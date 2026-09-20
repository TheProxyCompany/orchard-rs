//! Replaying what a model generated as the token ids it produced.
//!
//! A reply the model wrote comes back on the next turn as part of the prompt. Rendered
//! as text and encoded again it is not always the same ids (a merge across the seam,
//! a special token, a split the model chose that the tokenizer would not), and one
//! different id ends prefix-cache reuse for the rest of the conversation. An assistant
//! message that carries a `generation` record is sent as those exact ids instead: the
//! template renders a marker in its place, and the marker becomes a `tokens` layout
//! segment. The next prompt is then the last prompt plus the reply, id for id.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::ipc::serialization::LayoutEntry;

type Message = HashMap<String, Value>;

// Private-use code points: no template or tokenizer gives them a meaning.
const MARK_OPEN: char = '\u{E000}';
const MARK_CLOSE: char = '\u{E001}';

/// Replace each assistant message's `generation` record with a marker for the template
/// (`generated`) and the thinking mode it was generated in (`generated_thinking`), returning the ids the
/// markers stand for. `model` is the model being asked, when its engine takes token
/// segments; records from any other model are dropped and the message renders as text.
pub(super) fn take_replays(
    messages: &[Message],
    model: Option<&str>,
) -> (Vec<Message>, Vec<Vec<i32>>) {
    let mut replays = Vec::new();
    let messages = messages
        .iter()
        .map(|message| {
            let mut message = message.clone();
            let Some(generation) = message.remove("generation") else {
                return message;
            };
            let tokens: Vec<i32> = generation
                .get("tokens")
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(|id| id.as_i64().map(|id| id as i32))
                        .collect()
                })
                .unwrap_or_default();
            let same_model =
                model.is_some() && generation.get("model").and_then(Value::as_str) == model;
            if same_model && !tokens.is_empty() {
                message.insert(
                    "generated".into(),
                    json!(format!("{MARK_OPEN}{}{MARK_CLOSE}", replays.len())),
                );
                message.insert(
                    "thinking".into(),
                    generation.get("thinking").cloned().unwrap_or(json!(false)),
                );
                replays.push(tokens);
            }
            message
        })
        .collect();
    (messages, replays)
}

/// Cut the markers out of the prompt text and put a `tokens` segment where each one was.
/// Returns the prompt, the layout, and the token segments in layout order.
pub(super) fn splice_replays(
    prompt: &str,
    layout: &[LayoutEntry],
    replays: &[Vec<i32>],
) -> (String, Vec<LayoutEntry>, Vec<Vec<i32>>) {
    let mut text = String::with_capacity(prompt.len());
    let (mut spliced, mut token_segments) = (Vec::with_capacity(layout.len()), Vec::new());
    let mut cursor = 0;
    for entry in layout {
        if entry.segment_type != "text" {
            spliced.push(entry.clone());
            continue;
        }
        let mut rest = &prompt[cursor..cursor + entry.length];
        cursor += entry.length;
        let mut push_text = |part: &str, spliced: &mut Vec<LayoutEntry>| {
            if !part.is_empty() {
                text.push_str(part);
                spliced.push(LayoutEntry {
                    segment_type: "text".into(),
                    length: part.len(),
                });
            }
        };
        while let Some((before, marked)) = rest.split_once(MARK_OPEN) {
            // Anything that is not a marker this request made stays as text.
            let replay = marked.split_once(MARK_CLOSE).and_then(|(index, after)| {
                Some((replays.get(index.parse::<usize>().ok()?)?, after))
            });
            let Some((ids, after)) = replay else {
                push_text(&rest[..before.len() + MARK_OPEN.len_utf8()], &mut spliced);
                rest = marked;
                continue;
            };
            push_text(before, &mut spliced);
            spliced.push(LayoutEntry {
                segment_type: "tokens".into(),
                length: ids.len(),
            });
            token_segments.push(ids.clone());
            rest = after;
        }
        push_text(rest, &mut spliced);
    }
    (text, spliced, token_segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(generation: Value) -> Message {
        HashMap::from([
            ("role".to_string(), json!("assistant")),
            ("content".to_string(), json!("hi")),
            ("generation".to_string(), generation),
        ])
    }

    #[test]
    fn a_reply_from_the_same_model_becomes_a_token_segment() {
        let history = [assistant(
            json!({"model": "m", "tokens": [7, 8, 9], "thinking": true}),
        )];
        let (messages, replays) = take_replays(&history, Some("m"));
        assert_eq!(replays, vec![vec![7, 8, 9]]);
        assert_eq!(messages[0]["generated_thinking"], json!(true));
        assert!(!messages[0].contains_key("generation"));

        let marker = messages[0]["generated"].as_str().unwrap();
        let prompt = format!("<a>{marker}<end><b>");
        let layout = [LayoutEntry {
            segment_type: "text".into(),
            length: prompt.len(),
        }];
        let (text, layout, token_segments) = splice_replays(&prompt, &layout, &replays);
        assert_eq!(text, "<a><end><b>");
        let shape: Vec<_> = layout
            .iter()
            .map(|e| (e.segment_type.as_str(), e.length))
            .collect();
        assert_eq!(shape, [("text", 3), ("tokens", 3), ("text", 8)]);
        assert_eq!(token_segments, replays);
    }

    #[test]
    fn a_reply_from_another_model_or_an_older_engine_stays_text() {
        let history = [assistant(json!({"model": "other", "tokens": [7]}))];
        for model in [Some("m"), None] {
            let (messages, replays) = take_replays(&history, model);
            assert!(replays.is_empty());
            assert!(
                !messages[0].contains_key("generated") && !messages[0].contains_key("generation")
            );
        }
    }

    #[test]
    fn marker_characters_a_user_typed_are_left_alone() {
        let prompt = format!("a{MARK_OPEN}9{MARK_CLOSE}b{MARK_OPEN}c");
        let layout = [LayoutEntry {
            segment_type: "text".into(),
            length: prompt.len(),
        }];
        let (text, layout, token_segments) = splice_replays(&prompt, &layout, &[vec![1]]);
        assert_eq!(text, prompt);
        assert!(token_segments.is_empty());
        assert_eq!(layout.iter().map(|e| e.length).sum::<usize>(), prompt.len());
    }
}
