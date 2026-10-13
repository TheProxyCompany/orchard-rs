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
/// (`generated`) and the thinking mode it was generated in (`generated_thinking`),
/// returning the ids the markers stand for. `model` is the model being asked, when its
/// engine takes token segments. A record from any other model, or one that is not a
/// non-empty list of token ids, is dropped and the message renders as text; so is every
/// record when the conversation already contains a marker character, since text that
/// looks like a marker must never be swapped for ids.
pub(super) fn take_replays(
    messages: &[Message],
    model: Option<&str>,
) -> (Vec<Message>, Vec<Vec<i32>>) {
    let marker_free = || {
        !messages
            .iter()
            .flat_map(|m| m.values())
            .any(|v| v.to_string().contains(MARK_OPEN))
    };
    let model = model.filter(|_| marker_free());
    let mut replays = Vec::new();
    let messages = messages
        .iter()
        .map(|message| {
            let mut message = message.clone();
            let Some(generation) = message.remove("generation") else {
                return message;
            };
            let same_model =
                model.is_some() && generation.get("model").and_then(Value::as_str) == model;
            let tokens = generation
                .get("tokens")
                .and_then(Value::as_array)
                .and_then(|ids| {
                    ids.iter()
                        .map(|id| id.as_u64().and_then(|id| i32::try_from(id).ok()))
                        .collect::<Option<Vec<i32>>>()
                });
            if let Some(tokens) = tokens.filter(|tokens| same_model && !tokens.is_empty()) {
                message.insert(
                    "generated".into(),
                    json!(format!("{MARK_OPEN}{}{MARK_CLOSE}", replays.len())),
                );
                message.insert(
                    "generated_thinking".into(),
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
/// Returns the prompt, the layout, and the token segments in layout order. Markers are
/// taken in order, once each; anything else stays in the text around it, and a request
/// with nothing to replay comes back exactly as it went in.
pub(super) fn splice_replays(
    prompt: &str,
    layout: &[LayoutEntry],
    replays: &[Vec<i32>],
) -> (String, Vec<LayoutEntry>, Vec<Vec<i32>>) {
    if replays.is_empty() {
        return (prompt.to_string(), layout.to_vec(), Vec::new());
    }
    let mut text = String::with_capacity(prompt.len());
    let (mut spliced, mut token_segments) = (Vec::with_capacity(layout.len()), Vec::new());
    let mut cursor = 0;
    for entry in layout {
        if entry.segment_type != "text" {
            spliced.push(entry.clone());
            continue;
        }
        let segment = &prompt[cursor..cursor + entry.length];
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
        let (mut run_start, mut scan) = (0, 0);
        while let Some(open) = segment[scan..].find(MARK_OPEN).map(|at| scan + at) {
            let digits_start = open + MARK_OPEN.len_utf8();
            let digits = segment[digits_start..]
                .bytes()
                .take(9)
                .take_while(u8::is_ascii_digit)
                .count();
            let close = digits_start + digits;
            let next = token_segments.len();
            let ids = segment[close..]
                .starts_with(MARK_CLOSE)
                .then(|| segment[digits_start..close].parse::<usize>().ok())
                .flatten()
                .filter(|index| *index == next)
                .and_then(|index| replays.get(index));
            scan = digits_start;
            if let Some(ids) = ids {
                push_text(&segment[run_start..open], &mut spliced);
                spliced.push(LayoutEntry {
                    segment_type: "tokens".into(),
                    length: ids.len(),
                });
                token_segments.push(ids.clone());
                run_start = close + MARK_CLOSE.len_utf8();
                scan = run_start;
            }
        }
        push_text(&segment[run_start..], &mut spliced);
    }
    (text, spliced, token_segments)
}
