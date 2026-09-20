//! Replies written out delta by delta, in the shape PIE sends them, for the
//! tests of every place that assembles message text.
//!
//! Every token delta carries two views of the text. `content` is the decoded
//! text of the sampled token. The state events are the engine's text stream:
//! it holds back the end of the text while that end could still become a stop
//! sequence (or the newline in front of `</think>`), never contains a stop
//! sequence, and hands held text over in a later delta.
//!
//! An engine that advertises the `released_text` capability completes that
//! stream: when a reply ends without a stop (token limit, cancel) the held text
//! arrives in the completion delta, which has no tokens and no `content`. Its
//! replies are assembled from the message spans alone. An engine without the
//! capability never hands that text over, and its replies are assembled exactly
//! as they were before the capability existed.

use serde_json::{json, Value};

use crate::ipc::client::{ResponseDelta, ResponseStateEvent};

pub(crate) struct RecordedReply {
    pub(crate) name: &'static str,
    /// The deltas as an engine that advertises `released_text` sends them.
    pub(crate) deltas: Vec<ResponseDelta>,
    /// The message text every assembly site makes of them.
    pub(crate) text: &'static str,
    /// The tool calls of the reply, by name.
    pub(crate) calls: &'static [&'static str],
    /// What main made of the same reply from an engine without `released_text`
    /// (`deltas_without_released_text`), where that is not `text`.
    pub(crate) old: Old,
}

/// The text of a reply from an engine without `released_text`, at the sites that
/// did not agree with `RecordedReply::text` on main. `None`: the same text.
#[derive(Default)]
pub(crate) struct Old {
    /// `achat` and `achat_batch`: `content` wherever a delta had no events of
    /// another item.
    pub(crate) chat: Option<&'static str>,
    /// The Responses paths (the non-streaming output and the final snapshot of
    /// the stream): the spans, `content` of deltas without events added to the
    /// message, and the value of `item_completed` in place of both.
    pub(crate) responses: Option<&'static str>,
    /// The text deltas of the streaming Responses path, where they are not
    /// `responses`.
    pub(crate) responses_streamed: Option<&'static str>,
    /// The `output_text.done` text of the streaming Responses path, where it is
    /// not `responses`.
    pub(crate) responses_done: Option<&'static str>,
}

impl RecordedReply {
    /// The reply as an engine without `released_text` sends it. Such an engine
    /// hands held text over only while the reply goes on: the completion delta
    /// of a reply that ended without a stop carries no events.
    pub(crate) fn deltas_without_released_text(&self) -> Vec<ResponseDelta> {
        let mut deltas = self.deltas.clone();
        for delta in &mut deltas {
            let cut_off = matches!(delta.finish_reason.as_deref(), Some("length" | "user"));
            if delta.is_final_delta && cut_off {
                delta.state_events.clear();
            }
        }
        deltas
    }
}

fn event(item_type: &str, output_index: u32, event_type: &str) -> ResponseStateEvent {
    ResponseStateEvent {
        event_type: event_type.to_string(),
        item_type: item_type.to_string(),
        output_index,
        identifier: item_type.to_string(),
        ..Default::default()
    }
}

fn started() -> ResponseStateEvent {
    event("message", 0, "item_started")
}

fn span(delta: &str) -> ResponseStateEvent {
    ResponseStateEvent {
        delta: delta.to_string(),
        ..event("message", 0, "content_delta")
    }
}

fn completed(value: &str) -> ResponseStateEvent {
    ResponseStateEvent {
        value: Some(Value::String(value.to_string())),
        ..event("message", 0, "item_completed")
    }
}

const ARGUMENTS: &str = r#"{"city":"Paris"}"#;

/// The events of the tool call that follows the message, the second output item.
fn call_events() -> [ResponseStateEvent; 3] {
    let call = |event_type: &str, identifier: &str| ResponseStateEvent {
        identifier: identifier.to_string(),
        ..event("tool_call", 1, event_type)
    };
    [
        call("item_started", "tool_call:get_weather"),
        ResponseStateEvent {
            delta: ARGUMENTS.to_string(),
            ..call("content_delta", "arguments")
        },
        ResponseStateEvent {
            value: Some(json!({"name": "get_weather", "arguments": {"city": "Paris"}})),
            ..call("item_completed", "tool_call:get_weather")
        },
    ]
}

fn token(content: &str, state_events: Vec<ResponseStateEvent>) -> ResponseDelta {
    ResponseDelta {
        request_id: 1,
        candidate_index: Some(0),
        content: Some(content.to_string()),
        tokens: vec![11],
        state_events,
        ..Default::default()
    }
}

fn completion(finish_reason: &str, state_events: Vec<ResponseStateEvent>) -> ResponseDelta {
    ResponseDelta {
        request_id: 1,
        candidate_index: Some(0),
        content: Some(String::new()),
        is_final_delta: true,
        finish_reason: Some(finish_reason.to_string()),
        state_events,
        ..Default::default()
    }
}

fn hello() -> ResponseDelta {
    token("Hello", vec![started(), span("Hello")])
}

pub(crate) fn all() -> Vec<RecordedReply> {
    let [call_started, call_arguments, call_completed] = call_events();
    vec![
        // Stop sequence "END", cut off by the token limit after " the E". "E"
        // could start the stop sequence, so its span waits; the completion delta
        // releases it.
        RecordedReply {
            name: "length_limit_releases_held_stop_start",
            deltas: vec![
                hello(),
                token(" the E", vec![span(" the ")]),
                completion("length", vec![span("E")]),
            ],
            text: "Hello the E",
            calls: &[],
            old: Old {
                // The held "E" never arrives: only `content` has it.
                responses: Some("Hello the "),
                ..Default::default()
            },
        },
        // A bare newline is held in front of a possible `</think>`: the token
        // delta has no events at all, and the completion delta releases the
        // newline.
        RecordedReply {
            name: "length_limit_releases_held_newline",
            deltas: vec![
                hello(),
                token("\n", Vec::new()),
                completion("length", vec![span("\n")]),
            ],
            text: "Hello\n",
            calls: &[],
            old: Old::default(),
        },
        // The reply ends on the stop sequence "END". Its tokens are decoded into
        // `content`, but the text stream never shows them.
        RecordedReply {
            name: "stop_sequence_stays_out",
            deltas: vec![
                hello(),
                token("EN", Vec::new()),
                token("D", vec![completed("Hello")]),
                completion("stop", Vec::new()),
            ],
            text: "Hello",
            calls: &[],
            old: Old {
                chat: Some("HelloEND"),
                // "EN" was streamed as message text; the completed value then
                // replaced it in the message.
                responses_streamed: Some("HelloEN"),
                ..Default::default()
            },
        },
        // The stop sequence "END" inside a token: the span ends in front of it,
        // `content` spells it.
        RecordedReply {
            name: "stop_sequence_inside_a_token",
            deltas: vec![
                hello(),
                token(" END", vec![span(" "), completed("Hello ")]),
                completion("stop", Vec::new()),
            ],
            text: "Hello ",
            calls: &[],
            old: Old {
                chat: Some("Hello END"),
                ..Default::default()
            },
        },
        // Held text released in the middle of a reply: "e" of " the" could start
        // "END", and the next token's span hands it over in front of its own text.
        RecordedReply {
            name: "held_text_released_mid_reply",
            deltas: vec![
                hello(),
                token(" the", vec![span(" th")]),
                token(" cat", vec![span("e cat")]),
                completion("stop", Vec::new()),
            ],
            text: "Hello the cat",
            calls: &[],
            old: Old::default(),
        },
        // An engine (or a model without a text state machine) that sends no state
        // events: `content` is all there is.
        RecordedReply {
            name: "no_state_events",
            deltas: vec![
                token("Hello", Vec::new()),
                token(" world", Vec::new()),
                completion("stop", Vec::new()),
            ],
            text: "Hello world",
            calls: &[],
            old: Old::default(),
        },
        // The reply ends on a stop token while a newline is held: the text state
        // machine ends with the reply, so the completion delta carries the
        // newline and the finished message.
        RecordedReply {
            name: "stop_token_releases_held_newline",
            deltas: vec![
                hello(),
                token("\n", Vec::new()),
                completion("stop", vec![span("\n"), completed("Hello\n")]),
            ],
            text: "Hello\n",
            calls: &[],
            old: Old {
                // `content` of the held newline, then its span; on the Responses
                // paths the completed value then replaced both in the message.
                chat: Some("Hello\n\n"),
                responses_streamed: Some("Hello\n\n"),
                ..Default::default()
            },
        },
        // A tool call after held text: the opening marker ends the message, so its
        // delta releases the held newline, completes the message and starts the
        // call. The markers are tokens like any other and are decoded into
        // `content`.
        RecordedReply {
            name: "tool_call_after_held_text",
            deltas: vec![
                hello(),
                token("\n", Vec::new()),
                token(
                    "<tool_call>",
                    vec![span("\n"), completed("Hello\n"), call_started.clone()],
                ),
                token(ARGUMENTS, vec![call_arguments.clone()]),
                token("</tool_call>", vec![call_completed.clone()]),
                completion("tool_use", Vec::new()),
            ],
            text: "Hello\n",
            calls: &["get_weather"],
            old: Old {
                chat: Some("Hello\n\n"),
                responses_streamed: Some("Hello\n\n"),
                ..Default::default()
            },
        },
        // The markers of a tool call as deltas without events, after the message
        // item exists: they are not message text.
        RecordedReply {
            name: "markers_without_events_after_a_message",
            deltas: vec![
                hello(),
                token("<tool_call>", Vec::new()),
                token(
                    ARGUMENTS,
                    vec![
                        completed("Hello"),
                        call_started,
                        call_arguments,
                        call_completed,
                    ],
                ),
                token("</tool_call>", Vec::new()),
                completion("tool_use", Vec::new()),
            ],
            text: "Hello",
            calls: &["get_weather"],
            old: Old {
                // Both markers were message text; on the Responses paths the
                // completed value replaced the first, and the second followed it.
                chat: Some("Hello<tool_call></tool_call>"),
                responses: Some("Hello</tool_call>"),
                responses_streamed: Some("Hello<tool_call></tool_call>"),
                responses_done: Some("Hello"),
            },
        },
        // Cancelled by the client (the engine's finish reason is "user") after
        // " the E": like the token limit, a cancel ends the reply without a stop,
        // and the completion delta releases the held "E".
        RecordedReply {
            name: "cancel_releases_held_stop_start",
            deltas: vec![
                hello(),
                token(" the E", vec![span(" the ")]),
                completion("user", vec![span("E")]),
            ],
            text: "Hello the E",
            calls: &[],
            old: Old {
                responses: Some("Hello the "),
                ..Default::default()
            },
        },
        // Structured output: the spans carry the JSON and the finished item
        // repeats it as its value.
        RecordedReply {
            name: "structured_json_complete",
            deltas: vec![
                token(r#"{"answer":"#, vec![started(), span(r#"{"answer":"#)]),
                token(
                    r#""A"}"#,
                    vec![span(r#""A"}"#), completed(r#"{"answer":"A"}"#)],
                ),
                completion("stop", Vec::new()),
            ],
            text: r#"{"answer":"A"}"#,
            calls: &[],
            old: Old::default(),
        },
        // Structured output cut by the token limit in the middle of the JSON.
        RecordedReply {
            name: "structured_json_cut_by_length",
            deltas: vec![
                token(r#"{"answer":"#, vec![started(), span(r#"{"answer":"#)]),
                token(r#""A"#, vec![span(r#""A"#)]),
                completion("length", Vec::new()),
            ],
            text: r#"{"answer":"A"#,
            calls: &[],
            old: Old::default(),
        },
    ]
}
