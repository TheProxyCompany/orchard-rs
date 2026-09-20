//! Replies written out delta by delta, in the shape PIE sends them, for the
//! tests of every place that assembles message text.
//!
//! Every token delta carries two views of the text. `content` is the decoded
//! text of the sampled token. The state events are the engine's text stream:
//! it holds back the end of the text while that end could still become a stop
//! sequence (or the newline in front of `</think>`), never contains a stop
//! sequence, and hands held text over in a later delta. When a reply ends
//! without a stop (token limit, cancel) the held text arrives in the completion
//! delta, which has no tokens and no `content`.

use serde_json::Value;

use crate::ipc::client::{ResponseDelta, ResponseStateEvent};

pub(crate) struct RecordedReply {
    pub(crate) deltas: Vec<ResponseDelta>,
    /// The message text every assembly site must arrive at.
    pub(crate) text: &'static str,
}

fn message_event(event_type: &str, delta: &str, value: Option<&str>) -> ResponseStateEvent {
    ResponseStateEvent {
        event_type: event_type.to_string(),
        item_type: "message".to_string(),
        output_index: 0,
        identifier: "message".to_string(),
        delta: delta.to_string(),
        value: value.map(|value| Value::String(value.to_string())),
    }
}

fn span(delta: &str) -> ResponseStateEvent {
    message_event("content_delta", delta, None)
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
    token(
        "Hello",
        vec![message_event("item_started", "", None), span("Hello")],
    )
}

/// Stop sequence "END", cut off by the token limit after " the E". "E" could
/// start the stop sequence, so its span waits; the completion delta releases it.
pub(crate) fn length_limit_releases_held_stop_start() -> RecordedReply {
    RecordedReply {
        deltas: vec![
            hello(),
            token(" the E", vec![span(" the ")]),
            completion("length", vec![span("E")]),
        ],
        text: "Hello the E",
    }
}

/// A bare newline is held in front of a possible `</think>`: the token delta
/// has no events at all, and the completion delta releases the newline.
pub(crate) fn length_limit_releases_held_newline() -> RecordedReply {
    RecordedReply {
        deltas: vec![
            hello(),
            token("\n", Vec::new()),
            completion("length", vec![span("\n")]),
        ],
        text: "Hello\n",
    }
}

/// The reply ends on the stop sequence "END". Its tokens are decoded into
/// `content`, but the text stream never shows them.
pub(crate) fn stop_sequence_stays_out() -> RecordedReply {
    RecordedReply {
        deltas: vec![
            hello(),
            token("EN", Vec::new()),
            token(
                "D",
                vec![message_event("item_completed", "", Some("Hello"))],
            ),
            completion("stop", Vec::new()),
        ],
        text: "Hello",
    }
}

/// Held text released in the middle of a reply: "e" of " the" could start
/// "END", and the next token's span hands it over in front of its own text.
pub(crate) fn held_text_released_mid_reply() -> RecordedReply {
    RecordedReply {
        deltas: vec![
            hello(),
            token(" the", vec![span(" th")]),
            token(" cat", vec![span("e cat")]),
            completion("stop", Vec::new()),
        ],
        text: "Hello the cat",
    }
}

/// An engine (or a model without a text state machine) that sends no state
/// events: `content` is all there is.
pub(crate) fn no_state_events() -> RecordedReply {
    RecordedReply {
        deltas: vec![
            token("Hello", Vec::new()),
            token(" world", Vec::new()),
            completion("stop", Vec::new()),
        ],
        text: "Hello world",
    }
}
