use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use orchard::{
    OutputFunctionCall, ResponseCompletedEvent, ResponseEvent, ResponseOutputItem, ResponseUsage,
};
use serde_json::{Map, Value};

const ID_KEYS: &[&str] = &["id", "item_id", "call_id", "response_id"];
const TIMESTAMP_KEYS: &[&str] = &["created_at", "completed_at"];

#[derive(Debug, Default)]
pub(crate) struct Turn {
    pub(crate) events: Vec<ResponseEvent>,
    pub(crate) order: Vec<&'static str>,
    pub(crate) counts: HashMap<&'static str, usize>,
    pub(crate) added: HashMap<&'static str, usize>,
    pub(crate) reasoning: String,
    pub(crate) reasoning_done: Option<String>,
    pub(crate) content: String,
    pub(crate) content_done: Option<String>,
    pub(crate) args_done: Option<String>,
    pub(crate) field_args: HashMap<String, String>,
    pub(crate) generated: String,
    pub(crate) stop_token: Option<String>,
    pub(crate) function_calls: Vec<OutputFunctionCall>,
    pub(crate) items_added: Vec<ResponseOutputItem>,
    pub(crate) items_done: Vec<ResponseOutputItem>,
}

pub(crate) async fn drain_stream(mut stream: tokio::sync::mpsc::Receiver<ResponseEvent>) -> Turn {
    let mut turn = Turn::default();

    while let Some(event) = stream.recv().await {
        let event_type = event.event_type();
        turn.order.push(event_type);
        *turn.counts.entry(event_type).or_insert(0) += 1;

        match &event {
            ResponseEvent::OutputItemAdded(added) => {
                *turn.added.entry(added.item.item_type()).or_insert(0) += 1;
                turn.items_added.push(added.item.clone());
            }
            ResponseEvent::ReasoningDelta(delta) => {
                turn.reasoning.push_str(&delta.delta);
            }
            ResponseEvent::ReasoningDone(done) => {
                turn.reasoning_done = Some(done.text.clone());
            }
            ResponseEvent::OutputToken(token) => {
                if let Some(content) = &token.content {
                    turn.generated.push_str(content);
                }
            }
            ResponseEvent::OutputTextDelta(delta) => {
                turn.content.push_str(&delta.delta);
            }
            ResponseEvent::OutputTextDone(done) => {
                turn.content_done = Some(done.text.clone());
            }
            ResponseEvent::FunctionCallArgumentsDelta(delta) => {
                if let Some(field_path) = &delta.field_path {
                    turn.field_args
                        .entry(field_path.clone())
                        .or_default()
                        .push_str(&delta.delta);
                }
            }
            ResponseEvent::FunctionCallArgumentsDone(done) => {
                turn.args_done = Some(done.arguments.clone());
            }
            ResponseEvent::OutputItemDone(done) => {
                if let ResponseOutputItem::FunctionCall(call) = &done.item {
                    turn.function_calls.push(call.clone());
                }
                turn.items_done.push(done.item.clone());
            }
            ResponseEvent::ResponseCompleted(completed) => {
                turn.stop_token = completed.response.stop_token.clone();
            }
            ResponseEvent::Done => {
                turn.events.push(event);
                break;
            }
            _ => {}
        }

        turn.events.push(event);
    }

    turn
}

pub(crate) fn reasoning_tokens(turn: &Turn) -> u32 {
    let completed = turn
        .events
        .iter()
        .filter_map(|event| match event {
            ResponseEvent::ResponseCompleted(completed) => Some(completed),
            _ => None,
        })
        .collect::<Vec<&ResponseCompletedEvent>>();
    assert_eq!(
        completed.len(),
        1,
        "expected exactly one response.completed"
    );
    let usage = completed[0]
        .response
        .usage
        .as_ref()
        .expect("response.completed carried no usage");
    usage
        .output_tokens_details
        .as_ref()
        .map(|details| details.reasoning_tokens)
        .unwrap_or(0)
}

pub(crate) fn assert_or_record(
    template_type: &str,
    scenario: &str,
    turn: &str,
    events: &[ResponseEvent],
) {
    let live = normalize(events);
    let path = golden_path(template_type, scenario);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("missing golden {}: {}", path.display(), err));
    let data: Value = serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("invalid golden {}: {}", path.display(), err));
    let recorded = data
        .get(turn)
        .unwrap_or_else(|| panic!("missing golden turn {template_type}/{scenario}/{turn}"));

    if recorded == &Value::Array(live.clone()) {
        return;
    }

    let recorded_events = recorded
        .as_array()
        .unwrap_or_else(|| panic!("golden turn {template_type}/{scenario}/{turn} is not an array"));
    let mut detail = if recorded_events.len() != live.len() {
        format!(
            "event count: golden={} live={}",
            recorded_events.len(),
            live.len()
        )
    } else {
        "event count matches but contents differ".to_string()
    };

    for (index, (expected, actual)) in recorded_events.iter().zip(live.iter()).enumerate() {
        if expected != actual {
            detail.push_str(&format!(
                "; first diff at index {index}:\n  golden: {expected}\n  live:   {actual}"
            ));
            break;
        }
    }

    panic!("golden drift {template_type}/{scenario}/{turn}: {detail}");
}

fn golden_path(template_type: &str, scenario: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/data")
        .join(template_type)
        .join(format!("{scenario}.json"))
}

fn normalize(events: &[ResponseEvent]) -> Vec<Value> {
    let mut ids: HashMap<String, String> = HashMap::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    events
        .iter()
        .map(event_to_value)
        .map(|value| canon(value, &mut ids, &mut counts))
        .collect()
}

fn event_to_value(event: &ResponseEvent) -> Value {
    match event {
        ResponseEvent::ResponseCreated(event) => wrap_event("response.created", event),
        ResponseEvent::ResponseInProgress(event) => wrap_event("response.in_progress", event),
        ResponseEvent::ResponseCompleted(event) => wrap_event("response.completed", event),
        ResponseEvent::ResponseFailed(event) => wrap_event("response.failed", event),
        ResponseEvent::ResponseIncomplete(event) => wrap_event("response.incomplete", event),
        ResponseEvent::OutputItemAdded(event) => wrap_event("response.output_item.added", event),
        ResponseEvent::OutputItemDone(event) => wrap_event("response.output_item.done", event),
        ResponseEvent::ContentPartAdded(event) => wrap_event("response.content_part.added", event),
        ResponseEvent::ContentPartDone(event) => wrap_event("response.content_part.done", event),
        ResponseEvent::OutputTextDelta(event) => wrap_event("response.output_text.delta", event),
        ResponseEvent::OutputTextDone(event) => wrap_event("response.output_text.done", event),
        ResponseEvent::OutputToken(event) => wrap_event("response.output_token", event),
        ResponseEvent::FunctionCallArgumentsDelta(event) => {
            wrap_event("response.function_call_arguments.delta", event)
        }
        ResponseEvent::FunctionCallArgumentsDone(event) => {
            wrap_event("response.function_call_arguments.done", event)
        }
        ResponseEvent::ReasoningDelta(event) => wrap_event("response.reasoning.delta", event),
        ResponseEvent::ReasoningDone(event) => wrap_event("response.reasoning.done", event),
        ResponseEvent::ReasoningSummaryTextDelta(event) => {
            wrap_event("response.reasoning_summary_text.delta", event)
        }
        ResponseEvent::ReasoningSummaryTextDone(event) => {
            wrap_event("response.reasoning_summary_text.done", event)
        }
        ResponseEvent::Error(event) => wrap_event("error", event),
        ResponseEvent::Done => serde_json::json!({"type": "done"}),
    }
}

fn wrap_event<T: serde::Serialize>(event_type: &str, event: &T) -> Value {
    let value = serde_json::to_value(event).expect("event serializes");
    let Value::Object(fields) = value else {
        panic!("event serialized to non-object");
    };
    let mut wrapped = Map::new();
    wrapped.insert("type".to_string(), Value::String(event_type.to_string()));
    for (key, value) in fields {
        wrapped.insert(key, value);
    }
    Value::Object(wrapped)
}

fn canon(
    value: Value,
    ids: &mut HashMap<String, String>,
    counts: &mut HashMap<String, usize>,
) -> Value {
    match value {
        Value::Object(object) => {
            let mapped = object
                .into_iter()
                .map(|(key, value)| {
                    let value = if TIMESTAMP_KEYS.contains(&key.as_str()) {
                        Value::Null
                    } else if ID_KEYS.contains(&key.as_str()) {
                        match value {
                            Value::String(text) => Value::String(token(&text, ids, counts)),
                            other => canon(other, ids, counts),
                        }
                    } else {
                        canon(value, ids, counts)
                    };
                    (key, value)
                })
                .collect::<Map<String, Value>>();
            Value::Object(mapped)
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|value| canon(value, ids, counts))
                .collect(),
        ),
        other => other,
    }
}

fn token(
    value: &str,
    ids: &mut HashMap<String, String>,
    counts: &mut HashMap<String, usize>,
) -> String {
    if let Some(token) = ids.get(value) {
        return token.clone();
    }

    let prefix = value
        .split_once('_')
        .map(|(prefix, _)| prefix)
        .unwrap_or("id");
    let count = counts.entry(prefix.to_string()).or_insert(0);
    let token = format!("{prefix}_{count}");
    *count += 1;
    ids.insert(value.to_string(), token.clone());
    token
}

#[allow(dead_code)]
pub(crate) fn usage_from_completed(turn: &Turn) -> Option<&ResponseUsage> {
    turn.events.iter().find_map(|event| match event {
        ResponseEvent::ResponseCompleted(completed) => completed.response.usage.as_ref(),
        _ => None,
    })
}
