//! End-to-end Responses API basic tests.
//!
//! Mirrors orchard-py/tests/test_e2e_responses_basic.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::{
    OutputStatus, ResponseEvent, ResponseOutputItem, ResponsesInput, ResponsesRequest,
    ResponsesResult,
};

use crate::fixture::{get_fixture, make_message};

const MODEL_ID: &str = "meta-llama/Llama-3.1-8B-Instruct";

fn first_message_text(output: &[ResponseOutputItem]) -> String {
    output
        .iter()
        .find_map(|item| match item {
            ResponseOutputItem::Message(message) => message.content.first().map(|c| c.text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

async fn collect_stream_events(
    mut stream: tokio::sync::mpsc::Receiver<ResponseEvent>,
) -> Vec<ResponseEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.recv().await {
        let is_done = matches!(event, ResponseEvent::Done);
        events.push(event);
        if is_done {
            break;
        }
    }
    events
}

#[tokio::test]
#[ignore]
async fn test_responses_non_streaming_string_input() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let mut request = ResponsesRequest::from_text("Say hello in one sentence.");
    request.max_output_tokens = Some(32);
    request.temperature = Some(0.0);

    let result = client.aresponses(MODEL_ID, request).await;
    assert!(
        result.is_ok(),
        "responses request failed: {:?}",
        result.err()
    );

    let response = match result.unwrap() {
        ResponsesResult::Complete(response) => response,
        ResponsesResult::Stream(_) => panic!("expected complete response, got stream"),
    };

    assert_eq!(response.object, "response");
    assert!(response.id.starts_with("resp_"));
    assert_eq!(response.status, OutputStatus::Completed);
    assert_eq!(response.model, MODEL_ID);
    assert!(response.created_at > 0);

    assert!(!response.output.is_empty());
    let text = first_message_text(&response.output);
    assert!(!text.is_empty(), "expected non-empty output text");

    let usage = response.usage.expect("usage should be present");
    assert!(usage.input_tokens > 0);
    assert!(usage.output_tokens > 0);
    assert_eq!(usage.total_tokens, usage.input_tokens + usage.output_tokens);
}

#[tokio::test]
#[ignore]
async fn test_responses_non_streaming_message_items() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let request = ResponsesRequest {
        input: ResponsesInput::Items(vec![orchard::ResponseInputItem::Message {
            role: "user".to_string(),
            content: serde_json::json!("What is 2+2? Answer with just the number."),
            tool_calls: None,
            tool_call_id: None,
        }]),
        stream: false,
        instructions: None,
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        min_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        max_output_tokens: Some(8),
        top_logprobs: None,
        tools: Vec::new(),
        tool_choice: None,
        max_tool_calls: None,
        text: None,
        reasoning_effort: None,
        metadata: None,
        parallel_tool_calls: false,
    };

    let result = client.aresponses(MODEL_ID, request).await;
    assert!(
        result.is_ok(),
        "responses request failed: {:?}",
        result.err()
    );

    let response = match result.unwrap() {
        ResponsesResult::Complete(response) => response,
        ResponsesResult::Stream(_) => panic!("expected complete response, got stream"),
    };

    assert_eq!(response.status, OutputStatus::Completed);

    let text = first_message_text(&response.output).to_lowercase();
    assert!(
        text.contains('4'),
        "expected answer to contain 4, got: {}",
        text
    );
}

#[tokio::test]
#[ignore]
async fn test_responses_echo_fields() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let mut request = ResponsesRequest::from_text("Hi");
    request.temperature = Some(0.5);
    request.top_p = Some(0.9);
    request.max_output_tokens = Some(4);
    request.metadata = Some(
        [("test_key".to_string(), "test_value".to_string())]
            .into_iter()
            .collect(),
    );

    let result = client.aresponses(MODEL_ID, request).await;
    assert!(
        result.is_ok(),
        "responses request failed: {:?}",
        result.err()
    );

    let response = match result.unwrap() {
        ResponsesResult::Complete(response) => response,
        ResponsesResult::Stream(_) => panic!("expected complete response, got stream"),
    };

    assert_eq!(response.temperature, Some(0.5));
    assert_eq!(response.top_p, Some(0.9));
    assert_eq!(
        response.metadata,
        Some(
            [("test_key".to_string(), "test_value".to_string())]
                .into_iter()
                .collect()
        )
    );
}

#[tokio::test]
#[ignore]
async fn test_responses_streaming_event_sequence() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let mut request = ResponsesRequest::from_text("Say hello in one sentence.");
    request.stream = true;
    request.temperature = Some(0.0);
    request.max_output_tokens = Some(32);

    let result = client.aresponses(MODEL_ID, request).await;
    assert!(
        result.is_ok(),
        "responses request failed: {:?}",
        result.err()
    );

    let stream = match result.unwrap() {
        ResponsesResult::Stream(stream) => stream,
        ResponsesResult::Complete(_) => panic!("expected stream, got complete response"),
    };

    let events = collect_stream_events(stream).await;
    let event_types = events
        .iter()
        .map(ResponseEvent::event_type)
        .collect::<Vec<_>>();

    assert_eq!(event_types[0], "response.created");
    assert_eq!(event_types[1], "response.in_progress");
    assert!(event_types.contains(&"response.output_item.added"));
    assert!(event_types.contains(&"response.content_part.added"));
    assert!(event_types.contains(&"response.output_text.delta"));
    assert!(event_types.contains(&"response.output_text.done"));
    assert!(event_types.contains(&"response.content_part.done"));
    assert!(event_types.contains(&"response.output_item.done"));
    assert_eq!(event_types[event_types.len() - 2], "response.completed");
    assert_eq!(event_types[event_types.len() - 1], "done");

    let mut sequence_numbers = Vec::new();
    for event in &events {
        match event {
            ResponseEvent::ResponseCreated(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::ResponseInProgress(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::ResponseCompleted(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::ResponseFailed(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::ResponseIncomplete(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::OutputItemAdded(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::OutputItemDone(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::ContentPartAdded(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::ContentPartDone(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::OutputTextDelta(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::OutputTextDone(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::FunctionCallArgumentsDelta(e) => {
                sequence_numbers.push(e.sequence_number)
            }
            ResponseEvent::FunctionCallArgumentsDone(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::ReasoningDelta(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::ReasoningDone(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::ReasoningSummaryTextDelta(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::ReasoningSummaryTextDone(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::Error(e) => sequence_numbers.push(e.sequence_number),
            ResponseEvent::Done => {}
        }
    }

    let sorted = {
        let mut copy = sequence_numbers.clone();
        copy.sort_unstable();
        copy
    };
    assert_eq!(sequence_numbers, sorted);

    let unique = sequence_numbers
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(sequence_numbers.len(), unique.len());
}

#[tokio::test]
#[ignore]
async fn test_responses_streaming_delta_accumulation() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let mut request = ResponsesRequest::from_text("Count from 1 to 5.");
    request.stream = true;
    request.temperature = Some(0.0);
    request.max_output_tokens = Some(64);

    let result = client.aresponses(MODEL_ID, request).await;
    assert!(
        result.is_ok(),
        "responses request failed: {:?}",
        result.err()
    );

    let stream = match result.unwrap() {
        ResponsesResult::Stream(stream) => stream,
        ResponsesResult::Complete(_) => panic!("expected stream, got complete response"),
    };

    let events = collect_stream_events(stream).await;

    let mut accumulated = String::new();
    let mut done_text = String::new();

    for event in events {
        if let ResponseEvent::OutputTextDelta(delta) = event {
            accumulated.push_str(&delta.delta);
        } else if let ResponseEvent::OutputTextDone(done) = event {
            done_text = done.text;
        }
    }

    assert!(
        !done_text.is_empty(),
        "expected output_text.done event text"
    );
    assert_eq!(accumulated, done_text);
}

#[tokio::test]
#[ignore]
async fn test_responses_streaming_completed_snapshot() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let mut request = ResponsesRequest::from_text("Hi");
    request.stream = true;
    request.temperature = Some(0.0);
    request.max_output_tokens = Some(64);

    let result = client.aresponses(MODEL_ID, request).await;
    assert!(
        result.is_ok(),
        "responses request failed: {:?}",
        result.err()
    );

    let stream = match result.unwrap() {
        ResponsesResult::Stream(stream) => stream,
        ResponsesResult::Complete(_) => panic!("expected stream, got complete response"),
    };

    let events = collect_stream_events(stream).await;
    let completed = events
        .iter()
        .find_map(|event| {
            if let ResponseEvent::ResponseCompleted(done) = event {
                Some(done)
            } else {
                None
            }
        })
        .expect("expected response.completed event");

    assert_eq!(completed.response.status, OutputStatus::Completed);
}

#[tokio::test]
#[ignore]
async fn test_responses_incomplete_non_streaming() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let mut request =
        ResponsesRequest::from_text("Write a very long essay about the history of mathematics.");
    request.temperature = Some(0.0);
    request.max_output_tokens = Some(1);

    let result = client.aresponses(MODEL_ID, request).await;
    assert!(
        result.is_ok(),
        "responses request failed: {:?}",
        result.err()
    );

    let response = match result.unwrap() {
        ResponsesResult::Complete(response) => response,
        ResponsesResult::Stream(_) => panic!("expected complete response, got stream"),
    };

    assert_eq!(response.status, OutputStatus::Incomplete);
    assert_eq!(
        response.incomplete_details.map(|d| d.reason),
        Some("max_output_tokens".to_string())
    );
}

#[tokio::test]
#[ignore]
async fn test_responses_incomplete_streaming() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let mut request =
        ResponsesRequest::from_text("Write a very long essay about the history of mathematics.");
    request.stream = true;
    request.temperature = Some(0.0);
    request.max_output_tokens = Some(1);

    let result = client.aresponses(MODEL_ID, request).await;
    assert!(
        result.is_ok(),
        "responses request failed: {:?}",
        result.err()
    );

    let stream = match result.unwrap() {
        ResponsesResult::Stream(stream) => stream,
        ResponsesResult::Complete(_) => panic!("expected stream, got complete response"),
    };

    let events = collect_stream_events(stream).await;
    let event_types = events
        .iter()
        .map(ResponseEvent::event_type)
        .collect::<Vec<_>>();

    assert!(event_types.contains(&"response.incomplete"));
    assert_eq!(event_types[event_types.len() - 1], "done");

    let incomplete = events
        .iter()
        .find_map(|event| {
            if let ResponseEvent::ResponseIncomplete(incomplete) = event {
                Some(incomplete)
            } else {
                None
            }
        })
        .expect("expected response.incomplete event");

    assert_eq!(incomplete.response.status, OutputStatus::Incomplete);
    assert_eq!(
        incomplete
            .response
            .incomplete_details
            .as_ref()
            .map(|d| d.reason.clone()),
        Some("max_output_tokens".to_string())
    );
}

#[tokio::test]
#[ignore]
async fn test_responses_instructions() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let mut request = ResponsesRequest::from_text("What is your name?");
    request.instructions = Some(
        "You are a helpful assistant named Orchard. Always introduce yourself by name.".to_string(),
    );
    request.temperature = Some(0.0);
    request.max_output_tokens = Some(64);

    let result = client.aresponses(MODEL_ID, request).await;
    assert!(
        result.is_ok(),
        "responses request failed: {:?}",
        result.err()
    );

    let response = match result.unwrap() {
        ResponsesResult::Complete(response) => response,
        ResponsesResult::Stream(_) => panic!("expected complete response, got stream"),
    };

    let text = first_message_text(&response.output).to_lowercase();
    assert!(
        text.contains("orchard"),
        "expected orchard in output, got: {}",
        text
    );
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_fixture_message_helper() {
        let message = make_message("user", "hello");
        assert_eq!(
            message.get("role").and_then(serde_json::Value::as_str),
            Some("user")
        );
    }
}
