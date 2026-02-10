//! End-to-end Responses API tool-calling tests.
//!
//! Mirrors orchard-py/tests/test_e2e_responses_tools.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::{
    ResponseEvent, ResponseInputItem, ResponseOutputItem, ResponsesInput, ResponsesRequest,
    ResponsesResult,
};

use crate::fixture::get_fixture;

const MODEL_ID: &str = "meta-llama/Llama-3.1-8B-Instruct";

fn weather_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco'"
                }
            },
            "required": ["location"]
        }
    })
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

fn base_input_items() -> Vec<ResponseInputItem> {
    vec![
        ResponseInputItem::Message {
            role: "system".to_string(),
            content: serde_json::json!(
                "You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original user question."
            ),
            tool_calls: None,
            tool_call_id: None,
        },
        ResponseInputItem::Message {
            role: "user".to_string(),
            content: serde_json::json!("What's the weather in San Francisco?"),
            tool_calls: None,
            tool_call_id: None,
        },
    ]
}

#[tokio::test]
#[ignore]
async fn test_responses_tool_call_non_streaming() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let request = ResponsesRequest {
        input: ResponsesInput::Items(base_input_items()),
        stream: false,
        instructions: None,
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        min_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        max_output_tokens: Some(128),
        top_logprobs: None,
        tools: vec![weather_tool()],
        tool_choice: Some(serde_json::json!("required")),
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
        ResponsesResult::Complete(response) => *response,
        ResponsesResult::Stream(_) => panic!("expected complete response, got stream"),
    };

    let tool_call = response
        .output
        .iter()
        .find_map(|item| {
            if let ResponseOutputItem::FunctionCall(call) = item {
                Some(call)
            } else {
                None
            }
        })
        .expect("expected a function_call output item");

    assert_eq!(tool_call.name, "get_weather");
    assert!(!tool_call.call_id.is_empty());
    assert_eq!(tool_call.status, orchard::OutputStatus::Completed);

    let args: serde_json::Value = serde_json::from_str(&tool_call.arguments)
        .unwrap_or_else(|e| panic!("invalid JSON arguments '{}': {}", tool_call.arguments, e));
    assert!(args.is_object());
    assert!(
        args.get("location").is_some(),
        "expected location in tool arguments"
    );
}

#[tokio::test]
#[ignore]
async fn test_responses_tool_call_streaming() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let request = ResponsesRequest {
        input: ResponsesInput::Items(base_input_items()),
        stream: true,
        instructions: None,
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        min_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        max_output_tokens: Some(128),
        top_logprobs: None,
        tools: vec![weather_tool()],
        tool_choice: Some(serde_json::json!("required")),
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

    let stream = match result.unwrap() {
        ResponsesResult::Stream(stream) => stream,
        ResponsesResult::Complete(_) => panic!("expected stream, got complete response"),
    };

    let events = collect_stream_events(stream).await;
    let event_types = events
        .iter()
        .map(ResponseEvent::event_type)
        .collect::<Vec<_>>();

    assert!(
        event_types.contains(&"response.function_call_arguments.delta"),
        "missing function_call_arguments.delta in {:?}",
        event_types
    );
    assert!(event_types.contains(&"response.function_call_arguments.done"));

    let mut accumulated_arguments = String::new();
    let mut done_arguments = String::new();
    let mut saw_tool_item_done = false;

    for event in events {
        match event {
            ResponseEvent::FunctionCallArgumentsDelta(delta) => {
                accumulated_arguments.push_str(&delta.delta);
            }
            ResponseEvent::FunctionCallArgumentsDone(done) => {
                done_arguments = done.arguments;
            }
            ResponseEvent::OutputItemDone(done) => {
                if let orchard::ResponseOutputItem::FunctionCall(call) = done.item {
                    saw_tool_item_done = true;
                    assert_eq!(call.name, "get_weather");
                }
            }
            _ => {}
        }
    }

    assert!(
        !done_arguments.is_empty(),
        "expected arguments.done payload"
    );
    assert_eq!(accumulated_arguments, done_arguments);
    let parsed: serde_json::Value = serde_json::from_str(&done_arguments)
        .unwrap_or_else(|e| panic!("invalid JSON arguments '{}': {}", done_arguments, e));
    assert!(parsed.is_object());
    assert!(
        saw_tool_item_done,
        "expected function_call output_item.done event"
    );
}

#[tokio::test]
#[ignore]
async fn test_responses_tool_result_continuation() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    // Turn 1: trigger a tool call
    let first_request = ResponsesRequest {
        input: ResponsesInput::Items(base_input_items()),
        stream: false,
        instructions: None,
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        min_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        max_output_tokens: Some(128),
        top_logprobs: None,
        tools: vec![weather_tool()],
        tool_choice: Some(serde_json::json!("required")),
        max_tool_calls: None,
        text: None,
        reasoning_effort: None,
        metadata: None,
        parallel_tool_calls: false,
    };

    let first_result = client.aresponses(MODEL_ID, first_request).await;
    assert!(
        first_result.is_ok(),
        "first responses request failed: {:?}",
        first_result.err()
    );

    let first_response = match first_result.unwrap() {
        ResponsesResult::Complete(response) => *response,
        ResponsesResult::Stream(_) => panic!("expected complete response, got stream"),
    };

    let tool_call = first_response
        .output
        .iter()
        .find_map(|item| {
            if let ResponseOutputItem::FunctionCall(call) = item {
                Some(call.clone())
            } else {
                None
            }
        })
        .expect("expected tool call in first response");

    // Turn 2: feed the function call and output back in.
    let mut items = base_input_items();
    items.push(ResponseInputItem::FunctionCall {
        call_id: tool_call.call_id.clone(),
        name: tool_call.name.clone(),
        arguments: tool_call.arguments.clone(),
    });
    items.push(ResponseInputItem::FunctionCallOutput {
        call_id: tool_call.call_id.clone(),
        output: serde_json::json!({
            "temperature": 65,
            "unit": "fahrenheit",
            "condition": "foggy"
        })
        .to_string(),
    });

    let second_request = ResponsesRequest {
        input: ResponsesInput::Items(items),
        stream: false,
        instructions: None,
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        min_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        max_output_tokens: Some(128),
        top_logprobs: None,
        tools: vec![weather_tool()],
        tool_choice: None,
        max_tool_calls: None,
        text: None,
        reasoning_effort: None,
        metadata: None,
        parallel_tool_calls: false,
    };

    let second_result = client.aresponses(MODEL_ID, second_request).await;
    assert!(
        second_result.is_ok(),
        "second responses request failed: {:?}",
        second_result.err()
    );

    let second_response = match second_result.unwrap() {
        ResponsesResult::Complete(response) => *response,
        ResponsesResult::Stream(_) => panic!("expected complete response, got stream"),
    };

    let message_text = second_response
        .output
        .iter()
        .find_map(|item| {
            if let ResponseOutputItem::Message(message) = item {
                message.content.first().map(|c| c.text.to_lowercase())
            } else {
                None
            }
        })
        .unwrap_or_default();

    assert!(
        ["65", "fog", "san francisco"]
            .iter()
            .any(|needle| message_text.contains(needle)),
        "model did not incorporate tool result: {}",
        message_text
    );
}
