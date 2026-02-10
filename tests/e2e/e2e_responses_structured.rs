//! End-to-end Responses API structured output tests.
//!
//! Mirrors orchard-py/tests/test_e2e_responses_structured.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::{OutputStatus, ResponseOutputItem, ResponsesRequest, ResponsesResult};

use crate::fixture::get_fixture;

const MODEL_ID: &str = "meta-llama/Llama-3.1-8B-Instruct";

#[tokio::test]
#[ignore]
async fn test_responses_structured_json_schema() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "population": {"type": "integer"}
        },
        "required": ["capital", "population"]
    });

    let mut request = ResponsesRequest::from_text(
        "What is the capital of France and its approximate population? Respond as JSON.",
    );
    request.temperature = Some(0.0);
    request.max_output_tokens = Some(64);
    request.text = Some(serde_json::json!({
        "format": {
            "type": "json_schema",
            "name": "city_info",
            "schema": schema,
            "strict": true
        }
    }));

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

    assert_eq!(response.status, OutputStatus::Completed);

    let raw_text = response
        .output
        .iter()
        .find_map(|item| {
            if let ResponseOutputItem::Message(message) = item {
                message.content.first().map(|content| content.text.clone())
            } else {
                None
            }
        })
        .unwrap_or_default();

    let start = raw_text.find('{');
    let end = raw_text.rfind('}');
    assert!(
        start.is_some() && end.is_some(),
        "no JSON object found in output: {}",
        raw_text
    );

    let start = start.unwrap();
    let end = end.unwrap();
    let json_payload = &raw_text[start..=end];

    let parsed: serde_json::Value = serde_json::from_str(json_payload)
        .unwrap_or_else(|e| panic!("failed to parse JSON '{}': {}", json_payload, e));

    assert!(parsed.is_object());
    let obj = parsed.as_object().expect("parsed JSON should be object");
    assert!(obj.contains_key("capital"));
    assert!(obj.contains_key("population"));
    assert!(
        obj.get("capital")
            .and_then(serde_json::Value::as_str)
            .is_some(),
        "capital should be a string"
    );
    assert!(
        obj.get("population")
            .map(serde_json::Value::is_i64)
            .unwrap_or(false),
        "population should be an integer"
    );

    let capital = obj
        .get("capital")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_lowercase();
    assert!(capital.contains("paris"), "capital should mention paris");
}
