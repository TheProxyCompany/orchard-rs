//! End-to-end batching tests.
//!
//! Mirrors orchard-py/tests/test_e2e_batching.py
//! Tests batched inference where multiple prompts are processed together.
//! Run with: cargo test --test e2e -- --ignored

use std::collections::HashMap;

use orchard::{BatchChatResult, SamplingParams};

use crate::fixture::{get_fixture, make_message};

const MODEL_ID: &str = "moondream3";

/// Test homogeneous batched chat completion with identical parameters.
/// Mirrors: test_e2e_batching.py::test_chat_completion_batched_homogeneous
#[tokio::test]
#[ignore]
async fn test_chat_completion_batched_homogeneous() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 10,
        temperature: 0.0,
        ..Default::default()
    };

    let conversations = vec![
        vec![make_message("user", "Say hello politely.")],
        vec![make_message("user", "Give me a fun fact about space.")],
    ];

    let result = client
        .achat_batch(MODEL_ID, conversations, params, false)
        .await;
    assert!(result.is_ok(), "Batched request failed: {:?}", result.err());

    let responses = match result.unwrap() {
        BatchChatResult::Complete(responses) => responses,
        BatchChatResult::Stream(_) => panic!("Expected complete result, got stream"),
    };
    assert_eq!(
        responses.len(),
        2,
        "Should have 2 responses (one per conversation)"
    );

    let mut output_lines = Vec::new();
    for (i, response) in responses.iter().enumerate() {
        output_lines.push(format!("Response {}: {}", i, response.text));
    }
    if !output_lines.is_empty() {
        println!("{}", output_lines.join("\n"));
    }

    for (i, response) in responses.iter().enumerate() {
        assert!(
            !response.text.is_empty(),
            "Response {} should have content",
            i
        );
        assert!(
            response.finish_reason.is_some(),
            "Response {} should have finish reason",
            i
        );
    }
}

/// Test batched requests with different content per conversation.
/// Mirrors: test_e2e_batching.py::test_chat_completion_batched_heterogeneous (prompts only)
#[tokio::test]
#[ignore]
async fn test_chat_completion_batched_different_content() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 20,
        temperature: 0.0,
        ..Default::default()
    };

    let conversations = vec![
        vec![make_message("user", "Respond with a single word greeting.")],
        vec![make_message(
            "user",
            "List three colors separated by commas.",
        )],
    ];

    let result = client
        .achat_batch(MODEL_ID, conversations, params, false)
        .await;
    assert!(result.is_ok(), "Batched request failed: {:?}", result.err());

    let responses = match result.unwrap() {
        BatchChatResult::Complete(responses) => responses,
        BatchChatResult::Stream(_) => panic!("Expected complete result, got stream"),
    };
    assert_eq!(responses.len(), 2, "Should have 2 responses");

    let output_lines = [
        format!("Greeting: {}", responses[0].text.trim()),
        format!("Colors: {}", responses[1].text.trim()),
    ];
    println!("{}", output_lines.join("\n"));

    assert!(
        responses[0].finish_reason.is_some(),
        "Greeting should have finish reason"
    );
    assert!(
        responses[1].finish_reason.is_some(),
        "Colors should have finish reason"
    );
}

/// Test that empty batch returns empty responses.
#[tokio::test]
#[ignore]
async fn test_empty_batch() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams::default();
    let conversations: Vec<Vec<HashMap<String, serde_json::Value>>> = vec![];

    let result = client
        .achat_batch(MODEL_ID, conversations, params, false)
        .await;
    match result {
        Ok(BatchChatResult::Complete(responses)) => {
            assert!(
                responses.is_empty(),
                "Empty batch should return empty responses"
            )
        }
        Ok(BatchChatResult::Stream(_)) => panic!("Expected complete result, got stream"),
        Err(e) => panic!("Empty batch should succeed: {:?}", e),
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use orchard::ipc::serialization::{build_batch_request_payload, PromptPayload, RequestType};

    /// Test that batch serialization produces valid metadata with prompt_index.
    #[test]
    fn test_batch_serialization_prompt_index() {
        let prompts = vec![
            PromptPayload {
                prompt: "First prompt".to_string(),
                max_generated_tokens: 10,
                ..Default::default()
            },
            PromptPayload {
                prompt: "Second prompt".to_string(),
                max_generated_tokens: 20,
                ..Default::default()
            },
        ];

        let payload = build_batch_request_payload(
            1,
            "test-model",
            "/path/to/model",
            RequestType::Generation,
            12345,
            &prompts,
        )
        .expect("Failed to build batch payload");

        // Parse metadata to verify prompt_index
        let length = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
        let metadata: serde_json::Value = serde_json::from_slice(&payload[4..4 + length]).unwrap();

        let prompts_meta = metadata["prompts"]
            .as_array()
            .expect("prompts should be an array");
        assert_eq!(prompts_meta.len(), 2, "Should have 2 prompt entries");
        assert_eq!(
            prompts_meta[0]["prompt_index"], 0,
            "First prompt should have index 0"
        );
        assert_eq!(
            prompts_meta[1]["prompt_index"], 1,
            "Second prompt should have index 1"
        );
    }

    /// Test conversation construction.
    #[test]
    fn test_batch_conversation_construction() {
        let conversations = [
            vec![make_message("user", "Hello")],
            vec![make_message("user", "World")],
        ];

        assert_eq!(conversations.len(), 2);
        assert_eq!(conversations[0].len(), 1);
        assert_eq!(conversations[1].len(), 1);
    }
}
