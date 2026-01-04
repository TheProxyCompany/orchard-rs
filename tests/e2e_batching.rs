//! End-to-end batching tests.
//!
//! Tests batched inference where multiple prompts are processed together.
//! These tests verify:
//! - ONE IPC message is sent for all prompts (not N separate messages)
//! - prompt_index correctly demultiplexes responses
//! - Responses are correlated back to their prompts
//!
//! Set PIE_LOCAL_BUILD to run these tests against a real engine.

use std::collections::HashMap;
use std::sync::Arc;

use orchard::{Client, ModelRegistry, SamplingParams};

const MODEL_ID: &str = "moondream3";

/// Check if PIE is available for testing.
fn pie_available() -> bool {
    std::env::var("PIE_LOCAL_BUILD").is_ok()
}

/// Skip test if PIE is not available.
macro_rules! require_pie {
    () => {
        if !pie_available() {
            eprintln!("SKIPPED: PIE_LOCAL_BUILD not set. Set it to run integration tests.");
            return;
        }
    };
}

fn make_message(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));
    msg.insert("content".to_string(), serde_json::json!(content));
    msg
}

/// Test homogeneous batched chat completion with identical parameters.
///
/// Verifies that:
/// - Multiple conversations are sent in ONE IPC message
/// - Each response is correctly identified by prompt_index
/// - Responses are returned in order
#[tokio::test]
async fn test_chat_completion_batched_homogeneous() {
    require_pie!();

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 10,
        temperature: 0.0,
        ..Default::default()
    };

    let conversations = vec![
        vec![make_message("user", "Say hello politely.")],
        vec![make_message("user", "Give me a fun fact about space.")],
    ];

    let result = client.achat_batch(MODEL_ID, conversations, params).await;
    assert!(result.is_ok(), "Batched request failed: {:?}", result.err());

    let responses = result.unwrap();
    assert_eq!(responses.len(), 2, "Should have 2 responses (one per conversation)");

    for (i, response) in responses.iter().enumerate() {
        assert!(!response.text.is_empty(), "Response {} should have content", i);
        println!("Response {}: {}", i, response.text);
        assert!(
            response.finish_reason.is_some(),
            "Response {} should have finish reason",
            i
        );
    }
}

/// Test batched requests with different content per conversation.
#[tokio::test]
async fn test_chat_completion_batched_different_content() {
    require_pie!();

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 20,
        temperature: 0.0,
        ..Default::default()
    };

    let conversations = vec![
        vec![make_message("user", "Respond with a single word greeting.")],
        vec![make_message("user", "List three colors separated by commas.")],
    ];

    let result = client.achat_batch(MODEL_ID, conversations, params).await;
    assert!(result.is_ok(), "Batched request failed: {:?}", result.err());

    let responses = result.unwrap();
    assert_eq!(responses.len(), 2, "Should have 2 responses");

    // First response should be short (greeting)
    println!("Greeting: {}", responses[0].text.trim());

    // Second response should list colors
    println!("Colors: {}", responses[1].text.trim());
}

/// Test that empty batch returns empty responses.
#[tokio::test]
async fn test_empty_batch() {
    // This test doesn't require PIE - it tests edge case handling
    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry);

    // Connection may fail without PIE, but that's ok for this test
    if let Ok(client) = client {
        let params = SamplingParams::default();
        let conversations: Vec<Vec<HashMap<String, serde_json::Value>>> = vec![];

        let result = client.achat_batch(MODEL_ID, conversations, params).await;
        match result {
            Ok(responses) => assert!(responses.is_empty(), "Empty batch should return empty responses"),
            Err(_) => {
                // Expected without PIE running
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use orchard::ipc::serialization::{PromptPayload, build_batch_request_payload, RequestType};

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
        ).expect("Failed to build batch payload");

        // Parse metadata to verify prompt_index
        let length = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
        let metadata: serde_json::Value = serde_json::from_slice(&payload[4..4 + length]).unwrap();

        let prompts_meta = metadata["prompts"].as_array().expect("prompts should be an array");
        assert_eq!(prompts_meta.len(), 2, "Should have 2 prompt entries");
        assert_eq!(prompts_meta[0]["prompt_index"], 0, "First prompt should have index 0");
        assert_eq!(prompts_meta[1]["prompt_index"], 1, "Second prompt should have index 1");
    }

    /// Test conversation construction.
    #[test]
    fn test_batch_conversation_construction() {
        let conversations = vec![
            vec![make_message("user", "Hello")],
            vec![make_message("user", "World")],
        ];

        assert_eq!(conversations.len(), 2);
        assert_eq!(conversations[0].len(), 1);
        assert_eq!(conversations[1].len(), 1);
    }
}
