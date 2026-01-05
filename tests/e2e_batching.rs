//! End-to-end batching tests.
//!
//! Mirrors orchard-py/tests/test_e2e_batching.py
//! Tests batched inference where multiple prompts are processed together.
//! Run with: cargo test -- --ignored

use std::collections::HashMap;
use std::sync::Arc;

use orchard::{BatchChatResult, Client, InferenceEngine, ModelRegistry, SamplingParams};

const MODEL_ID: &str = "moondream3";

fn make_message(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));
    msg.insert("content".to_string(), serde_json::json!(content));
    msg
}

/// Test homogeneous batched chat completion with identical parameters.
/// Mirrors: test_e2e_batching.py::test_chat_completion_batched_homogeneous
#[tokio::test]
#[ignore]
async fn test_chat_completion_batched_homogeneous() {
    let _engine = InferenceEngine::new().await.expect("Failed to start engine");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 10,
        temperature: 0.0,
        ..Default::default()
    };

    let conversations = vec![
        vec![make_message("user", "Say hello politely.")],
        vec![make_message("user", "Give me a fun fact about space.")],
    ];

    let result = client.achat_batch(MODEL_ID, conversations, params, false).await;
    assert!(result.is_ok(), "Batched request failed: {:?}", result.err());

    let responses = match result.unwrap() {
        BatchChatResult::Complete(responses) => responses,
        BatchChatResult::Stream(_) => panic!("Expected complete result, got stream"),
    };
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
/// Mirrors: test_e2e_batching.py::test_chat_completion_batched_heterogeneous (prompts only)
#[tokio::test]
#[ignore]
async fn test_chat_completion_batched_different_content() {
    let _engine = InferenceEngine::new().await.expect("Failed to start engine");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 20,
        temperature: 0.0,
        ..Default::default()
    };

    let conversations = vec![
        vec![make_message("user", "Respond with a single word greeting.")],
        vec![make_message("user", "List three colors separated by commas.")],
    ];

    let result = client.achat_batch(MODEL_ID, conversations, params, false).await;
    assert!(result.is_ok(), "Batched request failed: {:?}", result.err());

    let responses = match result.unwrap() {
        BatchChatResult::Complete(responses) => responses,
        BatchChatResult::Stream(_) => panic!("Expected complete result, got stream"),
    };
    assert_eq!(responses.len(), 2, "Should have 2 responses");

    println!("Greeting: {}", responses[0].text.trim());
    println!("Colors: {}", responses[1].text.trim());

    // Both should have content
    assert!(!responses[0].text.trim().is_empty(), "Greeting should not be empty");
    assert!(!responses[1].text.trim().is_empty(), "Colors should not be empty");
}

/// Test that empty batch returns empty responses.
#[tokio::test]
#[ignore]
async fn test_empty_batch() {
    let _engine = InferenceEngine::new().await.expect("Failed to start engine");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let params = SamplingParams::default();
    let conversations: Vec<Vec<HashMap<String, serde_json::Value>>> = vec![];

    let result = client.achat_batch(MODEL_ID, conversations, params, false).await;
    match result {
        Ok(BatchChatResult::Complete(responses)) => {
            assert!(responses.is_empty(), "Empty batch should return empty responses")
        }
        Ok(BatchChatResult::Stream(_)) => panic!("Expected complete result, got stream"),
        Err(e) => panic!("Empty batch should succeed: {:?}", e),
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
