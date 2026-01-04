//! End-to-end basic chat completion tests.
//!
//! These tests require a running PIE instance and model weights.
//! Set PIE_LOCAL_BUILD to run integration tests.

use std::collections::HashMap;
use std::sync::Arc;

use orchard::{Client, ModelRegistry, SamplingParams};

const MODEL_ID: &str = "meta-llama/Llama-3.1-8B-Instruct";

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

/// Test basic non-streaming chat completion with a single token.
#[tokio::test]
async fn test_chat_completion_first_token() {
    require_pie!();

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 1,
        temperature: 1.0,
        ..Default::default()
    };

    let messages = vec![make_message("user", "Hello!")];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            assert!(!response.text.is_empty(), "Response text should not be empty");
            assert!(
                response.finish_reason.is_some(),
                "Should have a finish reason"
            );
            let reason = response.finish_reason.unwrap();
            assert!(
                reason == "length" || reason == "stop",
                "Unexpected finish reason: {}",
                reason
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test multi-token generation with deterministic sampling.
#[tokio::test]
async fn test_chat_completion_multi_token() {
    require_pie!();

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 64,
        temperature: 0.0, // Greedy sampling
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "Provide one friendly sentence introducing yourself.",
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            assert!(!response.text.is_empty(), "Response text should not be empty");
            println!("Generated text: {}", response.text);
            assert!(
                response.usage.completion_tokens > 0,
                "Should have generated tokens"
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test synchronous chat interface.
#[test]
fn test_sync_chat_completion() {
    require_pie!();

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 32,
        temperature: 0.7,
        ..Default::default()
    };

    let messages = vec![make_message("user", "Say hello in one sentence.")];

    let result = client.chat(MODEL_ID, messages, params);
    assert!(result.is_ok(), "Sync chat request failed: {:?}", result.err());

    let response = result.unwrap();
    assert!(!response.text.is_empty(), "Response text should not be empty");
    println!("Sync response: {}", response.text);
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_make_message() {
        let msg = make_message("user", "Hello");
        assert_eq!(msg.get("role").unwrap().as_str(), Some("user"));
        assert_eq!(msg.get("content").unwrap().as_str(), Some("Hello"));
    }

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert_eq!(params.max_tokens, 1024);
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.n, 1);
    }
}
