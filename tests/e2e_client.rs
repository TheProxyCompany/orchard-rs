//! End-to-end embedded client tests.
//!
//! Mirrors orchard-py/tests/test_e2e_client.py
//! Tests the embedded IPC client with both streaming and non-streaming.
//! Run with: cargo test -- --ignored

use std::collections::HashMap;
use std::sync::Arc;

use orchard::{Client, InferenceEngine, ModelRegistry, SamplingParams};

fn make_message(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));
    msg.insert("content".to_string(), serde_json::json!(content));
    msg
}

/// Test non-streaming chat with exact token count - Llama model
/// Mirrors: test_e2e_client.py::test_client_chat_non_streaming (Llama variant)
#[tokio::test]
#[ignore]
async fn test_client_chat_non_streaming_llama_poem() {
    let _engine = InferenceEngine::new().await.expect("Failed to start engine");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 5,
        temperature: 0.0,
        ..Default::default()
    };

    let prompt = "You have 5 output tokens. Respond with a 5 token poem.";
    let messages = vec![make_message("user", prompt)];

    println!("User: {}", prompt);

    let result = client.achat("meta-llama/Llama-3.1-8B-Instruct", messages, params, false).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            assert!(!response.text.trim().is_empty(), "Response should have content");
            println!("meta-llama/Llama-3.1-8B-Instruct: {}", response.text);
            assert!(
                response.usage.completion_tokens > 0,
                "Should have generated tokens"
            );
            assert_eq!(
                response.usage.completion_tokens, 5,
                "Expected exactly 5 completion tokens, got {}",
                response.usage.completion_tokens
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test non-streaming chat with exact token count - Llama model (plea variant)
/// Mirrors: test_e2e_client.py::test_client_chat_non_streaming (Llama plea variant)
#[tokio::test]
#[ignore]
async fn test_client_chat_non_streaming_llama_plea() {
    let _engine = InferenceEngine::new().await.expect("Failed to start engine");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 5,
        temperature: 0.0,
        ..Default::default()
    };

    let prompt = "You have 5 output tokens. Respond with a 5 token plea for more tokens.";
    let messages = vec![make_message("user", prompt)];

    println!("User: {}", prompt);

    let result = client.achat("meta-llama/Llama-3.1-8B-Instruct", messages, params, false).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            assert!(!response.text.trim().is_empty(), "Response should have content");
            println!("meta-llama/Llama-3.1-8B-Instruct: {}", response.text);
            assert_eq!(
                response.usage.completion_tokens, 5,
                "Expected exactly 5 completion tokens, got {}",
                response.usage.completion_tokens
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test non-streaming chat with exact token count - moondream3 model
/// Mirrors: test_e2e_client.py::test_client_chat_non_streaming (moondream3 variant)
#[tokio::test]
#[ignore]
async fn test_client_chat_non_streaming_moondream_poem() {
    let _engine = InferenceEngine::new().await.expect("Failed to start engine");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 5,
        temperature: 0.0,
        ..Default::default()
    };

    let prompt = "You have 5 output tokens. Respond with a 5 token poem.";
    let messages = vec![make_message("user", prompt)];

    println!("User: {}", prompt);

    let result = client.achat("moondream3", messages, params, false).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            assert!(!response.text.trim().is_empty(), "Response should have content");
            println!("moondream3: {}", response.text);
            assert_eq!(
                response.usage.completion_tokens, 5,
                "Expected exactly 5 completion tokens, got {}",
                response.usage.completion_tokens
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test streaming chat - Llama model (musical artist)
/// Mirrors: test_e2e_client.py::test_client_chat_streaming (Llama artist variant)
#[tokio::test]
#[ignore]
async fn test_client_chat_streaming_llama_artist() {
    let _engine = InferenceEngine::new().await.expect("Failed to start engine");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 96,
        temperature: 0.7,
        ..Default::default()
    };

    let prompt = "Respond with your favorite musical artist of the last 10 years.";
    let messages = vec![make_message("user", prompt)];

    println!("User: {}", prompt);
    print!("meta-llama/Llama-3.1-8B-Instruct: ");

    let result = client.achat("meta-llama/Llama-3.1-8B-Instruct", messages, params, true).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Stream(mut stream) => {
            let mut deltas = Vec::new();
            let mut content = String::new();

            while let Some(delta) = stream.recv().await {
                if let Some(text) = &delta.content {
                    print!("{}", text);
                    content.push_str(text);
                }
                deltas.push(delta);
            }
            println!();

            assert!(deltas.len() > 1, "Expected multiple deltas, got {}", deltas.len());
            assert!(!content.trim().is_empty(), "Expected non-empty content");
        }
        orchard::ChatResult::Complete(_) => {
            panic!("Expected stream, got complete response");
        }
    }
}

/// Test streaming chat - Llama model (movie)
/// Mirrors: test_e2e_client.py::test_client_chat_streaming (Llama movie variant)
#[tokio::test]
#[ignore]
async fn test_client_chat_streaming_llama_movie() {
    let _engine = InferenceEngine::new().await.expect("Failed to start engine");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 96,
        temperature: 0.7,
        ..Default::default()
    };

    let prompt = "Respond with your favorite movie of the last 10 years.";
    let messages = vec![make_message("user", prompt)];

    println!("User: {}", prompt);
    print!("meta-llama/Llama-3.1-8B-Instruct: ");

    let result = client.achat("meta-llama/Llama-3.1-8B-Instruct", messages, params, true).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Stream(mut stream) => {
            let mut deltas = Vec::new();
            let mut content = String::new();

            while let Some(delta) = stream.recv().await {
                if let Some(text) = &delta.content {
                    print!("{}", text);
                    content.push_str(text);
                }
                deltas.push(delta);
            }
            println!();

            assert!(deltas.len() > 1, "Expected multiple deltas, got {}", deltas.len());
            assert!(!content.trim().is_empty(), "Expected non-empty content");
        }
        orchard::ChatResult::Complete(_) => {
            panic!("Expected stream, got complete response");
        }
    }
}

/// Test streaming chat - moondream3 model (artist)
/// Mirrors: test_e2e_client.py::test_client_chat_streaming (moondream3 artist variant)
#[tokio::test]
#[ignore]
async fn test_client_chat_streaming_moondream_artist() {
    let _engine = InferenceEngine::new().await.expect("Failed to start engine");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 96,
        temperature: 0.7,
        ..Default::default()
    };

    let prompt = "Respond with your favorite musical artist of the last 10 years.";
    let messages = vec![make_message("user", prompt)];

    println!("User: {}", prompt);
    print!("moondream3: ");

    let result = client.achat("moondream3", messages, params, true).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Stream(mut stream) => {
            let mut deltas = Vec::new();
            let mut content = String::new();

            while let Some(delta) = stream.recv().await {
                if let Some(text) = &delta.content {
                    print!("{}", text);
                    content.push_str(text);
                }
                deltas.push(delta);
            }
            println!();

            assert!(deltas.len() > 1, "Expected multiple deltas, got {}", deltas.len());
            assert!(!content.trim().is_empty(), "Expected non-empty content");
        }
        orchard::ChatResult::Complete(_) => {
            panic!("Expected stream, got complete response");
        }
    }
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
}
