//! End-to-end embedded client tests.
//!
//! Mirrors orchard-py/tests/test_e2e_client.py
//! Tests the embedded IPC client with both streaming and non-streaming.
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message, ALL_MODELS};

async fn run_client_chat_non_streaming(prompt: &str) {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    for &model_id in ALL_MODELS {
        let params = SamplingParams {
            max_tokens: 5,
            temperature: 0.0,
            ..Default::default()
        };

        let messages = vec![make_message("user", prompt)];
        let mut output_lines = vec![format!("User: {}", prompt)];

        let result = client.achat(model_id, messages, params, false).await;
        assert!(
            result.is_ok(),
            "Chat request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Complete(response) => {
                output_lines.push(format!("{}: {}", model_id, response.text));
                println!("{}", output_lines.join("\n"));
                assert!(
                    !response.text.trim().is_empty(),
                    "Response should have content for {}",
                    model_id
                );
                assert!(
                    response.usage.completion_tokens > 0,
                    "Should have generated tokens for {}",
                    model_id
                );
                assert_eq!(
                    response.usage.completion_tokens, 5,
                    "Expected exactly 5 completion tokens for {}, got {}",
                    model_id, response.usage.completion_tokens
                );
            }
            orchard::ChatResult::Stream(_) => {
                panic!("Expected complete response, got stream for {}", model_id);
            }
        }
    }
}

async fn run_client_chat_streaming(prompt: &str) {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    for &model_id in ALL_MODELS {
        let params = SamplingParams {
            max_tokens: 96,
            temperature: 0.7,
            ..Default::default()
        };

        let messages = vec![make_message("user", prompt)];
        let mut output_lines = vec![format!("User: {}", prompt)];

        let result = client.achat(model_id, messages, params, true).await;
        assert!(
            result.is_ok(),
            "Chat request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Stream(mut stream) => {
                let mut deltas = Vec::new();
                let mut content = String::new();

                while let Some(delta) = stream.recv().await {
                    if let Some(text) = &delta.content {
                        content.push_str(text);
                    }
                    deltas.push(delta);
                }

                output_lines.push(format!("{}: {}", model_id, content));
                println!("{}", output_lines.join("\n"));

                assert!(
                    deltas.len() > 1,
                    "Expected multiple deltas for {}, got {}",
                    model_id,
                    deltas.len()
                );
                assert!(
                    !content.trim().is_empty(),
                    "Expected non-empty content for {}",
                    model_id
                );
            }
            orchard::ChatResult::Complete(_) => {
                panic!("Expected stream, got complete response for {}", model_id);
            }
        }
    }
}

/// Test non-streaming chat with exact token count.
/// Mirrors: test_e2e_client.py::test_client_chat_non_streaming
#[tokio::test]
#[ignore]
async fn test_client_chat_non_streaming_poem() {
    run_client_chat_non_streaming("You have 5 output tokens. Respond with a 5 token poem.")
        .await;
}

/// Test non-streaming chat with exact token count.
/// Mirrors: test_e2e_client.py::test_client_chat_non_streaming
#[tokio::test]
#[ignore]
async fn test_client_chat_non_streaming_plea() {
    run_client_chat_non_streaming(
        "You have 5 output tokens. Respond with a 5 token plea for more tokens.",
    )
    .await;
}

/// Test streaming chat.
/// Mirrors: test_e2e_client.py::test_client_chat_streaming
#[tokio::test]
#[ignore]
async fn test_client_chat_streaming_artist() {
    run_client_chat_streaming("Respond with your favorite musical artist of the last 10 years.")
        .await;
}

/// Test streaming chat.
/// Mirrors: test_e2e_client.py::test_client_chat_streaming
#[tokio::test]
#[ignore]
async fn test_client_chat_streaming_movie() {
    run_client_chat_streaming("Respond with your favorite movie of the last 10 years.").await;
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
