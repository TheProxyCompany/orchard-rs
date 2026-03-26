//! End-to-end multi-token generation tests.
//!
//! Mirrors orchard-py/tests/test_e2e_multi_token.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message, TEXT_MODELS};

/// Test multi-token non-streaming - "What is the capital of France?" should produce "Paris".
/// Mirrors: test_e2e_multi_token.py::test_chat_completion_multi_token_non_streaming
#[tokio::test]
#[ignore]
async fn test_chat_completion_multi_token_non_streaming() {
    let fixture = get_fixture().await;
    let client = &fixture.client;
    for &model_id in TEXT_MODELS {
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 10,
            top_logprobs: 5,
            ..Default::default()
        };

        let messages = vec![make_message("user", "What is the capital of France?")];

        let result = client.achat(model_id, messages, params, false).await;
        assert!(
            result.is_ok(),
            "Chat request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Complete(response) => {
                let content = &response.text;

                if !content.contains("Paris") {
                    println!("{} response did not contain 'Paris': {}", model_id, content);
                }
                assert!(
                    content.contains("Paris"),
                    "Expected 'Paris' in response for {} but got: '{}'",
                    model_id,
                    content
                );
                println!("{}: {}", model_id, content);

                let finish_reason = response
                    .finish_reason
                    .as_deref()
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                assert!(
                    finish_reason == "stop" || finish_reason == "length",
                    "Unexpected finish reason for {}: {}",
                    model_id,
                    finish_reason
                );

                assert!(
                    response.usage.prompt_tokens > 0,
                    "Expected input_tokens > 0 for {}",
                    model_id
                );
                assert_eq!(
                    response.usage.total_tokens,
                    response.usage.prompt_tokens + response.usage.completion_tokens,
                    "total_tokens should equal input_tokens + output_tokens for {}",
                    model_id
                );
            }
            orchard::ChatResult::Stream(_) => {
                panic!("Expected complete response, got stream for {}", model_id);
            }
        }
    }
}

/// Test multi-token streaming chat completion.
/// Mirrors: test_e2e_multi_token.py::test_chat_completion_multi_token_streaming
#[tokio::test]
#[ignore]
async fn test_chat_completion_multi_token_streaming() {
    let fixture = get_fixture().await;
    let client = &fixture.client;
    for &model_id in TEXT_MODELS {
        let params = SamplingParams {
            max_tokens: 10,
            temperature: 0.0,
            ..Default::default()
        };

        let messages = vec![make_message(
            "user",
            "Tell me a very short story in one sentence.",
        )];

        let result = client.achat(model_id, messages, params, true).await;
        assert!(
            result.is_ok(),
            "Chat request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Stream(mut stream) => {
                let mut chunks_received = 0;
                let mut full_content = String::new();
                let mut finish_reason: Option<String> = None;

                while let Some(delta) = stream.recv().await {
                    chunks_received += 1;
                    if let Some(content) = &delta.content {
                        full_content.push_str(content);
                    }
                    if let Some(reason) = &delta.finish_reason {
                        finish_reason = Some(reason.clone());
                    }
                }

                assert!(
                    chunks_received > 1,
                    "Expected multiple chunks for {}, but got {}",
                    model_id,
                    chunks_received
                );
                assert!(
                    !full_content.is_empty(),
                    "Expected non-empty content for {}",
                    model_id
                );
                assert!(finish_reason.is_some(), "Expected a finish reason");

                let reason = finish_reason.unwrap().to_lowercase();
                assert!(
                    reason == "stop" || reason == "length",
                    "Unexpected finish reason for {}: {}",
                    model_id,
                    reason
                );
            }
            orchard::ChatResult::Complete(_) => {
                panic!("Expected stream, got complete response for {}", model_id);
            }
        }
    }
}
