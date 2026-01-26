//! End-to-end multi-token generation tests.
//!
//! Mirrors orchard-py/tests/test_e2e_multi_token.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message};

const MODEL_ID: &str = "meta-llama/Llama-3.1-8B-Instruct";

/// Test multi-token non-streaming - "What is the capital of France?" should produce "Paris".
/// Mirrors: test_e2e_multi_token.py::test_chat_completion_multi_token_non_streaming
#[tokio::test]
#[ignore]
async fn test_chat_completion_multi_token_non_streaming() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: 10,  // max_completion_tokens in Python
        top_logprobs: 5, // logprobs=True, top_logprobs=5 in Python
        ..Default::default()
    };

    let messages = vec![make_message("user", "What is the capital of France?")];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            let content = &response.text;

            // The model's greedy output should contain "Paris"
            if !content.contains("Paris") {
                // Print logprobs for debugging (matching Python behavior)
                println!("Response did not contain 'Paris': {}", content);
            }
            assert!(
                content.contains("Paris"),
                "Expected 'Paris' in response but got: '{}'",
                content
            );
            println!("{}", content);

            // Assert finish reason
            let finish_reason = response
                .finish_reason
                .as_deref()
                .map(|s| s.to_lowercase())
                .unwrap_or_default();
            assert!(
                finish_reason == "stop" || finish_reason == "length",
                "Unexpected finish reason: {}",
                finish_reason
            );

            // Assert usage fields
            assert!(
                response.usage.prompt_tokens > 0,
                "Expected input_tokens > 0"
            );
            assert_eq!(
                response.usage.total_tokens,
                response.usage.prompt_tokens + response.usage.completion_tokens,
                "total_tokens should equal input_tokens + output_tokens"
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
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

    let params = SamplingParams {
        max_tokens: 10, // max_completion_tokens in Python
        temperature: 0.0,
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "Tell me a very short story in one sentence.",
    )];

    let result = client.achat(MODEL_ID, messages, params, true).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

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
                "Expected multiple chunks, but got {}",
                chunks_received
            );
            assert!(!full_content.is_empty(), "Expected non-empty content");
            assert!(finish_reason.is_some(), "Expected a finish reason");

            let reason = finish_reason.unwrap().to_lowercase();
            assert!(
                reason == "stop" || reason == "length",
                "Unexpected finish reason: {}",
                reason
            );
        }
        orchard::ChatResult::Complete(_) => {
            panic!("Expected stream, got complete response");
        }
    }
}
