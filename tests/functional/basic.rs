//! End-to-end basic chat completion tests.
//!
//! Mirrors orchard-py/tests/functional/test_basic.py
//! Run with: cargo test --test functional

use orchard::SamplingParams;

use crate::fixture::{fanout, get_fixture, make_message, TEXT_MODELS};

/// Test basic non-streaming chat completion with a single token.
/// Mirrors: test_basic.py::test_chat_completion_first_token
#[tokio::test]
async fn test_chat_completion_first_token() {
    let fixture = get_fixture().await;
    fanout(TEXT_MODELS.iter().map(|&model_id| async move {
        let params = SamplingParams {
            max_tokens: 1,
            temperature: 1.0,
            reasoning: Some(false),
            ..Default::default()
        };

        let messages = vec![make_message("user", "Hello!")];

        let result = fixture
            .client
            .achat(model_id, messages, params, false)
            .await;
        assert!(
            result.is_ok(),
            "Chat request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Complete(response) => {
                assert!(
                    !response.text.is_empty(),
                    "Response text should not be empty for {}",
                    model_id
                );
                assert!(
                    response.finish_reason.is_some(),
                    "Should have a finish reason for {}",
                    model_id
                );
                let reason = response.finish_reason.unwrap().to_lowercase();
                assert!(
                    reason == "length" || reason == "stop",
                    "Unexpected finish reason for {}: {}",
                    model_id,
                    reason
                );
            }
            orchard::ChatResult::Stream(_) => {
                panic!("Expected complete response, got stream for {}", model_id);
            }
        }
    }))
    .await;
}

/// Test multi-token generation with deterministic sampling.
/// Mirrors: test_basic.py::test_chat_completion_multi_token
#[tokio::test]
async fn test_chat_completion_multi_token() {
    let fixture = get_fixture().await;
    fanout(TEXT_MODELS.iter().map(|&model_id| async move {
        let params = SamplingParams {
            max_tokens: 64, // max_completion_tokens in Python
            temperature: 0.0,
            reasoning: Some(false),
            ..Default::default()
        };

        let messages = vec![make_message(
            "user",
            "Provide one friendly sentence introducing yourself.",
        )];

        let result = fixture
            .client
            .achat(model_id, messages, params, false)
            .await;
        assert!(
            result.is_ok(),
            "Chat request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Complete(response) => {
                assert!(
                    !response.text.is_empty(),
                    "Response text should not be empty for {}",
                    model_id
                );
                println!("{}: {}", model_id, response.text);
            }
            orchard::ChatResult::Stream(_) => {
                panic!("Expected complete response, got stream for {}", model_id);
            }
        }
    }))
    .await;
}
