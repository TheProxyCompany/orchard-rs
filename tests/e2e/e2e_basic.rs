//! End-to-end basic chat completion tests.
//!
//! Mirrors orchard-py/tests/test_e2e_basic.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message, TEXT_MODELS};

/// Test basic non-streaming chat completion with a single token.
/// Mirrors: test_e2e_basic.py::test_chat_completion_first_token
#[tokio::test]
#[ignore]
async fn test_chat_completion_first_token() {
    let fixture = get_fixture().await;
    for &model_id in TEXT_MODELS {
        let params = SamplingParams {
            max_tokens: 1,
            temperature: 1.0,
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
    }
}

/// Test multi-token generation with deterministic sampling.
/// Mirrors: test_e2e_basic.py::test_chat_completion_multi_token
#[tokio::test]
#[ignore]
async fn test_chat_completion_multi_token() {
    let fixture = get_fixture().await;
    for &model_id in TEXT_MODELS {
        let params = SamplingParams {
            max_tokens: 64, // max_completion_tokens in Python
            temperature: 0.0,
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
    }
}
