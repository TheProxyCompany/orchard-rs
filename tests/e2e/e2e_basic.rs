//! End-to-end basic chat completion tests.
//!
//! Mirrors orchard-py/tests/test_e2e_basic.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message};

const MODEL_ID: &str = "meta-llama/Llama-3.1-8B-Instruct";

/// Test basic non-streaming chat completion with a single token.
/// Mirrors: test_e2e_basic.py::test_chat_completion_first_token
#[tokio::test]
#[ignore]
async fn test_chat_completion_first_token() {
    let fixture = get_fixture().await;

    let params = SamplingParams {
        max_tokens: 1, // max_completion_tokens in Python
        temperature: 1.0,
        ..Default::default()
    };

    let messages = vec![make_message("user", "Hello!")];

    let result = fixture
        .client
        .achat(MODEL_ID, messages, params, false)
        .await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            assert!(
                !response.text.is_empty(),
                "Response text should not be empty"
            );
            assert!(
                response.finish_reason.is_some(),
                "Should have a finish reason"
            );
            let reason = response.finish_reason.unwrap().to_lowercase();
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
/// Mirrors: test_e2e_basic.py::test_chat_completion_multi_token
#[tokio::test]
#[ignore]
async fn test_chat_completion_multi_token() {
    let fixture = get_fixture().await;

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
        .achat(MODEL_ID, messages, params, false)
        .await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            assert!(
                !response.text.is_empty(),
                "Response text should not be empty"
            );
            println!("{}", response.text);
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}
