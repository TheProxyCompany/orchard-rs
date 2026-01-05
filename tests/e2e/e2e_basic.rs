//! End-to-end basic chat completion tests.
//!
//! Mirrors orchard-py/tests/test_e2e_basic.py and test_e2e_multi_token.py
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
        max_tokens: 1,
        temperature: 1.0,
        ..Default::default()
    };

    let messages = vec![make_message("user", "Hello!")];

    let result = fixture.client.achat(MODEL_ID, messages, params, false).await;
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

/// Test multi-token generation - "What is the capital of France?" should produce "Paris"
/// Mirrors: test_e2e_multi_token.py::test_chat_completion_multi_token_non_streaming
#[tokio::test]
#[ignore]
async fn test_chat_completion_capital_of_france() {
    let fixture = get_fixture().await;

    let params = SamplingParams {
        max_tokens: 10,
        temperature: 0.0,
        ..Default::default()
    };

    let messages = vec![make_message("user", "What is the capital of France?")];

    let result = fixture.client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            println!("Response: {}", response.text);
            assert!(
                response.text.contains("Paris"),
                "Expected 'Paris' in response but got: '{}'",
                response.text
            );
            assert!(
                response.usage.prompt_tokens > 0,
                "Expected prompt_tokens > 0"
            );
            assert!(
                response.usage.completion_tokens > 0,
                "Expected completion_tokens > 0"
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
        max_tokens: 64,
        temperature: 0.0,
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "Provide one friendly sentence introducing yourself.",
    )];

    let result = fixture.client.achat(MODEL_ID, messages, params, false).await;
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
