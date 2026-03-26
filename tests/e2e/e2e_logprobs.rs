//! End-to-end logprobs tests.
//!
//! Mirrors orchard-py/tests/test_e2e_logprobs.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message, TEXT_MODELS};

/// Test chat completion with logprobs enabled.
/// Mirrors: test_e2e_logprobs.py::test_chat_completion_with_logprobs
#[tokio::test]
#[ignore]
async fn test_chat_completion_with_logprobs() {
    let fixture = get_fixture().await;
    let client = &fixture.client;
    for &model_id in TEXT_MODELS {
        let params = SamplingParams {
            max_tokens: 3,
            temperature: 1.0,
            top_logprobs: 5,
            ..Default::default()
        };

        let messages = vec![make_message("user", "Say hello")];

        let result = client.achat(model_id, messages, params, false).await;
        assert!(
            result.is_ok(),
            "Chat request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Complete(response) => {
                assert!(!response.text.is_empty(), "Response should have content");

                let mut found_logprobs = false;
                for delta in &response.deltas {
                    if !delta.top_logprobs.is_empty() {
                        found_logprobs = true;
                        for token_logprob in &delta.top_logprobs {
                            assert!(
                                token_logprob.logprob.is_finite(),
                                "Logprob for token '{}' should be finite for {}: {}",
                                token_logprob.token,
                                model_id,
                                token_logprob.logprob
                            );
                        }
                    }
                }
                assert!(
                    found_logprobs,
                    "Should have received logprobs in at least one delta for {}",
                    model_id
                );
            }
            orchard::ChatResult::Stream(_) => {
                panic!("Expected complete response, got stream for {}", model_id);
            }
        }
    }
}

/// Test that when logprobs is not requested, they are not included.
/// Mirrors: test_e2e_logprobs.py::test_chat_completion_without_logprobs
#[tokio::test]
#[ignore]
async fn test_chat_completion_without_logprobs() {
    let fixture = get_fixture().await;
    let client = &fixture.client;
    for &model_id in TEXT_MODELS {
        let params = SamplingParams {
            max_tokens: 3,
            temperature: 1.0,
            top_logprobs: 0,
            ..Default::default()
        };

        let messages = vec![make_message("user", "Say hello")];

        let result = client.achat(model_id, messages, params, false).await;
        assert!(
            result.is_ok(),
            "Chat request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Complete(response) => {
                for delta in &response.deltas {
                    assert!(
                        delta.top_logprobs.is_empty(),
                        "Logprobs should be empty when not requested for {}",
                        model_id
                    );
                }
            }
            orchard::ChatResult::Stream(_) => {
                panic!("Expected complete response, got stream for {}", model_id);
            }
        }
    }
}

/// Test that logprobs work correctly with streaming responses.
/// Mirrors: test_e2e_logprobs.py::test_chat_completion_logprobs_streaming
#[tokio::test]
#[ignore]
async fn test_chat_completion_logprobs_streaming() {
    let fixture = get_fixture().await;
    let client = &fixture.client;
    for &model_id in TEXT_MODELS {
        let params = SamplingParams {
            max_tokens: 5,
            temperature: 1.0,
            top_logprobs: 3,
            ..Default::default()
        };

        let messages = vec![make_message("user", "Count to three")];

        let result = client.achat(model_id, messages, params, true).await;
        assert!(
            result.is_ok(),
            "Chat request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Stream(mut stream) => {
                let mut chunks = Vec::new();

                while let Some(delta) = stream.recv().await {
                    chunks.push(delta);
                }

                assert!(!chunks.is_empty(), "Should have received chunks");

                for chunk in &chunks {
                    if !chunk.top_logprobs.is_empty() {
                        for token_logprob in &chunk.top_logprobs {
                            assert!(
                                token_logprob.logprob.is_finite(),
                                "Logprob should be finite for {}",
                                model_id
                            );
                        }
                    }
                }
            }
            orchard::ChatResult::Complete(_) => {
                panic!("Expected stream, got complete response for {}", model_id);
            }
        }
    }
}
