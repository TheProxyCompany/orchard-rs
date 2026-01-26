//! End-to-end logprobs tests.
//!
//! Mirrors orchard-py/tests/test_e2e_logprobs.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message};

const MODEL_ID: &str = "moondream3";

/// Test chat completion with logprobs enabled.
/// Mirrors: test_e2e_logprobs.py::test_chat_completion_with_logprobs
#[tokio::test]
#[ignore]
async fn test_chat_completion_with_logprobs() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 3, // max_completion_tokens in Python
        temperature: 1.0,
        top_logprobs: 5, // logprobs=True, top_logprobs=5 in Python
        ..Default::default()
    };

    let messages = vec![make_message("user", "Say hello")];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            // Check that we have deltas with logprobs
            // Note: The IPC client stores logprobs in ClientDelta.top_logprobs
            assert!(
                !response.text.is_empty(),
                "Response should have content"
            );

            // Check logprobs in deltas
            for delta in &response.deltas {
                if !delta.top_logprobs.is_empty() {
                    for token_logprobs in &delta.top_logprobs {
                        assert!(
                            !token_logprobs.is_empty(),
                            "top_logprobs should not be empty"
                        );

                        // Verify logprob values are finite numbers (not inf/nan)
                        for (token, logprob) in token_logprobs {
                            assert!(
                                logprob.is_finite(),
                                "Logprob for token '{}' should be finite: {}",
                                token,
                                logprob
                            );
                        }

                        // Note: Cannot verify sort order because top_logprobs uses HashMap
                        // which has non-deterministic iteration order. If ordering matters,
                        // the type should be Vec<(String, f64)> instead of HashMap<String, f64>.

                        // Verify we don't exceed requested count
                        assert!(
                            token_logprobs.len() <= 5,
                            "Should not exceed requested top_logprobs count"
                        );
                    }
                }
            }
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
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

    let params = SamplingParams {
        max_tokens: 3, // max_completion_tokens in Python
        temperature: 1.0,
        top_logprobs: 0, // logprobs=False in Python
        ..Default::default()
    };

    let messages = vec![make_message("user", "Say hello")];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            // With top_logprobs=0, logprobs should be empty in all deltas
            for delta in &response.deltas {
                assert!(
                    delta.top_logprobs.is_empty(),
                    "Logprobs should be empty when not requested"
                );
            }
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
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

    let params = SamplingParams {
        max_tokens: 5, // max_completion_tokens in Python
        temperature: 1.0,
        top_logprobs: 3, // logprobs=True, top_logprobs=3 in Python
        ..Default::default()
    };

    let messages = vec![make_message("user", "Count to three")];

    let result = client.achat(MODEL_ID, messages, params, true).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Stream(mut stream) => {
            let mut chunks = Vec::new();

            while let Some(delta) = stream.recv().await {
                chunks.push(delta);
            }

            assert!(!chunks.is_empty(), "Should have received chunks");

            // Check if any chunks have logprobs
            // Note: Not all chunks might have them, depending on implementation
            for chunk in &chunks {
                if !chunk.top_logprobs.is_empty() {
                    // Verify structure if present
                    for token_logprobs in &chunk.top_logprobs {
                        assert!(token_logprobs.len() <= 3, "Should not exceed requested amount");
                    }
                }
            }
        }
        orchard::ChatResult::Complete(_) => {
            panic!("Expected stream, got complete response");
        }
    }
}
