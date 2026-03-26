//! End-to-end best_of tests.
//!
//! Mirrors orchard-py/tests/test_e2e_best_of.py
//! Run with: cargo test --test e2e -- --ignored
//!
//! Note: Validation tests (best_of < n returns 422, streaming+best_of returns 422)
//! are HTTP-specific and not ported.

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message, TEXT_MODELS};

/// Test that best_of fan-out returns only the top-n candidates while reflecting total work in usage.
/// Mirrors: test_e2e_best_of.py::test_chat_completion_best_of_selects_top_n
#[tokio::test]
#[ignore]
async fn test_chat_completion_best_of_selects_top_n() {
    let fixture = get_fixture().await;
    let client = &fixture.client;
    for &model_id in TEXT_MODELS {
        let best_of = 3;
        let params = SamplingParams {
            max_tokens: 8, // max_completion_tokens in Python
            temperature: 0.2,
            n: 1,
            best_of: Some(best_of),
            ..Default::default()
        };

        let messages = vec![make_message("user", "List one fun fact about penguins.")];

        let result = client.achat(model_id, messages, params, false).await;
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
                    "Response should have content for {}",
                    model_id
                );
                println!("{}: {}", model_id, response.text);

                assert!(
                    response.usage.completion_tokens >= best_of as u32,
                    "output_tokens should reflect all best_of candidates for {}",
                    model_id
                );
                assert!(
                    response.usage.total_tokens >= response.usage.completion_tokens,
                    "total_tokens should be >= output_tokens for {}",
                    model_id
                );
            }
            orchard::ChatResult::Stream(_) => {
                panic!("Expected complete response, got stream for {}", model_id);
            }
        }
    }
}

// test_chat_completion_best_of_validation_less_than_n is HTTP-specific (validates 422 response)
// test_chat_completion_best_of_streaming_disallowed is HTTP-specific (validates 422 response)
