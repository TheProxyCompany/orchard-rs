//! End-to-end stop sequence tests.
//!
//! Mirrors orchard-py/tests/test_e2e_stop_sequences.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message};

const MODEL_ID: &str = "moondream3";

/// Test stop sequence on "blue" - should output red, white, blue and stop at blue.
/// Mirrors: test_e2e_stop_sequences.py::test_chat_completion_respects_stop_sequence
#[tokio::test]
#[ignore]
async fn test_chat_completion_respects_stop_sequence() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        temperature: 0.0,
        stop: vec!["blue".to_string()],
        top_logprobs: 10, // logprobs=True, top_logprobs=10 in Python
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "What are the national colors of the United States of America?",
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(
        result.is_ok(),
        "Stop sequence test failed: {:?}",
        result.err()
    );

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            let content = response.text.to_lowercase();
            println!("{}", response.text);

            assert!(content.contains("red"), "Expected 'red' in response");
            assert!(content.contains("white"), "Expected 'white' in response");
            assert!(content.contains("blue"), "Expected 'blue' in response");
            assert!(
                content.ends_with("blue"),
                "Expected response to end with 'blue' but got: '{}'",
                response.text
            );
            assert_eq!(
                response.finish_reason.as_deref().map(|s| s.to_lowercase()),
                Some("stop".to_string()),
                "Finish reason should be 'stop'"
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}
