//! End-to-end batching tests.
//!
//! Mirrors orchard-py/tests/test_e2e_batching.py
//! Run with: cargo test --test e2e -- --ignored
//!
//! Note: test_chat_completion_batch_length_mismatch_returns_422 is HTTP-specific
//! (validates 422 response code) and is not ported.

use orchard::{BatchChatResult, SamplingParams};

use crate::fixture::{get_fixture, make_message};

const MODEL_ID: &str = "moondream3";

/// Test homogeneous batched chat completion with identical parameters.
/// Mirrors: test_e2e_batching.py::test_chat_completion_batched_homogeneous
#[tokio::test]
#[ignore]
async fn test_chat_completion_batched_homogeneous() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 10, // max_completion_tokens in Python
        temperature: 0.0,
        ..Default::default()
    };

    let conversations = vec![
        vec![make_message("user", "Say hello politely.")],
        vec![make_message("user", "Give me a fun fact about space.")],
    ];

    let result = client
        .achat_batch(MODEL_ID, conversations, params, false)
        .await;
    assert!(result.is_ok(), "Batched request failed: {:?}", result.err());

    let responses = match result.unwrap() {
        BatchChatResult::Complete(responses) => responses,
        BatchChatResult::Stream(_) => panic!("Expected complete result, got stream"),
    };
    assert_eq!(responses.len(), 2);

    for (index, response) in responses.iter().enumerate() {
        println!("{}", response.text);
        assert!(!response.text.is_empty(), "Response {} should have content", index);
        assert!(response.finish_reason.is_some(), "Response {} should have finish_reason", index);
    }
}

/// Test batched requests with different content and parameters per conversation.
/// Mirrors: test_e2e_batching.py::test_chat_completion_batched_heterogeneous
///
/// Note: Python test uses per-prompt parameter arrays. The IPC client takes uniform
/// params, so we use uniform parameters but the same prompts.
#[tokio::test]
#[ignore]
async fn test_chat_completion_batched_heterogeneous() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 4,
        temperature: 0.0,
        ..Default::default()
    };

    let conversations = vec![
        vec![make_message("user", "Respond with a single word greeting.")],
        vec![make_message(
            "user",
            "List three colors separated by commas.",
        )],
    ];

    let result = client
        .achat_batch(MODEL_ID, conversations, params, false)
        .await;
    assert!(result.is_ok(), "Batched request failed: {:?}", result.err());

    let responses = match result.unwrap() {
        BatchChatResult::Complete(responses) => responses,
        BatchChatResult::Stream(_) => panic!("Expected complete result, got stream"),
    };
    assert_eq!(responses.len(), 2);

    assert!(responses[0].finish_reason.is_some());
    assert!(responses[1].finish_reason.is_some());
}

// test_chat_completion_batch_length_mismatch_returns_422 is HTTP-specific (validates
// 422 response for mismatched parameter array lengths) - not applicable to IPC client.
