//! End-to-end multi-candidate generation tests.
//!
//! Mirrors orchard-py/tests/test_e2e_multi_candidate.py
//! Run with: cargo test --test e2e -- --ignored

use std::collections::HashMap;

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message};

const MODEL_ID: &str = "meta-llama/Llama-3.1-8B-Instruct";

/// Test non-streaming multi-candidate responses return the expected number of choices.
/// Mirrors: test_e2e_multi_candidate.py::test_chat_completion_multi_candidate_non_streaming
#[tokio::test]
#[ignore]
async fn test_chat_completion_multi_candidate_non_streaming() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let candidate_count = 3;
    let params = SamplingParams {
        max_tokens: 10, // max_completion_tokens in Python
        temperature: 0.0,
        n: candidate_count,
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "Provide three brief facts about the moon.",
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            // Verify we got content
            assert!(!response.text.is_empty(), "Response should have content");
            assert!(
                response.finish_reason.is_some(),
                "Response should have finish_reason"
            );

            // Verify candidate count by checking deltas
            let candidate_indices: std::collections::HashSet<u32> = response
                .deltas
                .iter()
                .map(|d| d.candidate_index.unwrap_or(0))
                .collect();

            assert_eq!(
                candidate_indices.len(),
                candidate_count as usize,
                "Should have {} distinct candidates, got {:?}",
                candidate_count,
                candidate_indices
            );

            // Verify candidate indices are 0..candidate_count
            let expected_indices: std::collections::HashSet<u32> =
                (0..candidate_count as u32).collect();
            assert_eq!(
                candidate_indices, expected_indices,
                "Candidate indices should be 0..{}",
                candidate_count
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test streaming multi-candidate responses can be reconstructed per candidate index.
/// Mirrors: test_e2e_multi_candidate.py::test_chat_completion_multi_candidate_streaming
#[tokio::test]
#[ignore]
async fn test_chat_completion_multi_candidate_streaming() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let candidate_count = 3;
    let params = SamplingParams {
        max_tokens: 10, // max_completion_tokens in Python
        temperature: 0.0,
        n: candidate_count,
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "Stream three short tips for studying effectively.",
    )];

    let result = client.achat(MODEL_ID, messages, params, true).await;
    assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Stream(mut stream) => {
            let mut candidate_contents: HashMap<u32, Vec<String>> = HashMap::new();
            let mut finish_reasons: HashMap<u32, String> = HashMap::new();

            while let Some(delta) = stream.recv().await {
                let index = delta.candidate_index.unwrap_or(0);
                if let Some(content) = &delta.content {
                    candidate_contents
                        .entry(index)
                        .or_default()
                        .push(content.clone());
                }
                if let Some(reason) = &delta.finish_reason {
                    finish_reasons.insert(index, reason.clone());
                }
            }

            // Verify we received the expected number of candidates
            assert_eq!(
                candidate_contents.len(),
                candidate_count as usize,
                "Should have received {} candidates",
                candidate_count
            );

            // Verify candidate indices are 0..candidate_count
            let expected_indices: std::collections::HashSet<u32> =
                (0..candidate_count as u32).collect();
            let actual_indices: std::collections::HashSet<u32> =
                candidate_contents.keys().copied().collect();
            assert_eq!(
                actual_indices, expected_indices,
                "Candidate indices should be 0..{}",
                candidate_count
            );

            // Verify finish reasons exist for all candidates
            for idx in 0..candidate_count as u32 {
                assert!(
                    finish_reasons.contains_key(&idx),
                    "Candidate {} missing finish reason",
                    idx
                );
            }
        }
        orchard::ChatResult::Complete(_) => {
            panic!("Expected stream, got complete response");
        }
    }
}
