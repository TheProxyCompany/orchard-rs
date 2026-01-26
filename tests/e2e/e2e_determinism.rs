//! End-to-end determinism tests.
//!
//! Mirrors orchard-py/tests/test_e2e_determinism.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message};

/// Test multi-candidate determinism - all candidates should be identical with temp=0.
/// Mirrors: test_e2e_determinism.py::test_multi_candidate_determinism
///
/// Python parametrizes over models and batch_sizes [2, 4, 8, 16].
/// We test a subset here.
#[tokio::test]
#[ignore]
async fn test_multi_candidate_determinism_llama_n2() {
    run_multi_candidate_determinism("meta-llama/Llama-3.1-8B-Instruct", 2).await;
}

#[tokio::test]
#[ignore]
async fn test_multi_candidate_determinism_llama_n4() {
    run_multi_candidate_determinism("meta-llama/Llama-3.1-8B-Instruct", 4).await;
}

#[tokio::test]
#[ignore]
async fn test_multi_candidate_determinism_llama_n8() {
    run_multi_candidate_determinism("meta-llama/Llama-3.1-8B-Instruct", 8).await;
}

#[tokio::test]
#[ignore]
async fn test_multi_candidate_determinism_llama_n16() {
    run_multi_candidate_determinism("meta-llama/Llama-3.1-8B-Instruct", 16).await;
}

#[tokio::test]
#[ignore]
async fn test_multi_candidate_determinism_moondream_n2() {
    run_multi_candidate_determinism("moondream3", 2).await;
}

#[tokio::test]
#[ignore]
async fn test_multi_candidate_determinism_moondream_n4() {
    run_multi_candidate_determinism("moondream3", 4).await;
}

#[tokio::test]
#[ignore]
async fn test_multi_candidate_determinism_moondream_n8() {
    run_multi_candidate_determinism("moondream3", 8).await;
}

#[tokio::test]
#[ignore]
async fn test_multi_candidate_determinism_moondream_n16() {
    run_multi_candidate_determinism("moondream3", 16).await;
}

async fn run_multi_candidate_determinism(model_id: &str, batch_size: i32) {
    use std::collections::HashMap;

    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 64, // max_completion_tokens in Python
        temperature: 0.0,
        n: batch_size,
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "Provide one friendly sentence introducing yourself.",
    )];

    // Use streaming to collect all candidates
    let result = client.achat(model_id, messages, params, true).await;
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

            // Verify we received all candidates
            assert_eq!(
                candidate_contents.len(),
                batch_size as usize,
                "Should have received {} candidates",
                batch_size
            );

            // Reconstruct each candidate's full content
            let reconstructed: Vec<String> = (0..batch_size as u32)
                .map(|idx| {
                    candidate_contents
                        .get(&idx)
                        .map(|parts| parts.join(""))
                        .unwrap_or_default()
                })
                .collect();

            let first_content = &reconstructed[0];
            println!(
                "First content:\n{}\n seen {} candidates",
                first_content, batch_size
            );

            // With temp=0, all candidates should produce identical output
            let mut drifts = 0;
            for (idx, content) in reconstructed.iter().enumerate().skip(1) {
                if content != first_content {
                    drifts += 1;
                    println!("Drift detected at candidate {}:\n{}", idx, content);
                } else {
                    print!("✓");
                }
            }
            println!();

            assert_eq!(drifts, 0, "Expected 0 drifts, got {}", drifts);

            // Verify finish reasons
            for idx in 0..batch_size as u32 {
                let reason = finish_reasons
                    .get(&idx)
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                assert!(
                    reason == "length" || reason == "stop",
                    "Candidate {} has unexpected finish reason: {}",
                    idx,
                    reason
                );
            }
        }
        orchard::ChatResult::Complete(_) => {
            panic!("Expected stream, got complete response");
        }
    }
}

/// Test sequential request determinism - same request multiple times should yield identical results.
/// Mirrors: test_e2e_determinism.py::test_sequential_request_determinism
#[tokio::test]
#[ignore]
async fn test_sequential_request_determinism_llama() {
    run_sequential_request_determinism("meta-llama/Llama-3.1-8B-Instruct").await;
}

#[tokio::test]
#[ignore]
async fn test_sequential_request_determinism_moondream() {
    run_sequential_request_determinism("moondream3").await;
}

async fn run_sequential_request_determinism(model_id: &str) {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let num_requests = 3;
    let mut first_response: Option<String> = None;
    let mut valid_responses = 0;

    let params = SamplingParams {
        max_tokens: 64, // max_completion_tokens in Python
        temperature: 0.0,
        ..Default::default()
    };

    for i in 0..num_requests {
        let messages = vec![make_message(
            "user",
            "Provide one friendly sentence introducing yourself.",
        )];

        let result = client
            .achat(model_id, messages, params.clone(), false)
            .await;
        assert!(result.is_ok(), "Chat request failed: {:?}", result.err());

        match result.unwrap() {
            orchard::ChatResult::Complete(response) => {
                let content = response.text.clone();

                if first_response.is_none() {
                    first_response = Some(content.clone());
                    valid_responses += 1;
                } else if first_response.as_ref() == Some(&content) {
                    valid_responses += 1;
                } else {
                    println!("Drift detected!:\n{} at index {}", content, i);
                    break;
                }
            }
            orchard::ChatResult::Stream(_) => {
                panic!("Expected complete response, got stream");
            }
        }
    }

    if let Some(ref first) = first_response {
        print!("First response:\n{}\n seen {} ", first, valid_responses);
        for i in 0..num_requests {
            if i == 0 {
                print!("✓");
            } else {
                print!(".");
            }
        }
        println!();
    }

    assert_eq!(
        valid_responses, num_requests,
        "Expected {} valid responses, got {}",
        num_requests, valid_responses
    );
}
