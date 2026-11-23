//! End-to-end determinism tests.
//!
//! Mirrors orchard-py/tests/functional/test_determinism.py
//! Run with: cargo test --test functional

use crate::fixture::semantic_token_limit;
use orchard::SamplingParams;

use crate::fixture::{fanout, get_fixture, make_message, ALL_MODELS};

async fn run_multi_candidate_determinism(batch_size: i32) {
    use std::collections::HashMap;

    let fixture = get_fixture().await;
    let client = &fixture.client;

    fanout(ALL_MODELS.iter().map(|&model_id| async move {
        let params = SamplingParams {
            max_tokens: semantic_token_limit(model_id, 64),
            temperature: 0.0,
            deterministic: true,
            n: batch_size,
            reasoning: Some(false),
            ..Default::default()
        };

        let messages = vec![make_message(
            "user",
            "Provide one friendly sentence introducing yourself.",
        )];

        let result = client.achat(model_id, messages, params, true).await;
        assert!(
            result.is_ok(),
            "Chat request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Stream(mut stream) => {
                let mut candidate_contents: HashMap<u32, Vec<String>> = HashMap::new();
                let mut finish_reasons: HashMap<u32, String> = HashMap::new();

                let mut sequence_ids = HashMap::new();
                while let Some(delta) = stream.recv().await {
                    let index = delta.candidate_index.expect("missing candidate index");
                    assert!(index < batch_size as u32);
                    let id = delta.sequence_id.expect("missing sequence ID");
                    assert_eq!(*sequence_ids.entry(index).or_insert(id), id);
                    assert!(
                        !finish_reasons.contains_key(&index),
                        "delta after terminal for {model_id} candidate {index}"
                    );
                    if let Some(content) = &delta.content {
                        candidate_contents
                            .entry(index)
                            .or_default()
                            .push(content.clone());
                    }
                    if delta.is_final_delta {
                        finish_reasons.insert(
                            index,
                            delta
                                .finish_reason
                                .expect("terminal without a finish reason"),
                        );
                    }
                }

                assert_eq!(
                    candidate_contents.len(),
                    batch_size as usize,
                    "Should have received {} candidates for {}",
                    batch_size,
                    model_id
                );

                let reconstructed: Vec<String> = (0..batch_size as u32)
                    .map(|idx| {
                        candidate_contents
                            .get(&idx)
                            .map(|parts| parts.join(""))
                            .unwrap_or_default()
                    })
                    .collect();

                assert_eq!(
                    sequence_ids
                        .values()
                        .collect::<std::collections::HashSet<_>>()
                        .len(),
                    batch_size as usize,
                    "each candidate must have its own sequence ID for {model_id}"
                );
                let first_content = &reconstructed[0];
                println!(
                    "{} first content:\n{}\n seen {} candidates",
                    model_id, first_content, batch_size
                );

                let mut drifts = 0;
                for (idx, content) in reconstructed.iter().enumerate().skip(1) {
                    if content != first_content {
                        drifts += 1;
                        println!(
                            "{} drift detected at candidate {}:\n{}",
                            model_id, idx, content
                        );
                    } else {
                        print!("✓");
                    }
                }
                println!();

                assert_eq!(
                    drifts, 0,
                    "Expected 0 drifts for {}, got {}",
                    model_id, drifts
                );

                for idx in 0..batch_size as u32 {
                    let reason = finish_reasons
                        .get(&idx)
                        .map(|s| s.to_lowercase())
                        .unwrap_or_default();
                    assert!(
                        reason == "length" || reason == "stop",
                        "Candidate {} has unexpected finish reason for {}: {}",
                        idx,
                        model_id,
                        reason
                    );
                }
            }
            orchard::ChatResult::Complete(_) => {
                panic!("Expected stream, got complete response for {}", model_id);
            }
        }
    }))
    .await;
}

async fn run_sequential_request_determinism() {
    let fixture = get_fixture().await;
    let client = &fixture.client;
    let num_requests = 3;

    // Sequential-identity is a per-model property: the three repeats stay
    // serial within a model while the models themselves run concurrently.
    fanout(ALL_MODELS.iter().map(|&model_id| async move {
        let mut first_response: Option<String> = None;
        let mut valid_responses = 0;

        let params = SamplingParams {
            max_tokens: semantic_token_limit(model_id, 64),
            temperature: 0.0,
            deterministic: true,
            reasoning: Some(false),
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
            assert!(
                result.is_ok(),
                "Chat request failed for {}: {:?}",
                model_id,
                result.err()
            );

            match result.unwrap() {
                orchard::ChatResult::Complete(response) => {
                    let content = response.text.clone();
                    println!(
                        "{} repeat {} sequence {:?}, usage {:?}, token IDs {:?}",
                        model_id,
                        i,
                        response.deltas.iter().find_map(|delta| delta.sequence_id),
                        response.usage,
                        response
                            .deltas
                            .iter()
                            .flat_map(|delta| &delta.tokens)
                            .collect::<Vec<_>>()
                    );

                    if first_response.is_none() {
                        first_response = Some(content.clone());
                        valid_responses += 1;
                    } else if first_response.as_ref() == Some(&content) {
                        valid_responses += 1;
                    } else {
                        println!("{} drift detected:\n{} at index {}", model_id, content, i);
                        break;
                    }
                }
                orchard::ChatResult::Stream(_) => {
                    panic!("Expected complete response, got stream for {}", model_id);
                }
            }
        }

        if let Some(ref first) = first_response {
            print!(
                "{} first response:\n{}\n seen {} ",
                model_id, first, valid_responses
            );
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
            "Expected {} valid responses for {}, got {}",
            num_requests, model_id, valid_responses
        );
    }))
    .await;
}

/// Test multi-candidate determinism - all candidates should be identical with temp=0.
/// Mirrors: test_determinism.py::test_multi_candidate_determinism
#[tokio::test]
async fn test_multi_candidate_determinism_n2() {
    run_multi_candidate_determinism(2).await;
}

#[tokio::test]
async fn test_multi_candidate_determinism_n4() {
    run_multi_candidate_determinism(4).await;
}

#[tokio::test]
async fn test_multi_candidate_determinism_n8() {
    run_multi_candidate_determinism(8).await;
}

#[tokio::test]
async fn test_multi_candidate_determinism_n16() {
    run_multi_candidate_determinism(16).await;
}

/// Each prompt has n=1 and enters PIE as its own physical decode row.
#[tokio::test]
async fn test_sixteen_independent_greedy_rows_agree() {
    let fixture = get_fixture().await;
    fanout(ALL_MODELS.iter().map(|&model_id| async move {
        let prompt = vec![make_message(
            "user",
            "Provide one friendly sentence introducing yourself.",
        )];
        let params = SamplingParams {
            max_tokens: 8,
            temperature: 0.0,
            deterministic: true,
            n: 1,
            reasoning: Some(false),
            prefix_cache: Some(false),
            ..Default::default()
        };
        let orchard::BatchChatResult::Stream(mut stream) = fixture
            .client
            .achat_batch(model_id, vec![prompt; 16], params, true)
            .await
            .expect("independent batch failed")
        else {
            panic!("expected an independent-row stream");
        };
        let mut tokens = vec![Vec::new(); 16];
        let mut terminals = vec![None; 16];
        let mut ids = vec![None; 16];
        while let Some(delta) = stream.recv().await {
            let row = delta.prompt_index.expect("missing prompt index") as usize;
            assert_eq!(delta.candidate_index, Some(0));
            let id = delta.sequence_id.expect("missing sequence ID");
            assert_eq!(*ids[row].get_or_insert(id), id);
            assert!(
                terminals[row].is_none(),
                "delta after terminal on {model_id} row {row}"
            );
            tokens[row].extend(delta.tokens);
            if delta.is_final {
                let reason = delta
                    .finish_reason
                    .expect("terminal without a finish reason");
                assert!(
                    matches!(reason.to_lowercase().as_str(), "stop" | "length"),
                    "{model_id}: {reason}"
                );
                terminals[row] = Some(reason);
            }
        }
        assert_eq!(
            ids.into_iter()
                .flatten()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            16
        );
        assert!(terminals[0].is_some() && !tokens[0].is_empty());
        for row in 1..16 {
            assert_eq!(
                tokens[row], tokens[0],
                "token drift on {model_id} row {row}"
            );
            assert_eq!(
                terminals[row], terminals[0],
                "finish reason drift on {model_id} row {row}"
            );
        }
    }))
    .await;
}

/// Test sequential request determinism - same request multiple times should yield identical results.
/// Mirrors: test_determinism.py::test_sequential_request_determinism
#[tokio::test]
async fn test_sequential_request_determinism() {
    run_sequential_request_determinism().await;
}
