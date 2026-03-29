//! End-to-end unicode payload tests.
//!
//! Mirrors orchard-py/tests/test_e2e_unicode_payload.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message, TEXT_MODELS};

/// Test that unicode (emoji) payloads round-trip correctly without corruption.
/// Mirrors: test_e2e_unicode_payload.py::test_unicode_payload_round_trip
#[tokio::test]
#[ignore]
async fn test_unicode_payload_round_trip() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let target = "😊".repeat(40); // 40 multi-byte characters (> MAX_INLINE_CONTENT_BYTES)
    let replacement_char = '\u{FFFD}';

    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: 10, // max_completion_tokens in Python
        ..Default::default()
    };

    let prompt = format!("Respond with this emoji: {}", target);
    let messages = vec![make_message("user", &prompt)];

    for &model_id in TEXT_MODELS {
        let result = client
            .achat(model_id, messages.clone(), params.clone(), true)
            .await;
        assert!(
            result.is_ok(),
            "Chat request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Stream(mut stream) => {
                let mut chunks = Vec::new();
                let mut finish_reason: Option<String> = None;

                while let Some(delta) = stream.recv().await {
                    if let Some(content) = &delta.content {
                        assert!(
                            !content.contains(replacement_char),
                            "Encountered replacement char in streamed chunk for {}",
                            model_id
                        );
                        chunks.push(content.clone());
                    }
                    if let Some(reason) = &delta.finish_reason {
                        finish_reason = Some(reason.clone());
                    }
                }

                assert!(!chunks.is_empty(), "No streamed content chunks received");
                assert!(
                    finish_reason.is_some(),
                    "Expected finish reason in streamed response for {}",
                    model_id
                );

                let full_content: String = chunks.join("");
                assert!(
                    !full_content.is_empty(),
                    "Expected non-empty streamed content for {}",
                    model_id
                );
                assert!(
                    !full_content.contains(replacement_char),
                    "Replacement char found in final content for {}",
                    model_id
                );

                let total_emojis = full_content.matches("😊").count();
                assert!(
                    total_emojis > 0,
                    "Expected at least one emoji in the content for {}",
                    model_id
                );
            }
            orchard::ChatResult::Complete(_) => {
                panic!("Expected stream, got complete response for {}", model_id);
            }
        }
    }
}
