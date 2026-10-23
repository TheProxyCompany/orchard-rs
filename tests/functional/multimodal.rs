//! End-to-end multimodal tests.
//!
//! Mirrors orchard-py/tests/functional/test_multimodal.py
//! Tests image-based inference with vision-capable models using real test images.
//! Run with: cargo test --test functional

use std::collections::HashMap;
use std::path::PathBuf;

use base64::Engine;
use orchard::SamplingParams;

use crate::fixture::{fanout, get_fixture, VISION_MODELS};

fn get_test_assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/assets")
}

fn load_image_base64(filename: &str) -> String {
    let path = get_test_assets_dir().join(filename);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("Failed to read test image {}: {}", path.display(), e));
    base64::engine::general_purpose::STANDARD.encode(&bytes)
}

#[allow(dead_code)]
fn make_text_message(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));
    msg.insert("content".to_string(), serde_json::json!(content));
    msg
}

fn make_image_message(
    role: &str,
    text: &str,
    image_base64: &str,
    image_first: bool,
) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));

    let image = serde_json::json!({
        "type": "image_url",
        "image_url": {"url": format!("data:image/jpeg;base64,{}", image_base64)}
    });
    let text = serde_json::json!({"type": "text", "text": text});
    let content = if image_first {
        serde_json::json!([image, text])
    } else {
        serde_json::json!([text, image])
    };

    msg.insert("content".to_string(), content);
    msg
}

/// Test image captioning with apple.jpg - should identify "apple"
/// Mirrors: test_multimodal.py::test_multimodal_e2e_apple_image
#[tokio::test]
async fn test_multimodal_apple_image() {
    let fixture = get_fixture().await;
    let client = &fixture.client;
    fanout(VISION_MODELS.iter().map(|&model_id| async move {
        let params = SamplingParams {
            max_tokens: 50,
            temperature: 0.0,
            reasoning: Some(false),
            ..Default::default()
        };

        let image_base64 = load_image_base64("apple.jpg");
        let messages = vec![make_image_message(
            "user",
            "What is in this image: ",
            &image_base64,
            false,
        )];

        let result = client.achat(model_id, messages, params, false).await;
        assert!(
            result.is_ok(),
            "Multimodal request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Complete(response) => {
                let output_text = response.text.to_lowercase();
                let output_lines = [format!("{} output text: {}", model_id, output_text)];
                println!("{}", output_lines.join("\n"));
                assert!(
                    output_text.contains("apple"),
                    "Expected 'apple' in response for {} but got: '{}'",
                    model_id,
                    response.text
                );
            }
            orchard::ChatResult::Stream(_) => {
                panic!("Expected complete response, got stream for {}", model_id);
            }
        }
    }))
    .await;
}

/// Test image captioning with moondream.jpg - should identify "burger"
/// Mirrors: test_multimodal.py::test_multimodal_e2e_moondream_image
#[tokio::test]
async fn test_multimodal_moondream_image() {
    let fixture = get_fixture().await;
    let client = &fixture.client;
    fanout(VISION_MODELS.iter().map(|&model_id| async move {
        let params = SamplingParams {
            max_tokens: 50,
            temperature: 0.0,
            reasoning: Some(false),
            ..Default::default()
        };

        let image_base64 = load_image_base64("moondream.jpg");
        let messages = vec![make_image_message(
            "user",
            "What is the girl doing?",
            &image_base64,
            true,
        )];

        let result = client.achat(model_id, messages, params, false).await;
        assert!(
            result.is_ok(),
            "Multimodal request failed for {}: {:?}",
            model_id,
            result.err()
        );

        match result.unwrap() {
            orchard::ChatResult::Complete(response) => {
                let output_text = response.text.to_lowercase();
                let output_lines = [format!("{} output text: {}", model_id, output_text)];
                println!("{}", output_lines.join("\n"));
                assert!(
                    output_text.contains("burger"),
                    "Expected 'burger' in response for {} but got: '{}'",
                    model_id,
                    response.text
                );
            }
            orchard::ChatResult::Stream(_) => {
                panic!("Expected complete response, got stream for {}", model_id);
            }
        }
    }))
    .await;
}
