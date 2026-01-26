//! End-to-end multimodal tests.
//!
//! Mirrors orchard-py/tests/test_e2e_multimodal.py
//! Tests image-based inference with vision-capable models using real test images.
//! Run with: cargo test --test e2e -- --ignored

use std::collections::HashMap;
use std::path::PathBuf;

use base64::Engine;
use orchard::SamplingParams;

use crate::fixture::get_fixture;

const MODEL_ID: &str = "moondream3";

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
) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));

    let content = serde_json::json!([
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{}", image_base64)}}
    ]);

    msg.insert("content".to_string(), content);
    msg
}

/// Test image captioning with apple.jpg - should identify "apple"
/// Mirrors: test_e2e_multimodal.py::test_multimodal_e2e_apple_image
#[tokio::test]
#[ignore]
async fn test_multimodal_apple_image() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 50,
        temperature: 0.0,
        ..Default::default()
    };

    let image_base64 = load_image_base64("apple.jpg");
    let messages = vec![make_image_message(
        "user",
        "What is in this image: ",
        &image_base64,
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(
        result.is_ok(),
        "Multimodal request failed: {:?}",
        result.err()
    );

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            let output_text = response.text.to_lowercase();
            let output_lines = [format!("Output text: {}", output_text)];
            println!("{}", output_lines.join("\n"));
            assert!(
                output_text.contains("apple"),
                "Expected 'apple' in response but got: '{}'",
                response.text
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test image captioning with moondream.jpg - should identify "burger"
/// Mirrors: test_e2e_multimodal.py::test_multimodal_e2e_moondream_image
#[tokio::test]
#[ignore]
async fn test_multimodal_moondream_image() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 50,
        temperature: 0.0,
        ..Default::default()
    };

    let image_base64 = load_image_base64("moondream.jpg");
    let messages = vec![make_image_message(
        "user",
        "What is the girl doing?",
        &image_base64,
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(
        result.is_ok(),
        "Multimodal request failed: {:?}",
        result.err()
    );

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            let output_text = response.text.to_lowercase();
            let output_lines = [format!("Output text: {}", output_text)];
            println!("{}", output_lines.join("\n"));
            assert!(
                output_text.contains("burger"),
                "Expected 'burger' in response but got: '{}'",
                response.text
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_assets_exist() {
        let assets_dir = get_test_assets_dir();
        assert!(
            assets_dir.join("apple.jpg").exists(),
            "apple.jpg should exist"
        );
        assert!(
            assets_dir.join("moondream.jpg").exists(),
            "moondream.jpg should exist"
        );
    }

    #[test]
    fn test_load_image_base64() {
        let base64 = load_image_base64("apple.jpg");
        assert!(!base64.is_empty());

        // Should be valid base64
        let decoded = base64::engine::general_purpose::STANDARD.decode(&base64);
        assert!(decoded.is_ok());

        // Should start with JPEG magic bytes
        let bytes = decoded.unwrap();
        assert!(bytes.len() >= 2);
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1], 0xD8);
    }

    #[test]
    fn test_make_image_message() {
        let msg = make_image_message("user", "Describe this.", "abc123");

        assert_eq!(msg.get("role").unwrap().as_str(), Some("user"));

        let content = msg.get("content").unwrap().as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0].get("type").unwrap().as_str(), Some("text"));
        assert_eq!(content[1].get("type").unwrap().as_str(), Some("image_url"));
    }
}
