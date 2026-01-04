//! End-to-end multimodal tests.
//!
//! Tests image-based inference with vision-capable models.
//! Set PIE_LOCAL_BUILD to run integration tests.

use std::collections::HashMap;
use std::sync::Arc;

use base64::Engine;
use orchard::{Client, ModelRegistry, SamplingParams};

const MODEL_ID: &str = "moondream3";

/// Check if PIE is available for testing.
fn pie_available() -> bool {
    std::env::var("PIE_LOCAL_BUILD").is_ok()
}

/// Skip test if PIE is not available.
macro_rules! require_pie {
    () => {
        if !pie_available() {
            eprintln!("SKIPPED: PIE_LOCAL_BUILD not set. Set it to run integration tests.");
            return;
        }
    };
}

#[allow(dead_code)]
fn make_text_message(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));
    msg.insert("content".to_string(), serde_json::json!(content));
    msg
}

fn make_image_message(role: &str, text: &str, image_base64: &str) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));

    let content = serde_json::json!([
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{}", image_base64)}}
    ]);

    msg.insert("content".to_string(), content);
    msg
}

/// Create a simple 2x2 red test image in JPEG format as base64.
fn create_test_image_base64() -> String {
    // This is a minimal valid JPEG: 2x2 pixels, all red
    // In a real test, you'd load a proper test image
    let minimal_jpeg: &[u8] = &[
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x02,
        0x00, 0x02, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
        0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
        0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD3, 0xED, 0x6B, 0x8A, 0x28, 0xAF, 0xFF,
        0xD9,
    ];

    base64::engine::general_purpose::STANDARD.encode(minimal_jpeg)
}

/// Test image captioning with moondream3.
#[tokio::test]
async fn test_image_captioning() {
    require_pie!();

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 50,
        temperature: 0.0,
        ..Default::default()
    };

    let image_base64 = create_test_image_base64();
    let messages = vec![make_image_message(
        "user",
        "Describe this image briefly.",
        &image_base64,
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Multimodal request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            assert!(!response.text.is_empty(), "Response should have content");
            println!("Caption: {}", response.text);
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test visual question answering.
#[tokio::test]
async fn test_visual_question_answering() {
    require_pie!();

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 30,
        temperature: 0.0,
        ..Default::default()
    };

    let image_base64 = create_test_image_base64();
    let messages = vec![make_image_message(
        "user",
        "What is the dominant color in this image?",
        &image_base64,
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "VQA request failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            assert!(!response.text.is_empty(), "Response should have content");
            println!("Answer: {}", response.text);
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
    fn test_create_test_image_base64() {
        let base64 = create_test_image_base64();
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
