//! End-to-end structured generation tests.
//!
//! Mirrors orchard-py/tests/test_e2e_structured_generation.py
//! Tests JSON schema-constrained generation.
//! Run with: cargo test -- --ignored

use std::collections::HashMap;
use std::sync::Arc;

use orchard::{Client, InferenceEngine, ModelRegistry, SamplingParams};

const MODEL_ID: &str = "moondream3";

fn make_message(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));
    msg.insert("content".to_string(), serde_json::json!(content));
    msg
}

/// Test generation with JSON schema response format.
/// Mirrors: test_e2e_structured_generation.py::test_chat_completion_structured_json_response
#[tokio::test]
#[ignore]
async fn test_json_structured_output() {
    let _engine = InferenceEngine::new().await.expect("Failed to start engine");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "color": {
                "type": "object",
                "properties": {
                    "R": {"type": "integer", "minimum": 0, "maximum": 255},
                    "G": {"type": "integer", "minimum": 0, "maximum": 255},
                    "B": {"type": "integer", "minimum": 0, "maximum": 255},
                },
                "required": ["R", "G", "B"],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["color", "confidence"],
    });

    let prompt = format!(
        "Respond with a JSON object with a rgb(r, g, b) color and a confidence score. use this schema: {}. Make it pretty, this should reflect your internal associations with the color.",
        serde_json::to_string_pretty(&schema).unwrap()
    );

    let params = SamplingParams {
        max_tokens: 100,
        temperature: 0.0,
        ..Default::default()
    };

    let messages = vec![make_message("user", &prompt)];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Structured generation failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            let text = response.text.trim();
            println!("Structured output: {}", text);

            // Find JSON in response (may be wrapped in markdown code blocks)
            let json_str = if text.starts_with("```") {
                text.lines()
                    .skip(1)
                    .take_while(|l| !l.starts_with("```"))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                // Find the first { and last }
                let start = text.find('{');
                let end = text.rfind('}');
                match (start, end) {
                    (Some(s), Some(e)) if e > s => text[s..=e].to_string(),
                    _ => text.to_string(),
                }
            };

            // Parse as JSON
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json_str);
            assert!(parsed.is_ok(), "Output should be valid JSON: {}", json_str);

            let value = parsed.unwrap();
            assert!(value.is_object(), "Output should be a JSON object");
            let obj = value.as_object().unwrap();

            // Verify structure
            assert!(obj.contains_key("color"), "Should have 'color' field");
            let color = obj.get("color").unwrap();
            assert!(color.is_object(), "'color' should be an object");
            let color_obj = color.as_object().unwrap();
            assert!(color_obj.contains_key("R"), "Should have 'R' field");
            assert!(color_obj.contains_key("G"), "Should have 'G' field");
            assert!(color_obj.contains_key("B"), "Should have 'B' field");

            // Extract and display color
            let r = color_obj.get("R").and_then(|v| v.as_i64()).unwrap_or(0) as u8;
            let g = color_obj.get("G").and_then(|v| v.as_i64()).unwrap_or(0) as u8;
            let b = color_obj.get("B").and_then(|v| v.as_i64()).unwrap_or(0) as u8;

            if let Some(confidence) = obj.get("confidence").and_then(|v| v.as_f64()) {
                println!(
                    "\x1b[38;2;{};{};{}m{}'s color is rgb({}, {}, {}) with confidence {}\x1b[0m",
                    r, g, b, MODEL_ID, r, g, b, confidence
                );
            } else {
                println!(
                    "\x1b[38;2;{};{};{}m{}'s color is rgb({}, {}, {})\x1b[0m",
                    r, g, b, MODEL_ID, r, g, b
                );
            }
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test generation constrained to a simple list format.
#[tokio::test]
#[ignore]
async fn test_list_structured_output() {
    let _engine = InferenceEngine::new().await.expect("Failed to start engine");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 50,
        temperature: 0.0,
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "List exactly 3 colors, one per line, no other text.",
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "List generation failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            let text = response.text.trim();
            println!("List output:\n{}", text);

            let lines: Vec<_> = text.lines().filter(|l| !l.is_empty()).collect();
            assert!(
                lines.len() >= 1,
                "Should have at least one line of output"
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

#[cfg(test)]
mod unit_tests {
    #[test]
    fn test_json_schema_construction() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"}
            },
            "required": ["name", "age", "active"]
        });

        assert!(schema.is_object());
        let props = schema.get("properties").unwrap();
        assert!(props.get("name").is_some());
        assert!(props.get("age").is_some());
        assert!(props.get("active").is_some());
    }
}
