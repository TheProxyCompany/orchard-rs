//! End-to-end structured generation tests.
//!
//! Mirrors orchard-py/tests/test_e2e_structured_generation.py
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message};

const MODEL_ID: &str = "moondream3";

/// Test generation with JSON schema response format.
/// Mirrors: test_e2e_structured_generation.py::test_chat_completion_structured_json_response
#[tokio::test]
#[ignore]
async fn test_chat_completion_structured_json_response() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

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

    let temperature = 0.0; // parametrized in Python as [0.0]

    let prompt = format!(
        "Respond with a JSON object with a rgb(r, g, b) color and a confidence score. use this schema: {}. Make it pretty, this should reflect your internal associations with the color.",
        serde_json::to_string_pretty(&schema).unwrap()
    );

    let params = SamplingParams {
        temperature,
        top_logprobs: 3, // logprobs=True, top_logprobs=3 in Python
        response_format: Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "color_summary",
                "strict": true,
                "schema": schema,
            },
        })),
        ..Default::default()
    };

    let messages = vec![make_message("user", &prompt)];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(
        result.is_ok(),
        "Structured generation failed: {:?}",
        result.err()
    );

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            let content = response.text.trim();
            assert!(!content.is_empty(), "Expected structured content in completion.");

            // Find JSON in response
            let start = content.find('{');
            let end = content.rfind('}');
            let json_str = match (start, end) {
                (Some(s), Some(e)) if e > s => &content[s..=e],
                _ => panic!("Invalid JSON in completion: {}", content),
            };

            let parsed: serde_json::Value = serde_json::from_str(json_str)
                .unwrap_or_else(|_| panic!("Failed to parse JSON: {}", json_str));

            assert!(parsed.is_object(), "Output should be a JSON object");
            let obj = parsed.as_object().unwrap();

            // Verify color structure
            assert!(obj.contains_key("color"), "Should have 'color' field");
            let color = obj.get("color").unwrap();
            assert!(color.is_object(), "'color' should be an object");
            let color_obj = color.as_object().unwrap();
            assert!(color_obj.contains_key("R"), "Should have 'R' field");
            assert!(
                color_obj.get("R").unwrap().is_number(),
                "'R' should be a number"
            );
            assert!(color_obj.contains_key("G"), "Should have 'G' field");
            assert!(
                color_obj.get("G").unwrap().is_number(),
                "'G' should be a number"
            );
            assert!(color_obj.contains_key("B"), "Should have 'B' field");
            assert!(
                color_obj.get("B").unwrap().is_number(),
                "'B' should be a number"
            );

            // Verify confidence if present
            if let Some(confidence) = obj.get("confidence") {
                assert!(
                    confidence.is_number(),
                    "'confidence' should be a number"
                );
            }

            // Print colored output like Python
            let r = color_obj.get("R").and_then(|v| v.as_i64()).unwrap_or(0) as u8;
            let g = color_obj.get("G").and_then(|v| v.as_i64()).unwrap_or(0) as u8;
            let b = color_obj.get("B").and_then(|v| v.as_i64()).unwrap_or(0) as u8;
            if let Some(confidence) = obj.get("confidence").and_then(|v| v.as_f64()) {
                let opacity = (confidence * 255.0) as u8;
                println!(
                    "\x1b[38;2;{};{};{};{}m{}'s color is rgb({}, {}, {}) with confidence {} at temperature {}.\x1b[0m",
                    r, g, b, opacity, MODEL_ID, r, g, b, confidence, temperature
                );
            } else {
                println!(
                    "\x1b[38;2;{};{};{}m{}'s color is rgb({}, {}, {}) at temperature {}. No confidence score provided.\x1b[0m",
                    r, g, b, MODEL_ID, r, g, b, temperature
                );
            }
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}
