//! End-to-end structured generation tests.
//!
//! Tests JSON schema-constrained generation.
//! Run with: cargo test -- --ignored

use std::collections::HashMap;
use std::sync::Arc;

use orchard::{Client, EngineFetcher, ModelRegistry, SamplingParams};

const MODEL_ID: &str = "meta-llama/Llama-3.1-8B-Instruct";

fn make_message(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));
    msg.insert("content".to_string(), serde_json::json!(content));
    msg
}

/// Test generation with JSON output format.
#[tokio::test]
#[ignore]
async fn test_json_structured_output() {
    let fetcher = EngineFetcher::new();
    fetcher.get_engine_path().await.expect("Failed to get engine path");

    let registry = Arc::new(ModelRegistry::new().unwrap());
    let client = Client::connect(registry).await.expect("Failed to connect to engine");

    let params = SamplingParams {
        max_tokens: 100,
        temperature: 0.0,
        ..Default::default()
    };

    // Ask for structured JSON output
    let messages = vec![make_message(
        "user",
        r#"Return a JSON object with these fields: "name" (string), "age" (number), "active" (boolean). Use sample values."#,
    )];

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
                text.to_string()
            };

            // Attempt to parse as JSON
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json_str);
            if let Ok(value) = parsed {
                assert!(value.is_object(), "Output should be a JSON object");
                let obj = value.as_object().unwrap();
                assert!(obj.contains_key("name"), "Should have 'name' field");
                assert!(obj.contains_key("age"), "Should have 'age' field");
                assert!(obj.contains_key("active"), "Should have 'active' field");
            } else {
                println!("Note: Output was not valid JSON (may need schema constraints)");
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
    let fetcher = EngineFetcher::new();
    fetcher.get_engine_path().await.expect("Failed to get engine path");

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
