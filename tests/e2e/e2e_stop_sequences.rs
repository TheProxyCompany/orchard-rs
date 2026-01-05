//! End-to-end stop sequence tests.
//!
//! Mirrors orchard-py/tests/test_e2e_stop_sequences.py
//! Tests that generation stops at specified sequences with real content assertions.
//! Run with: cargo test --test e2e -- --ignored

use orchard::SamplingParams;

use crate::fixture::{get_fixture, make_message};

const MODEL_ID: &str = "moondream3";

/// Test stop sequence on "blue" - should output red, white, blue and stop at blue
/// Mirrors: test_e2e_stop_sequences.py::test_chat_completion_respects_stop_sequence
#[tokio::test]
#[ignore]
async fn test_stop_sequence_national_colors() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 100,
        temperature: 0.0,
        stop: vec!["blue".to_string()],
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "What are the national colors of the United States of America?",
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Stop sequence test failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            let text = response.text.to_lowercase();
            println!("Output: {}", response.text);

            // Should contain red, white, and blue
            assert!(
                text.contains("red"),
                "Expected 'red' in response but got: '{}'",
                response.text
            );
            assert!(
                text.contains("white"),
                "Expected 'white' in response but got: '{}'",
                response.text
            );
            assert!(
                text.contains("blue"),
                "Expected 'blue' in response but got: '{}'",
                response.text
            );

            // Should end with "blue" (the stop sequence)
            assert!(
                text.trim().ends_with("blue"),
                "Expected response to end with 'blue' but got: '{}'",
                response.text
            );

            // Finish reason should be "stop"
            assert_eq!(
                response.finish_reason.as_deref(),
                Some("stop"),
                "Finish reason should be 'stop'"
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test stop sequence on newline.
#[tokio::test]
#[ignore]
async fn test_stop_on_newline() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 100,
        temperature: 0.0,
        stop: vec!["\n".to_string()],
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "Write a short poem with multiple lines. Start with 'Roses are red'",
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Stop sequence test failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            let text = response.text.trim();
            println!("Output: {}", text);

            // Should have stopped before second line
            let newline_count = text.matches('\n').count();
            assert!(
                newline_count == 0,
                "Should have stopped at first newline, but got {} newlines",
                newline_count
            );

            assert_eq!(
                response.finish_reason.as_deref(),
                Some("stop"),
                "Finish reason should be 'stop'"
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test stop sequence on specific word.
#[tokio::test]
#[ignore]
async fn test_stop_on_word() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 200,
        temperature: 0.0,
        stop: vec!["END".to_string()],
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "Count from 1 to 5, then write END, then count from 6 to 10.",
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Stop sequence test failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            let text = response.text;
            println!("Output: {}", text);

            // Should not contain numbers after 5
            assert!(
                !text.contains("6") && !text.contains("7") && !text.contains("8"),
                "Should have stopped before continuing count"
            );
        }
        orchard::ChatResult::Stream(_) => {
            panic!("Expected complete response, got stream");
        }
    }
}

/// Test multiple stop sequences.
#[tokio::test]
#[ignore]
async fn test_multiple_stop_sequences() {
    let fixture = get_fixture().await;
    let client = &fixture.client;

    let params = SamplingParams {
        max_tokens: 100,
        temperature: 0.0,
        stop: vec!["!".to_string(), "?".to_string()],
        ..Default::default()
    };

    let messages = vec![make_message(
        "user",
        "Ask a question or make an excited statement.",
    )];

    let result = client.achat(MODEL_ID, messages, params, false).await;
    assert!(result.is_ok(), "Multiple stop test failed: {:?}", result.err());

    match result.unwrap() {
        orchard::ChatResult::Complete(response) => {
            let text = response.text.trim();
            println!("Output: {}", text);

            assert!(
                text.ends_with('!') || text.ends_with('?'),
                "Expected response to end with stop sequence"
            );
            assert_eq!(
                response.finish_reason.as_deref(),
                Some("stop"),
                "Finish reason should be 'stop'"
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
    fn test_stop_sequences_in_params() {
        let params = SamplingParams {
            stop: vec!["end".to_string(), "\n".to_string()],
            ..Default::default()
        };

        assert_eq!(params.stop.len(), 2);
        assert!(params.stop.contains(&"end".to_string()));
        assert!(params.stop.contains(&"\n".to_string()));
    }
}
