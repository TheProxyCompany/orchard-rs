//! The same conversation as `replay_turns`, through the Responses API, which is how the
//! Proxy app talks to local models. `MODE=replay` sends each finished response back with
//! `response_input_items`; `MODE=answer` sends back the visible answer only.
//!
//!   MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 cargo run --release --example replay_responses

mod common;

use common::QUESTIONS;
use orchard::{
    response_input_items, ResponseInputItem, ResponseOutputItem, ResponsesInput, ResponsesRequest,
    ResponsesResult,
};
use serde_json::Value;

fn message(role: &str, text: &str) -> ResponseInputItem {
    ResponseInputItem::Message {
        role: role.to_string(),
        content: Value::String(text.to_string()),
        tool_calls: None,
        tool_call_id: None,
    }
}

#[tokio::main]
async fn main() -> Result<(), common::Error> {
    let model =
        std::env::var("MODEL").unwrap_or_else(|_| "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16".into());
    let replay = std::env::var("MODE")
        .map(|mode| mode != "answer")
        .unwrap_or(true);
    let turns: usize = std::env::var("TURNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let (_engine, registry, client) = common::connect().await?;
    registry.ensure_loaded(&model).await?;

    let mut items = vec![message(
        "system",
        &"You are a concise science explainer. ".repeat(10),
    )];
    let (mut total_prompt, mut total_cached) = (0u64, 0u64);
    println!(
        "mode={} model={model}",
        if replay { "replay" } else { "answer" }
    );
    println!("turn | prompt tok | cached | reuse | recomputed | gen");
    for (i, question) in QUESTIONS.iter().take(turns).enumerate() {
        items.push(message("user", question));
        let mut request = ResponsesRequest::from_text("");
        request.input = ResponsesInput::Items(items.clone());
        request.temperature = Some(0.0);
        request.max_output_tokens = Some(400);
        request.reasoning = Some(true.into());
        let ResponsesResult::Complete(response) = client.aresponses(&model, request).await? else {
            unreachable!("asked for a complete response");
        };
        let usage = response.usage.clone().unwrap_or_default();
        let (prompt, cached) = (
            usage.input_tokens as u64,
            usage
                .input_tokens_details
                .as_ref()
                .map_or(0, |d| d.cached_tokens as u64),
        );
        println!(
            "{:>4} | {:>10} | {:>6} | {:>4.0}% | {:>10} | {:>3}",
            i + 1,
            prompt,
            cached,
            100.0 * cached as f64 / prompt.max(1) as f64,
            prompt - cached,
            usage.output_tokens
        );
        total_prompt += prompt;
        total_cached += cached;
        if replay {
            items.extend(response_input_items(
                &response.output,
                response.generation.as_ref(),
            ));
        } else {
            for item in &response.output {
                if let ResponseOutputItem::Message(answer) = item {
                    let text: String = answer
                        .content
                        .iter()
                        .map(|part| part.text.as_str())
                        .collect();
                    items.push(message("assistant", &text));
                }
            }
        }
    }
    println!(
        "total: {total_cached} of {total_prompt} prompt tokens served from cache ({:.0}%)",
        100.0 * total_cached as f64 / total_prompt.max(1) as f64
    );
    Ok(())
}
