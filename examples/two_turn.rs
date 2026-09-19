//! Two turns: turn 2 = turn 1's transcript + the model's own reply + a new question.
//! Shows whether the engine reuses KV for the *generated* text, not just the prompt.
//!
//!   MODEL=google/gemma-4-26B-A4B-it cargo run --release --example two_turn

mod common;

use std::time::Instant;

use common::{msg, run_turn, Message};
use orchard::{Client, SamplingParams};

async fn turn(
    client: &Client,
    model: &str,
    messages: Vec<Message>,
) -> Result<String, common::Error> {
    let params = SamplingParams {
        max_tokens: 96,
        temperature: 0.0,
        reasoning: Some(false),
        ..Default::default()
    };
    let turn = run_turn(client, model, messages, params, |_| {}).await?;
    let (n, prompt_tokens, cached) = (turn.ids.len(), turn.prompt_tokens, turn.cached);
    println!(
        "  ttft {:>5.0}ms | {n:>3} gen tokens | prompt {prompt_tokens:>3} tok, {cached:>3} cached ({:.0}%)",
        turn.ttft_ms,
        100.0 * cached as f64 / prompt_tokens.max(1) as f64
    );
    Ok(turn.text)
}

#[tokio::main]
async fn main() -> Result<(), common::Error> {
    let model = std::env::var("MODEL").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let (_engine, registry, client) = common::connect().await?;
    let t = Instant::now();
    registry.ensure_loaded(&model).await?;
    eprintln!("[model ready in {:.1}s]", t.elapsed().as_secs_f64());

    let system = "You are a terse assistant. ".repeat(12); // ~100 tokens of shared prefix
    let mut transcript = vec![
        msg("system", &system),
        msg("user", "Name three planets, one per line."),
    ];

    println!("turn 1 (cold or warm from earlier runs):");
    let a1 = turn(&client, &model, transcript.clone()).await?;
    transcript.push(msg("assistant", &a1));
    transcript.push(msg("user", "Now name three more."));

    println!("turn 2 (prefix = system + user1 + model's own reply):");
    let _a2 = turn(&client, &model, transcript.clone()).await?;

    println!("turn 2 again (identical prompt):");
    let _ = turn(&client, &model, transcript).await?;
    Ok(())
}
