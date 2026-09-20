//! Ten-turn conversation: each turn's prompt is the whole transcript so far plus a new
//! question. Prints, per turn, how much of the prompt came from the prefix cache.
//!
//!   MODEL=google/gemma-4-E2B-it cargo run --release --example ten_turn

mod common;

use common::{msg, run_turn, QUESTIONS};
use orchard::SamplingParams;

#[tokio::main]
async fn main() -> Result<(), common::Error> {
    let model = std::env::var("MODEL").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let (_engine, registry, client) = common::connect().await?;
    registry.ensure_loaded(&model).await?;

    let system = "You are a concise science explainer. ".repeat(10);
    let mut transcript = vec![msg("system", &system)];
    let (mut total_prompt, mut total_cached) = (0usize, 0usize);
    println!("turn | prompt tok | cached | reuse | new+recomputed | ttft ms | gen");
    for (i, question) in QUESTIONS.iter().enumerate() {
        transcript.push(msg("user", question));
        let params = SamplingParams {
            max_tokens: 140,
            temperature: 0.0,
            reasoning: Some(false),
            ..Default::default()
        };
        let turn = run_turn(&client, &model, transcript.clone(), params, |_| {}).await?;
        let (prompt, cached) = (turn.prompt_tokens, turn.cached);
        println!(
            "{:>4} | {:>10} | {:>6} | {:>4.0}% | {:>14} | {:>7.0} | {:>3}",
            i + 1,
            prompt,
            cached,
            100.0 * cached as f64 / prompt.max(1) as f64,
            prompt - cached,
            turn.ttft_ms,
            turn.ids.len()
        );
        total_prompt += prompt;
        total_cached += cached;
        transcript.push(msg("assistant", &turn.text));
    }
    println!(
        "total: {total_cached} of {total_prompt} prompt tokens served from cache ({:.0}%)",
        100.0 * total_cached as f64 / total_prompt.max(1) as f64
    );
    Ok(())
}
