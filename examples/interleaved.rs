//! Two conversations alternating turns on one model (A1, B1, A2, B2, ...). Each turn's
//! prompt is its own transcript plus a new question. Shows whether one conversation's
//! cached prefix survives the other's allocations.
//!
//!   MODEL=google/gemma-4-E2B-it cargo run --release --example interleaved

mod common;

use common::{msg, run_turn, QUESTIONS};
use orchard::SamplingParams;

#[tokio::main]
async fn main() -> Result<(), common::Error> {
    let model = std::env::var("MODEL").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let (_engine, registry, client) = common::connect().await?;
    registry.ensure_loaded(&model).await?;

    let systems = [
        "You are a concise science explainer. ".repeat(10),
        "You are a patient history tutor. ".repeat(10),
    ];
    let mut transcripts = [
        vec![msg("system", &systems[0])],
        vec![msg("system", &systems[1])],
    ];
    let (mut total_prompt, mut total_cached) = (0usize, 0usize);
    println!("conv turn | prompt tok | cached | reuse | new+recomputed | ttft ms | gen");
    for step in 0..12 {
        let (conv, i) = (step % 2, step / 2);
        let question = QUESTIONS[i];
        let transcript = &mut transcripts[conv];
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
            "{:>4} {:>4} | {:>10} | {:>6} | {:>4.0}% | {:>14} | {:>7.0} | {:>3}",
            if conv == 0 { "A" } else { "B" },
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
