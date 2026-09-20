//! A conversation with a reasoning model, carried forward three ways.
//!
//! `MODE=replay` appends `Client::assistant_message` after every turn: the reply comes
//! back with its reasoning, as the token ids the model produced. `MODE=reasoning` sends
//! the same message without the ids, so the reasoning comes back as text. `MODE=answer`
//! sends the visible answer only, which is what callers did before. The table shows how
//! much of each prompt the engine served from its prefix cache.
//!
//!   MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 MODE=replay cargo run --release --example replay_turns

mod common;

use common::{msg, run_turn, QUESTIONS};
use orchard::SamplingParams;

#[tokio::main]
async fn main() -> Result<(), common::Error> {
    let model =
        std::env::var("MODEL").unwrap_or_else(|_| "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16".into());
    let mode = std::env::var("MODE").unwrap_or_else(|_| "replay".into());
    let reasoning = std::env::var("REASONING").map(|v| v != "0").unwrap_or(true);
    let turns: usize = std::env::var("TURNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let (_engine, registry, client) = common::connect().await?;
    registry.ensure_loaded(&model).await?;

    let mut transcript = vec![msg(
        "system",
        &"You are a concise science explainer. ".repeat(10),
    )];
    let (mut total_prompt, mut total_cached, mut last) = (0usize, 0usize, None::<(usize, usize)>);
    println!("mode={mode} reasoning={reasoning} model={model}");
    println!("turn | prompt tok | cached | reuse | recomputed | beyond last prompt+reply | ttft ms | gen");
    for (i, question) in QUESTIONS.iter().take(turns).enumerate() {
        transcript.push(msg("user", question));
        let params = SamplingParams {
            max_tokens: 400,
            temperature: 0.0,
            reasoning: Some(reasoning),
            ..Default::default()
        };
        let turn = run_turn(&client, &model, transcript.clone(), params.clone(), |_| {}).await?;
        let (prompt, cached) = (turn.prompt_tokens, turn.cached);
        // With replay this is the new question and its markers, nothing else.
        let beyond = last.map(|(p, g)| (prompt as i64 - (p + g) as i64).to_string());
        println!(
            "{:>4} | {:>10} | {:>6} | {:>4.0}% | {:>10} | {:>24} | {:>7.0} | {:>3}",
            i + 1,
            prompt,
            cached,
            100.0 * cached as f64 / prompt.max(1) as f64,
            prompt - cached,
            beyond.unwrap_or_default(),
            turn.ttft_ms,
            turn.ids.len()
        );
        total_prompt += prompt;
        total_cached += cached;
        last = Some((prompt, turn.ids.len()));

        let mut reply = client
            .assistant_message(&model, &params, turn.deltas)
            .await?;
        if mode != "replay" {
            reply.remove("generation");
        }
        if mode == "answer" {
            reply.remove("reasoning_content");
        }
        transcript.push(reply);
    }
    println!(
        "total: {total_cached} of {total_prompt} prompt tokens served from cache ({:.0}%)",
        100.0 * total_cached as f64 / total_prompt.max(1) as f64
    );
    Ok(())
}
