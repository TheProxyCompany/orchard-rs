//! Asynchronous prefill: keep a second model ready to pick up a conversation.
//!
//! Five turns run on MODEL_A. With MODE=warm, MODEL_B is prefilled in the background after
//! each turn (`Client::awarm_prefix`, not awaited). Turn 6 is then handed to MODEL_B.
//!
//!   MODE=cold MODEL_A=meta-llama/Llama-3.1-8B-Instruct MODEL_B=google/gemma-4-E2B-it \
//!     cargo run --release --example warm_handoff
//!   MODE=warm ... (same)

mod common;

use common::{msg, run_turn, Message, QUESTIONS};
use orchard::{Client, SamplingParams};

async fn turn(
    client: &Client,
    model: &str,
    messages: Vec<Message>,
) -> Result<(String, f64, usize, usize), common::Error> {
    let params = SamplingParams {
        max_tokens: 140,
        temperature: 0.0,
        reasoning: Some(false),
        ..Default::default()
    };
    let turn = run_turn(client, model, messages, params, |_| {}).await?;
    Ok((turn.text, turn.ttft_ms, turn.prompt_tokens, turn.cached))
}

#[tokio::main]
async fn main() -> Result<(), common::Error> {
    let model_a =
        std::env::var("MODEL_A").unwrap_or_else(|_| "meta-llama/Llama-3.1-8B-Instruct".into());
    let model_b = std::env::var("MODEL_B").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let warm = std::env::var("MODE").map(|m| m == "warm").unwrap_or(false);

    let (_engine, registry, client) = common::connect().await?;
    registry.ensure_loaded(&model_a).await?;
    registry.ensure_loaded(&model_b).await?;

    println!(
        "mode={} | A={model_a} | B={model_b}",
        if warm { "warm" } else { "cold" }
    );
    let system = "You are a concise science explainer. ".repeat(10);
    let mut transcript = vec![msg("system", &system)];
    let mut warmers = Vec::new();
    for (i, question) in QUESTIONS.iter().take(5).enumerate() {
        transcript.push(msg("user", question));
        let (text, ttft, prompt, cached) = turn(&client, &model_a, transcript.clone()).await?;
        println!(
            "  A turn {} | prompt {prompt:>4} cached {cached:>4} | ttft {ttft:>5.0} ms",
            i + 1
        );
        transcript.push(msg("assistant", &text));
        if warm {
            // Fire and forget: B catches up on the transcript while the user reads.
            let (client, model_b, snapshot) = (client.clone(), model_b.clone(), transcript.clone());
            warmers.push(tokio::spawn(async move {
                client.awarm_prefix(&[&model_b], snapshot).await
            }));
        }
    }
    let mut warmed = 0u32;
    for handle in warmers {
        for result in handle.await? {
            if let Some(error) = result.error {
                println!("  warm error: {error}");
            }
            warmed += result.prompt_tokens - result.cached_tokens.min(result.prompt_tokens);
        }
    }
    if warm {
        println!("  B prefilled {warmed} tokens in the background across 5 warm-ups");
    }

    // The closing "summarize everything above" question.
    transcript.push(msg("user", QUESTIONS[9]));
    let (_, ttft, prompt, cached) = turn(&client, &model_b, transcript).await?;
    println!(
        "HANDOFF to B | prompt {prompt} tok, {cached} cached ({:.0}%) | ttft {ttft:.0} ms",
        100.0 * cached as f64 / prompt.max(1) as f64
    );
    Ok(())
}
