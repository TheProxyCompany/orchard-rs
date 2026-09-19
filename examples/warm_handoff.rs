//! Asynchronous prefill: keep a second model ready to pick up a conversation.
//!
//! Five turns run on MODEL_A. With MODE=warm, MODEL_B is prefilled in the background after
//! each turn (`Client::awarm_prefix`, not awaited). Turn 6 is then handed to MODEL_B.
//!
//!   MODE=cold MODEL_A=meta-llama/Llama-3.1-8B-Instruct MODEL_B=google/gemma-4-E2B-it \
//!     cargo run --release --example warm_handoff
//!   MODE=warm ... (same)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use orchard::{ChatResult, Client, InferenceEngine, ModelRegistry, SamplingParams};

type Message = HashMap<String, serde_json::Value>;

fn msg(role: &str, content: &str) -> Message {
    HashMap::from([
        ("role".to_string(), serde_json::json!(role)),
        ("content".to_string(), serde_json::json!(content)),
    ])
}

const QUESTIONS: [&str; 6] = [
    "In about 80 words, explain why the sky is blue.",
    "In about 80 words, why are sunsets red?",
    "In about 80 words, why does the ocean look blue?",
    "In about 80 words, why is the grass green?",
    "In about 80 words, why is snow white?",
    "In about 80 words, summarize everything above in one paragraph.",
];

async fn turn(
    client: &Client,
    model: &str,
    messages: Vec<Message>,
) -> Result<(String, f64, u32, u32), Box<dyn std::error::Error>> {
    let params = SamplingParams {
        max_tokens: 140,
        temperature: 0.0,
        reasoning: Some(false),
        ..Default::default()
    };
    let t = Instant::now();
    let ChatResult::Stream(mut stream) = client.achat(model, messages, params, true).await? else {
        unreachable!()
    };
    let (mut ttft, mut prompt, mut cached, mut text) = (None, 0, 0, String::new());
    while let Some(d) = stream.recv().await {
        if let Some(e) = d.error {
            return Err(e.into());
        }
        if ttft.is_none() && !d.tokens.is_empty() {
            ttft = Some(t.elapsed().as_secs_f64() * 1000.0);
        }
        prompt = prompt.max(d.prompt_token_count.unwrap_or(0));
        cached = cached.max(d.cached_token_count.unwrap_or(0));
        if let Some(c) = d.content {
            text.push_str(&c);
        }
        if d.is_final_delta {
            break;
        }
    }
    Ok((text, ttft.unwrap_or(0.0), prompt, cached))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_a =
        std::env::var("MODEL_A").unwrap_or_else(|_| "meta-llama/Llama-3.1-8B-Instruct".into());
    let model_b = std::env::var("MODEL_B").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let warm = std::env::var("MODE").map(|m| m == "warm").unwrap_or(false);

    let _engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(Arc::clone(&registry)).await?;
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

    transcript.push(msg("user", QUESTIONS[5]));
    let (_, ttft, prompt, cached) = turn(&client, &model_b, transcript).await?;
    println!(
        "HANDOFF to B | prompt {prompt} tok, {cached} cached ({:.0}%) | ttft {ttft:.0} ms",
        100.0 * cached as f64 / prompt.max(1) as f64
    );
    Ok(())
}
