//! Two turns: turn 2 = turn 1's transcript + the model's own reply + a new question.
//! Shows whether the engine reuses KV for the *generated* text, not just the prompt.
//!
//!   MODEL=google/gemma-4-26B-A4B-it cargo run --release --example two_turn

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use orchard::{ChatResult, Client, InferenceEngine, ModelRegistry, SamplingParams};

fn msg(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    HashMap::from([
        ("role".to_string(), serde_json::json!(role)),
        ("content".to_string(), serde_json::json!(content)),
    ])
}

async fn turn(
    client: &Client,
    model: &str,
    messages: Vec<HashMap<String, serde_json::Value>>,
) -> Result<String, Box<dyn std::error::Error>> {
    let params = SamplingParams {
        max_tokens: 96,
        temperature: 0.0,
        reasoning: Some(false),
        ..Default::default()
    };
    let t = Instant::now();
    let ChatResult::Stream(mut stream) = client.achat(model, messages, params, true).await? else {
        unreachable!()
    };
    let (mut ttft, mut prompt_tokens, mut cached, mut n, mut text) = (None, 0, 0, 0, String::new());
    while let Some(d) = stream.recv().await {
        if let Some(e) = d.error {
            return Err(e.into());
        }
        if ttft.is_none() && !d.tokens.is_empty() {
            ttft = Some(t.elapsed());
        }
        n += d.tokens.len();
        prompt_tokens = prompt_tokens.max(d.prompt_token_count.unwrap_or(0));
        cached = cached.max(d.cached_token_count.unwrap_or(0));
        if let Some(c) = d.content {
            text.push_str(&c);
        }
        if d.is_final_delta {
            break;
        }
    }
    println!(
        "  ttft {:>5.0}ms | {n:>3} gen tokens | prompt {prompt_tokens:>3} tok, {cached:>3} cached ({:.0}%)",
        ttft.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0),
        100.0 * cached as f64 / prompt_tokens.max(1) as f64
    );
    Ok(text)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("MODEL").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let _engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(Arc::clone(&registry)).await?;
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
