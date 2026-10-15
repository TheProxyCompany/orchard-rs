//! Two conversations alternating turns on one model (A1, B1, A2, B2, ...). Each turn's
//! prompt is its own transcript plus a new question. Shows whether one conversation's
//! cached prefix survives the other's allocations.
//!
//!   MODEL=google/gemma-4-E2B-it cargo run --release --example interleaved

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

const QUESTIONS: [&str; 10] = [
    "In about 80 words, explain why the sky is blue.",
    "In about 80 words, why are sunsets red?",
    "In about 80 words, why does the ocean look blue?",
    "In about 80 words, why is the grass green?",
    "In about 80 words, why is snow white?",
    "In about 80 words, why do rainbows form?",
    "In about 80 words, why is the moon sometimes orange?",
    "In about 80 words, why do stars twinkle?",
    "In about 80 words, why is fire yellow and sometimes blue?",
    "In about 80 words, summarize everything above in one paragraph.",
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("MODEL").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let _engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(Arc::clone(&registry)).await?;
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
        let t = Instant::now();
        let ChatResult::Stream(mut stream) = client
            .achat(&model, transcript.clone(), params, true)
            .await?
        else {
            unreachable!()
        };
        let (mut ttft, mut prompt, mut cached, mut n, mut text) =
            (None, 0usize, 0usize, 0usize, String::new());
        while let Some(d) = stream.recv().await {
            if let Some(e) = d.error {
                return Err(e.into());
            }
            if ttft.is_none() && !d.tokens.is_empty() {
                ttft = Some(t.elapsed());
            }
            n += d.tokens.len();
            prompt = prompt.max(d.prompt_token_count.unwrap_or(0) as usize);
            cached = cached.max(d.cached_token_count.unwrap_or(0) as usize);
            if let Some(c) = d.content {
                text.push_str(&c);
            }
            if d.is_final_delta {
                break;
            }
        }
        println!(
            "{:>4} {:>4} | {:>10} | {:>6} | {:>4.0}% | {:>14} | {:>7.0} | {:>3}",
            if conv == 0 { "A" } else { "B" },
            i + 1,
            prompt,
            cached,
            100.0 * cached as f64 / prompt.max(1) as f64,
            prompt - cached,
            ttft.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0),
            n
        );
        total_prompt += prompt;
        total_cached += cached;
        transcript.push(msg("assistant", &text));
    }
    println!(
        "total: {total_cached} of {total_prompt} prompt tokens served from cache ({:.0}%)",
        100.0 * total_cached as f64 / total_prompt.max(1) as f64
    );
    Ok(())
}
