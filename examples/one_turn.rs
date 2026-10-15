//! One turn against a local model. Prints tokens as they stream, then TTFT and
//! how many prompt tokens hit the prefix cache. Run it twice to see the cache.
//!
//!   cargo run --example one_turn
//!   MODEL=google/gemma-4-26B-A4B-it cargo run --example one_turn

#[path = "../tests/project/gpu_lease.rs"]
mod gpu_lease;

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use orchard::{ChatResult, Client, InferenceEngine, ModelRegistry, SamplingParams};

fn msg(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    HashMap::from([
        ("role".to_string(), serde_json::json!(role)),
        ("content".to_string(), serde_json::json!(content)),
    ])
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    gpu_lease::hold(); // waits for any test or benchmark engine on this machine
    let model = std::env::var("MODEL").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Say hello in one sentence.".into());

    let t = Instant::now();
    let _engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(Arc::clone(&registry)).await?;
    registry.ensure_loaded(&model).await?;
    eprintln!("[engine+model ready in {:.1}s]", t.elapsed().as_secs_f64());

    let messages = vec![
        msg("system", "You are a terse assistant."),
        msg("user", &prompt),
    ];
    let params = SamplingParams {
        max_tokens: 64,
        temperature: 0.0,
        ..Default::default()
    };

    let t = Instant::now();
    let ChatResult::Stream(mut stream) = client.achat(&model, messages, params, true).await? else {
        unreachable!("asked for a stream");
    };

    let (mut ttft, mut prompt_tokens, mut cached, mut n) = (None, 0, 0, 0);
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
            print!("{c}");
            std::io::stdout().flush()?;
        }
        if d.is_final_delta {
            break;
        }
    }
    println!();
    eprintln!(
        "[ttft {:.0}ms | {n} tokens in {:.2}s | prompt {prompt_tokens} tok, {cached} from prefix cache]",
        ttft.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}
