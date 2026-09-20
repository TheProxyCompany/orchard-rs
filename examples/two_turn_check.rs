//! Correctness check for prefix-cache reuse: generate turn 2 with the cache, then the
//! identical turn 2 with the prefix cache disabled, and compare generated token ids.
//!
//!   MODEL=allenai/Olmo-Hybrid-Instruct-DPO-7B cargo run --release --example two_turn_check
//!
//! Exits non-zero unless the RESULT is identical.

use std::collections::HashMap;
use std::sync::Arc;

use orchard::{ChatResult, Client, InferenceEngine, ModelRegistry, SamplingParams};

type Message = HashMap<String, serde_json::Value>;

fn msg(role: &str, content: &str) -> Message {
    HashMap::from([
        ("role".to_string(), serde_json::json!(role)),
        ("content".to_string(), serde_json::json!(content)),
    ])
}

async fn turn(
    client: &Client,
    model: &str,
    messages: Vec<Message>,
    prefix_cache: bool,
) -> Result<(String, Vec<i32>, usize, usize, Vec<f64>, String), Box<dyn std::error::Error>> {
    let params = SamplingParams {
        max_tokens: 160,
        temperature: 0.0,
        reasoning: Some(false),
        prefix_cache: Some(prefix_cache),
        top_logprobs: 5,
        // DETERMINISTIC=1: the request must be reproducible, so the engine may only
        // reuse prefill-produced KV and warm must equal cold exactly.
        deterministic: std::env::var("DETERMINISTIC").is_ok(),
        ..Default::default()
    };
    let ChatResult::Stream(mut stream) = client.achat(model, messages, params, true).await? else {
        unreachable!()
    };
    let (mut ids, mut text, mut prompt_tokens, mut cached) = (Vec::new(), String::new(), 0, 0);
    // Per-token log-prob of the chosen token, from the running cumulative value.
    let (mut per_token, mut last_cum, mut first_top) = (Vec::<f64>::new(), 0.0f64, String::new());
    while let Some(d) = stream.recv().await {
        if let Some(e) = d.error {
            return Err(e.into());
        }
        if first_top.is_empty() && !d.top_logprobs.is_empty() {
            first_top = d
                .top_logprobs
                .iter()
                .map(|t| format!("{:?}:{:.4}", t.token, t.logprob))
                .collect::<Vec<_>>()
                .join("  ");
        }
        if let Some(cum) = d.cumulative_logprob {
            if !d.tokens.is_empty() {
                let each = (cum - last_cum) / d.tokens.len() as f64;
                per_token.extend(std::iter::repeat_n(each, d.tokens.len()));
                last_cum = cum;
            }
        }
        ids.extend(d.tokens.iter().copied());
        prompt_tokens = prompt_tokens.max(d.prompt_token_count.unwrap_or(0));
        cached = cached.max(d.cached_token_count.unwrap_or(0));
        if let Some(c) = d.content {
            text.push_str(&c);
        }
        if d.is_final_delta {
            break;
        }
    }
    Ok((
        text,
        ids,
        prompt_tokens as usize,
        cached as usize,
        per_token,
        first_top,
    ))
}

/// The exit status follows the RESULT line.
fn verdict(differing: usize) -> Result<(), Box<dyn std::error::Error>> {
    if differing > 0 {
        return Err("the warm turn differs from the cache-off turn".into());
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("MODEL").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let _engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(Arc::clone(&registry)).await?;
    registry.ensure_loaded(&model).await?;

    let system = "You are a terse assistant. ".repeat(12);
    let mut transcript = vec![
        msg("system", &system),
        msg(
            "user",
            "Explain in about 150 words why the sky is blue and sunsets are red.",
        ),
    ];
    let (a1, _, _, _, _, _) = turn(&client, &model, transcript.clone(), true).await?;
    transcript.push(msg("assistant", &a1));
    transcript.push(msg("user", "Now do the same for why the ocean looks blue."));

    let (warm_text, warm, prompt, warm_cached, warm_lp, warm_top) =
        turn(&client, &model, transcript.clone(), true).await?;
    let (cold_text, cold, _, cold_cached, cold_lp, cold_top) =
        turn(&client, &model, transcript, false).await?;
    // Log-prob agreement over the shared prefix of generated tokens: kernel noise is
    // ~1e-3; a recurrent state that is one token off is orders of magnitude larger.
    let shared = warm
        .iter()
        .zip(cold.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let n = shared.min(warm_lp.len()).min(cold_lp.len());
    if n > 0 {
        let diffs: Vec<f64> = (0..n).map(|i| (warm_lp[i] - cold_lp[i]).abs()).collect();
        let max = diffs.iter().cloned().fold(0.0, f64::max);
        let mean = diffs.iter().sum::<f64>() / n as f64;
        println!("logprob |warm-cold| over first {n} shared tokens: max {max:.5} mean {mean:.5}  first-token {:.5}", diffs[0]);
    }
    println!("first step top-5 warm: {warm_top}");
    println!("first step top-5 cold: {cold_top}");
    println!("prompt {prompt} tok | warm cached {warm_cached} | cold cached {cold_cached}");
    println!(
        "warm {} gen tokens | cold {} gen tokens",
        warm.len(),
        cold.len()
    );
    match warm.iter().zip(cold.iter()).position(|(a, b)| a != b) {
        None if warm.len() == cold.len() => println!("RESULT: identical"),
        None => println!(
            "RESULT: identical for {} tokens, lengths differ",
            warm.len().min(cold.len())
        ),
        Some(i) => {
            println!(
                "RESULT: first divergence at generated token {i} of {}",
                cold.len()
            );
            println!(
                "  warm: {}",
                warm_text
                    .chars()
                    .take(160)
                    .collect::<String>()
                    .replace('\n', " ")
            );
            println!(
                "  cold: {}",
                cold_text
                    .chars()
                    .take(160)
                    .collect::<String>()
                    .replace('\n', " ")
            );
        }
    }
    verdict(usize::from(warm != cold))
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_differing_run_fails_the_process() {
        assert!(super::verdict(0).is_ok());
        assert!(super::verdict(1).is_err());
    }
}
