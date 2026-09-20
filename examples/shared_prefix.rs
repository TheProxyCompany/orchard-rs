//! Two different requests that share a long system prompt, on a sliding-window model.
//! The second request branches off well before the first one ends, so everything it
//! reuses must be valid at ITS window, not just at the first request's. Compares the
//! second request with the prefix cache on against the same request with it off.
//!
//!   MODEL=google/gemma-4-E2B-it cargo run --release --example shared_prefix
//!   REPRODUCIBLE=0 ...   ordinary greedy requests instead of reproducible ones
//!
//! Exits non-zero when either request differs from its cache-off run.

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

struct Run {
    tokens: Vec<i32>,
    tops: Vec<String>,
    prompt: usize,
    cached: usize,
}

async fn run(
    client: &Client,
    model: &str,
    messages: Vec<Message>,
    prefix_cache: bool,
    reproducible: bool,
) -> Result<Run, Box<dyn std::error::Error>> {
    let params = SamplingParams {
        max_tokens: 32,
        temperature: 0.0,
        reasoning: Some(false),
        prefix_cache: Some(prefix_cache),
        deterministic: reproducible,
        top_logprobs: 3,
        ..Default::default()
    };
    let ChatResult::Stream(mut stream) = client.achat(model, messages, params, true).await? else {
        unreachable!()
    };
    let mut out = Run {
        tokens: Vec::new(),
        tops: Vec::new(),
        prompt: 0,
        cached: 0,
    };
    while let Some(d) = stream.recv().await {
        if let Some(e) = d.error {
            return Err(e.into());
        }
        if !d.top_logprobs.is_empty() {
            out.tops.push(
                d.top_logprobs
                    .iter()
                    .map(|t| format!("{:?} {:.4}", t.token, t.logprob))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
        out.tokens.extend(d.tokens.iter().copied());
        out.prompt = out.prompt.max(d.prompt_token_count.unwrap_or(0) as usize);
        out.cached = out.cached.max(d.cached_token_count.unwrap_or(0) as usize);
        if d.is_final_delta {
            break;
        }
    }
    Ok(out)
}

fn numbered(count: usize, what: &str, offset: usize) -> String {
    (0..count)
        .map(|i| {
            format!(
                "Entry {}: the {what} ledger lists item {} at shelf {}. ",
                i + offset,
                i * 7 + 3,
                i * 3 + 11
            )
        })
        .collect()
}

/// The exit status follows the RESULT line.
fn verdict(bad: usize) -> Result<(), Box<dyn std::error::Error>> {
    if bad > 0 {
        return Err(format!("{bad} shared-prefix requests differ from the cache-off run").into());
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("MODEL").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let reproducible = std::env::var("REPRODUCIBLE")
        .map(|v| v != "0")
        .unwrap_or(true);
    let env_usize = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let _engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(Arc::clone(&registry)).await?;
    registry.ensure_loaded(&model).await?;

    println!("shared system tok (approx) | first request | second: prompt, cached | tokens match | log-probs match");
    let mut bad = 0;
    for shared in [
        env_usize("SHARED_ENTRIES", 12),
        env_usize("SHARED_ENTRIES_2", 40),
    ] {
        let system = format!(
            "You are the archive clerk. Known records follow. {}",
            numbered(shared, "harbour", 0)
        );
        let first = vec![
            msg("system", &system),
            msg(
                "user",
                &format!(
                    "{}\nWhich shelf holds item 3 of the harbour ledger?",
                    numbered(env_usize("TAIL_ENTRIES", 34), "mill", 500)
                ),
            ),
        ];
        let second = vec![
            msg("system", &system),
            msg(
                "user",
                &format!(
                    "{}\nWhich shelf holds item 10 of the harbour ledger?",
                    numbered(env_usize("TAIL_ENTRIES", 34), "quarry", 900)
                ),
            ),
        ];
        let donor = run(&client, &model, first, true, reproducible).await?;
        let warm = run(&client, &model, second.clone(), true, reproducible).await?;
        let cold = run(&client, &model, second, false, reproducible).await?;
        let tokens_match = warm.tokens == cold.tokens;
        let logprobs_match = warm.tops == cold.tops;
        bad += usize::from(!(tokens_match && logprobs_match));
        println!(
            "{:>26} | {:>13} | {:>6}, {:>6} | {:>12} | {}",
            shared * 22,
            donor.prompt,
            warm.prompt,
            warm.cached,
            if tokens_match { "yes" } else { "NO" },
            if logprobs_match { "yes" } else { "NO" }
        );
        if !logprobs_match {
            for step in 0..warm.tops.len().min(cold.tops.len()).min(2) {
                println!("    step {step} cache on : {}", warm.tops[step]);
                println!("    step {step} cache off: {}", cold.tops[step]);
            }
        }
    }
    println!("RESULT: {bad} of 2 shared-prefix requests differ from the cache-off run (reproducible={reproducible})");
    verdict(bad)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_differing_run_fails_the_process() {
        assert!(super::verdict(0).is_ok());
        assert!(super::verdict(1).is_err());
    }
}
