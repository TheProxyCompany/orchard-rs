//! Shared plumbing for the examples: connect to the engine, build a message, and
//! drain one streamed turn while tracking TTFT and prefix-cache reuse.
#![allow(dead_code)] // every example uses a different subset

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use orchard::{ChatResult, Client, InferenceEngine, ModelRegistry, ResponseDelta, SamplingParams};

pub type Message = HashMap<String, serde_json::Value>;
pub type Error = Box<dyn std::error::Error>;

pub const QUESTIONS: [&str; 10] = [
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

pub fn msg(role: &str, content: &str) -> Message {
    HashMap::from([
        ("role".to_string(), serde_json::json!(role)),
        ("content".to_string(), serde_json::json!(content)),
    ])
}

/// Start the engine and connect a client. Keep the engine alive for the whole run.
pub async fn connect() -> Result<(InferenceEngine, Arc<ModelRegistry>, Client), Error> {
    let engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(Arc::clone(&registry)).await?;
    Ok((engine, registry, client))
}

/// What one streamed turn produced.
pub struct Turn {
    pub text: String,
    /// Generated token ids.
    pub ids: Vec<i32>,
    pub prompt_tokens: usize,
    /// Prompt tokens served from the prefix cache.
    pub cached: usize,
    /// Time to the first generated token in milliseconds; 0 if none arrived.
    pub ttft_ms: f64,
    pub elapsed: Duration,
}

/// Stream one chat turn to completion. `on_delta` sees every delta before it is
/// folded into the returned `Turn`.
pub async fn run_turn(
    client: &Client,
    model: &str,
    messages: Vec<Message>,
    params: SamplingParams,
    mut on_delta: impl FnMut(&ResponseDelta),
) -> Result<Turn, Error> {
    let t = Instant::now();
    let ChatResult::Stream(mut stream) = client.achat(model, messages, params, true).await? else {
        unreachable!("asked for a stream");
    };
    let (mut text, mut ids, mut prompt_tokens, mut cached, mut ttft) =
        (String::new(), Vec::new(), 0, 0, None);
    while let Some(d) = stream.recv().await {
        if let Some(e) = &d.error {
            return Err(e.clone().into());
        }
        if ttft.is_none() && !d.tokens.is_empty() {
            ttft = Some(t.elapsed());
        }
        on_delta(&d);
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
    Ok(Turn {
        text,
        ids,
        prompt_tokens: prompt_tokens as usize,
        cached: cached as usize,
        ttft_ms: ttft.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0),
        elapsed: t.elapsed(),
    })
}
