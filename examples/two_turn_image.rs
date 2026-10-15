//! Two turns with images: turn 1 asks about one image; turn 2 is turn 1's transcript
//! plus the model's own reply plus a SECOND image. Shows how much of an image-bearing
//! conversation the engine reuses (image pages, the reply, and the text around them).
//!
//!   MODEL=google/gemma-4-E2B-it cargo run --release --example two_turn_image

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use orchard::{ChatResult, Client, InferenceEngine, ModelRegistry, SamplingParams};

type Message = HashMap<String, serde_json::Value>;

fn text_msg(role: &str, content: &str) -> Message {
    HashMap::from([
        ("role".to_string(), serde_json::json!(role)),
        ("content".to_string(), serde_json::json!(content)),
    ])
}

fn image_msg(text: &str, asset: &str) -> Message {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/assets")
        .join(asset);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let image = serde_json::json!({
        "type": "image_url",
        "image_url": {"url": format!("data:image/jpeg;base64,{}", BASE64.encode(bytes))}
    });
    HashMap::from([
        ("role".to_string(), serde_json::json!("user")),
        (
            "content".to_string(),
            serde_json::json!([{"type": "text", "text": text}, image]),
        ),
    ])
}

async fn turn(
    client: &Client,
    model: &str,
    messages: Vec<Message>,
) -> Result<String, Box<dyn std::error::Error>> {
    let params = SamplingParams {
        max_tokens: 160,
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
        "  ttft {:>5.0}ms | {n:>3} gen tokens | prompt {prompt_tokens:>4} tok, {cached:>4} cached ({:.0}%)",
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
    registry.ensure_loaded(&model).await?;

    let system = "You are a careful visual assistant. ".repeat(8);
    let mut transcript = vec![
        text_msg("system", &system),
        image_msg("Describe this image in about 80 words.", "apple.jpg"),
    ];

    println!("turn 1 (one image):");
    let a1 = turn(&client, &model, transcript.clone()).await?;
    transcript.push(text_msg("assistant", &a1));
    transcript.push(image_msg(
        "Now compare it with this second image in about 60 words.",
        "bottles.jpg",
    ));

    println!("turn 2 (turn 1 + the model's reply + a second image):");
    let _ = turn(&client, &model, transcript.clone()).await?;

    println!("turn 2 again (identical prompt):");
    let _ = turn(&client, &model, transcript).await?;
    Ok(())
}
