//! One turn against a local model. Prints tokens as they stream, then TTFT and
//! how many prompt tokens hit the prefix cache. Run it twice to see the cache.
//!
//!   cargo run --example one_turn
//!   MODEL=google/gemma-4-26B-A4B-it cargo run --example one_turn

mod common;

use std::io::Write;
use std::time::Instant;

use common::{msg, run_turn};
use orchard::SamplingParams;

#[tokio::main]
async fn main() -> Result<(), common::Error> {
    let model = std::env::var("MODEL").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Say hello in one sentence.".into());

    let t = Instant::now();
    let (_engine, registry, client) = common::connect().await?;
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

    let turn = run_turn(&client, &model, messages, params, |d| {
        if let Some(c) = &d.content {
            print!("{c}");
            let _ = std::io::stdout().flush();
        }
    })
    .await?;
    println!();
    let (n, prompt_tokens, cached) = (turn.ids.len(), turn.prompt_tokens, turn.cached);
    eprintln!(
        "[ttft {:.0}ms | {n} tokens in {:.2}s | prompt {prompt_tokens} tok, {cached} from prefix cache]",
        turn.ttft_ms,
        turn.elapsed.as_secs_f64()
    );
    Ok(())
}
