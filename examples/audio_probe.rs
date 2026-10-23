//! Grounded local audio probe. Feed a mono 16 kHz float32-le file whose words
//! are not supplied in the prompt. Use isolated Orchard IPC/cache roots.
mod common;
use orchard::SamplingParams;
use serde_json::json;
use std::collections::HashMap;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), common::Error> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: audio_probe MODEL PCM_F32LE [--stt]".into());
    }
    let model = &args[1];
    let raw = std::fs::read(&args[2])?;
    if raw.len() % 4 != 0 {
        return Err("float32 PCM alignment".into());
    }
    let pcm: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let (_engine, registry, client) = common::connect().await?;
    let info =
        tokio::time::timeout(Duration::from_secs(180), registry.ensure_loaded(model)).await??;
    println!(
        "MODEL {}",
        json!({"model":info.model_id,"path":info.model_path,"capabilities":info.capabilities})
    );
    if args.get(3).map(String::as_str) == Some("--stt") {
        let text = tokio::time::timeout(
            Duration::from_secs(120),
            client.atranscribe_audio(model, &pcm),
        )
        .await??;
        println!("TRANSCRIPT {}", json!({"text":text}));
    } else {
        let message = HashMap::from([
            ("role".into(), json!("user")),
            (
                "content".into(),
                json!([
                    {"type":"text","text":"Transcribe the audio verbatim. Output only the words you actually hear."},
                    {"type":"input_audio","data":pcm}
                ]),
            ),
        ]);
        let params = SamplingParams {
            max_tokens: 128,
            temperature: 0.0,
            reasoning: Some(false),
            // Grounding probes must not reuse KV from a previous engine math revision.
            prefix_cache: Some(false),
            ..Default::default()
        };
        let result = tokio::time::timeout(
            Duration::from_secs(120),
            common::run_turn(&client, model, vec![message], params, |_| {}),
        )
        .await??;
        println!(
            "TRANSCRIPT {}",
            json!({"text":result.text,"elapsed_ms":result.elapsed.as_secs_f64()*1000.0,"cached_tokens":result.cached,"prompt_tokens":result.prompt_tokens})
        );
    }
    Ok(())
}
