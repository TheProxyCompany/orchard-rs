# orchard-rs

[![Crates.io](https://img.shields.io/crates/v/orchard-rs.svg)](https://crates.io/crates/orchard-rs)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![macOS](https://img.shields.io/badge/macOS-14%2B-111111.svg)](#requirements)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-required-024645.svg)](#requirements)

Embeddable Rust client for Orchard, the local inference runtime for Apple
Silicon.

The crate is published as `orchard-rs` and imported as `orchard`. It manages
the local Proxy Inference Engine process, downloads or resolves model weights,
formats prompts with Pantheon profiles, and talks to the engine over local IPC.
Use `orchard-rs` when Orchard is part of a Rust application or service. If you
want a standalone Python package or an optional OpenAI-compatible HTTP server,
use [`orchard`](https://github.com/TheProxyCompany/orchard-py).

[Official docs](https://docs.theproxycompany.com/orchard/) cover the shared Orchard API,
models, and deployment patterns.

## Install

```toml
[dependencies]
orchard-rs = "2026.5.6"
base64 = "0.22"
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## Quickstart

```rust
use std::collections::HashMap;
use std::sync::Arc;

use orchard::{ChatResult, Client, InferenceEngine, ModelRegistry, SamplingParams};

fn message(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    let mut message = HashMap::new();
    message.insert("role".to_string(), serde_json::json!(role));
    message.insert("content".to_string(), serde_json::json!(content));
    message
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(Arc::clone(&registry)).await?;

    let model = "google/gemma-4-E2B-it";
    registry.ensure_loaded(model).await?;

    let params = SamplingParams {
        max_tokens: 64,
        temperature: 0.0,
        ..Default::default()
    };

    let result = client
        .achat(
            model,
            vec![message("user", "Write one sentence about local AI.")],
            params,
            true,
        )
        .await?;

    if let ChatResult::Stream(mut stream) = result {
        while let Some(delta) = stream.recv().await {
            if let Some(text) = delta.content {
                print!("{text}");
            }
        }
        println!();
    }

    Ok(())
}
```

## Responses API

```rust
use std::sync::Arc;

use orchard::{
    Client, InferenceEngine, ModelRegistry, ResponseOutputItem, ResponsesRequest,
    ResponsesResult,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(Arc::clone(&registry)).await?;

    let model = "google/gemma-4-E2B-it";
    registry.ensure_loaded(model).await?;

    let mut request = ResponsesRequest::from_text(
        "Explain why local inference is useful in two sentences.",
    );
    request.temperature = Some(0.0);
    request.max_output_tokens = Some(96);

    let result = client.aresponses(model, request).await?;
    if let ResponsesResult::Complete(response) = result {
        for item in &response.output {
            if let ResponseOutputItem::Message(message) = item {
                for content in &message.content {
                    print!("{}", content.text);
                }
            }
        }
        println!();
    }

    Ok(())
}
```

## Batching

Use `achat_batch()` to send multiple conversations in one request. The engine
schedules them together and Orchard returns responses in prompt order.

```rust
let params = SamplingParams {
    max_tokens: 24,
    temperature: 0.0,
    ..Default::default()
};

let conversations = vec![
    vec![message("user", "Say hello politely.")],
    vec![message("user", "Give me a fun fact about space.")],
];

let result = client
    .achat_batch("google/gemma-4-E2B-it", conversations, params, false)
    .await?;
```

## Multi-turn conversations

Append `assistant_message` after each reply and send the conversation back as it is:

```rust
let ChatResult::Stream(mut stream) = client.achat(model, messages.clone(), params.clone(), true).await? else {
    unreachable!()
};
let mut deltas = Vec::new();
while let Some(delta) = stream.recv().await {
    let done = delta.is_final_delta;
    deltas.push(delta);
    if done {
        break;
    }
}
messages.push(client.assistant_message(model, &params, deltas).await?);
messages.push(user("and the next question"));
```

The message keeps the reply's reasoning and tool calls, and a `generation` record with
the exact token ids the model produced. Sent back to the same model, the reply is
replayed id for id, so the engine serves the whole conversation so far from its prefix
cache and computes only the new question. Any other model reads the text fields. With the
Responses API, `response_input_items(&response.output, response.generation.as_ref())`
does the same. `examples/replay_turns.rs` and `examples/replay_responses.rs` print the
cache reuse per turn.

## Multimodal

Vision-capable models accept OpenAI-style content parts. Use data URLs for
local images.

```rust
use std::collections::HashMap;

use base64::Engine;
use orchard::SamplingParams;

fn image_message(
    path: &str,
) -> Result<HashMap<String, serde_json::Value>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let image_base64 = base64::engine::general_purpose::STANDARD.encode(bytes);

    let mut message = HashMap::new();
    message.insert("role".to_string(), serde_json::json!("user"));
    message.insert(
        "content".to_string(),
        serde_json::json!([
            {"type": "text", "text": "Describe this image in one sentence."},
            {
                "type": "image_url",
                "image_url": {"url": format!("data:image/jpeg;base64,{image_base64}")}
            }
        ]),
    );
    Ok(message)
}

let params = SamplingParams {
    max_tokens: 96,
    temperature: 0.0,
    ..Default::default()
};

let result = client
    .achat(
        "google/gemma-3-4b-it",
        vec![image_message("apple.jpg")?],
        params,
        false,
    )
    .await?;
```

## Features

- Embedded Rust API for apps that own their process lifecycle.
- Engine lifecycle management with automatic binary fetch.
- Hugging Face and local-path model resolution.
- Async chat and Responses APIs.
- Streaming token deltas over local IPC.
- Batched chat requests.
- Structured output, tool-call schemas, reasoning effort, and multimodal layout.
- Pantheon-backed chat templates and control tokens shared with Orchard Python.

## Requirements

- macOS 14 or newer
- Apple Silicon Mac
- Rust 1.70 or newer
- A local Orchard engine binary, downloaded automatically on first use

## Development

```bash
cargo check
cargo test
```

End-to-end tests start the local engine and are ignored by default:

```bash
cargo test --test e2e -- --ignored
```

Inside the Proxy Company hyper-repo, use the full Orchard gate when changing
client behavior:

```bash
./scripts/pie_cycle.sh --rs-only
```

## Related

- [Orchard Python](https://github.com/TheProxyCompany/orchard-py)
- [Official Orchard docs](https://docs.theproxycompany.com/orchard/)
- [Pantheon](https://github.com/TheProxyCompany/Pantheon)
- [Proxy Inference Engine](https://github.com/TheProxyCompany/proxy-inference-engine)

## License

Apache-2.0

## Native voice and speaker streams

Build with `--features duplex,diarization`. PIE owns Moshi/Mimi and Nemotron3
model weights, streaming state, inference and native backend libraries. The Rust
SDK resolves pinned assets, publishes immutable descriptors and transports
audio/events over bounded IPC. `duplex-metal` remains a source
compatibility alias for `duplex`.

```rust
use orchard::duplex::{DuplexOptions, DEFAULT_MODEL};
use orchard::diarization::DiarizationOptions;

let mut diarizer = client.diarization(
    "nvidia/Nemotron-3-Diarization", DiarizationOptions::default()
).await?;
let mut voice = client.duplex(DEFAULT_MODEL, DuplexOptions::default()).await?;
let control = voice.control();
control.push_audio(0, vec![0.0; 1920])?; // 80 ms, mono float32 at 24 kHz
let epoch = control.epoch();
control.reference("The test flag is purple.".into(), epoch)?;
// Drain voice.next_event() concurrently with input.
let next_epoch = control.interrupt()?; // acknowledged by PIE before returning
```

This native implementation supports the pinned MoshiRAG `v0_1` safetensors
checkpoint and ARC assets declared in Pantheon. The source repository name
`kyutai/moshika-rag-candle-bf16` identifies the checkpoint's layout; inference is
C++/Carbon in PIE. Source weights stay unchanged, and the SDK creates a small,
versioned descriptor in its private cache. No implicit Q8 conversion is made.
Unsupported base/GGUF formats and legacy CPU/cadence overrides fail explicitly.

The default voice is autonomous, full-duplex generation: a pending backbone
request does not mute its PCM. Concise factual references condition the trained
RAG channel asynchronously. Generated voice text is distinct from the backbone's
answer and microphone ASR; it cannot execute tools or authorize actions.
`reference_applied` reports actual consumption, not merely receipt of a command.
Interrupt and reset epochs fence already-buffered old audio. Model loading,
readiness, cancellation, session cleanup and unload are owned by PIE.

When Ready advertises `supports_response_hold`, call `hold_response(epoch)`
explicitly while waiting for factual context. Its acknowledgment returns the
native reference version without advancing the epoch or stopping microphone
input and the audio clock. A matching `grounded_reply(context, epoch)` applies
the context and releases the hold for a model-authored response. Background
`reference` updates do not release it; interrupt and reset retire the old hold.
The caller still owns which pending result is current. Retrieval events alone
never trigger a hold in the SDK, and a control acknowledgment does not confirm
speech generation or playback.

With `realtime: false` (the default), microphone frames including silence drive
the clock. `realtime: true` lets the engine synthesize clock ticks when no frame
is available. Input and output queues are bounded; metrics report actual frame
drops, queue delay and compute time. The client binds a `pull_v1` response route
before opening the session, so audio does not depend on lossy PUB/SUB delivery.
A connected PIE build with native Moshi and Nemotron3 support is required.

Nemotron 3 returns eight anonymous, session-local speaker channels with 10 ms
posterior frames and overlapping segments. A speaker channel is not a person's
identity. Entity binding belongs to the application and needs confirmed evidence.
Device and streaming geometry select an immutable engine model descriptor;
the SDK's default CPU selection also executes inside PIE. Diarization input
receives a bounded queue admission ACK. A full queue rejects the frame without
advancing its sequence; retry it or let the next admission report the gap.
`finish()` gates new input and drains accepted frames through the final update,
metrics and `Closed`. `shutdown()` waits for PIE's terminal confirmation.

The PIE artifact includes the pinned NVIDIA backend and its dylib dependencies.
`orchard-duplex` and `orchard-diarize` are thin JSON-lines clients for the engine's
voice and diarization sessions. Client applications need no NeMo/GGML inference
library loaded into their own process.
