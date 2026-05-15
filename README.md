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
