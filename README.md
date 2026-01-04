# orchard-rs

Rust client for [Orchard](https://github.com/TheProxyCompany) - high-performance LLM inference on Apple Silicon.

## Installation

```toml
[dependencies]
orchard-rs = "2025.12"
```

## Usage

```rust
use orchard::{IPCClient, RequestOptions};

#[tokio::main]
async fn main() -> Result<(), orchard::Error> {
    // Connect to PIE (Proxy Inference Engine)
    let mut client = IPCClient::new();
    client.connect()?;

    // Send inference request
    let request_id = client.next_request_id();
    let mut stream = client.send_request(
        request_id,
        "qwen-2.5-coder-32b",
        "/path/to/model",
        "Explain quantum computing in simple terms.",
        RequestOptions {
            max_tokens: 500,
            temperature: 0.7,
            ..Default::default()
        },
    )?;

    // Stream response tokens
    while let Some(delta) = stream.recv().await {
        if let Some(content) = delta.content {
            print!("{}", content);
        }
        if delta.is_final_delta {
            println!();
            break;
        }
    }

    client.disconnect();
    Ok(())
}
```

## Features

- **High-performance IPC** - NNG (nanomsg-next-gen) for minimal latency
- **Streaming responses** - Async token streaming via tokio channels
- **Thread-safe** - Lock-based design for concurrent access
- **Wire-compatible** - Same binary protocol as orchard-py and orchard-swift

## Requirements

- Rust 1.70+
- PIE (Proxy Inference Engine) running locally
- macOS 14+ (Apple Silicon)

## Model Profiles

Chat templates and control tokens are loaded from the [orchard-models](https://github.com/TheProxyCompany/orchard-models) submodule at `profiles/`. This provides a single source of truth shared across all Orchard SDKs (Python, Rust, Swift).

## License

Apache-2.0
