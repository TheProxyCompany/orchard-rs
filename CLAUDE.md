# CLAUDE.md

Rust client for Orchard - high-performance LLM inference on Apple Silicon.

## What is orchard-rs?

orchard-rs is the Rust client library for communicating with PIE (Proxy Inference Engine). It provides:

- **NNG IPC** - High-performance inter-process communication using nanomsg-next-gen
- **Streaming responses** - Async token streaming via tokio channels
- **Binary serialization** - Wire protocol matching orchard-py and orchard-swift

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   orchard-rs                     │
├─────────────────────────────────────────────────┤
│  IPCClient                                       │
│    ├── Push socket → PIE request queue          │
│    ├── Sub socket  ← PIE response stream        │
│    └── Req socket  ↔ PIE management (sync)      │
├─────────────────────────────────────────────────┤
│  Serialization                                   │
│    └── Binary wire format (16-byte aligned)     │
└─────────────────────────────────────────────────┘
```

## Build

```bash
cargo build --release
cargo test
```

## Usage

```rust
use orchard::{IPCClient, RequestOptions};

let mut client = IPCClient::new();
client.connect()?;

let request_id = client.next_request_id();
let mut stream = client.send_request(
    request_id,
    "qwen-2.5-coder-32b",
    "/path/to/model",
    "Hello, world!",
    RequestOptions::default(),
)?;

while let Some(delta) = stream.recv().await {
    if let Some(content) = delta.content {
        print!("{}", content);
    }
}
```

## Related Crates

- **orchard-py** - Python client (PyPI)
- **orchard-swift** - Swift client (SwiftPM)
- **PIE** - C++ inference engine (private)

## Publish

```bash
gh workflow run publish.yml -R TheProxyCompany/orchard-rs
```

Publishes to crates.io with calver versioning (YYYY.MM.PATCH).
