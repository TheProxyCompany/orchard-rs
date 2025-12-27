//! Orchard - Rust client for high-performance LLM inference on Apple Silicon.
//!
//! This crate provides IPC communication with PIE (Proxy Inference Engine) using NNG.
//!
//! # Example
//!
//! ```no_run
//! use orchard::{IPCClient, ResponseDelta};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), orchard::Error> {
//!     let mut client = IPCClient::new();
//!     client.connect()?;
//!
//!     let request_id = client.next_request_id();
//!     let mut stream = client.send_request(
//!         request_id,
//!         "qwen-2.5-coder-32b",
//!         "/path/to/model",
//!         "Hello, world!",
//!         Default::default(),
//!     )?;
//!
//!     while let Some(delta) = stream.recv().await {
//!         if let Some(content) = delta.content {
//!             print!("{}", content);
//!         }
//!     }
//!
//!     client.disconnect();
//!     Ok(())
//! }
//! ```

mod error;
pub mod ipc;

pub use error::Error;
pub use ipc::client::{IPCClient, RequestOptions, ResponseDelta};
pub use ipc::endpoints;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
