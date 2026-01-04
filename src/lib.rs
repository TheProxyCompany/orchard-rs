//! Orchard - Rust client for high-performance LLM inference on Apple Silicon.
//!
//! This crate provides a complete client library for communicating with PIE (Proxy Inference Engine),
//! including:
//!
//! - **IPC Transport**: High-performance NNG-based communication
//! - **Engine Management**: Binary fetching, process lifecycle, and coordination
//! - **Model Management**: Resolution, registry, and state tracking
//! - **Chat Formatting**: Jinja2-compatible template rendering
//! - **Multimodal Support**: Image and capability handling
//! - **Client API**: High-level chat interface
//!
//! # Example
//!
//! ```no_run
//! use orchard::{Client, ClientError, SamplingParams};
//! use std::collections::HashMap;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), ClientError> {
//!     // Create registry and client
//!     let registry = Arc::new(orchard::ModelRegistry::new());
//!     let client = Client::connect(registry)?;
//!
//!     // Build messages
//!     let mut msg = HashMap::new();
//!     msg.insert("role".to_string(), serde_json::json!("user"));
//!     msg.insert("content".to_string(), serde_json::json!("Hello!"));
//!
//!     // Generate response
//!     let response = client.chat(
//!         "meta-llama/Llama-3.1-8B-Instruct",
//!         vec![msg],
//!         SamplingParams::default(),
//!     )?;
//!
//!     println!("{}", response.text);
//!     Ok(())
//! }
//! ```

mod error;

pub mod client;
pub mod engine;
pub mod formatter;
pub mod ipc;
pub mod model;

// Re-export main types
pub use error::Error;

// IPC
pub use ipc::client::{IPCClient, RequestOptions, ResponseDelta};
pub use ipc::endpoints;
pub use ipc::serialization::{
    build_batch_request_payload, CapabilityEntry, LayoutEntry, PromptPayload, RequestType, SegmentType,
};

// Engine
pub use engine::fetch::{EngineFetcher, FetchError};
pub use engine::lifecycle::{EnginePaths, InferenceEngine, LifecycleError};
pub use engine::multiprocess;

// Model
pub use model::registry::{ModelEntry, ModelInfo, ModelLoadState, ModelRegistry};
pub use model::resolver::{ModelResolutionError, ModelResolver, ResolvedModel};

// Formatter
pub use formatter::control_tokens::{ControlTokens, Role, RoleTags};
pub use formatter::multimodal::{
    build_multimodal_layout, build_multimodal_messages, CapabilityInput, ContentType,
    LayoutSegment, MultimodalError,
};
pub use formatter::{ChatFormatter, FormatterError};

// Client
pub use client::{ChatResult, Client, ClientDelta, ClientError, ClientResponse, SamplingParams, UsageStats};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
