//! Error types for Orchard.

use thiserror::Error;

/// Orchard error type.
#[derive(Error, Debug)]
pub enum Error {
    /// IPC client is not connected
    #[error("IPC client not connected")]
    NotConnected,

    /// Invalid response from PIE
    #[error("Invalid response from PIE")]
    InvalidResponse,

    /// NNG socket error
    #[error("NNG error: {0}")]
    Nng(#[from] nng::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Request timeout
    #[error("Request timeout")]
    Timeout,

    /// Channel closed
    #[error("Channel closed")]
    ChannelClosed,

    /// Engine fetch error
    #[error("Fetch error: {0}")]
    Fetch(#[from] crate::engine::fetch::FetchError),

    /// Engine lifecycle error
    #[error("Lifecycle error: {0}")]
    Lifecycle(#[from] crate::engine::lifecycle::LifecycleError),

    /// Model resolution error
    #[error("Model resolution error: {0}")]
    ModelResolution(#[from] crate::model::resolver::ModelResolutionError),

    /// Formatter error
    #[error("Formatter error: {0}")]
    Formatter(#[from] crate::formatter::FormatterError),

    /// Multimodal error
    #[error("Multimodal error: {0}")]
    Multimodal(#[from] crate::formatter::multimodal::MultimodalError),
}

/// Result type alias for Orchard operations.
pub type Result<T> = std::result::Result<T, Error>;
