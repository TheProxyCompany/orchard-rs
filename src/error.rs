//! Unified error type for Orchard.

use std::path::PathBuf;
use thiserror::Error;

/// Orchard error type.
#[derive(Error, Debug)]
pub enum Error {
    // === IPC Errors ===
    #[error("IPC client not connected")]
    NotConnected,
    #[error("Invalid response from PIE")]
    InvalidResponse,
    #[error("NNG error: {0}")]
    Nng(#[from] nng::Error),
    #[error("Request timeout")]
    Timeout,
    #[error("Channel closed")]
    ChannelClosed,

    // === Serialization Errors ===
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    // === Model Errors ===
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Model not ready: {0}")]
    ModelNotReady(String),
    #[error("Model identifier cannot be empty")]
    EmptyModelId,
    #[error("Model directory '{0}' is missing config.json")]
    MissingConfig(PathBuf),
    #[error("Failed to download model '{0}': {1}")]
    DownloadFailed(String, String),
    #[error("Failed to initialize HuggingFace API: {0}")]
    HfApiInit(String),

    // === Formatter Errors ===
    #[error("Formatter config not found: {0}")]
    FormatterConfigNotFound(String),
    #[error("Formatter profile not found: {0}")]
    FormatterProfileNotFound(String),
    #[error("Template error: {0}")]
    Template(String),

    // === Multimodal Errors ===
    #[error("Invalid image data URL format")]
    InvalidImageUrl,
    #[error("Invalid base64-encoded image content")]
    InvalidBase64,
    #[error("Content part {0} in message {1} is missing a valid 'type'")]
    MissingContentType(usize, usize),
    #[error("Message content must be a string or list of content parts")]
    InvalidContent,
    #[error("Mismatch between image placeholders ({0}) and supplied images ({1})")]
    PlaceholderMismatch(usize, usize),
    #[error("Response request must include at least one content segment")]
    EmptyRequest,

    // === Engine Lifecycle Errors ===
    #[error("Failed to acquire lock: {0}")]
    LockFailed(String),
    #[error("Engine startup failed: {0}")]
    StartupFailed(String),
    #[error("Engine shutdown failed: {0}")]
    ShutdownFailed(String),
    #[error("Engine already closed")]
    EngineClosed,

    // === Fetch Errors ===
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Network error: {0}")]
    Network(String),
    #[error("Integrity check failed: expected {expected}, got {actual}")]
    Integrity { expected: String, actual: String },
    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),
    #[error("No compatible binary for this platform")]
    NoBinaryForPlatform,
    #[error("Extract error: {0}")]
    Extract(String),

    // === Generic ===
    #[error("{0}")]
    Other(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;
