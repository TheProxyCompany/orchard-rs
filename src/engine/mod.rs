//! Engine management for PIE (Proxy Inference Engine).

pub mod fetch;
pub mod lifecycle;
pub mod multiprocess;

pub use fetch::EngineFetcher;
pub use lifecycle::InferenceEngine;
