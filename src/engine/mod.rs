//! Engine management for PIE (Proxy Inference Engine).
//!
//! This module provides:
//! - Binary fetching and installation (`fetch`)
//! - Process lifecycle management (`lifecycle`)
//! - Cross-process coordination (`multiprocess`)

pub mod fetch;
pub mod lifecycle;
pub mod multiprocess;

pub use fetch::{EngineFetcher, FetchError};
pub use lifecycle::InferenceEngine;
