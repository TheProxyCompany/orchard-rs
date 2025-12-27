//! IPC communication with PIE (Proxy Inference Engine).
//!
//! This module provides NNG-based IPC for communicating with the inference engine.
//! It mirrors the implementations in orchard-py and orchard-swift.

pub mod client;
pub mod endpoints;
pub mod serialization;
