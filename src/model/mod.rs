//! Model management for Orchard.
//!
//! This module provides:
//! - Model resolution (`resolver`) - mapping identifiers to local paths
//! - Model registry (`registry`) - tracking loaded models and their state

pub mod registry;
pub mod resolver;

pub use registry::{ModelEntry, ModelInfo, ModelLoadState, ModelRegistry};
pub use resolver::{ModelResolutionError, ModelResolver, ResolvedModel};
