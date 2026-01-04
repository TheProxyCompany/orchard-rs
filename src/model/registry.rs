//! Model registry for tracking loaded models and their state.
//!
//! Provides a state machine for model lifecycle:
//! IDLE -> DOWNLOADING -> LOADING -> ACTIVATING -> READY -> FAILED

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Notify, RwLock};

use crate::formatter::ChatFormatter;
use crate::model::resolver::{ModelResolver, ResolvedModel};

/// Model load state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelLoadState {
    /// Model not yet requested
    Idle,
    /// Downloading from HuggingFace
    Downloading,
    /// Loading weights into engine
    Loading,
    /// Waiting for engine activation
    Activating,
    /// Ready for inference
    Ready,
    /// Failed to load
    Failed,
}

impl std::fmt::Display for ModelLoadState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "IDLE"),
            Self::Downloading => write!(f, "DOWNLOADING"),
            Self::Loading => write!(f, "LOADING"),
            Self::Activating => write!(f, "ACTIVATING"),
            Self::Ready => write!(f, "READY"),
            Self::Failed => write!(f, "FAILED"),
        }
    }
}

/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model identifier
    pub model_id: String,
    /// Path to model weights
    pub model_path: String,
    /// Chat formatter for this model
    pub formatter: Arc<ChatFormatter>,
    /// Token ID capabilities (e.g., EOS, BOS token IDs)
    pub capabilities: Option<HashMap<String, Vec<i32>>>,
}

/// Entry in the model registry tracking a model's state.
pub struct ModelEntry {
    /// Current state
    pub state: ModelLoadState,
    /// Model info (available after loading)
    pub info: Option<ModelInfo>,
    /// Error message if failed
    pub error: Option<String>,
    /// Notification for state changes
    pub notify: Arc<Notify>,
    /// Resolution result
    pub resolved: Option<ResolvedModel>,
    /// Download progress (bytes downloaded)
    pub bytes_downloaded: Option<u64>,
    /// Download progress (total bytes)
    pub bytes_total: Option<u64>,
}

impl Default for ModelEntry {
    fn default() -> Self {
        Self {
            state: ModelLoadState::Idle,
            info: None,
            error: None,
            notify: Arc::new(Notify::new()),
            resolved: None,
            bytes_downloaded: None,
            bytes_total: None,
        }
    }
}

/// Registry of loaded models.
///
/// Tracks model state and provides methods for ensuring models are loaded.
pub struct ModelRegistry {
    /// Model entries by canonical ID
    entries: RwLock<HashMap<String, ModelEntry>>,
    /// Model resolver
    resolver: Mutex<ModelResolver>,
    /// Alias cache: short name -> canonical ID
    alias_cache: RwLock<HashMap<String, String>>,
}

impl ModelRegistry {
    /// Create a new model registry.
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            resolver: Mutex::new(ModelResolver::new()),
            alias_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Ensure a model is loaded and ready.
    ///
    /// This will:
    /// 1. Resolve the model identifier
    /// 2. Download if needed
    /// 3. Load the formatter
    /// 4. Wait for engine activation
    pub async fn ensure_loaded(&self, requested_model_id: &str) -> Result<ModelInfo, String> {
        let (_state, canonical_id) = self.schedule_model(requested_model_id, false).await?;

        // Wait for local readiness
        let (state, info, error) = self.await_model(&canonical_id, None).await?;

        if state == ModelLoadState::Failed {
            return Err(error.unwrap_or_else(|| format!("Model '{}' failed to load", canonical_id)));
        }

        if state == ModelLoadState::Ready {
            return info.ok_or_else(|| "Model ready but info missing".to_string());
        }

        // Model is in Loading state, need to activate with engine
        // This would normally involve IPC communication
        // For now, we just return the info if available
        info.ok_or_else(|| format!("Model '{}' not ready", canonical_id))
    }

    /// Schedule a model for loading.
    ///
    /// Returns the current state and canonical ID.
    pub async fn schedule_model(
        &self,
        requested_model_id: &str,
        force_reload: bool,
    ) -> Result<(ModelLoadState, String), String> {
        // Resolve the model
        let resolved = {
            let mut resolver = self.resolver.lock().await;
            resolver
                .resolve(requested_model_id)
                .await
                .map_err(|e| e.to_string())?
        };

        let canonical_id = resolved.canonical_id.clone();

        // Update alias cache
        {
            let mut alias_cache = self.alias_cache.write().await;
            alias_cache.insert(requested_model_id.to_lowercase(), canonical_id.clone());
            alias_cache
                .entry(canonical_id.to_lowercase())
                .or_insert_with(|| canonical_id.clone());
        }

        // Check/create entry
        let mut entries = self.entries.write().await;
        let entry = entries
            .entry(canonical_id.clone())
            .or_insert_with(ModelEntry::default);

        // Check if already ready
        if entry.state == ModelLoadState::Ready && !force_reload {
            return Ok((ModelLoadState::Ready, canonical_id));
        }

        // Check if already loading
        if matches!(
            entry.state,
            ModelLoadState::Loading | ModelLoadState::Downloading | ModelLoadState::Activating
        ) && !force_reload
        {
            return Ok((entry.state, canonical_id));
        }

        // Check if failed
        if entry.state == ModelLoadState::Failed && !force_reload {
            return Ok((ModelLoadState::Failed, canonical_id));
        }

        // Reset for loading
        entry.error = None;
        entry.info = None;
        entry.resolved = Some(resolved.clone());
        entry.bytes_downloaded = None;
        entry.bytes_total = None;
        entry.notify = Arc::new(Notify::new());

        // If model is already local, build formatter
        if resolved.source == "local" || resolved.source == "hf_cache" {
            match ChatFormatter::new(&resolved.model_path) {
                Ok(formatter) => {
                    entry.info = Some(ModelInfo {
                        model_id: canonical_id.clone(),
                        model_path: resolved.model_path.to_string_lossy().to_string(),
                        formatter: Arc::new(formatter),
                        capabilities: None,
                    });
                    entry.state = ModelLoadState::Loading;
                    entry.notify.notify_waiters();
                    return Ok((ModelLoadState::Loading, canonical_id));
                }
                Err(e) => {
                    entry.error = Some(e.to_string());
                    entry.state = ModelLoadState::Failed;
                    entry.notify.notify_waiters();
                    return Ok((ModelLoadState::Failed, canonical_id));
                }
            }
        }

        // Need to download
        entry.state = ModelLoadState::Downloading;
        Ok((ModelLoadState::Downloading, canonical_id))
    }

    /// Wait for a model to finish loading.
    pub async fn await_model(
        &self,
        model_id: &str,
        timeout: Option<std::time::Duration>,
    ) -> Result<(ModelLoadState, Option<ModelInfo>, Option<String>), String> {
        let canonical_id = self.canonicalize(model_id).await?;

        let notify = {
            let entries = self.entries.read().await;
            let entry = entries
                .get(&canonical_id)
                .ok_or_else(|| format!("Model '{}' has not been scheduled", model_id))?;
            entry.notify.clone()
        };

        // Wait for notification or timeout
        if let Some(duration) = timeout {
            if tokio::time::timeout(duration, notify.notified())
                .await
                .is_err()
            {
                let entries = self.entries.read().await;
                if let Some(entry) = entries.get(&canonical_id) {
                    return Ok((entry.state, entry.info.clone(), entry.error.clone()));
                }
            }
        } else {
            notify.notified().await;
        }

        let entries = self.entries.read().await;
        let entry = entries
            .get(&canonical_id)
            .ok_or_else(|| format!("Model '{}' not found", canonical_id))?;

        Ok((entry.state, entry.info.clone(), entry.error.clone()))
    }

    /// Get model info if ready.
    pub async fn get_if_ready(&self, model_id: &str) -> Option<ModelInfo> {
        let canonical_id = self.canonicalize(model_id).await.ok()?;
        let entries = self.entries.read().await;
        let entry = entries.get(&canonical_id)?;

        if entry.state == ModelLoadState::Ready {
            entry.info.clone()
        } else {
            None
        }
    }

    /// Get model status.
    pub async fn get_status(
        &self,
        model_id: &str,
    ) -> (ModelLoadState, Option<String>, Option<(u64, u64)>) {
        let canonical_id = match self.canonicalize(model_id).await {
            Ok(id) => id,
            Err(_) => return (ModelLoadState::Idle, None, None),
        };

        let entries = self.entries.read().await;
        let entry = match entries.get(&canonical_id) {
            Some(e) => e,
            None => return (ModelLoadState::Idle, None, None),
        };

        let progress = match (entry.bytes_downloaded, entry.bytes_total) {
            (Some(d), Some(t)) => Some((d, t)),
            _ => None,
        };

        (entry.state, entry.error.clone(), progress)
    }

    /// Update capabilities for a loaded model.
    pub async fn update_capabilities(
        &self,
        model_id: &str,
        capabilities: HashMap<String, Vec<i32>>,
    ) {
        let canonical_id = match self.canonicalize(model_id).await {
            Ok(id) => id,
            Err(_) => {
                log::warn!("Received capabilities for unknown model '{}'", model_id);
                return;
            }
        };

        let mut entries = self.entries.write().await;
        if let Some(entry) = entries.get_mut(&canonical_id) {
            if let Some(ref mut info) = entry.info {
                info.capabilities = Some(capabilities);
            }
        }
    }

    /// Mark a model as ready (called when engine confirms activation).
    pub async fn mark_ready(&self, model_id: &str) {
        let canonical_id = match self.canonicalize(model_id).await {
            Ok(id) => id,
            Err(_) => return,
        };

        let mut entries = self.entries.write().await;
        if let Some(entry) = entries.get_mut(&canonical_id) {
            entry.state = ModelLoadState::Ready;
            entry.notify.notify_waiters();
        }
    }

    /// Mark a model as failed.
    pub async fn mark_failed(&self, model_id: &str, error: String) {
        let canonical_id = match self.canonicalize(model_id).await {
            Ok(id) => id,
            Err(_) => return,
        };

        let mut entries = self.entries.write().await;
        if let Some(entry) = entries.get_mut(&canonical_id) {
            entry.state = ModelLoadState::Failed;
            entry.error = Some(error);
            entry.notify.notify_waiters();
        }
    }

    /// List all registered models.
    pub async fn list_models(&self) -> Vec<HashMap<String, String>> {
        let entries = self.entries.read().await;
        let mut catalog = Vec::new();

        for (canonical_id, entry) in entries.iter() {
            if let Some(ref resolved) = entry.resolved {
                let mut payload: HashMap<String, String> = resolved.metadata.clone();
                payload.insert("canonical_id".to_string(), canonical_id.clone());
                payload.insert(
                    "model_path".to_string(),
                    resolved.model_path.to_string_lossy().to_string(),
                );
                payload.insert("source".to_string(), resolved.source.clone());
                payload.insert(
                    "hf_repo".to_string(),
                    resolved.hf_repo.clone().unwrap_or_default(),
                );
                payload.insert("state".to_string(), entry.state.to_string());
                catalog.push(payload);
            }
        }

        catalog
    }

    async fn canonicalize(&self, model_id: &str) -> Result<String, String> {
        // Check if it's already a canonical ID
        {
            let entries = self.entries.read().await;
            if entries.contains_key(model_id) {
                return Ok(model_id.to_string());
            }
        }

        // Check alias cache
        {
            let alias_cache = self.alias_cache.read().await;
            if let Some(canonical) = alias_cache.get(&model_id.to_lowercase()) {
                return Ok(canonical.clone());
            }
        }

        Err(format!("Model '{}' not found in registry", model_id))
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_creation() {
        let registry = ModelRegistry::new();
        let models = registry.list_models().await;
        assert!(models.is_empty());
    }

    #[test]
    fn test_model_load_state_display() {
        assert_eq!(ModelLoadState::Idle.to_string(), "IDLE");
        assert_eq!(ModelLoadState::Ready.to_string(), "READY");
        assert_eq!(ModelLoadState::Failed.to_string(), "FAILED");
    }
}
