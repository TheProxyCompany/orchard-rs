//! Model registry for tracking loaded models and their state.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use hf_hub::api::tokio::ApiBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::{Mutex, Notify, RwLock, oneshot};

use crate::error::Error;
use crate::formatter::ChatFormatter;
use crate::ipc::client::IPCClient;
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
    pub model_id: String,
    pub model_path: String,
    pub formatter: Arc<ChatFormatter>,
    pub capabilities: Option<HashMap<String, Vec<i32>>>,
}

/// Entry in the model registry tracking a model's state.
pub struct ModelEntry {
    pub state: ModelLoadState,
    pub info: Option<ModelInfo>,
    pub error: Option<String>,
    pub notify: Arc<Notify>,
    pub resolved: Option<ResolvedModel>,
    pub bytes_downloaded: Option<u64>,
    pub bytes_total: Option<u64>,
    /// Oneshot sender for activation completion (set when ACTIVATING)
    pub activation_tx: Option<oneshot::Sender<Result<(), String>>>,
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
            activation_tx: None,
        }
    }
}

/// Registry of loaded models.
pub struct ModelRegistry {
    entries: Arc<RwLock<HashMap<String, ModelEntry>>>,
    resolver: Mutex<ModelResolver>,
    alias_cache: RwLock<HashMap<String, String>>,
    /// IPC client for sending management commands to PIE
    ipc_client: RwLock<Option<Arc<IPCClient>>>,
}

impl ModelRegistry {
    /// Create a new model registry.
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            resolver: Mutex::new(ModelResolver::new()?),
            alias_cache: RwLock::new(HashMap::new()),
            ipc_client: RwLock::new(None),
        })
    }

    /// Set the IPC client for sending management commands to PIE.
    pub async fn set_ipc_client(&self, client: Arc<IPCClient>) {
        let mut ipc = self.ipc_client.write().await;
        *ipc = Some(client);
    }

    /// Ensure a model is loaded and ready.
    ///
    /// This will:
    /// 1. Resolve the model identifier
    /// 2. Download if needed
    /// 3. Load the formatter
    /// 4. Send load_model command to PIE
    /// 5. Wait for engine activation
    pub async fn ensure_loaded(&self, requested_model_id: &str) -> Result<ModelInfo, Error> {
        let (_state, canonical_id) = self.schedule_model(requested_model_id, false).await
            .map_err(|e| Error::ModelNotReady(e))?;

        // Wait for local readiness (download + formatter)
        let (state, info, error) = self.await_model(&canonical_id, None).await
            .map_err(|e| Error::ModelNotReady(e))?;

        if state == ModelLoadState::Failed {
            return Err(Error::ModelNotReady(
                error.unwrap_or_else(|| format!("Model '{}' failed to load", canonical_id))
            ));
        }

        if state == ModelLoadState::Ready {
            return info.ok_or_else(|| Error::ModelNotReady("Model ready but info missing".to_string()));
        }

        // At this point we have LOADING state with info - need to activate on PIE
        let info = info.ok_or_else(|| Error::ModelNotReady(format!("Model '{}' info missing", canonical_id)))?;

        // Check if already activating or ready
        {
            let entries = self.entries.read().await;
            if let Some(entry) = entries.get(&canonical_id) {
                if entry.state == ModelLoadState::Ready {
                    return entry.info.clone().ok_or_else(|| Error::ModelNotReady("Ready but no info".to_string()));
                }
            }
        }

        // Send load_model command and wait for activation
        let activation_rx = self.send_load_model_command(
            requested_model_id,
            &canonical_id,
            &info,
        ).await.map_err(|e| Error::ModelNotReady(e))?;

        // Wait for activation to complete
        match activation_rx.await {
            Ok(Ok(())) => {
                // Activation succeeded, get the ready info
                self.get_if_ready(&canonical_id).await
                    .ok_or_else(|| Error::ModelNotReady(format!("Model '{}' failed to activate", canonical_id)))
            }
            Ok(Err(e)) => Err(Error::ModelNotReady(e)),
            Err(_) => Err(Error::ModelNotReady(format!("Activation channel closed for '{}'", canonical_id))),
        }
    }

    /// Send the load_model command to PIE.
    async fn send_load_model_command(
        &self,
        requested_id: &str,
        canonical_id: &str,
        info: &ModelInfo,
    ) -> Result<oneshot::Receiver<Result<(), String>>, String> {
        // Create activation channel
        let (tx, rx) = oneshot::channel();

        // Set up activation state
        {
            let mut entries = self.entries.write().await;
            let entry = entries.get_mut(canonical_id)
                .ok_or_else(|| format!("Model '{}' not in registry", canonical_id))?;

            // If already activating, we can't start another activation
            if entry.state == ModelLoadState::Activating {
                return Err(format!("Model '{}' is already activating", canonical_id));
            }

            entry.state = ModelLoadState::Activating;
            entry.activation_tx = Some(tx);
        }

        // Get IPC client
        let ipc = {
            let guard = self.ipc_client.read().await;
            guard.clone().ok_or_else(|| "IPC client not set".to_string())?
        };

        // Build and send the command
        let command = json!({
            "type": "load_model",
            "requested_id": requested_id,
            "canonical_id": canonical_id,
            "model_path": info.model_path,
            "wait_for_completion": false,
        });

        let response = ipc.send_management_command_async(command, Duration::from_secs(30))
            .await
            .map_err(|e| format!("Failed to send load_model command: {}", e))?;

        // Check response status
        let status = response.get("status").and_then(|v| v.as_str()).unwrap_or("");

        match status {
            "ok" => {
                // Immediate success - extract capabilities and mark ready
                let capabilities = self.parse_capabilities(&response);
                self.complete_activation(canonical_id, capabilities).await;
            }
            "accepted" => {
                // Async activation - wait for model_loaded event
                log::debug!("Model '{}' activation accepted, waiting for model_loaded event", canonical_id);
            }
            _ => {
                let message = response.get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown error");
                self.fail_activation(canonical_id, &format!("Engine rejected load_model: {}", message)).await;
                return Err(format!("Engine rejected load_model for '{}': {}", requested_id, message));
            }
        }

        Ok(rx)
    }

    /// Parse capabilities from management response.
    fn parse_capabilities(&self, response: &Value) -> Option<HashMap<String, Vec<i32>>> {
        response.get("data")
            .and_then(|d| d.get("load_model"))
            .and_then(|lm| lm.get("capabilities"))
            .and_then(|c| c.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        let vals: Vec<i32> = if let Some(arr) = v.as_array() {
                            arr.iter().filter_map(|x| x.as_i64().map(|n| n as i32)).collect()
                        } else if let Some(n) = v.as_i64() {
                            vec![n as i32]
                        } else {
                            return None;
                        };
                        Some((k.clone(), vals))
                    })
                    .collect()
            })
    }

    /// Complete activation successfully.
    async fn complete_activation(&self, model_id: &str, capabilities: Option<HashMap<String, Vec<i32>>>) {
        let mut entries = self.entries.write().await;
        if let Some(entry) = entries.get_mut(model_id) {
            if let Some(ref mut info) = entry.info {
                info.capabilities = capabilities;
            }
            entry.state = ModelLoadState::Ready;
            entry.notify.notify_waiters();

            // Signal activation complete
            if let Some(tx) = entry.activation_tx.take() {
                let _ = tx.send(Ok(()));
            }
        }
    }

    /// Fail activation with error.
    async fn fail_activation(&self, model_id: &str, error: &str) {
        let mut entries = self.entries.write().await;
        if let Some(entry) = entries.get_mut(model_id) {
            entry.state = ModelLoadState::Failed;
            entry.error = Some(error.to_string());
            entry.notify.notify_waiters();

            // Signal activation failed
            if let Some(tx) = entry.activation_tx.take() {
                let _ = tx.send(Err(error.to_string()));
            }
        }
    }

    /// Schedule a model for loading.
    ///
    /// Returns the current state and canonical ID.
    pub async fn schedule_model(
        &self,
        requested_model_id: &str,
        force_reload: bool,
    ) -> Result<(ModelLoadState, String), String> {
        let resolved = {
            let mut resolver = self.resolver.lock().await;
            resolver
                .resolve(requested_model_id)
                .await
                .map_err(|e| e.to_string())?
        };

        let canonical_id = resolved.canonical_id.clone();

        {
            let mut alias_cache = self.alias_cache.write().await;
            alias_cache.insert(requested_model_id.to_lowercase(), canonical_id.clone());
            alias_cache
                .entry(canonical_id.to_lowercase())
                .or_insert_with(|| canonical_id.clone());
        }

        let mut entries = self.entries.write().await;
        let entry = entries
            .entry(canonical_id.clone())
            .or_insert_with(ModelEntry::default);

        if entry.state == ModelLoadState::Ready && !force_reload {
            return Ok((ModelLoadState::Ready, canonical_id));
        }

        if matches!(
            entry.state,
            ModelLoadState::Loading | ModelLoadState::Downloading | ModelLoadState::Activating
        ) && !force_reload
        {
            return Ok((entry.state, canonical_id));
        }

        if entry.state == ModelLoadState::Failed && !force_reload {
            return Ok((ModelLoadState::Failed, canonical_id));
        }

        entry.error = None;
        entry.info = None;
        entry.resolved = Some(resolved.clone());
        entry.bytes_downloaded = None;
        entry.bytes_total = None;
        entry.notify = Arc::new(Notify::new());

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

        // Model needs to be downloaded from HuggingFace
        entry.state = ModelLoadState::Downloading;
        let notify = entry.notify.clone();

        // Drop entries lock before spawning to avoid deadlock
        drop(entries);

        // Spawn download task
        let hf_repo = resolved.hf_repo.clone().unwrap_or_else(|| resolved.canonical_id.clone());
        let canonical_id_for_task = canonical_id.clone();
        let entries_ref = self.entries.clone();

        tokio::spawn(async move {
            let result = Self::download_model(&hf_repo).await;

            let mut entries: tokio::sync::RwLockWriteGuard<'_, HashMap<String, ModelEntry>> = entries_ref.write().await;
            if let Some(entry) = entries.get_mut(&canonical_id_for_task) {
                match result {
                    Ok(download_path) => {
                        // Update resolved path
                        if let Some(ref mut resolved) = entry.resolved {
                            resolved.model_path = download_path.clone();
                            resolved.source = "hf_cache".to_string();
                        }

                        // Create formatter
                        match ChatFormatter::new(&download_path) {
                            Ok(formatter) => {
                                entry.info = Some(ModelInfo {
                                    model_id: canonical_id_for_task.clone(),
                                    model_path: download_path.to_string_lossy().to_string(),
                                    formatter: Arc::new(formatter),
                                    capabilities: None,
                                });
                                entry.state = ModelLoadState::Loading;
                            }
                            Err(e) => {
                                entry.error = Some(format!("Failed to create formatter: {}", e));
                                entry.state = ModelLoadState::Failed;
                            }
                        }
                    }
                    Err(e) => {
                        entry.error = Some(format!("Download failed: {}", e));
                        entry.state = ModelLoadState::Failed;
                    }
                }
                notify.notify_waiters();
            }
        });

        Ok((ModelLoadState::Downloading, canonical_id))
    }

    /// Download a model from HuggingFace Hub.
    async fn download_model(repo_id: &str) -> Result<std::path::PathBuf, String> {
        log::info!("Downloading model from HuggingFace: {}", repo_id);

        let api = ApiBuilder::new()
            .build()
            .map_err(|e| format!("Failed to create HF API: {}", e))?;

        let repo = api.model(repo_id.to_string());

        // Download config.json first to verify it's a valid model
        repo.get("config.json")
            .await
            .map_err(|e| format!("Failed to download config.json: {}", e))?;

        // Get the cache directory where files are stored
        let cache_dir = repo.get(".")
            .await
            .map_err(|e| format!("Failed to access repo: {}", e))?;

        // The parent directory is the model directory
        let model_dir = cache_dir.parent()
            .ok_or_else(|| "Invalid cache path".to_string())?
            .to_path_buf();

        // Download common model files (if they exist)
        let files_to_download = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
        ];

        for file in &files_to_download {
            if let Err(e) = repo.get(file).await {
                log::debug!("Optional file {} not found: {}", file, e);
            }
        }

        log::info!("Model downloaded to {:?}", model_dir);
        Ok(model_dir)
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
            entry.error = Some(error.clone());
            entry.notify.notify_waiters();

            // Signal activation failed if waiting
            if let Some(tx) = entry.activation_tx.take() {
                let _ = tx.send(Err(error));
            }
        }
    }

    /// Handle model_loaded event from PIE.
    ///
    /// Called by the event callback when a model_loaded event is received.
    pub async fn handle_model_loaded(&self, payload: &Value) {
        let model_id = match payload.get("model_id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => {
                log::warn!("Received model_loaded event without model_id");
                return;
            }
        };

        // Extract and update capabilities
        if let Some(caps) = payload.get("capabilities").and_then(|c| c.as_object()) {
            let capabilities: HashMap<String, Vec<i32>> = caps.iter()
                .filter_map(|(k, v)| {
                    let vals: Vec<i32> = if let Some(arr) = v.as_array() {
                        arr.iter().filter_map(|x| x.as_i64().map(|n| n as i32)).collect()
                    } else if let Some(n) = v.as_i64() {
                        vec![n as i32]
                    } else {
                        return None;
                    };
                    Some((k.clone(), vals))
                })
                .collect();

            if !capabilities.is_empty() {
                self.update_capabilities(model_id, capabilities).await;
            }
        }

        // Canonicalize the model_id first (fix for issue #4)
        let canonical_id = match self.canonicalize(model_id).await {
            Ok(id) => id,
            Err(_) => {
                // Try the raw model_id as fallback
                log::debug!("Model '{}' not found in alias cache, using as-is", model_id);
                model_id.to_string()
            }
        };

        // Complete the activation
        self.complete_activation(&canonical_id, None).await;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_creation() {
        let registry = ModelRegistry::new().unwrap();
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
