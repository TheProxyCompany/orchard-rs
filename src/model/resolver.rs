//! Model resolution utilities.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use hf_hub::api::tokio::Api;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

const ALIASES: &[(&str, &str)] = &[("moondream3", "moondream/moondream3-preview")];

/// Result of resolving a model identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedModel {
    pub canonical_id: String,
    pub model_path: PathBuf,
    pub source: String,
    pub metadata: HashMap<String, String>,
    pub hf_repo: Option<String>,
}

/// Resolves model identifiers to local filesystem paths.
pub struct ModelResolver {
    resolved_cache: HashMap<String, ResolvedModel>,
    hf_api: Api,
}

impl ModelResolver {
    /// Create a new model resolver.
    pub fn new() -> Result<Self> {
        Ok(Self {
            resolved_cache: HashMap::new(),
            hf_api: Api::new().map_err(|e| Error::HfApiInit(e.to_string()))?,
        })
    }

    /// Resolve a model identifier to a local filesystem path.
    ///
    /// # Arguments
    /// * `requested_id` - Model identifier, which can be:
    ///   - Local path: `/path/to/model` or `./relative/path`
    ///   - HF repo ID: `meta-llama/Llama-3.1-8B-Instruct` (primary interface)
    ///   - Alias: `moondream3` (only for unambiguous models)
    pub async fn resolve(&mut self, requested_id: &str) -> Result<ResolvedModel> {
        let identifier = requested_id.trim();
        if identifier.is_empty() {
            return Err(Error::EmptyModelId);
        }

        // Check cache first
        let cache_key = identifier.to_lowercase();
        if let Some(cached) = self.resolved_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // 1. Try as local path
        if let Some(resolved) = self.try_local_path(identifier).await? {
            self.resolved_cache.insert(cache_key, resolved.clone());
            return Ok(resolved);
        }

        // 2. Check for known alias
        let hf_repo = ALIASES
            .iter()
            .find(|(alias, _)| alias.eq_ignore_ascii_case(identifier))
            .map(|(_, repo)| *repo)
            .unwrap_or(identifier);

        // 3. Resolve via HuggingFace
        let resolved = self
            .resolve_huggingface(hf_repo, if hf_repo != identifier { Some(identifier) } else { None })
            .await?;

        self.resolved_cache.insert(cache_key, resolved.clone());
        Ok(resolved)
    }

    /// Clear the resolution cache.
    pub fn clear_cache(&mut self) {
        self.resolved_cache.clear();
    }

    async fn try_local_path(&self, identifier: &str) -> Result<Option<ResolvedModel>> {
        let path = PathBuf::from(identifier);

        // Check absolute path
        if path.is_absolute() && path.is_dir() {
            return Ok(Some(self.build_resolved_model(path, "local", None, None).await?));
        }

        // Check relative path
        if path.is_dir() {
            let resolved = std::fs::canonicalize(&path)?;
            return Ok(Some(self.build_resolved_model(resolved, "local", None, None).await?));
        }

        Ok(None)
    }

    async fn resolve_huggingface(
        &self,
        repo_id: &str,
        requested_alias: Option<&str>,
    ) -> Result<ResolvedModel> {
        let repo = self.hf_api.model(repo_id.to_string());

        // Try to get from cache first, then download if needed
        let path = match repo.get("config.json").await {
            Ok(config_path) => config_path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| config_path),
            Err(e) => {
                return Err(Error::DownloadFailed(
                    repo_id.to_string(),
                    e.to_string(),
                ));
            }
        };

        let source = if path.to_string_lossy().contains("cache") {
            "hf_cache"
        } else {
            "hf_hub"
        };

        let canonical_id = requested_alias.unwrap_or(repo_id);
        self.build_resolved_model(path, source, Some(canonical_id), Some(repo_id))
            .await
    }

    async fn build_resolved_model(
        &self,
        model_path: PathBuf,
        source: &str,
        canonical_id: Option<&str>,
        hf_repo: Option<&str>,
    ) -> Result<ResolvedModel> {
        let model_path = if model_path.is_absolute() {
            model_path
        } else {
            std::fs::canonicalize(&model_path)?
        };

        // Load and parse config
        let config = self.load_config(&model_path)?;
        let metadata = Self::collect_metadata(&config);

        // Determine canonical ID
        let canonical_id = canonical_id
            .map(String::from)
            .or_else(|| Self::determine_canonical_id(&config, &model_path))
            .unwrap_or_else(|| {
                model_path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            });

        // Infer HF repo
        let hf_repo = hf_repo
            .map(String::from)
            .or_else(|| Self::infer_hf_repo(&config));

        Ok(ResolvedModel {
            canonical_id,
            model_path,
            source: source.to_string(),
            metadata,
            hf_repo,
        })
    }

    fn load_config(&self, model_dir: &Path) -> Result<serde_json::Value> {
        let config_file = model_dir.join("config.json");
        if !config_file.exists() {
            return Err(Error::MissingConfig(model_dir.to_path_buf()));
        }

        let content = std::fs::read_to_string(&config_file)?;
        serde_json::from_str(&content).map_err(Error::from)
    }

    fn determine_canonical_id(config: &serde_json::Value, model_dir: &Path) -> Option<String> {
        config
            .get("_name_or_path")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .or_else(|| {
                config
                    .get("model_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .or_else(|| {
                model_dir
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
            })
    }

    fn infer_hf_repo(config: &serde_json::Value) -> Option<String> {
        let candidate = config
            .get("_name_or_path")
            .or_else(|| config.get("original_repo"))
            .and_then(|v| v.as_str());

        candidate
            .filter(|s| s.contains('/') && !s.starts_with('/'))
            .map(String::from)
    }

    fn collect_metadata(config: &serde_json::Value) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        let keys = [
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "architecture",
        ];

        for key in keys {
            if let Some(value) = config.get(key) {
                let str_value = match value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        serde_json::to_string(value).unwrap_or_default()
                    }
                    serde_json::Value::Null => continue,
                };
                metadata.insert(key.to_string(), str_value);
            }
        }

        // Handle quantization config
        if let Some(quant_cfg) = config
            .get("quantization_config")
            .or_else(|| config.get("quantization"))
        {
            if let Some(bits) = quant_cfg
                .get("bits")
                .or_else(|| quant_cfg.get("num_bits"))
                .and_then(|v| v.as_u64())
            {
                metadata.insert("quantization_bits".to_string(), bits.to_string());
            }
        }

        metadata
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolver_creation() {
        let resolver = ModelResolver::new().unwrap();
        assert!(resolver.resolved_cache.is_empty());
    }

    #[test]
    fn test_collect_metadata() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "quantization_config": {
                "bits": 4
            }
        });

        let metadata = ModelResolver::collect_metadata(&config);
        assert_eq!(metadata.get("model_type"), Some(&"llama".to_string()));
        assert_eq!(metadata.get("hidden_size"), Some(&"4096".to_string()));
        assert_eq!(metadata.get("quantization_bits"), Some(&"4".to_string()));
    }

    #[test]
    fn test_infer_hf_repo() {
        let config = serde_json::json!({
            "_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"
        });

        let repo = ModelResolver::infer_hf_repo(&config);
        assert_eq!(repo, Some("meta-llama/Llama-3.1-8B-Instruct".to_string()));
    }

    #[test]
    fn test_infer_hf_repo_local_path() {
        let config = serde_json::json!({
            "_name_or_path": "/local/path/to/model"
        });

        let repo = ModelResolver::infer_hf_repo(&config);
        assert_eq!(repo, None);
    }
}
