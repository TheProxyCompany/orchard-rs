//! Model resolution utilities.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use hf_hub::api::tokio::{Api, ApiBuilder};
use hf_hub::Cache;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Result of resolving a model identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedModel {
    pub canonical_id: String,
    pub model_path: PathBuf,
    pub source: String,
    pub metadata: HashMap<String, String>,
    pub hf_repo: Option<String>,
    #[serde(default)]
    pub formatter_config: Option<serde_json::Value>,
}

/// Resolves model identifiers to local filesystem paths.
pub struct ModelResolver {
    resolved_cache: HashMap<String, ResolvedModel>,
    hf_api: Api,
    hf_cache: Cache,
}

#[derive(Debug, Deserialize)]
struct HubRepoFile {
    rfilename: String,
}

#[derive(Debug, Deserialize)]
struct HubRepoInfo {
    siblings: Vec<HubRepoFile>,
}

impl ModelResolver {
    /// Create a new model resolver.
    pub fn new() -> Result<Self> {
        let hf_cache = Cache::from_env();
        Ok(Self {
            resolved_cache: HashMap::new(),
            hf_api: ApiBuilder::from_env()
                .build()
                .map_err(|e| Error::HfApiInit(e.to_string()))?,
            hf_cache,
        })
    }

    #[cfg(test)]
    fn new_with_cache_for_tests(cache_dir: PathBuf) -> Result<Self> {
        let hf_cache = Cache::new(cache_dir);
        Ok(Self {
            resolved_cache: HashMap::new(),
            hf_api: ApiBuilder::from_cache(hf_cache.clone())
                .with_endpoint("http://127.0.0.1:9".to_string())
                .build()
                .map_err(|e| Error::HfApiInit(e.to_string()))?,
            hf_cache,
        })
    }

    /// Resolve a model identifier to a local filesystem path.
    ///
    /// # Arguments
    /// * `requested_id` - Model identifier, which can be:
    ///   - Local path: `/path/to/model` or `./relative/path`
    ///   - HF repo ID: `meta-llama/Llama-3.1-8B-Instruct`
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

        // 2. Resolve via HuggingFace
        let resolved = self.resolve_huggingface(identifier).await?;

        self.resolved_cache.insert(cache_key, resolved.clone());
        Ok(resolved)
    }

    /// Clear the resolution cache.
    pub fn clear_cache(&mut self) {
        self.resolved_cache.clear();
    }

    async fn try_local_path(&self, identifier: &str) -> Result<Option<ResolvedModel>> {
        let path = PathBuf::from(identifier);

        if path.is_absolute() && path.exists() {
            if path.is_dir()
                && (path.join("config.json").exists() || path.join("model_index.json").exists())
            {
                return Ok(Some(
                    self.build_resolved_model(path, "local", None, None).await?,
                ));
            }
            if path.is_dir() || path.is_file() {
                return Ok(Some(Self::build_local_source_model(path)));
            }
        }

        if path.exists() {
            let resolved = std::fs::canonicalize(&path)?;
            if resolved.is_dir()
                && (resolved.join("config.json").exists()
                    || resolved.join("model_index.json").exists())
            {
                return Ok(Some(
                    self.build_resolved_model(resolved, "local", None, None)
                        .await?,
                ));
            }
            if resolved.is_dir() || resolved.is_file() {
                return Ok(Some(Self::build_local_source_model(resolved)));
            }
        }

        Ok(None)
    }

    async fn resolve_huggingface(&self, repo_id: &str) -> Result<ResolvedModel> {
        if let Some(path) = self.resolve_cached_hf_snapshot_root(repo_id)? {
            return self
                .build_resolved_model(path, "hf_cache", Some(repo_id), Some(repo_id))
                .await;
        }

        let path = self.resolve_remote_hf_snapshot_root(repo_id).await?;

        let source = if path.to_string_lossy().contains("cache") {
            "hf_cache"
        } else {
            "hf_hub"
        };

        self.build_resolved_model(path, source, Some(repo_id), Some(repo_id))
            .await
    }

    fn resolve_cached_hf_snapshot_root(&self, repo_id: &str) -> Result<Option<PathBuf>> {
        let repo = self.hf_cache.model(repo_id.to_string());
        let root_file = repo
            .get("config.json")
            .or_else(|| repo.get("model_index.json"));
        let Some(root_file) = root_file else {
            return Ok(None);
        };
        let Some(root) = root_file.parent().map(Path::to_path_buf) else {
            return Ok(None);
        };

        if Self::is_hf_snapshot_complete(&root)? {
            Ok(Some(root))
        } else {
            Ok(None)
        }
    }

    async fn resolve_remote_hf_snapshot_root(&self, repo_id: &str) -> Result<PathBuf> {
        let repo = self.hf_api.model(repo_id.to_string());
        let mut repo_info: HubRepoInfo = repo
            .info_request()
            .query(&[("blobs", "true")])
            .send()
            .await
            .map_err(|e| Error::DownloadFailed(repo_id.to_string(), e.to_string()))?
            .json()
            .await
            .map_err(|e| Error::DownloadFailed(repo_id.to_string(), e.to_string()))?;

        repo_info
            .siblings
            .retain(|file| Self::should_download_hf_file(file.rfilename.as_str()));
        repo_info.siblings.sort_by(|left, right| {
            Self::hf_file_priority(left.rfilename.as_str())
                .cmp(&Self::hf_file_priority(right.rfilename.as_str()))
                .then_with(|| left.rfilename.cmp(&right.rfilename))
        });

        let mut config_path = None;
        let mut model_index_path = None;
        for file in repo_info.siblings {
            let path = repo
                .get(file.rfilename.as_str())
                .await
                .map_err(|e| Error::DownloadFailed(repo_id.to_string(), e.to_string()))?;
            match file.rfilename.as_str() {
                "config.json" => config_path = Some(path),
                "model_index.json" => model_index_path = Some(path),
                _ => {}
            }
        }

        let root_file = config_path.or(model_index_path).ok_or_else(|| {
            Error::DownloadFailed(
                repo_id.to_string(),
                "repository is missing config.json or model_index.json".to_string(),
            )
        })?;
        root_file
            .parent()
            .map(|path| path.to_path_buf())
            .ok_or_else(|| {
                Error::DownloadFailed(
                    repo_id.to_string(),
                    "resolved root has no parent".to_string(),
                )
            })
    }

    fn is_hf_snapshot_complete(model_dir: &Path) -> Result<bool> {
        let index_files = Self::find_files_with_suffix(model_dir, ".safetensors.index.json")?;
        if !index_files.is_empty() {
            for index_file in index_files {
                let content = std::fs::read_to_string(&index_file)?;
                let payload: serde_json::Value = serde_json::from_str(&content)?;
                let Some(weight_map) = payload
                    .get("weight_map")
                    .and_then(|value| value.as_object())
                else {
                    return Ok(false);
                };
                let mut shard_names = BTreeSet::new();
                for shard in weight_map.values().filter_map(|value| value.as_str()) {
                    shard_names.insert(shard.to_string());
                }
                if shard_names.is_empty() {
                    return Ok(false);
                }
                let Some(index_dir) = index_file.parent() else {
                    return Ok(false);
                };
                for shard_name in shard_names {
                    if !Self::is_materialized_weight_file(&index_dir.join(shard_name)) {
                        return Ok(false);
                    }
                }
            }
            return Ok(true);
        }

        let single_file = model_dir.join("model.safetensors");
        if single_file.exists() || single_file.is_symlink() {
            return Ok(Self::is_materialized_weight_file(&single_file));
        }

        let safetensors = Self::find_files_with_suffix(model_dir, ".safetensors")?;
        if safetensors.is_empty() {
            return Ok(false);
        }
        Ok(safetensors
            .iter()
            .all(|path| Self::is_materialized_weight_file(path)))
    }

    fn find_files_with_suffix(root: &Path, suffix: &str) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        Self::collect_files_with_suffix(root, suffix, &mut files)?;
        files.sort();
        Ok(files)
    }

    fn collect_files_with_suffix(
        root: &Path,
        suffix: &str,
        files: &mut Vec<PathBuf>,
    ) -> Result<()> {
        if !root.is_dir() {
            return Ok(());
        }
        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::collect_files_with_suffix(&path, suffix, files)?;
            } else if path.to_string_lossy().ends_with(suffix) {
                files.push(path);
            }
        }
        Ok(())
    }

    fn is_materialized_weight_file(path: &Path) -> bool {
        let candidate = if path.is_symlink() {
            match std::fs::canonicalize(path) {
                Ok(path) => path,
                Err(_) => return false,
            }
        } else {
            path.to_path_buf()
        };

        candidate.is_file()
            && !candidate
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".incomplete"))
            && candidate
                .metadata()
                .is_ok_and(|metadata| metadata.len() > 0)
    }

    fn should_download_hf_file(path: &str) -> bool {
        let file_name = path.rsplit('/').next().unwrap_or(path);
        path.ends_with(".json")
            || path.ends_with(".safetensors")
            || path.ends_with(".py")
            || path.ends_with(".tiktoken")
            || path.ends_with(".txt")
            || path.ends_with(".jsonl")
            || path.ends_with(".jinja")
            || file_name == "tokenizer.model"
            || file_name == "tiktoken.model"
    }

    fn hf_file_priority(path: &str) -> u8 {
        match path {
            "config.json" => 0,
            "model_index.json" => 1,
            _ => 2,
        }
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
        let config = Self::normalize_config(self.load_config(&model_path)?);
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
        let formatter_config = hf_repo.as_ref().map(|repo| {
            let mut config = config.clone();
            if config
                .get("_name_or_path")
                .and_then(|v| v.as_str())
                .is_none()
            {
                config["_name_or_path"] = serde_json::Value::String(repo.clone());
            }
            config
        });
        Ok(ResolvedModel {
            canonical_id,
            model_path,
            source: source.to_string(),
            metadata,
            hf_repo,
            formatter_config,
        })
    }

    fn build_local_source_model(model_path: PathBuf) -> ResolvedModel {
        let canonical_id = if model_path.is_file() {
            model_path
                .file_stem()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        } else {
            model_path
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        };

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "local_source".to_string());
        ResolvedModel {
            canonical_id,
            model_path,
            source: "local_source".to_string(),
            metadata,
            hf_repo: None,
            formatter_config: None,
        }
    }

    fn load_config(&self, model_dir: &Path) -> Result<serde_json::Value> {
        let config_file = model_dir.join("config.json");
        if !config_file.exists() {
            let model_index_file = model_dir.join("model_index.json");
            if model_index_file.exists() {
                let content = std::fs::read_to_string(&model_index_file)?;
                let mut model_index: serde_json::Value =
                    serde_json::from_str(&content).map_err(Error::from)?;
                let pipeline_class = model_index
                    .get("_class_name")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                let model_type = match pipeline_class {
                    "Ideogram4Pipeline" => Some("ideogram4"),
                    "QwenImageEditPipeline" => Some("qwen_image_edit"),
                    "FluxPipeline" | "Flux2KleinPipeline" => Some("flux"),
                    _ => None,
                };
                if let Some(model_type) = model_type {
                    if model_index.get("model_type").is_none() {
                        model_index["model_type"] =
                            serde_json::Value::String(model_type.to_string());
                    }
                    if model_index.get("source_format").is_none() {
                        model_index["source_format"] =
                            serde_json::Value::String("diffusers_directory".to_string());
                    }
                    return Ok(model_index);
                }
            }
            return Err(Error::MissingConfig(model_dir.to_path_buf()));
        }

        let content = std::fs::read_to_string(&config_file)?;
        serde_json::from_str(&content).map_err(Error::from)
    }

    fn normalize_config(mut config: serde_json::Value) -> serde_json::Value {
        if Self::is_parakeet_tdt_config(&config) {
            config["model_type"] = serde_json::Value::String("parakeet_tdt".to_string());
        }
        config
    }

    fn is_parakeet_tdt_config(config: &serde_json::Value) -> bool {
        let target_matches = config
            .get("target")
            .and_then(|value| value.as_str())
            .is_some_and(|target| target.ends_with("EncDecRNNTBPEModel"));
        let has_tdt_durations = config
            .get("model_defaults")
            .and_then(|defaults| defaults.get("tdt_durations"))
            .and_then(|durations| durations.as_array())
            .is_some();
        target_matches && has_tdt_durations
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
    use hf_hub::Cache;
    use tempfile::tempdir;

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

    #[tokio::test]
    async fn test_resolves_local_file_as_engine_inspected_source() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"GGUF").unwrap();

        let mut resolver = ModelResolver::new().unwrap();
        let resolved = resolver
            .resolve(model_path.to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(resolved.source, "local_source");
        assert_eq!(resolved.canonical_id, "model");
        assert_eq!(resolved.model_path, model_path);
        assert!(resolved.formatter_config.is_none());
    }

    #[tokio::test]
    async fn test_hf_resolved_model_passes_repo_hint_to_formatter_config() {
        let repo_id = "meta-llama/Llama-3.1-8B-Instruct";
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::json!({"model_type": "llama"}).to_string(),
        )
        .unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();

        let resolver = ModelResolver::new().unwrap();
        let resolved = resolver
            .build_resolved_model(
                dir.path().to_path_buf(),
                "hf_cache",
                Some(repo_id),
                Some(repo_id),
            )
            .await
            .unwrap();
        let config = resolved.formatter_config.as_ref().unwrap();

        assert_eq!(config["model_type"], "llama");
        assert_eq!(config["_name_or_path"], repo_id);
    }

    #[tokio::test]
    async fn test_resolves_cached_hf_snapshot_without_remote_metadata() {
        let repo_id = "meta-llama/Llama-3.1-8B-Instruct";
        let cache_dir = tempdir().unwrap();
        let cache = Cache::new(cache_dir.path().to_path_buf());
        let cache_repo = cache.model(repo_id.to_string());
        let commit = "0123456789abcdef0123456789abcdef01234567";
        cache_repo.create_ref(commit).unwrap();
        let snapshot = cache_repo.pointer_path(commit);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(
            snapshot.join("config.json"),
            serde_json::json!({"model_type": "llama"}).to_string(),
        )
        .unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(snapshot.join("model.safetensors"), b"weights").unwrap();

        let mut resolver =
            ModelResolver::new_with_cache_for_tests(cache_dir.path().to_path_buf()).unwrap();
        let resolved = resolver.resolve(repo_id).await.unwrap();

        assert_eq!(resolved.source, "hf_cache");
        assert_eq!(resolved.canonical_id, repo_id);
        assert_eq!(resolved.model_path, snapshot);
        assert_eq!(
            resolved.formatter_config.as_ref().unwrap()["_name_or_path"],
            repo_id
        );
    }

    #[tokio::test]
    async fn test_resolves_local_ideogram_model_index_directory() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("model_index.json"),
            serde_json::json!({"_class_name": "Ideogram4Pipeline"}).to_string(),
        )
        .unwrap();

        let mut resolver = ModelResolver::new().unwrap();
        let resolved = resolver
            .resolve(dir.path().to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(resolved.source, "local");
        assert_eq!(
            resolved.metadata.get("model_type"),
            Some(&"ideogram4".to_string())
        );
        assert_eq!(resolved.formatter_config, None);
    }

    #[tokio::test]
    async fn test_resolves_local_qwen_image_edit_model_index_directory() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("model_index.json"),
            serde_json::json!({"_class_name": "QwenImageEditPipeline"}).to_string(),
        )
        .unwrap();

        let mut resolver = ModelResolver::new().unwrap();
        let resolved = resolver
            .resolve(dir.path().to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(resolved.source, "local");
        assert_eq!(
            resolved.metadata.get("model_type"),
            Some(&"qwen_image_edit".to_string())
        );
        assert_eq!(resolved.formatter_config, None);
    }

    #[tokio::test]
    async fn test_resolves_local_flux2_model_index_directory() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("model_index.json"),
            serde_json::json!({"_class_name": "Flux2KleinPipeline"}).to_string(),
        )
        .unwrap();

        let mut resolver = ModelResolver::new().unwrap();
        let resolved = resolver
            .resolve(dir.path().to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(resolved.source, "local");
        assert_eq!(
            resolved.metadata.get("model_type"),
            Some(&"flux".to_string())
        );
        assert_eq!(resolved.formatter_config, None);
    }

    #[tokio::test]
    async fn test_resolves_parakeet_tdt_config_with_audio_profile() {
        let repo_id = "mlx-community/parakeet-tdt-0.6b-v3";
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::json!({
                "_name_or_path": repo_id,
                "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
                "model_defaults": {
                    "tdt_durations": [0, 1, 2, 3, 4]
                }
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();

        let resolver = ModelResolver::new().unwrap();
        let resolved = resolver
            .build_resolved_model(
                dir.path().to_path_buf(),
                "hf_cache",
                Some(repo_id),
                Some(repo_id),
            )
            .await
            .unwrap();
        let config = resolved.formatter_config.as_ref().unwrap();

        assert_eq!(
            resolved.metadata.get("model_type"),
            Some(&"parakeet_tdt".to_string())
        );
        assert_eq!(config["model_type"], "parakeet_tdt");
        assert!(config.get("template_type").is_none());
        assert_eq!(config["_name_or_path"], repo_id);
    }
}
