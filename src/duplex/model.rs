//! Resolve pinned Moshi assets for PIE. No model tensors or device live in the SDK.

use hf_hub::{Cache, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::{Error, Result};
use sha2::{Digest, Sha256};

pub const RAG_MODEL: &str = "kyutai/moshika-rag-candle-bf16";
pub const DEFAULT_MODEL: &str = RAG_MODEL;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DuplexArchitecture {
    pub sample_rate: u32,
    pub frame_samples: usize,
    pub codebooks: usize,
    pub architecture_version: String,
    pub model_file: String,
    pub codec_file: String,
    pub tokenizer_file: String,
    pub checkpoints: HashMap<String, Checkpoint>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Checkpoint {
    pub revision: String,
    #[serde(default)]
    pub model_file: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub acoustic_delay: Option<usize>,
    #[serde(default)]
    pub reference_encoder: Option<ModelAsset>,
    #[serde(default)]
    pub reference_tokenizer: Option<ModelAsset>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelAsset {
    pub repo: String,
    pub revision: String,
    pub filename: String,
}

pub fn architecture() -> Result<DuplexArchitecture> {
    #[derive(Deserialize)]
    struct Profile {
        duplex: DuplexArchitecture,
    }
    let profile = crate::formatter::embedded_profiles::find_embedded_profile("moshi")
        .ok_or_else(|| Error::FormatterProfileNotFound("moshi".into()))?;
    let profile: Profile = serde_yaml::from_str(profile.capabilities)
        .map_err(|e| Error::Other(format!("Invalid Pantheon Moshi profile: {e}")))?;
    if profile.duplex.architecture_version != "v0_1"
        || profile.duplex.sample_rate != 24_000
        || profile.duplex.frame_samples != 1_920
        || profile.duplex.codebooks != 8
    {
        return Err(Error::Other(
            "Pantheon Moshi profile is incompatible with the native v0.1 architecture".into(),
        ));
    }
    Ok(profile.duplex)
}

pub fn is_moshi_model(model_id: &str) -> bool {
    if let Ok(bytes) = std::fs::read(Path::new(model_id).join("config.json")) {
        if serde_json::from_slice::<serde_json::Value>(&bytes)
            .is_ok_and(|config| config["model_type"] == "moshi")
        {
            return true;
        }
    }
    let Ok(profile) = architecture() else {
        return false;
    };
    if profile.checkpoints.contains_key(model_id) {
        return true;
    }
    let path = Path::new(model_id);
    path.is_dir()
        && [
            profile.model_file,
            profile.codec_file,
            profile.tokenizer_file,
        ]
        .iter()
        .all(|name| path.join(name).is_file())
}

/// Resolve a pinned remote checkpoint or an explicit native model descriptor.
/// Header/shape validation and tensor hydration are always performed by PIE.
pub(crate) async fn engine_source(model_id: &str) -> Result<(PathBuf, u64)> {
    let directory = Path::new(model_id);
    let config_path = directory.join("config.json");
    if config_path.is_file() {
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(config_path)?)?;
        if config["model_type"] != "moshi"
            || config["moshi_schema_version"] != 1
            || config["rag"] != true
        {
            return Err(Error::ModelNotFound(
                "Expected a schema-1 native MoshiRAG descriptor".into(),
            ));
        }
        let mut bytes = 0u64;
        for key in [
            "model_file",
            "codec_file",
            "tokenizer_file",
            "reference_encoder_file",
            "reference_tokenizer_file",
        ] {
            let path = config[key]
                .as_str()
                .ok_or_else(|| Error::ModelNotFound(format!("Moshi descriptor has no {key}")))?;
            let path = Path::new(path);
            if !path.is_absolute() {
                return Err(Error::ModelNotFound(format!(
                    "Moshi descriptor {key} must be absolute"
                )));
            }
            let metadata = std::fs::metadata(path)?;
            if !metadata.is_file() || metadata.len() == 0 {
                return Err(Error::ModelNotFound(format!(
                    "Moshi descriptor {key} is empty"
                )));
            }
            bytes = bytes
                .checked_add(metadata.len())
                .ok_or_else(|| Error::Other("Moshi asset size overflow".into()))?;
        }
        return Ok((directory.canonicalize()?, bytes));
    }
    let files = resolve_files(model_id).await?;
    Ok((files.engine_directory()?, files.weight_bytes))
}

#[derive(Debug)]
pub(crate) struct ModelFiles {
    model: PathBuf,
    codec: PathBuf,
    tokenizer: PathBuf,
    pub weight_bytes: u64,
    reference_encoder: Option<PathBuf>,
    reference_tokenizer: Option<PathBuf>,
    acoustic_delay: usize,
    rag: bool,
}

async fn asset(repo_id: &str, revision: &str, filename: &str) -> Result<PathBuf> {
    let root = Cache::from_env();
    let direct = root
        .path()
        .join(format!("models--{}", repo_id.replace('/', "--")))
        .join("snapshots")
        .join(revision)
        .join(filename);
    if direct.is_file() {
        return Ok(direct);
    }
    let api = hf_hub::api::tokio::ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .map_err(|e| Error::HfApiInit(e.to_string()))?;
    api.repo(Repo::with_revision(
        repo_id.to_owned(),
        RepoType::Model,
        revision.to_owned(),
    ))
    .get(filename)
    .await
    .map_err(|e| Error::DownloadFailed(repo_id.into(), format!("{filename}: {e}")))
}
pub(crate) async fn resolve_files(model_id: &str) -> Result<ModelFiles> {
    let profile = architecture()?;
    let local = Path::new(model_id);
    let checkpoint = profile.checkpoints.get(model_id);
    if checkpoint
        .is_none_or(|checkpoint| checkpoint.mode.as_deref() != Some("reference_conditioning"))
    {
        return Err(Error::ModelNotFound("Native PIE duplex currently supports the pinned MoshiRAG safetensors checkpoint; base/GGUF checkpoints require a matching native implementation".into()));
    }
    let model_file = checkpoint
        .and_then(|c| c.model_file.as_deref())
        .unwrap_or(&profile.model_file);
    let paths = if local.is_dir() {
        [
            model_file,
            profile.codec_file.as_str(),
            profile.tokenizer_file.as_str(),
        ]
        .iter()
        .map(|n| local.join(n))
        .collect::<Vec<_>>()
    } else {
        let checkpoint = checkpoint.ok_or_else(|| {
            Error::ModelNotFound(format!("No Pantheon Moshi checkpoint for {model_id}"))
        })?;
        let mut paths = Vec::new();
        for filename in [
            model_file,
            profile.codec_file.as_str(),
            profile.tokenizer_file.as_str(),
        ] {
            paths.push(asset(model_id, &checkpoint.revision, filename).await?);
        }
        paths
    };
    let mut weight_bytes = 0;
    for path in &paths {
        let metadata = std::fs::metadata(path)?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(Error::ModelNotFound(format!(
                "Empty Moshi asset {}",
                path.display()
            )));
        }
        weight_bytes += metadata.len();
    }
    let mut reference_encoder = None;
    let mut reference_tokenizer = None;
    if let Some(checkpoint) = checkpoint {
        if checkpoint.mode.as_deref() == Some("reference_conditioning") {
            let encoder = checkpoint
                .reference_encoder
                .as_ref()
                .ok_or_else(|| Error::Other("MoshiRAG profile has no ARC encoder".into()))?;
            let tokenizer = checkpoint
                .reference_tokenizer
                .as_ref()
                .ok_or_else(|| Error::Other("MoshiRAG profile has no ARC tokenizer".into()))?;
            let encoder_path = asset(&encoder.repo, &encoder.revision, &encoder.filename).await?;
            weight_bytes += std::fs::metadata(&encoder_path)?.len();
            reference_encoder = Some(encoder_path);
            reference_tokenizer =
                Some(asset(&tokenizer.repo, &tokenizer.revision, &tokenizer.filename).await?);
        }
    }
    Ok(ModelFiles {
        model: paths[0].clone(),
        codec: paths[1].clone(),
        tokenizer: paths[2].clone(),
        weight_bytes,
        reference_encoder,
        reference_tokenizer,
        acoustic_delay: checkpoint.and_then(|c| c.acoustic_delay).unwrap_or(2),
        rag: checkpoint.is_some_and(|c| c.mode.as_deref() == Some("reference_conditioning")),
    })
}

impl ModelFiles {
    /// A descriptor names immutable assets; PIE owns loading, residency and inference.
    pub fn engine_directory(&self) -> Result<PathBuf> {
        let absolute = |path: &Path| -> Result<String> {
            if !path.is_file() {
                return Err(Error::ModelNotFound(format!(
                    "Missing Moshi asset {}",
                    path.display()
                )));
            }
            // Preserve the logical .safetensors/.model filename: HF blobs are
            // extensionless, and both upstream and native loaders use format suffixes.
            let full = if path.is_absolute() {
                path.to_owned()
            } else {
                std::env::current_dir()?.join(path)
            };
            Ok(full.to_string_lossy().into_owned())
        };
        let mut descriptor = serde_json::json!({
            "model_type":"moshi", "architectures":[if self.rag {
                "MoshiRagForConditionalGeneration"
            } else { "MoshiForConditionalGeneration" }],
            "moshi_schema_version":1, "architecture_version":"v0_1",
            "rag":self.rag, "sample_rate":24_000, "frame_samples":1_920,
            "channels":1, "codebooks":8, "acoustic_delay":self.acoustic_delay,
            "model_file":absolute(&self.model)?, "codec_file":absolute(&self.codec)?,
            "tokenizer_file":absolute(&self.tokenizer)?,
        });
        if self.rag {
            let encoder = self.reference_encoder.as_deref().ok_or_else(|| {
                Error::ModelNotFound("MoshiRAG reference encoder is missing".into())
            })?;
            let tokenizer = self.reference_tokenizer.as_deref().ok_or_else(|| {
                Error::ModelNotFound("MoshiRAG reference tokenizer is missing".into())
            })?;
            descriptor["reference_encoder_file"] = absolute(encoder)?.into();
            descriptor["reference_tokenizer_file"] = absolute(tokenizer)?.into();
        }
        let bytes = serde_json::to_vec_pretty(&descriptor)
            .map_err(|error| Error::Other(format!("Moshi descriptor: {error}")))?;
        let key = format!("{:x}", Sha256::digest(&bytes));
        let directory = crate::EnginePaths::new()?
            .cache_dir
            .join("model-descriptors/moshi")
            .join(key);
        std::fs::create_dir_all(&directory)?;
        let target = directory.join("config.json");
        if target.is_file() {
            if std::fs::read(&target)? != bytes {
                return Err(Error::Other(
                    "Moshi descriptor content hash mismatch".into(),
                ));
            }
            return Ok(directory);
        }
        // A complete same-directory rename avoids exposing a partially written model config.
        let temporary = directory.join(format!(
            ".config-{}-{}.tmp",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::write(&temporary, &bytes)?;
        if let Err(error) = std::fs::rename(&temporary, &target) {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.into());
        }
        Ok(directory)
    }
}
