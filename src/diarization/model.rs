//! Pinned assets and immutable PIE descriptors. No native library or tensor is loaded here.
use super::{architecture, DiarizationOptions, DEFAULT_MODEL};
use crate::{Error, Result};
use hf_hub::{Cache, Repo, RepoType};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn properties(options: &DiarizationOptions) -> Value {
    json!({"device":options.device,"chunk_frames":options.chunk_frames,
        "right_context_frames":options.right_context_frames,"fifo_frames":options.fifo_frames,
        "speaker_cache_frames":options.speaker_cache_frames,"update_period_frames":options.update_period_frames})
}

pub(crate) fn runtime_key(model_id: &str, options: &DiarizationOptions) -> String {
    if properties(options) == properties(&DiarizationOptions::default()) {
        return model_id.to_owned();
    }
    let digest = Sha256::digest(properties(options).to_string().as_bytes());
    format!("{model_id}:diarization:{digest:x}")
}

pub(crate) async fn engine_source(
    model_id: &str,
    options: &DiarizationOptions,
) -> Result<(PathBuf, u64)> {
    options.validate()?;
    let profile = architecture()?;
    let local = Path::new(model_id);
    let config_path = local.join("config.json");
    if config_path.is_file() {
        let config: Value = serde_json::from_slice(&std::fs::read(config_path)?)?;
        if config["model_type"] != "nemotron3_diarization"
            || config["diarization_schema_version"] != 1
            || config["model_sha256"] != profile.sha256
            || config["model_size_bytes"] != profile.size_bytes
        {
            return Err(Error::ModelNotFound(
                "Expected the pinned schema-1 Nemotron3 descriptor".into(),
            ));
        }
        for (key, expected) in properties(options).as_object().expect("properties object") {
            if config[key] != *expected {
                return Err(Error::Other(format!(
                    "Diarization descriptor {key} does not match the requested options"
                )));
            }
        }
        let file = config["model_file"].as_str().ok_or_else(|| {
            Error::ModelNotFound("Diarization descriptor has no model_file".into())
        })?;
        verify_asset_path(Path::new(file), profile.size_bytes)?;
        return Ok((local.canonicalize()?, profile.size_bytes));
    }
    let file = if local.is_file() {
        local.to_owned()
    } else {
        if model_id != DEFAULT_MODEL {
            return Err(Error::ModelNotFound(model_id.into()));
        }
        let direct = Cache::from_env()
            .path()
            .join(format!("models--{}", model_id.replace('/', "--")))
            .join("snapshots")
            .join(&profile.revision)
            .join(&profile.model_file);
        if direct.is_file() {
            direct
        } else {
            let api = hf_hub::api::tokio::ApiBuilder::from_env()
                .with_progress(false)
                .build()
                .map_err(|error| Error::HfApiInit(error.to_string()))?;
            api.repo(Repo::with_revision(
                model_id.into(),
                RepoType::Model,
                profile.revision.clone(),
            ))
            .get(&profile.model_file)
            .await
            .map_err(|error| Error::DownloadFailed(model_id.into(), error.to_string()))?
        }
    };
    // Keep the logical .gguf path when an HF snapshot points at an extensionless blob.
    let file = if file.is_absolute() {
        file
    } else {
        std::env::current_dir()?.join(file)
    };
    verify_asset_path(&file, profile.size_bytes)?;
    let mut config = properties(options);
    let object = config.as_object_mut().expect("properties object");
    object.extend(
        json!({"model_type":"nemotron3_diarization","diarization_schema_version":1,
            "model_file":file.to_string_lossy(),"model_sha256":profile.sha256,
            "model_size_bytes":profile.size_bytes})
        .as_object()
        .expect("descriptor object")
        .clone(),
    );
    let bytes = serde_json::to_vec_pretty(&config)?;
    let key = format!("{:x}", Sha256::digest(&bytes));
    let directory = crate::EnginePaths::new()?
        .cache_dir
        .join("model-descriptors/nemotron3_diarization")
        .join(key);
    std::fs::create_dir_all(&directory)?;
    let target = directory.join("config.json");
    if target.exists() {
        if std::fs::read(&target)? != bytes {
            return Err(Error::Other(
                "Diarization descriptor content hash mismatch".into(),
            ));
        }
    } else {
        let temporary = directory.join(format!(
            ".config-{}-{}.tmp",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::write(&temporary, &bytes)?;
        if let Err(error) = std::fs::rename(&temporary, &target) {
            let _ = std::fs::remove_file(temporary);
            return Err(error.into());
        }
    }
    Ok((directory, profile.size_bytes))
}

fn verify_asset_path(path: &Path, expected_bytes: u64) -> Result<()> {
    if !path.is_absolute() {
        return Err(Error::ModelNotFound(
            "Diarization model_file must be absolute".into(),
        ));
    }
    let metadata = std::fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() != expected_bytes {
        return Err(Error::ModelNotFound(
            "Nemotron3 asset size differs from the pinned checkpoint".into(),
        ));
    }
    // The engine verifies SHA256 and GGUF geometry while hydrating its model.
    Ok(())
}
