//! Native Moshi/Mimi model ownership. We use Kyutai's pinned implementation,
//! not a text model masquerading as an audio endpoint.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{Cache, Repo, RepoType};
use serde::{Deserialize, Serialize};

use super::{DuplexDevice, DuplexOptions};
use crate::{Error, Result};

pub const DEFAULT_MODEL: &str = "kyutai/moshiko-candle-q8";

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

#[derive(Debug)]
pub(crate) struct ModelFiles {
    pub model_id: String,
    pub directory: PathBuf,
    model: PathBuf,
    codec: PathBuf,
    tokenizer: PathBuf,
    pub weight_bytes: u64,
}

pub(crate) async fn resolve_files(model_id: &str) -> Result<ModelFiles> {
    let profile = architecture()?;
    let local = Path::new(model_id);
    let names = [
        &profile.model_file,
        &profile.codec_file,
        &profile.tokenizer_file,
    ];
    let paths = if local.is_dir() {
        names
            .iter()
            .map(|name| local.join(name))
            .collect::<Vec<_>>()
    } else {
        let checkpoint = profile.checkpoints.get(model_id).ok_or_else(|| {
            Error::ModelNotFound(format!("No Pantheon Moshi checkpoint for {model_id}"))
        })?;
        let repo = Repo::with_revision(
            model_id.to_owned(),
            RepoType::Model,
            checkpoint.revision.clone(),
        );
        let cache_root = Cache::from_env();
        let snapshot = cache_root
            .path()
            .join(format!("models--{}", model_id.replace('/', "--")))
            .join("snapshots")
            .join(&checkpoint.revision);
        let cache = cache_root.repo(repo.clone());
        let api = hf_hub::api::tokio::ApiBuilder::from_env()
            .with_progress(false)
            .build()
            .map_err(|e| Error::HfApiInit(e.to_string()))?;
        let api_repo = api.repo(repo);
        let mut paths = Vec::new();
        for name in names {
            // Python's hub cache omits refs/<commit> for an immutable revision;
            // hf-hub 0.5's CacheRepo::get only consults refs and would download an
            // already cached multi-GB model again. Check the exact snapshot first.
            let pinned = snapshot.join(name);
            paths.push(
                match pinned
                    .is_file()
                    .then_some(pinned)
                    .or_else(|| cache.get(name))
                {
                    Some(path) => path,
                    None => api_repo.get(name).await.map_err(|e| {
                        Error::DownloadFailed(model_id.into(), format!("{name}: {e}"))
                    })?,
                },
            );
        }
        paths
    };
    let mut bytes = 0;
    for path in &paths {
        let metadata = std::fs::metadata(path)
            .map_err(|e| Error::ModelNotFound(format!("Moshi asset {}: {e}", path.display())))?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(Error::ModelNotFound(format!(
                "Moshi asset {} is empty or not a file",
                path.display()
            )));
        }
        bytes += metadata.len();
    }
    Ok(ModelFiles {
        model_id: model_id.to_owned(),
        directory: paths[0].parent().unwrap_or(local).to_path_buf(),
        model: paths[0].clone(),
        codec: paths[1].clone(),
        tokenizer: paths[2].clone(),
        weight_bytes: bytes,
    })
}

pub(crate) struct MoshiModel {
    lm: moshi::lm::LmModel,
    mimi: moshi::mimi::Mimi,
    tokenizer: Arc<sentencepiece::SentencePieceProcessor>,
    device: Device,
    mimi_device: Device,
    pub model_id: String,
    pub directory: PathBuf,
    pub weight_bytes: u64,
    active_sessions: AtomicUsize,
}

fn native_error(error: impl std::fmt::Display) -> Error {
    Error::Other(format!("Moshi: {error}"))
}

impl MoshiModel {
    pub fn load(files: ModelFiles, device: DuplexDevice, mimi_cpu: bool) -> Result<Self> {
        let device = match device {
            DuplexDevice::Cpu => Device::Cpu,
            DuplexDevice::Metal => Device::new_metal(0).map_err(native_error)?,
            DuplexDevice::Auto if candle_core::utils::metal_is_available() => {
                Device::new_metal(0).map_err(native_error)?
            }
            DuplexDevice::Auto => Device::Cpu,
        };
        // This mirrors the official Candle backend: F32 on Metal/CPU, quantized
        // GGUF for Moshi and F32 Mimi. No MLX/Candle tensor conversion is involved.
        let lm =
            moshi::lm::load_streaming(&files.model, DType::F32, &device).map_err(native_error)?;
        let mimi_device = if mimi_cpu {
            Device::Cpu
        } else {
            device.clone()
        };
        let mimi = moshi::mimi::load(
            files
                .codec
                .to_str()
                .ok_or_else(|| Error::Other("Mimi path must be valid UTF-8".into()))?,
            Some(8),
            &mimi_device,
        )
        .map_err(native_error)?;
        let tokenizer =
            sentencepiece::SentencePieceProcessor::open(&files.tokenizer).map_err(native_error)?;
        let model = Arc::new(Self {
            lm,
            mimi,
            tokenizer: Arc::new(tokenizer),
            device,
            mimi_device,
            model_id: files.model_id,
            directory: files.directory,
            weight_bytes: files.weight_bytes,
            active_sessions: AtomicUsize::new(0),
        });
        // Compile the actual Metal kernels and exercise both codec directions
        // before reporting readiness. Otherwise the first microphone frames
        // arrive during a multi-second first-use compilation and are discarded.
        {
            let mut warmup = model.acquire(&DuplexOptions::default())?;
            for _ in 0..6 {
                warmup.process(&vec![0.0; super::FRAME_SAMPLES], Some(3))?;
            }
        }
        Arc::try_unwrap(model)
            .map_err(|_| Error::Internal("Moshi warm-up retained an unexpected model owner".into()))
    }

    pub fn acquire(self: &Arc<Self>, options: &DuplexOptions) -> Result<ModelSession> {
        // One real-time stream per loaded backend. More streams need measured
        // admission, not silently multiplying the GPU load and audio latency.
        self.active_sessions
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                Error::ModelNotReady("Moshi already has an active duplex session".into())
            })?;
        Ok(ModelSession::new(Arc::clone(self), options))
    }

    pub fn is_busy(&self) -> bool {
        self.active_sessions.load(Ordering::Acquire) != 0
    }
}

pub(crate) struct ModelSession {
    owner: Arc<MoshiModel>,
    state: moshi::lm_generate_multistream::State,
    mimi: moshi::mimi::Mimi,
    previous_text: u32,
    previous_text_piece: u32,
}

pub(crate) struct ModelFrame {
    pub pcm: Option<Vec<f32>>,
    pub text: Option<String>,
}

impl ModelSession {
    fn new(owner: Arc<MoshiModel>, options: &DuplexOptions) -> Self {
        let mut lm = owner.lm.clone();
        lm.reset_state();
        let mut mimi = owner.mimi.clone();
        mimi.reset_state();
        let config = moshi::lm_generate_multistream::Config::v0_1();
        let previous_text = config.text_start_token;
        let sampling = |temperature: f64, k: usize| {
            if temperature <= 0.0 {
                Sampling::ArgMax
            } else {
                Sampling::TopK { k, temperature }
            }
        };
        let state = moshi::lm_generate_multistream::State::new(
            lm,
            options.max_steps,
            LogitsProcessor::from_sampling(
                options.seed,
                sampling(options.audio_temperature, options.audio_top_k),
            ),
            LogitsProcessor::from_sampling(
                options.seed.wrapping_add(1),
                sampling(options.text_temperature, options.text_top_k),
            ),
            None,
            None,
            None,
            config,
        );
        Self {
            owner,
            state,
            mimi,
            previous_text,
            previous_text_piece: previous_text,
        }
    }

    pub fn reset(&mut self, options: &DuplexOptions) {
        // The replacement takes over this session's admission. Dropping the
        // old codec/KV state balances this temporary increment normally.
        self.owner.active_sessions.fetch_add(1, Ordering::AcqRel);
        *self = Self::new(Arc::clone(&self.owner), options);
    }

    pub fn step_idx(&self) -> usize {
        self.state.step_idx()
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.owner
            .tokenizer
            .encode(text)
            .map(|pieces| pieces.into_iter().map(|piece| piece.id).collect())
            .map_err(native_error)
    }

    pub fn process(&mut self, pcm: &[f32], forced_text: Option<u32>) -> Result<ModelFrame> {
        let input = Tensor::from_slice(pcm, (1, 1, pcm.len()), &self.owner.mimi_device)
            .map_err(native_error)?;
        let encoded = self
            .mimi
            .encode_step(&input.into(), &().into())
            .map_err(native_error)?;
        let Some(encoded) = encoded.as_option() else {
            return Ok(ModelFrame {
                pcm: None,
                text: None,
            });
        };
        let (_, codebooks, frames) = encoded.dims3().map_err(native_error)?;
        if codebooks != 8 || frames != 1 {
            return Err(Error::Other(format!(
                "Mimi emitted {codebooks} codebooks and {frames} frames for one 80ms input"
            )));
        }
        let codes = encoded
            .i((0, .., 0))
            .and_then(|v| v.to_vec1::<u32>())
            .map_err(native_error)?;
        let text_token = self
            .state
            .step(self.previous_text, &codes, forced_text, None)
            .map_err(native_error)?;
        let config = self.state.config();
        let text = if [
            config.text_pad_token,
            config.text_eop_token,
            config.text_start_token,
        ]
        .contains(&text_token)
        {
            None
        } else {
            // Decode the growing piece pair as upstream does, preserving spaces
            // between SentencePiece words without emitting special tokens.
            if self.previous_text_piece == config.text_start_token {
                Some(
                    self.owner
                        .tokenizer
                        .decode_piece_ids(&[text_token])
                        .map_err(native_error)?,
                )
            } else {
                let current = self
                    .owner
                    .tokenizer
                    .decode_piece_ids(&[self.previous_text_piece, text_token])
                    .map_err(native_error)?;
                let previous = self
                    .owner
                    .tokenizer
                    .decode_piece_ids(&[self.previous_text_piece])
                    .map_err(native_error)?;
                Some(
                    current
                        .strip_prefix(&previous)
                        .unwrap_or(&current)
                        .to_owned(),
                )
            }
        };
        if text.is_some() {
            self.previous_text_piece = text_token;
        }
        self.previous_text = text_token;
        let pcm = if let Some(tokens) = self.state.last_audio_tokens() {
            let tokens = Tensor::from_slice(&tokens[..8], (1, 8, 1), &self.owner.mimi_device)
                .map_err(native_error)?;
            let decoded = self
                .mimi
                .decode_step(&tokens.into(), &().into())
                .map_err(native_error)?;
            match decoded.as_option() {
                Some(decoded) => Some(
                    decoded
                        .i((0, 0))
                        .and_then(|v| v.to_vec1::<f32>())
                        .map_err(native_error)?,
                ),
                None => None,
            }
        } else {
            None
        };
        self.owner.device.synchronize().map_err(native_error)?;
        Ok(ModelFrame { pcm, text })
    }
}

impl Drop for ModelSession {
    fn drop(&mut self) {
        self.owner.active_sessions.fetch_sub(1, Ordering::AcqRel);
    }
}
