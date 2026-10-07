//! Native Moshi/Mimi model ownership. We use Kyutai's pinned implementation,
//! not a text model masquerading as an audio endpoint.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor};
use hf_hub::{Cache, Repo, RepoType};
use serde::{Deserialize, Serialize};

use super::{language::LanguageModel, DuplexDevice, DuplexOptions};
use crate::{Error, Result};

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
    reference_encoder: Option<PathBuf>,
    reference_tokenizer: Option<PathBuf>,
    acoustic_delay: usize,
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
        model_id: model_id.into(),
        directory: paths[0].parent().unwrap_or(local).to_path_buf(),
        model: paths[0].clone(),
        codec: paths[1].clone(),
        tokenizer: paths[2].clone(),
        weight_bytes,
        reference_encoder,
        reference_tokenizer,
        acoustic_delay: checkpoint.and_then(|c| c.acoustic_delay).unwrap_or(2),
    })
}

pub(crate) struct MoshiModel {
    lm: LanguageModel,
    mimi: moshi::mimi::Mimi,
    tokenizer: Arc<sentencepiece::SentencePieceProcessor>,
    device: Device,
    mimi_device: Device,
    pub model_id: String,
    pub directory: PathBuf,
    pub weight_bytes: u64,
    active_sessions: AtomicUsize,
    active_references: AtomicUsize,
    pub acoustic_delay: usize,
    reference_encoder: Option<ReferenceEncoder>,
}

struct ReferenceActivity<'a>(&'a AtomicUsize);
impl Drop for ReferenceActivity<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

struct ReferenceEncoder {
    encoder: moshi_rag::conditioner::ArcEncoderConditioner,
    device: Device,
}
impl ReferenceEncoder {
    fn encode(&self, text: &str) -> Result<Tensor> {
        let tensor = self
            .encoder
            .condition(text, &self.device)
            .map_err(native_error)?;
        self.device.synchronize().map_err(native_error)?;
        // Only CPU tensors cross the worker boundary. Uploading or synchronizing
        // the voice device here could commit its in-progress Metal encoder.
        let tensor = tensor.to_device(&Device::Cpu).map_err(native_error)?;
        Ok(tensor)
    }
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
        let mut reference_encoder = None;
        let lm = match (&files.reference_encoder, &files.reference_tokenizer) {
            (Some(encoder), Some(tokenizer)) => {
                let dtype = if device.is_metal() {
                    DType::F16
                } else {
                    DType::F32
                };
                let tokenizer = tokenizer
                    .to_str()
                    .ok_or_else(|| Error::Other("ARC tokenizer path must be UTF-8".into()))?;
                let mut config = moshi_rag::lm::Config::v0_1_streaming_rag(8, tokenizer.to_owned());
                // The ARC encoder has a separate Metal queue and lifecycle from
                // the live voice decoder. Only the first-speaker LUT stays in
                // the voice model; prepared reference embeddings enter forward.
                let arc_config = match config
                    .conditioners
                    .as_mut()
                    .and_then(|c| c.remove("reference_with_time"))
                {
                    Some(moshi_rag::conditioner::ConditionerConfig::ArcEncoder(config)) => config,
                    _ => return Err(Error::Other("MoshiRAG has no ARC configuration".into())),
                };
                let voice_weights = super::quantization::q8_voice(&files.model)?;
                let model =
                    moshi_rag::lm::load_lm_model(config, &voice_weights, DType::F32, &device)
                        .map_err(native_error)?;
                if model.get_lut_condition("first_speaker", "user").is_none() {
                    return Err(Error::Other(
                        "MoshiRAG first-speaker conditioner missing".into(),
                    ));
                }
                let reference_device = if device.is_metal() {
                    Device::new_metal(0).map_err(native_error)?
                } else {
                    Device::Cpu
                };
                let vb = unsafe {
                    candle_nn::VarBuilder::from_mmaped_safetensors(
                        &[&files.model],
                        dtype,
                        &reference_device,
                    )
                }
                .map_err(native_error)?;
                let vb = moshi_rag::nn::MaybeQuantizedVarBuilder::Real(
                    vb.pp("condition_provider.conditioners.reference_with_time"),
                );
                let mut encoder_model =
                    moshi_rag::conditioner::ArcEncoderConditioner::new(4096, &arc_config, vb)
                        .map_err(native_error)?;
                encoder_model
                    .reload_from_hf(encoder, dtype, &reference_device)
                    .map_err(native_error)?;
                reference_encoder = Some(ReferenceEncoder {
                    encoder: encoder_model,
                    device: reference_device,
                });
                LanguageModel::Rag(model)
            }
            (None, None) => LanguageModel::Base(
                moshi::lm::load_streaming(&files.model, DType::F32, &device)
                    .map_err(native_error)?,
            ),
            _ => {
                return Err(Error::Other(
                    "Incomplete MoshiRAG conditioning assets".into(),
                ))
            }
        };
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
            active_references: AtomicUsize::new(0),
            acoustic_delay: files.acoustic_delay,
            reference_encoder,
        });
        // Compile the actual Metal kernels and exercise both codec directions
        // before reporting readiness. Otherwise the first microphone frames
        // arrive during a multi-second first-use compilation and are discarded.
        {
            if model.is_rag() {
                let reference =
                    model.encode_reference("The local assistant is preparing its tools.")?;
                model.device.synchronize().map_err(native_error)?;
                if reference.elem_count() == 0 {
                    return Err(Error::Other(
                        "MoshiRAG ARC warm-up returned no embeddings".into(),
                    ));
                }
            }
            super::codec_pool(DuplexOptions::default().codec_threads)?.install(
                || -> Result<()> {
                    let mut warmup = model.acquire(&DuplexOptions::default())?;
                    for _ in 0..6 {
                        warmup.process(&vec![0.0; super::FRAME_SAMPLES], Some(3))?;
                    }
                    Ok(())
                },
            )?;
        }
        Arc::try_unwrap(model)
            .map_err(|_| Error::Internal("Moshi warm-up retained an unexpected model owner".into()))
    }

    pub fn is_rag(&self) -> bool {
        self.lm.is_rag()
    }
    pub fn encode_reference(&self, text: &str) -> Result<Tensor> {
        self.active_references.fetch_add(1, Ordering::AcqRel);
        let _activity = ReferenceActivity(&self.active_references);
        self.reference_encoder
            .as_ref()
            .ok_or_else(|| {
                Error::Other("Choose MoshiRAG for trained reference conditioning".into())
            })?
            .encode(text)
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
            || self.active_references.load(Ordering::Acquire) != 0
    }
}

pub(crate) struct ModelSession {
    owner: Arc<MoshiModel>,
    state: super::generation::State,
    mimi: moshi::mimi::Mimi,
    previous_text: u32,
    previous_text_piece: u32,
}

pub(crate) struct ModelFrame {
    pub pcm: Option<Vec<f32>>,
    pub text: Option<String>,
    pub consumed_text: bool,
    pub retrieval_requested: bool,
}

impl ModelSession {
    fn new(owner: Arc<MoshiModel>, options: &DuplexOptions) -> Self {
        let mut lm = owner.lm.clone();
        lm.reset_state();
        let mut mimi = owner.mimi.clone();
        mimi.reset_state();
        let config = moshi::lm_generate_multistream::Config::v0_1();
        let previous_text = config.text_start_token;
        let state = super::generation::State::new(lm, options, owner.acoustic_delay);
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

    pub fn is_rag(&self) -> bool {
        self.owner.is_rag()
    }
    pub fn set_reference(&mut self, tensor: Tensor) -> Result<()> {
        // The quantized voice model operates in F32. This runs on its own actor,
        // never on the asynchronous ARC encoder thread.
        let tensor = tensor
            .to_dtype(DType::F32)
            .map_err(native_error)?
            .to_device(&self.owner.device)
            .map_err(native_error)?;
        self.state.set_reference(tensor);
        Ok(())
    }

    pub fn reference_remaining(&self) -> usize {
        self.state.reference_remaining()
    }
    pub fn clear_reference(&mut self) {
        self.state.clear_reference();
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
                consumed_text: false,
                retrieval_requested: false,
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
        let step = self
            .state
            .step(self.previous_text, &codes, forced_text)
            .map_err(native_error)?;
        let text_token = step.text_token;
        let config = self.state.config();
        let retrieval_requested = self.is_rag() && text_token == 4;
        let text = if retrieval_requested
            || [
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
        Ok(ModelFrame {
            pcm,
            text,
            consumed_text: step.consumed_text,
            retrieval_requested,
        })
    }
}

impl Drop for ModelSession {
    fn drop(&mut self) {
        self.owner.active_sessions.fetch_sub(1, Ordering::AcqRel);
    }
}
