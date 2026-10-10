//! Operation and residency contracts shared by hosts and native adapters.
//! A model's profile describes its wire geometry; its name is not a capability.
use crate::{Error, ModelInfo, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioGeometry {
    pub sample_rate: u32,
    pub channels: u32,
    pub frame_samples: Option<usize>,
    pub encoding: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelOperations {
    pub operations: BTreeSet<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioGeometry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_speakers: Option<usize>,
    /// Some profiles expose caption/point tasks in addition to image chat.
    #[serde(default)]
    pub vision_tasks: bool,
}

impl ModelOperations {
    pub fn supports(&self, operation: &str) -> bool {
        self.operations.contains(operation)
    }

    pub fn require(&self, operations: &[&str]) -> Result<()> {
        for operation in operations {
            if !self.supports(operation) {
                return Err(Error::Other(format!(
                    "Selected model does not support {operation}"
                )));
            }
        }
        Ok(())
    }

    /// Normalize the installed profile without model-family comparisons in a
    /// host. Native session Ready events remain authoritative for live controls.
    pub fn from_profile(profile: &Value, descriptor: &Value, has_formatter: bool) -> Self {
        let mut result = Self::default();
        let native = |name: &str| profile[name]["native"] == true;
        let mut add = |operation: &str| {
            result.operations.insert(operation.into());
        };
        if has_formatter
            && !native("speech_to_text")
            && profile.get("duplex").is_none()
            && profile.get("diarization").is_none()
        {
            add("text_generation");
            if profile["tool_calling"]["formats"]
                .as_array()
                .is_some_and(|formats| !formats.is_empty())
            {
                add("tool_calling");
            }
        }
        if native("vision") {
            add("vision_query");
        }
        if native("pointing") {
            add("vision_point");
        }
        if native("captioning") {
            add("vision_caption");
            result.vision_tasks = true;
        }
        if native("speech_to_text") {
            add("audio_transcription");
            if let Some(rate) = profile["speech_to_text"]["input"]["sample_rate"].as_u64() {
                result.audio = Some(AudioGeometry {
                    sample_rate: rate as u32,
                    channels: 1,
                    frame_samples: None,
                    encoding: "pcm_f32le".into(),
                });
            }
        }
        if let Some(duplex) = profile.get("duplex") {
            add("duplex");
            add("forced_text");
            if descriptor["rag"] == true {
                add("speech_reference");
            }
            if let (Some(rate), Some(frame)) = (
                duplex["sample_rate"].as_u64(),
                duplex["frame_samples"].as_u64(),
            ) {
                result.audio = Some(AudioGeometry {
                    sample_rate: rate as u32,
                    channels: 1,
                    frame_samples: Some(frame as usize),
                    encoding: "pcm_f32le".into(),
                });
            }
        }
        if let Some(diarization) = profile.get("diarization") {
            add("speaker_diarization");
            // The native streaming API accepts/resamples the input rate in the
            // request. Its current protocol consumes 80 ms mono PCM frames.
            result.audio = Some(AudioGeometry {
                sample_rate: 24_000,
                channels: 1,
                frame_samples: Some(1_920),
                encoding: "pcm_f32le".into(),
            });
            result.max_speakers = diarization["speakers"].as_u64().map(|value| value as usize);
        }
        result
    }

    pub fn for_loaded(info: &ModelInfo) -> Result<Self> {
        let descriptor: Value = serde_json::from_slice(&std::fs::read(
            std::path::Path::new(&info.model_path).join("config.json"),
        )?)?;
        let profile = if let Some(formatter) = &info.formatter {
            formatter.operation_profile().clone()
        } else {
            let model_type = descriptor["model_type"]
                .as_str()
                .ok_or_else(|| Error::Other("Model descriptor has no model_type".into()))?;
            let embedded = crate::formatter::embedded_profiles::find_embedded_profile(model_type)
                .ok_or_else(|| Error::FormatterProfileNotFound(model_type.into()))?;
            serde_yaml::from_str(embedded.capabilities)
                .map_err(|error| Error::Other(format!("Invalid operation profile: {error}")))?
        };
        Ok(Self::from_profile(
            &profile,
            &descriptor,
            info.formatter.is_some(),
        ))
    }
}

/// A preparation request carries exactly the immutable options the eventual
/// session uses. In particular, speaker device/chunk options must not be lost.
#[derive(Clone, Debug)]
pub struct ModelLoadRequest {
    pub model: String,
    pub options: ModelLoadOptions,
}

#[derive(Clone, Debug, Default)]
pub enum ModelLoadOptions {
    #[default]
    Automatic,
    #[cfg(feature = "duplex")]
    Duplex(crate::duplex::DuplexOptions),
    #[cfg(feature = "diarization")]
    Diarization(crate::diarization::DiarizationOptions),
}

impl ModelLoadRequest {
    pub fn automatic(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            options: ModelLoadOptions::Automatic,
        }
    }
}

/// Specialized source preparation lives at the adapter boundary. The registry
/// and every host can schedule the resulting descriptor through one path.
pub(crate) fn source_options(model: &str) -> ModelLoadOptions {
    #[cfg(feature = "diarization")]
    if crate::diarization::is_diarization_model(model) {
        return ModelLoadOptions::Diarization(Default::default());
    }
    #[cfg(feature = "duplex")]
    if crate::duplex::is_moshi_model(model) {
        return ModelLoadOptions::Duplex(Default::default());
    }
    let _ = model;
    ModelLoadOptions::Automatic
}
