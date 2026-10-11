//! Offline defaults and adapter capability hints. Live operation checks use
//! the activated descriptor; a catalog label never grants runtime authority.
use super::operations::ModelOperations;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{path::Path, sync::OnceLock};

#[derive(Clone, Debug, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub family: String,
    pub name: String,
    pub model: String,
    pub profile: String,
    pub capabilities: Value,
}

pub fn catalog_models() -> &'static [CatalogModel] {
    static MODELS: OnceLock<Vec<CatalogModel>> = OnceLock::new();
    MODELS.get_or_init(|| {
        serde_json::from_str(include_str!("catalog.json")).expect("bundled model catalog")
    })
}

fn profile_operations(model_type: &str, descriptor: &Value) -> Option<ModelOperations> {
    let embedded = crate::formatter::embedded_profiles::find_embedded_profile(model_type)?;
    let profile: Value = serde_yaml::from_str(embedded.capabilities).ok()?;
    let formatter = profile.get("duplex").is_none() && profile.get("diarization").is_none();
    Some(ModelOperations::from_profile(
        &profile, descriptor, formatter,
    ))
}

/// Works offline for installed descriptors and bundled defaults. Unknown
/// remote models remain unadvertised until their provider resolves a profile.
pub fn catalog_operations(identifier: &str) -> Option<ModelOperations> {
    let local = Path::new(identifier).join("config.json");
    let config = if local.is_file() {
        Some(local)
    } else {
        hf_hub::Cache::from_env()
            .model(identifier.into())
            .get("config.json")
    };
    if let Some(config) = config {
        let descriptor: Value = serde_json::from_slice(&std::fs::read(config).ok()?).ok()?;
        if let Some(model_type) = descriptor["model_type"].as_str() {
            if let Some(operations) = profile_operations(model_type, &descriptor) {
                return Some(operations);
            }
        }
    }
    let model = catalog_models()
        .iter()
        .find(|model| model.model == identifier || model.id == identifier)?;
    profile_operations(
        &model.profile,
        &json!({"rag":model.capabilities["speech_reference"] == true}),
    )
}

/// Compatibility projection for registry UIs. Operation names are canonical;
/// the older category flags remain readable by existing catalog consumers.
pub fn catalog_capabilities(identifier: &str) -> Option<Value> {
    let operations = catalog_operations(identifier)?;
    let mut value = json!({});
    for operation in &operations.operations {
        value[operation] = true.into();
    }
    for (operation, flags) in [
        ("text_generation", &["language"][..]),
        (
            "duplex",
            &[
                "voice",
                "full_duplex",
                "speech_to_speech",
                "audio_input",
                "audio_output",
                "media_only",
                "orchard_duplex",
            ][..],
        ),
        ("vision_query", &["vision", "orchard_vision"][..]),
        (
            "speaker_diarization",
            &[
                "diarization",
                "streaming",
                "audio_input",
                "media_only",
                "orchard_diarization",
            ][..],
        ),
        (
            "audio_transcription",
            &[
                "speech_to_text",
                "transcription",
                "media_only",
                "orchard_transcription",
            ][..],
        ),
    ] {
        if operations.supports(operation) {
            for flag in flags {
                value[*flag] = true.into();
            }
        }
    }
    if let Some(audio) = operations.audio {
        value["audio_geometry"] = serde_json::to_value(audio).ok()?;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn installed_alternate_profile_advertises_operations_without_an_identifier_allowlist() {
        let folder = tempfile::tempdir().unwrap();
        std::fs::write(
            folder.path().join("config.json"),
            r#"{"model_type":"gemma4"}"#,
        )
        .unwrap();
        let report = catalog_operations(folder.path().to_str().unwrap()).unwrap();
        report
            .require(&["text_generation", "tool_calling", "vision_query"])
            .unwrap();
        assert!(!report.supports("vision_point"));
        assert!(catalog_operations("not-installed/not-in-catalog").is_none());
    }
}
