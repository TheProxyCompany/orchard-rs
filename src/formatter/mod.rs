//! Chat formatting for LLM prompts.

pub mod control_tokens;
pub mod multimodal;

use std::collections::HashMap;
use std::path::Path;

use minijinja::{context, Environment, Value};

pub use control_tokens::{ControlTokens, Role, RoleTags};
pub use multimodal::{build_multimodal_layout, build_multimodal_messages, CapabilityInput};

use crate::error::{Error, Result};

/// Determine model type from config.
fn determine_model_type(config: &serde_json::Value) -> &str {
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    match model_type {
        "llama" | "llama3" => "llama3",
        "moondream3" | "moondream" => "moondream3",
        "gemma3" | "gemma" => "gemma3",
        "qwen2" | "qwen" => "qwen2",
        other => other,
    }
}

/// Chat formatter using Jinja2-compatible templates.
///
/// Handles the application of chat templates to conversation histories.
#[derive(Clone)]
pub struct ChatFormatter {
    /// Model path
    model_path: String,
    /// Model type (e.g., "llama3", "moondream3")
    model_type: String,
    /// Control tokens
    pub control_tokens: ControlTokens,
    /// Compiled template
    template_source: String,
}

impl ChatFormatter {
    /// Create a new chat formatter for a model.
    pub fn new(model_path: &Path) -> Result<Self> {
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(Error::FormatterConfigNotFound(
                model_path.to_string_lossy().to_string(),
            ));
        }

        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

        let model_type = determine_model_type(&config).to_string();
        let profile_dir = Self::find_profile_dir(&model_type)?;
        let control_tokens = ControlTokens::load(&profile_dir)?;

        let template_path = profile_dir.join("chat_template.jinja");
        let template_source = std::fs::read_to_string(&template_path).map_err(|_| {
            Error::Template(format!("Failed to load template from {:?}", template_path))
        })?;

        Ok(Self {
            model_path: model_path.to_string_lossy().to_string(),
            model_type,
            control_tokens,
            template_source,
        })
    }

    /// Whether to clip the image placeholder from prompt text.
    ///
    /// If the model doesn't have a start image token, we need to clip
    /// the placeholder from the rendered text.
    pub fn should_clip_image_placeholder(&self) -> bool {
        self.control_tokens.start_image_token.is_none()
            || self
                .control_tokens
                .start_image_token
                .as_ref()
                .map(|s| s.is_empty())
                .unwrap_or(true)
    }

    /// The default image placeholder token.
    pub fn default_image_placeholder(&self) -> &str {
        "<|image|>"
    }

    /// Apply the chat template to a conversation.
    ///
    /// # Arguments
    /// * `conversation` - List of message dictionaries with "role" and "content"
    /// * `add_generation_prompt` - Whether to add the assistant prompt turn
    /// * `reasoning` - Whether to add reasoning tokens
    /// * `task` - Optional task name for task-specific formatting
    pub fn apply_template(
        &self,
        conversation: &[HashMap<String, serde_json::Value>],
        add_generation_prompt: bool,
        reasoning: bool,
        task: Option<&str>,
    ) -> Result<String> {
        let mut env = Environment::new();
        env.add_template("chat", &self.template_source)
            .map_err(|e| Error::Template(e.to_string()))?;

        let template = env
            .get_template("chat")
            .map_err(|e| Error::Template(e.to_string()))?;

        let interactions: Vec<Value> = conversation
            .iter()
            .map(|msg| {
                let role = msg
                    .get("role")
                    .and_then(|v| v.as_str())
                    .unwrap_or("user");
                let content = msg.get("content").cloned().unwrap_or(serde_json::Value::Null);
                let content_value = Self::json_to_minijinja(&content);

                let map: std::collections::BTreeMap<String, Value> = [
                    ("role".to_string(), Value::from(role)),
                    ("content".to_string(), content_value),
                ]
                .into_iter()
                .collect();
                Value::from_object(map)
            })
            .collect();

        let roles = self.control_tokens.roles.to_value();

        let ctx = context! {
            interactions => interactions,
            add_generation_prompt => add_generation_prompt,
            begin_of_text => self.control_tokens.begin_of_text,
            end_of_sequence => self.control_tokens.end_of_sequence,
            end_of_message => self.control_tokens.end_of_message,
            start_image_token => self.control_tokens.start_image_token,
            end_image_token => self.control_tokens.end_image_token,
            thinking_start_token => self.control_tokens.thinking_start_token,
            thinking_end_token => self.control_tokens.thinking_end_token,
            reasoning => reasoning,
            task => task,
            roles => roles,
        };

        template
            .render(ctx)
            .map_err(|e| Error::Template(e.to_string()))
    }

    fn find_profile_dir(model_type: &str) -> Result<std::path::PathBuf> {
        let candidates = [
            Some(
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("profiles")
                    .join(model_type),
            ),
            std::env::current_dir()
                .ok()
                .map(|p| p.join("profiles").join(model_type)),
        ];

        for candidate in candidates.into_iter().flatten() {
            if candidate.is_dir() {
                return Ok(candidate);
            }
        }

        Err(Error::FormatterProfileNotFound(model_type.to_string()))
    }

    fn json_to_minijinja(value: &serde_json::Value) -> Value {
        match value {
            serde_json::Value::Null => Value::UNDEFINED,
            serde_json::Value::Bool(b) => Value::from(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::from(i)
                } else if let Some(f) = n.as_f64() {
                    Value::from(f)
                } else {
                    Value::UNDEFINED
                }
            }
            serde_json::Value::String(s) => Value::from(s.clone()),
            serde_json::Value::Array(arr) => {
                Value::from(arr.iter().map(Self::json_to_minijinja).collect::<Vec<_>>())
            }
            serde_json::Value::Object(obj) => {
                let map: std::collections::BTreeMap<String, Value> = obj
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::json_to_minijinja(v)))
                    .collect();
                Value::from_object(map)
            }
        }
    }
}

impl std::fmt::Debug for ChatFormatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatFormatter")
            .field("model_path", &self.model_path)
            .field("model_type", &self.model_type)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_model_type() {
        let config = serde_json::json!({"model_type": "llama"});
        assert_eq!(determine_model_type(&config), "llama3");

        let config = serde_json::json!({"model_type": "moondream3"});
        assert_eq!(determine_model_type(&config), "moondream3");

        let config = serde_json::json!({});
        assert_eq!(determine_model_type(&config), "llama3");
    }
}
