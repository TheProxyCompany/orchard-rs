//! Chat formatting for LLM prompts.

pub mod control_tokens;
pub mod multimodal;

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use minijinja::value::{Kwargs, Object};
use minijinja::{context, Environment, Value};
use minijinja_contrib::pycompat::unknown_method_callback;

pub use control_tokens::{ControlTokens, Role, RoleTags};
pub use multimodal::{
    build_multimodal_layout, build_multimodal_messages, CapabilityInput, LayoutSegment,
};

use crate::error::{Error, Result};
use crate::ipc::serialization::{ToolCallFormat, ToolCallingTokens};

/// Wrapper that renders as text but exposes an indexable `type` field for Jinja.
/// Mirrors Python's _RenderableText behavior.
#[derive(Debug, Clone)]
struct RenderableText(String);

impl fmt::Display for RenderableText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Object for RenderableText {
    fn get_value(self: &std::sync::Arc<Self>, key: &Value) -> Option<Value> {
        match key.as_str()? {
            "type" => Some(Value::from("text")),
            "text" => Some(Value::from(self.0.clone())),
            _ => None,
        }
    }

    fn render(self: &std::sync::Arc<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Placeholder wrapper that renders as empty text and reports `type=image`.
#[derive(Debug, Clone)]
struct RenderableImage;

impl fmt::Display for RenderableImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

impl Object for RenderableImage {
    fn get_value(self: &std::sync::Arc<Self>, key: &Value) -> Option<Value> {
        match key.as_str()? {
            "type" => Some(Value::from("image")),
            _ => None,
        }
    }
}

/// Placeholder wrapper for capability inputs. Renders as empty.
#[derive(Debug, Clone)]
struct RenderableCapability;

impl fmt::Display for RenderableCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

impl Object for RenderableCapability {
    fn get_value(self: &std::sync::Arc<Self>, key: &Value) -> Option<Value> {
        match key.as_str()? {
            "type" => Some(Value::from("capability")),
            _ => None,
        }
    }
}

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

fn tojson_filter(
    value: &Value,
    indent: Option<Value>,
    kwargs: Kwargs,
) -> std::result::Result<Value, minijinja::Error> {
    if kwargs.has("ensure_ascii") {
        let _: Option<bool> = kwargs.get("ensure_ascii")?;
    }
    minijinja::filters::tojson(value, indent, kwargs)
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
    /// Parsed capabilities.yaml content.
    capabilities: serde_json::Value,
    /// Placeholder tokens from capabilities manifest.
    capability_placeholders: Vec<String>,
    /// Cached tool-calling delimiters from capabilities manifest.
    tool_calling_tokens: ToolCallingTokens,
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
        let capabilities = Self::load_capabilities(&profile_dir)?;
        let capability_placeholders = Self::extract_capability_placeholders(&capabilities);
        let tool_calling_tokens = Self::extract_tool_calling_tokens(&capabilities);

        Ok(Self {
            model_path: model_path.to_string_lossy().to_string(),
            model_type,
            control_tokens,
            template_source,
            capabilities,
            capability_placeholders,
            tool_calling_tokens,
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

    /// Image placeholder token to use for layout detection.
    pub fn image_placeholder_token(&self) -> &str {
        self.control_tokens
            .start_image_token
            .as_deref()
            .filter(|token| !token.is_empty())
            .unwrap_or(self.default_image_placeholder())
    }

    /// Placeholder token used for inline capability markers in templates.
    ///
    /// This excludes image placeholders, which are handled separately.
    pub fn capability_placeholder_token(&self) -> Option<&str> {
        let image_token = self.image_placeholder_token();
        self.capability_placeholders
            .iter()
            .map(String::as_str)
            .find(|token| *token != image_token && *token != self.default_image_placeholder())
    }

    /// Strip placeholders that should not be sent to PIE as text bytes.
    pub fn strip_template_placeholders(&self, prompt: &str) -> String {
        let mut stripped = prompt.to_string();

        if self.should_clip_image_placeholder() {
            stripped = stripped.replace(self.default_image_placeholder(), "");
        }

        let image_token = self.image_placeholder_token();
        for placeholder in &self.capability_placeholders {
            if placeholder == image_token || placeholder == self.default_image_placeholder() {
                continue;
            }
            stripped = stripped.replace(placeholder, "");
        }

        stripped
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
        self.apply_template_with_tools(conversation, add_generation_prompt, reasoning, task, None)
    }

    /// Apply the chat template with tool schemas and capabilities context.
    pub fn apply_template_with_tools(
        &self,
        conversation: &[HashMap<String, serde_json::Value>],
        add_generation_prompt: bool,
        reasoning: bool,
        task: Option<&str>,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<String> {
        self.render_template(conversation, add_generation_prompt, reasoning, task, tools)
    }

    /// Tool-calling delimiters from capabilities.yaml.
    pub fn get_tool_calling_tokens(&self) -> &ToolCallingTokens {
        &self.tool_calling_tokens
    }

    fn render_template(
        &self,
        conversation: &[HashMap<String, serde_json::Value>],
        add_generation_prompt: bool,
        reasoning: bool,
        task: Option<&str>,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<String> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_filter("tojson", tojson_filter);
        // Match Python's Jinja2 Environment config
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.add_template("chat", &self.template_source)
            .map_err(|e| Error::Template(e.to_string()))?;

        let template = env
            .get_template("chat")
            .map_err(|e| Error::Template(e.to_string()))?;

        let mut interactions: Vec<Value> = Vec::with_capacity(conversation.len());
        for msg in conversation {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = msg
                .get("content")
                .cloned()
                .unwrap_or(serde_json::Value::Null);

            let mut map = std::collections::BTreeMap::new();
            map.insert("role".to_string(), Value::from(role));
            map.insert("content".to_string(), Self::json_to_minijinja(&content));

            if let Some(tool_calls) = msg.get("tool_calls") {
                let normalized_tool_calls = Self::normalize_tool_calls(tool_calls)?;
                map.insert(
                    "tool_calls".to_string(),
                    Self::json_to_minijinja(&normalized_tool_calls),
                );
            }

            interactions.push(Value::from_object(map));
        }

        let roles = self.control_tokens.roles.to_value();
        let tools_value = tools
            .map(|items| Self::json_to_minijinja(&serde_json::Value::Array(items.to_vec())))
            .unwrap_or(Value::UNDEFINED);
        let capabilities_value = Self::json_to_minijinja(&self.capabilities);

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
            tools => tools_value,
            capabilities => capabilities_value,
        };

        template
            .render(ctx)
            .map_err(|e| Error::Template(e.to_string()))
    }

    fn load_capabilities(profile_dir: &Path) -> Result<serde_json::Value> {
        let capabilities_path = profile_dir.join("capabilities.yaml");
        if !capabilities_path.exists() {
            return Ok(serde_json::json!({}));
        }

        let content = std::fs::read_to_string(&capabilities_path)?;
        let parsed_yaml: serde_yaml::Value = serde_yaml::from_str(&content).map_err(|e| {
            Error::Template(format!(
                "Failed to parse capabilities from {:?}: {}",
                capabilities_path, e
            ))
        })?;

        serde_json::to_value(parsed_yaml).map_err(|e| {
            Error::Template(format!(
                "Failed to convert capabilities from {:?}: {}",
                capabilities_path, e
            ))
        })
    }

    fn extract_capability_placeholders(capabilities: &serde_json::Value) -> Vec<String> {
        let mut placeholders = Vec::new();

        if let Some(capability_map) = capabilities.as_object() {
            for capability in capability_map.values() {
                let Some(placeholder_map) = capability
                    .get("placeholders")
                    .and_then(serde_json::Value::as_object)
                else {
                    continue;
                };

                for placeholder in placeholder_map
                    .values()
                    .filter_map(serde_json::Value::as_str)
                {
                    if placeholder.is_empty() || placeholders.iter().any(|p| p == placeholder) {
                        continue;
                    }
                    placeholders.push(placeholder.to_string());
                }
            }
        }

        placeholders
    }

    fn extract_tool_calling_tokens(capabilities: &serde_json::Value) -> ToolCallingTokens {
        let format_entries = capabilities
            .get("tool_calling")
            .and_then(|cap| cap.get("formats"))
            .and_then(serde_json::Value::as_array);

        let mut formats = Vec::new();
        let mut section_start = String::new();
        let mut section_end = String::new();

        if let Some(entries) = format_entries {
            for (index, format) in entries.iter().enumerate() {
                let token_map = format.get("tokens").and_then(serde_json::Value::as_object);
                let token = |key: &str| {
                    token_map
                        .and_then(|tokens| tokens.get(key))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string()
                };

                formats.push(ToolCallFormat {
                    call_start: token("start"),
                    call_end: token("end"),
                });
                if index == 0 {
                    section_start = token("section_start");
                    section_end = token("section_end");
                }
            }
        }

        ToolCallingTokens {
            formats,
            section_start,
            section_end,
        }
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
                // Check if this is a content part with a "type" field
                // If so, wrap it in the appropriate Renderable type
                if let Some(type_val) = obj.get("type").and_then(|v| v.as_str()) {
                    match type_val {
                        "text" => {
                            if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                                return Value::from_object(RenderableText(text.to_string()));
                            }
                        }
                        "image" => {
                            return Value::from_object(RenderableImage);
                        }
                        "capability" => {
                            return Value::from_object(RenderableCapability);
                        }
                        _ => {}
                    }
                }
                // Preserve object key order using minijinja's serde conversion path.
                Value::from_serialize(obj)
            }
        }
    }

    fn normalize_tool_calls(tool_calls: &serde_json::Value) -> Result<serde_json::Value> {
        let Some(calls) = tool_calls.as_array() else {
            return Ok(tool_calls.clone());
        };

        let mut normalized_calls = Vec::with_capacity(calls.len());
        for (call_idx, call) in calls.iter().enumerate() {
            let Some(call_obj) = call.as_object() else {
                normalized_calls.push(call.clone());
                continue;
            };

            let mut call_obj = call_obj.clone();
            if let Some(function_obj) = call_obj
                .get_mut("function")
                .and_then(serde_json::Value::as_object_mut)
            {
                if let Some(arguments) = function_obj
                    .get("arguments")
                    .and_then(serde_json::Value::as_str)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
                {
                    let parsed = serde_json::from_str(&arguments).map_err(|error| {
                        Error::Other(format!(
                            "Failed to parse tool call arguments at index {}: {}",
                            call_idx, error
                        ))
                    })?;
                    function_obj.insert("arguments".to_string(), parsed);
                }
            }

            normalized_calls.push(serde_json::Value::Object(call_obj));
        }

        Ok(serde_json::Value::Array(normalized_calls))
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
    use std::collections::HashMap;

    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_determine_model_type() {
        let config = serde_json::json!({"model_type": "llama"});
        assert_eq!(determine_model_type(&config), "llama3");

        let config = serde_json::json!({"model_type": "moondream3"});
        assert_eq!(determine_model_type(&config), "moondream3");

        let config = serde_json::json!({});
        assert_eq!(determine_model_type(&config), "llama3");
    }

    #[test]
    fn test_renders_tool_call_delimiters_with_normalized_arguments() {
        let model_dir = tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            serde_json::json!({"model_type": "llama3"}).to_string(),
        )
        .unwrap();

        let formatter = ChatFormatter::new(model_dir.path()).unwrap();
        let call_id = "call_weather_1";
        let items = vec![
            HashMap::from([
                ("role".to_string(), serde_json::json!("assistant")),
                ("content".to_string(), serde_json::Value::Null),
                (
                    "tool_calls".to_string(),
                    serde_json::json!([{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": "{\"city\":\"San Francisco\"}",
                        }
                    }]),
                ),
            ]),
            HashMap::from([
                ("role".to_string(), serde_json::json!("tool")),
                (
                    "content".to_string(),
                    serde_json::json!("{\"status\":\"ok\"}"),
                ),
            ]),
        ];

        let (messages, _, _, _) = build_multimodal_messages(&formatter, &items, None).unwrap();
        assert!(messages[0].contains_key("tool_calls"));

        let rendered = formatter
            .apply_template_with_tools(
                &messages,
                false,
                false,
                None,
                Some(&[serde_json::json!({
                    "name": "share_to_party",
                    "type": "object",
                    "description": "Post a message to the party.",
                    "properties": {
                        "name": { "const": "share_to_party" },
                        "arguments": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string" }
                            },
                            "required": ["content"]
                        }
                    },
                    "strict": true,
                    "required": ["name", "arguments"]
                })]),
            )
            .unwrap();
        let format = formatter
            .get_tool_calling_tokens()
            .formats
            .first()
            .expect("llama3 profile should define tool-calling format tokens");

        assert!(rendered.contains(&format.call_start));
        assert!(rendered.contains(&format.call_end));
        assert!(rendered.contains("\"name\":\"lookup_weather\""));
        assert!(rendered.contains("\"arguments\":{\"city\":\"San Francisco\"}"));
        assert!(!rendered.contains("\"arguments\":\"{\\\"city\\\":\\\"San Francisco\\\"}\""));

        let properties_start = rendered
            .find("\"properties\": {")
            .expect("expected tool schema properties block in rendered prompt");
        let properties_block = &rendered[properties_start..];
        let name_idx = properties_block
            .find("\"name\": {")
            .expect("expected name key in tool schema properties");
        let arguments_idx = properties_block
            .find("\"arguments\": {")
            .expect("expected arguments key in tool schema properties");
        assert!(
            name_idx < arguments_idx,
            "expected name to appear before arguments in tool schema properties block"
        );
    }
}
