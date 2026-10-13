//! Chat formatting for LLM prompts.

pub mod control_tokens;
pub(crate) mod embedded_profiles {
    include!(concat!(env!("OUT_DIR"), "/embedded_profiles.rs"));
}
pub mod multimodal;

use std::collections::{BTreeMap, HashMap};
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
use crate::ipc::serialization::{ThinkingTokens, ToolCallFormat, ToolCallingTokens};

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

/// Placeholder wrapper that renders as empty text and reports `type=audio`.
#[derive(Debug, Clone)]
struct RenderableAudio;

impl fmt::Display for RenderableAudio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

impl Object for RenderableAudio {
    fn get_value(self: &std::sync::Arc<Self>, key: &Value) -> Option<Value> {
        match key.as_str()? {
            "type" => Some(Value::from("audio")),
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

/// Determine architecture model type from config.
fn determine_model_type(config: &serde_json::Value) -> Result<&str> {
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            Error::FormatterConfigNotFound("config.json missing model_type".to_string())
        })?;

    Ok(model_type)
}

/// Determine Pantheon template type from config.
fn determine_template_type(config: &serde_json::Value) -> Result<&str> {
    if let Some(template_type) = config
        .get("template_type")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        return Ok(template_type);
    }

    let _ = determine_model_type(config)?;

    Ok("default")
}

/// Determine Pantheon profile from config.
fn determine_pantheon_profile(config: &serde_json::Value) -> Result<&str> {
    let model_type = determine_model_type(config)?;

    Ok(match model_type {
        "llama" | "llama3" => "llama3",
        "moondream3" | "moondream" => "moondream3",
        "gemma3" | "gemma3_text" | "gemma" => "gemma3",
        "gemma4" | "gemma4_text" => "gemma4",
        "gemma4u" | "gemma4_unified" | "gemma4_unified_text" => "gemma4u",
        "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" => "qwen3_5",
        "qwen2" | "qwen" => "qwen2",
        "lfm2_moe" => "lfm2_5",
        other => other,
    })
}

fn tojson_filter(
    value: &Value,
    indent: Option<Value>,
    kwargs: Kwargs,
) -> std::result::Result<Value, minijinja::Error> {
    if indent.is_some() || kwargs.has("separators") || kwargs.has("sort_keys") {
        return Err(minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "Orchard template tojson currently supports only json.dumps defaults plus ensure_ascii=False",
        ));
    }
    if kwargs.has("ensure_ascii") {
        let ensure_ascii: bool = kwargs.get("ensure_ascii")?;
        if ensure_ascii {
            return Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                "tojson(ensure_ascii=True) is not supported by Orchard templates",
            ));
        }
    }

    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, TemplateJsonFormatter);
    serde::Serialize::serialize(value, &mut serializer).map_err(|err| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("failed to serialize JSON value: {err}"),
        )
    })?;
    let text = String::from_utf8(bytes).map_err(|err| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("failed to encode JSON value: {err}"),
        )
    })?;
    Ok(Value::from_safe_string(text))
}

struct TemplateJsonFormatter;

impl serde_json::ser::Formatter for TemplateJsonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
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
    /// Prompt template variant (e.g., "default", "reasoning")
    template_type: String,
    /// Control tokens
    pub control_tokens: ControlTokens,
    /// Compiled template
    template_source: String,
    /// Shared tool macro template source.
    shared_tool_macros_source: Option<String>,
    /// Parsed capabilities.yaml content.
    capabilities: serde_json::Value,
    /// Parsed generation.yaml content.
    generation: serde_json::Value,
    /// Parsed model config.json content.
    model_config: serde_json::Value,
    /// Placeholder tokens from capabilities manifest.
    capability_placeholders: Vec<String>,
    /// Cached tool-calling delimiters from capabilities manifest.
    tool_calling_tokens: ToolCallingTokens,
    /// Cached output-frame syntax and semantic channel values from capabilities manifest.
    output_frame_tokens: BTreeMap<String, String>,
    /// Cached generated-output thinking delimiters from capabilities manifest.
    thinking_tokens: ThinkingTokens,
}

impl ChatFormatter {
    /// Installed profile used to negotiate operations with a host. This is
    /// descriptive metadata, not a substitute for engine activation.
    pub fn operation_profile(&self) -> &serde_json::Value {
        &self.capabilities
    }
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

        Self::from_config(model_path, config)
    }

    /// Create a chat formatter from engine-inspected model config.
    pub fn from_config(model_path: &Path, config: serde_json::Value) -> Result<Self> {
        let model_type = determine_model_type(&config)?.to_string();
        let template_type = determine_template_type(&config)?.to_string();
        let pantheon_profile = determine_pantheon_profile(&config)?.to_string();
        let profile = Self::find_profile(&pantheon_profile)?;
        let control_tokens =
            ControlTokens::from_json_str(profile.control_tokens, profile.profile_name)?;
        let template_source = Self::load_profile_template(profile.profile_name, &template_type)?;
        let shared_tool_macros_source = Self::load_shared_template("tool_macros.jinja")?;
        let capabilities = Self::load_capabilities(&profile)?;
        let generation = Self::load_generation(&profile)?;
        let capability_placeholders = Self::extract_capability_placeholders(&capabilities);
        let tool_calling_tokens = Self::extract_tool_calling_tokens(&capabilities);
        let output_frame_tokens = Self::extract_output_frame_tokens(&capabilities);
        let thinking_tokens = Self::extract_thinking_tokens(&capabilities);

        Ok(Self {
            model_path: model_path.to_string_lossy().to_string(),
            model_type,
            template_type,
            control_tokens,
            template_source,
            shared_tool_macros_source,
            capabilities,
            generation,
            model_config: config,
            capability_placeholders,
            tool_calling_tokens,
            output_frame_tokens,
            thinking_tokens,
        })
    }

    /// Whether to clip the image placeholder from prompt text.
    ///
    /// If the placeholder is a real token (vision.tokens.start or start_image_token),
    /// it stays in the text for tokenization. If it's a synthetic placeholder
    /// (like <|image|>), it gets clipped.
    pub fn should_clip_image_placeholder(&self) -> bool {
        let has_placeholder = self.capabilities["vision"]["placeholders"]["image"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
            || self.capabilities["vision"]["tokens"]["placeholder"]
                .as_str()
                .is_some_and(|s| !s.is_empty());
        if has_placeholder {
            return true;
        }
        let has_vision_tokens = self.capabilities["vision"]["tokens"]["start"]
            .as_str()
            .is_some_and(|s| !s.is_empty());
        let has_start_token = self
            .control_tokens
            .start_image_token
            .as_ref()
            .is_some_and(|s| !s.is_empty());
        !(has_vision_tokens || has_start_token)
    }

    /// Resolve the image placeholder token from capabilities or control tokens.
    pub fn image_placeholder_token(&self) -> &str {
        // Explicit placeholder (e.g. moondream: vision.placeholders.image)
        if let Some(p) = self.capabilities["vision"]["placeholders"]["image"]
            .as_str()
            .filter(|s| !s.is_empty())
        {
            return p;
        }
        if let Some(p) = self.capabilities["vision"]["tokens"]["placeholder"]
            .as_str()
            .filter(|s| !s.is_empty())
        {
            return p;
        }
        // Vision start token (e.g. gemma: vision.tokens.start)
        if let Some(t) = self.capabilities["vision"]["tokens"]["start"]
            .as_str()
            .filter(|s| !s.is_empty())
        {
            return t;
        }
        // Legacy fallback to control_tokens
        self.control_tokens
            .start_image_token
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or("<|image|>")
    }

    /// Placeholder token used for inline capability markers in templates.
    ///
    /// This excludes image placeholders, which are handled separately.
    pub fn capability_placeholder_token(&self) -> Option<&str> {
        let image_token = self.image_placeholder_token();
        let audio_token = self.audio_placeholder_token();
        self.capability_placeholders
            .iter()
            .map(String::as_str)
            .find(|token| *token != image_token && Some(*token) != audio_token)
    }

    pub fn coord_placeholder_token(&self) -> Option<&str> {
        ["pointing", "gaze_detection"].into_iter().find_map(|name| {
            self.capabilities
                .get(name)
                .and_then(|cap| cap.get("placeholders"))
                .and_then(|placeholders| placeholders.get("coord"))
                .and_then(serde_json::Value::as_str)
                .filter(|token| !token.is_empty())
        })
    }

    pub fn audio_placeholder_token(&self) -> Option<&str> {
        self.capabilities
            .get("audio")
            .and_then(|cap| cap.get("placeholders"))
            .and_then(|placeholders| placeholders.get("audio"))
            .and_then(serde_json::Value::as_str)
            .filter(|token| !token.is_empty())
            .or_else(|| {
                self.capabilities
                    .get("audio")
                    .and_then(|cap| cap.get("tokens"))
                    .and_then(|tokens| tokens.get("placeholder"))
                    .and_then(serde_json::Value::as_str)
                    .filter(|token| !token.is_empty())
            })
    }

    /// Strip placeholders that should not be sent to PIE as text bytes.
    pub fn strip_template_placeholders(&self, prompt: &str) -> String {
        let mut stripped = prompt.to_string();

        if self.should_clip_image_placeholder() {
            stripped = stripped.replace(self.image_placeholder_token(), "");
        }

        let image_token = self.image_placeholder_token();
        for placeholder in &self.capability_placeholders {
            if placeholder == image_token {
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
        reasoning_effort: Option<&str>,
    ) -> Result<String> {
        self.apply_template_with_tools(
            conversation,
            add_generation_prompt,
            reasoning,
            task,
            reasoning_effort,
            None,
        )
    }

    /// Apply the chat template with tool schemas and capabilities context.
    pub fn apply_template_with_tools(
        &self,
        conversation: &[HashMap<String, serde_json::Value>],
        add_generation_prompt: bool,
        reasoning: bool,
        task: Option<&str>,
        reasoning_effort: Option<&str>,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<String> {
        self.render_template(
            conversation,
            add_generation_prompt,
            reasoning,
            task,
            reasoning_effort,
            tools,
        )
    }

    /// Tool-calling delimiters from capabilities.yaml.
    pub fn get_tool_calling_tokens(&self) -> &ToolCallingTokens {
        &self.tool_calling_tokens
    }

    /// Output-frame syntax and semantic channel values from capabilities.yaml.
    pub fn get_output_frame_tokens(&self) -> &BTreeMap<String, String> {
        &self.output_frame_tokens
    }

    /// Generated-output thinking delimiters from capabilities.yaml.
    pub fn get_thinking_tokens(&self) -> &ThinkingTokens {
        &self.thinking_tokens
    }

    /// Whether this model profile has native generated-output thinking support.
    pub fn supports_native_thinking(&self) -> bool {
        self.capabilities["thinking"]["native"]
            .as_bool()
            .unwrap_or(false)
    }

    /// Numeric f64 value from generation.yaml's default lane.
    pub fn generation_default_f64(&self, key: &str) -> Option<f64> {
        self.generation
            .get("default")
            .and_then(|value| value.get(key))
            .and_then(serde_json::Value::as_f64)
    }

    /// Numeric i32 value from generation.yaml's default lane.
    pub fn generation_default_i32(&self, key: &str) -> Option<i32> {
        self.generation
            .get("default")
            .and_then(|value| value.get(key))
            .and_then(serde_json::Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
    }

    fn render_template(
        &self,
        conversation: &[HashMap<String, serde_json::Value>],
        add_generation_prompt: bool,
        reasoning: bool,
        task: Option<&str>,
        reasoning_effort: Option<&str>,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<String> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_filter("tojson", tojson_filter);
        // Match Python's Jinja2 Environment config
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        if let Some(source) = &self.shared_tool_macros_source {
            env.add_template("tool_macros.jinja", source)
                .map_err(|e| Error::Template(e.to_string()))?;
        }
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
            for (key, value) in msg {
                map.insert(key.clone(), Self::json_to_minijinja(value));
            }
            map.insert("role".to_string(), Value::from(role));
            map.insert("content".to_string(), Self::json_to_minijinja(&content));

            if let Some(tool_calls) = msg.get("tool_calls") {
                let normalized_tool_calls = Self::normalize_tool_calls(tool_calls)?;
                map.insert(
                    "tool_calls".to_string(),
                    Self::json_to_minijinja(&normalized_tool_calls),
                );
            }
            if let Some(tool_call_id) = msg.get("tool_call_id") {
                map.insert(
                    "tool_call_id".to_string(),
                    Self::json_to_minijinja(tool_call_id),
                );
            }
            if let Some(name) = msg.get("name") {
                map.insert("name".to_string(), Self::json_to_minijinja(name));
            }

            interactions.push(Value::from_object(map));
        }

        let roles = self.control_tokens.roles.to_value();
        let tools_value = tools
            .map(|items| Self::json_to_minijinja(&serde_json::Value::Array(items.to_vec())))
            .unwrap_or(Value::UNDEFINED);
        let capabilities_value = Self::json_to_minijinja(&self.capabilities);
        let model_config_value = Self::json_to_minijinja(&self.model_config);

        let ctx = context! {
            messages => interactions.clone(),
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
            reasoning_effort => reasoning_effort.unwrap_or("medium"),
            task => task,
            roles => roles,
            tools => tools_value,
            capabilities => capabilities_value,
            model_config => model_config_value,
        };

        template
            .render(ctx)
            .map_err(|e| Error::Template(e.to_string()))
    }

    fn load_capabilities(
        profile: &embedded_profiles::EmbeddedProfile,
    ) -> Result<serde_json::Value> {
        let parsed_yaml: serde_yaml::Value =
            serde_yaml::from_str(profile.capabilities).map_err(|e| {
                Error::Template(format!(
                    "Failed to parse embedded capabilities for {}: {}",
                    profile.model_type, e
                ))
            })?;

        serde_json::to_value(parsed_yaml).map_err(|e| {
            Error::Template(format!(
                "Failed to convert embedded capabilities for {}: {}",
                profile.model_type, e
            ))
        })
    }

    fn load_generation(profile: &embedded_profiles::EmbeddedProfile) -> Result<serde_json::Value> {
        let parsed_yaml: serde_yaml::Value =
            serde_yaml::from_str(profile.generation).map_err(|e| {
                Error::Template(format!(
                    "Failed to parse embedded generation for {}: {}",
                    profile.model_type, e
                ))
            })?;

        serde_json::to_value(parsed_yaml).map_err(|e| {
            Error::Template(format!(
                "Failed to convert embedded generation for {}: {}",
                profile.model_type, e
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
                    name: format
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    call_start: token("start"),
                    inline_start: token("inline_start"),
                    channel: token("channel"),
                    recipient_prefix: token("recipient_prefix"),
                    constraint_prefix: token("constraint_prefix"),
                    constraint: token("constraint"),
                    message: token("message"),
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

    fn extract_output_frame_tokens(capabilities: &serde_json::Value) -> BTreeMap<String, String> {
        let mut output_frame_tokens = BTreeMap::new();
        if let Some(markers) = capabilities
            .get("output_framing")
            .and_then(|cap| cap.get("markers"))
            .and_then(serde_json::Value::as_object)
        {
            for (name, value) in markers {
                if let Some(token) = value.as_str() {
                    output_frame_tokens.insert(format!("marker.{}", name), token.to_string());
                }
            }
        }
        if let Some(channels) = capabilities
            .get("output_framing")
            .and_then(|cap| cap.get("channels"))
            .and_then(serde_json::Value::as_object)
        {
            for (name, value) in channels {
                if let Some(token) = value.as_str() {
                    output_frame_tokens.insert(format!("channel.{}", name), token.to_string());
                }
            }
        }
        output_frame_tokens
    }

    fn extract_thinking_tokens(capabilities: &serde_json::Value) -> ThinkingTokens {
        let tokens = capabilities
            .get("thinking")
            .and_then(|cap| cap.get("tokens"))
            .and_then(serde_json::Value::as_object);
        let token = |key: &str| {
            tokens
                .and_then(|map| map.get(key))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string()
        };

        ThinkingTokens {
            start: token("start"),
            end: token("end"),
            required: capabilities
                .get("thinking")
                .and_then(|cap| cap.get("required"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        }
    }

    fn find_profile(model_type: &str) -> Result<embedded_profiles::EmbeddedProfile> {
        embedded_profiles::find_embedded_profile(model_type)
            .ok_or_else(|| Error::FormatterProfileNotFound(model_type.to_string()))
    }

    fn load_profile_template(profile_name: &str, template_type: &str) -> Result<String> {
        embedded_profiles::load_profile_template(profile_name, template_type)
            .map(str::to_owned)
            .ok_or_else(|| {
                Error::FormatterProfileNotFound(format!("{profile_name}/{template_type}"))
            })
    }

    fn load_shared_template(template_name: &str) -> Result<Option<String>> {
        Ok(embedded_profiles::load_shared_template(template_name).map(str::to_owned))
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
                        "audio" => {
                            return Value::from_object(RenderableAudio);
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
            .field("template_type", &self.template_type)
            .finish()
    }
}
