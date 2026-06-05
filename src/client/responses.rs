//! OpenAI Responses API surface for Orchard.
//!
//! This module maps PIE state-events carried over IPC into typed Responses API events.

use std::collections::{BTreeMap, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::{distributions::Alphanumeric, Rng};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use super::{native_reasoning_settings, tool_choice_to_string, Client, ClientError, Result};
use crate::formatter::multimodal::{build_multimodal_layout, build_multimodal_messages};
use crate::ipc::client::{ResponseDelta, ResponseStateEvent};
use crate::ipc::serialization::{PromptPayload, ToolCallingTokens};

const RESPONSE_ID_PREFIX: &str = "resp_";
const MESSAGE_ID_PREFIX: &str = "msg_";
const FUNCTION_CALL_ID_PREFIX: &str = "fc_";
const TOOL_CALL_ID_PREFIX: &str = "call_";
const DEFAULT_REASONING_EFFORT: &str = "medium";

fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn generate_id(prefix: &str) -> String {
    let random: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(22)
        .map(char::from)
        .collect();
    format!("{}{}", prefix, random)
}

fn generate_response_id() -> String {
    generate_id(RESPONSE_ID_PREFIX)
}

fn generate_message_id() -> String {
    generate_id(MESSAGE_ID_PREFIX)
}

fn generate_function_call_id() -> String {
    generate_id(FUNCTION_CALL_ID_PREFIX)
}

fn generate_tool_call_id() -> String {
    generate_id(TOOL_CALL_ID_PREFIX)
}

fn finish_reason_to_incomplete(reason: Option<&str>) -> Option<IncompleteDetails> {
    let normalized = reason.unwrap_or_default().to_lowercase();
    if matches!(
        normalized.as_str(),
        "length" | "max_tokens" | "max_output_tokens"
    ) {
        Some(IncompleteDetails {
            reason: "max_output_tokens".to_string(),
        })
    } else if normalized == "content_filter" {
        Some(IncompleteDetails {
            reason: "content_filter".to_string(),
        })
    } else if matches!(normalized.as_str(), "user" | "cancelled" | "canceled") {
        Some(IncompleteDetails {
            reason: "cancelled".to_string(),
        })
    } else {
        None
    }
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => value.to_string(),
    }
}

#[derive(Debug, Deserialize)]
struct CompletedToolCallValue {
    name: String,
    #[serde(default = "default_tool_call_arguments")]
    arguments: Value,
}

fn default_tool_call_arguments() -> Value {
    Value::Object(Default::default())
}

fn parse_tool_call_completion_value(value: &Value) -> Option<(String, String)> {
    let structured_value = match value {
        Value::String(text) => serde_json::from_str::<Value>(text).ok()?,
        Value::Object(_) => value.clone(),
        _ => return None,
    };

    let tool_call: CompletedToolCallValue = serde_json::from_value(structured_value).ok()?;
    Some((
        tool_call.name,
        response_arguments_json(&tool_call.arguments),
    ))
}

fn response_arguments_json(value: &Value) -> String {
    match value {
        Value::Array(values) => {
            let values = values
                .iter()
                .map(response_arguments_json)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{values}]")
        }
        Value::Object(object) => {
            let fields = object
                .iter()
                .map(|(key, value)| {
                    let key = serde_json::to_string(key).expect("JSON object key serializes");
                    format!("{key}: {}", response_arguments_json(value))
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{fields}}}")
        }
        other => serde_json::to_string(other).expect("JSON value serializes"),
    }
}

fn normalize_response_tool_schema(tool: &Value) -> Value {
    let Some(obj) = tool.as_object() else {
        return tool.clone();
    };

    let type_name = obj.get("type").and_then(Value::as_str);
    let name = obj.get("name").and_then(Value::as_str);
    let parameters = obj.get("parameters");

    if type_name != Some("function") || name.is_none() || parameters.is_none() {
        return tool.clone();
    }

    let name = name.unwrap_or_default();
    let description = obj
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or(name);
    let strict = obj.get("strict").and_then(Value::as_bool).unwrap_or(true);
    let parameters = parameters
        .cloned()
        .unwrap_or(Value::Object(Default::default()));

    serde_json::json!({
        "name": name,
        "type": "object",
        "description": description,
        "properties": {
            "name": {"const": name},
            "arguments": parameters,
        },
        "strict": strict,
        "required": ["name", "arguments"],
    })
}

fn response_tool_prompt_payload(tool: &Value) -> Value {
    let Some(obj) = tool.as_object() else {
        return tool.clone();
    };

    let type_name = obj.get("type").and_then(Value::as_str);
    let Some(name) = obj.get("name").and_then(Value::as_str) else {
        return tool.clone();
    };
    let parameters = obj.get("parameters");
    if type_name != Some("function") || parameters.is_none() {
        return tool.clone();
    }

    let description = obj
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or(name);
    let strict = obj.get("strict").and_then(Value::as_bool).unwrap_or(true);
    let mut payload = serde_json::Map::new();
    payload.insert("name".to_string(), Value::String(name.to_string()));
    payload.insert("type".to_string(), Value::String("function".to_string()));
    payload.insert(
        "description".to_string(),
        Value::String(description.to_string()),
    );
    payload.insert("strict".to_string(), Value::Bool(strict));
    payload.insert(
        "parameters".to_string(),
        parameters
            .cloned()
            .unwrap_or(Value::Object(Default::default())),
    );
    Value::Object(payload)
}

fn response_tool_schema_name(tool: &Value) -> &str {
    tool.get("name").and_then(Value::as_str).unwrap_or_default()
}

fn normalize_response_format(value: Value) -> Value {
    let Value::Object(object) = value else {
        return value;
    };
    if object.get("type").and_then(Value::as_str) != Some("json_schema") {
        return Value::Object(object);
    }

    let schema = object
        .get("json_schema")
        .or_else(|| object.get("schema"))
        .cloned()
        .unwrap_or(Value::Object(Default::default()));
    let name = object
        .get("name")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let description = object
        .get("description")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let strict = object
        .get("strict")
        .cloned()
        .unwrap_or_else(|| Value::Bool(false));

    let mut payload = serde_json::Map::new();
    payload.insert("type".to_string(), Value::String("json_schema".to_string()));
    payload.insert("json_schema".to_string(), schema);
    payload.insert("name".to_string(), name);
    payload.insert("description".to_string(), description);
    payload.insert("strict".to_string(), strict);
    Value::Object(payload)
}

fn response_tool_schemas(
    core_source: &[Value],
    active_source: &[Value],
) -> (Vec<Value>, Vec<Value>) {
    let mut tool_schemas = core_source
        .iter()
        .map(response_tool_prompt_payload)
        .collect::<Vec<_>>();
    tool_schemas.sort_by(|a, b| response_tool_schema_name(a).cmp(response_tool_schema_name(b)));

    let active_source = if active_source.is_empty() {
        core_source
    } else {
        active_source
    };
    let mut active_tool_schemas = active_source
        .iter()
        .map(normalize_response_tool_schema)
        .collect::<Vec<_>>();
    active_tool_schemas
        .sort_by(|a, b| response_tool_schema_name(a).cmp(response_tool_schema_name(b)));

    (tool_schemas, active_tool_schemas)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputItem {
    Message {
        role: String,
        content: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<Value>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: FunctionCallOutputContent,
    },
    Reasoning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<Vec<Value>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FunctionCallOutputContent {
    Text(String),
    Content(Vec<Value>),
}

impl FunctionCallOutputContent {
    fn to_message_content(&self) -> Value {
        match self {
            Self::Text(text) => Value::String(text.clone()),
            Self::Content(parts) => Value::Array(parts.clone()),
        }
    }
}

impl From<String> for FunctionCallOutputContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for FunctionCallOutputContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReasoningConfig {
    Bool(bool),
    Effort(String),
    Object { effort: String },
}

impl ReasoningConfig {
    fn effort(&self) -> Option<String> {
        match self {
            Self::Bool(true) => Some(DEFAULT_REASONING_EFFORT.to_string()),
            Self::Bool(false) => None,
            Self::Effort(effort) => Some(effort.clone()),
            Self::Object { effort } => Some(effort.clone()),
        }
    }

    fn explicit_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }
}

impl From<bool> for ReasoningConfig {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<&str> for ReasoningConfig {
    fn from(value: &str) -> Self {
        Self::Effort(value.to_string())
    }
}

impl From<String> for ReasoningConfig {
    fn from(value: String) -> Self {
        Self::Effort(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub input: ResponsesInput,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub deterministic: bool,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub max_output_tokens: Option<i32>,
    #[serde(default)]
    pub top_logprobs: Option<i32>,
    #[serde(default)]
    pub core_tools: Vec<Value>,
    #[serde(default)]
    pub active_tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub min_tool_calls: Option<i32>,
    #[serde(default)]
    pub max_tool_calls: Option<i32>,
    #[serde(default)]
    pub text: Option<Value>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(default)]
    pub parallel_tool_calls: bool,
    /// None => use the shared prefix cache (default). Some(false) => skip read+write.
    #[serde(default)]
    pub prefix_cache: Option<bool>,
    /// Client-side only: emit one OutputToken event per raw generated token id.
    /// Deliberately never placed on the wire / PromptPayload.
    #[serde(default)]
    pub stream_tokens: bool,
}

impl ResponsesRequest {
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            input: ResponsesInput::Text(text.into()),
            stream: false,
            instructions: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            deterministic: false,
            frequency_penalty: None,
            presence_penalty: None,
            max_output_tokens: None,
            top_logprobs: None,
            core_tools: Vec::new(),
            active_tools: Vec::new(),
            tool_choice: None,
            min_tool_calls: None,
            max_tool_calls: None,
            text: None,
            reasoning: None,
            reasoning_effort: None,
            metadata: None,
            parallel_tool_calls: false,
            prefix_cache: None,
            stream_tokens: false,
        }
    }

    fn to_messages(&self) -> Vec<HashMap<String, Value>> {
        match &self.input {
            ResponsesInput::Text(text) => {
                let mut message = HashMap::new();
                message.insert("role".to_string(), Value::String("user".to_string()));
                message.insert("content".to_string(), Value::String(text.clone()));
                vec![message]
            }
            ResponsesInput::Items(items) => {
                let mut messages = Vec::new();
                for item in items {
                    match item {
                        ResponseInputItem::Message {
                            role,
                            content,
                            tool_calls,
                            tool_call_id,
                        } => {
                            let mut message = HashMap::new();
                            message.insert("role".to_string(), Value::String(role.clone()));
                            message.insert("content".to_string(), content.clone());
                            if let Some(calls) = tool_calls {
                                message
                                    .insert("tool_calls".to_string(), Value::Array(calls.clone()));
                            }
                            if let Some(call_id) = tool_call_id {
                                message.insert(
                                    "tool_call_id".to_string(),
                                    Value::String(call_id.clone()),
                                );
                            }
                            messages.push(message);
                        }
                        ResponseInputItem::FunctionCall {
                            call_id,
                            name,
                            arguments,
                        } => {
                            let mut message = HashMap::new();
                            message
                                .insert("role".to_string(), Value::String("assistant".to_string()));
                            message.insert("content".to_string(), Value::String(String::new()));
                            message.insert(
                                "tool_calls".to_string(),
                                Value::Array(vec![serde_json::json!({
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": arguments,
                                    }
                                })]),
                            );
                            messages.push(message);
                        }
                        ResponseInputItem::FunctionCallOutput { call_id, output } => {
                            let mut message = HashMap::new();
                            message.insert("role".to_string(), Value::String("tool".to_string()));
                            message.insert("content".to_string(), output.to_message_content());
                            message
                                .insert("tool_call_id".to_string(), Value::String(call_id.clone()));
                            messages.push(message);
                        }
                        ResponseInputItem::Reasoning { .. } => {
                            // Reasoning items are not directly representable in template messages.
                        }
                    }
                }
                messages
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum OutputStatus {
    #[default]
    Completed,
    Incomplete,
    InProgress,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputTextContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
    #[serde(default)]
    pub annotations: Vec<Value>,
    #[serde(default)]
    pub logprobs: Vec<Value>,
}

impl OutputTextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content_type: "output_text".to_string(),
            text: text.into(),
            annotations: Vec::new(),
            logprobs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl ReasoningContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content_type: "reasoning_text".to_string(),
            text: text.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningSummaryTextContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputMessage {
    #[serde(rename = "type")]
    pub output_type: String,
    pub id: String,
    pub status: OutputStatus,
    pub role: String,
    pub content: Vec<OutputTextContent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputFunctionCall {
    #[serde(rename = "type")]
    pub output_type: String,
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    pub status: OutputStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputReasoning {
    #[serde(rename = "type")]
    pub output_type: String,
    pub id: String,
    #[serde(default)]
    pub status: OutputStatus,
    #[serde(default)]
    pub summary: Vec<ReasoningSummaryTextContent>,
    #[serde(default)]
    pub content: Vec<ReasoningContent>,
    #[serde(default)]
    pub encrypted_content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseOutputItem {
    Message(OutputMessage),
    FunctionCall(OutputFunctionCall),
    Reasoning(OutputReasoning),
}

impl ResponseOutputItem {
    pub fn item_type(&self) -> &'static str {
        match self {
            Self::Message(_) => "message",
            Self::FunctionCall(_) => "function_call",
            Self::Reasoning(_) => "reasoning",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncompleteDetails {
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    #[serde(default)]
    pub completed_at: Option<i64>,
    pub status: OutputStatus,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    pub error: Option<ResponseError>,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    #[serde(default)]
    pub usage: Option<ResponseUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    pub parallel_tool_calls: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_tool_calls: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<Value>,
    /// Model stop/EOS token id that ended generation, if any. Proxy extension.
    #[serde(default)]
    pub stop_token_id: Option<i32>,
    /// Decoded text of that stop token (e.g. "<|eom_id|>"). Proxy extension.
    #[serde(default)]
    pub stop_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseSnapshot {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    #[serde(default)]
    pub completed_at: Option<i64>,
    pub status: OutputStatus,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    #[serde(default)]
    pub usage: Option<ResponseUsage>,
    /// Model stop/EOS token id that ended generation, if any. Proxy extension.
    #[serde(default)]
    pub stop_token_id: Option<i32>,
    /// Decoded text of that stop token (e.g. "<|eom_id|>"). Proxy extension.
    #[serde(default)]
    pub stop_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseContentPart {
    OutputText(OutputTextContent),
    Reasoning(ReasoningContent),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseCreatedEvent {
    pub sequence_number: u64,
    pub response: ResponseSnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseInProgressEvent {
    pub sequence_number: u64,
    pub response: ResponseSnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseCompletedEvent {
    pub sequence_number: u64,
    pub response: ResponseSnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseFailedEvent {
    pub sequence_number: u64,
    pub response: ResponseSnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseIncompleteEvent {
    pub sequence_number: u64,
    pub response: ResponseSnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputItemAddedEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item: ResponseOutputItem,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputItemDoneEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item: ResponseOutputItem,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentPartAddedEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub part: ResponseContentPart,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentPartDoneEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub part: ResponseContentPart,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputTextDeltaEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub delta: String,
    #[serde(default)]
    pub logprobs: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputTextDoneEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub text: String,
    #[serde(default)]
    pub logprobs: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputTokenEvent {
    pub sequence_number: u64,
    pub token_id: i32,
    /// Engine run-decoded text for the delta (faithful decode PIE already
    /// computed); None/empty when the token does not yet complete a UTF-8 run.
    pub content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallArgumentsDeltaEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub delta: String,
    #[serde(default)]
    pub field_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallArgumentsDoneEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningDeltaEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub delta: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningDoneEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningSummaryTextDeltaEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub summary_index: u32,
    pub delta: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningSummaryTextDoneEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub summary_index: u32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamErrorEvent {
    pub sequence_number: u64,
    pub error: StreamErrorDetail,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseEvent {
    ResponseCreated(ResponseCreatedEvent),
    ResponseInProgress(ResponseInProgressEvent),
    ResponseCompleted(ResponseCompletedEvent),
    ResponseFailed(ResponseFailedEvent),
    ResponseIncomplete(ResponseIncompleteEvent),
    OutputItemAdded(OutputItemAddedEvent),
    OutputItemDone(OutputItemDoneEvent),
    ContentPartAdded(ContentPartAddedEvent),
    ContentPartDone(ContentPartDoneEvent),
    OutputTextDelta(OutputTextDeltaEvent),
    OutputTextDone(OutputTextDoneEvent),
    OutputToken(OutputTokenEvent),
    FunctionCallArgumentsDelta(FunctionCallArgumentsDeltaEvent),
    FunctionCallArgumentsDone(FunctionCallArgumentsDoneEvent),
    ReasoningDelta(ReasoningDeltaEvent),
    ReasoningDone(ReasoningDoneEvent),
    ReasoningSummaryTextDelta(ReasoningSummaryTextDeltaEvent),
    ReasoningSummaryTextDone(ReasoningSummaryTextDoneEvent),
    Error(StreamErrorEvent),
    Done,
}

impl ResponseEvent {
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::ResponseCreated(_) => "response.created",
            Self::ResponseInProgress(_) => "response.in_progress",
            Self::ResponseCompleted(_) => "response.completed",
            Self::ResponseFailed(_) => "response.failed",
            Self::ResponseIncomplete(_) => "response.incomplete",
            Self::OutputItemAdded(_) => "response.output_item.added",
            Self::OutputItemDone(_) => "response.output_item.done",
            Self::ContentPartAdded(_) => "response.content_part.added",
            Self::ContentPartDone(_) => "response.content_part.done",
            Self::OutputTextDelta(_) => "response.output_text.delta",
            Self::OutputTextDone(_) => "response.output_text.done",
            Self::OutputToken(_) => "response.output_token",
            Self::FunctionCallArgumentsDelta(_) => "response.function_call_arguments.delta",
            Self::FunctionCallArgumentsDone(_) => "response.function_call_arguments.done",
            Self::ReasoningDelta(_) => "response.reasoning.delta",
            Self::ReasoningDone(_) => "response.reasoning.done",
            Self::ReasoningSummaryTextDelta(_) => "response.reasoning_summary_text.delta",
            Self::ReasoningSummaryTextDone(_) => "response.reasoning_summary_text.done",
            Self::Error(_) => "error",
            Self::Done => "done",
        }
    }
}

pub enum ResponsesResult {
    Complete(Box<ResponseObject>),
    Stream {
        request_id: u64,
        events: mpsc::Receiver<ResponseEvent>,
    },
}

#[derive(Debug, Clone)]
struct StreamingOutputItem {
    item_id: String,
    item_type: String,
    call_id: Option<String>,
    function_name: Option<String>,
    accumulated_content: String,
    accumulated_arguments: String,
    status: OutputStatus,
}

impl StreamingOutputItem {
    fn new(item_type: &str) -> Self {
        let item_id = match item_type {
            "tool_call" => generate_function_call_id(),
            "reasoning" => generate_id("reasoning_"),
            _ => generate_message_id(),
        };

        Self {
            item_id,
            item_type: item_type.to_string(),
            call_id: None,
            function_name: None,
            accumulated_content: String::new(),
            accumulated_arguments: String::new(),
            status: OutputStatus::InProgress,
        }
    }

    fn to_skeleton(&self) -> ResponseOutputItem {
        match self.item_type.as_str() {
            "tool_call" => ResponseOutputItem::FunctionCall(OutputFunctionCall {
                output_type: "function_call".to_string(),
                id: self.item_id.clone(),
                call_id: self.call_id.clone().unwrap_or_else(generate_tool_call_id),
                name: self.function_name.clone().unwrap_or_default(),
                arguments: String::new(),
                metadata: None,
                status: OutputStatus::InProgress,
            }),
            "reasoning" => ResponseOutputItem::Reasoning(OutputReasoning {
                output_type: "reasoning".to_string(),
                id: self.item_id.clone(),
                status: OutputStatus::InProgress,
                summary: Vec::new(),
                content: Vec::new(),
                encrypted_content: None,
            }),
            _ => ResponseOutputItem::Message(OutputMessage {
                output_type: "message".to_string(),
                id: self.item_id.clone(),
                status: OutputStatus::InProgress,
                role: "assistant".to_string(),
                content: Vec::new(),
            }),
        }
    }

    fn to_completed(&self) -> ResponseOutputItem {
        match self.item_type.as_str() {
            "tool_call" => ResponseOutputItem::FunctionCall(OutputFunctionCall {
                output_type: "function_call".to_string(),
                id: self.item_id.clone(),
                call_id: self.call_id.clone().unwrap_or_else(generate_tool_call_id),
                name: self.function_name.clone().unwrap_or_default(),
                arguments: self.accumulated_arguments.clone(),
                metadata: None,
                status: OutputStatus::Completed,
            }),
            "reasoning" => ResponseOutputItem::Reasoning(OutputReasoning {
                output_type: "reasoning".to_string(),
                id: self.item_id.clone(),
                status: OutputStatus::Completed,
                summary: Vec::new(),
                content: if self.accumulated_content.is_empty() {
                    Vec::new()
                } else {
                    vec![ReasoningContent::new(self.accumulated_content.clone())]
                },
                encrypted_content: None,
            }),
            _ => ResponseOutputItem::Message(OutputMessage {
                output_type: "message".to_string(),
                id: self.item_id.clone(),
                status: OutputStatus::Completed,
                role: "assistant".to_string(),
                content: if self.accumulated_content.is_empty() {
                    Vec::new()
                } else {
                    vec![OutputTextContent::new(self.accumulated_content.clone())]
                },
            }),
        }
    }
}

#[derive(Debug, Clone)]
struct ResponseStreamState {
    response_id: String,
    model: String,
    created_at: i64,
    completed_at: Option<i64>,
    items: BTreeMap<u32, StreamingOutputItem>,
    sequence_number: u64,
    status: OutputStatus,
    incomplete_details: Option<IncompleteDetails>,
    usage: Option<ResponseUsage>,
    stop_token_id: Option<i32>,
    stop_token: Option<String>,
}

impl ResponseStreamState {
    fn new(response_id: String, model: String) -> Self {
        Self {
            response_id,
            model,
            created_at: current_timestamp(),
            completed_at: None,
            items: BTreeMap::new(),
            sequence_number: 0,
            status: OutputStatus::InProgress,
            incomplete_details: None,
            usage: None,
            stop_token_id: None,
            stop_token: None,
        }
    }

    fn next_sequence_number(&mut self) -> u64 {
        let current = self.sequence_number;
        self.sequence_number = self.sequence_number.saturating_add(1);
        current
    }

    fn get_or_create_item(
        &mut self,
        output_index: u32,
        item_type: &str,
        identifier: &str,
    ) -> &mut StreamingOutputItem {
        self.items.entry(output_index).or_insert_with(|| {
            let mut item = StreamingOutputItem::new(item_type);
            if item_type == "tool_call" {
                item.call_id = Some(generate_tool_call_id());
                if !identifier.is_empty() {
                    item.function_name =
                        Some(identifier.trim_start_matches("tool_call:").to_string());
                }
            }
            item
        })
    }

    fn snapshot(&self) -> ResponseSnapshot {
        let output = self
            .items
            .values()
            .map(|item| {
                if item.status == OutputStatus::Completed {
                    item.to_completed()
                } else {
                    item.to_skeleton()
                }
            })
            .collect::<Vec<_>>();

        ResponseSnapshot {
            id: self.response_id.clone(),
            object: "response".to_string(),
            created_at: self.created_at,
            completed_at: self.completed_at,
            status: self.status,
            incomplete_details: self.incomplete_details.clone(),
            model: self.model.clone(),
            output,
            usage: self.usage.clone(),
            stop_token_id: self.stop_token_id,
            stop_token: self.stop_token.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct AggregatedOutputItem {
    item_type: String,
    content: String,
    arguments: String,
    identifier: String,
    function_name: String,
}

fn append_raw_message_output(
    output_items: &mut BTreeMap<u32, AggregatedOutputItem>,
    content: &str,
) {
    if content.is_empty() {
        return;
    }

    let output_index = output_items
        .iter()
        .rev()
        .find_map(|(index, item)| (item.item_type == "message").then_some(*index))
        .unwrap_or_else(|| {
            output_items
                .keys()
                .next_back()
                .map(|index| index.saturating_add(1))
                .unwrap_or(0)
        });
    let item = output_items
        .entry(output_index)
        .or_insert_with(|| AggregatedOutputItem {
            item_type: "message".to_string(),
            content: String::new(),
            arguments: String::new(),
            identifier: "message".to_string(),
            function_name: String::new(),
        });
    item.content.push_str(content);
}

fn process_state_event_for_output(
    event: &ResponseStateEvent,
    output_items: &mut BTreeMap<u32, AggregatedOutputItem>,
) {
    let output_index = event.output_index;
    let item_type = if event.item_type.is_empty() {
        "message"
    } else {
        event.item_type.as_str()
    };

    let item = output_items
        .entry(output_index)
        .or_insert_with(|| AggregatedOutputItem {
            item_type: item_type.to_string(),
            content: String::new(),
            arguments: String::new(),
            identifier: event.identifier.clone(),
            function_name: if item_type == "tool_call" {
                event
                    .identifier
                    .trim_start_matches("tool_call:")
                    .to_string()
            } else {
                String::new()
            },
        });

    if item.item_type == "tool_call" && event.identifier == "arguments" {
        if event.event_type == "content_delta" {
            item.arguments.push_str(&event.delta);
        } else if event.event_type == "item_completed" {
            if let Some(value) = &event.value {
                item.arguments = value_to_string(value);
            }
        }
        return;
    }

    if event.event_type == "content_delta" {
        item.content.push_str(&event.delta);
    } else if event.event_type == "item_completed" {
        if item.item_type == "tool_call" {
            item.identifier = event.identifier.clone();
            if let Some(value) = &event.value {
                if let Some((function_name, arguments)) = parse_tool_call_completion_value(value) {
                    item.function_name = function_name;
                    item.arguments = arguments;
                }
            }
            if item.function_name.is_empty() {
                item.function_name = event
                    .identifier
                    .trim_start_matches("tool_call:")
                    .to_string();
            }
        } else if let Some(value) = &event.value {
            item.content = value_to_string(value);
        }
    }
}

fn parse_raw_tool_call_message(
    content: &str,
    tool_calling_tokens: &ToolCallingTokens,
) -> Option<(String, String)> {
    let mut text = content.trim();
    if text.is_empty() {
        return None;
    }

    for format in &tool_calling_tokens.formats {
        let call_end = format.call_end.as_str();
        if call_end.is_empty() || !text.ends_with(call_end) {
            continue;
        }
        text = text[..text.len() - call_end.len()].trim_end();
        break;
    }

    let section_end = tool_calling_tokens.section_end.as_str();
    if !section_end.is_empty() && text.ends_with(section_end) {
        text = text[..text.len() - section_end.len()].trim_end();
    }

    let candidate = Value::String(text.to_string());
    if let Some(parsed) = parse_tool_call_completion_value(&candidate) {
        return Some(parsed);
    }

    if !text.starts_with('{') {
        return None;
    }
    let json_end = text.rfind('}')?;
    parse_tool_call_completion_value(&Value::String(text[..=json_end].to_string()))
}

fn build_output_items(
    output_items: &BTreeMap<u32, AggregatedOutputItem>,
    raw_tool_call_tokens: Option<&ToolCallingTokens>,
) -> Vec<ResponseOutputItem> {
    let mut output = Vec::new();
    for item in output_items.values() {
        match item.item_type.as_str() {
            "tool_call" => {
                output.push(ResponseOutputItem::FunctionCall(OutputFunctionCall {
                    output_type: "function_call".to_string(),
                    id: generate_function_call_id(),
                    call_id: generate_tool_call_id(),
                    name: item.function_name.clone(),
                    arguments: item.arguments.clone(),
                    metadata: None,
                    status: OutputStatus::Completed,
                }));
            }
            "reasoning" => {
                output.push(ResponseOutputItem::Reasoning(OutputReasoning {
                    output_type: "reasoning".to_string(),
                    id: generate_id("reasoning_"),
                    status: OutputStatus::Completed,
                    summary: Vec::new(),
                    content: if item.content.is_empty() {
                        Vec::new()
                    } else {
                        vec![ReasoningContent::new(item.content.clone())]
                    },
                    encrypted_content: None,
                }));
            }
            _ => {
                if let Some((function_name, arguments)) = raw_tool_call_tokens
                    .and_then(|tokens| parse_raw_tool_call_message(&item.content, tokens))
                {
                    output.push(ResponseOutputItem::FunctionCall(OutputFunctionCall {
                        output_type: "function_call".to_string(),
                        id: generate_function_call_id(),
                        call_id: generate_tool_call_id(),
                        name: function_name,
                        arguments,
                        metadata: None,
                        status: OutputStatus::Completed,
                    }));
                    continue;
                }
                output.push(ResponseOutputItem::Message(OutputMessage {
                    output_type: "message".to_string(),
                    id: generate_message_id(),
                    status: OutputStatus::Completed,
                    role: "assistant".to_string(),
                    content: if item.content.is_empty() {
                        Vec::new()
                    } else {
                        vec![OutputTextContent::new(item.content.clone())]
                    },
                }));
            }
        }
    }

    if output.is_empty() {
        output.push(ResponseOutputItem::Message(OutputMessage {
            output_type: "message".to_string(),
            id: generate_message_id(),
            status: OutputStatus::Completed,
            role: "assistant".to_string(),
            content: Vec::new(),
        }));
    }

    output
}

fn update_usage_from_delta(delta: &ResponseDelta, usage: &mut ResponseUsage) {
    if let Some(prompt_tokens) = delta.prompt_token_count {
        usage.input_tokens = usage.input_tokens.max(prompt_tokens);
    }
    if let Some(cached_tokens) = delta.cached_token_count {
        usage.input_tokens_details = Some(InputTokensDetails { cached_tokens });
    }
    if let Some(reasoning_tokens) = delta.reasoning_tokens {
        usage.output_tokens_details = Some(OutputTokensDetails { reasoning_tokens });
    }
    if let Some(generation_len) = delta.generation_len {
        let reasoning_tokens = usage
            .output_tokens_details
            .as_ref()
            .map(|details| details.reasoning_tokens)
            .unwrap_or(0);
        usage.output_tokens = usage
            .output_tokens
            .max(generation_len.saturating_sub(reasoning_tokens));
    }
    usage.total_tokens = usage.input_tokens + usage.output_tokens;
}

fn append_raw_message_stream_delta(
    stream_state: &mut ResponseStreamState,
    content: &str,
    events: &mut Vec<ResponseEvent>,
) {
    if content.is_empty() {
        return;
    }

    let output_index = stream_state
        .items
        .iter()
        .rev()
        .find_map(|(index, item)| (item.item_type == "message").then_some(*index))
        .unwrap_or_else(|| {
            stream_state
                .items
                .keys()
                .next_back()
                .map(|index| index.saturating_add(1))
                .unwrap_or(0)
        });
    let item_is_new = !stream_state.items.contains_key(&output_index);
    let item_id = {
        let item = stream_state.get_or_create_item(output_index, "message", "message");
        item.accumulated_content.push_str(content);
        item.item_id.clone()
    };

    if item_is_new {
        let sequence_number = stream_state.next_sequence_number();
        let item = stream_state
            .items
            .get(&output_index)
            .expect("message item was just inserted")
            .to_skeleton();
        events.push(ResponseEvent::OutputItemAdded(OutputItemAddedEvent {
            sequence_number,
            output_index,
            item,
        }));

        let sequence_number = stream_state.next_sequence_number();
        events.push(ResponseEvent::ContentPartAdded(ContentPartAddedEvent {
            sequence_number,
            item_id: item_id.clone(),
            output_index,
            content_index: 0,
            part: ResponseContentPart::OutputText(OutputTextContent::new("")),
        }));
    }

    let sequence_number = stream_state.next_sequence_number();
    events.push(ResponseEvent::OutputTextDelta(OutputTextDeltaEvent {
        sequence_number,
        item_id,
        output_index,
        content_index: 0,
        delta: content.to_string(),
        logprobs: Vec::new(),
    }));
}

fn process_state_event_for_streaming(
    event: &ResponseStateEvent,
    stream_state: &mut ResponseStreamState,
    events: &mut Vec<ResponseEvent>,
) {
    let item_type = if event.item_type.is_empty() {
        "message"
    } else {
        event.item_type.as_str()
    };

    let output_index = event.output_index;
    let identifier = event.identifier.clone();

    if item_type == "tool_call" && identifier == "arguments" {
        if event.event_type == "content_delta" {
            let item_id = {
                let item = stream_state.get_or_create_item(output_index, item_type, &identifier);
                item.accumulated_arguments.push_str(&event.delta);
                item.item_id.clone()
            };
            let sequence_number = stream_state.next_sequence_number();
            events.push(ResponseEvent::FunctionCallArgumentsDelta(
                FunctionCallArgumentsDeltaEvent {
                    sequence_number,
                    item_id,
                    output_index,
                    delta: event.delta.clone(),
                    field_path: None,
                },
            ));
        } else if event.event_type == "item_completed" {
            if let Some(value) = &event.value {
                let item = stream_state.get_or_create_item(output_index, item_type, &identifier);
                item.accumulated_arguments = value_to_string(value);
            }
        }
        return;
    }

    if item_type == "tool_call"
        && event.event_type == "content_delta"
        && !identifier.is_empty()
        && !identifier.starts_with("tool_call:")
    {
        let item_id = {
            let item = stream_state.get_or_create_item(output_index, item_type, &identifier);
            item.item_id.clone()
        };
        let sequence_number = stream_state.next_sequence_number();
        events.push(ResponseEvent::FunctionCallArgumentsDelta(
            FunctionCallArgumentsDeltaEvent {
                sequence_number,
                item_id,
                output_index,
                delta: event.delta.clone(),
                field_path: Some(identifier),
            },
        ));
        return;
    }

    if item_type == "tool_call" && !identifier.is_empty() && !identifier.starts_with("tool_call:") {
        return;
    }

    if event.event_type == "item_started" {
        let (item_id, skeleton_item) = {
            let item = stream_state.get_or_create_item(output_index, item_type, &identifier);
            (item.item_id.clone(), item.to_skeleton())
        };
        let sequence_number = stream_state.next_sequence_number();
        events.push(ResponseEvent::OutputItemAdded(OutputItemAddedEvent {
            sequence_number,
            output_index,
            item: skeleton_item,
        }));

        if item_type == "message" {
            let sequence_number = stream_state.next_sequence_number();
            events.push(ResponseEvent::ContentPartAdded(ContentPartAddedEvent {
                sequence_number,
                item_id,
                output_index,
                content_index: 0,
                part: ResponseContentPart::OutputText(OutputTextContent::new("")),
            }));
        }
        return;
    }

    if event.event_type == "content_delta" {
        let item_id = {
            let item = stream_state.get_or_create_item(output_index, item_type, &identifier);
            item.accumulated_content.push_str(&event.delta);
            item.item_id.clone()
        };
        if item_type == "message" {
            let sequence_number = stream_state.next_sequence_number();
            events.push(ResponseEvent::OutputTextDelta(OutputTextDeltaEvent {
                sequence_number,
                item_id,
                output_index,
                content_index: 0,
                delta: event.delta.clone(),
                logprobs: Vec::new(),
            }));
        } else if item_type == "reasoning" {
            let sequence_number = stream_state.next_sequence_number();
            events.push(ResponseEvent::ReasoningDelta(ReasoningDeltaEvent {
                sequence_number,
                item_id,
                output_index,
                content_index: 0,
                delta: event.delta.clone(),
            }));
        }
        return;
    }

    if event.event_type == "item_completed" {
        let (item_id, accumulated_content, completed_item) = {
            let item = stream_state.get_or_create_item(output_index, item_type, &identifier);
            if let Some(value) = &event.value {
                if item_type == "tool_call" {
                    if let Some((function_name, arguments)) =
                        parse_tool_call_completion_value(value)
                    {
                        item.function_name = Some(function_name);
                        item.accumulated_arguments = arguments;
                    }
                } else {
                    item.accumulated_content = value_to_string(value);
                }
            }
            if item_type == "reasoning" {
                item.accumulated_content = item.accumulated_content.trim().to_string();
            }
            item.status = OutputStatus::Completed;
            if item_type == "tool_call" && item.function_name.is_none() {
                item.function_name = Some(identifier.trim_start_matches("tool_call:").to_string());
            }
            (
                item.item_id.clone(),
                item.accumulated_content.clone(),
                item.to_completed(),
            )
        };

        if item_type == "message" {
            let sequence_number = stream_state.next_sequence_number();
            events.push(ResponseEvent::OutputTextDone(OutputTextDoneEvent {
                sequence_number,
                item_id: item_id.clone(),
                output_index,
                content_index: 0,
                text: accumulated_content.clone(),
                logprobs: Vec::new(),
            }));

            let sequence_number = stream_state.next_sequence_number();
            events.push(ResponseEvent::ContentPartDone(ContentPartDoneEvent {
                sequence_number,
                item_id: item_id.clone(),
                output_index,
                content_index: 0,
                part: ResponseContentPart::OutputText(OutputTextContent::new(accumulated_content)),
            }));
        } else if item_type == "reasoning" {
            let sequence_number = stream_state.next_sequence_number();
            events.push(ResponseEvent::ReasoningDone(ReasoningDoneEvent {
                sequence_number,
                item_id,
                output_index,
                content_index: 0,
                text: accumulated_content,
            }));
        } else if item_type == "tool_call" {
            let arguments = {
                let item = stream_state.get_or_create_item(output_index, item_type, &identifier);
                item.accumulated_arguments.clone()
            };
            let sequence_number = stream_state.next_sequence_number();
            events.push(ResponseEvent::FunctionCallArgumentsDone(
                FunctionCallArgumentsDoneEvent {
                    sequence_number,
                    item_id: item_id.clone(),
                    output_index,
                    arguments,
                },
            ));
        }

        let sequence_number = stream_state.next_sequence_number();
        events.push(ResponseEvent::OutputItemDone(OutputItemDoneEvent {
            sequence_number,
            output_index,
            item: completed_item,
        }));
    }
}

fn emit_stream_fallback_item_done(
    stream_state: &mut ResponseStreamState,
    events: &mut Vec<ResponseEvent>,
    complete_tool_calls: bool,
) {
    let indexes = stream_state
        .items
        .iter()
        .filter_map(|(index, item)| {
            if item.status != OutputStatus::Completed {
                Some(*index)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    for output_index in indexes {
        if let Some(item) = stream_state.items.get_mut(&output_index) {
            if item.item_type == "tool_call" && !complete_tool_calls {
                continue;
            }

            let item_type = item.item_type.clone();
            if item_type == "reasoning" {
                item.accumulated_content = item.accumulated_content.trim().to_string();
            }
            item.status = OutputStatus::Completed;
            let item_id = item.item_id.clone();
            let accumulated_content = item.accumulated_content.clone();
            let accumulated_arguments = item.accumulated_arguments.clone();
            let completed_item = item.to_completed();

            if item_type == "message" {
                let sequence_number = stream_state.next_sequence_number();
                events.push(ResponseEvent::OutputTextDone(OutputTextDoneEvent {
                    sequence_number,
                    item_id: item_id.clone(),
                    output_index,
                    content_index: 0,
                    text: accumulated_content.clone(),
                    logprobs: Vec::new(),
                }));
                let sequence_number = stream_state.next_sequence_number();
                events.push(ResponseEvent::ContentPartDone(ContentPartDoneEvent {
                    sequence_number,
                    item_id: item_id.clone(),
                    output_index,
                    content_index: 0,
                    part: ResponseContentPart::OutputText(OutputTextContent::new(
                        accumulated_content,
                    )),
                }));
            } else if item_type == "tool_call" {
                let sequence_number = stream_state.next_sequence_number();
                events.push(ResponseEvent::FunctionCallArgumentsDone(
                    FunctionCallArgumentsDoneEvent {
                        sequence_number,
                        item_id: item_id.clone(),
                        output_index,
                        arguments: accumulated_arguments,
                    },
                ));
            } else if item_type == "reasoning" {
                let sequence_number = stream_state.next_sequence_number();
                events.push(ResponseEvent::ReasoningDone(ReasoningDoneEvent {
                    sequence_number,
                    item_id: item_id.clone(),
                    output_index,
                    content_index: 0,
                    text: accumulated_content,
                }));
            }

            let sequence_number = stream_state.next_sequence_number();
            events.push(ResponseEvent::OutputItemDone(OutputItemDoneEvent {
                sequence_number,
                output_index,
                item: completed_item,
            }));
        }
    }
}

async fn stream_response_events(
    mut delta_rx: mpsc::UnboundedReceiver<ResponseDelta>,
    event_tx: mpsc::Sender<ResponseEvent>,
    response_id: String,
    model: String,
    stream_tokens: bool,
) {
    let mut stream_state = ResponseStreamState::new(response_id, model);

    if event_tx
        .send(ResponseEvent::ResponseCreated(ResponseCreatedEvent {
            sequence_number: stream_state.next_sequence_number(),
            response: stream_state.snapshot(),
        }))
        .await
        .is_err()
    {
        return;
    }

    if event_tx
        .send(ResponseEvent::ResponseInProgress(ResponseInProgressEvent {
            sequence_number: stream_state.next_sequence_number(),
            response: stream_state.snapshot(),
        }))
        .await
        .is_err()
    {
        return;
    }

    let mut error_detail: Option<String> = None;
    let mut finish_reason: Option<String> = None;
    let mut usage = ResponseUsage::default();
    let mut pending_raw_content = String::new();

    while let Some(delta) = delta_rx.recv().await {
        if let Some(error) = &delta.error {
            error_detail = Some(error.clone());
            break;
        }

        if stream_tokens {
            for token_id in &delta.tokens {
                let sequence_number = stream_state.next_sequence_number();
                if event_tx
                    .send(ResponseEvent::OutputToken(OutputTokenEvent {
                        sequence_number,
                        token_id: *token_id,
                        content: delta.content.clone(),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
        }

        let mut mapped_events = Vec::new();
        if delta.state_events.is_empty() {
            if let Some(content) = delta.content.as_deref() {
                if stream_state
                    .items
                    .values()
                    .any(|item| item.item_type == "message")
                {
                    append_raw_message_stream_delta(&mut stream_state, content, &mut mapped_events);
                } else {
                    pending_raw_content.push_str(content);
                }
            }
        } else {
            for event in &delta.state_events {
                process_state_event_for_streaming(event, &mut stream_state, &mut mapped_events);
            }
        }

        for event in mapped_events {
            if event_tx.send(event).await.is_err() {
                return;
            }
        }

        update_usage_from_delta(&delta, &mut usage);
        if usage.total_tokens > 0 {
            stream_state.usage = Some(usage.clone());
        }

        if let Some(reason) = &delta.finish_reason {
            finish_reason = Some(reason.to_lowercase());
        }

        if let Some(id) = delta.matched_stop_token_id {
            stream_state.stop_token_id = Some(id);
        }
        if let Some(tok) = &delta.matched_stop_token {
            stream_state.stop_token = Some(tok.clone());
        }

        if delta.is_final_delta {
            break;
        }
    }

    if let Some(detail) = error_detail {
        stream_state.status = OutputStatus::Failed;

        if event_tx
            .send(ResponseEvent::ResponseFailed(ResponseFailedEvent {
                sequence_number: stream_state.next_sequence_number(),
                response: stream_state.snapshot(),
            }))
            .await
            .is_err()
        {
            return;
        }

        if event_tx
            .send(ResponseEvent::Error(StreamErrorEvent {
                sequence_number: stream_state.next_sequence_number(),
                error: StreamErrorDetail {
                    message: detail,
                    error_type: "inference_error".to_string(),
                },
            }))
            .await
            .is_err()
        {
            return;
        }
    } else {
        stream_state.completed_at = Some(current_timestamp());

        let incomplete_details = finish_reason_to_incomplete(finish_reason.as_deref());
        let mut completion_events = Vec::new();
        if !pending_raw_content.is_empty() && stream_state.items.is_empty() {
            append_raw_message_stream_delta(
                &mut stream_state,
                &pending_raw_content,
                &mut completion_events,
            );
        }
        emit_stream_fallback_item_done(
            &mut stream_state,
            &mut completion_events,
            incomplete_details.is_none(),
        );
        for event in completion_events {
            if event_tx.send(event).await.is_err() {
                return;
            }
        }

        if let Some(incomplete_details) = incomplete_details {
            stream_state.status = OutputStatus::Incomplete;
            stream_state.incomplete_details = Some(incomplete_details);
            if event_tx
                .send(ResponseEvent::ResponseIncomplete(ResponseIncompleteEvent {
                    sequence_number: stream_state.next_sequence_number(),
                    response: stream_state.snapshot(),
                }))
                .await
                .is_err()
            {
                return;
            }
        } else {
            stream_state.status = OutputStatus::Completed;
            if event_tx
                .send(ResponseEvent::ResponseCompleted(ResponseCompletedEvent {
                    sequence_number: stream_state.next_sequence_number(),
                    response: stream_state.snapshot(),
                }))
                .await
                .is_err()
            {
                return;
            }
        }
    }

    let _ = event_tx.send(ResponseEvent::Done).await;
}

async fn gather_non_streaming_response(
    mut delta_rx: mpsc::UnboundedReceiver<ResponseDelta>,
    model: &str,
    request: &ResponsesRequest,
    raw_tool_call_tokens: Option<&ToolCallingTokens>,
) -> Result<ResponseObject> {
    let created_at = current_timestamp();
    let mut completed_at: Option<i64> = None;
    let mut output_items: BTreeMap<u32, AggregatedOutputItem> = BTreeMap::new();
    let mut fallback_content = String::new();
    let mut usage = ResponseUsage::default();
    let mut error_detail: Option<String> = None;
    let mut finish_reason: Option<String> = None;
    let mut stop_token_id: Option<i32> = None;
    let mut stop_token: Option<String> = None;

    while let Some(delta) = delta_rx.recv().await {
        if let Some(error) = &delta.error {
            error_detail = Some(error.clone());
        }

        if delta.state_events.is_empty() {
            if let Some(content) = &delta.content {
                if output_items.is_empty() {
                    fallback_content.push_str(content);
                } else {
                    append_raw_message_output(&mut output_items, content);
                }
            }
        } else {
            for event in &delta.state_events {
                process_state_event_for_output(event, &mut output_items);
            }
        }

        update_usage_from_delta(&delta, &mut usage);

        if let Some(reason) = &delta.finish_reason {
            finish_reason = Some(reason.to_lowercase());
        }

        if let Some(id) = delta.matched_stop_token_id {
            stop_token_id = Some(id);
        }
        if let Some(tok) = &delta.matched_stop_token {
            stop_token = Some(tok.clone());
        }

        if delta.is_final_delta {
            completed_at = Some(current_timestamp());
            break;
        }
    }

    if let Some(error) = error_detail {
        return Err(ClientError::RequestFailed(error));
    }

    let incomplete_details = finish_reason_to_incomplete(finish_reason.as_deref());
    let output = if output_items.is_empty() && !fallback_content.is_empty() {
        vec![ResponseOutputItem::Message(OutputMessage {
            output_type: "message".to_string(),
            id: generate_message_id(),
            status: OutputStatus::Completed,
            role: "assistant".to_string(),
            content: vec![OutputTextContent::new(fallback_content)],
        })]
    } else {
        build_output_items(&output_items, raw_tool_call_tokens)
    };

    Ok(ResponseObject {
        id: generate_response_id(),
        object: "response".to_string(),
        created_at,
        completed_at,
        status: if incomplete_details.is_some() {
            OutputStatus::Incomplete
        } else {
            OutputStatus::Completed
        },
        incomplete_details,
        error: None,
        model: model.to_string(),
        output,
        usage: Some(usage),
        metadata: request.metadata.clone(),
        parallel_tool_calls: request.parallel_tool_calls,
        temperature: request.temperature,
        top_p: request.top_p,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        top_k: request.top_k,
        min_p: request.min_p,
        instructions: request.instructions.clone(),
        max_output_tokens: request.max_output_tokens,
        top_logprobs: request.top_logprobs,
        tool_choice: request.tool_choice.clone(),
        tools: request.core_tools.clone(),
        max_tool_calls: request.max_tool_calls,
        min_tool_calls: request.min_tool_calls,
        text: request.text.clone(),
        stop_token_id,
        stop_token,
    })
}

impl Client {
    /// Perform an asynchronous Responses API request.
    pub async fn aresponses(
        &self,
        model_id: &str,
        request: ResponsesRequest,
    ) -> Result<ResponsesResult> {
        let info = self.registry.ensure_loaded(model_id).await?;
        let formatter = info.require_formatter()?;
        let request_model_id = info.model_id.as_str();
        let request_id = self.ipc.next_request_id();
        tracing::debug!(
            request_id,
            model_id = %model_id,
            stream = request.stream,
            "Building responses request"
        );
        let messages = request.to_messages();
        tracing::trace!(
            request_id,
            model_id = %model_id,
            messages = ?messages,
            "Responses messages before multimodal expansion"
        );
        let request_reasoning_effort = request
            .reasoning_effort
            .clone()
            .or_else(|| request.reasoning.as_ref().and_then(ReasoningConfig::effort));
        let explicit_reasoning_false = request
            .reasoning
            .as_ref()
            .and_then(ReasoningConfig::explicit_bool)
            == Some(false);
        let default_reasoning = formatter.supports_native_thinking()
            && request.reasoning.is_none()
            && request_reasoning_effort.is_none();
        let requested_reasoning =
            !explicit_reasoning_false && (request_reasoning_effort.is_some() || default_reasoning);
        let (reasoning_flag, reasoning_effort, thinking_tokens) =
            native_reasoning_settings(formatter, requested_reasoning, &request_reasoning_effort);

        let (messages_for_template, image_buffers, audio_buffers, capabilities, content_order) =
            build_multimodal_messages(formatter, &messages, request.instructions.as_deref())
                .map_err(|e| ClientError::Multimodal(e.to_string()))?;

        if messages_for_template.is_empty() {
            return Err(ClientError::RequestFailed(
                "Response request must include at least one content segment.".into(),
            ));
        }
        tracing::trace!(
            request_id,
            model_id = %model_id,
            messages_for_template = ?messages_for_template,
            "Responses messages after multimodal expansion"
        );

        let (tool_schemas, active_tool_schemas) =
            response_tool_schemas(&request.core_tools, &request.active_tools);
        let tool_schemas_json = if tool_schemas.is_empty() {
            String::new()
        } else {
            serde_json::to_string(&tool_schemas).unwrap_or_default()
        };
        let active_tool_schemas_json = if active_tool_schemas.is_empty() {
            String::new()
        } else {
            serde_json::to_string(&active_tool_schemas).unwrap_or_default()
        };
        let tool_schemas_chars = tool_schemas_json.chars().count();
        let template_tools = (!tool_schemas.is_empty()).then_some(tool_schemas.as_slice());

        let prompt_text = formatter
            .apply_template_with_tools(
                &messages_for_template,
                true,
                reasoning_flag,
                None,
                reasoning_effort.as_deref(),
                template_tools,
            )
            .map_err(|e| ClientError::Formatter(e.to_string()))?;

        let layout_segments = build_multimodal_layout(
            formatter,
            &prompt_text,
            &image_buffers,
            &audio_buffers,
            &capabilities,
            &content_order,
        )
        .map_err(|e| ClientError::Multimodal(e.to_string()))?;

        let final_prompt = formatter.strip_template_placeholders(&prompt_text);
        tracing::debug!(
            request_id,
            model_id = %model_id,
            prompt_chars = final_prompt.chars().count(),
            image_count = image_buffers.len(),
            capability_count = capabilities.len(),
            layout_segment_count = layout_segments.len(),
            tool_schema_chars = tool_schemas_chars,
            message_count = messages_for_template.len(),
            "Prepared responses prompt payload"
        );
        let response_format_json = request
            .text
            .as_ref()
            .map(|text| text.get("format").cloned().unwrap_or_else(|| text.clone()))
            .map(normalize_response_format)
            .map(|response_format| serde_json::to_string(&response_format).unwrap_or_default())
            .unwrap_or_default();

        let rng_seed = rand::thread_rng().gen::<u64>();
        let temperature = request.temperature.unwrap_or_else(|| {
            formatter
                .generation_default_f64("temperature")
                .unwrap_or(1.0)
        });
        let top_p = request
            .top_p
            .unwrap_or_else(|| formatter.generation_default_f64("top_p").unwrap_or(1.0));
        let top_k = request
            .top_k
            .unwrap_or_else(|| formatter.generation_default_i32("top_k").unwrap_or(-1));
        let min_p = request
            .min_p
            .unwrap_or_else(|| formatter.generation_default_f64("min_p").unwrap_or(0.0));
        let frequency_penalty = request.frequency_penalty.unwrap_or_else(|| {
            formatter
                .generation_default_f64("frequency_penalty")
                .unwrap_or(0.0)
        });
        let presence_penalty = request.presence_penalty.unwrap_or_else(|| {
            formatter
                .generation_default_f64("presence_penalty")
                .unwrap_or(0.0)
        });
        let repetition_penalty = formatter
            .generation_default_f64("repetition_penalty")
            .unwrap_or(1.0);
        let repetition_context_size = formatter
            .generation_default_i32("repetition_context_size")
            .unwrap_or(60);
        let prompt_payload = PromptPayload {
            prompt: final_prompt,
            image_buffers,
            audio_buffers,
            capabilities: super::convert_capabilities(&capabilities),
            layout: super::convert_layout(&layout_segments),
            max_generated_tokens: request.max_output_tokens.unwrap_or(0),
            temperature,
            top_p,
            top_k,
            min_p,
            rng_seed,
            deterministic: request.deterministic,
            stop_sequences: if formatter.control_tokens.end_of_sequence.is_empty() {
                Vec::new()
            } else {
                vec![formatter.control_tokens.end_of_sequence.clone()]
            },
            num_candidates: 1,
            best_of: Some(1),
            final_candidates: Some(1),
            frequency_penalty,
            presence_penalty,
            repetition_penalty,
            repetition_context_size,
            top_logprobs: request.top_logprobs.unwrap_or(0),
            logit_bias: HashMap::new(),
            tool_schemas_json,
            active_tool_schemas_json,
            tool_calling_tokens: formatter.get_tool_calling_tokens().clone(),
            output_frame_tokens: formatter.get_output_frame_tokens().clone(),
            thinking_tokens,
            tool_choice: tool_choice_to_string(request.tool_choice.as_ref()),
            min_tool_calls: request.min_tool_calls.unwrap_or(1).max(1),
            max_tool_calls: request.max_tool_calls.unwrap_or(0).max(0),
            response_format_json,
            modal_options_json: String::new(),
            task_name: None,
            reasoning_effort,
            prefix_cache: request.prefix_cache,
        };
        tracing::debug!(
            request_id,
            model_id = %model_id,
            stream = request.stream,
            "Dispatching responses request to PIE"
        );
        let (_batch_size, stream) = self.ipc.send_batch_request(
            request_id,
            request_model_id,
            &info.model_path,
            &[prompt_payload],
        )?;

        if request.stream {
            let (event_tx, event_rx) = mpsc::channel(256);
            tokio::spawn(stream_response_events(
                stream,
                event_tx,
                generate_response_id(),
                model_id.to_string(),
                request.stream_tokens,
            ));
            Ok(ResponsesResult::Stream {
                request_id,
                events: event_rx,
            })
        } else {
            let raw_tool_call_tokens = if request.core_tools.is_empty()
                || tool_choice_to_string(request.tool_choice.as_ref()) == "none"
            {
                None
            } else {
                Some(formatter.get_tool_calling_tokens())
            };
            let response =
                gather_non_streaming_response(stream, model_id, &request, raw_tool_call_tokens)
                    .await?;
            Ok(ResponsesResult::Complete(Box::new(response)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_event_type_names() {
        let snapshot = ResponseSnapshot {
            id: "resp_1".to_string(),
            object: "response".to_string(),
            created_at: 1,
            completed_at: None,
            status: OutputStatus::InProgress,
            incomplete_details: None,
            model: "model".to_string(),
            output: Vec::new(),
            usage: None,
            stop_token_id: None,
            stop_token: None,
        };

        let event = ResponseEvent::ResponseCreated(ResponseCreatedEvent {
            sequence_number: 0,
            response: snapshot,
        });
        assert_eq!(event.event_type(), "response.created");
        assert_eq!(ResponseEvent::Done.event_type(), "done");
    }

    #[test]
    fn test_finish_reason_to_incomplete() {
        assert_eq!(
            finish_reason_to_incomplete(Some("length")),
            Some(IncompleteDetails {
                reason: "max_output_tokens".to_string(),
            })
        );
        assert_eq!(
            finish_reason_to_incomplete(Some("content_filter")),
            Some(IncompleteDetails {
                reason: "content_filter".to_string(),
            })
        );
        assert_eq!(
            finish_reason_to_incomplete(Some("user")),
            Some(IncompleteDetails {
                reason: "cancelled".to_string(),
            })
        );
        assert_eq!(finish_reason_to_incomplete(Some("stop")), None);
    }

    #[test]
    fn test_response_tool_schemas_keep_core_raw_and_wrap_active() {
        let weather_tool = serde_json::json!({
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        });

        let (core_tools, active_tools) =
            response_tool_schemas(std::slice::from_ref(&weather_tool), &[]);

        assert_eq!(
            core_tools,
            vec![serde_json::json!({
                "name": "get_weather",
                "type": "function",
                "description": "Get the current weather for a location.",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            })]
        );
        assert_eq!(
            active_tools,
            vec![serde_json::json!({
                "name": "get_weather",
                "type": "object",
                "description": "Get the current weather for a location.",
                "properties": {
                    "name": {"const": "get_weather"},
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                },
                "strict": true,
                "required": ["name", "arguments"]
            })]
        );
    }

    #[test]
    fn test_request_input_conversion_string() {
        let request = ResponsesRequest::from_text("hello");
        let messages = request.to_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].get("role").and_then(Value::as_str),
            Some("user")
        );
    }

    #[test]
    fn test_request_input_conversion_tool_items() {
        let request = ResponsesRequest {
            input: ResponsesInput::Items(vec![
                ResponseInputItem::FunctionCall {
                    call_id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: "{\"location\":\"SF\"}".to_string(),
                },
                ResponseInputItem::FunctionCallOutput {
                    call_id: "call_1".to_string(),
                    output: "{\"temperature\":65}".into(),
                },
            ]),
            stream: false,
            instructions: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            deterministic: false,
            frequency_penalty: None,
            presence_penalty: None,
            max_output_tokens: None,
            top_logprobs: None,
            core_tools: Vec::new(),
            active_tools: Vec::new(),
            tool_choice: None,
            min_tool_calls: None,
            max_tool_calls: None,
            text: None,
            reasoning: Some(false.into()),
            reasoning_effort: None,
            metadata: None,
            parallel_tool_calls: false,
            prefix_cache: None,
            stream_tokens: false,
        };

        let messages = request.to_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0].get("role").and_then(Value::as_str),
            Some("assistant")
        );
        assert_eq!(
            messages[1].get("role").and_then(Value::as_str),
            Some("tool")
        );
    }

    #[test]
    fn test_request_input_conversion_multimodal_tool_output() {
        let request = ResponsesRequest {
            input: ResponsesInput::Items(vec![ResponseInputItem::FunctionCallOutput {
                call_id: "call_image".to_string(),
                output: FunctionCallOutputContent::Content(vec![serde_json::json!({
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AA==",
                    "detail": "auto"
                })]),
            }]),
            stream: false,
            instructions: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            deterministic: false,
            frequency_penalty: None,
            presence_penalty: None,
            max_output_tokens: None,
            top_logprobs: None,
            core_tools: Vec::new(),
            active_tools: Vec::new(),
            tool_choice: None,
            min_tool_calls: None,
            max_tool_calls: None,
            text: None,
            reasoning: Some(false.into()),
            reasoning_effort: None,
            metadata: None,
            parallel_tool_calls: false,
            prefix_cache: None,
            stream_tokens: false,
        };

        let messages = request.to_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].get("role").and_then(Value::as_str),
            Some("tool")
        );
        assert_eq!(
            messages[0].get("content"),
            Some(&serde_json::json!([{
                "type": "input_image",
                "image_url": "data:image/png;base64,AA==",
                "detail": "auto"
            }]))
        );
        assert_eq!(
            messages[0].get("tool_call_id").and_then(Value::as_str),
            Some("call_image")
        );
    }

    #[test]
    fn test_response_delta_state_event_deserialization_shape() {
        let json = serde_json::json!({
            "request_id": 1,
            "is_final_delta": false,
            "state_events": [
                {
                    "event_type": "item_started",
                    "item_type": "message",
                    "output_index": 0,
                    "identifier": "",
                    "delta": ""
                }
            ]
        });
        let parsed: ResponseDelta = serde_json::from_value(json).expect("deserialize failed");
        assert_eq!(parsed.state_events.len(), 1);
        assert_eq!(parsed.state_events[0].event_type, "item_started");
    }

    #[test]
    fn test_build_output_items_uses_structured_tool_call_completion_value() {
        let mut output_items = BTreeMap::new();

        process_state_event_for_output(
            &ResponseStateEvent {
                event_type: "content_delta".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "arguments".to_string(),
                delta: r#"location="Tokyo", verbose=True"#.to_string(),
                value: None,
            },
            &mut output_items,
        );
        process_state_event_for_output(
            &ResponseStateEvent {
                event_type: "item_completed".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "tool_call:get_weather".to_string(),
                delta: String::new(),
                value: Some(Value::String(
                    serde_json::json!({
                        "name": "get_weather",
                        "arguments": {
                            "location": "Tokyo",
                            "verbose": true,
                            "limit": null,
                        },
                    })
                    .to_string(),
                )),
            },
            &mut output_items,
        );

        let output = build_output_items(&output_items, None);
        let ResponseOutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call output");
        };

        assert_eq!(call.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).expect("valid JSON arguments"),
            serde_json::json!({
                "location": "Tokyo",
                "verbose": true,
                "limit": null,
            })
        );
    }

    #[test]
    fn test_build_output_items_promotes_raw_tool_call_message() {
        let output_items = BTreeMap::from([(
            0,
            AggregatedOutputItem {
                item_type: "message".to_string(),
                content: r#"{"name":"get_weather","arguments":{"location":"San Francisco"}}"#
                    .to_string()
                    + "<|tool_call_end|>",
                arguments: String::new(),
                identifier: "message".to_string(),
                function_name: String::new(),
            },
        )]);
        let tokens = ToolCallingTokens {
            formats: vec![crate::ipc::serialization::ToolCallFormat {
                name: "json".to_string(),
                call_end: "<|tool_call_end|>".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };

        let output = build_output_items(&output_items, Some(&tokens));
        let ResponseOutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected raw tool call message to be promoted");
        };

        assert_eq!(call.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).expect("valid JSON arguments"),
            serde_json::json!({"location": "San Francisco"})
        );
    }

    #[test]
    fn test_build_output_items_preserves_parallel_tool_calls() {
        let mut output_items = BTreeMap::new();
        for (output_index, name, arguments) in [
            (1, "get_weather", serde_json::json!({"location": "Tokyo"})),
            (2, "set_alarm", serde_json::json!({"time": "7am"})),
        ] {
            process_state_event_for_output(
                &ResponseStateEvent {
                    event_type: "item_completed".to_string(),
                    item_type: "tool_call".to_string(),
                    output_index,
                    identifier: format!("tool_call:{name}"),
                    delta: String::new(),
                    value: Some(Value::String(
                        serde_json::json!({
                            "name": name,
                            "arguments": arguments,
                        })
                        .to_string(),
                    )),
                },
                &mut output_items,
            );
        }

        let output = build_output_items(&output_items, None);
        let calls = output
            .iter()
            .filter_map(|item| match item {
                ResponseOutputItem::FunctionCall(call) => Some(call.name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(calls, vec!["get_weather", "set_alarm"]);
    }

    #[test]
    fn test_streaming_tool_call_done_uses_structured_completion_value() {
        let mut stream_state = ResponseStreamState::new("resp_1".to_string(), "model".to_string());
        let mut events = Vec::new();

        process_state_event_for_streaming(
            &ResponseStateEvent {
                event_type: "item_started".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "tool_call:get_weather".to_string(),
                delta: String::new(),
                value: None,
            },
            &mut stream_state,
            &mut events,
        );
        process_state_event_for_streaming(
            &ResponseStateEvent {
                event_type: "content_delta".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "location".to_string(),
                delta: "Tokyo".to_string(),
                value: None,
            },
            &mut stream_state,
            &mut events,
        );
        process_state_event_for_streaming(
            &ResponseStateEvent {
                event_type: "item_completed".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "location".to_string(),
                delta: String::new(),
                value: Some(Value::String("Tokyo".to_string())),
            },
            &mut stream_state,
            &mut events,
        );
        process_state_event_for_streaming(
            &ResponseStateEvent {
                event_type: "content_delta".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "arguments".to_string(),
                delta: r#"location="Tokyo", verbose=True"#.to_string(),
                value: None,
            },
            &mut stream_state,
            &mut events,
        );
        process_state_event_for_streaming(
            &ResponseStateEvent {
                event_type: "item_completed".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "arguments".to_string(),
                delta: String::new(),
                value: Some(Value::String(
                    r#"location="Tokyo", verbose=True"#.to_string(),
                )),
            },
            &mut stream_state,
            &mut events,
        );
        process_state_event_for_streaming(
            &ResponseStateEvent {
                event_type: "item_completed".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "tool_call:get_weather".to_string(),
                delta: String::new(),
                value: Some(Value::String(
                    serde_json::json!({
                        "name": "get_weather",
                        "arguments": {
                            "location": "Tokyo",
                            "verbose": true,
                        },
                    })
                    .to_string(),
                )),
            },
            &mut stream_state,
            &mut events,
        );

        let argument_done_events = events
            .iter()
            .filter_map(|event| match event {
                ResponseEvent::FunctionCallArgumentsDone(done) => Some(done),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(argument_done_events.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&argument_done_events[0].arguments)
                .expect("valid JSON arguments"),
            serde_json::json!({
                "location": "Tokyo",
                "verbose": true,
            })
        );

        let completed_calls = events
            .iter()
            .filter_map(|event| match event {
                ResponseEvent::OutputItemDone(done) => match &done.item {
                    ResponseOutputItem::FunctionCall(call) => Some(call),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(completed_calls.len(), 1);
        assert_eq!(completed_calls[0].name, "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(&completed_calls[0].arguments)
                .expect("valid JSON arguments"),
            serde_json::json!({
                "location": "Tokyo",
                "verbose": true,
            })
        );
    }

    #[test]
    fn test_harmony_reasoning_state_events_stream_deltas() {
        let mut stream_state =
            ResponseStreamState::new("resp_1".to_string(), "gpt-oss-test".to_string());
        let mut events = Vec::new();

        for event in [
            ResponseStateEvent {
                event_type: "item_started".to_string(),
                item_type: "reasoning".to_string(),
                output_index: 0,
                identifier: "reasoning".to_string(),
                delta: String::new(),
                value: None,
            },
            ResponseStateEvent {
                event_type: "content_delta".to_string(),
                item_type: "reasoning".to_string(),
                output_index: 0,
                identifier: "reasoning".to_string(),
                delta: "First thought. ".to_string(),
                value: None,
            },
            ResponseStateEvent {
                event_type: "content_delta".to_string(),
                item_type: "reasoning".to_string(),
                output_index: 0,
                identifier: "reasoning".to_string(),
                delta: "Second thought.".to_string(),
                value: None,
            },
            ResponseStateEvent {
                event_type: "item_completed".to_string(),
                item_type: "reasoning".to_string(),
                output_index: 0,
                identifier: "reasoning".to_string(),
                delta: String::new(),
                value: Some(Value::String("First thought. Second thought.".to_string())),
            },
        ] {
            process_state_event_for_streaming(&event, &mut stream_state, &mut events);
        }

        let reasoning_deltas = events
            .iter()
            .filter_map(|event| match event {
                ResponseEvent::ReasoningDelta(delta) => Some(delta.delta.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(reasoning_deltas, vec!["First thought. ", "Second thought."]);

        let reasoning_done = events
            .iter()
            .find_map(|event| match event {
                ResponseEvent::ReasoningDone(done) => Some(done),
                _ => None,
            })
            .expect("reasoning done event");
        assert_eq!(reasoning_done.text, "First thought. Second thought.");

        let output_item_done = events
            .iter()
            .find_map(|event| match event {
                ResponseEvent::OutputItemDone(done) => match &done.item {
                    ResponseOutputItem::Reasoning(reasoning) => Some(reasoning),
                    _ => None,
                },
                _ => None,
            })
            .expect("reasoning output item done");
        assert_eq!(
            output_item_done.content[0].text,
            "First thought. Second thought."
        );
    }

    #[test]
    fn test_streaming_incomplete_tool_call_is_not_synthesized_done() {
        let mut stream_state = ResponseStreamState::new("resp_1".to_string(), "model".to_string());
        let mut events = Vec::new();

        process_state_event_for_streaming(
            &ResponseStateEvent {
                event_type: "item_started".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "tool_call:share_to_party".to_string(),
                delta: String::new(),
                value: None,
            },
            &mut stream_state,
            &mut events,
        );
        process_state_event_for_streaming(
            &ResponseStateEvent {
                event_type: "content_delta".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "arguments".to_string(),
                delta: r#"{"content":"hel"#.to_string(),
                value: None,
            },
            &mut stream_state,
            &mut events,
        );

        let mut fallback_events = Vec::new();
        emit_stream_fallback_item_done(&mut stream_state, &mut fallback_events, false);

        assert!(!fallback_events.iter().any(|event| {
            matches!(
                event,
                ResponseEvent::FunctionCallArgumentsDone(_)
                    | ResponseEvent::OutputItemDone(OutputItemDoneEvent {
                        item: ResponseOutputItem::FunctionCall(_),
                        ..
                    })
            )
        }));
        assert_eq!(stream_state.items[&0].status, OutputStatus::InProgress);
    }

    #[test]
    fn test_streaming_tool_argument_field_delta_preserves_semantic_field_path() {
        let mut stream_state = ResponseStreamState::new("resp_1".to_string(), "model".to_string());
        let mut events = Vec::new();

        process_state_event_for_streaming(
            &ResponseStateEvent {
                event_type: "item_started".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "tool_call:share_to_party".to_string(),
                delta: String::new(),
                value: None,
            },
            &mut stream_state,
            &mut events,
        );
        process_state_event_for_streaming(
            &ResponseStateEvent {
                event_type: "content_delta".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "content".to_string(),
                delta: "Hello".to_string(),
                value: None,
            },
            &mut stream_state,
            &mut events,
        );

        let argument_delta = events
            .iter()
            .find_map(|event| match event {
                ResponseEvent::FunctionCallArgumentsDelta(delta) => Some(delta),
                _ => None,
            })
            .expect("function arguments delta");
        let started_call = events
            .iter()
            .find_map(|event| match event {
                ResponseEvent::OutputItemAdded(OutputItemAddedEvent {
                    item: ResponseOutputItem::FunctionCall(call),
                    ..
                }) => Some(call),
                _ => None,
            })
            .expect("started function call");
        assert_eq!(argument_delta.delta, "Hello");
        assert_eq!(argument_delta.item_id, started_call.id);
        assert_eq!(argument_delta.field_path.as_deref(), Some("content"));
        assert_eq!(started_call.name, "share_to_party");
    }
}
