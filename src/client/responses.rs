//! OpenAI Responses API surface for Orchard.
//!
//! This module maps PIE state-events carried over IPC into typed Responses API events.

use std::collections::{BTreeMap, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::{distributions::Alphanumeric, Rng};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use super::{tool_choice_to_string, Client, ClientError, Result};
use crate::formatter::multimodal::{build_multimodal_layout, build_multimodal_messages};
use crate::ipc::client::{ResponseDelta, ResponseStateEvent};
use crate::ipc::serialization::PromptPayload;

const RESPONSE_ID_PREFIX: &str = "resp_";
const MESSAGE_ID_PREFIX: &str = "msg_";
const FUNCTION_CALL_ID_PREFIX: &str = "fc_";
const TOOL_CALL_ID_PREFIX: &str = "call_";

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
        output: String,
    },
    Reasoning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<Vec<Value>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
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
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub max_output_tokens: Option<i32>,
    #[serde(default)]
    pub top_logprobs: Option<i32>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub max_tool_calls: Option<i32>,
    #[serde(default)]
    pub text: Option<Value>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(default)]
    pub parallel_tool_calls: bool,
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
            frequency_penalty: None,
            presence_penalty: None,
            max_output_tokens: None,
            top_logprobs: None,
            tools: Vec::new(),
            tool_choice: None,
            max_tool_calls: None,
            text: None,
            reasoning_effort: None,
            metadata: None,
            parallel_tool_calls: false,
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
                            message.insert("content".to_string(), Value::String(output.clone()));
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
    pub status: OutputStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputReasoning {
    #[serde(rename = "type")]
    pub output_type: String,
    pub id: String,
    pub status: OutputStatus,
    pub summary: Vec<ReasoningSummaryTextContent>,
    pub content: Vec<ReasoningContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
    pub status: OutputStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
    pub text: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseSnapshot {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
    pub status: OutputStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallArgumentsDeltaEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub delta: String,
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
    Stream(mpsc::Receiver<ResponseEvent>),
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
        }
    }
}

#[derive(Debug, Clone)]
struct AggregatedOutputItem {
    item_type: String,
    content: String,
    arguments: String,
    identifier: String,
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
        } else if let Some(value) = &event.value {
            item.content = value_to_string(value);
        }
    }
}

fn build_output_items(
    output_items: &BTreeMap<u32, AggregatedOutputItem>,
) -> Vec<ResponseOutputItem> {
    let mut output = Vec::new();
    for item in output_items.values() {
        match item.item_type.as_str() {
            "tool_call" => {
                let name = item.identifier.trim_start_matches("tool_call:").to_string();
                output.push(ResponseOutputItem::FunctionCall(OutputFunctionCall {
                    output_type: "function_call".to_string(),
                    id: generate_function_call_id(),
                    call_id: generate_tool_call_id(),
                    name,
                    arguments: item.arguments.clone(),
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
    if let Some(generation_len) = delta.generation_len {
        usage.output_tokens = usage.output_tokens.max(generation_len);
    }
    if let Some(cached_tokens) = delta.cached_token_count {
        usage.input_tokens_details = Some(InputTokensDetails { cached_tokens });
    }
    if let Some(reasoning_tokens) = delta.reasoning_tokens {
        usage.output_tokens_details = Some(OutputTokensDetails { reasoning_tokens });
    }
    usage.total_tokens = usage.input_tokens + usage.output_tokens;
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
                },
            ));
        } else if event.event_type == "item_completed" {
            let (item_id, arguments) = {
                let item = stream_state.get_or_create_item(output_index, item_type, &identifier);
                if let Some(value) = &event.value {
                    item.accumulated_arguments = value_to_string(value);
                }
                (item.item_id.clone(), item.accumulated_arguments.clone())
            };
            let sequence_number = stream_state.next_sequence_number();
            events.push(ResponseEvent::FunctionCallArgumentsDone(
                FunctionCallArgumentsDoneEvent {
                    sequence_number,
                    item_id,
                    output_index,
                    arguments,
                },
            ));
        }
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
        } else if item_type == "reasoning" {
            let sequence_number = stream_state.next_sequence_number();
            events.push(ResponseEvent::ContentPartAdded(ContentPartAddedEvent {
                sequence_number,
                item_id,
                output_index,
                content_index: 0,
                part: ResponseContentPart::Reasoning(ReasoningContent::new("")),
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
                if item_type != "tool_call" {
                    item.accumulated_content = value_to_string(value);
                }
            }
            item.status = OutputStatus::Completed;
            if item_type == "tool_call" {
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
            item.status = OutputStatus::Completed;
            let item_type = item.item_type.clone();
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

    while let Some(delta) = delta_rx.recv().await {
        if let Some(error) = &delta.error {
            error_detail = Some(error.clone());
            break;
        }

        let mut mapped_events = Vec::new();
        for event in &delta.state_events {
            process_state_event_for_streaming(event, &mut stream_state, &mut mapped_events);
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

        if delta.is_final_delta {
            break;
        }
    }

    let mut completion_events = Vec::new();
    emit_stream_fallback_item_done(&mut stream_state, &mut completion_events);
    for event in completion_events {
        if event_tx.send(event).await.is_err() {
            return;
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

        if let Some(incomplete_details) = finish_reason_to_incomplete(finish_reason.as_deref()) {
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
) -> Result<ResponseObject> {
    let created_at = current_timestamp();
    let mut completed_at: Option<i64> = None;
    let mut output_items: BTreeMap<u32, AggregatedOutputItem> = BTreeMap::new();
    let mut fallback_content = String::new();
    let mut usage = ResponseUsage::default();
    let mut error_detail: Option<String> = None;
    let mut finish_reason: Option<String> = None;

    while let Some(delta) = delta_rx.recv().await {
        if let Some(error) = &delta.error {
            error_detail = Some(error.clone());
        }

        if let Some(content) = &delta.content {
            fallback_content.push_str(content);
        }

        for event in &delta.state_events {
            process_state_event_for_output(event, &mut output_items);
        }

        update_usage_from_delta(&delta, &mut usage);

        if let Some(reason) = &delta.finish_reason {
            finish_reason = Some(reason.to_lowercase());
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
        build_output_items(&output_items)
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
        tools: request.tools.clone(),
        max_tool_calls: request.max_tool_calls,
        text: request.text.clone(),
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
        let request_id = self.ipc.next_request_id();
        let messages = request.to_messages();
        let reasoning_flag = request.reasoning_effort.is_some();

        let (messages_for_template, image_buffers, capabilities, content_order) =
            build_multimodal_messages(&info.formatter, &messages, request.instructions.as_deref())
                .map_err(|e| ClientError::Multimodal(e.to_string()))?;

        if messages_for_template.is_empty() {
            return Err(ClientError::RequestFailed(
                "Response request must include at least one content segment.".into(),
            ));
        }

        let tool_schemas = request
            .tools
            .iter()
            .map(normalize_response_tool_schema)
            .collect::<Vec<_>>();
        let tool_schemas_json = if tool_schemas.is_empty() {
            String::new()
        } else {
            serde_json::to_string(&tool_schemas).unwrap_or_default()
        };
        let template_tools = (!tool_schemas.is_empty()).then_some(tool_schemas.as_slice());

        let prompt_text = info
            .formatter
            .apply_template_with_tools(
                &messages_for_template,
                true,
                reasoning_flag,
                None,
                template_tools,
            )
            .map_err(|e| ClientError::Formatter(e.to_string()))?;

        let capability_placeholder = info.formatter.capability_placeholder_token();

        let layout_segments = build_multimodal_layout(
            &prompt_text,
            &image_buffers,
            &capabilities,
            &content_order,
            info.formatter.image_placeholder_token(),
            info.formatter.should_clip_image_placeholder(),
            capability_placeholder,
        )
        .map_err(|e| ClientError::Multimodal(e.to_string()))?;

        let final_prompt = info.formatter.strip_template_placeholders(&prompt_text);

        let response_format_json = request
            .text
            .as_ref()
            .map(|text| text.get("format").cloned().unwrap_or_else(|| text.clone()))
            .map(|response_format| serde_json::to_string(&response_format).unwrap_or_default())
            .unwrap_or_default();

        let rng_seed = rand::thread_rng().gen::<u64>();
        let prompt_payload = PromptPayload {
            prompt: final_prompt,
            image_buffers,
            capabilities: super::convert_capabilities(&capabilities),
            layout: super::convert_layout(&layout_segments),
            max_generated_tokens: request.max_output_tokens.unwrap_or(0),
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(-1),
            min_p: request.min_p.unwrap_or(0.0),
            rng_seed,
            stop_sequences: Vec::new(),
            num_candidates: 1,
            best_of: Some(1),
            final_candidates: Some(1),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: 1.0,
            repetition_context_size: 60,
            top_logprobs: request.top_logprobs.unwrap_or(0),
            logit_bias: HashMap::new(),
            tool_schemas_json,
            tool_calling_tokens: info.formatter.get_tool_calling_tokens().clone(),
            tool_choice: tool_choice_to_string(request.tool_choice.as_ref()),
            max_tool_calls: request.max_tool_calls.unwrap_or(0).max(0),
            response_format_json,
            task_name: None,
            reasoning_effort: request.reasoning_effort.clone(),
        };
        let (_batch_size, stream) = self.ipc.send_batch_request(
            request_id,
            model_id,
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
            ));
            Ok(ResponsesResult::Stream(event_rx))
        } else {
            let response = gather_non_streaming_response(stream, model_id, &request).await?;
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
        assert_eq!(finish_reason_to_incomplete(Some("stop")), None);
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
                    output: "{\"temperature\":65}".to_string(),
                },
            ]),
            stream: false,
            instructions: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            max_output_tokens: None,
            top_logprobs: None,
            tools: Vec::new(),
            tool_choice: None,
            max_tool_calls: None,
            text: None,
            reasoning_effort: None,
            metadata: None,
            parallel_tool_calls: false,
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
}
