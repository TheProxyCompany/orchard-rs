//! High-level client API for Orchard.
//!
//! Provides the main user-facing interface for LLM inference.

mod moondream;
mod privacy_filter;
mod response;
mod responses;

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::mpsc;

use crate::defaults;

/// Global runtime for synchronous operations.
/// Uses current_thread for efficiency - sync callers don't need multi-thread.
static SYNC_RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

fn get_sync_runtime() -> &'static tokio::runtime::Runtime {
    SYNC_RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create sync runtime")
    })
}

use crate::formatter::multimodal::{
    build_multimodal_layout, build_multimodal_messages, CapabilityInput, LayoutSegment,
};
use crate::ipc::client::{EventCallback, IPCClient, ResponseDelta};
use crate::ipc::serialization::{
    CapabilityEntry, LayoutEntry, PromptPayload, RequestType, ThinkingTokens,
};
use crate::model::registry::ModelRegistry;

pub use moondream::{
    BoundingBox, CaptionResult, DetectResult, DetectedObject, GazeResult, GroundingSpan,
    MoondreamClient, Point, PointResult, QueryResult, ReasoningOutput, SpatialRef,
    MOONDREAM_MODEL_ID,
};
pub use privacy_filter::{OpenAIPrivacyFilterClient, OPENAI_PRIVACY_FILTER_MODEL_ID};
pub use response::{BatchChatResult, ClientDelta, ClientResponse, ClientToolCall, UsageStats};
pub use responses::{
    ContentPartAddedEvent, ContentPartDoneEvent, FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent, IncompleteDetails, InputTokensDetails, OutputFunctionCall,
    OutputItemAddedEvent, OutputItemDoneEvent, OutputMessage, OutputReasoning, OutputStatus,
    OutputTextContent, OutputTextDeltaEvent, OutputTextDoneEvent, OutputTokensDetails,
    ReasoningConfig, ReasoningContent, ReasoningDeltaEvent, ReasoningDoneEvent,
    ReasoningSummaryTextContent, ReasoningSummaryTextDeltaEvent, ReasoningSummaryTextDoneEvent,
    ResponseCompletedEvent, ResponseCreatedEvent, ResponseError, ResponseEvent,
    ResponseFailedEvent, ResponseInProgressEvent, ResponseIncompleteEvent, ResponseInputItem,
    ResponseObject, ResponseOutputItem, ResponseSnapshot, ResponseUsage, ResponsesInput,
    ResponsesRequest, ResponsesResult, StreamErrorDetail, StreamErrorEvent,
};

/// Errors that can occur during client operations.
#[derive(Error, Debug)]
pub enum ClientError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("{0}")]
    ModelNotReady(String),

    #[error("{0}")]
    Ipc(String),

    #[error("{0}")]
    Formatter(String),

    #[error("{0}")]
    Multimodal(String),

    #[error("{0}")]
    RequestFailed(String),
}

impl From<crate::error::Error> for ClientError {
    fn from(err: crate::error::Error) -> Self {
        use crate::error::Error;
        match err {
            Error::ModelNotFound(s) => ClientError::ModelNotFound(s),
            Error::ModelNotReady(s) => ClientError::ModelNotReady(s),
            Error::NotConnected
            | Error::InvalidResponse
            | Error::Nng(_)
            | Error::Timeout
            | Error::ChannelClosed => ClientError::Ipc(err.to_string()),
            Error::Template(s) => ClientError::Formatter(s),
            Error::InvalidImageUrl
            | Error::InvalidBase64
            | Error::MissingContentType(_, _)
            | Error::InvalidContent
            | Error::PlaceholderMismatch(_, _)
            | Error::EmptyRequest => ClientError::Multimodal(err.to_string()),
            _ => ClientError::RequestFailed(err.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, ClientError>;

const DEFAULT_REASONING_EFFORT: &str = "medium";

fn native_reasoning_settings(
    formatter: &crate::formatter::ChatFormatter,
    requested_reasoning: bool,
    requested_reasoning_effort: &Option<String>,
) -> (bool, Option<String>, ThinkingTokens) {
    let supports_native_thinking = formatter.supports_native_thinking();
    let requested_reasoning = requested_reasoning || requested_reasoning_effort.is_some();
    let reasoning_flag = requested_reasoning && supports_native_thinking;
    let reasoning_effort = if reasoning_flag {
        requested_reasoning_effort
            .clone()
            .or_else(|| Some(DEFAULT_REASONING_EFFORT.to_string()))
    } else {
        None
    };
    let thinking_tokens = if supports_native_thinking {
        formatter.get_thinking_tokens().clone()
    } else {
        ThinkingTokens::default()
    };
    (reasoning_flag, reasoning_effort, thinking_tokens)
}

fn sampling_with_profile_defaults(
    formatter: &crate::formatter::ChatFormatter,
    params: &SamplingParams,
) -> SamplingParams {
    let mut resolved = params.clone();
    if resolved.temperature == defaults::TEMPERATURE {
        if let Some(value) = formatter.generation_default_f64("temperature") {
            resolved.temperature = value;
        }
    }
    if resolved.top_p == defaults::TOP_P {
        if let Some(value) = formatter.generation_default_f64("top_p") {
            resolved.top_p = value;
        }
    }
    if resolved.top_k == defaults::TOP_K {
        if let Some(value) = formatter.generation_default_i32("top_k") {
            resolved.top_k = value;
        }
    }
    if resolved.min_p == 0.0 {
        if let Some(value) = formatter.generation_default_f64("min_p") {
            resolved.min_p = value;
        }
    }
    if resolved.frequency_penalty == 0.0 {
        if let Some(value) = formatter.generation_default_f64("frequency_penalty") {
            resolved.frequency_penalty = value;
        }
    }
    if resolved.presence_penalty == 0.0 {
        if let Some(value) = formatter.generation_default_f64("presence_penalty") {
            resolved.presence_penalty = value;
        }
    }
    if resolved.repetition_penalty == defaults::REPETITION_PENALTY {
        if let Some(value) = formatter.generation_default_f64("repetition_penalty") {
            resolved.repetition_penalty = value;
        }
    }
    if resolved.repetition_context_size == defaults::REPETITION_CONTEXT_SIZE {
        if let Some(value) = formatter.generation_default_i32("repetition_context_size") {
            resolved.repetition_context_size = value;
        }
    }
    resolved
}

/// Sampling parameters for generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    #[serde(default = "defaults::max_tokens")]
    pub max_tokens: i32,
    #[serde(default = "defaults::temperature")]
    pub temperature: f64,
    #[serde(default = "defaults::top_p")]
    pub top_p: f64,
    #[serde(default = "defaults::top_k")]
    pub top_k: i32,
    #[serde(default)]
    pub min_p: f64,
    #[serde(default)]
    pub rng_seed: u64,
    #[serde(default)]
    pub deterministic: bool,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub frequency_penalty: f64,
    #[serde(default)]
    pub presence_penalty: f64,
    #[serde(default = "defaults::repetition_penalty")]
    pub repetition_penalty: f64,
    #[serde(default = "defaults::repetition_context_size")]
    pub repetition_context_size: i32,
    #[serde(default = "defaults::num_candidates")]
    pub n: i32,
    #[serde(default)]
    pub best_of: Option<i32>,
    #[serde(default)]
    pub final_candidates: Option<i32>,
    #[serde(default)]
    pub top_logprobs: i32,
    #[serde(default)]
    pub logit_bias: HashMap<i32, f64>,
    #[serde(default)]
    pub core_tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub active_tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub max_tool_calls: Option<i32>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub task_name: Option<String>,
    /// None => use the shared prefix cache (default). Some(false) => skip read+write.
    #[serde(default)]
    pub prefix_cache: Option<bool>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: defaults::MAX_TOKENS,
            temperature: defaults::TEMPERATURE,
            top_p: defaults::TOP_P,
            top_k: defaults::TOP_K,
            min_p: 0.0,
            rng_seed: 0,
            deterministic: false,
            stop: Vec::new(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: defaults::REPETITION_PENALTY,
            repetition_context_size: defaults::REPETITION_CONTEXT_SIZE,
            n: defaults::NUM_CANDIDATES,
            best_of: None,
            final_candidates: None,
            top_logprobs: 0,
            logit_bias: HashMap::new(),
            core_tools: Vec::new(),
            active_tools: Vec::new(),
            tool_choice: None,
            max_tool_calls: None,
            response_format: None,
            reasoning: false,
            reasoning_effort: None,
            instructions: None,
            task_name: None,
            prefix_cache: None,
        }
    }
}

fn tool_choice_to_string(tool_choice: Option<&Value>) -> String {
    match tool_choice {
        None | Some(Value::Null) => "auto".to_string(),
        Some(Value::String(value)) => value.clone(),
        Some(Value::Object(value)) => serde_json::to_string(value).unwrap_or_default(),
        Some(other) => other.to_string(),
    }
}

fn tool_schema_name(tool: &Value) -> &str {
    tool.get("name")
        .and_then(Value::as_str)
        .or_else(|| {
            tool.get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
        })
        .unwrap_or("")
}

fn sorted_tool_schemas(tools: &[Value]) -> Vec<Value> {
    let mut schemas = tools.to_vec();
    schemas.sort_by(|a, b| tool_schema_name(a).cmp(tool_schema_name(b)));
    schemas
}

fn serialize_tool_schemas(tools: &[Value]) -> String {
    if tools.is_empty() {
        String::new()
    } else {
        serde_json::to_string(tools).unwrap_or_default()
    }
}

fn core_and_active_tool_schemas(params: &SamplingParams) -> (Vec<Value>, Vec<Value>) {
    let core_source = &params.core_tools;
    let active_source = if params.active_tools.is_empty() {
        core_source
    } else {
        &params.active_tools
    };
    (
        sorted_tool_schemas(core_source),
        sorted_tool_schemas(active_source),
    )
}

/// A high-level client for the Proxy Inference Engine.
///
/// Provides both synchronous and asynchronous interfaces for LLM inference.
#[derive(Clone)]
pub struct Client {
    ipc: Arc<IPCClient>,
    registry: Arc<ModelRegistry>,
}

impl Client {
    /// Create a new client with the given IPC client and model registry.
    pub fn new(ipc: Arc<IPCClient>, registry: Arc<ModelRegistry>) -> Self {
        Self { ipc, registry }
    }

    /// Create a client and connect to the engine (async).
    ///
    /// This sets up:
    /// - Event callback for handling model lifecycle events
    /// - IPC client shared with registry for management commands
    pub async fn connect(registry: Arc<ModelRegistry>) -> Result<Self> {
        // Create event callback that routes model lifecycle events to registry
        let registry_for_events = Arc::clone(&registry);
        let runtime_handle = tokio::runtime::Handle::current();
        let event_callback: EventCallback =
            Arc::new(move |event_name: &str, payload: &Value| match event_name {
                "model_loaded" => {
                    let registry = Arc::clone(&registry_for_events);
                    let payload = payload.clone();
                    let handle = runtime_handle.clone();
                    handle.spawn(async move {
                        registry.handle_model_loaded(&payload).await;
                    });
                }
                "model_load_failed" => {
                    let registry = Arc::clone(&registry_for_events);
                    let payload = payload.clone();
                    let handle = runtime_handle.clone();
                    handle.spawn(async move {
                        registry.handle_model_load_failed(&payload).await;
                    });
                }
                _ => {}
            });

        let mut ipc = IPCClient::with_event_callback(event_callback);
        ipc.connect()?;
        let ipc = Arc::new(ipc);

        // Share the IPC client with the registry for management commands
        registry.set_ipc_client(Arc::clone(&ipc)).await;

        Ok(Self { ipc, registry })
    }

    /// Resolve control token capabilities for a model.
    pub async fn resolve_capabilities(&self, model_id: &str) -> Result<HashMap<String, i32>> {
        let info = self.registry.ensure_loaded(model_id).await?;

        let capabilities = info.capabilities.as_ref().cloned().unwrap_or_default();
        let mut resolved = HashMap::new();

        for (name, token_ids) in capabilities {
            if let Some(&first) = token_ids.first() {
                resolved.insert(name, first);
            }
        }

        Ok(resolved)
    }

    /// Perform asynchronous chat completion.
    ///
    /// # Arguments
    /// * `model_id` - Model to use for generation
    /// * `messages` - Conversation messages
    /// * `params` - Sampling parameters
    /// * `stream` - Whether to stream responses
    pub async fn achat(
        &self,
        model_id: &str,
        messages: Vec<HashMap<String, serde_json::Value>>,
        params: SamplingParams,
        stream: bool,
    ) -> Result<ChatResult> {
        let info = self.registry.ensure_loaded(model_id).await?;
        let formatter = info.require_formatter()?;
        let request_model_id = info.model_id.as_str();
        let params = sampling_with_profile_defaults(formatter, &params);

        let request_id = self.ipc.next_request_id();
        tracing::debug!(
            request_id,
            model_id = %model_id,
            stream,
            message_count = messages.len(),
            "Building chat request"
        );
        tracing::trace!(
            request_id,
            model_id = %model_id,
            messages = ?messages,
            "Chat messages before template application"
        );

        let (reasoning_flag, reasoning_effort, thinking_tokens) = native_reasoning_settings(
            formatter,
            params.reasoning || params.reasoning_effort.is_some(),
            &params.reasoning_effort,
        );

        // Build multimodal content (pass instructions if provided)
        let (messages_for_template, image_buffers, capabilities, content_order) =
            build_multimodal_messages(formatter, &messages, params.instructions.as_deref())
                .map_err(|e| ClientError::Multimodal(e.to_string()))?;

        if messages_for_template.is_empty() {
            return Err(ClientError::RequestFailed(
                "Chat request must include at least one message".into(),
            ));
        }
        tracing::trace!(
            request_id,
            model_id = %model_id,
            messages_for_template = ?messages_for_template,
            "Chat messages after multimodal expansion"
        );
        let (core_tool_schemas, active_tool_schemas) = core_and_active_tool_schemas(&params);
        let template_tools =
            (!core_tool_schemas.is_empty()).then_some(core_tool_schemas.as_slice());

        // Apply template with reasoning flag
        let prompt_text = formatter
            .apply_template_with_tools(
                &messages_for_template,
                true,
                reasoning_flag,
                params.task_name.as_deref(),
                reasoning_effort.as_deref(),
                template_tools,
            )
            .map_err(|e| ClientError::Formatter(e.to_string()))?;

        let capability_placeholder = formatter.capability_placeholder_token();

        // Build layout for multimodal content
        let layout_segments = build_multimodal_layout(
            &prompt_text,
            &image_buffers,
            &capabilities,
            &content_order,
            formatter.image_placeholder_token(),
            formatter.should_clip_image_placeholder(),
            capability_placeholder,
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
            "Prepared chat prompt payload"
        );
        tracing::trace!(
            request_id,
            model_id = %model_id,
            prompt = %final_prompt,
            "Chat prompt sent to PIE"
        );

        // Core tools are rendered in the prompt; active tools drive PSE grammar.
        let tool_schemas_json = serialize_tool_schemas(&core_tool_schemas);
        let active_tool_schemas_json = serialize_tool_schemas(&active_tool_schemas);
        let response_format_json = params
            .response_format
            .as_ref()
            .map(|rf| serde_json::to_string(rf).unwrap_or_default())
            .unwrap_or_default();
        let tool_calling_tokens = formatter.get_tool_calling_tokens().clone();
        let output_frame_tokens = formatter.get_output_frame_tokens().clone();
        let tool_choice = tool_choice_to_string(params.tool_choice.as_ref());
        let max_tool_calls = params.max_tool_calls.unwrap_or(0).max(0);
        // Build PromptPayload with full multimodal data
        // Generate unique RNG seed if not explicitly provided
        let rng_seed = if params.rng_seed == 0 {
            rand::thread_rng().gen::<u64>()
        } else {
            params.rng_seed
        };

        let prompt_payload = PromptPayload {
            prompt: final_prompt,
            image_buffers,
            capabilities: convert_capabilities(&capabilities),
            layout: convert_layout(&layout_segments),
            max_generated_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            min_p: params.min_p,
            rng_seed,
            deterministic: params.deterministic,
            stop_sequences: params.stop.clone(),
            num_candidates: params.n,
            best_of: params.best_of,
            final_candidates: params.final_candidates,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            repetition_penalty: params.repetition_penalty,
            repetition_context_size: params.repetition_context_size,
            top_logprobs: params.top_logprobs,
            logit_bias: params.logit_bias.clone(),
            tool_schemas_json,
            active_tool_schemas_json,
            tool_calling_tokens,
            output_frame_tokens,
            thinking_tokens,
            tool_choice,
            max_tool_calls,
            response_format_json,
            task_name: params.task_name.clone(),
            reasoning_effort,
            prefix_cache: params.prefix_cache,
        };

        // Use unified batch request path (even for single prompts)
        tracing::debug!(
            request_id,
            model_id = %model_id,
            stream,
            "Dispatching chat request to PIE"
        );
        let (_batch_size, rx) = self.ipc.send_batch_request(
            request_id,
            request_model_id,
            &info.model_path,
            &[prompt_payload],
        )?;

        if stream {
            Ok(ChatResult::Stream(rx))
        } else {
            // Determine how many candidates to expect
            let best_of = params.best_of.unwrap_or(params.n).max(1) as usize;
            let final_candidates = params.final_candidates.unwrap_or(params.n).max(1) as usize;

            // Collect deltas grouped by candidate_index (matching Python's gather_non_streaming_batch_response)
            let mut candidate_states: Vec<CandidateState> =
                (0..best_of).map(|_| CandidateState::default()).collect();
            let mut remaining_sequences = best_of;
            let mut rx = rx;

            while remaining_sequences > 0 {
                match rx.recv().await {
                    Some(delta) => {
                        let candidate_index = delta.candidate_index.unwrap_or(0) as usize;
                        if candidate_index >= candidate_states.len() {
                            continue;
                        }

                        let state = &mut candidate_states[candidate_index];

                        if let Some(content) = &delta.content {
                            state.content.push_str(content);
                        }
                        state.completion_tokens += delta.tokens.len() as u32;
                        if let Some(count) = delta.prompt_token_count {
                            state.prompt_tokens = state.prompt_tokens.max(count);
                        }
                        if let Some(reasoning_tokens) = delta.reasoning_tokens {
                            state.reasoning_tokens = state.reasoning_tokens.max(reasoning_tokens);
                        }

                        let client_delta = ClientDelta::from(delta.clone());
                        state.deltas.push(client_delta);

                        if delta.is_final_delta && !state.completed {
                            state.completed = true;
                            state.finish_reason = delta.finish_reason.clone();
                            state.cumulative_logprob = delta.cumulative_logprob;
                            state.generation_len = delta.generation_len;
                            if let Some(generation_len) = delta.generation_len {
                                state.completion_tokens =
                                    generation_len.saturating_sub(state.reasoning_tokens);
                            }
                            remaining_sequences -= 1;
                        }
                    }
                    None => break,
                }
            }

            let total_completion_tokens: u32 =
                candidate_states.iter().map(|c| c.completion_tokens).sum();

            // Score and select best candidates (matching Python logic)
            let selected = select_best_candidates(candidate_states, best_of, final_candidates);

            Ok(ChatResult::Complete(build_response_from_candidates(
                selected,
                total_completion_tokens,
            )))
        }
    }

    /// Perform synchronous chat completion (blocking).
    ///
    /// Handles nested async contexts properly - safe to call from any context.
    pub fn chat(
        &self,
        model_id: &str,
        messages: Vec<HashMap<String, serde_json::Value>>,
        params: SamplingParams,
    ) -> Result<ClientResponse> {
        let future = async {
            match self.achat(model_id, messages, params, false).await? {
                ChatResult::Complete(response) => Ok(response),
                ChatResult::Stream(_) => Err(ClientError::RequestFailed(
                    "Unexpected stream result".into(),
                )),
            }
        };

        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                // Already in async context - use block_in_place to avoid panic
                tokio::task::block_in_place(|| handle.block_on(future))
            }
            Err(_) => {
                // Not in async context - use the global sync runtime
                get_sync_runtime().block_on(future)
            }
        }
    }

    /// Perform batched chat completion.
    ///
    /// This sends ALL conversations in ONE IPC message, allowing the engine
    /// to schedule them together efficiently. Responses are demultiplexed
    /// by prompt_index and returned in order.
    ///
    /// # Arguments
    /// * `model_id` - Model to use for generation
    /// * `conversations` - List of conversation message lists
    /// * `params` - Sampling parameters
    /// * `stream` - Whether to stream responses (deltas contain prompt_index)
    pub async fn achat_batch(
        &self,
        model_id: &str,
        conversations: Vec<Vec<HashMap<String, serde_json::Value>>>,
        params: SamplingParams,
        stream: bool,
    ) -> Result<BatchChatResult> {
        let batch_size = conversations.len();
        self.achat_batch_with_params(model_id, conversations, vec![params; batch_size], stream)
            .await
    }

    /// Perform batched chat completion with per-prompt sampling parameters.
    pub async fn achat_batch_with_params(
        &self,
        model_id: &str,
        conversations: Vec<Vec<HashMap<String, serde_json::Value>>>,
        params_by_prompt: Vec<SamplingParams>,
        stream: bool,
    ) -> Result<BatchChatResult> {
        if conversations.is_empty() {
            return Ok(BatchChatResult::Complete(Vec::new()));
        }
        if params_by_prompt.len() != conversations.len() {
            return Err(ClientError::RequestFailed(format!(
                "params_by_prompt length ({}) does not match batch size ({})",
                params_by_prompt.len(),
                conversations.len()
            )));
        }

        let info = self.registry.ensure_loaded(model_id).await?;
        let formatter = info.require_formatter()?;
        let request_model_id = info.model_id.as_str();

        let request_id = self.ipc.next_request_id();
        let num_prompts = conversations.len();
        tracing::debug!(
            request_id,
            model_id = %model_id,
            stream,
            prompt_count = num_prompts,
            "Building batched chat request"
        );

        let tool_calling_tokens = formatter.get_tool_calling_tokens().clone();
        let output_frame_tokens = formatter.get_output_frame_tokens().clone();

        // Build all prompt payloads
        let mut prompt_payloads = Vec::with_capacity(num_prompts);

        for (prompt_index, messages) in conversations.iter().enumerate() {
            let params = sampling_with_profile_defaults(formatter, &params_by_prompt[prompt_index]);
            let (reasoning_flag, reasoning_effort, thinking_tokens) = native_reasoning_settings(
                formatter,
                params.reasoning || params.reasoning_effort.is_some(),
                &params.reasoning_effort,
            );
            let (core_tool_schemas, active_tool_schemas) = core_and_active_tool_schemas(&params);
            let tool_schemas_json = serialize_tool_schemas(&core_tool_schemas);
            let active_tool_schemas_json = serialize_tool_schemas(&active_tool_schemas);
            let response_format_json = params
                .response_format
                .as_ref()
                .map(|rf| serde_json::to_string(rf).unwrap_or_default())
                .unwrap_or_default();
            let tool_choice = tool_choice_to_string(params.tool_choice.as_ref());
            let max_tool_calls = params.max_tool_calls.unwrap_or(0).max(0);

            // Build multimodal content (pass instructions if provided)
            let (messages_for_template, image_buffers, capabilities, content_order) =
                build_multimodal_messages(formatter, messages, params.instructions.as_deref())
                    .map_err(|e| ClientError::Multimodal(e.to_string()))?;

            if messages_for_template.is_empty() {
                return Err(ClientError::RequestFailed(
                    "Chat request must include at least one message".into(),
                ));
            }
            tracing::trace!(
                request_id,
                model_id = %model_id,
                prompt_index,
                messages = ?messages,
                messages_for_template = ?messages_for_template,
                "Prepared batch messages for prompt"
            );
            let template_tools =
                (!core_tool_schemas.is_empty()).then_some(core_tool_schemas.as_slice());

            // Apply template with reasoning flag
            let prompt_text = formatter
                .apply_template_with_tools(
                    &messages_for_template,
                    true,
                    reasoning_flag,
                    params.task_name.as_deref(),
                    reasoning_effort.as_deref(),
                    template_tools,
                )
                .map_err(|e| ClientError::Formatter(e.to_string()))?;

            let capability_placeholder = formatter.capability_placeholder_token();

            // Build layout for multimodal content
            let layout_segments = build_multimodal_layout(
                &prompt_text,
                &image_buffers,
                &capabilities,
                &content_order,
                formatter.image_placeholder_token(),
                formatter.should_clip_image_placeholder(),
                capability_placeholder,
            )
            .map_err(|e| ClientError::Multimodal(e.to_string()))?;

            let final_prompt = formatter.strip_template_placeholders(&prompt_text);
            tracing::debug!(
                request_id,
                model_id = %model_id,
                prompt_index,
                prompt_chars = final_prompt.chars().count(),
                image_count = image_buffers.len(),
                capability_count = capabilities.len(),
                layout_segment_count = layout_segments.len(),
                "Prepared batched prompt payload"
            );
            tracing::trace!(
                request_id,
                model_id = %model_id,
                prompt_index,
                prompt = %final_prompt,
                "Batch prompt sent to PIE"
            );

            // Generate unique RNG seed for EACH prompt in batch
            let rng_seed = if params.rng_seed == 0 {
                rand::thread_rng().gen::<u64>()
            } else {
                params.rng_seed
            };
            prompt_payloads.push(PromptPayload {
                prompt: final_prompt,
                image_buffers,
                capabilities: convert_capabilities(&capabilities),
                layout: convert_layout(&layout_segments),
                max_generated_tokens: params.max_tokens,
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                min_p: params.min_p,
                rng_seed,
                deterministic: params.deterministic,
                stop_sequences: params.stop.clone(),
                num_candidates: params.n,
                best_of: params.best_of,
                final_candidates: params.final_candidates,
                frequency_penalty: params.frequency_penalty,
                presence_penalty: params.presence_penalty,
                repetition_penalty: params.repetition_penalty,
                repetition_context_size: params.repetition_context_size,
                top_logprobs: params.top_logprobs,
                logit_bias: params.logit_bias.clone(),
                tool_schemas_json: tool_schemas_json.clone(),
                active_tool_schemas_json: active_tool_schemas_json.clone(),
                tool_calling_tokens: tool_calling_tokens.clone(),
                output_frame_tokens: output_frame_tokens.clone(),
                thinking_tokens,
                tool_choice: tool_choice.clone(),
                max_tool_calls,
                response_format_json: response_format_json.clone(),
                task_name: params.task_name.clone(),
                reasoning_effort,
                prefix_cache: params.prefix_cache,
            });
        }

        // Send ONE batch request with all prompts
        tracing::debug!(
            request_id,
            model_id = %model_id,
            stream,
            prompt_count = prompt_payloads.len(),
            "Dispatching batched chat request to PIE"
        );
        let (_batch_size, rx) = self.ipc.send_batch_request(
            request_id,
            request_model_id,
            &info.model_path,
            &prompt_payloads,
        )?;

        if stream {
            // Convert ResponseDelta receiver to ClientDelta receiver
            let (tx, client_rx) = mpsc::channel(256);
            tokio::spawn(async move {
                let mut rx = rx;
                while let Some(delta) = rx.recv().await {
                    if tx.send(ClientDelta::from(delta)).await.is_err() {
                        break;
                    }
                }
            });
            return Ok(BatchChatResult::Stream(client_rx));
        }

        // Collect responses grouped by prompt_index
        let mut deltas_by_prompt: HashMap<u32, Vec<ClientDelta>> = HashMap::new();
        let mut finals_received = 0usize;
        let mut rx = rx;

        while finals_received < num_prompts {
            match rx.recv().await {
                Some(delta) => {
                    let prompt_index = delta.prompt_index.unwrap_or(0);
                    let is_final = delta.is_final_delta;

                    deltas_by_prompt
                        .entry(prompt_index)
                        .or_default()
                        .push(ClientDelta::from(delta));

                    if is_final {
                        finals_received += 1;
                    }
                }
                None => break, // Channel closed
            }
        }

        // Build responses in order
        let mut responses = Vec::with_capacity(num_prompts);
        for idx in 0..num_prompts {
            let deltas = deltas_by_prompt.remove(&(idx as u32)).unwrap_or_default();
            responses.push(aggregate_response(deltas));
        }

        Ok(BatchChatResult::Complete(responses))
    }

    /// Generate an embedding for a single text input.
    pub async fn aembed(&self, model_id: &str, text: &str) -> Result<Vec<f32>> {
        let mut embeddings = self.aembed_batch(model_id, vec![text.to_string()]).await?;
        embeddings.pop().ok_or_else(|| {
            ClientError::RequestFailed("Embedding response missing result".to_string())
        })
    }

    /// Generate embeddings for multiple text inputs in a single IPC request.
    pub async fn aembed_batch(&self, model_id: &str, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let info = self.registry.ensure_loaded(model_id).await?;
        let request_model_id = info.model_id.as_str();

        let request_id = self.ipc.next_request_id();
        tracing::debug!(
            request_id,
            model_id = %model_id,
            prompt_count = texts.len(),
            "Building batched embedding request"
        );

        let mut prompt_payloads = Vec::with_capacity(texts.len());
        for (prompt_index, text) in texts.into_iter().enumerate() {
            let prompt_chars = text.chars().count();
            tracing::debug!(
                request_id,
                model_id = %model_id,
                prompt_index,
                prompt_chars,
                "Prepared embedding prompt payload"
            );
            tracing::trace!(
                request_id,
                model_id = %model_id,
                prompt_index,
                prompt = %text,
                "Embedding prompt sent to PIE"
            );

            prompt_payloads.push(build_embedding_prompt_payload(text));
        }

        tracing::debug!(
            request_id,
            model_id = %model_id,
            prompt_count = prompt_payloads.len(),
            "Dispatching batched embedding request to PIE"
        );
        let (_batch_size, rx) = self.ipc.send_batch_request_with_type(
            request_id,
            request_model_id,
            &info.model_path,
            RequestType::Embedding,
            &prompt_payloads,
        )?;

        collect_embeddings(rx, prompt_payloads.len()).await
    }

    /// Transcribe float32 PCM audio with a local speech-to-text model.
    pub async fn atranscribe_audio(&self, model_id: &str, pcm: &[f32]) -> Result<String> {
        if pcm.is_empty() {
            return Ok(String::new());
        }

        let info = self.registry.ensure_loaded(model_id).await?;
        let request_model_id = info.model_id.as_str();

        let request_id = self.ipc.next_request_id();
        tracing::debug!(
            request_id,
            model_id = %model_id,
            sample_count = pcm.len(),
            "Building speech-to-text request"
        );

        let prompt_payload = build_stt_prompt_payload(pcm);

        tracing::debug!(
            request_id,
            model_id = %model_id,
            payload_bytes = prompt_payload.capabilities[0].payload.len(),
            "Dispatching speech-to-text request to PIE"
        );
        let (_batch_size, rx) = self.ipc.send_batch_request_with_type(
            request_id,
            request_model_id,
            &info.model_path,
            RequestType::Omni,
            &[prompt_payload],
        )?;

        collect_transcription(rx).await
    }

    /// Synchronous speech-to-text wrapper.
    pub fn transcribe_audio(&self, model_id: &str, pcm: &[f32]) -> Result<String> {
        let model_id = model_id.to_string();
        let pcm = pcm.to_vec();
        let future = async move { self.atranscribe_audio(&model_id, &pcm).await };

        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(future)),
            Err(_) => get_sync_runtime().block_on(future),
        }
    }

    /// Run a prefill-only task and return the raw response deltas.
    pub async fn aprefill_task(
        &self,
        model_id: &str,
        text: &str,
        task_name: &str,
    ) -> Result<Vec<ClientDelta>> {
        let mut results = self
            .aprefill_task_batch(model_id, vec![text.to_string()], task_name)
            .await?;
        results.pop().ok_or_else(|| {
            ClientError::RequestFailed("Prefill task response missing result".to_string())
        })
    }

    /// Run a prefill-only task for multiple texts in one IPC request.
    pub async fn aprefill_task_batch(
        &self,
        model_id: &str,
        texts: Vec<String>,
        task_name: &str,
    ) -> Result<Vec<Vec<ClientDelta>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let info = self.registry.ensure_loaded(model_id).await?;
        let request_id = self.ipc.next_request_id();
        let prompt_payloads: Vec<_> = texts
            .iter()
            .map(|text| build_prefill_task_prompt_payload(text, task_name))
            .collect();
        let (_batch_size, mut rx) = self.ipc.send_batch_request_with_type(
            request_id,
            info.model_id.as_str(),
            &info.model_path,
            RequestType::PrefillTask,
            &prompt_payloads,
        )?;

        let mut deltas_by_prompt = vec![Vec::new(); prompt_payloads.len()];
        let mut completed_prompts = vec![false; prompt_payloads.len()];
        let mut finals_received = 0usize;
        while finals_received < prompt_payloads.len() {
            match rx.recv().await {
                Some(delta) => {
                    if let Some(error) = delta.error.clone() {
                        return Err(ClientError::RequestFailed(error));
                    }
                    let prompt_index = delta.prompt_index.unwrap_or(0) as usize;
                    let is_final = delta.is_final_delta;
                    if prompt_index < deltas_by_prompt.len() {
                        deltas_by_prompt[prompt_index].push(ClientDelta::from(delta));
                        if is_final && !completed_prompts[prompt_index] {
                            completed_prompts[prompt_index] = true;
                            finals_received += 1;
                        }
                    }
                }
                None => {
                    return Err(ClientError::RequestFailed(
                        "Prefill task response channel closed before completion".to_string(),
                    ));
                }
            }
        }
        Ok(deltas_by_prompt)
    }
}

/// Convert CapabilityInput from multimodal to CapabilityEntry for serialization.
/// Position is always 0 (matching Python behavior).
fn convert_capabilities(capabilities: &[CapabilityInput]) -> Vec<CapabilityEntry> {
    capabilities
        .iter()
        .map(|cap| CapabilityEntry {
            name: cap.name.clone(),
            position: 0, // Always 0, matching Python
            payload: cap.payload.clone(),
        })
        .collect()
}

/// Convert LayoutSegment from multimodal to LayoutEntry for serialization.
fn convert_layout(segments: &[LayoutSegment]) -> Vec<LayoutEntry> {
    segments
        .iter()
        .map(|seg| LayoutEntry {
            segment_type: seg.segment_type.clone(),
            length: seg.length,
        })
        .collect()
}

fn build_embedding_prompt_payload(prompt: String) -> PromptPayload {
    let prompt_len = prompt.len();

    PromptPayload {
        prompt,
        image_buffers: Vec::new(),
        capabilities: Vec::new(),
        layout: vec![LayoutEntry {
            segment_type: "text".to_string(),
            length: prompt_len,
        }],
        max_generated_tokens: 0,
        temperature: defaults::TEMPERATURE,
        top_p: defaults::TOP_P,
        top_k: defaults::TOP_K,
        min_p: 0.0,
        rng_seed: rand::thread_rng().gen::<u64>(),
        deterministic: false,
        stop_sequences: Vec::new(),
        num_candidates: 1,
        best_of: Some(1),
        final_candidates: Some(1),
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: defaults::REPETITION_PENALTY,
        repetition_context_size: 0,
        top_logprobs: 0,
        logit_bias: HashMap::new(),
        tool_schemas_json: String::new(),
        active_tool_schemas_json: String::new(),
        tool_calling_tokens: Default::default(),
        output_frame_tokens: Default::default(),
        thinking_tokens: Default::default(),
        tool_choice: "auto".to_string(),
        max_tool_calls: 0,
        response_format_json: String::new(),
        task_name: None,
        reasoning_effort: None,
        prefix_cache: None,
    }
}

fn build_prefill_task_prompt_payload(prompt: &str, task_name: &str) -> PromptPayload {
    PromptPayload {
        task_name: Some(task_name.to_string()),
        ..build_embedding_prompt_payload(prompt.to_string())
    }
}

fn build_stt_prompt_payload(pcm: &[f32]) -> PromptPayload {
    let audio_payload = encode_float32_pcm_bytes(pcm);
    let audio_payload_size = audio_payload.len();

    PromptPayload {
        prompt: String::new(),
        image_buffers: Vec::new(),
        capabilities: vec![CapabilityEntry {
            name: "audio".to_string(),
            position: 0,
            payload: audio_payload,
        }],
        layout: vec![
            LayoutEntry {
                segment_type: "text".to_string(),
                length: 0,
            },
            LayoutEntry {
                segment_type: "capability".to_string(),
                length: audio_payload_size,
            },
        ],
        max_generated_tokens: 0,
        temperature: defaults::TEMPERATURE,
        top_p: defaults::TOP_P,
        top_k: defaults::TOP_K,
        min_p: 0.0,
        rng_seed: rand::thread_rng().gen::<u64>(),
        deterministic: false,
        stop_sequences: Vec::new(),
        num_candidates: 1,
        best_of: Some(1),
        final_candidates: Some(1),
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: defaults::REPETITION_PENALTY,
        repetition_context_size: 0,
        top_logprobs: 0,
        logit_bias: HashMap::new(),
        tool_schemas_json: String::new(),
        active_tool_schemas_json: String::new(),
        tool_calling_tokens: Default::default(),
        output_frame_tokens: Default::default(),
        thinking_tokens: Default::default(),
        tool_choice: "auto".to_string(),
        max_tool_calls: 0,
        response_format_json: String::new(),
        task_name: None,
        reasoning_effort: None,
        prefix_cache: None,
    }
}

fn encode_float32_pcm_bytes(pcm: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(pcm));
    for sample in pcm {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

async fn collect_embeddings(
    mut rx: mpsc::UnboundedReceiver<ResponseDelta>,
    prompt_count: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut embeddings_by_prompt: Vec<Option<Vec<f32>>> = vec![None; prompt_count];
    let mut completed_prompts = vec![false; prompt_count];
    let mut finals_received = 0usize;

    while finals_received < prompt_count {
        match rx.recv().await {
            Some(delta) => {
                if let Some(error) = delta.error {
                    return Err(ClientError::RequestFailed(error));
                }

                let prompt_index = delta.prompt_index.unwrap_or(0) as usize;
                if prompt_index >= prompt_count {
                    continue;
                }

                if let Some(bytes) = delta.embedding_bytes.as_deref() {
                    embeddings_by_prompt[prompt_index] = Some(decode_embedding_bytes(bytes)?);
                }

                if delta.is_final_delta && !completed_prompts[prompt_index] {
                    completed_prompts[prompt_index] = true;
                    finals_received += 1;
                }
            }
            None => {
                return Err(ClientError::RequestFailed(
                    "Embedding response channel closed before completion".to_string(),
                ));
            }
        }
    }

    embeddings_by_prompt
        .into_iter()
        .enumerate()
        .map(|(prompt_index, embedding)| {
            embedding.ok_or_else(|| {
                ClientError::RequestFailed(format!(
                    "Embedding response missing bytes for prompt_index={}",
                    prompt_index
                ))
            })
        })
        .collect()
}

fn decode_embedding_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    let mut chunks = bytes.chunks_exact(std::mem::size_of::<f32>());
    if !chunks.remainder().is_empty() {
        return Err(ClientError::RequestFailed(format!(
            "Embedding payload length {} is not divisible by {}",
            bytes.len(),
            std::mem::size_of::<f32>()
        )));
    }

    Ok(chunks
        .by_ref()
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk size")))
        .collect())
}

async fn collect_transcription(mut rx: mpsc::UnboundedReceiver<ResponseDelta>) -> Result<String> {
    let mut transcription = String::new();

    loop {
        match rx.recv().await {
            Some(delta) => {
                if let Some(error) = delta.error {
                    return Err(ClientError::RequestFailed(error));
                }

                if let Some(content) = delta.content {
                    transcription.push_str(&content);
                }

                if delta.is_final_delta {
                    return Ok(transcription);
                }
            }
            None => {
                return Err(ClientError::RequestFailed(
                    "Speech-to-text response channel closed before completion".to_string(),
                ));
            }
        }
    }
}

/// Result of a chat operation.
pub enum ChatResult {
    /// Complete response (non-streaming)
    Complete(ClientResponse),
    /// Streaming response receiver
    Stream(mpsc::UnboundedReceiver<ResponseDelta>),
}

/// State for a single candidate during best_of collection.
/// Mirrors Python's candidate state dict in gather_non_streaming_batch_response.
#[derive(Default)]
struct CandidateState {
    content: String,
    finish_reason: Option<String>,
    completion_tokens: u32,
    reasoning_tokens: u32,
    prompt_tokens: u32,
    cumulative_logprob: Option<f64>,
    generation_len: Option<u32>,
    completed: bool,
    deltas: Vec<ClientDelta>,
}

impl CandidateState {
    /// Score for best_of selection: cumulative_logprob / generation_len
    #[inline]
    fn score(&self) -> f64 {
        match (self.cumulative_logprob, self.generation_len) {
            (Some(cumulative), Some(gen_len)) if gen_len > 0 => cumulative / gen_len as f64,
            _ => f64::NEG_INFINITY,
        }
    }
}

/// Select the best candidates based on cumulative_logprob / generation_len.
/// Sorts in-place and truncates to avoid allocations.
fn select_best_candidates(
    mut candidates: Vec<CandidateState>,
    fanout: usize,
    final_target: usize,
) -> Vec<CandidateState> {
    let final_target = final_target.min(candidates.len()).max(1);

    if final_target >= fanout {
        return candidates;
    }

    // Sort in-place by score descending
    candidates.sort_by(|a, b| {
        b.score()
            .partial_cmp(&a.score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates.truncate(final_target);
    candidates
}

/// Build a ClientResponse from selected candidates, using completion tokens from the full best_of fan-out.
fn build_response_from_candidates(
    candidates: Vec<CandidateState>,
    total_completion_tokens: u32,
) -> ClientResponse {
    let prompt_tokens = candidates
        .iter()
        .map(|c| c.prompt_tokens)
        .max()
        .unwrap_or(0);

    let capacity: usize = candidates.iter().map(|c| c.deltas.len()).sum();
    let mut all_deltas = Vec::with_capacity(capacity);
    let mut text = String::new();
    let mut finish_reason = None;

    for candidate in candidates {
        text.push_str(&aggregate_message_text(&candidate.deltas));
        if candidate.finish_reason.is_some() {
            finish_reason = candidate.finish_reason;
        }
        all_deltas.extend(candidate.deltas);
    }
    let (reasoning, tool_calls) = aggregate_structured_items(&all_deltas);

    ClientResponse {
        text,
        finish_reason,
        usage: UsageStats {
            prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: prompt_tokens + total_completion_tokens,
        },
        reasoning,
        tool_calls,
        deltas: all_deltas,
    }
}

fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn aggregate_message_text(deltas: &[ClientDelta]) -> String {
    let has_state_events = deltas.iter().any(|delta| !delta.state_events.is_empty());
    if !has_state_events {
        return deltas
            .iter()
            .filter_map(|delta| delta.content.as_ref())
            .cloned()
            .collect();
    }

    let has_structured_items = deltas
        .iter()
        .flat_map(|delta| &delta.state_events)
        .any(|event| event.item_type != "message");
    if has_structured_items {
        return aggregate_message_state_text_with_raw_suffix(deltas);
    }

    let mut text = String::new();
    let mut completed_value = None;

    for delta in deltas {
        if delta.state_events.is_empty() {
            if let Some(content) = &delta.content {
                text.push_str(content);
            }
            continue;
        }

        let message_delta = delta
            .state_events
            .iter()
            .find(|event| event.item_type == "message" && event.event_type == "content_delta")
            .map(|event| event.delta.as_str());
        let delta_content = delta
            .content
            .as_deref()
            .filter(|content| !content.is_empty())
            .or(message_delta)
            .unwrap_or_default();

        for event in &delta.state_events {
            if event.item_type == "message"
                && event.event_type == "item_completed"
                && event.value.is_some()
                && text.is_empty()
                && delta_content.is_empty()
            {
                completed_value = event.value.as_ref().map(value_to_text);
            }
        }

        if !delta_content.is_empty() {
            text.push_str(delta_content);
        }
    }

    if !text.is_empty() {
        return text;
    }
    completed_value.unwrap_or_default()
}

fn aggregate_message_state_text(deltas: &[ClientDelta]) -> String {
    let mut text = String::new();
    let mut completed_value = None;
    for event in deltas.iter().flat_map(|delta| &delta.state_events) {
        if event.item_type != "message" {
            continue;
        }
        if event.event_type == "content_delta" {
            text.push_str(&event.delta);
        } else if event.event_type == "item_completed" {
            if let Some(value) = &event.value {
                completed_value = Some(value_to_text(value));
            }
        }
    }

    if !text.is_empty() {
        return text;
    }
    completed_value.unwrap_or_default()
}

fn aggregate_message_state_text_with_raw_suffix(deltas: &[ClientDelta]) -> String {
    let state_text = aggregate_message_state_text(deltas);
    if state_text.is_empty() {
        return state_text;
    }

    let raw_text: String = deltas
        .iter()
        .filter_map(|delta| delta.content.as_ref())
        .cloned()
        .collect();
    if raw_text.len() > state_text.len() {
        if let Some(start) = raw_text.rfind(&state_text) {
            let suffix_start = start + state_text.len();
            if suffix_start < raw_text.len() {
                return format!("{}{}", state_text, &raw_text[suffix_start..]);
            }
        }
    }

    state_text
}

fn aggregate_structured_items(deltas: &[ClientDelta]) -> (Vec<String>, Vec<ClientToolCall>) {
    let mut reasoning_by_identifier: HashMap<String, String> = HashMap::new();
    let mut tool_calls_by_index: HashMap<u32, ClientToolCall> = HashMap::new();

    for event in deltas.iter().flat_map(|delta| &delta.state_events) {
        match event.item_type.as_str() {
            "reasoning" => {
                let entry = reasoning_by_identifier
                    .entry(event.identifier.clone())
                    .or_default();
                if event.event_type == "content_delta" {
                    entry.push_str(&event.delta);
                } else if event.event_type == "item_completed" {
                    if let Some(value) = &event.value {
                        *entry = value_to_text(value);
                    }
                }
            }
            "tool_call" => {
                let tool_call = tool_calls_by_index
                    .entry(event.output_index)
                    .or_insert_with(|| ClientToolCall {
                        name: event
                            .identifier
                            .trim_start_matches("tool_call:")
                            .to_string(),
                        arguments: Value::String(String::new()),
                    });
                if event.identifier.starts_with("tool_call:") {
                    tool_call.name = event
                        .identifier
                        .trim_start_matches("tool_call:")
                        .to_string();
                }
                if event.identifier == "arguments" && event.event_type == "content_delta" {
                    match &mut tool_call.arguments {
                        Value::String(arguments) => arguments.push_str(&event.delta),
                        _ => tool_call.arguments = Value::String(event.delta.clone()),
                    }
                } else if event.event_type == "item_completed" {
                    let parsed_value = match &event.value {
                        Some(Value::String(value)) => serde_json::from_str::<Value>(value).ok(),
                        _ => None,
                    };
                    let value = parsed_value.as_ref().or(event.value.as_ref());
                    if let Some(Value::Object(object)) = value {
                        if let Some(Value::String(name)) = object.get("name") {
                            tool_call.name = name.clone();
                        }
                        if let Some(arguments) = object.get("arguments") {
                            tool_call.arguments = arguments.clone();
                        }
                    } else if event.identifier == "arguments" {
                        if let Some(value) = &event.value {
                            tool_call.arguments = value.clone();
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut reasoning_entries: Vec<_> = reasoning_by_identifier.into_iter().collect();
    reasoning_entries.sort_by(|a, b| a.0.cmp(&b.0));
    let reasoning = reasoning_entries
        .into_iter()
        .map(|(_, value)| value)
        .filter(|value| !value.is_empty())
        .collect();

    let mut tool_call_entries: Vec<_> = tool_calls_by_index.into_iter().collect();
    tool_call_entries.sort_by_key(|(index, _)| *index);
    let tool_calls = tool_call_entries
        .into_iter()
        .map(|(_, value)| value)
        .filter(|value| !value.name.is_empty())
        .collect();

    (reasoning, tool_calls)
}

/// Aggregate deltas into a complete response.
fn aggregate_response(deltas: Vec<ClientDelta>) -> ClientResponse {
    let text = aggregate_message_text(&deltas);

    let finish_reason = deltas
        .iter()
        .rev()
        .find_map(|d| d.finish_reason.as_ref())
        .cloned();

    let usage = extract_usage(&deltas);
    let (reasoning, tool_calls) = aggregate_structured_items(&deltas);

    ClientResponse {
        text,
        finish_reason,
        usage,
        reasoning,
        tool_calls,
        deltas,
    }
}

fn extract_usage(deltas: &[ClientDelta]) -> UsageStats {
    let mut usage = UsageStats::default();
    let mut reasoning_tokens = 0;

    for delta in deltas {
        if let Some(count) = delta.prompt_token_count {
            usage.prompt_tokens = usage.prompt_tokens.max(count);
        }
        if let Some(count) = delta.reasoning_tokens {
            reasoning_tokens = reasoning_tokens.max(count);
        }
        if let Some(len) = delta.generation_len {
            usage.completion_tokens = usage
                .completion_tokens
                .max(len.saturating_sub(reasoning_tokens));
        }
    }

    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
    usage
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert_eq!(params.max_tokens, 8192);
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.top_k, -1);
        assert_eq!(params.repetition_context_size, 60);
        assert_eq!(params.top_logprobs, 0);
        assert!(params.logit_bias.is_empty());
        assert!(params.core_tools.is_empty());
        assert!(params.response_format.is_none());
        assert!(!params.deterministic);
        assert!(!params.reasoning);
        assert!(params.reasoning_effort.is_none());
        assert!(params.instructions.is_none());
    }

    #[test]
    fn test_native_reasoning_settings_drop_llama3_fallback_tokens() {
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            serde_json::json!({"model_type": "llama3"}).to_string(),
        )
        .unwrap();
        let formatter = crate::formatter::ChatFormatter::new(model_dir.path()).unwrap();

        let (reasoning_flag, reasoning_effort, thinking_tokens) =
            native_reasoning_settings(&formatter, true, &Some("high".to_string()));

        assert!(!reasoning_flag);
        assert!(reasoning_effort.is_none());
        assert_eq!(thinking_tokens.start, "");
        assert_eq!(thinking_tokens.end, "");
    }

    #[test]
    fn test_native_reasoning_settings_keep_gemma4_native_tokens() {
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            serde_json::json!({"model_type": "gemma4"}).to_string(),
        )
        .unwrap();
        let formatter = crate::formatter::ChatFormatter::new(model_dir.path()).unwrap();

        let (reasoning_flag, reasoning_effort, thinking_tokens) =
            native_reasoning_settings(&formatter, true, &Some("high".to_string()));

        assert!(reasoning_flag);
        assert_eq!(reasoning_effort.as_deref(), Some("high"));
        assert_eq!(thinking_tokens.start, "<|channel>thought\n");
        assert_eq!(thinking_tokens.end, "<channel|>");
    }

    #[test]
    fn test_native_reasoning_settings_effort_enables_gemma4_reasoning() {
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            serde_json::json!({"model_type": "gemma4"}).to_string(),
        )
        .unwrap();
        let formatter = crate::formatter::ChatFormatter::new(model_dir.path()).unwrap();

        let (reasoning_flag, reasoning_effort, thinking_tokens) =
            native_reasoning_settings(&formatter, false, &Some("high".to_string()));

        assert!(reasoning_flag);
        assert_eq!(reasoning_effort.as_deref(), Some("high"));
        assert_eq!(thinking_tokens.start, "<|channel>thought\n");
        assert_eq!(thinking_tokens.end, "<channel|>");
    }

    #[test]
    fn test_native_reasoning_settings_default_enables_gemma4_reasoning() {
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            serde_json::json!({"model_type": "gemma4"}).to_string(),
        )
        .unwrap();
        let formatter = crate::formatter::ChatFormatter::new(model_dir.path()).unwrap();

        let (reasoning_flag, reasoning_effort, thinking_tokens) =
            native_reasoning_settings(&formatter, formatter.supports_native_thinking(), &None);

        assert!(reasoning_flag);
        assert_eq!(reasoning_effort.as_deref(), Some("medium"));
        assert_eq!(thinking_tokens.start, "<|channel>thought\n");
        assert_eq!(thinking_tokens.end, "<channel|>");
    }

    #[test]
    fn test_native_reasoning_settings_boolean_reasoning_uses_default_effort() {
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            serde_json::json!({"model_type": "gemma4"}).to_string(),
        )
        .unwrap();
        let formatter = crate::formatter::ChatFormatter::new(model_dir.path()).unwrap();

        let (reasoning_flag, reasoning_effort, thinking_tokens) =
            native_reasoning_settings(&formatter, true, &None);

        assert!(reasoning_flag);
        assert_eq!(reasoning_effort.as_deref(), Some("medium"));
        assert_eq!(thinking_tokens.start, "<|channel>thought\n");
        assert_eq!(thinking_tokens.end, "<channel|>");
    }

    #[test]
    fn test_native_reasoning_settings_native_thinking_enables_afmoe_reasoning() {
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            serde_json::json!({"model_type": "afmoe"}).to_string(),
        )
        .unwrap();
        let formatter = crate::formatter::ChatFormatter::new(model_dir.path()).unwrap();

        let (reasoning_flag, reasoning_effort, thinking_tokens) =
            native_reasoning_settings(&formatter, formatter.supports_native_thinking(), &None);

        assert!(reasoning_flag);
        assert_eq!(reasoning_effort.as_deref(), Some("medium"));
        assert_eq!(thinking_tokens.start, "<think>\n");
        assert_eq!(thinking_tokens.end, "\n</think>");
    }

    #[test]
    fn test_aggregate_response() {
        let deltas = vec![
            ClientDelta {
                content: Some("Hello".to_string()),
                is_final: false,
                ..Default::default()
            },
            ClientDelta {
                content: Some(" World".to_string()),
                is_final: true,
                finish_reason: Some("stop".to_string()),
                ..Default::default()
            },
        ];

        let response = aggregate_response(deltas);
        assert_eq!(response.text, "Hello World");
        assert_eq!(response.finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_aggregate_response_uses_message_state_events() {
        let deltas = vec![
            ClientDelta {
                content: Some("hidden thought".to_string()),
                state_events: vec![crate::ipc::client::ResponseStateEvent {
                    event_type: "content_delta".to_string(),
                    item_type: "reasoning".to_string(),
                    identifier: "reasoning".to_string(),
                    delta: "hidden thought".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            },
            ClientDelta {
                content: Some("visible answer".to_string()),
                state_events: vec![crate::ipc::client::ResponseStateEvent {
                    event_type: "content_delta".to_string(),
                    item_type: "message".to_string(),
                    identifier: "message".to_string(),
                    delta: "visible answer".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            },
        ];

        let response = aggregate_response(deltas);

        assert_eq!(response.text, "visible answer");
        assert_eq!(response.reasoning, vec!["hidden thought".to_string()]);
    }

    #[test]
    fn test_aggregate_response_exposes_tool_calls() {
        let deltas = vec![ClientDelta {
            state_events: vec![crate::ipc::client::ResponseStateEvent {
                event_type: "item_completed".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "tool_call:share_to_party".to_string(),
                value: Some(serde_json::json!({
                    "name": "share_to_party",
                    "arguments": {"content": "hi"}
                })),
                ..Default::default()
            }],
            ..Default::default()
        }];

        let response = aggregate_response(deltas);

        assert_eq!(response.text, "");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "share_to_party");
        assert_eq!(
            response.tool_calls[0].arguments,
            serde_json::json!({"content": "hi"})
        );
    }

    #[test]
    fn test_aggregate_response_parses_tool_call_json_string_completion() {
        let deltas = vec![ClientDelta {
            state_events: vec![crate::ipc::client::ResponseStateEvent {
                event_type: "item_completed".to_string(),
                item_type: "tool_call".to_string(),
                output_index: 0,
                identifier: "tool_call:share_to_party".to_string(),
                value: Some(Value::String(
                    serde_json::json!({
                        "name": "share_to_party",
                        "arguments": {"content": "hi"}
                    })
                    .to_string(),
                )),
                ..Default::default()
            }],
            ..Default::default()
        }];

        let response = aggregate_response(deltas);

        assert_eq!(response.text, "");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "share_to_party");
        assert_eq!(
            response.tool_calls[0].arguments,
            serde_json::json!({"content": "hi"})
        );
    }

    #[test]
    fn test_build_embedding_prompt_payload() {
        let payload = build_embedding_prompt_payload("hello".to_string());

        assert_eq!(payload.prompt, "hello");
        assert_eq!(payload.max_generated_tokens, 0);
        assert_eq!(payload.layout.len(), 1);
        assert_eq!(payload.layout[0].segment_type, "text");
        assert_eq!(payload.layout[0].length, 5);
        assert_eq!(payload.num_candidates, 1);
        assert_eq!(payload.best_of, Some(1));
        assert_eq!(payload.final_candidates, Some(1));
    }

    #[test]
    fn test_decode_embedding_bytes() {
        let bytes = [
            0.0f32.to_le_bytes(),
            1.5f32.to_le_bytes(),
            (-2.25f32).to_le_bytes(),
        ]
        .concat();

        let embedding = decode_embedding_bytes(&bytes).expect("embedding should decode");
        assert_eq!(embedding, vec![0.0, 1.5, -2.25]);
    }

    #[test]
    fn test_decode_embedding_bytes_rejects_partial_float() {
        let error = decode_embedding_bytes(&[0, 0, 128]).expect_err("decode should fail");
        assert!(matches!(error, ClientError::RequestFailed(_)));
    }

    #[test]
    fn test_build_stt_prompt_payload() {
        let payload = build_stt_prompt_payload(&[0.25, -0.5]);

        assert!(payload.prompt.is_empty());
        assert_eq!(payload.capabilities.len(), 1);
        assert_eq!(payload.capabilities[0].name, "audio");
        assert_eq!(payload.capabilities[0].payload.len(), 8);
        assert_eq!(payload.layout.len(), 2);
        assert_eq!(payload.layout[0].segment_type, "text");
        assert_eq!(payload.layout[0].length, 0);
        assert_eq!(payload.layout[1].segment_type, "capability");
        assert_eq!(payload.layout[1].length, 8);
    }

    #[test]
    fn test_encode_float32_pcm_bytes() {
        let bytes = encode_float32_pcm_bytes(&[0.0, 1.5, -2.25]);
        let decoded = decode_embedding_bytes(&bytes).expect("audio bytes should decode");
        assert_eq!(decoded, vec![0.0, 1.5, -2.25]);
    }

    #[test]
    fn test_build_response_from_candidates_uses_total_completion_tokens() {
        let response = build_response_from_candidates(
            vec![CandidateState {
                content: "winner".to_string(),
                finish_reason: Some("stop".to_string()),
                completion_tokens: 2,
                prompt_tokens: 5,
                deltas: vec![ClientDelta {
                    content: Some("winner".to_string()),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            /*total_completion_tokens=*/ 7,
        );

        assert_eq!(response.text, "winner");
        assert_eq!(response.finish_reason, Some("stop".to_string()));
        assert_eq!(response.usage.prompt_tokens, 5);
        assert_eq!(response.usage.completion_tokens, 7);
        assert_eq!(response.usage.total_tokens, 12);
    }

    #[test]
    fn test_build_response_from_candidates_uses_message_state_events() {
        let response = build_response_from_candidates(
            vec![CandidateState {
                content: "<think>private</think>{\"ok\":true}".to_string(),
                finish_reason: Some("stop".to_string()),
                prompt_tokens: 5,
                deltas: vec![
                    ClientDelta {
                        state_events: vec![crate::ipc::client::ResponseStateEvent {
                            event_type: "content_delta".to_string(),
                            item_type: "reasoning".to_string(),
                            identifier: "reasoning".to_string(),
                            delta: "private".to_string(),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    ClientDelta {
                        state_events: vec![crate::ipc::client::ResponseStateEvent {
                            event_type: "content_delta".to_string(),
                            item_type: "message".to_string(),
                            identifier: "message".to_string(),
                            delta: "{\"ok\":true}".to_string(),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            2,
        );

        assert_eq!(response.text, "{\"ok\":true}");
        assert_eq!(response.reasoning, vec!["private".to_string()]);
    }

    #[test]
    fn test_build_response_from_candidates_keeps_plain_raw_content() {
        let response = build_response_from_candidates(
            vec![CandidateState {
                content: "red, white, blue".to_string(),
                finish_reason: Some("stop".to_string()),
                prompt_tokens: 5,
                deltas: vec![ClientDelta {
                    content: Some("red, white, blue".to_string()),
                    state_events: vec![crate::ipc::client::ResponseStateEvent {
                        event_type: "content_delta".to_string(),
                        item_type: "message".to_string(),
                        identifier: "message".to_string(),
                        delta: "red, white, ".to_string(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            3,
        );

        assert_eq!(response.text, "red, white, blue");
    }

    #[test]
    fn test_build_response_from_candidates_uses_message_delta_when_raw_content_empty() {
        let response = build_response_from_candidates(
            vec![CandidateState {
                content: "red, white, ".to_string(),
                finish_reason: Some("stop".to_string()),
                prompt_tokens: 5,
                deltas: vec![
                    ClientDelta {
                        content: Some("red, white, ".to_string()),
                        ..Default::default()
                    },
                    ClientDelta {
                        content: Some(String::new()),
                        state_events: vec![crate::ipc::client::ResponseStateEvent {
                            event_type: "content_delta".to_string(),
                            item_type: "message".to_string(),
                            identifier: "message".to_string(),
                            delta: "blue".to_string(),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            3,
        );

        assert_eq!(response.text, "red, white, blue");
    }

    #[test]
    fn test_build_response_from_candidates_preserves_raw_suffix_after_state_text() {
        let response = build_response_from_candidates(
            vec![CandidateState {
                content: "red, white, blue".to_string(),
                finish_reason: Some("stop".to_string()),
                prompt_tokens: 5,
                deltas: vec![
                    ClientDelta {
                        content: Some("<think>hidden</think>red, white, ".to_string()),
                        state_events: vec![
                            crate::ipc::client::ResponseStateEvent {
                                event_type: "content_delta".to_string(),
                                item_type: "reasoning".to_string(),
                                identifier: "reasoning".to_string(),
                                delta: "hidden".to_string(),
                                ..Default::default()
                            },
                            crate::ipc::client::ResponseStateEvent {
                                event_type: "content_delta".to_string(),
                                item_type: "message".to_string(),
                                identifier: "message".to_string(),
                                delta: "red, white, ".to_string(),
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                    ClientDelta {
                        content: Some("blue".to_string()),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            3,
        );

        assert_eq!(response.text, "red, white, blue");
        assert_eq!(response.reasoning, vec!["hidden".to_string()]);
    }
}
