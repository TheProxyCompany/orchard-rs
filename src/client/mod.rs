//! High-level client API for Orchard.
//!
//! Provides the main user-facing interface for LLM inference.

mod moondream;
mod response;
mod responses;

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::mpsc;

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
use crate::ipc::serialization::{CapabilityEntry, LayoutEntry, PromptPayload};
use crate::model::registry::ModelRegistry;

pub use moondream::{
    BoundingBox, CaptionResult, DetectResult, DetectedObject, GazeResult, GroundingSpan,
    MoondreamClient, Point, PointResult, QueryResult, ReasoningOutput, SpatialRef,
    MOONDREAM_MODEL_ID,
};
pub use response::{BatchChatResult, ClientDelta, ClientResponse, UsageStats};
pub use responses::{
    ContentPartAddedEvent, ContentPartDoneEvent, FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent, IncompleteDetails, InputTokensDetails, OutputFunctionCall,
    OutputItemAddedEvent, OutputItemDoneEvent, OutputMessage, OutputReasoning, OutputStatus,
    OutputTextContent, OutputTextDeltaEvent, OutputTextDoneEvent, OutputTokensDetails,
    ReasoningContent, ReasoningDeltaEvent, ReasoningDoneEvent, ReasoningSummaryTextContent,
    ReasoningSummaryTextDeltaEvent, ReasoningSummaryTextDoneEvent, ResponseCompletedEvent,
    ResponseCreatedEvent, ResponseError, ResponseEvent, ResponseFailedEvent,
    ResponseInProgressEvent, ResponseIncompleteEvent, ResponseInputItem, ResponseObject,
    ResponseOutputItem, ResponseSnapshot, ResponseUsage, ResponsesInput, ResponsesRequest,
    ResponsesResult, StreamErrorDetail, StreamErrorEvent,
};

/// Errors that can occur during client operations.
#[derive(Error, Debug)]
pub enum ClientError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model not ready: {0}")]
    ModelNotReady(String),

    #[error("IPC error: {0}")]
    Ipc(String),

    #[error("Formatter error: {0}")]
    Formatter(String),

    #[error("Multimodal error: {0}")]
    Multimodal(String),

    #[error("Request failed: {0}")]
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

use crate::defaults;

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
    pub tools: Vec<serde_json::Value>,
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
            tools: Vec::new(),
            tool_choice: None,
            max_tool_calls: None,
            response_format: None,
            reasoning: false,
            reasoning_effort: None,
            instructions: None,
            task_name: None,
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

/// A high-level client for the Proxy Inference Engine.
///
/// Provides both synchronous and asynchronous interfaces for LLM inference.
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
    /// - Event callback for handling model_loaded events
    /// - IPC client shared with registry for management commands
    pub async fn connect(registry: Arc<ModelRegistry>) -> Result<Self> {
        // Create event callback that routes model_loaded events to registry
        let registry_for_events = Arc::clone(&registry);
        let runtime_handle = tokio::runtime::Handle::current();
        let event_callback: EventCallback = Arc::new(move |event_name: &str, payload: &Value| {
            if event_name == "model_loaded" {
                let registry = Arc::clone(&registry_for_events);
                let payload = payload.clone();
                // Spawn a task to handle the event (handle_model_loaded is async)
                let handle = runtime_handle.clone();
                handle.spawn(async move {
                    registry.handle_model_loaded(&payload).await;
                });
            }
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

        // Compute reasoning flag (same as Python: reasoning OR reasoning_effort present)
        let reasoning_flag = params.reasoning || params.reasoning_effort.is_some();

        // Build multimodal content (pass instructions if provided)
        let (messages_for_template, image_buffers, capabilities, content_order) =
            build_multimodal_messages(&info.formatter, &messages, params.instructions.as_deref())
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

        // Apply template with reasoning flag
        let prompt_text = info
            .formatter
            .apply_template(
                &messages_for_template,
                true,
                reasoning_flag,
                params.task_name.as_deref(),
            )
            .map_err(|e| ClientError::Formatter(e.to_string()))?;

        let capability_placeholder = info.formatter.capability_placeholder_token();

        // Build layout for multimodal content
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

        // Serialize tools and response_format to JSON strings (matching Python)
        let tool_schemas_json = if params.tools.is_empty() {
            String::new()
        } else {
            serde_json::to_string(&params.tools).unwrap_or_default()
        };
        let response_format_json = params
            .response_format
            .as_ref()
            .map(|rf| serde_json::to_string(rf).unwrap_or_default())
            .unwrap_or_default();
        let tool_calling_tokens = info.formatter.get_tool_calling_tokens().clone();
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
            tool_calling_tokens,
            tool_choice,
            max_tool_calls,
            response_format_json,
            task_name: params.task_name.clone(),
            reasoning_effort: params.reasoning_effort.clone(),
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
            model_id,
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

                        let client_delta = ClientDelta::from(delta.clone());
                        state.deltas.push(client_delta);

                        if delta.is_final_delta && !state.completed {
                            state.completed = true;
                            state.finish_reason = delta.finish_reason.clone();
                            state.cumulative_logprob = delta.cumulative_logprob;
                            state.generation_len = delta.generation_len;
                            remaining_sequences -= 1;
                        }
                    }
                    None => break,
                }
            }

            // Score and select best candidates (matching Python logic)
            let selected = select_best_candidates(candidate_states, best_of, final_candidates);

            Ok(ChatResult::Complete(build_response_from_candidates(
                selected,
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
        if conversations.is_empty() {
            return Ok(BatchChatResult::Complete(Vec::new()));
        }

        let info = self.registry.ensure_loaded(model_id).await?;

        let request_id = self.ipc.next_request_id();
        let num_prompts = conversations.len();
        tracing::debug!(
            request_id,
            model_id = %model_id,
            stream,
            prompt_count = num_prompts,
            "Building batched chat request"
        );

        // Compute reasoning flag (same as Python: reasoning OR reasoning_effort present)
        let reasoning_flag = params.reasoning || params.reasoning_effort.is_some();

        // Serialize tools and response_format to JSON strings (matching Python)
        let tool_schemas_json = if params.tools.is_empty() {
            String::new()
        } else {
            serde_json::to_string(&params.tools).unwrap_or_default()
        };
        let response_format_json = params
            .response_format
            .as_ref()
            .map(|rf| serde_json::to_string(rf).unwrap_or_default())
            .unwrap_or_default();
        let tool_calling_tokens = info.formatter.get_tool_calling_tokens().clone();
        let tool_choice = tool_choice_to_string(params.tool_choice.as_ref());
        let max_tool_calls = params.max_tool_calls.unwrap_or(0).max(0);

        // Build all prompt payloads
        let mut prompt_payloads = Vec::with_capacity(num_prompts);

        for (prompt_index, messages) in conversations.iter().enumerate() {
            // Build multimodal content (pass instructions if provided)
            let (messages_for_template, image_buffers, capabilities, content_order) =
                build_multimodal_messages(
                    &info.formatter,
                    messages,
                    params.instructions.as_deref(),
                )
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

            // Apply template with reasoning flag
            let prompt_text = info
                .formatter
                .apply_template(
                    &messages_for_template,
                    true,
                    reasoning_flag,
                    params.task_name.as_deref(),
                )
                .map_err(|e| ClientError::Formatter(e.to_string()))?;

            let capability_placeholder = info.formatter.capability_placeholder_token();

            // Build layout for multimodal content
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
                tool_calling_tokens: tool_calling_tokens.clone(),
                tool_choice: tool_choice.clone(),
                max_tool_calls,
                response_format_json: response_format_json.clone(),
                task_name: params.task_name.clone(),
                reasoning_effort: params.reasoning_effort.clone(),
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
            model_id,
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

/// Build a ClientResponse from selected candidates, consuming them to avoid clones.
fn build_response_from_candidates(candidates: Vec<CandidateState>) -> ClientResponse {
    let prompt_tokens = candidates
        .iter()
        .map(|c| c.prompt_tokens)
        .max()
        .unwrap_or(0);
    let completion_tokens: u32 = candidates.iter().map(|c| c.completion_tokens).sum();

    let capacity: usize = candidates.iter().map(|c| c.deltas.len()).sum();
    let mut all_deltas = Vec::with_capacity(capacity);
    let mut text = String::new();
    let mut finish_reason = None;

    for candidate in candidates {
        text.push_str(&candidate.content);
        if candidate.finish_reason.is_some() {
            finish_reason = candidate.finish_reason;
        }
        all_deltas.extend(candidate.deltas);
    }

    ClientResponse {
        text,
        finish_reason,
        usage: UsageStats {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        deltas: all_deltas,
    }
}

/// Aggregate deltas into a complete response.
fn aggregate_response(deltas: Vec<ClientDelta>) -> ClientResponse {
    let text: String = deltas
        .iter()
        .filter_map(|d| d.content.as_ref())
        .cloned()
        .collect();

    let finish_reason = deltas
        .iter()
        .rev()
        .find_map(|d| d.finish_reason.as_ref())
        .cloned();

    let usage = extract_usage(&deltas);

    ClientResponse {
        text,
        finish_reason,
        usage,
        deltas,
    }
}

fn extract_usage(deltas: &[ClientDelta]) -> UsageStats {
    let mut usage = UsageStats::default();

    for delta in deltas {
        if let Some(count) = delta.prompt_token_count {
            usage.prompt_tokens = usage.prompt_tokens.max(count);
        }
        if let Some(len) = delta.generation_len {
            usage.completion_tokens = usage.completion_tokens.max(len);
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
        assert_eq!(params.max_tokens, 1024);
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.top_k, -1);
        assert_eq!(params.repetition_context_size, 60);
        assert_eq!(params.top_logprobs, 0);
        assert!(params.logit_bias.is_empty());
        assert!(params.tools.is_empty());
        assert!(params.response_format.is_none());
        assert!(!params.reasoning);
        assert!(params.reasoning_effort.is_none());
        assert!(params.instructions.is_none());
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
}
