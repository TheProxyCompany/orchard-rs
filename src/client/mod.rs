//! High-level client API for Orchard.
//!
//! Provides the main user-facing interface for LLM inference.

mod moondream;
mod response;

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
            response_format: None,
            reasoning: false,
            reasoning_effort: None,
            instructions: None,
            task_name: None,
        }
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

        // Build layout for multimodal content
        let layout_segments = build_multimodal_layout(
            &prompt_text,
            &image_buffers,
            &capabilities,
            &content_order,
            info.formatter.image_placeholder_token(),
            info.formatter.should_clip_image_placeholder(),
            info.formatter.control_tokens.coord_placeholder.as_deref(),
        )
        .map_err(|e| ClientError::Multimodal(e.to_string()))?;

        // Build final prompt (with placeholders removed if needed)
        let mut final_prompt = prompt_text;
        if info.formatter.should_clip_image_placeholder() {
            final_prompt = final_prompt.replace(info.formatter.default_image_placeholder(), "");
        }
        if let Some(coord) = &info.formatter.control_tokens.coord_placeholder {
            final_prompt = final_prompt.replace(coord, "");
        }

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
            response_format_json,
            task_name: params.task_name.clone(),
            reasoning_effort: params.reasoning_effort.clone(),
        };

        // Use unified batch request path (even for single prompts)
        let (_batch_size, rx) = self.ipc.send_batch_request(
            request_id,
            model_id,
            &info.model_path,
            &[prompt_payload],
        )?;

        if stream {
            Ok(ChatResult::Stream(rx))
        } else {
            // Collect all deltas
            let mut deltas = Vec::new();
            let mut rx = rx;
            while let Some(delta) = rx.recv().await {
                let is_final = delta.is_final_delta;
                deltas.push(ClientDelta::from(delta));
                if is_final {
                    break;
                }
            }

            Ok(ChatResult::Complete(aggregate_response(deltas)))
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

        // Build all prompt payloads
        let mut prompt_payloads = Vec::with_capacity(num_prompts);

        for messages in &conversations {
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

            // Build layout for multimodal content
            let layout_segments = build_multimodal_layout(
                &prompt_text,
                &image_buffers,
                &capabilities,
                &content_order,
                info.formatter.image_placeholder_token(),
                info.formatter.should_clip_image_placeholder(),
                info.formatter.control_tokens.coord_placeholder.as_deref(),
            )
            .map_err(|e| ClientError::Multimodal(e.to_string()))?;

            // Build final prompt (with placeholders removed if needed)
            let mut final_prompt = prompt_text;
            if info.formatter.should_clip_image_placeholder() {
                final_prompt = final_prompt.replace(info.formatter.default_image_placeholder(), "");
            }
            if let Some(coord) = &info.formatter.control_tokens.coord_placeholder {
                final_prompt = final_prompt.replace(coord, "");
            }

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
                response_format_json: response_format_json.clone(),
                task_name: params.task_name.clone(),
                reasoning_effort: params.reasoning_effort.clone(),
            });
        }

        // Send ONE batch request with all prompts
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
