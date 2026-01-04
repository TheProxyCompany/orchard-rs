//! High-level client API for Orchard.
//!
//! Provides the main user-facing interface for LLM inference.

mod moondream;
mod response;

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::formatter::multimodal::{build_multimodal_layout, build_multimodal_messages};
use crate::ipc::client::{IPCClient, RequestOptions, ResponseDelta};
use crate::ipc::serialization::PromptPayload;
use crate::model::registry::ModelRegistry;

pub use moondream::{
    BoundingBox, CaptionResult, DetectResult, DetectedObject, GazeResult, GroundingSpan,
    MoondreamClient, Point, PointResult, QueryResult, ReasoningOutput, SpatialRef,
    MOONDREAM_MODEL_ID,
};
pub use response::{ClientDelta, ClientResponse, UsageStats};

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
        ClientError::Ipc(err.to_string())
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
    ipc: IPCClient,
    registry: Arc<ModelRegistry>,
}

impl Client {
    /// Create a new client with the given IPC client and model registry.
    pub fn new(ipc: IPCClient, registry: Arc<ModelRegistry>) -> Self {
        Self { ipc, registry }
    }

    /// Create a client and connect to the engine.
    pub fn connect(registry: Arc<ModelRegistry>) -> Result<Self> {
        let mut ipc = IPCClient::new();
        ipc.connect()?;
        Ok(Self::new(ipc, registry))
    }

    /// Disconnect from the engine.
    pub fn disconnect(&mut self) {
        self.ipc.disconnect();
    }

    /// Resolve control token capabilities for a model.
    pub async fn resolve_capabilities(&self, model_id: &str) -> Result<HashMap<String, i32>> {
        let info = self
            .registry
            .ensure_loaded(model_id)
            .await
            .map_err(ClientError::ModelNotReady)?;

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
        let info = self
            .registry
            .ensure_loaded(model_id)
            .await
            .map_err(ClientError::ModelNotReady)?;

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
            .apply_template(&messages_for_template, true, reasoning_flag, params.task_name.as_deref())
            .map_err(|e| ClientError::Formatter(e.to_string()))?;

        // Build layout (currently unused, but validates the multimodal structure)
        let _layout_segments = build_multimodal_layout(
            &prompt_text,
            &image_buffers,
            &capabilities,
            &content_order,
            info.formatter
                .control_tokens
                .start_image_token
                .as_deref()
                .unwrap_or(info.formatter.default_image_placeholder()),
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

        // Send request
        let options = RequestOptions {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            stop_sequences: params.stop.clone(),
        };

        let rx = self.ipc.send_request(
            request_id,
            model_id,
            &info.model_path,
            &final_prompt,
            options,
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
    pub fn chat(
        &self,
        model_id: &str,
        messages: Vec<HashMap<String, serde_json::Value>>,
        params: SamplingParams,
    ) -> Result<ClientResponse> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ClientError::RequestFailed(e.to_string()))?;

        rt.block_on(async {
            match self.achat(model_id, messages, params, false).await? {
                ChatResult::Complete(response) => Ok(response),
                ChatResult::Stream(_) => Err(ClientError::RequestFailed(
                    "Unexpected stream result".into(),
                )),
            }
        })
    }

    /// Perform batched chat completion.
    ///
    /// This sends ALL conversations in ONE IPC message, allowing the engine
    /// to schedule them together efficiently. Responses are demultiplexed
    /// by prompt_index and returned in order.
    pub async fn achat_batch(
        &self,
        model_id: &str,
        conversations: Vec<Vec<HashMap<String, serde_json::Value>>>,
        params: SamplingParams,
    ) -> Result<Vec<ClientResponse>> {
        if conversations.is_empty() {
            return Ok(Vec::new());
        }

        let info = self
            .registry
            .ensure_loaded(model_id)
            .await
            .map_err(ClientError::ModelNotReady)?;

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
            let (messages_for_template, _image_buffers, _capabilities, _content_order) =
                build_multimodal_messages(&info.formatter, messages, params.instructions.as_deref())
                    .map_err(|e| ClientError::Multimodal(e.to_string()))?;

            if messages_for_template.is_empty() {
                return Err(ClientError::RequestFailed(
                    "Chat request must include at least one message".into(),
                ));
            }

            // Apply template with reasoning flag
            let prompt_text = info
                .formatter
                .apply_template(&messages_for_template, true, reasoning_flag, params.task_name.as_deref())
                .map_err(|e| ClientError::Formatter(e.to_string()))?;

            // Build final prompt (with placeholders removed if needed)
            let mut final_prompt = prompt_text;
            if info.formatter.should_clip_image_placeholder() {
                final_prompt = final_prompt.replace(info.formatter.default_image_placeholder(), "");
            }
            if let Some(coord) = &info.formatter.control_tokens.coord_placeholder {
                final_prompt = final_prompt.replace(coord, "");
            }

            prompt_payloads.push(PromptPayload {
                prompt: final_prompt,
                max_generated_tokens: params.max_tokens,
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                min_p: params.min_p,
                rng_seed: params.rng_seed,
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
                ..Default::default()
            });
        }

        // Send ONE batch request with all prompts
        let (_batch_size, mut rx) = self.ipc.send_batch_request(
            request_id,
            model_id,
            &info.model_path,
            &prompt_payloads,
        )?;

        // Collect responses grouped by prompt_index
        let mut deltas_by_prompt: HashMap<u32, Vec<ClientDelta>> = HashMap::new();
        let mut finals_received = 0usize;

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

        Ok(responses)
    }
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
