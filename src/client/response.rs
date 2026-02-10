//! Response types for the client API.

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::ipc::client::ResponseDelta;

/// Result of a batch chat completion that can be streamed or complete.
pub enum BatchChatResult {
    /// Complete responses for all prompts
    Complete(Vec<ClientResponse>),
    /// Stream of deltas (contains prompt_index to identify which prompt)
    Stream(mpsc::Receiver<ClientDelta>),
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// A single delta from a streaming response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientDelta {
    pub request_id: u64,
    pub sequence_id: Option<u64>,
    pub prompt_index: Option<u32>,
    pub candidate_index: Option<u32>,
    pub content: Option<String>,
    pub content_len: Option<u32>,
    pub inline_content_bytes: Option<u32>,
    pub is_final: bool,
    pub finish_reason: Option<String>,
    pub error: Option<String>,
    pub prompt_token_count: Option<u32>,
    pub num_tokens_in_delta: Option<u32>,
    pub generation_len: Option<u32>,
    pub tokens: Vec<i32>,
    pub top_logprobs: Vec<crate::ipc::client::TokenLogProb>,
    pub cumulative_logprob: Option<f64>,
    pub modal_decoder_id: Option<String>,
    pub modal_bytes_b64: Option<String>,
}

impl From<ResponseDelta> for ClientDelta {
    fn from(delta: ResponseDelta) -> Self {
        Self {
            request_id: delta.request_id,
            sequence_id: delta.sequence_id,
            prompt_index: delta.prompt_index,
            candidate_index: delta.candidate_index,
            content: delta.content,
            content_len: delta.content_len,
            inline_content_bytes: delta.inline_content_bytes,
            is_final: delta.is_final_delta,
            finish_reason: delta.finish_reason,
            error: delta.error,
            prompt_token_count: delta.prompt_token_count,
            num_tokens_in_delta: delta.num_tokens_in_delta,
            generation_len: delta.generation_len,
            tokens: delta.tokens,
            top_logprobs: delta.top_logprobs,
            cumulative_logprob: delta.cumulative_logprob,
            modal_decoder_id: delta.modal_decoder_id,
            modal_bytes_b64: delta.modal_bytes_b64,
        }
    }
}

/// A complete response from a chat completion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientResponse {
    pub text: String,
    pub finish_reason: Option<String>,
    pub usage: UsageStats,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub deltas: Vec<ClientDelta>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_stats_default() {
        let usage = UsageStats::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_client_delta_from_response_delta() {
        let response = ResponseDelta {
            request_id: 123,
            sequence_id: Some(1),
            prompt_index: Some(0),
            candidate_index: Some(0),
            content: Some("Hello".to_string()),
            content_len: Some(5),
            inline_content_bytes: Some(5),
            is_final_delta: false,
            finish_reason: None,
            error: None,
            prompt_token_count: Some(10),
            num_tokens_in_delta: Some(3),
            generation_len: Some(5),
            tokens: vec![1, 2, 3],
            top_logprobs: vec![],
            cumulative_logprob: Some(-1.5),
            modal_decoder_id: Some("moondream3.coord".to_string()),
            modal_bytes_b64: Some("AAAA".to_string()),
            state_events: vec![],
            cached_token_count: Some(0),
            reasoning_tokens: Some(0),
        };

        let delta = ClientDelta::from(response);
        assert_eq!(delta.request_id, 123);
        assert_eq!(delta.sequence_id, Some(1));
        assert_eq!(delta.prompt_index, Some(0));
        assert_eq!(delta.candidate_index, Some(0));
        assert_eq!(delta.content, Some("Hello".to_string()));
        assert_eq!(delta.content_len, Some(5));
        assert_eq!(delta.num_tokens_in_delta, Some(3));
        assert!(!delta.is_final);
        assert_eq!(delta.tokens, vec![1, 2, 3]);
        assert_eq!(delta.cumulative_logprob, Some(-1.5));
        assert_eq!(delta.modal_decoder_id, Some("moondream3.coord".to_string()));
    }
}
