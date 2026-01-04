//! Response types for the client API.

use serde::{Deserialize, Serialize};

use crate::ipc::client::ResponseDelta;

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageStats {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens generated
    pub completion_tokens: u32,
    /// Total tokens (prompt + completion)
    pub total_tokens: u32,
}

/// A single delta from a streaming response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientDelta {
    /// Request ID
    pub request_id: u64,
    /// Prompt index (for batched requests)
    pub prompt_index: Option<u32>,
    /// Generated content
    pub content: Option<String>,
    /// Whether this is the final delta
    pub is_final: bool,
    /// Finish reason (e.g., "stop", "length")
    pub finish_reason: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// Prompt token count
    pub prompt_token_count: Option<u32>,
    /// Current generation length
    pub generation_len: Option<u32>,
}

impl From<ResponseDelta> for ClientDelta {
    fn from(delta: ResponseDelta) -> Self {
        Self {
            request_id: delta.request_id,
            prompt_index: delta.prompt_index,
            content: delta.content,
            is_final: delta.is_final_delta,
            finish_reason: delta.finish_reason,
            error: delta.error,
            prompt_token_count: None,
            generation_len: None,
        }
    }
}

/// A complete response from a chat completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientResponse {
    /// Aggregated text content
    pub text: String,
    /// Finish reason
    pub finish_reason: Option<String>,
    /// Token usage statistics
    pub usage: UsageStats,
    /// Individual deltas (for inspection)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub deltas: Vec<ClientDelta>,
}

impl Default for ClientResponse {
    fn default() -> Self {
        Self {
            text: String::new(),
            finish_reason: None,
            usage: UsageStats::default(),
            deltas: Vec::new(),
        }
    }
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
            prompt_index: Some(0),
            content: Some("Hello".to_string()),
            is_final_delta: false,
            finish_reason: None,
            error: None,
        };

        let delta = ClientDelta::from(response);
        assert_eq!(delta.request_id, 123);
        assert_eq!(delta.prompt_index, Some(0));
        assert_eq!(delta.content, Some("Hello".to_string()));
        assert!(!delta.is_final);
    }
}
