use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde_json::Value;

use crate::client::{Client, ClientDelta, ClientError, Result};
use crate::model::registry::ModelRegistry;

pub const OPENAI_PRIVACY_FILTER_MODEL_ID: &str = "openai/privacy-filter";

pub struct OpenAIPrivacyFilterClient {
    client: Client,
}

impl OpenAIPrivacyFilterClient {
    pub const MAX_INPUT_BYTES: usize = 16 * 1024;

    pub async fn new(client: Client, registry: Arc<ModelRegistry>) -> Result<Self> {
        registry
            .ensure_loaded(OPENAI_PRIVACY_FILTER_MODEL_ID)
            .await?;
        Ok(Self { client })
    }

    pub async fn analyze(&self, text: &str) -> Result<Value> {
        Self::check_input_size(text)?;
        let deltas = self
            .client
            .aprefill_task(OPENAI_PRIVACY_FILTER_MODEL_ID, text, "privacy_filter")
            .await?;
        Self::payload_from_deltas(deltas)
    }

    pub async fn analyze_batch(&self, texts: Vec<String>) -> Result<Vec<Value>> {
        for text in &texts {
            Self::check_input_size(text)?;
        }
        let deltas_by_prompt = self
            .client
            .aprefill_task_batch(OPENAI_PRIVACY_FILTER_MODEL_ID, texts, "privacy_filter")
            .await?;
        deltas_by_prompt
            .into_iter()
            .map(Self::payload_from_deltas)
            .collect()
    }

    fn payload_from_deltas(deltas: Vec<ClientDelta>) -> Result<Value> {
        for delta in deltas {
            if delta.modal_decoder_id.as_deref() == Some("privacy_filter") {
                if let Some(payload_b64) = delta.modal_bytes_b64 {
                    let payload = BASE64.decode(payload_b64).map_err(|e| {
                        ClientError::Multimodal(format!(
                            "Failed to decode privacy filter payload: {}",
                            e
                        ))
                    })?;
                    return serde_json::from_slice(&payload).map_err(|e| {
                        ClientError::Multimodal(format!(
                            "Failed to parse privacy filter payload: {}",
                            e
                        ))
                    });
                }
            }
        }
        Err(ClientError::RequestFailed(
            "privacy filter response did not include a result payload".to_string(),
        ))
    }

    fn check_input_size(text: &str) -> Result<()> {
        if text.len() > Self::MAX_INPUT_BYTES {
            return Err(ClientError::RequestFailed(format!(
                "privacy filter input exceeds {} bytes; callers must chunk larger inputs",
                Self::MAX_INPUT_BYTES
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn decodes_modal_payload() {
        let payload = json!({
            "type": "openai_privacy_filter.token_classification",
            "token_count": 1,
            "label_ids": [0],
            "labels": ["O"],
            "scores": [0.25],
        });
        let encoded = BASE64.encode(payload.to_string());
        let decoded = OpenAIPrivacyFilterClient::payload_from_deltas(vec![ClientDelta {
            request_id: 1,
            prompt_index: Some(0),
            modal_decoder_id: Some("privacy_filter".to_string()),
            modal_bytes_b64: Some(encoded),
            ..Default::default()
        }])
        .unwrap();

        assert_eq!(decoded, payload);
    }
}
