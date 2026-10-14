//! Profile-selected image operations. Hosts select a model and an operation;
//! prompt tasks and spatial decoding stay with the compatible SDK adapter.
use super::{
    CaptionWithMetrics, ChatResult, Client, ClientDelta, MoondreamClient, PointResult,
    SamplingParams,
};
use crate::{Error, ModelLoadRequest, ModelOperations, ModelRegistry, Result};
use serde_json::json;
use std::{collections::HashMap, sync::Arc, time::Instant};

pub const DEFAULT_VISION_MODEL: &str = super::MOONDREAM_MODEL_ID;

enum Adapter {
    Tasks(MoondreamClient),
    ImageChat(Client),
}

pub struct VisionClient {
    model: String,
    operations: ModelOperations,
    adapter: Adapter,
}

impl VisionClient {
    pub async fn with_model(
        client: Client,
        registry: Arc<ModelRegistry>,
        model: &str,
    ) -> Result<Self> {
        let operations = registry
            .model_operations(&ModelLoadRequest::automatic(model))
            .await?;
        operations.require(&["vision_query"])?;
        let adapter = if operations.vision_tasks {
            Adapter::Tasks(MoondreamClient::with_model(client, registry, model).await?)
        } else {
            Adapter::ImageChat(client)
        };
        Ok(Self {
            model: model.into(),
            operations,
            adapter,
        })
    }

    pub fn operations(&self) -> &ModelOperations {
        &self.operations
    }

    pub async fn query(
        &self,
        image: &str,
        question: &str,
        params: SamplingParams,
    ) -> Result<String> {
        match &self.adapter {
            Adapter::Tasks(client) => client
                .query(question, Some(image), &[], false, params)
                .await
                .map(|result| result.answer),
            Adapter::ImageChat(client) => self
                .image_chat(client, image, question, params)
                .await
                .map(|result| result.caption),
        }
    }

    pub async fn describe(
        &self,
        image: &str,
        params: SamplingParams,
    ) -> Result<CaptionWithMetrics> {
        match &self.adapter {
            Adapter::Tasks(client) => client.caption_with_metrics(image, "short", params).await,
            Adapter::ImageChat(client) => {
                self.image_chat(
                    client,
                    image,
                    "Describe the visible scene briefly and factually.",
                    params,
                )
                .await
            }
        }
    }

    pub async fn point(
        &self,
        image: &str,
        object: &str,
        params: SamplingParams,
    ) -> Result<PointResult> {
        self.operations.require(&["vision_point"])?;
        match &self.adapter {
            Adapter::Tasks(client) => client.point(image, object, params).await,
            Adapter::ImageChat(_) => Err(Error::Other(
                "The selected vision adapter has no normalized-point operation".into(),
            )),
        }
    }

    async fn image_chat(
        &self,
        client: &Client,
        image: &str,
        question: &str,
        mut params: SamplingParams,
    ) -> Result<CaptionWithMetrics> {
        let began = Instant::now();
        // No vendor task name or implicit reasoning changes in ordinary image chat.
        params.task_name = None;
        params.reasoning = Some(false);
        let messages = vec![HashMap::from([
            ("role".into(), json!("user")),
            (
                "content".into(),
                json!([
                    {"type":"input_image", "image_url":image},
                    {"type":"input_text", "text":question},
                ]),
            ),
        ])];
        let ChatResult::Stream(mut stream) =
            client.achat(&self.model, messages, params, true).await?
        else {
            return Err(Error::Other("Expected a streaming image response".into()));
        };
        let mut result = CaptionWithMetrics {
            caption: String::new(),
            model_id: self.model.clone(),
            prompt_tokens: 0,
            cached_tokens: 0,
            completion_tokens: 0,
            elapsed_ms: 0.0,
            first_token_ms: None,
        };
        while let Some(delta) = stream.recv().await {
            if let Some(error) = &delta.error {
                return Err(Error::Other(error.clone()));
            }
            result.prompt_tokens = result
                .prompt_tokens
                .max(delta.prompt_token_count.unwrap_or(0));
            result.cached_tokens = result
                .cached_tokens
                .max(delta.cached_token_count.unwrap_or(0));
            result.completion_tokens = result
                .completion_tokens
                .max(delta.generation_len.unwrap_or(0));
            let delta = ClientDelta::from(delta);
            if let Some(content) = delta.content {
                if !content.is_empty() && result.first_token_ms.is_none() {
                    result.first_token_ms = Some(began.elapsed().as_secs_f64() * 1000.0);
                }
                result.caption.push_str(&content);
            }
            if delta.is_final {
                result.caption = result.caption.trim().into();
                result.elapsed_ms = began.elapsed().as_secs_f64() * 1000.0;
                return Ok(result);
            }
        }
        Err(Error::Other(
            "Image response ended before its final delta".into(),
        ))
    }
}
