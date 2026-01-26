//! Moondream-specific client for vision-language tasks.
//!
//! Provides specialized methods for image captioning, object detection,
//! pointing, and visual question answering.

use std::collections::HashMap;
use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};

use crate::client::response::ClientDelta;
use crate::client::{Client, ClientError, Result, SamplingParams};
use crate::model::registry::ModelRegistry;

/// Model ID for Moondream.
pub const MOONDREAM_MODEL_ID: &str = "moondream3";

/// A point coordinate (x, y) normalized to 0-1.
pub type Point = (f64, f64);

/// A bounding box (x_min, y_min, x_max, y_max) normalized to 0-1.
pub type BoundingBox = (f64, f64, f64, f64);

/// Spatial reference: either a point or a bounding box.
#[derive(Debug, Clone)]
pub enum SpatialRef {
    Point(Point),
    Box(BoundingBox),
}

impl From<(f64, f64)> for SpatialRef {
    fn from(p: (f64, f64)) -> Self {
        SpatialRef::Point(p)
    }
}

impl From<(f64, f64, f64, f64)> for SpatialRef {
    fn from(b: (f64, f64, f64, f64)) -> Self {
        SpatialRef::Box(b)
    }
}

/// Grounding information linking text to spatial coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingSpan {
    /// Start index in the reasoning text
    pub start_idx: usize,
    /// End index in the reasoning text
    pub end_idx: usize,
    /// Points associated with this span
    pub points: Vec<Point>,
}

/// Reasoning output with grounding information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningOutput {
    /// The reasoning text
    pub text: String,
    /// Grounding spans linking text to coordinates
    pub grounding: Vec<GroundingSpan>,
}

/// Result from a query operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// The answer text
    pub answer: String,
    /// Optional reasoning with grounding (if reasoning=true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningOutput>,
}

/// Result from a caption operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionResult {
    /// The generated caption
    pub caption: String,
}

/// A point coordinate (x, y) normalized to 0-1 with named fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCoord {
    /// X coordinate (0-1)
    pub x: f64,
    /// Y coordinate (0-1)
    pub y: f64,
}

/// Result from a point operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointResult {
    /// Points where the object was found
    pub points: Vec<PointCoord>,
}

/// A detected object with bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

/// Result from a detect operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectResult {
    /// Detected objects with bounding boxes
    pub objects: Vec<DetectedObject>,
}

/// Result from a gaze detection operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GazeResult {
    /// Gaze target point, or None if not detected
    pub gaze: Option<PointCoord>,
}

/// Moondream-specific client for vision-language tasks.
///
/// Wraps the base client with specialized methods for Moondream's capabilities.
pub struct MoondreamClient {
    /// Underlying client
    client: Client,
    /// Capability token IDs
    capability_token_ids: HashMap<String, i32>,
}

impl MoondreamClient {
    /// Create a new MoondreamClient.
    ///
    /// This will ensure the moondream3 model is loaded and ready.
    pub async fn new(client: Client, registry: Arc<ModelRegistry>) -> Result<Self> {
        let model_info = registry.ensure_loaded(MOONDREAM_MODEL_ID).await?;

        // Build capability token ID map with fallbacks
        let mut capability_token_ids: HashMap<String, i32> = HashMap::from([
            ("start_ground".to_string(), 7),
            ("placeholder".to_string(), 8),
            ("end_ground".to_string(), 9),
            ("coord".to_string(), 5),
            ("answer".to_string(), 3),
        ]);

        // Override with actual capabilities from model
        if let Some(ref caps) = model_info.capabilities {
            for (name, ids) in caps {
                if let Some(&first_id) = ids.first() {
                    capability_token_ids.insert(name.clone(), first_id);
                }
            }
        }

        Ok(Self {
            client,
            capability_token_ids,
        })
    }

    /// Decode a coordinate value from base64-encoded bytes.
    ///
    /// The payload is 4 bytes representing a little-endian f32.
    fn decode_coordinate(payload_b64: &str) -> Result<f64> {
        let raw_bytes = BASE64
            .decode(payload_b64)
            .map_err(|e| ClientError::Multimodal(format!("Failed to decode base64: {}", e)))?;

        if raw_bytes.len() != 4 {
            return Err(ClientError::Multimodal(format!(
                "Coordinate payload must be 4 bytes; received {} bytes",
                raw_bytes.len()
            )));
        }

        let value = f32::from_le_bytes([raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]]);
        Ok(value as f64)
    }

    /// Decode a size value from base64-encoded bytes.
    ///
    /// The payload is 8 bytes representing two little-endian f32 values (width, height).
    fn decode_size(payload_b64: &str) -> Result<(f64, f64)> {
        let raw_bytes = BASE64
            .decode(payload_b64)
            .map_err(|e| ClientError::Multimodal(format!("Failed to decode base64: {}", e)))?;

        if raw_bytes.len() != 8 {
            return Err(ClientError::Multimodal(format!(
                "Size payload must be 8 bytes; received {} bytes",
                raw_bytes.len()
            )));
        }

        let w = f32::from_le_bytes([raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]]);
        let h = f32::from_le_bytes([raw_bytes[4], raw_bytes[5], raw_bytes[6], raw_bytes[7]]);
        Ok((w as f64, h as f64))
    }

    /// Build messages with image and optional spatial refs.
    fn build_query_messages(
        &self,
        prompt: &str,
        image_data_url: Option<&str>,
        spatial_refs: &[SpatialRef],
    ) -> Vec<HashMap<String, serde_json::Value>> {
        let mut content: Vec<serde_json::Value> = Vec::new();

        // Add image if present
        if let Some(data_url) = image_data_url {
            content.push(serde_json::json!({
                "type": "input_image",
                "image_url": data_url
            }));
        }

        // Add spatial refs as capability inputs
        for spatial_ref in spatial_refs {
            match spatial_ref {
                SpatialRef::Point((x, y)) => {
                    content.push(serde_json::json!({
                        "type": "capability",
                        "name": "coord",
                        "data": [x, y]
                    }));
                }
                SpatialRef::Box((x_min, y_min, x_max, y_max)) => {
                    // Convert box to center + size
                    let x_c = (x_min + x_max) / 2.0;
                    let y_c = (y_min + y_max) / 2.0;
                    let w = x_max - x_min;
                    let h = y_max - y_min;

                    content.push(serde_json::json!({
                        "type": "capability",
                        "name": "coord",
                        "data": [x_c, y_c]
                    }));
                    content.push(serde_json::json!({
                        "type": "capability",
                        "name": "size",
                        "data": [w, h]
                    }));
                }
            }
        }

        // Add text prompt
        content.push(serde_json::json!({
            "type": "input_text",
            "text": prompt
        }));

        vec![HashMap::from([
            ("role".to_string(), serde_json::json!("user")),
            ("content".to_string(), serde_json::json!(content)),
        ])]
    }

    /// Query the model with an image and prompt.
    ///
    /// This is the main method for visual question answering with optional
    /// grounding/reasoning output.
    pub async fn query(
        &self,
        prompt: &str,
        image_data_url: Option<&str>,
        spatial_refs: &[SpatialRef],
        reasoning: bool,
        params: SamplingParams,
    ) -> Result<QueryResult> {
        let messages = self.build_query_messages(prompt, image_data_url, spatial_refs);

        // Get streaming response
        let result = self
            .client
            .achat(MOONDREAM_MODEL_ID, messages, params, true)
            .await?;

        let mut rx = match result {
            crate::client::ChatResult::Stream(rx) => rx,
            _ => return Err(ClientError::RequestFailed("Expected stream".into())),
        };

        // Process stream
        let mut answer_parts: Vec<String> = Vec::new();
        let mut grounding: Vec<GroundingSpan> = Vec::new();
        let mut reasoning_parts: Vec<String> = Vec::new();
        let mut current_text_parts: Vec<String> = Vec::new();
        let mut current_coords: Vec<f64> = Vec::new();
        let mut in_ground_block = false;
        let mut in_answer_block = false;
        let mut ground_start_idx: Option<usize> = None;
        let mut reasoning_text_len: usize = 0;

        let start_ground_id = *self.capability_token_ids.get("start_ground").unwrap_or(&7);
        let end_ground_id = *self.capability_token_ids.get("end_ground").unwrap_or(&9);
        let answer_id = *self.capability_token_ids.get("answer").unwrap_or(&3);
        let placeholder_id = *self.capability_token_ids.get("placeholder").unwrap_or(&8);

        while let Some(delta) = rx.recv().await {
            let client_delta = ClientDelta::from(delta);
            let mut append_content = true;

            // Check for coordinate modal decoder output
            if let (Some(ref decoder_id), Some(ref bytes_b64)) = (
                &client_delta.modal_decoder_id,
                &client_delta.modal_bytes_b64,
            ) {
                if decoder_id.ends_with(".coord") && in_ground_block {
                    if let Ok(coord_value) = Self::decode_coordinate(bytes_b64) {
                        current_coords.push(coord_value);
                        append_content = false;
                    }
                }
            }

            // Process tokens
            for &token_id in &client_delta.tokens {
                if token_id == start_ground_id {
                    if in_ground_block {
                        // Finalize previous grounding
                        Self::finalize_grounding(
                            &mut grounding,
                            &mut current_text_parts,
                            &mut current_coords,
                            &mut ground_start_idx,
                        );
                    }
                    in_ground_block = true;
                    ground_start_idx = Some(reasoning_text_len);
                    current_text_parts.clear();
                    append_content = false;
                } else if token_id == end_ground_id {
                    if in_ground_block {
                        Self::finalize_grounding(
                            &mut grounding,
                            &mut current_text_parts,
                            &mut current_coords,
                            &mut ground_start_idx,
                        );
                    }
                    in_ground_block = false;
                    append_content = false;
                } else if token_id == answer_id {
                    in_answer_block = true;
                    append_content = false;
                } else if token_id == placeholder_id {
                    append_content = false;
                }
            }

            // Append content to appropriate buffer
            if let Some(ref content) = client_delta.content {
                if append_content {
                    if in_answer_block {
                        answer_parts.push(content.clone());
                    } else if !content.is_empty() {
                        current_text_parts.push(content.clone());
                        reasoning_parts.push(content.clone());
                        reasoning_text_len += content.len();
                    }
                }
            }

            if client_delta.is_final {
                break;
            }
        }

        // Finalize any remaining grounding
        if !current_text_parts.is_empty() {
            Self::finalize_grounding(
                &mut grounding,
                &mut current_text_parts,
                &mut current_coords,
                &mut ground_start_idx,
            );
        }

        // Build result
        let final_answer = answer_parts.join("").trim().to_string();
        let reasoning_text = reasoning_parts.join("").trim().to_string();

        let answer = if final_answer.is_empty() && !reasoning_text.is_empty() {
            reasoning_text.clone()
        } else {
            final_answer
        };

        let reasoning_output = if reasoning && !grounding.is_empty() {
            Some(ReasoningOutput {
                text: reasoning_text,
                grounding,
            })
        } else {
            None
        };

        Ok(QueryResult {
            answer,
            reasoning: reasoning_output,
        })
    }

    /// Finalize a grounding span and add it to the list.
    fn finalize_grounding(
        grounding: &mut Vec<GroundingSpan>,
        current_text_parts: &mut Vec<String>,
        current_coords: &mut Vec<f64>,
        ground_start_idx: &mut Option<usize>,
    ) {
        if current_coords.len() < 2 {
            current_text_parts.clear();
            current_coords.clear();
            *ground_start_idx = None;
            return;
        }

        let text_block = current_text_parts.join("");
        let start_idx = ground_start_idx.unwrap_or(0);

        // Pair up coordinates
        let points: Vec<Point> = current_coords
            .chunks(2)
            .filter(|chunk| chunk.len() == 2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect();

        grounding.push(GroundingSpan {
            start_idx,
            end_idx: start_idx + text_block.len(),
            points,
        });

        current_text_parts.clear();
        current_coords.clear();
        *ground_start_idx = None;
    }

    /// Generate a caption for an image.
    ///
    /// # Arguments
    /// * `image_data_url` - Data URL of the image (e.g., "data:image/jpeg;base64,...")
    /// * `length` - Caption length: "normal", "short", or "long"
    /// * `params` - Sampling parameters
    pub async fn caption(
        &self,
        image_data_url: &str,
        length: &str,
        mut params: SamplingParams,
    ) -> Result<CaptionResult> {
        let content = vec![serde_json::json!({
            "type": "input_image",
            "image_url": image_data_url
        })];

        let messages = vec![HashMap::from([
            ("role".to_string(), serde_json::json!("user")),
            ("content".to_string(), serde_json::json!(content)),
        ])];

        // Set task_name based on length
        params.task_name = Some(format!("caption_{}", length));

        let result = self
            .client
            .achat(MOONDREAM_MODEL_ID, messages, params, true)
            .await?;

        let mut rx = match result {
            crate::client::ChatResult::Stream(rx) => rx,
            _ => return Err(ClientError::RequestFailed("Expected stream".into())),
        };

        let mut caption_parts: Vec<String> = Vec::new();

        while let Some(delta) = rx.recv().await {
            let client_delta = ClientDelta::from(delta);

            if let Some(content) = client_delta.content {
                caption_parts.push(content);
            }

            if client_delta.is_final {
                break;
            }
        }

        Ok(CaptionResult {
            caption: caption_parts.join("").trim().to_string(),
        })
    }

    /// Find points where an object appears in an image.
    ///
    /// # Arguments
    /// * `image_data_url` - Data URL of the image
    /// * `object` - Object to find (e.g., "dog", "face")
    /// * `params` - Sampling parameters
    pub async fn point(
        &self,
        image_data_url: &str,
        object: &str,
        mut params: SamplingParams,
    ) -> Result<PointResult> {
        let content = vec![
            serde_json::json!({
                "type": "input_image",
                "image_url": image_data_url
            }),
            serde_json::json!({
                "type": "input_text",
                "text": object
            }),
        ];

        let messages = vec![HashMap::from([
            ("role".to_string(), serde_json::json!("user")),
            ("content".to_string(), serde_json::json!(content)),
        ])];

        params.task_name = Some("point".to_string());

        let result = self
            .client
            .achat(MOONDREAM_MODEL_ID, messages, params, true)
            .await?;

        let mut rx = match result {
            crate::client::ChatResult::Stream(rx) => rx,
            _ => return Err(ClientError::RequestFailed("Expected stream".into())),
        };

        let mut coords: Vec<f64> = Vec::new();

        while let Some(delta) = rx.recv().await {
            let client_delta = ClientDelta::from(delta);

            // Check for coordinate modal decoder output
            if let (Some(ref decoder_id), Some(ref bytes_b64)) = (
                &client_delta.modal_decoder_id,
                &client_delta.modal_bytes_b64,
            ) {
                if decoder_id.ends_with(".coord") {
                    if let Ok(coord_value) = Self::decode_coordinate(bytes_b64) {
                        coords.push(coord_value);
                    }
                }
            }

            if client_delta.is_final {
                break;
            }
        }

        // Pair up coordinates into points
        let points: Vec<PointCoord> = coords
            .chunks(2)
            .filter(|chunk| chunk.len() == 2)
            .map(|chunk| PointCoord {
                x: chunk[0],
                y: chunk[1],
            })
            .collect();

        Ok(PointResult { points })
    }

    /// Detect objects in an image with bounding boxes.
    ///
    /// # Arguments
    /// * `image_data_url` - Data URL of the image
    /// * `object` - Object to detect (e.g., "dog", "car")
    /// * `params` - Sampling parameters
    pub async fn detect(
        &self,
        image_data_url: &str,
        object: &str,
        mut params: SamplingParams,
    ) -> Result<DetectResult> {
        let content = vec![
            serde_json::json!({
                "type": "input_image",
                "image_url": image_data_url
            }),
            serde_json::json!({
                "type": "input_text",
                "text": object
            }),
        ];

        let messages = vec![HashMap::from([
            ("role".to_string(), serde_json::json!("user")),
            ("content".to_string(), serde_json::json!(content)),
        ])];

        params.task_name = Some("detect".to_string());

        let result = self
            .client
            .achat(MOONDREAM_MODEL_ID, messages, params, true)
            .await?;

        let mut rx = match result {
            crate::client::ChatResult::Stream(rx) => rx,
            _ => return Err(ClientError::RequestFailed("Expected stream".into())),
        };

        let mut coords: Vec<f64> = Vec::new();
        let mut sizes: Vec<(f64, f64)> = Vec::new();

        while let Some(delta) = rx.recv().await {
            let client_delta = ClientDelta::from(delta);

            if let (Some(ref decoder_id), Some(ref bytes_b64)) = (
                &client_delta.modal_decoder_id,
                &client_delta.modal_bytes_b64,
            ) {
                if decoder_id.ends_with(".coord") {
                    if let Ok(coord_value) = Self::decode_coordinate(bytes_b64) {
                        coords.push(coord_value);
                    }
                } else if decoder_id.ends_with(".size") {
                    if let Ok((w, h)) = Self::decode_size(bytes_b64) {
                        sizes.push((w, h));
                    }
                }
            }

            if client_delta.is_final {
                break;
            }
        }

        // Build bounding boxes from center coords + sizes
        let num_objects = (coords.len() / 2).min(sizes.len());
        let objects: Vec<DetectedObject> = (0..num_objects)
            .map(|i| {
                let x_c = coords[i * 2];
                let y_c = coords[i * 2 + 1];
                let (w, h) = sizes[i];

                DetectedObject {
                    x_min: x_c - w / 2.0,
                    y_min: y_c - h / 2.0,
                    x_max: x_c + w / 2.0,
                    y_max: y_c + h / 2.0,
                }
            })
            .collect();

        Ok(DetectResult { objects })
    }

    /// Detect where a person is looking in an image.
    ///
    /// # Arguments
    /// * `image_data_url` - Data URL of the image
    /// * `eye` - (x, y) coordinates of the eye/face position (normalized 0-1)
    /// * `params` - Sampling parameters
    pub async fn detect_gaze(
        &self,
        image_data_url: &str,
        eye: Point,
        mut params: SamplingParams,
    ) -> Result<GazeResult> {
        let content = vec![
            serde_json::json!({
                "type": "input_image",
                "image_url": image_data_url
            }),
            serde_json::json!({
                "type": "capability",
                "name": "coord",
                "data": [eye.0, eye.1]
            }),
        ];

        let messages = vec![HashMap::from([
            ("role".to_string(), serde_json::json!("user")),
            ("content".to_string(), serde_json::json!(content)),
        ])];

        params.task_name = Some("detect_gaze".to_string());

        let result = self
            .client
            .achat(MOONDREAM_MODEL_ID, messages, params, true)
            .await?;

        let mut rx = match result {
            crate::client::ChatResult::Stream(rx) => rx,
            _ => return Err(ClientError::RequestFailed("Expected stream".into())),
        };

        let mut coords: Vec<f64> = Vec::new();

        while let Some(delta) = rx.recv().await {
            let client_delta = ClientDelta::from(delta);

            if let (Some(ref decoder_id), Some(ref bytes_b64)) = (
                &client_delta.modal_decoder_id,
                &client_delta.modal_bytes_b64,
            ) {
                if decoder_id.ends_with(".coord") {
                    if let Ok(coord_value) = Self::decode_coordinate(bytes_b64) {
                        coords.push(coord_value);
                    }
                }
            }

            if client_delta.is_final {
                break;
            }
        }

        if coords.len() >= 2 {
            Ok(GazeResult {
                gaze: Some(PointCoord {
                    x: coords[0],
                    y: coords[1],
                }),
            })
        } else {
            Ok(GazeResult { gaze: None })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_coordinate() {
        // f32 value 0.5 in little-endian
        let half = 0.5f32;
        let bytes = half.to_le_bytes();
        let encoded = BASE64.encode(bytes);

        let decoded = MoondreamClient::decode_coordinate(&encoded).unwrap();
        assert!((decoded - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_decode_coordinate_invalid_length() {
        let encoded = BASE64.encode([0u8, 1, 2]); // Only 3 bytes
        let result = MoondreamClient::decode_coordinate(&encoded);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_size() {
        let w = 0.25f32;
        let h = 0.75f32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&w.to_le_bytes());
        bytes.extend_from_slice(&h.to_le_bytes());
        let encoded = BASE64.encode(&bytes);

        let (dw, dh) = MoondreamClient::decode_size(&encoded).unwrap();
        assert!((dw - 0.25).abs() < 1e-6);
        assert!((dh - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_spatial_ref_from_point() {
        let spatial: SpatialRef = (0.5, 0.5).into();
        match spatial {
            SpatialRef::Point((x, y)) => {
                assert!((x - 0.5).abs() < 1e-6);
                assert!((y - 0.5).abs() < 1e-6);
            }
            _ => panic!("Expected Point"),
        }
    }

    #[test]
    fn test_spatial_ref_from_box() {
        let spatial: SpatialRef = (0.1, 0.2, 0.3, 0.4).into();
        match spatial {
            SpatialRef::Box((x_min, y_min, x_max, y_max)) => {
                assert!((x_min - 0.1).abs() < 1e-6);
                assert!((y_min - 0.2).abs() < 1e-6);
                assert!((x_max - 0.3).abs() < 1e-6);
                assert!((y_max - 0.4).abs() < 1e-6);
            }
            _ => panic!("Expected Box"),
        }
    }

    #[test]
    fn test_finalize_grounding() {
        let mut grounding = Vec::new();
        let mut text_parts = vec!["hello".to_string(), " world".to_string()];
        let mut coords = vec![0.1, 0.2, 0.3, 0.4];
        let mut start_idx = Some(0usize);

        MoondreamClient::finalize_grounding(
            &mut grounding,
            &mut text_parts,
            &mut coords,
            &mut start_idx,
        );

        assert_eq!(grounding.len(), 1);
        assert_eq!(grounding[0].start_idx, 0);
        assert_eq!(grounding[0].end_idx, 11); // "hello world".len()
        assert_eq!(grounding[0].points.len(), 2);
        assert!(text_parts.is_empty());
        assert!(coords.is_empty());
        assert!(start_idx.is_none());
    }

    #[test]
    fn test_finalize_grounding_insufficient_coords() {
        let mut grounding = Vec::new();
        let mut text_parts = vec!["hello".to_string()];
        let mut coords = vec![0.1]; // Only one coord
        let mut start_idx = Some(0usize);

        MoondreamClient::finalize_grounding(
            &mut grounding,
            &mut text_parts,
            &mut coords,
            &mut start_idx,
        );

        // Should not add grounding with insufficient coords
        assert!(grounding.is_empty());
        assert!(text_parts.is_empty());
        assert!(coords.is_empty());
    }
}
