//! Binary serialization for PIE IPC protocol.
//!
//! Wire format: [4 bytes: metadata length][JSON metadata][16-byte aligned binary blobs]

use crate::defaults;
use crate::error::{Error, Result};
use serde::{ser::Serialize as SerializeTrait, Deserialize, Serialize};
use serde_json::{json, Value};

/// A single prompt payload for batched requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptPayload {
    pub prompt: String,
    #[serde(default)]
    pub image_buffers: Vec<Vec<u8>>,
    #[serde(default)]
    pub capabilities: Vec<CapabilityEntry>,
    #[serde(default)]
    pub layout: Vec<LayoutEntry>,
    #[serde(default)]
    pub max_generated_tokens: i32,
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
    pub stop_sequences: Vec<String>,
    #[serde(default = "defaults::num_candidates")]
    pub num_candidates: i32,
    #[serde(default)]
    pub best_of: Option<i32>,
    #[serde(default)]
    pub final_candidates: Option<i32>,
    #[serde(default)]
    pub frequency_penalty: f64,
    #[serde(default)]
    pub presence_penalty: f64,
    #[serde(default = "defaults::repetition_penalty")]
    pub repetition_penalty: f64,
    #[serde(default = "defaults::repetition_context_size")]
    pub repetition_context_size: i32,
    #[serde(default)]
    pub top_logprobs: i32,
    #[serde(default)]
    pub logit_bias: std::collections::HashMap<i32, f64>,
    #[serde(default)]
    pub tool_schemas_json: String,
    #[serde(default)]
    pub response_format_json: String,
    #[serde(default)]
    pub task_name: Option<String>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

/// Capability entry for multimodal content.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilityEntry {
    pub name: String,
    pub position: usize,
    pub payload: Vec<u8>,
}

/// Layout entry describing content ordering.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayoutEntry {
    #[serde(rename = "type")]
    pub segment_type: String,
    pub length: usize,
}

/// Payload alignment boundary (16 bytes)
const PAYLOAD_ALIGNMENT: usize = 16;

/// Layout segment size: 1 byte type + 7 padding + 8 bytes length
const LAYOUT_SEGMENT_SIZE: usize = 16;

/// Segment types matching C++ SerializedSegmentType
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SegmentType {
    Text = 0,
    Image = 1,
    Capability = 2,
}

/// Request type codes matching PIE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RequestType {
    Generation = 0,
    Embedding = 1,
    Query = 2,
    Point = 3,
    Detect = 4,
    Agent = 5,
    Omni = 6,
}

/// Align offset to payload alignment boundary
fn align(offset: usize) -> usize {
    let remainder = offset % PAYLOAD_ALIGNMENT;
    if remainder == 0 {
        offset
    } else {
        offset + (PAYLOAD_ALIGNMENT - remainder)
    }
}

/// Serialize JSON to compact format (no spaces after separators, matching Python).
fn serialize_json_compact(value: &Value) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    let formatter = serde_json::ser::CompactFormatter;
    let mut serializer = serde_json::Serializer::with_formatter(&mut buffer, formatter);
    SerializeTrait::serialize(value, &mut serializer)?;
    Ok(buffer)
}

/// Encode layout segments.
fn encode_layout(segments: &[(SegmentType, usize)]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(segments.len() * LAYOUT_SEGMENT_SIZE);

    for (segment_type, length) in segments {
        // 1 byte type
        buffer.push(*segment_type as u8);
        // 7 bytes padding
        buffer.extend_from_slice(&[0u8; 7]);
        // 8 bytes length (little-endian)
        buffer.extend_from_slice(&(*length as u64).to_le_bytes());
    }

    buffer
}

/// Parse a response delta from JSON.
pub fn parse_response_delta(data: &[u8]) -> Result<Value> {
    serde_json::from_slice(data).map_err(Error::from)
}

/// Build a batched request payload with multiple prompts.
///
/// This is the correct implementation that sends all prompts in ONE IPC message,
/// allowing the engine to schedule them together efficiently.
#[allow(clippy::too_many_arguments)]
pub fn build_batch_request_payload(
    request_id: u64,
    model_id: &str,
    model_path: &str,
    request_type: RequestType,
    response_channel_id: u64,
    prompts: &[PromptPayload],
) -> Result<Vec<u8>> {
    if prompts.is_empty() {
        return Err(Error::Serialization(
            "At least one prompt is required".to_string(),
        ));
    }

    // Track blob fragments: (offset, data)
    let mut blob_fragments: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut total_size = 0usize;

    // Reserve blob space with alignment
    let mut reserve_blob = |data: Vec<u8>| -> (usize, usize) {
        if data.is_empty() {
            return (0, 0);
        }
        total_size = align(total_size);
        let offset = total_size;
        let size = data.len();
        blob_fragments.push((offset, data));
        total_size += size;
        (offset, size)
    };

    // Build metadata for each prompt
    let mut prompt_metadata_list: Vec<Value> = Vec::with_capacity(prompts.len());

    for (index, prompt) in prompts.iter().enumerate() {
        let text_bytes = prompt.prompt.as_bytes().to_vec();

        // Encode image buffers
        let (image_span_bytes, image_count, image_data_bytes) =
            encode_image_buffers(&prompt.image_buffers);

        // Encode capabilities
        let (capability_metadata, capability_data_bytes) =
            encode_capabilities(&prompt.capabilities);

        // Build layout
        let layout_data = if prompt.layout.is_empty() {
            // Default layout: text followed by images
            let mut segments = vec![(SegmentType::Text, text_bytes.len())];
            for img in &prompt.image_buffers {
                segments.push((SegmentType::Image, img.len()));
            }
            encode_layout(&segments)
        } else {
            let segments: Vec<(SegmentType, usize)> = prompt
                .layout
                .iter()
                .map(|e| {
                    let seg_type = match e.segment_type.as_str() {
                        "image" => SegmentType::Image,
                        "capability" => SegmentType::Capability,
                        _ => SegmentType::Text,
                    };
                    (seg_type, e.length)
                })
                .collect();
            encode_layout(&segments)
        };
        let layout_count = if prompt.layout.is_empty() {
            1 + prompt.image_buffers.len()
        } else {
            prompt.layout.len()
        };

        // Validate layout if explicitly provided
        if !prompt.layout.is_empty() {
            let total_image_size: usize = prompt.image_buffers.iter().map(|b| b.len()).sum();
            validate_layout(text_bytes.len(), total_image_size, &prompt.layout, index)?;
        }

        // Reserve space for all blob data
        let (text_offset, text_size) = reserve_blob(text_bytes);
        let (image_sizes_offset, _) = reserve_blob(image_span_bytes);
        let (image_data_offset, image_data_size) = reserve_blob(image_data_bytes);
        let (capability_data_offset, capability_data_size) = reserve_blob(capability_data_bytes);
        let (layout_offset, _) = reserve_blob(layout_data);

        // Compute best_of and final_candidates with proper defaults
        let best_of = prompt.best_of.unwrap_or(prompt.num_candidates.max(1));
        let final_candidates = prompt.final_candidates.unwrap_or(best_of);

        // Convert logit_bias HashMap to array of {token, bias} objects (matching Python)
        let logit_bias: Vec<Value> = prompt
            .logit_bias
            .iter()
            .map(|(&k, &v)| json!({"token": k, "bias": v}))
            .collect();

        let prompt_meta = json!({
            "prompt_index": index,
            "num_candidates": prompt.num_candidates.max(1),
            "best_of": best_of,
            "final_candidates": final_candidates,
            "max_generated_tokens": prompt.max_generated_tokens,
            "text_offset": text_offset,
            "text_size": text_size,
            "image_data_offset": image_data_offset,
            "image_data_size": image_data_size,
            "image_sizes_offset": image_sizes_offset,
            "image_count": image_count,
            "capability_data_offset": capability_data_offset,
            "capability_data_size": capability_data_size,
            "capabilities": capability_metadata,
            "layout_offset": layout_offset,
            "layout_count": layout_count,
            "temperature": prompt.temperature,
            "top_p": prompt.top_p,
            "top_k": prompt.top_k,
            "min_p": prompt.min_p,
            "rng_seed": (prompt.rng_seed & 0xFFFFFFFF) as u32,
            "top_logprobs": prompt.top_logprobs,
            "frequency_penalty": prompt.frequency_penalty,
            "presence_penalty": prompt.presence_penalty,
            "repetition_context_size": prompt.repetition_context_size,
            "repetition_penalty": prompt.repetition_penalty,
            "stop_sequences": prompt.stop_sequences,
            "tool_schemas_json": prompt.tool_schemas_json,
            "response_format_json": prompt.response_format_json,
            "logit_bias": logit_bias,
            "task_name": prompt.task_name,
            "reasoning_effort": prompt.reasoning_effort,
        });

        prompt_metadata_list.push(prompt_meta);
    }

    // Build full metadata
    let metadata = json!({
        "request_id": request_id,
        "model_id": model_id,
        "model_path": model_path,
        "request_type": request_type as i32,
        "request_channel_id": 0,
        "response_channel_id": response_channel_id,
        "prompts": prompt_metadata_list,
    });

    // Use compact JSON serialization (no spaces, matching Python)
    let metadata_bytes = serialize_json_compact(&metadata)?;

    if metadata_bytes.len() > u32::MAX as usize {
        return Err(Error::Serialization(
            "Metadata exceeds 4-byte length prefix capacity".to_string(),
        ));
    }

    // Build payload buffer
    let mut payload = vec![0u8; total_size];
    for (offset, data) in blob_fragments {
        payload[offset..offset + data.len()].copy_from_slice(&data);
    }

    // Build frame: [4 bytes length][metadata][payload]
    let mut frame = Vec::with_capacity(4 + metadata_bytes.len() + payload.len());
    let length = metadata_bytes.len() as u32;
    frame.extend_from_slice(&length.to_le_bytes());
    frame.extend_from_slice(&metadata_bytes);
    frame.extend_from_slice(&payload);

    Ok(frame)
}

/// Validate that layout bytes match actual content sizes.
fn validate_layout(
    text_size: usize,
    image_data_size: usize,
    layout: &[LayoutEntry],
    prompt_index: usize,
) -> Result<()> {
    let mut layout_text_bytes = 0usize;
    let mut layout_image_bytes = 0usize;

    for entry in layout {
        match entry.segment_type.as_str() {
            "text" => layout_text_bytes += entry.length,
            "image" => layout_image_bytes += entry.length,
            _ => {} // capabilities are handled separately
        }
    }

    if layout_text_bytes != text_size {
        return Err(Error::Serialization(format!(
            "Prompt {}: Layout text bytes ({}) != actual text size ({})",
            prompt_index, layout_text_bytes, text_size
        )));
    }

    if layout_image_bytes != image_data_size {
        return Err(Error::Serialization(format!(
            "Prompt {}: Layout image bytes ({}) != actual image size ({})",
            prompt_index, layout_image_bytes, image_data_size
        )));
    }

    Ok(())
}

/// Encode image buffers into span array and concatenated data.
fn encode_image_buffers(buffers: &[Vec<u8>]) -> (Vec<u8>, usize, Vec<u8>) {
    if buffers.is_empty() {
        return (Vec::new(), 0, Vec::new());
    }

    // Span array: 8 bytes per image (length as u64 LE)
    let mut span_buffer = Vec::with_capacity(buffers.len() * 8);
    let total_data_size: usize = buffers.iter().map(|b| b.len()).sum();
    let mut data_buffer = Vec::with_capacity(total_data_size);

    for buffer in buffers {
        span_buffer.extend_from_slice(&(buffer.len() as u64).to_le_bytes());
        data_buffer.extend_from_slice(buffer);
    }

    (span_buffer, buffers.len(), data_buffer)
}

/// Encode capability entries.
fn encode_capabilities(capabilities: &[CapabilityEntry]) -> (Vec<Value>, Vec<u8>) {
    if capabilities.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut metadata_list = Vec::with_capacity(capabilities.len());
    let mut data_buffer = Vec::new();

    for cap in capabilities {
        metadata_list.push(json!({
            "name": cap.name,
            "position": cap.position,
            "payload_size": cap.payload.len(),
        }));
        data_buffer.extend_from_slice(&cap.payload);
    }

    (metadata_list, data_buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align() {
        assert_eq!(align(0), 0);
        assert_eq!(align(1), 16);
        assert_eq!(align(16), 16);
        assert_eq!(align(17), 32);
    }

    #[test]
    fn test_build_batch_request_payload() {
        let prompt = PromptPayload {
            prompt: "Hello, world!".to_string(),
            max_generated_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            ..Default::default()
        };

        let payload = build_batch_request_payload(
            1,
            "test-model",
            "/path/to/model",
            RequestType::Generation,
            12345,
            &[prompt],
        )
        .unwrap();

        // Should have 4-byte length prefix
        assert!(payload.len() > 4);

        // Read length prefix
        let length = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;

        // Metadata should be valid JSON
        let metadata: Value = serde_json::from_slice(&payload[4..4 + length]).unwrap();
        assert_eq!(metadata["request_id"], 1);
        assert_eq!(metadata["model_id"], "test-model");
        assert_eq!(metadata["prompts"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_validate_layout_success() {
        let layout = vec![
            LayoutEntry {
                segment_type: "text".to_string(),
                length: 100,
            },
            LayoutEntry {
                segment_type: "image".to_string(),
                length: 5000,
            },
        ];

        let result = validate_layout(100, 5000, &layout, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_layout_text_mismatch() {
        let layout = vec![LayoutEntry {
            segment_type: "text".to_string(),
            length: 50, // Mismatch!
        }];

        let result = validate_layout(100, 0, &layout, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("text bytes"));
    }

    #[test]
    fn test_validate_layout_image_mismatch() {
        let layout = vec![
            LayoutEntry {
                segment_type: "text".to_string(),
                length: 100,
            },
            LayoutEntry {
                segment_type: "image".to_string(),
                length: 1000, // Mismatch!
            },
        ];

        let result = validate_layout(100, 5000, &layout, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("image"));
    }

    #[test]
    fn test_compact_json_serialization() {
        let value = serde_json::json!({
            "key": "value",
            "number": 42
        });

        let compact = serialize_json_compact(&value).unwrap();
        let compact_str = String::from_utf8(compact).unwrap();

        // Should not have spaces after colons or commas
        assert!(!compact_str.contains(": "));
        assert!(!compact_str.contains(", "));
        assert!(compact_str.contains(":") && compact_str.contains(","));
    }
}
