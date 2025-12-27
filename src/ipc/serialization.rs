//! Binary serialization for PIE IPC protocol.
//!
//! Wire format: [4 bytes: metadata length][JSON metadata][16-byte aligned binary blobs]

use crate::error::{Error, Result};
use serde_json::{json, Value};

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

/// Build a request payload for PIE.
///
/// # Arguments
/// * `request_id` - Unique request identifier
/// * `model_id` - Model identifier string
/// * `model_path` - Path to model weights
/// * `request_type` - Type of request (generation, embedding, etc.)
/// * `response_channel_id` - Channel ID for routing responses
/// * `prompt` - The prompt text
/// * `options` - Sampling and generation options
#[allow(clippy::too_many_arguments)]
pub fn build_request_payload(
    request_id: u64,
    model_id: &str,
    model_path: &str,
    request_type: RequestType,
    response_channel_id: u64,
    prompt: &str,
    max_tokens: i32,
    temperature: f64,
    top_p: f64,
    stop_sequences: &[String],
) -> Result<Vec<u8>> {
    let prompt_bytes = prompt.as_bytes();

    // Calculate blob layout
    let text_offset = 0usize;
    let text_size = prompt_bytes.len();
    let total_size = align(text_size);

    // Build layout for text segment
    let layout_data = encode_layout(&[(SegmentType::Text, text_size)]);

    // Reserve blob space
    let layout_offset = align(total_size);
    let final_size = layout_offset + layout_data.len();

    // Build prompt metadata
    let prompt_metadata = json!({
        "prompt_index": 0,
        "num_candidates": 1,
        "best_of": 1,
        "final_candidates": 1,
        "max_generated_tokens": max_tokens,
        "text_offset": text_offset,
        "text_size": text_size,
        "image_data_offset": 0,
        "image_data_size": 0,
        "image_sizes_offset": 0,
        "image_count": 0,
        "capability_data_offset": 0,
        "capability_data_size": 0,
        "capabilities": [],
        "layout_offset": layout_offset,
        "layout_count": 1,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        "min_p": 0.0,
        "rng_seed": 0,
        "top_logprobs": 0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_context_size": 0,
        "repetition_penalty": 1.0,
        "stop_sequences": stop_sequences,
        "tool_schemas_json": "",
        "response_format_json": "",
        "logit_bias": [],
    });

    // Build full metadata
    let metadata = json!({
        "request_id": request_id,
        "model_id": model_id,
        "model_path": model_path,
        "request_type": request_type as i32,
        "request_channel_id": 0,
        "response_channel_id": response_channel_id,
        "prompts": [prompt_metadata],
    });

    let metadata_bytes = serde_json::to_vec(&metadata)?;

    if metadata_bytes.len() > u32::MAX as usize {
        return Err(Error::Serialization(
            "Metadata exceeds 4-byte length prefix capacity".to_string(),
        ));
    }

    // Build payload buffer
    let mut payload = vec![0u8; final_size];

    // Copy text
    payload[text_offset..text_offset + text_size].copy_from_slice(prompt_bytes);

    // Copy layout
    payload[layout_offset..layout_offset + layout_data.len()].copy_from_slice(&layout_data);

    // Build frame: [4 bytes length][metadata][payload]
    let mut frame = Vec::with_capacity(4 + metadata_bytes.len() + payload.len());

    // Write length as little-endian u32
    let length = metadata_bytes.len() as u32;
    frame.extend_from_slice(&length.to_le_bytes());
    frame.extend_from_slice(&metadata_bytes);
    frame.extend_from_slice(&payload);

    Ok(frame)
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
    fn test_build_request_payload() {
        let payload = build_request_payload(
            1,
            "test-model",
            "/path/to/model",
            RequestType::Generation,
            12345,
            "Hello, world!",
            100,
            0.7,
            0.9,
            &[],
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
    }
}
