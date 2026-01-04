//! Multimodal content handling.

use std::collections::HashMap;

use base64::Engine;
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::ChatFormatter;
use crate::error::{Error, Result};

/// A capability input with name and binary payload.
#[derive(Debug, Clone)]
pub struct CapabilityInput {
    pub name: String,
    pub payload: Vec<u8>,
}

/// Content type for layout segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    Text,
    Image,
    Capability,
}

/// A segment in the multimodal layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutSegment {
    #[serde(rename = "type")]
    pub segment_type: String,
    pub length: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Build multimodal messages for template rendering.
///
/// Returns a tuple of:
/// - Messages for the template
/// - Image buffers (raw bytes)
/// - Capability inputs
/// - Content order (type, index) for layout building
pub fn build_multimodal_messages(
    formatter: &ChatFormatter,
    items: &[HashMap<String, serde_json::Value>],
    instructions: Option<&str>,
) -> Result<(
    Vec<HashMap<String, serde_json::Value>>,
    Vec<Vec<u8>>,
    Vec<CapabilityInput>,
    Vec<(ContentType, usize)>,
)> {
    let available_roles: std::collections::HashSet<String> = [
        formatter.control_tokens.roles.system.as_ref().map(|_| "system"),
        formatter.control_tokens.roles.agent.as_ref().map(|_| "agent"),
        formatter.control_tokens.roles.user.as_ref().map(|_| "user"),
        formatter.control_tokens.roles.tool.as_ref().map(|_| "tool"),
    ]
    .into_iter()
    .flatten()
    .map(String::from)
    .collect();

    let mut messages = Vec::new();
    let mut image_buffers = Vec::new();
    let mut capabilities = Vec::new();
    let mut content_order = Vec::new();

    if let Some(instructions) = instructions {
        let system_role = if available_roles.contains("system") {
            "system"
        } else {
            "user"
        };
        let mut msg = HashMap::new();
        msg.insert("role".to_string(), serde_json::json!(system_role));
        msg.insert("content".to_string(), serde_json::json!(instructions));
        messages.push(msg);
    }

    for (msg_idx, message) in items.iter().enumerate() {
        let role = message
            .get("role")
            .and_then(|v| v.as_str())
            .map(|r| normalize_role(r, &available_roles))
            .unwrap_or_else(|| "user".to_string());

        let content = message.get("content").cloned().unwrap_or(serde_json::Value::Null);

        if let Some(text) = content.as_str() {
            let mut msg = HashMap::new();
            msg.insert("role".to_string(), serde_json::json!(role));
            msg.insert("content".to_string(), serde_json::json!(text));
            messages.push(msg);
            continue;
        }

        if let Some(parts) = content.as_array() {
            let mut rendered_parts: Vec<serde_json::Value> = Vec::new();

            for (part_idx, content_part) in parts.iter().enumerate() {
                let part_type = content_part
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or(Error::MissingContentType(part_idx, msg_idx))?
                    .to_lowercase();

                match part_type.as_str() {
                    "text" | "input_text" => {
                        let text = content_part
                            .get("text")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                Error::Other(format!(
                                    "Text content missing for part {} in message {}",
                                    part_idx, msg_idx
                                ))
                            })?;
                        rendered_parts.push(serde_json::json!({"type": "text", "text": text}));
                    }
                    "image" | "image_url" | "input_image" => {
                        let image_url = get_image_url(content_part)?;
                        let decoded = decode_image_payload(&image_url)?;

                        content_order.push((ContentType::Image, image_buffers.len()));
                        image_buffers.push(decoded);
                        rendered_parts.push(serde_json::json!({"type": "image"}));
                    }
                    "capability" => {
                        let name = content_part
                            .get("name")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                Error::Other(format!(
                                    "Capability part {} in message {} missing 'name'",
                                    part_idx, msg_idx
                                ))
                            })?;

                        let data = content_part
                            .get("data")
                            .and_then(|v| v.as_array())
                            .ok_or_else(|| {
                                Error::Other(format!(
                                    "Capability part {} in message {} missing 'data' array",
                                    part_idx, msg_idx
                                ))
                            })?;

                        let mut payload = Vec::with_capacity(data.len() * 4);
                        for val in data {
                            let f = val.as_f64().unwrap_or(0.0) as f32;
                            payload.extend_from_slice(&f.to_le_bytes());
                        }

                        content_order.push((ContentType::Capability, capabilities.len()));
                        capabilities.push(CapabilityInput {
                            name: name.to_string(),
                            payload,
                        });
                        rendered_parts.push(serde_json::json!({"type": "capability"}));
                    }
                    _ => {
                        return Err(Error::Other(format!(
                            "Unsupported content type: {}",
                            part_type
                        )));
                    }
                }
            }

            let mut msg = HashMap::new();
            msg.insert("role".to_string(), serde_json::json!(role));
            msg.insert("content".to_string(), serde_json::json!(rendered_parts));
            messages.push(msg);
        } else if !content.is_null() {
            return Err(Error::InvalidContent);
        }
    }

    Ok((messages, image_buffers, capabilities, content_order))
}

/// Build the multimodal layout for PIE.
///
/// Creates layout segments that describe the binary data structure.
pub fn build_multimodal_layout(
    prompt_text: &str,
    image_buffers: &[Vec<u8>],
    capabilities: &[CapabilityInput],
    content_order: &[(ContentType, usize)],
    placeholder_token: &str,
    exclude_image_placeholder: bool,
    coord_placeholder: Option<&str>,
) -> Result<Vec<LayoutSegment>> {
    let mut layout = Vec::new();

    if image_buffers.is_empty() && capabilities.is_empty() {
        let text_bytes = prompt_text.as_bytes();
        if text_bytes.is_empty() {
            return Err(Error::EmptyRequest);
        }
        layout.push(LayoutSegment {
            segment_type: "text".to_string(),
            length: text_bytes.len(),
            name: None,
        });
        return Ok(layout);
    }

    let image_regex = Regex::new(&regex::escape(placeholder_token))
        .expect("escaped regex is always valid");
    let image_matches: Vec<_> = image_regex.find_iter(prompt_text).collect();

    if image_matches.len() != image_buffers.len() {
        return Err(Error::PlaceholderMismatch(
            image_matches.len(),
            image_buffers.len(),
        ));
    }

    let coord_placeholder = coord_placeholder.unwrap_or("<|coord|>");
    let coord_regex = Regex::new(&regex::escape(coord_placeholder))
        .expect("escaped regex is always valid");
    let coord_matches: Vec<_> = coord_regex.find_iter(prompt_text).collect();
    let use_coord_placeholders = !coord_matches.is_empty();

    if use_coord_placeholders {
        let coord_caps: Vec<_> = capabilities.iter().filter(|c| c.name == "coord").collect();

        if coord_matches.len() != coord_caps.len() {
            return Err(Error::Other(format!(
                "Mismatch between coord placeholders ({}) and coord capabilities ({})",
                coord_matches.len(),
                coord_caps.len()
            )));
        }

        let mut all_placeholders: Vec<(usize, usize, &str, usize)> = Vec::new();

        for (idx, m) in image_matches.iter().enumerate() {
            all_placeholders.push((m.start(), m.end(), "image", idx));
        }
        for (idx, m) in coord_matches.iter().enumerate() {
            all_placeholders.push((m.start(), m.end(), "coord", idx));
        }

        all_placeholders.sort_by_key(|p| p.0);

        let mut cursor = 0;
        let mut coord_cap_idx = 0;

        for (start, end, ptype, idx) in all_placeholders {
            let text_end = if ptype == "image" && !exclude_image_placeholder {
                end
            } else {
                start
            };

            let text_segment = &prompt_text[cursor..text_end];
            let segment_bytes = text_segment.as_bytes();
            if !segment_bytes.is_empty() {
                layout.push(LayoutSegment {
                    segment_type: "text".to_string(),
                    length: segment_bytes.len(),
                    name: None,
                });
            }

            if ptype == "image" {
                layout.push(LayoutSegment {
                    segment_type: "image".to_string(),
                    length: image_buffers[idx].len(),
                    name: None,
                });
            } else {
                let cap = &coord_caps[coord_cap_idx];
                layout.push(LayoutSegment {
                    segment_type: "capability".to_string(),
                    length: cap.payload.len(),
                    name: Some(cap.name.clone()),
                });
                coord_cap_idx += 1;
            }

            cursor = end;
        }

        if cursor < prompt_text.len() {
            let tail = &prompt_text[cursor..];
            let tail_bytes = tail.as_bytes();
            if !tail_bytes.is_empty() {
                layout.push(LayoutSegment {
                    segment_type: "text".to_string(),
                    length: tail_bytes.len(),
                    name: None,
                });
            }
        }
    } else {
        let mut cursor = 0;
        let mut image_idx = 0;
        let mut cap_idx = 0;

        for (content_type, _) in content_order {
            match content_type {
                ContentType::Image => {
                    let m = &image_matches[image_idx];
                    let text_end = if exclude_image_placeholder {
                        m.start()
                    } else {
                        m.end()
                    };

                    let text_segment = &prompt_text[cursor..text_end];
                    let segment_bytes = text_segment.as_bytes();
                    if !segment_bytes.is_empty() {
                        layout.push(LayoutSegment {
                            segment_type: "text".to_string(),
                            length: segment_bytes.len(),
                            name: None,
                        });
                    }

                    layout.push(LayoutSegment {
                        segment_type: "image".to_string(),
                        length: image_buffers[image_idx].len(),
                        name: None,
                    });

                    cursor = m.end();
                    image_idx += 1;
                }
                ContentType::Capability => {
                    let cap = &capabilities[cap_idx];
                    layout.push(LayoutSegment {
                        segment_type: "capability".to_string(),
                        length: cap.payload.len(),
                        name: Some(cap.name.clone()),
                    });
                    cap_idx += 1;
                }
                ContentType::Text => {}
            }
        }

        if cursor < prompt_text.len() {
            let tail = &prompt_text[cursor..];
            let tail_bytes = tail.as_bytes();
            if !tail_bytes.is_empty() {
                layout.push(LayoutSegment {
                    segment_type: "text".to_string(),
                    length: tail_bytes.len(),
                    name: None,
                });
            }
        }
    }

    if layout.is_empty() {
        return Err(Error::EmptyRequest);
    }

    Ok(layout)
}

fn normalize_role(raw_role: &str, available_roles: &std::collections::HashSet<String>) -> String {
    let role_lower = raw_role.to_lowercase();

    let normalized = match role_lower.as_str() {
        "assistant" | "model" => "agent",
        "developer" => "system",
        other => other,
    };

    if !available_roles.contains(normalized) {
        log::debug!(
            "Role '{}' not found in formatter profile; using as-is",
            normalized
        );
    }

    normalized.to_string()
}

fn get_image_url(content_part: &serde_json::Value) -> Result<String> {
    if let Some(url) = content_part.get("image_url").and_then(|v| v.as_str()) {
        return Ok(url.to_string());
    }

    if let Some(obj) = content_part.get("image_url").and_then(|v| v.as_object()) {
        if let Some(url) = obj.get("url").and_then(|v| v.as_str()) {
            return Ok(url.to_string());
        }
        if let Some(data) = obj.get("data").and_then(|v| v.as_str()) {
            return Ok(data.to_string());
        }
    }

    Err(Error::Other("Image content part missing image_url".into()))
}

fn decode_image_payload(data_url: &str) -> Result<Vec<u8>> {
    let base64_prefix = ";base64,";
    if let Some(idx) = data_url.find(base64_prefix) {
        let base64_data = &data_url[idx + base64_prefix.len()..];
        base64::engine::general_purpose::STANDARD
            .decode(base64_data)
            .map_err(|_| Error::InvalidBase64)
    } else {
        Err(Error::InvalidImageUrl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_image_payload() {
        let data_url = "data:image/png;base64,SGVsbG8gV29ybGQ=";
        let decoded = decode_image_payload(data_url).unwrap();
        assert_eq!(decoded, b"Hello World");
    }

    #[test]
    fn test_decode_invalid_base64() {
        let data_url = "data:image/png;base64,!!!invalid!!!";
        let result = decode_image_payload(data_url);
        assert!(matches!(result, Err(Error::InvalidBase64)));
    }

    #[test]
    fn test_normalize_role() {
        let roles: std::collections::HashSet<String> =
            ["system", "agent", "user"].iter().map(|s| s.to_string()).collect();

        assert_eq!(normalize_role("assistant", &roles), "agent");
        assert_eq!(normalize_role("model", &roles), "agent");
        assert_eq!(normalize_role("developer", &roles), "system");
        assert_eq!(normalize_role("user", &roles), "user");
    }
}
