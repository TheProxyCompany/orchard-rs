//! Control tokens for different model templates.

use std::collections::HashMap;
use std::path::Path;

use minijinja::Value;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Role definition for a message type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name (e.g., "user", "assistant", "system")
    pub role_name: String,
    /// Tag that starts the role section
    pub role_start_tag: String,
    /// Tag that ends the role section
    pub role_end_tag: String,
    /// End of message token (optional, defaults to end_of_sequence)
    #[serde(default)]
    pub end_of_message: Option<String>,
}

impl Role {
    pub fn to_value(&self) -> Value {
        let mut map = std::collections::BTreeMap::new();
        map.insert("role_name".to_string(), Value::from(self.role_name.clone()));
        map.insert(
            "role_start_tag".to_string(),
            Value::from(self.role_start_tag.clone()),
        );
        map.insert(
            "role_end_tag".to_string(),
            Value::from(self.role_end_tag.clone()),
        );
        if let Some(ref eom) = self.end_of_message {
            map.insert("end_of_message".to_string(), Value::from(eom.clone()));
        }
        Value::from_object(map)
    }
}

/// Collection of role definitions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoleTags {
    /// System role
    #[serde(default)]
    pub system: Option<Role>,
    /// Agent/assistant role
    #[serde(default)]
    pub agent: Option<Role>,
    /// User role
    #[serde(default)]
    pub user: Option<Role>,
    /// Tool role
    #[serde(default)]
    pub tool: Option<Role>,
}

impl RoleTags {
    pub fn to_value(&self) -> Value {
        let mut map = std::collections::BTreeMap::new();

        if let Some(ref role) = self.system {
            map.insert("system".to_string(), role.to_value());
        }
        if let Some(ref role) = self.agent {
            map.insert("agent".to_string(), role.to_value());
        }
        if let Some(ref role) = self.user {
            map.insert("user".to_string(), role.to_value());
        }
        if let Some(ref role) = self.tool {
            map.insert("tool".to_string(), role.to_value());
        }

        Value::from_object(map)
    }

    pub fn get(&self, role_name: &str) -> Option<&Role> {
        match role_name.to_lowercase().as_str() {
            "system" => self.system.as_ref(),
            "agent" | "assistant" => self.agent.as_ref(),
            "user" => self.user.as_ref(),
            "tool" | "ipython" => self.tool.as_ref(),
            _ => None,
        }
    }
}

/// Control tokens for a model template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlTokens {
    /// Template type identifier
    pub template_type: String,
    /// Beginning of text token
    pub begin_of_text: String,
    /// End of message token
    pub end_of_message: String,
    /// End of sequence token
    pub end_of_sequence: String,
    /// Start image token (for multimodal models)
    #[serde(default)]
    pub start_image_token: Option<String>,
    /// End image token (for multimodal models)
    #[serde(default)]
    pub end_image_token: Option<String>,
    /// Start thinking token (for reasoning models)
    #[serde(default)]
    pub thinking_start_token: Option<String>,
    /// End thinking token (for reasoning models)
    #[serde(default)]
    pub thinking_end_token: Option<String>,
    /// Coordinate placeholder (for object detection)
    #[serde(default)]
    pub coord_placeholder: Option<String>,
    /// Named capabilities (token name -> token string)
    #[serde(default)]
    pub capabilities: HashMap<String, String>,
    /// Role tags
    pub roles: RoleTags,
}

impl ControlTokens {
    /// Load control tokens from a profile directory.
    pub fn load(profile_dir: &Path) -> Result<Self> {
        let tokens_path = profile_dir.join("control_tokens.json");
        if !tokens_path.exists() {
            return Err(Error::FormatterConfigNotFound(format!(
                "control_tokens.json not found in {:?}",
                profile_dir
            )));
        }

        let content = std::fs::read_to_string(&tokens_path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_tags_get() {
        let role_tags = RoleTags {
            system: Some(Role {
                role_name: "system".to_string(),
                role_start_tag: "<|system|>".to_string(),
                role_end_tag: "".to_string(),
                end_of_message: None,
            }),
            agent: Some(Role {
                role_name: "assistant".to_string(),
                role_start_tag: "<|assistant|>".to_string(),
                role_end_tag: "".to_string(),
                end_of_message: None,
            }),
            user: None,
            tool: None,
        };

        assert!(role_tags.get("system").is_some());
        assert!(role_tags.get("agent").is_some());
        assert!(role_tags.get("assistant").is_some()); // Alias
        assert!(role_tags.get("user").is_none());
    }

    #[test]
    fn test_control_tokens_deserialize() {
        let json = r#"{
            "template_type": "llama",
            "begin_of_text": "<|begin_of_text|>",
            "end_of_message": "<|eom_id|>",
            "end_of_sequence": "<|eot_id|>",
            "roles": {
                "agent": {
                    "role_name": "assistant",
                    "role_start_tag": "<|start_header_id|>",
                    "role_end_tag": "<|end_header_id|>\n\n"
                }
            }
        }"#;

        let tokens: ControlTokens = serde_json::from_str(json).unwrap();
        assert_eq!(tokens.template_type, "llama");
        assert_eq!(tokens.begin_of_text, "<|begin_of_text|>");
        assert!(tokens.roles.agent.is_some());
    }
}
