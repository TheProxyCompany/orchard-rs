//! IPC endpoint definitions for PIE communication.
//!
//! These endpoints mirror the Python/Swift implementations.
//! PIE uses NNG (nanomsg-next-gen) for high-performance IPC.

use std::path::PathBuf;

/// Get the IPC root directory for socket files.
///
/// Determines the stable, user-specific root directory for IPC socket files.
/// This ensures that all Orchard processes communicate through a predictable,
/// private location, avoiding pollution of system-wide directories like /tmp.
pub fn ipc_root() -> PathBuf {
    // ORCHARD_IPC_ROOT is an escape hatch for development or containerized environments.
    if let Ok(root) = std::env::var("ORCHARD_IPC_ROOT") {
        return PathBuf::from(root);
    }

    // Default to the standard application cache directory.
    let home = dirs::home_dir().expect("Could not determine home directory");

    // macOS: ~/Library/Caches, others: ~/.cache
    let base = if cfg!(target_os = "macos") {
        let mac_cache = home.join("Library/Caches");
        if mac_cache.exists() {
            mac_cache
        } else {
            home.join(".cache")
        }
    } else {
        home.join(".cache")
    };

    let path = base.join("com.theproxycompany/ipc");

    // Ensure directory exists
    std::fs::create_dir_all(&path).ok();

    path
}

/// Format a filesystem path into an NNG ipc:// transport URL.
fn as_ipc_url(path: PathBuf) -> String {
    format!("ipc://{}", path.display())
}

/// The endpoint for submitting inference requests to the engine.
/// Pattern: PUSH/PULL (Many clients PUSH, one engine PULLs)
pub fn request_url() -> String {
    as_ipc_url(ipc_root().join("pie_requests.ipc"))
}

/// The endpoint for receiving responses and broadcast events from the engine.
/// Pattern: PUB/SUB (One engine PUBlishes, many clients SUBscribe)
/// Topics are used to route messages to the correct consumer.
pub fn response_url() -> String {
    as_ipc_url(ipc_root().join("pie_responses.ipc"))
}

/// The endpoint for synchronous management commands (e.g., load_model).
/// Pattern: REQ/REP (One client sends a REQ, one engine sends a REP)
pub fn management_url() -> String {
    as_ipc_url(ipc_root().join("pie_management.ipc"))
}

// --- Topic Prefixes for the PUB/SUB Channel ---

/// Topic prefix for response deltas targeted at a specific client.
/// A client subscribes to RESPONSE_TOPIC_PREFIX + its_channel_id_hex.
pub const RESPONSE_TOPIC_PREFIX: &[u8] = b"resp:";

/// Topic prefix for global, broadcast events (e.g., engine_ready).
/// Clients subscribe to this prefix to receive all system-wide notifications.
pub const EVENT_TOPIC_PREFIX: &[u8] = b"__PIE_EVENT__:";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_root_is_valid() {
        let root = ipc_root();
        assert!(root.to_string_lossy().contains("com.theproxycompany/ipc"));
    }

    #[test]
    fn test_urls_are_valid() {
        assert!(request_url().starts_with("ipc://"));
        assert!(response_url().starts_with("ipc://"));
        assert!(management_url().starts_with("ipc://"));
    }
}
