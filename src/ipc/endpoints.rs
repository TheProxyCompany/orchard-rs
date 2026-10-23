//! IPC endpoint definitions for PIE communication.
//!
//! These endpoints mirror the engine's `pie::utils::get_ipc_dir()`
//! (`src/pie/src/utils/platform_utils.cpp`) byte for byte. PIE uses NNG
//! (nanomsg-next-gen) for high-performance IPC over unix sockets, and
//! `sockaddr_un` caps the whole socket path at 104 bytes on macOS, so the
//! engine projects any root that would overflow onto a compact, hashed
//! directory under `/tmp`. A client that does not apply the same projection
//! dials a socket the engine never binds and times out waiting for a heartbeat.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// PIE's bound on the entire filesystem path of a socket (`IPC_SOCKET_PATH_MAX_BYTES`).
const MAX_SOCKET_PATH_BYTES: usize = 103;
/// The longest socket filename the engine ever creates (per-channel response sockets).
const LONGEST_SOCKET_NAME: &str = "pie_response_ffffffffffffffff.ipc";

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV1A64_OFFSET_BASIS, |digest, &byte| {
        (digest ^ u64::from(byte)).wrapping_mul(FNV1A64_PRIME)
    })
}

/// Lexical normalization: drop `.` components and resolve `..` against the
/// preceding component, like `std::filesystem::path::lexically_normal`.
fn lexically_normal(path: &Path) -> PathBuf {
    let absolute = path.is_absolute();
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                // `..` at an absolute root is clamped (std::filesystem does the
                // same); a relative path keeps leading `..` components.
                let at_root = out.parent().is_none();
                if at_root {
                    if !absolute {
                        out.push("..");
                    }
                } else {
                    out.pop();
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

/// `std::filesystem::weakly_canonical`: canonicalize the longest existing
/// prefix, then append and lexically normalize the missing suffix. Falls back
/// to an absolute, normalized path when even the prefix cannot be resolved.
fn canonical_path(path: &Path) -> PathBuf {
    let mut prefix = absolute(path);
    let mut suffix: Vec<OsString> = Vec::new();
    while !prefix.exists() {
        // Walk whole components, not file_name(): a trailing `..` or `.` must
        // stay in the suffix so the search continues to the real existing prefix.
        let last = match prefix.components().next_back() {
            Some(component) => component.as_os_str().to_owned(),
            None => break,
        };
        if !prefix.pop() {
            break;
        }
        suffix.push(last);
    }
    match std::fs::canonicalize(&prefix) {
        Ok(mut resolved) => {
            for part in suffix.iter().rev() {
                resolved.push(part);
            }
            lexically_normal(&resolved)
        }
        Err(_) => lexically_normal(&absolute(path)),
    }
}

fn socket_paths_fit(root: &Path) -> bool {
    root.join(LONGEST_SOCKET_NAME).as_os_str().len() <= MAX_SOCKET_PATH_BYTES
}

/// Project a requested root the way PIE does: keep it when every socket path
/// fits, otherwise use `/tmp/orchard-ipc-<uid>-<fnv1a64 of the canonical root>`.
fn bounded_ipc_root(candidate: &Path) -> PathBuf {
    let canonical = canonical_path(candidate);
    if socket_paths_fit(&canonical) {
        return canonical;
    }
    let digest = fnv1a64(canonical.as_os_str().as_encoded_bytes());
    // SAFETY: getuid has no preconditions and cannot fail.
    let uid = unsafe { libc::getuid() };
    canonical_path(Path::new("/tmp")).join(format!("orchard-ipc-{uid}-{digest:016x}"))
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// PIE's cache directory: `ORCHARD_CACHE_ROOT`, else the platform cache
/// directory under `com.theproxycompany`. Created before use, as PIE does.
fn cache_root() -> PathBuf {
    if let Some(root) = non_empty_env("ORCHARD_CACHE_ROOT") {
        return PathBuf::from(root);
    }
    let home = std::env::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    platform_cache_dir(&home).join("com.theproxycompany")
}

/// The per-user cache directory: `~/Library/Caches` on macOS, else
/// `XDG_CACHE_HOME` or `~/.cache`.
pub(crate) fn platform_cache_dir(home: &Path) -> PathBuf {
    if cfg!(target_os = "macos") {
        home.join("Library/Caches")
    } else {
        non_empty_env("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".cache"))
    }
}

/// Get the IPC root directory for socket files.
///
/// `ORCHARD_IPC_ROOT` (non-empty) is honored first, then
/// `ORCHARD_CACHE_ROOT/ipc`, then the platform default. Whichever applies is
/// bounded exactly like the engine bounds it, so both sides always agree.
pub fn ipc_root() -> PathBuf {
    let requested = match non_empty_env("ORCHARD_IPC_ROOT") {
        Some(root) => PathBuf::from(root),
        None => {
            let cache = cache_root();
            // PIE creates its cache directory before projecting the IPC child;
            // weak canonicalization depends on which prefix exists.
            let _ = std::fs::create_dir_all(&cache);
            cache.join("ipc")
        }
    };
    let root = bounded_ipc_root(&requested);
    if !root.exists() && std::fs::create_dir_all(&root).is_ok() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700));
        }
    }
    root
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

/// A flow-controlled, client-owned response route on the existing IPC root.
pub fn pull_response_url(channel_id: u64) -> String {
    as_ipc_url(ipc_root().join(format!("pie_response_{channel_id:x}.ipc")))
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
