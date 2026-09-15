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
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let base = if cfg!(target_os = "macos") {
        home.join("Library/Caches")
    } else {
        non_empty_env("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".cache"))
    };
    base.join("com.theproxycompany")
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
    fn fnv1a64_matches_pie() {
        // Same digest PIE's platform_utils.cpp produces for these bytes.
        assert_eq!(fnv1a64(b"abc"), 0xe71f_a219_0541_574b);
        let long = format!("/private/tmp/{}", "x".repeat(95));
        assert_eq!(fnv1a64(long.as_bytes()), 0x6afb_dbbb_b309_ece4);
    }

    #[test]
    fn short_root_is_kept_verbatim_after_canonicalization() {
        let root = bounded_ipc_root(Path::new("/tmp/orc-abl/com.theproxycompany/ipc"));
        // /tmp resolves to /private/tmp on macOS; the suffix is preserved.
        assert!(
            root.ends_with("orc-abl/com.theproxycompany/ipc"),
            "{root:?}"
        );
        assert!(socket_paths_fit(&root));
    }

    #[test]
    fn long_root_projects_to_hashed_tmp_directory() {
        let long = format!("/private/tmp/{}", "x".repeat(95));
        let root = bounded_ipc_root(Path::new(&long));
        let uid = unsafe { libc::getuid() };
        let expected_name = format!("orchard-ipc-{uid}-6afbdbbbb309ece4");
        assert_eq!(root.file_name().unwrap().to_str().unwrap(), expected_name);
        assert!(
            root.starts_with(canonical_path(Path::new("/tmp"))),
            "{root:?}"
        );
        assert!(socket_paths_fit(&root));
    }

    #[test]
    fn boundary_is_the_full_socket_path_at_103_bytes() {
        // Build a root under /private/tmp whose longest socket path is exactly 103 bytes,
        // then one byte longer.
        let base = canonical_path(Path::new("/tmp"));
        let fill = MAX_SOCKET_PATH_BYTES
            - base.as_os_str().len()
            - 1 // separator before the root name
            - 1 // separator before the socket name
            - LONGEST_SOCKET_NAME.len();
        let fits = base.join("y".repeat(fill));
        let over = base.join("y".repeat(fill + 1));
        assert_eq!(bounded_ipc_root(&fits), fits);
        assert_ne!(bounded_ipc_root(&over), over);
    }

    #[test]
    fn missing_suffix_is_normalized_not_resolved() {
        let candidate = Path::new("/tmp/./orc-missing-a/../orc-missing-b/ipc");
        let root = canonical_path(candidate);
        assert!(root.ends_with("orc-missing-b/ipc"), "{root:?}");
        assert!(!root.to_string_lossy().contains(".."));
    }

    #[test]
    fn trailing_parent_dir_still_resolves_the_existing_symlink_prefix() {
        // /tmp is a symlink to /private/tmp on macOS; the missing child plus a
        // trailing `..` must not stop the walk before the symlink is resolved.
        let root = canonical_path(Path::new("/tmp/orc-does-not-exist-zzz/.."));
        assert_eq!(root, canonical_path(Path::new("/tmp")));
        let deeper = canonical_path(Path::new("/tmp/orc-missing-a/orc-missing-b/../.."));
        assert_eq!(deeper, canonical_path(Path::new("/tmp")));
    }

    #[test]
    fn parent_dir_is_clamped_at_an_absolute_root() {
        assert_eq!(
            lexically_normal(Path::new("/../../tmp")),
            PathBuf::from("/tmp")
        );
        assert_eq!(
            lexically_normal(Path::new("/a/../../b")),
            PathBuf::from("/b")
        );
        assert_eq!(lexically_normal(Path::new("../x")), PathBuf::from("../x"));
        assert_eq!(
            canonical_path(Path::new("/../../tmp")),
            canonical_path(Path::new("/tmp"))
        );
    }

    #[test]
    fn urls_are_ipc_scheme() {
        assert!(request_url().starts_with("ipc://"));
        assert!(response_url().starts_with("ipc://"));
        assert!(management_url().starts_with("ipc://"));
        assert!(request_url().len() - "ipc://".len() <= MAX_SOCKET_PATH_BYTES);
    }
}
