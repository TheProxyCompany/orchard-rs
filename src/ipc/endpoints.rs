//! IPC endpoint definitions for PIE communication.
//!
//! These endpoints mirror the Python/Swift implementations.
//! PIE uses NNG (nanomsg-next-gen) for high-performance IPC.

use std::os::unix::{ffi::OsStrExt, fs::DirBuilderExt};
use std::path::{Component, Path, PathBuf};

const IPC_SOCKET_PATH_MAX_BYTES: usize = 103;
const LONGEST_SOCKET_NAME: &str = "pie_response_ffffffffffffffff.ipc";

fn fnv1a64(value: &[u8]) -> u64 {
    value.iter().fold(0xcbf29ce484222325, |digest, byte| {
        (digest ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn append_normalized(mut root: PathBuf, suffix: &Path, trailing_directory: bool) -> PathBuf {
    for component in suffix.components() {
        match component {
            Component::ParentDir => {
                root.pop();
            }
            Component::CurDir => {}
            other => root.push(other.as_os_str()),
        }
    }
    // C++ lexically_normal retains a trailing directory separator; it affects
    // both a missing suffix's hash and the canonicalization-error fallback.
    if trailing_directory && !suffix.as_os_str().is_empty() {
        root.push("");
    }
    root
}

fn canonical_path(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .expect("Could not determine current directory for IPC root")
            .join(path)
    };
    // Path::strip_prefix normalizes away a trailing '/.', so retain this from
    // the original spelling before splitting off the existing prefix.
    let bytes = absolute.as_os_str().as_bytes();
    let trailing_directory =
        bytes.ends_with(b"/") || bytes.ends_with(b"/.") || bytes.ends_with(b"/..");

    // Match PIE's weakly_canonical: resolve the existing prefix, then normalize
    // the not-yet-created suffix. Resolving symlinks must precede removing '..'.
    for prefix in absolute.ancestors() {
        match std::fs::canonicalize(prefix) {
            Ok(canonical) => {
                return append_normalized(
                    canonical,
                    absolute.strip_prefix(prefix).unwrap(),
                    trailing_directory,
                );
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound
                    || error.raw_os_error() == Some(libc::ENOTDIR) => {}
            // PIE falls back without resolving aliases on EACCES/ELOOP and
            // other errors; they are not evidence of an absent suffix.
            Err(_) => return append_normalized(PathBuf::new(), &absolute, trailing_directory),
        }
    }
    append_normalized(PathBuf::new(), &absolute, trailing_directory)
}

fn socket_paths_fit(root: &Path) -> bool {
    root.join(LONGEST_SOCKET_NAME).as_os_str().as_bytes().len() <= IPC_SOCKET_PATH_MAX_BYTES
}

fn bounded_ipc_root(candidate: &Path) -> PathBuf {
    let canonical = canonical_path(candidate);
    if socket_paths_fit(&canonical) {
        return canonical;
    }

    let digest = fnv1a64(canonical.as_os_str().as_bytes());
    // getuid has no preconditions and matches PIE/Python's per-user namespace.
    let uid = unsafe { libc::getuid() };
    let compact =
        canonical_path(Path::new("/tmp")).join(format!("orchard-ipc-{uid}-{digest:016x}"));
    assert!(
        socket_paths_fit(&compact),
        "Could not construct a bounded IPC root for {}",
        canonical.display()
    );
    compact
}

fn private_ipc_root(candidate: &Path) -> PathBuf {
    let path = bounded_ipc_root(candidate);
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&path)
        .expect("Could not create private IPC directory");
    path
}

fn cache_ipc_root(cache: &Path) -> PathBuf {
    // PIE get_cache_dir creates the original cache before resolving IPC paths.
    std::fs::create_dir_all(cache).expect("Could not create cache directory");
    private_ipc_root(&cache.join("ipc"))
}

/// Get the IPC root directory for socket files.
///
/// Determines the stable, user-specific root directory for IPC socket files.
/// This ensures that all Orchard processes communicate through a predictable,
/// private location, avoiding pollution of system-wide directories like /tmp.
pub fn ipc_root() -> PathBuf {
    // ORCHARD_IPC_ROOT is an escape hatch for development or containerized environments.
    if let Some(root) = std::env::var_os("ORCHARD_IPC_ROOT").filter(|root| !root.is_empty()) {
        return private_ipc_root(Path::new(&root));
    }

    if let Some(cache_root) = std::env::var_os("ORCHARD_CACHE_ROOT").filter(|root| !root.is_empty())
    {
        return cache_ipc_root(Path::new(&cache_root));
    }

    // Default to the standard application cache directory.
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));

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

    cache_ipc_root(&base.join("com.theproxycompany"))
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
        assert!(root.is_absolute());
        assert!(root.is_dir());
        assert!(socket_paths_fit(&root));
    }

    #[test]
    fn test_urls_are_valid() {
        assert!(request_url().starts_with("ipc://"));
        assert!(response_url().starts_with("ipc://"));
        assert!(management_url().starts_with("ipc://"));
    }

    #[test]
    fn fnv_matches_python_fixture() {
        assert_eq!(fnv1a64(b"/deterministic/orchard/ipc"), 0x9262f36ffeae1c45);
    }

    #[test]
    fn short_root_preserves_canonical_path() {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        assert_eq!(
            bounded_ipc_root(root.path()),
            root.path().canonicalize().unwrap()
        );
    }

    #[test]
    fn long_roots_are_bounded_deterministic_and_distinct() {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let first = root.path().join("a".repeat(140)).join("ipc");
        let second = root.path().join("b".repeat(140)).join("ipc");
        let compact = bounded_ipc_root(&first);
        assert_eq!(compact, bounded_ipc_root(&first));
        assert_ne!(compact, bounded_ipc_root(&second));
        assert_ne!(compact, canonical_path(&first));
        assert!(socket_paths_fit(&compact));
    }

    #[test]
    fn byte_limit_is_inclusive_and_counts_utf8_bytes() {
        let root = canonical_path(Path::new("/tmp"));
        let available = IPC_SOCKET_PATH_MAX_BYTES
            - root.as_os_str().as_bytes().len()
            - LONGEST_SOCKET_NAME.len()
            - 2;
        let fits = root.join("a".repeat(available));
        let over = root.join("a".repeat(available + 1));
        assert_eq!(bounded_ipc_root(&fits), fits);
        assert_ne!(bounded_ipc_root(&over), over);
        let unicode = root.join("\u{e9}".repeat(available));
        assert_ne!(bounded_ipc_root(&unicode), unicode);
    }

    #[test]
    fn symlink_resolution_precedes_parent_normalization() {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let real = root.path().join("real/nested");
        std::fs::create_dir_all(&real).unwrap();
        let alias = root.path().join("alias");
        std::os::unix::fs::symlink(&real, &alias).unwrap();
        let expected = root.path().canonicalize().unwrap().join("real/new");
        assert_eq!(canonical_path(&alias.join("../new")), expected);
        assert_eq!(
            bounded_ipc_root(&alias.join("x".repeat(140))),
            bounded_ipc_root(&real.join("x".repeat(140)))
        );
    }

    #[test]
    fn missing_suffix_preserves_pie_trailing_directory_separator() {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let missing = root.path().join("missing");
        let mut expected = root.path().canonicalize().unwrap().join("missing");
        expected.push("");
        for suffix in ["missing/", "missing/.", "missing/child/.."] {
            assert_eq!(
                canonical_path(&root.path().join(suffix)).as_os_str(),
                expected.as_os_str()
            );
        }
        std::fs::create_dir(&missing).unwrap();
        assert_eq!(
            canonical_path(&root.path().join("missing/")),
            missing.canonicalize().unwrap()
        );
    }

    #[test]
    fn new_directory_is_private_without_chmod_of_existing_directory() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let candidate = root.path().join("private");
        let actual = private_ipc_root(&candidate);
        assert_eq!(
            std::fs::metadata(&actual).unwrap().permissions().mode() & 0o777,
            0o700
        );
        std::fs::set_permissions(&actual, std::fs::Permissions::from_mode(0o750)).unwrap();
        assert_eq!(private_ipc_root(&candidate), actual);
        assert_eq!(
            std::fs::metadata(&actual).unwrap().permissions().mode() & 0o777,
            0o750
        );
    }

    #[test]
    fn cache_fallback_creates_original_before_hashing() {
        const CHILD_EXPECTED: &str = "ORCHARD_IPC_CACHE_REGRESSION_EXPECTED";
        if let Some(expected) = std::env::var_os(CHILD_EXPECTED) {
            assert_eq!(ipc_root(), PathBuf::from(expected));
            return;
        }
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let real = root.path().join("real/nested");
        std::fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, root.path().join("alias")).unwrap();
        let long = "x".repeat(140);
        let cache = root.path().join("missing/../alias").join(&long);
        let expected = bounded_ipc_root(&real.join(&long).join("ipc"));
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "cache_fallback_creates_original_before_hashing",
                "--nocapture",
            ])
            .env_remove("ORCHARD_IPC_ROOT")
            .env("ORCHARD_CACHE_ROOT", &cache)
            .env(CHILD_EXPECTED, &expected)
            .output()
            .unwrap();
        // The child owns environment changes; parallel library tests are unaffected.
        let _ = std::fs::remove_dir(&expected);
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(cache.is_dir());
    }

    #[test]
    fn canonical_symlink_loop_uses_absolute_lexical_fallback() {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let base = root.path().canonicalize().unwrap();
        let real = base.join("real/nested");
        std::fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, base.join("alias")).unwrap();
        std::os::unix::fs::symlink("loop", real.join("loop")).unwrap();
        let candidate = base.join("alias/loop").join("x".repeat(140));
        assert_eq!(
            std::fs::canonicalize(&candidate)
                .unwrap_err()
                .raw_os_error(),
            Some(libc::ELOOP)
        );
        assert_eq!(canonical_path(&candidate), candidate);
    }
}
