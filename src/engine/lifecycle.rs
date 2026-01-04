//! Inference engine process lifecycle management.
//!
//! Provides the `InferenceEngine` struct which manages:
//! - PIE process spawning
//! - Reference counting across multiple clients
//! - Graceful shutdown with proper cleanup

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use fs4::fs_std::FileExt;
use thiserror::Error;

use nng::options::Options;

use crate::engine::fetch::EngineFetcher;
use crate::engine::multiprocess::{
    filter_alive_pids, pid_is_alive, read_pid_file, read_ref_pids, reap_engine_process,
    stop_engine_process, write_pid_file, write_ref_pids,
};
use crate::ipc::endpoints::{response_url, EVENT_TOPIC_PREFIX};

const DEFAULT_STARTUP_TIMEOUT_SECS: u64 = 60;

/// Errors that can occur during engine lifecycle management.
#[derive(Error, Debug)]
pub enum LifecycleError {
    #[error("Failed to acquire lock: {0}")]
    LockFailed(String),

    #[error("Engine startup failed: {0}")]
    StartupFailed(String),

    #[error("Engine shutdown failed: {0}")]
    ShutdownFailed(String),

    #[error("Engine already closed")]
    AlreadyClosed,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Fetch error: {0}")]
    Fetch(#[from] crate::engine::fetch::FetchError),
}

pub type Result<T> = std::result::Result<T, LifecycleError>;

/// File paths used by the engine lifecycle manager.
#[derive(Debug, Clone)]
pub struct EnginePaths {
    pub cache_dir: PathBuf,
    pub pid_file: PathBuf,
    pub refs_file: PathBuf,
    pub lock_file: PathBuf,
    pub ready_file: PathBuf,
    pub engine_log_file: PathBuf,
    pub client_log_file: PathBuf,
}

impl EnginePaths {
    pub fn new() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("com.theproxycompany");

        Self {
            pid_file: cache_dir.join("pie.pid"),
            refs_file: cache_dir.join("pie.refs"),
            lock_file: cache_dir.join("pie.lock"),
            ready_file: cache_dir.join("pie.ready"),
            engine_log_file: cache_dir.join("pie.log"),
            client_log_file: cache_dir.join("client.log"),
            cache_dir,
        }
    }
}

impl Default for EnginePaths {
    fn default() -> Self {
        Self::new()
    }
}

/// Global context shared across all InferenceEngine instances in a process.
struct GlobalContext {
    ref_count: AtomicU32,
    initialized: AtomicBool,
}

static GLOBAL_CONTEXT: GlobalContext = GlobalContext {
    ref_count: AtomicU32::new(0),
    initialized: AtomicBool::new(false),
};

/// Manages the PIE (Proxy Inference Engine) process lifecycle.
///
/// Handles:
/// - Binary fetching if not installed
/// - Process spawning with proper daemonization
/// - Reference counting across multiple clients
/// - Graceful shutdown when the last client disconnects
pub struct InferenceEngine {
    paths: EnginePaths,
    fetcher: EngineFetcher,
    startup_timeout: Duration,
    lease_active: bool,
    closed: bool,
    launch_process: Option<Child>,
}

impl InferenceEngine {
    /// Create a new InferenceEngine and connect to (or spawn) the engine process.
    pub async fn new() -> Result<Self> {
        Self::with_options(Default::default(), None).await
    }

    /// Create with custom options.
    pub async fn with_options(
        paths: EnginePaths,
        startup_timeout: Option<Duration>,
    ) -> Result<Self> {
        let fetcher = EngineFetcher::new();
        let startup_timeout =
            startup_timeout.unwrap_or(Duration::from_secs(DEFAULT_STARTUP_TIMEOUT_SECS));

        let mut engine = Self {
            paths,
            fetcher,
            startup_timeout,
            lease_active: false,
            closed: false,
            launch_process: None,
        };

        engine.acquire_lease().await?;

        Ok(engine)
    }

    /// Close this engine instance.
    ///
    /// Decrements the reference count and shuts down the engine if this
    /// was the last reference.
    pub fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }

        let should_release = if self.lease_active {
            let prev = GLOBAL_CONTEXT.ref_count.fetch_sub(1, Ordering::SeqCst);
            prev == 1 // Was the last reference
        } else {
            false
        };

        if !self.lease_active || !should_release {
            self.closed = true;
            self.lease_active = false;
            return Ok(());
        }

        // Acquire lock for cleanup
        let lock_file = std::fs::File::create(&self.paths.lock_file)?;
        lock_file
            .lock_exclusive()
            .map_err(|e| LifecycleError::LockFailed(e.to_string()))?;

        // Read current refs
        let refs = read_ref_pids(&self.paths.refs_file);
        let current_pid = std::process::id();
        let alive_refs: Vec<_> = filter_alive_pids(&refs)
            .into_iter()
            .filter(|&p| p != current_pid)
            .collect();

        let engine_pid = read_pid_file(&self.paths.pid_file);
        let engine_running = engine_pid.map(pid_is_alive).unwrap_or(false);

        if alive_refs.is_empty() {
            if engine_running {
                if let Some(pid) = engine_pid {
                    self.stop_engine_locked(pid)?;
                }
            } else {
                let _ = std::fs::remove_file(&self.paths.pid_file);
                let _ = std::fs::remove_file(&self.paths.ready_file);
            }
            let _ = write_ref_pids(&self.paths.refs_file, &[]);
        } else {
            let _ = write_ref_pids(&self.paths.refs_file, &alive_refs);
        }

        drop(lock_file);

        self.lease_active = false;
        self.closed = true;

        Ok(())
    }

    /// Force shutdown the engine regardless of reference count.
    pub fn shutdown(timeout: Duration) -> Result<()> {
        let paths = EnginePaths::new();

        let lock_file = std::fs::File::create(&paths.lock_file)?;
        lock_file
            .lock_exclusive()
            .map_err(|e| LifecycleError::LockFailed(e.to_string()))?;

        let pid = read_pid_file(&paths.pid_file);

        if pid.is_none() || !pid_is_alive(pid.unwrap()) {
            log::info!("Engine is not running. Cleaning up stale files.");
            let _ = std::fs::remove_file(&paths.pid_file);
            let _ = std::fs::remove_file(&paths.ready_file);
            let _ = std::fs::remove_file(&paths.refs_file);
            return Ok(());
        }

        let pid = pid.unwrap();
        log::info!("Sending shutdown signal to engine process {}", pid);

        if stop_engine_process(pid, timeout) {
            let _ = std::fs::remove_file(&paths.pid_file);
            let _ = std::fs::remove_file(&paths.ready_file);
            let _ = std::fs::remove_file(&paths.refs_file);
            reap_engine_process(pid);
            log::info!("Engine process {} terminated gracefully", pid);
            Ok(())
        } else {
            Err(LifecycleError::ShutdownFailed(format!(
                "Failed to stop engine process {}",
                pid
            )))
        }
    }

    async fn acquire_lease(&mut self) -> Result<()> {
        if self.closed || self.lease_active {
            return Ok(());
        }

        // Ensure cache directory exists
        std::fs::create_dir_all(&self.paths.cache_dir)?;

        // Acquire file lock
        let lock_file = std::fs::File::create(&self.paths.lock_file)?;
        lock_file
            .lock_exclusive()
            .map_err(|e| LifecycleError::LockFailed(e.to_string()))?;

        // Read current state
        let refs = read_ref_pids(&self.paths.refs_file);
        let alive_refs = filter_alive_pids(&refs);

        let engine_pid = read_pid_file(&self.paths.pid_file);
        let engine_running = engine_pid.map(pid_is_alive).unwrap_or(false);

        // Launch engine if needed
        if !engine_running && alive_refs.is_empty() {
            log::debug!("Inference engine not running. Launching new instance.");

            // Clean up stale files
            let _ = std::fs::remove_file(&self.paths.pid_file);
            let _ = std::fs::remove_file(&self.paths.ready_file);

            self.launch_engine().await?;
            self.wait_for_engine_ready().await?;
        }

        // Register this process
        let current_pid = std::process::id();
        let mut new_refs = alive_refs;
        if !new_refs.contains(&current_pid) {
            new_refs.push(current_pid);
        }
        write_ref_pids(&self.paths.refs_file, &new_refs)?;

        // Update global context
        GLOBAL_CONTEXT.ref_count.fetch_add(1, Ordering::SeqCst);
        GLOBAL_CONTEXT.initialized.store(true, Ordering::SeqCst);

        drop(lock_file);

        self.lease_active = true;
        Ok(())
    }

    async fn launch_engine(&mut self) -> Result<()> {
        let engine_path = self.fetcher.get_engine_path().await?;

        log::info!("Launching PIE from {:?}", engine_path);

        // Open log file for engine output
        let log_file = std::fs::File::create(&self.paths.engine_log_file)?;

        let child = Command::new(&engine_path)
            .stdout(Stdio::from(log_file.try_clone()?))
            .stderr(Stdio::from(log_file))
            .spawn()
            .map_err(|e| LifecycleError::StartupFailed(format!("Failed to spawn engine: {}", e)))?;

        self.launch_process = Some(child);
        Ok(())
    }

    async fn wait_for_engine_ready(&self) -> Result<()> {
        log::info!("Waiting for telemetry heartbeat from engine...");

        // Subscribe to telemetry topic via NNG
        let telemetry_topic = [EVENT_TOPIC_PREFIX, b"telemetry"].concat();
        let response_url = response_url();

        let socket = nng::Socket::new(nng::Protocol::Sub0)
            .map_err(|e| LifecycleError::StartupFailed(format!("Failed to create Sub0 socket: {}", e)))?;

        socket
            .set_opt::<nng::options::protocol::pubsub::Subscribe>(telemetry_topic.clone())
            .map_err(|e| LifecycleError::StartupFailed(format!("Failed to subscribe: {}", e)))?;

        socket
            .set_opt::<nng::options::RecvTimeout>(Some(Duration::from_millis(250)))
            .map_err(|e| LifecycleError::StartupFailed(format!("Failed to set timeout: {}", e)))?;

        socket
            .dial(&response_url)
            .map_err(|e| LifecycleError::StartupFailed(format!("Failed to dial {}: {}", response_url, e)))?;

        let deadline = Instant::now() + self.startup_timeout;

        while Instant::now() < deadline {
            // Check if launched process died
            if let Some(ref child) = self.launch_process {
                if !pid_is_alive(child.id()) {
                    return Err(LifecycleError::StartupFailed(
                        "Engine process exited before signaling readiness; check the engine log".into()
                    ));
                }
            }

            // Try to receive a message
            let msg = match socket.recv() {
                Ok(msg) => msg,
                Err(nng::Error::TimedOut) => continue,
                Err(e) => {
                    log::debug!("Error receiving telemetry: {}", e);
                    continue;
                }
            };

            // Parse message: topic\x00json_body
            let bytes = msg.as_slice();
            let parts: Vec<&[u8]> = bytes.splitn(2, |&b| b == 0).collect();
            if parts.len() < 2 {
                log::warn!("Discarding malformed event message while waiting for telemetry");
                continue;
            }

            let (topic_part, json_body) = (parts[0], parts[1]);
            if topic_part != telemetry_topic.as_slice() {
                log::debug!(
                    "Ignoring unexpected startup topic '{}'",
                    String::from_utf8_lossy(topic_part)
                );
                continue;
            }

            // Parse JSON payload
            let payload: serde_json::Value = match serde_json::from_slice(json_body) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!("Discarding malformed telemetry payload: {}", e);
                    continue;
                }
            };

            // Extract health.pid
            let engine_pid = payload
                .get("health")
                .and_then(|h| h.get("pid"))
                .and_then(|p| p.as_u64())
                .map(|p| p as u32);

            match engine_pid {
                Some(pid) if pid > 0 => {
                    if let Err(e) = write_pid_file(&self.paths.pid_file, pid) {
                        log::warn!("Failed to write PID file: {}", e);
                    }
                    log::info!("Received telemetry heartbeat. Engine PID {} recorded.", pid);
                    return Ok(());
                }
                _ => {
                    log::warn!("Telemetry payload missing valid PID; waiting for next heartbeat");
                    continue;
                }
            }
        }

        Err(LifecycleError::StartupFailed(format!(
            "Timed out after {:?}s waiting for telemetry heartbeat from engine",
            self.startup_timeout.as_secs()
        )))
    }

    fn stop_engine_locked(&mut self, pid: u32) -> Result<()> {
        if !pid_is_alive(pid) {
            log::debug!("Engine PID {} already exited", pid);
            let _ = std::fs::remove_file(&self.paths.pid_file);
            let _ = std::fs::remove_file(&self.paths.ready_file);
            return Ok(());
        }

        if !stop_engine_process(pid, Duration::from_secs(5)) {
            return Err(LifecycleError::ShutdownFailed(format!(
                "Failed to stop engine PID {}",
                pid
            )));
        }

        reap_engine_process(pid);
        let _ = std::fs::remove_file(&self.paths.pid_file);
        let _ = std::fs::remove_file(&self.paths.ready_file);

        log::info!("Engine PID {} stopped", pid);
        Ok(())
    }

    /// Generate a unique response channel ID for this client.
    ///
    /// Format: (PID << 32) | random_32_bits
    /// Uses true randomness to avoid collisions between rapid successive calls.
    pub fn generate_response_channel_id() -> u64 {
        use rand::Rng;

        let pid = std::process::id() as u64 & 0xFFFFFFFF;
        let random: u32 = rand::thread_rng().gen();

        let channel_id = (pid << 32) | (random as u64);
        if channel_id == 0 {
            1
        } else {
            channel_id
        }
    }
}

impl Drop for InferenceEngine {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_paths() {
        let paths = EnginePaths::new();
        assert!(paths.cache_dir.to_string_lossy().contains("com.theproxycompany"));
    }

    #[test]
    fn test_generate_channel_id_uniqueness() {
        use std::collections::HashSet;

        // Generate 1000 channel IDs in rapid succession
        let ids: HashSet<u64> = (0..1000)
            .map(|_| InferenceEngine::generate_response_channel_id())
            .collect();

        // All IDs must be unique (HashSet dedupes)
        assert_eq!(ids.len(), 1000, "Channel IDs must be unique across rapid calls");

        // All IDs must be non-zero
        assert!(!ids.contains(&0), "Channel ID must never be zero");

        // All IDs should have the current PID in upper 32 bits
        let expected_pid = std::process::id() as u64 & 0xFFFFFFFF;
        for id in &ids {
            let id_pid = id >> 32;
            assert_eq!(id_pid, expected_pid, "Upper 32 bits must be current PID");
        }
    }
}
