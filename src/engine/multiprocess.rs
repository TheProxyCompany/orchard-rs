//! Cross-process coordination utilities.
//!
//! Provides PID file management, process liveness checks, and signal handling
//! for coordinating multiple clients with a shared engine process.

use std::fs;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

/// Check if a process is still alive.
pub fn pid_is_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }

    #[cfg(unix)]
    {
        #[cfg(target_os = "macos")]
        {
            if pid_is_zombie(pid) {
                return false;
            }
        }

        // Signal 0 doesn't send a signal but checks if the process exists
        unsafe {
            let result = libc::kill(pid as libc::pid_t, 0);
            if result == 0 {
                return true;
            }
            // ESRCH means process doesn't exist
            // EPERM means it exists but we don't have permission
            *libc::__error() == libc::EPERM
        }
    }

    #[cfg(not(unix))]
    {
        true // Non-Unix: can't check liveness, assume running
    }
}

#[cfg(target_os = "macos")]
fn pid_is_zombie(pid: u32) -> bool {
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let info_size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    let result = unsafe {
        libc::proc_pidinfo(
            pid as libc::c_int,
            libc::PROC_PIDTBSDINFO,
            0,
            &mut info as *mut _ as *mut libc::c_void,
            info_size,
        )
    };
    result == info_size && info.pbi_status == libc::SZOMB
}

#[cfg(not(target_os = "macos"))]
fn pid_is_zombie(_pid: u32) -> bool {
    false
}

/// Check if a PID belongs to the PIE engine binary.
#[cfg(target_os = "macos")]
pub fn pid_is_engine(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }

    let mut buf = vec![0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    let result = unsafe {
        libc::proc_pidpath(
            pid as libc::c_int,
            buf.as_mut_ptr() as *mut libc::c_void,
            buf.len() as u32,
        )
    };
    if result <= 0 {
        return false;
    }

    let cstr = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr() as *const i8) };
    let path = cstr.to_string_lossy();
    path.ends_with("/proxy_inference_engine")
}

#[cfg(not(target_os = "macos"))]
pub fn pid_is_engine(_pid: u32) -> bool {
    true
}

/// Read a PID from a file.
pub fn read_pid_file(path: &Path) -> Option<u32> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .filter(|&pid| pid > 0)
}

/// Write a PID to a file.
pub fn write_pid_file(path: &Path, pid: u32) -> io::Result<()> {
    fs::write(path, format!("{}\n", pid))
}

/// Read reference PIDs from a JSON file.
pub fn read_ref_pids(path: &Path) -> Vec<u32> {
    match fs::read_to_string(path) {
        Ok(content) if !content.is_empty() => {
            serde_json::from_str::<Vec<u32>>(&content).unwrap_or_default()
        }
        _ => Vec::new(),
    }
}

/// Write reference PIDs to a JSON file.
pub fn write_ref_pids(path: &Path, pids: &[u32]) -> io::Result<()> {
    // Deduplicate and filter
    let mut unique: Vec<u32> = Vec::new();
    for &pid in pids {
        if pid > 0 && !unique.contains(&pid) {
            unique.push(pid);
        }
    }

    if unique.is_empty() {
        // Remove the file if no PIDs
        let _ = fs::remove_file(path);
        return Ok(());
    }

    // Atomic write via temp file
    let tmp_path = path.with_extension("tmp");
    fs::write(&tmp_path, serde_json::to_string(&unique)?)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Filter a list of PIDs to only those that are still alive.
pub fn filter_alive_pids(pids: &[u32]) -> Vec<u32> {
    pids.iter()
        .copied()
        .filter(|&pid| pid_is_alive(pid))
        .collect()
}

/// Stop an engine process gracefully.
///
/// Sends SIGINT, waits, then SIGTERM, then SIGKILL if necessary.
/// Returns true if the process was stopped successfully.
#[cfg(unix)]
pub fn stop_engine_process(pid: u32, timeout: Duration) -> bool {
    use libc::{SIGINT, SIGKILL, SIGTERM};

    let pid = pid as libc::pid_t;

    // Try SIGINT first
    if unsafe { libc::kill(pid, SIGINT) } != 0 {
        return !pid_is_alive(pid as u32);
    }

    if wait_for_exit(pid as u32, timeout) {
        return true;
    }
    if pid_is_zombie(pid as u32) {
        return true;
    }

    log::warn!("Engine did not respond to SIGINT, sending SIGTERM");

    // Try SIGTERM
    if unsafe { libc::kill(pid, SIGTERM) } != 0 {
        return !pid_is_alive(pid as u32);
    }

    if wait_for_exit(pid as u32, timeout) {
        return true;
    }
    if pid_is_zombie(pid as u32) {
        return true;
    }

    log::error!("Engine did not respond to SIGTERM, sending SIGKILL");

    // Last resort: SIGKILL
    unsafe { libc::kill(pid, SIGKILL) };

    if wait_for_exit(pid as u32, Duration::from_secs(5)) {
        return true;
    }

    pid_is_zombie(pid as u32)
}

#[cfg(not(unix))]
pub fn stop_engine_process(_pid: u32, _timeout: Duration) -> bool {
    // On non-Unix systems, we can't send signals
    false
}

/// Wait for a process to exit.
pub fn wait_for_exit(pid: u32, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;

    while Instant::now() < deadline {
        if !pid_is_alive(pid) {
            return true;
        }

        #[cfg(unix)]
        {
            // Try to reap the process if it's our child
            let result = unsafe {
                let mut status: libc::c_int = 0;
                libc::waitpid(pid as libc::pid_t, &mut status, libc::WNOHANG)
            };

            if result == pid as libc::pid_t {
                return true;
            }
        }

        std::thread::sleep(Duration::from_millis(50));
    }

    !pid_is_alive(pid)
}

/// Reap a zombie process.
#[cfg(unix)]
pub fn reap_engine_process(pid: u32) {
    loop {
        let result = unsafe {
            let mut status: libc::c_int = 0;
            libc::waitpid(pid as libc::pid_t, &mut status, 0)
        };

        if result >= 0 {
            break;
        }

        let errno = unsafe { *libc::__error() };
        if errno == libc::EINTR {
            continue; // Interrupted, retry
        }
        break; // ECHILD or other error
    }
}

#[cfg(not(unix))]
pub fn reap_engine_process(_pid: u32) {
    // No-op on non-Unix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_is_alive_current() {
        let pid = std::process::id();
        assert!(pid_is_alive(pid));
    }

    #[test]
    fn test_pid_is_alive_zero() {
        assert!(!pid_is_alive(0));
    }

    #[test]
    fn test_ref_pids_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("refs.json");

        let pids = vec![1234, 5678];
        write_ref_pids(&path, &pids).unwrap();

        let read = read_ref_pids(&path);
        assert_eq!(read, pids);
    }

    #[test]
    fn test_ref_pids_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("refs.json");

        write_ref_pids(&path, &[]).unwrap();
        assert!(!path.exists());

        let read = read_ref_pids(&path);
        assert!(read.is_empty());
    }

    #[test]
    fn test_filter_alive_pids() {
        let current = std::process::id();
        let pids = vec![current, 999999999]; // Current PID should be alive, huge PID should not

        let alive = filter_alive_pids(&pids);
        assert!(alive.contains(&current));
    }
}
