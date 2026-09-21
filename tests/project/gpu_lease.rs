//! The machine's GPU lease: one heavy GPU tenant at a time.
//!
//! Apple's GPU firmware on the M3 Ultra test machine locks up when several
//! processes stream large weight sets at once, even when each is clean alone.
//! That machine is also the CI runner, so CI engines, local test runs and
//! benchmarks all meet on one GPU. Every Proxy repo therefore implements the
//! same contract: anything outside production that starts a real engine takes
//! an exclusive flock on `/tmp/proxy-gpu.lease` and keeps it until the process
//! exits. Here the engine it starts keeps the lease too (see [`hold_at`]).
//!
//! Production never takes the lease, so this is not part of the `orchard`
//! library. It is one file, pulled in with `#[path]` by the test fixture and by
//! the examples.

use std::fs::{File, Permissions, TryLockError};
use std::io::{Read, Seek, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;

/// Block until this process is the machine's one heavy GPU tenant. Call it
/// before starting an engine; there is no timeout, a CI job waits for a local
/// run and the other way round.
///
/// Does nothing under `PROXY_GPU_LEASE_HELD` (a parent holds the lease, and
/// taking it again from a child would deadlock: a flock belongs to the open
/// file, not the process tree) or under `PROXY_GPU_LEASE=0` (two tenants on
/// purpose, for an experiment).
pub fn hold() {
    hold_at("/tmp/proxy-gpu.lease");
}

/// [`hold`] on another file. The path is an argument only so that
/// tests/gpu_lease.rs can check this against a temp file.
pub fn hold_at(path: &str) {
    if std::env::var("PROXY_GPU_LEASE").as_deref() == Ok("0")
        || std::env::var_os("PROXY_GPU_LEASE_HELD").is_some_and(|held| !held.is_empty())
    {
        return;
    }

    let mut file = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .unwrap_or_else(|e| panic!("cannot open the GPU lease {path}: {e}"));
    // Mode 0666 whatever the umask: another user's session has to open the
    // same file read-write. Only the file's owner may chmod; the rest skip it.
    let _ = file.set_permissions(Permissions::from_mode(0o666));
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            let mut holder = String::new();
            let _ = (&file).take(300).read_to_string(&mut holder);
            // Not eprintln!: libtest captures that, and a test run would sit
            // silent for as long as it waits.
            let _ = writeln!(
                std::io::stderr(),
                "waiting for the GPU lease ({path}) held by: {}",
                holder.trim()
            );
            file.lock()
                .unwrap_or_else(|e| panic!("cannot lock the GPU lease {path}: {e}"));
        }
        Err(TryLockError::Error(e)) => panic!("cannot lock the GPU lease {path}: {e}"),
    }

    // The GPU tenant is the engine, and it outlives this process: an example
    // only deregisters on exit, and a killed test binary never reaches the
    // shutdown. std opens files close-on-exec, which would free the lease with
    // the engine still on the GPU, so let children inherit this descriptor:
    // the engine, spawned later by the library's plain Command, shares the
    // open file, and the flock lasts until this process AND its engine are
    // gone. The price: a wedged engine keeps the lease until it is killed, and
    // the holder line may then name a dead pid (`lsof /tmp/proxy-gpu.lease`
    // shows the real holder). Not covered: an engine an example adopts rather
    // than starts; one it starts keeps the lease while any later client keeps
    // it running.
    // SAFETY: fcntl(F_SETFD) on a descriptor this function owns; no memory.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, 0) } == -1 {
        let e = std::io::Error::last_os_error();
        panic!("cannot pass the GPU lease {path} on to the engine: {e}");
    }

    // CARGO_CRATE_NAME is the target this file was compiled into: the test
    // binary (buckshot, golden, ...) or the example.
    file.set_len(0)
        .and_then(|()| file.rewind())
        .and_then(|()| {
            writeln!(
                file,
                "pid={} orchard-rs {} since={}",
                std::process::id(),
                env!("CARGO_CRATE_NAME"),
                local_time()
            )
        })
        .unwrap_or_else(|e| panic!("cannot write the GPU lease {path}: {e}"));

    // Children inherit the mark from this process's own environment. The
    // engine is spawned by the library's production code, which this must not
    // touch, so there is no Command::env to hang it on. Other threads do exist
    // by now. What set_var relies on instead: the engine-backed tests get no
    // further than the fixture's OnceLock until this returns, std's own env
    // accessors serialize on std's env lock, and nothing in these targets is
    // known to read the environment from C (getenv, which std cannot lock)
    // this early. ensure_test_namespace() sets ORCHARD_CACHE_ROOT at the same
    // point on the same terms.
    std::env::set_var("PROXY_GPU_LEASE_HELD", "1");
    // Open until exit: the kernel drops the flock when the last process
    // holding this open file is gone.
    std::mem::forget(file);
}

/// Local time as ISO-8601 to the second, like the other repos' holder lines.
fn local_time() -> String {
    let mut buf = [0u8; 20]; // "2026-09-20T14:03:11" and its NUL
    let format = c"%Y-%m-%dT%H:%M:%S";
    // SAFETY: localtime_r and strftime write only into `tm` and `buf`, which
    // live here and are as large as the calls are told.
    let len = unsafe {
        let now = libc::time(std::ptr::null_mut());
        let mut tm = std::mem::zeroed();
        libc::localtime_r(&now, &mut tm);
        libc::strftime(buf.as_mut_ptr().cast(), buf.len(), format.as_ptr(), &tm)
    };
    String::from_utf8_lossy(&buf[..len]).into_owned()
}
