//! Check for the GPU lease (tests/project/gpu_lease.rs) against a temp lease
//! file: no engine, no GPU, and never the machine's real lease.
//!
//!   cargo test --test gpu_lease
//!
//! A session is this same binary run as `gpu_lease session <lease path>`: it
//! takes the lease, prints `through`, and stays alive until its stdin closes.
//! Run as `gpu_lease engine-session <lease path>` it exits at once instead and
//! leaves a stub engine behind (`cat`, which lives until that same stdin
//! closes). That is why this target has its own main (`harness = false`).

#[allow(dead_code)] // hold() is for real sessions; the check only uses hold_at()
#[path = "project/gpu_lease.rs"]
mod gpu_lease;

use std::io::{BufRead, BufReader, Read};
use std::os::unix::fs::PermissionsExt;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let run_as = args.get(1).map(String::as_str);
    if let Some("session" | "engine-session") = run_as {
        gpu_lease::hold_at(&args[2]);
        if run_as == Some("engine-session") {
            // Spawned the way the library spawns the engine: a plain Command,
            // after the lease is held, that outlives this process.
            #[allow(clippy::zombie_processes)] // outliving us is the point
            Command::new("cat").stdout(Stdio::null()).spawn().unwrap();
        }
        println!(
            "through held={}",
            std::env::var("PROXY_GPU_LEASE_HELD").unwrap_or_default()
        );
        if run_as == Some("session") {
            let _ = std::io::stdin().read_line(&mut String::new());
        }
        return;
    }

    // A broken lease shows up as a session that blocks forever; fail instead.
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(60));
        eprintln!("gpu_lease: timed out; a session blocked that should not have");
        std::process::exit(1);
    });

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gpu.lease");
    let path = path.to_str().unwrap();

    // The first session gets the lease at once, names itself in a file any
    // user can open, and marks its own environment for its children.
    let mut first = session("session", path, None, Stdio::piped());
    let first_line = format!("pid={} orchard-rs gpu_lease since=", first.id());
    let mut first_out = BufReader::new(first.stdout.take().unwrap());
    assert_eq!(line(&mut first_out), "through held=1");
    assert!(holder(path).starts_with(&first_line), "{}", holder(path));
    let mode = std::fs::metadata(path).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o666);

    // While it holds: a child of a holder and an explicit opt-out both go
    // straight through, without waiting and without touching the file.
    for skip in [("PROXY_GPU_LEASE_HELD", "1"), ("PROXY_GPU_LEASE", "0")] {
        let out = session("session", path, Some(skip), Stdio::null())
            .wait_with_output()
            .unwrap();
        assert!(out.stdout.starts_with(b"through"), "{skip:?}: {out:?}");
        assert!(out.stderr.is_empty(), "{skip:?}: {out:?}");
        assert!(holder(path).starts_with(&first_line), "{}", holder(path));
    }

    // A second session says who it is waiting for, blocks, and gets the lease
    // only once the first session is gone.
    let second = blocked_session(path, &first_line);
    drop(first.stdin.take());
    first.wait().unwrap();
    gets_the_lease(second, path);

    // The GPU tenant is the engine, a child that outlives its session (an
    // example's normal exit, a killed test binary). It keeps the lease: the
    // next session waits on a holder line that names a dead pid, until the
    // stub engine is gone too.
    let mut third = session("engine-session", path, None, Stdio::piped());
    let third_line = format!("pid={} orchard-rs gpu_lease since=", third.id());
    let stub_engine_stdin = third.stdin.take(); // wait() would close it
    assert_eq!(
        line(&mut BufReader::new(third.stdout.take().unwrap())),
        "through held=1"
    );
    third.wait().unwrap();
    let fourth = blocked_session(path, &third_line);
    drop(stub_engine_stdin);
    gets_the_lease(fourth, path);

    println!(
        "gpu_lease: sessions took turns; a holder's child and PROXY_GPU_LEASE=0 did not wait; \
         an engine kept the lease after its session"
    );
}

fn session(mode: &str, path: &str, env: Option<(&str, &str)>, stdin: Stdio) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args([mode, path])
        // This check may itself run under a lease holder; its sessions must not inherit that.
        .env_remove("PROXY_GPU_LEASE")
        .env_remove("PROXY_GPU_LEASE_HELD")
        .envs(env)
        .stdin(stdin)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// A new session that says it is waiting for `held_by`, then blocks.
fn blocked_session(path: &str, held_by: &str) -> Child {
    let mut waiter = session("session", path, None, Stdio::null());
    let waiting = line(&mut BufReader::new(waiter.stderr.take().unwrap()));
    let expected = format!("waiting for the GPU lease ({path}) held by: {held_by}");
    assert!(
        waiting.starts_with(&expected),
        "no waiting line: {waiting:?}, wanted: {expected:?}"
    );
    std::thread::sleep(Duration::from_millis(300));
    assert!(waiter.try_wait().unwrap().is_none(), "did not block");
    assert!(holder(path).starts_with(held_by), "{}", holder(path));
    waiter
}

/// That session goes through now that nothing holds the lease, and names itself.
fn gets_the_lease(mut waiter: Child, path: &str) {
    let mut out = String::new();
    let mut stdout = waiter.stdout.take().unwrap();
    stdout.read_to_string(&mut out).unwrap();
    assert_eq!(out.trim(), "through held=1");
    let waiter_line = format!("pid={} orchard-rs gpu_lease since=", waiter.id());
    assert!(holder(path).starts_with(&waiter_line), "{}", holder(path));
    waiter.wait().unwrap();
}

fn holder(path: &str) -> String {
    std::fs::read_to_string(path).unwrap()
}

fn line(reader: &mut impl BufRead) -> String {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    line.trim_end().to_string()
}
