//! Check for the GPU lease (tests/project/gpu_lease.rs) against a temp lease
//! file: no engine, no GPU, and never the machine's real lease.
//!
//!   cargo test --test gpu_lease
//!
//! A session is this same binary run as `gpu_lease session <lease path>`: it
//! takes the lease, prints `through`, and stays alive until its stdin closes.
//! That is why this target has its own main (`harness = false`).

#[allow(dead_code)] // hold() is for real sessions; the check only uses hold_at()
#[path = "project/gpu_lease.rs"]
mod gpu_lease;

use std::io::{BufRead, BufReader, Read};
use std::os::unix::fs::PermissionsExt;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("session") {
        gpu_lease::hold_at(&args[2]);
        println!(
            "through held={}",
            std::env::var("PROXY_GPU_LEASE_HELD").unwrap_or_default()
        );
        let _ = std::io::stdin().read_line(&mut String::new());
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
    let holder = || std::fs::read_to_string(path).unwrap();

    // The first session gets the lease at once, names itself in a file any
    // user can open, and marks its own environment for its children.
    let mut first = session(path, None, Stdio::piped());
    let first_line = format!("pid={} orchard-rs gpu_lease since=", first.id());
    let mut first_out = BufReader::new(first.stdout.take().unwrap());
    assert_eq!(line(&mut first_out), "through held=1");
    assert!(holder().starts_with(&first_line), "{}", holder());
    let mode = std::fs::metadata(path).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o666);

    // While it holds: a child of a holder and an explicit opt-out both go
    // straight through, without waiting and without touching the file.
    for skip in [("PROXY_GPU_LEASE_HELD", "1"), ("PROXY_GPU_LEASE", "0")] {
        let out = session(path, Some(skip), Stdio::null())
            .wait_with_output()
            .unwrap();
        assert!(out.stdout.starts_with(b"through"), "{skip:?}: {out:?}");
        assert!(out.stderr.is_empty(), "{skip:?}: {out:?}");
        assert!(holder().starts_with(&first_line), "{}", holder());
    }

    // A second session says who it is waiting for, then blocks.
    let mut second = session(path, None, Stdio::null());
    let second_line = format!("pid={} orchard-rs gpu_lease since=", second.id());
    let waiting = line(&mut BufReader::new(second.stderr.take().unwrap()));
    let expected = format!("waiting for the GPU lease ({path}) held by: {first_line}");
    assert!(
        waiting.starts_with(&expected),
        "no waiting line: {waiting:?}"
    );
    std::thread::sleep(Duration::from_millis(300));
    assert!(second.try_wait().unwrap().is_none(), "second did not block");
    assert!(holder().starts_with(&first_line), "{}", holder());

    // It gets the lease only once the first session is gone.
    drop(first.stdin.take());
    first.wait().unwrap();
    let mut second_out = String::new();
    let mut stdout = second.stdout.take().unwrap();
    stdout.read_to_string(&mut second_out).unwrap();
    assert_eq!(second_out.trim(), "through held=1");
    assert!(holder().starts_with(&second_line), "{}", holder());
    second.wait().unwrap();

    println!("gpu_lease: sessions took turns; a holder's child and PROXY_GPU_LEASE=0 did not wait");
}

fn session(path: &str, env: Option<(&str, &str)>, stdin: Stdio) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["session", path])
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

fn line(reader: &mut impl BufRead) -> String {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    line.trim_end().to_string()
}
