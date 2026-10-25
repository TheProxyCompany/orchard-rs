//! Real PIE lease handoff while a close waits for the lifecycle lock.

use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use orchard::engine::lifecycle::EnginePaths;
use orchard::engine::multiprocess::{pid_is_alive, read_pid_file, wait_for_exit};
use orchard::{Client, InferenceEngine, ModelRegistry};

#[test]
fn concurrent_close_preserves_the_new_instance_lease() {
    let namespace = tempfile::Builder::new().prefix("orl-").tempdir().unwrap();
    std::env::set_var("ORCHARD_CACHE_ROOT", namespace.path());
    std::env::set_var("ORCHARD_IPC_ROOT", namespace.path().join("ipc"));
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut engine = runtime.block_on(InferenceEngine::new()).unwrap();
    let paths = EnginePaths::new().unwrap();
    let pid = read_pid_file(&paths.pid_file).unwrap();

    let lock = std::fs::File::create(&paths.lock_file).unwrap();
    lock.lock().unwrap();
    let (started, entered) = mpsc::channel();
    let (finished, done) = mpsc::channel();
    let closing = std::thread::spawn(move || {
        started.send(()).unwrap();
        let result = engine.close();
        finished.send(()).unwrap();
        result
    });
    entered.recv_timeout(Duration::from_secs(2)).unwrap();
    // Release during the blocked close's backoff so the new constructor can
    // acquire the real lock first. Neither registration nor IPC is mocked.
    std::thread::sleep(Duration::from_millis(200));
    assert!(done.try_recv().is_err());
    drop(lock);
    let mut replacement = runtime.block_on(InferenceEngine::new()).unwrap();
    let constructed_before_close = done.try_recv().is_err();
    closing.join().unwrap().unwrap();

    let result = runtime.block_on(async {
        assert!(
            constructed_before_close,
            "The constructor did not overtake the waiting close"
        );
        let registry = Arc::new(ModelRegistry::new().unwrap());
        let client = Client::connect(registry).await?;
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut replies = 0;
        while Instant::now() < deadline {
            client.cancel_request(999999).await?;
            assert_eq!(read_pid_file(&paths.pid_file), Some(pid));
            assert!(pid_is_alive(pid));
            replies += 1;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok::<_, orchard::Error>(replies)
    });
    replacement.close().unwrap();
    let exited = wait_for_exit(pid, Duration::from_secs(10));
    if !exited {
        InferenceEngine::shutdown(Duration::from_secs(10)).unwrap();
    }
    assert!(exited, "The final lease did not release the engine");
    let replies = result.expect("The new instance lost its live PIE lease");
    println!(
        "{}",
        serde_json::json!({"passed": true, "pid": pid, "live_replies_after_close": replies, "graceful_final_exit": true})
    );
}
