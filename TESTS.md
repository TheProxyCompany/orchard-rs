# E2E Test Issues - January 5, 2026

## Current State

- **Sequential tests work:** `cargo test --test e2e_basic -- --ignored --test-threads=1` passes in ~2.3s
- **Parallel tests hang:** `cargo test --test e2e_basic -- --ignored` hangs indefinitely
- **Test fixture implemented:** Shared `TestFixture` via `OnceLock` reduces setup from 9s to 2.3s

## The Parallel Problem

When tests run in parallel, multiple calls to `client.achat()` race on model activation:

```
Test A: ensure_loaded → send_load_model_command → sets state to Activating, sends IPC
Test B: ensure_loaded → send_load_model_command → sees Activating, returns ERROR
Test C: ensure_loaded → send_load_model_command → sees Activating, returns ERROR
```

The bug is in `src/model/registry.rs` line ~197:
```rust
if entry.state == ModelLoadState::Activating {
    return Err(format!("Model '{}' is already activating", canonical_id));
}
```

**Expected behavior:** Test B and C should WAIT for Test A's activation to complete, not fail.

## Why This Only Affects Rust (Not Python)

- **Python** uses `asyncio.Event` which is **level-triggered**: if set before you wait, `event.wait()` returns immediately
- **Rust** uses `tokio::sync::Notify` which is **edge-triggered**: if `notify_waiters()` fires before you call `notified().await`, the signal is LOST

## Attempted Fix: Switch to `watch` Channel

`tokio::sync::watch` is level-triggered like `asyncio.Event`:
- Holds current value
- `rx.changed().await` checks current state first, never misses updates

### Changes Made (then reverted)

1. Replace `Notify` with `watch::Sender<ModelLoadState>` in `ModelEntry`
2. Add helper methods: `state()`, `set_state()`, `subscribe()`
3. Update `await_model` to use watch receiver
4. Remove all `notify.notify_waiters()` calls (watch auto-notifies on send)

### Why It Failed

Tests failed with "Model failed to activate". The conversion had a bug somewhere - likely in:
- How `await_model` waits for state changes
- The interaction between `set_state()` and the oneshot `activation_tx` channel
- Some race condition in the state machine

The watch changes were reverted. Would need more careful debugging.

## Alternative Fix (Not Implemented)

Smaller change - keep Notify but handle "already activating" case:

```rust
// In send_load_model_command, instead of returning error:
if entry.state == ModelLoadState::Activating {
    let notify = entry.notify.clone();
    drop(entries);  // Release write lock
    notify.notified().await;  // Wait for activation to complete
    let (tx, rx) = oneshot::channel();
    let _ = tx.send(Ok(()));  // Pre-completed
    return Ok(rx);
}
```

**Caveat:** This still has the edge-trigger problem. If activation completes between `drop(entries)` and `notify.notified().await`, the signal is lost and we wait forever.

## Files Involved

- `tests/e2e_basic.rs` - Test fixture with `OnceLock<TestFixture>`
- `src/model/registry.rs` - `ModelRegistry`, `ModelEntry`, `await_model`, `send_load_model_command`
- `src/client/mod.rs` - `Client::connect` captures runtime handle for event callbacks

## To Reproduce

```bash
# This works (~2.3s)
cargo test --test e2e_basic -- --ignored --test-threads=1

# This hangs
cargo test --test e2e_basic -- --ignored
```

## Next Steps

1. **Debug the watch conversion** - figure out why "failed to activate" happens
2. **Or** implement the alternative fix with proper state checking to avoid edge-trigger issues
3. **Or** restructure to use a different synchronization primitive entirely

## Limitation: Test Fixture Not Shared Across Files

The current `TestFixture` in `tests/e2e_basic.rs` only applies to that one file. Rust has no built-in equivalent to pytest's `conftest.py` for sharing fixtures across test files.

**Potential approaches for cross-file fixtures:**

1. **Shared test module** - Create `tests/common/mod.rs` with the fixture, import from each test file. But each file still gets its own `OnceLock` instance unless using process-level synchronization.

2. **Custom test harness** - Replace the default test runner with a custom one that manages shared state. More complex setup.

3. **`ctor` crate** - Global constructors that run before main. Could initialize shared resources, but cleanup is tricky.

4. **Integration test binary** - Single `tests/integration.rs` that contains all e2e tests in one file. Simple but less organized.

5. **Workspace-level test utilities crate** - A separate crate in the workspace that provides test fixtures. Other test files depend on it.

For now, only `e2e_basic.rs` has the fixture. Other test files would need to either copy the pattern or we'd need to implement one of the above approaches.

## Reference: orchard-py Pattern

```python
# conftest.py - pytest fixtures are session-scoped, created once
@pytest.fixture(scope="session")
def engine() -> Generator[InferenceEngine, None, None]:
    engine_instance = InferenceEngine(load_models=MODEL_IDS)
    yield engine_instance
    engine_instance.close()

@pytest.fixture(scope="session")
def client(engine: InferenceEngine) -> Generator[Client, None]:
    client = engine.client()
    yield client
    client.close()
```

Python's `asyncio.Event` in the registry handles concurrent callers correctly because it's level-triggered.
