#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use ctor::dtor;
use futures::future::try_join_all;
use orchard::{
    install_request_attempt_observer, Client, InferenceEngine, ModelRegistry,
    RequestAttemptObserver,
};
use tokio::sync::Notify;

#[dtor]
fn cleanup_engine() {
    // Only clean up an engine this test process actually used. If the fixture
    // never initialized, no test touched an engine -- and without the
    // fixture's namespace setup a shutdown here would target the machine's
    // default engine namespace, killing an engine that belongs to someone
    // else (e.g. Proxy.app's Grand Central engine).
    if FIXTURE.get().is_some() {
        if !shared_owner_enabled() {
            if let Some(start) = VOLLEY_START.get() {
                // Mirrors orchard-py's BUCKSHOT wall print: volley only, engine
                // setup and shutdown excluded, so the harnesses compare directly.
                println!(
                    "BUCKSHOT wall: {:.1}s (volley only)",
                    start.elapsed().as_secs_f64()
                );
            }
        }

        if shared_owner_enabled() {
            if let Some(fixture) = FIXTURE.get() {
                fixture.close_engine();
            }
        } else {
            let _ = InferenceEngine::shutdown(Duration::from_secs(30));
        }
    }
}

pub(crate) const LLAMA_MODEL_ID: &str = "meta-llama/Llama-3.1-8B-Instruct";
pub(crate) const GEMMA4_MODEL_ID: &str = "google/gemma-4-E2B-it";
pub(crate) const QWEN_MODEL_ID: &str = "Qwen/Qwen3.5-4B";
pub(crate) const MOONDREAM_MODEL_ID: &str = "moondream/moondream3-preview";
pub(crate) const TRINITY_MODEL_ID: &str = "mlx-community/Trinity-Mini-4bit";
pub(crate) const LFM2_5_MODEL_ID: &str = "LiquidAI/LFM2.5-8B-A1B";
pub(crate) const OLMO_HYBRID_MODEL_ID: &str = "allenai/Olmo-Hybrid-Instruct-DPO-7B";
pub(crate) const NEMOTRON_H_MODEL_ID: &str = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16";
pub(crate) const GRANITE_MODEL_ID: &str = "mlx-community/granite-4.1-30b-4bit";
pub(crate) const GPT_OSS_MODEL_ID: &str = "openai/gpt-oss-20b";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Thinking {
    Off,
    On,
    Required,
}

impl Thinking {
    pub(crate) fn enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Model {
    pub(crate) template_type: &'static str,
    pub(crate) checkpoint: &'static str,
    pub(crate) thinking: Thinking,
    #[allow(dead_code)]
    pub(crate) vision: bool,
    pub(crate) tools: bool,
}

pub(crate) const MODELS: &[Model] = &[
    Model {
        template_type: "llama3",
        checkpoint: LLAMA_MODEL_ID,
        thinking: Thinking::Off,
        vision: false,
        tools: true,
    },
    Model {
        template_type: "gemma4",
        checkpoint: GEMMA4_MODEL_ID,
        thinking: Thinking::On,
        vision: true,
        tools: true,
    },
    Model {
        template_type: "qwen3_5",
        checkpoint: QWEN_MODEL_ID,
        thinking: Thinking::On,
        vision: false,
        tools: true,
    },
    Model {
        template_type: "moondream3",
        checkpoint: MOONDREAM_MODEL_ID,
        thinking: Thinking::On,
        vision: true,
        tools: false,
    },
    Model {
        template_type: "afmoe",
        checkpoint: TRINITY_MODEL_ID,
        thinking: Thinking::On,
        vision: false,
        tools: true,
    },
    Model {
        template_type: "lfm2_5",
        checkpoint: LFM2_5_MODEL_ID,
        thinking: Thinking::Required,
        vision: false,
        tools: true,
    },
    Model {
        template_type: "olmo_hybrid",
        checkpoint: OLMO_HYBRID_MODEL_ID,
        thinking: Thinking::Off,
        vision: false,
        tools: true,
    },
    Model {
        template_type: "nemotron_h",
        checkpoint: NEMOTRON_H_MODEL_ID,
        thinking: Thinking::On,
        vision: false,
        tools: true,
    },
    Model {
        template_type: "granite_switch",
        checkpoint: GRANITE_MODEL_ID,
        thinking: Thinking::Off,
        vision: false,
        tools: true,
    },
    Model {
        template_type: "gpt_oss",
        checkpoint: GPT_OSS_MODEL_ID,
        thinking: Thinking::On,
        vision: false,
        tools: true,
    },
];

pub(crate) const TEXT_MODELS: &[&str] = &[
    LLAMA_MODEL_ID,
    GEMMA4_MODEL_ID,
    QWEN_MODEL_ID,
    MOONDREAM_MODEL_ID,
    TRINITY_MODEL_ID,
    LFM2_5_MODEL_ID,
    OLMO_HYBRID_MODEL_ID,
    NEMOTRON_H_MODEL_ID,
    GRANITE_MODEL_ID,
    GPT_OSS_MODEL_ID,
];
pub(crate) const VISION_MODELS: &[&str] = &[GEMMA4_MODEL_ID, MOONDREAM_MODEL_ID];
pub(crate) const ALL_MODELS: &[&str] = TEXT_MODELS;

pub(crate) const IDEOGRAM4_MODEL_ID: &str = "ideogram-ai/ideogram-4-fp8";
pub(crate) const FLUX2_MODEL_ID: &str = "black-forest-labs/FLUX.2-klein-4B";
pub(crate) const QWEN_IMAGE_EDIT_MODEL_ID: &str = "Qwen/Qwen-Image-Edit";
pub(crate) const PARAKEET_MODEL_ID: &str = "mlx-community/parakeet-tdt-0.6b-v3";
pub(crate) const QWEN3_TTS_0_6B_MODEL_ID: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";
pub(crate) const QWEN3_TTS_1_7B_MODEL_ID: &str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";
pub(crate) const QWEN3_ASR_0_6B_MODEL_ID: &str = "Qwen/Qwen3-ASR-0.6B";
pub(crate) const QWEN3_ASR_1_7B_MODEL_ID: &str = "Qwen/Qwen3-ASR-1.7B";

/// Modal (diffusion/TTS/STT) checkpoints the golden suite exercises. When
/// [`PRELOAD_MODAL_MODELS`] is set, the fixture hydrates this entire set
/// concurrently so the release suite covers heterogeneous model activation
/// under unified-memory pressure.
pub(crate) const MODAL_MODELS: &[&str] = &[
    IDEOGRAM4_MODEL_ID,
    FLUX2_MODEL_ID,
    QWEN_IMAGE_EDIT_MODEL_ID,
    QWEN3_TTS_0_6B_MODEL_ID,
    QWEN3_TTS_1_7B_MODEL_ID,
    PARAKEET_MODEL_ID,
    QWEN3_ASR_0_6B_MODEL_ID,
    QWEN3_ASR_1_7B_MODEL_ID,
];

/// Set from a `#[ctor]` in targets whose tests hit modal checkpoints
/// (golden, buckshot) before any test runs; functional-only runs skip the
/// multi-minute modal hydration.
pub(crate) static PRELOAD_MODAL_MODELS: AtomicBool = AtomicBool::new(false);

/// Checkpoints whose profile supports tool calling — the orchard-py matrix
/// gates tool cases on the same flag (models.py `tools=`), so suites stay
/// in parity by filtering here instead of looping raw TEXT_MODELS.
pub(crate) fn tool_model_ids() -> impl Iterator<Item = &'static str> {
    MODELS.iter().filter(|m| m.tools).map(|m| m.checkpoint)
}

/// Drive every independent per-model test chain concurrently; each chain's
/// internal turn order is untouched.
pub(crate) async fn fanout<I>(chains: I)
where
    I: IntoIterator,
    I::Item: std::future::Future<Output = ()>,
{
    futures::future::join_all(chains).await;
}

pub(crate) struct TestFixture {
    _runtime: tokio::runtime::Runtime,
    engine: Mutex<Option<InferenceEngine>>,
    pub(crate) client: Client,
    pub(crate) registry: Arc<ModelRegistry>,
}

impl TestFixture {
    fn close_engine(&self) {
        let mut guard = self
            .engine
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(mut engine) = guard.take() {
            if let Err(error) = engine.close() {
                eprintln!("Failed to close shared-owner Orchard engine lease: {error}");
            }
        }
    }
}

static FIXTURE: OnceLock<TestFixture> = OnceLock::new();
static VOLLEY_START: OnceLock<Instant> = OnceLock::new();
static ADMISSION: OnceLock<Mutex<AdmissionState>> = OnceLock::new();
static REQUEST_RELEASED: OnceLock<Notify> = OnceLock::new();

const EXPECTED_RS_PLAN_ENV: &str = "ORCHARD_TEST_EXPECTED_RS_REQUEST_PLAN_JSON";
const PROTOCOL_VERSION: u8 = 3;

pub(crate) fn shared_owner_enabled() -> bool {
    std::env::var("ORCHARD_TEST_SHARED_OWNER").as_deref() == Ok("1")
}

struct AdmissionState {
    expected: BTreeSet<String>,
    request_plan: BTreeMap<String, BTreeMap<String, Vec<usize>>>,
    wire_plan: BTreeMap<String, Vec<usize>>,
    expected_phase_counts: Vec<usize>,
    started: BTreeSet<String>,
    completed: BTreeSet<String>,
    thread_cases: HashMap<ThreadId, String>,
    request_counts: HashMap<(String, String), usize>,
    actual_phase_counts: Vec<usize>,
    request_attempts: usize,
    retries: usize,
    released: bool,
    release_in_progress: bool,
    release_error: Option<String>,
    done_sent: bool,
    session: Option<UnixStream>,
}

pub(crate) struct BuckshotCaseGuard {
    case_id: Option<String>,
    thread_id: Option<ThreadId>,
}

impl BuckshotCaseGuard {
    fn disabled() -> Self {
        Self {
            case_id: None,
            thread_id: None,
        }
    }
}

impl Drop for BuckshotCaseGuard {
    fn drop(&mut self) {
        let Some(case_id) = self.case_id.take() else {
            return;
        };
        let lock = admission_state();
        let mut state = lock.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(thread_id) = self.thread_id.take() {
            let mapped_case = state.thread_cases.remove(&thread_id);
            assert_eq!(mapped_case.as_deref(), Some(case_id.as_str()));
        }
        assert!(
            state.completed.insert(case_id.clone()),
            "Rust Buckshot case completed more than once: {case_id}"
        );
        if state.completed.len() != state.expected.len() || state.done_sent {
            return;
        }

        let completed = state.completed.len();
        let success = state
            .request_counts
            .iter()
            .all(|((case_id, model_id), actual)| {
                state
                    .request_plan
                    .get(case_id)
                    .and_then(|models| models.get(model_id))
                    .is_some_and(|phases| phases.len() == *actual)
            })
            && state.request_plan.iter().all(|(case_id, models)| {
                models.iter().all(|(model_id, phases)| {
                    state
                        .request_counts
                        .get(&(case_id.clone(), model_id.clone()))
                        .copied()
                        .unwrap_or_default()
                        == phases.len()
                })
            });
        let done = serde_json::json!({
            "type": "done",
            "protocol": PROTOCOL_VERSION,
            "role": "orchard-rs",
            "cases_completed": completed,
            "request_attempts": state.request_attempts,
            "phase_counts": state.actual_phase_counts,
            "filtered": 0,
            "skipped": 0,
            "inference_retries": state.retries,
            "success": success,
        });
        let session = state
            .session
            .as_mut()
            .expect("Rust Buckshot completed without a root-owner session");
        write_protocol_frame(session, &done)
            .expect("Failed to send Rust Buckshot completion attestation");
        state.done_sent = true;
    }
}

fn admission_state() -> &'static Mutex<AdmissionState> {
    ADMISSION.get_or_init(|| {
        let encoded = std::env::var(EXPECTED_RS_PLAN_ENV).unwrap_or_else(|_| {
            panic!("{EXPECTED_RS_PLAN_ENV} is required when ORCHARD_TEST_SHARED_OWNER=1")
        });
        let root_plan: serde_json::Value = serde_json::from_str(&encoded)
            .unwrap_or_else(|error| panic!("Invalid {EXPECTED_RS_PLAN_ENV}: {error}"));
        let request_plan = expanded_rust_request_plan();
        let wire_plan = aggregate_request_plan(&request_plan);
        let expected_wire = serde_json::to_value(&wire_plan).expect("Rust plan must serialize");
        assert_eq!(
            root_plan, expected_wire,
            "Rust Buckshot request plan disagrees with the root owner's manifest"
        );
        let expected = request_plan.keys().cloned().collect::<BTreeSet<_>>();
        assert!(!expected.is_empty(), "Rust Buckshot case manifest is empty");
        let expected_phase_counts = sum_phase_counts(wire_plan.values());
        assert!(
            expected_phase_counts.first().copied().unwrap_or_default() > 0,
            "Rust Buckshot request plan has no phase-0 inference requests"
        );
        Mutex::new(AdmissionState {
            expected,
            request_plan,
            wire_plan,
            expected_phase_counts: expected_phase_counts.clone(),
            started: BTreeSet::new(),
            completed: BTreeSet::new(),
            thread_cases: HashMap::new(),
            request_counts: HashMap::new(),
            actual_phase_counts: vec![0; expected_phase_counts.len()],
            request_attempts: 0,
            retries: 0,
            released: false,
            release_in_progress: false,
            release_error: None,
            done_sent: false,
            session: None,
        })
    })
}

fn expanded_rust_request_plan() -> BTreeMap<String, BTreeMap<String, Vec<usize>>> {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../buckshot_request_plan.json"))
            .expect("Invalid tests/buckshot_request_plan.json");
    let groups = manifest["groups"]
        .as_object()
        .expect("Rust Buckshot groups must be an object");
    let cases = manifest["cases"]
        .as_object()
        .expect("Rust Buckshot cases must be an object");
    let mut expanded = BTreeMap::new();
    for (case_id, entries) in cases {
        let entries = entries
            .as_object()
            .unwrap_or_else(|| panic!("Rust Buckshot case {case_id} must be an object"));
        let mut models: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (group_or_model, phases) in entries {
            let phases = phases
                .as_array()
                .unwrap_or_else(|| panic!("Rust Buckshot phases for {case_id} must be an array"))
                .iter()
                .map(|phase| {
                    phase.as_u64().unwrap_or_else(|| {
                        panic!("Rust Buckshot phase for {case_id} must be non-negative")
                    }) as usize
                })
                .collect::<Vec<_>>();
            let model_ids = groups.get(group_or_model).map_or_else(
                || vec![group_or_model.clone()],
                |group| {
                    group
                        .as_array()
                        .unwrap_or_else(|| {
                            panic!("Rust Buckshot group {group_or_model} must be an array")
                        })
                        .iter()
                        .map(|model| {
                            model
                                .as_str()
                                .unwrap_or_else(|| {
                                    panic!("Rust Buckshot model IDs must be strings")
                                })
                                .to_owned()
                        })
                        .collect()
                },
            );
            for model_id in model_ids {
                assert!(
                    models.insert(model_id.clone(), phases.clone()).is_none(),
                    "Rust Buckshot case {case_id} declares {model_id} more than once"
                );
            }
        }
        expanded.insert(case_id.clone(), models);
    }
    expanded
}

fn aggregate_request_plan(
    plan: &BTreeMap<String, BTreeMap<String, Vec<usize>>>,
) -> BTreeMap<String, Vec<usize>> {
    plan.iter()
        .map(|(case_id, models)| {
            let width = models
                .values()
                .flat_map(|phases| phases.iter())
                .max()
                .map_or(0, |phase| phase + 1);
            let mut counts = vec![0; width];
            for phase in models.values().flatten() {
                counts[*phase] += 1;
            }
            (case_id.clone(), counts)
        })
        .collect()
}

fn sum_phase_counts<'a>(plans: impl Iterator<Item = &'a Vec<usize>>) -> Vec<usize> {
    let plans = plans.collect::<Vec<_>>();
    let width = plans
        .iter()
        .map(|phases| phases.len())
        .max()
        .unwrap_or_default();
    (0..width)
        .map(|phase| {
            plans
                .iter()
                .map(|counts| counts.get(phase).copied().unwrap_or_default())
                .sum()
        })
        .collect()
}

fn normalize_case_id(case_id: &str) -> String {
    case_id
        .split_once("::")
        .map_or_else(|| case_id.to_owned(), |(_, test_id)| test_id.to_owned())
}

fn write_protocol_frame(stream: &mut UnixStream, frame: &serde_json::Value) -> std::io::Result<()> {
    serde_json::to_writer(&mut *stream, frame)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    stream.write_all(b"\n")
}

fn buckshot_model_ids() -> Vec<&'static str> {
    TEXT_MODELS
        .iter()
        .chain(MODAL_MODELS.iter())
        .copied()
        .collect()
}

pub(crate) fn admit_buckshot_case(case_id: &str) -> BuckshotCaseGuard {
    if !shared_owner_enabled() {
        return BuckshotCaseGuard::disabled();
    }

    let case_id = normalize_case_id(case_id);
    // `#[tokio::test]` uses a current-thread runtime by default. Its root
    // future is not a spawned Tokio task, so `tokio::task::try_id()` is None.
    // Libtest's worker thread is the real case owner, and inline fanout futures
    // remain on that thread for the lifetime of this guard.
    let thread_id = std::thread::current().id();
    let release_owner = {
        let lock = admission_state();
        let mut state = lock.lock().unwrap_or_else(|error| error.into_inner());
        assert!(
            state.expected.contains(&case_id),
            "Unmanifested Rust Buckshot case: {case_id}"
        );
        assert!(
            state.started.insert(case_id.clone()),
            "Duplicate Rust Buckshot case admission: {case_id}"
        );
        let previous = state.thread_cases.insert(thread_id, case_id.clone());
        assert!(
            previous.is_none(),
            "Rust Buckshot worker admitted more than one case"
        );
        let release_owner = state.started == state.expected && !state.release_in_progress;
        if release_owner {
            state.release_in_progress = true;
        }
        release_owner
    };

    if release_owner {
        let (session, release_error) = match announce_rust_ready_and_wait() {
            Ok(session) => (Some(session), None),
            Err(error) => (None, Some(error)),
        };
        let released = release_error.is_none();
        {
            let lock = admission_state();
            let mut state = lock.lock().unwrap_or_else(|error| error.into_inner());
            state.session = session;
            state.release_error = release_error;
            state.released = true;
        }
        if released {
            let _ = VOLLEY_START.set(Instant::now());
        }
        REQUEST_RELEASED.get_or_init(Notify::new).notify_waiters();
    }

    BuckshotCaseGuard {
        case_id: Some(case_id),
        thread_id: Some(thread_id),
    }
}

pub(crate) async fn admit_buckshot_case_async(case_id: &'static str) -> BuckshotCaseGuard {
    if !shared_owner_enabled() {
        return BuckshotCaseGuard::disabled();
    }
    let _ = get_fixture().await;
    let guard = admit_buckshot_case(case_id);
    wait_for_buckshot_release()
        .await
        .unwrap_or_else(|error| panic!("Rust Buckshot case release failed: {error}"));
    guard
}

async fn wait_for_buckshot_release() -> Result<(), String> {
    loop {
        let notified = REQUEST_RELEASED.get_or_init(Notify::new).notified();
        let release = {
            let state = admission_state()
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.released.then(|| state.release_error.clone())
        };
        if let Some(release_error) = release {
            return release_error.map_or(Ok(()), Err);
        }
        notified.await;
    }
}

async fn observe_buckshot_request(model_id: String) -> Result<(), String> {
    let thread_id = std::thread::current().id();
    {
        let lock = admission_state();
        let mut state = lock.lock().unwrap_or_else(|error| error.into_inner());
        if !state.released {
            return Err(format!(
                "Rust Buckshot request started before case release: {model_id}"
            ));
        }
        if let Some(error) = state.release_error.as_ref() {
            return Err(format!("Rust Buckshot case release failed: {error}"));
        }
        let case_id = state.thread_cases.get(&thread_id).cloned().ok_or_else(|| {
            format!("Rust Buckshot request has no case on worker {thread_id:?}: {model_id}")
        })?;
        let key = (case_id.clone(), model_id.clone());
        let ordinal = state.request_counts.get(&key).copied().unwrap_or_default();
        let phase = state
            .request_plan
            .get(&case_id)
            .and_then(|models| models.get(&model_id))
            .and_then(|phases| phases.get(ordinal))
            .copied();
        let Some(phase) = phase else {
            state.request_attempts += 1;
            state.retries += 1;
            return Err(format!(
                "Unplanned Rust Buckshot request attempt for {case_id} model={model_id} ordinal={ordinal}"
            ));
        };
        state.request_attempts += 1;
        state.request_counts.insert(key, ordinal + 1);
        state.actual_phase_counts[phase] += 1;
    }
    Ok(())
}

fn announce_rust_ready_and_wait() -> Result<UnixStream, String> {
    assert!(
        PRELOAD_MODAL_MODELS.load(Ordering::Relaxed),
        "Combined Buckshot did not enable the modal model population"
    );
    let barrier_path = std::env::var("ORCHARD_TEST_BARRIER_SOCKET")
        .map_err(|_| "ORCHARD_TEST_BARRIER_SOCKET is required".to_string())?;
    let (cases, request_plan, phase_counts, phase0_requests, request_attempts, retries) = {
        let state = admission_state()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        assert_eq!(
            state.started, state.expected,
            "Rust Buckshot announced before every case was admitted"
        );
        assert_eq!(
            state.request_attempts, 0,
            "Rust Buckshot executed requests before case release"
        );
        (
            state.started.iter().cloned().collect::<Vec<_>>(),
            state.wire_plan.clone(),
            state.expected_phase_counts.clone(),
            state.expected_phase_counts[0],
            state.request_attempts,
            state.retries,
        )
    };
    let mut session = UnixStream::connect(&barrier_path)
        .map_err(|error| format!("Failed to connect to Buckshot barrier: {error}"))?;
    let cases_admitted = cases.len();
    let ready = serde_json::json!({
        "type": "ready",
        "protocol": PROTOCOL_VERSION,
        "role": "orchard-rs",
        "models": buckshot_model_ids(),
        "cases": cases,
        "cases_admitted": cases_admitted,
        "request_plan": request_plan,
        "phase_counts": phase_counts,
        "phase0_requests": phase0_requests,
        "request_attempts": request_attempts,
        "filtered": 0,
        "skipped": 0,
        "inference_retries": retries,
    });
    write_protocol_frame(&mut session, &ready)
        .map_err(|error| format!("Failed to send Rust Buckshot readiness: {error}"))?;
    let mut release = [0_u8; 1];
    session
        .read_exact(&mut release)
        .map_err(|error| format!("Combined Buckshot owner closed before release: {error}"))?;
    if release != *b"G" {
        return Err("Combined Buckshot sent an invalid release".to_string());
    }
    Ok(session)
}

/// Tests must never operate in the default engine namespace: that namespace
/// belongs to whatever long-lived engine this machine runs (Proxy.app's
/// Grand Central engine in production), and test-side engine shutdowns
/// force-stop the namespace's engine. Unless the caller pinned a namespace
/// explicitly (pie_cycle.sh exports ORCHARD_CACHE_ROOT), give this test
/// process a private one before anything resolves engine paths or IPC
/// endpoints from the environment.
pub(crate) fn ensure_test_namespace() {
    if std::env::var_os("ORCHARD_CACHE_ROOT").is_none() {
        // Keep the name short: the engine listens on a unix socket at
        // <namespace>/ipc/pie_requests.ipc, and sockaddr_un caps the whole
        // path at 104 bytes on macOS. The temp dir alone is ~50 bytes; the
        // previous orchard-rs-test-<pid>-<16-hex-nanos> name pushed the
        // socket path over the cap and every engine boot died with
        // "nng_listen ... Address invalid" (the whole rs suite then times
        // out waiting for a heartbeat that can never come). Pid plus the
        // low 32 bits of the boot nanos keeps it unique per test process.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let namespace = std::env::temp_dir().join(format!(
            "orc-{}-{:x}",
            std::process::id(),
            (nanos & 0xFFFF_FFFF) as u32
        ));
        std::fs::create_dir_all(&namespace).expect("Failed to create test engine namespace");
        std::env::set_var("ORCHARD_CACHE_ROOT", &namespace);
    }
}

fn init_fixture() -> TestFixture {
    ensure_test_namespace();

    if shared_owner_enabled() {
        let observer: RequestAttemptObserver =
            Arc::new(|model_id| Box::pin(observe_buckshot_request(model_id)));
        install_request_attempt_observer(observer)
            .expect("Failed to install the Rust Buckshot request observer");
    }

    if !shared_owner_enabled() {
        if let Err(e) = InferenceEngine::shutdown(Duration::from_secs(30)) {
            panic!(
                "Failed to stop existing engine before starting tests: {}",
                e
            );
        }
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Failed to create runtime");

    let (engine, client, registry) = rt.block_on(async {
        let engine = InferenceEngine::new()
            .await
            .expect("Failed to start engine");
        let registry = Arc::new(ModelRegistry::new().unwrap());
        let client = Client::connect(Arc::clone(&registry))
            .await
            .expect("Failed to connect");
        let mut preload_models = Vec::from(ALL_MODELS);
        if PRELOAD_MODAL_MODELS.load(Ordering::Relaxed) {
            preload_models.extend_from_slice(MODAL_MODELS);
        }
        try_join_all(
            preload_models
                .into_iter()
                .map(|model_id| registry.ensure_loaded(model_id)),
        )
        .await
        .expect("Failed to attach test models");

        (engine, client, registry)
    });

    if !shared_owner_enabled() {
        let _ = VOLLEY_START.set(Instant::now());
    }

    TestFixture {
        _runtime: rt,
        engine: Mutex::new(Some(engine)),
        client,
        registry,
    }
}

pub(crate) async fn get_fixture() -> &'static TestFixture {
    tokio::task::spawn_blocking(|| FIXTURE.get_or_init(init_fixture))
        .await
        .expect("spawn_blocking failed")
}

pub(crate) fn make_message(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));
    msg.insert("content".to_string(), serde_json::json!(content));
    msg
}
