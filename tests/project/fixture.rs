#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use ctor::dtor;
use futures::future::try_join_all;
use orchard::{Client, InferenceEngine, ModelRegistry};

#[dtor]
fn cleanup_engine() {
    // Only clean up an engine this test process actually used. If the fixture
    // never initialized, no test touched an engine -- and without the
    // fixture's namespace setup a shutdown here would target the machine's
    // default engine namespace, killing an engine that belongs to someone
    // else (e.g. Proxy.app's Grand Central engine).
    if FIXTURE.get().is_some() {
        if let Some(start) = VOLLEY_START.get() {
            // Mirrors orchard-py's BUCKSHOT wall print: volley only, engine
            // setup and shutdown excluded, so the harnesses compare directly.
            println!(
                "BUCKSHOT wall: {:.1}s (volley only)",
                start.elapsed().as_secs_f64()
            );
        }
        let _ = InferenceEngine::shutdown(Duration::from_secs(30));
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

/// Modal (diffusion/TTS/STT) checkpoints the golden suite exercises.
/// Hydrated serially at fixture init when [`PRELOAD_MODAL_MODELS`] is set:
/// activating them lazily mid-volley — while chat decode is in flight —
/// spikes wired memory, and concurrent hydration of several at once is the
/// documented engine killer (same constraint orchard-py's buckshot preload
/// obeys). Serial, up-front, on an idle GPU.
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

/// Cap on concurrently in-flight chains a single test fans out across models.
/// Uncapped by default: the old cap of 4 guarded against an IPC livelock
/// (RequestPreprocessor/ResponseProcessor spinning with committed==completed,
/// pt1 evidence: /tmp/carbon-prod/trackc_fmod2_wedge_sample.txt) that
/// predates engine pacing — orchard-py now fires ~587 concurrent requests in
/// one volley and the engine batches up to 64 sequences. ORCHARD_TEST_FANOUT
/// restores a cap as fallback if a burst regression reappears.
pub(crate) fn test_fanout() -> usize {
    std::env::var("ORCHARD_TEST_FANOUT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(usize::MAX)
}

/// Drive independent per-model test chains concurrently, at most
/// [`test_fanout`] in flight; each chain's internal turn order is untouched.
pub(crate) async fn fanout<I>(chains: I)
where
    I: IntoIterator,
    I::Item: std::future::Future<Output = ()>,
{
    use futures::stream::StreamExt;
    futures::stream::iter(chains)
        .buffer_unordered(test_fanout())
        .for_each(|()| async {})
        .await;
}

pub(crate) struct TestFixture {
    _runtime: tokio::runtime::Runtime,
    _engine: InferenceEngine,
    pub(crate) client: Client,
    pub(crate) registry: Arc<ModelRegistry>,
}

static FIXTURE: OnceLock<TestFixture> = OnceLock::new();
static VOLLEY_START: OnceLock<Instant> = OnceLock::new();

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

    if let Err(e) = InferenceEngine::shutdown(Duration::from_secs(30)) {
        panic!(
            "Failed to stop existing engine before starting tests: {}",
            e
        );
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
        try_join_all(
            ALL_MODELS
                .iter()
                .map(|model_id| registry.ensure_loaded(model_id)),
        )
        .await
        .expect("Failed to preload test models");

        if PRELOAD_MODAL_MODELS.load(Ordering::Relaxed) {
            for &model_id in MODAL_MODELS {
                registry
                    .ensure_loaded(model_id)
                    .await
                    .unwrap_or_else(|err| {
                        panic!("Failed to preload modal model {model_id}: {err:?}")
                    });
            }
        }

        (engine, client, registry)
    });

    let _ = VOLLEY_START.set(Instant::now());

    TestFixture {
        _runtime: rt,
        _engine: engine,
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
