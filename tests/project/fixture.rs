#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use ctor::dtor;
use orchard::{Client, InferenceEngine, ModelRegistry};

#[dtor]
fn cleanup_engine() {
    let _ = InferenceEngine::shutdown(Duration::from_secs(30));
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

pub(crate) struct TestFixture {
    _runtime: tokio::runtime::Runtime,
    _engine: InferenceEngine,
    pub(crate) client: Client,
    pub(crate) registry: Arc<ModelRegistry>,
}

static FIXTURE: OnceLock<TestFixture> = OnceLock::new();

fn init_fixture() -> TestFixture {
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

        (engine, client, registry)
    });

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
