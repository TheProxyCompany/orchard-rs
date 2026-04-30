use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use ctor::dtor;
use futures::future::join_all;
use orchard::{Client, InferenceEngine, ModelRegistry};

#[dtor]
fn cleanup_engine() {
    let _ = InferenceEngine::shutdown(Duration::from_secs(30));
}

pub(crate) const LLAMA_MODEL_ID: &str = "meta-llama/Llama-3.1-8B-Instruct";
pub(crate) const GEMMA_MODEL_ID: &str = "google/gemma-3-4b-it";
pub(crate) const GEMMA4_MODEL_ID: &str = "google/gemma-4-E2B-it";
pub(crate) const QWEN_MODEL_ID: &str = "Qwen/Qwen3.5-4B";
pub(crate) const MOONDREAM_MODEL_ID: &str = "moondream/moondream3-preview";

pub(crate) const TEXT_MODELS: &[&str] = &[
    LLAMA_MODEL_ID,
    GEMMA_MODEL_ID,
    GEMMA4_MODEL_ID,
    QWEN_MODEL_ID,
    MOONDREAM_MODEL_ID,
];
pub(crate) const VISION_MODELS: &[&str] = &[GEMMA_MODEL_ID, MOONDREAM_MODEL_ID];
pub(crate) const ALL_MODELS: &[&str] = &[
    LLAMA_MODEL_ID,
    GEMMA_MODEL_ID,
    GEMMA4_MODEL_ID,
    QWEN_MODEL_ID,
    MOONDREAM_MODEL_ID,
];

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

        let preload_results = join_all(ALL_MODELS.iter().map(|&model_id| {
            let registry = Arc::clone(&registry);
            async move {
                registry
                    .ensure_loaded(model_id)
                    .await
                    .map(|_| model_id)
                    .map_err(|e| (model_id, e))
            }
        }))
        .await;

        for result in preload_results {
            if let Err((model_id, error)) = result {
                panic!("Failed to preload model {}: {}", model_id, error);
            }
        }

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
