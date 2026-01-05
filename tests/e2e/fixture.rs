use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Duration;

use orchard::{Client, InferenceEngine, ModelRegistry};

const PRELOAD_MODELS: [&str; 2] = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "moondream3",
];

pub(crate) struct TestFixture {
    _runtime: tokio::runtime::Runtime,
    _engine: InferenceEngine,
    pub(crate) client: Client,
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

    let (engine, client) = rt.block_on(async {
        let engine = InferenceEngine::new().await.expect("Failed to start engine");
        let registry = std::sync::Arc::new(ModelRegistry::new().unwrap());
        let client = Client::connect(std::sync::Arc::clone(&registry))
            .await
            .expect("Failed to connect");

        for model_id in PRELOAD_MODELS {
            registry
                .ensure_loaded(model_id)
                .await
                .unwrap_or_else(|e| panic!("Failed to preload model {}: {}", model_id, e));
        }

        (engine, client)
    });

    TestFixture {
        _runtime: rt,
        _engine: engine,
        client,
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
