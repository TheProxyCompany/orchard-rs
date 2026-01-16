//! Minimal reproduction test for concurrency bug.
//!
//! Simulates cargo's parallel test threads using std::thread with separate tokio runtimes.
//! This reproduces the bug where parallel execution produces different outputs than sequential.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use base64::Engine;
use orchard::{Client, InferenceEngine, ModelRegistry, SamplingParams};

const MODEL_ID: &str = "moondream3";

fn get_test_assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/assets")
}

fn load_image_base64(filename: &str) -> String {
    let path = get_test_assets_dir().join(filename);
    let bytes = std::fs::read(&path).unwrap();
    base64::engine::general_purpose::STANDARD.encode(&bytes)
}

fn make_image_message(role: &str, text: &str, image_base64: &str) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));
    let content = serde_json::json!([
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{}", image_base64)}}
    ]);
    msg.insert("content".to_string(), content);
    msg
}

fn make_text_message(role: &str, content: &str) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));
    msg.insert("content".to_string(), serde_json::json!(content));
    msg
}

fn run_moondream(registry: Arc<ModelRegistry>) -> String {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let client = Client::connect(registry).await.unwrap();
        let params = SamplingParams {
            max_tokens: 50,
            temperature: 0.0,
            ..Default::default()
        };
        let image_base64 = load_image_base64("moondream.jpg");
        let messages = vec![make_image_message("user", "What is the girl doing?", &image_base64)];
        match client.achat(MODEL_ID, messages, params, false).await.unwrap() {
            orchard::ChatResult::Complete(r) => r.text,
            _ => panic!("Expected complete"),
        }
    })
}

fn run_apple(registry: Arc<ModelRegistry>) -> String {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let client = Client::connect(registry).await.unwrap();
        let params = SamplingParams {
            max_tokens: 50,
            temperature: 0.0,
            ..Default::default()
        };
        let image_base64 = load_image_base64("apple.jpg");
        let messages = vec![make_image_message("user", "What is in this image?", &image_base64)];
        match client.achat(MODEL_ID, messages, params, false).await.unwrap() {
            orchard::ChatResult::Complete(r) => r.text,
            _ => panic!("Expected complete"),
        }
    })
}

fn run_text_question(registry: Arc<ModelRegistry>) -> String {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let client = Client::connect(registry).await.unwrap();
        let params = SamplingParams {
            max_tokens: 20,
            temperature: 0.0,
            ..Default::default()
        };
        let messages = vec![make_text_message("user", "What is the capital of France?")];
        match client.achat(MODEL_ID, messages, params, false).await.unwrap() {
            orchard::ChatResult::Complete(r) => r.text,
            _ => panic!("Expected complete"),
        }
    })
}

#[test]
#[ignore]
fn test_concurrency_bug() {
    // Start engine
    let _ = InferenceEngine::shutdown(Duration::from_secs(10));
    let rt = tokio::runtime::Runtime::new().unwrap();
    let registry = rt.block_on(async {
        let _engine = InferenceEngine::new().await.unwrap();
        let registry = Arc::new(ModelRegistry::new().unwrap());
        let _client = Client::connect(Arc::clone(&registry)).await.unwrap();
        // Keep engine alive
        std::mem::forget(_engine);
        registry.ensure_loaded(MODEL_ID).await.unwrap();
        registry
    });

    // Sequential
    let seq_moondream = run_moondream(Arc::clone(&registry));
    let seq_apple = run_apple(Arc::clone(&registry));
    let seq_text = run_text_question(Arc::clone(&registry));

    // Parallel - separate OS threads with separate runtimes (like cargo)
    let reg1 = Arc::clone(&registry);
    let reg2 = Arc::clone(&registry);
    let reg3 = Arc::clone(&registry);

    let handle1 = thread::spawn(move || run_moondream(reg1));
    let handle2 = thread::spawn(move || run_apple(reg2));
    let handle3 = thread::spawn(move || run_text_question(reg3));

    let par_moondream = handle1.join().unwrap();
    let par_apple = handle2.join().unwrap();
    let par_text = handle3.join().unwrap();

    // Print results
    println!("\n=== MOONDREAM ===");
    println!("Sequential: {}", seq_moondream);
    println!("Parallel:   {}", par_moondream);

    println!("\n=== APPLE ===");
    println!("Sequential: {}", seq_apple);
    println!("Parallel:   {}", par_apple);

    println!("\n=== TEXT ===");
    println!("Sequential: {}", seq_text);
    println!("Parallel:   {}", par_text);

    let _ = InferenceEngine::shutdown(Duration::from_secs(10));

    // Check for differences
    let moondream_match = seq_moondream == par_moondream;
    let apple_match = seq_apple == par_apple;
    let text_match = seq_text == par_text;

    println!("\n=== RESULTS ===");
    println!("Moondream match: {}", moondream_match);
    println!("Apple match:     {}", apple_match);
    println!("Text match:      {}", text_match);

    if !moondream_match || !apple_match || !text_match {
        panic!("Concurrency bug reproduced: parallel outputs differ from sequential");
    }
}
