//! Turn-by-turn long-context shared-prefix benchmark.
//!
//! Run with:
//! PIE_LOCAL_BUILD=/path/to/pie/release \
//!   cargo test --test e2e test_shared_prefix_turn_by_turn_benchmark -- --ignored --nocapture --test-threads=1

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use orchard::{ChatResult, Client, InferenceEngine, ModelRegistry, SamplingParams};

use crate::fixture::QWEN_MODEL_ID;

const DEFAULT_TURNS: usize = 10;
const DEFAULT_CONTEXT_PARAGRAPHS: usize = 24;
const DEFAULT_MAX_TOKENS: i32 = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolMode {
    None,
    Messages,
    Grammar,
}

#[derive(Default)]
struct TurnStats {
    turn: usize,
    messages: usize,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    ttft_ms: f64,
    total_ms: f64,
    assistant_text: String,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn tool_mode_from_env() -> ToolMode {
    match std::env::var("ORCHARD_SHARED_PREFIX_BENCH_TOOL_MODE")
        .unwrap_or_else(|_| "grammar".to_string())
        .as_str()
    {
        "none" => ToolMode::None,
        "messages" => ToolMode::Messages,
        "grammar" => ToolMode::Grammar,
        value => panic!("unsupported ORCHARD_SHARED_PREFIX_BENCH_TOOL_MODE={value}"),
    }
}

fn message(role: &str, content: impl Into<String>) -> HashMap<String, serde_json::Value> {
    let mut msg = HashMap::new();
    msg.insert("role".to_string(), serde_json::json!(role));
    msg.insert("content".to_string(), serde_json::json!(content.into()));
    msg
}

fn tool_message(call_id: &str, content: impl Into<String>) -> HashMap<String, serde_json::Value> {
    let mut msg = message("tool", content);
    msg.insert("tool_call_id".to_string(), serde_json::json!(call_id));
    msg.insert(
        "name".to_string(),
        serde_json::json!("lookup_project_context"),
    );
    msg
}

fn synthetic_context(paragraphs: usize) -> String {
    let mut text = String::new();
    for i in 0..paragraphs {
        text.push_str(&format!(
            "Context shard {i}: Orchard routes local inference through a proxy engine. \
             The transcript benchmark keeps a stable prefix, appends small user turns, \
             occasionally includes tool observations, and expects the engine to reuse \
             sealed KV pages instead of refilling the full history. The important facts \
             are project=Orchard, cache=shared-prefix, mode=incremental-multi-turn, \
             invariant={}. ",
            i % 17
        ));
    }
    text
}

fn benchmark_tools() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "name": "lookup_project_context",
            "description": "Look up a deterministic project-context shard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "turn": {"type": "integer"},
                    "topic": {"type": "string"}
                },
                "required": ["turn", "topic"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "record_benchmark_note",
            "description": "Record a short benchmark note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"}
                },
                "required": ["note"]
            }
        }),
    ]
}

async fn run_turn(
    client: &Client,
    model_id: &str,
    messages: Vec<HashMap<String, serde_json::Value>>,
    params: SamplingParams,
    turn: usize,
) -> TurnStats {
    let started = Instant::now();
    let result = client
        .achat(model_id, messages.clone(), params, true)
        .await
        .unwrap_or_else(|e| panic!("turn {turn} request failed: {e:?}"));

    let ChatResult::Stream(mut stream) = result else {
        panic!("turn {turn} expected streaming result");
    };

    let mut stats = TurnStats {
        turn,
        messages: messages.len(),
        ..Default::default()
    };
    let mut saw_first_token = false;

    while let Some(delta) = stream.recv().await {
        if delta.error.is_some() {
            panic!("turn {turn} failed: {:?}", delta.error);
        }
        if !saw_first_token && !delta.tokens.is_empty() {
            saw_first_token = true;
            stats.ttft_ms = started.elapsed().as_secs_f64() * 1000.0;
        }
        if let Some(content) = &delta.content {
            stats.assistant_text.push_str(content);
        }
        stats.completion_tokens += delta.tokens.len() as u32;
        if let Some(prompt_tokens) = delta.prompt_token_count {
            stats.prompt_tokens = stats.prompt_tokens.max(prompt_tokens);
        }
        if let Some(cached_tokens) = delta.cached_token_count {
            stats.cached_tokens = stats.cached_tokens.max(cached_tokens);
        }
        if delta.is_final_delta {
            break;
        }
    }

    stats.total_ms = started.elapsed().as_secs_f64() * 1000.0;
    stats
}

#[tokio::test]
#[ignore]
async fn test_shared_prefix_turn_by_turn_benchmark() {
    let turns = env_usize("ORCHARD_SHARED_PREFIX_BENCH_TURNS", DEFAULT_TURNS);
    let context_paragraphs = env_usize(
        "ORCHARD_SHARED_PREFIX_BENCH_CONTEXT_PARAGRAPHS",
        DEFAULT_CONTEXT_PARAGRAPHS,
    );
    let max_tokens = env_i32("ORCHARD_SHARED_PREFIX_BENCH_MAX_TOKENS", DEFAULT_MAX_TOKENS);
    let model_id = std::env::var("ORCHARD_SHARED_PREFIX_BENCH_MODEL")
        .unwrap_or_else(|_| QWEN_MODEL_ID.to_string());
    let tool_mode = tool_mode_from_env();
    let persistence_mode = std::env::var("PIE_PROMPT_CACHE_PERSISTENCE").unwrap_or_else(|_| {
        std::env::set_var("PIE_PROMPT_CACHE_PERSISTENCE", "0");
        "0".to_string()
    });

    InferenceEngine::shutdown(Duration::from_secs(30))
        .expect("failed to stop existing engine before benchmark");
    let _engine = InferenceEngine::new().await.expect("failed to start PIE");
    let registry = Arc::new(ModelRegistry::new().expect("failed to create model registry"));
    let client = Client::connect(Arc::clone(&registry))
        .await
        .expect("failed to connect client");
    registry
        .ensure_loaded(&model_id)
        .await
        .unwrap_or_else(|e| panic!("failed to load {model_id}: {e}"));

    let mut transcript = vec![message(
        "system",
        format!(
            "You are running a deterministic benchmark. Reply with one concise sentence. \
             Preserve the turn number and do not elaborate.\n\n{}",
            synthetic_context(context_paragraphs)
        ),
    )];

    let mut all_stats = Vec::with_capacity(turns);
    let tools = benchmark_tools();

    for turn in 1..=turns {
        if tool_mode != ToolMode::None && matches!(turn, 4 | 7 | 10) {
            transcript.push(tool_message(
                &format!("call_shared_prefix_{turn}"),
                format!(
                    "{{\"turn\":{turn},\"result\":\"tool observation for shared prefix benchmark\",\"cache_hint\":\"reuse prior sealed pages\"}}"
                ),
            ));
        }

        transcript.push(message(
            "user",
            format!(
                "Turn {turn}: summarize the current benchmark state in one sentence. \
                 Include marker shared-prefix-turn-{turn}."
            ),
        ));

        let params = SamplingParams {
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            rng_seed: 42 + turn as u64,
            tools: if tool_mode == ToolMode::Grammar {
                tools.clone()
            } else {
                Vec::new()
            },
            tool_choice: (tool_mode == ToolMode::Grammar).then(|| serde_json::json!("auto")),
            max_tool_calls: (tool_mode == ToolMode::Grammar).then_some(1),
            ..Default::default()
        };

        let stats = run_turn(&client, &model_id, transcript.clone(), params, turn).await;
        println!(
            "turn={},messages={},prompt_tokens={},cached_tokens={},completion_tokens={},ttft_ms={:.3},total_ms={:.3}",
            stats.turn,
            stats.messages,
            stats.prompt_tokens,
            stats.cached_tokens,
            stats.completion_tokens,
            stats.ttft_ms,
            stats.total_ms
        );
        let assistant_text = if stats.assistant_text.trim().is_empty() {
            format!("Benchmark response for shared-prefix-turn-{turn}.")
        } else {
            stats.assistant_text.clone()
        };
        transcript.push(message(
            "assistant",
            format!(
                "{}\n\n[benchmark marker: shared-prefix-turn-{turn}; cached_tokens={}]",
                assistant_text, stats.cached_tokens
            ),
        ));
        all_stats.push(stats);
    }

    let total_ms: f64 = all_stats.iter().map(|stats| stats.total_ms).sum();
    let total_prompt_tokens: u32 = all_stats.iter().map(|stats| stats.prompt_tokens).sum();
    let total_cached_tokens: u32 = all_stats.iter().map(|stats| stats.cached_tokens).sum();
    let cache_ratio = if total_prompt_tokens == 0 {
        0.0
    } else {
        total_cached_tokens as f64 / total_prompt_tokens as f64
    };

    println!(
        "shared_prefix_benchmark model={} turns={} context_paragraphs={} max_tokens={} tool_mode={:?} prompt_cache_persistence={}",
        model_id, turns, context_paragraphs, max_tokens, tool_mode, persistence_mode
    );
    println!("turn,messages,prompt_tokens,cached_tokens,completion_tokens,ttft_ms,total_ms");
    for stats in &all_stats {
        println!(
            "{},{},{},{},{},{:.3},{:.3}",
            stats.turn,
            stats.messages,
            stats.prompt_tokens,
            stats.cached_tokens,
            stats.completion_tokens,
            stats.ttft_ms,
            stats.total_ms
        );
    }
    println!(
        "summary,total_ms={:.3},total_prompt_tokens={},total_cached_tokens={},cache_ratio={:.4}",
        total_ms, total_prompt_tokens, total_cached_tokens, cache_ratio
    );

    assert_eq!(all_stats.len(), turns);
    if turns > 1 {
        assert!(
            all_stats
                .iter()
                .skip(1)
                .any(|stats| stats.cached_tokens > 0),
            "expected at least one warm turn to report cached tokens"
        );
    }
}
