//! TRACK B equivalence driver: replays the orchard-py buckshot chat volley's
//! load shape through orchard-rs — N concurrent streaming chat requests fired
//! all at once, round-robin across the full 10-model matrix — and checks every
//! accumulated response for cross-stream contamination.
//!
//! Contamination oracles, strongest first:
//! 1. foreign code: every prompt asks the model to echo a unique per-request
//!    code (VLY-0042); any other request's code in this response's text is
//!    hard proof of cross-stream leakage.
//! 2. foreign template marker: control-token spellings unique to one model
//!    family (verified against profiles/*/chat_template.jinja and
//!    control_tokens.json) appearing in another family's output.
//! 3. doubled-word ratio: fraction of adjacent word pairs that are identical;
//!    a zip-merge of two near-identical streams pushes this toward 0.5. Soft
//!    signal only — reported, never asserted.
//!
//! Every request is a datapoint appended as one JSON line to `VOLLEY_OUT`;
//! failures are recorded, never panicked on.
//!
//! Run alone, never alongside the rest of the suite:
//!
//! ```text
//! PIE_LOCAL_BUILD=$PWD/../proxy-inference-engine/release \
//! VOLLEY_STEPS=64,128,256,512 \
//! cargo test --test volley_stress -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Knobs (all env, all optional):
//! - `VOLLEY_STEPS`: comma-separated concurrency rungs (default 64,128,256,512)
//! - `VOLLEY_MAX_TOKENS`: max_output_tokens per request (default 64)
//! - `VOLLEY_TIMEOUT_SECS`: per-request wedge backstop (default 480)
//! - `VOLLEY_SETTLE_SECS`: idle gap between rungs (default 15)
//! - `VOLLEY_STAGGER_MS`: per-slot submission delay (default 0 = one burst,
//!   the py-volley shape; suite-style pacing is ~100-500)
//! - `VOLLEY_OUT`: results JSONL path (default /tmp/volley-stress/results.jsonl)

#[path = "project/fixture.rs"]
mod fixture;

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use orchard::{
    ReasoningConfig, ResponseEvent, ResponseInputItem, ResponsesInput, ResponsesRequest,
    ResponsesResult,
};
use serde_json::{json, Value};

use crate::fixture::{get_fixture, Model, MODELS};

const BASE_PROMPTS: [&str; 12] = [
    "Give me one sentence about the ocean.",
    "Name three prime numbers and add them together.",
    "Summarize the plot of a heist movie in two sentences.",
    "What is a good name for a gray cat? Answer briefly.",
    "Explain what a hash map is in one sentence.",
    "Write a haiku about winter mornings.",
    "List two uses for a paperclip.",
    "What color is the sky at noon on a clear day? Answer briefly.",
    "Give one tip for remembering names.",
    "Describe a lighthouse in one sentence.",
    "What is seven times eight? Answer briefly.",
    "Name a fruit that is yellow. Answer briefly.",
];

/// Template markers unique to one family across the 10-model matrix, verified
/// against profiles/: llama3 (control_tokens.json + capabilities.yaml
/// python_tag), gemma4 (chat_template.jinja open-form tokens), the shared
/// ChatML family (qwen3_5/afmoe/lfm2_5/olmo_hybrid/nemotron_h all use
/// im_start), moondream3 (md_reserved), granite_switch (start_of_role),
/// gpt_oss (harmony channel/message/return). Shared spellings like
/// <|endoftext|> and <|begin_of_text|> are deliberately excluded.
/// Note <|channel> (gemma4, open form) and <|channel|> (gpt_oss) are distinct
/// strings; neither is a substring of the other.
const MARKER_FAMILIES: &[(&str, &[&str])] = &[
    (
        "llama3",
        &[
            "<|start_header_id|>",
            "<|eot_id|>",
            "<|eom_id|>",
            "<|python_tag|>",
        ],
    ),
    (
        "gemma4",
        &[
            "<|turn>",
            "<|think|>",
            "<|tool_call>",
            "<|channel>",
            "<|\"|>",
        ],
    ),
    ("chatml", &["<|im_start|>", "<|im_end|>"]),
    ("moondream3", &["<|md_reserved_"]),
    ("granite_switch", &["<|start_of_role|>", "<|end_of_role|>"]),
    ("gpt_oss", &["<|channel|>", "<|message|>", "<|return|>"]),
];

fn marker_family(template_type: &str) -> &'static str {
    match template_type {
        "qwen3_5" | "afmoe" | "lfm2_5" | "olmo_hybrid" | "nemotron_h" => "chatml",
        "llama3" => "llama3",
        "gemma4" => "gemma4",
        "moondream3" => "moondream3",
        "granite_switch" => "granite_switch",
        "gpt_oss" => "gpt_oss",
        other => panic!("unknown template_type {other}"),
    }
}

fn foreign_markers(template_type: &str, text: &str) -> Vec<String> {
    let own = marker_family(template_type);
    let mut found = Vec::new();
    for (family, markers) in MARKER_FAMILIES {
        if *family == own {
            continue;
        }
        for marker in *markers {
            if text.contains(marker) {
                found.push(format!("{family}:{marker}"));
            }
        }
    }
    found
}

/// Fraction of adjacent word pairs that are identical. A zip-merge of two
/// streams generating similar text doubles most words ("TheThe user user"),
/// pushing this toward 0.5; honest generations sit near 0.
fn doubled_word_ratio(text: &str) -> (usize, f64) {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 2 {
        return (words.len(), 0.0);
    }
    let doubled = words.windows(2).filter(|pair| pair[0] == pair[1]).count();
    (words.len(), doubled as f64 / (words.len() - 1) as f64)
}

struct Config {
    steps: Vec<usize>,
    max_tokens: i32,
    timeout: Duration,
    settle: Duration,
    stagger: Duration,
    out_path: PathBuf,
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}

impl Config {
    fn from_env() -> Self {
        let steps = std::env::var("VOLLEY_STEPS")
            .unwrap_or_else(|_| "64,128,256,512".to_string())
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(|entry| {
                entry
                    .parse::<usize>()
                    .expect("VOLLEY_STEPS entries are usize")
            })
            .collect::<Vec<_>>();
        assert!(!steps.is_empty(), "VOLLEY_STEPS resolved to no rungs");
        Self {
            steps,
            max_tokens: env_parse("VOLLEY_MAX_TOKENS", 64i32),
            timeout: Duration::from_secs(env_parse("VOLLEY_TIMEOUT_SECS", 480u64)),
            settle: Duration::from_secs(env_parse("VOLLEY_SETTLE_SECS", 15u64)),
            stagger: Duration::from_millis(env_parse("VOLLEY_STAGGER_MS", 0u64)),
            out_path: PathBuf::from(
                std::env::var("VOLLEY_OUT")
                    .unwrap_or_else(|_| "/tmp/volley-stress/results.jsonl".to_string()),
            ),
        }
    }
}

fn request_for(model: Model, prompt: String, max_tokens: i32) -> ResponsesRequest {
    ResponsesRequest {
        input: ResponsesInput::Items(vec![ResponseInputItem::Message {
            role: "user".to_string(),
            content: json!(prompt),
            tool_calls: None,
            tool_call_id: None,
        }]),
        stream: true,
        instructions: None,
        temperature: None,
        top_p: None,
        top_k: None,
        min_p: None,
        deterministic: true,
        frequency_penalty: None,
        presence_penalty: None,
        max_output_tokens: Some(max_tokens),
        top_logprobs: None,
        core_tools: Vec::new(),
        active_tools: Vec::new(),
        tool_choice: None,
        min_tool_calls: None,
        max_tool_calls: None,
        text: None,
        reasoning: model.thinking.enabled().then(|| ReasoningConfig::Object {
            effort: "medium".to_string(),
        }),
        reasoning_effort: None,
        metadata: None,
        parallel_tool_calls: false,
        // Engine default (shared prefix cache on), matching the orchard-py
        // volley's cases.
        prefix_cache: None,
        stream_tokens: true,
    }
}

#[derive(Default)]
struct Capture {
    generated: String,
    content: String,
    reasoning: String,
    events: usize,
    completed: bool,
    error: Option<String>,
}

async fn drain(mut events: tokio::sync::mpsc::Receiver<ResponseEvent>) -> Capture {
    let mut capture = Capture::default();
    while let Some(event) = events.recv().await {
        capture.events += 1;
        match &event {
            ResponseEvent::OutputToken(token) => {
                if let Some(content) = &token.content {
                    capture.generated.push_str(content);
                }
            }
            ResponseEvent::OutputTextDelta(delta) => {
                capture.content.push_str(&delta.delta);
            }
            ResponseEvent::ReasoningDelta(delta) => {
                capture.reasoning.push_str(&delta.delta);
            }
            ResponseEvent::ResponseCompleted(_) => {
                capture.completed = true;
            }
            ResponseEvent::Error(error) => {
                capture.error = Some(error.error.message.clone());
            }
            ResponseEvent::Done => break,
            _ => {}
        }
    }
    capture
}

struct WorkerResult {
    index: usize,
    model: Model,
    code: String,
    request_id: Option<u64>,
    ms: u64,
    capture: Capture,
}

async fn run_worker(
    index: usize,
    model: Model,
    code: String,
    max_tokens: i32,
    timeout: Duration,
    delay: Duration,
) -> WorkerResult {
    if !delay.is_zero() {
        tokio::time::sleep(delay).await;
    }
    let fixture = get_fixture().await;
    let prompt = format!(
        "{} End your reply with the exact code {}.",
        BASE_PROMPTS[index % BASE_PROMPTS.len()],
        code
    );
    let request = request_for(model, prompt, max_tokens);
    let started = Instant::now();
    let mut request_id = None;
    let capture = match tokio::time::timeout(timeout, async {
        match fixture.client.aresponses(model.checkpoint, request).await {
            Ok(ResponsesResult::Stream {
                request_id: id,
                events,
            }) => (Some(id), drain(events).await),
            Ok(_) => (
                None,
                Capture {
                    error: Some("expected stream result".to_string()),
                    ..Capture::default()
                },
            ),
            Err(err) => (
                None,
                Capture {
                    error: Some(format!("aresponses failed: {err:?}")),
                    ..Capture::default()
                },
            ),
        }
    })
    .await
    {
        Ok((id, capture)) => {
            request_id = id;
            capture
        }
        Err(_) => Capture {
            error: Some(format!("timeout after {timeout:?}")),
            ..Capture::default()
        },
    };
    WorkerResult {
        index,
        model,
        code,
        request_id,
        ms: started.elapsed().as_millis() as u64,
        capture,
    }
}

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

fn append_line(path: &Path, record: &Value) {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|err| panic!("open {}: {err}", path.display()));
    writeln!(file, "{record}").expect("append volley record");
}

fn truncated(text: &str, limit: usize) -> String {
    text.chars().take(limit).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "manual buckshot-equivalence stress driver; run alone with --ignored"]
async fn volley_stress_chat_matrix() {
    let config = Config::from_env();
    if let Some(parent) = config.out_path.parent() {
        fs::create_dir_all(parent).expect("create VOLLEY_OUT directory");
    }

    let boot_started = Instant::now();
    let _fixture = get_fixture().await;
    let boot_ms = boot_started.elapsed().as_millis() as u64;

    let header = json!({
        "kind": "run_header",
        "started_at": epoch_secs(),
        "pid": std::process::id(),
        "boot_ms": boot_ms,
        "steps": config.steps,
        "max_tokens": config.max_tokens,
        "timeout_secs": config.timeout.as_secs(),
        "stagger_ms": config.stagger.as_millis() as u64,
        "models": MODELS.iter().map(|m| m.template_type).collect::<Vec<_>>(),
        "pie_local_build": std::env::var("PIE_LOCAL_BUILD").ok(),
    });
    append_line(&config.out_path, &header);
    println!("{header}");

    // Codes are globally unique across rungs so late-arriving leakage from a
    // previous rung is still attributable.
    let mut next_code_index = 0usize;
    let mut all_codes: Vec<String> = Vec::new();

    for &n in &config.steps {
        let rung_started = Instant::now();
        let mut handles = Vec::with_capacity(n);
        for slot in 0..n {
            let model = MODELS[slot % MODELS.len()];
            let code = format!("VLY-{next_code_index:04}");
            next_code_index += 1;
            all_codes.push(code.clone());
            handles.push(tokio::spawn(run_worker(
                slot,
                model,
                code,
                config.max_tokens,
                config.timeout,
                config.stagger * slot as u32,
            )));
        }

        let mut results = Vec::with_capacity(n);
        for handle in handles {
            results.push(handle.await.expect("worker task panicked"));
        }
        let wall_ms = rung_started.elapsed().as_millis() as u64;

        let mut completed = 0usize;
        let mut errored = 0usize;
        let mut dead_errors = 0usize;
        let mut flagged_codes = 0usize;
        let mut flagged_markers = 0usize;
        let mut flagged_dup = 0usize;
        let mut max_ms = 0u64;
        let mut sum_ms = 0u64;

        for result in &results {
            let combined = format!(
                "{}\n{}\n{}",
                result.capture.generated, result.capture.content, result.capture.reasoning
            );
            let foreign_codes: Vec<&String> = all_codes
                .iter()
                .filter(|code| **code != result.code && combined.contains(code.as_str()))
                .collect();
            let markers = foreign_markers(result.model.template_type, &combined);
            let (words, dup_ratio) = doubled_word_ratio(&result.capture.generated);
            let dup_flag = words >= 30 && dup_ratio >= 0.12;

            let ok = result.capture.error.is_none() && result.capture.completed;
            if ok {
                completed += 1;
            } else {
                errored += 1;
                let error_text = result.capture.error.as_deref().unwrap_or("no completion");
                if error_text.contains("disconnected")
                    || error_text.contains("dead")
                    || error_text.contains("Engine process")
                {
                    dead_errors += 1;
                }
            }
            if !foreign_codes.is_empty() {
                flagged_codes += 1;
            }
            if !markers.is_empty() {
                flagged_markers += 1;
            }
            if dup_flag {
                flagged_dup += 1;
            }
            max_ms = max_ms.max(result.ms);
            sum_ms += result.ms;

            let flagged = !foreign_codes.is_empty() || !markers.is_empty() || dup_flag;
            let sample_limit = if flagged { 4000 } else { 200 };
            let record = json!({
                "kind": "request",
                "step_n": n,
                "slot": result.index,
                "model": result.model.template_type,
                "code": result.code,
                "request_id": result.request_id,
                "ms": result.ms,
                "ok": ok,
                "error": result.capture.error,
                "events": result.capture.events,
                "generated_len": result.capture.generated.chars().count(),
                "dup_ratio": (dup_ratio * 1000.0).round() / 1000.0,
                "dup_flag": dup_flag,
                "foreign_codes": foreign_codes,
                "foreign_markers": markers,
                "generated": truncated(&result.capture.generated, sample_limit),
                "content": truncated(&result.capture.content, sample_limit),
            });
            append_line(&config.out_path, &record);
        }

        let summary = json!({
            "kind": "step_summary",
            "n": n,
            "completed": completed,
            "errored": errored,
            "engine_dead_errors": dead_errors,
            "flagged_foreign_codes": flagged_codes,
            "flagged_foreign_markers": flagged_markers,
            "flagged_dup_ratio": flagged_dup,
            "wall_ms": wall_ms,
            "max_request_ms": max_ms,
            "mean_request_ms": if results.is_empty() { 0 } else { sum_ms / results.len() as u64 },
        });
        append_line(&config.out_path, &summary);
        println!("{summary}");

        if dead_errors > n / 2 {
            let abort = json!({
                "kind": "aborted",
                "after_step_n": n,
                "reason": "majority of rung failed with engine-dead errors",
            });
            append_line(&config.out_path, &abort);
            println!("{abort}");
            break;
        }

        tokio::time::sleep(config.settle).await;
    }
}
