//! Fast repro driver for the gpt_oss tool_chaining/turn3 concurrency drift.
//!
//! Replays the exact golden tool_chaining conversation (turn1 -> turn2 ->
//! turn3) against gpt-oss while N synthetic short chat streams run against
//! other already-loaded models. Drift is never an assertion failure here:
//! every trial is a datapoint appended as one JSON line to `DRIFT_OUT`.
//!
//! Request construction deliberately duplicates golden_path.rs
//! (test_tool_chaining + `request()`) instead of sharing its helpers, so the
//! probe stays byte-identical to the drifting suite while that file keeps
//! evolving. If test_tool_chaining's prompts or sampling setup change, the
//! goldens get re-recorded and this file must be updated to match.
//!
//! Run alone, never alongside the rest of the suite:
//!
//! ```text
//! PIE_LOCAL_BUILD=$PWD/../proxy-inference-engine/release \
//! DRIFT_N_LOAD=4 DRIFT_TRIALS=4 \
//! cargo test --test golden drift_repro -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Knobs (all env, all optional):
//! - `DRIFT_N_LOAD`: parallel load streams; 0 = serial control (default 0)
//! - `DRIFT_TRIALS`: trials in this process, sharing one engine boot (default 1)
//! - `DRIFT_TOP_LOGPROBS`: top-K logprobs requested on probe turns; 0 = off
//!   (default 5). Keep a 0 arm in every sweep: the logprob side-channel adds
//!   kernels to the compute stream and must be shown not to move the flip rate.
//! - `DRIFT_LOAD_MODELS`: `mixed` (llama3+gemma4+qwen3.5) | `same` (gpt-oss
//!   co-batching) | comma-separated checkpoint ids (default `mixed`)
//! - `DRIFT_LOAD_MAX_TOKENS`: max_output_tokens per load request (default 48)
//! - `DRIFT_LOAD_PROMPT_REPEAT`: prefill inflation factor for load prompts
//!   (default 1)
//! - `DRIFT_TURN_TIMEOUT_SECS`: per-turn wedge backstop (default 600)
//! - `DRIFT_OUT`: results JSONL path (default /tmp/drift-repro/results.jsonl)
//! - `DRIFT_RELOAD_BEFORE_TRIAL`: trial index before which gpt-oss is
//!   force-unloaded and reloaded through the engine management socket
//!   (reload discriminator for resident-state corruption); -1 = off

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use orchard::{
    IPCClient, ReasoningConfig, ResponseEvent, ResponseInputItem, ResponsesInput, ResponsesRequest,
    ResponsesResult,
};
use serde_json::{json, Value};
use tokio::sync::watch;

use crate::fixture::{
    get_fixture, Model, GEMMA4_MODEL_ID, GPT_OSS_MODEL_ID, LLAMA_MODEL_ID, MODELS, QWEN_MODEL_ID,
};
use crate::golden_io::{drain_stream, golden_path, normalize, Turn};

/// Sampled positions of turn3 captured per trial. The known flip is at probe
/// position 3 (event index 5): token 3575 "You" -> 976 "The" right after the
/// `<|channel|>final<|message|>` header.
const PROBE_POSITIONS: usize = 10;

const KEY: &str = "K7-MAGENTA-9931";
const CHEST_CONTENTS: &str = "a jade dragon figurine";
const SYSTEM_PROMPT: &str = "You are a helpful assistant with tool calling. Use the tools in the right order, passing each tool's result into the next, then answer the request.";
const USER_PROMPT: &str = "Find the key hidden in the library, then unlock the treasure chest with it and tell me what's inside.";

const LOAD_PROMPTS: [&str; 6] = [
    "Give me one sentence about the ocean.",
    "Name three prime numbers and add them together.",
    "Summarize the plot of a heist movie in two sentences.",
    "What is a good name for a gray cat? Answer briefly.",
    "Explain what a hash map is in one sentence.",
    "Write a haiku about winter mornings.",
];

fn model_for(checkpoint: &str) -> Model {
    *MODELS
        .iter()
        .find(|model| model.checkpoint == checkpoint)
        .unwrap_or_else(|| panic!("unknown test model {checkpoint}"))
}

// --- golden_path.rs mirrors (keep byte-identical to the suite) ---

fn message(role: &str, content: &str) -> ResponseInputItem {
    ResponseInputItem::Message {
        role: role.to_string(),
        content: json!(content),
        tool_calls: None,
        tool_call_id: None,
    }
}

fn function_call(call_id: &str, name: &str, arguments: &str) -> ResponseInputItem {
    ResponseInputItem::FunctionCall {
        call_id: call_id.to_string(),
        name: name.to_string(),
        arguments: arguments.to_string(),
    }
}

fn function_output(call_id: &str, output: Value) -> ResponseInputItem {
    ResponseInputItem::FunctionCallOutput {
        call_id: call_id.to_string(),
        output: tool_output_json(&output).into(),
    }
}

fn tool_output_json(value: &Value) -> String {
    match value {
        Value::Array(values) => {
            let values = values
                .iter()
                .map(tool_output_json)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{values}]")
        }
        Value::Object(object) => {
            let fields = object
                .iter()
                .map(|(key, value)| {
                    let key = serde_json::to_string(key).expect("JSON object key serializes");
                    format!("{key}: {}", tool_output_json(value))
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{fields}}}")
        }
        other => serde_json::to_string(other).expect("JSON value serializes"),
    }
}

fn request(input: Vec<ResponseInputItem>) -> ResponsesRequest {
    ResponsesRequest {
        input: ResponsesInput::Items(input),
        stream: true,
        instructions: None,
        temperature: None,
        top_p: None,
        top_k: None,
        min_p: None,
        deterministic: true,
        frequency_penalty: None,
        presence_penalty: None,
        max_output_tokens: Some(512),
        top_logprobs: None,
        core_tools: Vec::new(),
        active_tools: Vec::new(),
        tool_choice: None,
        min_tool_calls: None,
        max_tool_calls: None,
        text: None,
        reasoning: None,
        reasoning_effort: None,
        metadata: None,
        parallel_tool_calls: false,
        prefix_cache: Some(false),
        stream_tokens: true,
    }
}

fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    })
}

fn chaining_tools() -> Vec<Value> {
    vec![
        tool(
            "find_key",
            "Search a room and return the key hidden there.",
            json!({"room": {"type": "string"}}),
            &["room"],
        ),
        tool(
            "unlock_chest",
            "Unlock the treasure chest with a key and return its contents.",
            json!({"key": {"type": "string"}}),
            &["key"],
        ),
    ]
}

fn reasoning_for(model: Model) -> Option<ReasoningConfig> {
    model.thinking.enabled().then(|| ReasoningConfig::Object {
        effort: "medium".to_string(),
    })
}

// --- driver ---

struct Config {
    n_load: usize,
    trials: usize,
    top_logprobs: i32,
    load_models: Vec<Model>,
    load_max_tokens: i32,
    load_prompt_repeat: usize,
    turn_timeout: Duration,
    out_path: PathBuf,
    reload_before_trial: i64,
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}

impl Config {
    fn from_env() -> Self {
        let spec = std::env::var("DRIFT_LOAD_MODELS").unwrap_or_else(|_| "mixed".to_string());
        let load_models = match spec.as_str() {
            "mixed" => vec![
                model_for(LLAMA_MODEL_ID),
                model_for(GEMMA4_MODEL_ID),
                model_for(QWEN_MODEL_ID),
            ],
            "same" => vec![model_for(GPT_OSS_MODEL_ID)],
            list => list
                .split(',')
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(model_for)
                .collect(),
        };
        assert!(
            !load_models.is_empty(),
            "DRIFT_LOAD_MODELS resolved to no models"
        );
        Self {
            n_load: env_parse("DRIFT_N_LOAD", 0usize),
            trials: env_parse("DRIFT_TRIALS", 1usize),
            top_logprobs: env_parse("DRIFT_TOP_LOGPROBS", 5i32),
            load_models,
            load_max_tokens: env_parse("DRIFT_LOAD_MAX_TOKENS", 48i32),
            load_prompt_repeat: env_parse("DRIFT_LOAD_PROMPT_REPEAT", 1usize),
            turn_timeout: Duration::from_secs(env_parse("DRIFT_TURN_TIMEOUT_SECS", 600u64)),
            out_path: PathBuf::from(
                std::env::var("DRIFT_OUT")
                    .unwrap_or_else(|_| "/tmp/drift-repro/results.jsonl".to_string()),
            ),
            reload_before_trial: env_parse("DRIFT_RELOAD_BEFORE_TRIAL", -1i64),
        }
    }
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

/// Force-unload gpt-oss and reload it from disk through the engine's
/// management socket, bypassing the orchard-rs registry (which keeps
/// believing the model is Ready — the engine ends up back in the same
/// state, so that belief stays true). Discriminates resident-state
/// corruption: if post-reload trials match golden again, the corrupted
/// state lived in the unloaded runtime (weights/cache), not the client.
async fn reload_gpt_oss() -> Result<Value, String> {
    let fixture = get_fixture().await;
    let info = fixture
        .registry
        .ensure_loaded(GPT_OSS_MODEL_ID)
        .await
        .map_err(|err| format!("ensure_loaded failed: {err:?}"))?;

    let mut mgmt = IPCClient::new();
    mgmt.connect()
        .map_err(|err| format!("management connect failed: {err:?}"))?;

    let listing = mgmt
        .send_management_command_async(json!({"type": "list_models"}), Duration::from_secs(30))
        .await
        .map_err(|err| format!("list_models failed: {err:?}"))?;
    let canonical = listing["data"]["list_models"]["models"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|entry| {
            entry["requested_id"]
                .as_str()
                .is_some_and(|id| id.contains("gpt-oss"))
                || entry["canonical_id"]
                    .as_str()
                    .is_some_and(|id| id.contains("gpt-oss"))
        })
        .and_then(|entry| entry["canonical_id"].as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("gpt-oss not in engine inventory: {listing}"))?;

    let unload = mgmt
        .send_management_command_async(
            json!({"type": "unload_model", "requested_id": canonical}),
            Duration::from_secs(120),
        )
        .await
        .map_err(|err| format!("unload_model failed: {err:?}"))?;

    let load = mgmt
        .send_management_command_async(
            json!({
                "type": "load_model",
                "requested_id": GPT_OSS_MODEL_ID,
                "canonical_id": canonical,
                "model_path": info.model_path,
                "wait_for_completion": false,
            }),
            Duration::from_secs(120),
        )
        .await
        .map_err(|err| format!("load_model failed: {err:?}"))?;

    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let listing = mgmt
            .send_management_command_async(json!({"type": "list_models"}), Duration::from_secs(30))
            .await
            .map_err(|err| format!("list_models poll failed: {err:?}"))?;
        let ready = listing["data"]["list_models"]["models"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|entry| {
                entry["canonical_id"].as_str() == Some(canonical.as_str())
                    && entry["load_state"].as_str() == Some("ready")
            });
        if ready {
            break;
        }
        if Instant::now() > deadline {
            return Err("gpt-oss did not return to ready after reload".to_string());
        }
    }

    // The manager reports "ready" when weights are loaded; runtime activation
    // trails it. Warm with a tiny request until the runtime answers.
    let gpt_oss = model_for(GPT_OSS_MODEL_ID);
    let mut warm_error = String::new();
    for _ in 0..20 {
        let mut warm_request = request(vec![message("user", "Say OK.")]);
        warm_request.max_output_tokens = Some(8);
        warm_request.stream_tokens = false;
        match probe_stream(gpt_oss, warm_request, Duration::from_secs(120)).await {
            Ok(_) => {
                return Ok(json!({
                    "canonical_id": canonical,
                    "unload": unload,
                    "load": load,
                }));
            }
            Err(reason) => {
                warm_error = reason;
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        }
    }
    Err(format!(
        "gpt-oss runtime never answered after reload: {warm_error}"
    ))
}

fn load_request(prompt: String, max_tokens: i32) -> ResponsesRequest {
    let mut req = request(vec![message("user", &prompt)]);
    req.max_output_tokens = Some(max_tokens);
    req.stream_tokens = false;
    req
}

async fn load_worker(
    index: usize,
    model: Model,
    max_tokens: i32,
    prompt_repeat: usize,
    stop: watch::Receiver<bool>,
    iterations: Arc<AtomicU64>,
    errors: Arc<AtomicU64>,
) {
    let fixture = get_fixture().await;
    // Stagger starts so the load streams sit in different prefill/decode phases.
    tokio::time::sleep(Duration::from_millis(250 * index as u64)).await;
    let mut round = index;
    while !*stop.borrow() {
        let base = LOAD_PROMPTS[round % LOAD_PROMPTS.len()];
        let prompt = vec![base; prompt_repeat.max(1)].join(" ");
        round += 1;
        match fixture
            .client
            .aresponses(model.checkpoint, load_request(prompt, max_tokens))
            .await
        {
            Ok(ResponsesResult::Stream { events, .. }) => {
                if drain_stream(events).await.error.is_some() {
                    errors.fetch_add(1, Ordering::Relaxed);
                } else {
                    iterations.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(_) => {
                errors.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                errors.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
}

/// Non-panicking, timeout-guarded variant of golden_path.rs `run_stream`: in
/// an unattended sweep a wedged or failed stream is a datapoint, not a crash.
async fn probe_stream(
    model: Model,
    request: ResponsesRequest,
    timeout: Duration,
) -> Result<Turn, String> {
    let fixture = get_fixture().await;
    match tokio::time::timeout(timeout, async {
        match fixture.client.aresponses(model.checkpoint, request).await {
            Ok(ResponsesResult::Stream { events, .. }) => {
                let turn = drain_stream(events).await;
                // A mid-stream engine failure (sequence error delta or channel
                // close without a final delta) is an aborted trial, never a
                // short-but-complete turn.
                match &turn.error {
                    Some(error) => Err(format!("engine_error: {error}")),
                    None => Ok(turn),
                }
            }
            Ok(_) => Err("expected stream result".to_string()),
            Err(err) => Err(format!("aresponses failed: {err:?}")),
        }
    })
    .await
    {
        Ok(result) => result,
        Err(_) => Err(format!("stream did not complete within {timeout:?}")),
    }
}

/// Remove instrumentation-only fields before golden comparison so a probe with
/// `top_logprobs` requested still compares clean against goldens recorded
/// without it (`output_token.top_logprobs` arrays plus the response snapshots'
/// `top_logprobs: null` -> `5` request echo).
fn strip_top_logprobs(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("top_logprobs");
            for child in map.values_mut() {
                strip_top_logprobs(child);
            }
        }
        Value::Array(values) => {
            for child in values.iter_mut() {
                strip_top_logprobs(child);
            }
        }
        _ => {}
    }
}

struct TurnComparison {
    matches: bool,
    first_diff: Option<usize>,
    live_events: usize,
    golden_events: usize,
}

fn compare_turn(golden_turn: &Value, turn: &Turn) -> TurnComparison {
    let mut golden_events = golden_turn.as_array().cloned().unwrap_or_default();
    for event in &mut golden_events {
        strip_top_logprobs(event);
    }
    let mut live_events = normalize(&turn.events);
    for event in &mut live_events {
        strip_top_logprobs(event);
    }
    let shared = golden_events.len().min(live_events.len());
    let first_diff = (0..shared)
        .find(|&index| golden_events[index] != live_events[index])
        .or_else(|| (golden_events.len() != live_events.len()).then_some(shared));
    TurnComparison {
        matches: first_diff.is_none(),
        first_diff,
        live_events: live_events.len(),
        golden_events: golden_events.len(),
    }
}

fn turn_fields(record: &mut Value, label: &str, comparison: &TurnComparison, elapsed_ms: u64) {
    let object = record.as_object_mut().expect("trial record is an object");
    object.insert(format!("{label}_match"), json!(comparison.matches));
    object.insert(format!("{label}_first_diff"), json!(comparison.first_diff));
    object.insert(
        format!("{label}_events_live"),
        json!(comparison.live_events),
    );
    object.insert(
        format!("{label}_events_golden"),
        json!(comparison.golden_events),
    );
    object.insert(format!("{label}_ms"), json!(elapsed_ms));
}

/// First `PROBE_POSITIONS` sampled token ids of a live turn plus, when logprob
/// capture is on, the top-K `[token_id, logprob]` pairs at each position.
fn probe_rows(turn: &Turn) -> (Vec<i64>, Vec<Value>) {
    let mut token_ids = Vec::new();
    let mut top_k = Vec::new();
    for event in &turn.events {
        if let ResponseEvent::OutputToken(token) = event {
            if token_ids.len() == PROBE_POSITIONS {
                break;
            }
            token_ids.push(i64::from(token.token_id));
            top_k.push(Value::Array(
                token
                    .top_logprobs
                    .iter()
                    .map(|entry| json!([entry.token.parse::<i64>().unwrap_or(-1), entry.logprob]))
                    .collect(),
            ));
        }
    }
    (token_ids, top_k)
}

fn golden_probe_ids(golden_turn: &Value) -> Vec<i64> {
    golden_turn
        .as_array()
        .into_iter()
        .flatten()
        .filter(|event| event.get("type").and_then(Value::as_str) == Some("response.output_token"))
        .filter_map(|event| event.get("token_id").and_then(Value::as_i64))
        .take(PROBE_POSITIONS)
        .collect()
}

async fn run_trial(gpt_oss: Model, golden: &Value, config: &Config) -> Value {
    let tools = chaining_tools();
    let reasoning = reasoning_for(gpt_oss);
    let top_logprobs = (config.top_logprobs > 0).then_some(config.top_logprobs);
    let mut record = json!({ "kind": "trial" });

    // turn1: find_key (mirrors golden_path.rs test_tool_chaining)
    let mut conversation = vec![
        message("system", SYSTEM_PROMPT),
        message("user", USER_PROMPT),
    ];
    let mut turn1_request = request(conversation.clone());
    turn1_request.core_tools = tools.clone();
    turn1_request.tool_choice = Some(json!("required"));
    turn1_request.reasoning = reasoning.clone();
    turn1_request.top_logprobs = top_logprobs;
    let started = Instant::now();
    let turn1 = match probe_stream(gpt_oss, turn1_request, config.turn_timeout).await {
        Ok(turn) => turn,
        Err(reason) => {
            record["aborted_at"] = json!("turn1");
            record["abort_reason"] = json!(reason);
            return record;
        }
    };
    turn_fields(
        &mut record,
        "turn1",
        &compare_turn(&golden["turn1"], &turn1),
        started.elapsed().as_millis() as u64,
    );
    let Some(find) = turn1.function_calls.first().cloned() else {
        record["aborted_at"] = json!("turn1");
        record["abort_reason"] = json!("turn1 produced no function call");
        return record;
    };

    // turn2: unlock_chest
    conversation.push(function_call(&find.call_id, &find.name, &find.arguments));
    conversation.push(function_output(&find.call_id, json!({"key": KEY})));
    let mut turn2_request = request(conversation.clone());
    turn2_request.core_tools = tools.clone();
    turn2_request.tool_choice = Some(json!("required"));
    turn2_request.reasoning = reasoning.clone();
    turn2_request.top_logprobs = top_logprobs;
    let started = Instant::now();
    let turn2 = match probe_stream(gpt_oss, turn2_request, config.turn_timeout).await {
        Ok(turn) => turn,
        Err(reason) => {
            record["aborted_at"] = json!("turn2");
            record["abort_reason"] = json!(reason);
            return record;
        }
    };
    turn_fields(
        &mut record,
        "turn2",
        &compare_turn(&golden["turn2"], &turn2),
        started.elapsed().as_millis() as u64,
    );
    let Some(unlock) = turn2.function_calls.first().cloned() else {
        record["aborted_at"] = json!("turn2");
        record["abort_reason"] = json!("turn2 produced no function call");
        return record;
    };

    // turn3: the drifting message turn
    conversation.push(function_call(
        &unlock.call_id,
        &unlock.name,
        &unlock.arguments,
    ));
    conversation.push(function_output(
        &unlock.call_id,
        json!({"contents": CHEST_CONTENTS}),
    ));
    let mut turn3_request = request(conversation);
    turn3_request.core_tools = tools;
    turn3_request.tool_choice = Some(json!("none"));
    turn3_request.reasoning = reasoning;
    turn3_request.top_logprobs = top_logprobs;
    let started = Instant::now();
    let turn3 = match probe_stream(gpt_oss, turn3_request, config.turn_timeout).await {
        Ok(turn) => turn,
        Err(reason) => {
            record["aborted_at"] = json!("turn3");
            record["abort_reason"] = json!(reason);
            return record;
        }
    };
    turn_fields(
        &mut record,
        "turn3",
        &compare_turn(&golden["turn3"], &turn3),
        started.elapsed().as_millis() as u64,
    );

    let (probe_live, probe_top_k) = probe_rows(&turn3);
    let probe_golden = golden_probe_ids(&golden["turn3"]);
    let probe_first_diff = (0..probe_golden.len().min(probe_live.len()))
        .find(|&index| probe_golden[index] != probe_live[index]);
    let object = record.as_object_mut().expect("trial record is an object");
    object.insert(
        "valid_chain".to_string(),
        json!(find.name == "find_key" && unlock.name == "unlock_chest"),
    );
    object.insert("probe_golden".to_string(), json!(probe_golden));
    object.insert("probe_live".to_string(), json!(probe_live));
    object.insert("probe_first_diff".to_string(), json!(probe_first_diff));
    object.insert("probe_top_logprobs".to_string(), Value::Array(probe_top_k));
    object.insert("stop_token".to_string(), json!(turn3.stop_token));
    object.insert(
        "answer_prefix".to_string(),
        json!(turn3
            .content_done
            .as_deref()
            .unwrap_or_default()
            .chars()
            .take(80)
            .collect::<String>()),
    );
    record
}

fn append_line(path: &Path, record: &Value) {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|err| panic!("open {}: {err}", path.display()));
    writeln!(file, "{record}").expect("append drift record");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual concurrency-drift repro driver; run alone with --ignored (see /tmp/drift-repro/runbook.md)"]
async fn drift_repro_tool_chaining_turn3() {
    let config = Config::from_env();
    if let Some(parent) = config.out_path.parent() {
        fs::create_dir_all(parent).expect("create DRIFT_OUT directory");
    }
    let gpt_oss = model_for(GPT_OSS_MODEL_ID);
    let golden_file = golden_path("gpt_oss", "tool_chaining");
    let golden: Value = serde_json::from_str(
        &fs::read_to_string(&golden_file)
            .unwrap_or_else(|err| panic!("missing golden {}: {err}", golden_file.display())),
    )
    .expect("golden tool_chaining.json parses");

    let boot_started = Instant::now();
    let _fixture = get_fixture().await;
    let boot_ms = boot_started.elapsed().as_millis() as u64;

    let (stop_tx, stop_rx) = watch::channel(false);
    let iterations = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    let mut workers = Vec::new();
    for index in 0..config.n_load {
        let model = config.load_models[index % config.load_models.len()];
        workers.push(tokio::spawn(load_worker(
            index,
            model,
            config.load_max_tokens,
            config.load_prompt_repeat,
            stop_rx.clone(),
            Arc::clone(&iterations),
            Arc::clone(&errors),
        )));
    }
    if config.n_load > 0 {
        // Let the load reach steady state before the first probe.
        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    let header = json!({
        "kind": "run_header",
        "started_at": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0),
        "pid": std::process::id(),
        "boot_ms": boot_ms,
        "n_load": config.n_load,
        "load_models": config
            .load_models
            .iter()
            .map(|model| model.checkpoint)
            .collect::<Vec<_>>(),
        "load_max_tokens": config.load_max_tokens,
        "load_prompt_repeat": config.load_prompt_repeat,
        "top_logprobs": config.top_logprobs,
        "trials": config.trials,
        "reload_before_trial": config.reload_before_trial,
        "pie_local_build": std::env::var("PIE_LOCAL_BUILD").ok(),
    });
    append_line(&config.out_path, &header);
    println!("{header}");

    for trial in 0..config.trials {
        if config.reload_before_trial == trial as i64 {
            let reload_started = epoch_ms();
            let reload_record = match reload_gpt_oss().await {
                Ok(detail) => json!({
                    "kind": "reload",
                    "before_trial": trial,
                    "ok": true,
                    "ts_start_ms": reload_started,
                    "ts_end_ms": epoch_ms(),
                    "detail": detail,
                }),
                Err(reason) => json!({
                    "kind": "reload",
                    "before_trial": trial,
                    "ok": false,
                    "ts_start_ms": reload_started,
                    "ts_end_ms": epoch_ms(),
                    "error": reason,
                }),
            };
            append_line(&config.out_path, &reload_record);
            println!("{reload_record}");
        }
        let load_before = iterations.load(Ordering::Relaxed);
        let ts_start_ms = epoch_ms();
        let mut record = run_trial(gpt_oss, &golden, &config).await;
        let object = record.as_object_mut().expect("trial record is an object");
        object.insert("ts_start_ms".to_string(), json!(ts_start_ms));
        object.insert("ts_end_ms".to_string(), json!(epoch_ms()));
        object.insert("trial".to_string(), json!(trial));
        object.insert("n_load".to_string(), json!(config.n_load));
        object.insert(
            "load_iterations_during_trial".to_string(),
            json!(iterations.load(Ordering::Relaxed) - load_before),
        );
        object.insert(
            "load_errors_total".to_string(),
            json!(errors.load(Ordering::Relaxed)),
        );
        append_line(&config.out_path, &record);
        println!("{record}");
    }

    let _ = stop_tx.send(true);
    for worker in workers {
        let _ = worker.await;
    }
}
