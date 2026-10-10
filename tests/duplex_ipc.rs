//! Hermetic native-duplex wire tests. Each scenario runs in a new test process
//! whose entire IPC/cache/HF environment is a private TempDir. No Client or
//! InferenceEngine lifecycle API is used, and no real engine/model is launched.
#![cfg(feature = "duplex")]

use base64::{engine::general_purpose::STANDARD, Engine as _};
use nng::{options::Options, Protocol, Socket};
use orchard::duplex::{DuplexControl, DuplexEvent, DuplexOptions, DuplexSession, FRAME_SAMPLES};
use orchard::ipc::serialization::PromptPayload;
use orchard::{endpoints, Client, Error, IPCClient, ModelRegistry};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const CASE_ENV: &str = "ORCHARD_DUPLEX_HERMETIC_CASE";
const ROOT_ENV: &str = "ORCHARD_DUPLEX_HERMETIC_ROOT";
const DEADLINE: Duration = Duration::from_secs(4);

#[test]
fn native_duplex_ipc_contract() {
    if let Ok(case) = std::env::var(CASE_ENV) {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(run_case(&case));
        return;
    }

    // Environment is set on child creation, before NNG/Tokio starts threads.
    // Never temporarily change the parent test process's endpoint environment.
    for case in [
        "wire_and_lifecycle",
        "grounded_reply",
        "grounded_reply_unsupported",
        "grounded_reply_legacy",
        "grounded_reply_epoch_race",
        "grounded_reply_bad_ack",
        "bounded_audio",
        "bounded_input",
        "bounded_control_events",
        "engine_death",
        "generic_open_error",
        "bad_geometry",
        "bad_channels",
        "missing_metrics",
        "wrong_request",
        "malformed_pull_json",
        "rollover_ack_race",
        "management_lock_isolation",
        "request_cancellation_scope",
    ] {
        let temporary = tempfile::Builder::new()
            .prefix("odx-")
            .tempdir_in("/tmp")
            .unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "native_duplex_ipc_contract",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CASE_ENV, case)
            .env(ROOT_ENV, &root)
            .env("ORCHARD_CACHE_ROOT", root.join("cache"))
            .env("ORCHARD_IPC_ROOT", root.join("ipc"))
            .env("HF_HOME", root.join("hf"))
            .env("HF_HUB_OFFLINE", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "Hermetic case {case} failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        println!("native duplex fake-PIE case passed: {case}");
    }
}

#[derive(Clone)]
struct Route {
    socket: Arc<Socket>,
    request_id: u64,
    channel_id: u64,
    model_id: String,
}

struct FakeState {
    stop: AtomicBool,
    epoch: AtomicU64,
    reference_version: AtomicU64,
    input_acks: AtomicU64,
    route: Mutex<Option<Route>>,
    opens: Mutex<Vec<Value>>,
    commands: Mutex<Vec<Value>>,
    wake: Condvar,
    gate_wake: Condvar,
    gate: Mutex<bool>,
    case: String,
}

impl FakeState {
    fn route(&self) -> Route {
        self.route.lock().unwrap().as_ref().unwrap().clone()
    }

    fn emit(
        &self,
        name: &str,
        epoch: u64,
        fields: Value,
        content: Option<&str>,
        pcm: Option<f32>,
        final_delta: bool,
    ) {
        let route = self.route();
        let mut metadata = json!({"duplex_version":1,"epoch":epoch});
        metadata
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let mut event = json!({
            "request_id":route.request_id,"response_channel_id":route.channel_id,
            "sequence_id":route.request_id + 100,"prompt_index":0,"candidate_index":0,
            "is_final_delta":final_delta,"finish_reason":if final_delta {"stop"} else {"delta"},
            "modal_type":"audio","modal_event":name,"modal_decoder_id":"moshi.duplex",
            "modal_mime_type":"audio/pcm;format=f32le;rate=24000;channels=1",
            "modal_metadata_json":metadata.to_string()
        });
        if let Some(content) = content {
            event["content"] = content.into();
        }
        if let Some(value) = pcm {
            let bytes = vec![value; FRAME_SAMPLES]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            event["modal_bytes_b64"] = STANDARD.encode(bytes).into();
        }
        self.send(&route, event);
    }

    fn send(&self, route: &Route, event: Value) {
        let message = format!("resp:{:x}:{}", route.channel_id, event);
        if let Err((_, error)) = route.socket.send(message.as_bytes()) {
            assert!(
                self.stop.load(Ordering::Acquire),
                "Fake response send: {error}"
            );
        }
    }

    fn reply(&self, socket: &Socket, response: Value) -> bool {
        match socket.send(response.to_string().as_bytes()) {
            Ok(()) => true,
            Err((_, error)) => {
                assert!(
                    self.stop.load(Ordering::Acquire),
                    "Fake management reply: {error}"
                );
                false
            }
        }
    }

    fn audio(&self, epoch: u64, sequence: u64, value: f32) {
        self.emit(
            "duplex.audio",
            epoch,
            json!({
                "sequence":sequence,"input_sequence":sequence,"model_step":sequence,
                "speaking":true,"compute_ms":7.5,"queue_ms":1.25
            }),
            None,
            Some(value),
            false,
        );
    }

    fn text(&self, epoch: u64, sequence: u64, value: &str) {
        self.emit(
            "duplex.text",
            epoch,
            json!({
                "sequence":sequence,"input_sequence":sequence,"model_step":sequence,
                "speaking":true,"compute_ms":7.5,"queue_ms":1.25
            }),
            Some(value),
            None,
            false,
        );
    }

    fn metrics(&self, epoch: u64, output_frames: u64) {
        self.emit("duplex.metrics", epoch, json!({
            "input_frames":output_frames,"output_frames":output_frames,"synthetic_silence_frames":0,
            "dropped_input_frames":0,"dropped_output_frames":0,"compute_ms_total":7.5 * output_frames as f64,
            "compute_ms_max":7.5,"queue_ms_max":1.25,"elapsed_ms":80.0 * output_frames as f64
        }), None, None, false);
    }

    fn closed(&self, epoch: u64, reason: &str) {
        self.emit(
            "duplex.closed",
            epoch,
            json!({"reason":reason,"dropped_input_frames":0}),
            None,
            None,
            true,
        );
    }

    fn command_count(&self, kind: &str) -> usize {
        self.commands
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c["type"] == kind)
            .count()
    }

    fn wait_commands(&self, kind: &str, count: usize) {
        let commands = self.commands.lock().unwrap();
        let (_commands, timeout) = self
            .wake
            .wait_timeout_while(commands, DEADLINE, |commands| {
                commands.iter().filter(|c| c["type"] == kind).count() < count
            })
            .unwrap();
        assert!(
            !timeout.timed_out(),
            "Timed out waiting for {kind} #{count}"
        );
    }
}

struct FakePie {
    state: Arc<FakeState>,
    request: Arc<Socket>,
    management: Arc<Socket>,
    _publish: Socket,
    workers: Vec<JoinHandle<()>>,
}

impl FakePie {
    fn start(case: &str, root: &Path) -> Self {
        let actual_root = endpoints::ipc_root();
        assert!(
            actual_root.starts_with(root),
            "IPC escaped the private test root: {actual_root:?}"
        );
        let cache = root.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        // The fake server lives in this child process. Removing only this private
        // PID file below simulates engine death without signalling any process.
        std::fs::write(cache.join("engine.pid"), std::process::id().to_string()).unwrap();
        let socket = |protocol, url: String| {
            let socket = Socket::new(protocol).unwrap();
            socket
                .set_opt::<nng::options::RecvTimeout>(Some(Duration::from_millis(20)))
                .unwrap();
            socket
                .set_opt::<nng::options::SendTimeout>(Some(DEADLINE))
                .unwrap();
            socket.listen(&url).unwrap();
            socket
        };
        let request = Arc::new(socket(Protocol::Pull0, endpoints::request_url()));
        let management = Arc::new(socket(Protocol::Rep0, endpoints::management_url()));
        let publish = socket(Protocol::Pub0, endpoints::response_url());
        let state = Arc::new(FakeState {
            stop: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
            reference_version: AtomicU64::new(0),
            input_acks: AtomicU64::new(0),
            route: Mutex::new(None),
            opens: Mutex::new(vec![]),
            commands: Mutex::new(vec![]),
            wake: Condvar::new(),
            gate_wake: Condvar::new(),
            gate: Mutex::new(case == "bounded_input" || case == "rollover_ack_race"),
            case: case.into(),
        });
        let mut workers = vec![];
        {
            let state = Arc::clone(&state);
            let request = Arc::clone(&request);
            workers.push(thread::spawn(move || {
                while !state.stop.load(Ordering::Acquire) {
                    let message = match request.recv() {
                        Ok(message) => message,
                        Err(nng::Error::TimedOut) => continue,
                        Err(_) if state.stop.load(Ordering::Acquire) => break,
                        Err(error) => panic!("Fake request recv: {error}"),
                    };
                    let bytes = message.as_slice();
                    assert!(bytes.len() >= 4);
                    let length = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
                    assert!(length <= bytes.len() - 4);
                    let meta: Value = serde_json::from_slice(&bytes[4..4 + length]).unwrap();
                    if state.case == "request_cancellation_scope" {
                        assert_eq!(meta["request_type"], 0);
                        assert_eq!(meta["prompts"].as_array().unwrap().len(), 1);
                        state.opens.lock().unwrap().push(meta);
                        continue;
                    }
                    assert_eq!(meta["request_type"], 10);
                    assert_eq!(meta["response_transport"], "pull_v1");
                    let prompts = meta["prompts"].as_array().unwrap();
                    assert_eq!(prompts.len(), 1);
                    assert_eq!(prompts[0]["text_size"], 0);
                    assert_eq!(prompts[0]["best_of"], 1);
                    assert_eq!(prompts[0]["final_candidates"], 1);
                    let options: Value = serde_json::from_str(prompts[0]["modal_options_json"].as_str().unwrap()).unwrap();
                    assert_eq!(options["duplex_version"], 1);
                    assert_eq!(options["autonomous"], true);
                    assert_eq!(options["channels"], 1);
                    for legacy in ["device", "mimi_cpu", "codec_threads", "text_token_interval"] {
                        assert!(options.get(legacy).is_none());
                    }
                    let channel_id = meta["response_channel_id"].as_u64().unwrap();
                    let url = endpoints::pull_response_url(channel_id);
                    assert!(Path::new(url.strip_prefix("ipc://").unwrap()).exists(), "Client must listen before open");
                    let response = Socket::new(Protocol::Push0).unwrap();
                    response.set_opt::<nng::options::SendTimeout>(Some(DEADLINE)).unwrap();
                    response.set_opt::<nng::options::SendBufferSize>(0).unwrap();
                    response.dial(&url).unwrap();
                    let route = Route { socket: Arc::new(response), request_id: meta["request_id"].as_u64().unwrap(),
                        channel_id, model_id: meta["model_id"].as_str().unwrap().to_owned() };
                    *state.route.lock().unwrap() = Some(route.clone());
                    state.opens.lock().unwrap().push(meta);
                    if state.case == "generic_open_error" {
                        state.send(&route, json!({"request_id":route.request_id,"is_final_delta":true,
                            "finish_reason":"error","error_message":"fixture open rejected"}));
                    } else {
                        state.emit("duplex.ready", 0, json!({"model_id":route.model_id,
                            "sample_rate":if state.case == "bad_geometry" {16000} else {24000},
                            "channels":if state.case == "bad_channels" {2} else {1},"frame_samples":1920,"supports_reference":true,"supports_forced_text":true,
                            "supports_grounded_response":if state.case == "grounded_reply_legacy" {Value::Null} else {json!(state.case != "grounded_reply_unsupported")},
                            "max_pending_frames":options["max_pending_frames"],"max_pending_output_frames":64
                        }), None, None, false);
                    }
                }
            }));
        }
        {
            let state = Arc::clone(&state);
            let management = Arc::clone(&management);
            workers.push(thread::spawn(move || {
                while !state.stop.load(Ordering::Acquire) {
                    let message = match management.recv() {
                        Ok(message) => message,
                        Err(nng::Error::TimedOut) => continue,
                        Err(_) if state.stop.load(Ordering::Acquire) => break,
                        Err(error) => panic!("Fake management recv: {error}"),
                    };
                    let command: Value = serde_json::from_slice(&message).unwrap();
                    let kind = command["type"].as_str().unwrap();
                    state.commands.lock().unwrap().push(command.clone());
                    state.wake.notify_all();
                    if state.case == "request_cancellation_scope" {
                        assert_eq!(kind, "cancel_request");
                        assert!(command["response_channel_id"].as_u64().unwrap_or(0) > 0);
                        assert!(state.opens.lock().unwrap().iter().any(|request| {
                            request["request_id"] == command["request_id"]
                                && request["response_channel_id"] == command["response_channel_id"]
                        }));
                        assert!(state.reply(&management, json!({"status":"accepted"})));
                        continue;
                    }
                    if kind == "load_model" {
                        assert!(Path::new(command["model_path"].as_str().unwrap())
                            .join("config.json")
                            .is_file());
                        if !state.reply(
                            &management,
                            json!({"status":"ok","data":{"load_model":{"runtime_started":true,
                            "bound_runtime_id":command["canonical_id"],"minimum_memory_bytes":5}}}),
                        ) {
                            break;
                        }
                        continue;
                    }
                    let route = state.route();
                    assert_eq!(command["request_id"], route.request_id);
                    assert_eq!(command["response_channel_id"], route.channel_id);
                    assert_eq!(command["model_id"], route.model_id);
                    let previous = state.epoch.load(Ordering::Acquire);
                    if state.case == "rollover_ack_race"
                        && kind == "duplex_input"
                        && command["epoch"].as_u64().unwrap() < previous
                    {
                        let gate = state.gate.lock().unwrap();
                        let _guard = state
                            .gate_wake
                            .wait_while(gate, |blocked| {
                                *blocked && !state.stop.load(Ordering::Acquire)
                            })
                            .unwrap();
                        if !state.reply(
                            &management,
                            json!({"status":"error","message":"Duplex command epoch is stale",
                            "data":{"duplex":{"error_code":"stale_epoch","epoch":previous,
                                "queue_depth":0,"dropped_input_frames":0,"session_state":"open"}}}),
                        ) {
                            break;
                        }
                        continue;
                    }
                    if state.case == "grounded_reply_epoch_race" && kind == "duplex_reply" {
                        state.epoch.store(previous + 1, Ordering::Release);
                        assert!(state.reply(
                            &management,
                            json!({"status":"error","message":"Duplex command epoch is stale",
                            "data":{"duplex":{"error_code":"stale_epoch","epoch":previous+1}}})
                        ));
                        continue;
                    }
                    if kind != "duplex_close" {
                        assert_eq!(command["epoch"], previous);
                    }
                    let mut data = json!({"epoch":previous,"queue_depth":0,"dropped_input_frames":0,
                        "session_state":if kind == "duplex_close" {"closing"} else {"open"}});
                    match kind {
                        "duplex_input" => {
                            let bytes = STANDARD
                                .decode(command["pcm_f32_b64"].as_str().unwrap())
                                .unwrap();
                            assert_eq!(bytes.len(), FRAME_SAMPLES * 4);
                            assert!(bytes.chunks_exact(4).all(|bytes| {
                                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                                value.is_finite() && value.abs() <= 1.0
                            }));
                            if state.case == "bounded_input" {
                                let gate = state.gate.lock().unwrap();
                                let _guard = state
                                    .gate_wake
                                    .wait_while(gate, |blocked| {
                                        *blocked && !state.stop.load(Ordering::Acquire)
                                    })
                                    .unwrap();
                            }
                            data["accepted_sequence"] = command["sequence"].clone();
                        }
                        "duplex_reference" | "duplex_reply" => {
                            assert!(!command["text"].as_str().unwrap().is_empty());
                            data["reference_version"] =
                                (state.reference_version.fetch_add(1, Ordering::AcqRel) + 1).into();
                            if state.case == "grounded_reply_bad_ack" && kind == "duplex_reply" {
                                data["reference_version"] = Value::Null;
                            }
                        }
                        "duplex_interrupt" | "duplex_reset" => {
                            data["epoch"] = (state.epoch.fetch_add(1, Ordering::AcqRel) + 1).into();
                        }
                        "duplex_close" | "duplex_speak" => {}
                        other => panic!("Unexpected management command {other}"),
                    }
                    if !state.reply(&management, json!({"status":"ok","data":{"duplex":data}})) {
                        break;
                    }
                    if kind == "duplex_input" {
                        state.input_acks.fetch_add(1, Ordering::Release);
                    }
                    if kind == "duplex_close" {
                        state.closed(state.epoch.load(Ordering::Acquire), "client_close");
                    }
                }
            }));
        }
        Self {
            state,
            request,
            management,
            _publish: publish,
            workers,
        }
    }
}

impl Drop for FakePie {
    fn drop(&mut self) {
        self.state.stop.store(true, Ordering::Release);
        self.state.wake.notify_all();
        self.state.gate_wake.notify_all();
        self.request.close();
        self.management.close();
        if let Some(route) = self.state.route.lock().unwrap().as_ref() {
            route.socket.close();
        }
        for worker in self.workers.drain(..) {
            let result = worker.join();
            if !thread::panicking() {
                result.expect("Fake PIE worker panicked");
            }
        }
    }
}

fn descriptor(root: &Path) -> String {
    let model = root.join("native-moshi");
    std::fs::create_dir_all(&model).unwrap();
    let mut config = json!({"model_type":"moshi","moshi_schema_version":1,"rag":true});
    for key in [
        "model_file",
        "codec_file",
        "tokenizer_file",
        "reference_encoder_file",
        "reference_tokenizer_file",
    ] {
        let path = model.join(key);
        std::fs::write(&path, [1]).unwrap();
        config[key] = path.to_string_lossy().into_owned().into();
    }
    std::fs::write(model.join("config.json"), config.to_string()).unwrap();
    model.to_string_lossy().into_owned()
}

async fn next(session: &mut DuplexSession) -> DuplexEvent {
    tokio::time::timeout(DEADLINE, session.next_event())
        .await
        .expect("duplex event deadline")
        .expect("duplex stream ended early")
}

async fn terminal_events(session: &mut DuplexSession) -> Vec<DuplexEvent> {
    let mut events = vec![];
    loop {
        let event = tokio::time::timeout(DEADLINE, session.next_event())
            .await
            .expect("terminal deadline");
        match event {
            Some(event) => events.push(event),
            None => break,
        }
        assert!(
            events.len() <= 258,
            "SDK event queue exceeded its documented bound"
        );
    }
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, DuplexEvent::Closed))
            .count(),
        1
    );
    events
}

async fn wait_closed(control: &DuplexControl) {
    tokio::time::timeout(DEADLINE, async {
        loop {
            // Invalid PCM is never admitted. Once the prior wire final has been
            // processed, require_open changes this error to ChannelClosed.
            if matches!(
                control.push_audio(u64::MAX, vec![]),
                Err(Error::ChannelClosed)
            ) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("PULL callback did not process its final delta");
}

async fn run_case(case: &str) {
    let root = PathBuf::from(std::env::var(ROOT_ENV).expect("private root env"));
    assert_eq!(
        std::env::var("ORCHARD_CACHE_ROOT").unwrap(),
        root.join("cache").to_str().unwrap()
    );
    let backend = FakePie::start(case, &root);
    let mut ipc = IPCClient::new();
    ipc.connect().unwrap();
    let ipc = Arc::new(ipc);
    if case == "request_cancellation_scope" {
        let mut other = IPCClient::new();
        other.connect().unwrap();
        let other = Arc::new(other);
        let first_id = ipc.next_request_id();
        let other_id = other.next_request_id();
        assert_eq!(first_id, other_id, "Request IDs are local to each client");
        let prompt = PromptPayload {
            prompt: "A private cancellation wire fixture".into(),
            max_generated_tokens: 1,
            ..Default::default()
        };
        let (_, _first_events) = ipc
            .send_batch_request(first_id, "first", "/unused", std::slice::from_ref(&prompt))
            .unwrap();
        let (_, _other_events) = other
            .send_batch_request(other_id, "other", "/unused", &[prompt])
            .unwrap();
        tokio::time::timeout(DEADLINE, async {
            while backend.state.opens.lock().unwrap().len() != 2 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("Both clients must reach the actual request socket");
        let channels: Vec<_> = ["first", "other"]
            .iter()
            .map(|model| {
                backend
                    .state
                    .opens
                    .lock()
                    .unwrap()
                    .iter()
                    .find(|request| request["model_id"] == *model)
                    .unwrap()["response_channel_id"]
                    .as_u64()
                    .unwrap()
            })
            .collect();
        assert_ne!(channels[0], channels[1]);
        let registry = Arc::new(ModelRegistry::new().unwrap());
        for (ipc, request_id, channel) in
            [(ipc, first_id, channels[0]), (other, other_id, channels[1])]
        {
            Client::new(ipc, Arc::clone(&registry))
                .cancel_request(request_id)
                .await
                .unwrap();
            let commands = backend.state.commands.lock().unwrap();
            let command = commands.last().unwrap();
            assert_eq!(command["request_id"], request_id);
            assert_eq!(command["response_channel_id"], channel);
        }
        assert_eq!(backend.state.command_count("cancel_request"), 2);
        return;
    }
    let registry = ModelRegistry::new().unwrap();
    registry.set_ipc_client(Arc::clone(&ipc)).await;
    let model_id = descriptor(&root);
    let options = DuplexOptions {
        max_pending_frames: 3,
        ..Default::default()
    };
    if case == "wire_and_lifecycle" {
        let request = orchard::ModelLoadRequest {
            model: model_id.clone(),
            options: orchard::ModelLoadOptions::Duplex(options.clone()),
        };
        let (first, second) = tokio::join!(
            registry.ensure_prepared(&request),
            registry.ensure_prepared(&request)
        );
        assert_eq!(first.unwrap().model_id, second.unwrap().model_id);
        assert_eq!(backend.state.command_count("load_model"), 1);
        assert!(
            backend.state.opens.lock().unwrap().is_empty(),
            "Preparation must not open a speech session"
        );
        registry
            .model_operations(&request)
            .await
            .unwrap()
            .require(&["duplex", "speech_reference"])
            .unwrap();
    }
    let opened = tokio::time::timeout(DEADLINE, registry.duplex(&model_id, options))
        .await
        .expect("open deadline");
    if case == "generic_open_error" || case == "bad_geometry" || case == "bad_channels" {
        let error = match opened {
            Err(error) => error.to_string(),
            Ok(_) => panic!("Invalid engine reply was accepted"),
        };
        assert!(
            error.contains(if case == "generic_open_error" {
                "fixture open rejected"
            } else if case == "bad_channels" {
                "mono"
            } else {
                "geometry"
            }),
            "{error}"
        );
        return;
    }
    let mut session = opened.unwrap();
    let control = session.control();
    assert!(matches!(
        next(&mut session).await,
        DuplexEvent::Ready {
            sample_rate: 24_000,
            frame_samples: 1920,
            epoch: 0,
            ..
        }
    ));
    assert!(control.supports_reference());
    assert_eq!(backend.state.command_count("load_model"), 1);
    assert_eq!(backend.state.opens.lock().unwrap().len(), 1);

    match case {
        "grounded_reply"
        | "grounded_reply_unsupported"
        | "grounded_reply_legacy"
        | "grounded_reply_epoch_race"
        | "grounded_reply_bad_ack" => {
            let supported = !matches!(case, "grounded_reply_unsupported" | "grounded_reply_legacy");
            assert_eq!(control.supports_grounded_response(), supported);
            assert!(control.grounded_reply(" ".into(), 0).is_err());
            assert!(control.grounded_reply("é".repeat(4097), 0).is_err());
            assert_eq!(backend.state.command_count("duplex_reply"), 0);
            let result = control.grounded_reply(
                "The package arrives Friday; it is delayed by one day.".into(),
                0,
            );
            if case == "grounded_reply" {
                assert_eq!(result.unwrap(), 1);
                assert_eq!(backend.state.command_count("duplex_reply"), 1);
                assert_eq!(backend.state.command_count("duplex_reference"), 0);
                assert_eq!(backend.state.command_count("duplex_speak"), 0);
                assert!(
                    tokio::time::timeout(Duration::from_millis(40), session.next_event())
                        .await
                        .is_err(),
                    "Grounded response ACK must not fabricate audio or reference application"
                );
                assert_eq!(
                    control
                        .reference("Background context only".into(), 0)
                        .unwrap(),
                    2
                );
                assert_eq!(backend.state.command_count("duplex_reference"), 1);
                assert_eq!(backend.state.command_count("duplex_reply"), 1);
                assert_eq!(control.interrupt().unwrap(), 1);
                assert!(control.grounded_reply("Old context".into(), 0).is_err());
                assert_eq!(backend.state.command_count("duplex_reply"), 1);
            } else {
                assert!(result.is_err());
                assert_eq!(
                    backend.state.command_count("duplex_reply"),
                    usize::from(supported)
                );
                assert_eq!(
                    control.epoch(),
                    u64::from(case == "grounded_reply_epoch_race")
                );
            }
            control.close().unwrap();
            terminal_events(&mut session).await;
        }
        "management_lock_isolation" => {
            // A generic management exchange can hold this lock throughout a
            // slow model load. Duplex controls must use an independent route.
            let generic_socket = ipc.management_socket();
            let generic_guard = generic_socket.lock().unwrap();
            let started = std::time::Instant::now();
            assert_eq!(control.interrupt().unwrap(), 1);
            control.close().unwrap();
            assert!(started.elapsed() < Duration::from_secs(1));
            drop(generic_guard);
            terminal_events(&mut session).await;
            backend.state.wait_commands("duplex_close", 1);
        }
        "wire_and_lifecycle" => {
            control.push_audio(1, vec![0.25; FRAME_SAMPLES]).unwrap();
            backend.state.wait_commands("duplex_input", 1);
            backend.state.audio(0, 1, 0.125);
            match next(&mut session).await {
                DuplexEvent::Audio {
                    epoch: 0,
                    sequence: 1,
                    pcm,
                    compute_ms,
                    queue_ms,
                } => {
                    assert_eq!(pcm.len(), FRAME_SAMPLES);
                    assert!(pcm.iter().all(|value| *value == 0.125));
                    assert_eq!(compute_ms, 7.5);
                    assert_eq!(queue_ms, 1.25);
                }
                event => panic!("Expected native nonzero audio, got {event:?}"),
            }
            let version = control
                .reference("A factual backbone reference".into(), 0)
                .unwrap();
            assert_eq!(version, 1);
            assert!(
                tokio::time::timeout(Duration::from_millis(40), session.next_event())
                    .await
                    .is_err(),
                "Reference ACK must not invent an applied event"
            );
            backend.state.emit(
                "duplex.reference_queued",
                0,
                json!({"reference_version":version}),
                None,
                None,
                false,
            );
            assert!(matches!(
                next(&mut session).await,
                DuplexEvent::ReferenceQueued {
                    epoch: 0,
                    version: 1
                }
            ));
            backend.state.emit(
                "duplex.reference_applied",
                0,
                json!({"reference_version":version,"steps":4,
                "reference_remaining_steps":3,"encode_ms":12.5}),
                None,
                None,
                false,
            );
            assert!(
                matches!(next(&mut session).await, DuplexEvent::ReferenceApplied { epoch:0, version:1, steps:4, encode_ms } if encode_ms == 12.5)
            );
            // One old frame is already queued; another arrives after the ACK.
            backend.state.audio(0, 2, 0.25);
            assert_eq!(control.interrupt().unwrap(), 1);
            backend.state.audio(0, 3, 0.375);
            backend
                .state
                .emit("duplex.interrupted", 1, json!({}), None, None, false);
            backend.state.audio(1, 4, 0.5);
            assert!(matches!(
                next(&mut session).await,
                DuplexEvent::Interrupted { epoch: 1 }
            ));
            assert!(matches!(
                next(&mut session).await,
                DuplexEvent::Audio {
                    epoch: 1,
                    sequence: 4,
                    ..
                }
            ));
            backend.state.emit(
                "duplex.retrieval_requested",
                1,
                json!({"sequence":4,"input_sequence":123,"model_step":987}),
                None,
                None,
                false,
            );
            assert!(matches!(
                next(&mut session).await,
                DuplexEvent::RetrievalRequested {
                    epoch: 1,
                    sequence: 4,
                    input_sequence: Some(123),
                    model_step: Some(987),
                }
            ));
            // Native clock ticks may use synthetic silence; old engines send
            // neither correlation field. Neither may inherit the last input.
            for fields in [
                json!({"sequence":4,"input_sequence":null,"model_step":988}),
                json!({"sequence":4}),
            ] {
                let expected_step = fields["model_step"].as_u64();
                backend
                    .state
                    .emit("duplex.retrieval_requested", 1, fields, None, None, false);
                let event = next(&mut session).await;
                assert!(matches!(&event, DuplexEvent::RetrievalRequested {
                    epoch: 1, sequence: 4, input_sequence: None, model_step,
                } if *model_step == expected_step));
                let forwarded = serde_json::to_value(&event).unwrap();
                assert!(forwarded["input_sequence"].is_null());
                assert_eq!(forwarded["model_step"].as_u64(), expected_step);
            }
            assert_eq!(control.reset().unwrap(), 2);
            backend.state.emit(
                "duplex.reset",
                2,
                json!({"reason":"client_reset"}),
                None,
                None,
                false,
            );
            assert!(
                matches!(next(&mut session).await, DuplexEvent::Reset { epoch:2, reason } if reason == "client_reset")
            );
            control.close().unwrap();
            let events = terminal_events(&mut session).await;
            assert!(
                !events
                    .iter()
                    .any(|e| matches!(e, DuplexEvent::Error { .. })),
                "{events:?}"
            );
            backend.state.wait_commands("duplex_close", 1);
        }
        "bounded_audio" => {
            for sequence in 1..=100 {
                backend.state.audio(0, sequence, 0.125);
            }
            backend.state.metrics(0, 100);
            backend.state.closed(0, "client_close");
            wait_closed(&control).await;
            let events = terminal_events(&mut session).await;
            let sequences: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    DuplexEvent::Audio { sequence, .. } => Some(*sequence),
                    _ => None,
                })
                .collect();
            assert_eq!(sequences, (37..=100).collect::<Vec<_>>());
            assert!(events
                .iter()
                .any(|event| matches!(event, DuplexEvent::Metrics { metrics }
                if metrics.output_frames == 100 && metrics.dropped_output_frames == 36)));
        }
        "bounded_input" => {
            control.push_audio(1, vec![0.25; FRAME_SAMPLES]).unwrap();
            backend.state.wait_commands("duplex_input", 1);
            let dropped: u64 = (2..=20)
                .map(|sequence| {
                    control
                        .push_audio(sequence, vec![0.25; FRAME_SAMPLES])
                        .unwrap()
                        .dropped_frames
                })
                .sum();
            assert_eq!(dropped, 16);
            *backend.state.gate.lock().unwrap() = false;
            backend.state.gate_wake.notify_all();
            backend.state.wait_commands("duplex_input", 4);
            let sequences: Vec<_> = backend
                .state
                .commands
                .lock()
                .unwrap()
                .iter()
                .filter(|c| c["type"] == "duplex_input")
                .map(|c| c["sequence"].as_u64().unwrap())
                .collect();
            assert_eq!(sequences, vec![1, 18, 19, 20]);
            control.close().unwrap();
            terminal_events(&mut session).await;
        }
        "bounded_control_events" => {
            for sequence in 1..=257 {
                backend.state.text(0, sequence, "bounded");
            }
            wait_closed(&control).await;
            let events = terminal_events(&mut session).await;
            assert!(events.len() <= 2);
            assert!(events.iter().any(|e| matches!(e, DuplexEvent::Error { message } if message.contains("bounded event queue"))));
            backend.state.wait_commands("duplex_close", 1);
        }
        "missing_metrics" => {
            backend.state.emit(
                "duplex.metrics",
                0,
                json!({"input_frames":1}),
                None,
                None,
                false,
            );
            let events = terminal_events(&mut session).await;
            assert!(!events
                .iter()
                .any(|e| matches!(e, DuplexEvent::Metrics { .. })));
            assert!(events
                .iter()
                .any(|e| matches!(e, DuplexEvent::Error { message }
                if message.contains("metrics missing"))));
            backend.state.wait_commands("duplex_close", 1);
        }
        "wrong_request" => {
            let route = backend.state.route();
            backend.state.send(&route, json!({"request_id":route.request_id + 1,"is_final_delta":false,
                "modal_type":"audio","modal_decoder_id":"moshi.duplex","modal_event":"duplex.text",
                "modal_metadata_json":json!({"duplex_version":1,"epoch":0,"sequence":1}).to_string(),
                "content":"This must not reach the current session"}));
            let events = terminal_events(&mut session).await;
            assert!(!events
                .iter()
                .any(|e| matches!(e, DuplexEvent::TextDelta { .. })));
            assert!(events
                .iter()
                .any(|e| matches!(e, DuplexEvent::Error { message }
                if message.contains("different request"))));
            backend.state.wait_commands("duplex_close", 1);
        }
        "malformed_pull_json" => {
            let route = backend.state.route();
            let malformed = format!("resp:{:x}:{{this is not JSON", route.channel_id);
            route.socket.send(malformed.as_bytes()).unwrap();
            let events = terminal_events(&mut session).await;
            assert!(events
                .iter()
                .any(|e| matches!(e, DuplexEvent::Error { message }
                if message.contains("Invalid PIE duplex response"))));
            assert!(!events
                .iter()
                .any(|e| matches!(e, DuplexEvent::Audio { .. })));
            // A locally synthesized error must not pretend PIE sent a terminal
            // delta. The engine is still alive and must receive an explicit close.
            assert_eq!(
                std::fs::read_to_string(root.join("cache/engine.pid")).unwrap(),
                std::process::id().to_string()
            );
            backend.state.wait_commands("duplex_close", 1);
            assert_eq!(backend.state.command_count("duplex_close"), 1);
        }
        "rollover_ack_race" => {
            assert!(control.is_autonomous());
            for sequence in 1..=3 {
                control
                    .push_audio(sequence, vec![0.25; FRAME_SAMPLES])
                    .unwrap();
                backend
                    .state
                    .wait_commands("duplex_input", sequence as usize);
                tokio::time::timeout(DEADLINE, async {
                    while backend.state.input_acks.load(Ordering::Acquire) < sequence {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                })
                .await
                .expect("captured frame was not ACKed before simulated rollover");
                backend.state.audio(0, sequence, 0.125);
                assert!(matches!(
                    next(&mut session).await,
                    DuplexEvent::Audio { epoch: 0, .. }
                ));
            }
            // PIE has reset its bounded model timeline, but the reset event is
            // deliberately delayed on PULL. The next old-epoch microphone ACK
            // must update the SDK epoch without terminating this live request.
            backend.state.epoch.store(1, Ordering::Release);
            control.push_audio(4, vec![0.25; FRAME_SAMPLES]).unwrap();
            backend.state.wait_commands("duplex_input", 4);
            control.push_audio(5, vec![0.25; FRAME_SAMPLES]).unwrap();
            control.push_audio(6, vec![0.25; FRAME_SAMPLES]).unwrap();
            *backend.state.gate.lock().unwrap() = false;
            backend.state.gate_wake.notify_all();
            tokio::time::timeout(DEADLINE, async {
                while control.epoch() != 1 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .expect("stale ACK did not advance SDK epoch");
            assert!(
                tokio::time::timeout(Duration::from_millis(40), session.next_event())
                    .await
                    .is_err(),
                "A stale microphone ACK must not fabricate Reset or terminate the session"
            );
            backend.state.emit(
                "duplex.reset",
                1,
                json!({"reason":"timeline_rollover","rollover_index":1,
                "previous_model_step":3,"max_steps":3}),
                None,
                None,
                false,
            );
            assert!(
                matches!(next(&mut session).await, DuplexEvent::Reset { epoch:1, reason }
                if reason == "timeline_rollover")
            );
            control.push_audio(7, vec![0.25; FRAME_SAMPLES]).unwrap();
            backend.state.wait_commands("duplex_input", 5);
            let sequences: Vec<_> = backend
                .state
                .commands
                .lock()
                .unwrap()
                .iter()
                .filter(|command| command["type"] == "duplex_input")
                .map(|command| command["sequence"].as_u64().unwrap())
                .collect();
            assert_eq!(sequences, vec![1, 2, 3, 4, 7]);
            backend.state.emit(
                "duplex.audio",
                1,
                json!({"sequence":4,"input_sequence":7,"model_step":1,
                "speaking":true,"compute_ms":7.5,"queue_ms":1.25}),
                None,
                Some(0.25),
                false,
            );
            assert!(matches!(
                next(&mut session).await,
                DuplexEvent::Audio {
                    epoch: 1,
                    sequence: 4,
                    ..
                }
            ));
            backend.state.metrics(1, 4);
            assert!(
                matches!(next(&mut session).await, DuplexEvent::Metrics { metrics }
                if metrics.input_frames == 4 && metrics.dropped_input_frames == 3)
            );
            control.close().unwrap();
            let events = terminal_events(&mut session).await;
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, DuplexEvent::Error { .. })),
                "{events:?}"
            );
        }
        "engine_death" => {
            std::fs::remove_file(root.join("cache/engine.pid")).unwrap();
            let events = terminal_events(&mut session).await;
            assert!(events.iter().any(
                |e| matches!(e, DuplexEvent::Error { message } if message.contains("engine"))
            ));
        }
        other => panic!("Unknown case {other}"),
    }
}
