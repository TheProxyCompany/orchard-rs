//! Hermetic wire/lifecycle checks. No native library, model hydration or GPU is used.
#![cfg(feature = "diarization")]
use base64::{engine::general_purpose::STANDARD, Engine as _};
use nng::{options::Options, Protocol, Socket};
use orchard::diarization::{
    DiarizationDevice, DiarizationEvent, DiarizationOptions, DiarizationSession,
};
use orchard::ipc::{client::IPCClient, endpoints};
use orchard::ModelRegistry;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

const DEADLINE: Duration = Duration::from_secs(6);

#[test]
fn native_diarization_ipc_contract() {
    if let Ok(case) = std::env::var("ORCHARD_DIAR_TEST_CASE") {
        let root = PathBuf::from(std::env::var_os("ORCHARD_DIAR_TEST_ROOT").unwrap());
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
            .block_on(run_case(&case, &root));
        return;
    }
    for case in [
        "wire_finish",
        "queue_full",
        "malformed_ack",
        "lost_ack",
        "nonfinal_error",
        "bad_ready",
        "bad_probabilities",
        "engine_death",
        "variant",
        "descriptor_mismatch",
        "close",
    ] {
        let root = tempfile::Builder::new()
            .prefix("diar-ipc-")
            .tempdir_in("/tmp")
            .unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "native_diarization_ipc_contract", "--nocapture"])
            .env("ORCHARD_DIAR_TEST_CASE", case)
            .env("ORCHARD_DIAR_TEST_ROOT", root.path())
            .env("ORCHARD_CACHE_ROOT", root.path().join("cache"))
            .env("ORCHARD_IPC_ROOT", root.path().join("ipc"))
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{case}:\n{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        println!("native diarization fake-PIE case passed: {case}");
    }
}

#[derive(Clone)]
struct Route {
    socket: Arc<Socket>,
    request: u64,
    channel: u64,
    model: String,
    session: String,
}
struct State {
    case: String,
    stop: AtomicBool,
    route: Mutex<Option<Route>>,
    commands: Mutex<Vec<Value>>,
    loads: Mutex<Vec<Value>>,
    wake: Condvar,
    input_calls: AtomicU64,
    last_sequence: Mutex<Option<u64>>,
}
impl State {
    fn route(&self) -> Route {
        self.route.lock().unwrap().as_ref().unwrap().clone()
    }
    fn send(&self, value: Value) {
        let route = self.route();
        let payload = format!("resp:{:x}:{value}", route.channel);
        // Closing a client deliberately ends its exclusive route before the
        // fake server necessarily finishes sending the terminal message.
        let _ = route.socket.send(payload.as_bytes());
    }
    fn emit(&self, kind: &str, mut metadata: Value, final_delta: bool) {
        let route = self.route();
        metadata["diarization_version"] = 1.into();
        metadata["session_id"] = route.session.into();
        self.send(
            json!({"request_id":route.request,"is_final_delta":final_delta,
            "modal_type":"audio","modal_event":format!("diarization.{kind}"),
            "modal_decoder_id":"nemotron.diarization","modal_mime_type":"application/json",
            "modal_metadata_json":metadata.to_string()}),
        );
    }
    fn update(&self, finished: bool, probability: f64) {
        let route = self.route();
        self.emit("update", json!({"frame_start":0,"frame_seconds":0.01,
            "probabilities":[[probability,0.2,0.3,0.4,0.5,0.6,0.7,0.8]],
            "segments":[{"speaker_id":format!("{}:speaker-1",route.session),"speaker_index":1,
                "start_seconds":0.0,"end_seconds":0.01,"mean_probability":0.1,"provisional":!finished}],
            "replace_segments_from_seconds":0.0,"compute_ms":2.5,"audio_seconds":0.08,"finished":finished}), false);
    }
    fn closed(&self) {
        self.emit("closed", json!({}), true);
    }
    fn wait_commands(&self, kind: &str, count: usize) {
        let commands = self.commands.lock().unwrap();
        let (_commands, timeout) = self
            .wake
            .wait_timeout_while(commands, DEADLINE, |commands| {
                commands.iter().filter(|row| row["type"] == kind).count() < count
            })
            .unwrap();
        assert!(!timeout.timed_out(), "waiting for {kind} {count}");
    }
}
struct FakePie {
    state: Arc<State>,
    request: Arc<Socket>,
    management: Arc<Socket>,
    _publish: Socket,
    workers: Vec<JoinHandle<()>>,
}
impl FakePie {
    fn start(case: &str, root: &Path) -> Self {
        std::fs::create_dir_all(root.join("cache")).unwrap();
        std::fs::write(
            root.join("cache/engine.pid"),
            std::process::id().to_string(),
        )
        .unwrap();
        assert!(endpoints::ipc_root().starts_with(root.canonicalize().unwrap()));
        let socket = |protocol, url: String| {
            let socket = Socket::new(protocol).unwrap();
            socket
                .set_opt::<nng::options::RecvTimeout>(Some(Duration::from_millis(20)))
                .unwrap();
            socket
                .set_opt::<nng::options::SendTimeout>(Some(Duration::from_millis(300)))
                .unwrap();
            socket.listen(&url).unwrap();
            socket
        };
        let request = Arc::new(socket(Protocol::Pull0, endpoints::request_url()));
        let management = Arc::new(socket(Protocol::Rep0, endpoints::management_url()));
        let publish = socket(Protocol::Pub0, endpoints::response_url());
        let state = Arc::new(State {
            case: case.into(),
            stop: AtomicBool::new(false),
            route: Mutex::new(None),
            commands: Mutex::new(vec![]),
            loads: Mutex::new(vec![]),
            wake: Condvar::new(),
            input_calls: AtomicU64::new(0),
            last_sequence: Mutex::new(None),
        });
        let mut workers = vec![];
        {
            let state = Arc::clone(&state);
            let request = Arc::clone(&request);
            workers.push(std::thread::spawn(move || {
                while !state.stop.load(Ordering::Acquire) {
                    let message = match request.recv() {
                        Ok(value)=>value, Err(nng::Error::TimedOut)=>continue,
                        Err(_) if state.stop.load(Ordering::Acquire)=>break, Err(error)=>panic!("fake request: {error}"),
                    };
                    let bytes=message.as_slice(); let length=u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
                    let metadata:Value=serde_json::from_slice(&bytes[4..4+length]).unwrap();
                    assert_eq!(metadata["request_type"],11); assert_eq!(metadata["response_transport"],"pull_v1");
                    assert_eq!(metadata["prompts"].as_array().unwrap().len(),1);
                    assert_eq!(metadata["prompts"][0]["text_size"],0);
                    let options:Value=serde_json::from_str(metadata["prompts"][0]["modal_options_json"].as_str().unwrap()).unwrap();
                    assert_eq!(options["diarization_version"],1);
                    for key in ["device","chunk_frames","fifo_frames","speaker_cache_frames"] { assert!(options.get(key).is_none()); }
                    let channel=metadata["response_channel_id"].as_u64().unwrap();
                    let response=Socket::new(Protocol::Push0).unwrap();
                    response.set_opt::<nng::options::SendTimeout>(Some(Duration::from_millis(300))).unwrap();
                    response.set_opt::<nng::options::SendBufferSize>(0).unwrap();
                    response.dial(&endpoints::pull_response_url(channel)).unwrap();
                    let route=Route{socket:Arc::new(response),request:metadata["request_id"].as_u64().unwrap(),channel,
                        model:metadata["model_id"].as_str().unwrap().into(),session:options["session_id"].as_str().unwrap().into()};
                    *state.route.lock().unwrap()=Some(route.clone());
                    let rate=options["sample_rate"].as_u64().unwrap();
                    state.emit("ready",json!({"model_id":route.model,"speakers":if state.case=="bad_ready" {4}else{8},
                        "sample_rate":rate,"frame_samples":rate*8/100,"output_frame_seconds":0.01}),false);
                }
            }));
        }
        {
            let state = Arc::clone(&state);
            let management = Arc::clone(&management);
            workers.push(std::thread::spawn(move||{
                while !state.stop.load(Ordering::Acquire) {
                    let message=match management.recv(){Ok(value)=>value,Err(nng::Error::TimedOut)=>continue,
                        Err(_) if state.stop.load(Ordering::Acquire)=>break,Err(error)=>panic!("fake management: {error}")};
                    let command:Value=serde_json::from_slice(&message).unwrap();let kind=command["type"].as_str().unwrap();
                    state.commands.lock().unwrap().push(command.clone());state.wake.notify_all();
                    let reply=|value:Value|{let _=management.send(value.to_string().as_bytes());};
                    if kind=="load_model" {
                        let config:Value=serde_json::from_slice(&std::fs::read(Path::new(command["model_path"].as_str().unwrap()).join("config.json")).unwrap()).unwrap();
                        assert_eq!(config["model_type"],"nemotron3_diarization");
                        assert_eq!(config["diarization_schema_version"],1);
                        assert!(config.get("library").is_none());
                        state.loads.lock().unwrap().push(config);
                        reply(json!({"status":"ok","data":{"load_model":{"runtime_started":true,
                            "bound_runtime_id":command["canonical_id"],"capabilities":{"speaker_diarization":[8,100]},"minimum_memory_bytes":107012128}}}));
                        continue;
                    }
                    if kind=="unload_model" {reply(json!({"status":"ok"}));continue;}
                    let route=state.route();
                    assert_eq!(command["model_id"],route.model);assert_eq!(command["request_id"],route.request);
                    assert_eq!(command["response_channel_id"],route.channel);assert!(command.get("epoch").is_none());
                    let mut data=json!({"queue_depth":0,"session_state":"open","missing_frames":0});
                    if kind=="diarization_input" {
                        let call=state.input_calls.fetch_add(1,Ordering::AcqRel);
                        if state.case=="queue_full" && call==0 {
                            reply(json!({"status":"error","message":"Diarization input queue is full",
                                "data":{"diarization":{"error_code":"queue_full","queue_depth":2,"session_state":"open"}}}));continue;
                        }
                        let bytes=STANDARD.decode(command["pcm_f32_b64"].as_str().unwrap()).unwrap();
                        assert_eq!(bytes.len(),1920*4);
                        let sequence=command["sequence"].as_u64().unwrap();
                        let mut last=state.last_sequence.lock().unwrap();
                        let missing=last.map_or(0,|last|sequence-last-1);*last=Some(sequence);drop(last);
                        data["accepted_sequence"]=sequence.into();data["missing_frames"]=missing.into();
                        if state.case=="lost_ack" {std::thread::sleep(Duration::from_millis(2250));}
                        if state.case=="malformed_ack" {data=json!({});}
                        reply(json!({"status":"ok","data":{"diarization":data}}));
                        if state.case=="nonfinal_error" {
                            state.send(json!({"request_id":route.request,"is_final_delta":false,"error_message":"fixture non-final engine fault"}));
                        } else if state.case=="bad_probabilities" {state.update(false,2.0);}
                        continue;
                    }
                    if kind=="diarization_finish" {
                        data["session_state"]="finishing".into();
                        reply(json!({"status":"ok","data":{"diarization":data}}));
                        state.update(true,0.1);
                        state.emit("metrics",json!({"input_frames":2,"missing_input_frames":2,"output_frames":1,
                            "compute_ms_total":5.0,"compute_ms_max":2.5,"elapsed_ms":160.0,"model_weight_bytes":107012128}),false);
                        state.closed();continue;
                    }
                    assert_eq!(kind,"diarization_close");data["session_state"]="closing".into();
                    reply(json!({"status":"ok","data":{"diarization":data}}));state.closed();
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
        self.request.close();
        self.management.close();
        if let Some(route) = self.state.route.lock().unwrap().as_ref() {
            route.socket.close();
        }
        for worker in self.workers.drain(..) {
            let result = worker.join();
            if !std::thread::panicking() {
                result.unwrap();
            }
        }
    }
}

async fn terminal(session: &mut DiarizationSession) -> Vec<DiarizationEvent> {
    let mut events = vec![];
    while let Some(event) = tokio::time::timeout(DEADLINE, session.next_event())
        .await
        .unwrap()
    {
        events.push(event);
        assert!(events.len() <= 64);
    }
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, DiarizationEvent::Closed))
            .count(),
        1
    );
    events
}
async fn run_case(case: &str, root: &Path) {
    let backend = FakePie::start(case, root);
    let profile = orchard::diarization::architecture().unwrap();
    let file = root.join(&profile.model_file);
    std::fs::File::create(&file)
        .unwrap()
        .set_len(profile.size_bytes)
        .unwrap();
    let mut model = file.to_string_lossy().into_owned();
    if case == "descriptor_mismatch" {
        let dir = root.join("explicit-descriptor");
        std::fs::create_dir_all(&dir).unwrap();
        let config = json!({"model_type":"nemotron3_diarization","diarization_schema_version":1,"model_file":model,
            "model_size_bytes":profile.size_bytes,"model_sha256":profile.sha256,"device":"cpu","chunk_frames":6,
            "right_context_frames":2,"fifo_frames":264,"speaker_cache_frames":264,"update_period_frames":222});
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        model = dir.to_string_lossy().into_owned();
    }
    let mut ipc = IPCClient::new();
    ipc.connect().unwrap();
    let registry = ModelRegistry::new().unwrap();
    registry.set_ipc_client(Arc::new(ipc)).await;
    let mut options = DiarizationOptions {
        max_pending_frames: 2,
        session_id: Some("fixture".into()),
        ..Default::default()
    };
    if case == "variant" {
        registry.ensure_loaded(&model).await.unwrap();
        options.device = DiarizationDevice::Metal;
    }
    let opened = tokio::time::timeout(DEADLINE, registry.diarization(&model, options.clone()))
        .await
        .unwrap();
    if case == "bad_ready" {
        assert!(opened.is_err());
        backend.state.wait_commands("diarization_close", 1);
        return;
    }
    let mut session = opened.unwrap();
    let control = session.control();
    match case {
        "wire_finish" => {
            assert_eq!(
                control
                    .push_audio(100, vec![0.25; 1920])
                    .unwrap()
                    .missing_frames,
                0
            );
            assert!(control.push_audio(100, vec![0.25; 1920]).is_err());
            assert_eq!(
                control
                    .push_audio(103, vec![0.25; 1920])
                    .unwrap()
                    .missing_frames,
                2
            );
            assert!(control.push_audio(160, vec![0.25; 1920]).is_err());
            control.finish().await.unwrap();
            assert!(control.push_audio(104, vec![0.25; 1920]).is_err());
            let events = terminal(&mut session).await;
            assert!(events.iter().any(|event|matches!(event,DiarizationEvent::Update{finished:true,probabilities,..} if probabilities.len()==1)));
            assert!(!events
                .iter()
                .any(|event| matches!(event, DiarizationEvent::Error { .. })));
            session.shutdown().await.unwrap();
        }
        "queue_full" => {
            assert!(control
                .push_audio(7, vec![0.0; 1920])
                .unwrap_err()
                .to_string()
                .contains("queue is full"));
            assert_eq!(
                control
                    .push_audio(7, vec![0.0; 1920])
                    .unwrap()
                    .missing_frames,
                0
            );
            assert_eq!(
                control
                    .push_audio(9, vec![0.0; 1920])
                    .unwrap()
                    .missing_frames,
                1
            );
            control.finish().await.unwrap();
            assert!(!terminal(&mut session)
                .await
                .iter()
                .any(|event| matches!(event, DiarizationEvent::Error { .. })));
        }
        "malformed_ack" | "lost_ack" => {
            assert!(control.push_audio(0, vec![0.0; 1920]).is_err());
            assert!(control.push_audio(1, vec![0.0; 1920]).is_err());
            backend.state.wait_commands("diarization_close", 1);
            assert!(terminal(&mut session)
                .await
                .iter()
                .any(|event| matches!(event, DiarizationEvent::Error { .. })));
        }
        "nonfinal_error" | "bad_probabilities" => {
            control.push_audio(0, vec![0.0; 1920]).unwrap();
            assert!(terminal(&mut session)
                .await
                .iter()
                .any(|event| matches!(event, DiarizationEvent::Error { .. })));
            backend.state.wait_commands("diarization_close", 1);
        }
        "engine_death" => {
            std::fs::remove_file(root.join("cache/engine.pid")).unwrap();
            assert!(terminal(&mut session)
                .await
                .iter()
                .any(|event| matches!(event, DiarizationEvent::Error { .. })));
        }
        "variant" => {
            {
                let loads = backend.state.loads.lock().unwrap();
                assert_eq!(loads.len(), 2);
                assert_eq!(loads[0]["device"], "cpu");
                assert_eq!(loads[1]["device"], "metal");
            }
            control.finish().await.unwrap();
            terminal(&mut session).await;
            registry.unload_diarization(&model).await.unwrap();
            backend.state.wait_commands("unload_model", 2);
        }
        "descriptor_mismatch" => {
            control.finish().await.unwrap();
            terminal(&mut session).await;
            options.device = DiarizationDevice::Metal;
            let error = registry.diarization(&model, options).await.err().unwrap();
            assert!(error.to_string().contains("does not match"));
            assert_eq!(backend.state.loads.lock().unwrap().len(), 1);
        }
        "close" => {
            control.close();
            let events = terminal(&mut session).await;
            assert!(!events
                .iter()
                .any(|event| matches!(event, DiarizationEvent::Error { .. })));
            backend.state.wait_commands("diarization_close", 1);
            session.shutdown().await.unwrap();
        }
        _ => unreachable!(),
    }
    drop(session);
    drop(control);
    drop(registry);
}
