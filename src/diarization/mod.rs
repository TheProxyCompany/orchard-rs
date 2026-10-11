//! Thin IPC for NVIDIA Nemotron3 diarization. PIE exclusively owns its model,
//! native library, AOSC/FIFO state and inference. Speaker IDs remain anonymous.
mod model;
pub(crate) use model::{engine_source, runtime_key};

use crate::ipc::client::{IPCClient, ResponseDelta, StreamResponseRoute};
use crate::ipc::serialization::PromptPayload;
use crate::{Error, ModelInfo, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

pub const DEFAULT_MODEL: &str = "nvidia/Nemotron-3-Diarization";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiarizationArchitecture {
    pub sample_rate: u32,
    pub speakers: usize,
    pub output_frame_seconds: f64,
    pub encoder_frame_seconds: f64,
    pub chunk_frames: i32,
    pub right_context_frames: i32,
    pub left_context_frames: i32,
    pub fifo_frames: i32,
    pub speaker_cache_frames: i32,
    pub update_period_frames: i32,
    pub model_id: String,
    pub revision: String,
    pub model_file: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub runtime_revision: String,
}

pub fn architecture() -> Result<DiarizationArchitecture> {
    #[derive(Deserialize)]
    struct Profile {
        diarization: DiarizationArchitecture,
    }
    let profile =
        crate::formatter::embedded_profiles::find_embedded_profile("nemotron3_diarization")
            .ok_or_else(|| Error::FormatterProfileNotFound("nemotron3_diarization".into()))?;
    let profile: Profile = serde_yaml::from_str(profile.capabilities)
        .map_err(|e| Error::Other(format!("Nemotron Pantheon profile: {e}")))?;
    if profile.diarization.speakers != 8 || profile.diarization.output_frame_seconds != 0.01 {
        return Err(Error::Other(
            "Nemotron 3 profile must describe eight speakers at 10 ms".into(),
        ));
    }
    Ok(profile.diarization)
}

pub fn is_diarization_model(model_id: &str) -> bool {
    if model_id == DEFAULT_MODEL {
        return true;
    }
    if let Ok(bytes) = std::fs::read(Path::new(model_id).join("config.json")) {
        if serde_json::from_slice::<Value>(&bytes)
            .is_ok_and(|config| config["model_type"] == "nemotron3_diarization")
        {
            return true;
        }
    }
    let Ok(profile) = architecture() else {
        return false;
    };
    let path = Path::new(model_id);
    path.is_file()
        && path
            .file_name()
            .is_some_and(|name| name == profile.model_file.as_str())
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiarizationDevice {
    #[default]
    Cpu,
    Metal,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct DiarizationOptions {
    pub device: DiarizationDevice,
    /// Input may share Moshi's 24 kHz stream; NVIDIA resamples internally.
    pub sample_rate: u32,
    pub max_pending_frames: usize,
    pub chunk_frames: i32,
    pub right_context_frames: i32,
    pub fifo_frames: i32,
    pub speaker_cache_frames: i32,
    pub update_period_frames: i32,
    /// Optional unique ID supplied by the owner; reuse does not restore state.
    pub session_id: Option<String>,
}

impl Default for DiarizationOptions {
    fn default() -> Self {
        Self {
            device: DiarizationDevice::Cpu,
            sample_rate: 24_000,
            max_pending_frames: 32,
            chunk_frames: 6,
            right_context_frames: 2,
            fifo_frames: 264,
            speaker_cache_frames: 264,
            update_period_frames: 222,
            session_id: None,
        }
    }
}
impl DiarizationOptions {
    pub fn frame_samples(&self) -> usize {
        self.sample_rate as usize * 8 / 100
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if !(8_000..=96_000).contains(&self.sample_rate)
            || !self.sample_rate.is_multiple_of(100)
            || !(1..=64).contains(&self.max_pending_frames)
            || !(1..=340).contains(&self.chunk_frames)
            || !(1..=40).contains(&self.right_context_frames)
            || !(1..=512).contains(&self.fifo_frames)
            || !(64..=512).contains(&self.speaker_cache_frames)
            || self.update_period_frames < 1
            || self.update_period_frames > self.fifo_frames
        {
            return Err(Error::Other(
                "Invalid streaming diarization geometry or audio rate".into(),
            ));
        }
        if self.session_id.as_ref().is_some_and(|id| {
            id.is_empty()
                || id.len() > 128
                || !id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_:.".contains(c))
        }) {
            return Err(Error::Other(
                "Diarization session ID must be a bounded unique identifier".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpeakerSegment {
    pub speaker_id: String,
    pub speaker_index: u8,
    pub start_seconds: f64,
    pub end_seconds: f64,
    /// Mean model posterior on this segment, not confidence in a person's name.
    pub mean_probability: Option<f32>,
    pub provisional: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DiarizationEvent {
    Ready {
        session_id: String,
        model_id: String,
        speakers: usize,
        sample_rate: u32,
        frame_samples: usize,
        output_frame_seconds: f64,
    },
    Update {
        session_id: String,
        frame_start: u64,
        frame_seconds: f64,
        probabilities: Vec<[f32; 8]>,
        segments: Vec<SpeakerSegment>,
        replace_segments_from_seconds: f64,
        compute_ms: f64,
        audio_seconds: f64,
        finished: bool,
    },
    /// Missing input is represented by silence only to preserve the clock.
    /// Identity/transcription for this interval is unknown, never fabricated.
    Gap {
        session_id: String,
        start_seconds: f64,
        end_seconds: f64,
        missing_frames: u64,
    },
    Metrics {
        session_id: String,
        input_frames: u64,
        missing_input_frames: u64,
        output_frames: u64,
        compute_ms_total: f64,
        compute_ms_max: f64,
        elapsed_ms: f64,
        model_weight_bytes: u64,
    },
    Error {
        message: String,
    },
    Closed,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DiarizationAdmission {
    pub missing_frames: u64,
}

const CONTROL_TIMEOUT: Duration = Duration::from_secs(2);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_EVENTS: usize = 64;
const MAX_EVENT_BYTES: usize = 16 * 1024 * 1024;
pub const IPC_PROTOCOL_VERSION: u32 = 1;

#[derive(Default)]
struct AdmissionState {
    last_sequence: Option<u64>,
    finish_confirmed: bool,
}
#[derive(Default)]
struct OutputQueue {
    events: VecDeque<(DiarizationEvent, usize)>,
    bytes: usize,
    done: bool,
}
struct CommandFailure {
    error: Error,
    rejected: bool,
}
impl From<Error> for CommandFailure {
    fn from(error: Error) -> Self {
        Self {
            error,
            rejected: false,
        }
    }
}
struct Shared {
    ipc: Arc<IPCClient>,
    model_id: String,
    public_model_id: String,
    session_id: String,
    request_id: u64,
    channel_id: AtomicU64,
    sample_rate: u32,
    frame_samples: usize,
    ready: AtomicBool,
    closed: AtomicBool,
    finishing: AtomicBool,
    close_requested: AtomicBool,
    remote_final: AtomicBool,
    ended: AtomicBool,
    admission: Mutex<AdmissionState>,
    output: Mutex<OutputQueue>,
    output_ready: Notify,
    terminal: Notify,
    failure: Mutex<Option<String>>,
}
impl Shared {
    fn command(&self, name: &str, fields: Value) -> std::result::Result<Value, CommandFailure> {
        let mut command = json!({"type":name,"model_id":self.model_id,
            "request_id":self.request_id,"response_channel_id":self.channel_id.load(Ordering::Acquire)});
        if let Some(fields) = fields.as_object() {
            command
                .as_object_mut()
                .expect("command object")
                .extend(fields.clone());
        }
        let response = self
            .ipc
            .send_management_command_blocking(command, CONTROL_TIMEOUT)?;
        if response["status"] != "ok" {
            let message = response["message"]
                .as_str()
                .unwrap_or("PIE rejected diarization command")
                .to_owned();
            return Err(CommandFailure {
                error: Error::Other(message),
                rejected: matches!(
                    response["data"]["diarization"]["error_code"].as_str(),
                    Some("queue_full" | "invalid_sequence" | "invalid_frame" | "gap_too_large")
                ),
            });
        }
        let data = response["data"]["diarization"].clone();
        if !data.is_object()
            || data["queue_depth"].as_u64().is_none()
            || !matches!(
                data["session_state"].as_str(),
                Some("open" | "finishing" | "closing" | "closed")
            )
        {
            return Err(Error::Other("PIE diarization ACK has no valid typed state".into()).into());
        }
        Ok(data)
    }

    fn require_open(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) || self.finishing.load(Ordering::Acquire) {
            return Err(Error::ChannelClosed);
        }
        if !self.ready.load(Ordering::Acquire) {
            return Err(Error::ModelNotReady(
                "PIE diarization session is not ready".into(),
            ));
        }
        Ok(())
    }

    fn request_close(self: &Arc<Self>) {
        self.closed.store(true, Ordering::Release);
        if self.close_requested.swap(true, Ordering::AcqRel)
            || self.remote_final.load(Ordering::Acquire)
        {
            return;
        }
        let shared = Arc::clone(self);
        if let Err(error) = std::thread::Builder::new()
            .name("orchard-diarization-close".into())
            .spawn(move || {
                let _serial = shared
                    .admission
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                if !shared.remote_final.load(Ordering::Acquire) {
                    if let Err(error) = shared.command("diarization_close", json!({})) {
                        if !shared.remote_final.load(Ordering::Acquire) {
                            shared
                                .finish(Some(format!("Closing PIE diarization: {}", error.error)));
                        }
                    }
                }
            })
        {
            self.finish(Some(format!("Starting diarization close: {error}")));
        }
    }

    fn emit(self: &Arc<Self>, event: DiarizationEvent, bytes: usize) {
        let mut output = self
            .output
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if output.done {
            return;
        }
        if output.events.len() >= MAX_EVENTS - 2
            || output.bytes.saturating_add(bytes) > MAX_EVENT_BYTES
        {
            output.events.clear();
            output.bytes = 0;
            drop(output);
            self.finish(Some(
                "Diarization consumer is not draining its bounded event queue".into(),
            ));
            self.request_close();
            return;
        }
        output.bytes += bytes;
        output.events.push_back((event, bytes));
        drop(output);
        self.output_ready.notify_one();
    }

    fn finish(&self, failure: Option<String>) {
        if self.ended.swap(true, Ordering::AcqRel) {
            return;
        }
        self.closed.store(true, Ordering::Release);
        let mut output = self
            .output
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(message) = failure {
            *self
                .failure
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(message.clone());
            output
                .events
                .push_back((DiarizationEvent::Error { message }, 0));
        }
        output.events.push_back((DiarizationEvent::Closed, 0));
        output.done = true;
        drop(output);
        self.output_ready.notify_one();
        self.terminal.notify_waiters();
    }

    fn receive(self: &Arc<Self>, delta: ResponseDelta, from_engine: bool) {
        if delta.request_id != self.request_id {
            self.finish(Some(
                "PIE diarization response belongs to a different request".into(),
            ));
            self.request_close();
            return;
        }
        if delta.is_final_delta && from_engine {
            self.remote_final.store(true, Ordering::Release);
            self.terminal.notify_waiters();
        }
        if self.ended.load(Ordering::Acquire) {
            return;
        }
        if let Some(error) = delta.error.as_ref() {
            self.finish(Some(error.clone()));
            if !delta.is_final_delta || !from_engine {
                self.request_close();
            }
            return;
        }
        let bytes = delta.modal_metadata_json.as_ref().map_or(0, String::len);
        let result = decode_event(&delta, &self.session_id);
        match result {
            Ok(Some(mut event)) => {
                match &mut event {
                    DiarizationEvent::Ready {
                        model_id,
                        sample_rate,
                        frame_samples,
                        ..
                    } => {
                        if *model_id != self.model_id
                            || *sample_rate != self.sample_rate
                            || *frame_samples != self.frame_samples
                            || self.ready.swap(true, Ordering::AcqRel)
                        {
                            self.finish(Some(
                                "PIE returned incompatible diarization readiness".into(),
                            ));
                            self.request_close();
                            return;
                        }
                        *model_id = self.public_model_id.clone();
                    }
                    DiarizationEvent::Closed => {
                        if !delta.is_final_delta {
                            self.finish(Some(
                                "PIE diarization closed without a terminal lifecycle marker".into(),
                            ));
                            self.request_close();
                        } else {
                            self.finish(None);
                        }
                        return;
                    }
                    DiarizationEvent::Error { message } => {
                        self.finish(Some(message.clone()));
                        if !delta.is_final_delta {
                            self.request_close();
                        }
                        return;
                    }
                    _ if !self.ready.load(Ordering::Acquire) => {
                        self.finish(Some("PIE diarization emitted output before ready".into()));
                        self.request_close();
                        return;
                    }
                    _ => {}
                }
                self.emit(event, bytes);
            }
            Ok(None) => {}
            Err(error) => {
                self.finish(Some(error.to_string()));
                self.request_close();
                return;
            }
        }
        if delta.is_final_delta {
            self.finish(None);
        }
    }
}

fn decode_event(delta: &ResponseDelta, session_id: &str) -> Result<Option<DiarizationEvent>> {
    let Some(name) = delta.modal_event.as_deref() else {
        return Ok(None);
    };
    let suffix = name
        .strip_prefix("diarization.")
        .ok_or_else(|| Error::Other("Unexpected diarization event type".into()))?;
    let raw = delta.modal_metadata_json.as_deref().unwrap_or("{}");
    if raw.len() > MAX_EVENT_BYTES {
        return Err(Error::Other(
            "Diarization event exceeds its metadata bound".into(),
        ));
    }
    let mut value: Value = serde_json::from_str(raw)?;
    if !value.is_object() || value["diarization_version"] != IPC_PROTOCOL_VERSION {
        return Err(Error::Other("Unsupported diarization event version".into()));
    }
    if value["session_id"].as_str() != Some(session_id) {
        return Err(Error::Other(
            "Diarization event belongs to a different session".into(),
        ));
    }
    value["type"] = suffix.into();
    if suffix == "error" {
        value["message"] = delta
            .content
            .clone()
            .unwrap_or_else(|| "PIE diarization failed".into())
            .into();
    }
    let event: DiarizationEvent = serde_json::from_value(value)
        .map_err(|error| Error::Other(format!("Invalid diarization event: {error}")))?;
    let nonnegative = |value: f64| value.is_finite() && value >= 0.0;
    match &event {
        DiarizationEvent::Ready {
            speakers,
            output_frame_seconds,
            ..
        } if *speakers != 8 || *output_frame_seconds != 0.01 => {
            return Err(Error::Other(
                "PIE diarization requires eight speakers at 10 ms".into(),
            ))
        }
        DiarizationEvent::Update {
            frame_seconds,
            probabilities,
            segments,
            replace_segments_from_seconds,
            compute_ms,
            audio_seconds,
            ..
        } => {
            if *frame_seconds != 0.01
                || probabilities.len() > 240_000
                || segments.len() > 1_000_000
                || !nonnegative(*replace_segments_from_seconds)
                || !nonnegative(*compute_ms)
                || !nonnegative(*audio_seconds)
                || probabilities
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                || segments.iter().any(|segment| {
                    !(1..=8).contains(&segment.speaker_index)
                        || segment.speaker_id
                            != format!("{session_id}:speaker-{}", segment.speaker_index)
                        || !nonnegative(segment.start_seconds)
                        || !nonnegative(segment.end_seconds)
                        || segment.end_seconds < segment.start_seconds
                        || segment.mean_probability.is_some_and(|value| {
                            !value.is_finite() || !(0.0..=1.0).contains(&value)
                        })
                })
            {
                return Err(Error::Other(
                    "PIE diarization returned invalid probabilities, segments or timing".into(),
                ));
            }
        }
        DiarizationEvent::Gap {
            start_seconds,
            end_seconds,
            missing_frames,
            ..
        } if !nonnegative(*start_seconds)
            || !nonnegative(*end_seconds)
            || end_seconds < start_seconds
            || *missing_frames > 50 =>
        {
            return Err(Error::Other(
                "PIE diarization returned an invalid gap".into(),
            ))
        }
        DiarizationEvent::Metrics {
            compute_ms_total,
            compute_ms_max,
            elapsed_ms,
            ..
        } if !nonnegative(*compute_ms_total)
            || !nonnegative(*compute_ms_max)
            || !nonnegative(*elapsed_ms) =>
        {
            return Err(Error::Other(
                "PIE diarization returned invalid metrics".into(),
            ))
        }
        _ => {}
    }
    Ok(Some(event))
}

#[derive(Clone)]
pub struct DiarizationControl {
    shared: Arc<Shared>,
}
impl DiarizationControl {
    /// Admit one 80 ms frame after PIE's bounded queue ACK, without waiting for
    /// inference. A rejected queue-full sequence may be retried unchanged.
    pub fn push_audio(&self, sequence: u64, pcm: Vec<f32>) -> Result<DiarizationAdmission> {
        if pcm.len() != self.shared.frame_samples
            || pcm.iter().any(|x| !x.is_finite() || x.abs() > 1.0)
        {
            return Err(Error::Other(
                "Diarization requires one 80 ms frame of finite mono PCM in [-1, 1]".into(),
            ));
        }
        let mut state = self
            .shared
            .admission
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.shared.require_open()?;
        if state.last_sequence.is_some_and(|last| sequence <= last) {
            return Err(Error::Other(
                "Diarization sequence must increase strictly".into(),
            ));
        }
        let missing_frames = state.last_sequence.map_or(0, |last| sequence - last - 1);
        if missing_frames > 50 {
            return Err(Error::Other(
                "Diarization input gap exceeds four seconds; open a new session".into(),
            ));
        }
        let bytes = pcm
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let result = self.shared.command(
            "diarization_input",
            json!({"sequence":sequence,"pcm_f32_b64":STANDARD.encode(bytes)}),
        );
        let data = match result {
            Ok(data) => data,
            Err(failure) => {
                if !failure.rejected {
                    self.shared.finish(Some(format!(
                        "PIE diarization admission: {}",
                        failure.error
                    )));
                    self.shared.request_close();
                }
                return Err(failure.error);
            }
        };
        if data["accepted_sequence"].as_u64() != Some(sequence)
            || data["missing_frames"].as_u64() != Some(missing_frames)
        {
            self.shared.finish(Some(
                "PIE diarization ACK did not confirm the admitted sequence and gap".into(),
            ));
            self.shared.request_close();
            return Err(Error::Other(
                "PIE diarization admission ACK mismatch".into(),
            ));
        }
        state.last_sequence = Some(sequence);
        Ok(DiarizationAdmission { missing_frames })
    }

    /// Gate further input and flush every accepted frame, including the last
    /// partial inference chunk. Continue consuming events through final Closed.
    pub async fn finish(&self) -> Result<()> {
        let shared = Arc::clone(&self.shared);
        tokio::task::spawn_blocking(move || {
            let mut state = shared
                .admission
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if state.finish_confirmed {
                return Ok(());
            }
            if shared.closed.load(Ordering::Acquire) {
                return Err(Error::ChannelClosed);
            }
            shared.finishing.store(true, Ordering::Release);
            let data = match shared.command("diarization_finish", json!({})) {
                Ok(data) => data,
                Err(failure) => {
                    shared.finish(Some(format!(
                        "Finishing PIE diarization: {}",
                        failure.error
                    )));
                    shared.request_close();
                    return Err(failure.error);
                }
            };
            if data["session_state"] == "open" {
                return Err(Error::Other(
                    "PIE did not gate diarization input on finish".into(),
                ));
            }
            state.finish_confirmed = true;
            Ok(())
        })
        .await
        .map_err(|error| Error::Internal(format!("Diarization finish task: {error}")))?
    }

    pub fn close(&self) {
        self.shared.request_close();
    }
}

pub struct DiarizationSession {
    shared: Arc<Shared>,
    _route: StreamResponseRoute,
}
impl DiarizationSession {
    pub fn control(&self) -> DiarizationControl {
        DiarizationControl {
            shared: Arc::clone(&self.shared),
        }
    }
    pub async fn next_event(&mut self) -> Option<DiarizationEvent> {
        loop {
            let notified = self.shared.output_ready.notified();
            {
                let mut output = self
                    .shared
                    .output
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                if let Some((event, bytes)) = output.events.pop_front() {
                    output.bytes -= bytes;
                    return Some(event);
                }
                if output.done {
                    return None;
                }
            }
            notified.await;
        }
    }

    /// Close the engine session and wait for its terminal lifecycle confirmation.
    /// This SDK owns only IPC handles; all native state and weights belong to PIE.
    pub async fn shutdown(&mut self) -> Result<()> {
        self.shared.request_close();
        tokio::time::timeout(SHUTDOWN_TIMEOUT, async {
            loop {
                let notified = self.shared.terminal.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if self.shared.remote_final.load(Ordering::Acquire) {
                    return Ok(());
                }
                if let Some(message) = self
                    .shared
                    .failure
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .clone()
                {
                    return Err(Error::Other(message));
                }
                notified.await;
            }
        })
        .await
        .map_err(|_| Error::Other("PIE diarization shutdown deadline exceeded".into()))?
    }

    pub(crate) async fn open(
        ipc: Arc<IPCClient>,
        info: ModelInfo,
        public_model_id: String,
        options: DiarizationOptions,
    ) -> Result<Self> {
        options.validate()?;
        let request_id = ipc.next_request_id();
        let session_id = options
            .session_id
            .clone()
            .unwrap_or_else(|| format!("diar-{:016x}", rand::random::<u64>()));
        let shared = Arc::new(Shared {
            ipc: Arc::clone(&ipc),
            model_id: info.model_id.clone(),
            public_model_id,
            session_id: session_id.clone(),
            request_id,
            channel_id: AtomicU64::new(0),
            sample_rate: options.sample_rate,
            frame_samples: options.frame_samples(),
            ready: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            finishing: AtomicBool::new(false),
            close_requested: AtomicBool::new(false),
            remote_final: AtomicBool::new(false),
            ended: AtomicBool::new(false),
            admission: Mutex::new(AdmissionState::default()),
            output: Mutex::new(OutputQueue::default()),
            output_ready: Notify::new(),
            terminal: Notify::new(),
            failure: Mutex::new(None),
        });
        let weak = Arc::downgrade(&shared);
        let route = ipc.bind_diarization_route(
            request_id,
            Arc::new(move |delta, from_engine| {
                if let Some(shared) = weak.upgrade() {
                    shared.receive(delta, from_engine);
                }
            }),
        )?;
        shared.channel_id.store(route.channel_id, Ordering::Release);
        let prompt = PromptPayload {
            num_candidates: 1,
            max_generated_tokens: 1,
            modal_options_json:
                json!({"diarization_version":IPC_PROTOCOL_VERSION,"sample_rate":options.sample_rate,
                "max_pending_frames":options.max_pending_frames,"session_id":session_id})
                .to_string(),
            ..Default::default()
        };
        ipc.send_diarization_request(
            request_id,
            &info.model_id,
            &info.model_path,
            route.channel_id,
            prompt,
        )?;
        let session = Self {
            shared,
            _route: route,
        };
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let notified = session.shared.output_ready.notified();
                if session.shared.ended.load(Ordering::Acquire) {
                    return Err(Error::Other(
                        session
                            .shared
                            .failure
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .clone()
                            .unwrap_or_else(|| "PIE closed diarization before ready".into()),
                    ));
                }
                if session.shared.ready.load(Ordering::Acquire) {
                    return Ok(());
                }
                notified.await;
            }
        })
        .await
        .map_err(|_| Error::Other("PIE diarization ready deadline exceeded".into()))??;
        Ok(session)
    }
}
impl Drop for DiarizationSession {
    fn drop(&mut self) {
        self.shared.request_close();
    }
}

/// Match a word/utterance interval to anonymous tracks, retaining overlaps.
/// This is temporal attribution only; it never establishes a person's name.
pub fn speakers_for_interval(segments: &[SpeakerSegment], start: f64, end: f64) -> Vec<String> {
    let mut durations: HashMap<&str, f64> = HashMap::new();
    if !start.is_finite() || !end.is_finite() || end <= start {
        return Vec::new();
    }
    for segment in segments {
        let overlap = (segment.end_seconds.min(end) - segment.start_seconds.max(start)).max(0.0);
        if overlap > 0.0 {
            *durations.entry(&segment.speaker_id).or_default() += overlap;
        }
    }
    let mut matches: Vec<_> = durations.into_iter().collect();
    matches.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    matches
        .into_iter()
        .map(|(speaker, _)| speaker.to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn profile_is_the_exact_new_eight_speaker_model() {
        let p = architecture().unwrap();
        assert_eq!(p.speakers, 8);
        assert_eq!(p.output_frame_seconds, 0.01);
        assert_eq!(p.size_bytes, 107_012_128);
        assert_eq!(p.sha256.len(), 64);
        assert!(!is_diarization_model(
            "nvidia/diar_streaming_sortformer_4spk-v2"
        ));
    }
    #[test]
    fn overlapping_tracks_are_not_collapsed_into_a_person() {
        let segment = |id: &str, a: f64, b: f64| SpeakerSegment {
            speaker_id: id.into(),
            speaker_index: 1,
            start_seconds: a,
            end_seconds: b,
            mean_probability: Some(0.9),
            provisional: false,
        };
        let segments = vec![
            segment("session:speaker-1", 0.0, 2.0),
            segment("session:speaker-2", 1.0, 3.0),
        ];
        assert_eq!(
            speakers_for_interval(&segments, 1.2, 1.8),
            ["session:speaker-1", "session:speaker-2"]
        );
        assert!(speakers_for_interval(&segments, 4.0, 5.0).is_empty());
    }
}
