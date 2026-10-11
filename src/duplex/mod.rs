//! Stateful speech-to-speech over the existing PIE transport.
//! All Moshi, Mimi and reference tensors, inference, scheduling and residency
//! belong to PIE. This module only validates PCM and manages bounded IPC queues.

mod model;
pub(crate) use model::engine_source;
pub use model::{
    architecture, is_moshi_model, Checkpoint, DuplexArchitecture, DEFAULT_MODEL, RAG_MODEL,
};

use crate::ipc::client::{DuplexResponseRoute, IPCClient, ResponseDelta};
use crate::ipc::serialization::PromptPayload;
use crate::{Error, ModelInfo, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

/// Version of the native PIE streaming contract required by this client.
pub const IPC_PROTOCOL_VERSION: u32 = 1;
pub const SAMPLE_RATE: u32 = 24_000;
pub const FRAME_SAMPLES: usize = 1_920;
const CONTROL_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_EVENTS: usize = 256;
const MAX_OUTPUT_AUDIO: usize = 64;
const MAX_REFERENCE_BYTES: usize = 8_192;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DuplexDevice {
    #[default]
    Auto,
    Cpu,
    Metal,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct DuplexOptions {
    pub device: DuplexDevice,
    /// Legacy override; native PIE currently requires engine-managed GPU placement.
    pub mimi_cpu: bool,
    /// Legacy compatibility field; only the default value is accepted by PIE.
    pub codec_threads: usize,
    /// False means only queued PCM advances the model, useful for file clients.
    pub realtime: bool,
    /// True lets Moshi choose its own words; false speaks only queued text.
    pub autonomous: bool,
    pub max_pending_frames: usize,
    /// At this frame limit PIE resets the timeline without reloading weights.
    pub max_steps: usize,
    /// Legacy compatibility field; only the default value is accepted by PIE.
    pub text_token_interval: usize,
    pub audio_temperature: f64,
    pub audio_top_k: usize,
    pub text_temperature: f64,
    pub text_top_k: usize,
    pub seed: u64,
}

impl Default for DuplexOptions {
    fn default() -> Self {
        Self {
            device: DuplexDevice::Auto,
            mimi_cpu: false,
            codec_threads: 4,
            realtime: false,
            autonomous: true,
            max_pending_frames: 6,
            max_steps: 45_000,
            text_token_interval: 12,
            audio_temperature: 0.8,
            audio_top_k: 250,
            text_temperature: 0.8,
            text_top_k: 250,
            seed: 299_792_458,
        }
    }
}

impl DuplexOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.device == DuplexDevice::Cpu || self.mimi_cpu {
            return Err(Error::Other("PIE duplex currently uses engine-managed GPU placement; CPU overrides are unsupported".into()));
        }
        if self.codec_threads != 4 || self.text_token_interval != 12 {
            return Err(Error::Other("Legacy SDK codec thread/text cadence overrides are unsupported by native PIE duplex".into()));
        }
        if !(1..=16).contains(&self.codec_threads)
            || !(1..=64).contains(&self.max_pending_frames)
            || !(256..=45_000).contains(&self.max_steps)
            || !(1..=25).contains(&self.text_token_interval)
            || !(0..=2_048).contains(&self.audio_top_k)
            || !(0..=32_000).contains(&self.text_top_k)
            || !self.audio_temperature.is_finite()
            || self.audio_temperature < 0.0
            || !self.text_temperature.is_finite()
            || self.text_temperature < 0.0
        {
            return Err(Error::Other("Invalid duplex options: bounded queues, timeline, sampling and text cadence are required".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct DuplexMetrics {
    pub input_frames: u64,
    pub output_frames: u64,
    pub synthetic_silence_frames: u64,
    pub dropped_input_frames: u64,
    pub dropped_output_frames: u64,
    pub compute_ms_total: f64,
    pub compute_ms_max: f64,
    pub queue_ms_max: f64,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DuplexEvent {
    Ready {
        model_id: String,
        sample_rate: u32,
        frame_samples: usize,
        epoch: u64,
    },
    Audio {
        epoch: u64,
        sequence: u64,
        pcm: Vec<f32>,
        compute_ms: f64,
        queue_ms: f64,
    },
    TextDelta {
        epoch: u64,
        sequence: u64,
        text: String,
    },
    SpeechQueued {
        epoch: u64,
        tokens: usize,
    },
    SpeechDone {
        epoch: u64,
    },
    Interrupted {
        epoch: u64,
    },
    Reset {
        epoch: u64,
        reason: String,
    },
    Metrics {
        #[serde(flatten)]
        metrics: DuplexMetrics,
    },
    Error {
        message: String,
    },
    ReferenceQueued {
        epoch: u64,
        version: u64,
    },
    ReferenceApplied {
        epoch: u64,
        version: u64,
        steps: usize,
        encode_ms: f64,
    },
    RetrievalRequested {
        epoch: u64,
        /// Output sequence is a presentation counter, not microphone identity.
        sequence: u64,
        /// Exact host input frame used by this step. Native synthetic silence
        /// and older engines may have no captured input to correlate.
        #[serde(default)]
        input_sequence: Option<u64>,
        /// Native model clock, independent of input and output sequences.
        #[serde(default)]
        model_step: Option<u64>,
    },
    Closed,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct AudioAdmission {
    pub epoch: u64,
    /// Number of older pending microphone frames discarded by this admission.
    pub dropped_frames: u64,
}

struct InputFrame {
    sequence: u64,
    epoch: u64,
    pcm: Vec<f32>,
}
#[derive(Default)]
struct InputQueue {
    frames: VecDeque<InputFrame>,
    last_sequence: Option<u64>,
}
#[derive(Default)]
struct OutputQueue {
    events: VecDeque<DuplexEvent>,
    done: bool,
}

struct Shared {
    ipc: Arc<IPCClient>,
    model_id: String,
    request_id: u64,
    response_channel_id: AtomicU64,
    remote_final: AtomicBool,
    epoch: AtomicU64,
    ready: AtomicBool,
    closed: AtomicBool,
    ended: AtomicBool,
    supports_reference: AtomicBool,
    supports_grounded_response: AtomicBool,
    supports_response_hold: AtomicBool,
    autonomous: bool,
    input: Mutex<InputQueue>,
    input_wake: Condvar,
    output: Mutex<OutputQueue>,
    output_ready: Notify,
    opening_error: Mutex<Option<String>>,
    max_pending_frames: usize,
    dropped_input: AtomicU64,
    dropped_output: AtomicU64,
}

impl Shared {
    fn command(&self, name: &str, epoch: u64, fields: Value) -> Result<Value> {
        let mut command = json!({"type":name,"model_id":self.model_id,
            "request_id":self.request_id,"response_channel_id":self.response_channel_id.load(Ordering::Acquire),
            "epoch":epoch});
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
            // A native timeline rollover may reach the management ACK before
            // its ordered reset event reaches PULL. Adopt the engine's newer
            // epoch, so the input pump discards that old frame without killing
            // the session. The actual reset event still owns user-visible state.
            if response["data"]["duplex"]["error_code"] == "stale_epoch" {
                if let Some(epoch) = response["data"]["duplex"]["epoch"].as_u64() {
                    if epoch > self.epoch.load(Ordering::Acquire) {
                        self.update_epoch(epoch);
                    }
                }
            }
            return Err(Error::Other(
                response["message"]
                    .as_str()
                    .unwrap_or("PIE rejected duplex command")
                    .to_owned(),
            ));
        }
        let data = response["data"]["duplex"].clone();
        if !data.is_object() {
            return Err(Error::Other("PIE duplex ACK has no typed data".into()));
        }
        Ok(data)
    }

    fn update_epoch(&self, epoch: u64) {
        self.epoch.fetch_max(epoch, Ordering::AcqRel);
        let mut input = self.input.lock().unwrap_or_else(|e| e.into_inner());
        let current = self.epoch.load(Ordering::Acquire);
        let before = input.frames.len();
        input.frames.retain(|frame| frame.epoch == current);
        self.dropped_input
            .fetch_add((before - input.frames.len()) as u64, Ordering::AcqRel);
        drop(input);
        let mut output = self.output.lock().unwrap_or_else(|e| e.into_inner());
        let current = self.epoch.load(Ordering::Acquire);
        output
            .events
            .retain(|event| event_epoch(event).is_none_or(|value| value >= current));
    }

    fn emit(&self, event: DuplexEvent) {
        let mut output = self.output.lock().unwrap_or_else(|e| e.into_inner());
        if output.done {
            return;
        }
        if event_epoch(&event).is_some_and(|epoch| epoch != self.epoch.load(Ordering::Acquire)) {
            return;
        }
        if matches!(event, DuplexEvent::Audio { .. }) {
            let count = output
                .events
                .iter()
                .filter(|event| matches!(event, DuplexEvent::Audio { .. }))
                .count();
            if count >= MAX_OUTPUT_AUDIO {
                if let Some(at) = output
                    .events
                    .iter()
                    .position(|event| matches!(event, DuplexEvent::Audio { .. }))
                {
                    output.events.remove(at);
                    self.dropped_output.fetch_add(1, Ordering::AcqRel);
                }
            }
        }
        let overflowed = output.events.len() >= MAX_EVENTS;
        if overflowed {
            // Losing control/text events would silently corrupt the dialogue.
            // Terminate instead; only audio frames have explicit drop accounting.
            output.events.clear();
            output.events.push_back(DuplexEvent::Error {
                message: "Duplex consumer is not draining its bounded event queue".into(),
            });
            output.events.push_back(DuplexEvent::Closed);
            output.done = true;
        } else {
            output.events.push_back(event);
        }
        drop(output);
        if overflowed {
            self.close();
        }
        self.output_ready.notify_one();
    }

    fn finish(&self, error: Option<String>) {
        if self.ended.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Some(message) = error {
            *self.opening_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(message.clone());
            self.emit(DuplexEvent::Error { message });
        }
        self.emit(DuplexEvent::Closed);
        self.close();
        self.output.lock().unwrap_or_else(|e| e.into_inner()).done = true;
        self.output_ready.notify_one();
    }

    fn close(&self) {
        // The worker checks this predicate while holding input. Publish closure
        // under that same mutex so its check-to-wait transition cannot lose the wake.
        let mut input = self.input.lock().unwrap_or_else(|e| e.into_inner());
        self.closed.store(true, Ordering::Release);
        input.frames.clear();
        drop(input);
        self.input_wake.notify_all();
    }

    fn receive(&self, delta: ResponseDelta, from_engine: bool) {
        if delta.request_id != self.request_id {
            self.finish(Some(
                "PIE duplex response belongs to a different request".into(),
            ));
            return;
        }
        if delta.is_final_delta && from_engine {
            self.remote_final.store(true, Ordering::Release);
        }
        if self.ended.load(Ordering::Acquire) {
            return;
        }
        if let Some(error) = &delta.error {
            self.finish(Some(error.clone()));
            return;
        }
        let result = decode_event(&delta);
        match result {
            Ok(Some(event)) => {
                match &event {
                    DuplexEvent::Ready {
                        sample_rate,
                        frame_samples,
                        epoch,
                        ..
                    } => {
                        if *sample_rate != SAMPLE_RATE || *frame_samples != FRAME_SAMPLES {
                            self.finish(Some(
                                "PIE returned incompatible duplex audio geometry".into(),
                            ));
                            return;
                        }
                        let meta: Value = serde_json::from_str(
                            delta.modal_metadata_json.as_deref().unwrap_or("{}"),
                        )
                        .unwrap_or(Value::Null);
                        self.supports_reference.store(
                            meta["supports_reference"].as_bool().unwrap_or(false),
                            Ordering::Release,
                        );
                        self.supports_grounded_response.store(
                            meta["supports_grounded_response"]
                                .as_bool()
                                .unwrap_or(false),
                            Ordering::Release,
                        );
                        self.supports_response_hold.store(
                            meta["supports_response_hold"].as_bool().unwrap_or(false),
                            Ordering::Release,
                        );
                        self.update_epoch(*epoch);
                        self.ready.store(true, Ordering::Release);
                    }
                    DuplexEvent::Interrupted { epoch } | DuplexEvent::Reset { epoch, .. } => {
                        self.update_epoch(*epoch)
                    }
                    DuplexEvent::Closed => {
                        self.finish(None);
                        return;
                    }
                    DuplexEvent::Error { message } => {
                        self.finish(Some(message.clone()));
                        return;
                    }
                    _ => {}
                }
                if event_epoch(&event)
                    .is_some_and(|epoch| epoch != self.epoch.load(Ordering::Acquire))
                {
                    return;
                }
                let event = match event {
                    DuplexEvent::Metrics { mut metrics } => {
                        metrics.dropped_input_frames += self.dropped_input.load(Ordering::Acquire);
                        metrics.dropped_output_frames +=
                            self.dropped_output.load(Ordering::Acquire);
                        DuplexEvent::Metrics { metrics }
                    }
                    event => event,
                };
                self.emit(event);
            }
            Ok(None) => {}
            Err(error) => {
                self.finish(Some(error.to_string()));
                return;
            }
        }
        if delta.is_final_delta {
            self.finish(None);
        }
    }
}

fn event_epoch(event: &DuplexEvent) -> Option<u64> {
    match event {
        DuplexEvent::Ready { epoch, .. }
        | DuplexEvent::Audio { epoch, .. }
        | DuplexEvent::TextDelta { epoch, .. }
        | DuplexEvent::SpeechQueued { epoch, .. }
        | DuplexEvent::SpeechDone { epoch }
        | DuplexEvent::Interrupted { epoch }
        | DuplexEvent::Reset { epoch, .. }
        | DuplexEvent::ReferenceQueued { epoch, .. }
        | DuplexEvent::ReferenceApplied { epoch, .. }
        | DuplexEvent::RetrievalRequested { epoch, .. } => Some(*epoch),
        _ => None,
    }
}

fn decode_event(delta: &ResponseDelta) -> Result<Option<DuplexEvent>> {
    let Some(name) = delta.modal_event.as_deref() else {
        return Ok(None);
    };
    let suffix = name
        .strip_prefix("duplex.")
        .ok_or_else(|| Error::Other("Unexpected duplex event type".into()))?;
    let mut value: Value =
        serde_json::from_str(delta.modal_metadata_json.as_deref().unwrap_or("{}"))
            .map_err(|error| Error::Other(format!("Invalid duplex event metadata: {error}")))?;
    if !value.is_object() || value["duplex_version"] != 1 {
        return Err(Error::Other("Unsupported duplex event version".into()));
    }
    if suffix == "ready" && value["channels"] != 1 {
        return Err(Error::Other(
            "PIE duplex requires mono audio (channels=1)".into(),
        ));
    }
    if suffix == "metrics" {
        for key in [
            "input_frames",
            "output_frames",
            "synthetic_silence_frames",
            "dropped_input_frames",
            "dropped_output_frames",
        ] {
            if value[key].as_u64().is_none() {
                return Err(Error::Other(format!(
                    "PIE duplex metrics missing integer {key}"
                )));
            }
        }
        for key in [
            "compute_ms_total",
            "compute_ms_max",
            "queue_ms_max",
            "elapsed_ms",
        ] {
            if !value[key]
                .as_f64()
                .is_some_and(|number| number.is_finite() && number >= 0.0)
            {
                return Err(Error::Other(format!(
                    "PIE duplex metrics missing valid {key}"
                )));
            }
        }
    }
    value["type"] = match suffix {
        "text" => "text_delta",
        other => other,
    }
    .into();
    if suffix == "audio" {
        let encoded = delta
            .modal_bytes_b64
            .as_deref()
            .ok_or_else(|| Error::Other("Duplex audio has no PCM payload".into()))?;
        if encoded.len() > FRAME_SAMPLES * 8 {
            return Err(Error::Other(
                "Duplex PCM payload exceeds its frame bound".into(),
            ));
        }
        let bytes = STANDARD
            .decode(encoded)
            .map_err(|error| Error::Other(format!("Invalid duplex PCM: {error}")))?;
        if bytes.len() != FRAME_SAMPLES * 4 {
            return Err(Error::Other(
                "PIE duplex PCM frame must contain 1920 float32 samples".into(),
            ));
        }
        let pcm = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect::<Vec<_>>();
        if pcm.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::Other("PIE duplex PCM is not finite".into()));
        }
        value["pcm"] = json!(pcm);
    }
    if suffix == "text" {
        value["text"] = delta.content.clone().unwrap_or_default().into();
    }
    if suffix.starts_with("reference_") {
        value["version"] = value["reference_version"].clone();
    }
    if suffix == "error" {
        value["message"] = delta
            .content
            .clone()
            .unwrap_or_else(|| "PIE duplex failed".into())
            .into();
    }
    serde_json::from_value(value)
        .map(Some)
        .map_err(|error| Error::Other(format!("Invalid duplex event: {error}")))
}

fn input_worker(shared: Arc<Shared>) {
    loop {
        let frame = {
            let mut input = shared.input.lock().unwrap_or_else(|e| e.into_inner());
            while input.frames.is_empty() && !shared.closed.load(Ordering::Acquire) {
                input = shared
                    .input_wake
                    .wait(input)
                    .unwrap_or_else(|e| e.into_inner());
            }
            if shared.closed.load(Ordering::Acquire) {
                None
            } else {
                input.frames.pop_front()
            }
        };
        let Some(frame) = frame else {
            break;
        };
        if frame.epoch != shared.epoch.load(Ordering::Acquire) {
            shared.dropped_input.fetch_add(1, Ordering::AcqRel);
            continue;
        }
        let bytes = frame
            .pcm
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let result = shared.command(
            "duplex_input",
            frame.epoch,
            json!({"sequence":frame.sequence,"pcm_f32_b64":STANDARD.encode(bytes)}),
        );
        if let Err(error) = result {
            if frame.epoch != shared.epoch.load(Ordering::Acquire) {
                shared.dropped_input.fetch_add(1, Ordering::AcqRel);
            }
            if frame.epoch == shared.epoch.load(Ordering::Acquire)
                && !shared.closed.load(Ordering::Acquire)
            {
                shared.finish(Some(format!("PIE duplex audio admission: {error}")));
                break;
            }
        }
    }
    if !shared.remote_final.load(Ordering::Acquire) {
        let result = shared.command(
            "duplex_close",
            shared.epoch.load(Ordering::Acquire),
            json!({}),
        );
        if let Err(error) = result {
            shared.finish(Some(format!("Closing PIE duplex: {error}")));
        }
    }
}

#[derive(Clone)]
pub struct DuplexControl {
    shared: Arc<Shared>,
}
impl DuplexControl {
    pub fn epoch(&self) -> u64 {
        self.shared.epoch.load(Ordering::Acquire)
    }
    pub fn supports_reference(&self) -> bool {
        self.shared.supports_reference.load(Ordering::Acquire)
    }
    /// Whether this live session can begin a model-authored response to context.
    /// Older native engines omit the Ready capability and report false.
    pub fn supports_grounded_response(&self) -> bool {
        self.shared
            .supports_grounded_response
            .load(Ordering::Acquire)
    }
    /// Whether this live session can wait for context before responding.
    /// Older native engines omit the Ready capability and report false.
    pub fn supports_response_hold(&self) -> bool {
        self.shared.supports_response_hold.load(Ordering::Acquire)
    }
    pub fn is_autonomous(&self) -> bool {
        self.shared.autonomous
    }
    fn require_open(&self) -> Result<()> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(Error::ChannelClosed);
        }
        if !self.shared.ready.load(Ordering::Acquire) {
            return Err(Error::ModelNotReady(
                "PIE duplex session is not ready".into(),
            ));
        }
        Ok(())
    }
    pub fn push_audio(&self, sequence: u64, pcm: Vec<f32>) -> Result<AudioAdmission> {
        self.require_open()?;
        if pcm.len() != FRAME_SAMPLES
            || pcm
                .iter()
                .any(|sample| !sample.is_finite() || sample.abs() > 1.0)
        {
            return Err(Error::Other(
                "Expected 1920 finite mono float32 samples in [-1,1] at 24 kHz".into(),
            ));
        }
        let mut input = self.shared.input.lock().unwrap_or_else(|e| e.into_inner());
        self.require_open()?;
        if input.last_sequence.is_some_and(|last| sequence <= last) {
            return Err(Error::Other("Duplex input sequence must increase".into()));
        }
        let epoch = self.epoch();
        let mut dropped = 0;
        while input.frames.len() >= self.shared.max_pending_frames {
            input.frames.pop_front();
            dropped += 1;
        }
        self.shared
            .dropped_input
            .fetch_add(dropped, Ordering::AcqRel);
        input.last_sequence = Some(sequence);
        input.frames.push_back(InputFrame {
            sequence,
            epoch,
            pcm,
        });
        drop(input);
        self.shared.input_wake.notify_one();
        Ok(AudioAdmission {
            epoch,
            dropped_frames: dropped,
        })
    }
    pub fn reference(&self, text: String, expected_epoch: u64) -> Result<u64> {
        self.require_open()?;
        if !self.supports_reference() {
            return Err(Error::Other(
                "This PIE duplex model does not support references".into(),
            ));
        }
        if expected_epoch != self.epoch() {
            return Err(Error::Other(
                "Speech reference belongs to an interrupted epoch".into(),
            ));
        }
        if text.trim().is_empty() || text.len() > MAX_REFERENCE_BYTES {
            return Err(Error::Other(
                "Reference must contain 1..8192 UTF-8 bytes".into(),
            ));
        }
        let data = self
            .shared
            .command("duplex_reference", expected_epoch, json!({"text":text}))?;
        data["reference_version"]
            .as_u64()
            .ok_or_else(|| Error::Other("PIE reference ACK has no version".into()))
    }
    /// Ask the native session to wait for a grounded reply without changing epoch.
    /// Returns its reference version; the audio clock and microphone remain active.
    pub fn hold_response(&self, expected_epoch: u64) -> Result<u64> {
        self.require_open()?;
        if !self.supports_response_hold() {
            return Err(Error::Other(
                "This PIE duplex session does not support response holds".into(),
            ));
        }
        if expected_epoch != self.epoch() {
            return Err(Error::Other(
                "Response hold belongs to an interrupted epoch".into(),
            ));
        }
        let data = self
            .shared
            .command("duplex_hold_response", expected_epoch, json!({}))?;
        if data["epoch"].as_u64() != Some(expected_epoch) || self.epoch() != expected_epoch {
            return Err(Error::Other(
                "Response hold epoch changed during admission".into(),
            ));
        }
        data["reference_version"]
            .as_u64()
            .filter(|version| *version > 0)
            .ok_or_else(|| {
                Error::Other("PIE response hold ACK has no valid reference version".into())
            })
    }
    /// Queue release of this hold without cueing speech or changing the epoch.
    /// False means the hold was superseded. A later control can still supersede
    /// an accepted release before it reaches the next complete frame boundary.
    pub fn release_response_hold(&self, expected_epoch: u64, hold_version: u64) -> Result<bool> {
        self.require_open()?;
        if !self.supports_response_hold() {
            return Err(Error::Other(
                "This PIE duplex session does not support response holds".into(),
            ));
        }
        if hold_version == 0 {
            return Err(Error::Other(
                "Response hold version must be positive".into(),
            ));
        }
        if expected_epoch != self.epoch() {
            return Ok(false);
        }
        let data = self.shared.command(
            "duplex_release_response_hold",
            expected_epoch,
            json!({"hold_version":hold_version}),
        )?;
        let epoch = data["epoch"]
            .as_u64()
            .ok_or_else(|| Error::Other("PIE response release ACK has no epoch".into()))?;
        let queued = data["hold_release_queued"]
            .as_bool()
            .ok_or_else(|| Error::Other("PIE response release ACK has no result".into()))?;
        if epoch > self.epoch() {
            self.shared.update_epoch(epoch);
        }
        Ok(queued && epoch == expected_epoch && self.epoch() == expected_epoch)
    }
    /// Supply factual context and request a natural response in the model's own words.
    /// Returns the queued reference version, not evidence of generated or played audio.
    /// Background `reference` calls never request this response cue.
    pub fn grounded_reply(&self, context: String, expected_epoch: u64) -> Result<u64> {
        self.require_open()?;
        if !self.supports_grounded_response() {
            return Err(Error::Other(
                "This PIE duplex session does not support grounded responses".into(),
            ));
        }
        if expected_epoch != self.epoch() {
            return Err(Error::Other(
                "Grounded response belongs to an interrupted epoch".into(),
            ));
        }
        if context.trim().is_empty() || context.len() > MAX_REFERENCE_BYTES {
            return Err(Error::Other(
                "Grounded response context must contain 1..8192 UTF-8 bytes".into(),
            ));
        }
        let data = self
            .shared
            .command("duplex_reply", expected_epoch, json!({"text":context}))?;
        if data["epoch"].as_u64() != Some(expected_epoch) || self.epoch() != expected_epoch {
            return Err(Error::Other(
                "Grounded response epoch changed during admission".into(),
            ));
        }
        data["reference_version"]
            .as_u64()
            .filter(|version| *version > 0)
            .ok_or_else(|| {
                Error::Other("PIE grounded response ACK has no valid reference version".into())
            })
    }
    fn advance(&self, name: &str) -> Result<u64> {
        self.require_open()?;
        let previous = self.epoch();
        let data = self.shared.command(name, previous, json!({}))?;
        let epoch = data["epoch"]
            .as_u64()
            .ok_or_else(|| Error::Other("PIE control ACK has no epoch".into()))?;
        if epoch <= previous {
            return Err(Error::Other("PIE control did not advance its epoch".into()));
        }
        self.shared.update_epoch(epoch);
        Ok(epoch)
    }
    pub fn interrupt(&self) -> Result<u64> {
        self.advance("duplex_interrupt")
    }
    pub fn reset(&self) -> Result<u64> {
        self.advance("duplex_reset")
    }
    /// Queue supplied speech without interrupting the autonomous session.
    pub fn speak(&self, text: String, replace: bool) -> Result<u64> {
        self.speak_in_epoch(text, replace, self.epoch())
    }
    /// Reject a supplied reply if its original conversation epoch has changed.
    pub fn speak_in_epoch(&self, text: String, replace: bool, expected_epoch: u64) -> Result<u64> {
        self.require_open()?;
        if expected_epoch != self.epoch() {
            return Err(Error::Other(
                "Supplied speech belongs to an interrupted epoch".into(),
            ));
        }
        if text.trim().is_empty() || text.len() > MAX_REFERENCE_BYTES {
            return Err(Error::Other(
                "Speech text exceeds 8192 UTF-8 bytes or is empty".into(),
            ));
        }
        let data = self.shared.command(
            "duplex_speak",
            expected_epoch,
            json!({"text":text,"replace":replace}),
        )?;
        if data["epoch"].as_u64() != Some(expected_epoch) || self.epoch() != expected_epoch {
            return Err(Error::Other(
                "Supplied speech epoch changed during admission".into(),
            ));
        }
        Ok(expected_epoch)
    }
    pub fn close(&self) -> Result<()> {
        self.shared.close();
        Ok(())
    }
}

pub struct DuplexSession {
    shared: Arc<Shared>,
    _route: DuplexResponseRoute,
}
impl DuplexSession {
    pub fn control(&self) -> DuplexControl {
        DuplexControl {
            shared: Arc::clone(&self.shared),
        }
    }
    pub async fn next_event(&mut self) -> Option<DuplexEvent> {
        loop {
            let notified = self.shared.output_ready.notified();
            {
                let mut output = self.shared.output.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(event) = output.events.pop_front() {
                    return Some(event);
                }
                if output.done {
                    return None;
                }
            }
            notified.await;
        }
    }
    pub(crate) async fn open(
        ipc: Arc<IPCClient>,
        info: ModelInfo,
        options: DuplexOptions,
    ) -> Result<Self> {
        options.validate()?;
        let request_id = ipc.next_request_id();
        let shared = Arc::new(Shared {
            ipc: Arc::clone(&ipc),
            model_id: info.model_id.clone(),
            request_id,
            response_channel_id: AtomicU64::new(0),
            remote_final: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
            ready: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            ended: AtomicBool::new(false),
            supports_reference: AtomicBool::new(false),
            supports_grounded_response: AtomicBool::new(false),
            supports_response_hold: AtomicBool::new(false),
            autonomous: options.autonomous,
            input: Mutex::new(InputQueue::default()),
            input_wake: Condvar::new(),
            output: Mutex::new(OutputQueue::default()),
            output_ready: Notify::new(),
            opening_error: Mutex::new(None),
            max_pending_frames: options.max_pending_frames,
            dropped_input: AtomicU64::new(0),
            dropped_output: AtomicU64::new(0),
        });
        let prompt = PromptPayload { num_candidates: 1, max_generated_tokens: 1,
            modal_options_json: json!({"duplex_version":1,"sample_rate":SAMPLE_RATE,"frame_samples":FRAME_SAMPLES,
                "channels":1,"max_pending_frames":options.max_pending_frames,"realtime":options.realtime,
                "autonomous":options.autonomous,"seed":options.seed,"max_steps":options.max_steps,
                "audio_temperature":options.audio_temperature,"audio_top_k":options.audio_top_k,
                "text_temperature":options.text_temperature,"text_top_k":options.text_top_k}).to_string(),
            ..Default::default() };
        let weak = Arc::downgrade(&shared);
        let route = ipc.bind_duplex_route(
            request_id,
            Arc::new(move |delta, from_engine| {
                if let Some(shared) = weak.upgrade() {
                    shared.receive(delta, from_engine);
                }
            }),
        )?;
        shared
            .response_channel_id
            .store(route.channel_id, Ordering::Release);
        ipc.send_duplex_request(
            request_id,
            &info.model_id,
            &info.model_path,
            route.channel_id,
            prompt,
        )?;
        let worker = Arc::clone(&shared);
        if let Err(error) = std::thread::Builder::new()
            .name("orchard-duplex-ipc".into())
            .spawn(move || input_worker(worker))
        {
            let _ = shared.command("duplex_close", 0, json!({}));
            return Err(error.into());
        }
        let session = Self {
            shared,
            _route: route,
        };
        let ready = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let notified = session.shared.output_ready.notified();
                if session.shared.ready.load(Ordering::Acquire) {
                    return Ok(());
                }
                if session.shared.ended.load(Ordering::Acquire) {
                    return Err(Error::Other(
                        session
                            .shared
                            .opening_error
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .clone()
                            .unwrap_or_else(|| "PIE closed duplex before ready".into()),
                    ));
                }
                notified.await;
            }
        })
        .await;
        match ready {
            Ok(Ok(())) => Ok(session),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(Error::Other("PIE duplex ready deadline exceeded".into())),
        }
    }
}
impl Drop for DuplexSession {
    fn drop(&mut self) {
        self.shared.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared() -> Arc<Shared> {
        Arc::new(Shared {
            ipc: Arc::new(IPCClient::new()),
            model_id: "test".into(),
            request_id: 1,
            response_channel_id: AtomicU64::new(2),
            remote_final: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
            ready: AtomicBool::new(true),
            closed: AtomicBool::new(false),
            ended: AtomicBool::new(false),
            supports_reference: AtomicBool::new(true),
            supports_grounded_response: AtomicBool::new(false),
            supports_response_hold: AtomicBool::new(false),
            autonomous: true,
            input: Mutex::new(InputQueue::default()),
            input_wake: Condvar::new(),
            output: Mutex::new(OutputQueue::default()),
            output_ready: Notify::new(),
            opening_error: Mutex::new(None),
            max_pending_frames: 2,
            dropped_input: AtomicU64::new(0),
            dropped_output: AtomicU64::new(0),
        })
    }

    fn audio(epoch: u64, sequence: u64) -> DuplexEvent {
        DuplexEvent::Audio {
            epoch,
            sequence,
            pcm: vec![0.25; FRAME_SAMPLES],
            compute_ms: 12.5,
            queue_ms: 0.5,
        }
    }

    #[test]
    fn native_options_are_explicit_and_reject_unsupported_legacy_execution() {
        let mut options = DuplexOptions::default();
        assert!(options.autonomous && !options.realtime && !options.mimi_cpu);
        assert!(options.validate().is_ok());
        options.max_steps = 45_001;
        assert!(options.validate().is_err());
        options.max_steps = 45_000;
        options.audio_top_k = 0;
        options.text_top_k = 0;
        assert!(options.validate().is_ok());
        options.mimi_cpu = true;
        assert!(options.validate().is_err());
    }

    #[test]
    fn bounded_pcm_admission_counts_overflow_and_rejects_invalid_sequences() {
        let shared = shared();
        let control = DuplexControl {
            shared: Arc::clone(&shared),
        };
        assert!(control
            .push_audio(0, vec![f32::NAN; FRAME_SAMPLES])
            .is_err());
        assert!(control.push_audio(0, vec![0.0; FRAME_SAMPLES - 1]).is_err());
        assert_eq!(
            control
                .push_audio(0, vec![0.0; FRAME_SAMPLES])
                .unwrap()
                .dropped_frames,
            0
        );
        assert!(control.push_audio(0, vec![0.0; FRAME_SAMPLES]).is_err());
        assert_eq!(
            control
                .push_audio(1, vec![0.0; FRAME_SAMPLES])
                .unwrap()
                .dropped_frames,
            0
        );
        assert_eq!(
            control
                .push_audio(2, vec![0.0; FRAME_SAMPLES])
                .unwrap()
                .dropped_frames,
            1
        );
        assert_eq!(shared.input.lock().unwrap().frames.len(), 2);
        assert_eq!(shared.dropped_input.load(Ordering::Acquire), 1);
        shared.ready.store(false, Ordering::Release);
        assert!(control.interrupt().is_err());
        assert_eq!(control.epoch(), 0);
    }

    #[test]
    fn late_epoch_events_cannot_erase_new_audio_or_reintroduce_old_output() {
        let shared = shared();
        let control = DuplexControl {
            shared: Arc::clone(&shared),
        };
        shared.emit(audio(0, 0));
        shared.update_epoch(2);
        control.push_audio(1, vec![0.0; FRAME_SAMPLES]).unwrap();
        shared.emit(audio(2, 1));
        shared.update_epoch(1);
        shared.emit(audio(1, 2));
        assert_eq!(control.epoch(), 2);
        assert_eq!(
            shared.input.lock().unwrap().frames.front().unwrap().epoch,
            2
        );
        let output = shared.output.lock().unwrap();
        assert_eq!(output.events.len(), 1);
        assert_eq!(event_epoch(output.events.front().unwrap()), Some(2));
    }

    #[test]
    fn continuous_output_is_bounded_with_explicit_drop_accounting() {
        let shared = shared();
        for sequence in 0..70 {
            shared.emit(audio(0, sequence));
        }
        assert_eq!(shared.output.lock().unwrap().events.len(), MAX_OUTPUT_AUDIO);
        assert_eq!(shared.dropped_output.load(Ordering::Acquire), 6);
        for _ in 0..MAX_EVENTS {
            shared.emit(DuplexEvent::SpeechDone { epoch: 0 });
        }
        assert!(shared.closed.load(Ordering::Acquire));
        assert!(shared.output.lock().unwrap().done);
        assert_eq!(shared.output.lock().unwrap().events.len(), 2);
    }

    #[test]
    fn every_terminal_path_synchronizes_with_the_input_wait_predicate() {
        for terminal in ["close", "finish", "overflow"] {
            let shared = shared();
            if terminal == "overflow" {
                for _ in 0..MAX_EVENTS {
                    shared.emit(DuplexEvent::SpeechDone { epoch: 0 });
                }
            }
            // Reproduce the worker's critical interval after checking closed
            // and before Condvar::wait atomically releases this mutex.
            let input = shared.input.lock().unwrap();
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let worker_shared = Arc::clone(&shared);
            let worker = std::thread::spawn(move || {
                started_tx.send(()).unwrap();
                match terminal {
                    "close" => worker_shared.close(),
                    "finish" => worker_shared.finish(None),
                    "overflow" => worker_shared.emit(DuplexEvent::SpeechDone { epoch: 0 }),
                    _ => unreachable!(),
                }
                done_tx.send(()).unwrap();
            });
            started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            let finished_early = done_rx.recv_timeout(Duration::from_millis(100)).is_ok();
            let closed_early = shared.closed.load(Ordering::Acquire);
            drop(input);
            if !finished_early {
                done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            }
            worker.join().unwrap();
            assert!(
                !finished_early && !closed_early,
                "{terminal} bypassed the wait mutex"
            );
            assert!(shared.closed.load(Ordering::Acquire));
        }
    }

    #[test]
    fn wire_pcm_is_preserved_and_required_timing_is_not_fabricated() {
        let mut delta = ResponseDelta {
            modal_event: Some("duplex.audio".into()),
            modal_metadata_json: Some(
                json!({"duplex_version":1,"epoch":3,"sequence":7,"compute_ms":12.5,"queue_ms":0.5})
                    .to_string(),
            ),
            modal_bytes_b64: Some(
                STANDARD.encode(
                    vec![0.25f32; FRAME_SAMPLES]
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ),
            ..Default::default()
        };
        match decode_event(&delta).unwrap().unwrap() {
            DuplexEvent::Audio {
                epoch,
                sequence,
                pcm,
                compute_ms,
                queue_ms,
            } => {
                assert_eq!((epoch, sequence), (3, 7));
                assert_eq!(pcm, vec![0.25; FRAME_SAMPLES]);
                assert_eq!((compute_ms, queue_ms), (12.5, 0.5));
            }
            _ => panic!("expected actual PCM event"),
        }
        delta.modal_metadata_json =
            Some(json!({"duplex_version":1,"epoch":3,"sequence":7}).to_string());
        assert!(decode_event(&delta).is_err());
    }

    #[test]
    fn retrieval_correlation_preserves_distinct_native_counters_and_unknown_input() {
        let decode = |metadata: Value| {
            decode_event(&ResponseDelta {
                modal_event: Some("duplex.retrieval_requested".into()),
                modal_metadata_json: Some(metadata.to_string()),
                ..Default::default()
            })
        };
        for (input, step) in [(0, 1), (123, 987), (u64::MAX, u64::MAX)] {
            let event = decode(json!({"duplex_version":1,"epoch":3,"sequence":7,
                "input_sequence":input,"model_step":step}))
            .unwrap()
            .unwrap();
            assert!(matches!(&event, DuplexEvent::RetrievalRequested {
                epoch: 3, sequence: 7, input_sequence: Some(actual_input), model_step: Some(actual_step),
            } if *actual_input == input && *actual_step == step));
            // The CLI serializes this same event. Neither input identity nor
            // the native clock may disappear on the next protocol boundary.
            let forwarded = serde_json::to_value(&event).unwrap();
            assert_eq!(forwarded["type"], "retrieval_requested");
            assert_eq!(forwarded["input_sequence"], input);
            assert_eq!(forwarded["model_step"], step);
        }
        for metadata in [
            json!({"duplex_version":1,"epoch":3,"sequence":7}),
            json!({"duplex_version":1,"epoch":3,"sequence":7,"input_sequence":null,"model_step":null}),
        ] {
            assert!(matches!(
                decode(metadata).unwrap().unwrap(),
                DuplexEvent::RetrievalRequested {
                    epoch: 3,
                    sequence: 7,
                    input_sequence: None,
                    model_step: None,
                }
            ));
        }
        assert!(matches!(
            decode(json!({"duplex_version":1,"epoch":3,"sequence":7,
            "input_sequence":null,"model_step":987}))
            .unwrap()
            .unwrap(),
            DuplexEvent::RetrievalRequested {
                input_sequence: None,
                model_step: Some(987),
                ..
            }
        ));
        // Invalid identity must fail closed, not silently become unknown.
        for field in ["input_sequence", "model_step"] {
            for bad in [json!(-1), json!(1.5), json!("123")] {
                let mut metadata = json!({"duplex_version":1,"epoch":3,"sequence":7});
                metadata[field] = bad;
                assert!(decode(metadata).is_err());
            }
        }
    }

    #[test]
    fn generic_engine_failure_before_ready_terminates_waiters() {
        let shared = shared();
        shared.ready.store(false, Ordering::Release);
        shared.receive(
            ResponseDelta {
                request_id: 1,
                is_final_delta: true,
                error: Some("native model rejected".into()),
                ..Default::default()
            },
            true,
        );
        assert!(shared.ended.load(Ordering::Acquire));
        assert!(shared.remote_final.load(Ordering::Acquire));
        assert!(!shared.ready.load(Ordering::Acquire));
        assert_eq!(
            shared.opening_error.lock().unwrap().as_deref(),
            Some("native model rejected")
        );
    }
}
