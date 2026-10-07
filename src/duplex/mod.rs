//! Stateful, full-duplex local audio inference. Moshi and Mimi run in native
//! Rust beside PIE; the same Orchard registry owns all resident models.
//!
//! PCM is mono float32 at 24 kHz, in exactly 1,920-sample frames. Input and
//! output are independent streams. `text_delta` describes the assistant's
//! generated speech text, **not** a transcript of the microphone. The default
//! MoshiRAG model receives factual references from a separate thinking model
//! while continuing to hear PCM; its spoken wording remains model-generated.

mod generation;
mod language;
mod model;
mod quantization;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

use crate::{Error, Result};
pub use model::{
    architecture, is_moshi_model, Checkpoint, DuplexArchitecture, DEFAULT_MODEL, RAG_MODEL,
};
pub(crate) use model::{resolve_files, MoshiModel};

pub const SAMPLE_RATE: u32 = 24_000;
pub const FRAME_SAMPLES: usize = 1_920;
const FRAME_DURATION: Duration = Duration::from_millis(80);
const MAX_SPEECH_CHARS: usize = 8_192;
const MAX_QUEUED_TEXT_TOKENS: usize = 4_096;

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
    /// Run the small streaming codec on CPU while the language model uses Metal.
    pub mimi_cpu: bool,
    /// Bound the CPU codec pool so one audio stream does not consume every core.
    pub codec_threads: usize,
    /// False means only queued PCM advances the model, useful for file clients.
    pub realtime: bool,
    /// True lets Moshi choose its own words; false speaks only queued text.
    pub autonomous: bool,
    pub max_pending_frames: usize,
    /// A bounded timeline is rolled over between utterances without reloading weights.
    pub max_steps: usize,
    /// Maximum natural text-pad wait, in 80 ms frames, before advancing a queued token.
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
            mimi_cpu: true,
            codec_threads: 4,
            realtime: true,
            autonomous: false,
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
        if !(1..=16).contains(&self.codec_threads)
            || !(1..=32).contains(&self.max_pending_frames)
            || !(256..=90_000).contains(&self.max_steps)
            || !(1..=25).contains(&self.text_token_interval)
            || !(1..=2_048).contains(&self.audio_top_k)
            || !(1..=32_000).contains(&self.text_top_k)
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
        sequence: u64,
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
    pcm: Vec<f32>,
    received_at: Instant,
}
enum Command {
    ReferenceReady {
        epoch: u64,
        version: u64,
        encode_ms: f64,
        result: Result<candle_core::Tensor>,
    },
    Speak {
        epoch: u64,
        text: String,
        replace: bool,
    },
    Interrupt {
        epoch: u64,
    },
    Reset {
        epoch: u64,
    },
}
#[derive(Default)]
struct InputQueue {
    frames: VecDeque<InputFrame>,
    commands: VecDeque<Command>,
    last_sequence: Option<u64>,
}
#[derive(Default)]
struct OutputQueue {
    events: VecDeque<DuplexEvent>,
    done: bool,
}

struct ReferenceJob {
    epoch: u64,
    version: u64,
    text: String,
}

struct Shared {
    input: Mutex<InputQueue>,
    wake: Condvar,
    output: Mutex<OutputQueue>,
    output_ready: Notify,
    epoch: AtomicU64,
    closed: AtomicBool,
    dropped_input: AtomicU64,
    dropped_output: AtomicU64,
    max_pending_frames: usize,
    supports_reference: bool,
    reference_input: Mutex<Option<ReferenceJob>>,
    reference_wake: Condvar,
    reference_version: AtomicU64,
}

impl Shared {
    fn new(max_pending_frames: usize, supports_reference: bool) -> Self {
        Self {
            input: Mutex::new(InputQueue::default()),
            wake: Condvar::new(),
            output: Mutex::new(OutputQueue::default()),
            output_ready: Notify::new(),
            epoch: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            dropped_input: AtomicU64::new(0),
            dropped_output: AtomicU64::new(0),
            max_pending_frames,
            supports_reference,
            reference_input: Mutex::new(None),
            reference_wake: Condvar::new(),
            reference_version: AtomicU64::new(0),
        }
    }

    fn stop(&self) {
        self.closed.store(true, Ordering::Release);
        self.wake.notify_all();
        self.reference_wake.notify_all();
    }

    fn emit(&self, event: DuplexEvent) {
        let mut output = self.output.lock().unwrap_or_else(|p| p.into_inner());
        // Cut obsolete output before considering capacity. The frontend also
        // receives epochs because an old frame may already be on the wire.
        let current_epoch = self.epoch.load(Ordering::Acquire);
        output.events.retain(
            |event| !matches!(event, DuplexEvent::Audio { epoch, .. } if *epoch != current_epoch),
        );
        if matches!(&event, DuplexEvent::Audio { epoch, .. } if *epoch != current_epoch) {
            return;
        }
        let capacity = self.max_pending_frames * 4 + 8;
        if output.events.len() >= capacity {
            let oldest_audio = output
                .events
                .iter()
                .position(|event| matches!(event, DuplexEvent::Audio { .. }));
            if let Some(index) = oldest_audio {
                output.events.remove(index);
                self.dropped_output.fetch_add(1, Ordering::Relaxed);
            } else {
                output.events.pop_front();
            }
        }
        output.events.push_back(event);
        drop(output);
        self.output_ready.notify_one();
    }

    fn finish(&self) {
        self.stop();
        self.emit(DuplexEvent::Closed);
        self.output.lock().unwrap_or_else(|p| p.into_inner()).done = true;
        self.output_ready.notify_one();
    }
}

/// Cloneable producer/control half; none of its methods waits for inference.
#[derive(Clone)]
pub struct DuplexControl {
    shared: Arc<Shared>,
}

impl DuplexControl {
    pub fn supports_reference(&self) -> bool {
        self.shared.supports_reference
    }

    /// Supply facts/tool results through the trained MoshiRAG reference encoder.
    /// Epoch checks prevent a late result from a cancelled turn steering speech.
    pub fn reference(&self, text: String, expected_epoch: u64) -> Result<u64> {
        if !self.supports_reference() {
            return Err(Error::Other(
                "Select MoshiRAG for trained reference conditioning".into(),
            ));
        }
        if text.trim().is_empty() || text.chars().count() > 4096 {
            return Err(Error::Other(
                "A speech reference must contain 1 to 4096 characters".into(),
            ));
        }
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(Error::ChannelClosed);
        }
        let _input = self.shared.input.lock().unwrap_or_else(|p| p.into_inner());
        if self.epoch() != expected_epoch {
            return Err(Error::Other(
                "Speech reference belongs to an interrupted epoch".into(),
            ));
        }
        // Match the factual-document format used to train and serve MoshiRAG.
        // Callers should keep one reference concise (about fifty words).
        let text = if text.trim_start().starts_with("Reference:") {
            text.trim().to_owned()
        } else {
            format!("Reference: {}", text.trim())
        };
        let version = self.shared.reference_version.fetch_add(1, Ordering::AcqRel) + 1;
        *self
            .shared
            .reference_input
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(ReferenceJob {
            epoch: expected_epoch,
            version,
            text,
        });
        self.shared.emit(DuplexEvent::ReferenceQueued {
            epoch: expected_epoch,
            version,
        });
        self.shared.reference_wake.notify_one();
        Ok(version)
    }

    pub fn epoch(&self) -> u64 {
        self.shared.epoch.load(Ordering::Acquire)
    }

    pub fn push_audio(&self, sequence: u64, pcm: Vec<f32>) -> Result<AudioAdmission> {
        if pcm.len() != FRAME_SAMPLES || pcm.iter().any(|x| !x.is_finite() || x.abs() > 1.0) {
            return Err(Error::Other("Duplex PCM must be exactly 1920 finite mono float32 samples in [-1, 1] at 24000 Hz".into()));
        }
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(Error::ChannelClosed);
        }
        let mut input = self.shared.input.lock().unwrap_or_else(|p| p.into_inner());
        if input.last_sequence.is_some_and(|last| sequence <= last) {
            return Err(Error::Other(
                "Duplex input sequence must increase strictly".into(),
            ));
        }
        input.last_sequence = Some(sequence);
        let dropped_frames = if input.frames.len() >= self.shared.max_pending_frames {
            input.frames.pop_front();
            self.shared.dropped_input.fetch_add(1, Ordering::Relaxed);
            1
        } else {
            0
        };
        input.frames.push_back(InputFrame {
            sequence,
            pcm,
            received_at: Instant::now(),
        });
        drop(input);
        self.shared.wake.notify_one();
        Ok(AudioAdmission {
            epoch: self.epoch(),
            dropped_frames,
        })
    }

    /// Append a bounded chunk of the backbone's speech. `replace` starts a new
    /// epoch, so audio already sent for the previous reply can be discarded.
    pub fn speak(&self, text: String, replace: bool) -> Result<u64> {
        if self.supports_reference() {
            let epoch = self.epoch();
            self.reference(text, epoch)?;
            return Ok(epoch);
        }
        if text.trim().is_empty() || text.chars().count() > MAX_SPEECH_CHARS {
            return Err(Error::Other(
                "Speech text must contain 1 to 8192 characters".into(),
            ));
        }
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(Error::ChannelClosed);
        }
        let mut input = self.shared.input.lock().unwrap_or_else(|p| p.into_inner());
        if input.commands.len() >= 32 {
            return Err(Error::Other("Duplex speech/control queue is full".into()));
        }
        let epoch = if replace {
            input
                .commands
                .retain(|c| !matches!(c, Command::Speak { .. }));
            self.shared.epoch.fetch_add(1, Ordering::AcqRel) + 1
        } else {
            self.epoch()
        };
        input.commands.push_back(Command::Speak {
            epoch,
            text,
            replace,
        });
        drop(input);
        self.shared.wake.notify_one();
        Ok(epoch)
    }

    pub fn interrupt(&self) -> Result<u64> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(Error::ChannelClosed);
        }
        let mut input = self.shared.input.lock().unwrap_or_else(|p| p.into_inner());
        let epoch = self.shared.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        input.commands.clear();
        input.commands.push_front(Command::Interrupt { epoch });
        drop(input);
        self.shared.wake.notify_one();
        Ok(epoch)
    }

    pub fn reset(&self) -> Result<u64> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(Error::ChannelClosed);
        }
        let mut input = self.shared.input.lock().unwrap_or_else(|p| p.into_inner());
        let epoch = self.shared.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        input.commands.clear();
        input.frames.clear();
        input.last_sequence = None;
        input.commands.push_front(Command::Reset { epoch });
        drop(input);
        self.shared.wake.notify_one();
        Ok(epoch)
    }

    pub fn close(&self) -> Result<()> {
        self.shared.stop();
        Ok(())
    }
}

/// An Orchard-owned inference task. Dropping it closes the native audio worker.
pub struct DuplexSession {
    shared: Arc<Shared>,
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
                let mut output = self.shared.output.lock().unwrap_or_else(|p| p.into_inner());
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

    pub(crate) fn start(model: Arc<MoshiModel>, options: DuplexOptions) -> Result<Self> {
        options.validate()?;
        let pool = codec_pool(options.codec_threads)?;
        let model_session = model.acquire(&options)?;
        let shared = Arc::new(Shared::new(options.max_pending_frames, model.is_rag()));
        if model.is_rag() {
            let encoder_shared = Arc::clone(&shared);
            let encoder_model = Arc::clone(&model);
            std::thread::Builder::new()
                .name("orchard-moshi-reference".into())
                .spawn(move || reference_worker(encoder_shared, encoder_model))?;
        }
        let worker_shared = Arc::clone(&shared);
        let model_id = model.model_id.clone();
        let worker = std::thread::Builder::new()
            .name("orchard-moshi".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    pool.install(|| run(worker_shared.as_ref(), model_id, model_session, &options))
                }));
                match result {
                    Ok(Err(error)) => worker_shared.emit(DuplexEvent::Error {
                        message: error.to_string(),
                    }),
                    Err(_) => worker_shared.emit(DuplexEvent::Error {
                        message: "Moshi native inference task panicked".into(),
                    }),
                    Ok(Ok(())) => {}
                }
                worker_shared.finish();
            });
        if let Err(error) = worker {
            shared.stop();
            return Err(Error::Io(error));
        }
        Ok(Self { shared })
    }
}

impl Drop for DuplexSession {
    fn drop(&mut self) {
        self.shared.stop();
    }
}

pub(super) fn codec_pool(threads: usize) -> Result<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("orchard-mimi-{i}"))
        .build()
        .map_err(|e| Error::Other(format!("Mimi CPU pool: {e}")))
}

fn reference_worker(shared: Arc<Shared>, model: Arc<MoshiModel>) {
    loop {
        let job = {
            let mut input = shared
                .reference_input
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            while input.is_none() && !shared.closed.load(Ordering::Acquire) {
                input = shared
                    .reference_wake
                    .wait(input)
                    .unwrap_or_else(|p| p.into_inner());
            }
            if shared.closed.load(Ordering::Acquire) {
                return;
            }
            input.take().expect("reference available")
        };
        if job.epoch != shared.epoch.load(Ordering::Acquire)
            || job.version != shared.reference_version.load(Ordering::Acquire)
        {
            continue;
        }
        let began = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.encode_reference(&job.text)
        }))
        .unwrap_or_else(|_| Err(Error::Other("MoshiRAG reference encoder panicked".into())));
        let encode_ms = began.elapsed().as_secs_f64() * 1000.0;
        if shared.closed.load(Ordering::Acquire) {
            return;
        }
        let mut input = shared.input.lock().unwrap_or_else(|p| p.into_inner());
        input.commands.push_back(Command::ReferenceReady {
            epoch: job.epoch,
            version: job.version,
            encode_ms,
            result,
        });
        drop(input);
        shared.wake.notify_one();
    }
}

struct Speech {
    tokens: VecDeque<u32>,
    tail: usize,
    end_pending: bool,
}

impl Speech {
    fn new() -> Self {
        Self {
            tokens: VecDeque::new(),
            tail: 0,
            end_pending: false,
        }
    }
    fn active(&self) -> bool {
        !self.tokens.is_empty() || self.tail != 0 || self.end_pending
    }
    fn requested(&self) -> u32 {
        self.tokens
            .front()
            .copied()
            .unwrap_or(if self.end_pending { 0 } else { 3 })
    }
    fn advance(&mut self, consumed: bool) {
        if !self.tokens.is_empty() {
            if consumed {
                self.tokens.pop_front();
                self.tail = 8;
                self.end_pending = self.tokens.is_empty();
            }
        } else if self.end_pending {
            if consumed {
                self.end_pending = false;
            }
        } else {
            self.tail = self.tail.saturating_sub(1);
        }
    }
}

enum Work {
    Command(Command),
    Frame(InputFrame),
    Silence,
    End,
}

fn take_work(shared: &Shared, options: &DuplexOptions, deadline: Instant) -> Work {
    let mut input = shared.input.lock().unwrap_or_else(|p| p.into_inner());
    loop {
        if shared.closed.load(Ordering::Acquire) {
            return Work::End;
        }
        if let Some(command) = input.commands.pop_front() {
            return Work::Command(command);
        }
        let now = Instant::now();
        if !options.realtime || now >= deadline {
            if let Some(frame) = input.frames.pop_front() {
                return Work::Frame(frame);
            }
            if options.realtime {
                return Work::Silence;
            }
        }
        input = if options.realtime {
            shared
                .wake
                .wait_timeout(input, deadline.saturating_duration_since(now))
                .unwrap_or_else(|p| p.into_inner())
                .0
        } else {
            shared.wake.wait(input).unwrap_or_else(|p| p.into_inner())
        };
    }
}

fn run(
    shared: &Shared,
    model_id: String,
    mut model: model::ModelSession,
    options: &DuplexOptions,
) -> Result<()> {
    shared.emit(DuplexEvent::Ready {
        model_id,
        sample_rate: SAMPLE_RATE,
        frame_samples: FRAME_SAMPLES,
        epoch: 0,
    });
    let started = Instant::now();
    let mut next_deadline = started + FRAME_DURATION;
    let mut metrics = DuplexMetrics::default();
    let mut speech = Speech::new();
    let mut output_sequence = 0;
    let mut frame_count = 0;
    let is_rag = model.is_rag();
    let mut rag_can_speak = options.autonomous;
    let mut reference_receipt: Option<(u64, u64, usize, f64)> = None;
    loop {
        let (pcm, queue_ms) = match take_work(shared, options, next_deadline) {
            Work::End => break,
            Work::Command(command) => {
                match command {
                    Command::ReferenceReady {
                        epoch,
                        version,
                        encode_ms,
                        result,
                    } => {
                        if epoch != shared.epoch.load(Ordering::Acquire)
                            || version != shared.reference_version.load(Ordering::Acquire)
                        {
                            continue;
                        }
                        match result {
                            Ok(embedding) => {
                                let steps =
                                    embedding.dim(1).map_err(|e| Error::Other(e.to_string()))?;
                                model.set_reference(embedding)?;
                                reference_receipt = Some((epoch, version, steps, encode_ms));
                            }
                            Err(error) => shared.emit(DuplexEvent::Error {
                                message: error.to_string(),
                            }),
                        }
                    }
                    Command::Speak {
                        epoch,
                        text,
                        replace,
                    } => {
                        if epoch != shared.epoch.load(Ordering::Acquire) {
                            continue;
                        }
                        let tokens = model.encode_text(&text)?;
                        if tokens.len() + if replace { 0 } else { speech.tokens.len() }
                            > MAX_QUEUED_TEXT_TOKENS
                        {
                            shared.emit(DuplexEvent::Error {
                                message: "Speech queue exceeds 4096 text tokens".into(),
                            });
                            continue;
                        }
                        if replace {
                            model.reset(options);
                            speech = Speech::new();
                        }
                        let count = tokens.len();
                        speech.tokens.extend(tokens);
                        speech.end_pending = false;
                        shared.emit(DuplexEvent::SpeechQueued {
                            epoch,
                            tokens: count,
                        });
                    }
                    Command::Interrupt { epoch } => {
                        if is_rag {
                            model.clear_reference();
                            rag_can_speak = false;
                            reference_receipt = None;
                        } else {
                            model.reset(options);
                        }
                        speech = Speech::new();
                        shared.emit(DuplexEvent::Interrupted { epoch });
                    }
                    Command::Reset { epoch } => {
                        model.reset(options);
                        rag_can_speak = options.autonomous;
                        reference_receipt = None;
                        speech = Speech::new();
                        shared.emit(DuplexEvent::Reset {
                            epoch,
                            reason: "requested".into(),
                        });
                    }
                }
                continue;
            }
            Work::Frame(frame) => {
                let _input_sequence = frame.sequence;
                metrics.input_frames += 1;
                (
                    frame.pcm,
                    frame.received_at.elapsed().as_secs_f64() * 1_000.0,
                )
            }
            Work::Silence => {
                metrics.synthetic_silence_frames += 1;
                (vec![0.0; FRAME_SAMPLES], 0.0)
            }
        };
        // Rotate finite upstream generation bookkeeping, retaining the queued
        // external-language-model speech and resident model weights.
        if model.step_idx() + 16 >= options.max_steps {
            model.reset(options);
            if is_rag {
                rag_can_speak = options.autonomous;
                reference_receipt = None;
            }
            let epoch = shared.epoch.fetch_add(1, Ordering::AcqRel) + 1;
            shared.emit(DuplexEvent::Reset {
                epoch,
                reason: "timeline_rollover".into(),
            });
        }
        let epoch = shared.epoch.load(Ordering::Acquire);
        let was_speaking = speech.active();
        let forced_text = if is_rag {
            if rag_can_speak {
                None
            } else {
                Some(3)
            }
        } else if options.autonomous && !speech.active() {
            None
        } else {
            Some(speech.requested())
        };
        let began = Instant::now();
        let mut frame = model.process(&pcm, forced_text)?;
        speech.advance(frame.consumed_text);
        let compute_ms = began.elapsed().as_secs_f64() * 1_000.0;
        metrics.compute_ms_total += compute_ms;
        metrics.compute_ms_max = metrics.compute_ms_max.max(compute_ms);
        metrics.queue_ms_max = metrics.queue_ms_max.max(queue_ms);
        if shared.epoch.load(Ordering::Acquire) == epoch {
            if let Some(pcm) = frame.pcm.as_mut() {
                if (is_rag && !rag_can_speak) || (!is_rag && !options.autonomous && !was_speaking) {
                    pcm.fill(0.0);
                }
                if pcm.iter().any(|sample| !sample.is_finite()) {
                    return Err(Error::Other("Moshi decoder produced non-finite PCM".into()));
                }
                shared.emit(DuplexEvent::Audio {
                    epoch,
                    sequence: output_sequence,
                    pcm: std::mem::take(pcm),
                    compute_ms,
                    queue_ms,
                });
                metrics.output_frames += 1;
            }
            if let Some(text) = frame.text {
                if (is_rag && rag_can_speak) || (!is_rag && (options.autonomous || was_speaking)) {
                    shared.emit(DuplexEvent::TextDelta {
                        epoch,
                        sequence: output_sequence,
                        text,
                    });
                }
            }
            if frame.retrieval_requested && is_rag && rag_can_speak {
                shared.emit(DuplexEvent::RetrievalRequested {
                    epoch,
                    sequence: output_sequence,
                });
            }
            if model.reference_remaining() == 0 {
                if let Some((reference_epoch, version, steps, encode_ms)) = reference_receipt.take()
                {
                    if reference_epoch == epoch {
                        rag_can_speak = true;
                        shared.emit(DuplexEvent::ReferenceApplied {
                            epoch,
                            version,
                            steps,
                            encode_ms,
                        });
                    }
                }
            }
            if was_speaking && !speech.active() {
                shared.emit(DuplexEvent::SpeechDone { epoch });
            }
        }
        output_sequence += 1;
        frame_count += 1;
        if frame_count % 25 == 0 {
            metrics.elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            metrics.dropped_input_frames = shared.dropped_input.load(Ordering::Relaxed);
            metrics.dropped_output_frames = shared.dropped_output.load(Ordering::Relaxed);
            shared.emit(DuplexEvent::Metrics {
                metrics: metrics.clone(),
            });
        }
        // Keep the standalone clock anchored: after a slow frame, overdue
        // steps can catch up instead of permanently adding input latency.
        // With realtime=false the microphone clock owns pacing entirely.
        next_deadline += FRAME_DURATION;
    }
    metrics.elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    metrics.dropped_input_frames = shared.dropped_input.load(Ordering::Relaxed);
    metrics.dropped_output_frames = shared.dropped_output.load(Ordering::Relaxed);
    shared.emit(DuplexEvent::Metrics { metrics });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn control(capacity: usize) -> DuplexControl {
        DuplexControl {
            shared: Arc::new(Shared::new(capacity, false)),
        }
    }

    #[test]
    fn pcm_queue_discards_oldest_and_rejects_invalid_or_repeated_frames() {
        let control = control(2);
        assert!(control
            .push_audio(0, vec![f32::NAN; FRAME_SAMPLES])
            .is_err());
        assert!(control.push_audio(0, vec![0.0; FRAME_SAMPLES - 1]).is_err());
        control.push_audio(0, vec![0.0; FRAME_SAMPLES]).unwrap();
        control.push_audio(1, vec![0.0; FRAME_SAMPLES]).unwrap();
        assert_eq!(
            control
                .push_audio(2, vec![0.0; FRAME_SAMPLES])
                .unwrap()
                .dropped_frames,
            1
        );
        let input = control.shared.input.lock().unwrap();
        assert_eq!(
            input.frames.iter().map(|f| f.sequence).collect::<Vec<_>>(),
            [1, 2]
        );
        drop(input);
        assert!(control.push_audio(2, vec![0.0; FRAME_SAMPLES]).is_err());
    }

    #[test]
    fn barge_in_invalidates_audio_already_waiting_without_waiting_for_inference() {
        let control = control(2);
        let old = control.speak("An old answer".into(), true).unwrap();
        control.shared.emit(DuplexEvent::Audio {
            epoch: old,
            sequence: 0,
            pcm: vec![0.5],
            compute_ms: 1.0,
            queue_ms: 0.0,
        });
        let new = control.interrupt().unwrap();
        assert!(new > old);
        control.shared.emit(DuplexEvent::Audio {
            epoch: old,
            sequence: 1,
            pcm: vec![0.5],
            compute_ms: 1.0,
            queue_ms: 0.0,
        });
        assert!(control.shared.output.lock().unwrap().events.is_empty());
        let input = control.shared.input.lock().unwrap();
        assert_eq!(input.commands.len(), 1);
        assert!(matches!(input.commands[0], Command::Interrupt { epoch } if epoch == new));
    }

    #[test]
    fn a_waiting_word_survives_natural_padding_and_audio_has_a_bounded_tail() {
        let mut speech = Speech::new();
        speech.tokens.extend([100, 101]);
        assert_eq!(speech.requested(), 100);
        speech.advance(false);
        assert_eq!(speech.requested(), 100);
        speech.advance(true);
        assert_eq!(speech.requested(), 101);
        speech.advance(true);
        assert_eq!(speech.requested(), 0);
        speech.advance(false);
        assert_eq!(speech.requested(), 0);
        speech.advance(true);
        assert_eq!(speech.requested(), 3);
        for _ in 0..8 {
            speech.advance(false);
        }
        assert!(!speech.active());
    }

    #[test]
    fn reference_results_cannot_cross_barge_in_and_pending_context_is_bounded() {
        let control = DuplexControl {
            shared: Arc::new(Shared::new(2, true)),
        };
        let first = control.reference("old fact".into(), 0).unwrap();
        let second = control.reference("new fact".into(), 0).unwrap();
        assert!(second > first);
        {
            let pending = control.shared.reference_input.lock().unwrap();
            let pending = pending.as_ref().unwrap();
            assert_eq!(pending.version, second);
            assert_eq!(pending.text, "Reference: new fact");
        }
        let epoch = control.interrupt().unwrap();
        assert!(control.reference("late tool result".into(), 0).is_err());
        control
            .reference("Reference: current fact".into(), epoch)
            .unwrap();
        let pending = control.shared.reference_input.lock().unwrap();
        assert_eq!(pending.as_ref().unwrap().text, "Reference: current fact");
        assert_eq!(pending.as_ref().unwrap().epoch, epoch);
    }

    #[test]
    fn native_profile_has_matching_codec_and_model_revisions() {
        let profile = architecture().unwrap();
        assert_eq!(profile.frame_samples * 25, profile.sample_rate as usize * 2);
        assert_eq!(profile.checkpoints[DEFAULT_MODEL].revision.len(), 40);
        assert!(is_moshi_model(DEFAULT_MODEL));
        assert!(!is_moshi_model("google/gemma-4-E2B-it"));
    }
}
