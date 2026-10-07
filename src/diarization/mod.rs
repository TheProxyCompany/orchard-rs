//! NVIDIA Nemotron 3 streaming speaker diarization, owned by Orchard.
//!
//! Eight arrival-order acoustic speaker tracks at 10 ms resolution, with the
//! actual NVIDIA AOSC/FIFO state. Speaker IDs are anonymous and session-local;
//! this module never invents a person's identity or collapses overlapping speech.
mod native;

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hf_hub::{Cache, Repo, RepoType};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use crate::{Error, Result};
pub(crate) use native::NativeModel;
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
    pub(crate) fn model_key(&self, model_id: &str) -> String {
        format!(
            "{model_id}:{:?}:{}:{}:{}:{}:{}",
            self.device,
            self.chunk_frames,
            self.right_context_frames,
            self.fifo_frames,
            self.speaker_cache_frames,
            self.update_period_frames
        )
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

enum Command {
    Audio { pcm: Vec<f32>, missing_frames: u64 },
    Finish,
    Close,
}

#[derive(Clone)]
pub struct DiarizationControl {
    input: mpsc::Sender<Command>,
    closed: Arc<AtomicBool>,
    last_sequence: Arc<Mutex<Option<u64>>>,
    frame_samples: usize,
}
impl DiarizationControl {
    /// Accept one 80 ms frame without waiting for inference. Queue pressure is
    /// explicit. If the caller skips a rejected sequence, the next admission
    /// records a gap and preserves the timestamp with the missing duration.
    pub fn push_audio(&self, sequence: u64, pcm: Vec<f32>) -> Result<DiarizationAdmission> {
        if pcm.len() != self.frame_samples || pcm.iter().any(|x| !x.is_finite() || x.abs() > 1.0) {
            return Err(Error::Other(
                "Diarization requires one 80 ms frame of finite mono PCM in [-1, 1]".into(),
            ));
        }
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::ChannelClosed);
        }
        let mut last = self.last_sequence.lock().unwrap_or_else(|p| p.into_inner());
        if last.is_some_and(|last| sequence <= last) {
            return Err(Error::Other(
                "Diarization sequence must increase strictly".into(),
            ));
        }
        let missing_frames = last.map_or(0, |last| sequence - last - 1);
        if missing_frames > 50 {
            return Err(Error::Other(
                "Diarization input gap exceeds four seconds; open a new session".into(),
            ));
        }
        self.input
            .try_send(Command::Audio {
                pcm,
                missing_frames,
            })
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => {
                    Error::Other("Diarization input queue is full".into())
                }
                mpsc::error::TrySendError::Closed(_) => Error::ChannelClosed,
            })?;
        *last = Some(sequence);
        Ok(DiarizationAdmission { missing_frames })
    }
    /// Flush every accepted frame, including the last partial inference chunk.
    pub async fn finish(&self) -> Result<()> {
        self.input
            .send(Command::Finish)
            .await
            .map_err(|_| Error::ChannelClosed)
    }
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        let _ = self.input.try_send(Command::Close);
    }
}

pub struct DiarizationSession {
    control: DiarizationControl,
    events: mpsc::Receiver<DiarizationEvent>,
}
impl DiarizationSession {
    pub fn control(&self) -> DiarizationControl {
        self.control.clone()
    }
    pub async fn next_event(&mut self) -> Option<DiarizationEvent> {
        self.events.recv().await
    }
    pub(crate) fn start(model: Arc<NativeModel>, options: DiarizationOptions) -> Result<Self> {
        options.validate()?;
        let session_id = options
            .session_id
            .clone()
            .unwrap_or_else(|| format!("diar-{:016x}", rand::random::<u64>()));
        let (input, receiver) = mpsc::channel(options.max_pending_frames);
        let (output, events) = mpsc::channel(32);
        let closed = Arc::new(AtomicBool::new(false));
        let control = DiarizationControl {
            input,
            closed: Arc::clone(&closed),
            last_sequence: Arc::new(Mutex::new(None)),
            frame_samples: options.frame_samples(),
        };
        std::thread::Builder::new()
            .name("orchard-nemotron-diarization".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(
                        Arc::clone(&model),
                        &options,
                        &session_id,
                        receiver,
                        &output,
                        closed.as_ref(),
                    )
                }));
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        let _ = output.blocking_send(DiarizationEvent::Error {
                            message: e.to_string(),
                        });
                    }
                    Err(_) => {
                        let _ = output.blocking_send(DiarizationEvent::Error {
                            message: "Native diarization task panicked".into(),
                        });
                    }
                }
                closed.store(true, Ordering::Release);
                let _ = output.blocking_send(DiarizationEvent::Closed);
            })?;
        Ok(Self { control, events })
    }
}
impl Drop for DiarizationSession {
    fn drop(&mut self) {
        self.control.close();
    }
}

fn publish(
    stream: &native::NativeStream,
    session_id: &str,
    last_frame: &mut u64,
    output: &mpsc::Sender<DiarizationEvent>,
    compute_ms: f64,
    audio_seconds: f64,
    finished: bool,
) -> Result<()> {
    if stream.frame_count()? == *last_frame && !finished {
        return Ok(());
    }
    let snapshot = stream.snapshot()?;
    let frame_start = (*last_frame).max(snapshot.first_frame);
    let offset = (frame_start - snapshot.first_frame) as usize;
    let probabilities = snapshot.probabilities[offset..].to_vec();
    let labeled_seconds = snapshot.frame_count as f64 * 0.01;
    // The native V3 stream commits chunk probabilities (no legacy V2 birth
    // heuristic). Segmentation can still widen/merge a recent edge, so upsert
    // the last two seconds instead of pretending those edges are immutable.
    let replace_from = if finished {
        0.0
    } else {
        (labeled_seconds - 2.0).max(0.0)
    };
    let mut segments = Vec::new();
    for segment in snapshot.segments {
        if segment.end_time < replace_from {
            continue;
        }
        if !(1..=8).contains(&segment.speaker)
            || !segment.start_time.is_finite()
            || !segment.end_time.is_finite()
            || segment.end_time < segment.start_time
        {
            return Err(Error::Other("Native diarization segment is invalid".into()));
        }
        let start = (segment.start_time / 0.01).floor().max(0.0) as u64;
        let end = ((segment.end_time / 0.01).ceil() as u64).min(snapshot.frame_count);
        let mean_probability = if start >= snapshot.first_frame && end > start {
            let sum: f32 = snapshot.probabilities
                [(start - snapshot.first_frame) as usize..(end - snapshot.first_frame) as usize]
                .iter()
                .map(|frame| frame[(segment.speaker - 1) as usize])
                .sum();
            Some(sum / (end - start) as f32)
        } else {
            None
        };
        segments.push(SpeakerSegment {
            speaker_id: format!("{session_id}:speaker-{}", segment.speaker),
            speaker_index: segment.speaker as u8,
            start_seconds: segment.start_time,
            end_seconds: segment.end_time,
            mean_probability,
            provisional: !finished && segment.end_time > labeled_seconds - 1.25,
        });
    }
    *last_frame = snapshot.frame_count;
    output
        .blocking_send(DiarizationEvent::Update {
            session_id: session_id.to_owned(),
            frame_start,
            frame_seconds: 0.01,
            probabilities,
            segments,
            replace_segments_from_seconds: replace_from,
            compute_ms,
            audio_seconds,
            finished,
        })
        .map_err(|_| Error::ChannelClosed)
}

fn run(
    model: Arc<NativeModel>,
    options: &DiarizationOptions,
    session_id: &str,
    mut input: mpsc::Receiver<Command>,
    output: &mpsc::Sender<DiarizationEvent>,
    closed: &AtomicBool,
) -> Result<()> {
    let started = Instant::now();
    let mut stream = native::NativeStream::open(Arc::clone(&model))?;
    output
        .blocking_send(DiarizationEvent::Ready {
            session_id: session_id.into(),
            model_id: DEFAULT_MODEL.into(),
            speakers: 8,
            sample_rate: options.sample_rate,
            frame_samples: options.frame_samples(),
            output_frame_seconds: 0.01,
        })
        .map_err(|_| Error::ChannelClosed)?;
    let (mut frames, mut missing, mut last_frame) = (0_u64, 0_u64, 0_u64);
    let (mut total_ms, mut max_ms) = (0.0_f64, 0.0_f64);
    while let Some(command) = input.blocking_recv() {
        if closed.load(Ordering::Acquire) {
            break;
        }
        match command {
            Command::Close => break,
            Command::Finish => {
                let begin = Instant::now();
                stream.finish()?;
                let ms = begin.elapsed().as_secs_f64() * 1000.0;
                total_ms += ms;
                max_ms = max_ms.max(ms);
                publish(
                    &stream,
                    session_id,
                    &mut last_frame,
                    output,
                    ms,
                    (frames + missing) as f64 * 0.08,
                    true,
                )?;
                break;
            }
            Command::Audio {
                pcm,
                missing_frames,
            } => {
                if missing_frames != 0 {
                    let start = (frames + missing) as f64 * 0.08;
                    output
                        .blocking_send(DiarizationEvent::Gap {
                            session_id: session_id.into(),
                            start_seconds: start,
                            end_seconds: start + missing_frames as f64 * 0.08,
                            missing_frames,
                        })
                        .map_err(|_| Error::ChannelClosed)?;
                }
                let begin = Instant::now();
                for _ in 0..missing_frames {
                    stream.push(&vec![0.0; options.frame_samples()], options.sample_rate)?;
                }
                stream.push(&pcm, options.sample_rate)?;
                let ms = begin.elapsed().as_secs_f64() * 1000.0;
                total_ms += ms;
                max_ms = max_ms.max(ms);
                frames += 1;
                missing += missing_frames;
                publish(
                    &stream,
                    session_id,
                    &mut last_frame,
                    output,
                    ms,
                    (frames + missing) as f64 * 0.08,
                    false,
                )?;
            }
        }
    }
    let _ = output.blocking_send(DiarizationEvent::Metrics {
        session_id: session_id.into(),
        input_frames: frames,
        missing_input_frames: missing,
        output_frames: last_frame,
        compute_ms_total: total_ms,
        compute_ms_max: max_ms,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        model_weight_bytes: model.weight_bytes,
    });
    Ok(())
}

pub(crate) async fn resolve(model_id: &str) -> Result<(PathBuf, PathBuf, DiarizationArchitecture)> {
    let profile = architecture()?;
    let path = if Path::new(model_id).is_file() {
        PathBuf::from(model_id)
    } else {
        if model_id != profile.model_id {
            return Err(Error::ModelNotFound(model_id.into()));
        }
        let cache = Cache::from_env();
        let direct = cache
            .path()
            .join(format!("models--{}", model_id.replace('/', "--")))
            .join("snapshots")
            .join(&profile.revision)
            .join(&profile.model_file);
        if direct.is_file() {
            direct
        } else {
            let api = hf_hub::api::tokio::ApiBuilder::from_env()
                .with_progress(false)
                .build()
                .map_err(|e| Error::HfApiInit(e.to_string()))?;
            api.repo(Repo::with_revision(
                model_id.into(),
                RepoType::Model,
                profile.revision.clone(),
            ))
            .get(&profile.model_file)
            .await
            .map_err(|e| Error::DownloadFailed(model_id.into(), e.to_string()))?
        }
    };
    #[cfg(target_os = "macos")]
    let library_name = "libnemo_speech_asr_c.dylib";
    #[cfg(target_os = "windows")]
    let library_name = "nemo_speech_asr_c.dll";
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let library_name = "libnemo_speech_asr_c.so";
    let bundled = std::env::current_exe().ok().and_then(|exe| {
        let directory = exe.parent()?;
        [
            directory.join(library_name),
            directory.join("../Frameworks").join(library_name),
        ]
        .into_iter()
        .find(|path| path.is_file())
    });
    let library = if let Some(path) = std::env::var_os("ORCHARD_NEMO_LIBRARY") {
        PathBuf::from(path)
    } else if let Some(path) = bundled {
        path
    } else {
        let engine = crate::EngineFetcher::new().get_engine_path().await?;
        let root = engine
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| Error::Other("Invalid Orchard engine bundle".into()))?;
        root.join("lib").join(library_name)
    };
    if !library.is_file() {
        return Err(Error::ModelNotReady(format!("This Orchard engine lacks the native Nemotron3 architecture: {}. Update or build the engine bundle.",library.display())));
    }
    Ok((library, path, profile))
}

pub(crate) fn load_verified(
    library: PathBuf,
    path: PathBuf,
    profile: DiarizationArchitecture,
    options: DiarizationOptions,
) -> Result<NativeModel> {
    let mut file = std::fs::File::open(&path)?;
    if file.metadata()?.len() != profile.size_bytes {
        return Err(Error::Other(
            "Nemotron3 model size differs from the pinned official checkpoint".into(),
        ));
    }
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    let actual = format!("{:x}", hash.finalize());
    if actual != profile.sha256 {
        return Err(Error::Integrity {
            expected: profile.sha256,
            actual,
        });
    }
    NativeModel::load(library, path, &options)
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
