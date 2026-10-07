//! Safe ownership around NVIDIA's pinned, stable diarization C ABI.
//! Header: NeMo-Speech.cpp/include/nemo_speech/diar.h at b809bbb4.
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::{DiarizationDevice, DiarizationOptions};
use crate::{Error, Result};
use libloading::Library;

#[repr(C)]
struct ModelConfig {
    size: usize,
    model_path: *const c_char,
    gpu: i32,
    preset: *const c_char,
    chunk_frames: i32,
    right_context_frames: i32,
    left_context_frames: i32,
    fifo_frames: i32,
    spkcache_frames: i32,
    update_period_frames: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Segment {
    pub start_time: f64,
    pub end_time: f64,
    pub speaker: i32,
}

type Create = unsafe extern "C" fn(*const ModelConfig, *mut *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Speakers = unsafe extern "C" fn(*const c_void) -> i32;
type FrameSeconds = unsafe extern "C" fn(*const c_void) -> f64;
type Open = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32;
type Push = unsafe extern "C" fn(*mut c_void, *const f32, usize, i32) -> i32;
type Finish = unsafe extern "C" fn(*mut c_void) -> i32;
type FrameCount = unsafe extern "C" fn(*const c_void) -> i64;
type Probabilities = unsafe extern "C" fn(*const c_void, *mut f32, usize) -> i32;
type Segments =
    unsafe extern "C" fn(*const c_void, *const c_void, *mut Segment, usize, *mut usize) -> i32;
type LastError = unsafe extern "C" fn() -> *const c_char;

struct Api {
    _library: Library,
    create: Create,
    destroy: Destroy,
    speakers: Speakers,
    frame_seconds: FrameSeconds,
    open: Open,
    push: Push,
    finish: Finish,
    close: Destroy,
    frame_count: FrameCount,
    probs_start: FrameCount,
    probabilities: Probabilities,
    segments: Segments,
    last_error: LastError,
}

impl Api {
    fn load(path: &Path) -> Result<Self> {
        // The library is the reviewed architecture packaged beside PIE, or an
        // explicit developer override. It stays alive longer than every handle.
        let library = unsafe { Library::new(path) }.map_err(|e| {
            Error::Other(format!("Load native diarization {}: {e}", path.display()))
        })?;
        unsafe {
            macro_rules! symbol {
                ($name:literal, $ty:ty) => {
                    *library
                        .get::<$ty>(concat!($name, "\0").as_bytes())
                        .map_err(|e| {
                            Error::Other(format!("Native diarization ABI {}: {e}", $name))
                        })?
                };
            }
            Ok(Self {
                create: symbol!("nemo_speech_diar_create", Create),
                destroy: symbol!("nemo_speech_diar_destroy", Destroy),
                speakers: symbol!("nemo_speech_diar_num_speakers", Speakers),
                frame_seconds: symbol!("nemo_speech_diar_seconds_per_frame", FrameSeconds),
                open: symbol!("nemo_speech_diar_stream_open", Open),
                push: symbol!("nemo_speech_diar_stream_push_f32", Push),
                finish: symbol!("nemo_speech_diar_stream_finish", Finish),
                close: symbol!("nemo_speech_diar_stream_close", Destroy),
                frame_count: symbol!("nemo_speech_diar_frame_count", FrameCount),
                probs_start: symbol!("nemo_speech_diar_frame_probs_start", FrameCount),
                probabilities: symbol!("nemo_speech_diar_frame_probs", Probabilities),
                segments: symbol!("nemo_speech_diar_segments", Segments),
                last_error: symbol!("nemo_speech_asr_last_error", LastError),
                _library: library,
            })
        }
    }

    fn check(&self, status: i32) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let message = unsafe {
            let pointer = (self.last_error)();
            if pointer.is_null() {
                format!("status {status}")
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        Err(Error::Other(format!("Nemotron 3 diarization: {message}")))
    }
}

pub(crate) struct NativeModel {
    api: Arc<Api>,
    pointer: *mut c_void,
    pub model_path: PathBuf,
    pub weight_bytes: u64,
    active: AtomicUsize,
}

// NVIDIA's C contract permits independent streams on different threads and
// serializes the shared compute backend internally. Handles are never freed
// until the last stream's Arc is released.
unsafe impl Send for NativeModel {}
unsafe impl Sync for NativeModel {}

impl NativeModel {
    pub fn load(
        library: PathBuf,
        model_path: PathBuf,
        options: &DiarizationOptions,
    ) -> Result<Self> {
        let weight_bytes = std::fs::metadata(&model_path)?.len();
        let api = Arc::new(Api::load(&library)?);
        let path = CString::new(model_path.to_string_lossy().as_bytes())
            .map_err(|_| Error::Other("NUL byte in diarization model path".into()))?;
        let preset = CString::new("v3-streaming").expect("constant");
        let config = ModelConfig {
            size: std::mem::size_of::<ModelConfig>(),
            model_path: path.as_ptr(),
            gpu: match options.device {
                DiarizationDevice::Cpu => -1,
                DiarizationDevice::Metal => 0,
            },
            preset: preset.as_ptr(),
            chunk_frames: options.chunk_frames,
            right_context_frames: options.right_context_frames,
            left_context_frames: 0,
            fifo_frames: options.fifo_frames,
            spkcache_frames: options.speaker_cache_frames,
            update_period_frames: options.update_period_frames,
        };
        let mut pointer = ptr::null_mut();
        api.check(unsafe { (api.create)(&config, &mut pointer) })?;
        if pointer.is_null() {
            return Err(Error::Other(
                "Native diarization returned a null model".into(),
            ));
        }
        let model = Self {
            api,
            pointer,
            weight_bytes,
            model_path,
            active: AtomicUsize::new(0),
        };
        let speakers = unsafe { (model.api.speakers)(pointer) };
        let cadence = unsafe { (model.api.frame_seconds)(pointer) };
        if speakers != 8 || (cadence - 0.01).abs() > 1e-8 {
            return Err(Error::Other(format!("Expected Nemotron 3's 8 speakers/10ms outputs; model reports {speakers} speakers/{cadence}s")));
        }
        Ok(model)
    }

    pub fn is_busy(&self) -> bool {
        self.active.load(Ordering::Acquire) != 0
    }
}

impl Drop for NativeModel {
    fn drop(&mut self) {
        unsafe { (self.api.destroy)(self.pointer) };
    }
}

pub(crate) struct NativeStream {
    owner: Arc<NativeModel>,
    pointer: *mut c_void,
}
// A stream is used only by its one Orchard worker, never concurrently.
unsafe impl Send for NativeStream {}

pub(crate) struct Snapshot {
    pub first_frame: u64,
    pub frame_count: u64,
    pub probabilities: Vec<[f32; 8]>,
    pub segments: Vec<Segment>,
}

impl NativeStream {
    pub fn open(owner: Arc<NativeModel>) -> Result<Self> {
        let mut pointer = ptr::null_mut();
        owner
            .api
            .check(unsafe { (owner.api.open)(owner.pointer, &mut pointer) })?;
        if pointer.is_null() {
            return Err(Error::Other(
                "Native diarization returned a null stream".into(),
            ));
        }
        owner.active.fetch_add(1, Ordering::AcqRel);
        Ok(Self { owner, pointer })
    }
    pub fn push(&mut self, pcm: &[f32], sample_rate: u32) -> Result<()> {
        self.owner.api.check(unsafe {
            (self.owner.api.push)(self.pointer, pcm.as_ptr(), pcm.len(), sample_rate as i32)
        })
    }
    pub fn finish(&mut self) -> Result<()> {
        self.owner
            .api
            .check(unsafe { (self.owner.api.finish)(self.pointer) })
    }
    pub fn frame_count(&self) -> Result<u64> {
        let count = unsafe { (self.owner.api.frame_count)(self.pointer) };
        if count < 0 {
            return Err(Error::Other("Invalid native frame count".into()));
        }
        Ok(count as u64)
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        let count = unsafe { (self.owner.api.frame_count)(self.pointer) };
        let start = unsafe { (self.owner.api.probs_start)(self.pointer) };
        if count < 0 || start < 0 || start > count || count - start > 240_000 {
            return Err(Error::Other(
                "Invalid native diarization probability span".into(),
            ));
        }
        let mut probabilities = vec![[0.0_f32; 8]; (count - start) as usize];
        if !probabilities.is_empty() {
            self.owner.api.check(unsafe {
                (self.owner.api.probabilities)(
                    self.pointer,
                    probabilities.as_mut_ptr().cast(),
                    probabilities.len() * 8,
                )
            })?;
        }
        if probabilities
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(Error::Other(
                "Nemotron emitted an invalid speaker probability".into(),
            ));
        }
        let mut segment_count = 0;
        self.owner.api.check(unsafe {
            (self.owner.api.segments)(
                self.pointer,
                ptr::null(),
                ptr::null_mut(),
                0,
                &mut segment_count,
            )
        })?;
        if segment_count > 1_000_000 {
            return Err(Error::Other(
                "Native diarization segment count is invalid".into(),
            ));
        }
        let mut segments = vec![Segment::default(); segment_count];
        if !segments.is_empty() {
            self.owner.api.check(unsafe {
                (self.owner.api.segments)(
                    self.pointer,
                    ptr::null(),
                    segments.as_mut_ptr(),
                    segments.len(),
                    &mut segment_count,
                )
            })?;
            segments.truncate(segment_count);
        }
        Ok(Snapshot {
            first_frame: start as u64,
            frame_count: count as u64,
            probabilities,
            segments,
        })
    }
}

impl Drop for NativeStream {
    fn drop(&mut self) {
        unsafe { (self.owner.api.close)(self.pointer) };
        self.owner.active.fetch_sub(1, Ordering::AcqRel);
    }
}
