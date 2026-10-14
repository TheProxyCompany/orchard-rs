//! High-performance IPC client for communicating with PIE.
//!
//! Uses NNG sockets with a dedicated listener thread for response handling.

use crate::engine::lifecycle::{current_engine_pid_file, EnginePaths};
use crate::engine::multiprocess::{pid_is_alive, read_pid_file};
use crate::error::{Error, Result};
use crate::ipc::endpoints::{management_url, request_url, response_url, EVENT_TOPIC_PREFIX};
use crate::ipc::serialization::{build_batch_request_payload, PromptPayload, RequestType};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use nng::options::Options;
use nng::{Protocol, Socket};
use serde::de::Error as DeError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Callback type for engine events (telemetry, model_loaded, etc.)
pub type EventCallback = Arc<dyn Fn(&str, &Value) + Send + Sync>;

const ENGINE_LIVENESS_POLL_INTERVAL: Duration = Duration::from_secs(10);
const RESPONSE_RECV_TIMEOUT: Duration = Duration::from_millis(10);
const RESPONSE_SOCKET_BUFFER_MESSAGES: i32 = 1024;
/// Cap on how long a PUSH send may block. With no live peer (engine dead
/// before the liveness poll notices), nng blocks the send forever otherwise.
const REQUEST_SEND_TIMEOUT: Duration = Duration::from_secs(30);
/// Recv slice for management replies; between slices we poll engine liveness
/// so a reply that will never come fails loudly instead of blocking.
const MANAGEMENT_LIVENESS_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// A single token's log probability info from PIE.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TokenLogProb {
    /// The token string (or token ID as string)
    pub token: String,
    /// The log probability value. PIE's wire format names this field
    /// `probability` (transport.cpp writes `{"token":...,"probability":...}`);
    /// the alias maps it here so the value survives deserialization.
    #[serde(alias = "probability")]
    pub logprob: f64,
    /// Optional bytes representation
    pub bytes: Option<Vec<u8>>,
}

/// A single state transition event emitted by PIE/PSE for structured outputs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ResponseStateEvent {
    /// Event type (e.g., item_started, content_delta, item_completed)
    pub event_type: String,
    /// Item type (e.g., message, tool_call, reasoning)
    pub item_type: String,
    /// Output index for this item in the response output array
    pub output_index: u32,
    /// Identifier for sub-items or tool names
    pub identifier: String,
    /// Delta text for streaming content updates
    pub delta: String,
    /// Optional final value for completion events
    pub value: Option<Value>,
}

fn deserialize_optional_bytes<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<Vec<u8>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BytePayload {
        Bytes(Vec<u8>),
        Base64(String),
    }

    match Option::<BytePayload>::deserialize(deserializer)? {
        None => Ok(None),
        Some(BytePayload::Bytes(bytes)) => Ok(Some(bytes)),
        Some(BytePayload::Base64(encoded)) => {
            BASE64.decode(encoded).map(Some).map_err(D::Error::custom)
        }
    }
}

/// Response delta from PIE.
///
/// Uses serde for deserialization with sensible defaults for missing fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ResponseDelta {
    /// Request ID this delta belongs to
    pub request_id: u64,
    /// Sequence ID for ordering
    pub sequence_id: Option<u64>,
    /// Prompt index for batched requests (identifies which prompt in the batch)
    pub prompt_index: Option<u32>,
    /// Candidate index (for multi-candidate generation)
    pub candidate_index: Option<u32>,
    /// Generated content (token text)
    pub content: Option<String>,
    /// Content length in characters
    pub content_len: Option<u32>,
    /// Inline content bytes
    pub inline_content_bytes: Option<u32>,
    /// Whether this is the final delta
    pub is_final_delta: bool,
    /// Finish reason (e.g., "stop", "length")
    pub finish_reason: Option<String>,
    /// Matched stop/EOS token id that ended generation, present only on the
    /// final delta when a sampled eos token triggered completion.
    pub matched_stop_token_id: Option<i32>,
    /// Decoded text of that stop token (e.g. "<|eom_id|>"), decoded engine-side.
    #[serde(default)]
    pub matched_stop_token: Option<String>,
    /// Error message if request failed
    #[serde(alias = "error_message")]
    pub error: Option<String>,
    /// Prompt token count
    pub prompt_token_count: Option<u32>,
    /// Number of tokens in this delta
    pub num_tokens_in_delta: Option<u32>,
    /// Generation length so far
    pub generation_len: Option<u32>,
    /// Token IDs in this delta
    pub tokens: Vec<i32>,
    /// Top log probabilities for each token position
    pub top_logprobs: Vec<TokenLogProb>,
    /// Cumulative log probability
    pub cumulative_logprob: Option<f64>,
    /// Modal decoder identifier (e.g., "moondream3.coord")
    pub modal_decoder_id: Option<String>,
    /// Modal artifact type, such as "audio" or "image".
    pub modal_type: Option<String>,
    /// Modal artifact lifecycle event, such as "artifact.delta" or "artifact.done".
    pub modal_event: Option<String>,
    /// MIME type for modal artifact bytes.
    pub modal_mime_type: Option<String>,
    /// JSON-encoded modal artifact metadata.
    pub modal_metadata_json: Option<String>,
    /// Base64-encoded modal decoder output bytes
    pub modal_bytes_b64: Option<String>,
    /// Raw embedding bytes from PIE, when request_type is embedding.
    #[serde(default, deserialize_with = "deserialize_optional_bytes")]
    pub embedding_bytes: Option<Vec<u8>>,
    /// Structured state transition events used by Responses API.
    pub state_events: Vec<ResponseStateEvent>,
    /// Cached token count (input token cache hits).
    pub cached_token_count: Option<u32>,
    /// Reasoning token count, when available.
    pub reasoning_tokens: Option<u32>,
}

/// High-performance IPC client for communicating with PIE.
///
/// Uses a lock-based design instead of actors to minimize overhead in the hot path.
/// All socket operations are thread-safe via internal locks.
pub struct IPCClient {
    request_socket: Option<Socket>,
    response_socket: Option<Socket>,
    /// Management socket wrapped in Arc<Mutex> for async access via spawn_blocking
    management_socket: Arc<Mutex<Option<Socket>>>,
    response_channel_id: u64,
    request_id_counter: AtomicU64,
    active_requests: Arc<Mutex<HashMap<u64, ActiveRequest>>>,
    listener_handle: Option<JoinHandle<()>>,
    should_stop: Arc<AtomicBool>,
    /// Set by the listener when the engine process dies or the response
    /// socket is gone for good; send paths fail fast instead of hanging.
    engine_dead: Arc<AtomicBool>,
    engine_pid_file: Option<PathBuf>,
    event_callback: Option<EventCallback>,
}

struct ActiveRequest {
    sender: mpsc::UnboundedSender<ResponseDelta>,
    remaining_finals: usize,
}

impl IPCClient {
    /// Create a new IPC client (not connected).
    pub fn new() -> Self {
        Self {
            request_socket: None,
            response_socket: None,
            management_socket: Arc::new(Mutex::new(None)),
            response_channel_id: rand_u64(),
            request_id_counter: AtomicU64::new(0),
            active_requests: Arc::new(Mutex::new(HashMap::new())),
            listener_handle: None,
            should_stop: Arc::new(AtomicBool::new(false)),
            engine_dead: Arc::new(AtomicBool::new(false)),
            engine_pid_file: None,
            event_callback: None,
        }
    }

    /// Create a new IPC client with an event callback.
    pub fn with_event_callback(callback: EventCallback) -> Self {
        let mut client = Self::new();
        client.event_callback = Some(callback);
        client
    }

    /// Get a clone of the management socket Arc for async operations.
    pub fn management_socket(&self) -> Arc<Mutex<Option<Socket>>> {
        Arc::clone(&self.management_socket)
    }

    pub(crate) fn response_channel_id(&self) -> u64 {
        self.response_channel_id
    }

    /// Connect to PIE IPC endpoints.
    pub fn connect(&mut self) -> Result<()> {
        let engine_pid_file = current_engine_pid_file()
            .or_else(|| EnginePaths::new().ok().map(|paths| paths.pid_file))
            .ok_or_else(|| Error::Internal("Cannot determine engine PID file path".into()))?;

        // Create and connect request socket (PUSH)
        let request_socket = Socket::new(Protocol::Push0)?;
        request_socket.set_opt::<nng::options::SendTimeout>(Some(REQUEST_SEND_TIMEOUT))?;
        request_socket.dial(&request_url())?;
        self.request_socket = Some(request_socket);

        // Create response socket (SUB) - subscribe BEFORE dial
        let response_socket = Socket::new(Protocol::Sub0)?;
        response_socket.set_opt::<nng::options::RecvBufferSize>(RESPONSE_SOCKET_BUFFER_MESSAGES)?;

        // Subscribe to our response topic
        let response_topic = format!("resp:{:x}:", self.response_channel_id);
        response_socket.set_opt::<nng::options::protocol::pubsub::Subscribe>(
            response_topic.as_bytes().to_vec(),
        )?;

        // Subscribe to global events
        response_socket
            .set_opt::<nng::options::protocol::pubsub::Subscribe>(EVENT_TOPIC_PREFIX.to_vec())?;

        response_socket.dial(&response_url())?;
        self.response_socket = Some(response_socket);

        // Create management socket (REQ)
        let management_socket = Socket::new(Protocol::Req0)?;
        management_socket.dial(&management_url())?;
        {
            let mut mgmt = self
                .management_socket
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *mgmt = Some(management_socket);
        }

        // Start listener thread
        self.should_stop.store(false, Ordering::SeqCst);
        self.engine_dead.store(false, Ordering::SeqCst);
        self.engine_pid_file = Some(engine_pid_file.clone());
        self.start_listener(engine_pid_file);

        Ok(())
    }

    /// Disconnect from PIE with graceful shutdown.
    ///
    /// Sends error deltas to all pending requests before closing.
    pub fn disconnect(&mut self) {
        self.should_stop.store(true, Ordering::SeqCst);

        // Send error deltas to all pending requests (graceful shutdown)
        {
            let requests = self
                .active_requests
                .lock()
                .unwrap_or_else(|e| e.into_inner());

            for (request_id, entry) in requests.iter() {
                let error_delta = ResponseDelta {
                    request_id: *request_id,
                    is_final_delta: true,
                    finish_reason: Some("error".to_string()),
                    content: Some("Engine process disconnected.".to_string()),
                    error: Some("Engine process disconnected.".to_string()),
                    ..Default::default()
                };
                let _ = entry.sender.send(error_delta);
            }
        }

        if let Some(handle) = self.listener_handle.take() {
            let _ = handle.join();
        }

        self.request_socket = None;
        self.response_socket = None;
        {
            let mut mgmt = self
                .management_socket
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *mgmt = None;
        }

        if let Ok(mut requests) = self.active_requests.lock() {
            requests.clear();
        }
    }

    /// Get the next request ID.
    pub fn next_request_id(&self) -> u64 {
        let id = self.request_id_counter.fetch_add(1, Ordering::SeqCst);
        if id >= u64::MAX - 1 {
            self.request_id_counter.store(1, Ordering::SeqCst);
        }
        id + 1
    }

    /// Send a batched request with multiple prompts in ONE IPC message.
    pub fn send_batch_request(
        &self,
        request_id: u64,
        model_id: &str,
        model_path: &str,
        prompts: &[PromptPayload],
    ) -> Result<(usize, mpsc::UnboundedReceiver<ResponseDelta>)> {
        self.send_batch_request_with_type(
            request_id,
            model_id,
            model_path,
            RequestType::Generation,
            prompts,
        )
    }

    pub(crate) fn send_batch_request_with_type(
        &self,
        request_id: u64,
        model_id: &str,
        model_path: &str,
        request_type: RequestType,
        prompts: &[PromptPayload],
    ) -> Result<(usize, mpsc::UnboundedReceiver<ResponseDelta>)> {
        let socket = self.request_socket.as_ref().ok_or(Error::NotConnected)?;
        tracing::debug!(
            request_id,
            model_id = %model_id,
            ?request_type,
            prompt_count = prompts.len(),
            "Serializing and sending IPC batch request"
        );

        let payload = build_batch_request_payload(
            request_id,
            model_id,
            model_path,
            request_type,
            self.response_channel_id,
            prompts,
        )?;
        tracing::debug!(
            request_id,
            model_id = %model_id,
            ?request_type,
            payload_bytes = payload.len(),
            "Built IPC batch payload"
        );

        let (tx, rx) = mpsc::unbounded_channel();
        let remaining_finals = prompts
            .iter()
            .map(|prompt| {
                let num_candidates = prompt.num_candidates.max(1);
                let best_of = prompt.best_of.unwrap_or(num_candidates).max(1);
                let final_candidates = prompt.final_candidates.unwrap_or(best_of).max(1);
                final_candidates as usize
            })
            .sum::<usize>()
            .max(1);

        {
            // Checked under the same lock the listener holds while draining on
            // engine death, so a request can never register after the drain and
            // hang with nobody left to fail it.
            let mut requests = self
                .active_requests
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if self.engine_dead.load(Ordering::SeqCst) {
                return Err(Error::EngineDead);
            }
            requests.insert(
                request_id,
                ActiveRequest {
                    sender: tx,
                    remaining_finals,
                },
            );
        }

        let msg = nng::Message::from(payload.as_slice());
        if let Err((_, error)) = socket.send(msg) {
            self.active_requests
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&request_id);
            if self.engine_dead.load(Ordering::SeqCst)
                || !self
                    .engine_pid_file
                    .as_deref()
                    .map(engine_process_is_alive)
                    .unwrap_or(true)
            {
                return Err(Error::EngineDead);
            }
            return Err(Error::Nng(error));
        }
        tracing::debug!(
            request_id,
            model_id = %model_id,
            ?request_type,
            expected_final_count = remaining_finals,
            "IPC batch request sent"
        );

        Ok((prompts.len(), rx))
    }

    /// Send a management command asynchronously.
    ///
    /// Uses spawn_blocking internally since NNG sockets are sync.
    pub async fn send_management_command_async(
        &self,
        command: Value,
        timeout: Duration,
    ) -> Result<Value> {
        let socket_arc = Arc::clone(&self.management_socket);
        let engine_dead = Arc::clone(&self.engine_dead);
        let engine_pid_file = self.engine_pid_file.clone();

        tokio::task::spawn_blocking(move || {
            blocking_management_exchange(
                &socket_arc,
                &engine_dead,
                engine_pid_file.as_deref(),
                &command,
                timeout,
            )
        })
        .await
        .map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }

    #[cfg(any(feature = "duplex", feature = "diarization"))]
    pub(crate) fn send_management_command_blocking(
        &self,
        command: Value,
        timeout: Duration,
    ) -> Result<Value> {
        let engine_pid_file = self.engine_pid_file.as_deref().ok_or(Error::NotConnected)?;
        // Continuous audio and priority controls must not wait behind the
        // generic management mutex (model activation may hold it for seconds).
        // Each exchange owns its REQ socket, including its request/reply state,
        // so a timed-out input cannot strand or cancel a later close/interrupt.
        let deadline = Instant::now() + timeout;
        let socket = Socket::new(Protocol::Req0)?;
        socket.set_opt::<nng::options::RecvMaxSize>(1024 * 1024)?;
        socket.dial_async(&management_url())?;
        let socket = Mutex::new(Some(socket));
        blocking_management_exchange(
            &socket,
            &self.engine_dead,
            Some(engine_pid_file),
            &command,
            deadline.saturating_duration_since(Instant::now()),
        )
    }

    /// Bind before open. Dedicated PULL flow control protects continuous audio
    /// from PUB/SUB drops and avoids the generic unbounded text receiver.
    #[cfg(feature = "duplex")]
    pub(crate) fn bind_duplex_route(
        &self,
        request_id: u64,
        handler: Arc<dyn Fn(ResponseDelta, bool) + Send + Sync>,
    ) -> Result<DuplexResponseRoute> {
        self.bind_stream_route(request_id, "duplex", 1024 * 1024, handler)
    }

    #[cfg(feature = "diarization")]
    pub(crate) fn bind_diarization_route(
        &self,
        request_id: u64,
        handler: Arc<dyn Fn(ResponseDelta, bool) + Send + Sync>,
    ) -> Result<StreamResponseRoute> {
        // Metadata is bounded at 16 MiB by PIE. Its JSON-string encoding in
        // ResponseDelta can double in size before decoding that metadata.
        self.bind_stream_route(request_id, "diarization", 33 * 1024 * 1024, handler)
    }

    #[cfg(any(feature = "duplex", feature = "diarization"))]
    fn bind_stream_route(
        &self,
        request_id: u64,
        kind: &'static str,
        max_bytes: usize,
        handler: Arc<dyn Fn(ResponseDelta, bool) + Send + Sync>,
    ) -> Result<StreamResponseRoute> {
        if self.engine_dead.load(Ordering::Acquire) {
            return Err(Error::EngineDead);
        }
        let pid_file = self.engine_pid_file.clone().ok_or(Error::NotConnected)?;
        let channel_id = rand_u64();
        let url = crate::ipc::endpoints::pull_response_url(channel_id);
        let socket = Socket::new(Protocol::Pull0)?;
        socket.set_opt::<nng::options::RecvTimeout>(Some(Duration::from_millis(50)))?;
        socket.set_opt::<nng::options::RecvBufferSize>(64)?;
        socket.set_opt::<nng::options::RecvMaxSize>(max_bytes)?;
        socket.listen(&url)?;
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let engine_dead = Arc::clone(&self.engine_dead);
        let original_pid = read_pid_file(&pid_file);
        let handle = thread::Builder::new()
            .name(format!("orchard-{kind}-response"))
            .spawn(move || {
                let prefix = format!("resp:{channel_id:x}:");
                while !worker_stop.load(Ordering::Acquire) {
                    match socket.recv() {
                        Ok(message) => {
                            let bytes = message.as_slice();
                            let Some(body) = bytes.strip_prefix(prefix.as_bytes()) else {
                                continue;
                            };
                            match serde_json::from_slice::<ResponseDelta>(body) {
                                Ok(delta) => {
                                    let finished = delta.is_final_delta;
                                    handler(delta, true);
                                    if finished {
                                        break;
                                    }
                                }
                                Err(error) => {
                                    handler(
                                        ResponseDelta {
                                            request_id,
                                            is_final_delta: true,
                                            error: Some(format!(
                                                "Invalid PIE {kind} response: {error}"
                                            )),
                                            ..Default::default()
                                        },
                                        false,
                                    );
                                    break;
                                }
                            }
                        }
                        Err(nng::Error::TimedOut) => {}
                        Err(error) => {
                            if !worker_stop.load(Ordering::Acquire) {
                                handler(
                                    ResponseDelta {
                                        request_id,
                                        is_final_delta: true,
                                        error: Some(format!(
                                            "PIE {kind} response route closed: {error}"
                                        )),
                                        ..Default::default()
                                    },
                                    false,
                                );
                            }
                            break;
                        }
                    }
                    if engine_dead.load(Ordering::Acquire)
                        || read_pid_file(&pid_file) != original_pid
                        || !engine_process_is_alive(&pid_file)
                    {
                        handler(
                            ResponseDelta {
                                request_id,
                                is_final_delta: true,
                                error: Some(format!("PIE {kind} engine exited or changed")),
                                ..Default::default()
                            },
                            false,
                        );
                        break;
                    }
                }
            })
            .map_err(Error::Io)?;
        Ok(StreamResponseRoute {
            channel_id,
            stop,
            handle: Some(handle),
        })
    }

    #[cfg(feature = "duplex")]
    pub(crate) fn send_duplex_request(
        &self,
        request_id: u64,
        model_id: &str,
        model_path: &str,
        channel_id: u64,
        prompt: PromptPayload,
    ) -> Result<()> {
        self.send_stream_request(
            request_id,
            model_id,
            model_path,
            channel_id,
            RequestType::Duplex,
            prompt,
        )
    }

    #[cfg(feature = "diarization")]
    pub(crate) fn send_diarization_request(
        &self,
        request_id: u64,
        model_id: &str,
        model_path: &str,
        channel_id: u64,
        prompt: PromptPayload,
    ) -> Result<()> {
        self.send_stream_request(
            request_id,
            model_id,
            model_path,
            channel_id,
            RequestType::Diarization,
            prompt,
        )
    }

    #[cfg(any(feature = "duplex", feature = "diarization"))]
    fn send_stream_request(
        &self,
        request_id: u64,
        model_id: &str,
        model_path: &str,
        channel_id: u64,
        request_type: RequestType,
        prompt: PromptPayload,
    ) -> Result<()> {
        if self.engine_dead.load(Ordering::Acquire) {
            return Err(Error::EngineDead);
        }
        let socket = self.request_socket.as_ref().ok_or(Error::NotConnected)?;
        let payload = crate::ipc::serialization::build_batch_request_payload_with_transport(
            request_id,
            model_id,
            model_path,
            request_type,
            channel_id,
            &[prompt],
            Some("pull_v1"),
        )?;
        socket
            .send(nng::Message::from(payload.as_slice()))
            .map_err(|(_, error)| Error::from(error))
    }

    /// Start the response listener thread.
    fn start_listener(&mut self, engine_pid_file: PathBuf) {
        let response_socket = self.response_socket.take();
        let active_requests = Arc::clone(&self.active_requests);
        let should_stop = Arc::clone(&self.should_stop);
        let engine_dead = Arc::clone(&self.engine_dead);
        let response_channel_id = self.response_channel_id;
        let event_callback = self.event_callback.clone();

        let handle = thread::Builder::new()
            .name("orchard-ipc-listener".to_string())
            .spawn(move || {
                if let Some(socket) = response_socket {
                    run_response_listener(
                        socket,
                        active_requests,
                        should_stop,
                        engine_dead,
                        response_channel_id,
                        engine_pid_file,
                        event_callback,
                    );
                }
            });

        match handle {
            Ok(h) => self.listener_handle = Some(h),
            Err(e) => tracing::error!("Failed to spawn IPC listener thread: {}", e),
        }
    }
}

#[cfg(feature = "duplex")]
pub(crate) type DuplexResponseRoute = StreamResponseRoute;
#[cfg(any(feature = "duplex", feature = "diarization"))]
pub(crate) struct StreamResponseRoute {
    pub channel_id: u64,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}
#[cfg(any(feature = "duplex", feature = "diarization"))]
impl Drop for StreamResponseRoute {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Default for IPCClient {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for IPCClient {
    fn drop(&mut self) {
        self.disconnect();
    }
}

fn engine_process_is_alive(engine_pid_file: &Path) -> bool {
    read_pid_file(engine_pid_file)
        .map(pid_is_alive)
        .unwrap_or(false)
}

/// One request/reply exchange on the management REQ socket.
///
/// A bare recv blocks forever when the engine exits between send and reply —
/// and it holds the management socket lock, wedging every queued command
/// behind it. Receive in short slices and poll engine liveness between them,
/// failing loudly the moment the engine is gone.
fn blocking_management_exchange(
    socket_arc: &Mutex<Option<Socket>>,
    engine_dead: &AtomicBool,
    engine_pid_file: Option<&Path>,
    command: &Value,
    timeout: Duration,
) -> Result<Value> {
    let guard = socket_arc.lock().unwrap_or_else(|e| e.into_inner());
    let socket = guard.as_ref().ok_or(Error::NotConnected)?;

    let engine_is_dead = || {
        engine_dead.load(Ordering::SeqCst)
            || !engine_pid_file.map(engine_process_is_alive).unwrap_or(true)
    };
    if engine_is_dead() {
        return Err(Error::EngineDead);
    }

    // REQ send also blocks forever when the peer is gone (verified: it parks
    // in nng_aio_wait), so both directions run in timed slices.
    socket.set_opt::<nng::options::SendTimeout>(Some(MANAGEMENT_LIVENESS_POLL_INTERVAL))?;
    socket.set_opt::<nng::options::RecvTimeout>(Some(MANAGEMENT_LIVENESS_POLL_INTERVAL))?;
    let deadline = Instant::now() + timeout;

    let data = serde_json::to_vec(command)?;
    let mut msg = nng::Message::from(data.as_slice());
    loop {
        match socket.send(msg) {
            Ok(()) => break,
            Err((returned, nng::Error::TimedOut)) => {
                if engine_is_dead() {
                    return Err(Error::EngineDead);
                }
                if Instant::now() >= deadline {
                    return Err(Error::Nng(nng::Error::TimedOut));
                }
                msg = returned;
            }
            Err((_, error)) => return Err(Error::Nng(error)),
        }
    }

    loop {
        match socket.recv() {
            Ok(response) => return Ok(serde_json::from_slice(&response)?),
            Err(nng::Error::TimedOut) => {
                if engine_is_dead() {
                    return Err(Error::EngineDead);
                }
                if Instant::now() >= deadline {
                    return Err(Error::Nng(nng::Error::TimedOut));
                }
            }
            Err(error) => return Err(Error::Nng(error)),
        }
    }
}

/// Response listener - runs on dedicated thread for minimal latency.
fn run_response_listener(
    socket: Socket,
    active_requests: Arc<Mutex<HashMap<u64, ActiveRequest>>>,
    should_stop: Arc<AtomicBool>,
    engine_dead: Arc<AtomicBool>,
    response_channel_id: u64,
    engine_pid_file: PathBuf,
    event_callback: Option<EventCallback>,
) {
    let response_topic = format!("resp:{:x}:", response_channel_id);
    let response_topic_bytes = response_topic.as_bytes();

    // Set receive timeout for responsive polling (10ms for better latency)
    let _ = socket.set_opt::<nng::options::RecvTimeout>(Some(RESPONSE_RECV_TIMEOUT));
    let mut last_engine_check = Instant::now();
    let mut death_reason: Option<String> = None;

    while !should_stop.load(Ordering::SeqCst) {
        match socket.recv() {
            Ok(msg) => {
                let data = msg.as_slice();

                // Check if it's a response for us
                if data.starts_with(response_topic_bytes) {
                    let json_data = &data[response_topic_bytes.len()..];

                    if let Ok(delta) = serde_json::from_slice::<ResponseDelta>(json_data) {
                        let request_id = delta.request_id;
                        let is_final = delta.is_final_delta;

                        let sender = {
                            let mut requests =
                                active_requests.lock().unwrap_or_else(|e| e.into_inner());
                            if let Some(entry) = requests.get_mut(&request_id) {
                                if is_final {
                                    entry.remaining_finals =
                                        entry.remaining_finals.saturating_sub(1);
                                    if entry.remaining_finals == 0 {
                                        let sender = entry.sender.clone();
                                        requests.remove(&request_id);
                                        Some(sender)
                                    } else {
                                        Some(entry.sender.clone())
                                    }
                                } else {
                                    Some(entry.sender.clone())
                                }
                            } else {
                                None
                            }
                        };

                        if let Some(tx) = sender {
                            let _ = tx.send(delta);
                        }
                    } else {
                        tracing::warn!(
                            response_channel_id,
                            payload_bytes = json_data.len(),
                            "Failed to deserialize IPC response payload"
                        );
                    }
                }
                // Check if it's an engine event
                else if data.starts_with(EVENT_TOPIC_PREFIX) {
                    handle_engine_event(data, &event_callback);
                }
            }
            Err(nng::Error::TimedOut) => {
                if last_engine_check.elapsed() >= ENGINE_LIVENESS_POLL_INTERVAL {
                    last_engine_check = Instant::now();
                    if !engine_process_is_alive(&engine_pid_file) {
                        tracing::error!(
                            pid_file = %engine_pid_file.display(),
                            "PIE is no longer alive; shutting down IPC listener"
                        );
                        death_reason = Some("Engine process died".to_string());
                        should_stop.store(true, Ordering::SeqCst);
                        break;
                    }
                }
                continue;
            }
            Err(error) => {
                if should_stop.load(Ordering::SeqCst) {
                    break;
                }
                if !engine_process_is_alive(&engine_pid_file) {
                    tracing::error!(
                        pid_file = %engine_pid_file.display(),
                        error = %error,
                        "PIE is no longer alive; shutting down IPC listener"
                    );
                    death_reason = Some(format!("Engine process died ({})", error));
                    should_stop.store(true, Ordering::SeqCst);
                    break;
                }
            }
        }
    }

    // Shutdown: fail every pending request so no caller waits on a reply that
    // will never come, and close the door on new registrations.
    tracing::info!("IPC listener shutting down");
    let reason = death_reason.unwrap_or_else(|| "Engine process disconnected.".to_string());
    {
        let mut requests = active_requests.lock().unwrap_or_else(|e| e.into_inner());
        // Under the same lock send_batch_request checks, so no request can
        // slip in after this drain.
        engine_dead.store(true, Ordering::SeqCst);

        if !requests.is_empty() {
            tracing::warn!(
                "IPC listener exiting with {} active requests; failing them.",
                requests.len()
            );

            for (request_id, entry) in requests.iter() {
                let error_delta = ResponseDelta {
                    request_id: *request_id,
                    is_final_delta: true,
                    finish_reason: Some("error".to_string()),
                    content: Some(reason.clone()),
                    error: Some(reason.clone()),
                    ..Default::default()
                };
                let _ = entry.sender.send(error_delta);
            }
            requests.clear();
        }
    }

    // Engine-side model activations can never complete now; let the registry
    // fail its waiters instead of leaving them parked forever.
    if let Some(callback) = &event_callback {
        callback("engine_died", &serde_json::json!({ "error": reason }));
    }
}

/// Handle an engine event (telemetry, model_loaded, etc.)
fn handle_engine_event(data: &[u8], event_callback: &Option<EventCallback>) {
    // Event format: __PIE_EVENT__:<event_name>\x00<json_body>
    let parts: Vec<&[u8]> = data.splitn(2, |&b| b == 0).collect();
    if parts.len() != 2 {
        tracing::warn!("Received malformed event message");
        return;
    }

    let (topic_part, json_body) = (parts[0], parts[1]);

    // Extract event name from topic: "__PIE_EVENT__:<event_name>"
    let event_name = if topic_part.len() > EVENT_TOPIC_PREFIX.len() {
        String::from_utf8_lossy(&topic_part[EVENT_TOPIC_PREFIX.len()..]).to_string()
    } else {
        tracing::warn!("Event message has empty event name");
        return;
    };

    // Parse JSON payload
    let payload: Value = match serde_json::from_slice(json_body) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("Failed to parse engine event payload: {}", e);
            return;
        }
    };

    if event_name != "telemetry" {
        tracing::debug!("Received engine event: {}", event_name);
    }

    // Dispatch to callback if registered
    if let Some(callback) = event_callback {
        callback(&event_name, &payload);
    }
}

/// Generate a unique response channel ID.
/// Format: (PID << 32) | random_32_bits
fn rand_u64() -> u64 {
    use rand::Rng;

    let pid = std::process::id() as u64 & 0xFFFFFFFF;
    let random: u32 = rand::thread_rng().gen();

    let channel_id = (pid << 32) | (random as u64);
    if channel_id == 0 {
        1
    } else {
        channel_id
    }
}
