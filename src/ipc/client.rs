//! High-performance IPC client for communicating with PIE.
//!
//! Uses NNG sockets with a dedicated listener thread for response handling.
//!
//! PUB/SUB drops the oldest queued messages, silently on both sides, once a
//! subscriber is about a thousand messages behind, which shortens a reply with
//! no error anywhere. So requests ask for the engine's flow-controlled route
//! instead (`"response_transport": "pull_v1"`): this client listens on its own
//! PULL endpoint, the engine dials it and PUSHes this client's deltas, and a
//! client that falls behind makes the engine wait instead of losing anything.
//!
//! They ask only where that is safe for the engine. An engine keeps a socket,
//! a thread and a queue per client on that route, and one that does not reap
//! them leaks all three, plus the undelivered deltas, for every client process
//! that goes away, until it stops. So a request asks for the route when the
//! engine advertises `lossless_responses` (it reaps stalled routes with an
//! explicit error and reports a response endpoint it cannot reach), or when
//! this process launched the engine itself, which bounds the leak to this one
//! client for the life of that engine. Otherwise requests go out as they
//! always did and are answered over PUB/SUB, which also still carries the
//! engine's broadcast events.

use crate::engine::lifecycle::{
    current_engine_pid_file, engine_launched_by_this_process, EnginePaths,
};
use crate::engine::multiprocess::{pid_is_alive, read_pid_file};
use crate::error::{Error, Result};
use crate::ipc::endpoints::{
    as_ipc_url, ipc_root, management_url_in, request_url_in, response_route_path, response_url_in,
    EVENT_TOPIC_PREFIX, RESPONSE_ROUTE_SOCKET_PREFIX, RESPONSE_ROUTE_SOCKET_SUFFIX,
};
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
/// Events are rare and `disconnect` wakes their reader by closing its socket,
/// so this only bounds how long the reader outlives a listener that stopped
/// by itself.
const EVENT_RECV_TIMEOUT: Duration = Duration::from_secs(1);
const EVENT_SOCKET_BUFFER_MESSAGES: i32 = 1024;
/// Cap on how long a PUSH send may block. With no live peer (engine dead
/// before the liveness poll notices), nng blocks the send forever otherwise.
const REQUEST_SEND_TIMEOUT: Duration = Duration::from_secs(30);
/// Recv slice for management replies; between slices we poll engine liveness
/// so a reply that will never come fails loudly instead of blocking.
const MANAGEMENT_LIVENESS_POLL_INTERVAL: Duration = Duration::from_millis(250);

// This client listens, and the NNG that nng-sys 1.4.0-rc.0 bundles has a
// listener that stops accepting for good once a peer connects and is gone
// before the handshake (upstream #1518), as an engine killed while dialling
// is: later deltas are then lost with no error on either side. The nng-sys
// vendored in this repository carries the fix and defines this constant.
//
// If this line does not compile, the workspace being built resolves nng-sys
// to a copy without the fix. Point its [patch.crates-io] nng-sys at this
// orchard-rs commit or a later one, in the same change that moves orchard-rs:
// grand-central pins both by rev (bump both to the same commit); Proxy/Glue
// patches from branch main and needs `cargo update -p nng-sys`.
const _: () = nng_sys::ORCHARD_VENDORED_WITH_LISTENER_FIX_1518;

/// The capability an engine lists in a load_model reply and in a
/// `model_loaded` event, with value 1, once it reaps a stalled response route
/// with an explicit error and reports a response endpoint it cannot reach.
pub(crate) const LOSSLESS_RESPONSES_CAPABILITY: &str = "lossless_responses";

/// Longest wait for a request's first delta before the request fails.
///
/// The engine sends nothing while a request is queued or prefilling, so this
/// has to hold the longest prefill a caller can legitimately ask for, not a
/// latency budget. It does not have to hold a model load: every client path
/// awaits `ModelRegistry::ensure_loaded` before it sends, the engine
/// publishes `model_loaded` only after the weights are materialized, and it
/// answers a request for a model it has not loaded with an error delta at
/// once. orchard-py caps the same wait at 300 s, but the app sends far longer
/// prompts to far larger models: 128k tokens at 100 tokens/s is 21 minutes of
/// prefill with no delta.
const DEFAULT_FIRST_DELTA_TIMEOUT: Duration = Duration::from_secs(1800);
/// Longest silence between two deltas of a request with one stream before it
/// fails; a request with several is held to the first-delta bound throughout.
///
/// Once a request decodes, a gap is a decode step behind other requests'
/// prefill chunks, or a preempted sequence waiting for cache pages. This is
/// orchard-py's ceiling on any single delta wait (`DELTA_HARD_TIMEOUT_S`).
const DEFAULT_DELTA_TIMEOUT: Duration = Duration::from_secs(300);
/// How often the listener looks for requests the engine went silent on. A
/// request therefore fails up to two intervals after its bound.
const DELTA_WATCHDOG_INTERVAL: Duration = Duration::from_secs(1);
/// While deltas arrive back to back the listener never sees a receive
/// timeout, so it also considers the watchdog once per this many deltas.
const DELTAS_PER_WATCHDOG_CHECK: u32 = 1024;

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
    /// PULL socket listening on this client's own endpoint; the engine dials
    /// it and pushes this client's response deltas.
    response_socket: Option<Socket>,
    /// SUB socket for the engine's broadcast events.
    event_socket: Option<Socket>,
    /// Management socket wrapped in Arc<Mutex> for async access via spawn_blocking
    management_socket: Arc<Mutex<Option<Socket>>>,
    pub(crate) response_channel_id: u64,
    request_id_counter: AtomicU64,
    active_requests: Arc<Mutex<HashMap<u64, ActiveRequest>>>,
    listener_handle: Option<JoinHandle<()>>,
    event_listener_handle: Option<JoinHandle<()>>,
    should_stop: Arc<AtomicBool>,
    /// Set by the listener when the engine process dies or the response
    /// socket is gone for good; send paths fail fast instead of hanging.
    engine_dead: Arc<AtomicBool>,
    engine_pid_file: Option<PathBuf>,
    event_callback: Option<EventCallback>,
    delta_timeouts: DeltaTimeouts,
    /// This process launched the engine it is connected to.
    own_engine: bool,
    /// The engine has advertised `lossless_responses`.
    engine_advertises_lossless: Arc<AtomicBool>,
}

/// How long a request may wait on a silent engine: for its first delta, and
/// between two deltas.
#[derive(Clone, Copy)]
struct DeltaTimeouts {
    first: Duration,
    between: Duration,
}

struct ActiveRequest {
    sender: mpsc::UnboundedSender<ResponseDelta>,
    remaining_finals: usize,
    /// The request expects more than one stream (prompts, candidates). One of
    /// them can be queued or prefilling, which is silent, after another has
    /// sent deltas, so only the first-delta bound holds for such a request.
    multi_stream: bool,
    /// Deltas routed to this request so far. The watchdog compares it with
    /// `watched_deltas` once per interval, so the receive loop pays one
    /// addition per delta and never reads the clock for it.
    deltas: u64,
    watched_deltas: u64,
    /// When the request was sent, then when the watchdog last saw new deltas.
    last_progress: Instant,
}

/// The delta that ends a request this client fails by itself.
fn terminal_error_delta(request_id: u64, reason: &str) -> ResponseDelta {
    ResponseDelta {
        request_id,
        is_final_delta: true,
        finish_reason: Some("error".to_string()),
        content: Some(reason.to_string()),
        error: Some(reason.to_string()),
        ..Default::default()
    }
}

impl IPCClient {
    /// Create a new IPC client (not connected).
    pub fn new() -> Self {
        Self {
            request_socket: None,
            response_socket: None,
            event_socket: None,
            management_socket: Arc::new(Mutex::new(None)),
            response_channel_id: rand_u64(),
            request_id_counter: AtomicU64::new(0),
            active_requests: Arc::new(Mutex::new(HashMap::new())),
            listener_handle: None,
            event_listener_handle: None,
            should_stop: Arc::new(AtomicBool::new(false)),
            engine_dead: Arc::new(AtomicBool::new(false)),
            engine_pid_file: None,
            event_callback: None,
            delta_timeouts: DeltaTimeouts {
                first: DEFAULT_FIRST_DELTA_TIMEOUT,
                between: DEFAULT_DELTA_TIMEOUT,
            },
            own_engine: false,
            engine_advertises_lossless: Arc::new(AtomicBool::new(false)),
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

    /// Set the event callback for handling engine events.
    pub fn set_event_callback(&mut self, callback: EventCallback) {
        self.event_callback = Some(callback);
    }

    /// Takes effect at the next `connect`.
    #[cfg(test)]
    fn set_delta_timeouts(&mut self, first_delta: Duration, between_deltas: Duration) {
        self.delta_timeouts = DeltaTimeouts {
            first: first_delta,
            between: between_deltas,
        };
    }

    /// The engine advertised `lossless_responses` to an earlier client of the
    /// registry this client now serves (`ModelRegistry::set_ipc_client`).
    pub(crate) fn note_lossless_responses(&self) {
        self.engine_advertises_lossless
            .store(true, Ordering::SeqCst);
    }

    /// Connect to PIE IPC endpoints.
    pub fn connect(&mut self) -> Result<()> {
        let engine_pid_file = current_engine_pid_file()
            .or_else(|| EnginePaths::new().ok().map(|paths| paths.pid_file))
            .ok_or_else(|| Error::Internal("Cannot determine engine PID file path".into()))?;
        let own_engine = engine_launched_by_this_process(&engine_pid_file);
        self.connect_in(&ipc_root(), engine_pid_file, own_engine)
    }

    fn connect_in(
        &mut self,
        ipc_root: &Path,
        engine_pid_file: PathBuf,
        own_engine: bool,
    ) -> Result<()> {
        if self.response_socket.is_some() {
            // Our endpoint is still bound: a second listen on it would fail.
            self.disconnect();
        }
        remove_stale_response_endpoints(ipc_root);
        self.own_engine = own_engine;
        // What an earlier engine advertised says nothing about this one.
        self.engine_advertises_lossless
            .store(false, Ordering::SeqCst);

        // Response socket (PULL). It listens whether or not requests will ask
        // for it, and before the request socket exists: what the engine
        // advertises is only known after the first load_model, and the engine
        // dials this endpoint on a request's first delta and drops the deltas
        // it produces while nothing listens there.
        let response_socket = Socket::new(Protocol::Pull0)?;
        response_socket.set_opt::<nng::options::RecvMaxSize>(0)?;
        response_socket.listen(&as_ipc_url(response_route_path(
            ipc_root,
            self.response_channel_id,
        )))?;

        // Create and connect request socket (PUSH)
        let request_socket = Socket::new(Protocol::Push0)?;
        request_socket.set_opt::<nng::options::SendTimeout>(Some(REQUEST_SEND_TIMEOUT))?;
        request_socket.dial(&request_url_in(ipc_root))?;

        // Create event socket (SUB) - subscribe BEFORE dial
        let event_socket = Socket::new(Protocol::Sub0)?;
        event_socket.set_opt::<nng::options::RecvBufferSize>(EVENT_SOCKET_BUFFER_MESSAGES)?;
        event_socket
            .set_opt::<nng::options::protocol::pubsub::Subscribe>(EVENT_TOPIC_PREFIX.to_vec())?;
        // A request that does not ask for the flow-controlled route is
        // answered here, on our response topic, and so is one that asked an
        // engine too old to know the route. Without this subscription those
        // requests would never be answered.
        event_socket.set_opt::<nng::options::protocol::pubsub::Subscribe>(
            format!("resp:{:x}:", self.response_channel_id).into_bytes(),
        )?;
        event_socket.dial(&response_url_in(ipc_root))?;

        // Create management socket (REQ)
        let management_socket = Socket::new(Protocol::Req0)?;
        management_socket.dial(&management_url_in(ipc_root))?;
        {
            let mut mgmt = self
                .management_socket
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *mgmt = Some(management_socket);
        }
        self.request_socket = Some(request_socket);
        self.response_socket = Some(response_socket);
        self.event_socket = Some(event_socket);

        // Start listener threads
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
                let _ = entry.sender.send(terminal_error_delta(
                    *request_id,
                    "Engine process disconnected.",
                ));
            }
        }

        // Closing wakes both readers now instead of at their next receive
        // timeout, and closing the listening socket removes our endpoint file.
        for socket in [&self.response_socket, &self.event_socket]
            .into_iter()
            .flatten()
        {
            socket.close();
        }
        for handle in [
            self.listener_handle.take(),
            self.event_listener_handle.take(),
        ]
        .into_iter()
        .flatten()
        {
            let _ = handle.join();
        }

        self.request_socket = None;
        self.response_socket = None;
        self.event_socket = None;
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
            self.own_engine || self.engine_advertises_lossless.load(Ordering::SeqCst),
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
                    multi_stream: remaining_finals > 1,
                    deltas: 0,
                    watched_deltas: 0,
                    last_progress: Instant::now(),
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
        let engine_advertises_lossless = Arc::clone(&self.engine_advertises_lossless);

        tokio::task::spawn_blocking(move || {
            blocking_management_exchange(
                &socket_arc,
                &engine_dead,
                engine_pid_file.as_deref(),
                &command,
                timeout,
                &engine_advertises_lossless,
            )
        })
        .await
        .map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }

    /// Send a management command synchronously (blocking).
    ///
    /// Prefer `send_management_command_async` in async contexts.
    pub fn send_management_command(&self, command: &Value, timeout: Duration) -> Result<Value> {
        blocking_management_exchange(
            &self.management_socket,
            &self.engine_dead,
            self.engine_pid_file.as_deref(),
            command,
            timeout,
            &self.engine_advertises_lossless,
        )
    }

    /// Start the response listener thread and the event listener thread.
    fn start_listener(&mut self, engine_pid_file: PathBuf) {
        let response_channel_id = self.response_channel_id;

        if let Some(socket) = self.response_socket.clone() {
            let active_requests = Arc::clone(&self.active_requests);
            let should_stop = Arc::clone(&self.should_stop);
            let engine_dead = Arc::clone(&self.engine_dead);
            let event_callback = self.event_callback.clone();
            let delta_timeouts = self.delta_timeouts;
            let management_socket = Arc::clone(&self.management_socket);
            let engine_advertises_lossless = Arc::clone(&self.engine_advertises_lossless);
            let handle = thread::Builder::new()
                .name("orchard-ipc-listener".to_string())
                .spawn(move || {
                    run_response_listener(
                        socket,
                        active_requests,
                        should_stop,
                        engine_dead,
                        response_channel_id,
                        engine_pid_file,
                        event_callback,
                        delta_timeouts,
                        management_socket,
                        engine_advertises_lossless,
                    );
                });
            match handle {
                Ok(h) => self.listener_handle = Some(h),
                Err(e) => tracing::error!("Failed to spawn IPC listener thread: {}", e),
            }
        }

        if let Some(socket) = self.event_socket.clone() {
            let active_requests = Arc::clone(&self.active_requests);
            let should_stop = Arc::clone(&self.should_stop);
            let event_callback = self.event_callback.clone();
            let engine_advertises_lossless = Arc::clone(&self.engine_advertises_lossless);
            let handle = thread::Builder::new()
                .name("orchard-ipc-events".to_string())
                .spawn(move || {
                    run_event_listener(
                        socket,
                        active_requests,
                        should_stop,
                        response_channel_id,
                        event_callback,
                        engine_advertises_lossless,
                    );
                });
            match handle {
                Ok(h) => self.event_listener_handle = Some(h),
                Err(e) => tracing::error!("Failed to spawn IPC event thread: {}", e),
            }
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
    engine_advertises_lossless: &AtomicBool,
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
            Ok(response) => {
                let reply: Value = serde_json::from_slice(&response)?;
                // A load_model reply for a model that is already up carries
                // the capabilities; a load that was only accepted brings
                // them in its `model_loaded` event.
                note_engine_capabilities(
                    reply.pointer("/data/load_model/capabilities"),
                    engine_advertises_lossless,
                );
                return Ok(reply);
            }
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

/// Whether `capabilities` (name to integers, from a load_model reply or a
/// `model_loaded` event) lists `lossless_responses` with value 1.
pub(crate) fn advertises_lossless_responses(capabilities: Option<&Value>) -> bool {
    capabilities
        .and_then(|capabilities| capabilities.get(LOSSLESS_RESPONSES_CAPABILITY))
        .map(|value| value.get(0).unwrap_or(value))
        .and_then(Value::as_i64)
        == Some(1)
}

fn note_engine_capabilities(capabilities: Option<&Value>, engine_advertises_lossless: &AtomicBool) {
    if advertises_lossless_responses(capabilities) {
        engine_advertises_lossless.store(true, Ordering::SeqCst);
    }
}

/// The management command that cancels one request of the client that
/// listens on `response_channel_id`. Every client counts request ids from 1:
/// an engine that advertises `lossless_responses` cancels by channel and id,
/// an older one ignores the channel and cancels every client's request with
/// that id.
pub(crate) fn cancel_request_command(response_channel_id: u64, request_id: u64) -> Value {
    serde_json::json!({
        "type": "cancel_request",
        "request_id": request_id,
        "response_channel_id": response_channel_id,
    })
}

/// Hand one response delta (the JSON after the topic) to the request it
/// belongs to. Never waits on a consumer: the per-request channel is unbounded,
/// so whatever a consumer has not read yet sits here, not in the engine.
fn route_response_delta(
    json_data: &[u8],
    active_requests: &Mutex<HashMap<u64, ActiveRequest>>,
    response_channel_id: u64,
) {
    let Ok(delta) = serde_json::from_slice::<ResponseDelta>(json_data) else {
        tracing::warn!(
            response_channel_id,
            payload_bytes = json_data.len(),
            "Failed to deserialize IPC response payload"
        );
        return;
    };
    let request_id = delta.request_id;
    let is_final = delta.is_final_delta;

    let sender = {
        let mut requests = active_requests.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = requests.get_mut(&request_id) {
            entry.deltas += 1;
            if is_final {
                entry.remaining_finals = entry.remaining_finals.saturating_sub(1);
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
}

/// Fail every request the engine has been silent on for longer than its
/// bound, and return their ids. A request that waits forever is what an engine
/// that cannot reach this client's response endpoint looks like from here: it
/// drops that client's deltas and nothing else tells the client.
fn fail_silent_requests(
    active_requests: &Mutex<HashMap<u64, ActiveRequest>>,
    timeouts: DeltaTimeouts,
) -> Vec<u64> {
    let now = Instant::now();
    let mut failed = Vec::new();
    let mut requests = active_requests.lock().unwrap_or_else(|e| e.into_inner());
    requests.retain(|request_id, entry| {
        if entry.deltas != entry.watched_deltas {
            entry.watched_deltas = entry.deltas;
            entry.last_progress = now;
            return true;
        }
        let awaited = if entry.deltas == 0 {
            "its first response delta"
        } else {
            "its next response delta"
        };
        let limit = if entry.deltas == 0 || entry.multi_stream {
            timeouts.first
        } else {
            timeouts.between
        };
        let silent_for = now.duration_since(entry.last_progress);
        if silent_for < limit {
            return true;
        }
        let reason = format!(
            "Timed out after {} s waiting for {awaited} from PIE.",
            silent_for.as_secs()
        );
        tracing::error!(request_id, deltas_received = entry.deltas, "{reason}");
        let _ = entry
            .sender
            .send(terminal_error_delta(*request_id, &reason));
        failed.push(*request_id);
        false
    });
    failed
}

/// Response listener - runs on dedicated thread for minimal latency.
///
/// `socket` is this client's PULL endpoint, so every message on it is one of
/// this client's deltas: the same `resp:<channel hex>:` topic and JSON body
/// the engine publishes on PUB/SUB.
#[allow(clippy::too_many_arguments)]
fn run_response_listener(
    socket: Socket,
    active_requests: Arc<Mutex<HashMap<u64, ActiveRequest>>>,
    should_stop: Arc<AtomicBool>,
    engine_dead: Arc<AtomicBool>,
    response_channel_id: u64,
    engine_pid_file: PathBuf,
    event_callback: Option<EventCallback>,
    delta_timeouts: DeltaTimeouts,
    management_socket: Arc<Mutex<Option<Socket>>>,
    engine_advertises_lossless: Arc<AtomicBool>,
) {
    let response_topic = format!("resp:{:x}:", response_channel_id);
    let response_topic_bytes = response_topic.as_bytes();

    // Set receive timeout for responsive polling (10ms for better latency)
    let _ = socket.set_opt::<nng::options::RecvTimeout>(Some(RESPONSE_RECV_TIMEOUT));
    let mut last_engine_check = Instant::now();
    let mut last_watchdog = Instant::now();
    let mut deltas_since_watchdog_check = 0u32;
    let mut death_reason: Option<String> = None;

    while !should_stop.load(Ordering::SeqCst) {
        match socket.recv() {
            Ok(msg) => {
                match msg.as_slice().strip_prefix(response_topic_bytes) {
                    Some(json_data) => {
                        route_response_delta(json_data, &active_requests, response_channel_id)
                    }
                    None => tracing::warn!(
                        response_channel_id,
                        payload_bytes = msg.len(),
                        "Ignoring a message for another channel on our response endpoint"
                    ),
                }
                deltas_since_watchdog_check += 1;
                if deltas_since_watchdog_check < DELTAS_PER_WATCHDOG_CHECK {
                    continue;
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

        // Reached on every receive timeout (every 10 ms while nothing
        // arrives) and once per DELTAS_PER_WATCHDOG_CHECK deltas otherwise.
        deltas_since_watchdog_check = 0;
        if last_watchdog.elapsed() >= DELTA_WATCHDOG_INTERVAL {
            last_watchdog = Instant::now();
            let timed_out = fail_silent_requests(&active_requests, delta_timeouts);
            // The engine would decode them to their token limit and keep
            // their cache pages. An engine that does not advertise
            // `lossless_responses` is not told: it would cancel every
            // client's request with that id. On a thread of its own: the
            // exchange waits for the management socket, which a load_model
            // can hold for minutes, and this thread receives the deltas.
            if !timed_out.is_empty() && engine_advertises_lossless.load(Ordering::SeqCst) {
                let management_socket = Arc::clone(&management_socket);
                let engine_dead = Arc::clone(&engine_dead);
                let engine_pid_file = engine_pid_file.clone();
                let engine_advertises_lossless = Arc::clone(&engine_advertises_lossless);
                let _ = thread::Builder::new()
                    .name("orchard-ipc-cancel".to_string())
                    .spawn(move || {
                        for request_id in timed_out {
                            let _ = blocking_management_exchange(
                                &management_socket,
                                &engine_dead,
                                Some(&engine_pid_file),
                                &cancel_request_command(response_channel_id, request_id),
                                Duration::from_secs(2),
                                &engine_advertises_lossless,
                            );
                        }
                    });
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
                let _ = entry
                    .sender
                    .send(terminal_error_delta(*request_id, &reason));
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

/// Event listener - the engine's broadcast events (telemetry, model_loaded,
/// engine_ready, ...) only ever come over PUB/SUB.
fn run_event_listener(
    socket: Socket,
    active_requests: Arc<Mutex<HashMap<u64, ActiveRequest>>>,
    should_stop: Arc<AtomicBool>,
    response_channel_id: u64,
    event_callback: Option<EventCallback>,
    engine_advertises_lossless: Arc<AtomicBool>,
) {
    let response_topic = format!("resp:{:x}:", response_channel_id);
    let response_topic_bytes = response_topic.as_bytes();
    let mut warned_about_lossy_route = false;

    let _ = socket.set_opt::<nng::options::RecvTimeout>(Some(EVENT_RECV_TIMEOUT));
    while !should_stop.load(Ordering::SeqCst) {
        match socket.recv() {
            Ok(msg) => {
                let data = msg.as_slice();
                if data.starts_with(EVENT_TOPIC_PREFIX) {
                    handle_engine_event(data, &event_callback, &engine_advertises_lossless);
                } else if let Some(json_data) = data.strip_prefix(response_topic_bytes) {
                    // An engine that advertises the lossless route publishes
                    // here only what it could not push to our endpoint.
                    if !warned_about_lossy_route
                        && !engine_advertises_lossless.load(Ordering::SeqCst)
                    {
                        warned_about_lossy_route = true;
                        tracing::warn!(
                            "PIE answers over PUB/SUB, where deltas are lost without an \
                             error whenever this process falls behind: no load_model \
                             reply or model_loaded event has advertised \
                             lossless_responses."
                        );
                    }
                    route_response_delta(json_data, &active_requests, response_channel_id);
                }
            }
            Err(nng::Error::TimedOut) => {}
            Err(error) => {
                if !should_stop.load(Ordering::SeqCst) {
                    tracing::error!(error = %error, "IPC event listener stopped");
                }
                break;
            }
        }
    }
}

/// Remove the response endpoints of clients that are gone.
///
/// A client that is killed leaves its socket file behind, and nothing listens
/// on that name again (it holds a process id and a random number), so NNG
/// never cleans it up. A file is stale when the process in the top half of
/// its channel id, where orchard-rs and orchard-py put their own pid (see
/// `rand_u64`), no longer exists; an id with nothing there is left alone.
///
/// Never probe an endpoint by connecting to it instead. A connection that
/// goes away before the handshake pauses a live listener's accepts for
/// 100 ms, and an NNG without the fix for upstream #1518 stops accepting for
/// good: the engine still connects, and every delta it sends is lost.
fn remove_stale_response_endpoints(ipc_root: &Path) {
    let Ok(entries) = std::fs::read_dir(ipc_root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let owner_pid = name
            .to_str()
            .and_then(|name| name.strip_prefix(RESPONSE_ROUTE_SOCKET_PREFIX))
            .and_then(|name| name.strip_suffix(RESPONSE_ROUTE_SOCKET_SUFFIX))
            .and_then(|channel_hex| u64::from_str_radix(channel_hex, 16).ok())
            .map(|channel_id| (channel_id >> 32) as u32);
        if owner_pid.is_some_and(|pid| pid != 0 && !pid_is_alive(pid)) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Handle an engine event (telemetry, model_loaded, etc.)
fn handle_engine_event(
    data: &[u8],
    event_callback: &Option<EventCallback>,
    engine_advertises_lossless: &AtomicBool,
) {
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
    if event_name == "model_loaded" {
        // Before the callback: it wakes the caller that waits for this model,
        // and that caller's first request must already see the capability.
        note_engine_capabilities(payload.get("capabilities"), engine_advertises_lossless);
    }

    // Dispatch to callback if registered
    if let Some(callback) = event_callback {
        callback(&event_name, &payload);
    }
}

/// Generate a unique response channel ID.
/// Format: (PID << 32) | random_32_bits
///
/// The pid is how a later client tells that the endpoint file named after
/// this id was left behind by a process that is gone.
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_client_creation() {
        let client = IPCClient::new();
        assert!(client.request_socket.is_none());
        assert!(client.response_channel_id > 0);
    }

    #[test]
    fn test_request_id_increment() {
        let client = IPCClient::new();
        let id1 = client.next_request_id();
        let id2 = client.next_request_id();
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_response_delta_default() {
        let delta = ResponseDelta::default();
        assert_eq!(delta.request_id, 0);
        assert!(!delta.is_final_delta);
        assert!(delta.tokens.is_empty());
        assert!(delta.top_logprobs.is_empty());
        assert!(delta.embedding_bytes.is_none());
        assert!(delta.state_events.is_empty());
    }

    #[test]
    fn test_response_delta_deserialize() {
        let json = serde_json::json!({
            "request_id": 123,
            "sequence_id": 1,
            "prompt_index": 0,
            "candidate_index": 0,
            "content": "Hello",
            "content_len": 5,
            "inline_content_bytes": 5,
            "is_final_delta": false,
            "num_tokens_in_delta": 3,
            "tokens": [1, 2, 3],
            "top_logprobs": [{"token": "hello", "logprob": -0.5}, {"token": "world", "logprob": -1.0}],
            "cumulative_logprob": -1.5,
            "modal_decoder_id": "moondream3.coord",
            "modal_bytes_b64": "AAAA",
            "embedding_bytes": [0, 0, 128, 63]
        });
        let delta: ResponseDelta = serde_json::from_value(json).expect("deserialize failed");
        assert_eq!(delta.request_id, 123);
        assert_eq!(delta.sequence_id, Some(1));
        assert_eq!(delta.candidate_index, Some(0));
        assert_eq!(delta.content_len, Some(5));
        assert_eq!(delta.num_tokens_in_delta, Some(3));
        assert_eq!(delta.tokens, vec![1, 2, 3]);
        assert_eq!(delta.top_logprobs.len(), 2);
        assert_eq!(delta.cumulative_logprob, Some(-1.5));
        assert_eq!(delta.modal_decoder_id, Some("moondream3.coord".to_string()));
        assert_eq!(delta.embedding_bytes, Some(vec![0, 0, 128, 63]));
        assert!(delta.state_events.is_empty());
    }

    #[test]
    fn test_top_logprobs_deserialize_pie_wire_format() {
        // Exact frame shape PIE emits (transport.cpp): token id rendered as a
        // string, value under `probability`. The values must land in
        // TokenLogProb::logprob, not default to 0.0.
        let wire = br#"{"request_id":7,"sequence_id":1,"prompt_index":0,"candidate_index":0,"content":"a","is_final_delta":false,"finish_reason":"DELTA","tokens":[976],"top_logprobs":[{"token":"976","probability":-0.052},{"token":"3575","probability":-3.25}]}"#;
        let delta: ResponseDelta = serde_json::from_slice(wire).expect("deserialize failed");
        assert_eq!(delta.tokens, vec![976]);
        assert_eq!(delta.top_logprobs.len(), 2);
        assert_eq!(delta.top_logprobs[0].token, "976");
        assert_eq!(delta.top_logprobs[0].logprob, -0.052);
        assert_eq!(delta.top_logprobs[1].token, "3575");
        assert_eq!(delta.top_logprobs[1].logprob, -3.25);
    }

    #[test]
    fn test_response_delta_deserialize_with_defaults() {
        // Test that missing fields get sensible defaults
        let json = serde_json::json!({
            "request_id": 42,
            "is_final_delta": true
        });
        let delta: ResponseDelta = serde_json::from_value(json).expect("deserialize failed");
        assert_eq!(delta.request_id, 42);
        assert!(delta.is_final_delta);
        assert!(delta.tokens.is_empty());
        assert!(delta.content.is_none());
        assert!(delta.state_events.is_empty());
    }

    #[test]
    fn test_response_delta_deserialize_embedding_bytes_from_base64() {
        let json = serde_json::json!({
            "request_id": 42,
            "is_final_delta": true,
            "embedding_bytes": "AAAAAA==",
        });

        let delta: ResponseDelta = serde_json::from_value(json).expect("deserialize failed");
        assert_eq!(delta.embedding_bytes, Some(vec![0, 0, 0, 0]));
    }

    #[test]
    fn test_response_delta_deserialize_error_message_alias() {
        let json = serde_json::json!({
            "request_id": 42,
            "is_final_delta": true,
            "error_message": "boom",
        });

        let delta: ResponseDelta = serde_json::from_value(json).expect("deserialize failed");
        assert_eq!(delta.error.as_deref(), Some("boom"));
    }

    #[test]
    fn test_listener_shutdown_fails_pending_requests_and_marks_engine_dead() {
        let socket = Socket::new(Protocol::Pull0).expect("pull socket");
        let active_requests: Arc<Mutex<HashMap<u64, ActiveRequest>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let (tx, mut rx) = mpsc::unbounded_channel();
        active_requests.lock().unwrap().insert(
            7,
            ActiveRequest {
                sender: tx,
                remaining_finals: 1,
                multi_stream: false,
                deltas: 0,
                watched_deltas: 0,
                last_progress: Instant::now(),
            },
        );

        let engine_dead = Arc::new(AtomicBool::new(false));
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_for_callback = Arc::clone(&events);
        let callback: EventCallback = Arc::new(move |name: &str, _payload: &Value| {
            events_for_callback.lock().unwrap().push(name.to_string());
        });

        let dir = tempdir().expect("tempdir should be available");
        run_response_listener(
            socket,
            Arc::clone(&active_requests),
            Arc::new(AtomicBool::new(true)),
            Arc::clone(&engine_dead),
            1,
            dir.path().join("engine.pid"),
            Some(callback),
            IPCClient::new().delta_timeouts,
            Arc::new(Mutex::new(None)),
            Arc::new(AtomicBool::new(false)),
        );

        assert!(engine_dead.load(Ordering::SeqCst));
        assert!(active_requests.lock().unwrap().is_empty());
        let delta = rx.try_recv().expect("terminal error delta");
        assert!(delta.is_final_delta);
        assert_eq!(delta.finish_reason.as_deref(), Some("error"));
        assert!(delta.error.is_some());
        assert_eq!(*events.lock().unwrap(), vec!["engine_died".to_string()]);
    }

    #[test]
    fn test_management_exchange_fails_fast_when_engine_flagged_dead() {
        let socket = Socket::new(Protocol::Req0).expect("req socket");
        let socket_arc = Mutex::new(Some(socket));
        let engine_dead = AtomicBool::new(true);

        let result = blocking_management_exchange(
            &socket_arc,
            &engine_dead,
            None,
            &serde_json::json!({"type": "ping"}),
            Duration::from_secs(1),
            &AtomicBool::new(false),
        );

        assert!(matches!(result, Err(Error::EngineDead)));
    }

    #[test]
    fn test_management_exchange_detects_engine_exit_while_awaiting_reply() {
        let socket = Socket::new(Protocol::Req0).expect("req socket");
        let socket_arc = Mutex::new(Some(socket));
        let engine_dead = AtomicBool::new(false);
        let dir = tempdir().expect("tempdir should be available");

        let result = blocking_management_exchange(
            &socket_arc,
            &engine_dead,
            Some(&dir.path().join("missing.pid")),
            &serde_json::json!({"type": "ping"}),
            Duration::from_secs(5),
            &AtomicBool::new(false),
        );

        assert!(matches!(result, Err(Error::EngineDead)));
    }

    #[test]
    fn test_engine_process_is_alive_reads_pid_file() {
        let dir = tempdir().expect("tempdir should be available");
        let pid_file = dir.path().join("engine.pid");
        std::fs::write(&pid_file, format!("{}\n", std::process::id()))
            .expect("pid file should be written");

        assert!(engine_process_is_alive(&pid_file));
    }

    #[test]
    fn test_engine_process_is_alive_handles_missing_pid_file() {
        let dir = tempdir().expect("tempdir should be available");
        let pid_file = dir.path().join("missing.pid");

        assert!(!engine_process_is_alive(&pid_file));
    }

    #[test]
    fn test_response_delta_deserialize_with_state_events() {
        let json = serde_json::json!({
            "request_id": 7,
            "is_final_delta": false,
            "state_events": [
                {
                    "event_type": "item_started",
                    "item_type": "message",
                    "output_index": 0,
                    "identifier": "",
                    "delta": ""
                },
                {
                    "event_type": "content_delta",
                    "item_type": "message",
                    "output_index": 0,
                    "identifier": "",
                    "delta": "hello"
                }
            ]
        });

        let delta: ResponseDelta = serde_json::from_value(json).expect("deserialize failed");
        assert_eq!(delta.request_id, 7);
        assert_eq!(delta.state_events.len(), 2);
        assert_eq!(delta.state_events[0].event_type, "item_started");
        assert_eq!(delta.state_events[1].delta, "hello");
    }

    // --- The response route, against a stand-in for the engine's wire side ---

    /// What PIE does on the wire, without a model. It PULLs requests, PUBlishes
    /// events, and answers each request on the route the request asked for:
    /// PUSH to the client's own endpoint for "pull_v1" (dialled on the first
    /// delta, nothing buffered on the send side), PUB/SUB otherwise.
    struct FakeEngine {
        root: PathBuf,
        /// False stands in for an engine older than the lossless route: it
        /// ignores the field it does not know and answers on PUB/SUB.
        knows_pull_route: bool,
        requests: Socket,
        publisher: Socket,
        management: Socket,
        routes: HashMap<u64, Socket>,
    }

    impl FakeEngine {
        fn start(root: &Path) -> Self {
            let requests = Socket::new(Protocol::Pull0).expect("pull socket");
            requests
                .set_opt::<nng::options::RecvTimeout>(Some(Duration::from_secs(10)))
                .expect("recv timeout");
            requests.listen(&request_url_in(root)).expect("listen");
            let publisher = Socket::new(Protocol::Pub0).expect("pub socket");
            // The engine's own NNG_OPT_SENDBUF on this socket.
            publisher
                .set_opt::<nng::options::SendBufferSize>(1024)
                .expect("send buffer");
            publisher.listen(&response_url_in(root)).expect("listen");
            let management = Socket::new(Protocol::Rep0).expect("rep socket");
            management
                .set_opt::<nng::options::RecvTimeout>(Some(Duration::from_secs(10)))
                .expect("recv timeout");
            management.listen(&management_url_in(root)).expect("listen");
            Self {
                root: root.to_path_buf(),
                knows_pull_route: true,
                requests,
                publisher,
                management,
                routes: HashMap::new(),
            }
        }

        /// Metadata of the next request a client sent.
        fn next_request(&self) -> Value {
            let frame = self.requests.recv().expect("a request");
            let length = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize;
            serde_json::from_slice(&frame[4..4 + length]).expect("request metadata")
        }

        fn send_delta(&mut self, request: &Value, content: &str, is_final_delta: bool) {
            let channel_id = request["response_channel_id"].as_u64().expect("channel id");
            let mut message = format!("resp:{channel_id:x}:").into_bytes();
            let delta = serde_json::json!({
                "request_id": request["request_id"],
                "content": content,
                "is_final_delta": is_final_delta,
            });
            message.extend_from_slice(&serde_json::to_vec(&delta).unwrap());

            if self.knows_pull_route && request["response_transport"] == "pull_v1" {
                let root = &self.root;
                let route = self.routes.entry(channel_id).or_insert_with(|| {
                    let route = Socket::new(Protocol::Push0).expect("push socket");
                    route
                        .set_opt::<nng::options::SendBufferSize>(0)
                        .expect("send buffer");
                    route
                        .set_opt::<nng::options::SendTimeout>(Some(Duration::from_secs(10)))
                        .expect("send timeout");
                    route
                        .dial(&as_ipc_url(response_route_path(root, channel_id)))
                        .expect("the client listens on its endpoint before it sends");
                    route
                });
                route
                    .send(message.as_slice())
                    .map_err(|(_, error)| error)
                    .expect("push a delta");
            } else {
                self.publisher
                    .send(message.as_slice())
                    .map_err(|(_, error)| error)
                    .expect("publish a delta");
            }
        }

        /// Answer with `count` deltas whose contents are "0", "1", ...
        fn answer(&mut self, request: &Value, count: usize) {
            for index in 0..count {
                self.send_delta(request, &index.to_string(), index + 1 == count);
            }
        }

        /// Answer the next management command with `reply`, and return it.
        fn answer_management(&self, reply: &Value) -> Value {
            let command = self.management.recv().expect("a management command");
            self.management
                .send(serde_json::to_vec(reply).unwrap().as_slice())
                .map_err(|(_, error)| error)
                .expect("reply");
            serde_json::from_slice(&command).expect("management command")
        }

        fn publish_event(&self, name: &str, payload: &Value) {
            let mut message = EVENT_TOPIC_PREFIX.to_vec();
            message.extend_from_slice(name.as_bytes());
            message.push(0);
            message.extend_from_slice(&serde_json::to_vec(payload).unwrap());
            let _ = self.publisher.send(message.as_slice());
        }
    }

    /// An IPC root short enough for a socket path whatever TMPDIR is.
    fn ipc_dir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("orc-")
            .tempdir_in("/tmp")
            .expect("tempdir should be available")
    }

    /// Connect to an engine this process launched itself, which is asked for
    /// the lossless route whatever it advertises.
    fn connect(client: &mut IPCClient, root: &Path) {
        connect_as(client, root, true);
    }

    /// Connect to an engine that was already running.
    fn connect_shared(client: &mut IPCClient, root: &Path) {
        connect_as(client, root, false);
    }

    fn connect_as(client: &mut IPCClient, root: &Path, own_engine: bool) {
        let pid_file = root.join("engine.pid");
        std::fs::write(&pid_file, format!("{}\n", std::process::id())).expect("pid file");
        client
            .connect_in(root, pid_file, own_engine)
            .expect("connect");
    }

    /// Name and payload of every event a client handled.
    type RecordedEvents = Arc<Mutex<Vec<(String, Value)>>>;

    /// A client that records the engine's events, and those events.
    fn client_recording_events() -> (IPCClient, RecordedEvents) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_for_callback = Arc::clone(&events);
        let client =
            IPCClient::with_event_callback(Arc::new(move |name: &str, payload: &Value| {
                events_for_callback
                    .lock()
                    .unwrap()
                    .push((name.to_string(), payload.clone()));
            }));
        (client, events)
    }

    /// Publish `payload` as `name` until the client has handled it: PUB/SUB
    /// does not queue for a subscriber it has not seen yet.
    fn publish_until_handled(
        engine: &FakeEngine,
        events: &RecordedEvents,
        name: &str,
        payload: &Value,
    ) {
        let deadline = Instant::now() + Duration::from_secs(10);
        let handled = || {
            events
                .lock()
                .unwrap()
                .iter()
                .any(|(seen_name, seen)| seen_name == name && seen == payload)
        };
        while !handled() {
            assert!(Instant::now() < deadline, "the event never arrived");
            engine.publish_event(name, payload);
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn send(client: &IPCClient, request_id: u64) -> mpsc::UnboundedReceiver<ResponseDelta> {
        let (_, deltas) = client
            .send_batch_request(request_id, "model", "/model", &[PromptPayload::default()])
            .expect("send");
        deltas
    }

    /// The final delta of `deltas`, and the contents before it.
    fn contents_until_error(
        deltas: &mut mpsc::UnboundedReceiver<ResponseDelta>,
    ) -> (Vec<String>, ResponseDelta) {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut contents = Vec::new();
        loop {
            match deltas.try_recv() {
                Ok(delta) if delta.is_final_delta => return (contents, delta),
                Ok(delta) => contents.push(delta.content.unwrap_or_default()),
                Err(_) => {
                    assert!(
                        Instant::now() < deadline,
                        "the request never ended; got {} deltas",
                        contents.len()
                    );
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }

    /// Contents of every delta up to and including the final one.
    fn contents(deltas: &mut mpsc::UnboundedReceiver<ResponseDelta>) -> Vec<String> {
        let (mut contents, last) = contents_until_error(deltas);
        contents.push(last.content.unwrap_or_default());
        contents
    }

    fn counting(count: usize) -> Vec<String> {
        (0..count).map(|index| index.to_string()).collect()
    }

    #[test]
    fn test_deltas_pushed_to_the_endpoint_reach_their_request_in_order() {
        let dir = ipc_dir();
        let engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        connect(&mut client, dir.path());

        let mut first = send(&client, 1);
        let mut second = send(&client, 2);
        let requests = [engine.next_request(), engine.next_request()];
        assert_eq!(requests[0]["request_id"], 1);
        assert_eq!(requests[1]["request_id"], 2);

        // Straight to the endpoint, whatever the request asked for.
        let endpoint = as_ipc_url(response_route_path(dir.path(), client.response_channel_id));
        let pusher = Socket::new(Protocol::Push0).expect("push socket");
        pusher
            .dial(&endpoint)
            .expect("the endpoint is up before a request is sent");
        let topic = format!("resp:{:x}:", client.response_channel_id);
        for index in 0..200 {
            for request_id in [1, 2] {
                let delta = serde_json::json!({
                    "request_id": request_id,
                    "content": format!("{request_id}.{index}"),
                    "is_final_delta": index == 199,
                });
                pusher
                    .send(format!("{topic}{delta}").as_bytes())
                    .map_err(|(_, error)| error)
                    .expect("push a delta");
            }
        }

        let expected = |request_id: u64| -> Vec<String> {
            (0..200)
                .map(|index| format!("{request_id}.{index}"))
                .collect()
        };
        assert_eq!(contents(&mut first), expected(1));
        assert_eq!(contents(&mut second), expected(2));
    }

    #[test]
    fn test_a_stalled_receive_loop_loses_no_delta() {
        const DELTAS: usize = 6000;
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        connect(&mut client, dir.path());

        let mut deltas = send(&client, 1);
        let request = engine.next_request();
        let engine = thread::spawn(move || {
            engine.answer(&request, DELTAS);
            engine
        });

        // The receive loop takes this lock for every delta: holding it is a
        // client that falls 1.5 s behind while the engine keeps producing.
        let stall = client.active_requests.lock().unwrap();
        thread::sleep(Duration::from_millis(1500));
        drop(stall);

        let received = contents(&mut deltas);
        assert_eq!(received.len(), DELTAS, "deltas were lost during the stall");
        assert_eq!(received, counting(DELTAS));
        drop(engine.join().expect("engine thread"));
    }

    #[test]
    fn test_two_clients_on_one_root_get_only_their_own_deltas() {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let mut clients = [IPCClient::new(), IPCClient::new()];
        for client in &mut clients {
            connect(client, dir.path());
        }

        // Every client counts request ids from 1: only the channel tells them apart.
        let mut streams = [send(&clients[0], 1), send(&clients[1], 1)];
        for _ in 0..2 {
            let request = engine.next_request();
            let channel_id = request["response_channel_id"].as_u64().unwrap();
            for index in 0..50 {
                engine.send_delta(&request, &format!("{channel_id:x}.{index}"), index == 49);
            }
        }

        for (client, deltas) in clients.iter().zip(&mut streams) {
            let expected: Vec<String> = (0..50)
                .map(|index| format!("{:x}.{index}", client.response_channel_id))
                .collect();
            assert_eq!(contents(deltas), expected);
            assert!(deltas.try_recv().is_err(), "a delta of the other client");
        }
    }

    #[test]
    fn test_disconnect_closes_and_removes_the_endpoint() {
        let dir = ipc_dir();
        let _engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        let endpoint = response_route_path(dir.path(), client.response_channel_id);

        connect(&mut client, dir.path());
        assert!(endpoint.exists(), "the endpoint is up once connect returns");

        client.disconnect();
        assert!(!endpoint.exists(), "disconnect removes the socket file");
        let pusher = Socket::new(Protocol::Push0).expect("push socket");
        assert!(pusher.dial(&as_ipc_url(endpoint.clone())).is_err());

        connect(&mut client, dir.path());
        assert!(endpoint.exists());
        drop(client);
        assert!(!endpoint.exists(), "drop removes the socket file");
    }

    #[test]
    fn test_connecting_again_replaces_the_endpoint() {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        connect(&mut client, dir.path());
        // No disconnect in between: the endpoint of the first connect is
        // still bound to the path the second one listens on.
        connect(&mut client, dir.path());

        let mut deltas = send(&client, 1);
        let request = engine.next_request();
        engine.answer(&request, 3);
        assert_eq!(contents(&mut deltas), counting(3));
    }

    #[test]
    fn test_the_same_listener_hears_a_restarted_engine() {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        connect(&mut client, dir.path());

        let mut deltas = send(&client, 1);
        let request = engine.next_request();
        engine.answer(&request, 3);
        assert_eq!(contents(&mut deltas), counting(3));

        // A new engine in the same root knows nothing of this client until its
        // first delta for it, and then dials the endpoint that is still up.
        drop(engine);
        let mut engine = FakeEngine::start(dir.path());
        let mut deltas = send(&client, 2);
        let request = engine.next_request();
        engine.answer(&request, 3);
        assert_eq!(contents(&mut deltas), counting(3));
    }

    #[test]
    fn test_reconnect_after_an_engine_restart_still_receives() {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        connect(&mut client, dir.path());

        let mut deltas = send(&client, 1);
        let request = engine.next_request();
        engine.answer(&request, 3);
        assert_eq!(contents(&mut deltas), counting(3));

        drop(engine);
        client.disconnect();
        let mut engine = FakeEngine::start(dir.path());
        connect(&mut client, dir.path());

        let mut deltas = send(&client, 2);
        let request = engine.next_request();
        engine.answer(&request, 3);
        assert_eq!(contents(&mut deltas), counting(3));
    }

    #[test]
    fn test_events_and_an_older_engines_deltas_arrive_over_pub_sub() {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        engine.knows_pull_route = false;

        let (mut client, events) = client_recording_events();
        connect(&mut client, dir.path());
        publish_until_handled(
            &engine,
            &events,
            "model_loaded",
            &serde_json::json!({"model_id": "model"}),
        );

        let mut deltas = send(&client, 1);
        let request = engine.next_request();
        engine.answer(&request, 3);
        assert_eq!(contents(&mut deltas), counting(3));
    }

    #[test]
    fn test_an_engine_this_process_launched_is_asked_for_the_lossless_route() {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        connect(&mut client, dir.path());

        // No capability was ever advertised: launching the engine is enough.
        let mut deltas = send(&client, 1);
        let request = engine.next_request();
        assert_eq!(request["response_transport"], "pull_v1");
        engine.answer(&request, 3);
        assert_eq!(contents(&mut deltas), counting(3));
    }

    #[test]
    fn test_a_shared_engine_is_asked_for_the_lossless_route_once_a_model_loaded_event_advertises_it(
    ) {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let (mut client, events) = client_recording_events();
        connect_shared(&mut client, dir.path());

        // A model of an engine that does not advertise the capability: the
        // request is the one on main, and its answer comes over PUB/SUB.
        let older = serde_json::json!({"model_id": "a", "capabilities": {"answer": [3]}});
        publish_until_handled(&engine, &events, "model_loaded", &older);
        let mut deltas = send(&client, 1);
        let request = engine.next_request();
        assert!(request.get("response_transport").is_none(), "{request}");
        engine.answer(&request, 3);
        assert_eq!(contents(&mut deltas), counting(3));

        let reaping = serde_json::json!({
            "model_id": "b",
            "capabilities": {"answer": [3], "lossless_responses": [1]},
        });
        publish_until_handled(&engine, &events, "model_loaded", &reaping);
        let mut deltas = send(&client, 2);
        let request = engine.next_request();
        assert_eq!(request["response_transport"], "pull_v1");
        engine.answer(&request, 3);
        assert_eq!(contents(&mut deltas), counting(3));
    }

    #[test]
    fn test_a_shared_engine_is_asked_for_the_lossless_route_once_a_load_model_reply_advertises_it()
    {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        connect_shared(&mut client, dir.path());

        let load_model = |engine: FakeEngine, client: &IPCClient, value: Value| {
            let reply = serde_json::json!({
                "status": "ok",
                "data": {"load_model": {
                    "runtime_started": true,
                    "capabilities": {"answer": [3], "lossless_responses": value},
                }},
            });
            let engine = thread::spawn(move || {
                engine.answer_management(&reply);
                engine
            });
            client
                .send_management_command(
                    &serde_json::json!({"type": "load_model"}),
                    Duration::from_secs(5),
                )
                .expect("load_model reply");
            engine.join().expect("engine thread")
        };

        // Listed, but not with the value that promises reaping.
        engine = load_model(engine, &client, serde_json::json!([0]));
        let _deltas = send(&client, 1);
        let request = engine.next_request();
        assert!(request.get("response_transport").is_none(), "{request}");

        engine = load_model(engine, &client, serde_json::json!([1]));
        let mut deltas = send(&client, 2);
        let request = engine.next_request();
        assert_eq!(request["response_transport"], "pull_v1");
        engine.answer(&request, 3);
        assert_eq!(contents(&mut deltas), counting(3));

        // What one engine advertised says nothing about the next one.
        client.disconnect();
        connect_shared(&mut client, dir.path());
        let _deltas = send(&client, 3);
        let request = engine.next_request();
        assert!(request.get("response_transport").is_none(), "{request}");
    }

    #[tokio::test]
    async fn test_a_new_client_of_a_registry_that_heard_the_capability_asks_for_the_lossless_route()
    {
        let dir = ipc_dir();
        let engine = FakeEngine::start(dir.path());
        // Heard through an earlier client of this registry. Its models stay
        // Ready, so it never sends the load_model whose reply would tell the
        // next client.
        let registry = crate::model::registry::ModelRegistry::new().expect("registry");
        registry
            .handle_model_loaded(&serde_json::json!({
                "model_id": "model",
                "capabilities": {"answer": [3], "lossless_responses": [1]},
            }))
            .await;

        let mut client = IPCClient::new();
        connect_shared(&mut client, dir.path());
        let client = Arc::new(client);
        registry.set_ipc_client(Arc::clone(&client)).await;

        let _deltas = send(&client, 1);
        assert_eq!(engine.next_request()["response_transport"], "pull_v1");
    }

    #[test]
    fn test_only_lossless_responses_with_value_one_counts_as_advertised() {
        let advertised = |capabilities: Value| {
            let flag = AtomicBool::new(false);
            note_engine_capabilities(Some(&capabilities), &flag);
            flag.load(Ordering::SeqCst)
        };
        // The engine writes every capability as a list of integers.
        assert!(advertised(serde_json::json!({"lossless_responses": [1]})));
        assert!(advertised(serde_json::json!({"lossless_responses": 1})));
        assert!(!advertised(serde_json::json!({"lossless_responses": [0]})));
        assert!(!advertised(serde_json::json!({"lossless_responses": []})));
        assert!(!advertised(serde_json::json!({"lossless_responses": "1"})));
        assert!(!advertised(serde_json::json!({"answer": [1]})));

        let flag = AtomicBool::new(false);
        note_engine_capabilities(None, &flag);
        assert!(!flag.load(Ordering::SeqCst));
    }

    #[test]
    fn test_a_request_the_engine_never_answers_fails_after_the_first_delta_timeout() {
        let dir = ipc_dir();
        let engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        client.set_delta_timeouts(Duration::from_millis(200), Duration::from_secs(60));
        connect(&mut client, dir.path());

        // The engine has the request and never answers it: from here that is
        // an engine that could not dial our endpoint and dropped the deltas.
        let mut deltas = send(&client, 1);
        engine.next_request();

        let (contents, error) = contents_until_error(&mut deltas);
        assert!(contents.is_empty());
        assert_eq!(error.request_id, 1);
        assert_eq!(error.finish_reason.as_deref(), Some("error"));
        let message = error.error.expect("an error the caller can see");
        assert!(message.contains("first response delta"), "{message}");
        assert!(client.active_requests.lock().unwrap().is_empty());
        assert!(deltas.try_recv().is_err(), "the request ends exactly once");
    }

    #[test]
    fn test_a_request_the_watchdog_fails_is_cancelled_in_an_engine_that_cancels_by_channel() {
        let dir = ipc_dir();
        let engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        client.set_delta_timeouts(Duration::from_millis(200), Duration::from_secs(60));
        connect(&mut client, dir.path());

        // Nothing says yet that this engine cancels by channel: it may cancel
        // every client's request 1, so it is not asked to.
        let mut deltas = send(&client, 1);
        engine.next_request();
        contents_until_error(&mut deltas);

        client.note_lossless_responses();
        let mut deltas = send(&client, 2);
        engine.next_request();
        contents_until_error(&mut deltas);

        let command = engine.answer_management(&serde_json::json!({"status": "accepted"}));
        assert_eq!(
            command,
            serde_json::json!({
                "type": "cancel_request",
                "request_id": 2,
                "response_channel_id": client.response_channel_id,
            })
        );
    }

    #[test]
    fn test_a_stream_that_goes_silent_fails_after_the_delta_timeout() {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        client.set_delta_timeouts(Duration::from_secs(60), Duration::from_millis(200));
        connect(&mut client, dir.path());

        let mut deltas = send(&client, 1);
        let request = engine.next_request();
        for index in 0..3 {
            engine.send_delta(&request, &index.to_string(), false);
        }

        let (contents, error) = contents_until_error(&mut deltas);
        assert_eq!(contents, counting(3));
        assert_eq!(error.finish_reason.as_deref(), Some("error"));
        let message = error.error.expect("an error the caller can see");
        assert!(message.contains("next response delta"), "{message}");
        assert!(client.active_requests.lock().unwrap().is_empty());
    }

    #[test]
    fn test_a_request_with_several_streams_is_held_to_the_first_delta_timeout_only() {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        client.set_delta_timeouts(Duration::from_secs(4), Duration::from_millis(100));
        connect(&mut client, dir.path());

        let prompts = [PromptPayload::default(), PromptPayload::default()];
        let (_, mut deltas) = client
            .send_batch_request(1, "model", "/model", &prompts)
            .expect("send");
        let request = engine.next_request();
        // One prompt's stream runs and ends while the other is still queued
        // or prefilling, which the engine is silent through.
        engine.answer(&request, 3);
        assert_eq!(contents(&mut deltas), counting(3));

        thread::sleep(Duration::from_millis(2300));
        assert!(
            deltas.try_recv().is_err(),
            "failed by the between-deltas bound while a stream had yet to start"
        );

        let (_, error) = contents_until_error(&mut deltas);
        let message = error.error.expect("an error the caller can see");
        assert!(message.contains("next response delta"), "{message}");
    }

    #[test]
    fn test_a_stream_that_keeps_arriving_outlives_the_delta_timeout() {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        // Both bounds are far shorter than the stream: what they bound is
        // the engine's silence, not how long a request takes.
        client.set_delta_timeouts(Duration::from_millis(1500), Duration::from_millis(1500));
        connect(&mut client, dir.path());

        let mut deltas = send(&client, 1);
        let request = engine.next_request();
        for index in 0..8 {
            thread::sleep(Duration::from_millis(500));
            engine.send_delta(&request, &index.to_string(), index == 7);
        }
        // Read only now: a consumer that is slow to read is not silence.
        assert_eq!(contents(&mut deltas), counting(8));
    }

    #[test]
    fn test_a_silent_request_fails_while_another_streams_without_a_pause() {
        let dir = ipc_dir();
        let mut engine = FakeEngine::start(dir.path());
        let mut client = IPCClient::new();
        client.set_delta_timeouts(Duration::from_millis(200), Duration::from_secs(60));
        connect(&mut client, dir.path());

        let mut silent = send(&client, 1);
        let mut busy = send(&client, 2);
        engine.next_request();
        let request = engine.next_request();

        // Deltas back to back for three seconds: the receive loop never sees
        // a receive timeout, which is where it otherwise runs the watchdog.
        let streaming = Arc::new(AtomicBool::new(true));
        let streaming_in_engine = Arc::clone(&streaming);
        let engine = thread::spawn(move || {
            let until = Instant::now() + Duration::from_secs(3);
            let mut sent = 0usize;
            while Instant::now() < until {
                engine.send_delta(&request, "", false);
                sent += 1;
            }
            streaming_in_engine.store(false, Ordering::SeqCst);
            engine.send_delta(&request, "", true);
            (engine, sent + 1)
        });

        let (_, error) = contents_until_error(&mut silent);
        assert!(
            streaming.load(Ordering::SeqCst),
            "the silent request only failed once the other stream paused"
        );
        assert!(error.error.is_some());

        let (engine, sent) = engine.join().expect("engine thread");
        assert_eq!(contents(&mut busy).len(), sent);
        drop(engine);
    }

    #[test]
    fn test_connect_removes_the_endpoints_of_dead_clients_only() {
        use std::os::unix::net::UnixListener;

        let dir = ipc_dir();
        let _engine = FakeEngine::start(dir.path());

        let mut child = std::process::Command::new("true").spawn().expect("spawn");
        let dead_pid = u64::from(child.id());
        child.wait().expect("wait");
        // std leaves the socket file behind when the listener goes, as a
        // killed client does.
        let stale = response_route_path(dir.path(), (dead_pid << 32) | 7);
        drop(UnixListener::bind(&stale).expect("bind"));
        let alive = response_route_path(dir.path(), (u64::from(std::process::id()) << 32) | 7);
        drop(UnixListener::bind(&alive).expect("bind"));
        let foreign = response_route_path(dir.path(), 7);
        drop(UnixListener::bind(&foreign).expect("bind"));

        let mut client = IPCClient::new();
        connect(&mut client, dir.path());

        assert!(!stale.exists(), "its owner is gone");
        assert!(alive.exists(), "its owner is alive");
        assert!(foreign.exists(), "its id names no owner");
    }

    #[test]
    fn test_a_peer_gone_before_the_handshake_does_not_deafen_a_listener() {
        // A peer that connects and is gone before NNG's handshake, as an
        // engine killed while dialling a client's endpoint is. NNG 1.4.0 took
        // the resulting "closed" for its own listener closing and never
        // accepted again (upstream #1518): later peers still connected, and
        // what they sent was never received.
        let dir = ipc_dir();
        let path = dir.path().join("listener.ipc");
        let url = as_ipc_url(path.clone());
        let listener = Socket::new(Protocol::Pull0).expect("pull socket");
        listener
            .set_opt::<nng::options::RecvTimeout>(Some(Duration::from_secs(5)))
            .expect("recv timeout");
        listener.listen(&url).expect("listen");

        // Whether NNG sees "closed" depends on the peer closing before NNG
        // writes its half of the handshake, which a single try can miss.
        for _ in 0..3 {
            drop(std::os::unix::net::UnixStream::connect(&path).expect("connect"));
            thread::sleep(Duration::from_millis(150));
        }

        let peer = Socket::new(Protocol::Push0).expect("push socket");
        peer.dial(&url).expect("dial");
        peer.send(b"delta".as_slice())
            .map_err(|(_, error)| error)
            .expect("send");
        let received = listener.recv().expect("the listener still accepts");
        assert_eq!(received.as_slice(), b"delta");
    }
}
