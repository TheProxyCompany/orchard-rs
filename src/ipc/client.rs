//! High-performance IPC client for communicating with PIE.
//!
//! Uses NNG sockets with a dedicated listener thread for response handling.

use crate::error::{Error, Result};
use crate::ipc::endpoints::{management_url, request_url, response_url, EVENT_TOPIC_PREFIX};
use crate::ipc::serialization::{build_batch_request_payload, PromptPayload, RequestType};

use nng::options::Options;
use nng::{Protocol, Socket};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tokio::sync::mpsc;

/// Callback type for engine events (telemetry, model_loaded, etc.)
pub type EventCallback = Arc<dyn Fn(&str, &Value) + Send + Sync>;

/// Response delta from PIE.
///
/// Uses serde for deserialization with sensible defaults for missing fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Error message if request failed
    pub error: Option<String>,
    /// Prompt token count
    pub prompt_token_count: Option<u32>,
    /// Number of tokens in this delta
    pub num_tokens_in_delta: Option<u32>,
    /// Generation length so far
    pub generation_len: Option<u32>,
    /// Token IDs in this delta
    pub tokens: Vec<i32>,
    /// Top log probabilities for each token
    pub top_logprobs: Vec<HashMap<String, f64>>,
    /// Cumulative log probability
    pub cumulative_logprob: Option<f64>,
    /// Modal decoder identifier (e.g., "moondream3.coord")
    pub modal_decoder_id: Option<String>,
    /// Base64-encoded modal decoder output bytes
    pub modal_bytes_b64: Option<String>,
}

impl Default for ResponseDelta {
    fn default() -> Self {
        Self {
            request_id: 0,
            sequence_id: None,
            prompt_index: None,
            candidate_index: None,
            content: None,
            content_len: None,
            inline_content_bytes: None,
            is_final_delta: false,
            finish_reason: None,
            error: None,
            prompt_token_count: None,
            num_tokens_in_delta: None,
            generation_len: None,
            tokens: Vec::new(),
            top_logprobs: Vec::new(),
            cumulative_logprob: None,
            modal_decoder_id: None,
            modal_bytes_b64: None,
        }
    }
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
            event_callback: None,
        }
    }

    /// Create a new IPC client with an event callback.
    pub fn with_event_callback(callback: EventCallback) -> Self {
        Self {
            request_socket: None,
            response_socket: None,
            management_socket: Arc::new(Mutex::new(None)),
            response_channel_id: rand_u64(),
            request_id_counter: AtomicU64::new(0),
            active_requests: Arc::new(Mutex::new(HashMap::new())),
            listener_handle: None,
            should_stop: Arc::new(AtomicBool::new(false)),
            event_callback: Some(callback),
        }
    }

    /// Get a clone of the management socket Arc for async operations.
    pub fn management_socket(&self) -> Arc<Mutex<Option<Socket>>> {
        Arc::clone(&self.management_socket)
    }

    /// Set the event callback for handling engine events.
    pub fn set_event_callback(&mut self, callback: EventCallback) {
        self.event_callback = Some(callback);
    }

    /// Connect to PIE IPC endpoints.
    pub fn connect(&mut self) -> Result<()> {
        // Create and connect request socket (PUSH)
        let request_socket = Socket::new(Protocol::Push0)?;
        request_socket.dial(&request_url())?;
        self.request_socket = Some(request_socket);

        // Create response socket (SUB) - subscribe BEFORE dial
        let response_socket = Socket::new(Protocol::Sub0)?;

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
            let mut mgmt = self.management_socket.lock().unwrap_or_else(|e| e.into_inner());
            *mgmt = Some(management_socket);
        }

        // Start listener thread
        self.should_stop.store(false, Ordering::SeqCst);
        self.start_listener();

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
            let mut mgmt = self.management_socket.lock().unwrap_or_else(|e| e.into_inner());
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
        let socket = self.request_socket.as_ref().ok_or(Error::NotConnected)?;

        let payload = build_batch_request_payload(
            request_id,
            model_id,
            model_path,
            RequestType::Generation,
            self.response_channel_id,
            prompts,
        )?;

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

        self.active_requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(request_id, ActiveRequest {
                sender: tx,
                remaining_finals,
            });

        let msg = nng::Message::from(payload.as_slice());
        socket.send(msg).map_err(|(_, e)| Error::Nng(e))?;

        Ok((prompts.len(), rx))
    }

    /// Send a management command asynchronously.
    ///
    /// Uses spawn_blocking internally since NNG sockets are sync.
    pub async fn send_management_command_async(&self, command: Value, timeout: Duration) -> Result<Value> {
        let socket_arc = Arc::clone(&self.management_socket);

        tokio::task::spawn_blocking(move || {
            let guard = socket_arc.lock().unwrap_or_else(|e| e.into_inner());
            let socket = guard.as_ref().ok_or(Error::NotConnected)?;

            // Set timeout
            socket.set_opt::<nng::options::RecvTimeout>(Some(timeout))?;

            // Send command
            let data = serde_json::to_vec(&command)?;
            let msg = nng::Message::from(data.as_slice());
            socket.send(msg).map_err(|(_, e)| Error::Nng(e))?;

            // Receive response
            let response = socket.recv()?;
            let json: Value = serde_json::from_slice(&response)?;

            Ok(json)
        })
        .await
        .map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }

    /// Send a management command synchronously (blocking).
    ///
    /// Prefer `send_management_command_async` in async contexts.
    pub fn send_management_command(&self, command: &Value, timeout: Duration) -> Result<Value> {
        let guard = self.management_socket.lock().unwrap_or_else(|e| e.into_inner());
        let socket = guard.as_ref().ok_or(Error::NotConnected)?;

        // Set timeout
        socket.set_opt::<nng::options::RecvTimeout>(Some(timeout))?;

        // Send command
        let data = serde_json::to_vec(command)?;
        let msg = nng::Message::from(data.as_slice());
        socket.send(msg).map_err(|(_, e)| Error::Nng(e))?;

        // Receive response
        let response = socket.recv()?;
        let json: Value = serde_json::from_slice(&response)?;

        Ok(json)
    }

    /// Start the response listener thread.
    fn start_listener(&mut self) {
        let response_socket = self.response_socket.take();
        let active_requests = Arc::clone(&self.active_requests);
        let should_stop = Arc::clone(&self.should_stop);
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
                        response_channel_id,
                        event_callback,
                    );
                }
            });

        match handle {
            Ok(h) => self.listener_handle = Some(h),
            Err(e) => log::error!("Failed to spawn IPC listener thread: {}", e),
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

/// Response listener - runs on dedicated thread for minimal latency.
fn run_response_listener(
    socket: Socket,
    active_requests: Arc<Mutex<HashMap<u64, ActiveRequest>>>,
    should_stop: Arc<AtomicBool>,
    response_channel_id: u64,
    event_callback: Option<EventCallback>,
) {
    let response_topic = format!("resp:{:x}:", response_channel_id);
    let response_topic_bytes = response_topic.as_bytes();

    // Set receive timeout for responsive polling (10ms for better latency)
    let _ = socket.set_opt::<nng::options::RecvTimeout>(Some(Duration::from_millis(10)));

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
                            let mut requests = active_requests
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            if let Some(entry) = requests.get_mut(&request_id) {
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
                }
                // Check if it's an engine event
                else if data.starts_with(EVENT_TOPIC_PREFIX) {
                    handle_engine_event(data, &event_callback);
                }
            }
            Err(nng::Error::TimedOut) => {
                // Normal timeout, continue polling
                continue;
            }
            Err(_) => {
                // Other error, check if we should stop
                if should_stop.load(Ordering::SeqCst) {
                    break;
                }
            }
        }
    }

    // Graceful shutdown: notify any remaining pending requests
    log::info!("IPC listener shutting down");
    let requests = active_requests
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    if !requests.is_empty() {
        log::warn!(
            "IPC listener exiting with {} active requests; failing them.",
            requests.len()
        );

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
}

/// Handle an engine event (telemetry, model_loaded, etc.)
fn handle_engine_event(data: &[u8], event_callback: &Option<EventCallback>) {
    // Event format: __PIE_EVENT__:<event_name>\x00<json_body>
    let parts: Vec<&[u8]> = data.splitn(2, |&b| b == 0).collect();
    if parts.len() != 2 {
        log::warn!("Received malformed event message");
        return;
    }

    let (topic_part, json_body) = (parts[0], parts[1]);

    // Extract event name from topic: "__PIE_EVENT__:<event_name>"
    let event_name = if topic_part.len() > EVENT_TOPIC_PREFIX.len() {
        String::from_utf8_lossy(&topic_part[EVENT_TOPIC_PREFIX.len()..]).to_string()
    } else {
        log::warn!("Event message has empty event name");
        return;
    };

    // Parse JSON payload
    let payload: Value = match serde_json::from_slice(json_body) {
        Ok(v) => v,
        Err(e) => {
            log::error!("Failed to parse engine event payload: {}", e);
            return;
        }
    };

    log::debug!("Received engine event: {}", event_name);

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
    if channel_id == 0 { 1 } else { channel_id }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            "top_logprobs": [{"hello": -0.5}, {"world": -1.0}],
            "cumulative_logprob": -1.5,
            "modal_decoder_id": "moondream3.coord",
            "modal_bytes_b64": "AAAA"
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
    }
}
