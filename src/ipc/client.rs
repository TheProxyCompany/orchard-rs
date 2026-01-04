//! High-performance IPC client for communicating with PIE.
//!
//! Uses NNG sockets with a dedicated listener thread for response handling.

use crate::error::{Error, Result};
use crate::ipc::endpoints::{management_url, request_url, response_url, EVENT_TOPIC_PREFIX};
use crate::ipc::serialization::{build_request_payload, build_batch_request_payload, PromptPayload, RequestType};

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

/// Response delta from PIE.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl ResponseDelta {
    /// Parse from JSON value.
    pub fn from_json(json: &Value) -> Self {
        let tokens = json["tokens"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|n| n as i32))
                    .collect()
            })
            .unwrap_or_default();

        let top_logprobs = json["top_logprobs"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        v.as_object().map(|obj| {
                            obj.iter()
                                .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                                .collect()
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Self {
            request_id: json["request_id"].as_u64().unwrap_or(0),
            sequence_id: json["sequence_id"].as_u64(),
            prompt_index: json["prompt_index"].as_u64().map(|v| v as u32),
            candidate_index: json["candidate_index"].as_u64().map(|v| v as u32),
            content: json["content"].as_str().map(String::from),
            content_len: json["content_len"].as_u64().map(|v| v as u32),
            inline_content_bytes: json["inline_content_bytes"].as_u64().map(|v| v as u32),
            is_final_delta: json["is_final_delta"].as_bool().unwrap_or(false),
            finish_reason: json["finish_reason"].as_str().map(String::from),
            error: json["error"].as_str().map(String::from),
            prompt_token_count: json["prompt_token_count"].as_u64().map(|v| v as u32),
            num_tokens_in_delta: json["num_tokens_in_delta"].as_u64().map(|v| v as u32),
            generation_len: json["generation_len"].as_u64().map(|v| v as u32),
            tokens,
            top_logprobs,
            cumulative_logprob: json["cumulative_logprob"].as_f64(),
            modal_decoder_id: json["modal_decoder_id"].as_str().map(String::from),
            modal_bytes_b64: json["modal_bytes_b64"].as_str().map(String::from),
        }
    }
}

/// Request options for inference.
#[derive(Debug, Clone)]
pub struct RequestOptions {
    /// Maximum tokens to generate
    pub max_tokens: i32,
    /// Sampling temperature
    pub temperature: f64,
    /// Top-p (nucleus) sampling
    pub top_p: f64,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

impl Default for RequestOptions {
    fn default() -> Self {
        Self {
            max_tokens: 0, // 0 means no limit
            temperature: 1.0,
            top_p: 1.0,
            stop_sequences: Vec::new(),
        }
    }
}

/// High-performance IPC client for communicating with PIE.
///
/// Uses a lock-based design instead of actors to minimize overhead in the hot path.
/// All socket operations are thread-safe via internal locks.
pub struct IPCClient {
    /// Push socket for requests
    request_socket: Option<Socket>,
    /// Sub socket for responses
    response_socket: Option<Socket>,
    /// Req socket for management
    management_socket: Option<Socket>,

    /// Unique channel ID for this client
    response_channel_id: u64,
    /// Request ID counter
    request_id_counter: AtomicU64,

    /// Active request senders - protected by mutex
    active_requests: Arc<Mutex<HashMap<u64, mpsc::UnboundedSender<ResponseDelta>>>>,

    /// Listener thread handle
    listener_handle: Option<JoinHandle<()>>,
    /// Signal to stop listener
    should_stop: Arc<AtomicBool>,
}

impl IPCClient {
    /// Create a new IPC client (not connected).
    pub fn new() -> Self {
        Self {
            request_socket: None,
            response_socket: None,
            management_socket: None,
            response_channel_id: rand_u64(),
            request_id_counter: AtomicU64::new(0),
            active_requests: Arc::new(Mutex::new(HashMap::new())),
            listener_handle: None,
            should_stop: Arc::new(AtomicBool::new(false)),
        }
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
        self.management_socket = Some(management_socket);

        // Start listener thread
        self.should_stop.store(false, Ordering::SeqCst);
        self.start_listener();

        Ok(())
    }

    /// Disconnect from PIE.
    pub fn disconnect(&mut self) {
        self.should_stop.store(true, Ordering::SeqCst);

        if let Some(handle) = self.listener_handle.take() {
            let _ = handle.join();
        }

        self.request_socket = None;
        self.response_socket = None;
        self.management_socket = None;

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

    /// Send an inference request and receive streaming responses.
    pub fn send_request(
        &self,
        request_id: u64,
        model_id: &str,
        model_path: &str,
        prompt: &str,
        options: RequestOptions,
    ) -> Result<mpsc::UnboundedReceiver<ResponseDelta>> {
        let socket = self.request_socket.as_ref().ok_or(Error::NotConnected)?;

        // Build request payload
        let payload = build_request_payload(
            request_id,
            model_id,
            model_path,
            RequestType::Generation,
            self.response_channel_id,
            prompt,
            options.max_tokens,
            options.temperature,
            options.top_p,
            &options.stop_sequences,
        )?;

        let (tx, rx) = mpsc::unbounded_channel();

        self.active_requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(request_id, tx);

        let msg = nng::Message::from(payload.as_slice());
        socket.send(msg).map_err(|(_, e)| Error::Nng(e))?;

        Ok(rx)
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

        self.active_requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(request_id, tx);

        let msg = nng::Message::from(payload.as_slice());
        socket.send(msg).map_err(|(_, e)| Error::Nng(e))?;

        Ok((prompts.len(), rx))
    }

    /// Send a management command (e.g., load_model).
    /// This is synchronous and blocking - appropriate for setup operations.
    pub fn send_management_command(&self, command: &Value, timeout: Duration) -> Result<Value> {
        let socket = self.management_socket.as_ref().ok_or(Error::NotConnected)?;

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

        let handle = thread::Builder::new()
            .name("orchard-ipc-listener".to_string())
            .spawn(move || {
                if let Some(socket) = response_socket {
                    run_response_listener(
                        socket,
                        active_requests,
                        should_stop,
                        response_channel_id,
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
    active_requests: Arc<Mutex<HashMap<u64, mpsc::UnboundedSender<ResponseDelta>>>>,
    should_stop: Arc<AtomicBool>,
    response_channel_id: u64,
) {
    let response_topic = format!("resp:{:x}:", response_channel_id);
    let response_topic_bytes = response_topic.as_bytes();

    // Set receive timeout for polling
    let _ = socket.set_opt::<nng::options::RecvTimeout>(Some(Duration::from_millis(100)));

    while !should_stop.load(Ordering::SeqCst) {
        match socket.recv() {
            Ok(msg) => {
                let data = msg.as_slice();

                // Check if it's a response for us
                if data.starts_with(response_topic_bytes) {
                    let json_data = &data[response_topic_bytes.len()..];

                    if let Ok(json) = serde_json::from_slice::<Value>(json_data) {
                        let delta = ResponseDelta::from_json(&json);
                        let request_id = delta.request_id;
                        let is_final = delta.is_final_delta;

                        let sender = active_requests
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .get(&request_id)
                            .cloned();

                        if let Some(tx) = sender {
                            let _ = tx.send(delta);

                            if is_final {
                                active_requests
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .remove(&request_id);
                            }
                        }
                    }
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
    fn test_default_options() {
        let options = RequestOptions::default();
        assert_eq!(options.max_tokens, 0);
        assert_eq!(options.temperature, 1.0);
        assert_eq!(options.top_p, 1.0);
        assert!(options.stop_sequences.is_empty());
    }

    #[test]
    fn test_response_delta_from_json() {
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
        let delta = ResponseDelta::from_json(&json);
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
}
