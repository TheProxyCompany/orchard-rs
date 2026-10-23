//! The nng sockets Orchard speaks to PIE over.

use std::ffi::{CStr, CString};
use std::fmt;
use std::ops::Deref;
use std::ptr;
use std::sync::Once;
use std::time::Duration;

use nng_sys::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Push0,
    Pull0,
    Req0,
    Rep0,
    Pub0,
    Sub0,
}

/// An nng error code, as nng reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SocketError(nng_err);

impl SocketError {
    pub const TIMED_OUT: SocketError = SocketError(nng_err::NNG_ETIMEDOUT);
    pub const CLOSED: SocketError = SocketError(nng_err::NNG_ECLOSED);
}

impl fmt::Display for SocketError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = unsafe { CStr::from_ptr(nng_strerror(self.0)) };
        formatter.write_str(&text.to_string_lossy())
    }
}

impl std::error::Error for SocketError {}

fn check(code: std::ffi::c_int) -> Result<(), SocketError> {
    if code == 0 {
        Ok(())
    } else {
        Err(SocketError(nng_err(code as std::ffi::c_uint)))
    }
}

fn initialize() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        nng_init(ptr::null());
    });
}

fn url(text: &str) -> Result<CString, SocketError> {
    CString::new(text).map_err(|_| SocketError(nng_err::NNG_EADDRINVAL))
}

fn milliseconds(duration: Duration) -> nng_duration {
    duration.as_millis().min(i32::MAX as u128) as nng_duration
}

/// One nng socket. nng sockets are thread safe, so a `Socket` is `Send` and
/// `Sync`; it closes when dropped.
pub struct Socket(nng_socket);

unsafe impl Send for Socket {}
unsafe impl Sync for Socket {}

impl Socket {
    pub fn new(protocol: Protocol) -> Result<Self, SocketError> {
        initialize();
        let mut socket = nng_socket::default();
        let open = match protocol {
            Protocol::Push0 => nng_push0_open,
            Protocol::Pull0 => nng_pull0_open,
            Protocol::Req0 => nng_req0_open,
            Protocol::Rep0 => nng_rep0_open,
            Protocol::Pub0 => nng_pub0_open,
            Protocol::Sub0 => nng_sub0_open,
        };
        check(unsafe { open(&mut socket) })?;
        Ok(Socket(socket))
    }

    /// Connect, waiting for the first connection to succeed.
    pub fn dial(&self, address: &str) -> Result<(), SocketError> {
        let address = url(address)?;
        check(unsafe { nng_dial(self.0, address.as_ptr(), ptr::null_mut(), 0) })
    }

    /// Connect in the background; nng keeps retrying until the peer appears.
    pub fn dial_async(&self, address: &str) -> Result<(), SocketError> {
        let address = url(address)?;
        check(unsafe {
            nng_dial(
                self.0,
                address.as_ptr(),
                ptr::null_mut(),
                NNG_FLAG_NONBLOCK as std::ffi::c_int,
            )
        })
    }

    pub fn listen(&self, address: &str) -> Result<(), SocketError> {
        let address = url(address)?;
        check(unsafe { nng_listen(self.0, address.as_ptr(), ptr::null_mut(), 0) })
    }

    pub fn set_send_timeout(&self, timeout: Duration) -> Result<(), SocketError> {
        self.set_duration(NNG_OPT_SENDTIMEO, timeout)
    }

    pub fn set_receive_timeout(&self, timeout: Duration) -> Result<(), SocketError> {
        self.set_duration(NNG_OPT_RECVTIMEO, timeout)
    }

    /// How many messages nng queues for sending before `send` blocks.
    pub fn set_send_buffer(&self, messages: i32) -> Result<(), SocketError> {
        self.set_int(NNG_OPT_SENDBUF, messages)
    }

    /// How many received messages nng queues before dropping or blocking.
    pub fn set_receive_buffer(&self, messages: i32) -> Result<(), SocketError> {
        self.set_int(NNG_OPT_RECVBUF, messages)
    }

    /// The largest message nng accepts; larger ones are dropped.
    pub fn set_receive_max_size(&self, bytes: usize) -> Result<(), SocketError> {
        check(unsafe { nng_socket_set_size(self.0, NNG_OPT_RECVMAXSZ.as_ptr().cast(), bytes) })
    }

    /// Subscribe a Sub0 socket to every message starting with `topic`.
    pub fn subscribe(&self, topic: &[u8]) -> Result<(), SocketError> {
        check(unsafe { nng_sub0_socket_subscribe(self.0, topic.as_ptr().cast(), topic.len()) })
    }

    /// Send a copy of `data`, blocking up to the send timeout.
    pub fn send(&self, data: &[u8]) -> Result<(), SocketError> {
        check(unsafe { nng_send(self.0, data.as_ptr().cast(), data.len(), 0) })
    }

    /// Receive one message, blocking up to the receive timeout.
    pub fn receive(&self) -> Result<Message, SocketError> {
        let mut message: *mut nng_msg = ptr::null_mut();
        check(unsafe { nng_recvmsg(self.0, &mut message, 0) })?;
        Ok(Message(message))
    }

    /// Close now, waking every thread blocked in `send` or `receive` on this
    /// socket with `SocketError::CLOSED`. Dropping the socket closes it too.
    pub fn close(&self) {
        unsafe {
            nng_socket_close(self.0);
        }
    }

    fn set_duration(&self, option: &[u8], duration: Duration) -> Result<(), SocketError> {
        check(unsafe { nng_socket_set_ms(self.0, option.as_ptr().cast(), milliseconds(duration)) })
    }

    fn set_int(&self, option: &[u8], value: i32) -> Result<(), SocketError> {
        check(unsafe { nng_socket_set_int(self.0, option.as_ptr().cast(), value) })
    }
}

impl Drop for Socket {
    fn drop(&mut self) {
        self.close();
    }
}

/// A received message; its body reads as a byte slice.
pub struct Message(*mut nng_msg);

unsafe impl Send for Message {}

impl Message {
    pub fn as_slice(&self) -> &[u8] {
        let length = unsafe { nng_msg_len(self.0) };
        if length == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(nng_msg_body(self.0).cast::<u8>(), length) }
    }
}

impl Deref for Message {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Drop for Message {
    fn drop(&mut self) {
        unsafe {
            nng_msg_free(self.0);
        }
    }
}
