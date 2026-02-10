//! Orchard - Rust client for high-performance LLM inference on Apple Silicon.

mod defaults;
pub mod error;

pub mod client;
pub mod engine;
pub mod formatter;
pub mod ipc;
pub mod model;

pub use error::{Error, Result};

pub use ipc::client::{EventCallback, IPCClient, ResponseDelta, TokenLogProb};
pub use ipc::endpoints;
pub use ipc::serialization::{
    build_batch_request_payload, CapabilityEntry, LayoutEntry, PromptPayload, RequestType,
    SegmentType,
};

pub use engine::fetch::EngineFetcher;
pub use engine::lifecycle::{EnginePaths, InferenceEngine};
pub use engine::multiprocess;

pub use model::registry::{ModelEntry, ModelInfo, ModelLoadState, ModelRegistry};
pub use model::resolver::{ModelResolver, ResolvedModel};

pub use formatter::control_tokens::{ControlTokens, Role, RoleTags};
pub use formatter::multimodal::{
    build_multimodal_layout, build_multimodal_messages, CapabilityInput, ContentType, LayoutSegment,
};
pub use formatter::ChatFormatter;

pub use client::{
    BatchChatResult, ChatResult, Client, ClientDelta, ClientResponse, ContentPartAddedEvent,
    ContentPartDoneEvent, FunctionCallArgumentsDeltaEvent, FunctionCallArgumentsDoneEvent,
    IncompleteDetails, InputTokensDetails, OutputFunctionCall, OutputItemAddedEvent,
    OutputItemDoneEvent, OutputMessage, OutputReasoning, OutputStatus, OutputTextContent,
    OutputTextDeltaEvent, OutputTextDoneEvent, OutputTokensDetails, ReasoningContent,
    ReasoningDeltaEvent, ReasoningDoneEvent, ReasoningSummaryTextContent,
    ReasoningSummaryTextDeltaEvent, ReasoningSummaryTextDoneEvent, ResponseCompletedEvent,
    ResponseCreatedEvent, ResponseError, ResponseEvent, ResponseFailedEvent,
    ResponseInProgressEvent, ResponseIncompleteEvent, ResponseInputItem, ResponseObject,
    ResponseOutputItem, ResponseSnapshot, ResponseUsage, ResponsesInput, ResponsesRequest,
    ResponsesResult, SamplingParams, StreamErrorDetail, StreamErrorEvent, UsageStats,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
