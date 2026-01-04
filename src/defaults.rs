//! Default values for sampling and generation parameters.

pub const MAX_TOKENS: i32 = 1024;
pub const TEMPERATURE: f64 = 1.0;
pub const TOP_P: f64 = 1.0;
pub const TOP_K: i32 = -1;
pub const REPETITION_PENALTY: f64 = 1.0;
pub const REPETITION_CONTEXT_SIZE: i32 = 60;
pub const NUM_CANDIDATES: i32 = 1;

pub fn max_tokens() -> i32 { MAX_TOKENS }
pub fn temperature() -> f64 { TEMPERATURE }
pub fn top_p() -> f64 { TOP_P }
pub fn top_k() -> i32 { TOP_K }
pub fn repetition_penalty() -> f64 { REPETITION_PENALTY }
pub fn repetition_context_size() -> i32 { REPETITION_CONTEXT_SIZE }
pub fn num_candidates() -> i32 { NUM_CANDIDATES }
