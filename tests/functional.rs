//! Functional tests mirroring orchard-py/tests/functional/*.py.

#[path = "project/fixture.rs"]
mod fixture;

#[path = "functional/basic.rs"]
mod basic;
#[path = "functional/batching.rs"]
mod batching;
#[path = "functional/best_of.rs"]
mod best_of;
#[path = "functional/capabilities.rs"]
mod capabilities;
#[path = "functional/client.rs"]
mod client;
#[path = "functional/determinism.rs"]
mod determinism;
#[path = "functional/logprobs.rs"]
mod logprobs;
#[path = "functional/multi_candidate.rs"]
mod multi_candidate;
#[path = "functional/multi_token.rs"]
mod multi_token;
#[path = "functional/multimodal.rs"]
mod multimodal;
#[path = "functional/responses_basic.rs"]
mod responses_basic;
#[path = "functional/responses_structured.rs"]
mod responses_structured;
#[path = "functional/responses_tools.rs"]
mod responses_tools;
#[path = "functional/stop_sequences.rs"]
mod stop_sequences;
#[path = "functional/structured_generation.rs"]
mod structured_generation;
#[path = "functional/unicode_payload.rs"]
mod unicode_payload;
