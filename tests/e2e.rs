//! End-to-end tests mirroring orchard-py/tests/test_e2e_*.py
//!
//! Run with: cargo test --test e2e -- --ignored

#[path = "e2e/fixture.rs"]
mod fixture;

#[path = "e2e/e2e_basic.rs"]
mod e2e_basic;
#[path = "e2e/e2e_batching.rs"]
mod e2e_batching;
#[path = "e2e/e2e_best_of.rs"]
mod e2e_best_of;
#[path = "e2e/e2e_capabilities.rs"]
mod e2e_capabilities;
#[path = "e2e/e2e_client.rs"]
mod e2e_client;
#[path = "e2e/e2e_determinism.rs"]
mod e2e_determinism;
#[path = "e2e/e2e_logprobs.rs"]
mod e2e_logprobs;
#[path = "e2e/e2e_multi_candidate.rs"]
mod e2e_multi_candidate;
#[path = "e2e/e2e_multi_token.rs"]
mod e2e_multi_token;
#[path = "e2e/e2e_multimodal.rs"]
mod e2e_multimodal;
#[path = "e2e/e2e_responses_basic.rs"]
mod e2e_responses_basic;
#[path = "e2e/e2e_responses_structured.rs"]
mod e2e_responses_structured;
#[path = "e2e/e2e_responses_tools.rs"]
mod e2e_responses_tools;
#[path = "e2e/e2e_stop_sequences.rs"]
mod e2e_stop_sequences;
#[path = "e2e/e2e_structured_generation.rs"]
mod e2e_structured_generation;
#[path = "e2e/e2e_unicode_payload.rs"]
mod e2e_unicode_payload;
