//! Buckshot: every functional and golden test against one engine in one
//! continuous volley, mirroring orchard-py/tests/test_buckshot.py.
//!
//! `--test functional` / `--test golden` each boot their own engine, so run
//! sequentially their walls add. This target compiles both suites into one
//! binary sharing one fixture; libtest's default parallelism plus the
//! uncapped per-test fanout turns the whole matrix into a single volley.
//! Run with: cargo test --release --test buckshot

#[path = "project/fixture.rs"]
mod fixture;
#[path = "golden/golden_io.rs"]
mod golden_io;

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

#[path = "golden/golden_path.rs"]
mod golden_path;

#[ctor::ctor]
fn preload_modal_models() {
    fixture::PRELOAD_MODAL_MODELS.store(true, std::sync::atomic::Ordering::Relaxed);
}
