# Incremental Qwen audio

The opt-in `agenerate_audio_streaming_cancellable` API delivers native audio artifacts through a bounded caller channel while synthesis runs. The existing complete-WAV APIs retain their behavior.

The method requires `qwen3_tts_streaming: [1]` from the running engine's model-load receipt. `ModelOperations` exposes `text_to_speech_streaming` only for that native capability; a static profile cannot establish engine support. Each artifact is limited to 1 MiB, the response to 32 MiB, the queue to eight artifacts and generation to 8192 codec tokens. GC uses a two-artifact queue and a stricter clause/token budget.

A completed audio artifact is separate from native request retirement. Success requires the engine's terminal response. Cancellation, consumer closure, or artifact parsing failure stops forwarding, requests cancellation with a two-second outer bound, and waits up to five seconds for a terminal response. An absent terminal remains `CancellationUnconfirmed`. The caller must await the producer after rejecting an artifact; dropping its future is not cancellation acknowledgement.

## Compatibility

This requires PIE revision [`b7b75ae3d484bd27ae1a335cbd5c24e7f4461ab2`](https://github.com/TheProxyCompany/proxy-inference-engine/commit/b7b75ae3d484bd27ae1a335cbd5c24e7f4461ab2) or a compatible engine that advertises the same native marker. The tested engine SHA-256 is `d824c85c5d37c1d7667cbac4e2d8a6ac4f4a719794e1156d60a86089a36e6b92`. Older engines fail before a stream request is sent. Orchard's existing engine selection remains local override, application bundle, existing cache, then stable-manifest download only when absent. This library change does not update a shared or bundled engine automatically; the application must package the compatible engine through its normal release path.

## Historical native evidence

The source compiled with `cargo +1.97.1 check --lib --features duplex,diarization`, and independent review checked cancellation, consumer failure and native capability gating. The companion [PIE evidence packet](https://github.com/TheProxyCompany/proxy-inference-engine/blob/b7b75ae3d484bd27ae1a335cbd5c24e7f4461ab2/docs/reviews/qwen-streaming-evidence.json) records the actual model/IPC proof: legacy full WAV, incremental chunks, empty final tail, cancellation, a different next request, and invalid-option rejection all passed. Streamed and full output each contain 157,440 samples; the maximum difference is one PCM16 unit, RMS error 0.0153 units, and each checked boundary window matches exactly. The next request after cancellation matches its fresh same-seed counterpart byte for byte.

Independent Parakeet ASR preserves “Morgan,” “Do not send,” and the amount $15.50. The first chunk arrived after 175 ms in the already-loaded renderer. This is not speech-end-to-reply, GC, iPhone, physical playback or preferred-timbre acceptance. These component tests use Orchard Python and native IPC; they do not substitute for the pending GC → new Rust streaming API integration run. The broader PIE client cycle was attempted but could not pass because an unrelated Llama model was unavailable. The [local proof index](incremental-qwen-evidence.json) records exact receipt hashes without embedding audio or personal data.

## October 23 rebase verification

Production source `8a5cacd35f87860ae292a05af58879db84c8ae3e` rebases the four PR commits onto main `7e2c98af18c168ba5e7e3d054662eaca27070087`. The range diff is unchanged and the rebase had no conflicts. Upstream CI's engine revision, fixture isolation and failed-initialization cleanup, Cargo manifests, and the committed profiles revision remain intact. No test was deleted, ignored, or weakened.

`cargo fmt --check`, `cargo test --locked --all-features --no-run`, and `cargo clippy --locked --all-targets --all-features -- -D warnings` passed. The existing `diarization_ipc` contract passed with all features (one test, zero failures or ignored tests). Library and documentation targets each contain zero tests; their successful exit is not additional behavior coverage. The evidence JSON records exact commands and limitations.

The complete native model suite and the new Rust streaming API's live integration remain unverified at this revision. No engine was launched, replaced, or supplied through an executable override for these checks. The normal app-bundle path can select a reviewed PR73 engine, but a remote Orchard Cargo revision alone does not bundle or download that engine. Verify the packaged engine revision and its `qwen3_tts_streaming: [1]` receipt before claiming compatibility or latency. The older native timings above are historical, not results for this rebase.
