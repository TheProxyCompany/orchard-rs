# Incremental Qwen audio

The opt-in `agenerate_audio_streaming_cancellable` API delivers native audio artifacts through a bounded caller channel while synthesis runs. The existing complete-WAV APIs retain their behavior.

The method requires `qwen3_tts_streaming: [1]` from the running engine's model-load receipt. `ModelOperations` exposes `text_to_speech_streaming` only for that native capability; a static profile cannot establish engine support. Each artifact is limited to 1 MiB, the response to 32 MiB, the queue to eight artifacts and generation to 8192 codec tokens. GC uses a two-artifact queue and a stricter clause/token budget.

A completed audio artifact is separate from native request retirement. Success requires the engine's terminal response. Cancellation, consumer closure, or artifact parsing failure stops forwarding, requests cancellation with a two-second outer bound, and waits up to five seconds for a terminal response. An absent terminal remains `CancellationUnconfirmed`. The caller must await the producer after rejecting an artifact; dropping its future is not cancellation acknowledgement.

## Compatibility

This requires the companion PIE incremental-Qwen implementation. Older engines fail before a stream request is sent. Orchard's existing engine selection remains local override, application bundle, existing cache, then stable-manifest download only when absent. This library change does not update a shared or bundled engine automatically. The final review must record the compatible PIE revision and measured runtime artifact.

## Evidence

The source compiled with `cargo +1.97.1 check --lib --features duplex,diarization`, and independent review checked cancellation, consumer failure and native capability gating. Actual PIE chunk continuity, legacy/full-WAV comparison, cancellation and spoken output evidence are pending the companion binary; compilation alone is not audio acceptance.
