"""Real MoshiRAG reference grounding and barge-in receipt (no microphone)."""

import argparse
import asyncio
import array
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import time
import wave


def read_pcm(path):
    with wave.open(str(path), "rb") as source:
        assert (
            source.getframerate() == 24000
            and source.getnchannels() == 1
            and source.getsampwidth() == 2
        )
        return [
            x / 32768.0
            for x in array.array("h", source.readframes(source.getnframes()))
        ]


def write_wav(path, values):
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24000)
        target.writeframes(
            array.array(
                "h", [max(-32768, min(32767, round(x * 32767))) for x in values]
            ).tobytes()
        )


async def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    options = {"mimi_cpu": not args.mimi_metal, "max_pending_frames": 6}
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        args.binary,
        "--model",
        args.model,
        "--options-json",
        json.dumps(options),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1048576,
    )
    ready = asyncio.Event()
    epochs = {}
    events = []
    measurements = []
    errors = []
    ready_time = None

    async def read():
        nonlocal ready_time
        async for line in process.stdout:
            event = json.loads(line)
            event["observed_seconds"] = time.monotonic() - started
            kind = event.get("type")
            if kind == "audio":
                epochs.setdefault(event["epoch"], []).extend(event.pop("pcm"))
                measurements.append(event)
            elif kind != "audio_ack":
                events.append(event)
                if kind not in ["loading", "text_delta"]:
                    print(json.dumps(event), flush=True)
            if kind == "ready":
                ready_time = time.monotonic()
                ready.set()
            if kind in ["error", "request_error"]:
                errors.append(event)
        ready.set()

    reader = asyncio.create_task(read())

    async def stderr():
        with (out / "stderr.log").open("wb") as target:
            async for line in process.stderr:
                target.write(line)
                target.flush()

    errreader = asyncio.create_task(stderr())

    async def send(command):
        process.stdin.write((json.dumps(command) + "\n").encode())
        await process.stdin.drain()

    try:
        await asyncio.wait_for(ready.wait(), 300)
        if ready_time is None:
            raise RuntimeError("Model failed before readiness")
        first = read_pcm(args.first)
        second = read_pcm(args.second)
        begin = time.monotonic()
        for i in range(math.ceil(args.seconds / 0.08)):
            t = i * 0.08
            offset = round(
                (t - (args.seconds / 2 if t >= args.seconds / 2 else 0)) * 24000
            )
            values = second if t >= args.seconds / 2 else first
            frame = values[offset : offset + 1920]
            frame += [0.0] * (1920 - len(frame))
            if i == round(args.seconds / 0.16):
                await send({"type": "interrupt"})
            await send({"type": "audio", "sequence": i, "pcm": frame})
            if i == round(len(first) / 1920):
                await send(
                    {
                        "type": "speak",
                        "text": "The one-word answer to this test is telescope. There is no other information about the test.",
                        "replace": True,
                    }
                )
            if i == round(args.seconds / 0.16) + round(len(second) / 1920):
                await send(
                    {
                        "type": "speak",
                        "text": "The flag in this test is plain purple. It is not a national flag, and no country is associated with it.",
                        "replace": True,
                    }
                )
            await asyncio.sleep(max(0, begin + (i + 1) * 0.08 - time.monotonic()))
        await send({"type": "close"})
        process.stdin.close()
        await asyncio.wait_for(process.wait(), 30)
        await reader
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await reader
        await errreader
        for epoch, values in epochs.items():
            write_wav(out / f"epoch-{epoch}.wav", values)
        compute = [x["compute_ms"] for x in measurements]
        report = {
            "model": args.model,
            "options": options,
            "load_seconds": None if ready_time is None else ready_time - started,
            "returncode": process.returncode,
            "events": events,
            "measurements": measurements,
            "errors": errors,
            "epoch_audio_samples": {k: len(v) for k, v in epochs.items()},
            "compute_ms_median": statistics.median(compute) if compute else None,
            "compute_ms_max": max(compute, default=0),
        }
        (out / "receipt.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ["events", "measurements"]
                },
                indent=2,
            ),
            flush=True,
        )
    if errors or process.returncode or not measurements:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="kyutai/moshika-rag-candle-bf16")
    parser.add_argument("--mimi-metal", action="store_true")
    parser.add_argument("--seconds", type=float, default=28)
    args = parser.parse_args()
    with open("/tmp/proxy-gpu.lease", "a+") as lease:
        print("Waiting for shared GPU lease", flush=True)
        fcntl.flock(lease, fcntl.LOCK_EX)
        lease.seek(0)
        lease.truncate()
        lease.write(
            f"pid={os.getpid()} orchard MoshiRAG grounding {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        )
        lease.flush()
        print("Acquired shared GPU lease", flush=True)
        asyncio.run(run(args))
