"""Real Moshi/Mimi acceptance; manually run under the shared T1 GPU lease.

Needs the release orchard-duplex and a known, spoken 24 kHz mono WAV. Never uses
microphone hardware or a cloud model. Writes PCM and measured receipts.
"""
import argparse
import asyncio
import array
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import wave

async def run(args):
    with wave.open(args.input, "rb") as source:
        assert source.getframerate() == 24000 and source.getnchannels() == 1 and source.getsampwidth() == 2
        samples = array.array("h", source.readframes(source.getnframes()))
    pcm = [sample / 32768.0 for sample in samples]
    options = {"autonomous": args.autonomous, "mimi_cpu": not args.mimi_metal, "max_pending_frames": 6}
    process = await asyncio.create_subprocess_exec(
        args.binary, "--model", args.model, "--options-json", json.dumps(options),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        limit=1048576,
    )
    ready = asyncio.Event()
    audio = []
    measurements = []
    texts = []
    events = []
    failures = []
    epochs = {}
    ready_time = None
    async def read():
        nonlocal ready_time
        async for line in process.stdout:
            event = json.loads(line)
            kind = event.get("type")
            if kind == "ready":
                ready_time = time.monotonic();ready.set()
                print("READY", json.dumps(event), flush=True)
            if kind == "audio":
                epoch=event["epoch"]
                epochs.setdefault(epoch, []).extend(event["pcm"])
                audio.extend(event["pcm"])
                measurements.append({k: event[k] for k in ["epoch", "sequence", "compute_ms", "queue_ms"]})
            elif kind == "text_delta":texts.append(event)
            elif kind not in ["audio_ack"]:events.append(event)
            if kind in ["error", "request_error"]:
                failures.append(event);print("ERROR",json.dumps(event),flush=True)
        ready.set()
    reader=asyncio.create_task(read())
    errreader=asyncio.create_task(process.stderr.read())
    async def send(command):
        process.stdin.write((json.dumps(command)+"\n").encode());await process.stdin.drain()
    try:
        await asyncio.wait_for(ready.wait(),300)
        if ready_time is None:raise RuntimeError("Model failed before readiness")
        if not args.autonomous:
            # First one second must stay silent while input and temporal state advance.
            await asyncio.sleep(1)
            await send({"type":"speak","text":"Hello Jack. The local voice is ready.","replace":True})
        start=time.monotonic()
        for index in range(math.ceil(args.seconds/0.08)):
            frame=pcm[index*1920:(index+1)*1920]
            frame += [0.0]*(1920-len(frame))
            await send({"type":"audio","sequence":index,"pcm":frame})
            if not args.autonomous and index == round(args.seconds/0.16):
                await send({"type":"interrupt"})
                await send({"type":"speak","text":"Telescope. The microphone is still active.","replace":True})
            await asyncio.sleep(max(0,start+(index+1)*0.08-time.monotonic()))
        await send({"type":"close"})
        process.stdin.close()
        await asyncio.wait_for(process.wait(),20)
        await reader
    finally:
        if process.returncode is None:
            process.kill();await process.wait()
        await reader
    stderr=(await errreader).decode(errors="replace")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    def write_wav(path,values):
        with wave.open(str(path),"wb") as target:
            target.setnchannels(1);target.setsampwidth(2);target.setframerate(24000)
            raw=array.array("h",[max(-32768,min(32767,round(x*32767))) for x in values]);target.writeframes(raw.tobytes())
    write_wav(out/'speech.wav',audio)
    for epoch,values in epochs.items():write_wav(out/f'epoch-{epoch}.wav',values)
    compute=[x['compute_ms'] for x in measurements]
    report={"model":args.model,"input_wav":args.input,"options":options,"returncode":process.returncode,
            "output_frames":len(measurements),"output_samples":len(audio),"nonzero_output_samples":sum(abs(x)>0.001 for x in audio),
            "compute_ms_median":statistics.median(compute) if compute else None,"compute_ms_max":max(compute,default=0),
            "text":texts,"events":events,"errors":failures,"stderr":stderr,
            "epoch_audio_samples":{str(k):len(v) for k,v in epochs.items()},
            "idle_epoch_max_abs":max((abs(x) for x in epochs.get(0,[])),default=0),"measurements":measurements}
    (out/'receipt.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['measurements','stderr','events']},indent=2))
    if failures or process.returncode or not compute or not any(abs(x)>0.001 for x in audio):raise SystemExit(1)
    if not args.autonomous and report['idle_epoch_max_abs'] != 0:raise SystemExit('Controlled idle leaked speech')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',required=True);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--model',default='kyutai/moshiko-candle-q8');parser.add_argument('--seconds',type=float,default=16)
    parser.add_argument('--autonomous',action='store_true');parser.add_argument('--mimi-metal',action='store_true')
    args=parser.parse_args()
    with open('/tmp/proxy-gpu.lease','a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lease.seek(0);lease.truncate();lease.write(f'pid={os.getpid()} orchard native duplex acceptance {time.strftime("%Y-%m-%dT%H:%M:%S")}\n');lease.flush()
        asyncio.run(run(args))
