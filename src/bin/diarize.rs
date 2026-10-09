//! Thin JSON-lines transport for PIE-owned Nemotron3 sessions.
use orchard::diarization::{self, DiarizationDevice, DiarizationEvent, DiarizationOptions};
use orchard::{Client, InferenceEngine, ModelRegistry};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::sync::Arc;

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Command {
    Audio { sequence: u64, pcm: Vec<f32> },
    Finish,
    Close,
}
#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    id: Option<u64>,
    #[serde(flatten)]
    command: Command,
}
fn write_json(value: &impl serde::Serialize) -> std::io::Result<()> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer(&mut out, value)?;
    out.write_all(b"\n")?;
    out.flush()
}
fn error(message: impl std::fmt::Display) {
    let _ = write_json(&DiarizationEvent::Error {
        message: message.to_string(),
    });
}

async fn run() -> orchard::Result<()> {
    let mut model = diarization::DEFAULT_MODEL.to_owned();
    let mut options = DiarizationOptions::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model = args
                    .next()
                    .ok_or_else(|| orchard::Error::Other("--model requires a model id".into()))?
            }
            "--metal" => options.device = DiarizationDevice::Metal,
            "--cpu" => options.device = DiarizationDevice::Cpu,
            "--options-json" => {
                options =
                    serde_json::from_str(&args.next().ok_or_else(|| {
                        orchard::Error::Other("--options-json requires JSON".into())
                    })?)?
            }
            "--capabilities" => {
                write_json(
                    &json!({"architecture":"nemotron3_diarization","backend":"pie","diarization":diarization::architecture()?}),
                )?;
                return Ok(());
            }
            "--help" | "-h" => {
                println!("orchard-diarize [--model ID] [--metal|--cpu] [--options-json JSON]\n\nNative eight-speaker Nemotron3. Reads audio/finish/close JSON lines; emits timestamped probabilities and anonymous speaker segments.\nInput: mono float32, 80ms frames at sample_rate (24000Hz by default).\n--capabilities reads the shared Pantheon model descriptor without loading weights.");
                return Ok(());
            }
            _ => return Err(orchard::Error::Other(format!("Unknown argument {arg}"))),
        }
    }
    let _engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(registry).await?;
    let mut session = client.diarization(&model, options).await?;
    let control = session.control();
    let runtime = tokio::runtime::Handle::current();
    std::thread::Builder::new().name("orchard-diar-input".into()).spawn(move||{
        for line in std::io::stdin().lock().lines(){
            let line=match line{Ok(line)=>line,Err(e)=>{error(e);break;}};
            if line.len()>1_048_576{error("Diarization command exceeds1MiB");continue;}
            if line.trim().is_empty(){continue;}
            let request:Request=match serde_json::from_str(&line){Ok(v)=>v,Err(e)=>{error(e);continue;}};
            let id=request.id;
            let result:orchard::Result<Value>=match request.command{
                Command::Audio{sequence,pcm}=>control.push_audio(sequence,pcm)
                    .map(|v|json!({"type":"audio_ack","id":id,"sequence":sequence,"missing_frames":v.missing_frames})),
                Command::Finish=>runtime.block_on(control.finish())
                    .map(|_|json!({"type":"control_ack","id":id,"action":"finish"})),
                Command::Close=>{control.close();break;}
            };
            match result{
                Ok(v)=>{if write_json(&v).is_err(){break;}},
                Err(e)=>{let _=write_json(&json!({"type":"request_error","id":id,"message":e.to_string()}));}
            }
        }
        control.close();
    })?;
    let mut failed = false;
    while let Some(event) = session.next_event().await {
        failed |= matches!(event, DiarizationEvent::Error { .. });
        write_json(&event)?;
    }
    if failed {
        return Err(orchard::Error::Other("Native diarization failed".into()));
    }
    Ok(())
}
#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(e) = run().await {
        error(e);
        std::process::exit(1);
    }
}
