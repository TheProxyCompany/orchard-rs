//! Thin JSON-lines transport for native PIE duplex sessions.
//! Model tensors and inference live exclusively in the PIE process.
use orchard::duplex::{self, DuplexDevice, DuplexEvent, DuplexOptions};
use orchard::{Client, InferenceEngine, ModelLoadOptions, ModelLoadRequest, ModelRegistry};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::sync::Arc;

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Command {
    Audio {
        sequence: u64,
        pcm: Vec<f32>,
    },
    Speak {
        text: String,
        #[serde(default)]
        replace: bool,
    },
    Reference {
        text: String,
        expected_epoch: u64,
    },
    GroundedReply {
        text: String,
        expected_epoch: u64,
    },
    HoldResponse {
        expected_epoch: u64,
    },
    Interrupt,
    Reset,
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
    let _ = write_json(&DuplexEvent::Error {
        message: message.to_string(),
    });
}

async fn run() -> orchard::Result<()> {
    let mut model = duplex::DEFAULT_MODEL.to_owned();
    let mut options = DuplexOptions::default();
    let mut prepare_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--prepare-only" => prepare_only = true,
            "--model" => {
                model = args.next().ok_or_else(|| {
                    orchard::Error::Other("--model requires an identifier or directory".into())
                })?
            }
            "--cpu" => options.device = DuplexDevice::Cpu,
            "--mimi-metal" => options.mimi_cpu = false,
            "--autonomous" => options.autonomous = true,
            "--offline" => options.realtime = false,
            "--options-json" => {
                options =
                    serde_json::from_str(&args.next().ok_or_else(|| {
                        orchard::Error::Other("--options-json requires JSON".into())
                    })?)?
            }
            "--capabilities" => {
                let profile = duplex::architecture()?;
                write_json(&json!({"architecture":"moshi","backend":"pie","duplex":profile}))?;
                return Ok(());
            }
            "--help" | "-h" => {
                println!("orchard-duplex [--model ID|DIRECTORY] [--cpu] [--mimi-metal] [--autonomous] [--offline] [--options-json JSON] [--prepare-only]\n\nReads audio/speak/reference/hold_response/grounded_reply/interrupt/reset/close JSON lines from stdin and emits duplex JSON events.\nAudio is 1920 mono float32 samples per frame at 24000 Hz.\nDefault speech is native autonomous duplex; references provide context only; hold_response waits for context; grounded_reply asks for a model-authored response, not a scripted reading.\nModel assets are the exact revision declared by the bundled Pantheon profile.\n--capabilities lists architecture and available checkpoints without loading weights.");
                return Ok(());
            }
            _ => return Err(orchard::Error::Other(format!("Unknown argument {arg}"))),
        }
    }
    let _engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(Arc::clone(&registry)).await?;
    if prepare_only {
        let request = ModelLoadRequest {
            model: model.clone(),
            options: ModelLoadOptions::Duplex(options),
        };
        let info = registry.ensure_prepared(&request).await?;
        let operations = registry.model_operations(&request).await?;
        write_json(
            &json!({"type":"prepared", "model_id":model, "runtime_model_id":info.model_id, "model_path":info.model_path, "operations":operations}),
        )?;
        return Ok(());
    }
    let mut session = client.duplex(&model, options).await?;
    let control = session.control();
    let supports_grounded_response = control.supports_grounded_response();
    let supports_response_hold = control.supports_response_hold();
    std::thread::Builder::new().name("orchard-duplex-input".into()).spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let line = match line { Ok(line) => line, Err(e) => { error(e); break; } };
            if line.len() > 1_048_576 { error("Duplex command exceeds 1 MiB"); continue; }
            if line.trim().is_empty() { continue; }
            let request: Request = match serde_json::from_str(&line) { Ok(request) => request, Err(e) => { error(e); continue; } };
            let id = request.id;
            let result: orchard::Result<Option<Value>> = match request.command {
                Command::Audio { sequence, pcm } => control.push_audio(sequence, pcm)
                    .map(|admission| Some(json!({"type":"audio_ack","id":id,"sequence":sequence,"epoch":admission.epoch,"dropped_frames":admission.dropped_frames}))),
                Command::Speak { text, replace } => control.speak(text, replace)
                    .map(|epoch| Some(json!({"type":"control_ack","id":id,"action":"speak","epoch":epoch}))),
                Command::Reference { text, expected_epoch } => control.reference(text, expected_epoch)
                    .map(|version| Some(json!({"type":"control_ack","id":id,"action":"reference","epoch":expected_epoch,"version":version}))),
                Command::GroundedReply { text, expected_epoch } => control.grounded_reply(text, expected_epoch)
                    .map(|version| Some(json!({"type":"control_ack","id":id,"action":"grounded_reply","epoch":expected_epoch,"version":version,"playback_confirmed":false}))),
                Command::HoldResponse { expected_epoch } => control.hold_response(expected_epoch)
                    .map(|version| Some(json!({"type":"control_ack","id":id,"action":"hold_response","epoch":expected_epoch,"version":version}))),
                Command::Interrupt => control.interrupt()
                    .map(|epoch| Some(json!({"type":"control_ack","id":id,"action":"interrupt","epoch":epoch}))),
                Command::Reset => control.reset()
                    .map(|epoch| Some(json!({"type":"control_ack","id":id,"action":"reset","epoch":epoch}))),
                Command::Close => { let _ = control.close(); break; },
            };
            match result {
                Ok(Some(value)) => { if write_json(&value).is_err() { break; } },
                Ok(None) => {},
                Err(e) => { let _ = write_json(&json!({"type":"request_error","id":id,"message":e.to_string()})); },
            }
        }
        let _ = control.close();
    })?;
    let mut failed = false;
    while let Some(event) = session.next_event().await {
        failed |= matches!(event, DuplexEvent::Error { .. });
        if matches!(event, DuplexEvent::Ready { .. }) {
            let mut ready = serde_json::to_value(&event)?;
            ready["supports_grounded_response"] = json!(supports_grounded_response);
            ready["supports_response_hold"] = json!(supports_response_hold);
            write_json(&ready)?;
        } else {
            write_json(&event)?;
        }
    }
    if failed {
        return Err(orchard::Error::Other("Duplex inference failed".into()));
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
