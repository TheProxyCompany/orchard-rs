//! Real resident Gemma + MoshiRAG + Moondream + Nemotron3 + Parakeet gate.
//! Uses local speech/image fixtures, never microphone/camera hardware or cloud inference.
mod common;
use base64::Engine;
use orchard::{
    client::MoondreamClient,
    diarization::{DiarizationEvent, DiarizationOptions},
    duplex::{DuplexEvent, DuplexOptions},
    Client, SamplingParams,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::Write,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
const BRAIN: &str = "google/gemma-4-E2B-it";
const VISION: &str = "moondream/moondream3-preview";
const VOICE: &str = "kyutai/moshika-rag-candle-bf16";
const ASR: &str = "mlx-community/parakeet-tdt-0.6b-v3";
fn wav(path: &Path, rate: u32) -> Result<Vec<f32>, common::Error> {
    let mut wav = hound::WavReader::open(path)?;
    let spec = wav.spec();
    if spec.sample_rate != rate || spec.channels != 1 || spec.bits_per_sample != 16 {
        return Err("Wrong fixture PCM format".into());
    }
    Ok(wav
        .samples::<i16>()
        .map(|s| s.map(|x| x as f32 / 32768.))
        .collect::<Result<_, _>>()?)
}
fn params() -> SamplingParams {
    SamplingParams {
        temperature: 0.,
        max_tokens: 96,
        reasoning: Some(false),
        prefix_cache: Some(true),
        ..Default::default()
    }
}
#[tokio::main]
async fn main() -> Result<(), common::Error> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: media_concurrency FIRST_24K.wav SECOND_24K.wav ASR_16K.wav OUTPUT_DIR".into(),
        );
    }
    let first = wav(Path::new(&args[1]), 24000)?;
    let second = wav(Path::new(&args[2]), 24000)?;
    let asr_pcm = wav(Path::new(&args[3]), 16000)?;
    let output = Path::new(&args[4]);
    std::fs::create_dir_all(output)?;
    let (_engine, registry, client) = common::connect().await?;
    for model in [BRAIN, VISION, ASR] {
        let t = Instant::now();
        let info =
            tokio::time::timeout(Duration::from_secs(300), registry.ensure_loaded(model)).await??;
        println!(
            "MODEL {}",
            json!({"id":model,"path":info.model_path,"load_ms":t.elapsed().as_secs_f64()*1000.})
        );
    }
    let vision = MoondreamClient::with_model(
        Client::connect(Arc::clone(&registry)).await?,
        Arc::clone(&registry),
        VISION,
    )
    .await?;
    let image =
        std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/apple.jpg"))?;
    let image_url = format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(image)
    );
    let mut cold_params = params();
    cold_params.prefix_cache = Some(false);
    let warm_caption = tokio::time::timeout(
        Duration::from_secs(180),
        vision.caption_with_metrics(&image_url, "normal", cold_params),
    )
    .await??;
    println!("VISION_WARM {}", serde_json::to_string(&warm_caption)?);
    let warm_brain = common::run_turn(
        &client,
        BRAIN,
        vec![common::msg("user", "Reply only: ready")],
        params(),
        |_| {},
    )
    .await?;
    println!("BRAIN_WARM {}", json!({"text":warm_brain.text}));
    let warm_asr = client.atranscribe_audio(ASR, &asr_pcm).await?;
    println!("ASR_WARM {}", json!({"text":warm_asr}));
    let mut diar = registry
        .diarization(
            "nvidia/Nemotron-3-Diarization",
            DiarizationOptions::default(),
        )
        .await?;
    let dc = diar.control();
    let mut voice = registry
        .duplex(
            VOICE,
            DuplexOptions {
                realtime: false,
                ..Default::default()
            },
        )
        .await?;
    let vc = voice.control();
    let began = Instant::now();
    println!(
        "ALL_MODELS_READY {}",
        json!({"pid":std::process::id(),"models":[BRAIN,VISION,VOICE,"nvidia/Nemotron-3-Diarization",ASR]})
    );
    let feeding = async {
        for i in 0..400u64 {
            if i == 200 {
                vc.interrupt()?;
            }
            let values = if i < 200 { &first } else { &second };
            let offset = (i % 200) as usize * 1920;
            let mut frame = if offset < values.len() {
                values[offset..(offset + 1920).min(values.len())].to_vec()
            } else {
                vec![]
            };
            frame.resize(1920, 0.);
            vc.push_audio(i, frame.clone())?;
            dc.push_audio(i, frame)?;
            tokio::time::sleep_until((began + Duration::from_millis((i + 1) * 80)).into()).await;
        }
        vc.close()?;
        dc.finish().await?;
        Ok::<_, orchard::Error>(())
    };
    let thinking = async {
        let mut turns = vec![];
        for (at,question,fact) in [(3.,"What is the secret word for this test?","The one-word answer is telescope. Nothing else is known about this test."),(18.,"What color is the flag?","This test flag is plain purple. It is not a national flag and no country is associated with it.")] {
            tokio::time::sleep_until((began+Duration::from_secs_f64(at)).into()).await;
            let turn=common::run_turn(&client,BRAIN,vec![common::msg("system","Write a factual reference for a voice model. Use only the supplied facts, no additional claims. Fewer than forty words. No markdown."),common::msg("user",&format!("Question: {question}\nVerified local facts: {fact}"))],params(), |_|{}).await?;
            let reference=turn.text.clone();let epoch=vc.epoch();let version=vc.reference(reference.clone(),epoch)?;
            let receipt=json!({"epoch":epoch,"version":version,"reference":reference,"elapsed_ms":turn.elapsed.as_secs_f64()*1000.,"cached_tokens":turn.cached,"observed_seconds":began.elapsed().as_secs_f64()});
            println!("BRAIN {}",receipt);turns.push(receipt);
        }
        Ok::<_, common::Error>(turns)
    };
    let seeing = async {
        tokio::time::sleep_until((began + Duration::from_secs(6)).into()).await;
        let mut captions = vec![];
        for _ in 0..2 {
            let caption = vision
                .caption_with_metrics(&image_url, "normal", params())
                .await?;
            println!("VISION {}", serde_json::to_string(&caption)?);
            captions.push(caption);
        }
        if captions
            .iter()
            .any(|caption| caption.caption != warm_caption.caption)
        {
            return Err("Cached Moondream caption differed from the uncached baseline".into());
        }
        let points = vision.point(&image_url, "apple", params()).await?;
        println!("POINTS {}", serde_json::to_string(&points)?);
        if points.points.is_empty() {
            return Err("Moondream returned no apple point".into());
        }
        if captions.last().is_none_or(|c| c.cached_tokens == 0) {
            return Err("Repeated Moondream frame had no measured prefix hit".into());
        }
        Ok::<_, common::Error>((captions, points))
    };
    let listening = async {
        tokio::time::sleep_until((began + Duration::from_secs(9)).into()).await;
        let t = Instant::now();
        let text = client.atranscribe_audio(ASR, &asr_pcm).await?;
        println!(
            "ASR {}",
            json!({"text":text,"elapsed_ms":t.elapsed().as_secs_f64()*1000.})
        );
        Ok::<_, common::Error>(text)
    };
    let voice_events = async {
        let mut events =
            std::io::BufWriter::new(std::fs::File::create(output.join("voice.jsonl"))?);
        let mut audio: BTreeMap<u64, Vec<f32>> = BTreeMap::new();
        let mut errors = vec![];
        let mut metrics = None;
        while let Some(event) = voice.next_event().await {
            match &event {
                DuplexEvent::Audio {
                    epoch,
                    pcm,
                    sequence,
                    compute_ms,
                    queue_ms,
                } => {
                    audio.entry(*epoch).or_default().extend_from_slice(pcm);
                    writeln!(
                        events,
                        "{}",
                        json!({"type":"audio","epoch":epoch,"sequence":sequence,"compute_ms":compute_ms,"queue_ms":queue_ms,"observed_seconds":began.elapsed().as_secs_f64()})
                    )?;
                }
                _ => {
                    serde_json::to_writer(&mut events, &event)?;
                    writeln!(events)?;
                }
            }
            if let DuplexEvent::Error { message } = event {
                errors.push(message);
            } else if let DuplexEvent::Metrics { metrics: m } = event {
                metrics = Some(m);
            }
        }
        for (epoch, pcm) in audio {
            let mut wav = hound::WavWriter::create(
                output.join(format!("voice-epoch-{epoch}.wav")),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 24000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )?;
            for s in pcm {
                wav.write_sample((s.clamp(-1., 1.) * 32767.) as i16)?;
            }
            wav.finalize()?;
        }
        events.flush()?;
        Ok::<_, common::Error>((metrics, errors))
    };
    let diar_events = async {
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(output.join("diarization.jsonl"))?);
        let mut frames = 0;
        let mut errors = vec![];
        while let Some(event) = diar.next_event().await {
            serde_json::to_writer(&mut file, &event)?;
            writeln!(file)?;
            match event {
                DiarizationEvent::Update { probabilities, .. } => frames += probabilities.len(),
                DiarizationEvent::Error { message } => errors.push(message),
                _ => {}
            }
        }
        file.flush()?;
        Ok::<_, common::Error>((frames, errors))
    };
    let result = tokio::time::timeout(Duration::from_secs(90), async {
        tokio::join!(
            feeding,
            thinking,
            seeing,
            listening,
            voice_events,
            diar_events
        )
    })
    .await;
    let _ = vc.close();
    dc.close();
    let (feed, brain, vision, asr, voice, diar) = result?;
    feed?;
    let brain = brain?;
    let vision = vision?;
    let asr = asr?;
    let voice = voice?;
    let diar = diar?;
    let receipt = json!({"brain":brain,"vision":vision,"asr":asr,"voice_metrics":voice.0,"voice_errors":voice.1,"diarization_frames":diar.0,"diarization_errors":diar.1,"elapsed_seconds":began.elapsed().as_secs_f64()});
    std::fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("RECEIPT {}", receipt);
    if !voice.1.is_empty() || !diar.1.is_empty() || diar.0 == 0 {
        return Err("Concurrent media failed".into());
    }
    if voice
        .0
        .is_none_or(|m| m.dropped_input_frames > 0 || m.dropped_output_frames > 0)
    {
        return Err("Concurrent voice missed audio frames".into());
    }
    Ok(())
}
