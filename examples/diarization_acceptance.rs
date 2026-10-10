//! Real eight-speaker Nemotron3 streaming acceptance on an annotated WAV.
//! No microphone capture or cloud inference. Keep the shared GPU lease outside
//! this process and use the exact PIE bundle through PIE_LOCAL_BUILD.
use orchard::diarization::{
    DiarizationDevice, DiarizationEvent, DiarizationOptions, DEFAULT_MODEL,
};
use orchard::{Client, InferenceEngine, ModelRegistry};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        return Err(
            "usage: diarization_acceptance INPUT.wav OUTPUT.jsonl [--metal] [--realtime]".into(),
        );
    }
    let realtime = args.iter().any(|s| s == "--realtime");
    let device = if args.iter().any(|s| s == "--metal") {
        DiarizationDevice::Metal
    } else {
        DiarizationDevice::Cpu
    };
    let mut wav = hound::WavReader::open(&args[1])?;
    let spec = wav.spec();
    if spec.channels != 1 || spec.bits_per_sample != 16 {
        return Err("Expected mono PCM16 WAV".into());
    }
    let pcm: Vec<f32> = wav
        .samples::<i16>()
        .map(|s| s.map(|s| f32::from(s) / 32768.0))
        .collect::<Result<_, _>>()?;
    let options = DiarizationOptions {
        device,
        sample_rate: spec.sample_rate,
        session_id: Some(format!("ami-native-{:016x}", rand::random::<u64>())),
        ..Default::default()
    };
    let frame_samples = options.frame_samples();
    let _engine = InferenceEngine::new().await?;
    let registry = Arc::new(ModelRegistry::new()?);
    let client = Client::connect(registry).await?;
    let mut session = client.diarization(DEFAULT_MODEL, options).await?;
    let control = session.control();
    let out = PathBuf::from(&args[2]);
    let receiver = tokio::spawn(async move {
        use std::io::Write;
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(&out).map_err(|e| e.to_string())?);
        let mut frames = 0;
        let mut errors = Vec::new();
        while let Some(event) = session.next_event().await {
            serde_json::to_writer(&mut file, &event).map_err(|e| e.to_string())?;
            writeln!(&mut file).map_err(|e| e.to_string())?;
            match event{
                DiarizationEvent::Ready{speakers,output_frame_seconds,..}=>{
                    assert_eq!(speakers,8);assert_eq!(output_frame_seconds,0.01);
                    eprintln!("READY {speakers} speakers at {output_frame_seconds}s");
                }
                DiarizationEvent::Update{probabilities,..}=>frames+=probabilities.len(),
                DiarizationEvent::Metrics{compute_ms_total,elapsed_ms,input_frames,missing_input_frames,..}=>
                    eprintln!("METRICS compute_ms={compute_ms_total:.2} elapsed_ms={elapsed_ms:.2} input_frames={input_frames} gaps={missing_input_frames}"),
                DiarizationEvent::Error{message}=>errors.push(message),
                _=>{},
            }
        }
        file.flush().map_err(|e| e.to_string())?;
        Ok::<_, String>((frames, errors))
    });
    let started = Instant::now();
    for (sequence, frame) in pcm.chunks(frame_samples).enumerate() {
        let mut input = frame.to_vec();
        input.resize(frame_samples, 0.0);
        loop {
            match control.push_audio(sequence as u64, input.clone()) {
                Ok(_) => break,
                Err(error) if error.to_string() == "Diarization input queue is full" => {
                    tokio::time::sleep(Duration::from_millis(2)).await
                }
                Err(error) => return Err(error.into()),
            }
        }
        if realtime {
            let target = started + Duration::from_millis((sequence as u64 + 1) * 80);
            tokio::time::sleep_until(target.into()).await;
        }
    }
    control.finish().await?;
    let (frames, errors) = tokio::time::timeout(Duration::from_secs(300), receiver).await???;
    println!(
        "frames={frames} audio_seconds={:.2} wall_seconds={:.2} errors={errors:?}",
        pcm.len() as f64 / spec.sample_rate as f64,
        started.elapsed().as_secs_f64()
    );
    if frames == 0 || !errors.is_empty() {
        return Err("Diarization acceptance failed".into());
    }
    Ok(())
}
