use std::collections::{HashMap, HashSet};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use orchard::{
    FunctionCallOutputContent, ModalArtifact, OutputFunctionCall, OutputMessage, OutputReasoning,
    OutputStatus, ReasoningConfig, ResponseInputItem, ResponseOutputItem, ResponsesInput,
    ResponsesRequest, ResponsesResult,
};
use serde_json::{json, Value};

use crate::fixture::{get_fixture, Model, Thinking, GEMMA4_MODEL_ID, MODELS, MOONDREAM_MODEL_ID};
use crate::golden_io::{assert_or_record, drain_stream, reasoning_tokens, Turn};

const WEATHER_SYSTEM: &str = "You are a helpful assistant with tool calling. Reason about the request, then call a tool when needed and use its result to answer.";
const IDEOGRAM4_MODEL_ID: &str = "ideogram-ai/ideogram-4-fp8";
const FLUX2_MODEL_ID: &str = "black-forest-labs/FLUX.2-klein-4B";
const QWEN_IMAGE_EDIT_MODEL_ID: &str = "Qwen/Qwen-Image-Edit";
const PARAKEET_MODEL_ID: &str = "mlx-community/parakeet-tdt-0.6b-v3";
const TTS_MODELS: [(&str, &str); 2] = [
    ("qwen3_tts_0_6b", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
    ("qwen3_tts_1_7b", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
];
const STT_MODELS: [(&str, &str); 3] = [
    ("parakeet", PARAKEET_MODEL_ID),
    ("qwen3_asr_0_6b", "Qwen/Qwen3-ASR-0.6B"),
    ("qwen3_asr_1_7b", "Qwen/Qwen3-ASR-1.7B"),
];
const AUDIO_TELEPHONE_PHRASES: [&str; 6] = [
    "hello this is a test",
    "the quick brown fox jumps over the lazy dog",
    "today we test local speech in a quiet room",
    "set the kitchen timer for tomorrow morning after breakfast",
    "proxy orchard handles audio images and text together",
    "blue square red circle green triangle",
];
const AUDIO_TELEPHONE_SAMPLE_RATE: usize = 16_000;
const SHAPES_IMAGE_USER: &str = "Use generate_image to create a simple flat icon on a plain white background: a red circle on the left and a blue square on the right. No text, no shadows.";
const SWAP_COLORS_PROMPT: &str = "Make the left circle bright blue. Make the right square bright red. Keep the background white.";

fn message(role: &str, content: &str) -> ResponseInputItem {
    ResponseInputItem::Message {
        role: role.to_string(),
        content: json!(content),
        tool_calls: None,
        tool_call_id: None,
    }
}

fn function_call(call_id: &str, name: &str, arguments: &str) -> ResponseInputItem {
    ResponseInputItem::FunctionCall {
        call_id: call_id.to_string(),
        name: name.to_string(),
        arguments: arguments.to_string(),
    }
}

fn function_output(call_id: &str, output: Value) -> ResponseInputItem {
    ResponseInputItem::FunctionCallOutput {
        call_id: call_id.to_string(),
        output: tool_output_json(&output).into(),
    }
}

fn tool_output_json(value: &Value) -> String {
    match value {
        Value::Array(values) => {
            let values = values
                .iter()
                .map(tool_output_json)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{values}]")
        }
        Value::Object(object) => {
            let fields = object
                .iter()
                .map(|(key, value)| {
                    let key = serde_json::to_string(key).expect("JSON object key serializes");
                    format!("{key}: {}", tool_output_json(value))
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{fields}}}")
        }
        other => serde_json::to_string(other).expect("JSON value serializes"),
    }
}

fn normalize_transcript(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn tts_options(_tts_label: &str) -> Value {
    json!({
        "language": "English",
        "speaker": "Aiden",
        "sample_rate": 24000,
        "max_output_tokens": 128,
        "temperature": 0.9,
        "top_k": 50,
        "seed": 1337,
        "deterministic": true,
    })
}

fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn resample_linear(samples: &[f32], source_rate: usize, target_rate: usize) -> Vec<f32> {
    if source_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let target_count = ((samples.len() * target_rate) + (source_rate / 2)) / source_rate;
    let target_count = target_count.max(1);
    if target_count == 1 {
        return vec![samples[0]];
    }

    let ratio = source_rate as f64 / target_rate as f64;
    let last = samples.len() - 1;
    (0..target_count)
        .map(|index| {
            let position = index as f64 * ratio;
            let left = (position as usize).min(last);
            let right = (left + 1).min(last);
            let mix = (position - left as f64) as f32;
            samples[left] * (1.0 - mix) + samples[right] * mix
        })
        .collect()
}

fn wav_to_float32_pcm(wav_bytes: &[u8], target_rate: usize) -> Vec<f32> {
    assert!(
        wav_bytes.starts_with(b"RIFF"),
        "TTS WAV missing RIFF header"
    );
    assert_eq!(&wav_bytes[8..12], b"WAVE", "TTS WAV missing WAVE tag");

    let mut offset = 12usize;
    let mut channels = None;
    let mut sample_rate = None;
    let mut sample_width = None;
    let mut data = None;

    while offset + 8 <= wav_bytes.len() {
        let chunk_id = &wav_bytes[offset..offset + 4];
        let chunk_len = read_u32_le(wav_bytes, offset + 4) as usize;
        let chunk_start = offset + 8;
        let chunk_end = chunk_start + chunk_len;
        assert!(
            chunk_end <= wav_bytes.len(),
            "TTS WAV chunk extends past file"
        );

        match chunk_id {
            b"fmt " => {
                assert!(chunk_len >= 16, "TTS WAV fmt chunk too small");
                let format = read_u16_le(wav_bytes, chunk_start);
                assert_eq!(format, 1, "expected PCM WAV format");
                channels = Some(read_u16_le(wav_bytes, chunk_start + 2) as usize);
                sample_rate = Some(read_u32_le(wav_bytes, chunk_start + 4) as usize);
                sample_width = Some(read_u16_le(wav_bytes, chunk_start + 14) as usize / 8);
            }
            b"data" => {
                data = Some(&wav_bytes[chunk_start..chunk_end]);
            }
            _ => {}
        }

        offset = chunk_end + (chunk_len % 2);
    }

    let channels = channels.expect("TTS WAV missing channel count");
    let source_rate = sample_rate.expect("TTS WAV missing sample rate");
    let sample_width = sample_width.expect("TTS WAV missing sample width");
    let data = data.expect("TTS WAV missing data chunk");
    assert!(channels >= 1, "TTS WAV must have at least one channel");
    assert_eq!(sample_width, 2, "expected 16-bit PCM WAV");
    assert_eq!(data.len() % (sample_width * channels), 0);

    let mono = if channels == 1 {
        data.chunks_exact(2)
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0)
            .collect::<Vec<_>>()
    } else {
        data.chunks_exact(sample_width * channels)
            .map(|frame| {
                let sum = (0..channels)
                    .map(|channel| {
                        let start = channel * sample_width;
                        i16::from_le_bytes([frame[start], frame[start + 1]]) as f32
                    })
                    .sum::<f32>();
                sum / (channels as f32 * 32768.0)
            })
            .collect::<Vec<_>>()
    };

    resample_linear(&mono, source_rate, target_rate)
}

fn reasoning_for(model: Model) -> Option<ReasoningConfig> {
    model.thinking.enabled().then(|| ReasoningConfig::Object {
        effort: "medium".to_string(),
    })
}

fn request(input: Vec<ResponseInputItem>) -> ResponsesRequest {
    ResponsesRequest {
        input: ResponsesInput::Items(input),
        stream: true,
        instructions: None,
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        min_p: None,
        deterministic: true,
        frequency_penalty: None,
        presence_penalty: None,
        max_output_tokens: Some(512),
        top_logprobs: None,
        core_tools: Vec::new(),
        active_tools: Vec::new(),
        tool_choice: None,
        min_tool_calls: None,
        max_tool_calls: None,
        text: None,
        reasoning: None,
        reasoning_effort: None,
        metadata: None,
        parallel_tool_calls: false,
        prefix_cache: Some(false),
        stream_tokens: true,
    }
}

fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    })
}

fn weather_tool() -> Value {
    tool(
        "get_weather",
        "Get the current weather for a location.",
        json!({"location": {"type": "string"}}),
        &["location"],
    )
}

fn get_time_tool() -> Value {
    tool(
        "get_time",
        "Get the current local time in a timezone.",
        json!({
            "timezone": {
                "type": "string",
                "description": "IANA timezone identifier, for example America/New_York.",
            }
        }),
        &["timezone"],
    )
}

fn generate_image_tool() -> Value {
    tool(
        "generate_image",
        "Generate an image from a text prompt.",
        json!({
            "prompt": {
                "type": "string",
                "description": "A concise visual prompt for the image generator.",
            }
        }),
        &["prompt"],
    )
}

async fn run_stream(model: Model, request: ResponsesRequest) -> Turn {
    let fixture = get_fixture().await;
    let result = fixture
        .client
        .aresponses(model.checkpoint, request)
        .await
        .unwrap_or_else(|err| {
            panic!(
                "responses request failed for {}: {err:?}",
                model.template_type
            )
        });
    let ResponsesResult::Stream { events: stream, .. } = result else {
        panic!("expected stream for {}", model.template_type);
    };
    drain_stream(stream).await
}

fn count(turn: &Turn, event: &'static str) -> usize {
    turn.counts.get(event).copied().unwrap_or(0)
}

fn added(turn: &Turn, item_type: &'static str) -> usize {
    turn.added.get(item_type).copied().unwrap_or(0)
}

fn function_call_items_added(turn: &Turn) -> Vec<&OutputFunctionCall> {
    turn.items_added
        .iter()
        .filter_map(|item| match item {
            ResponseOutputItem::FunctionCall(call) => Some(call),
            _ => None,
        })
        .collect()
}

fn message_items_added(turn: &Turn) -> Vec<&OutputMessage> {
    turn.items_added
        .iter()
        .filter_map(|item| match item {
            ResponseOutputItem::Message(message) => Some(message),
            _ => None,
        })
        .collect()
}

fn message_items_done(turn: &Turn) -> Vec<&OutputMessage> {
    turn.items_done
        .iter()
        .filter_map(|item| match item {
            ResponseOutputItem::Message(message) => Some(message),
            _ => None,
        })
        .collect()
}

fn reasoning_items_added(turn: &Turn) -> Vec<&OutputReasoning> {
    turn.items_added
        .iter()
        .filter_map(|item| match item {
            ResponseOutputItem::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect()
}

fn assert_response_lifecycle(turn: &Turn) {
    assert_eq!(turn.order.first().copied(), Some("response.created"));
    assert_eq!(turn.order.last().copied(), Some("done"));
    assert_eq!(count(turn, "response.created"), 1);
    assert_eq!(count(turn, "response.in_progress"), 1);
    assert_eq!(count(turn, "response.completed"), 1);
}

fn assert_optional_reasoning(turn: &Turn, model: Model, label: &str, max_one_message: &str) {
    let reasoning_blocks = added(turn, "reasoning");
    if !model.thinking.enabled() {
        assert_eq!(reasoning_blocks, 0, "{label}: non-reasoning model reasoned");
    }
    if reasoning_blocks > 0 {
        assert_eq!(reasoning_blocks, 1, "{label}: {max_one_message}");
        assert_eq!(count(turn, "response.reasoning.done"), 1);
        assert!(count(turn, "response.reasoning.delta") >= 1);
        assert_eq!(
            turn.reasoning.trim(),
            turn.reasoning_done.as_deref().unwrap_or_default(),
            "{label}: reasoning deltas != reasoning.done"
        );
    } else {
        assert_eq!(count(turn, "response.reasoning.delta"), 0);
    }
}

fn assert_message_lifecycle(turn: &Turn, label: &str) {
    assert_eq!(
        count(turn, "response.output_text.done"),
        1,
        "{label}: expected one message"
    );
    assert_eq!(
        turn.content,
        turn.content_done.as_deref().unwrap_or_default(),
        "{label}: content deltas != output_text.done"
    );
    let msg_open = message_items_added(turn);
    let msg_done = message_items_done(turn);
    assert!(
        msg_open.len() == 1 && msg_done.len() == 1,
        "{label}: expected one message item"
    );
    assert_eq!(msg_open[0].role, "assistant");
    assert!(
        msg_open[0].content.is_empty(),
        "{label}: message must open with empty content"
    );
    assert_eq!(msg_open[0].status, OutputStatus::InProgress);
    assert_eq!(msg_done[0].status, OutputStatus::Completed);
}

fn parse_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments)
        .unwrap_or_else(|err| panic!("invalid tool arguments {arguments:?}: {err}"))
}

fn image_part(artifact: &ModalArtifact) -> Value {
    json!({
        "type": "input_image",
        "image_url": format!(
            "data:{};base64,{}",
            artifact.mime_type,
            BASE64.encode(&artifact.data)
        ),
        "detail": "auto",
    })
}

fn model_by_checkpoint(checkpoint: &str) -> Model {
    *MODELS
        .iter()
        .find(|model| model.checkpoint == checkpoint)
        .unwrap_or_else(|| panic!("missing test model {checkpoint}"))
}

#[tokio::test]
async fn test_thinking_on_off() {
    const SYSTEM: &str = "You are a careful assistant. Answer the user's question correctly.";
    const QUESTION: &str = "What is 17 + 26? Reply with just the number.";
    const ANSWER: &str = "43";

    for &model in MODELS {
        if !model.thinking.enabled() || model.thinking == Thinking::Required {
            continue;
        }

        let conversation = vec![message("system", SYSTEM), message("user", QUESTION)];

        let mut on_request = request(conversation.clone());
        on_request.reasoning = Some(ReasoningConfig::Object {
            effort: "medium".to_string(),
        });
        let on = run_stream(model, on_request).await;
        assert_or_record(model.template_type, "thinking_on_off", "on", &on.events);

        assert_response_lifecycle(&on);
        assert_eq!(
            added(&on, "reasoning"),
            1,
            "on: expected exactly one reasoning block"
        );
        assert_eq!(count(&on, "response.reasoning.done"), 1);
        assert!(
            count(&on, "response.reasoning.delta") >= 1,
            "on: reasoning produced no deltas"
        );
        assert_eq!(
            on.reasoning.trim(),
            on.reasoning_done.as_deref().unwrap_or_default(),
            "on: reasoning deltas != reasoning.done"
        );
        assert!(
            !on.reasoning.contains("<|") && !on.reasoning.contains("</"),
            "on: control leak in reasoning"
        );
        let on_reasoning = reasoning_items_added(&on);
        assert_eq!(
            on_reasoning.len(),
            1,
            "on: expected one reasoning item opened"
        );
        assert_eq!(on_reasoning[0].status, OutputStatus::InProgress);
        assert!(
            reasoning_tokens(&on) > 0,
            "on: usage reported zero reasoning tokens while thinking"
        );
        assert_message_lifecycle(&on, "on");
        assert!(
            on.content_done
                .as_deref()
                .unwrap_or_default()
                .contains(ANSWER),
            "on: wrong answer: {:?}",
            on.content_done
        );

        let mut off_request = request(conversation);
        off_request.reasoning = Some(false.into());
        let off = run_stream(model, off_request).await;
        assert_or_record(model.template_type, "thinking_on_off", "off", &off.events);

        assert_response_lifecycle(&off);
        assert_eq!(
            added(&off, "reasoning"),
            0,
            "off: thinking still produced a reasoning block"
        );
        assert_eq!(
            count(&off, "response.reasoning.delta"),
            0,
            "off: reasoning deltas leaked"
        );
        assert_eq!(
            count(&off, "response.reasoning.done"),
            0,
            "off: reasoning.done leaked"
        );
        assert!(
            reasoning_items_added(&off).is_empty(),
            "off: a reasoning item was opened"
        );
        assert_eq!(
            reasoning_tokens(&off),
            0,
            "off: usage charged reasoning tokens with thinking off"
        );
        assert_message_lifecycle(&off, "off");
        assert!(
            off.content_done
                .as_deref()
                .unwrap_or_default()
                .contains(ANSWER),
            "off: wrong answer: {:?}",
            off.content_done
        );
    }
}

#[tokio::test]
async fn test_reason_then_structured() {
    const SYSTEM: &str = "You are a helpful assistant. Reason about the request, then return the answer as a single JSON object that matches the requested schema exactly.";
    const USER: &str = "Return the capital of France and its population. Use the capital string \"Paris\" and the integer literal 2148327 (no decimal point).";
    let expected = json!({"capital": "Paris", "population": 2148327});
    let city_schema = json!({
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "population": {"type": "integer"},
        },
        "required": ["capital", "population"],
        "additionalProperties": false,
    });

    for &model in MODELS {
        if !model.thinking.enabled() {
            continue;
        }

        let mut turn1_request = request(vec![message("system", SYSTEM), message("user", USER)]);
        turn1_request.text = Some(json!({
            "format": {
                "type": "json_schema",
                "name": "city_info",
                "schema": city_schema,
                "strict": true,
            }
        }));
        turn1_request.reasoning = Some(ReasoningConfig::Object {
            effort: "medium".to_string(),
        });
        let turn1 = run_stream(model, turn1_request).await;
        assert_or_record(
            model.template_type,
            "reason_then_structured",
            "turn1",
            &turn1.events,
        );

        assert_response_lifecycle(&turn1);
        assert_eq!(
            added(&turn1, "reasoning"),
            1,
            "turn1: expected exactly one reasoning block"
        );
        assert_eq!(
            count(&turn1, "response.reasoning.done"),
            1,
            "turn1: reasoning did not terminate cleanly"
        );
        assert!(count(&turn1, "response.reasoning.delta") >= 1);
        assert_eq!(
            turn1.reasoning.trim(),
            turn1.reasoning_done.as_deref().unwrap_or_default(),
            "turn1: reasoning deltas != reasoning.done"
        );
        assert!(
            !turn1.reasoning.contains("<|") && !turn1.reasoning.contains("</"),
            "turn1: control leak in reasoning"
        );
        assert_eq!(
            count(&turn1, "response.function_call_arguments.done"),
            0,
            "turn1: unexpected tool call"
        );
        assert_message_lifecycle(&turn1, "turn1");
        let parsed: Value = serde_json::from_str(turn1.content_done.as_deref().unwrap_or_default())
            .unwrap_or_else(|err| {
                panic!("{}: invalid structured output: {err}", model.template_type)
            });
        assert_eq!(
            parsed, expected,
            "{}: structured output != expected: {:?}",
            model.template_type, turn1.content_done
        );
    }
}

#[tokio::test]
async fn test_reason_then_tool() {
    for &model in MODELS {
        if !model.tools {
            continue;
        }
        let reasoning = reasoning_for(model);
        let mut conversation = vec![
            message("system", WEATHER_SYSTEM),
            message("user", "What's the weather in San Francisco?"),
        ];

        let mut turn1_request = request(conversation.clone());
        turn1_request.core_tools = vec![weather_tool()];
        turn1_request.tool_choice = Some(json!("required"));
        turn1_request.reasoning = reasoning.clone();
        let turn1 = run_stream(model, turn1_request).await;
        assert_or_record(
            model.template_type,
            "reason_then_tool",
            "turn1",
            &turn1.events,
        );

        assert_response_lifecycle(&turn1);
        assert_optional_reasoning(
            &turn1,
            model,
            "turn1",
            "expected exactly one reasoning block",
        );
        assert!(
            !turn1.counts.contains_key("response.output_text.delta"),
            "turn1: leaked message text on a tool turn"
        );
        assert_eq!(
            added(&turn1, "function_call"),
            1,
            "turn1: expected exactly one function_call opened"
        );
        assert_eq!(
            count(&turn1, "response.function_call_arguments.done"),
            1,
            "turn1: expected one arguments.done"
        );
        assert_eq!(turn1.function_calls.len(), 1);
        let call = &turn1.function_calls[0];
        let opened = function_call_items_added(&turn1);
        assert_eq!(opened.len(), 1, "turn1: expected one function_call opened");
        assert_eq!(opened[0].name, "get_weather");
        assert_eq!(opened[0].call_id, call.call_id);
        assert_eq!(
            opened[0].arguments, "",
            "turn1: function_call must open with empty arguments"
        );
        assert_eq!(opened[0].status, OutputStatus::InProgress);
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.status, OutputStatus::Completed);
        assert_eq!(
            parse_arguments(&call.arguments),
            json!({"location": "San Francisco"})
        );
        assert_eq!(
            turn1.field_args,
            HashMap::from([("location".to_string(), "San Francisco".to_string())]),
            "{}: per-argument field_path tagging wrong (expected location='San Francisco' value-only): {:?}",
            model.template_type,
            turn1.field_args
        );

        conversation.push(function_call(
            &call.call_id,
            &call.name,
            turn1.args_done.as_deref().unwrap_or_default(),
        ));
        conversation.push(function_output(
            &call.call_id,
            json!({"temperature": 65, "unit": "fahrenheit", "condition": "foggy"}),
        ));
        let mut turn2_request = request(conversation);
        turn2_request.core_tools = vec![weather_tool()];
        turn2_request.tool_choice = Some(json!("none"));
        turn2_request.reasoning = reasoning;
        let turn2 = run_stream(model, turn2_request).await;
        assert_or_record(
            model.template_type,
            "reason_then_tool",
            "turn2",
            &turn2.events,
        );

        assert_response_lifecycle(&turn2);
        assert_optional_reasoning(
            &turn2,
            model,
            "turn2",
            "expected exactly one reasoning block",
        );
        assert_eq!(
            count(&turn2, "response.function_call_arguments.done"),
            0,
            "turn2: unexpected tool call"
        );
        assert_message_lifecycle(&turn2, "turn2");
        let answer = turn2
            .content_done
            .as_deref()
            .unwrap_or_default()
            .to_lowercase();
        assert!(
            answer.contains("65") || answer.contains("fog"),
            "{}: answer ignored the tool result: {answer:?}",
            model.template_type
        );
    }
}

#[tokio::test]
async fn test_tool_result_grounding() {
    let surprise_result = json!({"temperature": 9, "unit": "celsius", "condition": "snowing"});

    for &model in MODELS {
        if !model.tools {
            continue;
        }
        let reasoning = reasoning_for(model);
        let mut conversation = vec![
            message("system", WEATHER_SYSTEM),
            message("user", "What's the weather in San Francisco?"),
        ];

        let mut turn1_request = request(conversation.clone());
        turn1_request.core_tools = vec![weather_tool()];
        turn1_request.tool_choice = Some(json!("required"));
        turn1_request.reasoning = reasoning.clone();
        let turn1 = run_stream(model, turn1_request).await;
        assert_or_record(
            model.template_type,
            "tool_result_grounding",
            "turn1",
            &turn1.events,
        );

        assert_response_lifecycle(&turn1);
        assert_optional_reasoning(
            &turn1,
            model,
            "turn1",
            "expected at most one reasoning block",
        );
        assert!(
            !turn1.counts.contains_key("response.output_text.delta"),
            "turn1: leaked message text on a tool turn"
        );
        assert_eq!(
            added(&turn1, "function_call"),
            1,
            "turn1: expected exactly one function_call opened"
        );
        assert_eq!(
            count(&turn1, "response.function_call_arguments.done"),
            1,
            "turn1: expected one arguments.done"
        );
        assert_eq!(turn1.function_calls.len(), 1);
        let call = &turn1.function_calls[0];
        let opened = function_call_items_added(&turn1);
        assert_eq!(opened.len(), 1, "turn1: expected one function_call opened");
        assert_eq!(opened[0].name, "get_weather");
        assert_eq!(opened[0].call_id, call.call_id);
        assert_eq!(
            opened[0].arguments, "",
            "turn1: function_call must open with empty arguments"
        );
        assert_eq!(opened[0].status, OutputStatus::InProgress);
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.status, OutputStatus::Completed);
        assert_eq!(
            parse_arguments(&call.arguments),
            json!({"location": "San Francisco"})
        );
        assert_eq!(
            turn1.field_args,
            HashMap::from([("location".to_string(), "San Francisco".to_string())]),
            "{}: per-argument field_path tagging wrong (expected location='San Francisco' value-only): {:?}",
            model.template_type,
            turn1.field_args
        );

        conversation.push(function_call(
            &call.call_id,
            &call.name,
            turn1.args_done.as_deref().unwrap_or_default(),
        ));
        conversation.push(function_output(&call.call_id, surprise_result.clone()));
        let mut turn2_request = request(conversation);
        turn2_request.core_tools = vec![weather_tool()];
        turn2_request.tool_choice = Some(json!("none"));
        turn2_request.reasoning = reasoning;
        let turn2 = run_stream(model, turn2_request).await;
        assert_or_record(
            model.template_type,
            "tool_result_grounding",
            "turn2",
            &turn2.events,
        );

        assert_response_lifecycle(&turn2);
        assert_optional_reasoning(
            &turn2,
            model,
            "turn2",
            "expected exactly one reasoning block",
        );
        assert_eq!(
            count(&turn2, "response.function_call_arguments.done"),
            0,
            "turn2: unexpected tool call"
        );
        assert_message_lifecycle(&turn2, "turn2");
        let answer = turn2
            .content_done
            .as_deref()
            .unwrap_or_default()
            .to_lowercase();
        assert!(
            answer.contains('9'),
            "{}: answer dropped the injected temperature: {answer:?}",
            model.template_type
        );
        assert!(
            answer.contains("snow"),
            "{}: answer dropped the injected condition: {answer:?}",
            model.template_type
        );
        for prior in ["65", "fog"] {
            assert!(
                !answer.contains(prior),
                "{}: answer leaked the hallucinated SF prior ({prior:?}) instead of grounding on the tool result: {answer:?}",
                model.template_type
            );
        }
    }
}

#[tokio::test]
async fn test_image_tool_self_loop_and_blind_verifier() {
    const SYSTEM: &str = "You are a multimodal assistant with image-generation tools. Use the tool when the user asks you to create an image. After the tool result is returned, inspect the image and answer from the image.";
    const USER: &str = "Use generate_image to create a simple image of one red apple centered on a plain white background. After the tool returns, tell me what object is in the generated image.";

    let gemma = model_by_checkpoint(GEMMA4_MODEL_ID);
    let moondream = model_by_checkpoint(MOONDREAM_MODEL_ID);
    let mut conversation = vec![message("system", SYSTEM), message("user", USER)];

    let mut generator_request = request(conversation.clone());
    generator_request.core_tools = vec![generate_image_tool()];
    generator_request.tool_choice = Some(json!("required"));
    generator_request.reasoning = Some(ReasoningConfig::Object {
        effort: "medium".to_string(),
    });
    let generator = run_stream(gemma, generator_request).await;
    assert_or_record("gemma4", "image_tool_self_loop", "turn1", &generator.events);
    assert_or_record(
        "gemma4",
        "image_tool_blind_verifier",
        "generator",
        &generator.events,
    );

    assert_response_lifecycle(&generator);
    assert_optional_reasoning(
        &generator,
        gemma,
        "generator",
        "expected at most one reasoning block",
    );
    assert!(
        !generator.counts.contains_key("response.output_text.delta"),
        "generator: leaked message text on a tool turn"
    );
    assert_eq!(
        added(&generator, "function_call"),
        1,
        "generator: expected one function_call"
    );
    assert_eq!(
        count(&generator, "response.function_call_arguments.done"),
        1,
        "generator: expected one arguments.done"
    );
    assert_eq!(generator.function_calls.len(), 1);

    let opened = function_call_items_added(&generator);
    assert_eq!(
        opened.len(),
        1,
        "generator: expected one function_call opened"
    );
    let call = &generator.function_calls[0];
    assert_eq!(opened[0].name, "generate_image");
    assert_eq!(opened[0].call_id, call.call_id);
    assert_eq!(
        opened[0].arguments, "",
        "generator: function_call must open with empty arguments"
    );
    assert_eq!(opened[0].status, OutputStatus::InProgress);
    assert_eq!(call.name, "generate_image");
    assert_eq!(call.status, OutputStatus::Completed);
    let arguments = parse_arguments(&call.arguments);
    let prompt = arguments
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("generator: missing prompt argument: {:?}", call.arguments));
    assert!(
        prompt.to_lowercase().contains("apple"),
        "generator prompt lost the requested object: {prompt:?}"
    );

    let fixture = get_fixture().await;
    let artifacts = fixture
        .client
        .agenerate_image(
            IDEOGRAM4_MODEL_ID,
            prompt,
            Some(json!({
                "height": 512,
                "width": 512,
                "num_steps": 12,
                "guidance_scale": 7.0,
                "mu": 0.5,
                "std": 1.75,
                "seed": 17,
            })),
        )
        .await
        .unwrap_or_else(|err| panic!("ideogram image generation failed: {err:?}"));
    assert_eq!(artifacts.len(), 1, "ideogram returned unexpected artifacts");
    let artifact = &artifacts[0];
    assert_eq!(artifact.mime_type, "image/png");
    assert!(
        artifact.data.starts_with(b"\x89PNG\r\n\x1a\n"),
        "ideogram did not return a PNG"
    );
    assert!(
        artifact.data.len() > 1024,
        "ideogram returned a suspiciously small PNG"
    );
    let image = image_part(artifact);

    conversation.push(function_call(
        &call.call_id,
        &call.name,
        generator.args_done.as_deref().unwrap_or_default(),
    ));
    conversation.push(ResponseInputItem::FunctionCallOutput {
        call_id: call.call_id.clone(),
        output: FunctionCallOutputContent::Content(vec![image.clone()]),
    });

    let mut self_loop_request = request(conversation);
    self_loop_request.core_tools = vec![generate_image_tool()];
    self_loop_request.tool_choice = Some(json!("none"));
    self_loop_request.reasoning = Some(ReasoningConfig::Object {
        effort: "medium".to_string(),
    });
    let self_loop = run_stream(gemma, self_loop_request).await;
    assert_or_record("gemma4", "image_tool_self_loop", "turn2", &self_loop.events);

    assert_response_lifecycle(&self_loop);
    assert_eq!(
        count(&self_loop, "response.function_call_arguments.done"),
        0,
        "turn2: unexpected tool call"
    );
    assert_message_lifecycle(&self_loop, "turn2");
    let self_loop_answer = self_loop
        .content_done
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();
    assert!(
        self_loop_answer.contains("apple"),
        "gemma4 did not ground on the generated image: {self_loop_answer:?}"
    );

    let mut verifier_request = request(vec![ResponseInputItem::Message {
        role: "user".to_string(),
        content: json!([
            {
                "type": "input_text",
                "text": "What is in this image? Answer with the main object.",
            },
            image,
        ]),
        tool_calls: None,
        tool_call_id: None,
    }]);
    verifier_request.reasoning = Some(ReasoningConfig::Object {
        effort: "medium".to_string(),
    });
    let verifier = run_stream(moondream, verifier_request).await;
    assert_or_record(
        "moondream3",
        "image_tool_blind_verifier",
        "verifier",
        &verifier.events,
    );

    assert_response_lifecycle(&verifier);
    assert_message_lifecycle(&verifier, "verifier");
    let verifier_answer = verifier
        .content_done
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();
    assert!(
        verifier_answer.contains("apple"),
        "moondream3 did not identify the generated image: {verifier_answer:?}"
    );
}

#[tokio::test]
async fn test_image_tool_self_loop_and_blind_verifier_flux() {
    const SYSTEM: &str = "You are a multimodal assistant with image-generation tools. Use the tool when the user asks you to create an image. After the tool result is returned, inspect the image and answer from the image.";
    const USER: &str = "Use generate_image to create a simple image of one red apple centered on a plain white background. After the tool returns, tell me what object is in the generated image.";

    let gemma = model_by_checkpoint(GEMMA4_MODEL_ID);
    let moondream = model_by_checkpoint(MOONDREAM_MODEL_ID);
    let mut conversation = vec![message("system", SYSTEM), message("user", USER)];

    let mut generator_request = request(conversation.clone());
    generator_request.core_tools = vec![generate_image_tool()];
    generator_request.tool_choice = Some(json!("required"));
    generator_request.reasoning = Some(ReasoningConfig::Object {
        effort: "medium".to_string(),
    });
    let generator = run_stream(gemma, generator_request).await;
    assert_or_record(
        "gemma4",
        "image_tool_self_loop_flux",
        "turn1",
        &generator.events,
    );
    assert_or_record(
        "gemma4",
        "image_tool_blind_verifier_flux",
        "generator",
        &generator.events,
    );

    assert_response_lifecycle(&generator);
    assert_optional_reasoning(
        &generator,
        gemma,
        "generator",
        "expected at most one reasoning block",
    );
    assert!(
        !generator.counts.contains_key("response.output_text.delta"),
        "generator: leaked message text on a tool turn"
    );
    assert_eq!(
        added(&generator, "function_call"),
        1,
        "generator: expected one function_call"
    );
    assert_eq!(
        count(&generator, "response.function_call_arguments.done"),
        1,
        "generator: expected one arguments.done"
    );
    assert_eq!(generator.function_calls.len(), 1);

    let opened = function_call_items_added(&generator);
    assert_eq!(
        opened.len(),
        1,
        "generator: expected one function_call opened"
    );
    let call = &generator.function_calls[0];
    assert_eq!(opened[0].name, "generate_image");
    assert_eq!(opened[0].call_id, call.call_id);
    assert_eq!(
        opened[0].arguments, "",
        "generator: function_call must open with empty arguments"
    );
    assert_eq!(opened[0].status, OutputStatus::InProgress);
    assert_eq!(call.name, "generate_image");
    assert_eq!(call.status, OutputStatus::Completed);
    let arguments = parse_arguments(&call.arguments);
    let prompt = arguments
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("generator: missing prompt argument: {:?}", call.arguments));
    assert!(
        prompt.to_lowercase().contains("apple"),
        "generator prompt lost the requested object: {prompt:?}"
    );

    let fixture = get_fixture().await;
    let artifacts = fixture
        .client
        .agenerate_image(
            FLUX2_MODEL_ID,
            prompt,
            Some(json!({
                "height": 512,
                "width": 512,
                "num_steps": 8,
                "guidance_scale": 3.5,
                "seed": 17,
            })),
        )
        .await
        .unwrap_or_else(|err| panic!("flux image generation failed: {err:?}"));
    assert_eq!(artifacts.len(), 1, "flux returned unexpected artifacts");
    let artifact = &artifacts[0];
    assert_eq!(artifact.mime_type, "image/png");
    assert_eq!(artifact.decoder_id, "flux");
    assert!(
        artifact.data.starts_with(b"\x89PNG\r\n\x1a\n"),
        "flux did not return a PNG"
    );
    assert!(
        artifact.data.len() > 1024,
        "flux returned a suspiciously small PNG"
    );
    let image = image_part(artifact);

    conversation.push(function_call(
        &call.call_id,
        &call.name,
        generator.args_done.as_deref().unwrap_or_default(),
    ));
    conversation.push(ResponseInputItem::FunctionCallOutput {
        call_id: call.call_id.clone(),
        output: FunctionCallOutputContent::Content(vec![image.clone()]),
    });

    let mut self_loop_request = request(conversation);
    self_loop_request.core_tools = vec![generate_image_tool()];
    self_loop_request.tool_choice = Some(json!("none"));
    self_loop_request.reasoning = Some(ReasoningConfig::Object {
        effort: "medium".to_string(),
    });
    let self_loop = run_stream(gemma, self_loop_request).await;
    assert_or_record(
        "gemma4",
        "image_tool_self_loop_flux",
        "turn2",
        &self_loop.events,
    );

    assert_response_lifecycle(&self_loop);
    assert_eq!(
        count(&self_loop, "response.function_call_arguments.done"),
        0,
        "turn2: unexpected tool call"
    );
    assert_message_lifecycle(&self_loop, "turn2");
    let self_loop_answer = self_loop
        .content_done
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();
    assert!(
        self_loop_answer.contains("apple"),
        "gemma4 did not ground on the Flux-generated image: {self_loop_answer:?}"
    );

    let mut verifier_request = request(vec![ResponseInputItem::Message {
        role: "user".to_string(),
        content: json!([
            {
                "type": "input_text",
                "text": "What is in this image? Answer with the main object.",
            },
            image,
        ]),
        tool_calls: None,
        tool_call_id: None,
    }]);
    verifier_request.reasoning = Some(ReasoningConfig::Object {
        effort: "medium".to_string(),
    });
    let verifier = run_stream(moondream, verifier_request).await;
    assert_or_record(
        "moondream3",
        "image_tool_blind_verifier_flux",
        "verifier",
        &verifier.events,
    );

    assert_response_lifecycle(&verifier);
    assert_message_lifecycle(&verifier, "verifier");
    let verifier_answer = verifier
        .content_done
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();
    assert!(
        verifier_answer.contains("apple"),
        "moondream3 did not identify the Flux-generated image: {verifier_answer:?}"
    );
}

#[tokio::test]
async fn test_image_edit_tool_blind_verifier() {
    const SYSTEM: &str = "You are a multimodal assistant with image-generation tools. Use the tool when the user asks you to create an image. After the tool result is returned, inspect the image and answer from the image.";

    let gemma = model_by_checkpoint(GEMMA4_MODEL_ID);
    let moondream = model_by_checkpoint(MOONDREAM_MODEL_ID);
    let conversation = vec![
        message("system", SYSTEM),
        message("user", SHAPES_IMAGE_USER),
    ];

    let mut generator_request = request(conversation);
    generator_request.core_tools = vec![generate_image_tool()];
    generator_request.tool_choice = Some(json!("required"));
    generator_request.reasoning = Some(ReasoningConfig::Object {
        effort: "medium".to_string(),
    });
    let generator = run_stream(gemma, generator_request).await;
    assert_or_record(
        "gemma4",
        "image_edit_tool_blind_verifier",
        "generator",
        &generator.events,
    );

    assert_response_lifecycle(&generator);
    assert_optional_reasoning(
        &generator,
        gemma,
        "generator",
        "expected at most one reasoning block",
    );
    assert!(
        !generator.counts.contains_key("response.output_text.delta"),
        "generator: leaked message text on a tool turn"
    );
    assert_eq!(
        added(&generator, "function_call"),
        1,
        "generator: expected one function_call"
    );
    assert_eq!(
        count(&generator, "response.function_call_arguments.done"),
        1,
        "generator: expected one arguments.done"
    );
    assert_eq!(generator.function_calls.len(), 1);

    let opened = function_call_items_added(&generator);
    assert_eq!(
        opened.len(),
        1,
        "generator: expected one function_call opened"
    );
    let call = &generator.function_calls[0];
    assert_eq!(opened[0].name, "generate_image");
    assert_eq!(opened[0].call_id, call.call_id);
    assert_eq!(
        opened[0].arguments, "",
        "generator: function_call must open with empty arguments"
    );
    assert_eq!(opened[0].status, OutputStatus::InProgress);
    assert_eq!(call.name, "generate_image");
    assert_eq!(call.status, OutputStatus::Completed);
    let arguments = parse_arguments(&call.arguments);
    let prompt = arguments
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("generator: missing prompt argument: {:?}", call.arguments));
    let prompt_lower = prompt.to_lowercase();
    for term in ["red", "circle", "blue", "square"] {
        assert!(
            prompt_lower.contains(term),
            "generator prompt lost requested term {term:?}: {prompt:?}"
        );
    }

    let fixture = get_fixture().await;
    let source = fixture
        .client
        .agenerate_image(
            IDEOGRAM4_MODEL_ID,
            prompt,
            Some(json!({
                "height": 512,
                "width": 512,
                "num_steps": 12,
                "guidance_scale": 7.0,
                "mu": 0.5,
                "std": 1.75,
                "seed": 17,
            })),
        )
        .await
        .unwrap_or_else(|err| panic!("ideogram image generation failed: {err:?}"));
    assert_eq!(source.len(), 1, "ideogram returned unexpected artifacts");
    let source = &source[0];
    assert_eq!(source.mime_type, "image/png");
    assert!(
        source.data.starts_with(b"\x89PNG\r\n\x1a\n"),
        "ideogram did not return a PNG"
    );
    assert!(
        source.data.len() > 1024,
        "ideogram returned a suspiciously small PNG"
    );

    let edited = fixture
        .client
        .aedit_image(
            QWEN_IMAGE_EDIT_MODEL_ID,
            &source.data,
            SWAP_COLORS_PROMPT,
            Some(json!({
                "height": 512,
                "width": 512,
                "num_steps": 8,
                "true_cfg_scale": 1.0,
                "negative_prompt": "",
                "seed": 1337,
            })),
        )
        .await
        .unwrap_or_else(|err| panic!("qwen image edit failed: {err:?}"));
    assert_eq!(
        edited.len(),
        1,
        "qwen image edit returned unexpected artifacts"
    );
    let edited = &edited[0];
    assert_eq!(edited.mime_type, "image/png");
    assert_eq!(edited.decoder_id, "qwen_image_edit");
    assert!(
        edited.data.starts_with(b"\x89PNG\r\n\x1a\n"),
        "qwen image edit did not return a PNG"
    );
    assert!(
        edited.data.len() > 1024,
        "qwen image edit returned a suspiciously small PNG"
    );
    let image = image_part(edited);

    let mut verifier_request = request(vec![ResponseInputItem::Message {
        role: "user".to_string(),
        content: json!([
            {
                "type": "input_text",
                "text": "What colors and shapes are in this image? Answer briefly.",
            },
            image,
        ]),
        tool_calls: None,
        tool_call_id: None,
    }]);
    verifier_request.reasoning = Some(ReasoningConfig::Object {
        effort: "medium".to_string(),
    });
    let verifier = run_stream(moondream, verifier_request).await;
    assert_or_record(
        "moondream3",
        "image_edit_tool_blind_verifier",
        "verifier",
        &verifier.events,
    );

    assert_response_lifecycle(&verifier);
    assert_message_lifecycle(&verifier, "verifier");
    let verifier_answer = verifier
        .content_done
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();
    for term in ["blue", "circle", "red", "square"] {
        assert!(
            verifier_answer.contains(term),
            "moondream3 did not identify the edited image as blue circle/red square: {verifier_answer:?}"
        );
    }
}

#[tokio::test]
async fn test_image_edit_tool_blind_verifier_flux() {
    const SYSTEM: &str = "You are a multimodal assistant with image-generation tools. Use the tool when the user asks you to create an image. After the tool result is returned, inspect the image and answer from the image.";

    let gemma = model_by_checkpoint(GEMMA4_MODEL_ID);
    let moondream = model_by_checkpoint(MOONDREAM_MODEL_ID);
    let conversation = vec![
        message("system", SYSTEM),
        message("user", SHAPES_IMAGE_USER),
    ];

    let mut generator_request = request(conversation);
    generator_request.core_tools = vec![generate_image_tool()];
    generator_request.tool_choice = Some(json!("required"));
    generator_request.reasoning = Some(ReasoningConfig::Object {
        effort: "medium".to_string(),
    });
    let generator = run_stream(gemma, generator_request).await;
    assert_or_record(
        "gemma4",
        "image_edit_tool_blind_verifier_flux",
        "generator",
        &generator.events,
    );

    assert_response_lifecycle(&generator);
    assert_optional_reasoning(
        &generator,
        gemma,
        "generator",
        "expected at most one reasoning block",
    );
    assert!(
        !generator.counts.contains_key("response.output_text.delta"),
        "generator: leaked message text on a tool turn"
    );
    assert_eq!(
        added(&generator, "function_call"),
        1,
        "generator: expected one function_call"
    );
    assert_eq!(
        count(&generator, "response.function_call_arguments.done"),
        1,
        "generator: expected one arguments.done"
    );
    assert_eq!(generator.function_calls.len(), 1);

    let opened = function_call_items_added(&generator);
    assert_eq!(
        opened.len(),
        1,
        "generator: expected one function_call opened"
    );
    let call = &generator.function_calls[0];
    assert_eq!(opened[0].name, "generate_image");
    assert_eq!(opened[0].call_id, call.call_id);
    assert_eq!(
        opened[0].arguments, "",
        "generator: function_call must open with empty arguments"
    );
    assert_eq!(opened[0].status, OutputStatus::InProgress);
    assert_eq!(call.name, "generate_image");
    assert_eq!(call.status, OutputStatus::Completed);
    let arguments = parse_arguments(&call.arguments);
    let prompt = arguments
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("generator: missing prompt argument: {:?}", call.arguments));
    let prompt_lower = prompt.to_lowercase();
    for term in ["red", "circle", "blue", "square"] {
        assert!(
            prompt_lower.contains(term),
            "generator prompt lost requested term {term:?}: {prompt:?}"
        );
    }

    let fixture = get_fixture().await;
    let source = fixture
        .client
        .agenerate_image(
            FLUX2_MODEL_ID,
            prompt,
            Some(json!({
                "height": 512,
                "width": 512,
                "num_steps": 8,
                "guidance_scale": 3.5,
                "seed": 17,
            })),
        )
        .await
        .unwrap_or_else(|err| panic!("flux image generation failed: {err:?}"));
    assert_eq!(source.len(), 1, "flux returned unexpected artifacts");
    let source = &source[0];
    assert_eq!(source.mime_type, "image/png");
    assert_eq!(source.decoder_id, "flux");
    assert!(
        source.data.starts_with(b"\x89PNG\r\n\x1a\n"),
        "flux did not return a PNG"
    );
    assert!(
        source.data.len() > 1024,
        "flux returned a suspiciously small PNG"
    );

    let edited = fixture
        .client
        .aedit_image(
            QWEN_IMAGE_EDIT_MODEL_ID,
            &source.data,
            SWAP_COLORS_PROMPT,
            Some(json!({
                "height": 512,
                "width": 512,
                "num_steps": 8,
                "true_cfg_scale": 1.0,
                "negative_prompt": "",
                "seed": 1337,
            })),
        )
        .await
        .unwrap_or_else(|err| panic!("qwen image edit failed: {err:?}"));
    assert_eq!(
        edited.len(),
        1,
        "qwen image edit returned unexpected artifacts"
    );
    let edited = &edited[0];
    assert_eq!(edited.mime_type, "image/png");
    assert_eq!(edited.decoder_id, "qwen_image_edit");
    assert!(
        edited.data.starts_with(b"\x89PNG\r\n\x1a\n"),
        "qwen image edit did not return a PNG"
    );
    assert!(
        edited.data.len() > 1024,
        "qwen image edit returned a suspiciously small PNG"
    );
    let image = image_part(edited);

    let mut verifier_request = request(vec![ResponseInputItem::Message {
        role: "user".to_string(),
        content: json!([
            {
                "type": "input_text",
                "text": "What colors and shapes are in this image? Answer briefly.",
            },
            image,
        ]),
        tool_calls: None,
        tool_call_id: None,
    }]);
    verifier_request.reasoning = Some(ReasoningConfig::Object {
        effort: "medium".to_string(),
    });
    let verifier = run_stream(moondream, verifier_request).await;
    assert_or_record(
        "moondream3",
        "image_edit_tool_blind_verifier_flux",
        "verifier",
        &verifier.events,
    );

    assert_response_lifecycle(&verifier);
    assert_message_lifecycle(&verifier, "verifier");
    let verifier_answer = verifier
        .content_done
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();
    for term in ["blue", "circle", "red", "square"] {
        assert!(
            verifier_answer.contains(term),
            "moondream3 did not identify the edited Flux source image as blue circle/red square: {verifier_answer:?}"
        );
    }
}

#[tokio::test]
async fn test_audio_telephone_tts_to_speech_to_text() {
    let fixture = get_fixture().await;
    for (tts_label, tts_model_id) in TTS_MODELS {
        for phrase in AUDIO_TELEPHONE_PHRASES {
            let artifacts = fixture
                .client
                .agenerate_audio(tts_model_id, phrase, Some(tts_options(tts_label)))
                .await
                .unwrap_or_else(|err| panic!("{tts_label} generation failed: {err:?}"));

            assert_eq!(
                artifacts.len(),
                1,
                "{tts_label} returned unexpected artifacts"
            );
            let artifact = &artifacts[0];
            assert_eq!(artifact.modal_type, "audio");
            assert_eq!(artifact.mime_type, "audio/wav");
            assert!(
                artifact.data.starts_with(b"RIFF"),
                "{tts_label} did not return a WAV"
            );
            assert!(
                artifact.data.len() > 44,
                "{tts_label} returned a suspiciously small WAV"
            );

            let pcm = wav_to_float32_pcm(&artifact.data, AUDIO_TELEPHONE_SAMPLE_RATE);
            assert!(!pcm.is_empty(), "{tts_label} produced no audio samples");

            for (stt_label, stt_model_id) in STT_MODELS {
                let transcript = fixture
                    .client
                    .atranscribe_audio(stt_model_id, &pcm)
                    .await
                    .unwrap_or_else(|err| {
                        panic!("{tts_label} -> {stt_label} transcription failed: {err:?}")
                    });
                let normalized = normalize_transcript(&transcript);
                assert_eq!(
                    normalized, phrase,
                    "{tts_label} -> {stt_label} telephone transcript drifted: {transcript:?}"
                );
            }
        }
    }
}

#[tokio::test]
async fn test_tool_selection() {
    let distractors = vec![
        tool(
            "get_time",
            "Get the current time in a location.",
            json!({"location": {"type": "string"}}),
            &["location"],
        ),
        tool(
            "get_news",
            "Get the latest news headlines for a topic.",
            json!({"topic": {"type": "string"}}),
            &["topic"],
        ),
        tool(
            "send_email",
            "Send an email to a recipient.",
            json!({"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}),
            &["to", "subject", "body"],
        ),
        tool(
            "set_timer",
            "Set a countdown timer for a number of seconds.",
            json!({"seconds": {"type": "integer"}}),
            &["seconds"],
        ),
        tool(
            "get_stock_price",
            "Get the current stock price for a ticker symbol.",
            json!({"ticker": {"type": "string"}}),
            &["ticker"],
        ),
        tool(
            "translate_text",
            "Translate text into a target language.",
            json!({"text": {"type": "string"}, "target_language": {"type": "string"}}),
            &["text", "target_language"],
        ),
        tool(
            "get_directions",
            "Get driving directions between two locations.",
            json!({"origin": {"type": "string"}, "destination": {"type": "string"}}),
            &["origin", "destination"],
        ),
        tool(
            "create_calendar_event",
            "Create a calendar event.",
            json!({"title": {"type": "string"}, "start": {"type": "string"}}),
            &["title", "start"],
        ),
        tool(
            "get_calendar_events",
            "List calendar events for a date.",
            json!({"date": {"type": "string"}}),
            &["date"],
        ),
        tool(
            "play_music",
            "Play a song or playlist.",
            json!({"query": {"type": "string"}}),
            &["query"],
        ),
        tool(
            "set_reminder",
            "Set a reminder at a given time.",
            json!({"text": {"type": "string"}, "time": {"type": "string"}}),
            &["text", "time"],
        ),
        tool(
            "get_air_quality",
            "Get the air quality index for a location.",
            json!({"location": {"type": "string"}}),
            &["location"],
        ),
        tool(
            "search_web",
            "Search the web for a query.",
            json!({"query": {"type": "string"}}),
            &["query"],
        ),
        tool(
            "convert_currency",
            "Convert an amount between currencies.",
            json!({"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}}),
            &["amount", "from", "to"],
        ),
        tool(
            "get_sports_score",
            "Get the latest score for a team.",
            json!({"team": {"type": "string"}}),
            &["team"],
        ),
        tool(
            "book_flight",
            "Book a flight between two cities.",
            json!({"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}),
            &["origin", "destination", "date"],
        ),
        tool(
            "get_traffic",
            "Get current traffic conditions for a location.",
            json!({"location": {"type": "string"}}),
            &["location"],
        ),
    ];
    let mut tools = vec![weather_tool()];
    tools.extend(distractors.clone());
    let distractor_names = distractors
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect::<HashSet<_>>();

    for &model in MODELS {
        if !model.tools {
            continue;
        }
        let mut turn1_request = request(vec![
            message("system", "You are a helpful assistant with tool calling. You have many tools available; select the single most appropriate one for the request."),
            message("user", "What's the weather in San Francisco?"),
        ]);
        turn1_request.core_tools = tools.clone();
        turn1_request.tool_choice = Some(json!("required"));
        turn1_request.reasoning = reasoning_for(model);
        let turn1 = run_stream(model, turn1_request).await;
        assert_or_record(
            model.template_type,
            "tool_selection",
            "turn1",
            &turn1.events,
        );

        assert_response_lifecycle(&turn1);
        assert_optional_reasoning(
            &turn1,
            model,
            "turn1",
            "expected at most one reasoning block",
        );
        assert!(
            !turn1.counts.contains_key("response.output_text.delta"),
            "turn1: leaked message text on a tool turn"
        );
        assert_eq!(
            added(&turn1, "function_call"),
            1,
            "turn1: expected exactly one function_call opened"
        );
        assert_eq!(
            count(&turn1, "response.function_call_arguments.done"),
            1,
            "turn1: expected one arguments.done"
        );
        assert_eq!(turn1.function_calls.len(), 1);
        let call = &turn1.function_calls[0];
        let opened = function_call_items_added(&turn1);
        assert_eq!(opened.len(), 1, "turn1: expected one function_call opened");
        assert_eq!(
            opened[0].name, "get_weather",
            "{}: selected the wrong tool out of 18: {:?}",
            model.template_type, opened[0].name
        );
        assert_eq!(opened[0].call_id, call.call_id);
        assert_eq!(
            opened[0].arguments, "",
            "turn1: function_call must open with empty arguments"
        );
        assert_eq!(opened[0].status, OutputStatus::InProgress);
        assert_eq!(
            call.name, "get_weather",
            "{}: picked a distractor: {:?}",
            model.template_type, call.name
        );
        assert!(!distractor_names.contains(call.name.as_str()));
        assert_eq!(call.status, OutputStatus::Completed);
        assert_eq!(
            parse_arguments(&call.arguments),
            json!({"location": "San Francisco"})
        );
        assert_eq!(
            turn1.field_args,
            HashMap::from([("location".to_string(), "San Francisco".to_string())]),
            "{}: per-argument field_path tagging wrong (expected location='San Francisco' value-only): {:?}",
            model.template_type,
            turn1.field_args
        );
    }
}

#[tokio::test]
async fn test_tool_chaining() {
    const KEY: &str = "K7-MAGENTA-9931";
    const CHEST_CONTENTS: &str = "a jade dragon figurine";
    let find_key = tool(
        "find_key",
        "Search a room and return the key hidden there.",
        json!({"room": {"type": "string"}}),
        &["room"],
    );
    let unlock_chest = tool(
        "unlock_chest",
        "Unlock the treasure chest with a key and return its contents.",
        json!({"key": {"type": "string"}}),
        &["key"],
    );
    let tools = vec![find_key, unlock_chest];

    for &model in MODELS {
        if !model.tools {
            continue;
        }
        let reasoning = reasoning_for(model);
        let mut conversation = vec![
            message("system", "You are a helpful assistant with tool calling. Use the tools in the right order, passing each tool's result into the next, then answer the request."),
            message("user", "Find the key hidden in the library, then unlock the treasure chest with it and tell me what's inside."),
        ];

        let mut turn1_request = request(conversation.clone());
        turn1_request.core_tools = tools.clone();
        turn1_request.tool_choice = Some(json!("required"));
        turn1_request.reasoning = reasoning.clone();
        let turn1 = run_stream(model, turn1_request).await;
        assert_or_record(model.template_type, "tool_chaining", "turn1", &turn1.events);

        assert_response_lifecycle(&turn1);
        assert_optional_reasoning(
            &turn1,
            model,
            "turn1",
            "expected at most one reasoning block",
        );
        assert!(
            !turn1.counts.contains_key("response.output_text.delta"),
            "turn1: leaked message text on a tool turn"
        );
        assert_eq!(
            added(&turn1, "function_call"),
            1,
            "turn1: expected exactly one function_call"
        );
        assert_eq!(count(&turn1, "response.function_call_arguments.done"), 1);
        assert_eq!(turn1.function_calls.len(), 1);
        let find = &turn1.function_calls[0];
        assert_eq!(
            find.name, "find_key",
            "turn1: must call find_key first, got {:?}",
            find.name
        );
        assert_eq!(find.status, OutputStatus::Completed);
        assert!(
            parse_arguments(&find.arguments)
                .get("room")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_lowercase()
                .contains("library"),
            "turn1: find_key should search the library: {:?}",
            find.arguments
        );

        conversation.push(function_call(&find.call_id, &find.name, &find.arguments));
        conversation.push(function_output(&find.call_id, json!({"key": KEY})));
        let mut turn2_request = request(conversation.clone());
        turn2_request.core_tools = tools.clone();
        turn2_request.tool_choice = Some(json!("required"));
        turn2_request.reasoning = reasoning.clone();
        let turn2 = run_stream(model, turn2_request).await;
        assert_or_record(model.template_type, "tool_chaining", "turn2", &turn2.events);

        assert_eq!(turn2.order.first().copied(), Some("response.created"));
        assert_eq!(turn2.order.last().copied(), Some("done"));
        assert_eq!(count(&turn2, "response.completed"), 1);
        assert_optional_reasoning(
            &turn2,
            model,
            "turn2",
            "expected at most one reasoning block",
        );
        assert!(
            !turn2.counts.contains_key("response.output_text.delta"),
            "turn2: leaked message text on a tool turn"
        );
        assert_eq!(
            added(&turn2, "function_call"),
            1,
            "turn2: expected exactly one function_call"
        );
        assert_eq!(turn2.function_calls.len(), 1);
        let unlock = &turn2.function_calls[0];
        assert_eq!(
            unlock.name, "unlock_chest",
            "turn2: must call unlock_chest, got {:?}",
            unlock.name
        );
        assert_eq!(unlock.status, OutputStatus::Completed);
        assert_eq!(
            parse_arguments(&unlock.arguments),
            json!({"key": KEY}),
            "{}: did not chain the returned key into unlock_chest: {:?}",
            model.template_type,
            unlock.arguments
        );
        assert_eq!(
            turn2.field_args,
            HashMap::from([("key".to_string(), KEY.to_string())]),
            "{}: key field_path tagging wrong: {:?}",
            model.template_type,
            turn2.field_args
        );

        conversation.push(function_call(
            &unlock.call_id,
            &unlock.name,
            &unlock.arguments,
        ));
        conversation.push(function_output(
            &unlock.call_id,
            json!({"contents": CHEST_CONTENTS}),
        ));
        let mut turn3_request = request(conversation);
        turn3_request.core_tools = tools.clone();
        turn3_request.tool_choice = Some(json!("none"));
        turn3_request.reasoning = reasoning;
        let turn3 = run_stream(model, turn3_request).await;
        assert_or_record(model.template_type, "tool_chaining", "turn3", &turn3.events);

        assert_eq!(turn3.order.first().copied(), Some("response.created"));
        assert_eq!(turn3.order.last().copied(), Some("done"));
        assert_eq!(count(&turn3, "response.completed"), 1);
        assert_optional_reasoning(
            &turn3,
            model,
            "turn3",
            "expected at most one reasoning block",
        );
        assert_eq!(
            count(&turn3, "response.function_call_arguments.done"),
            0,
            "turn3: unexpected tool call"
        );
        assert_message_lifecycle(&turn3, "turn3");
        let answer = turn3
            .content_done
            .as_deref()
            .unwrap_or_default()
            .to_lowercase();
        assert!(
            answer.contains("jade dragon") || answer.contains("figurine"),
            "{}: answer ignored the chest contents: {answer:?}",
            model.template_type
        );
    }
}

#[tokio::test]
async fn test_multi_tool() {
    let tools = vec![weather_tool(), get_time_tool()];
    const SYSTEM: &str = "You are a helpful assistant with tool calling. Call the tools you need to answer the request, then use their results to give the final answer.";

    for &model in MODELS {
        if !model.tools {
            continue;
        }
        let reasoning = reasoning_for(model);
        let mut conversation = vec![
            message("system", SYSTEM),
            message(
                "user",
                "What's the weather in San Francisco and what time is it in Tokyo?",
            ),
        ];

        let mut turn1_request = request(conversation.clone());
        turn1_request.core_tools = tools.clone();
        turn1_request.tool_choice = Some(json!("required"));
        turn1_request.reasoning = reasoning.clone();
        let turn1 = run_stream(model, turn1_request).await;
        assert_or_record(model.template_type, "multi_tool", "turn1", &turn1.events);

        assert_response_lifecycle(&turn1);
        assert_optional_reasoning(
            &turn1,
            model,
            "turn1",
            "expected at most one reasoning block",
        );
        assert!(
            !turn1.counts.contains_key("response.output_text.delta"),
            "turn1: leaked message text on a tool turn"
        );
        let n_calls = added(&turn1, "function_call");
        assert!(
            n_calls >= 1,
            "turn1: expected at least one function_call opened"
        );
        assert_eq!(
            count(&turn1, "response.function_call_arguments.done"),
            n_calls
        );
        assert_eq!(turn1.function_calls.len(), n_calls);
        let opened = function_call_items_added(&turn1);
        assert_eq!(
            opened.len(),
            n_calls,
            "turn1: opened function_call count mismatch"
        );
        let mut open_ids = HashSet::new();
        for item in &opened {
            assert!(
                ["get_weather", "get_time"].contains(&item.name.as_str()),
                "turn1: unexpected tool {:?}",
                item.name
            );
            assert_eq!(
                item.arguments, "",
                "turn1: function_call must open with empty arguments"
            );
            assert_eq!(item.status, OutputStatus::InProgress);
            open_ids.insert(item.call_id.clone());
        }
        assert_eq!(
            open_ids.len(),
            n_calls,
            "turn1: duplicate call_id across opened calls"
        );

        let mut by_name: HashMap<String, OutputFunctionCall> = HashMap::new();
        for call in &turn1.function_calls {
            assert_eq!(call.status, OutputStatus::Completed);
            assert!(
                open_ids.contains(&call.call_id),
                "turn1: done call_id has no matching open"
            );
            by_name.insert(call.name.clone(), call.clone());
        }
        if let Some(weather) = by_name.get("get_weather") {
            assert_eq!(
                parse_arguments(&weather.arguments),
                json!({"location": "San Francisco"})
            );
        }
        if let Some(time) = by_name.get("get_time") {
            assert_eq!(
                parse_arguments(&time.arguments),
                json!({"timezone": "Tokyo"})
            );
        }

        let results = HashMap::from([
            (
                "get_weather".to_string(),
                json!({"temperature": 65, "unit": "fahrenheit", "condition": "foggy"}),
            ),
            (
                "get_time".to_string(),
                json!({"time": "23:00", "timezone": "Tokyo", "utc_offset": "+09:00"}),
            ),
        ]);
        for call in &turn1.function_calls {
            conversation.push(function_call(&call.call_id, &call.name, &call.arguments));
            conversation.push(function_output(&call.call_id, results[&call.name].clone()));
        }

        let remaining = tools
            .iter()
            .filter(|tool| {
                let name = tool.get("name").and_then(Value::as_str).unwrap_or_default();
                !by_name.contains_key(name)
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut turn2_request = request(conversation.clone());
        turn2_request.core_tools = tools.clone();
        turn2_request.tool_choice = Some(json!(if remaining.is_empty() {
            "none"
        } else {
            "required"
        }));
        turn2_request.reasoning = reasoning.clone();
        let turn2 = run_stream(model, turn2_request).await;
        assert_or_record(model.template_type, "multi_tool", "turn2", &turn2.events);

        assert_response_lifecycle(&turn2);
        assert_optional_reasoning(
            &turn2,
            model,
            "turn2",
            "expected at most one reasoning block",
        );

        let final_turn = if remaining.is_empty() {
            assert_eq!(
                count(&turn2, "response.function_call_arguments.done"),
                0,
                "turn2: unexpected tool call"
            );
            assert_message_lifecycle(&turn2, "turn2");
            assert_eq!(
                by_name.keys().cloned().collect::<HashSet<_>>(),
                HashSet::from(["get_weather".to_string(), "get_time".to_string()]),
                "turn1: parallel arch must emit both calls"
            );
            turn2
        } else {
            assert!(
                added(&turn2, "function_call") >= 1,
                "turn2: expected the remaining tool call"
            );
            assert_eq!(
                count(&turn2, "response.output_text.done"),
                0,
                "turn2: leaked message on a tool turn"
            );
            for call in &turn2.function_calls {
                assert_eq!(call.status, OutputStatus::Completed);
                let remaining_names = remaining
                    .iter()
                    .filter_map(|tool| tool.get("name").and_then(Value::as_str))
                    .collect::<HashSet<_>>();
                assert!(
                    remaining_names.contains(call.name.as_str()),
                    "turn2: unexpected tool {:?}",
                    call.name
                );
                by_name.insert(call.name.clone(), call.clone());
                conversation.push(function_call(&call.call_id, &call.name, &call.arguments));
                conversation.push(function_output(&call.call_id, results[&call.name].clone()));
            }
            assert_eq!(
                parse_arguments(&by_name["get_weather"].arguments),
                json!({"location": "San Francisco"})
            );
            assert_eq!(
                parse_arguments(&by_name["get_time"].arguments),
                json!({"timezone": "Asia/Tokyo"})
            );

            let mut turn3_request = request(conversation);
            turn3_request.core_tools = tools.clone();
            turn3_request.tool_choice = Some(json!("none"));
            turn3_request.reasoning = reasoning;
            let turn3 = run_stream(model, turn3_request).await;
            assert_or_record(model.template_type, "multi_tool", "turn3", &turn3.events);
            assert_eq!(turn3.order.first().copied(), Some("response.created"));
            assert_eq!(turn3.order.last().copied(), Some("done"));
            assert_eq!(count(&turn3, "response.completed"), 1);
            assert_optional_reasoning(
                &turn3,
                model,
                "turn3",
                "expected at most one reasoning block",
            );
            assert_eq!(
                count(&turn3, "response.function_call_arguments.done"),
                0,
                "turn3: unexpected tool call"
            );
            assert_message_lifecycle(&turn3, "turn3");
            turn3
        };

        let answer = final_turn
            .content_done
            .as_deref()
            .unwrap_or_default()
            .to_lowercase();
        assert!(
            answer.contains("65") || answer.contains("fog"),
            "{}: final answer dropped weather result: {answer:?}",
            model.template_type
        );
        assert!(
            answer.contains("23:00") || answer.contains("11") || answer.contains("tokyo"),
            "{}: final answer dropped time result: {answer:?}",
            model.template_type
        );
    }
}
