//! End-to-end moondream capabilities tests.
//!
//! Mirrors orchard-py/tests/test_e2e_capabilities.py
//! Run with: cargo test --test e2e -- --ignored
//!
//! Note: MoondreamClient takes ownership of Client, so we create a new IPC connection
//! per test. This mirrors Python's behavior where each test gets a fresh client.

use std::path::PathBuf;
use std::sync::Arc;

use base64::Engine;
use orchard::client::{MoondreamClient, SpatialRef};
use orchard::{Client, ModelRegistry, SamplingParams};

use crate::fixture::get_fixture;

fn get_test_assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/assets")
}

fn load_image_data_url(filename: &str) -> String {
    let path = get_test_assets_dir().join(filename);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("Failed to read test image {}: {}", path.display(), e));
    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
    format!("data:image/jpeg;base64,{}", encoded)
}

async fn create_moondream_client(registry: Arc<ModelRegistry>) -> MoondreamClient {
    let client = Client::connect(Arc::clone(&registry))
        .await
        .expect("Failed to connect client");
    MoondreamClient::new(client, registry)
        .await
        .expect("Failed to create MoondreamClient")
}

/// Test moondream reasoning with grounding on bottles image.
/// Mirrors: test_e2e_capabilities.py::test_moondream_reasoning_grounding
#[tokio::test]
#[ignore]
async fn test_moondream_reasoning_grounding() {
    let fixture = get_fixture().await;
    let moondream = create_moondream_client(Arc::clone(&fixture.registry)).await;

    let image_data_url = load_image_data_url("bottles.jpg");

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    let result = moondream
        .query(
            "How many bottles are shown?",
            Some(&image_data_url),
            &[],
            true, // reasoning=True
            params,
        )
        .await;

    assert!(result.is_ok(), "Query failed: {:?}", result.err());
    let response = result.unwrap();

    let answer_text = response.answer.to_lowercase();
    println!("{}", answer_text);

    assert!(
        answer_text.contains('6'),
        "Answer text should contain '6', but got {}",
        answer_text
    );

    // Check reasoning output
    if let Some(reasoning) = &response.reasoning {
        let reasoning_text = reasoning.text.to_lowercase();
        println!("{}", reasoning_text);

        let grounding = &reasoning.grounding;
        println!("{:?}", grounding);

        assert!(
            !grounding.is_empty(),
            "Grounding output should not be empty"
        );

        for ground in grounding {
            assert!(!ground.points.is_empty(), "Should have points");
        }

        assert!(
            reasoning_text.contains("duckhorn"),
            "Model should mention the label on the bottle, but got:\n{}",
            reasoning_text
        );
    }
}

/// Test moondream caption generation.
/// Mirrors: test_e2e_capabilities.py::test_moondream_caption
#[tokio::test]
#[ignore]
async fn test_moondream_caption_normal() {
    run_moondream_caption("normal").await;
}

#[tokio::test]
#[ignore]
async fn test_moondream_caption_short() {
    run_moondream_caption("short").await;
}

#[tokio::test]
#[ignore]
async fn test_moondream_caption_long() {
    run_moondream_caption("long").await;
}

async fn run_moondream_caption(length: &str) {
    let fixture = get_fixture().await;
    let moondream = create_moondream_client(Arc::clone(&fixture.registry)).await;

    let image_data_url = load_image_data_url("apple.jpg");

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    let result = moondream.caption(&image_data_url, length, params).await;

    assert!(result.is_ok(), "Caption failed: {:?}", result.err());
    let response = result.unwrap();

    assert!(!response.caption.is_empty());
    let caption = response.caption.to_lowercase();
    assert!(caption.contains("apple"), "Caption should contain 'apple'");
    println!("{} caption: {}", length, caption);
}

/// Test moondream object detection.
/// Mirrors: test_e2e_capabilities.py::test_moondream_detect
#[tokio::test]
#[ignore]
async fn test_moondream_detect() {
    let fixture = get_fixture().await;
    let moondream = create_moondream_client(Arc::clone(&fixture.registry)).await;

    let image_data_url = load_image_data_url("apple.jpg");

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    let result = moondream.detect(&image_data_url, "apple", params).await;

    assert!(result.is_ok(), "Detect failed: {:?}", result.err());
    let response = result.unwrap();

    assert!(!response.objects.is_empty());
    println!("{:?}", response.objects);
}

/// Test moondream query with spatial reference (point).
/// Mirrors: test_e2e_capabilities.py::test_moondream_query_with_spatial_refs
#[tokio::test]
#[ignore]
async fn test_moondream_query_with_spatial_refs() {
    let fixture = get_fixture().await;
    let moondream = create_moondream_client(Arc::clone(&fixture.registry)).await;

    let image_data_url = load_image_data_url("apple.jpg");

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    let spatial_refs: Vec<SpatialRef> = vec![(0.5, 0.5).into()];

    let result = moondream
        .query(
            "What is at this location?",
            Some(&image_data_url),
            &spatial_refs,
            false,
            params,
        )
        .await;

    assert!(result.is_ok(), "Query failed: {:?}", result.err());
    let response = result.unwrap();

    let answer = response.answer.to_lowercase();
    println!("{}", answer);
    assert!(answer.contains("apple"), "Answer should contain 'apple'");
}

/// Test moondream pointing to objects.
/// Mirrors: test_e2e_capabilities.py::test_moondream_point
#[tokio::test]
#[ignore]
async fn test_moondream_point() {
    let fixture = get_fixture().await;
    let moondream = create_moondream_client(Arc::clone(&fixture.registry)).await;

    let image_data_url = load_image_data_url("moondream.jpg");

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    let result = moondream.point(&image_data_url, "Eyes", params).await;

    assert!(result.is_ok(), "Point failed: {:?}", result.err());
    let response = result.unwrap();

    assert!(!response.points.is_empty());
    println!("{:?}", response.points);
}

/// Test moondream gaze detection.
/// Mirrors: test_e2e_capabilities.py::test_moondream_detect_gaze
#[tokio::test]
#[ignore]
async fn test_moondream_detect_gaze() {
    let fixture = get_fixture().await;
    let moondream = create_moondream_client(Arc::clone(&fixture.registry)).await;

    let image_data_url = load_image_data_url("moondream.jpg");

    // Eye position - center of the character's face area
    let eye_position = (0.5419921875, 0.5419921875);

    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: 100,
        ..Default::default()
    };

    let result = moondream
        .detect_gaze(&image_data_url, eye_position, params)
        .await;

    assert!(result.is_ok(), "Detect gaze failed: {:?}", result.err());
    let response = result.unwrap();

    assert!(response.gaze.is_some(), "Should have gaze result");
    let gaze = response.gaze.unwrap();

    // Baseline implementation returns center of image
    assert!((gaze.x - 0.5).abs() < 0.01, "Gaze x should be ~0.5");
    assert!((gaze.y - 0.5).abs() < 0.01, "Gaze y should be ~0.5");
    println!("Gaze: ({}, {})", gaze.x, gaze.y);
}
