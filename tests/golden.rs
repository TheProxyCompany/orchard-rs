//! Golden tests mirroring orchard-py/tests/golden/*.py.

#[path = "project/fixture.rs"]
mod fixture;
#[path = "golden/golden_io.rs"]
mod golden_io;

#[path = "golden/golden_path.rs"]
mod golden_path;

#[path = "golden/drift_repro.rs"]
mod drift_repro;

#[ctor::ctor]
fn preload_modal_models() {
    fixture::PRELOAD_MODAL_MODELS.store(true, std::sync::atomic::Ordering::Relaxed);
}

#[test]
fn verified_image_streams_require_a_complete_match() {
    let path = golden_io::golden_path("moondream3", "image_edit_tool_blind_verifier_flux");
    let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let approved = &fixture["verifier"];
    let variants = approved.as_array().unwrap();
    let diagnostics = tempfile::tempdir().unwrap();
    for variant in variants {
        golden_io::compare_strictly(
            approved,
            variant.as_array().unwrap(),
            "image edit",
            diagnostics.path(),
        )
        .unwrap();
    }
    let mut altered = variants[1].as_array().unwrap().clone();
    altered[11] = variants[0][11].clone();
    assert!(
        golden_io::compare_strictly(approved, &altered, "image edit", diagnostics.path()).is_err()
    );
}
