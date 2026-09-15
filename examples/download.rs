//! Download (or resume) model weights into the HF cache using Orchard's own
//! resolver, so `ensure_loaded` finds them exactly where it looks.
//!
//!   cargo run --release --example download -- google/gemma-4-26B-A4B-it Qwen/Qwen3.6-35B-A3B

use std::time::Instant;

use orchard::ModelResolver;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ids: Vec<String> = std::env::args().skip(1).collect();
    if ids.is_empty() {
        eprintln!("usage: download <hf-repo-id>...");
        std::process::exit(2);
    }
    let mut resolver = ModelResolver::new()?;
    for id in ids {
        let t = Instant::now();
        eprintln!("[{id}] resolving/downloading...");
        let resolved = resolver.resolve(&id).await?;
        eprintln!(
            "[{id}] ready in {:.0}s -> {} (source: {})",
            t.elapsed().as_secs_f64(),
            resolved.model_path.display(),
            resolved.source
        );
    }
    Ok(())
}
