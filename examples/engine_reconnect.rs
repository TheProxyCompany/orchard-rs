//! Exercise rapid sequential client leases against an isolated real engine.
use orchard::InferenceEngine;
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    for cycle in 0..10 {
        let mut engine = InferenceEngine::new().await?;
        engine.close()?;
        println!("admitted and released {cycle}");
    }
    Ok(())
}
