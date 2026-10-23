//! Sliding-window regression: resume a prompt longer than the window from a cached
//! tail at every token offset within a page, and check the output matches a run with
//! the prefix cache off. The resume position that lands one token short of a page
//! boundary is the one where the prefill kernel still reads the block just below the
//! attention window.
//!
//!   MODEL=google/gemma-4-E2B-it cargo run --release --example window_boundary
//!
//! Exits non-zero when any run differs from its cache-off run.

mod common;

use common::{msg, run_turn, Message};
use orchard::{Client, SamplingParams};

async fn run(
    client: &Client,
    model: &str,
    messages: Vec<Message>,
    prefix_cache: bool,
) -> Result<(Vec<i32>, usize, usize, Vec<String>), common::Error> {
    let params = SamplingParams {
        max_tokens: 24,
        temperature: 0.0,
        reasoning: Some(false),
        prefix_cache: Some(prefix_cache),
        deterministic: true,
        top_logprobs: 3,
        ..Default::default()
    };
    let mut tops = Vec::new();
    let turn = run_turn(client, model, messages, params, |d| {
        if !d.top_logprobs.is_empty() {
            tops.push(
                d.top_logprobs
                    .iter()
                    .map(|t| format!("{:?} {:.4}", t.token, t.logprob))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
    })
    .await?;
    Ok((turn.ids, turn.prompt_tokens, turn.cached, tops))
}

#[tokio::main]
async fn main() -> Result<(), common::Error> {
    let model = std::env::var("MODEL").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let page: usize = 16;
    let (_engine, registry, client) = common::connect().await?;
    registry.ensure_loaded(&model).await?;

    // Comfortably past a 512-token window.
    let repeats: usize = std::env::var("BACKGROUND_REPEATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(44);
    // UNIQUE=1 numbers every sentence so no two pages hold the same tokens.
    let background: String = if std::env::var("UNIQUE").is_ok() {
        (0..repeats)
            .map(|i| {
                format!(
                    "In year {} the harbour town logged tide {} and storm {}. ",
                    1700 + i * 7,
                    i * 13 + 5,
                    i * 3 + 1
                )
            })
            .collect()
    } else {
        "The harbour town kept careful records of every tide, ship and storm. ".repeat(repeats)
    };
    let fillers = [
        "red", "blue", "green", "gold", "grey", "pink", "teal", "plum", "lime", "rust", "sand",
        "jade", "rose", "wine", "mint", "coal", "snow", "moss", "clay", "fern",
    ];

    println!("prompt tok | resumed at | offset in page | matches cache-off run");
    let (mut mismatches, mut hit_boundary) = (0, false);
    // ONLY=3,4 restricts the sweep to those filler counts, for reruns.
    let only: Option<Vec<usize>> = std::env::var("ONLY")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect());
    for n in (0..fillers.len()).filter(|n| only.as_ref().is_none_or(|o| o.contains(n))) {
        let question = format!(
            "{background}\nIn one sentence, what did the town record? Ignore these words: {}.",
            fillers[..n].join(" ")
        );
        let messages = vec![msg("user", &question)];
        let (first, _, first_cached, first_tops) =
            run(&client, &model, messages.clone(), true).await?;
        let (warm, prompt, cached, warm_tops) =
            run(&client, &model, messages.clone(), true).await?;
        let (cold, _, _, cold_tops) = run(&client, &model, messages, false).await?;
        let same = warm == cold;
        mismatches += usize::from(!same || first != cold);
        hit_boundary |= cached % page == page - 1;
        println!(
            "{prompt:>10} | {cached:>10} | {:>14} | {}",
            cached % page,
            if same { "yes" } else { "NO" }
        );
        if std::env::var("SUMMARY").is_ok() {
            println!(
                "SUMMARY prompt {prompt} first-resumed-at {first_cached} first==cold {} warm==cold {}",
                first == cold && first_tops == cold_tops,
                warm == cold && warm_tops == cold_tops
            );
        }
        if std::env::var("VERBOSE").is_ok() {
            for (label, tops) in [
                ("first", &first_tops),
                ("warm ", &warm_tops),
                ("cold ", &cold_tops),
            ] {
                println!(
                    "           {label} resumed/step1: {}",
                    tops.get(1).cloned().unwrap_or_default()
                );
            }
        }
        if first != cold {
            println!("           the first warm run (resumed at {first_cached}) already differs from the cache-off run");
        }
        if !same {
            let at = warm.iter().zip(cold.iter()).position(|(a, b)| a != b);
            println!("           first difference at generated token {at:?}");
            for (label, tops) in [("warm", &warm_tops), ("cold", &cold_tops)] {
                for (step, top) in tops.iter().enumerate().take(at.unwrap_or(0) + 1) {
                    println!("           {label} step {step}: {top}");
                }
            }
        }
    }
    println!(
        "RESULT: {mismatches} mismatches; resume one token short of a page boundary {}",
        if hit_boundary {
            "was exercised"
        } else {
            "was NOT exercised"
        }
    );
    // Whether the boundary resume was exercised is information only.
    common::verdict(mismatches, "prompt lengths differ from the cache-off run")
}
