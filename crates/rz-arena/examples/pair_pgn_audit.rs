//! Bounded post-run PGN audit using the existing A-owned Rules implementation.
//! This checks paired games; it grants no launch, timing or strength authority.
use rz_arena::{PairSpec, PgnLimits, PgnOutcomePolicy, audit_pair_pgn_for_spec, decode_json};
use rz_experiments::OutcomePolicy;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    pair: PairSpec,
    max_plies: u32,
}

fn read_bounded(path: &Path, limit: u64) -> Result<String, Box<dyn std::error::Error>> {
    let mut text = String::new();
    File::open(path)?
        .take(limit + 1)
        .read_to_string(&mut text)?;
    if text.len() as u64 > limit {
        return Err("audit input exceeds its byte ceiling".into());
    }
    Ok(text)
}

fn execute() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("Usage: pair_pgn_audit PAIR_INPUT_JSON TWO_GAME_PGN".into());
    }
    let input: Input = decode_json(&read_bounded(Path::new(&args[0]), 64 * 1024)?)?;
    if input.schema_version != 1 {
        return Err("unsupported pair audit schema".into());
    }
    let pgn = read_bounded(Path::new(&args[1]), 4 * 1024 * 1024)?;
    let audit = audit_pair_pgn_for_spec(
        &input.pair,
        &pgn,
        PgnLimits {
            max_bytes: 4 * 1024 * 1024,
            max_plies: input.max_plies,
        },
        PgnOutcomePolicy {
            engine_failure: OutcomePolicy::Loss,
            max_plies_outcome: OutcomePolicy::Incomplete,
            max_game_plies: input.max_plies,
        },
    )?;
    println!("{}", serde_json::to_string_pretty(&audit)?);
    Ok(())
}

fn main() -> std::process::ExitCode {
    match execute() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("pair_pgn_audit: {error}");
            std::process::ExitCode::from(2)
        }
    }
}
