//! Offline accounting of a pinned runner's startup failure, never a successful pair.
use rz_arena::{PairSpec, audit_pilot_startup_failure_trace, decode_json};
use std::{fs::File, io::Read, path::Path};

fn bounded(path: &Path, limit: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("failure audit input exceeds its byte ceiling".into());
    }
    Ok(bytes)
}
fn execute() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("Usage: pilot_failure_audit PAIR_SPEC_JSON RUNNER_STDOUT_LOG".into());
    }
    let pair: PairSpec = decode_json(std::str::from_utf8(&bounded(
        Path::new(&args[0]),
        64 * 1024,
    )?)?)?;
    let stdout = bounded(Path::new(&args[1]), 64 * 1024 * 1024)?;
    let audit = audit_pilot_startup_failure_trace(&stdout, &pair)?;
    println!("{}", serde_json::to_string_pretty(&audit)?);
    Ok(())
}
fn main() -> std::process::ExitCode {
    match execute() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("pilot_failure_audit: {error}");
            std::process::ExitCode::from(2)
        }
    }
}
