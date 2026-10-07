//! Finite own-data producer. No external teachers, weights, optimizer or GPU.
use rz_arena::pals_collect::{
    OwnCpuCollectionDriver, OwnPalsMockCollectionDriver, PalsCollectionConfig,
    PalsCollectionDriver, collect_pals_own_data, own_collection_registry,
};
use rz_experiments::PalsSplit;
use rz_search::{cpu::CpuConfig, pals::engine::PalsConfig};
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
};

fn usage() {
    println!(
        "pals_collect --output-root ABSOLUTE_OUTSIDE_REPOSITORY [--mode cpu|pals-mock]\n\
        [--run-id ID] [--games N] [--max-plies N] [--depth N] [--nodes-per-job N]\n\
        [--max-total-nodes N] [--max-seconds N] [--max-output-bytes N] [--seed N]\n\
        [--exploration-plies N] [--no-conditional] [--fen FEN]\n\
        [--opening-moves \"e2e4 e7e5\"] [--opening-id ID] [--split train|validation|holdout]\n\
        CPU/mock sources have no neural weights. Limits are mandatory finite defaults.\n\
        Output preserves seals, native Rust Rules tensors, own observations, masks, PGN, and failure receipt."
    );
}
fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let mut config = PalsCollectionConfig::default();
    let mut output = None;
    let mut mode = "cpu".to_owned();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            usage();
            return Ok(true);
        }
        if flag == "--no-conditional" {
            config.collect_conditional_repair = false;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--output-root" => output = Some(PathBuf::from(value)),
            "--mode" => mode = value,
            "--run-id" => config.run_id = value,
            "--games" => config.games = value.parse()?,
            "--max-plies" => config.max_plies = value.parse()?,
            "--depth" => config.cpu_depth = value.parse()?,
            "--nodes-per-job" => config.nodes_per_job = value.parse()?,
            "--max-total-nodes" => config.max_total_nodes = value.parse()?,
            "--max-seconds" => {
                config.max_wall_time_ms = value
                    .parse::<u64>()?
                    .checked_mul(1000)
                    .ok_or("wall time overflow")?
            }
            "--max-output-bytes" => config.max_output_bytes = value.parse()?,
            "--seed" => config.seed = value.parse()?,
            "--exploration-plies" => config.exploration_plies = value.parse()?,
            "--fen" => config.openings[0].initial_fen = Some(value),
            "--opening-moves" => {
                config.openings[0].moves = value.split_whitespace().map(str::to_owned).collect()
            }
            "--opening-id" => config.openings[0].id = value,
            "--split" => {
                config.openings[0].split = match value.as_str() {
                    "train" => PalsSplit::Train,
                    "validation" => PalsSplit::Validation,
                    "holdout" => PalsSplit::Holdout,
                    _ => return Err("unsupported split".into()),
                }
            }
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    let output =
        output.ok_or("--output-root is required and must be absolute outside the checkout")?;
    config.validate()?;
    let mut driver: Box<dyn PalsCollectionDriver> = match mode.as_str() {
        "cpu" => Box::new(OwnCpuCollectionDriver::new(CpuConfig::default())?),
        "pals-mock" => Box::new(OwnPalsMockCollectionDriver::new(
            CpuConfig::default(),
            PalsConfig::default(),
        )?),
        _ => return Err("mode must be cpu or pals-mock; no implicit provider fallback".into()),
    };
    let registry = own_collection_registry(driver.as_ref())?;
    let cancelled = Arc::new(AtomicBool::new(false));
    #[cfg(unix)]
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signal_hook::flag::register(signal, Arc::clone(&cancelled))?;
    }
    let receipt = collect_pals_own_data(
        config,
        &output,
        driver.as_mut(),
        &registry,
        cancelled.as_ref(),
    )?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(receipt.complete)
}
fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(2),
        Err(error) => {
            eprintln!("pals_collect: {error}");
            std::process::exit(2);
        }
    }
}
