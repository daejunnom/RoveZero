//! Finite own-data producer. No external teachers, optimizer or cloud launch.
use rz_arena::pals_collect::{
    OwnCpuCollectionDriver, OwnPalsMockCollectionDriver, PalsCollectionConfig,
    PalsCollectionDriver, collect_pals_own_data, own_collection_registry,
    pals_collection_registration_description, validate_pals_collection_output,
};
#[cfg(feature = "pals-collection-onnx")]
use rz_arena::pals_collect::{OwnPalsOnnxCollectionDriver, PalsNativeCollectionRegistry};
#[cfg(feature = "pals-collection-onnx")]
use rz_eval::runtime_pin::RuntimeCache;
use rz_experiments::PalsSplit;
use rz_search::{cpu::CpuConfig, pals::engine::PalsConfig};
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
};

fn usage() {
    println!(
        "pals_collect --output-root ABSOLUTE_OUTSIDE_REPOSITORY [--mode cpu|pals-mock|pals-onnx]\n\
        [--run-id ID] [--games N] [--max-plies N] [--depth N] [--nodes-per-job N]\n\
        [--max-total-nodes N] [--max-seconds N] [--max-output-bytes N] [--seed N]\n\
        [--exploration-plies N] [--no-conditional] [--fen FEN]\n\
        [--opening-moves \"e2e4 e7e5\"] [--opening-id ID] [--split train|validation|holdout]\n\
        pals-onnx requires feature pals-collection-onnx, --provider cpu, --export ABSOLUTE,\n\
        --checkpoint ABSOLUTE --runtime ABSOLUTE --source-registry ABSOLUTE\n\
        --source-registry-sha256 SHA256 [--runtime-cache ABSOLUTE]\n\
        [--beam-width N] [--line-plies N] [--max-role-calls N] [--pals-rounds N]\n\
        CPU/mock sources have no neural weights. Limits are mandatory finite defaults.\n\
        Actual ONNX mode accepts independently registered Untrained P/C on explicit CPU only.\n\
        --describe-registration prints read-only executable/CPU/encoder/model-config facts and exits.\n\
        Output preserves seals, actual native tensors, physical raw/delivery/consumption, masks, PGN and failure receipt."
    );
}
fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let mut config = PalsCollectionConfig::default();
    let mut output = None;
    let mut mode = "cpu".to_owned();
    let mut provider = None;
    let mut export = None;
    let mut checkpoint = None;
    let mut runtime = None;
    let mut runtime_cache = None;
    let mut source_registry = None;
    let mut source_registry_sha256 = None;
    let mut pals = PalsConfig::default();
    let mut pals_rounds = 2_u64;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--describe-registration" {
            println!(
                "{}",
                serde_json::to_string(&pals_collection_registration_description()?)?
            );
            return Ok(true);
        }
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
            "--provider" => provider = Some(value),
            "--export" => export = Some(PathBuf::from(value)),
            "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
            "--runtime" => runtime = Some(PathBuf::from(value)),
            "--runtime-cache" => runtime_cache = Some(PathBuf::from(value)),
            "--source-registry" => source_registry = Some(PathBuf::from(value)),
            "--source-registry-sha256" => source_registry_sha256 = Some(value),
            "--beam-width" => pals.beam_width = value.parse()?,
            "--line-plies" => pals.line_plies = value.parse()?,
            "--max-role-calls" => pals.max_role_calls = value.parse()?,
            "--pals-rounds" => pals_rounds = value.parse()?,
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
    if !(1..=16).contains(&pals_rounds) {
        return Err("--pals-rounds must be 1..=16".into());
    }
    validate_pals_collection_output(&output, &config.run_id)?;
    let native_flags = provider.is_some()
        || export.is_some()
        || checkpoint.is_some()
        || runtime.is_some()
        || runtime_cache.is_some()
        || source_registry.is_some()
        || source_registry_sha256.is_some();
    if mode != "pals-onnx" && native_flags {
        return Err(
            "native assets/provider flags require pals-onnx; no weights source downcast".into(),
        );
    }
    let (mut driver, registry): (Box<dyn PalsCollectionDriver>, _) = match mode.as_str() {
        "cpu" => {
            let driver = Box::new(OwnCpuCollectionDriver::new(CpuConfig::default())?);
            let registry = own_collection_registry(driver.as_ref())?;
            (driver, registry)
        }
        "pals-mock" => {
            let driver = Box::new(OwnPalsMockCollectionDriver::new(
                CpuConfig::default(),
                pals,
            )?);
            let registry = own_collection_registry(driver.as_ref())?;
            (driver, registry)
        }
        "pals-onnx" => {
            #[cfg(feature = "pals-collection-onnx")]
            {
                if provider.as_deref() != Some("cpu") {
                    return Err("pals-onnx requires explicit --provider cpu; GPU/fallback collection not admitted".into());
                }
                let expected = PalsNativeCollectionRegistry::read_pinned(
                    source_registry
                        .as_deref()
                        .ok_or("--source-registry is required")?,
                    source_registry_sha256
                        .as_deref()
                        .ok_or("--source-registry-sha256 is required")?,
                )?;
                let cache = match runtime_cache {
                    Some(path) => RuntimeCache::open(&path)?,
                    None => RuntimeCache::for_user()?,
                };
                let runtime = runtime.ok_or("--runtime is required")?;
                if !runtime.is_absolute() {
                    return Err("--runtime must be absolute".into());
                }
                let pin = cache.library(&runtime, &expected.runtime_sha256)?;
                let driver = Box::new(OwnPalsOnnxCollectionDriver::load_cpu(
                    export.as_deref().ok_or("--export is required")?,
                    checkpoint.as_deref().ok_or("--checkpoint is required")?,
                    &pin,
                    &expected,
                    CpuConfig::default(),
                    pals,
                    pals_rounds,
                )?);
                (driver, expected.owned_sources()?)
            }
            #[cfg(not(feature = "pals-collection-onnx"))]
            {
                return Err(
                    "pals-onnx requires feature pals-collection-onnx; no implicit mock fallback"
                        .into(),
                );
            }
        }
        _ => {
            return Err(
                "mode must be cpu, pals-mock or pals-onnx; no implicit provider fallback".into(),
            );
        }
    };
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
