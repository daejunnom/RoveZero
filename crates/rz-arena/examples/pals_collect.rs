//! Finite own-data producer. No external teachers, optimizer or cloud launch.
use rz_arena::pals_collect::{
    OwnCpuCollectionDriver, OwnPalsMockCollectionDriver, PalsCollectionConfig,
    PalsCollectionDriver, PalsProducerCollectionConfig, collect_pals_own_data_with_producer,
    own_collection_registry, pals_collection_registration_description,
    pals_producer_registration_description, validate_pals_collection_output,
};
#[cfg(feature = "pals-collection-onnx")]
use rz_arena::pals_collect::{
    OwnPalsOnnxCollectionDriver, PalsNativeCollectionRegistry, PalsNativeContinuationRegistration,
    PalsNativeRefinementRegistration,
};
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
        [--producer-registration ABSOLUTE --producer-registration-sha256 SHA256]\n\
        [--producer-journal-bytes N --producer-capture-bytes N]\n\
        [--post-repair-recheck same-repaired-line-once-v1|actual-opponent-continuation-v1\n\
         --refinement-registration ABSOLUTE --refinement-registration-sha256 SHA256]\n\
        CPU/mock sources have no neural weights. Limits are mandatory finite defaults.\n\
        Actual ONNX mode accepts independently registered Untrained P/C on explicit CPU only.\n\
        --describe-registration prints read-only executable/CPU/encoder/model-config facts and exits.\n\
        --describe-producer ID constructs the checked CPU/native owner, prints registration facts and exits.\n\
        Strict producer mode requires independent prior registration bytes; failure has no legacy fallback.\n\
        Output preserves seals, actual native tensors, physical raw/delivery/consumption, masks, PGN and failure receipt."
    );
}

// Selection admission precedes every owner/runtime/model dispatch. The only
// non-strict exception is read-only producer-registration bootstrap metadata.
fn validate_refinement_cli(
    mode: &str,
    provider: Option<&str>,
    policy: Option<&str>,
    registration: Option<&PathBuf>,
    registration_sha256: Option<&str>,
    strict_producer: bool,
    describe_producer: bool,
) -> Result<bool, String> {
    match (policy, registration, registration_sha256) {
        (None, None, None) => Ok(false),
        (Some("same-repaired-line-once-v1" | "actual-opponent-continuation-v1"), Some(path), Some(sha)) => {
            if mode != "pals-onnx" || provider != Some("cpu") || !path.is_absolute()
                || sha.len() != 64
                || !sha.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                return Err("explicit recheck requires CPU pals-onnx, an absolute registration and exact lowercase SHA256".into());
            }
            if !strict_producer && !describe_producer {
                return Err("explicit recheck collection requires independent strict producer registration".into());
            }
            Ok(true)
        }
        _ => Err("post-Repair selection requires the exact policy, registration path and independent SHA256 together".into()),
    }
}

fn set_refinement_arg<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("duplicate {flag}"));
    }
    *slot = Some(value);
    Ok(())
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
    let mut post_repair_recheck = None;
    let mut refinement_registration = None;
    let mut refinement_registration_sha256 = None;
    let mut producer_registration = None;
    let mut producer_registration_sha256 = None;
    let mut producer_journal_bytes = None;
    let mut producer_capture_bytes = None;
    let mut describe_producer = None;
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
            "--post-repair-recheck" => {
                set_refinement_arg(&mut post_repair_recheck, value, &flag)?;
            }
            "--refinement-registration" => {
                set_refinement_arg(&mut refinement_registration, PathBuf::from(value), &flag)?;
            }
            "--refinement-registration-sha256" => {
                set_refinement_arg(&mut refinement_registration_sha256, value, &flag)?;
            }
            "--producer-registration" => producer_registration = Some(PathBuf::from(value)),
            "--producer-registration-sha256" => producer_registration_sha256 = Some(value),
            "--producer-journal-bytes" => producer_journal_bytes = Some(value.parse::<u64>()?),
            "--producer-capture-bytes" => producer_capture_bytes = Some(value.parse::<u64>()?),
            "--describe-producer" => describe_producer = Some(value),
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
    let recheck_selected = validate_refinement_cli(
        &mode,
        provider.as_deref(),
        post_repair_recheck.as_deref(),
        refinement_registration.as_ref(),
        refinement_registration_sha256.as_deref(),
        producer_registration.is_some() && producer_registration_sha256.is_some(),
        describe_producer.is_some(),
    )?;
    if describe_producer.is_some()
        && (producer_registration.is_some()
            || producer_registration_sha256.is_some()
            || producer_journal_bytes.is_some()
            || producer_capture_bytes.is_some())
    {
        return Err("--describe-producer cannot enroll or consume a strict registration".into());
    }
    let producer_config = match (producer_registration, producer_registration_sha256) {
        (None, None) => {
            if producer_journal_bytes.is_some() || producer_capture_bytes.is_some() {
                return Err("producer metadata limits require independent registration".into());
            }
            None
        }
        (Some(path), Some(sha256)) => Some(
            PalsProducerCollectionConfig::read_pinned(&path, &sha256)?.with_metadata_limits(
                producer_journal_bytes.unwrap_or(512 * 1024),
                producer_capture_bytes.unwrap_or(512 * 1024),
            )?,
        ),
        _ => {
            return Err(
                "producer registration path and independent SHA256 are both required".into(),
            );
        }
    };
    if describe_producer.is_none() && output.is_none() {
        return Err("--output-root is required and must be absolute outside the checkout".into());
    }
    config.validate()?;
    if !(1..=16).contains(&pals_rounds) {
        return Err("--pals-rounds must be 1..=16".into());
    }
    if let Some(output) = &output {
        validate_pals_collection_output(output, &config.run_id)?;
    }
    let native_flags = provider.is_some()
        || export.is_some()
        || checkpoint.is_some()
        || runtime.is_some()
        || runtime_cache.is_some()
        || source_registry.is_some()
        || source_registry_sha256.is_some()
        || recheck_selected;
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
                let continuation_selected =
                    post_repair_recheck.as_deref() == Some("actual-opponent-continuation-v1");
                let refinement = if recheck_selected && !continuation_selected {
                    let registration = PalsNativeRefinementRegistration::read_pinned(
                        refinement_registration
                            .as_deref()
                            .ok_or("--refinement-registration is required")?,
                        refinement_registration_sha256
                            .as_deref()
                            .ok_or("--refinement-registration-sha256 is required")?,
                    )?;
                    registration.validate_against(&expected)?;
                    Some(registration)
                } else {
                    None
                };
                let continuation = if continuation_selected {
                    let registration = PalsNativeContinuationRegistration::read_pinned(
                        refinement_registration
                            .as_deref()
                            .ok_or("--refinement-registration is required")?,
                        refinement_registration_sha256
                            .as_deref()
                            .ok_or("--refinement-registration-sha256 is required")?,
                    )?;
                    registration.validate_against(&expected)?;
                    Some(registration)
                } else {
                    None
                };
                let cache = match runtime_cache {
                    Some(path) => RuntimeCache::open(&path)?,
                    None => RuntimeCache::for_user()?,
                };
                let runtime = runtime.ok_or("--runtime is required")?;
                if !runtime.is_absolute() {
                    return Err("--runtime must be absolute".into());
                }
                let pin = cache.library(&runtime, &expected.runtime_sha256)?;
                let export = export.as_deref().ok_or("--export is required")?;
                let checkpoint = checkpoint.as_deref().ok_or("--checkpoint is required")?;
                let driver = Box::new(match (refinement.as_ref(), continuation.as_ref()) {
                    (None, None) => OwnPalsOnnxCollectionDriver::load_cpu(
                        export,
                        checkpoint,
                        &pin,
                        &expected,
                        CpuConfig::default(),
                        pals,
                        pals_rounds,
                    )?,
                    (Some(registration), None) => {
                        OwnPalsOnnxCollectionDriver::load_cpu_with_refinement_policy(
                            export,
                            checkpoint,
                            &pin,
                            &expected,
                            CpuConfig::default(),
                            pals,
                            pals_rounds,
                            registration,
                        )?
                    }
                    (None, Some(registration)) => {
                        OwnPalsOnnxCollectionDriver::load_cpu_with_continuation_policy(
                            export,
                            checkpoint,
                            &pin,
                            &expected,
                            CpuConfig::default(),
                            pals,
                            pals_rounds,
                            registration,
                        )?
                    }
                    (Some(_), Some(_)) => {
                        return Err("mixed continuation/refinement registrations".into());
                    }
                });
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
    if let Some(producer_id) = describe_producer {
        println!(
            "{}",
            serde_json::to_string(&pals_producer_registration_description(
                driver.as_mut(),
                &producer_id
            )?)?
        );
        return Ok(true);
    }
    let output = output.ok_or("--output-root is required")?;
    let cancelled = Arc::new(AtomicBool::new(false));
    #[cfg(unix)]
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signal_hook::flag::register(signal, Arc::clone(&cancelled))?;
    }
    let receipt = collect_pals_own_data_with_producer(
        config,
        &output,
        driver.as_mut(),
        &registry,
        cancelled.as_ref(),
        producer_config.as_ref(),
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

#[cfg(test)]
mod tests {
    use super::*;
    fn public_registration() -> PathBuf {
        std::env::temp_dir().join("public-refinement-registration.json")
    }

    #[test]
    fn refinement_cli_preserves_absent_legacy_selection() {
        for mode in ["cpu", "pals-mock", "pals-onnx"] {
            assert!(!validate_refinement_cli(mode, None, None, None, None, false, false).unwrap());
        }
    }
    #[test]
    fn actual_continuation_cli_requires_exact_explicit_cpu_strict_selection() {
        let path = public_registration();
        let sha = "07".repeat(32);
        assert!(
            validate_refinement_cli(
                "pals-onnx",
                Some("cpu"),
                Some("actual-opponent-continuation-v1"),
                Some(&path),
                Some(&sha),
                true,
                false
            )
            .unwrap()
        );
        assert!(
            validate_refinement_cli(
                "pals-onnx",
                Some("cpu"),
                Some("actual-opponent-continuation-v1"),
                Some(&path),
                Some(&sha),
                false,
                false
            )
            .is_err()
        );
        assert!(
            validate_refinement_cli(
                "pals-onnx",
                Some("cuda"),
                Some("actual-opponent-continuation-v1"),
                Some(&path),
                Some(&sha),
                true,
                false
            )
            .is_err()
        );
        assert!(
            validate_refinement_cli(
                "pals-mock",
                Some("cpu"),
                Some("actual-opponent-continuation-v1"),
                Some(&path),
                Some(&sha),
                true,
                false
            )
            .is_err()
        );
        assert!(
            validate_refinement_cli(
                "pals-onnx",
                Some("cpu"),
                Some("actual_opponent_continuation_v1"),
                Some(&path),
                Some(&sha),
                true,
                false
            )
            .is_err()
        );
    }

    #[test]
    fn refinement_cli_admits_strict_collection_and_metadata_bootstrap_only() {
        let path = public_registration();
        let sha = "07".repeat(32);
        for (strict, bootstrap) in [(true, false), (false, true)] {
            assert!(
                validate_refinement_cli(
                    "pals-onnx",
                    Some("cpu"),
                    Some("same-repaired-line-once-v1"),
                    Some(&path),
                    Some(&sha),
                    strict,
                    bootstrap
                )
                .unwrap()
            );
        }
        assert!(
            validate_refinement_cli(
                "pals-onnx",
                Some("cpu"),
                Some("same-repaired-line-once-v1"),
                Some(&path),
                Some(&sha),
                false,
                false
            )
            .is_err()
        );
    }

    #[test]
    fn refinement_cli_rejects_partial_alias_provider_and_pin_drift() {
        let path = public_registration();
        let sha = "07".repeat(32);
        let selected = Some("same-repaired-line-once-v1");
        for policy in [
            None,
            Some("disabled"),
            Some("same_repaired_line_once_v1"),
            Some("unknown"),
        ] {
            assert!(
                validate_refinement_cli(
                    "pals-onnx",
                    Some("cpu"),
                    policy,
                    Some(&path),
                    Some(&sha),
                    true,
                    false
                )
                .is_err()
            );
        }
        assert!(
            validate_refinement_cli(
                "pals-onnx",
                Some("cpu"),
                selected,
                None,
                Some(&sha),
                true,
                false
            )
            .is_err()
        );
        assert!(
            validate_refinement_cli(
                "pals-onnx",
                Some("cpu"),
                selected,
                Some(&path),
                None,
                true,
                false
            )
            .is_err()
        );
        for (mode, provider) in [
            ("cpu", Some("cpu")),
            ("pals-mock", Some("cpu")),
            ("pals-onnx", Some("cuda")),
            ("pals-onnx", None),
        ] {
            assert!(
                validate_refinement_cli(
                    mode,
                    provider,
                    selected,
                    Some(&path),
                    Some(&sha),
                    true,
                    false
                )
                .is_err()
            );
        }
        let relative = PathBuf::from("relative.json");
        assert!(
            validate_refinement_cli(
                "pals-onnx",
                Some("cpu"),
                selected,
                Some(&relative),
                Some(&sha),
                true,
                false
            )
            .is_err()
        );
        assert!(
            validate_refinement_cli(
                "pals-onnx",
                Some("cpu"),
                selected,
                Some(&path),
                Some("00"),
                true,
                false
            )
            .is_err()
        );
        assert!(
            validate_refinement_cli(
                "pals-onnx",
                Some("cpu"),
                selected,
                Some(&path),
                Some(&"AA".repeat(32)),
                true,
                false
            )
            .is_err()
        );
    }

    #[test]
    fn refinement_flags_reject_duplicate_without_overwriting_first_value() {
        let mut policy = None;
        set_refinement_arg(&mut policy, "first".to_owned(), "--post-repair-recheck").unwrap();
        assert!(
            set_refinement_arg(&mut policy, "second".to_owned(), "--post-repair-recheck").is_err()
        );
        assert_eq!(policy.as_deref(), Some("first"));
        let mut path = Some(public_registration());
        assert!(
            set_refinement_arg(
                &mut path,
                PathBuf::from("other.json"),
                "--refinement-registration"
            )
            .is_err()
        );
        let mut sha = Some("07".repeat(32));
        assert!(
            set_refinement_arg(
                &mut sha,
                "08".repeat(32),
                "--refinement-registration-sha256"
            )
            .is_err()
        );
    }
}
