//! Independent PyTorch reference -> frozen Rust/ORT P/C acceptance.
//! Generated fixtures and reports belong to caller-owned external roots.
use rz_eval::asset;
use rz_eval::onnx::{OrtRuntime, Provider};
use rz_eval::pals_model::{
    PalsModelConfig, PalsModelInput, PalsRawOutput, PalsRole, PALS_MODEL_SCHEMA,
};
use rz_eval::pals_onnx::{
    PalsBackendStats, PalsNativeCommand, PalsNativeResult, PalsOnnxBackend, PalsOnnxConfig,
    PalsPublicMemoryWitness,
};
use rz_eval::runtime_pin::{CudaRuntimeBundleSpec, RuntimeBundleFileRole, RuntimeCache};
use rz_eval::worker::PhysicalPoll;
use serde::Deserialize;
use serde_json::json;
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::task::Poll;
use std::time::{Duration, Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixtures {
    schema: String,
    checkpoint_sha256: String,
    rules_certified: bool,
    trained: bool,
    reference: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    input: PalsModelInput,
    expected: PalsRawOutput,
    public_memory: PalsPublicMemoryWitness,
}

fn validate_fixture_coverage(fixtures: &Fixtures) -> Result<(), Box<dyn Error>> {
    let required = [
        ("empty_context", PalsRole::Proposer, 0, 0, 0),
        ("critic_divergence", PalsRole::Critic, 3, 7, 4),
        ("proposer_order", PalsRole::Proposer, 3, 7, 0),
        ("proposer_order_reversed", PalsRole::Proposer, 3, 7, 0),
        ("same_board_other_history", PalsRole::Proposer, 3, 7, 0),
        ("all_promotions", PalsRole::Critic, 1, 4, 1),
    ];
    if fixtures.schema != PALS_MODEL_SCHEMA
        || fixtures.rules_certified
        || fixtures.reference != "pytorch_fp32_tf32_off"
        || fixtures.cases.len() != required.len()
    {
        return Err("expected registered six-case P/C numeric-only reference fixtures".into());
    }
    for (name, role, records, candidates, divergences) in required {
        let matching: Vec<_> = fixtures
            .cases
            .iter()
            .filter(|case| case.name == name)
            .collect();
        if matching.len() != 1 {
            return Err("P/C numeric reference case is missing or duplicated".into());
        }
        let input = &matching[0].input;
        input.validate(&PalsModelConfig::baseline())?;
        if input.role != role
            || input.records.len() != records
            || input.candidates.len() != candidates
            || input.divergence_features.len() != divergences
        {
            return Err("P/C numeric reference case does not cover its declared role/shape".into());
        }
        let witness = &matching[0].public_memory;
        let tokens = PalsModelConfig::baseline().public_memory_tokens(records)?;
        if witness.tokens != tokens
            || witness.memory_key.len() != 2 * tokens * 64
            || witness.memory_value.len() != 2 * tokens * 64
            || witness.memory_mask.len() != tokens
            || witness
                .memory_key
                .iter()
                .chain(&witness.memory_value)
                .any(|value| !value.is_finite())
            || witness.memory_mask[..66].iter().any(|value| !*value)
            || witness.memory_mask[66..]
                .iter()
                .filter(|value| **value)
                .count()
                != records
        {
            return Err("public-memory reference witness shape/finite/mask differs".into());
        }
    }
    let case = |name: &str| {
        fixtures
            .cases
            .iter()
            .find(|case| case.name == name)
            .expect("coverage verified")
    };
    let ordered = &case("proposer_order").input;
    let mut reversed = ordered.clone();
    reversed.candidates.reverse();
    if reversed != case("proposer_order_reversed").input {
        return Err("candidate order comparison changed another input condition".into());
    }
    let history = &case("same_board_other_history").input;
    let mut other_history = ordered.clone();
    other_history.history_digest = history.history_digest;
    if ordered.history_digest == history.history_digest || other_history != *history {
        return Err("history comparison did not preserve the same board/other inputs".into());
    }
    let promotions = &case("all_promotions").input.candidates;
    if !promotions.iter().enumerate().all(|(index, candidate)| {
        candidate.from == 48 && candidate.to == 56 && candidate.promotion == index as u8 + 1
    }) {
        return Err("promotion reference does not cover queen/rook/bishop/knight order".into());
    }
    Ok(())
}

fn compare(actual: &[f32], expected: &[f32], atol: f64, rtol: f64) -> Result<f64, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err("reference tensor shape differs".into());
    }
    let mut maximum: f64 = 0.;
    for (a, e) in actual.iter().zip(expected) {
        let delta = (f64::from(*a) - f64::from(*e)).abs();
        if !a.is_finite() || !e.is_finite() || delta > atol + rtol * f64::from(*e).abs() {
            return Err(
                format!("PALS numeric mismatch: actual={a}, expected={e}, abs={delta}").into(),
            );
        }
        maximum = maximum.max(delta);
    }
    Ok(maximum)
}
fn verify(raw: &PalsRawOutput, case: &Case) -> Result<serde_json::Value, Box<dyn Error>> {
    let config = PalsModelConfig::baseline();
    let actual = raw.decode(&case.input, &config)?;
    let expected = case.expected.decode(&case.input, &config)?;
    let candidate = compare(
        &raw.candidate_logits,
        &case.expected.candidate_logits,
        1e-4,
        1e-3,
    )?;
    let wdl_logits = compare(&raw.wdl_logits, &case.expected.wdl_logits, 1e-4, 1e-3)?;
    let latent = compare(
        &raw.private_latent,
        &case.expected.private_latent,
        1e-4,
        1e-3,
    )?;
    let policy = compare(
        &actual.candidate_policy,
        &expected.candidate_policy,
        1e-4,
        0.,
    )?;
    let wdl = compare(&actual.wdl, &expected.wdl, 1e-4, 0.)?;
    let divergence = match (&raw.divergence_logits, &case.expected.divergence_logits) {
        (Some(a), Some(e)) => Some(compare(a, e, 1e-4, 1e-3)?),
        (None, None) => None,
        _ => return Err("private critic head differs".into()),
    };
    Ok(
        json!({"case":case.name,"candidate_logit_max_abs":candidate,"wdl_logit_max_abs":wdl_logits,
        "latent_max_abs":latent,"policy_max_abs":policy,"wdl_max_abs":wdl,"divergence_logit_max_abs":divergence}),
    )
}
fn verify_public(
    backend: &PalsOnnxBackend,
    case: &Case,
) -> Result<serde_json::Value, Box<dyn Error>> {
    match backend.public_memory_witness()? {
        Some(actual) => {
            if actual.tokens != case.public_memory.tokens
                || actual.memory_mask != case.public_memory.memory_mask
            {
                return Err("independent public-memory shape/mask mismatch".into());
            }
            let key = compare(
                &actual.memory_key,
                &case.public_memory.memory_key,
                1e-4,
                1e-3,
            )?;
            let value = compare(
                &actual.memory_value,
                &case.public_memory.memory_value,
                1e-4,
                1e-3,
            )?;
            Ok(
                json!({"status":"passed","key_max_abs":key,"value_max_abs":value,"mask":"exact","source":"completed_host_cache"}),
            )
        }
        None if backend.config().device_public_memory => Ok(
            json!({"status":"not_run","cause":"device_KV_host_witness_unavailable","key_max_abs":null,"value_max_abs":null}),
        ),
        None => Err("completed public-memory cache witness missing".into()),
    }
}
fn outside_git(path: &Path) -> Result<(), Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("external artifact path must be absolute".into());
    }
    for ancestor in path.canonicalize()?.ancestors() {
        if ancestor.join(".git").try_exists()? {
            return Err("generated artifacts must be outside Git".into());
        }
    }
    Ok(())
}
fn hex_digest(value: &[u8; 32]) -> String {
    value.iter().map(|v| format!("{v:02x}")).collect()
}
fn stats_json(stats: &PalsBackendStats) -> serde_json::Value {
    json!({"admitted_role_requests":stats.admitted_role_requests,
        "public_cache_hits":stats.public_cache_hits,"public_cache_misses":stats.public_cache_misses,
        "public_nn_runs_attempted":stats.public_nn_runs_attempted,"public_nn_runs_completed":stats.public_nn_runs_completed,
        "public_nn_runs_failed_known":stats.public_nn_runs_failed_known,
        "role_nn_runs_attempted":stats.role_nn_runs_attempted,"role_nn_runs_completed":stats.role_nn_runs_completed,
        "role_nn_runs_failed_known":stats.role_nn_runs_failed_known,"completed_nn_inputs":stats.completed_nn_inputs,
        "validated_public_outputs":stats.validated_public_outputs,"validated_role_outputs":stats.validated_role_outputs,
        "new_game_resets":stats.new_game_resets,"live_public_cache_entries":stats.live_public_cache_entries})
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(8..=9).contains(&args.len()) {
        return Err("usage: pals_model_check EXPORT.json EXPORT_SHA256 ORT_LIBRARY ORT_SHA256 FIXTURES.json REPORT.json cpu|cuda|cuda-device CACHE_ROOT [CUDA_BUNDLE.json]".into());
    }
    let is_cuda = match (args[6].as_str(), args.len()) {
        ("cpu", 8) => false,
        ("cuda" | "cuda-device", 9) => true,
        _ => return Err("CUDA requires its explicit pinned bundle; CPU excludes it".into()),
    };
    let report = Path::new(&args[5]);
    outside_git(report.parent().ok_or("report has no existing parent")?)?;
    if report.exists() {
        return Err("refusing to overwrite registered PALS acceptance report".into());
    }
    let fixture_path = Path::new(&args[4]);
    if !fixture_path.is_absolute() {
        return Err("fixture must be absolute".into());
    }
    let fixture_bytes = asset::read_bounded(fixture_path, 16 * 1024 * 1024)?;
    let fixture_digest = asset::sha256(&fixture_bytes);
    let fixtures: Fixtures = serde_json::from_slice(&fixture_bytes)?;
    validate_fixture_coverage(&fixtures)?;
    let cache = RuntimeCache::open(Path::new(&args[7]))?;
    let pin = if is_cuda {
        let spec_bytes = asset::read_bounded(Path::new(&args[8]), 64 * 1024)?;
        let spec = CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&spec_bytes)?)?;
        let core = spec
            .files
            .iter()
            .find(|f| f.role == RuntimeBundleFileRole::Core)
            .ok_or("bundle core absent")?;
        let path = Path::new(&args[2]);
        if path.file_name().and_then(|n| n.to_str()) != Some(core.filename.as_str())
            || args[3] != core.sha256
        {
            return Err("ORT declaration differs from explicit CUDA bundle".into());
        }
        cache.cuda_bundle(path.parent().ok_or("runtime core has no parent")?, &spec)?
    } else {
        cache.library(Path::new(&args[2]), &args[3])?
    };
    let runtime = OrtRuntime::load(&pin).map_err(|mut failure| {
        if is_cuda {
            failure.detail = "PALS CUDA runtime bootstrap failed before session creation";
        }
        failure
    })?;
    let runtime_digest = runtime.binary_digest();
    let mut config = PalsOnnxConfig::cpu();
    if is_cuda {
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 2 * 1024 * 1024 * 1024,
        };
        config.device_public_memory = args[6] == "cuda-device";
    }
    let mut backend =
        PalsOnnxBackend::load(Path::new(&args[0]), &args[1], runtime.clone(), config)?;
    let epoch = backend.model_epoch();
    if epoch != asset::parse_sha256(&fixtures.checkpoint_sha256)?
        || backend.is_trained() != fixtures.trained
    {
        return Err("reference weight epoch differs".into());
    }
    for case in &fixtures.cases {
        if case.name.is_empty() || case.name.len() > 128 || case.input.model_epoch != epoch {
            return Err("fixture identity differs".into());
        }
        case.input.validate(&PalsModelConfig::baseline())?;
    }
    // CUDA tests are bounded by the shell/process owner in addition to these
    // logical deadlines. This check never claims a deadline stopped native Run.
    let began = Instant::now();
    let mut reports = Vec::new();
    let mut raw_outputs = Vec::new();
    for case in &fixtures.cases {
        if began.elapsed() >= Duration::from_secs(120) {
            return Err("finite numeric window exhausted before next physical Run".into());
        }
        let raw = backend.run(&case.input)?;
        let mut report = verify(&raw, case)?;
        report["public_memory"] = verify_public(&backend, case)?;
        reports.push(report);
        let repeated = backend.run(&case.input)?;
        if repeated != raw {
            return Err("fresh private-role repeat changed exact PALS result".into());
        }
        raw_outputs.push(raw);
    }
    let mut validator = fixtures.cases[0].input.clone();
    validator.role = PalsRole::Validator;
    validator.divergence_features.clear();
    if backend.run(&validator).is_ok() {
        return Err("P/C product admitted private V role".into());
    }
    let public_encodes = backend.public_encodes;
    let public_cache_hits = backend.public_cache_hits;
    backend.verify_runtime()?;
    let trained = backend.is_trained();
    let residency = backend.residency().clone();
    let encodes_before_new_game = backend.public_encodes;
    backend.clear_public_memory()?;
    if backend.public_encodes != encodes_before_new_game {
        return Err("new game cache reset executed a neural graph".into());
    }
    if backend.run(&fixtures.cases[0].input)? != raw_outputs[0]
        || backend.public_encodes != encodes_before_new_game + 1
    {
        return Err("new game did not force a fresh equivalent public-memory encode".into());
    }
    // Separate sessions verify cache eviction/fresh encode equivalence. No
    // measured performance improvement is inferred from these correctness calls.
    config.cache_public_memory = false;
    let mut fresh = PalsOnnxBackend::load(Path::new(&args[0]), &args[1], runtime, config)?;
    for (case, expected) in fixtures.cases.iter().zip(&raw_outputs) {
        if fresh.run(&case.input)? != *expected {
            return Err("public memory cache changed exact PALS output".into());
        }
    }
    fresh.verify_runtime()?;
    drop(fresh);
    let before_worker_stats = backend.snapshot_stats()?;
    let mut worker = backend.controlled_worker()?;
    let mut reset = worker.submit(PalsNativeCommand::NewGame)?;
    let reset_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match reset.poll() {
            PhysicalPoll::Ready(Ok(PalsNativeResult::NewGame)) => break,
            PhysicalPoll::Ready(Err(error)) => return Err(error.into()),
            PhysicalPoll::Ready(Ok(PalsNativeResult::Evaluation(_))) => {
                return Err("cache reset unexpectedly returned a neural evaluation".into());
            }
            PhysicalPoll::Ready(Ok(PalsNativeResult::Stats(_))) => {
                return Err("cache reset unexpectedly returned native statistics".into())
            }
            PhysicalPoll::Ready(Ok(PalsNativeResult::RuntimeVerified)) => {
                return Err("cache reset unexpectedly returned runtime verification".into())
            }
            PhysicalPoll::Quarantined => {
                return Err("new game cache reset remains quarantined".into())
            }
            PhysicalPoll::Consumed => return Err("new game cache reset completed twice".into()),
            PhysicalPoll::Pending if Instant::now() < reset_deadline => {
                std::thread::sleep(Duration::from_millis(1));
            }
            PhysicalPoll::Pending => {
                return Err("new game cache reset has not physically completed".into())
            }
        }
    }
    let mut lease = worker.submit(PalsNativeCommand::Evaluate(fixtures.cases[0].input.clone()))?;
    let physical_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match lease.poll() {
            PhysicalPoll::Ready(result) => {
                match result? {
                    PalsNativeResult::Evaluation(raw) => {
                        verify(&raw, &fixtures.cases[0])?;
                    }
                    PalsNativeResult::NewGame => {
                        return Err("evaluation unexpectedly returned a cache reset".into())
                    }
                    PalsNativeResult::Stats(_) => {
                        return Err("evaluation unexpectedly returned native statistics".into())
                    }
                    PalsNativeResult::RuntimeVerified => {
                        return Err("evaluation unexpectedly returned runtime verification".into())
                    }
                }
                break;
            }
            PhysicalPoll::Quarantined => {
                return Err("physical completion is unknown; quarantined job retained".into())
            }
            PhysicalPoll::Consumed => return Err("physical lease completed twice".into()),
            PhysicalPoll::Pending => {}
        }
        if Instant::now() >= physical_deadline {
            return Err("physical lease still live at finite check deadline".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let mut verification = worker.submit(PalsNativeCommand::VerifyRuntime)?;
    let verification_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match verification.poll() {
            PhysicalPoll::Ready(Ok(PalsNativeResult::RuntimeVerified)) => break,
            PhysicalPoll::Ready(Err(error)) => return Err(error.into()),
            PhysicalPoll::Ready(Ok(_)) => {
                return Err("runtime verification returned another command response".into())
            }
            PhysicalPoll::Quarantined => {
                return Err(
                    "runtime verification cannot confirm unknown physical completion".into(),
                )
            }
            PhysicalPoll::Consumed => return Err("runtime verification completed twice".into()),
            PhysicalPoll::Pending if Instant::now() < verification_deadline => {
                std::thread::sleep(Duration::from_millis(1))
            }
            PhysicalPoll::Pending => {
                return Err("runtime verification has no physical ACK before deadline".into())
            }
        }
    }
    let mut snapshot = worker.submit(PalsNativeCommand::SnapshotStats)?;
    let stats_deadline = Instant::now() + Duration::from_secs(30);
    let stats = loop {
        match snapshot.poll() {
            PhysicalPoll::Ready(Ok(PalsNativeResult::Stats(stats))) => break stats,
            PhysicalPoll::Ready(Err(error)) => return Err(error.into()),
            PhysicalPoll::Ready(Ok(_)) => {
                return Err("native statistics returned another command response".into())
            }
            PhysicalPoll::Quarantined => {
                return Err("native statistics cannot confirm unknown physical completion".into())
            }
            PhysicalPoll::Consumed => return Err("native statistics completed twice".into()),
            PhysicalPoll::Pending if Instant::now() < stats_deadline => {
                std::thread::sleep(Duration::from_millis(1))
            }
            PhysicalPoll::Pending => {
                return Err("native statistics have no physical ACK before deadline".into())
            }
        }
    };
    stats.validate()?;
    if stats.admitted_role_requests != before_worker_stats.admitted_role_requests + 1
        || stats.role_nn_runs_attempted != before_worker_stats.role_nn_runs_attempted + 1
        || stats.role_nn_runs_completed != before_worker_stats.role_nn_runs_completed + 1
        || stats.public_nn_runs_attempted != before_worker_stats.public_nn_runs_attempted + 1
        || stats.public_nn_runs_completed != before_worker_stats.public_nn_runs_completed + 1
        || stats.completed_nn_inputs != before_worker_stats.completed_nn_inputs + 2
        || stats.new_game_resets != before_worker_stats.new_game_resets + 1
        || stats.live_public_cache_entries != 1
    {
        return Err("NewGame/SnapshotStats control changed NN accounting or failed to force fresh public encode".into());
    }
    let shutdown_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match worker.try_shutdown() {
            Poll::Ready(result) => {
                result?;
                break;
            }
            Poll::Pending if Instant::now() < shutdown_deadline => {
                std::thread::sleep(Duration::from_millis(1))
            }
            Poll::Pending => return Err("physical owner has not completed shutdown".into()),
        }
    }
    let receipt = json!({"schema": PALS_MODEL_SCHEMA,"status":"passed","trained":trained,"rules_certified":false,
        "model_epoch":hex_digest(&epoch),"manifest_sha256":args[1],"fixture_sha256":hex_digest(&fixture_digest),
        "runtime_sha256":hex_digest(&runtime_digest),"runtime":"1.22.0","rust_ort":"2.0.0-rc.10",
        "provider":if is_cuda {"CUDAExecutionProvider"} else {"CPUExecutionProvider"},"precision":"fp32","tf32":false,
        "physical_completion":"confirmed","physical_shutdown":"confirmed","public_encodes":public_encodes,
        "public_cache_hits":public_cache_hits,"fresh_cache_equivalence":"passed","private_role_repeat":"passed",
        "public_counter_scope":"six_reference_cases_before_new_game_checks",
        "v_rejection":"passed","new_game_cache_reset":"passed","new_game_physical_ack":"confirmed",
        "stats_physical_ack":"confirmed","control_neural_runs":"zero","backend_stats":stats_json(&stats),
        "runtime_verify_physical_ack":"confirmed",
        "backend_stats_scope":"backend_lifetime_through_final_snapshot_ack",
        "device_public_memory":config.device_public_memory,"session_residency":residency,"vram_peak":"unknown","cases":reports});
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(report)?;
    serde_json::to_writer_pretty(&mut output, &receipt)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    println!("PALS P/C numeric/physical checks passed; trained={trained}; GPU peak unknown");
    Ok(())
}
