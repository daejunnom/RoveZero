//! Independent PyTorch reference -> frozen Rust/ORT P/C acceptance.
//! Generated fixtures and reports belong to caller-owned external roots.
use rz_eval::asset;
use rz_eval::onnx::{OrtRuntime, Provider};
use rz_eval::pals_model::{PalsModelConfig, PalsModelInput, PalsRawOutput, PalsRole, PALS_MODEL_SCHEMA};
use rz_eval::pals_onnx::{PalsOnnxBackend, PalsOnnxConfig};
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
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case { name: String, input: PalsModelInput, expected: PalsRawOutput }

fn compare(actual: &[f32], expected: &[f32], atol: f64, rtol: f64) -> Result<f64, Box<dyn Error>> {
    if actual.len() != expected.len() { return Err("reference tensor shape differs".into()); }
    let mut maximum: f64 = 0.;
    for (a, e) in actual.iter().zip(expected) {
        let delta = (f64::from(*a) - f64::from(*e)).abs();
        if !a.is_finite() || !e.is_finite() || delta > atol + rtol * f64::from(*e).abs() {
            return Err(format!("PALS numeric mismatch: actual={a}, expected={e}, abs={delta}").into());
        }
        maximum = maximum.max(delta);
    }
    Ok(maximum)
}
fn verify(raw: &PalsRawOutput, case: &Case) -> Result<serde_json::Value, Box<dyn Error>> {
    let config = PalsModelConfig::baseline();
    let actual = raw.decode(&case.input, &config)?;
    let expected = case.expected.decode(&case.input, &config)?;
    let candidate = compare(&raw.candidate_logits, &case.expected.candidate_logits, 1e-4, 1e-3)?;
    let wdl_logits = compare(&raw.wdl_logits, &case.expected.wdl_logits, 1e-4, 1e-3)?;
    let latent = compare(&raw.private_latent, &case.expected.private_latent, 1e-4, 1e-3)?;
    let policy = compare(&actual.candidate_policy, &expected.candidate_policy, 1e-4, 0.)?;
    let wdl = compare(&actual.wdl, &expected.wdl, 1e-4, 0.)?;
    let divergence = match (&raw.divergence_logits, &case.expected.divergence_logits) {
        (Some(a), Some(e)) => Some(compare(a, e, 1e-4, 1e-3)?),
        (None, None) => None,
        _ => return Err("private critic head differs".into()),
    };
    Ok(json!({"case":case.name,"candidate_logit_max_abs":candidate,"wdl_logit_max_abs":wdl_logits,
        "latent_max_abs":latent,"policy_max_abs":policy,"wdl_max_abs":wdl,"divergence_logit_max_abs":divergence}))
}
fn outside_git(path: &Path) -> Result<(), Box<dyn Error>> {
    if !path.is_absolute() { return Err("external artifact path must be absolute".into()); }
    for ancestor in path.canonicalize()?.ancestors() {
        if ancestor.join(".git").try_exists()? { return Err("generated artifacts must be outside Git".into()); }
    }
    Ok(())
}
fn hex_digest(value: &[u8;32]) -> String { value.iter().map(|v| format!("{v:02x}")).collect() }
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(8..=9).contains(&args.len()) {
        return Err("usage: pals_model_check EXPORT.json EXPORT_SHA256 ORT_LIBRARY ORT_SHA256 FIXTURES.json REPORT.json cpu|cuda CACHE_ROOT [CUDA_BUNDLE.json]".into());
    }
    let is_cuda = match (args[6].as_str(), args.len()) {
        ("cpu",8) => false, ("cuda",9) => true, _ => return Err("CUDA requires its explicit pinned bundle; CPU excludes it".into()),
    };
    let report = Path::new(&args[5]);
    outside_git(report.parent().ok_or("report has no existing parent")?)?;
    if report.exists() { return Err("refusing to overwrite registered PALS acceptance report".into()); }
    let fixture_path = Path::new(&args[4]);
    if !fixture_path.is_absolute() { return Err("fixture must be absolute".into()); }
    let fixture_bytes = asset::read_bounded(fixture_path, 16*1024*1024)?;
    let fixture_digest = asset::sha256(&fixture_bytes);
    let fixtures: Fixtures = serde_json::from_slice(&fixture_bytes)?;
    if fixtures.schema != PALS_MODEL_SCHEMA || fixtures.rules_certified || fixtures.cases.is_empty() || fixtures.cases.len() > 32 {
        return Err("expected finite numeric-only PALS reference fixtures".into());
    }
    let cache = RuntimeCache::open(Path::new(&args[7]))?;
    let pin = if is_cuda {
        let spec_bytes = asset::read_bounded(Path::new(&args[8]), 64*1024)?;
        let spec = CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&spec_bytes)?)?;
        let core = spec.files.iter().find(|f| f.role == RuntimeBundleFileRole::Core).ok_or("bundle core absent")?;
        let path = Path::new(&args[2]);
        if path.file_name().and_then(|n| n.to_str()) != Some(core.filename.as_str()) || args[3] != core.sha256 {
            return Err("ORT declaration differs from explicit CUDA bundle".into());
        }
        cache.cuda_bundle(path.parent().ok_or("runtime core has no parent")?, &spec)?
    } else { cache.library(Path::new(&args[2]), &args[3])? };
    let runtime = OrtRuntime::load(&pin)?;
    let runtime_digest = runtime.binary_digest();
    let mut config = PalsOnnxConfig::cpu();
    if is_cuda { config.provider = Provider::Cuda { device_id:0, arena_bytes:2*1024*1024*1024 }; }
    let mut backend = PalsOnnxBackend::load(Path::new(&args[0]), &args[1], runtime.clone(), config)?;
    let epoch = backend.model_epoch();
    if epoch != asset::parse_sha256(&fixtures.checkpoint_sha256)? { return Err("reference weight epoch differs".into()); }
    for case in &fixtures.cases {
        if case.name.is_empty() || case.name.len() > 128 || case.input.model_epoch != epoch { return Err("fixture identity differs".into()); }
        case.input.validate(&PalsModelConfig::baseline())?;
    }
    // CUDA tests are bounded by the shell/process owner in addition to these
    // logical deadlines. This check never claims a deadline stopped native Run.
    let began = Instant::now();
    let mut reports = Vec::new();
    let mut raw_outputs = Vec::new();
    for case in &fixtures.cases {
        if began.elapsed() >= Duration::from_secs(120) { return Err("finite numeric window exhausted before next physical Run".into()); }
        let raw = backend.run(&case.input)?;
        reports.push(verify(&raw, case)?);
        let repeated = backend.run(&case.input)?;
        if repeated != raw { return Err("fresh private-role repeat changed exact PALS result".into()); }
        raw_outputs.push(raw);
    }
    let mut validator = fixtures.cases[0].input.clone(); validator.role = PalsRole::Validator; validator.divergence_features.clear();
    if backend.run(&validator).is_ok() { return Err("P/C product admitted private V role".into()); }
    let public_encodes = backend.public_encodes; let public_cache_hits = backend.public_cache_hits;
    backend.verify_runtime()?;
    let trained = backend.is_trained();
    // Separate sessions verify cache eviction/fresh encode equivalence. No
    // measured performance improvement is inferred from these correctness calls.
    config.cache_public_memory = false;
    let mut fresh = PalsOnnxBackend::load(Path::new(&args[0]), &args[1], runtime, config)?;
    for (case, expected) in fixtures.cases.iter().zip(&raw_outputs) {
        if fresh.run(&case.input)? != *expected { return Err("public memory cache changed exact PALS output".into()); }
    }
    fresh.verify_runtime()?;
    drop(fresh);
    let mut worker = backend.worker()?;
    let mut lease = worker.submit(fixtures.cases[0].input.clone())?;
    let physical_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match lease.poll() {
            PhysicalPoll::Ready(result) => { verify(&result?, &fixtures.cases[0])?; break; }
            PhysicalPoll::Quarantined => return Err("physical completion is unknown; quarantined job retained".into()),
            PhysicalPoll::Consumed => return Err("physical lease completed twice".into()),
            PhysicalPoll::Pending => {}
        }
        if Instant::now() >= physical_deadline { return Err("physical lease still live at finite check deadline".into()); }
        std::thread::sleep(Duration::from_millis(1));
    }
    let shutdown_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match worker.try_shutdown() {
            Poll::Ready(result) => { result?; break; }
            Poll::Pending if Instant::now() < shutdown_deadline => std::thread::sleep(Duration::from_millis(1)),
            Poll::Pending => return Err("physical owner has not completed shutdown".into()),
        }
    }
    let receipt = json!({"schema": PALS_MODEL_SCHEMA,"status":"passed","trained":trained,"rules_certified":false,
        "model_epoch":hex_digest(&epoch),"manifest_sha256":args[1],"fixture_sha256":hex_digest(&fixture_digest),
        "runtime_sha256":hex_digest(&runtime_digest),"runtime":"1.22.0","rust_ort":"2.0.0-rc.10",
        "provider":if is_cuda {"CUDAExecutionProvider"} else {"CPUExecutionProvider"},"precision":"fp32","tf32":false,
        "physical_completion":"confirmed","physical_shutdown":"confirmed","public_encodes":public_encodes,
        "public_cache_hits":public_cache_hits,"fresh_cache_equivalence":"passed","private_role_repeat":"passed",
        "v_rejection":"passed","vram_peak":"unknown","cases":reports});
    let mut output = std::fs::OpenOptions::new().create_new(true).write(true).open(report)?;
    serde_json::to_writer_pretty(&mut output, &receipt)?; output.write_all(b"\n")?; output.sync_all()?;
    println!("PALS P/C numeric/physical checks passed; trained={trained}; GPU peak unknown");
    Ok(())
}
