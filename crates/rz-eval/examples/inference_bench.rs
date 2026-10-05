//! User-run inference-only sweep. No numerical suite, cache or deduplication.
//! Fresh CUDA placement/session per batch; first failure stops the whole sweep.
use rz_encoding::classical::{self, Frame, HistoryFill, Input};
use rz_eval::{
    asset::{self, AssetProfile, MaiaAsset},
    onnx::{BackendConfig, OnnxBackend, OrtRuntime, Provider},
    runtime_pin::{CudaRuntimeBundleSpec, RuntimeCache},
};
use serde_json::{json, Value};
use std::{
    error::Error,
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

fn ns(duration: Duration) -> u64 {
    duration.as_nanos().try_into().unwrap_or(u64::MAX)
}
fn percentile(samples: &[u64], percent: usize) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[(sorted.len() * percent).div_ceil(100).saturating_sub(1)]
}
fn start_input() -> Result<classical::EncodedInput, Box<dyn Error>> {
    let frame = Frame {
        pieces: [
            [0xff00, 0x42, 0x24, 0x81, 0x08, 0x10],
            [
                0x00ff000000000000,
                0x4200000000000000,
                0x2400000000000000,
                0x8100000000000000,
                0x0800000000000000,
                0x1000000000000000,
            ],
        ],
        repeated: false,
        en_passant_target: None,
    };
    Ok(classical::encode(Input {
        history: &[frame],
        black_to_move: false,
        castling: [true; 4],
        halfmove_clock: 0,
        history_fill: HistoryFill::No,
    })?)
}
fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn memory_evidence() -> Value {
    // One final self observation, not a periodic PID/GPU scan. These counters
    // cannot identify CUDA VRAM or prove a Windows commit limit was respected.
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok();
        let field = |key: &str| {
            status.as_ref().and_then(|text| {
                text.lines()
                    .find_map(|line| line.strip_prefix(key).map(str::trim).map(String::from))
            })
        };
        json!({"self_vm_hwm":field("VmHWM:"),"self_vm_rss":field("VmRSS:"),
            "peak_vram":"unknown","scope":"final_proc_self_rss_not_gpu_or_global_commit"})
    }
    #[cfg(not(target_os = "linux"))]
    {
        json!({"self_vm_hwm":"unknown","peak_vram":"unknown"})
    }
}
fn save(file: &mut File, report: &Value) -> Result<(), Box<dyn Error>> {
    file.write_all(&serde_json::to_vec_pretty(report)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err("usage: inference_bench SOURCE ONNX EXPORT CUDA_CORE CUDA_BUNDLE REPORT (all absolute; fresh REPORT outside Git)".into());
    }
    if args.iter().any(|arg| !Path::new(arg).is_absolute()) {
        return Err("all paths must be absolute".into());
    }
    let parent = Path::new(&args[5])
        .parent()
        .ok_or("report parent missing")?
        .canonicalize()?;
    if parent.ancestors().any(|p| p.join(".git").exists()) {
        return Err("report must be outside Git".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[5])?;
    let mut report = json!({"schema_version":1,"kind":"inference_only_batch_sweep",
        "accepted":false,"batches":[1,2,4,8,16],"warmup_runs":3,"measured_runs":20,
        "precision":"fp32","tf32":false,"history_fill":"no","cache":false,"dedup":false,
        "io_binding":false,"cuda_graph":false,"fresh_session_per_batch":true,
        "numerical_regression":"separate_required_not_run_here", "samples":[],
        "measurement":"host_wall_including_run_io_and_physical_completion_not_kernel_time",
        "memory":null,"failure":null});
    let started = Instant::now();
    let outcome = sweep(&args, &mut report);
    report["memory"] = memory_evidence();
    report["total_wall_ns"] = json!(ns(started.elapsed()));
    match outcome {
        Ok(()) => {
            report["accepted"] = json!(true);
            save(&mut file, &report)?;
        }
        Err(error) => {
            report["failure"] = json!(error.to_string());
            save(&mut file, &report)?;
            return Err(error);
        }
    }
    Ok(())
}
fn sweep(args: &[String], report: &mut Value) -> Result<(), Box<dyn Error>> {
    let prepare = Instant::now();
    let bundle = CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&asset::read_bounded(
        Path::new(&args[4]),
        64 * 1024,
    )?)?)?;
    // Validate the caller's core declaration before any native library load.
    let core = bundle
        .files
        .iter()
        .find(|f| f.role == rz_eval::runtime_pin::RuntimeBundleFileRole::Core)
        .ok_or("missing core")?;
    if Path::new(&args[3]).file_name().and_then(|n| n.to_str()) != Some(core.filename.as_str()) {
        return Err("CUDA_CORE must name the bundle's declared core".into());
    }
    let cache = RuntimeCache::for_user()?;
    let pin = cache.cuda_bundle(
        Path::new(&args[3]).parent().ok_or("core parent missing")?,
        &bundle,
    )?;
    let runtime = OrtRuntime::load(&pin)?;
    let input = start_input()?;
    let input_bytes: Vec<_> = input
        .values()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    report["input_sha256"] = json!(hex(asset::sha256(&input_bytes)));
    report["runtime_bundle_sha256"] = json!(runtime.bundle_digest().map(hex));
    report["runtime_prepare_ns"] = json!(ns(prepare.elapsed()));
    for width in [1, 2, 4, 8, 16] {
        report["active_batch"] = json!(width);
        let preparation = Instant::now();
        let model = MaiaAsset::load(
            Path::new(&args[0]),
            Path::new(&args[1]),
            Path::new(&args[2]),
        )?;
        if model.profile() != AssetProfile::Bt4It332 {
            return Err("this sweep requires BT4-it332".into());
        }
        report["model_sha256"] = json!(model.manifest().onnx_sha256);
        report["export_manifest_sha256"] = json!(hex(model.manifest_digest()));
        let mut config = BackendConfig::cpu();
        config.max_batch = width;
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: model.profile().cuda_arena_bytes(),
        };
        let profile_dir = Path::new(&args[5]).with_extension(format!("b{width}.placement"));
        std::fs::create_dir(&profile_dir)?;
        config.profiling_prefix = Some(profile_dir.join("placement"));
        let (_, mut backend) = OnnxBackend::load_owned(&runtime, model, config)?;
        let prep_ns = ns(preparation.elapsed());
        let placement = backend
            .cuda_evidence()
            .ok_or("no actual CUDA placement evidence")?;
        let placement_record = json!({"profile_sha256":hex(placement.profile_sha256),
            "executed_cuda_nodes":placement.executed_cuda_nodes});
        let inputs = vec![&input; width];
        let warmup = Instant::now();
        for _ in 0..3 {
            let raw = backend.run(&inputs)?;
            if raw.len() != width {
                return Err("incomplete warmup".into());
            }
        }
        let warmup_ns = ns(warmup.elapsed());
        let mut times = Vec::with_capacity(20);
        for _ in 0..20 {
            let run = Instant::now();
            let outputs = backend.run(&inputs)?;
            let elapsed = ns(run.elapsed());
            if outputs.len() != width || backend.has_unconfirmed_physical_completion() {
                return Err("incomplete outputs or unconfirmed physical completion".into());
            }
            times.push(elapsed);
        }
        backend.verify_cuda_runtime_mappings()?;
        let total: u64 = times.iter().sum();
        report["samples"].as_array_mut().unwrap().push(json!({
            "batch":width,"prepare_ns":prep_ns,"warmup_ns":warmup_ns,"latency_ns":times,
            "p50_ns":percentile(&times,50),"p95_ns":percentile(&times,95),
            "completed_nn_items":width*20,"warmup_nn_items":width*3,
            "mean_ns_per_nn_item":total as f64/(width*20) as f64,
            "cuda_placement":placement_record,"memory":memory_evidence(),
            "physical_completion_confirmed":true
        }));
        // Normal drop only after confirmed completion. On an error the backend
        // retains its own unconfirmed tensors/session; no larger batch runs.
        drop(backend);
    }
    report["active_batch"] = Value::Null;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_input_matches_previous_no_history_tensor_identity() {
        let input = start_input().unwrap();
        let bytes: Vec<_> = input
            .values()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            hex(asset::sha256(&bytes)),
            "c33ae28cfa8081c3ff247f2ee36881e28520e4be8d585df726b8728b7dac22b6"
        );
    }
    #[test]
    fn reported_percentiles_use_nearest_rank_on_twenty_measured_runs() {
        let samples: Vec<_> = (1..=20).rev().collect();
        assert_eq!(percentile(&samples, 50), 10);
        assert_eq!(percentile(&samples, 95), 19);
    }
}
