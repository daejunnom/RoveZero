//! User-run inference-only sweep. No numerical suite, cache or deduplication.
//! Fresh CUDA placement/session per batch; first failure stops the whole sweep.
use rz_encoding::classical::{self, Frame, HistoryFill, Input};
use rz_eval::{
    asset::{self, AssetProfile, MaiaAsset},
    onnx::{BackendConfig, ExecutionExperiments, OnnxBackend, OrtRuntime, Provider},
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
            "self_rss_anon":field("RssAnon:"),"self_rss_file":field("RssFile:"),
            "self_rss_shmem":field("RssShmem:"),"self_vm_size":field("VmSize:"),
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
fn observe_memory(points: &mut Vec<Value>, started: Instant, phase: &str) {
    points.push(
        json!({"phase":phase,"since_prepare_ns":ns(started.elapsed()),"memory":memory_evidence()}),
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    BatchSweep,
    B1Buffers { reuse: bool },
    B1CpuArena { disabled: bool },
    B1OrtModel { direct: bool },
}
impl RunMode {
    fn parse(option: Option<&str>) -> Result<Self, Box<dyn Error>> {
        match option {
            None => Ok(Self::BatchSweep),
            Some("--b1-buffers=baseline") => Ok(Self::B1Buffers { reuse: false }),
            Some("--b1-buffers=reuse") if cfg!(feature = "experimental-io-buffers") => {
                Ok(Self::B1Buffers { reuse: true })
            }
            Some("--b1-cpu-arena=baseline") => Ok(Self::B1CpuArena { disabled: false }),
            Some("--b1-cpu-arena=disabled") if cfg!(feature = "experimental-ort-cpu-arena") => {
                Ok(Self::B1CpuArena { disabled: true })
            }
            Some("--b1-ort-model=baseline") if cfg!(feature = "experimental-ort-model") => {
                Ok(Self::B1OrtModel { direct: false })
            }
            Some("--b1-ort-model=direct") if cfg!(feature = "experimental-ort-model") => {
                Ok(Self::B1OrtModel { direct: true })
            }
            _ => {
                Err("unknown comparison mode or unavailable experimental-io-buffers feature".into())
            }
        }
    }
    fn widths(self) -> &'static [usize] {
        match self {
            Self::BatchSweep => &[1, 2, 4, 8, 16],
            Self::B1Buffers { .. } | Self::B1CpuArena { .. } | Self::B1OrtModel { .. } => &[1],
        }
    }
    fn observes_startup(self) -> bool {
        matches!(self, Self::B1CpuArena { .. } | Self::B1OrtModel { .. })
    }
}

fn recycle_checked(
    backend: &mut OnnxBackend,
    outputs: Vec<rz_eval::RawOutput>,
    mode: RunMode,
    expected: &mut Option<String>,
) -> Result<(), Box<dyn Error>> {
    if mode != RunMode::BatchSweep {
        let bytes: Vec<_> = outputs
            .iter()
            .flat_map(|raw| raw.policy_logits.iter().chain(&raw.wdl))
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let digest = hex(asset::sha256(&bytes));
        if expected.as_ref().is_some_and(|prior| prior != &digest) {
            return Err("B1 fixed input outputs changed during buffer comparison".into());
        }
        *expected = Some(digest);
        // Both comparison arms return ownership at the same point. The default
        // batch sweep retains its original measurement/drop behavior.
        backend.recycle_outputs(outputs);
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(args.len() == 6 || args.len() == 7 || args.len() == 9) {
        return Err("usage: inference_bench SOURCE ONNX EXPORT CUDA_CORE CUDA_BUNDLE REPORT [--b1-buffers=baseline|reuse|--b1-cpu-arena=baseline|disabled] (all six paths absolute; fresh REPORT outside Git)".into());
    }
    let mode = RunMode::parse(args.get(6).map(String::as_str))?;
    if matches!(mode, RunMode::B1OrtModel { .. }) != (args.len() == 9)
        || (args.len() == 9 && args[7..].iter().any(|p| !Path::new(p).is_absolute()))
    {
        return Err(
            "ORT comparison requires absolute ORT/model-manifest paths after its explicit mode"
                .into(),
        );
    }
    if args[..6].iter().any(|arg| !Path::new(arg).is_absolute()) {
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
    if let RunMode::B1Buffers { reuse } = mode {
        report["kind"] = json!("b1_buffer_reuse_fixed_work");
        report["batches"] = json!([1]);
        report["reuse_buffers"] = json!(reuse);
        report["measurement"] = json!("host_wall_including_run_physical_completion_output_digest_and_ownership_return_not_kernel_time");
    }
    if let RunMode::B1CpuArena { disabled } = mode {
        report["kind"] = json!("b1_cuda_cpu_arena_fixed_work");
        report["batches"] = json!([1]);
        report["disable_cuda_cpu_arena"] = json!(disabled);
        report["cpu_ep_fallback"] = json!(false);
        report["reuse_buffers"] = json!(false);
        report["startup_memory_observations"] = json!("bounded self RSS endpoints and process lifetime HWM; not phase peaks, live heap, cgroup file cache, VRAM or Windows commit");
        report["measurement"] = json!("host_wall_including_run_physical_completion_output_digest_and_ownership_return_not_kernel_time; same startup memory observations in both arms");
    }
    if let RunMode::B1OrtModel { direct } = mode {
        report["kind"] = json!("b1_owned_ort_fixed_work");
        report["batches"] = json!([1]);
        report["zero_copy_ort"] = json!(direct);
        report["cpu_ep_fallback"] = json!(false);
        report["reuse_buffers"] = json!(false);
        report["disable_cuda_cpu_arena"] = json!(false);
        report["measurement"] = json!("host_wall_including_run_physical_completion_output_digest_and_ownership_return_not_kernel_time; same bounded startup observer in both arms");
    }
    let started = Instant::now();
    let outcome = sweep(&args, &mut report, mode);
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
fn sweep(args: &[String], report: &mut Value, mode: RunMode) -> Result<(), Box<dyn Error>> {
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
    for &width in mode.widths() {
        report["active_batch"] = json!(width);
        let preparation = Instant::now();
        let mut memory_points = Vec::new();
        if mode.observes_startup() {
            observe_memory(&mut memory_points, preparation, "BeforeAssetLoad");
            report["startup_memory"] = json!(memory_points);
        }
        let direct = matches!(mode, RunMode::B1OrtModel { direct: true });
        let model = if direct {
            None
        } else {
            Some(MaiaAsset::load(
                Path::new(&args[0]),
                Path::new(&args[1]),
                Path::new(&args[2]),
            )?)
        };
        #[cfg(feature = "experimental-ort-model")]
        let derived = if direct {
            Some(rz_eval::ort_model::OwnedOrtAsset::load(
                Path::new(&args[0]),
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[7]),
                Path::new(&args[8]),
            )?)
        } else {
            None
        };
        let metadata = if let Some(model) = model.as_ref() {
            model.metadata()
        } else {
            #[cfg(feature = "experimental-ort-model")]
            {
                derived.as_ref().ok_or("derived model missing")?.metadata()
            }
            #[cfg(not(feature = "experimental-ort-model"))]
            {
                return Err("derived ORT feature unavailable".into());
            }
        };
        if metadata.profile() != AssetProfile::Bt4It332 {
            return Err("this sweep requires BT4-it332".into());
        }
        report["model_sha256"] = json!(metadata.manifest().onnx_sha256);
        report["export_manifest_sha256"] = json!(hex(metadata.manifest_digest()));
        let mut config = BackendConfig::cpu();
        config.max_batch = width;
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: metadata.profile().cuda_arena_bytes(),
        };
        if let RunMode::B1Buffers { reuse } = mode {
            config.experiments = ExecutionExperiments {
                reuse_buffers: reuse,
                ..ExecutionExperiments::default()
            };
        }
        if let RunMode::B1CpuArena { disabled } = mode {
            config.experiments.disable_cuda_cpu_arena = disabled;
        }
        config.experiments.zero_copy_ort = direct;
        let profile_dir = Path::new(&args[5]).with_extension(format!("b{width}.placement"));
        std::fs::create_dir(&profile_dir)?;
        config.profiling_prefix = Some(profile_dir.join("placement"));
        let loaded = if direct {
            #[cfg(feature = "experimental-ort-model")]
            {
                let derived = derived.ok_or("derived model missing")?;
                report["serialized_model_sha256"] = json!(derived.manifest().ort_sha256);
                report["derived_manifest_sha256"] = json!(hex(derived.manifest_digest()));
                let loaded =
                    OnnxBackend::load_owned_ort_observed(&runtime, derived, config, |phase| {
                        observe_memory(&mut memory_points, preparation, &format!("{phase:?}"))
                    });
                report["startup_memory"] = json!(memory_points);
                loaded
            }
            #[cfg(not(feature = "experimental-ort-model"))]
            {
                return Err("derived ORT feature unavailable".into());
            }
        } else if mode.observes_startup() {
            let loaded = OnnxBackend::load_owned_observed(
                &runtime,
                model.ok_or("original model missing")?,
                config,
                |phase| observe_memory(&mut memory_points, preparation, &format!("{phase:?}")),
            );
            report["startup_memory"] = json!(memory_points);
            loaded
        } else {
            OnnxBackend::load_owned(&runtime, model.ok_or("original model missing")?, config)
        };
        let (_, mut backend) = loaded?;
        if mode.observes_startup() {
            report["backend_sha256"] = json!(hex(backend.identity()));
            report["retained_model_bytes"] = json!(backend.retained_model_bytes());
        }
        let prep_ns = ns(preparation.elapsed());
        let placement = backend
            .cuda_evidence()
            .ok_or("no actual CUDA placement evidence")?;
        let placement_record = json!({"profile_sha256":hex(placement.profile_sha256),
            "executed_cuda_nodes":placement.executed_cuda_nodes});
        let inputs = vec![&input; width];
        let mut output_digest = None;
        let warmup = Instant::now();
        for _ in 0..3 {
            let raw = backend.run(&inputs)?;
            if raw.len() != width {
                return Err("incomplete warmup".into());
            }
            if backend.has_unconfirmed_physical_completion() {
                return Err("unconfirmed warmup physical completion".into());
            }
            if mode != RunMode::BatchSweep {
                recycle_checked(&mut backend, raw, mode, &mut output_digest)?;
            }
        }
        let warmup_ns = ns(warmup.elapsed());
        if mode.observes_startup() {
            observe_memory(&mut memory_points, preparation, "AfterWarmup");
            report["startup_memory"] = json!(memory_points);
        }
        let mut times = Vec::with_capacity(20);
        for _ in 0..20 {
            let run = Instant::now();
            let outputs = backend.run(&inputs)?;
            let run_ns = ns(run.elapsed());
            if outputs.len() != width || backend.has_unconfirmed_physical_completion() {
                return Err("incomplete outputs or unconfirmed physical completion".into());
            }
            if mode == RunMode::BatchSweep {
                times.push(run_ns);
            } else {
                recycle_checked(&mut backend, outputs, mode, &mut output_digest)?;
                times.push(ns(run.elapsed()));
            }
        }
        backend.verify_cuda_runtime_mappings()?;
        if mode.observes_startup() {
            observe_memory(&mut memory_points, preparation, "AfterMeasuredRuns");
            report["startup_memory"] = json!(memory_points);
        }
        let total: u64 = times.iter().sum();
        report["samples"].as_array_mut().unwrap().push(json!({
            "batch":width,"prepare_ns":prep_ns,"warmup_ns":warmup_ns,"latency_ns":times,
            "p50_ns":percentile(&times,50),"p95_ns":percentile(&times,95),
            "completed_nn_items":width*20,"warmup_nn_items":width*3,
            "mean_ns_per_nn_item":total as f64/(width*20) as f64,
            "cuda_placement":placement_record,"memory":memory_evidence(),
            "physical_completion_confirmed":true
        }));
        if mode != RunMode::BatchSweep {
            report["output_sha256"] = json!(output_digest);
        }
        // Normal drop only after confirmed completion. On an error the backend
        // retains its own unconfirmed tensors/session; no larger batch runs.
        drop(backend);
        if mode.observes_startup() {
            observe_memory(&mut memory_points, preparation, "AfterBackendDrop");
            report["startup_memory"] = json!(memory_points);
        }
    }
    report["active_batch"] = Value::Null;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn b1_comparison_is_explicit_and_does_not_change_default_sweep() {
        assert_eq!(RunMode::parse(None).unwrap().widths(), &[1, 2, 4, 8, 16]);
        assert_eq!(
            RunMode::parse(Some("--b1-buffers=baseline")).unwrap(),
            RunMode::B1Buffers { reuse: false }
        );
        let reuse = RunMode::parse(Some("--b1-buffers=reuse"));
        assert_eq!(reuse.is_ok(), cfg!(feature = "experimental-io-buffers"));
        if let Ok(mode) = reuse {
            assert_eq!(mode.widths(), &[1]);
        }
        for bad in ["--b1-buffers=", "--b1-buffers=4", "--b1-buffers=notify"] {
            assert!(RunMode::parse(Some(bad)).is_err());
        }
        assert_eq!(
            RunMode::parse(Some("--b1-cpu-arena=baseline"))
                .unwrap()
                .widths(),
            &[1]
        );
        assert_eq!(
            RunMode::parse(Some("--b1-cpu-arena=disabled")).is_ok(),
            cfg!(feature = "experimental-ort-cpu-arena")
        );
        assert!(RunMode::parse(Some("--b1-cpu-arena=auto")).is_err());
        for mode in ["--b1-ort-model=baseline", "--b1-ort-model=direct"] {
            assert_eq!(
                RunMode::parse(Some(mode)).is_ok(),
                cfg!(feature = "experimental-ort-model")
            );
        }
        assert!(RunMode::parse(Some("--b1-ort-model=auto")).is_err());
    }
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
