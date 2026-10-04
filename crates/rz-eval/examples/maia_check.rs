//! Bounded CPU or CUDA acceptance entry point; actual assets stay external.
//!
//! REPORT must be absolute. Its existing parent is also the bootstrap output
//! root: the caller owns this private directory and controls its ancestors.
//! It must be outside Git. No temporary-directory fallback is selected, and
//! native libraries use a separate bounded cache and retain process-lifetime pins.
use rz_encoding::classical::{self, Frame, HistoryFill, Input};
use rz_encoding::policy;
use rz_eval::asset::{self, MaiaAsset};
use rz_eval::onnx::{BackendConfig, ExecutionExperiments, OnnxBackend, OrtRuntime, Provider};
use rz_eval::runtime_pin::{CudaRuntimeBundleSpec, RuntimeBundleFileRole, RuntimeCache};
use rz_eval::{output, RawOutput};
use serde::Deserialize;
use serde_json::json;
use std::error::Error;
use std::path::Path;

#[derive(Deserialize)]
struct Fixtures {
    schema: u32,
    reference: String,
    reference_commit: String,
    reference_module_sha256: String,
    source_sha256: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    frames: Vec<FixtureFrame>,
    black_to_move: bool,
    castling: [bool; 4],
    halfmove_clock: u32,
    history_fill: String,
    input: Vec<f32>,
    legal_moves: Vec<FixtureMove>,
    indices: Vec<usize>,
    policy_logits: Vec<f32>,
    wdl: Vec<f32>,
    legal_policy: Vec<f32>,
}

#[derive(Deserialize)]
struct FixtureFrame {
    pieces: [[u64; 6]; 2],
    repeated: bool,
    en_passant_target: Option<u8>,
}
#[derive(Deserialize)]
struct FixtureMove {
    from_square: u8,
    to_square: u8,
    promotion: Option<char>,
    castle: bool,
}

fn compare(actual: &[f32], expected: &[f32], atol: f64, rtol: f64) -> Result<f64, Box<dyn Error>> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("comparison shape differs".into());
    }
    let mut max: f64 = 0.0;
    for (&a, &e) in actual.iter().zip(expected) {
        let error = (f64::from(a) - f64::from(e)).abs();
        if !a.is_finite() || !e.is_finite() || error > atol + rtol * f64::from(e).abs() {
            return Err(
                format!("numerical mismatch: actual={a}, reference={e}, error={error}").into(),
            );
        }
        max = max.max(error);
    }
    Ok(max)
}

fn verify(raw: &RawOutput, case: &Case) -> Result<serde_json::Value, Box<dyn Error>> {
    let heads = output::validate_maia(raw, &case.indices)?;
    let logits = compare(&raw.policy_logits, &case.policy_logits, 1e-4, 1e-3)?;
    let wdl = compare(&raw.wdl, &case.wdl, 1e-4, 0.0)?;
    let policy = compare(heads.policy(), &case.legal_policy, 1e-4, 0.0)?;
    let reverse = case.indices.iter().copied().rev().collect::<Vec<_>>();
    let permuted = output::validate_maia(raw, &reverse)?;
    let expected = heads.policy().iter().copied().rev().collect::<Vec<_>>();
    compare(permuted.policy(), &expected, 1e-7, 0.0)?;
    assert_eq!(permuted.wdl(), heads.wdl());
    Ok(
        json!({"case": case.name, "logit_max_abs": logits, "wdl_max_abs": wdl, "legal_policy_max_abs": policy}),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let option_start = args
        .iter()
        .position(|arg| arg.starts_with("--"))
        .unwrap_or(args.len());
    let (args, options) = args.split_at(option_start);
    let mut benchmark_rounds = None;
    let mut runtime_cache_root = None;
    let mut execution_options = Vec::new();
    for option in options {
        if let Some(value) = option.strip_prefix("--benchmark-rounds=") {
            let count = value.parse::<usize>()?;
            if !(1..=100).contains(&count) || benchmark_rounds.replace(count).is_some() {
                return Err("benchmark rounds must be a unique 1..=100 limit".into());
            }
        } else if let Some(value) = option.strip_prefix("--runtime-cache-root=") {
            if value.is_empty() || runtime_cache_root.replace(Path::new(value)).is_some() {
                return Err("runtime cache root must be a unique absolute path".into());
            }
        } else {
            execution_options.push(option.clone());
        }
    }
    let experiments = experimental_options(&execution_options)?;
    if benchmark_rounds.is_some() && experiments != ExecutionExperiments::default() {
        return Err("inference baseline benchmark excludes execution experiments".into());
    }
    if !(8..=10).contains(&args.len()) {
        return Err("usage: maia_check SOURCE.pb.gz MODEL.onnx MANIFEST.json ORT_LIBRARY ORT_SHA256 FIXTURES.json REPORT.json cpu | cuda PROFILE_DIRECTORY CUDA_BUNDLE.json [--experimental-io-buffers] [--experimental-io-binding] [--experimental-cuda-graph]".into());
    }
    let is_cuda = match args[7].as_str() {
        "cpu" if (8..=9).contains(&args.len()) => false,
        "cuda" if args.len() == 10 => true,
        _ => return Err("CPU keeps its existing arguments; CUDA requires a fresh profile directory and explicit bundle manifest".into()),
    };
    if experiments.cuda_graph && !is_cuda {
        return Err("CUDA Graph numerical check requires explicit CUDA provider".into());
    }
    let bytes = asset::read_bounded(Path::new(&args[5]), 8 * 1024 * 1024)?;
    let fixtures: Fixtures = serde_json::from_slice(&bytes)?;
    let model = MaiaAsset::load(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
    )?;
    if fixtures.schema != 1
        || fixtures.reference != "lc0-v0.32.1-eigen-original-protobuf"
        || fixtures.reference_commit != asset::CONVERTER_COMMIT
        || fixtures.source_sha256 != model.profile().gzip_sha256()
        || fixtures.cases.len() != 12
    {
        return Err("reference metadata or finite fixture count differs".into());
    }
    asset::parse_sha256(&fixtures.reference_module_sha256)?;
    let report_path = Path::new(&args[6]);
    if !report_path.is_absolute() {
        return Err("REPORT must be absolute and have an existing caller-owned parent".into());
    }
    let output_root = report_path
        .parent()
        .ok_or("REPORT has no bootstrap output parent")?;
    let canonical_root = output_root
        .canonicalize()
        .map_err(|_| "bootstrap output parent must already exist")?;
    for ancestor in canonical_root.ancestors() {
        if ancestor
            .join(".git")
            .try_exists()
            .map_err(|_| "cannot verify bootstrap root is outside Git")?
        {
            return Err("runtime/report root must be outside a Git checkout".into());
        }
    }
    let mut config = BackendConfig::cpu();
    config.experiments = experiments;
    if experiments.cuda_graph {
        // Graph has one fixed B1 shape; larger batches are explicitly excluded.
        config.max_batch = 1;
    }
    let bundle_spec = if is_cuda {
        let spec_bytes = asset::read_bounded(Path::new(&args[9]), 64 * 1024)?;
        let spec = CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&spec_bytes)?)?;
        let core = spec
            .files
            .iter()
            .find(|file| file.role == RuntimeBundleFileRole::Core)
            .ok_or("CUDA bundle has no core declaration")?;
        let declared_core = Path::new(&args[3]);
        if declared_core.file_name().and_then(|name| name.to_str()) != Some(core.filename.as_str())
            || core.sha256 != args[4]
        {
            return Err("ORT_LIBRARY/ORT_SHA256 must match the explicit CUDA bundle core".into());
        }
        Some(spec)
    } else {
        None
    };
    let runtime_cache =
        runtime_cache_root.map_or_else(RuntimeCache::for_user, RuntimeCache::open)?;
    let pin = if let Some(spec) = &bundle_spec {
        runtime_cache.cuda_bundle(
            Path::new(&args[3])
                .parent()
                .ok_or("CUDA core has no bundle source parent")?,
            spec,
        )?
    } else {
        runtime_cache.library(Path::new(&args[3]), &args[4])?
    };
    if is_cuda {
        let profile_directory = Path::new(&args[8]);
        if !profile_directory.is_absolute()
            || profile_directory
                .parent()
                .ok_or("profile directory has no parent")?
                .canonicalize()?
                != canonical_root
        {
            return Err("CUDA profile directory must be a fresh direct child of REPORT's caller-owned parent".into());
        }
        let builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        let builder = {
            use std::os::unix::fs::DirBuilderExt;
            let mut builder = builder;
            builder.mode(0o700);
            builder
        };
        // create, rather than create_dir_all: an existing probe namespace
        // cannot supply stale placement evidence or overwrite earlier logs.
        builder.create(profile_directory)?;
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: model.profile().cuda_arena_bytes(),
        };
        config.profiling_prefix = Some(profile_directory.join("placement"));
    }
    let runtime = OrtRuntime::load(&pin)?;
    let mut backend = OnnxBackend::load(&runtime, &model, config)?;
    let mut encoded = Vec::new();
    for case in &fixtures.cases {
        let history = case
            .frames
            .iter()
            .map(|f| Frame {
                pieces: f.pieces,
                repeated: f.repeated,
                en_passant_target: f.en_passant_target,
            })
            .collect::<Vec<_>>();
        let history_fill = match case.history_fill.as_str() {
            "no" => HistoryFill::No,
            "repeat_oldest" => HistoryFill::RepeatOldest,
            _ => return Err("unknown history profile".into()),
        };
        let input = classical::encode(Input {
            history: &history,
            black_to_move: case.black_to_move,
            castling: case.castling,
            halfmove_clock: case.halfmove_clock,
            history_fill,
        })?;
        compare(input.values(), &case.input, 0.0, 0.0)
            .map_err(|e| format!("{} encoding: {e}", case.name))?;
        let mut indices = Vec::new();
        for m in &case.legal_moves {
            let from = policy::canonical_square(m.from_square, case.black_to_move)?;
            let to = policy::canonical_square(m.to_square, case.black_to_move)?;
            indices.push(if m.castle {
                policy::castling_index(
                    from,
                    if to == 6 {
                        7
                    } else if to == 2 {
                        0
                    } else {
                        return Err("invalid standard castle".into());
                    },
                )?
            } else {
                policy::index(from, to, m.promotion)?
            });
        }
        if indices != case.indices {
            return Err(format!("{} legal action indices differ", case.name).into());
        }
        encoded.push(input);
    }
    // Same board with known repetition/history must remain a different NN input.
    assert_ne!(encoded[3].values(), encoded[4].values());
    let mut singles = Vec::new();
    let mut errors = Vec::new();
    for (input, case) in encoded.iter().zip(&fixtures.cases) {
        let raw = backend.run(&[input])?.remove(0);
        errors.push(verify(&raw, case).map_err(|e| format!("{}: {e}", case.name))?);
        singles.push(raw);
    }
    // Same immutable start-position tensor; B1 warmed above, first B16 included.
    // These are host wall intervals around C Run, not device kernel timings.
    let mut inference_benchmark = Vec::new();
    if let Some(rounds) = benchmark_rounds {
        for size in [1, 16] {
            let inputs = vec![&encoded[0]; size];
            for round in 0..rounds {
                let started = std::time::Instant::now();
                let results = backend.run(&inputs)?;
                let elapsed = started.elapsed();
                if results.len() != size {
                    return Err("benchmark batch incomplete".into());
                }
                for raw in &results {
                    verify(raw, &fixtures.cases[0])?;
                }
                inference_benchmark.push(
                    json!({"batch":size,"round":round,"completed_inferences":size,
                    "host_run_ns":elapsed.as_nanos()}),
                );
            }
        }
    }
    let mut batch_checks = Vec::new();
    let batch_sizes: &[usize] = if experiments.cuda_graph {
        &[1]
    } else {
        &[1, 2, 4, 8, 16]
    };
    for &size in batch_sizes {
        let order = (0..size)
            .map(|i| (size - 1 - i) % encoded.len())
            .collect::<Vec<_>>();
        let inputs = order.iter().map(|&i| &encoded[i]).collect::<Vec<_>>();
        let results = backend.run(&inputs)?;
        let mut max_abs: f64 = 0.0;
        for (raw, &i) in results.iter().zip(&order) {
            verify(raw, &fixtures.cases[i])?;
            max_abs = max_abs.max(compare(
                &raw.policy_logits,
                &singles[i].policy_logits,
                1e-4,
                1e-3,
            )?);
            compare(&raw.wdl, &singles[i].wdl, 1e-4, 0.0)?;
        }
        batch_checks.push(json!({"batch":size,"single_vs_batch_logit_max_abs":max_abs}));
    }
    let mut experimental_checks = None;
    if experiments != ExecutionExperiments::default() {
        let mut phases = Vec::new();
        // Same session/shape and varying actual inputs: retain independent
        // outputs, then move only the new outputs back to the bounded pool.
        for round in 0..32 {
            let i = round % encoded.len();
            let results = backend.run(&[&encoded[i]])?;
            verify(&results[0], &fixtures.cases[i])?;
            if let Some(timing) = backend.last_io_timings() {
                phases.push(json!({"round":round, "case":fixtures.cases[i].name,
                    "host_stage_seconds":timing.host_stage.as_secs_f64(),
                    "transfer_in_seconds":timing.transfer_in.map(|d| d.as_secs_f64()),
                    "run_seconds":timing.run.as_secs_f64(),
                    "output_fence_seconds":timing.output_fence.map(|d| d.as_secs_f64()),
                    "transfer_out_seconds":timing.transfer_out.map(|d| d.as_secs_f64()),
                    "own_outputs_seconds":timing.own_outputs.as_secs_f64()}));
            }
            backend.recycle_outputs(results);
        }
        // Later reuse must not mutate outputs which the caller still owns.
        for (raw, case) in singles.iter().zip(&fixtures.cases) {
            verify(raw, case)?;
        }
        experimental_checks = Some(json!({
            "reuse_buffers":experiments.reuse_buffers,
            "io_binding":experiments.io_binding,
            "cuda_graph_requested":experiments.cuda_graph,
            "repeated_B1_calls":32, "retained_outputs_unchanged":true,
            "binding_runs":backend.binding_runs(),
            "phase_clock":"CPU wall boundaries; Run includes kernels/synchronization",
            "phases":phases,
            "excluded_batch_sizes":if experiments.cuda_graph { vec![2,4,8,16] } else { vec![] },
            "capture_replay_observed":false,
            "formal_performance_acceptance":false
        }));
    }
    assert!(backend.run(&[]).is_err());
    assert!(backend.run(&[&encoded[0]; 17]).is_err());
    let mut invalid = encoded[0].values().to_vec();
    invalid[0] = f32::NAN;
    assert!(backend.run_values(&[&invalid]).is_err());
    // A rejected call must leave the session usable.
    verify(&backend.run(&[&encoded[0]])?.remove(0), &fixtures.cases[0])?;
    let cuda_executed_nodes = backend.cuda_evidence().map(|e| e.executed_cuda_nodes);
    let cuda_profile_sha256 = backend.cuda_evidence().map(|e| {
        e.profile_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    });
    let backend_sha256 = backend
        .identity()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    #[cfg(feature = "contracts")]
    let contract_worker = verify_contract_worker(
        backend,
        &fixtures.cases,
        &encoded,
        if experiments.cuda_graph { 1 } else { 4 },
    )?;
    #[cfg(not(feature = "contracts"))]
    let contract_worker = json!({"status":"not_enabled"});
    if is_cuda {
        // Batch-dependent or later lazy loads must still belong to the pinned
        // closure when final evidence is admitted, not only at the B1 probe.
        runtime.verify_cuda_runtime_mappings()?;
    }
    let mut report = json!({"status":"passed", "provider": args[7], "precision":"fp32", "tf32":false,
        "reference":fixtures.reference, "fixture_sha256":asset::hex_sha256(&bytes),
        "reference_commit":fixtures.reference_commit,"reference_module_sha256":fixtures.reference_module_sha256,
        "runtime_sha256":args[4], "runtime_build":runtime.build_info(),
        "runtime_storage":pin.storage(),
        "cuda_bundle_sha256":runtime.bundle_digest().map(|digest| digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>()),
        "cuda_bundle_files":pin.bundle_files().map(|files| files.iter().map(|file| {
            let role = match file.role {
                RuntimeBundleFileRole::Core => "core",
                RuntimeBundleFileRole::ProvidersShared => "providers_shared",
                RuntimeBundleFileRole::ProvidersCuda => "providers_cuda",
                RuntimeBundleFileRole::NvidiaDependency => "nvidia_dependency",
            };
            json!({"role":role,"filename":file.filename,"bytes":file.bytes,"sha256":file.sha256})
        }).collect::<Vec<_>>()),
        "backend_sha256":backend_sha256,
        "onnx_sha256":model.manifest().onnx_sha256,
        "tolerances":{"raw_logits_atol":1e-4,"raw_logits_rtol":1e-3,"wdl_max_abs":1e-4,"legal_policy_max_abs":1e-4},
        "case_errors":errors,"batch_checks":batch_checks,
        "cuda_executed_nodes":cuda_executed_nodes,"cuda_profile_sha256":cuda_profile_sha256, "contract_worker": contract_worker,
        "gpu_acceptance": if args[7] == "cpu" { "not_run" } else { "numerical_and_provider_probe_only" }});
    if !inference_benchmark.is_empty() {
        report["inference_benchmark"] = json!({
            "scope":"same fixed start-position input; synchronous host C Run includes provider transfers and inference",
            "input_f32_le_sha256":asset::hex_sha256(&encoded[0].values().iter()
                .flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>()),
            "history_fill":"no","cache":"disabled","warmup":"numerical B1 cases precede timing; first B16 included",
            "samples":inference_benchmark,"device_kernel_timing":"not measured","device_transfer_timing":"not measured"});
    }
    if let Some(checks) = experimental_checks {
        report["experimental_checks"] = checks;
    }
    std::fs::write(&args[6], serde_json::to_vec_pretty(&report)?)?;
    println!(
        "passed: {} cases, batches {}, provider={}",
        encoded.len(),
        batch_sizes
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("/"),
        args[7]
    );
    Ok(())
}

fn experimental_options(options: &[String]) -> Result<ExecutionExperiments, Box<dyn Error>> {
    let mut experiments = ExecutionExperiments::default();
    for option in options {
        let (enabled, compiled) = match option.as_str() {
            "--experimental-io-buffers" => (
                &mut experiments.reuse_buffers,
                cfg!(feature = "experimental-io-buffers"),
            ),
            "--experimental-io-binding" => (
                &mut experiments.io_binding,
                cfg!(feature = "experimental-io-binding"),
            ),
            "--experimental-cuda-graph" => (
                &mut experiments.cuda_graph,
                cfg!(feature = "experimental-cuda-graph"),
            ),
            _ => return Err("unknown execution experiment".into()),
        };
        if *enabled || !compiled {
            return Err("duplicate or uncompiled execution experiment".into());
        }
        *enabled = true;
    }
    if experiments.cuda_graph && !experiments.io_binding {
        return Err("CUDA Graph numerical check requires explicit I/O Binding".into());
    }
    Ok(experiments)
}

#[cfg(feature = "contracts")]
fn verify_contract_worker(
    backend: OnnxBackend,
    cases: &[Case],
    encoded: &[classical::EncodedInput],
    width: usize,
) -> Result<serde_json::Value, Box<dyn Error>> {
    use rz_contracts::*;
    use rz_eval::contracts::{self, MaiaBinding, PreparedBatch};
    use rz_eval::worker::PhysicalPoll;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let encoding = EncodingHandle {
        owner: OwnerId(1),
        slot: 1,
        generation: SlotGeneration(1),
        manifest: contracts::encoding_manifest(HistoryFill::No),
    };
    let model = ModelHandle {
        owner: OwnerId(1),
        slot: 1,
        generation: SlotGeneration(1),
        manifest: Digest(backend.asset_identity()),
    };
    let binding = MaiaBinding::for_backend(&backend, model, encoding, HistoryFill::No)?;
    let epoch = ProcessEpoch(100);
    let mut prepared = Vec::new();
    let mut requests = Vec::new();
    // Immutable fixture Rules views from the external oracle, not an A integration claim.
    for (i, case) in cases.iter().take(width).enumerate() {
        let state = StateIdentity {
            owner: OwnerId(2),
            revision: StateRevision(i as u64),
            semantic: Digest(asset::sha256(case.name.as_bytes())),
        };
        let position = PositionSnapshot::try_new(
            state,
            Arc::new(()),
            if case.black_to_move {
                Color::Black
            } else {
                Color::White
            },
            PositionClassification {
                play_status: PlayStatus::Ongoing,
                claims: Arc::from([]),
                history: HistoryCompleteness::UnknownPrefix,
                rules_profile: Digest([3; 32]),
            },
        )?;
        let mut moves = Vec::new();
        for m in &case.legal_moves {
            let promotion = match m.promotion {
                Some('q') => Some(Promotion::Queen),
                Some('r') => Some(Promotion::Rook),
                Some('b') => Some(Promotion::Bishop),
                Some('n') => Some(Promotion::Knight),
                None => None,
                _ => return Err("bad fixture promotion".into()),
            };
            moves.push(Move::new(
                Square::try_new(m.from_square)?,
                Square::try_new(m.to_square)?,
                promotion,
            )?);
        }
        let legal = LegalMoveView::try_new(state, LegalOrderIdentity(state.semantic), moves, 256)?;
        let context = EvalContext {
            revision: CONTRACT_REVISION,
            request: RequestId::new(epoch, i as u64 + 1),
            selection: SelectionId::new(epoch, i as u64 + 1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            state,
            legal_order: legal.order(),
            input: contracts::input_key(encoding, &encoded[i]),
            model,
            encoding,
            precision: PrecisionProfile::Fp32,
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            backend: binding.backend(),
        };
        let request = Arc::new(EvalRequest::try_new(
            context,
            position,
            legal,
            Arc::clone(binding.model()),
            Deadline {
                clock: ClockDomain(epoch),
                at: MonotonicTick(100),
            },
            CancelToken::new(),
            ByteBudget {
                host: contracts::HOST_BYTES_PER_ITEM,
                device: 0,
                pinned: 0,
            },
        )?);
        let frames = case
            .frames
            .iter()
            .map(|f| Frame {
                pieces: f.pieces,
                repeated: f.repeated,
                en_passant_target: f.en_passant_target,
            })
            .collect::<Vec<_>>();
        prepared.push(binding.prepare(
            Arc::clone(&request),
            Input {
                history: &frames,
                black_to_move: case.black_to_move,
                castling: case.castling,
                halfmove_clock: case.halfmove_clock,
                history_fill: HistoryFill::No,
            },
        )?);
        requests.push(request);
    }
    let execution = ExecutionId::new(epoch, 50);
    let mut worker = contracts::spawn_onnx_worker(backend)?;
    let mut lease = worker.submit(PreparedBatch::new(execution, prepared)?)?;
    if width > 1 {
        requests[0].cancel_token().cancel();
    }
    let stop = Instant::now() + Duration::from_secs(5);
    let outputs = loop {
        match lease.poll() {
            PhysicalPoll::Ready(result) => break result?,
            PhysicalPoll::Pending if Instant::now() < stop => std::thread::yield_now(),
            PhysicalPoll::Quarantined => {
                return Err(lease.quarantine_cause()?.unwrap_or_else(|| {
                    rz_eval::error::BackendError::new(
                        rz_eval::error::FailureKind::BackendFailure,
                        rz_eval::error::FailureStage::Backend,
                        "physical worker completion is unknown and no quarantine receipt is available",
                    )
                }).into());
            }
            _ => return Err("physical worker did not complete within fixture budget".into()),
        }
    };
    if outputs.len() != requests.len() {
        return Err("physical result count differs".into());
    }
    for (i, (output, request)) in outputs.iter().zip(&requests).enumerate() {
        assert_eq!(output.context, request.context());
        assert_eq!(output.actual.execution, Some(execution));
        compare(
            &output
                .policy
                .probabilities()
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>(),
            &cases[i].legal_policy,
            1e-4,
            0.0,
        )?;
        let scope = AcceptanceScope {
            game: GameGeneration(1),
            root: RootGeneration(1),
            model,
            encoding,
            backend: binding.backend(),
        };
        let acceptance = output.validate_for(request, scope, ClockDomain(epoch), MonotonicTick(1));
        if i == 0 && width > 1 {
            assert_eq!(acceptance.unwrap_err().code, ErrorCode::Canceled);
        } else {
            acceptance?;
        }
    }
    assert!(matches!(lease.poll(), PhysicalPoll::Consumed));
    Ok(
        json!({"status":"passed","physical_outputs":outputs.len(),"canceled_rejected":usize::from(width > 1),
        "scope":"C worker plus contract 0.1; fixture Rules view; D scheduler not linked"}),
    )
}
