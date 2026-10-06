//! Actual A Rules → explicit C ONNX CPU/CUDA → D finalization numerical acceptance.
//!
//! External reference frames never construct the product position or legal view.
//! Model/encoding/position owners are assigned locally for this finite fixture;
//! this example makes no production registry, B UCI, or strength claim. CUDA is
//! separately selected and requires its own fresh pinned runtime/profile inputs.
use rz_contracts::*;
use rz_encoding::classical::{HistoryFill, HISTORY_FRAMES};
use rz_eval::asset::{self, MaiaAsset};
use rz_eval::contracts::{encoding_manifest, MaiaBinding, HOST_BYTES_PER_ITEM};
use rz_eval::native_runtime_bridge::{NativeRuntimeBackend, NativeWorkerOrigin, NativeWorkerOwner};
use rz_eval::onnx::{BackendConfig, OnnxBackend, OrtRuntime, Provider};
use rz_eval::rules_projection::ClassicalProjection;
use rz_eval::runtime_pin::{CudaRuntimeBundleSpec, RuntimeBundleFileRole, RuntimeCache};
use rz_position::contracts::{ContractPosition, ContractState, RulesState};
use rz_position::{BoardMove, Position};
use rz_runtime::contracts::{
    ContractClock, ContractEvaluator, ContractSystemClock, ContractsAdapter, SharedScope,
};
use rz_runtime::{DrainState, Limits, Resources};
use serde::Deserialize;
use serde_json::{json, Value};
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type NativeEvaluator =
    ContractEvaluator<RulesState, NativeRuntimeBackend<ContractSystemClock>, ContractSystemClock>;
const REQUEST_SECONDS: u64 = 5;
const SHUTDOWN_SECONDS: u64 = 5;
const CASE_NAMES: [&str; 12] = [
    "start",
    "black-after-e4",
    "eight-ply-opening",
    "repeated-start-with-history",
    "same-board-unknown-prefix",
    "white-en-passant",
    "black-en-passant",
    "white-both-castles",
    "black-both-castles",
    "white-all-promotions",
    "black-all-promotions",
    "short-fen-history",
];

fn hex_digest(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixtures {
    schema: u32,
    reference: String,
    reference_commit: String,
    reference_module_sha256: String,
    source_sha256: String,
    fixture_oracle: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    fen: String,
    moves: Vec<String>,
    frames: Vec<ReferenceFrame>,
    black_to_move: bool,
    castling: [bool; 4],
    halfmove_clock: u32,
    history_fill: String,
    input: Vec<f32>,
    legal_moves: Vec<ReferenceMove>,
    indices: Vec<usize>,
    policy_logits: Vec<f32>,
    wdl: Vec<f32>,
    legal_policy: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceFrame {
    pieces: [[u64; 6]; 2],
    repeated: bool,
    en_passant_target: Option<u8>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceMove {
    from_square: u8,
    to_square: u8,
    promotion: Option<char>,
    castle: bool,
}

fn compare(actual: &[f32], expected: &[f32], atol: f64, rtol: f64) -> Result<f64> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("comparison shape differs".into());
    }
    let mut maximum: f64 = 0.0;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let error = (f64::from(actual) - f64::from(expected)).abs();
        if !actual.is_finite()
            || !expected.is_finite()
            || error > atol + rtol * f64::from(expected).abs()
        {
            return Err(format!("numerical mismatch: absolute error={error}").into());
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}

fn probability_vector(values: &[f32], length: usize) -> bool {
    values.len() == length
        && !values.is_empty()
        && values
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        && (values.iter().map(|&value| f64::from(value)).sum::<f64>() - 1.0).abs() <= 1e-5
}

fn reference_move(movement: &ReferenceMove) -> Result<Move> {
    let promotion = match movement.promotion {
        None => None,
        Some('q') => Some(Promotion::Queen),
        Some('r') => Some(Promotion::Rook),
        Some('b') => Some(Promotion::Bishop),
        Some('n') => Some(Promotion::Knight),
        _ => return Err("unsupported reference promotion".into()),
    };
    if movement.castle && promotion.is_some() {
        return Err("reference castle has a promotion".into());
    }
    Ok(Move::new(
        Square::try_new(movement.from_square)?,
        Square::try_new(movement.to_square)?,
        promotion,
    )?)
}

fn validate_reference(fixtures: &Fixtures, model: &MaiaAsset) -> Result<()> {
    if fixtures.schema != 1
        || fixtures.reference != "lc0-v0.32.1-eigen-original-protobuf"
        || fixtures.reference_commit != asset::CONVERTER_COMMIT
        || fixtures.source_sha256 != model.profile().gzip_sha256()
        || fixtures.fixture_oracle != "python-chess-1.999/chess-1.11.2"
        || fixtures.cases.len() != CASE_NAMES.len()
    {
        return Err("reference metadata or finite case set differs".into());
    }
    asset::parse_sha256(&fixtures.reference_module_sha256)?;
    for (i, (case, name)) in fixtures.cases.iter().zip(CASE_NAMES).enumerate() {
        let count = case.legal_moves.len();
        let expected_profile = if i < 5 { "no" } else { "repeat_oldest" };
        if case.name != name
            || case.fen.len() > 256
            || case.moves.len() > 8
            || case
                .moves
                .iter()
                .any(|movement| !(4..=5).contains(&movement.len()))
            || case.frames.len() != (case.moves.len() + 1).min(HISTORY_FRAMES)
            || case
                .frames
                .iter()
                .any(|frame| frame.en_passant_target.is_some_and(|sq| sq >= 64))
            || case.input.len() != 112 * 64
            || case.input.iter().any(|value| !value.is_finite())
            || case.policy_logits.len() != 1858
            || case.policy_logits.iter().any(|value| !value.is_finite())
            || count == 0
            || count > rz_position::contracts::MAX_CONTRACT_LEGAL_MOVES
            || case.indices.len() != count
            || case.indices.iter().any(|&index| index >= 1858)
            || !probability_vector(&case.wdl, 3)
            || !probability_vector(&case.legal_policy, count)
            || case.history_fill != expected_profile
        {
            return Err(format!("invalid reference dimensions/profile: {}", case.name).into());
        }
        let moves = case
            .legal_moves
            .iter()
            .map(reference_move)
            .collect::<Result<Vec<_>>>()?;
        if moves
            .iter()
            .enumerate()
            .any(|(i, movement)| moves[..i].contains(movement))
            || case
                .indices
                .iter()
                .enumerate()
                .any(|(i, index)| case.indices[..i].contains(index))
        {
            return Err("duplicate reference move or action index".into());
        }
    }
    if fixtures.cases[3].input == fixtures.cases[4].input {
        return Err("known repeated history and unknown prefix collapsed to one NN input".into());
    }
    Ok(())
}

fn backend_receipt(error: &rz_eval::error::BackendError) -> Value {
    json!({"kind":format!("{:?}",error.kind),"stage":format!("{:?}",error.stage),
        "detail":error.detail,
        "cause":error.cause.map(|cause| json!({"code":format!("{:?}",cause.code),
            "prefix_sha256":cause.prefix_sha256,"hashed_bytes":cause.hashed_bytes,
            "truncated":cause.truncated,"formatting_failed":cause.formatting_failed,
            "output":cause.output.map(|output| format!("{output:?}"))})),
        "native":error.native.as_ref().map(|native| json!({"code":native.code,
            "message_bytes":native.message.len(),"truncated":native.truncated,
            "raw_text":"bounded native text is excluded from this report"}))})
}

fn error_receipt(error: &(dyn Error + 'static)) -> Value {
    if let Some(physical) = error.downcast_ref::<rz_eval::contracts::PhysicalFailure>() {
        json!({"kind":"PhysicalFailure","contract":format!("{:?}",physical.contract),
            "backend":physical.backend.as_ref().map(backend_receipt)})
    } else if let Some(backend) = error.downcast_ref::<rz_eval::error::BackendError>() {
        backend_receipt(backend)
    } else {
        json!({"kind":"fixture_or_contract_failure","detail":error.to_string()})
    }
}

fn restore(case: &Case, owner: OwnerId) -> Result<ContractState> {
    // Literal FEN imports remain unknown-prefix. Do not turn model padding into
    // Rules history, or claim FEN-before history is complete because it looks like startpos.
    let mut position = Position::from_fen(&case.fen)?;
    for movement in &case.moves {
        position.make_move(BoardMove::from_uci(movement)?)?;
    }
    let mut live = ContractPosition::new(owner, position);
    let frozen = live.export()?;
    if frozen.terminal_wdl().is_some() {
        return Err("reference unexpectedly restores a terminal".into());
    }
    // Advance the live owner after export; every later comparison consumes only
    // the already-owned immutable A snapshot and its exact legal attestation.
    let first = *frozen
        .legal_moves()
        .moves()
        .first()
        .ok_or("empty A legal view")?;
    live.make_from_view(&frozen, first)?;
    Ok(frozen)
}

fn compare_projection(
    projection: &ClassicalProjection,
    frozen: &ContractState,
    case: &Case,
) -> Result<Vec<f32>> {
    let projected = projection.project(frozen.rules())?;
    let input = projected.input();
    if input.black_to_move != case.black_to_move
        || input.castling != case.castling
        || input.halfmove_clock != case.halfmove_clock
        || input.history.len() != case.frames.len()
    {
        return Err(format!("actual A frame metadata differs: {}", case.name).into());
    }
    for (actual, expected) in input.history.iter().zip(&case.frames) {
        if actual.pieces != expected.pieces
            || actual.repeated != expected.repeated
            || actual.en_passant_target != expected.en_passant_target
        {
            return Err(format!("actual A known frame differs: {}", case.name).into());
        }
    }
    let encoded = projection.encode(frozen.rules())?;
    compare(encoded.values(), &case.input, 0.0, 0.0)?;
    let actual_moves = frozen.legal_moves().moves();
    let reference_moves = case
        .legal_moves
        .iter()
        .map(reference_move)
        .collect::<Result<Vec<_>>>()?;
    if actual_moves.len() != reference_moves.len() {
        return Err("A/reference legal move counts differ".into());
    }
    let actual_indices = projection.legal_indices(frozen.rules(), actual_moves)?;
    let mut reordered = Vec::with_capacity(actual_moves.len());
    for (movement, actual_index) in actual_moves.iter().zip(actual_indices) {
        let reference_index = reference_moves
            .iter()
            .position(|reference| reference == movement)
            .ok_or("actual A legal move absent from independent reference")?;
        if actual_index != case.indices[reference_index] {
            return Err("actual A legal action mapping differs".into());
        }
        reordered.push(case.legal_policy[reference_index]);
    }
    Ok(reordered)
}

fn deadline(clock: &ContractSystemClock) -> Result<Deadline> {
    Ok(Deadline {
        clock: clock.domain(),
        at: MonotonicTick(
            clock
                .try_now()?
                .0
                .checked_add(REQUEST_SECONDS * 1_000_000_000)
                .ok_or("fixture deadline overflow")?,
        ),
    })
}

fn request(
    projection: &ClassicalProjection,
    frozen: &ContractState,
    clock: &ContractSystemClock,
    sequence: u64,
) -> Result<Arc<EvalRequest<RulesState>>> {
    let epoch = clock.domain().0;
    let context = EvalContext {
        revision: CONTRACT_REVISION,
        request: RequestId::new(epoch, sequence),
        selection: SelectionId::new(epoch, sequence),
        game: GameGeneration(1),
        root: RootGeneration(1),
        state: frozen.snapshot().identity(),
        legal_order: frozen.legal_moves().order(),
        input: projection.input_key(frozen.rules(), frozen.legal_moves().moves())?,
        model: projection.model().handle(),
        encoding: projection.model().encoding().handle,
        precision: PrecisionProfile::Fp32,
        compute: ComputeBudget {
            min_steps: 1,
            max_steps: 1,
            require_full: true,
        },
        backend: projection.backend(),
    };
    Ok(Arc::new(EvalRequest::try_new(
        context,
        frozen.snapshot().clone(),
        frozen.legal_moves().clone(),
        Arc::clone(projection.model()),
        deadline(clock)?,
        CancelToken::new(),
        ByteBudget {
            host: HOST_BYTES_PER_ITEM + 4096,
            device: 0,
            pinned: 0,
        },
    )?))
}

fn drain(evaluator: &mut NativeEvaluator, clock: &ContractSystemClock) -> Result<Value> {
    evaluator.begin_shutdown(deadline(clock)?)?;
    let until = Instant::now() + Duration::from_secs(REQUEST_SECONDS);
    let mut extra_deliveries = Vec::new();
    let mut extra_overflow = false;
    loop {
        evaluator.pump();
        // Preserve unexpected deliveries while still finishing physical cleanup.
        if let Some(result) = evaluator.poll() {
            if extra_deliveries.len() < CASE_NAMES.len() {
                extra_deliveries.push(format!("{result:?}"));
            } else {
                extra_overflow = true;
            }
        }
        let snapshot = evaluator.shutdown_snapshot();
        if snapshot.drain == DrainState::Drained {
            let state = snapshot.state;
            if state.logical_requests != 0
                || state.reserved_requests != 0
                || state.queued != 0
                || state.executions != 0
                || state.reserved != Resources::default()
            {
                return Err("physical drain left logical/delivery reservations".into());
            }
            return Ok(
                json!({"physical":"drained","logical_requests":0,"executions":0,
                "reserved_requests":0,"reserved_host":0,"reserved_device":0,"reserved_pinned":0,
                "unexpected_deliveries":extra_deliveries,"unexpected_delivery_overflow":extra_overflow}),
            );
        }
        if Instant::now() >= until {
            return Err(format!("native drain incomplete: {snapshot:?}").into());
        }
        std::thread::yield_now();
    }
}

fn shutdown_native(owner: &NativeWorkerOwner, budget: Duration) -> Result<Value> {
    let until = Instant::now() + budget;
    loop {
        match owner.try_shutdown() {
            std::task::Poll::Ready(result) => {
                result?;
                return Ok(json!({"native":"joined","session_destruction":"complete",
                    "worker_thread_exit":"joined"}));
            }
            std::task::Poll::Pending if Instant::now() < until => {
                std::thread::sleep(Duration::from_millis(1));
            }
            std::task::Poll::Pending => {
                return Err("native worker shutdown is unconfirmed within fixture budget".into());
            }
        }
    }
}

struct ProfileExecution<'a> {
    fill: HistoryFill,
    profile_slot: u64,
    cuda_profile_directory: Option<&'a Path>,
    raw_cache: bool,
}

fn case_sequence(index: usize, raw_cache: bool) -> u64 {
    // Reserve adjacent IDs for both replays before the next physical case.
    index as u64 * if raw_cache { 3 } else { 1 } + 1
}

fn finalized(evaluator: &mut NativeEvaluator, case: &str) -> Result<EvalOutput> {
    let until = Instant::now() + Duration::from_secs(REQUEST_SECONDS);
    loop {
        if let Some(result) = evaluator.poll() {
            return match result {
                EvalResult::Completed(output) => Ok(output),
                other => Err(format!("native finalized failure: {other:?}").into()),
            };
        }
        if Instant::now() >= until {
            return Err(format!("native completion timeout: {case}").into());
        }
        std::thread::yield_now();
    }
}

fn evaluate_profile(
    runtime: &OrtRuntime,
    model: &MaiaAsset,
    fixtures: &Fixtures,
    execution: ProfileExecution<'_>,
    reports: &mut Vec<Value>,
    profiles: &mut Vec<Value>,
) -> Result<()> {
    let ProfileExecution {
        fill,
        profile_slot,
        cuda_profile_directory,
        raw_cache,
    } = execution;
    let mut config = BackendConfig::cpu();
    config.max_batch = 1;
    if let Some(directory) = cuda_profile_directory {
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: model.profile().cuda_arena_bytes(),
        };
        config.profiling_prefix = Some(directory.join(format!("placement-{profile_slot}")));
    }
    let loaded = OnnxBackend::load(runtime, model, config)?;
    // Explicit local fixture ownership; these are not production-issued handles.
    let encoding = EncodingHandle {
        owner: OwnerId(900),
        slot: profile_slot,
        generation: SlotGeneration(1),
        manifest: encoding_manifest(fill),
    };
    let handle = ModelHandle {
        owner: OwnerId(900),
        slot: profile_slot,
        generation: SlotGeneration(1),
        manifest: Digest(loaded.asset_identity()),
    };
    let projection =
        ClassicalProjection::new(MaiaBinding::for_backend(&loaded, handle, encoding, fill)?);
    #[cfg(feature = "experimental-raw-cache")]
    if raw_cache {
        projection.configure_raw_cache(rz_eval::raw_cache::RawCacheLimits {
            max_entries: 16,
            max_bytes: 2 * 1024 * 1024,
        })?;
    }
    let profile = match fill {
        HistoryFill::No => "no",
        HistoryFill::RepeatOldest => "repeat_oldest",
    };
    let is_cuda = cuda_profile_directory.is_some();
    let owner = if is_cuda {
        NativeWorkerOwner::from_cuda_onnx(loaded, projection.clone(), CASE_NAMES.len())?
    } else {
        NativeWorkerOwner::from_onnx(loaded, projection.clone(), CASE_NAMES.len())?
    };
    let expected_origin = if is_cuda {
        NativeWorkerOrigin::CudaOnnx
    } else {
        NativeWorkerOrigin::CpuOnnx
    };
    if owner.origin() != expected_origin {
        return Err("injected worker is not actual requested-provider ONNX evidence".into());
    }
    let clock = ContractSystemClock::new(ProcessEpoch(910 + profile_slot));
    let scope = AcceptanceScope {
        game: GameGeneration(1),
        root: RootGeneration(1),
        model: handle,
        encoding,
        backend: projection.backend(),
    };
    let adapter =
        ContractsAdapter::with_execution_high_water(SharedScope::new(scope), clock.clone(), 1, 0)?;
    let backend = NativeRuntimeBackend::new(owner.clone(), clock.clone())?;
    let limits = Limits {
        max_requests: 1,
        max_batch_items: 1,
        max_executions: 1,
        max_batch_wait: Duration::ZERO,
        max_queue_age: Duration::from_secs(REQUEST_SECONDS),
        deadline_reserve: Duration::ZERO,
        memory: Resources {
            host_bytes: 4 * HOST_BYTES_PER_ITEM + 8192,
            device_bytes: owner.admission_policy().execution_resources().device_bytes,
            pinned_bytes: 0,
        },
    };
    let mut evaluator = ContractEvaluator::new(adapter, backend, limits, 64)?;
    #[cfg(feature = "experimental-raw-cache")]
    if raw_cache {
        evaluator.set_raw_reuse(Box::new(projection.raw_cache_provider()));
    }
    #[cfg(feature = "experimental-raw-cache")]
    let mut replay_count = 0_u64;
    #[cfg(not(feature = "experimental-raw-cache"))]
    let replay_count = 0_u64;
    let work = (|| -> Result<()> {
        for (i, case) in fixtures
            .cases
            .iter()
            .enumerate()
            .filter(|(_, case)| case.history_fill == profile)
        {
            let frozen = restore(case, OwnerId(1000 + i as u64))?;
            let reference_policy = compare_projection(&projection, &frozen, case)?;
            let sequence = case_sequence(i, raw_cache);
            let request = request(&projection, &frozen, &clock, sequence)?;
            evaluator.submit(Arc::clone(&request))?;
            let output = finalized(&mut evaluator, &case.name)?;
            output.validate_for(&request, scope, clock.domain(), clock.try_now()?)?;
            if output.actual.execution.is_none()
                || output.actual.provenance != CacheProvenance::Computed
                || output.actual.precision != PrecisionProfile::Fp32
                || output.actual.steps != 1
                || !output.actual.full
                || output.viewpoint != Viewpoint::SideToMove
            {
                return Err("D result is not fresh full FP32 physical inference".into());
            }
            let policy = output
                .policy
                .probabilities()
                .iter()
                .map(|&value| value as f32)
                .collect::<Vec<_>>();
            let policy_error = compare(&policy, &reference_policy, 1e-4, 0.0)?;
            let wdl_error = compare(&output.wdl.probabilities(), &case.wdl, 1e-4, 0.0)?;
            #[cfg(feature = "experimental-raw-cache")]
            if raw_cache {
                for replay in 0..2 {
                    let reused =
                        self::request(&projection, &frozen, &clock, sequence + replay + 1)?;
                    evaluator.submit(Arc::clone(&reused))?;
                    let cached = finalized(&mut evaluator, &case.name)?;
                    cached.validate_for(&reused, scope, clock.domain(), clock.try_now()?)?;
                    if cached.policy != output.policy
                        || cached.wdl != output.wdl
                        || cached.actual.execution.is_some()
                        || cached.actual.provenance
                            != (CacheProvenance::RawEvalHit {
                                source_execution: output.actual.execution,
                            })
                        || evaluator.state().executions != 0
                        || evaluator.state().reserved != Resources::default()
                    {
                        return Err("raw-cache replay differs or retains physical work".into());
                    }
                    replay_count += 1;
                }
            }
            reports.push(json!({"case":case.name,"history_fill":profile,
                "rules_origin":"literal_fen_unknown_prefix_with_actual_trace",
                "known_frames":case.frames.len(),"frozen_after_live_advance":true,
                "dense_input_max_abs":0.0,"ordered_legal_count":policy.len(),
                "legal_policy_max_abs":policy_error,"wdl_max_abs":wdl_error,
                "physical_execution":format!("{:?}",output.actual.execution),"finalized_by":"D"}));
        }
        Ok(())
    })();
    // Both failure and cleanup receipts survive even when the profile fails.
    let cleanup = drain(&mut evaluator, &clock);
    // One bounded checkpoint survives a fatal native destructor before the
    // final report can be written. This fixture has no periodic PID/GPU logger.
    eprintln!(
        "{}",
        json!({"stage":"before_native_shutdown","history_fill":profile,
        "work_passed":work.is_ok(),"physical_drain_confirmed":cleanup.is_ok(),
        "fresh_cases_completed":reports.iter().filter(|report|report["history_fill"]==profile).count(),
        "raw_cache_replays":replay_count})
    );
    // Empty request reservations do not include session destruction or native
    // thread exit. Join on failed work too, before another profile or main exit.
    let shutdown = shutdown_native(&owner, Duration::from_secs(SHUTDOWN_SECONDS));
    // Current resident identity is a separate proof after the worker shutdown
    // work. It cannot release quarantined inputs or turn failed work into pass.
    let mapping_audit = if is_cuda {
        runtime
            .verify_cuda_runtime_mappings()
            .map_err(|error| Box::new(error) as Box<dyn Error>)
    } else {
        Ok(())
    };
    let diagnostics = owner.take_diagnostics();
    let diagnostic_report = match &diagnostics {
        Ok(batch) => json!({"entries":batch.entries.iter().map(|receipt| json!({
            "ticket":format!("{:?}",receipt.ticket),"context":format!("{:?}",receipt.context),
            "kind":format!("{:?}",receipt.kind),"contract":format!("{:?}",receipt.failure.contract),
            "backend":receipt.failure.backend.as_ref().map(backend_receipt)})).collect::<Vec<_>>(),
            "boundary_error":batch.boundary_error.as_ref().map(|error| format!("{error:?}")),
            "poison_error":batch.poison_error.as_ref().map(|error| format!("{error:?}"))}),
        Err(error) => {
            json!({"transfer_failed":format!("{error:?}"),"originals":"retained by owner"})
        }
    };
    let diagnostics_clean = diagnostics.as_ref().is_ok_and(|batch| {
        batch.entries.is_empty() && batch.boundary_error.is_none() && batch.poison_error.is_none()
    });
    let cleanup_clean = cleanup.as_ref().is_ok_and(|receipt| {
        receipt["unexpected_deliveries"]
            .as_array()
            .is_some_and(Vec::is_empty)
            && receipt["unexpected_delivery_overflow"] == false
    });
    let cuda_metadata = owner.cuda_metadata().map(|metadata| {
        json!({
        "device_id":metadata.device_id,"arena_bytes":metadata.arena_bytes,
        "runtime_bundle_sha256":hex_digest(metadata.runtime_bundle_digest.0),
        "placement_profile_sha256":hex_digest(metadata.placement_profile_digest.0),
        "executed_cuda_nodes":metadata.executed_cuda_nodes})
    });
    let execution_admission = owner.admission_policy().execution_resources();
    let session_admission = owner.admission_policy().session_resident_admission();
    #[cfg(feature = "experimental-raw-cache")]
    let cache_stats: Result<Value> = if raw_cache {
        // A poisoned stats owner or accounting error must not bypass the
        // original work/cleanup receipts or preservation of an unconfirmed
        // native owner. Collect the error here and return it after those steps.
        (|| {
            let stats = projection.raw_cache_stats()?;
            if work.is_ok() && (stats.hits != replay_count || stats.staged != 0) {
                return Err("raw-cache replay accounting differs".into());
            }
            Ok(
                json!({"hits":stats.hits,"misses":stats.misses,"entries":stats.entries,
                "staged":stats.staged,"retained_bytes":stats.retained_bytes}),
            )
        })()
    } else {
        Ok(Value::Null)
    };
    #[cfg(not(feature = "experimental-raw-cache"))]
    let cache_stats: Result<Value> = Ok(Value::Null);
    profiles.push(json!({"history_fill":profile,"batch_size":1,
        "raw_cache":raw_cache,"raw_cache_replays":replay_count,
        "cache_stats":cache_stats.as_ref().ok(),
        "cache_stats_failure":cache_stats.as_ref().err().map(|error|error_receipt(error.as_ref())),
        "provider":if is_cuda{"cuda"}else{"cpu"},
        "native_origin":if is_cuda{"cuda_onnx"}else{"cpu_onnx"},
        "cuda":cuda_metadata,
        "current_mapping_audit":if !is_cuda{"not_applicable_cpu"}else if mapping_audit.is_ok(){"passed"}else{"failed"},
        "mapping_failure":mapping_audit.as_ref().err().map(|error|error_receipt(error.as_ref())),
        "execution_admission":{"host_bytes":execution_admission.host_bytes,
            "device_bytes":execution_admission.device_bytes,"pinned_bytes":execution_admission.pinned_bytes},
        "session_resident_admission":{"host_bytes":session_admission.host_bytes,
            "device_bytes":session_admission.device_bytes,"pinned_bytes":session_admission.pinned_bytes},
        "admission_is_measured_residency_or_total_native_hard_cap":false,
        "work_status":if work.is_ok(){"passed"}else{"failed"},
        "work_failure":work.as_ref().err().map(|error| error_receipt(error.as_ref())),
        "drain":cleanup.as_ref().ok(),
        "drain_failure":cleanup.as_ref().err().map(|error| error_receipt(error.as_ref())),
        "native_shutdown":shutdown.as_ref().ok(),
        "native_shutdown_failure":shutdown.as_ref().err().map(|error|error_receipt(error.as_ref())),
        "native_diagnostics":diagnostic_report}));
    if diagnostics.is_err() || shutdown.is_err() {
        // Preserve originals or an unconfirmed native owner through process
        // exit; a report or logical drain never permits reclaiming its pins.
        std::mem::forget(owner);
    }
    work?;
    cleanup?;
    shutdown?;
    mapping_audit?;
    cache_stats?;
    if !diagnostics_clean || !cleanup_clean {
        return Err(
            "native diagnostics or unexpected deliveries prevent profile acceptance".into(),
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let option_start = args
        .iter()
        .position(|arg| arg.starts_with("--"))
        .unwrap_or(args.len());
    let (args, options) = args.split_at(option_start);
    let mut runtime_cache_root = None;
    let mut raw_cache = false;
    for option in options {
        if option == "--experimental-raw-cache" {
            if !cfg!(feature = "experimental-raw-cache") || raw_cache {
                return Err("raw-cache gate requires its feature and a unique option".into());
            }
            raw_cache = true;
            continue;
        }
        let value = option
            .strip_prefix("--runtime-cache-root=")
            .ok_or("unsupported rules gate option")?;
        if value.is_empty() || runtime_cache_root.replace(Path::new(value)).is_some() {
            return Err("runtime cache root must be a unique absolute path".into());
        }
    }
    let is_cuda = match args.get(7).map(String::as_str) {
        Some("cpu") if args.len() == 8 => false,
        Some("cuda") if args.len() == 10 => true,
        _ => return Err("usage: rules_maia_check SOURCE.pb.gz MODEL.onnx MANIFEST.json ORT_LIBRARY ORT_SHA256 FIXTURES.json REPORT.json cpu | cuda PROFILE_DIRECTORY CUDA_BUNDLE.json".into()),
    };
    let report_path = Path::new(&args[6]);
    if !report_path.is_absolute() {
        return Err("REPORT must be absolute".into());
    }
    let output_root = report_path.parent().ok_or("REPORT has no parent")?;
    let canonical = output_root
        .canonicalize()
        .map_err(|_| "REPORT parent must already exist")?;
    for ancestor in canonical.ancestors() {
        if ancestor.join(".git").try_exists()? {
            return Err("runtime/report root must be outside Git".into());
        }
    }
    let mut report_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report_path)?;
    let mut cases = Vec::new();
    let mut profiles = Vec::new();
    let mut identity = json!({});
    let result = (|| -> Result<()> {
        let bytes = asset::read_bounded(Path::new(&args[5]), 8 * 1024 * 1024)?;
        let fixtures: Fixtures = serde_json::from_slice(&bytes)?;
        let model = MaiaAsset::load(
            Path::new(&args[0]),
            Path::new(&args[1]),
            Path::new(&args[2]),
        )?;
        validate_reference(&fixtures, &model)?;
        let cuda_profile_directory = if is_cuda {
            let directory = Path::new(&args[8]);
            if !directory.is_absolute()
                || directory
                    .parent()
                    .ok_or("profile directory has no parent")?
                    .canonicalize()?
                    != canonical
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
            builder.create(directory)?;
            Some(directory)
        } else {
            None
        };
        let runtime_cache =
            runtime_cache_root.map_or_else(RuntimeCache::for_user, RuntimeCache::open)?;
        let pin = if is_cuda {
            let bytes = asset::read_bounded(Path::new(&args[9]), 64 * 1024)?;
            let spec = CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&bytes)?)?;
            let core = spec
                .files
                .iter()
                .find(|file| file.role == RuntimeBundleFileRole::Core)
                .ok_or("CUDA bundle has no core declaration")?;
            let core_path = Path::new(&args[3]);
            if core_path.file_name().and_then(|name| name.to_str()) != Some(core.filename.as_str())
                || core.sha256 != args[4]
            {
                return Err(
                    "ORT_LIBRARY/ORT_SHA256 must match the explicit CUDA bundle core".into(),
                );
            }
            runtime_cache.cuda_bundle(
                core_path.parent().ok_or("CUDA core has no bundle parent")?,
                &spec,
            )?
        } else {
            runtime_cache.library(Path::new(&args[3]), &args[4])?
        };
        let runtime = OrtRuntime::load(&pin)?;
        identity = json!({"reference":fixtures.reference,"reference_commit":fixtures.reference_commit,
            "reference_module_sha256":fixtures.reference_module_sha256,"source_sha256":fixtures.source_sha256,
            "fixture_sha256":asset::hex_sha256(&bytes),"onnx_sha256":model.manifest().onnx_sha256,
            "runtime_sha256":args[4],"runtime_build":runtime.build_info(),
            "runtime_storage":pin.storage(),
            "runtime_bundle_sha256":runtime.bundle_digest().map(hex_digest)});
        for (fill, slot) in [(HistoryFill::No, 1), (HistoryFill::RepeatOldest, 2)] {
            evaluate_profile(
                &runtime,
                &model,
                &fixtures,
                ProfileExecution {
                    fill,
                    profile_slot: slot,
                    cuda_profile_directory,
                    raw_cache,
                },
                &mut cases,
                &mut profiles,
            )?;
        }
        if cases.len() != CASE_NAMES.len() {
            return Err("not all twelve actual A cases completed".into());
        }
        if is_cuda {
            runtime.verify_cuda_runtime_mappings()?;
        }
        Ok(())
    })();
    let device_admission = profiles
        .first()
        .and_then(|profile| profile.pointer("/execution_admission/device_bytes"))
        .cloned();
    let report = json!({"status":if result.is_ok(){"passed"}else{"failed"},
        "failure":result.as_ref().err().map(|error| error_receipt(error.as_ref())),"identity":identity,
        "scope":if is_cuda{"actual A immutable Rules projection, C ONNX CUDA and D finalization"}else{"actual A immutable Rules projection, C ONNX CPU and D finalization"},
        "handles":"explicit local fixture ownership; no production registry issuance claim",
        "provider":if is_cuda{"cuda"}else{"cpu"},"precision":"fp32","case_count":cases.len(),"case_results":cases,
        "model_rights":"external selected weight; license status is recorded in the pinned export manifest",
        "profiles":profiles,"limits":{"fixture_bytes":8*1024*1024,"cases":12,"trace_plies":8,
            "native_batch_size":1,"max_executions":1,"request_deadline_seconds":REQUEST_SECONDS,
            "native_shutdown_deadline_seconds":SHUTDOWN_SECONDS,
            "scheduler_host_reservation_limit":4*HOST_BYTES_PER_ITEM+8192,
            "scheduler_device_admission_limit":device_admission,
            "bootstrap_session_resident_device_admission":device_admission,
            "scheduler_reservations_are_resident_memory_measurements":false},
        "tolerances":{"dense_input_atol":0.0,"legal_policy_max_abs":1e-4,"wdl_max_abs":1e-4},
        "raw_logits":"separate maia_check gate","uci":"not_run","gpu":if is_cuda{"see_actual_profile_receipts"}else{"not_run"},
        "maia_training":"not_run","arena":"not_run","strength":"not_run"});
    report_file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    report_file.write_all(b"\n")?;
    result
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use rz_runtime::contracts::RuntimeRequest;
    use rz_runtime::Adapter;

    #[test]
    fn physical_and_replay_ids_pass_actual_monotonic_admission() {
        let clock = ContractSystemClock::new(ProcessEpoch(211));
        let encoding = EncodingHandle {
            owner: OwnerId(212),
            slot: 1,
            generation: SlotGeneration(1),
            manifest: encoding_manifest(HistoryFill::No),
        };
        let model = ModelHandle {
            owner: OwnerId(212),
            slot: 2,
            generation: SlotGeneration(1),
            manifest: Digest([213; 32]),
        };
        let projection = ClassicalProjection::new(
            MaiaBinding::new(model, encoding, HistoryFill::No, Digest([214; 32]), 1).unwrap(),
        );
        let frozen = ContractPosition::new(OwnerId(215), Position::startpos())
            .export()
            .unwrap();
        let scope = AcceptanceScope {
            game: GameGeneration(1),
            root: RootGeneration(1),
            model,
            encoding,
            backend: projection.backend(),
        };
        for raw_cache in [false, true] {
            let mut adapter =
                ContractsAdapter::new(SharedScope::new(scope), clock.clone(), 1).unwrap();
            for index in 0..CASE_NAMES.len() {
                let sequence = case_sequence(index, raw_cache);
                for offset in 0..if raw_cache { 3 } else { 1 } {
                    let request = request(&projection, &frozen, &clock, sequence + offset).unwrap();
                    adapter
                        .validate_admission(&RuntimeRequest::new(request))
                        .unwrap();
                }
            }
        }
    }

    #[test]
    fn drained_request_does_not_admit_delayed_native_shutdown_on_success_or_failure() {
        use rz_eval::error::{BackendError, FailureKind, FailureStage};
        use rz_eval::worker::SingleWorker;
        use std::sync::mpsc;

        struct NativeDestructor(mpsc::Sender<()>, mpsc::Receiver<()>);
        impl Drop for NativeDestructor {
            fn drop(&mut self) {
                self.0.send(()).unwrap();
                self.1.recv_timeout(Duration::from_secs(5)).unwrap();
            }
        }

        for fail in [false, true] {
            let clock = ContractSystemClock::new(ProcessEpoch(321));
            let model = ModelHandle {
                owner: OwnerId(322),
                slot: 1,
                generation: SlotGeneration(1),
                manifest: Digest([123; 32]),
            };
            let encoding = EncodingHandle {
                owner: OwnerId(322),
                slot: 2,
                generation: SlotGeneration(1),
                manifest: encoding_manifest(HistoryFill::No),
            };
            let projection = ClassicalProjection::new(
                MaiaBinding::new(model, encoding, HistoryFill::No, Digest([124; 32]), 1).unwrap(),
            );
            let (began, destruction_started) = mpsc::channel();
            let (release, released) = mpsc::channel();
            let native = NativeDestructor(began, released);
            let worker = SingleWorker::spawn(
                move |batch: &rz_eval::contracts::PreparedBatch<RulesState>| {
                    let _keep = &native;
                    if fail {
                        Err(BackendError::new(
                            FailureKind::BackendFailure,
                            FailureStage::Backend,
                            "injected completed physical failure",
                        )
                        .into())
                    } else {
                        let raw = rz_eval::RawOutput {
                            policy_logits: vec![0.0; 1858],
                            wdl: vec![0.5, 0.25, 0.25],
                        };
                        batch
                            .requests()
                            .iter()
                            .map(|request| request.physical_output(&raw, batch.execution()))
                            .collect()
                    }
                },
            )
            .unwrap();
            let owner = NativeWorkerOwner::from_worker(worker, projection.clone(), 2).unwrap();
            let scope = AcceptanceScope {
                game: GameGeneration(1),
                root: RootGeneration(1),
                model,
                encoding,
                backend: projection.backend(),
            };
            let adapter = ContractsAdapter::new(SharedScope::new(scope), clock.clone(), 1).unwrap();
            let backend = NativeRuntimeBackend::new(owner.clone(), clock.clone()).unwrap();
            let limits = Limits {
                max_requests: 1,
                max_batch_items: 1,
                max_executions: 1,
                max_batch_wait: Duration::ZERO,
                max_queue_age: Duration::from_secs(REQUEST_SECONDS),
                deadline_reserve: Duration::ZERO,
                memory: Resources {
                    host_bytes: 4 * HOST_BYTES_PER_ITEM + 8192,
                    device_bytes: 0,
                    pinned_bytes: 0,
                },
            };
            let mut evaluator = ContractEvaluator::new(adapter, backend, limits, 64).unwrap();
            let frozen = ContractPosition::new(OwnerId(323), Position::startpos())
                .export()
                .unwrap();
            evaluator
                .submit(request(&projection, &frozen, &clock, 1).unwrap())
                .unwrap();
            let work = finalized(&mut evaluator, "native-shutdown-regression");
            assert_eq!(work.is_err(), fail);
            assert_eq!(
                drain(&mut evaluator, &clock).unwrap()["physical"],
                "drained"
            );
            let unconfirmed = shutdown_native(&owner, Duration::ZERO);
            destruction_started
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
            release.send(()).unwrap();
            assert_eq!(
                unconfirmed.unwrap_err().to_string(),
                "native worker shutdown is unconfirmed within fixture budget"
            );
            assert_eq!(
                shutdown_native(&owner, Duration::from_secs(SHUTDOWN_SECONDS)).unwrap()["native"],
                "joined"
            );
            let diagnostics = owner.take_diagnostics().unwrap();
            assert_eq!(diagnostics.entries.len(), usize::from(fail));
            if fail {
                assert_eq!(
                    diagnostics.entries[0]
                        .failure
                        .backend
                        .as_ref()
                        .unwrap()
                        .detail,
                    "injected completed physical failure"
                );
            }
        }
    }
}
