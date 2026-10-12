//! Bounded diagnostic of the public exact192 CPU evaluator, separate from Rules
//! replay, active engine throughput and PALS GPU inference. No fallback weights.
//!
//! Build separately, then supervise the binary for at most 60 seconds with two
//! CPU IDs. The program itself admits at most 45 seconds and 128 iterations.
//! --checkpoint is an explicit external float checkpoint. Without it the report
//! says not_run; --active-value-identity is a declaration, not a runtime witness.
#![forbid(unsafe_code)]

use rz_position::{BoardMove, PlayStatus, Position};
use rz_search::cpu_value::{
    CPU_VALUE_MAX_BYTES, CpuAccumulator, CpuFloatValue, CpuValueEvaluator, CpuValueIdentity,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    error::Error,
    fs::{self, File, OpenOptions},
    hint::black_box,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TRACE: &str = include_str!("../../rz-position/tests/fixtures/performance_trace.txt");
const FIXTURE_PLIES: [usize; 5] = [0, 16, 64, 128, 256];
const MAX_REPORT_BYTES: usize = 1024 * 1024;
const MAX_IDENTITY_BYTES: usize = 16 * 1024;
const MAX_BINARY_BYTES: usize = 128 * 1024 * 1024;

struct Args {
    checkpoint: Option<PathBuf>,
    active_identity: Option<PathBuf>,
    output: PathBuf,
    iterations: usize,
    wall_ms: u64,
}
impl Args {
    fn parse() -> Result<Self> {
        let arguments: Vec<_> = std::env::args().skip(1).collect();
        if arguments.len() > 10 || arguments.len() % 2 != 0 {
            return Err("usage: cpu_value_cost --output NEW_EXTERNAL_JSON [--checkpoint EXTERNAL_FLOAT_JSON] [--active-value-identity EXTERNAL_IDENTITY_JSON] [--iterations 1..128] [--wall-ms 1..45000]".into());
        }
        let mut args = Self {
            checkpoint: None,
            active_identity: None,
            output: PathBuf::new(),
            iterations: 16,
            wall_ms: 45_000,
        };
        let mut names = BTreeSet::new();
        for pair in arguments.chunks_exact(2) {
            if !names.insert(pair[0].as_str()) || pair[1].len() > 4096 {
                return Err("duplicate or unbounded diagnostic argument".into());
            }
            match pair[0].as_str() {
                "--checkpoint" => args.checkpoint = Some(pair[1].clone().into()),
                "--active-value-identity" => args.active_identity = Some(pair[1].clone().into()),
                "--output" => args.output = pair[1].clone().into(),
                "--iterations" => args.iterations = pair[1].parse()?,
                "--wall-ms" => args.wall_ms = pair[1].parse()?,
                _ => return Err("unknown diagnostic argument".into()),
            }
        }
        if !(1..=128).contains(&args.iterations) || !(1..=45_000).contains(&args.wall_ms) {
            return Err("diagnostic iteration/wall bound exceeded".into());
        }
        Ok(args)
    }
}

fn external_path(path: &Path) -> Result<()> {
    if !path.is_absolute() || path.components().count() > 64 {
        return Err("diagnostic artifact path must be bounded and absolute".into());
    }
    for ancestor in path.ancestors() {
        let name = ancestor
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        let lower = name.to_ascii_lowercase();
        if lower == ".env"
            || lower.starts_with(".env.")
            || lower.contains("credential")
            || lower.contains("service-account")
            || lower.contains("service_account")
            || lower.starts_with("id_ed25519")
            || lower.starts_with("id_rsa")
        {
            return Err("secret paths are outside diagnostic scope".into());
        }
        if ancestor.join(".git").exists() {
            return Err("diagnostic artifacts must stay outside Git".into());
        }
        match fs::symlink_metadata(ancestor) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err("linked diagnostic artifact paths are unsupported".into());
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn read_regular(path: &Path, maximum: usize) -> Result<Vec<u8>> {
    external_path(path)?;
    let named = fs::symlink_metadata(path)?;
    if !named.is_file() || named.len() > maximum as u64 {
        return Err("diagnostic input must be a bounded regular file".into());
    }
    let mut bytes = Vec::new();
    File::open(path)?
        .take(maximum as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > maximum {
        return Err("diagnostic input grew beyond its bound".into());
    }
    Ok(bytes)
}

fn hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn source_pins() -> Value {
    let sources: [(&str, &[u8]); 7] = [
        ("cpu_value.rs", include_bytes!("../src/cpu_value.rs")),
        ("cpu_value_cost.rs", include_bytes!("cpu_value_cost.rs")),
        (
            "position.rs",
            include_bytes!("../../rz-position/src/position.rs"),
        ),
        (
            "movegen.rs",
            include_bytes!("../../rz-position/src/movegen.rs"),
        ),
        ("types.rs", include_bytes!("../../rz-position/src/types.rs")),
        (
            "outcome.rs",
            include_bytes!("../../rz-position/src/outcome.rs"),
        ),
        ("fen.rs", include_bytes!("../../rz-position/src/fen.rs")),
    ];
    Value::Object(
        sources
            .into_iter()
            .map(|(name, bytes)| (name.to_owned(), json!(hex(bytes))))
            .collect(),
    )
}

#[cfg(target_os = "linux")]
fn affinity() -> Result<Vec<u32>> {
    let mut status = String::new();
    File::open("/proc/self/status")?
        .take(128 * 1024)
        .read_to_string(&mut status)?;
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:\t"))
        .ok_or("actual Linux CPU affinity is unavailable")?;
    let mut cpus = BTreeSet::new();
    for item in list.split(',') {
        let mut range = item.split('-');
        let first: u32 = range.next().ok_or("invalid CPU range")?.parse()?;
        let last: u32 = range.next().map(str::parse).transpose()?.unwrap_or(first);
        if range.next().is_some() || first > last || last > 4095 || last - first > 1 {
            return Err("exactly two observed CPU IDs are required".into());
        }
        cpus.extend(first..=last);
    }
    if cpus.len() != 2 {
        return Err("run the diagnostic under taskset with exactly two CPU IDs".into());
    }
    Ok(cpus.into_iter().collect())
}
#[cfg(not(target_os = "linux"))]
fn affinity() -> Result<Vec<u32>> {
    Err("this bounded diagnostic requires observed two-CPU Linux affinity".into())
}

fn check_wall(started: Instant, limit: Duration) -> Result<()> {
    if started.elapsed() >= limit {
        return Err("diagnostic wall limit exceeded; incomplete rows are not success".into());
    }
    Ok(())
}

fn fixtures(started: Instant, limit: Duration) -> Result<Vec<(usize, Position, BoardMove)>> {
    let moves: Vec<_> = TRACE.split_whitespace().collect();
    if moves.len() != 256 {
        return Err("registered Rules trace length changed".into());
    }
    let mut position = Position::startpos();
    let mut result = Vec::with_capacity(FIXTURE_PLIES.len());
    for ply in 0..=moves.len() {
        check_wall(started, limit)?;
        if FIXTURE_PLIES.contains(&ply) {
            if position.classify_position()?.play_status != PlayStatus::Ongoing {
                return Err("registered timing trace is no longer ongoing".into());
            }
            let movement = *position
                .legal_moves()
                .first()
                .ok_or("trace has no legal move")?;
            result.push((ply, position.clone(), movement));
        }
        if let Some(movement) = moves.get(ply) {
            position.make_uci(movement)?;
        }
    }
    Ok(result)
}

fn same_bits(a: &CpuAccumulator, b: &CpuAccumulator) -> bool {
    a.values()
        .iter()
        .flatten()
        .zip(b.values().iter().flatten())
        .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn sample<T, E: Error + 'static>(
    samples: &mut Vec<u64>,
    run: impl FnOnce() -> std::result::Result<T, E>,
) -> Result<T> {
    let began = Instant::now();
    let result = run();
    let elapsed: u64 = began.elapsed().as_nanos().try_into()?;
    let value = result?;
    samples.push(elapsed);
    Ok(black_box(value))
}

fn metrics(samples: &[u64]) -> Value {
    if samples.is_empty() {
        return json!({"status":"unknown", "calls":0});
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = |percent: usize| sorted[(sorted.len() * percent).div_ceil(100) - 1];
    json!({"calls":samples.len(), "total_ns":samples.iter().sum::<u64>(),
        "min_ns":sorted[0], "median_ns":rank(50), "p95_ns":rank(95),
        "percentile_method":"nearest_rank", "samples_ns":samples})
}

fn measure_fixture(
    value: &CpuFloatValue,
    ply: usize,
    mut position: Position,
    movement: BoardMove,
    iterations: usize,
    started: Instant,
    limit: Duration,
) -> Result<Value> {
    check_wall(started, limit)?;
    let began = Instant::now();
    let before = position.snapshot();
    let baseline = value.initialize(&position)?;
    let baseline_forward = value.forward(&baseline, position.side_to_move())?.to_bits();
    let mut accumulator = baseline.clone();
    let mut rows: Vec<_> = (0..8).map(|_| Vec::with_capacity(iterations)).collect();
    // One untimed full cycle warms each existing API and independently checks
    // rounded values/raw forward. Private Signed192 limbs are not exposed here.
    let rule_undo = position.make_move(movement)?;
    let kind = rule_undo.delta().kind;
    let piece_removals = rule_undo.delta().removals.len();
    let piece_additions = rule_undo.delta().additions.len();
    let value_undo = value.apply_delta(&mut accumulator, rule_undo.delta())?;
    let fresh = value.initialize(&position)?;
    if !same_bits(&accumulator, &fresh)
        || value
            .forward(&accumulator, position.side_to_move())?
            .to_bits()
            != value.forward(&fresh, position.side_to_move())?.to_bits()
    {
        return Err("warmup update differs from fresh FP32 bits".into());
    }
    position.unmake(rule_undo)?;
    value.restore(&mut accumulator, value_undo, &position)?;
    if !same_bits(&accumulator, &baseline)
        || value
            .forward(&accumulator, position.side_to_move())?
            .to_bits()
            != baseline_forward
        || !before.same_state(&position.snapshot())
    {
        return Err("warmup restore differs from original Rules/evaluator bits".into());
    }
    drop(fresh);
    for _ in 0..iterations {
        check_wall(started, limit)?;
        let fresh_root = sample(&mut rows[2], || value.initialize(black_box(&position)))?;
        if !same_bits(&fresh_root, &baseline) {
            return Err("fresh root initializer differs from original bits".into());
        }
        let raw = sample(&mut rows[5], || {
            value.forward(black_box(&accumulator), position.side_to_move())
        })?;
        if raw.to_bits() != baseline_forward {
            return Err("reused root forward differs from original bits".into());
        }
        let (fresh, raw) = sample(&mut rows[6], || {
            let fresh = value.initialize(black_box(&position))?;
            let raw = value.forward(&fresh, position.side_to_move())?;
            Ok::<_, rz_search::cpu_value::CpuValueError>((fresh, raw))
        })?;
        if !same_bits(&fresh, &baseline) || raw.to_bits() != baseline_forward {
            return Err("fresh initializer plus forward differs from original bits".into());
        }
        drop((fresh_root, fresh)); // Destruction is outside the measured regions.
        check_wall(started, limit)?;
        let rule_undo = sample(&mut rows[0], || position.make_move(black_box(movement)))?;
        let value_undo = sample(&mut rows[3], || {
            value.apply_delta(black_box(&mut accumulator), rule_undo.delta())
        })?;
        let raw = sample(&mut rows[7], || {
            value.forward(black_box(&accumulator), position.side_to_move())
        })?;
        let fresh = value.initialize(&position)?; // Untimed independent validation.
        if !same_bits(&accumulator, &fresh)
            || raw.to_bits() != value.forward(&fresh, position.side_to_move())?.to_bits()
        {
            return Err("measured update differs from fresh FP32 bits".into());
        }
        sample(&mut rows[1], || position.unmake(rule_undo))?;
        sample(&mut rows[4], || {
            value.restore(black_box(&mut accumulator), value_undo, &position)
        })?;
        if !same_bits(&accumulator, &baseline)
            || value
                .forward(&accumulator, position.side_to_move())?
                .to_bits()
                != baseline_forward
            || !before.same_state(&position.snapshot())
        {
            return Err("measured restore differs from original Rules/evaluator bits".into());
        }
    }
    let names = [
        "rules_make",
        "rules_unmake",
        "exact_initialize",
        "exact_apply_delta",
        "exact_restore",
        "forward_reused_root",
        "fresh_initialize_plus_forward",
        "forward_after_update",
    ];
    let measured_ns: u64 = rows.iter().flatten().sum();
    let fixture_wall_ns: u64 = began.elapsed().as_nanos().try_into()?;
    Ok(
        json!({"trace_ply":ply, "known_history_len":position.known_history_len(), "movement":movement.to_string(),
        "move_kind":{"capture":kind.capture,"castling":kind.castling,"en_passant":kind.en_passant,"promotion":kind.promotion},
        "piece_removals":piece_removals,"piece_additions":piece_additions,
        "rows":Value::Object(names.into_iter().zip(&rows).map(|(name, samples)| (name.to_owned(), metrics(samples))).collect()),
        "fixture_wall_ns":fixture_wall_ns,"measured_nonoverlapping_calls_ns":measured_ns,
        "unmeasured_setup_validation_drop_and_timer_bookkeeping_ns":fixture_wall_ns.saturating_sub(measured_ns),
        "bit_equivalence":{"warmup_cycles":1,"measured_update_restore_cycles":iterations,
            "rounded_accumulator_bits":"passed","raw_forward_bits":"passed","rules_restore":"passed",
            "private_signed192_limbs":"not_exposed_by_public_api"},
        "untimed_validation_initialize_calls":iterations+2,"untimed_validation_forward_calls":2*iterations+4}),
    )
}

fn execute(args: &Args, started: Instant, report: &mut Value) -> Result<()> {
    let limit = Duration::from_millis(args.wall_ms);
    let Some(checkpoint_path) = &args.checkpoint else {
        report["status"] = json!("not_run");
        report["reason"] =
            json!("no_explicit_registered_float_checkpoint; no_bootstrap_or_zero_fallback");
        return Ok(());
    };
    report["cpu_affinity"] = json!(affinity()?);
    check_wall(started, limit)?;
    let bytes = read_regular(checkpoint_path, CPU_VALUE_MAX_BYTES)?;
    let value = CpuFloatValue::from_reader(bytes.as_slice())?;
    report["checkpoint_file_sha256"] = json!(hex(&bytes));
    report["checkpoint_value_identity"] = serde_json::to_value(value.identity())?;
    report["training_metadata_is_declaration"] = json!(true);
    drop(bytes);
    let active = args
        .active_identity
        .as_ref()
        .map(|path| -> Result<CpuValueIdentity> {
            let identity: CpuValueIdentity =
                serde_json::from_slice(&read_regular(path, MAX_IDENTITY_BYTES)?)?;
            identity.validate()?;
            Ok(identity)
        })
        .transpose()?;
    report["scope"] = json!(match &active {
        None => "standalone_diagnostic;active_engine_value_identity_unknown",
        Some(identity) if identity == value.identity() =>
            "diagnostic_matches_declared_value_identity;runtime_active_profile_unattested",
        Some(_) => "standalone_diagnostic;different_declared_active_engine_value_profile",
    });
    report["declared_active_value_identity"] = serde_json::to_value(&active)?;
    report["matches_declared_active_value_identity"] =
        json!(active.as_ref().map(|identity| identity == value.identity()));
    let executable = std::env::current_exe()?;
    report["binary_sha256"] = json!(hex(&read_regular(&executable, MAX_BINARY_BYTES)?));
    let mut overhead = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let timer = Instant::now();
        overhead.push(timer.elapsed().as_nanos().try_into()?);
    }
    report["timer_pair_overhead"] = metrics(&overhead);
    report["rows"] = json!([]);
    for (ply, position, movement) in fixtures(started, limit)? {
        check_wall(started, limit)?;
        let row = measure_fixture(
            &value,
            ply,
            position,
            movement,
            args.iterations,
            started,
            limit,
        )?;
        report["rows"]
            .as_array_mut()
            .ok_or("report rows lost their type")?
            .push(row);
    }
    check_wall(started, limit)?;
    report["status"] = json!("measured");
    Ok(())
}

fn run() -> Result<bool> {
    let started = Instant::now();
    let args = Args::parse()?;
    external_path(&args.output)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let mut report = json!({"schema":"rz-cpu-exact192-api-cost-v1", "status":"pending",
        "source_pins":source_pins(), "trace_sha256":hex(TRACE.as_bytes()),
        "iterations_per_fixture":args.iterations,"wall_limit_ms":args.wall_ms,"required_supervisor_limit_seconds":60,
        "worker_threads":1,"cpu_limit":2,"gpu_execution":false,"python_or_ort_execution":false,
        "active_engine_bottleneck_claim":false,"scope":"standalone_diagnostic",
        "cost_scope":"public_api_calls;identity_validation_feature_changes_exact192_rounding_and_initialize_snapshot_included;rules_make_unmake_separate;allocation_in_initialize_included;destruction_excluded",
        "signed192_limb_only_time":"unknown;private_implementation_not_instrumented",
        "timings":"raw_serial_wall_samples;timer_overhead_recorded_not_subtracted;not_end_to_end_search_or_strength",
        "rows":[]});
    let succeeded = match execute(&args, started, &mut report) {
        Ok(()) => true,
        Err(error) => {
            report["status"] = json!("failed");
            report["failure"] = json!(error.to_string());
            false
        }
    };
    report["program_wall_ns_before_report_write"] =
        json!(u64::try_from(started.elapsed().as_nanos())?);
    let bytes = serde_json::to_vec_pretty(&report)?;
    if bytes.len() >= MAX_REPORT_BYTES {
        return Err("diagnostic report exceeded 1 MiB".into());
    }
    output.write_all(&bytes)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(succeeded)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("cpu_value_cost: {error}");
            ExitCode::FAILURE
        }
    }
}
