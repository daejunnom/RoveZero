//! CPU/mock on/off runner check; no weights, CUDA, strength or fairness acceptance.
use rz_experiments::{
    ArtifactRef, EngineComputeKind, EngineResourceRequest, MatchExecutionV1, NativeEngineRole,
    ResourceSharing,
};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    ffi::OsString,
    fs::{self, File},
    io::Read,
    path::Path,
};
fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("fixture file exceeds its byte ceiling".into());
    }
    Ok(bytes)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(
            "usage: ponder_runner_check FASTCHESS RZ_UCI MODEL_PAIR NEW_OUTPUT_DIRECTORY".into(),
        );
    }
    for arg in &args {
        if !Path::new(arg).is_absolute() {
            return Err("absolute paths required".into());
        }
    }
    let root = Path::new(&args[3]);
    for parent in root.ancestors() {
        if parent.join(".git").exists() {
            return Err("output stays outside Git".into());
        }
    }
    fs::create_dir(root)?;
    let hardware = rz_arena::probe_match_hardware()?.hardware;
    let request = |role| EngineResourceRequest {
        role,
        kind: EngineComputeKind::Cpu,
        cpu_ids: None,
        threads: None,
        gpu_ids: None,
        gpu_memory_bytes: None,
        cpu_weight: None,
        gpu_weight: None,
    };
    let bytes = read_bounded(Path::new(&args[2]), 64 * 1024 * 1024)?;
    let executor = ArtifactRef {
        path: "fixture/model-pair".into(),
        sha256: format!("{:x}", Sha256::digest(&bytes)),
        bytes: bytes.len() as u64,
        source: "https://github.com/daejunnom/RoveZero".into(),
        license: "MIT".into(),
    };
    let execution = MatchExecutionV1 {
        ponder: true,
        sharing: ResourceSharing::Isolated,
        hardware,
        engines: [
            request(NativeEngineRole::Baseline),
            request(NativeEngineRole::Candidate),
        ],
        executor,
        resolved_resources: None,
    };
    let plan = execution.plan()?;
    let opening = rz_experiments::OpeningSpec {
        id: "ponder-mock-startpos".into(),
        initial: rz_experiments::InitialPosition::Startpos,
        fen: None,
        moves: vec![],
        history: rz_experiments::HistoryCompleteness::Complete,
        history_origin: "startpos-complete-trace".into(),
    };
    let pair = rz_arena::PairSpec {
        id: "ponder-mock-pair".into(),
        ordinal: 0,
        opening: opening.clone(),
        opening_input_sha256: rz_arena::canonical_sha256(&opening)?,
        games: std::array::from_fn(|i| rz_arena::GameSpec {
            id: format!("ponder-mock-{i}"),
            white_engine: if i == 0 { "a" } else { "b" }.into(),
            black_engine: if i == 0 { "b" } else { "a" }.into(),
            engine_seeds: Default::default(),
            engine_slots: Default::default(),
        }),
        execution_order: [0, 1],
    };
    let mut binaries = Vec::new();
    for path in &args[..3] {
        binaries.push(serde_json::json!({"sha256":format!("{:x}",Sha256::digest(read_bounded(Path::new(path),64 * 1024 * 1024)?))}));
    }
    let mut reports = Vec::new();
    for ponder in [false, true] {
        let mut command: Vec<OsString> = Vec::new();
        for (i, allocation) in plan.engines.iter().enumerate() {
            let inner = vec![
                OsString::from("engine-exec"),
                allocation
                    .cpu_ids
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
                    .into(),
                "-".into(),
                args[1].clone().into(),
                "--cpu-mock".into(),
                "--mock-delay-ms=0".into(),
            ];
            command.extend([
                "-engine".into(),
                format!("cmd={}", args[2]).into(),
                format!("name={}", if i == 0 { "a" } else { "b" }).into(),
                rz_arena::encode_fastchess_native_args(&inner)?,
                format!("option.Ponder={ponder}").into(),
            ]);
        }
        let pgn = root.join(if ponder { "on.pgn" } else { "off.pgn" });
        command.extend(
            [
                "-each",
                "proto=uci",
                "tc=5+0.1",
                "restart=on",
                "-games",
                "2",
                "-rounds",
                "1",
                "-repeat",
                "-concurrency",
                "1",
                "-maxmoves",
                "6",
                "-pgnout",
            ]
            .into_iter()
            .map(OsString::from),
        );
        command.push(format!("file={}", pgn.display()).into());
        command.extend(
            [
                "notation=uci",
                "-log",
                "file=/proc/self/fd/1",
                "level=trace",
                "realtime=true",
                "engine=true",
            ]
            .into_iter()
            .map(OsString::from),
        );
        let mut runner_command = vec![
            "engine-exec".into(),
            plan.runner_cpu_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
                .into(),
            "-".into(),
            args[0].clone().into(),
        ];
        runner_command.extend(command);
        let output = rz_arena::supervise(
            &File::open(&args[2])?,
            &runner_command,
            root,
            rz_arena::ProcessLimits {
                wall_ms: 20_000,
                shutdown_grace_ms: 2_000,
                max_output_bytes: 2 * 1024 * 1024,
                max_child_processes: 5,
            },
            None,
        )?;
        fs::write(
            root.join(if ponder { "on.log" } else { "off.log" }),
            &output.stdout,
        )?;
        fs::write(
            root.join(if ponder { "on.stderr" } else { "off.stderr" }),
            &output.stderr,
        )?;
        if output.receipt.exit_code != Some(0)
            || output.receipt.group_cleanup != rz_arena::CleanupStatus::Gone
            || output.pending_child.is_some()
            || !output.receipt.errors.is_empty()
        {
            return Err(Box::new(rz_arena::ArenaError::HardwareProbe {
                cause: "CPU/mock runner lifecycle failed".into(),
                process: Box::new(output),
            }));
        }
        let audit = if ponder {
            Some(rz_arena::audit_ponder_protocol(&output.stdout, ["a", "b"])?)
        } else {
            if std::str::from_utf8(&output.stdout)?.contains("go ponder") {
                return Err("ponder-off runner computed speculative moves".into());
            }
            None
        };
        if audit
            .as_ref()
            .is_some_and(|a| a.started == 0 || a.hits == 0 || a.stopped == 0)
        {
            return Err("no actual start/hit/stop coverage in CPU/mock pair".into());
        }
        let pgn_bytes = read_bounded(&pgn, 4 * 1024 * 1024)?;
        let pgn_audit = rz_arena::audit_pair_pgn_for_spec(
            &pair,
            std::str::from_utf8(&pgn_bytes)?,
            rz_arena::PgnLimits {
                max_bytes: 4 * 1024 * 1024,
                max_plies: 12,
            },
            rz_arena::PgnOutcomePolicy {
                engine_failure: rz_experiments::OutcomePolicy::Loss,
                max_plies_outcome: rz_experiments::OutcomePolicy::Incomplete,
                claim_policy: rz_experiments::ClaimPolicy::AutomaticAcceptance,
                max_game_plies: 12,
            },
        )?;
        let clock = rz_arena::validate_pilot_clock_trace(
            &output.stdout,
            &pgn_audit,
            &opening,
            rz_experiments::NativeGameClockV3 {
                base_ms: 5000,
                increment_ms: 100,
            },
        )?;
        reports.push(serde_json::json!({"ponder":ponder,"protocol":audit,"process":output.receipt,"pgn_sha256":format!("{:x}",Sha256::digest(&pgn_bytes)),"pgn":pgn_audit,"clock":clock}));
    }
    let report = serde_json::json!({"scope":"CPU/mock only; no NN/GPU/strength acceptance","binary_order":["fastchess","rz-uci","model-pair"],"binaries":binaries,"placement":plan,"runs":reports});
    fs::write(
        root.join("summary.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
