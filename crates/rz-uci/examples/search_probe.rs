//! Opt-in finite diagnostic of the existing B1 Rules/C/D/search path.
use rz_contracts::{PlayStatus, ProcessEpoch};
use rz_search::contracts::ContractPosition as _;
use rz_uci::{
    PositionBase, PositionPort, PositionSpec,
    engine::{EvaluatorFactory, OwnerRegistry, ProcessClock, RulesUciPort, diagnostics, move_text},
    native_bootstrap::{NativeConfig, NativeCpuFactory, NativeProvider},
    native_profile::ProfileWriter,
};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut wall = 30_000;
    let mut cases = "terminal".to_owned();
    let mut arguments = Vec::new();
    for arg in std::env::args().skip(1) {
        if let Some(value) = arg.strip_prefix("--probe-wall-ms=") {
            wall = value.parse::<u64>()?;
        } else if let Some(value) = arg.strip_prefix("--probe-cases=") {
            cases = value.to_owned();
        } else {
            arguments.push(arg);
        }
    }
    if wall == 0 || wall > 60_000 {
        return Err("probe wall must be 1..=60000ms".into());
    }
    let config = NativeConfig::parse(arguments)?;
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    match config.provider() {
        NativeProvider::Cpu => {
            let factory = NativeCpuFactory::load(&owners, &config)?;
            probe(&factory, &owners, &clock, &config, &cases, wall)?;
            let finished = factory.finish(Ok(()));
            if config.profiling_requested() {
                ProfileWriter::open(&config)?
                    .publish(&factory.source_trace().unwrap(), &finished)?;
            }
            eprintln!("{}", finished?);
        }
        NativeProvider::Cuda => {
            #[cfg(feature = "onnx-cuda")]
            {
                let factory = rz_uci::native_bootstrap::NativeCudaFactory::load(&owners, &config)?;
                probe(&factory, &owners, &clock, &config, &cases, wall)?;
                let finished = factory.finish(Ok(()));
                if config.profiling_requested() {
                    ProfileWriter::open(&config)?
                        .publish(&factory.source_trace().unwrap(), &finished)?;
                }
                eprintln!("{}", finished?);
            }
            #[cfg(not(feature = "onnx-cuda"))]
            return Err("CUDA probe requires onnx-cuda".into());
        }
    }
    Ok(())
}

fn probe(
    factory: &dyn EvaluatorFactory,
    owners: &Arc<OwnerRegistry>,
    clock: &ProcessClock,
    config: &NativeConfig,
    cases: &str,
    wall: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let profile_cases = [
        ("startpos", "startpos", ""),
        ("ruy-black", "startpos", "e2e4 e7e5 g1f3 b8c6 f1b5"),
    ];
    let parity_cases = [
        (
            "pirc-eight-ply",
            "startpos",
            "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 g1f3 f8g7",
        ),
        (
            "kings-indian-eight-ply",
            "startpos",
            "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",
        ),
        (
            "italian-eight-ply",
            "startpos",
            "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 d2d3 g8f6",
        ),
        (
            "french-eight-ply",
            "startpos",
            "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 e4e5 f6d7",
        ),
    ];
    let fixtures: Vec<_> = match cases {
        "terminal" => diagnostics::TERMINAL_CASES
            .iter()
            .map(|&(name, fen)| (name, fen, ""))
            .collect(),
        "profile" => profile_cases.to_vec(),
        "parity" => parity_cases.to_vec(),
        _ => return Err("probe cases must be terminal, profile or parity".into()),
    };
    let port = RulesUciPort::new(owners.clone(), Default::default());
    for (index, &(name, fen, moves)) in fixtures.iter().enumerate() {
        factory.reset_game()?;
        let spec = PositionSpec {
            base: if fen == "startpos" {
                PositionBase::StartPos
            } else {
                PositionBase::Fen(fen.into())
            },
            moves: moves.split_whitespace().map(String::from).collect(),
        };
        let root = port.prepare(&spec)?.snapshot;
        let input_dump = if root.snapshot().classification().play_status == PlayStatus::Ongoing {
            let profile = factory.profile();
            let projection = rz_eval::rules_projection::ClassicalProjection::new(
                rz_eval::contracts::MaiaBinding::new(
                    profile.model.handle(),
                    profile.model.encoding().handle,
                    rz_encoding::classical::HistoryFill::No,
                    profile.backend,
                    1,
                )?,
            );
            let state = root.state().rules();
            let legal = root.legal().moves();
            let key = projection.input_key(state, legal)?;
            if key != factory.input_key(state, legal)? {
                return Err("diagnostic input differs from native factory".into());
            }
            Some(
                json!({"history_fill":"no", "tensor":projection.encode(state)?.values(),
                "indices":projection.legal_indices(state, legal)?,
                "input_digest":format!("{key:?}"), "model":format!("{:?}",profile.model.handle()),
                "precision":format!("{:?}",profile.precision)}),
            )
        } else {
            None
        };
        let candidates: Vec<Value> = root.legal().moves().iter().map(|mv| {
            let child = root.play(mv)?;
            Ok(json!({"move":move_text(*mv)?, "child":format!("{:?}", child.snapshot().classification().play_status)}))
        }).collect::<Result<_, rz_contracts::ContractError>>()?;
        let probe = diagnostics::run(
            factory,
            clock,
            root.clone(),
            index as u64 + 1,
            config.engine_settings().search.max_simulations,
            Duration::from_millis(wall),
        )?;
        let final_status = probe
            .outcome
            .best_move
            .map(|mv| root.play(&mv))
            .transpose()?
            .map(|p| p.snapshot().classification().play_status);
        let terminals: Vec<Value> = probe.terminals.iter().map(|trace| {
            let obs = &trace.observation;
            json!({"selection": obs.selection.sequence,
                "path":obs.path.iter().map(|mv| move_text(*mv)).collect::<Result<Vec<_>,_>>().unwrap(),
                "leaf":format!("{:?}", obs.classification), "leaf_side":format!("{:?}", obs.leaf_side),
                "leaf_value":obs.leaf_value, "root_visit_delta":trace.root_visit_delta,
                "root_value_delta":trace.root_value_delta, "expected_root_value_delta":trace.expected_root_value_delta})
        }).collect();
        let stats: Vec<Value> = probe.outcome.root_stats.iter().map(|(mv, s)| json!({
            "move":move_text(*mv).unwrap(), "prior":s.prior, "visits":s.visits, "value_sum":s.value_sum, "q":s.q()
        })).collect();
        println!(
            "{}",
            json!({"schema":1, "name":name, "fen":fen, "moves":spec.moves,
            "simulations":config.engine_settings().search.max_simulations, "wall_ms":wall,
            "policy":format!("{:?}",probe.outcome.policy_identity), "status":format!("{:?}",probe.outcome.status),
            "best_move":probe.outcome.best_move.map(move_text).transpose()?,
            "chosen_child":final_status.map(|s| format!("{s:?}")),
            "chosen_is_mate":matches!(final_status,Some(PlayStatus::Terminal{reason:rz_contracts::TerminalReason::Checkmate,..})),
            "candidates":candidates, "root_stats":stats, "terminal_traces":terminals,
            "input":input_dump,
            "root_evaluation":probe.root_evaluation.as_ref().map(|output| json!({
                "policy":output.policy.probabilities(), "wdl":output.wdl.probabilities(),
                "viewpoint":format!("{:?}",output.viewpoint), "actual":format!("{:?}",output.actual)})),
            "backup_errors":probe.backup_errors, "counters":format!("{:?}",probe.outcome.counters),
            "metrics":format!("{:?}",probe.outcome.metrics), "elapsed_ns":probe.elapsed.as_nanos(),
            "shutdown_ns":probe.shutdown_elapsed.as_nanos()})
        );
    }
    Ok(())
}
