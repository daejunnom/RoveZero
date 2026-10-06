//! Finite correctness witness for old/new CPU mock composition. No timing claim.
//! Keep this source compatible with the registered pre-adapter library APIs so
//! the identical observer can be compiled against each independently pinned tree.
use rz_contracts::{PlayStatus, ProcessEpoch};
use rz_search::{
    contracts::{ContractPosition as _, ContractSearchStatus},
    tree::FinalMovePolicy,
};
use rz_uci::{
    Command, ParserLimits, PositionPort,
    bootstrap::CpuMockFactory,
    engine::{EvaluatorFactory, OwnerRegistry, ProcessClock, RulesUciPort, diagnostics, move_text},
    parse,
};
use serde_json::json;
use std::{sync::Arc, time::Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().len() != 1 {
        return Err("mock_adapter_check accepts no arguments or variable work budget".into());
    }
    let mut inputs = vec![
        ("start".to_owned(), "position startpos".to_owned()),
        (
            "ruy-sixteen".to_owned(),
            "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8".to_owned(),
        ),
    ];
    inputs.extend(
        diagnostics::TERMINAL_CASES
            .iter()
            .map(|&(name, fen)| (name.to_owned(), format!("position fen {fen}"))),
    );
    for repeat in 0..2 {
        for (name, command) in &inputs {
            // A fresh factory, rules owner, clock and root for every witness.
            let owners = Arc::new(OwnerRegistry::default());
            let factory = CpuMockFactory::new(&owners, Duration::ZERO)?;
            let clock = ProcessClock::new(ProcessEpoch(1));
            let spec = match parse(command, ParserLimits::default())? {
                Command::Position(spec) => spec,
                _ => return Err("fixture is not a position command".into()),
            };
            let root = RulesUciPort::new(owners, Default::default())
                .prepare(&spec)?
                .snapshot;
            let input_key =
                factory.input_key(root.state().rules(), root.state().legal_moves().moves())?;
            let probe = diagnostics::run(
                &factory,
                &clock,
                root.clone(),
                1,
                128,
                Duration::from_secs(5),
                FinalMovePolicy::Visits,
            )?;
            let completed = matches!(
                probe.outcome.status,
                ContractSearchStatus::Completed | ContractSearchStatus::Terminal { .. }
            );
            if !completed || probe.backup_errors != 0 {
                return Err(format!(
                    "{name}: incomplete or invalid backup: {:?}",
                    probe.outcome.status
                )
                .into());
            }
            if root.state().snapshot().classification().play_status == PlayStatus::Ongoing
                && probe.outcome.counters.completed_visits != 128
            {
                return Err(format!("{name}: fixed work was not completed").into());
            }
            let chosen_child = probe
                .outcome
                .best_move
                .map(|mv| root.play(&mv))
                .transpose()?;
            let stats = probe.outcome.root_stats.iter().map(|(mv, edge)| Ok(json!({
                "move":move_text(*mv)?, "prior_f64_bits":format!("{:016x}", edge.prior.to_bits()),
                "visits":edge.visits, "value_sum_f64_bits":format!("{:016x}", edge.value_sum.to_bits()),
                "q_f64_bits":format!("{:016x}", edge.q().to_bits())
            }))).collect::<Result<Vec<_>, rz_contracts::ContractError>>()?;
            println!(
                "{}",
                json!({
                    "schema":"rz-mock-adapter-equivalence/1", "name":name, "repeat":repeat,
                    "position":command, "simulation_limit":128, "final_selection":"visits",
                    "input_key":format!("{input_key:?}"), "status":format!("{:?}", probe.outcome.status),
                    "best_move":probe.outcome.best_move.map(move_text).transpose()?,
                    "chosen_child":chosen_child.map(|child|format!("{:?}", child.snapshot().classification().play_status)),
                    "root_stats":stats, "counters":format!("{:?}", probe.outcome.counters),
                    "metrics":format!("{:?}", probe.outcome.metrics),
                    "policy":format!("{:?}", probe.outcome.policy_identity),
                    "completed_simulations":probe.outcome.counters.completed_visits,
                    "consumed_evaluations":probe.outcome.metrics.accepted_outputs,
                    "backup_errors":probe.backup_errors,
                })
            );
        }
    }
    Ok(())
}
