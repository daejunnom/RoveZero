//! Research-only CPU/mock facade. Both callers execute precisely this body.
//! The shared diagnostic pump includes its existing 1ms wait and observations;
//! these measurements do not estimate the native UCI worker's throughput.
#![forbid(unsafe_code)]

use rz_contracts::ProcessEpoch;
use rz_search::{contracts::ContractSearchStatus, tree::FinalMovePolicy};
use rz_uci::{
    Command, ParserLimits, PositionPort,
    bootstrap::CpuMockFactory,
    engine::{OwnerRegistry, ProcessClock, RulesUciPort, diagnostics, move_text},
    parse,
};
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchResult {
    pub best_move: Option<String>,
    pub terminal: String,
    pub completed_simulations: u64,
    pub consumed_evaluations: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchMeasurement {
    pub result: SearchResult,
    pub initialization_ns: u64,
    pub search_body_ns: u64,
    pub shutdown_ns: u64,
    pub rust_call_ns: u64,
}

fn nanos(value: Duration) -> Result<u64, String> {
    value
        .as_nanos()
        .try_into()
        .map_err(|_| "duration overflow".into())
}

/// Fresh game/root/evaluator on every call; no process-wide search state.
pub fn search_once(
    position_command: &str,
    simulation_limit: u64,
) -> Result<SearchMeasurement, String> {
    let started = Instant::now();
    if !(1..=128).contains(&simulation_limit) {
        return Err("simulation_limit must be 1..=128 (bounded CPU mock)".into());
    }
    let spec = match parse(position_command, ParserLimits::default()).map_err(|e| e.to_string())? {
        Command::Position(spec) => spec,
        _ => return Err("only a complete UCI position command is accepted".into()),
    };
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let port = RulesUciPort::new(owners.clone(), Default::default());
    let root = port.prepare(&spec).map_err(|e| e.to_string())?.snapshot;
    let factory = CpuMockFactory::new(&owners, Duration::ZERO).map_err(|e| e.to_string())?;
    let initialized = started.elapsed();
    let probe = diagnostics::run(
        &factory,
        &clock,
        root,
        1,
        simulation_limit,
        Duration::from_secs(5),
        FinalMovePolicy::Visits,
    )
    .map_err(|e| e.to_string())?;
    if probe.backup_errors != 0 {
        return Err("terminal backup contract violation".into());
    }
    let terminal = match probe.outcome.status {
        ContractSearchStatus::Completed => "ongoing".into(),
        ContractSearchStatus::Terminal { reason, winner, .. } => format!("{reason:?}:{winner:?}"),
        other => return Err(format!("search did not complete: {other:?}")),
    };
    Ok(SearchMeasurement {
        result: SearchResult {
            best_move: probe
                .outcome
                .best_move
                .map(move_text)
                .transpose()
                .map_err(|e| e.to_string())?,
            terminal,
            completed_simulations: probe.outcome.counters.completed_visits,
            consumed_evaluations: probe.outcome.metrics.accepted_outputs,
        },
        initialization_ns: nanos(initialized)?,
        search_body_ns: nanos(probe.elapsed)?,
        shutdown_ns: nanos(probe.shutdown_elapsed)?,
        rust_call_ns: nanos(started.elapsed())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_wrong_command_invalid_position_and_budget() {
        for line in [
            "go nodes 1",
            "position fen broken",
            "position startpos moves e2e5",
        ] {
            assert!(search_once(line, 1).is_err());
        }
        assert!(search_once("position startpos", 0).is_err());
        assert!(search_once("position startpos", 129).is_err());
    }
    #[test]
    fn repeated_and_concurrent_calls_keep_game_state_isolated() {
        let inputs = [
            "position startpos",
            "position fen 7k/5KQ1/8/8/8/8/8/8 b - - 0 1",
            "position fen 8/8/8/8/8/8/5kq1/7K w - - 0 1",
            "position fen 7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
        ];
        for line in inputs {
            let expected = search_once(line, 8).unwrap().result;
            let children: Vec<_> = (0..2)
                .map(|_| std::thread::spawn(move || search_once(line, 8).unwrap().result))
                .collect();
            for child in children {
                assert_eq!(child.join().unwrap(), expected);
            }
            assert_eq!(search_once(line, 8).unwrap().result, expected);
        }
    }
}
