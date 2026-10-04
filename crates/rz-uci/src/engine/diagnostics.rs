//! Finite, passive regression probe of the production Rules/search/factory path.
//! No solver, mate-distance preference or alternative chess implementation lives here.

use super::*;
use rz_search::contracts::{
    ContractPumpEvent, ContractSearch, ContractSearchConfig, ContractSearchOutcome,
    TerminalBackupObservation,
};

#[derive(Debug)]
pub struct TerminalTrace {
    pub observation: TerminalBackupObservation,
    pub root_visit_delta: u64,
    pub root_value_delta: f64,
    pub expected_root_value_delta: f64,
}

#[derive(Debug)]
pub struct SearchProbe {
    pub outcome: ContractSearchOutcome,
    pub root_evaluation: Option<contract::EvalOutput>,
    pub terminals: Vec<TerminalTrace>,
    pub elapsed: Duration,
    pub shutdown_elapsed: Duration,
    pub backup_errors: usize,
}

struct ObservedPort<'a> {
    runtime: &'a mut dyn ManagedEvaluator,
    root_state: contract::StateIdentity,
    candidate: &'a mut Option<contract::EvalOutput>,
}
impl contract::Evaluator<RulesState> for ObservedPort<'_> {
    fn submit(
        &mut self,
        request: Arc<contract::EvalRequest<RulesState>>,
    ) -> Result<(), contract::ContractError> {
        self.runtime.submit(request)
    }
    fn poll(&mut self) -> Option<contract::EvalResult> {
        let result = self.runtime.poll()?;
        if let contract::EvalResult::Completed(output) = &result
            && output.context.state == self.root_state
        {
            *self.candidate = Some(output.clone());
        }
        Some(result)
    }
    fn cancel(&mut self, request: contract::RequestId) -> Result<(), contract::ContractError> {
        self.runtime.cancel(request)
    }
}

/// Authored fixed positions; multiple mates are graded by Rules child outcome.
pub const TERMINAL_CASES: &[(&str, &str)] = &[
    ("white-mate-one", "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1"),
    ("black-mate-one", "8/8/8/8/8/6q1/5k2/7K b - - 0 1"),
    ("black-stalemate-root", "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"),
    ("white-stalemate-root", "8/8/8/8/8/6q1/5k2/7K w - - 0 1"),
    ("black-checkmate-root", "7k/5KQ1/8/8/8/8/8/8 b - - 0 1"),
    ("white-checkmate-root", "8/8/8/8/8/8/5kq1/7K w - - 0 1"),
    ("white-avoid-mate", "5qK1/8/5k2/8/8/8/8/8 w - - 0 1"),
    ("black-avoid-mate", "8/8/8/8/8/5K2/8/5Qk1 b - - 0 1"),
];

/// Explicit bounds prevent this diagnostic API from becoming an unbounded runner.
/// Uses the same B1 pump/progress query/1ms wait/drain as the native UCI worker.
pub fn run(
    factory: &dyn EvaluatorFactory,
    clock: &ProcessClock,
    root: RulesSearchPosition,
    root_generation: u64,
    simulations: u64,
    wall: Duration,
    final_move_policy: rz_search::tree::FinalMovePolicy,
) -> Result<SearchProbe, contract::ContractError> {
    #[cfg(feature = "experimental-batch")]
    let parallelism = factory.parallelism();
    #[cfg(not(feature = "experimental-batch"))]
    let parallelism = 1;
    if !(1..=4096).contains(&simulations)
        || wall.is_zero()
        || wall > Duration::from_secs(60)
        || parallelism != 1
    {
        return Err(failure(
            contract::ErrorCode::InvalidInput,
            "probe requires bounded B1",
        ));
    }
    let profile = factory.profile();
    let scope = contract::AcceptanceScope {
        game: contract::GameGeneration(1),
        root: contract::RootGeneration(root_generation),
        model: profile.model.handle(),
        encoding: profile.model.encoding().handle,
        backend: profile.backend,
    };
    let authority = SearchAuthority {
        #[cfg(feature = "experimental-notify")]
        signal: None,
        current: Arc::new(Mutex::new(scope)),
        cancel: contract::CancelToken::new(),
    };
    let started = Instant::now();
    let deadline = clock.deadline(started + wall)?;
    let config = ContractSearchConfig {
        scope,
        model: profile.model.clone(),
        precision: profile.precision,
        compute: profile.compute,
        bytes: profile.bytes,
        policy_tolerance: 1e-5,
        wdl_tolerance: 1e-5,
        cancellation: authority.cancel_token(),
        deadlines: rz_search::contract_time::ContractDeadlines {
            soft: deadline,
            admission: deadline,
            hard: deadline,
            output: deadline,
        },
        ids: clock.ids.clone(),
        max_simulations: simulations,
        tree_limits: EngineSettings::default().tree,
    };
    // Construct/validate Rules before opening the runtime. Every later exit drains it.
    let root_state = root.state.snapshot().identity();
    let mut search = ContractSearch::new(root, config)?;
    search.set_final_move_policy(final_move_policy)?;
    search.observe_terminal_backups(true);
    search.set_source_trace(factory.source_trace());
    let mut runtime = factory.create(clock.clone(), authority.clone())?;
    let result = (|| {
        let mut previous = search.outcome();
        let mut root_candidate = None;
        let mut root_evaluation = None;
        let mut terminals = Vec::new();
        terminals.try_reserve(simulations as usize).map_err(|_| {
            failure(
                contract::ErrorCode::ResourceExhausted,
                "probe trace allocation failed",
            )
        })?;
        let mut backup_errors = 0;
        while !search.is_finished() {
            let event = search.pump(
                &mut ObservedPort {
                    runtime: runtime.as_mut(),
                    root_state,
                    candidate: &mut root_candidate,
                },
                clock,
                || scope,
                |position, encoding| {
                    if encoding != profile.model.encoding() {
                        return Err(failure(
                            contract::ErrorCode::IdentityMismatch,
                            "probe encoding changed",
                        ));
                    }
                    factory.input_key(position.state.rules(), position.state.legal_moves().moves())
                },
            );
            match &event {
                ContractPumpEvent::Accepted {
                    evaluation: Some(value),
                    traversed_edges,
                    ..
                } => {
                    factory.observe_search_acceptance(value, *traversed_edges)?;
                    if *traversed_edges == 0
                        && root_candidate
                            .as_ref()
                            .is_some_and(|output: &contract::EvalOutput| {
                                output.context == value.context.request
                            })
                    {
                        root_evaluation = root_candidate.take();
                    }
                }
                ContractPumpEvent::Diagnostic(error)
                | ContractPumpEvent::RejectedResult { error, .. } => return Err(*error),
                _ => {}
            }
            let next = search.outcome();
            if let Some(observation) = search.take_terminal_observation() {
                let first = observation.path.first().ok_or_else(|| {
                    failure(
                        contract::ErrorCode::InvalidInput,
                        "terminal traversal has no edge",
                    )
                })?;
                let before = previous
                    .root_stats
                    .iter()
                    .find(|(mv, _)| mv == first)
                    .map(|(_, s)| *s);
                let after = next
                    .root_stats
                    .iter()
                    .find(|(mv, _)| mv == first)
                    .map(|(_, s)| *s)
                    .ok_or_else(|| {
                        failure(
                            contract::ErrorCode::IdentityMismatch,
                            "terminal root edge missing",
                        )
                    })?;
                let root_visit_delta = after.visits.saturating_sub(before.map_or(0, |s| s.visits));
                let root_value_delta = after.value_sum - before.map_or(0.0, |s| s.value_sum);
                let expected_root_value_delta = observation.leaf_value
                    * if observation.path.len() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                if root_visit_delta != 1
                    || (root_value_delta - expected_root_value_delta).abs() > 1e-9
                {
                    backup_errors += 1;
                }
                terminals.push(TerminalTrace {
                    observation,
                    root_visit_delta,
                    root_value_delta,
                    expected_root_value_delta,
                });
            }
            previous = next;
            if matches!(event, ContractPumpEvent::Waiting) {
                thread::sleep(Duration::from_millis(1));
            }
        }
        Ok(SearchProbe {
            outcome: search.outcome(),
            root_evaluation,
            terminals,
            elapsed: started.elapsed(),
            shutdown_elapsed: Duration::ZERO,
            backup_errors,
        })
    })();
    // Close pending logical work and fence physical buffers on success and errors.
    authority.cancel();
    let shutdown = Instant::now();
    runtime.shutdown(shutdown + Duration::from_secs(2))?;
    result.map(|mut probe| {
        probe.shutdown_elapsed = shutdown.elapsed();
        probe
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::CpuMockFactory;
    use rz_search::contracts::{ContractPosition as _, ContractSearchStatus};

    #[test]
    fn rules_terminal_candidates_backup_and_final_choice() {
        for final_policy in [
            rz_search::tree::FinalMovePolicy::Visits,
            rz_search::tree::FinalMovePolicy::ExactTerminal,
        ] {
            for &(name, fen) in TERMINAL_CASES {
                let owners = Arc::new(OwnerRegistry::default());
                let factory = CpuMockFactory::new(&owners, Duration::ZERO).unwrap();
                let clock = ProcessClock::new(contract::ProcessEpoch(1));
                let port = RulesUciPort::new(owners, PositionLimits::default());
                let root = port
                    .prepare(&PositionSpec {
                        base: PositionBase::Fen(fen.into()),
                        moves: vec![],
                    })
                    .unwrap()
                    .snapshot;
                let child_status = |mv: contract::Move| {
                    rz_search::contracts::ContractPosition::play(&root, &mv)
                        .unwrap()
                        .snapshot()
                        .classification()
                        .play_status
                };
                if name.ends_with("mate-one") {
                    assert!(
                        root.legal().moves().iter().any(|&mv| matches!(
                            child_status(mv),
                            contract::PlayStatus::Terminal {
                                reason: contract::TerminalReason::Checkmate,
                                ..
                            }
                        )),
                        "{name}"
                    );
                }
                let probe = run(
                    &factory,
                    &clock,
                    root.clone(),
                    1,
                    128,
                    Duration::from_secs(5),
                    final_policy,
                )
                .unwrap();
                assert_eq!(probe.backup_errors, 0, "{name}");
                assert_eq!(
                    probe.outcome.counters.completed_visits,
                    probe.outcome.counters.accepted_backups,
                    "{name}"
                );
                if name.ends_with("-root") {
                    assert!(
                        matches!(probe.outcome.status, ContractSearchStatus::Terminal { .. }),
                        "{name}"
                    );
                    assert!(probe.outcome.best_move.is_none(), "{name}");
                    assert_eq!(probe.outcome.metrics.submissions, 0, "{name}");
                } else {
                    let chosen = child_status(probe.outcome.best_move.expect(name));
                    let reason = if name.ends_with("mate-one") {
                        contract::TerminalReason::Checkmate
                    } else {
                        contract::TerminalReason::DeadPosition
                    };
                    assert!(
                        matches!(chosen, contract::PlayStatus::Terminal { reason: found, .. } if found == reason),
                        "{name}: {chosen:?}"
                    );
                    assert!(!probe.terminals.is_empty(), "{name}");
                }
            }
        }
    }
}
