//! Bounded aggregate evidence: no per-go journal grows with game length.
//! None means an observation was unavailable; an observed no-work path is zero.
use super::{SearchKind, SearchSessionContext, SearchSessionFailure, SearchSessionReport};
use rz_contracts::{ContractError, ErrorCode, Stage, pals::SearchAuthority};
use rz_position::BoardMove;
use rz_search::{
    cpu::{CpuReport, CpuScoreScope, CpuWork},
    pals::engine::PalsCounters,
};
use std::sync::Mutex;

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "search-work-receipts", derive(serde::Serialize))]
pub struct ProcessSearchWorkReceipt {
    pub schema_version: u32,
    pub search_kind: SearchKind,
    pub go_invocations: u64,
    pub successful_returns: u64,
    pub failed_returns: u64,
    pub canceled_returns: u64,
    pub deadline_returns: u64,
    pub physical_unknown_returns: u64,
    pub active_invocations: u64,
    pub unobserved_work_invocations: u64,
    pub cpu: Option<CpuWorkTotals>,
    pub pals: Option<PalsWorkTotals>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "search-work-receipts", derive(serde::Serialize))]
pub struct CpuWorkTotals {
    pub tasks_requested: Option<u64>,
    pub reports_returned: Option<u64>,
    pub requested_depth_completed: Option<u64>,
    pub shallower_completed_iterations: Option<u64>,
    pub frontier_only_reports: Option<u64>,
    pub rules_terminal_reports: Option<u64>,
    /// Exact natural result accepted by UCI's owner after its physical fence.
    /// Transport write/exit success remains a separate process observation.
    pub reports_accepted_for_uci_output: Option<u64>,
    pub completed_reports_accepted_for_uci_output: Option<u64>,
    pub nodes: Option<u64>,
    pub quiescence_nodes: Option<u64>,
    pub tt_hits: Option<u64>,
    pub max_completed_depth: Option<u16>,
}
impl CpuWorkTotals {
    fn zero() -> Self {
        Self {
            tasks_requested: Some(0),
            reports_returned: Some(0),
            requested_depth_completed: Some(0),
            shallower_completed_iterations: Some(0),
            frontier_only_reports: Some(0),
            rules_terminal_reports: Some(0),
            reports_accepted_for_uci_output: Some(0),
            completed_reports_accepted_for_uci_output: Some(0),
            nodes: Some(0),
            quiescence_nodes: Some(0),
            tt_hits: Some(0),
            max_completed_depth: Some(0),
        }
    }
    fn observe(&mut self, report: &CpuObservation) {
        add(&mut self.tasks_requested, 1);
        add(&mut self.reports_returned, 1);
        add(&mut self.nodes, report.nodes);
        add(&mut self.quiescence_nodes, report.quiescence_nodes);
        add(&mut self.tt_hits, report.tt_hits);
        match report.score_scope {
            CpuScoreScope::CompletedIteration
                if report.completed_depth >= report.requested_depth =>
            {
                add(&mut self.requested_depth_completed, 1)
            }
            CpuScoreScope::CompletedIteration => add(&mut self.shallower_completed_iterations, 1),
            CpuScoreScope::FrontierOnly => add(&mut self.frontier_only_reports, 1),
            CpuScoreScope::RulesTerminal => add(&mut self.rules_terminal_reports, 1),
        }
        self.max_completed_depth = self
            .max_completed_depth
            .map(|value| value.max(report.completed_depth));
    }
    fn failed_attempt(&mut self) {
        add(&mut self.tasks_requested, 1);
        // No actual work snapshot was observed; earlier successful totals cannot be
        // presented as the full sum of this process after the missing attempt.
        self.nodes = None;
        self.quiescence_nodes = None;
        self.tt_hits = None;
        self.max_completed_depth = None;
    }
}

macro_rules! pals_totals {
    ($($field:ident),+ $(,)?) => {
        #[derive(Clone, Debug, Eq, PartialEq)]
        #[cfg_attr(feature = "search-work-receipts", derive(serde::Serialize))]
        pub struct PalsWorkTotals {
            $(pub $field: Option<u64>,)+
            pub retained_situations_peak: Option<usize>,
            pub unknown_root_children: Option<u64>,
        }
        impl PalsWorkTotals {
            fn zero() -> Self { Self { $($field: Some(0),)+ retained_situations_peak: Some(0), unknown_root_children: Some(0) } }
            fn observe(&mut self, counters: PalsCounters, root_coverage_observed: bool) {
                $(add(&mut self.$field, counters.$field);)+
                self.retained_situations_peak = self.retained_situations_peak.map(|value| value.max(counters.retained_situations));
                if root_coverage_observed { add(&mut self.unknown_root_children, counters.unknown_root_children as u64); }
                else { self.unknown_root_children = None; }
                if counters.cpu_work_observation_incomplete {
                    self.cpu_nodes = None; self.cpu_quiescence_nodes = None; self.cpu_tt_hits = None;
                }
            }
            fn unknown(&mut self) { $(self.$field = None;)+ self.retained_situations_peak = None; self.unknown_root_children = None; }
        }
    };
}
pals_totals!(
    rounds,
    proposals,
    refutations,
    repairs,
    supported_refutations,
    supported_repairs,
    role_calls,
    consumed_role_outputs,
    proposer_calls,
    critic_calls,
    repair_calls,
    completed_proposer_calls,
    completed_critic_calls,
    completed_repair_calls,
    accepted_proposer_outputs,
    accepted_critic_outputs,
    accepted_repair_outputs,
    cpu_tasks_requested,
    cpu_tasks,
    cpu_nodes,
    cpu_quiescence_nodes,
    cpu_tt_hits,
    completed_cpu_tasks,
    partial_cpu_iterations,
    consumed_cpu_tasks,
    reused_completed_cpu_tasks_consumed,
    consumed_partial_cpu_values,
    consumed_frontier_cpu_values,
    consumed_cached_cpu_values,
    evidence_cache_hits,
    examined_edges
);

fn add(total: &mut Option<u64>, increment: u64) {
    *total = total.and_then(|value| value.checked_add(increment));
}
fn fault() -> ContractError {
    ContractError::new(
        ErrorCode::BackendFailure,
        Stage::Output,
        "search work receipt owner unavailable or counter exhausted",
    )
}
// One fixed-size counter snapshot lives on the stack per admitted go, never
// per node. Keeping it inline avoids a heap allocation on failure/cancel
// accounting paths; this bounded scalar snapshot has no owned buffers.
#[allow(clippy::large_enum_variant)]
pub(super) enum AttemptObservation {
    NoWork,
    CpuStarted,
    CpuFailed(CpuWork),
    Cpu(CpuObservation),
    PalsStarted,
    Pals {
        counters: PalsCounters,
        root_coverage_observed: bool,
    },
}
pub(super) struct CpuObservation {
    nodes: u64,
    quiescence_nodes: u64,
    tt_hits: u64,
    completed_depth: u16,
    score_scope: CpuScoreScope,
    requested_depth: u16,
    best_move: Option<BoardMove>,
}
impl AttemptObservation {
    pub fn cpu(report: &CpuReport, requested_depth: u16) -> Self {
        Self::Cpu(CpuObservation {
            nodes: report.nodes,
            quiescence_nodes: report.quiescence_nodes,
            tt_hits: report.tt_hits,
            completed_depth: report.completed_depth,
            score_scope: report.score_scope,
            requested_depth,
            best_move: report.best_move,
        })
    }
}
#[derive(Clone, Copy)]
struct PendingCpuOutput {
    authority: SearchAuthority,
    best_move: Option<BoardMove>,
    complete: bool,
}
pub(super) struct ProcessWorkJournal {
    receipt: Mutex<ProcessSearchWorkReceipt>,
    // A bounded set of unacknowledged reports, never a growing go journal.
    pending_cpu: Mutex<Vec<PendingCpuOutput>>,
}
impl ProcessWorkJournal {
    pub fn new(kind: SearchKind) -> Self {
        Self {
            receipt: Mutex::new(ProcessSearchWorkReceipt {
                schema_version: 1,
                search_kind: kind,
                go_invocations: 0,
                successful_returns: 0,
                failed_returns: 0,
                canceled_returns: 0,
                deadline_returns: 0,
                physical_unknown_returns: 0,
                active_invocations: 0,
                unobserved_work_invocations: 0,
                cpu: (kind == SearchKind::Cpu).then(CpuWorkTotals::zero),
                pals: (kind == SearchKind::Pals).then(PalsWorkTotals::zero),
            }),
            pending_cpu: Mutex::new(Vec::with_capacity(16)),
        }
    }
    pub fn begin(&self) -> Result<(), ContractError> {
        let mut receipt = self.receipt.lock().map_err(|_| fault())?;
        receipt.go_invocations = receipt.go_invocations.checked_add(1).ok_or_else(fault)?;
        receipt.active_invocations = receipt
            .active_invocations
            .checked_add(1)
            .ok_or_else(fault)?;
        Ok(())
    }
    pub fn finish(
        &self,
        context: &SearchSessionContext,
        result: &Result<SearchSessionReport, SearchSessionFailure>,
        observation: AttemptObservation,
    ) -> Result<(), ContractError> {
        let mut receipt = self.receipt.lock().map_err(|_| fault())?;
        receipt.active_invocations = receipt
            .active_invocations
            .checked_sub(1)
            .ok_or_else(fault)?;
        if result.is_ok() {
            receipt.successful_returns = receipt
                .successful_returns
                .checked_add(1)
                .ok_or_else(fault)?;
        } else {
            receipt.failed_returns = receipt.failed_returns.checked_add(1).ok_or_else(fault)?;
        }
        if context
            .control
            .cancellation
            .load(std::sync::atomic::Ordering::Acquire)
            || context.cancellation.is_canceled()
        {
            receipt.canceled_returns = receipt.canceled_returns.checked_add(1).ok_or_else(fault)?;
        } else if std::time::Instant::now() >= context.control.admission_deadline {
            receipt.deadline_returns = receipt.deadline_returns.checked_add(1).ok_or_else(fault)?;
        }
        if result.as_ref().err().is_some_and(|failure| {
            failure.physical_completion == super::DriverPhysicalCompletion::Unknown
        }) {
            receipt.physical_unknown_returns = receipt
                .physical_unknown_returns
                .checked_add(1)
                .ok_or_else(fault)?;
        }
        let mut unknown = false;
        match observation {
            AttemptObservation::NoWork => {}
            AttemptObservation::Cpu(report) => {
                receipt.cpu.as_mut().ok_or_else(fault)?.observe(&report);
                if result.is_ok() && context.accepts()? {
                    let mut pending = self.pending_cpu.lock().map_err(|_| fault())?;
                    // A single UCI owner cannot subsequently accept a report
                    // from a replaced generation. Prune only that known scope.
                    pending.retain(|entry| {
                        entry.authority.epoch != context.authority.epoch
                            || entry.authority.implementation != context.authority.implementation
                            || (entry.authority.game == context.authority.game
                                && entry.authority.root == context.authority.root)
                    });
                    if pending.len() < 16 {
                        pending.push(PendingCpuOutput {
                            authority: context.authority,
                            best_move: report.best_move,
                            complete: report.score_scope == CpuScoreScope::RulesTerminal
                                || (report.score_scope == CpuScoreScope::CompletedIteration
                                    && report.completed_depth >= report.requested_depth),
                        });
                    } else {
                        let totals = receipt.cpu.as_mut().ok_or_else(fault)?;
                        totals.reports_accepted_for_uci_output = None;
                        totals.completed_reports_accepted_for_uci_output = None;
                        unknown = true;
                    }
                }
            }
            AttemptObservation::CpuStarted => {
                receipt.cpu.as_mut().ok_or_else(fault)?.failed_attempt();
                unknown = true;
            }
            AttemptObservation::CpuFailed(work) => {
                let totals = receipt.cpu.as_mut().ok_or_else(fault)?;
                add(&mut totals.tasks_requested, 1);
                add(&mut totals.nodes, work.nodes);
                add(&mut totals.quiescence_nodes, work.quiescence_nodes);
                add(&mut totals.tt_hits, work.tt_hits);
                if work.nodes > 0 {
                    // The invocation observed actual work but supplied no
                    // report of its last completed iteration. Preserve that
                    // distinct missing gauge rather than inventing depth 0.
                    totals.max_completed_depth = None;
                    unknown = true;
                }
            }
            AttemptObservation::Pals {
                counters,
                root_coverage_observed,
            } => {
                unknown = counters.cpu_work_observation_incomplete;
                receipt
                    .pals
                    .as_mut()
                    .ok_or_else(fault)?
                    .observe(counters, root_coverage_observed);
            }
            AttemptObservation::PalsStarted => {
                receipt.pals.as_mut().ok_or_else(fault)?.unknown();
                unknown = true;
            }
        }
        if unknown {
            receipt.unobserved_work_invocations = receipt
                .unobserved_work_invocations
                .checked_add(1)
                .ok_or_else(fault)?;
        }
        Ok(())
    }
    pub fn snapshot(&self) -> Result<ProcessSearchWorkReceipt, ContractError> {
        self.receipt
            .lock()
            .map(|receipt| receipt.clone())
            .map_err(|_| fault())
    }
    pub fn accept_cpu_output(
        &self,
        authority: SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        let best_move = if bestmove == "0000" {
            None
        } else {
            Some(BoardMove::from_uci(bestmove).map_err(|_| fault())?)
        };
        let accepted = {
            let mut pending = self.pending_cpu.lock().map_err(|_| fault())?;
            pending
                .iter()
                .position(|entry| entry.authority == authority && entry.best_move == best_move)
                .map(|index| pending.swap_remove(index))
        };
        if let Some(accepted) = accepted {
            let mut receipt = self.receipt.lock().map_err(|_| fault())?;
            let totals = receipt.cpu.as_mut().ok_or_else(fault)?;
            if from_report {
                add(&mut totals.reports_accepted_for_uci_output, 1);
                if accepted.complete {
                    add(&mut totals.completed_reports_accepted_for_uci_output, 1);
                }
            } else {
                // A frozen fallback can coincide with the final reported move.
                // Without the exact completion handoff, coincidence does not
                // prove that UCI consumed this task instead of earlier progress.
                totals.reports_accepted_for_uci_output = None;
                if accepted.complete {
                    totals.completed_reports_accepted_for_uci_output = None;
                }
                receipt.unobserved_work_invocations = receipt
                    .unobserved_work_invocations
                    .checked_add(1)
                    .ok_or_else(fault)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        contracts::SessionScope,
        search_driver::{DriverWork, SearchKind},
    };
    use rz_contracts::{
        CancelToken, Digest, GameGeneration, ProcessEpoch, RootGeneration, pals::SearchAuthority,
    };
    use rz_search::driver::SearchControl;
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };
    fn context() -> SearchSessionContext {
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([1; 32]),
        };
        let until = Instant::now() + Duration::from_secs(5);
        SearchSessionContext {
            authority,
            control: SearchControl::new(until, 10),
            cancellation: CancelToken::new(),
            shutdown_deadline: until,
            current: Arc::new(Mutex::new(SessionScope::Search(authority))),
        }
    }
    #[test]
    fn failed_and_canceled_calls_keep_real_work_and_distinguish_root_unknown() {
        let journal = ProcessWorkJournal::new(SearchKind::Pals);
        let context = context();
        journal.begin().unwrap();
        let counters = PalsCounters {
            proposer_calls: 2,
            completed_proposer_calls: 1,
            accepted_proposer_outputs: 1,
            role_calls: 2,
            consumed_role_outputs: 1,
            cpu_tasks_requested: 1,
            cpu_tasks: 1,
            cpu_nodes: 4,
            partial_cpu_iterations: 1,
            consumed_partial_cpu_values: 1,
            retained_situations: 3,
            ..PalsCounters::default()
        };
        journal
            .finish(
                &context,
                &Err(SearchSessionFailure::physical_completion_unknown()),
                AttemptObservation::Pals {
                    counters,
                    root_coverage_observed: false,
                },
            )
            .unwrap();
        journal.begin().unwrap();
        context.control.cancel();
        journal
            .finish(
                &context,
                &Ok(SearchSessionReport {
                    best_move: None,
                    work: DriverWork::Pals {
                        rounds: 0,
                        cpu_nodes: 0,
                        completed_tasks: 0,
                        consumed_role_outputs: 0,
                        retained_situations: 0,
                    },
                }),
                AttemptObservation::NoWork,
            )
            .unwrap();
        let receipt = journal.snapshot().unwrap();
        assert_eq!(
            (
                receipt.go_invocations,
                receipt.successful_returns,
                receipt.failed_returns
            ),
            (2, 1, 1)
        );
        assert_eq!(
            (
                receipt.active_invocations,
                receipt.canceled_returns,
                receipt.physical_unknown_returns
            ),
            (0, 1, 1)
        );
        assert_eq!(receipt.unobserved_work_invocations, 0);
        let totals = receipt.pals.unwrap();
        assert_eq!(totals.cpu_nodes, Some(4));
        assert_eq!(totals.partial_cpu_iterations, Some(1));
        assert_eq!(totals.completed_cpu_tasks, Some(0));
        assert_eq!(totals.proposer_calls, Some(2));
        assert_eq!(totals.accepted_proposer_outputs, Some(1));
        assert_eq!(totals.unknown_root_children, None);
        assert_eq!(totals.retained_situations_peak, Some(3));
    }
    #[test]
    fn missing_failed_cpu_observation_is_unknown_not_zero() {
        let journal = ProcessWorkJournal::new(SearchKind::Cpu);
        let context = context();
        journal.begin().unwrap();
        journal
            .finish(
                &context,
                &Err(SearchSessionFailure::physical_completion_unknown()),
                AttemptObservation::CpuStarted,
            )
            .unwrap();
        let receipt = journal.snapshot().unwrap();
        assert_eq!(receipt.unobserved_work_invocations, 1);
        let totals = receipt.cpu.unwrap();
        assert_eq!(totals.tasks_requested, Some(1));
        assert_eq!(totals.reports_returned, Some(0));
        assert_eq!(totals.nodes, None);
        assert_eq!(totals.quiescence_nodes, None);
    }
    #[test]
    fn failed_cpu_work_is_retained_without_inventing_a_completed_report() {
        for (work, unknown, depth) in [
            (
                CpuWork {
                    nodes: 3,
                    quiescence_nodes: 1,
                    tt_hits: 2,
                },
                1,
                None,
            ),
            (CpuWork::default(), 0, Some(0)),
        ] {
            let journal = ProcessWorkJournal::new(SearchKind::Cpu);
            let context = context();
            journal.begin().unwrap();
            journal
                .finish(
                    &context,
                    &Err(SearchSessionFailure::debug(
                        "CpuSearch",
                        &"observed failure",
                    )),
                    AttemptObservation::CpuFailed(work),
                )
                .unwrap();
            let receipt = journal.snapshot().unwrap();
            assert_eq!(receipt.unobserved_work_invocations, unknown);
            let totals = receipt.cpu.unwrap();
            assert_eq!(totals.tasks_requested, Some(1));
            assert_eq!(totals.reports_returned, Some(0));
            assert_eq!(totals.nodes, Some(work.nodes));
            assert_eq!(totals.quiescence_nodes, Some(work.quiescence_nodes));
            assert_eq!(totals.tt_hits, Some(work.tt_hits));
            assert_eq!(totals.max_completed_depth, depth);
            assert_eq!(totals.completed_reports_accepted_for_uci_output, Some(0));
        }
    }
    #[test]
    fn cpu_completion_is_consumed_only_at_the_exact_natural_output_handoff() {
        let journal = ProcessWorkJournal::new(SearchKind::Cpu);
        let context = context();
        let movement = BoardMove::from_uci("e2e4").unwrap();
        journal.begin().unwrap();
        let report = Ok(SearchSessionReport {
            best_move: Some(movement),
            work: DriverWork::Cpu {
                nodes: 1,
                completed_depth: 2,
            },
        });
        journal
            .finish(
                &context,
                &report,
                AttemptObservation::Cpu(CpuObservation {
                    nodes: 1,
                    quiescence_nodes: 0,
                    tt_hits: 0,
                    completed_depth: 2,
                    requested_depth: 2,
                    score_scope: CpuScoreScope::CompletedIteration,
                    best_move: Some(movement),
                }),
            )
            .unwrap();
        let mut stale = context.authority;
        stale.root = RootGeneration(99);
        journal.accept_cpu_output(stale, "e2e4", true).unwrap();
        assert_eq!(
            journal
                .snapshot()
                .unwrap()
                .cpu
                .unwrap()
                .completed_reports_accepted_for_uci_output,
            Some(0)
        );
        journal
            .accept_cpu_output(context.authority, "e2e4", true)
            .unwrap();
        journal
            .accept_cpu_output(context.authority, "e2e4", true)
            .unwrap();
        let totals = journal.snapshot().unwrap().cpu.unwrap();
        assert_eq!(totals.requested_depth_completed, Some(1));
        assert_eq!(totals.completed_reports_accepted_for_uci_output, Some(1));
    }
}
