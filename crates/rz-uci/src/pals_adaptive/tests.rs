//! Actual manager/owner integration, with authored neural outputs only.
//! The fixture never certifies a completed PALS refinement: its finite
//! semantic scope ends as explicit Unknown after checked worker evidence.
#[path = "fixed_trace_tests.rs"]
mod fixed_trace;

use super::*;
use crate::pals_adaptive::AdaptivePortDriver;
use crate::pals_native::ticketed::{
    TicketExecutionEvidence, TicketNativeBackend, TicketPhysicalResult, TicketPreparation,
    TicketServiceLimits, TicketedNativeService,
};
use rz_contracts::pals::{Move16, RepresentationKey, RoleOutput, RolePayload, SearchAuthority};
use rz_contracts::pals_adaptive::{CpuWorkerId, ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::{
    Digest, ExecutionId, GameGeneration, OwnerId, PrecisionProfile, ProcessEpoch, RootGeneration,
    Wdl,
};
use rz_position::contracts::{ContractPosition, ContractState};
use rz_position::{BoardMove, Position};
use rz_runtime::adaptive::{ControllerConfig, CoordinatorPolicy};
use rz_runtime::contracts::ContractSystemClock;
use rz_search::cpu::{
    CpuCompletion, CpuConfig, CpuEngine, CpuLimits, CpuOrderingPolicy, CpuProfile, CpuReport,
    CpuResumePolicy,
};
use rz_search::cpu_value::BootstrapCpuValue;
use rz_search::cpu_worker::{CpuSliceReport, CpuWorker, CpuWorkerProfile};
use rz_search::external_cpu::worker::ExternalWorkerReport;
use rz_search::external_cpu::{CpuEndpointSelection, ExternalCpuConfig};
use rz_search::pals::coordinator::{
    ClosedFrame, CoordinatorWork, FrameBudget, FrameInput, FrameTransition, PhysicalRoleCompletion,
    ProposerRankingPublished, RefinementStage, RoleFailure,
};
use rz_search::pals::engine::RoleQueryPurpose;
use rz_search::pals::review_context::{
    ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy, ReviewView,
};
use rz_search::pals::store::{PalsStores, StateId, StoreLimits};
use sha2::Sha256;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const WINDOW: Duration = Duration::from_secs(5);
const TOTAL_NODES: u64 = 100_000;

#[derive(Default)]
struct NativeFacts {
    started: Mutex<Vec<RoleTicketEnvelope>>,
    accepted: Mutex<Vec<RoleTicketEnvelope>>,
    child_gate: AtomicBool,
    finish_called: AtomicBool,
    factories: std::sync::atomic::AtomicU64,
}
struct NativeFixture {
    clock: ContractSystemClock,
    facts: Arc<NativeFacts>,
    finish_ok: bool,
}
impl<Q: Clone + Send + Sync + 'static> TicketNativeBackend<ReviewView, Q> for NativeFixture {
    type Prepared = RepresentationKey;
    type Output = RoleOutput;
    fn clock(&self) -> ContractSystemClock {
        self.clock.clone()
    }
    fn prepare(
        &mut self,
        question: &TicketPreparation<ReviewView, Q>,
        _: &AtomicBool,
    ) -> Result<(Self::Prepared, RepresentationKey), RoleFailure> {
        let mut candidates = Sha256::new();
        for movement in &question.view.query().legal {
            candidates.update(
                Move16::encode(rz_contracts::Move::try_from(*movement).unwrap())
                    .bits()
                    .to_le_bytes(),
            );
        }
        let key = RepresentationKey {
            input: Digest(question.view.digest()),
            history: question.seal.root_state.semantic,
            candidates: Digest(candidates.finalize().into()),
            mask_positions: Digest([4; 32]),
            model: Digest([5; 32]),
            encoding: Digest([6; 32]),
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 1,
            record_revision: 0,
            role: question.role,
        };
        Ok((key, key))
    }
    fn prepared_host_bytes(&self, _: &Self::Prepared) -> u64 {
        std::mem::size_of::<RepresentationKey>() as u64
    }
    fn run(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, Q>,
        _: &Self::Prepared,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<Self::Output> {
        self.facts.started.lock().unwrap().push(*ticket);
        let until = Instant::now() + Duration::from_secs(2);
        while !question.view.query().prefix.is_empty()
            && !self.facts.child_gate.load(Ordering::Acquire)
        {
            if canceled.load(Ordering::Acquire) || Instant::now() >= until {
                return TicketPhysicalResult {
                    physical: PhysicalRoleCompletion::Confirmed,
                    output: Err(RoleFailure::Canceled),
                };
            }
            thread::sleep(Duration::from_millis(1));
        }
        TicketPhysicalResult {
            physical: PhysicalRoleCompletion::Confirmed,
            output: Ok(RoleOutput {
                id: ticket.id,
                authority: ticket.authority,
                key: ticket.representation,
                payload: RolePayload::Proposal {
                    candidate_logits: (0..question.view.query().legal.len())
                        .map(|i| i as f32)
                        .collect(),
                    wdl: Wdl::try_new(0.25, 0.5, 0.25, 1e-4).unwrap(),
                },
                private_latent: vec![0.; 16 * 384],
            }),
        }
    }
    fn output_host_bytes(&self, output: &Self::Output) -> u64 {
        let RolePayload::Proposal {
            candidate_logits, ..
        } = &output.payload
        else {
            unreachable!("fixture emits proposer output only")
        };
        ((candidate_logits.len() + output.private_latent.len()) * 4) as u64
    }
    fn acceptance_output_digest(&self, output: &Self::Output) -> Option<Digest> {
        rz_search::pals::coordinator::native_acceptance_output_digest(output).ok()
    }
    fn execution_evidence(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Option<TicketExecutionEvidence> {
        Some(TicketExecutionEvidence {
            execution: Some(ExecutionId {
                epoch: ticket.id.epoch,
                sequence: ticket.id.sequence,
            }),
            // This is an authored CPU neural-service fixture, not GPU evidence.
            completed_nn_inputs: Some(1),
            measurement_complete: true,
            physical_fresh: Some(true),
            accounting: None,
        })
    }
    fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, Q>,
        _: &Self::Prepared,
        output: &Self::Output,
    ) -> Result<(), RoleFailure> {
        if output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != ticket.representation
            || ticket.view.view_digest.0 != question.view.seal_digest()
        {
            return Err(RoleFailure::InvalidOutput);
        }
        self.facts.accepted.lock().unwrap().push(*ticket);
        Ok(())
    }
    fn discard(&mut self, _: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
        Ok(())
    }
    fn finish(&mut self, _: Instant) -> bool {
        self.facts.finish_called.store(true, Ordering::Release);
        self.finish_ok
    }
    fn quiesce(&mut self, _: Instant) -> bool {
        // This authored CPU backend has no pending physical buffers after
        // service.run returns. It is not GPU completion evidence.
        self.finish_ok
    }
}

#[derive(Clone, Copy)]
enum FixtureScope {
    TwoIndependentRoles,
    OwnCpuResume,
}
struct DriverFixture {
    authority: SearchAuthority,
    rules: Arc<ContractState>,
    stores: PalsStores,
    index: ReviewContextIndex,
    state: StateId,
    scope: FixtureScope,
    consumed: Vec<RoleTicketEnvelope>,
    closed: Vec<ClosedFrame>,
    expected_cpu: BTreeMap<u16, CpuReport>,
    actual_cpu: Vec<CpuSliceReport>,
    fail_raw_once: bool,
    fail_role_once: bool,
    fail_after_role_consumption_once: bool,
    raw_cpu_receipts: Vec<ExecutionId>,
}
impl DriverFixture {
    fn input(
        &mut self,
        stage: RefinementStage,
        first: Option<BoardMove>,
    ) -> FrameInput<ReviewView, u64> {
        let mut endpoint = Position::from_snapshot(
            self.rules.rules().snapshot(),
            rz_position::PositionLimits::default(),
        )
        .unwrap();
        let prefix = first.into_iter().collect::<Vec<_>>();
        for movement in &prefix {
            endpoint.make_move(*movement).unwrap();
        }
        let legal = endpoint.legal_moves();
        let view = Arc::new(
            self.index
                .select(
                    &self.stores,
                    self.state,
                    ReviewQuerySequences {
                        purpose: RoleQueryPurpose::ProposePolicy,
                        prefix: &prefix,
                        proposal: &[],
                        counterexample: None,
                        legal: &legal,
                        divergences: &[],
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let view_seal = ReviewViewSeal {
            authority: self.authority,
            view_sequence: view.sequence(),
            context_policy: Digest(view.policy_digest()),
            view_digest: Digest(view.seal_digest()),
            root_state: self.rules.snapshot().identity(),
            actual_tokens: view.token_count(),
            selected_records: view.records().len() as u16,
            anchors: view.anchors().len() as u8,
            length_bucket: view.length_bucket(),
        };
        let query = first.map_or(0, |mv| {
            u64::from(Move16::encode(rz_contracts::Move::try_from(mv).unwrap()).bits())
        });
        FrameInput {
            stage,
            question_digest: Digest(view.digest()),
            question: Arc::new(query),
            public_context: Digest(view.record_scope_digest()),
            view_seal,
            view,
        }
    }
}
impl AdaptiveSearchDriver<u64> for DriverFixture {
    fn root_rules(&self) -> Arc<ContractState> {
        Arc::clone(&self.rules)
    }
    fn ranking_published(
        &mut self,
        ranking: &ProposerRankingPublished,
    ) -> Result<Vec<CandidateFrame<u64>>, AdaptiveManagerError> {
        assert_eq!(ranking.source().ticket.authority, self.authority);
        assert!(
            ranking
                .rules()
                .rules()
                .snapshot()
                .same_state(self.rules.rules().snapshot())
        );
        let width = match self.scope {
            FixtureScope::TwoIndependentRoles => 2,
            FixtureScope::OwnCpuResume => 1,
        };
        ranking
            .ranked_candidates()
            .iter()
            .take(width)
            .map(|movement| {
                let first = BoardMove::try_from(movement.decode().unwrap()).unwrap();
                Ok(CandidateFrame {
                    first_move: *movement,
                    input: self.input(RefinementStage::ProposalContinuation, Some(first)),
                    budget: FrameBudget {
                        role_submissions_remaining: 4,
                        original_cpu_credit: TOTAL_NODES,
                        original_store_byte_limit: 1 << 20,
                    },
                    deferred: None,
                    initialized: None,
                })
            })
            .collect()
    }
    fn validate_role(
        &self,
        work: &CoordinatorWork<ReviewView, u64>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        if output.id != ticket.id
            || output.authority != self.authority
            || output.key != ticket.representation
            || ticket.question != work.input.question_digest
            || ticket.view != work.input.view_seal
            || !work.input.view.validates(&self.stores)
        {
            return Err(AdaptiveManagerError::Invalid(
                "fixture original question/view",
            ));
        }
        let RolePayload::Proposal {
            candidate_logits, ..
        } = &output.payload
        else {
            return Err(AdaptiveManagerError::Invalid("fixture proposer head"));
        };
        if candidate_logits.len() != work.input.view.query().legal.len()
            || candidate_logits.iter().any(|v| !v.is_finite())
        {
            return Err(AdaptiveManagerError::Invalid("fixture legal head"));
        }
        Ok(())
    }
    fn commit_role(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        _: Option<TicketExecutionEvidence>,
    ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
        self.validate_role(work, ticket, output)?;
        if self.fail_role_once {
            self.fail_role_once = false;
            return Err(AdaptiveManagerError::Semantic(
                "authored role hot-commit failure".into(),
            ));
        }
        if self.fail_after_role_consumption_once {
            self.fail_after_role_consumption_once = false;
            self.role_consumed(ticket, &work.input.view)?;
            return Err(AdaptiveManagerError::Semantic(
                "authored next-view materialization failure after actual consumption".into(),
            ));
        }
        let transition = match self.scope {
            FixtureScope::TwoIndependentRoles => FrameTransition::Unknown {
                reason: "authored integration fixture ends after actual role acceptance",
            },
            FixtureScope::OwnCpuResume => {
                let mut input = work.input.clone();
                input.stage = RefinementStage::CpuValidation;
                FrameTransition::Continue(input)
            }
        };
        Ok(CheckedStep {
            evidence: Digest(ticket.representation.input.0),
            transition,
            fresh_pair: None,
            rules_pair: None,
            queued_recheck: None,
        })
    }
    fn cpu_question(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        _: &[CpuWorkerProfile],
        _: Option<&ExternalCpuConfig>,
        _: CpuEndpointSelection,
    ) -> Result<CpuQuestion, AdaptiveManagerError> {
        let mut endpoint = Position::from_snapshot(
            self.rules.rules().snapshot(),
            rz_position::PositionLimits::default(),
        )
        .unwrap();
        for movement in &work.input.view.query().prefix {
            endpoint.make_move(*movement).unwrap();
        }
        let restriction = endpoint
            .legal_moves()
            .first()
            .copied()
            .into_iter()
            .collect();
        Ok(CpuQuestion::Own(OwnCpuQuestion {
            position: ContractPosition::new(OwnerId(71), endpoint),
            root_moves: Some(restriction),
            target_depth: 2,
            total_native_node_credit: TOTAL_NODES,
        }))
    }
    fn commit_cpu(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        report: &CpuSliceReport,
    ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
        assert!(report.terminal && !report.continuation_retained);
        let actual = &report.result.as_ref().unwrap().report;
        let expected = &self.expected_cpu[&work.original_first_move.bits()];
        assert_eq!(actual.completion, CpuCompletion::DepthLimit);
        assert_eq!(actual.completed_depth, expected.completed_depth);
        assert_eq!(actual.score, expected.score);
        assert_eq!(actual.pv, expected.pv);
        assert_eq!(report.cumulative_nodes, expected.nodes);
        self.actual_cpu.push(report.clone());
        Ok(CheckedStep {
            evidence: Digest([42; 32]),
            transition: FrameTransition::Unknown {
                reason: "authored integration fixture ends after actual private CPU completion",
            },
            fresh_pair: None,
            rules_pair: None,
            queued_recheck: None,
        })
    }
    fn cpu_slice_observed(
        &mut self,
        _: &CoordinatorWork<ReviewView, u64>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        if self.fail_raw_once {
            self.fail_raw_once = false;
            return Err(AdaptiveManagerError::Semantic(
                "authored hot-commit failure".into(),
            ));
        }
        assert!(!self.raw_cpu_receipts.contains(&report.grant.execution));
        self.raw_cpu_receipts.push(report.grant.execution);
        Ok(())
    }
    fn commit_external(
        &mut self,
        _: &CoordinatorWork<ReviewView, u64>,
        _: &ExternalWorkerReport,
    ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "fixture forbids external execution",
        ))
    }
    fn publish_closed(&mut self, closed: &ClosedFrame) -> Result<(), AdaptiveManagerError> {
        assert!(closed.unknown.is_some());
        assert!(closed.completion_scope.is_none());
        self.closed.push(closed.clone());
        Ok(())
    }
    fn role_consumed(
        &mut self,
        ticket: &RoleTicketEnvelope,
        view: &Arc<ReviewView>,
    ) -> Result<(), AdaptiveManagerError> {
        assert!(self.consumed.iter().all(|old| old.id != ticket.id));
        assert_eq!(ticket.view.view_digest.0, view.seal_digest());
        self.index.consume(view, &self.stores).unwrap();
        self.consumed.push(*ticket);
        Ok(())
    }
    fn role_consumption_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        view: &ReviewView,
    ) -> Result<bool, AdaptiveManagerError> {
        assert_eq!(ticket.view.view_digest.0, view.seal_digest());
        Ok(self.consumed.iter().any(|old| old == ticket))
    }
}

type Manager = AdaptiveExecutionManager<u64, NativeFixture, DriverFixture>;

fn cpu_engine() -> CpuEngine {
    CpuEngine::with_evaluator_ordering_and_resume_policy(
        CpuConfig {
            profile: CpuProfile::PlanAssisted,
            tt_entries: 128,
            max_depth: 2,
            quiescence_ply: 4,
        },
        Arc::new(BootstrapCpuValue::default()),
        CpuOrderingPolicy::LegacyMvvLvaV1,
        CpuResumePolicy::PausedStack,
    )
    .unwrap()
}
fn fixture(
    scope: FixtureScope,
    finish_ok: bool,
    credit: u64,
) -> (
    Manager,
    Arc<NativeFacts>,
    FrameInput<ReviewView, u64>,
    Instant,
) {
    fixture_with_resource_limits(scope, finish_ok, credit, None)
}
fn fixture_with_resource_limits(
    scope: FixtureScope,
    finish_ok: bool,
    credit: u64,
    limits: Option<AdaptiveExecutionLimits>,
) -> (
    Manager,
    Arc<NativeFacts>,
    FrameInput<ReviewView, u64>,
    Instant,
) {
    let position = Position::from_fen("3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
    let mut expected_cpu = BTreeMap::new();
    if matches!(scope, FixtureScope::OwnCpuResume) {
        // The independent direct search runs before the manager exists. It is
        // not a synchronous CPU callback from an adaptive manager turn.
        for first in position.legal_moves() {
            let mut endpoint = position.clone();
            endpoint.make_move(first).unwrap();
            let restriction = vec![endpoint.legal_moves()[0]];
            let report = cpu_engine()
                .analyze_root_moves(
                    &endpoint,
                    &restriction,
                    CpuLimits {
                        max_depth: 2,
                        max_nodes: TOTAL_NODES,
                        deadline: None,
                    },
                    &AtomicBool::new(false),
                )
                .unwrap();
            expected_cpu.insert(
                Move16::encode(rz_contracts::Move::try_from(first).unwrap()).bits(),
                report,
            );
        }
    }
    let clock = ContractSystemClock::new(ProcessEpoch(117));
    let until = clock.origin() + WINDOW;
    let authority = SearchAuthority {
        epoch: ProcessEpoch(117),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([19; 32]),
    };
    let rules = Arc::new(
        ContractPosition::new(OwnerId(17), position.clone())
            .export()
            .unwrap(),
    );
    let mut stores = PalsStores::new(StoreLimits::default());
    let root = stores.focus_actual_moves(position.snapshot()).unwrap();
    let state = stores.situations.get(root).unwrap().state;
    let mut driver = DriverFixture {
        authority,
        rules,
        stores,
        index: ReviewContextIndex::default(),
        state,
        scope,
        consumed: Vec::new(),
        closed: Vec::new(),
        expected_cpu,
        actual_cpu: Vec::new(),
        fail_raw_once: false,
        fail_role_once: false,
        fail_after_role_consumption_once: false,
        raw_cpu_receipts: Vec::new(),
    };
    let root_input = driver.input(RefinementStage::ProposerRanking, None);
    let facts = Arc::new(NativeFacts::default());
    let thread_facts = Arc::clone(&facts);
    let native_clock = clock.clone();
    let factory = move || {
        thread_facts.factories.fetch_add(1, Ordering::Relaxed);
        Ok(NativeFixture {
            clock: native_clock,
            facts: thread_facts,
            finish_ok,
        })
    };
    let native = match limits {
        Some(limits) => TicketedNativeService::spawn_with_capacity(
            factory,
            TicketServiceLimits::default(),
            limits.role_tickets_max,
            limits.worker_commands_max,
        ),
        None => TicketedNativeService::spawn(factory, TicketServiceLimits::default()),
    }
    .unwrap();
    let mut cpu = cpu_engine();
    let worker = match limits {
        Some(limits) => {
            cpu.set_paused_stack_max_bytes(limits.paused_bytes_per_worker_max)
                .unwrap();
            CpuWorker::spawn_notified_retaining_with_capacity(
                CpuWorkerId(0),
                cpu,
                native.signal(),
                limits.worker_commands_max,
            )
            .unwrap_or_else(|(_, e)| panic!("fixture owner start failed: {e:?}"))
        }
        None => CpuWorker::spawn_notified(CpuWorkerId(0), cpu, native.signal()).unwrap(),
    };
    let registration = AdaptiveManagerRegistration {
        authority,
        hard_deadline: until,
        token_budget: 512,
        policy: CoordinatorPolicy::Fixed,
        first_request_sequence: 1,
        native_cpu_node_limit: TOTAL_NODES,
        endpoints: CpuEndpointSelection::deployment(),
        controller: ControllerConfig {
            fixed_node_credit: credit,
            ..ControllerConfig::default()
        },
        device_digest: Digest([71; 32]),
        cpu_device_digest: Digest([72; 32]),
    };
    let manager = match limits {
        Some(limits) => AdaptiveExecutionManager::new_retaining_with_limits(
            registration,
            clock,
            limits,
            AdaptiveManagerSources {
                native,
                own_workers: vec![worker],
                external: None,
                driver,
            },
        )
        .unwrap_or_else(|e| panic!("actual resource admission failed: {}", e.error)),
        None => {
            AdaptiveExecutionManager::new(registration, clock, native, vec![worker], None, driver)
                .unwrap()
        }
    };
    (manager, facts, root_input, until)
}

#[test]
fn explicit_lower_caps_bound_actual_admission_and_complete_both_independent_questions() {
    let limits = AdaptiveExecutionLimits {
        waiting_jobs_max: 1,
        worker_commands_max: 1,
        role_tickets_max: 1,
        paused_bytes_per_worker_max: 1 << 20,
    };
    let (mut manager, facts, root, until) =
        fixture_with_resource_limits(FixtureScope::TwoIndependentRoles, true, 13, Some(limits));
    assert_eq!(manager.limits, limits);
    assert_eq!(manager.native.command_capacity(), 1);
    assert_eq!(manager.native.pending_capacity(), 1);
    assert_eq!(
        manager.own[0].worker.as_ref().unwrap().command_capacity(),
        1
    );
    assert_eq!(
        manager.own[0].worker.as_ref().unwrap().paused_bytes_limit(),
        1 << 20
    );
    manager.prepare_root_ranking(root.clone()).unwrap();
    assert!(matches!(
        manager.prepare_root_ranking(root),
        Err(AdaptiveManagerError::Invalid(
            "root ranking admission order"
        ))
    ));
    facts.child_gate.store(true, Ordering::Release);
    pump(&mut manager, until, |manager| {
        assert!(manager.roles.len() <= limits.role_tickets_max);
        assert!(manager.ready.len() + manager.validator_waiting.len() <= limits.waiting_jobs_max);
        manager.driver().closed.len() == 2 && !manager.has_pending_work()
    });
    assert_eq!(manager.driver().consumed.len(), 3);
    assert_eq!(manager.counters().consumed_role_outputs, 3);
    assert!(manager.failures().next().is_none());
    assert!(shutdown(&mut manager, until).permits_archive_close());
}

#[test]
fn explicit_resource_limits_reject_each_out_of_range_axis() {
    let defaults = AdaptiveExecutionLimits::default();
    assert!(defaults.validate().is_ok());
    for limits in [
        AdaptiveExecutionLimits {
            waiting_jobs_max: 0,
            ..defaults
        },
        AdaptiveExecutionLimits {
            waiting_jobs_max: 65,
            ..defaults
        },
        AdaptiveExecutionLimits {
            worker_commands_max: 0,
            ..defaults
        },
        AdaptiveExecutionLimits {
            worker_commands_max: 9,
            ..defaults
        },
        AdaptiveExecutionLimits {
            role_tickets_max: 0,
            ..defaults
        },
        AdaptiveExecutionLimits {
            role_tickets_max: 17,
            ..defaults
        },
        AdaptiveExecutionLimits {
            paused_bytes_per_worker_max: 0,
            ..defaults
        },
        AdaptiveExecutionLimits {
            paused_bytes_per_worker_max: (8 << 20) + 1,
            ..defaults
        },
    ] {
        assert!(limits.validate().is_err());
    }
}

#[test]
fn explicit_one_command_owner_repeated_resume_preserves_credit_and_completion() {
    let limits = AdaptiveExecutionLimits {
        waiting_jobs_max: 1,
        worker_commands_max: 1,
        role_tickets_max: 1,
        paused_bytes_per_worker_max: 1 << 20,
    };
    let (mut manager, facts, root, until) =
        fixture_with_resource_limits(FixtureScope::OwnCpuResume, true, 1, Some(limits));
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        assert!(manager.waiting_jobs_occupancy() <= limits.waiting_jobs_max);
        manager.driver().closed.len() == 1 && !manager.has_pending_work()
    });
    assert!(manager.counters().cpu_yields > 0);
    assert_eq!(manager.driver().actual_cpu.len(), 1);
    let nodes = manager.driver().actual_cpu[0].cumulative_nodes;
    assert_eq!(manager.native_cpu_usage().unwrap(), (nodes, 0));
    assert!(manager.failures().next().is_none());
    let receipt = shutdown(&mut manager, until);
    assert!(receipt.permits_archive_close());
    assert_eq!(
        (receipt.native_cpu_spent, receipt.native_cpu_reserved),
        (nodes, 0)
    );
}
fn pump(manager: &mut Manager, until: Instant, predicate: impl Fn(&Manager) -> bool) {
    while !predicate(manager) {
        assert!(
            Instant::now() < until,
            "manager fixture timeout: {:?}",
            manager.failures().collect::<Vec<_>>()
        );
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
}
fn shutdown(manager: &mut Manager, until: Instant) -> AdaptiveShutdownReceipt {
    manager.begin_shutdown(until).unwrap();
    pump(manager, until, |manager| {
        let receipt = manager.shutdown_receipt().unwrap();
        receipt.native_owner_joined && receipt.own_cpu_joined == 1
    });
    manager.shutdown_receipt().unwrap()
}

#[test]
fn actual_search_port_commits_native_and_private_cpu_without_inline_model_calls() {
    use rz_runtime::contracts::ContractClock as _;
    use rz_search::pals::engine::adaptive_port::{
        AdaptiveEnginePort, AdaptiveModelNamespace, AdaptivePortModel, AdaptivePortQuestion,
    };
    use rz_search::pals::engine::{PalsCompletion, PalsConfig, PalsEngine, PalsLimits};
    use rz_search::pals::review_context::ReviewIndexLimits;
    use rz_search::pals::value::{MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity};
    use std::sync::atomic::AtomicU64;

    let position = Position::from_fen("3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
    let clock = ContractSystemClock::new(ProcessEpoch(221));
    let until = clock.origin() + WINDOW;
    let authority = SearchAuthority {
        epoch: ProcessEpoch(221),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([89; 32]),
    };
    // This model has metadata only: every physical method is Unsupported.
    // Therefore an accidental synchronous model call cannot satisfy the test.
    let value = ModelValueIdentity {
        semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
        model: "authored-native-fixture".into(),
        encoding: "authored-fixture-encoding".into(),
        precision: "fp32".into(),
        model_epoch: [5; 32],
    };
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        Digest([5; 32]),
        Digest([6; 32]),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let model =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace).unwrap();
    let mut engine = PalsEngine::new(
        PalsConfig {
            beam_width: 1,
            line_plies: 1,
            ..PalsConfig::default()
        },
        model,
        cpu_engine(),
    )
    .unwrap()
    .with_selected_review_context(
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
    )
    .unwrap();
    engine.new_game();
    let issuer = Arc::new(AtomicU64::new(100));
    let port = AdaptiveEnginePort::new(
        engine,
        position.clone(),
        authority,
        rz_contracts::Deadline {
            clock: clock.domain(),
            at: MonotonicTick(WINDOW.as_nanos() as u64),
        },
        PalsLimits {
            deadline: until,
            max_rounds: 1,
            max_cpu_nodes: TOTAL_NODES,
            cpu_depth: 1,
        },
        Arc::new(move || Ok(OwnerId(issuer.fetch_add(1, Ordering::Relaxed)))),
    )
    .unwrap();
    let facts = Arc::new(NativeFacts::default());
    facts.child_gate.store(true, Ordering::Release);
    let owner_facts = Arc::clone(&facts);
    let owner_clock = clock.clone();
    let native = TicketedNativeService::<ReviewView, AdaptivePortQuestion, NativeFixture>::spawn(
        move || {
            Ok(NativeFixture {
                clock: owner_clock,
                facts: owner_facts,
                finish_ok: true,
            })
        },
        TicketServiceLimits::default(),
    )
    .unwrap();
    let worker = CpuWorker::spawn_notified(CpuWorkerId(0), cpu_engine(), native.signal()).unwrap();
    let mut driver = AdaptivePortDriver::new(
        port,
        until,
        Arc::new(native.signal()),
        rz_search::external_cpu::ExternalSlicePolicy::default(),
    )
    .unwrap();
    let root = driver.root_input().unwrap();
    let mut manager = AdaptiveExecutionManager::new(
        AdaptiveManagerRegistration {
            authority,
            hard_deadline: until,
            token_budget: 512,
            policy: CoordinatorPolicy::Fixed,
            first_request_sequence: 1,
            native_cpu_node_limit: TOTAL_NODES,
            endpoints: CpuEndpointSelection::deployment(),
            controller: ControllerConfig::default(),
            device_digest: Digest([90; 32]),
            cpu_device_digest: Digest([91; 32]),
        },
        clock,
        native,
        vec![worker],
        None,
        driver,
    )
    .unwrap();
    manager.prepare_root_ranking(root).unwrap();
    while manager.has_pending_work() {
        assert!(
            Instant::now() < until,
            "actual port timeout: {:?}",
            manager.failures().collect::<Vec<_>>()
        );
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
    assert!(manager.failures().next().is_none());
    let result = manager.driver().result(PalsCompletion::RoundLimit).unwrap();
    assert!(position.legal_moves().contains(&result.best_move.unwrap()));
    assert_eq!(result.counters.rounds, 1);
    assert!(result.counters.completed_cpu_tasks > 0);
    assert!(manager.counters().consumed_role_outputs > 0);
    assert!(manager.native_cpu_usage().unwrap().0 > 0);
    manager.begin_shutdown(until).unwrap();
    while manager.phase() != ManagerPhase::Finished {
        assert!(Instant::now() < until);
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
    assert!(manager.shutdown_receipt().unwrap().permits_archive_close());
}

#[test]
fn root_ranking_fans_out_two_original_views_without_staling_or_double_consumption() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.counters().submitted_role_tickets == 3
    });
    assert_eq!(manager.counters().consumed_role_outputs, 1);
    assert_eq!(manager.driver().consumed.len(), 1);
    facts.child_gate.store(true, Ordering::Release);
    pump(&mut manager, until, |manager| {
        manager.driver().closed.len() == 2 && !manager.has_pending_work()
    });
    let consumed = &manager.driver().consumed;
    assert_eq!(consumed.len(), 3);
    assert_eq!(consumed[1].authority, consumed[2].authority);
    assert_eq!(consumed[1].authority.root, RootGeneration(1));
    assert_ne!(consumed[1].id, consumed[2].id);
    assert_ne!(consumed[1].question, consumed[2].question);
    assert_ne!(consumed[1].view.view_digest, consumed[2].view.view_digest);
    assert_eq!(facts.accepted.lock().unwrap().len(), 3);
    assert_eq!(manager.counters().consumed_role_outputs, 3);
    assert_eq!(manager.counters().completed_nn_inputs, 3);
    assert_eq!(manager.native_cpu_usage().unwrap(), (0, 0));
    assert!(manager.failures().next().is_none());
    let receipt = shutdown(&mut manager, until);
    assert!(receipt.permits_archive_close());
    assert_eq!(receipt.pending_tickets, 0);
}

#[test]
fn two_actual_root_fences_reuse_one_native_factory_and_preserve_cpu_engine_work() {
    // This case observes a completed attempt across root reuse. Yield and its
    // final cancellation attempt are exercised separately below.
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuResume, true, 13);
    facts.child_gate.store(true, Ordering::Release);
    let clock = manager.clock.clone();
    let mut registration = manager.registration;
    manager.prepare_root_ranking(root).unwrap();
    let first = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(first.permits_root_reuse());
    assert!(!first.native_owner_joined);
    assert!(!facts.finish_called.load(Ordering::Acquire));
    let parts = match manager.into_persistent_parts() {
        Ok(parts) => parts,
        Err(rejected) => panic!("persistent transfer: {}", rejected.error),
    };
    let (native, mut cpu, external, mut driver, continuation, first) = parts.into_components();
    assert!(external.is_none());
    assert_eq!(cpu.len(), 1);
    let owner = cpu.pop().unwrap();
    let actual_cpu_work = owner
        .engine
        .last_attempt_work()
        .expect("actual CPU searched first root");
    assert!(actual_cpu_work.nodes > 0);
    registration.authority.root = RootGeneration(registration.authority.root.0 + 1);
    registration.first_request_sequence = continuation.first_request_sequence();
    let old_high_water = first.request_sequence_high_water;
    driver.authority = registration.authority;
    driver.scope = FixtureScope::TwoIndependentRoles;
    let focused = driver
        .stores
        .focus_actual_moves(driver.rules.rules().snapshot().clone())
        .unwrap();
    driver.state = driver.stores.situations.get(focused).unwrap().state;
    driver.closed.clear();
    driver.consumed.clear();
    let input = driver.input(RefinementStage::ProposerRanking, None);
    let worker =
        CpuWorker::spawn_notified(owner.profile.owner, owner.engine, native.signal()).unwrap();
    let mut next = AdaptiveExecutionManager::new_continued(
        registration,
        clock,
        native,
        vec![worker],
        None,
        driver,
        continuation,
    )
    .unwrap();
    next.prepare_root_ranking(input).unwrap();
    let second = next.run_root_until(until, &AtomicBool::new(false)).unwrap();
    assert!(second.permits_root_reuse());
    assert!(second.request_sequence_high_water > old_high_water);
    assert!(
        next.driver()
            .consumed
            .iter()
            .all(|ticket| ticket.id.sequence > old_high_water
                && ticket.authority.root == RootGeneration(2))
    );
    assert_eq!(facts.factories.load(Ordering::Acquire), 1);
    let parts = match next.into_persistent_parts() {
        Ok(parts) => parts,
        Err(rejected) => panic!("second persistent transfer: {}", rejected.error),
    };
    let (mut native, cpu, external, _, _, _) = parts.into_components();
    assert!(external.is_none());
    assert_eq!(cpu[0].engine.last_attempt_work(), Some(actual_cpu_work));
    // Process teardown is separate from both root quiescence receipts.
    native.begin_shutdown(until);
    while !native.try_join().unwrap() {
        assert!(Instant::now() < until);
        while native.poll().is_some() {}
        thread::sleep(Duration::from_millis(1));
    }
    assert!(facts.finish_called.load(Ordering::Acquire));
}

#[test]
fn actual_private_cpu_yield_resume_settles_exact_work_and_leaves_no_reservation() {
    // One native node forces a real checkpoint before this fixed job completes.
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuResume, true, 1);
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.driver().closed.len() == 1 && !manager.has_pending_work()
    });
    assert_eq!(manager.driver().actual_cpu.len(), 1);
    let actual = &manager.driver().actual_cpu[0];
    assert!(
        manager.counters().cpu_yields > 0,
        "actual traversal must cross slice boundaries"
    );
    assert_eq!(
        manager.counters().cpu_slice_receipts,
        manager.counters().cpu_yields + 1
    );
    assert_eq!(
        manager.native_cpu_usage().unwrap(),
        (actual.cumulative_nodes, 0)
    );
    assert_eq!(manager.counters().consumed_role_outputs, 2);
    assert!(manager.failures().next().is_none());
    let spent = actual.cumulative_nodes;
    let receipt = shutdown(&mut manager, until);
    assert_eq!(receipt.native_cpu_spent, spent);
    assert_eq!(receipt.native_cpu_reserved, 0);
    assert!(receipt.permits_archive_close());
}

#[test]
fn shutdown_without_physical_native_fence_quarantines_and_refuses_archive_close() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, false, 4096);
    drop(root);
    let receipt = shutdown(&mut manager, until);
    assert!(facts.finish_called.load(Ordering::Acquire));
    assert!(receipt.native_owner_joined);
    assert_eq!(receipt.own_cpu_joined, 1);
    assert!(!receipt.native_physical_shutdown_confirmed);
    assert!(receipt.quarantined);
    assert!(!receipt.permits_archive_close());
    assert_eq!(manager.phase(), ManagerPhase::Quarantined);
    let rejected = manager
        .driver_mut()
        .input(RefinementStage::ProposerRanking, None);
    assert!(manager.prepare_root_ranking(rejected).is_err());
    assert_eq!(receipt.counters.consumed_role_outputs, 0);
}

#[test]
fn failed_role_hot_commit_retains_original_ticket_until_actual_owner_drain() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    facts.child_gate.store(true, Ordering::Release);
    manager.driver_mut().fail_role_once = true;
    manager.prepare_root_ranking(root).unwrap();
    let receipt = manager.run_until(until, &AtomicBool::new(false)).unwrap();
    assert!(receipt.permits_archive_close());
    assert_eq!(receipt.pending_tickets, 0);
    assert_eq!(receipt.native_cpu_reserved, 0);
    assert_eq!(receipt.counters.consumed_role_outputs, 1);
    assert!(manager.failures().any(|error| matches!(error,
        AdaptiveManagerError::Semantic(message) if message == "authored role hot-commit failure")));
    assert!(!manager.failures().any(|error| matches!(
        error,
        AdaptiveManagerError::Invalid("unknown semantic retirement")
    )));
    assert!(facts.finish_called.load(Ordering::Acquire));
    assert!(manager.roles.is_empty());
}

#[test]
fn invalid_continued_registration_returns_live_native_and_same_cpu_owner() {
    let (mut manager, facts, _, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    let mut registration = manager.registration;
    let clock = manager.clock.clone();
    manager.begin_root_quiesce(until).unwrap();
    pump(&mut manager, until, |manager| {
        manager.phase() == ManagerPhase::Finished
    });
    let parts = match manager.into_persistent_parts() {
        Ok(parts) => parts,
        Err(rejected) => panic!("initial root fence: {}", rejected.error),
    };
    let (native, mut cpu, external, driver, continuation, receipt) = parts.into_components();
    assert!(receipt.permits_root_reuse());
    let cpu = cpu.remove(0);
    let owner = cpu.profile.owner;
    let worker = CpuWorker::spawn_notified(owner, cpu.engine, native.signal()).unwrap();
    registration.authority.root = RootGeneration(2);
    registration.first_request_sequence = continuation.first_request_sequence();
    registration.native_cpu_node_limit = 0;
    let rejected = match AdaptiveExecutionManager::new_continued_retaining(
        registration,
        clock,
        AdaptiveManagerSources {
            native,
            own_workers: vec![worker],
            external,
            driver,
        },
        continuation,
    ) {
        Ok(_) => panic!("invalid node registration admitted"),
        Err(rejected) => rejected,
    };
    assert!(rejected.continuation.is_some());
    assert_eq!(rejected.sources.own_workers[0].profile().owner, owner);
    assert!(
        rejected
            .sources
            .native
            .root_fence()
            .unwrap()
            .permits_reuse()
    );
    assert!(!facts.finish_called.load(Ordering::Acquire));
    assert_eq!(facts.factories.load(Ordering::Acquire), 1);
    let mut sources = rejected.sources;
    sources.native.begin_shutdown(until);
    sources.own_workers[0].request_shutdown();
    while !sources.native.try_join().unwrap() {
        while sources.native.poll().is_some() {}
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(1));
    }
    loop {
        if let Some(stopped) = sources.own_workers[0].try_join().unwrap() {
            assert!(stopped.thread_joined && stopped.engine_recoverable);
            break;
        }
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn next_view_failure_preserves_actual_original_role_consumption() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    manager.driver_mut().fail_after_role_consumption_once = true;
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.failures().next().is_some()
    });
    let receipt = shutdown(&mut manager, until);
    assert!(receipt.permits_archive_close());
    assert_eq!(manager.counters().consumed_role_outputs, 2);
    assert_eq!(manager.driver().consumed.len(), 2);
    assert_eq!(manager.driver().closed.len(), 0);
    assert!(manager.roles.is_empty());
    assert!(matches!(
        manager.failures().next(),
        Some(AdaptiveManagerError::Semantic(_))
    ));
    assert!(facts.finish_called.load(Ordering::Acquire));
}

#[test]
fn changed_tick_observer_can_close_root_before_late_child_consumption() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    assert!(manager.root_observation_snapshot().is_err());
    manager.prepare_root_ranking(root).unwrap();
    let mut saw_original_ranking = false;
    let receipt = manager
        .run_root_until_observed(until, &AtomicBool::new(false), &mut |manager| {
            if !saw_original_ranking && manager.counters().consumed_role_outputs == 1 {
                saw_original_ranking = true;
                assert_eq!(manager.phase(), ManagerPhase::Running);
                manager.begin_root_quiesce(until)?;
            }
            Ok(())
        })
        .unwrap();
    assert!(saw_original_ranking);
    assert!(receipt.permits_root_reuse());
    assert_eq!(manager.counters().consumed_role_outputs, 1);
    assert!(!facts.finish_called.load(Ordering::Acquire));
    let observed = manager.root_observation_snapshot().unwrap();
    assert_eq!(observed.authority(), receipt.authority);
    assert_eq!(observed.close_mode(), AdaptiveCloseMode::RootQuiesce);
    assert_eq!(observed.manager_counters(), manager.counters());
    assert_eq!(observed.cpu_outstanding_reserved(), 0);
    assert!(observed.role_ticket_accounting().admitted >= 1);
    assert_eq!(observed.cache_evaluations_accepted(), None);
    let parts = match manager.into_persistent_parts() {
        Ok(parts) => parts,
        Err(rejected) => panic!("observed root owner recovery: {}", rejected.error),
    };
    let (mut native, _, _, _, _, _) = parts.into_components();
    native.begin_shutdown(until);
    while !native.try_join().unwrap() {
        while native.poll().is_some() {}
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(facts.finish_called.load(Ordering::Acquire));
}

#[test]
fn cancel_during_actual_cpu_slice_preserves_final_receipts_before_archive_close() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuResume, true, 13);
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.own[0].binding.is_some()
    });
    manager.begin_shutdown(until).unwrap();
    assert!(!manager.shutdown_receipt().unwrap().permits_archive_close());
    let receipt = shutdown(&mut manager, until);
    assert_eq!(receipt.native_cpu_reserved, 0);
    assert_eq!(receipt.own_cpu_joined, receipt.expected_own_cpu_owners);
    assert!(receipt.permits_archive_close());
    assert!(
        manager.failures().next().is_none(),
        "{:?}",
        manager.failures().collect::<Vec<_>>()
    );
    // Revoke cannot accept the late CPU value as another semantic step.
    assert!(manager.driver().actual_cpu.is_empty());
}

#[test]
fn failed_hot_yield_commit_retains_actual_receipt_until_cancel_settlement() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuResume, true, 1);
    manager.driver_mut().fail_raw_once = true;
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.failures().next().is_some()
    });
    assert!(matches!(
        manager.failures().next(),
        Some(AdaptiveManagerError::Semantic(_))
    ));
    let receipt = shutdown(&mut manager, until);
    assert_eq!(receipt.native_cpu_reserved, 0);
    assert!(receipt.permits_archive_close());
    assert!(!manager.driver().raw_cpu_receipts.is_empty());
    assert!(manager.driver().actual_cpu.is_empty());
}

#[cfg(test)]
mod adaptive_partial_feedback {
    use super::*;
    use rz_search::cpu::{CpuIterationProgress, CpuScoreScope};
    use rz_search::cpu_worker::{CpuJobTermination, CpuWorkerEvent, CpuWorkerProgress};

    // Only the continuation is authored. Every tested slice and retained token
    // comes from the same real PausedStack owner used by the existing fixture.
    struct PartialDriver {
        inner: DriverFixture,
        admissions: Vec<(CoordinatorWork<ReviewView, u64>, CpuJobEnvelope, SliceGrant)>,
        partial_commits: Vec<(CoordinatorWork<ReviewView, u64>, CpuSliceReport)>,
        terminal_commits: Vec<CpuSliceReport>,
        detached: Vec<(CoordinatorWork<ReviewView, u64>, CpuSliceReport)>,
        discarded: Vec<(CoordinatorWork<ReviewView, u64>, CpuJobTermination)>,
    }

    impl AdaptiveSearchDriver<u64> for PartialDriver {
        fn root_rules(&self) -> Arc<ContractState> {
            self.inner.root_rules()
        }
        fn ranking_published(
            &mut self,
            ranking: &ProposerRankingPublished,
        ) -> Result<Vec<CandidateFrame<u64>>, AdaptiveManagerError> {
            self.inner.ranking_published(ranking)
        }
        fn validate_role(
            &self,
            work: &CoordinatorWork<ReviewView, u64>,
            ticket: &RoleTicketEnvelope,
            output: &RoleOutput,
        ) -> Result<(), AdaptiveManagerError> {
            self.inner.validate_role(work, ticket, output)
        }
        fn commit_role(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            ticket: &RoleTicketEnvelope,
            output: &RoleOutput,
            physical: Option<TicketExecutionEvidence>,
        ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
            self.inner.commit_role(work, ticket, output, physical)
        }
        fn cpu_question(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            profiles: &[CpuWorkerProfile],
            external: Option<&ExternalCpuConfig>,
            selection: CpuEndpointSelection,
        ) -> Result<CpuQuestion, AdaptiveManagerError> {
            self.inner.cpu_question(work, profiles, external, selection)
        }
        fn cpu_admitted(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            job: &CpuJobEnvelope,
            grant: &SliceGrant,
        ) -> Result<(), AdaptiveManagerError> {
            if let Some((original, original_job, original_grant)) = self.admissions.first() {
                assert_eq!(work.frame, original.frame);
                assert_eq!(work.input.stage, original.input.stage);
                assert_eq!(work.input.question_digest, original.input.question_digest);
                assert_eq!(work.input.view_seal, original.input.view_seal);
                assert_eq!(job, original_job);
                assert_eq!(grant.owner, original_grant.owner);
            }
            self.admissions.push((work.clone(), *job, *grant));
            Ok(())
        }
        fn cpu_partial_continuation_eligible(
            &self,
            work: &CoordinatorWork<ReviewView, u64>,
            report: &CpuSliceReport,
        ) -> bool {
            work.input.stage == RefinementStage::CpuValidation
                && !report.terminal
                && !report.admission_rejected
                && report.continuation_retained
                && report.actual_work.is_some()
                && report.result.as_ref().is_ok_and(|actual| {
                    actual.report.completion == CpuCompletion::Yielded
                        && actual.report.score_scope == CpuScoreScope::CompletedIteration
                        && actual.report.completed_depth > 0
                })
        }
        fn commit_cpu(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            report: &CpuSliceReport,
        ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
            let transition = if report.terminal {
                assert!(!report.continuation_retained);
                self.terminal_commits.push(report.clone());
                FrameTransition::Unknown {
                    reason: "authored terminal-only control ends after the actual CPU receipt",
                }
            } else {
                assert!(self.cpu_partial_continuation_eligible(work, report));
                assert!(self.partial_commits.is_empty());
                assert!(
                    !self
                        .inner
                        .raw_cpu_receipts
                        .contains(&report.grant.execution)
                );
                self.partial_commits.push((work.clone(), report.clone()));
                let first =
                    BoardMove::try_from(work.original_first_move.decode().unwrap()).unwrap();
                let mut input = self
                    .inner
                    .input(RefinementStage::CriticDivergence, Some(first));
                input.question_digest =
                    receipt_digest(b"authored-partial-next-question", report.grant.execution);
                FrameTransition::Continue(input)
            };
            Ok(CheckedStep {
                evidence: receipt_digest(b"authored-partial-feedback", report.grant.execution),
                transition,
                fresh_pair: None,
                rules_pair: None,
                queued_recheck: None,
            })
        }
        fn cpu_slice_observed(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            report: &CpuSliceReport,
        ) -> Result<(), AdaptiveManagerError> {
            self.inner.cpu_slice_observed(work, report)
        }
        fn cpu_detached_slice_observed(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            report: &CpuSliceReport,
        ) -> Result<(), AdaptiveManagerError> {
            self.inner.cpu_slice_observed(work, report)?;
            self.detached.push((work.clone(), report.clone()));
            Ok(())
        }
        fn cpu_token_discarded(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            termination: &CpuJobTermination,
        ) -> Result<(), AdaptiveManagerError> {
            self.discarded.push((work.clone(), termination.clone()));
            Ok(())
        }
        fn commit_external(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            report: &ExternalWorkerReport,
        ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
            self.inner.commit_external(work, report)
        }
        fn publish_closed(&mut self, closed: &ClosedFrame) -> Result<(), AdaptiveManagerError> {
            self.inner.publish_closed(closed)
        }
        fn role_consumed(
            &mut self,
            ticket: &RoleTicketEnvelope,
            view: &Arc<ReviewView>,
        ) -> Result<(), AdaptiveManagerError> {
            self.inner.role_consumed(ticket, view)
        }
        fn role_consumption_recorded(
            &self,
            ticket: &RoleTicketEnvelope,
            view: &ReviewView,
        ) -> Result<bool, AdaptiveManagerError> {
            self.inner.role_consumption_recorded(ticket, view)
        }
    }

    type PartialManager = AdaptiveExecutionManager<u64, NativeFixture, PartialDriver>;

    fn partial_fixture(
        policy: CoordinatorPolicy,
        root_generation: RootGeneration,
    ) -> (
        PartialManager,
        Arc<NativeFacts>,
        FrameInput<ReviewView, u64>,
        Instant,
    ) {
        let position = Position::from_fen("3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let clock = ContractSystemClock::new(ProcessEpoch(117));
        let until = clock.origin() + WINDOW;
        let authority = SearchAuthority {
            epoch: ProcessEpoch(117),
            game: GameGeneration(1),
            root: root_generation,
            implementation: Digest([19; 32]),
        };
        let rules = Arc::new(
            ContractPosition::new(OwnerId(17), position.clone())
                .export()
                .unwrap(),
        );
        let mut stores = PalsStores::new(StoreLimits::default());
        let root_registration = stores
            .prepare_hot_root_registration(position.snapshot(), root_generation.0)
            .unwrap();
        let root = stores.hot_focus_registered_root(root_registration).unwrap();
        assert_eq!(stores.generation(), authority.root.0);
        let state = stores.situations.get(root).unwrap().state;
        let mut driver = PartialDriver {
            inner: DriverFixture {
                authority,
                rules,
                stores,
                index: ReviewContextIndex::default(),
                state,
                scope: FixtureScope::OwnCpuResume,
                consumed: Vec::new(),
                closed: Vec::new(),
                expected_cpu: BTreeMap::new(),
                actual_cpu: Vec::new(),
                fail_raw_once: false,
                fail_role_once: false,
                fail_after_role_consumption_once: false,
                raw_cpu_receipts: Vec::new(),
            },
            admissions: Vec::new(),
            partial_commits: Vec::new(),
            terminal_commits: Vec::new(),
            detached: Vec::new(),
            discarded: Vec::new(),
        };
        let input = driver.inner.input(RefinementStage::ProposerRanking, None);
        let facts = Arc::new(NativeFacts::default());
        let thread_facts = Arc::clone(&facts);
        let native_clock = clock.clone();
        let native = TicketedNativeService::spawn(
            move || {
                thread_facts.factories.fetch_add(1, Ordering::Relaxed);
                Ok(NativeFixture {
                    clock: native_clock,
                    facts: thread_facts,
                    finish_ok: true,
                })
            },
            TicketServiceLimits::default(),
        )
        .unwrap();
        let worker =
            CpuWorker::spawn_notified(CpuWorkerId(0), cpu_engine(), native.signal()).unwrap();
        let registration = AdaptiveManagerRegistration {
            authority,
            hard_deadline: until,
            token_budget: 512,
            policy,
            first_request_sequence: 1,
            native_cpu_node_limit: TOTAL_NODES,
            endpoints: CpuEndpointSelection::deployment(),
            controller: ControllerConfig {
                policy,
                fixed_node_credit: 1,
                ..ControllerConfig::default()
            },
            device_digest: Digest([71; 32]),
            cpu_device_digest: Digest([72; 32]),
        };
        let manager =
            AdaptiveExecutionManager::new(registration, clock, native, vec![worker], None, driver)
                .unwrap();
        (manager, facts, input, until)
    }

    fn pump_partial(
        manager: &mut PartialManager,
        until: Instant,
        predicate: impl Fn(&PartialManager) -> bool,
    ) {
        while !predicate(manager) {
            assert!(Instant::now() < until, "partial fixture timeout");
            manager.tick().unwrap();
            assert!(manager.failures().next().is_none());
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn start_cpu(
        manager: &mut PartialManager,
        facts: &NativeFacts,
        root: FrameInput<ReviewView, u64>,
        until: Instant,
    ) {
        facts.child_gate.store(true, Ordering::Release);
        manager.prepare_root_ranking(root).unwrap();
        pump_partial(manager, until, |manager| manager.own[0].binding.is_some());
    }

    fn next_slice(manager: &mut PartialManager, until: Instant) -> CpuWorkerEventObservation {
        loop {
            assert!(Instant::now() < until, "actual CPU receipt timeout");
            if let Some(event) = manager.own[0]
                .worker
                .as_ref()
                .unwrap()
                .try_event_observed()
                .unwrap()
            {
                return event;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn forward_first(manager: &mut PartialManager, until: Instant) -> CpuSliceReport {
        loop {
            let event = next_slice(manager, until);
            let (event, _) = event.into_parts();
            let CpuWorkerEvent::Slice(report) = event else {
                panic!("live original owner terminated before its first completed iteration");
            };
            assert!(
                !report.terminal,
                "one-node slices must retain the depth-two traversal"
            );
            manager
                .own_event(0, CpuWorkerEvent::Slice(report.clone()))
                .unwrap();
            if manager.own[0].binding.as_ref().unwrap().feedback_forwarded {
                return report;
            }
            manager.resume_cpu(0, 1, Duration::from_millis(8)).unwrap();
        }
    }

    fn frame_fingerprint(
        manager: &PartialManager,
        frame: FrameId,
    ) -> Option<(FramePhase, RefinementStage, Digest, ReviewViewSeal, Digest)> {
        manager.machine.frame(frame).map(|current| {
            (
                current.phase,
                current.input.stage,
                current.input.question_digest,
                current.input.view_seal,
                current.input.public_context,
            )
        })
    }

    fn shutdown_partial(manager: &mut PartialManager, until: Instant) -> AdaptiveShutdownReceipt {
        manager.begin_shutdown(until).unwrap();
        while {
            let receipt = manager.shutdown_receipt().unwrap();
            !receipt.native_owner_joined || receipt.own_cpu_joined != 1
        } {
            assert!(Instant::now() < until, "partial owner shutdown timeout");
            manager.tick().unwrap();
            thread::sleep(Duration::from_millis(1));
        }
        manager.shutdown_receipt().unwrap()
    }

    #[test]
    fn real_completed_iteration_handoff_resumes_only_original_custody() {
        let (mut manager, facts, root, until) =
            partial_fixture(CoordinatorPolicy::Adaptive, RootGeneration(1));
        start_cpu(&mut manager, &facts, root, until);
        let first = forward_first(&mut manager, until);
        let original = manager.own[0].binding.as_ref().unwrap().work.clone();
        let original_job = manager.own[0].binding.as_ref().unwrap().envelope;
        let fingerprint = frame_fingerprint(&manager, original.frame);
        assert_eq!(fingerprint.as_ref().unwrap().0, FramePhase::Ready);
        assert_eq!(
            fingerprint.as_ref().unwrap().1,
            RefinementStage::CriticDivergence
        );
        assert_ne!(
            fingerprint.as_ref().unwrap().2,
            original.input.question_digest
        );
        assert_eq!(manager.driver.partial_commits.len(), 1);
        assert!(
            !manager
                .driver
                .inner
                .raw_cpu_receipts
                .contains(&first.grant.execution)
        );

        // A repeated publication of this already-settled real iteration cannot
        // bind its execution to the newly materialized logical question.
        let actual = &first.result.as_ref().unwrap().report;
        let progress = CpuWorkerProgress {
            job: first.grant.job,
            execution: first.grant.execution,
            owner: first.grant.owner,
            sequence: first.progress_sequence,
            iteration: CpuIterationProgress {
                best_move: actual.best_move,
                score: actual.score,
                completed_depth: actual.completed_depth,
                nodes: actual.nodes,
            },
        };
        manager.own_progress(0, progress.clone()).unwrap();
        manager.own_progress(0, progress).unwrap();
        assert_eq!(frame_fingerprint(&manager, original.frame), fingerprint);

        let terminal = loop {
            manager.resume_cpu(0, 1, Duration::from_millis(8)).unwrap();
            let binding = manager.own[0].binding.as_ref().unwrap();
            assert!(binding.feedback_forwarded);
            assert_eq!(binding.envelope, original_job);
            assert_eq!(
                binding.work.input.question_digest,
                original.input.question_digest
            );
            assert_eq!(binding.work.input.view_seal, original.input.view_seal);
            assert_eq!(frame_fingerprint(&manager, original.frame), fingerprint);
            let (event, _) = next_slice(&mut manager, until).into_parts();
            let CpuWorkerEvent::Slice(report) = event else {
                panic!("resumed original traversal terminated without its slice receipt");
            };
            manager
                .own_event(0, CpuWorkerEvent::Slice(report.clone()))
                .unwrap();
            assert_eq!(frame_fingerprint(&manager, original.frame), fingerprint);
            assert_eq!(manager.driver.partial_commits.len(), 1);
            assert!(manager.driver.terminal_commits.is_empty());
            assert!(manager.driver.inner.closed.is_empty());
            if report.terminal {
                break report;
            }
        };
        assert_eq!(
            terminal.result.as_ref().unwrap().report.completion,
            CpuCompletion::DepthLimit
        );
        assert!(manager.own[0].binding.is_none());
        assert!(manager.driver.detached.iter().all(|(work, report)| {
            work.frame == original.frame
                && work.input.question_digest == original.input.question_digest
                && work.input.view_seal == original.input.view_seal
                && report.grant.job == original_job.id
                && report.grant.owner == first.grant.owner
        }));
        assert_eq!(
            manager.native_cpu_usage().unwrap(),
            (terminal.cumulative_nodes, 0)
        );
        assert_eq!(
            manager.counters.cpu_slice_receipts,
            manager.counters.cpu_yields + 1
        );

        // A duplicate terminal is rejected before another semantic consumption.
        assert!(
            manager
                .own_event(0, CpuWorkerEvent::Slice(terminal))
                .is_err()
        );
        assert_eq!(frame_fingerprint(&manager, original.frame), fingerprint);
        assert_eq!(manager.driver.partial_commits.len(), 1);
        assert!(manager.driver.terminal_commits.is_empty());
        let receipt = shutdown_partial(&mut manager, until);
        assert_eq!(receipt.native_cpu_reserved, 0);
        assert!(!receipt.permits_archive_close());
    }

    #[test]
    fn forwarded_paused_token_discard_after_cancel_never_retires_new_question() {
        let (mut manager, facts, root, until) =
            partial_fixture(CoordinatorPolicy::Adaptive, RootGeneration(1));
        start_cpu(&mut manager, &facts, root, until);
        let first = forward_first(&mut manager, until);
        let original = manager.own[0].binding.as_ref().unwrap().work.clone();
        manager.begin_shutdown(until).unwrap();
        let revoked = frame_fingerprint(&manager, original.frame);
        let closed = manager.driver.inner.closed.len();
        // Check the old token discard before the separate, intentional root
        // retirement at the all-owner join fence clears every frame.
        let (event, timing) = next_slice(&mut manager, until).into_parts();
        assert!(matches!(&event, CpuWorkerEvent::JobTerminated(_)));
        assert!(timing.is_none());
        manager.own_event(0, event).unwrap();
        manager.flush_deferred().unwrap();
        assert_eq!(frame_fingerprint(&manager, original.frame), revoked);
        assert_eq!(manager.driver.inner.closed.len(), closed);
        assert_eq!(manager.driver.discarded.len(), 1);
        let receipt = shutdown_partial(&mut manager, until);
        assert!(frame_fingerprint(&manager, original.frame).is_none());
        assert_eq!(manager.driver.inner.closed.len(), closed);
        assert_eq!(manager.driver.partial_commits.len(), 1);
        assert!(manager.driver.terminal_commits.is_empty());
        assert_eq!(manager.driver.discarded.len(), 1);
        let (work, termination) = &manager.driver.discarded[0];
        assert_eq!(work.input.question_digest, original.input.question_digest);
        assert_eq!(work.input.view_seal, original.input.view_seal);
        assert_eq!(termination.job, first.grant.job);
        assert_eq!(termination.owner, first.grant.owner);
        assert_eq!(termination.last_execution, first.grant.execution);
        assert_eq!(receipt.native_cpu_reserved, 0);
        assert!(receipt.permits_archive_close());
        assert!(manager.failures().next().is_none());
    }

    #[test]
    fn old_root_yield_receipt_cannot_change_a_new_root_live_frame() {
        let (mut old, facts, root, until) =
            partial_fixture(CoordinatorPolicy::Adaptive, RootGeneration(1));
        start_cpu(&mut old, &facts, root, until);
        let old_report = forward_first(&mut old, until);
        assert!(shutdown_partial(&mut old, until).permits_archive_close());

        let (mut next, facts, root, until) =
            partial_fixture(CoordinatorPolicy::Adaptive, RootGeneration(2));
        start_cpu(&mut next, &facts, root, until);
        let original = next.own[0].binding.as_ref().unwrap().work.clone();
        let fingerprint = frame_fingerprint(&next, original.frame);
        assert_ne!(old_report.grant.job.authority, original.frame.authority);
        assert!(
            next.own_event(0, CpuWorkerEvent::Slice(old_report))
                .is_err()
        );
        assert_eq!(frame_fingerprint(&next, original.frame), fingerprint);
        assert_eq!(
            next.own[0]
                .binding
                .as_ref()
                .unwrap()
                .work
                .input
                .question_digest,
            original.input.question_digest
        );
        assert!(next.driver.partial_commits.is_empty());
        assert!(next.driver.terminal_commits.is_empty());
        assert!(next.driver.detached.is_empty());
        let receipt = shutdown_partial(&mut next, until);
        assert_eq!(receipt.native_cpu_reserved, 0);
        assert!(!receipt.permits_archive_close());
    }

    #[test]
    fn fixed_and_synchronous_keep_partial_receipts_on_the_original_cpu_step() {
        for policy in [CoordinatorPolicy::Fixed, CoordinatorPolicy::Synchronous] {
            let (mut manager, facts, root, until) = partial_fixture(policy, RootGeneration(1));
            start_cpu(&mut manager, &facts, root, until);
            let original = manager.own[0].binding.as_ref().unwrap().work.clone();
            let terminal = loop {
                let (event, _) = next_slice(&mut manager, until).into_parts();
                let CpuWorkerEvent::Slice(report) = event else {
                    panic!("control owner terminated without its slice receipt");
                };
                manager
                    .own_event(0, CpuWorkerEvent::Slice(report.clone()))
                    .unwrap();
                assert!(manager.driver.partial_commits.is_empty());
                assert!(manager.driver.detached.is_empty());
                if report.terminal {
                    break report;
                }
                let binding = manager.own[0].binding.as_ref().unwrap();
                assert!(!binding.feedback_forwarded);
                assert_eq!(
                    binding.work.input.question_digest,
                    original.input.question_digest
                );
                assert_eq!(
                    manager.machine.frame(original.frame).unwrap().input.stage,
                    RefinementStage::CpuValidation
                );
                manager.resume_cpu(0, 1, Duration::from_millis(8)).unwrap();
            };
            assert_eq!(manager.driver.terminal_commits.len(), 1);
            assert_eq!(manager.driver.inner.closed.len(), 1);
            assert_eq!(
                manager.native_cpu_usage().unwrap(),
                (terminal.cumulative_nodes, 0)
            );
            assert_eq!(
                manager.counters.cpu_slice_receipts,
                manager.counters.cpu_yields + 1
            );
            assert!(shutdown_partial(&mut manager, until).permits_archive_close());
            assert!(manager.failures().next().is_none());
        }
    }

    #[test]
    fn scheduling_yield_ignores_closed_owners_and_preserves_live_control_errors() {
        use rz_runtime::CompletionSignal;
        use rz_runtime::adaptive::ControlMailbox;

        let terminal = ControlMailbox::new((), CompletionSignal::default());
        let terminal_before = terminal.seal_terminal();
        assert!(!scheduling_yield(&terminal, 1).unwrap());
        assert_eq!(terminal.snapshot(), terminal_before);
        assert_eq!(terminal.snapshot().yield_sequence, 0);
        assert_eq!(terminal.snapshot().yield_ack, 0);

        let canceled = ControlMailbox::new((), CompletionSignal::default());
        canceled.request_cancel(1).unwrap();
        let canceled_before = canceled.snapshot();
        assert!(!scheduling_yield(&canceled, 2).unwrap());
        assert_eq!(canceled.snapshot(), canceled_before);
        assert_eq!(canceled.snapshot().yield_sequence, 0);
        assert_eq!(canceled.snapshot().yield_ack, 0);

        let live = ControlMailbox::new((), CompletionSignal::default());
        assert!(scheduling_yield(&live, 1).unwrap());
        assert_eq!(live.snapshot().yield_sequence, 1);
        assert_eq!(live.snapshot().yield_ack, 0);
        let requested = live.snapshot();
        assert!(matches!(
            scheduling_yield(&live, 2),
            Err(AdaptiveManagerError::Invalid("adaptive Yield control"))
        ));
        assert!(matches!(
            scheduling_yield(&live, 1),
            Err(AdaptiveManagerError::Invalid("adaptive Yield control"))
        ));
        assert_eq!(live.snapshot(), requested);
    }
}
