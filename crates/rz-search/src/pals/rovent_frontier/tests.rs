//! Authored contract evidence, not actual model/GPU or speed acceptance.
use super::*;
use crate::contract_time::{ContractClock, InstantClock};
use crate::cpu::{CpuError, CpuWork};
use crate::cpu_worker::CpuCreditLedger;
use crate::pals::store::{SituationId, StoreLimits};
use crate::pals::value::{MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity};
use rz_contracts::pals_adaptive::{CpuJobId, CpuWorkerId};
use rz_contracts::{
    ExecutionId, GameGeneration, OwnerId, PrecisionProfile, ProcessEpoch, RootGeneration,
};
use rz_position::contracts::ContractPosition;
use std::time::{Duration, Instant};

fn mv(text: &str) -> BoardMove {
    BoardMove::from_uci(text).unwrap()
}

struct Fixture {
    stores: PalsStores,
    clock: InstantClock,
    frontier: RoventFrontier,
    root_line: LineId,
    root_situation: SituationId,
}

impl Fixture {
    fn new(position: Position, goal: SearchGoal, nodes: Option<u64>, native: bool) -> Self {
        let mut stores = PalsStores::new(StoreLimits {
            states: 128,
            situations: 128,
            ..Default::default()
        });
        let root_situation = stores.hot_focus_actual_moves(position.snapshot()).unwrap();
        let root_state = stores.situations.get(root_situation).unwrap().state;
        let root_line = stores.lines.root(root_state).unwrap();
        let clock = InstantClock::new(ClockDomain(ProcessEpoch(71)), Instant::now());
        let authority = SearchAuthority {
            epoch: clock.domain().0,
            game: GameGeneration(1),
            root: RootGeneration(stores.generation()),
            implementation: Digest([7; 32]),
        };
        let rules = Arc::new(
            ContractPosition::new(OwnerId(71), position)
                .export()
                .unwrap(),
        );
        let moves = Arc::new(RootMoveDomain::new(&rules, None).unwrap());
        let namespace = Arc::new(
            AdaptiveModelNamespace::new(
                ModelValueIdentity {
                    semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
                    model: "authored-frontier-fixture".into(),
                    encoding: "authored-selected-review".into(),
                    precision: "fp32".into(),
                    model_epoch: [8; 32],
                },
                Digest([5; 32]),
                Digest([6; 32]),
                PrecisionProfile::Fp32,
                1,
            )
            .unwrap(),
        );
        let ledger = native.then(|| CpuCreditLedger::new(authority, 100, clock).unwrap());
        let frontier = RoventFrontier::new(
            FrontierRegistration {
                authority,
                mode: if native {
                    ExecutionMode::CpuPcr
                } else {
                    ExecutionMode::CpuPc
                },
                original_deadline: clock
                    .deadline_at(clock.origin() + Duration::from_secs(30))
                    .unwrap(),
                root_budget: Digest([9; 32]),
                goal,
                nodes,
                selective_policy: Digest([10; 32]),
                root_rules: rules,
                root_moves: moves,
                root_state,
                model_namespace: namespace,
                native_ledger: ledger.as_ref().map(CpuCreditLedger::identity),
            },
            FrontierLimits {
                maximum_goal_plies: NonZeroU16::new(32).unwrap(),
                maximum_frames: 64,
                maximum_paths: 128,
                maximum_mate_nodes: 128,
                maximum_native_receipts: 16,
            },
            &stores,
            clock.domain(),
            clock.now().unwrap(),
        )
        .unwrap();
        Self {
            stores,
            clock,
            frontier,
            root_line,
            root_situation,
        }
    }

    fn line(&mut self, moves: &[BoardMove]) -> LineId {
        self.stores.hot_append_line(self.root_line, moves).unwrap()
    }

    fn observe(&mut self, moves: &[BoardMove]) -> Result<FrontierPath, FrontierError> {
        let line = self.line(moves);
        self.frontier.observe_stored_line(
            &self.stores,
            line,
            self.clock.domain(),
            self.clock.now().unwrap(),
        )
    }
}

#[test]
fn zero_one_nodes_and_cached_edges_preserve_the_registered_unit() {
    for nodes in [0, 1] {
        let mut fixture = Fixture::new(
            Position::startpos(),
            SearchGoal::default(),
            Some(nodes),
            false,
        );
        assert!(matches!(
            fixture.observe(&[mv("e2e4")]),
            Err(FrontierError::NodeBudget)
        ));
        let snapshot = fixture.frontier.snapshot();
        assert_eq!(snapshot.work.total_node_visits(), Some(nodes));
        assert_eq!(snapshot.work.checked_transitions, 0);
        assert_eq!(
            snapshot.goal,
            FrontierGoalStatus::Stopped(FrontierStop::WorkBudget)
        );
    }
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), Some(10), false);
    let first = fixture.observe(&[mv("e2e4"), mv("e7e5")]).unwrap();
    let second = fixture.observe(&[mv("e2e4"), mv("e7e5")]).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        fixture.frontier.snapshot().work.total_node_visits(),
        Some(3)
    );
    assert_eq!(fixture.frontier.remaining_depth(first).unwrap(), None);
    assert_eq!(fixture.frontier.snapshot().horizon.declared_obligations, 0);
}

#[test]
fn unexamined_mate_defenses_are_returned_as_questions_and_depth_does_not_shrink_mate() {
    let mut root = Position::from_fen("6k1/p7/5K2/8/8/8/P7/5Q2 w - - 0 1").unwrap();
    for movement in ["a2a3", "a7a6", "f1g1"] {
        root.make_uci(movement).unwrap();
    }
    let mut fixture = Fixture::new(
        root,
        SearchGoal {
            depth_plies: NonZeroU16::new(1),
            mate_moves: NonZeroU16::new(2),
        },
        None,
        false,
    );
    fixture.observe(&[mv("g8h8")]).unwrap();
    let questions = fixture
        .frontier
        .missing_mate_questions(64, fixture.clock.domain(), fixture.clock.now().unwrap())
        .unwrap();
    assert!(
        questions
            .iter()
            .all(|q| fixture.frontier.path_ply(q.anchor()).unwrap() == ChessPly(0))
    );
    assert!(
        questions
            .iter()
            .any(|q| q.kind() == MateQuestionKind::RequiredDefense)
    );
    for lane in fixture.frontier.mate.iter().flatten() {
        assert_eq!(
            lane.proof.root_conclusion().conclusion,
            MateConclusion::Unknown
        );
    }
    assert_eq!(
        fixture.frontier.observe_prefix(
            &[mv("g8h8"), mv("g1g7")],
            fixture.clock.domain(),
            fixture.clock.now().unwrap()
        ),
        Err(FrontierError::DepthHorizon)
    );
    assert_eq!(
        fixture.frontier.snapshot().horizon.maximum_examined_ply,
        ChessPly(1)
    );
    assert_eq!(
        fixture.frontier.snapshot().mate[0].unwrap().conclusion,
        MateConclusion::Unknown
    );
}

fn native_job(fixture: &Fixture) -> CpuJobEnvelope {
    CpuJobEnvelope {
        id: CpuJobId {
            authority: fixture.frontier.registration.authority,
            sequence: 1,
        },
        state_identity: fixture
            .frontier
            .registration
            .root_rules
            .snapshot()
            .identity(),
        question: Digest([30; 32]),
        root_restriction: Digest([31; 32]),
        evaluation_namespace: Digest([32; 32]),
        search_conditions: Digest([33; 32]),
        total_native_node_credit: 100,
        hard_deadline: fixture.frontier.original_deadline(),
    }
}
fn native_grant(job: CpuJobEnvelope, sequence: u64, credit: u64) -> SliceGrant {
    SliceGrant {
        job: job.id,
        execution: ExecutionId {
            epoch: job.id.authority.epoch,
            sequence,
        },
        owner: CpuWorkerId(0),
        node_credit: credit,
        soft_lease: Duration::from_millis(8),
        control_sequence: sequence,
    }
}
fn failed_native_report(grant: SliceGrant, nodes: u64, qnodes: u64) -> CpuSliceReport {
    CpuSliceReport {
        grant,
        reciprocal_request: None,
        actual_work: Some(CpuWork {
            nodes,
            quiescence_nodes: qnodes,
            tt_hits: 0,
        }),
        result: Err(CpuError::Allocation),
        cumulative_nodes: nodes,
        progress_sequence: 1,
        continuation_retained: false,
        terminal: true,
        admission_rejected: false,
        controls: None,
    }
}

#[test]
fn native_accounting_requires_the_original_ledger_receipt_and_charges_actual_once() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), Some(10), true);
    let job = native_job(&fixture);
    let mut ledger = CpuCreditLedger::new(job.id.authority, 100, fixture.clock).unwrap();
    fixture.frontier.registration.native_ledger = Some(ledger.identity());
    ledger.register(job).unwrap();
    assert_eq!(
        fixture
            .frontier
            .native_capacity(100, fixture.clock.domain(), fixture.clock.now().unwrap())
            .unwrap(),
        9
    );
    let grant = native_grant(job, 1, 9);
    ledger.reserve(grant).unwrap();
    let reservation = fixture
        .frontier
        .reserve_native(
            &job,
            grant,
            fixture.clock.domain(),
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    assert_eq!(
        fixture.observe(&[mv("e2e4")]),
        Err(FrontierError::NodesReserved)
    );
    let report = failed_native_report(grant, 5, 2);
    let settled = ledger.settle_attested(&report).unwrap();
    assert_eq!(
        fixture
            .frontier
            .settle_native(&reservation, &report, &settled)
            .unwrap(),
        NativeCloseReason::Executed
    );
    let snapshot = fixture.frontier.snapshot();
    assert_eq!(snapshot.work.total_node_visits(), Some(6));
    assert_eq!(snapshot.work.native_qsearch_nodes, 2);
    assert_eq!(snapshot.work.reserved_native_nodes, 0);
    assert_eq!(
        fixture
            .frontier
            .settle_native(&reservation, &report, &settled),
        Err(FrontierError::Duplicate)
    );
    assert_eq!(
        fixture
            .frontier
            .native_capacity(100, fixture.clock.domain(), fixture.clock.now().unwrap())
            .unwrap(),
        4
    );
}

#[test]
fn foreign_and_unknown_native_work_quarantine_without_refunding() {
    for foreign in [false, true] {
        let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), Some(10), true);
        let job = native_job(&fixture);
        let mut ledger = CpuCreditLedger::new(job.id.authority, 100, fixture.clock).unwrap();
        ledger.register(job).unwrap();
        let grant = native_grant(job, 1, 8);
        ledger.reserve(grant).unwrap();
        let reservation = fixture
            .frontier
            .reserve_native(
                &job,
                grant,
                fixture.clock.domain(),
                fixture.clock.now().unwrap(),
            )
            .unwrap();
        if foreign {
            let report = failed_native_report(grant, 3, 1);
            let settled = ledger.settle_attested(&report).unwrap();
            assert_eq!(
                fixture
                    .frontier
                    .settle_native(&reservation, &report, &settled),
                Err(FrontierError::NativeUnknown)
            );
        } else {
            fixture.frontier.native_usage_unknown(&reservation).unwrap();
        }
        let snapshot = fixture.frontier.snapshot();
        assert_eq!(snapshot.work.reserved_native_nodes, 8);
        assert_eq!(snapshot.work.native_nodes, 0);
        assert!(!snapshot.work.native_usage_known);
        assert_eq!(
            snapshot.goal,
            FrontierGoalStatus::Stopped(FrontierStop::NativeUnknown)
        );
    }
}

#[test]
fn attested_owner_zero_work_rejection_is_distinct_and_does_not_count_as_a_visit() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), Some(10), true);
    let job = native_job(&fixture);
    let mut ledger = CpuCreditLedger::new(job.id.authority, 100, fixture.clock).unwrap();
    fixture.frontier.registration.native_ledger = Some(ledger.identity());
    ledger.register(job).unwrap();
    let grant = native_grant(job, 1, 8);
    ledger.reserve(grant).unwrap();
    let reservation = fixture
        .frontier
        .reserve_native(
            &job,
            grant,
            fixture.clock.domain(),
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    let mut report = failed_native_report(grant, 0, 0);
    report.admission_rejected = true;
    report.terminal = false;
    let settled = ledger.settle_attested(&report).unwrap();
    assert_eq!(
        fixture
            .frontier
            .settle_native(&reservation, &report, &settled)
            .unwrap(),
        NativeCloseReason::OwnerRejectedBeforeWork
    );
    assert_eq!(
        fixture.frontier.snapshot().work.total_node_visits(),
        Some(1)
    );
    assert_eq!(fixture.frontier.snapshot().work.reserved_native_nodes, 0);
}

#[test]
fn actual_store_generation_expiry_and_foreign_root_assembly_fail_closed() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    let root_rules = Arc::clone(&fixture.frontier.registration.root_rules);
    let root_moves = Arc::clone(fixture.frontier.root_moves());
    let namespace = Arc::clone(&fixture.frontier.registration.model_namespace);
    let equivalent_domain = Arc::new(RootMoveDomain::new(&root_rules, None).unwrap());
    let attachment = FrontierAttachment {
        authority: fixture.frontier.registration.authority,
        mode: ExecutionMode::CpuPc,
        original_deadline: fixture.frontier.original_deadline(),
        root_budget: fixture.frontier.registration.root_budget,
        root_rules: &root_rules,
        root_moves: &root_moves,
        root_state: fixture.frontier.registration.root_state,
        model_namespace: &namespace,
        native_ledger: None,
    };
    fixture
        .frontier
        .validate_attachment(attachment, &fixture.stores)
        .unwrap();
    assert_eq!(
        fixture.frontier.validate_attachment(
            FrontierAttachment {
                root_moves: &equivalent_domain,
                ..attachment
            },
            &fixture.stores
        ),
        Err(FrontierError::ForeignRoot)
    );
    assert_eq!(fixture.stores.root(), Some(fixture.root_situation));
    fixture
        .stores
        .hot_focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let line = fixture.line(&[mv("e2e4")]);
    assert_eq!(
        fixture.frontier.observe_stored_line(
            &fixture.stores,
            line,
            fixture.clock.domain(),
            fixture.clock.now().unwrap()
        ),
        Err(FrontierError::StoreUnavailable)
    );
    assert_eq!(
        fixture.frontier.snapshot().goal,
        FrontierGoalStatus::Stopped(FrontierStop::EvidenceReclaimed)
    );
}
