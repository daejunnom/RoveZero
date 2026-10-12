//! Exact-credit and private-worker regressions. Native own-CPU execution is
//! bounded by the immutable ten-second fixture Job, not a benchmark window.

use super::*;
use crate::cpu::{CpuConfig, CpuOrderingPolicy, CpuProfile, CpuScoreScope, CpuYieldCause};
use crate::cpu_value::{
    BootstrapCpuValue, CpuAccumulator, CpuAccumulatorUndo, CpuValueError, CpuValueEvaluator,
    CpuValueIdentity,
};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::{ClockDomain, GameGeneration, OwnerId, ProcessEpoch, RootGeneration};
use rz_position::RuleMoveDelta;

const FIXTURE_WINDOW: Duration = Duration::from_secs(10);

fn engine(ordering: CpuOrderingPolicy, tt_entries: usize) -> CpuEngine {
    CpuEngine::with_evaluator_ordering_and_resume_policy(
        CpuConfig {
            profile: CpuProfile::PlanAssisted,
            tt_entries,
            max_depth: 2,
            quiescence_ply: 4,
        },
        Arc::new(BootstrapCpuValue::default()),
        ordering,
        CpuResumePolicy::PausedStack,
    )
    .unwrap()
}

fn clock() -> InstantClock {
    InstantClock::new(ClockDomain(ProcessEpoch(41)), Instant::now())
}

fn envelope(
    cpu: &CpuEngine,
    position: &ContractPosition,
    restriction: Option<&[BoardMove]>,
    clock: &InstantClock,
    total_credit: u64,
) -> CpuJobEnvelope {
    CpuJobEnvelope {
        id: CpuJobId {
            authority: SearchAuthority {
                epoch: clock.domain().0,
                game: GameGeneration(1),
                root: RootGeneration(1),
                implementation: Digest([9; 32]),
            },
            sequence: 1,
        },
        state_identity: position.export().unwrap().snapshot().identity(),
        question: Digest([7; 32]),
        root_restriction: cpu_root_restriction_digest(restriction),
        evaluation_namespace: cpu_evaluation_namespace_digest(cpu).unwrap(),
        search_conditions: cpu_search_conditions_digest(cpu),
        total_native_node_credit: total_credit,
        hard_deadline: clock.deadline_at(clock.origin() + FIXTURE_WINDOW).unwrap(),
    }
}

fn job(
    cpu: &CpuEngine,
    position: Position,
    restriction: Option<Vec<BoardMove>>,
    clock: &InstantClock,
    owner: CpuWorkerId,
    total_credit: u64,
) -> CpuJob {
    let position = ContractPosition::new(OwnerId(71), position);
    let envelope = envelope(cpu, &position, restriction.as_deref(), clock, total_credit);
    CpuJob::new(
        envelope,
        position,
        restriction,
        2,
        clock,
        cpu,
        owner,
        CompletionSignal::default(),
    )
    .unwrap()
}

fn grant(job: CpuJobEnvelope, owner: CpuWorkerId, sequence: u64, credit: u64) -> SliceGrant {
    SliceGrant {
        job: job.id,
        execution: ExecutionId {
            epoch: job.id.authority.epoch,
            sequence,
        },
        owner,
        node_credit: credit,
        soft_lease: Duration::from_millis(100),
        control_sequence: sequence,
    }
}

fn ledger(envelope: CpuJobEnvelope, clock: InstantClock) -> CpuCreditLedger {
    CpuCreditLedger::new(
        envelope.id.authority,
        envelope.total_native_node_credit,
        clock,
    )
    .unwrap()
}

fn wait_event(worker: &CpuWorker, deadline: Instant) -> CpuWorkerEvent {
    loop {
        // Observe before checking the queue: publication in this gap changes
        // the version and cannot turn the wait into a lost wakeup.
        let observed = worker.signal.version();
        if let Some(report) = worker.try_event().unwrap() {
            return report;
        }
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .expect("CPU fixture failed to report before its immutable deadline");
        worker
            .signal
            .wait_changed(observed, remaining.min(Duration::from_millis(10)));
    }
}

fn wait_report(worker: &CpuWorker, deadline: Instant) -> CpuSliceReport {
    match wait_event(worker, deadline) {
        CpuWorkerEvent::Slice(report) => report,
        CpuWorkerEvent::JobTerminated(termination) => {
            panic!("unexpected idle termination during a live slice: {termination:?}")
        }
    }
}

fn wait_reports_queued(worker: &CpuWorker, count: usize, deadline: Instant) {
    loop {
        let observed = worker.signal.version();
        if worker.mailbox.occupancy().reliable_queued >= count {
            return;
        }
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .expect("CPU fixture did not publish its reserved report seats");
        worker
            .signal
            .wait_changed(observed, remaining.min(Duration::from_millis(10)));
    }
}

fn observed_report(
    grant: SliceGrant,
    nodes: u64,
    cumulative: u64,
    terminal: bool,
) -> CpuSliceReport {
    CpuSliceReport {
        grant,
        actual_work: Some(CpuWork {
            nodes,
            quiescence_nodes: 0,
            tt_hits: 0,
        }),
        // This authored ledger fixture validates arithmetic and report shape.
        // Native work is separately checked by the actual worker tests below.
        result: Ok(CpuControlledReport {
            report: crate::cpu::CpuReport {
                best_move: None,
                pv: Vec::new(),
                score: 0,
                completed_depth: 0,
                nodes,
                quiescence_nodes: 0,
                tt_hits: 0,
                completion: if terminal {
                    CpuCompletion::NodeLimit
                } else {
                    CpuCompletion::Yielded
                },
                score_scope: CpuScoreScope::FrontierOnly,
                profile: CpuProfile::PlanAssisted,
                score_provenance: "authored-ledger-fixture",
                value_identity: BootstrapCpuValue::default().identity().clone(),
                search_version: crate::cpu::CPU_PAUSED_STACK_SEARCH_VERSION,
                root_restricted: false,
                elapsed: Duration::ZERO,
                reused_completed_depth: 0,
                resume: None,
            },
            yield_cause: (!terminal).then_some(CpuYieldCause::NodeCredit),
            yield_ack_sequence: None,
            publish_ack_sequence: 0,
        }),
        cumulative_nodes: cumulative,
        progress_sequence: 0,
        continuation_retained: !terminal,
        terminal,
        admission_rejected: false,
        controls: None,
    }
}

#[test]
fn native_private_worker_slices_settle_once_and_match_uninterrupted_work() {
    for ordering in [
        CpuOrderingPolicy::LegacyMvvLvaV1,
        CpuOrderingPolicy::LegalSeeV1,
    ] {
        for credit in [1, 3, 13] {
            let position = Position::from_fen("3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
            let restriction = vec![BoardMove::from_uci("e4d5").unwrap()];
            let mut direct = engine(ordering, 16);
            let expected = direct
                .analyze_root_moves(
                    &position,
                    &restriction,
                    CpuLimits {
                        max_depth: 2,
                        max_nodes: 100_000,
                        deadline: None,
                    },
                    &AtomicBool::new(false),
                )
                .unwrap();
            assert_eq!(expected.completion, CpuCompletion::DepthLimit);
            let cpu = engine(ordering, 16);
            let clock = clock();
            let owner = CpuWorkerId(0);
            let job = job(&cpu, position, Some(restriction), &clock, owner, 100_000);
            let envelope = job.envelope();
            let controls = job.mailbox.clone();
            let worker = CpuWorker::spawn(owner, cpu).unwrap();
            let mut ledger = ledger(envelope, clock);
            ledger.register(envelope).unwrap();
            let first = grant(envelope, owner, 1, credit);
            ledger.reserve(first).unwrap();
            worker.begin(job, first).unwrap();
            let mut sequence = 1;
            let mut accumulated = CpuWork::default();
            loop {
                let report = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
                assert_eq!(report.grant.owner, owner);
                assert_eq!(report.grant.execution.sequence, sequence);
                let work = report
                    .actual_work
                    .expect("normal own worker observes a delta");
                accumulated.nodes += work.nodes;
                accumulated.quiescence_nodes += work.quiescence_nodes;
                accumulated.tt_hits += work.tt_hits;
                assert!(work.nodes <= credit);
                assert_eq!(report.cumulative_nodes, accumulated.nodes);
                ledger.settle(&report).unwrap();
                assert_eq!(
                    ledger.usage(envelope.id).unwrap(),
                    (accumulated.nodes, 0, false)
                );
                let actual = report.result.as_ref().unwrap();
                // Opaque continuation ownership is private to the worker. A
                // caller can request another grant but cannot copy its token.
                assert!(actual.report.resume.is_none());
                assert_eq!(
                    (
                        actual.report.nodes,
                        actual.report.quiescence_nodes,
                        actual.report.tt_hits
                    ),
                    (work.nodes, work.quiescence_nodes, work.tt_hits)
                );
                assert_eq!(ledger.settle(&report), Err(CpuWorkerError::Duplicate));
                if report.terminal {
                    assert!(!report.continuation_retained);
                    assert_eq!(actual.report.completion, CpuCompletion::DepthLimit);
                    assert_eq!(
                        (
                            actual.report.completed_depth,
                            actual.report.score,
                            &actual.report.pv
                        ),
                        (expected.completed_depth, expected.score, &expected.pv)
                    );
                    break;
                }
                assert!(report.continuation_retained);
                assert_eq!(actual.report.completion, CpuCompletion::Yielded);
                assert!(matches!(
                    actual.yield_cause,
                    Some(CpuYieldCause::NodeCredit | CpuYieldCause::SoftLease)
                ));
                sequence += 1;
                assert!(sequence < 100_000);
                let next = grant(envelope, owner, sequence, credit);
                ledger.reserve(next).unwrap();
                worker.resume(next).unwrap();
            }
            assert_eq!(accumulated, direct.last_attempt_work().unwrap());
            assert!(controls.snapshot().terminal);
            if let Some(progress) = worker.latest_progress() {
                assert_eq!(progress.job, envelope.id);
                assert_eq!(progress.owner, owner);
                assert!(progress.sequence > 0);
                assert!(progress.iteration.completed_depth > 0);
                assert!(progress.iteration.nodes <= credit);
            }
            worker.shutdown().unwrap();
        }
    }
}

#[test]
fn ledger_rejects_overreservation_duplicate_and_owner_migration_without_refund() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let clock = clock();
    let job = job(&cpu, Position::startpos(), None, &clock, CpuWorkerId(0), 5);
    let envelope = job.envelope();
    let mut ledger = ledger(envelope, clock);
    ledger.register(envelope).unwrap();
    assert_eq!(ledger.register(envelope), Err(CpuWorkerError::Duplicate));
    assert!(matches!(
        ledger.reserve(grant(envelope, CpuWorkerId(0), 1, 6)),
        Err(CpuWorkerError::Invalid(_))
    ));
    assert_eq!(ledger.usage(envelope.id).unwrap(), (0, 0, false));
    let first = grant(envelope, CpuWorkerId(0), 1, 4);
    ledger.reserve(first).unwrap();
    assert_eq!(ledger.usage(envelope.id).unwrap(), (0, 4, false));
    assert_eq!(
        ledger.reserve(grant(envelope, CpuWorkerId(0), 2, 1)),
        Err(CpuWorkerError::AlreadyReserved)
    );
    let first_report = observed_report(first, 2, 2, false);
    ledger.settle(&first_report).unwrap();
    assert_eq!(ledger.usage(envelope.id).unwrap(), (2, 0, false));
    assert_eq!(ledger.settle(&first_report), Err(CpuWorkerError::Duplicate));
    assert!(matches!(
        ledger.reserve(grant(envelope, CpuWorkerId(1), 2, 1)),
        Err(CpuWorkerError::Invalid(_))
    ));
    assert!(matches!(
        ledger.reserve(grant(envelope, CpuWorkerId(0), 2, 4)),
        Err(CpuWorkerError::Invalid(_))
    ));
    assert_eq!(ledger.reserve(first), Err(CpuWorkerError::Duplicate));
    assert_eq!(ledger.usage(envelope.id).unwrap(), (2, 0, false));
    let last = grant(envelope, CpuWorkerId(0), 2, 3);
    ledger.reserve(last).unwrap();
    ledger.settle(&observed_report(last, 3, 5, true)).unwrap();
    assert_eq!(ledger.usage(envelope.id).unwrap(), (5, 0, false));
    assert!(matches!(
        ledger.reserve(grant(envelope, CpuWorkerId(0), 3, 1)),
        Err(CpuWorkerError::Invalid(_))
    ));
}

#[test]
fn unknown_usage_and_inconsistent_cumulative_usage_quarantine_reserved_credit() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let clock = clock();
    let envelope = job(&cpu, Position::startpos(), None, &clock, CpuWorkerId(0), 10).envelope();
    for unknown in [true, false] {
        let mut ledger = ledger(envelope, clock);
        ledger.register(envelope).unwrap();
        let first = grant(envelope, CpuWorkerId(0), 1, 4);
        ledger.reserve(first).unwrap();
        let mut report = observed_report(first, 1, 1, true);
        if unknown {
            report.actual_work = None;
            assert_eq!(ledger.settle(&report), Err(CpuWorkerError::UnknownUsage));
        } else {
            report.cumulative_nodes = 0;
            assert!(matches!(
                ledger.settle(&report),
                Err(CpuWorkerError::Invalid(_))
            ));
        }
        assert_eq!(ledger.usage(envelope.id).unwrap(), (0, 4, true));
        assert_eq!(ledger.settle(&report), Err(CpuWorkerError::Quarantined));
        assert_eq!(
            ledger.reserve(grant(envelope, CpuWorkerId(0), 2, 1)),
            Err(CpuWorkerError::Quarantined)
        );
    }
}

#[test]
fn ledger_rejects_a_report_from_an_unreserved_execution_without_changing_usage() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let clock = clock();
    let envelope = job(&cpu, Position::startpos(), None, &clock, CpuWorkerId(0), 10).envelope();
    let mut ledger = ledger(envelope, clock);
    ledger.register(envelope).unwrap();
    let reserved = grant(envelope, CpuWorkerId(0), 1, 4);
    ledger.reserve(reserved).unwrap();
    let other = grant(envelope, CpuWorkerId(0), 2, 4);
    assert_eq!(
        ledger.settle(&observed_report(other, 1, 1, false)),
        Err(CpuWorkerError::Duplicate)
    );
    assert_eq!(ledger.usage(envelope.id).unwrap(), (0, 4, false));
    ledger
        .settle(&observed_report(reserved, 1, 1, false))
        .unwrap();
    assert_eq!(ledger.usage(envelope.id).unwrap(), (1, 0, false));
}

#[test]
fn global_credit_covers_all_jobs_and_one_owner_cannot_hold_two_primaries() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let clock = clock();
    let first = job(&cpu, Position::startpos(), None, &clock, CpuWorkerId(0), 10).envelope();
    let mut second = first;
    second.id.sequence = 2;
    second.question = Digest([8; 32]);
    let mut ledger = ledger(first, clock);
    ledger.register(first).unwrap();
    ledger.register(second).unwrap();
    let first_grant = grant(first, CpuWorkerId(0), 1, 6);
    ledger.reserve(first_grant).unwrap();
    ledger
        .settle(&observed_report(first_grant, 4, 4, false))
        .unwrap();
    assert_eq!(ledger.global_usage().unwrap(), (4, 0));
    assert!(matches!(
        ledger.reserve(grant(second, CpuWorkerId(0), 2, 1)),
        Err(CpuWorkerError::Invalid(_))
    ));
    let second_grant = grant(second, CpuWorkerId(1), 2, 6);
    ledger.reserve(second_grant).unwrap();
    assert_eq!(ledger.global_usage().unwrap(), (4, 6));
    assert!(matches!(
        ledger.reserve(grant(first, CpuWorkerId(0), 3, 1)),
        Err(CpuWorkerError::Invalid(_))
    ));
    assert_eq!(ledger.global_usage().unwrap(), (4, 6));
    ledger
        .settle(&observed_report(second_grant, 6, 6, true))
        .unwrap();
    assert_eq!(ledger.global_usage().unwrap(), (10, 0));
    assert!(matches!(
        ledger.reserve(grant(first, CpuWorkerId(0), 3, 1)),
        Err(CpuWorkerError::Invalid(_))
    ));
}

#[test]
fn root_ledger_rejects_stale_authority_expiry_lease_and_control_sequences() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let clock = clock();
    let envelope = job(&cpu, Position::startpos(), None, &clock, CpuWorkerId(0), 10).envelope();
    assert!(CpuCreditLedger::new(envelope.id.authority, 0, clock).is_err());
    assert!(
        CpuCreditLedger::new(
            envelope.id.authority,
            10,
            InstantClock::new(ClockDomain(ProcessEpoch(42)), Instant::now())
        )
        .is_err()
    );
    let mut ledger = ledger(envelope, clock);
    let mut changed = envelope;
    changed.id.authority.root = RootGeneration(2);
    assert!(ledger.register(changed).is_err());
    let mut expired = envelope;
    expired.hard_deadline.at = clock.now().unwrap();
    assert!(matches!(
        ledger.register(expired),
        Err(CpuWorkerError::Invalid(_))
    ));
    ledger.register(envelope).unwrap();
    let mut long_lease = grant(envelope, CpuWorkerId(0), 1, 1);
    long_lease.soft_lease = Duration::from_secs(60);
    assert!(matches!(
        ledger.reserve(long_lease),
        Err(CpuWorkerError::Invalid(_))
    ));
    let first = grant(envelope, CpuWorkerId(0), 1, 1);
    ledger.reserve(first).unwrap();
    ledger.settle(&observed_report(first, 1, 1, false)).unwrap();
    let mut stale_control = grant(envelope, CpuWorkerId(0), 2, 1);
    stale_control.control_sequence = 1;
    assert!(matches!(
        ledger.reserve(stale_control),
        Err(CpuWorkerError::Invalid(_))
    ));
    assert_eq!(ledger.usage(envelope.id).unwrap(), (1, 0, false));
}

#[test]
fn immutable_job_validation_rejects_namespace_history_restriction_and_clock_mismatch() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 16);
    let clock = clock();
    let position = ContractPosition::new(OwnerId(71), Position::startpos());
    let original = envelope(&cpu, &position, None, &clock, 100);
    for mutation in 0..6 {
        let mut invalid = original;
        match mutation {
            0 => invalid.state_identity.revision = rz_contracts::StateRevision(100),
            1 => invalid.root_restriction = Digest([2; 32]),
            2 => invalid.evaluation_namespace = Digest([3; 32]),
            3 => invalid.search_conditions = Digest([4; 32]),
            4 => invalid.hard_deadline.clock = ClockDomain(ProcessEpoch(42)),
            _ => invalid.total_native_node_credit = 0,
        }
        let fresh = ContractPosition::new(OwnerId(71), Position::startpos());
        assert!(matches!(
            CpuJob::new(
                invalid,
                fresh,
                None,
                2,
                &clock,
                &cpu,
                CpuWorkerId(0),
                CompletionSignal::default()
            ),
            Err(CpuWorkerError::Invalid(_))
        ));
    }
    let illegal = vec![BoardMove::from_uci("e2e5").unwrap()];
    let mut illegal_envelope = original;
    illegal_envelope.root_restriction = cpu_root_restriction_digest(Some(&illegal));
    assert!(matches!(
        CpuJob::new(
            illegal_envelope,
            ContractPosition::new(OwnerId(71), Position::startpos()),
            Some(illegal),
            2,
            &clock,
            &cpu,
            CpuWorkerId(0),
            CompletionSignal::default()
        ),
        Err(CpuWorkerError::Invalid(_))
    ));
}

#[test]
fn worker_rejects_foreign_owner_before_admission_and_keeps_original_job_private() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let clock = clock();
    let foreign_job = job(
        &cpu,
        Position::startpos(),
        None,
        &clock,
        CpuWorkerId(1),
        100,
    );
    let foreign_envelope = foreign_job.envelope();
    let foreign_controls = foreign_job.mailbox.clone();
    let worker = CpuWorker::spawn(CpuWorkerId(0), cpu).unwrap();
    assert!(matches!(
        worker.begin(foreign_job, grant(foreign_envelope, CpuWorkerId(1), 1, 1)),
        Err(CpuWorkerError::Invalid(_))
    ));
    assert!(worker.try_event().unwrap().is_none());
    assert_eq!(worker.mailbox.occupancy().reliable_reserved, 0);
    assert_eq!(foreign_controls.snapshot().yield_ack, 0);
    worker.shutdown().unwrap();
}

#[test]
fn second_begin_cannot_replace_or_run_a_paused_immutable_job() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let clock = clock();
    let original = job(
        &cpu,
        Position::startpos(),
        None,
        &clock,
        CpuWorkerId(0),
        100,
    );
    let envelope = original.envelope();
    let mut conflicting = job(
        &cpu,
        Position::startpos(),
        None,
        &clock,
        CpuWorkerId(0),
        100,
    );
    conflicting.envelope.question = Digest([99; 32]);
    let worker = CpuWorker::spawn(CpuWorkerId(0), cpu).unwrap();
    worker
        .begin(original, grant(envelope, CpuWorkerId(0), 1, 1))
        .unwrap();
    let first = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
    assert!(first.continuation_retained && !first.terminal);
    let attempted = worker.begin(conflicting, grant(envelope, CpuWorkerId(0), 2, 1));
    match attempted {
        Err(CpuWorkerError::Invalid(_)) => {}
        Ok(()) => {
            let refused = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
            assert!(refused.admission_rejected && !refused.terminal);
            assert!(refused.result.is_err());
            assert_eq!(refused.actual_work, Some(CpuWork::default()));
            assert_eq!(refused.cumulative_nodes, first.cumulative_nodes);
        }
        other => panic!("second Begin had unexpected admission status: {other:?}"),
    }
    worker
        .resume(grant(envelope, CpuWorkerId(0), 3, 1))
        .unwrap();
    let resumed = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
    assert!(resumed.result.is_ok());
    assert!(resumed.continuation_retained && !resumed.terminal);
    assert_eq!(resumed.cumulative_nodes, first.cumulative_nodes + 1);
    worker.shutdown().unwrap();
}

#[test]
fn paused_idle_expiry_has_its_own_terminal_receipt_and_does_not_recharge_nodes() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    // A ten-second owned Job whose first nine seconds preceded this test's
    // admission. The remaining second expires naturally; no deadline mutation
    // or fabricated worker report stands in for the real idle expiry path.
    let clock = InstantClock::new(
        ClockDomain(ProcessEpoch(41)),
        Instant::now().checked_sub(Duration::from_secs(9)).unwrap(),
    );
    let original = job(
        &cpu,
        Position::startpos(),
        None,
        &clock,
        CpuWorkerId(0),
        100,
    );
    let envelope = original.envelope();
    let controls = original.mailbox.clone();
    let first = grant(envelope, CpuWorkerId(0), 1, 1);
    let mut ledger = ledger(envelope, clock);
    ledger.register(envelope).unwrap();
    ledger.reserve(first).unwrap();
    let worker = CpuWorker::spawn(CpuWorkerId(0), cpu).unwrap();
    worker.begin(original, first).unwrap();
    let yielded = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
    assert!(yielded.continuation_retained && !yielded.terminal);
    ledger.settle(&yielded).unwrap();
    let before = ledger.global_usage().unwrap();
    // Cleanup observation has a separate finite reserve after the unchanged
    // hard deadline, so a poll near expiry can still receive the terminal ack.
    let termination = match wait_event(
        &worker,
        clock.origin() + FIXTURE_WINDOW + Duration::from_secs(1),
    ) {
        CpuWorkerEvent::JobTerminated(termination) => termination,
        CpuWorkerEvent::Slice(_) => panic!("idle expiry unexpectedly minted another slice"),
    };
    assert_eq!(termination.completion, CpuCompletion::Deadline);
    assert_eq!(termination.last_execution, first.execution);
    assert_eq!(termination.retained_bytes, 0);
    assert!(termination.controls.terminal);
    assert_eq!(termination.controls.cancel_sequence, 0);
    ledger.accept_job_termination(&termination).unwrap();
    assert_eq!(ledger.global_usage().unwrap(), before);
    assert_eq!(ledger.usage(envelope.id).unwrap(), (1, 0, false));
    assert!(controls.snapshot().terminal);
    worker.shutdown().unwrap();
}

#[test]
fn full_report_queue_does_not_block_atomic_cancellation_or_drop_reserved_reports() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let clock = clock();
    let original = job(
        &cpu,
        Position::startpos(),
        None,
        &clock,
        CpuWorkerId(0),
        100,
    );
    let envelope = original.envelope();
    let controls = original.mailbox.clone();
    let worker = CpuWorker::spawn(CpuWorkerId(0), cpu).unwrap();
    worker
        .begin(original, grant(envelope, CpuWorkerId(0), 1, 1))
        .unwrap();
    wait_reports_queued(&worker, 1, clock.origin() + FIXTURE_WINDOW);
    let mut last_sequence = 1;
    for sequence in 2..=32 {
        match worker.resume(grant(envelope, CpuWorkerId(0), sequence, 1)) {
            Ok(()) => {
                last_sequence = sequence;
                wait_reports_queued(&worker, sequence as usize, clock.origin() + FIXTURE_WINDOW);
            }
            Err(CpuWorkerError::Capacity) => break,
            other => panic!("report-seat saturation had unexpected status: {other:?}"),
        }
    }
    assert!(last_sequence >= CPU_WORKER_COMMAND_CAPACITY as u64);
    assert!(
        last_sequence < 32,
        "bounded report capacity was not enforced"
    );
    assert_eq!(
        worker.resume(grant(envelope, CpuWorkerId(0), last_sequence + 1, 1)),
        Err(CpuWorkerError::Capacity)
    );
    // The atomic mailbox has no dependency on a command/reliable queue seat.
    controls.request_cancel(100).unwrap();
    assert_eq!(controls.snapshot().cancel_sequence, 100);
    for sequence in 1..=last_sequence {
        let report = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
        assert_eq!(report.grant.execution.sequence, sequence);
        assert_eq!(report.actual_work.unwrap().nodes, 1);
    }
    // The paused owner's reserved terminal seat survives even when ordinary
    // admission is full. No Resume command is needed to read cancellation.
    let termination = match wait_event(&worker, clock.origin() + FIXTURE_WINDOW) {
        CpuWorkerEvent::JobTerminated(termination) => termination,
        CpuWorkerEvent::Slice(_) => panic!("idle cancel was incorrectly charged as a new slice"),
    };
    assert_eq!(termination.job, envelope.id);
    assert_eq!(termination.owner, CpuWorkerId(0));
    assert_eq!(termination.last_execution.sequence, last_sequence);
    assert_eq!(termination.completion, CpuCompletion::Canceled);
    assert_eq!(termination.retained_bytes, 0);
    assert!(controls.snapshot().terminal);
    assert_eq!(controls.snapshot().cancel_ack, 100);
    worker.shutdown().unwrap();
}

struct PanicDuringDelta(BootstrapCpuValue);
impl CpuValueEvaluator for PanicDuringDelta {
    fn identity(&self) -> &CpuValueIdentity {
        self.0.identity()
    }
    fn provenance(&self) -> &'static str {
        self.0.provenance()
    }
    fn initialize(&self, position: &Position) -> Result<CpuAccumulator, CpuValueError> {
        self.0.initialize(position)
    }
    fn score(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
    ) -> Result<i32, CpuValueError> {
        self.0.score(accumulator, position)
    }
    fn apply_delta(
        &self,
        _: &mut CpuAccumulator,
        _: &RuleMoveDelta,
    ) -> Result<CpuAccumulatorUndo, CpuValueError> {
        panic!("worker failure after a real entered root; actual delta is unreturned")
    }
}

#[test]
fn native_worker_panic_preserves_unknown_usage_and_quarantines_the_live_reservation() {
    let cpu = CpuEngine::with_evaluator_ordering_and_resume_policy(
        CpuConfig {
            max_depth: 2,
            quiescence_ply: 4,
            tt_entries: 0,
            ..CpuConfig::default()
        },
        Arc::new(PanicDuringDelta(BootstrapCpuValue::default())),
        CpuOrderingPolicy::LegacyMvvLvaV1,
        CpuResumePolicy::PausedStack,
    )
    .unwrap();
    let clock = clock();
    let original = job(
        &cpu,
        Position::startpos(),
        None,
        &clock,
        CpuWorkerId(0),
        100,
    );
    let envelope = original.envelope();
    let controls = original.mailbox.clone();
    let first = grant(envelope, CpuWorkerId(0), 1, 16);
    let mut ledger = ledger(envelope, clock);
    ledger.register(envelope).unwrap();
    ledger.reserve(first).unwrap();
    let worker = CpuWorker::spawn(CpuWorkerId(0), cpu).unwrap();
    worker.begin(original, first).unwrap();
    let failed = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
    assert!(failed.terminal && !failed.continuation_retained);
    assert!(failed.result.is_err());
    assert_eq!(failed.actual_work, None);
    assert_eq!(ledger.settle(&failed), Err(CpuWorkerError::UnknownUsage));
    assert_eq!(ledger.usage(envelope.id).unwrap(), (0, 16, true));
    assert!(controls.snapshot().terminal);
    assert!(matches!(
        worker.resume(grant(envelope, CpuWorkerId(0), 2, 1)),
        Err(CpuWorkerError::Disconnected)
    ));
    worker.shutdown().unwrap();
}

#[test]
fn shutdown_seals_paused_job_and_joins_without_exposing_its_token() {
    let cpu = engine(CpuOrderingPolicy::LegalSeeV1, 16);
    let clock = clock();
    let original = job(
        &cpu,
        Position::startpos(),
        None,
        &clock,
        CpuWorkerId(0),
        100,
    );
    let envelope = original.envelope();
    let controls = original.mailbox.clone();
    let first = grant(envelope, CpuWorkerId(0), 1, 1);
    let mut ledger = ledger(envelope, clock);
    ledger.register(envelope).unwrap();
    ledger.reserve(first).unwrap();
    let worker = CpuWorker::spawn(CpuWorkerId(0), cpu).unwrap();
    worker.begin(original, first).unwrap();
    let paused = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
    assert!(paused.continuation_retained && !paused.terminal);
    assert!(paused.result.as_ref().unwrap().report.resume.is_none());
    ledger.settle(&paused).unwrap();
    let before = ledger.global_usage().unwrap();
    let stopped = worker.shutdown().unwrap();
    assert_eq!(stopped.owner, CpuWorkerId(0));
    assert!(stopped.thread_joined && !stopped.thread_panicked);
    assert!(stopped.mailbox.closed);
    assert_eq!(stopped.mailbox.commands, 0);
    assert_eq!(stopped.mailbox.reliable_queued, 0);
    assert_eq!(stopped.mailbox.reliable_reserved, 0);
    assert_eq!(stopped.events.len(), 1);
    let termination = match &stopped.events[0] {
        CpuWorkerEvent::JobTerminated(termination) => termination,
        CpuWorkerEvent::Slice(_) => panic!("paused shutdown should not invent another slice"),
    };
    assert_eq!(termination.job, envelope.id);
    assert_eq!(termination.owner, CpuWorkerId(0));
    assert_eq!(termination.last_execution, first.execution);
    assert_eq!(termination.completion, CpuCompletion::Canceled);
    assert!(termination.shutdown && termination.controls.terminal);
    assert_eq!(termination.retained_bytes, 0);
    ledger.accept_job_termination(termination).unwrap();
    assert_eq!(ledger.global_usage().unwrap(), before);
    assert_eq!(ledger.usage(envelope.id).unwrap(), (1, 0, false));
    assert_eq!(
        ledger.accept_job_termination(termination),
        Err(CpuWorkerError::Duplicate)
    );
    assert!(controls.snapshot().terminal);
    assert!(controls.request_yield(1).is_err());
    let mut successor = envelope;
    successor.id.sequence = 2;
    ledger.register(successor).unwrap();
    // The completed owner can be assigned again without dropping spent work.
    ledger
        .reserve(grant(successor, CpuWorkerId(0), 2, 1))
        .unwrap();
    assert_eq!(ledger.global_usage().unwrap(), (before.0, 1));
}

#[test]
fn first_begin_rejection_releases_owner_but_rejected_resume_preserves_primary() {
    for previously_started in [false, true] {
        let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let clock = clock();
        let position = ContractPosition::new(OwnerId(71), Position::startpos());
        let first_job = envelope(&cpu, &position, None, &clock, 10);
        let mut second_job = first_job;
        second_job.id.sequence = 2;
        second_job.question = Digest([8; 32]);
        let owner = CpuWorkerId(0);
        let mut ledger = ledger(first_job, clock);
        ledger.register(first_job).unwrap();
        ledger.register(second_job).unwrap();

        let first = grant(first_job, owner, 1, 2);
        ledger.reserve(first).unwrap();
        let (rejected_grant, spent) = if previously_started {
            ledger.settle(&observed_report(first, 1, 1, false)).unwrap();
            let resume = grant(first_job, owner, 2, 2);
            ledger.reserve(resume).unwrap();
            (resume, 1)
        } else {
            (first, 0)
        };
        // This is an authored exact-admission receipt, not native work evidence.
        // A rejected command has no delta and cannot destroy an older token.
        // The rejecting owner may no longer observe the old cumulative snapshot.
        let mut rejection = observed_report(rejected_grant, 0, 0, false);
        rejection.admission_rejected = true;
        rejection.continuation_retained = false;
        rejection.result = Err(CpuError::InvalidLimits("command rejected before execution"));
        ledger.settle(&rejection).unwrap();
        assert_eq!(ledger.usage(first_job.id).unwrap(), (spent, 0, false));
        assert_eq!(ledger.global_usage().unwrap(), (spent, 0));
        assert_eq!(ledger.settle(&rejection), Err(CpuWorkerError::Duplicate));

        let next_sequence = rejected_grant.execution.sequence + 1;
        let other_job = grant(second_job, owner, next_sequence, 1);
        if previously_started {
            assert!(matches!(
                ledger.retire_unstarted(first_job.id),
                Err(CpuWorkerError::Invalid(_))
            ));
            assert!(matches!(
                ledger.reserve(other_job),
                Err(CpuWorkerError::Invalid(_))
            ));
            assert_eq!(ledger.global_usage().unwrap(), (1, 0));
            // Rejected Resume retains the same Job's private owner authority.
            let resumed = grant(first_job, owner, next_sequence, 1);
            ledger.reserve(resumed).unwrap();
            ledger
                .settle(&observed_report(resumed, 1, 2, true))
                .unwrap();
            assert_eq!(ledger.global_usage().unwrap(), (2, 0));
        } else {
            // Rejected first Begin never owned a stack, so another Job can run.
            ledger.retire_unstarted(first_job.id).unwrap();
            let mut newly_registered = second_job;
            newly_registered.id.sequence = 3;
            ledger.register(newly_registered).unwrap();
            assert_eq!(ledger.register(first_job), Err(CpuWorkerError::Duplicate));
            assert_eq!(
                ledger.reserve(grant(first_job, owner, next_sequence, 1)),
                Err(CpuWorkerError::UnknownJob)
            );
            ledger.reserve(other_job).unwrap();
            ledger
                .settle(&observed_report(other_job, 1, 1, true))
                .unwrap();
            assert_eq!(ledger.global_usage().unwrap(), (1, 0));
        }
    }
}

#[test]
fn accepting_the_same_paused_job_termination_twice_is_duplicate_without_recharging() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let clock = clock();
    let position = ContractPosition::new(OwnerId(71), Position::startpos());
    let envelope = envelope(&cpu, &position, None, &clock, 10);
    let owner = CpuWorkerId(0);
    let mut ledger = ledger(envelope, clock);
    ledger.register(envelope).unwrap();
    let first = grant(envelope, owner, 1, 2);
    ledger.reserve(first).unwrap();
    ledger.settle(&observed_report(first, 1, 1, false)).unwrap();
    let before = ledger.global_usage().unwrap();

    // Exercise the real mailbox's sequenced ack; the native execution delta
    // remains an authored ledger fixture and is not charged at idle termination.
    let controls = CpuControlMailbox::new(
        CpuMailboxScope {
            job: envelope.id,
            owner,
        },
        CompletionSignal::default(),
    );
    controls.request_cancel(1).unwrap();
    controls.acknowledge_cancel(1).unwrap();
    let termination = CpuJobTermination {
        job: envelope.id,
        owner,
        last_execution: first.execution,
        completion: CpuCompletion::Canceled,
        controls: controls.seal_terminal(),
        retained_bytes: 0,
        shutdown: false,
    };
    ledger.accept_job_termination(&termination).unwrap();
    assert_eq!(ledger.global_usage().unwrap(), before);
    assert_eq!(ledger.usage(envelope.id).unwrap(), (1, 0, false));
    assert_eq!(
        ledger.accept_job_termination(&termination),
        Err(CpuWorkerError::Duplicate)
    );
    assert_eq!(ledger.global_usage().unwrap(), before);
    assert!(matches!(
        ledger.reserve(grant(envelope, owner, 2, 1)),
        Err(CpuWorkerError::Invalid(_))
    ));
}

#[test]
fn pre_enqueue_rejection_releases_credit_once_and_returns_the_immutable_job() {
    let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let clock = clock();
    let original = job(&cpu, Position::startpos(), None, &clock, CpuWorkerId(0), 10);
    let envelope = original.envelope();
    let original_controls = original.mailbox.clone();
    let original_deadline = original.hard_deadline;
    let original_target_depth = original.target_depth;
    let mut second_job = envelope;
    second_job.id.sequence = 2;
    second_job.question = Digest([8; 32]);
    let mut ledger = ledger(envelope, clock);
    ledger.register(envelope).unwrap();
    ledger.register(second_job).unwrap();
    let first = grant(envelope, CpuWorkerId(0), 1, 2);
    ledger.reserve(first).unwrap();
    assert_eq!(ledger.global_usage().unwrap(), (0, 2));

    // The mismatched worker owner rejects before enqueue; no CPU or NN search
    // starts and the proof is generated by the actual admission boundary.
    let worker = CpuWorker::spawn(CpuWorkerId(1), cpu).unwrap();
    let rejected = worker.begin_retaining(original, first).unwrap_err();
    assert!(matches!(rejected.reason(), CpuWorkerError::Invalid(_)));
    ledger.release_unadmitted(rejected.proof()).unwrap();
    assert_eq!(ledger.global_usage().unwrap(), (0, 0));
    assert_eq!(ledger.usage(envelope.id).unwrap(), (0, 0, false));
    let (reason, proof, retained) = rejected.into_parts();
    assert!(matches!(reason, CpuWorkerError::Invalid(_)));
    let retained = retained.expect("a rejected Begin must return its exact immutable Job");
    assert_eq!(retained.envelope(), envelope);
    assert_eq!(retained.hard_deadline, original_deadline);
    assert_eq!(retained.target_depth, original_target_depth);
    assert!(retained.root_moves.is_none());
    assert!(Arc::ptr_eq(&retained.mailbox, &original_controls));
    assert_eq!(
        ledger.release_unadmitted(&proof),
        Err(CpuWorkerError::Duplicate)
    );
    assert_eq!(ledger.global_usage().unwrap(), (0, 0));
    assert!(worker.try_event().unwrap().is_none());
    assert_eq!(worker.occupancy().reliable_reserved, 0);

    let next = grant(second_job, CpuWorkerId(0), 2, 1);
    ledger.reserve(next).unwrap();
    ledger.settle(&observed_report(next, 1, 1, true)).unwrap();
    assert_eq!(ledger.global_usage().unwrap(), (1, 0));
    worker.shutdown().unwrap();
}

#[test]
fn worker_profile_prepares_sequential_jobs_without_retaining_an_engine_reference() {
    let owner = CpuWorkerId(0);
    let shared_signal = CompletionSignal::default();
    // Move the only engine and TT into the worker before constructing any Job.
    let worker = CpuWorker::spawn_notified(
        owner,
        engine(CpuOrderingPolicy::LegalSeeV1, 16),
        shared_signal.clone(),
    )
    .unwrap();
    let profile = worker.profile();
    assert_eq!(profile.owner, owner);
    assert_eq!(profile.max_depth, 2);
    let clock = clock();
    let fixtures = [
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "e2e3"),
        ("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1", "e7e6"),
    ];
    for (index, (fen, movement)) in fixtures.into_iter().enumerate() {
        let position = Position::from_fen(fen).unwrap();
        let create_position = || ContractPosition::new(OwnerId(71), position.clone());
        let movement = BoardMove::from_uci(movement).unwrap();
        let restricted = vec![movement];
        let envelope = CpuJobEnvelope {
            id: CpuJobId {
                authority: SearchAuthority {
                    epoch: clock.domain().0,
                    game: GameGeneration(1),
                    root: RootGeneration(1),
                    implementation: Digest([9; 32]),
                },
                sequence: index as u64 + 1,
            },
            state_identity: create_position().export().unwrap().snapshot().identity(),
            question: Digest([index as u8 + 7; 32]),
            root_restriction: cpu_root_restriction_digest(Some(&restricted)),
            evaluation_namespace: profile.evaluation_namespace,
            search_conditions: profile.search_conditions,
            total_native_node_credit: 4096,
            hard_deadline: clock.deadline_at(clock.origin() + FIXTURE_WINDOW).unwrap(),
        };
        for mismatch in [true, false] {
            let mut wrong_profile = envelope;
            if mismatch {
                wrong_profile.evaluation_namespace.0[0] ^= 1;
            } else {
                wrong_profile.search_conditions.0[0] ^= 1;
            }
            assert!(matches!(
                worker.prepare_job(
                    wrong_profile,
                    create_position(),
                    Some(restricted.clone()),
                    1,
                    &clock,
                ),
                Err(CpuWorkerError::Invalid("CPU immutable namespace mismatch"))
            ));
        }
        assert!(matches!(
            worker.prepare_job(
                envelope,
                create_position(),
                Some(restricted.clone()),
                profile.max_depth + 1,
                &clock,
            ),
            Err(CpuWorkerError::Invalid("CPU cooperative depth or policy"))
        ));
        assert!(worker.try_event().unwrap().is_none());

        let prepared = worker
            .prepare_job(envelope, create_position(), Some(restricted), 1, &clock)
            .unwrap();
        assert_eq!(prepared.envelope(), envelope);
        let controls = prepared.mailbox.clone();
        assert_eq!(
            controls.scope(),
            &CpuMailboxScope {
                job: envelope.id,
                owner
            }
        );
        let before_control = shared_signal.version();
        if index == 0 {
            controls.request_publish_progress(1).unwrap();
        } else {
            controls.request_yield(1).unwrap();
        }
        // The immutable Job's control and private command/event mailbox share
        // the external manager notification without a relay thread.
        assert!(shared_signal.wait_changed(before_control, Duration::ZERO));
        let execution = index as u64 + 1;
        worker
            .begin(prepared, grant(envelope, owner, execution, 4096))
            .unwrap();
        let mut completed = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
        if index == 1 {
            assert!(!completed.admission_rejected && !completed.terminal);
            assert_eq!(completed.actual_work, Some(CpuWork::default()));
            let yielded = completed.result.as_ref().unwrap();
            assert_eq!(yielded.report.completion, CpuCompletion::Yielded);
            assert_eq!(yielded.yield_cause, Some(CpuYieldCause::Requested));
            assert_eq!(yielded.yield_ack_sequence, Some(1));
            assert!(yielded.report.resume.is_none());
            worker
                .resume(grant(envelope, owner, execution + 1, 4096))
                .unwrap();
            completed = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
        }
        assert!(!completed.admission_rejected);
        assert!(completed.terminal && !completed.continuation_retained);
        let result = completed.result.unwrap();
        assert_eq!(result.report.completion, CpuCompletion::DepthLimit);
        assert_eq!(result.report.completed_depth, 1);
        assert_eq!(result.report.best_move, Some(movement));
        assert!(result.report.resume.is_none());
        assert!(completed.actual_work.unwrap().nodes > 0);
        if index == 0 {
            assert_eq!(result.publish_ack_sequence, 1);
        }
        assert!(controls.snapshot().terminal);
        assert!(controls.request_cancel(2).is_err());
        assert_eq!(
            worker.profile().evaluation_namespace,
            profile.evaluation_namespace
        );
        assert_eq!(
            worker.profile().search_conditions,
            profile.search_conditions
        );
    }
    worker.shutdown().unwrap();
}

#[test]
fn shutdown_returns_unsettled_begin_events_so_exact_reservations_can_be_drained() {
    let cpu = engine(CpuOrderingPolicy::LegalSeeV1, 16);
    let clock = clock();
    let owner = CpuWorkerId(0);
    let original = job(&cpu, Position::startpos(), None, &clock, owner, 100);
    let envelope = original.envelope();
    let controls = original.mailbox.clone();
    let first = grant(envelope, owner, 1, 1);
    let mut ledger = ledger(envelope, clock);
    ledger.register(envelope).unwrap();
    ledger.reserve(first).unwrap();
    let worker = CpuWorker::spawn(owner, cpu).unwrap();
    worker.begin(original, first).unwrap();
    // Do not poll or settle before shutdown: whether Begin already ran is an
    // actor race, and every possible outcome must retain its real receipt.
    let stopped = worker.shutdown().unwrap();
    assert_eq!(stopped.owner, owner);
    assert!(stopped.thread_joined && !stopped.thread_panicked);
    assert!(stopped.mailbox.closed);
    assert_eq!(stopped.mailbox.commands, 0);
    assert_eq!(stopped.mailbox.reliable_queued, 0);
    assert_eq!(stopped.mailbox.reliable_reserved, 0);
    assert!(!stopped.events.is_empty());
    assert!(stopped.events.len() <= 2);
    let mut slice_count = 0;
    let mut observed_nodes = 0;
    let mut rejected_before_start = false;
    let mut terminal_observed = false;
    for event in &stopped.events {
        match event {
            CpuWorkerEvent::Slice(report) => {
                slice_count += 1;
                assert_eq!(report.grant, first);
                if let Ok(value) = &report.result {
                    assert!(value.report.resume.is_none());
                }
                let work = report
                    .actual_work
                    .expect("finite native fixture must report its work");
                observed_nodes += work.nodes;
                rejected_before_start = report.admission_rejected;
                terminal_observed |= report.terminal;
                if report.admission_rejected {
                    assert_eq!(work, CpuWork::default());
                    assert!(report.result.is_err() && !report.terminal);
                }
                ledger.settle(report).unwrap();
                assert_eq!(ledger.settle(report), Err(CpuWorkerError::Duplicate));
            }
            CpuWorkerEvent::JobTerminated(termination) => {
                assert_eq!(termination.job, envelope.id);
                assert_eq!(termination.last_execution, first.execution);
                assert!(termination.shutdown && termination.controls.terminal);
                assert_eq!(termination.retained_bytes, 0);
                ledger.accept_job_termination(termination).unwrap();
                terminal_observed = true;
            }
        }
    }
    assert_eq!(slice_count, 1);
    assert!(observed_nodes <= first.node_credit);
    assert_eq!(ledger.global_usage().unwrap(), (observed_nodes, 0));
    assert_eq!(
        ledger.usage(envelope.id).unwrap(),
        (observed_nodes, 0, false)
    );
    if rejected_before_start {
        ledger.retire_unstarted(envelope.id).unwrap();
    } else {
        assert!(
            terminal_observed,
            "started work must return a terminal receipt"
        );
    }
    assert!(controls.snapshot().terminal);
    let mut successor = envelope;
    successor.id.sequence = 2;
    ledger.register(successor).unwrap();
    ledger.reserve(grant(successor, owner, 2, 1)).unwrap();
    assert_eq!(ledger.global_usage().unwrap(), (observed_nodes, 1));
}

#[test]
fn manager_polls_nonblocking_join_and_settles_shutdown_events_without_reopening_controls() {
    for wait_for_native_pause in [false, true] {
        let cpu = engine(CpuOrderingPolicy::LegalSeeV1, 16);
        let clock = clock();
        let owner = CpuWorkerId(0);
        let original = job(&cpu, Position::startpos(), None, &clock, owner, 100);
        let envelope = original.envelope();
        let controls = original.mailbox.clone();
        let first = grant(envelope, owner, 1, 1);
        let mut ledger = ledger(envelope, clock);
        ledger.register(envelope).unwrap();
        ledger.reserve(first).unwrap();
        let mut worker = CpuWorker::spawn(owner, cpu).unwrap();
        worker.begin(original, first).unwrap();
        let mut events = Vec::with_capacity(2);
        if wait_for_native_pause {
            let paused = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
            assert!(paused.continuation_retained && !paused.terminal);
            assert!(!paused.admission_rejected);
            // Retain this real report without settlement until the final drain.
            events.push(CpuWorkerEvent::Slice(paused));
        }
        worker.request_shutdown();
        let stopped = loop {
            let observed = worker.signal.version();
            if let Some(stopped) = worker.try_join().unwrap() {
                break stopped;
            }
            match worker.try_event() {
                Ok(Some(event)) => {
                    events.push(event);
                    assert!(events.len() <= 2);
                    continue;
                }
                Ok(None) => {}
                // The actor can finish between try_join and try_event. The
                // next nonblocking join must recover its remaining receipts.
                Err(CpuWorkerError::Disconnected) => continue,
                other => panic!("nonblocking shutdown drain failed: {other:?}"),
            }
            let remaining = (clock.origin() + FIXTURE_WINDOW)
                .checked_duration_since(Instant::now())
                .expect("native fixture did not finish its bounded shutdown");
            worker
                .signal
                .wait_changed(observed, remaining.min(Duration::from_millis(10)));
        };
        assert_eq!(stopped.owner, owner);
        assert!(stopped.thread_joined && !stopped.thread_panicked);
        assert!(stopped.mailbox.closed);
        assert_eq!(stopped.mailbox.commands, 0);
        assert_eq!(stopped.mailbox.reliable_queued, 0);
        assert_eq!(stopped.mailbox.reliable_reserved, 0);
        events.extend(stopped.events);
        assert!(!events.is_empty() && events.len() <= 2);
        assert!(matches!(worker.try_join(), Err(CpuWorkerError::Duplicate)));

        let mut slices = 0;
        let mut used_nodes = 0;
        let mut rejected_before_start = false;
        let mut terminal_observed = false;
        for event in &events {
            match event {
                CpuWorkerEvent::Slice(report) => {
                    slices += 1;
                    assert_eq!(report.grant, first);
                    let work = report
                        .actual_work
                        .expect("finite native fixture has exact work");
                    used_nodes += work.nodes;
                    if let Ok(value) = &report.result {
                        assert!(value.report.resume.is_none());
                    }
                    rejected_before_start = report.admission_rejected;
                    terminal_observed |= report.terminal;
                    if report.admission_rejected {
                        assert_eq!(work, CpuWork::default());
                        assert!(report.result.is_err() && !report.terminal);
                        assert!(!report.continuation_retained);
                    }
                    ledger.settle(report).unwrap();
                    assert_eq!(ledger.settle(report), Err(CpuWorkerError::Duplicate));
                }
                CpuWorkerEvent::JobTerminated(termination) => {
                    assert_eq!(termination.job, envelope.id);
                    assert_eq!(termination.last_execution, first.execution);
                    assert!(termination.shutdown && termination.controls.terminal);
                    assert_eq!(termination.retained_bytes, 0);
                    ledger.accept_job_termination(termination).unwrap();
                    terminal_observed = true;
                }
            }
        }
        assert_eq!(slices, 1);
        assert!(used_nodes <= first.node_credit);
        assert_eq!(ledger.global_usage().unwrap(), (used_nodes, 0));
        assert_eq!(ledger.usage(envelope.id).unwrap(), (used_nodes, 0, false));
        if rejected_before_start {
            assert!(!wait_for_native_pause);
            ledger.retire_unstarted(envelope.id).unwrap();
        } else {
            assert!(
                terminal_observed,
                "started work must return its terminal receipt"
            );
        }
        assert!(controls.snapshot().terminal);
        assert!(controls.request_yield(1).is_err());
        assert!(controls.request_publish_progress(2).is_err());
        assert!(controls.request_cancel(3).is_err());
    }
}

#[test]
fn ready_readback_is_observed_once_by_the_owner_without_blocking_shared_control() {
    let caller = observe_current_cpu_thread(CpuActor::Manager);
    for owner in [CpuWorkerId(0), CpuWorkerId(256)] {
        let cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let clock = clock();
        let original = job(&cpu, Position::startpos(), None, &clock, owner, 100);
        let envelope = original.envelope();
        let signal = CompletionSignal::default();
        let worker = CpuWorker::spawn_notified(owner, cpu, signal.clone()).unwrap();
        let observed = loop {
            let version = signal.version();
            if let Some(readback) = worker.ready_readback() {
                break readback.clone();
            }
            let remaining = (clock.origin() + FIXTURE_WINDOW)
                .checked_duration_since(Instant::now())
                .expect("worker ready observation must have a finite boundary");
            signal.wait_changed(version, remaining.min(Duration::from_millis(10)));
        };
        if owner.0 <= u16::from(u8::MAX) {
            if let Ok(readback) = &observed {
                assert_eq!(readback.actor, CpuActor::OwnWorker(owner.0 as u8));
                assert_ne!(readback.thread_id, 0);
                if let Ok(caller) = &caller {
                    assert_ne!(readback.thread_id, caller.thread_id);
                }
            }
            // A platform probe may fail. It is retained as Err and is not a
            // fabricated complete inventory or an installed affinity claim.
        } else {
            assert_eq!(observed, Err(CpuResourceError::InvalidDeclaration));
        }
        worker
            .begin(original, grant(envelope, owner, 1, 1))
            .unwrap();
        let report = wait_report(&worker, clock.origin() + FIXTURE_WINDOW);
        assert!(!report.admission_rejected);
        assert_eq!(
            report.result.unwrap().report.completion,
            CpuCompletion::Yielded
        );
        assert_eq!(worker.ready_readback(), Some(&observed));
        let stopped = worker.shutdown().unwrap();
        assert!(stopped.thread_joined && !stopped.thread_panicked);
        assert_eq!(stopped.ready_readback, Some(observed));
    }
}
