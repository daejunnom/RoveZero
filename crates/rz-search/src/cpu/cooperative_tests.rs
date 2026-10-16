//! Regression coverage for the real PausedStack cooperative entry point.
//!
//! These are CPU/Rules tests. No mocked completion, sleep-based timing, NN,
//! native runtime or scheduler is used to substitute for entered CPU work.

use super::*;

// Keep fixture-only Job identity construction at this boundary. The production
// worker owns issued identities; these tests do not pretend to issue its IDs.
fn mailbox() -> Arc<CpuControlMailbox> {
    Arc::new(CpuControlMailbox::new(
        CpuMailboxScope {
            job: rz_contracts::pals_adaptive::CpuJobId {
                authority: rz_contracts::pals::SearchAuthority {
                    epoch: rz_contracts::ProcessEpoch(1),
                    game: rz_contracts::GameGeneration(1),
                    root: rz_contracts::RootGeneration(1),
                    implementation: rz_contracts::Digest([1; 32]),
                },
                sequence: 1,
            },
            owner: rz_contracts::pals_adaptive::CpuWorkerId(0),
        },
        rz_runtime::CompletionSignal::default(),
    ))
}

fn config(tt_entries: usize) -> CpuConfig {
    CpuConfig {
        profile: CpuProfile::PlanAssisted,
        tt_entries,
        max_depth: 2,
        quiescence_ply: 4,
    }
}

fn engine(ordering: CpuOrderingPolicy, tt_entries: usize) -> CpuEngine {
    CpuEngine::with_evaluator_ordering_and_resume_policy(
        config(tt_entries),
        Arc::new(BootstrapCpuValue::default()),
        ordering,
        CpuResumePolicy::PausedStack,
    )
    .unwrap()
}

fn limits() -> CpuLimits {
    CpuLimits {
        max_depth: 2,
        max_nodes: 100_000,
        deadline: None,
    }
}

fn slice(credit: u64, mailbox: &Arc<CpuControlMailbox>) -> CpuSliceControl {
    CpuSliceControl::new(credit, None, Arc::clone(mailbox)).unwrap()
}

fn add(total: &mut CpuWork, delta: CpuWork) {
    total.nodes = total.nodes.checked_add(delta.nodes).unwrap();
    total.quiescence_nodes = total
        .quiescence_nodes
        .checked_add(delta.quiescence_nodes)
        .unwrap();
    total.tt_hits = total.tt_hits.checked_add(delta.tt_hits).unwrap();
}

fn same_tt_and_history(left: &CpuEngine, right: &CpuEngine) {
    assert_eq!(left.history, right.history);
    assert_eq!(left.tt.len(), right.tt.len());
    for (left, right) in left.tt.iter().zip(&right.tt) {
        match (left, right) {
            (None, None) => {}
            (Some(left), Some(right)) => {
                assert_eq!(
                    (
                        left.hash,
                        left.depth,
                        left.score,
                        left.bound,
                        left.best_move,
                    ),
                    (
                        right.hash,
                        right.depth,
                        right.score,
                        right.bound,
                        right.best_move,
                    )
                );
                assert_eq!(left.identity, right.identity);
                assert_eq!(left.value_identity, right.value_identity);
            }
            _ => panic!("cooperative traversal changed a completed TT entry"),
        }
    }
}

/// Return only after the real traversal completes. The fixed finite call bound
/// catches an acknowledged Yield being applied again without new control.
fn run_slices(
    cpu: &mut CpuEngine,
    position: &Position,
    restriction: Option<&[BoardMove]>,
    credit: u64,
) -> (CpuReport, CpuWork) {
    let mailbox = mailbox();
    let cancel = AtomicBool::new(false);
    let before = position.snapshot();
    let owner = cpu.paused_stack_snapshot().unwrap().owner_id;
    let mut token = None;
    let mut total = CpuWork::default();
    for _ in 0..100_000 {
        let previous = token.take();
        let remaining = limits().max_nodes.checked_sub(total.nodes).unwrap();
        assert!(remaining > 0, "tiny workload exhausted its original budget");
        let controlled = cpu
            .analyze_controlled(
                position,
                restriction,
                CpuLimits {
                    max_nodes: remaining,
                    ..limits()
                },
                &cancel,
                &slice(credit, &mailbox),
                previous.as_ref(),
                |_| {},
            )
            .unwrap();
        let mut report = controlled.report;
        let delta = cpu.last_attempt_work().unwrap();
        assert_eq!(
            (report.nodes, report.quiescence_nodes, report.tt_hits),
            (delta.nodes, delta.quiescence_nodes, delta.tt_hits)
        );
        assert!(report.nodes <= credit);
        assert!(report.quiescence_nodes <= report.nodes);
        add(&mut total, delta);
        assert!(before.same_state(&position.snapshot()));
        if let Some(previous) = previous.as_ref() {
            assert!(!cpu.token_is_current(previous));
        }
        if report.completion == CpuCompletion::DepthLimit {
            assert_eq!(controlled.yield_cause, None);
            assert_eq!(controlled.yield_ack_sequence, None);
            assert_eq!(cpu.paused_stack_bytes(), None);
            let observation = cpu.paused_stack_snapshot().unwrap();
            assert_eq!(observation.owner_id, owner);
            assert_eq!(observation.tokens_retained, 0);
            assert_eq!(observation.replayed_consumed_work, 0);
            return (report, total);
        }
        assert_eq!(report.completion, CpuCompletion::Yielded);
        assert_eq!(controlled.yield_cause, Some(CpuYieldCause::NodeCredit));
        assert_eq!(controlled.yield_ack_sequence, None);
        let next = report.resume.take().expect("Yield owns a continuation");
        assert!(next.is_paused_stack());
        assert!(cpu.token_is_current(&next));
        assert_eq!(next.cumulative_work(), Some(total));
        assert!(next.paused_stack_bytes().unwrap() <= CPU_PAUSED_STACK_MAX_BYTES);
        let observation = cpu.paused_stack_snapshot().unwrap();
        assert_eq!(observation.owner_id, owner);
        assert_eq!(observation.tokens_retained, 1);
        assert_eq!(observation.replayed_consumed_work, 0);
        token = Some(next);
    }
    panic!("bounded cooperative workload made no terminal progress");
}

#[test]
fn cooperative_credit_slices_preserve_depth_score_pv_work_tt_and_history() {
    for ordering in [
        CpuOrderingPolicy::LegacyMvvLvaV1,
        CpuOrderingPolicy::LegalSeeV1,
    ] {
        for tt_entries in [0, 64] {
            for fen in [
                "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
                "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
                "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            ] {
                let position = Position::from_fen(fen).unwrap();
                let mut uninterrupted = engine(ordering, tt_entries);
                let expected = uninterrupted
                    .analyze(&position, limits(), &AtomicBool::new(false))
                    .unwrap();
                assert_eq!(expected.completion, CpuCompletion::DepthLimit);
                assert!(expected.quiescence_nodes > 0);
                for credit in [1, 3, 13] {
                    let mut controlled = engine(ordering, tt_entries);
                    let (actual, total) = run_slices(&mut controlled, &position, None, credit);
                    assert_eq!(
                        (actual.completed_depth, actual.score, &actual.pv),
                        (expected.completed_depth, expected.score, &expected.pv)
                    );
                    assert_eq!(total, uninterrupted.last_attempt_work().unwrap());
                    same_tt_and_history(&controlled, &uninterrupted);
                }
            }
        }
    }
}

#[test]
fn cooperative_see_and_restricted_root_resume_without_recharging() {
    let position = Position::from_fen("3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
    let restriction = [BoardMove::from_uci("e4d5").unwrap()];
    let mut uninterrupted = engine(CpuOrderingPolicy::LegalSeeV1, 16);
    let expected = uninterrupted
        .analyze_root_moves(&position, &restriction, limits(), &AtomicBool::new(false))
        .unwrap();
    assert_eq!(expected.completion, CpuCompletion::DepthLimit);
    // A legal tactical root under LegalSee necessarily enters the owned
    // Exchange path before that move can be searched. One-node grants divide
    // its initial and recapture entries into distinct invocations.
    for credit in [1, 3, 13] {
        let mut controlled = engine(CpuOrderingPolicy::LegalSeeV1, 16);
        let (actual, total) = run_slices(&mut controlled, &position, Some(&restriction), credit);
        assert!(actual.root_restricted);
        assert_eq!(
            (actual.completed_depth, actual.score, &actual.pv),
            (expected.completed_depth, expected.score, &expected.pv)
        );
        assert_eq!(total, uninterrupted.last_attempt_work().unwrap());
        same_tt_and_history(&controlled, &uninterrupted);
    }
}

#[test]
fn publish_progress_continues_and_never_acknowledges_a_depth_zero_fallback() {
    let position = Position::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
    let mailbox = mailbox();
    mailbox.request_publish_progress(1).unwrap();
    let mut controlled = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 16);
    let first = controlled
        .analyze_controlled(
            &position,
            None,
            limits(),
            &AtomicBool::new(false),
            &slice(1, &mailbox),
            None,
            |progress| {
                assert_eq!(progress.completed_depth, 0);
                assert_eq!(mailbox.snapshot().publish_ack, 0);
            },
        )
        .unwrap();
    assert_eq!(first.report.completion, CpuCompletion::Yielded);
    assert_eq!(first.publish_ack_sequence, 0);
    assert_eq!(mailbox.snapshot().publish_ack, 0);
    let token = first.report.resume.unwrap();
    let mut progress = Vec::new();
    let rest = controlled
        .analyze_controlled(
            &position,
            None,
            limits(),
            &AtomicBool::new(false),
            &slice(100_000, &mailbox),
            Some(&token),
            |value| progress.push(value),
        )
        .unwrap();
    assert_eq!(rest.report.completion, CpuCompletion::DepthLimit);
    assert_eq!(rest.yield_cause, None);
    assert_eq!(rest.yield_ack_sequence, None);
    assert_eq!(rest.publish_ack_sequence, 1);
    assert_eq!(mailbox.snapshot().publish_ack, 1);
    assert!(progress.iter().any(|value| value.completed_depth == 1));
    let mut uninterrupted = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 16);
    let expected = uninterrupted
        .analyze(&position, limits(), &AtomicBool::new(false))
        .unwrap();
    assert_eq!(
        (
            rest.report.completed_depth,
            rest.report.score,
            &rest.report.pv
        ),
        (expected.completed_depth, expected.score, &expected.pv)
    );
    let mut total = CpuWork {
        nodes: first.report.nodes,
        quiescence_nodes: first.report.quiescence_nodes,
        tt_hits: first.report.tt_hits,
    };
    add(&mut total, controlled.last_attempt_work().unwrap());
    assert_eq!(total, uninterrupted.last_attempt_work().unwrap());
    same_tt_and_history(&controlled, &uninterrupted);
}

#[test]
fn repeated_publications_do_not_stop_or_charge_extra_search_work() {
    let position = Position::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
    let mailbox = mailbox();
    mailbox.request_publish_progress(1).unwrap();
    let mut requested_second = false;
    let mut publications = Vec::new();
    let mut controlled = engine(CpuOrderingPolicy::LegalSeeV1, 64);
    let actual = controlled
        .analyze_controlled(
            &position,
            None,
            limits(),
            &AtomicBool::new(false),
            &slice(100_000, &mailbox),
            None,
            |value| {
                publications.push(value);
                if value.completed_depth == 2 && !requested_second {
                    mailbox.request_publish_progress(2).unwrap();
                    requested_second = true;
                }
            },
        )
        .unwrap();
    assert!(requested_second);
    assert_eq!(actual.report.completion, CpuCompletion::DepthLimit);
    assert_eq!(actual.publish_ack_sequence, 2);
    assert!(publications.iter().all(|value| {
        value.completed_depth <= actual.report.completed_depth && value.nodes <= actual.report.nodes
    }));
    let mut uninterrupted = engine(CpuOrderingPolicy::LegalSeeV1, 64);
    let expected = uninterrupted
        .analyze(&position, limits(), &AtomicBool::new(false))
        .unwrap();
    assert_eq!(
        (
            actual.report.completed_depth,
            actual.report.score,
            &actual.report.pv
        ),
        (expected.completed_depth, expected.score, &expected.pv)
    );
    assert_eq!(
        controlled.last_attempt_work(),
        uninterrupted.last_attempt_work()
    );
    same_tt_and_history(&controlled, &uninterrupted);
}

#[test]
fn requested_yield_before_start_is_acknowledged_once_and_resume_progresses() {
    let position = Position::startpos();
    let mailbox = mailbox();
    mailbox.request_yield(7).unwrap();
    let mut cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let first = cpu
        .analyze_controlled(
            &position,
            None,
            limits(),
            &AtomicBool::new(false),
            &slice(100_000, &mailbox),
            None,
            |_| {},
        )
        .unwrap();
    assert_eq!(first.report.completion, CpuCompletion::Yielded);
    assert_eq!(first.yield_cause, Some(CpuYieldCause::Requested));
    assert_eq!(first.yield_ack_sequence, Some(7));
    assert_eq!(first.report.nodes, 0);
    assert_eq!(first.report.score_scope, CpuScoreScope::FrontierOnly);
    assert_eq!(mailbox.snapshot().yield_ack, 7);
    let token = first.report.resume.unwrap();
    assert!(token.is_paused_stack() && cpu.token_is_current(&token));
    let next = cpu
        .analyze_controlled(
            &position,
            None,
            limits(),
            &AtomicBool::new(false),
            &slice(1, &mailbox),
            Some(&token),
            |_| {},
        )
        .unwrap();
    assert_eq!(next.report.completion, CpuCompletion::Yielded);
    assert_eq!(next.yield_cause, Some(CpuYieldCause::NodeCredit));
    assert_eq!(next.yield_ack_sequence, None);
    assert_eq!(next.report.nodes, 1);
    assert_eq!(mailbox.snapshot().yield_ack, 7);
    assert!(!cpu.token_is_current(&token));
    assert!(mailbox.request_yield(7).is_err());
    assert!(mailbox.request_yield(0).is_err());
}

#[test]
fn soft_lease_yields_without_expiring_the_job() {
    let position = Position::startpos();
    let mailbox = mailbox();
    let mut cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let expired_lease =
        CpuSliceControl::new(100, Some(Instant::now()), Arc::clone(&mailbox)).unwrap();
    let first = cpu
        .analyze_controlled(
            &position,
            None,
            limits(),
            &AtomicBool::new(false),
            &expired_lease,
            None,
            |_| {},
        )
        .unwrap();
    assert_eq!(first.report.completion, CpuCompletion::Yielded);
    assert_eq!(first.yield_cause, Some(CpuYieldCause::SoftLease));
    assert_eq!(first.yield_ack_sequence, None);
    assert_eq!(first.report.nodes, 0);
    let token = first.report.resume.unwrap();
    let next = cpu
        .analyze_controlled(
            &position,
            None,
            limits(),
            &AtomicBool::new(false),
            &slice(1, &mailbox),
            Some(&token),
            |_| {},
        )
        .unwrap();
    assert_eq!(next.yield_cause, Some(CpuYieldCause::NodeCredit));
    assert_eq!(next.report.nodes, 1);
}

#[test]
fn cancellation_and_hard_expiry_win_over_pending_yield_and_discard_tokens() {
    let position = Position::startpos();
    for completion in [CpuCompletion::Canceled, CpuCompletion::Deadline] {
        let mailbox = mailbox();
        let mut cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let first = cpu
            .analyze_controlled(
                &position,
                None,
                limits(),
                &AtomicBool::new(false),
                &slice(1, &mailbox),
                None,
                |_| {},
            )
            .unwrap();
        let token = first.report.resume.unwrap();
        mailbox.request_yield(1).unwrap();
        if completion == CpuCompletion::Canceled {
            mailbox.request_cancel(2).unwrap();
        }
        let terminal_limits = CpuLimits {
            deadline: (completion == CpuCompletion::Deadline).then(Instant::now),
            ..limits()
        };
        let terminal = cpu
            .analyze_controlled(
                &position,
                None,
                terminal_limits,
                &AtomicBool::new(false),
                &slice(1, &mailbox),
                Some(&token),
                |_| {},
            )
            .unwrap();
        assert_eq!(terminal.report.completion, completion);
        assert_eq!(terminal.report.nodes, 0);
        assert_eq!(terminal.yield_cause, None);
        assert_eq!(terminal.yield_ack_sequence, None);
        assert_eq!(mailbox.snapshot().yield_ack, 0);
        assert!(terminal.report.resume.is_none());
        assert!(!cpu.token_is_current(&token));
        assert_eq!(cpu.paused_stack_bytes(), None);
    }
    let mut cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let mailbox = mailbox();
    mailbox.request_yield(1).unwrap();
    let terminal = cpu
        .analyze_controlled(
            &position,
            None,
            limits(),
            &AtomicBool::new(true),
            &slice(1, &mailbox),
            None,
            |_| {},
        )
        .unwrap();
    assert_eq!(terminal.report.completion, CpuCompletion::Canceled);
    assert_eq!(terminal.yield_ack_sequence, None);
    assert!(terminal.report.resume.is_none());
}

#[test]
fn hard_node_limit_discards_cooperative_work_but_legacy_limit_still_pauses() {
    let position = Position::startpos();
    let hard = CpuLimits {
        max_nodes: 1,
        ..limits()
    };
    let mailbox = mailbox();
    let mut cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let actual = cpu
        .analyze_controlled(
            &position,
            None,
            hard,
            &AtomicBool::new(false),
            &slice(1, &mailbox),
            None,
            |_| {},
        )
        .unwrap();
    assert_eq!(actual.report.completion, CpuCompletion::NodeLimit);
    assert_eq!(actual.report.nodes, 1);
    assert_eq!(actual.yield_cause, None);
    assert!(actual.report.resume.is_none());
    assert_eq!(cpu.paused_stack_bytes(), None);
    let legacy = cpu
        .analyze(&position, hard, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(legacy.completion, CpuCompletion::NodeLimit);
    assert!(legacy.resume.as_ref().unwrap().is_paused_stack());
    let legacy_token = legacy.resume.unwrap();
    let expired = cpu
        .resume(
            &position,
            &legacy_token,
            CpuLimits {
                deadline: Some(Instant::now()),
                ..hard
            },
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(expired.completion, CpuCompletion::Deadline);
    assert!(expired.resume.as_ref().unwrap().is_paused_stack());
}

#[test]
fn natural_completion_does_not_acknowledge_a_late_yield_as_worker_release() {
    let position = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
    let mailbox = mailbox();
    let mut requested = false;
    let mut cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
    let actual = cpu
        .analyze_controlled(
            &position,
            None,
            CpuLimits {
                max_depth: 1,
                ..limits()
            },
            &AtomicBool::new(false),
            &slice(100_000, &mailbox),
            None,
            |progress| {
                if progress.completed_depth == 1 && !requested {
                    mailbox.request_yield(9).unwrap();
                    requested = true;
                }
            },
        )
        .unwrap();
    assert!(requested);
    assert_eq!(actual.report.completion, CpuCompletion::DepthLimit);
    assert_eq!(actual.yield_cause, None);
    assert_eq!(actual.yield_ack_sequence, None);
    assert_eq!(mailbox.snapshot().yield_ack, 0);
    assert_eq!(cpu.paused_stack_bytes(), None);
    let mut result = position.clone();
    result.make_move(actual.report.best_move.unwrap()).unwrap();
    assert!(matches!(
        result
            .play_status_from_view(&result.ordered_legal_moves())
            .unwrap(),
        PlayStatus::Terminal {
            reason: TerminalReason::Checkmate,
            ..
        }
    ));
}

#[test]
fn restriction_history_and_same_owner_tokens_are_required_for_resume() {
    let mut position = Position::startpos();
    position
        .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8"])
        .unwrap();
    let restriction = [BoardMove::from_uci("e2e4").unwrap()];
    let mailbox = mailbox();
    let mut cpu = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 16);
    let first = cpu
        .analyze_controlled(
            &position,
            Some(&restriction),
            limits(),
            &AtomicBool::new(false),
            &slice(1, &mailbox),
            None,
            |_| {},
        )
        .unwrap();
    let token = first.report.resume.unwrap();
    let control = slice(1, &mailbox);
    assert!(matches!(
        cpu.analyze_controlled(
            &position,
            None,
            limits(),
            &AtomicBool::new(false),
            &control,
            Some(&token),
            |_| {}
        ),
        Err(CpuError::ResumeMismatch(_))
    ));
    assert!(cpu.token_is_current(&token));
    let imported = Position::from_fen(&position.to_fen()).unwrap();
    assert!(matches!(
        cpu.analyze_controlled(
            &imported,
            Some(&restriction),
            limits(),
            &AtomicBool::new(false),
            &control,
            Some(&token),
            |_| {}
        ),
        Err(CpuError::ResumeMismatch(_))
    ));
    assert!(cpu.token_is_current(&token));
    let mut foreign = engine(CpuOrderingPolicy::LegacyMvvLvaV1, 16);
    assert!(matches!(
        foreign.analyze_controlled(
            &position,
            Some(&restriction),
            limits(),
            &AtomicBool::new(false),
            &control,
            Some(&token),
            |_| {}
        ),
        Err(CpuError::ResumeMismatch(_))
    ));
    assert!(cpu.token_is_current(&token));
    let next = cpu
        .analyze_controlled(
            &position,
            Some(&restriction),
            limits(),
            &AtomicBool::new(false),
            &slice(1, &mailbox),
            Some(&token),
            |_| {},
        )
        .unwrap();
    assert_eq!(next.report.nodes, 1);
    assert!(!cpu.token_is_current(&token));
    let next_token = next.report.resume.unwrap();
    cpu.clear();
    assert!(!cpu.token_is_current(&next_token));
    assert!(matches!(
        cpu.analyze_controlled(
            &position,
            Some(&restriction),
            limits(),
            &AtomicBool::new(false),
            &control,
            Some(&next_token),
            |_| {}
        ),
        Err(CpuError::ResumeMismatch(_))
    ));
}

#[test]
fn cooperative_control_rejects_completed_iteration_only_engine_and_bad_credit() {
    let position = Position::startpos();
    let mailbox = mailbox();
    assert!(matches!(
        CpuSliceControl::new(0, None, Arc::clone(&mailbox)),
        Err(CpuError::InvalidLimits(_))
    ));
    let mut legacy = CpuEngine::new(config(0)).unwrap();
    assert!(matches!(
        legacy.analyze_controlled(
            &position,
            None,
            limits(),
            &AtomicBool::new(false),
            &slice(1, &mailbox),
            None,
            |_| {}
        ),
        Err(CpuError::Unsupported(_))
    ));
    assert!(mailbox.request_publish_progress(0).is_err());
    mailbox.request_publish_progress(3).unwrap();
    assert!(mailbox.request_publish_progress(2).is_err());
    assert!(mailbox.request_publish_progress(3).is_err());
    assert_eq!(mailbox.snapshot().publish_ack, 0);
}
