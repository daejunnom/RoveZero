//! Rules replay relative to the private original preparation, never Debug text.
//! The old model counterline/Repair record revision are absent from this wire;
//! therefore actual anchor SELECTION and native execution remain unadmitted.
use super::*;
use crate::pals_cpu_task;
use crate::pals_cpu_task::strategic_action::replay_inputs::{
    PreparedReplayRequest, ReplayInputError, ReplayResultRoot,
};
use rz_position::{PlayStatus, Position};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Keep the original root reconstruction cause, rather than flattening it to a
/// generic report refusal or losing the failing original input stage.
#[derive(Debug)]
pub enum ReportedRulesError {
    Report(AdmissionFault),
    Original(Box<ReplayInputError>),
}
impl From<AdmissionFault> for ReportedRulesError {
    fn from(error: AdmissionFault) -> Self {
        Self::Report(error)
    }
}

/// Only the report and small Rules facts survive. Temporary original/branch
/// history graphs are released before return; no model/session or imported owner.
/// No Deserialize/Clone/into-projection or native/Query capability constructor.
pub struct ReportedRulesConsistency {
    report: ReportedPriorConsistency,
    repaired_endpoint: PlayStatus,
    opponent_endpoint: PlayStatus,
    anchor_ply: usize,
}
impl ReportedRulesConsistency {
    pub fn reported(&self) -> &ReportedPriorConsistency {
        &self.report
    }
    pub fn original_input_artifact(&self) -> &ArtifactPin {
        &self.report.input_artifact
    }
    pub fn repaired_endpoint(&self) -> PlayStatus {
        self.repaired_endpoint
    }
    pub fn opponent_endpoint(&self) -> PlayStatus {
        self.opponent_endpoint
    }
    pub fn opponent_anchor_ply(&self) -> usize {
        self.anchor_ply
    }
    pub fn assurance_scope(&self) -> &'static str {
        "reported_lines_rules_checked_pending_repair_anchor_selection_native_witness_and_caller_chronology"
    }
}
impl ReportedPriorConsistency {
    /// Consumes a consistent report, but obtains Rules solely from the private
    /// previously admitted original backing. This is result verification in W,
    /// not fresh CPU/native work in E or a rerun of admission with a new clock.
    pub fn check_original_rules(
        self,
        prepared: &PreparedReplayRequest,
        cancel: &AtomicBool,
    ) -> Result<ReportedRulesConsistency, ReportedRulesError> {
        control(prepared.deadline(), cancel)?;
        let audit = &prepared.audit().input_admission;
        if self.input_artifact != *prepared.artifact()
            || audit.input_artifact != *prepared.artifact()
            || audit.mode != ReplayInputMode::RepairOpponent4n
            || self.whole_wall_ms != audit.whole_wall_ms
            || self.cleanup_reserve_ms != audit.cleanup_reserve_ms
        {
            return Err(fault("reported Rules original input/clock binding differs").into());
        }
        let root = prepared
            .reconstruct_result_root(cancel)
            .map_err(|e| ReportedRulesError::Original(Box::new(e)))?;
        let facts = check_lines(&root, &self.projection, cancel)?;
        // No graph is stored in the returned report. S/W/E remain unmodified.
        drop(root);
        control(prepared.deadline(), cancel)?;
        Ok(ReportedRulesConsistency {
            report: self,
            repaired_endpoint: facts.repaired_endpoint,
            opponent_endpoint: facts.opponent_endpoint,
            anchor_ply: facts.anchor_ply,
        })
    }
}
struct RulesFacts {
    repaired_endpoint: PlayStatus,
    opponent_endpoint: PlayStatus,
    anchor_ply: usize,
}
fn control(deadline: Instant, cancel: &AtomicBool) -> Result<(), AdmissionFault> {
    if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
        Err(fault("reported Rules original window/cancellation"))
    } else {
        Ok(())
    }
}
fn status(position: &Position) -> Result<PlayStatus, AdmissionFault> {
    let view = position.ordered_legal_moves();
    position
        .play_status_from_view(&view)
        .map_err(|_| fault("reported Rules status failed"))
}
fn replay(
    root: &Position,
    line: &[BoardMove],
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<Position, AdmissionFault> {
    let mut position = root.clone();
    for movement in line {
        control(deadline, cancel)?;
        let view = position.ordered_legal_moves();
        if position
            .play_status_from_view(&view)
            .map_err(|_| fault("reported Rules status failed"))?
            != PlayStatus::Ongoing
        {
            return Err(fault("reported line continues after Rules terminal"));
        }
        position
            .make_from_view(&view, *movement)
            .map_err(|_| fault("reported line contains illegal Rules move"))?;
    }
    control(deadline, cancel)?;
    Ok(position)
}
fn check_lines(
    origin: &ReplayResultRoot,
    p: &ReplayQueryPrior,
    cancel: &AtomicBool,
) -> Result<RulesFacts, AdmissionFault> {
    control(origin.whole_deadline, cancel)?;
    if !matches!(
        p.outcome,
        PriorOutcome::CompletedOpponentEndpoint | PriorOutcome::RulesTerminalOpponentEndpoint
    ) || !p.endpoint_observed
        || origin.prefix.is_empty()
        || !(1..=MAX_LINE_MOVES).contains(&origin.line_plies)
        || p.repaired_line.len == 0
        || p.repaired_line.len > origin.line_plies
        || p.opponent_counterline.len == 0
        || p.opponent_counterline.len > origin.line_plies
    {
        return Err(fault("reported Rules completed endpoint/line scope"));
    }
    let repair = pals_cpu_task::decode_moves(p.repaired_line.as_slice())
        .map_err(|_| fault("reported Rules repaired move codec"))?;
    let counter = pals_cpu_task::decode_moves(p.opponent_counterline.as_slice())
        .map_err(|_| fault("reported Rules counterline move codec"))?;
    let anchor = p
        .opponent_anchor_ply
        .ok_or_else(|| fault("reported Rules opponent anchor missing"))?;
    if anchor <= origin.prefix.len()
        || anchor >= repair.len()
        || anchor >= counter.len()
        || !repair.starts_with(&origin.prefix)
        || !counter.starts_with(&origin.prefix)
        || repair[..anchor] != counter[..anchor]
        || repair[anchor] == counter[anchor]
        || p.opponent_role_steps != counter.len() - anchor
        || p.accepted_opponent_steps != p.opponent_role_steps
    {
        return Err(fault(
            "reported Rules original prefix/anchor/alternative/role extent",
        ));
    }
    let at_anchor = replay(
        &origin.root,
        &repair[..anchor],
        origin.whole_deadline,
        cancel,
    )?;
    if at_anchor.side_to_move() == origin.root.side_to_move()
        || status(&at_anchor)? != PlayStatus::Ongoing
    {
        return Err(fault(
            "reported Rules anchor is not an ongoing opponent turn",
        ));
    }
    let repaired = replay(&origin.root, &repair, origin.whole_deadline, cancel)?;
    let repaired_endpoint = status(&repaired)?;
    let opponent = replay(&origin.root, &counter, origin.whole_deadline, cancel)?;
    let opponent_endpoint = status(&opponent)?;
    let terminal = opponent_endpoint != PlayStatus::Ongoing;
    if p.endpoint_rules_terminal != terminal
        || (p.outcome == PriorOutcome::RulesTerminalOpponentEndpoint) != terminal
        || (!terminal && counter.len() != origin.line_plies)
    {
        return Err(fault("reported Rules terminal/outcome/horizon differs"));
    }
    control(origin.whole_deadline, cancel)?;
    Ok(RulesFacts {
        repaired_endpoint,
        opponent_endpoint,
        anchor_ply: anchor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    fn moves(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|mv| BoardMove::from_uci(mv).unwrap())
            .collect()
    }
    fn fixture() -> (ReplayResultRoot, ReplayQueryPrior) {
        let root = ReplayResultRoot {
            root: Position::startpos(),
            prefix: moves("e2e4"),
            line_plies: 6,
            whole_deadline: Instant::now() + Duration::from_secs(5),
        };
        let mut p = ReplayQueryPrior {
            outcome: PriorOutcome::CompletedOpponentEndpoint,
            opponent_anchor_ply: Some(3),
            endpoint_observed: true,
            opponent_role_steps: 3,
            accepted_opponent_steps: 3,
            ..ReplayQueryPrior::default()
        };
        p.repaired_line
            .capture(&moves("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6"), MAX_LINE_MOVES)
            .unwrap();
        p.opponent_counterline
            .capture(&moves("e2e4 e7e5 g1f3 g8f6 f1c4 f6e4"), MAX_LINE_MOVES)
            .unwrap();
        (root, p)
    }
    #[test]
    fn reported_rules_lines_keep_original_prefix_and_legal_opponent_alternative() {
        let (root, p) = fixture();
        let before = root.root.snapshot();
        let facts = check_lines(&root, &p, &AtomicBool::new(false)).unwrap();
        assert_eq!(facts.anchor_ply, 3);
        assert_eq!(facts.opponent_endpoint, PlayStatus::Ongoing);
        assert!(before.same_state(&root.root.snapshot()));
        assert_eq!(p.authorities, ReplayAuthorities::default());
        assert_eq!(p.actual_utility_groups, 0);
    }
    #[test]
    fn reported_rules_reject_illegal_prefix_anchor_counts_and_false_terminal() {
        for case in 0..11 {
            let (root, mut p) = fixture();
            match case {
                0 => p
                    .repaired_line
                    .capture(&moves("d2d4 e7e5 g1f3 b8c6 f1b5 a7a6"), MAX_LINE_MOVES)
                    .unwrap(),
                1 => {
                    p.repaired_line
                        .capture(&moves("e2e4 e7e5 f1f3 b8c6 f1b5 a7a6"), MAX_LINE_MOVES)
                        .unwrap();
                    p.opponent_counterline
                        .capture(&moves("e2e4 e7e5 f1f3 g8f6 f1c4 f6e4"), MAX_LINE_MOVES)
                        .unwrap();
                }
                2 => {
                    p.opponent_anchor_ply = Some(2);
                    p.opponent_counterline
                        .capture(&moves("e2e4 e7e5 g1h3 g8f6 f1c4 f6e4"), MAX_LINE_MOVES)
                        .unwrap();
                    p.opponent_role_steps = 4;
                    p.accepted_opponent_steps = 4;
                }
                3 => p.opponent_anchor_ply = Some(1),
                4 => p
                    .opponent_counterline
                    .capture(&moves("e2e4 e7e5 g1h3 g8f6 f1c4 f6e4"), MAX_LINE_MOVES)
                    .unwrap(),
                5 => p
                    .opponent_counterline
                    .capture(&moves("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6"), MAX_LINE_MOVES)
                    .unwrap(),
                6 => {
                    p.endpoint_rules_terminal = true;
                    p.outcome = PriorOutcome::RulesTerminalOpponentEndpoint;
                }
                7 => p.opponent_role_steps = 4,
                8 => p.accepted_opponent_steps = 2,
                9 => p.endpoint_observed = false,
                _ => {
                    p.opponent_counterline
                        .capture(&moves("e2e4 e7e5 g1f3 g8f6 f1c4"), MAX_LINE_MOVES)
                        .unwrap();
                    p.opponent_role_steps = 2;
                    p.accepted_opponent_steps = 2;
                }
            }
            assert!(
                check_lines(&root, &p, &AtomicBool::new(false)).is_err(),
                "case {case}"
            );
        }
    }
    #[test]
    fn reported_rules_actual_mate_stalemate_and_terminal_suffix_are_rules_owned() {
        let (mut root, mut p) = fixture();
        root.prefix = moves("f2f3");
        p.repaired_line
            .capture(&moves("f2f3 e7e5 g2g4 d8f6"), MAX_LINE_MOVES)
            .unwrap();
        p.opponent_counterline
            .capture(&moves("f2f3 e7e5 g2g4 d8h4"), MAX_LINE_MOVES)
            .unwrap();
        p.opponent_role_steps = 1;
        p.accepted_opponent_steps = 1;
        p.endpoint_rules_terminal = true;
        p.outcome = PriorOutcome::RulesTerminalOpponentEndpoint;
        let facts = check_lines(&root, &p, &AtomicBool::new(false)).unwrap();
        assert_eq!(
            facts.opponent_endpoint,
            PlayStatus::Terminal {
                reason: rz_position::TerminalReason::Checkmate,
                winner: Some(rz_position::Color::Black),
            }
        );
        p.endpoint_rules_terminal = false;
        p.outcome = PriorOutcome::CompletedOpponentEndpoint;
        assert!(check_lines(&root, &p, &AtomicBool::new(false)).is_err());
        p.endpoint_rules_terminal = true;
        p.outcome = PriorOutcome::RulesTerminalOpponentEndpoint;
        p.opponent_counterline
            .capture(&moves("f2f3 e7e5 g2g4 d8h4 a2a3"), MAX_LINE_MOVES)
            .unwrap();
        p.opponent_role_steps = 2;
        p.accepted_opponent_steps = 2;
        assert!(check_lines(&root, &p, &AtomicBool::new(false)).is_err());
        let stalemate = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1").unwrap();
        assert_eq!(
            status(&stalemate).unwrap(),
            PlayStatus::Terminal {
                reason: rz_position::TerminalReason::Stalemate,
                winner: None,
            }
        );
        assert!(
            replay(
                &stalemate,
                &moves("h8h7"),
                root.whole_deadline,
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
    #[test]
    fn reported_rules_original_window_and_cancellation_do_not_reopen_work() {
        let (mut root, p) = fixture();
        assert!(check_lines(&root, &p, &AtomicBool::new(true)).is_err());
        root.whole_deadline = Instant::now();
        assert!(check_lines(&root, &p, &AtomicBool::new(false)).is_err());
    }

    #[test]
    fn reported_rules_private_original_binding_keeps_native_and_anchor_selection_pending() {
        let prepared =
            pals_cpu_task::strategic_action::replay_inputs::tests::controlled_prepared_result_rules(
            );
        let make_report = || {
            let (_, mut projection) = fixture();
            projection
                .repaired_line
                .capture(&moves("e2e4 e7e5 g1f3 b8c6"), MAX_LINE_MOVES)
                .unwrap();
            projection
                .opponent_counterline
                .capture(&moves("e2e4 e7e5 g1f3 g8f6"), MAX_LINE_MOVES)
                .unwrap();
            projection.opponent_role_steps = 1;
            projection.accepted_opponent_steps = 1;
            ReportedPriorConsistency {
                projection,
                readiness: PriorReadiness::NativeClosureUnobserved,
                input_artifact: prepared.artifact().clone(),
                whole_wall_ms: prepared.audit().input_admission.whole_wall_ms,
                cleanup_reserve_ms: prepared.audit().input_admission.cleanup_reserve_ms,
            }
        };
        let result = make_report()
            .check_original_rules(&prepared, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(result.original_input_artifact(), prepared.artifact());
        assert_eq!(result.opponent_endpoint(), PlayStatus::Ongoing);
        assert_eq!(
            result.reported().recomputed_reported_readiness(),
            PriorReadiness::NativeClosureUnobserved
        );
        assert_eq!(
            result.assurance_scope(),
            "reported_lines_rules_checked_pending_repair_anchor_selection_native_witness_and_caller_chronology"
        );
        let mut bad = make_report();
        bad.input_artifact.sha256 = "0".repeat(64);
        assert!(matches!(
            bad.check_original_rules(&prepared, &AtomicBool::new(false)),
            Err(ReportedRulesError::Report(_))
        ));
        let mut bad = make_report();
        bad.whole_wall_ms += 1;
        assert!(matches!(
            bad.check_original_rules(&prepared, &AtomicBool::new(false)),
            Err(ReportedRulesError::Report(_))
        ));
        assert!(
            make_report()
                .check_original_rules(&prepared, &AtomicBool::new(true))
                .is_err()
        );
    }
}
