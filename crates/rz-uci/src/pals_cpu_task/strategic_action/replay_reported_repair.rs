//! Closed /4 report and original Rules anchor check; no native witness import.
use super::super::repair_evidence::RepairAnchorEvidence;
use super::*;
use crate::pals_cpu_task::strategic_action::replay_inputs::{
    PreparedReplayRequest, ReplayResultRoot,
};
use rz_position::PlayStatus;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RepairWire {
    schema: String,
    assurance_scope: String,
    source_sha256: [u8; 32],
    source_bytes: u64,
    model_counterline: Vec<u16>,
    repaired_line: Vec<u16>,
    repair_record_observed: bool,
    repair_record_revision: Option<u64>,
    opponent_anchor_ply: Option<usize>,
    actual_utility_groups: u8,
    authorities: ReplayAuthorities,
}
/// These fields are a checked REPORT. No accessor moves an actual owner evidence
/// object into a NativeReplayObservation; no Deserialize/Clone/Serialize.
pub struct ReportedRepairConsistency {
    report: ReportedPriorConsistency,
    evidence: RepairAnchorEvidence,
}
impl ReportedRepairConsistency {
    pub fn reported(&self) -> &ReportedPriorConsistency {
        &self.report
    }
    pub fn reported_record_revision(&self) -> Option<u64> {
        self.evidence.repair_record_revision
    }
    pub fn assurance_scope(&self) -> &'static str {
        "reported_repair_origin_pending_original_rules_native_witness_and_caller_chronology"
    }
    pub fn check_original_rules(
        self,
        prepared: &PreparedReplayRequest,
        cancel: &AtomicBool,
    ) -> Result<ReportedRepairRulesConsistency, ReportedRulesError> {
        let Self { report, evidence } = self;
        let (rules, anchor) =
            report.check_original_rules_with(prepared, cancel, |root, prior| {
                check_anchor(root, prior, &evidence, cancel)
            })?;
        Ok(ReportedRepairRulesConsistency {
            rules,
            first_anchor: anchor,
            reported_record_revision: evidence
                .repair_record_revision
                .expect("checked actual record report"),
        })
    }
}
pub struct ReportedRepairRulesConsistency {
    rules: ReportedRulesConsistency,
    first_anchor: usize,
    reported_record_revision: u64,
}
impl ReportedRepairRulesConsistency {
    pub fn rules(&self) -> &ReportedRulesConsistency {
        &self.rules
    }
    pub fn first_eligible_anchor_ply(&self) -> usize {
        self.first_anchor
    }
    /// Local revision reported by producer; not an independently witnessed record.
    pub fn reported_record_revision(&self) -> u64 {
        self.reported_record_revision
    }
    pub fn assurance_scope(&self) -> &'static str {
        "reported_first_repair_anchor_rules_checked_pending_native_witness_and_caller_chronology"
    }
}

pub fn check_reported_repair_consistency(
    mut body: Value,
    expected: &ReplayExpectedPins,
    profile_pin: &ArtifactPin,
    profile: &CpuFreshAssetProfile,
    projection_source: &ArtifactPin,
    repair_source: &ArtifactPin,
) -> Result<ReportedRepairConsistency, AdmissionFault> {
    bounded_json_extent(
        &body,
        super::super::super::super::replay_inputs::MAX_OUTPUT_BYTES,
    )?;
    let raw = body
        .as_object_mut()
        .and_then(|o| o.remove("repair_anchor_evidence"))
        .ok_or_else(|| fault("reported explicit Repair evidence missing"))?;
    bounded_json_extent(&raw, MAX_PROJECTED_JSON_BYTES)?;
    let wire: RepairWire = serde_json::from_value(raw)
        .map_err(|_| fault("reported Repair closed field/type shape"))?;
    let digest = wire
        .source_sha256
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    if wire.schema != REPAIR_EVIDENCE_SCHEMA
        || wire.assurance_scope != REPAIR_EVIDENCE_SCOPE
        || digest != repair_source.sha256
        || wire.source_bytes != repair_source.bytes
        || repair_source.bytes == 0
        || repair_source.bytes > 512 * 1024
    {
        return Err(fault(
            "reported independently registered Repair source differs",
        ));
    }
    let report = check_reported_prior_schema(
        body,
        expected,
        profile_pin,
        profile,
        projection_source,
        super::super::super::REPAIR_ANCHOR_OBSERVATION_SCHEMA,
    )?;
    let evidence = RepairAnchorEvidence {
        schema: REPAIR_EVIDENCE_SCHEMA,
        assurance_scope: REPAIR_EVIDENCE_SCOPE,
        source_sha256: wire.source_sha256,
        source_bytes: wire.source_bytes,
        model_counterline: packed(wire.model_counterline, MAX_LINE_MOVES)?,
        repaired_line: packed(wire.repaired_line, MAX_LINE_MOVES)?,
        repair_record_observed: wire.repair_record_observed,
        repair_record_revision: wire.repair_record_revision,
        opponent_anchor_ply: wire.opponent_anchor_ply,
        actual_utility_groups: wire.actual_utility_groups,
        authorities: wire.authorities,
    };
    evidence.check_projection(report.reported_projection())?;
    Ok(ReportedRepairConsistency { report, evidence })
}

fn control(root: &ReplayResultRoot, cancel: &AtomicBool) -> Result<(), AdmissionFault> {
    if cancel.load(Ordering::Acquire) || Instant::now() >= root.whole_deadline {
        Err(fault("reported Repair original W/cancellation"))
    } else {
        Ok(())
    }
}
fn check_anchor(
    root: &ReplayResultRoot,
    prior: &ReplayQueryPrior,
    evidence: &RepairAnchorEvidence,
    cancel: &AtomicBool,
) -> Result<usize, AdmissionFault> {
    control(root, cancel)?;
    evidence.check_projection(prior)?;
    let model = crate::pals_cpu_task::decode_moves(evidence.model_counterline.as_slice())
        .map_err(|_| fault("reported original model counterline move codec"))?;
    let repair = crate::pals_cpu_task::decode_moves(evidence.repaired_line.as_slice())
        .map_err(|_| fault("reported original Repair move codec"))?;
    let attack = root.prefix.len();
    if !evidence.repair_record_observed
        || repair.len() <= attack
        || model.len() <= attack
        || repair.len() > root.line_plies
        || model.len() > root.line_plies
        || !model.starts_with(&root.prefix)
        || repair[..=attack] != model[..=attack]
    {
        return Err(fault("reported original model C/Repair prefix differs"));
    }
    // Rules are the sole owner of legality/terminal state; never trust endpoint
    // text or CPU/model scores. The model line must be complete or Rules terminal.
    let mut original = root.root.clone();
    for movement in &model {
        control(root, cancel)?;
        let view = original.ordered_legal_moves();
        if original
            .play_status_from_view(&view)
            .map_err(|_| fault("model Rules status"))?
            != PlayStatus::Ongoing
        {
            return Err(fault("model line after Rules terminal"));
        }
        original
            .make_from_view(&view, *movement)
            .map_err(|_| fault("illegal original model line"))?;
    }
    let view = original.ordered_legal_moves();
    if model.len() != root.line_plies
        && original
            .play_status_from_view(&view)
            .map_err(|_| fault("model endpoint Rules status"))?
            == PlayStatus::Ongoing
    {
        return Err(fault("partial original model endpoint"));
    }
    drop(original);
    let own = root.root.side_to_move();
    let mut state = root.root.clone();
    let mut changed_own = false;
    for (ply, movement) in repair.iter().enumerate() {
        control(root, cancel)?;
        let view = state.ordered_legal_moves();
        let ongoing = state
            .play_status_from_view(&view)
            .map_err(|_| fault("Repair Rules status"))?
            == PlayStatus::Ongoing;
        if !ongoing {
            return Err(fault("Repair line after Rules terminal"));
        }
        if changed_own && state.side_to_move() != own {
            if Some(ply) != evidence.opponent_anchor_ply {
                return Err(fault("reported anchor skips first eligible opponent turn"));
            }
            control(root, cancel)?;
            return Ok(ply);
        }
        if ply > attack && state.side_to_move() == own && model.get(ply) != Some(movement) {
            changed_own = true;
        }
        state
            .make_from_view(&view, *movement)
            .map_err(|_| fault("illegal Repair anchor path"))?;
    }
    Err(fault(
        "reported Repair has no first eligible opponent anchor",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    fn moves(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|m| BoardMove::from_uci(m).unwrap())
            .collect()
    }
    fn fixture() -> (ReplayResultRoot, ReplayQueryPrior, RepairAnchorEvidence) {
        let root = ReplayResultRoot {
            root: rz_position::Position::startpos(),
            prefix: moves("e2e4"),
            line_plies: 6,
            whole_deadline: Instant::now() + Duration::from_secs(5),
        };
        let mut prior = ReplayQueryPrior {
            outcome: PriorOutcome::CompletedOpponentEndpoint,
            opponent_anchor_ply: Some(3),
            endpoint_observed: true,
            opponent_role_steps: 3,
            accepted_opponent_steps: 3,
            ..ReplayQueryPrior::default()
        };
        prior
            .repaired_line
            .capture(&moves("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6"), MAX_LINE_MOVES)
            .unwrap();
        prior
            .opponent_counterline
            .capture(&moves("e2e4 e7e5 g1f3 g8f6 f1c4 f6e4"), MAX_LINE_MOVES)
            .unwrap();
        let mut evidence = RepairAnchorEvidence {
            repair_record_observed: true,
            repair_record_revision: Some(7),
            opponent_anchor_ply: Some(3),
            ..RepairAnchorEvidence::default()
        };
        evidence
            .model_counterline
            .capture(&moves("e2e4 e7e5 d2d4 b8c6 b1c3 g8f6"), MAX_LINE_MOVES)
            .unwrap();
        evidence
            .repaired_line
            .capture(&moves("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6"), MAX_LINE_MOVES)
            .unwrap();
        (root, prior, evidence)
    }
    #[test]
    fn reported_repair_recomputes_first_own_change_then_opponent_anchor() {
        let (root, prior, evidence) = fixture();
        let before = root.root.snapshot();
        assert_eq!(
            check_anchor(&root, &prior, &evidence, &AtomicBool::new(false)).unwrap(),
            3
        );
        assert!(before.same_state(&root.root.snapshot()));
        assert_eq!(evidence.actual_utility_groups, 0);
        assert_eq!(evidence.authorities, ReplayAuthorities::default());
    }
    #[test]
    fn reported_repair_black_own_turn_uses_original_rules_side_and_history() {
        let (mut root, mut prior, mut evidence) = fixture();
        let view = root.root.ordered_legal_moves();
        root.root
            .make_from_view(&view, BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        root.prefix = moves("e7e5");
        let before = root.root.snapshot();
        assert_eq!(root.root.side_to_move(), rz_position::Color::Black);
        assert_eq!(root.root.known_history_len(), 2);
        let repair = moves("e7e5 g1f3 g8f6 f1b5 b8c6 b5a4");
        prior
            .repaired_line
            .capture(&repair, MAX_LINE_MOVES)
            .unwrap();
        evidence
            .repaired_line
            .capture(&repair, MAX_LINE_MOVES)
            .unwrap();
        evidence
            .model_counterline
            .capture(&moves("e7e5 g1f3 b8c6 f1b5 a7a6 b5a4"), MAX_LINE_MOVES)
            .unwrap();
        prior
            .opponent_counterline
            .capture(&moves("e7e5 g1f3 g8f6 f1c4 b8c6 d2d3"), MAX_LINE_MOVES)
            .unwrap();
        assert_eq!(
            check_anchor(&root, &prior, &evidence, &AtomicBool::new(false)).unwrap(),
            3
        );
        assert!(before.same_state(&root.root.snapshot()));
        assert_eq!(root.root.known_history_len(), 2);
    }
    #[test]
    fn reported_repair_short_original_model_mate_is_rules_terminal_not_partial() {
        let (mut root, mut prior, mut evidence) = fixture();
        root.prefix = moves("f2f3");
        let repair = moves("f2f3 e7e5 g2g3 d8f6 e2e3 f6g6");
        prior
            .repaired_line
            .capture(&repair, MAX_LINE_MOVES)
            .unwrap();
        evidence
            .repaired_line
            .capture(&repair, MAX_LINE_MOVES)
            .unwrap();
        prior
            .opponent_counterline
            .capture(&moves("f2f3 e7e5 g2g3 d8h4 e2e3 h4g5"), MAX_LINE_MOVES)
            .unwrap();
        evidence
            .model_counterline
            .capture(&moves("f2f3 e7e5 g2g4 d8h4"), MAX_LINE_MOVES)
            .unwrap();
        assert_eq!(
            check_anchor(&root, &prior, &evidence, &AtomicBool::new(false)).unwrap(),
            3
        );
        evidence
            .model_counterline
            .capture(&moves("f2f3 e7e5 g2g4 d8h4 a2a3"), MAX_LINE_MOVES)
            .unwrap();
        assert!(check_anchor(&root, &prior, &evidence, &AtomicBool::new(false)).is_err());
    }
    #[test]
    fn reported_repair_refuses_later_anchor_original_c_illegal_or_partial_model_line() {
        for case in 0..6 {
            let (root, mut prior, mut evidence) = fixture();
            match case {
                0 => {
                    prior.opponent_anchor_ply = Some(5);
                    evidence.opponent_anchor_ply = Some(5);
                }
                1 => evidence
                    .model_counterline
                    .capture(&moves("e2e4 e7e6 d2d4 d7d5 b1c3 g8f6"), MAX_LINE_MOVES)
                    .unwrap(),
                2 => evidence
                    .model_counterline
                    .capture(&moves("e2e4 e7e5 f1f3 b8c6 b1c3 g8f6"), MAX_LINE_MOVES)
                    .unwrap(),
                3 => evidence
                    .model_counterline
                    .capture(&moves("e2e4 e7e5 d2d4 b8c6"), MAX_LINE_MOVES)
                    .unwrap(),
                4 => evidence
                    .model_counterline
                    .capture(&moves("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6"), MAX_LINE_MOVES)
                    .unwrap(),
                _ => evidence.repair_record_revision = None,
            }
            assert!(
                check_anchor(&root, &prior, &evidence, &AtomicBool::new(false)).is_err(),
                "case {case}"
            );
        }
    }
    #[test]
    fn reported_repair_first_anchor_check_never_extends_original_window() {
        let (mut root, prior, evidence) = fixture();
        assert!(check_anchor(&root, &prior, &evidence, &AtomicBool::new(true)).is_err());
        root.whole_deadline = Instant::now();
        assert!(check_anchor(&root, &prior, &evidence, &AtomicBool::new(false)).is_err());
    }
    #[test]
    fn reported_repair_original_private_preparation_checks_binding_without_native_promotion() {
        let prepared = crate::pals_cpu_task::strategic_action::replay_inputs::tests::controlled_prepared_result_rules();
        let build = || {
            let mut o = native_fixture();
            o.input_admission = prepared.audit().input_admission.clone();
            o.original_whole_wall_ms = o.input_admission.whole_wall_ms;
            o.cleanup_reserve_ms = o.input_admission.cleanup_reserve_ms;
            let p = o.query_prior.as_mut().unwrap();
            p.repaired_line
                .capture(&moves("e2e4 e7e5 g1f3 b8c6"), MAX_LINE_MOVES)
                .unwrap();
            p.opponent_counterline
                .capture(&moves("e2e4 e7e5 g1f3 g8f6"), MAX_LINE_MOVES)
                .unwrap();
            p.opponent_role_steps = 1;
            p.accepted_opponent_steps = 1;
            let e = o.repair_anchor_evidence.as_mut().unwrap();
            e.model_counterline
                .capture(&moves("e2e4 e7e5 d2d4 b8c6"), MAX_LINE_MOVES)
                .unwrap();
            e.repaired_line
                .capture(&moves("e2e4 e7e5 g1f3 b8c6"), MAX_LINE_MOVES)
                .unwrap();
            let readiness = consume_native_repair_replay_prior(
                &o,
                &super::super::super::tests::controlled_expected(&o),
                &o.asset_profile_artifact,
                &super::super::super::super::tests::profile(false),
            )
            .unwrap()
            .readiness();
            o.query_prior.as_mut().unwrap().readiness = readiness;
            reported(serde_json::to_value(&o).unwrap(), &o).unwrap()
        };
        let checked = build()
            .check_original_rules(&prepared, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(checked.first_eligible_anchor_ply(), 3);
        assert_eq!(checked.reported_record_revision(), 7);
        assert_eq!(
            checked.rules().original_input_artifact(),
            prepared.artifact()
        );
        assert!(
            checked
                .assurance_scope()
                .contains("pending_native_witness_and_caller_chronology")
        );
        let mut changed = build();
        changed.report.original_binding.registration.sha256 = "0".repeat(64);
        assert!(
            changed
                .check_original_rules(&prepared, &AtomicBool::new(false))
                .is_err()
        );
        assert!(
            build()
                .check_original_rules(&prepared, &AtomicBool::new(true))
                .is_err()
        );
    }
    fn native_fixture() -> NativeReplayObservation {
        let (_, prior, evidence) = fixture();
        let mut o = super::super::super::tests::fixture();
        o.schema = super::super::super::super::REPAIR_ANCHOR_OBSERVATION_SCHEMA;
        let p = o.query_prior.as_mut().unwrap();
        p.repaired_line = prior.repaired_line;
        p.opponent_counterline = prior.opponent_counterline;
        p.opponent_role_steps = 3;
        p.accepted_opponent_steps = 3;
        o.repair_anchor_evidence = Some(evidence);
        let readiness = consume_native_repair_replay_prior(
            &o,
            &super::super::super::tests::controlled_expected(&o),
            &o.asset_profile_artifact,
            &super::super::super::super::tests::profile(false),
        )
        .unwrap()
        .readiness();
        o.query_prior.as_mut().unwrap().readiness = readiness;
        o
    }
    fn reported(
        body: Value,
        o: &NativeReplayObservation,
    ) -> Result<ReportedRepairConsistency, AdmissionFault> {
        check_reported_repair_consistency(
            body,
            &super::super::super::tests::controlled_expected(o),
            &o.asset_profile_artifact,
            &super::super::super::super::tests::profile(false),
            &super::super::super::super::super::pin(include_bytes!("replay_prior.rs")),
            &super::super::super::super::super::pin(include_bytes!("replay_repair_evidence.rs")),
        )
    }
    #[test]
    fn reported_repair_new_lane_is_closed_and_old_consumers_refuse_it() {
        let o = native_fixture();
        let value = serde_json::to_value(&o).unwrap();
        let checked = reported(value.clone(), &o).unwrap();
        assert_eq!(checked.reported_record_revision(), Some(7));
        assert!(
            checked
                .assurance_scope()
                .contains("pending_original_rules_native_witness")
        );
        assert!(
            consume_native_replay_prior(
                &o,
                &super::super::super::tests::controlled_expected(&o),
                &o.asset_profile_artifact,
                &super::super::super::super::tests::profile(false)
            )
            .is_err()
        );
        assert!(
            check_reported_prior_consistency(
                value,
                &super::super::super::tests::controlled_expected(&o),
                &o.asset_profile_artifact,
                &super::super::super::super::tests::profile(false),
                &super::super::super::super::super::pin(include_bytes!("replay_prior.rs"))
            )
            .is_err()
        );
    }
    #[test]
    fn reported_repair_source_shape_revision_and_authority_substitution_are_refused() {
        let o = native_fixture();
        for case in 0..8 {
            let mut body = serde_json::to_value(&o).unwrap();
            let wire = &mut body["repair_anchor_evidence"];
            match case {
                0 => {
                    wire["source_sha256"][0] = serde_json::json!(
                        o.repair_anchor_evidence.as_ref().unwrap().source_sha256[0] ^ 1
                    )
                }
                1 => wire["source_bytes"] = serde_json::json!(0),
                2 => wire["repair_record_revision"] = Value::Null,
                3 => wire["opponent_anchor_ply"] = serde_json::json!(5),
                4 => wire["repaired_line"][0] = serde_json::json!(0),
                5 => wire["actual_utility_groups"] = serde_json::json!(1),
                6 => wire["hidden_native_authority"] = serde_json::json!(true),
                _ => wire["model_counterline"] = serde_json::json!(vec![0; 17]),
            }
            assert!(reported(body, &o).is_err(), "case {case}");
        }
        let mut expected_source =
            super::super::super::super::super::pin(include_bytes!("replay_repair_evidence.rs"));
        expected_source.sha256 = "0".repeat(64);
        assert!(
            check_reported_repair_consistency(
                serde_json::to_value(&o).unwrap(),
                &super::super::super::tests::controlled_expected(&o),
                &o.asset_profile_artifact,
                &super::super::super::super::tests::profile(false),
                &super::super::super::super::super::pin(include_bytes!("replay_prior.rs")),
                &expected_source
            )
            .is_err()
        );
    }
}
