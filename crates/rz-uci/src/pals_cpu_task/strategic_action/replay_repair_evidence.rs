//! Small actual-owner origin projection. No portable record handle, V feature,
//! imported native capability, CPU/model rerun or new execution window.
use super::*;
use rz_search::pals::engine::replay::ReplayError;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

pub const REPAIR_EVIDENCE_SCHEMA: &str = "rz-pals-replay-repair-anchor-evidence/1";
pub const REPAIR_EVIDENCE_SCOPE: &str =
    "actual_owner_repair_origin_pending_caller_rules_and_chronology";
pub fn repair_evidence_source_digest() -> [u8; 32] {
    Sha256::digest(include_bytes!("replay_repair_evidence.rs")).into()
}
pub fn repair_evidence_source_bytes() -> u64 {
    include_bytes!("replay_repair_evidence.rs").len() as u64
}

/// Constructed only by the actual producer. Wire parsing is separately labelled
/// as a report and never yields this owner observation to a caller.
#[derive(Debug, Serialize)]
pub struct RepairAnchorEvidence {
    pub(super) schema: &'static str,
    pub(super) assurance_scope: &'static str,
    pub(super) source_sha256: [u8; 32],
    pub(super) source_bytes: u64,
    pub(super) model_counterline: PackedMoves,
    pub(super) repaired_line: PackedMoves,
    pub(super) repair_record_observed: bool,
    pub(super) repair_record_revision: Option<u64>,
    pub(super) opponent_anchor_ply: Option<usize>,
    pub(super) actual_utility_groups: u8,
    pub(super) authorities: ReplayAuthorities,
}
impl Default for RepairAnchorEvidence {
    fn default() -> Self {
        Self {
            schema: REPAIR_EVIDENCE_SCHEMA,
            assurance_scope: REPAIR_EVIDENCE_SCOPE,
            source_sha256: repair_evidence_source_digest(),
            source_bytes: repair_evidence_source_bytes(),
            model_counterline: PackedMoves::default(),
            repaired_line: PackedMoves::default(),
            repair_record_observed: false,
            repair_record_revision: None,
            opponent_anchor_ply: None,
            actual_utility_groups: 0,
            authorities: ReplayAuthorities::default(),
        }
    }
}
impl RepairAnchorEvidence {
    /// Reads the live owner's actual accepted record in original W. Fixed move
    /// arrays were reserved before model loading. No graph or role context copy.
    pub(in super::super) fn capture_owner<M: RoleModel>(
        &mut self,
        owner: &FreshReplayOwner<M>,
        whole: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), ReplayError> {
        let origin = owner.opponent_repair_origin(whole, cancel)?;
        // Owner has already checked W/cancellation, including the no-anchor case.
        self.model_counterline
            .capture(owner.model_counterline(), MAX_LINE_MOVES)
            .map_err(|_| ReplayError::InvalidPlan("bounded original model counterline"))?;
        if let Some(origin) = origin {
            self.repaired_line
                .capture(origin.repaired_line(), MAX_LINE_MOVES)
                .map_err(|_| ReplayError::InvalidPlan("bounded actual Repair line"))?;
            self.repair_record_revision = Some(origin.repair_record_revision());
            self.opponent_anchor_ply = Some(origin.anchor_ply());
            self.repair_record_observed = true;
        }
        Ok(())
    }
    pub(super) fn check_projection(&self, p: &ReplayQueryPrior) -> Result<(), AdmissionFault> {
        if self.schema != REPAIR_EVIDENCE_SCHEMA
            || self.assurance_scope != REPAIR_EVIDENCE_SCOPE
            || self.source_bytes == 0
            || self.source_bytes > 512 * 1024
            || self.authorities != ReplayAuthorities::default()
            || self.actual_utility_groups != 0
            || self.model_counterline.len > MAX_LINE_MOVES
            || self.repaired_line.len > MAX_LINE_MOVES
            || self.opponent_anchor_ply != p.opponent_anchor_ply
        {
            return Err(fault(
                "repair evidence closed source/extent/authority/anchor",
            ));
        }
        if self.repair_record_observed {
            if self.repair_record_revision.is_none_or(|r| r == 0)
                || self.opponent_anchor_ply.is_none()
                || self.model_counterline.len == 0
                || self.repaired_line.as_slice() != p.repaired_line.as_slice()
            {
                return Err(fault("repair evidence record/revision/actual line"));
            }
        } else if self.repair_record_revision.is_some()
            || self.opponent_anchor_ply.is_some()
            || self.repaired_line.len != 0
        {
            return Err(fault("repair evidence false record absence"));
        }
        Ok(())
    }
}

/// Actual producer-side check. The old /3 consumer continues to reject /4.
/// Does not certify the external child's loaded image or next Query chronology.
pub fn consume_native_repair_replay_prior<'a>(
    observation: &'a NativeReplayObservation,
    expected: &ReplayExpectedPins,
    profile_pin: &super::super::ArtifactPin,
    profile: &CpuFreshAssetProfile,
) -> Result<CheckedNativeReplayPrior<'a>, AdmissionFault> {
    let checked = consume_native_replay_prior_schema(
        observation,
        expected,
        profile_pin,
        profile,
        super::super::REPAIR_ANCHOR_OBSERVATION_SCHEMA,
    )?;
    let evidence = observation
        .repair_anchor_evidence
        .as_ref()
        .ok_or_else(|| fault("explicit Repair origin evidence missing"))?;
    if evidence.source_sha256 != repair_evidence_source_digest()
        || evidence.source_bytes != repair_evidence_source_bytes()
    {
        return Err(fault("actual Repair evidence source differs"));
    }
    evidence.check_projection(checked.projection())?;
    Ok(checked)
}
