//! PALS data and training-lifecycle preparation. This module never runs an optimizer.
//!
//! Exact chess legality remains with Rules. These persisted records preserve the
//! exact snapshot supplied by its owner, rather than reimplementing chess here.
use crate::{ManifestError, decode_json, digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PALS_DATA_DOMAIN: &str = "rz-pals-data/1";
pub const PALS_CHECKPOINT_DOMAIN: &str = "rz-pals-training-checkpoint/1";
const MAX_HISTORY: usize = 16_384;
const MAX_RECORDS: usize = 65_536;

fn ensure(ok: bool, reason: &str) -> Result<(), ManifestError> {
    if ok {
        Ok(())
    } else {
        Err(ManifestError::Integrity(reason.into()))
    }
}
fn identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}
fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn canonical_sha<T: Serialize>(domain: &str, value: &T) -> Result<String, ManifestError> {
    let bytes = serde_json::to_vec(&(domain, value))
        .map_err(|e| ManifestError::Integrity(e.to_string()))?;
    Ok(digest(&bytes))
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsDataRole {
    Proposer,
    Critic,
    Verifier,
}

/// Public records available at capture time. Future labels and V controls are
/// structurally absent from this type and therefore from its input digest.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPublicRecord {
    pub observation_sha256: String,
    pub situation_revision: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsInputSnapshot {
    pub game_id: String,
    pub opening_id: String,
    pub line_genealogy_id: String,
    /// Complete initial state plus actual played moves, supplied by Rules/UCI.
    pub position_command: String,
    pub board_fen: String,
    pub actual_history: Vec<u16>,
    pub rules_state_sha256: String,
    pub rules_history_sha256: String,
    /// Rules-provided grouping identity for transpositions, excluding clocks
    /// and move genealogy. Used only for leakage checks, never cache reuse.
    pub transposition_sha256: String,
    pub encoding_sha256: String,
    pub model_configuration_sha256: String,
    pub model_weights_sha256: String,
    pub frozen_epoch: u64,
    pub input_revision: u64,
    pub capture_sequence: u64,
    pub white_to_move: bool,
    pub role: PalsDataRole,
    pub legal_moves: Vec<u16>,
    pub public_records: Vec<PalsPublicRecord>,
}
impl PalsInputSnapshot {
    pub fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            [&self.game_id, &self.opening_id, &self.line_genealogy_id]
                .into_iter()
                .all(|s| identifier(s)),
            "invalid data identity",
        )?;
        ensure(
            self.position_command.starts_with("position ")
                && self.position_command.len() <= 256 * 1024
                && !self.position_command.chars().any(char::is_control),
            "invalid or unbounded position snapshot",
        )?;
        ensure(
            !self.board_fen.is_empty()
                && self.board_fen.len() <= 512
                && !self.board_fen.chars().any(char::is_control),
            "invalid FEN snapshot",
        )?;
        ensure(
            self.board_fen.split_whitespace().nth(1)
                == Some(if self.white_to_move { "w" } else { "b" }),
            "captured FEN side-to-move mismatch",
        )?;
        ensure(
            self.actual_history.len() <= MAX_HISTORY
                && self.legal_moves.len() <= 256
                && self.public_records.len() <= 128,
            "snapshot allocation limit",
        )?;
        ensure(
            [
                &self.rules_state_sha256,
                &self.rules_history_sha256,
                &self.transposition_sha256,
                &self.encoding_sha256,
                &self.model_configuration_sha256,
                &self.model_weights_sha256,
            ]
            .into_iter()
            .all(|s| sha(s)),
            "invalid snapshot digest",
        )?;
        ensure(
            self.legal_moves
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                == self.legal_moves.len(),
            "duplicate legal move",
        )?;
        ensure(
            self.public_records
                .iter()
                .all(|r| sha(&r.observation_sha256) && r.situation_revision <= self.input_revision),
            "public record was unavailable at input revision",
        )
    }
    pub fn seal(self) -> Result<PalsFrozenInput, ManifestError> {
        self.validate()?;
        let sha256 = canonical_sha(PALS_DATA_DOMAIN, &self)?;
        Ok(PalsFrozenInput {
            snapshot: self,
            sha256,
        })
    }
}

/// In-memory callers can inspect the snapshot. Every persistence/label boundary
/// verifies the seal so a changed historical input cannot masquerade as original.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsFrozenInput {
    snapshot: PalsInputSnapshot,
    sha256: String,
}
impl PalsFrozenInput {
    pub fn snapshot(&self) -> &PalsInputSnapshot {
        &self.snapshot
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn verify(&self) -> Result<(), ManifestError> {
        self.snapshot.validate()?;
        ensure(
            self.sha256 == canonical_sha(PALS_DATA_DOMAIN, &self.snapshot)?,
            "historical input seal mismatch",
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsOutcome {
    WhiteWin,
    BlackWin,
    Draw,
    Unknown,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsGameEnd {
    Checkmate,
    Stalemate,
    Repetition,
    FiftyMove,
    SeventyFiveMove,
    InsufficientMaterial,
    DeadPosition,
    TimeForfeit,
    IllegalMove,
    EngineCrash,
    PlyLimit,
    WallTimeLimit,
    InfrastructureFailure,
    UserStop,
    Unresolved,
}
impl PalsGameEnd {
    pub fn supplies_outcome(self) -> bool {
        !matches!(
            self,
            Self::PlyLimit
                | Self::WallTimeLimit
                | Self::InfrastructureFailure
                | Self::UserStop
                | Self::Unresolved
        )
    }
    fn rules_terminal(self) -> bool {
        matches!(
            self,
            Self::Checkmate
                | Self::Stalemate
                | Self::Repetition
                | Self::FiftyMove
                | Self::SeventyFiveMove
                | Self::InsufficientMaterial
                | Self::DeadPosition
        )
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsGameResult {
    pub game_id: String,
    pub outcome: PalsOutcome,
    pub ending: PalsGameEnd,
    pub raw_evidence_sha256: String,
}
impl PalsGameResult {
    pub fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            identifier(&self.game_id) && sha(&self.raw_evidence_sha256),
            "invalid game result evidence",
        )?;
        ensure(
            self.ending.supplies_outcome() == (self.outcome != PalsOutcome::Unknown),
            "interrupted game cannot supply a value target",
        )?;
        ensure(
            !matches!(
                self.ending,
                PalsGameEnd::Stalemate
                    | PalsGameEnd::Repetition
                    | PalsGameEnd::FiftyMove
                    | PalsGameEnd::SeventyFiveMove
                    | PalsGameEnd::InsufficientMaterial
                    | PalsGameEnd::DeadPosition
            ) || self.outcome == PalsOutcome::Draw,
            "draw termination requires a draw outcome",
        )?;
        ensure(
            self.ending != PalsGameEnd::Checkmate
                || matches!(self.outcome, PalsOutcome::WhiteWin | PalsOutcome::BlackWin),
            "checkmate requires a winner",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsTargetProvenance {
    OwnedCpu {
        engine_sha256: String,
        profile_sha256: String,
        task_sha256: String,
        completed_depth: u32,
        nodes: u64,
        raw_evidence_sha256: String,
    },
    RulesTerminal {
        rules_state_sha256: String,
        result: PalsGameResult,
    },
    ActualGame {
        result: PalsGameResult,
    },
    /// Persisted as a rejected source instead of silently relabeling it as own CPU.
    ExternalTeacher {
        identity: String,
    },
}
impl PalsTargetProvenance {
    fn validate(&self, input: &PalsInputSnapshot) -> Result<(), ManifestError> {
        match self {
            Self::OwnedCpu {
                engine_sha256,
                profile_sha256,
                task_sha256,
                raw_evidence_sha256,
                completed_depth,
                ..
            } => ensure(
                [
                    engine_sha256,
                    profile_sha256,
                    task_sha256,
                    raw_evidence_sha256,
                ]
                .into_iter()
                .all(|s| sha(s))
                    && *completed_depth > 0,
                "invalid owned CPU evidence",
            ),
            Self::RulesTerminal {
                rules_state_sha256,
                result,
            } => {
                result.validate()?;
                ensure(
                    !matches!(
                        result.ending,
                        PalsGameEnd::Checkmate | PalsGameEnd::Stalemate
                    ) || input.legal_moves.is_empty(),
                    "terminal mate/stalemate input still has legal moves",
                )?;
                ensure(
                    rules_state_sha256 == &input.rules_state_sha256
                        && result.game_id == input.game_id
                        && result.ending.rules_terminal(),
                    "terminal label must describe this exact Rules state",
                )
            }
            Self::ActualGame { result } => {
                result.validate()?;
                ensure(
                    result.game_id == input.game_id,
                    "outcome belongs to a different game",
                )
            }
            Self::ExternalTeacher { .. } => Err(ManifestError::Integrity(
                "initial PALS data rejects external teachers".into(),
            )),
        }
    }
    fn result(&self) -> Option<&PalsGameResult> {
        match self {
            Self::RulesTerminal { result, .. } | Self::ActualGame { result } => Some(result),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsPolicyTarget {
    pub moves: Vec<u16>,
    pub probabilities: Vec<f64>,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsConditionalValidity {
    SupportedAfterRepair,
    RefutedByRepair,
    NotExamined,
    Disputed,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCounterexampleTarget {
    pub challenged_line_sha256: String,
    pub divergence_ply: u32,
    pub response_line: Vec<u16>,
    pub repair_line: Option<Vec<u16>>,
    pub validity: PalsConditionalValidity,
    pub input_revision: u64,
    /// Rules-owner attestation to sequential legality, not a strategic proof.
    pub legality_evidence_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsVerifierTaskKind {
    DefendResponse,
    AttackRepair,
    WidenResponses,
    LowerSelectivity,
    ResumeTask,
    CrossProfileRecheck,
    Defer,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsVerifierTaskTarget {
    pub task: PalsVerifierTaskKind,
    pub branch_sha256: String,
    pub cpu_profile_sha256: String,
    pub budget_bucket: u32,
    /// None keeps unresolved utility masked. CPU/model agreement is not a rank.
    pub preference_rank: Option<u32>,
    pub information_gain_evidence_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsFutureLabel {
    pub observed_sequence: u64,
    pub provenance: PalsTargetProvenance,
    pub policy: Option<PalsPolicyTarget>,
    /// Strictly one-hot actual outcome, ordered W/D/L from captured side-to-move.
    pub value_wdl: Option<[f64; 3]>,
    pub white_to_move: bool,
    pub counterexample: Option<PalsCounterexampleTarget>,
    pub verifier_tasks: Option<Vec<PalsVerifierTaskTarget>>,
    pub supersedes_label_sha256: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsVerifierPrivate {
    pub task_kind: String,
    pub control_sha256: String,
    pub private_latent: Vec<f32>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsLearningRecord {
    pub input: PalsFrozenInput,
    pub future_label: Option<PalsFutureLabel>,
    pub verifier_private: Option<PalsVerifierPrivate>,
}
impl PalsLearningRecord {
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.input.verify()?;
        let input = self.input.snapshot();
        if let Some(private) = &self.verifier_private {
            ensure(
                input.role == PalsDataRole::Verifier
                    && identifier(&private.task_kind)
                    && sha(&private.control_sha256)
                    && private.private_latent.len() <= 16 * 384
                    && private.private_latent.iter().all(|v| v.is_finite()),
                "V private data cannot enter P/C records",
            )?;
        }
        let Some(label) = &self.future_label else {
            return Ok(());
        };
        ensure(
            label.observed_sequence >= input.capture_sequence,
            "future label precedes its immutable input",
        )?;
        ensure(
            label.white_to_move == input.white_to_move,
            "label viewpoint differs from captured state",
        )?;
        label.provenance.validate(input)?;
        ensure(
            label
                .supersedes_label_sha256
                .as_ref()
                .is_none_or(|s| sha(s)),
            "invalid label revision digest",
        )?;
        if let Some(policy) = &label.policy {
            ensure(
                input.role != PalsDataRole::Verifier,
                "V task ranking is not a legal-move policy head",
            )?;
            ensure(
                policy.moves == input.legal_moves
                    && policy.moves.len() == policy.probabilities.len()
                    && !policy.moves.is_empty(),
                "policy target legal order mismatch",
            )?;
            ensure(
                policy
                    .probabilities
                    .iter()
                    .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
                    && (policy.probabilities.iter().sum::<f64>() - 1.0).abs() <= 1e-6,
                "invalid policy target",
            )?;
        }
        if let Some(value) = &label.value_wdl {
            let result = label.provenance.result().ok_or_else(|| {
                ManifestError::Integrity("CPU estimates cannot masquerade as outcome WDL".into())
            })?;
            ensure(
                result.outcome != PalsOutcome::Unknown,
                "unresolved outcome requires a masked value target",
            )?;
            let win = matches!(
                (result.outcome, label.white_to_move),
                (PalsOutcome::WhiteWin, true) | (PalsOutcome::BlackWin, false)
            );
            let expected = if result.outcome == PalsOutcome::Draw {
                [0.0, 1.0, 0.0]
            } else if win {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            ensure(value == &expected, "outcome WDL or viewpoint mismatch")?;
        }
        if let Some(c) = &label.counterexample {
            ensure(
                input.role == PalsDataRole::Critic
                    && sha(&c.challenged_line_sha256)
                    && sha(&c.legality_evidence_sha256)
                    && c.input_revision == input.input_revision
                    && !c.response_line.is_empty()
                    && c.response_line.len() <= 256
                    && c.repair_line
                        .as_ref()
                        .is_none_or(|l| !l.is_empty() && l.len() <= 256),
                "invalid conditional counterexample target",
            )?;
            ensure(
                !matches!(
                    c.validity,
                    PalsConditionalValidity::SupportedAfterRepair
                        | PalsConditionalValidity::RefutedByRepair
                ) || c.repair_line.is_some(),
                "repair-dependent label requires the actual repair",
            )?;
        }
        if let Some(tasks) = &label.verifier_tasks {
            ensure(
                input.role == PalsDataRole::Verifier && !tasks.is_empty() && tasks.len() <= 64,
                "invalid verifier task target extent",
            )?;
            for task in tasks {
                ensure(
                    sha(&task.branch_sha256)
                        && sha(&task.cpu_profile_sha256)
                        && task.budget_bucket <= 16
                        && task
                            .preference_rank
                            .is_none_or(|rank| rank < tasks.len() as u32)
                        && task
                            .information_gain_evidence_sha256
                            .as_ref()
                            .is_none_or(|s| sha(s)),
                    "invalid verifier task ranking",
                )?;
                ensure(
                    task.preference_rank.is_none()
                        || task.information_gain_evidence_sha256.is_some(),
                    "ranked verifier task needs observed information-gain evidence",
                )?;
            }
        }
        Ok(())
    }
    pub fn to_json(&self) -> Result<String, ManifestError> {
        self.validate()?;
        serde_json::to_string(self).map_err(|e| ManifestError::Integrity(e.to_string()))
    }
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let result: Self = decode_json(input)?;
        result.validate()?;
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum PalsSplit {
    Train,
    Validation,
    Holdout,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsDatasetSplit {
    /// One assignment per complete game; openings and line ancestry may not cross.
    pub games: BTreeMap<String, PalsSplit>,
}
impl PalsDatasetSplit {
    pub fn validate(&self, records: &[PalsLearningRecord]) -> Result<(), ManifestError> {
        ensure(
            !records.is_empty() && records.len() <= MAX_RECORDS && self.games.len() <= MAX_RECORDS,
            "dataset record limit",
        )?;
        let mut openings = BTreeMap::new();
        let mut genealogies = BTreeMap::new();
        let mut states = BTreeMap::new();
        let mut transpositions = BTreeMap::new();
        let mut present = BTreeSet::new();
        for record in records {
            record.validate()?;
            let s = record.input.snapshot();
            let split = self
                .games
                .get(&s.game_id)
                .ok_or_else(|| ManifestError::Integrity("unassigned game".into()))?;
            present.insert(s.game_id.as_str());
            for (map, identity) in [
                (&mut openings, &s.opening_id),
                (&mut genealogies, &s.line_genealogy_id),
                (&mut states, &s.rules_state_sha256),
                (&mut transpositions, &s.transposition_sha256),
            ] {
                ensure(
                    map.insert(identity.clone(), *split)
                        .is_none_or(|prior| prior == *split),
                    "game/opening/line or duplicate state leaks across splits",
                )?;
            }
        }
        ensure(
            self.games.keys().all(|g| present.contains(g.as_str())),
            "split plan contains unobserved games",
        )
    }
}

/// Authority supplied by the collector after verifying actual asset hashes.
/// A row's own provenance declaration cannot enroll a new engine or model.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsOwnedSources {
    pub cpu_binary_sha256: BTreeSet<String>,
    pub model_weights_sha256: BTreeSet<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsDatasetAudit {
    pub records: u64,
    pub labeled_records: u64,
    pub masked_value_records: u64,
    pub training_records: u64,
    pub validation_records: u64,
    pub holdout_records: u64,
    pub canonical_dataset_sha256: String,
    pub canonical_split_sha256: String,
}
impl PalsOwnedSources {
    pub fn audit(
        &self,
        records: &[PalsLearningRecord],
        split: &PalsDatasetSplit,
    ) -> Result<PalsDatasetAudit, ManifestError> {
        ensure(
            !self.cpu_binary_sha256.is_empty()
                && self.cpu_binary_sha256.len() <= 64
                && !self.model_weights_sha256.is_empty()
                && self.model_weights_sha256.len() <= 256
                && self
                    .cpu_binary_sha256
                    .iter()
                    .chain(self.model_weights_sha256.iter())
                    .all(|s| sha(s)),
            "invalid owned source registry",
        )?;
        split.validate(records)?;
        let mut report = PalsDatasetAudit {
            records: records.len() as u64,
            labeled_records: 0,
            masked_value_records: 0,
            training_records: 0,
            validation_records: 0,
            holdout_records: 0,
            canonical_dataset_sha256: canonical_sha("rz-pals-dataset/1", &records)?,
            canonical_split_sha256: canonical_sha("rz-pals-split/1", split)?,
        };
        for record in records {
            ensure(
                self.model_weights_sha256
                    .contains(&record.input.snapshot().model_weights_sha256),
                "unregistered model weights in dataset",
            )?;
            match split.games[&record.input.snapshot().game_id] {
                PalsSplit::Train => report.training_records += 1,
                PalsSplit::Validation => report.validation_records += 1,
                PalsSplit::Holdout => report.holdout_records += 1,
            }
            if let Some(label) = &record.future_label {
                report.labeled_records += 1;
                if let PalsTargetProvenance::OwnedCpu { engine_sha256, .. } = &label.provenance {
                    ensure(
                        self.cpu_binary_sha256.contains(engine_sha256),
                        "unregistered CPU teacher in dataset",
                    )?;
                }
            }
            if record
                .future_label
                .as_ref()
                .is_none_or(|l| l.value_wdl.is_none())
            {
                report.masked_value_records += 1;
            }
        }
        Ok(report)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsComputeProfile {
    pub width: u32,
    pub latent_slots: u32,
    pub recurrent_blocks: u32,
    pub recurrent_iterations: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dimension: u32,
    pub ffn_width: u32,
    /// FMA is two FLOPs. Uncounted ops must remain visible, not silently zero.
    pub fma_flops: u32,
    pub counted_operator_sha256: String,
    pub uncounted_operators: Vec<String>,
    pub forward_flops_per_sample: u64,
    pub backward_flops_per_sample: u64,
}
impl PalsComputeProfile {
    pub fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            (
                self.width,
                self.latent_slots,
                self.recurrent_blocks,
                self.recurrent_iterations,
                self.query_heads,
                self.kv_heads,
                self.head_dimension,
                self.ffn_width,
                self.fma_flops,
            ) == (384, 16, 2, 2, 6, 2, 64, 1024, 2),
            "first PALS model configuration is fixed; resource pressure cannot shrink it",
        )?;
        ensure(
            sha(&self.counted_operator_sha256)
                && self.forward_flops_per_sample > 0
                && self.backward_flops_per_sample > 0
                && self.uncounted_operators.len() <= 64
                && self.uncounted_operators.iter().all(|s| identifier(s)),
            "invalid FLOPs cost declaration",
        )
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsTrainingBudget {
    pub max_flops: u64,
    pub max_steps: u64,
    pub max_games: u64,
    pub max_wall_time_ms: u64,
    pub max_cpu_nodes: u64,
    pub max_cpu_time_ms: u64,
    pub max_compute_units_milli: u64,
    pub max_spend_usd_micros: u64,
    pub max_output_bytes: u64,
}
impl PalsTrainingBudget {
    pub fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            [
                self.max_flops,
                self.max_steps,
                self.max_games,
                self.max_wall_time_ms,
                self.max_cpu_nodes,
                self.max_cpu_time_ms,
                self.max_compute_units_milli,
                self.max_spend_usd_micros,
                self.max_output_bytes,
            ]
            .into_iter()
            .all(|v| v > 0),
            "training requires finite registered compute/time/CPU/output/cost limits",
        )
    }
    pub fn permits(&self, usage: &PalsTrainingUsage) -> Result<(), ManifestError> {
        self.validate()?;
        let flops = usage.total_flops()?;
        ensure(
            flops <= self.max_flops
                && usage.steps <= self.max_steps
                && usage.games <= self.max_games
                && usage.wall_time_ms <= self.max_wall_time_ms
                && usage.cpu_nodes <= self.max_cpu_nodes
                && usage.cpu_time_ms <= self.max_cpu_time_ms
                && usage.compute_units_milli <= self.max_compute_units_milli
                && usage.spend_usd_micros <= self.max_spend_usd_micros
                && usage.output_bytes <= self.max_output_bytes,
            "registered training budget exhausted",
        )
    }
}
#[derive(Clone, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsTrainingUsage {
    pub forward_flops: u64,
    pub backward_flops: u64,
    pub validation_flops: u64,
    pub recompute_flops: u64,
    pub steps: u64,
    pub games: u64,
    pub wall_time_ms: u64,
    pub cpu_nodes: u64,
    pub cpu_time_ms: u64,
    pub compute_units_milli: u64,
    pub spend_usd_micros: u64,
    pub output_bytes: u64,
}
impl PalsTrainingUsage {
    pub fn total_flops(&self) -> Result<u64, ManifestError> {
        [
            self.forward_flops,
            self.backward_flops,
            self.validation_flops,
            self.recompute_flops,
        ]
        .into_iter()
        .try_fold(0_u64, |total, v| {
            total
                .checked_add(v)
                .ok_or_else(|| ManifestError::Integrity("FLOPs counter overflow".into()))
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsTrainingRecipe {
    pub optimizer: String,
    pub learning_rate: f64,
    pub betas: [f64; 2],
    pub epsilon: f64,
    pub weight_decay: f64,
    pub gradient_accumulation_steps: u32,
    pub role_order: Vec<PalsDataRole>,
    pub shared_encoder_frozen_for_verifier: bool,
    pub model: PalsComputeProfile,
    pub budget: PalsTrainingBudget,
    pub implementation_sha256: String,
    pub dataset_sha256: String,
    pub split_sha256: String,
}
impl PalsTrainingRecipe {
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.model.validate()?;
        self.budget.validate()?;
        ensure(
            self.learning_rate.is_finite()
                && self.learning_rate > 0.0
                && self.learning_rate <= 1.0
                && self
                    .betas
                    .iter()
                    .all(|b| b.is_finite() && (0.0..1.0).contains(b))
                && self.epsilon.is_finite()
                && self.epsilon > 0.0
                && self.weight_decay.is_finite()
                && self.weight_decay >= 0.0
                && self.gradient_accumulation_steps > 0,
            "invalid AdamW optimizer recipe",
        )?;
        ensure(
            self.optimizer == "adamw"
                && self.role_order
                    == [
                        PalsDataRole::Proposer,
                        PalsDataRole::Critic,
                        PalsDataRole::Verifier,
                    ]
                && self.shared_encoder_frozen_for_verifier,
            "unsupported initial role/optimizer recipe",
        )?;
        ensure(
            [
                &self.implementation_sha256,
                &self.dataset_sha256,
                &self.split_sha256,
            ]
            .into_iter()
            .all(|s| sha(s)),
            "invalid recipe provenance",
        )
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        canonical_sha("rz-pals-training-recipe/1", self)
    }
}

/// Asset references contain digests and sizes only. Storage ownership and actual
/// byte verification are performed by the collector/model asset boundary.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCheckpointAsset {
    pub sha256: String,
    pub bytes: u64,
}
impl PalsCheckpointAsset {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ManifestError> {
        ensure(!bytes.is_empty(), "empty checkpoint asset")?;
        Ok(Self {
            sha256: digest(bytes),
            bytes: bytes.len() as u64,
        })
    }
    pub fn verify_bytes(&self, bytes: &[u8]) -> Result<(), ManifestError> {
        ensure(
            self.bytes == bytes.len() as u64 && self.sha256 == digest(bytes),
            "checkpoint asset bytes differ from registered identity",
        )
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsSamplerState {
    pub algorithm: String,
    pub seed: u64,
    pub order_sha256: String,
    pub cursor: u64,
    pub dataset_rows: u64,
}
/// Complete deterministic RNG state, not just an initial seed.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRngState {
    pub algorithm: String,
    pub words: Vec<u64>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsResumeContext {
    pub model_configuration_sha256: String,
    pub recipe_sha256: String,
    pub dataset_sha256: String,
    pub split_sha256: String,
    pub frozen_epoch: u64,
    pub raw_evidence_sha256: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsTrainingCheckpoint {
    pub context: PalsResumeContext,
    pub model: PalsCheckpointAsset,
    pub optimizer_state: Option<PalsCheckpointAsset>,
    pub rng: BTreeMap<String, PalsRngState>,
    pub sampler: PalsSamplerState,
    pub usage: PalsTrainingUsage,
    pub condition_sha256: String,
}
impl PalsTrainingCheckpoint {
    pub fn prepare(
        context: PalsResumeContext,
        model: PalsCheckpointAsset,
        optimizer_state: Option<PalsCheckpointAsset>,
        rng: BTreeMap<String, PalsRngState>,
        sampler: PalsSamplerState,
        usage: PalsTrainingUsage,
    ) -> Result<Self, ManifestError> {
        let condition_sha256 = canonical_sha("rz-pals-resume-conditions/1", &context)?;
        let value = Self {
            context,
            model,
            optimizer_state,
            rng,
            sampler,
            usage,
            condition_sha256,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        let c = &self.context;
        ensure(
            [
                &c.model_configuration_sha256,
                &c.recipe_sha256,
                &c.dataset_sha256,
                &c.split_sha256,
                &c.raw_evidence_sha256,
                &self.model.sha256,
                &self.sampler.order_sha256,
            ]
            .into_iter()
            .all(|s| sha(s)),
            "invalid checkpoint provenance",
        )?;
        ensure(
            self.model.bytes > 0
                && self
                    .optimizer_state
                    .as_ref()
                    .is_none_or(|a| sha(&a.sha256) && a.bytes > 0)
                && (self.usage.steps == 0 || self.optimizer_state.is_some()),
            "checkpoint is missing model/optimizer state",
        )?;
        ensure(
            !self.rng.is_empty()
                && self.rng.len() <= 16
                && self.rng.iter().all(|(name, state)| {
                    identifier(name)
                        && identifier(&state.algorithm)
                        && !state.words.is_empty()
                        && state.words.len() <= 1024
                })
                && identifier(&self.sampler.algorithm)
                && self.sampler.dataset_rows > 0
                && self.sampler.cursor <= self.sampler.dataset_rows,
            "invalid RNG/sampler resume state",
        )?;
        self.usage.total_flops()?;
        ensure(
            self.condition_sha256 == canonical_sha("rz-pals-resume-conditions/1", c)?,
            "checkpoint resume condition mismatch",
        )
    }
    pub fn seal(self) -> Result<PalsSealedCheckpoint, ManifestError> {
        self.validate()?;
        let sha256 = canonical_sha(PALS_CHECKPOINT_DOMAIN, &self)?;
        Ok(PalsSealedCheckpoint {
            checkpoint: self,
            sha256,
        })
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsSealedCheckpoint {
    checkpoint: PalsTrainingCheckpoint,
    sha256: String,
}
impl PalsSealedCheckpoint {
    pub fn checkpoint(&self) -> &PalsTrainingCheckpoint {
        &self.checkpoint
    }
    pub fn verify_resume(
        &self,
        context: &PalsResumeContext,
        budget: &PalsTrainingBudget,
    ) -> Result<(), ManifestError> {
        self.checkpoint.validate()?;
        ensure(
            self.sha256 == canonical_sha(PALS_CHECKPOINT_DOMAIN, &self.checkpoint)?,
            "checkpoint seal mismatch",
        )?;
        ensure(
            &self.checkpoint.context == context,
            "model/recipe/dataset/epoch/evidence changed at resume",
        )?;
        budget.permits(&self.checkpoint.usage)
    }
    pub fn to_json(&self) -> Result<String, ManifestError> {
        self.checkpoint.validate()?;
        ensure(
            self.sha256 == canonical_sha(PALS_CHECKPOINT_DOMAIN, &self.checkpoint)?,
            "checkpoint seal mismatch",
        )?;
        serde_json::to_string(self).map_err(|e| ManifestError::Integrity(e.to_string()))
    }
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        value.checkpoint.validate()?;
        ensure(
            value.sha256 == canonical_sha(PALS_CHECKPOINT_DOMAIN, &value.checkpoint)?,
            "checkpoint seal mismatch",
        )?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hash(value: &str) -> String {
        digest(value.as_bytes())
    }
    fn snapshot(game: &str, opening: &str, line: &str) -> PalsInputSnapshot {
        PalsInputSnapshot {
            game_id: game.into(),
            opening_id: opening.into(),
            line_genealogy_id: line.into(),
            position_command: "position startpos moves e2e4".into(),
            board_fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1".into(),
            actual_history: vec![1804],
            rules_state_sha256: hash(game),
            rules_history_sha256: hash("history"),
            transposition_sha256: hash(game),
            encoding_sha256: hash("encoding"),
            model_configuration_sha256: hash("model"),
            model_weights_sha256: hash("weights"),
            frozen_epoch: 1,
            input_revision: 3,
            capture_sequence: 8,
            white_to_move: false,
            role: PalsDataRole::Proposer,
            legal_moves: vec![100, 200],
            public_records: vec![PalsPublicRecord {
                observation_sha256: hash("obs"),
                situation_revision: 2,
            }],
        }
    }
    fn record(game: &str, opening: &str, line: &str) -> PalsLearningRecord {
        PalsLearningRecord {
            input: snapshot(game, opening, line).seal().unwrap(),
            future_label: None,
            verifier_private: None,
        }
    }
    fn budget() -> PalsTrainingBudget {
        PalsTrainingBudget {
            max_flops: 100_000,
            max_steps: 10,
            max_games: 10,
            max_wall_time_ms: 1000,
            max_cpu_nodes: 1000,
            max_cpu_time_ms: 1000,
            max_compute_units_milli: 1000,
            max_spend_usd_micros: 1000,
            max_output_bytes: 1000,
        }
    }
    fn outcome_label(
        ending: PalsGameEnd,
        outcome: PalsOutcome,
        value: Option<[f64; 3]>,
    ) -> PalsFutureLabel {
        PalsFutureLabel {
            observed_sequence: 10,
            provenance: PalsTargetProvenance::ActualGame {
                result: PalsGameResult {
                    game_id: "a".into(),
                    outcome,
                    ending,
                    raw_evidence_sha256: hash("pgn"),
                },
            },
            policy: None,
            value_wdl: value,
            white_to_move: false,
            counterexample: None,
            verifier_tasks: None,
            supersedes_label_sha256: None,
        }
    }
    #[test]
    fn input_is_sealed_and_future_labels_do_not_change_it() {
        let mut a = record("a", "oa", "la");
        let original = a.input.sha256().to_owned();
        a.future_label = Some(outcome_label(
            PalsGameEnd::Checkmate,
            PalsOutcome::WhiteWin,
            Some([0.0, 0.0, 1.0]),
        ));
        a.validate().unwrap();
        assert_eq!(original, a.input.sha256());
        let encoded = a.to_json().unwrap();
        assert_eq!(PalsLearningRecord::from_json(&encoded).unwrap(), a);
        let corrupt = encoded.replace("e2e4", "d2d4");
        assert!(PalsLearningRecord::from_json(&corrupt).is_err());
        assert!(
            PalsLearningRecord::from_json(&encoded.replacen("{", "{\"future_label\":null,", 1))
                .is_err()
        );
    }
    #[test]
    fn interrupted_games_are_unknown_not_draw_targets() {
        let mut a = record("a", "oa", "la");
        for ending in [
            PalsGameEnd::PlyLimit,
            PalsGameEnd::WallTimeLimit,
            PalsGameEnd::InfrastructureFailure,
            PalsGameEnd::UserStop,
        ] {
            a.future_label = Some(outcome_label(ending, PalsOutcome::Unknown, None));
            a.validate().unwrap();
            a.future_label = Some(outcome_label(
                ending,
                PalsOutcome::Draw,
                Some([0.0, 1.0, 0.0]),
            ));
            assert!(a.validate().is_err());
        }
    }
    #[test]
    fn external_teacher_and_cpu_value_guess_are_rejected() {
        let mut a = record("a", "oa", "la");
        let mut label = outcome_label(PalsGameEnd::Checkmate, PalsOutcome::WhiteWin, None);
        label.provenance = PalsTargetProvenance::ExternalTeacher {
            identity: "stockfish".into(),
        };
        a.future_label = Some(label.clone());
        assert!(a.validate().is_err());
        label.provenance = PalsTargetProvenance::OwnedCpu {
            engine_sha256: hash("own"),
            profile_sha256: hash("profile"),
            task_sha256: hash("task"),
            completed_depth: 4,
            nodes: 200,
            raw_evidence_sha256: hash("raw"),
        };
        a.future_label = Some(label.clone());
        a.validate().unwrap();
        label.value_wdl = Some([1.0, 0.0, 0.0]);
        a.future_label = Some(label);
        assert!(a.validate().is_err());
    }
    #[test]
    fn verifier_private_and_repair_scope_are_enforced() {
        let mut a = record("a", "oa", "la");
        a.verifier_private = Some(PalsVerifierPrivate {
            task_kind: "resume_task".into(),
            control_sha256: hash("control"),
            private_latent: vec![0.0; 16 * 384],
        });
        assert!(a.validate().is_err());
        a.verifier_private = None;
        let mut input = a.input.snapshot().clone();
        input.role = PalsDataRole::Critic;
        a.input = input.seal().unwrap();
        let mut label = outcome_label(PalsGameEnd::Checkmate, PalsOutcome::WhiteWin, None);
        label.counterexample = Some(PalsCounterexampleTarget {
            challenged_line_sha256: hash("line"),
            divergence_ply: 1,
            response_line: vec![100],
            repair_line: None,
            validity: PalsConditionalValidity::SupportedAfterRepair,
            input_revision: 3,
            legality_evidence_sha256: hash("rules"),
        });
        a.future_label = Some(label);
        assert!(a.validate().is_err());
        a.future_label
            .as_mut()
            .unwrap()
            .counterexample
            .as_mut()
            .unwrap()
            .repair_line = Some(vec![200]);
        a.validate().unwrap();
    }
    #[test]
    fn split_rejects_opening_line_and_state_leaks() {
        let plan = PalsDatasetSplit {
            games: BTreeMap::from([
                ("a".into(), PalsSplit::Train),
                ("b".into(), PalsSplit::Holdout),
            ]),
        };
        let a = record("a", "oa", "la");
        let b = record("b", "ob", "lb");
        plan.validate(&[a.clone(), b.clone()]).unwrap();
        assert!(
            plan.validate(&[a.clone(), record("b", "oa", "lb")])
                .is_err()
        );
        assert!(
            plan.validate(&[a.clone(), record("b", "ob", "la")])
                .is_err()
        );
        let mut sb = b.input.snapshot().clone();
        sb.rules_state_sha256 = a.input.snapshot().rules_state_sha256.clone();
        assert!(
            plan.validate(&[
                a,
                PalsLearningRecord {
                    input: sb.seal().unwrap(),
                    ..b
                }
            ])
            .is_err()
        );
    }
    fn checkpoint() -> PalsSealedCheckpoint {
        let context = PalsResumeContext {
            model_configuration_sha256: hash("model"),
            recipe_sha256: hash("recipe"),
            dataset_sha256: hash("dataset"),
            split_sha256: hash("split"),
            frozen_epoch: 1,
            raw_evidence_sha256: hash("raw"),
        };
        PalsTrainingCheckpoint::prepare(
            context,
            PalsCheckpointAsset {
                sha256: hash("modelbytes"),
                bytes: 100,
            },
            None,
            BTreeMap::from([
                (
                    "model".into(),
                    PalsRngState {
                        algorithm: "splitmix64-v1".into(),
                        words: vec![1],
                    },
                ),
                (
                    "data".into(),
                    PalsRngState {
                        algorithm: "splitmix64-v1".into(),
                        words: vec![2],
                    },
                ),
            ]),
            PalsSamplerState {
                algorithm: "seeded-permutation-v1".into(),
                seed: 2,
                order_sha256: hash("order"),
                cursor: 0,
                dataset_rows: 20,
            },
            PalsTrainingUsage::default(),
        )
        .unwrap()
        .seal()
        .unwrap()
    }
    #[test]
    fn checkpoint_roundtrip_resume_conditions_and_corruption() {
        let checkpoint = checkpoint();
        let encoded = checkpoint.to_json().unwrap();
        let loaded = PalsSealedCheckpoint::from_json(&encoded).unwrap();
        assert_eq!(loaded, checkpoint);
        let context = checkpoint.checkpoint().context.clone();
        loaded.verify_resume(&context, &budget()).unwrap();
        let mut changed = context;
        changed.frozen_epoch += 1;
        assert!(loaded.verify_resume(&changed, &budget()).is_err());
        assert!(
            PalsSealedCheckpoint::from_json(&encoded.replace("\"cursor\":0", "\"cursor\":1"))
                .is_err()
        );
    }
    #[test]
    fn registered_limits_and_flops_overflow_fail_closed() {
        let mut usage = PalsTrainingUsage {
            forward_flops: 500,
            backward_flops: 1000,
            cpu_nodes: 10,
            ..Default::default()
        };
        assert_eq!(usage.total_flops().unwrap(), 1500);
        budget().permits(&usage).unwrap();
        usage.backward_flops = 100_000;
        assert!(budget().permits(&usage).is_err());
        usage.forward_flops = u64::MAX;
        assert!(usage.total_flops().is_err());
        let mut bounds = budget();
        bounds.max_compute_units_milli = 0;
        assert!(bounds.validate().is_err());
    }
    #[test]
    fn dataset_authority_rejects_unregistered_source_despite_own_cpu_tag() {
        let mut a = record("a", "oa", "la");
        let mut label = outcome_label(PalsGameEnd::Checkmate, PalsOutcome::WhiteWin, None);
        label.provenance = PalsTargetProvenance::OwnedCpu {
            engine_sha256: hash("spoofed-external"),
            profile_sha256: hash("profile"),
            task_sha256: hash("task"),
            completed_depth: 1,
            nodes: 2,
            raw_evidence_sha256: hash("raw"),
        };
        a.future_label = Some(label);
        let registry = PalsOwnedSources {
            cpu_binary_sha256: BTreeSet::from([hash("own")]),
            model_weights_sha256: BTreeSet::from([hash("weights")]),
        };
        let split = PalsDatasetSplit {
            games: BTreeMap::from([("a".into(), PalsSplit::Train)]),
        };
        assert!(registry.audit(&[a.clone()], &split).is_err());
        if let PalsTargetProvenance::OwnedCpu { engine_sha256, .. } =
            &mut a.future_label.as_mut().unwrap().provenance
        {
            *engine_sha256 = hash("own");
        }
        let audited = registry.audit(&[a], &split).unwrap();
        assert_eq!(audited.labeled_records, 1);
        assert_eq!(audited.masked_value_records, 1);
        assert_eq!(audited.training_records, 1);
    }
    #[test]
    fn actual_viewpoint_and_future_public_record_leaks_fail() {
        let mut a = record("a", "oa", "la");
        let mut label = outcome_label(
            PalsGameEnd::Checkmate,
            PalsOutcome::WhiteWin,
            Some([1.0, 0.0, 0.0]),
        );
        label.white_to_move = true;
        a.future_label = Some(label);
        assert!(a.validate().is_err());
        let mut snapshot = a.input.snapshot().clone();
        snapshot.public_records[0].situation_revision = 4;
        assert!(snapshot.seal().is_err());
    }
    #[test]
    fn checkpoint_preserves_full_rng_and_requires_optimizer_after_steps() {
        let mut state = checkpoint().checkpoint().clone();
        state.rng.get_mut("model").unwrap().words = vec![1, 2, 3, 4];
        state.usage.steps = 1;
        assert!(state.validate().is_err());
        state.optimizer_state = Some(PalsCheckpointAsset {
            sha256: hash("adamw-state"),
            bytes: 32,
        });
        let sealed = state.seal().unwrap();
        let loaded = PalsSealedCheckpoint::from_json(&sealed.to_json().unwrap()).unwrap();
        assert_eq!(loaded.checkpoint().rng["model"].words, vec![1, 2, 3, 4]);
        let mut smaller = budget();
        smaller.max_steps = 1;
        loaded
            .verify_resume(&loaded.checkpoint().context, &smaller)
            .unwrap();
        smaller.max_flops = 1;
        let mut checkpoint = loaded.checkpoint().clone();
        checkpoint.usage.validation_flops = 2;
        let checkpoint = checkpoint.seal().unwrap();
        assert!(
            checkpoint
                .verify_resume(&checkpoint.checkpoint().context, &smaller)
                .is_err()
        );
    }
    #[test]
    fn verifier_targets_need_information_gain_not_just_agreement() {
        let mut a = record("a", "oa", "la");
        let mut input = a.input.snapshot().clone();
        input.role = PalsDataRole::Verifier;
        a.input = input.seal().unwrap();
        let mut label = outcome_label(PalsGameEnd::Checkmate, PalsOutcome::WhiteWin, None);
        label.verifier_tasks = Some(vec![PalsVerifierTaskTarget {
            task: PalsVerifierTaskKind::AttackRepair,
            branch_sha256: hash("branch"),
            cpu_profile_sha256: hash("profile"),
            budget_bucket: 1,
            preference_rank: None,
            information_gain_evidence_sha256: None,
        }]);
        a.future_label = Some(label);
        a.validate().unwrap();
        a.future_label
            .as_mut()
            .unwrap()
            .verifier_tasks
            .as_mut()
            .unwrap()[0]
            .preference_rank = Some(0);
        assert!(a.validate().is_err());
        a.future_label
            .as_mut()
            .unwrap()
            .verifier_tasks
            .as_mut()
            .unwrap()[0]
            .information_gain_evidence_sha256 = Some(hash("gain"));
        a.validate().unwrap();
    }
}
