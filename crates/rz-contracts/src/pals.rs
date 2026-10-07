//! PALS-specific contracts. The existing evaluation revision 0.1 is unchanged.
//! IDs and digests are issued by owners; estimates never certify Rules facts.

use crate::{
    ByteBudget, CancelToken, ClockDomain, Color, ContractError, Deadline, Digest, ErrorCode,
    GameGeneration, MonotonicTick, Move, ProcessEpoch, Promotion, RequestId, RootGeneration,
    SchemaVersion, Square, Stage, StateIdentity, Wdl,
};
use std::sync::Arc;

pub const PALS_CONTRACT_REVISION: SchemaVersion = SchemaVersion { major: 0, minor: 1 };

/// Search fencing is independent of a neural model registry. CPU-only search
/// has no fictitious model/encoding/backend handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SearchAuthority {
    pub epoch: ProcessEpoch,
    pub game: GameGeneration,
    pub root: RootGeneration,
    pub implementation: Digest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Role {
    Proposer,
    Critic,
    Validator,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ExecutionMode {
    Deployment,
    Collection,
}

/// a1=0, h8=63. Castle/EP meaning remains owned by Rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Move16(u16);

impl Move16 {
    pub fn encode(movement: Move) -> Self {
        let promotion = match movement.promotion {
            None => 0,
            Some(Promotion::Queen) => 1,
            Some(Promotion::Rook) => 2,
            Some(Promotion::Bishop) => 3,
            Some(Promotion::Knight) => 4,
        };
        Self(u16::from(movement.from.index()) | (u16::from(movement.to.index()) << 6)
            | (promotion << 12))
    }

    pub fn try_from_bits(bits: u16) -> Result<Self, ContractError> {
        if bits & 0x8000 != 0 || (bits >> 12) > 4 {
            return Err(invalid("invalid Move16 reserved bits or promotion"));
        }
        let movement = Self(bits);
        movement.decode()?;
        Ok(movement)
    }

    pub const fn bits(self) -> u16 {
        self.0
    }

    pub fn decode(self) -> Result<Move, ContractError> {
        let promotion = match self.0 >> 12 {
            0 => None,
            1 => Some(Promotion::Queen),
            2 => Some(Promotion::Rook),
            3 => Some(Promotion::Bishop),
            4 => Some(Promotion::Knight),
            _ => return Err(invalid("invalid Move16 promotion")),
        };
        Move::new(Square::try_new((self.0 & 63) as u8)?,
            Square::try_new(((self.0 >> 6) & 63) as u8)?, promotion)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SituationHandle {
    pub slot: u32,
    pub generation: u64,
}

/// Every item participates in exact representation-cache identity. Role-neutral
/// memory may use a separate key only after its actual mask/content is equal.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RepresentationKey {
    pub input: Digest,
    pub history: Digest,
    pub candidates: Digest,
    pub mask_positions: Digest,
    pub model: Digest,
    pub encoding: Digest,
    pub precision: crate::PrecisionProfile,
    pub frozen_epoch: u64,
    pub record_revision: u64,
    pub role: Role,
}

#[derive(Clone, Debug)]
pub struct RoleRequest<P> {
    pub revision: SchemaVersion,
    pub id: RequestId,
    pub authority: SearchAuthority,
    pub situation: SituationHandle,
    pub state_identity: StateIdentity,
    pub state: Arc<P>,
    pub key: RepresentationKey,
    pub role: Role,
    pub mode: ExecutionMode,
    pub candidates: Arc<[Move16]>,
    pub public_records: Arc<[Digest]>,
    pub max_records: u16,
    pub recurrent_steps: u16,
    pub deadline: Deadline,
    pub cancel: CancelToken,
    pub resources: ByteBudget,
}

impl<P> RoleRequest<P> {
    pub fn validate(&self, clock: ClockDomain, now: MonotonicTick) -> Result<(), ContractError> {
        if self.revision != PALS_CONTRACT_REVISION {
            return Err(ContractError::new(ErrorCode::UnsupportedContract, Stage::Contract,
                "unsupported PALS contract"));
        }
        if self.id.epoch != self.authority.epoch || self.deadline.clock.0 != self.authority.epoch
            || self.role != self.key.role {
            return Err(ContractError::new(ErrorCode::IdentityMismatch, Stage::Admission,
                "PALS request identity mismatch"));
        }
        if self.mode == ExecutionMode::Deployment && self.role == Role::Validator {
            return Err(invalid("Validator is collection-only"));
        }
        if self.max_records == 0 || self.max_records > 128
            || self.public_records.len() > usize::from(self.max_records)
            || self.recurrent_steps == 0 || self.recurrent_steps > 16
            || self.candidates.len() > 256 {
            return Err(invalid("PALS input exceeds declared finite shape"));
        }
        for movement in self.candidates.iter() { movement.decode()?; }
        let mut seen = std::collections::HashSet::new();
        if self.candidates.iter().any(|movement| !seen.insert(*movement)) {
            return Err(invalid("duplicate legal candidate"));
        }
        if self.cancel.is_canceled() {
            return Err(ContractError::new(ErrorCode::Canceled, Stage::Admission,
                "PALS request canceled"));
        }
        self.deadline.accepts(clock, now)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RolePayload {
    Proposal { candidate_logits: Vec<f32>, wdl: Wdl },
    Counterexample { candidate_logits: Vec<f32>, divergence_logits: Vec<f32>, wdl: Wdl },
    TaskRanking { task_logits: Vec<f32> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoleOutput {
    pub id: RequestId,
    pub authority: SearchAuthority,
    pub key: RepresentationKey,
    pub payload: RolePayload,
    /// An approximation state, never an exact cache key or a Rules state.
    pub private_latent: Vec<f32>,
}

impl RoleOutput {
    pub fn validate_for<P>(&self, request: &RoleRequest<P>) -> Result<(), ContractError> {
        if self.id != request.id || self.authority != request.authority || self.key != request.key {
            return Err(ContractError::new(ErrorCode::IdentityMismatch, Stage::Output,
                "PALS output request mismatch"));
        }
        let finite = |values: &[f32]| values.iter().all(|value| value.is_finite());
        let valid = match (&self.payload, request.role) {
            (RolePayload::Proposal { candidate_logits, .. }, Role::Proposer) =>
                candidate_logits.len() == request.candidates.len() && finite(candidate_logits),
            (RolePayload::Counterexample { candidate_logits, divergence_logits, .. }, Role::Critic) =>
                candidate_logits.len() == request.candidates.len() && finite(candidate_logits)
                    && (!candidate_logits.is_empty() || !divergence_logits.is_empty())
                    && divergence_logits.len() <= 128
                    && finite(divergence_logits),
            (RolePayload::TaskRanking { task_logits }, Role::Validator) =>
                !task_logits.is_empty() && task_logits.len() <= 7 && finite(task_logits),
            _ => false,
        };
        if !valid || self.private_latent.len() != 16 * 384 || !finite(&self.private_latent) {
            return Err(ContractError::new(ErrorCode::NumericalFailure, Stage::Output,
                "PALS role output shape or finite-value violation"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CpuProblemKind { AnalyzePosition, RootMoves, Divergence, Resume }

#[derive(Clone, Debug)]
pub struct CpuProblem<P> {
    pub id: RequestId,
    pub authority: SearchAuthority,
    pub state: Arc<P>,
    pub state_identity: StateIdentity,
    pub kind: CpuProblemKind,
    pub prefix: Arc<[Move16]>,
    pub root_moves: Arc<[Move16]>,
    pub requested_depth: u16,
    pub max_nodes: u64,
    pub deadline: Deadline,
    pub cancel: CancelToken,
    pub profile: Digest,
    pub conditions: Digest,
    /// Resume is an opaque engine-owned condition-bound token, not a foreign TT.
    pub resume: Option<Digest>,
}

impl<P> CpuProblem<P> {
    pub fn validate(&self, clock: ClockDomain, now: MonotonicTick) -> Result<(), ContractError> {
        if self.id.epoch != self.authority.epoch || self.deadline.clock.0 != self.authority.epoch {
            return Err(ContractError::new(ErrorCode::IdentityMismatch, Stage::Admission,
                "CPU request authority mismatch"));
        }
        if self.requested_depth == 0 || self.requested_depth > 128 || self.max_nodes == 0
            || self.prefix.len() > 256 || self.root_moves.len() > 256 {
            return Err(invalid("CPU problem exceeds finite limits"));
        }
        if (self.kind == CpuProblemKind::Resume) != self.resume.is_some()
            || (self.kind == CpuProblemKind::RootMoves && self.root_moves.is_empty()) {
            return Err(invalid("CPU problem payload does not match capability"));
        }
        for movement in self.prefix.iter().chain(self.root_moves.iter()) { movement.decode()?; }
        if self.cancel.is_canceled() {
            return Err(ContractError::new(ErrorCode::Canceled, Stage::Admission,
                "CPU problem canceled"));
        }
        self.deadline.accepts(clock, now)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bound { ExactAtDepth, LowerAtDepth, UpperAtDepth, Unknown }

/// Raw bootstrap/model/engine scores retain their scale and perspective.
/// A CPU mate report is not an all-defenses Rules proof.
#[derive(Clone, Debug, PartialEq)]
pub enum RawScore {
    Bootstrap { value: i32, evaluator: Digest, perspective: Color },
    LearnedScalar { value: f32, evaluator: Digest, perspective: Color },
    ReportedMate { plies: i32, engine: Digest, perspective: Color },
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuCoverage {
    pub requested_depth: u16,
    pub completed_depth: u16,
    pub bound: Bound,
    pub profile: Digest,
    pub conditions: Digest,
    pub selective: bool,
}

impl CpuCoverage {
    pub fn satisfies(self, requested: Self) -> bool {
        self.profile == requested.profile && self.conditions == requested.conditions
            && self.selective == requested.selective && self.completed_depth >= requested.requested_depth
            && self.bound == Bound::ExactAtDepth
    }
}

fn invalid(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::InvalidInput, Stage::Contract, detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move16_round_trip_all_promotions_and_reserved_rejection() {
        for promotion in [None, Some(Promotion::Queen), Some(Promotion::Rook),
            Some(Promotion::Bishop), Some(Promotion::Knight)] {
            let movement = Move::new(Square::try_new(48).unwrap(),
                Square::try_new(56).unwrap(), promotion).unwrap();
            assert_eq!(Move16::encode(movement).decode().unwrap(), movement);
        }
        assert!(Move16::try_from_bits(0x8001).is_err());
        assert!(Move16::try_from_bits(0x5001).is_err());
        assert!(Move16::try_from_bits(0).is_err());
    }

    #[test]
    fn insufficient_or_changed_cpu_coverage_is_not_complete() {
        let coverage = CpuCoverage { requested_depth: 8, completed_depth: 8,
            bound: Bound::ExactAtDepth, profile: Digest([1; 32]),
            conditions: Digest([2; 32]), selective: false };
        assert!(coverage.satisfies(coverage));
        assert!(!coverage.satisfies(CpuCoverage { requested_depth: 10, ..coverage }));
        assert!(!coverage.satisfies(CpuCoverage { conditions: Digest([3; 32]), ..coverage }));
        assert!(!CpuCoverage { bound: Bound::LowerAtDepth, ..coverage }.satisfies(coverage));
    }
}
