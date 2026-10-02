use crate::*;
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalReason {
    Checkmate,
    Stalemate,
    DeadPosition,
    FivefoldRepetition,
    SeventyFiveMove,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlayStatus {
    Ongoing,
    Terminal {
        reason: TerminalReason,
        winner: Option<Color>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimRule {
    Threefold,
    FiftyMove,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimSource {
    CurrentPosition,
    IntendedMove(Move),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimAvailability {
    Available,
    Unavailable,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClaimEvidence {
    pub rule: ClaimRule,
    pub source: ClaimSource,
    pub availability: ClaimAvailability,
}

/// Orthogonal Rules facts. Claims and unknown history do not imply termination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositionClassification {
    pub play_status: PlayStatus,
    pub claims: Arc<[ClaimEvidence]>,
    pub history: HistoryCompleteness,
    /// Rules profile includes the supported dead-position subset, not an oracle claim.
    pub rules_profile: Digest,
}

impl PositionClassification {
    pub fn validate(&self) -> Result<(), ContractError> {
        if let PlayStatus::Terminal { reason, winner } = self.play_status {
            if (reason == TerminalReason::Checkmate) != winner.is_some() {
                return Err(ContractError::new(
                    ErrorCode::InvalidInput,
                    Stage::Contract,
                    "mate requires a winner; automatic draw requires none",
                ));
            }
        }
        Ok(())
    }
}

/// P is an owned, frozen Rules view. No borrowed mutable board crosses this API.
/// Rules adapters remain responsible for deriving all metadata from that view.
#[derive(Debug)]
pub struct PositionSnapshot<P> {
    identity: StateIdentity,
    state: Arc<P>,
    side: Color,
    classification: PositionClassification,
}

impl<P> Clone for PositionSnapshot<P> {
    fn clone(&self) -> Self {
        Self {
            identity: self.identity,
            state: Arc::clone(&self.state),
            side: self.side,
            classification: self.classification.clone(),
        }
    }
}

impl<P> PositionSnapshot<P> {
    pub fn try_new(
        identity: StateIdentity,
        state: Arc<P>,
        side: Color,
        classification: PositionClassification,
    ) -> Result<Self, ContractError> {
        classification.validate()?;
        Ok(Self {
            identity,
            state,
            side,
            classification,
        })
    }
    pub fn identity(&self) -> StateIdentity {
        self.identity
    }
    pub fn state(&self) -> &Arc<P> {
        &self.state
    }
    pub fn side_to_move(&self) -> Color {
        self.side
    }
    pub fn classification(&self) -> &PositionClassification {
        &self.classification
    }
}

/// Preserves Rules ordering. This checks shape/duplicates, not chess legality.
/// The order digest is an owner attestation; equality also compares actual moves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegalMoveView {
    state: StateIdentity,
    order: LegalOrderIdentity,
    moves: Arc<[Move]>,
}

impl LegalMoveView {
    pub fn try_new(
        state: StateIdentity,
        order: LegalOrderIdentity,
        moves: Vec<Move>,
        max_moves: usize,
    ) -> Result<Self, ContractError> {
        if max_moves == 0 || moves.len() > max_moves {
            return Err(ContractError::new(
                ErrorCode::ResourceExhausted,
                Stage::Contract,
                "legal view exceeds declared finite move limit",
            ));
        }
        if moves.iter().any(|mv| mv.from == mv.to)
            || moves.iter().collect::<HashSet<_>>().len() != moves.len()
        {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Contract,
                "invalid or duplicate semantic move",
            ));
        }
        Ok(Self {
            state,
            order,
            moves: moves.into(),
        })
    }
    pub fn state(&self) -> StateIdentity {
        self.state
    }
    pub fn order(&self) -> LegalOrderIdentity {
        self.order
    }
    pub fn moves(&self) -> &[Move] {
        &self.moves
    }
}
