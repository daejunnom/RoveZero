//! Rules attestation for the coordinator's shared contract revision 0.1.
//!
//! Enable `contracts` to consume PR #7 at the pinned source revision. Owners are
//! issued by the actual registry, never by a hash or pointer cast in this crate.
//! No mutable board borrow crosses the evaluator boundary. Model input keys,
//! action maps, request IDs, generations and clocks remain their owners' work.
//!
//! ```
//! use rz_position::{contracts::ContractPosition, BoardMove, Position};
//! let mut live = ContractPosition::new(rz_contracts::OwnerId(1), Position::startpos());
//! let frozen = live.export()?;
//! let mv = rz_contracts::Move::try_from(BoardMove::from_uci("e2e4")?)?;
//! let undo = live.make_from_view(&frozen, mv)?;
//! assert_eq!(frozen.snapshot().side_to_move(), rz_contracts::Color::White);
//! assert_eq!(live.position().side_to_move(), rz_position::Color::Black);
//! live.unmake(undo)?;
//! assert!(live.make_from_view(&frozen, mv).is_err());
//! # Ok::<(), rz_contracts::ContractError>(())
//! ```

use crate::{
    Availability, BoardMove, ClaimEvidence, ClaimReason, Color, HistoryCompleteness, HistoryOrigin,
    LegalMoveView, PieceKind, PlayStatus, Position, PositionClassification, PositionError,
    PositionSnapshot, TerminalReason, UndoToken,
};
use rz_contracts as shared;
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

pub const CONTRACT_SOURCE_REVISION: &str = "67284c4f66f7a7ae9f46fa63dfd50e7410eb6845";

/// SHA-256 input profile. Changing any field requires a new profile identifier.
/// The framed fields below are bytes with u64 little-endian lengths, not a wire
/// codec. State history is newest first and includes irreversible boundaries.
pub const IDENTITY_PROFILE: &str = concat!(
    "rz-position-contracts/1;contract=0.1;rules=rz-position/0.1.0;",
    "variant=standard_chess;squares=a1:0,h8:63;",
    "moves=from,to,promotion:none,Q,R,B,N;",
    "dead=proven_material_subset_v1;claims=3fold,100ply,current+intended;",
    "automatic=5fold,150ply;mate-precedes-150ply;",
    "digest=sha256;framing=u64le;history=newest-first,canonical-fen,irreversible;",
    "origin+completeness=preserved"
);

// A validated board has at most 16 pieces per side. This conservative finite
// bound also covers promotion choices without assuming retrograde reachability.
pub const MAX_CONTRACT_LEGAL_MOVES: usize = 512;

/// Concrete immutable payload, including evidence absent from the shared enum.
#[derive(Clone, Debug)]
pub struct RulesState {
    snapshot: PositionSnapshot,
    classification: PositionClassification,
}

impl RulesState {
    pub fn snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    pub fn classification(&self) -> &PositionClassification {
        &self.classification
    }
}

/// One Rules-attested state and ordered legal array. Only Rules can create this
/// bundle; the generic shared constructors alone do not attest chess legality.
#[derive(Clone, Debug)]
pub struct ContractState {
    snapshot: shared::PositionSnapshot<RulesState>,
    legal: shared::LegalMoveView,
    local_legal: LegalMoveView,
}

impl ContractState {
    pub fn snapshot(&self) -> &shared::PositionSnapshot<RulesState> {
        &self.snapshot
    }
    pub fn legal_moves(&self) -> &shared::LegalMoveView {
        &self.legal
    }
    pub fn rules(&self) -> &RulesState {
        self.snapshot.state()
    }
    /// Exact terminal value in the frozen state's side-to-move perspective.
    /// Ongoing states, including available claims, have no exact WDL here.
    pub fn terminal_wdl(&self) -> Option<shared::Wdl> {
        let PlayStatus::Terminal { winner, .. } = self.rules().classification.play_status else {
            return None;
        };
        let probabilities = match winner {
            Some(color) if color == self.rules().snapshot.side_to_move() => [1.0, 0.0, 0.0],
            Some(_) => [0.0, 0.0, 1.0],
            None => [0.0, 1.0, 0.0],
        };
        Some(
            shared::Wdl::try_new(probabilities[0], probabilities[1], probabilities[2], 0.0)
                .expect("exact terminal probabilities"),
        )
    }
}

/// A mutable Rules owner. The registry must assign a fresh, nonreused OwnerId to
/// each instance, including a fork/new game. There is deliberately no Clone or
/// mutable escape hatch: every exported revision names this owned position.
#[derive(Debug)]
pub struct ContractPosition {
    owner: shared::OwnerId,
    position: Position,
}

impl ContractPosition {
    pub fn new(owner: shared::OwnerId, position: Position) -> Self {
        Self { owner, position }
    }
    pub fn position(&self) -> &Position {
        &self.position
    }
    pub fn into_position(self) -> Position {
        self.position
    }
    /// Derive metadata from one current Rules state; no caller-provided digest,
    /// classification or move array is accepted as an attestation.
    pub fn export(&self) -> Result<ContractState, shared::ContractError> {
        let classification = self.position.classify_position()?;
        let local_legal = self.position.ordered_legal_moves();
        let local_snapshot = local_legal.snapshot();
        let identity = shared::StateIdentity {
            owner: self.owner,
            revision: shared::StateRevision(local_snapshot.revision()),
            semantic: state_digest(local_snapshot),
        };
        let moves: Vec<shared::Move> = local_legal
            .moves()
            .iter()
            .copied()
            .map(shared::Move::try_from)
            .collect::<Result<_, _>>()?;
        let legal = shared::LegalMoveView::try_new(
            identity,
            order_digest(&moves),
            moves,
            MAX_CONTRACT_LEGAL_MOVES,
        )?;
        let common_classification = map_classification(&classification)?;
        let side = local_snapshot.side_to_move().into();
        let payload = RulesState {
            snapshot: local_snapshot.clone(),
            classification,
        };
        let snapshot = shared::PositionSnapshot::try_new(
            identity,
            Arc::new(payload),
            side,
            common_classification,
        )?;
        Ok(ContractState {
            snapshot,
            legal,
            local_legal,
        })
    }
    /// Live application checks exact concrete owner/revision/history, not just
    /// the shared digest. A restored state cannot reuse an earlier live view.
    /// Exact terminals bypass both further game moves and neural evaluation.
    pub fn make_from_view(
        &mut self,
        view: &ContractState,
        mv: shared::Move,
    ) -> Result<UndoToken, shared::ContractError> {
        if view.snapshot.identity().owner != self.owner
            || !self.position.matches_snapshot(view.local_legal.snapshot())
        {
            return Err(PositionError::StaleView.into());
        }
        if view.rules().classification.play_status != PlayStatus::Ongoing {
            return Err(shared::ContractError::new(
                shared::ErrorCode::InvalidInput,
                shared::Stage::Admission,
                "exact terminal cannot accept a further game move",
            ));
        }
        Ok(self
            .position
            .make_from_view(&view.local_legal, BoardMove::try_from(mv)?)?)
    }
    pub fn unmake(&mut self, token: UndoToken) -> Result<(), shared::ContractError> {
        Ok(self.position.unmake(token)?)
    }
}

impl From<Color> for shared::Color {
    fn from(color: Color) -> Self {
        match color {
            Color::White => Self::White,
            Color::Black => Self::Black,
        }
    }
}

impl TryFrom<BoardMove> for shared::Move {
    type Error = shared::ContractError;
    fn try_from(mv: BoardMove) -> Result<Self, Self::Error> {
        let promotion = match mv.promotion {
            None => None,
            Some(PieceKind::Queen) => Some(shared::Promotion::Queen),
            Some(PieceKind::Rook) => Some(shared::Promotion::Rook),
            Some(PieceKind::Bishop) => Some(shared::Promotion::Bishop),
            Some(PieceKind::Knight) => Some(shared::Promotion::Knight),
            Some(_) => return Err(PositionError::InvalidMove("invalid promotion").into()),
        };
        shared::Move::new(
            shared::Square::try_new(mv.from.index())?,
            shared::Square::try_new(mv.to.index())?,
            promotion,
        )
    }
}

impl TryFrom<shared::Move> for BoardMove {
    type Error = shared::ContractError;
    fn try_from(mv: shared::Move) -> Result<Self, Self::Error> {
        let promotion = mv.promotion.map(|promotion| match promotion {
            shared::Promotion::Queen => PieceKind::Queen,
            shared::Promotion::Rook => PieceKind::Rook,
            shared::Promotion::Bishop => PieceKind::Bishop,
            shared::Promotion::Knight => PieceKind::Knight,
        });
        Ok(Self::new(
            crate::Square::new(mv.from.index())?,
            crate::Square::new(mv.to.index())?,
            promotion,
        )?)
    }
}

impl From<PositionError> for shared::ContractError {
    fn from(error: PositionError) -> Self {
        use shared::{ErrorCode, Stage};
        let (code, detail) = match error {
            PositionError::InvalidFen(detail) | PositionError::InvalidMove(detail) => {
                (ErrorCode::InvalidInput, detail)
            }
            PositionError::IllegalMove => (ErrorCode::InvalidInput, "illegal move"),
            PositionError::CounterOverflow => {
                (ErrorCode::ResourceExhausted, "rule counter overflow")
            }
            PositionError::RevisionExhausted => {
                (ErrorCode::ResourceExhausted, "state revision exhausted")
            }
            PositionError::ResourceLimit(detail) => (ErrorCode::ResourceExhausted, detail),
            PositionError::UndoMismatch => {
                (ErrorCode::IdentityMismatch, "undo owner or child mismatch")
            }
            PositionError::StaleView => (ErrorCode::Stale, "stale Rules view"),
        };
        Self::new(code, Stage::Admission, detail)
    }
}

fn map_classification(
    classification: &PositionClassification,
) -> Result<shared::PositionClassification, shared::ContractError> {
    let play_status = match classification.play_status {
        PlayStatus::Ongoing => shared::PlayStatus::Ongoing,
        PlayStatus::Terminal { reason, winner } => shared::PlayStatus::Terminal {
            reason: match reason {
                TerminalReason::Checkmate => shared::TerminalReason::Checkmate,
                TerminalReason::Stalemate => shared::TerminalReason::Stalemate,
                TerminalReason::DeadPosition => shared::TerminalReason::DeadPosition,
                TerminalReason::FivefoldRepetition => shared::TerminalReason::FivefoldRepetition,
                TerminalReason::SeventyFiveMove => shared::TerminalReason::SeventyFiveMove,
            },
            winner: winner.map(Into::into),
        },
    };
    let claims = classification
        .claim_availability
        .iter()
        .map(|claim| {
            Ok(shared::ClaimEvidence {
                rule: match claim.reason {
                    ClaimReason::ThreefoldRepetition => shared::ClaimRule::Threefold,
                    ClaimReason::FiftyMove => shared::ClaimRule::FiftyMove,
                },
                source: match claim.evidence {
                    ClaimEvidence::CurrentPosition => shared::ClaimSource::CurrentPosition,
                    ClaimEvidence::IntendedMove(mv) => {
                        shared::ClaimSource::IntendedMove(mv.try_into()?)
                    }
                },
                availability: match claim.availability {
                    Availability::Available => shared::ClaimAvailability::Available,
                    Availability::Unavailable => shared::ClaimAvailability::Unavailable,
                    Availability::Unknown => shared::ClaimAvailability::Unknown,
                },
            })
        })
        .collect::<Result<Vec<_>, shared::ContractError>>()?;
    Ok(shared::PositionClassification {
        play_status,
        claims: claims.into(),
        history: match classification.history_evidence.completeness {
            HistoryCompleteness::Complete => shared::HistoryCompleteness::Complete,
            HistoryCompleteness::UnknownPrefix => shared::HistoryCompleteness::UnknownPrefix,
        },
        rules_profile: profile_digest(),
    })
}

fn frame(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

/// Profile identity is independent of runtime owner, model and game generation.
pub fn profile_digest() -> shared::Digest {
    shared::Digest(Sha256::digest(IDENTITY_PROFILE.as_bytes()).into())
}

fn state_digest(snapshot: &PositionSnapshot) -> shared::Digest {
    let mut hash = Sha256::new();
    frame(&mut hash, b"rz-position-state/1");
    hash.update(profile_digest().0);
    hash.update([match snapshot.history_origin() {
        HistoryOrigin::StartPosition => 0,
        HistoryOrigin::Fen => 1,
    }]);
    hash.update([match snapshot.history_completeness() {
        HistoryCompleteness::Complete => 0,
        HistoryCompleteness::UnknownPrefix => 1,
    }]);
    hash.update((snapshot.known_history_len() as u64).to_le_bytes());
    for state in snapshot.known_history() {
        hash.update([u8::from(state.is_irreversible_boundary())]);
        frame(&mut hash, state.to_fen().as_bytes());
    }
    shared::Digest(hash.finalize().into())
}

fn order_digest(moves: &[shared::Move]) -> shared::LegalOrderIdentity {
    let mut hash = Sha256::new();
    frame(&mut hash, b"rz-position-legal-order/1");
    hash.update(profile_digest().0);
    hash.update((moves.len() as u64).to_le_bytes());
    for mv in moves {
        hash.update([
            mv.from.index(),
            mv.to.index(),
            match mv.promotion {
                None => 0,
                Some(shared::Promotion::Queen) => 1,
                Some(shared::Promotion::Rook) => 2,
                Some(shared::Promotion::Bishop) => 3,
                Some(shared::Promotion::Knight) => 4,
            },
        ]);
    }
    shared::LegalOrderIdentity(shared::Digest(hash.finalize().into()))
}
