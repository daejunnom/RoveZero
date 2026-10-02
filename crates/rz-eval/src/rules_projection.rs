//! Concrete A → C projection. Rules retains legality/history/terminal authority.
//! This consumes owned snapshots and reads their exact raw EP and known history;
//! missing-prefix history remains unknown, while model padding stays an encoder
//! option. The original C model-input codec and ordered action map are reused.

use crate::contracts::{input_key, ordered_policy_indices, MaiaBinding, PreparedRequest};
use rz_contracts::{
    ContractError, Digest, ErrorCode, EvalInputKey, EvalRequest, ModelDescriptor, Move, Stage,
};
use rz_encoding::classical::{self, EncodedInput, Frame, HistoryFill, Input, HISTORY_FRAMES};
use rz_position::{contracts::RulesState, Color, PieceKind, Square};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ClassicalProjection {
    binding: MaiaBinding,
}

impl ClassicalProjection {
    pub fn new(binding: MaiaBinding) -> Self {
        Self { binding }
    }
    pub fn model(&self) -> &Arc<ModelDescriptor> {
        self.binding.model()
    }
    pub fn backend(&self) -> Digest {
        self.binding.backend()
    }
    pub fn binding(&self) -> &MaiaBinding {
        &self.binding
    }

    pub fn project(&self, state: &RulesState) -> Result<RulesProjection, ContractError> {
        let snapshot = state.snapshot();
        let mut frames = Vec::new();
        frames.try_reserve_exact(HISTORY_FRAMES).map_err(|_| {
            failed(
                ErrorCode::ResourceExhausted,
                "Rules projection allocation failed",
            )
        })?;
        for history in snapshot.known_history().take(HISTORY_FRAMES) {
            let mut pieces = [[0; 6]; 2];
            for index in 0..64 {
                if let Some(piece) = history.piece_at(Square::new(index)?) {
                    let color = usize::from(piece.color == Color::Black);
                    let kind = match piece.kind {
                        PieceKind::Pawn => 0,
                        PieceKind::Knight => 1,
                        PieceKind::Bishop => 2,
                        PieceKind::Rook => 3,
                        PieceKind::Queen => 4,
                        PieceKind::King => 5,
                    };
                    pieces[color][kind] |= 1u64 << index;
                }
            }
            let identity = history.repetition_identity();
            let repeated = history
                .known_history()
                .skip(1)
                .any(|prior| prior.repetition_identity() == identity);
            frames.push(Frame {
                pieces,
                repeated,
                en_passant_target: history.en_passant_target().map(Square::index),
            });
        }
        let rights = snapshot.castling_rights();
        Ok(RulesProjection {
            frames,
            black_to_move: snapshot.side_to_move() == Color::Black,
            castling: [
                rights & 2 != 0,
                rights & 1 != 0,
                rights & 8 != 0,
                rights & 4 != 0,
            ],
            halfmove_clock: snapshot.halfmove_clock(),
            history_fill: self.binding.history_fill(),
        })
    }

    pub fn encode(&self, state: &RulesState) -> Result<EncodedInput, ContractError> {
        let projection = self.project(state)?;
        classical::encode(projection.input()).map_err(|_| {
            failed(
                ErrorCode::InvalidInput,
                "Rules projection cannot be encoded by this profile",
            )
        })
    }

    pub fn input_key(
        &self,
        state: &RulesState,
        ordered_legal: &[Move],
    ) -> Result<EvalInputKey, ContractError> {
        let projection = self.project(state)?;
        // Keep the original dense-input identity codec. Legal order is a
        // separate attestation and is nevertheless checked for representability.
        ordered_policy_indices(projection.input(), ordered_legal)?;
        let encoded = classical::encode(projection.input()).map_err(|_| {
            failed(
                ErrorCode::InvalidInput,
                "Rules projection cannot be encoded by this profile",
            )
        })?;
        Ok(input_key(self.model().encoding().handle, &encoded))
    }

    pub fn legal_indices(
        &self,
        state: &RulesState,
        ordered_legal: &[Move],
    ) -> Result<Vec<usize>, ContractError> {
        ordered_policy_indices(self.project(state)?.input(), ordered_legal)
    }

    pub fn prepare(
        &self,
        request: Arc<EvalRequest<RulesState>>,
    ) -> Result<PreparedRequest<RulesState>, ContractError> {
        let projection = self.project(request.position().state())?;
        self.binding.prepare(request, projection.input())
    }
}

pub struct RulesProjection {
    frames: Vec<Frame>,
    black_to_move: bool,
    castling: [bool; 4],
    halfmove_clock: u32,
    history_fill: HistoryFill,
}

impl RulesProjection {
    pub fn input(&self) -> Input<'_> {
        Input {
            history: &self.frames,
            black_to_move: self.black_to_move,
            castling: self.castling,
            halfmove_clock: self.halfmove_clock,
            history_fill: self.history_fill,
        }
    }
}

fn failed(code: ErrorCode, detail: &'static str) -> ContractError {
    ContractError::new(code, Stage::Admission, detail)
}
