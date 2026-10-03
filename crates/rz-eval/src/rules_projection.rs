//! Concrete A → C projection. Rules retains legality/history/terminal authority.
//! This consumes owned snapshots and reads their exact raw EP and known history;
//! missing-prefix history remains unknown, while model padding stays an encoder
//! option. The original C model-input codec and ordered action map are reused.

use crate::contracts::{input_key, ordered_policy_indices, MaiaBinding, PreparedRequest};
use rz_contracts::{
    ContractError, Digest, ErrorCode, EvalInputKey, EvalRequest, ModelDescriptor, Move, Stage,
};
use rz_encoding::classical::{self, EncodedInput, Frame, HistoryFill, Input, HISTORY_FRAMES};
use rz_position::{contracts::RulesState, Color, Piece, PieceKind, Square};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ClassicalProjection {
    binding: MaiaBinding,
    #[cfg(feature = "experimental-prepared-input")]
    prepared: Arc<std::sync::Mutex<Option<Arc<PreparedRulesInput>>>>,
    #[cfg(feature = "experimental-raw-cache")]
    raw_cache: crate::raw_cache::RawCache,
}

impl ClassicalProjection {
    pub fn new(binding: MaiaBinding) -> Self {
        Self {
            binding,
            #[cfg(feature = "experimental-prepared-input")]
            prepared: Arc::default(),
            #[cfg(feature = "experimental-raw-cache")]
            raw_cache: crate::raw_cache::RawCache::disabled(),
        }
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

    #[cfg(feature = "experimental-raw-cache")]
    pub fn configure_raw_cache(
        &self,
        limits: crate::raw_cache::RawCacheLimits,
    ) -> Result<(), ContractError> {
        self.raw_cache.configure(limits)
    }
    #[cfg(feature = "experimental-raw-cache")]
    pub fn clear_raw_cache(&self) -> Result<(), ContractError> {
        self.raw_cache.clear()
    }
    #[cfg(feature = "experimental-raw-cache")]
    pub fn raw_cache_stats(&self) -> Result<crate::raw_cache::RawCacheStats, ContractError> {
        self.raw_cache.stats()
    }
    #[cfg(feature = "experimental-raw-cache")]
    pub fn raw_cache_provider(&self) -> crate::raw_cache::RawCacheProvider {
        crate::raw_cache::RawCacheProvider {
            projection: self.clone(),
            cache: self.raw_cache.clone(),
        }
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
        #[cfg(feature = "experimental-history-frames")]
        for history in snapshot
            .recent_history_frames::<HISTORY_FRAMES>()
            .iter()
            .flatten()
        {
            frames.push(Frame {
                pieces: read_pieces(history.piece_bitboards(), |square| history.piece_at(square))?,
                repeated: history.repeated(),
                en_passant_target: history.en_passant_target().map(Square::index),
            });
        }
        #[cfg(not(feature = "experimental-history-frames"))]
        for history in snapshot.known_history().take(HISTORY_FRAMES) {
            let pieces = read_pieces(history.piece_bitboards(), |square| history.piece_at(square))?;
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
        #[cfg(feature = "experimental-prepared-input")]
        {
            return Ok(self.prepared_input(state, ordered_legal, None)?.key);
        }
        #[cfg(not(feature = "experimental-prepared-input"))]
        {
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
        #[cfg(feature = "experimental-prepared-input")]
        {
            let prepared = self.prepared_input(
                request.position().state(),
                request.legal().moves(),
                Some(&request),
            )?;
            return self.attach_raw_cache(self.binding.prepare_rules(request, &prepared)?);
        }
        #[cfg(not(feature = "experimental-prepared-input"))]
        {
            let projection = self.project(request.position().state())?;
            self.attach_raw_cache(self.binding.prepare(request, projection.input())?)
        }
    }

    fn attach_raw_cache(
        &self,
        prepared: PreparedRequest<RulesState>,
    ) -> Result<PreparedRequest<RulesState>, ContractError> {
        #[cfg(feature = "experimental-raw-cache")]
        let prepared = {
            let mut prepared = prepared;
            if self.raw_cache.enabled() {
                prepared.raw_cache = Some(self.raw_cache.clone());
            }
            prepared
        };
        Ok(prepared)
    }

    #[cfg(feature = "experimental-prepared-input")]
    fn prepared_input(
        &self,
        state: &RulesState,
        legal: &[Move],
        request: Option<&EvalRequest<RulesState>>,
    ) -> Result<Arc<PreparedRulesInput>, ContractError> {
        let mut memo = self
            .prepared
            .lock()
            .map_err(|_| failed(ErrorCode::BackendFailure, "prepared input owner poisoned"))?;
        if let Some(prepared) = memo.as_ref() {
            if prepared.identity.matches(state.snapshot()) && prepared.legal.as_ref() == legal {
                if let Some(request) = request {
                    self.binding
                        .validate_preparation(request, prepared.projection.input())?;
                    check_prepared_key(prepared.key, request)?;
                }
                return Ok(Arc::clone(prepared));
            }
        }
        let projection = self.project(state)?;
        if let Some(request) = request {
            self.binding
                .validate_preparation(request, projection.input())?;
        }
        // Match the original error order: input_key maps moves first, while
        // prepare validates the actual input key before mapping legal indices.
        let indices = if request.is_none() {
            Some(ordered_policy_indices(projection.input(), legal)?)
        } else {
            None
        };
        let encoded = classical::encode(projection.input())
            .map_err(|_| failed(ErrorCode::InvalidInput, "invalid Rules input projection"))?;
        let key = input_key(self.model().encoding().handle, &encoded);
        if let Some(request) = request {
            check_prepared_key(key, request)?;
        }
        let indices = match indices {
            Some(indices) => indices,
            None => ordered_policy_indices(projection.input(), legal)?,
        };
        let prepared = Arc::new(PreparedRulesInput {
            identity: state.snapshot().weak_position_identity(),
            legal: legal.into(),
            projection,
            key,
            encoded: Arc::new(encoded),
            indices: indices.into(),
        });
        // One slot, at most 112*64 f32 values + HISTORY_FRAMES + POLICY_SIZE
        // bounded move/index arrays. The weak identity does not retain Rules.
        *memo = Some(Arc::clone(&prepared));
        Ok(prepared)
    }
}

#[cfg(feature = "experimental-prepared-input")]
fn check_prepared_key(
    key: EvalInputKey,
    request: &EvalRequest<RulesState>,
) -> Result<(), ContractError> {
    if key != request.context().input {
        return Err(failed(
            ErrorCode::IdentityMismatch,
            "actual tensor identity differs from request",
        ));
    }
    Ok(())
}

#[cfg(feature = "experimental-prepared-input")]
pub(crate) struct PreparedRulesInput {
    identity: rz_position::WeakPositionIdentity,
    legal: Arc<[Move]>,
    pub(crate) projection: RulesProjection,
    pub(crate) key: EvalInputKey,
    pub(crate) encoded: Arc<EncodedInput>,
    pub(crate) indices: Arc<[usize]>,
}

#[cfg(feature = "experimental-prepared-input")]
impl std::fmt::Debug for PreparedRulesInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedRulesInput")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

fn read_pieces(
    bitboards: &[u64; 12],
    piece_at: impl Fn(Square) -> Option<Piece>,
) -> Result<[[u64; 6]; 2], ContractError> {
    if cfg!(feature = "experimental-bitboards") {
        return Ok(std::array::from_fn(|color| {
            std::array::from_fn(|kind| bitboards[color * 6 + kind])
        }));
    }
    let mut pieces = [[0; 6]; 2];
    for index in 0..64 {
        if let Some(piece) = piece_at(Square::new(index)?) {
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
    Ok(pieces)
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
