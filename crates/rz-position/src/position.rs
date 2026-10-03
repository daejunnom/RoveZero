use crate::{fen, movegen, types::*};
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct RepetitionIdentity {
    pieces: [u64; 12],
    side: Color,
    castling: u8,
    legal_ep: Option<Square>,
}
impl RepetitionIdentity {
    fn of(state: &CoreState) -> Self {
        Self {
            pieces: state.board.bitboards,
            side: state.side,
            castling: state.castling,
            legal_ep: movegen::legal_ep(state),
        }
    }
}

struct HistoryNode {
    state: CoreState,
    repetition: RepetitionIdentity,
    previous: Option<Arc<HistoryNode>>,
    len: usize,
    irreversible: bool,
}
impl std::fmt::Debug for HistoryNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HistoryNode")
            .field("fen", &fen::format(&self.state))
            .field("len", &self.len)
            .field("irreversible", &self.irreversible)
            .finish_non_exhaustive()
    }
}
impl Drop for HistoryNode {
    fn drop(&mut self) {
        // Releasing a long unique prefix must not recursively consume stack.
        // Shared prefixes remain owned by snapshots/branches until they drop.
        let mut previous = self.previous.take();
        while let Some(node) = previous {
            match Arc::try_unwrap(node) {
                Ok(mut owned) => previous = owned.previous.take(),
                Err(_) => break,
            }
        }
    }
}

pub const RULES_VERSION: &str = "rz-position/0.1.0";
pub const RULES_VARIANT: &str = "standard_chess";
pub const DEAD_POSITION_PROFILE: &str = "proven_material_subset_v1";
const MAX_PERFT_STACK_DEPTH: u32 = 64;

/// Exact, position-local rule identity. Equality verifies full known history;
/// a short hash is never sufficient proof that two states are interchangeable.
#[derive(Clone, Debug)]
pub struct PositionIdentity {
    history: Arc<HistoryNode>,
    completeness: HistoryCompleteness,
    origin: HistoryOrigin,
}
impl PartialEq for PositionIdentity {
    fn eq(&self, other: &Self) -> bool {
        if self.completeness != other.completeness
            || self.origin != other.origin
            || self.history.len != other.history.len
        {
            return false;
        }
        let mut a = Some(self.history.as_ref());
        let mut b = Some(other.history.as_ref());
        while let (Some(x), Some(y)) = (a, b) {
            if std::ptr::eq(x, y) {
                return true;
            }
            if x.state != y.state || x.irreversible != y.irreversible {
                return false;
            }
            a = x.previous.as_deref();
            b = y.previous.as_deref();
        }
        a.is_none() && b.is_none()
    }
}
impl Eq for PositionIdentity {}

/// An owned immutable view, including all known raw rule/model history.
/// Its revision is a live-view token, not a repetition or model cache key.
#[derive(Clone, Debug)]
pub struct PositionSnapshot {
    identity: PositionIdentity,
    owner: Arc<()>,
    revision: u64,
}
impl PositionSnapshot {
    pub fn rules_version(&self) -> &'static str {
        RULES_VERSION
    }
    pub fn variant(&self) -> &'static str {
        RULES_VARIANT
    }
    pub fn to_fen(&self) -> String {
        fen::format(&self.identity.history.state)
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn side_to_move(&self) -> Color {
        self.identity.history.state.side
    }
    pub fn piece_at(&self, s: Square) -> Option<Piece> {
        self.identity.history.state.board.get(s)
    }
    pub fn halfmove_clock(&self) -> u32 {
        self.identity.history.state.halfmove
    }
    pub fn fullmove_number(&self) -> u32 {
        self.identity.history.state.fullmove
    }
    pub fn castling_rights(&self) -> u8 {
        self.identity.history.state.castling
    }
    pub fn en_passant_target(&self) -> Option<Square> {
        self.identity.history.state.ep
    }
    pub fn history_completeness(&self) -> HistoryCompleteness {
        self.identity.completeness
    }
    pub fn history_origin(&self) -> HistoryOrigin {
        self.identity.origin
    }
    pub fn known_history_len(&self) -> usize {
        self.identity.history.len
    }
    #[cfg(all(feature = "contracts", not(feature = "experimental-history-digest")))]
    pub(crate) fn is_irreversible_boundary(&self) -> bool {
        self.identity.history.irreversible
    }
    #[cfg(feature = "experimental-history-digest")]
    pub(crate) fn history_states(&self) -> impl Iterator<Item = (&CoreState, bool)> + '_ {
        let mut at = Some(self.identity.history.as_ref());
        std::iter::from_fn(move || {
            let node = at?;
            at = node.previous.as_deref();
            Some((&node.state, node.irreversible))
        })
    }
    /// Explicit audit helper. Encoding reads raw history without fabricating fill.
    pub fn known_history_fens(&self) -> Vec<String> {
        let mut result = Vec::with_capacity(self.known_history_len());
        let mut at = Some(self.identity.history.as_ref());
        while let Some(node) = at {
            result.push(fen::format(&node.state));
            at = node.previous.as_deref();
        }
        result
    }
    /// Newest first; each view owns its prefix and remains valid independently.
    pub fn known_history(&self) -> impl Iterator<Item = PositionSnapshot> + '_ {
        let mut at = Some(self.identity.history.clone());
        std::iter::from_fn(move || {
            let node = at.take()?;
            at = node.previous.clone();
            Some(Self {
                identity: PositionIdentity {
                    history: node,
                    completeness: self.identity.completeness,
                    origin: self.identity.origin,
                },
                owner: self.owner.clone(),
                revision: self.revision,
            })
        })
    }
    pub fn position_identity(&self) -> PositionIdentity {
        self.identity.clone()
    }
    pub fn repetition_identity(&self) -> RepetitionIdentity {
        self.identity.history.repetition.clone()
    }
    pub fn same_state(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

#[derive(Clone, Debug)]
pub struct LegalMoveView {
    snapshot: PositionSnapshot,
    moves: Arc<[BoardMove]>,
}
impl LegalMoveView {
    pub fn snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    pub fn moves(&self) -> &[BoardMove] {
        &self.moves
    }
    /// Exact ordered semantic moves; adapters must not treat a hash as proof.
    pub fn same_order(&self, other: &Self) -> bool {
        self.snapshot.same_state(&other.snapshot) && self.moves == other.moves
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PieceChange {
    pub square: Square,
    pub piece: Piece,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleMoveKind {
    pub capture: bool,
    pub castling: bool,
    pub en_passant: bool,
    pub promotion: bool,
}
#[derive(Clone, Debug)]
pub struct RuleMoveDelta {
    pub mv: BoardMove,
    pub kind: RuleMoveKind,
    pub removals: Vec<PieceChange>,
    pub additions: Vec<PieceChange>,
    pub before: PositionSnapshot,
    pub after: PositionSnapshot,
}
/// Single-use LIFO undo proof. A foreign owner or diverged child is rejected.
#[derive(Debug)]
pub struct UndoToken {
    owner: Arc<()>,
    parent: Arc<HistoryNode>,
    child: Arc<HistoryNode>,
    delta: RuleMoveDelta,
}
impl UndoToken {
    pub fn delta(&self) -> &RuleMoveDelta {
        &self.delta
    }
}

#[derive(Debug)]
pub struct Position {
    state: CoreState,
    history: Arc<HistoryNode>,
    completeness: HistoryCompleteness,
    origin: HistoryOrigin,
    owner: Arc<()>,
    revision: u64,
    limits: PositionLimits,
}
/// One checked transition, shared by live mutation and immutable previews.
/// Undo deltas belong only to the mutation path, not to claim/search previews.
struct GeneratedChild {
    state: CoreState,
    history: Arc<HistoryNode>,
    revision: u64,
    kind: RuleMoveKind,
}
// Checked board transition without a published owner or allocated history node.
// Both materialized children and experimental claim evidence use these checks.
struct CheckedTransition {
    state: CoreState,
    repetition: RepetitionIdentity,
    revision: u64,
    irreversible: bool,
    kind: RuleMoveKind,
}

#[cfg(feature = "experimental-claim-preview")]
pub(crate) struct RepetitionEvidence {
    pub count: usize,
    pub complete: bool,
}
impl Clone for Position {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            history: self.history.clone(),
            completeness: self.completeness,
            origin: self.origin,
            owner: Arc::new(()),
            revision: self.revision,
            limits: self.limits,
        }
    }
}
impl Default for Position {
    fn default() -> Self {
        Self::startpos()
    }
}
impl Position {
    pub fn startpos() -> Self {
        Self::startpos_with_limits(PositionLimits::default())
            .expect("default limits admit startpos")
    }
    pub fn startpos_with_limits(limits: PositionLimits) -> Result<Self, PositionError> {
        Self::new(
            fen::parse(fen::START_FEN, limits)?,
            limits,
            HistoryCompleteness::Complete,
            HistoryOrigin::StartPosition,
        )
    }
    pub fn from_fen(fen: &str) -> Result<Self, PositionError> {
        Self::from_fen_with_limits(fen, PositionLimits::default())
    }
    pub fn from_fen_with_limits(fen: &str, limits: PositionLimits) -> Result<Self, PositionError> {
        Self::new(
            fen::parse(fen, limits)?,
            limits,
            HistoryCompleteness::UnknownPrefix,
            HistoryOrigin::Fen,
        )
    }
    fn new(
        state: CoreState,
        limits: PositionLimits,
        completeness: HistoryCompleteness,
        origin: HistoryOrigin,
    ) -> Result<Self, PositionError> {
        if limits.max_perft_depth > MAX_PERFT_STACK_DEPTH {
            return Err(PositionError::ResourceLimit(
                "perft depth configuration exceeds 64",
            ));
        }
        if limits.max_history_positions == 0 {
            return Err(PositionError::ResourceLimit("history positions"));
        }
        let history = Arc::new(HistoryNode {
            repetition: RepetitionIdentity::of(&state),
            state: state.clone(),
            previous: None,
            len: 1,
            irreversible: completeness == HistoryCompleteness::Complete,
        });
        Ok(Self {
            state,
            history,
            completeness,
            origin,
            owner: Arc::new(()),
            revision: 0,
            limits,
        })
    }
    pub fn to_fen(&self) -> String {
        fen::format(&self.state)
    }
    pub fn side_to_move(&self) -> Color {
        self.state.side
    }
    pub fn piece_at(&self, s: Square) -> Option<Piece> {
        self.state.board.get(s)
    }
    pub fn halfmove_clock(&self) -> u32 {
        self.state.halfmove
    }
    pub fn fullmove_number(&self) -> u32 {
        self.state.fullmove
    }
    pub fn castling_rights(&self) -> u8 {
        self.state.castling
    }
    pub fn en_passant_target(&self) -> Option<Square> {
        self.state.ep
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn history_completeness(&self) -> HistoryCompleteness {
        self.completeness
    }
    pub fn history_origin(&self) -> HistoryOrigin {
        self.origin
    }
    pub fn known_history_len(&self) -> usize {
        self.history.len
    }
    pub fn position_identity(&self) -> PositionIdentity {
        self.snapshot().identity
    }
    pub fn repetition_identity(&self) -> RepetitionIdentity {
        self.history.repetition.clone()
    }
    pub fn in_check(&self) -> bool {
        movegen::is_attacked(
            &self.state,
            self.state.board.king(self.state.side),
            self.state.side.opposite(),
        )
    }
    pub fn is_attacked(&self, square: Square, by: Color) -> bool {
        movegen::is_attacked(&self.state, square, by)
    }
    pub fn legal_moves(&self) -> Vec<BoardMove> {
        movegen::legal(&self.state)
    }
    pub fn snapshot(&self) -> PositionSnapshot {
        PositionSnapshot {
            identity: PositionIdentity {
                history: self.history.clone(),
                completeness: self.completeness,
                origin: self.origin,
            },
            owner: self.owner.clone(),
            revision: self.revision,
        }
    }
    pub fn ordered_legal_moves(&self) -> LegalMoveView {
        LegalMoveView {
            snapshot: self.snapshot(),
            moves: self.legal_moves().into(),
        }
    }
    pub fn matches_snapshot(&self, snapshot: &PositionSnapshot) -> bool {
        Arc::ptr_eq(&self.owner, &snapshot.owner)
            && self.revision == snapshot.revision
            && Arc::ptr_eq(&self.history, &snapshot.identity.history)
    }
    pub fn make_from_view(
        &mut self,
        view: &LegalMoveView,
        mv: BoardMove,
    ) -> Result<UndoToken, PositionError> {
        if !self.matches_snapshot(&view.snapshot) {
            return Err(PositionError::StaleView);
        }
        if !view.moves.contains(&mv) {
            return Err(PositionError::IllegalMove);
        }
        self.make_generated(mv)
    }
    /// A fork checks the live source's view before cloning its immutable history.
    /// The clone owns a fresh concrete owner; the source view cannot mutate it.
    #[cfg(feature = "contracts")]
    pub(crate) fn fork_from_view(
        &self,
        view: &LegalMoveView,
        mv: BoardMove,
    ) -> Result<Self, PositionError> {
        if !self.matches_snapshot(&view.snapshot) {
            return Err(PositionError::StaleView);
        }
        if !view.moves.contains(&mv) {
            return Err(PositionError::IllegalMove);
        }
        self.preview_generated(mv)
    }
    /// Applies board-legal moves. Search/arena must consume classification first
    /// to enforce terminal/claim game policy; perft counts board-legal moves.
    pub fn make_move(&mut self, mv: BoardMove) -> Result<UndoToken, PositionError> {
        if !self.legal_moves().contains(&mv) {
            return Err(PositionError::IllegalMove);
        }
        self.make_generated(mv)
    }
    pub fn make_uci(&mut self, mv: &str) -> Result<UndoToken, PositionError> {
        self.make_move(mv.parse()?)
    }
    fn make_generated(&mut self, mv: BoardMove) -> Result<UndoToken, PositionError> {
        let generated = self.prepare_generated(mv)?;
        let before = self.snapshot();
        let mut removals = Vec::new();
        let mut additions = Vec::new();
        for i in 0..64 {
            let square = Square(i);
            let old = self.state.board.get(square);
            let new = generated.state.board.get(square);
            if old != new {
                if let Some(piece) = old {
                    removals.push(PieceChange { square, piece });
                }
                if let Some(piece) = new {
                    additions.push(PieceChange { square, piece });
                }
            }
        }
        let parent = self.history.clone();
        let child = generated.history.clone();
        let kind = generated.kind;
        self.install_generated(generated);
        let delta = RuleMoveDelta {
            mv,
            kind,
            removals,
            additions,
            before,
            after: self.snapshot(),
        };
        Ok(UndoToken {
            owner: self.owner.clone(),
            parent,
            child,
            delta,
        })
    }
    /// No state escapes until all history, revision and counter checks pass.
    fn prepare_generated(&self, mv: BoardMove) -> Result<GeneratedChild, PositionError> {
        let checked = self.check_generated(mv)?;
        let history = Arc::new(HistoryNode {
            state: checked.state.clone(),
            repetition: checked.repetition,
            previous: Some(self.history.clone()),
            len: self.history.len + 1,
            irreversible: checked.irreversible,
        });
        Ok(GeneratedChild {
            state: checked.state,
            history,
            revision: checked.revision,
            kind: checked.kind,
        })
    }

    fn check_generated(&self, mv: BoardMove) -> Result<CheckedTransition, PositionError> {
        if self.history.len >= self.limits.max_history_positions {
            return Err(PositionError::ResourceLimit("history positions"));
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(PositionError::RevisionExhausted)?;
        let piece = self.state.board.get(mv.from).expect("checked legal move");
        let mut child_state = self.state.clone();
        let ep = piece.kind == PieceKind::Pawn
            && self.state.ep == Some(mv.to)
            && self.state.board.get(mv.to).is_none()
            && mv.from.file() != mv.to.file();
        let capture = movegen::apply(&mut child_state, mv);
        if piece.kind != PieceKind::Pawn && !capture {
            child_state.halfmove = self
                .state
                .halfmove
                .checked_add(1)
                .ok_or(PositionError::CounterOverflow)?;
        }
        if self.state.side == Color::Black {
            child_state.fullmove = self
                .state
                .fullmove
                .checked_add(1)
                .ok_or(PositionError::CounterOverflow)?;
        }
        let irreversible =
            piece.kind == PieceKind::Pawn || capture || child_state.castling != self.state.castling;
        Ok(CheckedTransition {
            repetition: RepetitionIdentity::of(&child_state),
            state: child_state,
            revision,
            irreversible,
            kind: RuleMoveKind {
                capture,
                castling: piece.kind == PieceKind::King
                    && mv.from.file().abs_diff(mv.to.file()) == 2,
                en_passant: ep,
                promotion: mv.promotion.is_some(),
            },
        })
    }
    fn install_generated(&mut self, generated: GeneratedChild) {
        self.state = generated.state;
        self.history = generated.history;
        self.revision = generated.revision;
    }
    pub fn unmake(&mut self, token: UndoToken) -> Result<(), PositionError> {
        if !Arc::ptr_eq(&self.owner, &token.owner) || !Arc::ptr_eq(&self.history, &token.child) {
            return Err(PositionError::UndoMismatch);
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(PositionError::RevisionExhausted)?;
        self.state = token.parent.state.clone();
        self.history = token.parent;
        self.revision = revision;
        Ok(())
    }
    /// All-or-nothing input transaction. Intermediate states never escape.
    pub fn apply_uci_moves(&mut self, moves: &[&str]) -> Result<(), PositionError> {
        if moves.is_empty() {
            return Ok(());
        }
        let mut trial = self.clone();
        for mv in moves {
            trial.make_uci(mv)?;
        }
        self.state = trial.state;
        self.history = trial.history;
        self.revision = trial.revision;
        // A transaction replaces the mutable owner; earlier live views/undo proofs
        // remain valid immutable evidence but cannot mutate this new owner.
        self.owner = trial.owner;
        Ok(())
    }
    pub fn preview_move(&self, mv: BoardMove) -> Result<Self, PositionError> {
        let mut child = self.clone();
        child.make_move(mv)?;
        Ok(child)
    }
    /// Classifier already owns a generated legal list for this immutable state.
    pub(crate) fn preview_generated(&self, mv: BoardMove) -> Result<Self, PositionError> {
        let mut child = self.clone();
        let generated = child.prepare_generated(mv)?;
        child.install_generated(generated);
        Ok(child)
    }

    #[cfg(feature = "experimental-claim-preview")]
    pub(crate) fn repetition_evidence(&self) -> RepetitionEvidence {
        repetition_evidence(Some(self.history.as_ref()), &self.history.repetition, 0)
    }

    /// Check the same transition as a real move, then compare its repetition
    /// identity against borrowed history. The intended child counts once even
    /// though no owner/history node is constructed. An irreversible child cuts
    /// the repetition window, without completing imported model/game history.
    #[cfg(feature = "experimental-claim-preview")]
    pub(crate) fn intended_claim_evidence(
        &self,
        mv: BoardMove,
    ) -> Result<(RepetitionEvidence, u32), PositionError> {
        let checked = self.check_generated(mv)?;
        let evidence = if checked.irreversible {
            RepetitionEvidence {
                count: 1,
                complete: true,
            }
        } else {
            repetition_evidence(Some(self.history.as_ref()), &checked.repetition, 1)
        };
        Ok((evidence, checked.state.halfmove))
    }
    pub fn known_repetition_count(&self) -> usize {
        let mut count = 0;
        let mut at = Some(self.history.as_ref());
        while let Some(node) = at {
            if node.repetition == self.history.repetition {
                count += 1;
            }
            if node.irreversible {
                break;
            }
            at = node.previous.as_deref();
        }
        count
    }
    pub fn repetition_history_complete(&self) -> bool {
        let mut at = Some(self.history.as_ref());
        while let Some(node) = at {
            if node.irreversible {
                return true;
            }
            at = node.previous.as_deref();
        }
        false
    }
    pub fn perft(&self, depth: u32) -> Result<u64, PositionError> {
        if depth > self.limits.max_perft_depth {
            return Err(PositionError::ResourceLimit("perft depth"));
        }
        let mut budget = self.limits.max_perft_nodes;
        count_nodes(&self.state, depth, &mut budget)
    }
    pub fn divide(&self, depth: u32) -> Result<Vec<(BoardMove, u64)>, PositionError> {
        if depth == 0 {
            return Err(PositionError::InvalidMove("divide requires positive depth"));
        }
        if depth > self.limits.max_perft_depth {
            return Err(PositionError::ResourceLimit("perft depth"));
        }
        let mut budget = self
            .limits
            .max_perft_nodes
            .checked_sub(1)
            .ok_or(PositionError::ResourceLimit("perft traversal nodes"))?;
        let mut result = Vec::new();
        for mv in self.legal_moves() {
            let mut child = self.state.clone();
            movegen::apply(&mut child, mv);
            result.push((mv, count_nodes(&child, depth - 1, &mut budget)?));
        }
        Ok(result)
    }
}

#[cfg(feature = "experimental-claim-preview")]
fn repetition_evidence(
    mut at: Option<&HistoryNode>,
    identity: &RepetitionIdentity,
    mut count: usize,
) -> RepetitionEvidence {
    while let Some(node) = at {
        if node.repetition == *identity {
            count += 1;
        }
        if node.irreversible {
            return RepetitionEvidence {
                count,
                complete: true,
            };
        }
        at = node.previous.as_deref();
    }
    RepetitionEvidence {
        count,
        complete: false,
    }
}

fn count_nodes(state: &CoreState, depth: u32, budget: &mut u64) -> Result<u64, PositionError> {
    *budget = budget
        .checked_sub(1)
        .ok_or(PositionError::ResourceLimit("perft traversal nodes"))?;
    if depth == 0 {
        return Ok(1);
    }
    let moves = movegen::legal(state);
    if depth == 1 {
        let count = moves.len() as u64;
        *budget = budget
            .checked_sub(count)
            .ok_or(PositionError::ResourceLimit("perft traversal nodes"))?;
        return Ok(count);
    }
    let mut total = 0u64;
    for mv in moves {
        let mut child = state.clone();
        movegen::apply(&mut child, mv);
        total = total
            .checked_add(count_nodes(&child, depth - 1, budget)?)
            .ok_or(PositionError::ResourceLimit("perft count overflow"))?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhausted_revision_never_wraps_or_partially_changes_state() {
        let mut p = Position::startpos();
        p.revision = u64::MAX;
        let view = p.ordered_legal_moves();
        let before = p.snapshot();
        assert!(matches!(
            p.make_uci("e2e4"),
            Err(PositionError::RevisionExhausted)
        ));
        assert!(matches!(
            p.apply_uci_moves(&["e2e4"]),
            Err(PositionError::RevisionExhausted)
        ));
        assert!(matches!(
            p.preview_generated(BoardMove::from_uci("e2e4").unwrap()),
            Err(PositionError::RevisionExhausted)
        ));
        assert!(matches!(
            p.classify_position(),
            Err(PositionError::RevisionExhausted)
        ));
        assert!(p.snapshot().same_state(&before));
        assert!(p.matches_snapshot(view.snapshot()));

        p.revision = u64::MAX - 1;
        let token = p.make_uci("e2e4").unwrap();
        let child = p.snapshot();
        assert!(matches!(
            p.unmake(token),
            Err(PositionError::RevisionExhausted)
        ));
        assert!(p.snapshot().same_state(&child));
        assert_eq!(p.revision(), u64::MAX);
    }

    #[cfg(feature = "experimental-claim-preview")]
    #[test]
    fn borrowed_claim_evidence_matches_owned_previews_and_preserves_live_views() {
        let mut cases = vec![
            Position::startpos(),
            Position::from_fen(fen::START_FEN).unwrap(),
        ];
        for input in [
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 2",
            "4k3/8/8/r4pPK/8/8/8/8 w - f6 0 2",
            "1r2k3/P7/8/8/8/8/7p/R3K3 w Q - 99 1",
            "7k/8/8/8/8/8/P7/KR6 w - - 99 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 4294967295",
            "7k/8/8/8/8/8/P7/KR6 w - - 4294967295 1",
        ] {
            cases.push(Position::from_fen(input).unwrap());
        }
        for start in [
            Position::startpos(),
            Position::from_fen(fen::START_FEN).unwrap(),
        ] {
            let mut repeated = start;
            for index in 0..15 {
                repeated
                    .make_uci(["g1f3", "g8f6", "f3g1", "f6g8"][index % 4])
                    .unwrap();
                cases.push(repeated.clone());
            }
        }

        for mut position in cases {
            for step in 0..24 {
                let before = position.ordered_legal_moves();
                let evidence = position.repetition_evidence();
                assert_eq!(evidence.count, position.known_repetition_count());
                assert_eq!(evidence.complete, position.repetition_history_complete());
                for &mv in before.moves() {
                    let expected = position.preview_generated(mv).map(|child| {
                        (
                            child.known_repetition_count(),
                            child.repetition_history_complete(),
                            child.halfmove_clock(),
                        )
                    });
                    let actual = position
                        .intended_claim_evidence(mv)
                        .map(|(evidence, halfmove)| (evidence.count, evidence.complete, halfmove));
                    assert_eq!(actual, expected, "{} / {mv}", position.to_fen());
                    assert!(position.matches_snapshot(before.snapshot()));
                }
                if before.moves().is_empty() {
                    break;
                }
                let mv = before.moves()[(step * 17 + 3) % before.moves().len()];
                if position.make_from_view(&before, mv).is_err() {
                    break;
                }
            }
        }
    }

    #[cfg(feature = "experimental-claim-preview")]
    #[test]
    fn claim_preview_keeps_checked_error_precedence_without_mutation() {
        let mv = BoardMove::from_uci("e2e4").unwrap();
        let mut position = Position::startpos_with_limits(PositionLimits {
            max_history_positions: 1,
            ..PositionLimits::default()
        })
        .unwrap();
        position.revision = u64::MAX;
        let before = position.snapshot();
        assert!(matches!(
            position.intended_claim_evidence(mv),
            Err(PositionError::ResourceLimit("history positions"))
        ));
        position.limits.max_history_positions = 2;
        assert!(matches!(
            position.intended_claim_evidence(mv),
            Err(PositionError::RevisionExhausted)
        ));
        assert!(position.matches_snapshot(&before));
        assert!(position.snapshot().same_state(&before));
    }
}
