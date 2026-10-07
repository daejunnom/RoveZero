//! Finite, training-private Rules branch preparation; no search or model call.
//!
//! The descriptors and ordered Move16 tokens identify a checked branch. They
//! are not neural features, targets, ranks, utility estimates or tactical proof.
//! Parent-input and before-result anchors are caller declarations, sealed into
//! request context but not independently registered here. The binary argument
//! is compared exactly; the CLI owns hashing its running image on the original
//! clock. This module alone does not connect V, a private frontend or DG04.

use super::{
    CpuTaskError, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, MAX_WALL_TIME_MS, canonical, decode_moves,
    history_sha, json_digest, milliseconds, pack_moves, replay_checked, state_sha, valid_sha,
};
use crate::engine::{OwnerRegistry, RulesUciPort};
use crate::pals_native::PALS_RULES_ENCODING;
use crate::{Command, ParserLimits, PositionPort, PositionSpec, parse};
use rz_position::{
    BoardMove, Color, HistoryCompleteness, HistoryOrigin, PieceKind, PlayStatus, Position,
    PositionLimits, Square,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const SEMANTIC_SCHEMA: &str = "rz-pals-private-semantic-branch/1";
pub const SEMANTIC_VERSION: &str = "rz-pals-rules-branch-preparation/1";
pub const DESCRIPTOR_SCHEMA: &str = "rz-pals-private-rules-descriptor/1";
pub const MOVE_ORDER_DOMAIN: &str = "rz-pals-private-semantic-move-order/1";
pub const BRANCH_MEANING_DOMAIN: &str = "rz-pals-private-semantic-branch-meaning/1";
pub const DECLARATION_SCOPE: &str = "caller_declared_not_independently_registered";
pub const BINARY_PIN_SCOPE: &str = "dispatcher_compared_verified_argument";
pub const MEANING_SCOPE: &str = "rules_checked_branch_only;hashes_are_identity_not_neural_features;no_search_no_model_no_target_no_rank_no_utility_no_tactical_proof";
pub const MAX_PREFIX_PLIES: usize = 64;
pub const MAX_CLAIM_PLIES: usize = 64;
pub const MAX_ROOT_MOVES: usize = 256;
pub const BOARD_PIECE_CODE_SEMANTICS: &str =
    "a1=0,h8=63;empty=0;white_PNBRQK=1..6;black_PNBRQK=7..12;exact_rules_piece_at";
pub const CASTLING_BIT_SEMANTICS: &str =
    "bit0:white_king;bit1:white_queen;bit2:black_king;bit3:black_queen;exact_rules_rights";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticQuestion {
    RestrictedResponse,
    ContinuationChallenge,
    UnrestrictedRecheck,
}

/// Bounded declarations only. No CPU task, score, depth, NN input or target is
/// accepted by this schema. Empty claimed_line is explicitly an unknown claim.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticRequest {
    pub schema: String,
    pub question: SemanticQuestion,
    pub parent_input_sha256: String,
    pub before_result_anchor_sha256: String,
    pub position_command: String,
    pub expected_board_fen: String,
    pub rules_state_sha256: String,
    pub rules_history_sha256: String,
    pub prefix: Vec<u16>,
    pub root_moves: Vec<u16>,
    pub claimed_line: Vec<u16>,
    pub current_binary_sha256: String,
    pub max_wall_time_ms: u64,
    pub max_output_bytes: usize,
    pub context_sha256: String,
}

#[derive(Clone, Copy, Debug)]
pub struct SemanticAdmission {
    pub deadline: Instant,
    pub output_limit: usize,
}

/// A failed preparation remains a failed run. A captured descriptor, when
/// present, is diagnostic evidence rather than an admitted semantic result.
#[derive(Clone, Debug, Serialize)]
pub struct SemanticError {
    pub schema: &'static str,
    pub code: &'static str,
    pub stage: &'static str,
    pub message: Box<str>,
    pub cpu_checks: u8,
    pub elapsed_ms: Option<u64>,
    pub deadline_exceeded: bool,
    pub receipt: Option<Box<SemanticReceipt>>,
    #[serde(skip)]
    pub output_limit: usize,
}

impl SemanticError {
    pub fn new(stage: &'static str, message: impl fmt::Display) -> Self {
        Self {
            schema: SEMANTIC_SCHEMA,
            code: "semantic_branch_failed",
            stage,
            message: message
                .to_string()
                .chars()
                .take(512)
                .collect::<String>()
                .into_boxed_str(),
            cpu_checks: 0,
            elapsed_ms: None,
            deadline_exceeded: false,
            receipt: None,
            output_limit: 1024,
        }
    }

    fn with_receipt(mut self, receipt: SemanticReceipt) -> Self {
        self.receipt = Some(Box::new(receipt));
        self
    }
}

impl fmt::Display for SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.stage, self.message)
    }
}
impl std::error::Error for SemanticError {}

fn shared_error(error: CpuTaskError) -> SemanticError {
    SemanticError::new(error.stage, error.message)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SideToMove {
    White,
    Black,
}
impl From<Color> for SideToMove {
    fn from(color: Color) -> Self {
        match color {
            Color::White => Self::White,
            Color::Black => Self::Black,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoveTokenKind {
    RootLegal,
    TargetLegal,
    Prefix,
    RootRestriction,
    EffectiveRootLegal,
    ClaimedContinuation,
    ClaimEndLegal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionMeaning {
    Queen,
    Rook,
    Bishop,
    Knight,
}

/// slot is this declared stream's order, not a priority/rank. a1=0,h8=63.
/// For masks, legal_order_slot links to the target's complete Rules order.
/// Castle/EP legality is established by Rules replay, not inferred from bits.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoveMeaning {
    pub token_kind: MoveTokenKind,
    pub slot: u16,
    pub legal_order_slot: Option<u16>,
    pub move16: u16,
    pub from: u8,
    pub to: u8,
    pub promotion: Option<PromotionMeaning>,
    pub uci: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RulesDescriptor {
    pub schema: String,
    pub rules_version: String,
    pub rules_variant: String,
    pub rules_state_sha256: String,
    pub rules_history_sha256: String,
    pub board_fen: String,
    pub side_to_move: SideToMove,
    /// Exact Rules representation, not a model tensor or FEN reconstruction.
    pub board64_piece_codes: Vec<u8>,
    pub piece_code_semantics: String,
    pub pals_rules_encoding: String,
    pub castling_rights: u8,
    pub castling_bit_semantics: String,
    pub en_passant_square: Option<u8>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
    pub in_check: bool,
    pub history_completeness: String,
    pub history_origin: String,
    pub known_history_positions: usize,
    /// Exact count over Rules' known repetition scope; unknown prefix is not
    /// invented. Rules stops at its irreversible boundary, never at a hash guess.
    pub known_repetition_count: usize,
    pub repetition_history_complete: bool,
    pub repetition_scope: String,
    /// Geometric legal order remains complete even at automatic terminals.
    pub legal_moves: Vec<u16>,
    pub legal_order_sha256: String,
    pub legal_tokens: Vec<MoveMeaning>,
    pub play_status: String,
    pub terminal_reason: Option<String>,
    pub terminal_winner: Option<SideToMove>,
    pub terminal_source: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootRestriction {
    /// Request order is preserved independently of Rules' effective root order.
    pub declared_order: Vec<MoveMeaning>,
    pub effective_legal_order: Vec<MoveMeaning>,
    pub declared_order_sha256: String,
    pub effective_order_sha256: String,
    pub scope: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    NoClaim,
    LegalContinuation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimDescriptor {
    pub status: ClaimStatus,
    /// None for no_claim; true only means the provided continuation is legal.
    pub legality_verified: Option<bool>,
    /// Always unknown: replay is not a truth/score/tactical-utility check.
    pub claim_truth: String,
    pub movements: Vec<MoveMeaning>,
    pub restriction_checked: bool,
    pub final_state: Option<RulesDescriptor>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticResourcePolicy {
    pub max_wall_time_ms: u64,
    pub max_output_bytes: usize,
    pub max_prefix_plies: usize,
    pub max_claim_plies: usize,
    pub max_root_moves: usize,
    pub cpu_checks: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticReceipt {
    pub schema: String,
    pub implementation: String,
    pub context_sha256: String,
    pub current_binary_sha256: String,
    pub binary_pin_scope: String,
    pub parent_input_sha256: String,
    pub before_result_anchor_sha256: String,
    pub caller_declaration_scope: String,
    pub meaning_scope: String,
    pub question: SemanticQuestion,
    pub root: RulesDescriptor,
    pub prefix: Vec<MoveMeaning>,
    pub target: RulesDescriptor,
    pub root_restriction: Option<RootRestriction>,
    pub claimed_line: ClaimDescriptor,
    pub branch_meaning_sha256: String,
    pub resource_policy: SemanticResourcePolicy,
    pub cpu_checks: u8,
    pub search_executed: bool,
    pub model_executed: bool,
    pub training_target_created: bool,
    pub product_verifier_enabled: bool,
    pub elapsed_ms: u64,
    pub deadline_exceeded: bool,
}

/// Canonical [schema, complete request without context_sha256]. Every ordered
/// control, declaration, question and original budget is sealed; no sorting of
/// prefix, mask or claimed continuation is allowed.
pub fn request_context_sha256(request: &SemanticRequest) -> Result<String, SemanticError> {
    let mut value = serde_json::to_value(request)
        .map_err(|error| SemanticError::new("context_identity", error))?;
    value
        .as_object_mut()
        .ok_or_else(|| SemanticError::new("context_identity", "request is not an object"))?
        .remove("context_sha256");
    json_digest(&json!([SEMANTIC_SCHEMA, value])).map_err(shared_error)
}

fn decode_request(bytes: &[u8]) -> Result<SemanticRequest, SemanticError> {
    if bytes.is_empty() || bytes.len() > MAX_REQUEST_BYTES {
        return Err(SemanticError::new(
            "admission",
            "stdin must contain 1..=512KiB",
        ));
    }
    // Direct struct deserialization rejects both unknown and duplicate fields.
    let request: SemanticRequest = serde_json::from_slice(bytes)
        .map_err(|error| SemanticError::new("json_admission", error))?;
    if request.schema != SEMANTIC_SCHEMA
        || !(1..=MAX_WALL_TIME_MS).contains(&request.max_wall_time_ms)
        || !(1024..=MAX_RESPONSE_BYTES).contains(&request.max_output_bytes)
        || request.prefix.len() > MAX_PREFIX_PLIES
        || request.root_moves.len() > MAX_ROOT_MOVES
        || request.claimed_line.len() > MAX_CLAIM_PLIES
        || request.position_command.len() > ParserLimits::default().max_line_bytes
        || request.expected_board_fen.len() > PositionLimits::default().max_fen_bytes
    {
        return Err(SemanticError::new(
            "admission",
            "unsupported schema or finite bounds",
        ));
    }
    for pin in [
        &request.parent_input_sha256,
        &request.before_result_anchor_sha256,
        &request.rules_state_sha256,
        &request.rules_history_sha256,
        &request.current_binary_sha256,
        &request.context_sha256,
    ] {
        if !valid_sha(pin) {
            return Err(SemanticError::new(
                "admission",
                "identity must be lowercase SHA256",
            ));
        }
    }
    let payload_ok = match request.question {
        SemanticQuestion::RestrictedResponse => !request.root_moves.is_empty(),
        SemanticQuestion::ContinuationChallenge | SemanticQuestion::UnrestrictedRecheck => {
            request.root_moves.is_empty()
        }
    };
    if !payload_ok {
        return Err(SemanticError::new(
            "question_admission",
            "missing or unused root restriction",
        ));
    }
    decode_moves(&request.prefix).map_err(shared_error)?;
    decode_moves(&request.claimed_line).map_err(shared_error)?;
    let roots = decode_moves(&request.root_moves).map_err(shared_error)?;
    if roots
        .iter()
        .enumerate()
        .any(|(at, mv)| roots[..at].contains(mv))
    {
        return Err(SemanticError::new(
            "root_admission",
            "duplicate root restriction",
        ));
    }
    Ok(request)
}

fn absolute_deadline(
    request: &SemanticRequest,
    started: Instant,
) -> Result<Instant, SemanticError> {
    started
        .checked_add(Duration::from_millis(request.max_wall_time_ms))
        .ok_or_else(|| SemanticError::new("admission", "absolute deadline overflow"))
}

fn check_deadline(deadline: Instant, stage: &'static str) -> Result<(), SemanticError> {
    if Instant::now() >= deadline {
        let mut error = SemanticError::new(stage, "original absolute wall allowance expired");
        error.deadline_exceeded = true;
        Err(error)
    } else {
        Ok(())
    }
}

/// CLI calls this before streaming self-binary hashing; started includes stdin.
/// No extra clock is granted to hashing, parsing, replay or serialization.
pub fn request_admission(
    bytes: &[u8],
    started: Instant,
) -> Result<SemanticAdmission, SemanticError> {
    let request = decode_request(bytes)?;
    let deadline = absolute_deadline(&request, started)?;
    check_deadline(deadline, "admission_deadline")?;
    Ok(SemanticAdmission {
        deadline,
        output_limit: request.max_output_bytes,
    })
}

fn meanings(
    moves: &[BoardMove],
    kind: MoveTokenKind,
    legal_order: Option<&[BoardMove]>,
    deadline: Instant,
) -> Result<Vec<MoveMeaning>, SemanticError> {
    let bits = pack_moves(moves).map_err(shared_error)?;
    let mut result = Vec::with_capacity(moves.len());
    for (index, (&mv, &move16)) in moves.iter().zip(&bits).enumerate() {
        check_deadline(deadline, "token_deadline")?;
        let promotion = match mv.promotion {
            None => None,
            Some(PieceKind::Queen) => Some(PromotionMeaning::Queen),
            Some(PieceKind::Rook) => Some(PromotionMeaning::Rook),
            Some(PieceKind::Bishop) => Some(PromotionMeaning::Bishop),
            Some(PieceKind::Knight) => Some(PromotionMeaning::Knight),
            Some(PieceKind::Pawn | PieceKind::King) => {
                return Err(SemanticError::new(
                    "token_admission",
                    "unsupported promotion",
                ));
            }
        };
        let legal_order_slot = legal_order
            .map(|legal| {
                legal
                    .iter()
                    .position(|candidate| *candidate == mv)
                    .ok_or_else(|| {
                        SemanticError::new("token_admission", "move outside legal order")
                    })
                    .and_then(|slot| {
                        u16::try_from(slot)
                            .map_err(|error| SemanticError::new("token_admission", error))
                    })
            })
            .transpose()?;
        result.push(MoveMeaning {
            token_kind: kind,
            slot: u16::try_from(index)
                .map_err(|error| SemanticError::new("token_admission", error))?,
            legal_order_slot,
            move16,
            from: mv.from.index(),
            to: mv.to.index(),
            promotion,
            uci: mv.to_string(),
        });
    }
    check_deadline(deadline, "token_deadline")?;
    Ok(result)
}

fn describe(
    position: &Position,
    owners: &OwnerRegistry,
    legal_kind: MoveTokenKind,
    deadline: Instant,
) -> Result<RulesDescriptor, SemanticError> {
    check_deadline(deadline, "descriptor_deadline")?;
    let view = position.ordered_legal_moves();
    if view.moves().len() > MAX_ROOT_MOVES {
        return Err(SemanticError::new(
            "rules_admission",
            "complete legal order exceeds finite bound",
        ));
    }
    let status = position
        .play_status_from_view(&view)
        .map_err(|error| SemanticError::new("rules_classification", error))?;
    let snapshot = position.snapshot();
    let mut board64_piece_codes = Vec::with_capacity(64);
    for index in 0..64 {
        check_deadline(deadline, "descriptor_deadline")?;
        let square = Square::new(index)
            .map_err(|error| SemanticError::new("rules_representation", error))?;
        board64_piece_codes.push(match snapshot.piece_at(square) {
            None => 0,
            Some(piece) => {
                let kind = match piece.kind {
                    PieceKind::Pawn => 1,
                    PieceKind::Knight => 2,
                    PieceKind::Bishop => 3,
                    PieceKind::Rook => 4,
                    PieceKind::Queen => 5,
                    PieceKind::King => 6,
                };
                kind + if piece.color == Color::White { 0 } else { 6 }
            }
        });
    }
    let legal_moves = pack_moves(view.moves()).map_err(shared_error)?;
    let legal_order_sha256 =
        json_digest(&json!([MOVE_ORDER_DOMAIN, legal_moves])).map_err(shared_error)?;
    let legal_tokens = meanings(view.moves(), legal_kind, Some(view.moves()), deadline)?;
    let rules_state_sha256 = state_sha(position, owners).map_err(shared_error)?;
    let rules_history_sha256 = history_sha(position).map_err(shared_error)?;
    let (play_status, terminal_reason, terminal_winner, terminal_source) = match status {
        PlayStatus::Ongoing => ("ongoing".into(), None, None, None),
        PlayStatus::Terminal { reason, winner } => (
            "rules_terminal".into(),
            Some(format!("{reason:?}")),
            winner.map(SideToMove::from),
            Some("rz-position-rules".into()),
        ),
    };
    let descriptor = RulesDescriptor {
        schema: DESCRIPTOR_SCHEMA.into(),
        rules_version: snapshot.rules_version().into(),
        rules_variant: snapshot.variant().into(),
        rules_state_sha256,
        rules_history_sha256,
        board_fen: position.to_fen(),
        side_to_move: position.side_to_move().into(),
        board64_piece_codes,
        piece_code_semantics: BOARD_PIECE_CODE_SEMANTICS.into(),
        pals_rules_encoding: PALS_RULES_ENCODING.into(),
        castling_rights: snapshot.castling_rights(),
        castling_bit_semantics: CASTLING_BIT_SEMANTICS.into(),
        en_passant_square: snapshot.en_passant_target().map(|square| square.index()),
        halfmove_clock: snapshot.halfmove_clock(),
        fullmove_number: snapshot.fullmove_number(),
        in_check: position.in_check(),
        history_completeness: match snapshot.history_completeness() {
            HistoryCompleteness::Complete => "complete".into(),
            HistoryCompleteness::UnknownPrefix => "unknown_prefix".into(),
        },
        history_origin: match snapshot.history_origin() {
            HistoryOrigin::StartPosition => "start_position".into(),
            HistoryOrigin::Fen => "fen".into(),
        },
        known_history_positions: snapshot.known_history_len(),
        known_repetition_count: position.known_repetition_count(),
        repetition_history_complete: position.repetition_history_complete(),
        repetition_scope:
            "rules_known_prefix_until_irreversible_boundary;not_inferred_from_fen_or_digest".into(),
        legal_moves,
        legal_order_sha256,
        legal_tokens,
        play_status,
        terminal_reason,
        terminal_winner,
        terminal_source,
    };
    check_deadline(deadline, "descriptor_deadline")?;
    Ok(descriptor)
}

fn prepare_root(
    request: &SemanticRequest,
    owners: &Arc<OwnerRegistry>,
    deadline: Instant,
) -> Result<Position, SemanticError> {
    check_deadline(deadline, "position_parse_deadline")?;
    let command = parse(&request.position_command, ParserLimits::default())
        .map_err(|error| SemanticError::new("position_admission", error))?;
    check_deadline(deadline, "position_parse_deadline")?;
    let Command::Position(spec) = command else {
        return Err(SemanticError::new(
            "position_admission",
            "only a position command is accepted",
        ));
    };
    // Reuse RulesUciPort for the exact start/FEN origin, then the parent's
    // checked replay for every known move so the original deadline is polled.
    let base = PositionSpec {
        base: spec.base,
        moves: Vec::new(),
    };
    let prepared = RulesUciPort::new(Arc::clone(owners), PositionLimits::default())
        .prepare(&base)
        .map_err(|error| SemanticError::new("rules_prepare", error))?;
    check_deadline(deadline, "rules_prepare_deadline")?;
    let mut moves = Vec::with_capacity(spec.moves.len());
    for text in &spec.moves {
        check_deadline(deadline, "rules_replay_deadline")?;
        moves.push(
            BoardMove::from_uci(text)
                .map_err(|error| SemanticError::new("position_admission", error))?,
        );
    }
    let root = replay_checked(prepared.snapshot.rules_position(), &moves, owners, deadline)
        .map_err(shared_error)?;
    check_deadline(deadline, "rules_replay_deadline")?;
    Ok(root)
}

fn prepare(
    request: &SemanticRequest,
    verified_binary: &str,
    started: Instant,
    deadline: Instant,
) -> Result<SemanticReceipt, SemanticError> {
    check_deadline(deadline, "binary_identity_deadline")?;
    if !valid_sha(verified_binary) || request.current_binary_sha256 != verified_binary {
        return Err(SemanticError::new(
            "binary_identity",
            "current executable digest differs from admitted binary",
        ));
    }
    check_deadline(deadline, "context_identity_deadline")?;
    if request_context_sha256(request)? != request.context_sha256 {
        return Err(SemanticError::new(
            "context_identity",
            "complete request context digest mismatch",
        ));
    }
    check_deadline(deadline, "context_identity_deadline")?;
    let owners = Arc::new(OwnerRegistry::default());
    let root_position = prepare_root(request, &owners, deadline)?;
    let root = describe(&root_position, &owners, MoveTokenKind::RootLegal, deadline)?;
    if root.rules_state_sha256 != request.rules_state_sha256
        || root.rules_history_sha256 != request.rules_history_sha256
        || root.board_fen != request.expected_board_fen
    {
        return Err(SemanticError::new(
            "rules_identity",
            "actual root board/full known history differs from sealed input",
        ));
    }
    let prefix = decode_moves(&request.prefix).map_err(shared_error)?;
    check_deadline(deadline, "prefix_deadline")?;
    let target_position =
        replay_checked(&root_position, &prefix, &owners, deadline).map_err(shared_error)?;
    check_deadline(deadline, "prefix_deadline")?;
    let target = describe(
        &target_position,
        &owners,
        MoveTokenKind::TargetLegal,
        deadline,
    )?;
    let target_legal = decode_moves(&target.legal_moves).map_err(shared_error)?;
    let roots = decode_moves(&request.root_moves).map_err(shared_error)?;
    let root_restriction = if roots.is_empty() {
        None
    } else {
        if target.play_status != "ongoing" || roots.iter().any(|mv| !target_legal.contains(mv)) {
            return Err(SemanticError::new(
                "root_admission",
                "restriction is illegal or target is Rules terminal",
            ));
        }
        let effective: Vec<_> = target_legal
            .iter()
            .copied()
            .filter(|mv| roots.contains(mv))
            .collect();
        Some(RootRestriction {
            declared_order: meanings(
                &roots,
                MoveTokenKind::RootRestriction,
                Some(&target_legal),
                deadline,
            )?,
            effective_legal_order: meanings(
                &effective,
                MoveTokenKind::EffectiveRootLegal,
                Some(&target_legal),
                deadline,
            )?,
            declared_order_sha256: json_digest(&json!([MOVE_ORDER_DOMAIN, request.root_moves]))
                .map_err(shared_error)?,
            effective_order_sha256: json_digest(&json!([
                MOVE_ORDER_DOMAIN,
                pack_moves(&effective).map_err(shared_error)?
            ]))
            .map_err(shared_error)?,
            scope: "exact_target_root_only;request_order_preserved;effective_order_is_rules_order"
                .into(),
        })
    };
    let claimed = decode_moves(&request.claimed_line).map_err(shared_error)?;
    if !roots.is_empty() && claimed.first().is_some_and(|mv| !roots.contains(mv)) {
        return Err(SemanticError::new(
            "claim_admission",
            "claim's first move escapes root restriction",
        ));
    }
    let claimed_line = if claimed.is_empty() {
        ClaimDescriptor {
            status: ClaimStatus::NoClaim,
            legality_verified: None,
            claim_truth: "unknown".into(),
            movements: Vec::new(),
            restriction_checked: false,
            final_state: None,
        }
    } else {
        check_deadline(deadline, "claim_deadline")?;
        let final_position =
            replay_checked(&target_position, &claimed, &owners, deadline).map_err(shared_error)?;
        check_deadline(deadline, "claim_deadline")?;
        ClaimDescriptor {
            status: ClaimStatus::LegalContinuation,
            legality_verified: Some(true),
            claim_truth: "unknown".into(),
            movements: meanings(&claimed, MoveTokenKind::ClaimedContinuation, None, deadline)?,
            restriction_checked: !roots.is_empty(),
            final_state: Some(describe(
                &final_position,
                &owners,
                MoveTokenKind::ClaimEndLegal,
                deadline,
            )?),
        }
    };
    let prefix = meanings(&prefix, MoveTokenKind::Prefix, None, deadline)?;
    // Actual descriptors and token streams are sealed separately from request
    // budget identity. A budget edit changes context, not chess branch meaning.
    let branch_meaning_sha256 = json_digest(&json!([BRANCH_MEANING_DOMAIN, {
        "question": request.question, "root": root, "prefix": prefix,
        "target": target, "root_restriction": root_restriction, "claimed_line": claimed_line
    }]))
    .map_err(shared_error)?;
    check_deadline(deadline, "meaning_identity_deadline")?;
    Ok(SemanticReceipt {
        schema: SEMANTIC_SCHEMA.into(),
        implementation: SEMANTIC_VERSION.into(),
        context_sha256: request.context_sha256.clone(),
        current_binary_sha256: verified_binary.into(),
        binary_pin_scope: BINARY_PIN_SCOPE.into(),
        parent_input_sha256: request.parent_input_sha256.clone(),
        before_result_anchor_sha256: request.before_result_anchor_sha256.clone(),
        caller_declaration_scope: DECLARATION_SCOPE.into(),
        meaning_scope: MEANING_SCOPE.into(),
        question: request.question,
        root,
        prefix,
        target,
        root_restriction,
        claimed_line,
        branch_meaning_sha256,
        resource_policy: SemanticResourcePolicy {
            max_wall_time_ms: request.max_wall_time_ms,
            max_output_bytes: request.max_output_bytes,
            max_prefix_plies: MAX_PREFIX_PLIES,
            max_claim_plies: MAX_CLAIM_PLIES,
            max_root_moves: MAX_ROOT_MOVES,
            cpu_checks: 0,
        },
        cpu_checks: 0,
        search_executed: false,
        model_executed: false,
        training_target_created: false,
        product_verifier_enabled: false,
        elapsed_ms: milliseconds(started.elapsed()),
        deadline_exceeded: false,
    })
}

/// Returns exactly one newline-terminated JSON receipt. The dispatcher never
/// writes stdout itself. Admission failures, original-clock expiry and output
/// overflow are typed failures, not zero-score or truncated-success receipts.
pub fn prepare_started(
    bytes: &[u8],
    verified_current_binary_sha256: &str,
    started: Instant,
) -> Result<Vec<u8>, SemanticError> {
    let request = decode_request(bytes)?;
    let deadline = absolute_deadline(&request, started)?;
    let result = (|| {
        check_deadline(deadline, "admission_deadline")?;
        let mut receipt = prepare(&request, verified_current_binary_sha256, started, deadline)?;
        let serialization = (|| {
            check_deadline(deadline, "serialization_deadline")?;
            receipt.elapsed_ms = milliseconds(started.elapsed());
            let value = serde_json::to_value(&receipt)
                .map_err(|error| SemanticError::new("serialization", error))?;
            let mut output = canonical(&value).map_err(shared_error)?;
            output.push(b'\n');
            if output.len() > request.max_output_bytes {
                return Err(SemanticError::new(
                    "output_bound",
                    "actual semantic receipt exceeds admitted output bound",
                ));
            }
            check_deadline(deadline, "serialization_deadline")?;
            Ok(output)
        })();
        serialization.map_err(|error| error.with_receipt(receipt))
    })();
    result.map_err(|mut error: SemanticError| {
        error.output_limit = request.max_output_bytes;
        error.elapsed_ms = Some(milliseconds(started.elapsed()));
        error.deadline_exceeded |= Instant::now() >= deadline;
        error
    })
}

/// Static preparation capabilities only. No checker, model, optimizer or
/// evaluator is constructed; this does not register or admit a request.
pub fn capabilities() -> Result<Vec<u8>, SemanticError> {
    let description = json!({
        "schema": SEMANTIC_SCHEMA,
        "implementation": SEMANTIC_VERSION,
        "questions": ["restricted_response", "continuation_challenge", "unrestricted_recheck"],
        "max_request_bytes": MAX_REQUEST_BYTES,
        "max_output_bytes": MAX_RESPONSE_BYTES,
        "max_wall_time_ms": MAX_WALL_TIME_MS,
        "max_prefix_plies": MAX_PREFIX_PLIES,
        "max_claim_plies": MAX_CLAIM_PLIES,
        "max_root_moves": MAX_ROOT_MOVES,
        "max_known_position_moves": ParserLimits::default().max_moves,
        "deadline_scope": "original_instant_including_stdin_and_cli_self_image_hashing",
        "deadline_enforcement": "cooperative_phase_and_each_rules_replay_move_boundaries",
        "root_restriction": "required_only_for_restricted_response;unique;legal;request_order_preserved",
        "claimed_line": "empty_no_claim_unknown_truth;otherwise_rules_legal_continuation_unknown_truth",
        "structured_rules_fields": ["board64_piece_codes", "side_to_move", "castling_rights", "en_passant_square", "halfmove_clock", "fullmove_number", "known_repetition_count", "history_completeness", "complete_legal_order"],
        "piece_code_semantics": BOARD_PIECE_CODE_SEMANTICS,
        "castling_bit_semantics": CASTLING_BIT_SEMANTICS,
        "caller_declaration_scope": DECLARATION_SCOPE,
        "binary_pin_scope": BINARY_PIN_SCOPE,
        "meaning_scope": MEANING_SCOPE,
        "cpu_checks": 0,
        "model_calls": 0,
        "training_target_created": false,
        "product_verifier_enabled": false,
        "native_v_supported": false,
        "cancellation_input_supported": false
    });
    let mut output = canonical(&description).map_err(shared_error)?;
    output.push(b'\n');
    if output.len() > 8192 {
        return Err(SemanticError::new(
            "capabilities",
            "bounded capability receipt exceeds 8KiB",
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BINARY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn bits(text: &str) -> u16 {
        pack_moves(&[BoardMove::from_uci(text).unwrap()]).unwrap()[0]
    }
    fn seal(request: &mut SemanticRequest) {
        request.context_sha256 = request_context_sha256(request).unwrap();
    }
    fn request(command: &str) -> SemanticRequest {
        let Command::Position(spec) = parse(command, ParserLimits::default()).unwrap() else {
            panic!("position fixture")
        };
        let owners = Arc::new(OwnerRegistry::default());
        let prepared = RulesUciPort::new(Arc::clone(&owners), PositionLimits::default())
            .prepare(&spec)
            .unwrap();
        let root = prepared.snapshot.rules_position();
        let mut request = SemanticRequest {
            schema: SEMANTIC_SCHEMA.into(),
            question: SemanticQuestion::UnrestrictedRecheck,
            parent_input_sha256: "b".repeat(64),
            before_result_anchor_sha256: "c".repeat(64),
            position_command: command.into(),
            expected_board_fen: root.to_fen(),
            rules_state_sha256: state_sha(root, &owners).unwrap(),
            rules_history_sha256: history_sha(root).unwrap(),
            prefix: Vec::new(),
            root_moves: Vec::new(),
            claimed_line: Vec::new(),
            current_binary_sha256: BINARY.into(),
            max_wall_time_ms: 5000,
            max_output_bytes: MAX_RESPONSE_BYTES,
            context_sha256: String::new(),
        };
        seal(&mut request);
        request
    }
    fn run(request: &SemanticRequest) -> SemanticReceipt {
        let output = prepare_started(
            &serde_json::to_vec(request).unwrap(),
            BINARY,
            Instant::now(),
        )
        .unwrap();
        assert_eq!(output.last(), Some(&b'\n'));
        assert_eq!(output.iter().filter(|&&byte| byte == b'\n').count(), 1);
        serde_json::from_slice(&output).unwrap()
    }
    fn reject(request: &SemanticRequest) -> SemanticError {
        prepare_started(
            &serde_json::to_vec(request).unwrap(),
            BINARY,
            Instant::now(),
        )
        .unwrap_err()
    }

    #[test]
    fn fresh_branch_is_rules_only_and_empty_claim_remains_unknown() {
        let receipt = run(&request("position startpos"));
        assert_eq!(receipt.cpu_checks, 0);
        assert!(
            !receipt.search_executed && !receipt.model_executed && !receipt.training_target_created
        );
        assert!(!receipt.product_verifier_enabled);
        assert_eq!(receipt.claimed_line.status, ClaimStatus::NoClaim);
        assert_eq!(receipt.claimed_line.legality_verified, None);
        assert_eq!(receipt.claimed_line.claim_truth, "unknown");
        assert!(receipt.claimed_line.final_state.is_none());
        assert_eq!(receipt.root.board_fen, receipt.target.board_fen);
        assert_eq!(
            receipt.root.rules_history_sha256,
            receipt.target.rules_history_sha256
        );
        assert_eq!(receipt.target.history_completeness, "complete");
        assert_eq!(receipt.target.legal_moves.len(), 20);
        assert_eq!(receipt.target.board64_piece_codes.len(), 64);
        assert_eq!(receipt.target.board64_piece_codes[0], 4);
        assert_eq!(receipt.target.board64_piece_codes[4], 6);
        assert_eq!(receipt.target.board64_piece_codes[60], 12);
        assert_eq!(receipt.target.castling_rights, 15);
        assert_eq!(receipt.target.en_passant_square, None);
        assert_eq!(receipt.target.halfmove_clock, 0);
        assert_eq!(receipt.target.fullmove_number, 1);
        assert_eq!(receipt.target.known_repetition_count, 1);
        for (slot, token) in receipt.target.legal_tokens.iter().enumerate() {
            assert_eq!(usize::from(token.slot), slot);
            assert_eq!(usize::from(token.legal_order_slot.unwrap()), slot);
            assert_eq!(token.move16, receipt.target.legal_moves[slot]);
            assert_eq!(token.token_kind, MoveTokenKind::TargetLegal);
            assert_eq!(bits(&token.uci), token.move16);
        }
        let value = serde_json::to_value(receipt).unwrap();
        for name in [
            "raw_score",
            "rank",
            "target_label",
            "policy",
            "value",
            "model_tensor",
        ] {
            assert!(value.get(name).is_none(), "{name}");
        }
    }

    #[test]
    fn prefix_question_masks_and_budgets_seal_distinct_ordered_conditions() {
        let mut base = request("position startpos");
        base.prefix = vec![bits("e2e4")];
        base.question = SemanticQuestion::RestrictedResponse;
        base.root_moves = vec![bits("e7e5"), bits("c7c5")];
        base.claimed_line = vec![bits("e7e5"), bits("g1f3")];
        seal(&mut base);
        let before = run(&base);
        assert_eq!(before.target.side_to_move, SideToMove::Black);
        assert_eq!(before.target.known_history_positions, 2);
        assert_eq!(before.prefix[0].token_kind, MoveTokenKind::Prefix);
        assert_eq!(before.claimed_line.status, ClaimStatus::LegalContinuation);
        assert_eq!(before.claimed_line.claim_truth, "unknown");
        assert!(before.claimed_line.restriction_checked);
        assert_eq!(
            before
                .claimed_line
                .final_state
                .as_ref()
                .unwrap()
                .known_history_positions,
            4
        );
        let mask = before.root_restriction.as_ref().unwrap();
        assert_eq!(
            mask.declared_order
                .iter()
                .map(|token| token.move16)
                .collect::<Vec<_>>(),
            base.root_moves
        );
        let effective = before
            .target
            .legal_moves
            .iter()
            .copied()
            .filter(|mv| base.root_moves.contains(mv))
            .collect::<Vec<_>>();
        assert_eq!(
            mask.effective_legal_order
                .iter()
                .map(|token| token.move16)
                .collect::<Vec<_>>(),
            effective
        );

        let mut order = base.clone();
        order.root_moves.reverse();
        seal(&mut order);
        let reordered = run(&order);
        assert_ne!(before.context_sha256, reordered.context_sha256);
        assert_ne!(
            before.branch_meaning_sha256,
            reordered.branch_meaning_sha256
        );
        assert_eq!(
            before.target.rules_state_sha256,
            reordered.target.rules_state_sha256
        );

        let mut budget = base.clone();
        budget.max_wall_time_ms += 1;
        seal(&mut budget);
        let budget = run(&budget);
        assert_ne!(before.context_sha256, budget.context_sha256);
        assert_eq!(before.branch_meaning_sha256, budget.branch_meaning_sha256);

        let mut question = base.clone();
        question.root_moves.clear();
        question.question = SemanticQuestion::ContinuationChallenge;
        seal(&mut question);
        let challenge = run(&question);
        question.question = SemanticQuestion::UnrestrictedRecheck;
        seal(&mut question);
        let recheck = run(&question);
        assert_ne!(challenge.context_sha256, recheck.context_sha256);
        assert_ne!(
            challenge.branch_meaning_sha256,
            recheck.branch_meaning_sha256
        );

        let mut changed_prefix = base;
        changed_prefix.prefix = vec![bits("d2d4")];
        seal(&mut changed_prefix);
        let changed_prefix = run(&changed_prefix);
        assert_ne!(
            before.target.rules_state_sha256,
            changed_prefix.target.rules_state_sha256
        );
        assert_ne!(
            before.target.rules_history_sha256,
            changed_prefix.target.rules_history_sha256
        );
    }

    #[test]
    fn caller_anchors_are_sealed_declarations_not_verified_parent_results() {
        let original = request("position startpos");
        let before = run(&original);
        for parent in [true, false] {
            let mut changed = original.clone();
            if parent {
                changed.parent_input_sha256 = "d".repeat(64);
            } else {
                changed.before_result_anchor_sha256 = "d".repeat(64);
            }
            assert_eq!(reject(&changed).stage, "context_identity");
            seal(&mut changed);
            let receipt = run(&changed);
            assert_ne!(receipt.context_sha256, before.context_sha256);
            assert_eq!(receipt.branch_meaning_sha256, before.branch_meaning_sha256);
            assert_eq!(receipt.caller_declaration_scope, DECLARATION_SCOPE);
        }
    }

    #[test]
    fn same_fen_different_known_history_is_not_interchangeable() {
        let trace = request("position startpos moves g1f3 g8f6 f3g1 f6g8");
        let mut fen = request(&format!("position fen {}", trace.expected_board_fen));
        let a = run(&trace);
        let b = run(&fen);
        assert_eq!(a.root.board_fen, b.root.board_fen);
        assert_ne!(a.root.rules_history_sha256, b.root.rules_history_sha256);
        assert_ne!(a.branch_meaning_sha256, b.branch_meaning_sha256);
        assert_eq!(a.root.history_completeness, "complete");
        assert_eq!(b.root.history_completeness, "unknown_prefix");
        assert_eq!(a.root.known_repetition_count, 2);
        assert_eq!(b.root.known_repetition_count, 1);
        fen.rules_history_sha256 = trace.rules_history_sha256;
        seal(&mut fen);
        assert_eq!(reject(&fen).stage, "rules_identity");
    }

    #[test]
    fn all_four_promotion_tokens_have_exact_square_and_piece_meanings() {
        let original = request("position fen 7k/P7/8/8/8/8/8/4K3 w - - 0 1");
        let mut identities = Vec::new();
        for (suffix, promotion) in [
            ('q', PromotionMeaning::Queen),
            ('r', PromotionMeaning::Rook),
            ('b', PromotionMeaning::Bishop),
            ('n', PromotionMeaning::Knight),
        ] {
            let mut request = original.clone();
            request.question = SemanticQuestion::RestrictedResponse;
            request.root_moves = vec![bits(&format!("a7a8{suffix}"))];
            request.claimed_line = request.root_moves.clone();
            seal(&mut request);
            let receipt = run(&request);
            let token = &receipt.root_restriction.unwrap().declared_order[0];
            assert_eq!(token.from, 48);
            assert_eq!(token.to, 56);
            assert_eq!(token.promotion, Some(promotion));
            assert_eq!(token.token_kind, MoveTokenKind::RootRestriction);
            assert_eq!(receipt.claimed_line.movements[0].promotion, Some(promotion));
            assert_eq!(receipt.claimed_line.claim_truth, "unknown");
            identities.push(receipt.branch_meaning_sha256);
        }
        assert!(
            identities
                .iter()
                .enumerate()
                .all(|(at, identity)| !identities[..at].contains(identity))
        );
    }

    #[test]
    fn illegal_prefix_mask_claim_and_history_pins_fail_without_search() {
        let original = request("position startpos");
        let mut prefix = original.clone();
        prefix.prefix = vec![bits("e2e5")];
        seal(&mut prefix);
        assert_eq!(reject(&prefix).stage, "rules_replay");
        let mut mask = original.clone();
        mask.question = SemanticQuestion::RestrictedResponse;
        mask.root_moves = vec![bits("e2e5")];
        seal(&mut mask);
        assert_eq!(reject(&mask).stage, "root_admission");
        mask.root_moves = vec![bits("e2e4"), bits("e2e4")];
        seal(&mut mask);
        assert_eq!(reject(&mask).stage, "root_admission");
        mask.root_moves = vec![bits("e2e4")];
        mask.claimed_line = vec![bits("d2d4")];
        seal(&mut mask);
        assert_eq!(reject(&mask).stage, "claim_admission");
        let mut claim = original.clone();
        claim.claimed_line = vec![bits("e2e4"), bits("e7e6"), bits("e4e6")];
        seal(&mut claim);
        assert_eq!(reject(&claim).stage, "rules_replay");
        for key in [
            "expected_board_fen",
            "rules_state_sha256",
            "rules_history_sha256",
        ] {
            let mut changed = original.clone();
            match key {
                "expected_board_fen" => changed.expected_board_fen = "not a FEN".into(),
                "rules_state_sha256" => changed.rules_state_sha256 = "d".repeat(64),
                _ => changed.rules_history_sha256 = "d".repeat(64),
            }
            seal(&mut changed);
            let failure = reject(&changed);
            assert_eq!(failure.stage, "rules_identity", "{key}");
            assert_eq!(failure.cpu_checks, 0);
        }
    }

    #[test]
    fn exact_bounds_unknown_duplicates_and_binary_pins_do_not_fallback() {
        let original = request("position startpos");
        for (wall, output) in [
            (0, 1024),
            (MAX_WALL_TIME_MS + 1, 1024),
            (5000, 0),
            (5000, MAX_RESPONSE_BYTES + 1),
        ] {
            let mut changed = original.clone();
            changed.max_wall_time_ms = wall;
            changed.max_output_bytes = output;
            seal(&mut changed);
            assert_eq!(reject(&changed).stage, "admission");
        }
        for selector in 0..3 {
            let mut changed = original.clone();
            match selector {
                0 => changed.prefix = vec![bits("e2e4"); MAX_PREFIX_PLIES + 1],
                1 => {
                    changed.question = SemanticQuestion::RestrictedResponse;
                    changed.root_moves = vec![bits("e2e4"); MAX_ROOT_MOVES + 1];
                }
                _ => changed.claimed_line = vec![bits("e2e4"); MAX_CLAIM_PLIES + 1],
            }
            seal(&mut changed);
            assert_eq!(reject(&changed).stage, "admission");
        }
        let raw = serde_json::to_string(&original).unwrap();
        let duplicate = raw.replacen(
            "\"max_wall_time_ms\":5000",
            "\"max_wall_time_ms\":5000,\"max_wall_time_ms\":5000",
            1,
        );
        assert_eq!(
            prepare_started(duplicate.as_bytes(), BINARY, Instant::now())
                .unwrap_err()
                .stage,
            "json_admission"
        );
        let mut value = serde_json::to_value(&original).unwrap();
        for (field, payload) in [
            ("cancel", json!(false)),
            ("horizon", json!(1)),
            ("value", json!(0)),
            ("policy", json!([])),
        ] {
            value[field] = payload;
            assert_eq!(
                prepare_started(&serde_json::to_vec(&value).unwrap(), BINARY, Instant::now())
                    .unwrap_err()
                    .stage,
                "json_admission"
            );
            value.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(
            prepare_started(raw.as_bytes(), &"d".repeat(64), Instant::now())
                .unwrap_err()
                .stage,
            "binary_identity"
        );
        assert!(super::super::request_admission(raw.as_bytes(), Instant::now()).is_err());
    }

    #[test]
    fn original_clock_and_output_bound_preserve_typed_failed_preparation() {
        let original = request("position startpos");
        let bytes = serde_json::to_vec(&original).unwrap();
        let started = Instant::now().checked_sub(Duration::from_secs(6)).unwrap();
        let admission = request_admission(&bytes, started).unwrap_err();
        assert_eq!(admission.stage, "admission_deadline");
        assert!(admission.deadline_exceeded);
        let preparation = prepare_started(&bytes, BINARY, started).unwrap_err();
        assert!(preparation.deadline_exceeded);
        assert_eq!(preparation.cpu_checks, 0);
        let mut small = original;
        small.max_output_bytes = 1024;
        seal(&mut small);
        let failure = reject(&small);
        assert_eq!(failure.stage, "output_bound");
        assert_eq!(failure.output_limit, 1024);
        let preserved = failure.receipt.unwrap();
        assert_eq!(preserved.target.legal_moves.len(), 20);
        assert_eq!(preserved.cpu_checks, 0);
        assert!(!preserved.model_executed && !preserved.search_executed);
    }

    #[test]
    fn terminal_source_is_rules_fact_and_never_proves_an_unknown_claim() {
        let request = request("position fen 7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
        let receipt = run(&request);
        assert_eq!(receipt.target.play_status, "rules_terminal");
        assert_eq!(receipt.target.terminal_reason.as_deref(), Some("Checkmate"));
        assert_eq!(
            receipt.target.terminal_source.as_deref(),
            Some("rz-position-rules")
        );
        assert_eq!(receipt.target.terminal_winner, Some(SideToMove::White));
        assert_eq!(receipt.claimed_line.status, ClaimStatus::NoClaim);
        assert_eq!(receipt.claimed_line.claim_truth, "unknown");
        let mut illegal = request;
        illegal.claimed_line = vec![bits("h8h7")];
        seal(&mut illegal);
        assert_eq!(reject(&illegal).stage, "rules_replay");
    }

    #[test]
    fn structured_rules_fields_follow_actual_prefix_castling_and_en_passant() {
        let mut ep = request("position startpos moves e2e4 a7a6 e4e5 d7d5");
        ep.prefix = vec![bits("e5d6")];
        seal(&mut ep);
        let ep = run(&ep);
        assert_eq!(ep.root.en_passant_square, Some(43));
        assert_eq!(ep.target.en_passant_square, None);
        assert_eq!(ep.target.board64_piece_codes[43], 1);
        assert_eq!(ep.target.board64_piece_codes[35], 0);
        assert_eq!(ep.target.board64_piece_codes[36], 0);
        assert_eq!(ep.target.halfmove_clock, 0);
        assert_eq!(ep.target.fullmove_number, 3);
        assert_eq!(ep.target.side_to_move, SideToMove::Black);
        assert_eq!(ep.prefix[0].from, 36);
        assert_eq!(ep.prefix[0].to, 43);

        let mut castle = request("position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        castle.prefix = vec![bits("e1g1")];
        seal(&mut castle);
        let castle = run(&castle);
        assert_eq!(castle.root.castling_rights, 15);
        assert_eq!(castle.target.castling_rights, 12);
        assert_eq!(castle.target.board64_piece_codes[6], 6);
        assert_eq!(castle.target.board64_piece_codes[5], 4);
        assert_eq!(castle.target.board64_piece_codes[4], 0);
        assert_eq!(castle.target.board64_piece_codes[7], 0);
        assert_eq!(castle.target.history_completeness, "unknown_prefix");
        assert_eq!(castle.target.halfmove_clock, 1);
        assert_eq!(castle.target.known_repetition_count, 1);
    }

    #[test]
    fn capabilities_are_static_rules_only_and_do_not_enable_v_or_cpu_search() {
        let bytes = capabilities().unwrap();
        assert!(bytes.len() <= 8192);
        let capability: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(capability["schema"], SEMANTIC_SCHEMA);
        assert_eq!(capability["cpu_checks"], 0);
        assert_eq!(capability["model_calls"], 0);
        assert_eq!(capability["native_v_supported"], false);
        assert_eq!(capability["product_verifier_enabled"], false);
        assert_eq!(capability["cancellation_input_supported"], false);
    }
}
