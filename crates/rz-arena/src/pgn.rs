//! Bounded PGN auditing against A-owned Rules and the pinned shared contract.
//! PGN notation is E's responsibility; every move and terminal decision comes
//! from `rz-position`, never from a second chess rules implementation.
use crate::{ArenaError, ArenaPlan, EngineFailureKind, GameResult, GameSpec, PairSpec};
use rz_experiments::{
    ArtifactRef, ClaimPolicy, HistoryCompleteness, InitialPosition, OpeningSpec, OutcomePolicy,
};
use rz_position::contracts::{ContractPosition, ContractState};
use rz_position::{
    Availability, BoardMove, ClaimEvidence, ClaimReason, Color, PieceKind, PlayStatus, Position,
    PositionLimits, TerminalReason,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub const PGN_AUDIT_VERSION: u32 = 1;
pub const RULES_SOURCE_COMMIT: &str = "118dc0311a88e143be285703940321dc16261f6a";
pub const CONTRACT_SOURCE_COMMIT: &str = rz_position::contracts::CONTRACT_SOURCE_REVISION;
const MAX_PGN_BYTES: usize = 4 * 1024 * 1024;
const MAX_PGN_PLIES: u32 = 4095;
const MAX_TAGS: usize = 64;
const MAX_TAG_VALUE_BYTES: usize = 8192;
const MAX_COMMENT_BYTES: usize = 16384;

// Process-local audit registry: issue fresh IDs, never derive owner/revision
// from game names or content hashes and never reset/recycle a live owner.
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug)]
pub struct PgnLimits {
    pub max_bytes: usize,
    /// Ceiling on all movetext plies per game, including the opening prefix.
    pub max_plies: u32,
}

/// Outcome declarations for a caller-owned pair, independent of its launch purpose.
/// This controls PGN classification only: it grants no launch or scoring authority.
/// Native integration callers must keep the cutoff `Incomplete` and retain raw
/// failures/incomplete games without promoting them to formal strength results.
#[derive(Clone, Copy, Debug)]
pub struct PgnOutcomePolicy {
    pub engine_failure: OutcomePolicy,
    pub max_plies_outcome: OutcomePolicy,
    /// Only AutomaticAcceptance admits pinned runner claims checked by A.
    /// ExplicitClaim remains unsupported without a separate claim receipt.
    pub claim_policy: ClaimPolicy,
    /// Ceiling on the complete game, including every declared opening ply.
    pub max_game_plies: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct PairPgnAudit {
    pub audit_version: u32,
    pub rules_source_commit: String,
    pub contract_source_commit: String,
    pub engine_contract_revision: String,
    pub opening_input_sha256: String,
    /// A's contract semantic digest, including origin, counters and history.
    /// This is not the opening declaration hash or a registry owner ID.
    pub start_state_semantic_sha256: String,
    /// Exact declared execution order, with each game matched to its plan ID.
    pub games: Vec<GamePgnAudit>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GamePgnAudit {
    pub game_id: String,
    pub white_engine: String,
    pub black_engine: String,
    /// Retained header metadata; only the V3 whole-clock gate attests it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_control: Option<String>,
    pub uci_moves: Vec<String>,
    /// Observed runner declaration. An `incomplete` cutoff can declare Draw
    /// but never grants a scored draw; callers must dispatch by classification.
    pub result: GameResult,
    pub termination: String,
    pub final_fen: String,
    pub classification: String,
    pub loser_engine: Option<String>,
    pub terminal_reason: Option<String>,
    pub engine_failure: Option<EngineFailureKind>,
    /// A pinned runner may record a late reply without applying it to its board.
    /// This evidence retains that reply separately from the committed game moves.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_evidence: Option<PgnFailureEvidence>,
}

/// PGN failure accounting only; independent clock evidence remains required.
#[derive(Clone, Debug, Serialize)]
pub struct PgnFailureEvidence {
    pub profile: String,
    pub unplayed_uci_reply: String,
    /// The runner counts its recorded, unplayed reply in this original header.
    pub declared_ply_count: u32,
    pub overrun_ms: u64,
}

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Invalid(reason.into())
}
fn rules(error: rz_position::PositionError) -> ArenaError {
    ArenaError::Contract(error.into())
}

fn fresh_owner() -> Result<rz_contracts::OwnerId, ArenaError> {
    let mut current = NEXT_OWNER.load(Ordering::Relaxed);
    loop {
        let next = current
            .checked_add(1)
            .ok_or_else(|| ArenaError::Budget("audit owner registry exhausted".into()))?;
        match NEXT_OWNER.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(owner) => return Ok(rz_contracts::OwnerId(owner)),
            Err(actual) => current = actual,
        }
    }
}

fn initial_position(opening: &OpeningSpec, max_plies: u32) -> Result<Position, ArenaError> {
    let limits = PositionLimits {
        max_fen_bytes: MAX_TAG_VALUE_BYTES,
        // Contract export checks one intended-move lookahead for claims.
        // That inspection is not an admitted movetext ply.
        max_history_positions: max_plies as usize + 2,
        ..PositionLimits::default()
    };
    let position = match opening.initial {
        InitialPosition::Startpos => Position::startpos_with_limits(limits).map_err(rules)?,
        InitialPosition::Fen => Position::from_fen_with_limits(
            opening
                .fen
                .as_deref()
                .ok_or_else(|| invalid("FEN opening lacks its origin FEN"))?,
            limits,
        )
        .map_err(rules)?,
    };
    let expected = match opening.history {
        HistoryCompleteness::Complete => rz_position::HistoryCompleteness::Complete,
        HistoryCompleteness::UnknownPrefix => rz_position::HistoryCompleteness::UnknownPrefix,
    };
    if position.history_completeness() != expected {
        return Err(invalid(
            "opening history declaration cannot be attested by A: startpos is complete; an imported FEN has an unknown prefix",
        ));
    }
    Ok(position)
}

fn contract(position: Position) -> Result<ContractPosition, ArenaError> {
    Ok(ContractPosition::new(fresh_owner()?, position))
}
fn export(position: &ContractPosition) -> Result<ContractState, ArenaError> {
    position.export().map_err(ArenaError::Contract)
}
fn make(
    position: &mut ContractPosition,
    state: &ContractState,
    mv: BoardMove,
) -> Result<(), ArenaError> {
    make_with_undo(position, state, mv).map(|_| ())
}
fn make_with_undo(
    position: &mut ContractPosition,
    state: &ContractState,
    mv: BoardMove,
) -> Result<rz_position::UndoToken, ArenaError> {
    let shared = rz_contracts::Move::try_from(mv).map_err(ArenaError::Contract)?;
    position
        .make_from_view(state, shared)
        .map_err(ArenaError::Contract)
}
fn digest(state: &ContractState) -> String {
    state
        .snapshot()
        .identity()
        .semantic
        .0
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn piece_letter(piece: PieceKind) -> char {
    match piece {
        PieceKind::Pawn => 'P',
        PieceKind::Knight => 'N',
        PieceKind::Bishop => 'B',
        PieceKind::Rook => 'R',
        PieceKind::Queen => 'Q',
        PieceKind::King => 'K',
    }
}

/// Format one already legal move. Disambiguation and check suffix inspect A's
/// board/legal array/checked preview; this code has no attack or move generator.
pub(crate) fn san(position: &Position, mv: BoardMove) -> Result<String, ArenaError> {
    let legal = position.legal_moves();
    if !legal.contains(&mv) {
        return Err(invalid("cannot serialize illegal opening move"));
    }
    let piece = position
        .piece_at(mv.from)
        .ok_or_else(|| invalid("legal move lacks source piece"))?;
    let castle = piece.kind == PieceKind::King && mv.from.file().abs_diff(mv.to.file()) == 2;
    let mut text = if castle {
        if mv.to.file() > mv.from.file() {
            "O-O".into()
        } else {
            "O-O-O".into()
        }
    } else {
        let capture = position.piece_at(mv.to).is_some()
            || (piece.kind == PieceKind::Pawn && mv.from.file() != mv.to.file());
        let mut text = String::new();
        if piece.kind == PieceKind::Pawn {
            if capture {
                text.push((b'a' + mv.from.file()) as char);
            }
        } else {
            text.push(piece_letter(piece.kind));
            let peers: Vec<_> = legal
                .iter()
                .filter(|other| {
                    other.from != mv.from
                        && other.to == mv.to
                        && position
                            .piece_at(other.from)
                            .is_some_and(|source| source.kind == piece.kind)
                })
                .collect();
            if !peers.is_empty() {
                if peers
                    .iter()
                    .all(|other| other.from.file() != mv.from.file())
                {
                    text.push((b'a' + mv.from.file()) as char);
                } else if peers
                    .iter()
                    .all(|other| other.from.rank() != mv.from.rank())
                {
                    text.push((b'1' + mv.from.rank()) as char);
                } else {
                    text.push((b'a' + mv.from.file()) as char);
                    text.push((b'1' + mv.from.rank()) as char);
                }
            }
        }
        if capture {
            text.push('x');
        }
        text.push_str(&mv.to.to_string());
        if let Some(promotion) = mv.promotion {
            text.push('=');
            text.push(piece_letter(promotion));
        }
        text
    };
    let preview = position.preview_move(mv).map_err(rules)?;
    if preview.in_check() {
        text.push(if preview.legal_moves().is_empty() {
            '#'
        } else {
            '+'
        });
    }
    Ok(text)
}

fn quote_tag(value: &str) -> Result<String, ArenaError> {
    if value.chars().any(char::is_control) {
        return Err(invalid("control character in PGN tag"));
    }
    Ok(value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn validate_pgn_contract() -> Result<(), ArenaError> {
    let revision = rz_contracts::CONTRACT_REVISION;
    revision.validate().map_err(ArenaError::Contract)?;
    if revision.major != 0 || revision.minor != 1 {
        return Err(rz_contracts::ContractError::new(
            rz_contracts::ErrorCode::UnsupportedContract,
            rz_contracts::Stage::Contract,
            "PGN Rules adapter requires the native contract revision 0.1",
        )
        .into());
    }
    Ok(())
}

fn validate_opening_spec(opening: &OpeningSpec, max_plies: u32) -> Result<(), ArenaError> {
    if !(1..=MAX_PGN_PLIES).contains(&max_plies) || opening.moves.len() > max_plies as usize {
        return Err(ArenaError::Budget(
            "opening requires a positive full-history PGN ceiling <=4095".into(),
        ));
    }
    if opening.id.is_empty()
        || opening.id.len() > MAX_TAG_VALUE_BYTES
        || opening.id.chars().any(char::is_control)
        || opening.history_origin.is_empty()
        || opening.history_origin.len() > MAX_TAG_VALUE_BYTES
        || opening.history_origin.chars().any(char::is_control)
    {
        return Err(invalid(
            "opening identifiers must be bounded nonempty single-line values",
        ));
    }
    if !matches!(
        (opening.initial, opening.fen.as_ref()),
        (InitialPosition::Startpos, None) | (InitialPosition::Fen, Some(_))
    ) {
        return Err(invalid(
            "opening must declare exactly one startpos or full FEN origin",
        ));
    }
    Ok(())
}

pub fn opening_pgn(plan: &ArenaPlan, pair_id: &str) -> Result<String, ArenaError> {
    crate::validate_engine_contract_revision(plan)?;
    let pair = plan
        .pair(pair_id)
        .ok_or_else(|| invalid("unknown pair ID"))?;
    opening_pgn_for_spec(&pair.opening, MAX_PGN_PLIES)
}

/// Serialize the complete opening through the same A-owned Rules as the plan wrapper.
/// A FEN remains an unknown-prefix origin; its counters, side and literal move
/// history are never replaced by the final board alone. This is an input artifact,
/// not validation of a caller's locked launch manifest or permission to run a game.
pub fn opening_pgn_for_spec(opening: &OpeningSpec, max_plies: u32) -> Result<String, ArenaError> {
    validate_pgn_contract()?;
    validate_opening_spec(opening, max_plies)?;
    let mut live = contract(initial_position(opening, max_plies)?)?;
    let mut text = format!(
        "[Event \"RoveZero opening input\"]\n[White \"Opening\"]\n[Black \"Opening\"]\n[Result \"*\"]\n[OpeningId \"{}\"]\n",
        quote_tag(&opening.id)?
    );
    if opening.initial == InitialPosition::Fen {
        text.push_str(&format!(
            "[SetUp \"1\"]\n[FEN \"{}\"]\n",
            quote_tag(&live.position().to_fen())?
        ));
    }
    text.push('\n');
    for (index, input) in opening.moves.iter().enumerate() {
        let state = export(&live)?;
        if live.position().side_to_move() == Color::White {
            text.push_str(&format!("{}. ", live.position().fullmove_number()));
        } else if index == 0 {
            text.push_str(&format!("{}... ", live.position().fullmove_number()));
        }
        let mv = BoardMove::from_uci(input).map_err(rules)?;
        text.push_str(&san(live.position(), mv)?);
        text.push(' ');
        make(&mut live, &state, mv)?;
    }
    // Also attest the final start state. Exact terminals are valid evidence for
    // auditing, but they cannot be submitted to an external engine as a book.
    if export(&live)?.rules().classification().play_status != PlayStatus::Ongoing {
        return Err(invalid(
            "opening ends at an exact terminal; no game launch is permitted",
        ));
    }
    text.push_str("*\n");
    if text.len() > MAX_PGN_BYTES {
        return Err(ArenaError::Budget(
            "serialized opening PGN exceeds 4 MiB".into(),
        ));
    }
    Ok(text)
}

/// Check a declared book against the Rules-owned serialization before locking
/// or preparing a launch. Matching the final board alone is insufficient.
pub fn validate_opening_artifact_for_spec(
    opening: &OpeningSpec,
    max_plies: u32,
    artifact: &ArtifactRef,
) -> Result<String, ArenaError> {
    let text = opening_pgn_for_spec(opening, max_plies)?;
    if text.len() as u64 != artifact.bytes
        || format!("{:x}", Sha256::digest(text.as_bytes())) != artifact.sha256
    {
        return Err(ArenaError::Integrity(
            "opening artifact differs from complete A-validated trace serialization".into(),
        ));
    }
    Ok(text)
}

#[derive(Debug)]
enum Token<'a> {
    Word(&'a str),
    Comment(&'a str),
    Header,
    End,
}
struct Parser<'a> {
    input: &'a str,
    offset: usize,
}
impl<'a> Parser<'a> {
    fn whitespace(&mut self) {
        while self
            .input
            .as_bytes()
            .get(self.offset)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.offset += 1;
        }
    }
    fn header_pending(&mut self) -> bool {
        self.whitespace();
        self.input.as_bytes().get(self.offset) == Some(&b'[')
    }
    fn header(&mut self) -> Result<(String, String), ArenaError> {
        self.whitespace();
        let bytes = self.input.as_bytes();
        if bytes.get(self.offset) != Some(&b'[') {
            return Err(invalid("expected PGN header"));
        }
        self.offset += 1;
        let start = self.offset;
        while bytes
            .get(self.offset)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            self.offset += 1;
        }
        if start == self.offset
            || self.offset - start > 64
            || !bytes.get(self.offset).is_some_and(u8::is_ascii_whitespace)
        {
            return Err(invalid("invalid PGN tag name/separator"));
        }
        let key = self.input[start..self.offset].to_owned();
        self.whitespace();
        if bytes.get(self.offset) != Some(&b'"') {
            return Err(invalid("PGN tag requires a quoted value"));
        }
        self.offset += 1;
        let mut value = String::new();
        loop {
            let c = self.input[self.offset..]
                .chars()
                .next()
                .ok_or_else(|| invalid("unterminated PGN tag value"))?;
            self.offset += c.len_utf8();
            if c == '"' {
                break;
            }
            if c == '\\' {
                let escaped = bytes
                    .get(self.offset)
                    .copied()
                    .ok_or_else(|| invalid("unterminated PGN escape"))?;
                if !matches!(escaped, b'"' | b'\\') {
                    return Err(invalid("unsupported PGN tag escape"));
                }
                self.offset += 1;
                value.push(escaped as char);
            } else {
                if c.is_control() {
                    return Err(invalid("control character in PGN tag"));
                }
                value.push(c);
            }
            if value.len() > MAX_TAG_VALUE_BYTES {
                return Err(ArenaError::Budget("PGN tag value exceeds ceiling".into()));
            }
        }
        self.whitespace();
        if bytes.get(self.offset) != Some(&b']') {
            return Err(invalid("expected closing PGN tag bracket"));
        }
        self.offset += 1;
        Ok((key, value))
    }
    fn token(&mut self) -> Result<Token<'a>, ArenaError> {
        self.whitespace();
        let bytes = self.input.as_bytes();
        let Some(&first) = bytes.get(self.offset) else {
            return Ok(Token::End);
        };
        if first == b'[' {
            return Ok(Token::Header);
        }
        if first == b'{' {
            self.offset += 1;
            let start = self.offset;
            while let Some(&byte) = bytes.get(self.offset) {
                if byte == b'}' {
                    let comment = &self.input[start..self.offset];
                    self.offset += 1;
                    return Ok(Token::Comment(comment));
                }
                if byte == b'{' || (byte.is_ascii_control() && !byte.is_ascii_whitespace()) {
                    return Err(invalid("nested PGN comment/control character"));
                }
                self.offset += 1;
                if self.offset - start > MAX_COMMENT_BYTES {
                    return Err(ArenaError::Budget("PGN comment exceeds ceiling".into()));
                }
            }
            return Err(invalid("unterminated PGN comment"));
        }
        if matches!(first, b'(' | b')' | b';' | b'$' | b'%' | b'}') {
            return Err(invalid(
                "unsupported PGN variation, NAG, line escape or comment syntax",
            ));
        }
        let start = self.offset;
        while bytes.get(self.offset).is_some_and(|byte| {
            !byte.is_ascii_whitespace()
                && !matches!(byte, b'[' | b'{' | b'}' | b'(' | b')' | b';' | b'$' | b'%')
        }) {
            self.offset += 1;
        }
        if self.offset - start > 64 {
            return Err(ArenaError::Budget("PGN token exceeds ceiling".into()));
        }
        let word = &self.input[start..self.offset];
        if word.is_empty() || !word.is_ascii() {
            return Err(invalid("invalid PGN movetext token"));
        }
        Ok(Token::Word(word))
    }
}

fn result_token(value: &str) -> Option<GameResult> {
    match value {
        "1-0" => Some(GameResult::WhiteWin),
        "0-1" => Some(GameResult::BlackWin),
        "1/2-1/2" => Some(GameResult::Draw),
        _ => None,
    }
}
fn require_tag<'a>(tags: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, ArenaError> {
    tags.get(key)
        .map(String::as_str)
        .ok_or_else(|| invalid(format!("PGN lacks {key} tag")))
}
fn resolve(position: &Position, word: &str) -> Result<BoardMove, ArenaError> {
    if let Ok(mv) = BoardMove::from_uci(word) {
        if position.legal_moves().contains(&mv) {
            return Ok(mv);
        }
        return Err(invalid(format!("PGN contains illegal UCI move {word}")));
    }
    let mut resolved = None;
    for mv in position.legal_moves() {
        if san(position, mv)? == word {
            if resolved.is_some() {
                return Err(invalid("ambiguous SAN in PGN"));
            }
            resolved = Some(mv);
        }
    }
    resolved.ok_or_else(|| invalid(format!("PGN SAN is not a canonical A-legal move: {word}")))
}

fn check_move_number<'a>(
    position: &Position,
    word: &'a str,
) -> Result<Option<&'a str>, ArenaError> {
    let digit_count = word.bytes().take_while(u8::is_ascii_digit).count();
    if digit_count == 0 || word.as_bytes().get(digit_count) != Some(&b'.') {
        return Ok(None);
    }
    let number: u32 = word[..digit_count]
        .parse()
        .map_err(|_| invalid("PGN move number overflow"))?;
    let dots = word[digit_count..]
        .bytes()
        .take_while(|byte| *byte == b'.')
        .count();
    let expected_dots = if position.side_to_move() == Color::White {
        1
    } else {
        3
    };
    if number != position.fullmove_number() || dots != expected_dots {
        return Err(invalid(
            "PGN move number/side disagrees with restored A state",
        ));
    }
    Ok(Some(&word[digit_count + dots..]))
}

fn terminal_reason(reason: TerminalReason) -> &'static str {
    match reason {
        TerminalReason::Checkmate => "checkmate",
        TerminalReason::Stalemate => "stalemate",
        TerminalReason::DeadPosition => "dead_position",
        TerminalReason::FivefoldRepetition => "fivefold_repetition",
        TerminalReason::SeventyFiveMove => "seventy_five_move",
    }
}

struct OutcomeClassification {
    classification: String,
    loser_engine: Option<String>,
    terminal_reason: Option<String>,
    engine_failure: Option<EngineFailureKind>,
}

fn exact_reason_suffix<'a>(comment: &'a str, marker: &str) -> Option<&'a str> {
    if let Some(suffix) = comment.strip_prefix(marker) {
        return Some(suffix);
    }
    comment
        .rsplit_once(", ")
        .and_then(|(_, reason)| reason.strip_prefix(marker))
}

fn pinned_timeout_reason(comment: &str) -> Option<(Color, u64)> {
    [(Color::White, "White"), (Color::Black, "Black")]
        .into_iter()
        .find_map(|(side, color)| {
            let marker = format!("{color} loses on time (");
            let overrun = exact_reason_suffix(comment, &marker)?.strip_suffix("ms overrun)")?;
            if overrun.is_empty() || !overrun.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            Some((side, overrun.parse().ok()?))
        })
}

fn classify_outcome(
    live: &ContractPosition,
    game: &GameSpec,
    termination: &str,
    result: GameResult,
    final_comment: &str,
    policy: PgnOutcomePolicy,
    max_plies_cutoff: bool,
) -> Result<OutcomeClassification, ArenaError> {
    let state = export(live)?;
    match state.rules().classification().play_status {
        PlayStatus::Terminal { reason, winner } => {
            let expected = match winner {
                Some(Color::White) => GameResult::WhiteWin,
                Some(Color::Black) => GameResult::BlackWin,
                None => GameResult::Draw,
            };
            if termination != "normal" || expected != result {
                return Err(invalid(
                    "PGN result/termination contradicts A exact terminal",
                ));
            }
            Ok(OutcomeClassification {
                classification: "rules_terminal".into(),
                loser_engine: None,
                terminal_reason: Some(terminal_reason(reason).into()),
                engine_failure: None,
            })
        }
        PlayStatus::Ongoing => {
            if policy.claim_policy == ClaimPolicy::AutomaticAcceptance
                && termination == "normal"
                && result == GameResult::Draw
            {
                let claim = if exact_reason_suffix(final_comment, "Draw by 3-fold repetition")
                    == Some("")
                {
                    Some((ClaimReason::ThreefoldRepetition, "threefold_repetition"))
                } else if exact_reason_suffix(final_comment, "Draw by fifty moves rule") == Some("")
                {
                    Some((ClaimReason::FiftyMove, "fifty_move"))
                } else {
                    None
                };
                if let Some((rule, reason)) = claim {
                    if !state
                        .rules()
                        .classification()
                        .claim_availability
                        .iter()
                        .any(|evidence| {
                            evidence.reason == rule
                                && evidence.evidence == ClaimEvidence::CurrentPosition
                                && evidence.availability == Availability::Available
                        })
                    {
                        return Err(invalid("runner draw claim is not currently available in A"));
                    }
                    return Ok(OutcomeClassification {
                        classification: "accepted_claim".into(),
                        loser_engine: None,
                        terminal_reason: Some(reason.into()),
                        engine_failure: None,
                    });
                }
            }
            // Pinned Fastchess reports its maxmoves ceiling as adjudicated Draw.
            // RoveZero's manifest classifies this exact limit as Incomplete.
            // Preserve the observed declaration without assigning draw points.
            if termination == "adjudication"
                && result == GameResult::Draw
                && max_plies_cutoff
                && exact_reason_suffix(final_comment, "Draw by adjudication") == Some("")
            {
                return Ok(OutcomeClassification {
                    classification: "incomplete".into(),
                    loser_engine: None,
                    terminal_reason: None,
                    engine_failure: None,
                });
            }
            if policy.engine_failure != OutcomePolicy::Loss {
                return Err(invalid("manifest does not classify engine failure as loss"));
            }
            let side = live.position().side_to_move();
            let color = if side == Color::White {
                "White"
            } else {
                "Black"
            };
            let (loser, expected) = if side == Color::White {
                (&game.white_engine, GameResult::BlackWin)
            } else {
                (&game.black_engine, GameResult::WhiteWin)
            };
            if result != expected {
                return Err(invalid(
                    "PGN engine loss winner disagrees with A side to move",
                ));
            }
            let failure = match termination {
                "illegal move" => {
                    let marker = format!("{color} makes an illegal move: ");
                    let reply = exact_reason_suffix(final_comment, &marker).ok_or_else(|| {
                        invalid("PGN illegal-move loss lacks pinned Fastchess reason/reply")
                    })?;
                    if reply.is_empty()
                        || reply.len() > 64
                        || reply.chars().any(char::is_whitespace)
                    {
                        return Err(invalid("PGN illegal reply is not a bounded single token"));
                    }
                    if BoardMove::from_uci(reply)
                        .is_ok_and(|mv| live.position().legal_moves().contains(&mv))
                    {
                        return Err(invalid("PGN labels an A-legal reply as an illegal move"));
                    }
                    EngineFailureKind::IllegalMove
                }
                "abandoned"
                    if exact_reason_suffix(final_comment, &format!("{color} disconnects"))
                        == Some("") =>
                {
                    EngineFailureKind::Crash
                }
                "time forfeit" => {
                    if pinned_timeout_reason(final_comment).map(|(loser, _)| loser) != Some(side) {
                        return Err(invalid("PGN timeout lacks pinned Fastchess reason text"));
                    }
                    EngineFailureKind::Timeout
                }
                _ => {
                    return Err(invalid(
                        "unsupported PGN outcome: claims, adjudication, stalled connection, incomplete or unspecified results are not exact Rules terminals",
                    ));
                }
            };
            Ok(OutcomeClassification {
                classification: "engine_loss".into(),
                loser_engine: Some(loser.clone()),
                terminal_reason: None,
                engine_failure: Some(failure),
            })
        }
    }
}

pub fn audit_pair_pgn(
    plan: &ArenaPlan,
    pair_id: &str,
    pgn: &str,
    limits: PgnLimits,
) -> Result<PairPgnAudit, ArenaError> {
    crate::validate_engine_contract_revision(plan)?;
    let pair = plan
        .pair(pair_id)
        .ok_or_else(|| invalid("unknown pair ID"))?;
    let protocol = &plan.manifest().input().protocol;
    audit_pair_pgn_for_spec(
        pair,
        pgn,
        limits,
        PgnOutcomePolicy {
            engine_failure: protocol.engine_failure,
            max_plies_outcome: protocol.max_plies_outcome,
            claim_policy: protocol.claim_policy,
            max_game_plies: protocol.max_plies,
        },
    )
}

fn validate_pair_spec(pair: &PairSpec) -> Result<(), ArenaError> {
    if !matches!(pair.execution_order, [0, 1] | [1, 0]) {
        return Err(invalid(
            "PGN pair execution order must contain each game exactly once",
        ));
    }
    let [first, second] = &pair.games;
    if first.id == second.id
        || first.white_engine == first.black_engine
        || first.white_engine != second.black_engine
        || first.black_engine != second.white_engine
    {
        return Err(invalid(
            "PGN pair requires distinct game IDs and exactly swapped engine colors",
        ));
    }
    for value in [
        &pair.id,
        &first.id,
        &second.id,
        &first.white_engine,
        &first.black_engine,
    ] {
        if value.is_empty()
            || value.len() > MAX_TAG_VALUE_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(invalid(
                "PGN pair identifiers must be bounded nonempty single-line values",
            ));
        }
    }
    if pair.opening_input_sha256.len() != 64
        || !pair
            .opening_input_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(
            "declared opening input identity must be a lowercase SHA-256",
        ));
    }
    Ok(())
}

/// Audit a complete pair using the same bounded parser and A-owned transitions.
/// Callers supply a previously locked pair and its classification policy. This
/// helper validates the pair's game order/colors, origin and literal history;
/// it does not attest binaries, a launch purpose, clock/resource fairness or
/// formal scoring. `opening_input_sha256` remains the caller's declaration
/// identity, separate from the A-derived semantic identity returned below.
pub fn audit_pair_pgn_for_spec(
    pair: &PairSpec,
    pgn: &str,
    limits: PgnLimits,
    policy: PgnOutcomePolicy,
) -> Result<PairPgnAudit, ArenaError> {
    validate_pgn_contract()?;
    if !(1..=MAX_PGN_BYTES).contains(&limits.max_bytes)
        || !(1..=MAX_PGN_PLIES).contains(&limits.max_plies)
        || policy.max_game_plies == 0
    {
        return Err(ArenaError::Budget(
            "positive PGN ceilings require bytes <=4 MiB and plies <=4095".into(),
        ));
    }
    if pgn.len() > limits.max_bytes {
        return Err(ArenaError::Budget(
            "PGN exceeds explicit byte ceiling".into(),
        ));
    }
    validate_pair_spec(pair)?;
    let limits = PgnLimits {
        max_plies: limits.max_plies.min(policy.max_game_plies),
        ..limits
    };
    validate_opening_spec(&pair.opening, limits.max_plies)?;
    let initial = initial_position(&pair.opening, limits.max_plies)?;
    let initial_fen = initial.to_fen();
    let mut expected_start = contract(initial)?;
    for word in &pair.opening.moves {
        let state = export(&expected_start)?;
        make(
            &mut expected_start,
            &state,
            BoardMove::from_uci(word).map_err(rules)?,
        )?;
    }
    let expected_digest = digest(&export(&expected_start)?);
    let mut parser = Parser {
        input: pgn,
        offset: 0,
    };
    let mut games = Vec::with_capacity(2);
    for index in pair.execution_order {
        let game = &pair.games[index];
        let mut tags = BTreeMap::new();
        while parser.header_pending() {
            let (key, value) = parser.header()?;
            if tags.insert(key, value).is_some() {
                return Err(invalid("duplicate PGN tag"));
            }
            if tags.len() > MAX_TAGS {
                return Err(ArenaError::Budget(
                    "PGN header count exceeds ceiling".into(),
                ));
            }
        }
        if require_tag(&tags, "White")? != game.white_engine
            || require_tag(&tags, "Black")? != game.black_engine
        {
            return Err(invalid("PGN engine colors/order disagree with pair plan"));
        }
        if tags.contains_key("Variant") {
            return Err(invalid(
                "PGN variants are unsupported: only standard chess is admitted",
            ));
        }
        match pair.opening.initial {
            InitialPosition::Fen => {
                if require_tag(&tags, "SetUp")? != "1" || require_tag(&tags, "FEN")? != initial_fen
                {
                    return Err(invalid("PGN does not preserve full origin FEN/counters"));
                }
            }
            InitialPosition::Startpos => {
                if tags.contains_key("FEN") || tags.contains_key("SetUp") {
                    return Err(invalid(
                        "PGN replaced startpos origin with a FEN declaration",
                    ));
                }
            }
        }
        let header_result = result_token(require_tag(&tags, "Result")?)
            .ok_or_else(|| invalid("PGN has incomplete/unsupported Result"))?;
        let termination = require_tag(&tags, "Termination")?.to_owned();
        let mut live = contract(initial_position(&pair.opening, limits.max_plies)?)?;
        let mut uci_moves = Vec::new();
        let mut last_move_undo = None;
        let mut final_comment = "";
        let mut result_seen = false;
        let mut pending_number = false;
        loop {
            match parser.token()? {
                Token::Comment(comment) => final_comment = comment,
                Token::Word(word) => {
                    if let Some(result) = result_token(word) {
                        if result != header_result || pending_number {
                            return Err(invalid(
                                "PGN Result header/movetext mismatch or dangling move number",
                            ));
                        }
                        result_seen = true;
                        break;
                    }
                    if word == "*" {
                        return Err(invalid(
                            "incomplete PGN lacks a complete paired outcome declaration",
                        ));
                    }
                    let actual = if let Some(rest) = check_move_number(live.position(), word)? {
                        if pending_number {
                            return Err(invalid("duplicate PGN move number"));
                        }
                        pending_number = true;
                        if rest.is_empty() {
                            continue;
                        }
                        rest
                    } else {
                        word
                    };
                    if uci_moves.len() >= limits.max_plies as usize {
                        return Err(ArenaError::Budget(
                            "PGN exceeds explicit ply ceiling".into(),
                        ));
                    }
                    let mv = resolve(live.position(), actual)?;
                    let input = mv.to_string();
                    let ply = uci_moves.len();
                    if ply < pair.opening.moves.len() && input != pair.opening.moves[ply] {
                        return Err(invalid("PGN omitted or changed the literal opening prefix"));
                    }
                    let state = export(&live)?;
                    last_move_undo = Some(make_with_undo(&mut live, &state, mv)?);
                    uci_moves.push(input);
                    final_comment = "";
                    pending_number = false;
                    if uci_moves.len() == pair.opening.moves.len()
                        && digest(&export(&live)?) != expected_digest
                    {
                        return Err(invalid("PGN Rules start-state semantic identity mismatch"));
                    }
                }
                Token::Header | Token::End => break,
            }
        }
        if !result_seen || uci_moves.len() < pair.opening.moves.len() {
            return Err(invalid("missing game result or opening prefix"));
        }
        let declared_ply_count = tags
            .get("PlyCount")
            .map(|count| {
                count
                    .parse::<u32>()
                    .map_err(|_| invalid("invalid PGN PlyCount"))
            })
            .transpose()?;
        if let Some(count) = declared_ply_count {
            // Fastchess includes an unplayed illegal reply in PlyCount.
            let expected = uci_moves.len() + usize::from(termination == "illegal move");
            if count as usize != expected {
                return Err(invalid(
                    "PGN PlyCount disagrees with parsed legal moves/failure profile",
                ));
            }
        }
        let mut failure_evidence = None;
        if termination == "time forfeit"
            && let Some((loser, overrun_ms)) = pinned_timeout_reason(final_comment)
            && loser != live.position().side_to_move()
        {
            // Fastchess f618e345 records best_move in addMoveData before checking
            // the charged clock. On timeout it returns before board.makeMove.
            // Its final reply is A-legal but was never a committed game move.
            // Require the exact reason and recorded-ply header for this profile;
            // a no-reply timeout retains the current A state instead.
            let declared_ply_count = declared_ply_count.ok_or_else(|| {
                invalid("recorded timeout reply requires pinned Fastchess PlyCount")
            })?;
            if uci_moves.len() <= pair.opening.moves.len() {
                return Err(invalid("timeout cannot unplay the declared opening prefix"));
            }
            let undo = last_move_undo
                .take()
                .ok_or_else(|| invalid("recorded timeout reply lacks an A undo proof"))?;
            live.unmake(undo).map_err(ArenaError::Contract)?;
            if live.position().side_to_move() != loser {
                return Err(invalid(
                    "recorded timeout reply does not belong to the A loser",
                ));
            }
            failure_evidence = Some(PgnFailureEvidence {
                profile: "fastchess_recorded_unplayed_timeout_reply_v1".into(),
                unplayed_uci_reply: uci_moves
                    .pop()
                    .ok_or_else(|| invalid("recorded timeout reply is missing"))?,
                declared_ply_count,
                overrun_ms,
            });
        }
        let outcome = classify_outcome(
            &live,
            game,
            &termination,
            header_result,
            final_comment,
            policy,
            policy.max_plies_outcome == OutcomePolicy::Incomplete
                && uci_moves.len() == policy.max_game_plies as usize,
        )?;
        games.push(GamePgnAudit {
            game_id: game.id.clone(),
            white_engine: game.white_engine.clone(),
            black_engine: game.black_engine.clone(),
            time_control: tags.get("TimeControl").cloned(),
            uci_moves,
            result: header_result,
            termination,
            final_fen: live.position().to_fen(),
            classification: outcome.classification,
            loser_engine: outcome.loser_engine,
            terminal_reason: outcome.terminal_reason,
            engine_failure: outcome.engine_failure,
            failure_evidence,
        });
    }
    if !matches!(parser.token()?, Token::End) {
        return Err(invalid(
            "PGN contains extra game or trailing unsupported content",
        ));
    }
    let revision = rz_contracts::CONTRACT_REVISION;
    Ok(PairPgnAudit {
        audit_version: PGN_AUDIT_VERSION,
        rules_source_commit: RULES_SOURCE_COMMIT.into(),
        contract_source_commit: CONTRACT_SOURCE_COMMIT.into(),
        engine_contract_revision: format!("{}.{}", revision.major, revision.minor),
        opening_input_sha256: pair.opening_input_sha256.clone(),
        start_state_semantic_sha256: expected_digest,
        games,
    })
}
