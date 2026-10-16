//! Actual collection clocks and PGN, separate from neural training targets.
//! The caller measures position submission through the final engine decision.
//! This ledger charges that whole interval; it never substitutes a per-move
//! budget or an engine-reported search time for the observed wall duration.
use crate::ArenaError;
use rz_position::{
    Availability, BoardMove, ClaimEvidence, ClaimReason, Color, HistoryOrigin, PlayStatus,
    Position, TerminalReason,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::time::Duration;

const BASE: Duration = Duration::from_secs(120);
const INCREMENT: Duration = Duration::from_secs(1);
const MAX_OPENING_PLIES: usize = 256;
const MAX_PLAYED_PLIES: usize = 512;
const MAX_PGN_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionPgnResultV3 {
    WhiteWin,
    BlackWin,
    Draw,
    Incomplete,
}
impl CollectionPgnResultV3 {
    fn tag(self) -> &'static str {
        match self {
            Self::WhiteWin => "1-0",
            Self::BlackWin => "0-1",
            Self::Draw => "1/2-1/2",
            Self::Incomplete => "*",
        }
    }
    fn winner(color: Color) -> Self {
        match color {
            Color::White => Self::WhiteWin,
            Color::Black => Self::BlackWin,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionPgnTerminationV3 {
    RulesAutomatic,
    RulesClaim(ClaimReason),
    TimeForfeit { loser: Color },
    IllegalMove { loser: Color },
    EngineCrash { loser: Color },
    Infrastructure,
    PlyLimit,
    Cancelled,
}

// Keep wire names local to collection receipts. Rules owns Color and
// ClaimReason, and neither Rules serde support nor a decoded receipt establishes
// that the indicated termination actually happened.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum CollectionPgnTerminationWireV3 {
    // Struct variants also reject additional fields for payload-free endings.
    RulesAutomatic {},
    RulesClaim { reason: CollectionClaimReasonWireV3 },
    TimeForfeit { loser: CollectionColorWireV3 },
    IllegalMove { loser: CollectionColorWireV3 },
    EngineCrash { loser: CollectionColorWireV3 },
    Infrastructure {},
    PlyLimit {},
    Cancelled {},
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum CollectionClaimReasonWireV3 {
    ThreefoldRepetition,
    FiftyMove,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum CollectionColorWireV3 {
    White,
    Black,
}

// Derived externally tagged unit enums also accept object forms in JSON. The
// receipt payload deliberately permits only these exact string spellings.
impl<'de> Deserialize<'de> for CollectionClaimReasonWireV3 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "threefold_repetition" => Ok(Self::ThreefoldRepetition),
            "fifty_move" => Ok(Self::FiftyMove),
            _ => Err(serde::de::Error::unknown_variant(
                &value,
                &["threefold_repetition", "fifty_move"],
            )),
        }
    }
}

impl<'de> Deserialize<'de> for CollectionColorWireV3 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "white" => Ok(Self::White),
            "black" => Ok(Self::Black),
            _ => Err(serde::de::Error::unknown_variant(
                &value,
                &["white", "black"],
            )),
        }
    }
}

impl From<Color> for CollectionColorWireV3 {
    fn from(color: Color) -> Self {
        match color {
            Color::White => Self::White,
            Color::Black => Self::Black,
        }
    }
}

impl From<CollectionColorWireV3> for Color {
    fn from(color: CollectionColorWireV3) -> Self {
        match color {
            CollectionColorWireV3::White => Self::White,
            CollectionColorWireV3::Black => Self::Black,
        }
    }
}

impl From<CollectionPgnTerminationV3> for CollectionPgnTerminationWireV3 {
    fn from(termination: CollectionPgnTerminationV3) -> Self {
        match termination {
            CollectionPgnTerminationV3::RulesAutomatic => Self::RulesAutomatic {},
            CollectionPgnTerminationV3::RulesClaim(reason) => Self::RulesClaim {
                reason: match reason {
                    ClaimReason::ThreefoldRepetition => {
                        CollectionClaimReasonWireV3::ThreefoldRepetition
                    }
                    ClaimReason::FiftyMove => CollectionClaimReasonWireV3::FiftyMove,
                },
            },
            CollectionPgnTerminationV3::TimeForfeit { loser } => Self::TimeForfeit {
                loser: loser.into(),
            },
            CollectionPgnTerminationV3::IllegalMove { loser } => Self::IllegalMove {
                loser: loser.into(),
            },
            CollectionPgnTerminationV3::EngineCrash { loser } => Self::EngineCrash {
                loser: loser.into(),
            },
            CollectionPgnTerminationV3::Infrastructure => Self::Infrastructure {},
            CollectionPgnTerminationV3::PlyLimit => Self::PlyLimit {},
            CollectionPgnTerminationV3::Cancelled => Self::Cancelled {},
        }
    }
}

impl From<CollectionPgnTerminationWireV3> for CollectionPgnTerminationV3 {
    fn from(termination: CollectionPgnTerminationWireV3) -> Self {
        match termination {
            CollectionPgnTerminationWireV3::RulesAutomatic {} => Self::RulesAutomatic,
            CollectionPgnTerminationWireV3::RulesClaim { reason } => {
                Self::RulesClaim(match reason {
                    CollectionClaimReasonWireV3::ThreefoldRepetition => {
                        ClaimReason::ThreefoldRepetition
                    }
                    CollectionClaimReasonWireV3::FiftyMove => ClaimReason::FiftyMove,
                })
            }
            CollectionPgnTerminationWireV3::TimeForfeit { loser } => Self::TimeForfeit {
                loser: loser.into(),
            },
            CollectionPgnTerminationWireV3::IllegalMove { loser } => Self::IllegalMove {
                loser: loser.into(),
            },
            CollectionPgnTerminationWireV3::EngineCrash { loser } => Self::EngineCrash {
                loser: loser.into(),
            },
            CollectionPgnTerminationWireV3::Infrastructure {} => Self::Infrastructure,
            CollectionPgnTerminationWireV3::PlyLimit {} => Self::PlyLimit,
            CollectionPgnTerminationWireV3::Cancelled {} => Self::Cancelled,
        }
    }
}

impl Serialize for CollectionPgnTerminationV3 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CollectionPgnTerminationWireV3::from(*self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CollectionPgnTerminationV3 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        CollectionPgnTerminationWireV3::deserialize(deserializer).map(Self::from)
    }
}

/// A failed response has no accepted move and earns no Fischer increment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionDecisionFailureV3 {
    Timeout,
    EngineCrash,
    Infrastructure,
    Cancelled,
}

#[derive(Clone, Debug, Serialize)]
pub struct CollectionClockAuditV3 {
    pub schema: &'static str,
    pub base_ms: u64,
    pub increment_ms: u64,
    pub opening_plies: usize,
    pub accepted_plies: usize,
    pub white_remaining_ns: u64,
    pub black_remaining_ns: u64,
    pub white_wall_charged_ns: u64,
    pub black_wall_charged_ns: u64,
    pub white_increments: u16,
    pub black_increments: u16,
}

#[derive(Clone, Default)]
struct SideClock {
    remaining: Duration,
    used: Duration,
    increments: u16,
}

/// An owned live Rules trace prevents a repeated FEN with a different history
/// from being accepted as the next actual position. PGN serialization reuses
/// the arena's canonical SAN adapter and the Rules owner's legal transitions.
pub struct CollectionGamePgnV3 {
    game_id: String,
    white: String,
    black: String,
    position: Position,
    initial_fen: String,
    replay_start_fen: String,
    replay_origin: HistoryOrigin,
    opening_moves: String,
    opening_plies: usize,
    accepted_plies: usize,
    clocks: [SideClock; 2],
    movetext: String,
    observed_failure: Option<CollectionPgnTerminationV3>,
}

impl CollectionGamePgnV3 {
    pub fn begin(
        game_id: impl Into<String>,
        white_name: impl Into<String>,
        black_name: impl Into<String>,
        position: &Position,
    ) -> Result<Self, ArenaError> {
        let game_id = bounded_tag(game_id.into())?;
        let white = bounded_tag(white_name.into())?;
        let black = bounded_tag(black_name.into())?;
        let replay = position
            .snapshot()
            .uci_replay(MAX_OPENING_PLIES)
            .map_err(rules)?;
        let mut start = match replay.origin {
            HistoryOrigin::StartPosition => Position::startpos(),
            HistoryOrigin::Fen => Position::from_fen(&replay.start_fen).map_err(rules)?,
        };
        let mut movetext = String::new();
        for &movement in &replay.moves {
            append_san(&mut movetext, &start, movement, None)?;
            start.make_move(movement).map_err(rules)?;
        }
        if start.position_identity() != position.position_identity() {
            return Err(invalid(
                "PGN opening replay differs from the exact Rules history",
            ));
        }
        let opening_moves = replay
            .moves
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        Ok(Self {
            game_id,
            white,
            black,
            position: position.clone(),
            initial_fen: position.to_fen(),
            replay_start_fen: replay.start_fen,
            replay_origin: replay.origin,
            opening_moves,
            opening_plies: replay.moves.len(),
            accepted_plies: 0,
            clocks: [
                SideClock {
                    remaining: BASE,
                    ..SideClock::default()
                },
                SideClock {
                    remaining: BASE,
                    ..SideClock::default()
                },
            ],
            movetext,
            observed_failure: None,
        })
    }

    /// UCI milliseconds round down, while this owner charges exact durations.
    pub fn remaining_ms(&self, color: Color) -> u64 {
        self.remaining(color).as_millis() as u64
    }
    pub fn remaining(&self, color: Color) -> Duration {
        self.clocks[index(color)].remaining
    }
    pub fn used(&self, color: Color) -> Duration {
        self.clocks[index(color)].used
    }
    pub fn position(&self) -> &Position {
        &self.position
    }
    /// A failed append can be distinguished from a validation error without
    /// parsing human-readable diagnostics or inventing a forfeiting side.
    pub fn observed_failure(&self) -> Option<CollectionPgnTerminationV3> {
        self.observed_failure
    }
    pub fn clock_audit(&self) -> Result<CollectionClockAuditV3, ArenaError> {
        Ok(CollectionClockAuditV3 {
            schema: "rz-pals-collection-clock/3",
            base_ms: BASE.as_millis() as u64,
            increment_ms: INCREMENT.as_millis() as u64,
            opening_plies: self.opening_plies,
            accepted_plies: self.accepted_plies,
            white_remaining_ns: nanos(self.clocks[0].remaining)?,
            black_remaining_ns: nanos(self.clocks[1].remaining)?,
            white_wall_charged_ns: nanos(self.clocks[0].used)?,
            black_wall_charged_ns: nanos(self.clocks[1].used)?,
            white_increments: self.clocks[0].increments,
            black_increments: self.clocks[1].increments,
        })
    }

    /// Only a legal, on-time accepted actual move earns its increment. A late
    /// or illegal reply is preserved as a failure and cannot advance the trace.
    pub fn push_actual_move(
        &mut self,
        before: &Position,
        movement: BoardMove,
        elapsed: Duration,
    ) -> Result<(), ArenaError> {
        self.ensure_current(before)?;
        if self.accepted_plies >= MAX_PLAYED_PLIES {
            return Err(ArenaError::Budget(
                "actual collection reached 512 played plies".into(),
            ));
        }
        let legal = self.position.ordered_legal_moves();
        if self.position.play_status_from_view(&legal).map_err(rules)? != PlayStatus::Ongoing {
            return Err(invalid("cannot append a move after a Rules ending"));
        }
        let side = self.position.side_to_move();
        let charged = self.charged_clock(side, elapsed)?;
        if elapsed >= self.remaining(side) {
            self.clocks[index(side)] = charged;
            self.observed_failure = Some(CollectionPgnTerminationV3::TimeForfeit { loser: side });
            return Err(ArenaError::Budget(
                "actual position-to-decision interval exhausted game clock".into(),
            ));
        }
        if !legal.moves().contains(&movement) {
            self.clocks[index(side)] = charged;
            self.observed_failure = Some(CollectionPgnTerminationV3::IllegalMove { loser: side });
            return Err(invalid("actual engine response is not Rules-legal"));
        }
        let mut credited = charged;
        credited.remaining = credited
            .remaining
            .checked_add(INCREMENT)
            .ok_or_else(|| invalid("collection clock overflow"))?;
        credited.increments = credited
            .increments
            .checked_add(1)
            .ok_or_else(|| invalid("collection increment count overflow"))?;
        let next = self.position.preview_move(movement).map_err(rules)?;
        let mut move_text = String::new();
        append_san(
            &mut move_text,
            &self.position,
            movement,
            Some((credited.remaining, elapsed)),
        )?;
        ensure_output_bound(self.movetext.len().saturating_add(move_text.len()))?;
        self.movetext.push_str(&move_text);
        self.clocks[index(side)] = credited;
        self.position = next;
        self.accepted_plies += 1;
        Ok(())
    }

    /// Use when there is no accepted bestmove. The caller preserves the actual
    /// process/transport failure separately; this receipt is not WDL evidence.
    pub fn record_failed_decision(
        &mut self,
        before: &Position,
        elapsed: Duration,
        failure: CollectionDecisionFailureV3,
    ) -> Result<(), ArenaError> {
        self.ensure_current(before)?;
        let legal = self.position.ordered_legal_moves();
        if self.position.play_status_from_view(&legal).map_err(rules)? != PlayStatus::Ongoing {
            return Err(invalid(
                "cannot record an engine decision after a Rules ending",
            ));
        }
        let side = self.position.side_to_move();
        if failure == CollectionDecisionFailureV3::Timeout && elapsed < self.remaining(side) {
            return Err(invalid(
                "timeout cannot be inferred before the actual game clock expires",
            ));
        }
        let charged = self.charged_clock(side, elapsed)?;
        self.clocks[index(side)] = charged;
        self.observed_failure = Some(match failure {
            CollectionDecisionFailureV3::Timeout => {
                CollectionPgnTerminationV3::TimeForfeit { loser: side }
            }
            CollectionDecisionFailureV3::EngineCrash => {
                CollectionPgnTerminationV3::EngineCrash { loser: side }
            }
            CollectionDecisionFailureV3::Infrastructure => {
                CollectionPgnTerminationV3::Infrastructure
            }
            CollectionDecisionFailureV3::Cancelled => CollectionPgnTerminationV3::Cancelled,
        });
        Ok(())
    }

    /// Preserve raw accepted moves after a runner window, serialization or
    /// physical-closure failure. This adds neither a move nor a clock charge
    /// and grants no Rules WDL authority, even for a terminal final board.
    /// A previously observed engine loss keeps its actual result/termination.
    pub fn abort_infrastructure(self) -> Result<String, ArenaError> {
        let termination = self
            .observed_failure
            .unwrap_or(CollectionPgnTerminationV3::Infrastructure);
        let result = match termination {
            CollectionPgnTerminationV3::TimeForfeit { loser }
            | CollectionPgnTerminationV3::IllegalMove { loser }
            | CollectionPgnTerminationV3::EngineCrash { loser } => {
                CollectionPgnResultV3::winner(loser.opposite())
            }
            CollectionPgnTerminationV3::Infrastructure | CollectionPgnTerminationV3::Cancelled => {
                CollectionPgnResultV3::Incomplete
            }
            _ => return Err(invalid("unexpected recorded collection failure")),
        };
        self.finish(result, termination)
    }

    pub fn finish(
        self,
        result: CollectionPgnResultV3,
        termination: CollectionPgnTerminationV3,
    ) -> Result<String, ArenaError> {
        let ending = self.validate_ending(result, termination)?;
        let audit = self.clock_audit()?;
        let wall = self
            .used(Color::White)
            .checked_add(self.used(Color::Black))
            .ok_or_else(|| invalid("collection wall duration overflow"))?;
        let mut output = format!(
            "[Event \"RoveZero actual adaptive selfplay\"]\n[Site \"?\"]\n[Date \"????.??.??\"]\n[Round \"{}\"]\n[White \"{}\"]\n[Black \"{}\"]\n[Result \"{}\"]\n[Termination \"{}\"]\n[TimeControl \"120+1\"]\n[RoveZeroClockBasis \"position-to-final-decision wall\"]\n[RoveZeroCollectionInitialFEN \"{}\"]\n[RoveZeroOpeningMoves \"{}\"]\n[RoveZeroOpeningPlies \"{}\"]\n[RoveZeroPlayedPlies \"{}\"]\n[RoveZeroDecisionWallNs \"{}\"]\n[RoveZeroWhiteRemainingNs \"{}\"]\n[RoveZeroBlackRemainingNs \"{}\"]\n[RoveZeroWhiteWallNs \"{}\"]\n[RoveZeroBlackWallNs \"{}\"]\n[RoveZeroWhiteIncrements \"{}\"]\n[RoveZeroBlackIncrements \"{}\"]\n",
            self.game_id,
            self.white,
            self.black,
            result.tag(),
            ending,
            self.initial_fen,
            self.opening_moves,
            self.opening_plies,
            self.accepted_plies,
            nanos(wall)?,
            audit.white_remaining_ns,
            audit.black_remaining_ns,
            audit.white_wall_charged_ns,
            audit.black_wall_charged_ns,
            audit.white_increments,
            audit.black_increments,
        );
        if self.replay_origin == HistoryOrigin::Fen {
            output.push_str(&format!(
                "[SetUp \"1\"]\n[FEN \"{}\"]\n",
                self.replay_start_fen
            ));
        }
        output.push_str(&format!("\n{}{}\n\n", self.movetext, result.tag()));
        ensure_output_bound(output.len())?;
        Ok(output)
    }

    fn ensure_current(&self, before: &Position) -> Result<(), ArenaError> {
        if self.observed_failure.is_some() {
            return Err(invalid(
                "failed collection game cannot accept another decision",
            ));
        }
        if self.position.position_identity() != before.position_identity() {
            return Err(invalid(
                "actual move before-state differs in Rules state or history",
            ));
        }
        Ok(())
    }

    fn charged_clock(&self, color: Color, elapsed: Duration) -> Result<SideClock, ArenaError> {
        let clock = &self.clocks[index(color)];
        let used = clock
            .used
            .checked_add(elapsed)
            .ok_or_else(|| invalid("collection charged duration overflow"))?;
        nanos(used)?;
        Ok(SideClock {
            remaining: clock.remaining.saturating_sub(elapsed),
            used,
            increments: clock.increments,
        })
    }

    fn validate_ending(
        &self,
        result: CollectionPgnResultV3,
        termination: CollectionPgnTerminationV3,
    ) -> Result<&'static str, ArenaError> {
        let expected_failure = self.observed_failure;
        match termination {
            CollectionPgnTerminationV3::RulesAutomatic if expected_failure.is_none() => {
                let legal = self.position.ordered_legal_moves();
                let PlayStatus::Terminal { reason, winner } =
                    self.position.play_status_from_view(&legal).map_err(rules)?
                else {
                    return Err(invalid(
                        "Rules automatic ending requires a terminal actual position",
                    ));
                };
                require_result(
                    result,
                    winner
                        .map(CollectionPgnResultV3::winner)
                        .unwrap_or(CollectionPgnResultV3::Draw),
                )?;
                Ok(match reason {
                    TerminalReason::Checkmate => "checkmate",
                    TerminalReason::Stalemate => "stalemate",
                    TerminalReason::DeadPosition => "dead position",
                    TerminalReason::FivefoldRepetition => "fivefold repetition",
                    TerminalReason::SeventyFiveMove => "seventy-five-move rule",
                })
            }
            CollectionPgnTerminationV3::RulesClaim(reason) if expected_failure.is_none() => {
                let status = self.position.classify_position().map_err(rules)?;
                if status.play_status != PlayStatus::Ongoing
                    || !status.claim_availability.iter().any(|claim| {
                        claim.reason == reason
                            && claim.evidence == ClaimEvidence::CurrentPosition
                            && claim.availability == Availability::Available
                    })
                {
                    return Err(invalid("draw claim lacks current actual Rules evidence"));
                }
                require_result(result, CollectionPgnResultV3::Draw)?;
                Ok(match reason {
                    ClaimReason::ThreefoldRepetition => "threefold repetition claim",
                    ClaimReason::FiftyMove => "fifty-move rule claim",
                })
            }
            CollectionPgnTerminationV3::TimeForfeit { loser }
            | CollectionPgnTerminationV3::IllegalMove { loser }
            | CollectionPgnTerminationV3::EngineCrash { loser }
                if expected_failure == Some(termination) =>
            {
                require_result(result, CollectionPgnResultV3::winner(loser.opposite()))?;
                Ok(match termination {
                    CollectionPgnTerminationV3::TimeForfeit { .. } => "time forfeit",
                    CollectionPgnTerminationV3::IllegalMove { .. } => "illegal move",
                    _ => "engine crash",
                })
            }
            CollectionPgnTerminationV3::PlyLimit
                if expected_failure.is_none() && self.accepted_plies == MAX_PLAYED_PLIES =>
            {
                let legal = self.position.ordered_legal_moves();
                if self.position.play_status_from_view(&legal).map_err(rules)?
                    != PlayStatus::Ongoing
                {
                    return Err(invalid(
                        "terminal Rules state must not be recorded as a ply-limit ending",
                    ));
                }
                require_result(result, CollectionPgnResultV3::Incomplete)?;
                Ok("collection ply limit")
            }
            CollectionPgnTerminationV3::Infrastructure | CollectionPgnTerminationV3::Cancelled
                if expected_failure.is_none() || expected_failure == Some(termination) =>
            {
                require_result(result, CollectionPgnResultV3::Incomplete)?;
                Ok(
                    if termination == CollectionPgnTerminationV3::Infrastructure {
                        "infrastructure failure"
                    } else {
                        "cancelled"
                    },
                )
            }
            _ => Err(invalid(
                "PGN result/termination differs from the observed actual ending",
            )),
        }
    }
}

fn index(color: Color) -> usize {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}
fn rules(error: rz_position::PositionError) -> ArenaError {
    invalid(error.to_string())
}
fn invalid(message: impl Into<String>) -> ArenaError {
    ArenaError::Invalid(message.into())
}
fn nanos(duration: Duration) -> Result<u64, ArenaError> {
    u64::try_from(duration.as_nanos())
        .map_err(|_| invalid("collection nanosecond evidence overflow"))
}
fn require_result(
    actual: CollectionPgnResultV3,
    expected: CollectionPgnResultV3,
) -> Result<(), ArenaError> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid("PGN result contradicts actual termination"))
    }
}
fn bounded_tag(value: String) -> Result<String, ArenaError> {
    if value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control) {
        return Err(invalid("collection PGN tag size/control character"));
    }
    Ok(value.replace('\\', "\\\\").replace('"', "\\\""))
}
fn ensure_output_bound(bytes: usize) -> Result<(), ArenaError> {
    if bytes <= MAX_PGN_BYTES {
        Ok(())
    } else {
        Err(ArenaError::Budget("collection PGN exceeds 1MiB".into()))
    }
}
fn clock_text(duration: Duration) -> String {
    let millis = duration.as_millis();
    format!(
        "{}:{:02}:{:02}.{:03}",
        millis / 3_600_000,
        (millis / 60_000) % 60,
        (millis / 1000) % 60,
        millis % 1000
    )
}
fn append_san(
    text: &mut String,
    before: &Position,
    movement: BoardMove,
    clock: Option<(Duration, Duration)>,
) -> Result<(), ArenaError> {
    if before.side_to_move() == Color::White {
        text.push_str(&format!("{}. ", before.fullmove_number()));
    } else if text.is_empty() {
        text.push_str(&format!("{}... ", before.fullmove_number()));
    }
    text.push_str(&crate::pgn::san(before, movement)?);
    if let Some((remaining, elapsed)) = clock {
        text.push_str(&format!(
            " {{[%clk {}] [%emt {}] [%rz_wall_ns {}]}}",
            clock_text(remaining),
            clock_text(elapsed),
            nanos(elapsed)?
        ));
    }
    text.push(' ');
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn movement(uci: &str) -> BoardMove {
        BoardMove::from_uci(uci).unwrap()
    }

    #[test]
    fn termination_wire_round_trips_rules_claims_and_each_forfeiting_color() {
        let mut cases = vec![
            CollectionPgnTerminationV3::RulesAutomatic,
            CollectionPgnTerminationV3::RulesClaim(ClaimReason::ThreefoldRepetition),
            CollectionPgnTerminationV3::RulesClaim(ClaimReason::FiftyMove),
            CollectionPgnTerminationV3::Infrastructure,
            CollectionPgnTerminationV3::PlyLimit,
            CollectionPgnTerminationV3::Cancelled,
        ];
        for loser in [Color::White, Color::Black] {
            cases.extend([
                CollectionPgnTerminationV3::TimeForfeit { loser },
                CollectionPgnTerminationV3::IllegalMove { loser },
                CollectionPgnTerminationV3::EngineCrash { loser },
            ]);
        }
        for termination in cases {
            let encoded = serde_json::to_string(&termination).unwrap();
            let decoded: CollectionPgnTerminationV3 = serde_json::from_str(&encoded).unwrap();
            assert_eq!(decoded, termination);
            assert_eq!(serde_json::to_string(&decoded).unwrap(), encoded);
        }
        assert_eq!(
            serde_json::to_string(&CollectionPgnTerminationV3::RulesClaim(
                ClaimReason::FiftyMove
            ))
            .unwrap(),
            r#"{"kind":"rules_claim","reason":"fifty_move"}"#,
        );
        assert_eq!(
            serde_json::to_string(&CollectionPgnTerminationV3::TimeForfeit {
                loser: Color::Black,
            })
            .unwrap(),
            r#"{"kind":"time_forfeit","loser":"black"}"#,
        );
    }

    #[test]
    fn termination_wire_rejects_unknown_fields_variants_and_payloads() {
        for input in [
            r#"{"kind":"unknown"}"#,
            r#"{"kind":"rules_automatic","extra":true}"#,
            r#"{"kind":"rules_claim","reason":"fifty_move","extra":0}"#,
            r#"{"kind":"time_forfeit","loser":"white","extra":null}"#,
            r#"{"kind":"illegal_move","loser":"black","extra":[]}"#,
            r#"{"kind":"engine_crash","loser":"white","extra":{}}"#,
            r#"{"kind":"infrastructure","extra":1}"#,
            r#"{"kind":"ply_limit","extra":1}"#,
            r#"{"kind":"cancelled","extra":1}"#,
            r#"{"kind":"rules_claim","reason":"unknown"}"#,
            r#"{"kind":"rules_claim","reason":"FiftyMove"}"#,
            r#"{"kind":"rules_claim","reason":{"fifty_move":null}}"#,
            r#"{"kind":"time_forfeit","loser":"unknown"}"#,
            r#"{"kind":"time_forfeit","loser":"White"}"#,
            r#"{"kind":"time_forfeit","loser":{"white":null}}"#,
            r#"{"kind":"time_forfeit"}"#,
            r#"{"kind":"time_forfeit","loser":null}"#,
            r#"{"kind":"rules_claim"}"#,
            r#"{"kind":"cancelled","kind":"infrastructure"}"#,
            r#""cancelled""#,
        ] {
            assert!(
                serde_json::from_str::<CollectionPgnTerminationV3>(input).is_err(),
                "unexpectedly accepted {input}",
            );
        }
    }

    #[test]
    fn whole_game_clock_charges_fractional_wall_before_increment() {
        let initial = Position::startpos();
        let mut pgn = CollectionGamePgnV3::begin("g1", "PALS-A", "PALS-B", &initial).unwrap();
        pgn.push_actual_move(
            &initial,
            movement("e2e4"),
            Duration::from_nanos(2_500_000_001),
        )
        .unwrap();
        assert_eq!(
            pgn.remaining(Color::White),
            Duration::from_nanos(118_499_999_999)
        );
        assert_eq!(pgn.remaining_ms(Color::White), 118_499);
        assert_eq!(pgn.remaining(Color::Black), BASE);
        assert_eq!(pgn.clock_audit().unwrap().white_increments, 1);
        let output = pgn
            .finish(
                CollectionPgnResultV3::Incomplete,
                CollectionPgnTerminationV3::Cancelled,
            )
            .unwrap();
        assert!(output.contains("[TimeControl \"120+1\"]"));
        assert!(output.contains("1. e4 {[%clk 0:01:58.499]"));
        assert!(output.contains("[%rz_wall_ns 2500000001]"));
        assert!(output.contains("[White \"PALS-A\"]\n[Black \"PALS-B\"]"));
        assert!(output.ends_with("*\n\n"));
    }

    #[test]
    fn late_move_does_not_earn_increment_or_enter_pgn() {
        let initial = Position::startpos();
        let mut pgn = CollectionGamePgnV3::begin("g1", "A", "B", &initial).unwrap();
        assert!(
            pgn.push_actual_move(&initial, movement("e2e4"), BASE)
                .is_err()
        );
        assert_eq!(pgn.remaining(Color::White), Duration::ZERO);
        assert_eq!(pgn.clock_audit().unwrap().white_increments, 0);
        assert_eq!(
            pgn.position().position_identity(),
            initial.position_identity()
        );
        assert!(
            pgn.push_actual_move(&initial, movement("d2d4"), Duration::ZERO)
                .is_err()
        );
        let output = pgn
            .finish(
                CollectionPgnResultV3::BlackWin,
                CollectionPgnTerminationV3::TimeForfeit {
                    loser: Color::White,
                },
            )
            .unwrap();
        assert!(output.contains("[Termination \"time forfeit\"]"));
        assert!(!output.ends_with("\n1-0\n\n"));
        assert!(!output.contains("1. e4"));
    }

    #[test]
    fn illegal_response_preserves_charged_wall_and_explicit_failure() {
        let initial = Position::startpos();
        let mut pgn = CollectionGamePgnV3::begin("g1", "A", "B", &initial).unwrap();
        assert!(
            pgn.push_actual_move(&initial, movement("e2e5"), Duration::from_secs(3))
                .is_err()
        );
        assert_eq!(pgn.used(Color::White), Duration::from_secs(3));
        assert_eq!(pgn.remaining(Color::White), Duration::from_secs(117));
        assert_eq!(pgn.clock_audit().unwrap().white_increments, 0);
        assert!(
            pgn.finish(
                CollectionPgnResultV3::Draw,
                CollectionPgnTerminationV3::IllegalMove {
                    loser: Color::White
                }
            )
            .is_err()
        );
    }

    #[test]
    fn exact_before_history_and_opening_san_are_preserved() {
        let mut initial = Position::startpos();
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            initial.make_move(movement(uci)).unwrap();
        }
        let imported = Position::from_fen(&initial.to_fen()).unwrap();
        let mut pgn = CollectionGamePgnV3::begin("g1", "A", "B", &initial).unwrap();
        assert!(
            pgn.push_actual_move(&imported, movement("e2e4"), Duration::from_secs(2))
                .is_err()
        );
        assert_eq!(pgn.used(Color::White), Duration::ZERO);
        pgn.push_actual_move(&initial, movement("e2e4"), Duration::from_secs(2))
            .unwrap();
        let output = pgn
            .finish(
                CollectionPgnResultV3::Incomplete,
                CollectionPgnTerminationV3::Cancelled,
            )
            .unwrap();
        assert!(output.contains("1. Nf3 Nf6 2. Ng1 Ng8 3. e4"));
        assert!(output.contains("[RoveZeroOpeningMoves \"g1f3 g8f6 f3g1 f6g8\"]"));
        assert!(output.contains("[RoveZeroOpeningPlies \"4\"]"));
        assert!(output.contains("[RoveZeroPlayedPlies \"1\"]"));
    }

    #[test]
    fn actual_rules_mate_has_canonical_san_and_result() {
        let mut live = Position::startpos();
        let mut pgn = CollectionGamePgnV3::begin("mate", "A", "B", &live).unwrap();
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"] {
            let mv = movement(uci);
            pgn.push_actual_move(&live, mv, Duration::from_secs(1))
                .unwrap();
            live.make_move(mv).unwrap();
        }
        let output = pgn
            .finish(
                CollectionPgnResultV3::BlackWin,
                CollectionPgnTerminationV3::RulesAutomatic,
            )
            .unwrap();
        assert!(output.contains("Qh4#"));
        assert!(output.contains("[Result \"0-1\"]\n[Termination \"checkmate\"]"));
    }

    #[test]
    fn current_rules_claim_required_and_infrastructure_stays_incomplete() {
        let initial = Position::startpos();
        let pgn = CollectionGamePgnV3::begin("g1", "A", "B", &initial).unwrap();
        assert!(
            pgn.finish(
                CollectionPgnResultV3::Draw,
                CollectionPgnTerminationV3::RulesClaim(ClaimReason::ThreefoldRepetition)
            )
            .is_err()
        );
        let mut pgn = CollectionGamePgnV3::begin("g1", "A", "B", &initial).unwrap();
        assert!(
            pgn.record_failed_decision(
                &initial,
                Duration::from_secs(1),
                CollectionDecisionFailureV3::Timeout
            )
            .is_err()
        );
        pgn.record_failed_decision(
            &initial,
            Duration::from_secs(1),
            CollectionDecisionFailureV3::Infrastructure,
        )
        .unwrap();
        let output = pgn
            .finish(
                CollectionPgnResultV3::Incomplete,
                CollectionPgnTerminationV3::Infrastructure,
            )
            .unwrap();
        assert!(output.contains("[Result \"*\"]\n[Termination \"infrastructure failure\"]"));
    }

    #[test]
    fn infrastructure_abort_keeps_terminal_trace_without_claiming_rules_result() {
        let mut live = Position::startpos();
        let mut pgn = CollectionGamePgnV3::begin("mate", "A", "B", &live).unwrap();
        for (uci, seconds) in [("f2f3", 3), ("e7e5", 2), ("g2g4", 3), ("d8h4", 2)] {
            let mv = movement(uci);
            pgn.push_actual_move(&live, mv, Duration::from_secs(seconds))
                .unwrap();
            live.make_move(mv).unwrap();
        }
        let audit = pgn.clock_audit().unwrap();
        let output = pgn.abort_infrastructure().unwrap();
        assert!(output.contains("Qh4#"));
        assert!(output.contains("[Result \"*\"]\n[Termination \"infrastructure failure\"]"));
        assert!(output.contains("[RoveZeroPlayedPlies \"4\"]"));
        assert!(output.contains("[RoveZeroDecisionWallNs \"10000000000\"]"));
        assert!(output.contains(&format!(
            "[RoveZeroWhiteRemainingNs \"{}\"]",
            audit.white_remaining_ns
        )));
        assert!(output.contains(&format!(
            "[RoveZeroBlackRemainingNs \"{}\"]",
            audit.black_remaining_ns
        )));
        assert!(output.ends_with("*\n\n"));
    }

    #[test]
    fn infrastructure_abort_preserves_previously_observed_engine_loss() {
        let initial = Position::startpos();
        for (failure, label) in [
            (CollectionDecisionFailureV3::Timeout, "time forfeit"),
            (CollectionDecisionFailureV3::EngineCrash, "engine crash"),
        ] {
            let mut pgn = CollectionGamePgnV3::begin("failed", "A", "B", &initial).unwrap();
            pgn.record_failed_decision(&initial, BASE, failure).unwrap();
            let output = pgn.abort_infrastructure().unwrap();
            assert!(output.contains(&format!("[Result \"0-1\"]\n[Termination \"{label}\"]")));
            assert!(output.contains("[RoveZeroPlayedPlies \"0\"]"));
            assert!(output.contains("[RoveZeroWhiteWallNs \"120000000000\"]"));
            assert!(!output.contains("[Termination \"infrastructure failure\"]"));
        }
        let mut pgn = CollectionGamePgnV3::begin("illegal", "A", "B", &initial).unwrap();
        assert!(
            pgn.push_actual_move(&initial, movement("e2e5"), Duration::from_secs(1))
                .is_err()
        );
        let output = pgn.abort_infrastructure().unwrap();
        assert!(output.contains("[Result \"0-1\"]\n[Termination \"illegal move\"]"));
        assert!(output.contains("[RoveZeroWhiteWallNs \"1000000000\"]"));
    }

    #[test]
    fn fen_black_start_and_engine_name_escaping_are_valid_pgn() {
        let initial =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
                .unwrap();
        let mut pgn = CollectionGamePgnV3::begin("g1", "PALS \"A\"", "PALS \\B", &initial).unwrap();
        pgn.push_actual_move(&initial, movement("e7e5"), Duration::ZERO)
            .unwrap();
        let output = pgn
            .finish(
                CollectionPgnResultV3::Incomplete,
                CollectionPgnTerminationV3::Cancelled,
            )
            .unwrap();
        assert!(output.contains("[SetUp \"1\"]\n[FEN \""));
        assert!(output.contains("[White \"PALS \\\"A\\\"\"]"));
        assert!(output.contains("1... e5"));
        assert!(CollectionGamePgnV3::begin("bad\ngame", "A", "B", &initial).is_err());
    }
}
