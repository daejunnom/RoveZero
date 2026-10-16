//! Fixed pilot cohort and pinned runner's whole-engine clock evidence.
//! Clock logs are supervisor-owned evidence, never engine-reported UCI time.

use crate::{ArenaError, PairPgnAudit};
use rz_experiments::NativeGameClockV3;
use rz_experiments::OpeningSpec;
#[cfg(feature = "native-cuda")]
use rz_experiments::{CudaIntegrationPairSpec, CudaLaunchProfile};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(feature = "native-cuda")]
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativePilotOpeningCohortV3 {
    pub schema_version: u32,
    pub openings: Vec<OpeningSpec>,
}
#[cfg(feature = "native-cuda")]
pub(crate) fn validate_pilot_cohort<P: CudaLaunchProfile>(
    bytes: &[u8],
    input: &CudaIntegrationPairSpec<P>,
) -> Result<(), ArenaError> {
    let p = input
        .pilot
        .as_ref()
        .ok_or_else(|| invalid("pilot protocol absent"))?;
    if bytes.len() > 64 * 1024
        || bytes.len() as u64 != p.opening_cohort.bytes
        || format!("{:x}", Sha256::digest(bytes)) != p.opening_cohort.sha256
    {
        return Err(invalid("pilot cohort differs from its immutable artifact"));
    }
    let cohort: NativePilotOpeningCohortV3 = crate::decode_json(
        std::str::from_utf8(bytes).map_err(|_| invalid("pilot cohort is not UTF-8"))?,
    )?;
    if cohort.schema_version != 3
        || cohort.openings.len() != p.total_pairs as usize
        || cohort.openings.get(p.pair_ordinal as usize) != Some(&input.opening)
    {
        return Err(invalid(
            "pilot opening differs from the registered ordinal/cohort",
        ));
    }
    let mut ids = BTreeSet::new();
    let mut traces = BTreeSet::new();
    for o in &cohort.openings {
        if o.initial != rz_experiments::InitialPosition::Startpos
            || o.history != rz_experiments::HistoryCompleteness::Complete
            || !(6..=12).contains(&o.moves.len())
            || o.moves.len() % 2 != 0
            || !ids.insert(&o.id)
            || !traces.insert(&o.moves)
        {
            return Err(invalid(
                "pilot cohort requires distinct complete 6..12 ply startpos traces",
            ));
        }
        // Legality and complete state are owned by A, including all book plies.
        crate::opening_pgn_for_spec(o, input.max_plies)?;
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
pub struct NativePilotClockAudit {
    pub audit_version: u32,
    pub boundary: String,
    pub source: String,
    pub rounding: String,
    pub read_margin_ms: u64,
    pub trace_sha256: String,
    pub expected_pgn_time_control: String,
    /// Header differences are metadata diagnostics, not clock rejection gates.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pgn_time_control_warnings: Vec<String>,
    pub games: Vec<NativePilotGameClockAudit>,
}
#[derive(Clone, Debug, Serialize)]
pub struct NativePilotGameClockAudit {
    pub game_id: String,
    pub searched_plies: u32,
    pub white_remaining_ms: u64,
    pub black_remaining_ms: u64,
    pub charged_ms: [u64; 2],
}

/// Closed runner observations for losses that occur before any game PGN exists.
/// The original attempt still fails its process/provider/clock acceptance gates.
#[derive(Clone, Debug, Serialize)]
pub struct NativePilotFailureAudit {
    pub audit_version: u32,
    pub source: String,
    pub stdout_sha256: String,
    pub startup_losses: Vec<NativePilotStartupLoss>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativePilotStartupLoss {
    pub game_id: String,
    pub white_engine: String,
    pub black_engine: String,
    pub loser_engine: String,
    pub declared_result: crate::GameResult,
    pub failure_stage: String,
    pub cause: String,
}

/// Decode only the pinned runner's FATAL startup renderer and its ordered game
/// starts. Engine stderr can explain a cause but cannot manufacture a loss.
/// Unsupported fatal messages remain errors, never silently excluded outcomes.
pub fn audit_pilot_startup_failure_trace(
    stdout: &[u8],
    pair: &crate::PairSpec,
) -> Result<Option<NativePilotFailureAudit>, ArenaError> {
    if stdout.is_empty()
        || stdout.len() > 64 * 1024 * 1024
        || !stdout.ends_with(b"\n")
        || !matches!(pair.execution_order, [0, 1] | [1, 0])
    {
        return Err(invalid("startup failure trace bounds or game order differ"));
    }
    let text =
        std::str::from_utf8(stdout).map_err(|_| invalid("startup failure trace is not UTF-8"))?;
    let mut started = 0_usize;
    let mut active = None;
    let mut losses = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index >= 131_072 || line.len() > 4096 {
            return Err(invalid("startup failure trace line budget exceeded"));
        }
        if line.starts_with("[TRACE ] [") {
            let Some(message) = crate::native_exit::cuda_exit_trace_message(line) else {
                continue;
            };
            if message.starts_with("Game ") && message.ends_with(" starting") {
                if active.is_some() || started >= 2 || !losses.is_empty() {
                    return Err(invalid(
                        "startup failure trace has overlapping/extra starts",
                    ));
                }
                let game = &pair.games[pair.execution_order[started]];
                let expected = format!(
                    "Game {} between {} and {} starting",
                    started + 1,
                    game.white_engine,
                    game.black_engine
                );
                if message != expected {
                    return Err(invalid("startup failure game colors/order differ"));
                }
                active = Some(started);
                started += 1;
            } else if message.starts_with("Game ") && message.ends_with(" finished") {
                let ordinal = active
                    .ok_or_else(|| invalid("startup failure trace has an unmatched game end"))?;
                let game = &pair.games[pair.execution_order[ordinal]];
                if message
                    != format!(
                        "Game {} between {} and {} finished",
                        ordinal + 1,
                        game.white_engine,
                        game.black_engine
                    )
                {
                    return Err(invalid("startup failure game end differs"));
                }
                active = None;
            }
        } else if line.starts_with("[FATAL") {
            let message = crate::native_exit::native_runner_message(line, "[FATAL ] [", false)
                .ok_or_else(|| invalid("startup failure FATAL renderer is malformed"))?;
            let ordinal =
                active.ok_or_else(|| invalid("startup failure has no matching active game"))?;
            let game = &pair.games[pair.execution_order[ordinal]];
            let loser = message
                .strip_prefix("Fatal; ")
                .and_then(|v| {
                    v.strip_suffix(
                        " engine startup failure: \"Engine didn't respond to uciok after startup\"",
                    )
                })
                .ok_or_else(|| invalid("unsupported runner fatal cause"))?;
            if !losses.is_empty() || (loser != game.white_engine && loser != game.black_engine) {
                return Err(invalid("startup failure has a duplicate/unknown loser"));
            }
            losses.push(NativePilotStartupLoss {
                game_id: game.id.clone(),
                white_engine: game.white_engine.clone(),
                black_engine: game.black_engine.clone(),
                loser_engine: loser.into(),
                declared_result: if loser == game.white_engine {
                    crate::GameResult::BlackWin
                } else {
                    crate::GameResult::WhiteWin
                },
                failure_stage: "startup_uci_handshake".into(),
                cause: "pinned_runner_rejected_uci_startup".into(),
            });
        }
    }
    if losses.is_empty() {
        Ok(None)
    } else {
        Ok(Some(NativePilotFailureAudit {
            audit_version: 1,
            source: "pinned_fastchess_game_start_and_fatal_renderer".into(),
            stdout_sha256: format!("{:x}", Sha256::digest(stdout)),
            startup_losses: losses,
        }))
    }
}
fn invalid(message: &str) -> ArenaError {
    ArenaError::Integrity(message.into())
}

#[derive(Debug)]
struct ClockRecord<'a> {
    engine: &'a str,
    before: u64,
    increment: u64,
    elapsed_ns: u64,
    after: i64,
    valid: bool,
}
fn record(message: &str) -> Result<ClockRecord<'_>, ArenaError> {
    let fields: Vec<_> = message.split(' ').collect();
    if fields.len() != 7 || fields[0] != "RZ_CLOCK_V1" {
        return Err(invalid("clock record has extra/missing fields"));
    }
    let field = |n: usize, name: &str| {
        fields[n]
            .strip_prefix(name)
            .ok_or_else(|| invalid("clock record field order differs"))
    };
    let number = |n: usize, name: &str| -> Result<u64, ArenaError> {
        let text = field(n, name)?;
        let v = text
            .parse::<u64>()
            .map_err(|_| invalid("clock record integer is malformed"))?;
        if text != v.to_string() {
            return Err(invalid("clock record integer is not canonical"));
        }
        Ok(v)
    };
    let after_text = field(5, "after=")?;
    let after = after_text
        .parse::<i64>()
        .map_err(|_| invalid("clock remaining is malformed"))?;
    if after_text != after.to_string() {
        return Err(invalid("clock remaining is not canonical"));
    }
    let valid = match field(6, "valid=")? {
        "true" => true,
        "false" => false,
        _ => return Err(invalid("clock result is malformed")),
    };
    Ok(ClockRecord {
        engine: field(1, "engine=")?,
        before: number(2, "before=")?,
        increment: number(3, "increment=")?,
        elapsed_ns: number(4, "elapsed_ns=")?,
        after,
        valid,
    })
}

/// Validate every monotonic charge against the independently A-audited PGN and
/// the same two reset clocks. Quoted [Engine] payloads cannot forge this trace.
/// Engine-loss PGNs are retained by the caller; an incomplete clock/provider
/// gate stops the pilot and cannot turn an observed loss into an excluded win.
pub fn validate_pilot_clock_trace(
    stdout: &[u8],
    pgn: &PairPgnAudit,
    opening: &OpeningSpec,
    clock: NativeGameClockV3,
) -> Result<NativePilotClockAudit, ArenaError> {
    validate_pilot_clock_trace_with_ply_limit(stdout, pgn, opening, clock, 256)
}

/// Same pinned runner codec and Rules-audited charges, with the independently
/// declared V5 collection ceiling. The legacy entry retains its 256-ply limit.
/// This never infers charges from engine-reported time or a PGN header.
pub fn validate_pilot_clock_trace_with_ply_limit(
    stdout: &[u8],
    pgn: &PairPgnAudit,
    opening: &OpeningSpec,
    clock: NativeGameClockV3,
    max_game_plies: u32,
) -> Result<NativePilotClockAudit, ArenaError> {
    if !matches!(max_game_plies, 256 | 512) {
        return Err(invalid(
            "pilot clock ply ceiling is not a registered 256/512 profile",
        ));
    }
    if stdout.is_empty()
        || stdout.len() > 64 * 1024 * 1024
        || !stdout.ends_with(b"\n")
        || pgn.games.len() != 2
    {
        return Err(invalid("pilot clock trace bounds or game count differ"));
    }
    let text =
        std::str::from_utf8(stdout).map_err(|_| invalid("pilot clock trace is not UTF-8"))?;
    let mut records = Vec::new();
    let mut trace = Sha256::new();
    for (index, line) in text.lines().enumerate() {
        if index >= 131_072 || line.len() > 4096 {
            return Err(invalid("pilot clock trace line budget exceeded"));
        }
        if !line.starts_with("[TRACE") || !line.contains("RZ_CLOCK_V1") {
            continue;
        }
        let message = crate::native_exit::cuda_exit_trace_message(line)
            .ok_or_else(|| invalid("pilot clock renderer differs from pinned runner"))?;
        records.push(record(message)?);
        trace.update(line.as_bytes());
        trace.update(b"\n");
        if records.len() > max_game_plies as usize * 2 {
            return Err(invalid(if max_game_plies == 256 {
                "pilot clock exceeds two 256 ply games"
            } else {
                "pilot clock exceeds two 512 ply games"
            }));
        }
    }
    let mut cursor = 0;
    let black_origin = match opening.initial {
        rz_experiments::InitialPosition::Startpos => false,
        rz_experiments::InitialPosition::Fen => {
            rz_position::Position::from_fen(
                opening
                    .fen
                    .as_deref()
                    .ok_or_else(|| invalid("clock FEN opening lacks exact origin"))?,
            )
            .map_err(|_| invalid("clock FEN opening is invalid"))?
            .side_to_move()
                == rz_position::Color::Black
        }
    };
    let mut games = Vec::new();
    let time_control = clock.pgn_time_control();
    let mut pgn_time_control_warnings = Vec::new();
    for g in &pgn.games {
        if g.uci_moves.len() > max_game_plies as usize {
            return Err(invalid(
                "one audited game exceeds its registered ply ceiling",
            ));
        }
        if g.time_control.as_deref() != Some(time_control.as_str()) {
            pgn_time_control_warnings.push(format!(
                "{}: PGN TimeControl {:?} differs from actual clock {}; header diagnostic only",
                g.game_id, g.time_control, time_control
            ));
        }
        if g.classification == "engine_loss" {
            return Err(invalid(
                "engine loss retained; full pilot clock/provider gate is not reusable",
            ));
        }
        if g.uci_moves.get(..opening.moves.len()) != Some(opening.moves.as_slice()) {
            return Err(invalid("clock PGN opening differs"));
        }
        let mut remaining = [clock.base_ms; 2];
        let mut charged = [0_u64; 2];
        for ply in opening.moves.len()..g.uci_moves.len() {
            let side = (ply + usize::from(black_origin)) % 2;
            let engine = if side == 0 {
                &g.white_engine
            } else {
                &g.black_engine
            };
            let r = records
                .get(cursor)
                .ok_or_else(|| invalid("pilot clock evidence is missing"))?;
            cursor += 1;
            let elapsed = r
                .elapsed_ns
                .checked_add(999_999)
                .ok_or_else(|| invalid("clock duration overflow"))?
                / 1_000_000;
            let after = remaining[side]
                .checked_sub(elapsed)
                .and_then(|v| v.checked_add(clock.increment_ms))
                .ok_or_else(|| invalid("late bestmove cannot pass the pilot clock gate"))?;
            if r.engine != engine
                || r.before != remaining[side]
                || r.increment != clock.increment_ms
                || !r.valid
                || i64::try_from(after).ok() != Some(r.after)
            {
                return Err(invalid(
                    "pilot clock charge, identity, increment or reset differs",
                ));
            }
            charged[side] = charged[side]
                .checked_add(elapsed)
                .ok_or_else(|| invalid("clock charge overflow"))?;
            remaining[side] = after;
        }
        games.push(NativePilotGameClockAudit {
            game_id: g.game_id.clone(),
            searched_plies: (g.uci_moves.len() - opening.moves.len()) as u32,
            white_remaining_ms: remaining[0],
            black_remaining_ms: remaining[1],
            charged_ms: charged,
        });
    }
    if cursor != records.len() || cursor == 0 {
        return Err(invalid("pilot clock has extra, foreign or no move records"));
    }
    Ok(NativePilotClockAudit {
        audit_version: 1,
        boundary: "before position transmission through bestmove reception".into(),
        source: "pinned runner steady_clock".into(),
        rounding: "ceil nanoseconds to milliseconds".into(),
        read_margin_ms: 100,
        trace_sha256: format!("{:x}", trace.finalize()),
        expected_pgn_time_control: time_control,
        pgn_time_control_warnings,
        games,
    })
}

#[cfg(test)]
mod ply_ceiling_tests {
    use super::*;
    use crate::{GamePgnAudit, GameResult};
    use rz_experiments::{HistoryCompleteness, InitialPosition};
    fn fixture(count: usize, black_origin: bool) -> (OpeningSpec, PairPgnAudit, Vec<u8>) {
        let opening = OpeningSpec {
            id: "clock-codec-fixture".into(),
            initial: if black_origin {
                InitialPosition::Fen
            } else {
                InitialPosition::Startpos
            },
            fen: black_origin
                .then(|| "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1".into()),
            moves: vec![],
            history: if black_origin {
                HistoryCompleteness::UnknownPrefix
            } else {
                HistoryCompleteness::Complete
            },
            history_origin: "authored clock-only fixture".into(),
        };
        // This is a clock-codec fixture, not an independently Rules-audited
        // game. Production receives the existing Rules audit before this gate.
        let mut trace = String::new();
        let mut games = Vec::new();
        for game in 0..2 {
            let white = if game == 0 { "a" } else { "b" };
            let black = if game == 0 { "b" } else { "a" };
            let mut remaining = [120000u64; 2];
            for ply in 0..count {
                let side = (ply + usize::from(black_origin)) % 2;
                let before = remaining[side];
                remaining[side] = before - 1 + 1000;
                let name = if side == 0 { white } else { black };
                trace.push_str(&format!("[TRACE ] [10:51:44.181984] <     137433152747200> fastchess --- RZ_CLOCK_V1 engine={name} before={before} increment=1000 elapsed_ns=1 after={} valid=true\n",remaining[side]));
            }
            games.push(GamePgnAudit {
                game_id: format!("g{game}"),
                white_engine: white.into(),
                black_engine: black.into(),
                time_control: Some("120+1".into()),
                uci_moves: vec!["g8f6".into(); count],
                result: GameResult::Draw,
                termination: "fixture-only".into(),
                final_fen: "not a Rules claim".into(),
                classification: "fixture-only".into(),
                loser_engine: None,
                terminal_reason: None,
                engine_failure: None,
                failure_evidence: None,
            });
        }
        (
            opening,
            PairPgnAudit {
                audit_version: 1,
                rules_source_commit: "fixture".into(),
                contract_source_commit: "fixture".into(),
                engine_contract_revision: "0.1".into(),
                opening_input_sha256: "0".repeat(64),
                start_state_semantic_sha256: "0".repeat(64),
                games,
            },
            trace.into_bytes(),
        )
    }
    fn clock() -> NativeGameClockV3 {
        NativeGameClockV3 {
            base_ms: 120000,
            increment_ms: 1000,
        }
    }
    #[test]
    fn explicit_collection_clock_does_not_expand_legacy_ceiling() {
        let (o, p, t) = fixture(257, false);
        assert!(validate_pilot_clock_trace(&t, &p, &o, clock()).is_err());
        assert!(validate_pilot_clock_trace_with_ply_limit(&t, &p, &o, clock(), 512).is_ok());
        assert!(validate_pilot_clock_trace_with_ply_limit(&t, &p, &o, clock(), 1024).is_err());
    }
    #[test]
    fn clock_replay_preserves_black_fen_origin() {
        let (o, p, t) = fixture(1, true);
        let a = validate_pilot_clock_trace_with_ply_limit(&t, &p, &o, clock(), 256).unwrap();
        assert_eq!(a.games[0].charged_ms, [0, 1]);
        assert_eq!(a.games[0].white_remaining_ms, 120000);
        assert_eq!(a.games[0].black_remaining_ms, 120999);
    }
}
