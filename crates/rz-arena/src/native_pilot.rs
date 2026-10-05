//! Fixed pilot cohort and pinned runner's whole-engine clock evidence.
//! Clock logs are supervisor-owned evidence, never engine-reported UCI time.

#[cfg(feature = "native-cuda")]
use crate::{ArenaError, PairPgnAudit};
use rz_experiments::OpeningSpec;
#[cfg(feature = "native-cuda")]
use rz_experiments::{CudaIntegrationPairSpec, CudaLaunchProfile, NativeGameClockV3};
use serde::{Deserialize, Serialize};
#[cfg(feature = "native-cuda")]
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
#[cfg(feature = "native-cuda")]
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
            let Some(message) = crate::native_cuda::cuda_exit_trace_message(line) else {
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
            let message = crate::native_cuda::native_runner_message(line, "[FATAL ] [", false)
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
#[cfg(feature = "native-cuda")]
fn invalid(message: &str) -> ArenaError {
    ArenaError::Integrity(message.into())
}

#[cfg(feature = "native-cuda")]
#[derive(Debug)]
struct ClockRecord<'a> {
    engine: &'a str,
    before: u64,
    increment: u64,
    elapsed_ns: u64,
    after: i64,
    valid: bool,
}
#[cfg(feature = "native-cuda")]
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
#[cfg(feature = "native-cuda")]
pub fn validate_pilot_clock_trace(
    stdout: &[u8],
    pgn: &PairPgnAudit,
    opening: &OpeningSpec,
    clock: NativeGameClockV3,
) -> Result<NativePilotClockAudit, ArenaError> {
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
        let message = crate::native_cuda::cuda_exit_trace_message(line)
            .ok_or_else(|| invalid("pilot clock renderer differs from pinned runner"))?;
        records.push(record(message)?);
        trace.update(line.as_bytes());
        trace.update(b"\n");
        if records.len() > 512 {
            return Err(invalid("pilot clock exceeds two 256 ply games"));
        }
    }
    let mut cursor = 0;
    let mut games = Vec::new();
    for g in &pgn.games {
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
            let side = ply % 2;
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
        games,
    })
}
