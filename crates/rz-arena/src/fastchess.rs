//! Invocation construction for one pinned, CPU-only synthetic fixture pair.
//! This builds arguments; it does not attest executables, PGN bytes, applied
//! options, physical resource enforcement or the shared clock/rules contract.

use crate::{ArenaError, ArenaPlan};
use rz_experiments::{
    ClockSpec, Comparison, Device, EngineKind, ResearchPath, Residency, RunPurpose,
};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub const FASTCHESS_SOURCE_URL: &str = "https://github.com/Disservin/fastchess";
pub const FASTCHESS_SOURCE_COMMIT: &str = "f618e34540f94f4719ad3817950618dabe441318";
pub const FASTCHESS_VERSION: &str = "fastchess alpha 1.8.2 20260726-f618e34";

#[derive(Clone, Debug)]
pub struct FastchessInvocation {
    pub args: Vec<OsString>,
    pub limitations: Vec<String>,
}

fn invalid(reason: &str) -> ArenaError {
    ArenaError::Invalid(reason.into())
}

fn path_argument(prefix: &str, path: &Path) -> Result<OsString, ArenaError> {
    // Keep the native path bytes. These arguments are passed directly to the
    // runner, never to a shell; Fastchess's key/value parser still requires
    // rejecting delimiter-like bytes and whitespace in this bounded profile.
    let bytes = path.as_os_str().as_encoded_bytes();
    if !path.is_absolute()
        || bytes.len() > 4096
        || bytes
            .iter()
            .any(|byte| *byte <= b' ' || *byte == 127 || b"=\"'\\".contains(byte))
        || path
            .to_str()
            .is_some_and(|text| text.chars().any(char::is_whitespace))
    {
        return Err(invalid(
            "Fastchess requires absolute native paths without token delimiters or whitespace",
        ));
    }
    let mut value = OsString::from(prefix);
    value.push(path);
    Ok(value)
}

fn deadline(value: u64) -> Result<String, ArenaError> {
    if value == 0 || value > u64::from(u32::MAX) {
        return Err(invalid(
            "Fastchess fixture deadlines must fit positive u32 milliseconds",
        ));
    }
    Ok(value.to_string())
}

/// Build one two-game fixture invocation in the pair's declared execution order.
///
/// The caller must create `opening_file` from `opening_pgn(plan, pair_id)`,
/// retain pinned executable handles, then audit the actual PGN against A Rules.
/// `plies` selects the full declared prefix; no board-only EPD shortcut is used.
/// The returned limitations prevent this fixture path being an execution permit
/// or evidence of official strength, GPU fairness or clock-contract compliance.
pub fn build_fastchess_invocation(
    plan: &ArenaPlan,
    pair_id: &str,
    engine_commands: &BTreeMap<String, PathBuf>,
    opening_file: &Path,
    pgn_file: &Path,
) -> Result<FastchessInvocation, ArenaError> {
    let input = plan.manifest().input();
    input.validate()?;
    let revision = rz_contracts::CONTRACT_REVISION;
    if revision.major != 0
        || revision.minor != 1
        || input.contract_revision.as_deref() != Some("0.1")
    {
        return Err(invalid(
            "Fastchess fixture adapter requires the native contract revision 0.1",
        ));
    }
    if input.purpose != RunPurpose::Fixture || input.comparison != Comparison::Fixture {
        return Err(invalid(
            "Fastchess fixture adapter rejects development/formal strength runs",
        ));
    }
    if input.research_path != ResearchPath::Cpu
        || input.hardware.residency != Residency::CpuOnly
        || input.hardware.gpu.is_some()
        || input.hardware.tablebase_enabled
        || input.protocol.tablebase_policy != "disabled-both-engines"
        || input.protocol.adjudication_enabled
        || input.lifecycle.warmup_input.is_some()
        || input.lifecycle.warmup_policy != "disabled"
        || input.lifecycle.uci_overhead_ms != 0
    {
        return Err(invalid(
            "Fastchess fixture profile requires CPU-only execution, no warmup/tablebase/score adjudication and zero declared UCI overhead",
        ));
    }
    if input.budget.workers != 1 || input.budget.max_child_processes < 3 {
        return Err(ArenaError::Budget(
            "Fastchess fixture requires one worker and capacity for the runner plus two engines"
                .into(),
        ));
    }
    let runner = &input.protocol.runner;
    if runner.source_url != FASTCHESS_SOURCE_URL
        || runner.source_commit != FASTCHESS_SOURCE_COMMIT
        || runner.version != FASTCHESS_VERSION
        || runner.dirty
        || runner.dirty_patch.is_some()
    {
        return Err(invalid(
            "unsupported Fastchess source/version or dirty runner",
        ));
    }
    let base_ms = match input.clock {
        ClockSpec::T1 {
            base_ms,
            increment_ms: 0,
        } if base_ms % 1000 == 0 && base_ms <= 86_400_000 => base_ms,
        _ => {
            return Err(invalid(
                "Fastchess fixture supports only T1 with zero increment and an integral-second base up to 24 hours",
            ));
        }
    };
    let pair = plan
        .pair(pair_id)
        .ok_or_else(|| invalid("unknown pair ID"))?;
    let prefix_plies = u32::try_from(pair.opening.moves.len())
        .map_err(|_| invalid("opening prefix exceeds Fastchess ply range"))?;
    let searched_plies = input
        .protocol
        .max_plies
        .checked_sub(prefix_plies)
        .ok_or_else(|| invalid("opening prefix exceeds the declared full game ply ceiling"))?;
    if searched_plies == 0 || searched_plies % 2 != 0 || searched_plies / 2 > i32::MAX as u32 {
        return Err(invalid(
            "Fastchess maxmoves requires a positive even searched-ply remainder after the full opening prefix",
        ));
    }
    if opening_file.extension() != Some(std::ffi::OsStr::new("pgn"))
        || pgn_file.extension() != Some(std::ffi::OsStr::new("pgn"))
        || opening_file == pgn_file
    {
        return Err(invalid(
            "opening and output must be distinct PGN paths; board-only EPD is unsupported",
        ));
    }
    let opening_argument = path_argument("file=", opening_file)?;
    let pgn_argument = path_argument("file=", pgn_file)?;
    if engine_commands.len() != 2
        || input
            .engines
            .iter()
            .any(|engine| !engine_commands.contains_key(&engine.id))
    {
        return Err(invalid(
            "exactly the two manifest engine command paths are required",
        ));
    }

    // In this pinned source, the first -engine is white in game 1 and the
    // second is white in game 2. Choose the first planned execution's colors;
    // this also supports [1,0] without relying on another version's -reverse.
    let first_game = &pair.games[pair.execution_order[0]];
    let mut args = Vec::new();
    for id in [&first_game.white_engine, &first_game.black_engine] {
        let engine = input
            .engines
            .iter()
            .find(|engine| &engine.id == id)
            .ok_or_else(|| invalid("pair references an undeclared engine"))?;
        if engine.kind != EngineKind::Fixture
            || engine.weight.is_some()
            || engine.device != Device::Cpu
        {
            return Err(invalid(
                "Fastchess fixture profile requires two CPU fixture engines without weights",
            ));
        }
        if engine.requested_options.get("Threads").map(String::as_str) != Some("1")
            || engine.requested_options.get("Ponder").map(String::as_str) != Some("false")
        {
            return Err(invalid(
                "fixture engines must explicitly request canonical Threads=1 and Ponder=false",
            ));
        }
        args.push("-engine".into());
        args.push(path_argument("cmd=", &engine_commands[id])?);
        args.push(format!("name={id}").into());
        for (name, value) in &engine.requested_options {
            if name.is_empty()
                || value.is_empty()
                || name
                    .chars()
                    .chain(value.chars())
                    .any(|character| character.is_whitespace() || character.is_control())
                || !name.bytes().all(|byte| byte.is_ascii_alphanumeric())
                || !value.bytes().all(|byte| byte.is_ascii_alphanumeric())
            {
                return Err(invalid(
                    "fixture option names/values must be single bounded alphanumeric tokens",
                ));
            }
            match name.as_str() {
                "Threads" if value == "1" => {}
                "Ponder" if value == "false" => {}
                "Hash" => {
                    let megabytes = value
                        .parse::<u32>()
                        .map_err(|_| invalid("Hash must be positive canonical integer MiB"))?;
                    if !(1..=1024).contains(&megabytes) || megabytes.to_string() != *value {
                        return Err(invalid("Hash must be canonical integer MiB in 1..=1024"));
                    }
                }
                _ => {
                    return Err(invalid(
                        "unsupported fixture engine option; per-game RNG seeds are not applied by a two-game invocation",
                    ));
                }
            }
            args.push(format!("option.{name}={value}").into());
        }
    }
    if input.lifecycle.timeout_grace_ms > i32::MAX as u64 {
        return Err(invalid(
            "Fastchess timemargin exceeds its signed integer millisecond range",
        ));
    }
    args.extend([
        OsString::from("-each"),
        "proto=uci".into(),
        format!("tc={}", base_ms / 1000).into(),
        "restart=on".into(),
        format!("timemargin={}", input.lifecycle.timeout_grace_ms).into(),
        "-openings".into(),
        opening_argument,
        "format=pgn".into(),
        "order=sequential".into(),
        format!("plies={prefix_plies}").into(),
        "start=1".into(),
        "-rounds".into(),
        "1".into(),
        "-games".into(),
        "2".into(),
        "-repeat".into(),
        "-concurrency".into(),
        "1".into(),
        "-srand".into(),
        input.input.seed.to_string().into(),
        "-maxmoves".into(),
        (searched_plies / 2).to_string().into(),
        "-pgnout".into(),
        pgn_argument,
        "append=false".into(),
        "notation=uci".into(),
        "min=false".into(),
        "timeleft=true".into(),
        "latency=true".into(),
        "-autosaveinterval".into(),
        "0".into(),
        "-ratinginterval".into(),
        "0".into(),
        "-startup-ms".into(),
        deadline(input.lifecycle.handshake_timeout_ms)?.into(),
        "-ping-ms".into(),
        deadline(input.lifecycle.handshake_timeout_ms)?.into(),
        "-ucinewgame-ms".into(),
        deadline(input.lifecycle.drain_timeout_ms)?.into(),
        "-strict".into(),
    ]);
    Ok(FastchessInvocation {
        args,
        limitations: [
            "fixture-development-only; execution_ready=false; no strength, GPU, backend or applied-option attestation",
            "Fastchess measures after go write/read setup and before bestmove validation; shared clock boundary is unverified and its read timeout has an internal 100 ms margin",
            "Fastchess automatic threefold/50-move/insufficient-material rules differ from A's explicit-claim contract; independently replay and audit each PGN",
            "maxmoves counts searched plies excluding the full prefix and reports adjudication draws; the audit must classify the limit as incomplete, never score it as a draw",
            "per-game derived engine seeds are not applied; this profile requires deterministic scripted fixtures and rejects RNG options",
            "opening path bytes are not attested by this pure builder; caller must write the complete A-validated opening_pgn and compare actual PGN/full prefix after execution",
            "process-group child/output/artifact snapshots are observed limits, not kernel resource caps; escaped or transient children require a future cgroup adapter",
        ].map(String::from).to_vec(),
    })
}
