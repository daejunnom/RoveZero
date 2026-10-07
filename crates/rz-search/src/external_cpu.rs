//! Startup-selected external UCI CPU_R checker, independent of arena opponents.
//!
//! External cp/mate/bounds and reported selective depth remain external reports.
//! A bestmove does not certify minimax completion or a Rules game outcome. One
//! owner issues one active `go`; cancellation is followed by bounded stop/drain.
//! Linux execution uses a pinned native ELF, an owned process group, nonblocking
//! pipes and waitid(WNOWAIT). Windows remains explicitly unsupported here.

use crate::cpu::{CpuLimits, CpuResumeToken};
use crate::cpu_checker::*;
use rz_position::{BoardMove, HistoryOrigin, PlayStatus, Position};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsString;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const EXTERNAL_UCI_CHECKER_VERSION: &str = "rz-external-uci-checker/0.1";
pub const EXTERNAL_UCI_SCORE_SEMANTICS: &str = "uci-native-cp-or-signed-mate-moves;root-side-to-move;reported-bounds;selectivity-unknown;no-cp-wdl-calibration;no-rules-proof";

/// Files are opened, streamed and verified before spawn. Identity declarations
/// do not prove training, license rights, an applied option or actual precision.
#[derive(Clone, Debug)]
pub struct ExternalCpuConfig {
    pub program: PathBuf,
    pub arguments: Vec<OsString>,
    pub working_directory: PathBuf,
    pub identity: ExternalCheckerIdentity,
    pub expected_uci_name: Option<String>,
    pub max_depth: u16,
    pub max_prefix_plies: usize,
    pub handshake_timeout: Duration,
    pub max_task_wall_time: Duration,
    pub stop_grace: Duration,
    pub shutdown_grace: Duration,
    /// Combined stdout/stderr bytes over this process's whole lifetime.
    pub max_output_bytes: usize,
    pub max_line_bytes: usize,
}

impl ExternalCpuConfig {
    pub fn validate(&self) -> Result<(), CheckerError> {
        self.identity.validate()?;
        if self.identity.adapter_semantics != EXTERNAL_UCI_SCORE_SEMANTICS {
            return Err(CheckerError::Invalid("external score semantics mismatch"));
        }
        if !self.program.is_absolute() || !self.working_directory.is_absolute() {
            return Err(CheckerError::Invalid("external paths must be absolute"));
        }
        if self.max_depth == 0 || self.max_depth > 64 || self.max_prefix_plies > 4096 {
            return Err(CheckerError::Invalid("external depth/history cap"));
        }
        if self.arguments.len() > 64
            || self
                .arguments
                .iter()
                .try_fold(0usize, |total, arg| {
                    total.checked_add(arg.as_encoded_bytes().len())
                })
                .is_none_or(|bytes| bytes > 16_384)
            || self
                .arguments
                .iter()
                .any(|arg| arg.as_encoded_bytes().contains(&0))
        {
            return Err(CheckerError::Invalid("external argument budget"));
        }
        if arguments_sha256(&self.arguments) != self.identity.launch_arguments_sha256 {
            return Err(CheckerError::Invalid("external argument identity mismatch"));
        }
        if !self.identity.assets.is_empty() {
            return Err(CheckerError::Unsupported(
                "external separately loaded assets require a pinned launch profile",
            ));
        }
        for time in [
            self.handshake_timeout,
            self.max_task_wall_time,
            self.stop_grace,
            self.shutdown_grace,
        ] {
            if time.is_zero() || time > Duration::from_secs(180) {
                return Err(CheckerError::Invalid("external finite wall budget"));
            }
        }
        if self.max_output_bytes == 0
            || self.max_output_bytes > 16 * 1024 * 1024
            || self.max_line_bytes == 0
            || self.max_line_bytes > 65_536
            || self.max_line_bytes > self.max_output_bytes
        {
            return Err(CheckerError::Invalid("external output budget"));
        }
        if self
            .expected_uci_name
            .as_ref()
            .is_some_and(|name| !bounded_text(name, 256))
        {
            return Err(CheckerError::Invalid("external expected name"));
        }
        Ok(())
    }
}

/// Domain-separated byte identity; argument boundaries are preserved.
pub fn arguments_sha256(arguments: &[OsString]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"rz-external-uci-arguments/1\0");
    hash.update((arguments.len() as u64).to_le_bytes());
    for arg in arguments {
        hash.update((arg.as_encoded_bytes().len() as u64).to_le_bytes());
        hash.update(arg.as_encoded_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn bounded_text(text: &String, limit: usize) -> bool {
    !text.trim().is_empty()
        && text.len() <= limit
        && text.capacity() <= limit
        && !text.chars().any(char::is_control)
}

fn error(stage: &'static str, code: &'static str) -> CheckerError {
    CheckerError::External { stage, code }
}

fn bounded_deadline(
    start: Instant,
    duration: Duration,
    supplied: Option<Instant>,
) -> Result<Instant, CheckerError> {
    let local = start
        .checked_add(duration)
        .ok_or(CheckerError::Invalid("external deadline overflow"))?;
    Ok(supplied.map_or(local, |deadline| deadline.min(local)))
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum OptionKind {
    Spin { min: i64, max: i64 },
    Check,
    Combo(Vec<String>),
    String,
    Button,
}

fn parse_option(line: &str) -> Result<(String, OptionKind), CheckerError> {
    let (name, declaration) = line
        .strip_prefix("option name ")
        .and_then(|body| body.split_once(" type "))
        .ok_or_else(|| error("handshake", "malformed_option"))?;
    if name.is_empty() || name.len() > 256 || name.chars().any(char::is_control) {
        return Err(error("handshake", "invalid_option_name"));
    }
    let tokens: Vec<&str> = declaration.split_ascii_whitespace().collect();
    let kind = match tokens.first().copied() {
        Some("spin") => {
            let field = |key: &str| -> Result<i64, CheckerError> {
                let index = tokens
                    .iter()
                    .position(|&word| word == key)
                    .ok_or_else(|| error("handshake", "spin_range_missing"))?;
                tokens
                    .get(index + 1)
                    .and_then(|value| value.parse().ok())
                    .ok_or_else(|| error("handshake", "spin_range_invalid"))
            };
            let min = field("min")?;
            let max = field("max")?;
            if min > max {
                return Err(error("handshake", "spin_range_invalid"));
            }
            OptionKind::Spin { min, max }
        }
        Some("check") => OptionKind::Check,
        Some("combo") => {
            let mut values = Vec::new();
            let mut at = 0;
            while at < tokens.len() {
                if tokens[at] == "var" {
                    let begin = at + 1;
                    at = begin;
                    while at < tokens.len() && tokens[at] != "var" {
                        at += 1;
                    }
                    if begin == at {
                        return Err(error("handshake", "empty_combo_value"));
                    }
                    values.push(tokens[begin..at].join(" "));
                } else {
                    at += 1;
                }
            }
            if values.is_empty() || values.len() > 64 {
                return Err(error("handshake", "combo_values_invalid"));
            }
            OptionKind::Combo(values)
        }
        Some("string") => OptionKind::String,
        Some("button") => OptionKind::Button,
        _ => return Err(error("handshake", "unsupported_option_type")),
    };
    Ok((name.to_owned(), kind))
}

fn validate_option(kind: &OptionKind, value: &str) -> bool {
    match kind {
        OptionKind::Spin { min, max } => value
            .parse::<i64>()
            .is_ok_and(|number| number >= *min && number <= *max),
        OptionKind::Check => matches!(value, "true" | "false"),
        OptionKind::Combo(values) => values.iter().any(|candidate| candidate == value),
        OptionKind::String => value.len() <= 4096 && !value.chars().any(char::is_control),
        OptionKind::Button => value.is_empty(),
    }
}

#[derive(Clone, Debug, Default)]
struct InfoFrame {
    depth: Option<u16>,
    seldepth: Option<u16>,
    nodes: Option<u64>,
    score: Option<ExternalRawScore>,
    bound: Option<ExternalBound>,
    wdl: Option<[u16; 3]>,
    pv: Option<Vec<BoardMove>>,
    multipv: u16,
}

/// Each accepted score/PV/depth tuple comes from one complete info line. Sparse
/// lines update observed work only; they cannot stitch a score onto a later PV.
fn parse_info(line: &str, max_pv: usize) -> Result<InfoFrame, CheckerError> {
    let words: Vec<&str> = line.split_ascii_whitespace().collect();
    if words.first() != Some(&"info") {
        return Err(error("analysis", "not_info"));
    }
    let mut frame = InfoFrame {
        multipv: 1,
        ..InfoFrame::default()
    };
    let mut at = 1;
    while at < words.len() {
        match words[at] {
            "string" => break,
            "depth" | "seldepth" | "multipv" => {
                let number = words
                    .get(at + 1)
                    .and_then(|word| word.parse::<u16>().ok())
                    .ok_or_else(|| error("analysis", "invalid_depth_field"))?;
                match words[at] {
                    "depth" => frame.depth = Some(number),
                    "seldepth" => frame.seldepth = Some(number),
                    _ => frame.multipv = number,
                }
                at += 2;
            }
            "nodes" => {
                frame.nodes = Some(
                    words
                        .get(at + 1)
                        .and_then(|word| word.parse().ok())
                        .ok_or_else(|| error("analysis", "invalid_nodes"))?,
                );
                at += 2;
            }
            "score" => {
                let value = words
                    .get(at + 2)
                    .and_then(|word| word.parse::<i32>().ok())
                    .ok_or_else(|| error("analysis", "invalid_score"))?;
                frame.score = Some(match words.get(at + 1).copied() {
                    Some("cp") => ExternalRawScore::Centipawns(value),
                    Some("mate") => ExternalRawScore::MateMoves(value),
                    _ => return Err(error("analysis", "unsupported_score_unit")),
                });
                frame.bound = Some(ExternalBound::ExactReported);
                at += 3;
                if let Some(&kind) = words.get(at) {
                    if matches!(kind, "lowerbound" | "upperbound") {
                        frame.bound = Some(if kind == "lowerbound" {
                            ExternalBound::Lower
                        } else {
                            ExternalBound::Upper
                        });
                        at += 1;
                    }
                }
            }
            "wdl" => {
                let mut values = [0u16; 3];
                for (offset, value) in values.iter_mut().enumerate() {
                    *value = words
                        .get(at + 1 + offset)
                        .and_then(|word| word.parse().ok())
                        .ok_or_else(|| error("analysis", "invalid_wdl"))?;
                }
                if values.iter().map(|&value| u32::from(value)).sum::<u32>() != 1000 {
                    return Err(error("analysis", "invalid_wdl_sum"));
                }
                frame.wdl = Some(values);
                at += 4;
            }
            "pv" => {
                if words.len() - at - 1 > max_pv {
                    return Err(error("analysis", "pv_budget"));
                }
                let pv = words[at + 1..]
                    .iter()
                    .map(|word| {
                        BoardMove::from_uci(word).map_err(|_| error("analysis", "invalid_pv_move"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if pv.is_empty() {
                    return Err(error("analysis", "empty_pv"));
                }
                frame.pv = Some(pv);
                break;
            }
            // These fields are preserved in the bounded transcript. Unknown
            // extension fields do not supply score/depth/coverage observations.
            _ => at += 1,
        }
    }
    if frame.multipv == 0 {
        return Err(error("analysis", "invalid_multipv"));
    }
    Ok(frame)
}

fn validate_pv(
    position: &Position,
    pv: &[BoardMove],
    roots: Option<&[BoardMove]>,
) -> Result<(), CheckerError> {
    if roots.is_some_and(|mask| pv.first().is_some_and(|mv| !mask.contains(mv))) {
        return Err(error("analysis", "root_mask_escape"));
    }
    let mut current = position.clone();
    for &mv in pv {
        let view = current.ordered_legal_moves();
        if current
            .play_status_from_view(&view)
            .map_err(|_| error("analysis", "rules_classification"))?
            != PlayStatus::Ongoing
        {
            return Err(error("analysis", "pv_after_rules_terminal"));
        }
        current
            .make_from_view(&view, mv)
            .map_err(|_| error("analysis", "illegal_pv"))?;
    }
    Ok(())
}

pub struct ExternalUciCpuChecker {
    config: ExternalCpuConfig,
    identity: CheckerIdentity,
    conditions: String,
    observed_uci: ExternalUciIdentity,
    owner: Option<process::OwnedProcess>,
    last_attempt: Option<CheckerAttempt>,
    request_id: u64,
    active: bool,
    failed: bool,
    failed_output: Option<(Vec<u8>, Vec<u8>)>,
}

impl ExternalUciCpuChecker {
    /// Create an unstarted owner so failed startup evidence remains available.
    /// The caller must use `start` and inspect `last_attempt` on failure.
    pub fn create(config: ExternalCpuConfig) -> Result<Self, CheckerError> {
        config.validate()?;
        let conditions = format!(
            "{};score:{};depth-cap:{};prefix-cap:{};resume:unsupported;root:searchmoves;go:depth+reported-node-limit+finite-wall;single-active;game:ucinewgame+ready;output-cap:{};line-cap:{};task-ms:{};stop-ms:{};shutdown-ms:{}",
            EXTERNAL_UCI_CHECKER_VERSION,
            EXTERNAL_UCI_SCORE_SEMANTICS,
            config.max_depth,
            config.max_prefix_plies,
            config.max_output_bytes,
            config.max_line_bytes,
            config.max_task_wall_time.as_millis(),
            config.stop_grace.as_millis(),
            config.shutdown_grace.as_millis()
        );
        Ok(Self {
            identity: CheckerIdentity::ExternalUci(config.identity.clone()),
            config,
            conditions,
            observed_uci: ExternalUciIdentity {
                name: String::new(),
                author: None,
            },
            owner: None,
            last_attempt: None,
            request_id: 0,
            active: false,
            failed: false,
            failed_output: None,
        })
    }

    /// Streamed executable validation, spawn and handshake are on the caller's
    /// finite startup clock. This object survives failure with evidence intact.
    pub fn start(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        self.last_attempt = None;
        if self.failed || self.owner.is_some() {
            return Err(error("startup", "already_started_or_failed"));
        }
        let start = Instant::now();
        let deadline = bounded_deadline(start, self.config.handshake_timeout, Some(deadline))?;
        if cancel.load(Ordering::Acquire) || start >= deadline {
            return Err(error("startup", "cancel_or_deadline"));
        }
        match process::OwnedProcess::spawn(&self.config, deadline, cancel) {
            Ok(owner) => self.owner = Some(owner),
            Err(failure) => {
                self.failed = true;
                self.last_attempt = Some(CheckerAttempt {
                    work: CheckerWork::default(),
                    elapsed: start.elapsed(),
                    external: Some(ExternalAttemptEvidence {
                        request_id: 0,
                        partial_report: None,
                        process: failure.process,
                    }),
                });
                self.failed_output = Some((failure.stdout, failure.stderr));
                return Err(failure.error);
            }
        }
        let result = self.handshake(deadline, cancel);
        if result.is_err() {
            self.failed = true;
            let cleanup = Instant::now()
                .checked_add(self.config.shutdown_grace)
                .unwrap_or_else(Instant::now);
            let _ = self.shutdown(cleanup);
        }
        self.last_attempt = Some(CheckerAttempt {
            work: CheckerWork::default(),
            elapsed: start.elapsed(),
            external: Some(ExternalAttemptEvidence {
                request_id: 0,
                partial_report: None,
                process: self
                    .owner
                    .as_ref()
                    .map_or_else(empty_shutdown, |owner| owner.evidence()),
            }),
        });
        result
    }

    pub fn config(&self) -> &ExternalCpuConfig {
        &self.config
    }

    /// Transcript storage is bounded over the entire process lifetime. Absence
    /// of an option application response remains unknown, even after readyok.
    pub fn diagnostic_output(&self) -> (&[u8], &[u8]) {
        self.owner.as_ref().map_or_else(
            || {
                self.failed_output
                    .as_ref()
                    .map_or((&[][..], &[][..]), |(stdout, stderr)| {
                        (stdout.as_slice(), stderr.as_slice())
                    })
            },
            |owner| owner.output(),
        )
    }

    fn owner(&mut self) -> Result<&mut process::OwnedProcess, CheckerError> {
        self.owner
            .as_mut()
            .ok_or_else(|| error("process", "owner_closed"))
    }

    fn handshake(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        self.owner()?.send("uci", deadline)?;
        let mut options = BTreeMap::new();
        loop {
            let line = self
                .owner()?
                .next_line(deadline, cancel)?
                .ok_or_else(|| error("handshake", "timeout_or_cancel"))?;
            if line == "uciok" {
                break;
            }
            if let Some(name) = line.strip_prefix("id name ") {
                if !self.observed_uci.name.is_empty() || name.is_empty() || name.len() > 256 {
                    return Err(error("handshake", "invalid_engine_name"));
                }
                self.observed_uci.name = name.to_owned();
            } else if let Some(author) = line.strip_prefix("id author ") {
                if self.observed_uci.author.is_some() || author.len() > 256 {
                    return Err(error("handshake", "invalid_engine_author"));
                }
                self.observed_uci.author = Some(author.to_owned());
            } else if line.starts_with("option ") {
                let (name, kind) = parse_option(&line)?;
                if options.len() >= 256 || options.insert(name, kind).is_some() {
                    return Err(error("handshake", "duplicate_or_excess_options"));
                }
            } else if line.starts_with("bestmove") || line == "readyok" {
                return Err(error("handshake", "unexpected_response"));
            }
        }
        if self.observed_uci.name.is_empty()
            || self
                .config
                .expected_uci_name
                .as_ref()
                .is_some_and(|expected| *expected != self.observed_uci.name)
        {
            return Err(error("handshake", "engine_name_mismatch"));
        }
        for (name, value) in self.config.identity.options.clone() {
            let kind = options
                .get(&name)
                .ok_or_else(|| error("handshake", "unsupported_requested_option"))?;
            if !validate_option(kind, &value) {
                return Err(error("handshake", "requested_option_out_of_range"));
            }
            let command = if matches!(kind, OptionKind::Button) {
                format!("setoption name {name}")
            } else {
                format!("setoption name {name} value {value}")
            };
            self.owner()?.send(&command, deadline)?;
        }
        self.ready(deadline, cancel)
    }

    fn ready(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
            return Err(error("readiness", "cancel_or_deadline_before_send"));
        }
        self.owner()?.send("isready", deadline)?;
        loop {
            let line = self
                .owner()?
                .next_line(deadline, cancel)?
                .ok_or_else(|| error("readiness", "timeout_or_cancel"))?;
            if line == "readyok" {
                if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                    return Err(error("readiness", "late_readyok"));
                }
                return Ok(());
            }
            if line.starts_with("bestmove") || line == "uciok" {
                return Err(error("readiness", "late_or_duplicate_response"));
            }
            // Engine diagnostics cannot confer work or analysis evidence here.
        }
    }

    fn run(
        &mut self,
        position: &Position,
        roots: Option<&[BoardMove]>,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.last_attempt = None;
        if self.failed || self.active {
            return Err(error("analysis", "checker_not_available"));
        }
        if limits.max_depth == 0
            || limits.max_depth > self.config.max_depth
            || limits.max_nodes == 0
        {
            return Err(CheckerError::Invalid("external requested depth/nodes"));
        }
        let legal_view = position.ordered_legal_moves();
        if position
            .play_status_from_view(&legal_view)
            .map_err(|_| error("analysis", "rules_classification"))?
            != PlayStatus::Ongoing
        {
            return Err(CheckerError::Unsupported(
                "Rules terminal is resolved without external UCI",
            ));
        }
        if let Some(mask) = roots {
            let legal = legal_view.moves();
            if mask.is_empty()
                || mask.len() > 256
                || mask
                    .iter()
                    .enumerate()
                    .any(|(at, mv)| !legal.contains(mv) || mask[..at].contains(mv))
            {
                return Err(CheckerError::Invalid("external root mask"));
            }
        }
        let start = Instant::now();
        let deadline = bounded_deadline(start, self.config.max_task_wall_time, limits.deadline)?;
        if cancel.load(Ordering::Acquire) || start >= deadline {
            return Err(error("analysis", "cancel_or_deadline_before_go"));
        }
        let replay = position
            .snapshot()
            .uci_replay(self.config.max_prefix_plies)
            .map_err(|_| error("position", "history_replay_refused"))?;
        let mut command = if replay.origin == HistoryOrigin::StartPosition {
            "position startpos".to_owned()
        } else {
            format!("position fen {}", replay.start_fen)
        };
        if !replay.moves.is_empty() {
            command.push_str(" moves");
            for mv in replay.moves {
                command.push(' ');
                command.push_str(&mv.to_string());
            }
        }
        self.request_id = self
            .request_id
            .checked_add(1)
            .ok_or(CheckerError::Invalid("external request id overflow"))?;
        let mut report = ExternalCheckerReport {
            identity: self.config.identity.clone(),
            observed_uci: self.observed_uci.clone(),
            request_id: self.request_id,
            best_move: None,
            pv: Vec::new(),
            score: ExternalRawScore::Unknown,
            bound: ExternalBound::Unknown,
            wdl_per_mille: None,
            perspective: position.side_to_move(),
            requested_depth: limits.max_depth,
            reported_depth: None,
            seldepth: None,
            root_restricted: roots.is_some(),
            completion: ExternalCompletion::Pending,
            work: CheckerWork {
                nodes: None,
                qnodes: None,
                tt_hits: None,
            },
            elapsed: Duration::ZERO,
        };
        self.capture(&report, start);
        let result = self.run_active(
            position,
            roots,
            limits,
            cancel,
            start,
            deadline,
            &command,
            &mut report,
        );
        report.elapsed = start.elapsed();
        self.capture(&report, start);
        if let Err(failure) = result {
            self.failed = true;
            let cleanup = Instant::now()
                .checked_add(self.config.shutdown_grace)
                .unwrap_or_else(Instant::now);
            let _ = self.shutdown(cleanup);
            self.capture(&report, start);
            return Err(failure);
        }
        Ok(CheckerReport::ExternalUci(report))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_active(
        &mut self,
        position: &Position,
        roots: Option<&[BoardMove]>,
        limits: CpuLimits,
        cancel: &AtomicBool,
        start: Instant,
        deadline: Instant,
        position_command: &str,
        report: &mut ExternalCheckerReport,
    ) -> Result<(), CheckerError> {
        self.ready(deadline, cancel)?;
        self.owner()?.send(position_command, deadline)?;
        let remaining_ms = deadline
            .saturating_duration_since(Instant::now())
            .as_millis();
        if remaining_ms == 0 || cancel.load(Ordering::Acquire) {
            return Err(error("analysis", "cancel_or_deadline_before_go"));
        }
        let mut go = format!(
            "go depth {} nodes {} movetime {}",
            limits.max_depth, limits.max_nodes, remaining_ms
        );
        if let Some(mask) = roots {
            go.push_str(" searchmoves");
            for mv in mask {
                go.push(' ');
                go.push_str(&mv.to_string());
            }
        }
        self.owner()?.send(&go, deadline)?;
        self.active = true;
        let no_cancel = AtomicBool::new(false);
        let mut stopped_deadline = None;
        loop {
            let now = Instant::now();
            if stopped_deadline.is_none() && (cancel.load(Ordering::Acquire) || now >= deadline) {
                report.completion = if cancel.load(Ordering::Acquire) {
                    ExternalCompletion::StoppedCanceled
                } else {
                    ExternalCompletion::StoppedDeadline
                };
                let stop_deadline = now
                    .checked_add(self.config.stop_grace)
                    .ok_or(CheckerError::Invalid("external stop overflow"))?;
                self.owner()?.send("stop", stop_deadline)?;
                self.owner()?.mark_stop();
                stopped_deadline = Some(stop_deadline);
            }
            let wait_deadline = stopped_deadline.unwrap_or(deadline);
            let token = if stopped_deadline.is_some() {
                &no_cancel
            } else {
                cancel
            };
            let Some(line) = self.owner()?.next_line(wait_deadline, token)? else {
                if stopped_deadline.is_some() {
                    return Err(error("stop", "bestmove_not_drained"));
                }
                continue;
            };
            if line.starts_with("info ") {
                let frame = parse_info(&line, 128)?;
                if let Some(nodes) = frame.nodes {
                    if report.work.nodes.is_some_and(|old| nodes < old) {
                        return Err(error("analysis", "nodes_regressed"));
                    }
                    report.work.nodes = Some(nodes);
                    self.capture(report, start);
                    if nodes > limits.max_nodes {
                        return Err(error("analysis", "reported_node_budget_exceeded"));
                    }
                }
                if frame.multipv != 1 {
                    continue;
                }
                if let Some(pv) = &frame.pv {
                    validate_pv(position, pv, roots)?;
                }
                if let (Some(score), Some(bound), Some(depth), Some(pv)) =
                    (frame.score, frame.bound, frame.depth, frame.pv)
                {
                    report.score = score;
                    report.bound = bound;
                    report.reported_depth = Some(depth);
                    report.seldepth = frame.seldepth;
                    report.pv = pv;
                    report.wdl_per_mille = frame.wdl;
                    self.capture(report, start);
                }
            } else if let Some(body) = line.strip_prefix("bestmove ") {
                let parts: Vec<&str> = body.split_ascii_whitespace().collect();
                if parts.len() != 1 && (parts.len() != 3 || parts[1] != "ponder") {
                    return Err(error("analysis", "malformed_bestmove"));
                }
                let best = BoardMove::from_uci(parts[0])
                    .map_err(|_| error("analysis", "invalid_bestmove"))?;
                validate_pv(position, &[best], roots)?;
                if report.pv.first() != Some(&best) {
                    // Preserve the original tuple only in prior evidence; a
                    // bestmove mismatch cannot borrow the last line's score.
                    report.pv = vec![best];
                    report.score = ExternalRawScore::Unknown;
                    report.bound = ExternalBound::Unknown;
                    report.reported_depth = None;
                    report.seldepth = None;
                    report.wdl_per_mille = None;
                }
                report.best_move = Some(best);
                if matches!(report.completion, ExternalCompletion::Pending) {
                    report.completion = ExternalCompletion::BestMove;
                }
                self.active = false;
                self.ready(wait_deadline, &no_cancel)?;
                // Final cancellation/deadline remains a partial external
                // observation even when bestmove races with the stop boundary.
                if matches!(report.completion, ExternalCompletion::BestMove) {
                    if cancel.load(Ordering::Acquire) {
                        report.completion = ExternalCompletion::StoppedCanceled;
                    } else if Instant::now() >= deadline {
                        report.completion = ExternalCompletion::StoppedDeadline;
                    }
                }
                return Ok(());
            } else if matches!(line.as_str(), "uciok" | "readyok") {
                return Err(error("analysis", "unexpected_barrier_response"));
            }
        }
    }

    fn capture(&mut self, report: &ExternalCheckerReport, start: Instant) {
        let process = self
            .owner
            .as_ref()
            .map_or_else(empty_shutdown, |owner| owner.evidence());
        self.last_attempt = Some(CheckerAttempt {
            work: report.work,
            elapsed: start.elapsed(),
            external: Some(ExternalAttemptEvidence {
                request_id: report.request_id,
                partial_report: Some(report.clone()),
                process,
            }),
        });
    }
}

fn empty_shutdown() -> CheckerShutdown {
    CheckerShutdown {
        stop_sent: false,
        quit_sent: false,
        exit_observed: false,
        exit_code: None,
        exit_signal: None,
        ownership_lost: false,
        stdout_drained: false,
        stderr_drained: false,
        cleanup_complete: false,
        quarantined: false,
        stdout_bytes: 0,
        stderr_bytes: 0,
    }
}

impl CpuChecker for ExternalUciCpuChecker {
    fn identity(&self) -> &CheckerIdentity {
        &self.identity
    }
    fn conditions(&self) -> &str {
        &self.conditions
    }
    fn capabilities(&self) -> CheckerCapabilities {
        CheckerCapabilities {
            max_depth: self.config.max_depth,
            max_prefix_plies: self.config.max_prefix_plies,
            max_root_moves: 256,
            root_moves: true,
            divergence: true,
            resume: false,
            selective_search: None,
        }
    }
    fn last_attempt(&self) -> Option<&CheckerAttempt> {
        self.last_attempt.as_ref()
    }
    fn reset_attempt(&mut self) {
        self.last_attempt = None;
    }
    fn analyze(
        &mut self,
        p: &Position,
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.run(p, None, l, c)
    }
    fn analyze_root_moves(
        &mut self,
        p: &Position,
        m: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.run(p, Some(m), l, c)
    }
    fn analyze_divergence(
        &mut self,
        p: &Position,
        prefix: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.last_attempt = None;
        if prefix.len() > self.config.max_prefix_plies {
            return Err(CheckerError::Invalid("external divergence prefix cap"));
        }
        let mut target = p.clone();
        for &mv in prefix {
            target
                .make_move(mv)
                .map_err(|_| error("divergence", "illegal_prefix"))?;
        }
        self.run(&target, None, l, c)
    }
    fn resume(
        &mut self,
        _p: &Position,
        _t: &CpuResumeToken,
        _l: CpuLimits,
        _c: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.last_attempt = None;
        Err(CheckerError::Unsupported(
            "external UCI internal stack resume",
        ))
    }
    fn new_game(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        if self.failed || self.active {
            return Err(error("new_game", "checker_not_available"));
        }
        if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
            return Err(error("new_game", "cancel_or_deadline_before_reset"));
        }
        self.owner()?.send("ucinewgame", deadline)?;
        if let Err(failure) = self.ready(deadline, cancel) {
            self.failed = true;
            return Err(failure);
        }
        self.last_attempt = None;
        Ok(())
    }
    fn shutdown(&mut self, deadline: Instant) -> Result<CheckerShutdown, CheckerError> {
        let Some(owner) = self.owner.as_mut() else {
            return Err(error("shutdown", "owner_closed"));
        };
        let result = owner.shutdown(deadline, self.active);
        self.active = false;
        if let Some(attempt) = self
            .last_attempt
            .as_mut()
            .and_then(|attempt| attempt.external.as_mut())
        {
            attempt.process = owner.evidence();
        }
        result
    }
}

#[cfg(target_os = "linux")]
mod process {
    use super::*;
    use nix::errno::Errno;
    use nix::fcntl::{FcntlArg, OFlag, fcntl};
    use nix::sys::signal::{Signal, killpg};
    use nix::sys::wait::{Id, WaitPidFlag, WaitStatus, waitid};
    use nix::unistd::Pid;
    use std::collections::VecDeque;
    use std::io::{ErrorKind, Write};
    use std::os::fd::{AsFd, AsRawFd};
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    use std::os::unix::process::CommandExt;
    use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};

    // A quarantined owner keeps this slot occupied. Repeated failed creation
    // cannot accumulate arbitrarily many child groups or retained handles.
    static PROCESS_SLOT: AtomicBool = AtomicBool::new(false);
    static QUARANTINE_PRESENT: AtomicBool = AtomicBool::new(false);

    pub(super) struct OwnedProcess {
        child: Option<Child>,
        stdin: Option<ChildStdin>,
        stdout: Option<ChildStdout>,
        stderr: Option<ChildStderr>,
        stdin_nonblocking: bool,
        stdout_nonblocking: bool,
        stderr_nonblocking: bool,
        _program: File,
        _directory: File,
        pid: Pid,
        start_time: u64,
        authority: bool,
        max_output: usize,
        max_line: usize,
        stdout_log: Vec<u8>,
        stderr_log: Vec<u8>,
        pending_line: Vec<u8>,
        lines: VecDeque<String>,
        state: CheckerShutdown,
        shutdown_grace: Duration,
    }

    pub(super) struct SpawnFailure {
        pub(super) error: CheckerError,
        pub(super) process: CheckerShutdown,
        pub(super) stdout: Vec<u8>,
        pub(super) stderr: Vec<u8>,
    }

    fn nonblocking<F: AsFd>(fd: &F) -> Result<(), CheckerError> {
        let flags = fcntl(fd, FcntlArg::F_GETFL).map_err(|_| error("spawn", "pipe_flags"))?;
        fcntl(
            fd,
            FcntlArg::F_SETFL(OFlag::from_bits_truncate(flags) | OFlag::O_NONBLOCK),
        )
        .map_err(|_| error("spawn", "pipe_nonblocking"))?;
        Ok(())
    }

    #[derive(Clone, Copy)]
    struct ProcIdentity {
        group: i32,
        start: u64,
    }

    fn proc_identity(pid: i32) -> Result<Option<ProcIdentity>, CheckerError> {
        let file = match File::open(format!("/proc/{pid}/stat")) {
            Ok(file) => file,
            Err(failure) if failure.kind() == ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err(error("ownership", "proc_stat_unavailable")),
        };
        let mut bytes = Vec::new();
        file.take(4097)
            .read_to_end(&mut bytes)
            .map_err(|_| error("ownership", "proc_stat_read"))?;
        if bytes.len() > 4096 {
            return Err(error("ownership", "proc_stat_budget"));
        }
        let text =
            std::str::from_utf8(&bytes).map_err(|_| error("ownership", "proc_stat_encoding"))?;
        let body = text
            .rsplit_once(") ")
            .map(|(_, body)| body)
            .ok_or_else(|| error("ownership", "proc_stat_format"))?;
        let fields: Vec<&str> = body.split_ascii_whitespace().collect();
        let group = fields
            .get(2)
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| error("ownership", "proc_group"))?;
        let start = fields
            .get(19)
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| error("ownership", "proc_start"))?;
        Ok(Some(ProcIdentity { group, start }))
    }

    fn other_group_members(pid: Pid) -> Result<usize, CheckerError> {
        let mut scanned = 0usize;
        let mut count = 0usize;
        for entry in std::fs::read_dir("/proc").map_err(|_| error("cleanup", "proc_group_scan"))? {
            let entry = entry.map_err(|_| error("cleanup", "proc_entry"))?;
            let Some(id) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<i32>().ok())
            else {
                continue;
            };
            scanned += 1;
            if scanned > 65_536 {
                return Err(error("cleanup", "proc_scan_budget"));
            }
            if id != pid.as_raw()
                && proc_identity(id)?.is_some_and(|identity| identity.group == pid.as_raw())
            {
                count += 1;
            }
        }
        Ok(count)
    }

    impl OwnedProcess {
        pub(super) fn spawn(
            config: &ExternalCpuConfig,
            deadline: Instant,
            cancel: &AtomicBool,
        ) -> Result<Self, Box<SpawnFailure>> {
            if PROCESS_SLOT
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return Err(Box::new(SpawnFailure {
                    error: error("spawn", "process_slot_unavailable"),
                    process: empty_shutdown(),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                }));
            }
            let mut failure_evidence = None;
            let result = Self::spawn_owned(config, deadline, cancel, &mut failure_evidence);
            if result.is_err() && !QUARANTINE_PRESENT.load(Ordering::Acquire) {
                PROCESS_SLOT.store(false, Ordering::Release);
            }
            result.map_err(|failure| {
                Box::new(failure_evidence.unwrap_or_else(|| {
                    let mut process = empty_shutdown();
                    process.cleanup_complete = true;
                    SpawnFailure {
                        error: failure,
                        process,
                        stdout: Vec::new(),
                        stderr: Vec::new(),
                    }
                }))
            })
        }
        fn spawn_owned(
            config: &ExternalCpuConfig,
            deadline: Instant,
            cancel: &AtomicBool,
            failure_evidence: &mut Option<SpawnFailure>,
        ) -> Result<Self, CheckerError> {
            let program = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(nix::libc::O_NOFOLLOW)
                .open(&config.program)
                .map_err(|_| error("spawn", "program_open"))?;
            let metadata = program
                .metadata()
                .map_err(|_| error("spawn", "program_metadata"))?;
            if !metadata.is_file()
                || metadata.len() > 256 * 1024 * 1024
                || metadata.mode() & 0o111 == 0
            {
                return Err(error("spawn", "program_kind_or_size"));
            }
            let mut reader = &program;
            let mut header = [0u8; 4];
            reader
                .read_exact(&mut header)
                .map_err(|_| error("spawn", "program_header"))?;
            if header != *b"\x7fELF" {
                return Err(error("spawn", "native_elf_required"));
            }
            let mut hash = Sha256::new();
            hash.update(header);
            let mut buffer = [0u8; 65_536];
            let mut total = 4u64;
            loop {
                if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                    return Err(error("spawn", "hash_cancel_or_deadline"));
                }
                let count = reader
                    .read(&mut buffer)
                    .map_err(|_| error("spawn", "program_hash_read"))?;
                if count == 0 {
                    break;
                }
                total = total
                    .checked_add(count as u64)
                    .ok_or_else(|| error("spawn", "program_hash_overflow"))?;
                if total > 256 * 1024 * 1024 {
                    return Err(error("spawn", "program_hash_budget"));
                }
                hash.update(&buffer[..count]);
            }
            let after = program
                .metadata()
                .map_err(|_| error("spawn", "program_recheck"))?;
            if total != metadata.len()
                || after.len() != metadata.len()
                || after.dev() != metadata.dev()
                || after.ino() != metadata.ino()
                || after.mtime() != metadata.mtime()
                || after.mtime_nsec() != metadata.mtime_nsec()
                || after.ctime() != metadata.ctime()
                || after.ctime_nsec() != metadata.ctime_nsec()
            {
                return Err(error("spawn", "program_changed_during_hash"));
            }
            if format!("{:x}", hash.finalize()) != config.identity.binary_sha256 {
                return Err(error("spawn", "program_hash_mismatch"));
            }
            let directory = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(nix::libc::O_DIRECTORY | nix::libc::O_NOFOLLOW)
                .open(&config.working_directory)
                .map_err(|_| error("spawn", "directory_open"))?;
            let mut command = Command::new(format!("/proc/self/fd/{}", program.as_raw_fd()));
            if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                return Err(error("spawn", "cancel_or_deadline_before_spawn"));
            }
            command
                .args(&config.arguments)
                .current_dir(format!("/proc/self/fd/{}", directory.as_raw_fd()))
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("LC_ALL", "C")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .process_group(0);
            let mut child = command
                .spawn()
                .map_err(|_| error("spawn", "native_spawn"))?;
            // Linux kernel PIDs fit signed pid_t. Install the owned child before
            // any fallible setup and retain optional pipes even on setup error.
            let pid = Pid::from_raw(child.id() as i32);
            let stdin = child.stdin.take();
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            // Child ownership is installed before any fallible pipe/proc setup.
            // An initialization failure receives the same bounded cleanup.
            let mut owner = Self {
                child: Some(child),
                stdin,
                stdout,
                stderr,
                stdin_nonblocking: false,
                stdout_nonblocking: false,
                stderr_nonblocking: false,
                _program: program,
                _directory: directory,
                pid,
                start_time: 0,
                authority: true,
                max_output: config.max_output_bytes,
                max_line: config.max_line_bytes,
                stdout_log: Vec::new(),
                stderr_log: Vec::new(),
                pending_line: Vec::new(),
                lines: VecDeque::new(),
                state: empty_shutdown(),
                shutdown_grace: config.shutdown_grace,
            };
            let setup = (|| {
                let identity = proc_identity(pid.as_raw())?
                    .ok_or_else(|| error("spawn", "leader_identity_missing"))?;
                if identity.group != pid.as_raw() {
                    return Err(error("spawn", "leader_group_mismatch"));
                }
                owner.start_time = identity.start;
                nonblocking(
                    owner
                        .stdin
                        .as_ref()
                        .ok_or_else(|| error("spawn", "stdin_missing"))?,
                )?;
                owner.stdin_nonblocking = true;
                nonblocking(
                    owner
                        .stdout
                        .as_ref()
                        .ok_or_else(|| error("spawn", "stdout_missing"))?,
                )?;
                owner.stdout_nonblocking = true;
                nonblocking(
                    owner
                        .stderr
                        .as_ref()
                        .ok_or_else(|| error("spawn", "stderr_missing"))?,
                )?;
                owner.stderr_nonblocking = true;
                Ok::<(), CheckerError>(())
            })();
            if let Err(failure) = setup {
                let deadline = Instant::now()
                    .checked_add(owner.shutdown_grace)
                    .unwrap_or_else(Instant::now);
                let _ = owner.shutdown(deadline, false);
                *failure_evidence = Some(SpawnFailure {
                    error: failure.clone(),
                    process: owner.evidence(),
                    stdout: owner.stdout_log.clone(),
                    stderr: owner.stderr_log.clone(),
                });
                return Err(failure);
            }
            Ok(owner)
        }
        fn status(&mut self) -> Result<WaitStatus, CheckerError> {
            if !self.authority {
                return Err(error("ownership", "authority_lost"));
            }
            match waitid(
                Id::Pid(self.pid),
                WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT,
            ) {
                Ok(status) => Ok(status),
                Err(Errno::ECHILD) => {
                    self.authority = false;
                    self.state.ownership_lost = true;
                    self.state.quarantined = true;
                    Err(error("ownership", "foreign_reaper_or_no_child"))
                }
                Err(_) => Err(error("ownership", "waitid_failed")),
            }
        }
        fn verify_signal_authority(&mut self) -> Result<(), CheckerError> {
            self.status()?;
            let identity = proc_identity(self.pid.as_raw())?
                .ok_or_else(|| error("ownership", "leader_disappeared"))?;
            if identity.start != self.start_time || identity.group != self.pid.as_raw() {
                self.authority = false;
                self.state.ownership_lost = true;
                self.state.quarantined = true;
                return Err(error("ownership", "leader_identity_changed"));
            }
            Ok(())
        }
        fn signal(&mut self, signal: Signal) -> Result<(), CheckerError> {
            self.verify_signal_authority()?;
            killpg(self.pid, signal).map_err(|_| error("cleanup", "group_signal_failed"))
        }
        pub(super) fn send(&mut self, line: &str, deadline: Instant) -> Result<(), CheckerError> {
            if !self.stdin_nonblocking {
                return Err(error("stdin", "nonblocking_not_ready"));
            }
            if line.len() > 65_536 || line.contains(['\r', '\n', '\0']) {
                return Err(error("stdin", "command_budget_or_injection"));
            }
            let mut bytes = line.as_bytes().to_vec();
            bytes.push(b'\n');
            let mut written = 0;
            while written < bytes.len() {
                if Instant::now() >= deadline {
                    return Err(error("stdin", "write_deadline"));
                }
                match self
                    .stdin
                    .as_mut()
                    .ok_or_else(|| error("stdin", "closed"))?
                    .write(&bytes[written..])
                {
                    Ok(0) => return Err(error("stdin", "zero_write")),
                    Ok(count) => written += count,
                    Err(failure) if failure.kind() == ErrorKind::WouldBlock => {
                        self.pump(false)?;
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(failure) if failure.kind() == ErrorKind::Interrupted => {}
                    Err(_) => return Err(error("stdin", "write_failure")),
                }
            }
            Ok(())
        }
        fn pump(&mut self, discard: bool) -> Result<(), CheckerError> {
            let mut buffer = [0u8; 4096];
            // Fair finite reads; a streaming stdout cannot starve stderr/clock.
            for stdout in [true, false] {
                if stdout && !self.stdout_nonblocking || !stdout && !self.stderr_nonblocking {
                    continue;
                }
                for _ in 0..16 {
                    let result = if stdout {
                        self.stdout
                            .as_mut()
                            .ok_or_else(|| error("stdout", "pipe_missing"))?
                            .read(&mut buffer)
                    } else {
                        self.stderr
                            .as_mut()
                            .ok_or_else(|| error("stderr", "pipe_missing"))?
                            .read(&mut buffer)
                    };
                    match result {
                        Ok(0) => {
                            if stdout {
                                self.state.stdout_drained = true;
                            } else {
                                self.state.stderr_drained = true;
                            }
                            break;
                        }
                        Ok(count) => {
                            if stdout {
                                self.state.stdout_bytes =
                                    self.state.stdout_bytes.saturating_add(count as u64);
                            } else {
                                self.state.stderr_bytes =
                                    self.state.stderr_bytes.saturating_add(count as u64);
                            }
                            if discard {
                                continue;
                            }
                            if self
                                .stdout_log
                                .len()
                                .checked_add(self.stderr_log.len())
                                .and_then(|bytes| bytes.checked_add(count))
                                .is_none_or(|bytes| bytes > self.max_output)
                            {
                                return Err(error("output", "combined_output_budget"));
                            }
                            if stdout {
                                self.stdout_log.extend_from_slice(&buffer[..count]);
                                for &byte in &buffer[..count] {
                                    if byte == b'\n' {
                                        if self.pending_line.last() == Some(&b'\r') {
                                            self.pending_line.pop();
                                        }
                                        let text = std::str::from_utf8(&self.pending_line)
                                            .map_err(|_| error("stdout", "invalid_utf8"))?
                                            .to_owned();
                                        self.pending_line.clear();
                                        if self.lines.len() >= 256 {
                                            return Err(error("stdout", "pending_line_budget"));
                                        }
                                        if text.chars().any(|c| c.is_control() && c != '\t') {
                                            return Err(error("stdout", "control_character"));
                                        }
                                        self.lines.push_back(text);
                                    } else {
                                        if self.pending_line.len() >= self.max_line {
                                            return Err(error("stdout", "line_budget"));
                                        }
                                        self.pending_line.push(byte);
                                    }
                                }
                            } else {
                                self.stderr_log.extend_from_slice(&buffer[..count]);
                            }
                        }
                        Err(failure) if failure.kind() == ErrorKind::WouldBlock => break,
                        Err(failure) if failure.kind() == ErrorKind::Interrupted => continue,
                        Err(_) => return Err(error("output", "pipe_read")),
                    }
                }
            }
            Ok(())
        }
        pub(super) fn next_line(
            &mut self,
            deadline: Instant,
            cancel: &AtomicBool,
        ) -> Result<Option<String>, CheckerError> {
            loop {
                if let Some(line) = self.lines.pop_front() {
                    return Ok(Some(line));
                }
                if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                    return Ok(None);
                }
                self.pump(false)?;
                if let Some(line) = self.lines.pop_front() {
                    return Ok(Some(line));
                }
                match self.status()? {
                    WaitStatus::StillAlive => {}
                    _ => return Err(error("process", "unexpected_exit")),
                }
                if self.state.stdout_drained {
                    return Err(error("stdout", "eof_before_response"));
                }
                if other_group_members(self.pid)? > 0 {
                    return Err(error("process", "child_process_limit"));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        pub(super) fn mark_stop(&mut self) {
            self.state.stop_sent = true;
        }
        pub(super) fn output(&self) -> (&[u8], &[u8]) {
            (&self.stdout_log, &self.stderr_log)
        }
        pub(super) fn evidence(&self) -> CheckerShutdown {
            self.state
        }
        pub(super) fn shutdown(
            &mut self,
            deadline: Instant,
            active: bool,
        ) -> Result<CheckerShutdown, CheckerError> {
            let result = self.shutdown_inner(deadline, active);
            if result.is_err() {
                self.state.quarantined = true;
            }
            result
        }
        fn shutdown_inner(
            &mut self,
            deadline: Instant,
            active: bool,
        ) -> Result<CheckerShutdown, CheckerError> {
            if self.state.cleanup_complete {
                return Ok(self.state);
            }
            if !self.authority {
                self.state.quarantined = true;
                return Err(error("shutdown", "ownership_lost"));
            }
            if active && self.send("stop", deadline).is_ok() {
                self.state.stop_sent = true;
            }
            if self.send("quit", deadline).is_ok() {
                self.state.quit_sent = true;
            }
            self.stdin.take();
            let begin = Instant::now();
            let mut killed = false;
            let kill_at = begin
                .checked_add(deadline.saturating_duration_since(begin) / 2)
                .unwrap_or(begin);
            while Instant::now() < deadline {
                // Drain without retaining more output after any failure. Byte
                // observations remain actual reads; discarded content is not a
                // fabricated full transcript or a claim about unread bytes.
                let _ = self.pump(true);
                let status = self.status()?;
                let others = other_group_members(self.pid)?;
                if !matches!(status, WaitStatus::StillAlive) {
                    self.state.exit_observed = true;
                    match status {
                        WaitStatus::Exited(_, code) => self.state.exit_code = Some(code),
                        WaitStatus::Signaled(_, signal, _) => {
                            self.state.exit_signal = Some(signal as i32)
                        }
                        _ => {}
                    }
                    if others == 0 && self.state.stdout_drained && self.state.stderr_drained {
                        // The unreaped leader reserves PID/PGID until this final
                        // check. No numeric group operation occurs after reap.
                        self.child
                            .as_mut()
                            .ok_or_else(|| error("shutdown", "missing_child"))?
                            .wait()
                            .map_err(|_| error("shutdown", "leader_reap"))?;
                        self.child.take();
                        self.authority = false;
                        self.state.cleanup_complete = true;
                        PROCESS_SLOT.store(false, Ordering::Release);
                        return Ok(self.state);
                    }
                }
                if !killed && (others > 0 || Instant::now() >= kill_at) {
                    self.signal(Signal::SIGKILL)?;
                    killed = true;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            self.state.quarantined = true;
            Err(error("shutdown", "cleanup_unconfirmed"))
        }
    }
    impl Drop for OwnedProcess {
        fn drop(&mut self) {
            if self.state.cleanup_complete {
                return;
            }
            let deadline = Instant::now()
                .checked_add(self.shutdown_grace)
                .unwrap_or_else(Instant::now);
            if !self.state.quarantined {
                let _ = self.shutdown(deadline, true);
            }
            if !self.state.cleanup_complete {
                // Preserve an unreaped handle; a failed owner cannot create a
                // replacement. No kill/reap/drop follows lost PID authority.
                if let Some(child) = self.child.take() {
                    std::mem::forget(child);
                }
                QUARANTINE_PRESENT.store(true, Ordering::Release);
                PROCESS_SLOT.store(true, Ordering::Release);
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod process {
    use super::*;
    pub(super) struct OwnedProcess;
    pub(super) struct SpawnFailure {
        pub(super) error: CheckerError,
        pub(super) process: CheckerShutdown,
        pub(super) stdout: Vec<u8>,
        pub(super) stderr: Vec<u8>,
    }
    impl OwnedProcess {
        pub(super) fn spawn(
            _config: &ExternalCpuConfig,
            _deadline: Instant,
            _cancel: &AtomicBool,
        ) -> Result<Self, Box<SpawnFailure>> {
            Err(Box::new(SpawnFailure {
                error: CheckerError::Unsupported("external UCI checker requires Linux/WSL"),
                process: empty_shutdown(),
                stdout: Vec::new(),
                stderr: Vec::new(),
            }))
        }
        pub(super) fn send(&mut self, _line: &str, _deadline: Instant) -> Result<(), CheckerError> {
            Err(CheckerError::Unsupported("external UCI Linux process"))
        }
        pub(super) fn next_line(
            &mut self,
            _deadline: Instant,
            _cancel: &AtomicBool,
        ) -> Result<Option<String>, CheckerError> {
            Err(CheckerError::Unsupported("external UCI Linux process"))
        }
        pub(super) fn mark_stop(&mut self) {}
        pub(super) fn output(&self) -> (&[u8], &[u8]) {
            (&[], &[])
        }
        pub(super) fn evidence(&self) -> CheckerShutdown {
            empty_shutdown()
        }
        pub(super) fn shutdown(
            &mut self,
            _deadline: Instant,
            _active: bool,
        ) -> Result<CheckerShutdown, CheckerError> {
            Err(CheckerError::Unsupported("external UCI Linux process"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_scores_remain_native_units_bounds_and_unknown_work() {
        let cp = parse_info("info depth 8 seldepth 14 nodes 432 score cp -731 upperbound wdl 10 200 790 pv e2e4 e7e5", 16).unwrap();
        assert_eq!(cp.score, Some(ExternalRawScore::Centipawns(-731)));
        assert_eq!(cp.bound, Some(ExternalBound::Upper));
        assert_eq!(cp.wdl, Some([10, 200, 790]));
        assert_eq!(cp.nodes, Some(432));
        let mate = parse_info("info depth 10 score mate -3 lowerbound pv e2e4", 16).unwrap();
        assert_eq!(mate.score, Some(ExternalRawScore::MateMoves(-3)));
        assert_eq!(mate.nodes, None);
        assert_eq!(mate.bound, Some(ExternalBound::Lower));
        let sparse = parse_info("info nodes 0 string score cp 99 pv e2e4", 16).unwrap();
        assert_eq!(sparse.nodes, Some(0));
        assert_eq!(sparse.score, None);
        assert!(parse_info("info score mate NaN pv e2e4", 16).is_err());
        assert!(parse_info("info wdl 1 2 3 pv e2e4", 16).is_err());
    }
    #[test]
    fn root_mask_and_rules_pv_are_independently_checked() {
        let p = Position::startpos();
        let e4 = BoardMove::from_uci("e2e4").unwrap();
        let d4 = BoardMove::from_uci("d2d4").unwrap();
        validate_pv(&p, &[e4], Some(&[e4])).unwrap();
        assert!(validate_pv(&p, &[d4], Some(&[e4])).is_err());
        assert!(validate_pv(&p, &[e4, d4], None).is_err());
    }
    #[test]
    fn option_ranges_and_argument_boundaries_are_not_inferred() {
        let (_, kind) =
            parse_option("option name Threads type spin default 1 min 1 max 1024").unwrap();
        assert!(validate_option(&kind, "2"));
        assert!(!validate_option(&kind, "0"));
        assert!(!validate_option(&kind, "2\nquit"));
        let (_, combo) =
            parse_option("option name NumaPolicy type combo default auto var auto var none")
                .unwrap();
        assert!(validate_option(&combo, "none"));
        assert!(!validate_option(&combo, "unknown"));
        assert_ne!(
            arguments_sha256(&["ab".into(), "c".into()]),
            arguments_sha256(&["a".into(), "bc".into()])
        );
    }

    #[cfg(target_os = "linux")]
    static FIXTURE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(target_os = "linux")]
    fn fixture_mode(name: &str) -> bool {
        let args: Vec<_> = std::env::args().collect();
        args.iter().any(|arg| arg == "--exact") && args.iter().any(|arg| arg == name)
    }

    #[cfg(target_os = "linux")]
    fn fixture_loop(mode: u8) {
        use std::io::{BufRead, Write};
        let mut position = Position::startpos();
        let mut readiness_count = 0usize;
        for line in std::io::stdin().lock().lines() {
            let line = line.expect("fixture stdin");
            if line == "uci" {
                println!("id name RZ external checker fixture");
                println!("id author RoveZero tests");
                println!("option name Threads type spin default 1 min 1 max 2");
                println!("uciok");
            } else if line == "isready" {
                readiness_count += 1;
                if mode == 3 && readiness_count > 1 {
                    println!("uciok");
                } else {
                    println!("readyok");
                }
            } else if let Some(body) = line.strip_prefix("position startpos") {
                position = Position::startpos();
                if let Some(moves) = body.trim().strip_prefix("moves ") {
                    for mv in moves.split_ascii_whitespace() {
                        position
                            .make_move(BoardMove::from_uci(mv).expect("fixture move"))
                            .expect("fixture legal replay");
                    }
                }
            } else if line.starts_with("go ") {
                let words: Vec<&str> = line.split_ascii_whitespace().collect();
                let depth = words
                    .iter()
                    .position(|word| *word == "depth")
                    .map(|at| words[at + 1])
                    .unwrap();
                let requested: u64 = words
                    .iter()
                    .position(|word| *word == "nodes")
                    .map(|at| words[at + 1].parse().unwrap())
                    .unwrap();
                let best = words
                    .iter()
                    .position(|word| *word == "searchmoves")
                    .map(|at| BoardMove::from_uci(words[at + 1]).unwrap())
                    .unwrap_or_else(|| position.legal_moves()[0]);
                let nodes = if mode == 1 {
                    requested + 1
                } else {
                    requested.min(10)
                };
                if mode == 2 {
                    println!("info depth {depth} nodes {nodes} score cp NaN pv {best}");
                } else {
                    println!("info depth {depth} seldepth 12 nodes {nodes} score cp 21 pv {best}");
                }
                println!("bestmove {best}");
            } else if line == "quit" {
                break;
            }
            std::io::stdout().flush().unwrap();
        }
    }

    // These are native test-binary subprocess modes, not shell/Python engines.
    // Ordinary all-target test execution returns immediately in these tests.
    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_normal() {
        if fixture_mode("external_cpu::tests::fake_uci_normal") {
            fixture_loop(0);
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_overshoot() {
        if fixture_mode("external_cpu::tests::fake_uci_overshoot") {
            fixture_loop(1);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_parse_failure() {
        if fixture_mode("external_cpu::tests::fake_uci_parse_failure") {
            fixture_loop(2);
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_pre_go_failure() {
        if fixture_mode("external_cpu::tests::fake_uci_pre_go_failure") {
            fixture_loop(3);
        }
    }

    #[cfg(target_os = "linux")]
    fn fixture_config(mode: &str) -> ExternalCpuConfig {
        let program = std::env::current_exe().unwrap();
        let arguments = vec![
            "--exact".into(),
            format!("external_cpu::tests::{mode}").into(),
            "--nocapture".into(),
            "--quiet".into(),
            "--test-threads=1".into(),
        ];
        let mut file = File::open(&program).unwrap();
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65_536];
        loop {
            let count = file.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        let identity = ExternalCheckerIdentity {
            adapter_semantics: EXTERNAL_UCI_SCORE_SEMANTICS.to_owned(),
            binary_sha256: format!("{:x}", hash.finalize()),
            launch_arguments_sha256: arguments_sha256(&arguments),
            declared_name: "RZ external checker fixture".to_owned(),
            declared_version: "native-test/1".to_owned(),
            declared_source: "RoveZero MIT fixture".to_owned(),
            declared_license: "MIT".to_owned(),
            options: BTreeMap::from([("Threads".to_owned(), "2".to_owned())]),
            assets: Vec::new(),
            model_metadata: ExternalModelMetadata {
                weights_sha256: None,
                training: ExternalTrainingKnowledge::Unknown,
                declared_rights: None,
                precision: None,
            },
        };
        ExternalCpuConfig {
            working_directory: program.parent().unwrap().to_owned(),
            program,
            arguments,
            identity,
            expected_uci_name: Some("RZ external checker fixture".to_owned()),
            max_depth: 16,
            max_prefix_plies: 4096,
            handshake_timeout: Duration::from_secs(10),
            max_task_wall_time: Duration::from_secs(2),
            stop_grace: Duration::from_secs(1),
            shutdown_grace: Duration::from_secs(2),
            max_output_bytes: 65_536,
            max_line_bytes: 4096,
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_fixture_root_divergence_and_reset_are_external_reports() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let cancel = AtomicBool::new(false);
        let mut checker = ExternalUciCpuChecker::create(fixture_config("fake_uci_normal")).unwrap();
        checker
            .start(Instant::now() + Duration::from_secs(10), &cancel)
            .unwrap();
        let limits = CpuLimits {
            max_depth: 8,
            max_nodes: 80,
            deadline: Some(Instant::now() + Duration::from_secs(2)),
        };
        let d4 = BoardMove::from_uci("d2d4").unwrap();
        let CheckerReport::ExternalUci(report) = checker
            .analyze_root_moves(&Position::startpos(), &[d4], limits, &cancel)
            .unwrap()
        else {
            panic!("foreign converted to own");
        };
        assert_eq!(report.best_move, Some(d4));
        assert_eq!(report.score, ExternalRawScore::Centipawns(21));
        assert_eq!(report.reported_depth, Some(8));
        assert_eq!(report.work.nodes, Some(10));
        assert_eq!(report.work.qnodes, None);
        assert_eq!(report.work.tt_hits, None);
        let before = checker.diagnostic_output().0.len();
        assert!(
            checker
                .new_game(
                    Instant::now() + Duration::from_secs(1),
                    &AtomicBool::new(true)
                )
                .is_err()
        );
        assert_eq!(checker.diagnostic_output().0.len(), before);
        checker
            .new_game(Instant::now() + Duration::from_secs(1), &cancel)
            .unwrap();
        let CheckerReport::ExternalUci(branch) = checker
            .analyze_divergence(
                &Position::startpos(),
                &[BoardMove::from_uci("e2e4").unwrap()],
                CpuLimits {
                    deadline: Some(Instant::now() + Duration::from_secs(2)),
                    ..limits
                },
                &cancel,
            )
            .unwrap()
        else {
            panic!("foreign converted to own");
        };
        assert_eq!(branch.perspective, rz_position::Color::Black);
        let process = checker
            .shutdown(Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert!(
            process.cleanup_complete
                && process.exit_observed
                && process.stdout_drained
                && process.stderr_drained
        );
        assert!(!process.ownership_lost && !process.quarantined);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn node_overshoot_fails_and_preserves_actual_capture() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let cancel = AtomicBool::new(false);
        let mut checker =
            ExternalUciCpuChecker::create(fixture_config("fake_uci_overshoot")).unwrap();
        checker
            .start(Instant::now() + Duration::from_secs(10), &cancel)
            .unwrap();
        let result = checker.analyze(
            &Position::startpos(),
            CpuLimits {
                max_depth: 8,
                max_nodes: 80,
                deadline: Some(Instant::now() + Duration::from_secs(2)),
            },
            &cancel,
        );
        assert!(matches!(
            result,
            Err(CheckerError::External {
                code: "reported_node_budget_exceeded",
                ..
            })
        ));
        let attempt = checker.last_attempt().unwrap();
        assert_eq!(attempt.work.nodes, Some(81));
        assert_eq!(attempt.work.qnodes, None);
        assert!(attempt.external.as_ref().unwrap().process.cleanup_complete);
        let partial = attempt
            .external
            .as_ref()
            .unwrap()
            .partial_report
            .as_ref()
            .unwrap();
        assert_eq!(partial.completion, ExternalCompletion::Pending);
        assert_eq!(partial.best_move, None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_and_pre_go_failures_do_not_invent_bestmove_completion() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        for (mode, expected) in [
            ("fake_uci_parse_failure", "invalid_score"),
            ("fake_uci_pre_go_failure", "late_or_duplicate_response"),
        ] {
            let cancel = AtomicBool::new(false);
            let mut checker = ExternalUciCpuChecker::create(fixture_config(mode)).unwrap();
            checker
                .start(Instant::now() + Duration::from_secs(10), &cancel)
                .unwrap();
            let result = checker.analyze(
                &Position::startpos(),
                CpuLimits {
                    max_depth: 8,
                    max_nodes: 80,
                    deadline: Some(Instant::now() + Duration::from_secs(2)),
                },
                &cancel,
            );
            assert!(matches!(result, Err(CheckerError::External { code, .. }) if code == expected));
            let attempt = checker.last_attempt().unwrap().external.as_ref().unwrap();
            let partial = attempt.partial_report.as_ref().unwrap();
            assert_eq!(partial.completion, ExternalCompletion::Pending);
            assert_eq!(partial.best_move, None);
            assert!(attempt.process.cleanup_complete);
        }
    }
}
