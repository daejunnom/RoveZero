use crate::{ArenaError, ArenaPlan, canonical_sha256, decode_json};
use rz_experiments::ArtifactRef;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;
const CANONICALIZATION: &str = "rz-e02-ledger-json-v1";

#[derive(Clone, Copy, Debug)]
pub struct LedgerLimits {
    pub max_events: u64,
    pub max_bytes: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GameResult {
    WhiteWin,
    Draw,
    BlackWin,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EngineFailureKind {
    IllegalMove,
    Crash,
    Timeout,
}

/// Declared outcomes only. Evidence metadata and internal consistency are checked;
/// this type does not establish that a rules engine or subprocess observed them.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GameOutcome {
    RulesTerminal {
        result: GameResult,
        reason: String,
        evidence: ArtifactRef,
    },
    ProtocolAdjudicated {
        result: GameResult,
        policy_id: String,
        evidence: ArtifactRef,
    },
    EngineLoss {
        loser_engine: String,
        reason: EngineFailureKind,
        evidence: ArtifactRef,
    },
    InfrastructureInvalid {
        cause: String,
        evidence: ArtifactRef,
    },
    Incomplete {
        reason: String,
        evidence: ArtifactRef,
    },
    ContractInvalid {
        reason: String,
        evidence: ArtifactRef,
    },
    SimultaneousFailure {
        reason: String,
        evidence: ArtifactRef,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    PairStarted {
        pair_id: String,
        attempt: u32,
        process_run_id: String,
    },
    GameRecorded {
        pair_id: String,
        attempt: u32,
        game_id: String,
        outcome: GameOutcome,
    },
    PairClosed {
        pair_id: String,
        attempt: u32,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Header {
    ledger_version: u32,
    canonicalization: String,
    execution_ready: bool,
    input_sha256: String,
    plan_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Record {
    seq: u64,
    prev_sha256: String,
    event_sha256: String,
    event: Event,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct RecordInput<'a> {
    seq: u64,
    prev_sha256: &'a str,
    event: &'a Event,
}

#[derive(Clone, Debug)]
struct Attempt {
    number: u32,
    games: [Option<GameOutcome>; 2],
    closed: bool,
}

#[derive(Clone, Debug, Default)]
struct State {
    pairs: BTreeMap<String, Vec<Attempt>>,
    process_run_ids: BTreeSet<String>,
    evidence: BTreeMap<String, ArtifactRef>,
    evidence_bytes: u64,
    interpretation_blocked: bool,
}

#[derive(Clone, Debug)]
pub struct Ledger {
    plan: ArenaPlan,
    limits: LedgerLimits,
    header: Header,
    header_sha256: String,
    records: Vec<Record>,
    state: State,
    bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FailureCounts {
    pub illegal_move: u64,
    pub crash: u64,
    pub timeout: u64,
    pub infrastructure_invalid: u64,
    pub incomplete: u64,
    pub contract_invalid: u64,
    pub simultaneous_failure: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LedgerSummary {
    pub execution_ready: bool,
    pub input_sha256: String,
    pub plan_sha256: String,
    pub candidate_engine: String,
    pub planned_pairs: u64,
    pub completed_pairs: u64,
    pub excluded_pairs: u64,
    pub pending_pairs: u64,
    pub attempts: u64,
    pub closed_attempts: u64,
    pub recorded_games: u64,
    pub wins: u64,
    pub draws: u64,
    pub losses: u64,
    /// Opening-pair scores in half-point units: LL, DL, (DD or WL), WD, WW.
    pub n: [u64; 5],
    pub failure_counts: FailureCounts,
    pub interpretation_blocked: bool,
}

impl Ledger {
    pub fn new(plan: &ArenaPlan, limits: LedgerLimits) -> Result<Self, ArenaError> {
        if limits.max_events == 0 || limits.max_bytes == 0 {
            return Err(ArenaError::Budget("ledger limits must be positive".into()));
        }
        let limits = LedgerLimits {
            max_events: limits.max_events,
            max_bytes: limits
                .max_bytes
                .min(plan.manifest().input().budget.max_output_bytes),
        };
        let header = Header {
            ledger_version: 1,
            canonicalization: CANONICALIZATION.into(),
            execution_ready: false,
            input_sha256: plan.manifest().sha256().into(),
            plan_sha256: plan.sha256().into(),
        };
        let header_sha256 = canonical_sha256(&header)?;
        let bytes = line_bytes(&header)?;
        if bytes > limits.max_bytes {
            return Err(ArenaError::Budget(
                "header exceeds ledger byte limit".into(),
            ));
        }
        Ok(Self {
            plan: plan.clone(),
            limits,
            header,
            header_sha256,
            records: Vec::new(),
            state: State::default(),
            bytes,
        })
    }

    /// A rejected append leaves the previous state and serialized ledger unchanged.
    /// Process IDs are declaration tokens; appending cannot authorize runner resume.
    pub fn append(&mut self, event: Event) -> Result<(), ArenaError> {
        let seq = u64::try_from(self.records.len())
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| ArenaError::Budget("event sequence overflow".into()))?;
        if seq > self.limits.max_events {
            return Err(ArenaError::Budget("ledger event limit exceeded".into()));
        }
        let mut next = self.state.clone();
        apply(&self.plan, &mut next, &event)?;
        let previous = self
            .records
            .last()
            .map_or(self.header_sha256.as_str(), |r| r.event_sha256.as_str());
        let event_sha256 = canonical_sha256(&RecordInput {
            seq,
            prev_sha256: previous,
            event: &event,
        })?;
        let record = Record {
            seq,
            prev_sha256: previous.into(),
            event_sha256,
            event,
        };
        let bytes = self
            .bytes
            .checked_add(line_bytes(&record)?)
            .ok_or_else(|| ArenaError::Budget("ledger byte count overflow".into()))?;
        if bytes > self.limits.max_bytes {
            return Err(ArenaError::Budget("ledger byte limit exceeded".into()));
        }
        self.records.push(record);
        self.state = next;
        self.bytes = bytes;
        Ok(())
    }

    pub fn to_jsonl(&self) -> Result<String, ArenaError> {
        let mut output = serde_json::to_string(&self.header)
            .map_err(|e| ArenaError::Integrity(e.to_string()))?;
        output.push('\n');
        for record in &self.records {
            output.push_str(
                &serde_json::to_string(record).map_err(|e| ArenaError::Integrity(e.to_string()))?,
            );
            output.push('\n');
        }
        Ok(output)
    }

    /// Replays declarations, enforcing the state machine and semantic hash chain.
    /// The chain is not a signature: it cannot authenticate an external producer.
    /// A complete record prefix remains structurally valid; verify a separately
    /// trusted tip to detect removal of an entire suffix. This does not attest
    /// process continuity or permit a real runner to resume a partial pair.
    pub fn from_jsonl(
        plan: &ArenaPlan,
        input: &str,
        limits: LedgerLimits,
    ) -> Result<Self, ArenaError> {
        let mut ledger = Self::new(plan, limits)?;
        if u64::try_from(input.len()).map_or(true, |n| n > ledger.limits.max_bytes) {
            return Err(ArenaError::Budget("ledger input exceeds byte limit".into()));
        }
        if !input.ends_with('\n') {
            return Err(ArenaError::Integrity(
                "ledger must end with a complete newline-terminated record".into(),
            ));
        }
        let mut lines = input.split_terminator('\n');
        let header: Header = decode_json(
            lines
                .next()
                .ok_or_else(|| ArenaError::Integrity("missing ledger header".into()))?,
        )?;
        if header != ledger.header {
            return Err(ArenaError::Integrity(
                "ledger header, input lock or pair plan mismatch".into(),
            ));
        }
        for line in lines {
            let record: Record = decode_json(line)?;
            let next = u64::try_from(ledger.records.len())
                .ok()
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| ArenaError::Budget("event sequence overflow".into()))?;
            let previous = ledger
                .records
                .last()
                .map_or(ledger.header_sha256.as_str(), |r| r.event_sha256.as_str());
            if record.seq != next || record.prev_sha256 != previous {
                return Err(ArenaError::Integrity(
                    "event sequence or previous hash mismatch".into(),
                ));
            }
            let digest = canonical_sha256(&RecordInput {
                seq: record.seq,
                prev_sha256: &record.prev_sha256,
                event: &record.event,
            })?;
            if record.event_sha256 != digest {
                return Err(ArenaError::Integrity("event SHA-256 mismatch".into()));
            }
            ledger.append(record.event)?;
        }
        Ok(ledger)
    }

    pub fn events(&self) -> impl Iterator<Item = &Event> {
        self.records.iter().map(|r| &r.event)
    }

    pub fn tip_sha256(&self) -> &str {
        self.records
            .last()
            .map_or(self.header_sha256.as_str(), |record| {
                record.event_sha256.as_str()
            })
    }

    /// The expected tip must come from a separately trusted checkpoint.
    pub fn verify_tip(&self, expected: &str) -> Result<(), ArenaError> {
        if expected != self.tip_sha256() {
            return Err(ArenaError::Integrity(
                "ledger tip checkpoint mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn summary(&self) -> Result<LedgerSummary, ArenaError> {
        let candidate = &self.plan.manifest().input().engines[0].id;
        let planned_pairs = u64::try_from(self.plan.pairs().len())
            .map_err(|_| ArenaError::Budget("pair count overflow".into()))?;
        let mut summary = LedgerSummary {
            execution_ready: false,
            input_sha256: self.header.input_sha256.clone(),
            plan_sha256: self.header.plan_sha256.clone(),
            candidate_engine: candidate.clone(),
            planned_pairs,
            completed_pairs: 0,
            excluded_pairs: 0,
            pending_pairs: 0,
            attempts: 0,
            closed_attempts: 0,
            recorded_games: 0,
            wins: 0,
            draws: 0,
            losses: 0,
            n: [0; 5],
            failure_counts: FailureCounts::default(),
            interpretation_blocked: self.state.interpretation_blocked,
        };
        for pair in self.plan.pairs() {
            let Some(attempts) = self.state.pairs.get(&pair.id) else {
                summary.pending_pairs += 1;
                continue;
            };
            for attempt in attempts {
                summary.attempts += 1;
                summary.closed_attempts += u64::from(attempt.closed);
                for outcome in attempt.games.iter().flatten() {
                    summary.recorded_games += 1;
                    count_failure(&mut summary.failure_counts, outcome);
                }
            }
            let current = attempts.last().expect("nonempty started attempts");
            if !current.closed {
                summary.pending_pairs += 1;
                continue;
            }
            let scores: Option<Vec<u8>> = current
                .games
                .iter()
                .zip(&pair.games)
                .map(|(outcome, game)| score(outcome.as_ref()?, game, candidate))
                .collect();
            let Some(scores) = scores else {
                summary.excluded_pairs += 1;
                continue;
            };
            summary.completed_pairs += 1;
            let total = usize::from(scores[0] + scores[1]);
            summary.n[total] += 1;
            for points in scores {
                match points {
                    0 => summary.losses += 1,
                    1 => summary.draws += 1,
                    2 => summary.wins += 1,
                    _ => unreachable!("score is a bounded half-point integer"),
                }
            }
        }
        Ok(summary)
    }
}

fn line_bytes<T: Serialize>(value: &T) -> Result<u64, ArenaError> {
    let bytes = serde_json::to_vec(value).map_err(|e| ArenaError::Integrity(e.to_string()))?;
    if bytes.len() > MAX_LINE_BYTES {
        return Err(ArenaError::Budget("ledger line exceeds 4 MiB limit".into()));
    }
    u64::try_from(bytes.len())
        .ok()
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| ArenaError::Budget("ledger line length overflow".into()))
}

fn apply(plan: &ArenaPlan, state: &mut State, event: &Event) -> Result<(), ArenaError> {
    let (pair_id, attempt_number) = match event {
        Event::PairStarted {
            pair_id, attempt, ..
        }
        | Event::GameRecorded {
            pair_id, attempt, ..
        }
        | Event::PairClosed { pair_id, attempt } => (pair_id, *attempt),
    };
    let pair = plan
        .pair(pair_id)
        .ok_or_else(|| ArenaError::Invalid("unknown planned pair".into()))?;
    if attempt_number == 0 {
        return Err(ArenaError::Invalid("attempt numbers start at 1".into()));
    }
    match event {
        Event::PairStarted { process_run_id, .. } => {
            if state.interpretation_blocked {
                return Err(ArenaError::Invalid(
                    "contract-invalid run cannot start additional pair attempts".into(),
                ));
            }
            if process_run_id.is_empty()
                || process_run_id.len() > 128
                || !process_run_id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
            {
                return Err(ArenaError::Invalid("invalid process run ID".into()));
            }
            if state.process_run_ids.contains(process_run_id) {
                return Err(ArenaError::Invalid(
                    "process run ID was already used".into(),
                ));
            }
            let attempts = state.pairs.entry(pair_id.clone()).or_default();
            let expected = attempts
                .last()
                .map_or(Some(1), |a| a.number.checked_add(1))
                .ok_or_else(|| ArenaError::Budget("attempt count overflow".into()))?;
            if attempt_number != expected
                || attempt_number - 1 > plan.manifest().input().plan.max_retries_per_pair
            {
                return Err(ArenaError::Invalid(
                    "invalid or exhausted retry number".into(),
                ));
            }
            if let Some(previous) = attempts.last() {
                if !previous.closed || !retryable(previous) {
                    return Err(ArenaError::Invalid(
                        "retry requires a closed infrastructure-invalid whole pair".into(),
                    ));
                }
            }
            attempts.push(Attempt {
                number: attempt_number,
                games: [None, None],
                closed: false,
            });
            state.process_run_ids.insert(process_run_id.clone());
        }
        Event::GameRecorded {
            game_id, outcome, ..
        } => {
            let game_index = pair
                .games
                .iter()
                .position(|g| &g.id == game_id)
                .ok_or_else(|| ArenaError::Invalid("game does not belong to pair".into()))?;
            let attempt = state
                .pairs
                .get(pair_id)
                .and_then(|v| v.last())
                .ok_or_else(|| ArenaError::Invalid("pair attempt was not started".into()))?;
            if attempt.closed || attempt.number != attempt_number {
                return Err(ArenaError::Invalid(
                    "game references inactive pair attempt".into(),
                ));
            }
            let recorded = attempt.games.iter().filter(|g| g.is_some()).count();
            if recorded >= 2 || pair.execution_order[recorded] != game_index {
                return Err(ArenaError::Invalid(
                    "game duplicate or planned execution order mismatch".into(),
                ));
            }
            validate_outcome(&pair.games[game_index], outcome)?;
            register_evidence(plan, state, outcome.evidence())?;
            if matches!(
                outcome,
                GameOutcome::ContractInvalid { .. } | GameOutcome::SimultaneousFailure { .. }
            ) {
                state.interpretation_blocked = true;
            }
            state
                .pairs
                .get_mut(pair_id)
                .unwrap()
                .last_mut()
                .unwrap()
                .games[game_index] = Some(outcome.clone());
        }
        Event::PairClosed { .. } => {
            let attempt = state
                .pairs
                .get_mut(pair_id)
                .and_then(|v| v.last_mut())
                .ok_or_else(|| ArenaError::Invalid("pair attempt was not started".into()))?;
            if attempt.number != attempt_number || attempt.closed {
                return Err(ArenaError::Invalid(
                    "pair attempt is inactive or already closed".into(),
                ));
            }
            if attempt.games.iter().any(Option::is_none) {
                return Err(ArenaError::Invalid("partial pair cannot be closed".into()));
            }
            attempt.closed = true;
        }
    }
    Ok(())
}

fn retryable(attempt: &Attempt) -> bool {
    let mut infrastructure = false;
    for outcome in attempt.games.iter().flatten() {
        match outcome {
            GameOutcome::InfrastructureInvalid { .. } => infrastructure = true,
            GameOutcome::RulesTerminal { .. } | GameOutcome::ProtocolAdjudicated { .. } => {}
            _ => return false,
        }
    }
    infrastructure
}

fn text(value: &str, field: &str) -> Result<(), ArenaError> {
    if value.trim().is_empty() || value.len() > 1024 || value.chars().any(char::is_control) {
        return Err(ArenaError::Invalid(format!("invalid {field}")));
    }
    Ok(())
}

fn validate_outcome(game: &crate::GameSpec, outcome: &GameOutcome) -> Result<(), ArenaError> {
    let evidence = outcome.evidence();
    if evidence.path.len() > 1024
        || evidence.source.len() > 1024
        || evidence.license.len() > 1024
        || evidence.sha256.len() != 64
    {
        return Err(ArenaError::Invalid(
            "evidence metadata field exceeds limits".into(),
        ));
    }
    outcome.evidence().validate()?;
    match outcome {
        GameOutcome::RulesTerminal { result, reason, .. } => {
            let consistent = match reason.as_str() {
                "checkmate" => matches!(result, GameResult::WhiteWin | GameResult::BlackWin),
                "stalemate" | "dead_position" | "fivefold_repetition" | "seventy_five_move" => {
                    *result == GameResult::Draw
                }
                _ => false,
            };
            if !consistent {
                return Err(ArenaError::Invalid(
                    "unsupported or inconsistent terminal reason/result".into(),
                ));
            }
        }
        GameOutcome::ProtocolAdjudicated { policy_id, .. } => {
            text(policy_id, "adjudication policy ID")?;
            // E01 has no locked evaluation thresholds, tablebase provenance or
            // typed claim policy identity. A boolean cannot authorize arbitrary
            // policy scores. Keep the wire variant for a later policy adapter.
            return Err(ArenaError::Invalid(
                "protocol adjudication requires a concrete locked policy adapter".into(),
            ));
        }
        GameOutcome::EngineLoss { loser_engine, .. } => {
            if loser_engine != &game.white_engine && loser_engine != &game.black_engine {
                return Err(ArenaError::Invalid(
                    "losing engine is absent from game".into(),
                ));
            }
        }
        GameOutcome::InfrastructureInvalid { cause, .. } => text(cause, "infrastructure cause")?,
        GameOutcome::Incomplete { reason, .. }
        | GameOutcome::ContractInvalid { reason, .. }
        | GameOutcome::SimultaneousFailure { reason, .. } => text(reason, "failure reason")?,
    }
    Ok(())
}

impl GameOutcome {
    pub fn evidence(&self) -> &ArtifactRef {
        match self {
            Self::RulesTerminal { evidence, .. }
            | Self::ProtocolAdjudicated { evidence, .. }
            | Self::EngineLoss { evidence, .. }
            | Self::InfrastructureInvalid { evidence, .. }
            | Self::Incomplete { evidence, .. }
            | Self::ContractInvalid { evidence, .. }
            | Self::SimultaneousFailure { evidence, .. } => evidence,
        }
    }
}

fn register_evidence(
    plan: &ArenaPlan,
    state: &mut State,
    evidence: &ArtifactRef,
) -> Result<(), ArenaError> {
    if let Some(previous) = state.evidence.get(&evidence.path) {
        if previous != evidence {
            return Err(ArenaError::Invalid(
                "conflicting evidence identity at same path".into(),
            ));
        }
        return Ok(());
    }
    let bytes = state
        .evidence_bytes
        .checked_add(evidence.bytes)
        .ok_or_else(|| ArenaError::Budget("evidence byte count overflow".into()))?;
    if bytes > plan.manifest().input().budget.max_artifact_bytes {
        return Err(ArenaError::Budget(
            "evidence artifact byte limit exceeded".into(),
        ));
    }
    state
        .evidence
        .insert(evidence.path.clone(), evidence.clone());
    state.evidence_bytes = bytes;
    Ok(())
}

fn score(outcome: &GameOutcome, game: &crate::GameSpec, candidate: &str) -> Option<u8> {
    let result = match outcome {
        GameOutcome::RulesTerminal { result, .. }
        | GameOutcome::ProtocolAdjudicated { result, .. } => *result,
        GameOutcome::EngineLoss { loser_engine, .. } => {
            if loser_engine == &game.white_engine {
                GameResult::BlackWin
            } else {
                GameResult::WhiteWin
            }
        }
        _ => return None,
    };
    match result {
        GameResult::Draw => Some(1),
        GameResult::WhiteWin => Some(if candidate == game.white_engine { 2 } else { 0 }),
        GameResult::BlackWin => Some(if candidate == game.black_engine { 2 } else { 0 }),
    }
}

fn count_failure(counts: &mut FailureCounts, outcome: &GameOutcome) {
    match outcome {
        GameOutcome::EngineLoss {
            reason: EngineFailureKind::IllegalMove,
            ..
        } => counts.illegal_move += 1,
        GameOutcome::EngineLoss {
            reason: EngineFailureKind::Crash,
            ..
        } => counts.crash += 1,
        GameOutcome::EngineLoss {
            reason: EngineFailureKind::Timeout,
            ..
        } => counts.timeout += 1,
        GameOutcome::InfrastructureInvalid { .. } => counts.infrastructure_invalid += 1,
        GameOutcome::Incomplete { .. } => counts.incomplete += 1,
        GameOutcome::ContractInvalid { .. } => counts.contract_invalid += 1,
        GameOutcome::SimultaneousFailure { .. } => counts.simultaneous_failure += 1,
        _ => {}
    }
}
