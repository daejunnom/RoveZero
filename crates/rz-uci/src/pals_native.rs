//! Exact Rules-to-PALS model composition and a single native physical owner.
//! P/C heads cross the typed PALS scheduler; Python is never a product callback.
use rz_eval::pals_model::{
    PalsCandidateToken, PalsModelConfig, PalsModelInput, PalsRecordToken, PalsRole,
};
use rz_position::{
    BoardMove, Color, HistoryCompleteness, HistoryOrigin, PieceKind, Position, Square,
};
use rz_search::cpu::CpuScoreScope;
use rz_search::pals::engine::{DivergenceQuery, RecordKind, RoleError, RoleQuery, RoleRecord};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::time::Instant;

pub const PALS_RULES_ENCODING: &str = "rz-pals-rules-fields-v1";
pub const PALS_RULES_BASE_SEMANTICS: &str = "board:a1-h8;empty=0;white_PNBRQK=1..6;black_PNBRQK=7..12;history:newest_first_exact_fen;records:latest_kinds_and_author_critical_pinned;candidate_order:Rules;divergence_query:fields_1,3,4,8..14=0";
const MAX_HISTORY: usize = 4096;

/// Index order, normalization and missing-value meaning are model semantics.
/// A source digest is recorded separately: changing a worker or a comment must
/// not silently change the exact-input cache's encoding meaning.
pub const PALS_METADATA_FEATURES: [&str; 16] = [
    "white_to_move:boolean",
    "white_king_castle:boolean",
    "white_queen_castle:boolean",
    "black_king_castle:boolean",
    "black_queen_castle:boolean",
    "ep_file:/7;absent=-1",
    "ep_rank:/7;absent=-1",
    "halfmove_clock:/150",
    "fullmove_number:/512",
    "known_history_frames:/4096",
    "unknown_history_prefix:boolean",
    "known_repetition_count:exact_count_as_f32",
    "in_check:boolean",
    "white_piece_count:/16",
    "black_piece_count:/16",
    "constant:1",
];
pub const PALS_RECORD_FEATURES: [&str; 16] = [
    "kind:proposal=0,counterexample=1,repair=2,cpu_verification=3",
    "has_completed_value:boolean",
    "completed_value:/30000;missing=0",
    "completed_depth:/64",
    "score_scope:absent=-1,frontier_only=0,completed=1,rules_terminal=2",
    "white_score_perspective:boolean",
    "origin_state_id:advisory_f32_not_identity",
    "line_length:/256",
    "first_from:/63;missing=0",
    "first_to:/63;missing=0",
    "first_promotion:/4;none=0,Q=1,R=2,B=3,N=4",
    "has_first_move:boolean",
    "last_from:/63;missing=0",
    "last_to:/63;missing=0",
    "last_promotion:/4;none=0,Q=1,R=2,B=3,N=4",
    "unknown_value:boolean",
];
pub const PALS_QUERY_FEATURES: [&str; 16] = [
    "mode:propose=0,reply=1,repair=2,divergence=3",
    "prefix_length:/256",
    "proposal_length:/256",
    "has_counterexample:boolean",
    "counterexample_length:/256",
    "candidate_count:role=/256,divergence=/128",
    "situation_revision:advisory_f32_not_identity",
    "deadline_remaining:seconds",
    "white_to_move:boolean",
    "prefix_last_from:/63;missing=0",
    "prefix_last_to:/63;missing=0",
    "prefix_last_promotion:/4;none=0,Q=1,R=2,B=3,N=4",
    "proposal_first_from:/63;missing=0",
    "proposal_first_to:/63;missing=0",
    "proposal_first_promotion:/4;none=0,Q=1,R=2,B=3,N=4",
    "constant:1",
];
pub const PALS_DIVERGENCE_FEATURES: [&str; 8] = [
    "ply_index:/256",
    "opponent_to_move_relative_root:boolean",
    "prefix_in_check:boolean",
    "movement_from:/63",
    "movement_to:/63",
    "promotion:/4;none=0,Q=1,R=2,B=3,N=4",
    "prefix_halfmove_clock:/150",
    "prefix_legal_candidate_count:/256",
];
pub fn pals_rules_semantic_fields() -> impl Iterator<Item = &'static str> {
    std::iter::once(PALS_RULES_ENCODING)
        .chain(std::iter::once(PALS_RULES_BASE_SEMANTICS))
        .chain(PALS_METADATA_FEATURES)
        .chain(PALS_RECORD_FEATURES)
        .chain(PALS_QUERY_FEATURES)
        .chain(PALS_DIVERGENCE_FEATURES)
}
pub fn pals_rules_encoding_semantic_digest() -> [u8; 32] {
    let mut h = Sha256::new();
    for field in pals_rules_semantic_fields() {
        h.update((field.len() as u64).to_le_bytes());
        h.update(field.as_bytes());
    }
    h.finalize().into()
}
pub fn pals_native_source_digest() -> [u8; 32] {
    Sha256::digest(include_bytes!("pals_native.rs")).into()
}
/// Shape compatibility alone does not prove that a model was trained on the
/// same feature meanings. This pin verifies the declared Rules transformation;
/// actual learned-input suitability still needs separate data/model evidence.
pub fn verify_pals_rules_profile(
    profile: Option<&str>,
    semantic_digest: Option<[u8; 32]>,
) -> Result<(), RoleError> {
    match (profile, semantic_digest) {
        (Some(PALS_RULES_ENCODING), Some(digest)) if digest == pals_rules_encoding_semantic_digest() => Ok(()),
        (None, _) | (_, None) => Err(RoleError::Backend("PALS product export lacks the required Rules input semantic profile; legacy numeric fixtures do not authorize product execution".into())),
        _ => Err(RoleError::Backend("PALS export Rules input semantic profile differs from this encoder".into())),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeQueryKind {
    Propose,
    Reply,
    Repair,
    Divergence,
}

fn model_error(error: impl std::fmt::Display) -> RoleError {
    RoleError::Backend(error.to_string())
}
fn flag(value: bool) -> f32 {
    f32::from(u8::from(value))
}
fn candidate(movement: BoardMove) -> PalsCandidateToken {
    PalsCandidateToken {
        from: movement.from.index(),
        to: movement.to.index(),
        promotion: match movement.promotion {
            None => 0,
            Some(PieceKind::Queen) => 1,
            Some(PieceKind::Rook) => 2,
            Some(PieceKind::Bishop) => 3,
            Some(PieceKind::Knight) => 4,
            _ => 255,
        },
    }
}
/// Streaming exact known history. Unknown prefixes remain explicitly distinct;
/// compact network features never substitute for repetition/terminal authority.
pub fn pals_history_digest(position: &Position) -> Result<[u8; 32], RoleError> {
    if position.known_history_len() > MAX_HISTORY {
        return Err(RoleError::Backend(
            "PALS exact history budget exceeded".into(),
        ));
    }
    let snapshot = position.snapshot();
    let mut h = Sha256::new();
    h.update(PALS_RULES_ENCODING);
    h.update([match snapshot.history_completeness() {
        HistoryCompleteness::Complete => 0,
        HistoryCompleteness::UnknownPrefix => 1,
    }]);
    h.update([match snapshot.history_origin() {
        HistoryOrigin::StartPosition => 0,
        HistoryOrigin::Fen => 1,
    }]);
    h.update((snapshot.known_history_len() as u64).to_le_bytes());
    for frame in snapshot.known_history() {
        let fen = frame.to_fen();
        h.update((fen.len() as u64).to_le_bytes());
        h.update(fen.as_bytes());
    }
    Ok(h.finalize().into())
}
fn board_and_metadata(position: &Position) -> Result<(Vec<u8>, [f32; 16]), RoleError> {
    let mut board = Vec::with_capacity(64);
    let mut pieces = [0_u32; 2];
    for index in 0..64 {
        let piece = position.piece_at(Square::new(index).map_err(model_error)?);
        board.push(match piece {
            None => 0,
            Some(piece) => {
                let offset = if piece.color == Color::White {
                    pieces[0] += 1;
                    0
                } else {
                    pieces[1] += 1;
                    6
                };
                offset
                    + match piece.kind {
                        PieceKind::Pawn => 1,
                        PieceKind::Knight => 2,
                        PieceKind::Bishop => 3,
                        PieceKind::Rook => 4,
                        PieceKind::Queen => 5,
                        PieceKind::King => 6,
                    }
            }
        });
    }
    let rights = position.castling_rights();
    let ep = position.en_passant_target();
    let mut metadata = [0.0; 16];
    metadata[0] = flag(position.side_to_move() == Color::White);
    for i in 0..4 {
        metadata[i + 1] = flag(rights & (1 << i) != 0);
    }
    metadata[5] = ep.map_or(-1.0, |s| f32::from(s.file()) / 7.0);
    metadata[6] = ep.map_or(-1.0, |s| f32::from(s.rank()) / 7.0);
    metadata[7] = position.halfmove_clock() as f32 / 150.0;
    metadata[8] = position.fullmove_number() as f32 / 512.0;
    metadata[9] = position.known_history_len() as f32 / MAX_HISTORY as f32;
    metadata[10] = flag(position.history_completeness() == HistoryCompleteness::UnknownPrefix);
    metadata[11] = position.known_repetition_count() as f32;
    metadata[12] = flag(position.in_check());
    metadata[13] = pieces[0] as f32 / 16.0;
    metadata[14] = pieces[1] as f32 / 16.0;
    metadata[15] = 1.0;
    Ok((board, metadata))
}
fn records(
    records: &[RoleRecord],
    revision: u64,
) -> Result<(Vec<PalsRecordToken>, Vec<u64>), RoleError> {
    if records.len() > 16_384 {
        return Err(RoleError::Backend(
            "PALS public record scan budget exceeded".into(),
        ));
    }
    let mut critical = BTreeSet::new();
    for kind in [
        RecordKind::Proposal,
        RecordKind::Counterexample,
        RecordKind::Repair,
        RecordKind::CpuVerification,
    ] {
        if let Some(index) = records.iter().rposition(|r| r.kind == kind) {
            critical.insert(index);
        }
    }
    for (index, record) in records.iter().enumerate() {
        if record.revision > revision {
            return Err(RoleError::Backend(
                "future public record in PALS input".into(),
            ));
        }
        if record.critical {
            critical.insert(index);
        }
    }
    if critical.len() > 128 {
        return Err(RoleError::Backend(
            "PALS critical context exceeds registered record width".into(),
        ));
    }
    let mut admitted = critical.clone();
    for index in (0..records.len()).rev() {
        if admitted.len() == 128 {
            break;
        }
        admitted.insert(index);
    }
    let mut result = Vec::with_capacity(admitted.len());
    let mut required = Vec::with_capacity(critical.len());
    for index in admitted {
        let record = &records[index];
        if record.line.len() > 256 {
            return Err(RoleError::Backend(
                "PALS record line budget exceeded".into(),
            ));
        }
        let id = (index as u64)
            .checked_add(1)
            .ok_or_else(|| RoleError::Backend("PALS record ID overflow".into()))?;
        let is_critical = critical.contains(&index);
        if is_critical {
            required.push(id);
        }
        let mut features = [0.0; 16];
        features[0] = match record.kind {
            RecordKind::Proposal => 0.0,
            RecordKind::Counterexample => 1.0,
            RecordKind::Repair => 2.0,
            RecordKind::CpuVerification => 3.0,
        };
        features[1] = flag(record.value.is_some());
        features[2] = record.value.unwrap_or(0) as f32 / 30_000.0;
        features[3] = f32::from(record.completed_depth) / 64.0;
        features[4] = match record.score_scope {
            None => -1.0,
            Some(CpuScoreScope::FrontierOnly) => 0.0,
            Some(CpuScoreScope::CompletedIteration) => 1.0,
            Some(CpuScoreScope::RulesTerminal) => 2.0,
        };
        features[5] = flag(record.perspective == Color::White);
        // An anchor identifier is a provenance field, never a calibrated score.
        features[6] = record.origin_state.0 as f32;
        features[7] = record.line.len() as f32 / 256.0;
        if let Some(movement) = record.line.first() {
            let c = candidate(*movement);
            c.validate().map_err(model_error)?;
            features[8] = f32::from(c.from) / 63.0;
            features[9] = f32::from(c.to) / 63.0;
            features[10] = f32::from(c.promotion) / 4.0;
            features[11] = 1.0;
        }
        if let Some(movement) = record.line.last() {
            let c = candidate(*movement);
            c.validate().map_err(model_error)?;
            features[12] = f32::from(c.from) / 63.0;
            features[13] = f32::from(c.to) / 63.0;
            features[14] = f32::from(c.promotion) / 4.0;
        }
        features[15] = flag(record.value.is_none());
        result.push(PalsRecordToken {
            record_id: id,
            revision: record.revision,
            critical: is_critical,
            features,
        });
    }
    Ok((result, required))
}
pub fn prepare_role_input(
    query: &RoleQuery<'_>,
    kind: NativeQueryKind,
    model_epoch: [u8; 32],
) -> Result<PalsModelInput, RoleError> {
    query.check_control()?;
    if kind == NativeQueryKind::Divergence
        || query.prefix.len() > 256
        || query.proposal.len() > 256
        || query.counterexample.is_some_and(|line| line.len() > 256)
    {
        return Err(RoleError::InvalidOutput);
    }
    if query.legal != query.position.legal_moves() {
        return Err(RoleError::Backend(
            "PALS model candidates differ from exact Rules legal order".into(),
        ));
    }
    let (board, metadata) = board_and_metadata(query.position)?;
    let (records, required_critical_records) = records(query.records, query.revision)?;
    let candidates = query.legal.iter().copied().map(candidate).collect();
    let mut features = [0.0; 16];
    features[0] = match kind {
        NativeQueryKind::Propose => 0.0,
        NativeQueryKind::Reply => 1.0,
        NativeQueryKind::Repair => 2.0,
        NativeQueryKind::Divergence => unreachable!(),
    };
    features[1] = query.prefix.len() as f32 / 256.0;
    features[2] = query.proposal.len() as f32 / 256.0;
    features[3] = flag(query.counterexample.is_some());
    features[4] = query.counterexample.map_or(0.0, |l| l.len() as f32 / 256.0);
    features[5] = query.legal.len() as f32 / 256.0;
    features[6] = query.revision as f32;
    features[7] = query
        .deadline
        .saturating_duration_since(Instant::now())
        .as_secs_f32();
    features[8] = flag(query.position.side_to_move() == Color::White);
    if let Some(movement) = query.prefix.last() {
        let c = candidate(*movement);
        features[9] = f32::from(c.from) / 63.0;
        features[10] = f32::from(c.to) / 63.0;
        features[11] = f32::from(c.promotion) / 4.0;
    }
    if let Some(movement) = query.proposal.first() {
        let c = candidate(*movement);
        features[12] = f32::from(c.from) / 63.0;
        features[13] = f32::from(c.to) / 63.0;
        features[14] = f32::from(c.promotion) / 4.0;
    }
    features[15] = 1.0;
    let input = PalsModelInput {
        role: if kind == NativeQueryKind::Reply {
            PalsRole::Critic
        } else {
            PalsRole::Proposer
        },
        board,
        metadata,
        records,
        required_critical_records,
        candidates,
        divergence_features: vec![],
        query: features,
        situation_revision: query.revision,
        history_digest: pals_history_digest(query.position)?,
        model_epoch,
    };
    input
        .validate(&PalsModelConfig::baseline())
        .map_err(model_error)?;
    Ok(input)
}
pub fn prepare_divergence_input(
    query: &DivergenceQuery<'_>,
    model_epoch: [u8; 32],
) -> Result<PalsModelInput, RoleError> {
    if query.cancel.load(std::sync::atomic::Ordering::Acquire) {
        return Err(RoleError::Canceled);
    }
    if Instant::now() >= query.deadline {
        return Err(RoleError::Deadline);
    }
    if query.proposal.len() > 256
        || query.candidates.is_empty()
        || query.candidates.len() > 128
        || query
            .candidates
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != query.candidates.len()
    {
        return Err(RoleError::InvalidOutput);
    }
    let (board, metadata) = board_and_metadata(query.root)?;
    let (records, required_critical_records) = records(query.records, query.revision)?;
    let mut prefix = query.root.clone();
    let mut by_ply = Vec::with_capacity(query.proposal.len());
    for (ply, movement) in query.proposal.iter().enumerate() {
        let c = candidate(*movement);
        c.validate().map_err(model_error)?;
        by_ply.push([
            ply as f32 / 256.0,
            flag(prefix.side_to_move() != query.root.side_to_move()),
            flag(prefix.in_check()),
            f32::from(c.from) / 63.0,
            f32::from(c.to) / 63.0,
            f32::from(c.promotion) / 4.0,
            prefix.halfmove_clock() as f32 / 150.0,
            prefix.legal_moves().len() as f32 / 256.0,
        ]);
        prefix.make_move(*movement).map_err(model_error)?;
    }
    let mut divergence_features = Vec::with_capacity(query.candidates.len());
    for &ply in query.candidates {
        let features = by_ply.get(ply).ok_or(RoleError::InvalidOutput)?;
        if features[1] != 1.0 {
            return Err(RoleError::Backend(
                "critic divergence is not an opponent turn".into(),
            ));
        }
        divergence_features.push(*features);
    }
    let mut features = [0.0; 16];
    features[0] = 3.0;
    features[2] = query.proposal.len() as f32 / 256.0;
    features[5] = query.candidates.len() as f32 / 128.0;
    features[6] = query.revision as f32;
    features[7] = query
        .deadline
        .saturating_duration_since(Instant::now())
        .as_secs_f32();
    features[15] = 1.0;
    let input = PalsModelInput {
        role: PalsRole::Critic,
        board,
        metadata,
        records,
        required_critical_records,
        candidates: vec![],
        divergence_features,
        query: features,
        situation_revision: query.revision,
        history_digest: pals_history_digest(query.root)?,
        model_epoch,
    };
    input
        .validate(&PalsModelConfig::baseline())
        .map_err(model_error)?;
    Ok(input)
}

#[cfg(feature = "onnx-cpu")]
mod native {
    use super::*;
    use rz_contracts::pals::{
        ExecutionMode, Move16, PALS_CONTRACT_REVISION, RepresentationKey, Role, RoleOutput,
        RolePayload, RoleRequest, SearchAuthority, SituationHandle,
    };
    use rz_contracts::{
        ByteBudget, CancelToken, ContractError, Deadline, Digest, ErrorCode, ExecutionId,
        GameGeneration, MonotonicTick, OwnerId, PrecisionProfile, ProcessEpoch, RequestId,
        RootGeneration, Stage, Wdl,
    };
    use rz_eval::error::BackendError;
    use rz_eval::pals_model::PALS_ENCODING_SCHEMA;
    use rz_eval::pals_onnx::{
        PalsBackendStats, PalsCudaControlPolicy, PalsCudaPlacementWitness, PalsGraphOptimization,
        PalsNativeCommand, PalsNativeResult, PalsOnnxBackend, PalsSessionResidency,
    };
    use rz_eval::worker::{PhysicalLease, PhysicalPoll, SingleWorker};
    use rz_position::contracts::ContractPosition;
    use rz_runtime::contracts::{ContractClock, ContractSystemClock};
    use rz_runtime::pals::{
        PalsAdapter, PalsRuntime, PalsScope, PalsTerminal, PhysicalRoleOutput, RepresentationScope,
        RuntimeRoleRequest, SharedPalsScope, SharedRepresentationScope,
    };
    use rz_runtime::{Backend, BackendResult, Clock, Limits, Resources};
    use rz_search::pals::engine::{RoleEvaluation, RoleModel, RoleSearchClosure};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::Poll;
    use std::time::Duration;

    const REQUEST_BYTES: u64 = 1024 * 1024;
    const DEFAULT_DRAIN_LIMIT: Duration = Duration::from_secs(2);
    const DEPLOYMENT_FROZEN_EPOCH: u64 = 1;
    static EPOCHS: AtomicU64 = AtomicU64::new(0x5041_4c53_0000_0000);
    static RULE_OWNERS: AtomicU64 = AtomicU64::new(0x5041_4c52_0000_0000);
    type NativeWorker = SingleWorker<PalsNativeCommand, Result<PalsNativeResult, BackendError>>;
    type NativePhysicalLease =
        PhysicalLease<PalsNativeCommand, Result<PalsNativeResult, BackendError>>;
    type Runtime = PalsRuntime<PalsModelInput, WorkerBackend, ContractSystemClock>;
    fn fault(code: ErrorCode, detail: &'static str) -> ContractError {
        ContractError::new(code, Stage::Backend, detail)
    }
    fn next(counter: &AtomicU64) -> Result<u64, RoleError> {
        counter
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_add(1))
            .map_err(|_| RoleError::Backend("PALS owner ID epoch exhausted".into()))
    }
    struct WorkerOwner {
        worker: Mutex<NativeWorker>,
        physical_completed: AtomicU64,
        completed: AtomicU64,
        physical_failed: AtomicU64,
        validation_failed: AtomicU64,
        delivered: AtomicU64,
        consumed: AtomicU64,
        canceled: AtomicU64,
        expired: AtomicU64,
        new_game_resets: AtomicU64,
        in_flight: AtomicU64,
        game_generation: AtomicU64,
        request_high_water: AtomicU64,
        execution_high_water: AtomicU64,
        quarantined: AtomicBool,
        finishing: AtomicBool,
        shutdown: AtomicBool,
        model_epoch: [u8; 32],
        manifest_digest: [u8; 32],
        encoding_semantic_digest: [u8; 32],
        adapter_source_digest: [u8; 32],
        trained: bool,
        residency: PalsSessionResidency,
        process_epoch: ProcessEpoch,
        last_failure: Mutex<Option<NativeFailureReceipt>>,
        final_stats: Mutex<Option<NativeBackendStatsReceipt>>,
        stats_lease: Mutex<Option<NativePhysicalLease>>,
        execution: NativeExecutionReceipt,
        startup_probe: Mutex<Option<NativeStartupProbeReceipt>>,
        startup_ready: AtomicBool,
        final_mapping_confirmed: AtomicBool,
        observer: Mutex<Option<Box<dyn NativeRoleObserver>>>,
        observer_failures: AtomicU64,
        last_observer_failure: Mutex<Option<NativeFailureReceipt>>,
    }
    #[derive(Clone, Copy)]
    pub enum NativePreparedContext<'a> {
        Role {
            query: &'a RoleQuery<'a>,
            kind: NativeQueryKind,
        },
        Divergence {
            query: &'a DivergenceQuery<'a>,
        },
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NativeRoleRejection {
        Admission,
        Canceled,
        Deadline,
        Stale,
        Backend,
        InvalidOutput,
        PhysicalCompletionUnknown,
        Observer,
        SupersededUnconsumed,
        SearchFinishedUnconsumed,
        SearchFailedUnconsumed,
        GameReset,
    }
    /// Optional bounded collector. All callbacks run on the calling/search
    /// thread and use its clock; startup probe inputs never enter this stream.
    pub trait NativeRoleObserver: Send {
        fn prepared(
            &mut self,
            id: RequestId,
            input: &PalsModelInput,
            context: NativePreparedContext<'_>,
        ) -> Result<(), RoleError>;
        fn physically_completed(
            &mut self,
            _id: RequestId,
            _result: Result<&rz_eval::pals_model::PalsRawOutput, &BackendError>,
        ) -> Result<(), RoleError> {
            Ok(())
        }
        fn delivered(&mut self, _id: RequestId) -> Result<(), RoleError> {
            Ok(())
        }
        fn rejected(
            &mut self,
            _id: RequestId,
            _reason: NativeRoleRejection,
        ) -> Result<(), RoleError> {
            Ok(())
        }
        fn accepted(&mut self, _id: RequestId) -> Result<(), RoleError> {
            Ok(())
        }
    }
    /// Provider and admission declarations are not measured native/VRAM peaks.
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeExecutionReceipt {
        pub provider: &'static str,
        pub device_id: Option<i32>,
        /// ORT allocator limit per session, excluding unobserved allocations.
        pub session_arena_bytes: Option<u64>,
        pub runtime_sha256: [u8; 32],
        pub runtime_bundle_sha256: Option<[u8; 32]>,
        /// Identity of the explicitly loaded metadata-only control inventory.
        /// None means strict CUDA or CPU, not an observed placement witness.
        pub cuda_control_inventory_sha256: Option<[u8; 32]>,
        pub transient_request_device_bytes: u64,
        pub transient_execution_device_bytes: u64,
        pub pinned_request_bytes: u64,
        pub device_public_memory: bool,
    }
    /// Initialization work is never converted into search role consumption.
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeStartupProbeReceipt {
        pub completed_proposer_calls: u64,
        pub completed_critic_calls: u64,
        pub runtime_mapping_confirmed: bool,
        /// Same-worker NN-zero ACK after physically completed public/P/C probes.
        /// Library origin confirmation alone never supplies this evidence.
        pub cuda_placement_witness: Option<Box<PalsCudaPlacementWitness>>,
        pub reset_completed: bool,
        /// Actual cumulative snapshot after probe and reset; final statistics
        /// include this work and must not be reported as search-only inputs.
        pub backend_stats: Option<NativeBackendStatsReceipt>,
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeRoleSourceIdentity {
        pub checkpoint_sha256: [u8; 32],
        pub frozen_epoch: u64,
        pub export_manifest_sha256: [u8; 32],
        pub encoding_semantic_sha256: [u8; 32],
        pub adapter_source_sha256: [u8; 32],
        pub model_configuration: PalsModelConfig,
        pub trained: bool,
        pub execution: NativeExecutionReceipt,
    }
    fn transient_device_layout() -> Result<(u64, u64), RoleError> {
        use rz_eval::pals_model::{
            DIVERGENCE_FEATURES, METADATA_FEATURES, QUERY_FEATURES, RECORD_FEATURES,
        };
        let c = PalsModelConfig::baseline();
        let to_u64 = |n| u64::try_from(n).map_err(|_| RoleError::InvalidOutput);
        let r = to_u64(c.max_records)?;
        let candidates = to_u64(c.max_candidates)?;
        let divergences = to_u64(c.max_divergences)?;
        // Actual bounded ActiveInputs tensor layout: i64 board/candidates,
        // f32 numeric fields, bool masks and the shared P/C scalar condition.
        let inputs = 64 * 8
            + to_u64(METADATA_FEATURES)? * 4
            + r * to_u64(RECORD_FEATURES)? * 4
            + r
            + candidates * 3 * 8
            + candidates
            + divergences * to_u64(DIVERGENCE_FEATURES)? * 4
            + divergences
            + to_u64(QUERY_FEATURES)? * 4
            + 1;
        let tokens = to_u64(c.public_memory_tokens(c.max_records).map_err(model_error)?)?;
        // Two public K/V heads and mask plus private latent/policy/WDL/critic
        // outputs. Native graph workspace/weights remain the separately
        // declared session allocator domain; these are tensor reservations.
        let outputs = 2 * to_u64(c.kv_heads)? * tokens * to_u64(c.head_dimension)? * 4
            + tokens
            + to_u64(c.latent_elements())? * 4
            + candidates * 4
            + 3 * 4
            + divergences * 4;
        Ok((inputs, outputs))
    }
    fn execution_receipt(backend: &PalsOnnxBackend) -> Result<NativeExecutionReceipt, RoleError> {
        let (provider, device_id, session_arena_bytes, request, physical) =
            match backend.config().provider {
                rz_eval::onnx::Provider::Cpu => ("cpu", None, None, 0, 0),
                rz_eval::onnx::Provider::Cuda {
                    device_id,
                    arena_bytes,
                } => {
                    if !cfg!(all(feature = "onnx-cuda", target_os = "linux")) {
                        return Err(RoleError::Backend(
                            "PALS CUDA requires onnx-cuda on Linux; CPU fallback forbidden".into(),
                        ));
                    }
                    let (request, physical) = transient_device_layout()?;
                    (
                        "cuda",
                        Some(device_id),
                        Some(u64::try_from(arena_bytes).map_err(|_| RoleError::InvalidOutput)?),
                        request,
                        physical,
                    )
                }
            };
        Ok(NativeExecutionReceipt {
            provider,
            device_id,
            session_arena_bytes,
            runtime_sha256: backend.runtime_binary_digest(),
            runtime_bundle_sha256: backend.runtime_bundle_digest(),
            cuda_control_inventory_sha256: backend.cuda_control_inventory_digest(),
            transient_request_device_bytes: request,
            transient_execution_device_bytes: physical,
            // Rust owns pageable host tensors in this first host-K/V path.
            // Unknown internal ORT staging is not a claimed observed zero.
            pinned_request_bytes: 0,
            device_public_memory: backend.config().device_public_memory,
        })
    }
    fn validate_startup_cuda_witness(
        witness: &PalsCudaPlacementWitness,
        execution: &NativeExecutionReceipt,
        manifest: [u8; 32],
        residency: &PalsSessionResidency,
    ) -> Result<(), RoleError> {
        // The backend owns exact opcode/provenance/profile validation. This
        // worker boundary separately binds its typed ACK to this actual owner.
        let matching_graphs = witness.initialization.len() == 2
            && ["public", "shared_pc"].iter().all(|role| {
                let mut placements = witness
                    .initialization
                    .iter()
                    .filter(|placement| placement.role == *role);
                let Some(placement) = placements.next() else {
                    return false;
                };
                placements.next().is_none()
                    && placement.assigned_nodes > 0
                    && placement.cuda_nodes > 0
                    && placement.optimization == PalsGraphOptimization::Disable
                    && placement
                        .cuda_nodes
                        .checked_add(placement.approved_cpu_control_nodes)
                        == Some(placement.assigned_nodes)
                    && residency
                        .graphs
                        .iter()
                        .any(|graph| graph.role == *role && graph.sha256 == placement.graph_sha256)
            });
        if execution.provider != "cuda"
            || execution.cuda_control_inventory_sha256 != Some(witness.inventory_sha256)
            || witness.manifest_sha256 != manifest
            || witness.runtime_sha256 != execution.runtime_sha256
            || execution.runtime_bundle_sha256 != Some(witness.runtime_bundle_sha256)
            || witness.schema != "rovezero.pals-cuda-metadata-control.v2"
            || witness.optimization != PalsGraphOptimization::Disable
            || !matching_graphs
            || witness.public.cuda_kernels == 0
            || witness.public.cuda_transfer_kernels != 0
            || witness.public.neural_kernels == 0
            || witness.shared_pc.cuda_kernels == 0
            || witness.shared_pc.cuda_transfer_kernels > witness.shared_pc.cuda_kernels
            || witness.shared_pc.proposer_private_kernels == 0
            || witness.shared_pc.critic_private_kernels == 0
        {
            return Err(RoleError::Backend(
                "PALS CUDA placement ACK is incomplete or belongs to a different pinned owner"
                    .into(),
            ));
        }
        Ok(())
    }
    /// Only an acknowledged exclusive worker snapshot supplies these counters.
    /// No semantic role completion count is expanded into guessed graph work.
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeBackendStatsReceipt {
        pub admitted_role_requests: u64,
        pub public_cache_hits: u64,
        pub public_cache_misses: u64,
        pub public_nn_runs_attempted: u64,
        pub public_nn_runs_completed: u64,
        pub public_nn_runs_failed_known: u64,
        pub role_nn_runs_attempted: u64,
        pub role_nn_runs_completed: u64,
        pub role_nn_runs_failed_known: u64,
        pub completed_nn_inputs: u64,
        pub validated_public_outputs: u64,
        pub validated_role_outputs: u64,
        pub new_game_resets: u64,
        pub live_public_cache_entries: u64,
    }
    impl From<PalsBackendStats> for NativeBackendStatsReceipt {
        fn from(value: PalsBackendStats) -> Self {
            Self {
                admitted_role_requests: value.admitted_role_requests,
                public_cache_hits: value.public_cache_hits,
                public_cache_misses: value.public_cache_misses,
                public_nn_runs_attempted: value.public_nn_runs_attempted,
                public_nn_runs_completed: value.public_nn_runs_completed,
                public_nn_runs_failed_known: value.public_nn_runs_failed_known,
                role_nn_runs_attempted: value.role_nn_runs_attempted,
                role_nn_runs_completed: value.role_nn_runs_completed,
                role_nn_runs_failed_known: value.role_nn_runs_failed_known,
                completed_nn_inputs: value.completed_nn_inputs,
                validated_public_outputs: value.validated_public_outputs,
                validated_role_outputs: value.validated_role_outputs,
                new_game_resets: value.new_game_resets,
                live_public_cache_entries: value.live_public_cache_entries,
            }
        }
    }
    /// Bounded typed diagnostics. Library message text and paths are never
    /// copied into a portable receipt.
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeFailureReceipt {
        pub kind: String,
        pub stage: String,
        pub detail: &'static str,
        pub cause_code: Option<String>,
        pub cause_prefix_sha256: Option<[u8; 32]>,
        pub cause_hashed_bytes: Option<u16>,
        pub cause_truncated: Option<bool>,
    }
    impl WorkerOwner {
        fn observe(
            &self,
            stage: &'static str,
            callback: impl FnOnce(&mut dyn NativeRoleObserver) -> Result<(), RoleError>,
        ) -> Result<(), RoleError> {
            let result = {
                let mut observer = self
                    .observer
                    .lock()
                    .map_err(|_| RoleError::Backend("PALS observer owner poisoned".into()))?;
                match observer.as_mut() {
                    Some(observer) => callback(observer.as_mut()),
                    None => Ok(()),
                }
            };
            if result.is_err() {
                self.observer_failures.fetch_add(1, Ordering::AcqRel);
                *self
                    .last_observer_failure
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(NativeFailureReceipt {
                    kind: "Observer".into(),
                    stage: stage.into(),
                    detail: "bounded native role observer failed",
                    cause_code: None,
                    cause_prefix_sha256: None,
                    cause_hashed_bytes: None,
                    cause_truncated: None,
                });
                // A logging/sink error cannot retroactively make an already
                // completed physical native run become completion-unknown.
                return Err(RoleError::Backend(
                    "PALS native role observer failed; actual physical work is retained".into(),
                ));
            }
            Ok(())
        }
        fn remember_failure(&self, error: &BackendError) {
            let receipt = NativeFailureReceipt {
                kind: format!("{:?}", error.kind),
                stage: format!("{:?}", error.stage),
                detail: error.detail,
                cause_code: error.cause.map(|c| format!("{:?}", c.code)),
                cause_prefix_sha256: error.cause.map(|c| c.prefix_sha256),
                cause_hashed_bytes: error.cause.map(|c| c.hashed_bytes),
                cause_truncated: error.cause.map(|c| c.truncated),
            };
            *self.last_failure.lock().unwrap_or_else(|e| e.into_inner()) = Some(receipt);
        }
    }
    #[derive(Clone)]
    pub struct NativeRoleFinishHandle {
        owner: Arc<WorkerOwner>,
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeRoleReceipt {
        /// Completed native role calls, including a physically completed failure.
        pub physically_completed_role_calls: u64,
        /// Successful raw native role callbacks, distinct from actual ONNX
        /// graph inputs. Resets, Stats controls and failed callbacks are excluded.
        pub completed_role_inputs: u64,
        pub failed_physical_role_calls: u64,
        pub invalid_role_outputs: u64,
        pub delivered_role_inputs: u64,
        pub search_consumed_role_inputs: u64,
        pub canceled_requests: u64,
        pub expired_requests: u64,
        pub completed_new_game_resets: u64,
        pub process_epoch: u64,
        pub game_generation: u64,
        /// Actual immutable deployment epoch used by both scope and input keys.
        pub frozen_epoch: u64,
        pub request_high_water: u64,
        pub execution_high_water: u64,
        pub physical_runs_in_flight: u64,
        pub quarantined: bool,
        pub physical_shutdown_confirmed: bool,
        /// Native backend closure/session destruction is confirmed by a join;
        /// this does not attest destruction of unrelated Rust search records.
        pub native_buffers_released: bool,
        pub model_epoch: [u8; 32],
        pub export_manifest_sha256: [u8; 32],
        pub encoding_semantic_sha256: [u8; 32],
        pub adapter_source_sha256: [u8; 32],
        /// Export declaration, not independent proof that optimizer steps ran.
        pub trained: bool,
        pub residency: PalsSessionResidency,
        pub last_failure: Option<NativeFailureReceipt>,
        /// None means unobserved. A pre-shutdown cache count is not a claim that
        /// native cache buffers survive the subsequent confirmed worker join.
        pub backend_stats: Option<NativeBackendStatsReceipt>,
        pub backend_stats_observation: Option<&'static str>,
        pub execution: NativeExecutionReceipt,
        pub startup_probe: Option<NativeStartupProbeReceipt>,
        /// None means not observed/applicable, not a failed or successful audit.
        pub final_runtime_mapping_confirmed: Option<bool>,
        pub observer_failures: u64,
        pub last_observer_failure: Option<NativeFailureReceipt>,
    }
    impl NativeRoleFinishHandle {
        pub fn receipt(&self) -> NativeRoleReceipt {
            let backend_stats = self
                .owner
                .final_stats
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let backend_stats_observation = backend_stats
                .as_ref()
                .map(|_| "exclusive_worker_before_shutdown");
            NativeRoleReceipt {
                physically_completed_role_calls: self
                    .owner
                    .physical_completed
                    .load(Ordering::Acquire),
                completed_role_inputs: self.owner.completed.load(Ordering::Acquire),
                failed_physical_role_calls: self.owner.physical_failed.load(Ordering::Acquire),
                invalid_role_outputs: self.owner.validation_failed.load(Ordering::Acquire),
                delivered_role_inputs: self.owner.delivered.load(Ordering::Acquire),
                search_consumed_role_inputs: self.owner.consumed.load(Ordering::Acquire),
                canceled_requests: self.owner.canceled.load(Ordering::Acquire),
                expired_requests: self.owner.expired.load(Ordering::Acquire),
                completed_new_game_resets: self.owner.new_game_resets.load(Ordering::Acquire),
                process_epoch: self.owner.process_epoch.0,
                game_generation: self.owner.game_generation.load(Ordering::Acquire),
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                request_high_water: self.owner.request_high_water.load(Ordering::Acquire),
                execution_high_water: self.owner.execution_high_water.load(Ordering::Acquire),
                physical_runs_in_flight: self.owner.in_flight.load(Ordering::Acquire),
                quarantined: self.owner.quarantined.load(Ordering::Acquire),
                physical_shutdown_confirmed: self.owner.shutdown.load(Ordering::Acquire),
                native_buffers_released: self.owner.shutdown.load(Ordering::Acquire)
                    && !self.owner.quarantined.load(Ordering::Acquire)
                    && self.owner.in_flight.load(Ordering::Acquire) == 0,
                model_epoch: self.owner.model_epoch,
                export_manifest_sha256: self.owner.manifest_digest,
                encoding_semantic_sha256: self.owner.encoding_semantic_digest,
                adapter_source_sha256: self.owner.adapter_source_digest,
                trained: self.owner.trained,
                residency: self.owner.residency.clone(),
                last_failure: self
                    .owner
                    .last_failure
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone(),
                backend_stats,
                backend_stats_observation,
                execution: self.owner.execution.clone(),
                startup_probe: self
                    .owner
                    .startup_probe
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone(),
                final_runtime_mapping_confirmed: self
                    .owner
                    .final_mapping_confirmed
                    .load(Ordering::Acquire)
                    .then_some(true),
                observer_failures: self.owner.observer_failures.load(Ordering::Acquire),
                last_observer_failure: self
                    .owner
                    .last_observer_failure
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone(),
            }
        }
        fn collect_final_stats(&self, until: Instant) -> Result<(), RoleError> {
            loop {
                if self.owner.quarantined.load(Ordering::Acquire) {
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                if self
                    .owner
                    .final_stats
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_some()
                {
                    return Ok(());
                }
                let mut slot = self
                    .owner
                    .stats_lease
                    .lock()
                    .map_err(|_| RoleError::PhysicalCompletionUnknown)?;
                if slot.is_none() && self.owner.in_flight.load(Ordering::Acquire) == 0 {
                    let lease = self
                        .owner
                        .worker
                        .lock()
                        .map_err(|_| RoleError::PhysicalCompletionUnknown)?
                        .submit(
                            if self.owner.execution.provider == "cuda"
                                && !self.owner.final_mapping_confirmed.load(Ordering::Acquire)
                            {
                                PalsNativeCommand::VerifyRuntime
                            } else {
                                PalsNativeCommand::SnapshotStats
                            },
                        )
                        .map_err(|error| {
                            self.owner.remember_failure(&error);
                            model_error(error)
                        })?;
                    self.owner.in_flight.fetch_add(1, Ordering::AcqRel);
                    *slot = Some(lease);
                }
                if let Some(lease) = slot.as_mut() {
                    match lease.poll() {
                        PhysicalPoll::Ready(result) => {
                            self.owner.in_flight.fetch_sub(1, Ordering::AcqRel);
                            *slot = None;
                            match result {
                                Ok(PalsNativeResult::RuntimeVerified)
                                    if self.owner.execution.provider == "cuda" =>
                                {
                                    self.owner
                                        .final_mapping_confirmed
                                        .store(true, Ordering::Release);
                                }
                                Ok(PalsNativeResult::Stats(stats)) => {
                                    stats.validate().map_err(|error| {
                                        self.owner.remember_failure(&error);
                                        model_error(error)
                                    })?;
                                    *self
                                        .owner
                                        .final_stats
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner()) = Some(stats.into());
                                    return Ok(());
                                }
                                Ok(_) => {
                                    return Err(RoleError::Backend(
                                        "PALS Stats lease received a different control response"
                                            .into(),
                                    ));
                                }
                                Err(error) => {
                                    self.owner.remember_failure(&error);
                                    return Err(model_error(error));
                                }
                            }
                        }
                        PhysicalPoll::Quarantined | PhysicalPoll::Consumed => {
                            if let Ok(Some(error)) = lease.quarantine_cause() {
                                self.owner.remember_failure(&error);
                            }
                            self.owner.quarantined.store(true, Ordering::Release);
                            return Err(RoleError::PhysicalCompletionUnknown);
                        }
                        PhysicalPoll::Pending => {}
                    }
                }
                if Instant::now() >= until {
                    // The owner retains an admitted Stats lease on timeout, or
                    // the role runtime retains the preceding active role lease.
                    self.owner.quarantined.store(true, Ordering::Release);
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                drop(slot);
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        pub fn finish(&self, until: Instant) -> Result<NativeRoleReceipt, RoleError> {
            self.owner.finishing.store(true, Ordering::Release);
            if self.owner.quarantined.load(Ordering::Acquire) {
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            if self.owner.shutdown.load(Ordering::Acquire) {
                let receipt = self.receipt();
                return if receipt.backend_stats.is_some() {
                    Ok(receipt)
                } else {
                    Err(RoleError::Backend(
                        "PALS shutdown completed without an accepted final backend snapshot".into(),
                    ))
                };
            }
            let stats_failure = match self.collect_final_stats(until) {
                Ok(()) => None,
                Err(RoleError::PhysicalCompletionUnknown) => {
                    self.owner.quarantined.store(true, Ordering::Release);
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                // A failed but physically completed Stats control does not
                // authorize metrics. Still join/drop the idle native owner in
                // the same cleanup window and preserve the original failure.
                Err(error) => Some(error),
            };
            loop {
                if self.owner.quarantined.load(Ordering::Acquire) {
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                let outcome = self
                    .owner
                    .worker
                    .lock()
                    .map_err(|_| RoleError::Backend("PALS worker shutdown lock poisoned".into()))?
                    .try_shutdown();
                match outcome {
                    Poll::Ready(Ok(())) => {
                        if self.owner.in_flight.load(Ordering::Acquire) != 0 {
                            self.owner.quarantined.store(true, Ordering::Release);
                            return Err(RoleError::PhysicalCompletionUnknown);
                        }
                        self.owner.shutdown.store(true, Ordering::Release);
                        return stats_failure.map_or_else(|| Ok(self.receipt()), Err);
                    }
                    Poll::Ready(Err(error)) => {
                        self.owner.remember_failure(&error);
                        self.owner.quarantined.store(true, Ordering::Release);
                        return Err(RoleError::PhysicalCompletionUnknown);
                    }
                    Poll::Pending if Instant::now() >= until => {
                        self.owner.quarantined.store(true, Ordering::Release);
                        return Err(RoleError::PhysicalCompletionUnknown);
                    }
                    Poll::Pending => std::thread::sleep(Duration::from_millis(1)),
                }
            }
        }
    }
    struct WorkerBackend {
        owner: Arc<WorkerOwner>,
    }
    struct Lease {
        physical: NativePhysicalLease,
        request: Arc<RuntimeRoleRequest<PalsModelInput>>,
        execution: ExecutionId,
    }
    impl Backend<PalsAdapter<PalsModelInput, ContractSystemClock>> for WorkerBackend {
        type Lease = Lease;
        fn additional_resources(&self, _: &[Arc<RuntimeRoleRequest<PalsModelInput>>]) -> Resources {
            Resources {
                host_bytes: REQUEST_BYTES,
                device_bytes: self.owner.execution.transient_execution_device_bytes,
                pinned_bytes: 0,
            }
        }
        fn dispatch(
            &mut self,
            execution: &ExecutionId,
            requests: &[Arc<RuntimeRoleRequest<PalsModelInput>>],
        ) -> Result<Self::Lease, ContractError> {
            if requests.len() != 1
                || self.owner.quarantined.load(Ordering::Acquire)
                || self.owner.finishing.load(Ordering::Acquire)
                || self.owner.shutdown.load(Ordering::Acquire)
            {
                return Err(fault(
                    ErrorCode::BackendUnavailable,
                    "PALS native worker is unavailable or batch is unsupported",
                ));
            }
            let request = Arc::clone(&requests[0]);
            // One bounded owned copy crosses SingleWorker's immutable input slot;
            // it is reserved separately and survives logical consumer cancellation.
            let input = request.role().state.as_ref().clone();
            let physical = self
                .owner
                .worker
                .lock()
                .map_err(|_| {
                    fault(
                        ErrorCode::BackendFailure,
                        "PALS physical owner lock poisoned",
                    )
                })?
                .submit(PalsNativeCommand::Evaluate(input))
                .map_err(|_| {
                    fault(
                        ErrorCode::BackendFailure,
                        "PALS physical worker rejected native admission",
                    )
                })?;
            self.owner.in_flight.fetch_add(1, Ordering::AcqRel);
            self.owner
                .execution_high_water
                .store(execution.sequence, Ordering::Release);
            Ok(Lease {
                physical,
                request,
                execution: *execution,
            })
        }
        fn poll(
            &mut self,
            lease: &mut Lease,
        ) -> Poll<Vec<BackendResult<PalsAdapter<PalsModelInput, ContractSystemClock>>>> {
            let result = match lease.physical.poll() {
                PhysicalPoll::Pending => return Poll::Pending,
                PhysicalPoll::Quarantined | PhysicalPoll::Consumed => {
                    if let Ok(Some(error)) = lease.physical.quarantine_cause() {
                        self.owner.remember_failure(&error);
                    }
                    self.owner.quarantined.store(true, Ordering::Release);
                    return Poll::Pending;
                }
                PhysicalPoll::Ready(result) => result,
            };
            self.owner.in_flight.fetch_sub(1, Ordering::AcqRel);
            self.owner.physical_completed.fetch_add(1, Ordering::AcqRel);
            let observer_failed = match &result {
                Ok(PalsNativeResult::Evaluation(raw)) => self
                    .owner
                    .observe("physical_return", |observer| {
                        observer.physically_completed(lease.request.role().id, Ok(raw))
                    })
                    .is_err(),
                Err(error) => self
                    .owner
                    .observe("physical_return", |observer| {
                        observer.physically_completed(lease.request.role().id, Err(error))
                    })
                    .is_err(),
                _ => false,
            };
            let raw = match result {
                Ok(PalsNativeResult::Evaluation(raw)) => {
                    self.owner.completed.fetch_add(1, Ordering::AcqRel);
                    Ok(raw)
                }
                Ok(
                    PalsNativeResult::NewGame
                    | PalsNativeResult::Stats(_)
                    | PalsNativeResult::RuntimeVerified
                    | PalsNativeResult::CudaPlacementVerified(_),
                ) => {
                    self.owner.validation_failed.fetch_add(1, Ordering::AcqRel);
                    Err(fault(
                        ErrorCode::UnsupportedContract,
                        "PALS role lease received a control response",
                    ))
                }
                Err(error) => {
                    self.owner.remember_failure(&error);
                    self.owner.physical_failed.fetch_add(1, Ordering::AcqRel);
                    Err(fault(
                        ErrorCode::BackendFailure,
                        "PALS native role execution failed after physical completion",
                    ))
                }
            };
            let raw_was_successful = raw.is_ok();
            let output = raw.and_then(|raw| {
                let request = lease.request.role();
                let decoded = raw
                    .decode(&request.state, &PalsModelConfig::baseline())
                    .map_err(|_| {
                        fault(
                            ErrorCode::NumericalFailure,
                            "PALS native role head validation failed",
                        )
                    })?;
                let wdl = Wdl::try_new(decoded.wdl[0], decoded.wdl[1], decoded.wdl[2], 1e-4)?;
                let payload = match request.role {
                    Role::Proposer => RolePayload::Proposal {
                        candidate_logits: raw.candidate_logits,
                        wdl,
                    },
                    Role::Critic => RolePayload::Counterexample {
                        candidate_logits: raw.candidate_logits,
                        divergence_logits: raw.divergence_logits.ok_or_else(|| {
                            fault(
                                ErrorCode::NumericalFailure,
                                "PALS C divergence head missing",
                            )
                        })?,
                        wdl,
                    },
                    Role::Validator => {
                        return Err(fault(
                            ErrorCode::UnsupportedContract,
                            "V is absent from PALS product export",
                        ));
                    }
                };
                Ok(PhysicalRoleOutput {
                    execution: lease.execution,
                    output: RoleOutput {
                        id: request.id,
                        authority: request.authority,
                        key: request.key,
                        payload,
                        private_latent: raw.private_latent,
                    },
                })
            });
            if raw_was_successful && output.is_err() {
                self.owner.validation_failed.fetch_add(1, Ordering::AcqRel);
            }
            let output = if observer_failed && output.is_ok() {
                Err(fault(
                    ErrorCode::BackendFailure,
                    "PALS physical result observer failed after confirmed native completion",
                ))
            } else {
                output
            };
            Poll::Ready(vec![BackendResult {
                request_id: lease.request.role().id,
                output,
            }])
        }
    }

    pub struct NativeRoleModel {
        identity: String,
        epoch: ProcessEpoch,
        model_epoch: [u8; 32],
        model: Digest,
        encoding: Digest,
        clock: ContractSystemClock,
        scope: SharedPalsScope,
        runtime: Runtime,
        owner: Arc<WorkerOwner>,
        sequence: u64,
        game: u64,
        unusable: bool,
        drain_limit: Duration,
        pending_new_game: bool,
        control_lease: Option<NativePhysicalLease>,
        startup_attempted: bool,
        delivered_request: Option<RequestId>,
    }
    impl NativeRoleModel {
        pub fn load_pinned(
            export: &std::path::Path,
            expected_export_sha256: &str,
            pin: &rz_eval::runtime_pin::RuntimeLibraryPin,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            drain_limit: Duration,
        ) -> Result<Self, RoleError> {
            let runtime = rz_eval::onnx::OrtRuntime::load(pin).map_err(model_error)?;
            let backend = PalsOnnxBackend::load(export, expected_export_sha256, runtime, config)
                .map_err(model_error)?;
            Self::new_with_drain_limit(backend, drain_limit)
        }
        pub fn load_pinned_with_cuda_control_policy(
            export: &std::path::Path,
            expected_export_sha256: &str,
            pin: &rz_eval::runtime_pin::RuntimeLibraryPin,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            policy: PalsCudaControlPolicy,
            profile_root: &std::path::Path,
            drain_limit: Duration,
        ) -> Result<Self, RoleError> {
            let runtime = rz_eval::onnx::OrtRuntime::load(pin).map_err(model_error)?;
            let backend = PalsOnnxBackend::load_with_cuda_control_policy(
                export,
                expected_export_sha256,
                runtime,
                config,
                policy,
                profile_root,
            )
            .map_err(model_error)?;
            Self::new_with_drain_limit(backend, drain_limit)
        }
        pub fn new(backend: PalsOnnxBackend) -> Result<Self, RoleError> {
            Self::new_with_drain_limit(backend, DEFAULT_DRAIN_LIMIT)
        }
        /// The product's UCI shutdown window supplies the same finite drain
        /// limit. There is no independent 30-second native tail hidden behind
        /// a shorter protocol fence.
        pub fn new_with_drain_limit(
            backend: PalsOnnxBackend,
            drain_limit: Duration,
        ) -> Result<Self, RoleError> {
            if drain_limit.is_zero() || Instant::now().checked_add(drain_limit).is_none() {
                return Err(RoleError::Backend(
                    "PALS drain limit must be positive and finite".into(),
                ));
            }
            let execution = execution_receipt(&backend)?;
            let is_cuda = execution.provider == "cuda";
            if !is_cuda {
                backend.verify_runtime().map_err(model_error)?;
            }
            // CUDA load already verifies pinned dependencies. The full origin
            // audit requires the first physical Run, performed by this same
            // worker in prepare_startup before protocol readiness is served.
            verify_pals_rules_profile(
                backend.rules_input_profile(),
                backend.rules_input_semantic_sha256(),
            )?;
            let model_epoch = backend.model_epoch();
            let model = Digest(backend.manifest_digest());
            let mut encoding_hasher = Sha256::new();
            encoding_hasher.update(PALS_ENCODING_SCHEMA);
            encoding_hasher.update(pals_rules_encoding_semantic_digest());
            let encoding = Digest(encoding_hasher.finalize().into());
            let adapter_source_digest = pals_native_source_digest();
            let trained = backend.is_trained();
            let residency = backend.residency().clone();
            let mut identity = format!(
                "pals-onnx-pc-fp32-{}",
                model
                    .0
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            );
            if is_cuda {
                identity.push_str(&format!(
                    "-cuda-device{}-arena{}",
                    execution.device_id.unwrap_or(-1),
                    execution.session_arena_bytes.unwrap_or(0)
                ));
            }
            let epoch = ProcessEpoch(next(&EPOCHS)?);
            let clock = ContractSystemClock::new(epoch);
            let authority = SearchAuthority {
                epoch,
                game: GameGeneration(1),
                root: RootGeneration(0),
                implementation: Digest(adapter_source_digest),
            };
            let scope = SharedPalsScope::new(PalsScope {
                authority,
                model,
                encoding,
                precision: PrecisionProfile::Fp32,
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                mode: ExecutionMode::Deployment,
            });
            let owner = Arc::new(WorkerOwner {
                worker: Mutex::new(backend.controlled_worker().map_err(model_error)?),
                physical_completed: AtomicU64::new(0),
                completed: AtomicU64::new(0),
                physical_failed: AtomicU64::new(0),
                validation_failed: AtomicU64::new(0),
                delivered: AtomicU64::new(0),
                consumed: AtomicU64::new(0),
                canceled: AtomicU64::new(0),
                expired: AtomicU64::new(0),
                new_game_resets: AtomicU64::new(0),
                in_flight: AtomicU64::new(0),
                game_generation: AtomicU64::new(1),
                request_high_water: AtomicU64::new(0),
                execution_high_water: AtomicU64::new(0),
                quarantined: AtomicBool::new(false),
                finishing: AtomicBool::new(false),
                shutdown: AtomicBool::new(false),
                model_epoch,
                manifest_digest: model.0,
                encoding_semantic_digest: encoding.0,
                adapter_source_digest,
                trained,
                residency,
                process_epoch: epoch,
                last_failure: Mutex::new(None),
                final_stats: Mutex::new(None),
                stats_lease: Mutex::new(None),
                execution: execution.clone(),
                startup_probe: Mutex::new(is_cuda.then_some(NativeStartupProbeReceipt {
                    completed_proposer_calls: 0,
                    completed_critic_calls: 0,
                    runtime_mapping_confirmed: false,
                    cuda_placement_witness: None,
                    reset_completed: false,
                    backend_stats: None,
                })),
                startup_ready: AtomicBool::new(!is_cuda),
                final_mapping_confirmed: AtomicBool::new(false),
                observer: Mutex::new(None),
                observer_failures: AtomicU64::new(0),
                last_observer_failure: Mutex::new(None),
            });
            let runtime = PalsRuntime::new(
                PalsAdapter::new(scope.clone(), clock.clone()).map_err(model_error)?,
                WorkerBackend {
                    owner: Arc::clone(&owner),
                },
                Limits {
                    max_requests: 1,
                    max_batch_items: 1,
                    max_executions: 1,
                    max_batch_wait: Duration::ZERO,
                    max_queue_age: Duration::from_secs(180),
                    deadline_reserve: Duration::ZERO,
                    memory: Resources {
                        host_bytes: 4 * REQUEST_BYTES,
                        device_bytes: execution
                            .transient_request_device_bytes
                            .checked_add(execution.transient_execution_device_bytes)
                            .ok_or(RoleError::InvalidOutput)?,
                        pinned_bytes: execution.pinned_request_bytes,
                    },
                },
                64,
            )
            .map_err(model_error)?;
            Ok(Self {
                identity,
                epoch,
                model_epoch,
                model,
                encoding,
                clock,
                scope,
                runtime,
                owner,
                sequence: 0,
                game: 1,
                unusable: false,
                drain_limit,
                pending_new_game: false,
                control_lease: None,
                startup_attempted: false,
                delivered_request: None,
            })
        }
        pub fn finish_handle(&self) -> NativeRoleFinishHandle {
            NativeRoleFinishHandle {
                owner: Arc::clone(&self.owner),
            }
        }
        pub fn model_epoch(&self) -> [u8; 32] {
            self.model_epoch
        }
        pub fn source_identity(&self) -> NativeRoleSourceIdentity {
            NativeRoleSourceIdentity {
                checkpoint_sha256: self.model_epoch,
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                export_manifest_sha256: self.owner.manifest_digest,
                encoding_semantic_sha256: self.owner.encoding_semantic_digest,
                adapter_source_sha256: self.owner.adapter_source_digest,
                model_configuration: PalsModelConfig::baseline(),
                trained: self.owner.trained,
                execution: self.owner.execution.clone(),
            }
        }
        pub fn set_observer(
            &mut self,
            observer: Box<dyn NativeRoleObserver>,
        ) -> Result<(), RoleError> {
            if self.runtime.state().executions != 0
                || self.owner.in_flight.load(Ordering::Acquire) != 0
                || self.delivered_request.is_some()
            {
                return Err(RoleError::Unavailable);
            }
            *self
                .owner
                .observer
                .lock()
                .map_err(|_| RoleError::Unavailable)? = Some(observer);
            Ok(())
        }
        /// Close only the most recent delivered, unconsumed output's accounting
        /// token. No physical lease, cancellation, drain or buffer lifetime is
        /// changed; the search owner calls this before its collector drains.
        pub fn close_unconsumed(&mut self, reason: NativeRoleRejection) -> Result<(), RoleError> {
            if let Some(id) = self.delivered_request.take() {
                if let Err(error) = self.owner.observe("search_output_closed", |observer| {
                    observer.rejected(id, reason)
                }) {
                    self.unusable = true;
                    return Err(error);
                }
            }
            Ok(())
        }
        /// CUDA readiness is obtained by the existing exclusive worker, not
        /// by an untracked temporary session or fabricated search consumption.
        pub fn prepare_startup(&mut self, until: Instant) -> Result<(), RoleError> {
            if self.owner.startup_ready.load(Ordering::Acquire) {
                return Ok(());
            }
            if self.startup_attempted || self.unusable {
                return Err(RoleError::Unavailable);
            }
            self.startup_attempted = true;
            let outcome = (|| {
                let position = Position::startpos();
                let legal = position.legal_moves();
                let cancel = AtomicBool::new(false);
                for kind in [NativeQueryKind::Propose, NativeQueryKind::Reply] {
                    let input = prepare_role_input(
                        &RoleQuery {
                            position: &position,
                            legal: &legal,
                            prefix: &[],
                            proposal: &[],
                            counterexample: None,
                            records: &[],
                            revision: 0,
                            deadline: until,
                            cancel: &cancel,
                        },
                        kind,
                        self.model_epoch,
                    )?;
                    if !matches!(
                        self.startup_command(PalsNativeCommand::Evaluate(input), until)?,
                        PalsNativeResult::Evaluation(_)
                    ) {
                        return Err(RoleError::InvalidOutput);
                    }
                }
                if !matches!(
                    self.startup_command(PalsNativeCommand::VerifyRuntime, until)?,
                    PalsNativeResult::RuntimeVerified
                ) {
                    return Err(RoleError::InvalidOutput);
                }
                self.owner
                    .startup_probe
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .as_mut()
                    .ok_or(RoleError::Unavailable)?
                    .runtime_mapping_confirmed = true;
                if self.owner.execution.provider == "cuda" {
                    let witness = match self
                        .startup_command(PalsNativeCommand::VerifyCudaPlacement, until)?
                    {
                        PalsNativeResult::CudaPlacementVerified(witness) => witness,
                        _ => return Err(RoleError::InvalidOutput),
                    };
                    validate_startup_cuda_witness(
                        &witness,
                        &self.owner.execution,
                        self.owner.manifest_digest,
                        &self.owner.residency,
                    )?;
                    self.owner
                        .startup_probe
                        .lock()
                        .map_err(|_| RoleError::Unavailable)?
                        .as_mut()
                        .ok_or(RoleError::Unavailable)?
                        .cuda_placement_witness = Some(witness);
                }
                if !matches!(
                    self.startup_command(PalsNativeCommand::NewGame, until)?,
                    PalsNativeResult::NewGame
                ) {
                    return Err(RoleError::InvalidOutput);
                }
                self.owner
                    .startup_probe
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .as_mut()
                    .ok_or(RoleError::Unavailable)?
                    .reset_completed = true;
                let stats = match self.startup_command(PalsNativeCommand::SnapshotStats, until)? {
                    PalsNativeResult::Stats(stats) => stats,
                    _ => return Err(RoleError::InvalidOutput),
                };
                stats.validate().map_err(model_error)?;
                let cache_entries = stats.live_public_cache_entries;
                self.owner
                    .startup_probe
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .as_mut()
                    .ok_or(RoleError::Unavailable)?
                    .backend_stats = Some(stats.into());
                if cache_entries != 0 {
                    return Err(RoleError::InvalidOutput);
                }
                if Instant::now() >= until {
                    return Err(RoleError::Deadline);
                }
                self.owner.startup_ready.store(true, Ordering::Release);
                Ok(())
            })();
            if outcome.is_err() {
                self.unusable = true;
            }
            outcome
        }
        fn startup_command(
            &mut self,
            command: PalsNativeCommand,
            until: Instant,
        ) -> Result<PalsNativeResult, RoleError> {
            if self.owner.quarantined.load(Ordering::Acquire) {
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            if Instant::now() >= until {
                return Err(RoleError::Deadline);
            }
            if self.runtime.state().executions != 0
                || self.owner.in_flight.load(Ordering::Acquire) != 0
            {
                return Err(RoleError::Unavailable);
            }
            let physical_until = until
                .checked_add(self.drain_limit)
                .ok_or(RoleError::InvalidOutput)?;
            self.control_lease = Some(
                self.owner
                    .worker
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .submit(command)
                    .map_err(model_error)?,
            );
            self.owner.in_flight.fetch_add(1, Ordering::AcqRel);
            loop {
                let lease = self.control_lease.as_mut().expect("admitted startup lease");
                match lease.poll() {
                    PhysicalPoll::Ready(result) => {
                        self.owner.in_flight.fetch_sub(1, Ordering::AcqRel);
                        // The retained immutable job gives the exact input for
                        // output validation without recomputing/duplicating it.
                        let checked = match result {
                            Ok(result) => {
                                if let (
                                    PalsNativeCommand::Evaluate(input),
                                    PalsNativeResult::Evaluation(raw),
                                ) = (lease.input(), &result)
                                {
                                    let checked = raw
                                        .decode(input, &PalsModelConfig::baseline())
                                        .map_err(model_error);
                                    if checked.is_ok() {
                                        let mut probe = self
                                            .owner
                                            .startup_probe
                                            .lock()
                                            .map_err(|_| RoleError::Unavailable)?;
                                        let probe = probe.as_mut().ok_or(RoleError::Unavailable)?;
                                        if input.role == PalsRole::Critic {
                                            probe.completed_critic_calls += 1;
                                        } else {
                                            probe.completed_proposer_calls += 1;
                                        }
                                    }
                                    checked.map(|_| result)
                                } else {
                                    Ok(result)
                                }
                            }
                            Err(error) => {
                                self.owner.remember_failure(&error);
                                Err(model_error(error))
                            }
                        };
                        self.control_lease = None;
                        return checked;
                    }
                    PhysicalPoll::Quarantined | PhysicalPoll::Consumed => {
                        if let Ok(Some(error)) = lease.quarantine_cause() {
                            self.owner.remember_failure(&error);
                        }
                        self.owner.quarantined.store(true, Ordering::Release);
                        return Err(RoleError::PhysicalCompletionUnknown);
                    }
                    PhysicalPoll::Pending => {}
                }
                if Instant::now() >= physical_until {
                    self.owner.quarantined.store(true, Ordering::Release);
                    // Keep the lease and worker closure's session/input pins.
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        fn drain_deadline(&self, observed: Instant, query_until: Instant) -> Instant {
            let observed_end = observed.checked_add(self.drain_limit).unwrap_or(observed);
            query_until
                .checked_add(self.drain_limit)
                .map_or(observed_end, |end| end.min(observed_end))
        }
        fn ensure_new_game(
            &mut self,
            until: Instant,
            canceled: &AtomicBool,
        ) -> Result<(), RoleError> {
            if !self.pending_new_game {
                return Ok(());
            }
            if canceled.load(Ordering::Acquire) {
                return Err(RoleError::Canceled);
            }
            if Instant::now() >= until {
                return Err(RoleError::Deadline);
            }
            if self.runtime.state().executions != 0
                || self.owner.in_flight.load(Ordering::Acquire) != 0
            {
                self.unusable = true;
                self.owner.quarantined.store(true, Ordering::Release);
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            self.control_lease = Some(
                self.owner
                    .worker
                    .lock()
                    .map_err(|_| RoleError::Backend("PALS cache-reset owner lock poisoned".into()))?
                    .submit(PalsNativeCommand::NewGame)
                    .map_err(model_error)?,
            );
            self.owner.in_flight.fetch_add(1, Ordering::AcqRel);
            let mut terminal = None;
            let mut drain_until = None;
            loop {
                let observed = Instant::now();
                if terminal.is_none() && canceled.load(Ordering::Acquire) {
                    terminal = Some(RoleError::Canceled);
                    self.owner.canceled.fetch_add(1, Ordering::AcqRel);
                    drain_until = Some(self.drain_deadline(observed, until));
                } else if terminal.is_none() && observed >= until {
                    terminal = Some(RoleError::Deadline);
                    self.owner.expired.fetch_add(1, Ordering::AcqRel);
                    drain_until = Some(self.drain_deadline(observed, until));
                }
                let outcome = self
                    .control_lease
                    .as_mut()
                    .expect("admitted reset lease")
                    .poll();
                match outcome {
                    PhysicalPoll::Ready(result) => {
                        self.owner.in_flight.fetch_sub(1, Ordering::AcqRel);
                        self.control_lease = None;
                        match result {
                            Ok(PalsNativeResult::NewGame) => {
                                self.pending_new_game = false;
                                self.owner.new_game_resets.fetch_add(1, Ordering::AcqRel);
                                return terminal.map_or(Ok(()), Err);
                            }
                            Ok(
                                PalsNativeResult::Evaluation(_)
                                | PalsNativeResult::Stats(_)
                                | PalsNativeResult::RuntimeVerified
                                | PalsNativeResult::CudaPlacementVerified(_),
                            ) => {
                                self.unusable = true;
                                return Err(RoleError::Backend(
                                    "PALS reset lease received a different response".into(),
                                ));
                            }
                            Err(error) => {
                                self.owner.remember_failure(&error);
                                self.unusable = true;
                                return Err(model_error(error));
                            }
                        }
                    }
                    PhysicalPoll::Quarantined | PhysicalPoll::Consumed => {
                        if let Ok(Some(error)) = self
                            .control_lease
                            .as_ref()
                            .expect("retained reset lease")
                            .quarantine_cause()
                        {
                            self.owner.remember_failure(&error);
                        }
                        self.unusable = true;
                        self.owner.quarantined.store(true, Ordering::Release);
                        // Retain this control lease as well as the worker's
                        // closure/job pins. No clear acknowledgement is invented.
                        return Err(RoleError::PhysicalCompletionUnknown);
                    }
                    PhysicalPoll::Pending => {}
                }
                if drain_until.is_some_and(|end| Instant::now() >= end) {
                    self.unusable = true;
                    self.owner.quarantined.store(true, Ordering::Release);
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        fn run(
            &mut self,
            input: PalsModelInput,
            position: &Position,
            until: Instant,
            canceled: &std::sync::atomic::AtomicBool,
            prepared_context: NativePreparedContext<'_>,
        ) -> Result<RoleOutput, RoleError> {
            if self.owner.quarantined.load(Ordering::Acquire) {
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            if self.unusable
                || !self.owner.startup_ready.load(Ordering::Acquire)
                || self.owner.finishing.load(Ordering::Acquire)
                || self.owner.shutdown.load(Ordering::Acquire)
            {
                return Err(RoleError::Unavailable);
            }
            if canceled.load(Ordering::Acquire) {
                return Err(RoleError::Canceled);
            }
            if Instant::now() >= until {
                return Err(RoleError::Deadline);
            }
            self.ensure_new_game(until, canceled)?;
            self.close_unconsumed(NativeRoleRejection::SupersededUnconsumed)?;
            self.sequence = self
                .sequence
                .checked_add(1)
                .ok_or_else(|| RoleError::Backend("PALS request sequence exhausted".into()))?;
            self.owner
                .request_high_water
                .store(self.sequence, Ordering::Release);
            let mut scope = self.scope.get().ok_or(RoleError::Unavailable)?;
            scope.authority.game = GameGeneration(self.game);
            scope.authority.root = RootGeneration(self.sequence);
            self.scope.update(scope).map_err(model_error)?;
            let now = self.clock.now();
            let remaining = u64::try_from(
                until.saturating_duration_since(Instant::now()).as_nanos(),
            )
            .map_err(|_| RoleError::Backend("PALS deadline outside finite clock domain".into()))?;
            let deadline = Deadline {
                clock: self.clock.domain(),
                at: MonotonicTick(now.0.checked_add(remaining).ok_or_else(|| {
                    RoleError::Backend("PALS monotonic deadline exhausted".into())
                })?),
            };
            let state_owner = OwnerId(next(&RULE_OWNERS)?);
            let exact = ContractPosition::new(state_owner, position.clone())
                .export()
                .map_err(model_error)?;
            let state_identity = exact.snapshot().identity();
            let role = if input.role == PalsRole::Critic {
                Role::Critic
            } else {
                Role::Proposer
            };
            let packed_candidates = input
                .candidates
                .iter()
                .map(|c| c.packed().map_err(model_error))
                .collect::<Result<Vec<_>, _>>()?;
            let key = RepresentationKey {
                input: Digest(
                    input
                        .canonical_input_key(&PalsModelConfig::baseline())
                        .map_err(model_error)?,
                ),
                history: Digest(input.history_digest),
                candidates: Digest(
                    Sha256::digest(
                        packed_candidates
                            .iter()
                            .flat_map(|c| c.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .into(),
                ),
                mask_positions: Digest(
                    Sha256::digest(
                        [
                            input.records.len() as u64,
                            input.candidates.len() as u64,
                            input.divergence_features.len() as u64,
                        ]
                        .into_iter()
                        .flat_map(u64::to_le_bytes)
                        .collect::<Vec<_>>(),
                    )
                    .into(),
                ),
                model: self.model,
                encoding: self.encoding,
                precision: PrecisionProfile::Fp32,
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                record_revision: input.situation_revision,
                role,
            };
            let candidates: Vec<_> = packed_candidates
                .into_iter()
                .map(|bits| Move16::try_from_bits(bits).map_err(model_error))
                .collect::<Result<_, _>>()?;
            let public_records: Vec<_> = input
                .records
                .iter()
                .map(|r| {
                    let mut h = Sha256::new();
                    h.update(r.record_id.to_le_bytes());
                    h.update(r.revision.to_le_bytes());
                    h.update([u8::from(r.critical)]);
                    for value in r.features {
                        h.update(value.to_bits().to_le_bytes());
                    }
                    Digest(h.finalize().into())
                })
                .collect();
            let cancel = CancelToken::new();
            let id = RequestId::new(self.epoch, self.sequence);
            let divergence_count = u16::try_from(input.divergence_features.len())
                .map_err(|_| RoleError::InvalidOutput)?;
            let situation = SituationHandle {
                slot: 0,
                generation: self.sequence,
            };
            let request = Arc::new(RoleRequest {
                revision: PALS_CONTRACT_REVISION,
                id,
                authority: scope.authority,
                situation,
                state_identity,
                state: Arc::new(input),
                key,
                role,
                mode: ExecutionMode::Deployment,
                candidates: candidates.into(),
                public_records: public_records.into(),
                max_records: 128,
                recurrent_steps: 2,
                divergence_count,
                deadline,
                cancel: cancel.clone(),
                resources: ByteBudget {
                    host: REQUEST_BYTES,
                    device: self.owner.execution.transient_request_device_bytes,
                    pinned: self.owner.execution.pinned_request_bytes,
                },
            });
            let representation = SharedRepresentationScope::new(RepresentationScope {
                situation,
                state: state_identity,
                key,
            });
            self.owner.observe("prepared", |observer| {
                observer.prepared(id, request.state.as_ref(), prepared_context)
            })?;
            if let Err(error) = self
                .runtime
                .submit(RuntimeRoleRequest::new(request, representation))
            {
                let _ = self.owner.observe("admission", |observer| {
                    observer.rejected(id, NativeRoleRejection::Admission)
                });
                return Err(model_error(error));
            }
            let mut outcome = None;
            let mut drain_until = None;
            loop {
                if canceled.load(Ordering::Acquire) && !cancel.is_canceled() {
                    drain_until = Some(self.drain_deadline(Instant::now(), until));
                    cancel.cancel();
                    self.owner.canceled.fetch_add(1, Ordering::AcqRel);
                    self.runtime.cancel(id).map_err(model_error)?;
                }
                if let Some(terminal) = self.runtime.poll() {
                    outcome = Some(match terminal {
                        PalsTerminal::Completed(output) => Ok(output.output),
                        PalsTerminal::Canceled(_) => Err(RoleError::Canceled),
                        PalsTerminal::Expired(_) => {
                            self.owner.expired.fetch_add(1, Ordering::AcqRel);
                            Err(RoleError::Deadline)
                        }
                        PalsTerminal::Stale(_) => Err(RoleError::Backend(
                            "PALS native result has stale authority".into(),
                        )),
                        PalsTerminal::Failed { error, .. } => Err(model_error(error)),
                    });
                    drain_until.get_or_insert(self.drain_deadline(Instant::now(), until));
                }
                if self.owner.quarantined.load(Ordering::Acquire) {
                    self.unusable = true;
                    cancel.cancel();
                    let _ = self.runtime.cancel(id);
                    self.runtime.pump();
                    let _ = self.owner.observe("physical_unknown", |observer| {
                        observer.rejected(id, NativeRoleRejection::PhysicalCompletionUnknown)
                    });
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                if outcome.is_some() && self.runtime.state().executions == 0 {
                    let result = outcome.take().expect("terminal outcome");
                    if canceled.load(Ordering::Acquire) {
                        let _ = self.owner.observe("canceled", |observer| {
                            observer.rejected(id, NativeRoleRejection::Canceled)
                        });
                        return Err(RoleError::Canceled);
                    }
                    if Instant::now() >= until {
                        let _ = self.owner.observe("deadline", |observer| {
                            observer.rejected(id, NativeRoleRejection::Deadline)
                        });
                        return Err(RoleError::Deadline);
                    }
                    if result.is_ok() {
                        self.owner
                            .observe("delivered", |observer| observer.delivered(id))?;
                        if canceled.load(Ordering::Acquire) || Instant::now() >= until {
                            let (reason, error) = if canceled.load(Ordering::Acquire) {
                                (NativeRoleRejection::Canceled, RoleError::Canceled)
                            } else {
                                (NativeRoleRejection::Deadline, RoleError::Deadline)
                            };
                            let _ = self.owner.observe("delivery_expired", |observer| {
                                observer.rejected(id, reason)
                            });
                            return Err(error);
                        }
                        self.owner.delivered.fetch_add(1, Ordering::AcqRel);
                        self.delivered_request = Some(id);
                    } else {
                        let reason = match &result {
                            Err(RoleError::Canceled) => NativeRoleRejection::Canceled,
                            Err(RoleError::Deadline) => NativeRoleRejection::Deadline,
                            Err(RoleError::InvalidOutput) => NativeRoleRejection::InvalidOutput,
                            Err(RoleError::PhysicalCompletionUnknown) => {
                                NativeRoleRejection::PhysicalCompletionUnknown
                            }
                            _ => NativeRoleRejection::Backend,
                        };
                        let _ = self
                            .owner
                            .observe("rejected", |observer| observer.rejected(id, reason));
                    }
                    return result;
                }
                if drain_until.is_some_and(|deadline| Instant::now() >= deadline) {
                    self.unusable = true;
                    self.owner.quarantined.store(true, Ordering::Release);
                    let now = self.clock.now();
                    let _ = self.runtime.begin_shutdown(Deadline {
                        clock: self.clock.domain(),
                        at: now,
                    });
                    let _ = self.owner.observe("physical_unknown", |observer| {
                        observer.rejected(id, NativeRoleRejection::PhysicalCompletionUnknown)
                    });
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        fn evaluate(
            &mut self,
            query: RoleQuery<'_>,
            kind: NativeQueryKind,
        ) -> Result<RoleEvaluation, RoleError> {
            let input = prepare_role_input(&query, kind, self.model_epoch)?;
            let output = self.run(
                input,
                query.position,
                query.deadline,
                query.cancel,
                NativePreparedContext::Role {
                    query: &query,
                    kind,
                },
            )?;
            match output.payload {
                RolePayload::Proposal {
                    candidate_logits,
                    wdl,
                }
                | RolePayload::Counterexample {
                    candidate_logits,
                    wdl,
                    ..
                } => Ok(RoleEvaluation {
                    logits: candidate_logits,
                    wdl: wdl.probabilities(),
                }),
                _ => Err(RoleError::InvalidOutput),
            }
        }
    }
    impl RoleModel for NativeRoleModel {
        fn identity(&self) -> &str {
            &self.identity
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.evaluate(query, NativeQueryKind::Propose)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.evaluate(query, NativeQueryKind::Reply)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.evaluate(query, NativeQueryKind::Repair)
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            let input = prepare_divergence_input(&query, self.model_epoch)?;
            let output = self.run(
                input,
                query.root,
                query.deadline,
                query.cancel,
                NativePreparedContext::Divergence { query: &query },
            )?;
            match output.payload {
                RolePayload::Counterexample {
                    divergence_logits, ..
                } if divergence_logits.len() == query.candidates.len() => Ok(divergence_logits),
                _ => Err(RoleError::InvalidOutput),
            }
        }
        fn new_game(&mut self) {
            let _ = self.close_unconsumed(NativeRoleRejection::GameReset);
            match self.game.checked_add(1) {
                Some(value) => {
                    self.game = value;
                    self.owner.game_generation.store(value, Ordering::Release);
                    self.pending_new_game = true;
                }
                None => self.unusable = true,
            }
            // Clear on the same physical worker before the next NN admission;
            // epoch and execution/request high-water marks never reset.
        }
        fn accepted_output(&mut self) {
            // The search owner calls this only after its late/cancel/shape gate;
            // delivering a result or clearing a cache never creates consumption.
            if let Some(id) = self.delivered_request.take() {
                self.owner.consumed.fetch_add(1, Ordering::AcqRel);
                if self
                    .owner
                    .observe("accepted", |observer| observer.accepted(id))
                    .is_err()
                {
                    self.unusable = true;
                }
            }
        }
        fn finish_search(&mut self, reason: RoleSearchClosure) {
            let reason = match reason {
                RoleSearchClosure::Completed => NativeRoleRejection::SearchFinishedUnconsumed,
                RoleSearchClosure::Canceled => NativeRoleRejection::Canceled,
                RoleSearchClosure::Deadline => NativeRoleRejection::Deadline,
                RoleSearchClosure::Failed => NativeRoleRejection::SearchFailedUnconsumed,
                RoleSearchClosure::PhysicalCompletionUnknown => {
                    NativeRoleRejection::PhysicalCompletionUnknown
                }
            };
            let _ = self.close_unconsumed(reason);
        }
    }

    #[cfg(test)]
    mod lifecycle_tests {
        use super::*;
        use rz_eval::error::{FailureKind, FailureStage};
        use rz_eval::pals_model::PalsRawOutput;
        use rz_eval::worker::PhysicalRun;

        fn unexpected_cuda_placement() -> PhysicalRun<Result<PalsNativeResult, BackendError>> {
            PhysicalRun::Complete(Err(BackendError::new(
                FailureKind::BackendUnavailable,
                FailureStage::Admission,
                "CPU lifecycle fixture has no CUDA placement witness",
            )))
        }

        #[test]
        fn startup_cuda_witness_requires_owner_pins_and_both_private_branches() {
            use rz_eval::pals_onnx::{PalsGraphPlacement, PalsKernelWitness};
            // Synthetic boundary evidence exercises admission only; it is not
            // a CUDA device, physical kernel or native placement attestation.
            let execution = NativeExecutionReceipt {
                provider: "cuda",
                device_id: Some(0),
                session_arena_bytes: Some(2 * 1024 * 1024 * 1024),
                runtime_sha256: [1; 32],
                runtime_bundle_sha256: Some([2; 32]),
                cuda_control_inventory_sha256: Some([3; 32]),
                transient_request_device_bytes: 0,
                transient_execution_device_bytes: 0,
                pinned_request_bytes: 0,
                device_public_memory: false,
            };
            let graphs: Vec<_> = ["public", "shared_pc"]
                .into_iter()
                .map(|role| rz_eval::pals_onnx::PalsGraphIdentity {
                    role: role.into(),
                    sha256: [4; 32],
                    serialized_bytes: 1,
                })
                .collect();
            let residency = PalsSessionResidency {
                graphs,
                native_sessions: 2,
                layout: "synthetic-boundary-test".into(),
                reader_initializer_bank: None,
                role_reader_weights_shared: None,
                native_resident_parameter_bytes: None,
                vram_peak_bytes: None,
            };
            let kernels = PalsKernelWitness {
                profile_sha256: [5; 32],
                cuda_kernels: 2,
                cuda_transfer_kernels: 0,
                approved_cpu_control_kernels: 1,
                neural_kernels: 2,
                proposer_private_kernels: 1,
                critic_private_kernels: 1,
            };
            let mut witness = PalsCudaPlacementWitness {
                schema: "rovezero.pals-cuda-metadata-control.v2".into(),
                inventory_sha256: [3; 32],
                manifest_sha256: [6; 32],
                runtime_sha256: [1; 32],
                runtime_bundle_sha256: [2; 32],
                optimization: PalsGraphOptimization::Disable,
                initialization: ["public", "shared_pc"]
                    .into_iter()
                    .map(|role| PalsGraphPlacement {
                        role: role.into(),
                        graph_sha256: [4; 32],
                        log_sha256: [7; 32],
                        assigned_nodes: 3,
                        cuda_nodes: 2,
                        approved_cpu_control_nodes: 1,
                        optimization: PalsGraphOptimization::Disable,
                        approved_transfers: Vec::new(),
                        recursive_coverage: "synthetic-boundary-test".into(),
                    })
                    .collect(),
                public: kernels.clone(),
                shared_pc: kernels,
                category_provenance: "synthetic-boundary-test".into(),
            };
            assert!(
                validate_startup_cuda_witness(&witness, &execution, [6; 32], &residency).is_ok()
            );
            assert!(
                validate_startup_cuda_witness(&witness, &execution, [8; 32], &residency).is_err()
            );
            witness.optimization = PalsGraphOptimization::Level1;
            assert!(
                validate_startup_cuda_witness(&witness, &execution, [6; 32], &residency).is_err()
            );
            witness.optimization = PalsGraphOptimization::Disable;
            witness.shared_pc.critic_private_kernels = 0;
            assert!(
                validate_startup_cuda_witness(&witness, &execution, [6; 32], &residency).is_err()
            );
            witness.shared_pc.critic_private_kernels = 1;
            witness.initialization[0].graph_sha256 = [8; 32];
            assert!(
                validate_startup_cuda_witness(&witness, &execution, [6; 32], &residency).is_err()
            );
        }

        fn fixture_model<F>(run: F) -> NativeRoleModel
        where
            F: FnMut(&PalsNativeCommand) -> PhysicalRun<Result<PalsNativeResult, BackendError>>
                + Send
                + 'static,
        {
            let epoch = ProcessEpoch(next(&EPOCHS).unwrap());
            let model = Digest([3; 32]);
            let encoding = Digest(pals_rules_encoding_semantic_digest());
            let clock = ContractSystemClock::new(epoch);
            let scope = SharedPalsScope::new(PalsScope {
                authority: SearchAuthority {
                    epoch,
                    game: GameGeneration(1),
                    root: RootGeneration(0),
                    implementation: Digest(pals_native_source_digest()),
                },
                model,
                encoding,
                precision: PrecisionProfile::Fp32,
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                mode: ExecutionMode::Deployment,
            });
            let owner = Arc::new(WorkerOwner {
                worker: Mutex::new(SingleWorker::spawn_with_outcome(run).unwrap()),
                physical_completed: AtomicU64::new(0),
                completed: AtomicU64::new(0),
                physical_failed: AtomicU64::new(0),
                validation_failed: AtomicU64::new(0),
                delivered: AtomicU64::new(0),
                consumed: AtomicU64::new(0),
                canceled: AtomicU64::new(0),
                expired: AtomicU64::new(0),
                new_game_resets: AtomicU64::new(0),
                in_flight: AtomicU64::new(0),
                game_generation: AtomicU64::new(1),
                request_high_water: AtomicU64::new(0),
                execution_high_water: AtomicU64::new(0),
                quarantined: AtomicBool::new(false),
                shutdown: AtomicBool::new(false),
                model_epoch: [3; 32],
                manifest_digest: [3; 32],
                encoding_semantic_digest: encoding.0,
                adapter_source_digest: pals_native_source_digest(),
                trained: false,
                finishing: AtomicBool::new(false),
                process_epoch: epoch,
                last_failure: Mutex::new(None),
                final_stats: Mutex::new(None),
                stats_lease: Mutex::new(None),
                residency: PalsSessionResidency {
                    graphs: vec![],
                    native_sessions: 0,
                    layout: "native-worker-fixture".into(),
                    reader_initializer_bank: None,
                    role_reader_weights_shared: None,
                    native_resident_parameter_bytes: None,
                    vram_peak_bytes: None,
                },
                execution: NativeExecutionReceipt {
                    provider: "cpu",
                    device_id: None,
                    session_arena_bytes: None,
                    runtime_sha256: [0; 32],
                    runtime_bundle_sha256: None,
                    cuda_control_inventory_sha256: None,
                    transient_request_device_bytes: 0,
                    transient_execution_device_bytes: 0,
                    pinned_request_bytes: 0,
                    device_public_memory: false,
                },
                startup_probe: Mutex::new(None),
                startup_ready: AtomicBool::new(true),
                final_mapping_confirmed: AtomicBool::new(false),
                observer: Mutex::new(None),
                observer_failures: AtomicU64::new(0),
                last_observer_failure: Mutex::new(None),
            });
            let runtime = PalsRuntime::new(
                PalsAdapter::new(scope.clone(), clock.clone()).unwrap(),
                WorkerBackend {
                    owner: Arc::clone(&owner),
                },
                Limits {
                    max_requests: 1,
                    max_batch_items: 1,
                    max_executions: 1,
                    max_batch_wait: Duration::ZERO,
                    max_queue_age: Duration::from_secs(2),
                    deadline_reserve: Duration::ZERO,
                    memory: Resources {
                        host_bytes: 4 * REQUEST_BYTES,
                        device_bytes: 0,
                        pinned_bytes: 0,
                    },
                },
                64,
            )
            .unwrap();
            NativeRoleModel {
                identity: "pals-native-lifecycle-fixture".into(),
                epoch,
                model_epoch: [3; 32],
                model,
                encoding,
                clock,
                scope,
                runtime,
                owner,
                sequence: 0,
                game: 1,
                unusable: false,
                drain_limit: Duration::from_secs(1),
                pending_new_game: false,
                control_lease: None,
                startup_attempted: false,
                delivered_request: None,
            }
        }
        fn output(input: &PalsModelInput) -> PalsNativeResult {
            PalsNativeResult::Evaluation(PalsRawOutput {
                candidate_logits: vec![0.0; input.candidates.len()],
                wdl_logits: [0.0; 3],
                divergence_logits: (input.role == PalsRole::Critic)
                    .then(|| vec![0.0; input.divergence_features.len()]),
                task_logits: None,
                private_latent: vec![0.0; PalsModelConfig::baseline().latent_elements()],
            })
        }
        fn query<'a>(
            position: &'a Position,
            legal: &'a [BoardMove],
            cancel: &'a AtomicBool,
        ) -> RoleQuery<'a> {
            RoleQuery {
                position,
                legal,
                prefix: &[],
                proposal: &[],
                counterexample: None,
                records: &[],
                revision: 0,
                deadline: Instant::now() + Duration::from_secs(2),
                cancel,
            }
        }
        #[test]
        fn transient_cuda_reservations_follow_bounded_tensor_layout() {
            let (request, execution) = transient_device_layout().unwrap();
            assert!(request > 0 && execution > request);
            assert!(request + execution < REQUEST_BYTES);
            // A transient tensor reservation never becomes a session/peak claim.
            assert!(execution < 2 * 1024 * 1024 * 1024);
        }
        #[test]
        fn startup_worker_probe_does_not_create_search_consumption() {
            let mut stats = PalsBackendStats::default();
            let mut cached = false;
            let mut model = fixture_model(move |command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::Evaluate(input) => {
                        stats.admitted_role_requests += 1;
                        if cached {
                            stats.public_cache_hits += 1;
                        } else {
                            stats.public_cache_misses += 1;
                            stats.public_nn_runs_attempted += 1;
                            stats.public_nn_runs_completed += 1;
                            stats.completed_nn_inputs += 1;
                            stats.validated_public_outputs += 1;
                            cached = true;
                        }
                        stats.role_nn_runs_attempted += 1;
                        stats.role_nn_runs_completed += 1;
                        stats.completed_nn_inputs += 1;
                        stats.validated_role_outputs += 1;
                        output(input)
                    }
                    PalsNativeCommand::NewGame => {
                        cached = false;
                        stats.new_game_resets += 1;
                        PalsNativeResult::NewGame
                    }
                    PalsNativeCommand::VerifyCudaPlacement => return unexpected_cuda_placement(),
                    PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                    PalsNativeCommand::SnapshotStats => PalsNativeResult::Stats(stats.clone()),
                }))
            });
            // Only the control-flow is simulated by this CPU worker fixture;
            // it is not provider/device/NN capability evidence.
            model.owner.startup_ready.store(false, Ordering::Release);
            *model.owner.startup_probe.lock().unwrap() = Some(NativeStartupProbeReceipt {
                completed_proposer_calls: 0,
                completed_critic_calls: 0,
                runtime_mapping_confirmed: false,
                cuda_placement_witness: None,
                reset_completed: false,
                backend_stats: None,
            });
            model
                .prepare_startup(Instant::now() + Duration::from_secs(2))
                .unwrap();
            let receipt = model.finish_handle().receipt();
            let startup = receipt.startup_probe.unwrap();
            assert_eq!(startup.completed_proposer_calls, 1);
            assert_eq!(startup.completed_critic_calls, 1);
            assert!(startup.runtime_mapping_confirmed && startup.reset_completed);
            assert_eq!(startup.backend_stats.unwrap().completed_nn_inputs, 3);
            assert_eq!(receipt.completed_role_inputs, 0);
            assert_eq!(receipt.delivered_role_inputs, 0);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
            assert_eq!(receipt.completed_new_game_resets, 0);
            assert_eq!(receipt.request_high_water, 0);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
        }
        struct InputObserver {
            input_keys: Arc<Mutex<Vec<[u8; 32]>>>,
            stages: Arc<Mutex<Vec<(u64, &'static str)>>>,
            fail_physical: bool,
        }
        impl NativeRoleObserver for InputObserver {
            fn prepared(
                &mut self,
                id: RequestId,
                input: &PalsModelInput,
                context: NativePreparedContext<'_>,
            ) -> Result<(), RoleError> {
                let NativePreparedContext::Role { query, .. } = context else {
                    return Err(RoleError::InvalidOutput);
                };
                assert_eq!(input.history_digest, pals_history_digest(query.position)?);
                assert_eq!(input.situation_revision, query.revision);
                self.input_keys.lock().unwrap().push(
                    input
                        .canonical_input_key(&PalsModelConfig::baseline())
                        .map_err(model_error)?,
                );
                self.stages.lock().unwrap().push((id.sequence, "prepared"));
                Ok(())
            }
            fn physically_completed(
                &mut self,
                id: RequestId,
                result: Result<&PalsRawOutput, &BackendError>,
            ) -> Result<(), RoleError> {
                assert!(result.is_ok());
                self.stages.lock().unwrap().push((id.sequence, "physical"));
                if self.fail_physical {
                    Err(RoleError::PhysicalCompletionUnknown)
                } else {
                    Ok(())
                }
            }
            fn delivered(&mut self, id: RequestId) -> Result<(), RoleError> {
                self.stages.lock().unwrap().push((id.sequence, "delivered"));
                Ok(())
            }
            fn accepted(&mut self, id: RequestId) -> Result<(), RoleError> {
                self.stages.lock().unwrap().push((id.sequence, "accepted"));
                Ok(())
            }
        }
        #[test]
        fn observer_sees_the_exact_once_prepared_input_and_authorized_consumption() {
            let actual = Arc::new(Mutex::new(Vec::new()));
            let executed = Arc::clone(&actual);
            let mut model = fixture_model(move |command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::Evaluate(input) => {
                        executed.lock().unwrap().push(
                            input
                                .canonical_input_key(&PalsModelConfig::baseline())
                                .unwrap(),
                        );
                        output(input)
                    }
                    PalsNativeCommand::NewGame => PalsNativeResult::NewGame,
                    PalsNativeCommand::VerifyCudaPlacement => return unexpected_cuda_placement(),
                    PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                    PalsNativeCommand::SnapshotStats => {
                        PalsNativeResult::Stats(PalsBackendStats::default())
                    }
                }))
            });
            let observed = Arc::new(Mutex::new(Vec::new()));
            let stages = Arc::new(Mutex::new(Vec::new()));
            model
                .set_observer(Box::new(InputObserver {
                    input_keys: Arc::clone(&observed),
                    stages: Arc::clone(&stages),
                    fail_physical: false,
                }))
                .unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            model.accepted_output();
            model.accepted_output(); // Duplicate semantic acceptance cannot consume twice.
            assert_eq!(*actual.lock().unwrap(), *observed.lock().unwrap());
            assert_eq!(
                *stages.lock().unwrap(),
                [
                    (1, "prepared"),
                    (1, "physical"),
                    (1, "delivered"),
                    (1, "accepted")
                ]
            );
            assert_eq!(
                model.finish_handle().receipt().search_consumed_role_inputs,
                1
            );
            model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
        }
        #[test]
        fn observer_error_cannot_reclassify_confirmed_physical_work_as_unknown() {
            let mut model = fixture_model(|command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::Evaluate(input) => output(input),
                    PalsNativeCommand::NewGame => PalsNativeResult::NewGame,
                    PalsNativeCommand::VerifyCudaPlacement => return unexpected_cuda_placement(),
                    PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                    PalsNativeCommand::SnapshotStats => {
                        PalsNativeResult::Stats(PalsBackendStats::default())
                    }
                }))
            });
            model
                .set_observer(Box::new(InputObserver {
                    input_keys: Arc::new(Mutex::new(Vec::new())),
                    stages: Arc::new(Mutex::new(Vec::new())),
                    fail_physical: true,
                }))
                .unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::Backend(_))
            ));
            let receipt = model.finish_handle().receipt();
            assert_eq!(receipt.completed_role_inputs, 1);
            assert_eq!(receipt.observer_failures, 1);
            assert_eq!(receipt.delivered_role_inputs, 0);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(!receipt.quarantined);
            assert!(receipt.last_observer_failure.is_some());
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert!(done.physical_shutdown_confirmed && done.native_buffers_released);
        }
        #[test]
        fn search_closure_rejects_unconsumed_delivery_before_the_next_root() {
            struct ClosureObserver(Arc<Mutex<Vec<(u64, NativeRoleRejection)>>>);
            impl NativeRoleObserver for ClosureObserver {
                fn prepared(
                    &mut self,
                    _: RequestId,
                    _: &PalsModelInput,
                    _: NativePreparedContext<'_>,
                ) -> Result<(), RoleError> {
                    Ok(())
                }
                fn rejected(
                    &mut self,
                    id: RequestId,
                    reason: NativeRoleRejection,
                ) -> Result<(), RoleError> {
                    self.0.lock().unwrap().push((id.sequence, reason));
                    Ok(())
                }
            }
            let mut model = fixture_model(|command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::Evaluate(input) => output(input),
                    PalsNativeCommand::NewGame => PalsNativeResult::NewGame,
                    PalsNativeCommand::VerifyCudaPlacement => return unexpected_cuda_placement(),
                    PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                    PalsNativeCommand::SnapshotStats => {
                        PalsNativeResult::Stats(PalsBackendStats::default())
                    }
                }))
            });
            let rejected = Arc::new(Mutex::new(Vec::new()));
            model
                .set_observer(Box::new(ClosureObserver(Arc::clone(&rejected))))
                .unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            model.finish_search(RoleSearchClosure::Deadline);
            model.finish_search(RoleSearchClosure::Deadline);
            let receipt = model.finish_handle().receipt();
            assert_eq!(
                *rejected.lock().unwrap(),
                [(1, NativeRoleRejection::Deadline)]
            );
            assert!(model.delivered_request.is_none());
            assert_eq!(receipt.physically_completed_role_calls, 1);
            assert_eq!(receipt.delivered_role_inputs, 1);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
            assert_eq!(receipt.canceled_requests, 0);
            assert_eq!(receipt.expired_requests, 0);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(!receipt.quarantined && !receipt.physical_shutdown_confirmed);
            model.new_game();
            model.propose(query(&position, &legal, &cancel)).unwrap();
            model.accepted_output();
            model.finish_search(RoleSearchClosure::Completed);
            assert_eq!(rejected.lock().unwrap().len(), 1);
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(done.search_consumed_role_inputs, 1);
            assert!(done.physical_shutdown_confirmed && done.native_buffers_released);
        }
        #[test]
        fn reset_ack_and_consumption_are_not_extra_nn_executions() {
            let resets = Arc::new(AtomicU64::new(0));
            let observed = Arc::clone(&resets);
            let mut model = fixture_model(move |command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::Evaluate(input) => output(input),
                    PalsNativeCommand::NewGame => {
                        observed.fetch_add(1, Ordering::SeqCst);
                        PalsNativeResult::NewGame
                    }
                    PalsNativeCommand::SnapshotStats => {
                        PalsNativeResult::Stats(PalsBackendStats::default())
                    }
                    PalsNativeCommand::VerifyCudaPlacement => return unexpected_cuda_placement(),
                    PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                }))
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            model.accepted_output();
            model.new_game();
            assert_eq!(resets.load(Ordering::SeqCst), 0);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            let receipt = model.finish_handle().receipt();
            assert_eq!(resets.load(Ordering::SeqCst), 1);
            assert_eq!(receipt.completed_new_game_resets, 1);
            assert_eq!(receipt.physically_completed_role_calls, 2);
            assert_eq!(receipt.completed_role_inputs, 2);
            assert_eq!(receipt.delivered_role_inputs, 2);
            assert_eq!(receipt.search_consumed_role_inputs, 1);
            assert_eq!(receipt.game_generation, 2);
            assert_eq!(receipt.request_high_water, 2);
            assert_eq!(receipt.execution_high_water, 2);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(receipt.backend_stats.is_none());
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert!(done.physical_shutdown_confirmed && done.native_buffers_released);
            assert!(done.backend_stats.is_some());
            assert_eq!(
                done.backend_stats_observation,
                Some("exclusive_worker_before_shutdown")
            );
            assert_eq!(done.completed_role_inputs, receipt.completed_role_inputs);
            assert_eq!(
                done.physically_completed_role_calls,
                receipt.physically_completed_role_calls
            );
        }
        #[test]
        fn canceled_logical_result_waits_for_physical_completion() {
            let cancel = Arc::new(AtomicBool::new(false));
            let signal = Arc::clone(&cancel);
            let mut model = fixture_model(move |command| match command {
                PalsNativeCommand::Evaluate(input) => {
                    signal.store(true, Ordering::Release);
                    std::thread::sleep(Duration::from_millis(5));
                    PhysicalRun::Complete(Ok(output(input)))
                }
                PalsNativeCommand::NewGame => PhysicalRun::Complete(Ok(PalsNativeResult::NewGame)),
                PalsNativeCommand::SnapshotStats => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::Stats(PalsBackendStats::default())))
                }
                PalsNativeCommand::VerifyCudaPlacement => unexpected_cuda_placement(),
                PalsNativeCommand::VerifyRuntime => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::RuntimeVerified))
                }
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::Canceled)
            ));
            let receipt = model.finish_handle().receipt();
            assert_eq!(receipt.completed_role_inputs, 1);
            assert_eq!(receipt.delivered_role_inputs, 0);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
            assert_eq!(receipt.canceled_requests, 1);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(!receipt.quarantined);
            model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
        }
        #[test]
        fn physical_failure_and_invalid_head_do_not_count_as_consumed_inputs() {
            for invalid_head in [false, true] {
                let mut model = fixture_model(move |command| match command {
                    PalsNativeCommand::NewGame => {
                        PhysicalRun::Complete(Ok(PalsNativeResult::NewGame))
                    }
                    PalsNativeCommand::SnapshotStats => PhysicalRun::Complete(Ok(
                        PalsNativeResult::Stats(PalsBackendStats::default()),
                    )),
                    PalsNativeCommand::VerifyCudaPlacement => unexpected_cuda_placement(),
                    PalsNativeCommand::VerifyRuntime => {
                        PhysicalRun::Complete(Ok(PalsNativeResult::RuntimeVerified))
                    }
                    PalsNativeCommand::Evaluate(input) => {
                        if invalid_head {
                            let PalsNativeResult::Evaluation(mut raw) = output(input) else {
                                unreachable!()
                            };
                            raw.wdl_logits[0] = f32::NAN;
                            PhysicalRun::Complete(Ok(PalsNativeResult::Evaluation(raw)))
                        } else {
                            PhysicalRun::Complete(Err(BackendError::new(
                                FailureKind::BackendFailure,
                                FailureStage::Backend,
                                "native lifecycle fixture failure",
                            )))
                        }
                    }
                });
                let position = Position::startpos();
                let legal = position.legal_moves();
                let cancel = AtomicBool::new(false);
                assert!(model.propose(query(&position, &legal, &cancel)).is_err());
                let receipt = model.finish_handle().receipt();
                assert_eq!(receipt.physically_completed_role_calls, 1);
                assert_eq!(receipt.completed_role_inputs, u64::from(invalid_head));
                assert_eq!(receipt.failed_physical_role_calls, u64::from(!invalid_head));
                assert_eq!(receipt.invalid_role_outputs, u64::from(invalid_head));
                assert_eq!(receipt.delivered_role_inputs, 0);
                assert_eq!(receipt.search_consumed_role_inputs, 0);
                assert_eq!(receipt.physical_runs_in_flight, 0);
                model
                    .finish_handle()
                    .finish(Instant::now() + Duration::from_secs(2))
                    .unwrap();
            }
        }
        #[test]
        fn unknown_completion_is_fatal_retains_pins_and_never_reopens_admission() {
            let mut model = fixture_model(|_| {
                PhysicalRun::Quarantined(BackendError::new(
                    FailureKind::BackendFailure,
                    FailureStage::Backend,
                    "unknown lifecycle fixture completion",
                ))
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            model.new_game();
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            let handle = model.finish_handle();
            assert!(matches!(
                handle.finish(Instant::now() + Duration::from_secs(1)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            let receipt = handle.receipt();
            assert!(
                receipt.quarantined
                    && !receipt.physical_shutdown_confirmed
                    && !receipt.native_buffers_released
            );
            assert_eq!(receipt.physical_runs_in_flight, 1);
            assert_eq!(receipt.completed_role_inputs, 0);
            assert_eq!(receipt.request_high_water, 1);
            assert!(receipt.last_failure.is_some());
            assert!(receipt.backend_stats.is_none());
        }
        #[test]
        fn unknown_stats_completion_does_not_fabricate_nn_counters() {
            let mut model = fixture_model(|command| match command {
                PalsNativeCommand::Evaluate(input) => PhysicalRun::Complete(Ok(output(input))),
                PalsNativeCommand::NewGame => PhysicalRun::Complete(Ok(PalsNativeResult::NewGame)),
                PalsNativeCommand::SnapshotStats => PhysicalRun::Quarantined(BackendError::new(
                    FailureKind::BackendFailure,
                    FailureStage::Backend,
                    "unknown Stats fixture completion",
                )),
                PalsNativeCommand::VerifyCudaPlacement => unexpected_cuda_placement(),
                PalsNativeCommand::VerifyRuntime => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::RuntimeVerified))
                }
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            model.accepted_output();
            let handle = model.finish_handle();
            assert!(matches!(
                handle.finish(Instant::now() + Duration::from_secs(2)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            let receipt = handle.receipt();
            assert_eq!(receipt.completed_role_inputs, 1);
            assert_eq!(receipt.search_consumed_role_inputs, 1);
            assert!(receipt.backend_stats.is_none());
            assert!(receipt.backend_stats_observation.is_none());
            assert!(receipt.quarantined && !receipt.native_buffers_released);
            assert_eq!(receipt.physical_runs_in_flight, 1);
        }
        #[test]
        fn invalid_stats_ack_preserves_failure_but_cleans_known_idle_owner() {
            let mut model = fixture_model(|command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::Evaluate(input) => output(input),
                    PalsNativeCommand::NewGame => PalsNativeResult::NewGame,
                    PalsNativeCommand::VerifyCudaPlacement => return unexpected_cuda_placement(),
                    PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                    PalsNativeCommand::SnapshotStats => PalsNativeResult::Stats(PalsBackendStats {
                        completed_nn_inputs: 1,
                        ..PalsBackendStats::default()
                    }),
                }))
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            let handle = model.finish_handle();
            assert!(matches!(
                handle.finish(Instant::now() + Duration::from_secs(2)),
                Err(RoleError::Backend(_))
            ));
            let receipt = handle.receipt();
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
            assert!(!receipt.quarantined);
            assert!(receipt.backend_stats.is_none() && receipt.last_failure.is_some());
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert_eq!(receipt.completed_role_inputs, 1);
            assert!(
                handle
                    .finish(Instant::now() + Duration::from_secs(1))
                    .is_err()
            );
        }
    }
}
#[cfg(feature = "onnx-cpu")]
pub use native::{
    NativeBackendStatsReceipt, NativeExecutionReceipt, NativeFailureReceipt, NativePreparedContext,
    NativeRoleFinishHandle, NativeRoleModel, NativeRoleObserver, NativeRoleReceipt,
    NativeRoleRejection, NativeRoleSourceIdentity, NativeStartupProbeReceipt,
};

#[cfg(test)]
mod tests {
    use super::*;
    use rz_search::pals::store::StateId;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;
    #[test]
    fn product_input_meaning_requires_the_exact_rules_profile() {
        let expected = pals_rules_encoding_semantic_digest();
        assert!(verify_pals_rules_profile(Some(PALS_RULES_ENCODING), Some(expected)).is_ok());
        assert!(verify_pals_rules_profile(None, None).is_err());
        assert!(verify_pals_rules_profile(Some(PALS_RULES_ENCODING), None).is_err());
        assert!(verify_pals_rules_profile(Some("different_fields"), Some(expected)).is_err());
        assert!(verify_pals_rules_profile(Some(PALS_RULES_ENCODING), Some([0; 32])).is_err());
        assert_ne!(pals_native_source_digest(), [0; 32]);
    }
    #[test]
    fn actual_rules_board_rights_history_and_legal_order() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 0,
            deadline: Instant::now() + Duration::from_secs(1),
            cancel: &cancel,
        };
        let input = prepare_role_input(&query, NativeQueryKind::Propose, [1; 32]).unwrap();
        assert_eq!(input.board[0], 4);
        assert_eq!(input.board[60], 12);
        assert_eq!(input.metadata[1..5], [1.0; 4]);
        assert_eq!(input.candidates.len(), legal.len());
        let fen = Position::from_fen(&position.to_fen()).unwrap();
        assert_ne!(
            pals_history_digest(&position).unwrap(),
            pals_history_digest(&fen).unwrap()
        );
    }
    #[test]
    fn unknown_records_are_critical_and_overflow_is_explicit() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let unknown: Vec<_> = (0..129)
            .map(|i| RoleRecord {
                revision: i,
                origin_state: StateId(0),
                kind: RecordKind::Counterexample,
                line: vec![],
                value: None,
                completed_depth: 0,
                score_scope: None,
                cpu_observation: None,
                perspective: Color::White,
                critical: true,
            })
            .collect();
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &unknown,
            revision: 129,
            deadline: Instant::now() + Duration::from_secs(1),
            cancel: &cancel,
        };
        assert!(prepare_role_input(&query, NativeQueryKind::Propose, [1; 32]).is_err());
        let query = RoleQuery {
            records: &unknown[..128],
            ..query
        };
        let input = prepare_role_input(&query, NativeQueryKind::Propose, [1; 32]).unwrap();
        assert_eq!(input.required_critical_records.len(), 128);
        assert_eq!(input.records[0].features[1], 0.0);
        assert_eq!(input.records[0].features[15], 1.0);
        let old_unknown: Vec<_> = unknown
            .iter()
            .cloned()
            .map(|mut record| {
                record.critical = false;
                record
            })
            .collect();
        let query = RoleQuery {
            records: &old_unknown,
            ..query
        };
        let input = prepare_role_input(&query, NativeQueryKind::Propose, [1; 32]).unwrap();
        // Historical unknowns are not all current critical commitments. Their
        // status remains explicit, but only the latest kind is mandatorily pinned.
        assert_eq!(input.records.len(), 128);
        assert_eq!(input.required_critical_records, [129]);
        assert!(input.records.iter().all(|r| r.features[15] == 1.0));
    }
    #[test]
    fn divergence_is_actual_replayed_opponent_prefix() {
        let root = Position::startpos();
        let proposal: Vec<_> = ["e2e4", "e7e5", "g1f3", "b8c6"]
            .into_iter()
            .map(|m| BoardMove::from_uci(m).unwrap())
            .collect();
        let cancel = AtomicBool::new(false);
        let query = DivergenceQuery {
            root: &root,
            proposal: &proposal,
            candidates: &[1, 3],
            records: &[],
            revision: 0,
            deadline: Instant::now() + Duration::from_secs(1),
            cancel: &cancel,
        };
        let input = prepare_divergence_input(&query, [1; 32]).unwrap();
        assert_eq!(input.divergence_features.len(), 2);
        assert!(input.candidates.is_empty());
        assert_eq!(input.divergence_features[0][1], 1.0);
        let query = DivergenceQuery {
            candidates: &[0],
            ..query
        };
        assert!(prepare_divergence_input(&query, [1; 32]).is_err());
    }
    #[test]
    fn all_promotion_tokens_roundtrip_to_common_move16() {
        for (piece, bits) in [
            (PieceKind::Queen, 1),
            (PieceKind::Rook, 2),
            (PieceKind::Bishop, 3),
            (PieceKind::Knight, 4),
        ] {
            let movement = BoardMove::new(
                Square::new(48).unwrap(),
                Square::new(56).unwrap(),
                Some(piece),
            )
            .unwrap();
            let token = candidate(movement);
            assert_eq!(token.promotion, bits);
            assert_eq!(
                PalsCandidateToken::from_packed(token.packed().unwrap()).unwrap(),
                token
            );
        }
    }
}
