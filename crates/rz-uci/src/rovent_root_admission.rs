//! Per-go Rovent root admission below an already admitted startup process.
//!
//! DynamicGo inherits the original source/assets/resources/profile declaration;
//! it never relabels another root as the startup's exact registered workload.
//! The original UCI owner supplies its ProcessClock clone. This context does
//! not carry an independent clock-origin capability, so this boundary cannot
//! identify a caller-created second clock with the same epoch. No new origin
//! or deadline is created here, and no raw Instant is serialized.

use crate::{engine::ProcessClock, parser::GoLimits, search_driver::SearchSessionContext};
use rz_contracts::{Deadline, Digest, pals::SearchAuthority, rovent::SearchGoal};
pub use rz_experiments::{
    ROVENT_ROOT_ADMISSION_DOMAIN, ROVENT_ROOT_BUDGET_DOMAIN, ROVENT_ROOT_DEADLINE_DOMAIN,
    RoventOriginalGoCoverage, RoventRootAdmissionPolicy,
};
use rz_experiments::{
    RoventInputLock, RoventParsedGo, RoventResolvedGo, RoventRootAuthority, RoventRootClocks,
    RoventRootDeadline, RoventRootDeclaredCeilings, RoventRootDescriptor,
    RoventRootDescriptorInput,
};
use rz_position::{BoardMove, Position, PositionSnapshot};
use rz_runtime::contracts::{ContractClock, ContractSystemClock};
use std::{fmt, sync::Arc};

pub mod preadmission;

const MAX_ROOT_MOVES: usize = 218;
const MAX_CANONICAL_BYTES: usize = 64 << 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventRootAdmissionFailure {
    InvalidLock,
    RevokedAuthority,
    OriginalGoMissing,
    OriginalGoMismatch,
    Clock,
    DeadlineOrder,
    DeclaredCeiling,
    Goal,
    RootMoves,
    RulesProfile,
    RegisteredWorkload,
    CanonicalEncoding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventRootAdmissionError {
    pub kind: RoventRootAdmissionFailure,
    pub detail: &'static str,
    pub metadata_error: Option<String>,
}
impl fmt::Display for RoventRootAdmissionError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            out,
            "Rovent root admission {:?}: {}",
            self.kind, self.detail
        )?;
        if let Some(source) = &self.metadata_error {
            write!(out, ": {source}")?;
        }
        Ok(())
    }
}
impl std::error::Error for RoventRootAdmissionError {}
fn fail(kind: RoventRootAdmissionFailure, detail: &'static str) -> RoventRootAdmissionError {
    RoventRootAdmissionError {
        kind,
        detail,
        metadata_error: None,
    }
}
fn require(
    valid: bool,
    kind: RoventRootAdmissionFailure,
    detail: &'static str,
) -> Result<(), RoventRootAdmissionError> {
    if valid {
        Ok(())
    } else {
        Err(fail(kind, detail))
    }
}

/// No public fields or deserializer can mint root authority from a digest.
/// Original Rules history, shared revocation and runtime clock owners remain
/// retained separately from the canonical descriptor bytes.
#[derive(Clone, Debug)]
pub struct RoventRootAdmission {
    policy: RoventRootAdmissionPolicy,
    startup_lock: Digest,
    state_and_history: Digest,
    rules_profile: Digest,
    snapshot: PositionSnapshot,
    context: SearchSessionContext,
    runtime_clock: ContractSystemClock,
    root_moves: Option<Arc<[BoardMove]>>,
    allowed_moves: Arc<[BoardMove]>,
    hard: Deadline,
    soft: Deadline,
    admission: Deadline,
    cleanup: Deadline,
    descriptor: RoventRootDescriptor,
    root_budget: Digest,
    deadline_seal: Digest,
    digest: Digest,
    canonical: Vec<u8>,
}
impl RoventRootAdmission {
    pub fn admit(
        lock: &RoventInputLock,
        policy: RoventRootAdmissionPolicy,
        position: &Position,
        context: &SearchSessionContext,
        clock: &ProcessClock,
    ) -> Result<Self, RoventRootAdmissionError> {
        lock.validate().map_err(|_| {
            fail(
                RoventRootAdmissionFailure::InvalidLock,
                "original startup input lock",
            )
        })?;
        accepts(context)?;
        let original_go = context.original_go.as_deref().ok_or_else(|| {
            fail(
                RoventRootAdmissionFailure::OriginalGoMissing,
                "actual admission requires the original parsed UCI go",
            )
        })?;
        // Normal product input was already bounded by the UCI parser. Keep
        // canonical construction bounded even if an internal caller forges a
        // much larger GoLimits; raw repetition is not a distinct-move limit.
        if let Some(moves) = &original_go.searchmoves {
            let bytes = moves.iter().try_fold(0usize, |bytes, movement| {
                bytes.checked_add(movement.len())?.checked_add(3)
            });
            require(
                bytes.is_some_and(|bytes| bytes <= MAX_CANONICAL_BYTES / 2),
                RoventRootAdmissionFailure::CanonicalEncoding,
                "original go exceeds the bounded root descriptor budget",
            )?;
        }
        require(
            original_go.nodes == context.requested_nodes
                && original_go.depth_plies == context.goal.depth_plies
                && original_go.mate_moves == context.goal.mate_moves,
            RoventRootAdmissionFailure::OriginalGoMismatch,
            "original parsed go nodes or goal differ from resolved search context",
        )?;
        let runtime_clock = clock.runtime_clock();
        require(
            clock.epoch().0 != 0
                && context.authority.epoch == clock.epoch()
                && runtime_clock.domain() == clock.domain()
                && context.authority.implementation != Digest([0; 32]),
            RoventRootAdmissionFailure::Clock,
            "original authority and process clock domains differ",
        )?;
        let convert = |instant| {
            clock.deadline(instant).map_err(|_| {
                fail(
                    RoventRootAdmissionFailure::Clock,
                    "deadline predates or overflows the original process origin",
                )
            })
        };
        require(
            convert(runtime_clock.origin())?.at.0 == 0,
            RoventRootAdmissionFailure::Clock,
            "runtime and UCI process origins differ",
        )?;
        let hard = convert(context.control.deadline)?;
        let soft = convert(context.control.soft_deadline)?;
        let admission = convert(context.control.admission_deadline)?;
        let cleanup = convert(context.shutdown_deadline)?;
        require(
            soft.at <= admission.at && admission.at <= hard.at && hard.at <= cleanup.at,
            RoventRootAdmissionFailure::DeadlineOrder,
            "original soft/admission/hard/cleanup deadline order",
        )?;
        let cleanup_limit = lock
            .manifest
            .cleanup_max_ms
            .checked_mul(1_000_000)
            .ok_or_else(|| {
                fail(
                    RoventRootAdmissionFailure::DeclaredCeiling,
                    "declared cleanup ceiling overflows nanoseconds",
                )
            })?;
        require(
            cleanup.at.0 - hard.at.0 <= cleanup_limit,
            RoventRootAdmissionFailure::DeclaredCeiling,
            "original cleanup exceeds the declared cleanup ceiling",
        )?;
        hard.accepts(
            clock.domain(),
            clock
                .now()
                .map_err(|_| fail(RoventRootAdmissionFailure::Clock, "original process now"))?,
        )
        .map_err(|_| {
            fail(
                RoventRootAdmissionFailure::RevokedAuthority,
                "original hard deadline expired",
            )
        })?;
        let registered = &lock.manifest.engine.search;
        context
            .goal
            .validate(registered.registered_max_chess_plies)
            .map_err(|_| {
                fail(
                    RoventRootAdmissionFailure::Goal,
                    "go exceeds registered chess-ply capability",
                )
            })?;
        let snapshot = position.snapshot();
        // This is exactly ContractState.identity.semantic, including history
        // origin/completeness, all known canonical states and irreversible marks.
        let state_and_history = rz_position::contracts::rules_snapshot_semantic_digest(&snapshot);
        let rules_profile = rz_position::contracts::profile_digest();
        require(
            registered.rules_profile_sha256 == hex(rules_profile),
            RoventRootAdmissionFailure::RulesProfile,
            "registered Rules profile differs from actual Rules",
        )?;
        let legal = position.ordered_legal_moves();
        let root_moves = canonical_subset(legal.moves(), context.root_moves.as_deref())?;
        let parsed_root_moves =
            canonical_original_subset(legal.moves(), original_go.searchmoves.as_deref())?;
        require(
            parsed_root_moves.as_deref() == root_moves.as_deref(),
            RoventRootAdmissionFailure::OriginalGoMismatch,
            "original parsed go root subset differs from resolved search context",
        )?;
        let allowed_moves = root_moves
            .clone()
            .unwrap_or_else(|| Arc::from(legal.moves()));
        if policy == RoventRootAdmissionPolicy::ExactRegistered {
            let declared = registered
                .searchmoves
                .as_ref()
                .map(|moves| {
                    moves
                        .iter()
                        .map(|movement| {
                            BoardMove::from_uci(movement).map_err(|_| {
                                fail(
                                    RoventRootAdmissionFailure::RegisteredWorkload,
                                    "registered root move decoding",
                                )
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?;
            let declared = canonical_subset(legal.moves(), declared.as_deref())?;
            require(
                registered.root_state_and_history_sha256 == hex(state_and_history)
                    && registered.requested_nodes == context.requested_nodes
                    && registered.goal() == context.goal
                    && declared.as_deref() == root_moves.as_deref(),
                RoventRootAdmissionFailure::RegisteredWorkload,
                "actual root/history/go/subset differs from exact registered workload",
            )?;
        }
        let startup_lock = parse_digest(&lock.canonical_sha256)?;
        let declared_ceilings = RoventRootDeclaredCeilings {
            resource_plan: lock.manifest.resources.clone(),
            registered_max_chess_plies: registered.registered_max_chess_plies,
            cpu_credit_limit: lock.manifest.coordinator.cpu_credit_limit,
            questions_max: lock.manifest.coordinator.questions_max,
            ancestor_depth_max: lock.manifest.coordinator.ancestor_depth_max,
            waiting_jobs_max: lock.manifest.coordinator.waiting_jobs_max,
            role_tickets_max: lock.manifest.coordinator.role_tickets_max,
            wall_time_max_ms: lock.manifest.wall_time_max_ms,
            cleanup_max_ms: lock.manifest.cleanup_max_ms,
            output_bytes_max: lock.manifest.output_bytes_max,
        };
        // Only this actual Rules/context/clock path can wrap pure metadata in
        // the live capability. A decoded descriptor has no reverse constructor.
        let descriptor = RoventRootDescriptor::seal(RoventRootDescriptorInput {
            policy,
            startup_lock_sha256: hex(startup_lock),
            state_and_history_sha256: hex(state_and_history),
            rules_profile_sha256: hex(rules_profile),
            authority: authority_descriptor(context.authority),
            node_unit: registered.node_unit.clone(),
            search_configuration_sha256: registered.search_configuration_sha256.clone(),
            coordinator_policy_sha256: lock.manifest.coordinator.policy_sha256.clone(),
            original_go: RoventResolvedGo {
                coverage: RoventOriginalGoCoverage::FullOriginalGo,
                requested_nodes: context.requested_nodes,
                depth_plies: context.goal.depth_plies,
                mate_moves: context.goal.mate_moves,
                searchmoves: root_moves.as_ref().map(|moves| move_texts(moves)),
                transport_simulation_cap: context.control.max_simulations,
                original_parsed_limits: Some(original_go_descriptor(original_go)),
            },
            allowed_root_moves: move_texts(&allowed_moves),
            clock: RoventRootClocks {
                hard: deadline_descriptor(hard),
                soft: deadline_descriptor(soft),
                admission: deadline_descriptor(admission),
                cleanup: deadline_descriptor(cleanup),
            },
            declared_ceilings,
        })
        .map_err(codec_error)?;
        descriptor.verify_against(lock).map_err(codec_error)?;
        let root_budget = descriptor.root_budget_digest().map_err(codec_error)?;
        let deadline_seal = descriptor.deadline_seal_digest().map_err(codec_error)?;
        let canonical = descriptor.canonical_bytes().map_err(codec_error)?;
        let digest = descriptor.digest().map_err(codec_error)?;
        // No old authority can become a successful admission while Rules work
        // or canonical encoding was in progress.
        accepts(context)?;
        Ok(Self {
            policy,
            startup_lock,
            state_and_history,
            rules_profile,
            snapshot,
            context: context.clone(),
            runtime_clock,
            root_moves,
            allowed_moves,
            hard,
            soft,
            admission,
            cleanup,
            descriptor,
            root_budget,
            deadline_seal,
            digest,
            canonical,
        })
    }
    pub fn policy(&self) -> RoventRootAdmissionPolicy {
        self.policy
    }
    pub fn original_go_coverage(&self) -> RoventOriginalGoCoverage {
        RoventOriginalGoCoverage::FullOriginalGo
    }
    pub fn startup_lock_digest(&self) -> Digest {
        self.startup_lock
    }
    pub fn state_and_history_digest(&self) -> Digest {
        self.state_and_history
    }
    pub fn rules_profile_digest(&self) -> Digest {
        self.rules_profile
    }
    pub fn root_snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    pub fn authority(&self) -> SearchAuthority {
        self.context.authority
    }
    pub fn original_context(&self) -> &SearchSessionContext {
        &self.context
    }
    pub fn original_go(&self) -> &GoLimits {
        self.context
            .original_go
            .as_deref()
            .expect("admitted original go")
    }
    pub fn requested_nodes(&self) -> Option<u64> {
        self.context.requested_nodes
    }
    pub fn goal(&self) -> SearchGoal {
        self.context.goal
    }
    pub fn root_moves(&self) -> Option<&Arc<[BoardMove]>> {
        self.root_moves.as_ref()
    }
    pub fn allowed_root_moves(&self) -> &Arc<[BoardMove]> {
        &self.allowed_moves
    }
    pub fn runtime_clock(&self) -> &ContractSystemClock {
        &self.runtime_clock
    }
    pub fn hard_deadline(&self) -> Deadline {
        self.hard
    }
    pub fn soft_deadline(&self) -> Deadline {
        self.soft
    }
    pub fn admission_deadline(&self) -> Deadline {
        self.admission
    }
    pub fn cleanup_deadline(&self) -> Deadline {
        self.cleanup
    }
    pub fn declared_ceilings(&self) -> &RoventRootDeclaredCeilings {
        &self.descriptor.input.declared_ceilings
    }
    pub fn descriptor(&self) -> &RoventRootDescriptor {
        &self.descriptor
    }
    pub fn root_budget_digest(&self) -> Digest {
        self.root_budget
    }
    pub fn deadline_seal_digest(&self) -> Digest {
        self.deadline_seal
    }
    pub fn digest(&self) -> Digest {
        self.digest
    }
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical
    }
}

fn accepts(context: &SearchSessionContext) -> Result<(), RoventRootAdmissionError> {
    require(
        context.accepts().map_err(|_| {
            fail(
                RoventRootAdmissionFailure::RevokedAuthority,
                "original UCI authority unavailable",
            )
        })?,
        RoventRootAdmissionFailure::RevokedAuthority,
        "original UCI authority revoked or expired",
    )
}
fn canonical_subset(
    legal: &[BoardMove],
    requested: Option<&[BoardMove]>,
) -> Result<Option<Arc<[BoardMove]>>, RoventRootAdmissionError> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    require(
        !requested.is_empty()
            && requested.len() <= MAX_ROOT_MOVES
            && requested.iter().enumerate().all(|(at, movement)| {
                legal.contains(movement) && !requested[..at].contains(movement)
            }),
        RoventRootAdmissionFailure::RootMoves,
        "root restriction is empty, duplicated, oversized or not Rules legal",
    )?;
    Ok(Some(
        legal
            .iter()
            .copied()
            .filter(|movement| requested.contains(movement))
            .collect::<Vec<_>>()
            .into(),
    ))
}
fn canonical_original_subset(
    legal: &[BoardMove],
    raw: Option<&[String]>,
) -> Result<Option<Arc<[BoardMove]>>, RoventRootAdmissionError> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    // Match Session::seal_root_moves: raw order/repetition is retained in the
    // wire, while only legal distinct moves form the resolved Rules subset.
    let mut distinct = Vec::new();
    for text in raw {
        let movement = BoardMove::from_uci(text).map_err(|_| {
            fail(
                RoventRootAdmissionFailure::OriginalGoMismatch,
                "original go root move decoding",
            )
        })?;
        require(
            legal.contains(&movement),
            RoventRootAdmissionFailure::RootMoves,
            "original go includes a Rules-illegal root move",
        )?;
        if !distinct.contains(&movement) {
            require(
                distinct.len() < MAX_ROOT_MOVES,
                RoventRootAdmissionFailure::RootMoves,
                "original go exceeds the distinct Rules root move bound",
            )?;
            distinct.push(movement);
        }
    }
    canonical_subset(legal, Some(&distinct))
}
fn move_texts(moves: &[BoardMove]) -> Vec<String> {
    moves.iter().map(ToString::to_string).collect()
}
fn original_go_descriptor(go: &GoLimits) -> RoventParsedGo {
    RoventParsedGo {
        searchmoves: go.searchmoves.clone(),
        depth_plies: go.depth_plies,
        mate_moves: go.mate_moves,
        movetime_ms: go.movetime_ms,
        white_time_ms: go.white_time_ms,
        black_time_ms: go.black_time_ms,
        white_increment_ms: go.white_increment_ms,
        black_increment_ms: go.black_increment_ms,
        moves_to_go: go.moves_to_go,
        nodes: go.nodes,
        infinite: go.infinite,
        ponder: go.ponder,
    }
}
fn authority_descriptor(authority: SearchAuthority) -> RoventRootAuthority {
    RoventRootAuthority {
        epoch: authority.epoch.0,
        game: authority.game.0,
        root: authority.root.0,
        implementation_sha256: hex(authority.implementation),
    }
}
fn deadline_descriptor(deadline: Deadline) -> RoventRootDeadline {
    RoventRootDeadline {
        clock_epoch: deadline.clock.0.0,
        at_ns: deadline.at.0,
    }
}
fn codec_error(error: rz_experiments::ManifestError) -> RoventRootAdmissionError {
    let mut failure = fail(
        RoventRootAdmissionFailure::CanonicalEncoding,
        "pure root metadata descriptor validation",
    );
    failure.metadata_error = Some(error.to_string());
    failure
}
fn parse_digest(value: &str) -> Result<Digest, RoventRootAdmissionError> {
    rz_eval::asset::parse_sha256(value)
        .map(Digest)
        .map_err(|_| {
            fail(
                RoventRootAdmissionFailure::InvalidLock,
                "validated startup lock digest",
            )
        })
}
fn hex(digest: Digest) -> String {
    digest.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
#[path = "rovent_root_admission/tests.rs"]
mod tests;
