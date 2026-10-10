//! Opt-in, approximate private warm-start ownership; never an exact input cache.
//!
//! The current native PALS export has no initial-latent input. This Rust unit
//! therefore supplies an explicit invocation/seed boundary and CPU fixtures, not
//! native graph warm-start support. Fresh canonical model and public-memory keys
//! remain unchanged. P/C seeds are isolated; Validator warm-start is unsupported.
//!
//! A caller must build the model input with its Rules-validated encoder and pass
//! the *same* immutable Rules snapshot. The weak exact history binding below is
//! an additional conservative guard, not proof that arbitrary supplied tensor
//! bytes describe that snapshot. Situation/prefix/focus IDs must come from the
//! corresponding Rules/search owner. Neither a tensor hash nor a history digest
//! alone authorizes reuse. Separately reconstructed identical histories miss.
//!
//! Physical completion acknowledgements must come from the actual backend fence.
//! Logical cancellation is insufficient. An abandoned/unknown lease retains its
//! owner and pins in a bounded self-owned quarantine, closes admission, and can
//! only be released by explicit known-completion recovery. If no fence ever
//! arrives, this intentionally remains retained for the process lifetime.

use crate::pals_model::{PalsModelConfig, PalsModelInput, PalsRole};
use rz_contracts::pals::SituationHandle;
use rz_contracts::CancelToken;
use rz_position::{PositionSnapshot, WeakPositionIdentity};
use sha2::{Digest as _, Sha256};
use std::mem::size_of;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

pub const PRIVATE_WARM_SEMANTICS: &str =
    "rz-pals-private-approx-warm/1;fp32-bits;same-exact-rules-context;role-isolated;accepted-seed-only;no-exact-cache";
/// Distinct CUDA invocation namespace. A bank lease alone does not attest a
/// loaded CUDA graph, I/O binding, device residency or physical completion.
pub const PRIVATE_CUDA_WARM_SEMANTICS: &str =
    "rz-pals-private-approx-cuda-warm/2;fp32-bits;same-exact-rules-context;role-isolated;accepted-seed-only;no-exact-cache;device-public-kv;physical-output-fence";
pub const CURRENT_NATIVE_PRIVATE_WARM_SUPPORTED: bool = false;
const MAX_LATENT_ELEMENTS: usize = 16 * 384;
// Conservative per-allocation ownership overhead. This is a module reservation,
// not observed RSS/allocator peak or a charge for the Rules-owned weak target.
const OWNER_OVERHEAD: u64 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrivateWarmSupport {
    RustConsumerBoundaryOnly,
    NativeExportUnsupported,
}

pub const fn native_warm_support() -> PrivateWarmSupport {
    PrivateWarmSupport::NativeExportUnsupported
}

pub fn require_native_warm_start() -> Result<(), PrivateSeedError> {
    Err(PrivateSeedError::UnsupportedNativeWarmStart)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrivatePrecision {
    Fp32,
}

/// Fixed-size declarations avoid caller-controlled string/backing allocations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivateModelIdentity {
    pub model: [u8; 32],
    pub encoding: [u8; 32],
    pub precision: PrivatePrecision,
    /// Checkpoint/model epoch digest, exactly `PalsModelInput.model_epoch`.
    pub model_epoch: [u8; 32],
    /// Numerical backend frozen epoch; distinct from the checkpoint digest.
    pub frozen_epoch: u64,
}

/// Owner-issued situation identity. Digests are not Rules validity proofs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivateSituationContext {
    pub situation: SituationHandle,
    pub prefix: [u8; 32],
    pub focus: [u8; 32],
}

/// Sealed actual prepared input plus conservative exact Rules/history binding.
/// Only public records/revision may vary between eligible warm invocations;
/// board, metadata, query, candidates and divergence feature bits are hashed
/// into `non_record_context`. This profile does not support parent-child seeds.
#[derive(Clone, Debug)]
pub struct PrivateRulesContext {
    rules: WeakPositionIdentity,
    rules_revision: u64,
    role: PalsRole,
    model_epoch: [u8; 32],
    prepared_input: [u8; 32],
    history: [u8; 32],
    non_record_context: [u8; 32],
    record_revision: u64,
    situation: PrivateSituationContext,
}

impl PrivateRulesContext {
    pub fn seal(
        snapshot: &PositionSnapshot,
        input: &PalsModelInput,
        config: &PalsModelConfig,
        situation: PrivateSituationContext,
    ) -> Result<Self, PrivateSeedError> {
        input
            .validate(config)
            .map_err(|_| PrivateSeedError::InvalidModelInput)?;
        let mut context = Sha256::new();
        context.update(b"rz-pals-private-non-record-context/1");
        context.update([role_index(input.role) as u8]);
        context.update(input.model_epoch);
        context.update(input.history_digest);
        context.update(&input.board);
        for value in input.metadata.iter().chain(input.query.iter()) {
            context.update(value.to_bits().to_le_bytes());
        }
        context.update((input.candidates.len() as u64).to_le_bytes());
        for candidate in &input.candidates {
            context.update(
                candidate
                    .packed()
                    .map_err(|_| PrivateSeedError::InvalidModelInput)?
                    .to_le_bytes(),
            );
        }
        context.update((input.divergence_features.len() as u64).to_le_bytes());
        for value in input.divergence_features.iter().flatten() {
            context.update(value.to_bits().to_le_bytes());
        }
        if !config.profile.is_legacy() {
            context.update(b"rz-pals-private-non-record-v2/2");
            context.update(config.profile.as_str().as_bytes());
            context.update(config.profile.encoding_schema().as_bytes());
            // Ordered query lines are part of the question, not public record
            // revision. Record lines and their local relationships may change.
            if let Some(lines) = &input.full_line {
                for line in [
                    &lines.query_prefix,
                    &lines.query_proposal,
                    &lines.query_counter,
                ] {
                    context.update((line.len() as u64).to_le_bytes());
                    for candidate in line {
                        context.update(
                            candidate
                                .packed()
                                .map_err(|_| PrivateSeedError::InvalidModelInput)?
                                .to_le_bytes(),
                        );
                    }
                }
            }
        }
        Ok(Self {
            rules: snapshot.weak_position_identity(),
            rules_revision: snapshot.revision(),
            role: input.role,
            model_epoch: input.model_epoch,
            prepared_input: input
                .canonical_input_key(config)
                .map_err(|_| PrivateSeedError::InvalidModelInput)?,
            history: input.history_digest,
            non_record_context: context.finalize().into(),
            record_revision: input.situation_revision,
            situation,
        })
    }

    pub fn prepared_input_key(&self) -> [u8; 32] {
        self.prepared_input
    }
    pub fn history_digest(&self) -> [u8; 32] {
        self.history
    }
    pub fn record_revision(&self) -> u64 {
        self.record_revision
    }

    fn matches_rules(&self, snapshot: &PositionSnapshot) -> bool {
        self.rules_revision == snapshot.revision() && self.rules.matches(snapshot)
    }

    fn same_invocation(&self, other: &Self, snapshot: &PositionSnapshot) -> bool {
        self.matches_rules(snapshot)
            && other.matches_rules(snapshot)
            && self.prepared_input == other.prepared_input
            && self.history == other.history
            && self.non_record_context == other.non_record_context
            && self.record_revision == other.record_revision
            && self.situation == other.situation
    }

    fn warm_eligible(&self, other: &Self, snapshot: &PositionSnapshot) -> bool {
        self.matches_rules(snapshot)
            && other.matches_rules(snapshot)
            && self.history == other.history
            && self.non_record_context == other.non_record_context
            && self.situation == other.situation
            && self.record_revision <= other.record_revision
    }
}

#[derive(Clone, Debug)]
pub struct PrivateSeedRequest {
    pub role: PalsRole,
    pub model: PrivateModelIdentity,
    pub game_generation: u64,
    pub search_generation: u64,
    pub context: PrivateRulesContext,
}

impl PrivateSeedRequest {
    fn same_invocation(&self, other: &Self, snapshot: &PositionSnapshot) -> bool {
        self.role == other.role
            && self.model == other.model
            && self.game_generation == other.game_generation
            && self.search_generation == other.search_generation
            && self.context.same_invocation(&other.context, snapshot)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrivateInvocation {
    /// Exactly the original prepared model input key; no fresh key rewriting.
    Fresh { input_key: [u8; 32] },
    /// A separate approximate namespace that includes the accepted seed seal.
    ApproxWarmV1 {
        input_key: [u8; 32],
        seed_seal: [u8; 32],
        invocation_key: [u8; 32],
    },
    /// CUDA Warm is explicitly admitted and never relabels a V1 invocation.
    ApproxCudaWarmV2 {
        input_key: [u8; 32],
        seed_seal: [u8; 32],
        invocation_key: [u8; 32],
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrivateInvocationMode {
    Fresh,
    ApproxWarmV1,
    ApproxCudaWarmV2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivateSeedProvenance {
    pub role: PalsRole,
    pub model: PrivateModelIdentity,
    pub game_generation: u64,
    pub search_generation: u64,
    pub source_input: [u8; 32],
    pub source_history: [u8; 32],
    pub source_non_record_context: [u8; 32],
    pub source_record_revision: u64,
    pub source_rules_revision: u64,
    pub source_situation: PrivateSituationContext,
    pub source_invocation: PrivateInvocation,
    pub source_lease_id: u64,
    pub seed_sequence: u64,
    pub latent_bits_digest: [u8; 32],
    pub seal: [u8; 32],
}

/// Exact post-commit withdrawal, not a physical completion or count rollback.
/// A missing/already-removed authority returns `revoked=false` and removes no
/// bytes. Remaining entries describe the bank after this single operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivateSeedRevocation {
    pub provenance: PrivateSeedProvenance,
    pub reason: PrivateCancelReason,
    pub revoked: bool,
    pub removed_bank_bytes: u64,
    pub remaining_entries_per_role: [usize; 3],
}

struct Seed {
    provenance: PrivateSeedProvenance,
    context: PrivateRulesContext,
    bits: Box<[u32]>,
    charged_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivateSeedLimits {
    /// One or two accepted seeds per role; no unbounded history of latents.
    pub slots_per_role: usize,
    /// Module-owned accepted seed/slot reservation, excluding backend buffers.
    /// Includes one outward lease handle, including its fixed admitted model.
    pub max_bank_bytes: u64,
    /// Overlapping output conversion/provisional ownership, not observed peak.
    pub max_transient_bytes: u64,
    /// Exact registered shape. No truncation or automatic OOM resizing.
    pub latent_elements: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalSeedCompletion {
    Succeeded,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrivateCancelReason {
    Cancelled,
    Stale,
    Deadline,
    Reset,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrivateSeedError {
    InvalidLimits,
    InvalidModelInput,
    Shape,
    NonFinite,
    Budget,
    Allocation,
    UnsupportedRole,
    UnsupportedNativeWarmStart,
    MissingSeed,
    StaleRules,
    StaleGame,
    StaleInvocation,
    Cancelled,
    Deadline,
    AdmissionClosed,
    ActivePhysicalLease,
    PendingFinalization,
    PhysicalCompletionUnknown,
    PhysicalFailure,
    InvalidLease,
    GenerationExhausted,
    Poisoned,
}
impl std::fmt::Display for PrivateSeedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PrivateSeedError {}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrivateSeedCounts {
    /// These are seed invocations, not NN executions, visits, or backups.
    pub fresh_admissions: u64,
    pub warm_admissions: u64,
    pub accepted_seeds: u64,
    pub known_completions: u64,
    pub recovered_unknown_completions: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivateSeedSnapshot {
    pub game_generation: u64,
    pub entries_per_role: [usize; 3],
    pub reserved_bank_bytes: u64,
    pub provisional_bytes: u64,
    pub pinned_entries: usize,
    pub active_lease: Option<u64>,
    pub active_seed_bytes: u64,
    pub active_cancel_reason: Option<PrivateCancelReason>,
    pub last_cancel_reason: Option<PrivateCancelReason>,
    pub last_completed_lease: Option<u64>,
    pub last_physical_completion: Option<PhysicalSeedCompletion>,
    pub quarantined: bool,
    pub admission_closed: bool,
    pub last_failure: Option<PrivateSeedError>,
    pub counts: PrivateSeedCounts,
    /// Withdrawn accepted entries; historical accepted/physical counts remain.
    pub revoked_seeds: u64,
    /// One bounded last successful removal, with its original reason.
    pub last_revocation: Option<PrivateSeedRevocation>,
}

struct Active {
    id: u64,
    deadline: Instant,
    cancelled: CancelToken,
    request: PrivateSeedRequest,
    invocation: PrivateInvocation,
    seed: Option<Arc<Seed>>,
    cancel_reason: Option<PrivateCancelReason>,
    unknown: bool,
}
struct Pending {
    id: u64,
    deadline: Instant,
    cancelled: CancelToken,
    request: PrivateSeedRequest,
    invocation: PrivateInvocation,
    completion: PhysicalSeedCompletion,
    cancel_reason: Option<PrivateCancelReason>,
    provisional: Option<Arc<Seed>>,
}
struct State {
    limits: PrivateSeedLimits,
    game: u64,
    next_id: u64,
    seed_sequence: u64,
    slots: [Vec<Arc<Seed>>; 3],
    base_bytes: u64,
    bank_bytes: u64,
    active: Option<Active>,
    pending: Option<Pending>,
    admission_closed: bool,
    last_failure: Option<PrivateSeedError>,
    last_cancel_reason: Option<PrivateCancelReason>,
    last_completed_lease: Option<u64>,
    last_physical_completion: Option<PhysicalSeedCompletion>,
    counts: PrivateSeedCounts,
    revoked_seeds: u64,
    last_revocation: Option<PrivateSeedRevocation>,
    // A single bounded cycle holds physical-unknown state even if user handles
    // are dropped. Known fence recovery explicitly breaks it; never auto-retry.
    quarantine_owner: Option<Arc<Mutex<State>>>,
}

/// One physical invocation at a time, with separate bounded role banks.
pub struct PrivateSeedBank {
    inner: Arc<Mutex<State>>,
}

impl PrivateSeedBank {
    pub fn new(limits: PrivateSeedLimits, game_generation: u64) -> Result<Self, PrivateSeedError> {
        if !(1..=2).contains(&limits.slots_per_role)
            || limits.latent_elements != MAX_LATENT_ELEMENTS
            || limits.max_bank_bytes == 0
            || limits.max_transient_bytes == 0
        {
            return Err(PrivateSeedError::InvalidLimits);
        }
        let mut slots: [Vec<Arc<Seed>>; 3] = std::array::from_fn(|_| Vec::new());
        for role in &mut slots {
            role.try_reserve_exact(limits.slots_per_role)
                .map_err(|_| PrivateSeedError::Allocation)?;
        }
        let backing = slots.iter().try_fold(0u64, |sum, role| {
            sum.checked_add(checked_bytes(role.capacity(), size_of::<Arc<Seed>>())?)
                .ok_or(PrivateSeedError::Budget)
        })?;
        let base_bytes = backing
            .checked_add(size_of::<State>() as u64)
            .and_then(|n| n.checked_add(size_of::<PrivateSeedLease>() as u64))
            .and_then(|n| n.checked_add(OWNER_OVERHEAD))
            .ok_or(PrivateSeedError::Budget)?;
        if base_bytes > limits.max_bank_bytes {
            return Err(PrivateSeedError::Budget);
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(State {
                limits,
                game: game_generation,
                next_id: 0,
                seed_sequence: 0,
                slots,
                base_bytes,
                bank_bytes: base_bytes,
                active: None,
                pending: None,
                admission_closed: false,
                last_failure: None,
                counts: PrivateSeedCounts::default(),
                revoked_seeds: 0,
                last_revocation: None,
                last_cancel_reason: None,
                last_completed_lease: None,
                last_physical_completion: None,
                quarantine_owner: None,
            })),
        })
    }

    pub const fn warm_support(&self) -> PrivateWarmSupport {
        PrivateWarmSupport::RustConsumerBoundaryOnly
    }

    /// Remove only an idle stored seed equal to the full authority returned by
    /// `commit`. A failed final caller guard may use this without pretending that
    /// commit or its physical completion never happened. No older evicted seed
    /// is restored, and a later miss must use an explicitly Fresh invocation.
    ///
    /// Active, pending or unknown ownership closes admission and refuses removal.
    /// Neither those pins nor a previous failure are replaced by a new fence,
    /// reset, cancellation token or generation. All fallible preflight precedes
    /// deletion; counter/byte failure leaves the exact seed pinned in the bank.
    pub fn revoke_committed(
        &self,
        provenance: &PrivateSeedProvenance,
        reason: PrivateCancelReason,
    ) -> Result<PrivateSeedRevocation, PrivateSeedError> {
        let mut state = lock(&self.inner)?;
        let result = (|| {
            if state.quarantine_owner.is_some()
                || state.active.as_ref().is_some_and(|active| active.unknown)
            {
                return Err(PrivateSeedError::PhysicalCompletionUnknown);
            }
            if state.active.is_some() {
                return Err(PrivateSeedError::ActivePhysicalLease);
            }
            if state.pending.is_some() {
                return Err(PrivateSeedError::PendingFinalization);
            }
            let role = role_index(provenance.role);
            let Some(index) = state.slots[role]
                .iter()
                .position(|seed| seed.provenance == *provenance)
            else {
                return Ok(PrivateSeedRevocation {
                    provenance: *provenance,
                    reason,
                    revoked: false,
                    removed_bank_bytes: 0,
                    remaining_entries_per_role: std::array::from_fn(|i| state.slots[i].len()),
                });
            };
            let seed = &state.slots[role][index];
            if Arc::strong_count(seed) != 1 {
                return Err(PrivateSeedError::PhysicalCompletionUnknown);
            }
            let removed_bank_bytes = seed.charged_bytes;
            let remaining_bytes = state
                .bank_bytes
                .checked_sub(removed_bank_bytes)
                .filter(|bytes| *bytes >= state.base_bytes)
                .ok_or(PrivateSeedError::Budget)?;
            let revoked_seeds = state
                .revoked_seeds
                .checked_add(1)
                .ok_or(PrivateSeedError::GenerationExhausted)?;
            // Exact full authority, exclusive ownership, count and reservation
            // are now checked. Vec removal cannot allocate or grow the bank.
            drop(state.slots[role].remove(index));
            state.bank_bytes = remaining_bytes;
            state.revoked_seeds = revoked_seeds;
            let receipt = PrivateSeedRevocation {
                provenance: *provenance,
                reason,
                revoked: true,
                removed_bank_bytes,
                remaining_entries_per_role: std::array::from_fn(|i| state.slots[i].len()),
            };
            state.last_revocation = Some(receipt);
            Ok(receipt)
        })();
        if let Err(error) = result {
            state.admission_closed = true;
            state.last_failure = Some(error);
        }
        result
    }

    pub fn begin(
        &self,
        request: PrivateSeedRequest,
        snapshot: &PositionSnapshot,
        mode: PrivateInvocationMode,
        cancelled: &CancelToken,
        deadline: Instant,
    ) -> Result<PrivateSeedLease, PrivateSeedError> {
        let mut state = lock(&self.inner)?;
        let result = (|| {
            check_control(cancelled, deadline)?;
            if state.admission_closed {
                return Err(if state.active.as_ref().is_some_and(|a| a.unknown) {
                    PrivateSeedError::PhysicalCompletionUnknown
                } else {
                    PrivateSeedError::AdmissionClosed
                });
            }
            if state.active.is_some() {
                return Err(PrivateSeedError::ActivePhysicalLease);
            }
            if state.pending.is_some() {
                return Err(PrivateSeedError::PendingFinalization);
            }
            if request.game_generation != state.game {
                return Err(PrivateSeedError::StaleGame);
            }
            if !request.context.matches_rules(snapshot) {
                return Err(PrivateSeedError::StaleRules);
            }
            if request.role != request.context.role
                || request.model.model_epoch != request.context.model_epoch
            {
                return Err(PrivateSeedError::InvalidModelInput);
            }
            let warm = mode != PrivateInvocationMode::Fresh;
            if warm && request.role == PalsRole::Validator {
                return Err(PrivateSeedError::UnsupportedRole);
            }
            let seed = if warm {
                state.slots[role_index(request.role)]
                    .iter()
                    .rev()
                    .find(|seed| {
                        let p = &seed.provenance;
                        p.model == request.model
                            && p.game_generation == request.game_generation
                            && p.search_generation == request.search_generation
                            && p.role == request.role
                            && seed.context.warm_eligible(&request.context, snapshot)
                    })
                    .cloned()
                    .ok_or(PrivateSeedError::MissingSeed)
                    .map(Some)?
            } else {
                None
            };
            let id = state
                .next_id
                .checked_add(1)
                .ok_or(PrivateSeedError::GenerationExhausted)?;
            let count = match mode {
                PrivateInvocationMode::Fresh => state.counts.fresh_admissions.checked_add(1),
                PrivateInvocationMode::ApproxWarmV1 | PrivateInvocationMode::ApproxCudaWarmV2 => {
                    state.counts.warm_admissions.checked_add(1)
                }
            }
            .ok_or(PrivateSeedError::GenerationExhausted)?;
            let invocation = invocation(&request, seed.as_deref(), mode);
            let model = request.model;
            check_control(cancelled, deadline)?;
            state.next_id = id;
            match mode {
                PrivateInvocationMode::Fresh => state.counts.fresh_admissions = count,
                PrivateInvocationMode::ApproxWarmV1 | PrivateInvocationMode::ApproxCudaWarmV2 => {
                    state.counts.warm_admissions = count
                }
            }
            state.active = Some(Active {
                id,
                deadline,
                cancelled: cancelled.clone(),
                request,
                invocation,
                seed: seed.clone(),
                cancel_reason: None,
                unknown: false,
            });
            Ok(PrivateSeedLease {
                inner: self.inner.clone(),
                id,
                invocation,
                model,
                seed,
                finalized: false,
            })
        })();
        if let Err(error) = result.as_ref() {
            state.last_failure = Some(*error);
        }
        result
    }

    pub fn cancel_active(
        &self,
        id: u64,
        reason: PrivateCancelReason,
    ) -> Result<(), PrivateSeedError> {
        let mut state = lock(&self.inner)?;
        let active = state
            .active
            .as_mut()
            .filter(|a| a.id == id)
            .ok_or(PrivateSeedError::InvalidLease)?;
        active.cancel_reason = Some(reason);
        state.last_cancel_reason = Some(reason);
        Ok(())
    }

    /// Use only after the actual fence for an abandoned/unknown invocation.
    /// Recovery releases the pin but never creates a reusable accepted seed.
    pub fn resolve_unknown_known_completion(
        &self,
        id: u64,
        outcome: PhysicalSeedCompletion,
    ) -> Result<(), PrivateSeedError> {
        let mut state = lock(&self.inner)?;
        if !state
            .active
            .as_ref()
            .is_some_and(|a| a.id == id && a.unknown)
        {
            return Err(PrivateSeedError::InvalidLease);
        }
        let count = state
            .counts
            .recovered_unknown_completions
            .checked_add(1)
            .ok_or(PrivateSeedError::GenerationExhausted)?;
        state.active.take();
        state.quarantine_owner.take();
        state.counts.recovered_unknown_completions = count;
        state.last_completed_lease = Some(id);
        state.last_physical_completion = Some(outcome);
        state.last_failure = Some(match outcome {
            PhysicalSeedCompletion::Succeeded => PrivateSeedError::PhysicalCompletionUnknown,
            PhysicalSeedCompletion::Failed => PrivateSeedError::PhysicalFailure,
        });
        // Admission stays closed. A caller must explicitly reset a new game.
        Ok(())
    }

    /// Admission closes even if reset is refused. Physical owners are never
    /// destroyed to make a new-game operation appear successful.
    pub fn reset_game(&self, new_generation: u64) -> Result<(), PrivateSeedError> {
        let mut state = lock(&self.inner)?;
        state.admission_closed = true;
        let reset_reason = if let Some(active) = state.active.as_mut() {
            active.cancelled.cancel();
            Some(
                *active
                    .cancel_reason
                    .get_or_insert(PrivateCancelReason::Reset),
            )
        } else if let Some(pending) = state.pending.as_mut() {
            pending.cancelled.cancel();
            Some(
                *pending
                    .cancel_reason
                    .get_or_insert(PrivateCancelReason::Reset),
            )
        } else {
            None
        };
        if let Some(reason) = reset_reason {
            state.last_cancel_reason = Some(reason);
        }
        let result = (|| {
            if state.active.is_some() {
                return Err(PrivateSeedError::ActivePhysicalLease);
            }
            if state.pending.is_some() {
                return Err(PrivateSeedError::PendingFinalization);
            }
            if new_generation <= state.game {
                return Err(PrivateSeedError::StaleGame);
            }
            for role in &mut state.slots {
                role.clear();
            }
            state.bank_bytes = state.base_bytes;
            state.game = new_generation;
            state.admission_closed = false;
            Ok(())
        })();
        if let Err(error) = result {
            state.last_failure = Some(error);
        }
        result
    }

    /// Close this same-game bank after the caller's actual physical owner
    /// fence. This method does not produce that fence or change the game. A
    /// refused close preserves all seed storage and permanently closes new
    /// admission; live leases, provisional seeds and outside pins cannot be
    /// discarded to manufacture release evidence.
    pub fn close_after_known_fence(&self) -> Result<PrivateSeedSnapshot, PrivateSeedError> {
        let mut state = lock(&self.inner)?;
        state.admission_closed = true;
        let result = (|| {
            if state.quarantine_owner.is_some()
                || state.active.as_ref().is_some_and(|active| active.unknown)
            { return Err(PrivateSeedError::PhysicalCompletionUnknown); }
            if state.active.is_some() { return Err(PrivateSeedError::ActivePhysicalLease); }
            if state.pending.is_some() { return Err(PrivateSeedError::PendingFinalization); }
            if state.slots.iter().flatten().any(|seed| Arc::strong_count(seed) != 1) {
                return Err(PrivateSeedError::PhysicalCompletionUnknown);
            }
            for role in &mut state.slots { role.clear(); }
            state.bank_bytes = state.base_bytes;
            Ok(())
        })();
        if let Err(error) = result { state.last_failure = Some(error); }
        drop(state);
        result?;
        self.snapshot()
    }

    pub fn snapshot(&self) -> Result<PrivateSeedSnapshot, PrivateSeedError> {
        let state = lock(&self.inner)?;
        Ok(PrivateSeedSnapshot {
            game_generation: state.game,
            entries_per_role: std::array::from_fn(|i| state.slots[i].len()),
            reserved_bank_bytes: state.bank_bytes,
            provisional_bytes: state
                .pending
                .as_ref()
                .and_then(|p| p.provisional.as_ref())
                .map_or(0, |s| s.charged_bytes),
            pinned_entries: state
                .slots
                .iter()
                .flatten()
                .filter(|seed| Arc::strong_count(seed) > 1)
                .count(),
            active_lease: state.active.as_ref().map(|a| a.id),
            active_seed_bytes: state
                .active
                .as_ref()
                .and_then(|a| a.seed.as_ref())
                .map_or(0, |s| s.charged_bytes),
            active_cancel_reason: state.active.as_ref().and_then(|a| {
                a.cancel_reason.or(a
                    .cancelled
                    .is_canceled()
                    .then_some(PrivateCancelReason::Cancelled))
            }),
            last_cancel_reason: state.last_cancel_reason,
            last_completed_lease: state.last_completed_lease,
            last_physical_completion: state.last_physical_completion,
            quarantined: state.active.as_ref().is_some_and(|a| a.unknown),
            admission_closed: state.admission_closed,
            last_failure: state.last_failure,
            counts: state.counts,
            revoked_seeds: state.revoked_seeds,
            last_revocation: state.last_revocation,
        })
    }
}

impl Drop for PrivateSeedBank {
    fn drop(&mut self) {
        match self.inner.lock() {
            Ok(mut state) => {
                state.admission_closed = true;
                if let Some(active) = state.active.as_mut() {
                    active.unknown = true;
                }
                if state.active.is_some() {
                    state.last_failure = Some(PrivateSeedError::PhysicalCompletionUnknown);
                    state.quarantine_owner = Some(self.inner.clone());
                }
            }
            Err(_) => {
                std::mem::forget(self.inner.clone());
            }
        }
    }
}

/// A seed pin and actual invocation identity. Not Clone; physical completion is
/// consumed once. The provider must retain its own input/output/session buffers.
pub struct PrivateSeedLease {
    inner: Arc<Mutex<State>>,
    id: u64,
    invocation: PrivateInvocation,
    model: PrivateModelIdentity,
    seed: Option<Arc<Seed>>,
    finalized: bool,
}
impl PrivateSeedLease {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn invocation(&self) -> PrivateInvocation {
        self.invocation
    }
    /// The full identity admitted by `begin`, including Fresh leases with no
    /// seed provenance. Fixed-size copying performs no allocation or locking.
    pub fn model_identity(&self) -> PrivateModelIdentity {
        self.model
    }
    pub fn seed_bits(&self) -> Option<&[u32]> {
        self.seed.as_ref().map(|s| s.bits.as_ref())
    }
    pub fn seed_provenance(&self) -> Option<&PrivateSeedProvenance> {
        self.seed.as_ref().map(|s| &s.provenance)
    }

    /// The caller asserts an actual physical fence, not logical cancellation.
    pub fn complete_known(
        mut self,
        outcome: PhysicalSeedCompletion,
    ) -> Result<CompletedPrivateInvocation, PrivateSeedError> {
        let mut state = lock(&self.inner)?;
        let active = state
            .active
            .as_ref()
            .filter(|a| a.id == self.id)
            .ok_or(PrivateSeedError::InvalidLease)?;
        let count = state
            .counts
            .known_completions
            .checked_add(1)
            .ok_or(PrivateSeedError::GenerationExhausted)?;
        let recovered = if active.unknown {
            Some(
                state
                    .counts
                    .recovered_unknown_completions
                    .checked_add(1)
                    .ok_or(PrivateSeedError::GenerationExhausted)?,
            )
        } else {
            None
        };
        let cancel_reason = active
            .cancel_reason
            .or(active
                .cancelled
                .is_canceled()
                .then_some(PrivateCancelReason::Cancelled))
            .or(active.unknown.then_some(PrivateCancelReason::Reset));
        let pending = Pending {
            id: active.id,
            deadline: active.deadline,
            cancelled: active.cancelled.clone(),
            request: active.request.clone(),
            invocation: active.invocation,
            completion: outcome,
            cancel_reason,
            provisional: None,
        };
        state.active.take();
        state.quarantine_owner.take();
        state.pending = Some(pending);
        state.counts.known_completions = count;
        if let Some(recovered) = recovered {
            state.counts.recovered_unknown_completions = recovered;
        }
        state.last_completed_lease = Some(self.id);
        state.last_physical_completion = Some(outcome);
        if let Some(reason) = cancel_reason {
            state.last_cancel_reason = Some(reason);
        }
        if outcome == PhysicalSeedCompletion::Failed {
            state.last_failure = Some(PrivateSeedError::PhysicalFailure);
        }
        self.finalized = true;
        Ok(CompletedPrivateInvocation {
            inner: self.inner.clone(),
            id: self.id,
            finalized: false,
        })
    }
}
impl Drop for PrivateSeedLease {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        match self.inner.lock() {
            Ok(mut state) => {
                if let Some(active) = state.active.as_mut().filter(|a| a.id == self.id) {
                    active.unknown = true;
                    state.admission_closed = true;
                    state.last_failure = Some(PrivateSeedError::PhysicalCompletionUnknown);
                    state.quarantine_owner = Some(self.inner.clone());
                }
            }
            Err(_) => {
                std::mem::forget(self.inner.clone());
            }
        }
    }
}

/// Physically complete, still provisional. No acceptance/search work is implied.
pub struct CompletedPrivateInvocation {
    inner: Arc<Mutex<State>>,
    id: u64,
    finalized: bool,
}
impl CompletedPrivateInvocation {
    pub fn stage_output(
        mut self,
        latent: &[f32],
    ) -> Result<ProvisionalPrivateSeed, PrivateSeedError> {
        let mut state = lock(&self.inner)?;
        let result = (|| {
            if state.admission_closed {
                return Err(PrivateSeedError::AdmissionClosed);
            }
            let pending = state
                .pending
                .as_ref()
                .filter(|p| p.id == self.id)
                .ok_or(PrivateSeedError::InvalidLease)?;
            if Instant::now() >= pending.deadline {
                return Err(PrivateSeedError::Deadline);
            }
            if pending.cancelled.is_canceled() {
                return Err(PrivateSeedError::Cancelled);
            }
            if pending.completion != PhysicalSeedCompletion::Succeeded {
                return Err(PrivateSeedError::PhysicalFailure);
            }
            if pending.cancel_reason.is_some() {
                return Err(PrivateSeedError::Cancelled);
            }
            if latent.len() != state.limits.latent_elements {
                return Err(PrivateSeedError::Shape);
            }
            if !latent.iter().all(|v| v.is_finite()) {
                return Err(PrivateSeedError::NonFinite);
            }
            let charged_bytes = checked_bytes(latent.len(), size_of::<u32>())?
                .checked_add(size_of::<Seed>() as u64 + OWNER_OVERHEAD)
                .ok_or(PrivateSeedError::Budget)?;
            // The conversion vector and boxed payload may overlap when shrinking.
            let conversion_reservation = charged_bytes
                .checked_add(checked_bytes(latent.len(), size_of::<u32>())?)
                .ok_or(PrivateSeedError::Budget)?;
            if conversion_reservation > state.limits.max_transient_bytes {
                return Err(PrivateSeedError::Budget);
            }
            let sequence = state
                .seed_sequence
                .checked_add(1)
                .ok_or(PrivateSeedError::GenerationExhausted)?;
            let mut bits = Vec::new();
            bits.try_reserve_exact(latent.len())
                .map_err(|_| PrivateSeedError::Allocation)?;
            let actual_conversion = charged_bytes
                .checked_add(checked_bytes(bits.capacity(), size_of::<u32>())?)
                .ok_or(PrivateSeedError::Budget)?;
            if actual_conversion > state.limits.max_transient_bytes {
                return Err(PrivateSeedError::Budget);
            }
            bits.extend(latent.iter().map(|v| v.to_bits()));
            let bits = bits.into_boxed_slice();
            let provenance = provenance(
                &pending.request,
                pending.invocation,
                self.id,
                sequence,
                &bits,
            );
            let seed = Arc::new(Seed {
                provenance,
                context: pending.request.context.clone(),
                bits,
                charged_bytes,
            });
            state
                .pending
                .as_mut()
                .ok_or(PrivateSeedError::InvalidLease)?
                .provisional = Some(seed);
            Ok(ProvisionalPrivateSeed {
                inner: self.inner.clone(),
                id: self.id,
                finalized: false,
            })
        })();
        if let Err(error) = result.as_ref() {
            state.last_failure = Some(*error);
        }
        if result.is_ok() {
            self.finalized = true;
        }
        result
    }
}
impl Drop for CompletedPrivateInvocation {
    fn drop(&mut self) {
        if !self.finalized {
            abort_pending(&self.inner, self.id);
        }
    }
}

pub struct ProvisionalPrivateSeed {
    inner: Arc<Mutex<State>>,
    id: u64,
    finalized: bool,
}
impl ProvisionalPrivateSeed {
    /// Commit only after the caller's accepted-output guard. This method repeats
    /// cancellation/deadline/generation/context checks; it creates no NN count,
    /// visit, backup, or Rules proof. A failed preflight preserves old entries.
    pub fn commit(
        mut self,
        current: &PrivateSeedRequest,
        snapshot: &PositionSnapshot,
        cancelled: &CancelToken,
        deadline: Instant,
    ) -> Result<PrivateSeedProvenance, PrivateSeedError> {
        let mut state = lock(&self.inner)?;
        let result = (|| {
            check_control(cancelled, deadline)?;
            if state.admission_closed {
                return Err(PrivateSeedError::AdmissionClosed);
            }
            if current.game_generation != state.game {
                return Err(PrivateSeedError::StaleGame);
            }
            let pending = state
                .pending
                .as_ref()
                .filter(|p| p.id == self.id)
                .ok_or(PrivateSeedError::InvalidLease)?;
            if Instant::now() >= pending.deadline {
                return Err(PrivateSeedError::Deadline);
            }
            if pending.cancelled.is_canceled() {
                return Err(PrivateSeedError::Cancelled);
            }
            if pending.completion != PhysicalSeedCompletion::Succeeded {
                return Err(PrivateSeedError::PhysicalFailure);
            }
            if pending.cancel_reason.is_some() {
                return Err(PrivateSeedError::Cancelled);
            }
            if !pending.request.same_invocation(current, snapshot) {
                return Err(PrivateSeedError::StaleInvocation);
            }
            let seed = pending
                .provisional
                .as_ref()
                .ok_or(PrivateSeedError::InvalidLease)?
                .clone();
            let role = role_index(seed.provenance.role);
            let mut evict = 0;
            let mut prospective = state
                .bank_bytes
                .checked_add(seed.charged_bytes)
                .ok_or(PrivateSeedError::Budget)?;
            while state.slots[role].len() + 1 - evict > state.limits.slots_per_role
                || prospective > state.limits.max_bank_bytes
            {
                let victim = state.slots[role]
                    .get(evict)
                    .ok_or(PrivateSeedError::Budget)?;
                if Arc::strong_count(victim) != 1 {
                    return Err(PrivateSeedError::Budget);
                }
                prospective = prospective
                    .checked_sub(victim.charged_bytes)
                    .ok_or(PrivateSeedError::Budget)?;
                evict += 1;
            }
            let count = state
                .counts
                .accepted_seeds
                .checked_add(1)
                .ok_or(PrivateSeedError::GenerationExhausted)?;
            check_control(cancelled, deadline.min(pending.deadline))?;
            check_control(&pending.cancelled, pending.deadline)?;
            // Every fallible preflight is complete before modifying valid entries.
            drop(state.slots[role].drain(..evict));
            state.slots[role].push(seed.clone());
            state.bank_bytes = prospective;
            state.seed_sequence = seed.provenance.seed_sequence;
            state.counts.accepted_seeds = count;
            state.pending.take();
            Ok(seed.provenance)
        })();
        if let Err(error) = result.as_ref() {
            state.last_failure = Some(*error);
        }
        if result.is_ok() {
            self.finalized = true;
        }
        result
    }
}
impl Drop for ProvisionalPrivateSeed {
    fn drop(&mut self) {
        if !self.finalized {
            abort_pending(&self.inner, self.id);
        }
    }
}

fn lock(inner: &Arc<Mutex<State>>) -> Result<MutexGuard<'_, State>, PrivateSeedError> {
    inner.lock().map_err(|_| PrivateSeedError::Poisoned)
}
fn checked_bytes(elements: usize, element_bytes: usize) -> Result<u64, PrivateSeedError> {
    elements
        .checked_mul(element_bytes)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(PrivateSeedError::Budget)
}
fn abort_pending(inner: &Arc<Mutex<State>>, id: u64) {
    if let Ok(mut state) = inner.lock() {
        if state.pending.as_ref().is_some_and(|p| p.id == id) {
            state.pending.take();
        }
    }
}
fn check_control(cancelled: &CancelToken, deadline: Instant) -> Result<(), PrivateSeedError> {
    if cancelled.is_canceled() {
        return Err(PrivateSeedError::Cancelled);
    }
    if Instant::now() >= deadline {
        return Err(PrivateSeedError::Deadline);
    }
    Ok(())
}
fn role_index(role: PalsRole) -> usize {
    match role {
        PalsRole::Proposer => 0,
        PalsRole::Critic => 1,
        PalsRole::Validator => 2,
    }
}
fn hash_request(hash: &mut Sha256, request: &PrivateSeedRequest) {
    hash.update([role_index(request.role) as u8]);
    hash.update(request.model.model);
    hash.update(request.model.encoding);
    hash.update(b"fp32");
    hash.update(request.model.model_epoch);
    hash.update(request.model.frozen_epoch.to_le_bytes());
    hash.update(request.game_generation.to_le_bytes());
    hash.update(request.search_generation.to_le_bytes());
    hash.update(request.context.prepared_input);
    hash.update(request.context.history);
    hash.update(request.context.non_record_context);
    hash.update(request.context.rules_revision.to_le_bytes());
    hash.update(request.context.record_revision.to_le_bytes());
    hash.update(request.context.situation.situation.slot.to_le_bytes());
    hash.update(request.context.situation.situation.generation.to_le_bytes());
    hash.update(request.context.situation.prefix);
    hash.update(request.context.situation.focus);
}
fn invocation(
    request: &PrivateSeedRequest,
    seed: Option<&Seed>,
    mode: PrivateInvocationMode,
) -> PrivateInvocation {
    let input_key = request.context.prepared_input;
    match seed {
        None => PrivateInvocation::Fresh { input_key },
        Some(seed) => {
            let mut hash = Sha256::new();
            hash.update(if mode == PrivateInvocationMode::ApproxCudaWarmV2 {
                PRIVATE_CUDA_WARM_SEMANTICS.as_bytes()
            } else {
                PRIVATE_WARM_SEMANTICS.as_bytes()
            });
            hash_request(&mut hash, request);
            hash.update(seed.provenance.seal);
            let invocation_key = hash.finalize().into();
            if mode == PrivateInvocationMode::ApproxCudaWarmV2 {
                PrivateInvocation::ApproxCudaWarmV2 {
                    input_key,
                    seed_seal: seed.provenance.seal,
                    invocation_key,
                }
            } else {
                PrivateInvocation::ApproxWarmV1 {
                    input_key,
                    seed_seal: seed.provenance.seal,
                    invocation_key,
                }
            }
        }
    }
}
fn provenance(
    request: &PrivateSeedRequest,
    invocation: PrivateInvocation,
    lease_id: u64,
    sequence: u64,
    bits: &[u32],
) -> PrivateSeedProvenance {
    let mut latent = Sha256::new();
    latent.update(b"rz-pals-private-finite-fp32-latent/1");
    latent.update((bits.len() as u64).to_le_bytes());
    for value in bits {
        latent.update(value.to_le_bytes());
    }
    let latent_bits_digest = latent.finalize().into();
    let mut seal = Sha256::new();
    seal.update(PRIVATE_WARM_SEMANTICS.as_bytes());
    hash_request(&mut seal, request);
    match invocation {
        PrivateInvocation::Fresh { input_key } => {
            seal.update([0]);
            seal.update(input_key);
        }
        PrivateInvocation::ApproxWarmV1 {
            invocation_key,
            seed_seal,
            ..
        } => {
            seal.update([1]);
            seal.update(invocation_key);
            seal.update(seed_seal);
        }
        PrivateInvocation::ApproxCudaWarmV2 {
            invocation_key,
            seed_seal,
            ..
        } => {
            seal.update([2]);
            seal.update(PRIVATE_CUDA_WARM_SEMANTICS.as_bytes());
            seal.update(invocation_key);
            seal.update(seed_seal);
        }
    }
    seal.update(sequence.to_le_bytes());
    seal.update(lease_id.to_le_bytes());
    seal.update(latent_bits_digest);
    PrivateSeedProvenance {
        role: request.role,
        model: request.model,
        game_generation: request.game_generation,
        search_generation: request.search_generation,
        source_input: request.context.prepared_input,
        source_history: request.context.history,
        source_non_record_context: request.context.non_record_context,
        source_record_revision: request.context.record_revision,
        source_rules_revision: request.context.rules_revision,
        source_situation: request.context.situation,
        source_invocation: invocation,
        source_lease_id: lease_id,
        seed_sequence: sequence,
        latent_bits_digest,
        seal: seal.finalize().into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_model::{PalsCandidateToken, PalsRecordToken};
    use rz_position::Position;
    use std::time::Duration;

    fn limits() -> PrivateSeedLimits {
        PrivateSeedLimits {
            slots_per_role: 2,
            max_bank_bytes: 256 * 1024,
            max_transient_bytes: 128 * 1024,
            latent_elements: MAX_LATENT_ELEMENTS,
        }
    }
    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(30)
    }
    fn input(role: PalsRole) -> PalsModelInput {
        PalsModelInput {
            role,
            board: vec![0; 64],
            metadata: [0.; 16],
            records: Vec::new(),
            required_critical_records: Vec::new(),
            candidates: vec![PalsCandidateToken {
                from: 12,
                to: 28,
                promotion: 0,
            }],
            divergence_features: Vec::new(),
            query: [0.; 16],
            situation_revision: 0,
            history_digest: [4; 32],
            model_epoch: [5; 32],
            full_line: None,
        }
    }
    fn request(position: &PositionSnapshot, input: &PalsModelInput) -> PrivateSeedRequest {
        PrivateSeedRequest {
            role: input.role,
            model: PrivateModelIdentity {
                model: [1; 32],
                encoding: [2; 32],
                precision: PrivatePrecision::Fp32,
                model_epoch: input.model_epoch,
                frozen_epoch: 9,
            },
            game_generation: 1,
            search_generation: 2,
            context: PrivateRulesContext::seal(
                position,
                input,
                &PalsModelConfig::baseline(),
                PrivateSituationContext {
                    situation: SituationHandle {
                        slot: 3,
                        generation: 4,
                    },
                    prefix: [6; 32],
                    focus: [7; 32],
                },
            )
            .unwrap(),
        }
    }
    /// Numerical boundary fixture only: reads *every* actual seed element.
    /// This is neither a trained model nor a registered native graph.
    fn consume(lease: &PrivateSeedLease) -> Vec<f32> {
        (0..MAX_LATENT_ELEMENTS)
            .map(|i| {
                let previous = lease.seed_bits().map_or(0., |bits| f32::from_bits(bits[i]));
                previous * 0.5 + (i % 7) as f32 * 0.125 + 0.25
            })
            .collect()
    }
    fn accepted(
        bank: &PrivateSeedBank,
        request: &PrivateSeedRequest,
        snapshot: &PositionSnapshot,
        mode: PrivateInvocationMode,
    ) -> (PrivateSeedProvenance, Vec<f32>) {
        let cancelled = CancelToken::new();
        let lease = bank
            .begin(request.clone(), snapshot, mode, &cancelled, deadline())
            .unwrap();
        let values = consume(&lease);
        let result = lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap()
            .stage_output(&values)
            .unwrap()
            .commit(request, snapshot, &cancelled, deadline())
            .unwrap();
        (result, values)
    }

    #[test]
    fn base_reservation_accounts_for_the_full_outward_lease_identity() {
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let reserved = bank.snapshot().unwrap().reserved_bank_bytes;
        let lease_and_owner =
            size_of::<State>() as u64 + size_of::<PrivateSeedLease>() as u64 + OWNER_OVERHEAD;
        assert!(reserved >= lease_and_owner);
        let mut exact = limits();
        exact.max_bank_bytes = reserved;
        assert!(PrivateSeedBank::new(exact, 1).is_ok());
        exact.max_bank_bytes -= 1;
        assert!(matches!(
            PrivateSeedBank::new(exact, 1),
            Err(PrivateSeedError::Budget)
        ));
    }
    #[test]
    fn cuda_invocation_has_a_distinct_key_and_keeps_unknown_seed_pinned() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let input = input(PalsRole::Proposer);
        let req = request(&snapshot, &input);
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let (accepted, _) = accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
        let cancel = CancelToken::new();
        let v1 = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancel,
                deadline(),
            )
            .unwrap();
        let PrivateInvocation::ApproxWarmV1 {
            invocation_key: v1_key,
            seed_seal,
            ..
        } = v1.invocation()
        else {
            panic!("expected v1");
        };
        assert_eq!(seed_seal, accepted.seal);
        drop(v1.complete_known(PhysicalSeedCompletion::Failed).unwrap());
        let cuda = bank
            .begin(
                req,
                &snapshot,
                PrivateInvocationMode::ApproxCudaWarmV2,
                &cancel,
                deadline(),
            )
            .unwrap();
        let PrivateInvocation::ApproxCudaWarmV2 {
            invocation_key: cuda_key,
            seed_seal,
            ..
        } = cuda.invocation()
        else {
            panic!("expected CUDA v2");
        };
        assert_eq!(seed_seal, accepted.seal);
        assert_ne!(cuda_key, v1_key);
        let id = cuda.id();
        drop(cuda); // Unknown is not a successful completion or seed publication.
        let state = bank.snapshot().unwrap();
        assert!(state.quarantined && state.admission_closed);
        assert_eq!(state.active_lease, Some(id));
        assert_eq!(state.pinned_entries, 1);
        assert_eq!(state.counts.accepted_seeds, 1);
        assert!(matches!(
            bank.reset_game(2),
            Err(PrivateSeedError::ActivePhysicalLease)
        ));
    }
    #[test]
    fn cuda_full_line_query_change_cannot_reuse_same_rules_seed() {
        use crate::pals_model::{PalsFullLineInput, PalsModelProfile};
        let snapshot = Position::startpos().snapshot();
        let config = PalsModelConfig::for_profile(PalsModelProfile::FullLineV2);
        let mut input = input(PalsRole::Proposer);
        input.full_line = Some(PalsFullLineInput {
            records: vec![],
            query_prefix: input.candidates.clone(),
            query_proposal: input.candidates.clone(),
            query_counter: vec![],
        });
        let mut req = request(&snapshot, &super::tests::input(PalsRole::Proposer));
        let situation = req.context.situation;
        req.context = PrivateRulesContext::seal(&snapshot, &input, &config, situation).unwrap();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
        input.situation_revision = 1;
        let current = PrivateRulesContext::seal(&snapshot, &input, &config, situation).unwrap();
        assert!(req.context.warm_eligible(&current, &snapshot));
        input.full_line.as_mut().unwrap().query_proposal[0].to = 20;
        req.context = PrivateRulesContext::seal(&snapshot, &input, &config, situation).unwrap();
        assert!(matches!(
            bank.begin(
                req,
                &snapshot,
                PrivateInvocationMode::ApproxCudaWarmV2,
                &CancelToken::new(),
                deadline()
            ),
            Err(PrivateSeedError::MissingSeed)
        ));
    }
    #[test]
    fn exact_revocation_is_idempotent_and_preserves_other_authorities_and_counts() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let proposer = request(&snapshot, &input(PalsRole::Proposer));
        let critic = request(&snapshot, &input(PalsRole::Critic));
        let (older, _) = accepted(&bank, &proposer, &snapshot, PrivateInvocationMode::Fresh);
        let (target, _) = accepted(&bank, &proposer, &snapshot, PrivateInvocationMode::Fresh);
        let (other_role, _) = accepted(&bank, &critic, &snapshot, PrivateInvocationMode::Fresh);
        let before = bank.snapshot().unwrap();
        let changed: [fn(&mut PrivateSeedProvenance); 5] = [
            |authority| authority.role = PalsRole::Critic,
            |authority| authority.model.encoding[0] ^= 1,
            |authority| authority.source_non_record_context[0] ^= 1,
            |authority| authority.source_lease_id += 1,
            |authority| authority.seal[0] ^= 1,
        ];
        for change in changed {
            let mut different = target;
            change(&mut different);
            let refusal = bank
                .revoke_committed(&different, PrivateCancelReason::Cancelled)
                .unwrap();
            assert!(!refusal.revoked);
            assert_eq!(refusal.removed_bank_bytes, 0);
            assert_eq!(refusal.remaining_entries_per_role, before.entries_per_role);
            assert_eq!(bank.snapshot().unwrap(), before);
        }
        let target_bytes = bank.inner.lock().unwrap().slots[0][1].charged_bytes;
        let removed = bank
            .revoke_committed(&target, PrivateCancelReason::Cancelled)
            .unwrap();
        assert!(removed.revoked);
        assert_eq!(removed.provenance, target);
        assert_eq!(removed.removed_bank_bytes, target_bytes);
        assert_eq!(removed.remaining_entries_per_role, [1, 1, 0]);
        let after = bank.snapshot().unwrap();
        assert_eq!(
            after.reserved_bank_bytes,
            before.reserved_bank_bytes - target_bytes
        );
        assert_eq!(after.counts, before.counts);
        assert_eq!(after.revoked_seeds, 1);
        assert_eq!(after.last_revocation, Some(removed));
        let repeated = bank
            .revoke_committed(&target, PrivateCancelReason::Deadline)
            .unwrap();
        assert!(!repeated.revoked);
        assert_eq!(repeated.removed_bank_bytes, 0);
        assert_eq!(bank.snapshot().unwrap(), after);
        let state = bank.inner.lock().unwrap();
        assert_eq!(state.slots[0][0].provenance, older);
        assert_eq!(state.slots[1][0].provenance, other_role);
    }
    #[test]
    fn revocation_does_not_restore_an_evicted_seed_and_a_missing_seed_can_start_fresh() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let mut one_slot = limits();
        one_slot.slots_per_role = 1;
        let bank = PrivateSeedBank::new(one_slot, 1).unwrap();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        let (evicted, _) = accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
        let (target, _) = accepted(&bank, &req, &snapshot, PrivateInvocationMode::ApproxWarmV1);
        let before = bank.snapshot().unwrap();
        assert!(
            bank.revoke_committed(&target, PrivateCancelReason::Deadline)
                .unwrap()
                .revoked
        );
        assert!(
            !bank
                .revoke_committed(&evicted, PrivateCancelReason::Deadline)
                .unwrap()
                .revoked
        );
        let after = bank.snapshot().unwrap();
        assert_eq!(after.entries_per_role, [0; 3]);
        assert_eq!(after.counts, before.counts);
        assert_eq!(after.revoked_seeds, 1);
        let cancelled = CancelToken::new();
        assert!(matches!(
            bank.begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancelled,
                deadline()
            ),
            Err(PrivateSeedError::MissingSeed)
        ));
        let fresh = bank
            .begin(
                req,
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancelled,
                deadline(),
            )
            .unwrap();
        assert!(matches!(
            fresh.invocation(),
            PrivateInvocation::Fresh { .. }
        ));
        assert!(fresh.seed_bits().is_none());
        // This bank-boundary fixture acknowledges failure only to clean up its
        // logical fixture lease; it is not Native numerical/fence evidence.
        drop(
            fresh
                .complete_known(PhysicalSeedCompletion::Failed)
                .unwrap(),
        );
    }
    #[test]
    fn revocation_closes_admission_without_releasing_active_pending_or_unknown_pins() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        for mode in 0..3 {
            let bank = PrivateSeedBank::new(limits(), 1).unwrap();
            let (target, _) = accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
            let cancelled = CancelToken::new();
            let lease = bank
                .begin(
                    req.clone(),
                    &snapshot,
                    PrivateInvocationMode::ApproxWarmV1,
                    &cancelled,
                    deadline(),
                )
                .unwrap();
            let id = lease.id();
            let mut active = Some(lease);
            let mut pending = None;
            let error = match mode {
                0 => PrivateSeedError::ActivePhysicalLease,
                1 => {
                    let lease = active.take().unwrap();
                    let values = consume(&lease);
                    pending = Some(
                        lease
                            .complete_known(PhysicalSeedCompletion::Succeeded)
                            .unwrap()
                            .stage_output(&values)
                            .unwrap(),
                    );
                    PrivateSeedError::PendingFinalization
                }
                _ => {
                    drop(active.take());
                    PrivateSeedError::PhysicalCompletionUnknown
                }
            };
            let before = bank.snapshot().unwrap();
            assert_eq!(
                bank.revoke_committed(&target, PrivateCancelReason::Cancelled),
                Err(error)
            );
            let after = bank.snapshot().unwrap();
            assert!(after.admission_closed);
            assert_eq!(after.last_failure, Some(error));
            assert_eq!(after.entries_per_role, before.entries_per_role);
            assert_eq!(after.reserved_bank_bytes, before.reserved_bank_bytes);
            assert_eq!(after.provisional_bytes, before.provisional_bytes);
            assert_eq!(after.active_lease, before.active_lease);
            assert_eq!(after.active_seed_bytes, before.active_seed_bytes);
            assert_eq!(after.pinned_entries, before.pinned_entries);
            assert_eq!(after.quarantined, before.quarantined);
            assert_eq!(after.counts, before.counts);
            assert_eq!(after.revoked_seeds, 0);
            assert!(after.last_revocation.is_none());
            assert!(!cancelled.is_canceled());
            if let Some(lease) = active.take() {
                assert_eq!(lease.seed_provenance(), Some(&target));
                drop(
                    lease
                        .complete_known(PhysicalSeedCompletion::Failed)
                        .unwrap(),
                );
            } else if mode == 2 {
                bank.resolve_unknown_known_completion(id, PhysicalSeedCompletion::Failed)
                    .unwrap();
            }
            drop(pending);
        }
    }
    #[test]
    fn revocation_counter_and_byte_preflight_failure_preserve_the_exact_seed() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        for count_overflow in [true, false] {
            let bank = PrivateSeedBank::new(limits(), 1).unwrap();
            let (target, _) = accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
            {
                let mut state = bank.inner.lock().unwrap();
                if count_overflow {
                    state.revoked_seeds = u64::MAX;
                } else {
                    state.bank_bytes = state.base_bytes + state.slots[0][0].charged_bytes - 1;
                }
            }
            let before = bank.snapshot().unwrap();
            let expected = if count_overflow {
                PrivateSeedError::GenerationExhausted
            } else {
                PrivateSeedError::Budget
            };
            assert_eq!(
                bank.revoke_committed(&target, PrivateCancelReason::Deadline),
                Err(expected)
            );
            let after = bank.snapshot().unwrap();
            assert!(after.admission_closed);
            assert_eq!(after.entries_per_role, before.entries_per_role);
            assert_eq!(after.reserved_bank_bytes, before.reserved_bank_bytes);
            assert_eq!(after.counts, before.counts);
            assert_eq!(after.revoked_seeds, before.revoked_seeds);
            assert!(after.last_revocation.is_none());
            assert_eq!(bank.inner.lock().unwrap().slots[0][0].provenance, target);
        }
    }
    #[test]
    fn actual_seed_consumption_changes_output_and_preserves_fresh_keys() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let data = input(PalsRole::Proposer);
        let req = request(&snapshot, &data);
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let cancelled = CancelToken::new();
        let fresh = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancelled,
                deadline(),
            )
            .unwrap();
        assert_eq!(
            fresh.invocation(),
            PrivateInvocation::Fresh {
                input_key: data
                    .canonical_input_key(&PalsModelConfig::baseline())
                    .unwrap()
            }
        );
        let first = consume(&fresh);
        let proof = fresh
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap()
            .stage_output(&first)
            .unwrap()
            .commit(&req, &snapshot, &cancelled, deadline())
            .unwrap();
        let warm = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancelled,
                deadline(),
            )
            .unwrap();
        assert!(
            matches!(warm.invocation(), PrivateInvocation::ApproxWarmV1 { seed_seal, .. } if seed_seal == proof.seal)
        );
        assert_eq!(warm.seed_bits().unwrap().len(), MAX_LATENT_ELEMENTS);
        assert_eq!(bank.snapshot().unwrap().pinned_entries, 1);
        let second = consume(&warm);
        assert_ne!(first, second);
        warm.complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap()
            .stage_output(&second)
            .unwrap()
            .commit(&req, &snapshot, &cancelled, deadline())
            .unwrap();
        assert_eq!(bank.snapshot().unwrap().pinned_entries, 0);
        assert_eq!(
            bank.snapshot().unwrap().counts,
            PrivateSeedCounts {
                fresh_admissions: 1,
                warm_admissions: 1,
                accepted_seeds: 2,
                known_completions: 2,
                recovered_unknown_completions: 0
            }
        );
        assert_eq!(
            bank.warm_support(),
            PrivateWarmSupport::RustConsumerBoundaryOnly
        );
        assert_eq!(
            require_native_warm_start(),
            Err(PrivateSeedError::UnsupportedNativeWarmStart)
        );
    }

    #[test]
    fn role_model_epoch_game_and_exact_history_are_not_shared() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
        let cancelled = CancelToken::new();
        let mut changed = request(&snapshot, &input(PalsRole::Critic));
        assert!(matches!(
            bank.begin(
                changed.clone(),
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancelled,
                deadline()
            ),
            Err(PrivateSeedError::MissingSeed)
        ));
        changed = request(&snapshot, &input(PalsRole::Validator));
        assert!(matches!(
            bank.begin(
                changed,
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancelled,
                deadline()
            ),
            Err(PrivateSeedError::UnsupportedRole)
        ));
        for mutate in [0, 1, 2] {
            let mut changed = req.clone();
            match mutate {
                0 => changed.model.model[0] ^= 1,
                1 => {
                    let mut data = input(PalsRole::Proposer);
                    data.model_epoch[0] ^= 1;
                    changed = request(&snapshot, &data);
                }
                _ => changed.search_generation += 1,
            }
            assert!(matches!(
                bank.begin(
                    changed,
                    &snapshot,
                    PrivateInvocationMode::ApproxWarmV1,
                    &cancelled,
                    deadline()
                ),
                Err(PrivateSeedError::MissingSeed)
            ));
        }
        let reconstructed = Position::startpos().snapshot();
        assert!(matches!(
            bank.begin(
                req.clone(),
                &reconstructed,
                PrivateInvocationMode::Fresh,
                &cancelled,
                deadline()
            ),
            Err(PrivateSeedError::StaleRules)
        ));
        bank.reset_game(2).unwrap();
        assert_eq!(bank.snapshot().unwrap().entries_per_role, [0; 3]);
        assert!(matches!(
            bank.begin(
                req,
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancelled,
                deadline()
            ),
            Err(PrivateSeedError::StaleGame)
        ));
    }

    #[test]
    fn record_revision_can_warm_but_focus_query_and_revision_rollback_cannot() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let mut data = input(PalsRole::Proposer);
        let original = request(&snapshot, &data);
        let (source, _) = accepted(&bank, &original, &snapshot, PrivateInvocationMode::Fresh);
        data.records.push(PalsRecordToken {
            record_id: 8,
            revision: 1,
            critical: true,
            features: [0.; 16],
        });
        data.required_critical_records.push(8);
        data.situation_revision = 1;
        let updated = request(&snapshot, &data);
        let (target, _) = accepted(
            &bank,
            &updated,
            &snapshot,
            PrivateInvocationMode::ApproxWarmV1,
        );
        assert_ne!(source.source_input, target.source_input);
        assert_ne!(source.seal, target.seal);
        let cancelled = CancelToken::new();
        let mut different = updated.clone();
        different.context.situation.focus[0] ^= 1;
        assert!(matches!(
            bank.begin(
                different,
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancelled,
                deadline()
            ),
            Err(PrivateSeedError::MissingSeed)
        ));
        data.query[0] = 1.;
        let different = request(&snapshot, &data);
        assert!(matches!(
            bank.begin(
                different,
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancelled,
                deadline()
            ),
            Err(PrivateSeedError::MissingSeed)
        ));
        // An old matching seed may still exist in slot 2; rollback is not allowed
        // to consume the *newer* seed. Its actual provenance stays explicit.
        let old = bank
            .begin(
                original,
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancelled,
                deadline(),
            )
            .unwrap();
        assert_eq!(old.seed_provenance().unwrap().source_record_revision, 0);
        old.complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap();
    }

    #[test]
    fn truncation_nonfinite_and_stale_or_cancelled_outputs_never_commit() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        let cancelled = CancelToken::new();
        for invalid in [
            vec![0.; MAX_LATENT_ELEMENTS - 1],
            vec![f32::NAN; MAX_LATENT_ELEMENTS],
            vec![f32::INFINITY; MAX_LATENT_ELEMENTS],
        ] {
            let lease = bank
                .begin(
                    req.clone(),
                    &snapshot,
                    PrivateInvocationMode::Fresh,
                    &cancelled,
                    deadline(),
                )
                .unwrap();
            assert!(lease
                .complete_known(PhysicalSeedCompletion::Succeeded)
                .unwrap()
                .stage_output(&invalid)
                .is_err());
            assert_eq!(bank.snapshot().unwrap().entries_per_role, [0; 3]);
        }
        let lease = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancelled,
                deadline(),
            )
            .unwrap();
        let values = consume(&lease);
        let provisional = lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap()
            .stage_output(&values)
            .unwrap();
        let mut changed = req.clone();
        changed.search_generation += 1;
        assert_eq!(
            provisional.commit(&changed, &snapshot, &cancelled, deadline()),
            Err(PrivateSeedError::StaleInvocation)
        );
        let lease = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancelled,
                deadline(),
            )
            .unwrap();
        let values = consume(&lease);
        let provisional = lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap()
            .stage_output(&values)
            .unwrap();
        cancelled.cancel();
        assert_eq!(
            provisional.commit(&req, &snapshot, &cancelled, deadline()),
            Err(PrivateSeedError::Cancelled)
        );
        assert_eq!(bank.snapshot().unwrap().entries_per_role, [0; 3]);
    }

    #[test]
    fn unknown_keeps_pin_and_rejects_reuse_reset_until_actual_fence() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
        let cancelled = CancelToken::new();
        let lease = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancelled,
                deadline(),
            )
            .unwrap();
        let id = lease.id();
        bank.cancel_active(id, PrivateCancelReason::Cancelled)
            .unwrap();
        drop(lease);
        let state = bank.snapshot().unwrap();
        assert!(state.quarantined);
        assert_eq!(state.pinned_entries, 1);
        assert!(matches!(
            bank.begin(
                req,
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancelled,
                deadline()
            ),
            Err(PrivateSeedError::PhysicalCompletionUnknown)
        ));
        assert_eq!(
            bank.reset_game(2),
            Err(PrivateSeedError::ActivePhysicalLease)
        );
        assert_eq!(bank.snapshot().unwrap().game_generation, 1);
        bank.resolve_unknown_known_completion(id, PhysicalSeedCompletion::Succeeded)
            .unwrap();
        assert_eq!(bank.snapshot().unwrap().pinned_entries, 0);
        assert!(bank.snapshot().unwrap().admission_closed);
        assert_eq!(
            bank.resolve_unknown_known_completion(id, PhysicalSeedCompletion::Succeeded),
            Err(PrivateSeedError::InvalidLease)
        );
        bank.reset_game(2).unwrap();
        assert_eq!(bank.snapshot().unwrap().entries_per_role, [0; 3]);
    }

    #[test]
    fn budgets_preserve_old_role_seeds_and_slots_are_bounded() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        for _ in 0..5 {
            accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
        }
        assert_eq!(bank.snapshot().unwrap().entries_per_role, [2, 0, 0]);
        let before = bank.snapshot().unwrap().reserved_bank_bytes;
        {
            let mut state = bank.inner.lock().unwrap();
            state.limits.max_bank_bytes = before;
        }
        let critic = request(&snapshot, &input(PalsRole::Critic));
        let cancelled = CancelToken::new();
        let lease = bank
            .begin(
                critic.clone(),
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancelled,
                deadline(),
            )
            .unwrap();
        let values = consume(&lease);
        let provisional = lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap()
            .stage_output(&values)
            .unwrap();
        assert_eq!(
            provisional.commit(&critic, &snapshot, &cancelled, deadline()),
            Err(PrivateSeedError::Budget)
        );
        let after = bank.snapshot().unwrap();
        assert_eq!(after.entries_per_role, [2, 0, 0]);
        assert_eq!(after.reserved_bank_bytes, before);
        let lease = bank
            .begin(
                req,
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancelled,
                deadline(),
            )
            .unwrap();
        lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap();
    }

    #[test]
    fn actual_bits_negative_zero_and_subnormal_are_sealed_without_normalization() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        let cancelled = CancelToken::new();
        let mut values = vec![0.; MAX_LATENT_ELEMENTS];
        values[0] = -0.;
        values[1] = f32::from_bits(1);
        let lease = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancelled,
                deadline(),
            )
            .unwrap();
        lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap()
            .stage_output(&values)
            .unwrap()
            .commit(&req, &snapshot, &cancelled, deadline())
            .unwrap();
        let warm = bank
            .begin(
                req,
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancelled,
                deadline(),
            )
            .unwrap();
        assert_eq!(warm.seed_bits().unwrap()[0], (-0f32).to_bits());
        assert_eq!(warm.seed_bits().unwrap()[1], 1);
        warm.complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap();
    }

    #[test]
    fn admitted_cancel_and_deadline_cannot_be_replaced_by_new_controls() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        let admitted = CancelToken::new();
        let replacement = CancelToken::new();
        let lease = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::Fresh,
                &admitted,
                deadline(),
            )
            .unwrap();
        let values = consume(&lease);
        let provisional = lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap()
            .stage_output(&values)
            .unwrap();
        admitted.cancel();
        assert_eq!(
            provisional.commit(&req, &snapshot, &replacement, deadline()),
            Err(PrivateSeedError::Cancelled)
        );
        assert_eq!(bank.snapshot().unwrap().entries_per_role, [0; 3]);
        let lease = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::Fresh,
                &replacement,
                deadline(),
            )
            .unwrap();
        let values = consume(&lease);
        let provisional = lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap()
            .stage_output(&values)
            .unwrap();
        // Make the admitted bound expired without sleeping or timing a workload.
        bank.inner
            .lock()
            .unwrap()
            .pending
            .as_mut()
            .unwrap()
            .deadline = Instant::now();
        assert_eq!(
            provisional.commit(&req, &snapshot, &replacement, deadline()),
            Err(PrivateSeedError::Deadline)
        );
        assert_eq!(bank.snapshot().unwrap().entries_per_role, [0; 3]);
    }

    #[test]
    fn known_failure_and_expired_output_never_become_accepted_seeds() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        let cancel = CancelToken::new();
        let lease = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancel,
                deadline(),
            )
            .unwrap();
        let id = lease.id();
        let values = consume(&lease);
        let complete = lease
            .complete_known(PhysicalSeedCompletion::Failed)
            .unwrap();
        assert!(matches!(
            complete.stage_output(&values),
            Err(PrivateSeedError::PhysicalFailure)
        ));
        let observed = bank.snapshot().unwrap();
        assert_eq!(observed.last_completed_lease, Some(id));
        assert_eq!(
            observed.last_physical_completion,
            Some(PhysicalSeedCompletion::Failed)
        );
        assert_eq!(
            observed.last_failure,
            Some(PrivateSeedError::PhysicalFailure)
        );
        let lease = bank
            .begin(
                req,
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancel,
                deadline(),
            )
            .unwrap();
        bank.inner.lock().unwrap().active.as_mut().unwrap().deadline = Instant::now();
        let complete = lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap();
        assert!(matches!(
            complete.stage_output(&values),
            Err(PrivateSeedError::Deadline)
        ));
        assert_eq!(bank.snapshot().unwrap().entries_per_role, [0; 3]);
        assert_eq!(bank.snapshot().unwrap().provisional_bytes, 0);
    }

    #[test]
    fn reset_during_active_work_closes_admission_and_preserves_until_known_completion() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        let cancel = CancelToken::new();
        accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
        let lease = bank
            .begin(
                req,
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancel,
                deadline(),
            )
            .unwrap();
        let values = consume(&lease);
        assert_eq!(
            bank.reset_game(2),
            Err(PrivateSeedError::ActivePhysicalLease)
        );
        let active = bank.snapshot().unwrap();
        assert!(active.admission_closed);
        assert_eq!(active.entries_per_role, [1, 0, 0]);
        assert_eq!(active.pinned_entries, 1);
        let complete = lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap();
        assert!(matches!(
            complete.stage_output(&values),
            Err(PrivateSeedError::AdmissionClosed)
        ));
        assert_eq!(bank.snapshot().unwrap().pinned_entries, 0);
        bank.reset_game(2).unwrap();
        assert_eq!(bank.snapshot().unwrap().entries_per_role, [0; 3]);
    }

    #[test]
    fn dropped_bank_keeps_unknown_owner_but_actual_fence_can_release_it() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        let cancel = CancelToken::new();
        accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
        let owner = Arc::downgrade(&bank.inner);
        let lease = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancel,
                deadline(),
            )
            .unwrap();
        drop(bank);
        assert!(owner.upgrade().is_some());
        let complete = lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap();
        drop(complete);
        assert!(owner.upgrade().is_none());

        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
        let owner = Arc::downgrade(&bank.inner);
        let lease = bank
            .begin(
                req,
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancel,
                deadline(),
            )
            .unwrap();
        let id = lease.id();
        drop(lease);
        drop(bank);
        let retained = owner
            .upgrade()
            .expect("actual seed pin retained in quarantine");
        assert!(retained
            .lock()
            .unwrap()
            .active
            .as_ref()
            .unwrap()
            .seed
            .is_some());
        // The test supplies a simulated actual fence; without it the owner is
        // intentionally retained. No new seed/admission follows this recovery.
        let recovered = PrivateSeedBank { inner: retained };
        recovered
            .resolve_unknown_known_completion(id, PhysicalSeedCompletion::Failed)
            .unwrap();
        drop(recovered);
        assert!(owner.upgrade().is_none());
    }

    #[test]
    fn tensor_hash_collision_of_supplied_history_does_not_authorize_rules_reuse() {
        let initial = Position::startpos();
        let initial_snapshot = initial.snapshot();
        let mut repeated = Position::startpos();
        repeated
            .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8"])
            .unwrap();
        let later = repeated.snapshot();
        let data = input(PalsRole::Proposer);
        let first = request(&initial_snapshot, &data);
        let second = request(&later, &data);
        // A faulty caller supplied identical compact input/history bytes. The
        // actual Rules history guard still prevents approximate seed reuse.
        assert_eq!(first.context.prepared_input, second.context.prepared_input);
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        accepted(
            &bank,
            &first,
            &initial_snapshot,
            PrivateInvocationMode::Fresh,
        );
        assert!(matches!(
            bank.begin(
                second,
                &later,
                PrivateInvocationMode::ApproxWarmV1,
                &CancelToken::new(),
                deadline()
            ),
            Err(PrivateSeedError::MissingSeed)
        ));
    }

    #[test]
    fn role_epoch_mismatch_exhaustion_and_transient_budget_are_explicit() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let req = request(&snapshot, &input(PalsRole::Proposer));
        let cancel = CancelToken::new();
        let mut wrong = req.clone();
        wrong.role = PalsRole::Critic;
        assert!(matches!(
            bank.begin(
                wrong,
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancel,
                deadline()
            ),
            Err(PrivateSeedError::InvalidModelInput)
        ));
        let mut wrong = req.clone();
        wrong.model.model_epoch[0] ^= 1;
        assert!(matches!(
            bank.begin(
                wrong,
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancel,
                deadline()
            ),
            Err(PrivateSeedError::InvalidModelInput)
        ));
        accepted(&bank, &req, &snapshot, PrivateInvocationMode::Fresh);
        let before = bank.snapshot().unwrap();
        {
            bank.inner.lock().unwrap().limits.max_transient_bytes = 1;
        }
        let lease = bank
            .begin(
                req.clone(),
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancel,
                deadline(),
            )
            .unwrap();
        let values = consume(&lease);
        assert!(matches!(
            lease
                .complete_known(PhysicalSeedCompletion::Succeeded)
                .unwrap()
                .stage_output(&values),
            Err(PrivateSeedError::Budget)
        ));
        assert_eq!(
            bank.snapshot().unwrap().entries_per_role,
            before.entries_per_role
        );
        assert_eq!(
            bank.snapshot().unwrap().reserved_bank_bytes,
            before.reserved_bank_bytes
        );
        {
            bank.inner.lock().unwrap().next_id = u64::MAX;
        }
        assert!(matches!(
            bank.begin(
                req,
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancel,
                deadline()
            ),
            Err(PrivateSeedError::GenerationExhausted)
        ));
        assert_eq!(bank.snapshot().unwrap().active_lease, None);
    }

    #[test]
    fn proposer_and_critic_read_only_their_own_actual_seed_bits() {
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let bank = PrivateSeedBank::new(limits(), 1).unwrap();
        let cancel = CancelToken::new();
        let proposer = request(&snapshot, &input(PalsRole::Proposer));
        let critic = request(&snapshot, &input(PalsRole::Critic));
        for (req, value) in [(&proposer, 1.25), (&critic, -2.5)] {
            let lease = bank
                .begin(
                    req.clone(),
                    &snapshot,
                    PrivateInvocationMode::Fresh,
                    &cancel,
                    deadline(),
                )
                .unwrap();
            lease
                .complete_known(PhysicalSeedCompletion::Succeeded)
                .unwrap()
                .stage_output(&vec![value; MAX_LATENT_ELEMENTS])
                .unwrap()
                .commit(req, &snapshot, &cancel, deadline())
                .unwrap();
        }
        for (req, expected) in [(proposer, 1.25f32), (critic, -2.5f32)] {
            let lease = bank
                .begin(
                    req.clone(),
                    &snapshot,
                    PrivateInvocationMode::ApproxWarmV1,
                    &cancel,
                    deadline(),
                )
                .unwrap();
            assert_eq!(lease.seed_provenance().unwrap().role, req.role);
            assert!(lease
                .seed_bits()
                .unwrap()
                .iter()
                .all(|bits| *bits == expected.to_bits()));
            assert_eq!(consume(&lease)[0], expected * 0.5 + 0.25);
            lease
                .complete_known(PhysicalSeedCompletion::Succeeded)
                .unwrap();
        }
        let mut epoch = request(&snapshot, &input(PalsRole::Proposer));
        epoch.model.frozen_epoch += 1;
        assert!(matches!(
            bank.begin(
                epoch,
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancel,
                deadline()
            ),
            Err(PrivateSeedError::MissingSeed)
        ));
    }
}
