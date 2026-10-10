//! Bounded CPU Native warm ownership, independent of the legacy Fresh path.
//! All sixteen query FP32 bits are frozen on the first actual preparation.
//! Current public records/revisions still enter the real model input and final
//! acceptance. This is ApproxWarmQueryV1, never an exact-cache optimization.
//! A single unknown/prelaunch-unconfirmed owner retains its actual prepared
//! input and bank; native session/command ownership stays with SingleWorker.

use rz_contracts::{CancelToken, RequestId};
use rz_eval::pals_model::{PalsModelConfig, PalsModelInput, PalsRole};
#[cfg(feature = "experimental-io-binding")]
use rz_eval::pals_onnx::PalsCudaWarmInput;
use rz_eval::pals_onnx::PalsWarmInput;
use rz_eval::pals_private::{
    PhysicalSeedCompletion, PrivateCancelReason, PrivateInvocation, PrivateInvocationMode,
    PrivateModelIdentity, PrivateRulesContext, PrivateSeedBank, PrivateSeedError, PrivateSeedLease,
    PrivateSeedLimits, PrivateSeedRequest, PrivateSituationContext, ProvisionalPrivateSeed,
};
use rz_position::{PositionSnapshot, WeakPositionIdentity};
use rz_search::pals::engine::{RoleAcceptance, RoleError, RoleLogicalContext, RoleQueryPurpose};
use sha2::{Digest as _, Sha256};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub(super) const MAX_FROZEN_CONTEXTS_PER_ROLE: usize = 2;
pub(super) const BANK_BYTES: u64 = 256 * 1024;
pub(super) const TRANSIENT_BYTES: u64 = 256 * 1024;
pub(super) const PAYLOAD_BYTES: u64 = 1024 * 1024;
pub(super) const OWNER_RESERVATION_BYTES: u64 = BANK_BYTES + TRANSIENT_BYTES + 2 * PAYLOAD_BYTES;
#[cfg(feature = "experimental-io-binding")]
pub(super) const CUDA_PAYLOAD_BYTES: u64 = 2 * 1024 * 1024;
#[cfg(feature = "experimental-io-binding")]
pub(super) const CUDA_OWNER_RESERVATION_BYTES: u64 =
    BANK_BYTES + TRANSIENT_BYTES + 2 * CUDA_PAYLOAD_BYTES;

pub(super) fn source_digest() -> [u8; 32] {
    Sha256::digest(include_bytes!("private_warm.rs")).into()
}
fn private_error(error: PrivateSeedError) -> RoleError {
    match error {
        PrivateSeedError::Cancelled => RoleError::Canceled,
        PrivateSeedError::Deadline => RoleError::Deadline,
        PrivateSeedError::PhysicalCompletionUnknown => RoleError::PhysicalCompletionUnknown,
        _ => RoleError::Backend(format!("Native private warm: {error}")),
    }
}
fn role_index(role: PalsRole) -> Result<usize, RoleError> {
    match role {
        PalsRole::Proposer => Ok(0),
        PalsRole::Critic => Ok(1),
        PalsRole::Validator => Err(RoleError::Backend("Native private warm excludes V".into())),
    }
}
fn logical_key(context: &RoleLogicalContext, role: PalsRole) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-native-frozen-query-context/1");
    hash.update([role_index(role).unwrap_or(255) as u8, context.purpose as u8]);
    for value in [
        context.game_generation,
        context.search_generation,
        context.situation.slot as u64,
        context.situation.generation,
        context.state.0 as u64,
        context.focus.0 as u64,
    ] {
        hash.update(value.to_le_bytes());
    }
    hash.update(context.focus_prefix_sha256());
    hash.update(context.proposal_sha256);
    hash.update(context.divergence_sha256);
    hash.update([u8::from(context.refutation_sha256.is_some())]);
    if let Some(refutation) = context.refutation_sha256 {
        hash.update(refutation);
    }
    // Public/situation revisions are current acceptance metadata. They remain
    // in the actual input/context; the query is the entire first actual tensor,
    // not a hash with just deadline/revision feature axes removed.
    hash.finalize().into()
}

struct FrozenQuery {
    key: [u8; 32],
    rules: WeakPositionIdentity,
    rules_revision: u64,
    bits: [u32; 16],
    first_input: [u8; 32],
}
pub(super) struct PreparedWarm {
    pub id: RequestId,
    pub input: Arc<PalsModelInput>,
    pub snapshot: PositionSnapshot,
    pub logical: RoleLogicalContext,
    pub until: Instant,
    pub cancel: CancelToken,
    external_cancel: usize,
    request: Option<PrivateSeedRequest>,
    frozen: Option<Arc<FrozenQuery>>,
    model_config: PalsModelConfig,
}
pub(super) struct WarmPhysical {
    pub prepared: PreparedWarm,
    seed: Option<PrivateSeedLease>,
    // Allocated before bank.begin; retained on unconfirmed handoff/physical
    // completion. The other full payload belongs to the real worker command.
    pub backup_payload: Option<PalsWarmInput>,
    #[cfg(feature = "experimental-io-binding")]
    pub backup_cuda_payload: Option<PalsCudaWarmInput>,
}
impl WarmPhysical {
    pub fn seed(&self) -> Option<&PrivateSeedLease> {
        self.seed.as_ref()
    }
    pub fn invocation(&self) -> Result<PrivateInvocation, RoleError> {
        if let Some(seed) = &self.seed {
            Ok(seed.invocation())
        } else {
            Ok(PrivateInvocation::Fresh {
                input_key: self
                    .prepared
                    .input
                    .canonical_input_key(&self.prepared.model_config)
                    .map_err(|error| RoleError::Backend(error.to_string()))?,
            })
        }
    }
}
struct PendingAcceptance {
    prepared: PreparedWarm,
    provisional: Option<ProvisionalPrivateSeed>,
}

pub(super) struct NativeWarmState {
    model: PrivateModelIdentity,
    model_config: PalsModelConfig,
    mode: PrivateInvocationMode,
    bank: PrivateSeedBank,
    frozen: Mutex<[Vec<Arc<FrozenQuery>>; 2]>,
    pending: Mutex<Option<PendingAcceptance>>,
    retained: Mutex<Option<WarmPhysical>>,
    quarantine_owner: Mutex<Option<Arc<Self>>>,
    closed: AtomicBool,
    query_freezes: AtomicU64,
    query_hits: AtomicU64,
    last_event: Mutex<Option<&'static str>>,
    last_revocation_failure: Mutex<Option<PrivateSeedError>>,
    #[cfg(feature = "experimental-io-binding")]
    followup: Mutex<Option<NativeWarmBankObservation>>,
    #[cfg(feature = "experimental-io-binding")]
    followup_complete: AtomicBool,
}

/// Metadata of the selected CUDA producer. Counters are advanced only by the
/// actual submitted-payload/Ready join, never by a prepared seed or callback.
#[cfg(feature = "experimental-io-binding")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeWarmBankObservation {
    pub owner_id: u64,
    pub event_sequence: u64,
    pub accepted_seed_consumptions: u64,
    pub accepted_seed_context_checked: u64,
    pub rejected_seed_contexts: u64,
    pub fresh_value_evaluations: u64,
    pub value_always_fresh: bool,
    pub closed: bool,
    pub complete: bool,
}
#[cfg(feature = "experimental-io-binding")]
static FOLLOWUP_BANK_OWNERS: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, serde::Serialize)]
pub struct NativePrivateWarmObservation {
    pub game_generation: u64,
    pub entries_per_role: [usize; 3],
    pub reserved_bank_bytes: u64,
    pub provisional_bytes: u64,
    pub pinned_entries: usize,
    pub active_seed_bytes: u64,
    pub active_lease: Option<u64>,
    pub admission_closed: bool,
    pub quarantined: bool,
    pub fresh_seed_admissions: u64,
    pub warm_seed_admissions: u64,
    pub accepted_seeds: u64,
    /// Actual bank publications and exact withdrawals are distinct; the
    /// latter never fabricate a physical completion or restore an old seed.
    pub revoked_seeds: u64,
    pub last_revocation: Option<NativePrivateWarmRevocation>,
    pub last_revocation_failure: Option<String>,
    pub known_seed_completions: u64,
    pub frozen_contexts_per_role: [usize; 2],
    pub query_freezes: u64,
    pub query_hits: u64,
    pub pending_acceptance: bool,
    pub retained_prepared_input: bool,
    pub retained_full_payload_bytes: u64,
    pub first_query_input_sha256_per_role: [Vec<[u8; 32]>; 2],
    pub last_event: Option<&'static str>,
    pub last_bank_failure: Option<String>,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct NativePrivateWarmRevocation {
    pub role: PalsRole,
    pub source_lease_id: u64,
    pub seed_sequence: u64,
    pub provenance_seal: [u8; 32],
    pub reason: String,
    pub revoked: bool,
    pub removed_bank_bytes: u64,
    pub remaining_entries_per_role: [usize; 3],
    pub prior_seed_restored: bool,
}

impl NativeWarmState {
    pub fn new(model: PrivateModelIdentity, game: u64) -> Result<Arc<Self>, RoleError> {
        Self::new_in_domain(
            model,
            game,
            PalsModelConfig::baseline(),
            PrivateInvocationMode::ApproxWarmV1,
        )
    }
    /// Exact-current CUDA query domain. Changed non-record bits and query lines
    /// cause a bank miss, never replacement with an old query tensor.
    #[cfg(feature = "experimental-io-binding")]
    pub fn new_cuda(
        model: PrivateModelIdentity,
        game: u64,
        model_config: PalsModelConfig,
    ) -> Result<Arc<Self>, RoleError> {
        Self::new_in_domain(
            model,
            game,
            model_config,
            PrivateInvocationMode::ApproxCudaWarmV2,
        )
    }
    fn new_in_domain(
        model: PrivateModelIdentity,
        game: u64,
        model_config: PalsModelConfig,
        mode: PrivateInvocationMode,
    ) -> Result<Arc<Self>, RoleError> {
        model_config
            .validate()
            .map_err(|e| RoleError::Backend(e.to_string()))?;
        let bank = PrivateSeedBank::new(
            PrivateSeedLimits {
                slots_per_role: 1,
                max_bank_bytes: BANK_BYTES,
                max_transient_bytes: TRANSIENT_BYTES,
                latent_elements: 16 * 384,
            },
            game,
        )
        .map_err(private_error)?;
        let mut frozen: [Vec<Arc<FrozenQuery>>; 2] = std::array::from_fn(|_| Vec::new());
        for role in &mut frozen {
            role.try_reserve_exact(MAX_FROZEN_CONTEXTS_PER_ROLE)
                .map_err(|_| RoleError::Backend("Native frozen-query allocation failed".into()))?;
        }
        Ok(Arc::new(Self {
            model,
            model_config,
            mode,
            bank,
            frozen: Mutex::new(frozen),
            pending: Mutex::new(None),
            retained: Mutex::new(None),
            quarantine_owner: Mutex::new(None),
            closed: AtomicBool::new(false),
            query_freezes: AtomicU64::new(0),
            query_hits: AtomicU64::new(0),
            last_event: Mutex::new(None),
            last_revocation_failure: Mutex::new(None),
            #[cfg(feature = "experimental-io-binding")]
            followup: Mutex::new(None),
            #[cfg(feature = "experimental-io-binding")]
            followup_complete: AtomicBool::new(true),
        }))
    }
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
    pub fn is_quarantined(&self) -> bool {
        self.quarantine_owner
            .lock()
            .map_or(true, |owner| owner.is_some())
    }
    // Keep the original request identity, Rules snapshot and two independent
    // cancellation authorities explicit at this ownership boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        id: RequestId,
        mut input: PalsModelInput,
        snapshot: PositionSnapshot,
        logical: &RoleLogicalContext,
        until: Instant,
        cancel: CancelToken,
        external_cancel: &AtomicBool,
    ) -> Result<PreparedWarm, RoleError> {
        if self.is_closed() {
            return Err(if self.is_quarantined() {
                RoleError::PhysicalCompletionUnknown
            } else {
                RoleError::Unavailable
            });
        }
        if logical.prefix.len() > 256
            || input.situation_revision != logical.public_revision
            || input.model_epoch != self.model.model_epoch
        {
            return Err(RoleError::Backend(
                "Native warm logical/input namespace mismatch".into(),
            ));
        }
        if cancel.is_canceled() {
            return Err(RoleError::Canceled);
        }
        if Instant::now() >= until {
            return Err(RoleError::Deadline);
        }
        input
            .validate(&self.model_config)
            .map_err(|e| RoleError::Backend(e.to_string()))?;
        let is_value = logical.purpose == RoleQueryPurpose::ValueFresh;
        let frozen = if is_value {
            if input.role != PalsRole::Proposer {
                return Err(RoleError::InvalidOutput);
            }
            None
        } else {
            let role = role_index(input.role)?;
            let expected = match logical.purpose {
                RoleQueryPurpose::ProposePolicy | RoleQueryPurpose::RepairPolicy => {
                    PalsRole::Proposer
                }
                RoleQueryPurpose::ReplyPolicy | RoleQueryPurpose::DivergencePolicy => {
                    PalsRole::Critic
                }
                RoleQueryPurpose::ValueFresh => unreachable!(),
            };
            if input.role != expected {
                return Err(RoleError::InvalidOutput);
            }
            if self.mode == PrivateInvocationMode::ApproxCudaWarmV2 {
                None
            } else {
                let key = logical_key(logical, input.role);
                let mut bank = self.frozen.lock().map_err(|_| RoleError::Unavailable)?;
                let existing = bank[role]
                    .iter()
                    .find(|entry| {
                        entry.key == key
                            && entry.rules_revision == snapshot.revision()
                            && entry.rules.matches(&snapshot)
                    })
                    .cloned();
                let entry = if let Some(entry) = existing {
                    self.query_hits.fetch_add(1, Ordering::AcqRel);
                    entry
                } else {
                    let entry = Arc::new(FrozenQuery {
                        key,
                        rules: snapshot.weak_position_identity(),
                        rules_revision: snapshot.revision(),
                        bits: input.query.map(f32::to_bits),
                        first_input: input
                            .canonical_input_key(&self.model_config)
                            .map_err(|e| RoleError::Backend(e.to_string()))?,
                    });
                    if bank[role].len() == MAX_FROZEN_CONTEXTS_PER_ROLE {
                        if Arc::strong_count(&bank[role][0]) != 1 {
                            return Err(RoleError::Backend(
                                "Native frozen-query pinned capacity exhausted".into(),
                            ));
                        }
                        bank[role].remove(0);
                    }
                    bank[role].push(entry.clone());
                    self.query_freezes.fetch_add(1, Ordering::AcqRel);
                    entry
                };
                input.query = entry.bits.map(f32::from_bits);
                Some(entry)
            }
        };
        let request = if is_value {
            None
        } else {
            let situation = PrivateSituationContext {
                situation: rz_contracts::pals::SituationHandle {
                    slot: u32::try_from(logical.situation.slot)
                        .map_err(|_| RoleError::InvalidOutput)?,
                    generation: logical.situation.generation,
                },
                prefix: logical.prefix_sha256,
                focus: logical_key(logical, input.role),
            };
            Some(PrivateSeedRequest {
                role: input.role,
                model: self.model,
                game_generation: logical.game_generation,
                search_generation: logical.search_generation,
                context: PrivateRulesContext::seal(
                    &snapshot,
                    &input,
                    &self.model_config,
                    situation,
                )
                .map_err(private_error)?,
            })
        };
        Ok(PreparedWarm {
            id,
            input: Arc::new(input),
            snapshot,
            logical: logical.clone(),
            until,
            cancel,
            external_cancel: std::ptr::from_ref(external_cancel) as usize,
            request,
            frozen,
            model_config: self.model_config.clone(),
        })
    }
    pub fn begin(&self, prepared: PreparedWarm) -> Result<WarmPhysical, RoleError> {
        let seed = if let Some(request) = &prepared.request {
            match self.bank.begin(
                request.clone(),
                &prepared.snapshot,
                self.mode,
                &prepared.cancel,
                prepared.until,
            ) {
                Ok(lease) => Some(lease),
                Err(PrivateSeedError::MissingSeed) => Some(
                    self.bank
                        .begin(
                            request.clone(),
                            &prepared.snapshot,
                            PrivateInvocationMode::Fresh,
                            &prepared.cancel,
                            prepared.until,
                        )
                        .map_err(private_error)?,
                ),
                Err(error) => return Err(private_error(error)),
            }
        } else {
            None
        };
        Ok(WarmPhysical {
            prepared,
            seed,
            backup_payload: None,
            #[cfg(feature = "experimental-io-binding")]
            backup_cuda_payload: None,
        })
    }
    /// Called exclusively after actual PhysicalPoll::Ready. Decode/observer
    /// failure is physically complete but cannot stage a successful seed.
    pub fn completed(
        &self,
        mut physical: WarmPhysical,
        latent: Option<&[f32]>,
    ) -> Result<(), RoleError> {
        let provisional = if let Some(seed) = physical.seed.take() {
            let completion = seed
                .complete_known(if latent.is_some() {
                    PhysicalSeedCompletion::Succeeded
                } else {
                    PhysicalSeedCompletion::Failed
                })
                .map_err(private_error)?;
            if let Some(latent) = latent {
                Some(completion.stage_output(latent).map_err(private_error)?)
            } else {
                None
            }
        } else {
            None
        };
        if latent.is_some() {
            let mut pending = self.pending.lock().map_err(|_| RoleError::Unavailable)?;
            if pending.is_some() {
                return Err(RoleError::Backend(
                    "Native warm acceptance slot occupied".into(),
                ));
            }
            *pending = Some(PendingAcceptance {
                prepared: physical.prepared,
                provisional,
            });
        }
        Ok(())
    }
    pub fn reject_output(&self, id: Option<RequestId>) {
        if let Ok(mut pending) = self.pending.lock() {
            if id.is_none() || pending.as_ref().is_some_and(|p| Some(p.prepared.id) == id) {
                pending.take();
            }
        } else {
            self.closed.store(true, Ordering::Release);
        }
    }
    pub fn accept(&self, id: RequestId, acceptance: &RoleAcceptance<'_>) -> Result<(), RoleError> {
        self.accept_with_final_guard(id, acceptance, || acceptance.check_control())
    }
    fn accept_with_final_guard(
        &self,
        id: RequestId,
        acceptance: &RoleAcceptance<'_>,
        final_guard: impl FnOnce() -> Result<(), RoleError>,
    ) -> Result<(), RoleError> {
        acceptance.check_control()?;
        if self.is_closed() {
            return Err(if self.is_quarantined() {
                RoleError::PhysicalCompletionUnknown
            } else {
                RoleError::Unavailable
            });
        }
        let mut pending = self.pending.lock().map_err(|_| RoleError::Unavailable)?;
        let current = pending
            .as_ref()
            .filter(|p| p.prepared.id == id)
            .ok_or_else(|| RoleError::Backend("Native warm unknown acceptance request".into()))?;
        if current.prepared.logical != *acceptance.context
            || current.prepared.until != acceptance.deadline
            || current.prepared.external_cancel != std::ptr::from_ref(acceptance.cancel) as usize
            || current.prepared.snapshot.revision() != acceptance.snapshot.revision()
            || !current
                .prepared
                .snapshot
                .weak_position_identity()
                .matches(acceptance.snapshot)
            || current
                .prepared
                .frozen
                .as_ref()
                .is_some_and(|f| f.bits != current.prepared.input.query.map(f32::to_bits))
        {
            return Err(RoleError::Backend(
                "Native warm stale acceptance context".into(),
            ));
        }
        let mut current = pending.take().expect("checked pending acceptance");
        let committed = if let (Some(seed), Some(request)) = (
            current.provisional.take(),
            current.prepared.request.as_ref(),
        ) {
            Some(
                seed.commit(
                    request,
                    acceptance.snapshot,
                    &current.prepared.cancel,
                    acceptance.deadline,
                )
                .map_err(private_error)?,
            )
        } else {
            None
        };
        if let Err(error) = final_guard() {
            self.closed.store(true, Ordering::Release);
            current.prepared.cancel.cancel();
            if let Some(provenance) = committed {
                let reason = if matches!(error, RoleError::Deadline) {
                    PrivateCancelReason::Deadline
                } else {
                    PrivateCancelReason::Cancelled
                };
                match self.bank.revoke_committed(&provenance, reason) {
                    Ok(revocation) if revocation.revoked => {
                        if let Ok(mut event) = self.last_event.lock() {
                            *event = Some(
                                "control_changed_exact_commit_revoked_prior_seed_not_restored",
                            );
                        }
                    }
                    result => {
                        if let Ok(mut failure) = self.last_revocation_failure.lock() {
                            *failure = Some(result.err().unwrap_or(PrivateSeedError::InvalidLease));
                        }
                        if let Ok(mut event) = self.last_event.lock() {
                            *event = Some("control_changed_exact_commit_withdrawal_unconfirmed");
                        }
                    }
                }
            } else if let Ok(mut event) = self.last_event.lock() {
                *event = Some("fresh_value_final_control_changed_no_seed");
            }
            return Err(error);
        }
        Ok(())
    }
    /// No fake completion is acknowledged on submit failure or unknown native
    /// work. Only one prepared-input retainer/cycle exists. A normal completed
    /// command releases its lease through completed(); this unknown owner has
    /// no automatic recovery, reset, or new dispatch permission.
    pub fn retain_unknown(self: &Arc<Self>, mut physical: WarmPhysical, stage: &'static str) {
        self.closed.store(true, Ordering::Release);
        physical.prepared.cancel.cancel();
        drop(physical.seed.take()); // Bank's bounded unknown quarantine/pin.
        if let Ok(mut retained) = self.retained.lock() {
            if retained.is_none() {
                *retained = Some(physical);
            }
        }
        if let Ok(mut owner) = self.quarantine_owner.lock() {
            if owner.is_none() {
                *owner = Some(self.clone());
            }
        }
        if let Ok(mut event) = self.last_event.lock() {
            *event = Some(stage);
        }
    }
    #[cfg(feature = "experimental-io-binding")]
    pub fn enable_followup_observations(&self) -> Result<(), RoleError> {
        let mut slot = self
            .followup
            .try_lock()
            .map_err(|_| RoleError::Unavailable)?;
        if slot.is_some() {
            return Err(RoleError::InvalidOutput);
        }
        let owner_id = FOLLOWUP_BANK_OWNERS
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| RoleError::Unavailable)?;
        *slot = Some(NativeWarmBankObservation {
            owner_id,
            event_sequence: 0,
            accepted_seed_consumptions: 0,
            accepted_seed_context_checked: 0,
            rejected_seed_contexts: 0,
            fresh_value_evaluations: 0,
            value_always_fresh: true,
            closed: false,
            complete: true,
        });
        Ok(())
    }
    #[cfg(feature = "experimental-io-binding")]
    pub fn followup_observation(&self) -> Option<NativeWarmBankObservation> {
        match self.followup.try_lock() {
            Ok(slot) => slot.map(|mut value| {
                value.complete &= self.followup_complete.load(Ordering::Acquire);
                value
            }),
            Err(_) => {
                self.followup_complete.store(false, Ordering::Release);
                None
            }
        }
    }
    #[cfg(feature = "experimental-io-binding")]
    pub fn observe_join_incomplete(&self) {
        self.followup_complete.store(false, Ordering::Release);
    }
    #[cfg(feature = "experimental-io-binding")]
    pub fn observe_ready_join(
        &self,
        invocation: PrivateInvocation,
        value: bool,
        private_completed: bool,
        exact: bool,
    ) {
        let Ok(mut slot) = self.followup.try_lock() else {
            self.followup_complete.store(false, Ordering::Release);
            return;
        };
        let Some(s) = slot.as_mut() else {
            return;
        };
        let Some(sequence) = s.event_sequence.checked_add(1) else {
            s.complete = false;
            return;
        };
        s.event_sequence = sequence;
        if !exact {
            s.complete = false;
            return;
        }
        if value && !matches!(invocation, PrivateInvocation::Fresh { .. }) {
            s.value_always_fresh = false;
            s.complete = false;
        }
        if !private_completed {
            return;
        }
        let counter = if value {
            Some(&mut s.fresh_value_evaluations)
        } else if matches!(invocation, PrivateInvocation::ApproxCudaWarmV2 { .. }) {
            Some(&mut s.accepted_seed_consumptions)
        } else {
            None
        };
        if let Some(counter) = counter {
            if let Some(next) = counter.checked_add(1) {
                *counter = next;
            } else {
                s.complete = false;
            }
        }
        // Every admitted Approx payload reached bind_lease with the actual
        // model/role/question/Rules seal. The exact Ready join retains it.
        if !value && matches!(invocation, PrivateInvocation::ApproxCudaWarmV2 { .. }) {
            if let Some(next) = s.accepted_seed_context_checked.checked_add(1) {
                s.accepted_seed_context_checked = next;
            } else {
                s.complete = false;
            }
        }
    }
    pub fn observe_context_rejection(&self) {
        #[cfg(feature = "experimental-io-binding")]
        {
            let Ok(mut slot) = self.followup.try_lock() else {
                self.followup_complete.store(false, Ordering::Release);
                return;
            };
            if let Some(s) = slot.as_mut() {
                match (
                    s.event_sequence.checked_add(1),
                    s.rejected_seed_contexts.checked_add(1),
                ) {
                    (Some(sequence), Some(count)) => {
                        s.event_sequence = sequence;
                        s.rejected_seed_contexts = count;
                    }
                    _ => s.complete = false,
                }
            }
        }
    }
    /// Caller must have joined the actual worker after a known physical fence.
    /// Same-game terminal closure does not advance the seed namespace.
    #[cfg(feature = "experimental-io-binding")]
    pub fn close_after_known_fence(&self) -> Result<(), RoleError> {
        let result = self.close_same_game_bank();
        if result.is_err() {
            self.followup_complete.store(false, Ordering::Release);
        }
        result
    }
    #[cfg(feature = "experimental-io-binding")]
    fn close_same_game_bank(&self) -> Result<(), RoleError> {
        self.closed.store(true, Ordering::Release);
        if self.is_quarantined()
            || self
                .retained
                .try_lock()
                .map_err(|_| RoleError::Unavailable)?
                .is_some()
            || self
                .pending
                .try_lock()
                .map_err(|_| RoleError::Unavailable)?
                .is_some()
        {
            return Err(RoleError::PhysicalCompletionUnknown);
        }
        self.bank.close_after_known_fence().map_err(private_error)?;
        for role in self
            .frozen
            .try_lock()
            .map_err(|_| RoleError::Unavailable)?
            .iter_mut()
        {
            role.clear();
        }
        let mut slot = self
            .followup
            .try_lock()
            .map_err(|_| RoleError::Unavailable)?;
        if let Some(s) = slot.as_mut() {
            if let Some(next) = s.event_sequence.checked_add(1) {
                s.event_sequence = next;
                s.closed = true;
            } else {
                s.complete = false;
                return Err(RoleError::Unavailable);
            }
        }
        Ok(())
    }
    pub fn reset_after_known_fence(&self, game: u64) -> Result<(), RoleError> {
        // Production calls this only after its actual worker NewGame Ready.
        // A logical final-control rejection with a successfully revoked seed
        // is recoverable; an unknown physical owner/pin remains quarantined.
        if self.is_quarantined() {
            return Err(RoleError::PhysicalCompletionUnknown);
        }
        self.reject_output(None);
        self.bank.reset_game(game).map_err(private_error)?;
        let mut frozen = self.frozen.lock().map_err(|_| RoleError::Unavailable)?;
        for role in &mut *frozen {
            role.clear();
        }
        self.closed.store(false, Ordering::Release);
        Ok(())
    }
    pub fn snapshot(&self) -> Result<NativePrivateWarmObservation, RoleError> {
        let snapshot = self.bank.snapshot().map_err(private_error)?;
        let frozen = self.frozen.lock().map_err(|_| RoleError::Unavailable)?;
        let retained = self.retained.lock().map_err(|_| RoleError::Unavailable)?;
        Ok(NativePrivateWarmObservation {
            game_generation: snapshot.game_generation,
            entries_per_role: snapshot.entries_per_role,
            reserved_bank_bytes: snapshot.reserved_bank_bytes,
            provisional_bytes: snapshot.provisional_bytes,
            pinned_entries: snapshot.pinned_entries,
            active_seed_bytes: snapshot.active_seed_bytes,
            active_lease: snapshot.active_lease,
            admission_closed: snapshot.admission_closed || self.is_closed(),
            quarantined: snapshot.quarantined || self.is_quarantined(),
            fresh_seed_admissions: snapshot.counts.fresh_admissions,
            warm_seed_admissions: snapshot.counts.warm_admissions,
            accepted_seeds: snapshot.counts.accepted_seeds,
            revoked_seeds: snapshot.revoked_seeds,
            last_revocation: snapshot
                .last_revocation
                .map(|r| NativePrivateWarmRevocation {
                    role: r.provenance.role,
                    source_lease_id: r.provenance.source_lease_id,
                    seed_sequence: r.provenance.seed_sequence,
                    provenance_seal: r.provenance.seal,
                    reason: format!("{:?}", r.reason),
                    revoked: r.revoked,
                    removed_bank_bytes: r.removed_bank_bytes,
                    remaining_entries_per_role: r.remaining_entries_per_role,
                    prior_seed_restored: false,
                }),
            last_revocation_failure: self
                .last_revocation_failure
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .map(|e| e.to_string()),
            known_seed_completions: snapshot.counts.known_completions,
            frozen_contexts_per_role: [frozen[0].len(), frozen[1].len()],
            query_freezes: self.query_freezes.load(Ordering::Acquire),
            query_hits: self.query_hits.load(Ordering::Acquire),
            pending_acceptance: self
                .pending
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .is_some(),
            retained_prepared_input: retained.is_some(),
            retained_full_payload_bytes: retained.as_ref().map_or(0, |r| {
                let bytes = r
                    .backup_payload
                    .as_ref()
                    .map_or(0, PalsWarmInput::owned_host_bytes);
                #[cfg(feature = "experimental-io-binding")]
                let bytes = bytes.saturating_add(
                    r.backup_cuda_payload
                        .as_ref()
                        .map_or(0, PalsCudaWarmInput::owned_host_bytes),
                );
                bytes
            }),
            first_query_input_sha256_per_role: std::array::from_fn(|i| {
                frozen[i].iter().map(|f| f.first_input).collect()
            }),
            last_event: *self.last_event.lock().map_err(|_| RoleError::Unavailable)?,
            last_bank_failure: snapshot.last_failure.map(|e| e.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::ProcessEpoch;
    use rz_eval::pals_private::PrivatePrecision;
    use rz_position::Position;
    use rz_search::pals::store::{LineId, SituationId, StateId};
    use std::time::Duration;

    // Real Rules/encoder inputs; the finite latent below is a Rust ownership
    // fixture, not a loaded native capability or a learned model.
    fn state() -> Arc<NativeWarmState> {
        NativeWarmState::new(
            PrivateModelIdentity {
                model: [1; 32],
                encoding: [2; 32],
                precision: PrivatePrecision::Fp32,
                model_epoch: [3; 32],
                frozen_epoch: 1,
            },
            0,
        )
        .unwrap()
    }
    fn context(role: PalsRole) -> RoleLogicalContext {
        RoleLogicalContext {
            game_generation: 0,
            search_generation: 2,
            situation: SituationId {
                slot: 0,
                generation: 2,
            },
            state: StateId(0),
            focus: LineId(0),
            purpose: if role == PalsRole::Proposer {
                RoleQueryPurpose::ProposePolicy
            } else {
                RoleQueryPurpose::ReplyPolicy
            },
            prefix: vec![],
            focus_sha256: [4; 32],
            prefix_sha256: [5; 32],
            proposal_sha256: [6; 32],
            refutation_sha256: None,
            divergence_sha256: [7; 32],
            public_revision: 0,
            situation_revision: 0,
        }
    }
    fn prepared(
        core: &NativeWarmState,
        position: &Position,
        context: &RoleLogicalContext,
        id: u64,
        cancel: &AtomicBool,
        until: Instant,
    ) -> PreparedWarm {
        let legal = position.legal_moves();
        let query = rz_search::pals::engine::RoleQuery {
            position,
            legal: &legal,
            prefix: &context.prefix,
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: context.public_revision,
            deadline: until,
            cancel,
        };
        let kind = match context.purpose {
            RoleQueryPurpose::ReplyPolicy => super::super::NativeQueryKind::Reply,
            RoleQueryPurpose::RepairPolicy => super::super::NativeQueryKind::Repair,
            _ => super::super::NativeQueryKind::Propose,
        };
        let input =
            super::super::prepare_role_input_for_config(&query, kind, [3; 32], &core.model_config)
                .unwrap();
        core.prepare(
            RequestId::new(ProcessEpoch(1), id),
            input,
            position.snapshot(),
            context,
            until,
            CancelToken::new(),
            cancel,
        )
        .unwrap()
    }
    fn accept(
        core: &NativeWarmState,
        position: &Position,
        context: &RoleLogicalContext,
        id: u64,
        cancel: &AtomicBool,
        until: Instant,
    ) -> Result<(), RoleError> {
        core.accept(
            RequestId::new(ProcessEpoch(1), id),
            &RoleAcceptance {
                snapshot: &position.snapshot(),
                context,
                deadline: until,
                cancel,
            },
        )
    }
    fn latent() -> Vec<f32> {
        (0..6144).map(|i| i as f32 * 0.000125 + 0.25).collect()
    }

    #[cfg(feature = "experimental-io-binding")]
    fn cuda_state() -> Arc<NativeWarmState> {
        let legacy = state();
        NativeWarmState::new_cuda(legacy.model, 0, PalsModelConfig::full_line_interaction_v2())
            .unwrap()
    }
    #[cfg(feature = "experimental-io-binding")]
    #[test]
    fn cuda_observation_contention_is_sticky_without_closing_model_admission() {
        for ready in [false, true] {
            let core = cuda_state();
            core.enable_followup_observations().unwrap();
            let guard = core.followup.lock().unwrap();
            if ready {
                core.observe_ready_join(
                    PrivateInvocation::Fresh { input_key: [8; 32] },
                    true,
                    true,
                    true,
                );
            } else {
                core.observe_context_rejection();
            }
            drop(guard);
            let observation = core.followup_observation().unwrap();
            assert!(!observation.complete);
            assert_eq!(observation.event_sequence, 0);
            assert_eq!(observation.fresh_value_evaluations, 0);
            assert_eq!(observation.rejected_seed_contexts, 0);
            assert!(!core.is_closed());
            core.observe_ready_join(
                PrivateInvocation::Fresh { input_key: [8; 32] },
                true,
                true,
                true,
            );
            assert!(!core.followup_observation().unwrap().complete);
        }
    }
    #[cfg(feature = "experimental-io-binding")]
    #[test]
    fn cuda_same_game_close_clears_only_fenced_accepted_seeds() {
        let core = cuda_state();
        core.enable_followup_observations().unwrap();
        let position = Position::startpos();
        let control = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        let logical = context(PalsRole::Proposer);
        let physical = core
            .begin(prepared(&core, &position, &logical, 1, &control, until))
            .unwrap();
        core.completed(physical, Some(&latent())).unwrap();
        accept(&core, &position, &logical, 1, &control, until).unwrap();
        let before = core.snapshot().unwrap();
        assert_eq!(before.entries_per_role[0], 1);
        core.close_after_known_fence().unwrap();
        let after = core.snapshot().unwrap();
        assert_eq!(after.game_generation, before.game_generation);
        assert_eq!(after.accepted_seeds, before.accepted_seeds);
        assert_eq!(after.known_seed_completions, before.known_seed_completions);
        assert_eq!(after.entries_per_role, [0; 3]);
        assert_eq!(after.pinned_entries, 0);
        assert!(!after.pending_acceptance && !after.retained_prepared_input);
        let observation = core.followup_observation().unwrap();
        assert!(observation.closed && observation.complete && after.admission_closed);

        let pending = cuda_state();
        pending.enable_followup_observations().unwrap();
        let physical = pending
            .begin(prepared(&pending, &position, &logical, 2, &control, until))
            .unwrap();
        pending.completed(physical, Some(&latent())).unwrap();
        let before = pending.snapshot().unwrap();
        assert!(matches!(
            pending.close_after_known_fence(),
            Err(RoleError::PhysicalCompletionUnknown)
        ));
        let after = pending.snapshot().unwrap();
        assert!(after.pending_acceptance && after.admission_closed);
        assert_eq!(after.provisional_bytes, before.provisional_bytes);
        assert_eq!(after.game_generation, before.game_generation);
        assert!(!pending.followup_observation().unwrap().complete);
    }
    #[cfg(feature = "experimental-io-binding")]
    #[test]
    fn cuda_current_query_reuses_only_accepted_seed_and_value_is_fresh() {
        let core = cuda_state();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        let mut context = context(PalsRole::Proposer);
        let first = prepared(&core, &position, &context, 1, &cancel, until);
        let query_bits = first.input.query.map(f32::to_bits);
        let physical = core.begin(first).unwrap();
        assert!(matches!(
            physical.invocation().unwrap(),
            PrivateInvocation::Fresh { .. }
        ));
        core.completed(physical, Some(&latent())).unwrap();
        assert_eq!(core.snapshot().unwrap().accepted_seeds, 0);
        accept(&core, &position, &context, 1, &cancel, until).unwrap();
        context.public_revision = 1;
        let second = prepared(&core, &position, &context, 2, &cancel, until);
        assert_eq!(second.input.query.map(f32::to_bits), query_bits);
        let physical = core.begin(second).unwrap();
        assert!(matches!(
            physical.invocation().unwrap(),
            PrivateInvocation::ApproxCudaWarmV2 { .. }
        ));
        core.completed(physical, Some(&latent())).unwrap();
        assert_eq!(core.snapshot().unwrap().accepted_seeds, 1);
        let mut stale = context.clone();
        stale.search_generation += 1;
        assert!(accept(&core, &position, &stale, 2, &cancel, until).is_err());
        core.reject_output(None);
        let observed = core.snapshot().unwrap();
        assert_eq!(observed.accepted_seeds, 1);
        assert_eq!(observed.query_freezes, 0);
        assert_eq!(observed.frozen_contexts_per_role, [0, 0]);
        context.purpose = RoleQueryPurpose::ValueFresh;
        let value = prepared(&core, &position, &context, 3, &cancel, until);
        let physical = core.begin(value).unwrap();
        assert!(physical.seed().is_none());
        assert!(matches!(
            physical.invocation().unwrap(),
            PrivateInvocation::Fresh { .. }
        ));
        core.completed(physical, Some(&latent())).unwrap();
        accept(&core, &position, &context, 3, &cancel, until).unwrap();
        assert_eq!(core.snapshot().unwrap().accepted_seeds, 1);
    }
    #[cfg(feature = "experimental-io-binding")]
    #[test]
    fn cuda_unknown_retains_exact_prepared_input_and_bank_without_new_fence() {
        let core = cuda_state();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        let context = context(PalsRole::Proposer);
        let first = prepared(&core, &position, &context, 1, &cancel, until);
        let input = Arc::downgrade(&first.input);
        let physical = core.begin(first).unwrap();
        core.retain_unknown(physical, "cuda_fixture_completion_unknown");
        let observed = core.snapshot().unwrap();
        assert!(
            observed.quarantined && observed.admission_closed && observed.retained_prepared_input
        );
        assert!(input.upgrade().is_some());
        assert!(observed.active_lease.is_some());
        assert_eq!(observed.accepted_seeds, 0);
        assert_eq!(observed.known_seed_completions, 0);
        assert!(matches!(
            core.reset_after_known_fence(1),
            Err(RoleError::PhysicalCompletionUnknown)
        ));
    }

    #[test]
    fn accepted_fresh_then_actual_warm_keeps_full_frozen_query_and_current_records() {
        let core = state();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        let mut context = context(PalsRole::Proposer);
        let mut first = prepared(&core, &position, &context, 1, &cancel, until);
        // All axes, including negative zero/subnormal, belong to the first
        // actual tensor. This is a boundary fixture for bit preservation.
        first.input = Arc::new({
            let mut input = first.input.as_ref().clone();
            input.query[0] = -0.0;
            input.query[1] = f32::from_bits(1);
            input
        });
        // Use a fresh core so its first preparation sees the changed full bits.
        let core = state();
        let first = core
            .prepare(
                first.id,
                first.input.as_ref().clone(),
                position.snapshot(),
                &context,
                until,
                CancelToken::new(),
                &cancel,
            )
            .unwrap();
        let bits = first.input.query.map(f32::to_bits);
        let fresh_key = first
            .input
            .canonical_input_key(&PalsModelConfig::baseline())
            .unwrap();
        let fresh_tensors = first
            .input
            .prepare_tensors(&PalsModelConfig::baseline())
            .unwrap();
        assert_eq!(fresh_tensors.record_mask, vec![false]);
        let physical = core.begin(first).unwrap();
        assert!(matches!(
            physical.invocation().unwrap(),
            PrivateInvocation::Fresh { .. }
        ));
        core.completed(physical, Some(&latent())).unwrap();
        assert_eq!(core.snapshot().unwrap().accepted_seeds, 0);
        accept(&core, &position, &context, 1, &cancel, until).unwrap();
        assert!(accept(&core, &position, &context, 1, &cancel, until).is_err());
        context.public_revision = 1;
        context.situation_revision = 9;
        let legal = position.legal_moves();
        let record = rz_search::pals::engine::RoleRecord {
            revision: 1,
            parent_revision: None,
            supersedes_revision: None,
            origin_state: context.state,
            kind: rz_search::pals::engine::RecordKind::Proposal,
            line: vec![legal[0]],
            value: None,
            completed_depth: 0,
            score_scope: None,
            cpu_observation: None,
            perspective: position.side_to_move(),
            critical: true,
        };
        let records = [record];
        let query = rz_search::pals::engine::RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &records,
            revision: 1,
            deadline: until,
            cancel: &cancel,
        };
        let actual = super::super::prepare_role_input(
            &query,
            super::super::NativeQueryKind::Propose,
            [3; 32],
        )
        .unwrap();
        let expected_records = actual.records.clone();
        assert_eq!(expected_records.len(), 1);
        let next = core
            .prepare(
                RequestId::new(ProcessEpoch(1), 2),
                actual,
                position.snapshot(),
                &context,
                until,
                CancelToken::new(),
                &cancel,
            )
            .unwrap();
        assert_eq!(next.input.query.map(f32::to_bits), bits);
        assert_eq!(next.input.situation_revision, 1);
        assert_eq!(next.input.records, expected_records);
        assert_eq!(next.input.records[0].revision, 1);
        assert!(next.input.records[0].critical);
        let next_tensors = next
            .input
            .prepare_tensors(&PalsModelConfig::baseline())
            .unwrap();
        assert_eq!(next_tensors.record_mask, vec![true]);
        assert_ne!(next_tensors.record_mask, fresh_tensors.record_mask);
        assert_ne!(next_tensors.records, fresh_tensors.records);
        let next_key = next
            .input
            .canonical_input_key(&PalsModelConfig::baseline())
            .unwrap();
        assert_ne!(next_key, fresh_key);
        let warm = core.begin(next).unwrap();
        assert_eq!(
            warm.seed().unwrap().seed_bits().unwrap(),
            latent().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let PrivateInvocation::ApproxWarmV1 {
            input_key,
            invocation_key,
            ..
        } = warm.invocation().unwrap()
        else {
            panic!("accepted seed must be actually consumed")
        };
        assert_eq!(input_key, next_key);
        assert_ne!(invocation_key, input_key);
        core.completed(warm, None).unwrap();
        let snapshot = core.snapshot().unwrap();
        assert_eq!(snapshot.accepted_seeds, 1);
        assert_eq!(snapshot.warm_seed_admissions, 1);
        assert_eq!(snapshot.query_hits, 1);
        assert_eq!(snapshot.known_seed_completions, 2);
        assert_eq!(snapshot.pinned_entries, 0);
        assert!(!snapshot.pending_acceptance);
    }
    #[test]
    fn value_fresh_never_admits_stages_or_commits_a_policy_seed() {
        let core = state();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        let mut context = context(PalsRole::Proposer);
        context.purpose = RoleQueryPurpose::ValueFresh;
        let physical = core
            .begin(prepared(&core, &position, &context, 1, &cancel, until))
            .unwrap();
        assert!(physical.seed().is_none());
        assert!(matches!(
            physical.invocation().unwrap(),
            PrivateInvocation::Fresh { .. }
        ));
        core.completed(physical, Some(&latent())).unwrap();
        accept(&core, &position, &context, 1, &cancel, until).unwrap();
        let snapshot = core.snapshot().unwrap();
        assert_eq!(snapshot.fresh_seed_admissions, 0);
        assert_eq!(snapshot.accepted_seeds, 0);
        assert_eq!(snapshot.known_seed_completions, 0);
        assert_eq!(snapshot.frozen_contexts_per_role, [0, 0]);
    }
    #[test]
    fn cancel_deadline_stale_and_replacement_controls_cannot_accept_seed() {
        for kind in 0..4 {
            let core = state();
            let position = Position::startpos();
            let cancel = AtomicBool::new(false);
            let replacement = AtomicBool::new(false);
            let until = Instant::now() + Duration::from_secs(30);
            let context = context(PalsRole::Proposer);
            let physical = core
                .begin(prepared(&core, &position, &context, 1, &cancel, until))
                .unwrap();
            core.completed(physical, Some(&latent())).unwrap();
            let mut wrong = context.clone();
            let result = match kind {
                0 => {
                    cancel.store(true, Ordering::Release);
                    accept(&core, &position, &context, 1, &cancel, until)
                }
                1 => accept(&core, &position, &context, 1, &cancel, Instant::now()),
                2 => {
                    wrong.public_revision += 1;
                    accept(&core, &position, &wrong, 1, &cancel, until)
                }
                _ => accept(&core, &position, &context, 1, &replacement, until),
            };
            assert!(result.is_err());
            core.reject_output(None);
            let snapshot = core.snapshot().unwrap();
            assert_eq!(snapshot.accepted_seeds, 0);
            assert_eq!(snapshot.pinned_entries, 0);
            assert!(!snapshot.pending_acceptance);
        }
    }
    #[test]
    fn roles_history_focus_generation_and_reset_are_distinct() {
        let core = state();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        let p = context(PalsRole::Proposer);
        let first = core
            .begin(prepared(&core, &position, &p, 1, &cancel, until))
            .unwrap();
        core.completed(first, Some(&latent())).unwrap();
        accept(&core, &position, &p, 1, &cancel, until).unwrap();
        let c = context(PalsRole::Critic);
        let critic = core
            .begin(prepared(&core, &position, &c, 2, &cancel, until))
            .unwrap();
        assert!(matches!(
            critic.invocation().unwrap(),
            PrivateInvocation::Fresh { .. }
        ));
        core.completed(critic, None).unwrap();
        let fen = Position::from_fen(&position.to_fen()).unwrap();
        let history = core
            .begin(prepared(&core, &fen, &p, 3, &cancel, until))
            .unwrap();
        assert!(matches!(
            history.invocation().unwrap(),
            PrivateInvocation::Fresh { .. }
        ));
        core.completed(history, None).unwrap();
        let mut changed = p.clone();
        changed.search_generation += 1;
        let generation = core
            .begin(prepared(&core, &position, &changed, 4, &cancel, until))
            .unwrap();
        assert!(matches!(
            generation.invocation().unwrap(),
            PrivateInvocation::Fresh { .. }
        ));
        core.completed(generation, None).unwrap();
        core.reset_after_known_fence(1).unwrap();
        assert_eq!(core.snapshot().unwrap().entries_per_role, [0, 0, 0]);
        assert_eq!(core.snapshot().unwrap().frozen_contexts_per_role, [0, 0]);
        let stale = prepared(&core, &position, &p, 5, &cancel, until);
        assert!(core.begin(stale).is_err());
    }
    #[test]
    fn frozen_context_pin_and_unknown_prelaunch_owner_survive_drop_and_refuse_reset() {
        let core = state();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        let context = context(PalsRole::Proposer);
        let first = core
            .begin(prepared(&core, &position, &context, 1, &cancel, until))
            .unwrap();
        core.completed(first, Some(&latent())).unwrap();
        accept(&core, &position, &context, 1, &cancel, until).unwrap();
        let physical = core
            .begin(prepared(&core, &position, &context, 2, &cancel, until))
            .unwrap();
        let input = Arc::downgrade(&physical.prepared.input);
        let owner = Arc::downgrade(&core);
        core.retain_unknown(physical, "test_submit_failure_no_physical_fence");
        let snapshot = core.snapshot().unwrap();
        assert!(snapshot.quarantined && snapshot.retained_prepared_input);
        assert_eq!(snapshot.known_seed_completions, 1);
        assert_eq!(snapshot.pinned_entries, 1);
        assert!(core.reset_after_known_fence(1).is_err());
        assert!(
            core.prepare(
                RequestId::new(ProcessEpoch(1), 3),
                prepared(&state(), &position, &context, 3, &cancel, until)
                    .input
                    .as_ref()
                    .clone(),
                position.snapshot(),
                &context,
                until,
                CancelToken::new(),
                &cancel
            )
            .is_err()
        );
        drop(core);
        assert!(owner.upgrade().is_some());
        assert!(input.upgrade().is_some());
        // Deliberate bounded quarantine: no fake Ready is invented to release it.
    }
    #[test]
    fn pinned_frozen_capacity_failure_preserves_existing_contexts() {
        let core = state();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        let context = context(PalsRole::Proposer);
        let held = prepared(&core, &position, &context, 1, &cancel, until);
        let mut second = context.clone();
        second.focus_sha256 = [8; 32];
        let _second = prepared(&core, &position, &second, 2, &cancel, until);
        let mut third = context.clone();
        third.focus_sha256 = [9; 32];
        let legal = position.legal_moves();
        let query = rz_search::pals::engine::RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 0,
            deadline: until,
            cancel: &cancel,
        };
        let input = super::super::prepare_role_input(
            &query,
            super::super::NativeQueryKind::Propose,
            [3; 32],
        )
        .unwrap();
        assert!(
            core.prepare(
                RequestId::new(ProcessEpoch(1), 3),
                input,
                position.snapshot(),
                &third,
                until,
                CancelToken::new(),
                &cancel
            )
            .is_err()
        );
        assert_eq!(core.snapshot().unwrap().frozen_contexts_per_role, [2, 0]);
        assert!(held.frozen.is_some());
    }
    #[test]
    fn final_control_change_revokes_exact_new_seed_without_faking_completion_or_reset() {
        for cancelled in [true, false] {
            let core = state();
            let position = Position::startpos();
            let cancel = AtomicBool::new(false);
            let until = Instant::now() + Duration::from_secs(30);
            let context = context(PalsRole::Proposer);
            let first = core
                .begin(prepared(&core, &position, &context, 1, &cancel, until))
                .unwrap();
            core.completed(first, Some(&latent())).unwrap();
            accept(&core, &position, &context, 1, &cancel, until).unwrap();
            let second = core
                .begin(prepared(&core, &position, &context, 2, &cancel, until))
                .unwrap();
            assert!(matches!(
                second.invocation().unwrap(),
                PrivateInvocation::ApproxWarmV1 { .. }
            ));
            core.completed(second, Some(&latent())).unwrap();
            let snapshot = position.snapshot();
            let acceptance = RoleAcceptance {
                snapshot: &snapshot,
                context: &context,
                deadline: until,
                cancel: &cancel,
            };
            let result = core.accept_with_final_guard(
                RequestId::new(ProcessEpoch(1), 2),
                &acceptance,
                || {
                    if cancelled {
                        cancel.store(true, Ordering::Release);
                        acceptance.check_control()
                    } else {
                        Err(RoleError::Deadline)
                    }
                },
            );
            assert!(matches!(
                result,
                Err(RoleError::Canceled | RoleError::Deadline)
            ));
            let observation = core.snapshot().unwrap();
            assert_eq!(observation.accepted_seeds, 2);
            assert_eq!(observation.revoked_seeds, 1);
            assert_eq!(observation.known_seed_completions, 2);
            assert_eq!(observation.game_generation, 0);
            assert_eq!(observation.entries_per_role, [0, 0, 0]);
            assert_eq!(observation.pinned_entries, 0);
            assert!(observation.admission_closed);
            assert!(!observation.quarantined);
            let revoke = observation.last_revocation.unwrap();
            assert!(revoke.revoked && !revoke.prior_seed_restored);
            assert_eq!(revoke.seed_sequence, 2);
            assert!(revoke.removed_bank_bytes > 0);
            assert!(observation.last_revocation_failure.is_none());
            assert_eq!(
                observation.last_event,
                Some("control_changed_exact_commit_revoked_prior_seed_not_restored")
            );
            // Known completion plus the caller's new-game fence permits a new
            // logical game. This Rust owner fixture does not attest a native
            // worker response; production supplies that fence before this call.
            core.reset_after_known_fence(1).unwrap();
            let reset = core.snapshot().unwrap();
            assert!(!reset.admission_closed && !reset.quarantined);
            assert_eq!(reset.game_generation, 1);
            assert_eq!(reset.frozen_contexts_per_role, [0, 0]);
            let next_cancel = AtomicBool::new(false);
            let mut next_context = context.clone();
            next_context.game_generation = 1;
            next_context.search_generation += 1;
            next_context.situation.generation += 1;
            let fresh = core
                .begin(prepared(
                    &core,
                    &position,
                    &next_context,
                    3,
                    &next_cancel,
                    until,
                ))
                .unwrap();
            assert!(matches!(
                fresh.invocation().unwrap(),
                PrivateInvocation::Fresh { .. }
            ));
            core.completed(fresh, None).unwrap();
            let fresh_result = core.snapshot().unwrap();
            assert_eq!(fresh_result.known_seed_completions, 3);
            assert_eq!(fresh_result.accepted_seeds, 2);
            assert_eq!(fresh_result.revoked_seeds, 1);
        }
    }
    #[test]
    fn precommit_cancel_keeps_prior_accepted_seed_and_releases_provisional_pin() {
        let core = state();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(30);
        let context = context(PalsRole::Proposer);
        let first = core
            .begin(prepared(&core, &position, &context, 1, &cancel, until))
            .unwrap();
        core.completed(first, Some(&latent())).unwrap();
        accept(&core, &position, &context, 1, &cancel, until).unwrap();
        let second = core
            .begin(prepared(&core, &position, &context, 2, &cancel, until))
            .unwrap();
        core.completed(second, Some(&latent())).unwrap();
        cancel.store(true, Ordering::Release);
        assert!(matches!(
            accept(&core, &position, &context, 2, &cancel, until),
            Err(RoleError::Canceled)
        ));
        core.reject_output(Some(RequestId::new(ProcessEpoch(1), 2)));
        let observation = core.snapshot().unwrap();
        assert_eq!(observation.accepted_seeds, 1);
        assert_eq!(observation.revoked_seeds, 0);
        assert_eq!(observation.entries_per_role, [1, 0, 0]);
        assert_eq!(observation.pinned_entries, 0);
        assert!(!observation.pending_acceptance);
    }
}
