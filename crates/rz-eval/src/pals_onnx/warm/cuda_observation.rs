//! Bounded metadata for the real CUDA Warm owner. No tensor, session, worker,
//! seed bank, allocator or native lifetime is owned by an observation handle.
//! Successful Tensor creation and its actual Drop determine payload bytes;
//! allocator caches, internal staging/workspace and VRAM remain unobserved.
use super::{PalsCudaWarmCapability, PrivateInvocation};
use crate::error::{BackendError, FailureKind as K, FailureStage as S};
use crate::pals_model::PalsRole;
use crate::pals_onnx::{fail, PalsNNInputEvidence};
use rz_contracts::{ExecutionId, RequestId};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};

pub const PRIVATE_CUDA_WARM_MEASUREMENT_CONTRACT: &str = "rz-pals-cuda-warm-owner-accounting-v4/1";
static BACKEND_OWNERS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsCudaWarmRuntimeLimits {
    pub max_leases: u32,
    /// Explicit CUDA K/V payload only. This is distinct from the runtime's
    /// declared memory reservation and the session arena/whole-device VRAM.
    pub device_bytes_max: u64,
}
impl PalsCudaWarmRuntimeLimits {
    pub fn validate(self) -> Result<Self, BackendError> {
        if self.max_leases != 1 || self.device_bytes_max == 0 {
            return Err(fail(K::InvalidInput, S::Admission,
                "CUDA Warm observations require exactly one physical lease and a positive device payload limit"));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PalsCudaWarmInvocationPurpose {
    Startup,
    Policy,
    ValueFresh,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsCudaWarmExecutionBinding {
    pub backend_owner_id: u64,
    pub purpose: PalsCudaWarmInvocationPurpose,
    /// No invented runtime ID for the two startup control invocations.
    pub request: Option<RequestId>,
    pub execution: Option<ExecutionId>,
    pub bank_lease_id: Option<u64>,
}
impl PalsCudaWarmExecutionBinding {
    pub(super) fn validate(
        &self,
        owner: u64,
        role: PalsRole,
        invocation: PrivateInvocation,
    ) -> Result<(), BackendError> {
        let fresh = matches!(invocation, PrivateInvocation::Fresh { .. });
        let actual_ids = self
            .request
            .zip(self.execution)
            .is_some_and(|(r, e)| r.epoch == e.epoch && r.sequence != 0 && e.sequence != 0);
        let valid = match self.purpose {
            PalsCudaWarmInvocationPurpose::Startup => {
                self.request.is_none()
                    && self.execution.is_none()
                    && self.bank_lease_id.is_none()
                    && fresh
            }
            PalsCudaWarmInvocationPurpose::Policy => {
                actual_ids && self.bank_lease_id.is_some_and(|id| id != 0)
            }
            PalsCudaWarmInvocationPurpose::ValueFresh => {
                actual_ids && self.bank_lease_id.is_none() && fresh && role == PalsRole::Proposer
            }
        };
        if owner == 0
            || self.backend_owner_id != owner
            || !valid
            || role == PalsRole::Validator
            || matches!(invocation, PrivateInvocation::ApproxWarmV1 { .. })
        {
            return Err(fail(K::IdentityMismatch, S::Admission,
                "CUDA Warm observation binding differs from the actual owner, invocation or Fresh value route"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PalsCudaWarmInvocationState {
    Active,
    CompletedKnown,
    Quarantined,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsCudaWarmInvocationObservation {
    pub ordinal: u64,
    pub binding: PalsCudaWarmExecutionBinding,
    pub role: PalsRole,
    pub input_key: [u8; 32],
    pub invocation: PrivateInvocation,
    pub state: PalsCudaWarmInvocationState,
    pub private_run_completed: bool,
    pub actual_completed_nn_inputs_delta: Option<u64>,
    pub nn_evidence_complete: bool,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PalsCudaWarmOwnerObservation {
    pub backend_owner_id: u64,
    pub event_sequence: u64,
    pub limits: PalsCudaWarmRuntimeLimits,
    pub leases_admitted: u64,
    pub leases_physically_completed: u64,
    pub leases_quarantined: u64,
    pub leases_active: u32,
    pub leases_peak: u32,
    pub startup_leases_admitted: u64,
    pub startup_leases_physically_completed: u64,
    pub startup_leases_quarantined: u64,
    pub device_bytes_live: u64,
    pub device_bytes_peak: u64,
    pub device_bytes_retained: u64,
    /// Prospective charge while Tensor allocation is in progress, never an
    /// observed allocated byte count or a device peak.
    pub pending_allocation_bytes: u64,
    pub device_tensors_created: u64,
    pub device_tensor_handles_released: u64,
    pub device_payload_limit_rejections: u64,
    pub approx_private_runs_completed: u64,
    pub fresh_value_private_runs_completed: u64,
    pub admission_closed: bool,
    pub backend_dropped: bool,
    /// Owned native values were actually dropped after a known fence. This
    /// does not claim cudaFree, ORT arena release, a bank close, or worker join.
    pub buffers_released: bool,
    pub accounting_complete: bool,
    pub last_invocation: Option<PalsCudaWarmInvocationObservation>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PalsCudaWarmObservationStatus {
    Available,
    Contended,
    Poisoned,
    EventSequenceExhausted,
    Incomplete,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PalsCudaWarmObservationSnapshot {
    /// Available describes metadata delivery, never CUDA capability support.
    pub status: PalsCudaWarmObservationStatus,
    pub attempted_event_sequence: u64,
    /// On missed/poisoned events this is the last earlier snapshot. Its own
    /// sequence must not be relabeled as the attempted current event.
    pub latest: Option<PalsCudaWarmOwnerObservation>,
}
struct Ledger {
    owner_id: u64,
    limits: PalsCudaWarmRuntimeLimits,
    capability: PalsCudaWarmCapability,
    latest: Mutex<PalsCudaWarmOwnerObservation>,
    event_sequence: AtomicU64,
    invocation_sequence: AtomicU64,
    active_leases: AtomicU32,
    charged_device_bytes: AtomicU64,
    owned_device_bytes: AtomicU64,
    peak_device_bytes: AtomicU64,
    closed: AtomicBool,
    quarantined: AtomicBool,
    incomplete: AtomicBool,
    contended: AtomicBool,
    poisoned: AtomicBool,
    exhausted: AtomicBool,
}
/// This Arc owns fixed-size metadata only. It remains readable after the
/// physical backend is dropped and never extends any native buffer lifetime.
#[derive(Clone)]
pub struct PalsCudaWarmObservationHandle(Arc<Ledger>);
impl PalsCudaWarmObservationHandle {
    pub(super) fn new(
        capability: PalsCudaWarmCapability,
        limits: PalsCudaWarmRuntimeLimits,
    ) -> Result<Self, BackendError> {
        let limits = limits.validate()?;
        let owner_id = BACKEND_OWNERS
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |id| id.checked_add(1))
            .map_err(|_| {
                fail(
                    K::ResourceExhausted,
                    S::Admission,
                    "CUDA Warm observation owner identity exhausted",
                )
            })?
            + 1;
        Ok(Self(Arc::new(Ledger {
            owner_id,
            limits,
            capability,
            latest: Mutex::new(PalsCudaWarmOwnerObservation {
                backend_owner_id: owner_id,
                event_sequence: 0,
                limits,
                leases_admitted: 0,
                leases_physically_completed: 0,
                leases_quarantined: 0,
                leases_active: 0,
                leases_peak: 0,
                startup_leases_admitted: 0,
                startup_leases_physically_completed: 0,
                startup_leases_quarantined: 0,
                device_bytes_live: 0,
                device_bytes_peak: 0,
                device_bytes_retained: 0,
                pending_allocation_bytes: 0,
                device_tensors_created: 0,
                device_tensor_handles_released: 0,
                device_payload_limit_rejections: 0,
                approx_private_runs_completed: 0,
                fresh_value_private_runs_completed: 0,
                admission_closed: false,
                backend_dropped: false,
                buffers_released: false,
                accounting_complete: true,
                last_invocation: None,
            }),
            event_sequence: AtomicU64::new(0),
            invocation_sequence: AtomicU64::new(0),
            active_leases: AtomicU32::new(0),
            charged_device_bytes: AtomicU64::new(0),
            owned_device_bytes: AtomicU64::new(0),
            peak_device_bytes: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            quarantined: AtomicBool::new(false),
            incomplete: AtomicBool::new(false),
            contended: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            exhausted: AtomicBool::new(false),
        })))
    }
    pub fn owner_id(&self) -> u64 {
        self.0.owner_id
    }
    pub fn limits(&self) -> PalsCudaWarmRuntimeLimits {
        self.0.limits
    }
    /// A loaded-session identity only. Native must join actual startup origin,
    /// placement and physical observations before reporting available=true.
    pub fn capability(&self) -> PalsCudaWarmCapability {
        self.0.capability
    }
    pub fn snapshot(&self) -> PalsCudaWarmObservationSnapshot {
        let attempted_event_sequence = self.0.event_sequence.load(Ordering::Acquire);
        let latest = match self.0.latest.try_lock() {
            Ok(latest) => Some(latest.clone()),
            Err(TryLockError::WouldBlock) => {
                self.0.contended.store(true, Ordering::Release);
                None
            }
            Err(TryLockError::Poisoned(error)) => {
                self.0.poisoned.store(true, Ordering::Release);
                Some(error.into_inner().clone())
            }
        };
        let status = if self.0.exhausted.load(Ordering::Acquire) {
            PalsCudaWarmObservationStatus::EventSequenceExhausted
        } else if self.0.poisoned.load(Ordering::Acquire) {
            PalsCudaWarmObservationStatus::Poisoned
        } else if self.0.contended.load(Ordering::Acquire) {
            PalsCudaWarmObservationStatus::Contended
        } else if self.0.incomplete.load(Ordering::Acquire)
            || latest.as_ref().is_some_and(|s| !s.accounting_complete)
        {
            PalsCudaWarmObservationStatus::Incomplete
        } else if latest
            .as_ref()
            .is_some_and(|s| s.event_sequence != attempted_event_sequence)
        {
            PalsCudaWarmObservationStatus::Contended
        } else {
            PalsCudaWarmObservationStatus::Available
        };
        PalsCudaWarmObservationSnapshot {
            status,
            attempted_event_sequence,
            latest,
        }
    }
    fn incomplete(&self) {
        self.0.incomplete.store(true, Ordering::Release);
    }
    fn update(&self, change: impl FnOnce(&mut PalsCudaWarmOwnerObservation) -> Option<()>) {
        let sequence = match self.0.event_sequence.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |sequence| sequence.checked_add(1),
        ) {
            Ok(sequence) => sequence + 1,
            Err(_) => {
                self.0.exhausted.store(true, Ordering::Release);
                return;
            }
        };
        let mut latest = match self.0.latest.try_lock() {
            Ok(latest) => latest,
            Err(TryLockError::WouldBlock) => {
                self.0.contended.store(true, Ordering::Release);
                return;
            }
            Err(TryLockError::Poisoned(_)) => {
                self.0.poisoned.store(true, Ordering::Release);
                return;
            }
        };
        let mut next = latest.clone();
        if change(&mut next).is_none() {
            self.incomplete();
            return;
        }
        next.event_sequence = sequence;
        next.leases_active = self.0.active_leases.load(Ordering::Acquire);
        next.device_bytes_live = self.0.owned_device_bytes.load(Ordering::Acquire);
        next.device_bytes_peak = self.0.peak_device_bytes.load(Ordering::Acquire);
        let charged = self.0.charged_device_bytes.load(Ordering::Acquire);
        let Some(pending) = charged.checked_sub(next.device_bytes_live) else {
            self.incomplete();
            return;
        };
        next.pending_allocation_bytes = pending;
        next.device_bytes_retained = if self.0.quarantined.load(Ordering::Acquire) {
            next.device_bytes_live
        } else {
            0
        };
        next.admission_closed = self.0.closed.load(Ordering::Acquire);
        next.accounting_complete &= !self.0.incomplete.load(Ordering::Acquire)
            && !self.0.contended.load(Ordering::Acquire)
            && !self.0.poisoned.load(Ordering::Acquire);
        *latest = next;
    }
    pub(super) fn begin(
        &self,
        binding: PalsCudaWarmExecutionBinding,
        role: PalsRole,
        input_key: [u8; 32],
        invocation: PrivateInvocation,
    ) -> Result<CudaWarmInvocationGuard, BackendError> {
        binding.validate(self.owner_id(), role, invocation)?;
        if self.0.closed.load(Ordering::Acquire) {
            return Err(fail(
                K::BackendFailure,
                S::Admission,
                "CUDA Warm observation physical owner is closed",
            ));
        }
        self.0
            .active_leases
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                fail(
                    K::ResourceExhausted,
                    S::Admission,
                    "CUDA Warm one-lease physical limit reached",
                )
            })?;
        let ordinal = match self.0.invocation_sequence.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |id| id.checked_add(1),
        ) {
            Ok(id) => id + 1,
            Err(_) => {
                self.0.active_leases.store(0, Ordering::Release);
                self.0.exhausted.store(true, Ordering::Release);
                return Err(fail(
                    K::ResourceExhausted,
                    S::Admission,
                    "CUDA Warm physical invocation sequence exhausted",
                ));
            }
        };
        self.update(|next| {
            next.leases_admitted = next.leases_admitted.checked_add(1)?;
            next.leases_peak = next.leases_peak.max(1);
            if binding.purpose == PalsCudaWarmInvocationPurpose::Startup {
                next.startup_leases_admitted = next.startup_leases_admitted.checked_add(1)?;
            }
            next.last_invocation = Some(PalsCudaWarmInvocationObservation {
                ordinal,
                binding,
                role,
                input_key,
                invocation,
                state: PalsCudaWarmInvocationState::Active,
                private_run_completed: false,
                actual_completed_nn_inputs_delta: None,
                nn_evidence_complete: false,
            });
            Some(())
        });
        Ok(CudaWarmInvocationGuard {
            handle: self.clone(),
            ordinal,
            binding,
            finished: false,
        })
    }
    /// Called only immediately after actual private Run + synchronize_outputs.
    pub(super) fn private_fence_completed(&self) {
        self.update(|next| {
            let invocation = next.last_invocation.as_mut()?;
            if invocation.state != PalsCudaWarmInvocationState::Active
                || invocation.private_run_completed
            {
                return None;
            }
            invocation.private_run_completed = true;
            if matches!(
                invocation.invocation,
                PrivateInvocation::ApproxCudaWarmV2 { .. }
            ) {
                next.approx_private_runs_completed =
                    next.approx_private_runs_completed.checked_add(1)?;
            }
            if invocation.binding.purpose == PalsCudaWarmInvocationPurpose::ValueFresh {
                next.fresh_value_private_runs_completed =
                    next.fresh_value_private_runs_completed.checked_add(1)?;
            }
            Some(())
        });
    }
    pub(in crate::pals_onnx) fn reserve_device_payload(
        &self,
        bytes: u64,
    ) -> Result<CudaWarmDeviceAllocationGuard, BackendError> {
        if bytes == 0 || self.0.closed.load(Ordering::Acquire) {
            return Err(fail(
                K::InvalidInput,
                S::Admission,
                "CUDA Warm device payload reservation is empty or closed",
            ));
        }
        if self
            .0
            .charged_device_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                live.checked_add(bytes)
                    .filter(|sum| *sum <= self.limits().device_bytes_max)
            })
            .is_err()
        {
            self.update(|next| {
                next.device_payload_limit_rejections =
                    next.device_payload_limit_rejections.checked_add(1)?;
                Some(())
            });
            return Err(fail(
                K::ResourceExhausted,
                S::Admission,
                "CUDA Warm explicit device payload limit reached before allocation",
            ));
        }
        self.update(|_| Some(()));
        Ok(CudaWarmDeviceAllocationGuard {
            handle: self.clone(),
            reserved_bytes: bytes,
            allocated_bytes: 0,
            allocated_tensors: 0,
        })
    }
    pub(in crate::pals_onnx) fn has_unfinished_invocation(&self) -> bool {
        self.0.active_leases.load(Ordering::Acquire) != 0
            || self.0.quarantined.load(Ordering::Acquire)
    }
    pub(in crate::pals_onnx) fn backend_dropped_known(&self) {
        self.0.closed.store(true, Ordering::Release);
        let released = !self.has_unfinished_invocation()
            && self.0.charged_device_bytes.load(Ordering::Acquire) == 0
            && self.0.owned_device_bytes.load(Ordering::Acquire) == 0;
        self.update(|next| {
            next.backend_dropped = true;
            next.buffers_released = released;
            if !released {
                next.accounting_complete = false;
            }
            Some(())
        });
    }
    pub(in crate::pals_onnx) fn backend_retained_unknown(&self) {
        self.0.closed.store(true, Ordering::Release);
        self.0.quarantined.store(true, Ordering::Release);
        self.update(|next| {
            next.buffers_released = false;
            Some(())
        });
    }
}

pub(super) struct CudaWarmInvocationGuard {
    handle: PalsCudaWarmObservationHandle,
    ordinal: u64,
    binding: PalsCudaWarmExecutionBinding,
    finished: bool,
}
impl CudaWarmInvocationGuard {
    pub(super) fn complete_known(mut self, evidence: PalsNNInputEvidence) {
        self.finished = true;
        if self
            .handle
            .0
            .active_leases
            .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            self.handle.incomplete();
        }
        self.handle.update(|next| {
            let invocation = next.last_invocation.as_mut()?;
            if invocation.ordinal != self.ordinal
                || invocation.binding != self.binding
                || invocation.state != PalsCudaWarmInvocationState::Active
            {
                return None;
            }
            invocation.state = PalsCudaWarmInvocationState::CompletedKnown;
            invocation.actual_completed_nn_inputs_delta = evidence.actual_completed_nn_inputs_delta;
            invocation.nn_evidence_complete = evidence.complete;
            next.accounting_complete &= evidence.complete;
            next.leases_physically_completed = next.leases_physically_completed.checked_add(1)?;
            if self.binding.purpose == PalsCudaWarmInvocationPurpose::Startup {
                next.startup_leases_physically_completed =
                    next.startup_leases_physically_completed.checked_add(1)?;
            }
            Some(())
        });
    }
    pub(super) fn quarantine(mut self) {
        self.retain_unknown();
        self.finished = true;
    }
    fn retain_unknown(&self) {
        self.handle.0.closed.store(true, Ordering::Release);
        self.handle.0.quarantined.store(true, Ordering::Release);
        self.handle.update(|next| {
            let invocation = next.last_invocation.as_mut()?;
            if invocation.ordinal != self.ordinal
                || invocation.binding != self.binding
                || invocation.state != PalsCudaWarmInvocationState::Active
            {
                return None;
            }
            invocation.state = PalsCudaWarmInvocationState::Quarantined;
            invocation.actual_completed_nn_inputs_delta = None;
            invocation.nn_evidence_complete = false;
            next.leases_quarantined = next.leases_quarantined.checked_add(1)?;
            if self.binding.purpose == PalsCudaWarmInvocationPurpose::Startup {
                next.startup_leases_quarantined = next.startup_leases_quarantined.checked_add(1)?;
            }
            next.buffers_released = false;
            Some(())
        });
    }
}
impl Drop for CudaWarmInvocationGuard {
    fn drop(&mut self) {
        // An unwind or missing explicit completion is never a completion ACK.
        if !self.finished {
            self.retain_unknown();
        }
    }
}

/// Stored after native Tensor fields: Drop observes actual native value-handle
/// destruction. On unknown work the aggregate and this guard are retained.
pub(in crate::pals_onnx) struct CudaWarmDeviceAllocationGuard {
    handle: PalsCudaWarmObservationHandle,
    reserved_bytes: u64,
    allocated_bytes: u64,
    allocated_tensors: u64,
}
impl CudaWarmDeviceAllocationGuard {
    pub(in crate::pals_onnx) fn tensor_created(&mut self, bytes: u64) {
        let Some(allocated) = self
            .allocated_bytes
            .checked_add(bytes)
            .filter(|sum| bytes != 0 && *sum <= self.reserved_bytes)
        else {
            self.handle.incomplete();
            return;
        };
        let Some(tensors) = self.allocated_tensors.checked_add(1) else {
            self.handle.incomplete();
            return;
        };
        self.allocated_bytes = allocated;
        self.allocated_tensors = tensors;
        let live = match self.handle.0.owned_device_bytes.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |live| live.checked_add(bytes),
        ) {
            Ok(old) => old
                .checked_add(bytes)
                .expect("checked atomic payload addition"),
            Err(_) => {
                self.handle.incomplete();
                return;
            }
        };
        self.handle
            .0
            .peak_device_bytes
            .fetch_max(live, Ordering::AcqRel);
        self.handle.update(|next| {
            next.device_tensors_created = next.device_tensors_created.checked_add(1)?;
            Some(())
        });
    }
}
impl Drop for CudaWarmDeviceAllocationGuard {
    fn drop(&mut self) {
        let owned = self.handle.0.owned_device_bytes.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |live| live.checked_sub(self.allocated_bytes),
        );
        let charged = self.handle.0.charged_device_bytes.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |charged| charged.checked_sub(self.reserved_bytes),
        );
        if owned.is_err() || charged.is_err() {
            self.handle.incomplete();
        }
        self.handle.update(|next| {
            next.device_tensor_handles_released = next
                .device_tensor_handles_released
                .checked_add(self.allocated_tensors)?;
            Some(())
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_model::PalsModelProfile;
    use crate::pals_private::{PrivateModelIdentity, PrivatePrecision};

    // These are CPU metadata/failure-path fixtures. They do not allocate a
    // CUDA tensor, execute an ORT graph or witness any real physical fence.
    fn observer(bytes: u64) -> PalsCudaWarmObservationHandle {
        let capability = PalsCudaWarmCapability {
            identity: PrivateModelIdentity {
                model: [1; 32],
                encoding: [2; 32],
                precision: PrivatePrecision::Fp32,
                model_epoch: [3; 32],
                frozen_epoch: 1,
            },
            rules: [4; 32],
            rules_source: [5; 32],
            profile: PalsModelProfile::FullLineInteractionV2,
            device_id: 0,
            runtime_binary: [6; 32],
            runtime_bundle: [7; 32],
            token: [8; 32],
        };
        PalsCudaWarmObservationHandle::new(
            capability,
            PalsCudaWarmRuntimeLimits {
                max_leases: 1,
                device_bytes_max: bytes,
            },
        )
        .unwrap()
    }
    fn binding(
        owner: &PalsCudaWarmObservationHandle,
        purpose: PalsCudaWarmInvocationPurpose,
    ) -> PalsCudaWarmExecutionBinding {
        let startup = purpose == PalsCudaWarmInvocationPurpose::Startup;
        PalsCudaWarmExecutionBinding {
            backend_owner_id: owner.owner_id(),
            purpose,
            request: (!startup).then_some(RequestId {
                epoch: rz_contracts::ProcessEpoch(7),
                sequence: 11,
            }),
            execution: (!startup).then_some(ExecutionId {
                epoch: rz_contracts::ProcessEpoch(7),
                sequence: 13,
            }),
            bank_lease_id: (purpose == PalsCudaWarmInvocationPurpose::Policy).then_some(17),
        }
    }
    fn fresh() -> PrivateInvocation {
        PrivateInvocation::Fresh { input_key: [9; 32] }
    }
    fn warm() -> PrivateInvocation {
        PrivateInvocation::ApproxCudaWarmV2 {
            input_key: [9; 32],
            seed_seal: [10; 32],
            invocation_key: [11; 32],
        }
    }
    fn latest(owner: &PalsCudaWarmObservationHandle) -> PalsCudaWarmOwnerObservation {
        let snapshot = owner.snapshot();
        assert_eq!(snapshot.status, PalsCudaWarmObservationStatus::Available);
        snapshot.latest.unwrap()
    }
    fn evidence(delta: u64) -> PalsNNInputEvidence {
        PalsNNInputEvidence {
            actual_completed_nn_inputs_delta: Some(delta),
            complete: true,
        }
    }
    #[test]
    fn owner_limits_reject_nonexclusive_or_empty_registration() {
        for limits in [
            PalsCudaWarmRuntimeLimits {
                max_leases: 0,
                device_bytes_max: 1,
            },
            PalsCudaWarmRuntimeLimits {
                max_leases: 2,
                device_bytes_max: 1,
            },
            PalsCudaWarmRuntimeLimits {
                max_leases: 1,
                device_bytes_max: 0,
            },
        ] {
            assert!(limits.validate().is_err());
        }
        let first = observer(128);
        let second = observer(128);
        assert_ne!(first.owner_id(), second.owner_id());
        assert_eq!(first.clone().owner_id(), first.owner_id());
        let initial = latest(&first);
        assert_eq!(initial.device_bytes_live, 0);
        assert_eq!(initial.leases_admitted, 0);
        assert!(!initial.backend_dropped && !initial.buffers_released);
    }
    #[test]
    fn binding_requires_actual_id_pair_exact_owner_and_fresh_value() {
        let owner = observer(128);
        let valid = binding(&owner, PalsCudaWarmInvocationPurpose::Policy);
        valid
            .validate(owner.owner_id(), PalsRole::Critic, warm())
            .unwrap();
        let mut changed = valid;
        changed.execution = None;
        assert!(changed
            .validate(owner.owner_id(), PalsRole::Critic, warm())
            .is_err());
        changed = valid;
        changed.execution.as_mut().unwrap().epoch = rz_contracts::ProcessEpoch(8);
        assert!(changed
            .validate(owner.owner_id(), PalsRole::Critic, warm())
            .is_err());
        changed = valid;
        changed.backend_owner_id += 1;
        assert!(changed
            .validate(owner.owner_id(), PalsRole::Critic, warm())
            .is_err());
        let value = binding(&owner, PalsCudaWarmInvocationPurpose::ValueFresh);
        assert!(value
            .validate(owner.owner_id(), PalsRole::Proposer, warm())
            .is_err());
        assert!(value
            .validate(owner.owner_id(), PalsRole::Critic, fresh())
            .is_err());
        value
            .validate(owner.owner_id(), PalsRole::Proposer, fresh())
            .unwrap();
        let startup = binding(&owner, PalsCudaWarmInvocationPurpose::Startup);
        assert!(startup
            .validate(owner.owner_id(), PalsRole::Proposer, warm())
            .is_err());
        assert!(startup
            .validate(owner.owner_id(), PalsRole::Validator, fresh())
            .is_err());
        assert!(valid
            .validate(
                owner.owner_id(),
                PalsRole::Critic,
                PrivateInvocation::ApproxWarmV1 {
                    input_key: [9; 32],
                    seed_seal: [10; 32],
                    invocation_key: [11; 32],
                }
            )
            .is_err());
    }
    #[test]
    fn admission_is_not_seed_consumption_and_only_one_physical_lease_is_admitted() {
        let owner = observer(128);
        let actual_binding = binding(&owner, PalsCudaWarmInvocationPurpose::Startup);
        let guard = owner
            .begin(actual_binding, PalsRole::Proposer, [9; 32], fresh())
            .unwrap();
        assert!(owner
            .begin(actual_binding, PalsRole::Critic, [9; 32], fresh())
            .is_err());
        let admitted = latest(&owner);
        assert_eq!(admitted.leases_admitted, 1);
        assert_eq!(admitted.leases_active, 1);
        assert_eq!(admitted.leases_peak, 1);
        assert_eq!(admitted.startup_leases_admitted, 1);
        assert_eq!(admitted.approx_private_runs_completed, 0);
        guard.complete_known(evidence(0));
        let complete = latest(&owner);
        assert_eq!(complete.startup_leases_physically_completed, 1);
        assert_eq!(complete.leases_active, 0);
        assert_eq!(complete.approx_private_runs_completed, 0);
        assert!(!complete.last_invocation.unwrap().private_run_completed);
    }
    #[test]
    fn private_run_fence_tracks_warm_consumption_separately_from_fresh_value() {
        let owner = observer(128);
        let policy = binding(&owner, PalsCudaWarmInvocationPurpose::Policy);
        let guard = owner
            .begin(policy, PalsRole::Critic, [9; 32], warm())
            .unwrap();
        assert_eq!(latest(&owner).approx_private_runs_completed, 0);
        owner.private_fence_completed();
        guard.complete_known(evidence(2));
        let seeded = latest(&owner);
        assert_eq!(seeded.approx_private_runs_completed, 1);
        assert_eq!(seeded.fresh_value_private_runs_completed, 0);
        let invocation = seeded.last_invocation.unwrap();
        assert_eq!(invocation.binding, policy);
        assert_eq!(invocation.invocation, warm());
        assert_eq!(invocation.actual_completed_nn_inputs_delta, Some(2));
        let guard = owner
            .begin(
                binding(&owner, PalsCudaWarmInvocationPurpose::ValueFresh),
                PalsRole::Proposer,
                [9; 32],
                fresh(),
            )
            .unwrap();
        owner.private_fence_completed();
        guard.complete_known(evidence(1));
        let value = latest(&owner);
        assert_eq!(value.approx_private_runs_completed, 1);
        assert_eq!(value.fresh_value_private_runs_completed, 1);
        assert_eq!(value.leases_physically_completed, 2);
    }
    #[test]
    fn payload_peak_includes_old_and_replacement_values_until_actual_drop() {
        let owner = observer(256);
        let mut old = owner.reserve_device_payload(128).unwrap();
        assert_eq!(latest(&owner).pending_allocation_bytes, 128);
        old.tensor_created(64);
        old.tensor_created(64);
        let mut replacement = owner.reserve_device_payload(128).unwrap();
        replacement.tensor_created(64);
        assert_eq!(latest(&owner).device_bytes_live, 192);
        replacement.tensor_created(64);
        let overlap = latest(&owner);
        assert_eq!(overlap.device_bytes_peak, 256);
        assert_eq!(overlap.device_tensors_created, 4);
        assert_eq!(overlap.device_tensor_handles_released, 0);
        drop(old);
        assert_eq!(latest(&owner).device_bytes_live, 128);
        drop(replacement);
        let released = latest(&owner);
        assert_eq!(released.device_bytes_live, 0);
        assert_eq!(released.device_bytes_peak, 256);
        assert_eq!(released.device_tensor_handles_released, 4);
        assert!(!released.buffers_released);
        owner.backend_dropped_known();
        let closed = latest(&owner);
        assert!(closed.backend_dropped && closed.admission_closed && closed.buffers_released);
        assert_eq!(closed.pending_allocation_bytes, 0);
    }
    #[test]
    fn partial_allocation_and_limit_failure_never_claim_unallocated_bytes() {
        let owner = observer(128);
        let mut partial = owner.reserve_device_payload(128).unwrap();
        partial.tensor_created(64);
        let first = latest(&owner);
        assert_eq!(first.device_bytes_live, 64);
        assert_eq!(first.pending_allocation_bytes, 64);
        assert!(owner.reserve_device_payload(1).is_err());
        let rejected = latest(&owner);
        assert_eq!(rejected.device_bytes_live, 64);
        assert_eq!(rejected.device_bytes_peak, 64);
        assert_eq!(rejected.device_payload_limit_rejections, 1);
        // Mirrors a known prelaunch failure after the first successful value:
        // aggregate value handles precede its metadata guard in Drop order.
        drop(partial);
        let released = latest(&owner);
        assert_eq!(released.device_bytes_live, 0);
        assert_eq!(released.pending_allocation_bytes, 0);
        assert_eq!(released.device_tensors_created, 1);
        assert_eq!(released.device_tensor_handles_released, 1);
        assert_eq!(released.leases_physically_completed, 0);
    }
    #[test]
    fn unknown_fence_retains_snapshot_and_refuses_future_admission() {
        let owner = observer(128);
        let mut payload = owner.reserve_device_payload(128).unwrap();
        payload.tensor_created(64);
        let actual_binding = binding(&owner, PalsCudaWarmInvocationPurpose::Policy);
        let guard = owner
            .begin(actual_binding, PalsRole::Proposer, [9; 32], warm())
            .unwrap();
        // An unwind/missing explicit completion uses the same quarantine event.
        drop(guard);
        owner.backend_retained_unknown();
        let unknown = latest(&owner);
        assert_eq!(unknown.leases_quarantined, 1);
        assert_eq!(unknown.leases_active, 1);
        assert_eq!(unknown.leases_physically_completed, 0);
        assert_eq!(unknown.device_bytes_retained, 64);
        assert_eq!(unknown.pending_allocation_bytes, 64);
        assert_eq!(unknown.approx_private_runs_completed, 0);
        assert!(!unknown.backend_dropped && !unknown.buffers_released);
        assert!(unknown.admission_closed && owner.has_unfinished_invocation());
        assert!(owner
            .begin(actual_binding, PalsRole::Proposer, [9; 32], warm())
            .is_err());
        assert!(owner.reserve_device_payload(1).is_err());
        assert_eq!(
            unknown.last_invocation.unwrap().state,
            PalsCudaWarmInvocationState::Quarantined
        );
        // Only metadata is cleaned up in this mock. The production aggregate
        // itself is retained, including this guard, on unknown CUDA work.
        drop(payload);
        assert!(!latest(&owner).buffers_released);
    }
    #[test]
    fn observation_contention_marks_earlier_snapshot_without_altering_physical_completion() {
        let owner = observer(128);
        let held = owner.0.latest.lock().unwrap();
        assert_eq!(
            owner.snapshot().status,
            PalsCudaWarmObservationStatus::Contended
        );
        let guard = owner
            .begin(
                binding(&owner, PalsCudaWarmInvocationPurpose::Policy),
                PalsRole::Proposer,
                [9; 32],
                fresh(),
            )
            .unwrap();
        drop(held);
        let earlier = owner.snapshot();
        assert_eq!(earlier.status, PalsCudaWarmObservationStatus::Contended);
        assert_eq!(earlier.latest.unwrap().event_sequence, 0);
        assert_eq!(earlier.attempted_event_sequence, 1);
        // Missing metadata must not quarantine actual known completed work.
        guard.complete_known(evidence(1));
        assert!(!owner.has_unfinished_invocation());
        assert_eq!(
            owner.snapshot().status,
            PalsCudaWarmObservationStatus::Contended
        );
    }
    #[test]
    fn metadata_overflow_and_poison_never_publish_invented_current_counters() {
        let owner = observer(128);
        owner.0.event_sequence.store(u64::MAX, Ordering::Release);
        owner.update(|_| Some(()));
        let exhausted = owner.snapshot();
        assert_eq!(
            exhausted.status,
            PalsCudaWarmObservationStatus::EventSequenceExhausted
        );
        assert_eq!(exhausted.latest.unwrap().event_sequence, 0);
        assert_eq!(exhausted.attempted_event_sequence, u64::MAX);
        let poisoned = observer(128);
        let _ = std::panic::catch_unwind(|| {
            let _held = poisoned.0.latest.lock().unwrap();
            panic!("metadata-only poison fixture");
        });
        let snapshot = poisoned.snapshot();
        assert_eq!(snapshot.status, PalsCudaWarmObservationStatus::Poisoned);
        assert_eq!(snapshot.latest.unwrap().leases_physically_completed, 0);
        assert!(!poisoned.has_unfinished_invocation());
    }
    #[test]
    fn incomplete_nn_accounting_is_separate_from_a_known_physical_fence() {
        let owner = observer(128);
        let guard = owner
            .begin(
                binding(&owner, PalsCudaWarmInvocationPurpose::Policy),
                PalsRole::Proposer,
                [9; 32],
                fresh(),
            )
            .unwrap();
        guard.complete_known(PalsNNInputEvidence {
            actual_completed_nn_inputs_delta: None,
            complete: false,
        });
        let snapshot = owner.snapshot();
        assert_eq!(snapshot.status, PalsCudaWarmObservationStatus::Incomplete);
        let physical = snapshot.latest.unwrap();
        assert_eq!(physical.leases_physically_completed, 1);
        assert_eq!(physical.leases_quarantined, 0);
        let invocation = physical.last_invocation.unwrap();
        assert_eq!(
            invocation.state,
            PalsCudaWarmInvocationState::CompletedKnown
        );
        assert!(!invocation.nn_evidence_complete);
        assert!(!owner.has_unfinished_invocation());
    }
}
