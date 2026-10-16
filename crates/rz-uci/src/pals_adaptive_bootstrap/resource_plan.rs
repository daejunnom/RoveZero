//! Private startup intent, not an affinity/quota owner or applied readback.
//! Shared bootstrap consumes the same bounded worker/queue declaration and
//! original control owner. Dedicated startup stays closed until U3--U6 exist.
use crate::pals_adaptive::AdaptiveExecutionLimits;
use rz_contracts::ProcessEpoch;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_experiments::{PalsCpuSlicesV5, PalsResourcePlanV5, PalsTopologyPolicyV5};
use rz_runtime::adaptive::CpuActor;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResourceIntentErrorV5 {
    InvalidDeclaration,
    DedicatedNotEnabled,
    InvalidEpoch,
    EpochMismatch,
    StartupCanceled,
    StartupExpired,
    WorkerOutsidePlan,
    WorkerBindingMissing,
    WorkerBindingIntentMismatch,
    WorkerOwnerMismatch,
    WorkerActorMismatch,
    ExternalSelectionMismatch,
    ExternalOwnerNotDeclared,
}

/// Not Clone/Deserialize: this private intent is shared only through its one
/// bootstrap Arc. It contains no OS handle, applied mask, success flag, or
/// proof of NativeCaller/physical ORT worker/manager actor placement.
pub(super) struct BootstrapResourceIntentV5 {
    epoch: ProcessEpoch,
    startup_until: Instant,
    startup_cancel: Arc<AtomicBool>,
    own_workers: u16,
    external_declared: bool,
    limits: AdaptiveExecutionLimits,
}
/// Immutable declaration binding minted only by the original startup intent.
/// No Clone/Deserialize or public fields: slot, owner and actor cannot be
/// reconstructed by casting an observed tag. This is not a thread inventory,
/// installation permit, PID/TID readback or exclusive worker reservation.
pub(super) struct OwnWorkerSlotBindingV5<'a> {
    intent: &'a BootstrapResourceIntentV5,
    epoch: ProcessEpoch,
    approved_slot: u8,
    engine_owner: CpuWorkerId,
}
impl OwnWorkerSlotBindingV5<'_> {
    pub(super) fn approved_slot(&self) -> u8 {
        self.approved_slot
    }
    pub(super) fn runtime_actor(&self) -> CpuActor {
        CpuActor::OwnWorker(self.approved_slot())
    }
}
impl std::fmt::Debug for OwnWorkerSlotBindingV5<'_> {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("OwnWorkerSlotBindingV5")
            .field("epoch", &self.epoch)
            .field("approved_slot", &self.approved_slot)
            .field("engine_owner", &self.engine_owner)
            .finish_non_exhaustive()
    }
}

impl BootstrapResourceIntentV5 {
    pub(super) fn prepare(
        resources: &PalsResourcePlanV5,
        slices: &PalsCpuSlicesV5,
        external_declared: bool,
        epoch: ProcessEpoch,
        startup_until: Instant,
        startup_cancel: Arc<AtomicBool>,
        now: Instant,
    ) -> Result<Self, ResourceIntentErrorV5> {
        if epoch.0 == 0 {
            return Err(ResourceIntentErrorV5::InvalidEpoch);
        }
        slices
            .validate()
            .and_then(|()| resources.validate(slices))
            .map_err(|_| ResourceIntentErrorV5::InvalidDeclaration)?;
        if resources.topology != PalsTopologyPolicyV5::SharedControl {
            return Err(ResourceIntentErrorV5::DedicatedNotEnabled);
        }
        let limits = AdaptiveExecutionLimits {
            waiting_jobs_max: usize::try_from(slices.waiting_jobs_max)
                .map_err(|_| ResourceIntentErrorV5::InvalidDeclaration)?,
            worker_commands_max: usize::try_from(slices.worker_commands_max)
                .map_err(|_| ResourceIntentErrorV5::InvalidDeclaration)?,
            role_tickets_max: usize::try_from(slices.role_tickets_max)
                .map_err(|_| ResourceIntentErrorV5::InvalidDeclaration)?,
            paused_bytes_per_worker_max: usize::try_from(slices.paused_bytes_per_worker_max)
                .map_err(|_| ResourceIntentErrorV5::InvalidDeclaration)?,
        };
        limits
            .validate()
            .map_err(|_| ResourceIntentErrorV5::InvalidDeclaration)?;
        let intent = Self {
            epoch,
            startup_until,
            startup_cancel,
            own_workers: u16::try_from(slices.workers)
                .map_err(|_| ResourceIntentErrorV5::InvalidDeclaration)?,
            external_declared,
            limits,
        };
        intent.require_active(epoch, now)?;
        Ok(intent)
    }

    pub(super) fn require_active(
        &self,
        epoch: ProcessEpoch,
        now: Instant,
    ) -> Result<(), ResourceIntentErrorV5> {
        if epoch != self.epoch {
            Err(ResourceIntentErrorV5::EpochMismatch)
        } else if self.startup_cancel.load(Ordering::Acquire) {
            Err(ResourceIntentErrorV5::StartupCanceled)
        } else if now >= self.startup_until {
            Err(ResourceIntentErrorV5::StartupExpired)
        } else {
            Ok(())
        }
    }

    pub(super) fn execution_limits(&self) -> AdaptiveExecutionLimits {
        self.limits
    }

    pub(super) fn own_worker_count(&self) -> usize {
        usize::from(self.own_workers)
    }

    /// Existing consumers keep their 1-based engine owner. Derive it through
    /// the explicit same-intent slot binding, without changing old ready tags
    /// or installing affinity. The runtime actor remains separately 0-based.
    pub(super) fn own_worker_owner(
        &self,
        index: usize,
        epoch: ProcessEpoch,
        now: Instant,
    ) -> Result<CpuWorkerId, ResourceIntentErrorV5> {
        let binding = self.bind_own_worker_slot(index, epoch, now)?;
        self.require_bound_own_worker(
            Some(&binding),
            binding.engine_owner,
            binding.runtime_actor(),
            epoch,
            now,
        )
    }

    pub(super) fn bind_own_worker_slot(
        &self,
        index: usize,
        epoch: ProcessEpoch,
        now: Instant,
    ) -> Result<OwnWorkerSlotBindingV5<'_>, ResourceIntentErrorV5> {
        self.require_active(epoch, now)?;
        if index >= self.own_worker_count() {
            return Err(ResourceIntentErrorV5::WorkerOutsidePlan);
        }
        let approved_slot =
            u8::try_from(index).map_err(|_| ResourceIntentErrorV5::WorkerOutsidePlan)?;
        Ok(OwnWorkerSlotBindingV5 {
            intent: self,
            epoch,
            approved_slot,
            engine_owner: CpuWorkerId(u16::from(approved_slot) + 1),
        })
    }

    /// Resolve a declaration only with a binding from this exact original
    /// intent. Missing/foreign mappings, owner/actor confusion and stale
    /// controls are errors. A successful join is still no OS observation or
    /// permission to enable DedicatedControl/use a raw affinity constructor.
    pub(super) fn require_bound_own_worker(
        &self,
        binding: Option<&OwnWorkerSlotBindingV5<'_>>,
        engine_owner: CpuWorkerId,
        runtime_actor: CpuActor,
        epoch: ProcessEpoch,
        now: Instant,
    ) -> Result<CpuWorkerId, ResourceIntentErrorV5> {
        self.require_active(epoch, now)?;
        let binding = binding.ok_or(ResourceIntentErrorV5::WorkerBindingMissing)?;
        if !std::ptr::eq(binding.intent, self) {
            return Err(ResourceIntentErrorV5::WorkerBindingIntentMismatch);
        }
        if binding.epoch != epoch {
            return Err(ResourceIntentErrorV5::EpochMismatch);
        }
        if usize::from(binding.approved_slot()) >= self.own_worker_count() {
            return Err(ResourceIntentErrorV5::WorkerOutsidePlan);
        }
        let expected_owner = CpuWorkerId(u16::from(binding.approved_slot()) + 1);
        if binding.engine_owner != expected_owner || engine_owner != binding.engine_owner {
            return Err(ResourceIntentErrorV5::WorkerOwnerMismatch);
        }
        if runtime_actor != binding.runtime_actor() {
            return Err(ResourceIntentErrorV5::WorkerActorMismatch);
        }
        Ok(binding.engine_owner)
    }

    pub(super) fn check_external_loaded(
        &self,
        loaded: bool,
        epoch: ProcessEpoch,
        now: Instant,
    ) -> Result<(), ResourceIntentErrorV5> {
        self.require_active(epoch, now)?;
        if loaded != self.external_declared {
            Err(ResourceIntentErrorV5::ExternalSelectionMismatch)
        } else {
            Ok(())
        }
    }

    pub(super) fn external_worker_owner(
        &self,
        epoch: ProcessEpoch,
        now: Instant,
    ) -> Result<CpuWorkerId, ResourceIntentErrorV5> {
        self.require_active(epoch, now)?;
        if !self.external_declared {
            return Err(ResourceIntentErrorV5::ExternalOwnerNotDeclared);
        }
        Ok(CpuWorkerId(self.own_workers + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn prepare(
        resources: &PalsResourcePlanV5,
        slices: &PalsCpuSlicesV5,
        external: bool,
        cancel: Arc<AtomicBool>,
        now: Instant,
    ) -> Result<BootstrapResourceIntentV5, ResourceIntentErrorV5> {
        BootstrapResourceIntentV5::prepare(
            resources,
            slices,
            external,
            ProcessEpoch(7),
            now + Duration::from_secs(1),
            cancel,
            now,
        )
    }

    #[test]
    fn shared_intent_preserves_actual_queue_limits_and_engine_owner_slots() {
        let now = Instant::now();
        let slices = PalsCpuSlicesV5::default();
        let intent = prepare(
            &PalsResourcePlanV5::default(),
            &slices,
            true,
            Arc::new(AtomicBool::new(false)),
            now,
        )
        .unwrap();
        assert_eq!(
            intent.execution_limits(),
            AdaptiveExecutionLimits::default()
        );
        assert_eq!(intent.own_worker_count(), 1);
        assert_eq!(
            intent.own_worker_owner(0, ProcessEpoch(7), now),
            Ok(CpuWorkerId(1))
        );
        assert_eq!(
            intent.external_worker_owner(ProcessEpoch(7), now),
            Ok(CpuWorkerId(2))
        );
        assert_eq!(
            intent.own_worker_owner(1, ProcessEpoch(7), now),
            Err(ResourceIntentErrorV5::WorkerOutsidePlan)
        );
    }

    #[test]
    fn separate_two_and_four_worker_profiles_preserve_unique_owner_slots() {
        for count in [2, 4] {
            let now = Instant::now();
            let slices = PalsCpuSlicesV5 {
                workers: count,
                ..Default::default()
            };
            let resources = PalsResourcePlanV5 {
                profile_id: format!("shared-control-workers{count}-v1"),
                cpu_quota_millis: 4000,
                ..Default::default()
            };
            let intent = prepare(
                &resources,
                &slices,
                true,
                Arc::new(AtomicBool::new(false)),
                now,
            )
            .unwrap();
            for slot in 0..count as usize {
                assert_eq!(
                    intent.own_worker_owner(slot, ProcessEpoch(7), now),
                    Ok(CpuWorkerId(slot as u16 + 1))
                );
            }
            assert_eq!(
                intent.external_worker_owner(ProcessEpoch(7), now),
                Ok(CpuWorkerId(count as u16 + 1))
            );
        }
    }

    #[test]
    fn invalid_resource_or_slice_input_is_rejected_before_owner_creation() {
        let now = Instant::now();
        let resources = PalsResourcePlanV5::default();
        let mut slices = PalsCpuSlicesV5 {
            workers: 2,
            ..PalsCpuSlicesV5::default()
        };
        assert!(matches!(
            prepare(
                &resources,
                &slices,
                false,
                Arc::new(AtomicBool::new(false)),
                now
            ),
            Err(ResourceIntentErrorV5::InvalidDeclaration)
        ));
        slices.workers = 1;
        slices.role_tickets_max = 17;
        assert!(matches!(
            prepare(
                &resources,
                &slices,
                false,
                Arc::new(AtomicBool::new(false)),
                now
            ),
            Err(ResourceIntentErrorV5::InvalidDeclaration)
        ));
        slices.role_tickets_max = 16;
        let mut invalid = resources;
        invalid.ort_spinning = true;
        assert!(matches!(
            prepare(
                &invalid,
                &slices,
                false,
                Arc::new(AtomicBool::new(false)),
                now
            ),
            Err(ResourceIntentErrorV5::InvalidDeclaration)
        ));
    }

    #[test]
    fn valid_dedicated_declaration_still_cannot_mint_an_actual_intent() {
        let resources = PalsResourcePlanV5 {
            topology: PalsTopologyPolicyV5::DedicatedControl,
            cloud_or_topology_unknown: false,
            allowed_logical_cpus: vec![0, 2],
            dedicated_manager_cpu: Some(0),
            worker_logical_cpus: vec![2],
            ..Default::default()
        };
        let slices = PalsCpuSlicesV5::default();
        resources.validate(&slices).unwrap();
        assert!(matches!(
            prepare(
                &resources,
                &slices,
                false,
                Arc::new(AtomicBool::new(false)),
                Instant::now()
            ),
            Err(ResourceIntentErrorV5::DedicatedNotEnabled)
        ));
    }

    #[test]
    fn original_cancel_deadline_and_epoch_are_not_replaced_by_new_controls() {
        let now = Instant::now();
        let cancel = Arc::new(AtomicBool::new(false));
        let intent = prepare(
            &PalsResourcePlanV5::default(),
            &PalsCpuSlicesV5::default(),
            false,
            Arc::clone(&cancel),
            now,
        )
        .unwrap();
        assert_eq!(
            intent.require_active(ProcessEpoch(8), now),
            Err(ResourceIntentErrorV5::EpochMismatch)
        );
        assert_eq!(
            intent.require_active(ProcessEpoch(7), now + Duration::from_secs(1)),
            Err(ResourceIntentErrorV5::StartupExpired)
        );
        cancel.store(true, Ordering::Release);
        assert_eq!(
            intent.own_worker_owner(0, ProcessEpoch(7), now),
            Err(ResourceIntentErrorV5::StartupCanceled)
        );
        assert_eq!(
            intent.external_worker_owner(ProcessEpoch(7), now),
            Err(ResourceIntentErrorV5::StartupCanceled)
        );
    }

    #[test]
    fn external_selection_and_owner_are_bound_to_original_declaration() {
        let now = Instant::now();
        for declared in [false, true] {
            let intent = prepare(
                &PalsResourcePlanV5::default(),
                &PalsCpuSlicesV5::default(),
                declared,
                Arc::new(AtomicBool::new(false)),
                now,
            )
            .unwrap();
            assert_eq!(
                intent.check_external_loaded(declared, ProcessEpoch(7), now),
                Ok(())
            );
            assert_eq!(
                intent.check_external_loaded(!declared, ProcessEpoch(7), now),
                Err(ResourceIntentErrorV5::ExternalSelectionMismatch)
            );
            if !declared {
                assert_eq!(
                    intent.external_worker_owner(ProcessEpoch(7), now),
                    Err(ResourceIntentErrorV5::ExternalOwnerNotDeclared)
                );
            }
        }
    }

    #[test]
    fn zero_epoch_or_original_canceled_or_expired_intent_is_rejected() {
        let now = Instant::now();
        let resources = PalsResourcePlanV5::default();
        let slices = PalsCpuSlicesV5::default();
        for (epoch, deadline, canceled, expected) in [
            (
                0,
                now + Duration::from_secs(1),
                false,
                ResourceIntentErrorV5::InvalidEpoch,
            ),
            (7, now, false, ResourceIntentErrorV5::StartupExpired),
            (
                7,
                now + Duration::from_secs(1),
                true,
                ResourceIntentErrorV5::StartupCanceled,
            ),
        ] {
            assert!(
                matches!(BootstrapResourceIntentV5::prepare(&resources, &slices, false, ProcessEpoch(epoch), deadline, Arc::new(AtomicBool::new(canceled)), now), Err(error) if error == expected)
            );
        }
    }

    #[test]
    fn approved_slots_bind_zero_based_actors_to_existing_one_based_engine_owners() {
        for count in [1, 2, 4] {
            let now = Instant::now();
            let slices = PalsCpuSlicesV5 {
                workers: count,
                ..Default::default()
            };
            let resources = PalsResourcePlanV5 {
                profile_id: format!("shared-control-workers{count}-slot-binding-v1"),
                cpu_quota_millis: 4000,
                ..Default::default()
            };
            let intent = prepare(
                &resources,
                &slices,
                false,
                Arc::new(AtomicBool::new(false)),
                now,
            )
            .unwrap();
            for slot in 0..count as usize {
                let binding = intent
                    .bind_own_worker_slot(slot, ProcessEpoch(7), now)
                    .unwrap();
                assert_eq!(binding.approved_slot(), slot as u8);
                assert_eq!(binding.runtime_actor(), CpuActor::OwnWorker(slot as u8));
                let expected_owner = CpuWorkerId(slot as u16 + 1);
                assert_eq!(
                    intent.require_bound_own_worker(
                        Some(&binding),
                        expected_owner,
                        CpuActor::OwnWorker(slot as u8),
                        ProcessEpoch(7),
                        now,
                    ),
                    Ok(expected_owner)
                );
                assert_eq!(
                    intent.own_worker_owner(slot, ProcessEpoch(7), now),
                    Ok(expected_owner)
                );
            }
            assert!(matches!(
                intent.bind_own_worker_slot(count as usize, ProcessEpoch(7), now),
                Err(ResourceIntentErrorV5::WorkerOutsidePlan)
            ));
            assert!(matches!(
                intent.bind_own_worker_slot(usize::MAX, ProcessEpoch(7), now),
                Err(ResourceIntentErrorV5::WorkerOutsidePlan)
            ));
        }
    }

    #[test]
    fn missing_foreign_and_owner_actor_mismatched_bindings_are_rejected() {
        let now = Instant::now();
        let resources = PalsResourcePlanV5::default();
        let slices = PalsCpuSlicesV5::default();
        let intent = prepare(
            &resources,
            &slices,
            false,
            Arc::new(AtomicBool::new(false)),
            now,
        )
        .unwrap();
        let other = prepare(
            &resources,
            &slices,
            false,
            Arc::new(AtomicBool::new(false)),
            now,
        )
        .unwrap();
        let binding = intent
            .bind_own_worker_slot(0, ProcessEpoch(7), now)
            .unwrap();
        assert_eq!(
            intent.require_bound_own_worker(
                None,
                CpuWorkerId(1),
                CpuActor::OwnWorker(0),
                ProcessEpoch(7),
                now
            ),
            Err(ResourceIntentErrorV5::WorkerBindingMissing)
        );
        // Equal declared values/epoch do not make a second startup intent the
        // same owner of this binding or its original cancellation authority.
        assert_eq!(
            other.require_bound_own_worker(
                Some(&binding),
                CpuWorkerId(1),
                CpuActor::OwnWorker(0),
                ProcessEpoch(7),
                now
            ),
            Err(ResourceIntentErrorV5::WorkerBindingIntentMismatch)
        );
        for owner in [CpuWorkerId(0), CpuWorkerId(2), CpuWorkerId(u16::MAX)] {
            assert_eq!(
                intent.require_bound_own_worker(
                    Some(&binding),
                    owner,
                    CpuActor::OwnWorker(0),
                    ProcessEpoch(7),
                    now
                ),
                Err(ResourceIntentErrorV5::WorkerOwnerMismatch)
            );
        }
        for actor in [
            CpuActor::OwnWorker(1),
            CpuActor::Manager,
            CpuActor::NativeCaller,
            CpuActor::ExternalEngine(0),
            CpuActor::RuntimeHelper(0),
        ] {
            assert_eq!(
                intent.require_bound_own_worker(
                    Some(&binding),
                    CpuWorkerId(1),
                    actor,
                    ProcessEpoch(7),
                    now
                ),
                Err(ResourceIntentErrorV5::WorkerActorMismatch)
            );
        }
    }

    #[test]
    fn binding_does_not_freeze_away_original_cancel_deadline_or_epoch() {
        let now = Instant::now();
        let cancel = Arc::new(AtomicBool::new(false));
        let intent = prepare(
            &PalsResourcePlanV5::default(),
            &PalsCpuSlicesV5::default(),
            false,
            Arc::clone(&cancel),
            now,
        )
        .unwrap();
        let binding = intent
            .bind_own_worker_slot(0, ProcessEpoch(7), now)
            .unwrap();
        assert!(matches!(
            intent.bind_own_worker_slot(0, ProcessEpoch(8), now),
            Err(ResourceIntentErrorV5::EpochMismatch)
        ));
        assert_eq!(
            intent.require_bound_own_worker(
                Some(&binding),
                CpuWorkerId(1),
                CpuActor::OwnWorker(0),
                ProcessEpoch(8),
                now
            ),
            Err(ResourceIntentErrorV5::EpochMismatch)
        );
        assert_eq!(
            intent.require_bound_own_worker(
                Some(&binding),
                CpuWorkerId(1),
                CpuActor::OwnWorker(0),
                ProcessEpoch(7),
                now + Duration::from_secs(1)
            ),
            Err(ResourceIntentErrorV5::StartupExpired)
        );
        cancel.store(true, Ordering::Release);
        assert_eq!(
            intent.require_bound_own_worker(
                Some(&binding),
                CpuWorkerId(1),
                CpuActor::OwnWorker(0),
                ProcessEpoch(7),
                now
            ),
            Err(ResourceIntentErrorV5::StartupCanceled)
        );
        assert!(matches!(
            intent.bind_own_worker_slot(0, ProcessEpoch(7), now),
            Err(ResourceIntentErrorV5::StartupCanceled)
        ));
    }

    #[test]
    fn malformed_private_fixture_does_not_authorize_a_slot_or_rewrite_owner_numbers() {
        // Only this same-module fixture can access binding fields. No product
        // caller can deserialize or assemble these malformed declarations.
        let now = Instant::now();
        let intent = prepare(
            &PalsResourcePlanV5::default(),
            &PalsCpuSlicesV5::default(),
            false,
            Arc::new(AtomicBool::new(false)),
            now,
        )
        .unwrap();
        let stale_epoch = OwnWorkerSlotBindingV5 {
            intent: &intent,
            epoch: ProcessEpoch(8),
            approved_slot: 0,
            engine_owner: CpuWorkerId(1),
        };
        assert_eq!(
            intent.require_bound_own_worker(
                Some(&stale_epoch),
                CpuWorkerId(1),
                CpuActor::OwnWorker(0),
                ProcessEpoch(7),
                now
            ),
            Err(ResourceIntentErrorV5::EpochMismatch)
        );
        let unapproved = OwnWorkerSlotBindingV5 {
            intent: &intent,
            epoch: ProcessEpoch(7),
            approved_slot: 1,
            engine_owner: CpuWorkerId(2),
        };
        assert_eq!(
            intent.require_bound_own_worker(
                Some(&unapproved),
                CpuWorkerId(2),
                CpuActor::OwnWorker(1),
                ProcessEpoch(7),
                now
            ),
            Err(ResourceIntentErrorV5::WorkerOutsidePlan)
        );
        let wrong_owner = OwnWorkerSlotBindingV5 {
            intent: &intent,
            epoch: ProcessEpoch(7),
            approved_slot: 0,
            engine_owner: CpuWorkerId(0),
        };
        assert_eq!(
            intent.require_bound_own_worker(
                Some(&wrong_owner),
                CpuWorkerId(0),
                CpuActor::OwnWorker(0),
                ProcessEpoch(7),
                now
            ),
            Err(ResourceIntentErrorV5::WorkerOwnerMismatch)
        );
    }
}
