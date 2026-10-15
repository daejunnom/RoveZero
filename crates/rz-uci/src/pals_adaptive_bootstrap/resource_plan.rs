//! Private startup intent, not an affinity/quota owner or applied readback.
//! Shared bootstrap consumes the same bounded worker/queue declaration and
//! original control owner. Dedicated startup stays closed until U3--U6 exist.
use crate::pals_adaptive::AdaptiveExecutionLimits;
use rz_contracts::ProcessEpoch;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_experiments::{PalsCpuSlicesV5, PalsResourcePlanV5, PalsTopologyPolicyV5};
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

    /// These are existing 1-based CPU engine owners, not runtime's 0-based
    /// OwnWorker actors. U3/U4 must join actual ready TIDs by the trusted slot.
    pub(super) fn own_worker_owner(
        &self,
        index: usize,
        epoch: ProcessEpoch,
        now: Instant,
    ) -> Result<CpuWorkerId, ResourceIntentErrorV5> {
        self.require_active(epoch, now)?;
        if index >= self.own_worker_count() {
            return Err(ResourceIntentErrorV5::WorkerOutsidePlan);
        }
        // prepare admitted only 1/2/4 workers; the checked bound precedes cast.
        Ok(CpuWorkerId(index as u16 + 1))
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
        let mut slices = PalsCpuSlicesV5::default();
        slices.workers = 2;
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
}
