//! Opt-in ownership of one actual worker thread's affinity.
//!
//! This owner is created inside the worker, never on its spawning thread. It
//! grants no actor-slot mapping, process quota or DedicatedControl admission.
use super::{LogicalCpu, CPU_PROBE_MAX_LOGICAL_CPUS};
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuThreadAffinityObservation {
    pub process_id: u32,
    pub thread_id: u64,
    pub cpus: Vec<LogicalCpu>,
    pub identity_stable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuThreadAffinityApplied {
    pub original: CpuThreadAffinityObservation,
    pub applied: CpuThreadAffinityObservation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuThreadAffinityStartError {
    InvalidRequest,
    UnsupportedPlatform,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    Native(Box<rz_platform::ThreadAffinityApplyError>),
}

impl CpuThreadAffinityStartError {
    /// Only pre-mutation rejection or identity/mask-verified rollback permits
    /// recovery of an unstarted CPU engine. This never turns failed placement
    /// into success. Unknown/failed rollback remains quarantined even though
    /// no search command was admitted.
    pub fn permits_unstarted_engine_recovery(&self) -> bool {
        match self {
            Self::InvalidRequest | Self::UnsupportedPlatform => true,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            Self::Native(error) => match &error.rollback {
                rz_platform::ThreadAffinityRollback::NotAttempted => matches!(
                    error.primary.stage,
                    rz_platform::ThreadAffinityStage::Validate
                        | rz_platform::ThreadAffinityStage::ObserveOriginal
                ),
                rz_platform::ThreadAffinityRollback::OriginalAlreadyObserved(observed)
                | rz_platform::ThreadAffinityRollback::Restored(observed) => {
                    let (Some(original), Some(target)) =
                        (error.original.as_ref(), error.rollback_target.as_ref())
                    else {
                        return false;
                    };
                    let original = observation(original);
                    let after = observation(observed);
                    original.identity_stable
                        && after.identity_stable
                        && original.process_id != 0
                        && original.thread_id != 0
                        && original.process_id == after.process_id
                        && original.thread_id == after.thread_id
                        && observed.mask() == *target
                }
                rz_platform::ThreadAffinityRollback::Failed(_) => false,
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuThreadAffinityFinishError {
    UnsupportedPlatform,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    Native(Box<rz_platform::ThreadAffinityRestoreError>),
}

/// A guard stays local to the actual worker. Only cloned observation receipts
/// may travel to its manager. Explicit finish is required; Drop cannot attest
/// restoration, and an explicit error remains an error after Drop's fallback.
///
/// ```compile_fail
/// fn requires_send<T: Send>() {}
/// requires_send::<rz_runtime::adaptive::CurrentCpuThreadAffinityOwner>();
/// ```
/// ```compile_fail
/// fn requires_sync<T: Sync>() {}
/// requires_sync::<rz_runtime::adaptive::CurrentCpuThreadAffinityOwner>();
/// ```
#[must_use = "keep this owner on the worker and retain its explicit finish result"]
pub struct CurrentCpuThreadAffinityOwner {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    guard: rz_platform::CurrentThreadAffinityGuard,
    evidence: CpuThreadAffinityApplied,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl CurrentCpuThreadAffinityOwner {
    /// Restrict only the calling thread to a nonempty subset of its original
    /// observed mask. The platform guard performs subset, identity, apply and
    /// readback verification and owns rollback on failed installation.
    pub fn begin(cpus: Vec<LogicalCpu>) -> Result<Self, CpuThreadAffinityStartError> {
        let cpus = validate_request(cpus)?;
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            let requested = platform_mask(&cpus);
            let guard = rz_platform::apply_current_thread_affinity(requested)
                .map_err(CpuThreadAffinityStartError::Native)?;
            let evidence = CpuThreadAffinityApplied {
                original: observation(guard.original()),
                applied: observation(guard.applied()),
            };
            Ok(Self {
                guard,
                evidence,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            let _ = cpus;
            Err(CpuThreadAffinityStartError::UnsupportedPlatform)
        }
    }

    pub fn applied(&self) -> &CpuThreadAffinityApplied {
        &self.evidence
    }

    /// Call on the same actual thread after its work/token cleanup. This does
    /// not dispose native sessions, join another worker, or attest GPU fences.
    pub fn finish(&mut self) -> Result<CpuThreadAffinityObservation, CpuThreadAffinityFinishError> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            self.guard
                .finish()
                .map(|snapshot| observation(&snapshot))
                .map_err(CpuThreadAffinityFinishError::Native)
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            Err(CpuThreadAffinityFinishError::UnsupportedPlatform)
        }
    }
}

fn validate_request(
    mut cpus: Vec<LogicalCpu>,
) -> Result<Vec<LogicalCpu>, CpuThreadAffinityStartError> {
    if cpus.is_empty() || cpus.len() > CPU_PROBE_MAX_LOGICAL_CPUS {
        return Err(CpuThreadAffinityStartError::InvalidRequest);
    }
    cpus.sort_unstable();
    if cpus.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(CpuThreadAffinityStartError::InvalidRequest);
    }
    #[cfg(target_os = "linux")]
    if cpus.iter().any(|cpu| cpu.group != 0 || cpu.index >= 4096) {
        return Err(CpuThreadAffinityStartError::InvalidRequest);
    }
    #[cfg(target_os = "windows")]
    if cpus
        .iter()
        .any(|cpu| cpu.group >= 64 || cpu.index >= 64 || cpu.group != cpus[0].group)
    {
        return Err(CpuThreadAffinityStartError::InvalidRequest);
    }
    Ok(cpus)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn platform_mask(cpus: &[LogicalCpu]) -> rz_platform::ThreadAffinityMask {
    #[cfg(target_os = "linux")]
    {
        rz_platform::ThreadAffinityMask::Linux {
            cpu_ids: cpus.iter().map(|cpu| cpu.index).collect(),
        }
    }
    #[cfg(target_os = "windows")]
    {
        rz_platform::ThreadAffinityMask::Windows {
            group: cpus[0].group,
            mask: cpus
                .iter()
                .fold(0u64, |mask, cpu| mask | (1u64 << cpu.index)),
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn observation(snapshot: &rz_platform::ThreadAffinitySnapshot) -> CpuThreadAffinityObservation {
    match snapshot {
        rz_platform::ThreadAffinitySnapshot::Linux(value) => CpuThreadAffinityObservation {
            process_id: value.process_id,
            thread_id: value.thread_id,
            cpus: value
                .cpu_ids
                .iter()
                .map(|index| LogicalCpu {
                    group: 0,
                    index: *index,
                })
                .collect(),
            identity_stable: value.identity_stable,
        },
        rz_platform::ThreadAffinitySnapshot::Windows(value) => CpuThreadAffinityObservation {
            process_id: value.process_id,
            thread_id: value.thread_id,
            cpus: (0u16..64)
                .filter(|index| value.group_mask.1 & (1u64 << *index) != 0)
                .map(|index| LogicalCpu {
                    group: value.group_mask.0,
                    index,
                })
                .collect(),
            identity_stable: value.identity_stable,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_requests_are_rejected_before_any_platform_call() {
        let cpu = LogicalCpu { group: 0, index: 0 };
        assert_eq!(
            validate_request(Vec::new()),
            Err(CpuThreadAffinityStartError::InvalidRequest)
        );
        assert_eq!(
            validate_request(vec![cpu, cpu]),
            Err(CpuThreadAffinityStartError::InvalidRequest)
        );
        assert_eq!(
            validate_request(vec![cpu; CPU_PROBE_MAX_LOGICAL_CPUS + 1]),
            Err(CpuThreadAffinityStartError::InvalidRequest)
        );
    }

    #[test]
    fn current_os_mask_constraints_are_checked_without_mutating_affinity() {
        #[cfg(target_os = "linux")]
        for cpu in [
            LogicalCpu { group: 1, index: 0 },
            LogicalCpu {
                group: 0,
                index: 4096,
            },
        ] {
            assert_eq!(
                validate_request(vec![cpu]),
                Err(CpuThreadAffinityStartError::InvalidRequest)
            );
        }
        #[cfg(target_os = "windows")]
        {
            for cpu in [
                LogicalCpu {
                    group: 64,
                    index: 0,
                },
                LogicalCpu {
                    group: 0,
                    index: 64,
                },
            ] {
                assert_eq!(
                    validate_request(vec![cpu]),
                    Err(CpuThreadAffinityStartError::InvalidRequest)
                );
            }
            assert_eq!(
                validate_request(vec![
                    LogicalCpu { group: 0, index: 0 },
                    LogicalCpu { group: 1, index: 0 }
                ]),
                Err(CpuThreadAffinityStartError::InvalidRequest)
            );
        }
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn recovery_snapshot() -> rz_platform::ThreadAffinitySnapshot {
        rz_platform::ThreadAffinitySnapshot::Linux(rz_platform::LinuxThreadAffinity {
            process_id: 101,
            thread_id: 102,
            cpu_ids: vec![0, 1],
            identity_stable: true,
        })
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn recovery_error(
        rollback: rz_platform::ThreadAffinityRollback,
    ) -> CpuThreadAffinityStartError {
        CpuThreadAffinityStartError::Native(Box::new(rz_platform::ThreadAffinityApplyError {
            primary: rz_platform::ThreadAffinityFailureRecord {
                stage: rz_platform::ThreadAffinityStage::VerifyApplied,
                failure: rz_platform::ThreadAffinityFailure::ReadbackMismatch,
                observed: None,
                set_conflict: None,
            },
            requested: rz_platform::ThreadAffinityMask::Linux { cpu_ids: vec![0] },
            original: Some(recovery_snapshot()),
            rollback_target: Some(rz_platform::ThreadAffinityMask::Linux {
                cpu_ids: vec![0, 1],
            }),
            rollback,
        }))
    }

    #[test]
    fn only_known_pre_mutation_rejection_can_recover_without_rollback() {
        assert!(CpuThreadAffinityStartError::InvalidRequest.permits_unstarted_engine_recovery());
        assert!(
            CpuThreadAffinityStartError::UnsupportedPlatform.permits_unstarted_engine_recovery()
        );
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            let CpuThreadAffinityStartError::Native(mut error) =
                recovery_error(rz_platform::ThreadAffinityRollback::NotAttempted)
            else {
                unreachable!();
            };
            assert!(!CpuThreadAffinityStartError::Native(error.clone())
                .permits_unstarted_engine_recovery());
            for stage in [
                rz_platform::ThreadAffinityStage::Validate,
                rz_platform::ThreadAffinityStage::ObserveOriginal,
            ] {
                error.primary.stage = stage;
                assert!(CpuThreadAffinityStartError::Native(error.clone())
                    .permits_unstarted_engine_recovery());
            }
            error.primary.stage = rz_platform::ThreadAffinityStage::Apply;
            assert!(!CpuThreadAffinityStartError::Native(error).permits_unstarted_engine_recovery());
        }
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[test]
    fn verified_rollback_requires_original_identity_and_exact_target_readback() {
        for rollback in [
            rz_platform::ThreadAffinityRollback::OriginalAlreadyObserved(recovery_snapshot()),
            rz_platform::ThreadAffinityRollback::Restored(recovery_snapshot()),
        ] {
            let error = recovery_error(rollback);
            assert!(error.permits_unstarted_engine_recovery());
        }
        let CpuThreadAffinityStartError::Native(mut error) = recovery_error(
            rz_platform::ThreadAffinityRollback::Restored(recovery_snapshot()),
        ) else {
            unreachable!();
        };
        error.rollback_target = None;
        assert!(
            !CpuThreadAffinityStartError::Native(error.clone()).permits_unstarted_engine_recovery()
        );
        error.rollback_target = Some(rz_platform::ThreadAffinityMask::Linux { cpu_ids: vec![0] });
        assert!(
            !CpuThreadAffinityStartError::Native(error.clone()).permits_unstarted_engine_recovery()
        );
        error.rollback_target = Some(rz_platform::ThreadAffinityMask::Linux {
            cpu_ids: vec![0, 1],
        });
        let Some(rz_platform::ThreadAffinitySnapshot::Linux(original)) = error.original.as_mut()
        else {
            unreachable!();
        };
        original.thread_id += 1;
        assert!(
            !CpuThreadAffinityStartError::Native(error.clone()).permits_unstarted_engine_recovery()
        );
        error.original = None;
        assert!(!CpuThreadAffinityStartError::Native(error).permits_unstarted_engine_recovery());
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[test]
    fn failed_rollback_is_not_recoverable_even_if_a_readback_happens_to_match() {
        let error = recovery_error(rz_platform::ThreadAffinityRollback::Failed(
            rz_platform::ThreadAffinityFailureRecord {
                stage: rz_platform::ThreadAffinityStage::Rollback,
                failure: rz_platform::ThreadAffinityFailure::ReadbackMismatch,
                observed: Some(recovery_snapshot()),
                set_conflict: None,
            },
        ));
        assert!(!error.permits_unstarted_engine_recovery());
    }
}
