//! Current-thread mutation owns its original mask until explicit restoration.
//! OS facts come from the existing observations, never from requested masks.
use crate::{LinuxThreadAffinity, PlatformCpuError, WindowsThreadAffinity};
use std::marker::PhantomData;
use std::rc::Rc;
use std::thread::ThreadId;

#[cfg(target_os = "linux")]
use crate::linux::NativeThreadAffinity;
#[cfg(target_os = "windows")]
use crate::windows::NativeThreadAffinity;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ThreadAffinityMask {
    Linux { cpu_ids: Vec<u16> },
    Windows { group: u16, mask: u64 },
}
impl ThreadAffinityMask {
    fn validate(&mut self) -> Result<(), ThreadAffinityFailure> {
        match self {
            Self::Linux { cpu_ids } => {
                if cpu_ids.is_empty() {
                    return Err(ThreadAffinityFailure::EmptyMask);
                }
                if cpu_ids.len() > 4096 || cpu_ids.iter().any(|cpu| *cpu >= 4096) {
                    return Err(ThreadAffinityFailure::CpuOutOfRange);
                }
                cpu_ids.sort_unstable();
                if cpu_ids.windows(2).any(|pair| pair[0] == pair[1]) {
                    return Err(ThreadAffinityFailure::DuplicateCpu);
                }
            }
            Self::Windows { group, mask } => {
                if *mask == 0 {
                    return Err(ThreadAffinityFailure::EmptyMask);
                }
                if *group >= 64 {
                    return Err(ThreadAffinityFailure::WindowsGroupOutOfRange);
                }
            }
        }
        Ok(())
    }
    fn is_subset_of(&self, allowed: &Self) -> bool {
        match (self, allowed) {
            (Self::Linux { cpu_ids }, Self::Linux { cpu_ids: allowed }) => {
                cpu_ids.iter().all(|cpu| allowed.binary_search(cpu).is_ok())
            }
            (
                Self::Windows { group, mask },
                Self::Windows {
                    group: other,
                    mask: allowed,
                },
            ) => group == other && mask & !allowed == 0,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ThreadAffinitySnapshot {
    Linux(LinuxThreadAffinity),
    Windows(WindowsThreadAffinity),
}
impl ThreadAffinitySnapshot {
    pub fn mask(&self) -> ThreadAffinityMask {
        match self {
            Self::Linux(value) => ThreadAffinityMask::Linux {
                cpu_ids: value.cpu_ids.clone(),
            },
            Self::Windows(value) => ThreadAffinityMask::Windows {
                group: value.group_mask.0,
                mask: value.group_mask.1,
            },
        }
    }
    fn identity(&self) -> (u32, u64) {
        match self {
            Self::Linux(value) => (value.process_id, value.thread_id),
            Self::Windows(value) => (value.process_id, value.thread_id),
        }
    }
    fn stable(&self) -> bool {
        match self {
            Self::Linux(value) => value.identity_stable,
            Self::Windows(value) => value.identity_stable,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadAffinityStage {
    Validate,
    ObserveOriginal,
    Apply,
    VerifyApplied,
    Rollback,
    Restore,
    VerifyRestored,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadAffinityFailure {
    Native(PlatformCpuError),
    EmptyMask,
    DuplicateCpu,
    CpuOutOfRange,
    WindowsGroupOutOfRange,
    WrongPlatformMask,
    OutsideOriginalMask,
    WrongThread,
    OriginChanged,
    MaskChangedExternally,
    ReadbackMismatch,
    AlreadyFinished,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadAffinityFailureRecord {
    pub stage: ThreadAffinityStage,
    pub failure: ThreadAffinityFailure,
    pub observed: Option<ThreadAffinitySnapshot>,
    /// Flat, bounded observations of a successful Set whose atomic previous
    /// mask disproved the preceding query. Compensation never makes it success.
    pub set_conflict: Option<Box<ThreadAffinitySetConflict>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadAffinityCause {
    pub stage: ThreadAffinityStage,
    pub failure: ThreadAffinityFailure,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadAffinityCompensationOutcome {
    NotAttempted,
    PreviousAlreadyObserved,
    PreviousRestored,
    Failed(ThreadAffinityCause),
}
/// This record does not contain another FailureRecord or another conflict.
/// Exactly one compensating Set is permitted, including a race in that Set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadAffinitySetConflict {
    pub expected_previous: ThreadAffinityMask,
    pub actual_previous: ThreadAffinityMask,
    pub written: ThreadAffinityMask,
    pub before_compensation: Option<ThreadAffinitySnapshot>,
    pub compensation_attempted: bool,
    pub compensation_previous: Option<ThreadAffinityMask>,
    pub compensation_error: Option<PlatformCpuError>,
    pub after_compensation: Option<ThreadAffinitySnapshot>,
    pub after_probe_failure: Option<ThreadAffinityCause>,
    pub outcome: ThreadAffinityCompensationOutcome,
}
fn failure(
    stage: ThreadAffinityStage,
    failure: ThreadAffinityFailure,
) -> ThreadAffinityFailureRecord {
    ThreadAffinityFailureRecord {
        stage,
        failure,
        observed: None,
        set_conflict: None,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ThreadAffinityRollback {
    NotAttempted,
    OriginalAlreadyObserved(ThreadAffinitySnapshot),
    Restored(ThreadAffinitySnapshot),
    Failed(ThreadAffinityFailureRecord),
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadAffinityApplyError {
    pub primary: ThreadAffinityFailureRecord,
    pub requested: ThreadAffinityMask,
    pub original: Option<ThreadAffinitySnapshot>,
    /// Windows supplies the actual previous mask atomically with Set. A race
    /// against the earlier query rolls back that mask, rather than stale data.
    pub rollback_target: Option<ThreadAffinityMask>,
    pub rollback: ThreadAffinityRollback,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadAffinityRestoreError {
    pub failure: ThreadAffinityFailureRecord,
    pub original: ThreadAffinitySnapshot,
    pub applied: ThreadAffinitySnapshot,
}

/// Implementations have no foreign-thread target: Linux uses pid=0 and Windows
/// uses GetCurrentThread(). This private seam also permits deterministic fault
/// injection without OS changes in failure-path tests.
pub(crate) trait ThreadAffinityAccess {
    fn accepts(&self, mask: &ThreadAffinityMask) -> bool;
    fn observe(&mut self) -> Result<ThreadAffinitySnapshot, PlatformCpuError>;
    fn set(
        &mut self,
        mask: &ThreadAffinityMask,
    ) -> Result<Option<ThreadAffinityMask>, PlatformCpuError>;
}

struct AffinityOwner<A: ThreadAffinityAccess> {
    access: A,
    origin_thread: ThreadId,
    original: ThreadAffinitySnapshot,
    applied: ThreadAffinitySnapshot,
    finished: bool,
    first_restore_failure: Option<ThreadAffinityFailureRecord>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

/// Owns only this calling thread's affinity. It is neither serializable nor
/// movable to another thread, and does not attest DedicatedControl admission.
/// Call `finish` on the origin thread and retain its error. Drop is a final
/// best-effort cleanup only; it emits no restoration success evidence.
///
/// ```compile_fail
/// fn requires_send<T: Send>() {}
/// requires_send::<rz_platform::CurrentThreadAffinityGuard>();
/// ```
/// ```compile_fail
/// fn requires_sync<T: Sync>() {}
/// requires_sync::<rz_platform::CurrentThreadAffinityGuard>();
/// ```
#[must_use = "retain the current-thread owner and call finish to observe restoration"]
pub struct CurrentThreadAffinityGuard {
    owner: AffinityOwner<NativeThreadAffinity>,
}

/// Restrict the actual current thread to a nonempty subset of its original
/// observed mask. No process-wide setting or other thread is changed. Linux
/// IDs remain bounded to 0..4096; Windows keeps group and mask separate and
/// rejects the existing ambiguous multi-group case.
pub fn apply_current_thread_affinity(
    requested: ThreadAffinityMask,
) -> Result<CurrentThreadAffinityGuard, Box<ThreadAffinityApplyError>> {
    AffinityOwner::begin(NativeThreadAffinity, requested)
        .map(|owner| CurrentThreadAffinityGuard { owner })
}
impl CurrentThreadAffinityGuard {
    pub fn original(&self) -> &ThreadAffinitySnapshot {
        &self.owner.original
    }
    pub fn applied(&self) -> &ThreadAffinitySnapshot {
        &self.owner.applied
    }
    pub fn first_restore_failure(&self) -> Option<&ThreadAffinityFailureRecord> {
        self.owner.first_restore_failure.as_ref()
    }
    /// Re-read the origin thread and applied mask before restoring. A changed
    /// mask is an error, not permission to overwrite another owner's setting.
    /// Failed explicit restoration remains visible even if Drop later retries.
    pub fn finish(&mut self) -> Result<ThreadAffinitySnapshot, Box<ThreadAffinityRestoreError>> {
        self.owner.finish()
    }
}

fn checked_observe<A: ThreadAffinityAccess>(
    access: &mut A,
    origin_thread: ThreadId,
    original: &ThreadAffinitySnapshot,
    stage: ThreadAffinityStage,
) -> Result<ThreadAffinitySnapshot, ThreadAffinityFailureRecord> {
    if std::thread::current().id() != origin_thread {
        return Err(failure(stage, ThreadAffinityFailure::WrongThread));
    }
    let observed = access
        .observe()
        .map_err(|error| failure(stage, ThreadAffinityFailure::Native(error)))?;
    if !observed.stable() || observed.identity() != original.identity() {
        return Err(ThreadAffinityFailureRecord {
            stage,
            failure: ThreadAffinityFailure::OriginChanged,
            observed: Some(observed),
            set_conflict: None,
        });
    }
    Ok(observed)
}

fn checked_set<A: ThreadAffinityAccess>(
    access: &mut A,
    origin_thread: ThreadId,
    original: &ThreadAffinitySnapshot,
    expected: &ThreadAffinityMask,
    target: &ThreadAffinityMask,
    stage: ThreadAffinityStage,
    verify_stage: ThreadAffinityStage,
) -> Result<ThreadAffinitySnapshot, ThreadAffinityFailureRecord> {
    let before = checked_observe(access, origin_thread, original, stage)?;
    if before.mask() != *expected {
        return Err(ThreadAffinityFailureRecord {
            stage,
            failure: ThreadAffinityFailure::MaskChangedExternally,
            observed: Some(before),
            set_conflict: None,
        });
    }
    let previous = access
        .set(target)
        .map_err(|error| failure(stage, ThreadAffinityFailure::Native(error)))?;
    if let Some(previous) = previous.filter(|previous| previous != expected) {
        // Set already changed this thread. Retain the original conflict even
        // if the one bounded attempt to restore its actual previous succeeds.
        let conflict =
            compensate_set_conflict(access, origin_thread, original, expected, &previous, target);
        return Err(ThreadAffinityFailureRecord {
            stage,
            failure: ThreadAffinityFailure::MaskChangedExternally,
            observed: conflict
                .after_compensation
                .clone()
                .or_else(|| conflict.before_compensation.clone()),
            set_conflict: Some(Box::new(conflict)),
        });
    }
    let observed = checked_observe(access, origin_thread, original, verify_stage)?;
    if observed.mask() != *target {
        return Err(ThreadAffinityFailureRecord {
            stage: verify_stage,
            failure: ThreadAffinityFailure::ReadbackMismatch,
            observed: Some(observed),
            set_conflict: None,
        });
    }
    Ok(observed)
}

fn compensate_set_conflict<A: ThreadAffinityAccess>(
    access: &mut A,
    origin_thread: ThreadId,
    original: &ThreadAffinitySnapshot,
    expected: &ThreadAffinityMask,
    previous: &ThreadAffinityMask,
    written: &ThreadAffinityMask,
) -> ThreadAffinitySetConflict {
    let stage = ThreadAffinityStage::Rollback;
    let failed =
        |failure| ThreadAffinityCompensationOutcome::Failed(ThreadAffinityCause { stage, failure });
    let mut conflict = ThreadAffinitySetConflict {
        expected_previous: expected.clone(),
        actual_previous: previous.clone(),
        written: written.clone(),
        before_compensation: None,
        compensation_attempted: false,
        compensation_previous: None,
        compensation_error: None,
        after_compensation: None,
        after_probe_failure: None,
        outcome: ThreadAffinityCompensationOutcome::NotAttempted,
    };
    let before = match checked_observe(access, origin_thread, original, stage) {
        Ok(observed) => observed,
        Err(error) => {
            conflict.before_compensation = error.observed;
            conflict.outcome = ThreadAffinityCompensationOutcome::Failed(ThreadAffinityCause {
                stage: error.stage,
                failure: error.failure,
            });
            return conflict;
        }
    };
    let current = before.mask();
    conflict.before_compensation = Some(before);
    if current == *previous {
        conflict.outcome = ThreadAffinityCompensationOutcome::PreviousAlreadyObserved;
        return conflict;
    }
    // An origin-checked fresh read must still match our just-written target.
    // A newer externally installed value is preserved without any extra Set.
    if current != *written {
        conflict.outcome = failed(ThreadAffinityFailure::MaskChangedExternally);
        return conflict;
    }
    conflict.compensation_attempted = true;
    match access.set(previous) {
        Ok(returned) => conflict.compensation_previous = returned,
        Err(error) => conflict.compensation_error = Some(error),
    }
    // Keep this observation even when compensation failed or its own returned
    // previous shows a second race. There is no recursive compensating write.
    match checked_observe(access, origin_thread, original, stage) {
        Ok(observed) => conflict.after_compensation = Some(observed),
        Err(error) => {
            conflict.after_compensation = error.observed;
            conflict.after_probe_failure = Some(ThreadAffinityCause {
                stage: error.stage,
                failure: error.failure,
            });
        }
    }
    conflict.outcome = if let Some(error) = conflict.compensation_error {
        failed(ThreadAffinityFailure::Native(error))
    } else if conflict
        .compensation_previous
        .as_ref()
        .is_some_and(|returned| returned != written)
    {
        failed(ThreadAffinityFailure::MaskChangedExternally)
    } else if let Some(error) = conflict.after_probe_failure {
        ThreadAffinityCompensationOutcome::Failed(error)
    } else if conflict
        .after_compensation
        .as_ref()
        .is_none_or(|observed| observed.mask() != *previous)
    {
        failed(ThreadAffinityFailure::ReadbackMismatch)
    } else {
        ThreadAffinityCompensationOutcome::PreviousRestored
    };
    conflict
}

impl<A: ThreadAffinityAccess> AffinityOwner<A> {
    // Preserve full failure/rollback observations in a cold, bounded error.
    // These values must not be collapsed to a single OS code to reduce size.
    fn begin(
        mut access: A,
        mut requested: ThreadAffinityMask,
    ) -> Result<Self, Box<ThreadAffinityApplyError>> {
        let invalid = requested.validate().err().or_else(|| {
            (!access.accepts(&requested)).then_some(ThreadAffinityFailure::WrongPlatformMask)
        });
        if let Some(error) = invalid {
            return Err(Box::new(ThreadAffinityApplyError {
                primary: failure(ThreadAffinityStage::Validate, error),
                requested,
                original: None,
                rollback_target: None,
                rollback: ThreadAffinityRollback::NotAttempted,
            }));
        }
        let origin_thread = std::thread::current().id();
        let original = access.observe().map_err(|error| {
            Box::new(ThreadAffinityApplyError {
                primary: failure(
                    ThreadAffinityStage::ObserveOriginal,
                    ThreadAffinityFailure::Native(error),
                ),
                requested: requested.clone(),
                original: None,
                rollback_target: None,
                rollback: ThreadAffinityRollback::NotAttempted,
            })
        })?;
        if !original.stable() || original.identity().0 == 0 || original.identity().1 == 0 {
            return Err(Box::new(ThreadAffinityApplyError {
                primary: failure(
                    ThreadAffinityStage::ObserveOriginal,
                    ThreadAffinityFailure::OriginChanged,
                ),
                requested,
                original: Some(original),
                rollback_target: None,
                rollback: ThreadAffinityRollback::NotAttempted,
            }));
        }
        if !requested.is_subset_of(&original.mask()) {
            return Err(Box::new(ThreadAffinityApplyError {
                primary: failure(
                    ThreadAffinityStage::Validate,
                    ThreadAffinityFailure::OutsideOriginalMask,
                ),
                requested,
                original: Some(original),
                rollback_target: None,
                rollback: ThreadAffinityRollback::NotAttempted,
            }));
        }
        let mut rollback_target = original.mask();
        let mut mutation_attempted = false;
        let outcome = (|| {
            // Recheck immediately before Set. These bounded observations are
            // not an atomic lock against another actor changing OS settings.
            let before = checked_observe(
                &mut access,
                origin_thread,
                &original,
                ThreadAffinityStage::Apply,
            )?;
            if before != original {
                return Err(ThreadAffinityFailureRecord {
                    stage: ThreadAffinityStage::Apply,
                    failure: ThreadAffinityFailure::MaskChangedExternally,
                    observed: Some(before),
                    set_conflict: None,
                });
            }
            mutation_attempted = true;
            if let Some(previous) = access.set(&requested).map_err(|error| {
                failure(
                    ThreadAffinityStage::Apply,
                    ThreadAffinityFailure::Native(error),
                )
            })? {
                if previous != rollback_target {
                    rollback_target = previous;
                    return Err(failure(
                        ThreadAffinityStage::Apply,
                        ThreadAffinityFailure::MaskChangedExternally,
                    ));
                }
            }
            let observed = checked_observe(
                &mut access,
                origin_thread,
                &original,
                ThreadAffinityStage::VerifyApplied,
            )?;
            if observed.mask() != requested {
                return Err(ThreadAffinityFailureRecord {
                    stage: ThreadAffinityStage::VerifyApplied,
                    failure: ThreadAffinityFailure::ReadbackMismatch,
                    observed: Some(observed),
                    set_conflict: None,
                });
            }
            Ok(observed)
        })();
        match outcome {
            Ok(applied) => Ok(Self {
                access,
                origin_thread,
                original,
                applied,
                finished: false,
                first_restore_failure: None,
                _not_send_or_sync: PhantomData,
            }),
            Err(primary) => {
                let rollback = if mutation_attempted {
                    Self::rollback(
                        &mut access,
                        origin_thread,
                        &original,
                        &requested,
                        &rollback_target,
                    )
                } else {
                    ThreadAffinityRollback::NotAttempted
                };
                Err(Box::new(ThreadAffinityApplyError {
                    primary,
                    requested,
                    original: Some(original),
                    rollback_target: Some(rollback_target),
                    rollback,
                }))
            }
        }
    }
    fn rollback(
        access: &mut A,
        origin_thread: ThreadId,
        original: &ThreadAffinitySnapshot,
        requested: &ThreadAffinityMask,
        target: &ThreadAffinityMask,
    ) -> ThreadAffinityRollback {
        let observed = match checked_observe(
            access,
            origin_thread,
            original,
            ThreadAffinityStage::Rollback,
        ) {
            Ok(observed) => observed,
            Err(error) => return ThreadAffinityRollback::Failed(error),
        };
        let actual = observed.mask();
        if actual == *target {
            return ThreadAffinityRollback::OriginalAlreadyObserved(observed);
        }
        // A kernel-restricted subset of the mask just set may be rolled back.
        // An unrelated externally installed mask must never be overwritten.
        if !actual.is_subset_of(requested) {
            return ThreadAffinityRollback::Failed(ThreadAffinityFailureRecord {
                stage: ThreadAffinityStage::Rollback,
                failure: ThreadAffinityFailure::MaskChangedExternally,
                observed: Some(observed),
                set_conflict: None,
            });
        }
        match checked_set(
            access,
            origin_thread,
            original,
            &actual,
            target,
            ThreadAffinityStage::Rollback,
            ThreadAffinityStage::Rollback,
        ) {
            Ok(observed) => ThreadAffinityRollback::Restored(observed),
            Err(error) => ThreadAffinityRollback::Failed(error),
        }
    }
    fn restore(&mut self) -> Result<ThreadAffinitySnapshot, ThreadAffinityFailureRecord> {
        if self.finished {
            return Err(failure(
                ThreadAffinityStage::Restore,
                ThreadAffinityFailure::AlreadyFinished,
            ));
        }
        let restored = checked_set(
            &mut self.access,
            self.origin_thread,
            &self.original,
            &self.applied.mask(),
            &self.original.mask(),
            ThreadAffinityStage::Restore,
            ThreadAffinityStage::VerifyRestored,
        )?;
        self.finished = true;
        Ok(restored)
    }
    fn finish(&mut self) -> Result<ThreadAffinitySnapshot, Box<ThreadAffinityRestoreError>> {
        self.restore().map_err(|failure| {
            self.first_restore_failure
                .get_or_insert_with(|| failure.clone());
            Box::new(ThreadAffinityRestoreError {
                failure,
                original: self.original.clone(),
                applied: self.applied.clone(),
            })
        })
    }
}
impl<A: ThreadAffinityAccess> Drop for AffinityOwner<A> {
    fn drop(&mut self) {
        if !self.finished {
            let _best_effort_only = self.restore();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    enum SetAction {
        Apply,
        Fail(u32),
        Clamp(ThreadAffinityMask),
        Previous(ThreadAffinityMask),
    }
    struct FakeState {
        mask: ThreadAffinityMask,
        tid: u64,
        reads: usize,
        writes: Vec<ThreadAffinityMask>,
        actions: VecDeque<SetAction>,
        fail_read: Option<usize>,
        change_on_read: Option<(usize, ThreadAffinityMask)>,
        change_tid_on_read: Option<(usize, u64)>,
    }
    #[derive(Clone)]
    struct FakeAccess(Rc<RefCell<FakeState>>);
    fn linux(cpus: &[u16]) -> ThreadAffinityMask {
        ThreadAffinityMask::Linux {
            cpu_ids: cpus.to_vec(),
        }
    }
    fn windows(mask: u64) -> ThreadAffinityMask {
        ThreadAffinityMask::Windows { group: 7, mask }
    }
    fn fake(mask: ThreadAffinityMask) -> (FakeAccess, Rc<RefCell<FakeState>>) {
        let state = Rc::new(RefCell::new(FakeState {
            mask,
            tid: 9,
            reads: 0,
            writes: Vec::new(),
            actions: VecDeque::new(),
            fail_read: None,
            change_on_read: None,
            change_tid_on_read: None,
        }));
        (FakeAccess(Rc::clone(&state)), state)
    }
    impl ThreadAffinityAccess for FakeAccess {
        fn accepts(&self, mask: &ThreadAffinityMask) -> bool {
            std::mem::discriminant(mask) == std::mem::discriminant(&self.0.borrow().mask)
        }
        fn observe(&mut self) -> Result<ThreadAffinitySnapshot, PlatformCpuError> {
            let mut state = self.0.borrow_mut();
            state.reads += 1;
            if state.fail_read == Some(state.reads) {
                return Err(PlatformCpuError::ReadUnavailable);
            }
            if let Some((at, mask)) = state.change_on_read.clone() {
                if at == state.reads {
                    state.mask = mask;
                }
            }
            if let Some((at, tid)) = state.change_tid_on_read {
                if at == state.reads {
                    state.tid = tid;
                }
            }
            Ok(match &state.mask {
                ThreadAffinityMask::Linux { cpu_ids } => {
                    ThreadAffinitySnapshot::Linux(LinuxThreadAffinity {
                        process_id: 3,
                        thread_id: state.tid,
                        cpu_ids: cpu_ids.clone(),
                        identity_stable: true,
                    })
                }
                ThreadAffinityMask::Windows { group, mask } => {
                    ThreadAffinitySnapshot::Windows(WindowsThreadAffinity {
                        process_id: 3,
                        thread_id: state.tid,
                        group_mask: (*group, *mask),
                        identity_stable: true,
                    })
                }
            })
        }
        fn set(
            &mut self,
            mask: &ThreadAffinityMask,
        ) -> Result<Option<ThreadAffinityMask>, PlatformCpuError> {
            let mut state = self.0.borrow_mut();
            state.writes.push(mask.clone());
            match state.actions.pop_front().unwrap_or(SetAction::Apply) {
                SetAction::Fail(os_code) => {
                    return Err(PlatformCpuError::NativeCall {
                        function: "fake_set_current",
                        os_code,
                    })
                }
                SetAction::Apply => state.mask = mask.clone(),
                SetAction::Clamp(mask) => state.mask = mask,
                SetAction::Previous(previous) => {
                    state.mask = mask.clone();
                    return Ok(Some(previous));
                }
            }
            Ok(None)
        }
    }

    #[test]
    fn narrowing_and_explicit_finish_preserve_the_exact_original_mask() {
        let original = linux(&[0, 1, 64, 4095]);
        let (access, state) = fake(original.clone());
        let mut owner = AffinityOwner::begin(access, linux(&[4095, 1]))
            .ok()
            .unwrap();
        assert_eq!(owner.original.mask(), original);
        assert_eq!(owner.applied.mask(), linux(&[1, 4095]));
        assert_eq!(owner.finish().unwrap().mask(), original);
        assert!(owner.first_restore_failure.is_none());
        drop(owner);
        assert_eq!(state.borrow().writes, vec![linux(&[1, 4095]), original]);
    }

    #[test]
    fn invalid_empty_duplicate_large_foreign_and_widening_masks_do_not_set() {
        for (request, expected) in [
            (linux(&[]), ThreadAffinityFailure::EmptyMask),
            (linux(&[0, 0]), ThreadAffinityFailure::DuplicateCpu),
            (linux(&[4096]), ThreadAffinityFailure::CpuOutOfRange),
            (linux(&[u16::MAX]), ThreadAffinityFailure::CpuOutOfRange),
            (linux(&[2]), ThreadAffinityFailure::OutsideOriginalMask),
            (
                ThreadAffinityMask::Windows { group: 0, mask: 0 },
                ThreadAffinityFailure::EmptyMask,
            ),
            (
                ThreadAffinityMask::Windows { group: 64, mask: 1 },
                ThreadAffinityFailure::WindowsGroupOutOfRange,
            ),
            (
                ThreadAffinityMask::Windows { group: 0, mask: 1 },
                ThreadAffinityFailure::WrongPlatformMask,
            ),
        ] {
            let (access, state) = fake(linux(&[0, 1]));
            let error = AffinityOwner::begin(access, request).err().unwrap();
            assert_eq!(error.primary.failure, expected);
            assert_eq!(error.rollback, ThreadAffinityRollback::NotAttempted);
            assert!(state.borrow().writes.is_empty());
        }
    }

    #[test]
    fn windows_group_is_preserved_and_a_different_group_is_not_a_subset() {
        let original = ThreadAffinityMask::Windows {
            group: 7,
            mask: 0x8000_0000_0000_0003,
        };
        let (access, state) = fake(original.clone());
        let mut owner =
            AffinityOwner::begin(access, ThreadAffinityMask::Windows { group: 7, mask: 1 })
                .ok()
                .unwrap();
        assert_eq!(owner.finish().unwrap().mask(), original);
        let (access, _) = fake(original);
        let error = AffinityOwner::begin(access, ThreadAffinityMask::Windows { group: 0, mask: 1 })
            .err()
            .unwrap();
        assert_eq!(
            error.primary.failure,
            ThreadAffinityFailure::OutsideOriginalMask
        );
        assert_eq!(state.borrow().writes.len(), 2);
    }

    #[test]
    fn kernel_restriction_is_a_failed_apply_with_verified_original_rollback() {
        let original = linux(&[0, 1, 2]);
        let (access, state) = fake(original.clone());
        state
            .borrow_mut()
            .actions
            .push_back(SetAction::Clamp(linux(&[0])));
        let error = AffinityOwner::begin(access, linux(&[0, 1])).err().unwrap();
        assert_eq!(error.primary.stage, ThreadAffinityStage::VerifyApplied);
        assert_eq!(
            error.primary.failure,
            ThreadAffinityFailure::ReadbackMismatch
        );
        assert!(matches!(
            error.rollback,
            ThreadAffinityRollback::Restored(_)
        ));
        assert_eq!(state.borrow().mask, original);
        assert_eq!(state.borrow().writes.len(), 2);
    }

    #[test]
    fn native_apply_error_and_failed_rollback_preserve_both_original_causes() {
        let (access, state) = fake(linux(&[0, 1, 2]));
        state.borrow_mut().actions.push_back(SetAction::Fail(22));
        let error = AffinityOwner::begin(access, linux(&[0])).err().unwrap();
        assert!(matches!(
            error.primary.failure,
            ThreadAffinityFailure::Native(PlatformCpuError::NativeCall { os_code: 22, .. })
        ));
        assert!(matches!(
            error.rollback,
            ThreadAffinityRollback::OriginalAlreadyObserved(_)
        ));

        let (access, state) = fake(linux(&[0, 1, 2]));
        state
            .borrow_mut()
            .actions
            .extend([SetAction::Clamp(linux(&[0])), SetAction::Fail(5)]);
        let error = AffinityOwner::begin(access, linux(&[0, 1])).err().unwrap();
        assert_eq!(
            error.primary.failure,
            ThreadAffinityFailure::ReadbackMismatch
        );
        assert!(matches!(
            error.rollback,
            ThreadAffinityRollback::Failed(ThreadAffinityFailureRecord {
                failure: ThreadAffinityFailure::Native(PlatformCpuError::NativeCall {
                    os_code: 5,
                    ..
                }),
                ..
            })
        ));
        assert_eq!(error.original.unwrap().mask(), linux(&[0, 1, 2]));
        assert_eq!(state.borrow().mask, linux(&[0]));
    }

    #[test]
    fn post_apply_query_failure_is_not_success_and_rollback_is_separately_observed() {
        let (access, state) = fake(linux(&[0, 1]));
        state.borrow_mut().fail_read = Some(3);
        let error = AffinityOwner::begin(access, linux(&[0])).err().unwrap();
        assert_eq!(error.primary.stage, ThreadAffinityStage::VerifyApplied);
        assert_eq!(
            error.primary.failure,
            ThreadAffinityFailure::Native(PlatformCpuError::ReadUnavailable)
        );
        assert!(matches!(
            error.rollback,
            ThreadAffinityRollback::Restored(_)
        ));
        assert_eq!(state.borrow().mask, linux(&[0, 1]));
    }

    #[test]
    fn pre_apply_external_change_is_preserved_without_a_rollback_write() {
        let (access, state) = fake(linux(&[0, 1, 2]));
        state.borrow_mut().change_on_read = Some((2, linux(&[0])));
        let error = AffinityOwner::begin(access, linux(&[0])).err().unwrap();
        assert_eq!(
            error.primary.failure,
            ThreadAffinityFailure::MaskChangedExternally
        );
        assert_eq!(error.rollback, ThreadAffinityRollback::NotAttempted);
        assert_eq!(state.borrow().mask, linux(&[0]));
        assert!(state.borrow().writes.is_empty());
    }

    #[test]
    fn atomic_windows_previous_mask_is_the_rollback_target_after_a_race() {
        let (access, state) = fake(linux(&[0, 1, 2]));
        state
            .borrow_mut()
            .actions
            .push_back(SetAction::Previous(linux(&[1, 2])));
        let error = AffinityOwner::begin(access, linux(&[0])).err().unwrap();
        assert_eq!(
            error.primary.failure,
            ThreadAffinityFailure::MaskChangedExternally
        );
        assert_eq!(error.rollback_target, Some(linux(&[1, 2])));
        assert!(matches!(
            error.rollback,
            ThreadAffinityRollback::Restored(_)
        ));
        assert_eq!(state.borrow().mask, linux(&[1, 2]));
    }

    #[test]
    fn explicit_restore_error_remains_sticky_after_retry_or_drop_cleanup() {
        let (access, state) = fake(linux(&[0, 1]));
        let mut owner = AffinityOwner::begin(access, linux(&[0])).ok().unwrap();
        state.borrow_mut().actions.push_back(SetAction::Fail(5));
        let explicit = owner.finish().unwrap_err();
        assert_eq!(
            owner.first_restore_failure.as_ref(),
            Some(&explicit.failure)
        );
        owner.finish().unwrap();
        assert_eq!(
            owner.first_restore_failure.as_ref(),
            Some(&explicit.failure)
        );
        assert_eq!(state.borrow().mask, linux(&[0, 1]));

        let (access, state) = fake(linux(&[0, 1]));
        let mut owner = AffinityOwner::begin(access, linux(&[0])).ok().unwrap();
        state.borrow_mut().actions.push_back(SetAction::Fail(5));
        let explicit = owner.finish().unwrap_err();
        drop(owner); // final best effort succeeds, without replacing `explicit`
        assert!(matches!(
            explicit.failure.failure,
            ThreadAffinityFailure::Native(PlatformCpuError::NativeCall { os_code: 5, .. })
        ));
        assert_eq!(state.borrow().mask, linux(&[0, 1]));
    }

    #[test]
    fn a_kernel_restricted_restore_is_an_explicit_failure_even_if_drop_later_recovers() {
        let (access, state) = fake(linux(&[0, 1]));
        let mut owner = AffinityOwner::begin(access, linux(&[0])).ok().unwrap();
        state
            .borrow_mut()
            .actions
            .push_back(SetAction::Clamp(linux(&[0])));
        let explicit = owner.finish().unwrap_err();
        assert_eq!(explicit.failure.stage, ThreadAffinityStage::VerifyRestored);
        assert_eq!(
            explicit.failure.failure,
            ThreadAffinityFailure::ReadbackMismatch
        );
        assert_eq!(explicit.original.mask(), linux(&[0, 1]));
        assert_eq!(
            owner.first_restore_failure.as_ref(),
            Some(&explicit.failure)
        );
        drop(owner);
        assert_eq!(state.borrow().mask, linux(&[0, 1]));
        assert_eq!(
            explicit.failure.failure,
            ThreadAffinityFailure::ReadbackMismatch
        );
    }

    #[test]
    fn restore_rejects_foreign_threads_changed_identity_and_external_masks() {
        let (access, state) = fake(linux(&[0, 1]));
        let mut owner = AffinityOwner::begin(access, linux(&[0])).ok().unwrap();
        owner.origin_thread = std::thread::spawn(|| std::thread::current().id())
            .join()
            .unwrap();
        assert_eq!(
            owner.finish().unwrap_err().failure.failure,
            ThreadAffinityFailure::WrongThread
        );
        drop(owner);
        assert_eq!(state.borrow().writes.len(), 1);

        let (access, state) = fake(linux(&[0, 1]));
        let mut owner = AffinityOwner::begin(access, linux(&[0])).ok().unwrap();
        state.borrow_mut().tid = 10;
        assert_eq!(
            owner.finish().unwrap_err().failure.failure,
            ThreadAffinityFailure::OriginChanged
        );
        drop(owner);
        assert_eq!(state.borrow().writes.len(), 1);

        let (access, state) = fake(linux(&[0, 1]));
        let mut owner = AffinityOwner::begin(access, linux(&[0])).ok().unwrap();
        state.borrow_mut().mask = linux(&[1]);
        assert_eq!(
            owner.finish().unwrap_err().failure.failure,
            ThreadAffinityFailure::MaskChangedExternally
        );
        drop(owner);
        assert_eq!(state.borrow().mask, linux(&[1]));
        assert_eq!(state.borrow().writes.len(), 1);
    }

    #[test]
    fn restore_atomic_previous_race_compensates_but_remains_a_sticky_error() {
        let (access, state) = fake(windows(15));
        let mut owner = AffinityOwner::begin(access, windows(3)).ok().unwrap();
        state.borrow_mut().actions.extend([
            SetAction::Previous(windows(4)),
            SetAction::Previous(windows(15)),
        ]);
        let explicit = owner.finish().unwrap_err();
        assert_eq!(explicit.failure.stage, ThreadAffinityStage::Restore);
        assert_eq!(
            explicit.failure.failure,
            ThreadAffinityFailure::MaskChangedExternally
        );
        let conflict = explicit.failure.set_conflict.as_ref().unwrap();
        assert_eq!(conflict.expected_previous, windows(3));
        assert_eq!(conflict.actual_previous, windows(4));
        assert_eq!(conflict.written, windows(15));
        assert!(conflict.compensation_attempted);
        assert_eq!(conflict.compensation_previous, Some(windows(15)));
        assert_eq!(
            conflict.after_compensation.as_ref().unwrap().mask(),
            windows(4)
        );
        assert_eq!(
            conflict.outcome,
            ThreadAffinityCompensationOutcome::PreviousRestored
        );
        assert!(!owner.finished);
        assert_eq!(
            owner.first_restore_failure.as_ref(),
            Some(&explicit.failure)
        );
        drop(owner);
        assert_eq!(state.borrow().mask, windows(4));
        assert_eq!(
            state.borrow().writes,
            vec![windows(3), windows(15), windows(4)]
        );
    }

    #[test]
    fn rollback_atomic_previous_race_preserves_apply_failure_and_compensation() {
        let (access, state) = fake(windows(15));
        state.borrow_mut().actions.extend([
            SetAction::Clamp(windows(1)),
            SetAction::Previous(windows(4)),
            SetAction::Previous(windows(15)),
        ]);
        let error = AffinityOwner::begin(access, windows(3)).err().unwrap();
        assert_eq!(error.primary.stage, ThreadAffinityStage::VerifyApplied);
        assert_eq!(
            error.primary.failure,
            ThreadAffinityFailure::ReadbackMismatch
        );
        let ThreadAffinityRollback::Failed(rollback) = &error.rollback else {
            panic!("rollback race must remain a failure");
        };
        assert_eq!(rollback.stage, ThreadAffinityStage::Rollback);
        assert_eq!(
            rollback.failure,
            ThreadAffinityFailure::MaskChangedExternally
        );
        let conflict = rollback.set_conflict.as_ref().unwrap();
        assert_eq!(conflict.actual_previous, windows(4));
        assert_eq!(
            conflict.outcome,
            ThreadAffinityCompensationOutcome::PreviousRestored
        );
        assert_eq!(state.borrow().mask, windows(4));
        assert_eq!(state.borrow().writes.len(), 3);
    }

    #[test]
    fn compensation_preserves_a_newer_external_mask_without_another_set() {
        let (access, state) = fake(windows(15));
        let mut owner = AffinityOwner::begin(access, windows(3)).ok().unwrap();
        state
            .borrow_mut()
            .actions
            .push_back(SetAction::Previous(windows(4)));
        // begin reads 1..3, restore pre-check is 4, compensation re-check is 5.
        state.borrow_mut().change_on_read = Some((5, windows(8)));
        let explicit = owner.finish().unwrap_err();
        let conflict = explicit.failure.set_conflict.as_ref().unwrap();
        assert_eq!(
            conflict.before_compensation.as_ref().unwrap().mask(),
            windows(8)
        );
        assert!(!conflict.compensation_attempted);
        assert_eq!(
            conflict.outcome,
            ThreadAffinityCompensationOutcome::Failed(ThreadAffinityCause {
                stage: ThreadAffinityStage::Rollback,
                failure: ThreadAffinityFailure::MaskChangedExternally
            })
        );
        drop(owner);
        assert_eq!(state.borrow().mask, windows(8));
        assert_eq!(state.borrow().writes.len(), 2);
    }

    #[test]
    fn compensation_rechecks_the_original_os_thread_before_another_set() {
        let (access, state) = fake(windows(15));
        let mut owner = AffinityOwner::begin(access, windows(3)).ok().unwrap();
        state
            .borrow_mut()
            .actions
            .push_back(SetAction::Previous(windows(4)));
        state.borrow_mut().change_tid_on_read = Some((5, 10));
        let explicit = owner.finish().unwrap_err();
        let conflict = explicit.failure.set_conflict.as_ref().unwrap();
        assert!(!conflict.compensation_attempted);
        assert_eq!(
            conflict.outcome,
            ThreadAffinityCompensationOutcome::Failed(ThreadAffinityCause {
                stage: ThreadAffinityStage::Rollback,
                failure: ThreadAffinityFailure::OriginChanged
            })
        );
        assert_eq!(
            conflict.before_compensation.as_ref().unwrap().identity(),
            (3, 10)
        );
        drop(owner);
        assert_eq!(state.borrow().writes.len(), 2);
    }

    #[test]
    fn compensation_native_and_readback_failures_do_not_replace_the_restore_race() {
        let (access, state) = fake(windows(15));
        let mut owner = AffinityOwner::begin(access, windows(3)).ok().unwrap();
        state
            .borrow_mut()
            .actions
            .extend([SetAction::Previous(windows(4)), SetAction::Fail(5)]);
        state.borrow_mut().fail_read = Some(6);
        let explicit = owner.finish().unwrap_err();
        assert_eq!(explicit.failure.stage, ThreadAffinityStage::Restore);
        assert_eq!(
            explicit.failure.failure,
            ThreadAffinityFailure::MaskChangedExternally
        );
        let conflict = explicit.failure.set_conflict.as_ref().unwrap();
        assert_eq!(conflict.actual_previous, windows(4));
        assert!(matches!(
            conflict.compensation_error,
            Some(PlatformCpuError::NativeCall { os_code: 5, .. })
        ));
        assert_eq!(
            conflict.after_probe_failure,
            Some(ThreadAffinityCause {
                stage: ThreadAffinityStage::Rollback,
                failure: ThreadAffinityFailure::Native(PlatformCpuError::ReadUnavailable)
            })
        );
        assert!(matches!(
            conflict.outcome,
            ThreadAffinityCompensationOutcome::Failed(ThreadAffinityCause {
                failure: ThreadAffinityFailure::Native(PlatformCpuError::NativeCall {
                    os_code: 5,
                    ..
                }),
                ..
            })
        ));
        assert!(!owner.finished);
        drop(owner);
        assert_eq!(state.borrow().writes.len(), 3);
    }

    #[test]
    fn compensation_previous_race_is_recorded_without_recursive_recovery() {
        let (access, state) = fake(windows(15));
        let mut owner = AffinityOwner::begin(access, windows(3)).ok().unwrap();
        state.borrow_mut().actions.extend([
            SetAction::Previous(windows(4)),
            SetAction::Previous(windows(8)),
        ]);
        let explicit = owner.finish().unwrap_err();
        let conflict = explicit.failure.set_conflict.as_ref().unwrap();
        assert_eq!(conflict.actual_previous, windows(4));
        assert_eq!(conflict.compensation_previous, Some(windows(8)));
        assert_eq!(
            conflict.after_compensation.as_ref().unwrap().mask(),
            windows(4)
        );
        assert_eq!(
            conflict.outcome,
            ThreadAffinityCompensationOutcome::Failed(ThreadAffinityCause {
                stage: ThreadAffinityStage::Rollback,
                failure: ThreadAffinityFailure::MaskChangedExternally
            })
        );
        assert!(!owner.finished);
        drop(owner);
        assert_eq!(state.borrow().writes.len(), 3);
    }

    #[test]
    fn actual_current_thread_round_trip_or_explicit_ambiguous_group_rejection() {
        let mut native = NativeThreadAffinity;
        let before = match native.observe() {
            Ok(observed) => observed,
            Err(PlatformCpuError::AmbiguousProcessGroups) => {
                let error = apply_current_thread_affinity(ThreadAffinityMask::Windows {
                    group: 0,
                    mask: 1,
                })
                .err()
                .unwrap();
                assert_eq!(
                    error.primary.failure,
                    ThreadAffinityFailure::Native(PlatformCpuError::AmbiguousProcessGroups)
                );
                assert_eq!(error.rollback, ThreadAffinityRollback::NotAttempted);
                return;
            }
            Err(error) => panic!("current-thread affinity observation failed: {error:?}"),
        };
        let narrowed = match before.mask() {
            ThreadAffinityMask::Linux { cpu_ids } => linux(&[cpu_ids[0]]),
            ThreadAffinityMask::Windows { group, mask } => ThreadAffinityMask::Windows {
                group,
                mask: 1u64 << mask.trailing_zeros(),
            },
        };
        let mut owner = apply_current_thread_affinity(narrowed.clone()).unwrap();
        assert_eq!(owner.original(), &before);
        assert_eq!(owner.applied().mask(), narrowed);
        assert_eq!(native.observe().unwrap(), *owner.applied());
        assert_eq!(owner.finish().unwrap(), before);
        assert_eq!(native.observe().unwrap(), before);
    }
}
