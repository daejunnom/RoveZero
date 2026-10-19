//! Staged product ownership only. No setup capability leaves this module, no
//! worker/Store is configured, and registration never means archive-on. The
//! native close flag is sufficient to release these unused owners only while
//! this module cannot issue setup; future worker integration must also join
//! that actual worker before releasing registration custody.

use super::RoventNativeProductDriver;
use crate::engine::ProcessClock;
use crate::rovent_bootstrap::launch::storage::{
    RoventStorageAdmissionOwners, RoventStorageCompiledCustody,
    RoventStorageOwnedRuntimeRegistration, RoventStorageRuntimeOwnershipFailure,
    RoventStorageRuntimeRegistrationFailure, own_rovent_storage_runtime_registration,
    register_rovent_storage_runtime,
};
use crate::rovent_bootstrap::process_boundary::VerifiedOwnedRoventProcessBoundary;
use crate::search_driver::{DriverPhysicalCompletion, SearchSessionFailure};
use rz_runtime::contracts::ContractClock;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Move-only custody. Unknown Drop deliberately retains the original value;
/// neither Drop nor a diagnostic can mark the actual product physically closed.
pub(super) struct RetainedOnUnknown<T> {
    original: Option<T>,
    physically_closed: Arc<AtomicBool>,
}
impl<T> RetainedOnUnknown<T> {
    fn new(original: T, physically_closed: Arc<AtomicBool>) -> Self {
        Self {
            original: Some(original),
            physically_closed,
        }
    }
    fn original(&self) -> &T {
        self.original.as_ref().expect("original storage owner")
    }
}
impl<T> Drop for RetainedOnUnknown<T> {
    fn drop(&mut self) {
        if !self.physically_closed.load(Ordering::Acquire) {
            if let Some(original) = self.original.take() {
                std::mem::forget(original);
            }
        }
    }
}
pub(super) type RetainedStorageRegistration =
    RetainedOnUnknown<RoventStorageOwnedRuntimeRegistration>;

// Preserve the exact partial proof/owners from each original API. No borrowed
// failure is decoded into a replacement registration or new FD capability.
#[allow(clippy::large_enum_variant)]
enum RejectedStorageOwners<'boundary, 'process> {
    Pending {
        custody: RoventStorageCompiledCustody<'boundary, 'process>,
        boundary: Arc<VerifiedOwnedRoventProcessBoundary>,
    },
    Registration {
        rejected: RoventStorageRuntimeRegistrationFailure<'boundary, 'process>,
        boundary: Arc<VerifiedOwnedRoventProcessBoundary>,
    },
    Ownership(RoventStorageRuntimeOwnershipFailure<'boundary, 'process>),
}

/// A rejection returns the same original partial custody and owned process
/// boundary. It shares only the real product's physical-close observation.
/// Dropping it before actual native closure quarantines those owners as well.
pub struct RoventProductStorageStartupRejection<'boundary, 'process> {
    pub failure: SearchSessionFailure,
    retained: RetainedOnUnknown<RejectedStorageOwners<'boundary, 'process>>,
}
impl RoventProductStorageStartupRejection<'_, '_> {
    pub fn owners(&self) -> &RoventStorageAdmissionOwners {
        match self.retained.original() {
            RejectedStorageOwners::Pending { custody, .. } => custody.owners(),
            RejectedStorageOwners::Registration { rejected, .. } => rejected.custody.owners(),
            RejectedStorageOwners::Ownership(rejected) => rejected.registration.owners(),
        }
    }
    pub fn process_boundary(&self) -> &Arc<VerifiedOwnedRoventProcessBoundary> {
        match self.retained.original() {
            RejectedStorageOwners::Pending { boundary, .. }
            | RejectedStorageOwners::Registration { boundary, .. } => boundary,
            RejectedStorageOwners::Ownership(rejected) => &rejected.candidate,
        }
    }
}

fn same_original_lock<T>(source: &T, storage: &Arc<T>, process: &Arc<T>) -> bool {
    std::ptr::eq(source, storage.as_ref()) && Arc::ptr_eq(storage, process)
}

fn validate_attachment_selection(
    already_installed: bool,
    physically_closed: bool,
    same_lock_owner: bool,
    actual_clock: &ProcessClock,
    requested_clock: &ProcessClock,
) -> Result<(), &'static str> {
    if already_installed {
        return Err("storage runtime registration is already installed");
    }
    if physically_closed {
        return Err("product is already physically closed");
    }
    if !same_lock_owner {
        return Err("product, storage and process do not retain the same original lock object");
    }
    let actual = actual_clock.runtime_clock();
    let requested = requested_clock.runtime_clock();
    if actual.domain() != requested.domain() || actual.origin() != requested.origin() {
        return Err("storage registration clock differs from the original product clock");
    }
    Ok(())
}

impl RoventNativeProductDriver {
    /// Explicit move-only ownership installation, not archive execution. The
    /// same prepared source Arc feeds this product and its actual inner driver.
    /// A caller cannot choose a replacement implementation/clock from a receipt:
    /// register uses only this product's original constructor clock and actual
    /// inner driver's implementation. No Store/worker/setup is started here.
    #[allow(clippy::result_large_err)]
    pub fn attach_storage_runtime_registration<'boundary, 'process>(
        &mut self,
        custody: RoventStorageCompiledCustody<'boundary, 'process>,
        boundary: Arc<VerifiedOwnedRoventProcessBoundary>,
        requested_clock: &ProcessClock,
    ) -> Result<(), RoventProductStorageStartupRejection<'boundary, 'process>> {
        if let Err(reason) = validate_attachment_selection(
            self.storage_registration.is_some(),
            self.physically_closed.load(Ordering::Acquire),
            same_original_lock(
                self.sources.lock(),
                custody.owners().input_lock(),
                boundary.input_lock(),
            ),
            &self.process_clock,
            requested_clock,
        ) {
            return Err(
                self.reject_storage(reason, RejectedStorageOwners::Pending { custody, boundary })
            );
        }
        let registration = match register_rovent_storage_runtime(
            custody,
            self.process_clock.clone(),
            self.actual().implementation(),
        ) {
            Ok(registration) => registration,
            Err(rejected) => {
                return Err(self.reject_storage(
                    rejected.error.to_string(),
                    RejectedStorageOwners::Registration { rejected, boundary },
                ));
            }
        };
        match own_rovent_storage_runtime_registration(registration, boundary) {
            Ok(registration) => {
                self.storage_registration = Some(RetainedOnUnknown::new(
                    registration,
                    Arc::clone(&self.physically_closed),
                ));
                Ok(())
            }
            Err(rejected) => Err(self.reject_storage(
                rejected.error.to_string(),
                RejectedStorageOwners::Ownership(rejected),
            )),
        }
    }

    fn reject_storage<'boundary, 'process>(
        &self,
        detail: impl AsRef<str>,
        owners: RejectedStorageOwners<'boundary, 'process>,
    ) -> RoventProductStorageStartupRejection<'boundary, 'process> {
        RoventProductStorageStartupRejection {
            failure: SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Unknown,
                code: "RoventStorageRuntimeRegistration",
                detail: detail.as_ref().chars().take(512).collect(),
            },
            retained: RetainedOnUnknown::new(owners, Arc::clone(&self.physically_closed)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::ProcessEpoch;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn lock_identity_rejects_equal_values_in_another_original_object() {
        // Pointer comparison only, no fake process/storage proof is issued.
        let original = Arc::new(7_u8);
        let alias = Arc::clone(&original);
        let same_bytes = Arc::new(*original);
        assert!(same_original_lock(original.as_ref(), &alias, &original));
        assert!(!same_original_lock(same_bytes.as_ref(), &alias, &original));
        assert!(!same_original_lock(
            original.as_ref(),
            &same_bytes,
            &original
        ));
        assert!(!same_original_lock(original.as_ref(), &alias, &same_bytes));
    }

    #[test]
    fn attach_selection_rejects_wrong_clock_closed_product_and_duplicate_installation() {
        let original = ProcessClock::new(ProcessEpoch(401));
        let same = original.clone();
        let wrong = ProcessClock::new(ProcessEpoch(402));
        assert!(validate_attachment_selection(false, false, true, &original, &same).is_ok());
        for (installed, closed, same_owner, clock) in [
            (true, false, true, &same),
            (false, true, true, &same),
            (false, false, false, &same),
            (false, false, true, &wrong),
        ] {
            assert!(
                validate_attachment_selection(installed, closed, same_owner, &original, clock)
                    .is_err()
            );
        }
        let same_epoch_new_origin = ProcessClock::new(original.epoch());
        if original.runtime_clock().origin() != same_epoch_new_origin.runtime_clock().origin() {
            assert!(
                validate_attachment_selection(
                    false,
                    false,
                    true,
                    &original,
                    &same_epoch_new_origin
                )
                .is_err()
            );
        }
    }

    struct DropProbe(Arc<AtomicUsize>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[test]
    fn unknown_drop_preserves_original_custody_without_setting_a_completion_flag() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let closed = Arc::new(AtomicBool::new(false));
        let held = RetainedOnUnknown::new(DropProbe(Arc::clone(&dropped)), Arc::clone(&closed));
        drop(held);
        assert_eq!(dropped.load(Ordering::Acquire), 0);
        assert!(!closed.load(Ordering::Acquire));
        assert_eq!(Arc::strong_count(&dropped), 2);
    }

    #[test]
    fn unused_custody_is_released_only_after_the_observed_native_close() {
        // Local holder policy only: this fixture does not issue an actual
        // DriverPhysicalCompletion or storage-worker join receipt.
        let dropped = Arc::new(AtomicUsize::new(0));
        let closed = Arc::new(AtomicBool::new(false));
        let held = RetainedOnUnknown::new(DropProbe(Arc::clone(&dropped)), Arc::clone(&closed));
        closed.store(true, Ordering::Release);
        drop(held);
        assert_eq!(dropped.load(Ordering::Acquire), 1);
        assert_eq!(Arc::strong_count(&dropped), 1);
    }
}
