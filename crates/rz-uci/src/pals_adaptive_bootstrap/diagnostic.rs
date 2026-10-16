//! Explicit observer seam for actual V-free Deployment diagnostics.
//!
//! This adds no Collection, V, teacher, resource or restored-owner authority.
//! Original source/content/binary admission, native factory, startup clock and
//! cancellation, root fences and process cleanup remain the normal owners.
//! It is a preparation seam, not an implemented Y1/T1/R1 runner or acceptance.
//! Missing case events must remain NOT_OBSERVED in a future bounded consumer.

use super::{
    ActualAdaptiveBackendV5, AdaptiveBootstrapError, DeploymentDiagnosticHooksV5,
    PreparedAdaptiveProcessV5, boundary, failure, prepare_actual, verify_actual_process_boundary,
};
use crate::EngineIdentity;
use crate::engine::{OwnerRegistry, ProcessClock};
use crate::pals_adaptive::{
    AdaptiveCollectionDispatchObserver, AdaptiveCompletedRootObserver, AdaptiveManagerError,
    AdaptiveRootAdmissionObserver, AdaptiveSessionManager,
};
use crate::pals_adaptive_startup::PinnedPalsStartupV5;
use crate::pals_native::ticketed::VirtualReviewCollectionObserver;
use crate::search_driver::SearchSessionDriver;
use rz_contracts::ProcessEpoch;
use rz_experiments::{PalsEngineModeV5, PalsViewSelectionV5};
use rz_search::pals::engine::PalsCompletion;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Instant;

/// Runs before root Rules/ranking on the original Search owner. The callback
/// can read the manager; it receives no mutable manager/control authority.
pub type DiagnosticRootAdmissionObserverV5 = Arc<
    dyn Fn(&AdaptiveSessionManager<ActualAdaptiveBackendV5>) -> Result<(), AdaptiveManagerError>
        + Send
        + Sync,
>;
/// Runs after the real root fence with the actual semantic result's completion.
/// A copied callback record cannot reconstruct a ticket, seed or root owner.
pub type DiagnosticRootCompletionObserverV5 = Arc<
    dyn Fn(
            &AdaptiveSessionManager<ActualAdaptiveBackendV5>,
            PalsCompletion,
        ) -> Result<(), AdaptiveManagerError>
        + Send
        + Sync,
>;

/// Callbacks reuse the existing immutable dispatch and original native seams.
/// Native callbacks execute on the native owner; root/dispatch callbacks run on
/// Search. They may only copy/enqueue bounded records, never block on I/O, run a
/// model, accept a ticket, synthesize work, or interpret preparation as execution.
/// A consumer must register its caps and fail closed on overflow or missing data.
pub struct DeploymentDiagnosticObserversV5 {
    pub native_observer: Box<dyn VirtualReviewCollectionObserver>,
    pub dispatch_observer: AdaptiveCollectionDispatchObserver,
    pub root_admission: DiagnosticRootAdmissionObserverV5,
    pub root_completion: DiagnosticRootCompletionObserverV5,
}

/// Prepare one real source-pinned V-free Deployment process for an in-process
/// diagnostic. The running diagnostic image must itself be the registered
/// endpoint binary. Changing the binary to an example requires a new real build
/// and registration; an existing rz-pals-adaptive image pin cannot be relabelled.
///
/// No source/asset/provider/domain is rewritten, no preflight/mock substitution
/// exists, and no fresh cancellation or later deadline replaces this owner.
/// PreparedAdaptiveProcessV5 context, driver/run and finish/retire are still
/// the original opaque owners. This function does not schedule Y1/T1/R1, expose
/// the private native service, change candidate order, or claim semantic consume.
pub fn prepare_deployment_diagnostic_actual_with_cancellation(
    pinned: Arc<PinnedPalsStartupV5>,
    epoch: ProcessEpoch,
    observers: DeploymentDiagnosticObserversV5,
    until: Instant,
    startup_cancel: Arc<AtomicBool>,
) -> Result<PreparedAdaptiveProcessV5, AdaptiveBootstrapError> {
    let started = Instant::now();
    let endpoint = &pinned.config.endpoint;
    if started >= until
        || startup_cancel.load(std::sync::atomic::Ordering::Acquire)
        || epoch.0 == 0
        || pinned.metadata_preflight
        || pinned.config.training_executed
        || endpoint.mode != PalsEngineModeV5::Deployment
        || endpoint.context.selection != PalsViewSelectionV5::VirtualReview
        || endpoint.v_generate
        || endpoint.v_load
        || endpoint.v_call
    {
        return Err(boundary("DeploymentDiagnosticActualAdmission"));
    }
    let binary = verify_actual_process_boundary(&pinned)?;
    let clock = ProcessClock::new(epoch);
    let admission = observers.root_admission;
    let completion = observers.root_completion;
    let root_admission: AdaptiveRootAdmissionObserver<ActualAdaptiveBackendV5> =
        Arc::new(move |manager| admission(manager));
    let root_closure: AdaptiveCompletedRootObserver<ActualAdaptiveBackendV5> =
        Arc::new(move |manager, actual| completion(manager, actual));
    let identity = EngineIdentity {
        name: format!("RoveZero PALS diagnostic {}", endpoint.id),
        author: "RoveZero contributors".into(),
    };
    let prepared = prepare_actual(
        pinned,
        clock,
        Arc::new(OwnerRegistry::default()),
        identity,
        binary,
        started,
        Some(until),
        None,
        Some(DeploymentDiagnosticHooksV5 {
            native_observer: observers.native_observer,
            dispatch_observer: observers.dispatch_observer,
            root_admission,
            root_closure,
        }),
        startup_cancel,
    )?;
    let published = prepared
        .driver
        .start_checker(
            prepared.startup_deadline,
            Arc::clone(&prepared.startup_cancel),
        )
        .map_err(|error| failure("CheckerStartupFence", &error))
        .and_then(|_| {
            prepared
                .publish_startup_resources()
                .map_err(|error| failure("StartupReceipt", &error))
        });
    if let Err(error) = published {
        if let Err(cleanup) = prepared.finish_process(until) {
            eprintln!(
                "pals_v5_diagnostic_startup_cleanup {}",
                failure("ProcessDrain", &cleanup)
            );
        }
        return Err(error);
    }
    Ok(prepared)
}
