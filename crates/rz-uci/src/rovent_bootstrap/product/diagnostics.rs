//! Bounded operational observations of the existing product owners. These
//! values are deliberately Serialize-only: no decoder can reconstruct a root,
//! fence, actor, acceptance permit or physical-completion authority from them.
use super::{ActualProductDriver, RoventNativeProductDriver};
use crate::pals_adaptive::{
    AdaptiveManagerCounters, AdaptiveShutdownReceipt, RoleOnlyShutdownReceipt,
};
use crate::rovent_root_admission::RoventRootAdmission;
use crate::search_driver::{DriverPhysicalCompletion, SearchSessionContext, SearchSessionFailure};
use rz_contracts::Digest;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::rovent::ExecutionMode;
use rz_search::cpu_value::nnuec::v2::review::PreparationStatus;
use rz_search::pals::rovent_frontier::{FrontierGoalStatus, FrontierSnapshot, FrontierStop};
use serde::Serialize;
use std::io::{self, Write};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

pub const ROVENT_LIFECYCLE_DIAGNOSTIC_DOMAIN: &str = "rz-rovent-product-lifecycle-observation/1";
// Operational output limits, independent of search/node/role credit. Keep one
// final row available even after an incomplete root stream reaches its bound.
const MAX_ROOT_ROWS: u64 = 256;
const MAX_ROW_BYTES: usize = 8192;
const MAX_STREAM_BYTES: usize = 1 << 20;

#[derive(Clone, Debug, Serialize)]
struct AuthorityObservation {
    epoch: u64,
    game: u64,
    root: u64,
    implementation_sha256: String,
}
impl From<SearchAuthority> for AuthorityObservation {
    fn from(value: SearchAuthority) -> Self {
        Self {
            epoch: value.epoch.0,
            game: value.game.0,
            root: value.root.0,
            implementation_sha256: hex(value.implementation),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct FailureObservation {
    code: String,
    code_truncated: bool,
    detail: String,
    detail_truncated: bool,
    physical_completion: &'static str,
}
impl From<&SearchSessionFailure> for FailureObservation {
    fn from(value: &SearchSessionFailure) -> Self {
        Self {
            code: bounded(value.code, 128),
            code_truncated: value.code.len() > 128,
            detail: bounded(&value.detail, 512),
            detail_truncated: value.detail.len() > 512,
            physical_completion: physical(value.physical_completion),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct CountersObservation {
    submitted_role_tickets: u64,
    physical_role_completions: u64,
    accepted_nn_inputs: Option<u64>,
    consumed_role_outputs: u64,
    completed_nn_inputs: Option<u64>,
    consumed_nn_inputs: Option<u64>,
    unused_role_results: u64,
    cpu_slice_receipts: u64,
    validator_ticket_admissions: u64,
}
impl From<&AdaptiveManagerCounters> for CountersObservation {
    fn from(value: &AdaptiveManagerCounters) -> Self {
        Self {
            submitted_role_tickets: value.submitted_role_tickets,
            physical_role_completions: value.physical_role_completions,
            accepted_nn_inputs: (value.accepted_nn_input_count_unknown == 0)
                .then_some(value.accepted_nn_inputs),
            consumed_role_outputs: value.consumed_role_outputs,
            completed_nn_inputs: (value.nn_input_count_unknown == 0)
                .then_some(value.completed_nn_inputs),
            consumed_nn_inputs: (value.consumed_nn_input_count_unknown == 0)
                .then_some(value.consumed_nn_inputs),
            unused_role_results: value.unused_role_results,
            cpu_slice_receipts: value.cpu_slice_receipts,
            validator_ticket_admissions: value.validator_ticket_admissions,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct RootPhysicalObservation {
    source: &'static str,
    semantic_closure_confirmed: bool,
    native_root_fence_present: Option<bool>,
    native_root_fence_reusable: bool,
    native_process_joined: bool,
    native_physical_shutdown_confirmed: bool,
    pending_tickets: usize,
    retained_outputs: Option<usize>,
    quarantined: bool,
    own_cpu_joined: Option<usize>,
    expected_own_cpu_owners: Option<usize>,
    own_cpu_panicked: Option<usize>,
    native_cpu_reserved: Option<u64>,
    shared_nnuec_closure_present: Option<bool>,
    shared_nnuec_issued_requests: Option<usize>,
    shared_nnuec_submitted_requests: Option<usize>,
    shared_nnuec_completed_requests: Option<usize>,
    shared_nnuec_cancelled_requests: Option<usize>,
    shared_nnuec_expired_requests: Option<usize>,
    shared_nnuec_failed_requests: Option<usize>,
    shared_nnuec_semantic_consumed_requests: Option<usize>,
}
impl RootPhysicalObservation {
    fn pc(value: &RoleOnlyShutdownReceipt) -> Self {
        let closure = value.shared_nnuec_closure();
        Self {
            source: "original_role_only_shutdown_receipt",
            semantic_closure_confirmed: value.permits_semantic_closure(),
            native_root_fence_present: Some(value.root_fence().is_some()),
            native_root_fence_reusable: value
                .root_fence()
                .is_some_and(|fence| fence.authority() == value.authority && fence.permits_reuse()),
            native_process_joined: value.native_process_joined,
            native_physical_shutdown_confirmed: value.native_physical_shutdown_confirmed,
            pending_tickets: value.pending_tickets,
            retained_outputs: Some(value.retained_outputs),
            quarantined: value.quarantined,
            own_cpu_joined: None,
            expected_own_cpu_owners: None,
            own_cpu_panicked: None,
            native_cpu_reserved: None,
            shared_nnuec_closure_present: Some(closure.is_some()),
            shared_nnuec_issued_requests: closure.map(|c| c.issued_requests()),
            shared_nnuec_submitted_requests: closure.map(|c| c.submitted_requests()),
            shared_nnuec_completed_requests: closure
                .map(|c| c.terminal_status_count(PreparationStatus::Completed)),
            shared_nnuec_cancelled_requests: closure
                .map(|c| c.terminal_status_count(PreparationStatus::Cancelled)),
            shared_nnuec_expired_requests: closure
                .map(|c| c.terminal_status_count(PreparationStatus::Expired)),
            shared_nnuec_failed_requests: closure
                .map(|c| c.terminal_status_count(PreparationStatus::Failed)),
            shared_nnuec_semantic_consumed_requests: closure
                .map(|c| c.semantic_consumed_requests()),
        }
    }
    fn pcr(value: &AdaptiveShutdownReceipt) -> Self {
        // The original predicate validates its private fence. No new fence is
        // inferred from thread counts or from a successful protocol write.
        Self {
            source: "original_adaptive_shutdown_receipt",
            semantic_closure_confirmed: value.permits_semantic_closure(),
            native_root_fence_present: None,
            native_root_fence_reusable: value.permits_root_reuse(),
            native_process_joined: value.native_owner_joined,
            native_physical_shutdown_confirmed: value.native_physical_shutdown_confirmed,
            pending_tickets: value.pending_tickets,
            retained_outputs: None,
            quarantined: value.quarantined,
            own_cpu_joined: Some(value.own_cpu_joined),
            expected_own_cpu_owners: Some(value.expected_own_cpu_owners),
            own_cpu_panicked: Some(value.own_cpu_panicked),
            native_cpu_reserved: Some(value.native_cpu_reserved),
            shared_nnuec_closure_present: None,
            shared_nnuec_issued_requests: None,
            shared_nnuec_submitted_requests: None,
            shared_nnuec_completed_requests: None,
            shared_nnuec_cancelled_requests: None,
            shared_nnuec_expired_requests: None,
            shared_nnuec_failed_requests: None,
            shared_nnuec_semantic_consumed_requests: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct SharedNnuecObservation {
    spawned_runtime_helpers: u8,
    worker_thread_id_observed: bool,
    worker_joined: bool,
    // No per-thread affinity readback is manufactured from the process quota.
    worker_thread_affinity_known: bool,
}

#[derive(Clone, Debug, Serialize)]
struct InitialObservation {
    receipt_domain: &'static str,
    source_seal_sha256: String,
    previous_clean_root_present: Option<bool>,
    used_original_startup_ready: Option<bool>,
    stopped_before_research_execution: Option<bool>,
    deadline_before_frontier: Option<bool>,
    original_nnuec_retirement_present: bool,
    original_researcher_release_present: Option<bool>,
    original_cache_release_epoch: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
struct RootObservation {
    receipt_domain: &'static str,
    root_admission_sha256: String,
    root_budget_sha256: String,
    deadline_seal_sha256: String,
    state_and_history_sha256: String,
    goal_status: &'static str,
    completed_depth: Option<u16>,
    maximum_examined_ply: u16,
    native_usage_known: bool,
    native_nodes: Option<u64>,
    native_qsearch_nodes: Option<u64>,
    reserved_native_nodes: u64,
    outstanding_native_executions: usize,
    counters: CountersObservation,
    physical: Option<RootPhysicalObservation>,
    // Only the concrete RoleOnly driver can supply this structural observation.
    // None means not observed/not applicable, never an inferred zero.
    r_free_confirmed: Option<bool>,
    shared_nnuec: Option<SharedNnuecObservation>,
}
impl RootObservation {
    fn new(
        domain: &'static str,
        admission: &RoventRootAdmission,
        frontier: &FrontierSnapshot,
        counters: &AdaptiveManagerCounters,
        physical: Option<RootPhysicalObservation>,
    ) -> Self {
        Self {
            receipt_domain: domain,
            root_admission_sha256: hex(admission.digest()),
            root_budget_sha256: hex(admission.root_budget_digest()),
            deadline_seal_sha256: hex(admission.deadline_seal_digest()),
            state_and_history_sha256: hex(admission.state_and_history_digest()),
            goal_status: goal(frontier.goal),
            completed_depth: frontier.horizon.completed_depth.map(|depth| depth.0),
            maximum_examined_ply: frontier.horizon.maximum_examined_ply.0,
            native_usage_known: frontier.work.native_usage_known,
            native_nodes: frontier
                .work
                .native_usage_known
                .then_some(frontier.work.native_nodes),
            native_qsearch_nodes: frontier
                .work
                .native_usage_known
                .then_some(frontier.work.native_qsearch_nodes),
            reserved_native_nodes: frontier.work.reserved_native_nodes,
            outstanding_native_executions: frontier.work.outstanding_native_executions,
            counters: counters.into(),
            physical,
            r_free_confirmed: None,
            shared_nnuec: None,
        }
    }
}

/// An observational copy only. Its private fields have no parser or conversion
/// into live search authority. A root fence and whole-session finish stay apart.
/// `diagnostics_complete` describes coverage before this row is written. It is
/// not a delivery acknowledgement: consumers must also observe successful
/// process termination and both stream EOFs, since the final flush can fail.
#[derive(Clone, Debug, Serialize)]
pub struct RoventProductLifecycleObservation {
    domain: &'static str,
    event: &'static str,
    sequence: u64,
    input_lock_sha256: String,
    implementation_sha256: String,
    mode: &'static str,
    authority: Option<AuthorityObservation>,
    source_status: &'static str,
    root: Option<RootObservation>,
    initial: Option<InitialObservation>,
    run_failure: Option<FailureObservation>,
    finish_observed: bool,
    finish_result: &'static str,
    finish_failure: Option<FailureObservation>,
    shared_nnuec_session: Option<SharedNnuecObservation>,
    whole_session_physical: &'static str,
    protocol_succeeded: Option<bool>,
    root_runs_observed: u64,
    root_run_error_results: u64,
    root_rows_emitted: u64,
    diagnostics_complete: bool,
    diagnostic_failure: Option<&'static str>,
}

#[derive(Default)]
struct State {
    root_runs: u64,
    root_run_errors: u64,
    root_rows: u64,
    bytes: usize,
    finish_observed: bool,
    finish_result: Option<&'static str>,
    finish_failure: Option<FailureObservation>,
    shared_nnuec_session: Option<SharedNnuecObservation>,
    first_issue: Option<&'static str>,
    latest: Option<RoventProductLifecycleObservation>,
}

/// A read-only handle to this product's diagnostic state. Keeping the handle
/// does not retain a ReviewView, worker, model or shutdown capability.
#[derive(Clone)]
pub struct RoventLifecycleDiagnostics {
    state: Arc<Mutex<State>>,
    poisoned: Arc<AtomicBool>,
    physical_close: Arc<AtomicBool>,
    lock: Digest,
    implementation: Digest,
    mode: ExecutionMode,
}
impl RoventLifecycleDiagnostics {
    pub(super) fn new_from_sources(
        lock: Digest,
        implementation: Digest,
        mode: ExecutionMode,
        physical_close: Arc<AtomicBool>,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(State::default())),
            poisoned: Arc::new(AtomicBool::new(false)),
            physical_close,
            lock,
            implementation,
            mode,
        }
    }
    /// Last attempted diagnostic row, not a new live root or shutdown receipt.
    pub fn latest_observation(
        &self,
    ) -> Result<Option<RoventProductLifecycleObservation>, &'static str> {
        self.state
            .lock()
            .map(|state| state.latest.clone())
            .map_err(|_| "diagnostic_state_poisoned")
    }
    pub(super) fn observe_root(
        &self,
        product: &RoventNativeProductDriver,
        context: &SearchSessionContext,
        failure: Option<&SearchSessionFailure>,
    ) {
        let (mut source_status, root) = project_root(product, context);
        let initial = project_initial(product, context);
        if root.is_none() && initial.is_some() {
            source_status = "original_initial_stop";
        }
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return;
        };
        state.root_runs = state.root_runs.saturating_add(1);
        if failure.is_some() {
            state.root_run_errors = state.root_run_errors.saturating_add(1);
        }
        if state.root_runs == u64::MAX {
            state.first_issue.get_or_insert("root_sequence_overflow");
        }
        if !matches!(
            source_status,
            "original_current_root" | "original_initial_stop"
        ) {
            state
                .first_issue
                .get_or_insert("original_current_root_receipt_unavailable");
        }
        if state.root_rows >= MAX_ROOT_ROWS {
            state.first_issue.get_or_insert("root_row_limit");
        }
        let mut row = self.row(&state, "root", Some(context.authority), source_status, root);
        row.initial = initial;
        row.run_failure = failure.map(Into::into);
        if state.root_rows >= MAX_ROOT_ROWS {
            state.latest = Some(row);
            return;
        }
        if let Err(cause) = write_row(&mut io::stderr().lock(), &row, &mut state, false) {
            state.first_issue.get_or_insert(cause);
        } else {
            state.root_rows += 1;
        }
        row.diagnostics_complete = state.first_issue.is_none();
        row.diagnostic_failure = state.first_issue;
        state.latest = Some(row);
    }
    pub(super) fn observe_finish(
        &self,
        product: &RoventNativeProductDriver,
        result: &Result<Option<DriverPhysicalCompletion>, SearchSessionFailure>,
    ) {
        let shared = match product.inner.as_ref() {
            ActualProductDriver::Pc(driver) | ActualProductDriver::PcNnuec(driver) => {
                driver.shared_nnuec_readback()
            }
            _ => Ok(None),
        };
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return;
        };
        state.finish_observed = true;
        state.finish_result = Some(match result {
            Ok(Some(value)) => physical(*value),
            Ok(None) => "unavailable",
            Err(_) => "failed",
        });
        state.finish_failure = result.as_ref().err().map(Into::into);
        match shared {
            Ok(value) => state.shared_nnuec_session = value.map(shared_observation),
            Err(_) => {
                state
                    .first_issue
                    .get_or_insert("shared_nnuec_session_readback_failed");
            }
        }
    }
    // Called after the existing UCI owner has completed its original cleanup.
    // Diagnostic failure cannot skip, replace or repeat that physical cleanup.
    pub(crate) fn complete_session(&self, protocol_succeeded: bool) -> Result<(), &'static str> {
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return Err("diagnostic_state_poisoned");
        };
        let mut row = self.row(
            &state,
            "session",
            None,
            "whole_session_finish_observation",
            None,
        );
        row.protocol_succeeded = Some(protocol_succeeded);
        if let Err(cause) = write_row(&mut io::stderr().lock(), &row, &mut state, true) {
            state.first_issue.get_or_insert(cause);
        }
        row.diagnostics_complete =
            state.first_issue.is_none() && !self.poisoned.load(Ordering::Acquire);
        row.diagnostic_failure = state.first_issue;
        state.latest = Some(row);
        match state.first_issue {
            Some(cause) => Err(cause),
            None => Ok(()),
        }
    }
    fn row(
        &self,
        state: &State,
        event: &'static str,
        authority: Option<SearchAuthority>,
        source_status: &'static str,
        root: Option<RootObservation>,
    ) -> RoventProductLifecycleObservation {
        RoventProductLifecycleObservation {
            domain: ROVENT_LIFECYCLE_DIAGNOSTIC_DOMAIN,
            event,
            sequence: state
                .root_runs
                .saturating_add(u64::from(event == "session")),
            input_lock_sha256: hex(self.lock),
            mode: mode(self.mode),
            implementation_sha256: hex(self.implementation),
            authority: authority.map(Into::into),
            source_status,
            root,
            initial: None,
            run_failure: None,
            finish_observed: state.finish_observed,
            finish_result: state.finish_result.unwrap_or("not_observed"),
            finish_failure: state.finish_failure.clone(),
            shared_nnuec_session: state.shared_nnuec_session.clone(),
            whole_session_physical: if self.physical_close.load(Ordering::Acquire) {
                "confirmed"
            } else {
                "unknown"
            },
            protocol_succeeded: None,
            root_runs_observed: state.root_runs,
            root_run_error_results: state.root_run_errors,
            root_rows_emitted: state.root_rows,
            diagnostics_complete: state.first_issue.is_none()
                && !self.poisoned.load(Ordering::Acquire),
            diagnostic_failure: state.first_issue,
        }
    }
}

fn project_initial(
    product: &RoventNativeProductDriver,
    context: &SearchSessionContext,
) -> Option<InitialObservation> {
    let lock = product.sources.lock_digest();
    match product.inner.as_ref() {
        ActualProductDriver::Pc(driver) | ActualProductDriver::PcNnuec(driver) => {
            let receipt = driver.latest_initial().ok()??;
            if receipt.authority() != context.authority || receipt.input_lock_digest() != lock {
                return None;
            }
            Some(InitialObservation {
                receipt_domain: receipt.domain(),
                source_seal_sha256: hex(receipt.source().digest()),
                previous_clean_root_present: Some(receipt.previous_clean_root().is_some()),
                used_original_startup_ready: Some(receipt.used_original_startup_ready()),
                stopped_before_research_execution: None,
                deadline_before_frontier: None,
                original_nnuec_retirement_present: false,
                original_researcher_release_present: None,
                original_cache_release_epoch: None,
            })
        }
        ActualProductDriver::Nnuec(driver) => {
            if let Some(receipt) = driver.latest_root().ok()? {
                if receipt.authority == context.authority
                    && receipt.input_lock == lock
                    && receipt.frontier.is_none()
                    && receipt.root_admission.authority() == context.authority
                    && receipt.root_admission.startup_lock_digest() == lock
                    && same_original_start(&receipt.root_admission, context)
                {
                    return Some(InitialObservation {
                        receipt_domain: receipt.domain,
                        source_seal_sha256: hex(receipt.root_admission.digest()),
                        previous_clean_root_present: None,
                        used_original_startup_ready: None,
                        stopped_before_research_execution: Some(
                            receipt.stopped_before_research_execution,
                        ),
                        deadline_before_frontier: Some(receipt.deadline_before_frontier),
                        original_nnuec_retirement_present: receipt.initial_retirement.is_some(),
                        original_researcher_release_present: receipt
                            .initial_retirement
                            .as_ref()
                            .map(|value| value.release().is_some()),
                        original_cache_release_epoch: receipt
                            .initial_retirement
                            .as_ref()
                            .map(|value| value.cache_release().epoch()),
                    });
                }
            }
            let receipt = driver.latest_initial().ok()??;
            if receipt.authority() != context.authority || receipt.input_lock_digest() != lock {
                return None;
            }
            Some(InitialObservation {
                receipt_domain: receipt.domain(),
                source_seal_sha256: hex(receipt.source().digest()),
                previous_clean_root_present: Some(receipt.previous_clean_root().is_some()),
                used_original_startup_ready: Some(receipt.used_original_startup_ready()),
                stopped_before_research_execution: None,
                deadline_before_frontier: None,
                original_nnuec_retirement_present: false,
                original_researcher_release_present: None,
                original_cache_release_epoch: None,
            })
        }
        ActualProductDriver::Pcr(driver) => {
            let receipt = driver.latest_rovent_initial().ok()??;
            if receipt.authority() != context.authority
                || receipt.source().startup_lock_digest() != lock
            {
                return None;
            }
            Some(InitialObservation {
                receipt_domain: receipt.domain(),
                source_seal_sha256: hex(receipt.source().digest()),
                previous_clean_root_present: Some(
                    receipt.previous_clean_root().is_some()
                        || receipt.previous_clean_profiled_root().is_some(),
                ),
                used_original_startup_ready: Some(receipt.used_original_startup_ready()),
                stopped_before_research_execution: None,
                deadline_before_frontier: None,
                original_nnuec_retirement_present: false,
                original_researcher_release_present: None,
                original_cache_release_epoch: None,
            })
        }
    }
}

fn original_scope(
    admission: &RoventRootAdmission,
    authority: SearchAuthority,
    lock: Digest,
    context: &SearchSessionContext,
    frontier: &FrontierSnapshot,
) -> bool {
    authority == context.authority
        && admission.authority() == context.authority
        && admission.startup_lock_digest() == lock
        && frontier.authority == context.authority
        && frontier.original_deadline == admission.hard_deadline()
        && same_original_start(admission, context)
}
fn same_original_start(admission: &RoventRootAdmission, context: &SearchSessionContext) -> bool {
    admission
        .original_context()
        .original_start
        .as_ref()
        .zip(context.original_start.as_ref())
        .is_some_and(|(original, current)| Arc::ptr_eq(original, current))
}

fn project_root(
    product: &RoventNativeProductDriver,
    context: &SearchSessionContext,
) -> (&'static str, Option<RootObservation>) {
    let lock = product.sources.lock_digest();
    match product.inner.as_ref() {
        ActualProductDriver::Pc(driver) | ActualProductDriver::PcNnuec(driver) => {
            let receipt = match driver.latest_root() {
                Ok(Some(receipt)) => receipt,
                Ok(None) => return ("no_original_root_receipt", None),
                Err(_) => return ("original_root_receipt_read_failed", None),
            };
            if receipt.input_lock != lock
                || !original_scope(
                    &receipt.root_admission,
                    receipt.authority,
                    lock,
                    context,
                    &receipt.frontier,
                )
                || receipt.physical.authority != context.authority
                || receipt.physical.root_budget != receipt.root_admission.root_budget_digest()
            {
                return ("original_root_scope_mismatch", None);
            }
            let mut row = RootObservation::new(
                receipt.domain,
                &receipt.root_admission,
                &receipt.frontier,
                &receipt.physical.counters,
                Some(RootPhysicalObservation::pc(&receipt.physical)),
            );
            row.r_free_confirmed = Some(
                !receipt.physical.mode.has_native_r()
                    && !product.sources.lock().manifest.engine.mode.has_native_r()
                    && receipt.frontier.work.native_usage_known
                    && receipt.frontier.work.native_nodes == 0
                    && receipt.frontier.work.native_qsearch_nodes == 0
                    && receipt.frontier.work.reserved_native_nodes == 0
                    && receipt.frontier.work.outstanding_native_executions == 0
                    && receipt.physical.counters.cpu_slice_receipts == 0,
            );
            row.shared_nnuec = match driver.shared_nnuec_readback() {
                Ok(value) => value.map(shared_observation),
                Err(_) => return ("shared_nnuec_readback_failed", None),
            };
            ("original_current_root", Some(row))
        }
        ActualProductDriver::Nnuec(driver) => {
            let receipt = match driver.latest_root() {
                Ok(Some(receipt)) => receipt,
                Ok(None) => return ("no_original_root_receipt", None),
                Err(_) => return ("original_root_receipt_read_failed", None),
            };
            let Some(frontier) = &receipt.frontier else {
                return ("original_frontier_not_admitted", None);
            };
            if receipt.input_lock != lock
                || !original_scope(
                    &receipt.root_admission,
                    receipt.authority,
                    lock,
                    context,
                    frontier,
                )
                || receipt
                    .physical
                    .as_ref()
                    .is_some_and(|value| value.authority != context.authority)
            {
                return ("original_root_scope_mismatch", None);
            }
            (
                "original_current_root",
                Some(RootObservation::new(
                    receipt.domain,
                    &receipt.root_admission,
                    frontier,
                    &receipt.counters,
                    receipt.physical.as_ref().map(RootPhysicalObservation::pcr),
                )),
            )
        }
        ActualProductDriver::Pcr(driver) => {
            let receipt = match driver.latest_rovent_root() {
                Ok(Some(receipt)) => receipt,
                Ok(None) => return ("no_original_root_receipt", None),
                Err(_) => return ("original_root_receipt_read_failed", None),
            };
            let (Some(admission), Some(frontier)) = (&receipt.root_admission, &receipt.frontier)
            else {
                return ("original_frontier_not_admitted", None);
            };
            if receipt.input_lock != lock
                || !original_scope(admission, receipt.authority, lock, context, frontier)
                || receipt
                    .physical
                    .as_ref()
                    .is_some_and(|value| value.authority != context.authority)
            {
                return ("original_root_scope_mismatch", None);
            }
            (
                "original_current_root",
                Some(RootObservation::new(
                    receipt.domain,
                    admission,
                    frontier,
                    &receipt.counters,
                    receipt.physical.as_ref().map(RootPhysicalObservation::pcr),
                )),
            )
        }
    }
}

// Allocate only the registered row bound before serialization. serde's writer
// cannot grow this buffer, even if a future DTO accidentally gains a long field.
struct RowBuffer(Vec<u8>);
impl Write for RowBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_ROW_BYTES.saturating_sub(self.0.len()) {
            return Err(io::Error::other("lifecycle diagnostic row byte limit"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn write_row(
    output: &mut impl Write,
    row: &RoventProductLifecycleObservation,
    state: &mut State,
    final_row: bool,
) -> Result<(), &'static str> {
    let mut buffer = RowBuffer(Vec::new());
    buffer
        .0
        .try_reserve_exact(MAX_ROW_BYTES)
        .map_err(|_| "diagnostic_allocation_failed")?;
    serde_json::to_writer(&mut buffer, row).map_err(|_| "diagnostic_row_encoding_or_byte_limit")?;
    buffer
        .write_all(b"\n")
        .map_err(|_| "diagnostic_row_byte_limit")?;
    let limit = if final_row {
        MAX_STREAM_BYTES
    } else {
        MAX_STREAM_BYTES - MAX_ROW_BYTES
    };
    if buffer.0.len() > limit.saturating_sub(state.bytes) {
        return Err("diagnostic_stream_byte_limit");
    }
    // Charge a partial/failed OS write once as a full attempted row. The final
    // receipt remains incomplete; retrying a torn JSONL record is prohibited.
    state.bytes += buffer.0.len();
    output
        .write_all(&buffer.0)
        .map_err(|_| "diagnostic_write_failed")?;
    output.flush().map_err(|_| "diagnostic_flush_failed")
}
fn physical(value: DriverPhysicalCompletion) -> &'static str {
    match value {
        DriverPhysicalCompletion::Confirmed => "confirmed",
        DriverPhysicalCompletion::Unknown => "unknown",
    }
}
fn shared_observation(
    value: super::super::pc_session::RoventPcSharedNnuecReadback,
) -> SharedNnuecObservation {
    SharedNnuecObservation {
        spawned_runtime_helpers: value.spawned_runtime_helpers,
        worker_thread_id_observed: value.worker_thread_id.is_some(),
        worker_joined: value.worker_joined,
        worker_thread_affinity_known: value.worker_thread_affinity.is_some(),
    }
}
fn mode(value: ExecutionMode) -> &'static str {
    match value {
        ExecutionMode::CpuPc => "cpu_pc",
        ExecutionMode::CpuPcr => "cpu_pcr",
        ExecutionMode::HybridPc => "hybrid_pc",
        ExecutionMode::HybridPcr => "hybrid_pcr",
    }
}
fn goal(value: FrontierGoalStatus) -> &'static str {
    match value {
        FrontierGoalStatus::Pending => "pending",
        FrontierGoalStatus::DepthSatisfied => "depth_satisfied",
        FrontierGoalStatus::MateSatisfied => "mate_satisfied",
        FrontierGoalStatus::GoalUnresolved => "goal_unresolved",
        FrontierGoalStatus::Stopped(reason) => match reason {
            FrontierStop::Canceled => "stopped_canceled",
            FrontierStop::Deadline => "stopped_deadline",
            FrontierStop::WorkBudget => "stopped_work_budget",
            FrontierStop::Capacity => "stopped_capacity",
            FrontierStop::NativeUnknown => "stopped_native_unknown",
            FrontierStop::EvidenceReclaimed => "stopped_evidence_reclaimed",
            FrontierStop::Failed => "stopped_failed",
        },
    }
}
fn bounded(value: &str, limit: usize) -> String {
    let mut end = value.len().min(limit);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].into()
}
fn hex(value: Digest) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(64);
    for byte in value.0 {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 15) as usize] as char);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagnostic() -> RoventLifecycleDiagnostics {
        RoventLifecycleDiagnostics::new_from_sources(
            Digest([3; 32]),
            Digest([4; 32]),
            ExecutionMode::CpuPc,
            Arc::new(AtomicBool::new(false)),
        )
    }

    #[test]
    fn protocol_success_and_root_rows_cannot_manufacture_a_session_finish() {
        let diagnostics = diagnostic();
        let state = State::default();
        let mut row = diagnostics.row(
            &state,
            "session",
            None,
            "whole_session_finish_observation",
            None,
        );
        row.protocol_succeeded = Some(true);
        let mut output = Vec::new();
        let mut accounting = State::default();
        write_row(&mut output, &row, &mut accounting, true).unwrap();
        let decoded: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(decoded["protocol_succeeded"], true);
        assert_eq!(decoded["whole_session_physical"], "unknown");
        assert_eq!(decoded["finish_observed"], false);
        assert_eq!(decoded["finish_result"], "not_observed");
        assert!(decoded["root"].is_null());
        assert!(decoded["initial"].is_null());
        assert_eq!(output.last(), Some(&b'\n'));
    }

    #[test]
    fn bounded_stream_retains_the_final_incomplete_observation() {
        let diagnostics = diagnostic();
        let mut state = State {
            root_runs: MAX_ROOT_ROWS + 1,
            root_rows: MAX_ROOT_ROWS,
            bytes: MAX_STREAM_BYTES - MAX_ROW_BYTES,
            first_issue: Some("root_row_limit"),
            ..State::default()
        };
        let row = diagnostics.row(
            &state,
            "session",
            None,
            "whole_session_finish_observation",
            None,
        );
        let mut output = Vec::new();
        assert_eq!(
            write_row(&mut output, &row, &mut state, false),
            Err("diagnostic_stream_byte_limit")
        );
        assert!(output.is_empty());
        write_row(&mut output, &row, &mut state, true).unwrap();
        let decoded: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(decoded["diagnostics_complete"], false);
        assert_eq!(decoded["diagnostic_failure"], "root_row_limit");
        assert_eq!(decoded["root_runs_observed"], MAX_ROOT_ROWS + 1);
        assert_eq!(decoded["root_rows_emitted"], MAX_ROOT_ROWS);
        assert_eq!(decoded["whole_session_physical"], "unknown");
        assert!(state.bytes <= MAX_STREAM_BYTES);
    }

    #[test]
    fn partial_diagnostic_io_cannot_advance_a_root_or_physical_completion() {
        struct TornOutput(usize);
        impl Write for TornOutput {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.0 == 0 {
                    return Err(io::Error::other("original FD rejected write"));
                }
                let count = bytes.len().min(self.0);
                self.0 -= count;
                Ok(count)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let diagnostics = diagnostic();
        let mut state = State::default();
        let row = diagnostics.row(&state, "root", None, "no_original_root_receipt", None);
        assert_eq!(
            write_row(&mut TornOutput(7), &row, &mut state, false),
            Err("diagnostic_write_failed")
        );
        assert!(state.bytes > 7); // Conservatively retain the entire attempted write charge.
        assert_eq!(state.root_rows, 0);
        assert!(!state.finish_observed);
        assert!(!diagnostics.physical_close.load(Ordering::Acquire));
    }
}
