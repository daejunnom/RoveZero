//! Parent-observed cost through result checking, on the original preparation
//! clock. This is neither an admitted native witness nor a next Query ledger.

use super::{BoundRepairReplayDelivery, OwnedReplayCapture};
use crate::{ArenaError, OriginalProcessTiming};
use rz_uci::pals_cpu_task::strategic_action::ArtifactPin;
use rz_uci::pals_cpu_task::strategic_action::native_replay::query_prior;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// No public constructor, Clone or serde. The borrow retains the actual capture
/// and its private original clock, raw bytes and any process custody.
#[derive(Debug)]
pub struct ReplayCallerTiming<'a> {
    process: OriginalProcessTiming<'a>,
    capture: &'a OwnedReplayCapture,
    caller_finished_ns: u64,
    checking_started_ns: u64,
    checking_finished_ns: u64,
    checking_finished_at: Instant,
    whole_ns: u64,
}
impl ReplayCallerTiming<'_> {
    pub fn capture(&self) -> &OwnedReplayCapture {
        self.capture
    }
    pub fn process(&self) -> &OriginalProcessTiming<'_> {
        &self.process
    }
    /// Includes original preparation and caller preflight. These cannot yet be
    /// split into separate phases from the available observed timestamps.
    pub fn preparation_and_preflight_ns(&self) -> u64 {
        self.process.supervisor_entry_ns()
    }
    pub fn supervisor_setup_ns(&self) -> u64 {
        self.process.launch_started_ns() - self.process.supervisor_entry_ns()
    }
    /// Includes spawn, stdin delivery, child work and the parent's wait. It is
    /// deliberately not labelled NN time or physically completed model work.
    pub fn launch_to_exit_observation_ns(&self) -> u64 {
        self.process.exit_observed_ns() - self.process.launch_started_ns()
    }
    pub fn drain_and_supervisor_return_ns(&self) -> u64 {
        self.process.supervisor_finished_ns() - self.process.exit_observed_ns()
    }
    pub fn caller_postflight_ns(&self) -> u64 {
        self.caller_finished_ns - self.process.supervisor_finished_ns()
    }
    /// Includes delivery binding, caller gaps and any earlier result checks.
    /// Delay after capture is charged, even when it was not useful work.
    pub fn after_capture_before_check_ns(&self) -> u64 {
        self.checking_started_ns - self.caller_finished_ns
    }
    pub fn result_check_ns(&self) -> u64 {
        self.checking_finished_ns - self.checking_started_ns
    }
    /// Cost through this check only. Later Query selection/serialization must be
    /// charged by the episode owner; this is not complete utility action cost.
    pub fn elapsed_through_check_ns(&self) -> u64 {
        self.checking_finished_ns
    }
    pub fn remaining_original_whole_ns(&self) -> u64 {
        self.whole_ns - self.checking_finished_ns
    }
    pub fn checking_finished_at(&self) -> Instant {
        self.checking_finished_at
    }
    pub fn assurance_scope(&self) -> &'static str {
        "actual_caller_transport_and_cost_through_report_check_pending_native_witness_and_next_query"
    }
}

/// Reported Repair facts checked by original Rules, paired with real parent
/// timestamps from the same capture. Still no native or Query capability.
pub struct CheckedRepairReplayObservation<'a> {
    report: query_prior::ReportedRepairRulesConsistency,
    timing: ReplayCallerTiming<'a>,
    body_artifact: ArtifactPin,
}
impl CheckedRepairReplayObservation<'_> {
    pub fn reported_rules(&self) -> &query_prior::ReportedRepairRulesConsistency {
        &self.report
    }
    pub fn caller_timing(&self) -> &ReplayCallerTiming<'_> {
        &self.timing
    }
    pub fn body_artifact(&self) -> &ArtifactPin {
        &self.body_artifact
    }
}

impl<'capture> BoundRepairReplayDelivery<'capture> {
    /// The actual checking interval is recorded here, not supplied by a report
    /// or caller-provided timestamp. All checks use the unchanged original W.
    pub fn check_reported_rules_with_timing(
        &self,
        cancel: &AtomicBool,
    ) -> Result<CheckedRepairReplayObservation<'capture>, ArenaError> {
        let started = Instant::now();
        let report = self.check_reported_rules_consistency(cancel)?;
        let body_artifact = self.body_artifact().clone();
        let finished = Instant::now();
        let timing = check_timing(self.capture(), started, finished, cancel)?;
        Ok(CheckedRepairReplayObservation {
            report,
            timing,
            body_artifact,
        })
    }
}

fn elapsed(start: Instant, at: Instant) -> Result<u64, ArenaError> {
    at.checked_duration_since(start)
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok())
        .ok_or_else(|| invalid("caller original clock extent/order"))
}
fn invalid(detail: &str) -> ArenaError {
    ArenaError::Invalid(detail.into())
}
fn check_timing<'a>(
    capture: &'a OwnedReplayCapture,
    started: Instant,
    finished: Instant,
    cancel: &AtomicBool,
) -> Result<ReplayCallerTiming<'a>, ArenaError> {
    if !capture.transport_complete() || cancel.load(Ordering::Acquire) {
        return Err(invalid("caller timing requires uncancelled actual closure"));
    }
    let process = capture.process().checked_timing()?;
    let bundle = capture.bundle();
    if process.original_started() != bundle.original_started()
        || process.execution_deadline() != bundle.execution_deadline()
        || process.whole_deadline() != bundle.deadline()
    {
        return Err(invalid("caller process/preparation original clock differs"));
    }
    let caller_finished_ns = elapsed(bundle.original_started(), capture.caller_finished)?;
    let checking_started_ns = elapsed(bundle.original_started(), started)?;
    let checking_finished_ns = elapsed(bundle.original_started(), finished)?;
    let whole_ns = elapsed(bundle.original_started(), bundle.deadline())?;
    check_order([
        process.supervisor_finished_ns(),
        caller_finished_ns,
        checking_started_ns,
        checking_finished_ns,
        whole_ns,
    ])?;
    // Check return time separately: a delayed checker cannot use its older local
    // finish stamp to revive the original window after W or after cancellation.
    if Instant::now() >= bundle.deadline() || cancel.load(Ordering::Acquire) {
        return Err(invalid("caller timing original W/cancellation at return"));
    }
    Ok(ReplayCallerTiming {
        process,
        capture,
        caller_finished_ns,
        checking_started_ns,
        checking_finished_ns,
        checking_finished_at: finished,
        whole_ns,
    })
}
fn check_order(stamps: [u64; 5]) -> Result<(), ArenaError> {
    if stamps[..4].windows(2).any(|p| p[0] > p[1]) || stamps[3] >= stamps[4] {
        Err(invalid("caller postflight/check chronology or original W"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // Pure boundary tests, not fabricated capture/process/native owners.
    #[test]
    fn caller_timing_refuses_reordered_postflight_check_and_whole_boundary() {
        assert!(check_order([10, 20, 30, 40, 50]).is_ok());
        assert!(check_order([10, 10, 10, 10, 11]).is_ok());
        for bad in [
            [20, 10, 30, 40, 50],
            [10, 30, 20, 40, 50],
            [10, 20, 40, 30, 50],
            [10, 20, 30, 50, 50],
            [10, 20, 30, 51, 50],
        ] {
            assert!(check_order(bad).is_err());
        }
    }
    #[test]
    fn caller_timing_preserves_original_cost_without_saturating_or_restarting() {
        let start = Instant::now();
        assert_eq!(
            elapsed(start, start + Duration::from_nanos(401)).unwrap(),
            401
        );
        // A 1ns subtraction is not distinct on every supported platform clock.
        assert!(elapsed(start, start - Duration::from_millis(1)).is_err());
        let captured = start + Duration::from_nanos(200);
        let checked = captured + Duration::from_nanos(300);
        assert_eq!(elapsed(start, checked).unwrap(), 500);
        assert_ne!(
            elapsed(start, checked).unwrap(),
            elapsed(captured, checked).unwrap()
        );
    }
}
