//! One-shot parent-observed material for a subsequent private Query. A native
//! report does not become an admitted native witness, Query or utility target.

use super::chronology::elapsed;
use super::{CheckedRepairReplayObservation, OwnedReplayCapture, RepairReplayDeliveryRegistration};
use crate::ArenaError;
use crate::process::OriginalLoadedImage;
use rz_uci::pals_cpu_task::strategic_action::replay_inputs::{
    PreparedReplayRequest, ReplayBindingPins, ReplayParentPins,
};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Instant;

/// Constructed only by the actual capture owner under its original W. No public
/// constructor, Clone or serde: the material borrows original input and stdout.
/// A separate next-Query checker must still admit the question, native witness,
/// prior ledger and whole causal costs before using these facts as a target.
pub struct ReportedRepairPriorMaterial<'capture> {
    observation: CheckedRepairReplayObservation<'capture>,
    original: &'capture PreparedReplayRequest,
    raw_native: &'capture [u8],
    loaded_image: OriginalLoadedImage<'capture>,
    caller_started_ns: u64,
    before_next_query_at: Instant,
    elapsed_before_next_query_ns: u64,
}
impl ReportedRepairPriorMaterial<'_> {
    pub fn checked_report(&self) -> &CheckedRepairReplayObservation<'_> {
        &self.observation
    }
    pub fn original_input(&self) -> &PreparedReplayRequest {
        self.original
    }
    pub fn original_parent(&self) -> &ReplayParentPins {
        &self.original.audit().input_admission.parent
    }
    pub fn previous_query_binding(&self) -> &ReplayBindingPins {
        &self.original.audit().input_admission.binding
    }
    pub fn raw_native_bytes(&self) -> &[u8] {
        self.raw_native
    }
    /// Same-inode parent observation before stdin, not content/source or native
    /// witness admission. Its owner is the same actual process capture.
    pub fn loaded_image(&self) -> &OriginalLoadedImage<'_> {
        &self.loaded_image
    }
    pub fn caller_started_ns(&self) -> u64 {
        self.caller_started_ns
    }
    /// Actual parent observation before returning material, not a reported or
    /// portable global sequence number. It cannot authorize a later execution.
    pub fn before_next_query_at(&self) -> Instant {
        self.before_next_query_at
    }
    pub fn elapsed_before_next_query_ns(&self) -> u64 {
        self.elapsed_before_next_query_ns
    }
    pub fn after_report_check_ns(&self) -> u64 {
        self.elapsed_before_next_query_ns
            - self.observation.caller_timing().elapsed_through_check_ns()
    }
    pub fn assurance_scope(&self) -> &'static str {
        "actual_caller_one_shot_reported_prior_material_pending_native_witness_and_next_query_admission"
    }
}

impl OwnedReplayCapture {
    /// Source registration, actual loaded-file identity, raw binding, original
    /// Rules and process timing checks happen before the single issue. Refusal
    /// leaves input/raw bytes and process custody intact. Issued material is never
    /// replenished by drop, changed registration or resetting cancellation.
    pub fn prepare_reported_repair_prior_material(
        &self,
        registration: &RepairReplayDeliveryRegistration,
        cancel: &AtomicBool,
    ) -> Result<ReportedRepairPriorMaterial<'_>, ArenaError> {
        let reservation = self
            .prior_material_issue
            .reserve(self.bundle().deadline(), cancel)?;
        let loaded_image = self.process().checked_loaded_image()?;
        let delivery = self.bind_repair_delivery(registration, cancel)?;
        let observation = delivery.check_reported_rules_with_timing(cancel)?;
        let mut originals = self
            .bundle()
            .payloads()
            .iter()
            .filter_map(|payload| payload.prepared_request());
        let original = originals
            .next()
            .ok_or_else(|| invalid("original input missing"))?;
        if originals.next().is_some()
            || original.deadline() != self.bundle().deadline()
            || observation
                .reported_rules()
                .rules()
                .original_input_artifact()
                != original.artifact()
        {
            return Err(invalid("prior material original input owner differs"));
        }
        let timing = observation.caller_timing();
        let caller_started_ns = elapsed(self.bundle().original_started(), self.caller_started)?;
        if caller_started_ns > timing.process().supervisor_entry_ns() {
            return Err(invalid("caller started after supervisor entry"));
        }
        reservation.issue(self.bundle().deadline(), cancel)?;
        let before_next_query_at = Instant::now();
        let elapsed_before_next_query_ns =
            elapsed(self.bundle().original_started(), before_next_query_at)?;
        // Once issued, a late/cancelled attempt remains spent. This prevents a
        // second material observation from the same physical child, including
        // through aliases or after an aborted downstream consumer.
        control(self.bundle().deadline(), cancel)?;
        if elapsed_before_next_query_ns < timing.elapsed_through_check_ns() {
            return Err(invalid("material precedes report check"));
        }
        Ok(ReportedRepairPriorMaterial {
            observation,
            original,
            raw_native: delivery.raw_native_bytes(),
            loaded_image,
            caller_started_ns,
            before_next_query_at,
            elapsed_before_next_query_ns,
        })
    }
}

fn invalid(detail: &str) -> ArenaError {
    ArenaError::Invalid(detail.into())
}
fn control(deadline: Instant, cancel: &AtomicBool) -> Result<(), ArenaError> {
    if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
        Err(invalid("prior material original W/cancellation"))
    } else {
        Ok(())
    }
}
const AVAILABLE: u8 = 0;
const CHECKING: u8 = 1;
const ISSUED: u8 = 2;

#[derive(Debug, Default)]
pub(super) struct MaterialIssueGate {
    state: AtomicU8,
}
impl MaterialIssueGate {
    fn reserve(
        &self,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<MaterialReservation<'_>, ArenaError> {
        control(deadline, cancel)?;
        self.state
            .compare_exchange(AVAILABLE, CHECKING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("same capture prior check in progress/already issued"))?;
        Ok(MaterialReservation {
            gate: self,
            issued: false,
        })
    }
}
// Excludes concurrent Rules reconstruction. Parser/source/binding/Rules refusal
// drops the reservation before issue and allows correction only within original
// W. After issue, even a late/cancelled return remains spent.
struct MaterialReservation<'a> {
    gate: &'a MaterialIssueGate,
    issued: bool,
}
impl MaterialReservation<'_> {
    fn issue(mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), ArenaError> {
        control(deadline, cancel)?;
        self.gate
            .state
            .compare_exchange(CHECKING, ISSUED, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("prior issue reservation ownership lost"))?;
        self.issued = true;
        Ok(())
    }
}
impl Drop for MaterialReservation<'_> {
    fn drop(&mut self) {
        if !self.issued {
            let _ = self.gate.state.compare_exchange(
                CHECKING,
                AVAILABLE,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // Only the private issue gate is exercised here. No native/capture/Query
    // authority is constructed from a fixture or from a received declaration.
    #[test]
    fn prior_material_issue_is_one_shot_even_after_consumer_drop() {
        let gate = MaterialIssueGate::default();
        let cancel = AtomicBool::new(false);
        let deadline = Instant::now() + Duration::from_secs(5);
        assert!(
            gate.reserve(deadline, &cancel)
                .unwrap()
                .issue(deadline, &cancel)
                .is_ok()
        );
        assert!(gate.reserve(deadline, &cancel).is_err());
    }
    #[test]
    fn prior_material_expiry_and_cancel_do_not_issue_or_restart_clock() {
        let gate = MaterialIssueGate::default();
        let cancel = AtomicBool::new(true);
        let deadline = Instant::now() + Duration::from_secs(5);
        assert!(gate.reserve(deadline, &cancel).is_err());
        assert_eq!(gate.state.load(Ordering::Acquire), AVAILABLE);
        cancel.store(false, Ordering::Release);
        assert!(gate.reserve(Instant::now(), &cancel).is_err());
        assert_eq!(gate.state.load(Ordering::Acquire), AVAILABLE);
        let reservation = gate.reserve(deadline, &cancel).unwrap();
        cancel.store(true, Ordering::Release);
        assert!(reservation.issue(deadline, &cancel).is_err());
        assert_eq!(gate.state.load(Ordering::Acquire), AVAILABLE);
        cancel.store(false, Ordering::Release);
        gate.reserve(deadline, &cancel)
            .unwrap()
            .issue(deadline, &cancel)
            .unwrap();
        cancel.store(true, Ordering::Release);
        assert!(gate.reserve(deadline, &cancel).is_err());
        cancel.store(false, Ordering::Release);
        assert!(gate.reserve(deadline, &cancel).is_err());
    }
    #[test]
    fn prior_material_concurrent_aliases_only_issue_once() {
        let gate = MaterialIssueGate::default();
        let cancel = AtomicBool::new(false);
        let deadline = Instant::now() + Duration::from_secs(5);
        let successes = std::thread::scope(|scope| {
            let run = || {
                gate.reserve(deadline, &cancel)
                    .and_then(|reservation| reservation.issue(deadline, &cancel))
                    .is_ok()
            };
            let first = scope.spawn(run);
            let second = scope.spawn(run);
            usize::from(first.join().unwrap()) + usize::from(second.join().unwrap())
        });
        assert_eq!(successes, 1);
        assert_eq!(gate.state.load(Ordering::Acquire), ISSUED);
    }
    #[test]
    fn prior_material_failed_check_releases_reservation_without_rearming_issue() {
        let gate = MaterialIssueGate::default();
        let cancel = AtomicBool::new(false);
        let deadline = Instant::now() + Duration::from_secs(5);
        {
            let _refused_check = gate.reserve(deadline, &cancel).unwrap();
            assert!(gate.reserve(deadline, &cancel).is_err());
        }
        assert_eq!(gate.state.load(Ordering::Acquire), AVAILABLE);
        gate.reserve(deadline, &cancel)
            .unwrap()
            .issue(deadline, &cancel)
            .unwrap();
        assert!(gate.reserve(deadline, &cancel).is_err());
    }
}
