//! Owner-produced ticket lifecycle observations, not inferred NN work counts.
//!
//! The registry records only successful transitions of its original tickets.
//! Retiring a ticket never removes these cumulative facts. A dispatch is not a
//! physical-start notification, a known physical completion is not logical
//! acceptance, and a release is not proof of neural execution or cache use.

use super::TicketPhase;

/// Current occupancy or independently observed high water for each phase.
/// Phase peaks need not have occurred at the same instant.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RoleTicketPhaseOccupancy {
    pub prepared: u64,
    pub submitted: u64,
    pub physically_running: u64,
    pub completed: u64,
    pub delivered: u64,
    pub quarantined: u64,
}

impl RoleTicketPhaseOccupancy {
    pub fn held(self) -> u64 {
        [
            self.prepared,
            self.submitted,
            self.physically_running,
            self.completed,
            self.delivered,
            self.quarantined,
        ]
        .into_iter()
        .try_fold(0_u64, |sum, count| sum.checked_add(count))
        .expect("bounded role-ticket phase occupancy")
    }

    pub(super) fn observe_phase(&mut self, phase: TicketPhase) {
        let count = match phase {
            TicketPhase::Prepared => &mut self.prepared,
            TicketPhase::Submitted => &mut self.submitted,
            TicketPhase::PhysicallyRunning => &mut self.physically_running,
            TicketPhase::Completed => &mut self.completed,
            TicketPhase::Delivered => &mut self.delivered,
            TicketPhase::Quarantined => &mut self.quarantined,
        };
        *count = count.checked_add(1).expect("at most 16 held role tickets");
    }

    fn retain_peaks(&mut self, current: Self) {
        self.prepared = self.prepared.max(current.prepared);
        self.submitted = self.submitted.max(current.submitted);
        self.physically_running = self.physically_running.max(current.physically_running);
        self.completed = self.completed.max(current.completed);
        self.delivered = self.delivered.max(current.delivered);
        self.quarantined = self.quarantined.max(current.quarantined);
    }
}

/// Actual transitions accumulated by one registry authority. No NN-input,
/// cache-hit or search-consumption totals are invented from ticket counts.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RoleTicketCumulativeObservation {
    pub admitted: u64,
    pub service_dispatched: u64,
    pub physical_start_notifications: u64,
    /// Confirmed physical completions, including confirmed error outputs.
    pub physically_completed: u64,
    pub delivered: u64,
    pub logically_accepted: u64,
    /// Prepared tickets revoked and released before acquiring a service lease.
    pub terminal_before_service_dispatch: u64,
    /// Actual removals, including rejected outputs, fence and joined cleanup.
    pub released: u64,
    /// Tickets whose original revocation flag changed from false to true.
    pub revoked: u64,
    pub physical_unknown_events: u64,
    pub current: RoleTicketPhaseOccupancy,
    pub phase_peaks: RoleTicketPhaseOccupancy,
    pub held_peak: u64,
}

pub(super) enum RoleTicketLifecycleEvent {
    Admitted,
    ServiceDispatched,
    PhysicalStartNotification,
    PhysicallyCompleted,
    Delivered,
    LogicallyAccepted,
    TerminalBeforeServiceDispatch,
    Released,
    Revoked,
    PhysicalUnknownEvent,
}

impl RoleTicketCumulativeObservation {
    pub(super) fn record(&mut self, event: RoleTicketLifecycleEvent, actual_count: u64) {
        let count = match event {
            RoleTicketLifecycleEvent::Admitted => &mut self.admitted,
            RoleTicketLifecycleEvent::ServiceDispatched => &mut self.service_dispatched,
            RoleTicketLifecycleEvent::PhysicalStartNotification => {
                &mut self.physical_start_notifications
            }
            RoleTicketLifecycleEvent::PhysicallyCompleted => &mut self.physically_completed,
            RoleTicketLifecycleEvent::Delivered => &mut self.delivered,
            RoleTicketLifecycleEvent::LogicallyAccepted => &mut self.logically_accepted,
            RoleTicketLifecycleEvent::TerminalBeforeServiceDispatch => {
                &mut self.terminal_before_service_dispatch
            }
            RoleTicketLifecycleEvent::Released => &mut self.released,
            RoleTicketLifecycleEvent::Revoked => &mut self.revoked,
            RoleTicketLifecycleEvent::PhysicalUnknownEvent => &mut self.physical_unknown_events,
        };
        // Each event occurs at most once per admitted RequestId. Admission's
        // checked, monotonically increasing nonzero sequence refuses issuance
        // at u64::MAX, so every counter is bounded by admitted < u64::MAX.
        // Bulk cleanup counts only the <=16 actually held original tickets.
        *count = count
            .checked_add(actual_count)
            .expect("lifecycle totals bounded by checked RequestId issuance");
    }

    pub(super) fn observe_occupancy(&mut self, current: RoleTicketPhaseOccupancy) {
        self.current = current;
        self.phase_peaks.retain_peaks(current);
        self.held_peak = self.held_peak.max(current.held());
    }
}

#[cfg(test)]
#[path = "ticket_observation_tests.rs"]
mod tests;
