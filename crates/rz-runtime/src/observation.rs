use crate::{Resources, RuntimeFault};
use rz_telemetry::FinishKind;
use std::collections::VecDeque;
use std::mem::size_of;

/// Passive scheduler facts. IDs and ticks belong to the embedding adapter.
/// Validation success does not authorize backup; Finished does not mean physical
/// completion. Dispatch timestamps bracket the backend call, not GPU transfer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObservationKind<R, E, T> {
    Admitted {
        request: R,
    },
    Rejected {
        request: R,
    },
    Dispatched {
        execution: E,
        items: usize,
        dispatch_started_at: T,
    },
    RequestDispatched {
        request: R,
        execution: E,
        dispatch_started_at: T,
    },
    PhysicalCompleted {
        execution: E,
    },
    ValidationStarted {
        request: R,
        execution: E,
    },
    ValidationFinished {
        request: R,
        execution: E,
        valid: bool,
    },
    Finished {
        request: R,
        kind: FinishKind,
        /// Enqueued in the completion mailbox; does not mean received/backed up.
        delivered: bool,
    },
    /// A changed owner-side ledger snapshot, not the exact time of a receiver's
    /// release. An unchanged idle/pending pump does not emit this event.
    ReservationChanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Observation<R, E, T> {
    pub at: T,
    /// Reservation ledger; excludes model, metadata and telemetry storage.
    pub reserved: Resources,
    /// Exact per-domain ledger high-water marks despite concurrent receipt release.
    pub peak_reserved: Resources,
    pub reserved_requests: usize,
    pub kind: ObservationKind<R, E, T>,
}

/// Events retained since the last drain, oldest first. A nonzero loss means
/// lifecycle reconstruction is incomplete. Counters describe this drain only.
#[derive(Debug)]
pub struct ObservationDrain<R, E, T> {
    pub events: Vec<Observation<R, E, T>>,
    pub dropped: u64,
    pub counter_overflow: bool,
}

pub(crate) struct ObservationBufferDrain<E> {
    pub(crate) events: Vec<E>,
    pub(crate) dropped: u64,
    pub(crate) counter_overflow: bool,
}

/// Shared bounded storage for the separate scheduler and delivery fact streams.
pub(crate) struct Observations<E> {
    events: VecDeque<E>,
    limit: usize,
    dropped: u64,
    counter_overflow: bool,
}

impl<E> Observations<E> {
    pub(crate) fn try_new(limit: usize) -> Result<Self, RuntimeFault> {
        let bytes = limit
            .checked_mul(size_of::<E>())
            .ok_or(RuntimeFault::ResourceOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(RuntimeFault::ResourceOverflow);
        }
        let mut events = VecDeque::new();
        events
            .try_reserve_exact(limit)
            .map_err(|_| RuntimeFault::ResourceLimit)?;
        Ok(Self {
            events,
            limit,
            dropped: 0,
            counter_overflow: false,
        })
    }

    pub(crate) fn push(&mut self, event: E) {
        if self.limit == 0 || self.events.len() == self.limit {
            if self.dropped == u64::MAX {
                self.counter_overflow = true;
            } else {
                self.dropped += 1;
            }
            if self.limit == 0 {
                return;
            }
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    pub(crate) fn drain(&mut self) -> ObservationBufferDrain<E> {
        ObservationBufferDrain {
            events: self.events.drain(..).collect(),
            dropped: std::mem::take(&mut self.dropped),
            counter_overflow: std::mem::take(&mut self.counter_overflow),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(at: u64) -> Observation<u64, u64, u64> {
        Observation {
            at,
            reserved: Resources::default(),
            peak_reserved: Resources::default(),
            reserved_requests: 0,
            kind: ObservationKind::ReservationChanged,
        }
    }

    #[test]
    fn retains_latest_events_in_order_and_reports_loss_per_drain() {
        let mut ring = Observations::try_new(2).unwrap();
        for tick in 1..=4 {
            ring.push(event(tick));
        }
        let first = ring.drain();
        assert_eq!(
            first
                .events
                .iter()
                .map(|event| event.at)
                .collect::<Vec<_>>(),
            [3, 4]
        );
        assert_eq!(first.dropped, 2);
        assert!(!first.counter_overflow);
        ring.push(event(5));
        let second = ring.drain();
        assert_eq!(second.events.len(), 1);
        assert_eq!(second.dropped, 0);
    }

    #[test]
    fn zero_capacity_records_loss_and_saturation_is_explicit() {
        let mut ring = Observations::try_new(0).unwrap();
        ring.push(event(1));
        assert_eq!(ring.drain().dropped, 1);
        ring.dropped = u64::MAX;
        ring.push(event(2));
        let result = ring.drain();
        assert!(result.events.is_empty());
        assert_eq!(result.dropped, u64::MAX);
        assert!(result.counter_overflow);
        assert!(!ring.drain().counter_overflow);
    }

    #[test]
    fn event_layout_overflow_is_rejected_before_allocation() {
        assert!(matches!(
            Observations::<Observation<u64, u64, u64>>::try_new(usize::MAX),
            Err(RuntimeFault::ResourceOverflow)
        ));
    }
}
