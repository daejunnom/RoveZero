//! Optional bounded source-clock journal. Recording never grants engine authority.
//! Spans may arrive out of order from different threads; original Instants are
//! retained and sorted only by the report consumer. No payload or device pin is held.

use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SourceStage {
    SearchPreparation,
    EncodingPreparation,
    NativePreparation,
    NativeInvocation,
    NativeOutput,
    PhysicalWorker,
    Admitted,
    DispatchStarted,
    DispatchFinished,
    PhysicalReadyObserved,
    ValidationStarted,
    ValidationFinished,
    LogicalFinished,
    DeliveryAccepted,
    DeliveryRejected,
    DeliveryTerminal,
    SearchBackup,
    ProtocolWrite,
}

#[derive(Clone, Debug)]
pub struct SourceSpan<K> {
    pub key: Option<K>,
    pub stage: SourceStage,
    pub start: Instant,
    pub end: Instant,
    pub succeeded: bool,
}

#[derive(Debug)]
pub struct SourceSnapshot<K> {
    pub origin: Instant,
    pub capacity: usize,
    pub records: Vec<SourceSpan<K>>,
    pub dropped: u64,
    pub invalid_intervals: u64,
    pub external_events_dropped: u64,
    pub counter_overflow: bool,
    pub poisoned: bool,
}

struct State<K> {
    records: Vec<SourceSpan<K>>,
    dropped: u64,
    invalid_intervals: u64,
    external_events_dropped: u64,
    counter_overflow: bool,
    poisoned: bool,
}

/// Fixed admission capacity, shared by the producer and final report consumer.
/// Full/poisoned observation storage never changes an evaluation or its lifetime.
pub struct SourceJournal<K> {
    origin: Instant,
    capacity: usize,
    state: Arc<Mutex<State<K>>>,
}

impl<K> Clone for SourceJournal<K> {
    fn clone(&self) -> Self {
        Self {
            origin: self.origin,
            capacity: self.capacity,
            state: Arc::clone(&self.state),
        }
    }
}

impl<K> SourceJournal<K> {
    pub fn try_new(capacity: usize) -> Result<Self, &'static str> {
        if !(1..=65_536).contains(&capacity)
            || std::mem::size_of::<SourceSpan<K>>()
                .checked_mul(capacity)
                .is_none_or(|bytes| bytes > 64 * 1024 * 1024)
        {
            return Err("source journal capacity exceeds its metadata budget");
        }
        let mut records = Vec::new();
        records
            .try_reserve_exact(capacity)
            .map_err(|_| "source journal allocation failed")?;
        Ok(Self {
            origin: Instant::now(),
            capacity,
            state: Arc::new(Mutex::new(State {
                records,
                dropped: 0,
                invalid_intervals: 0,
                external_events_dropped: 0,
                counter_overflow: false,
                poisoned: false,
            })),
        })
    }

    /// Preserve loss in the upstream bounded scheduler/delivery rings as well.
    pub fn note_external_loss(&self, dropped: u64, overflow: bool) {
        let mut state = self.state.lock().unwrap_or_else(|error| {
            let mut state = error.into_inner();
            state.poisoned = true;
            state
        });
        match state.external_events_dropped.checked_add(dropped) {
            Some(next) => state.external_events_dropped = next,
            None => state.counter_overflow = true,
        }
        state.counter_overflow |= overflow;
    }

    pub fn note_invalid_interval(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| {
            let mut state = error.into_inner();
            state.poisoned = true;
            state
        });
        increment(&mut state, true);
    }

    pub fn record(
        &self,
        key: Option<K>,
        stage: SourceStage,
        start: Instant,
        end: Instant,
        succeeded: bool,
    ) {
        let mut state = self.state.lock().unwrap_or_else(|error| {
            let mut state = error.into_inner();
            state.poisoned = true;
            state
        });
        if start < self.origin || end < start {
            increment(&mut state, true);
        } else if state.records.len() == self.capacity {
            increment(&mut state, false);
        } else {
            state.records.push(SourceSpan {
                key,
                stage,
                start,
                end,
                succeeded,
            });
        }
    }

    /// Call outside timed request paths after producer shutdown. A snapshot is
    /// a bounded copy; it neither drains physical work nor clears loss evidence.
    pub fn snapshot(&self) -> SourceSnapshot<K>
    where
        K: Clone,
    {
        let state = self.state.lock().unwrap_or_else(|error| {
            let mut state = error.into_inner();
            state.poisoned = true;
            state
        });
        SourceSnapshot {
            origin: self.origin,
            capacity: self.capacity,
            records: state.records.clone(),
            dropped: state.dropped,
            invalid_intervals: state.invalid_intervals,
            counter_overflow: state.counter_overflow,
            external_events_dropped: state.external_events_dropped,
            poisoned: state.poisoned,
        }
    }
}

fn increment<K>(state: &mut State<K>, invalid: bool) {
    let counter = if invalid {
        &mut state.invalid_intervals
    } else {
        &mut state.dropped
    };
    match counter.checked_add(1) {
        Some(next) => *counter = next,
        None => state.counter_overflow = true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_worker_time_survives_late_transport_and_capacity_loss() {
        let journal = SourceJournal::try_new(2).unwrap();
        let first = Instant::now();
        let second = Instant::now();
        journal.record(Some(2), SourceStage::PhysicalWorker, second, second, true);
        journal.record(Some(1), SourceStage::PhysicalWorker, first, first, true);
        journal.record(Some(3), SourceStage::PhysicalWorker, second, second, true);
        let snapshot = journal.snapshot();
        assert_eq!(snapshot.records[1].start, first);
        assert_eq!(snapshot.records[1].key, Some(1));
        assert_eq!(snapshot.dropped, 1);
        assert_eq!(snapshot.records.len(), 2);
    }

    #[test]
    fn invalid_or_poisoned_observation_is_visible_without_rewriting_records() {
        let journal = SourceJournal::try_new(2).unwrap();
        let now = Instant::now();
        journal.record(Some(1), SourceStage::NativeInvocation, now, now, true);
        journal.record(
            Some(2),
            SourceStage::NativeInvocation,
            now + std::time::Duration::from_nanos(1),
            now,
            true,
        );
        let shared = Arc::clone(&journal.state);
        assert!(std::thread::spawn(move || {
            let _guard = shared.lock().unwrap();
            panic!("observer poison");
        })
        .join()
        .is_err());
        journal.record(Some(3), SourceStage::NativeInvocation, now, now, false);
        let snapshot = journal.snapshot();
        assert!(snapshot.poisoned);
        assert_eq!(snapshot.invalid_intervals, 1);
        assert_eq!(snapshot.records.len(), 2);
        assert!(!snapshot.records[1].succeeded);
    }
}
