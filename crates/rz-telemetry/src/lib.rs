//! Passive counters and bounded samples for the CPU/mock runtime.
//!
//! Recording does not allocate or sort after construction. The owner supplies
//! monotonic elapsed durations and records each lifecycle event once. These
//! metrics neither validate nor control runtime state transitions.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;
use std::mem::size_of;
use std::time::Duration;

pub mod profile;

/// Failure to create the bounded sample storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapacityError {
    /// Element byte size overflows `usize` or exceeds the `Vec` byte limit.
    Overflow,
    /// The allocator could not reserve the validated sample storage.
    AllocationFailed,
}

impl fmt::Display for CapacityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => {
                formatter.write_str("telemetry sample capacity exceeds the byte limit")
            }
            Self::AllocationFailed => {
                formatter.write_str("telemetry sample storage allocation failed")
            }
        }
    }
}

impl std::error::Error for CapacityError {}

/// The logical terminal outcome of one admitted request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishKind {
    Completed,
    Canceled,
    Expired,
    Stale,
    Failed,
}

/// An observation emitted by the owner of a runtime lifecycle transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Event {
    Admitted,
    Rejected,
    /// One physical batch submission containing `items` logical requests.
    Dispatched {
        items: usize,
    },
    /// One physical batch is finished and its borrowed resources may be released.
    PhysicalCompleted,
    /// One logical request reached a terminal outcome.
    ///
    /// `latency` spans admission to logical completion, including cancellation
    /// and failure. It does not prove that search consumed or backed up a result.
    Finished {
        kind: FinishKind,
        latency: Duration,
    },
    ReceiverDropped,
    /// Physical executions whose resource ownership could not be safely drained.
    Quarantined {
        executions: usize,
    },
}

/// An immutable view of counters and the most recent retained samples.
///
/// Percentiles use the nearest-rank definition over the retained logical
/// completion latencies. The batch distribution also covers retained samples,
/// not all historical dispatches. A zero sample capacity still records counters.
/// A sticky `counter_overflow` marks saturated counters as inexact.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Snapshot {
    pub admitted: u64,
    pub rejected: u64,
    pub dispatches: u64,
    pub dispatched_items: u64,
    pub physical_completed: u64,
    pub finished: u64,
    pub completed: u64,
    pub canceled: u64,
    pub expired: u64,
    pub stale: u64,
    pub failed: u64,
    pub receiver_dropped: u64,
    pub quarantine_events: u64,
    pub quarantined_executions: u64,
    pub batch_distribution: BTreeMap<usize, usize>,
    pub batch_samples_total: u64,
    pub batch_samples_retained: usize,
    /// Samples overwritten by the ring or omitted due to a zero capacity.
    pub batch_samples_dropped: u64,
    pub latency_samples_total: u64,
    pub latency_samples_retained: usize,
    /// Samples overwritten by the ring or omitted due to a zero capacity.
    pub latency_samples_dropped: u64,
    pub latency_p50: Option<Duration>,
    pub latency_p95: Option<Duration>,
    pub latency_p99: Option<Duration>,
    pub counter_overflow: bool,
}

/// Single-owner metrics with a fixed storage budget for each sample stream.
///
/// Thread synchronization belongs to the caller. Both rings are allocated in
/// construction, with at most `sample_capacity` retained entries each. A snapshot
/// may allocate and sort; callers should take it outside request handling paths.
#[derive(Debug)]
pub struct Metrics {
    counters: Counters,
    batches: Ring<usize>,
    latencies: Ring<Duration>,
    batch_samples_dropped: u64,
    latency_samples_dropped: u64,
    counter_overflow: bool,
}

impl Metrics {
    /// Convenience constructor for trusted, bounded capacity configurations.
    ///
    /// # Panics
    ///
    /// Panics when the capacity exceeds the byte limit or storage allocation
    /// fails. Use [`Metrics::try_new`] for runtime or external configuration.
    pub fn new(sample_capacity: usize) -> Self {
        Self::try_new(sample_capacity).expect("trusted telemetry sample capacity must fit")
    }

    /// Check both sample layouts before allocating either ring.
    ///
    /// Allocation failures are returned as typed errors. This checks representable
    /// storage, not the owner's operational memory budget, which also includes
    /// runtime metadata and temporary snapshot storage.
    pub fn try_new(sample_capacity: usize) -> Result<Self, CapacityError> {
        validate_capacity::<usize>(sample_capacity)?;
        validate_capacity::<Duration>(sample_capacity)?;
        Ok(Self {
            counters: Counters::default(),
            batches: Ring::try_new(sample_capacity)?,
            latencies: Ring::try_new(sample_capacity)?,
            batch_samples_dropped: 0,
            latency_samples_dropped: 0,
            counter_overflow: false,
        })
    }

    /// Record one observation without changing runtime behavior.
    pub fn record(&mut self, event: Event) {
        let counters = &mut self.counters;
        let overflow = &mut self.counter_overflow;
        match event {
            Event::Admitted => add(&mut counters.admitted, 1, overflow),
            Event::Rejected => add(&mut counters.rejected, 1, overflow),
            Event::Dispatched { items } => {
                add(&mut counters.dispatches, 1, overflow);
                add_usize(&mut counters.dispatched_items, items, overflow);
                if self.batches.push(items) {
                    add(&mut self.batch_samples_dropped, 1, overflow);
                }
            }
            Event::PhysicalCompleted => add(&mut counters.physical_completed, 1, overflow),
            Event::Finished { kind, latency } => {
                add(&mut counters.finished, 1, overflow);
                let terminal_counter = match kind {
                    FinishKind::Completed => &mut counters.completed,
                    FinishKind::Canceled => &mut counters.canceled,
                    FinishKind::Expired => &mut counters.expired,
                    FinishKind::Stale => &mut counters.stale,
                    FinishKind::Failed => &mut counters.failed,
                };
                add(terminal_counter, 1, overflow);
                if self.latencies.push(latency) {
                    add(&mut self.latency_samples_dropped, 1, overflow);
                }
            }
            Event::ReceiverDropped => add(&mut counters.receiver_dropped, 1, overflow),
            Event::Quarantined { executions } => {
                add(&mut counters.quarantine_events, 1, overflow);
                add_usize(&mut counters.quarantined_executions, executions, overflow);
            }
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        let counters = &self.counters;
        let mut latencies = self.latencies.samples.clone();
        latencies.sort_unstable();
        let mut batch_distribution = BTreeMap::new();
        for &items in &self.batches.samples {
            *batch_distribution.entry(items).or_insert(0) += 1;
        }
        Snapshot {
            admitted: counters.admitted,
            rejected: counters.rejected,
            dispatches: counters.dispatches,
            dispatched_items: counters.dispatched_items,
            physical_completed: counters.physical_completed,
            finished: counters.finished,
            completed: counters.completed,
            canceled: counters.canceled,
            expired: counters.expired,
            stale: counters.stale,
            failed: counters.failed,
            receiver_dropped: counters.receiver_dropped,
            quarantine_events: counters.quarantine_events,
            quarantined_executions: counters.quarantined_executions,
            batch_distribution,
            batch_samples_total: counters.dispatches,
            batch_samples_retained: self.batches.samples.len(),
            batch_samples_dropped: self.batch_samples_dropped,
            latency_samples_total: counters.finished,
            latency_samples_retained: latencies.len(),
            latency_samples_dropped: self.latency_samples_dropped,
            latency_p50: percentile(&latencies, 50),
            latency_p95: percentile(&latencies, 95),
            latency_p99: percentile(&latencies, 99),
            counter_overflow: self.counter_overflow,
        }
    }
}

#[derive(Debug, Default)]
struct Counters {
    admitted: u64,
    rejected: u64,
    dispatches: u64,
    dispatched_items: u64,
    physical_completed: u64,
    finished: u64,
    completed: u64,
    canceled: u64,
    expired: u64,
    stale: u64,
    failed: u64,
    receiver_dropped: u64,
    quarantine_events: u64,
    quarantined_executions: u64,
}

/// Preallocated storage that replaces the oldest retained sample when full.
#[derive(Debug)]
struct Ring<T> {
    samples: Vec<T>,
    limit: usize,
    next: usize,
}

impl<T> Ring<T> {
    fn try_new(limit: usize) -> Result<Self, CapacityError> {
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(limit)
            .map_err(|_| CapacityError::AllocationFailed)?;
        Ok(Self {
            samples,
            limit,
            next: 0,
        })
    }

    /// Return true if one sample was omitted or an older sample was overwritten.
    fn push(&mut self, sample: T) -> bool {
        if self.limit == 0 {
            return true;
        }
        if self.samples.len() < self.limit {
            self.samples.push(sample);
            return false;
        }
        self.samples[self.next] = sample;
        self.next = if self.next == self.limit - 1 {
            0
        } else {
            self.next + 1
        };
        true
    }
}

fn validate_capacity<T>(capacity: usize) -> Result<(), CapacityError> {
    let bytes = capacity
        .checked_mul(size_of::<T>())
        .ok_or(CapacityError::Overflow)?;
    if bytes > isize::MAX as usize {
        return Err(CapacityError::Overflow);
    }
    Ok(())
}

fn add(counter: &mut u64, amount: u64, overflow: &mut bool) {
    match counter.checked_add(amount) {
        Some(value) => *counter = value,
        None => {
            *counter = u64::MAX;
            *overflow = true;
        }
    }
}

fn add_usize(counter: &mut u64, amount: usize, overflow: &mut bool) {
    match u64::try_from(amount) {
        Ok(amount) => add(counter, amount, overflow),
        Err(_) => {
            *counter = u64::MAX;
            *overflow = true;
        }
    }
}

fn percentile(sorted: &[Duration], percentage: usize) -> Option<Duration> {
    if sorted.is_empty() {
        return None;
    }
    // Split the product to avoid overflowing on len * percentage.
    let remainder_product = (sorted.len() % 100) * percentage;
    let rank = (sorted.len() / 100) * percentage + remainder_product.div_ceil(100);
    Some(sorted[rank - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finish(metrics: &mut Metrics, kind: FinishKind, milliseconds: u64) {
        metrics.record(Event::Finished {
            kind,
            latency: Duration::from_millis(milliseconds),
        });
    }

    #[test]
    fn empty_snapshot_has_no_invented_samples() {
        assert_eq!(Metrics::new(4).snapshot(), Snapshot::default());
    }

    #[test]
    fn impossible_capacity_is_rejected_before_either_ring_is_allocated() {
        // The second case fits the usize ring but exceeds the Duration ring's
        // isize byte limit. Prevalidation must reject it before any allocation.
        let duration_limit_exceeded = isize::MAX as usize / size_of::<Duration>() + 1;
        for capacity in [usize::MAX, duration_limit_exceeded] {
            assert!(matches!(
                Metrics::try_new(capacity),
                Err(CapacityError::Overflow)
            ));
        }
    }

    #[test]
    fn logical_outcomes_and_physical_completion_are_independent() {
        let mut metrics = Metrics::new(8);
        for _ in 0..5 {
            metrics.record(Event::Admitted);
        }
        metrics.record(Event::Rejected);
        metrics.record(Event::Dispatched { items: 4 });
        for kind in [
            FinishKind::Completed,
            FinishKind::Canceled,
            FinishKind::Expired,
            FinishKind::Stale,
            FinishKind::Failed,
        ] {
            finish(&mut metrics, kind, 3);
        }
        // Logical termination does not release a physical backend borrow.
        assert_eq!(metrics.snapshot().physical_completed, 0);
        metrics.record(Event::PhysicalCompleted);
        metrics.record(Event::ReceiverDropped);
        metrics.record(Event::Quarantined { executions: 2 });
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.admitted, 5);
        assert_eq!(snapshot.rejected, 1);
        assert_eq!(snapshot.dispatches, 1);
        assert_eq!(snapshot.dispatched_items, 4);
        assert_eq!(snapshot.physical_completed, 1);
        assert_eq!(snapshot.finished, 5);
        assert_eq!(snapshot.completed, 1);
        assert_eq!(snapshot.canceled, 1);
        assert_eq!(snapshot.expired, 1);
        assert_eq!(snapshot.stale, 1);
        assert_eq!(snapshot.failed, 1);
        assert_eq!(snapshot.receiver_dropped, 1);
        assert_eq!(snapshot.quarantine_events, 1);
        assert_eq!(snapshot.quarantined_executions, 2);
        assert!(!snapshot.counter_overflow);
    }

    #[test]
    fn ring_eviction_is_bounded_and_reported_for_both_sample_streams() {
        let mut metrics = Metrics::new(3);
        let latency_allocation = metrics.latencies.samples.capacity();
        let batch_allocation = metrics.batches.samples.capacity();
        for i in 1..=10 {
            finish(&mut metrics, FinishKind::Completed, i);
            metrics.record(Event::Dispatched {
                items: if i < 8 { 99 } else { i as usize % 2 + 1 },
            });
        }
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.latency_samples_total, 10);
        assert_eq!(snapshot.latency_samples_retained, 3);
        assert_eq!(snapshot.latency_samples_dropped, 7);
        assert_eq!(snapshot.latency_p50, Some(Duration::from_millis(9)));
        assert_eq!(snapshot.latency_p95, Some(Duration::from_millis(10)));
        assert_eq!(snapshot.latency_p99, Some(Duration::from_millis(10)));
        assert_eq!(snapshot.batch_samples_total, 10);
        assert_eq!(snapshot.batch_samples_retained, 3);
        assert_eq!(snapshot.batch_samples_dropped, 7);
        assert_eq!(
            snapshot.batch_distribution,
            BTreeMap::from([(1, 2), (2, 1)])
        );
        assert_eq!(metrics.latencies.samples.capacity(), latency_allocation);
        assert_eq!(metrics.batches.samples.capacity(), batch_allocation);
    }

    #[test]
    fn zero_capacity_keeps_counters_and_reports_all_samples_as_dropped() {
        let mut metrics = Metrics::try_new(0).expect("zero capacity requires no sample allocation");
        finish(&mut metrics, FinishKind::Canceled, 7);
        metrics.record(Event::Dispatched { items: 2 });
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.canceled, 1);
        assert_eq!(snapshot.latency_samples_total, 1);
        assert_eq!(snapshot.latency_samples_retained, 0);
        assert_eq!(snapshot.latency_samples_dropped, 1);
        assert_eq!(snapshot.latency_p50, None);
        assert_eq!(snapshot.latency_p95, None);
        assert_eq!(snapshot.latency_p99, None);
        assert_eq!(snapshot.batch_samples_total, 1);
        assert_eq!(snapshot.batch_samples_retained, 0);
        assert_eq!(snapshot.batch_samples_dropped, 1);
        assert!(snapshot.batch_distribution.is_empty());
    }

    #[test]
    fn nearest_rank_percentiles_sort_only_the_snapshot_and_preserve_durations() {
        let mut metrics = Metrics::new(100);
        for nanos in (1..=100).rev() {
            metrics.record(Event::Finished {
                kind: FinishKind::Completed,
                latency: Duration::from_nanos(nanos),
            });
        }
        let before = metrics.latencies.samples.clone();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.latency_p50, Some(Duration::from_nanos(50)));
        assert_eq!(snapshot.latency_p95, Some(Duration::from_nanos(95)));
        assert_eq!(snapshot.latency_p99, Some(Duration::from_nanos(99)));
        assert_eq!(metrics.latencies.samples, before);
        assert_eq!(snapshot, metrics.snapshot());
    }

    #[test]
    fn counters_saturate_and_keep_overflow_visible_after_later_events() {
        let mut metrics = Metrics::new(1);
        metrics.counters.admitted = u64::MAX;
        metrics.counters.dispatched_items = u64::MAX - 1;
        metrics.batch_samples_dropped = u64::MAX;
        metrics.record(Event::Admitted);
        metrics.record(Event::Dispatched { items: 2 });
        metrics.record(Event::Dispatched { items: 1 });
        metrics.record(Event::Rejected);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.admitted, u64::MAX);
        assert_eq!(snapshot.dispatched_items, u64::MAX);
        assert_eq!(snapshot.batch_samples_dropped, u64::MAX);
        assert_eq!(snapshot.rejected, 1);
        assert!(snapshot.counter_overflow);
        assert_eq!(snapshot.batch_distribution, BTreeMap::from([(1, 1)]));
    }

    #[test]
    fn one_retained_sample_including_zero_and_max_duration_has_all_percentiles() {
        let mut metrics = Metrics::new(1);
        for latency in [Duration::ZERO, Duration::MAX] {
            metrics.record(Event::Finished {
                kind: FinishKind::Completed,
                latency,
            });
            let snapshot = metrics.snapshot();
            assert_eq!(snapshot.latency_p50, Some(latency));
            assert_eq!(snapshot.latency_p95, Some(latency));
            assert_eq!(snapshot.latency_p99, Some(latency));
        }
    }
}
