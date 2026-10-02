//! Bounded, passive trace validation and CPU/mock profiling.
//!
//! Keys are supplied by the caller; this module defines no engine request or
//! execution contract. Rejected observations do not direct scheduler behavior.
//! Recently retired keys are remembered within a bounded duplicate-check window.

use super::{add, percentile, validate_capacity, CapacityError, FinishKind, Ring};
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Timepoint {
    pub domain: u64,
    pub elapsed: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfileConfig {
    pub max_requests: usize,
    pub max_executions: usize,
    /// Retained samples per stage, and retained batch-size samples.
    pub sample_capacity: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(usize)]
pub enum Stage {
    CpuPreparation,
    Queue,
    Dispatch,
    /// Actual provider transfer hooks only; CPU/mock traces leave this absent.
    Transfer,
    BackendPhysical,
    Validation,
    Backup,
    Output,
    EndToEnd,
}

impl Stage {
    pub const ALL: [Self; 9] = [
        Self::CpuPreparation,
        Self::Queue,
        Self::Dispatch,
        Self::Transfer,
        Self::BackendPhysical,
        Self::Validation,
        Self::Backup,
        Self::Output,
        Self::EndToEnd,
    ];
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(usize)]
pub enum TraceClass {
    Cold,
    Warm,
    ParentChild,
    Sibling,
    Transposition,
    Eviction,
    Long,
}

impl TraceClass {
    pub const ALL: [Self; 7] = [
        Self::Cold,
        Self::Warm,
        Self::ParentChild,
        Self::Sibling,
        Self::Transposition,
        Self::Eviction,
        Self::Long,
    ];
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(usize)]
pub enum BackendKind {
    CpuMock,
    Cpu,
    Gpu,
}

impl BackendKind {
    pub const ALL: [Self; 3] = [Self::CpuMock, Self::Cpu, Self::Gpu];
}

/// Caller-reported reservation ledger, not observed process RSS or actual VRAM.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryReservations {
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub pinned_bytes: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SampleSummary {
    pub total: u64,
    pub retained: usize,
    pub dropped: u64,
    pub p50: Option<Duration>,
    pub p95: Option<Duration>,
    pub p99: Option<Duration>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageReport {
    pub stage: Stage,
    pub latency: SampleSummary,
    pub by_class: BTreeMap<TraceClass, SampleSummary>,
    pub by_backend: BTreeMap<BackendKind, SampleSummary>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LogicalOutcomes {
    pub completed: u64,
    pub canceled: u64,
    pub expired: u64,
    pub stale: u64,
    pub failed: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Completeness {
    pub stage_samples_dropped: bool,
    pub batch_samples_dropped: bool,
    pub tracking_capacity_rejections: u64,
    pub request_ids_evicted: u64,
    pub execution_ids_evicted: u64,
    pub observation_errors: u64,
    pub domain_errors: u64,
    pub time_errors: u64,
    pub active_requests: usize,
    pub active_executions: usize,
    pub external_events_dropped: u64,
    /// Keys must never be reused, including after the bounded tombstone window.
    pub caller_ids_must_be_unique: bool,
    pub duplicate_checks_cover_all_ids: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProfileReport {
    pub domain: u64,
    pub first_point: Option<Timepoint>,
    pub last_point: Option<Timepoint>,
    /// Earliest accepted span start through the last accepted observation.
    pub wall_span: Option<Duration>,
    pub stages: Vec<StageReport>,
    pub batch_distribution: BTreeMap<usize, usize>,
    pub batch_samples_total: u64,
    pub batch_samples_retained: usize,
    pub batch_samples_dropped: u64,
    pub logical: LogicalOutcomes,
    pub rejected_requests: u64,
    pub physical_executions_started: u64,
    pub physical_executions_completed: u64,
    pub physical_items_started: u64,
    pub physical_items_completed: u64,
    pub consumed_requests: u64,
    pub physical_items_consumed: u64,
    pub cache_consumed_requests: u64,
    /// Completed physical items explicitly finalized without consumption.
    pub unused_items_finalized: u64,
    /// Completed but not retired physical items, still eligible for consumption.
    pub completed_unconsumed_items: u64,
    pub duplicate_finishes_rejected: u64,
    pub duplicate_consumes_rejected: u64,
    /// Based on `wall_span`, never a sum of stage durations.
    pub physical_items_per_second: Option<f64>,
    pub consumed_requests_per_second: Option<f64>,
    pub reserved_current: MemoryReservations,
    pub reserved_peak: MemoryReservations,
    pub completeness: Completeness,
    pub counter_overflow: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceError {
    DomainMismatch {
        expected: u64,
        observed: u64,
    },
    OutOfOrder {
        previous: Duration,
        observed: Duration,
    },
    InvalidInterval {
        start: Duration,
        end: Duration,
    },
    UnknownRequest,
    UnknownExecution,
    RequestCapacity,
    ExecutionCapacity,
    DuplicateRequest,
    DuplicateExecution,
    DuplicateFinish,
    DuplicatePhysicalFinish,
    DuplicateConsume,
    DuplicateStage {
        stage: Stage,
    },
    RequestNotFinished,
    RequestNotCompleted,
    RequestNotConsumed,
    ExecutionNotCompleted,
    ExecutionItemsExhausted,
    EmptyExecution,
    ManagedStage {
        stage: Stage,
    },
    TransferUnavailable,
}

impl fmt::Display for TraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid profiling observation: {self:?}")
    }
}

impl std::error::Error for TraceError {}

/// Finite tracking and fixed rings; generic keys remain the caller's IDs.
///
/// Duplicate validation is an observation check, not an exactly-once authority
/// for the engine. Keys must be unique for the entire caller trace. The report
/// exposes tombstone eviction and the resulting check-window limit.
#[derive(Debug)]
pub struct TraceCollector<R: Eq, E: Eq> {
    config: ProfileConfig,
    domain: u64,
    first: Option<Duration>,
    last: Option<Duration>,
    requests: Vec<RequestState<R>>,
    executions: Vec<ExecutionState<E>>,
    retired_requests: Ring<RetiredRequest<R>>,
    retired_executions: Ring<E>,
    stages: Vec<StageSamples>,
    batches: Ring<usize>,
    counters: Counters,
    reserved_current: MemoryReservations,
    reserved_peak: MemoryReservations,
    counter_overflow: bool,
}

impl<R: Eq, E: Eq> TraceCollector<R, E> {
    pub fn try_new(config: ProfileConfig, domain: u64) -> Result<Self, CapacityError> {
        // Check every layout before trying any allocation, including all stages.
        validate_capacity::<RequestState<R>>(config.max_requests)?;
        validate_capacity::<RetiredRequest<R>>(config.max_requests)?;
        validate_capacity::<ExecutionState<E>>(config.max_executions)?;
        validate_capacity::<E>(config.max_executions)?;
        validate_capacity::<Sample>(
            config
                .sample_capacity
                .checked_mul(Stage::ALL.len())
                .ok_or(CapacityError::Overflow)?,
        )?;
        validate_capacity::<usize>(config.sample_capacity)?;
        let requests = allocate(config.max_requests)?;
        let executions = allocate(config.max_executions)?;
        let retired_requests = Ring::try_new(config.max_requests)?;
        let retired_executions = Ring::try_new(config.max_executions)?;
        let mut stages = allocate(Stage::ALL.len())?;
        for _ in Stage::ALL {
            stages.push(StageSamples::try_new(config.sample_capacity)?);
        }
        Ok(Self {
            config,
            domain,
            first: None,
            last: None,
            requests,
            executions,
            retired_requests,
            retired_executions,
            stages,
            batches: Ring::try_new(config.sample_capacity)?,
            counters: Counters::default(),
            reserved_current: MemoryReservations::default(),
            reserved_peak: MemoryReservations::default(),
            counter_overflow: false,
        })
    }

    pub fn begin_request(
        &mut self,
        id: R,
        class: TraceClass,
        at: Timepoint,
    ) -> Result<(), TraceError> {
        self.checked(|this| {
            this.check_point(at)?;
            if this.requests.iter().any(|request| request.id == id)
                || this
                    .retired_requests
                    .samples
                    .iter()
                    .any(|request| request.id == id)
            {
                return Err(TraceError::DuplicateRequest);
            }
            if this.requests.len() == this.config.max_requests {
                return Err(TraceError::RequestCapacity);
            }
            this.requests.push(RequestState {
                id,
                class,
                start: at.elapsed,
                finished: None,
                consumed: false,
                stages: [false; Stage::ALL.len()],
            });
            this.commit_point(at);
            Ok(())
        })
    }

    /// Record one explicit request phase; phases may overlap.
    pub fn record_stage(
        &mut self,
        request: &R,
        stage: Stage,
        start: Timepoint,
        end: Timepoint,
    ) -> Result<(), TraceError> {
        self.checked(|this| {
            let duration = this.check_interval(start, end)?;
            let index = this.request_index(request)?;
            let state = &this.requests[index];
            if matches!(
                stage,
                Stage::Dispatch | Stage::Transfer | Stage::BackendPhysical | Stage::EndToEnd
            ) {
                return Err(TraceError::ManagedStage { stage });
            }
            this.check_not_before(start.elapsed, state.start)?;
            if state.stages[stage as usize] {
                return Err(TraceError::DuplicateStage { stage });
            }
            if stage == Stage::Backup && !state.consumed {
                return Err(TraceError::RequestNotConsumed);
            }
            let class = state.class;
            this.requests[index].stages[stage as usize] = true;
            this.sample(stage, class, None, duration);
            this.commit_interval(start, end);
            Ok(())
        })
    }

    pub fn finish_request(
        &mut self,
        request: &R,
        kind: FinishKind,
        at: Timepoint,
    ) -> Result<(), TraceError> {
        self.checked(|this| {
            this.check_point(at)?;
            if this
                .retired_requests
                .samples
                .iter()
                .any(|state| &state.id == request)
            {
                return Err(TraceError::DuplicateFinish);
            }
            let index = this.request_index(request)?;
            if this.requests[index].finished.is_some() {
                return Err(TraceError::DuplicateFinish);
            }
            this.requests[index].finished = Some(kind);
            let counter = match kind {
                FinishKind::Completed => &mut this.counters.logical.completed,
                FinishKind::Canceled => &mut this.counters.logical.canceled,
                FinishKind::Expired => &mut this.counters.logical.expired,
                FinishKind::Stale => &mut this.counters.logical.stale,
                FinishKind::Failed => &mut this.counters.logical.failed,
            };
            add(counter, 1, &mut this.counter_overflow);
            this.commit_point(at);
            Ok(())
        })
    }

    /// Observe one actual successful consumer/backup, after logical completion.
    ///
    /// `None` denotes a cache-only result with no current physical execution.
    /// The caller must validate the request's execution association and search
    /// consume token; this passive collector does not own those engine contracts.
    pub fn consume_request(
        &mut self,
        request: &R,
        execution: Option<&E>,
        at: Timepoint,
    ) -> Result<(), TraceError> {
        self.checked(|this| {
            this.check_point(at)?;
            if let Some(retired) = this
                .retired_requests
                .samples
                .iter()
                .find(|state| &state.id == request)
            {
                return Err(if retired.consumed {
                    TraceError::DuplicateConsume
                } else {
                    TraceError::RequestNotCompleted
                });
            }
            let index = this.request_index(request)?;
            if this.requests[index].consumed {
                return Err(TraceError::DuplicateConsume);
            }
            if this.requests[index].finished != Some(FinishKind::Completed) {
                return Err(TraceError::RequestNotCompleted);
            }
            if let Some(execution) = execution {
                let index = this.execution_index(execution)?;
                let state = &mut this.executions[index];
                if !state.completed {
                    return Err(TraceError::ExecutionNotCompleted);
                }
                if state.consumed == state.items {
                    return Err(TraceError::ExecutionItemsExhausted);
                }
                state.consumed += 1;
                add(
                    &mut this.counters.physical_items_consumed,
                    1,
                    &mut this.counter_overflow,
                );
            } else {
                add(
                    &mut this.counters.cache_consumed_requests,
                    1,
                    &mut this.counter_overflow,
                );
            }
            this.requests[index].consumed = true;
            add(
                &mut this.counters.consumed_requests,
                1,
                &mut this.counter_overflow,
            );
            this.commit_point(at);
            Ok(())
        })
    }

    /// Record end-to-end through output and release active request tracking.
    pub fn close_request(&mut self, request: &R, at: Timepoint) -> Result<(), TraceError> {
        self.checked(|this| {
            this.check_point(at)?;
            let index = this.request_index(request)?;
            let state = &this.requests[index];
            if state.finished.is_none() {
                return Err(TraceError::RequestNotFinished);
            }
            let duration = at.elapsed - state.start;
            let class = state.class;
            let state = this.requests.swap_remove(index);
            if this.retired_requests.push(RetiredRequest {
                id: state.id,
                consumed: state.consumed,
            }) {
                add(
                    &mut this.counters.request_ids_evicted,
                    1,
                    &mut this.counter_overflow,
                );
            }
            this.sample(Stage::EndToEnd, class, None, duration);
            this.commit_point(at);
            Ok(())
        })
    }

    /// `at` is the backend physical phase start, commonly dispatch completion.
    pub fn begin_execution(
        &mut self,
        id: E,
        class: TraceClass,
        backend: BackendKind,
        items: usize,
        at: Timepoint,
    ) -> Result<(), TraceError> {
        self.checked(|this| {
            this.check_point(at)?;
            if items == 0 {
                return Err(TraceError::EmptyExecution);
            }
            if this.executions.iter().any(|execution| execution.id == id)
                || this
                    .retired_executions
                    .samples
                    .iter()
                    .any(|execution| execution == &id)
            {
                return Err(TraceError::DuplicateExecution);
            }
            if this.executions.len() == this.config.max_executions {
                return Err(TraceError::ExecutionCapacity);
            }
            this.executions.push(ExecutionState {
                id,
                class,
                backend,
                items,
                consumed: 0,
                start: at.elapsed,
                completed: false,
                dispatch_recorded: false,
                transfer_recorded: false,
            });
            add(
                &mut this.counters.physical_executions_started,
                1,
                &mut this.counter_overflow,
            );
            add_count(
                &mut this.counters.physical_items_started,
                items,
                &mut this.counter_overflow,
            );
            if this.batches.push(items) {
                add(
                    &mut this.counters.batch_samples_dropped,
                    1,
                    &mut this.counter_overflow,
                );
            }
            this.commit_point(at);
            Ok(())
        })
    }

    /// Explicit submit interval, ending at backend phase start.
    /// Its earlier start is valid even though `begin_execution` was already seen.
    pub fn record_dispatch(
        &mut self,
        execution: &E,
        start: Timepoint,
        end: Timepoint,
    ) -> Result<(), TraceError> {
        self.checked(|this| {
            let duration = this.check_interval(start, end)?;
            let index = this.execution_index(execution)?;
            let state = &this.executions[index];
            if state.dispatch_recorded {
                return Err(TraceError::DuplicateStage {
                    stage: Stage::Dispatch,
                });
            }
            if end.elapsed > state.start {
                return Err(TraceError::InvalidInterval {
                    start: end.elapsed,
                    end: state.start,
                });
            }
            let (class, backend) = (state.class, state.backend);
            this.executions[index].dispatch_recorded = true;
            this.sample(Stage::Dispatch, class, Some(backend), duration);
            this.commit_interval(start, end);
            Ok(())
        })
    }

    /// Actual transfer observations require an explicitly selected GPU backend.
    pub fn record_transfer(
        &mut self,
        execution: &E,
        start: Timepoint,
        end: Timepoint,
    ) -> Result<(), TraceError> {
        self.checked(|this| {
            let duration = this.check_interval(start, end)?;
            let index = this.execution_index(execution)?;
            let state = &this.executions[index];
            if state.backend != BackendKind::Gpu {
                return Err(TraceError::TransferUnavailable);
            }
            if state.completed || state.transfer_recorded {
                return Err(TraceError::DuplicateStage {
                    stage: Stage::Transfer,
                });
            }
            this.check_not_before(start.elapsed, state.start)?;
            let class = state.class;
            this.executions[index].transfer_recorded = true;
            this.sample(Stage::Transfer, class, Some(BackendKind::Gpu), duration);
            this.commit_interval(start, end);
            Ok(())
        })
    }

    /// A canceled logical request does not prevent a late physical completion.
    pub fn finish_execution(&mut self, execution: &E, at: Timepoint) -> Result<(), TraceError> {
        self.checked(|this| {
            this.check_point(at)?;
            if this
                .retired_executions
                .samples
                .iter()
                .any(|id| id == execution)
            {
                return Err(TraceError::DuplicatePhysicalFinish);
            }
            let index = this.execution_index(execution)?;
            let state = &this.executions[index];
            if state.completed {
                return Err(TraceError::DuplicatePhysicalFinish);
            }
            let (class, backend, duration, items) = (
                state.class,
                state.backend,
                at.elapsed - state.start,
                state.items,
            );
            this.executions[index].completed = true;
            add(
                &mut this.counters.physical_executions_completed,
                1,
                &mut this.counter_overflow,
            );
            add_count(
                &mut this.counters.physical_items_completed,
                items,
                &mut this.counter_overflow,
            );
            this.sample(Stage::BackendPhysical, class, Some(backend), duration);
            this.commit_point(at);
            Ok(())
        })
    }

    /// Finalize unused work only once the caller knows no consumers remain.
    pub fn retire_execution(&mut self, execution: &E, at: Timepoint) -> Result<(), TraceError> {
        self.checked(|this| {
            this.check_point(at)?;
            let index = this.execution_index(execution)?;
            if !this.executions[index].completed {
                return Err(TraceError::ExecutionNotCompleted);
            }
            let state = this.executions.swap_remove(index);
            add_count(
                &mut this.counters.unused_items_finalized,
                state.items - state.consumed,
                &mut this.counter_overflow,
            );
            if this.retired_executions.push(state.id) {
                add(
                    &mut this.counters.execution_ids_evicted,
                    1,
                    &mut this.counter_overflow,
                );
            }
            this.commit_point(at);
            Ok(())
        })
    }

    pub fn record_rejection(&mut self, at: Timepoint) -> Result<(), TraceError> {
        self.checked(|this| {
            this.check_point(at)?;
            add(
                &mut this.counters.rejected_requests,
                1,
                &mut this.counter_overflow,
            );
            this.commit_point(at);
            Ok(())
        })
    }

    pub fn observe_reservations(
        &mut self,
        reserved: MemoryReservations,
        at: Timepoint,
    ) -> Result<(), TraceError> {
        self.checked(|this| {
            this.check_point(at)?;
            this.reserved_current = reserved;
            this.reserved_peak.host_bytes = this.reserved_peak.host_bytes.max(reserved.host_bytes);
            this.reserved_peak.device_bytes =
                this.reserved_peak.device_bytes.max(reserved.device_bytes);
            this.reserved_peak.pinned_bytes =
                this.reserved_peak.pinned_bytes.max(reserved.pinned_bytes);
            this.commit_point(at);
            Ok(())
        })
    }

    /// Observe a ledger's retained high-water mark without replacing current use.
    /// This can include transient receipt reservations between drained events.
    pub fn observe_reservation_peak(
        &mut self,
        peak: MemoryReservations,
        at: Timepoint,
    ) -> Result<(), TraceError> {
        self.checked(|this| {
            this.check_point(at)?;
            this.reserved_peak.host_bytes = this.reserved_peak.host_bytes.max(peak.host_bytes);
            this.reserved_peak.device_bytes =
                this.reserved_peak.device_bytes.max(peak.device_bytes);
            this.reserved_peak.pinned_bytes =
                this.reserved_peak.pinned_bytes.max(peak.pinned_bytes);
            this.commit_point(at);
            Ok(())
        })
    }

    /// Report event-ring loss from the caller's observation transport.
    pub fn note_event_loss(&mut self, count: u64) {
        add(
            &mut self.counters.external_events_dropped,
            count,
            &mut self.counter_overflow,
        );
    }

    pub fn report(&self) -> ProfileReport {
        let counters = &self.counters;
        let stages: Vec<_> = Stage::ALL
            .into_iter()
            .map(|stage| self.stages[stage as usize].report(stage))
            .collect();
        let mut batch_distribution = BTreeMap::new();
        for &items in &self.batches.samples {
            *batch_distribution.entry(items).or_insert(0) += 1;
        }
        let wall_span = self.first.zip(self.last).map(|(start, end)| end - start);
        let seconds = wall_span
            .map(|duration| duration.as_secs_f64())
            .filter(|seconds| *seconds > 0.0);
        let mut counter_overflow = self.counter_overflow;
        let mut completed_unconsumed_items = 0;
        for execution in self
            .executions
            .iter()
            .filter(|execution| execution.completed)
        {
            add_count(
                &mut completed_unconsumed_items,
                execution.items - execution.consumed,
                &mut counter_overflow,
            );
        }
        ProfileReport {
            domain: self.domain,
            first_point: self.first.map(|elapsed| Timepoint {
                domain: self.domain,
                elapsed,
            }),
            last_point: self.last.map(|elapsed| Timepoint {
                domain: self.domain,
                elapsed,
            }),
            wall_span,
            stages,
            batch_distribution,
            batch_samples_total: counters.physical_executions_started,
            batch_samples_retained: self.batches.samples.len(),
            batch_samples_dropped: counters.batch_samples_dropped,
            logical: counters.logical,
            rejected_requests: counters.rejected_requests,
            physical_executions_started: counters.physical_executions_started,
            physical_executions_completed: counters.physical_executions_completed,
            physical_items_started: counters.physical_items_started,
            physical_items_completed: counters.physical_items_completed,
            consumed_requests: counters.consumed_requests,
            physical_items_consumed: counters.physical_items_consumed,
            cache_consumed_requests: counters.cache_consumed_requests,
            unused_items_finalized: counters.unused_items_finalized,
            completed_unconsumed_items,
            duplicate_finishes_rejected: counters.duplicate_finishes_rejected,
            duplicate_consumes_rejected: counters.duplicate_consumes_rejected,
            physical_items_per_second: seconds
                .map(|seconds| counters.physical_items_completed as f64 / seconds),
            consumed_requests_per_second: seconds
                .map(|seconds| counters.consumed_requests as f64 / seconds),
            reserved_current: self.reserved_current,
            reserved_peak: self.reserved_peak,
            completeness: Completeness {
                stage_samples_dropped: stages_have_loss(&self.stages),
                batch_samples_dropped: counters.batch_samples_dropped != 0,
                tracking_capacity_rejections: counters.tracking_capacity_rejections,
                request_ids_evicted: counters.request_ids_evicted,
                execution_ids_evicted: counters.execution_ids_evicted,
                observation_errors: counters.observation_errors,
                domain_errors: counters.domain_errors,
                time_errors: counters.time_errors,
                active_requests: self.requests.len(),
                active_executions: self.executions.len(),
                external_events_dropped: counters.external_events_dropped,
                caller_ids_must_be_unique: true,
                duplicate_checks_cover_all_ids: counters.request_ids_evicted == 0
                    && counters.execution_ids_evicted == 0,
            },
            counter_overflow,
        }
    }

    fn checked(
        &mut self,
        update: impl FnOnce(&mut Self) -> Result<(), TraceError>,
    ) -> Result<(), TraceError> {
        let result = update(self);
        if let Err(error) = result {
            add(
                &mut self.counters.observation_errors,
                1,
                &mut self.counter_overflow,
            );
            let counter = match error {
                TraceError::DomainMismatch { .. } => Some(&mut self.counters.domain_errors),
                TraceError::OutOfOrder { .. } | TraceError::InvalidInterval { .. } => {
                    Some(&mut self.counters.time_errors)
                }
                TraceError::RequestCapacity | TraceError::ExecutionCapacity => {
                    Some(&mut self.counters.tracking_capacity_rejections)
                }
                TraceError::DuplicateFinish | TraceError::DuplicatePhysicalFinish => {
                    Some(&mut self.counters.duplicate_finishes_rejected)
                }
                TraceError::DuplicateConsume => {
                    Some(&mut self.counters.duplicate_consumes_rejected)
                }
                _ => None,
            };
            if let Some(counter) = counter {
                add(counter, 1, &mut self.counter_overflow);
            }
        }
        result
    }

    fn request_index(&self, id: &R) -> Result<usize, TraceError> {
        self.requests
            .iter()
            .position(|state| &state.id == id)
            .ok_or(TraceError::UnknownRequest)
    }

    fn execution_index(&self, id: &E) -> Result<usize, TraceError> {
        self.executions
            .iter()
            .position(|state| &state.id == id)
            .ok_or(TraceError::UnknownExecution)
    }

    fn check_domain(&self, at: Timepoint) -> Result<(), TraceError> {
        if at.domain != self.domain {
            return Err(TraceError::DomainMismatch {
                expected: self.domain,
                observed: at.domain,
            });
        }
        Ok(())
    }

    fn check_not_before(&self, observed: Duration, previous: Duration) -> Result<(), TraceError> {
        if observed < previous {
            return Err(TraceError::OutOfOrder { previous, observed });
        }
        Ok(())
    }

    fn check_point(&self, at: Timepoint) -> Result<(), TraceError> {
        self.check_domain(at)?;
        if let Some(previous) = self.last {
            self.check_not_before(at.elapsed, previous)?;
        }
        Ok(())
    }

    fn check_interval(&self, start: Timepoint, end: Timepoint) -> Result<Duration, TraceError> {
        self.check_domain(start)?;
        self.check_point(end)?;
        end.elapsed
            .checked_sub(start.elapsed)
            .ok_or(TraceError::InvalidInterval {
                start: start.elapsed,
                end: end.elapsed,
            })
    }

    fn commit_point(&mut self, at: Timepoint) {
        self.first.get_or_insert(at.elapsed);
        self.last = Some(at.elapsed);
    }

    fn commit_interval(&mut self, start: Timepoint, end: Timepoint) {
        self.first = Some(
            self.first
                .map_or(start.elapsed, |first| first.min(start.elapsed)),
        );
        self.last = Some(end.elapsed);
    }

    fn sample(
        &mut self,
        stage: Stage,
        class: TraceClass,
        backend: Option<BackendKind>,
        duration: Duration,
    ) {
        self.stages[stage as usize].record(
            Sample {
                class,
                backend,
                duration,
            },
            &mut self.counter_overflow,
        );
    }
}

#[derive(Debug)]
struct RequestState<R> {
    id: R,
    class: TraceClass,
    start: Duration,
    finished: Option<FinishKind>,
    consumed: bool,
    stages: [bool; Stage::ALL.len()],
}

#[derive(Debug)]
struct RetiredRequest<R> {
    id: R,
    consumed: bool,
}

#[derive(Debug)]
struct ExecutionState<E> {
    id: E,
    class: TraceClass,
    backend: BackendKind,
    items: usize,
    consumed: usize,
    start: Duration,
    completed: bool,
    dispatch_recorded: bool,
    transfer_recorded: bool,
}

#[derive(Debug, Default)]
struct Counters {
    logical: LogicalOutcomes,
    rejected_requests: u64,
    physical_executions_started: u64,
    physical_executions_completed: u64,
    physical_items_started: u64,
    physical_items_completed: u64,
    consumed_requests: u64,
    physical_items_consumed: u64,
    cache_consumed_requests: u64,
    unused_items_finalized: u64,
    duplicate_finishes_rejected: u64,
    duplicate_consumes_rejected: u64,
    batch_samples_dropped: u64,
    tracking_capacity_rejections: u64,
    request_ids_evicted: u64,
    execution_ids_evicted: u64,
    observation_errors: u64,
    domain_errors: u64,
    time_errors: u64,
    external_events_dropped: u64,
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    class: TraceClass,
    backend: Option<BackendKind>,
    duration: Duration,
}

#[derive(Debug)]
struct StageSamples {
    samples: Ring<Sample>,
    total: u64,
    dropped: u64,
    class_total: [u64; TraceClass::ALL.len()],
    class_dropped: [u64; TraceClass::ALL.len()],
    backend_total: [u64; BackendKind::ALL.len()],
    backend_dropped: [u64; BackendKind::ALL.len()],
}

impl StageSamples {
    fn try_new(capacity: usize) -> Result<Self, CapacityError> {
        Ok(Self {
            samples: Ring::try_new(capacity)?,
            total: 0,
            dropped: 0,
            class_total: [0; TraceClass::ALL.len()],
            class_dropped: [0; TraceClass::ALL.len()],
            backend_total: [0; BackendKind::ALL.len()],
            backend_dropped: [0; BackendKind::ALL.len()],
        })
    }

    fn record(&mut self, sample: Sample, overflow: &mut bool) {
        add(&mut self.total, 1, overflow);
        add(&mut self.class_total[sample.class as usize], 1, overflow);
        if let Some(backend) = sample.backend {
            add(&mut self.backend_total[backend as usize], 1, overflow);
        }
        let omitted = if self.samples.limit == 0 {
            Some(sample)
        } else if self.samples.samples.len() == self.samples.limit {
            Some(self.samples.samples[self.samples.next])
        } else {
            None
        };
        self.samples.push(sample);
        if let Some(sample) = omitted {
            add(&mut self.dropped, 1, overflow);
            add(&mut self.class_dropped[sample.class as usize], 1, overflow);
            if let Some(backend) = sample.backend {
                add(&mut self.backend_dropped[backend as usize], 1, overflow);
            }
        }
    }

    fn report(&self, stage: Stage) -> StageReport {
        let mut durations: Vec<_> = self
            .samples
            .samples
            .iter()
            .map(|sample| sample.duration)
            .collect();
        durations.sort_unstable();
        let mut by_class = BTreeMap::new();
        for class in TraceClass::ALL {
            if self.class_total[class as usize] != 0 {
                let mut values: Vec<_> = self
                    .samples
                    .samples
                    .iter()
                    .filter(|sample| sample.class == class)
                    .map(|sample| sample.duration)
                    .collect();
                values.sort_unstable();
                by_class.insert(
                    class,
                    summary(
                        self.class_total[class as usize],
                        self.class_dropped[class as usize],
                        &values,
                    ),
                );
            }
        }
        let mut by_backend = BTreeMap::new();
        for backend in BackendKind::ALL {
            if self.backend_total[backend as usize] != 0 {
                let mut values: Vec<_> = self
                    .samples
                    .samples
                    .iter()
                    .filter(|sample| sample.backend == Some(backend))
                    .map(|sample| sample.duration)
                    .collect();
                values.sort_unstable();
                by_backend.insert(
                    backend,
                    summary(
                        self.backend_total[backend as usize],
                        self.backend_dropped[backend as usize],
                        &values,
                    ),
                );
            }
        }
        StageReport {
            stage,
            latency: summary(self.total, self.dropped, &durations),
            by_class,
            by_backend,
        }
    }
}

fn allocate<T>(capacity: usize) -> Result<Vec<T>, CapacityError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| CapacityError::AllocationFailed)?;
    Ok(values)
}

fn add_count(counter: &mut u64, count: usize, overflow: &mut bool) {
    match u64::try_from(count) {
        Ok(count) => add(counter, count, overflow),
        Err(_) => {
            *counter = u64::MAX;
            *overflow = true;
        }
    }
}

fn summary(total: u64, dropped: u64, sorted: &[Duration]) -> SampleSummary {
    SampleSummary {
        total,
        retained: sorted.len(),
        dropped,
        p50: percentile(sorted, 50),
        p95: percentile(sorted, 95),
        p99: percentile(sorted, 99),
    }
}

fn stages_have_loss(stages: &[StageSamples]) -> bool {
    stages.iter().any(|stage| stage.dropped != 0)
}
