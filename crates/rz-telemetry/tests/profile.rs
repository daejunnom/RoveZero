use rz_telemetry::profile::{
    BackendKind, MemoryReservations, ProfileConfig, ProfileReport, Stage, Timepoint, TraceClass,
    TraceCollector, TraceError,
};
use rz_telemetry::{CapacityError, FinishKind};
use std::time::Duration;

type Collector = TraceCollector<u64, u64>;

fn at(seconds: u64) -> Timepoint {
    Timepoint {
        domain: 17,
        elapsed: Duration::from_secs(seconds),
    }
}

fn config(capacity: usize) -> ProfileConfig {
    ProfileConfig {
        max_requests: 4,
        max_executions: 4,
        sample_capacity: capacity,
    }
}

fn collector(capacity: usize) -> Collector {
    Collector::try_new(config(capacity), 17).unwrap()
}

fn stage(report: &ProfileReport, wanted: Stage) -> &rz_telemetry::profile::StageReport {
    report
        .stages
        .iter()
        .find(|row| row.stage == wanted)
        .unwrap()
}

#[test]
fn overlap_uses_wall_span_and_separates_dispatch_backend_and_consumption() {
    let mut trace = collector(8);
    trace.begin_request(1, TraceClass::Sibling, at(0)).unwrap();
    trace.begin_request(2, TraceClass::Sibling, at(0)).unwrap();
    for id in [1, 2] {
        trace
            .record_stage(&id, Stage::CpuPreparation, at(0), at(2))
            .unwrap();
    }
    trace
        .begin_execution(30, TraceClass::Sibling, BackendKind::CpuMock, 2, at(2))
        .unwrap();
    trace.record_dispatch(&30, at(1), at(2)).unwrap();
    trace.finish_execution(&30, at(4)).unwrap();
    for id in [1, 2] {
        trace
            .record_stage(&id, Stage::Validation, at(4), at(4))
            .unwrap();
        trace
            .finish_request(&id, FinishKind::Completed, at(4))
            .unwrap();
        trace.consume_request(&id, Some(&30), at(4)).unwrap();
        trace
            .record_stage(&id, Stage::Backup, at(4), at(4))
            .unwrap();
    }
    for id in [1, 2] {
        trace
            .record_stage(&id, Stage::Output, at(4), at(5))
            .unwrap();
        trace.close_request(&id, at(5)).unwrap();
    }
    trace.retire_execution(&30, at(5)).unwrap();
    let report = trace.report();
    assert_eq!(report.wall_span, Some(Duration::from_secs(5)));
    assert_eq!(report.physical_items_per_second, Some(0.4));
    assert_eq!(report.consumed_requests_per_second, Some(0.4));
    assert_eq!(report.physical_items_completed, 2);
    assert_eq!(report.physical_items_consumed, 2);
    assert_eq!(report.unused_items_finalized, 0);
    assert_eq!(
        stage(&report, Stage::Dispatch).latency.p50,
        Some(Duration::from_secs(1))
    );
    assert_eq!(
        stage(&report, Stage::BackendPhysical).latency.p50,
        Some(Duration::from_secs(2))
    );
    assert_eq!(
        stage(&report, Stage::EndToEnd).latency.p50,
        Some(Duration::from_secs(5))
    );
    assert_eq!(stage(&report, Stage::CpuPreparation).latency.total, 2);
    assert_eq!(stage(&report, Stage::Transfer).latency.total, 0);
    assert_eq!(stage(&report, Stage::Transfer).latency.p50, None);
    assert_eq!(
        stage(&report, Stage::BackendPhysical).by_backend[&BackendKind::CpuMock].total,
        1
    );
    assert_eq!(report.completeness.observation_errors, 0);
    assert_eq!(report.completeness.active_requests, 0);
    assert_eq!(report.completeness.active_executions, 0);
}

#[test]
fn execution_only_dispatch_moves_wall_start_to_the_earliest_accepted_span() {
    let mut trace = collector(2);
    trace
        .begin_execution(1, TraceClass::Cold, BackendKind::CpuMock, 1, at(5))
        .unwrap();
    // An invalid earlier interval must not alter the accepted measurement window.
    assert!(matches!(
        trace.record_dispatch(
            &1,
            Timepoint {
                domain: 99,
                ..at(0)
            },
            at(5)
        ),
        Err(TraceError::DomainMismatch { .. })
    ));
    assert_eq!(trace.report().first_point, Some(at(5)));
    trace.record_dispatch(&1, at(1), at(5)).unwrap();
    trace.finish_execution(&1, at(9)).unwrap();
    trace.retire_execution(&1, at(9)).unwrap();
    let report = trace.report();
    assert_eq!(report.first_point, Some(at(1)));
    assert_eq!(report.last_point, Some(at(9)));
    assert_eq!(report.wall_span, Some(Duration::from_secs(8)));
    assert_eq!(report.physical_items_per_second, Some(0.125));
    assert_eq!(
        stage(&report, Stage::Dispatch).latency.p50,
        Some(Duration::from_secs(4))
    );
    assert_eq!(
        stage(&report, Stage::BackendPhysical).latency.p50,
        Some(Duration::from_secs(4))
    );
}

#[test]
fn finish_backup_and_physical_work_are_exactly_once_within_the_check_window() {
    let mut trace = collector(4);
    trace.begin_request(1, TraceClass::Cold, at(0)).unwrap();
    trace
        .begin_execution(2, TraceClass::Cold, BackendKind::CpuMock, 1, at(1))
        .unwrap();
    trace.finish_execution(&2, at(2)).unwrap();
    trace
        .finish_request(&1, FinishKind::Completed, at(2))
        .unwrap();
    assert_eq!(
        trace.finish_request(&1, FinishKind::Failed, at(2)),
        Err(TraceError::DuplicateFinish)
    );
    trace.consume_request(&1, Some(&2), at(2)).unwrap();
    assert_eq!(
        trace.consume_request(&1, Some(&2), at(2)),
        Err(TraceError::DuplicateConsume)
    );
    trace.record_stage(&1, Stage::Backup, at(2), at(3)).unwrap();
    assert_eq!(
        trace.record_stage(&1, Stage::Backup, at(2), at(3)),
        Err(TraceError::DuplicateStage {
            stage: Stage::Backup
        })
    );
    trace.close_request(&1, at(3)).unwrap();
    assert_eq!(
        trace.finish_request(&1, FinishKind::Completed, at(3)),
        Err(TraceError::DuplicateFinish)
    );
    assert_eq!(
        trace.consume_request(&1, Some(&2), at(3)),
        Err(TraceError::DuplicateConsume)
    );
    trace.retire_execution(&2, at(3)).unwrap();
    assert_eq!(
        trace.finish_execution(&2, at(3)),
        Err(TraceError::DuplicatePhysicalFinish)
    );
    assert_eq!(
        trace.begin_execution(2, TraceClass::Cold, BackendKind::CpuMock, 1, at(3)),
        Err(TraceError::DuplicateExecution)
    );
    let report = trace.report();
    assert_eq!(report.logical.completed, 1);
    assert_eq!(report.logical.failed, 0);
    assert_eq!(report.physical_executions_completed, 1);
    assert_eq!(report.consumed_requests, 1);
    assert_eq!(report.duplicate_finishes_rejected, 3);
    assert_eq!(report.duplicate_consumes_rejected, 2);
    assert_eq!(report.completeness.observation_errors, 7);
}

#[test]
fn cancellation_closes_logically_before_late_physical_completion_and_unused_finalization() {
    let mut trace = collector(4);
    trace.begin_request(1, TraceClass::Warm, at(0)).unwrap();
    trace
        .begin_execution(2, TraceClass::Warm, BackendKind::Cpu, 3, at(1))
        .unwrap();
    trace
        .finish_request(&1, FinishKind::Canceled, at(2))
        .unwrap();
    assert_eq!(
        trace.consume_request(&1, Some(&2), at(2)),
        Err(TraceError::RequestNotCompleted)
    );
    trace.close_request(&1, at(2)).unwrap();
    assert_eq!(
        trace.retire_execution(&2, at(2)),
        Err(TraceError::ExecutionNotCompleted)
    );
    let logical = trace.report();
    assert_eq!(logical.logical.canceled, 1);
    assert_eq!(logical.physical_items_completed, 0);
    assert_eq!(logical.completeness.active_requests, 0);
    assert_eq!(logical.completeness.active_executions, 1);
    trace.finish_execution(&2, at(9)).unwrap();
    assert_eq!(trace.report().completed_unconsumed_items, 3);
    assert_eq!(trace.report().unused_items_finalized, 0);
    trace.retire_execution(&2, at(9)).unwrap();
    let physical = trace.report();
    assert_eq!(physical.physical_items_completed, 3);
    assert_eq!(physical.consumed_requests, 0);
    assert_eq!(physical.unused_items_finalized, 3);
    assert_eq!(physical.completed_unconsumed_items, 0);
    assert_eq!(physical.wall_span, Some(Duration::from_secs(9)));
    assert_eq!(
        stage(&physical, Stage::EndToEnd).latency.p50,
        Some(Duration::from_secs(2))
    );
    assert_eq!(
        stage(&physical, Stage::BackendPhysical).latency.p50,
        Some(Duration::from_secs(8))
    );
}

#[test]
fn domain_and_temporal_errors_leave_measurements_unchanged_and_report_incompleteness() {
    let mut trace = collector(4);
    trace
        .begin_request(1, TraceClass::ParentChild, at(5))
        .unwrap();
    assert!(matches!(
        trace.finish_request(
            &1,
            FinishKind::Completed,
            Timepoint {
                domain: 99,
                ..at(6)
            }
        ),
        Err(TraceError::DomainMismatch {
            expected: 17,
            observed: 99
        })
    ));
    assert!(matches!(
        trace.finish_request(&1, FinishKind::Completed, at(4)),
        Err(TraceError::OutOfOrder { .. })
    ));
    assert!(matches!(
        trace.record_stage(&1, Stage::Queue, at(7), at(6)),
        Err(TraceError::InvalidInterval { .. })
    ));
    assert!(matches!(
        trace.record_stage(&1, Stage::Queue, at(3), at(6)),
        Err(TraceError::OutOfOrder { .. })
    ));
    let report = trace.report();
    assert_eq!(report.logical.completed, 0);
    assert_eq!(report.first_point, Some(at(5)));
    assert_eq!(report.last_point, Some(at(5)));
    assert_eq!(report.physical_items_per_second, None);
    assert_eq!(stage(&report, Stage::Queue).latency.total, 0);
    assert_eq!(report.completeness.observation_errors, 4);
    assert_eq!(report.completeness.domain_errors, 1);
    assert_eq!(report.completeness.time_errors, 3);
    trace
        .finish_request(&1, FinishKind::Expired, at(6))
        .unwrap();
    trace.close_request(&1, at(6)).unwrap();
    assert_eq!(trace.report().logical.expired, 1);
}

#[test]
fn tracking_limits_are_explicit_and_retirement_reuses_slots_with_a_bounded_authority_window() {
    let config = ProfileConfig {
        max_requests: 1,
        max_executions: 1,
        sample_capacity: 4,
    };
    let mut trace = Collector::try_new(config, 17).unwrap();
    trace.begin_request(1, TraceClass::Eviction, at(0)).unwrap();
    assert_eq!(
        trace.begin_request(2, TraceClass::Eviction, at(0)),
        Err(TraceError::RequestCapacity)
    );
    trace
        .begin_execution(10, TraceClass::Eviction, BackendKind::CpuMock, 1, at(0))
        .unwrap();
    assert_eq!(
        trace.begin_execution(11, TraceClass::Eviction, BackendKind::CpuMock, 1, at(0)),
        Err(TraceError::ExecutionCapacity)
    );
    trace.finish_request(&1, FinishKind::Stale, at(1)).unwrap();
    trace.close_request(&1, at(1)).unwrap();
    trace.finish_execution(&10, at(1)).unwrap();
    trace.retire_execution(&10, at(1)).unwrap();
    assert_eq!(
        trace.begin_request(1, TraceClass::Eviction, at(1)),
        Err(TraceError::DuplicateRequest)
    );
    trace.begin_request(2, TraceClass::Eviction, at(1)).unwrap();
    trace.finish_request(&2, FinishKind::Failed, at(2)).unwrap();
    trace.close_request(&2, at(2)).unwrap();
    trace
        .begin_execution(11, TraceClass::Eviction, BackendKind::CpuMock, 1, at(2))
        .unwrap();
    trace.finish_execution(&11, at(2)).unwrap();
    trace.retire_execution(&11, at(2)).unwrap();
    let report = trace.report();
    assert_eq!(report.completeness.tracking_capacity_rejections, 2);
    assert_eq!(report.completeness.request_ids_evicted, 1);
    assert_eq!(report.completeness.execution_ids_evicted, 1);
    assert!(!report.completeness.duplicate_checks_cover_all_ids);
    assert!(report.completeness.caller_ids_must_be_unique);
}

#[test]
fn rings_keep_recent_samples_and_losses_are_attributed_to_stage_and_class() {
    let mut trace = collector(2);
    for id in 0..5 {
        let class = if id < 3 {
            TraceClass::Cold
        } else {
            TraceClass::Long
        };
        trace.begin_request(id, class, at(id * 10)).unwrap();
        trace
            .begin_execution(
                id,
                class,
                BackendKind::CpuMock,
                id as usize + 1,
                at(id * 10),
            )
            .unwrap();
        trace.finish_execution(&id, at(id * 10 + 1)).unwrap();
        trace
            .finish_request(&id, FinishKind::Completed, at(id * 10 + id + 1))
            .unwrap();
        trace
            .consume_request(&id, Some(&id), at(id * 10 + id + 1))
            .unwrap();
        trace.close_request(&id, at(id * 10 + id + 1)).unwrap();
        trace.retire_execution(&id, at(id * 10 + id + 1)).unwrap();
    }
    let report = trace.report();
    let end = stage(&report, Stage::EndToEnd);
    assert_eq!(end.latency.total, 5);
    assert_eq!(end.latency.retained, 2);
    assert_eq!(end.latency.dropped, 3);
    assert_eq!(end.latency.p50, Some(Duration::from_secs(4)));
    assert_eq!(end.latency.p95, Some(Duration::from_secs(5)));
    assert_eq!(end.latency.p99, Some(Duration::from_secs(5)));
    assert_eq!(end.by_class[&TraceClass::Cold].total, 3);
    assert_eq!(end.by_class[&TraceClass::Cold].dropped, 3);
    assert_eq!(end.by_class[&TraceClass::Cold].p50, None);
    assert_eq!(end.by_class[&TraceClass::Long].retained, 2);
    assert_eq!(report.batch_distribution, [(4, 1), (5, 1)].into());
    assert_eq!(report.batch_samples_total, 5);
    assert_eq!(report.batch_samples_retained, 2);
    assert_eq!(report.batch_samples_dropped, 3);
    assert!(report.completeness.stage_samples_dropped);
    assert!(report.completeness.batch_samples_dropped);
}

#[test]
fn checked_capacities_and_zero_sample_capacity_do_not_need_unsafe_allocations() {
    for bad in [
        ProfileConfig {
            max_requests: usize::MAX,
            ..config(2)
        },
        ProfileConfig {
            max_executions: usize::MAX,
            ..config(2)
        },
        ProfileConfig {
            sample_capacity: usize::MAX,
            ..config(2)
        },
    ] {
        assert!(matches!(
            Collector::try_new(bad, 17),
            Err(CapacityError::Overflow)
        ));
    }
    let mut trace = collector(0);
    trace.begin_request(1, TraceClass::Cold, at(0)).unwrap();
    trace
        .begin_execution(2, TraceClass::Cold, BackendKind::CpuMock, 1, at(0))
        .unwrap();
    trace.finish_execution(&2, at(1)).unwrap();
    trace
        .finish_request(&1, FinishKind::Completed, at(1))
        .unwrap();
    trace.consume_request(&1, Some(&2), at(1)).unwrap();
    trace.close_request(&1, at(1)).unwrap();
    trace.retire_execution(&2, at(1)).unwrap();
    trace.note_event_loss(3);
    let report = trace.report();
    assert_eq!(report.physical_items_completed, 1);
    assert_eq!(report.consumed_requests, 1);
    assert_eq!(stage(&report, Stage::EndToEnd).latency.total, 1);
    assert_eq!(stage(&report, Stage::EndToEnd).latency.dropped, 1);
    assert_eq!(stage(&report, Stage::EndToEnd).latency.p50, None);
    assert_eq!(report.batch_samples_dropped, 1);
    assert_eq!(report.completeness.external_events_dropped, 3);
}

#[test]
fn reservation_peaks_are_ledger_measurements_and_do_not_invent_actual_device_usage() {
    let mut trace = collector(2);
    let current = MemoryReservations {
        host_bytes: 20,
        device_bytes: 0,
        pinned_bytes: 5,
    };
    trace.observe_reservations(current, at(0)).unwrap();
    trace
        .observe_reservation_peak(
            MemoryReservations {
                host_bytes: 40,
                device_bytes: 70,
                pinned_bytes: 8,
            },
            at(1),
        )
        .unwrap();
    let report = trace.report();
    assert_eq!(report.reserved_current, current);
    assert_eq!(report.reserved_peak.host_bytes, 40);
    assert_eq!(report.reserved_peak.device_bytes, 70);
    assert_eq!(report.reserved_peak.pinned_bytes, 8);
    assert_eq!(report.physical_items_completed, 0);
}

#[test]
fn actual_transfer_requires_gpu_label_and_successful_validation_does_not_mean_consumption() {
    let mut trace = collector(2);
    trace
        .begin_request(1, TraceClass::Transposition, at(0))
        .unwrap();
    trace
        .begin_execution(2, TraceClass::Transposition, BackendKind::CpuMock, 1, at(0))
        .unwrap();
    assert_eq!(
        trace.record_transfer(&2, at(0), at(0)),
        Err(TraceError::TransferUnavailable)
    );
    trace.finish_execution(&2, at(1)).unwrap();
    trace
        .record_stage(&1, Stage::Validation, at(1), at(1))
        .unwrap();
    trace.finish_request(&1, FinishKind::Stale, at(1)).unwrap();
    assert_eq!(
        trace.record_stage(&1, Stage::Backup, at(1), at(1)),
        Err(TraceError::RequestNotConsumed)
    );
    trace.close_request(&1, at(1)).unwrap();
    trace.retire_execution(&2, at(1)).unwrap();
    assert_eq!(trace.report().consumed_requests, 0);
    assert_eq!(trace.report().unused_items_finalized, 1);

    trace
        .begin_execution(3, TraceClass::Cold, BackendKind::Gpu, 1, at(2))
        .unwrap();
    trace.record_transfer(&3, at(2), at(3)).unwrap();
    trace.finish_execution(&3, at(4)).unwrap();
    trace.retire_execution(&3, at(4)).unwrap();
    let report = trace.report();
    assert_eq!(
        stage(&report, Stage::Transfer).by_backend[&BackendKind::Gpu].total,
        1
    );
    assert_eq!(
        stage(&report, Stage::Transfer).latency.p50,
        Some(Duration::from_secs(1))
    );
}
