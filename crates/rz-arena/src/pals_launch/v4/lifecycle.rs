//! Typed consumption of Native's bounded process journal. Policy selection,
//! actual owner facts and physical completion retain separate authority.
use super::*;
use rz_uci::pals_attestation::followup::{
    NativeArchiveObservationV4, NativeCudaWarmObservationV4, NativeObservedV4,
    NativePausedStackObservationV4, PalsFollowupLifecycleV4,
};

fn typed(raw: &serde_json::Value) -> Result<PalsFollowupLifecycleV4, ArenaError> {
    let parsed: PalsFollowupLifecycleV4 = serde_json::from_value(raw.clone())
        .map_err(|e| invalid(format!("actual Native lifecycle codec:{e}")))?;
    require(
        serde_json::to_value(&parsed).map_err(|e| invalid(e.to_string()))? == *raw,
        "actual lifecycle has missing nullable fields or a noncanonical optional member",
    )?;
    Ok(parsed)
}
fn known<T>(o: &NativeObservedV4<T>) -> Option<&T> {
    match o {
        NativeObservedV4::Unknown => None,
        NativeObservedV4::Observed { value, .. } => Some(value),
    }
}
fn observed_method<T>(o: &NativeObservedV4<T>, expected: &str) -> Result<(), ArenaError> {
    if let NativeObservedV4::Observed { method, .. } = o {
        require(
            method == expected,
            "actual lifecycle observation method differs from its closed producer",
        )?;
    }
    Ok(())
}
fn owner_methods(raw: &PalsFollowupLifecycleV4) -> Result<PalsFollowupOwnerMethodsV4, ArenaError> {
    serde_json::from_value(
        serde_json::to_value(&raw.owner_methods).map_err(|e| invalid(e.to_string()))?,
    )
    .map_err(|e| invalid(format!("actual owner method codec:{e}")))
}
fn validate_scope(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    raw: &PalsFollowupLifecycleV4,
    envelope: &serde_json::Value,
) -> Result<(), ArenaError> {
    let b = lock
        .input
        .execution_binding(role_index(role))
        .ok_or_else(|| invalid("unselected actual lifecycle"))?;
    require(
        raw.domain == PALS_FOLLOWUP_LIFECYCLE_V4_DOMAIN
            && raw.schema_version == 1
            && raw.trace_capacity == PALS_FOLLOWUP_TRACE_CAPACITY_V4
            && raw.trace_bytes_max == PALS_FOLLOWUP_TRACE_BYTES_V4
            && owner_methods(raw)? == b.owner_methods
            && raw.repair_traces.len() <= raw.trace_capacity as usize
            && serde_json::to_vec(&raw.repair_traces)
                .map_err(|e| invalid(e.to_string()))?
                .len()
                <= raw.trace_bytes_max as usize,
        "actual producer scope/source/capture limits differ from execution binding",
    )?;
    if let Some(partial) = &raw.partial_facts {
        require(
            serde_json::to_vec(partial)
                .map_err(|e| invalid(e.to_string()))?
                .len()
                <= PALS_FOLLOWUP_PARTIAL_BYTES_V4,
            "actual partial owner facts exceed 8KiB",
        )?;
    }
    let native = &envelope["native"];
    if lock.input.endpoints[role_index(role)]
        .native_model()
        .is_some()
    {
        require(
            raw.process_epoch
                .is_some_and(|epoch| epoch > 0 && native["process_epoch"] == epoch)
                && native["game_generation"] == raw.game_generation,
            "actual lifecycle process/game scope differs from Native physical owner",
        )?;
        if raw.owner_shutdown_complete {
            require(
                native["physical_shutdown_confirmed"] == true
                    && native["native_buffers_released"] == true
                    && native["quarantined"] == false
                    && native["physical_runs_in_flight"] == 0,
                "followup owner seal cannot substitute for Native physical shutdown",
            )?;
        }
    } else {
        require(
            raw.process_epoch.is_none(),
            "mock lifecycle cannot invent a Native process epoch",
        )?;
    }
    observed_method(&raw.archive, "archive_actual_owner_events_v4")?;
    observed_method(&raw.paused_stack, "cpu_actual_paused_frame_owner_v4")?;
    observed_method(&raw.cuda_warm, "cuda_actual_warm_owner_events_v4")?;
    observed_method(&raw.repair_trace_total, "engine_actual_repair_recorder_v4")?;
    observed_method(
        &raw.pending_questions_peak,
        "engine_actual_repair_recorder_v4",
    )?;
    if let Some(peak) = &raw.repair_admissions_peak {
        observed_method(peak, "engine_actual_repair_recorder_v4")?;
    }
    if b.owner_methods.repair.is_none() {
        require(
            raw.repair_traces.is_empty()
                && known(&raw.repair_trace_total).is_none()
                && known(&raw.pending_questions_peak).is_none()
                && raw.repair_admissions_peak.is_none(),
            "unselected Repair lane emitted observed counters or traces",
        )?;
    }
    let PalsEngineV4::Pals(e) = &lock.input.semantic_lock.manifest.engines[role_index(role)] else {
        return Err(invalid("non-PALS actual followup producer"));
    };
    if matches!(
        e.policies.iterative_repair,
        PalsIterativeRepairPolicyV4::Disabled
    ) {
        require(
            raw.repair_admissions_peak.is_none(),
            "unselected iterative admission counter",
        )?;
    }
    if b.cold_archive.is_none() {
        require(
            known(&raw.archive).is_none()
                && raw.previous_closed_archive_owners.is_empty()
                && raw
                    .partial_facts
                    .as_ref()
                    .is_none_or(|p| p.archive.is_none()),
            "unselected archive observations",
        )?;
    }
    if b.owner_methods.paused_stack.is_none() {
        require(
            known(&raw.paused_stack).is_none()
                && raw
                    .partial_facts
                    .as_ref()
                    .is_none_or(|p| p.paused_stack.is_none()),
            "unselected stack observations",
        )?;
    }
    if b.cuda_warm.is_none() {
        require(
            known(&raw.cuda_warm).is_none()
                && raw
                    .partial_facts
                    .as_ref()
                    .is_none_or(|p| p.cuda_warm.is_none()),
            "unselected CUDA Warm observations",
        )?;
    }
    if let Some(a) = known(&raw.archive).or_else(|| raw.partial_facts.as_ref()?.archive.as_ref()) {
        validate_archive_binding(b, a)?;
    }
    require(
        raw.previous_closed_archive_owners.len() <= 2,
        "actual previous archive owner journal exceeds two",
    )?;
    let mut archive_ids = BTreeSet::new();
    if let Some(a) = known(&raw.archive).or_else(|| raw.partial_facts.as_ref()?.archive.as_ref()) {
        archive_ids.insert(a.owner_id);
    }
    for previous in &raw.previous_closed_archive_owners {
        validate_archive_binding(b, previous)?;
        require(
            archive_ids.insert(previous.owner_id) && archive_closed(previous),
            "archive predecessor is duplicate, incomplete, open or retains a transaction",
        )?;
    }
    if let Some(w) =
        known(&raw.cuda_warm).or_else(|| raw.partial_facts.as_ref()?.cuda_warm.as_ref())
    {
        let selected = b
            .cuda_warm
            .as_ref()
            .ok_or_else(|| invalid("unselected Warm owner"))?;
        require(
            w.max_leases == selected.max_leases && w.device_bytes_max == selected.device_bytes_max,
            "actual CUDA buffer/lease selector differs from bound limits",
        )?;
    }
    require(
        !raw.owner_shutdown_complete
            || (raw.capture_complete
                && raw.capture_failure.is_none()
                && raw.partial_facts.is_none()),
        "actual incomplete lifecycle claimed a completed owner seal",
    )
}
fn validate_archive_binding(
    b: &PalsEndpointExecutionBindingV4,
    a: &NativeArchiveObservationV4,
) -> Result<(), ArenaError> {
    let selected = b
        .cold_archive
        .as_ref()
        .ok_or_else(|| invalid("unselected archive owner"))?;
    let method = b
        .owner_methods
        .cold_archive
        .as_ref()
        .ok_or_else(|| invalid("archive method binding missing"))?;
    require(
        a.owner_id > 0
            && a.owner_method == method.method
            && a.compiled_source_sha256 == method.implementation_sha256,
        "actual archive source/owner identity differs",
    )?;
    if let Some(l) = &a.runtime_limits {
        let actual: PalsArchiveRuntimeLimitsV4 =
            serde_json::from_value(serde_json::to_value(l).map_err(|e| invalid(e.to_string()))?)
                .map_err(|e| invalid(e.to_string()))?;
        require(
            actual == selected.runtime_limits,
            "actual archive eight selectors differ from launch binding",
        )?;
    } else {
        require(!a.complete, "complete archive has unknown actual selectors")?;
    }
    require(
        a.session_write_ledger_id.is_none_or(|id| id > 0)
            && a.session_write_bytes_max
                .is_none_or(|max| max == selected.write_output_bytes_max)
            && a.session_write_bytes_consumed.is_none_or(|consumed| {
                consumed <= selected.write_output_bytes_max
                    && a.archive_write_bytes_total
                        .is_none_or(|written| written <= consumed)
            })
            && (!a.complete
                || (a.session_write_ledger_id.is_some()
                    && a.session_write_bytes_max.is_some()
                    && a.session_write_bytes_consumed.is_some())),
        "actual archive session write ledger differs from its original bound or is unknown",
    )?;
    Ok(())
}
fn monotonic(values: &[(u64, u64)], message: &str) -> Result<(), ArenaError> {
    require(values.iter().all(|(start, end)| start <= end), message)
}
fn stack_transition(
    a: &NativePausedStackObservationV4,
    b: &NativePausedStackObservationV4,
) -> Result<(), ArenaError> {
    require(
        a.owner_id == b.owner_id
            && !(a.admission_closed && !b.admission_closed)
            && !(a.owner_released && !b.owner_released),
        "paused owner moved or reopened",
    )?;
    monotonic(
        &[
            (a.event_sequence, b.event_sequence),
            (a.tokens_created, b.tokens_created),
            (a.tokens_resumed, b.tokens_resumed),
            (a.tokens_invalidated, b.tokens_invalidated),
            (a.bytes_peak, b.bytes_peak),
            (a.tokens_peak.into(), b.tokens_peak.into()),
            (a.stale_context_attempts, b.stale_context_attempts),
            (a.stale_context_rejections, b.stale_context_rejections),
            (a.replayed_consumed_work, b.replayed_consumed_work),
        ],
        "actual paused lifetime counters regressed",
    )
}
fn archive_transition(
    a: &NativeArchiveObservationV4,
    b: &NativeArchiveObservationV4,
) -> Result<(), ArenaError> {
    session_write_transition(a, b)?;
    require(
        a.owner_id == b.owner_id
            && a.root_sha256 == b.root_sha256
            && a.repository_sha256 == b.repository_sha256
            && a.runtime_limits == b.runtime_limits
            && a.compiled_source_sha256 == b.compiled_source_sha256
            && !(a.admission_closed && !b.admission_closed),
        "archive owner moved or reopened",
    )?;
    let mut counters = vec![
        (a.event_sequence, b.event_sequence),
        (a.reserved_generations, b.reserved_generations),
        (a.generation, b.generation),
        (a.committed_chunks, b.committed_chunks),
        (a.integrity_verified_chunks, b.integrity_verified_chunks),
        (a.committed_bytes, b.committed_bytes),
        (a.integrity_verified_bytes, b.integrity_verified_bytes),
        (a.ram_reclaim_events, b.ram_reclaim_events),
        (
            a.integrity_before_reclaim_events,
            b.integrity_before_reclaim_events,
        ),
        (a.unverified_reclaim_events, b.unverified_reclaim_events),
        (a.index_entries_peak.into(), b.index_entries_peak.into()),
        (a.index_bytes_peak, b.index_bytes_peak),
        (a.pinned_entries_peak.into(), b.pinned_entries_peak.into()),
        (a.loads_requested, b.loads_requested),
        (a.loads_completed, b.loads_completed),
        (a.load_bytes_total, b.load_bytes_total),
        (a.load_bytes_peak, b.load_bytes_peak),
        (a.load_elapsed_peak_ms, b.load_elapsed_peak_ms),
        (a.owner_generation_checks, b.owner_generation_checks),
        (
            a.owner_generation_check_failures,
            b.owner_generation_check_failures,
        ),
        (a.quota_failures, b.quota_failures),
        (a.io_failures, b.io_failures),
        (a.pin_saturation_failures, b.pin_saturation_failures),
    ];
    for (start, end) in [
        (a.ram_released_bytes, b.ram_released_bytes),
        (a.archive_write_bytes_total, b.archive_write_bytes_total),
    ] {
        if let (Some(start), Some(end)) = (start, end) {
            counters.push((start, end));
        }
        require(
            start.is_none() || end.is_some() || !b.complete,
            "archive complete lifetime lost a known metric",
        )?;
    }
    monotonic(&counters, "actual archive lifetime counters regressed")
}

fn session_write_transition(
    a: &NativeArchiveObservationV4,
    b: &NativeArchiveObservationV4,
) -> Result<(), ArenaError> {
    for (before, after) in [
        (a.session_write_ledger_id, b.session_write_ledger_id),
        (a.session_write_bytes_max, b.session_write_bytes_max),
    ] {
        require(
            before
                .zip(after)
                .is_none_or(|(before, after)| before == after)
                && (before.is_none() || after.is_some() || !b.complete),
            "actual archive new-game owner replaced the original session write ledger",
        )?;
    }
    require(
        a.session_write_bytes_consumed
            .zip(b.session_write_bytes_consumed)
            .is_none_or(|(before, after)| before <= after)
            && (a.session_write_bytes_consumed.is_none()
                || b.session_write_bytes_consumed.is_some()
                || !b.complete),
        "actual archive session write consumption regressed across owner snapshots",
    )
}
fn archive_closed(a: &NativeArchiveObservationV4) -> bool {
    a.complete
        && a.lifecycle == "closed"
        && a.admission_closed
        && a.cleanup_complete
        && !a.pending_buffers_retained
        && !a.pending_files_retained
        && a.pending_commit_bytes == Some(0)
}
fn warm_transition(
    a: &NativeCudaWarmObservationV4,
    b: &NativeCudaWarmObservationV4,
) -> Result<(), ArenaError> {
    require(
        a.backend_owner_id == b.backend_owner_id
            && a.bank_owner_id == b.bank_owner_id
            && !(a.admission_closed && !b.admission_closed),
        "CUDA backend/bank owner moved or reopened",
    )?;
    monotonic(
        &[
            (a.physical_event_sequence, b.physical_event_sequence),
            (a.seed_event_sequence, b.seed_event_sequence),
            (a.leases_admitted, b.leases_admitted),
            (a.leases_physically_completed, b.leases_physically_completed),
            (a.leases_quarantined, b.leases_quarantined),
            (a.device_bytes_peak, b.device_bytes_peak),
            (a.accepted_seed_consumptions, b.accepted_seed_consumptions),
            (a.rejected_seed_contexts, b.rejected_seed_contexts),
            (a.fresh_value_evaluations, b.fresh_value_evaluations),
        ],
        "actual CUDA lifetime counters regressed",
    )
}

pub(super) fn validate_pair(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    startup: &serde_json::Value,
    termination: &serde_json::Value,
) -> Result<Option<serde_json::Value>, ArenaError> {
    let (s, t) = (
        &startup["followup_lifecycle"],
        &termination["followup_lifecycle"],
    );
    if lock.input.execution_binding(role_index(role)).is_none() {
        require(
            s.is_null() && t.is_null(),
            "unselected/default-off V4 cannot inherit an actual lifecycle",
        )?;
        return Ok(None);
    }
    let start = if s.is_null() {
        None
    } else {
        let value = typed(s)?;
        validate_scope(lock, role, &value, startup)?;
        Some(value)
    };
    if t.is_null() {
        return Ok(None);
    }
    let end = typed(t)?;
    validate_scope(lock, role, &end, termination)?;
    if let Some(start) = &start {
        require(
            start.process_epoch == end.process_epoch
                && start.owner_methods == end.owner_methods
                && start.game_generation <= end.game_generation
                && !start.owner_shutdown_complete,
            "actual startup/final producer scope changed or startup claimed terminal release",
        )?;
        require(
            (start.capture_complete || !end.capture_complete)
                && (start.capture_failure.is_none() || end.capture_failure.is_some())
                && (start.partial_facts.is_none() || !end.capture_complete),
            "actual incomplete/unknown startup capture was overwritten by a later observation",
        )?;
        if let (Some(a), Some(b)) = (known(&start.paused_stack), known(&end.paused_stack)) {
            stack_transition(a, b)?;
        }
        if let (Some(a), Some(b)) = (known(&start.cuda_warm), known(&end.cuda_warm)) {
            warm_transition(a, b)?;
        }
        if let (Some(a), Some(b)) = (known(&start.archive), known(&end.archive)) {
            if a.owner_id == b.owner_id {
                archive_transition(a, b)?;
            }
            // A real ucinewgame owner transition is projected with its closed
            // predecessor by the bounded Native journal, never summed per go.
            else {
                require(
                    start.game_generation < end.game_generation,
                    "archive owner changed inside one game scope",
                )?;
                let previous = end
                    .previous_closed_archive_owners
                    .iter()
                    .find(|p| p.owner_id == a.owner_id)
                    .ok_or_else(|| {
                        invalid(
                            "startup archive owner transition lost its actual closed predecessor",
                        )
                    })?;
                archive_transition(a, previous)?;
                session_write_transition(previous, b)?;
            }
        }
        for previous in &start.previous_closed_archive_owners {
            let final_previous = end
                .previous_closed_archive_owners
                .iter()
                .find(|p| p.owner_id == previous.owner_id)
                .ok_or_else(|| invalid("final archive journal lost a closed startup owner"))?;
            require(
                previous == final_previous,
                "closed archive owner facts changed after release",
            )?;
        }
    }
    Ok(Some(t.clone()))
}

/// The original write budget bounds one restarted process session. Owner
/// replacement is not a new budget, and repeated go snapshots are not writes.
#[cfg(target_os = "linux")]
pub(super) fn archive_output_extent(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    raw: Option<&serde_json::Value>,
) -> Result<Option<u64>, ArenaError> {
    let Some(binding) = lock
        .input
        .execution_binding(role_index(role))
        .and_then(|b| b.cold_archive.as_ref())
    else {
        return Ok(Some(0));
    };
    let Some(raw) = raw else {
        return Ok(None);
    };
    let source = typed(raw)?;
    let current =
        known(&source.archive).or_else(|| source.partial_facts.as_ref()?.archive.as_ref());
    let Some(current) = current else {
        return Ok(None);
    };
    let mut owners = BTreeMap::new();
    let mut ledger_complete = true;
    let mut session = None;
    let mut prior_consumed = 0;
    for owner in source
        .previous_closed_archive_owners
        .iter()
        .chain(std::iter::once(current))
    {
        require(
            owner.owner_id > 0
                && owners
                    .insert(owner.owner_id, owner.archive_write_bytes_total)
                    .is_none(),
            "actual archive write owner reused within one process session",
        )?;
        if let Some((id, max, consumed)) = owner
            .session_write_ledger_id
            .zip(owner.session_write_bytes_max)
            .zip(owner.session_write_bytes_consumed)
            .map(|((id, max), consumed)| (id, max, consumed))
        {
            require(
                id > 0
                    && max == binding.write_output_bytes_max
                    && consumed <= max
                    && prior_consumed <= consumed
                    && owner
                        .archive_write_bytes_total
                        .is_none_or(|written| written <= consumed)
                    && session.is_none_or(|identity| identity == (id, max)),
                "actual archive owners reset or exceeded the original cumulative write ledger",
            )?;
            session = Some((id, max));
            prior_consumed = consumed;
        } else {
            ledger_complete = false;
        }
    }
    let total = owners
        .values()
        .try_fold(0u64, |total, extent| total.checked_add((*extent)?));
    if let Some(total) = total {
        require(
            total <= binding.write_output_bytes_max,
            "actual archive owners exceeded one original process-session write budget",
        )?;
        if let Some(consumed) = current.session_write_bytes_consumed {
            require(
                total <= consumed,
                "actual archive ledger lost previously written owner extents",
            )?;
            if source.capture_complete && source.partial_facts.is_none() && ledger_complete {
                require(
                    total == consumed,
                    "complete archive ledger differs from unique actual owner writes",
                )?;
            }
        }
    }
    Ok(total.filter(|_| {
        ledger_complete
            && source.capture_complete
            && source.capture_failure.is_none()
            && source.partial_facts.is_none()
    }))
}

#[cfg(target_os = "linux")]
mod projection {
    use super::*;
    use rz_uci::pals_attestation::followup::{
        NativeRecheckObservationV4, NativeRecheckValueV4, NativeRepairTraceV4, NativeRoleContextV4,
    };

    fn fact<T: Clone>(raw: &NativeObservedV4<T>) -> PalsObservedV3<T> {
        match raw {
            NativeObservedV4::Unknown => PalsObservedV3::Unknown,
            NativeObservedV4::Observed { value, method } => PalsObservedV3::Observed {
                value: value.clone(),
                method: method.clone(),
            },
        }
    }
    fn optional_metric<T>(raw: Option<T>, message: &str) -> Result<T, ArenaError> {
        raw.ok_or_else(|| invalid(message))
    }
    fn archive(a: &NativeArchiveObservationV4) -> Result<PalsArchiveObservationV4, ArenaError> {
        let runtime_limits = serde_json::from_value(
            serde_json::to_value(optional_metric(
                a.runtime_limits,
                "archive selectors unknown",
            )?)
            .map_err(|e| invalid(e.to_string()))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        Ok(PalsArchiveObservationV4 {
            owner_id: a.owner_id,
            generation: a.generation,
            lifecycle: match a.lifecycle.as_str() {
                "open" => PalsArchiveLifecycleV4::Open,
                "closed" => PalsArchiveLifecycleV4::Closed,
                "failed" => PalsArchiveLifecycleV4::Failed,
                _ => return Err(invalid("archive lifecycle unknown")),
            },
            committed_chunks: a.committed_chunks,
            integrity_verified_chunks: a.integrity_verified_chunks,
            committed_bytes: a.committed_bytes,
            integrity_verified_bytes: a.integrity_verified_bytes,
            ram_released_bytes: optional_metric(
                a.ram_released_bytes,
                "archive RAM subset measurement unknown",
            )?,
            pending_commit_bytes: optional_metric(
                a.pending_commit_bytes,
                "archive retained on-disk extent unknown",
            )?,
            pending_buffers_retained: a.pending_buffers_retained,
            global_managed_bytes: optional_metric(
                a.global_managed_bytes,
                "archive managed quota scan unknown",
            )?,
            index_entries_peak: a.index_entries_peak,
            index_bytes_peak: a.index_bytes_peak,
            pinned_entries_peak: a.pinned_entries_peak,
            loads_requested: a.loads_requested,
            loads_completed: a.loads_completed,
            load_bytes_total: a.load_bytes_total,
            load_bytes_peak: a.load_bytes_peak,
            load_elapsed_peak_ms: a.load_elapsed_peak_ms,
            owner_generation_checks: a.owner_generation_checks,
            quota_failures: a.quota_failures,
            io_failures: a.io_failures,
            pin_saturation_failures: a.pin_saturation_failures,
            cleanup_complete: a.cleanup_complete,
            actual_scope: Some(PalsArchiveActualScopeV4 {
                measurement_contract: a.measurement_contract.clone(),
                event_sequence: a.event_sequence,
                root_sha256: optional_metric(
                    a.root_sha256.clone(),
                    "archive actual canonical root unknown",
                )?,
                repository_sha256: optional_metric(
                    a.repository_sha256.clone(),
                    "archive actual repository boundary unknown",
                )?,
                runtime_limits,
                generation_kind: "reserved_generation_counter".into(),
                reserved_generations: a.reserved_generations,
                last_committed_generation: a.last_committed_generation,
                last_verified_generation: a.last_verified_generation,
                ram_release_method: a.ram_release_method.clone(),
                ram_reclaim_events: a.ram_reclaim_events,
                integrity_before_reclaim_events: a.integrity_before_reclaim_events,
                unverified_reclaim_events: a.unverified_reclaim_events,
                pending_files_retained: a.pending_files_retained,
                cold_index_kind: a.cold_index_kind.clone(),
                load_pins_max: a.load_pins_max,
                loaded_closure_bytes_peak: a.loaded_closure_bytes_peak,
                loaded_closure_bytes_method: a.loaded_closure_bytes_method.clone(),
                owner_generation_check_failures: a.owner_generation_check_failures,
                archive_write_bytes_total: a.archive_write_bytes_total,
                session_write_ledger_id: a.session_write_ledger_id,
                session_write_bytes_max: a.session_write_bytes_max,
                session_write_bytes_consumed: a.session_write_bytes_consumed,
                global_scan_event_sequence: a.global_scan_event_sequence,
                global_scan_complete: a.global_scan_complete,
                global_scan_after_last_commit: a.global_scan_after_last_commit,
                global_scope_kind: a.global_scope_kind.clone(),
                admission_closed: a.admission_closed,
                complete: a.complete,
            }),
        })
    }
    fn archive_partial(a: &NativeArchiveObservationV4) -> PalsArchivePartialFactsV4 {
        PalsArchivePartialFactsV4 {
            owner_id: Some(a.owner_id),
            reserved_generations: Some(a.reserved_generations),
            last_committed_generation: a.last_committed_generation,
            last_verified_generation: a.last_verified_generation,
            ram_released_bytes: a.ram_released_bytes,
            pending_commit_bytes: a.pending_commit_bytes,
            pending_files_retained: Some(a.pending_files_retained),
            global_managed_bytes: a.global_managed_bytes,
            archive_write_bytes_total: a.archive_write_bytes_total,
            session_write_ledger_id: a.session_write_ledger_id,
            session_write_bytes_max: a.session_write_bytes_max,
            session_write_bytes_consumed: a.session_write_bytes_consumed,
            loaded_closure_bytes_peak: a.loaded_closure_bytes_peak,
            complete: a.complete,
        }
    }
    fn stack(a: &NativePausedStackObservationV4) -> PalsPausedStackObservationV4 {
        PalsPausedStackObservationV4 {
            tokens_created: a.tokens_created,
            tokens_resumed: a.tokens_resumed,
            tokens_invalidated: a.tokens_invalidated,
            tokens_retained: a.tokens_retained,
            tokens_peak: a.tokens_peak,
            bytes_peak: a.bytes_peak,
            stale_context_attempts: a.stale_context_attempts,
            stale_context_rejections: a.stale_context_rejections,
            replayed_consumed_work: a.replayed_consumed_work,
            owner_released: a.owner_released,
            actual_scope: Some(PalsPausedStackActualScopeV4 {
                owner_id: a.owner_id,
                event_sequence: a.event_sequence,
                bytes_current: a.bytes_current,
                complete: a.complete,
                admission_closed: a.admission_closed,
            }),
        }
    }
    fn warm(a: &NativeCudaWarmObservationV4) -> PalsCudaWarmObservationV4 {
        let capability=match a.capability_available {
            Some(available)=>PalsObservedV3::Observed {value:PalsCudaWarmCapabilityV4 {
                semantic_id:a.capability_identity.clone(),device_identity:a.device_identity.clone(),available},
                method:"actual loaded CUDA capability and exercised startup/provider placement; no physical UUID or allocator VRAM claim".into()},
            None=>PalsObservedV3::Unknown,
        };
        PalsCudaWarmObservationV4 {
            capability,
            leases_admitted: a.leases_admitted,
            leases_physically_completed: a.leases_physically_completed,
            leases_quarantined: a.leases_quarantined,
            leases_peak: a.leases_peak,
            device_bytes_peak: a.device_bytes_peak,
            accepted_seed_consumptions: a.accepted_seed_consumptions,
            seed_context_checked_consumptions: a.seed_context_checked_consumptions,
            rejected_seed_contexts: a.rejected_seed_contexts,
            fresh_value_evaluations: a.fresh_value_evaluations,
            buffers_released: a.buffers_released,
            actual_scope: Some(PalsCudaWarmActualScopeV4 {
                measurement_contract: a.measurement_contract.clone(),
                backend_owner_id: a.backend_owner_id,
                bank_owner_id: a.bank_owner_id,
                physical_event_sequence: a.physical_event_sequence,
                seed_event_sequence: a.seed_event_sequence,
                leases_active: a.leases_active,
                device_bytes_current: a.device_bytes_current,
                device_bytes_retained: a.device_bytes_retained,
                device_bytes_scope: "explicit_cuda_kv_payload_bytes".into(),
                lease_accounting_scope: "backend_owner_lifetime_including_startup".into(),
                seed_accounting_scope: "search_only".into(),
                value_accounting_scope: "search_only".into(),
                rejection_scope: "selected_payload_seal_and_bind_validation".into(),
                startup_leases_admitted: a.startup_leases_admitted,
                startup_leases_completed: a.startup_leases_physically_completed,
                startup_leases_quarantined: a.startup_leases_quarantined,
                value_always_fresh: a.value_always_fresh,
                admission_closed: a.admission_closed,
                bank_closed: a.bank_closed,
                backend_dropped: a.backend_dropped,
                worker_joined: a.worker_joined,
                complete: a.complete,
            }),
        }
    }
    fn warm_partial(a: &NativeCudaWarmObservationV4) -> PalsCudaWarmPartialFactsV4 {
        PalsCudaWarmPartialFactsV4 {
            backend_owner_id: Some(a.backend_owner_id),
            bank_owner_id: Some(a.bank_owner_id),
            physical_event_sequence: Some(a.physical_event_sequence),
            seed_event_sequence: Some(a.seed_event_sequence),
            leases_admitted: Some(a.leases_admitted),
            leases_physically_completed: Some(a.leases_physically_completed),
            leases_quarantined: Some(a.leases_quarantined),
            device_bytes_peak: Some(a.device_bytes_peak),
            accepted_seed_consumptions: Some(a.accepted_seed_consumptions),
            fresh_value_evaluations: Some(a.fresh_value_evaluations),
            backend_dropped: Some(a.backend_dropped),
            worker_joined: Some(a.worker_joined),
            bank_closed: Some(a.bank_closed),
            complete: a.complete,
        }
    }
    fn perspective(raw: &str) -> Result<PalsPerspectiveV4, ArenaError> {
        match raw {
            "white" => Ok(PalsPerspectiveV4::White),
            "black" => Ok(PalsPerspectiveV4::Black),
            _ => Err(invalid("actual WDL perspective unsupported")),
        }
    }
    fn context(c: &NativeRoleContextV4, game: u64, purpose: &str) -> Result<(), ArenaError> {
        require(
            c.game_generation == game
                && c.purpose == purpose
                && [
                    &c.focus_sha256,
                    &c.prefix_sha256,
                    &c.proposal_sha256,
                    &c.divergence_sha256,
                ]
                .iter()
                .all(|s| hash(s))
                && c.refutation_sha256.as_ref().is_none_or(|s| hash(s)),
            "actual role context/game/purpose/input identity differs",
        )
    }
    /// Domain and canonical producer DTO bytes, each length-framed. No policy
    /// declaration, handle address or fabricated relation enters this identity.
    fn evidence_hash<T: Serialize>(domain: &str, raw: &T) -> Result<String, ArenaError> {
        let bytes = serde_json::to_vec(raw).map_err(|e| invalid(e.to_string()))?;
        let mut h = Sha256::new();
        for frame in [domain.as_bytes(), bytes.as_slice()] {
            h.update((frame.len() as u64).to_be_bytes());
            h.update(frame);
        }
        Ok(hex(h.finalize().into()))
    }
    fn recheck_value(
        raw: &NativeRecheckValueV4,
        e: &PalsEndpointV4,
        game: u64,
        epoch: u64,
        comparison: PalsPerspectiveV4,
        native: &serde_json::Value,
    ) -> Result<(PalsRecheckValueV4, Option<PalsRecheckModelProvenanceV4>), ArenaError> {
        match raw {
            NativeRecheckValueV4::Unknown => Ok((PalsRecheckValueV4::Unknown, None)),
            NativeRecheckValueV4::RulesTerminal {
                state_sha256,
                perspective: original,
                winner,
                ..
            } => {
                require(hash(state_sha256), "actual terminal state digest invalid")?;
                perspective(original)?;
                Ok((
                    PalsRecheckValueV4::RulesTerminal {
                        state_sha256: state_sha256.clone(),
                        perspective: comparison,
                        winner: winner.as_deref().map(perspective).transpose()?,
                    },
                    None,
                ))
            }
            NativeRecheckValueV4::FrozenWdl {
                model_value_semantics,
                model_identity,
                encoding_identity,
                precision,
                model_epoch_sha256,
                input_sha256,
                state_sha256,
                perspective: original,
                wdl_bits,
                context_revision,
                accepted_output,
                execution,
                ..
            } => {
                let physical = execution
                    .as_ref()
                    .ok_or_else(|| invalid("actual Fresh execution unknown"))?;
                let checkpoint = match &e.base.model.weights {
                    PalsWeightIdentityV3::Untrained { artifact, .. }
                    | PalsWeightIdentityV3::Trained { artifact, .. } => &artifact.sha256,
                    _ => return Err(invalid("Fresh WDL cannot inherit a mock checkpoint")),
                };
                let original_perspective = perspective(original)?;
                require(
                    model_value_semantics == rz_search::pals::value::MODEL_WDL_VALUE_SEMANTICS
                        && model_identity == &e.model_v2.model_identity
                        && encoding_identity == &e.model_v2.encoding_sha256
                        && model_epoch_sha256 == checkpoint
                        && model_epoch_sha256 == &physical.model_epoch_sha256
                        && input_sha256 == &physical.input_sha256
                        && hash(state_sha256)
                        && hash(input_sha256)
                        && physical.complete
                        && physical.fresh == Some(true)
                        && physical.physically_completed == Some(true)
                        && perspective(&physical.prepared_perspective)? == original_perspective
                        && physical.request.epoch == epoch
                        && physical.execution.epoch == epoch
                        && physical.request.sequence > 0
                        && physical.execution.sequence > 0
                        && physical.request.sequence <= count(native, "request_high_water")?
                        && physical.execution.sequence <= count(native, "execution_high_water")?
                        && count(native, "frozen_epoch")? == e.base.model.frozen_epoch,
                    "actual Fresh model/context/physical ID/deployment epoch differs",
                )?;
                context(&physical.context, game, "ValueFresh")?;
                let mut wdl = wdl_bits.map(f32::from_bits);
                require(
                    wdl.iter().all(|n| n.is_finite() && (0.0..=1.0).contains(n))
                        && (wdl.iter().sum::<f32>() - 1.0).abs() <= 1e-4,
                    "actual Fresh WDL is nonfinite or unnormalized",
                )?;
                if original_perspective != comparison {
                    wdl.swap(0, 2);
                }
                let provenance = PalsRecheckModelProvenanceV4 {
                    model_value_semantics: model_value_semantics.clone(),
                    model_identity: model_identity.clone(),
                    encoding_identity: encoding_identity.clone(),
                    model_epoch_sha256: model_epoch_sha256.clone(),
                    original_perspective,
                    wdl_bits: *wdl_bits,
                    state_sha256: state_sha256.clone(),
                    input_sha256: input_sha256.clone(),
                    context_revision: *context_revision,
                    prepared_state: PalsPreparedStateV4 {
                        owner: physical.prepared_state.owner,
                        revision: physical.prepared_state.revision,
                        semantic_sha256: physical.prepared_state.semantic_sha256.clone(),
                    },
                    fresh_call_attempted: true,
                    native_request: Some(PalsPhysicalIdV4 {
                        epoch: physical.request.epoch,
                        sequence: physical.request.sequence,
                    }),
                    native_execution: Some(PalsPhysicalIdV4 {
                        epoch: physical.execution.epoch,
                        sequence: physical.execution.sequence,
                    }),
                    completed_nn_inputs: physical.completed_nn_inputs,
                    physical_input_completed: true,
                    accepted_output: *accepted_output,
                };
                let value = if *accepted_output {
                    PalsRecheckValueV4::FrozenWdl {
                        model_sha256: checkpoint.clone(),
                        model_epoch: e.base.model.frozen_epoch,
                        encoding_sha256: encoding_identity.clone(),
                        precision: precision.clone(),
                        context_revision: *context_revision,
                        input_sha256: input_sha256.clone(),
                        perspective: comparison,
                        fresh: true,
                        wdl,
                    }
                } else {
                    PalsRecheckValueV4::Unknown
                };
                Ok((value, Some(provenance)))
            }
        }
    }
    fn recheck(
        raw: &NativeRecheckObservationV4,
        e: &PalsEndpointV4,
        game: u64,
        epoch: u64,
        native: &serde_json::Value,
    ) -> Result<PalsRecheckObservationV4, ArenaError> {
        require(raw.complete, "actual recheck comparison capture incomplete")?;
        let comparison = perspective(&raw.comparison_perspective)?;
        let (before, before_source) =
            recheck_value(&raw.before, e, game, epoch, comparison, native)?;
        let (after, after_source) = recheck_value(&raw.after, e, game, epoch, comparison, native)?;
        let ordering = match (&before, &after) {
            (
                PalsRecheckValueV4::FrozenWdl {
                    wdl: a,
                    context_revision: c,
                    ..
                },
                PalsRecheckValueV4::FrozenWdl {
                    wdl: b,
                    context_revision: d,
                    ..
                },
            ) => {
                require(
                    c == d && raw.context_revision == Some(*c),
                    "actual Fresh comparison context revision differs",
                )?;
                (a[0] - a[2]).partial_cmp(&(b[0] - b[2]))
            }
            (
                PalsRecheckValueV4::RulesTerminal { winner: a, .. },
                PalsRecheckValueV4::RulesTerminal { winner: b, .. },
            ) => {
                let score = |winner: &Option<PalsPerspectiveV4>| match winner {
                    None => 0,
                    Some(w) if *w == comparison => 1,
                    Some(_) => -1,
                };
                Some(score(a).cmp(&score(b)))
            }
            _ => None,
        };
        require(
            raw.comparable == ordering.is_some(),
            "actual comparison authority and common value basis differ",
        )?;
        let resolution = match ordering {
            Some(std::cmp::Ordering::Greater) => PalsRecheckResolutionV4::BeforePreferred,
            Some(std::cmp::Ordering::Less) => PalsRecheckResolutionV4::AfterPreferred,
            Some(std::cmp::Ordering::Equal) => PalsRecheckResolutionV4::Equal,
            None => PalsRecheckResolutionV4::Unresolved,
        };
        Ok(PalsRecheckObservationV4 {
            before,
            after,
            resolution,
            fresh_nn_inputs_charged: optional_metric(
                raw.completed_nn_inputs,
                "actual Fresh physical input cost unknown",
            )?,
            actual_scope: Some(PalsRecheckActualScopeV4 {
                comparison_perspective: comparison,
                before: before_source,
                after: after_source,
                fresh_requests_completed: optional_metric(
                    raw.fresh_requests_completed,
                    "actual Fresh request count unknown",
                )?,
                comparison_attempted: raw.comparison_attempted,
                publication: raw.publication,
                original_error: raw.original_error.clone(),
            }),
        })
    }
    fn trace(
        raw: &NativeRepairTraceV4,
        e: &PalsEndpointV4,
        epoch: u64,
        native: &serde_json::Value,
    ) -> Result<PalsRepairTraceV4, ArenaError> {
        let question = raw
            .accepted_question
            .as_ref()
            .ok_or_else(|| invalid("actual accepted Repair question missing"))?;
        context(question, raw.game_generation, "RepairPolicy")?;
        require(
            raw.repaired.first() == Some(&raw.first_move)
                && !raw.repaired.is_empty()
                && raw.repaired.len() <= 256
                && raw
                    .repaired
                    .iter()
                    .all(|mv| rz_position::BoardMove::from_uci(mv).is_ok()),
            "actual repaired Move16 path invalid",
        )?;
        let reply_recheck = match &raw.recheck {
            Some(r) => {
                context(&r.reply_context, raw.game_generation, "ReplyPolicy")?;
                let result = require(
                    r.prepared_accepted
                        && r.reply_call_attempted
                        && r.reply_accepted
                        && r.counterline_completed
                        && r.original_error.is_none()
                        && r.counterline.len() <= 256
                        && r.counterline
                            .iter()
                            .all(|mv| rz_position::BoardMove::from_uci(mv).is_ok())
                        && r.selected_response
                            .as_ref()
                            .is_some_and(|response| r.counterline.contains(response)),
                    "actual accepted Repair has no accepted/completed reply C",
                )
                .and_then(|()| recheck(&r.comparison, e, raw.game_generation, epoch, native));
                match result {
                Ok(value)=>PalsObservedV3::Observed{value,method:"actual accepted Repair → reply C → common frozen WDL/Rules comparison; native physical IDs and backend deltas verified".into()},
                Err(_)=>PalsObservedV3::Unknown,
            }
            }
            None => PalsObservedV3::Unknown,
        };
        let iterative_repair_comparison=raw.iterative_repair_comparison.as_ref().map(|comparison| {
            match recheck(comparison,e,raw.game_generation,epoch,native) {
                Ok(value)=>PalsObservedV3::Observed{value,method:"actual iterative Repair candidate-versus-prior-C endpoint comparison; independent physical Fresh bindings".into()},
                Err(_)=>PalsObservedV3::Unknown,
            }
        });
        Ok(PalsRepairTraceV4 {
            root_generation: raw.store_root_generation,
            first_move: raw.first_move.clone(),
            original_record: optional_metric(
                raw.parent_revision,
                "actual Counter parent revision unknown",
            )?,
            record_id: raw.record_revision,
            supersedes: raw.supersedes_revision,
            question_sha256: evidence_hash("rz-pals-followup-question-v4/1", question)?,
            evidence_revision: question.situation_revision,
            repair_ordinal: raw.repair_ordinal,
            evidence_sha256: evidence_hash("rz-pals-followup-repair-evidence-v4/1", raw)?,
            recheck: reply_recheck,
            iterative_repair_comparison,
            actual_scope: Some(PalsRepairActualScopeV4 {
                game_generation: raw.game_generation,
                search_attempt_sequence: raw.search_attempt_sequence,
                store_root_generation: raw.store_root_generation,
                lineage_root_record: raw.lineage_root_record,
                parent_revision: raw.parent_revision,
                supersedes_revision: raw.supersedes_revision,
            }),
        })
    }
    fn nonempty(p: &PalsFollowupPartialFactsV4) -> bool {
        p.archive.is_some()
            || p.paused_stack.is_some()
            || p.cuda_warm.is_some()
            || p.repair.is_some()
            || !p.previous_closed_archives.is_empty()
    }
    pub(in crate::pals_launch::v4) fn project(
        lock: &LockedPalsArenaLaunchV4,
        role: NativeEngineRole,
        work: &PalsProcessWorkAuditV4,
        native: Option<&PalsNativeSessionAuditV3>,
        out: &mut PalsEndpointReceiptV4,
    ) -> Result<(), ArenaError> {
        let Some(raw) = &work.followup_lifecycle else {
            return Ok(());
        };
        let source = typed(raw)?;
        let PalsEngineV4::Pals(e) = &lock.input.semantic_lock.manifest.engines[role_index(role)]
        else {
            return Err(invalid("non-PALS lifecycle projection"));
        };
        let mut partial = PalsFollowupPartialFactsV4::default();
        let mut previous_closed_archive_owners = Vec::new();
        for previous in &source.previous_closed_archive_owners {
            match archive(previous) {
                Ok(projected) => {
                    lock.verify_archive_scope_roots(
                        role,
                        projected.actual_scope.as_ref().unwrap(),
                    )?;
                    previous_closed_archive_owners.push(projected);
                }
                Err(_) => partial
                    .previous_closed_archives
                    .push(archive_partial(previous)),
            }
        }
        match &source.archive {
            NativeObservedV4::Observed { value, method } if value.complete => {
                if let Ok(a) = archive(value) {
                    lock.verify_archive_scope_roots(role, a.actual_scope.as_ref().unwrap())?;
                    out.archive = PalsObservedV3::Observed {
                        value: a,
                        method: method.clone(),
                    };
                } else {
                    partial.archive = Some(archive_partial(value));
                }
            }
            NativeObservedV4::Observed { value, .. } => {
                partial.archive = Some(archive_partial(value))
            }
            NativeObservedV4::Unknown => {}
        }
        match &source.paused_stack {
            NativeObservedV4::Observed { value, method } if value.complete => {
                out.paused_stack = PalsObservedV3::Observed {
                    value: stack(value),
                    method: method.clone(),
                }
            }
            NativeObservedV4::Observed { value, .. } => partial.paused_stack = Some(stack(value)),
            NativeObservedV4::Unknown => {}
        }
        match &source.cuda_warm {
            NativeObservedV4::Observed { value, method } if value.complete => {
                out.cuda_warm = PalsObservedV3::Observed {
                    value: warm(value),
                    method: method.clone(),
                }
            }
            NativeObservedV4::Observed { value, .. } => {
                partial.cuda_warm = Some(warm_partial(value))
            }
            NativeObservedV4::Unknown => {}
        }
        if let Some(p) = &source.partial_facts {
            if let Some(a) = &p.archive {
                out.archive = PalsObservedV3::Unknown;
                partial.archive = Some(archive_partial(a));
            }
            if let Some(a) = &p.paused_stack {
                out.paused_stack = PalsObservedV3::Unknown;
                partial.paused_stack = Some(stack(a));
            }
            if let Some(a) = &p.cuda_warm {
                out.cuda_warm = PalsObservedV3::Unknown;
                partial.cuda_warm = Some(warm_partial(a));
            }
        }
        if e.policies.recheck != PalsRecheckPolicyV4::Disabled {
            out.repair_trace_total = fact(&source.repair_trace_total);
            out.pending_questions_peak = fact(&source.pending_questions_peak);
            if !matches!(
                e.policies.iterative_repair,
                PalsIterativeRepairPolicyV4::Disabled
            ) {
                out.repair_admissions_peak = Some(
                    source
                        .repair_admissions_peak
                        .as_ref()
                        .map_or(PalsObservedV3::Unknown, fact),
                );
            }
            for raw in &source.repair_traces {
                if let Some((epoch, n)) = source.process_epoch.zip(native) {
                    match trace(raw, e, epoch, &n.raw_native) {
                        Ok(t) => out.repair_traces.push(t),
                        Err(_) => {
                            partial.repair = Some(PalsRepairPartialFactsV4 {
                                repair_trace_total: known(&source.repair_trace_total).copied(),
                                pending_questions_peak: known(&source.pending_questions_peak)
                                    .copied(),
                                repair_admissions_peak: source
                                    .repair_admissions_peak
                                    .as_ref()
                                    .and_then(known)
                                    .copied(),
                                complete: false,
                            });
                        }
                    }
                } else {
                    partial.repair = Some(PalsRepairPartialFactsV4 {
                        repair_trace_total: known(&source.repair_trace_total).copied(),
                        pending_questions_peak: known(&source.pending_questions_peak).copied(),
                        repair_admissions_peak: source
                            .repair_admissions_peak
                            .as_ref()
                            .and_then(known)
                            .copied(),
                        complete: false,
                    });
                }
            }
        }
        let projected_complete = !nonempty(&partial)
            && out.repair_traces.iter().all(|t| {
                matches!(t.recheck, PalsObservedV3::Observed { .. })
                    && t.iterative_repair_comparison
                        .as_ref()
                        .is_none_or(|c| matches!(c, PalsObservedV3::Observed { .. }))
            });
        let methods = owner_methods(&source)?;
        let capture_failure=source.capture_failure.map(|v|match v {
            rz_uci::pals_attestation::followup::PalsCaptureFailureV4::CaptureIncomplete=>PalsCaptureFailureV4::CaptureIncomplete,
            rz_uci::pals_attestation::followup::PalsCaptureFailureV4::CounterOverflow=>PalsCaptureFailureV4::CounterOverflow,
            rz_uci::pals_attestation::followup::PalsCaptureFailureV4::OwnerUnknown=>PalsCaptureFailureV4::OwnerUnknown,
            rz_uci::pals_attestation::followup::PalsCaptureFailureV4::PhysicalCompletionUnknown=>PalsCaptureFailureV4::PhysicalCompletionUnknown,
            rz_uci::pals_attestation::followup::PalsCaptureFailureV4::ContextMismatch=>PalsCaptureFailureV4::ContextMismatch,
        }).or_else(||(!projected_complete).then_some(PalsCaptureFailureV4::CaptureIncomplete));
        out.followup_lifecycle = Some(PalsFollowupLifecycleScopeV4 {
            domain: source.domain,
            schema_version: source.schema_version,
            process_epoch: source.process_epoch,
            game_generation: source.game_generation,
            capture_complete: source.capture_complete && projected_complete,
            owner_shutdown_complete: source.owner_shutdown_complete && projected_complete,
            trace_capacity: source.trace_capacity,
            trace_bytes_max: source.trace_bytes_max,
            owner_methods: methods,
            capture_failure,
            partial_facts: nonempty(&partial).then_some(partial),
            previous_closed_archive_owners,
        });
        Ok(())
    }
}
#[cfg(target_os = "linux")]
pub(super) use projection::project;

#[cfg(test)]
mod tests {
    use super::*;
    use rz_uci::pals_attestation::followup::{
        NativeArchiveRuntimeLimitsV4, NativeFollowupOwnerMethodsV4,
    };

    fn archive() -> NativeArchiveObservationV4 {
        NativeArchiveObservationV4 {
            measurement_contract: "rz-pals-cold-archive-owner-accounting-v4/1".into(),
            owner_method: "archive_actual_owner_events_v4".into(),
            compiled_source_sha256: "b".repeat(64),
            owner_id: 1,
            event_sequence: 1,
            root_sha256: Some("c".repeat(64)),
            repository_sha256: Some("d".repeat(64)),
            generation: 0,
            reserved_generations: 0,
            last_committed_generation: None,
            last_verified_generation: None,
            lifecycle: "open".into(),
            runtime_limits: Some(NativeArchiveRuntimeLimitsV4 {
                game_bytes_max: 4096,
                global_bytes_max: 8192,
                index_entries_max: 16,
                index_bytes_max: 1024,
                load_bytes_max: 1024,
                load_deadline_max_ms: 100,
                record_payload_bytes_max: 4096,
                max_load_pins: 16,
            }),
            root_boundary_checked: true,
            committed_chunks: 0,
            integrity_verified_chunks: 0,
            committed_bytes: 0,
            integrity_verified_bytes: 0,
            ram_released_bytes: Some(0),
            ram_release_method: "hot_unique_owned_capacity_subset".into(),
            ram_reclaim_events: 0,
            integrity_before_reclaim_events: 0,
            unverified_reclaim_events: 0,
            pending_commit_bytes: Some(0),
            pending_buffers_retained: false,
            pending_files_retained: false,
            global_managed_bytes: Some(0),
            global_scan_event_sequence: Some(1),
            global_scan_complete: true,
            global_scan_after_last_commit: true,
            global_scope_kind: "canonical_no_link_managed_root".into(),
            index_entries_peak: 0,
            index_bytes_peak: 0,
            cold_index_kind: "directory_scan".into(),
            pinned_entries_peak: 0,
            load_pins_max: 16,
            loaded_closure_bytes_peak: None,
            loaded_closure_bytes_method: None,
            loads_requested: 0,
            loads_completed: 0,
            load_bytes_total: 0,
            load_bytes_peak: 0,
            load_elapsed_peak_ms: 0,
            owner_generation_checks: 0,
            owner_generation_check_failures: 0,
            quota_failures: 0,
            io_failures: 0,
            pin_saturation_failures: 0,
            archive_write_bytes_total: Some(0),
            session_write_ledger_id: Some(17),
            session_write_bytes_max: Some(4096),
            session_write_bytes_consumed: Some(0),
            admission_closed: false,
            cleanup_complete: false,
            complete: true,
        }
    }

    fn raw_empty() -> PalsFollowupLifecycleV4 {
        PalsFollowupLifecycleV4 {
            domain: PALS_FOLLOWUP_LIFECYCLE_V4_DOMAIN.into(),
            schema_version: 1,
            process_epoch: None,
            game_generation: 1,
            capture_complete: true,
            owner_shutdown_complete: false,
            trace_capacity: 64,
            trace_bytes_max: 96 * 1024,
            owner_methods: NativeFollowupOwnerMethodsV4 {
                producer_domain: PALS_FOLLOWUP_LIFECYCLE_V4_DOMAIN.into(),
                producer_schema_version: 1,
                producer_implementation_sha256: "b".repeat(64),
                trace_capacity: 64,
                trace_bytes_max: 96 * 1024,
                repair: None,
                paused_stack: None,
                cold_archive: None,
                cuda_warm: None,
            },
            archive: NativeObservedV4::Unknown,
            paused_stack: NativeObservedV4::Unknown,
            cuda_warm: NativeObservedV4::Unknown,
            repair_traces: vec![],
            repair_trace_total: NativeObservedV4::Unknown,
            pending_questions_peak: NativeObservedV4::Unknown,
            repair_admissions_peak: None,
            partial_facts: None,
            capture_failure: None,
            previous_closed_archive_owners: vec![],
        }
    }
    #[test]
    fn actual_codec_requires_nullable_presence_and_rejects_unregistered_members() {
        let good = serde_json::to_value(raw_empty()).unwrap();
        typed(&good).unwrap();
        let mut missing = good.clone();
        missing.as_object_mut().unwrap().remove("process_epoch");
        assert!(typed(&missing).is_err());
        let mut injected = good.clone();
        injected["fabricated_owner_count"] = 1.into();
        assert!(typed(&injected).is_err());
        let mut missing_method = good;
        missing_method["owner_methods"]
            .as_object_mut()
            .unwrap()
            .remove("cold_archive");
        assert!(typed(&missing_method).is_err());
        let mut source = raw_empty();
        source.archive = NativeObservedV4::Observed {
            value: archive(),
            method: "archive_actual_owner_events_v4".into(),
        };
        let mut nullable = serde_json::to_value(source).unwrap();
        assert_eq!(
            known(&typed(&nullable).unwrap().archive)
                .unwrap()
                .session_write_bytes_consumed,
            Some(0)
        );
        nullable["archive"]["value"]["session_write_bytes_consumed"] = serde_json::Value::Null;
        assert_eq!(
            known(&typed(&nullable).unwrap().archive)
                .unwrap()
                .session_write_bytes_consumed,
            None
        );
        nullable["archive"]["value"]
            .as_object_mut()
            .unwrap()
            .remove("session_write_bytes_consumed");
        assert!(typed(&nullable).is_err());
    }
    fn stack() -> NativePausedStackObservationV4 {
        NativePausedStackObservationV4 {
            owner_id: 1,
            event_sequence: 1,
            tokens_created: 1,
            tokens_resumed: 0,
            tokens_invalidated: 0,
            tokens_retained: 1,
            tokens_peak: 1,
            bytes_current: 64,
            bytes_peak: 64,
            stale_context_attempts: 0,
            stale_context_rejections: 0,
            replayed_consumed_work: 0,
            admission_closed: false,
            owner_released: false,
            complete: true,
        }
    }
    #[test]
    fn actual_stack_transition_uses_owner_lifetime_without_summing_go_snapshots() {
        let before = stack();
        let mut after = before;
        after.tokens_resumed = 1;
        after.tokens_retained = 0;
        after.bytes_current = 0;
        after.event_sequence = 2;
        after.admission_closed = true;
        after.owner_released = true;
        stack_transition(&before, &after).unwrap();
        after.tokens_created = 0;
        assert!(stack_transition(&before, &after).is_err());
        after.tokens_created = 1;
        after.owner_id = 2;
        assert!(stack_transition(&before, &after).is_err());
        after.owner_id = 1;
        assert!(stack_transition(&after, &before).is_err());
    }

    #[test]
    fn actual_archive_owner_changes_keep_one_monotonic_session_write_ledger() {
        let mut before = archive();
        before.archive_write_bytes_total = Some(64);
        before.session_write_bytes_consumed = Some(64);
        let mut after = before.clone();
        after.owner_id = 2;
        after.session_write_bytes_consumed = Some(128);
        session_write_transition(&before, &after).unwrap();
        assert!(archive_transition(&before, &after).is_err());
        after.session_write_ledger_id = Some(18);
        assert!(session_write_transition(&before, &after).is_err());
        after.session_write_ledger_id = Some(17);
        after.session_write_bytes_consumed = Some(0);
        assert!(session_write_transition(&before, &after).is_err());
        after.session_write_bytes_consumed = None;
        assert!(session_write_transition(&before, &after).is_err());
        after.complete = false;
        session_write_transition(&before, &after).unwrap();
        after = before.clone();
        after.event_sequence += 1;
        after.archive_write_bytes_total = Some(96);
        after.session_write_bytes_consumed = Some(96);
        archive_transition(&before, &after).unwrap();
        after.archive_write_bytes_total = Some(32);
        assert!(archive_transition(&before, &after).is_err());
    }
}
