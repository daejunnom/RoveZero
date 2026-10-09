//! Actual-owner Repair -> next Query/3. Imported JSON remains a report. The
//! native observation is supplied by a separate live checked owner, under the
//! same finite episode; its physical IDs and timing are never equated to the
//! child. This admits a conditional prior, not utility, training or Rules proof.

use super::{ReportedRepairPriorMaterial, control, invalid};
use crate::ArenaError;
use rz_uci::pals_cpu_task::strategic_action::ArtifactPin;
use rz_uci::pals_cpu_task::strategic_action::native_replay::query_prior::{
    CheckedNativeReplayPrior, PriorReadiness,
};
use rz_uci::pals_cpu_task::strategic_action::replay_inputs::{ReplayBindingPins, ReplayParentPins};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

const MAX_PRIOR: usize = 16;
const MAX_BYTES: usize = 4 * 1024 * 1024;
const MAX_TRACE_ROWS: usize = 4096;
const SCHEMA: &str = "rz-pals-actual-repair-next-query/3";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeClockScope {
    /// S starts at action preparation; the earlier V/coverage choice is unknown.
    PreparationOnly,
    /// An actual callback was observed before preparation; this does not certify
    /// that the callback was a trained V or a particular selector algorithm.
    BeforeSelectionCallback,
}

/// Finite mutable ledger, with no Clone/serde or global sequence supplied by a
/// report. Each capture's issue gate is also spent across different episodes.
pub struct RepairQueryEpisode {
    parent: ReplayParentPins,
    source_query: String,
    ledger: String,
    ordinal: usize,
    maximum_prior: usize,
    started: Instant,
    whole: Instant,
    last_admitted: Instant,
    scope: EpisodeClockScope,
    selection_interval: Option<(u64, u64)>,
    seen_inputs: BTreeSet<String>,
}

/// Borrows both real owners. No deserialization, Clone or public constructor.
/// The independently re-executed native fact remains conditional and cannot
/// prove that a physical child executed the same operation or amount of work.
pub struct CheckedRepairQuery<'a> {
    material: &'a ReportedRepairPriorMaterial<'a>,
    native: Option<&'a CheckedNativeReplayPrior<'a>>,
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    captured: Option<&'a super::CheckedCapturedRepairWitness<'a>>,
    parent: ReplayParentPins,
    source_binding: ReplayBindingPins,
    query: String,
    ledger_before: String,
    ledger: String,
    ordinal: usize,
    started: Instant,
    whole: Instant,
    admitted: Instant,
    elapsed: u64,
    ledger_event_elapsed: u64,
    scope: EpisodeClockScope,
    selection_interval: Option<(u64, u64)>,
    fact: String,
    catalogue: ArtifactPin,
    reported_work_counts: Option<(u64, u64)>,
}

impl RepairQueryEpisode {
    /// Compatibility observation path. This is never relabelled a full episode
    /// because the earlier selector's execution has not been observed here.
    pub fn from_original(
        material: &ReportedRepairPriorMaterial<'_>,
        maximum_prior: usize,
        cancel: &AtomicBool,
    ) -> Result<Self, ArenaError> {
        let bundle = material.checked_report().caller_timing().capture().bundle();
        Self::new(
            material.original_parent(),
            material.previous_query_binding(),
            maximum_prior,
            bundle.original_started(),
            bundle.deadline(),
            EpisodeClockScope::PreparationOnly,
            None,
            cancel,
        )
    }

    /// Record an actual selection callback before any action preparation. The
    /// caller must use the returned decision to prepare the original action;
    /// source Query/parent/budget are fixed before this callback, never afterwards.
    pub fn begin_before_selection<T>(
        parent: &ReplayParentPins,
        source: &ReplayBindingPins,
        maximum_prior: usize,
        whole_budget: Duration,
        cancel: &AtomicBool,
        select: impl FnOnce(Instant, Instant) -> Result<T, ArenaError>,
    ) -> Result<(T, Self), ArenaError> {
        let started = Instant::now();
        if whole_budget.is_zero() || whole_budget > Duration::from_secs(300) {
            return Err(invalid("Repair episode finite whole budget"));
        }
        let whole = started
            .checked_add(whole_budget)
            .ok_or_else(|| invalid("Repair episode deadline extent"))?;
        let mut episode = Self::new(
            parent,
            source,
            maximum_prior,
            started,
            whole,
            EpisodeClockScope::BeforeSelectionCallback,
            None,
            cancel,
        )?;
        let value = select(started, whole)?;
        let returned = Instant::now();
        control(whole, cancel)?;
        episode.selection_interval = Some((0, nanos(started, returned)?));
        Ok((value, episode))
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        parent: &ReplayParentPins,
        source: &ReplayBindingPins,
        maximum_prior: usize,
        started: Instant,
        whole: Instant,
        scope: EpisodeClockScope,
        selection_interval: Option<(u64, u64)>,
        cancel: &AtomicBool,
    ) -> Result<Self, ArenaError> {
        control(whole, cancel)?;
        if !(1..=MAX_PRIOR).contains(&maximum_prior) || started >= whole {
            return Err(invalid("Repair episode prior extent/clock"));
        }
        for pin in [
            &parent.parent_input_sha256,
            &parent.current_view_sha256,
            &parent.frozen_admission_sha256,
            &parent.encoding_sha256,
            &source.query_sha256,
            &source.prior_ledger_sha256,
        ] {
            check_sha(pin)?;
        }
        Ok(Self {
            parent: parent.clone(),
            source_query: source.query_sha256.clone(),
            ledger: source.prior_ledger_sha256.clone(),
            ordinal: 0,
            maximum_prior,
            started,
            whole,
            last_admitted: started,
            scope,
            selection_interval,
            seen_inputs: BTreeSet::new(),
        })
    }

    pub fn original_started(&self) -> Instant {
        self.started
    }
    pub fn original_deadline(&self) -> Instant {
        self.whole
    }

    /// Admit the independently witnessed conditional result under actual parent
    /// chronology. The exact existing catalogue bytes are pinned, but its action
    /// semantics remain the original Query/2 consumer's responsibility.
    pub fn admit_repair_prior<'a>(
        &mut self,
        material: &'a ReportedRepairPriorMaterial<'a>,
        native: &'a CheckedNativeReplayPrior<'a>,
        catalogue_raw: &[u8],
        independently_expected_catalogue: &ArtifactPin,
        cancel: &AtomicBool,
    ) -> Result<CheckedRepairQuery<'a>, ArenaError> {
        control(self.whole, cancel)?;
        let original = material.original_input();
        let bundle = material.checked_report().caller_timing().capture().bundle();
        check_next_source(
            self,
            material.original_parent(),
            material.previous_query_binding(),
            original.artifact(),
            bundle.original_started(),
            bundle.deadline(),
            material.before_next_query_at(),
        )?;
        if self.selection_interval.is_some_and(|(_, ended)| {
            nanos(self.started, bundle.original_started())
                .and_then(|offset| {
                    offset
                        .checked_add(material.caller_started_ns())
                        .ok_or_else(|| invalid("Repair selection clock extent"))
                })
                .is_ok_and(|caller_started| caller_started < ended)
        }) {
            return Err(invalid(
                "Repair caller precedes observed selection completion",
            ));
        }
        let reservation = material
            .next_query_issue
            .reserve(bundle.deadline(), cancel)?;
        check_catalogue(
            catalogue_raw,
            independently_expected_catalogue,
            &self.parent,
            &material.previous_query_binding().catalogue_artifact,
        )?;
        let (fact, reported_work_counts) = check_independent_native(material, native)?;
        check_native_interval(
            native.projection().native_work_interval(),
            self.started,
            self.whole,
            Instant::now(),
        )?;
        control(self.whole, cancel)?;
        let ledger_event_elapsed = nanos(self.started, Instant::now())?;
        let ledger_before = self.ledger.clone();
        let event = json!({"ordinal": self.ordinal, "source_query": self.source_query,
            "previous_ledger": self.ledger, "source_input": original.artifact(),
            "source_output": material.checked_report().body_artifact(),
            "conditional_fact_sha256": fact, "ledger_event_elapsed_ns": ledger_event_elapsed});
        let ledger = digest("rz-pals-actual-repair-prior-ledger/3", &event)?;
        let identity = json!({"schema": SCHEMA, "parent": self.parent,
            "source_query_sha256": self.source_query, "prior_ledger_sha256": ledger,
            "decision_ordinal": self.ordinal + 1, "catalogue": independently_expected_catalogue});
        let query = digest(SCHEMA, &identity)?;
        // The capture is spent before publishing the new ledger. Downstream
        // cancellation/drop cannot let another episode issue this child again.
        reservation.issue(bundle.deadline(), cancel)?;
        control(self.whole, cancel)?;
        let admitted = Instant::now();
        let elapsed = nanos(self.started, admitted)?;
        let mut result = CheckedRepairQuery {
            material,
            native: Some(native),
            #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
            captured: None,
            parent: self.parent.clone(),
            source_binding: material.previous_query_binding().clone(),
            query: query.clone(),
            ledger_before,
            ledger: ledger.clone(),
            ordinal: self.ordinal,
            started: self.started,
            whole: self.whole,
            admitted,
            elapsed,
            ledger_event_elapsed,
            scope: self.scope,
            selection_interval: self.selection_interval,
            fact,
            catalogue: independently_expected_catalogue.clone(),
            reported_work_counts,
        };
        self.seen_inputs.insert(original.artifact().sha256.clone());
        self.ordinal += 1;
        self.ledger = ledger;
        self.source_query = query;
        control(self.whole, cancel)?;
        let returned = Instant::now();
        if returned >= self.whole {
            return Err(invalid("Repair Query admission passed original W"));
        }
        result.admitted = returned;
        result.elapsed = nanos(self.started, returned)?;
        self.last_admitted = returned;
        Ok(result)
    }

    /// Additive exact-captured-input lane. This does not pretend that a second
    /// live generation recreated the historical deadline feature or child score.
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    pub fn admit_captured_repair_prior<'a>(
        &mut self,
        material: &'a ReportedRepairPriorMaterial<'a>,
        captured: &'a super::CheckedCapturedRepairWitness<'a>,
        catalogue_raw: &[u8],
        independently_expected_catalogue: &ArtifactPin,
        cancel: &AtomicBool,
    ) -> Result<CheckedRepairQuery<'a>, ArenaError> {
        control(self.whole, cancel)?;
        let original = material.original_input();
        let bundle = material.checked_report().caller_timing().capture().bundle();
        check_next_source(
            self,
            material.original_parent(),
            material.previous_query_binding(),
            original.artifact(),
            bundle.original_started(),
            bundle.deadline(),
            material.before_next_query_at(),
        )?;
        if !std::ptr::eq(material, captured.material()) {
            return Err(invalid(
                "captured Repair witness belongs to another material owner",
            ));
        }
        if self.selection_interval.is_some_and(|(_, ended)| {
            nanos(self.started, bundle.original_started())
                .and_then(|offset| {
                    offset
                        .checked_add(material.caller_started_ns())
                        .ok_or_else(|| invalid("Repair selection clock extent"))
                })
                .is_ok_and(|at| at < ended)
        }) {
            return Err(invalid(
                "captured Repair caller precedes original selection",
            ));
        }
        check_native_interval(
            Some((captured.work_started_at(), captured.work_finished_at())),
            self.started,
            self.whole,
            Instant::now(),
        )?;
        let reservation = material
            .next_query_issue
            .reserve(bundle.deadline(), cancel)?;
        check_catalogue(
            catalogue_raw,
            independently_expected_catalogue,
            &self.parent,
            &material.previous_query_binding().catalogue_artifact,
        )?;
        let fact = captured.fact_sha256().to_owned();
        let ledger_event_elapsed = nanos(self.started, Instant::now())?;
        let ledger_before = self.ledger.clone();
        let event = json!({"ordinal":self.ordinal,"source_query":self.source_query,
            "previous_ledger":self.ledger,"source_input":original.artifact(),
            "source_output":material.checked_report().body_artifact(),
            "conditional_fact_sha256":fact,"ledger_event_elapsed_ns":ledger_event_elapsed});
        let ledger = digest("rz-pals-actual-repair-prior-ledger/3", &event)?;
        let identity = json!({"schema":SCHEMA,"parent":self.parent,"source_query_sha256":self.source_query,
            "prior_ledger_sha256":ledger,"decision_ordinal":self.ordinal+1,"catalogue":independently_expected_catalogue});
        let query = digest(SCHEMA, &identity)?;
        reservation.issue(bundle.deadline(), cancel)?;
        control(self.whole, cancel)?;
        let admitted = Instant::now();
        let mut result = CheckedRepairQuery {
            material,
            native: None,
            captured: Some(captured),
            parent: self.parent.clone(),
            source_binding: material.previous_query_binding().clone(),
            query: query.clone(),
            ledger_before,
            ledger: ledger.clone(),
            ordinal: self.ordinal,
            started: self.started,
            whole: self.whole,
            admitted,
            elapsed: nanos(self.started, admitted)?,
            ledger_event_elapsed,
            scope: self.scope,
            selection_interval: self.selection_interval,
            fact,
            catalogue: independently_expected_catalogue.clone(),
            reported_work_counts: captured.reported_work_counts(),
        };
        self.seen_inputs.insert(original.artifact().sha256.clone());
        self.ordinal += 1;
        self.ledger = ledger;
        self.source_query = query;
        control(self.whole, cancel)?;
        let returned = Instant::now();
        if returned >= self.whole {
            return Err(invalid("captured Repair Query admission passed original W"));
        }
        result.admitted = returned;
        result.elapsed = nanos(self.started, returned)?;
        self.last_admitted = returned;
        Ok(result)
    }
}

impl CheckedRepairQuery<'_> {
    pub fn material(&self) -> &ReportedRepairPriorMaterial<'_> {
        self.material
    }
    pub fn native(&self) -> Option<&CheckedNativeReplayPrior<'_>> {
        self.native
    }
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    pub fn captured_witness(&self) -> Option<&super::CheckedCapturedRepairWitness<'_>> {
        self.captured
    }
    pub fn assurance_scope(&self) -> &'static str {
        if self.native.is_some() {
            "actual_parent_chronology_and_independent_native_conditional_prior"
        } else {
            "captured_nn_input_reinference_and_independent_cpu_condition_reexecution"
        }
    }
    pub fn reported_work_scope(&self) -> &'static str {
        if self.native.is_some() {
            "child_report_correlated_to_independent_native_not_same_physical_execution"
        } else {
            "child_report_separate_from_captured_nn_and_independent_cpu_executions"
        }
    }
    pub fn parent(&self) -> &ReplayParentPins {
        &self.parent
    }
    pub fn source_binding(&self) -> &ReplayBindingPins {
        &self.source_binding
    }
    pub fn original_started(&self) -> Instant {
        self.started
    }
    pub fn original_deadline(&self) -> Instant {
        self.whole
    }
    pub fn admitted_at(&self) -> Instant {
        self.admitted
    }
    pub fn elapsed_through_admission_ns(&self) -> u64 {
        self.elapsed
    }
    pub fn clock_scope(&self) -> EpisodeClockScope {
        self.scope
    }
    pub fn selection_interval_ns(&self) -> Option<(u64, u64)> {
        self.selection_interval
    }
    pub fn prior_ordinal(&self) -> usize {
        self.ordinal
    }
    pub fn prior_ledger_sha256(&self) -> &str {
        &self.ledger
    }
    pub fn query_sha256(&self) -> &str {
        &self.query
    }
    pub fn fact_sha256(&self) -> &str {
        &self.fact
    }
    /// Independently corroborated child REPORT counters, not physical child work
    /// authority. Tuple is CPU attempted nodes and physically completed NN inputs.
    pub fn reported_work_counts(&self) -> Option<(u64, u64)> {
        self.reported_work_counts
    }
    /// Audit bytes do not transport the borrowed capability into Python or
    /// another process. They remain a conditional caller report with no reward.
    pub fn audit_json(&self) -> Result<Vec<u8>, ArenaError> {
        serde_json::to_vec(&json!({"schema": SCHEMA,
            "assurance_scope": self.assurance_scope(),
            "parent": self.parent, "source_binding": self.source_binding,
            "query_sha256": self.query, "prior_ledger_before_sha256": self.ledger_before,
            "prior_ledger_sha256": self.ledger, "prior_ordinal": self.ordinal,
            "catalogue": self.catalogue,
            "source_input": self.material.original_input().artifact(),
            "source_output": self.material.checked_report().body_artifact(),
            "conditional_fact_sha256": self.fact, "clock_scope": self.scope,
            "reported_work_counts": self.reported_work_counts,
            "reported_work_counts_scope": self.reported_work_scope(),
            "selection_interval_ns": self.selection_interval,
            "elapsed_through_admission_ns": self.elapsed,
            "ledger_event_elapsed_ns": self.ledger_event_elapsed,
            "whole_budget_ns": nanos(self.started, self.whole)?,
            "native_execution_is_same_physical_child": false,
            "query2_action_semantics_revalidated": false,
            "utility_authority": false, "target_authority": false,
            "training_authority": false, "product_authority": false}))
        .map_err(|_| invalid("Repair Query audit serialization"))
    }
}

#[allow(clippy::too_many_arguments)]
fn check_next_source(
    episode: &RepairQueryEpisode,
    parent: &ReplayParentPins,
    binding: &ReplayBindingPins,
    input: &ArtifactPin,
    started: Instant,
    whole: Instant,
    before: Instant,
) -> Result<(), ArenaError> {
    if episode.ordinal >= episode.maximum_prior
        || parent != &episode.parent
        || binding.query_sha256 != episode.source_query
        || binding.prior_ledger_sha256 != episode.ledger
        || started < episode.started
        || started >= whole
        || whole > episode.whole
        || before < episode.last_admitted
        || before >= whole
        || episode.seen_inputs.contains(&input.sha256)
    {
        return Err(invalid(
            "Repair Query parent/binding/prior/order/duplicate/original clock differs",
        ));
    }
    Ok(())
}

fn check_catalogue(
    raw: &[u8],
    expected: &ArtifactPin,
    parent: &ReplayParentPins,
    original: &ArtifactPin,
) -> Result<(), ArenaError> {
    if raw.is_empty()
        || raw.len() > MAX_BYTES
        || expected != original
        || expected.bytes != raw.len() as u64
        || expected.sha256 != format!("{:x}", Sha256::digest(raw))
    {
        return Err(invalid(
            "Repair Query original independently pinned catalogue bytes",
        ));
    }
    let v: Value =
        serde_json::from_slice(raw).map_err(|_| invalid("Repair Query catalogue JSON"))?;
    let expected_fields: BTreeSet<_> = [
        "schema",
        "parent",
        "recipe_id",
        "actions",
        "limits",
        "scope",
    ]
    .into();
    if v.as_object()
        .map(|m| m.keys().map(String::as_str).collect::<BTreeSet<_>>())
        != Some(expected_fields)
        || v["schema"] != "rz-pals-private-v-action-catalogue/1"
        || v["parent"]
            != serde_json::to_value(parent).map_err(|_| invalid("catalogue parent encode"))?
        || v["scope"] != "checked_private_action_query_only"
        || v["recipe_id"]
            .as_str()
            .is_none_or(|s| s.is_empty() || s.len() > 256)
        || v["actions"]
            .as_array()
            .is_none_or(|a| a.is_empty() || a.len() > 8)
        || v["limits"]
            != json!({"actions":8,"prior_observations":16,"aggregate_immutable_raw_bytes":MAX_BYTES})
    {
        return Err(invalid("Repair Query fixed catalogue parent/limits/shape"));
    }
    Ok(())
}

fn check_independent_native(
    material: &ReportedRepairPriorMaterial<'_>,
    native: &CheckedNativeReplayPrior<'_>,
) -> Result<(String, Option<(u64, u64)>), ArenaError> {
    let observed = native.observation();
    let source = &material.original_input().audit().input_admission;
    let reported = material
        .checked_report()
        .reported_rules()
        .rules()
        .reported();
    let complete = |r| {
        matches!(
            r,
            PriorReadiness::CompleteCpuScopePendingCallerChronology
                | PriorReadiness::RulesTerminalPendingCallerChronology
        )
    };
    if !complete(native.readiness())
        || !complete(reported.recomputed_reported_readiness())
        || observed.input_admission.input_artifact != *material.original_input().artifact()
        || observed.input_admission.parent != source.parent
        || observed.input_admission.binding != source.binding
        || observed.input_admission.registration_artifact != source.registration_artifact
        || observed.input_admission.prepared_action_artifact != source.prepared_action_artifact
        || observed.input_admission.semantic_receipt_artifact != source.semantic_receipt_artifact
        || observed.input_admission.registered_artifacts != source.registered_artifacts
        || observed.input_admission.provider_factory_id != source.provider_factory_id
        || observed.input_admission.legacy_cpu_profile_sha256 != source.legacy_cpu_profile_sha256
        || observed.original_whole_wall_ms != source.whole_wall_ms
        || observed.cleanup_reserve_ms != source.cleanup_reserve_ms
    {
        return Err(invalid(
            "Repair Query independent complete native/parent/model/encoding/input binding differs",
        ));
    }
    let raw: Value = serde_json::from_slice(material.raw_native_bytes())
        .map_err(|_| invalid("Repair Query native report JSON"))?;
    let witness = serde_json::to_value(observed).map_err(|_| invalid("native witness encode"))?;
    let work = comparable_work(&raw)?;
    if work != comparable_work(&witness)? {
        return Err(invalid(
            "Repair Query independent CPU/NN completion accounting differs",
        ));
    }
    if raw["asset_profile_artifact"] != witness["asset_profile_artifact"]
        || raw["repair_anchor_evidence"] != witness["repair_anchor_evidence"]
        || comparable_projection(raw["query_prior"].clone())?
            != comparable_projection(witness["query_prior"].clone())?
    {
        return Err(invalid(
            "Repair Query independently witnessed projection/Repair/model fact differs",
        ));
    }
    let reported_trace = raw["trace_jsonl"]
        .as_str()
        .ok_or_else(|| invalid("reported native trace missing"))?;
    let witnessed_trace = normalized_trace(&observed.trace_jsonl)?;
    if normalized_trace(reported_trace)? != witnessed_trace {
        return Err(invalid(
            "Repair Query actual input/raw/dispatch/consumption trace differs",
        ));
    }
    let fact = digest(
        "rz-pals-independent-repair-conditional-fact/3",
        &json!({
        "parent":source.parent,"binding":source.binding,"asset_profile":observed.asset_profile_artifact,
        "projection":comparable_projection(witness["query_prior"].clone())?,
        "repair_anchor":witness["repair_anchor_evidence"],"native_trace":witnessed_trace,"work":work}),
    )?;
    let counts = if raw["cpu_work_observation_incomplete"] == false {
        raw["cpu_nodes_lower_bound"]
            .as_u64()
            .zip(raw["native_receipt"]["backend_stats"]["completed_nn_inputs"].as_u64())
    } else {
        None
    };
    Ok((fact, counts))
}

fn comparable_work(v: &Value) -> Result<Value, ArenaError> {
    let mut work = serde_json::Map::new();
    for key in [
        "cpu_tasks_requested",
        "cpu_tasks_accounted",
        "cpu_reports_returned",
        "cpu_nodes_lower_bound",
        "cpu_work_observation_incomplete",
    ] {
        work.insert(
            key.into(),
            v.get(key)
                .ok_or_else(|| invalid("Repair work accounting missing"))?
                .clone(),
        );
    }
    let native = v
        .get("native_receipt")
        .filter(|n| n.is_object())
        .ok_or_else(|| invalid("Repair work native receipt missing"))?;
    for key in [
        "physically_completed_role_calls",
        "completed_role_inputs",
        "failed_physical_role_calls",
        "invalid_role_outputs",
        "delivered_role_inputs",
        "search_consumed_role_inputs",
        "canceled_requests",
        "expired_requests",
        "completed_new_game_resets",
    ] {
        work.insert(
            key.into(),
            native
                .get(key)
                .filter(|x| x.as_u64().is_some())
                .ok_or_else(|| invalid("Repair native work strict counter"))?
                .clone(),
        );
    }
    match native.get("backend_stats") {
        Some(Value::Null) => {
            work.insert("backend_stats".into(), Value::Null);
        }
        Some(stats) if stats.is_object() => {
            let mut counters = serde_json::Map::new();
            for key in [
                "admitted_role_requests",
                "public_cache_hits",
                "public_cache_misses",
                "public_nn_runs_attempted",
                "public_nn_runs_completed",
                "public_nn_runs_failed_known",
                "role_nn_runs_attempted",
                "role_nn_runs_completed",
                "role_nn_runs_failed_known",
                "completed_nn_inputs",
                "validated_public_outputs",
                "validated_role_outputs",
                "new_game_resets",
            ] {
                counters.insert(
                    key.into(),
                    stats
                        .get(key)
                        .filter(|x| x.as_u64().is_some())
                        .ok_or_else(|| invalid("Repair backend work strict counter"))?
                        .clone(),
                );
            }
            work.insert("backend_stats".into(), Value::Object(counters));
        }
        _ => {
            return Err(invalid(
                "Repair backend work explicit known/unknown required",
            ));
        }
    }
    Ok(Value::Object(work))
}

fn comparable_projection(mut v: Value) -> Result<Value, ArenaError> {
    let p = v
        .as_object_mut()
        .ok_or_else(|| invalid("Repair projection missing"))?;
    for key in [
        "readiness",
        "work_returned_elapsed_ms",
        "work_returned_within_execution_window",
    ] {
        p.remove(key);
    }
    let stages = p
        .get_mut("stages")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| invalid("Repair projection stages"))?;
    for stage in stages {
        stage
            .as_object_mut()
            .ok_or_else(|| invalid("Repair stage shape"))?
            .remove("report_elapsed_ms");
    }
    Ok(v)
}

// Each trace retains its own one-to-one causal IDs. Only after checking them do
// we replace them with local ordinals for independent-run content comparison.
pub(super) fn normalized_trace(raw: &str) -> Result<Value, ArenaError> {
    if raw.len() > MAX_BYTES {
        return Err(invalid("Repair trace byte bound"));
    }
    let mut requests: BTreeMap<String, (usize, u8, Option<String>)> = BTreeMap::new();
    let mut physical = BTreeSet::new();
    let mut result = Vec::new();
    for line in raw.lines() {
        if result.len() >= MAX_TRACE_ROWS || line.is_empty() {
            return Err(invalid("Repair trace row bound"));
        }
        let row: Value = serde_json::from_str(line).map_err(|_| invalid("Repair trace JSON"))?;
        let event = row["event"]
            .as_str()
            .ok_or_else(|| invalid("Repair trace event"))?;
        let request = if row.get("binding").is_some() {
            &row["binding"]["request"]
        } else {
            &row["request"]
        };
        let id = request_id(request)?;
        if event == "prepared" {
            let ordinal = requests.len();
            if !row["logical"].is_null() && row["logical"].as_str().is_none_or(str::is_empty) {
                return Err(invalid("Repair trace reported logical context shape"));
            }
            if requests.insert(id, (ordinal, 0, None)).is_some()
                || row["input"].is_null()
                || !matches!(
                    row["kind"].as_str(),
                    Some("propose" | "reply" | "repair" | "divergence")
                )
            {
                return Err(invalid("Repair trace duplicate/invalid prepared"));
            }
            result.push(
                json!({"event":event,"ordinal":ordinal,"kind":row["kind"],"input":row["input"]}),
            );
            continue;
        }
        let (ordinal, stage, binding) = requests
            .get_mut(&id)
            .ok_or_else(|| invalid("Repair trace orphan request"))?;
        let fact = match event {
            "dispatched" => {
                if *stage != 0
                    || row["binding"]["execution_namespace"]
                        != "rz_runtime_physical_execution_not_search_store_execution"
                    || row["fact"] != "actual_lease"
                {
                    return Err(invalid("Repair trace dispatch order/namespace"));
                }
                let execution = request_id(&row["binding"]["native_execution"])?;
                if !physical.insert(execution.clone()) {
                    return Err(invalid("Repair trace duplicate physical execution"));
                }
                *binding = Some(execution);
                *stage = 1;
                Value::Null
            }
            "physical_ready" => {
                if *stage != 1
                    || Some(request_id(&row["binding"]["native_execution"])?) != *binding
                    || row["fact"]["kind"] != "evaluation"
                    || row["fact"]["raw_projection"]["complete"] != true
                    || row["fact"]["evaluation_scalar_encoding"]
                        != "ordered_ieee754_f32_bits_as_u32_lower_hex"
                {
                    return Err(invalid(
                        "Repair trace late/missing/error physical completion",
                    ));
                }
                *stage = 2;
                row["fact"].clone()
            }
            "delivered" => {
                if *stage != 2 || row["fact"] != "actual_delivery" {
                    return Err(invalid("Repair trace delivery order"));
                }
                *stage = 3;
                Value::Null
            }
            "accepted_context" => {
                if *stage != 4 || row["context"].as_str().is_none_or(str::is_empty) {
                    return Err(invalid("Repair trace context order"));
                }
                *stage = 5;
                // Debug context is not interpreted as a Rules/store proof.
                json!({"reported_context_present":true})
            }
            "accepted" => {
                if *stage != 3 || row["fact"] != "actual_consumption_callback" {
                    return Err(invalid("Repair trace consumption order"));
                }
                *stage = 4;
                Value::Null
            }
            _ => return Err(invalid("Repair trace unsupported/failed/canceled event")),
        };
        result.push(json!({"event":event,"ordinal":ordinal,"fact":fact}));
    }
    if requests.is_empty() || requests.values().any(|(_, stage, _)| *stage < 4) {
        return Err(invalid("Repair trace unconsumed/missing native completion"));
    }
    Ok(Value::Array(result))
}

fn request_id(v: &Value) -> Result<String, ArenaError> {
    if v.as_object().is_none_or(|m| m.len() != 2)
        || v["epoch"].as_u64().is_none()
        || v["sequence"].as_u64().is_none()
    {
        return Err(invalid("Repair trace strict physical/request ID"));
    }
    Ok(format!("{}:{}", v["epoch"], v["sequence"]))
}
fn check_native_interval(
    interval: Option<(Instant, Instant)>,
    started: Instant,
    whole: Instant,
    now: Instant,
) -> Result<(), ArenaError> {
    match interval {
        Some((begin, end)) if begin >= started && begin <= end && end <= now && end < whole => {
            Ok(())
        }
        _ => Err(invalid(
            "Repair Query retained native witness outside original episode",
        )),
    }
}
fn nanos(start: Instant, at: Instant) -> Result<u64, ArenaError> {
    super::super::chronology::elapsed(start, at)
}
fn check_sha(value: &str) -> Result<(), ArenaError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Err(invalid("Repair Query lowercase SHA256"))
    } else {
        Ok(())
    }
}
pub(super) fn digest(domain: &str, value: &Value) -> Result<String, ArenaError> {
    let bytes = serde_json::to_vec(value).map_err(|_| invalid("Repair Query canonical encode"))?;
    if bytes.len() > MAX_BYTES {
        return Err(invalid("Repair Query byte bound"));
    }
    let mut hash = Sha256::new();
    hash.update(domain.as_bytes());
    hash.update([0]);
    hash.update(bytes);
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(epoch: u64, execution: u64) -> String {
        let request = json!({"epoch":epoch,"sequence":1});
        let binding = json!({"request":request,"native_execution":{"epoch":execution,"sequence":1},
            "execution_namespace":"rz_runtime_physical_execution_not_search_store_execution"});
        [json!({"event":"prepared","request":request,"kind":"reply","input":{"tokens":[1,2]},"logical":null}),
            json!({"event":"dispatched","binding":binding,"fact":"actual_lease"}),
            json!({"event":"physical_ready","binding":binding,"fact":{"kind":"evaluation",
                "raw_projection":{"text":"raw-bits","complete":true},
                "evaluation_scalar_encoding":"ordered_ieee754_f32_bits_as_u32_lower_hex","full_native_diagnostic_archive":false}}),
            json!({"event":"delivered","request":request,"fact":"actual_delivery"}),
            json!({"event":"accepted","request":request,"fact":"actual_consumption_callback"}),
            json!({"event":"accepted_context","request":request,"context":"local-record"})]
            .into_iter().map(|v|v.to_string()).collect::<Vec<_>>().join("\n")
    }
    #[test]
    fn independent_trace_compares_content_not_physical_ids() {
        assert_eq!(
            normalized_trace(&trace(1, 2)).unwrap(),
            normalized_trace(&trace(31, 42)).unwrap()
        );
        assert_ne!(
            normalized_trace(&trace(1, 2)).unwrap(),
            normalized_trace(&trace(1, 2).replace("raw-bits", "different-bits")).unwrap()
        );
    }
    #[test]
    fn independent_trace_refuses_duplicate_reordered_late_missing_and_wrong_binding() {
        let base = trace(1, 2);
        let lines = base.lines().collect::<Vec<_>>();
        for bad in [
            format!("{base}\n{}", lines[5]),
            [lines[0], lines[2], lines[1], lines[3], lines[4], lines[5]].join("\n"),
            lines[..4].join("\n"),
            base.replace("\"complete\":true", "\"complete\":false"),
            base.replace("actual_lease", "another_namespace"),
        ] {
            assert!(normalized_trace(&bad).is_err());
        }
        let mut bad = lines.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        bad[2] = bad[2].replace("\"epoch\":2", "\"epoch\":3");
        assert!(normalized_trace(&bad.join("\n")).is_err());
    }
    #[test]
    fn projection_ignores_elapsed_but_not_actual_moves_nodes_or_condition() {
        let a = json!({"readiness":"complete","work_returned_elapsed_ms":1,"work_returned_within_execution_window":true,
            "stages":[{"report_elapsed_ms":1,"pv":[1,2],"report_nodes":17,"task_condition_sha256":[1]}]});
        let mut b = a.clone();
        b["work_returned_elapsed_ms"] = json!(9);
        b["stages"][0]["report_elapsed_ms"] = json!(9);
        assert_eq!(
            comparable_projection(a.clone()).unwrap(),
            comparable_projection(b.clone()).unwrap()
        );
        b["stages"][0]["pv"] = json!([1, 3]);
        assert_ne!(
            comparable_projection(a).unwrap(),
            comparable_projection(b).unwrap()
        );
    }
    #[test]
    fn native_interval_refuses_old_missing_reordered_future_or_late_witness() {
        let s = Instant::now();
        let w = s + Duration::from_secs(5);
        let now = s + Duration::from_secs(2);
        assert!(check_native_interval(Some((s, s + Duration::from_secs(1))), s, w, now).is_ok());
        for bad in [
            None,
            Some((s - Duration::from_secs(1), s)),
            Some((s + Duration::from_secs(1), s)),
            Some((s, w)),
            Some((s, s + Duration::from_secs(3))),
        ] {
            assert!(check_native_interval(bad, s, w, now).is_err());
        }
    }
    #[test]
    fn trace_context_declaration_cannot_be_missing_or_changed_to_object() {
        let base = trace(1, 2);
        assert!(
            normalized_trace(&base.replace("\"context\":\"local-record\"", "\"context\":null"))
                .is_err()
        );
        assert!(
            normalized_trace(&base.replace(
                "\"context\":\"local-record\"",
                "\"context\":{\"complete\":false}"
            ))
            .is_err()
        );
    }
    fn pins() -> (ReplayParentPins, ReplayBindingPins, ArtifactPin) {
        let pin = ArtifactPin {
            bytes: 1,
            sha256: "1".repeat(64),
        };
        (
            ReplayParentPins {
                parent_input_sha256: "2".repeat(64),
                current_view_sha256: "3".repeat(64),
                frozen_admission_sha256: "4".repeat(64),
                encoding_sha256: "5".repeat(64),
            },
            ReplayBindingPins {
                query_sha256: "6".repeat(64),
                catalogue_artifact: pin.clone(),
                before_result_artifact: pin.clone(),
                prior_ledger_sha256: "7".repeat(64),
                semantic_input_sha256: "8".repeat(64),
                semantic_context_sha256: "9".repeat(64),
                semantic_branch_meaning_sha256: "a".repeat(64),
                semantic_before_result_anchor_sha256: "b".repeat(64),
                cpu_request_artifact: pin.clone(),
            },
            pin,
        )
    }
    #[test]
    fn episode_requires_original_parent_query_ledger_order_and_finite_extent() {
        let (parent, binding, pin) = pins();
        let cancel = AtomicBool::new(false);
        let start = Instant::now();
        let whole = start + Duration::from_secs(5);
        let mut e = RepairQueryEpisode::new(
            &parent,
            &binding,
            1,
            start,
            whole,
            EpisodeClockScope::PreparationOnly,
            None,
            &cancel,
        )
        .unwrap();
        let before = start + Duration::from_millis(10);
        assert!(check_next_source(&e, &parent, &binding, &pin, start, whole, before).is_ok());
        let mut other = parent.clone();
        other.encoding_sha256 = "c".repeat(64);
        assert!(check_next_source(&e, &other, &binding, &pin, start, whole, before).is_err());
        let mut other = binding.clone();
        other.prior_ledger_sha256 = "c".repeat(64);
        assert!(check_next_source(&e, &parent, &other, &pin, start, whole, before).is_err());
        other = binding.clone();
        other.query_sha256 = "c".repeat(64);
        assert!(check_next_source(&e, &parent, &other, &pin, start, whole, before).is_err());
        assert!(
            check_next_source(
                &e,
                &parent,
                &binding,
                &pin,
                start - Duration::from_millis(1),
                whole,
                before
            )
            .is_err()
        );
        assert!(
            check_next_source(
                &e,
                &parent,
                &binding,
                &pin,
                start,
                whole + Duration::from_millis(1),
                before
            )
            .is_err()
        );
        e.last_admitted = before + Duration::from_millis(1);
        assert!(check_next_source(&e, &parent, &binding, &pin, start, whole, before).is_err());
        e.last_admitted = start;
        e.seen_inputs.insert(pin.sha256.clone());
        assert!(check_next_source(&e, &parent, &binding, &pin, start, whole, before).is_err());
        e.seen_inputs.clear();
        e.ordinal = 1;
        assert!(check_next_source(&e, &parent, &binding, &pin, start, whole, before).is_err());
    }
    #[test]
    fn before_selection_callback_is_actual_bounded_observation_not_preparation_scope() {
        let (parent, binding, _) = pins();
        let cancel = AtomicBool::new(false);
        let ((s, w), episode) = RepairQueryEpisode::begin_before_selection(
            &parent,
            &binding,
            1,
            Duration::from_secs(5),
            &cancel,
            |s, w| Ok((s, w)),
        )
        .unwrap();
        assert_eq!(s, episode.original_started());
        assert_eq!(w, episode.original_deadline());
        assert_eq!(episode.scope, EpisodeClockScope::BeforeSelectionCallback);
        assert_eq!(episode.selection_interval.unwrap().0, 0);
        let canceled = AtomicBool::new(true);
        let calls = std::cell::Cell::new(0);
        assert!(
            RepairQueryEpisode::begin_before_selection(
                &parent,
                &binding,
                1,
                Duration::from_secs(5),
                &canceled,
                |_, _| {
                    calls.set(calls.get() + 1);
                    Ok(())
                }
            )
            .is_err()
        );
        assert_eq!(calls.get(), 0);
        assert!(
            RepairQueryEpisode::begin_before_selection(
                &parent,
                &binding,
                17,
                Duration::from_secs(5),
                &cancel,
                |_, _| Ok(())
            )
            .is_err()
        );
        assert!(
            RepairQueryEpisode::begin_before_selection(
                &parent,
                &binding,
                1,
                Duration::ZERO,
                &cancel,
                |_, _| Ok(())
            )
            .is_err()
        );
        assert!(
            RepairQueryEpisode::begin_before_selection(
                &parent,
                &binding,
                1,
                Duration::from_secs(301),
                &cancel,
                |_, _| Ok(())
            )
            .is_err()
        );
    }
    #[test]
    fn next_query_requires_actual_catalogue_body_and_original_pin() {
        let (parent, _, _) = pins();
        let raw=serde_json::to_vec(&json!({"schema":"rz-pals-private-v-action-catalogue/1","parent":parent,
            "recipe_id":"bounded","actions":[{"task":"defend_response"}],
            "limits":{"actions":8,"prior_observations":16,"aggregate_immutable_raw_bytes":MAX_BYTES},
            "scope":"checked_private_action_query_only"})).unwrap();
        let pin = ArtifactPin {
            bytes: raw.len() as u64,
            sha256: format!("{:x}", Sha256::digest(&raw)),
        };
        assert!(check_catalogue(&raw, &pin, &parent, &pin).is_ok());
        let mut wrong = pin.clone();
        wrong.sha256 = "0".repeat(64);
        assert!(check_catalogue(&raw, &wrong, &parent, &pin).is_err());
        let mut p = parent;
        p.current_view_sha256 = "c".repeat(64);
        assert!(check_catalogue(&raw, &pin, &p, &pin).is_err());
        assert!(check_catalogue(&raw[..raw.len() - 1], &pin, &p, &pin).is_err());
    }
}
