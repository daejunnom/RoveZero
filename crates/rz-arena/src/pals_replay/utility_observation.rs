//! Bounded, actual-clock coverage action costs and conditional cost dominance.
//! The selector is an explicit deterministic single-offer policy, not neural V.
//! Child work remains a registered report correlated to an independent live
//! native owner. Their IDs are never equated and both executions are charged.
//! Audit JSON cannot recreate any of the borrowed capabilities or targets.

use super::{CheckedRepairQuery, EpisodeClockScope, RepairQueryEpisode};
use crate::ArenaError;
use rz_uci::pals_cpu_task::strategic_action::native_replay::query_prior::{
    ConditionalEndpointFact, PriorPhase, PriorReadiness,
};
use rz_uci::pals_cpu_task::strategic_action::replay_inputs::{ReplayBindingPins, ReplayParentPins};
use rz_uci::pals_cpu_task::strategic_action::{ArtifactPin, StrategicAction, StrategicTaskKind};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const COVERAGE_POLICY: &str = "first_registered_defend_response_coverage/1";
pub const REGISTERED_COVERAGE_POLICY: &str = "registered_defend_response_coverage/2";
pub const WHOLE_COST_SCHEMA: &str = "rz-pals-coverage-whole-action-cost/2";
pub const UTILITY_SCHEMA: &str = "rz-pals-conditional-whole-cost-dominance/2";
pub const CAPTURED_WHOLE_COST_SCHEMA: &str = "rz-pals-coverage-whole-action-cost/3";
pub const CAPTURED_UTILITY_SCHEMA: &str = "rz-pals-conditional-whole-cost-dominance/3";
const COUNTER_SCOPE: &str =
    "registered_child_report_correlated_to_independent_native_plus_separate_live_native_execution";
const CAPTURED_COUNTER_SCOPE: &str =
    "child_work_report_plus_actual_captured_input_nn_replay_and_independent_cpu_work";

#[derive(Clone, Copy, Eq, PartialEq)]
enum WitnessCostScope {
    FullReplay,
    CapturedInputsAndCpu,
}
impl WitnessCostScope {
    fn counter_scope(self) -> &'static str {
        match self {
            Self::FullReplay => COUNTER_SCOPE,
            Self::CapturedInputsAndCpu => CAPTURED_COUNTER_SCOPE,
        }
    }
    fn whole_schema(self) -> &'static str {
        match self {
            Self::FullReplay => WHOLE_COST_SCHEMA,
            Self::CapturedInputsAndCpu => CAPTURED_WHOLE_COST_SCHEMA,
        }
    }
    fn utility_schema(self) -> &'static str {
        match self {
            Self::FullReplay => UTILITY_SCHEMA,
            Self::CapturedInputsAndCpu => CAPTURED_UTILITY_SCHEMA,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageChoice {
    DefendResponse,
    Defer,
}

/// A real pure selector invocation. No public constructor/Clone/deserializer.
/// The one-shot gate is spent even if a later selection/result check fails.
pub struct ObservedCoverageSelection {
    parent: ReplayParentPins,
    binding: ReplayBindingPins,
    started: Instant,
    returned: Instant,
    whole: Instant,
    action_index: usize,
    action: StrategicAction,
    policy: &'static str,
    prepared_action: ArtifactPin,
    covered_callback: bool,
    spent: AtomicBool,
}
impl ObservedCoverageSelection {
    pub fn choice(&self) -> CoverageChoice {
        CoverageChoice::DefendResponse
    }
    pub fn policy(&self) -> &'static str {
        self.policy
    }
    pub fn action_index(&self) -> usize {
        self.action_index
    }
}

/// Use inside RepairQueryEpisode::begin_before_selection. The supplied S/W
/// must be those exact callback arguments; an older window is never restarted.
/// There is one already registered DefendResponse offer. This policy makes no
/// learned ranking claim, launches no CPU search/NN, and has no opaque callback.
#[allow(clippy::too_many_arguments)]
pub fn select_registered_repair_coverage(
    parent: &ReplayParentPins,
    binding: &ReplayBindingPins,
    catalogue_raw: &[u8],
    prepared_action_raw: &[u8],
    expected_prepared_action: &ArtifactPin,
    episode_started: Instant,
    episode_whole: Instant,
    cancel: &AtomicBool,
) -> Result<ObservedCoverageSelection, ArenaError> {
    select_coverage(
        parent,
        binding,
        catalogue_raw,
        prepared_action_raw,
        expected_prepared_action,
        None,
        episode_started,
        episode_whole,
        cancel,
    )
}

#[allow(clippy::too_many_arguments)]
fn select_coverage(
    parent: &ReplayParentPins,
    binding: &ReplayBindingPins,
    catalogue_raw: &[u8],
    prepared_action_raw: &[u8],
    expected_prepared_action: &ArtifactPin,
    registered_action_index: Option<usize>,
    episode_started: Instant,
    episode_whole: Instant,
    cancel: &AtomicBool,
) -> Result<ObservedCoverageSelection, ArenaError> {
    control(episode_whole, cancel)?;
    let started = Instant::now();
    if started < episode_started || episode_started >= episode_whole {
        return Err(invalid("coverage selector original episode clock"));
    }
    let (action_index, action) = registered_repair_offer(
        catalogue_raw,
        binding,
        prepared_action_raw,
        expected_prepared_action,
        registered_action_index,
    )?;
    let parent = parent.clone();
    let binding = binding.clone();
    let returned = Instant::now();
    control(episode_whole, cancel)?;
    Ok(ObservedCoverageSelection {
        parent,
        binding,
        started,
        returned,
        whole: episode_whole,
        action_index,
        action,
        policy: if registered_action_index.is_some() {
            REGISTERED_COVERAGE_POLICY
        } else {
            COVERAGE_POLICY
        },
        prepared_action: expected_prepared_action.clone(),
        covered_callback: false,
        spent: AtomicBool::new(false),
    })
}

/// Complete selector accounting through a callback fixed in source. It runs
/// only this pure policy; arbitrary caller work cannot be labelled zero NN/CPU.
/// Lower-level selection observations do not admit whole-action costs.
#[allow(clippy::too_many_arguments)]
pub fn begin_registered_repair_coverage_episode(
    parent: &ReplayParentPins,
    binding: &ReplayBindingPins,
    catalogue_raw: &[u8],
    prepared_action_raw: &[u8],
    expected_prepared_action: &ArtifactPin,
    maximum_prior: usize,
    whole_budget: Duration,
    cancel: &AtomicBool,
) -> Result<(ObservedCoverageSelection, RepairQueryEpisode), ArenaError> {
    let (mut selection, episode) = RepairQueryEpisode::begin_before_selection(
        parent,
        binding,
        maximum_prior,
        whole_budget,
        cancel,
        |started, whole| {
            select_registered_repair_coverage(
                parent,
                binding,
                catalogue_raw,
                prepared_action_raw,
                expected_prepared_action,
                started,
                whole,
                cancel,
            )
        },
    )?;
    selection.covered_callback = true;
    Ok((selection, episode))
}

/// A bounded index is fixed before S, within the independently pinned catalogue.
/// It is an explicit untrained coverage policy, not a learned V or callback.
#[allow(clippy::too_many_arguments)]
pub fn begin_registered_defend_coverage_episode(
    parent: &ReplayParentPins,
    binding: &ReplayBindingPins,
    catalogue_raw: &[u8],
    prepared_action_raw: &[u8],
    expected_prepared_action: &ArtifactPin,
    registered_action_index: usize,
    maximum_prior: usize,
    whole_budget: Duration,
    cancel: &AtomicBool,
) -> Result<(ObservedCoverageSelection, RepairQueryEpisode), ArenaError> {
    let (mut selection, episode) = RepairQueryEpisode::begin_before_selection(
        parent,
        binding,
        maximum_prior,
        whole_budget,
        cancel,
        |started, whole| {
            select_coverage(
                parent,
                binding,
                catalogue_raw,
                prepared_action_raw,
                expected_prepared_action,
                Some(registered_action_index),
                started,
                whole,
                cancel,
            )
        },
    )?;
    selection.covered_callback = true;
    Ok((selection, episode))
}

#[cfg(test)]
fn first_registered_repair_offer(
    raw: &[u8],
    binding: &ReplayBindingPins,
    prepared_raw: &[u8],
    prepared_pin: &ArtifactPin,
) -> Result<usize, ArenaError> {
    registered_repair_offer(raw, binding, prepared_raw, prepared_pin, None).map(|(index, _)| index)
}

fn registered_repair_offer(
    raw: &[u8],
    binding: &ReplayBindingPins,
    prepared_raw: &[u8],
    prepared_pin: &ArtifactPin,
    registered_action_index: Option<usize>,
) -> Result<(usize, StrategicAction), ArenaError> {
    if raw.is_empty()
        || raw.len() > 4 * 1024 * 1024
        || u64::try_from(raw.len()).ok() != Some(binding.catalogue_artifact.bytes)
        || format!("{:x}", Sha256::digest(raw)) != binding.catalogue_artifact.sha256
    {
        return Err(invalid(
            "coverage selector original registered catalogue bytes differ",
        ));
    }
    let catalogue: serde_json::Value = serde_json::from_slice(raw)
        .map_err(|_| invalid("coverage selector original catalogue JSON"))?;
    let actions = catalogue["actions"]
        .as_array()
        .filter(|a| !a.is_empty() && a.len() <= 8)
        .ok_or_else(|| invalid("coverage selector bounded actual catalogue offers"))?;
    let selected = match registered_action_index {
        Some(index) if index < actions.len() && index < 8 => index,
        Some(_) => {
            return Err(invalid(
                "coverage registered index outside actual bounded catalogue",
            ));
        }
        None => actions
            .iter()
            .position(|a| a["task"] == "defend_response")
            .ok_or_else(|| invalid("coverage selector has no registered DefendResponse offer"))?,
    };
    if prepared_raw.is_empty()
        || prepared_raw.len() > 4 * 1024 * 1024
        || prepared_pin.bytes != prepared_raw.len() as u64
        || prepared_pin.sha256 != format!("{:x}", Sha256::digest(prepared_raw))
    {
        return Err(invalid(
            "coverage selector original independently pinned prepared action",
        ));
    }
    let prepared: serde_json::Value = serde_json::from_slice(prepared_raw)
        .map_err(|_| invalid("coverage selector original prepared JSON"))?;
    let offer: StrategicAction = serde_json::from_value(actions[selected].clone())
        .map_err(|_| invalid("coverage selector actual closed offer"))?;
    let prepared_offer: StrategicAction = serde_json::from_value(prepared["action"].clone())
        .map_err(|_| invalid("coverage selector actual closed prepared offer"))?;
    if prepared["schema"] != rz_uci::pals_cpu_task::strategic_action::SCHEMA
        || offer != prepared_offer
        || offer.task != StrategicTaskKind::DefendResponse
        || usize::from(offer.slot) != selected
        || prepared["query_sha256"] != binding.query_sha256
        || prepared["prior_ledger_sha256"] != binding.prior_ledger_sha256
        || prepared["catalogue_artifact"] != json!(binding.catalogue_artifact)
        || prepared["before_result_artifact"] != json!(binding.before_result_artifact)
        || prepared["cpu_request_artifact"] != json!(binding.cpu_request_artifact)
    {
        return Err(invalid(
            "coverage selector selected/prepared action or question/prior differs",
        ));
    }
    Ok((selected, offer))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct WholeActionCosts {
    pub whole_elapsed_ns: u64,
    pub cpu_nodes: u64,
    /// Actual backend graph rows, including public encodes and startup probes.
    /// Native role callbacks/cache hits are not used as a proxy for NN rows.
    pub physical_nn_inputs: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ObservedInterval {
    pub start_ns: u64,
    pub end_ns: u64,
}

/// The source Query and original selector remain alive. This is not a portable
/// action receipt or a training/loss/online task-selection capability.
pub struct CheckedRepairActionCost<'query, 'owner> {
    query: &'query CheckedRepairQuery<'owner>,
    selector: &'query ObservedCoverageSelection,
    next: ObservedInterval,
    native_witness: ObservedInterval,
    whole_elapsed_ns: u64,
    source_work: Option<(u64, u64)>,
    witness_work: Option<(u64, u64)>,
    witness_scope: WitnessCostScope,
    costs: Option<WholeActionCosts>,
    utility_spent: AtomicBool,
}
impl CheckedRepairActionCost<'_, '_> {
    pub fn costs(&self) -> Option<WholeActionCosts> {
        self.costs
    }
    pub fn next_choice(&self) -> CoverageChoice {
        CoverageChoice::Defer
    }
    pub fn query(&self) -> &CheckedRepairQuery<'_> {
        self.query
    }
    pub fn audit_json(&self) -> Result<Vec<u8>, ArenaError> {
        let timing = self.query.material().checked_report().caller_timing();
        let source_start = self
            .query
            .material()
            .checked_report()
            .caller_timing()
            .capture()
            .bundle()
            .original_started();
        let preparation_offset = nanos(self.query.original_started(), source_start)?;
        let (initial_start, initial_end) = self
            .query
            .selection_interval_ns()
            .ok_or_else(|| invalid("coverage initial selection interval missing"))?;
        let initial = ObservedInterval {
            start_ns: initial_start,
            end_ns: initial_end,
        };
        let through_check = preparation_offset
            .checked_add(timing.elapsed_through_check_ns())
            .ok_or_else(|| invalid("coverage check clock extent"))?;
        let mut audit = json!({"schema":self.witness_scope.whole_schema(),"policy":self.selector.policy(),
            "query_sha256":self.query.query_sha256(),"parent":self.query.parent(),
            "source_binding":self.query.source_binding(),
            "source_input":self.query.material().original_input().artifact(),
            "source_output":self.query.material().checked_report().body_artifact(),
            "conditional_fact_sha256":self.query.fact_sha256(),
            "selected_original_action_index":self.selector.action_index,
            "selected_action":self.selector.action,
            "whole_elapsed_ns":self.whole_elapsed_ns,"whole_cost":self.costs,
            "initial_selection":initial,"next_selection":self.next,
            "independent_native_work_interval":self.native_witness,
            "source_work":self.source_work,"independent_native_work":self.witness_work,
            "work_counter_scope":self.witness_scope.counter_scope(),
            "source_native_ids_equated_to_witness":false,
            "whole_causal_elapsed_observed":true,
            "preparation_transport_check_intervals_ns":{
                "preparation_and_preflight":timing.preparation_and_preflight_ns(),
                "supervisor_setup":timing.supervisor_setup_ns(),
                "launch_work_and_wait":timing.launch_to_exit_observation_ns(),
                "drain":timing.drain_and_supervisor_return_ns(),
                "postflight":timing.caller_postflight_ns(),
                "after_capture_before_check":timing.after_capture_before_check_ns(),
                "result_check":timing.result_check_ns(),
                "check_to_next_query_admission":self.query.elapsed_through_admission_ns().checked_sub(through_check)},
            "logical_phase_elapsed_ns":{
                "initial_v_or_coverage_selection":initial.end_ns-initial.start_ns,
                "preparation":null,"nn":null,"cpu":null,"transfer":null,"check":null,
                "next_selection":self.next.end_ns-self.next.start_ns},
            "logical_phase_scope":"combined_parent_intervals_observed;unseparated_subphases_unknown;no_additive_double_count",
            "neural_v_executed":false,"neural_v_skipped_by_policy":true,
            "selector_cpu_search_executed":false,"selector_nn_inputs":0,
            "next_selection_cpu_search_executed":false,"next_selection_nn_inputs":0,
            "next_choice":"defer","whole_work_counts_known":self.costs.is_some(),
            "utility_authority":false,"target_authority":false,"training_authority":false,
            "learned_utility_claim":false,"product_verifier_enabled":false,"optimizer_steps":0});
        if self.witness_scope == WitnessCostScope::CapturedInputsAndCpu {
            let map = audit
                .as_object_mut()
                .ok_or_else(|| invalid("captured cost audit object"))?;
            let interval = map
                .remove("independent_native_work_interval")
                .ok_or_else(|| invalid("captured cost witness interval"))?;
            map.insert("independent_witness_work_interval".into(), interval);
            map.insert(
                "source_child_work_provenance".into(),
                json!("registered_source_owned_report;physical_child_work_not_attested"),
            );
            map.insert("actual_independent_work_scope".into(),json!("fresh_independent_cpu_attempts_and_actual_recorded_input_nn_ready_stats_including_initialization_and_final_join"));
            map.insert("source_child_physical_work_attested".into(), json!(false));
            map.insert(
                "source_cpu_scalar_score_correlated".into(),
                serde_json::Value::Null,
            );
        }
        serde_json::to_vec(&audit).map_err(|_| invalid("coverage whole-cost audit serialization"))
    }
}

/// Observe an actual deterministic next choice after Query admission, inside
/// the original episode. No caller timestamps/counters/selector callback can
/// upgrade missing work. Unknown counters are retained as None, never zero.
pub fn observe_repair_coverage_cost<'query, 'owner>(
    query: &'query CheckedRepairQuery<'owner>,
    selector: &'query ObservedCoverageSelection,
    cancel: &AtomicBool,
) -> Result<CheckedRepairActionCost<'query, 'owner>, ArenaError> {
    control(query.original_deadline(), cancel)?;
    let origin = query.original_started();
    let Some((selection_start, selection_end)) = query.selection_interval_ns() else {
        return Err(invalid(
            "whole action excludes an unobserved initial selection",
        ));
    };
    let select_start = nanos(origin, selector.started)?;
    let select_end = nanos(origin, selector.returned)?;
    let capture = query.material().checked_report().caller_timing().capture();
    let (witness_scope, native_started, native_returned) = witness_interval(query)
        .ok_or_else(|| invalid("whole action requires original live native witness clock"))?;
    if !selector.covered_callback
        || query.clock_scope() != EpisodeClockScope::BeforeSelectionCallback
        || selector.whole != query.original_deadline()
        || selector.parent != *query.parent()
        || selector.binding != *query.source_binding()
        || selector.prepared_action
            != query
                .material()
                .original_input()
                .audit()
                .input_admission
                .prepared_action_artifact
        || selection_start > select_start
        || select_start > select_end
        || select_end > selection_end
        || selector.returned > capture.caller_started
        || native_started < origin
        || native_returned < native_started
        || native_returned > query.admitted_at()
        || capture.bundle().deadline() > query.original_deadline()
        || query.admitted_at() >= query.original_deadline()
        || !witness_completed(query)
    {
        return Err(invalid(
            "coverage actual selector/action/witness/episode chronology differs",
        ));
    }
    let source_work = query.reported_work_counts();
    let witness_work = witness_work(query);
    // Native-only counts never become whole-action work. The independently
    // correlated child and the actual live re-execution are charged separately.
    let whole_work = match (source_work, witness_work) {
        (Some(source), Some(witness))
            if witness_scope == WitnessCostScope::CapturedInputsAndCpu || source == witness =>
        {
            Some((
                source
                    .0
                    .checked_add(witness.0)
                    .ok_or_else(|| invalid("whole CPU work overflow"))?,
                source
                    .1
                    .checked_add(witness.1)
                    .ok_or_else(|| invalid("whole NN work overflow"))?,
            ))
        }
        _ => None,
    };
    selector
        .spent
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| invalid("coverage selector already consumed by a next choice"))?;
    let next_started = Instant::now();
    control(query.original_deadline(), cancel)?;
    // The registered single-offer policy stops after its complete conditional
    // observation. Defer has no NN/CPU search, and is actually selected here.
    let _choice = CoverageChoice::Defer;
    let next_returned = Instant::now();
    control(query.original_deadline(), cancel)?;
    let next = ObservedInterval {
        start_ns: nanos(origin, next_started)?,
        end_ns: nanos(origin, next_returned)?,
    };
    let costs = whole_work.map(|(cpu_nodes, physical_nn_inputs)| WholeActionCosts {
        whole_elapsed_ns: next.end_ns,
        cpu_nodes,
        physical_nn_inputs,
    });
    Ok(CheckedRepairActionCost {
        query,
        selector,
        next,
        native_witness: ObservedInterval {
            start_ns: nanos(origin, native_started)?,
            end_ns: nanos(origin, native_returned)?,
        },
        whole_elapsed_ns: next.end_ns,
        source_work,
        witness_work,
        witness_scope,
        costs,
        utility_spent: AtomicBool::new(false),
    })
}

fn native_work(query: &CheckedRepairQuery<'_>) -> Option<(u64, u64)> {
    let observation = query.native()?.observation();
    let receipt = observation.native_receipt.as_ref()?;
    let stats = receipt.backend_stats.as_ref()?;
    let mut nodes = 0u64;
    if observation.cpu_work_observation_incomplete != Some(false)
        || !receipt.physical_shutdown_confirmed
        || !receipt.native_buffers_released
        || receipt.physical_runs_in_flight != 0
        || receipt.quarantined
        || receipt.failed_physical_role_calls != 0
        || receipt.invalid_role_outputs != 0
        || receipt.canceled_requests != 0
        || receipt.expired_requests != 0
        || receipt.observer_failures != 0
        || receipt.last_failure.is_some()
        || receipt.last_observer_failure.is_some()
        || stats.public_nn_runs_attempted != stats.public_nn_runs_completed
        || stats.role_nn_runs_attempted != stats.role_nn_runs_completed
        || stats.public_nn_runs_failed_known != 0
        || stats.role_nn_runs_failed_known != 0
        || stats
            .public_nn_runs_completed
            .checked_add(stats.role_nn_runs_completed)?
            != stats.completed_nn_inputs
    {
        return None;
    }
    for stage in &observation.stages {
        let actual = stage.attempt_nodes?;
        if !stage.exact_completed
            || !stage.attempt_present
            || !stage.report_present
            || stage.report_nodes != Some(actual)
        {
            return None;
        }
        nodes = nodes.checked_add(actual)?;
    }
    if observation.cpu_nodes_lower_bound != Some(nodes) {
        return None;
    }
    Some((nodes, stats.completed_nn_inputs))
}

fn witness_interval(
    query: &CheckedRepairQuery<'_>,
) -> Option<(WitnessCostScope, Instant, Instant)> {
    if let Some(native) = query.native() {
        let (started, finished) = native.projection().native_work_interval()?;
        return Some((WitnessCostScope::FullReplay, started, finished));
    }
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    if let Some(captured) = query.captured_witness() {
        return Some((
            WitnessCostScope::CapturedInputsAndCpu,
            captured.work_started_at(),
            captured.work_finished_at(),
        ));
    }
    None
}
fn witness_completed(query: &CheckedRepairQuery<'_>) -> bool {
    if let Some(native) = query.native() {
        return native.readiness() == PriorReadiness::CompleteCpuScopePendingCallerChronology;
    }
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    if let Some(captured) = query.captured_witness() {
        let receipt = captured.final_native_receipt();
        return receipt.physical_shutdown_confirmed
            && receipt.native_buffers_released
            && receipt.physical_runs_in_flight == 0
            && !receipt.quarantined;
    }
    false
}
fn witness_work(query: &CheckedRepairQuery<'_>) -> Option<(u64, u64)> {
    if query.native().is_some() {
        return native_work(query);
    }
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    if let Some(captured) = query.captured_witness() {
        return Some(captured.actual_work_counts());
    }
    None
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CostPreference {
    Left,
    Right,
}

pub struct CheckedConditionalCostPair<'pair, 'query, 'owner> {
    left: &'pair CheckedRepairActionCost<'query, 'owner>,
    right: &'pair CheckedRepairActionCost<'query, 'owner>,
    preference: Option<CostPreference>,
    reason: &'static str,
}
impl CheckedConditionalCostPair<'_, '_, '_> {
    pub fn preference(&self) -> Option<CostPreference> {
        self.preference
    }
    pub fn audit_json(&self) -> Result<Vec<u8>, ArenaError> {
        let mut audit = json!({"schema":self.left.witness_scope.utility_schema(),
            "policy":"same_conditional_raw_endpoint_whole_cost_dominance/2",
            "scope":"conditional_cost_observation_only;not_paper_reward_or_learned_utility",
            "parent":self.left.query.parent(),"source_query_sha256":self.left.query.source_binding().query_sha256,
            "prior_ledger_before_sha256":self.left.query.source_binding().prior_ledger_sha256,
            "source_inputs":[self.left.query.material().original_input().artifact(),self.right.query.material().original_input().artifact()],
            "selected_original_action_indices":[self.left.selector.action_index,self.right.selector.action_index],
            "source_actions":[self.left.selector.action,self.right.selector.action],
            "whole_costs":[self.left.costs,self.right.costs],"work_counter_scope":self.left.witness_scope.counter_scope(),
            "preference":self.preference,"mask":self.preference.is_some(),"reason":self.reason,
            "actual_utility_groups":u8::from(self.preference.is_some()),
            "utility_observation_admitted":self.preference.is_some(),"target_authority":false,
            "training_authority":false,"neural_v_executed":false,"learned_utility_claim":false,
            "strategic_refutation_admitted":false,"rules_proof_admitted":false,
            "depth_or_cpu_agreement_reward":false,"no_counterexample_reward":false,
            "paper_reward_claim":false,"product_verifier_enabled":false,"optimizer_steps":0});
        if self.left.witness_scope == WitnessCostScope::CapturedInputsAndCpu {
            let map = audit
                .as_object_mut()
                .ok_or_else(|| invalid("captured utility audit object"))?;
            map.insert(
                "source_child_work_provenance".into(),
                json!("registered_source_owned_report;physical_child_work_not_attested"),
            );
            map.insert("source_child_physical_work_attested".into(), json!(false));
            map.insert(
                "source_cpu_scalar_score_correlated".into(),
                serde_json::Value::Null,
            );
            map.insert("conditional_fact_scope".into(),json!("actual_fresh_independent_cpu_endpoint_fact_only;not_correlated_child_scalar_score"));
        }
        serde_json::to_vec(&audit).map_err(|_| invalid("conditional cost pair audit serialization"))
    }
}

/// Two distinct actual source captures AND independent native owners are
/// required. A strict three-axis Pareto comparison is made only for the same
/// pre-result question/prior and the same completed conditional endpoint fact.
pub fn admit_conditional_cost_dominance<'pair, 'query, 'owner>(
    left: &'pair CheckedRepairActionCost<'query, 'owner>,
    right: &'pair CheckedRepairActionCost<'query, 'owner>,
) -> Result<CheckedConditionalCostPair<'pair, 'query, 'owner>, ArenaError> {
    let l = left.query;
    let r = right.query;
    if std::ptr::eq(left, right)
        || std::ptr::eq(l.material(), r.material())
        || same_witness_owner(l, r)
        || std::ptr::eq(left.selector, right.selector)
    {
        return Err(invalid(
            "same capture/native/selector cannot prove independent actions",
        ));
    }
    let same_question = l.parent() == r.parent()
        && l.source_binding().query_sha256 == r.source_binding().query_sha256
        && l.source_binding().catalogue_artifact == r.source_binding().catalogue_artifact
        && l.source_binding().before_result_artifact == r.source_binding().before_result_artifact
        && l.source_binding().prior_ledger_sha256 == r.source_binding().prior_ledger_sha256;
    let (preference, reason) = if left.witness_scope != right.witness_scope {
        (None, "different_independent_execution_cost_scope")
    } else if !same_question {
        (None, "different_pre_result_question_or_prior")
    } else if left.selector.action_index == right.selector.action_index
        || same_effective_action(&left.selector.action, &right.selector.action)
    {
        (None, "same_action_repetition_is_variance_observation_only")
    } else if !same_conditional_witness(l, r) {
        (
            None,
            "different_or_unobserved_conditional_counter_lower_fact",
        )
    } else {
        match (left.costs, right.costs) {
            (Some(a), Some(b)) => match pareto(a, b) {
                Some(p) => (Some(p), "same_conditional_fact_strict_whole_cost_dominance"),
                None => (None, "equal_or_crossing_whole_cost_axes"),
            },
            _ => (None, "whole_work_counter_unknown"),
        }
    };
    left.utility_spent
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| invalid("left whole cost already used in a utility pair"))?;
    right
        .utility_spent
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| {
            invalid("right whole cost already used in a utility pair; left remains spent")
        })?;
    Ok(CheckedConditionalCostPair {
        left,
        right,
        preference,
        reason,
    })
}

fn same_effective_action(a: &StrategicAction, b: &StrategicAction) -> bool {
    // Slot, legacy bucket and a profile registry index cannot establish different
    // actual execution by themselves. Until resolved profile controls are part
    // of a typed fact, an index-only difference remains repetition/unknown.
    a.task == b.task
        && a.semantic_input_sha256 == b.semantic_input_sha256
        && a.baseline_depth == b.baseline_depth
        && a.requested_depth == b.requested_depth
        && a.max_nodes_per_check == b.max_nodes_per_check
        && a.max_wall_time_ms == b.max_wall_time_ms
        && a.max_output_bytes == b.max_output_bytes
}

fn endpoints<'query, 'owner>(
    query: &'query CheckedRepairQuery<'owner>,
) -> Option<(
    &'query ConditionalEndpointFact,
    &'query ConditionalEndpointFact,
)> {
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    if let Some(captured) = query.captured_witness() {
        let cpu = captured.cpu();
        if !cpu.final_cpu_completion_checked() {
            return None;
        }
        return checked_endpoint_pair(cpu.repair_endpoint()?, cpu.opponent_endpoint()?);
    }
    let stages = query.native()?.projection().stages();
    let repaired = stages
        .iter()
        .find(|s| s.phase() == Some(PriorPhase::RepairEndpoint))?
        .conditional_endpoint()?;
    let counter = stages
        .iter()
        .find(|s| s.phase() == Some(PriorPhase::RepairOpponentEndpoint))?
        .conditional_endpoint()?;
    checked_endpoint_pair(repaired, counter)
}
fn checked_endpoint_pair<'a>(
    repaired: &'a ConditionalEndpointFact,
    counter: &'a ConditionalEndpointFact,
) -> Option<(&'a ConditionalEndpointFact, &'a ConditionalEndpointFact)> {
    if repaired.value_identity() != counter.value_identity()
        || repaired.registered_condition_sha256() != counter.registered_condition_sha256()
        || repaired.completed_depth() != counter.completed_depth()
        || repaired.root_perspective() != counter.root_perspective()
        || counter.raw_value_in_root_perspective() >= repaired.raw_value_in_root_perspective()
    {
        return None;
    }
    Some((repaired, counter))
}
fn same_conditional_witness(a: &CheckedRepairQuery<'_>, b: &CheckedRepairQuery<'_>) -> bool {
    let (Some((ar, ac)), Some((br, bc))) = (endpoints(a), endpoints(b)) else {
        return false;
    };
    same_witness_lines(a, b) && ar.same_conditional_fact(br) && ac.same_conditional_fact(bc)
}
fn same_witness_lines(a: &CheckedRepairQuery<'_>, b: &CheckedRepairQuery<'_>) -> bool {
    if let (Some(a), Some(b)) = (a.native(), b.native()) {
        return a.projection().repaired_line() == b.projection().repaired_line()
            && a.projection().opponent_counterline() == b.projection().opponent_counterline();
    }
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    if let (Some(a), Some(b)) = (a.captured_witness(), b.captured_witness()) {
        return a.cpu().repaired_line() == b.cpu().repaired_line()
            && a.cpu().opponent_counterline() == b.cpu().opponent_counterline();
    }
    false
}
fn same_witness_owner(a: &CheckedRepairQuery<'_>, b: &CheckedRepairQuery<'_>) -> bool {
    if let (Some(a), Some(b)) = (a.native(), b.native()) {
        return std::ptr::eq(a.observation(), b.observation());
    }
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    if let (Some(a), Some(b)) = (a.captured_witness(), b.captured_witness()) {
        return std::ptr::eq(a, b)
            || std::ptr::eq(a.cpu(), b.cpu())
            || a.nn_inputs().iter().any(|x| {
                b.nn_inputs().iter().any(|y| {
                    x.process_epoch() == y.process_epoch()
                        && x.lease_sequence() == y.lease_sequence()
                })
            });
    }
    false
}
fn pareto(a: WholeActionCosts, b: WholeActionCosts) -> Option<CostPreference> {
    let av = [a.whole_elapsed_ns, a.cpu_nodes, a.physical_nn_inputs];
    let bv = [b.whole_elapsed_ns, b.cpu_nodes, b.physical_nn_inputs];
    if av == bv {
        None
    } else if av.iter().zip(bv).all(|(a, b)| *a <= b) {
        Some(CostPreference::Left)
    } else if av.iter().zip(bv).all(|(a, b)| *a >= b) {
        Some(CostPreference::Right)
    } else {
        None
    }
}
fn control(deadline: Instant, cancel: &AtomicBool) -> Result<(), ArenaError> {
    if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
        Err(invalid("coverage original W/cancellation"))
    } else {
        Ok(())
    }
}
fn nanos(start: Instant, at: Instant) -> Result<u64, ArenaError> {
    at.checked_duration_since(start)
        .and_then(|d| u64::try_from(d.as_nanos()).ok())
        .ok_or_else(|| invalid("coverage actual clock extent/order"))
}
fn invalid(message: &str) -> ArenaError {
    ArenaError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pin(raw: &[u8]) -> ArtifactPin {
        ArtifactPin {
            bytes: raw.len() as u64,
            sha256: format!("{:x}", Sha256::digest(raw)),
        }
    }
    fn synthetic_offer_inputs() -> (Vec<u8>, Vec<u8>, ReplayBindingPins) {
        // Pure selection-byte fixture only. It never constructs a native,
        // process, Query, completed-action or utility capability.
        let offer = json!({"slot":1,"task":"defend_response","semantic_input_sha256":"1".repeat(64),
            "profile_registration":0,"baseline_depth":1,"requested_depth":2,"max_nodes_per_check":100,
            "max_wall_time_ms":1000,"max_output_bytes":4096,"budget_bucket":1});
        let mut defer = offer.clone();
        defer["slot"] = json!(0);
        defer["task"] = json!("defer");
        let catalogue = serde_json::to_vec(&json!({"actions":[defer,offer]})).unwrap();
        let binding = ReplayBindingPins {
            query_sha256: "2".repeat(64),
            catalogue_artifact: pin(&catalogue),
            before_result_artifact: pin(b"synthetic original before"),
            prior_ledger_sha256: "3".repeat(64),
            semantic_input_sha256: "1".repeat(64),
            semantic_context_sha256: "4".repeat(64),
            semantic_branch_meaning_sha256: "5".repeat(64),
            semantic_before_result_anchor_sha256: "6".repeat(64),
            cpu_request_artifact: pin(b"synthetic CPU request"),
        };
        let prepared=serde_json::to_vec(&json!({"schema":rz_uci::pals_cpu_task::strategic_action::SCHEMA,
            "query_sha256":binding.query_sha256,"catalogue_artifact":binding.catalogue_artifact,
            "before_result_artifact":binding.before_result_artifact,"prior_ledger_sha256":binding.prior_ledger_sha256,
            "cpu_request_artifact":binding.cpu_request_artifact,"action":offer})).unwrap();
        (catalogue, prepared, binding)
    }
    #[test]
    fn pure_coverage_selects_first_actual_offer_and_requires_unchanged_prepared_action() {
        let (catalogue, prepared, binding) = synthetic_offer_inputs();
        assert_eq!(
            first_registered_repair_offer(&catalogue, &binding, &prepared, &pin(&prepared))
                .unwrap(),
            1
        );
        let mut changed: serde_json::Value = serde_json::from_slice(&prepared).unwrap();
        changed["action"]["max_nodes_per_check"] = json!(101);
        let changed = serde_json::to_vec(&changed).unwrap();
        assert!(
            first_registered_repair_offer(&catalogue, &binding, &changed, &pin(&changed)).is_err()
        );
        assert!(
            first_registered_repair_offer(&catalogue, &binding, &prepared, &pin(&changed)).is_err()
        );
        let mut wrong = binding.clone();
        wrong.prior_ledger_sha256 = "7".repeat(64);
        assert!(
            first_registered_repair_offer(&catalogue, &wrong, &prepared, &pin(&prepared)).is_err()
        );
    }
    #[test]
    fn only_source_fixed_callback_covers_initial_selector_work() {
        let (catalogue, prepared, binding) = synthetic_offer_inputs();
        let parent = ReplayParentPins {
            parent_input_sha256: "8".repeat(64),
            current_view_sha256: "9".repeat(64),
            frozen_admission_sha256: "a".repeat(64),
            encoding_sha256: "b".repeat(64),
        };
        let cancel = AtomicBool::new(false);
        let started = Instant::now();
        let whole = started + Duration::from_secs(1);
        let generic = select_registered_repair_coverage(
            &parent,
            &binding,
            &catalogue,
            &prepared,
            &pin(&prepared),
            started,
            whole,
            &cancel,
        )
        .unwrap();
        assert!(!generic.covered_callback);
        let (fixed, episode) = begin_registered_repair_coverage_episode(
            &parent,
            &binding,
            &catalogue,
            &prepared,
            &pin(&prepared),
            1,
            Duration::from_secs(1),
            &cancel,
        )
        .unwrap();
        assert!(fixed.covered_callback);
        assert_eq!(fixed.action_index(), 1);
        assert_eq!(fixed.choice(), CoverageChoice::DefendResponse);
        assert_eq!(fixed.whole, episode.original_deadline());
        assert!(fixed.started >= episode.original_started());
        // No actual native owner, completed Query or utility is constructed.
    }
    #[test]
    fn registered_index_is_bounded_actual_offer_and_repetition_ignores_labels() {
        let (catalogue, prepared, mut binding) = synthetic_offer_inputs();
        assert!(
            registered_repair_offer(&catalogue, &binding, &prepared, &pin(&prepared), Some(0))
                .is_err()
        );
        assert!(
            registered_repair_offer(&catalogue, &binding, &prepared, &pin(&prepared), Some(8))
                .is_err()
        );
        let mut value: serde_json::Value = serde_json::from_slice(&catalogue).unwrap();
        let mut other = value["actions"][1].clone();
        other["slot"] = json!(2);
        other["max_wall_time_ms"] = json!(1100);
        value["actions"].as_array_mut().unwrap().push(other.clone());
        let catalogue = serde_json::to_vec(&value).unwrap();
        binding.catalogue_artifact = pin(&catalogue);
        let mut original: serde_json::Value = serde_json::from_slice(&prepared).unwrap();
        original["catalogue_artifact"] = json!(binding.catalogue_artifact);
        original["action"] = other;
        let prepared = serde_json::to_vec(&original).unwrap();
        let (index, offer) =
            registered_repair_offer(&catalogue, &binding, &prepared, &pin(&prepared), Some(2))
                .unwrap();
        assert_eq!(index, 2);
        let original_offer: StrategicAction =
            serde_json::from_value(value["actions"][1].clone()).unwrap();
        assert!(!same_effective_action(&offer, &original_offer));
        let mut labels = original_offer.clone();
        labels.slot = 2;
        labels.budget_bucket = 7;
        labels.profile_registration = 3;
        assert!(same_effective_action(&original_offer, &labels));
        let parent = ReplayParentPins {
            parent_input_sha256: "8".repeat(64),
            current_view_sha256: "9".repeat(64),
            frozen_admission_sha256: "a".repeat(64),
            encoding_sha256: "b".repeat(64),
        };
        let (selection, _) = begin_registered_defend_coverage_episode(
            &parent,
            &binding,
            &catalogue,
            &prepared,
            &pin(&prepared),
            2,
            1,
            Duration::from_secs(1),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(selection.policy(), REGISTERED_COVERAGE_POLICY);
        assert!(selection.covered_callback);
    }
    #[test]
    fn pure_coverage_never_clips_or_invents_missing_unbounded_offers() {
        let (catalogue, prepared, mut binding) = synthetic_offer_inputs();
        let mut value: serde_json::Value = serde_json::from_slice(&catalogue).unwrap();
        value["actions"][1]["task"] = json!("defer");
        let missing = serde_json::to_vec(&value).unwrap();
        binding.catalogue_artifact = pin(&missing);
        assert!(
            first_registered_repair_offer(&missing, &binding, &prepared, &pin(&prepared)).is_err()
        );
        let unbounded =
            serde_json::to_vec(&json!({"actions":vec![value["actions"][0].clone();9]})).unwrap();
        binding.catalogue_artifact = pin(&unbounded);
        assert!(
            first_registered_repair_offer(&unbounded, &binding, &prepared, &pin(&prepared))
                .is_err()
        );
    }
    #[test]
    fn pareto_requires_every_axis_and_one_strict_gain() {
        let a = WholeActionCosts {
            whole_elapsed_ns: 10,
            cpu_nodes: 20,
            physical_nn_inputs: 30,
        };
        assert_eq!(pareto(a, a), None);
        assert_eq!(
            pareto(
                WholeActionCosts {
                    whole_elapsed_ns: 9,
                    ..a
                },
                a
            ),
            Some(CostPreference::Left)
        );
        assert_eq!(
            pareto(
                a,
                WholeActionCosts {
                    physical_nn_inputs: 29,
                    ..a
                }
            ),
            Some(CostPreference::Right)
        );
        assert_eq!(
            pareto(
                WholeActionCosts {
                    whole_elapsed_ns: 9,
                    cpu_nodes: 21,
                    ..a
                },
                a
            ),
            None
        );
        assert_eq!(
            pareto(
                WholeActionCosts {
                    whole_elapsed_ns: u64::MAX,
                    cpu_nodes: 0,
                    physical_nn_inputs: 0
                },
                a
            ),
            None
        );
    }
    #[test]
    fn clock_refuses_reversal_expiry_and_cancel_without_new_window() {
        let now = Instant::now();
        let cancel = AtomicBool::new(false);
        let past = now
            .checked_sub(std::time::Duration::from_millis(1))
            .expect("clock fixture subtraction must be representable");
        assert!(past < now, "clock fixture must be strictly earlier");
        assert!(nanos(now, past).is_err());
        assert!(control(now, &cancel).is_err());
        cancel.store(true, Ordering::Release);
        assert!(control(now + std::time::Duration::from_secs(1), &cancel).is_err());
    }
}
