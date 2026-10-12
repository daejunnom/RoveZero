"""Strict, always-masked preparation for same_witness_cost_dominance/1.

This policy is a newly chosen conditional cost criterion, not the PALS paper's
Reward formula. It would need two independent completed actions from the same
pre-result question/prior, the same verified conditional witness, and whole
causal cost observations. None of that action bridge is implemented here.

Query/2 only admits a question/catalogue. CheckedNativeRecheckWitness admits a
conditional same-line observation, not the final search envelope or whole
action cost. Legacy CPU receipt bytes do not connect that native execution to
an action. Caller JSON, depth, CPU agreement, lack of a counterexample, or old
coverage-utility results cannot fill those gaps. This unit therefore ALWAYS
returns unknown/masked, preference/sign/cost=None and actual utility groups0.

Original raw bytes and independent pins are retained. The separate /2 report
inspectors below read actual Rust coverage-cost audits without importing their
capabilities. Reported numerical costs/preferences remain caller reports; these
inspectors never admit utility, a target, training, warm state or product V.
"""

import copy

from . import native_recheck_witness as native
from . import strategic_verifier_query as query


POLICY = "same_witness_cost_dominance/1"
CRITERION_SCHEMA = "rz-pals-private-v-strategic-cost-criterion/1"
BEFORE_SCHEMA = "rz-pals-private-v-strategic-utility-before-pair/1"
OBSERVATION_SCHEMA = "rz-pals-private-v-strategic-action-unknown/1"
ADMISSION_SCHEMA = "rz-pals-private-v-strategic-utility-masked/1"
SCOPE = "same_conditional_witness_whole_action_cost_policy_pending_causal_bridge"
CALLER_SCOPE = "independently_pinned_caller_declaration;no_causal_execution_or_cost_authority"
MAX_NEW_RAW_BYTES = 4 * 1024 * 1024
_FACTORY = object()
_STATES = ("unobserved", "missing", "partial", "canceled", "failed", "deferred", "declared_completed")
_COST_AXES = ("whole_elapsed_ms", "actual_cpu_nodes", "physical_nn_rows")
_DENIED = {"utility_authority": False, "target_authority": False, "training_authority": False,
           "action_completion_admitted": False, "whole_cost_admitted": False,
           "final_search_closure_admitted": False, "actual_training_executed": False,
           "backward_executed": False, "product_verifier_enabled": False,
           "coverage_utility_reused": False, "depth_or_cpu_agreement_reward": False,
           "no_counterexample_reward": False, "paper_reward_claim": False}


def _fields(value, names):
    # A caller-owned bundle reaches this guard before raw byte accounting.
    # Reject its cardinality before copying/traversing arbitrary unknown keys.
    if type(value) is not dict or len(value) != len(names) or any(name not in value for name in names):
        raise ValueError("strategic utility closed fields required")
    return value


def _uint(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("strategic utility exact bounded integer required")
    return value


def _aggregate(criterion, before, observations):
    if type(observations) not in (tuple, list) or len(observations) != 2:
        raise ValueError("strategic utility exactly two observation slots required")
    total = 0
    for raw in (criterion, before):
        if type(raw) is not bytes or not raw:
            raise ValueError("strategic utility nonempty immutable raw bytes required")
        total += len(raw)
    for bundle in observations:
        _fields(bundle, ("observation", "request", "response"))
        for name, raw in bundle.items():
            if raw is None and name != "observation":
                continue
            if type(raw) is not bytes or not raw:
                raise ValueError("strategic utility immutable raw observation/request/response required")
            total += len(raw)
    if total > MAX_NEW_RAW_BYTES:
        raise ValueError("strategic utility aggregate new immutable raw 4 MiB exceeded")
    return total


def _actual(raw, expected):
    query._pin(expected)
    if query.byte_pin(raw) != expected:
        raise ValueError("strategic utility independent original raw pin mismatch")


def _criterion(raw):
    value = _fields(query._parse(raw), ("schema", "policy", "scope", "cost_axes", "witness_scope",
        "requires_conditional_publication", "requires_same_pre_result_question_and_prior",
        "requires_independent_completed_actions", "requires_whole_causal_cost"))
    if (value["schema"] != CRITERION_SCHEMA or value["policy"] != POLICY or value["scope"] != SCOPE
            or type(value["cost_axes"]) is not list or value["cost_axes"] != list(_COST_AXES)
            or value["witness_scope"] != native.SCOPE or any(value[name] is not True for name in
                ("requires_conditional_publication", "requires_same_pre_result_question_and_prior",
                 "requires_independent_completed_actions", "requires_whole_causal_cost"))):
        raise ValueError("unsupported strategic same-witness cost criterion")
    return value


def _before(raw, checked, criterion_raw):
    value = _fields(query._parse(raw), ("schema", "query_sha256", "parent", "catalogue", "query_before_result",
        "prior_ledger_sha256", "criterion", "actions", "assurance_scope"))
    assets = checked.raw_assets()
    catalogue = checked.catalogue()
    original_before = query._parse(assets["before_result"])
    if (value["schema"] != BEFORE_SCHEMA or value["query_sha256"] != checked.sha256
            or value["parent"] != checked.audit()["parent"]
            or value["catalogue"] != query.byte_pin(assets["catalogue"])
            or value["query_before_result"] != query.byte_pin(assets["before_result"])
            or value["prior_ledger_sha256"] != original_before["prior_ledger_sha256"]
            or value["criterion"] != query.byte_pin(criterion_raw) or value["assurance_scope"] != CALLER_SCOPE):
        raise ValueError("strategic utility before query/current/prior/criterion binding mismatch")
    if type(value["actions"]) is not list or len(value["actions"]) != 2:
        raise ValueError("strategic utility pair must have exactly two preselected actions")
    semantics = checked.action_semantic_inputs()
    indices = []
    for action in value["actions"]:
        _fields(action, ("action_index", "semantic_input_sha256", "registered_profile"))
        index = _uint(action["action_index"], 0, len(catalogue["actions"]) - 1)
        spec = catalogue["actions"][index]
        registration = assets["profiles"][spec["profile_registration"]]
        if (action["semantic_input_sha256"] != semantics[index].sha256
                or action["registered_profile"] != {name: query.byte_pin(data) for name, data in registration.items()}):
            raise ValueError("strategic utility before action/actual registered profile mismatch")
        indices.append(index)
    if len(set(indices)) != 2:
        raise ValueError("strategic utility cannot compare one action to itself")
    return value, tuple(indices)


def _unknown_observation(bundle, expected, witness, checked, before_raw, action_index):
    _fields(expected, ("observation", "request", "response", "whole_witness_sha256"))
    for name in ("observation", "request", "response"):
        if bundle[name] is None:
            if name == "observation" or expected[name] is not None:
                raise ValueError("strategic utility missing original raw differs from independent pin")
        else:
            _actual(bundle[name], expected[name])
    value = _fields(query._parse(bundle["observation"]), ("schema", "query_sha256", "before_pair_sha256", "action_index",
        "semantic_input_sha256", "request", "response", "whole_witness_sha256", "state", "reason", "assurance_scope"))
    semantics = checked.action_semantic_inputs()
    if (value["schema"] != OBSERVATION_SCHEMA or value["query_sha256"] != checked.sha256
            or value["before_pair_sha256"] != query.byte_pin(before_raw)["sha256"]
            or type(value["action_index"]) is not int or value["action_index"] != action_index
            or value["semantic_input_sha256"] != semantics[action_index].sha256
            or value["assurance_scope"] != CALLER_SCOPE):
        raise ValueError("strategic utility original action/query/before binding mismatch")
    for name in ("request", "response"):
        if value[name] != expected[name]:
            raise ValueError("strategic utility observation changed its original raw identity")
    if type(value["state"]) is not str or value["state"] not in _STATES:
        raise ValueError("strategic utility unknown observation state")
    reason = value["reason"]
    if reason is not None and (type(reason) is not str or not 1 <= len(reason.encode("utf-8")) <= 256
                               or any(ord(char) < 32 for char in reason)):
        raise ValueError("strategic utility bounded unknown reason required")
    if value["state"] in ("missing", "partial", "canceled", "failed") and reason is None:
        raise ValueError("strategic utility incomplete raw observation needs reason")
    if value["response"] is not None and value["request"] is None:
        raise ValueError("strategic utility response without original request")
    if value["state"] == "missing" and value["response"] is not None:
        raise ValueError("strategic utility missing state invented a CPU receipt")
    if value["state"] == "declared_completed" and (value["request"] is None or value["response"] is None):
        raise ValueError("strategic utility declared completion lacks even original CPU bytes")
    assets, catalogue = checked.raw_assets(), checked.catalogue()
    spec = catalogue["actions"][action_index]
    profile_index = spec["profile_registration"]
    registration, cap, _ = query._registered(assets["profiles"][profile_index], assets["expected_pins"]["profiles"][profile_index])
    raw_cpu = None
    if bundle["request"] is not None:
        # Read-only reuse of Query/2's exact legacy raw validator. This never
        # changes request bytes, calls observed_gain, or admits a native action.
        request, _, _ = query._request(bundle["request"], semantics[action_index], registration, cap,
                                      checked._parent, checked._index, checked.audit()["parent"])
        if request["task"] != spec["task"] or request["max_output_bytes"] != spec["max_output_bytes"]:
            raise ValueError("strategic utility legacy request differs from selected action")
        if bundle["response"] is not None:
            if len(bundle["response"]) > request["max_output_bytes"]:
                raise ValueError("strategic utility legacy response output extent")
            response = query._parse(bundle["response"])
            query._incomplete_receipt(request, response)
            if value["state"] == "deferred" and response["status"] != "deferred":
                raise ValueError("strategic utility declared Defer contradicts raw CPU status")
            if value["state"] == "declared_completed" and response["deadline_exceeded"]:
                raise ValueError("strategic utility completion declaration contradicts raw deadline")
            raw_cpu = {"status": response["status"], "deadline_exceeded": response["deadline_exceeded"],
                       "nodes": response["nodes"], "elapsed_ms": response["elapsed_ms"]}
    if value["state"] == "deferred" and spec["task"] != "defer":
        raise ValueError("strategic utility Defer declaration differs from selected task")
    observed = None
    witness_relation = None
    if witness is None:
        if expected["whole_witness_sha256"] is not None or value["whole_witness_sha256"] is not None:
            raise ValueError("strategic utility unobserved whole witness carries a checked pin")
    else:
        if type(witness) is not native.CheckedNativeRecheckWitness:
            raise ValueError("strategic utility exact whole native recheck capability required")
        witness.verify()
        if expected["whole_witness_sha256"] != witness.sha256 or value["whole_witness_sha256"] != witness.sha256:
            raise ValueError("strategic utility independent whole witness pin mismatch")
        observed = witness.observation()
        if observed["search_completion_admitted"] is not False or observed["final_search_envelope_observed"] is not False:
            raise ValueError("strategic utility unsupported expanded native witness authority")
        # Narrow read-only coupling to the original checked Repair parent. Both
        # capabilities have already revalidated their own unchanged receipts;
        # this comparison does not create a causal query -> action -> witness.
        causal = witness._anchor._repair
        native_row = causal._parents.records[causal._repair_index]
        query_row = checked._parent.records[checked._index]
        witness_relation = {"parent_input_matches": native_row["input"]["sha256"] == query_row["input"]["sha256"],
            "source_matches": native_row["input"]["snapshot"]["source"] == query_row["input"]["snapshot"]["source"],
            "frozen_epoch_matches": native_row["input"]["snapshot"]["frozen_epoch"] == query_row["input"]["snapshot"]["frozen_epoch"],
            "current_view_matches": causal._parents.current_view.sha256 == checked._parent.current_view.sha256,
            "causal_action_binding_admitted": False}
    return {"action_index": action_index, "task": spec["task"], "declared_state": value["state"], "reason": reason,
            "raw_cpu_receipt": raw_cpu, "whole_witness_observation": observed, "witness_query_relation": witness_relation,
            "raw_cpu_is_whole_action_cost": False, "action_completion_admitted": False}


def _validate(checked, criterion_raw, before_raw, observations, witnesses, expected_pins):
    total = _aggregate(criterion_raw, before_raw, observations)
    if type(witnesses) not in (tuple, list) or len(witnesses) != 2:
        raise ValueError("strategic utility exactly two whole witness slots required")
    if type(checked) is not query.CheckedStrategicQuery:
        raise ValueError("strategic utility exact CheckedStrategicQuery required")
    # Reserve pin metadata BEFORE canonical serialization, including repeated
    # references. Existing Query/2 and native assets retain their own admission
    # limits and are borrowed, never reconstructed/sanitized or copied here.
    pin_raw = query.canonical(expected_pins, max_bytes=MAX_NEW_RAW_BYTES - total)
    total += len(pin_raw)
    pins = _fields(query._parse(pin_raw), ("query_sha256", "criterion", "before_pair", "observations"))
    if type(pins["observations"]) is not list or len(pins["observations"]) != 2:
        raise ValueError("strategic utility independent observation pin count")
    checked.verify()
    if pins["query_sha256"] != checked.sha256:
        raise ValueError("strategic utility independent checked query mismatch")
    _actual(criterion_raw, pins["criterion"])
    _actual(before_raw, pins["before_pair"])
    criterion = _criterion(criterion_raw)
    before, indices = _before(before_raw, checked, criterion_raw)
    values = tuple(_unknown_observation(bundle, pin, witness, checked, before_raw, index) for
                   bundle, pin, witness, index in zip(observations, pins["observations"], witnesses, indices))
    features = checked.features()
    reasons = ["native_action_causal_bridge_unobserved"]
    if features[indices[0]].branch != features[indices[1]].branch:
        reasons.append("different_checked_pre_result_question")
    if any(value["task"] == "defer" for value in values):
        reasons.append("defer_has_no_independent_conditional_witness_action")
    if any(value["declared_state"] != "declared_completed" for value in values):
        reasons.append("action_completion_not_observed")
    if any(value["whole_witness_observation"] is None or
           value["whole_witness_observation"]["status"] != "conditional_cp_observation" for value in values):
        reasons.append("missing_partial_or_unknown_conditional_witness")
    elif any(value["whole_witness_observation"].get("conditional_counter_lower") is not True for value in values):
        reasons.append("no_conditional_refutation_publication_witness")
    if witnesses[0] is not None and witnesses[1] is not None and witnesses[0].sha256 == witnesses[1].sha256:
        reasons.append("shared_witness_does_not_prove_independent_actions")
    if any(value["witness_query_relation"] is not None and any(value["witness_query_relation"][name] is not True
           for name in ("parent_input_matches", "source_matches", "frozen_epoch_matches", "current_view_matches")) for value in values):
        reasons.append("conditional_witness_parent_source_or_current_differs_from_query")
    return {"schema": ADMISSION_SCHEMA, "policy": criterion["policy"], "scope": SCOPE,
            "query_sha256": checked.sha256, "parent": copy.deepcopy(before["parent"]),
            "action_indices": indices, "observations": values, "status": "masked_unresolved", "known": False,
            "mask": False, "preference": None, "sign": None, "whole_action_cost": None,
            "same_conditional_fact_admitted": False, "reason": reasons[0], "additional_reasons": tuple(reasons[1:]),
            "new_immutable_raw_bytes": total, "actual_utility_groups": 0, **_DENIED}


class CheckedStrategicUtilityPair:
    """Factory-only, immutable, always masked; not a target/loss capability."""
    __slots__ = ("_query", "_criterion", "_before", "_observations", "_witnesses", "_pins", "_identity")

    def __init__(self, token=None, *, checked=None, criterion=None, before=None, observations=(), witnesses=(), pins=None):
        if token is not _FACTORY:
            raise ValueError("use admit_same_witness_cost_dominance; unchecked utility refused")
        for name, value in (("_query", checked), ("_criterion", criterion), ("_before", before),
                            ("_observations", tuple(tuple(sorted(bundle.items())) for bundle in observations)),
                            ("_witnesses", tuple(witnesses)), ("_pins", pins),
                            ("_identity", query.digest(ADMISSION_SCHEMA, query._parse(pins)))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("checked strategic masked utility is immutable")

    def _view(self):
        pins = query._parse(self._pins)
        if query.digest(ADMISSION_SCHEMA, pins) != self._identity:
            raise ValueError("strategic utility capability identity changed")
        return _validate(self._query, self._criterion, self._before,
                         tuple(dict(bundle) for bundle in self._observations), self._witnesses, pins)

    def verify(self):
        self._view()
        return self

    @property
    def sha256(self):
        self.verify()
        return self._identity

    def observation(self):
        return copy.deepcopy(self._view())

    def raw_assets(self):
        self.verify()
        return {"criterion": self._criterion, "before_pair": self._before,
                "observations": tuple(dict(bundle) for bundle in self._observations),
                "expected_pins": query._parse(self._pins)}


def admit_same_witness_cost_dominance(*, checked_query, criterion_bytes, before_pair_bytes,
                                    action_observations, whole_witnesses, expected_pins):
    """Strict unknown parser only; no caller declaration can enable a sign.

    Two slots can preserve honest missing/partial/deferred or even completed
    legacy CPU raw facts and exact whole native capabilities. The action bridge,
    final closure and whole cost are still unobserved, so every result remains
    masked. This function never imports old coverage utility as positive data.
    """
    _validate(checked_query, criterion_bytes, before_pair_bytes, action_observations, whole_witnesses, expected_pins)
    total = _aggregate(criterion_bytes, before_pair_bytes, action_observations)
    pins = query.canonical(expected_pins, max_bytes=MAX_NEW_RAW_BYTES - total)
    return CheckedStrategicUtilityPair(_FACTORY, checked=checked_query, criterion=criterion_bytes, before=before_pair_bytes,
                                      observations=action_observations, witnesses=whole_witnesses, pins=pins)


WHOLE_COST_REPORT_SCHEMA = "rz-pals-coverage-whole-action-cost/2"
CONDITIONAL_COST_REPORT_SCHEMA = "rz-pals-conditional-whole-cost-dominance/2"
CAPTURED_WHOLE_COST_REPORT_SCHEMA = "rz-pals-coverage-whole-action-cost/3"
CAPTURED_CONDITIONAL_COST_REPORT_SCHEMA = "rz-pals-conditional-whole-cost-dominance/3"
_COVERAGE_POLICY = "first_registered_defend_response_coverage/1"
_REGISTERED_COVERAGE_POLICY = "registered_defend_response_coverage/2"
_COUNTER_SCOPE = "registered_child_report_correlated_to_independent_native_plus_separate_live_native_execution"
_CAPTURED_COUNTER_SCOPE = "child_work_report_plus_actual_captured_input_nn_replay_and_independent_cpu_work"
_SOURCE_REPORT_SCOPE = "registered_source_owned_report;physical_child_work_not_attested"
_INDEPENDENT_WORK_SCOPE = "fresh_independent_cpu_attempts_and_actual_recorded_input_nn_ready_stats_including_initialization_and_final_join"
_FRESH_FACT_SCOPE = "actual_fresh_independent_cpu_endpoint_fact_only;not_correlated_child_scalar_score"
_WHOLE_AXES = ("whole_elapsed_ns", "cpu_nodes", "physical_nn_inputs")
_WHOLE_REPORT_FIELDS = ("schema", "policy", "query_sha256", "parent", "source_binding", "source_input",
    "source_output", "conditional_fact_sha256", "selected_original_action_index", "selected_action", "whole_elapsed_ns", "whole_cost",
    "initial_selection", "next_selection", "independent_native_work_interval", "source_work", "independent_native_work",
    "work_counter_scope", "source_native_ids_equated_to_witness", "whole_causal_elapsed_observed",
    "preparation_transport_check_intervals_ns", "logical_phase_elapsed_ns", "logical_phase_scope",
    "neural_v_executed", "neural_v_skipped_by_policy", "selector_cpu_search_executed", "selector_nn_inputs",
    "next_selection_cpu_search_executed", "next_selection_nn_inputs", "next_choice", "whole_work_counts_known",
    "utility_authority", "target_authority", "training_authority", "learned_utility_claim",
    "product_verifier_enabled", "optimizer_steps")
_PAIR_REPORT_FIELDS = ("schema", "policy", "scope", "parent", "source_query_sha256",
    "prior_ledger_before_sha256", "source_inputs", "selected_original_action_indices", "source_actions", "whole_costs", "work_counter_scope", "preference", "mask",
    "reason", "actual_utility_groups", "utility_observation_admitted", "target_authority", "training_authority",
    "neural_v_executed", "learned_utility_claim", "strategic_refutation_admitted", "rules_proof_admitted",
    "depth_or_cpu_agreement_reward", "no_counterexample_reward", "paper_reward_claim",
    "product_verifier_enabled", "optimizer_steps")
_CAPTURED_WHOLE_REPORT_FIELDS = tuple(name for name in _WHOLE_REPORT_FIELDS if name != "independent_native_work_interval") + (
    "independent_witness_work_interval", "source_child_work_provenance", "actual_independent_work_scope",
    "source_child_physical_work_attested", "source_cpu_scalar_score_correlated")
_CAPTURED_PAIR_REPORT_FIELDS = _PAIR_REPORT_FIELDS + ("source_child_work_provenance",
    "source_child_physical_work_attested", "source_cpu_scalar_score_correlated", "conditional_fact_scope")
_REPORT_PARENT_FIELDS = ("parent_input_sha256", "current_view_sha256", "frozen_admission_sha256", "encoding_sha256")
_REPORT_BINDING_SHA_FIELDS = ("query_sha256", "prior_ledger_sha256", "semantic_input_sha256",
    "semantic_context_sha256", "semantic_branch_meaning_sha256", "semantic_before_result_anchor_sha256")
_REPORT_BINDING_PIN_FIELDS = ("catalogue_artifact", "before_result_artifact", "cpu_request_artifact")
_REPORT_ACTION_FIELDS = ("slot", "task", "semantic_input_sha256", "profile_registration", "baseline_depth",
    "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "max_output_bytes", "budget_bucket")


def _report_original(raw, expected_pin, names, captured_names=None, captured_schema=None):
    # Validate bounded immutable bytes and the closed independent pin before
    # parsing/copying caller objects. A pin is provenance, never native authority.
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_NEW_RAW_BYTES:
        raise ValueError("coverage cost report bounded original immutable bytes required")
    _fields(expected_pin, ("bytes", "sha256"))
    _uint(expected_pin["bytes"], 1, MAX_NEW_RAW_BYTES)
    query._sha(expected_pin["sha256"])
    _actual(raw, expected_pin)
    value = query._parse(raw)
    if type(value) is dict and value.get("schema") == captured_schema and captured_names is not None:
        names = captured_names
    return _fields(value, names)


def _captured_report_provenance(value, *, pair=False):
    if (value["source_child_work_provenance"] != _SOURCE_REPORT_SCOPE
            or value["source_child_physical_work_attested"] is not False
            or value["source_cpu_scalar_score_correlated"] is not None
            or (pair and value["conditional_fact_scope"] != _FRESH_FACT_SCOPE)
            or (not pair and value["actual_independent_work_scope"] != _INDEPENDENT_WORK_SCOPE)):
        raise ValueError("captured-input report cannot promote child work or scalar score to physical proof")


def _whole_report_cost(value):
    if value is None:
        return None
    _fields(value, _WHOLE_AXES)
    return tuple(_uint(value[name], 0, 2**64 - 1) for name in _WHOLE_AXES)


def _report_parent(value):
    _fields(value, _REPORT_PARENT_FIELDS)
    for name in _REPORT_PARENT_FIELDS:
        query._sha(value[name])


def _report_binding(value):
    _fields(value, _REPORT_BINDING_SHA_FIELDS + _REPORT_BINDING_PIN_FIELDS)
    for name in _REPORT_BINDING_SHA_FIELDS:
        query._sha(value[name])
    for name in _REPORT_BINDING_PIN_FIELDS:
        query._pin(value[name])


def _report_action(value, index):
    _fields(value, _REPORT_ACTION_FIELDS)
    if value["task"] != "defend_response" or _uint(value["slot"], 0, 7) != index:
        raise ValueError("coverage report original selected DefendResponse action differs")
    query._sha(value["semantic_input_sha256"])
    for name in ("profile_registration", "budget_bucket"):
        _uint(value[name], 0, 255)
    baseline, requested = (_uint(value[name], 1, 64) for name in ("baseline_depth", "requested_depth"))
    if baseline > requested:
        raise ValueError("coverage report original CPU depth order differs")
    _uint(value["max_nodes_per_check"], 1, 2**64 - 1)
    _uint(value["max_wall_time_ms"], 1, 300000)
    _uint(value["max_output_bytes"], 1024, 1048576)
    # A registry index alone does not prove different resolved profile controls.
    return tuple(value[name] for name in _REPORT_ACTION_FIELDS if name not in ("slot", "budget_bucket", "profile_registration"))


def _report_interval(value, whole):
    _fields(value, ("start_ns", "end_ns"))
    start, end = (_uint(value[name], 0, whole) for name in ("start_ns", "end_ns"))
    if start > end:
        raise ValueError("coverage cost report observed interval order")
    return start, end


def _report_work(value):
    if value is None:
        return None
    if type(value) is not list or len(value) != 2:
        raise ValueError("coverage cost report exact source and native counter pair required")
    return tuple(_uint(number, 0, 2**64 - 1) for number in value)


def _reported_only(raw, value):
    return {"schema": "rz-pals-coverage-cost-report-inspection/1",
        "scope": "independently_pinned_rust_audit_report_only;no_capability_import",
        "original_report": query.byte_pin(raw), "reported": copy.deepcopy(value),
        "actual_utility_groups": 0, "utility_authority": False, "target_authority": False,
        "training_authority": False, "native_capability_imported": False,
        "physical_execution_attested_here": False, "optimizer_steps": 0}


def inspect_coverage_whole_cost_report(*, report_bytes, expected_report_pin):
    """Read a closed /2 Rust audit without reconstructing a completed action.

    Unknown counters/subphase durations stay None. The retained child report and
    independent native work are distinct executions; their counters are summed
    only when the source declares both and arithmetic/accounting agrees.
    """
    value = _report_original(report_bytes, expected_report_pin, _WHOLE_REPORT_FIELDS,
                            _CAPTURED_WHOLE_REPORT_FIELDS, CAPTURED_WHOLE_COST_REPORT_SCHEMA)
    captured = value["schema"] == CAPTURED_WHOLE_COST_REPORT_SCHEMA
    if captured:
        _captured_report_provenance(value)
    if (value["schema"] not in (WHOLE_COST_REPORT_SCHEMA, CAPTURED_WHOLE_COST_REPORT_SCHEMA)
            or value["policy"] not in (_COVERAGE_POLICY, _REGISTERED_COVERAGE_POLICY)
            or value["work_counter_scope"] != (_CAPTURED_COUNTER_SCOPE if captured else _COUNTER_SCOPE) or value["next_choice"] != "defer"
            or value["logical_phase_scope"] != "combined_parent_intervals_observed;unseparated_subphases_unknown;no_additive_double_count"):
        raise ValueError("unsupported coverage whole-cost report scope")
    for name in ("query_sha256", "conditional_fact_sha256"):
        query._sha(value[name])
    _report_parent(value["parent"])
    _report_binding(value["source_binding"])
    for name in ("source_input", "source_output"):
        query._pin(value[name])
    index = _uint(value["selected_original_action_index"], 0, 7)
    _report_action(value["selected_action"], index)
    whole = _uint(value["whole_elapsed_ns"], 0, 2**64 - 1)
    initial = _report_interval(value["initial_selection"], whole)
    next_selection = _report_interval(value["next_selection"], whole)
    _report_interval(value["independent_witness_work_interval" if captured else "independent_native_work_interval"], whole)
    if initial[1] > next_selection[0] or next_selection[1] != whole:
        raise ValueError("coverage whole-cost final selection clock differs")
    for name in ("source_native_ids_equated_to_witness", "neural_v_executed", "selector_cpu_search_executed",
                 "next_selection_cpu_search_executed", "utility_authority", "target_authority", "training_authority",
                 "learned_utility_claim", "product_verifier_enabled"):
        if value[name] is not False:
            raise ValueError("coverage report expanded authority or unrecorded selector work")
    for name in ("whole_causal_elapsed_observed", "neural_v_skipped_by_policy"):
        if value[name] is not True:
            raise ValueError("coverage report missing actual interval or explicit skipped V policy")
    for name in ("selector_nn_inputs", "next_selection_nn_inputs", "optimizer_steps"):
        if type(value[name]) is not int or value[name] != 0:
            raise ValueError("coverage report cannot add selector NN/training work")
    phases = _fields(value["logical_phase_elapsed_ns"],
                     ("initial_v_or_coverage_selection", "preparation", "nn", "cpu", "transfer", "check", "next_selection"))
    _uint(phases["initial_v_or_coverage_selection"], 0, whole)
    _uint(phases["next_selection"], 0, whole)
    if (phases["initial_v_or_coverage_selection"] != initial[1] - initial[0]
            or phases["next_selection"] != next_selection[1] - next_selection[0]
            or any(phases[name] is not None for name in ("preparation", "nn", "cpu", "transfer", "check"))):
        raise ValueError("coverage report invented unseparated subphase durations")
    intervals = _fields(value["preparation_transport_check_intervals_ns"],
        ("preparation_and_preflight", "supervisor_setup", "launch_work_and_wait", "drain", "postflight",
         "after_capture_before_check", "result_check", "check_to_next_query_admission"))
    for duration in intervals.values():
        if duration is not None:
            _uint(duration, 0, whole)
    source, native_work = _report_work(value["source_work"]), _report_work(value["independent_native_work"])
    costs = _whole_report_cost(value["whole_cost"])
    if type(value["whole_work_counts_known"]) is not bool or value["whole_work_counts_known"] != (costs is not None):
        raise ValueError("coverage report unknown whole counts cannot become zero or known")
    if costs is not None:
        if (source is None or native_work is None or (not captured and source != native_work) or costs[0] != whole
                or costs[1:] != (source[0] + native_work[0], source[1] + native_work[1])):
            raise ValueError("coverage report must charge source and independent witness separately")
    return _reported_only(report_bytes, value)


def inspect_conditional_cost_pair_report(*, report_bytes, expected_report_pin):
    """Retain a /2 conditional preference claim as a report, never a target."""
    value = _report_original(report_bytes, expected_report_pin, _PAIR_REPORT_FIELDS,
                            _CAPTURED_PAIR_REPORT_FIELDS, CAPTURED_CONDITIONAL_COST_REPORT_SCHEMA)
    captured = value["schema"] == CAPTURED_CONDITIONAL_COST_REPORT_SCHEMA
    if captured:
        _captured_report_provenance(value, pair=True)
    if (value["schema"] not in (CONDITIONAL_COST_REPORT_SCHEMA, CAPTURED_CONDITIONAL_COST_REPORT_SCHEMA)
            or value["policy"] != "same_conditional_raw_endpoint_whole_cost_dominance/2"
            or value["scope"] != "conditional_cost_observation_only;not_paper_reward_or_learned_utility"
            or value["work_counter_scope"] != (_CAPTURED_COUNTER_SCOPE if captured else _COUNTER_SCOPE)):
        raise ValueError("unsupported conditional cost pair report scope")
    for name in ("source_query_sha256", "prior_ledger_before_sha256"):
        query._sha(value[name])
    _report_parent(value["parent"])
    if type(value["source_inputs"]) is not list or len(value["source_inputs"]) != 2:
        raise ValueError("conditional cost report exact two source artifacts required")
    for pin in value["source_inputs"]:
        query._pin(pin)
    if type(value["selected_original_action_indices"]) is not list or len(value["selected_original_action_indices"]) != 2:
        raise ValueError("conditional cost report exact two original action indices required")
    indices = tuple(_uint(index, 0, 7) for index in value["selected_original_action_indices"])
    if type(value["source_actions"]) is not list or len(value["source_actions"]) != 2:
        raise ValueError("conditional cost report exact two original actions required")
    actions = tuple(_report_action(action, index) for action, index in zip(value["source_actions"], indices))
    if type(value["whole_costs"]) is not list or len(value["whole_costs"]) != 2:
        raise ValueError("conditional cost report exact two whole costs required")
    costs = tuple(_whole_report_cost(item) for item in value["whole_costs"])
    if type(value["mask"]) is not bool or type(value["utility_observation_admitted"]) is not bool:
        raise ValueError("conditional cost report exact boolean masks required")
    if (type(value["actual_utility_groups"]) is not int or value["actual_utility_groups"] != int(value["mask"])
            or value["utility_observation_admitted"] != value["mask"]):
        raise ValueError("conditional cost report group/mask disagreement")
    if type(value["reason"]) is not str or not 1 <= len(value["reason"]) <= 256:
        raise ValueError("conditional cost report bounded reason required")
    for name in ("target_authority", "training_authority", "neural_v_executed", "learned_utility_claim",
                 "strategic_refutation_admitted", "rules_proof_admitted", "depth_or_cpu_agreement_reward",
                 "no_counterexample_reward", "paper_reward_claim", "product_verifier_enabled"):
        if value[name] is not False:
            raise ValueError("conditional cost report cannot expand observation authority")
    if type(value["optimizer_steps"]) is not int or value["optimizer_steps"] != 0:
        raise ValueError("conditional cost report training execution unsupported")
    if value["mask"]:
        if indices[0] == indices[1] or actions[0] == actions[1] or value["source_inputs"][0] == value["source_inputs"][1]:
            raise ValueError("same action repetition cannot be a conditional utility preference")
        if None in costs or value["preference"] not in ("left", "right"):
            raise ValueError("conditional preference requires reported whole costs")
        left, right = costs
        actual = ("left" if left != right and all(a <= b for a, b in zip(left, right)) else
                  "right" if left != right and all(a >= b for a, b in zip(left, right)) else None)
        if value["preference"] != actual or value["reason"] != "same_conditional_fact_strict_whole_cost_dominance":
            raise ValueError("conditional cost report strict Pareto claim differs")
    elif value["preference"] is not None:
        raise ValueError("masked conditional cost report cannot carry a preference")
    return _reported_only(report_bytes, value)
