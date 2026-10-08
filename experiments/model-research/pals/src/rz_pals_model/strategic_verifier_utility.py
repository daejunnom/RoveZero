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

Original raw bytes and independent pins are retained. There is no new closure
wire, action producer, Pareto implementation, no-step loss, target, scorer call,
training, warm state, checkpoint or product V API in this module.
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
