"""Caller-bound Query/2 -> own CPU request preparation and raw observations.

The envelope is a result-free binding, not dispatch authority. Rust explicitly
does not revalidate Query/2. Its receipt can record entry to the CPU dispatcher
and original CPU reports, but cannot prove an independently observed process,
native action/witness, final closure, whole cost or strategic utility. No I/O,
launcher, model operation, target, loss or training is implemented here.

Existing Query/2 and semantic factories recheck their original current parents.
Only the legacy own-bootstrap vocabulary is supported. New buffers and JSON
serialization scratch have a finite 4 MiB credit; no action or wall is clipped.
"""

import copy
import hashlib
import json

from . import strategic_verifier_query as query


SCHEMA = "rz-pals-private-strategic-action/1"
SCOPE = "caller_registered_query_to_actual_own_cpu_dispatch"
MAX_BYTES = 4 * 1024 * 1024
MAX_CPU_REQUEST_BYTES = 512 * 1024
MAX_CPU_RESPONSE_BYTES = 1024 * 1024
_FACTORY = object()
_DENIED = {
    "query_revalidated_by_rust": False,
    "native_action_causal_bridge_observed": False,
    "conditional_witness_validated": False,
    "whole_action_cost_observed": False,
    "final_search_closure_observed": False,
    "action_completion_admitted": False,
    "utility_authority": False,
    "target_authority": False,
    "training_authority": False,
    "product_verifier_enabled": False,
    "actual_training_executed": False,
    "backward_executed": False,
    "optimizer_created": False,
    "gpu_used": False,
    "external_teacher_used": False,
}
_BINARY_SCOPES = ("library_caller_verified_digest", "current_exe_path_hash", "linux_loaded_executable_inode")
_ENVELOPE_FIELDS = ("schema", "query_sha256", "catalogue_artifact", "before_result_artifact",
    "prior_ledger_sha256", "action", "remaining", "cpu_request_raw", "cpu_request_artifact", "context_sha256")
_RECEIPT_FIELDS = ("schema", "status", "scope", "context_sha256", "query_sha256", "catalogue_artifact",
    "before_result_artifact", "prior_ledger_sha256", "action", "cpu_request_artifact", "cpu_response_raw",
    "cpu_response_artifact", "cpu_dispatch_attempted", "actual_cpu_nodes", "baseline_present", "after_present",
    "binary_pin_scope", "elapsed_ms", "deadline_exceeded", *_DENIED)


def _fields(value, names):
    # Caller dictionaries may be huge. Check extent before creating any key
    # collection; the names on this module's side are closed constant tuples.
    if type(value) is not dict or len(value) != len(names) or any(name not in value for name in names):
        raise ValueError("strategic CPU action closed fields required")
    return value


def _uint(value, low=0, high=2**64 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("strategic CPU action strict bounded integer required")
    return value


def _text(value, maximum=8192):
    if type(value) is not str or not value or len(value) > maximum:
        raise ValueError("strategic CPU action bounded text required")
    if len(value.encode("utf-8")) > maximum or any(ord(char) < 32 for char in value):
        raise ValueError("strategic CPU action bounded UTF-8 text required")
    return value


def _wire_tree(value, remaining, *, depth=0, nodes=None, raw_field=None):
    """Reserve compact UTF-8 JSON before dumps, including embedded raw JSON.

    Query/2's metadata string bound is 8 KiB. This wire separately embeds a
    bounded original CPU JSON string, so it needs a narrow larger allowance.
    All other scalar/node/depth rules preserve the existing metadata boundary.
    """
    if nodes is None:
        nodes = [65536]
    nodes[0] -= 1
    if nodes[0] < 0 or depth > 32:
        raise ValueError("strategic CPU action metadata extent")
    def charge(size):
        remaining[0] -= size
        if remaining[0] < 0:
            raise ValueError("strategic CPU action serialized byte credit before dumps")
    if value is None:
        charge(4)
    elif type(value) is bool:
        charge(4 if value else 5)
    elif type(value) is int:
        if not -(2**63) <= value <= 2**64 - 1:
            raise ValueError("strategic CPU action integer wire extent")
        charge(len(str(value)))
    elif type(value) is str:
        maximum = {"cpu_request_raw": MAX_CPU_REQUEST_BYTES, "cpu_response_raw": MAX_CPU_RESPONSE_BYTES}.get(raw_field, 8192)
        if len(value) > maximum:
            raise ValueError("strategic CPU action string extent")
        charge(2)
        extent = 0
        for char in value:
            code = ord(char)
            if 0xD800 <= code <= 0xDFFF:
                raise ValueError("strategic CPU action UTF-8 surrogate forbidden")
            size = 1 if code < 128 else 2 if code < 2048 else 3 if code < 65536 else 4
            extent += size
            if extent > maximum:
                raise ValueError("strategic CPU action string extent")
            charge(2 if char in ('"', "\\", "\b", "\f", "\n", "\r", "\t") else 6 if code < 32 else size)
    elif type(value) is list:
        charge(2 + max(0, len(value) - 1))
        for item in value:
            _wire_tree(item, remaining, depth=depth + 1, nodes=nodes)
    elif type(value) is dict:
        charge(2 + len(value) + max(0, len(value) - 1))
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("strategic CPU action string keys required")
            _wire_tree(key, remaining, depth=depth + 1, nodes=nodes)
            _wire_tree(item, remaining, depth=depth + 1, nodes=nodes, raw_field=key)
    else:
        raise ValueError("strategic CPU action float/object wire forbidden")


def _canonical(value, maximum=MAX_BYTES):
    _uint(maximum, 0, MAX_BYTES)
    _wire_tree(value, [maximum])
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > maximum:
        raise ValueError("strategic CPU action canonical byte extent")
    return raw


def _parse(raw):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_BYTES:
        raise ValueError("strategic CPU action bounded immutable raw required")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("strategic CPU action duplicate JSON field")
            value[key] = item
        return value
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("strategic CPU action nonfinite JSON")))
    _wire_tree(value, [MAX_BYTES])
    return value


def _pin(raw, expected):
    _bounded_pin(expected)
    if query.byte_pin(raw) != expected:
        raise ValueError("strategic CPU action independent original byte pin mismatch")


def _bounded_pin(value):
    _fields(value, ("bytes", "sha256"))
    _uint(value["bytes"], 1, MAX_BYTES)
    query._sha(value["sha256"])
    # The shared helper now sees exactly two bounded keys/scalars.
    return query._pin(value)


def _action_body(value):
    _fields(value, ("slot", "task", "semantic_input_sha256", "profile_registration", "baseline_depth",
        "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "max_output_bytes", "budget_bucket"))
    _uint(value["slot"], 0, 7)
    _uint(value["profile_registration"], 0, 7)
    _uint(value["baseline_depth"], 1, 63)
    _uint(value["requested_depth"], value["baseline_depth"] + 1, 64)
    _uint(value["max_nodes_per_check"], 1, 2**32 - 1)
    _uint(value["max_wall_time_ms"], 1, 300000)
    _uint(value["max_output_bytes"], 1024, MAX_CPU_RESPONSE_BYTES)
    _uint(value["budget_bucket"], 0, 16)
    query._sha(value["semantic_input_sha256"])
    if type(value["task"]) is not str or value["task"] not in query._TASK_CAPABILITIES or value["task"] == "lower_selectivity":
        raise ValueError("strategic CPU action unsupported actual task")
    return value


def _prepare(checked, index, request_raw, expected_pin):
    if type(request_raw) is not bytes or not 1 <= len(request_raw) <= MAX_CPU_REQUEST_BYTES:
        raise ValueError("strategic CPU action original request byte extent")
    # Validate a closed pin before serializing caller-owned metadata or querying
    # a capability. Repeated arbitrary metadata cannot spend a hidden buffer.
    _bounded_pin(expected_pin)
    pin_raw = query.canonical(expected_pin, max_bytes=128)
    _pin(request_raw, expected_pin)
    if type(checked) is not query.CheckedStrategicQuery:
        raise ValueError("strategic CPU action exact CheckedStrategicQuery required")
    checked.verify()
    catalogue, assets = checked.catalogue(), checked.raw_assets()
    _uint(index, 0, len(catalogue["actions"]) - 1)
    # Query/2 can describe equal horizons. The actual legacy CPU dispatcher
    # requires baseline 1..63 and a strictly higher requested horizon.
    action = _action_body(catalogue["actions"][index])
    semantics = checked.action_semantic_inputs()
    profile_index = action["profile_registration"]
    registration, capabilities, _ = query._registered(assets["profiles"][profile_index], assets["expected_pins"]["profiles"][profile_index])
    request, common, rules = query._request(request_raw, semantics[index], registration, capabilities,
        checked._parent, checked._index, checked.audit()["parent"])
    if request["task"] != action["task"] or request["max_output_bytes"] != action["max_output_bytes"]:
        raise ValueError("strategic CPU action selected task/output controls mismatch")
    for name in ("baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms"):
        if request[name] != action[name]:
            raise ValueError("strategic CPU action controls changed after catalogue")
    before = query._parse(assets["before_result"])
    remaining = {"steps": before["remaining_steps"], "nodes": before["remaining_nodes"],
        "wall_ms": before["remaining_wall_ms"], "output_bytes": before["remaining_output_bytes"]}
    # Match the actual bridge's independent outer declaration limits. Query/2
    # admission remains unchanged, and an unsupported allowance is not clipped.
    _uint(remaining["steps"], 1, 65_536)
    _uint(remaining["nodes"], 0, (1 << 63) - 1)
    _uint(remaining["wall_ms"], 1, 300_000)
    _uint(remaining["output_bytes"], 1, MAX_BYTES)
    if (remaining["steps"] < 1 or remaining["nodes"] < (0 if action["task"] == "defer" else 2 * request["max_nodes_per_check"])
            or remaining["wall_ms"] < request["max_wall_time_ms"] or remaining["output_bytes"] < 2 * request["max_output_bytes"]):
        raise ValueError("strategic CPU action fixed original remaining allowance")
    body = {"schema": SCHEMA, "query_sha256": checked.sha256,
        "catalogue_artifact": query.byte_pin(assets["catalogue"]), "before_result_artifact": query.byte_pin(assets["before_result"]),
        "prior_ledger_sha256": before["prior_ledger_sha256"], "action": action, "remaining": remaining,
        "cpu_request_raw": request_raw.decode("utf-8"), "cpu_request_artifact": copy.deepcopy(expected_pin)}
    # Both canonical buffers are prepaid. The first is temporary hash material,
    # not a re-rendering of the original CPU request kept in cpu_request_raw.
    available = MAX_BYTES - len(request_raw) - len(pin_raw)
    context_raw = _canonical([SCHEMA, body], available)
    body["context_sha256"] = hashlib.sha256(context_raw).hexdigest()
    # The embedded original request and its escaped outer envelope have separate
    # 512 KiB wire limits. Reserve the final serialized extent before dumps; do
    # not trim or normalize legitimate trailing whitespace in the CPU bytes.
    envelope = _canonical(body, min(MAX_CPU_REQUEST_BYTES, available - len(context_raw)))
    return envelope, request, rules, registration, common, len(request_raw) + len(pin_raw) + len(context_raw) + len(envelope)


class PreparedStrategicCpuAction:
    """Immutable result-free bytes; no callable dispatch or model operation."""
    __slots__ = ("_query", "_index", "_request", "_pin_raw", "_envelope", "_identity")

    def __init__(self, token=None, *, checked=None, index=None, request=None, pin_raw=None, envelope=None):
        if token is not _FACTORY:
            raise ValueError("use prepare_strategic_cpu_action; unchecked action refused")
        for name, value in (("_query", checked), ("_index", index), ("_request", request), ("_pin_raw", pin_raw),
                            ("_envelope", envelope), ("_identity", hashlib.sha256(envelope).hexdigest())):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("prepared strategic CPU action is immutable")

    def _view(self):
        values = _prepare(self._query, self._index, self._request, query._parse(self._pin_raw))
        if values[0] != self._envelope or hashlib.sha256(self._envelope).hexdigest() != self._identity:
            raise ValueError("strategic CPU action prepared bytes changed")
        return values

    def verify(self):
        self._view()
        return self

    @property
    def sha256(self):
        self.verify()
        return self._identity

    def envelope_bytes(self):
        self.verify()
        return self._envelope

    def envelope(self):
        self.verify()
        return _parse(self._envelope)

    def audit(self):
        values = self._view()
        return {"schema": SCHEMA, "scope": "prepared_caller_bound_cpu_request_only", "prepared": True,
            "new_raw_and_serialization_credit_bytes": values[-1], "actual_dispatch_observed": False,
            "independent_loaded_process_observed": False, "actual_utility_groups": 0, **_DENIED}

    def raw_assets(self):
        self.verify()
        return {"cpu_request": self._request, "expected_cpu_request_pin": query._parse(self._pin_raw), "envelope": self._envelope}


def prepare_strategic_cpu_action(*, checked_query, action_index, cpu_request_bytes, expected_cpu_request_pin):
    values = _prepare(checked_query, action_index, cpu_request_bytes, expected_cpu_request_pin)
    return PreparedStrategicCpuAction(_FACTORY, checked=checked_query, index=action_index, request=cpu_request_bytes,
        pin_raw=query.canonical(expected_cpu_request_pin, max_bytes=128), envelope=values[0])


def _report_target(report, rules):
    target = rules["target"]
    if any(report["conditions"][name] != target[name] for name in ("rules_state_sha256", "rules_history_sha256", "board_fen", "legal_moves")):
        raise ValueError("strategic CPU action actual Rules target mismatch")


def _cpu_error(value, request, rules):
    _, legacy, _ = query._modules()
    _fields(value, ("code", "stage", "message", "known_nodes", "failed_check_work", "baseline", "after", "elapsed_ms", "deadline_exceeded"))
    if value["code"] not in ("cpu_task_failed", "cpu_task_cli_failed") or type(value["deadline_exceeded"]) is not bool:
        raise ValueError("strategic CPU action typed CPU error required")
    _text(value["stage"], 256)
    _text(value["message"])
    for name in ("known_nodes", "elapsed_ms"):
        if value[name] is not None:
            _uint(value[name])
    work = value["failed_check_work"]
    if work is not None:
        _fields(work, ("nodes", "quiescence_nodes", "tt_hits"))
        _uint(work["nodes"], 0, request["max_nodes_per_check"])
        _uint(work["quiescence_nodes"], 0, work["nodes"])
        _uint(work["tt_hits"])
    if work is not None and value["after"] is not None:
        # Current Rust failure paths retain either after's actual RawReport or
        # failed-check work, never both representations of the same check.
        raise ValueError("strategic CPU action failed check duplicated after report")
    lower_bound = 0 if work is None else work["nodes"]
    for after, name in ((False, "baseline"), (True, "after")):
        report = value[name]
        if report is not None:
            profile = request["recheck_profile_sha256"] if after and request["task"] == "cross_profile_recheck" else request["cpu_profile_sha256"]
            legacy._report(report, request, profile, after=after)
            _report_target(report, rules)
            lower_bound += report["nodes"]
    if value["known_nodes"] is not None:
        _uint(value["known_nodes"], lower_bound, 2 * request["max_nodes_per_check"])
    # None is genuinely unknown. Rust can also retain known baseline nodes while
    # RawReport validation fails before either report/work is available; that
    # observed lower bound is preserved without inventing absent raw evidence.
    return copy.deepcopy(value)


def _observe(prepared, raw, expected_pin):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_BYTES:
        raise ValueError("strategic CPU action original result byte extent")
    _bounded_pin(expected_pin)
    pin_raw = query.canonical(expected_pin, max_bytes=128)
    _pin(raw, expected_pin)
    if type(prepared) is not PreparedStrategicCpuAction:
        raise ValueError("strategic CPU action exact prepared capability required")
    if len(raw) + len(pin_raw) + len(prepared._request) + len(prepared._envelope) > MAX_BYTES:
        raise ValueError("strategic CPU action aggregate result raw byte extent")
    _, request, rules, registration, _, _ = prepared._view()
    if len(raw) > 2 * request["max_output_bytes"]:
        raise ValueError("strategic CPU action outer result output extent")
    envelope = prepared.envelope()
    result = _parse(raw)
    error = None
    if type(result) is dict and result.get("code") == "strategic_action_failed":
        _fields(result, ("code", "stage", "message", "elapsed_ms", "deadline_exceeded", "cpu_error", "receipt"))
        _text(result["stage"], 256)
        _text(result["message"])
        if type(result["deadline_exceeded"]) is not bool:
            raise ValueError("strategic CPU action error deadline type")
        if result["elapsed_ms"] is not None:
            _uint(result["elapsed_ms"])
        error = copy.deepcopy(result)
        if result["cpu_error"] is not None:
            _cpu_error(result["cpu_error"], request, rules)
        result = result["receipt"]
        if result is None:
            return {"status": "unresolved", "reason": "strategic_failure_without_bound_receipt", "raw_cpu_response": None,
                "raw_cpu_error": error, "cpu_dispatch_function_entered": None, "cpu_reports_complete": False,
                "rust_binary_pin_scope": None, "independent_loaded_process_observed": False,
                "actual_utility_groups": 0, **_DENIED}
    result = _fields(result, _RECEIPT_FIELDS)
    if result["schema"] != SCHEMA or result["scope"] != SCOPE or result["status"] not in ("cpu_response_observed", "cpu_dispatch_failed"):
        raise ValueError("strategic CPU action unsupported Rust receipt scope/status")
    if any(result[name] is not False for name in _DENIED):
        raise ValueError("strategic CPU action receipt cannot expand authority")
    _action_body(result["action"])
    for name in ("catalogue_artifact", "before_result_artifact", "cpu_request_artifact"):
        _bounded_pin(result[name])
    for name in ("context_sha256", "query_sha256", "prior_ledger_sha256"):
        query._sha(result[name])
    for name in ("context_sha256", "query_sha256", "catalogue_artifact", "before_result_artifact", "prior_ledger_sha256", "action", "cpu_request_artifact"):
        if result[name] != envelope[name]:
            raise ValueError("strategic CPU action actual receipt binding mismatch")
    if type(result["cpu_dispatch_attempted"]) is not bool or type(result["deadline_exceeded"]) is not bool:
        raise ValueError("strategic CPU action typed dispatch/deadline flags required")
    _uint(result["elapsed_ms"])
    if result["actual_cpu_nodes"] is not None:
        _uint(result["actual_cpu_nodes"])
    if type(result["baseline_present"]) is not bool or type(result["after_present"]) is not bool:
        raise ValueError("strategic CPU action typed raw report presence required")
    if not result["deadline_exceeded"] and result["elapsed_ms"] > request["max_wall_time_ms"]:
        raise ValueError("strategic CPU action unreported outer deadline overrun")
    if result["binary_pin_scope"] not in _BINARY_SCOPES:
        raise ValueError("strategic CPU action unsupported Rust binary pin scope")
    if result["binary_pin_scope"] == "linux_loaded_executable_inode" and registration["platform"] != "linux":
        raise ValueError("strategic CPU action Rust binary scope/platform mismatch")
    response = None
    complete = False
    if result["status"] == "cpu_dispatch_failed":
        if error is None or error["cpu_error"] is None:
            raise ValueError("strategic CPU action dispatch failure lacks typed inner CPU error")
        if result["cpu_dispatch_attempted"] is not True:
            raise ValueError("strategic CPU action failed receipt lacks dispatcher entry")
        if result["cpu_response_raw"] is not None or result["cpu_response_artifact"] is not None:
            raise ValueError("strategic CPU action failed dispatch invented successful raw response")
        if error is not None and error["cpu_error"] is not None:
            inner = error["cpu_error"]
            if (result["actual_cpu_nodes"] != inner["known_nodes"] or result["baseline_present"] is not (inner["baseline"] is not None)
                    or result["after_present"] is not (inner["after"] is not None)):
                raise ValueError("strategic CPU action failed CPU raw summary mismatch")
    else:
        if result["cpu_dispatch_attempted"] is not True or type(result["cpu_response_raw"]) is not str:
            raise ValueError("strategic CPU action observed response contradicts actual Rust error/dispatch")
        response_raw = result["cpu_response_raw"].encode("utf-8")
        if not 1 <= len(response_raw) <= request["max_output_bytes"]:
            raise ValueError("strategic CPU action original CPU output extent")
        _pin(response_raw, result["cpu_response_artifact"])
        response = query._parse(response_raw)
        query._incomplete_receipt(request, response)
        if (result["actual_cpu_nodes"] != response["nodes"] or result["baseline_present"] is not (response["baseline"] is not None)
                or result["after_present"] is not (response["after"] is not None)):
            raise ValueError("strategic CPU action successful CPU raw summary mismatch")
        if response["elapsed_ms"] > result["elapsed_ms"]:
            raise ValueError("strategic CPU action inner receipt exceeds outer elapsed")
        reports = tuple(report for report in (response["baseline"], response["after"]) if report is not None)
        for report in reports:
            _report_target(report, rules)
        complete = (error is None and not result["deadline_exceeded"] and not response["deadline_exceeded"] and response["status"] == "observed"
            and len(reports) == 2 and all(report["score_scope"] == "completed_iteration" and report["completion"] == "depth_limit"
                and report["completed_depth"] == report["requested_depth"] for report in reports))
    return {"status": "cpu_binding_observation" if complete else "unresolved",
        "reason": "completed_cpu_reports_only" if complete else "failed_partial_deferred_or_unknown_cpu_binding",
        "raw_cpu_response": copy.deepcopy(response), "raw_cpu_error": error,
        "cpu_dispatch_function_entered": result["cpu_dispatch_attempted"], "cpu_reports_complete": complete,
        "rust_binary_pin_scope": result["binary_pin_scope"], "rust_elapsed_ms": result["elapsed_ms"],
        "reported_cpu_nodes": result["actual_cpu_nodes"], "baseline_present": result["baseline_present"], "after_present": result["after_present"],
        "reported_cpu_nodes_are_whole_action_cost": False,
        "rust_deadline_exceeded": result["deadline_exceeded"], "independent_loaded_process_observed": False,
        "caller_closure_observed": False, "actual_utility_groups": 0, **_DENIED}


class CheckedStrategicCpuObservation:
    """Original Rust binding facts only; never a whole action/utility capability."""
    __slots__ = ("_prepared", "_raw", "_pin_raw", "_identity")

    def __init__(self, token=None, *, prepared=None, raw=None, pin_raw=None):
        if token is not _FACTORY:
            raise ValueError("use admit_strategic_cpu_action_receipt; unchecked observation refused")
        for name, value in (("_prepared", prepared), ("_raw", raw), ("_pin_raw", pin_raw),
                            ("_identity", query.digest(SCHEMA, {"prepared": prepared.sha256, "result": query._parse(pin_raw)}))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("strategic CPU observation is immutable")

    def _view(self):
        pin = query._parse(self._pin_raw)
        if query.digest(SCHEMA, {"prepared": self._prepared.sha256, "result": pin}) != self._identity:
            raise ValueError("strategic CPU observation identity changed")
        return _observe(self._prepared, self._raw, pin)

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
        return {"result": self._raw, "expected_result_pin": query._parse(self._pin_raw), "prepared": self._prepared.raw_assets()}


def admit_strategic_cpu_action_receipt(*, prepared_action, receipt_bytes, expected_receipt_pin):
    """Consume original success stdout or typed StrategicError stderr bytes.

    No launch or caller closure is supplied here. Even fully completed reports
    are only CPU binding observations, never actual independent action/utility.
    """
    _observe(prepared_action, receipt_bytes, expected_receipt_pin)
    return CheckedStrategicCpuObservation(_FACTORY, prepared=prepared_action, raw=receipt_bytes,
        pin_raw=query.canonical(expected_receipt_pin, max_bytes=128))
