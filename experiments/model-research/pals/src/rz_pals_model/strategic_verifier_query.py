"""Private V query/2: checked action meaning and prior observations, no reward.

The legacy query16, semantic/1, TaskContext and public input/2 are unchanged.
This module does not run a model, checker, launcher or optimizer. Existing
semantic factories are reused; their independently pinned caller observations
remain conditional assurances, not a Python proof of process execution.

The first closed profile vocabulary is own bootstrap PVS, legacy ordering and
no reductions. The two historical profile names are identity namespaces only:
name/hash differences are not neural features or effective alternatives.
SEE, learned values, cross-child stack restoration and private warm state are
unsupported. A later action scorer and actual causal utility factory are
required. In particular this query cannot mint same_witness_cost_dominance/1.

All raw assets, including retained semantic-capability assets and pin metadata,
share a 4 MiB
admission bound. The 8 actions/16 prior observations are never truncated.
Large executable bytes are not copied here: the independent binary artifact
pin is a caller registration declaration, not loaded-image verification.
"""

import copy
from dataclasses import dataclass
import hashlib
import json
import math


QUERY_DOMAIN = "rz-pals-private-v-query/2"
CATALOGUE_SCHEMA = "rz-pals-private-v-action-catalogue/1"
BEFORE_SCHEMA = "rz-pals-private-v-query-before/2"
PROFILE_REGISTRATION_SCHEMA = "rz-pals-private-v-profile-registration/1"
PROFILE_SOURCE_SCHEMA = "rz-pals-private-v-profile-source/1"
LEGACY_PROFILE_SOURCE_SCHEMA = "rz-pals-verifier-utility-checker-source/1"
PRIOR_SCHEMA = "rz-pals-private-v-prior-observation/1"
LEDGER_DOMAIN = "rz-pals-private-v-prior-ledger/1"
FEATURE_DOMAIN = "rz-pals-private-v-action-features/1"
SCOPE = "checked_private_action_query_only"
CALLER_SCOPE = "independently_pinned_caller_registration_and_order_declaration"
MAX_ACTIONS = 8
MAX_PRIOR = 16
MAX_RAW_BYTES = 4 * 1024 * 1024
_FACTORY = object()


def _modules():
    from . import semantic_verifier as semantic
    from . import verifier_producer as legacy
    from . import verifier_feedback as feedback
    return semantic, legacy, feedback


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("strategic query exact fields required")
    return value


def _uint(value, low=0, high=2**64 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("strategic query strict bounded integer required")
    return value


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("strategic query lowercase SHA256 required")
    return value


def _tree(value, *, depth=0, credit=None, byte_credit=None):
    # Bound traversal before canonicalizing caller-owned declarations. Floats
    # are not part of this wire; actual numeric features are derived separately.
    if credit is None:
        credit = [65536]
    if byte_credit is None:
        byte_credit = [MAX_RAW_BYTES]
    def charge(size):
        byte_credit[0] -= size
        if byte_credit[0] < 0:
            raise ValueError("strategic query canonical byte extent before serialization")
    credit[0] -= 1
    if credit[0] < 0 or depth > 32:
        raise ValueError("strategic query metadata extent")
    if value is None:
        charge(4)
    elif type(value) is bool:
        charge(4 if value else 5)
    elif type(value) is int:
        if not -(2**63) <= value <= 2**64 - 1:
            raise ValueError("strategic query integer wire extent")
        charge(len(str(value)))
    elif type(value) is str:
        if len(value) > 8192:
            raise ValueError("strategic query string extent")
        # Count ensure_ascii=False JSON bytes without constructing a JSON or
        # UTF-8 buffer. Repeated references are charged for every occurrence.
        utf8_size = 0
        charge(2)
        for char in value:
            code = ord(char)
            if 0xD800 <= code <= 0xDFFF:
                raise ValueError("strategic query UTF-8 surrogate forbidden")
            size = 1 if code < 0x80 else 2 if code < 0x800 else 3 if code < 0x10000 else 4
            utf8_size += size
            if utf8_size > 8192:
                raise ValueError("strategic query string extent")
            charge(2 if char in ('"', "\\", "\b", "\f", "\n", "\r", "\t") else 6 if code < 0x20 else size)
    elif type(value) is list:
        charge(2 + max(0, len(value) - 1))
        for item in value:
            _tree(item, depth=depth + 1, credit=credit, byte_credit=byte_credit)
    elif type(value) is dict:
        charge(2 + len(value) + max(0, len(value) - 1))
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("strategic query string keys required")
            _tree(key, depth=depth + 1, credit=credit, byte_credit=byte_credit)
            _tree(item, depth=depth + 1, credit=credit, byte_credit=byte_credit)
    else:
        raise ValueError("strategic query float/object wire forbidden")


def canonical(value, *, max_bytes=MAX_RAW_BYTES):
    """Sorted compact UTF-8 JSON; integer precision is not converted to float."""
    _uint(max_bytes, 0, MAX_RAW_BYTES)
    _tree(value, byte_credit=[max_bytes])
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > max_bytes:
        raise ValueError("strategic query canonical byte extent")
    return raw


def digest(domain, value):
    return hashlib.sha256(canonical([domain, value])).hexdigest()


def byte_pin(raw):
    if type(raw) is not bytes or len(raw) > MAX_RAW_BYTES:
        raise ValueError("strategic query bounded immutable bytes required")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _parse(raw):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_RAW_BYTES:
        raise ValueError("strategic query bounded nonempty raw JSON required")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("strategic query duplicate JSON field")
            result[key] = value
        return result
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite strategic query JSON")))
    _tree(value)
    return value


def _pin(value, *, empty=False, binary=False):
    _fields(value, ("bytes", "sha256"))
    _uint(value["bytes"], 0 if empty else 1, (2**63 - 1) if binary else MAX_RAW_BYTES)
    _sha(value["sha256"])
    return value


def _actual(raw, expected, *, empty=False):
    _pin(expected, empty=empty)
    if byte_pin(raw) != expected:
        raise ValueError("strategic query independent actual bytes mismatch")


def _parent(parents, index):
    semantic, _, _ = _modules()
    from .training import ValidatedDataset
    if type(parents) is not ValidatedDataset:
        raise ValueError("strategic query exact strict parent required")
    parents._verify_raw_integrity()
    if parents.frozen_admission is None or parents._frozen_admission_identity is None:
        raise ValueError("strategic query requires actual frozen parent admission")
    _uint(index, 0, len(parents.records) - 1)
    if index not in parents.current_view.current_indices:
        raise ValueError("strategic query requires current parent leaf")
    row = parents.records[index]
    snapshot = row["input"]["snapshot"]
    if snapshot["role"] not in ("proposer", "critic"):
        raise ValueError("strategic query current P/C parent required")
    encoded = parents.encodings.get(row["input"]["sha256"])
    if encoded is None:
        raise ValueError("strategic query actual encoding missing")
    pins = {"parent_input_sha256": row["input"]["sha256"], "current_view_sha256": parents.current_view.sha256,
            "frozen_admission_sha256": parents._frozen_admission_identity, "encoding_sha256": encoded.encoding_sha256}
    for value in pins.values():
        semantic._sha(value)
    return row, pins


def _semantic(checked, parents, index, pins):
    semantic, _, _ = _modules()
    if type(checked) is not semantic.CheckedSemanticInput:
        raise ValueError("strategic query exact CheckedSemanticInput required")
    checked.verify()
    if checked.verify_parent(parents) != index or checked.parent_pins() != pins:
        raise ValueError("strategic query semantic parent/current/frozen binding mismatch")
    return checked.common_query(), checked.rules_receipt()


def _aggregate(catalogue_raw, before_raw, registrations, observations, semantics):
    """Check actual immutable buffers before parsing/copying any raw asset.

    CheckedSemanticInput has no public raw-buffer getter. This narrow read-only
    coupling counts its existing immutable _raws/_pins, then verify() validates
    them. It neither reconstructs nor changes that factory's admission.
    Repeated references are conservatively charged again; there is no dedup
    credit or hidden binary/read allocation in this query-only API.
    """
    semantic, _, _ = _modules()
    if type(registrations) not in (tuple, list) or not 1 <= len(registrations) <= MAX_ACTIONS:
        raise ValueError("strategic query profile registration count")
    if type(observations) not in (tuple, list) or len(observations) > MAX_PRIOR:
        raise ValueError("strategic query prior observation count")
    if type(semantics) not in (tuple, list) or not 1 <= len(semantics) <= MAX_ACTIONS:
        raise ValueError("strategic query action count")
    total = 0
    def charge(raw, nullable=False):
        nonlocal total
        if nullable and raw is None:
            return
        if type(raw) is not bytes:
            raise ValueError("strategic query immutable raw bytes required")
        total += len(raw)
        if total > MAX_RAW_BYTES:
            raise ValueError("strategic query aggregate immutable raw 4 MiB exceeded")
    charge(catalogue_raw)
    charge(before_raw)
    caps = list(semantics)
    for bundle in registrations:
        _fields(bundle, ("registration", "source", "capabilities"))
        for raw in bundle.values():
            charge(raw)
    for bundle in observations:
        _fields(bundle, ("observation", "request", "receipt", "stdout", "stderr", "launch", "semantic_input"))
        for name in ("observation", "request", "receipt", "stdout", "stderr", "launch"):
            charge(bundle[name], nullable=name in ("receipt", "launch"))
        caps.append(bundle["semantic_input"])
    for checked in caps:
        if type(checked) is not semantic.CheckedSemanticInput:
            raise ValueError("strategic query exact semantic capability required before raw reservation")
        if (type(checked._raws) is not tuple or len(checked._raws) != 7
                or {name for name, _ in checked._raws} != {"request", "receipt", "source", "registration", "before_result", "launch_observation", "common_query"}):
            raise ValueError("strategic query semantic raw extent")
        for _, raw in checked._raws:
            charge(raw)
        charge(checked._pins)
    return total


_TASK_CAPABILITIES = {
    "defend_response": "explicit_legal_prefix_and_restricted_response",
    "attack_repair": "explicit_legal_prefix", "widen_responses": "strict_legal_root_subset_to_unrestricted",
    "lower_selectivity": "unavailable_reductions_already_disabled",
    "resume_task": "conditional_owned_completed_iteration_token",
    "cross_profile_recheck": "fresh_own_independent_profile", "defer": "no_cpu_check",
}


@dataclass(frozen=True)
class ProfileFeatures:
    # Closed algorithm/value/ordering meanings, never profile name or digest.
    search_kind: int
    value_kind: int
    ordering_kind: int
    selective_reductions: bool
    tt_entries: int
    quiescence_ply: int
    requested_depth: int


def _registered(bundle, expected):
    _, legacy, _ = _modules()
    _fields(expected, ("registration", "source", "capabilities"))
    for name, raw in bundle.items():
        _actual(raw, expected[name])
    registration = _fields(_parse(bundle["registration"]), ("schema", "binary_artifact", "source_artifact",
        "capabilities_artifact", "profile", "profile_sha256", "platform", "binary_pin_scope", "assurance_scope"))
    source = _fields(_parse(bundle["source"]), ("schema", "profile", "profile_sha256", "search_version",
        "search_conditions", "value_identity", "search_implementation_sha256", "value_semantics_sha256"))
    if (registration["schema"] != PROFILE_REGISTRATION_SCHEMA or source["schema"] not in (PROFILE_SOURCE_SCHEMA, LEGACY_PROFILE_SOURCE_SCHEMA)
            or registration["assurance_scope"] != CALLER_SCOPE):
        raise ValueError("strategic query registered profile scope")
    _pin(registration["binary_artifact"], binary=True)
    if registration["source_artifact"] != expected["source"] or registration["capabilities_artifact"] != expected["capabilities"]:
        raise ValueError("strategic query registered actual source/capability pins")
    for key in ("search_implementation_sha256", "value_semantics_sha256"):
        _sha(source[key])
    profile = _fields(registration["profile"], ("domain", "search", "evaluator", "profile", "tt_entries",
                                              "max_depth", "quiescence_ply", "selective_reductions"))
    if profile["profile"] not in (legacy.PROFILE, legacy.RECHECK_PROFILE):
        raise ValueError("strategic query unknown registered profile vocabulary")
    _uint(profile["tt_entries"], 0, 1048576)
    _uint(profile["max_depth"], 1, 64)
    _uint(profile["quiescence_ply"], 0, 32)
    expected_profile = legacy.cpu_profile(profile["tt_entries"], profile["max_depth"], profile["quiescence_ply"], profile["profile"])
    if profile != expected_profile or profile["selective_reductions"] is not False:
        raise ValueError("strategic query unsupported algorithm/options; SEE is not admitted")
    profile_sha = legacy.profile_sha256(expected_profile)
    expected_conditions = legacy.cpu_conditions_with_ordering(profile["tt_entries"], profile["max_depth"], profile["quiescence_ply"], profile["profile"])
    identity = {"semantics": legacy.CPU_VALUE, "weights_sha256": None, "training": {"kind": "bootstrap"}}
    if (registration["profile_sha256"] != profile_sha or source["profile_sha256"] != profile_sha
            or canonical(source["profile"]) != canonical(expected_profile) or source["search_version"] != legacy.CPU_SEARCH
            or source["search_conditions"] != expected_conditions or source["value_identity"] != identity):
        raise ValueError("strategic query actual profile/source/value namespace mismatch")
    scopes = {"linux": "linux_loaded_executable_inode", "windows": "current_exe_path_hash", "macos": "current_exe_path_hash"}
    if type(registration["platform"]) is not str or scopes.get(registration["platform"]) != registration["binary_pin_scope"]:
        raise ValueError("strategic query registered platform scope")
    cap = _fields(_parse(bundle["capabilities"]), ("schema", "cpu_binary_sha256", "training_private_only",
        "product_verifier_enabled", "actual_training_executed", "backward_executed", "optimizer_created",
        "external_teacher_used", "gpu_used", "cpu_search", "value_identity", "profile", "max_request_bytes",
        "max_response_bytes", "max_wall_time_ms", "max_checks", "max_nodes_per_check", "max_depth",
        "max_prefix_plies", "max_root_moves", "selective_reductions", "resume_kind", "tasks"))
    if (cap["schema"] != legacy.CPU_SCHEMA or cap["cpu_binary_sha256"] != registration["binary_artifact"]["sha256"]
            or cap["training_private_only"] is not True or cap["cpu_search"] != legacy.CPU_SEARCH
            or cap["value_identity"] != identity or cap["profile"] != legacy.PROFILE
            or cap["resume_kind"] != "completed_iteration_same_invocation_only" or cap["tasks"] != _TASK_CAPABILITIES
            or type(cap["max_checks"]) is not int or cap["max_checks"] != 2
            or any(cap[name] is not False for name in ("product_verifier_enabled", "actual_training_executed",
                "backward_executed", "optimizer_created", "external_teacher_used", "gpu_used", "selective_reductions"))):
        raise ValueError("strategic query unsupported actual CPU capabilities")
    for key, maximum in (("max_request_bytes", 512 << 10), ("max_response_bytes", 1 << 20),
                         ("max_wall_time_ms", 300000), ("max_nodes_per_check", 2**32 - 1),
                         ("max_depth", 64), ("max_prefix_plies", 64), ("max_root_moves", 256)):
        _uint(cap[key], 1, maximum)
    if profile["max_depth"] > cap["max_depth"]:
        raise ValueError("strategic query profile exceeds registered horizon")
    features = ProfileFeatures(0, 0, 0, False, profile["tt_entries"], profile["quiescence_ply"], profile["max_depth"])
    return registration, cap, features


def _action(value, checked, registered, parents, index, pins):
    semantic, legacy, _ = _modules()
    _fields(value, ("slot", "task", "semantic_input_sha256", "profile_registration", "baseline_depth",
                    "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "max_output_bytes", "budget_bucket"))
    _uint(value["slot"], 0, MAX_ACTIONS - 1)
    _sha(value["semantic_input_sha256"])
    _uint(value["profile_registration"], 0, len(registered) - 1)
    registration, cap, profile_features = registered[value["profile_registration"]]
    common, rules = _semantic(checked, parents, index, pins)
    if value["semantic_input_sha256"] != checked.sha256:
        raise ValueError("strategic query action semantic identity mismatch")
    _uint(value["baseline_depth"], 1, 64)
    _uint(value["requested_depth"], 1, cap["max_depth"])
    _uint(value["max_nodes_per_check"], 1, cap["max_nodes_per_check"])
    _uint(value["max_wall_time_ms"], 1, cap["max_wall_time_ms"])
    _uint(value["max_output_bytes"], 1024, cap["max_response_bytes"])
    _uint(value["budget_bucket"], 0, 16)
    if (type(value["task"]) is not str or value["task"] not in _TASK_CAPABILITIES or value["task"] == "lower_selectivity"
            or value["baseline_depth"] > value["requested_depth"]):
        raise ValueError("strategic query unsupported actual task")
    for key in ("baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "budget_bucket"):
        if value[key] != common[key]:
            raise ValueError("strategic query exact common controls mismatch")
    if (common["cpu_profile_sha256"] != registration["profile_sha256"]
            or value["requested_depth"] != registration["profile"]["max_depth"]
            or len(common["prefix"]) > cap["max_prefix_plies"] or len(common["root_moves"]) > cap["max_root_moves"]):
        raise ValueError("strategic query semantic profile/capability mismatch")
    task_index = tuple(_TASK_CAPABILITIES).index(value["task"])
    if common["allowed_tasks"] != [value["task"]] or common["eligible_tasks"][task_index] is not True:
        raise ValueError("strategic query task not supported by actual checked branch")
    # Effective equality excludes every digest, profile name, slot and bucket
    # label. Real task kind/configuration/controls/resources remain explicit.
    effective = {"task": value["task"], "question": common["question"], "prefix": common["prefix"],
        "root_moves": common["root_moves"], "claimed_line": common["claimed_line"],
        "root_state": rules["root"]["rules_state_sha256"], "root_history": rules["root"]["rules_history_sha256"],
        "target_state": rules["target"]["rules_state_sha256"], "target_history": rules["target"]["rules_history_sha256"]}
    if value["task"] != "defer":
        effective.update(baseline_depth=value["baseline_depth"], requested_depth=value["requested_depth"],
            nodes=value["max_nodes_per_check"], wall=value["max_wall_time_ms"], output=value["max_output_bytes"],
            tt_entries=profile_features.tt_entries, quiescence_ply=profile_features.quiescence_ply)
    return common, rules, profile_features, canonical(effective), task_index


@dataclass(frozen=True)
class PriorFeatures:
    ordinal: int
    task: int
    state: str
    known: bool
    completed_depths: tuple
    completion: tuple
    pv: tuple
    reported_nodes: int | None
    reported_elapsed_ms: int | None
    caller_elapsed_ms: int | None
    raw_output_bytes: int


@dataclass(frozen=True)
class ActionFeatures:
    task: int
    branch: object
    profile: ProfileFeatures
    baseline_depth: int
    node_budget: int
    wall_ms: int
    output_bytes: int
    prior: tuple

    @property
    def cpu_check(self):
        return self.task != tuple(_TASK_CAPABILITIES).index("defer")

    def numeric_controls(self):
        """Actual controls only; profile name and legacy bucket ID are absent.

        The bucket still binds the old TaskContext/query16 without inventing a
        new meaning for it. Query/2 budget features are the actual H/N/wall and
        output package; changing only a bucket label is an effective duplicate.
        """
        if not self.cpu_check:
            # Keep the registered/raw descriptor for provenance, but no CPU
            # configuration becomes an executed feature or cost for Defer.
            return (0.0,) * 7
        return (self.baseline_depth / 64, self.profile.requested_depth / 64,
                math.log2(self.node_budget + 1) / 64, self.wall_ms / 300000,
                math.log2(self.profile.tt_entries + 1) / 21, self.profile.quiescence_ply / 32,
                self.output_bytes / (1 << 20))


def _request(raw, checked, registration, cap, parents, index, parent_pins):
    _, legacy, _ = _modules()
    value = _fields(_parse(raw), ("schema", "task", "parent_input_sha256", "position_command", "expected_board_fen",
        "rules_state_sha256", "rules_history_sha256", "cpu_binary_sha256", "branch_sha256", "prefix", "root_moves",
        "baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "tt_entries",
        "quiescence_ply", "cpu_profile_sha256", "recheck_profile_sha256", "max_output_bytes", "context_sha256"))
    common, rules = _semantic(checked, parents, index, parent_pins)
    snapshot = parents.records[index]["input"]["snapshot"]
    if value["schema"] != legacy.CPU_SCHEMA or type(value["task"]) is not str or value["task"] not in _TASK_CAPABILITIES or value["task"] == "lower_selectivity":
        raise ValueError("strategic prior unsupported request domain/task")
    if value["context_sha256"] != legacy._hash(legacy.CPU_SCHEMA, {k: v for k, v in value.items() if k != "context_sha256"}):
        raise ValueError("strategic prior request context mismatch")
    for key in ("position_command", "expected_board_fen", "rules_state_sha256", "rules_history_sha256"):
        expected = snapshot["board_fen"] if key == "expected_board_fen" else snapshot[key]
        if value[key] != expected:
            raise ValueError("strategic prior exact parent Rules identity mismatch")
    if (value["parent_input_sha256"] != parent_pins["parent_input_sha256"]
            or value["cpu_binary_sha256"] != registration["binary_artifact"]["sha256"]
            or value["prefix"] != common["prefix"] or value["root_moves"] != common["root_moves"]
            or value["branch_sha256"] != legacy._hash("rz-pals-private-cpu-branch/1", {
                "parent_input_sha256": value["parent_input_sha256"], "prefix": value["prefix"], "root_moves": value["root_moves"]})):
        raise ValueError("strategic prior actual branch/binary binding mismatch")
    for key in ("baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms"):
        if type(value[key]) is not int or value[key] != common[key]:
            raise ValueError("strategic prior request/common resources mismatch")
    if not common["eligible_tasks"][tuple(_TASK_CAPABILITIES).index(value["task"])]:
        raise ValueError("strategic prior unsupported checked task")
    profile = registration["profile"]
    for key in ("tt_entries", "quiescence_ply"):
        if type(value[key]) is not int or value[key] != profile[key]:
            raise ValueError("strategic prior registered actual options mismatch")
    for name, profile_name in (("cpu_profile_sha256", legacy.PROFILE), ("recheck_profile_sha256", legacy.RECHECK_PROFILE)):
        expected = legacy.profile_sha256(legacy.cpu_profile(value["tt_entries"], value["requested_depth"], value["quiescence_ply"], profile_name))
        if value[name] != expected:
            raise ValueError("strategic prior registered actual profile mismatch")
    effective_profile = value["recheck_profile_sha256"] if value["task"] == "cross_profile_recheck" else value["cpu_profile_sha256"]
    if common["cpu_profile_sha256"] != effective_profile or registration["profile_sha256"] != effective_profile:
        raise ValueError("strategic prior effective profile differs from semantic scope")
    _uint(value["baseline_depth"], 1, value["requested_depth"])
    _uint(value["requested_depth"], 1, cap["max_depth"])
    _uint(value["max_nodes_per_check"], 1, cap["max_nodes_per_check"])
    _uint(value["max_wall_time_ms"], 1, cap["max_wall_time_ms"])
    _uint(value["max_output_bytes"], 1024, cap["max_response_bytes"])
    if len(raw) > cap["max_request_bytes"]:
        raise ValueError("strategic prior registered input limit exceeded")
    return value, common, rules


def _incomplete_receipt(request, response):
    """Validate retained typed raw facts without the completed deadline gate.

    This is not a successful gain/admission validator. In particular an honest
    deadline_exceeded receipt remains unknown even when a report was completed
    before the caller deadline. Original bytes are never rewritten for reuse.
    """
    _, legacy, _ = _modules()
    _fields(response, ("schema", "task", "context_sha256", "cpu_binary_sha256", "rules_state_sha256",
        "rules_history_sha256", "board_fen", "branch_sha256", "status", "reason", "baseline", "after",
        "nodes", "elapsed_ms", "resume_kind", "product_verifier_enabled", "deadline_exceeded"))
    for key in ("schema", "task", "context_sha256", "cpu_binary_sha256", "rules_state_sha256", "rules_history_sha256", "branch_sha256"):
        if response[key] != request[key]:
            raise ValueError("strategic prior raw receipt request identity mismatch")
    if (response["board_fen"] != request["expected_board_fen"] or response["product_verifier_enabled"] is not False
            or type(response["deadline_exceeded"]) is not bool):
        raise ValueError("strategic prior raw receipt state/private/deadline mismatch")
    if response["status"] not in ("observed", "deferred", "unavailable"):
        raise ValueError("strategic prior raw receipt status invalid")
    reason = response["reason"]
    if reason is not None and (type(reason) is not str or not 1 <= len(reason) <= 256):
        raise ValueError("strategic prior raw receipt bounded reason required")
    if response["status"] == "unavailable" and reason is None:
        raise ValueError("strategic prior unavailable raw receipt reason missing")
    if (request["task"] == "defer") != (response["status"] == "deferred"):
        raise ValueError("strategic prior raw receipt Defer mismatch")
    if response["resume_kind"] not in (None, "completed_iteration"):
        raise ValueError("strategic prior raw receipt cannot claim stack restoration")
    _uint(response["nodes"], 0, 2 * request["max_nodes_per_check"])
    _uint(response["elapsed_ms"], 0, 2**32 - 1)
    if not response["deadline_exceeded"] and response["elapsed_ms"] > request["max_wall_time_ms"]:
        raise ValueError("strategic prior raw receipt unreported deadline overrun")
    before, after = response["baseline"], response["after"]
    base_profile = request["cpu_profile_sha256"]
    if response["status"] != "observed":
        if after is not None or (before is not None and response["status"] != "unavailable"):
            raise ValueError("strategic prior no-action raw receipt invented report")
        if before is not None:
            legacy._report(before, request, base_profile)
            if before["requested_depth"] != request["baseline_depth"] or response["nodes"] != before["nodes"]:
                raise ValueError("strategic prior unavailable raw baseline accounting mismatch")
        elif response["nodes"] != 0:
            raise ValueError("strategic prior no-action raw receipt invented node cost")
        return
    next_profile = request["recheck_profile_sha256"] if request["task"] == "cross_profile_recheck" else base_profile
    legacy._report(before, request, base_profile)
    legacy._report(after, request, next_profile, after=True)
    if (before["reused_completed_depth"] != 0 or before["requested_depth"] != request["baseline_depth"]
            or after["requested_depth"] != request["requested_depth"]
            or response["nodes"] != before["nodes"] + after["nodes"]):
        raise ValueError("strategic prior raw report depth/node accounting mismatch")
    if request["task"] == "resume_task":
        same = (before["conditions_sha256"] == after["conditions_sha256"]
                and before["profile_sha256"] == after["profile_sha256"]
                and before["white_to_move"] == after["white_to_move"]
                and before["root_restricted"] == after["root_restricted"])
        if (response["resume_kind"] != "completed_iteration" or not same or before["completed_depth"] < 1
                or after["reused_completed_depth"] != before["completed_depth"]):
            raise ValueError("strategic prior raw owned-resume evidence mismatch")
    elif response["resume_kind"] is not None or after["reused_completed_depth"] != 0:
        raise ValueError("strategic prior raw fresh task invented resume state")


def _prior(bundle, expected, parents, index, parent_pins, registrations, ordinal, previous_seal, previous_sequence):
    semantic, legacy, feedback = _modules()
    _fields(expected, ("observation", "request", "receipt", "stdout", "stderr", "launch", "semantic_input_sha256"))
    for name in ("observation", "request", "receipt", "stdout", "stderr", "launch"):
        if bundle[name] is None:
            if expected[name] is not None or name not in ("receipt", "launch"):
                raise ValueError("strategic prior missing raw pin differs")
        else:
            _actual(bundle[name], expected[name], empty=name in ("stdout", "stderr"))
    checked = bundle["semantic_input"]
    if expected["semantic_input_sha256"] != checked.sha256:
        raise ValueError("strategic prior semantic independent pin mismatch")
    event = _fields(_parse(bundle["observation"]), ("schema", "ordinal", "previous_ledger_sha256", "parent_input_sha256",
        "semantic_input_sha256", "profile_registration", "started_sequence", "finished_sequence", "state", "reason",
        "request", "receipt", "stdout", "stderr", "launch", "assurance_scope"))
    _uint(event["ordinal"], 0, MAX_PRIOR - 1)
    _uint(event["started_sequence"], 1)
    _uint(event["finished_sequence"], event["started_sequence"])
    if (event["schema"] != PRIOR_SCHEMA or event["ordinal"] != ordinal
            or event["previous_ledger_sha256"] != previous_seal or event["started_sequence"] <= previous_sequence
            or event["parent_input_sha256"] != parent_pins["parent_input_sha256"]
            or event["semantic_input_sha256"] != checked.sha256 or event["assurance_scope"] != CALLER_SCOPE):
        raise ValueError("strategic prior causal chronology/parent mismatch")
    for name in ("request", "receipt", "stdout", "stderr", "launch"):
        if event[name] != expected[name]:
            raise ValueError("strategic prior raw event binding mismatch")
    _uint(event["profile_registration"], 0, len(registrations) - 1)
    registration, cap, _ = registrations[event["profile_registration"]]
    request, _, rules = _request(bundle["request"], checked, registration, cap, parents, index, parent_pins)
    states = ("completed", "deferred", "unavailable", "partial", "canceled", "failed", "missing")
    if event["state"] not in states or (event["reason"] is not None and (type(event["reason"]) is not str or not 1 <= len(event["reason"]) <= 256)):
        raise ValueError("strategic prior explicit known/unknown state required")
    if len(bundle["stdout"]) + len(bundle["stderr"]) > request["max_output_bytes"]:
        raise ValueError("strategic prior aggregate observed output cap")
    known = event["state"] in ("completed", "deferred")
    if known and (event["reason"] is not None or bundle["receipt"] is None or bundle["launch"] is None
                  or bundle["stdout"] != bundle["receipt"] or bundle["stderr"] != b""):
        raise ValueError("strategic prior known raw evidence missing/dirty")
    response, reports, nodes, elapsed, caller_elapsed = None, (), None, None, None
    if bundle["receipt"] is not None:
        if bundle["stdout"] != bundle["receipt"]:
            raise ValueError("strategic prior receipt differs from retained stdout")
        response = _parse(bundle["receipt"])
        if known:
            # A known prior still requires the original completed deadline and
            # report/conditions/owned-resume validation without relaxation.
            legacy.observed_gain(request, response)
        else:
            _incomplete_receipt(request, response)
        nodes, elapsed = response["nodes"], response["elapsed_ms"]
        reports = tuple(r for r in (response["baseline"], response["after"]) if r is not None)
        target = rules["target"]
        for report in reports:
            conditions = report["conditions"]
            if any(conditions[key] != target[target_key] for key, target_key in (("rules_state_sha256", "rules_state_sha256"),
                    ("rules_history_sha256", "rules_history_sha256"), ("board_fen", "board_fen"), ("legal_moves", "legal_moves"))):
                raise ValueError("strategic prior report actual Rules target mismatch")
        complete = (response["status"] == "observed" and len(reports) == 2 and all(
            report["score_scope"] == "completed_iteration" and report["completion"] == "depth_limit"
            and report["completed_depth"] == report["requested_depth"] for report in reports))
        deferred = response["status"] == "deferred" and not reports and nodes == 0
        if (event["state"] == "completed" and not complete) or (event["state"] == "deferred" and not deferred):
            raise ValueError("strategic prior declared completion differs from actual raw evidence")
        # A complete CPU report can precede a failed/partial/canceled caller
        # capture. Retain both facts without promoting the overall observation.
    if not known and event["reason"] is None:
        raise ValueError("strategic prior unknown reason must be preserved")
    if bundle["launch"] is not None:
        initial_fields = {"schema", "pid", "spawned", "reaped", "pipes_finished",
            "owned_group_absent", "exit_code", "failure", "cleanup_error", "elapsed_ms", "original_deadline_met",
            "loaded_image_before_stdin", "loaded_image_sha256", "loaded_executable", "process_supervision_scope"}
        path_fields = {"binary_before", "binary_after", "source_before", "source_after", "binary_path_stable", "source_path_stable"}
        launch = _parse(bundle["launch"])
        if (type(launch) is not dict or not initial_fields <= set(launch) or set(launch) - initial_fields - path_fields
                or launch["schema"] != feedback.PROCESS_SCHEMA):
            raise ValueError("strategic prior unknown process wire")
        for key in ("spawned", "reaped", "pipes_finished", "owned_group_absent", "original_deadline_met", "loaded_image_before_stdin"):
            if type(launch[key]) is not bool:
                raise ValueError("strategic prior typed caller process flags")
        for key in ("binary_path_stable", "source_path_stable"):
            if key in launch and type(launch[key]) is not bool:
                raise ValueError("strategic prior typed caller path flags")
        if launch["elapsed_ms"] is not None:
            caller_elapsed = _uint(launch["elapsed_ms"], 0, 2**32 - 1)
    if known:
        if registration["platform"] != "linux":
            raise ValueError("strategic query first known prior requires caller Linux owned-group observation")
        launch = feedback._clean_process(bundle["launch"], registration["binary_artifact"], request["max_wall_time_ms"])
        if any(launch[key]["artifact"] != byte_pin(registration["_source_raw"]) for key in ("source_before", "source_after")):
            raise ValueError("strategic prior registered source current-path mismatch")
        if elapsed > launch["elapsed_ms"]:
            raise ValueError("strategic prior total receipt exceeds observed launch elapsed")
    if event["state"] == "deferred" and request["task"] != "defer":
        raise ValueError("strategic prior Defer request mismatch")
    feature = PriorFeatures(ordinal, tuple(_TASK_CAPABILITIES).index(request["task"]), event["state"], known,
        tuple(report["completed_depth"] for report in reports), tuple(report["completion"] for report in reports),
        tuple(tuple(tuple(semantic.move_components(move)) for move in report["pv"]) for report in reports),
        nodes, elapsed, caller_elapsed, len(bundle["stdout"]) + len(bundle["stderr"]))
    raw_pins = {key: expected[key] for key in ("observation", "request", "receipt", "stdout", "stderr", "launch", "semantic_input_sha256")}
    return feature, event, raw_pins


def _validate(parents, index, catalogue_raw, before_raw, profile_bundles, prior_bundles, semantics, expected_pins):
    total = _aggregate(catalogue_raw, before_raw, profile_bundles, prior_bundles, semantics)
    _fields(expected_pins, ("parent", "catalogue", "before_result", "profiles", "prior", "action_semantic_sha256"))
    if (type(expected_pins["profiles"]) is not list or len(expected_pins["profiles"]) != len(profile_bundles)
            or type(expected_pins["prior"]) is not list or len(expected_pins["prior"]) != len(prior_bundles)
            or type(expected_pins["action_semantic_sha256"]) is not list or len(expected_pins["action_semantic_sha256"]) != len(semantics)):
        raise ValueError("strategic query exact independent pin counts")
    # The immutable capability stores a canonical pin declaration too. Charge
    # that real allocation rather than treating it as free metadata.
    total += len(canonical(expected_pins, max_bytes=MAX_RAW_BYTES - total))
    if total > MAX_RAW_BYTES:
        raise ValueError("strategic query aggregate immutable raw 4 MiB exceeded")
    _actual(catalogue_raw, expected_pins["catalogue"])
    _actual(before_raw, expected_pins["before_result"])
    _, parent_pins = _parent(parents, index)
    if expected_pins["parent"] != parent_pins:
        raise ValueError("strategic query independent parent/current/frozen mismatch")
    registered = []
    for bundle, pin in zip(profile_bundles, expected_pins["profiles"]):
        registration, cap, features = _registered(bundle, pin)
        # Internal detached source bytes for the prior caller path observation.
        registration["_source_raw"] = bundle["source"]
        registered.append((registration, cap, features))
    catalogue = _fields(_parse(catalogue_raw), ("schema", "parent", "recipe_id", "actions", "limits", "scope"))
    if (catalogue["schema"] != CATALOGUE_SCHEMA or catalogue["parent"] != parent_pins or catalogue["scope"] != SCOPE
            or type(catalogue["recipe_id"]) is not str or not 1 <= len(catalogue["recipe_id"].encode()) <= 256):
        raise ValueError("strategic query catalogue fixed parent/recipe/scope")
    if catalogue["limits"] != {"actions": MAX_ACTIONS, "prior_observations": MAX_PRIOR, "aggregate_immutable_raw_bytes": MAX_RAW_BYTES}:
        raise ValueError("strategic query exact closed limits")
    if type(catalogue["actions"]) is not list or len(catalogue["actions"]) != len(semantics):
        raise ValueError("strategic query action catalogue extent")
    before = _fields(_parse(before_raw), ("schema", "parent", "catalogue", "prior_ledger_sha256", "decision_ordinal",
        "before_sequence", "remaining_steps", "remaining_nodes", "remaining_wall_ms", "remaining_output_bytes",
        "assurance_scope", "utility_authority", "target_authority", "training_authority"))
    if (before["schema"] != BEFORE_SCHEMA or before["parent"] != parent_pins or before["catalogue"] != expected_pins["catalogue"]
            or before["assurance_scope"] != CALLER_SCOPE or type(before["decision_ordinal"]) is not int
            or before["decision_ordinal"] != len(prior_bundles)
            or any(before[key] is not False for key in ("utility_authority", "target_authority", "training_authority"))):
        raise ValueError("strategic query result-free before declaration mismatch")
    _uint(before["before_sequence"], 1)
    _uint(before["remaining_steps"], 1, 65536)
    _uint(before["remaining_nodes"], 0, 2**63 - 1)
    _uint(before["remaining_wall_ms"], 1, 300000)
    _uint(before["remaining_output_bytes"], 1, MAX_RAW_BYTES)
    prior_features, events, history, seen_executions, seen_launch_meanings = [], [], [], set(), set()
    previous_sequence = 0
    for ordinal, (bundle, pin) in enumerate(zip(prior_bundles, expected_pins["prior"])):
        _fields(pin, ("observation", "request", "receipt", "stdout", "stderr", "launch", "semantic_input_sha256"))
        if pin.get("launch") is not None:
            _pin(pin["launch"])
            # A one-shot physical launch is not a new observation because the
            # request bytes use another valid JSON spelling (or another task).
            execution = (pin["launch"]["bytes"], pin["launch"]["sha256"])
            if execution in seen_executions:
                raise ValueError("strategic query duplicate prior execution observation")
            # Retain original raw bytes/pins, but JSON whitespace/key order is
            # not a new physical observation. Reserve canonical scratch plus
            # the bounded retained 32-byte meaning digests before serialization.
            scratch_limit = MAX_RAW_BYTES - total - 32 * (len(seen_launch_meanings) + 1)
            if scratch_limit < 0:
                raise ValueError("strategic query launch identity scratch byte extent")
            launch_identity = hashlib.sha256(canonical(_parse(bundle["launch"]), max_bytes=scratch_limit)).digest()
            if launch_identity in seen_launch_meanings:
                raise ValueError("strategic query duplicate prior execution observation: launch JSON alias")
            seen_executions.add(execution)
            seen_launch_meanings.add(launch_identity)
        feature, event, raw_pins = _prior(bundle, pin, parents, index, parent_pins, registered, ordinal,
            digest(LEDGER_DOMAIN, history), previous_sequence)
        if events and events[-1]["state"] == "deferred":
            raise ValueError("strategic query episode continues after Defer")
        previous_sequence = event["finished_sequence"]
        prior_features.append(feature)
        events.append(event)
        history.append(raw_pins)
    if before["prior_ledger_sha256"] != digest(LEDGER_DOMAIN, history) or before["before_sequence"] <= previous_sequence:
        raise ValueError("strategic query before/prior causal chronology mismatch")
    if events and events[-1]["state"] == "deferred":
        raise ValueError("strategic query terminated Defer episode has no next selection")
    semantic, _, _ = _modules()
    action_features, effective = [], set()
    for slot, (action, checked, expected_sha) in enumerate(zip(catalogue["actions"], semantics, expected_pins["action_semantic_sha256"])):
        if type(action) is not dict or type(action.get("slot")) is not int or action["slot"] != slot or expected_sha != checked.sha256:
            raise ValueError("strategic query action order/independent semantic pin mismatch")
        common, _, profile, key, task_index = _action(action, checked, registered, parents, index, parent_pins)
        if key in effective:
            raise ValueError("strategic query effective duplicate action; identity names are not alternatives")
        effective.add(key)
        node_reservation = 0 if action["task"] == "defer" else 2 * action["max_nodes_per_check"]
        if (node_reservation > before["remaining_nodes"]
                or action["max_wall_time_ms"] > before["remaining_wall_ms"]
                or 2 * action["max_output_bytes"] > before["remaining_output_bytes"]):
            raise ValueError("strategic query action cannot fit declared pre-dispatch allowance")
        # Existing known-depth header only summarizes admitted FULL completed
        # prior observations at this exact scope. Partial captures remain
        # unknown even if their retained baseline completed an iteration. A
        # missing prior snapshot is unknown, never global novelty or ranking.
        known_depth = 0
        for event, bundle in zip(events, prior_bundles):
            if event["state"] != "completed" or bundle["receipt"] is None:
                continue
            request = _parse(bundle["request"])
            registered_profile = registered[action["profile_registration"]][0]
            if (request["prefix"] != common["prefix"] or request["root_moves"] != common["root_moves"]
                    or any(request[name] != action[name] for name in ("requested_depth", "max_nodes_per_check", "max_wall_time_ms"))
                    or request["tt_entries"] != profile.tt_entries or request["quiescence_ply"] != profile.quiescence_ply
                    or request["cpu_binary_sha256"] != registered_profile["binary_artifact"]["sha256"]):
                continue
            response = _parse(bundle["receipt"])
            for report in (response["baseline"], response["after"]):
                if (report is not None and report["profile_sha256"] == registered_profile["profile_sha256"]
                        and report["score_scope"] == "completed_iteration" and report["completion"] == "depth_limit"
                        and report["completed_depth"] == report["requested_depth"]):
                    known_depth = max(known_depth, report["completed_depth"])
        if common["known_completed_depth"] != known_depth:
            raise ValueError("strategic query declared known coverage differs from actual retained scope")
        action_features.append(ActionFeatures(task_index, semantic.semantic_features(checked), profile,
            action["baseline_depth"], 0 if action["task"] == "defer" else action["max_nodes_per_check"], action["max_wall_time_ms"],
            action["max_output_bytes"], tuple(prior_features)))
    return tuple(action_features), tuple(events), parent_pins, total


class CheckedStrategicQuery:
    """Immutable query-only capability; returned metadata is detached.

    Every use rechecks original strict current parent, exact semantic factories
    and all independent pins. No selector, utility rank, target, task execution
    or trainable/warm-state authority is provided by this class.
    """
    __slots__ = ("_parent", "_index", "_catalogue", "_before", "_profiles", "_prior", "_semantics", "_pins", "_identity")

    def __init__(self, token=None, *, parent=None, index=None, catalogue=None, before=None,
                 profiles=None, prior=None, semantics=None, pins=None):
        if token is not _FACTORY:
            raise ValueError("use admit_strategic_query; unchecked query refused")
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_catalogue", catalogue)
        object.__setattr__(self, "_before", before)
        object.__setattr__(self, "_profiles", tuple(tuple(sorted(bundle.items())) for bundle in profiles))
        object.__setattr__(self, "_prior", tuple(tuple(sorted(bundle.items())) for bundle in prior))
        object.__setattr__(self, "_semantics", tuple(semantics))
        object.__setattr__(self, "_pins", canonical(pins))
        object.__setattr__(self, "_identity", digest(QUERY_DOMAIN, pins))

    def __setattr__(self, name, value):
        raise AttributeError("checked strategic query is immutable")

    def _views(self):
        pins = _parse(self._pins)
        if digest(QUERY_DOMAIN, pins) != self._identity:
            raise ValueError("strategic query capability identity changed")
        return _validate(self._parent, self._index, self._catalogue, self._before,
            tuple(dict(bundle) for bundle in self._profiles), tuple(dict(bundle) for bundle in self._prior), self._semantics, pins)

    def verify(self):
        self._views()
        return self

    @property
    def sha256(self):
        self.verify()
        return self._identity

    def features(self):
        return self._views()[0]

    def catalogue(self):
        self.verify()
        return _parse(self._catalogue)

    def prior_observations(self):
        return copy.deepcopy(self._views()[1])

    def raw_assets(self):
        self.verify()
        return {"catalogue": self._catalogue, "before_result": self._before,
                "profiles": tuple(dict(bundle) for bundle in self._profiles),
                "prior": tuple({key: value for key, value in bundle if key != "semantic_input"} for bundle in self._prior),
                "expected_pins": _parse(self._pins)}

    def action_semantic_inputs(self):
        self.verify()
        return self._semantics

    def audit(self):
        features, events, pins, total = self._views()
        return {"schema": QUERY_DOMAIN, "feature_schema": FEATURE_DOMAIN, "scope": SCOPE,
            "parent": pins, "actions": len(features), "prior_observations": len(events), "immutable_raw_bytes": total,
            "caller_assurance": CALLER_SCOPE, "binary_loaded_image_verified_here": False,
            "profile_registration_has_runtime_authority": False,
            "source_authority": False, "build_authority": False,
            "chronology_is_independent_runtime_clock_proof": False, "prior_snapshot_is_global_knowledge": False,
            "profile_names_are_neural_features": False, "actual_task_execution_authority": False,
            "legacy_budget_bucket_is_new_neural_feature": False,
            "utility_authority": False, "target_authority": False, "training_authority": False,
            "action_scoring_executed": False, "model_executed": False, "private_warm_continuation": False,
            "cpu_stack_restored": False, "product_verifier_enabled": False}


def admit_strategic_query(*, parents, parent_index, catalogue_bytes, before_result_bytes,
                          registered_profiles, prior_observations, action_semantic_inputs, expected_pins):
    """No asset I/O/dispatch: inputs are already immutable caller buffers.

    registered_profiles: 1..8 {registration,source,capabilities} byte bundles.
    prior_observations: 0..16 {observation,request,receipt,stdout,stderr,launch,
    semantic_input} bundles; receipt/launch can be None only for explicit unknown.
    Independently supplied expected_pins binds every raw asset and semantic
    capability. Pins are external caller assurances, never an auto-enrollment.
    """
    _validate(parents, parent_index, catalogue_bytes, before_result_bytes, registered_profiles,
              prior_observations, action_semantic_inputs, expected_pins)
    return CheckedStrategicQuery(_FACTORY, parent=parents, index=parent_index, catalogue=catalogue_bytes,
        before=before_result_bytes, profiles=registered_profiles, prior=prior_observations,
        semantics=action_semantic_inputs, pins=expected_pins)


# A separate /3 report lane. The Rust arena keeps actual native/capture owners;
# these bytes never reconstruct those capabilities or fabricate a legacy receipt.
REPAIR_QUERY_REPORT_SCHEMA = "rz-pals-actual-repair-next-query/3"
REPAIR_QUERY_REPORT_SCOPE = "actual_parent_chronology_and_independent_native_conditional_prior"
CAPTURED_REPAIR_QUERY_REPORT_SCOPE = "captured_nn_input_reinference_and_independent_cpu_condition_reexecution"
_REPAIR_REPORT_FACTORY = object()


def _repair_episode_digest(domain, value):
    """Matches the separately versioned Rust /3 domain-NUL-JSON identity."""
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + canonical(value)).hexdigest()


def _repair_query_report(base_query, raw, expected):
    if type(base_query) is not CheckedStrategicQuery:
        raise ValueError("Repair Query report requires original checked Query/2")
    base_query.verify()
    _actual(raw, expected)
    value = _fields(_parse(raw), ("schema", "assurance_scope", "parent", "source_binding",
        "query_sha256", "prior_ledger_before_sha256", "prior_ledger_sha256", "prior_ordinal",
        "catalogue", "source_input", "source_output", "conditional_fact_sha256", "clock_scope",
        "reported_work_counts", "reported_work_counts_scope", "selection_interval_ns",
        "elapsed_through_admission_ns", "ledger_event_elapsed_ns", "whole_budget_ns", "native_execution_is_same_physical_child",
        "query2_action_semantics_revalidated", "utility_authority", "target_authority",
        "training_authority", "product_authority"))
    assets = base_query.raw_assets()
    pins = assets["expected_pins"]
    before = _parse(assets["before_result"])
    if (value["schema"] != REPAIR_QUERY_REPORT_SCHEMA or value["assurance_scope"] not in (
            REPAIR_QUERY_REPORT_SCOPE, CAPTURED_REPAIR_QUERY_REPORT_SCOPE)
            or value["parent"] != pins["parent"] or value["catalogue"] != pins["catalogue"]
            or any(value[key] is not False for key in ("native_execution_is_same_physical_child",
                "query2_action_semantics_revalidated", "utility_authority", "target_authority",
                "training_authority", "product_authority"))):
        raise ValueError("Repair Query report fixed parent/catalogue/negative authority")
    source = _fields(value["source_binding"], ("query_sha256", "catalogue_artifact", "before_result_artifact",
        "prior_ledger_sha256", "semantic_input_sha256", "semantic_context_sha256",
        "semantic_branch_meaning_sha256", "semantic_before_result_anchor_sha256", "cpu_request_artifact"))
    if (source["query_sha256"] != base_query.sha256 or source["catalogue_artifact"] != pins["catalogue"]
            or source["before_result_artifact"] != pins["before_result"]
            or source["prior_ledger_sha256"] != before["prior_ledger_sha256"]
            or value["prior_ledger_before_sha256"] != source["prior_ledger_sha256"]
            or source["semantic_input_sha256"] not in pins["action_semantic_sha256"]):
        raise ValueError("Repair Query report original question/prior/semantic binding differs")
    matched = [cap for cap in base_query.action_semantic_inputs() if cap.sha256 == source["semantic_input_sha256"]]
    if not matched:
        raise ValueError("Repair Query report original semantic action missing")
    semantic_receipt = matched[0].rules_receipt()
    if any(cap.rules_receipt() != semantic_receipt for cap in matched[1:]):
        raise ValueError("Repair Query report repeated semantic identity differs")
    if (source["semantic_context_sha256"] != semantic_receipt["context_sha256"]
            or source["semantic_branch_meaning_sha256"] != semantic_receipt["branch_meaning_sha256"]
            or source["semantic_before_result_anchor_sha256"] != semantic_receipt["before_result_anchor_sha256"]):
        raise ValueError("Repair Query report original semantic context/branch/anchor differs")
    for key in ("query_sha256", "prior_ledger_before_sha256", "prior_ledger_sha256", "conditional_fact_sha256"):
        _sha(value[key])
    for key in ("semantic_input_sha256", "semantic_context_sha256", "semantic_branch_meaning_sha256",
            "semantic_before_result_anchor_sha256", "prior_ledger_sha256", "query_sha256"):
        _sha(source[key])
    for key in ("source_input", "source_output", "catalogue"):
        _pin(value[key])
    _pin(source["cpu_request_artifact"])
    # This consumer binds the first transition from Query/2. Later /3 transitions
    # stay in the live Rust episode; importing another JSON never extends it.
    if _uint(value["prior_ordinal"], 0, 15) != 0:
        raise ValueError("Repair Query report imported ledger cannot extend live episode")
    elapsed = _uint(value["elapsed_through_admission_ns"], 0)
    ledger_elapsed = _uint(value["ledger_event_elapsed_ns"], 0)
    whole = _uint(value["whole_budget_ns"], 1, 300_000_000_000)
    if elapsed >= whole or ledger_elapsed > elapsed:
        raise ValueError("Repair Query report original whole window expired")
    if value["clock_scope"] == "preparation_only":
        if value["selection_interval_ns"] is not None:
            raise ValueError("Repair Query report preparation scope invented selection")
    elif value["clock_scope"] == "before_selection_callback":
        interval = value["selection_interval_ns"]
        if type(interval) is not list or len(interval) != 2:
            raise ValueError("Repair Query report actual selection interval required")
        if _uint(interval[0]) != 0 or _uint(interval[1]) > elapsed:
            raise ValueError("Repair Query report selection/admission order")
    else:
        raise ValueError("Repair Query report unsupported clock scope")
    work_scope = ("child_report_correlated_to_independent_native_not_same_physical_execution"
        if value["assurance_scope"] == REPAIR_QUERY_REPORT_SCOPE
        else "child_report_separate_from_captured_nn_and_independent_cpu_executions")
    if value["reported_work_counts_scope"] != work_scope:
        raise ValueError("Repair Query report child work provenance")
    counts = value["reported_work_counts"]
    if counts is not None:
        if type(counts) is not list or len(counts) != 2:
            raise ValueError("Repair Query report known/unknown work counters")
        for count in counts:
            _uint(count)
    event = {"ordinal": value["prior_ordinal"], "source_query": source["query_sha256"],
        "previous_ledger": value["prior_ledger_before_sha256"], "source_input": value["source_input"],
        "source_output": value["source_output"], "conditional_fact_sha256": value["conditional_fact_sha256"],
        "ledger_event_elapsed_ns": ledger_elapsed}
    if _repair_episode_digest("rz-pals-actual-repair-prior-ledger/3", event) != value["prior_ledger_sha256"]:
        raise ValueError("Repair Query report event/ledger identity differs")
    identity = {"schema": REPAIR_QUERY_REPORT_SCHEMA, "parent": value["parent"],
        "source_query_sha256": source["query_sha256"], "prior_ledger_sha256": value["prior_ledger_sha256"],
        "decision_ordinal": value["prior_ordinal"] + 1, "catalogue": value["catalogue"]}
    if _repair_episode_digest(REPAIR_QUERY_REPORT_SCHEMA, identity) != value["query_sha256"]:
        raise ValueError("Repair Query report next question identity differs")
    return value


class CheckedRepairQueryReport:
    """Immutable checked report, explicitly distinct from a live Rust Query."""
    __slots__ = ("_base", "_raw", "_expected")

    def __init__(self, token=None, *, base=None, raw=None, expected=None):
        if token is not _REPAIR_REPORT_FACTORY:
            raise ValueError("use admit_repair_query_report; native capability cannot be imported")
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_expected", canonical(expected))

    def __setattr__(self, name, value):
        raise AttributeError("Repair Query report is immutable")

    def verify(self):
        _repair_query_report(self._base, self._raw, _parse(self._expected))
        return self

    @property
    def sha256(self):
        return _repair_query_report(self._base, self._raw, _parse(self._expected))["query_sha256"]

    def report(self):
        return copy.deepcopy(_repair_query_report(self._base, self._raw, _parse(self._expected)))

    def original_action_features(self):
        self.verify()
        return self._base.features()

    def audit(self):
        value = self.report()
        return {"schema": REPAIR_QUERY_REPORT_SCHEMA, "query_sha256": value["query_sha256"],
            "scope": "checked_pinned_caller_report_only_live_rust_owners_not_imported",
            "live_native_capability": False, "runtime_clock_proof": False,
            "legacy_cpu_receipt_created": False, "utility_authority": False,
            "target_authority": False, "training_authority": False, "product_authority": False}


def admit_repair_query_report(*, original_query, report_bytes, expected_report_pin):
    """Read the separately versioned /3 report alongside the original Query/2.

    Exact pins and causal metadata do not attest that Rust, a child or a native
    witness executed. Only the live Rust factory owns that admission capability.
    """
    _repair_query_report(original_query, report_bytes, expected_report_pin)
    return CheckedRepairQueryReport(_REPAIR_REPORT_FACTORY, base=original_query,
        raw=report_bytes, expected=expected_report_pin)
