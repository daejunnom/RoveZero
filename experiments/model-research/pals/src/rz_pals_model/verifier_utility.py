"""Declared-snapshot-relative fixed-obligation V utility, never a legacy rank.

The independent caller owns actual source/binary registration, result-before
plan publication and child launch observations. Hash-shaped declarations alone
prove none of those events. Only the existing checked semantic input capability
can open positive admission. Raw failed/partial observations remain preserved.
No CP, mate, nodes, V agreement, global novelty or optimality enters the target.
No model executes at import; the optional frozen loss helper has no update path.
"""
import copy
from dataclasses import dataclass
import math
import time

CRITERION_SCHEMA = "rz-pals-verifier-fixed-obligation-criterion/1"
PLAN_SCHEMA = "rz-pals-verifier-fixed-obligation-plan/1"
KNOWLEDGE_SCHEMA = "rz-pals-verifier-fixed-obligation-knowledge/1"
REGISTRATION_SCHEMA = "rz-pals-verifier-utility-checker-registration/1"
SOURCE_SCHEMA = "rz-pals-verifier-utility-checker-source/1"
LAUNCH_SCHEMA = "rz-pals-verifier-utility-launch/1"
ADMISSION_SCHEMA = "rz-pals-verifier-fixed-obligation-admission/1"
PREPARATION_SCHEMA = "rz-pals-verifier-fixed-obligation-frozen-preparation/1"
SCOPE = "declared_snapshot_relative_completed_iteration_coverage_only"
QUESTION = "unrestricted_completed_iteration_obligation"
ACTIONS = ("resume_task", "defer")
MAX_BYTES = 128 * 1024 * 1024
MAX_KNOWN = 256
_ADMISSION_CAPABILITY = object()


def _modules():
    from . import comparative_training as candidate
    from . import verifier_producer as legacy
    from . import training
    return candidate, legacy, training


def _fields(value, names, message):
    return _modules()[2]._fields(value, names, message)


def _sha(value):
    return _modules()[2]._sha(value)


def _uint(value, maximum=(1 << 64) - 1):
    return _modules()[2]._uint(value, maximum)


def _json(raw):
    return _modules()[0]._json(raw)


def canonical(value):
    return _modules()[0].canonical_wire(value)


def byte_pin(raw):
    return _modules()[0].byte_pin(raw)


def digest(value):
    return _modules()[0].wire_digest(value)


def _actual(raw, pin, *, empty=False):
    _modules()[0]._actual(raw, pin, MAX_BYTES, empty=empty)
    return raw


def _registered(registration_raw, source_raw, binary_raw, capabilities_raw, pins, criterion):
    _, legacy, _ = _modules()
    for name, raw in (("registration", registration_raw), ("source", source_raw),
                      ("binary", binary_raw), ("capabilities", capabilities_raw)):
        _actual(raw, pins[name])
    registration = _fields(_json(registration_raw),
         ("schema", "binary_artifact", "source_artifact", "capabilities_artifact", "profile", "profile_sha256",
          "recheck_profile_sha256", "platform", "binary_pin_scope", "search_implementation_sha256",
          "value_semantics_sha256"), "utility checker registration")
    source = _fields(_json(source_raw),
         ("schema", "profile", "profile_sha256", "search_version", "search_conditions", "value_identity",
          "search_implementation_sha256", "value_semantics_sha256"), "utility actual registered source")
    if registration["schema"] != REGISTRATION_SCHEMA or source["schema"] != SOURCE_SCHEMA:
        raise ValueError("utility registration/source schema")
    for name in ("search_implementation_sha256", "value_semantics_sha256"):
        _sha(registration[name])
        if source[name] != registration[name]:
            raise ValueError("utility registered implementation/semantics mismatch")
    for name, raw in (("binary", binary_raw), ("source", source_raw), ("capabilities", capabilities_raw)):
        if registration[name + "_artifact"] != byte_pin(raw):
            raise ValueError("utility registered actual byte artifact mismatch")
    profile = criterion["profile"]
    h = criterion["obligation"]["required_horizon"]
    expected = legacy.cpu_profile(profile["tt_entries"], h, profile["quiescence_ply"])
    expected_recheck = legacy.profile_sha256(legacy.cpu_profile(profile["tt_entries"], h, profile["quiescence_ply"], legacy.RECHECK_PROFILE))
    profile_sha = legacy.profile_sha256(expected)
    search_conditions = legacy.CPU_CONDITIONS + f";profile={legacy.PROFILE};max_depth={h};q_plies={profile['quiescence_ply']};tt_entries={profile['tt_entries']}"
    identity = {"semantics": legacy.CPU_VALUE, "weights_sha256": None, "training": {"kind": "bootstrap"}}
    if (canonical(profile) != canonical(expected) or canonical(registration["profile"]) != canonical(expected)
            or canonical(source["profile"]) != canonical(expected)
            or registration["profile_sha256"] != profile_sha or source["profile_sha256"] != profile_sha
            or registration["recheck_profile_sha256"] != expected_recheck
            or source["search_version"] != legacy.CPU_SEARCH or source["search_conditions"] != search_conditions
            or source["value_identity"] != identity):
        raise ValueError("utility exact registered profile/config/value namespace mismatch")
    platform, assurance = registration["platform"], registration["binary_pin_scope"]
    if ((platform == "linux" and assurance != "linux_loaded_executable_inode")
            or (platform in ("windows", "macos") and assurance != "current_exe_path_hash")
            or platform not in ("linux", "windows", "macos")):
        raise ValueError("utility platform assurance scope")
    capabilities = _json(capabilities_raw)
    # The actual --capabilities CLI adds its self-image pin. The static Rust
    # capabilities() descriptor alone is not the independently pinned CLI blob.
    names = ("schema", "cpu_binary_sha256", "training_private_only", "product_verifier_enabled", "actual_training_executed",
             "backward_executed", "optimizer_created", "external_teacher_used", "gpu_used", "cpu_search",
             "value_identity", "profile", "max_request_bytes", "max_response_bytes", "max_wall_time_ms",
             "max_checks", "max_nodes_per_check", "max_depth", "max_prefix_plies", "max_root_moves",
             "selective_reductions", "resume_kind", "tasks")
    _fields(capabilities, names, "registered actual legacy capabilities")
    if (capabilities["schema"] != legacy.CPU_SCHEMA or capabilities["cpu_binary_sha256"] != pins["binary"]["sha256"]
            or capabilities["training_private_only"] is not True
            or any(capabilities[name] is not False for name in ("product_verifier_enabled", "actual_training_executed",
                 "backward_executed", "optimizer_created", "external_teacher_used", "gpu_used", "selective_reductions"))
            or capabilities["cpu_search"] != legacy.CPU_SEARCH or capabilities["value_identity"] != identity
            or capabilities["profile"] != legacy.PROFILE or type(capabilities["max_checks"]) is not int or capabilities["max_checks"] != 2
            or capabilities["resume_kind"] != "completed_iteration_same_invocation_only"):
        raise ValueError("utility actual capability/private boundary mismatch")
    tasks = {"defend_response": "explicit_legal_prefix_and_restricted_response", "attack_repair": "explicit_legal_prefix",
             "widen_responses": "strict_legal_root_subset_to_unrestricted", "lower_selectivity": "unavailable_reductions_already_disabled",
             "resume_task": "conditional_owned_completed_iteration_token", "cross_profile_recheck": "fresh_own_independent_profile",
             "defer": "no_cpu_check"}
    if capabilities["tasks"] != tasks:
        raise ValueError("utility actual task capabilities mismatch")
    for name, maximum in (("max_request_bytes", 512 * 1024), ("max_response_bytes", 1024 * 1024),
                          ("max_wall_time_ms", 300000), ("max_nodes_per_check", (1 << 32) - 1),
                          ("max_depth", 64), ("max_prefix_plies", 64), ("max_root_moves", 256)):
        if not 1 <= _uint(capabilities[name], maximum):
            raise ValueError("utility finite registered capabilities")
    return registration, capabilities


def _criterion(raw):
    criterion = _fields(_json(raw),
       ("schema", "recipe_id", "scope", "question_id", "parent", "current", "semantic_input_sha256",
        "semantic_context_sha256", "branch_sha256", "profile", "obligation", "budget_bucket", "eligible_tasks", "max_output_bytes"),
       "fixed utility criterion")
    _modules()[2]._identity(criterion["recipe_id"])
    if (criterion["schema"] != CRITERION_SCHEMA or criterion["scope"] != SCOPE or criterion["question_id"] != QUESTION
            or criterion["eligible_tasks"] != list(ACTIONS)):
        raise ValueError("unsupported fixed utility criterion/question/action scope")
    for name in ("semantic_input_sha256", "semantic_context_sha256", "branch_sha256"):
        _sha(criterion[name])
    _uint(criterion["budget_bucket"], 16)
    if not 1024 <= _uint(criterion["max_output_bytes"], 1024 * 1024):
        raise ValueError("utility fixed output byte allowance")
    obligation = _fields(criterion["obligation"], ("conditions", "baseline_depth", "required_horizon", "score_scope", "completion"),
                         "fixed completed-iteration obligation")
    h = _uint(obligation["required_horizon"], 64)
    baseline = _uint(obligation["baseline_depth"], 63)
    if not 1 <= baseline < h or obligation["score_scope"] != "completed_iteration" or obligation["completion"] != "depth_limit":
        raise ValueError("fixed obligation horizon/completion")
    profile = _fields(criterion["profile"], ("domain", "search", "evaluator", "profile", "tt_entries", "max_depth",
                                            "quiescence_ply", "selective_reductions"), "fixed utility profile")
    _uint(profile["tt_entries"], 1048576)
    _uint(profile["quiescence_ply"], 32)
    if _uint(profile["max_depth"], 64) != h or profile["selective_reductions"] is not False:
        raise ValueError("utility profile exact current horizon/selectivity")
    return criterion


def _parent(parents, parent_artifacts, criterion):
    _, _, training = _modules()
    from . import frozen_producer as frozen
    # Consume the existing strict admission and DG05 view, without candidate
    # comparison's two-legal-move restriction or a second chain selector.
    names = ("receipt.json", "producer-roster.json", "producer-envelope.json", "producer-prepared.jsonl",
             "inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl")
    _fields(parent_artifacts, names, "strict utility parent artifacts")
    admission = parents.frozen_admission
    for name in names:
        _actual(parent_artifacts[name], admission["receipt"] if name == "receipt.json" else admission["artifacts"][name])
    roster = frozen.load_roster(parent_artifacts["producer-roster.json"])
    envelope = frozen.load_envelope(parent_artifacts["producer-envelope.json"])
    expected_parent = {"receipt_artifact": admission["receipt"], "raw_dataset_sha256": admission["raw_dataset_sha256"],
                       "split_sha256": admission["split_sha256"], "current_view_sha256": parents.current_view.sha256,
                       "producer_roster_sha256": roster["sha256"], "producer_envelope_sha256": envelope["sha256"]}
    if canonical(criterion["parent"]) != canonical(expected_parent):
        raise ValueError("utility actual strict parent pins mismatch")
    current = criterion["current"]
    identity = _sha(current.get("input_sha256")) if isinstance(current, dict) else None
    matching = [entry for entry in (parents.frozen_admission or {}).get("inputs", [])
                if entry["binding"]["input_sha256"] == identity]
    if len(matching) != 1:
        raise ValueError("utility current parent lacks strict prepared binding")
    entry = matching[0]
    journals = [_json(line) for line in parent_artifacts["producer-prepared.jsonl"].splitlines()]
    journal = next((value["prepared"] for value in journals if value["sha256"] == entry["prepared_evidence_sha256"]), None)
    if journal is None:
        raise ValueError("utility current parent missing actual prepared journal")
    for pin_name, name in (("input_json", "inputs.jsonl"), ("tensor_sidecar_json", "native-inputs.jsonl"),
                          ("lineage_json", "input-lineage.jsonl")):
        if not any(canonical(byte_pin(raw)) == canonical(journal[pin_name]) for raw in parent_artifacts[name].splitlines()):
            raise ValueError("utility missing exact prepared input/tensor/lineage bytes")
    selected = [index for index in parents.current_view.current_indices if parents.records[index]["input"]["sha256"] == identity]
    if len(selected) != 1:
        raise ValueError("utility stale current parent leaf")
    row = parents.records[selected[0]]
    expected = {"input_sha256": identity, "label_sha256": training.label_digest(row),
                **{name: row["input"]["snapshot"][name] for name in ("game_id", "role", "rules_state_sha256", "rules_history_sha256",
                    "encoding_sha256", "source", "frozen_epoch", "input_revision", "legal_moves")},
                "side_to_move": "white" if row["input"]["snapshot"]["white_to_move"] else "black"}
    if canonical(current) != canonical(expected):
        raise ValueError("utility current parent identity/label mismatch")
    return selected[0], row


def _conditions(criterion, row, registration):
    _, legacy, _ = _modules()
    conditions = _fields(criterion["obligation"]["conditions"],
       ("schema", "rules_state_sha256", "rules_history_sha256", "board_fen", "profile_sha256", "value_identity",
        "search_version", "search_conditions", "quiescence_ply", "resource_policy", "legal_moves", "root_moves",
        "root_order", "root_order_sha256", "root_selection", "white_to_move"), "fixed actual obligation conditions")
    snapshot = row["input"]["snapshot"]
    policy = _fields(conditions["resource_policy"],
                     ("max_wall_time_ms", "max_nodes_per_check", "max_checks", "search_deadline_reserve_ms"), "utility fixed resources")
    if (not 1 <= _uint(policy["max_wall_time_ms"], 300000) or not 1 <= _uint(policy["max_nodes_per_check"], (1 << 32) - 1)
            or type(policy["max_checks"]) is not int or policy["max_checks"] != 2
            or _uint(policy["search_deadline_reserve_ms"], 1000) != min(policy["max_wall_time_ms"] // 10, 1000)):
        raise ValueError("utility fixed original resource policy")
    h, p = criterion["obligation"]["required_horizon"], criterion["profile"]
    expected = {"schema": "rz-pals-private-cpu-conditions/1", "rules_state_sha256": snapshot["rules_state_sha256"],
                "rules_history_sha256": snapshot["rules_history_sha256"], "board_fen": snapshot["board_fen"],
                "profile_sha256": registration["profile_sha256"],
                "value_identity": {"semantics": legacy.CPU_VALUE, "weights_sha256": None, "training": {"kind": "bootstrap"}},
                "search_version": legacy.CPU_SEARCH,
                "search_conditions": legacy.CPU_CONDITIONS + f";profile={legacy.PROFILE};max_depth={h};q_plies={p['quiescence_ply']};tt_entries={p['tt_entries']}",
                "quiescence_ply": p["quiescence_ply"], "resource_policy": policy,
                "legal_moves": snapshot["legal_moves"], "root_moves": None, "root_order": snapshot["legal_moves"],
                "root_order_sha256": legacy._hash("rz-pals-private-cpu-root-order/1", snapshot["legal_moves"]),
                "root_selection": "unrestricted", "white_to_move": snapshot["white_to_move"]}
    if canonical(conditions) != canonical(expected):
        raise ValueError("utility exact Rules/order/perspective/profile/resources mismatch")
    return conditions


def _request(raw, task, row, criterion, registration, capabilities, binary_pin):
    _, legacy, training = _modules()
    request = _fields(_json(raw),
       ("schema", "task", "parent_input_sha256", "position_command", "expected_board_fen", "rules_state_sha256",
        "rules_history_sha256", "cpu_binary_sha256", "branch_sha256", "prefix", "root_moves", "baseline_depth",
        "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "max_output_bytes", "tt_entries",
        "quiescence_ply", "cpu_profile_sha256", "recheck_profile_sha256", "context_sha256"), "actual legacy utility request")
    without = {name: value for name, value in request.items() if name != "context_sha256"}
    snapshot = row["input"]["snapshot"]
    policy = criterion["obligation"]["conditions"]["resource_policy"]
    expected = {"schema": legacy.CPU_SCHEMA, "task": task, "parent_input_sha256": row["input"]["sha256"],
                "position_command": snapshot["position_command"], "expected_board_fen": snapshot["board_fen"],
                "rules_state_sha256": snapshot["rules_state_sha256"], "rules_history_sha256": snapshot["rules_history_sha256"],
                "cpu_binary_sha256": binary_pin["sha256"], "branch_sha256": criterion["branch_sha256"],
                "prefix": [], "root_moves": [], "baseline_depth": criterion["obligation"]["baseline_depth"],
                "requested_depth": criterion["obligation"]["required_horizon"],
                "max_nodes_per_check": policy["max_nodes_per_check"], "max_wall_time_ms": policy["max_wall_time_ms"],
                "max_output_bytes": criterion["max_output_bytes"], "tt_entries": criterion["profile"]["tt_entries"],
                "quiescence_ply": criterion["profile"]["quiescence_ply"], "cpu_profile_sha256": registration["profile_sha256"],
                "recheck_profile_sha256": registration["recheck_profile_sha256"]}
    if canonical(without) != canonical(expected) or request["context_sha256"] != digest([legacy.CPU_SCHEMA, without]):
        raise ValueError("utility actual request differs from fixed context/controls")
    if criterion["branch_sha256"] != legacy._hash("rz-pals-private-cpu-branch/1", {"parent_input_sha256": row["input"]["sha256"], "prefix": [], "root_moves": []}):
        raise ValueError("utility exact empty branch digest")
    for name in ("max_nodes_per_check", "max_wall_time_ms", "max_output_bytes"):
        cap = capabilities["max_response_bytes"] if name == "max_output_bytes" else capabilities[name]
        if not 1 <= _uint(request[name], cap) or (name == "max_output_bytes" and request[name] < 1024):
            raise ValueError("utility request exceeds registered capability")
    if len(raw) > capabilities["max_request_bytes"] or request["requested_depth"] > capabilities["max_depth"]:
        raise ValueError("utility finite request exceeds registered capability")
    return request


def _execution(buffers, pins, *, execution_id, request, registration, checker_pins, plan_sha, prior=False):
    _, legacy, _ = _modules()
    _fields(buffers, ("request", "receipt", "stderr", "launch_observation"), "utility actual execution buffers")
    _fields(pins, buffers.keys(), "independent utility execution pins")
    for name, raw in buffers.items():
        if raw is None:
            if name != "receipt" or pins[name] is not None:
                raise ValueError("utility missing raw byte pin")
        else:
            _actual(raw, pins[name], empty=name == "stderr")
    launch = _fields(_json(buffers["launch_observation"]),
       ("schema", "execution_id", "registration_sha256", "binary_sha256", "platform", "binary_pin_scope", "before_result_plan_sha256",
        "request", "receipt", "stderr", "assurance_scope", "anchor_durable_before_spawn", "criterion_fixed_before_spawn",
        "spawned", "reaped", "pipes_finished", "exit_code", "elapsed_ms", "timed_out", "canceled",
        "source_pre_artifact", "source_post_artifact", "binary_pre_artifact", "binary_post_artifact",
        "process_supervision_scope", "owned_group_absent"), "independent utility launch observation")
    if (launch["schema"] != LAUNCH_SCHEMA or launch["execution_id"] != execution_id
            or launch["registration_sha256"] != checker_pins["registration"]["sha256"]
            or launch["binary_sha256"] != checker_pins["binary"]["sha256"]
            or launch["platform"] != registration["platform"] or launch["binary_pin_scope"] != registration["binary_pin_scope"]
            or launch["assurance_scope"] != "independently_pinned_caller_observation"
            or any(canonical(launch[name]) != canonical(pins[name]) for name in ("request", "receipt", "stderr"))):
        raise ValueError("utility independent launch identity/bytes mismatch")
    _sha(launch["before_result_plan_sha256"])
    if not prior and launch["before_result_plan_sha256"] != plan_sha:
        raise ValueError("utility launch changed result-before plan")
    for name in ("anchor_durable_before_spawn", "criterion_fixed_before_spawn", "spawned", "reaped", "pipes_finished", "timed_out", "canceled"):
        if type(launch[name]) is not bool:
            raise ValueError("utility launch observation boolean")
    elapsed = _uint(launch["elapsed_ms"], request["max_wall_time_ms"])
    if launch["exit_code"] is not None and (type(launch["exit_code"]) is not int or not -(1 << 31) <= launch["exit_code"] < (1 << 31)):
        raise ValueError("utility actual child exit code")
    scope = launch["process_supervision_scope"]
    if registration["platform"] == "windows":
        if scope != "windows_direct_child_only_no_job_object" or launch["owned_group_absent"] is not None:
            raise ValueError("utility Windows supervision overclaim")
    elif scope != "posix_owned_process_group" or type(launch["owned_group_absent"]) is not bool:
        raise ValueError("utility POSIX supervision scope")
    stable = all(canonical(launch[name + suffix]) == canonical(checker_pins[name]) for name in ("source", "binary")
                 for suffix in ("_pre_artifact", "_post_artifact"))
    good = (stable and launch["anchor_durable_before_spawn"] and launch["criterion_fixed_before_spawn"]
            and launch["spawned"] and launch["reaped"] and launch["pipes_finished"] and launch["exit_code"] == 0
            and not launch["timed_out"] and not launch["canceled"] and launch["owned_group_absent"] is not False)
    if not good or buffers["receipt"] is None:
        return None, "failed_or_missing_launch"
    if len(buffers["receipt"]) + len(buffers["stderr"]) > request["max_output_bytes"]:
        raise ValueError("utility actual output exceeds fixed allowance")
    response = _json(buffers["receipt"])
    try:
        gain = legacy.observed_gain(request, response)
    except TimeoutError:
        return None, "absolute_deadline_exceeded"
    reports = tuple(report for report in (response["baseline"], response["after"]) if report is not None)
    if response["elapsed_ms"] > elapsed or sum(report["elapsed_ms"] for report in reports) > response["elapsed_ms"]:
        raise ValueError("utility CLI/search elapsed exceeds independent launch accounting")
    if response["status"] == "deferred":
        if (response["reason"] != "explicit_defer_no_cpu_check" or response["baseline"] is not None or response["after"] is not None
                or response["nodes"] != 0 or response["resume_kind"] is not None):
            raise ValueError("utility defer is not an explicit no-check receipt")
    return response, None if gain["status"] in ("observed", "deferred") else "unavailable"


def _coverage(response, request, conditions, *, prior=False):
    if response is None or response["status"] != "observed":
        return None if not prior else (False, 0)
    depth = 0
    for report in (response["baseline"], response["after"]):
        if report is None:
            continue
        if canonical(report["conditions"]) != canonical(conditions):
            raise ValueError("utility mixed actual observation conditions")
        if report["completion"] == "depth_limit" and report["completed_depth"] != report["requested_depth"]:
            raise ValueError("utility malformed full-depth completion")
        if report["score_scope"] == "completed_iteration":
            depth = max(depth, report["completed_depth"])
    if prior:
        # A valid earlier completed iteration remains known even when that
        # request stopped later at node/deadline/quiescence limits.
        return depth >= request["requested_depth"], depth
    before, after = response["baseline"], response["after"]
    if (before is None or after is None or before["score_scope"] != "completed_iteration"
            or before["completion"] != "depth_limit" or before["completed_depth"] != request["baseline_depth"]
            or after["score_scope"] != "completed_iteration" or after["completion"] != "depth_limit"
            or after["completed_depth"] != request["requested_depth"]):
        return None
    return 1


def _semantic(capability, parents, criterion, plan, known_depth):
    if capability is None:
        return None
    from .semantic_verifier import CheckedSemanticInput
    if type(capability) is not CheckedSemanticInput:
        raise ValueError("utility requires the existing immutable CheckedSemanticInput capability")
    capability.verify()
    capability.verify_parent(parents)
    query = capability.common_query()
    if (query["parent_input_sha256"] != criterion["current"]["input_sha256"]
            or query["current_view_sha256"] != parents.current_view.sha256
            or query["derived_input_sha256"] != criterion["semantic_input_sha256"]
            or query["semantic_context_sha256"] != criterion["semantic_context_sha256"]
            or query["common_query_sha256"] != plan["common_query_sha256"]
            or query["question"] != "unrestricted_recheck" or query["prefix"] != [] or query["root_moves"] != []
            or query["claimed_line"] != [] or query["allowed_tasks"] != list(ACTIONS)
            or query["baseline_depth"] != criterion["obligation"]["baseline_depth"]
            or query["requested_depth"] != criterion["obligation"]["required_horizon"]
            or query["known_completed_depth"] != known_depth or query["budget_bucket"] != criterion["budget_bucket"]
            or query["cpu_profile_sha256"] != criterion["obligation"]["conditions"]["profile_sha256"]
            or any(query[name] != criterion["obligation"]["conditions"]["resource_policy"][name]
                   for name in ("max_nodes_per_check", "max_wall_time_ms"))):
        raise ValueError("utility checked semantic common question/current/profile/knowledge mismatch")
    # Semantic question identity and the legacy CPU empty-branch digest are
    # different domains. Do not replace one with the other.
    return query


@dataclass(frozen=True)
class UtilityTarget:
    pair_id: str
    sign: int
    mask: bool
    resume_coverage: object
    defer_coverage: object
    reason: object
    split: str


class CheckedVerifierUtilityPairs:
    """Factory-only conditional admission with immutable byte/parent guards."""
    def __init__(self, *, parents, parent_artifacts, raw, executions, known, semantic, target, admission, _capability=None):
        if _capability is not _ADMISSION_CAPABILITY:
            raise ValueError("utility requires checked receipt admission factory")
        self.parents, self._semantic, self.target = parents, semantic, target
        self._parent_artifacts, self._raw = copy.deepcopy(parent_artifacts), copy.deepcopy(raw)
        self._executions, self._known = copy.deepcopy(executions), copy.deepcopy(known)
        self.admission = copy.deepcopy(admission)
        self._parent_identity = parents._frozen_admission_identity
        self._identity = self._seal()

    def __setattr__(self, name, value):
        if hasattr(self, "_identity"):
            raise AttributeError("checked utility capability is immutable")
        object.__setattr__(self, name, value)

    @property
    def sha256(self):
        self.verify()
        return self._identity

    def _seal(self):
        pins = lambda values: {name: None if raw is None else byte_pin(raw) for name, raw in values.items()}
        return digest([ADMISSION_SCHEMA, self.admission, self.target.__dict__, pins(self._raw),
                       pins(self._parent_artifacts), {name: pins(value) for name, value in self._executions.items()},
                       {name: pins(value) for name, value in self._known.items()}])

    def verify(self):
        self.parents._verify_raw_integrity()
        if self.parents._frozen_admission_identity != self._parent_identity or self._seal() != self._identity:
            raise ValueError("utility raw/target/parent/admission changed")
        if self._semantic is not None:
            self._semantic.verify()
            self._semantic.verify_parent(self.parents)
            current = self._semantic.common_query()
            if (current["common_query_sha256"] != self.admission["checked_common_query_sha256"]
                    or self._semantic.sha256 != self.admission["checked_semantic_admission_sha256"]):
                raise ValueError("utility checked semantic context changed")

    def raw_executions(self):
        self.verify()
        return copy.deepcopy({"current": self._executions, "known_snapshot": self._known})

    @property
    def checked_semantic(self):
        self.verify()
        if self._semantic is None:
            raise ValueError("utility requires checked semantic input")
        return self._semantic

    def collate(self, *, split="train"):
        self.verify()
        if self._semantic is None:
            raise ValueError("utility lacks checked semantic input; positive collation refused")
        if split != self.target.split:
            raise ValueError("utility immutable game split mismatch")
        _, _, training = _modules()
        query = self._semantic.common_query()
        row, encoding = self._semantic.derived_input(), self._semantic.encoded_snapshot()
        context = training.TaskContext(**query["context"])
        snapshot = row["input"]["snapshot"]
        # Source of the frozen V input remains the captured parent's producer;
        # the independent checker is a separate owner and must not replace it.
        authority = {"cpu_binary_sha256": self.parents.owned_sources["cpu_binary_sha256"], "input_sources": [snapshot["source"]]}
        dataset = training.ValidatedDataset([row], {"games": {snapshot["game_id"]: split}}, authority,
                                             {row["input"]["sha256"]: encoding})
        base = dataset.collate([0], "verifier", split=split, task_contexts=[context], device="cpu")
        from .config import TASKS
        return UtilityBatch(base, tuple(TASKS.index(action) for action in ACTIONS), self.target.sign,
                            self.target.mask, self.target.pair_id, tuple(query["eligible_tasks"]))


@dataclass(frozen=True)
class UtilityBatch:
    base: object
    task_slots: tuple
    sign: int
    mask: bool
    pair_id: str
    eligible_tasks: tuple


def admit_fixed_obligation_utility(*, parents, parent_artifacts, criterion_bytes, plan_bytes, knowledge_bytes,
                                  registration_bytes, source_bytes, binary_bytes, capabilities_bytes,
                                  executions, known_executions, independent_pins, checked_common_query=None):
    """Strict bytes/registered conditions first; semantic capability opens positives.

    Independently expected raw pins must come from the actual owners. Their
    comparison is conditional admission, not independent discovery of launches
    or builds. The known set defines this criterion's declared snapshot scope;
    an empty set is never asserted to be complete global knowledge.
    """
    _, _, training = _modules()
    if type(parents) is not training.ValidatedDataset or parents.frozen_admission is None:
        raise ValueError("utility requires existing strict frozen current parents")
    parents._verify_raw_integrity()
    raw = {"criterion": criterion_bytes, "plan": plan_bytes, "knowledge": knowledge_bytes,
           "registration": registration_bytes, "source": source_bytes, "binary": binary_bytes, "capabilities": capabilities_bytes}
    _fields(independent_pins, (*raw.keys(), "executions", "known_executions"), "independent utility byte authority")
    total = sum(len(value) for value in parent_artifacts.values() if isinstance(value, bytes))
    for name, value in raw.items():
        _actual(value, independent_pins[name])
        total += len(value)
    for group in (executions, known_executions):
        if not isinstance(group, dict) or len(group) > MAX_KNOWN:
            raise ValueError("utility finite actual execution count")
        for value in group.values():
            if not isinstance(value, dict):
                raise ValueError("utility actual execution buffers")
            total += sum(len(item) for item in value.values() if isinstance(item, bytes))
    if total > MAX_BYTES:
        raise ValueError("utility aggregate input byte allowance")
    criterion = _criterion(criterion_bytes)
    _, row = _parent(parents, parent_artifacts, criterion)
    registration, capabilities = _registered(registration_bytes, source_bytes, binary_bytes, capabilities_bytes,
                                               independent_pins, criterion)
    conditions = _conditions(criterion, row, registration)
    plan = _fields(_json(plan_bytes), ("schema", "scope", "pair_id", "parent", "current_input_sha256", "criterion_artifact",
                   "knowledge_artifact", "registration_artifact", "semantic_input_sha256", "semantic_context_sha256",
                   "common_query_sha256", "requests"), "result-free fixed utility plan")
    training._identity(plan["pair_id"])
    _sha(plan["common_query_sha256"])
    if (plan["schema"] != PLAN_SCHEMA or plan["scope"] != SCOPE or plan["parent"] != criterion["parent"]
            or plan["current_input_sha256"] != row["input"]["sha256"]
            or any(plan[name + "_artifact"] != byte_pin(raw[name]) for name in ("criterion", "knowledge", "registration"))
            or any(plan[name] != criterion[name] for name in ("semantic_input_sha256", "semantic_context_sha256"))):
        raise ValueError("utility result-before plan/criterion/current bytes mismatch")
    knowledge = _fields(_json(knowledge_bytes),
       ("schema", "scope", "current_input_sha256", "current_view_sha256", "branch_sha256", "execution_ids"), "declared prior knowledge snapshot")
    ids = knowledge["execution_ids"]
    if (knowledge["schema"] != KNOWLEDGE_SCHEMA or knowledge["scope"] != SCOPE
            or knowledge["current_input_sha256"] != row["input"]["sha256"]
            or knowledge["current_view_sha256"] != parents.current_view.sha256 or knowledge["branch_sha256"] != criterion["branch_sha256"]
            or not isinstance(ids, list) or len(ids) > MAX_KNOWN or len(set(ids)) != len(ids)
            or set(ids) != set(known_executions) or set(ids) != set(independent_pins["known_executions"])):
        raise ValueError("utility prior snapshot identity/duplicates/raw coverage mismatch")
    known_depth = 0
    for execution_id in ids:
        training._identity(execution_id)
        buffers = known_executions[execution_id]
        request = _request(buffers["request"], _json(buffers["request"])["task"], row, criterion, registration, capabilities, independent_pins["binary"])
        if request["task"] not in ACTIONS:
            raise ValueError("utility unsupported prior action")
        response, reason = _execution(buffers, independent_pins["known_executions"][execution_id], execution_id=execution_id,
               request=request, registration=registration, checker_pins=independent_pins, plan_sha=None, prior=True)
        if reason is not None:
            raise ValueError("utility prior observation is unknown; novel coverage refused")
        covered, depth = _coverage(response, request, conditions, prior=True)
        known_depth = max(known_depth, depth)
        if covered:
            raise ValueError("utility obligation already covered in declared prior snapshot")
    _fields(plan["requests"], ACTIONS, "exact utility action plan")
    results, reasons, used = {}, {}, set(ids)
    if set(executions) != set(ACTIONS) or set(independent_pins["executions"]) != set(ACTIONS):
        raise ValueError("utility requires exactly ResumeTask/Defer executions")
    for action in ACTIONS:
        spec = _fields(plan["requests"][action], ("execution_id", "request_artifact"), "fixed utility action request")
        training._identity(spec["execution_id"])
        if spec["execution_id"] in used:
            raise ValueError("utility duplicate current/prior execution ID")
        used.add(spec["execution_id"])
        buffers = executions[action]
        if spec["request_artifact"] != byte_pin(buffers["request"]):
            raise ValueError("utility action request bytes changed after planning")
        request = _request(buffers["request"], action, row, criterion, registration, capabilities, independent_pins["binary"])
        response, reason = _execution(buffers, independent_pins["executions"][action], execution_id=spec["execution_id"], request=request,
               registration=registration, checker_pins=independent_pins, plan_sha=byte_pin(plan_bytes)["sha256"])
        if reason is not None:
            coverage = None
        elif action == "defer":
            coverage = 0 if response["status"] == "deferred" else None
        else:
            coverage = _coverage(response, request, conditions)
        results[action], reasons[action] = coverage, reason or ("partial_terminal_or_unknown" if coverage is None else None)
    common = _semantic(checked_common_query, parents, criterion, plan, known_depth)
    mask = common is not None and results == {"resume_task": 1, "defer": 0}
    reason = None if mask else "requires_checked_semantic_input" if common is None else "masked_partial_failure_tie_or_unknown"
    target = UtilityTarget(plan["pair_id"], 1 if mask else 0, mask, results["resume_task"], results["defer"], reason,
                           parents.split[row["input"]["snapshot"]["game_id"]])
    admission = {"schema": ADMISSION_SCHEMA, "scope": SCOPE, "parent": criterion["parent"], "current": criterion["current"],
                 "semantic_input_sha256": criterion["semantic_input_sha256"], "semantic_context_sha256": criterion["semantic_context_sha256"],
                 "checked_common_query_sha256": None if common is None else common["common_query_sha256"], "known_completed_depth": known_depth,
                 "checked_semantic_admission_sha256": None if common is None else checked_common_query.sha256,
                 "known_snapshot_is_global_knowledge": False, "known_snapshot_empty": not ids,
                 "binary_sha256": independent_pins["binary"]["sha256"], "platform": registration["platform"],
                 "binary_pin_scope": registration["binary_pin_scope"], "execution_reasons": reasons,
                 "caller_assurance": "independently_pinned_caller_observation; registered source/build declarations",
                 "utility_claim": "fixed obligation coverage only; no optimality or global novelty",
                 "wdl_inferred": False, "rules_terminal_proof_inferred": False, "legacy_rank_modified": False,
                 "product_verifier_enabled": False, "actual_training_executed": False}
    return CheckedVerifierUtilityPairs(parents=parents, parent_artifacts=parent_artifacts, raw=raw, executions=executions,
            known=known_executions, semantic=checked_common_query, target=target, admission=admission, _capability=_ADMISSION_CAPABILITY)


def task_pairwise_softplus_loss(task_logits, batch):
    import torch
    from torch.nn import functional
    from .config import TASKS
    if (not isinstance(batch, UtilityBatch) or not isinstance(task_logits, torch.Tensor) or task_logits.shape != (1, len(TASKS))
            or task_logits.device.type != "cpu" or task_logits.dtype != torch.float32 or task_logits.requires_grad
            or task_logits.grad_fn is not None or not bool(torch.all(torch.isfinite(task_logits)))):
        raise ValueError("utility requires finite frozen CPU FP32 task logits")
    if (type(batch.mask) is not bool or type(batch.sign) is not int or batch.sign not in (0, 1) or batch.mask != (batch.sign != 0)
            or batch.task_slots != tuple(TASKS.index(action) for action in ACTIONS)
            or batch.eligible_tasks != tuple(action in ACTIONS for action in TASKS)):
        raise ValueError("utility task-head pair/mask alignment")
    with torch.inference_mode():
        left, right = batch.task_slots
        loss = functional.softplus(-(task_logits[0, left] - task_logits[0, right])) if batch.mask else torch.zeros((), dtype=torch.float32)
        if not bool(torch.isfinite(loss)):
            raise ValueError("utility pair loss became nonfinite")
        return loss


def frozen_verifier_utility_preparation(model, checked, *, parameter_observation_bytes, expected_observation_sha256,
                                        expected_parameter_sha256, checkpoint_sha256,
                                        split="train", max_wall_time_ms=300000, byte_limit=11 * 1024 * 1024):
    """Existing weights plus checked semantic features; no backward/step.

    Caller owns actual checkpoint reload and the independent observation pin.
    The semantic wrapper preserves existing weights but does not claim legacy
    V input/checkpoint/export compatibility or trained semantic utility.
    """
    started = time.monotonic()
    import torch
    from .preparation_check import _parameter_digest
    from .semantic_verifier import FrozenSemanticVerifier, PARAMETER_SCHEMA
    _sha(expected_parameter_sha256)
    _sha(checkpoint_sha256)
    if type(checked) is not CheckedVerifierUtilityPairs or not 1 <= _uint(max_wall_time_ms, 300000):
        raise ValueError("utility checked finite preparation required")
    deadline = started + max_wall_time_ms / 1000
    if not 1 <= _uint(byte_limit, 11 * 1024 * 1024):
        raise ValueError("utility finite semantic byte allowance")
    checked.verify()
    if not checked.target.mask:
        raise ValueError("all-masked utility cannot enter positive frozen preparation")
    observation_pin = {**byte_pin(parameter_observation_bytes), "sha256": _sha(expected_observation_sha256)}
    _modules()[0]._actual(parameter_observation_bytes, observation_pin, 8192)
    observation = _fields(_json(parameter_observation_bytes),
                          ("schema", "checkpoint_sha256", "parameter_sha256", "assurance_scope"), "registered frozen parameter observation")
    if (observation["schema"] != PARAMETER_SCHEMA or observation["checkpoint_sha256"] != checkpoint_sha256
            or observation["parameter_sha256"] != expected_parameter_sha256):
        raise ValueError("utility independently observed checkpoint/parameter pins differ")
    wrapper = FrozenSemanticVerifier(model, parameter_observation_bytes=parameter_observation_bytes,
                                      expected_observation_sha256=expected_observation_sha256)
    before = _parameter_digest(model)
    if before != expected_parameter_sha256:
        raise ValueError("utility reloaded parameter bytes differ from independent pin")
    if time.monotonic() >= deadline:
        raise TimeoutError("utility original preparation deadline expired before collation")
    batch = checked.collate(split=split)
    with torch.inference_mode():
        task_logits, _, semantic_audit = wrapper.forward([checked.checked_semantic], deadline=deadline, byte_limit=byte_limit)
        if semantic_audit["parameter_sha256"] != expected_parameter_sha256:
            raise ValueError("utility semantic forward parameter pin mismatch")
        loss = task_pairwise_softplus_loss(task_logits, batch)
        value = float(loss)
    after = _parameter_digest(model)
    checked.verify()
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("utility frozen preparation changed parameter/gradient state")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("utility positive preparation requires finite nonzero loss")
    if time.monotonic() >= deadline:
        raise ValueError("utility original preparation wall allowance expired")
    return {"schema": PREPARATION_SCHEMA, "scope": SCOPE, "admission_sha256": checked._identity,
            "checkpoint_sha256": checkpoint_sha256, "parameter_sha256_before": before, "parameter_sha256_after": after,
            "parameter_observation_sha256": _sha(expected_observation_sha256), "semantic_forward": semantic_audit,
            "current_view_sha256": checked.parents.current_view.sha256, "pair_id": checked.target.pair_id,
            "task_pairwise_loss": value, "known_pairs": 1, "actual_training_executed": False, "backward_executed": False,
            "optimizer_created": False, "gpu_executed": False, "product_verifier_enabled": False,
            "legacy_rank_modified": False, "known_snapshot_is_global_knowledge": False}
