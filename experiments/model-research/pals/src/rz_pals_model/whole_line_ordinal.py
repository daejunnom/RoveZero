"""Strict, conditional ordinal comparison of two forced whole-line plans.

The unchanged Rust continuation/1 wire reports an uncalibrated raw endpoint
side-to-move score. Only a separately pinned, durable, result-free criterion
may apply line parity and compare it. This is a line-conditioned surrogate,
never minimax/WDL/tactical truth, strategic repair validity or a role target.

Registration/build/launch observations are independent caller assurance. Their
coherent byte pins do not let generic metadata declare that a process ran.
The caller must observe the registered producer, executable and raw transport.
This module does not launch, replay chess, select DG05 leaves, collate, run
tensors/models, alter ordinary/auxiliary data, or promote old factual receipts.
Its existing strict-parent/Rules adapters require the managed CPU package.
"""
import copy
from dataclasses import dataclass

from . import comparative_training as cpu
from . import semantic_verifier as rules
from . import training

WIRE_SCHEMA = "rz-pals-owned-cpu-line-continuation/1"
CONDITIONS_SCHEMA = "rz-pals-owned-cpu-line-continuation-conditions/1"
LINE_DOMAIN = "rz-pals-owned-cpu-ordered-line/1"
CRITERION_SCHEMA = "rz-pals-whole-line-ordinal-criterion/1"
PLAN_SCHEMA = "rz-pals-whole-line-ordinal-before-plan/1"
REGISTRATION_SCHEMA = "rz-pals-whole-line-ordinal-checker-registration/1"
SOURCE_SCHEMA = "rz-pals-whole-line-ordinal-checker-source/1"
BUILD_SCHEMA = "rz-pals-whole-line-ordinal-build-observation/1"
LAUNCH_SCHEMA = "rz-pals-whole-line-ordinal-caller-launch/1"
PROCESS_SCHEMA = "rz-pals-whole-line-ordinal-process-observation/1"
ADMISSION_SCHEMA = "rz-pals-whole-line-ordinal-admission/1"
SCOPE = "line_conditioned_surrogate"
MAX_BYTES = 128 << 20
MAX_JSON = 4 << 20
ASSETS = ("criterion", "plan", "registration", "source", "binary", "capabilities", "build")
EXECUTION_ASSETS = ("request", "receipt", "stderr", "launch", "process_observation")
_FACTORY = object()


def canonical(value):
    """Sorted, compact UTF8 JSON, exact integer scalars; floats are forbidden."""
    return cpu.canonical_wire(value)


def digest(domain, value):
    return cpu.wire_digest([domain, value])


byte_pin = cpu.byte_pin


def _fields(value, names):
    return rules._fields(value, names)


def _int(value, low, high):
    return rules._int(value, low, high)


def _text(value, maximum=128):
    if type(value) is not str or not value.strip() or len(value.encode()) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError("bounded nonempty ordinal identifier required")
    return value


def _json(raw, maximum=MAX_JSON):
    if type(raw) is not bytes or not 1 <= len(raw) <= maximum:
        raise ValueError("bounded actual immutable whole-line bytes required")
    try:
        value = rules._parse(raw, maximum)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("bounded UTF8 whole-line JSON required") from error
    canonical(value)  # Also rejects float/int overflow anywhere in metadata.
    return value


def _pin(raw, expected, maximum=MAX_JSON, *, empty=False):
    cpu._actual(raw, expected, maximum, empty=empty)


def _moves(value, maximum=64, *, unique=False):
    rules._moves(value, maximum, unique)
    return value


def profile_description(horizon, tt_entries, quiescence_ply):
    _int(horizon, 1, 64)
    _int(tt_entries, 0, 1048576)
    _int(quiescence_ply, 0, 32)
    return {"domain": WIRE_SCHEMA, "search": cpu.SEARCH_VERSION, "evaluator": cpu.SEMANTICS,
            "profile": cpu.PROFILE, "tt_entries": tt_entries, "max_depth": horizon,
            "quiescence_ply": quiescence_ply, "selective_reductions": False}


def ordered_line_sha256(root_state, root_history, line):
    rules._sha(root_state)
    rules._sha(root_history)
    _moves(line)
    return cpu.wire_digest([LINE_DOMAIN, root_state, root_history, line])


def checker_namespace(registration):
    return digest(REGISTRATION_SCHEMA, {key: value for key, value in registration.items()
                                       if key != "checker_namespace_sha256"})


def input_anchor(parents, parent_index, query_anchor=None):
    """Read an already checked current anchor; this grants no role projection.

    Auxiliary public evidence remains the auxiliary input's own ordered seals.
    A current ordinary parent does not replace its false-learning query input.
    """
    if type(parents) is not training.ValidatedDataset:
        raise ValueError("existing strict current frozen parent required")
    parents._verify_raw_integrity()
    if parents.frozen_admission is None or parents._frozen_admission_identity is None:
        raise ValueError("metadata-only dataset cannot admit whole-line results")
    _int(parent_index, 0, len(parents.records) - 1)
    if parent_index not in parents.current_view.current_indices:
        raise ValueError("whole-line root parent must be a checked current leaf")
    row = parents.records[parent_index]
    frozen, kind, context, capability = row["input"], "ordinary_current", None, None
    board = parents.encodings[frozen["sha256"]].board
    if query_anchor is not None:
        from .native_divergence import CheckedNativeDivergence
        from .repair_context import CheckedRepairContext
        if type(query_anchor) is CheckedNativeDivergence:
            query_anchor.verify()
            if query_anchor._parents is not parents or query_anchor._index != parent_index:
                raise ValueError("D query must keep its exact strict current parent")
            frozen, descriptor, encoded, _ = query_anchor._views()
            board = encoded.board
            kind, context = "native_divergence", descriptor["sha256"]
        elif type(query_anchor) is CheckedRepairContext:
            query_anchor.verify()
            if query_anchor._parents is not parents or query_anchor._repair_index != parent_index:
                raise ValueError("Repair query must be its exact current ordinary row")
            kind, context = "native_initial_repair", query_anchor.admission()["context_sha256"]
        else:
            raise ValueError("unregistered query anchor/callback cannot admit whole-line evidence")
        capability = query_anchor.sha256
    snapshot = frozen["snapshot"]
    if snapshot["role"] not in ("proposer", "critic"):
        raise ValueError("owned P/C input root required")
    admission = parents.frozen_admission
    parent_binding = next((entry for entry in admission["inputs"]
                           if entry["binding"]["input_sha256"] == row["input"]["sha256"]), None)
    if parent_binding is None:
        raise ValueError("strict current prepared producer binding missing")
    return {"kind": kind, "parent_input_sha256": row["input"]["sha256"],
            "parent_label_sha256": training.label_digest(row), "query_input_sha256": frozen["sha256"],
            "query_context_sha256": context, "query_capability_sha256": capability,
            "frozen_admission_sha256": parents._frozen_admission_identity,
            "receipt_artifact": admission["receipt"], "raw_dataset_sha256": admission["raw_dataset_sha256"],
            "split_sha256": admission["split_sha256"], "current_view_sha256": parents.current_view.sha256,
            "producer_roster_sha256": admission["metadata_audit"]["roster_sha256"],
            "producer_envelope_sha256": admission["metadata_audit"]["envelope_sha256"],
            "parent_prepared_sha256": parent_binding["prepared_evidence_sha256"],
            "board64_piece_codes": list(board),
            "public_evidence": {"input_sha256": frozen["sha256"],
                "ordered_observation_sha256": [value["observation_sha256"] for value in snapshot["public_records"]]},
            **{name: copy.deepcopy(snapshot[name]) for name in ("game_id", "source", "frozen_epoch", "input_revision",
                 "encoding_sha256", "position_command", "board_fen", "rules_state_sha256", "rules_history_sha256", "legal_moves")},
            "side_to_move": "white" if snapshot["white_to_move"] else "black"}


def _criterion(raw):
    value = _fields(_json(raw), ("schema", "recipe_id", "scope", "direction", "line_plies", "horizon", "node_budget",
        "quiescence_ply", "tt_entries", "max_wall_time_ms", "max_output_bytes", "profile_sha256",
        "registration_sha256", "minimum_margin", "score_limit", "mate_threshold", "required_history_completeness"))
    if (value["schema"] != CRITERION_SCHEMA or value["scope"] != SCOPE
            or value["direction"] not in ("minimize_root_surrogate", "maximize_root_surrogate")
            or value["required_history_completeness"] != "complete"):
        raise ValueError("unsupported independently fixed line-conditioned ordinal criterion")
    _text(value["recipe_id"])
    for name, low, high in (("line_plies", 1, 64), ("horizon", 1, 64), ("node_budget", 1, 2**32 - 1),
        ("quiescence_ply", 0, 32), ("tt_entries", 0, 1048576), ("max_wall_time_ms", 1, 300000),
        ("max_output_bytes", 1024, 1048576), ("minimum_margin", 1, 40000), ("score_limit", 1, 20000),
        ("mate_threshold", 29000, 29000)):
        _int(value[name], low, high)
    if value["profile_sha256"] != cpu.wire_digest(profile_description(value["horizon"], value["tt_entries"], value["quiescence_ply"])):
        raise ValueError("criterion profile differs from exact Rust continuation namespace")
    rules._sha(value["registration_sha256"])
    return value


def _registered(raws, criterion):
    registration = _fields(_json(raws["registration"]), ("schema", "source_artifact", "binary_artifact", "capabilities_artifact",
        "build_artifact", "source_commit", "profile_sha256", "rules_version", "platform", "binary_pin_scope", "checker_namespace_sha256"))
    if registration["schema"] != REGISTRATION_SCHEMA or registration["checker_namespace_sha256"] != checker_namespace(registration):
        raise ValueError("registered whole-line namespace mismatch")
    for key in ("source", "binary", "capabilities", "build"):
        if canonical(registration[key + "_artifact"]) != canonical(byte_pin(raws[key])):
            raise ValueError("registered actual source/build/capability/binary bytes mismatch")
    if criterion["registration_sha256"] != byte_pin(raws["registration"])["sha256"] or registration["profile_sha256"] != criterion["profile_sha256"]:
        raise ValueError("criterion/checker registration was not independently fixed")
    if registration["platform"] != "linux" or registration["binary_pin_scope"] != "linux_loaded_executable_inode":
        raise ValueError("first whole-line unit requires actual Linux child image assurance")
    _text(registration["rules_version"])
    _text(registration["source_commit"])
    build = _fields(_json(raws["build"]), ("schema", "source_commit", "binary_artifact", "files_before", "files_after",
        "actual_build_exit_code", "source_verified_before_and_after_build", "assurance_scope"))
    if (build["schema"] != BUILD_SCHEMA or build["source_commit"] != registration["source_commit"]
            or canonical(build["binary_artifact"]) != canonical(byte_pin(raws["binary"]))
            or type(build["actual_build_exit_code"]) is not int or build["actual_build_exit_code"] != 0
            or build["source_verified_before_and_after_build"] is not True
            or build["assurance_scope"] != "independently_pinned_caller_build_observation"
            or canonical(build["files_before"]) != canonical(build["files_after"])):
        raise ValueError("actual caller registered build/source stability observation mismatch")
    files = build["files_before"]
    if type(files) is not list or not 1 <= len(files) <= 2048:
        raise ValueError("bounded original source manifest required")
    paths = set()
    for entry in files:
        _fields(entry, ("path", "bytes", "sha256"))
        path = _text(entry["path"], 512)
        if path in paths or path.startswith(("/", "\\")) or ":" in path or ".." in path.split("/"):
            raise ValueError("unique logical registered source paths required")
        paths.add(path)
        _int(entry["bytes"], 1, MAX_BYTES)
        rules._sha(entry["sha256"])
    source = _fields(_json(raws["source"]), ("schema", "source_commit", "build_artifact", "profile", "profile_sha256",
        "search_version", "search_conditions", "value_identity"))
    expected_profile = profile_description(criterion["horizon"], criterion["tt_entries"], criterion["quiescence_ply"])
    identity = {"semantics": cpu.SEMANTICS, "weights_sha256": None, "training": {"kind": "bootstrap"}}
    if (source["schema"] != SOURCE_SCHEMA or source["source_commit"] != registration["source_commit"]
            or canonical(source["build_artifact"]) != canonical(byte_pin(raws["build"])) or canonical(source["profile"]) != canonical(expected_profile)
            or source["profile_sha256"] != criterion["profile_sha256"] or source["search_version"] != cpu.SEARCH_VERSION
            or source["search_conditions"] != cpu.search_conditions(criterion["horizon"], criterion["tt_entries"], criterion["quiescence_ply"])
            or canonical(source["value_identity"]) != canonical(identity)):
        raise ValueError("actual registered Independent bootstrap profile/source mismatch")
    capabilities = _json(raws["capabilities"])
    expected = {"schema": WIRE_SCHEMA, "max_checks": 1, "terminal_checks": 0, "max_line_plies": 64, "max_horizon": 64,
        "max_node_budget": 2**32 - 1, "max_tt_entries": 1048576, "max_quiescence_ply": 32,
        "max_request_bytes": 524288, "max_response_bytes": 1048576, "max_wall_time_ms": 300000,
        "fresh_engine": True, "restriction": "unrestricted_endpoint", "resume": False, "profile": cpu.PROFILE,
        "search_version": cpu.SEARCH_VERSION, "score_semantics": cpu.SEMANTICS,
        "score_units": "uncalibrated_endpoint_side_to_move_raw", "rules_descriptor_schema": rules.DESCRIPTOR_SCHEMA,
        "line_domain": LINE_DOMAIN, "conditions_schema": CONDITIONS_SCHEMA,
        "caller_registration_scope": "declared_not_independently_verified", "binary_pin_scope": registration["binary_pin_scope"],
        "cpu_binary_sha256": byte_pin(raws["binary"])["sha256"], "training_target_created": False,
        "product_verifier_enabled": False, "teacher_gpu": False, "ranking_created": False}
    if canonical(capabilities) != canonical(expected):
        raise ValueError("actual registered CLI capabilities mismatch; static dispatcher facts insufficient")
    return registration, source


def _request(raw, planned, anchor, criterion, registration, plan_pin):
    value = _fields(_json(raw, 524288), ("schema", "task_id", "captured_input_sha256", "checker_namespace_sha256",
        "before_result_anchor_sha256", "frozen_epoch", "input_revision", "position_command", "expected_root_board_fen",
        "root_rules_state_sha256", "root_rules_history_sha256", "expected_root_legal_moves", "root_side_to_move",
        "line", "line_sha256", "expected_endpoint_board_fen", "endpoint_rules_state_sha256", "endpoint_rules_history_sha256",
        "expected_endpoint_legal_moves", "endpoint_side_to_move", "cpu_binary_sha256", "cpu_profile_sha256",
        "horizon", "node_budget", "tt_entries", "quiescence_ply", "max_wall_time_ms", "max_output_bytes", "context_sha256"))
    expected = {"schema": WIRE_SCHEMA, "task_id": planned["task_id"], "captured_input_sha256": anchor["query_input_sha256"],
        "checker_namespace_sha256": registration["checker_namespace_sha256"], "before_result_anchor_sha256": plan_pin["sha256"],
        "frozen_epoch": anchor["frozen_epoch"], "input_revision": anchor["input_revision"], "position_command": anchor["position_command"],
        "expected_root_board_fen": anchor["board_fen"], "root_rules_state_sha256": anchor["rules_state_sha256"],
        "root_rules_history_sha256": anchor["rules_history_sha256"], "expected_root_legal_moves": anchor["legal_moves"],
        "root_side_to_move": anchor["side_to_move"], "line": planned["line"], "line_sha256": planned["line_sha256"],
        "cpu_binary_sha256": registration["binary_artifact"]["sha256"], "cpu_profile_sha256": criterion["profile_sha256"],
        **{name: criterion[name] for name in ("horizon", "node_budget", "tt_entries", "quiescence_ply", "max_wall_time_ms", "max_output_bytes")},
        **{name: planned[name] for name in ("expected_endpoint_board_fen", "endpoint_rules_state_sha256", "endpoint_rules_history_sha256",
                                         "expected_endpoint_legal_moves", "endpoint_side_to_move")}}
    if canonical({key: item for key, item in value.items() if key != "context_sha256"}) != canonical(expected) or value["context_sha256"] != digest(WIRE_SCHEMA, expected):
        raise ValueError("actual whole-line request differs from pre-result plan/current captured input")
    return value


def _launch(buffers, request, registration, raws, pins):
    launch = _fields(_json(buffers["launch"]), ("schema", "task_id", "registration_sha256", "request", "receipt", "stderr",
        "process_observation", "criterion_artifact", "plan_artifact", "binary_artifact", "source_artifact",
        "anchor_durable_before_spawn", "criterion_fixed_before_spawn", "assurance_scope", "spawned", "reaped", "exit_code",
        "elapsed_ms", "timed_out", "platform", "binary_pin_scope"))
    if (launch["schema"] != LAUNCH_SCHEMA or launch["task_id"] != request["task_id"]
            or launch["registration_sha256"] != byte_pin(raws["registration"])["sha256"]
            or any(canonical(launch[name]) != canonical(pins[name]) for name in ("request", "receipt", "stderr", "process_observation"))
            or any(canonical(launch[name + "_artifact"]) != canonical(byte_pin(raws[name])) for name in ("criterion", "plan", "binary", "source"))
            or launch["assurance_scope"] != "independently_pinned_caller_execution_observation"
            or launch["anchor_durable_before_spawn"] is not True or launch["criterion_fixed_before_spawn"] is not True
            or launch["spawned"] is not True or launch["reaped"] is not True or type(launch["exit_code"]) is not int
            or launch["exit_code"] != 0 or launch["timed_out"] is not False
            or launch["platform"] != registration["platform"] or launch["binary_pin_scope"] != registration["binary_pin_scope"]):
        raise ValueError("actual durable anchor/registered launch/exit mismatch; cannot mask execution failure")
    _int(launch["elapsed_ms"], 0, request["max_wall_time_ms"] - 1)
    process = _fields(_json(buffers["process_observation"]), ("schema", "task_id", "spawned", "reaped", "exit_code",
        "pipes_finished", "owned_group_absent", "process_supervision_scope", "loaded_executable", "binary_before", "binary_after",
        "source_before", "source_after", "binary_path_stable", "source_path_stable", "transport_failure", "overflow",
        "original_deadline_met", "elapsed_ms"))
    _int(process["elapsed_ms"], 0, request["max_wall_time_ms"] - 1)
    if (process["schema"] != PROCESS_SCHEMA or process["task_id"] != request["task_id"]
            or any(process[name] is not True for name in ("spawned", "reaped", "pipes_finished", "owned_group_absent", "binary_path_stable",
                                                       "source_path_stable", "original_deadline_met"))
            or type(process["exit_code"]) is not int or process["exit_code"] != 0
            or process["process_supervision_scope"] != "posix_owned_process_group"
            or process["transport_failure"] is not None or process["overflow"] is not False
            or process["elapsed_ms"] != launch["elapsed_ms"]):
        raise ValueError("actual child/EOF/group/original allowance failure; cannot classify as masked")
    for name in ("binary_before", "binary_after", "source_before", "source_after"):
        if canonical(process[name]) != canonical(byte_pin(raws["binary" if name.startswith("binary") else "source"])):
            raise ValueError("registered live path stability failed")
    loaded = _fields(process["loaded_executable"], ("status", "artifact", "scope", "observed_before_stdin"))
    if (loaded["status"] != "checked" or canonical(loaded["artifact"]) != canonical(byte_pin(raws["binary"]))
            or loaded["scope"] != "caller_child_loaded_inode" or loaded["observed_before_stdin"] is not True):
        raise ValueError("independent actual loaded child image required; self-receipt alone insufficient")
    return launch["elapsed_ms"]


def _receipt(raw, request, registration, source, caller_elapsed, anchor):
    if raw is None:
        return None, "missing"
    value = _fields(_json(raw, request["max_output_bytes"]), ("schema", "task_id", "captured_input_sha256", "checker_namespace_sha256",
        "before_result_anchor_sha256", "frozen_epoch", "input_revision", "context_sha256", "cpu_binary_sha256", "binary_pin_scope",
        "caller_registration_scope", "status", "root", "endpoint", "line", "line_sha256", "line_rules_validated", "conditions",
        "conditions_sha256", "cpu_calls", "fresh_engine", "report", "elapsed_ms", "deadline_exceeded", "training_target_created", "product_verifier_enabled"))
    for name in ("schema", "task_id", "captured_input_sha256", "checker_namespace_sha256", "before_result_anchor_sha256",
                 "frozen_epoch", "input_revision", "context_sha256", "cpu_binary_sha256", "line", "line_sha256"):
        if canonical(value[name]) != canonical(request[name]):
            raise ValueError("actual continuation receipt request identity mismatch")
    if (value["binary_pin_scope"] != registration["binary_pin_scope"]
            or value["caller_registration_scope"] != "declared_not_independently_verified"
            or value["line_rules_validated"] is not True or any(value[name] is not False
                for name in ("deadline_exceeded", "training_target_created", "product_verifier_enabled"))):
        raise ValueError("factual continuation scope/image/Rules validation mismatch")
    _int(value["elapsed_ms"], 0, min(caller_elapsed, request["max_wall_time_ms"] - 1))
    for name, token_kind, prefix in (("root", "root_legal", "root"), ("endpoint", "claim_end_legal", "endpoint")):
        descriptor = value[name]
        rules._descriptor(descriptor, token_kind, registration)
        expected = {"board_fen": request["expected_" + prefix + "_board_fen"],
                    "rules_state_sha256": request[prefix + "_rules_state_sha256"],
                    "rules_history_sha256": request[prefix + "_rules_history_sha256"],
                    "legal_moves": request["expected_" + prefix + "_legal_moves"], "side_to_move": request[prefix + "_side_to_move"]}
        if any(descriptor[key] != item for key, item in expected.items()):
            raise ValueError("actual reconstructed root/endpoint full history/legal order/turn differs")
    root, endpoint = value["root"], value["endpoint"]
    if root["board64_piece_codes"] != anchor["board64_piece_codes"]:
        raise ValueError("actual root Rules board differs from captured checked encoding")
    if root["play_status"] != "ongoing":
        raise ValueError("a forced line cannot begin after a Rules terminal")
    for descriptor in (root, endpoint):
        # An irreversible ply may make repetition evidence complete even when
        # the full history before a FEN remains unknown. Keep both Rules facts.
        if descriptor["history_completeness"] == "complete" and descriptor["repetition_history_complete"] is not True:
            raise ValueError("actual Rules history completeness facts inconsistent")
    if (endpoint["history_completeness"] != root["history_completeness"]
            or endpoint["history_origin"] != root["history_origin"]):
        raise ValueError("forced replay cannot replace the known root history origin")
    expected_turn = root["side_to_move"] if len(request["line"]) % 2 == 0 else ("black" if root["side_to_move"] == "white" else "white")
    if endpoint["side_to_move"] != expected_turn or endpoint["known_history_positions"] != root["known_history_positions"] + len(request["line"]):
        raise ValueError("ordered whole-line parity/history length mismatch")
    conditions = {"schema": CONDITIONS_SCHEMA, "root_rules_state_sha256": root["rules_state_sha256"],
        "root_rules_history_sha256": root["rules_history_sha256"], "root_legal_order_sha256": root["legal_order_sha256"],
        "endpoint_rules_state_sha256": endpoint["rules_state_sha256"], "endpoint_rules_history_sha256": endpoint["rules_history_sha256"],
        "endpoint_legal_order_sha256": endpoint["legal_order_sha256"], "line_sha256": request["line_sha256"],
        "line_plies": len(request["line"]), "endpoint_side_to_move": endpoint["side_to_move"], "profile_sha256": request["cpu_profile_sha256"],
        "profile": cpu.PROFILE, "value_identity": source["value_identity"], "search_version": cpu.SEARCH_VERSION,
        "search_conditions": source["search_conditions"], **{key: request[key] for key in ("horizon", "node_budget", "tt_entries", "quiescence_ply")},
        "restriction": "unrestricted_endpoint", "perspective": "endpoint_side_to_move",
        "resource_policy": {"max_wall_time_ms": request["max_wall_time_ms"], "max_output_bytes": request["max_output_bytes"],
                            "max_checks": 1, "search_deadline_reserve_ms": min(request["max_wall_time_ms"] // 10, 1000)}}
    if canonical(value["conditions"]) != canonical(conditions) or value["conditions_sha256"] != cpu.wire_digest(conditions):
        raise ValueError("actual endpoint namespace/profile/resources/conditions mismatch")
    _int(value["cpu_calls"], 0, 1)
    if endpoint["play_status"] == "rules_terminal":
        if value["status"] != "rules_terminal" or value["report"] is not None or value["cpu_calls"] != 0 or value["fresh_engine"] is not False:
            raise ValueError("Rules terminal endpoint cannot manufacture CPU score/work")
        return None, "terminal"
    if root["play_status"] != "ongoing" or value["cpu_calls"] != 1 or value["fresh_engine"] is not True:
        raise ValueError("one actual fresh nonterminal endpoint check required")
    report = _fields(value["report"], ("raw_score", "best_move", "pv", "score_scope", "completion", "completed_depth", "requested_depth",
        "nodes", "quiescence_nodes", "tt_hits", "elapsed_ms", "reused_completed_depth", "root_restricted", "score_provenance", "pv_rules_validated"))
    _int(report["raw_score"], -(2**31), 2**31 - 1)
    _int(report["completed_depth"], 0, request["horizon"])
    _int(report["nodes"], 0, request["node_budget"])
    _int(report["quiescence_nodes"], 0, report["nodes"])
    _int(report["tt_hits"], 0, report["nodes"])
    _int(report["elapsed_ms"], 0, min(value["elapsed_ms"], request["max_wall_time_ms"] - conditions["resource_policy"]["search_deadline_reserve_ms"]))
    if (report["requested_depth"] != request["horizon"] or type(report["requested_depth"]) is not int
            or type(report["reused_completed_depth"]) is not int or report["reused_completed_depth"] != 0
            or report["root_restricted"] is not False or report["pv_rules_validated"] is not True
            or report["score_provenance"] != cpu.SEMANTICS
            or report["score_scope"] not in ("frontier_only", "completed_iteration")
            or (report["score_scope"] == "completed_iteration") != (report["completed_depth"] > 0)):
        raise ValueError("fresh actual endpoint raw score/PV/depth provenance mismatch")
    _moves(report["pv"], request["horizon"] + request["quiescence_ply"])
    _int(report["best_move"], 0, 65535)
    if not report["pv"] or report["best_move"] != report["pv"][0] or report["best_move"] not in endpoint["legal_moves"]:
        raise ValueError("known Rules-validated endpoint PV/best move required")
    if value["status"] == "canceled" and report["completion"] == "canceled":
        return report["raw_score"], "canceled"
    if value["status"] == "partial" and report["completion"] in ("node_limit", "deadline", "quiescence_limit"):
        return report["raw_score"], "partial"
    if (value["status"] != "completed" or report["completion"] != "depth_limit"
            or report["completed_depth"] != request["horizon"] or report["score_scope"] != "completed_iteration"):
        raise ValueError("receipt status/completion/exact requested depth inconsistent")
    if report["nodes"] == 0:
        raise ValueError("completed nonterminal fresh search must preserve observed node work")
    if root["history_completeness"] != "complete" or endpoint["history_completeness"] != "complete":
        return report["raw_score"], "unknown_history_prefix"
    return report["raw_score"], None


@dataclass(frozen=True)
class LinePairOutcome:
    pair_id: str
    mask: bool
    sign: int
    reason: str
    raw_endpoint_scores: tuple
    criterion_root_scores: tuple
    scope: str = SCOPE


def _validate(parents, index, query_anchor, raws, executions, pins):
    _fields(pins, (*ASSETS, "executions", "anchor"))
    total = sum(len(raw) for raw in raws.values()) + sum(len(raw) for buffers in executions.values() for raw in buffers.values() if raw is not None)
    if total > MAX_BYTES:
        raise ValueError("aggregate actual whole-line admission byte allowance")
    for name in ASSETS:
        _pin(raws[name], pins[name], MAX_BYTES if name == "binary" else MAX_JSON)
    anchor = input_anchor(parents, index, query_anchor)
    if canonical(anchor) != canonical(pins["anchor"]):
        raise ValueError("independent current input/label/producer/context/public evidence anchor mismatch")
    criterion = _criterion(raws["criterion"])
    registration, source = _registered(raws, criterion)
    plan = _fields(_json(raws["plan"]), ("schema", "pair_id", "anchor", "criterion_artifact", "registration_artifact", "direction", "lines"))
    if (plan["schema"] != PLAN_SCHEMA or canonical(plan["anchor"]) != canonical(anchor)
            or canonical(plan["criterion_artifact"]) != canonical(byte_pin(raws["criterion"]))
            or canonical(plan["registration_artifact"]) != canonical(byte_pin(raws["registration"])) or plan["direction"] != criterion["direction"]):
        raise ValueError("durable before-result plan/criterion/current anchor mismatch")
    _text(plan["pair_id"])
    if type(plan["lines"]) is not list or len(plan["lines"]) != 2:
        raise ValueError("exactly two ordered finite forced plans required")
    tasks, raw_scores, reasons = [], [], []
    for planned in plan["lines"]:
        _fields(planned, ("task_id", "line", "line_sha256", "expected_endpoint_board_fen", "endpoint_rules_state_sha256",
                         "endpoint_rules_history_sha256", "expected_endpoint_legal_moves", "endpoint_side_to_move"))
        task = _text(planned["task_id"])
        tasks.append(task)
        _moves(planned["line"])
        _moves(planned["expected_endpoint_legal_moves"], 256, unique=True)
        _text(planned["expected_endpoint_board_fen"], 512)
        rules._sha(planned["endpoint_rules_state_sha256"])
        rules._sha(planned["endpoint_rules_history_sha256"])
        expected_turn = anchor["side_to_move"] if criterion["line_plies"] % 2 == 0 else ("black" if anchor["side_to_move"] == "white" else "white")
        if planned["endpoint_side_to_move"] != expected_turn:
            raise ValueError("pre-result endpoint turn must match exact ordered-line parity")
        if (len(planned["line"]) != criterion["line_plies"] or planned["line_sha256"] != ordered_line_sha256(
                anchor["rules_state_sha256"], anchor["rules_history_sha256"], planned["line"])):
            raise ValueError("full ordered line/root history/L identity mismatch")
        if planned["line"][0] not in anchor["legal_moves"]:
            raise ValueError("forced plan must start with an actual Rules-ordered root legal move")
        buffers = _fields(executions[task], EXECUTION_ASSETS)
        execution_pins = _fields(pins["executions"][task], EXECUTION_ASSETS)
        for name in EXECUTION_ASSETS:
            if name == "receipt" and buffers[name] is None and execution_pins[name] is None:
                continue
            _pin(buffers[name], execution_pins[name], MAX_JSON if name not in ("request", "receipt", "stderr") else
                 524288 if name == "request" else criterion["max_output_bytes"], empty=name == "stderr")
        request = _request(buffers["request"], planned, anchor, criterion, registration, byte_pin(raws["plan"]))
        elapsed = _launch(buffers, request, registration, raws, execution_pins)
        score, reason = _receipt(buffers["receipt"], request, registration, source, elapsed, anchor)
        raw_scores.append(score)
        reasons.append(reason)
    if len(set(tasks)) != 2 or set(executions) != set(tasks) or set(pins["executions"]) != set(tasks) or plan["lines"][0]["line"] == plan["lines"][1]["line"]:
        raise ValueError("two unique task IDs/distinct full ordered plans and exact execution set required")
    root_scores = tuple(None if score is None else score * (-1 if criterion["line_plies"] % 2 else 1) for score in raw_scores)
    reason = next((item for item in reasons if item is not None), None)
    if reason is None:
        if any(abs(score) >= criterion["mate_threshold"] for score in raw_scores):
            reason = "mate_band"
        elif any(abs(score) > criterion["score_limit"] for score in raw_scores):
            reason = "out_of_range"
        elif abs(root_scores[0] - root_scores[1]) < criterion["minimum_margin"]:
            reason = "tie_or_below_margin"
    sign = 0
    if reason is None:
        left_better = root_scores[0] < root_scores[1] if criterion["direction"] == "minimize_root_surrogate" else root_scores[0] > root_scores[1]
        sign = 1 if left_better else -1
    outcome = LinePairOutcome(plan["pair_id"], reason is None, sign, reason or "known_line_conditioned_surrogate",
                              tuple(raw_scores), root_scores)
    admission = {"schema": ADMISSION_SCHEMA, "scope": SCOPE, "pair_id": plan["pair_id"], "anchor": anchor,
        "criterion_artifact": byte_pin(raws["criterion"]), "plan_artifact": byte_pin(raws["plan"]),
        "caller_assurance": "independently_pinned_caller_build_and_actual_execution_observations",
        "metadata_alone_is_execution_authority": False, "ordinal_comparison_known": outcome.mask,
        "requires_role_projection": True, "ordinary_policy_admitted": False, "auxiliary_has_current_label": False,
        "strategic_repair_validity_admitted": False, "counterexample_validity_admitted": False, "wdl_admitted": False,
        "minimax_admitted": False, "tactical_proof_admitted": False, "training_target_created": False,
        "actual_training_executed": False, "gpu_executed": False}
    return outcome, admission


class CheckedWholeLinePair:
    """Immutable conditional comparison, with no role-target/collation authority."""
    __slots__ = ("_parents", "_index", "_anchor", "_raws", "_executions", "_pins", "_identity")

    def __init__(self, token=None, *, parents=None, index=None, anchor=None, raws=None, executions=None, pins=None):
        if token is not _FACTORY:
            raise ValueError("whole-line capability requires actual strict factory admission")
        for name, value in (("_parents", parents), ("_index", index), ("_anchor", anchor), ("_raws", tuple(sorted(raws.items()))),
            ("_executions", tuple((task, tuple(sorted(buffers.items()))) for task, buffers in sorted(executions.items()))),
            ("_pins", canonical(pins)), ("_identity", digest(ADMISSION_SCHEMA, pins))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("checked whole-line comparison is immutable")

    def _views(self):
        pins = _json(self._pins)
        if digest(ADMISSION_SCHEMA, pins) != self._identity:
            raise ValueError("whole-line comparison identity changed")
        return _validate(self._parents, self._index, self._anchor, dict(self._raws),
                         {task: dict(buffers) for task, buffers in self._executions}, pins)

    def verify(self):
        self._views()
        return self

    @property
    def sha256(self):
        self.verify()
        return self._identity

    @property
    def outcome(self):
        return self._views()[0]

    @property
    def admission(self):
        return copy.deepcopy(self._views()[1])


def admit_whole_line_pair(*, parents, parent_index, criterion_bytes, plan_bytes, registration_bytes, source_bytes,
                         binary_bytes, capabilities_bytes, build_bytes, executions, expected_pins, query_anchor=None):
    raws = dict(criterion=criterion_bytes, plan=plan_bytes, registration=registration_bytes, source=source_bytes,
                binary=binary_bytes, capabilities=capabilities_bytes, build=build_bytes)
    if type(executions) is not dict or len(executions) != 2:
        raise ValueError("two actual observed execution byte bundles required")
    for buffers in executions.values():
        _fields(buffers, EXECUTION_ASSETS)
    if any(type(raw) is not bytes for raw in raws.values()) or any(type(raw) is not bytes and raw is not None
            for buffers in executions.values() if type(buffers) is dict for raw in buffers.values()):
        raise ValueError("immutable actual byte bundles required")
    pins = _json(canonical(expected_pins))
    _validate(parents, parent_index, query_anchor, raws, executions, pins)
    return CheckedWholeLinePair(_FACTORY, parents=parents, index=parent_index, anchor=query_anchor,
                               raws=raws, executions=executions, pins=pins)
