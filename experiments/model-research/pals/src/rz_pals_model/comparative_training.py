"""Own CPU candidate receipts -> separate ordinal P/C preparation targets.

This adapter executes no checker or launcher. Its independent byte pins must
come from the owner who registered and observed the real child invocation.
Hashes, JSON flags and the metadata overlay cannot establish that observation.
The checked scope is conditional on that caller assurance, including the
before-result publication and binary/platform guarantees stated below. Rust
owns Rules replay; Python checks its registered receipt and exact PV bytes,
and does not implement chess legality or turn raw units into WDL/proofs.

The existing strict frozen loader and DG05 current selector remain the base
admission. Comparative targets/losses are separate from ordinary policy/WDL,
repair-validity and divergence targets. No optimizer/backward/update exists.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

from . import comparative as metadata
from . import frozen_producer as frozen
from .training import (_CollectionArtifactReader, _fields, _identity, _sha,
                       _uint, _unique_json, label_digest,
                       load_frozen_collected_dataset, ValidatedDataset)

REQUEST_SCHEMA = "rz-pals-owned-cpu-candidate/1"
CONDITIONS_SCHEMA = "rz-pals-owned-cpu-candidate-conditions/1"
LEGAL_DOMAIN = "rz-pals-owned-cpu-candidate-legal-order/1"
REGISTRATION_SCHEMA = "rz-pals-owned-candidate-checker-registration/1"
SOURCE_SCHEMA = "rz-pals-owned-candidate-checker-source/1"
CRITERION_SCHEMA = "rz-pals-owned-candidate-ordinal-criterion/1"
PLAN_SCHEMA = "rz-pals-owned-candidate-pair-plan/1"
LAUNCH_SCHEMA = "rz-pals-owned-candidate-launch-observation/1"
ADMISSION_SCHEMA = "rz-pals-owned-candidate-pair-admission/1"
PREPARATION_SCHEMA = "rz-pals-owned-candidate-pair-preparation/1"
BANK_SCHEMA = "rz-pals-owned-candidate-pair-bank/1"
SEARCH_VERSION = "rz-cpu-pvs/0.1"
SEMANTICS = "bootstrap-material-pst-v1"
PROFILE = "cpu-independent-conservative-v1"
MAX_BYTES = 128 * 1024 * 1024
MAX_JSON = 4 * 1024 * 1024
SEARCH_CONDITIONS = ("iterative-deepening:1..requested;root-window:full-first,aspiration40-following,"
                     "full-when-mate-or-fail-inclusive;pvs:first-full,following-zero-window,strict-interior-research;"
                     "qsearch:tactical-capture-ep-promotion,all-check-evasions,no-check-standpat;q-limit:checked-abort;"
                     "tt:direct-mapped,full-history-value-profile,equal-remaining-depth,completed-nodes-only;"
                     "selectivity:no-reductions-no-nullmove;ties:Rules-order;score:side-to-move-raw")
_ADMISSION_CAPABILITY = object()


def _tree(value, depth=0):
    """Wire canonical admits bool/signed raw scores, but never floats."""
    if depth > 24:
        raise ValueError("candidate JSON nesting limit")
    if value is None or type(value) is bool:
        return
    if type(value) is int and -(1 << 63) <= value <= (1 << 64) - 1:
        return
    if isinstance(value, str) and len(value.encode("utf-8")) <= MAX_JSON:
        return
    if isinstance(value, list) and len(value) <= 65536:
        for item in value:
            _tree(item, depth + 1)
        return
    if isinstance(value, dict) and len(value) <= 256 and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _tree(key, depth + 1)
            _tree(item, depth + 1)
        return
    raise ValueError("candidate JSON requires bounded integer/bool/string/null scalars")


def canonical_wire(value):
    """Recursive sorted keys, compact UTF-8, retained arrays, no float coercion."""
    _tree(value)
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > MAX_JSON:
        raise ValueError("candidate canonical byte limit")
    return raw


def wire_digest(value):
    return hashlib.sha256(canonical_wire(value)).hexdigest()


def byte_pin(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
        raise ValueError("candidate actual bytes required within allocation limit")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _actual(raw, pin, maximum=MAX_JSON, *, empty=False):
    _fields(pin, ("bytes", "sha256"), "independent candidate actual byte pin")
    count = _uint(pin["bytes"], maximum)
    _sha(pin["sha256"])
    if (not empty and count == 0) or byte_pin(raw) != pin:
        raise ValueError("candidate actual bytes differ from independent pin")


def _json(raw):
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_JSON:
        raise ValueError("candidate JSON byte limit")
    try:
        value = _unique_json(raw.decode("utf-8"))
        _tree(value)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("candidate JSON encoding/nesting") from error
    return value


def profile_description(horizon, tt_entries, quiescence_ply):
    return {"domain": REQUEST_SCHEMA, "search": SEARCH_VERSION, "evaluator": SEMANTICS,
            "profile": PROFILE, "tt_entries": tt_entries, "max_depth": horizon,
            "quiescence_ply": quiescence_ply, "selective_reductions": False}


def search_conditions(horizon, tt_entries, quiescence_ply):
    return (f"{SEARCH_CONDITIONS};profile={PROFILE};max_depth={horizon};"
            f"q_plies={quiescence_ply};tt_entries={tt_entries}")


def _criterion(raw, expected):
    _actual(raw, expected)
    value = _fields(_json(raw), ("schema", "recipe_id", "method", "minimum_margin", "score_limit",
                                "mate_threshold", "score_scope", "completion", "perspective"), "fixed ordinal criterion")
    _identity(value["recipe_id"])
    if (value["schema"] != CRITERION_SCHEMA or value["method"] != "finite_depth_raw_ordinal"
            or not 1 <= _uint(value["minimum_margin"], 20000) or _uint(value["score_limit"]) != 20000
            or _uint(value["mate_threshold"]) != 29000 or value["score_scope"] != "completed_iteration"
            or value["completion"] != "depth_limit" or value["perspective"] != "captured_side_to_move"):
        raise ValueError("unsupported fixed finite-depth ordinal criterion")
    return value


def _registered_checker(raw, source_raw, binary_raw, expected, checker):
    """Registered bytes are caller authority, not a proof inferred from hashes."""
    for name, data in (("registration", raw), ("source", source_raw), ("binary", binary_raw)):
        _actual(data, expected[name], MAX_BYTES if name == "binary" else MAX_JSON)
    registration = _fields(_json(raw), ("schema", "checker", "source_artifact", "binary_artifact",
                                        "profile_sha256", "tt_entries", "quiescence_ply", "platform", "binary_pin_scope"),
                           "independently registered checker")
    if (registration["schema"] != REGISTRATION_SCHEMA or registration["source_artifact"] != expected["source"]
            or registration["binary_artifact"] != expected["binary"]):
        raise ValueError("registered actual checker source/binary pins mismatch")
    # Registration excludes its own digest to avoid a hash cycle.
    registered = registration["checker"]
    if not isinstance(registered, dict) or "registration_sha256" in registered:
        raise ValueError("registered checker identity must exclude its own byte digest")
    if canonical_wire({**registered, "registration_sha256": expected["registration"]["sha256"]}) != canonical_wire(checker):
        raise ValueError("actual checker registration differs from independent overlay checker")
    if checker["source"]["cpu_binary_sha256"] != expected["binary"]["sha256"] or checker["source"]["model_weights_sha256"] is not None:
        raise ValueError("candidate adapter requires actual registered bootstrap binary with no weights")
    horizon, budget = checker["horizon"], checker["node_budget"]
    if not 1 <= _uint(horizon, 64) or not 1 <= _uint(budget, (1 << 32) - 1) or checker["perspective"] != "captured_side_to_move":
        raise ValueError("actual candidate horizon/node/perspective limits")
    tt, q = _uint(registration["tt_entries"], 1048576), _uint(registration["quiescence_ply"], 32)
    profile = profile_description(horizon, tt, q)
    if (checker["profile"] != PROFILE or checker["search_conditions"] != search_conditions(horizon, tt, q)
            or _sha(registration["profile_sha256"]) != wire_digest(profile)):
        raise ValueError("registered fixed CPU profile/conditions mismatch")
    source = _fields(_json(source_raw), ("schema", "source", "profile", "profile_sha256", "search_version",
                                        "search_conditions", "value_identity", "search_implementation_sha256",
                                        "value_semantics_sha256"), "actual registered checker source facts")
    identity = {"semantics": SEMANTICS, "weights_sha256": None, "training": {"kind": "bootstrap"}}
    if (source["schema"] != SOURCE_SCHEMA or canonical_wire(source["source"]) != canonical_wire(checker["source"])
            or canonical_wire(source["profile"]) != canonical_wire(profile)
            or source["profile_sha256"] != registration["profile_sha256"] or source["search_version"] != SEARCH_VERSION
            or source["search_conditions"] != checker["search_conditions"] or canonical_wire(source["value_identity"]) != canonical_wire(identity)
            or source["search_implementation_sha256"] != checker["search_implementation_sha256"]
            or source["value_semantics_sha256"] != checker["value_semantics_sha256"]):
        raise ValueError("registered checker source facts mismatch")
    if checker["source"]["evaluator_configuration_sha256"] != wire_digest([SOURCE_SCHEMA, {key: source[key] for key in source if key != "source"}]):
        raise ValueError("registered checker configuration digest mismatch")
    platform, scope = registration["platform"], registration["binary_pin_scope"]
    if ((platform == "linux" and scope != "linux_loaded_executable_inode")
            or (platform in ("windows", "macos") and scope != "current_exe_path_hash")
            or platform not in ("linux", "windows", "macos")):
        raise ValueError("unsupported platform/binary assurance; dispatcher argument is not a child launch")
    return registration


def _parent_context(parents, parent_artifacts, body):
    if not isinstance(parents, ValidatedDataset) or parents.frozen_admission is None or parents._frozen_admission_identity is None:
        raise ValueError("comparative receipts require the existing strict frozen parent admission")
    parents._verify_raw_integrity()
    admission = parents.frozen_admission
    names = ("receipt.json", "producer-roster.json", "producer-envelope.json", "producer-prepared.jsonl",
             "inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl")
    _fields(parent_artifacts, names, "strict parent artifacts for comparative anchors")
    total = 0
    for name in names:
        raw = parent_artifacts[name]
        pin = admission["receipt"] if name == "receipt.json" else admission["artifacts"][name]
        _actual(raw, pin, MAX_BYTES)
        total += len(raw)
    if total > MAX_BYTES:
        raise ValueError("comparative parent artifact aggregate byte limit")
    roster = frozen.load_roster(parent_artifacts["producer-roster.json"])
    envelope = frozen.load_envelope(parent_artifacts["producer-envelope.json"])
    parent = {"receipt_artifact": admission["receipt"], "raw_dataset_sha256": admission["raw_dataset_sha256"],
              "split_sha256": admission["split_sha256"], "current_view_sha256": parents.current_view.sha256,
              "producer_roster_sha256": roster["sha256"], "producer_envelope_sha256": envelope["sha256"]}
    if parent != body["parent"]:
        raise ValueError("comparative actual strict parent pins mismatch")
    current = {}
    for index in parents.current_view.current_indices:
        row, snapshot = parents.records[index], parents.records[index]["input"]["snapshot"]
        if snapshot["role"] not in ("proposer", "critic") or len(snapshot["legal_moves"]) < 2:
            continue
        anchor = {"input_sha256": row["input"]["sha256"], "label_sha256": label_digest(row),
                  **{name: snapshot[name] for name in ("game_id", "role", "rules_state_sha256", "rules_history_sha256",
                     "encoding_sha256", "source", "frozen_epoch", "input_revision", "legal_moves")},
                  "side_to_move": "white" if snapshot["white_to_move"] else "black"}
        current[anchor["input_sha256"]] = (index, anchor)
    journals = {}
    for raw in parent_artifacts["producer-prepared.jsonl"].splitlines():
        journal = _unique_json(raw.decode("utf-8"))  # already checked by strict loader
        journals[journal["sha256"]] = journal["prepared"]
    rows = {}
    for name in ("inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl"):
        rows[name] = {tuple(byte_pin(raw).values()): raw for raw in parent_artifacts[name].splitlines()}
    admitted = {entry["binding"]["input_sha256"]: entry for entry in admission["inputs"]}
    actual = []
    for prepared in body["prepared_inputs"]:
        identity = prepared["current"]["input_sha256"]
        if identity not in current or prepared["current"] != current[identity][1] or identity not in admitted:
            raise ValueError("comparative current label/prepared input differs from strict current leaf")
        entry = admitted[identity]
        if (prepared["producer_id"] != entry["binding"]["producer_id"]
                or prepared["producer_registration_sha256"] != entry["producer"]["registration_sha256"]
                or prepared["capture_evidence_sha256"] != entry["prepared_evidence_sha256"]):
            raise ValueError("comparative actual prepared producer attribution mismatch")
        journal = journals[entry["prepared_evidence_sha256"]]
        item = {"input_sha256": identity}
        for pin_name, name in (("input_json", "inputs.jsonl"), ("tensor_sidecar_json", "native-inputs.jsonl"),
                               ("lineage_json", "input-lineage.jsonl")):
            if prepared[pin_name] != journal[pin_name]:
                raise ValueError("comparative prepared anchor differs from strict actual journal")
            key = tuple(journal[pin_name][name] for name in ("bytes", "sha256"))
            if key not in rows[name]:
                raise ValueError("comparative missing exact prepared row bytes")
            item[pin_name] = rows[name][key]
        actual.append(item)
    return current, actual


def _request(raw, current, candidate, task_id, checker, registration, plan_sha):
    if not 1 <= len(raw) <= 512 * 1024:
        raise ValueError("candidate request byte limit")
    value = _fields(_json(raw), ("schema", "task_id", "captured_input_sha256", "checker_namespace_sha256",
                               "before_result_anchor_sha256", "frozen_epoch", "input_revision", "position_command",
                               "expected_board_fen", "rules_state_sha256", "rules_history_sha256", "expected_legal_moves",
                               "white_to_move", "candidate", "cpu_binary_sha256", "cpu_profile_sha256", "horizon",
                               "node_budget", "tt_entries", "quiescence_ply", "max_wall_time_ms", "max_output_bytes",
                               "context_sha256"), "actual candidate request")
    snapshot = current["input"]["snapshot"]
    expected = {"schema": REQUEST_SCHEMA, "task_id": task_id, "captured_input_sha256": current["input"]["sha256"],
                "checker_namespace_sha256": metadata.checker_namespace(checker), "before_result_anchor_sha256": plan_sha,
                **{name: snapshot[name] for name in ("frozen_epoch", "input_revision", "position_command", "rules_state_sha256",
                   "rules_history_sha256", "white_to_move")}, "expected_board_fen": snapshot["board_fen"],
                "expected_legal_moves": snapshot["legal_moves"], "candidate": candidate,
                "cpu_binary_sha256": checker["source"]["cpu_binary_sha256"], "cpu_profile_sha256": registration["profile_sha256"],
                "horizon": checker["horizon"], "node_budget": checker["node_budget"],
                "tt_entries": registration["tt_entries"], "quiescence_ply": registration["quiescence_ply"]}
    if any(type(value[name]) is not type(wanted) or value[name] != wanted for name, wanted in expected.items()):
        raise ValueError("actual candidate request differs from captured input/checker/pre-result plan")
    for name, maximum, minimum in (("max_wall_time_ms", 300000, 1), ("max_output_bytes", 1048576, 1024)):
        if _uint(value[name], maximum) < minimum:
            raise ValueError("candidate request finite resource bounds")
    if not isinstance(task_id, str) or not 1 <= len(task_id) <= 128 or any(not 33 <= ord(c) <= 126 for c in task_id):
        raise ValueError("candidate request task ID extent")
    if _sha(value["context_sha256"]) != wire_digest([REQUEST_SCHEMA, {key: item for key, item in value.items() if key != "context_sha256"}]):
        raise ValueError("actual candidate request context mismatch")
    return value


def _receipt(raw, request, checker, registration, *, caller_elapsed_ms):
    receipt = _fields(_json(raw), ("schema", "task_id", "captured_input_sha256", "checker_namespace_sha256",
                                 "before_result_anchor_sha256", "frozen_epoch", "input_revision", "context_sha256",
                                 "cpu_binary_sha256", "binary_pin_scope", "caller_registration_scope", "status", "cpu_calls",
                                 "fresh_engine", "elapsed_ms", "deadline_exceeded", "product_verifier_enabled", "report"),
                      "actual candidate receipt")
    for name in ("schema", "task_id", "captured_input_sha256", "checker_namespace_sha256", "before_result_anchor_sha256",
                 "frozen_epoch", "input_revision", "context_sha256", "cpu_binary_sha256"):
        if type(receipt[name]) is not type(request[name]) or receipt[name] != request[name]:
            raise ValueError("candidate receipt request identity mismatch")
    if (receipt["binary_pin_scope"] != registration["binary_pin_scope"]
            or receipt["caller_registration_scope"] != "declared_not_independently_verified"
            or _uint(receipt["cpu_calls"], 1) != 1 or receipt["fresh_engine"] is not True
            or receipt["product_verifier_enabled"] is not False or type(receipt["deadline_exceeded"]) is not bool
            or receipt["status"] not in ("completed", "partial", "canceled")):
        raise ValueError("candidate receipt actual binary/isolated checker scope mismatch")
    elapsed = _uint(receipt["elapsed_ms"])
    if elapsed > caller_elapsed_ms:
        raise ValueError("caller launch elapsed time excludes total CLI receipt work")
    report = _fields(receipt["report"], ("conditions_sha256", "conditions", "raw_score", "best_move", "pv", "score_scope",
                                        "completion", "terminal_reason", "completed_depth", "requested_depth", "nodes",
                                        "quiescence_nodes", "tt_hits", "elapsed_ms", "reused_completed_depth", "root_restricted",
                                        "score_provenance", "pv_rules_validated"), "actual candidate raw report")
    conditions = _fields(report["conditions"], ("schema", "rules_state_sha256", "rules_history_sha256", "board_fen", "legal_moves",
                                               "legal_order_sha256", "history_completeness", "white_to_move", "profile_sha256",
                                               "profile", "value_identity", "search_version", "search_conditions", "horizon", "node_budget",
                                               "tt_entries", "quiescence_ply", "root_moves", "restriction", "perspective", "resource_policy"),
                         "actual candidate conditions")
    expected = {"schema": CONDITIONS_SCHEMA,
                **{name: request[name] for name in ("rules_state_sha256", "rules_history_sha256", "white_to_move", "horizon",
                   "node_budget", "tt_entries", "quiescence_ply")}, "board_fen": request["expected_board_fen"],
                "legal_moves": request["expected_legal_moves"], "legal_order_sha256": wire_digest([LEGAL_DOMAIN, request["expected_legal_moves"]]),
                "profile_sha256": request["cpu_profile_sha256"], "profile": PROFILE,
                "value_identity": {"semantics": SEMANTICS, "weights_sha256": None, "training": {"kind": "bootstrap"}},
                "search_version": SEARCH_VERSION, "search_conditions": checker["search_conditions"],
                "root_moves": [request["candidate"]], "restriction": "candidate_only", "perspective": "captured_side_to_move",
                "resource_policy": {"max_wall_time_ms": request["max_wall_time_ms"], "max_checks": 1,
                                    "search_deadline_reserve_ms": min(request["max_wall_time_ms"] // 10, 1000)}}
    # Canonical equality is type-sensitive throughout nested trees (True != 1).
    if any(canonical_wire(conditions[name]) != canonical_wire(wanted) for name, wanted in expected.items()):
        raise ValueError("candidate actual shared conditions/root/profile mismatch")
    if (conditions["history_completeness"] not in ("complete", "unknown_prefix")
            or _sha(report["conditions_sha256"]) != wire_digest(conditions)):
        raise ValueError("candidate exact conditions seal/history mismatch")
    for name in ("completed_depth", "requested_depth", "reused_completed_depth"):
        _uint(report[name], 64)
    nodes = _uint(report["nodes"], request["node_budget"])
    if (_uint(report["quiescence_nodes"], nodes) > nodes or _uint(report["tt_hits"], nodes) > nodes
            or _uint(report["elapsed_ms"]) > elapsed or report["requested_depth"] != request["horizon"]
            or report["completed_depth"] > request["horizon"] or report["reused_completed_depth"] != 0
            or report["root_restricted"] is not True or report["pv_rules_validated"] is not True
            or report["score_provenance"] != SEMANTICS):
        raise ValueError("candidate actual work/depth/PV/score provenance mismatch")
    score, pv = report["raw_score"], report["pv"]
    if type(score) is not int or not -(1 << 31) <= score < (1 << 31):
        raise ValueError("candidate raw score i32 required")
    if not isinstance(pv, list) or not 1 <= len(pv) <= request["horizon"] + request["quiescence_ply"]:
        raise ValueError("candidate exact PV extent")
    for move in pv:
        _uint(move, 65535)
    if _uint(report["best_move"], 65535) != request["candidate"] or pv[0] != request["candidate"]:
        raise ValueError("candidate Rules-validated PV/root mismatch")
    if report["score_scope"] not in ("completed_iteration", "frontier_only", "rules_terminal") or report["completion"] not in (
            "depth_limit", "node_limit", "deadline", "canceled", "quiescence_limit", "rules_terminal"):
        raise ValueError("candidate raw completion enum")
    if report["terminal_reason"] is not None and not isinstance(report["terminal_reason"], str):
        raise ValueError("candidate terminal reason")
    if report["completion"] == "rules_terminal" or report["score_scope"] == "rules_terminal" or report["terminal_reason"] is not None:
        return report, "terminal"
    complete = (report["completion"] == "depth_limit" and report["completed_depth"] == request["horizon"])
    if (receipt["status"] == "completed") != complete or (receipt["status"] == "canceled") != (report["completion"] == "canceled"):
        raise ValueError("candidate actual full completion/status mismatch")
    if complete and (report["score_scope"] != "completed_iteration" or nodes == 0):
        raise ValueError("candidate completed finite-depth report lacks real completed work")
    if receipt["deadline_exceeded"] or elapsed >= request["max_wall_time_ms"]:
        return report, "deadline"
    return report, None if complete else receipt["status"]


@dataclass(frozen=True)
class PairTarget:
    pair_id: str
    input_sha256: str
    row_index: int
    role: str
    split: str
    candidates: tuple
    sign: int  # +1 prefers left; -1 prefers right; 0 always masked.
    mask: bool
    reason: str
    receipt_sha256: tuple


class CheckedComparativePairs:
    """Conditional caller-observed receipt admission, never metadata promotion."""
    def __init__(self, parents, targets, admission, executions, *, _capability=None):
        if _capability is not _ADMISSION_CAPABILITY:
            raise ValueError("comparative receipt admission requires the checked adapter factory")
        self.parents, self.targets = parents, tuple(targets)
        self.admission = copy.deepcopy(admission)
        self._executions = copy.deepcopy(executions)
        self._identity = wire_digest([ADMISSION_SCHEMA, self.admission, [target.__dict__ | {"candidates": list(target.candidates),
                                                    "receipt_sha256": list(target.receipt_sha256)} for target in self.targets]])

    def verify(self):
        self.parents._verify_raw_integrity()
        actual = {task: {name: None if raw is None else byte_pin(raw) for name, raw in buffers.items()}
                  for task, buffers in self._executions.items()}
        if actual != self.admission["preserved_executions"]:
            raise ValueError("comparative preserved raw execution bytes changed")
        identity = wire_digest([ADMISSION_SCHEMA, self.admission, [target.__dict__ | {"candidates": list(target.candidates),
                                               "receipt_sha256": list(target.receipt_sha256)} for target in self.targets]])
        if identity != self._identity or self.parents._frozen_admission_identity != self.admission["strict_parent_admission_sha256"]:
            raise ValueError("comparative receipt admission/targets/parent changed")

    def raw_executions(self):
        """All immutable raw request/stdout/stderr/observation bytes, even failures."""
        self.verify()
        return copy.deepcopy(self._executions)

    def collate(self, indices, *, role, split="train"):
        self.verify()
        if role not in ("proposer", "critic") or not isinstance(indices, (list, tuple)) or not 1 <= len(indices) <= 256:
            raise ValueError("comparative batch role/extent")
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate comparative batch pair")
        selected = []
        for index in indices:
            target = self.targets[_uint(index, len(self.targets) - 1)]
            if target.role != role or target.split != split:
                raise ValueError("comparative pair role/split mismatch")
            selected.append(target)
        # Repeated input for different pairs has one ordinary frozen forward.
        rows = list(dict.fromkeys(target.row_index for target in selected))
        batch = self.parents.collate(rows, role, split=split, device="cpu")
        slots = tuple(rows.index(target.row_index) for target in selected)
        positions = tuple(tuple(self.parents.records[target.row_index]["input"]["snapshot"]["legal_moves"].index(candidate)
                                for candidate in target.candidates) for target in selected)
        return ComparativeBatch(batch, slots, positions, tuple(target.sign for target in selected),
                                tuple(target.mask for target in selected), tuple(target.pair_id for target in selected))


@dataclass(frozen=True)
class ComparativeBatch:
    base: object
    row_slots: tuple
    candidate_slots: tuple
    signs: tuple
    mask: tuple
    pair_ids: tuple


def admit_candidate_pairs(*, parents, parent_artifacts, overlay_bytes, expected_metadata_pins,
                          criterion_bytes, registration_bytes, source_bytes, binary_bytes, plan_bytes,
                          executions, independent_pins):
    """Reject identity mismatches; preserve failed/unknown evidence as masked.

    ``independent_pins`` contains overlay/criterion/registration/source/binary/
    plan byte pins and an executions map of request/receipt(stderr may empty)/
    stderr/launch_observation pins. No pin is inferred from an untrusted child.
    The caller must have independently observed each launch and prior durable
    plan/criterion publication; this function only checks that recorded scope.
    ``executions`` maps task IDs to those four actual byte buffers (receipt may
    be None). Synthetic fixtures exercise wiring, not that caller assurance.
    """
    _fields(independent_pins, ("overlay", "criterion", "registration", "source", "binary", "plan", "executions"),
            "independent actual comparative pins")
    total = sum(len(raw) for raw in (overlay_bytes, criterion_bytes, registration_bytes, source_bytes, binary_bytes, plan_bytes)
                if isinstance(raw, bytes))
    if total > MAX_BYTES:
        raise ValueError("candidate aggregate input byte budget")
    _actual(overlay_bytes, independent_pins["overlay"])
    body = metadata.load_overlay(overlay_bytes)["overlay"]
    current, actual_prepared = _parent_context(parents, parent_artifacts, body)
    audit = metadata.audit_metadata(overlay_bytes=overlay_bytes, expected_pins=expected_metadata_pins,
                                    checked_current_anchors=[value[1] for value in current.values()],
                                    independent_current_view_sha256=parents.current_view.sha256,
                                    actual_parent_receipt=parent_artifacts["receipt.json"], actual_prepared=actual_prepared)
    criterion = _criterion(criterion_bytes, independent_pins["criterion"])
    if body["criterion"] != {"recipe_id": criterion["recipe_id"], "recipe_sha256": independent_pins["criterion"]["sha256"]}:
        raise ValueError("overlay criterion differs from actual independently fixed bytes")
    registration = _registered_checker(registration_bytes, source_bytes, binary_bytes, independent_pins, body["checker"])
    _actual(plan_bytes, independent_pins["plan"])
    plan = _fields(_json(plan_bytes), ("schema", "parent", "criterion", "checker_namespace_sha256", "prepared_inputs", "pairs"),
                   "before-result candidate pair plan")
    expected_pairs = [{name: pair[name] for name in ("pair_id", "input_sha256", "kind", "candidates")}
                      | {"task_ids": [check["task_id"] for check in pair["checks"]]} for pair in body["pairs"]]
    if (plan["schema"] != PLAN_SCHEMA or canonical_wire(plan["parent"]) != canonical_wire(body["parent"])
            or canonical_wire(plan["criterion"]) != canonical_wire(body["criterion"])
            or plan["checker_namespace_sha256"] != metadata.checker_namespace(body["checker"])
            or canonical_wire(plan["prepared_inputs"]) != canonical_wire(body["prepared_inputs"])
            or canonical_wire(plan["pairs"]) != canonical_wire(expected_pairs)):
        raise ValueError("before-result plan differs from actual current anchors/fixed criterion/pair requests")
    if not isinstance(executions, dict) or not isinstance(independent_pins["executions"], dict):
        raise ValueError("independent candidate execution map required")
    tasks = [task for pair in expected_pairs for task in pair["task_ids"]]
    if len(set(tasks)) != len(tasks) or set(tasks) != set(executions) or set(tasks) != set(independent_pins["executions"]):
        raise ValueError("candidate tasks must be unique with exact observed execution set")
    reports, preserved, requests = {}, {}, {}
    for pair in body["pairs"]:
        row = parents.records[current[pair["input_sha256"]][0]]
        for candidate, check in zip(pair["candidates"], pair["checks"]):
            task = check["task_id"]
            execution = _fields(executions[task], ("request", "receipt", "stderr", "launch_observation"), "actual candidate execution")
            pins = _fields(independent_pins["executions"][task], ("request", "receipt", "stderr", "launch_observation"),
                           "independently observed candidate execution pins")
            for name in ("request", "stderr", "launch_observation"):
                _actual(execution[name], pins[name], empty=name == "stderr")
                total += len(execution[name])
            if (execution["receipt"] is None) != (pins["receipt"] is None):
                raise ValueError("missing candidate receipt actual pin mismatch")
            if execution["receipt"] is not None:
                _actual(execution["receipt"], pins["receipt"])
                total += len(execution["receipt"])
            if total > MAX_BYTES:
                raise ValueError("candidate aggregate actual execution byte budget")
            request = _request(execution["request"], row, candidate, task, body["checker"], registration,
                               independent_pins["plan"]["sha256"])
            requests[task] = request
            observation = _fields(_json(execution["launch_observation"]), ("schema", "task_id", "registration_sha256",
                                 "binary_sha256", "platform", "binary_pin_scope", "before_result_anchor_sha256",
                                 "request", "receipt", "stderr", "assurance_scope", "anchor_durable_before_spawn",
                                 "criterion_fixed_before_spawn", "spawned", "reaped", "exit_code", "elapsed_ms", "timed_out"),
                                 "independently pinned caller launch observation")
            if (observation["schema"] != LAUNCH_SCHEMA or observation["task_id"] != task
                    or observation["registration_sha256"] != independent_pins["registration"]["sha256"]
                    or observation["binary_sha256"] != body["checker"]["source"]["cpu_binary_sha256"]
                    or observation["platform"] != registration["platform"] or observation["binary_pin_scope"] != registration["binary_pin_scope"]
                    or observation["before_result_anchor_sha256"] != independent_pins["plan"]["sha256"]
                    or any(canonical_wire(observation[name]) != canonical_wire(pins[name]) for name in ("request", "receipt", "stderr"))
                    or observation["assurance_scope"] != "independently_pinned_caller_observation"
                    or observation["anchor_durable_before_spawn"] is not True or observation["criterion_fixed_before_spawn"] is not True):
                raise ValueError("caller-observed launch/pre-result/binary scope mismatch")
            for flag in ("spawned", "reaped", "timed_out"):
                if type(observation[flag]) is not bool:
                    raise ValueError("caller launch observation flags")
            elapsed = _uint(observation["elapsed_ms"])
            exit_code = observation["exit_code"]
            if exit_code is not None and (type(exit_code) is not int or not -(1 << 31) <= exit_code < (1 << 31)):
                raise ValueError("caller child exit status")
            if len(execution["stderr"]) > request["max_output_bytes"] or (execution["receipt"] is not None
                    and len(execution["receipt"]) > request["max_output_bytes"]):
                raise ValueError("actual candidate output cap mismatch")
            reason = None
            if not observation["spawned"] or not observation["reaped"] or exit_code != 0:
                reason = "launch_failed"
            elif observation["timed_out"] or elapsed >= request["max_wall_time_ms"]:
                reason = "deadline"
            if execution["receipt"] is None:
                report, reason = None, reason or "missing"
            elif reason is None:
                report, reason = _receipt(execution["receipt"], request, body["checker"], registration,
                                          caller_elapsed_ms=elapsed)
            else:
                report = None  # Preserve failed raw bytes; do not reinterpret error stdout as success.
            if reason is None:
                if check["completion"] != "completed" or check["completed_depth"] != request["horizon"]:
                    raise ValueError("overlay completion differs from actual complete candidate receipt")
                if check["evidence_artifact"] != pins["receipt"] or check["evidence_id"] != task:
                    raise ValueError("overlay evidence differs from independently observed raw receipt bytes")
            else:
                if pair["preference"] != "masked":
                    raise ValueError("unresolved actual candidate pair must remain metadata masked")
                if report is not None and reason in ("partial", "canceled"):
                    completion = "cancelled" if reason == "canceled" else "partial"
                    if check["completion"] != completion or check["completed_depth"] != report["completed_depth"]:
                        raise ValueError("masked overlay completion differs from actual candidate partial report")
                if check["evidence_artifact"] is not None and check["evidence_artifact"] != pins["receipt"]:
                    raise ValueError("masked overlay actual receipt pin mismatch")
            reports[task] = (report, reason)
            preserved[task] = copy.deepcopy(pins)
    targets = []
    for pair in body["pairs"]:
        checked = [reports[check["task_id"]] for check in pair["checks"]]
        reason = next((reason for _, reason in checked if reason is not None), None)
        sign = 0
        if reason is None:
            left, right = [report for report, _ in checked]
            common_requests = [{key: value for key, value in requests[check["task_id"]].items()
                                if key not in ("task_id", "candidate", "context_sha256")} for check in pair["checks"]]
            if canonical_wire(common_requests[0]) != canonical_wire(common_requests[1]):
                raise ValueError("actual candidate pair common request resource conditions mismatch")
            common_left = {key: value for key, value in left["conditions"].items() if key != "root_moves"}
            common_right = {key: value for key, value in right["conditions"].items() if key != "root_moves"}
            if canonical_wire(common_left) != canonical_wire(common_right):
                raise ValueError("actual candidate pair shared conditions mismatch")
            scores = left["raw_score"], right["raw_score"]
            if any(abs(score) >= criterion["mate_threshold"] for score in scores):
                reason = "mate_band"
            elif any(abs(score) > criterion["score_limit"] for score in scores):
                reason = "outside_ordinal_range"
            elif abs(scores[0] - scores[1]) < criterion["minimum_margin"]:
                reason = "tie_or_margin"
            else:
                sign = 1 if scores[0] > scores[1] else -1
        if sign and pair["preference"] != ("left" if sign == 1 else "right"):
            raise ValueError("overlay declared preference differs from actual independently fixed ordinal criterion")
        if not sign and pair["preference"] not in ("masked", "tie"):
            raise ValueError("ineligible ordinal pair must remain masked")
        index = current[pair["input_sha256"]][0]
        snapshot = parents.records[index]["input"]["snapshot"]
        targets.append(PairTarget(pair["pair_id"], pair["input_sha256"], index, snapshot["role"], parents.split[snapshot["game_id"]],
                                  tuple(pair["candidates"]), sign, bool(sign), reason or "fixed_finite_depth_ordinal",
                                  tuple(check["evidence_artifact"]["sha256"] if check["evidence_artifact"] else None for check in pair["checks"])))
    admission = {"schema": ADMISSION_SCHEMA, "scope": "caller_observed_owned_cpu_ordinal_receipts",
                 "caller_assurance": "independent_registration_and_launch_observation_required_not_inferred",
                 "rules_validation": "registered_rust_receipt_pv_attestation_not_python_rules_replay",
                 "metadata_audit": audit, "strict_parent_admission_sha256": parents._frozen_admission_identity,
                 "current_view_sha256": parents.current_view.sha256, "independent_pins": copy.deepcopy(independent_pins),
                 "preserved_executions": preserved, "binary_pin_scope": registration["binary_pin_scope"],
                 "platform": registration["platform"], "actual_training_executed": False,
                 "ordinary_policy_wdl_unchanged": True, "critic_repair_validity_admitted": False,
                 "divergence_adapter_admitted": False}
    result = CheckedComparativePairs(parents, targets, admission, executions, _capability=_ADMISSION_CAPABILITY)
    result.verify()
    return result


def load_checked_comparative_pairs(directory, *, strict_parent_options, **candidate_arguments):
    """Reload raw collector bytes through the existing required strict loader.

    Missing or failed strict evidence has no legacy fallback. Candidate evidence
    bytes and independent pins are supplied explicitly by the caller/launcher.
    They may be read from a persisted bank using a bounded reader; no filename
    discovery or execution occurs in this adapter.
    """
    parents = load_frozen_collected_dataset(directory, **strict_parent_options)
    reader = _CollectionArtifactReader(MAX_BYTES)
    root = Path(directory)
    names = ("receipt.json", "producer-roster.json", "producer-envelope.json", "producer-prepared.jsonl",
             "inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl")
    raw = {name: reader.read(root / name, parents.frozen_admission["receipt"] if name == "receipt.json"
                            else parents.frozen_admission["artifacts"][name]) for name in names}
    return admit_candidate_pairs(parents=parents, parent_artifacts=raw, **candidate_arguments)


def load_persisted_comparative_pairs(collection, evidence_directory, *, expected_bank_sha256,
                                    strict_parent_options, expected_metadata_pins, independent_pins):
    """Bounded reload of an explicitly pinned bank; never discovery or launch.

    The owner supplies both the bank SHA and independent asset/launch pins.
    A bank manifest cannot make its own declared observations independent.
    All files are direct children; registered-path rules reject secret names,
    links/junctions, and the reader rejects nonregular/changing/oversized files.
    The manifest contains filenames only, not derived targets or replacement
    raw/input/current seals. Actual targets are recomputed from the receipts.
    """
    from .preparation_check import _registered_path
    _sha(expected_bank_sha256)
    root = _registered_path(evidence_directory)
    reader = _CollectionArtifactReader(MAX_BYTES)
    manifest_raw = reader.read(root / "candidate-bank.json", maximum=MAX_JSON)
    if byte_pin(manifest_raw)["sha256"] != expected_bank_sha256:
        raise ValueError("candidate persisted bank differs from independent pin")
    manifest = _fields(_json(manifest_raw), ("schema", "files", "executions"), "persisted comparative bank")
    if manifest["schema"] != BANK_SCHEMA:
        raise ValueError("candidate persisted bank schema")
    _fields(independent_pins, ("overlay", "criterion", "registration", "source", "binary", "plan", "executions"),
            "independent persisted candidate pins")
    keys = ("overlay", "criterion", "registration", "source", "binary", "plan")
    files = _fields(manifest["files"], keys, "persisted candidate asset filenames")
    used = set()

    def read(name, pin, maximum=MAX_JSON, *, empty=False):
        if not isinstance(name, str) or not name or name in used or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("candidate persisted filenames must be unique direct children")
        used.add(name)
        path = _registered_path(root / name)
        if path.parent != root:
            raise ValueError("candidate persisted file escaped the registered bank")
        data = reader.read(path, pin, maximum=maximum)
        _actual(data, pin, maximum, empty=empty)
        return data

    raw = {key + "_bytes": read(files[key], independent_pins[key], MAX_BYTES if key == "binary" else MAX_JSON) for key in keys}
    if (not isinstance(manifest["executions"], dict) or not isinstance(independent_pins["executions"], dict)
            or set(manifest["executions"]) != set(independent_pins["executions"])
            or not 1 <= len(manifest["executions"]) <= metadata.MAX_PAIRS * 2):
        raise ValueError("candidate persisted observed execution extent/set")
    executions = {}
    for task, names in manifest["executions"].items():
        names = _fields(names, ("request", "receipt", "stderr", "launch_observation"), "persisted actual execution filenames")
        pins = _fields(independent_pins["executions"][task], tuple(names), "persisted independent execution pins")
        if (names["receipt"] is None) != (pins["receipt"] is None):
            raise ValueError("persisted missing receipt must remain explicit")
        executions[task] = {name: None if names[name] is None and name == "receipt"
                            else read(names[name], pins[name], empty=name == "stderr") for name in names}
    return load_checked_comparative_pairs(collection, strict_parent_options=strict_parent_options,
                                         expected_metadata_pins=expected_metadata_pins, independent_pins=independent_pins,
                                         executions=executions, **raw)


def pairwise_softplus_loss(candidate_logits, batch):
    """Separate masked ordinal loss; inference-only CPU FP32, no backward."""
    import torch
    from torch.nn import functional as functional
    if (not isinstance(batch, ComparativeBatch) or not isinstance(candidate_logits, torch.Tensor)
            or candidate_logits.device.type != "cpu" or candidate_logits.dtype != torch.float32
            or candidate_logits.requires_grad or candidate_logits.grad_fn is not None
            or candidate_logits.ndim != 2 or candidate_logits.shape != batch.base.inputs.candidate_mask.shape
            or not torch.all(torch.isfinite(candidate_logits))):
        raise ValueError("pairwise loss requires finite frozen CPU FP32 candidate logits")
    if not (len(batch.row_slots) == len(batch.candidate_slots) == len(batch.signs) == len(batch.mask) == len(batch.pair_ids)):
        raise ValueError("comparative batch alignment")
    with torch.inference_mode():
        losses = []
        for row, candidates, sign, active in zip(batch.row_slots, batch.candidate_slots, batch.signs, batch.mask):
            _uint(row, candidate_logits.shape[0] - 1)
            if type(active) is not bool or type(sign) is not int or sign not in (-1, 0, 1) or active != (sign != 0):
                raise ValueError("comparative ordinal mask/sign mismatch")
            if len(candidates) != 2 or candidates[0] == candidates[1]:
                raise ValueError("comparative candidate slots")
            for candidate in candidates:
                _uint(candidate, candidate_logits.shape[1] - 1)
                if not bool(batch.base.inputs.candidate_mask[row, candidate]):
                    raise ValueError("pairwise loss candidate outside captured legal mask")
            if active:
                margin = candidate_logits[row, candidates[0]] - candidate_logits[row, candidates[1]]
                losses.append(functional.softplus(-sign * margin))
        loss = torch.stack(losses).mean() if losses else torch.zeros((), dtype=torch.float32)
        if not bool(torch.isfinite(loss)):
            raise ValueError("pairwise loss became nonfinite")
        return loss


def frozen_pairwise_preparation(model, checked, *, expected_parameter_sha256, checkpoint_sha256,
                                split="train", required_roles=("proposer", "critic"), max_pairs=4096,
                                max_wall_time_ms=300000):
    """Forward/loss after caller checkpoint reload; never a training step.

    The caller must verify the checkpoint's actual bytes/metadata and supply the
    independently registered parameter digest of that loaded model. This helper
    pins its finite CPU FP32 state before/after, and requires every requested
    role to have a nonmasked target and finite nonzero loss. An all-masked bank
    is not accepted as positive preparation. Original preparation gates stay
    independent. This is not a checkpoint loader or actual launcher.
    """
    import torch
    from .preparation_check import _parameter_digest
    _sha(expected_parameter_sha256)
    _sha(checkpoint_sha256)
    if (not isinstance(checked, CheckedComparativePairs) or split not in ("train", "validation", "holdout")
            or not isinstance(required_roles, (tuple, list)) or not required_roles
            or len(set(required_roles)) != len(required_roles) or any(role not in ("proposer", "critic") for role in required_roles)
            or not 1 <= _uint(max_pairs, 4096) or not 1 <= _uint(max_wall_time_ms, 300000)):
        raise ValueError("finite comparative preparation role/split/pair/time limits")
    if model.training or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("comparative preparation requires caller-frozen eval model with no gradients")
    checked.verify()
    started = time.monotonic()
    before = _parameter_digest(model)
    if before != expected_parameter_sha256:
        raise ValueError("loaded frozen parameter bytes differ from independent pin")
    chosen = [index for index, target in enumerate(checked.targets) if target.role in required_roles and target.split == split]
    if len(chosen) > max_pairs:
        raise ValueError("comparative preparation pair budget exceeded")
    totals, known, consumed = {}, {}, []
    with torch.inference_mode():
        for role in required_roles:
            selected = [index for index in chosen if checked.targets[index].role == role]
            count = sum(checked.targets[index].mask for index in selected)
            if count == 0:
                raise ValueError("all-masked or missing requested comparative role")
            total = 0.0
            for start in range(0, len(selected), 256):
                if (time.monotonic() - started) * 1000 >= max_wall_time_ms:
                    raise ValueError("comparative preparation original wall allowance expired")
                indices = selected[start:start + 256]
                batch = checked.collate(indices, role=role, split=split)
                memory = model.public_encoder(*batch.base.inputs.public_args())
                outputs = model.role_graph(role)(*batch.base.inputs.role_args(memory))
                loss = pairwise_softplus_loss(outputs[0], batch)
                active = sum(batch.mask)
                total += float(loss) * active
                consumed.extend(batch.pair_ids)
            value = total / count
            if not 0 < value < float("inf"):
                raise ValueError("positive comparative preparation requires finite nonzero ordinal loss")
            totals[role], known[role] = value, count
    after = _parameter_digest(model)
    checked.verify()
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("frozen comparative preparation changed model parameters/gradients")
    if (time.monotonic() - started) * 1000 >= max_wall_time_ms:
        raise ValueError("comparative preparation original wall allowance expired")
    return {"schema": PREPARATION_SCHEMA, "scope": "frozen_cpu_fp32_no_step_ordinal_loss",
            "checkpoint_sha256": checkpoint_sha256, "parameter_sha256_before": before, "parameter_sha256_after": after,
            "admission_sha256": checked._identity, "current_view_sha256": checked.parents.current_view.sha256,
            "consumed_pair_ids": consumed, "known_pairs": known, "pairwise_losses": totals,
            "actual_training_executed": False, "backward_executed": False, "optimizer_created": False,
            "gpu_executed": False, "ordinary_policy_wdl_unchanged": True,
            "caller_assurance": checked.admission["caller_assurance"], "rules_validation": checked.admission["rules_validation"]}
