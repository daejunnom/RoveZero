"""F-owned envelope checks. Declared engine identities remain unverified.

This module validates persisted syntax and label correspondence. It does not
restore a chess position, generate legal moves, encode tensors or issue a shared
Rust contract revision. Those checks require the I/A/C implementations.
"""

from datetime import datetime
import math
import re

from .errors import DataError
from .serialization import canonical_bytes, digest

SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "holdout")
MOVE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _json_boundary(value: object, context: str) -> None:
    try:
        canonical_bytes(value)
    except DataError as exc:
        raise DataError(exc.code, context + "." + exc.context, exc.message) from exc


def obj(value: object, context: str, fields: tuple[str, ...]) -> dict:
    if not isinstance(value, dict):
        raise DataError("InvalidType", context, "expected an object")
    missing = set(fields) - value.keys()
    extra = value.keys() - set(fields)
    if missing or extra:
        raise DataError("InvalidFields", context,
                        f"missing={sorted(missing)}, unknown={sorted(extra)}")
    return value


def text(value: object, context: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise DataError("InvalidText", context, "expected nonempty text, max 2048 characters")


def choice(value: object, context: str, choices: tuple) -> None:
    if type(value) is not type(choices[0]) or value not in choices:
        raise DataError("InvalidEnum", context, f"expected one of {choices}")


def integer(value: object, context: str, *, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise DataError("InvalidInteger", context, f"expected integer in [{minimum}, 2^63-1]")


def number(value: object, context: str, *, minimum: float | None = None,
           maximum: float | None = None) -> None:
    if type(value) not in (int, float) or (type(value) is int and abs(value) > 2**63 - 1) or not math.isfinite(value):
        raise DataError("NonFinite", context, "expected a finite number, not bool")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise DataError("OutOfRange", context, f"expected range [{minimum}, {maximum}]")


def sha(value: object, context: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise DataError("InvalidDigest", context, "expected lowercase SHA-256 hex")


def array(value: object, context: str, *, maximum: int) -> list:
    if not isinstance(value, list) or len(value) > maximum:
        raise DataError("InvalidArray", context, f"expected array, max {maximum} entries")
    return value


def moves(value: object, context: str, *, maximum: int, unique: bool) -> list:
    items = array(value, context, maximum=maximum)
    for i, move in enumerate(items):
        if not isinstance(move, str) or MOVE.fullmatch(move) is None or move[:2] == move[2:4]:
            raise DataError("InvalidMoveSyntax", f"{context}[{i}]", "expected UCI move with explicit promotion")
    if unique and len(set(items)) != len(items):
        raise DataError("DuplicateMove", context, "ordered legal moves must be unique")
    return items


def rights(value: object, context: str) -> None:
    value = obj(value, context, ("status", "license", "reference"))
    choice(value["status"], context + ".status", ("confirmed", "unverified"))
    text(value["license"], context + ".license", nullable=True)
    text(value["reference"], context + ".reference", nullable=True)
    if value["status"] == "confirmed" and (value["license"] is None or value["reference"] is None):
        raise DataError("MissingRightsEvidence", context, "confirmed rights require license and reference")


def validate_manifest(value: object) -> dict:
    _json_boundary(value, "manifest")
    value = obj(value, "manifest", ("schema_version", "dataset_id", "kind",
                "engine_contract_revision", "execution_ready", "source", "teacher",
                "encoding", "probability_tolerance", "leakage_policy"))
    choice(value["schema_version"], "manifest.schema_version", (SCHEMA_VERSION,))
    text(value["dataset_id"], "manifest.dataset_id")
    choice(value["kind"], "manifest.kind", ("fixture", "teacher", "self_play"))
    text(value["engine_contract_revision"], "manifest.engine_contract_revision", nullable=True)
    if value["execution_ready"] is not False:
        raise DataError("EngineBindingUnavailable", "manifest.execution_ready",
                        "I/A/C bindings are unavailable; execution_ready must be false")
    source = obj(value["source"], "manifest.source", ("id", "uri", "digest", "rights"))
    text(source["id"], "manifest.source.id")
    text(source["uri"], "manifest.source.uri")
    sha(source["digest"], "manifest.source.digest")
    rights(source["rights"], "manifest.source.rights")
    teacher = obj(value["teacher"], "manifest.teacher", ("id", "version", "code_digest",
                  "binary_digest", "weights_digest", "search_id", "backend_id", "precision",
                  "device", "options", "budget", "rights"))
    for key in ("id", "version", "search_id", "backend_id", "precision", "device"):
        text(teacher[key], "manifest.teacher." + key)
    for key in ("code_digest", "binary_digest", "weights_digest"):
        sha(teacher[key], "manifest.teacher." + key)
    if not isinstance(teacher["options"], dict):
        raise DataError("InvalidType", "manifest.teacher.options", "expected an object")
    rights(teacher["rights"], "manifest.teacher.rights")
    budget = obj(teacher["budget"], "manifest.teacher.budget", ("max_positions", "max_nodes_per_position",
                 "max_time_ms_per_position", "workers", "max_memory_bytes"))
    for key, item in budget.items():
        integer(item, "manifest.teacher.budget." + key, minimum=1)
    encoding = obj(value["encoding"], "manifest.encoding", ("id", "version", "model_digest", "history_policy"))
    for key in ("id", "version", "history_policy"):
        text(encoding[key], "manifest.encoding." + key)
    sha(encoding["model_digest"], "manifest.encoding.model_digest")
    number(value["probability_tolerance"], "manifest.probability_tolerance", minimum=0, maximum=1e-3)
    policy = obj(value["leakage_policy"], "manifest.leakage_policy",
                 ("near_duplicate_key", "key_producer", "pretraining_overlap", "limitations", "holdout_usage"))
    choice(policy["near_duplicate_key"], "manifest.leakage_policy.near_duplicate_key", ("opening_family_id",))
    text(policy["key_producer"], "manifest.leakage_policy.key_producer")
    choice(policy["pretraining_overlap"], "manifest.leakage_policy.pretraining_overlap", ("unknown", "audited"))
    text(policy["limitations"], "manifest.leakage_policy.limitations")
    choice(policy["holdout_usage"], "manifest.leakage_policy.holdout_usage", ("final_evaluation_only",))
    return value


def _distribution(value: object, context: str, tolerance: float, *, length: int) -> list:
    values = array(value, context, maximum=length)
    if len(values) != length:
        raise DataError("InvalidShape", context, f"expected {length} values")
    for i, item in enumerate(values):
        number(item, f"{context}[{i}]", minimum=0, maximum=1)
    if abs(math.fsum(values) - 1) > tolerance:
        raise DataError("InvalidNormalization", context, "probabilities must sum to one within declared tolerance")
    return values


def validate_record(value: object, manifest: dict) -> dict:
    """Return the unchanged record; validation never repairs or fabricates labels."""
    validate_manifest(manifest)
    _json_boundary(value, "record")
    value = obj(value, "record", ("schema_version", "dataset_id", "record_id", "game_id",
                "opening_family_id", "lineage_id", "seed_group_id", "parent_record_id",
                "augmentation_id", "split", "input_identity", "state", "teacher_id", "label",
                "game_result", "failure_groups"))
    choice(value["schema_version"], "record.schema_version", (SCHEMA_VERSION,))
    for key in ("dataset_id", "record_id", "game_id", "teacher_id"):
        text(value[key], "record." + key)
    for key in ("opening_family_id", "lineage_id", "seed_group_id", "parent_record_id", "augmentation_id"):
        text(value[key], "record." + key, nullable=True)
    choice(value["split"], "record.split", SPLITS)
    if value["dataset_id"] != manifest["dataset_id"] or value["teacher_id"] != manifest["teacher"]["id"]:
        raise DataError("IdentityMismatch", "record", "dataset or teacher does not match manifest")
    identity = obj(value["input_identity"], "record.input_identity", ("digest", "encoding_id", "encoding_version"))
    sha(identity["digest"], "record.input_identity.digest")
    if (identity["encoding_id"], identity["encoding_version"]) != (manifest["encoding"]["id"], manifest["encoding"]["version"]):
        raise DataError("EncodingMismatch", "record.input_identity", "encoding does not match manifest")
    state = obj(value["state"], "record.state", ("rules_id", "rules_version", "initial_fen", "moves",
                "history_origin", "history_completeness", "state_digest", "board_digest", "side_to_move",
                "play_status", "termination_reason", "legal_moves", "legal_moves_digest"))
    for key in ("rules_id", "rules_version", "initial_fen"):
        text(state[key], "record.state." + key)
    choice(state["history_origin"], "record.state.history_origin", ("startpos", "fen"))
    choice(state["history_completeness"], "record.state.history_completeness", ("complete", "unknown_prefix"))
    if state["history_origin"] == "startpos":
        if state["initial_fen"] != "startpos" or state["history_completeness"] != "complete":
            raise DataError("HistoryMismatch", "record.state", "startpos requires full known trace")
    elif state["initial_fen"] == "startpos" or state["history_completeness"] != "unknown_prefix":
        raise DataError("UnknownHistory", "record.state", "FEN prefix must remain unknown")
    moves(state["moves"], "record.state.moves", maximum=10000, unique=False)
    for key in ("state_digest", "board_digest", "legal_moves_digest"):
        sha(state[key], "record.state." + key)
    choice(state["side_to_move"], "record.state.side_to_move", ("white", "black"))
    choice(state["play_status"], "record.state.play_status", ("ongoing", "terminal"))
    text(state["termination_reason"], "record.state.termination_reason", nullable=True)
    legal = moves(state["legal_moves"], "record.state.legal_moves", maximum=512, unique=True)
    if digest(legal) != state["legal_moves_digest"]:
        raise DataError("LegalOrderMismatch", "record.state.legal_moves_digest", "ordered move digest mismatch")
    if state["play_status"] == "ongoing" and (not legal or state["termination_reason"] is not None):
        raise DataError("TerminalMismatch", "record.state", "ongoing requires legal moves and no termination reason")
    if state["play_status"] == "terminal" and state["termination_reason"] is None:
        raise DataError("TerminalMismatch", "record.state", "terminal requires a reason; Rules verification is pending")
    groups = array(value["failure_groups"], "record.failure_groups", maximum=64)
    for group in groups:
        text(group, "record.failure_groups")
    if len(set(groups)) != len(groups):
        raise DataError("DuplicateGroup", "record.failure_groups", "duplicate stratum")
    _label(value["label"], legal, manifest)
    _result(value["game_result"], manifest["probability_tolerance"])
    return value


def _label(value: object, legal: list, manifest: dict) -> None:
    value = obj(value, "record.label", ("status", "failure_reason", "generated_at", "policy", "value",
                "actual_cost", "selection_reason", "confidence"))
    choice(value["status"], "record.label.status", ("complete", "failed", "partial", "canceled"))
    text(value["failure_reason"], "record.label.failure_reason", nullable=True)
    text(value["selection_reason"], "record.label.selection_reason")
    text(value["generated_at"], "record.label.generated_at")
    try:
        when = datetime.fromisoformat(value["generated_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataError("InvalidTimestamp", "record.label.generated_at", "expected ISO-8601 timestamp") from exc
    if when.tzinfo is None:
        raise DataError("InvalidTimestamp", "record.label.generated_at", "timezone is required")
    complete = value["status"] == "complete"
    if complete and value["failure_reason"] is not None or not complete and value["failure_reason"] is None:
        raise DataError("CompletionMismatch", "record.label", "status and failure reason disagree")
    if not complete and (value["policy"] is not None or value["value"] is not None):
        raise DataError("InvalidFailureLabel", "record.label", "failed/partial/canceled targets must be absent")
    if complete and value["policy"] is None and value["value"] is None:
        raise DataError("MissingTarget", "record.label", "complete label needs at least one target")
    cost = obj(value["actual_cost"], "record.label.actual_cost", ("time_ms", "nodes", "evaluations", "peak_memory_bytes"))
    for key, item in cost.items():
        integer(item, "record.label.actual_cost." + key)
    budget = manifest["teacher"]["budget"]
    if complete and (cost["time_ms"] > budget["max_time_ms_per_position"] or cost["nodes"] > budget["max_nodes_per_position"] or cost["peak_memory_bytes"] > budget["max_memory_bytes"]):
        raise DataError("BudgetExceeded", "record.label.actual_cost", "successful label exceeds declared budget")
    if value["confidence"] is not None:
        confidence = obj(value["confidence"], "record.label.confidence", ("value", "source"))
        number(confidence["value"], "record.label.confidence.value", minimum=0, maximum=1)
        text(confidence["source"], "record.label.confidence.source")
    tolerance = manifest["probability_tolerance"]
    if value["policy"] is not None:
        policy = obj(value["policy"], "record.label.policy", ("kind", "source", "moves", "values", "temperature", "normalization"))
        choice(policy["kind"], "record.label.policy.kind", ("probabilities", "visits"))
        choice(policy["source"], "record.label.policy.source", ("raw_nn", "search"))
        number(policy["temperature"], "record.label.policy.temperature", minimum=1e-12)
        if policy["moves"] != legal:
            raise DataError("PolicyMoveMismatch", "record.label.policy.moves", "targets must name every move in exact legal order")
        if not legal:
            raise DataError("EmptyPolicy", "record.label.policy", "no policy normalization for empty legal moves")
        if policy["kind"] == "probabilities":
            choice(policy["normalization"], "record.label.policy.normalization", ("sum_one",))
            _distribution(policy["values"], "record.label.policy.values", tolerance, length=len(legal))
        else:
            choice(policy["source"], "record.label.policy.source", ("search",))
            choice(policy["normalization"], "record.label.policy.normalization", ("visits_unscaled",))
            values = array(policy["values"], "record.label.policy.values", maximum=len(legal))
            if len(values) != len(legal):
                raise DataError("InvalidShape", "record.label.policy.values", "visit count shape mismatch")
            for item in values:
                integer(item, "record.label.policy.values")
            if sum(values) == 0:
                raise DataError("EmptyVisits", "record.label.policy.values", "visits must have a positive total")
    if value["value"] is not None:
        target = obj(value["value"], "record.label.value", ("kind", "value", "viewpoint", "scale"))
        choice(target["kind"], "record.label.value.kind", ("wdl", "q", "centipawn"))
        choice(target["viewpoint"], "record.label.value.viewpoint", ("side_to_move",))
        scale = {"wdl": "probability", "q": "wdl_difference", "centipawn": "centipawn"}[target["kind"]]
        choice(target["scale"], "record.label.value.scale", (scale,))
        if target["kind"] == "wdl":
            _distribution(target["value"], "record.label.value.value", tolerance, length=3)
        elif target["kind"] == "q":
            number(target["value"], "record.label.value.value", minimum=-1, maximum=1)
        else:
            number(target["value"], "record.label.value.value")


def _result(value: object, tolerance: float) -> None:
    value = obj(value, "record.game_result", ("status", "wdl", "viewpoint", "termination_reason"))
    choice(value["status"], "record.game_result.status", ("rule_terminal", "engine_loss", "adjudicated", "incomplete"))
    choice(value["viewpoint"], "record.game_result.viewpoint", ("white", "black", "side_to_move"))
    text(value["termination_reason"], "record.game_result.termination_reason")
    if value["status"] == "incomplete":
        if value["wdl"] is not None:
            raise DataError("IncompleteResult", "record.game_result.wdl", "unfinished game has no outcome target")
    else:
        values = _distribution(value["wdl"], "record.game_result.wdl", tolerance, length=3)
        if any(item not in (0, 1) for item in values):
            raise DataError("InvalidGameOutcome", "record.game_result.wdl", "actual result must be one-hot")
