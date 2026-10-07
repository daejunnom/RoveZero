"""Float-free comparative metadata, with no torch/model/checker execution.

Independent expected pins and checked current anchors must come from the
existing strict consumer over the same immutable dataset. Supplied anchors are
only checked for metadata consistency here: this module does not implement
DG05 selection, base label/chain/source/split/encoding validation, live checker
proof, target production, collation, loss or positive training admission.
"""
import copy
import hashlib
import json

from . import frozen_producer as _frozen

COMPARATIVE_DOMAIN = "rz-pals-comparative-overlay/1"
CHECKER_DOMAIN = "rz-pals-comparative-checker/1"
MAX_PAIRS = 4096
MAX_ACTUAL_BYTES = 64 * 1024 * 1024
_fields, _uint, _sha, _identity = _frozen._fields, _frozen._uint, _frozen._sha, _frozen._identity


def canonical_metadata(domain, value):
    """Sorted compact UTF-8, retained arrays, strict u64/string/null scalars."""
    _identity(domain)
    raw = _frozen.canonical_metadata(domain, value)
    if len(raw) > _frozen.MAX_MANIFEST_BYTES:
        raise ValueError("comparative canonical byte limit")
    return raw


def metadata_digest(domain, value):
    return hashlib.sha256(canonical_metadata(domain, value)).hexdigest()


def query_boundary(kind):
    if kind in ("proposer_candidates", "critic_responses"):
        return "candidate_pair_metadata_only"
    if kind in ("divergence", "critic_divergences"):
        return "requires_auxiliary_adapter"
    raise ValueError("unsupported comparative query")


def _pin(value):
    _fields(value, ("bytes", "sha256"), "comparative actual byte pin")
    if not 1 <= _uint(value["bytes"], _frozen.MAX_MANIFEST_BYTES):
        raise ValueError("comparative actual byte limit")
    _sha(value["sha256"])


def _parent(value):
    _fields(value, ("receipt_artifact", "raw_dataset_sha256", "split_sha256", "current_view_sha256",
                    "producer_roster_sha256", "producer_envelope_sha256"), "comparative parent")
    _pin(value["receipt_artifact"])
    for name in ("raw_dataset_sha256", "split_sha256", "current_view_sha256",
                 "producer_roster_sha256", "producer_envelope_sha256"):
        _sha(value[name])


def _criterion(value):
    _fields(value, ("recipe_id", "recipe_sha256"), "comparative fixed criterion")
    _identity(value["recipe_id"])
    _sha(value["recipe_sha256"])


def _moves(value):
    if not isinstance(value, list) or not 2 <= len(value) <= 256:
        raise ValueError("comparative legal moves extent")
    for move in value:
        _uint(move, 65535)
    if len(set(value)) != len(value):
        raise ValueError("duplicate comparative legal move")


def _current(value):
    _fields(value, ("input_sha256", "label_sha256", "game_id", "role", "rules_state_sha256",
                    "rules_history_sha256", "encoding_sha256", "source", "frozen_epoch", "input_revision",
                    "side_to_move", "legal_moves"), "comparative checked current anchor")
    for name in ("input_sha256", "rules_state_sha256", "rules_history_sha256", "encoding_sha256"):
        _sha(value[name])
    if value["label_sha256"] is not None:
        _sha(value["label_sha256"])
    _identity(value["game_id"])
    if value["role"] not in ("proposer", "critic") or value["side_to_move"] not in ("white", "black"):
        raise ValueError("comparative P/C current anchor; auxiliary adapter required")
    _frozen._source(value["source"])
    _uint(value["frozen_epoch"])
    _uint(value["input_revision"])
    _moves(value["legal_moves"])


def _prepared(value):
    _fields(value, ("current", "producer_id", "producer_registration_sha256", "capture_evidence_sha256",
                    "input_json", "tensor_sidecar_json", "lineage_json"), "comparative prepared anchor")
    _current(value["current"])
    _identity(value["producer_id"])
    _sha(value["producer_registration_sha256"])
    _sha(value["capture_evidence_sha256"])
    for name in ("input_json", "tensor_sidecar_json", "lineage_json"):
        _pin(value[name])


def _checker(value):
    _fields(value, ("registration_sha256", "source", "search_implementation_sha256", "value_semantics_sha256",
                    "profile", "search_conditions", "horizon", "node_budget", "perspective"), "comparative checker")
    for name in ("registration_sha256", "search_implementation_sha256", "value_semantics_sha256"):
        _sha(value[name])
    if _frozen._source(value["source"])["kind"] != "own_cpu":
        raise ValueError("comparative checker requires registered own CPU")
    _identity(value["profile"])
    conditions = value["search_conditions"]
    if not isinstance(conditions, str) or not 1 <= len(conditions.encode("utf-8")) <= 2048 or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in conditions):
        raise ValueError("comparative exact shared search conditions")
    if not 1 <= _uint(value["horizon"], 256) or not 1 <= _uint(value["node_budget"]):
        raise ValueError("comparative horizon/node budget")
    if value["perspective"] not in ("captured_side_to_move", "white", "black"):
        raise ValueError("comparative perspective")


def checker_namespace(checker):
    _checker(checker)
    return metadata_digest(CHECKER_DOMAIN, checker)


def task_conditions(current, checker):
    _current(current)
    return {"checker_namespace_sha256": checker_namespace(checker),
            **{name: current[name] for name in ("rules_state_sha256", "rules_history_sha256", "frozen_epoch", "input_revision")},
            **{name: checker[name] for name in ("profile", "search_conditions", "horizon", "node_budget", "perspective")}}


def normalize_overlay(body):
    _fields(body, ("version", "parent", "criterion", "checker", "prepared_inputs", "pairs"), "comparative overlay")
    anchors, pairs = body["prepared_inputs"], body["pairs"]
    if (body["version"] != COMPARATIVE_DOMAIN or not isinstance(anchors, list) or not 1 <= len(anchors) <= MAX_PAIRS
            or not isinstance(pairs, list) or not 1 <= len(pairs) <= MAX_PAIRS):
        raise ValueError("comparative version/extent")
    _parent(body["parent"])
    _criterion(body["criterion"])
    _checker(body["checker"])
    by_input = {}
    for anchor in anchors:
        _prepared(anchor)
        identity = anchor["current"]["input_sha256"]
        if identity in by_input:
            raise ValueError("duplicate comparative prepared input")
        by_input[identity] = anchor
    pair_ids, candidate_pairs, used, tasks, evidence = set(), set(), set(), {}, {}
    for pair in pairs:
        _fields(pair, ("pair_id", "input_sha256", "kind", "candidates", "checks", "preference"), "comparative pair")
        _identity(pair["pair_id"])
        identity = _sha(pair["input_sha256"])
        if pair["pair_id"] in pair_ids:
            raise ValueError("duplicate comparative pair ID")
        pair_ids.add(pair["pair_id"])
        if identity not in by_input:
            raise ValueError("missing comparative prepared input")
        current = by_input[identity]["current"]
        if (pair["kind"], current["role"]) not in (("proposer_candidates", "proposer"), ("critic_responses", "critic")):
            raise ValueError("comparative role mismatch; auxiliary adapter required")
        candidates, checks = pair["candidates"], pair["checks"]
        if not isinstance(candidates, list) or len(candidates) != 2 or not isinstance(checks, list) or len(checks) != 2:
            raise ValueError("comparative pair extent")
        for candidate in candidates:
            _uint(candidate, 65535)
        legal = current["legal_moves"]
        if any(candidate not in legal for candidate in candidates) or legal.index(candidates[0]) >= legal.index(candidates[1]):
            raise ValueError("comparative candidates duplicate/illegal/out of captured order")
        key = identity, pair["kind"], tuple(candidates)
        if key in candidate_pairs:
            raise ValueError("duplicate comparative candidate pair")
        candidate_pairs.add(key)
        shared = task_conditions(current, body["checker"])
        for candidate, check in zip(candidates, checks):
            _fields(check, ("candidate", "task_id", "conditions", "restriction", "completion", "completed_depth",
                           "evidence_id", "evidence_artifact"), "comparative candidate check")
            _identity(check["task_id"])
            # Validate the entire nested conditions tree as strict u64 metadata
            # before equality: Python True must never compare equal to u64 1.
            _frozen._metadata_tree(check["conditions"])
            restriction = _fields(check["restriction"], ("kind", "root_moves"), "candidate-only restriction")
            roots = restriction["root_moves"]
            if not isinstance(roots, list):
                raise ValueError("candidate-only roots")
            for root in roots:
                _uint(root, 65535)
            if (_uint(check["candidate"], 65535) != candidate or check["conditions"] != shared
                    or restriction["kind"] != "candidate_only" or roots != [candidate]):
                raise ValueError("comparative checker conditions/candidate restriction mismatch")
            completion = check["completion"]
            depth = _uint(check["completed_depth"], 256)
            if completion not in ("completed", "partial", "cancelled", "unknown", "missing"):
                raise ValueError("comparative completion")
            evidence_id, artifact = check["evidence_id"], check["evidence_artifact"]
            if evidence_id is not None:
                _identity(evidence_id)
            if (evidence_id is None) != (artifact is None):
                raise ValueError("comparative evidence identity/pin mismatch")
            if artifact is not None:
                _pin(artifact)
            if completion == "completed" and (depth < shared["horizon"] or artifact is None):
                raise ValueError("completed comparative check lacks horizon/evidence")
            if completion == "missing" and (depth != 0 or artifact is not None):
                raise ValueError("missing comparative evidence has fabricated completion")
            statement = identity, check
            if check["task_id"] in tasks and tasks[check["task_id"]] != statement:
                raise ValueError("comparative task ID reused across statements")
            tasks[check["task_id"]] = statement
            if evidence_id is not None:
                if evidence_id in evidence and evidence[evidence_id] != check["task_id"]:
                    raise ValueError("comparative evidence ID reused by different tasks")
                evidence[evidence_id] = check["task_id"]
        if checks[0]["task_id"] == checks[1]["task_id"]:
            raise ValueError("comparative candidate task IDs must differ")
        if pair["preference"] not in ("left", "right", "tie", "masked"):
            raise ValueError("comparative preference")
        if pair["preference"] != "masked" and any(check["completion"] != "completed" for check in checks):
            raise ValueError("unresolved comparative pair must remain masked")
        used.add(identity)
    if used != set(by_input):
        raise ValueError("unused comparative prepared input")
    result = copy.deepcopy(body)
    result["prepared_inputs"].sort(key=lambda value: value["current"]["input_sha256"])
    result["pairs"].sort(key=lambda value: value["pair_id"])
    return result


def seal_overlay(body):
    body = normalize_overlay(body)
    result = {"overlay": body, "sha256": metadata_digest(COMPARATIVE_DOMAIN, body)}
    # The same bounded compact serialization as Rust's sealed DTO.
    if len(json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")) > _frozen.MAX_MANIFEST_BYTES:
        raise ValueError("comparative serialized byte limit")
    return result


def load_overlay(raw):
    value = _fields(_frozen._parse(raw), ("overlay", "sha256"), "sealed comparative overlay")
    expected = seal_overlay(value["overlay"])
    if _sha(value["sha256"]) != expected["sha256"]:
        raise ValueError("comparative overlay seal mismatch")
    return expected


def _actual_pin(pin, raw):
    _pin(pin)
    if not isinstance(raw, bytes) or pin["bytes"] != len(raw) or pin["sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("comparative actual prepared bytes mismatch")


def audit_metadata(*, overlay_bytes, expected_pins, checked_current_anchors,
                   independent_current_view_sha256, actual_parent_receipt, actual_prepared):
    """Metadata only. A supplied current set is not a DG05/base audit here.

    The caller must supply the checked current anchor set and independent view
    pin from the existing fully validated strict consumer over the same raw
    history. Expected parent/producer/criterion/checker/prepared pins must also
    be independent. This function cannot establish that independence, execute
    checker requests or admit a declared preference to training.
    """
    overlay = load_overlay(overlay_bytes)
    body = overlay["overlay"]
    _fields(expected_pins, ("parent", "criterion", "checker", "prepared_inputs"), "comparative expected pins")
    if not isinstance(expected_pins["prepared_inputs"], list):
        raise ValueError("comparative expected input extent")
    expected = normalize_overlay({"version": COMPARATIVE_DOMAIN, **expected_pins, "pairs": body["pairs"]})
    if any(body[name] != expected[name] for name in ("parent", "criterion", "checker", "prepared_inputs")):
        raise ValueError("comparative independent expected pins mismatch")
    _actual_pin(body["parent"]["receipt_artifact"], actual_parent_receipt)
    if _sha(independent_current_view_sha256) != body["parent"]["current_view_sha256"]:
        raise ValueError("comparative stale independent current view")
    if not isinstance(checked_current_anchors, list) or not 1 <= len(checked_current_anchors) <= _frozen.MAX_RECORDS:
        raise ValueError("comparative checked current anchor extent")
    selected = {}
    for anchor in checked_current_anchors:
        _current(anchor)
        identity = anchor["input_sha256"]
        if identity in selected:
            raise ValueError("duplicate comparative checked current input")
        selected[identity] = anchor
    if not isinstance(actual_prepared, list) or len(actual_prepared) != len(body["prepared_inputs"]):
        raise ValueError("comparative actual prepared extent")
    actual, total = {}, 0
    for value in actual_prepared:
        _fields(value, ("input_sha256", "input_json", "tensor_sidecar_json", "lineage_json"), "actual prepared rows")
        identity = _sha(value["input_sha256"])
        if identity in actual:
            raise ValueError("duplicate comparative actual prepared input")
        for name in ("input_json", "tensor_sidecar_json", "lineage_json"):
            raw = value[name]
            if not isinstance(raw, bytes) or not 1 <= len(raw) <= _frozen.MAX_MANIFEST_BYTES:
                raise ValueError("comparative actual row byte limit")
            total += len(raw)
        if total > MAX_ACTUAL_BYTES:
            raise ValueError("comparative aggregate prepared byte limit")
        actual[identity] = value
    for anchor in body["prepared_inputs"]:
        current = anchor["current"]
        identity = current["input_sha256"]
        if selected.get(identity) != current:
            raise ValueError("comparative current label/input anchor mismatch")
        if identity not in actual:
            raise ValueError("missing comparative actual prepared bytes")
        rows = actual[identity]
        for name in ("input_json", "tensor_sidecar_json", "lineage_json"):
            _actual_pin(anchor[name], rows[name])
        captured = _fields(_frozen._parse(rows["input_json"]), ("snapshot", "sha256"), "actual frozen input")
        snapshot = captured["snapshot"]
        if not isinstance(snapshot, dict) or _sha(captured["sha256"]) != identity:
            raise ValueError("comparative captured input identity mismatch")
        # Only anchor facts are compared. The old input seal and native tensor
        # encoding are still verified by the existing strict consumer.
        for name in ("game_id", "role", "rules_state_sha256", "rules_history_sha256", "encoding_sha256",
                     "source", "frozen_epoch", "input_revision", "legal_moves"):
            if snapshot.get(name) != current[name]:
                raise ValueError("comparative captured input/current facts mismatch")
        white = snapshot.get("white_to_move")
        if type(white) is not bool or ("white" if white else "black") != current["side_to_move"]:
            raise ValueError("comparative captured side-to-move mismatch")
        for name in ("frozen_epoch", "input_revision"):
            _uint(snapshot[name])
        _moves(snapshot["legal_moves"])
        _frozen._source(snapshot["source"])
    declared = sum(pair["preference"] != "masked" for pair in body["pairs"])
    return {"scope": "metadata_only", "requires_actual_checker_admission": True,
            "overlay_sha256": overlay["sha256"], "current_view_sha256": independent_current_view_sha256,
            "prepared_inputs": len(body["prepared_inputs"]), "pairs": len(body["pairs"]),
            "declared_preferences": declared, "masked_pairs": len(body["pairs"]) - declared}
