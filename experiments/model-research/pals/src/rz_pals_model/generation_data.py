"""Actual generation data V3; no neural framework or Rules implementation.

Rust owns legality, Rules trajectory membership, and the submitted ReviewView.
This reader checks their sealed receipts rather than reconstructing legal chess
or adding future targets to encoder inputs. Diagnostic records remain ineligible.
"""
import copy
import hashlib
import json
import math
import struct
from collections import Counter

DATA_DOMAIN = "rz-pals-generation-data/3"
INPUT_DOMAIN = "rz-pals-generation-input/3"
VIEW_DOMAIN = "rz-pals-selected-review-view/1"
ACTUAL_SOURCE = "actual_selfplay"
ROLES = ("proposer", "critic", "validator")
TASKS = ("defend_response", "attack_repair", "widen_responses", "lower_selectivity",
         "resume_task", "cross_profile_recheck", "defer")
RULES_ENDINGS = {"checkmate", "stalemate", "repetition", "fifty_move", "seventy_five_move",
                "insufficient_material", "dead_position"}
DRAW_ENDINGS = RULES_ENDINGS - {"checkmate"}
BUCKETS = (8, 16, 32, 64, 128, 256)
EVIDENCE_DOMAINS = frozenset(("rz-pals-generation-head-evidence/1", "rz-pals-generation-causal-evidence/1",
                            "rz-pals-generation-witness-evidence/1", "rz-pals-rules-outcome-evidence/1",
                            "rz-pals-generation-submission-evidence/1", "rz-pals-generation-native-completion-evidence/1",
                            "rz-pals-generation-native-retirement-evidence/1", "rz-pals-generation-runtime-audit-evidence/1"))


def fields(value, names, description):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"{description} fields mismatch")
    return value


def uint(value, maximum=2**64 - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("invalid bounded unsigned integer")
    return value


def identifier(value):
    if type(value) is not str or not value or len(value.encode()) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError("invalid identifier")
    return value


def sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid SHA-256")
    return value


def digest(domain, value):
    return hashlib.sha256(json.dumps([domain, value], sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def validate_evidence_receipt(evidence_sha256, receipt):
    sha(evidence_sha256)
    if (type(receipt) is not dict or receipt.get("schema") not in EVIDENCE_DOMAINS
            or digest(receipt["schema"], receipt) != evidence_sha256):
        raise ValueError("recovered evidence receipt domain/digest mismatch")
    if receipt["schema"] == "rz-pals-generation-runtime-audit-evidence/1":
        fields(receipt, ("schema", "witness_json"), "raw runtime audit receipt")
        witness = receipt["witness_json"]
        if type(witness) is not str or len(witness.encode()) > 1024 * 1024:
            raise ValueError("bounded raw runtime audit JSON required")
        parsed = strict_json(witness)
        if type(parsed) is not dict or not parsed:
            raise ValueError("raw runtime audit witness must be a strict JSON object")
    return receipt


def _moves(value, maximum=256):
    if type(value) is not list or len(value) > maximum:
        raise ValueError("bounded Move16 list required")
    for move in value:
        uint(move, 65535)
        if move >> 12 > 4 or (move & 63) == ((move >> 6) & 63):
            raise ValueError("invalid Move16 encoding")
    return value


def _anchor(anchor):
    fields(anchor, ("rules_state_sha256", "rules_history_sha256", "board_fen", "white_to_move"), "Rules anchor")
    sha(anchor["rules_state_sha256"])
    sha(anchor["rules_history_sha256"])
    if (type(anchor["white_to_move"]) is not bool or type(anchor["board_fen"]) is not str
            or len(anchor["board_fen"].encode()) > 512 or any(ord(c) < 32 for c in anchor["board_fen"])):
        raise ValueError("invalid anchor board/viewpoint")
    fen = anchor["board_fen"].split()
    if len(fen) != 6 or fen[1] != ("w" if anchor["white_to_move"] else "b"):
        raise ValueError("anchor FEN/viewpoint mismatch")


def validate_view(view):
    """Validate physical selected input; actual game history is not a view field."""
    fields(view, ("context_policy_sha256", "anchors", "records", "query", "token_budget", "length_bucket"), "selected view")
    sha(view["context_policy_sha256"])
    if uint(view["token_budget"], 512) not in (128, 256, 512) or view["length_bucket"] not in BUCKETS:
        raise ValueError("unregistered context budget/bucket")
    anchors, records = view["anchors"], view["records"]
    if type(anchors) is not list or not 1 <= len(anchors) <= 4 or type(records) is not list or len(records) > 64:
        raise ValueError("selected anchors/records capacity")
    for anchor in anchors:
        _anchor(anchor)
    tokens, longest = 0, 0
    for index, record in enumerate(records):
        fields(record, ("record_id", "kind", "anchor", "moves", "parent", "supersedes", "observed_sequence",
                        "perspective", "completed_iteration", "rules_terminal"), "selected record")
        identifier(record["record_id"])
        if record["kind"] not in ("proposal", "counter", "repair", "continuation"):
            raise ValueError("unregistered record kind")
        uint(record["anchor"], len(anchors) - 1)
        _moves(record["moves"])
        tokens += len(record["moves"])
        longest = max(longest, len(record["moves"]))
        uint(record["observed_sequence"])
        if type(record["perspective"]) is not int or record["perspective"] not in (-1, 1) or type(record["completed_iteration"]) is not bool or type(record["rules_terminal"]) is not bool:
            raise ValueError("invalid compact record semantic features")
        for relation in ("parent", "supersedes"):
            if record[relation] is not None and uint(record[relation], len(records) - 1) == index:
                raise ValueError("record self relation")
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("duplicate selected record")
    query = fields(view["query"], ("purpose", "anchors", "prefix", "proposal", "counter", "counter_present", "white_to_move"), "selected query")
    identifier(query["purpose"])
    if type(query["anchors"]) is not list or len(query["anchors"]) != 3:
        raise ValueError("query requires prefix/proposal/counter anchor references")
    for ref in query["anchors"]:
        uint(ref, len(anchors) - 1)
    if (type(query["white_to_move"]) is not bool
            or query["white_to_move"] != (anchors[query["anchors"][0]]["white_to_move"] != bool(len(query["prefix"]) % 2))):
        raise ValueError("query anchor viewpoint mismatch")
    for name in ("prefix", "proposal", "counter"):
        _moves(query[name])
        tokens += len(query[name])  # Duplicated material actually read by NN is counted again.
        longest = max(longest, len(query[name]))
    if tokens > view["token_budget"] or view["length_bucket"] != next(b for b in BUCKETS if b >= longest):
        raise ValueError("view token or minimal length bucket mismatch")
    if type(query["counter_present"]) is not bool or not query["counter_present"] and query["counter"]:
        raise ValueError("query counter existence must preserve the submitted optional line")
    return view


def view_digest(view):
    validate_view(view)
    return digest(VIEW_DOMAIN, view)


INPUT_FIELDS = ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256", "role",
                "model_epoch", "context_epoch", "capture_sequence", "question_id", "worker_id",
                "job_id", "slice_id", "rules_anchor", "actual_history", "selected_view",
                "view_sha256", "legal_moves", "action_support_mask", "encoded_model_input", "divergence_record_ids")


def _components(move):
    return [move & 63, (move >> 6) & 63, move >> 12]


def _validate_encoded_input(snapshot):
    # Encoding originates from exact Rust Rules, never Python FEN reconstruction.
    from .virtual_review import validate_payload
    payload = snapshot["encoded_model_input"]
    validate_payload(payload)
    view = snapshot["selected_view"]
    if (payload["role"] != snapshot["role"] or payload["token_budget"] != view["token_budget"]
            or len(payload["anchors"]) != len(view["anchors"]) or len(payload["records"]) != len(view["records"])
            or payload["candidates"] != [_components(m) for m in snapshot["legal_moves"]]):
        raise ValueError("encoded input role/anchor/candidate allocation mismatch")
    kinds = {"proposal": "proposal", "counter": "counterexample", "repair": "repair", "continuation": "cpu_verification"}
    for source, encoded in zip(view["records"], payload["records"]):
        expected = {"kind": kinds[source["kind"]], "perspective": source["perspective"],
                    "completed_iteration": source["completed_iteration"], "rules_terminal": source["rules_terminal"],
                    "line": [_components(m) for m in source["moves"]], "anchor": source["anchor"],
                    "parent": source["parent"], "supersedes": source["supersedes"]}
        if encoded != expected:
            raise ValueError("encoded record changed selected-view semantics")
    source, encoded = view["query"], payload["query"]
    if (encoded["purpose"] != source["purpose"] or encoded["perspective"] != (1 if source["white_to_move"] else -1)
            or encoded["counter_present"] != source["counter_present"]
            or encoded["anchors"] != source["anchors"]
            or any(encoded[name] != [_components(m) for m in source[name]] for name in ("prefix", "proposal", "counter"))):
        raise ValueError("encoded query changed selected-view semantics")


def validate_input(value):
    fields(value, INPUT_FIELDS, "generation input")
    for name in ("game_id", "opening_id", "line_genealogy_id", "question_id", "worker_id", "job_id", "slice_id"):
        identifier(value[name])
    sha(value["transposition_sha256"])
    if value["role"] not in ROLES:
        raise ValueError("unregistered generation role")
    for name in ("model_epoch", "context_epoch", "capture_sequence"):
        uint(value[name])
    _anchor(value["rules_anchor"])
    _moves(value["actual_history"], 16384)  # Audit/Rules history, never appended to review tensor.
    validate_view(value["selected_view"])
    # The public root anchor and the exact NN head endpoint are different when
    # query.prefix is nonempty. Rust Rules owns replay/state/history; WDL may
    # be known only when this exact endpoint occurs in the actual trajectory.
    if (value["selected_view"]["query"]["white_to_move"] != value["rules_anchor"]["white_to_move"]
            or sha(value["view_sha256"]) != view_digest(value["selected_view"])):
        raise ValueError("exact query endpoint/view digest mismatch")
    if any(r["observed_sequence"] > value["capture_sequence"] for r in value["selected_view"]["records"]):
        raise ValueError("future observation leaked into submitted view")
    _moves(value["legal_moves"])
    if len(set(value["legal_moves"])) != len(value["legal_moves"]):
        raise ValueError("duplicate legal candidate")
    mask = value["action_support_mask"]
    if type(mask) is not list or len(mask) != len(TASKS) or any(type(v) is not bool for v in mask):
        raise ValueError("invalid action support mask")
    _validate_encoded_input(value)
    ids = value["divergence_record_ids"]
    if type(ids) is not list or len(ids) != len(value["encoded_model_input"]["divergences"]) or len(set(ids)) != len(ids):
        raise ValueError("divergence input/provenance ordering mismatch")
    visible = {r["record_id"] for r in value["selected_view"]["records"] if r["kind"] == "counter"}
    if any(identifier(v) not in visible for v in ids):
        raise ValueError("divergence refers to absent selected counter")
    return value


def seal_input(value):
    validate_input(value)
    projected = copy.deepcopy(value)
    # Rust/Python decimal exponent spelling differs. Only the seal projection
    # uses exact f32 bits; the actual submitted model payload stays numeric.
    payload = projected["encoded_model_input"]
    def f32_bits(number):
        try:
            return struct.pack(">f", float(number)).hex()
        except (OverflowError, ValueError, struct.error) as error:
            raise ValueError("model payload value cannot be represented as f32") from error
    for anchor in payload["anchors"]:
        anchor["metadata"] = [f32_bits(v) for v in anchor["metadata"]]
    payload["divergences"] = [[f32_bits(v) for v in row] for row in payload["divergences"]]
    return digest(INPUT_DOMAIN, projected)


def _label_base(label, snapshot, description):
    if label["observed_sequence"] < snapshot["capture_sequence"]:
        raise ValueError(f"{description} predates input")
    uint(label["observed_sequence"])
    if label["question_id"] != snapshot["question_id"] or label["view_sha256"] != snapshot["view_sha256"]:
        raise ValueError(f"{description} question/view mismatch")
    sha(label["raw_evidence_sha256"])


def policy_target(snapshot, label):
    fields(label, ("observed_sequence", "question_id", "view_sha256", "raw_evidence_sha256", "source",
                   "completed", "stale", "anchor", "ranked_moves", "teacher_binary_sha256", "teacher_profile_sha256",
                   "go_sequence", "completed_depth"), "conditional teacher policy")
    _label_base(label, snapshot, "policy")
    if label["source"] != "stockfish19_completed_multipv" or label["completed"] is not True or label["stale"] is not False or label["anchor"] != snapshot["rules_anchor"]:
        raise ValueError("policy requires completed nonstale same-anchor teacher opinion")
    sha(label["teacher_binary_sha256"])
    sha(label["teacher_profile_sha256"])
    if uint(label["go_sequence"]) == 0 or uint(label["completed_depth"], 2**32 - 1) == 0:
        raise ValueError("teacher go/depth completion authority required")
    moves = _moves(label["ranked_moves"], 4)
    if not moves or len(set(moves)) != len(moves) or any(m not in snapshot["legal_moves"] for m in moves):
        raise ValueError("teacher opinion contains invalid legal candidates")
    weights = [math.exp(-rank) for rank in range(len(moves))]
    total = sum(weights)
    return [weights[moves.index(m)] / total if m in moves else 0.0 for m in snapshot["legal_moves"]]


def wdl_target(snapshot, label):
    fields(label, ("observed_sequence", "question_id", "view_sha256", "raw_evidence_sha256", "source",
                   "game_id", "anchor", "trajectory_membership_sha256", "rules_result_sha256", "ending", "outcome"), "actual Rules WDL")
    _label_base(label, snapshot, "WDL")
    sha(label["trajectory_membership_sha256"])
    sha(label["rules_result_sha256"])
    if (label["source"] != "rules_selfplay_trajectory" or label["game_id"] != snapshot["game_id"]
            or label["anchor"] != snapshot["rules_anchor"] or label["ending"] not in RULES_ENDINGS
            or label["outcome"] not in ("white_win", "black_win", "draw")):
        raise ValueError("WDL requires exact actual trajectory anchor and Rules termination")
    if (label["ending"] in DRAW_ENDINGS) != (label["outcome"] == "draw"):
        raise ValueError("Rules termination/outcome mismatch")
    if label["outcome"] == "draw":
        return [0.0, 1.0, 0.0]
    win = (label["outcome"] == "white_win") == snapshot["rules_anchor"]["white_to_move"]
    return [1.0, 0.0, 0.0] if win else [0.0, 0.0, 1.0]


def trajectory_digest(authority):
    fields(authority, ("game_id", "ending", "outcome", "anchors", "result_evidence_sha256"), "actual trajectory membership")
    identifier(authority["game_id"])
    sha(authority["result_evidence_sha256"])
    if authority["ending"] not in RULES_ENDINGS or authority["outcome"] not in ("white_win", "black_win", "draw"):
        raise ValueError("actual trajectory needs Rules termination")
    if (authority["ending"] in DRAW_ENDINGS) != (authority["outcome"] == "draw"):
        raise ValueError("actual trajectory outcome/termination mismatch")
    if type(authority["anchors"]) is not list or not 1 <= len(authority["anchors"]) <= 4096:
        raise ValueError("bounded actual Rules trajectory anchors required")
    for anchor in authority["anchors"]:
        _anchor(anchor)
    return digest("rz-pals-actual-trajectory-membership/1", authority)


def divergence_target(snapshot, label):
    fields(label, ("observed_sequence", "question_id", "view_sha256", "raw_evidence_sha256", "source",
                   "counter_record_id", "repair_question_id", "status"), "conditional divergence")
    _label_base(label, snapshot, "divergence")
    if label["source"] != "actual_repair_continuation" or label["status"] not in ("supported_after_repair", "refuted_by_repair"):
        raise ValueError("unresolved/disputed divergence cannot become a target")
    identifier(label["repair_question_id"])
    if not any(r["record_id"] == label["counter_record_id"] and r["kind"] == "counter" for r in snapshot["selected_view"]["records"]):
        raise ValueError("counter was not visible in submitted input")
    if label["counter_record_id"] not in snapshot["divergence_record_ids"]:
        raise ValueError("counter has no submitted divergence-head candidate")
    return 1.0 if label["status"] == "supported_after_repair" else 0.0


def task_pairs(snapshot, label):
    """Only causal newly witnessed Pareto dominance yields a known action pair."""
    fields(label, ("observed_sequence", "question_id", "view_sha256", "raw_evidence_sha256", "source", "outcomes", "target_known_mask"), "action outcomes")
    _label_base(label, snapshot, "V task")
    if label["source"] != "actual_action_witness" or type(label["outcomes"]) is not list or len(label["outcomes"]) > len(TASKS):
        raise ValueError("actual bounded causal action outcomes required")
    result = derive_task_targets(snapshot, label["outcomes"])
    if (type(label["target_known_mask"]) is not list or len(label["target_known_mask"]) != len(TASKS)
            or any(type(v) is not bool for v in label["target_known_mask"]) or label["target_known_mask"] != result["target_known_mask"]):
        raise ValueError("V target-known mask differs from actual causal Pareto evidence")
    return result


def derive_task_targets(snapshot, observed_outcomes):
    """Derive a label mask after collection, without changing the sealed input."""
    if type(observed_outcomes) is not list or len(observed_outcomes) > len(TASKS):
        raise ValueError("bounded actual action outcomes required")
    outcomes = {}
    visible = {r["record_id"] for r in snapshot["selected_view"]["records"]}
    for outcome in observed_outcomes:
        fields(outcome, ("action", "execution_id", "completed", "wall_cost_ns", "new_witnesses", "causal_receipt_sha256"), "action outcome")
        action = outcome["action"]
        if action not in TASKS or action in outcomes or not snapshot["action_support_mask"][TASKS.index(action)]:
            raise ValueError("unsupported/duplicate action target")
        identifier(outcome["execution_id"])
        sha(outcome["causal_receipt_sha256"])
        if outcome["wall_cost_ns"] is not None:
            uint(outcome["wall_cost_ns"])
        if type(outcome["completed"]) is not bool or type(outcome["new_witnesses"]) is not list or len(outcome["new_witnesses"]) > 256:
            raise ValueError("invalid action completion/witness list")
        witnesses = set()
        for witness in outcome["new_witnesses"]:
            fields(witness, ("witness_id", "kind", "observed_sequence", "raw_evidence_sha256"), "new valid witness")
            identifier(witness["witness_id"])
            sha(witness["raw_evidence_sha256"])
            if (witness["kind"] not in ("counter", "repair") or witness["witness_id"] in visible
                    or witness["witness_id"] in witnesses or uint(witness["observed_sequence"]) <= snapshot["capture_sequence"]):
                raise ValueError("duplicate/old/noncausal witness cannot reward V")
            witnesses.add(witness["witness_id"])
        if outcome["completed"] and outcome["wall_cost_ns"] is not None:
            outcomes[action] = (len(witnesses), outcome["wall_cost_ns"])
    known, pairs = [False] * len(TASKS), []
    for preferred, a in outcomes.items():
        for other, b in outcomes.items():
            if preferred == other or max(a[0], b[0]) == 0:
                continue  # No new evidence, including Defer, supplies no utility supervision.
            if a[0] >= b[0] and a[1] <= b[1] and a != b:
                i, j = TASKS.index(preferred), TASKS.index(other)
                known[i] = known[j] = True
                pairs.append((i, j))
    return {"target_known_mask": known, "pairs": pairs, "new_witness_context": any(v[0] for v in outcomes.values())}


def validate_record(record):
    fields(record, ("schema", "source_kind", "eligibility", "input", "input_sha256", "future_targets"), "generation record V3")
    if record["schema"] != DATA_DOMAIN or record["source_kind"] != ACTUAL_SOURCE or record["eligibility"] != "actual_generation_candidate":
        raise ValueError("diagnostic/fixture assets cannot be promoted into actual generation")
    snapshot = validate_input(record["input"])
    if sha(record["input_sha256"]) != seal_input(snapshot):
        raise ValueError("selected input seal mismatch")
    targets = fields(record["future_targets"], ("policy", "wdl", "divergence", "task"), "head-separated future targets")
    result = dict.fromkeys(targets, None)
    for head, function in (("policy", policy_target), ("wdl", wdl_target), ("divergence", divergence_target), ("task", task_pairs)):
        if targets[head] is not None:
            if (snapshot["role"] == "validator" and head != "task") or (snapshot["role"] != "validator" and head == "task") or (head == "divergence" and snapshot["role"] != "critic"):
                raise ValueError("target head violates private role routing")
            result[head] = function(snapshot, targets[head])
    return result


def coverage(records):
    """Count unique actual inputs; duplicate rows never fill admission quotas."""
    counts, seen = Counter(), set()
    for record in records:
        targets = validate_record(record)
        identity = record["input_sha256"]
        if identity in seen:
            counts["duplicate_inputs_excluded"] += 1
            continue
        seen.add(identity)
        _count_target_coverage(counts, record, targets)
    return dict(counts)


def _count_target_coverage(counts, record, targets):
    """Count one previously verified unique input; disk index owns uniqueness."""
    role = record["input"]["role"]
    counts[f"{role}_unique_inputs"] += 1
    if role != "validator" and any(targets[h] is not None for h in ("policy", "wdl", "divergence")):
        counts[f"{role}_valid_samples"] += 1
    for head, target in targets.items():
        counts[f"{role}_{head}_{'masked' if target is None else 'valid'}"] += 1
    if role == "validator" and targets["task"]:
        counts["validator_known_pairs"] += len(targets["task"]["pairs"])
        counts["validator_new_witness_contexts"] += int(bool(targets["task"]["pairs"]) and targets["task"]["new_witness_context"])


def admit_phase(records, phase):
    observed = coverage(records)
    return _admit_coverage(observed, phase)


def _admit_coverage(observed, phase):
    required = {"pc": {"proposer_valid_samples": 1024, "critic_valid_samples": 1024},
                "v": {"validator_known_pairs": 128, "validator_new_witness_contexts": 32}}
    if phase not in required:
        raise ValueError("unknown generation phase")
    missing = {name: minimum - observed.get(name, 0) for name, minimum in required[phase].items() if observed.get(name, 0) < minimum}
    if missing:
        raise ValueError("InsufficientValidTargets: " + json.dumps(missing, sort_keys=True))
    return observed


def split_records(records, opening_splits):
    """Drop all known cross-split genealogy/transposition inputs symmetrically."""
    if type(opening_splits) is not dict or any(v not in ("train", "validation", "holdout") for v in opening_splits.values()):
        raise ValueError("opening split must be registered before outcomes")
    domains = {}
    for record in records:
        validate_record(record)
        s = record["input"]
        if s["opening_id"] not in opening_splits:
            raise ValueError("unregistered opening")
        split = opening_splits[s["opening_id"]]
        for key in (("genealogy", s["line_genealogy_id"]), ("transposition", s["transposition_sha256"]),
                    ("rules_state", s["rules_anchor"]["rules_state_sha256"]), ("game", s["game_id"])):
            domains.setdefault(key, set()).add(split)
    leaked = {key for key, splits in domains.items() if len(splits) > 1}
    outputs, excluded = {name: [] for name in ("train", "validation", "holdout")}, 0
    for record in records:
        s = record["input"]
        if any(k in leaked for k in (("genealogy", s["line_genealogy_id"]), ("transposition", s["transposition_sha256"]),
                                     ("rules_state", s["rules_anchor"]["rules_state_sha256"]), ("game", s["game_id"]))):
            excluded += 1
        else:
            outputs[opening_splits[s["opening_id"]]].append(copy.deepcopy(record))
    return outputs, {"cross_split_records_excluded": excluded, "cross_split_identities": len(leaked)}


class GenerationDataset:
    """Immutable admitted rows bound to real producer and trajectory receipts.

    The caller supplies independently recovered producer authorities. A row's
    own source_kind string is never sufficient evidence of an actual collection.
    This class neither fabricates a receipt nor promotes a diagnostic asset.
    """
    def __init__(self, records, *, producer, trajectories, opening_splits, allowed_evidence):
        fields(producer, ("schema", "source_kind", "source_sha256", "model_sha256", "runtime_sha256",
                          "teacher_binary_sha256", "teacher_profile_sha256", "opening_split_sha256",
                          "registered_before_outcomes", "input_seals", "actual_game_ids"), "generation producer authority")
        if (producer["schema"] != "rz-pals-actual-collection-authority/1" or producer["source_kind"] != ACTUAL_SOURCE
                or producer["registered_before_outcomes"] is not True):
            raise ValueError("actual producer authority with prior split registration required")
        for name in ("source_sha256", "model_sha256", "runtime_sha256", "teacher_binary_sha256", "teacher_profile_sha256", "opening_split_sha256"):
            sha(producer[name])
        if producer["opening_split_sha256"] != digest("rz-pals-opening-split/1", opening_splits):
            raise ValueError("outcome-selected or changed opening split")
        if type(records) is not list or len(records) > 262144 or type(trajectories) is not dict:
            raise ValueError("bounded actual records/trajectory authority required")
        if type(producer["input_seals"]) is not list or type(producer["actual_game_ids"]) is not list:
            raise ValueError("actual producer input/game identities required")
        if len(producer["input_seals"]) > 262144 or len(producer["actual_game_ids"]) > 8192 or len(trajectories) > 8192:
            raise ValueError("actual producer/trajectory authority capacity")
        seals = set(sha(v) for v in producer["input_seals"])
        games = set(identifier(v) for v in producer["actual_game_ids"])
        if len(seals) != len(producer["input_seals"]) or len(games) != len(producer["actual_game_ids"]):
            raise ValueError("duplicate producer publication input/game identity")
        if type(allowed_evidence) is not dict or len(allowed_evidence) > 1048576:
            raise ValueError("independently recovered full evidence receipts required, not digest strings")
        evidence = copy.deepcopy(allowed_evidence)
        for evidence_sha256, receipt in evidence.items():
            validate_evidence_receipt(evidence_sha256, receipt)
        for membership_sha256, authority in trajectories.items():
            if sha(membership_sha256) != trajectory_digest(authority):
                raise ValueError("actual trajectory membership seal mismatch")
        for record in records:
            self._validate_published_record(record, producer, seals, games, trajectories, evidence)
        split_rows, exclusions = split_records(records, opening_splits)
        self._records = copy.deepcopy(split_rows)
        self._producer = copy.deepcopy(producer)
        self._authority = copy.deepcopy(trajectories)
        self._evidence = evidence
        self._splits = copy.deepcopy(opening_splits)
        self.exclusions = exclusions
        self._seal = self._integrity()

    @staticmethod
    def _validate_published_record(record, producer, seals, games, trajectories, evidence):
        """Shared exact joins for in-memory and bounded disk-backed consumers."""
        validate_record(record)
        snapshot = record["input"]
        if record["input_sha256"] not in seals or snapshot["game_id"] not in games:
            raise ValueError("input absent from actual producer publication")
        for head, target in record["future_targets"].items():
            if target is not None:
                GenerationDataset._join_evidence(record["input_sha256"], head, target, evidence)
                if head == "policy" and (target["teacher_binary_sha256"] != producer["teacher_binary_sha256"]
                                         or target["teacher_profile_sha256"] != producer["teacher_profile_sha256"]):
                    raise ValueError("conditional policy teacher differs from registered CPU_T")
        label = record["future_targets"]["wdl"]
        if label is not None:
            authority = trajectories.get(label["trajectory_membership_sha256"])
            fields(authority, ("game_id", "ending", "outcome", "anchors", "result_evidence_sha256"), "actual Rules trajectory authority")
            if (authority["game_id"] != snapshot["game_id"] or authority["ending"] != label["ending"]
                    or authority["outcome"] != label["outcome"] or authority["result_evidence_sha256"] != label["rules_result_sha256"]
                    or snapshot["rules_anchor"] not in authority["anchors"]):
                raise ValueError("WDL anchor absent from independently recovered actual trajectory")
            outcome = {"schema": "rz-pals-rules-outcome-evidence/1", "game_id": authority["game_id"],
                       "ending": authority["ending"], "outcome": authority["outcome"], "rules_terminal": True}
            if evidence.get(authority["result_evidence_sha256"]) != outcome:
                raise ValueError("actual Rules termination evidence is missing or changed")

    @staticmethod
    def _join_evidence(input_sha256, head, target, receipts):
        content = {key: value for key, value in target.items() if key != "raw_evidence_sha256"}
        expected = {"schema": "rz-pals-generation-head-evidence/1", "input_sha256": input_sha256,
                    "head": head, "target": content}
        if receipts.get(target["raw_evidence_sha256"]) != expected:
            raise ValueError("target differs from independently recovered full head evidence")
        if head == "task":
            for outcome in target["outcomes"]:
                causal = {"schema": "rz-pals-generation-causal-evidence/1", "input_sha256": input_sha256,
                          "outcome": {key: value for key, value in outcome.items() if key != "causal_receipt_sha256"}}
                if receipts.get(outcome["causal_receipt_sha256"]) != causal:
                    raise ValueError("V action differs from recovered causal execution evidence")
                for witness in outcome["new_witnesses"]:
                    raw = {"schema": "rz-pals-generation-witness-evidence/1", "input_sha256": input_sha256,
                           "witness": {key: value for key, value in witness.items() if key != "raw_evidence_sha256"},
                           "rules_validated": True}
                    if receipts.get(witness["raw_evidence_sha256"]) != raw:
                        raise ValueError("V witness differs from recovered Rules validation evidence")

    def _integrity(self):
        return digest("rz-pals-generation-dataset/1", [self._records, self._producer, self._authority, self._evidence, self._splits, self.exclusions])

    def rows(self, split, *, optimization=False):
        if split not in self._records or (optimization and split != "train") or self._seal != self._integrity():
            raise ValueError("immutable dataset/split authority mismatch")
        return copy.deepcopy(self._records[split])

    def admit(self, phase):
        return admit_phase(self.rows("train", optimization=True), phase)

    @property
    def sha256(self):
        if self._seal != self._integrity():
            raise ValueError("immutable actual dataset changed")
        return self._seal

    @property
    def authority(self):
        if self._seal != self._integrity():
            raise ValueError("immutable actual dataset changed")
        return copy.deepcopy(self._producer)

    @property
    def opening_splits(self):
        if self._seal != self._integrity():
            raise ValueError("immutable actual dataset changed")
        return copy.deepcopy(self._splits)


def strict_json(text):
    """Reject duplicate keys/nonfinite values before either canonical seal."""
    if type(text) is not str or len(text.encode()) > 16 * 1024 * 1024:
        raise ValueError("bounded generation JSON required")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate generation JSON key")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError("nonfinite generation JSON: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)
