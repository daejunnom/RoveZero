"""PALS training *preparation*: audited tensors, losses, optimizer and resume.

This preparation module deliberately has no optimizer-update entry point. The
separate nonzero_training domain owns bounded actual updates and checkpoints. Rules
and the Rust collector own legality and feature encoding. A /2 historical row
cannot reconstruct public observation features from their hashes, so collation
requires a separately supplied, input-bound encoded snapshot. Targets never
enter its encoder input. The numerical helpers support CPU preparation; GPU
use requires explicitly named devices and does not fall back to CPU.
"""
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import stat
import struct
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from .artifacts import digest_file, output_directory
from .config import ModelConfig, TASKS, MAX_LINE_PLIES
from .model import TensorInput

DATA_DOMAIN = "rz-pals-data/2"
LABEL_DOMAIN = "rz-pals-label/1"
CURRENT_LABEL_VIEW_DOMAIN = "rz-pals-current-label-view/1"
CHECKPOINT_SCHEMA = "rz-pals-python-preparation-checkpoint/1"
U64_MAX = 2**64 - 1
ROLE_NAMES = {"proposer": "proposer", "critic": "critic", "verifier": "validator"}
SNAPSHOT_FIELDS = ("game_id", "opening_id", "line_genealogy_id", "position_command", "board_fen",
                   "actual_history", "rules_state_sha256", "rules_history_sha256", "transposition_sha256",
                   "encoding_sha256", "source", "frozen_epoch", "input_revision", "capture_sequence",
                   "white_to_move", "role", "legal_moves", "public_records")
LABEL_FIELDS = ("observed_sequence", "provenance", "policy", "value_wdl", "white_to_move",
                "counterexample", "verifier_tasks", "supersedes_label_sha256")


def _fields(value, names, description):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"{description} fields mismatch")
    return {name: value[name] for name in names}


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid SHA-256 identity")
    return value


def _uint(value, maximum=U64_MAX):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("invalid bounded unsigned integer")
    return value


def _vector(value, size, *, integers=False):
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError("encoded feature shape mismatch; broadcasting is prohibited")
    if integers:
        for item in value:
            _uint(item, 12)
    elif not all(type(item) in (int, float) and math.isfinite(item) and abs(item) <= np.finfo(np.float32).max for item in value):
        raise ValueError("encoded features must be finite FP32")


def _identity(value):
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 256 or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value):
        raise ValueError("invalid identifier")


def _canonical(domain, value):
    return hashlib.sha256(json.dumps([domain, value], ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _sorted_canonical(domain, value):
    return hashlib.sha256(json.dumps([domain, value], sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _private_fp32_bits(value):
    if type(value) not in (int, float):
        raise ValueError("V private latent requires finite FP32 numbers")
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("V private latent requires finite FP32 numbers")
        encoded = struct.pack(">f", number)
    except (OverflowError, struct.error) as error:
        raise ValueError("V private latent exceeds finite FP32") from error
    if not math.isfinite(struct.unpack(">f", encoded)[0]):
        raise ValueError("V private latent exceeds finite FP32")
    return encoded.hex()


def _source(value):
    if not isinstance(value, dict):
        raise ValueError("input source must be explicit")
    if value.get("kind") == "own_cpu":
        value = _fields(value, ("kind", "cpu_binary_sha256", "evaluator_configuration_sha256", "model_weights_sha256"), "CPU source")
        _sha(value["cpu_binary_sha256"])
        _sha(value["evaluator_configuration_sha256"])
        if value["model_weights_sha256"] is not None:
            _sha(value["model_weights_sha256"])
    elif value.get("kind") == "own_pals":
        value = _fields(value, ("kind", "model_configuration_sha256", "model_weights_sha256"), "PALS source")
        _sha(value["model_configuration_sha256"])
        _sha(value["model_weights_sha256"])
    else:
        raise ValueError("external or missing input source")
    return value


def move_components(value):
    """Decode the existing Move16 layout; this does not determine legality."""
    _uint(value, 65535)
    source, destination, promotion = value & 63, (value >> 6) & 63, value >> 12
    if promotion > 4 or source == destination:
        raise ValueError("invalid Move16 code")
    return source, destination, promotion


def seal_snapshot(snapshot):
    """Canonical /2 input seal matching Rust declaration order (no floats)."""
    value = _fields(snapshot, SNAPSHOT_FIELDS, "snapshot")
    value["source"] = _source(value["source"])
    value["public_records"] = [_fields(r, ("observation_sha256", "situation_revision"), "public record") for r in value["public_records"]]
    return _canonical(DATA_DOMAIN, value)


def _validate_record(record, owned_cpu, sources):
    record = _fields(record, ("input", "future_label", "verifier_private"), "learning record")
    frozen = _fields(record["input"], ("snapshot", "sha256"), "frozen input")
    s = _fields(frozen["snapshot"], SNAPSHOT_FIELDS, "snapshot")
    for name in ("game_id", "opening_id", "line_genealogy_id"):
        _identity(s[name])
    for name in ("rules_state_sha256", "rules_history_sha256", "transposition_sha256", "encoding_sha256"):
        _sha(s[name])
    for name in ("frozen_epoch", "input_revision", "capture_sequence"):
        _uint(s[name])
    if type(s["white_to_move"]) is not bool or s["role"] not in ROLE_NAMES:
        raise ValueError("invalid role/viewpoint")
    if not isinstance(s["position_command"], str) or not s["position_command"].startswith("position ") or len(s["position_command"].encode()) > 256 * 1024 or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in s["position_command"]):
        raise ValueError("invalid complete position command")
    if not isinstance(s["board_fen"], str) or not 1 <= len(s["board_fen"].encode()) <= 512 or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in s["board_fen"]):
        raise ValueError("invalid FEN snapshot")
    fields = s["board_fen"].split()
    if len(fields) != 6 or fields[1] != ("w" if s["white_to_move"] else "b"):
        raise ValueError("snapshot viewpoint mismatch")
    for name, maximum in (("actual_history", 16384), ("legal_moves", 256)):
        if not isinstance(s[name], list) or len(s[name]) > maximum:
            raise ValueError("move allocation bound")
        for move in s[name]:
            move_components(move)
    if len(set(s["legal_moves"])) != len(s["legal_moves"]):
        raise ValueError("duplicate legal move")
    if not isinstance(s["public_records"], list) or len(s["public_records"]) > 128:
        raise ValueError("public record allocation bound")
    for public in s["public_records"]:
        public = _fields(public, ("observation_sha256", "situation_revision"), "public record")
        _sha(public["observation_sha256"])
        if _uint(public["situation_revision"]) > s["input_revision"]:
            raise ValueError("future public record leaked into input")
    source = _source(s["source"])
    if (sources is not None and source not in sources) or (owned_cpu is not None and source["kind"] == "own_cpu" and source["cpu_binary_sha256"] not in owned_cpu):
        raise ValueError("unregistered own input source")
    if _sha(frozen["sha256"]) != seal_snapshot(s):
        raise ValueError("historical /2 input seal mismatch")
    private = record["verifier_private"]
    if private is not None:
        private = _fields(private, ("task_kind", "control_sha256", "private_latent"), "verifier private")
        _identity(private["task_kind"])
        _sha(private["control_sha256"])
        if s["role"] != "verifier" or not isinstance(private["private_latent"], list) or len(private["private_latent"]) > 16 * 384:
            raise ValueError("V private data is not permitted in P/C inputs")
        for value in private["private_latent"]:
            _private_fp32_bits(value)
    label = record["future_label"]
    if label is None:
        return
    label = _fields(label, LABEL_FIELDS, "future label")
    if _uint(label["observed_sequence"]) < s["capture_sequence"] or type(label["white_to_move"]) is not bool or label["white_to_move"] != s["white_to_move"]:
        raise ValueError("future label sequence/viewpoint mismatch")
    if label["supersedes_label_sha256"] is not None:
        _sha(label["supersedes_label_sha256"])
    p = label["provenance"]
    if not isinstance(p, dict):
        raise ValueError("missing target provenance")
    if p.get("source") == "owned_cpu":
        p = _fields(p, ("source", "engine_sha256", "profile_sha256", "task_sha256", "completed_depth", "nodes", "raw_evidence_sha256"), "owned CPU target")
        for name in ("engine_sha256", "profile_sha256", "task_sha256", "raw_evidence_sha256"):
            _sha(p[name])
        if (owned_cpu is not None and p["engine_sha256"] not in owned_cpu) or _uint(p["completed_depth"], 2**32 - 1) < 1:
            raise ValueError("unregistered or unfinished CPU target")
        _uint(p["nodes"])
        if label["value_wdl"] is not None:
            _vector(label["value_wdl"], 3)
            raise ValueError("CPU estimates cannot become actual outcome WDL")
    elif p.get("source") in ("actual_game", "rules_terminal"):
        names = ("source", "result") if p["source"] == "actual_game" else ("source", "rules_state_sha256", "result")
        p = _fields(p, names, "actual outcome provenance")
        result = _fields(p["result"], ("game_id", "outcome", "ending", "raw_evidence_sha256"), "actual result")
        _sha(result["raw_evidence_sha256"])
        interrupted = {"ply_limit", "wall_time_limit", "infrastructure_failure", "user_stop", "unresolved"}
        draws = {"stalemate", "repetition", "fifty_move", "seventy_five_move", "insufficient_material", "dead_position"}
        endings = interrupted | draws | {"checkmate", "time_forfeit", "illegal_move", "engine_crash"}
        if result["game_id"] != s["game_id"] or result["ending"] not in endings or result["outcome"] not in ("white_win", "black_win", "draw", "unknown") or ((result["ending"] in interrupted) != (result["outcome"] == "unknown")):
            raise ValueError("invalid actual outcome")
        if (result["ending"] in draws and result["outcome"] != "draw") or (result["ending"] == "checkmate" and result["outcome"] not in ("white_win", "black_win")):
            raise ValueError("result/termination mismatch")
        if p["source"] == "rules_terminal" and (p["rules_state_sha256"] != s["rules_state_sha256"] or result["ending"] not in draws | {"checkmate"} or (result["ending"] in {"checkmate", "stalemate"} and s["legal_moves"])):
            raise ValueError("terminal label does not describe captured Rules state")
        if p["source"] == "rules_terminal" and result["ending"] == "checkmate" and result["outcome"] != ("black_win" if s["white_to_move"] else "white_win"):
            raise ValueError("captured checkmate must defeat the side to move")
        if label["value_wdl"] is not None:
            _vector(label["value_wdl"], 3)
            win = (result["outcome"] == "white_win") == s["white_to_move"]
            expected = [0.0, 1.0, 0.0] if result["outcome"] == "draw" else ([1.0, 0.0, 0.0] if win else [0.0, 0.0, 1.0])
            if result["outcome"] == "unknown" or label["value_wdl"] != expected:
                raise ValueError("unknown/nonactual/viewpoint-invalid WDL target")
    else:
        raise ValueError("external or missing teacher provenance")
    if label["policy"] is not None:
        policy = _fields(label["policy"], ("moves", "probabilities"), "policy target")
        probabilities = policy["probabilities"]
        if not isinstance(policy["moves"], list) or len(policy["moves"]) > 256:
            raise ValueError("policy move allocation bound")
        for move in policy["moves"]:
            _uint(move, 65535)
        if s["role"] == "verifier" or not s["legal_moves"] or policy["moves"] != s["legal_moves"] or not isinstance(probabilities, list) or len(probabilities) != len(s["legal_moves"]) or not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in probabilities) or abs(sum(probabilities) - 1) > 1e-6:
            raise ValueError("invalid policy target/legal move order")
    if label["counterexample"] is not None:
        c = _fields(label["counterexample"], ("challenged_line_sha256", "divergence_ply", "response_line", "repair_line", "validity", "input_revision", "legality_evidence_sha256"), "counterexample")
        _sha(c["challenged_line_sha256"])
        _sha(c["legality_evidence_sha256"])
        _uint(c["divergence_ply"], 2**32 - 1)
        if s["role"] != "critic" or _uint(c["input_revision"]) != s["input_revision"] or c["validity"] not in ("supported_after_repair", "refuted_by_repair", "not_examined", "disputed"):
            raise ValueError("invalid conditional counterexample")
        for name in ("response_line", "repair_line"):
            line = c[name]
            if name == "repair_line" and line is None:
                continue
            if not isinstance(line, list) or not 1 <= len(line) <= 256:
                raise ValueError("counterexample line extent")
            for move in line:
                move_components(move)
        if c["validity"] in ("supported_after_repair", "refuted_by_repair") and c["repair_line"] is None:
            raise ValueError("conditional target requires its actual repair")
    if label["verifier_tasks"] is not None:
        tasks = label["verifier_tasks"]
        if s["role"] != "verifier" or not isinstance(tasks, list) or not 1 <= len(tasks) <= 64:
            raise ValueError("invalid V target extent")
        seen = set()
        for task in tasks:
            task = _fields(task, ("task", "branch_sha256", "cpu_profile_sha256", "budget_bucket", "preference_rank", "information_gain_evidence_sha256"), "V task")
            key = (TaskContext.from_target(task), task["task"])
            if task["task"] not in TASKS or key in seen:
                raise ValueError("unknown or duplicate V task/context")
            seen.add(key)
            rank = task["preference_rank"]
            if task["information_gain_evidence_sha256"] is not None:
                _sha(task["information_gain_evidence_sha256"])
            if rank is not None and (_uint(rank, len(tasks) - 1) < 0 or task["information_gain_evidence_sha256"] is None):
                raise ValueError("ranked V task needs observed information gain")


def _label_digest_value(input_sha256, label):
    value = copy.deepcopy(label)
    if value["policy"] is not None:
        value["policy"]["probabilities"] = [struct.pack(">d", float(v)).hex() for v in value["policy"]["probabilities"]]
    if value["value_wdl"] is not None:
        value["value_wdl"] = [struct.pack(">d", float(v)).hex() for v in value["value_wdl"]]
    return _sorted_canonical(LABEL_DOMAIN, {"input_sha256": input_sha256, "label": value})


def label_digest(record):
    """New input-bound identity matching Rust PalsLearningRecord.label_digest.

    Minimal UTF-8 JSON is [domain,{input_sha256,label}], with recursively sorted
    object keys and unchanged array order. Label probabilities/WDL alone become
    16-digit lowercase binary64 bit strings. None labels have no digest. Private
    V controls are compared separately by current_label_view. This validates the
    record shape/seal, but neither enrolls a source nor admits a dataset split.
    Existing input/raw-row/dataset digests are not converted to this domain.
    """
    _validate_record(record, None, None)
    label = record["future_label"]
    return None if label is None else _label_digest_value(record["input"]["sha256"], label)


def _private_context_digest(private):
    value = copy.deepcopy(private)
    if value is not None:
        value["private_latent"] = [_private_fp32_bits(v) for v in value["private_latent"]]
    return _sorted_canonical("rz-pals-verifier-private-context/1", value)


@dataclass(frozen=True)
class CurrentLabelView:
    current_indices: tuple
    sha256: str


def current_label_view(records):
    """Audit <=65,536 raw rows without mutation; select whole current labels.

    Input SHA order fixes view order regardless of raw row order. Sources/split
    still need independent admission through ValidatedDataset. The separate view
    digest binds each input, current label and exact V-private FP32 context.
    """
    if not isinstance(records, list) or not 1 <= len(records) <= 65536:
        raise ValueError("dataset record extent")
    groups, labels, digests, contexts = {}, {}, [], []
    for index, row in enumerate(records):
        digest = label_digest(row)
        if digest is not None:
            if digest in labels:
                raise ValueError("duplicate label identity")
            labels[digest] = index
        groups.setdefault(row["input"]["sha256"], []).append(index)
        digests.append(digest)
        contexts.append(_private_context_digest(row["verifier_private"]))
    children = {}
    for index, row in enumerate(records):
        label = row["future_label"]
        predecessor = None if label is None else label["supersedes_label_sha256"]
        if predecessor is not None:
            if predecessor not in labels:
                raise ValueError("missing predecessor label")
            prior = labels[predecessor]
            if row["input"] != records[prior]["input"]:
                raise ValueError("label predecessor belongs to a different input")
            if prior in children:
                raise ValueError("forked label supersession")
            children[prior] = index
    selected, descriptors = [], []
    for input_sha256, indices in sorted(groups.items()):
        first = indices[0]
        for index in indices:
            if records[index]["input"] != records[first]["input"]:
                raise ValueError("immutable input identity collision")
            if contexts[index] != contexts[first]:
                raise ValueError("different V private contexts share an input label chain")
        unlabeled = [i for i in indices if digests[i] is None]
        labeled = [i for i in indices if digests[i] is not None]
        if len(unlabeled) > 1:
            raise ValueError("duplicate unlabeled input")
        if not labeled:
            current = unlabeled[0]
        else:
            roots = [i for i in labeled if records[i]["future_label"]["supersedes_label_sha256"] is None]
            if len(roots) != 1:
                raise ValueError("cyclic or multiple-root label chain")
            visited, current = set(), roots[0]
            while True:
                if current in visited:
                    raise ValueError("cyclic label supersession")
                visited.add(current)
                if current not in children:
                    break
                successor = children[current]
                if records[successor]["future_label"]["observed_sequence"] <= records[current]["future_label"]["observed_sequence"]:
                    raise ValueError("label supersession sequence must strictly increase")
                current = successor
            if len(visited) != len(labeled):
                raise ValueError("disconnected or cyclic label chain")
        selected.append(current)
        descriptors.append({"input_sha256": input_sha256, "label_sha256": digests[current],
                            "verifier_private_sha256": contexts[current]})
    return CurrentLabelView(tuple(selected), _sorted_canonical(CURRENT_LABEL_VIEW_DOMAIN, descriptors))


@dataclass(frozen=True)
class TaskContext:
    branch_sha256: str
    cpu_profile_sha256: str
    budget_bucket: int

    def validate(self):
        _sha(self.branch_sha256)
        _sha(self.cpu_profile_sha256)
        _uint(self.budget_bucket, 16)

    @classmethod
    def from_target(cls, target):
        result = cls(target["branch_sha256"], target["cpu_profile_sha256"], target["budget_bucket"])
        result.validate()
        return result


@dataclass(frozen=True)
class DivergenceContext:
    challenged_line_sha256: str
    divergence_ply: int
    input_revision: int


@dataclass(frozen=True)
class EncodedSnapshot:
    """Collector/Rules supplied features; hashes are identities, never features.

    V queries must be encoded per explicit branch/profile/budget context. P/C
    queries and public records are frozen at capture, before labels are observed.
    """
    input_sha256: str
    encoding_sha256: str
    board: tuple
    metadata: tuple
    public_records: tuple  # (observation_sha256, situation_revision, 16 features)
    query: tuple
    divergences: tuple = ()  # (DivergenceContext, 8 features)
    task_queries: tuple = ()  # (TaskContext, 16 features)
    model_profile: str = "legacy_summary_v1"
    full_line: dict = None  # exact admitted /2 payload; never reconstructed from hashes


def _encoding_value(value):
    encoded = asdict(value)
    if value.model_profile == "legacy_summary_v1" and value.full_line is None:
        # Preserve the /1 derived encoding identity, not just the input seal.
        del encoded["model_profile"]
        del encoded["full_line"]
    return encoded


def _encoding_identity(value):
    return _canonical("rz-pals-python-immutable-encoding/1", _encoding_value(value))


def _validate_full_line(value, records):
    value = _fields(value, ("records", "query_prefix", "query_proposal", "query_counter"), "full-line input")
    if not isinstance(value["records"], list) or len(value["records"]) != records:
        raise ValueError("full-line record order/extent differs from captured summary records")
    def moves(line):
        if not isinstance(line, list) or len(line) > MAX_LINE_PLIES:
            raise ValueError("full-line ply allocation bound")
        for token in line:
            token = _fields(token, ("from", "to", "promotion"), "full-line move token")
            source, destination = _uint(token["from"], 63), _uint(token["to"], 63)
            _uint(token["promotion"], 4)
            if source == destination:
                raise ValueError("full-line move has identical from/to")
    relations = []
    for index, record in enumerate(value["records"]):
        if (not isinstance(record, dict) or not {"moves", "parent", "supersedes"} <= set(record)
                or set(record) - {"moves", "parent", "supersedes", "parent_required", "supersedes_required"}):
            raise ValueError("full-line record fields")
        moves(record["moves"])
        pair = []
        for name in ("parent", "supersedes"):
            required = record.get(name + "_required", False)
            if type(required) is not bool or (required and record[name] is None):
                raise ValueError("full-line required local relationship is missing")
            reference = record[name]
            if reference is not None and (_uint(reference, records - 1) == index):
                raise ValueError("full-line self relationship")
            pair.append(reference)
        relations.append(pair)
    complete, visiting = set(), set()
    def visit(index):
        if index in visiting:
            raise ValueError("full-line relationship cycle")
        if index in complete:
            return
        visiting.add(index)
        for reference in relations[index]:
            if reference is not None:
                visit(reference)
        visiting.remove(index)
        complete.add(index)
    for index in range(records):
        visit(index)
    for name in ("query_prefix", "query_proposal", "query_counter"):
        moves(value[name])


def _validate_encoding(e):
    _sha(e.input_sha256)
    _sha(e.encoding_sha256)
    _vector(e.board, 64, integers=True)
    _vector(e.metadata, 16)
    _vector(e.query, 16)
    config = ModelConfig.for_profile(e.model_profile)
    if config.full_line:
        if e.full_line is None:
            raise ValueError("full-line encoded snapshot requires explicit complete /2 payload")
    elif e.full_line is not None:
        raise ValueError("encoded snapshot profile rejects unexpected full-line payload")
    if not isinstance(e.public_records, (list, tuple)) or len(e.public_records) > 128 or not isinstance(e.divergences, (list, tuple)) or len(e.divergences) > 128 or not isinstance(e.task_queries, (list, tuple)) or len(e.task_queries) > 64:
        raise ValueError("encoded item allocation bound")
    for entry in e.public_records:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError("encoded public record descriptor")
        _sha(entry[0])
        _uint(entry[1])
        _vector(entry[2], 16)
    if config.full_line:
        _validate_full_line(e.full_line, len(e.public_records))
        if e.query[6] != 0 or e.query[7] != 0 or any(record[2][6] != 0 for record in e.public_records):
            raise ValueError("full-line captured encoding carries forbidden origin/revision/deadline semantic features")
    for entry in e.divergences:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("encoded divergence descriptor")
        _vector(entry[1], 8)
    for entry in e.task_queries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[0], TaskContext):
            raise ValueError("encoded V query descriptor")
        entry[0].validate()
        _vector(entry[1], 16)


def _unique_json(text):
    def pairs(entries):
        value = {}
        for key, item in entries:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value
    return json.loads(text, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _fen_board(fen):
    """Representation check only; Rules remains the legality authority."""
    rows = fen.split()[0].split("/")
    if len(rows) != 8:
        raise ValueError("FEN board representation")
    result = []
    pieces = {piece: i + 1 for i, piece in enumerate("PNBRQKpnbrqk")}
    for row in reversed(rows):
        squares = []
        for token in row:
            if token in "12345678":
                squares.extend([0] * int(token))
            elif token in pieces:
                squares.append(pieces[token])
            else:
                raise ValueError("FEN board representation")
        if len(squares) != 8:
            raise ValueError("FEN rank extent")
        result.extend(squares)
    return tuple(result)


def encoded_from_sidecar(sidecar, record, *, expected_encoder_source_sha256, expected_model_epoch=None):
    """Consume the collector's /1 exact-byte native tensor sidecar.

    Expected encoder source and neural epoch come from independently verified
    collection receipts, never from the row enrolling itself. Rust's float JSON
    is stored as a string: verify its bytes instead of reserializing FP32 values.
    Current native collection supplies role queries, without C divergence or V
    context descriptors; those conditional targets require explicit descriptors
    before they can be admitted by the collator.
    """
    version = sidecar.get("version") if isinstance(sidecar, dict) else None
    names = ("version", "input_sha256", "encoding_sha256", "encoder_source_sha256", "model_epoch_kind",
             "canonical_tensor_sha256", "tensor_json", "tensor_sha256", "record_sources")
    if version == "rz-pals-native-input-sidecar/2":
        names += ("model_profile", "encoding_profile")
    names += ("sha256",)
    value = _fields(sidecar, names, "native tensor sidecar")
    s = record["input"]["snapshot"]
    _sha(expected_encoder_source_sha256)
    if value["version"] not in ("rz-pals-native-input-sidecar/1", "rz-pals-native-input-sidecar/2") or value["input_sha256"] != record["input"]["sha256"] or value["encoding_sha256"] != s["encoding_sha256"] or value["encoder_source_sha256"] != expected_encoder_source_sha256:
        raise ValueError("native sidecar input/encoder identity mismatch")
    config = ModelConfig.for_profile(value["model_profile"]) if version == "rz-pals-native-input-sidecar/2" else ModelConfig.baseline()
    if version == "rz-pals-native-input-sidecar/2" and (config.profile == "legacy_summary_v1" or value["encoding_profile"] != config.encoding):
        raise ValueError("native /2 sidecar requires exact registered V2 model/encoding profile")
    for name in ("canonical_tensor_sha256", "tensor_sha256", "sha256"):
        _sha(value[name])
    if not isinstance(value["tensor_json"], str) or len(value["tensor_json"].encode()) > 2 * 1024 * 1024 or hashlib.sha256(value["tensor_json"].encode()).hexdigest() != value["tensor_sha256"]:
        raise ValueError("native tensor bytes changed or exceed allocation bound")
    sources = [_fields(r, ("observation_sha256", "situation_revision"), "native record source") for r in value["record_sources"]]
    outer = [value[name] if name != "record_sources" else sources for name in names[:-1]]
    outer_sha = hashlib.sha256(json.dumps(outer, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    if outer_sha != value["sha256"] or sources != s["public_records"]:
        raise ValueError("native sidecar seal or captured record order mismatch")
    tensor_names = ("role", "board", "metadata", "records", "required_critical_records", "candidates", "divergence_features", "query", "situation_revision", "history_digest", "model_epoch")
    if config.full_line:
        tensor_names += ("full_line",)
    tensor = _fields(_unique_json(value["tensor_json"]), tensor_names, "native input")
    if tensor["role"] != ROLE_NAMES[s["role"]] or _uint(tensor["situation_revision"]) != s["input_revision"] or tuple(tensor["board"]) != _fen_board(s["board_fen"]):
        raise ValueError("native input role/revision/board mismatch")
    for name in ("history_digest", "model_epoch"):
        if not isinstance(tensor[name], list) or len(tensor[name]) != 32:
            raise ValueError("native history/epoch shape")
        for byte in tensor[name]:
            _uint(byte, 255)
    if bytes(tensor["history_digest"]).hex() != s["rules_history_sha256"]:
        raise ValueError("native tensor exact history mismatch")
    if value["model_epoch_kind"] == "encoding_only_zero":
        if s["source"]["kind"] != "own_cpu" or any(tensor["model_epoch"]):
            raise ValueError("encoding-only epoch requires own CPU and zero native epoch")
    elif value["model_epoch_kind"] == "frozen_model_epoch":
        if s["source"]["kind"] != "own_pals" or expected_model_epoch is None or bytes(tensor["model_epoch"]).hex() != _sha(expected_model_epoch):
            raise ValueError("neural sidecar requires independently verified frozen epoch")
    else:
        raise ValueError("unknown native model epoch kind")
    if not isinstance(tensor["candidates"], list) or len(tensor["candidates"]) > 256:
        raise ValueError("native candidate allocation bound")
    candidates = []
    for candidate in tensor["candidates"]:
        candidate = _fields(candidate, ("from", "to", "promotion"), "native candidate")
        candidates.append((_uint(candidate["from"], 63), _uint(candidate["to"], 63), _uint(candidate["promotion"], 4)))
    if candidates != [move_components(v) for v in s["legal_moves"]]:
        raise ValueError("native legal candidate order mismatch")
    if not isinstance(tensor["records"], list) or len(tensor["records"]) != len(sources) or len(sources) > 128 or not isinstance(tensor["required_critical_records"], list) or len(tensor["required_critical_records"]) > 128:
        raise ValueError("native record allocation/order extent")
    critical, ids, public = set(), set(), []
    for source, token in zip(sources, tensor["records"]):
        token = _fields(token, ("record_id", "revision", "critical", "features"), "native record token")
        identity = _uint(token["record_id"])
        if identity in ids or type(token["critical"]) is not bool or _uint(token["revision"]) != _uint(source["situation_revision"]):
            raise ValueError("native record duplicate/revision/status")
        ids.add(identity)
        if token["critical"]:
            critical.add(identity)
        public.append((source["observation_sha256"], source["situation_revision"], tuple(token["features"])))
    required = tensor["required_critical_records"]
    for identity in required:
        _uint(identity)
    if len(set(required)) != len(required) or not set(required) <= critical:
        raise ValueError("native input missing required critical records")
    if tensor["divergence_features"]:
        raise ValueError("native divergence features require explicit challenged-line/ply descriptors")
    result = EncodedSnapshot(value["input_sha256"], value["encoding_sha256"], tuple(tensor["board"]), tuple(tensor["metadata"]), tuple(public), tuple(tensor["query"]),
                             model_profile=config.profile, full_line=copy.deepcopy(tensor.get("full_line")))
    _validate_encoding(result)
    return result


class _CollectionArtifactReader:
    """One aggregate budget for stable regular-file reads, including registrations."""

    def __init__(self, maximum):
        if _uint(maximum, 1024 * 1024 * 1024) == 0:
            raise ValueError("finite collection input byte limit required")
        self.maximum, self.consumed = maximum, 0

    def read(self, path, asset=None, *, maximum=None):
        path = Path(path)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("collection file must be a regular file")
        remaining = self.maximum - self.consumed
        if before.st_size > remaining or (maximum is not None and before.st_size > maximum):
            raise ValueError("collection input allocation budget exceeded")
        def identity(metadata):
            return (metadata.st_dev, metadata.st_ino, metadata.st_size,
                    metadata.st_mtime_ns, metadata.st_ctime_ns)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        try:
            stream = os.fdopen(descriptor, "rb", buffering=0)
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or identity(opened) != identity(before):
                raise ValueError("collection file changed before bounded read")
            limit = min(before.st_size + 1, remaining + 1)
            content = bytearray()
            while len(content) < limit:
                block = stream.read(min(65536, limit - len(content)))
                if not block:
                    break
                content.extend(block)
            data = bytes(content)
            after = os.fstat(stream.fileno())
        final = path.lstat()
        if (len(data) != before.st_size or len(data) > remaining
                or not stat.S_ISREG(after.st_mode) or not stat.S_ISREG(final.st_mode)
                or identity(after) != identity(before) or identity(final) != identity(before)):
            raise ValueError("collection file changed or exceeded allocation budget")
        self.consumed += len(data)
        if asset is not None:
            asset = _fields(asset, ("sha256", "bytes"), "collection artifact")
            if _uint(asset["bytes"]) != len(data) or _sha(asset["sha256"]) != hashlib.sha256(data).hexdigest():
                raise ValueError("collection artifact byte identity mismatch")
        return data


def load_collected_dataset(directory, *, expected_receipt_sha256, expected_encoder_source_sha256,
                           max_input_bytes=64 * 1024 * 1024):
    """Read only fixed collector filenames after independent receipt admission.

    Every consumed JSONL file is byte/digest checked before parsing. A failed or
    incomplete collection never becomes an admitted training dataset. This only
    loads data; it neither collects games nor updates an optimizer.
    """
    _sha(expected_receipt_sha256)
    _sha(expected_encoder_source_sha256)
    directory = Path(directory).expanduser().resolve()
    reader = _CollectionArtifactReader(max_input_bytes)

    def read(name, asset=None):
        return reader.read(directory / name, asset)

    receipt_bytes = read("receipt.json")
    if hashlib.sha256(receipt_bytes).hexdigest() != expected_receipt_sha256:
        raise ValueError("collection receipt differs from independently registered digest")
    receipt = _unique_json(receipt_bytes.decode("utf-8"))
    if not isinstance(receipt, dict) or receipt.get("version") != "rz-pals-own-collector/1" or receipt.get("complete") is not True or receipt.get("failure") is not None or receipt.get("actual_training_executed") is not False or receipt.get("external_teacher_used") is not False or not isinstance(receipt.get("audit"), dict) or receipt.get("source", {}).get("encoder_source_sha256") != expected_encoder_source_sha256:
        raise ValueError("collection incomplete, failed, external or unregistered encoder")
    artifacts = receipt.get("artifacts")
    required = ("records.jsonl", "native-inputs.jsonl", "source-registry.jsonl", "split.jsonl")
    if not isinstance(artifacts, dict) or not all(name in artifacts for name in required):
        raise ValueError("collection receipt missing required asset identities")
    values = {}
    for name in required:
        raw = read(name, artifacts[name])
        lines = raw.decode("utf-8").splitlines()
        if not lines or len(lines) > 65536 or any(not line or len(line.encode()) > 2 * 1024 * 1024 for line in lines):
            raise ValueError("collection JSONL record/line bound")
        values[name] = [_unique_json(line) for line in lines]
    if len(values["source-registry.jsonl"]) != 1 or len(values["split.jsonl"]) != 1:
        raise ValueError("collection requires exactly one source registry and split")
    rows = values["records.jsonl"]
    registry, split = values["source-registry.jsonl"][0], values["split.jsonl"][0]
    # Validate /2 rows and their independently captured authority before
    # consulting sidecars. Sidecars cannot enroll new model or CPU sources.
    result = ValidatedDataset(rows, split, registry, {})
    frozen = {row["input"]["sha256"]: row for row in rows}
    encodings = {}
    source_description = receipt["source"]
    model_epoch = source_description.get("model_epoch")
    expected_epoch = None
    if source_description.get("model_epoch_kind") == "frozen_model_epoch":
        if not isinstance(model_epoch, list) or len(model_epoch) != 32:
            raise ValueError("receipt model epoch shape")
        for byte in model_epoch:
            _uint(byte, 255)
        expected_epoch = bytes(model_epoch).hex()
    for sidecar in values["native-inputs.jsonl"]:
        row = frozen.get(sidecar.get("input_sha256")) if isinstance(sidecar, dict) else None
        if row is None:
            raise ValueError("sidecar references an uncaptured historical input")
        encoded = encoded_from_sidecar(sidecar, row, expected_encoder_source_sha256=expected_encoder_source_sha256,
                                       expected_model_epoch=expected_epoch)
        if encoded.input_sha256 in encodings and encodings[encoded.input_sha256] != encoded:
            raise ValueError("conflicting encodings for one immutable input")
        encodings[encoded.input_sha256] = encoded
    if set(encodings) != set(frozen):
        raise ValueError("collection lacks required input encodings")
    for name in ("canonical_dataset_sha256", "canonical_split_sha256"):
        _sha(receipt["audit"].get(name))
    if _uint(receipt["audit"].get("records")) != len(rows):
        raise ValueError("collection audit row count mismatch")
    result._attach_encodings(encodings)
    result.collection_receipt = copy.deepcopy(receipt)
    return result


FROZEN_ADMISSION_DOMAIN = "rz-pals-frozen-collected-admission/1"
PRODUCER_REGISTRATION_DOMAIN = "rz-pals-collector-producer-registration/1"
CHECKED_SOURCE_DOMAIN = "rz-pals-collector-checked-source/1"
PREPARED_PRODUCER_DOMAIN = "rz-pals-collector-prepared-producer/1"


def _collection_jsonl(raw, *, max_rows=65536):
    if not raw.endswith(b"\n"):
        raise ValueError("complete exact JSONL newline required")
    lines = raw.split(b"\n")[:-1]
    if not 1 <= len(lines) <= max_rows or any(not line or len(line) > 2 * 1024 * 1024 for line in lines):
        raise ValueError("collection JSONL record/line bound")
    return [(line, _unique_json(line.decode("utf-8"))) for line in lines]


def _byte_pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _epoch_hex(value):
    if not isinstance(value, list) or len(value) != 32:
        raise ValueError("checked native epoch shape")
    for byte in value:
        _uint(byte, 255)
    return bytes(value).hex()


def _checked_producer_source(raw, registration, pin):
    """Inspect actual independently pinned driver facts, never infer a teacher."""
    value = _unique_json(raw.decode("utf-8"))
    if not isinstance(value, list) or len(value) != 2 or value[0] != CHECKED_SOURCE_DOMAIN:
        raise ValueError("actual checked producer source artifact required")
    source = value[1]
    names = ("mode", "source", "cpu_profile_sha256", "implementation_sha256", "encoding_sha256",
             "encoder_source_sha256", "configuration", "model_epoch", "model_epoch_kind", "frozen_epoch")
    if not isinstance(source, dict) or not set(names) <= set(source) or set(source) - set(names) - {"native"}:
        raise ValueError("checked producer source facts fields")
    policy = pin["encoding_policy"]
    if policy["kind"] != "native_exact":
        raise ValueError("private producer requires an explicit checked derived adapter")
    if (source["source"] != pin["source"] or source["frozen_epoch"] != pin["frozen_epoch"]
            or source["encoding_sha256"] != policy["encoding_sha256"]
            or source["encoder_source_sha256"] != policy["encoder_source_sha256"]
            or _byte_pin(raw)["sha256"] != registration["checked_source_sha256"]):
        raise ValueError("actual checked source differs from independent producer pin")
    _identity(source["mode"])
    for name in ("cpu_profile_sha256", "implementation_sha256", "encoding_sha256", "encoder_source_sha256"):
        _sha(source[name])
    _uint(source["frozen_epoch"])
    if not isinstance(source["configuration"], dict):
        raise ValueError("actual producer configuration required")
    epoch = _epoch_hex(source["model_epoch"])
    if pin["source"]["kind"] == "own_cpu":
        identity = source["configuration"].get("value_identity")
        if (source["implementation_sha256"] != pin["source"]["cpu_binary_sha256"]
                or source["cpu_profile_sha256"] != pin["source"]["evaluator_configuration_sha256"]
                or not isinstance(identity, dict)
                or identity.get("weights_sha256") != pin["source"]["model_weights_sha256"]
                or not isinstance(identity.get("training"), dict) or not isinstance(identity.get("semantics"), str)
                or epoch != "0" * 64 or source["model_epoch_kind"] != "encoding_only_zero"
                or source["frozen_epoch"] != 0 or source.get("native") is not None):
            raise ValueError("actual CPU evaluator/binary/encoding-only source mismatch")
    else:
        native = source.get("native")
        if not isinstance(native, dict):
            raise ValueError("actual loaded native source facts required")
        registry, loaded = native.get("independent_registry"), native.get("loaded_source")
        if not isinstance(registry, dict) or not isinstance(loaded, dict) or not isinstance(loaded.get("execution"), dict):
            raise ValueError("native independent registry/loaded constructor facts required")
        execution = loaded["execution"]
        if (registry.get("version") != "rz-pals-native-collection-registry/1"
                or registry.get("training_state") != "untrained" or loaded.get("trained") is not False
                or native.get("training_state") != "Untrained" or native.get("actual_training_executed") is not False
                or native.get("precision") != "fp32" or execution.get("provider") != "cpu"
                or source["implementation_sha256"] != registry.get("collector_binary_sha256")
                or pin["source"]["model_weights_sha256"] != registry.get("checkpoint_sha256")
                or epoch != pin["source"]["model_weights_sha256"]
                or epoch != policy["native_model_epoch"].get("sha256")
                or epoch != _epoch_hex(loaded.get("checkpoint_sha256"))
                or source["model_epoch_kind"] != "frozen_model_epoch" or source["frozen_epoch"] == 0
                or source["frozen_epoch"] != loaded.get("frozen_epoch") or source["frozen_epoch"] != registry.get("frozen_epoch")
                or pin["source"]["model_configuration_sha256"] != registry.get("model_configuration_sha256")
                or source["configuration"] != registry.get("model_configuration")
                or source["configuration"] != loaded.get("model_configuration")
                or source["encoding_sha256"] != registry.get("encoding_sha256")
                or source["encoder_source_sha256"] != registry.get("encoder_source_sha256")
                or registry.get("export_manifest_sha256") != _epoch_hex(loaded.get("export_manifest_sha256"))
                or registry.get("runtime_sha256") != _epoch_hex(execution.get("runtime_sha256"))
                or registry.get("provider") != execution.get("provider") or native.get("provider") != execution.get("provider")):
            raise ValueError("actual native checkpoint/epoch/encoder/runtime source mismatch")
        declared_graphs, loaded_graphs = registry.get("graphs"), native.get("graphs")
        if not isinstance(declared_graphs, list) or not 1 <= len(declared_graphs) <= 4 or not isinstance(loaded_graphs, list):
            raise ValueError("actual native graph inventory required")
        def graph_key(graph):
            _fields(graph, ("role", "sha256", "serialized_bytes"), "native graph source")
            _identity(graph["role"])
            if _uint(graph["serialized_bytes"], 256 * 1024 * 1024) == 0:
                raise ValueError("actual native graph bytes required")
            return graph["role"], _sha(graph["sha256"]), graph["serialized_bytes"]
        expected = sorted(graph_key(graph) for graph in declared_graphs)
        actual = sorted(graph_key(graph) for graph in loaded_graphs)
        if len({key[0] for key in expected}) != len(expected) or actual != expected:
            raise ValueError("actual loaded graph identities differ from registry")
    return source


def load_frozen_collected_dataset(directory, *, expected_receipt_sha256, producer_registrations,
                                  max_input_bytes=64 * 1024 * 1024, continuation_registration=None):
    """Strict CPU preparation admission; a missing/failed artifact never retries legacy.

    Each independent entry supplies ``pin`` (the complete game/producer roster
    declaration), ``registration_path`` (prior registered bytes), and
    ``checked_source_path`` (actual checked driver facts). The registration file
    bytes SHA is the pin's registration_sha256. The owner's independent receipt
    pin must come from the completed collector/audit, not this loader's output.
    Native callbacks and raw row bytes are checked in addition to metadata.
    Private derived encodings require a separate adapter and are refused here.
    Multi-Reply continuation/2 requires its independent policy registration and
    original journals. Its extra audit is observation only, never target authority.
    """
    from . import frozen_producer as metadata

    _sha(expected_receipt_sha256)
    directory = Path(directory).expanduser().resolve()
    reader = _CollectionArtifactReader(max_input_bytes)
    cache = {}

    def read(path, asset=None, maximum=None):
        path = Path(path).expanduser().absolute()
        key = str(path)
        if key not in cache:
            cache[key] = reader.read(path, asset, maximum=maximum)
        raw = cache[key]
        if maximum is not None and len(raw) > maximum:
            raise ValueError("frozen artifact byte extent")
        if asset is not None and _fields(asset, ("bytes", "sha256"), "frozen artifact") != _byte_pin(raw):
            raise ValueError("frozen artifact differs from actual byte pin")
        return raw

    receipt_bytes = read(directory / "receipt.json", maximum=metadata.MAX_MANIFEST_BYTES)
    if _byte_pin(receipt_bytes)["sha256"] != expected_receipt_sha256:
        raise ValueError("frozen collection receipt differs from independent pin")
    receipt = _unique_json(receipt_bytes.decode("utf-8"))
    if (not isinstance(receipt, dict) or receipt.get("version") != "rz-pals-own-collector/1"
            or receipt.get("complete") is not True or receipt.get("failure") is not None
            or receipt.get("actual_training_executed") is not False or receipt.get("external_teacher_used") is not False
            or not isinstance(receipt.get("audit"), dict) or not isinstance(receipt.get("artifacts"), dict)):
        raise ValueError("strict frozen collection requires a completed owned raw audit")
    required = ("records.jsonl", "native-inputs.jsonl", "source-registry.jsonl", "split.jsonl",
                "inputs.jsonl", "input-lineage.jsonl", "producer-registration.json", "producer-source.json",
                "producer-roster.json", "producer-captures.json", "producer-envelope.json", "producer-audit.json",
                "producer-prepared.jsonl")
    artifacts, raw_files = receipt["artifacts"], {}
    for name in required:
        if name not in artifacts:
            raise ValueError("strict collection missing required producer artifact: " + name)
        cap = metadata.MAX_MANIFEST_BYTES if name.endswith(".json") else max_input_bytes
        raw_files[name] = read(directory / name, artifacts[name], cap)
    roster = metadata.load_roster(raw_files["producer-roster.json"])
    capture = metadata.load_capture(raw_files["producer-captures.json"])
    envelope = metadata.load_envelope(raw_files["producer-envelope.json"])
    if not isinstance(producer_registrations, (list, tuple)) or not 1 <= len(producer_registrations) <= 65536:
        raise ValueError("independent actual producer registrations required")
    supplied, sources, source_digests, registration_assets = [], {}, {}, []
    for entry in producer_registrations:
        _fields(entry, ("pin", "registration_path", "checked_source_path"), "independent producer registration")
        pin = metadata.normalize_roster({"version": metadata.ROSTER_DOMAIN, "game_producers": [entry["pin"]]})["game_producers"][0]
        if pin["encoding_policy"]["kind"] != "native_exact":
            raise ValueError("private producer requires an explicit checked derived adapter")
        registration_bytes = read(entry["registration_path"], maximum=metadata.MAX_MANIFEST_BYTES)
        if _byte_pin(registration_bytes)["sha256"] != pin["registration_sha256"]:
            raise ValueError("actual producer registration bytes differ from owner pin")
        registration = _fields(_unique_json(registration_bytes.decode("utf-8")),
                               ("version", "producer_id", "source", "frozen_epoch", "encoding_policy", "checked_source_sha256"),
                               "actual producer registration")
        _identity(registration["producer_id"])
        _uint(registration["frozen_epoch"])
        if registration["version"] != PRODUCER_REGISTRATION_DOMAIN or any(registration[name] != pin[name] for name in ("producer_id", "source", "frozen_epoch", "encoding_policy")):
            raise ValueError("actual prior registration differs from independent game producer")
        _sha(registration["checked_source_sha256"])
        checked_bytes = read(entry["checked_source_path"], maximum=metadata.MAX_MANIFEST_BYTES)
        checked = _checked_producer_source(checked_bytes, registration, pin)
        key = pin["game_id"], pin["producer_id"]
        if key in sources:
            raise ValueError("duplicate independent game producer registration")
        supplied.append(pin)
        sources[key] = checked
        source_digests[key] = _byte_pin(checked_bytes)["sha256"]
        registration_assets.append({"pin": pin, "registration": _byte_pin(registration_bytes), "checked_source": _byte_pin(checked_bytes)})
    declarations = roster["roster"]["game_producers"]
    if metadata.normalize_roster({"version": metadata.ROSTER_DOMAIN, "game_producers": supplied})["game_producers"] != declarations:
        raise ValueError("actual required producer roster differs from independent registration set")
    own_registration_sha = _byte_pin(raw_files["producer-registration.json"])["sha256"]
    own_source_sha = _byte_pin(raw_files["producer-source.json"])["sha256"]
    if not any(asset["registration"]["sha256"] == own_registration_sha and asset["checked_source"]["sha256"] == own_source_sha for asset in registration_assets):
        raise ValueError("collector's actual registration/source bytes lack independent authority")
    own_description = _unique_json(raw_files["producer-source.json"].decode("utf-8"))[1]
    if receipt.get("source") != own_description:
        raise ValueError("receipt source differs from actual checked collector source bytes")
    parsed = {name: _collection_jsonl(raw_files[name], max_rows=65536 * 8 if name in ("input-lineage.jsonl", "producer-prepared.jsonl") else 65536)
              for name in ("records.jsonl", "native-inputs.jsonl", "source-registry.jsonl", "split.jsonl", "inputs.jsonl", "input-lineage.jsonl", "producer-prepared.jsonl")}
    if len(parsed["source-registry.jsonl"]) != 1 or len(parsed["split.jsonl"]) != 1:
        raise ValueError("one actual source registry and split required")
    rows = [value for _, value in parsed["records.jsonl"]]
    registry, split = parsed["source-registry.jsonl"][0][1], parsed["split.jsonl"][0][1]
    result = ValidatedDataset(rows, split, registry, {})
    audit = metadata.audit_metadata(roster_bytes=raw_files["producer-roster.json"],
                                    envelope_bytes=raw_files["producer-envelope.json"], capture_bytes=raw_files["producer-captures.json"],
                                    independently_registered=supplied, source_registry_bytes=raw_files["source-registry.jsonl"],
                                    raw_receipt_bytes=receipt_bytes, records_bytes=raw_files["records.jsonl"],
                                    expected_raw_receipt_sha256=expected_receipt_sha256,
                                    checked_current_view_sha256=result.current_view.sha256, max_input_bytes=max_input_bytes)
    if audit["requires_derived_adapter"]:
        raise ValueError("metadata acceptance requires an explicit checked derived adapter")
    actual_audit = _fields(_unique_json(raw_files["producer-audit.json"].decode("utf-8")),
                           ("scope", "raw_records", "unique_inputs", "roster_sha256", "envelope_sha256", "capture_sha256",
                            "native_exact_metadata_inputs", "requires_derived_adapter"), "actual producer audit")
    for name in ("raw_records", "unique_inputs", "native_exact_metadata_inputs"):
        _uint(actual_audit[name], 65536)
    for name in ("roster_sha256", "envelope_sha256", "capture_sha256"):
        _sha(actual_audit[name])
    if actual_audit != audit:
        raise ValueError("actual producer audit differs from checked whole-history metadata")
    bindings = {value["input_sha256"]: value for value in capture["capture"]["bindings"]}
    pins = {(pin["game_id"], pin["producer_id"]): pin for pin in declarations}
    frozen = {row["input"]["sha256"]: row for row in rows}
    actual_game_values = {row["input"]["sha256"] for row in rows if row["future_label"] is not None
                          and row["future_label"]["provenance"].get("source") == "actual_game"
                          and row["future_label"]["value_wdl"] is not None}
    journals = {}
    for _, journal in parsed["producer-prepared.jsonl"]:
        _fields(journal, ("prepared", "sha256"), "actual prepared producer evidence")
        body = _fields(journal["prepared"], ("version", "producer_id", "registration_sha256", "roster_sha256", "checked_source_sha256", "game_id", "input_sha256", "capture_sequence", "input_json", "tensor_sidecar_json", "lineage_json", "native_request", "learning_input", "publication"), "prepared producer evidence")
        if body["version"] != PREPARED_PRODUCER_DOMAIN or _sha(journal["sha256"]) != _sorted_canonical(PREPARED_PRODUCER_DOMAIN, body):
            raise ValueError("actual prepared evidence seal mismatch")
        for name in ("producer_id", "game_id"):
            _identity(body[name])
        for name in ("registration_sha256", "roster_sha256", "checked_source_sha256"):
            _sha(body[name])
        _uint(body["capture_sequence"])
        _sha(body["input_sha256"])
        for name in ("input_json", "tensor_sidecar_json", "lineage_json"):
            asset = _fields(body[name], ("bytes", "sha256"), "prepared exact row bytes")
            if _uint(asset["bytes"], 2 * 1024 * 1024) == 0:
                raise ValueError("empty prepared exact row evidence")
            _sha(asset["sha256"])
        if type(body["learning_input"]) is not bool:
            raise ValueError("prepared learning admission flag")
        if body["publication"] not in ("append-before-analysis; sync-at-receipt-close", "seal-before-submit; prepaid-drain-after-search"):
            raise ValueError("prepared publication contract")
        if body["native_request"] is not None:
            if not isinstance(body["native_request"], list) or len(body["native_request"]) != 2:
                raise ValueError("prepared native request shape")
            for item in body["native_request"]:
                _uint(item)
        if journal["sha256"] in journals:
            raise ValueError("duplicate actual prepared evidence")
        journals[journal["sha256"]] = body
    def exact_rows(name, identity_name):
        result = {}
        for line, value in parsed[name]:
            identity = value.get(identity_name) if isinstance(value, dict) else None
            _sha(identity)
            result.setdefault(identity, []).append((_byte_pin(line), value))
        return result
    actual_inputs = exact_rows("inputs.jsonl", "sha256")
    actual_sidecars = exact_rows("native-inputs.jsonl", "input_sha256")
    actual_lineage = exact_rows("input-lineage.jsonl", "input_sha256")
    native_events = None
    encodings, admitted_inputs, full_line_parents = {}, [], {}
    for identity, binding in bindings.items():
        row = frozen[identity]
        pin = pins[(binding["game_id"], binding["producer_id"])]
        prepared = journals.get(binding["capture_evidence_sha256"])
        if prepared is None or prepared["learning_input"] is not True:
            raise ValueError("missing actual learning-input prepared evidence")
        if (prepared["input_sha256"] != identity or prepared["game_id"] != binding["game_id"]
                or prepared["producer_id"] != binding["producer_id"] or prepared["capture_sequence"] != binding["capture_sequence"]
                or prepared["registration_sha256"] != pin["registration_sha256"] or prepared["roster_sha256"] != roster["sha256"]
                or prepared["checked_source_sha256"] != source_digests[(binding["game_id"], binding["producer_id"])]):
            raise ValueError("actual capture journal producer/input/source attribution mismatch")
        def matching(name, pin_name, candidates):
            matches = [value for asset, value in candidates.get(identity, []) if asset == prepared[pin_name]]
            if len(matches) != 1:
                raise ValueError("prepared evidence lacks exact actual " + name + " row bytes")
            return matches[0]
        actual_input = matching("input", "input_json", actual_inputs)
        sidecar = matching("tensor sidecar", "tensor_sidecar_json", actual_sidecars)
        lineage = matching("lineage", "lineage_json", actual_lineage)
        if actual_input != row["input"] or lineage.get("game_id") != binding["game_id"] or lineage.get("actual_played_history") != row["input"]["snapshot"]["actual_history"]:
            raise ValueError("actual prepared input/lineage differs from admitted historical input")
        if type(lineage.get("actual_outcome_eligible")) is not bool:
            raise ValueError("actual lineage outcome eligibility must be explicit")
        if identity in actual_game_values and not lineage["actual_outcome_eligible"]:
            raise ValueError("counterfactual input cannot consume actual-game WDL")
        request = prepared["native_request"]
        if pin["source"]["kind"] == "own_cpu":
            if request is not None or prepared["publication"] != "append-before-analysis; sync-at-receipt-close":
                raise ValueError("CPU capture claims an unsupported native request/publication")
        else:
            if not isinstance(request, list) or len(request) != 2 or prepared["publication"] != "seal-before-submit; prepaid-drain-after-search":
                raise ValueError("native prepared request evidence required")
            for item in request:
                _uint(item)
            if lineage.get("process_epoch") != request[0] or lineage.get("request_sequence") != request[1]:
                raise ValueError("native prepared lineage request mismatch")
            _uint(lineage["process_epoch"])
            _uint(lineage["request_sequence"])
            if native_events is None:
                if "native-events.jsonl" not in artifacts:
                    raise ValueError("native prepared evidence lacks actual observer events")
                event_bytes = read(directory / "native-events.jsonl", artifacts["native-events.jsonl"])
                raw_files["native-events.jsonl"] = event_bytes
                native_events = {}
                for _, event in _collection_jsonl(event_bytes, max_rows=65536 * 8):
                    if not isinstance(event, dict) or event.get("domain") != "rz-pals-native-call-event/1":
                        raise ValueError("actual prepared observer event domain")
                    _uint(event.get("process_epoch"))
                    _uint(event.get("request_sequence"))
                    _sha(event.get("input_sha256"))
                    _identity(event.get("game_id"))
                    _identity(event.get("stage"))
                    if event["stage"] == "prepared":
                        key = event["game_id"], event["input_sha256"], event["process_epoch"], event["request_sequence"]
                        native_events[key] = native_events.get(key, 0) + 1
            if native_events.get((binding["game_id"], identity, request[0], request[1])) != 1:
                raise ValueError("native request lacks one actual prepared observer event")
        policy = pin["encoding_policy"]
        expected_epoch = policy["native_model_epoch"].get("sha256")
        if sidecar.get("model_epoch_kind") != policy["native_model_epoch"]["kind"]:
            raise ValueError("sidecar epoch kind differs from input's actual producer")
        encoding = encoded_from_sidecar(sidecar, row, expected_encoder_source_sha256=policy["encoder_source_sha256"], expected_model_epoch=expected_epoch)
        checked_source = sources[(binding["game_id"], binding["producer_id"])]
        if pin["source"]["kind"] == "own_pals":
            registered_profile = checked_source["configuration"].get("profile", "legacy_summary_v1")
            if registered_profile != encoding.model_profile:
                raise ValueError("actual sidecar model profile differs from independent loaded producer")
            if encoding.model_profile != "legacy_summary_v1":
                checked_config = ModelConfig(**checked_source["configuration"])
                checked_config.validate()
        encodings[identity] = encoding
        if encoding.full_line is not None:
            full_line_parents[identity] = {"sidecar": sidecar, "source": checked_source, "request": request}
        admitted_inputs.append({"binding": binding, "producer": pin, "prepared_evidence_sha256": binding["capture_evidence_sha256"]})
    if set(encodings) != set(frozen):
        raise ValueError("strict frozen collection lacks per-input producer encoding")
    full_line_observation, coverage_observation = None, None
    from . import target_coverage as coverage_consumer
    if full_line_parents and coverage_consumer.FULL_LINE_ARTIFACT not in artifacts:
        raise ValueError("strict V2 collection lacks exact full-line context artifact")
    if coverage_consumer.FULL_LINE_ARTIFACT in artifacts:
        name = coverage_consumer.FULL_LINE_ARTIFACT
        raw_files[name] = read(directory / name, artifacts[name])
        full_line_observation = coverage_consumer.inspect_full_line_contexts(
            raw_files[name], encoded=encodings, exact_parents=full_line_parents)
    if coverage_consumer.COVERAGE_ARTIFACT in artifacts:
        name = coverage_consumer.COVERAGE_ARTIFACT
        raw_files[name] = read(directory / name, artifacts[name])
        coverage_observation = coverage_consumer.inspect_collector_target_coverage(raw_files[name], dataset=result)
    elif full_line_parents:
        raise ValueError("strict V2 collection lacks actual target coverage artifact")
    continuation = None
    if "native-continuation-traces.jsonl" in artifacts:
        from . import native_continuation as continuation_consumer
        declaration = _fields(continuation_registration, ("registration_path", "sha256"),
                              "independent continuation registration")
        policy_path = Path(declaration["registration_path"])
        if not policy_path.is_absolute() or any(
                part.lower() == ".env" or part.lower().startswith((".env.", "id_rsa", "id_ed25519"))
                or part.lower().endswith((".key", ".pem")) or "credential" in part.lower()
                or "service-account" in part.lower() or "service_account" in part.lower() for part in policy_path.parts):
            raise ValueError("continuation registration requires an absolute public data path")
        policy_bytes = read(policy_path, maximum=32 * 1024)
        continuation_files = {name: read(directory / name, artifacts.get(name))
                              for name in continuation_consumer.REQUIRED_ARTIFACTS if name in artifacts}
        continuation = continuation_consumer.audit_native_continuation_collection(
            receipt_bytes=receipt_bytes, artifact_bytes=continuation_files,
            expected_receipt_sha256=expected_receipt_sha256, policy_registration_bytes=policy_bytes,
            expected_policy_registration_sha256=_sha(declaration["sha256"]), max_input_bytes=max_input_bytes)
    elif continuation_registration is not None:
        raise ValueError("continuation registration supplied without original continuation trace")
    result._attach_encodings(encodings)
    result.collection_receipt = copy.deepcopy(receipt)
    result.native_continuation_observation = copy.deepcopy(continuation)
    result.full_line_context_observation = copy.deepcopy(full_line_observation)
    result.target_coverage_observation = copy.deepcopy(coverage_observation)
    result._attach_frozen_admission({"version": FROZEN_ADMISSION_DOMAIN, "receipt": _byte_pin(receipt_bytes),
                                     "artifacts": {name: _byte_pin(raw) for name, raw in raw_files.items()},
                                     "producer_registrations": registration_assets, "inputs": admitted_inputs,
                                     "metadata_audit": audit, "owned_sources": registry,
                                     "raw_dataset_sha256": envelope["envelope"]["raw_dataset_sha256"],
                                     "split_sha256": envelope["envelope"]["split_sha256"],
                                     "current_view_sha256": result.current_view.sha256,
                                     **({"full_line_context_observation": full_line_observation} if full_line_observation is not None else {}),
                                     **({"target_coverage_observation": coverage_observation} if coverage_observation is not None else {}),
                                     **({"native_continuation_observation": continuation} if continuation is not None else {})})
    return result


@dataclass(frozen=True)
class TrainingBatch:
    role: str
    inputs: TensorInput
    input_sha256: tuple
    policy: torch.Tensor
    policy_mask: torch.Tensor
    wdl: torch.Tensor
    wdl_mask: torch.Tensor
    divergence: torch.Tensor
    divergence_mask: torch.Tensor
    task: torch.Tensor
    task_mask: torch.Tensor
    model_profile: str = "legacy_summary_v1"


class ValidatedDataset:
    def __init__(self, records, split, owned_sources, encodings):
        if not isinstance(records, list) or not 1 <= len(records) <= 65536:
            raise ValueError("dataset record extent")
        authority = _fields(owned_sources, ("cpu_binary_sha256", "input_sources"), "owned source registry")
        cpu = authority["cpu_binary_sha256"]
        sources = authority["input_sources"]
        if not isinstance(cpu, list) or len(cpu) > 64 or not isinstance(sources, list) or not 1 <= len(sources) <= 256:
            raise ValueError("source registry extent")
        for sha in cpu:
            _sha(sha)
        for source in sources:
            _source(source)
            if source["kind"] == "own_cpu" and source["cpu_binary_sha256"] not in cpu:
                raise ValueError("CPU source missing owned binary")
        split = _fields(split, ("games",), "split")["games"]
        if not isinstance(split, dict) or len(split) > 65536 or any(v not in ("train", "validation", "holdout") for v in split.values()):
            raise ValueError("invalid immutable split")
        groups = [dict() for _ in range(4)]
        present = set()
        for record in records:
            _validate_record(record, cpu, sources)
            s = record["input"]["snapshot"]
            assignment = split.get(s["game_id"])
            if assignment is None:
                raise ValueError("unassigned game")
            present.add(s["game_id"])
            for group, name in zip(groups, ("opening_id", "line_genealogy_id", "rules_state_sha256", "transposition_sha256")):
                if group.setdefault(s[name], assignment) != assignment:
                    raise ValueError("game/opening/line/state/transposition split leak")
        if set(split) != present:
            raise ValueError("split includes an unobserved game")
        self.records, self.split = copy.deepcopy(records), dict(split)
        self.owned_sources = copy.deepcopy(authority)
        self.frozen_admission, self._frozen_admission_identity = None, None
        self._row_identities = tuple(_canonical("rz-pals-python-immutable-row/1", row) for row in self.records)
        self._split_identity = _canonical("rz-pals-python-immutable-split/1", self.split)
        self._current_view = current_label_view(self.records)
        self._current_indices = frozenset(self._current_view.current_indices)
        self._attach_encodings(encodings)

    @property
    def current_view(self):
        return self._current_view

    def _verify_raw_integrity(self):
        if len(self.records) != len(self._row_identities) or any(
                _canonical("rz-pals-python-immutable-row/1", row) != identity
                for row, identity in zip(self.records, self._row_identities)):
            raise ValueError("immutable raw dataset history changed after admission")
        if _canonical("rz-pals-python-immutable-split/1", self.split) != self._split_identity:
            raise ValueError("immutable dataset split changed after admission")
        self._verify_frozen_admission()

    def _attach_frozen_admission(self, admission):
        if self._frozen_admission_identity is not None:
            raise ValueError("checked frozen admission cannot be replaced")
        self.frozen_admission = copy.deepcopy(admission)
        self.frozen_admission["encoding_identities_sha256"] = _sorted_canonical(
            "rz-pals-checked-native-encodings/1", self._encoding_identities)
        self._frozen_admission_identity = _sorted_canonical(FROZEN_ADMISSION_DOMAIN, self.frozen_admission)
        self._frozen_receipt_identity = _canonical("rz-pals-frozen-receipt-view/1", self.collection_receipt)
        self._verify_frozen_admission()

    def _verify_frozen_admission(self):
        if self._frozen_admission_identity is None:
            if self.frozen_admission is not None:
                raise ValueError("unchecked metadata cannot enroll a frozen dataset")
            return
        if (not isinstance(self.frozen_admission, dict)
                or _sorted_canonical(FROZEN_ADMISSION_DOMAIN, self.frozen_admission) != self._frozen_admission_identity):
            raise ValueError("checked frozen admission identity changed")
        if self.owned_sources != self.frozen_admission["owned_sources"]:
            raise ValueError("checked producer source authority changed")
        view = current_label_view(self.records)
        if (view != self._current_view or frozenset(view.current_indices) != self._current_indices
                or view.sha256 != self.frozen_admission["current_view_sha256"]):
            raise ValueError("checked frozen current label view changed")
        if _canonical("rz-pals-frozen-receipt-view/1", self.collection_receipt) != self._frozen_receipt_identity:
            raise ValueError("checked frozen receipt provenance changed")
        if (getattr(self, "native_continuation_observation", None)
                != self.frozen_admission.get("native_continuation_observation")):
            raise ValueError("checked continuation observation changed")
        if (getattr(self, "full_line_context_observation", None) != self.frozen_admission.get("full_line_context_observation")
                or getattr(self, "target_coverage_observation", None) != self.frozen_admission.get("target_coverage_observation")):
            raise ValueError("checked V2 full-line/coverage observation changed")
        if (_sorted_canonical("rz-pals-checked-native-encodings/1", self._encoding_identities)
                != self.frozen_admission["encoding_identities_sha256"]):
            raise ValueError("checked frozen encoding admission changed")
        if set(self.encodings) != set(self._encoding_identities) or any(
                not isinstance(value, EncodedSnapshot)
                or _encoding_identity(value) != self._encoding_identities.get(key)
                for key, value in self.encodings.items()):
            raise ValueError("checked frozen per-input encoding changed")

    def _attach_encodings(self, encodings):
        if self._frozen_admission_identity is not None:
            raise ValueError("checked frozen encodings cannot be replaced")
        self.encodings = copy.deepcopy(dict(encodings))
        self._encoding_identities = {}
        for key, value in self.encodings.items():
            if not isinstance(value, EncodedSnapshot):
                raise ValueError("collector encoded snapshot type required")
            _validate_encoding(value)
            if key != value.input_sha256:
                raise ValueError("encoded snapshot key mismatch")
            self._encoding_identities[key] = _encoding_identity(value)

    def indices(self, role, split="train"):
        if role not in ROLE_NAMES or split not in ("train", "validation", "holdout"):
            raise ValueError("invalid role/split selection")
        self._verify_raw_integrity()
        return [i for i in self._current_view.current_indices if self.records[i]["input"]["snapshot"]["role"] == role and self.split[self.records[i]["input"]["snapshot"]["game_id"]] == split]

    def collate(self, indices, role, *, split="train", task_contexts=None, device="cpu"):
        """V ranks use softmax(-rank) within one explicit context, temperature 1.

        Duplicate task kinds in a context are rejected. Unranked kinds are
        excluded from both the normalization and loss. No context is selected
        from sample position or a hash modulo seven.
        """
        if role not in ROLE_NAMES or split not in ("train", "validation", "holdout") or not 1 <= len(indices) <= 256:
            raise ValueError("batch extent/role/split")
        if role == "verifier" and (task_contexts is None or len(task_contexts) != len(indices)):
            raise ValueError("V requires one explicit branch/profile/budget context per row")
        if role != "verifier" and task_contexts is not None:
            raise ValueError("P/C must not consume V task controls")
        self._verify_raw_integrity()
        rows, encoded = [], []
        for index in indices:
            _uint(index, len(self.records) - 1)
            if index not in self._current_indices:
                raise ValueError("historical or superseded raw row cannot enter current training batches")
            row = self.records[index]
            if _canonical("rz-pals-python-immutable-row/1", row) != self._row_identities[index]:
                raise ValueError("immutable dataset input/target changed after admission")
            s = row["input"]["snapshot"]
            if s["role"] != role or self.split[s["game_id"]] != split:
                raise ValueError("role/split mismatch; holdout cannot enter train batches")
            e = self.encodings.get(row["input"]["sha256"])
            if not isinstance(e, EncodedSnapshot) or e.input_sha256 != row["input"]["sha256"] or e.encoding_sha256 != s["encoding_sha256"]:
                raise ValueError("missing or mismatched collector encoded snapshot")
            _validate_encoding(e)
            if _encoding_identity(e) != self._encoding_identities.get(e.input_sha256):
                raise ValueError("encoded input changed after admission")
            expected_records = [(v["observation_sha256"], v["situation_revision"]) for v in s["public_records"]]
            if [(v[0], v[1]) for v in e.public_records] != expected_records:
                raise ValueError("public features differ from captured observation order/revision")
            if len(e.divergences) > 128 or (role != "critic" and e.divergences):
                raise ValueError("divergence role/capacity")
            if len({v[0] for v in e.divergences}) != len(e.divergences) or len({v[0] for v in e.task_queries}) != len(e.task_queries):
                raise ValueError("ambiguous encoded query context")
            if role != "verifier" and e.task_queries:
                raise ValueError("V controls leaked into P/C encoded input")
            rows.append(row)
            encoded.append(e)
        if len({e.model_profile for e in encoded}) != 1:
            raise ValueError("training batch cannot mix legacy and V2 model/encoding profiles")
        config = ModelConfig.for_profile(encoded[0].model_profile)
        b = len(rows)
        r = max(1, max(len(e.public_records) for e in encoded))
        c = max(1, max(len(v["input"]["snapshot"]["legal_moves"]) for v in rows))
        d = max(1, max(len(e.divergences) for e in encoded))
        board = torch.zeros((b, 64), dtype=torch.int64, device=device)
        metadata = torch.zeros((b, 16), dtype=torch.float32, device=device)
        records = torch.zeros((b, r, 16), dtype=torch.float32, device=device)
        record_mask = torch.zeros((b, r), dtype=torch.bool, device=device)
        candidates = torch.zeros((b, c, 3), dtype=torch.int64, device=device)
        candidate_mask = torch.zeros((b, c), dtype=torch.bool, device=device)
        divergence_features = torch.zeros((b, d, 8), dtype=torch.float32, device=device)
        divergence_candidates = torch.zeros((b, d), dtype=torch.bool, device=device)
        query = torch.zeros((b, 16), dtype=torch.float32, device=device)
        policy, wdl, divergence, task = (torch.zeros(shape, dtype=torch.float32, device=device) for shape in ((b, c), (b, 3), (b, d), (b, len(TASKS))))
        policy_mask, wdl_mask = (torch.zeros(b, dtype=torch.bool, device=device) for _ in range(2))
        divergence_mask = torch.zeros((b, d), dtype=torch.bool, device=device)
        task_mask = torch.zeros((b, len(TASKS)), dtype=torch.bool, device=device)
        for i, (row, e) in enumerate(zip(rows, encoded)):
            board[i] = torch.tensor(e.board, dtype=torch.int64, device=device)
            metadata[i] = torch.tensor(e.metadata, dtype=torch.float32, device=device)
            for j, (_, _, features) in enumerate(e.public_records):
                records[i, j] = torch.tensor(features, dtype=torch.float32, device=device)
                record_mask[i, j] = True
            for j, move in enumerate(row["input"]["snapshot"]["legal_moves"]):
                candidates[i, j] = torch.tensor(move_components(move), device=device)
                candidate_mask[i, j] = True
            for j, (context, features) in enumerate(e.divergences):
                if not isinstance(context, DivergenceContext):
                    raise ValueError("divergence context must be explicit")
                _sha(context.challenged_line_sha256)
                _uint(context.divergence_ply, 2**32 - 1)
                if context.input_revision != row["input"]["snapshot"]["input_revision"]:
                    raise ValueError("divergence context revision mismatch")
                divergence_features[i, j] = torch.tensor(features, dtype=torch.float32, device=device)
                divergence_candidates[i, j] = True
            selected_context = task_contexts[i] if role == "verifier" else None
            if selected_context is not None:
                if not isinstance(selected_context, TaskContext):
                    raise ValueError("V context must use TaskContext")
                selected_context.validate()
                q = dict(e.task_queries).get(selected_context)
                if q is None:
                    raise ValueError("missing collector-encoded V context query")
            else:
                q = e.query
            query[i] = torch.tensor(q, dtype=torch.float32, device=device)
            label = row["future_label"]
            if label is None:
                continue
            if role != "verifier" and label["policy"] is not None:
                policy[i, :len(label["policy"]["probabilities"])] = torch.tensor(label["policy"]["probabilities"], device=device)
                policy_mask[i] = True
            if role != "verifier" and label["value_wdl"] is not None:
                wdl[i] = torch.tensor(label["value_wdl"], device=device)
                wdl_mask[i] = True
            counterexample = label["counterexample"]
            if counterexample is not None and counterexample["validity"] in ("supported_after_repair", "refuted_by_repair"):
                target_context = DivergenceContext(counterexample["challenged_line_sha256"], counterexample["divergence_ply"], counterexample["input_revision"])
                contexts = [v[0] for v in e.divergences]
                if target_context not in contexts:
                    raise ValueError("counterexample target has no matching captured divergence")
                j = contexts.index(target_context)
                divergence[i, j] = float(counterexample["validity"] == "supported_after_repair")
                divergence_mask[i, j] = True
            if role == "verifier":
                ranked = [v for v in (label["verifier_tasks"] or []) if TaskContext.from_target(v) == selected_context and v["preference_rank"] is not None]
                if ranked:
                    weights = torch.softmax(torch.tensor([-float(v["preference_rank"]) for v in ranked], dtype=torch.float32, device=device), dim=0)
                    for v, weight in zip(ranked, weights):
                        j = TASKS.index(v["task"])
                        task[i, j], task_mask[i, j] = weight, True
        inputs = TensorInput(board, metadata, records, record_mask, candidates, candidate_mask,
                             divergence_features, divergence_candidates, query)
        if config.full_line:
            record_lines = torch.zeros((b, r, MAX_LINE_PLIES, 3), dtype=torch.int64, device=device)
            record_line_mask = torch.zeros((b, r, MAX_LINE_PLIES), dtype=torch.bool, device=device)
            query_lines = torch.zeros((b, 3, MAX_LINE_PLIES, 3), dtype=torch.int64, device=device)
            query_line_mask = torch.zeros((b, 3, MAX_LINE_PLIES), dtype=torch.bool, device=device)
            relations = torch.full((b, r, 2), -1, dtype=torch.int64, device=device)
            for i, e in enumerate(encoded):
                for j, record in enumerate(e.full_line["records"]):
                    for ply, token in enumerate(record["moves"]):
                        record_lines[i, j, ply] = torch.tensor([token["from"], token["to"], token["promotion"]], device=device)
                        record_line_mask[i, j, ply] = True
                    for relation, name in enumerate(("parent", "supersedes")):
                        if record[name] is not None:
                            relations[i, j, relation] = record[name]
                for j, name in enumerate(("query_prefix", "query_proposal", "query_counter")):
                    for ply, token in enumerate(e.full_line[name]):
                        query_lines[i, j, ply] = torch.tensor([token["from"], token["to"], token["promotion"]], device=device)
                        query_line_mask[i, j, ply] = True
            inputs = TensorInput(**{**inputs.__dict__, "record_line_tokens": record_lines,
                                    "record_line_mask": record_line_mask, "query_line_tokens": query_lines,
                                    "query_line_mask": query_line_mask, "record_relations": relations})
        inputs.validate(config)
        return TrainingBatch(ROLE_NAMES[role], inputs, tuple(v["input"]["sha256"] for v in rows), policy,
                             policy_mask, wdl, wdl_mask, divergence, divergence_mask, task, task_mask, config.profile)


def _masked_distribution_loss(logits, target, active, rows):
    if logits.shape != target.shape or active.shape != target.shape or rows.shape != logits.shape[:1] or logits.dtype != torch.float32 or target.dtype != torch.float32 or active.dtype != torch.bool or rows.dtype != torch.bool:
        raise ValueError("loss shape/dtype mismatch")
    if not torch.all(torch.isfinite(logits)) or not torch.all(torch.isfinite(target)) or torch.any(target < 0) or torch.any(target[~active] != 0):
        raise ValueError("invalid loss logits/target")
    if torch.any(rows & (~active.any(dim=-1))) or torch.any(torch.abs(target[rows].sum(dim=-1) - 1) > 1e-6):
        raise ValueError("active target must have nonempty normalized support")
    if not rows.any():
        return logits.reshape(-1)[0] * 0.0
    selected = logits[rows].masked_fill(~active[rows], -torch.inf)
    log_probability = F.log_softmax(selected, dim=-1).masked_fill(~active[rows], 0.0)
    result = -(target[rows] * log_probability).sum(dim=-1).mean()
    if not torch.isfinite(result):
        raise ValueError("distribution loss exceeds finite FP32 range")
    return result


def masked_losses(outputs, batch, weights=None):
    """Mean per supervised row for distributions, per known divergence for BCE.

    SupportedAfterRepair labels supervise counterexample survival with 1;
    RefutedByRepair with 0. Unknown/disputed facts carry no binary target.
    An entirely masked loss is differentiable zero, with zero gradient.
    """
    weights = {"policy": 1.0, "wdl": 1.0, "divergence": 1.0, "task": 1.0} if weights is None else dict(weights)
    if set(weights) != {"policy", "wdl", "divergence", "task"} or not all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in weights.values()):
        raise ValueError("invalid registered loss weights")
    if batch.role not in ("proposer", "critic", "validator") or len(outputs) != (3 if batch.role == "proposer" else 4):
        raise ValueError("hard-routed output role mismatch")
    batch.inputs.validate(ModelConfig.for_profile(batch.model_profile))
    b, c, d = batch.inputs.board.shape[0], batch.inputs.candidates.shape[1], batch.inputs.divergence_features.shape[1]
    required_targets = ((batch.policy, (b, c), torch.float32), (batch.policy_mask, (b,), torch.bool),
                        (batch.wdl, (b, 3), torch.float32), (batch.wdl_mask, (b,), torch.bool),
                        (batch.divergence, (b, d), torch.float32), (batch.divergence_mask, (b, d), torch.bool),
                        (batch.task, (b, len(TASKS)), torch.float32), (batch.task_mask, (b, len(TASKS)), torch.bool))
    if any(not isinstance(v, torch.Tensor) or v.shape != shape or v.dtype != dtype for v, shape, dtype in required_targets):
        raise ValueError("registered target head shape/dtype mismatch")
    expected_shapes = ((b, c), (b, 3), (b, 16, 384))
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.all(torch.isfinite(value)) for value in outputs) or any(value.shape != shape for value, shape in zip(outputs[:3], expected_shapes)):
        raise ValueError("role output must be finite even on masked rows")
    all_values = (*outputs, *(value for value in batch.inputs.__dict__.values() if value is not None), batch.policy, batch.policy_mask, batch.wdl, batch.wdl_mask,
                  batch.divergence, batch.divergence_mask, batch.task, batch.task_mask)
    if len({v.device for v in all_values}) != 1:
        raise ValueError("loss tensors must share one explicit device")
    policy, wdl = outputs[:2]
    zero = policy.reshape(-1)[0] * 0.0 + wdl.reshape(-1)[0] * 0.0
    losses = dict.fromkeys(weights, zero)
    if batch.role != "validator":
        losses["policy"] = _masked_distribution_loss(policy, batch.policy, batch.inputs.candidate_mask, batch.policy_mask)
        losses["wdl"] = _masked_distribution_loss(wdl, batch.wdl, torch.ones_like(batch.wdl, dtype=torch.bool), batch.wdl_mask)
    if batch.role == "critic":
        logits = outputs[3]
        if logits.shape != batch.divergence.shape or logits.dtype != torch.float32 or batch.divergence.dtype != torch.float32 or batch.divergence_mask.dtype != torch.bool or batch.divergence_mask.shape != logits.shape or not torch.all(torch.isfinite(batch.divergence)) or torch.any((batch.divergence < 0) | (batch.divergence > 1)) or torch.any(batch.divergence_mask & ~batch.inputs.divergence_mask):
            raise ValueError("invalid divergence head/mask")
        if batch.divergence_mask.any():
            losses["divergence"] = F.binary_cross_entropy_with_logits(logits[batch.divergence_mask], batch.divergence[batch.divergence_mask])
        else:
            losses["divergence"] = logits.reshape(-1)[0] * 0.0
    if batch.role == "validator":
        losses["task"] = _masked_distribution_loss(outputs[3], batch.task, batch.task_mask, batch.task_mask.any(dim=-1))
    losses["total"] = sum(weights[name] * losses[name] for name in weights)
    if not all(torch.isfinite(value) for value in losses.values()):
        raise ValueError("combined loss exceeds finite FP32 range")
    return losses


USAGE_NAMES = ("forward_flops", "backward_flops", "validation_flops", "recompute_flops", "steps", "games",
               "wall_time_ms", "cpu_nodes", "cpu_time_ms", "compute_units_milli", "spend_usd_micros", "output_bytes")
BUDGET_NAMES = ("max_flops", "max_steps", "max_games", "max_wall_time_ms", "max_cpu_nodes", "max_cpu_time_ms",
                "max_compute_units_milli", "max_spend_usd_micros", "max_output_bytes")


def validate_recipe(recipe):
    names = ("optimizer", "learning_rate", "betas", "epsilon", "weight_decay", "gradient_accumulation_steps",
             "phase", "role_order", "verifier_freeze", "model", "budget", "implementation_sha256", "dataset_sha256", "split_sha256")
    _fields(recipe, names, "recipe /2")
    roles = {"pc_bootstrap": ["proposer", "critic"], "pcv_preparation": ["proposer", "critic", "verifier"]}
    expected_freeze = None if recipe["phase"] == "pc_bootstrap" else {"encoder": True, "public_reader": True, "move_embedding": True}
    if recipe["phase"] not in roles or recipe["role_order"] != roles[recipe["phase"]] or recipe["verifier_freeze"] != expected_freeze or recipe["optimizer"] != "adamw":
        raise ValueError("unsupported phase/role/freeze/optimizer recipe")
    if expected_freeze is not None and any(type(v) is not bool for v in recipe["verifier_freeze"].values()):
        raise ValueError("observed verifier freeze flags must be booleans")
    for name in ("implementation_sha256", "dataset_sha256", "split_sha256"):
        _sha(recipe[name])
    for name in ("learning_rate", "epsilon", "weight_decay"):
        value = recipe[name]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
            raise ValueError("invalid AdamW scalar")
    if recipe["learning_rate"] > 1 or not isinstance(recipe["betas"], list) or len(recipe["betas"]) != 2 or not all(type(b) in (float, int) and math.isfinite(b) and 0 <= b < 1 for b in recipe["betas"]) or _uint(recipe["gradient_accumulation_steps"], 2**32 - 1) == 0:
        raise ValueError("invalid AdamW betas/accumulation")
    profile_names = ("width", "latent_slots", "recurrent_blocks", "recurrent_iterations", "query_heads", "kv_heads",
                     "head_dimension", "ffn_width", "fma_flops", "counted_operator_sha256", "uncounted_operators",
                     "forward_flops_per_sample", "backward_flops_per_sample")
    profile = _fields(recipe["model"], profile_names, "compute profile")
    for name, expected in zip(profile_names[:9], (384, 16, 2, 2, 6, 2, 64, 1024, 2)):
        if type(profile[name]) is not int or profile[name] != expected:
            raise ValueError("registered model configuration cannot be resized")
    _sha(profile["counted_operator_sha256"])
    for name in ("forward_flops_per_sample", "backward_flops_per_sample"):
        if _uint(profile[name]) == 0:
            raise ValueError("forward/backward cost must be explicitly registered")
    if not isinstance(profile["uncounted_operators"], list) or len(profile["uncounted_operators"]) > 64:
        raise ValueError("uncounted operator declaration extent")
    for operator in profile["uncounted_operators"]:
        _identity(operator)
    budget = _fields(recipe["budget"], BUDGET_NAMES, "finite budget")
    for value in budget.values():
        if _uint(value) == 0:
            raise ValueError("every compute/time/output/cost limit must be finite and positive")


def recipe_sha256(recipe):
    validate_recipe(recipe)
    # The exact serialized recipe is carried in checkpoints, preventing any
    # dependence on Rust/Python floating-point JSON lexical equivalence.
    return _canonical("rz-pals-python-preparation-recipe/1", recipe)


@dataclass
class OptimizerPreparation:
    optimizer: torch.optim.Optimizer
    phase: str
    role: str
    parameter_names: tuple
    observed_freeze: dict


def prepare_adamw(model, recipe, role):
    """Configure trainability and AdamW state; never invoke an update.

    Partitioning is exhaustive. Shared tensors registered through aliases are
    grouped once by object identity; cross-owner aliases are rejected.
    """
    validate_recipe(recipe)
    model.config.validate()
    if role not in recipe["role_order"] or ROLE_NAMES[role] not in model.experts:
        raise ValueError("role absent from preparation phase/model")
    return _owned_adamw(model, role, recipe["phase"], recipe)


def prepare_nonzero_adamw(model, role):
    """The locked follow-up optimizer; creating it does not perform an update.

    Every selected tensor has weight decay 0.01 in this explicit smoke domain.
    The preparation recipe's historical matrix/vector decay split is unchanged.
    """
    return _owned_adamw(model, role, "bounded_nonzero_smoke", {
        "learning_rate": 1e-4, "betas": [0.9, 0.999], "epsilon": 1e-8,
        "weight_decay": 0.01}, decay_vectors=True)


def _owned_adamw(model, role, phase, settings, *, decay_vectors=False):
    model.config.validate()
    if role not in ROLE_NAMES or ROLE_NAMES[role] not in model.experts:
        raise ValueError("role absent from optimizer model")
    prefixes = ("public_encoder.", "reader_blocks.", "move_embedding.")
    mapped = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter.dtype != torch.float32 or not torch.all(torch.isfinite(parameter)):
            raise ValueError("first-profile preparation requires finite FP32 parameters")
        owner = "shared" if name.startswith(prefixes) else next((r for r in ("proposer", "critic", "validator") if name.startswith(f"experts.{r}.")), None)
        if owner is None:
            raise ValueError(f"unmapped parameter responsibility: {name}")
        existing = mapped.get(id(parameter))
        if existing is not None and existing[2] != owner:
            raise ValueError("parameter alias crosses shared/private ownership")
        if existing is None:
            mapped[id(parameter)] = (name, parameter, owner)
    if any(not any(name.startswith(prefix) for name, _, _ in mapped.values()) for prefix in prefixes):
        raise ValueError("shared parameter groups must be nonempty and observed")
    selected = []
    active_role = ROLE_NAMES[role]
    for name, parameter, owner in mapped.values():
        active = owner == active_role or (role != "verifier" and owner == "shared")
        parameter.requires_grad_(active)
        parameter.grad = None
        if active:
            selected.append((name, parameter))
    if not selected or not any(owner == active_role for _, _, owner in mapped.values()):
        raise ValueError("missing selected private expert parameters")
    observed = {label: all(not parameter.requires_grad for name, parameter, _ in mapped.values() if name.startswith(prefix))
                for label, prefix in zip(("encoder", "public_reader", "move_embedding"), prefixes)}
    if role == "verifier" and (not all(observed.values()) or any(owner != "validator" for name, parameter, owner in mapped.values() if parameter.requires_grad)):
        raise ValueError("V may update private V tensors only")
    decay = [p for name, p in selected if p.ndim >= 2]
    no_decay = [p for name, p in selected if p.ndim < 2]
    groups = [{"params": values, "weight_decay": settings["weight_decay"] if label == "matrix" or decay_vectors else 0.0, "group_name": label}
              for label, values in (("matrix", decay), ("vector", no_decay)) if values]
    optimizer = torch.optim.AdamW(groups, lr=settings["learning_rate"], betas=tuple(settings["betas"]), eps=settings["epsilon"], foreach=False)
    if len({id(p) for g in optimizer.param_groups for p in g["params"]}) != len(selected):
        raise ValueError("optimizer parameter membership duplication")
    return OptimizerPreparation(optimizer, phase, role, tuple(name for name, _ in selected), observed)


class BudgetLedger:
    """Reserve every bounded operation before dispatch; cancellation is terminal.

    FLOPs are registered operator costs, with exclusions preserved in recipe.
    This does not turn a matrix-only estimate into measured total training FLOPs.
    Wall time includes preparation since this ledger was created/resumed.
    """
    def __init__(self, recipe, usage=None, *, clock=time.monotonic):
        validate_recipe(recipe)
        self.recipe = copy.deepcopy(recipe)
        self.usage = dict.fromkeys(USAGE_NAMES, 0) if usage is None else dict(_fields(usage, USAGE_NAMES, "usage"))
        self.clock, self.started = clock, clock()
        self.initial_wall_ms = self.usage["wall_time_ms"]
        self.canceled = False
        self._check(self.usage)

    def _check(self, usage):
        for value in usage.values():
            _uint(value)
        total = sum(usage[n] for n in USAGE_NAMES[:4])
        _uint(total)
        observed = {"max_flops": total, **{"max_" + n: usage[n] for n in USAGE_NAMES[4:]}}
        if any(observed[name] > maximum for name, maximum in self.recipe["budget"].items()):
            raise ValueError("registered preparation budget exhausted")

    def snapshot_usage(self):
        elapsed = self.clock() - self.started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("monotonic clock moved backwards/nonfinite")
        self.usage["wall_time_ms"] = max(self.usage["wall_time_ms"], self.initial_wall_ms + math.ceil(elapsed * 1000))
        self._check(self.usage)
        return dict(self.usage)

    def cancel(self):
        self.canceled = True

    def admit(self, **increments):
        if self.canceled:
            raise ValueError("preparation canceled; no further operation admitted")
        if set(increments) - set(USAGE_NAMES):
            raise ValueError("unknown usage counter")
        if increments.get("steps", 0):
            raise ValueError("actual optimizer updates are excluded from preparation")
        projected = self.snapshot_usage()
        for name, value in increments.items():
            projected[name] += _uint(value)
        self._check(projected)
        self.usage = projected
        return dict(projected)

    def admit_batch(self, samples, *, backward=False, validation=False, recompute=False):
        if _uint(samples, 256) == 0 or sum((backward, validation, recompute)) > 1:
            raise ValueError("invalid bounded operation kind")
        profile = self.recipe["model"]
        forward = samples * profile["forward_flops_per_sample"]
        increments = {"validation_flops" if validation else "recompute_flops" if recompute else "forward_flops": forward}
        if backward:
            increments["backward_flops"] = samples * profile["backward_flops_per_sample"]
        return self.admit(**increments)


@dataclass
class ResumableSampler:
    """Persist the actual permutation, cursor, epoch and role batch position."""
    indices: tuple
    seed: int
    epoch: int = 0
    cursor: int = 0
    permutation: list = field(default_factory=list)
    role: str = "proposer"

    def __post_init__(self):
        _uint(self.seed, 2**63 - 1)
        _uint(self.epoch)
        if self.role not in ROLE_NAMES or not self.indices or len(self.indices) > 65536 or len(set(self.indices)) != len(self.indices):
            raise ValueError("sampler role/indices")
        for index in self.indices:
            _uint(index)
        if not self.permutation:
            self.permutation = list(self.indices)
            random.Random(_canonical("rz-pals-sampler-seed/1", [self.seed, self.epoch, self.role])).shuffle(self.permutation)
        for index in self.permutation:
            _uint(index)
        if sorted(self.permutation) != sorted(self.indices) or _uint(self.cursor, len(self.permutation)) > len(self.permutation):
            raise ValueError("sampler permutation/cursor")

    def next_batch(self, batch_size):
        if _uint(batch_size, 256) == 0:
            raise ValueError("bounded batch size required")
        result = self.permutation[self.cursor:self.cursor + batch_size]
        self.cursor += len(result)
        return result

    def state(self):
        value = asdict(self)
        value["indices"] = list(self.indices)
        value["algorithm"] = "python-mt19937-seeded-permutation-v1"
        value["order_sha256"] = _canonical("rz-pals-sampler-order/1", self.permutation)
        return value

    @classmethod
    def restore(cls, state, *, indices, role):
        value = _fields(state, ("indices", "seed", "epoch", "cursor", "permutation", "role", "algorithm", "order_sha256"), "sampler checkpoint")
        if not isinstance(value["permutation"], list) or len(value["permutation"]) != len(indices) or not isinstance(value["indices"], list) or any(type(index) is not int for index in value["permutation"] + value["indices"]) or value["algorithm"] != "python-mt19937-seeded-permutation-v1" or value["order_sha256"] != _canonical("rz-pals-sampler-order/1", value["permutation"]) or value["indices"] != list(indices) or value["role"] != role:
            raise ValueError("sampler identity/order/role mismatch")
        return cls(tuple(indices), value["seed"], value["epoch"], value["cursor"], list(value["permutation"]), role)


def capture_rng(cuda_devices=()):
    devices = tuple(cuda_devices)
    if len(set(devices)) != len(devices) or any(type(d) is not int or d < 0 for d in devices):
        raise ValueError("explicit device list required")
    if devices and (not torch.cuda.is_available() or any(d >= torch.cuda.device_count() for d in devices)):
        raise ValueError("requested CUDA RNG device unavailable; no CPU fallback")
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], torch.tensor(state[1].astype(np.int64)), state[2], state[3], state[4]),
            "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": {str(d): torch.cuda.get_rng_state(d).clone() for d in devices},
            "versions": {"python": list(sys.version_info[:3]), "numpy": np.__version__, "torch": str(torch.__version__)}}


def _validate_rng(state, cuda_devices=()):
    cuda_devices = tuple(cuda_devices)
    _fields(state, ("python", "numpy", "torch_cpu", "torch_cuda", "versions"), "full RNG")
    current = capture_rng(cuda_devices)
    if state["versions"] != current["versions"] or set(state["torch_cuda"]) != set(current["torch_cuda"]):
        raise ValueError("RNG software/device identity mismatch")
    # Validate on independent generators before changing any global state.
    probe_python = random.Random()
    probe_python.setstate(state["python"])
    ns = state["numpy"]
    if len(ns) != 5 or ns[0] != "MT19937" or not isinstance(ns[1], torch.Tensor) or ns[1].dtype != torch.int64 or ns[1].shape != (624,) or torch.any((ns[1] < 0) | (ns[1] > 2**32 - 1)):
        raise ValueError("incomplete NumPy RNG state")
    numpy_state = (ns[0], ns[1].cpu().numpy().astype(np.uint32), ns[2], ns[3], ns[4])
    np.random.RandomState().set_state(numpy_state)
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    for device in cuda_devices:
        torch.Generator(device=f"cuda:{device}").set_state(state["torch_cuda"][str(device)])
    return numpy_state


def restore_rng(state, cuda_devices=()):
    cuda_devices = tuple(cuda_devices)
    numpy_state = _validate_rng(state, cuda_devices)
    random.setstate(state["python"])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch_cpu"])
    for device in cuda_devices:
        torch.cuda.set_rng_state(state["torch_cuda"][str(device)], device)


def _clone_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone_state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_state(v) for v in value)
    return copy.deepcopy(value)


def save_preparation_checkpoint(path, model, preparation, recipe, sampler, ledger, *, cuda_devices=()):
    """Write model/AdamW/full RNG/sampler/usage atomically outside source.

    Gradient accumulation is not silently discarded: checkpoints are accepted
    only at a boundary with no pending gradients. Canceled work can still retain
    its checkpoint, subject to the same finite output/time budget.
    """
    validate_recipe(recipe)
    cuda_devices = tuple(cuda_devices)
    if preparation.phase != recipe["phase"] or preparation.role != sampler.role or ledger.recipe != recipe or any(p.grad is not None for p in model.parameters()):
        raise ValueError("checkpoint requires matching phase/role and no pending gradients")
    usage = ledger.snapshot_usage()
    if usage["steps"] != 0:
        raise ValueError("this preparation checkpoint cannot claim optimizer updates")
    active = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    members = [p for group in preparation.optimizer.param_groups for p in group["params"]]
    if active != set(preparation.parameter_names) or len({id(p) for p in members}) != len(members) or {id(p) for p in members} != {id(p) for name, p in model.named_parameters() if name in active} or preparation.optimizer.state:
        raise ValueError("checkpoint optimizer responsibility/state is inconsistent with zero-step preparation")
    expected = prepare_adamw(copy.deepcopy(model), recipe, preparation.role)
    if preparation.optimizer.state_dict() != expected.optimizer.state_dict():
        raise ValueError("checkpoint optimizer settings differ from admitted role/recipe")
    model.config.validate()
    tensor_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    if tensor_bytes > recipe["budget"]["max_output_bytes"] - usage["output_bytes"]:
        raise ValueError("model checkpoint tensor bytes exceed remaining output admission")
    payload = {"schema": CHECKPOINT_SCHEMA, "config": model.config.to_dict(), "recipe": copy.deepcopy(recipe),
               "recipe_sha256": recipe_sha256(recipe), "phase": preparation.phase, "role": preparation.role,
               "parameter_names": list(preparation.parameter_names), "model": _clone_state(model.state_dict()),
               "optimizer": _clone_state(preparation.optimizer.state_dict()), "rng": capture_rng(cuda_devices),
               "sampler": sampler.state(), "usage": usage, "canceled": ledger.canceled,
               "training_executed": False, "training_steps": 0}
    # Including this file's own size in persisted usage requires a small fixed
    # point: serialization size can change when an integer grows a byte.
    size = 0
    for _ in range(8):
        payload["usage"] = ledger.snapshot_usage()
        payload["usage"]["output_bytes"] += size
        ledger._check(payload["usage"])
        stream = io.BytesIO()
        torch.save(payload, stream)
        data = stream.getvalue()
        if len(data) == size:
            break
        size = len(data)
    else:
        raise ValueError("checkpoint serialization size did not stabilize")
    path = Path(path).expanduser().resolve()
    if path.suffix != ".pt":
        raise ValueError("checkpoint must use .pt")
    output_directory(path.parent)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError("checkpoint or incomplete checkpoint already exists")
    # Recheck wall deadline immediately before file admission.
    ledger.snapshot_usage()
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    ledger.usage["output_bytes"] += len(data)
    digest = digest_file(path)
    final_usage = ledger.snapshot_usage()  # Preserve artifact but report late save.
    # The returned receipt closes the file I/O/hash cost after the immutable
    # snapshot. Persist it with the run manifest and require it for full resume.
    return {"path": str(path), "sha256": digest, "bytes": len(data), "training_executed": False,
            "recipe_sha256": recipe_sha256(recipe), "phase": preparation.phase, "role": preparation.role,
            "usage": final_usage, "canceled": ledger.canceled}


def load_preparation_checkpoint(path, expected_sha256, model, recipe, *, usage_receipt, indices, role, cuda_devices=()):
    """Verify identity, restore all state, and return the exact next sampler batch.

    The caller supplies the trusted checkpoint digest and final usage receipt,
    preserving serialization/write/hash cost beyond the snapshot boundary.
    Pickle execution is
    prohibited by weights_only=True; CUDA use is explicit and validated.
    """
    validate_recipe(recipe)
    cuda_devices = tuple(cuda_devices)
    _sha(expected_sha256)
    path = Path(path).expanduser().resolve()
    if path.stat().st_size > recipe["budget"]["max_output_bytes"] or digest_file(path) != expected_sha256:
        raise ValueError("checkpoint bytes/digest differ from admitted artifact")
    # torch.save uses an uncompressed ZIP. Bound both payload entries and total
    # expanded bytes before the weights-only loader allocates tensor storage.
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not entries or len(entries) > 65536 or len({entry.filename for entry in entries}) != len(entries) or any(entry.compress_type != zipfile.ZIP_STORED for entry in entries) or sum(entry.file_size for entry in entries) > recipe["budget"]["max_output_bytes"]:
            raise ValueError("checkpoint expanded allocation or archive identity exceeds admission")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    names = ("schema", "config", "recipe", "recipe_sha256", "phase", "role", "parameter_names", "model", "optimizer", "rng", "sampler", "usage", "canceled", "training_executed", "training_steps")
    _fields(payload, names, "complete checkpoint")
    if payload["schema"] != CHECKPOINT_SCHEMA or payload["recipe"] != recipe or payload["recipe_sha256"] != recipe_sha256(recipe) or payload["phase"] != recipe["phase"] or payload["role"] != role or payload["config"] != model.config.to_dict() or payload["training_executed"] is not False or payload["training_steps"] != 0 or type(payload["canceled"]) is not bool:
        raise ValueError("checkpoint model/recipe/phase/role/provenance mismatch")
    receipt_names = ("path", "sha256", "bytes", "training_executed", "recipe_sha256", "phase", "role", "usage", "canceled")
    _fields(usage_receipt, receipt_names, "final checkpoint usage receipt")
    if usage_receipt["sha256"] != expected_sha256 or usage_receipt["bytes"] != path.stat().st_size or usage_receipt["training_executed"] is not False or usage_receipt["recipe_sha256"] != payload["recipe_sha256"] or usage_receipt["phase"] != payload["phase"] or usage_receipt["role"] != role or usage_receipt["canceled"] is not payload["canceled"]:
        raise ValueError("checkpoint final usage receipt identity mismatch")
    persisted = _fields(payload["usage"], USAGE_NAMES, "snapshot usage")
    finalized = _fields(usage_receipt["usage"], USAGE_NAMES, "final usage")
    if any(finalized[name] != persisted[name] for name in USAGE_NAMES if name != "wall_time_ms") or _uint(finalized["wall_time_ms"]) < _uint(persisted["wall_time_ms"]):
        raise ValueError("checkpoint final usage changed non-I/O counters or lost elapsed time")
    sampler = ResumableSampler.restore(payload["sampler"], indices=indices, role=role)
    ledger = BudgetLedger(recipe, finalized)
    if ledger.usage["steps"] != 0:
        raise ValueError("preparation resume cannot imply training steps")
    current = model.state_dict()
    if set(payload["model"]) != set(current) or any(not isinstance(v, torch.Tensor) or v.shape != current[k].shape or v.dtype != current[k].dtype or not torch.all(torch.isfinite(v)) for k, v in payload["model"].items()):
        raise ValueError("checkpoint parameter shape/dtype/value mismatch")
    # Every rejection before the commit boundary leaves caller trainability,
    # gradients, parameters, optimizer and global RNG untouched.
    scratch = copy.deepcopy(model)
    checked = prepare_adamw(scratch, recipe, role)
    if list(checked.parameter_names) != payload["parameter_names"]:
        raise ValueError("optimizer parameter responsibility changed")
    if payload["optimizer"] != checked.optimizer.state_dict():
        raise ValueError("zero-step optimizer state/settings/membership differ from recipe")
    checked.optimizer.load_state_dict(payload["optimizer"])
    _validate_rng(payload["rng"], cuda_devices)
    preparation = prepare_adamw(model, recipe, role)
    preparation.optimizer.load_state_dict(payload["optimizer"])
    model.load_state_dict(payload["model"], strict=True)
    restore_rng(payload["rng"], cuda_devices)
    ledger.canceled = payload["canceled"]
    return preparation, sampler, ledger
