"""PALS training *preparation*: audited tensors, losses, optimizer and resume.

There is deliberately no optimizer-update or training-loop entry point. Rules
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
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from .artifacts import digest_file, output_directory
from .config import ModelConfig, TASKS
from .model import TensorInput

DATA_DOMAIN = "rz-pals-data/2"
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
    if source not in sources or (source["kind"] == "own_cpu" and source["cpu_binary_sha256"] not in owned_cpu):
        raise ValueError("unregistered own input source")
    if _sha(frozen["sha256"]) != seal_snapshot(s):
        raise ValueError("historical /2 input seal mismatch")
    private = record["verifier_private"]
    if private is not None:
        private = _fields(private, ("task_kind", "control_sha256", "private_latent"), "verifier private")
        _identity(private["task_kind"])
        _sha(private["control_sha256"])
        if s["role"] != "verifier" or not isinstance(private["private_latent"], list) or len(private["private_latent"]) > 16 * 384 or not all(type(v) in (int, float) and math.isfinite(v) for v in private["private_latent"]):
            raise ValueError("V private data is not permitted in P/C inputs")
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
        if p["engine_sha256"] not in owned_cpu or _uint(p["completed_depth"], 2**32 - 1) < 1:
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


def _validate_encoding(e):
    _sha(e.input_sha256)
    _sha(e.encoding_sha256)
    _vector(e.board, 64, integers=True)
    _vector(e.metadata, 16)
    _vector(e.query, 16)
    if not isinstance(e.public_records, (list, tuple)) or len(e.public_records) > 128 or not isinstance(e.divergences, (list, tuple)) or len(e.divergences) > 128 or not isinstance(e.task_queries, (list, tuple)) or len(e.task_queries) > 64:
        raise ValueError("encoded item allocation bound")
    for entry in e.public_records:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError("encoded public record descriptor")
        _sha(entry[0])
        _uint(entry[1])
        _vector(entry[2], 16)
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
    names = ("version", "input_sha256", "encoding_sha256", "encoder_source_sha256", "model_epoch_kind",
             "canonical_tensor_sha256", "tensor_json", "tensor_sha256", "record_sources", "sha256")
    value = _fields(sidecar, names, "native tensor sidecar")
    s = record["input"]["snapshot"]
    _sha(expected_encoder_source_sha256)
    if value["version"] != "rz-pals-native-input-sidecar/1" or value["input_sha256"] != record["input"]["sha256"] or value["encoding_sha256"] != s["encoding_sha256"] or value["encoder_source_sha256"] != expected_encoder_source_sha256:
        raise ValueError("native sidecar input/encoder identity mismatch")
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
    result = EncodedSnapshot(value["input_sha256"], value["encoding_sha256"], tuple(tensor["board"]), tuple(tensor["metadata"]), tuple(public), tuple(tensor["query"]))
    _validate_encoding(result)
    return result


def load_collected_dataset(directory, *, expected_receipt_sha256, expected_encoder_source_sha256,
                           max_input_bytes=64 * 1024 * 1024):
    """Read only fixed collector filenames after independent receipt admission.

    Every consumed JSONL file is byte/digest checked before parsing. A failed or
    incomplete collection never becomes an admitted training dataset. This only
    loads data; it neither collects games nor updates an optimizer.
    """
    _sha(expected_receipt_sha256)
    _sha(expected_encoder_source_sha256)
    if _uint(max_input_bytes, 1024 * 1024 * 1024) == 0:
        raise ValueError("finite collection input byte limit required")
    directory = Path(directory).expanduser().resolve()
    consumed = 0
    def read(name, asset=None):
        nonlocal consumed
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("collection file must be a regular file")
        size = path.stat().st_size
        if consumed + size > max_input_bytes:
            raise ValueError("collection input allocation budget exceeded")
        data = path.read_bytes()
        consumed += len(data)
        if len(data) != size or consumed > max_input_bytes:
            raise ValueError("collection file changed or exceeded allocation budget")
        if asset is not None:
            asset = _fields(asset, ("sha256", "bytes"), "collection artifact")
            if _uint(asset["bytes"]) != len(data) or _sha(asset["sha256"]) != hashlib.sha256(data).hexdigest():
                raise ValueError("collection artifact byte identity mismatch")
        return data
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
        self._row_identities = tuple(_canonical("rz-pals-python-immutable-row/1", row) for row in self.records)
        self._split_identity = _canonical("rz-pals-python-immutable-split/1", self.split)
        self._attach_encodings(encodings)

    def _attach_encodings(self, encodings):
        self.encodings = copy.deepcopy(dict(encodings))
        self._encoding_identities = {}
        for key, value in self.encodings.items():
            if not isinstance(value, EncodedSnapshot):
                raise ValueError("collector encoded snapshot type required")
            _validate_encoding(value)
            if key != value.input_sha256:
                raise ValueError("encoded snapshot key mismatch")
            self._encoding_identities[key] = _canonical("rz-pals-python-immutable-encoding/1", asdict(value))

    def indices(self, role, split="train"):
        if role not in ROLE_NAMES or split not in ("train", "validation", "holdout"):
            raise ValueError("invalid role/split selection")
        return [i for i, r in enumerate(self.records) if r["input"]["snapshot"]["role"] == role and self.split[r["input"]["snapshot"]["game_id"]] == split]

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
        if _canonical("rz-pals-python-immutable-split/1", self.split) != self._split_identity:
            raise ValueError("immutable dataset split changed after admission")
        rows, encoded = [], []
        for index in indices:
            _uint(index, len(self.records) - 1)
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
            if _canonical("rz-pals-python-immutable-encoding/1", asdict(e)) != self._encoding_identities.get(e.input_sha256):
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
        inputs.validate()
        return TrainingBatch(ROLE_NAMES[role], inputs, tuple(v["input"]["sha256"] for v in rows), policy,
                             policy_mask, wdl, wdl_mask, divergence, divergence_mask, task, task_mask)


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
    batch.inputs.validate()
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
    all_values = (*outputs, *batch.inputs.__dict__.values(), batch.policy, batch.policy_mask, batch.wdl, batch.wdl_mask,
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
    groups = [{"params": values, "weight_decay": recipe["weight_decay"] if label == "matrix" else 0.0, "group_name": label}
              for label, values in (("matrix", decay), ("vector", no_decay)) if values]
    optimizer = torch.optim.AdamW(groups, lr=recipe["learning_rate"], betas=tuple(recipe["betas"]), eps=recipe["epsilon"], foreach=False)
    if len({id(p) for g in optimizer.param_groups for p in g["params"]}) != len(selected):
        raise ValueError("optimizer parameter membership duplication")
    return OptimizerPreparation(optimizer, recipe["phase"], role, tuple(name for name, _ in selected), observed)


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
