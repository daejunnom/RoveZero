"""Bounded independent CPU replay of one exact native CUDA Warm capture.

Admission and --help use only the standard library. Execution is opt-in and
requires separately scheduled CPU NN resources. Supplied Rules features are
replayed without reencoding or inventing legal moves. The native owner remains
responsible for seed eligibility, execution fences and final GPU acceptance.
"""
import argparse
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import struct
import time

from .config import MAX_LINE_PLIES, ModelConfig, public_tensor_names
from .warm_artifacts import (_read_immutable_bytes, _document_from_immutable_bytes,
                             _cpu_session, _descriptors, _feed, _reference,
                             MAX_WARM_GRAPH_BYTES, MAX_WARM_MANIFEST_BYTES,
                             validate_warm_export_manifest, warm_export_domain)

CAPTURE_SCHEMA = "rz-pals-native-cuda-warm-check/1"
REFERENCE_SCHEMA = "rz-pals-cuda-warm-actual-reference/1"
REFERENCE_EXECUTION = "onnxruntime_and_torch_same_actual_inputs_and_seed"
MAX_CAPTURE_BYTES = 128 * 1024 * 1024
MAX_CAPTURE_INPUT_BYTES = 8 * 1024 * 1024
MAX_REFERENCE_BYTES = 128 * 1024 * 1024
MIN_ROLE_FORWARDS = 16
MAX_ROLE_FORWARDS = 128
DEFAULT_ROLE_FORWARDS = 64  # Scenario default; admission requires the actual explicit captured value.
STARTUP_ROLE_FORWARDS = 2
MAX_CAPTURE_CALLS = MAX_ROLE_FORWARDS - STARTUP_ROLE_FORWARDS
LATENT_ELEMENTS = 16 * 384
MAX_REFERENCE_SECONDS = 900
CLEANUP_SECONDS = 30
ATOL, RTOL = 1e-4, 1e-3
PROBABILITY_ATOL = 1e-4
_WORKER_INTEROP = False
_RAW_FIELDS = {"candidate_logits", "wdl_logits", "divergence_logits", "task_logits", "private_latent"}
_INPUT_FIELDS = {"role", "board", "metadata", "records", "required_critical_records", "candidates",
                 "divergence_features", "query", "situation_revision", "history_digest", "model_epoch", "full_line"}


def _fields(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("missing or unknown capture fields")


def _uint(value, bits=64):
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise ValueError("capture integer is out of range or has the wrong type")
    return value


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("capture digest must be lowercase hexadecimal SHA256")
    return value


def _array(value, maximum, exact=None):
    if type(value) is not list or len(value) > maximum or exact is not None and len(value) != exact:
        raise ValueError("capture array length exceeds the registered shape")
    return value


def _f32_bytes(value):
    if type(value) not in (int, float):
        raise ValueError("capture requires finite FP32 numeric values")
    try:
        result = struct.pack("<f", value)
    except (OverflowError, struct.error):
        raise ValueError("capture numeric value exceeds finite FP32") from None
    if not math.isfinite(struct.unpack("<f", result)[0]):
        raise ValueError("capture requires finite FP32 numeric values")
    return result


def _features(value, count):
    for number in _array(value, count, count):
        _f32_bytes(number)


def _identity(value):
    _fields(value, ("epoch", "sequence"))
    return _uint(value["epoch"]), _uint(value["sequence"])


def _digest_bytes(value):
    return bytes(_uint(part, 8) for part in _array(value, 32, 32))


def _move(value):
    _fields(value, ("from", "to", "promotion"))
    result = (_uint(value["from"], 8), _uint(value["to"], 8), _uint(value["promotion"], 8))
    if result[0] > 63 or result[1] > 63 or result[0] == result[1] or result[2] > 4:
        raise ValueError("invalid supplied Move16 token")
    return result


def _line(value):
    return [_move(movement) for movement in _array(value, MAX_LINE_PLIES)]


def validate_captured_input(value, config):
    """Validate typed evidence, without claiming independent Rules legality."""
    config.validate()
    if not config.full_line:
        raise ValueError("actual CUDA Warm reference requires a full-line V2 profile")
    _fields(value, _INPUT_FIELDS)
    if value["role"] not in ("proposer", "critic"):
        raise ValueError("actual CUDA Warm reference is P/C only")
    if any(_uint(piece, 8) > 12 for piece in _array(value["board"], 64, 64)):
        raise ValueError("captured board piece code")
    _features(value["metadata"], 16)
    _features(value["query"], 16)
    _uint(value["situation_revision"])
    _digest_bytes(value["history_digest"])
    _digest_bytes(value["model_epoch"])
    records = _array(value["records"], config.max_records)
    record_ids, critical_ids = set(), set()
    for record in records:
        _fields(record, ("record_id", "revision", "critical", "features"))
        identifier = _uint(record["record_id"])
        _uint(record["revision"])
        if type(record["critical"]) is not bool or identifier in record_ids:
            raise ValueError("duplicate record ID or invalid critical flag")
        record_ids.add(identifier)
        if record["critical"]:
            critical_ids.add(identifier)
        _features(record["features"], 16)
        if record["features"][6] != 0:
            raise ValueError("V2 record feature cannot encode origin state ID")
    required = [_uint(identifier) for identifier in _array(value["required_critical_records"], config.max_records)]
    if len(set(required)) != len(required) or not set(required) <= critical_ids:
        raise ValueError("missing or duplicate required critical record")
    candidates = [_move(movement) for movement in _array(value["candidates"], config.max_candidates)]
    if len(set(candidates)) != len(candidates):
        raise ValueError("duplicate captured candidate")
    divergences = _array(value["divergence_features"], config.max_divergences)
    for features in divergences:
        _features(features, 8)
    if value["role"] != "critic" and divergences:
        raise ValueError("proposer input cannot carry private Critic divergences")
    if value["query"][6] != 0 or value["query"][7] != 0:
        raise ValueError("V2 query cannot encode revision or deadline")
    lines = value["full_line"]
    _fields(lines, ("records", "query_prefix", "query_proposal", "query_counter"))
    line_records = _array(lines["records"], config.max_records, len(records))
    for index, record in enumerate(line_records):
        _fields(record, ("moves", "parent", "supersedes"), ("parent_required", "supersedes_required"))
        _line(record["moves"])
        for name in ("parent", "supersedes"):
            target, required = record[name], record.get(name + "_required", False)
            if type(required) is not bool or required and target is None:
                raise ValueError("missing required local relationship")
            if target is not None and (_uint(target) >= len(records) or target == index):
                raise ValueError("local relationship is out of range or self-referencing")
    visiting, complete = set(), set()
    def visit(index):
        if index in visiting:
            raise ValueError("captured local relationship cycle")
        if index in complete:
            return
        visiting.add(index)
        for name in ("parent", "supersedes"):
            target = line_records[index][name]
            if target is not None:
                visit(target)
        visiting.remove(index)
        complete.add(index)
    for index in range(len(records)):
        visit(index)
    for name in ("query_prefix", "query_proposal", "query_counter"):
        _line(lines[name])


def canonical_captured_input_key(value, config):
    """Exact LE framing of PalsModelInput::canonical_input_key, not a cache lookup.

    IDs, revisions and history bind provenance only; none enter the NN tensors.
    Rust remains the consumer authority and independently checks the same key.
    """
    validate_captured_input(value, config)
    digest = hashlib.sha256()
    digest.update(config.model_semantics.encode("utf-8"))
    digest.update(config.encoding.encode("utf-8"))
    digest.update(config.profile.encode("utf-8"))
    digest.update(bytes([0 if value["role"] == "proposer" else 1]))
    digest.update(_digest_bytes(value["model_epoch"]))
    digest.update(_digest_bytes(value["history_digest"]))
    def u64(number):
        digest.update(struct.pack("<Q", number))
    u64(value["situation_revision"])
    digest.update(bytes(value["board"]))
    for number in [*value["metadata"], *value["query"]]:
        digest.update(_f32_bytes(number))
    u64(len(value["records"]))
    for record in value["records"]:
        u64(record["record_id"])
        u64(record["revision"])
        digest.update(bytes([int(record["critical"])]))
        for number in record["features"]:
            digest.update(_f32_bytes(number))
    required = sorted(value["required_critical_records"])
    u64(len(required))
    for number in required:
        u64(number)
    u64(len(value["candidates"]))
    for movement in value["candidates"]:
        source, destination, promotion = _move(movement)
        digest.update(struct.pack("<H", source | destination << 6 | promotion << 12))
    u64(len(value["divergence_features"]))
    for features in value["divergence_features"]:
        for number in features:
            digest.update(_f32_bytes(number))
    def line(moves):
        u64(len(moves))
        for movement in moves:
            digest.update(bytes(_move(movement)))
    lines = value["full_line"]
    u64(len(lines["records"]))
    for record in lines["records"]:
        line(record["moves"])
        for name in ("parent", "supersedes"):
            u64((1 << 64) - 1 if record[name] is None else record[name])
        digest.update(bytes([int(record.get("parent_required", False)), int(record.get("supersedes_required", False))]))
    for name in ("query_prefix", "query_proposal", "query_counter"):
        line(lines[name])
    return digest.hexdigest()


def latent_bytes(bits):
    contents = b"".join(struct.pack("<I", _uint(part, 32)) for part in _array(bits, LATENT_ELEMENTS, LATENT_ELEMENTS))
    if any(not math.isfinite(part[0]) for part in struct.iter_unpack("<f", contents)):
        raise ValueError("seed contains nonfinite FP32 bits")
    return contents


def _validate_raw_output(value, input_value, bits=None):
    _fields(value, _RAW_FIELDS)
    lengths = {"candidate_logits": len(input_value["candidates"]), "wdl_logits": 3,
               "divergence_logits": len(input_value["divergence_features"]) if input_value["role"] == "critic" else None,
               "task_logits": None, "private_latent": LATENT_ELEMENTS}
    if bits is not None:
        _fields(bits, _RAW_FIELDS)
    for name, count in lengths.items():
        if count is None:
            if value[name] is not None or bits is not None and bits[name] is not None:
                raise ValueError("unexpected private role head")
            continue
        numbers = _array(value[name], count, count)
        if bits is not None:
            encoded = _array(bits[name], count, count)
        for index, number in enumerate(numbers):
            actual = _f32_bytes(number)
            if bits is not None and actual != struct.pack("<I", _uint(encoded[index], 32)):
                raise ValueError("raw output JSON does not match captured FP32 bits")


def _input_fp32_bits(value, bits):
    _fields(bits, ("metadata", "query", "records", "divergence_features"))
    def compare(values, encoded, count):
        encoded = _array(encoded, count, count)
        for number, part in zip(values, encoded):
            if _f32_bytes(number) != struct.pack("<I", _uint(part, 32)):
                raise ValueError("captured input JSON differs from actual FP32 bits")
    for name in ("metadata", "query"):
        compare(value[name], bits[name], 16)
    for name, count in (("records", 16), ("divergence_features", 8)):
        encoded = _array(bits[name], len(value[name]), len(value[name]))
        for item, row in zip(value[name], encoded):
            compare(item["features"] if name == "records" else item, row, count)


def validate_actual_capture(capture, export_manifest_sha256):
    """Pure admission binds the exact full input, mode, seed and completed call."""
    _sha(export_manifest_sha256)
    if (type(capture) is not dict or capture.get("schema") != CAPTURE_SCHEMA
            or capture.get("phase") != "capture" or capture.get("execution_checks_passed") is not True
            or capture.get("export_manifest_sha256_hex") != export_manifest_sha256):
        raise ValueError("capture schema, execution gate or export identity mismatch")
    config = ModelConfig(**capture["model_configuration"])
    config.validate()
    if not config.full_line or capture["model_configuration"] != config.to_dict():
        raise ValueError("capture requires the exact registered full-line config")
    calls = _array(capture.get("calls"), MAX_CAPTURE_CALLS)
    if not calls:
        raise ValueError("actual capture must contain at least one completed call")
    role_budget = _uint(capture.get("max_actual_role_forwards_including_startup"))
    if not MIN_ROLE_FORWARDS <= role_budget <= MAX_ROLE_FORWARDS or len(calls) > role_budget - STARTUP_ROLE_FORWARDS:
        raise ValueError("capture exceeds its explicit registered role-forward budget")
    actual_forwards = len(calls) + STARTUP_ROLE_FORWARDS
    if (_uint(capture.get("actual_role_forwards_including_startup")) != actual_forwards
            or _uint(capture.get("actual_completed_role_forwards_including_startup")) != actual_forwards
            or _uint(capture.get("actual_known_completed_startup_role_forwards")) != STARTUP_ROLE_FORWARDS
            or capture.get("role_forward_accounting_checked") is not True):
        raise ValueError("capture attempted/completed/startup forward accounting mismatch")
    requests, executions = set(), set()
    prior_seeds = []
    for call in calls:
        if type(call) is not dict:
            raise ValueError("capture call must be an object")
        request, execution = _identity(call["request_id"]), _identity(call["execution_id"])
        if request in requests or execution in executions:
            raise ValueError("duplicate captured request or execution")
        requests.add(request)
        executions.add(execution)
        if (call.get("physical") != {"dispatched": True, "ready": True, "completion_unknown": False, "completed_ok": True}
                or call.get("accepted") is not True or call.get("delivered") is not True
                or type(call.get("warm_start")) is not bool):
            raise ValueError("capture call lacks completed accepted physical execution")
        key = canonical_captured_input_key(call["input"], config)
        if _sha(call["input_key_hex"]) != key:
            raise ValueError("captured canonical full input key mismatch")
        encoded_input = call["input_json_utf8"]
        if type(encoded_input) is not str:
            raise ValueError("capture must preserve the actual serde input UTF-8")
        encoded_input = encoded_input.encode("utf-8")
        if not 0 < len(encoded_input) <= MAX_CAPTURE_INPUT_BYTES or hashlib.sha256(encoded_input).hexdigest() != _sha(call["input_json_sha256_hex"]):
            raise ValueError("captured exact serde input bytes digest mismatch")
        parsed_input = _json_bytes(encoded_input)
        if parsed_input != call["input"] or canonical_captured_input_key(parsed_input, config) != key:
            raise ValueError("captured input object differs from its actual serde bytes")
        _input_fp32_bits(call["input"], call["input_f32_bits"])
        seed_bytes = latent_bytes(call["initial_latent_bits"])
        if _sha(call["initial_latent_sha256_hex"]) != hashlib.sha256(seed_bytes).hexdigest():
            raise ValueError("captured initial latent bits digest mismatch")
        invocation = call["invocation"]
        warm = call["warm_start"]
        _fields(invocation, ("mode", "input_key_hex", "invocation_key_hex"), ("seed_seal_hex",))
        if invocation["input_key_hex"] != key or invocation["mode"] != ("approx_cuda_warm_v2" if warm else "fresh"):
            raise ValueError("captured Warm/Fresh invocation input or mode mismatch")
        _sha(invocation["invocation_key_hex"])
        if warm:
            seal = _sha(invocation.get("seed_seal_hex"))
            provenance = call.get("seed_provenance")
            if (type(provenance) is not dict or provenance.get("seal_hex") != seal
                    or provenance.get("role") != call["input"]["role"]
                    or provenance.get("latent_bits_digest_hex") != call["initial_latent_sha256_hex"]
                    or provenance.get("model_manifest_sha256_hex") != export_manifest_sha256
                    or provenance.get("model_epoch_hex") != _digest_bytes(call["input"]["model_epoch"]).hex()):
                raise ValueError("captured seed provenance does not bind this role/model/full seed")
            source_input = _sha(provenance.get("source_input_hex"))
            if not any(role == call["input"]["role"] and source == source_input and bits == call["initial_latent_bits"]
                       for role, source, bits in prior_seeds):
                raise ValueError("Warm seed is not a prior accepted same-role full latent in this capture")
        elif ("seed_seal_hex" in invocation or invocation["invocation_key_hex"] != key
              or call.get("seed_provenance") is not None):
            raise ValueError("Fresh invocation carries Warm seed provenance")
        _validate_raw_output(call["raw_output"], call["input"], call["raw_output_bits"])
        final_bits = call["raw_output_bits"]["private_latent"]
        if _sha(call["final_latent_sha256_hex"]) != hashlib.sha256(latent_bytes(final_bits)).hexdigest():
            raise ValueError("captured final full latent digest mismatch")
        prior_seeds.append((call["input"]["role"], key, final_bits))
    return config


def _json_bytes(contents):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate capture/reference JSON field")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    result = json.loads(contents.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    if type(result) is not dict:
        raise ValueError("capture/reference artifact must be a JSON object")
    return result


def _external_path(path, new_file=False):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("actual reference paths must be absolute")
    def secret_parts(value):
        for part in value.parts:
            name = part.lower()
            if (name == ".env" or name.startswith(".env.") or "credential" in name
                    or "service-account" in name or "service_account" in name or "api_key" in name
                    or name.startswith("id_ed25519") or name.startswith("id_rsa")):
                raise ValueError("secret path refused")
    secret_parts(path)
    resolved = path.resolve(strict=not new_file)
    secret_parts(resolved)
    repository = next((parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()), None)
    if repository is not None and (resolved == repository or repository in resolved.parents):
        raise ValueError("actual reference artifacts must be outside the checkout")
    if new_file:
        if path.exists() or path.with_name(path.name + ".partial").exists() or not path.parent.is_dir():
            raise ValueError("actual reference output must be a new file in an existing external directory")
    return path


def _compare_raw(actual, expected):
    maxima = {}
    for name in _RAW_FIELDS:
        if actual[name] is None or expected[name] is None:
            if actual[name] is not None or expected[name] is not None:
                raise ValueError("reference role head presence mismatch")
            continue
        if len(actual[name]) != len(expected[name]):
            raise ValueError("reference output length mismatch")
        maximum = 0.0
        for a, b in zip(actual[name], expected[name]):
            a, b = struct.unpack("<f", _f32_bytes(a))[0], struct.unpack("<f", _f32_bytes(b))[0]
            difference = abs(a - b)
            if difference > ATOL + RTOL * abs(b):
                raise ValueError("actual capture/reference numeric mismatch: " + name)
            maximum = max(maximum, difference)
        maxima[name] = maximum
    for name, head in (("policy", "candidate_logits"), ("wdl", "wdl_logits")):
        def probabilities(values):
            if not values:
                return []
            values = [struct.unpack("<f", _f32_bytes(number))[0] for number in values]
            largest = max(values)
            exponential = [math.exp(number - largest) for number in values]
            total = math.fsum(exponential)
            return [number / total for number in exponential]
        a, b = probabilities(actual[head]), probabilities(expected[head])
        difference = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
        if difference > PROBABILITY_ATOL:
            raise ValueError("actual capture/reference normalized probability mismatch: " + name)
        maxima[name] = difference
    return maxima


def _tensor_input(value, config):
    import numpy as np
    import torch
    from .model import TensorInput
    r, c, d = max(1, len(value["records"])), max(1, len(value["candidates"])), max(1, len(value["divergence_features"]))
    arrays = {"board": np.array([value["board"]], dtype=np.int64),
              "metadata": np.array([value["metadata"]], dtype=np.float32),
              "query": np.array([value["query"]], dtype=np.float32),
              "records": np.zeros((1, r, 16), dtype=np.float32), "record_mask": np.zeros((1, r), dtype=np.bool_),
              "candidates": np.zeros((1, c, 3), dtype=np.int64), "candidate_mask": np.zeros((1, c), dtype=np.bool_),
              "divergence_features": np.zeros((1, d, 8), dtype=np.float32), "divergence_mask": np.zeros((1, d), dtype=np.bool_),
              "record_line_tokens": np.zeros((1, r, MAX_LINE_PLIES, 3), dtype=np.int64),
              "record_line_mask": np.zeros((1, r, MAX_LINE_PLIES), dtype=np.bool_),
              "query_line_tokens": np.zeros((1, 3, MAX_LINE_PLIES, 3), dtype=np.int64),
              "query_line_mask": np.zeros((1, 3, MAX_LINE_PLIES), dtype=np.bool_),
              "record_relations": np.full((1, r, 2), -1, dtype=np.int64)}
    for index, record in enumerate(value["records"]):
        arrays["records"][0, index] = record["features"]
        arrays["record_mask"][0, index] = True
    for index, movement in enumerate(value["candidates"]):
        arrays["candidates"][0, index] = _move(movement)
        arrays["candidate_mask"][0, index] = True
    for index, features in enumerate(value["divergence_features"]):
        arrays["divergence_features"][0, index] = features
        arrays["divergence_mask"][0, index] = True
    def write_line(moves, index, token_name, mask_name):
        for ply, movement in enumerate(moves):
            arrays[token_name][0, index, ply] = _move(movement)
            arrays[mask_name][0, index, ply] = True
    for index, record in enumerate(value["full_line"]["records"]):
        write_line(record["moves"], index, "record_line_tokens", "record_line_mask")
        arrays["record_relations"][0, index] = [record[name] if record[name] is not None else -1 for name in ("parent", "supersedes")]
    for index, name in enumerate(("query_prefix", "query_proposal", "query_counter")):
        write_line(value["full_line"][name], index, "query_line_tokens", "query_line_mask")
    data = TensorInput(**{name: torch.from_numpy(array) for name, array in arrays.items()})
    data.validate(config)
    return data, arrays


def _reference_call(call, raw_ort, raw_torch):
    return {"request_id": call["request_id"], "execution_id": call["execution_id"],
            "input_key_hex": call["input_key_hex"], "input_json_sha256_hex": call["input_json_sha256_hex"],
            "input_f32_bits": call["input_f32_bits"], "full_line": call["input"]["full_line"],
            "initial_latent_sha256_hex": call["initial_latent_sha256_hex"], "warm_start": call["warm_start"],
            "invocation": call["invocation"], "ort_raw_output": raw_ort, "torch_raw_output": raw_torch,
            "max_abs_difference": {"ort_vs_torch": _compare_raw(raw_ort, raw_torch),
                                   "capture_vs_ort": _compare_raw(call["raw_output"], raw_ort),
                                   "capture_vs_torch": _compare_raw(call["raw_output"], raw_torch)}}


def _raw_output(outputs, value):
    import numpy as np
    c, d = len(value["candidates"]), len(value["divergence_features"])
    shapes = ((1, max(c, 1)), (1, 3), (1, 16, 384), (1, max(d, 1)), ())
    if len(outputs) != 5:
        raise ValueError("actual reference graph output count mismatch")
    for index, (output, shape) in enumerate(zip(outputs, shapes)):
        expected_dtype = np.bool_ if index == 4 else np.float32
        if output.shape != shape or output.dtype != expected_dtype or not np.isfinite(output).all():
            raise ValueError("actual reference graph output shape/dtype/finite mismatch")
    if bool(outputs[4]) != (value["role"] == "critic"):
        raise ValueError("actual reference graph role tag mismatch")
    result = {"candidate_logits": outputs[0][0, :c].tolist(), "wdl_logits": outputs[1][0].tolist(),
              "private_latent": outputs[2].reshape(-1).tolist(),
              "divergence_logits": outputs[3][0, :d].tolist() if value["role"] == "critic" else None, "task_logits": None}
    _validate_raw_output(result, value)
    return result


def _load_checkpoint_bytes(contents, metadata, config):
    import torch
    from .model import PalsModel
    if (metadata.get("schema") != config.model_semantics or metadata.get("config") != config.to_dict()
            or metadata.get("external_weights") is not False or metadata.get("external_teacher") is not False):
        raise ValueError("reference checkpoint declaration is incompatible; diagnostic training checkpoints are unsupported")
    checkpoint = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    steps = checkpoint.get("training_steps")
    if (checkpoint.get("schema") != config.model_semantics or checkpoint.get("config") != config.to_dict()
            or type(steps) is not int or steps < 0 or checkpoint.get("trained") is not (steps > 0)
            or metadata.get("training_steps") != steps or metadata.get("trained") is not checkpoint["trained"]):
        raise ValueError("reference checkpoint/receipt training provenance mismatch")
    if type(checkpoint.get("seed")) is not int:
        raise ValueError("reference checkpoint seed must be an integer")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(checkpoint["seed"])
        model = PalsModel(config).float()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if "validator" in model.experts:
        del model.experts["validator"]
    return model.eval()


def _source_hashes():
    directory = Path(__file__).resolve().parent
    names = ("config.py", "model.py", "onnx_shared.py", "onnx_warm.py", "artifacts.py", "warm_artifacts.py",
             "cuda_warm_artifacts.py", "cuda_warm_actual_reference.py")
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def _bind_native_source_identity(capture, manifest, metadata, config):
    source = capture.get("source_identity")
    if (type(source) is not dict or _digest_bytes(source["checkpoint_sha256"]).hex() != manifest["checkpoint_sha256"]
            or _digest_bytes(source["export_manifest_sha256"]).hex() != capture["export_manifest_sha256_hex"]
            or source.get("model_configuration") != config.to_dict() or source.get("trained") is not metadata["trained"]):
        raise ValueError("capture native owner/checkpoint/export identity mismatch")
    # These are capability-owned combined namespaces, distinct from the
    # Rules-only declaration. Its encoding includes loaded runtime/options;
    # its adapter digest combines native + warm consumer + implementation.
    encoding = _digest_bytes(source["encoding_semantic_sha256"]).hex()
    _digest_bytes(source["adapter_source_sha256"])
    frozen_epoch = _uint(source["frozen_epoch"])
    for call in capture["calls"]:
        if _digest_bytes(call["input"]["model_epoch"]).hex() != manifest["checkpoint_sha256"]:
            raise ValueError("actual input model epoch differs from the registered checkpoint")
        if call["warm_start"]:
            provenance = call["seed_provenance"]
            if provenance.get("encoding_semantic_sha256_hex") != encoding or _uint(provenance["frozen_epoch"]) != frozen_epoch:
                raise ValueError("Warm seed provenance differs from the captured loaded owner namespace")
    return source


def _write_reference(path, report):
    contents = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    if len(contents) > MAX_REFERENCE_BYTES:
        raise ValueError("actual reference report exceeds its output limit")
    # Atomic exclusive publication must not replace an existing acceptance report.
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    return {"schema": REFERENCE_SCHEMA, "reference_passed": True, "calls": len(report["calls"]),
            "output_json": str(path), "reference_sha256_hex": hashlib.sha256(contents).hexdigest(),
            "reference_bytes": len(contents), "optimizer_steps": 0, "gpu_execution": "not_run_by_reference"}


def numeric_check_cuda_warm_actual_reference(checkpoint, export_manifest, export_manifest_sha256,
                                             capture_json, capture_sha256, output_json, *, max_seconds=300):
    """Replay captured P/C calls within its explicitly registered 16..128 budget.

    This function checks a cooperative wall deadline around every execution;
    embedded callers must provide a physical process timeout. The CLI provides
    a separate worker with max_seconds + 30 seconds for termination and cleanup.
    Successful publication does not replace Rust verify-reference acceptance.
    """
    if type(max_seconds) is not int or not 1 <= max_seconds <= MAX_REFERENCE_SECONDS:
        raise ValueError("actual reference wall budget must be an integer in 1..900 seconds")
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("actual CPU reference wall deadline exceeded")
    checkpoint, export_manifest, capture_json, output_json = (
        _external_path(checkpoint), _external_path(export_manifest), _external_path(capture_json), _external_path(output_json, True))
    sources_before = _source_hashes()
    capture_bytes, capture_pin = _read_immutable_bytes(capture_json, MAX_CAPTURE_BYTES, _sha(capture_sha256))
    manifest_bytes, manifest_pin = _read_immutable_bytes(export_manifest, MAX_WARM_MANIFEST_BYTES, _sha(export_manifest_sha256))
    capture, manifest = _json_bytes(capture_bytes), _json_bytes(manifest_bytes)
    config = validate_actual_capture(capture, export_manifest_sha256)
    domain = warm_export_domain(config, cuda=True)
    metadata_bytes, metadata_pin = _read_immutable_bytes(checkpoint.with_name("checkpoint.json"), MAX_WARM_MANIFEST_BYTES)
    metadata = _json_bytes(metadata_bytes)
    validate_warm_export_manifest(manifest, metadata)
    if manifest.get("schema") != domain["schema"] or manifest["config"] != config.to_dict():
        raise ValueError("actual capture/reference export domain or profile mismatch")
    checkpoint_bytes, checkpoint_pin = _read_immutable_bytes(checkpoint, MAX_WARM_GRAPH_BYTES, _sha(manifest["checkpoint_sha256"]))
    source = _bind_native_source_identity(capture, manifest, metadata, config)
    graph_sources, graph_pins = {}, []
    remaining = MAX_WARM_GRAPH_BYTES
    for graph in manifest["graphs"]:
        path = _external_path(export_manifest.parent / graph["file"])
        contents, pin = _read_immutable_bytes(path, remaining, graph["sha256"])
        remaining -= len(contents)
        graph_sources[graph["role"]] = contents
        graph_pins.append({"file": graph["file"], "role": graph["role"], **pin})
    deadline()
    # No neural dependency is imported before the bounded capture/artifact admission.
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    if _WORKER_INTEROP:
        torch.set_num_interop_threads(1)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "onnx": onnx.__version__, "ort": ort.__version__}
    if (torch.__version__.split("+")[0] != "2.8.0" or np.__version__ != "2.2.6"
            or onnx.__version__ != "1.19.0" or ort.__version__ != "1.22.0"):
        raise ValueError("actual CPU reference requires the registered Torch/NumPy/ONNX/ORT versions")
    original_threads, original_dtype = torch.get_num_threads(), torch.get_default_dtype()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    reports = []
    try:
        torch.set_num_threads(2)
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = _load_checkpoint_bytes(checkpoint_bytes, metadata, config)
        del checkpoint_bytes
        for graph in manifest["graphs"]:
            document = _document_from_immutable_bytes(graph_sources[graph["role"]])
            properties = {prop.key: prop.value for prop in document.metadata_props}
            expected = {"schema": domain["schema"], "layout": domain["layout"], "layout_revision": "2", "role": graph["role"],
                        "checkpoint_sha256": manifest["checkpoint_sha256"], "precision": "fp32", "expected_ort": "1.22.0",
                        "model_semantics": config.model_semantics, "model_profile": config.profile, "encoding_schema": config.encoding,
                        **{name: manifest[name] for name in ("warm_graph_semantics", "private_seed_policy", "query_semantics", "execution_domain",
                                                           "rules_input_profile", "rules_input_semantic_sha256", "rules_encoder_source_sha256")}}
            if (len(properties) != len(document.metadata_props) or any(properties.get(name) != value for name, value in expected.items())
                    or _descriptors(document.graph.input) != graph["inputs"] or _descriptors(document.graph.output) != graph["outputs"]):
                raise ValueError("actual reference graph metadata or tensor contract mismatch")
        public = _cpu_session(graph_sources["public"])
        private = _cpu_session(graph_sources[domain["private_role"]])
        deadline()
        with torch.no_grad(), torch.device("cpu"):
            for call in capture["calls"]:
                deadline()
                value = call["input"]
                data, arrays = _tensor_input(value, config)
                seed = np.frombuffer(latent_bytes(call["initial_latent_bits"]), dtype="<f4").copy().reshape(1, 16, 384)
                memory_torch = model.public_encoder(*data.public_args(config))
                deadline()
                raw_torch = _raw_output(_reference(model, data, memory_torch, value["role"], seed, call["warm_start"]), value)
                deadline()
                memory_ort = public.run(None, {name: arrays[name] for name in public_tensor_names(config)})
                deadline()
                for expected, actual in zip(memory_torch, memory_ort):
                    expected = expected.detach().cpu().numpy()
                    if expected.dtype == np.bool_:
                        if not np.array_equal(expected, actual):
                            raise ValueError("actual public memory masks differ")
                    else:
                        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
                raw_ort = _raw_output(private.run(None, _feed(data, memory_ort, seed, value["role"], call["warm_start"], config)), value)
                deadline()
                reports.append(_reference_call(call, raw_ort, raw_torch))
        del model, public, private
    finally:
        torch.set_num_threads(original_threads)
        torch.set_default_dtype(original_dtype)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original_tf32
    deadline()
    sources_after = _source_hashes()
    if sources_before != sources_after:
        raise ValueError("reference model/helper source changed during execution")
    report = {"schema": REFERENCE_SCHEMA, "capture_sha256_hex": capture_pin["sha256"],
              "export_manifest_sha256_hex": manifest_pin["sha256"], "model_configuration": config.to_dict(),
              "reference_provider": "cpu", "reference_graph_execution": REFERENCE_EXECUTION,
              "calls": reports, "tolerance": {"atol": ATOL, "rtol": RTOL, "policy_wdl_max_atol": PROBABILITY_ATOL}, "versions": versions,
              "scope": "same_actual_input_seed_and_mode_raw_fp32_only", "optimizer_steps": 0,
              "max_actual_role_forwards_including_startup": capture["max_actual_role_forwards_including_startup"],
              "actual_captured_role_forwards": len(reports),
              "actual_training": False, "arena_eligible": False, "diagnostic_only": True,
              "fresh_vs_seeded_equality_required": False, "native_seed_eligibility": "captured_owner_decision_not_recertified",
              "gpu_execution": "not_run_by_reference", "final_acceptance": "requires_rust_verify_reference",
              "source_hashes_before": sources_before, "source_hashes_after": sources_after, "sources_unchanged": True,
              "artifacts": {"capture": capture_pin, "export_manifest": manifest_pin, "checkpoint": checkpoint_pin,
                            "checkpoint_metadata": metadata_pin, "graphs": graph_pins},
              "rules_input_profile": manifest["rules_input_profile"], "rules_input_semantic_sha256": manifest["rules_input_semantic_sha256"],
              "rules_encoder_source_sha256": manifest["rules_encoder_source_sha256"],
              "captured_native_source_identity": source,
              "captured_compiled_component_pins": capture.get("compiled_component_pins"),
              "native_source_scope": "captured_capability_namespace;_compiled_component_proof_remains_native_owner_responsibility",
              "forward_counts": {"torch_public": len(reports), "torch_private": len(reports),
                                 "ort_public": len(reports), "ort_private": len(reports)},
              "wall_limit_seconds": max_seconds, "elapsed_seconds": time.monotonic() - started,
              "torch_intra_op_threads": 2, "torch_inter_op_threads": torch.get_num_interop_threads(),
              "limits": {"capture_bytes": MAX_CAPTURE_BYTES, "report_bytes": MAX_REFERENCE_BYTES,
                         "calls": capture["max_actual_role_forwards_including_startup"] - STARTUP_ROLE_FORWARDS,
                         "registered_maximum_calls": MAX_CAPTURE_CALLS, "combined_graph_bytes": MAX_WARM_GRAPH_BYTES}}
    return _write_reference(output_json, report)


def add_actual_reference_parser(commands):
    parser = commands.add_parser("actual-reference", help="finite CPU Torch/ORT replay of exact captured CUDA inputs and seed")
    _arguments(parser)


def _arguments(parser):
    for name in ("checkpoint", "export-manifest", "export-manifest-sha256", "capture-json", "capture-sha256", "output-json"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-seconds", type=int, default=300)


def _worker(arguments, sender):
    global _WORKER_INTEROP
    _WORKER_INTEROP = True
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    try:
        result = numeric_check_cuda_warm_actual_reference(**arguments)
        sender.send({"ok": True, "result": result})
    except BaseException as error:
        # A concise bounded failure is sufficient; paths and raw tensors are not stdout.
        sender.send({"ok": False, "error_type": type(error).__name__, "error": str(error)[:1024]})
    finally:
        sender.close()


def run_actual_reference_cli(arguments):
    """A single finite worker; no retry, optimizer or CUDA provider execution."""
    if type(arguments.max_seconds) is not int or not 1 <= arguments.max_seconds <= MAX_REFERENCE_SECONDS:
        raise ValueError("actual reference wall budget must be an integer in 1..900 seconds")
    values = {"checkpoint": arguments.checkpoint, "export_manifest": arguments.export_manifest,
              "export_manifest_sha256": arguments.export_manifest_sha256, "capture_json": arguments.capture_json,
              "capture_sha256": arguments.capture_sha256, "output_json": arguments.output_json, "max_seconds": arguments.max_seconds}
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_worker, args=(values, sender), name="rz-pals-actual-cpu-reference")
    worker.start()
    sender.close()
    cleanup_deadline = None
    try:
        if not receiver.poll(arguments.max_seconds):
            raise TimeoutError("actual CPU reference worker exceeded its finite wall budget")
        try:
            result = receiver.recv()
        except EOFError:
            raise RuntimeError("actual CPU reference worker exited without a result") from None
        cleanup_deadline = time.monotonic() + CLEANUP_SECONDS
        worker.join(1)
        if worker.is_alive() or worker.exitcode != 0 or not result.get("ok"):
            raise RuntimeError("actual CPU reference failed: " + result.get("error_type", "worker_exit") + ": " + result.get("error", ""))
        return result["result"]
    finally:
        if cleanup_deadline is None:
            cleanup_deadline = time.monotonic() + CLEANUP_SECONDS
        def cleanup_remaining():
            return max(0.0, cleanup_deadline - time.monotonic())
        if worker.is_alive():
            worker.terminate()
            worker.join(min(CLEANUP_SECONDS / 2, cleanup_remaining()))
        if worker.is_alive():
            worker.kill()
            worker.join(cleanup_remaining())
        receiver.close()
        if worker.is_alive():
            raise RuntimeError("actual CPU reference physical worker cleanup is unknown")
        worker.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    _arguments(parser)
    arguments = parser.parse_args(argv)
    print(json.dumps(run_actual_reference_cli(arguments), ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
