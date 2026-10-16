"""Finite CPU reference for the exact bits accepted by the V3 ticket service.

This separate domain never admits V5 Warm startup or manager semantic work.
Importing this module performs no framework/model/native/device execution.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import time

DOMAIN = "rz-pals-v3-actual-seed-numeric/1"
SCOPE = "typed_service_seed_acceptance_not_manager_semantic_consumption"
SEED_DOMAIN = b"rz-pals-review-accepted-final-latent/1"
MAX_FILE = 4 * 1024 * 1024
MAX_TOTAL = 32 * 1024 * 1024


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _read(path, expected_sha256, maximum=MAX_FILE):
    if not _sha(expected_sha256):
        raise ValueError("exact lowercase SHA256 required")
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("absolute regular unlinked input required")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= maximum:
            raise ValueError("bounded single-link original file required")
        chunks = []
        count = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, maximum + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            if count > maximum:
                raise ValueError("input grew beyond bound")
        data = b"".join(chunks)
        after = os.fstat(fd)
        named = path.stat(follow_symlinks=False)
        fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
        if fields(before) != fields(after) or fields(after) != fields(named) or hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("original FD/path/bytes changed")
        return data
    finally:
        os.close(fd)


def _json(data):
    return json.loads(data, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _bits(values):
    if type(values) is not list or len(values) != 6144 or any(type(v) is not int or not 0 <= v < 2**32 for v in values):
        raise ValueError("exact full 6144 u32 latent bits required")
    data = struct.pack("<6144I", *values)
    numbers = struct.unpack("<6144f", data)
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError("nonfinite actual latent")
    return data, hashlib.sha256(SEED_DOMAIN + struct.pack("<Q", 6144) + data).hexdigest()


def _canonical_keys(raw):
    """Independent Rust execution_prefix/context/approximate_warm_key framing.

    The historical Python execution_keys uses a different JSON frame; it must
    not certify a Rust invocation. Only the existing typed tensor input_digest
    is shared here. Product key domains and accepted seed bits are unchanged.
    """
    from .numeric_capture import python_input
    from .virtual_review import input_digest, PURPOSES
    value = python_input(raw["input"])
    identity = raw["identity"]
    histories = identity["anchor_history_seals"]
    if (identity["fp32"] is not True or type(histories) is not list
            or len(histories) != len(value["anchors"])
            or histories[0] != identity["rules_history"]
            or not all(_sha(v) for v in [identity["model_epoch"], identity["rules_history"],
                                       identity["backend"], *histories])):
        raise ValueError("canonical execution requires every ordered original identity digest")
    role = ("proposer", "critic", "validator").index(value["role"])
    def prefix(domain):
        return (domain + bytes.fromhex(identity["model_epoch"])
                + bytes.fromhex(identity["rules_history"]) + bytes.fromhex(identity["backend"])
                + b"fp32" + struct.pack("<Q", len(histories))
                + b"".join(bytes.fromhex(v) for v in histories) + bytes([role]))
    tensor = bytes.fromhex(input_digest(value))
    fresh = hashlib.sha256(prefix(b"rovezero.pals-review-fresh.v1\0") + tensor).hexdigest()
    seed = raw.get("accepted_seed_digest")
    if seed is None:
        return {"fresh_key": fresh, "warm_key": None}
    query = value["query"]
    if not _sha(seed) or value["role"] == "validator" or query["purpose"] == "value_fresh":
        raise ValueError("canonical ApproxWarm requires an eligible P/C query and actual seed digest")
    context = hashlib.sha256(prefix(b"rovezero.pals-review-context.v1\0"))
    context.update(bytes([PURPOSES.index(query["purpose"]), query["perspective"] & 0xff,
                          int(query["counter_present"])]))
    for name, index in zip(("prefix", "proposal", "counter"), query["anchors"]):
        line = query[name]
        context.update(struct.pack("<Q", len(line)))
        for movement in line:
            context.update(bytes(movement))
        anchor = value["anchors"][index]
        context.update(bytes(anchor["board"]))
        context.update(struct.pack("<16f", *anchor["metadata"]))
    context.update(struct.pack("<Q", len(value["candidates"])))
    for movement in value["candidates"]:
        context.update(bytes(movement))
    context.update(struct.pack("<Q", len(value["divergences"])))
    for fields in value["divergences"]:
        context.update(struct.pack("<8f", *fields))
    warm = hashlib.sha256(prefix(b"rovezero.pals-review-approx-warm.v1\0")
                          + context.digest() + tensor + bytes.fromhex(seed)).hexdigest()
    return {"fresh_key": fresh, "warm_key": warm}


def _validate_raw(raw):
    if (raw.get("domain") != DOMAIN or raw.get("scope") != SCOPE
            or raw.get("physical_completion_known") is not True or raw.get("numerical_acceptance") is not False):
        raise ValueError("original separate physical raw domain required")
    data, digest = _bits(raw["final_latent_bits"])
    values = raw["raw_output"]["private_latent"]
    if (digest != raw["final_latent_sha256"] or not any(v & 0x7fffffff for v in raw["final_latent_bits"])
            or type(values) is not list or len(values) != 6144
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or struct.pack("<6144f", *values) != data):
        raise ValueError("raw final latent and original bit digest differ")
    ticket = raw["ticket"]
    identity = raw["identity"]
    if (ticket["role"] not in ("Proposer", "Critic") or identity["fp32"] is not True
            or ticket["representation"]["precision"] != "fp32"
            or ticket["representation"]["model"] != identity["model_epoch"]
            or raw["bindings"]["checkpoint_sha256"] != identity["model_epoch"]
            or ticket["line_bucket"] != raw["line_bucket"]
            or raw["line_bucket"] not in (8, 16, 32, 64, 128, 256)
            or raw["input"]["role"] != ticket["role"].lower()):
        raise ValueError("original V-free query/model/role/L identity differs")
    output = raw["raw_output"]
    if (set(output) != {"candidate_logits", "wdl_logits", "private_latent", "divergence_logits", "task_logits"}
            or output["task_logits"] is not None
            or len(output["candidate_logits"]) != len(raw["input"]["candidates"])
            or len(output["wdl_logits"]) != 3
            or (ticket["role"] == "Proposer" and output["divergence_logits"] is not None)
            or (ticket["role"] == "Critic" and (type(output["divergence_logits"]) is not list
                or len(output["divergence_logits"]) != len(raw["input"]["divergences"])) )):
        raise ValueError("original P/C-only raw output heads differ")
    counters = raw["owner_accounting"]
    execution = raw["execution"]
    if (type(execution) is not dict or set(execution) != {"epoch", "sequence"}
            or any(type(v) is not int or v <= 0 for v in execution.values())
            or execution["epoch"] != ticket["process_epoch"]):
        raise ValueError("original owner execution identity differs")
    if (type(raw["completed_nn_inputs"]) is not int
            or any(type(v) is not int or v < 0 for v in counters.values())
            or counters["role_nn_runs_attempted"] != 1 or counters["role_nn_runs_completed"] != 1
            or counters["public_nn_runs_completed"] not in (0, 1)
            or counters["public_nn_runs_attempted"] != counters["public_nn_runs_completed"]
            or any(counters[k] != 0 for k in ("public_nn_runs_failed_known", "role_nn_runs_failed_known",
                       "validator_nn_runs_attempted", "validator_nn_runs_completed", "validator_nn_runs_failed_known"))
            or raw["completed_nn_inputs"] != counters["public_nn_runs_completed"] + 1):
        raise ValueError("original completed P/C-only owner accounting differs")
    keys = _canonical_keys(raw)
    if (type(raw["physical_fresh"]) is not bool
            or ticket["representation"]["input"] != keys["fresh_key"]
            or raw["invocation_key"] != (keys["fresh_key"] if raw["physical_fresh"] else keys["warm_key"])
            or (not raw["physical_fresh"] and keys["warm_key"] is None)):
        raise ValueError("original physical invocation identity differs")
    return digest


def _validate_chain(current, predecessor):
    _validate_raw(current)
    if predecessor is None:
        if (current.get("physical_fresh") is not True or current.get("actual_seed") is not None
                or current.get("accepted_seed_digest") is not None):
            raise ValueError("first invocation must be unseeded physical Fresh")
        return None
    old = predecessor["raw"]
    link = current.get("actual_seed")
    if current.get("physical_fresh") is not False or type(link) is not dict:
        raise ValueError("actual seeded Warm invocation required")
    _, digest = _bits(link["latent_bits"])
    if (not any(v & 0x7fffffff for v in link["latent_bits"])
            or digest != link["latent_sha256"] or digest != current["accepted_seed_digest"]
            or digest != old["final_latent_sha256"] or link["latent_bits"] != old["final_latent_bits"]
            or link["case"] != old["case"] or link["ticket"] != old["ticket"]
            or link["raw"] != predecessor["raw_pin"] or link["retirement"] != predecessor["retirement_pin"]
            or current["input"] != old["input"] or current["identity"] != old["identity"]
            or current["bindings"] != old["bindings"]):
        raise ValueError("Warm seed differs from the exact original accepted predecessor")
    a, b = old["ticket"], current["ticket"]
    for field in ("process_epoch", "game", "root", "authority_implementation", "role", "context_policy", "root_state"):
        if a[field] != b[field]:
            raise ValueError("seed crossed original authority/role/context: " + field)
    if a["request_sequence"] >= b["request_sequence"] or a["view_sequence"] >= b["view_sequence"]:
        raise ValueError("spent request/view sequence reused")
    if a["representation"] != b["representation"]:
        raise ValueError("same-input numeric control representation changed")
    return link["latent_bits"]


def _captures(config):
    receipt_path = Path(config["capture_receipt"])
    receipt = _json(_read(receipt_path, config["capture_receipt_sha256"]))
    if (receipt.get("domain") != DOMAIN or receipt.get("scope") != SCOPE
            or receipt.get("status") != "captured_actual_seed_numbers"
            or receipt.get("physical_shutdown_confirmed") is not True
            or receipt.get("native_owner_joined") is not True or receipt.get("quarantined") is not False
            or receipt.get("numerical_acceptance") is not False
            or receipt.get("production_manager_semantic_consumption") is not False
            or receipt.get("production_v5_warm_admission") != "NOT_READY"
            or receipt.get("provider") not in ("cpu", "cuda-control", "cuda-control-shim")
            or (receipt["provider"] != "cpu" and type(receipt.get("actual_virtual_cuda_placement")) is not dict)):
        raise ValueError("original completed typed-service capture/closure scope required")
    root = receipt_path.parent
    total = 0
    def original(pin):
        nonlocal total
        if (type(pin) is not dict or set(pin) != {"file", "bytes", "sha256"}
                or type(pin["file"]) is not str or Path(pin["file"]).name != pin["file"]
                or not pin["file"].endswith(".json") or type(pin["bytes"]) is not int
                or not 0 < pin["bytes"] <= MAX_FILE):
            raise ValueError("bounded original leaf artifact pin required")
        data = _read(root / pin["file"], pin["sha256"])
        total += len(data)
        if len(data) != pin["bytes"] or total > MAX_TOTAL:
            raise ValueError("original artifact size/aggregate bound differs")
        return _json(data)
    if type(receipt.get("artifacts")) is not list or len(receipt["artifacts"]) != 49:
        raise ValueError("exact raw/retirement and original negative-case roster required")
    captures = []
    duplicates = []
    canceled = []
    for item in receipt["artifacts"]:
        if "raw" not in item:
            value = original(item)
            if item["file"].endswith(".duplicate-rejected.json"):
                if value.get("domain") != DOMAIN or value.get("error") != "UnknownPreparation" or value.get("nn_execution_claim") is not False:
                    raise ValueError("original duplicate acknowledgement rejection differs")
                duplicates.append(value["ticket"])
            elif item["file"] == "canceled-before-execute.json":
                if (value.get("domain") != DOMAIN or value.get("original_failure") != "Canceled"
                        or value.get("native_execute_submitted") is not False):
                    raise ValueError("original prepared cancellation differs")
                canceled.append(value)
            else:
                raise ValueError("unexpected negative-case artifact")
            continue
        if set(item) != {"raw", "retirement"}:
            raise ValueError("exact raw/retirement pair required")
        raw = original(item["raw"])
        retired = original(item["retirement"])
        _validate_raw(raw)
        if (retired.get("domain") != DOMAIN or retired.get("scope") != SCOPE
                or retired.get("ticket") != raw["ticket"] or retired.get("raw") != item["raw"]
                or retired.get("native_retired_accepted") is not True
                or retired.get("registry_logical_acceptance") is not True
                or retired.get("production_manager_semantic_consumption") is not False):
            raise ValueError("original successful native retirement/logical acceptance required")
        captures.append({"raw": raw, "raw_pin": item["raw"], "retirement_pin": item["retirement"]})
    if len(captures) != 24:
        raise ValueError("exact twelve Fresh/Warm pairs required")
    if duplicates != [value["raw"]["ticket"] for value in captures] or len(canceled) != 1:
        raise ValueError("original negative-case ticket coverage differs")
    coverage = set()
    sequence = 0
    execution_sequence = 0
    for i in range(0, len(captures), 2):
        a, b = captures[i:i+2]
        _validate_chain(a["raw"], None)
        _validate_chain(b["raw"], a)
        key = (a["raw"]["ticket"]["role"], a["raw"]["line_bucket"])
        if key in coverage:
            raise ValueError("duplicate role/L coverage")
        coverage.add(key)
        for value in (a, b):
            if value["raw"]["ticket"]["request_sequence"] <= sequence:
                raise ValueError("global original request sequence regressed")
            sequence = value["raw"]["ticket"]["request_sequence"]
            if value["raw"]["execution"]["sequence"] <= execution_sequence:
                raise ValueError("original physical execution sequence regressed")
            execution_sequence = value["raw"]["execution"]["sequence"]
    if coverage != {(r, n) for r in ("Proposer", "Critic") for n in (8, 16, 32, 64, 128, 256)}:
        raise ValueError("P/C six-L coverage differs")
    return receipt, captures


class _Output:
    """Single numerical run's retained Linux output directory, never a registry."""
    def __init__(self, path):
        if sys.platform != "linux":
            raise ValueError("retained numerical output FD currently requires Linux")
        self.path = Path(path)
        if (not self.path.is_absolute() or self.path.exists()
                or any(p.is_symlink() or (p / ".git").exists() for p in self.path.parents)):
            raise ValueError("new outside-Git output root without linked ancestors required")
        self.path.mkdir()
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.total = 0
        try:
            self.verify()
        except Exception:
            os.close(self.fd)
            raise

    def verify(self):
        held = os.fstat(self.fd)
        named = self.path.stat(follow_symlinks=False)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_uid, s.st_gid)
        if (not stat.S_ISDIR(named.st_mode) or identity(held) != identity(named)
                or any(p.is_symlink() or (p / ".git").exists() for p in self.path.parents)):
            raise ValueError("original output directory FD/path changed")

    def publish(self, name, value):
        self.verify()
        if Path(name).name != name:
            raise ValueError("simple output leaf required")
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        if len(data) > MAX_FILE or self.total + len(data) > MAX_TOTAL:
            raise ValueError("bounded reference receipt exhausted")
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        with os.fdopen(fd, "wb") as file:
            if file.write(data) != len(data):
                raise OSError("incomplete reference publication")
            file.flush()
            os.fsync(file.fileno())
            held = os.fstat(file.fileno())
            named = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            if (held.st_dev, held.st_ino, held.st_nlink, held.st_size) != (named.st_dev, named.st_ino, 1, len(data)):
                raise ValueError("original output file identity changed")
        self.total += len(data)
        self.verify()
        return {"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def close(self):
        os.close(self.fd)


def run(config, *, execute=False):
    if execute is not True:
        raise ValueError("explicit execute=True required for actual CPU reference")
    expected = {"capture_receipt", "capture_receipt_sha256", "checkpoint", "export_manifest",
                "output_root", "reference_source_sha256", "max_seconds"}
    if type(config) is not dict or set(config) != expected or type(config["max_seconds"]) is not int or not 1 <= config["max_seconds"] <= 3600:
        raise ValueError("exact finite actual-seed reference config required")
    deadline = time.monotonic() + config["max_seconds"]
    _read(Path(__file__).resolve(), config["reference_source_sha256"])
    receipt, captures = _captures(config)
    bindings = captures[0]["raw"]["bindings"]
    if any(c["raw"]["bindings"] != bindings for c in captures):
        raise ValueError("capture inputs/source/checkpoint changed between calls")
    manifest = _json(_read(Path(config["export_manifest"]), bindings["export_manifest_sha256"], 256*1024))
    from .numeric_capture import python_input, source_pins
    for value in captures:
        python_input(value["raw"]["input"])
    sources = source_pins()
    if {k: bindings["source_components"].get(k) for k in sources} != sources:
        raise ValueError("current independent reference/source components differ")
    for key in ("checkpoint", "export_manifest", "output_root"):
        if not Path(config[key]).is_absolute():
            raise ValueError("absolute reference asset/output paths required")
    owner = _Output(config["output_root"])
    try:
        owner.publish("registration.json", {"domain": DOMAIN, "scope": SCOPE, "config": config,
                     "numerical_acceptance": False, "production_v5_warm_admission": "NOT_READY"})
        return _compare(config, captures, bindings, manifest, python_input, deadline, owner.publish)
    finally:
        owner.close()


def _compare(config, captures, bindings, manifest, python_input, deadline, publish):
    import torch
    from .artifacts import digest_file, load_checkpoint, validate_virtual_warm_export_manifest
    from .model import VirtualReviewTensorInput
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if digest_file(config["checkpoint"]) != bindings["checkpoint_sha256"]:
        raise ValueError("original checkpoint bytes differ")
    source_model, metadata = load_checkpoint(config["checkpoint"])
    model = source_model.virtual_review_pc_export_view()
    validate_virtual_warm_export_manifest(manifest, metadata)
    if model.training or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("frozen CPU FP32 reference required")
    artifacts = []
    with torch.no_grad():
        for i, capture in enumerate(captures):
            if time.monotonic() >= deadline:
                raise TimeoutError("finite reference window exhausted; prior originals retained")
            raw = capture["raw"]
            payload = python_input(raw["input"])
            data = VirtualReviewTensorInput.from_payload(payload)
            data.validate(model.config)
            memory = model.public_encoder(*data.public_args(model.config))
            seed = (_validate_chain(raw, captures[i-1]) if not raw["physical_fresh"] else None)
            # Fresh uses the Warm graph's declared unseeded branch. Warm uses
            # the exact prior GPU/CPU service output bits, never a new CPU seed.
            initial = torch.zeros((1, 16, 384), dtype=torch.float32) if seed is None else torch.tensor(struct.unpack("<6144f", _bits(seed)[0]), dtype=torch.float32).reshape(1,16,384)
            role = payload["role"]
            expected_raw = model.warm_role_graph(role)(*data.role_args(memory, model.config), initial, torch.tensor(seed is not None, dtype=torch.bool))
            if time.monotonic() >= deadline:
                raise TimeoutError("reference completed after its original deadline")
            expected_output = {"candidate_logits": expected_raw[0][0, :len(payload["candidates"])],
                               "wdl_logits": expected_raw[1][0], "private_latent": expected_raw[2][0].flatten()}
            if role == "critic":
                expected_output["divergence_logits"] = expected_raw[3][0, :len(payload["divergences"]) ]
            maxima = {}
            for field, expected_tensor in expected_output.items():
                actual_values = raw["raw_output"][field]
                if type(actual_values) is not list or any(type(v) not in (int,float) for v in actual_values):
                    raise ValueError("raw numeric vector required: " + field)
                actual_tensor = torch.tensor(actual_values, dtype=torch.float32)
                if (expected_tensor.shape != actual_tensor.shape or not torch.isfinite(expected_tensor).all()
                        or not torch.isfinite(actual_tensor).all()
                        or not torch.allclose(expected_tensor, actual_tensor, atol=1e-4, rtol=1e-3)):
                    raise ValueError("same-actual-seed raw tolerance: " + field)
                maxima[field] = float(torch.max(torch.abs(expected_tensor-actual_tensor))) if actual_tensor.numel() else 0.0
                if field in ("candidate_logits", "wdl_logits") and actual_tensor.numel():
                    if torch.max(torch.abs(torch.softmax(expected_tensor,-1)-torch.softmax(actual_tensor,-1))) > 1e-4:
                        raise ValueError("same-actual-seed policy/WDL tolerance: " + field)
            artifacts.append(publish(raw["case"] + ".reference.json", {
                "domain": DOMAIN, "scope": SCOPE, "producer": "independent_pytorch_cpu_fp32_same_actual_seed",
                "original_raw": capture["raw_pin"], "original_retirement": capture["retirement_pin"],
                "actual_seed_digest": raw["accepted_seed_digest"], "input": raw["input"],
                "raw_output": {k: v.tolist() for k,v in expected_output.items()}, "max_absolute_differences": maxima}))
            del data, memory, initial, expected_raw, expected_output
    return publish("comparison-receipt.json", {"domain": DOMAIN, "scope": SCOPE,
        "status": "same_actual_seed_raw_comparison_passed", "capture_receipt_sha256": config["capture_receipt_sha256"],
        "reference_source_sha256": config["reference_source_sha256"], "checkpoint_sha256": bindings["checkpoint_sha256"],
        "artifacts": artifacts, "numerical_acceptance": False, "production_v5_warm_admission": "NOT_READY",
        "production_manager_semantic_consumption": False, "performance_evidence_eligible": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = _json(_read(Path(args.config), args.config_sha256, 64*1024))
    print(json.dumps(run(config, execute=args.execute), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
