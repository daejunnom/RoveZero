"""Original G0 preparation/reference child; ordinary invocation is metadata only.

Preparation calls the registered fresh initializer itself. Its observation is
not a native capability: the Rust parent must keep the original child owner and
actual file descriptors before dispatching the separate reference child. JSON
cannot recreate that owner or rename a diagnostic, trained or legacy asset G0.
No optimizer, game dispatcher, native numerical acceptance or GPU is run here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

from .generation_data import digest, fields, identifier, sha, strict_json, uint
from .generation_training_cli import (
    MAX_JSON_BYTES, MAX_WALL_NS, _artifact,
    _bounded_checkpoint_snapshot, _file_identifier, _install_original_dependencies,
    _output_credit, _pinned_bytes, _regular_readback, _safe_absolute,
    _staging_bytes, _terminal_receipt,
)
from .generation_numerical_cli import (
    _model_bits_digest, _readback_pin, _runtime_readbacks, _torch_runtime_readback,
)
from .initialization_evidence import (
    EVIDENCE_NAME, MAX_CHECKPOINT_BYTES, MAX_EVIDENCE_BYTES, MAX_METADATA_BYTES,
    _registered_function, validate_source_registration,
)

REGISTRATION_DOMAIN = "rz-pals-g0-numerical-registration/1"
STARTUP_DOMAIN = "rz-pals-original-g0-numerical-startup/1"
RESULT_DOMAIN = "rz-pals-g0-numerical-cli-window/1"
SOURCE_COMPONENTS = ("model_native", "model_encoding", "nn_runtime", "model_reference", "model_export")
PHASES = ("prepare", "reference")
MAX_OUTPUT_BYTES = 18 * 1024 ** 3
MAX_GRAPH_BYTES = 512 * 1024 ** 2
MAX_PARAMETER_BYTES = 128 * 1024 ** 2
GRAPH_FILES = {"public": "public_memory.onnx", "shared_pc": "shared_pc_if.onnx", "validator": "role_validator.onnx"}
PREPARE_INPUTS = ("rules_profile", "case_plan_template")
REFERENCE_INPUTS = ("case_plan", "checkpoint", "checkpoint_metadata", "initialization_evidence",
                    "export_manifest", "export_graphs", "cuda_control_inventory")


def validate_registration(value):
    """Pure declarations only; never open artifacts or issue an owner."""
    fields(value, ("schema", "numerical_id", "phase", "source_components", "source_commit",
        "initialization_sources", "staging_root", "initialization_directory", "export_directory",
        "reference_output_root", "receipt_path", "max_wall_ns", "max_output_bytes", "inputs"),
        "original G0 numerical registration")
    if value["schema"] != REGISTRATION_DOMAIN or value["phase"] not in PHASES:
        raise ValueError("registered original G0 phase/domain required")
    _file_identifier(value["numerical_id"])
    if type(value["source_commit"]) is not str or re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]) is None:
        raise ValueError("full registered G0 source commit required")
    fields(value["source_components"], SOURCE_COMPONENTS, "exact G0 numerical source components")
    for pin in value["source_components"].values():
        sha(pin)
    validate_source_registration(value["initialization_sources"])
    if value["initialization_sources"]["source_commit"] != value["source_commit"]:
        raise ValueError("initializer and numerical source commits differ")
    if (not uint(value["max_wall_ns"], MAX_WALL_NS) or not uint(value["max_output_bytes"], MAX_OUTPUT_BYTES)
            or value["max_output_bytes"] <= MAX_JSON_BYTES):
        raise ValueError("finite original G0 time/output ceilings required")
    root = _safe_absolute(value["staging_root"])
    outputs = [_safe_absolute(value[name]) for name in (
        "initialization_directory", "export_directory", "reference_output_root", "receipt_path")]
    if any(path == root or not path.is_relative_to(root) for path in outputs):
        raise ValueError("every G0 output must be inside its original managed staging root")
    if any(left == right or left.is_relative_to(right) or right.is_relative_to(left)
           for i, left in enumerate(outputs) for right in outputs[i + 1:]):
        raise ValueError("G0 initialization/export/reference/terminal paths must be disjoint")
    if outputs[-1].suffix != ".json":
        raise ValueError("original G0 terminal receipt requires a JSON path")
    source_root = Path(__file__).absolute().parents[5]
    if root == source_root or root.is_relative_to(source_root):
        raise ValueError("G0 artifacts must remain outside the private source tree")
    names = PREPARE_INPUTS if value["phase"] == "prepare" else REFERENCE_INPUTS
    fields(value["inputs"], names, "exact phase-specific original G0 inputs")
    inputs = value["inputs"]
    if value["phase"] == "prepare":
        _artifact(inputs["rules_profile"], MAX_METADATA_BYTES)
        _artifact(inputs["case_plan_template"], 4 * 1024 ** 2)
        pins = list(inputs.values())
    else:
        for name, maximum in (("case_plan", 4 * 1024 ** 2), ("checkpoint", MAX_CHECKPOINT_BYTES),
                ("checkpoint_metadata", MAX_METADATA_BYTES), ("initialization_evidence", MAX_EVIDENCE_BYTES),
                ("export_manifest", 256 * 1024), ("cuda_control_inventory", 4 * 1024 ** 2)):
            _artifact(inputs[name], maximum)
        initial, export = outputs[:2]
        expected_paths = {"checkpoint": initial / "untrained.pt", "checkpoint_metadata": initial / "checkpoint.json",
            "initialization_evidence": initial / EVIDENCE_NAME, "export_manifest": export / "export.json",
            "cuda_control_inventory": export / "cuda-control-inventory.json"}
        if any(Path(inputs[name]["path"]) != path for name, path in expected_paths.items()):
            raise ValueError("G0 reference input escaped the exact original output roster")
        graphs = inputs["export_graphs"]
        if type(graphs) is not list or [g.get("role") for g in graphs if type(g) is dict] != list(GRAPH_FILES):
            raise ValueError("G0 Collection requires exactly the public/shared P-C/validator Fresh graphs")
        for graph in graphs:
            fields(graph, ("role", "file", "path", "sha256", "bytes"), "G0 original graph pin")
            _artifact({name: graph[name] for name in ("path", "sha256", "bytes")}, MAX_GRAPH_BYTES)
            if graph["file"] != GRAPH_FILES[graph["role"]] or Path(graph["path"]) != export / graph["file"]:
                raise ValueError("G0 graph path/name/role mismatch")
        pins = [inputs[name] for name in REFERENCE_INPUTS if name != "export_graphs"] + graphs
    paths = [pin["path"] for pin in pins]
    if len(set(paths)) != len(paths) or value["receipt_path"] in paths:
        raise ValueError("G0 original input/terminal paths cannot alias")
    generated = outputs[:3] if value["phase"] == "prepare" else outputs[2:3]
    if any(Path(path) == target or Path(path).is_relative_to(target) for path in paths for target in generated):
        raise ValueError("new G0 output cannot contain an original input")
    return copy.deepcopy(value)


def registration_digest(value):
    return digest(REGISTRATION_DOMAIN, validate_registration(value))


def validate_startup(value, *, expected_raw_sha256, raw_bytes, execution_until_ns, whole_until_ns, now_ns):
    """Join original immutable transport/deadlines; serialized pins grant no authority."""
    fields(value, ("schema", "execution_id", "phase", "registration_sha256", "registration",
        "original_started_ns", "execution_until_ns", "whole_until_ns", "dependency_directories",
        "python_runtime", "runtime_module_pins"), "original G0 numerical startup")
    if value["schema"] != STARTUP_DOMAIN:
        raise ValueError("registered original G0 startup required")
    identifier(value["execution_id"])
    if (type(raw_bytes) is not bytes or not 0 < len(raw_bytes) <= MAX_JSON_BYTES
            or hashlib.sha256(raw_bytes).hexdigest() != sha(expected_raw_sha256)):
        raise ValueError("G0 startup bytes differ from the original parent raw pin")
    registration = validate_registration(value["registration"])
    if value["phase"] != registration["phase"] or registration_digest(registration) != sha(value["registration_sha256"]):
        raise ValueError("G0 startup phase/registration differs")
    dependencies = value["dependency_directories"]
    if type(dependencies) is not list or not 1 <= len(dependencies) <= 8:
        raise ValueError("finite expressly registered G0 dependency directories required")
    paths, identities = set(), set()
    for entry in dependencies:
        fields(entry, ("path", "device", "inode"), "original G0 dependency directory")
        path, identity = str(_safe_absolute(entry["path"])), (uint(entry["device"]), uint(entry["inode"]))
        if not identity[1] or path in paths or identity in identities:
            raise ValueError("duplicate/aliased original G0 dependency directory")
        paths.add(path)
        identities.add(identity)
    runtime = value["python_runtime"]
    fields(runtime, ("version", "executable"), "registered original G0 Python runtime")
    if type(runtime["version"]) is not str or re.fullmatch(r"3\.[0-9]{1,2}\.[0-9]{1,3}", runtime["version"]) is None:
        raise ValueError("registered G0 Python release required")
    _artifact(runtime["executable"], 256 * 1024 ** 2)
    modules = value["runtime_module_pins"]
    if type(modules) is not list or not 1 <= len(modules) <= 64:
        raise ValueError("finite explicit G0 runtime import artifact roster required")
    module_paths = set()
    for pin in modules:
        _artifact(pin, 2 * 1024 ** 3)
        if pin["path"] in module_paths:
            raise ValueError("duplicate G0 runtime module artifact")
        module_paths.add(pin["path"])
    for number in (value["original_started_ns"], value["execution_until_ns"], value["whole_until_ns"],
                   execution_until_ns, whole_until_ns, now_ns):
        uint(number)
    if (value["execution_until_ns"] != execution_until_ns or value["whole_until_ns"] != whole_until_ns
            or not value["original_started_ns"] <= now_ns < execution_until_ns <= whole_until_ns
            or execution_until_ns - value["original_started_ns"] > registration["max_wall_ns"]
            or whole_until_ns - execution_until_ns > 30 * 10 ** 9):
        raise ValueError("G0 original deadline renewal/mismatch/expiration")
    return registration


def _json_pin(pin, maximum, *, until_ns, clock):
    return strict_json(_pinned_bytes(pin, maximum=maximum, until_ns=until_ns, clock=clock).decode("utf-8", "strict"))


def _original_inputs(registration, *, until_ns, clock):
    return {name: ([_readback_pin(pin, until_ns=until_ns, clock=clock) for pin in value]
                   if name == "export_graphs" else _readback_pin(value, until_ns=until_ns, clock=clock))
            for name, value in registration["inputs"].items()}


def _pin(path, maximum, *, until_ns, clock):
    return _regular_readback(path, maximum=maximum, until_ns=until_ns, clock=clock)


def _source_readback(registration, *, until_ns, clock):
    from .numeric_capture import source_pins
    # The training helper's inventory queries seven trainer/data components.
    # This child installs only the declared five numerical components; check
    # every actually loaded source module without silently asking for absent
    # collector sources or broadening the original parent's installation.
    package = Path(__file__).absolute().parent
    selected = sorted((name, module) for name, module in sys.modules.items()
                      if name == "rz_pals_model" or name.startswith("rz_pals_model."))
    modules = []
    if not 1 <= len(selected) <= 127:
        raise ValueError("original G0 active source module cap exceeded")
    for name, module in selected:
        path = _safe_absolute(getattr(module, "__file__", ""))
        if not path.is_relative_to(package) or path.suffix != ".py":
            raise ValueError("G0 imported source escaped the original private package boundary")
        modules.append({"module": name, **_pin(path, 4 * 1024 ** 2, until_ns=until_ns, clock=clock)})
    modules.append({"module": "__main__", **_pin(__file__, 4 * 1024 ** 2, until_ns=until_ns, clock=clock)})
    actual = {"source_components": source_pins(), "source_modules": modules}
    if actual["source_components"] != registration["source_components"]:
        raise ValueError("actual G0 child source roster differs from the original registration")
    from .initialization_evidence import observe_initialization_sources
    if observe_initialization_sources() != registration["initialization_sources"]["pins"]:
        raise ValueError("actual initializer source pins differ from original G0 registration")
    return actual


def _prepared_outputs(registration, *, until_ns, clock):
    initial, export = Path(registration["initialization_directory"]), Path(registration["export_directory"])
    outputs = {"checkpoint": _pin(initial / "untrained.pt", MAX_CHECKPOINT_BYTES, until_ns=until_ns, clock=clock),
        "checkpoint_metadata": _pin(initial / "checkpoint.json", MAX_METADATA_BYTES, until_ns=until_ns, clock=clock),
        "initialization_evidence": _pin(initial / EVIDENCE_NAME, MAX_EVIDENCE_BYTES, until_ns=until_ns, clock=clock),
        "export_manifest": _pin(export / "export.json", 256 * 1024, until_ns=until_ns, clock=clock),
        "export_graphs": [dict(_pin(export / file, MAX_GRAPH_BYTES, until_ns=until_ns, clock=clock), role=role, file=file)
                          for role, file in GRAPH_FILES.items()],
        "cuda_control_inventory": _pin(export / "cuda-control-inventory.json", 4 * 1024 ** 2, until_ns=until_ns, clock=clock)}
    return outputs


def _require_empty_output_directory(path):
    """The original parent retains each new empty directory descriptor."""
    path = _safe_absolute(str(path))
    if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
        raise FileExistsError("G0 requires original new empty output directories; preserve previous artifacts")
    return path


def _prepare(registration, execution_id, *, until_ns, clock, check_original, execution_observation):
    from . import artifacts, initialization_evidence, virtual_control_inventory
    initialize = _registered_function(initialization_evidence, "initialize_g0_with_evidence", "initialization_evidence.py")
    export = _registered_function(artifacts, "export_checkpoint", "artifacts.py")
    inventory = _registered_function(virtual_control_inventory, "produce_virtual_control_inventory", "virtual_control_inventory.py")
    initial = _require_empty_output_directory(registration["initialization_directory"])
    output = _require_empty_output_directory(registration["export_directory"])
    # These historical producer APIs do not expose a per-write guard. Reserve
    # their complete bounded checkpoint+graph envelopes and retain the original
    # parent's tree watcher; do not claim this is an exporter pre-write fence.
    reserved = MAX_CHECKPOINT_BYTES + len(GRAPH_FILES) * MAX_GRAPH_BYTES + MAX_EVIDENCE_BYTES + MAX_METADATA_BYTES
    _output_credit(registration, reserved_bytes=reserved)
    check_original()
    _json_pin(registration["inputs"]["case_plan_template"], 4 * 1024 ** 2, until_ns=until_ns, clock=clock)
    rules = registration["inputs"]["rules_profile"]
    _json_pin(rules, MAX_METADATA_BYTES, until_ns=until_ns, clock=clock)
    remaining = (until_ns - clock()) // 10 ** 9
    if remaining < 1:
        raise TimeoutError("G0 initializer has no complete original execution second remaining")
    execution_observation["initialization_dispatch_started"] = True
    observed = initialize(initial, registration["initialization_sources"], initialization_id=execution_id,
                          max_seconds=min(3600, remaining))
    execution_observation["actual_initialization_executed"] = True
    check_original()
    # The producer return remains an observation; native authority is held only
    # by the original parent child/process/file owner and is never reconstructed.
    metadata = observed["evidence"]["metadata_fields"]
    execution_observation["export_dispatch_started"] = True
    exported = export(initial / "untrained.pt", output, include_validator=True,
        layout="virtual_review_shared_pc_if_v1", rules_profile_json=rules["path"])
    execution_observation["actual_export_executed"] = True
    artifacts.validate_virtual_export_manifest(exported, metadata)
    check_original()
    manifest_pin = _pin(output / "export.json", 256 * 1024, until_ns=until_ns, clock=clock)
    remaining = (until_ns - clock()) // 10 ** 9
    if remaining < 1:
        raise TimeoutError("G0 control inventory has no complete original execution second remaining")
    produced = inventory(output / "export.json", manifest_pin["sha256"], output / "cuda-control-inventory.json",
        opt_in=True, max_seconds=min(30, remaining), until_ns=until_ns,
        output_guard=lambda additional: _inventory_credit(registration, additional, check_original))
    outputs = _prepared_outputs(registration, until_ns=until_ns, clock=clock)
    inventory_pin = {name: outputs["cuda_control_inventory"][name] for name in ("path", "sha256", "bytes")}
    inventory_value = _json_pin(inventory_pin, 4 * 1024 ** 2, until_ns=until_ns, clock=clock)
    if (produced["inventory_sha256"] != outputs["cuda_control_inventory"]["sha256"]
            or produced["inventory_bytes"] != outputs["cuda_control_inventory"]["bytes"]
            or inventory_value["manifest_sha256"] != outputs["export_manifest"]["sha256"]
            or inventory_value["scope"] != virtual_control_inventory.SCOPE
            or produced["actual_provider_placement"] != "not_observed" or produced["cpu_allowlist"] != "not_created"):
        raise ValueError("actual G0 control inventory differs from original serialized graph readback")
    check_original()
    return {"status": "prepared", "outputs": outputs, "initialization_observation": observed,
        "metadata_fields": metadata, "export_manifest_metadata": exported, "control_inventory_metadata": produced,
        "producer_output_guard": "conservative_phase_reservation_and_parent_tree_watch_only",
        "actual_initialization_executed": True, "actual_export_executed": True, "actual_reference_executed": False}


def _inventory_credit(registration, additional, check_original):
    check_original()
    if not uint(additional, MAX_GRAPH_BYTES) or _output_credit(registration) < additional:
        raise ValueError("G0 inventory lacks original managed pre-write byte credit")


def _validate_reference(registration, *, until_ns, clock):
    from .initialization_evidence import read_initialization_evidence, validate_checkpoint_metadata
    from .artifacts import validate_virtual_export_manifest
    from .numeric_capture import validate_case_plan
    inputs = registration["inputs"]
    observed = read_initialization_evidence(inputs["initialization_evidence"]["path"], join_files=True)
    evidence = observed["evidence"]
    if evidence["source"] != registration["initialization_sources"]:
        raise ValueError("reference evidence does not describe the original registered G0 initializer")
    metadata = _json_pin(inputs["checkpoint_metadata"], MAX_METADATA_BYTES, until_ns=until_ns, clock=clock)
    validate_checkpoint_metadata(metadata)
    if (metadata != evidence["metadata_fields"] or metadata["checkpoint_sha256"] != inputs["checkpoint"]["sha256"]
            or evidence["checkpoint"]["sha256"] != inputs["checkpoint"]["sha256"]
            or evidence["checkpoint"]["bytes"] != inputs["checkpoint"]["bytes"]):
        raise ValueError("reference G0 checkpoint/metadata/initializer byte joins differ")
    manifest = _json_pin(inputs["export_manifest"], 256 * 1024, until_ns=until_ns, clock=clock)
    validate_virtual_export_manifest(manifest, metadata)
    if manifest["validator_present"] is not True or manifest["roles"] != ["proposer", "critic", "validator"]:
        raise ValueError("G0 initial numerical acceptance requires a Collection P/C/V export")
    plan = _json_pin(inputs["case_plan"], 4 * 1024 ** 2, until_ns=until_ns, clock=clock)
    validate_case_plan(plan)
    bindings = plan["bindings"]
    expected_graphs = {pin["role"]: pin["sha256"] for pin in inputs["export_graphs"]}
    if (bindings["weights_sha256"] != inputs["checkpoint"]["sha256"]
            or bindings["export_manifest_sha256"] != inputs["export_manifest"]["sha256"]
            or bindings["graphs"] != expected_graphs
            or bindings["source_components"] != registration["source_components"]
            or {graph["role"]: graph["sha256"] for graph in manifest["graphs"]} != expected_graphs):
        raise ValueError("G0 case plan differs from the parent's actual original checkpoint/export/source pins")
    from .virtual_control_inventory import SCHEMA, SCOPE
    inventory = _json_pin(inputs["cuda_control_inventory"], 4 * 1024 ** 2, until_ns=until_ns, clock=clock)
    if inventory.get("schema") != SCHEMA or inventory.get("scope") != SCOPE or inventory.get("manifest_sha256") != inputs["export_manifest"]["sha256"]:
        raise ValueError("reference static inventory is not joined to the actual original G0 manifest")
    return plan, metadata


def _load_g0_reference(registration, metadata, *, until_ns, clock):
    """Load exact untrained mathematical bits; no optimizer/RNG restore API."""
    import torch
    from .config import ModelConfig
    from .model import PalsModel
    pin = registration["inputs"]["checkpoint"]
    _output_credit(registration, reserved_bytes=pin["bytes"])
    with _bounded_checkpoint_snapshot(pin, until_ns=until_ns, clock=clock) as (snapshot, size):
        if size != pin["bytes"]:
            raise ValueError("G0 reference snapshot differs from original checkpoint byte credit")
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    fields(payload, ("state_dict", "config", "seed", "schema", "training_steps", "trained"), "fresh G0 checkpoint")
    if (payload["config"] != metadata["config"] or payload["schema"] != metadata["schema"]
            or type(payload["seed"]) is not int or payload["seed"] != 23
            or type(payload["training_steps"]) is not int or payload["training_steps"] != 0
            or payload["trained"] is not False):
        raise ValueError("trained/diagnostic/legacy checkpoints cannot become a G0 reference")
    config = ModelConfig(**payload["config"])
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(23)
        model = PalsModel(config)
    expected, incoming, total = model.state_dict(), payload["state_dict"], 0
    # state_dict's historical OrderedDict is a valid mathematical mapping;
    # this does not loosen the surrounding exact checkpoint field schema.
    if not isinstance(incoming, dict) or set(incoming) != set(expected):
        raise ValueError("G0 reference parameter membership differs")
    for name, tensor in incoming.items():
        if clock() >= until_ns:
            raise TimeoutError("G0 reference checkpoint validation reached original deadline")
        if (not torch.is_tensor(tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32
                or tensor.shape != expected[name].shape or not torch.isfinite(tensor).all().item()):
            raise ValueError("G0 reference parameter shape/dtype/value differs: " + name)
        total += tensor.numel() * tensor.element_size()
        if total > MAX_PARAMETER_BYTES:
            raise ValueError("G0 reference mathematical parameter allocation cap exceeded")
    model.load_state_dict(incoming, strict=True)
    for name, actual in model.state_dict().items():
        if not torch.equal(actual.detach().contiguous().reshape(-1).view(torch.uint8), incoming[name].contiguous().reshape(-1).view(torch.uint8)):
            raise ValueError("G0 strict mathematical load changed original parameter bits: " + name)
    del expected, incoming, payload, tensor, actual
    return model.eval()


def _reference(registration, *, until_ns, clock, check_original, execution_observation):
    from . import numeric_capture
    capture = _registered_function(numeric_capture, "capture_reference", "numeric_capture.py")
    _require_empty_output_directory(registration["reference_output_root"])
    plan, metadata = _validate_reference(registration, until_ns=until_ns, clock=clock)
    model = _load_g0_reference(registration, metadata, until_ns=until_ns, clock=clock)
    before = _model_bits_digest(model, until_ns=until_ns, clock=clock)
    check_original()
    remaining = (until_ns - clock()) // 10 ** 9
    if remaining < 1:
        raise TimeoutError("G0 reference has no complete original execution second remaining")
    # capture_reference has its own 64MiB raw output envelope; its full reserve
    # is charged before dispatch in addition to the parent tree watcher.
    _output_credit(registration, reserved_bytes=numeric_capture.MAX_TOTAL_BYTES)
    execution_observation["reference_dispatch_started"] = True
    execution_observation["reference_neural_forwards_executed"] = None
    result = capture(model, metadata, registration["export_directory"], plan, registration["reference_output_root"],
        checkpoint_path=registration["inputs"]["checkpoint"]["path"],
        expected_reference_source_sha256=registration["source_components"]["model_reference"],
        max_seconds=min(3600, remaining), execute=True)
    execution_observation["reference_neural_forwards_executed"] = result["cases"]
    execution_observation["actual_reference_executed"] = True
    check_original()
    if _model_bits_digest(model, until_ns=until_ns, clock=clock) != before:
        raise ValueError("G0 CPU reference changed original frozen untrained model bits")
    return {"status": "captured", "capture_result": result, "model_state_sha256": before,
        "original_model_bits_preserved": True, "actual_initialization_executed": False,
        "actual_export_executed": False, "actual_reference_executed": True}


def _artifact_readbacks(registration, *, until_ns, clock):
    """Only expressly registered phase output roots, bounded files and entries."""
    roots = ("initialization_directory", "export_directory") if registration["phase"] == "prepare" else ("reference_output_root",)
    pins, total = [], 0
    for name in roots:
        root = Path(registration[name])
        if not root.exists():
            continue
        if not root.is_dir() or root.is_symlink():
            raise ValueError("G0 original output root changed identity")
        files = list(root.iterdir())
        if len(files) > 64:
            raise ValueError("G0 phase output entry cap exceeded")
        for path in sorted(files):
            maximum = MAX_CHECKPOINT_BYTES if path.name == "untrained.pt" else (MAX_GRAPH_BYTES if path.suffix == ".onnx" else 4 * 1024 ** 2)
            pin = _pin(path, maximum, until_ns=until_ns, clock=clock)
            pins.append(pin)
            total += pin["bytes"]
            if total > registration["max_output_bytes"]:
                raise ValueError("G0 original output readback byte cap exceeded")
    return pins


def main(argv=None, *, clock=time.monotonic_ns, stdin=None):
    import argparse
    parser = argparse.ArgumentParser(prog="g0-numerical-original-child")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--original-g0-child", action="store_true")
    parser.add_argument("--startup-sha256")
    parser.add_argument("--startup-bytes", type=int)
    parser.add_argument("--execution-until-ns", type=int)
    parser.add_argument("--whole-until-ns", type=int)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"schema": RESULT_DOMAIN, "status": "metadata_only",
            "actual_initialization_executed": False, "actual_export_executed": False,
            "actual_reference_executed": False, "optimizer_updates_executed": 0,
            "native_numeric_authority_issued": False, "native_game_authority_issued": False}, sort_keys=True))
        return 0
    if not args.original_g0_child or os.name != "posix" or not uint(args.startup_bytes, MAX_JSON_BYTES):
        raise ValueError("explicit original Linux G0 child and finite startup read credit required")
    stream = sys.stdin.buffer if stdin is None else stdin
    raw = stream.read(args.startup_bytes + 1)
    if type(raw) is not bytes or len(raw) != args.startup_bytes:
        raise ValueError("original G0 startup bytes/EOF differ from parent read credit")
    startup = strict_json(raw.decode("utf-8", "strict"))
    registration = validate_startup(startup, expected_raw_sha256=args.startup_sha256, raw_bytes=raw,
        execution_until_ns=args.execution_until_ns, whole_until_ns=args.whole_until_ns, now_ns=clock())
    cancelled, previous = [False], {}
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, lambda _s, _f: cancelled.__setitem__(0, True))
    result, interpreter, runtime_before, runtime_after, runtime = None, None, None, None, None
    exit_code, before_deadline = 0, False
    execution_observation = {"initialization_dispatch_started": False, "export_dispatch_started": False,
        "reference_dispatch_started": False, "actual_initialization_executed": False,
        "actual_export_executed": False, "actual_reference_executed": False,
        "reference_neural_forwards_executed": 0}
    def check_original():
        if cancelled[0] or clock() >= args.execution_until_ns:
            raise TimeoutError("original G0 window cancelled/expired; preserve partial output")
        _output_credit(registration)
    try:
        try:
            check_original()
            if Path(registration["receipt_path"]).exists():
                raise FileExistsError("G0 terminal receipt already exists; never overwrite")
            interpreter = _install_original_dependencies(startup, until_ns=args.execution_until_ns, clock=clock)
            runtime_before = _runtime_readbacks(startup, until_ns=args.execution_until_ns, clock=clock)
            _original_inputs(registration, until_ns=args.execution_until_ns, clock=clock)
            _source_readback(registration, until_ns=args.execution_until_ns, clock=clock)
            import torch
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            runtime = _torch_runtime_readback(torch, startup, until_ns=args.execution_until_ns, clock=clock)
            if registration["phase"] == "prepare":
                result = _prepare(registration, startup["execution_id"], until_ns=args.execution_until_ns,
                                  clock=clock, check_original=check_original, execution_observation=execution_observation)
            else:
                result = _reference(registration, until_ns=args.execution_until_ns, clock=clock,
                                    check_original=check_original, execution_observation=execution_observation)
            check_original()
            before_deadline = True
        except BaseException as error:
            result, exit_code = {"status": "failed", "error_type": type(error).__name__, "error": str(error)[:4096],
                "partial_output_retained_for_parent": True,
                "automatic_retry": False}, 1
        artifacts, inputs, active_source = [], {}, None
        try:
            artifacts = _artifact_readbacks(registration, until_ns=args.whole_until_ns, clock=clock)
            inputs = _original_inputs(registration, until_ns=args.whole_until_ns, clock=clock)
            active_source = _source_readback(registration, until_ns=args.whole_until_ns, clock=clock)
            runtime_after = _runtime_readbacks(startup, until_ns=args.whole_until_ns, clock=clock)
            _staging_bytes(registration)
        except BaseException as error:
            result.update(status="failed", readback_error_type=type(error).__name__, readback_error=str(error)[:4096])
            exit_code = 1
        success = result["status"] in ("prepared", "captured")
        result.update(execution_observation)
        result.update(schema=RESULT_DOMAIN, execution_id=startup["execution_id"], phase=registration["phase"],
            registration_sha256=startup["registration_sha256"], original_execution_until_ns=args.execution_until_ns,
            original_whole_until_ns=args.whole_until_ns, before_original_execution_deadline=before_deadline,
            artifact_pins=artifacts, original_input_readbacks=inputs, active_source=active_source,
            interpreter_readback=interpreter, runtime_module_readbacks_before=runtime_before,
            runtime_module_readbacks_after=runtime_after, runtime_readback=runtime,
            optimizer_updates_executed=0, neural_forwards_executed=(execution_observation["reference_neural_forwards_executed"]
                if execution_observation["reference_dispatch_started"] else 0),
            physical_completion="reference_completed_cpu" if success and registration["phase"] == "reference" else ("not_started" if success else "failed"),
            native_numeric_authority_issued=False, native_game_authority_issued=False,
            original_process_authority_issued=False, parent_must_verify_process_and_artifacts=True,
            native_status="not_run", cuda_status="not_run", arena_accepted=False)
        receipt = _terminal_receipt(registration, result)
        print(json.dumps({"schema": RESULT_DOMAIN, "status": result["status"], "phase": registration["phase"],
            "receipt": receipt, "actual_initialization_executed": result["actual_initialization_executed"],
            "actual_export_executed": result["actual_export_executed"], "actual_reference_executed": result["actual_reference_executed"],
            "optimizer_updates_executed": 0, "native_numeric_authority_issued": False,
            "before_original_execution_deadline": before_deadline}, sort_keys=True, allow_nan=False))
        return exit_code
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
