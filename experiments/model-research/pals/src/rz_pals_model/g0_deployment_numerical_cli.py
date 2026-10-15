"""Original V-free G0 Deployment export/reference child; no authority issuer.

The Rust parent must retain the live Collection owner and its original G0
checkpoint descriptor. This consumer reads those exact bytes, creates separate
P/C graphs and independent reference captures, and reports observations only.
It never initializes a G0, runs an optimizer, executes ORT, or converts stored
Collection observations into Deployment numerical/arena authority.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tempfile
import time

from .generation_data import digest, fields, identifier, sha, strict_json, uint
from .generation_training_cli import (
    MAX_JSON_BYTES, MAX_WALL_NS, _artifact, _file_identifier,
    _install_original_dependencies, _pinned_bytes,
    _regular_readback, _safe_absolute,
)
from .g0_cuda_artifact_tree import (
    output_credit as _output_credit, staging_bytes as _staging_bytes,
    terminal_receipt as _terminal_receipt, validate_prepared_cuda_tree,
)
from .generation_numerical_cli import (
    _model_bits_digest, _readback_pin, _runtime_readbacks, _torch_runtime_readback,
)
from .initialization_evidence import (
    EVIDENCE_NAME, MAX_CHECKPOINT_BYTES, MAX_EVIDENCE_BYTES, MAX_METADATA_BYTES,
    _registered_function, validate_source_registration,
)

REGISTRATION_DOMAIN = "rz-pals-g0-deployment-registration/1"
STARTUP_DOMAIN = "rz-pals-g0-deployment-original-startup/1"
RESULT_DOMAIN = "rz-pals-g0-deployment-original-result/1"
SOURCE_COMPONENTS = ("model_native", "model_encoding", "nn_runtime", "model_reference", "model_export")
PHASES = ("export", "reference")
MAX_OUTPUT_BYTES = 18 * 1024 ** 3
MAX_GRAPH_BYTES = 512 * 1024 ** 2
MAX_PARAMETER_BYTES = 128 * 1024 ** 2
GRAPH_FILES = {"public": "public_memory.onnx", "shared_pc": "shared_pc_if.onnx"}
EXPORT_INPUTS = ("checkpoint", "checkpoint_metadata", "initialization_evidence", "rules_profile")
REFERENCE_INPUTS = ("case_plan", "checkpoint", "checkpoint_metadata", "initialization_evidence",
                    "export_manifest", "export_graphs", "cuda_control_inventory")


def validate_registration(value):
    """Pure exact declarations; JSON cannot reconstruct the Collection owner."""
    optional = ("prepared_cuda_tree",) if type(value) is dict and "prepared_cuda_tree" in value else ()
    fields(value, ("schema", "numerical_id", "phase", "source_components", "source_commit",
        "initialization_sources", "staging_root", "initialization_directory", "export_directory",
        "reference_output_root", "receipt_path", "max_wall_ns", "max_output_bytes", "inputs", *optional),
        "original G0 Deployment registration")
    if value["schema"] != REGISTRATION_DOMAIN or value["phase"] not in PHASES:
        raise ValueError("separate original Deployment export/reference domain required")
    _file_identifier(value["numerical_id"])
    if type(value["source_commit"]) is not str or re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]) is None:
        raise ValueError("full registered Deployment source commit required")
    fields(value["source_components"], SOURCE_COMPONENTS, "exact Deployment numerical source components")
    for pin in value["source_components"].values():
        sha(pin)
    validate_source_registration(value["initialization_sources"])
    if value["initialization_sources"]["source_commit"] != value["source_commit"]:
        raise ValueError("original initializer and Deployment source commits differ")
    if (not uint(value["max_wall_ns"], MAX_WALL_NS) or not uint(value["max_output_bytes"], MAX_OUTPUT_BYTES)
            or value["max_output_bytes"] <= MAX_JSON_BYTES):
        raise ValueError("finite original Deployment time/output ceilings required")
    root = _safe_absolute(value["staging_root"])
    if optional:
        validate_prepared_cuda_tree(value["prepared_cuda_tree"], root)
    initial = _safe_absolute(value["initialization_directory"])
    outputs = [_safe_absolute(value[name]) for name in ("export_directory", "reference_output_root", "receipt_path")]
    if any(path == root or not path.is_relative_to(root) for path in outputs):
        raise ValueError("new Deployment outputs must stay under original managed staging")
    if any(left == right or left.is_relative_to(right) or right.is_relative_to(left)
           for i, left in enumerate([initial, *outputs]) for right in [initial, *outputs][i + 1:]):
        raise ValueError("original initialization and new Deployment namespaces must be disjoint")
    if initial == root or root.is_relative_to(initial) or outputs[-1].suffix != ".json":
        raise ValueError("separate original initialization and JSON terminal paths required")
    source = Path(__file__).absolute().parents[5]
    if root == source or root.is_relative_to(source) or initial == source or initial.is_relative_to(source):
        raise ValueError("Deployment staging and original weights must remain outside source")
    inputs = value["inputs"]
    fields(inputs, EXPORT_INPUTS if value["phase"] == "export" else REFERENCE_INPUTS,
           "phase-specific original Deployment inputs")
    expected_initial = {"checkpoint": initial / "untrained.pt", "checkpoint_metadata": initial / "checkpoint.json",
                        "initialization_evidence": initial / EVIDENCE_NAME}
    for name, expected in expected_initial.items():
        _artifact(inputs[name], {"checkpoint": MAX_CHECKPOINT_BYTES, "checkpoint_metadata": MAX_METADATA_BYTES,
                                "initialization_evidence": MAX_EVIDENCE_BYTES}[name])
        if Path(inputs[name]["path"]) != expected:
            raise ValueError("Deployment must read the original Collection initializer's exact file roster")
    if value["phase"] == "export":
        _artifact(inputs["rules_profile"], MAX_METADATA_BYTES)
        pins = list(inputs.values())
    else:
        for name, maximum in (("case_plan", 4 * 1024 ** 2), ("export_manifest", 256 * 1024),
                              ("cuda_control_inventory", 4 * 1024 ** 2)):
            _artifact(inputs[name], maximum)
        for name, filename in (("export_manifest", "export.json"), ("cuda_control_inventory", "cuda-control-inventory.json")):
            if Path(inputs[name]["path"]) != outputs[0] / filename:
                raise ValueError("Deployment reference must use its own original export outputs")
        graphs = inputs["export_graphs"]
        if type(graphs) is not list or [g.get("role") for g in graphs if type(g) is dict] != list(GRAPH_FILES):
            raise ValueError("Deployment requires exactly V-free public/shared P-C Fresh graphs")
        for graph in graphs:
            fields(graph, ("role", "file", "path", "sha256", "bytes"), "original Deployment graph pin")
            _artifact({key: graph[key] for key in ("path", "sha256", "bytes")}, MAX_GRAPH_BYTES)
            if graph["file"] != GRAPH_FILES[graph["role"]] or Path(graph["path"]) != outputs[0] / graph["file"]:
                raise ValueError("Deployment graph path/name/role mismatch")
        pins = [inputs[name] for name in REFERENCE_INPUTS if name != "export_graphs"] + graphs
    paths = [pin["path"] for pin in pins]
    if len(set(paths)) != len(paths) or value["receipt_path"] in paths:
        raise ValueError("original Deployment input/terminal paths cannot alias")
    generated = outputs[:2] if value["phase"] == "export" else outputs[1:2]
    if any(Path(path) == target or Path(path).is_relative_to(target) for path in paths for target in generated):
        raise ValueError("new Deployment output cannot contain an original input")
    return copy.deepcopy(value)


def registration_digest(value):
    return digest(REGISTRATION_DOMAIN, validate_registration(value))


def validate_startup(value, *, expected_raw_sha256, raw_bytes, execution_until_ns, whole_until_ns, now_ns):
    fields(value, ("schema", "execution_id", "phase", "registration_sha256", "registration",
        "original_started_ns", "execution_until_ns", "whole_until_ns", "dependency_directories",
        "python_runtime", "runtime_module_pins"), "original Deployment startup")
    if value["schema"] != STARTUP_DOMAIN:
        raise ValueError("separate original Deployment startup required")
    identifier(value["execution_id"])
    if (type(raw_bytes) is not bytes or not 0 < len(raw_bytes) <= MAX_JSON_BYTES
            or hashlib.sha256(raw_bytes).hexdigest() != sha(expected_raw_sha256)):
        raise ValueError("Deployment startup differs from original parent raw bytes")
    registration = validate_registration(value["registration"])
    if value["phase"] != registration["phase"] or registration_digest(registration) != sha(value["registration_sha256"]):
        raise ValueError("Deployment startup phase/registration differs")
    dependencies = value["dependency_directories"]
    if type(dependencies) is not list or not 1 <= len(dependencies) <= 8:
        raise ValueError("finite explicit Deployment dependency directories required")
    paths, identities = set(), set()
    for entry in dependencies:
        fields(entry, ("path", "device", "inode"), "original Deployment dependency directory")
        path = str(_safe_absolute(entry["path"])); identity = (uint(entry["device"]), uint(entry["inode"]))
        if not identity[1] or path in paths or identity in identities:
            raise ValueError("duplicate/aliased Deployment dependency directory")
        paths.add(path); identities.add(identity)
    runtime = value["python_runtime"]
    fields(runtime, ("version", "executable"), "registered Deployment Python runtime")
    if type(runtime["version"]) is not str or re.fullmatch(r"3\.[0-9]{1,2}\.[0-9]{1,3}", runtime["version"]) is None:
        raise ValueError("registered Deployment Python release required")
    _artifact(runtime["executable"], 256 * 1024 ** 2)
    modules = value["runtime_module_pins"]
    if type(modules) is not list or not 1 <= len(modules) <= 64:
        raise ValueError("finite explicit Deployment import artifact roster required")
    module_paths = set()
    for pin in modules:
        _artifact(pin, 2 * 1024 ** 3)
        if pin["path"] in module_paths:
            raise ValueError("duplicate Deployment runtime artifact")
        module_paths.add(pin["path"])
    for number in (value["original_started_ns"], value["execution_until_ns"], value["whole_until_ns"],
                   execution_until_ns, whole_until_ns, now_ns):
        uint(number)
    if (value["execution_until_ns"] != execution_until_ns or value["whole_until_ns"] != whole_until_ns
            or not value["original_started_ns"] <= now_ns < execution_until_ns <= whole_until_ns
            or execution_until_ns - value["original_started_ns"] > registration["max_wall_ns"]
            or whole_until_ns - execution_until_ns > 30 * 10 ** 9):
        raise ValueError("Deployment original deadline renewal/mismatch/expiration")
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
    from .initialization_evidence import observe_initialization_sources
    package = Path(__file__).absolute().parent
    selected = sorted((name, module) for name, module in sys.modules.items()
                      if name == "rz_pals_model" or name.startswith("rz_pals_model."))
    if not 1 <= len(selected) <= 127:
        raise ValueError("Deployment active source module cap exceeded")
    modules = []
    for name, module in selected:
        path = _safe_absolute(getattr(module, "__file__", ""))
        if not path.is_relative_to(package) or path.suffix != ".py":
            raise ValueError("Deployment imported source escaped original private package")
        modules.append({"module": name, **_pin(path, 4 * 1024 ** 2, until_ns=until_ns, clock=clock)})
    modules.append({"module": "__main__", **_pin(__file__, 4 * 1024 ** 2, until_ns=until_ns, clock=clock)})
    actual = {"source_components": source_pins(), "source_modules": modules}
    if actual["source_components"] != registration["source_components"]:
        raise ValueError("Deployment child source differs from original parent registration")
    if observe_initialization_sources() != registration["initialization_sources"]["pins"]:
        raise ValueError("original G0 initializer source pins differ")
    return actual


def _original_metadata(registration, *, until_ns, clock):
    from .initialization_evidence import read_initialization_evidence, validate_checkpoint_metadata
    inputs = registration["inputs"]
    pinned_evidence = _json_pin(inputs["initialization_evidence"], MAX_EVIDENCE_BYTES, until_ns=until_ns, clock=clock)
    evidence = read_initialization_evidence(inputs["initialization_evidence"]["path"], join_files=True)["evidence"]
    metadata = _json_pin(inputs["checkpoint_metadata"], MAX_METADATA_BYTES, until_ns=until_ns, clock=clock)
    validate_checkpoint_metadata(metadata)
    if (evidence != pinned_evidence or evidence["source"] != registration["initialization_sources"] or evidence["metadata_fields"] != metadata
            or metadata["checkpoint_sha256"] != inputs["checkpoint"]["sha256"]
            or any(evidence["checkpoint"][key] != inputs["checkpoint"][key] for key in ("sha256", "bytes"))):
        raise ValueError("Deployment must preserve original G0 checkpoint/metadata/initializer joins")
    return metadata


@contextmanager
def _original_checkpoint_snapshot(registration, *, until_ns, clock):
    """Hash/copy one original FD; temporary bytes stay in charged staging."""
    from .generation_export import _hash_file
    pin = registration["inputs"]["checkpoint"]
    source = _safe_absolute(pin["path"])
    staging = _safe_absolute(registration["staging_root"])
    for path in (source, *source.parents, staging, *staging.parents):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("original Deployment checkpoint has link/reparse ancestor")
    _output_credit(registration, reserved_bytes=pin["bytes"])
    with tempfile.TemporaryFile(mode="w+b", prefix="rz-g0-deployment-pinned-", dir=registration["staging_root"]) as snapshot:
        class OriginalCreditWriter:
            def write(self, block):
                if clock() >= until_ns or snapshot.tell() + len(block) > pin["bytes"]:
                    raise ValueError("Deployment original snapshot exceeded deadline/byte credit")
                return snapshot.write(block)
        observed, size = _hash_file(source, pin["bytes"], snapshot=OriginalCreditWriter())
        if observed != pin["sha256"] or size != pin["bytes"]:
            raise ValueError("Deployment snapshot differs from original checkpoint")
        snapshot.flush(); snapshot.seek(0)
        yield snapshot


def _load_original_g0_model(registration, metadata, *, until_ns, clock):
    """Existing strict G0 mathematical load, without initializer or optimizer."""
    import torch
    from .config import ModelConfig
    from .model import PalsModel
    with _original_checkpoint_snapshot(registration, until_ns=until_ns, clock=clock) as snapshot:
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    validate_original_checkpoint_header(payload, metadata)
    config = ModelConfig(**payload["config"]); config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(23)
        model = PalsModel(config)
    expected, incoming, total = model.state_dict(), payload["state_dict"], 0
    if not isinstance(incoming, dict) or set(incoming) != set(expected):
        raise ValueError("original G0 Deployment parameter membership differs")
    for name, tensor in incoming.items():
        if clock() >= until_ns:
            raise TimeoutError("Deployment mathematical load reached original deadline")
        if (not torch.is_tensor(tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32
                or tensor.shape != expected[name].shape or not torch.isfinite(tensor).all().item()):
            raise ValueError("original G0 Deployment parameter shape/dtype/value differs: " + name)
        total += tensor.numel() * tensor.element_size()
        if total > MAX_PARAMETER_BYTES:
            raise ValueError("original G0 mathematical allocation cap exceeded")
    model.load_state_dict(incoming, strict=True)
    for name, actual in model.state_dict().items():
        if not torch.equal(actual.detach().contiguous().reshape(-1).view(torch.uint8), incoming[name].contiguous().reshape(-1).view(torch.uint8)):
            raise ValueError("Deployment mathematical load changed original G0 bits: " + name)
    del expected, incoming, payload, tensor, actual
    return model.eval()


def validate_original_checkpoint_header(payload, metadata):
    """Pure G0 header check; no model, optimizer or eligibility conversion."""
    fields(payload, ("state_dict", "config", "seed", "schema", "training_steps", "trained"), "original G0 checkpoint")
    if (payload["config"] != metadata["config"] or payload["schema"] != metadata["schema"]
            or type(payload["seed"]) is not int or payload["seed"] != 23
            or type(payload["training_steps"]) is not int or payload["training_steps"] != 0
            or payload["trained"] is not False):
        raise ValueError("trained/diagnostic/legacy checkpoint cannot become G0 Deployment")
    return payload


def _require_empty_output_directory(path):
    path = _safe_absolute(str(path))
    if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
        raise FileExistsError("Deployment requires new empty outputs; preserve prior artifacts")
    return path


def _validate_vfree_manifest(manifest, metadata):
    from .artifacts import validate_virtual_export_manifest
    validate_virtual_export_manifest(manifest, metadata)
    return validate_vfree_manifest_roles(manifest)


def validate_vfree_manifest_roles(manifest):
    """Pure extra Deployment restriction on an existing export declaration."""
    if (manifest["roles"] != ["proposer", "critic"] or manifest["validator_present"] is not False
            or [graph["role"] for graph in manifest["graphs"]] != list(GRAPH_FILES)
            or set(manifest.get("initializer_banks", {})) != set(GRAPH_FILES)):
        raise ValueError("Deployment requires independently exported P/C graphs without any validator bank")
    return manifest


def _export_outputs(registration, *, until_ns, clock):
    root = Path(registration["export_directory"])
    # The original Rust parent joins each output to its retained file identity.
    return {"export_manifest": _pin(root / "export.json", 256 * 1024, until_ns=until_ns, clock=clock),
            "export_graphs": [dict(_pin(root / filename, MAX_GRAPH_BYTES, until_ns=until_ns, clock=clock), role=role, file=filename)
                              for role, filename in GRAPH_FILES.items()],
            "cuda_control_inventory": _pin(root / "cuda-control-inventory.json", 4 * 1024 ** 2, until_ns=until_ns, clock=clock)}


def _export(registration, *, until_ns, clock, check_original, execution_observation):
    from . import artifacts, virtual_control_inventory
    export = _registered_function(artifacts, "export_virtual_checkpoint", "artifacts.py")
    inventory = _registered_function(virtual_control_inventory, "produce_virtual_control_inventory", "virtual_control_inventory.py")
    output = _require_empty_output_directory(registration["export_directory"])
    metadata = _original_metadata(registration, until_ns=until_ns, clock=clock)
    model = _load_original_g0_model(registration, metadata, until_ns=until_ns, clock=clock)
    before = _model_bits_digest(model, until_ns=until_ns, clock=clock)
    _json_pin(registration["inputs"]["rules_profile"], MAX_METADATA_BYTES, until_ns=until_ns, clock=clock)
    check_original()
    execution_observation["export_dispatch_started"] = True
    exported = export(model, metadata, output, include_validator=False,
        rules_profile_json=registration["inputs"]["rules_profile"]["path"],
        output_guard=lambda amount: _write_credit(registration, amount, check_original))
    execution_observation["actual_export_executed"] = True
    _validate_vfree_manifest(exported, metadata)
    check_original()
    manifest_pin = _pin(output / "export.json", 256 * 1024, until_ns=until_ns, clock=clock)
    remaining = (until_ns - clock()) // 10 ** 9
    if remaining < 1:
        raise TimeoutError("Deployment inventory has no complete original second remaining")
    produced = inventory(output / "export.json", manifest_pin["sha256"], output / "cuda-control-inventory.json",
        opt_in=True, max_seconds=min(30, remaining), until_ns=until_ns,
        output_guard=lambda amount: _write_credit(registration, amount, check_original))
    outputs = _export_outputs(registration, until_ns=until_ns, clock=clock)
    # JSON consumes a strict artifact pin; retain device/inode in parent outputs.
    control_pin = {name: outputs["cuda_control_inventory"][name] for name in ("path", "sha256", "bytes")}
    control = _json_pin(control_pin, 4 * 1024 ** 2, until_ns=until_ns, clock=clock)
    if (produced["inventory_sha256"] != outputs["cuda_control_inventory"]["sha256"]
            or produced["inventory_bytes"] != outputs["cuda_control_inventory"]["bytes"]
            or control.get("manifest_sha256") != outputs["export_manifest"]["sha256"]
            or control.get("scope") != virtual_control_inventory.SCOPE
            or produced["actual_provider_placement"] != "not_observed" or produced["cpu_allowlist"] != "not_created"):
        raise ValueError("Deployment static inventory differs from its original exported graphs")
    check_original()
    if _model_bits_digest(model, until_ns=until_ns, clock=clock) != before:
        raise ValueError("Deployment export changed original G0 model bits")
    return {"status": "exported", "outputs": outputs, "metadata_fields": metadata,
            "export_manifest_metadata": exported, "control_inventory_metadata": produced,
            "original_model_bits_preserved": True, "model_state_sha256": before}


def _write_credit(registration, additional, check_original):
    check_original()
    if not uint(additional, MAX_GRAPH_BYTES) or _output_credit(registration) < additional:
        raise ValueError("Deployment lacks original managed pre-write byte credit")


def _validate_reference(registration, *, until_ns, clock):
    from .virtual_control_inventory import SCHEMA, SCOPE
    inputs = registration["inputs"]
    metadata = _original_metadata(registration, until_ns=until_ns, clock=clock)
    manifest = _json_pin(inputs["export_manifest"], 256 * 1024, until_ns=until_ns, clock=clock)
    _validate_vfree_manifest(manifest, metadata)
    plan = _json_pin(inputs["case_plan"], 4 * 1024 ** 2, until_ns=until_ns, clock=clock)
    validate_reference_plan(plan, inputs, manifest, registration["source_components"])
    control = _json_pin(inputs["cuda_control_inventory"], 4 * 1024 ** 2, until_ns=until_ns, clock=clock)
    if control.get("schema") != SCHEMA or control.get("scope") != SCOPE or control.get("manifest_sha256") != inputs["export_manifest"]["sha256"]:
        raise ValueError("Deployment reference inventory is not joined to its own original export")
    return plan, metadata


def validate_reference_plan(plan, inputs, manifest, source_components):
    """Pure complete P/C plan/binding check; no capture or passed JSON."""
    from .numeric_capture import make_case_plan, validate_case_plan
    cases = validate_case_plan(plan)
    graphs = {pin["role"]: pin["sha256"] for pin in inputs["export_graphs"]}
    bindings = plan["bindings"]
    if (bindings["weights_sha256"] != inputs["checkpoint"]["sha256"]
            or bindings["export_manifest_sha256"] != inputs["export_manifest"]["sha256"]
            or bindings["graphs"] != graphs or set(graphs) != set(GRAPH_FILES)
            or bindings["source_components"] != source_components
            or {graph["role"]: graph["sha256"] for graph in manifest["graphs"]} != graphs
            or plan != make_case_plan(bindings) or len(plan["groups"]) != 12 or cases != 100):
        raise ValueError("Deployment requires its independently bound complete P/C 12-group/100-case plan")
    return cases


def _reference(registration, *, until_ns, clock, check_original, execution_observation):
    from . import numeric_capture
    capture = _registered_function(numeric_capture, "capture_reference", "numeric_capture.py")
    _require_empty_output_directory(registration["reference_output_root"])
    plan, metadata = _validate_reference(registration, until_ns=until_ns, clock=clock)
    owner = _load_original_g0_model(registration, metadata, until_ns=until_ns, clock=clock)
    model = owner.virtual_review_pc_export_view()
    if set(model.experts) != {"proposer", "critic"}:
        raise ValueError("Deployment reference must expose P/C only")
    before = _model_bits_digest(model, until_ns=until_ns, clock=clock)
    check_original()
    remaining = (until_ns - clock()) // 10 ** 9
    if remaining < 1:
        raise TimeoutError("Deployment reference has no complete original second remaining")
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
    if result["cases"] != 100 or _model_bits_digest(model, until_ns=until_ns, clock=clock) != before:
        raise ValueError("Deployment reference incomplete or original P/C bits changed")
    return {"status": "captured", "capture_result": result, "model_state_sha256": before,
            "original_model_bits_preserved": True}


def _artifact_readbacks(registration, *, until_ns, clock):
    name = "export_directory" if registration["phase"] == "export" else "reference_output_root"
    root = Path(registration[name])
    if not root.exists():
        return []
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Deployment output directory identity changed")
    paths = list(root.iterdir())
    if len(paths) > 64:
        raise ValueError("Deployment output file count cap exceeded")
    pins, total = [], 0
    for path in sorted(paths):
        pin = _pin(path, MAX_GRAPH_BYTES if path.suffix == ".onnx" else 4 * 1024 ** 2, until_ns=until_ns, clock=clock)
        pins.append(pin); total += pin["bytes"]
        if total > registration["max_output_bytes"]:
            raise ValueError("Deployment output readback byte cap exceeded")
    return pins


def main(argv=None, *, clock=time.monotonic_ns, stdin=None):
    import argparse
    parser = argparse.ArgumentParser(prog="g0-deployment-original-child")
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
        raise ValueError("explicit original Linux Deployment child + finite startup credit required")
    stream = sys.stdin.buffer if stdin is None else stdin
    raw = stream.read(args.startup_bytes + 1)
    if type(raw) is not bytes or len(raw) != args.startup_bytes:
        raise ValueError("Deployment startup bytes/EOF differ from original parent credit")
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
            raise TimeoutError("original Deployment window cancelled/expired; preserve partial output")
        _output_credit(registration)
    try:
        try:
            check_original()
            if Path(registration["receipt_path"]).exists():
                raise FileExistsError("Deployment terminal already exists; never overwrite")
            interpreter = _install_original_dependencies(startup, until_ns=args.execution_until_ns, clock=clock)
            runtime_before = _runtime_readbacks(startup, until_ns=args.execution_until_ns, clock=clock)
            _original_inputs(registration, until_ns=args.execution_until_ns, clock=clock)
            _source_readback(registration, until_ns=args.execution_until_ns, clock=clock)
            import torch
            torch.set_num_threads(2); torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            runtime = _torch_runtime_readback(torch, startup, until_ns=args.execution_until_ns, clock=clock)
            operation = _export if registration["phase"] == "export" else _reference
            result = operation(registration, until_ns=args.execution_until_ns, clock=clock,
                               check_original=check_original, execution_observation=execution_observation)
            check_original(); before_deadline = True
        except BaseException as error:
            result, exit_code = {"status": "failed", "error_type": type(error).__name__, "error": str(error)[:4096],
                                "partial_output_retained_for_parent": True, "automatic_retry": False}, 1
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
        success = result["status"] in ("exported", "captured")
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
            "receipt": receipt, "actual_initialization_executed": False,
            "actual_export_executed": result["actual_export_executed"], "actual_reference_executed": result["actual_reference_executed"],
            "optimizer_updates_executed": 0, "native_numeric_authority_issued": False,
            "before_original_execution_deadline": before_deadline}, sort_keys=True, allow_nan=False))
        return exit_code
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
