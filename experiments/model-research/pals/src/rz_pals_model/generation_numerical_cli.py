"""Explicit CPU reference child for a parent-held actual generation export.

The Rust parent keeps its original export owner. JSON pins describe readback;
they never reconstruct that owner, authorize a native numerical pass or restore
an optimizer. The ordinary command is metadata-only and does not import Torch.
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
    MAX_JSON_BYTES, MAX_WALL_NS, _active_source_readback, _artifact,
    _bounded_checkpoint_snapshot, _file_identifier, _install_original_dependencies,
    _output_credit, _pinned_bytes, _regular_readback, _safe_absolute,
    _staging_bytes, _terminal_receipt,
)

REGISTRATION_DOMAIN = "rz-pals-generation-numerical-registration/1"
STARTUP_DOMAIN = "rz-pals-original-generation-numerical-startup/1"
RESULT_DOMAIN = "rz-pals-generation-numerical-cli-window/1"
SOURCE_COMPONENTS = ("model_native", "model_encoding", "nn_runtime", "model_reference", "model_export")
KINDS = ("frozen_pc_collection", "g1_actual_generation")
MAX_CHECKPOINT_BYTES = 1024 ** 3
MAX_PARAMETER_BYTES = 128 * 1024 ** 2
MAX_OUTPUT_BYTES = 18 * 1024 ** 3
CHECKPOINT_FIELDS = (
    "schema", "eligibility", "spec", "spec_sha256", "config", "model",
    "optimizers", "schedulers", "membership", "parameter_updates",
    "frozen_pc_sha256", "ledger", "sampler", "bucket", "rng", "cuda_devices",
)


def validate_registration(value):
    """Pure strict metadata validation; no source or model is executed."""
    fields(value, (
        "schema", "numerical_id", "template_startup", "case_plan", "checkpoint",
        "export_manifest", "export_graphs", "source_components", "frozen_pc_sha256",
        "model_epoch", "generation_export_semantic", "staging_root",
        "reference_output_root", "receipt_path", "max_wall_ns", "max_output_bytes",
    ), "original generation numerical registration")
    if value["schema"] != REGISTRATION_DOMAIN:
        raise ValueError("registered original generation numerical domain required")
    _file_identifier(value["numerical_id"])
    for name, maximum in (
        ("template_startup", MAX_JSON_BYTES), ("case_plan", 4 * 1024 ** 2),
        ("checkpoint", MAX_CHECKPOINT_BYTES), ("export_manifest", 256 * 1024),
    ):
        _artifact(value[name], maximum)
    if Path(value["checkpoint"]["path"]).suffix != ".pt":
        raise ValueError("actual original generation .pt checkpoint required")
    fields(value["source_components"], SOURCE_COMPONENTS, "exact numerical source components")
    for pin in value["source_components"].values():
        sha(pin)
    sha(value["frozen_pc_sha256"])
    uint(value["model_epoch"])
    if not uint(value["max_wall_ns"], MAX_WALL_NS):
        raise ValueError("finite original numerical window required")
    if not uint(value["max_output_bytes"], MAX_OUTPUT_BYTES):
        raise ValueError("finite managed numerical output ceiling required")
    semantic = value["generation_export_semantic"]
    fields(semantic, (
        "kind", "config", "checkpoint_sha256", "trained", "training_steps",
        "validator_present", "model_epoch", "frozen_pc_sha256",
    ), "original export mathematical metadata readback")
    expected_steps = {"frozen_pc_collection": 4000, "g1_actual_generation": 5000}
    if (semantic["kind"] not in KINDS or type(semantic["config"]) is not dict
            or semantic["checkpoint_sha256"] != value["checkpoint"]["sha256"]
            or semantic["trained"] is not True
            or type(semantic["training_steps"]) is not int
            or semantic["training_steps"] != expected_steps.get(semantic["kind"])
            or semantic["validator_present"] is not (semantic["kind"] == "frozen_pc_collection")
            or semantic["model_epoch"] != value["model_epoch"]
            or semantic["frozen_pc_sha256"] != value["frozen_pc_sha256"]):
        raise ValueError("original export/checkpoint mathematical metadata differs")
    graph_roles = ["public", "shared_pc"] + (["validator"] if semantic["validator_present"] else [])
    graphs = value["export_graphs"]
    if type(graphs) is not list or [g.get("role") for g in graphs if type(g) is dict] != graph_roles:
        raise ValueError("exact original Fresh graph roster required")
    directory = Path(value["export_manifest"]["path"]).parent
    if Path(value["export_manifest"]["path"]).name != "export.json":
        raise ValueError("original export.json manifest required")
    for graph in graphs:
        fields(graph, ("role", "file", "path", "sha256", "bytes"), "original Fresh graph pin")
        _artifact({name: graph[name] for name in ("path", "sha256", "bytes")}, 2 * 1024 ** 3)
        expected_file = {"public": "public_memory.onnx", "shared_pc": "shared_pc_if.onnx", "validator": "role_validator.onnx"}[graph["role"]]
        if graph["file"] != expected_file or Path(graph["path"]) != directory / expected_file:
            raise ValueError("original graph path/role differs from export manifest directory")
    root = _safe_absolute(value["staging_root"])
    output = _safe_absolute(value["reference_output_root"])
    receipt = _safe_absolute(value["receipt_path"])
    if (output == root or not output.is_relative_to(root) or receipt == root
            or not receipt.is_relative_to(root) or receipt.suffix != ".json"
            or receipt == output or receipt.is_relative_to(output)):
        raise ValueError("numerical output and terminal receipt must have separate managed paths")
    source_root = Path(__file__).absolute().parents[5]
    if any(path == source_root or path.is_relative_to(source_root) for path in (root, output, receipt)):
        raise ValueError("numerical generated artifacts must remain outside the private source tree")
    input_paths = [value[name]["path"] for name in ("template_startup", "case_plan", "checkpoint", "export_manifest")]
    input_paths.extend(graph["path"] for graph in graphs)
    if len(set(input_paths)) != len(input_paths) or str(receipt) in input_paths:
        raise ValueError("numerical inputs/terminal output cannot alias one another")
    if any(Path(path) == output or Path(path).is_relative_to(output) for path in input_paths):
        raise ValueError("numerical output cannot contain an original pinned input")
    return copy.deepcopy(value)


def numerical_registration_digest(value):
    return digest(REGISTRATION_DOMAIN, validate_registration(value))


def validate_startup(value, *, expected_raw_sha256, raw_bytes,
                     execution_until_ns, whole_until_ns, now_ns):
    """Check original child transport without making an export capability."""
    fields(value, (
        "schema", "execution_id", "registration_sha256", "registration",
        "original_started_ns", "execution_until_ns", "whole_until_ns",
        "dependency_directories", "python_runtime", "runtime_module_pins",
    ), "original generation numerical startup")
    if value["schema"] != STARTUP_DOMAIN:
        raise ValueError("registered original generation numerical startup required")
    identifier(value["execution_id"])
    if (type(raw_bytes) is not bytes or not 0 < len(raw_bytes) <= MAX_JSON_BYTES
            or hashlib.sha256(raw_bytes).hexdigest() != sha(expected_raw_sha256)):
        raise ValueError("original numerical startup bytes differ from parent raw pin")
    registration = validate_registration(value["registration"])
    if numerical_registration_digest(registration) != sha(value["registration_sha256"]):
        raise ValueError("original numerical registration differs from startup")
    dependencies = value["dependency_directories"]
    if type(dependencies) is not list or not 1 <= len(dependencies) <= 8:
        raise ValueError("finite explicitly registered dependency directories required")
    paths, identities = set(), set()
    for entry in dependencies:
        fields(entry, ("path", "device", "inode"), "registered numerical dependency directory")
        path = str(_safe_absolute(entry["path"]))
        identity = (uint(entry["device"]), uint(entry["inode"]))
        if not identity[1] or path in paths or identity in identities:
            raise ValueError("aliased/duplicate numerical dependency directory")
        paths.add(path)
        identities.add(identity)
    runtime = value["python_runtime"]
    fields(runtime, ("version", "executable"), "registered numerical Python runtime")
    if type(runtime["version"]) is not str or re.fullmatch(r"3\.[0-9]{1,2}\.[0-9]{1,3}", runtime["version"]) is None:
        raise ValueError("registered numerical Python release required")
    _artifact(runtime["executable"], 256 * 1024 ** 2)
    module_pins = value["runtime_module_pins"]
    if type(module_pins) is not list or not 1 <= len(module_pins) <= 64:
        raise ValueError("finite explicit numerical runtime module pins required")
    module_paths = set()
    for pin in module_pins:
        _artifact(pin, 2 * 1024 ** 3)
        if pin["path"] in module_paths:
            raise ValueError("duplicate numerical runtime module path")
        module_paths.add(pin["path"])
    for number in (value["original_started_ns"], value["execution_until_ns"], value["whole_until_ns"], execution_until_ns, whole_until_ns, now_ns):
        uint(number)
    if (value["execution_until_ns"] != execution_until_ns or value["whole_until_ns"] != whole_until_ns
            or not value["original_started_ns"] <= now_ns < execution_until_ns <= whole_until_ns
            or execution_until_ns - value["original_started_ns"] > registration["max_wall_ns"]
            or whole_until_ns - execution_until_ns > 30 * 10 ** 9):
        raise ValueError("original numerical deadline mismatch/renewal/expiration")
    return registration


def _readback_pin(pin, *, until_ns, clock):
    observed = _regular_readback(pin["path"], maximum=pin["bytes"], until_ns=until_ns, clock=clock)
    if any(observed[name] != pin[name] for name in ("path", "sha256", "bytes")):
        raise ValueError("actual original numerical artifact differs from exact pin")
    return observed


def _runtime_readbacks(startup, *, until_ns, clock):
    """Actual file readback only; do not claim every native module was mapped."""
    return [_readback_pin(pin, until_ns=until_ns, clock=clock) for pin in startup["runtime_module_pins"]]


def _torch_runtime_readback(torch, startup, *, until_ns, clock):
    """Join the actual imported Torch file to the original runtime pin roster."""
    path = str(_safe_absolute(str(torch.__file__)))
    matches = [pin for pin in startup["runtime_module_pins"] if pin["path"] == path]
    if len(matches) != 1:
        raise ValueError("actual imported Torch module is not exactly registered")
    _readback_pin(matches[0], until_ns=until_ns, clock=clock)
    observed = {
        "torch": str(torch.__version__), "torch_module_path": path,
        "device": "cpu", "precision": "fp32",
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }
    if (observed["cuda_matmul_tf32"] or observed["cudnn_tf32"]
            or observed["cpu_threads"] != 2 or observed["interop_threads"] != 1):
        raise ValueError("actual CPU FP32 numerical runtime settings changed")
    return observed


def _load_original_cpu_model(registration, *, until_ns, clock=time.monotonic_ns):
    """Restore mathematical weights only, never optimizer/sampler/RNG state."""
    from .generation_training import CHECKPOINT_DOMAIN, ELIGIBILITY, GenerationLedger, spec_digest
    from .config import ModelConfig
    from .model import PalsModel
    import torch
    semantic = registration["generation_export_semantic"]
    _output_credit(registration, reserved_bytes=registration["checkpoint"]["bytes"])
    with _bounded_checkpoint_snapshot(registration["checkpoint"], until_ns=until_ns, clock=clock) as (snapshot, size):
        if size != registration["checkpoint"]["bytes"] or clock() >= until_ns:
            raise TimeoutError("original numerical checkpoint window expired")
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
        fields(payload, CHECKPOINT_FIELDS, "actual numerical generation checkpoint")
        if (payload["schema"] != CHECKPOINT_DOMAIN or payload["eligibility"] != ELIGIBILITY
                or payload["config"] != semantic["config"]
                or payload["spec_sha256"] != spec_digest(payload["spec"])
                or payload["frozen_pc_sha256"] != registration["frozen_pc_sha256"]):
            raise ValueError("actual checkpoint schema/config/freeze differs from original export")
        ledger = GenerationLedger(**copy.deepcopy(payload["ledger"]))
        expected = {"proposer": 2000, "critic": 2000, "validator": 0 if semantic["kind"] == "frozen_pc_collection" else 1000}
        expected_phase = "v" if semantic["kind"] == "frozen_pc_collection" else "complete"
        if (ledger.spec != payload["spec"] or not ledger.checkpoint_boundary()
                or ledger.completed != expected or ledger.phase != expected_phase
                or sum(ledger.completed.values()) != semantic["training_steps"]
                or ledger.pending_microbatches != 0 or payload["cuda_devices"] != [0]
                or type(payload["cuda_devices"][0]) is not int
                or payload["bucket"] not in (8, 16, 32, 64, 128, 256)):
            raise ValueError("actual checkpoint is not the original clean generation export boundary")
        config = ModelConfig(**payload["config"])
        config.validate()
        if config.profile != "virtual_review_v1" or not config.virtual_review:
            raise ValueError("registered compact V3 checkpoint required")
        # A private scratch initialization cannot consume the embedding caller's
        # RNG. Every initialized parameter is overwritten by exact original bits.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(23)
            model = PalsModel(config)
        expected_state = model.state_dict()
        state = payload["model"]
        if type(state) is not dict or set(state) != set(expected_state):
            raise ValueError("actual generation model membership differs")
        total = 0
        for name, tensor in state.items():
            if clock() >= until_ns:
                raise TimeoutError("original numerical model validation window expired")
            if (not torch.is_tensor(tensor) or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32 or tensor.shape != expected_state[name].shape
                    or not torch.isfinite(tensor).all().item()):
                raise ValueError("actual generation model shape/dtype/value differs: " + name)
            total += tensor.numel() * tensor.element_size()
            if total > MAX_PARAMETER_BYTES:
                raise ValueError("actual generation model allocation cap exceeded")
        model.load_state_dict(state, strict=True)
        for name, observed in model.state_dict().items():
            original = state[name]
            if not torch.equal(observed.detach().contiguous().reshape(-1).view(torch.uint8), original.contiguous().reshape(-1).view(torch.uint8)):
                raise ValueError("original numerical strict load changed model bits: " + name)
        # Release full checkpoint/AdamW decoded tensors before a second independent
        # checkpoint proof in capture_reference or the first neural forward.
        del expected_state, state, payload, tensor, original, observed
        if semantic["validator_present"] is False:
            model = model.virtual_review_pc_export_view()
        metadata = {"config": config.to_dict(), "checkpoint_sha256": registration["checkpoint"]["sha256"],
                    "trained": True, "training_steps": semantic["training_steps"]}
        return model.eval(), metadata


def _model_bits_digest(model, *, until_ns, clock):
    """Bounded tensor-at-a-time identity; no optimizer, forward or GPU query."""
    import torch
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if clock() >= until_ns:
            raise TimeoutError("original model bits proof exceeded numerical window")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode("utf-8")
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        result.update(len(header).to_bytes(8, "big"))
        result.update(header)
        result.update(len(raw).to_bytes(8, "big"))
        result.update(raw)
        del raw
    return result.hexdigest()


def _validate_plan_and_manifest(registration, *, until_ns, clock):
    from .numeric_capture import source_pins, validate_case_plan
    from .artifacts import validate_virtual_export_manifest
    template = _pinned_bytes(registration["template_startup"], maximum=MAX_JSON_BYTES, until_ns=until_ns, clock=clock)
    # The original ActionCollection template is byte readback only. A JSON pin
    # cannot reconstruct its native owner and is never dispatched by this child.
    strict_json(template.decode("utf-8", "strict"))
    plan = strict_json(_pinned_bytes(registration["case_plan"], maximum=4 * 1024 ** 2, until_ns=until_ns, clock=clock).decode("utf-8", "strict"))
    validate_case_plan(plan)
    manifest = strict_json(_pinned_bytes(registration["export_manifest"], maximum=256 * 1024, until_ns=until_ns, clock=clock).decode("utf-8", "strict"))
    semantic = registration["generation_export_semantic"]
    metadata = {name: semantic[name] for name in ("config", "checkpoint_sha256", "trained", "training_steps")}
    validate_virtual_export_manifest(manifest, metadata)
    bindings = plan["bindings"]
    expected_graphs = {graph["role"]: graph["sha256"] for graph in registration["export_graphs"]}
    if (manifest["validator_present"] is not semantic["validator_present"]
            or bindings["weights_sha256"] != registration["checkpoint"]["sha256"]
            or bindings["export_manifest_sha256"] != registration["export_manifest"]["sha256"]
            or bindings["model_epoch"] != registration["model_epoch"]
            or bindings["graphs"] != expected_graphs
            or bindings["source_components"] != registration["source_components"]
            or source_pins() != registration["source_components"]
            or {graph["role"]: graph["sha256"] for graph in manifest["graphs"]} != expected_graphs):
        raise ValueError("actual export/plan/source roster differs from original numerical registration")
    for graph in registration["export_graphs"]:
        _readback_pin(graph, until_ns=until_ns, clock=clock)
    return plan


def main(argv=None, *, clock=time.monotonic_ns, stdin=None):
    import argparse
    parser = argparse.ArgumentParser(prog="generation-numerical-original-child")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--original-numerical-child", action="store_true")
    parser.add_argument("--startup-sha256")
    parser.add_argument("--startup-bytes", type=int)
    parser.add_argument("--execution-until-ns", type=int)
    parser.add_argument("--whole-until-ns", type=int)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"schema": RESULT_DOMAIN, "status": "metadata_only", "actual_reference_executed": False,
            "optimizer_updates_executed": 0, "native_numeric_authority_issued": False}, sort_keys=True))
        return 0
    if not args.original_numerical_child or os.name != "posix" or not uint(args.startup_bytes, MAX_JSON_BYTES):
        raise ValueError("explicit original Linux numerical child + finite startup credit required")
    stream = sys.stdin.buffer if stdin is None else stdin
    raw = stream.read(args.startup_bytes + 1)
    if type(raw) is not bytes or len(raw) != args.startup_bytes:
        raise ValueError("original numerical startup bytes/EOF differ from parent credit")
    startup = strict_json(raw.decode("utf-8", "strict"))
    registration = validate_startup(startup, expected_raw_sha256=args.startup_sha256, raw_bytes=raw,
        execution_until_ns=args.execution_until_ns, whole_until_ns=args.whole_until_ns, now_ns=clock())
    cancelled, previous = [False], {}
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, lambda _s, _f: cancelled.__setitem__(0, True))
    result, capture_result, interpreter, runtime_before, model = None, None, None, None, None
    runtime_readback = None
    exit_code, capture_dispatched, before_deadline = 0, False, False
    def check_original():
        if cancelled[0] or clock() >= args.execution_until_ns:
            raise TimeoutError("original numerical window cancelled/expired; preserve partial evidence")
        _output_credit(registration)
    try:
        try:
            check_original()
            if Path(registration["reference_output_root"]).exists() or Path(registration["receipt_path"]).exists():
                raise FileExistsError("original numerical evidence paths already exist; never overwrite")
            interpreter = _install_original_dependencies(startup, until_ns=args.execution_until_ns, clock=clock)
            runtime_before = _runtime_readbacks(startup, until_ns=args.execution_until_ns, clock=clock)
            plan = _validate_plan_and_manifest(registration, until_ns=args.execution_until_ns, clock=clock)
            import torch
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            runtime_readback = _torch_runtime_readback(torch, startup, until_ns=args.execution_until_ns, clock=clock)
            model, metadata = _load_original_cpu_model(registration, until_ns=args.execution_until_ns, clock=clock)
            before_bits = _model_bits_digest(model, until_ns=args.execution_until_ns, clock=clock)
            check_original()
            remaining_seconds = (args.execution_until_ns - clock()) // 10 ** 9
            if remaining_seconds < 1:
                raise TimeoutError("original numerical window has no full capture second remaining")
            from .numeric_capture import capture_reference
            capture_dispatched = True
            capture_result = capture_reference(model, metadata, Path(registration["export_manifest"]["path"]).parent,
                plan, registration["reference_output_root"], checkpoint_path=registration["checkpoint"]["path"],
                expected_reference_source_sha256=registration["source_components"]["model_reference"],
                max_seconds=min(3600, remaining_seconds), execute=True)
            check_original()
            if _model_bits_digest(model, until_ns=args.execution_until_ns, clock=clock) != before_bits:
                raise ValueError("CPU reference capture changed original mathematical model bits")
            runtime_readback = _torch_runtime_readback(torch, startup, until_ns=args.execution_until_ns, clock=clock)
            before_deadline = clock() < args.execution_until_ns
            result = {"schema": RESULT_DOMAIN, "status": "captured", "actual_reference_executed": True,
                "model_state_sha256": before_bits, "original_model_bits_preserved": True}
        except BaseException as error:
            result, exit_code = {"schema": RESULT_DOMAIN, "status": "failed", "actual_reference_executed": False,
                "error_type": type(error).__name__, "error": str(error)[:4096], "partial_artifacts_preserved": True,
                "automatic_retry": False}, 1
        artifact_pins, inputs, active_source, runtime_after = [], {}, None, None
        try:
            from .numeric_capture import source_pins
            from .generation_export import _atomic_json
            active_source = _active_source_readback(until_ns=args.whole_until_ns, clock=clock)
            if not any(entry["module"] == "__main__" for entry in active_source["source_modules"]):
                active_source["source_modules"].append({"module": "__main__", **_regular_readback(__file__, maximum=4 * 1024 ** 2,
                    until_ns=args.whole_until_ns, clock=clock)})
            active_source["source_components"] = source_pins()
            if active_source["source_components"] != registration["source_components"]:
                raise ValueError("terminal numerical source roster differs from original pin")
            runtime_after = _runtime_readbacks(startup, until_ns=args.whole_until_ns, clock=clock)
            for name in ("template_startup", "case_plan", "checkpoint", "export_manifest"):
                inputs[name] = _readback_pin(registration[name], until_ns=args.whole_until_ns, clock=clock)
            inputs["export_graphs"] = [_readback_pin(pin, until_ns=args.whole_until_ns, clock=clock) for pin in registration["export_graphs"]]
            root = Path(registration["reference_output_root"])
            if root.exists():
                if not root.is_dir() or root.is_symlink():
                    raise ValueError("numerical output root identity changed")
                files = sorted(root.iterdir())
                if len(files) > 64:
                    raise ValueError("numerical output file cap exceeded")
                for path in files:
                    artifact_pins.append(_regular_readback(path, maximum=4 * 1024 ** 2,
                        until_ns=args.whole_until_ns, clock=clock))
            _staging_bytes(registration)
        except BaseException as error:
            result.update(status="failed", readback_error_type=type(error).__name__, readback_error=str(error)[:4096])
            exit_code = 1
        successful = result["status"] == "captured"
        result.update(execution_id=startup["execution_id"], registration_sha256=startup["registration_sha256"],
            original_execution_until_ns=args.execution_until_ns, original_whole_until_ns=args.whole_until_ns,
            active_source=active_source, numerical_source_roster=registration["source_components"],
            original_input_readbacks=inputs, artifact_pins=artifact_pins, capture_result=capture_result,
            interpreter_readback=interpreter, runtime_module_readbacks_before=runtime_before,
            runtime_module_readbacks_after=runtime_after, runtime_readback=runtime_readback,
            optimizer_updates_executed=0,
            reference_neural_forwards_executed=None if capture_result is None else capture_result["cases"],
            reference_dispatch_started=capture_dispatched,
            physical_completion="reference_completed_cpu" if successful else "failed",
            native_numeric_authority_issued=False, parent_must_verify_process_and_artifacts=True,
            native_status="not_run", cuda_status="not_run", arena_accepted=False,
            before_original_execution_deadline=before_deadline,
            actual_reference_executed=successful)
        receipt = _terminal_receipt(registration, result)
        print(json.dumps({"schema": RESULT_DOMAIN, "status": result["status"], "receipt": receipt,
            "actual_reference_executed": successful, "before_original_execution_deadline": before_deadline},
            sort_keys=True, allow_nan=False))
        return exit_code
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
