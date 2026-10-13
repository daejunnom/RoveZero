"""Explicit original-checkpoint export child; no optimizer update or NN forward.

The launching Rust parent retains its opaque verified original trainer and
original game/action archives. JSON describes independently pinned inputs;
it never reconstructs that parent or issues neural/arena acceptance. The
registered private child reuses the actual model/checkpoint/export owners.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

from .generation_data import digest, fields, sha, strict_json, uint
from .generation_training import (CHECKPOINT_DOMAIN, GenerationOptimizers,
    configure_generation_runtime, restore_checkpoint, validate_checkpoint,
    generation_admission, parameter_owner, spec_digest, ELIGIBILITY)
from .generation_training_cli import (MAX_JSON_BYTES, MAX_WALL_NS, SOURCE_COMPONENTS,
    PinnedOriginalTrainerChild, _active_source_readback, _artifact, _file_identifier,
    _install_original_dependencies, _output_credit, _pinned_bytes, _regular_readback,
    _safe_absolute, _sampler_validator, _staging_bytes, _terminal_receipt,
    export_frozen_pc_collection, prepare_registration, registration_digest,
    validate_registration, validate_startup, STARTUP_DOMAIN)

REGISTRATION_DOMAIN = "rz-pals-generation-export-registration/1"
STARTUP = "rz-pals-original-generation-export-startup/1"
RESULT = "rz-pals-generation-export-cli-window/1"
EXPORT_COMPONENTS = SOURCE_COMPONENTS + ("model_export",)
KINDS = ("frozen_pc_collection", "g1_actual_generation")
FROZEN_V2 = "rz-pals-frozen-pc-collection-export/2"
FROZEN_CHECKPOINT = "rz-pals-verified-frozen-pc-checkpoint/1"
CONTROL_SEAL = "rz-pals-actual-export-control-inventory/1"


def _frozen_collection_v2(original, live, prepared, registration, *, output_guard=None):
    """Actual checked4000 checkpoint/admission bridge, separate from V1 receipt."""
    from .nonzero_training import _state_digest
    from .generation_export import _atomic_json
    payload, ledger = live.checkpoint_payload, live.ledger
    actual_owners = {parameter_owner(name) for name, count in payload["parameter_updates"].items() if count > 0}
    owners = ["shared", "proposer", "critic"]
    if actual_owners != set(owners):
        raise ValueError("frozen4000 requires actual updated shared/P/C parameter owners only")
    summary = {"schema": FROZEN_CHECKPOINT, "checkpoint_domain": CHECKPOINT_DOMAIN,
        "eligibility": ELIGIBILITY, "file": Path(registration["checkpoint"]["path"]).name,
        "sha256": registration["checkpoint"]["sha256"], "bytes": registration["checkpoint"]["bytes"],
        "phase": ledger.phase, "pending_microbatches": ledger.pending_microbatches,
        "completed_updates": dict(ledger.completed), "dispatched_updates": dict(ledger.dispatched),
        "ledger_sha256": digest("rz-pals-generation-ledger-state/1", ledger.state()), "cost": dict(ledger.cost),
        "state_sha256": {name: _state_digest(payload[name]) for name in ("model", "optimizers", "schedulers",
            "rng", "sampler", "parameter_updates", "membership", "bucket")},
        "config_sha256": digest("rz-pals-generation-model-config/1", live.model.config.to_dict()),
        "frozen_pc_sha256": payload["frozen_pc_sha256"], "frozen_pc_model_epoch": registration["model_epoch"],
        "context_epoch": prepared["spec"]["context_epoch"], "nonzero_parameter_owners": owners,
        "cuda_devices": list(payload["cuda_devices"])}
    spec = prepared["spec"]
    receipt = {"schema": FROZEN_V2, "eligibility": "frozen_pc_collection_only",
        "generation_id": spec["generation_id"], "spec": copy.deepcopy(spec), "spec_sha256": spec_digest(spec),
        "source_sha256": spec["source_sha256"], "runtime_sha256": spec["runtime_sha256"], "checkpoint": summary,
        "pc_admission": generation_admission(spec, live.pc_dataset, "pc"), "collection": original["collection"],
        "validator_present": True, "validator_training_updates": 0, "deployment_exported": False,
        "native_loaded_assets_verified": False, "numeric_status": "not_run", "cuda_status": "not_run", "arena_eligible": False}
    _atomic_json(Path(registration["output_directory"]) / "frozen-pc-collection-v2.json", receipt, output_guard=output_guard)
    return receipt


def validate_export_registration(value):
    """Metadata validation only. No native capability is constructed."""
    fields(value, ("schema", "export_id", "kind", "training_registration",
        "original_training_registration_sha256", "trainer_receipt", "original_training_execution_id",
        "checkpoint", "rules_profile", "export_source_components", "model_epoch", "index_root",
        "output_directory", "receipt_path", "max_wall_ns", "max_output_bytes"), "original export registration")
    if value["schema"] != REGISTRATION_DOMAIN or value["kind"] not in KINDS:
        raise ValueError("registered original export domain/kind required")
    _file_identifier(value["export_id"])
    _file_identifier(value["original_training_execution_id"])
    training = validate_registration(value["training_registration"])
    if registration_digest(training) != sha(value["original_training_registration_sha256"]):
        raise ValueError("original training registration semantic pin differs")
    for name, maximum in (("trainer_receipt", MAX_JSON_BYTES), ("checkpoint", 1024**3), ("rules_profile", MAX_JSON_BYTES)):
        _artifact(value[name], maximum)
    if value["checkpoint"]["path"].rsplit(".", 1)[-1] != "pt":
        raise ValueError("actual original checkpoint must be a separately pinned .pt")
    fields(value["export_source_components"], EXPORT_COMPONENTS, "exact export source components")
    for name, pin in value["export_source_components"].items():
        sha(pin)
        if name in SOURCE_COMPONENTS and training["source_components"][name] != pin:
            raise ValueError("export cannot silently change actual trainer/data/model semantics")
    if not uint(value["model_epoch"]) or not uint(value["max_wall_ns"], MAX_WALL_NS):
        raise ValueError("original export needs separate frozen model epoch and finite window")
    if value["max_output_bytes"] != training["max_output_bytes"]:
        raise ValueError("export must retain original managed staging ceiling")
    root = Path(training["staging_root"])
    for name in ("index_root", "output_directory", "receipt_path"):
        path = _safe_absolute(value[name])
        if path == root or not path.is_relative_to(root):
            raise ValueError("export output/index/receipt must remain in original owned staging root")
    if value["receipt_path"] == training["receipt_path"] or not value["receipt_path"].endswith(".json"):
        raise ValueError("export needs a new terminal receipt; do not overwrite original trainer evidence")
    if not Path(value["checkpoint"]["path"]).is_relative_to(Path(training["checkpoint_directory"])):
        raise ValueError("export input must come from the original trainer checkpoint directory")
    if value["kind"] == "frozen_pc_collection":
        if training["phase"] != "pc" or training["datasets"]["v"] is not None:
            raise ValueError("frozen4000 export must retain original P/C corpus only")
    elif (training["phase"] != "v" or training["datasets"]["v"] is None
            or training["model_epoch"] != value["model_epoch"]):
        raise ValueError("G1 export requires actual frozen V corpus/model epoch")
    return copy.deepcopy(value)


def export_registration_digest(value):
    return digest(REGISTRATION_DOMAIN, validate_export_registration(value))


def loading_registration(value):
    """Fixed model-load derivation, not a newly reconstructed native owner."""
    value = validate_export_registration(value)
    result = copy.deepcopy(value["training_registration"])
    result.update(window_id=value["export_id"], index_root=value["index_root"],
        receipt_path=value["receipt_path"], max_wall_ns=value["max_wall_ns"])
    result["checkpoint"] = {"artifact": copy.deepcopy(value["checkpoint"]), "domain": CHECKPOINT_DOMAIN, "mode": "resume"}
    return validate_registration(result)


def validate_original_terminal(value, registration):
    """Readback agreement; original-process authority remains with the parent."""
    original = registration["training_registration"]
    if (type(value) is not dict or value.get("execution_id") != registration["original_training_execution_id"]
            or value.get("registration_sha256") != registration["original_training_registration_sha256"]
            or value.get("physical_completion") != "confirmed" or value.get("native_game_authority_issued") is not False
            or value.get("original_process_authority_issued") is not False
            or value.get("parent_must_verify_process_and_artifacts") is not True):
        raise ValueError("export must join exact original terminal training receipt, not metadata success")
    window = value.get("window")
    expected = {"proposer": 2000, "critic": 2000, "validator": 0 if registration["kind"] == "frozen_pc_collection" else 1000}
    status = "await_actual_v_collection" if registration["kind"] == "frozen_pc_collection" else "generation_updates_complete"
    if type(window) is not dict or type(window.get("ledger")) is not dict:
        raise ValueError("original export terminal needs a typed actual window/ledger")
    for counts in (window.get("completed_updates"), window["ledger"].get("completed")):
        fields(counts, tuple(expected), "exact original completed update counters")
        if any(type(counts[role]) is not int or counts[role] != count for role, count in expected.items()):
            raise ValueError("original export terminal update counters differ")
    if (type(window) is not dict or window.get("status") != status or value.get("status") != status
            or window.get("completed_updates") != expected or window.get("ledger", {}).get("completed") != expected
            or type(window["ledger"].get("pending_microbatches")) is not int or window["ledger"]["pending_microbatches"] != 0
            or window.get("ledger", {}).get("spec", {}).get("model_sha256") != original["model_sha256"]
            or value.get("durable_checkpoint_boundary") is not True):
        raise ValueError("export needs exact4000PC or5000G1 clean durable original boundary")
    pins = value.get("checkpoint_pins")
    if type(pins) is not list or not 1 <= len(pins) <= 16:
        raise ValueError("original trainer did not hold a bounded actual checkpoint set")
    original_pin = registration["checkpoint"]
    matches = [pin for pin in pins if type(pin) is dict and all(pin.get(k) == original_pin[k] for k in ("path", "sha256", "bytes"))]
    if len(matches) != 1 or window.get("checkpoints", [])[-1:] != [original_pin["path"]]:
        raise ValueError("export checkpoint differs from the latest original durable boundary")
    return expected


def _actual_export_files(root, *, until_ns, clock):
    """Pin only explicitly produced output files, bounded by the managed cap."""
    result, total = [], 0
    for directory, directories, names in os.walk(root, followlinks=False):
        for name in directories:
            path = _safe_absolute(str(Path(directory) / name))
            if path.is_symlink() or getattr(path.stat(), "st_file_attributes", 0) & 0x400:
                raise ValueError("original export artifact directory cannot be linked/reparse")
        if len(result) + len(directories) + len(names) > 128:
            raise ValueError("original manual export artifact entry cap exceeded")
        for name in sorted(names):
            pin = _regular_readback(Path(directory) / name, maximum=1024**3, until_ns=until_ns, clock=clock)
            total += pin["bytes"]
            if total > 18 * 1024**3:
                raise ValueError("original export artifact byte cap exceeded")
            result.append(pin)
    if not result:
        raise ValueError("original manual exporter produced no artifact")
    return sorted(result, key=lambda value: value["path"])


def _rules_snapshot(registration, raw, execution_id, *, until_ns, clock):
    """Keep exact pinned Rules bytes as an exclusive read-only file input."""
    _file_identifier(execution_id)
    if type(raw) is not bytes or len(raw) != registration["rules_profile"]["bytes"] or hashlib.sha256(raw).hexdigest() != registration["rules_profile"]["sha256"]:
        raise ValueError("original pinned Rules snapshot bytes differ")
    _output_credit(registration["training_registration"], reserved_bytes=len(raw))
    path = _safe_absolute(str(Path(registration["training_registration"]["staging_root"]) / ("export-rules-" + execution_id + ".json")))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        for offset in range(0, len(raw), 1024 * 1024):
            if clock() >= until_ns:
                raise TimeoutError("original Rules snapshot exceeded original export deadline")
            stream.write(raw[offset:offset + 1024 * 1024])
        stream.flush()
        os.fsync(stream.fileno())
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222 or info.st_nlink != 1:
        raise ValueError("exclusive original Rules snapshot is not private read-only")
    return _regular_readback(path, maximum=MAX_JSON_BYTES, until_ns=until_ns, clock=clock)


def _export_control_inventories(semantic, registration, *, until_ns, clock, output_guard):
    """Original actual serialized graph inventory; no session or placement claim."""
    from .virtual_control_inventory import produce_virtual_control_inventory, SCHEMA, SCOPE, MAX_INVENTORY_BYTES
    root = Path(registration["output_directory"])
    seals = {}
    kinds = ("collection",) if registration["kind"] == "frozen_pc_collection" else ("collection", "deployment")
    for kind in kinds:
        manifest = semantic[kind]["manifest"]
        manifest_path = root / kind / "export.json"
        observed_manifest = _regular_readback(manifest_path, maximum=MAX_JSON_BYTES, until_ns=until_ns, clock=clock)
        if manifest["file"] != kind + "/export.json" or any(manifest[name] != observed_manifest[name] for name in ("sha256", "bytes")):
            raise ValueError("actual control inventory manifest differs from original export descriptor")
        remaining = until_ns - clock()
        if remaining <= 0:
            raise TimeoutError("actual export control inventory reached original deadline")
        relative = kind + "/cuda-control-inventory.json"
        produced = produce_virtual_control_inventory(manifest_path, observed_manifest["sha256"], root / relative,
            opt_in=True, max_seconds=min(30, max(1, remaining // 1_000_000_000)),
            until_ns=until_ns, output_guard=output_guard)
        observed = _regular_readback(root / relative, maximum=MAX_INVENTORY_BYTES, until_ns=until_ns, clock=clock)
        pin = {name: observed[name] for name in ("path", "sha256", "bytes")}
        inventory = strict_json(_pinned_bytes(pin, maximum=MAX_INVENTORY_BYTES, until_ns=until_ns, clock=clock).decode("utf-8", "strict"))
        if (produced["schema"] != SCHEMA or inventory["schema"] != SCHEMA or inventory["scope"] != SCOPE
                or produced["inventory_sha256"] != pin["sha256"] or produced["inventory_bytes"] != pin["bytes"]
                or inventory["manifest_sha256"] != manifest["sha256"]
                or produced["total_nodes"] != inventory["total_nodes"]
                or produced["actual_provider_placement"] != "not_observed" or produced["cpu_allowlist"] != "not_created"):
            raise ValueError("actual serialized control inventory and original producer readback differ")
        seals[kind] = {"schema": CONTROL_SEAL, "kind": kind, "manifest": copy.deepcopy(manifest),
            "inventory": {"file": relative, "sha256": pin["sha256"], "bytes": pin["bytes"]},
            "graphs": copy.deepcopy(semantic[kind]["graphs"]), "producer_source_sha256": inventory["producer_source_sha256"],
            "inventory_schema": SCHEMA, "scope": SCOPE, "total_nodes": inventory["total_nodes"],
            "actual_provider_placement": "not_observed", "cpu_allowlist": "not_created"}
    return seals


def main(argv=None, *, clock=time.monotonic_ns, stdin=None):
    import argparse
    parser = argparse.ArgumentParser(prog="generation-export-original-child")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--original-export-child", action="store_true")
    parser.add_argument("--startup-sha256")
    parser.add_argument("--startup-bytes", type=int)
    parser.add_argument("--execution-until-ns", type=int)
    parser.add_argument("--whole-until-ns", type=int)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"schema": RESULT, "status": "metadata_only", "actual_export_executed": False,
            "optimizer_updates_executed": 0, "native_game_authority_issued": False, "arena_accepted": False}, sort_keys=True))
        return 0
    if not args.original_export_child or os.name != "posix" or not uint(args.startup_bytes, MAX_JSON_BYTES):
        raise ValueError("explicit original Linux export child + finite startup byte credit required")
    stream = sys.stdin.buffer if stdin is None else stdin
    raw = stream.read(args.startup_bytes + 1)
    if type(raw) is not bytes or len(raw) != args.startup_bytes or hashlib.sha256(raw).hexdigest() != sha(args.startup_sha256):
        raise ValueError("export original startup bytes/EOF differ from parent raw pin")
    startup = strict_json(raw.decode("utf-8", "strict"))
    fields(startup, ("schema", "execution_id", "registration_sha256", "registration", "original_started_ns",
        "execution_until_ns", "whole_until_ns", "dataset_pin_scope", "dependency_directories", "python_runtime"), "original export startup")
    if startup["schema"] != STARTUP or startup["dataset_pin_scope"] != "original_trainer_held_checkpoint_readback_only":
        raise ValueError("unregistered original export startup/scope")
    registration = validate_export_registration(startup["registration"])
    if export_registration_digest(registration) != sha(startup["registration_sha256"]):
        raise ValueError("export registration differs from original parent startup")
    load = loading_registration(registration)
    # Reuse the isolated runtime/dependency + absolute-deadline validator. This
    # synthetic *validation input* never goes to an optimizer dispatcher.
    validation_startup = dict(startup, schema=STARTUP_DOMAIN, dataset_pin_scope="external_original_archive_pin_native_chain_readback_only",
        registration=load, registration_sha256=registration_digest(load))
    validation_raw = json.dumps(validation_startup, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    validate_startup(validation_startup, expected_raw_sha256=hashlib.sha256(validation_raw).hexdigest(), raw_bytes=validation_raw,
        execution_until_ns=args.execution_until_ns, whole_until_ns=args.whole_until_ns, now_ns=clock())
    interpreter = _install_original_dependencies(startup, until_ns=args.execution_until_ns, clock=clock)
    prepared = prepare_registration(load, expected_registration_sha256=registration_digest(load), until_ns=args.execution_until_ns, clock=clock)
    original_terminal = strict_json(_pinned_bytes(registration["trainer_receipt"], maximum=MAX_JSON_BYTES,
        until_ns=args.execution_until_ns, clock=clock).decode("utf-8", "strict"))
    expected_updates = validate_original_terminal(original_terminal, registration)
    rules_raw = _pinned_bytes(registration["rules_profile"], maximum=MAX_JSON_BYTES,
        until_ns=args.execution_until_ns, clock=clock)
    from .v5_source_registry import source_pins
    if source_pins(EXPORT_COMPONENTS) != registration["export_source_components"]:
        raise ValueError("actual export source bytes differ from original registration")
    if Path(registration["output_directory"]).exists():
        raise FileExistsError("export preserves prior output; never overwrite or eligibility-flip diagnostic artifacts")
    cancelled, previous = [False], {}
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, lambda _s, _f: cancelled.__setitem__(0, True))
    producer = PinnedOriginalTrainerChild()
    result, exit_code, rules_snapshot = None, 0, None
    def check_original():
        if cancelled[0] or clock() >= args.execution_until_ns:
            raise TimeoutError("original export window cancelled/expired; preserve original owner and partial output")
        return _output_credit(load, reserved_bytes=2 * registration["checkpoint"]["bytes"])
    def write_guard(additional_bytes):
        if not uint(additional_bytes, 1024**3) or check_original() < additional_bytes:
            raise ValueError("original exporter lacks exact pre-write managed byte credit")
    try:
        try:
            check_original()
            rules_snapshot = _rules_snapshot(registration, rules_raw, startup["execution_id"],
                until_ns=args.execution_until_ns, clock=clock)
            with producer.open_window(load, prepared, args.execution_until_ns) as live:
                configure_generation_runtime(prepared["spec"], execute=True, initialize_rng=False)
                live.optimizers = GenerationOptimizers(live.model, prepared["spec"], execute=True)
                validator = lambda value: _sampler_validator(value, live, live.checkpoint_payload)
                verified = validate_checkpoint(live.checkpoint_payload, live.model, live.optimizers, prepared["spec"], validator)
                if verified.completed != expected_updates or verified.pending_microbatches:
                    raise ValueError("decoded actual checkpoint differs from original terminal update boundary")
                live.ledger, live.sampler_states, _ = restore_checkpoint(live.checkpoint_payload, live.model,
                    live.optimizers, prepared["spec"], validator)
                live.model.eval()
                check_original()
                if registration["kind"] == "frozen_pc_collection":
                    exported = export_frozen_pc_collection(prepared["spec"], registration["checkpoint"]["path"],
                        registration["checkpoint"]["sha256"], live.model, live.optimizers, live.pc_dataset,
                        sampler_validator=validator, model_epoch=registration["model_epoch"],
                        directory=registration["output_directory"], rules_profile_json=rules_snapshot["path"],
                        execute=True, until_ns=args.execution_until_ns, clock=clock, output_guard=write_guard)
                    semantic_receipt = _frozen_collection_v2(exported["receipt"], live, prepared, registration, output_guard=write_guard)
                else:
                    from .generation_export import export_actual_generation
                    exported = export_actual_generation(registration["checkpoint"]["path"], expected_checkpoint_sha256=registration["checkpoint"]["sha256"],
                        model=live.model, optimizers=live.optimizers, spec=prepared["spec"],
                        pc_dataset=live.pc_dataset, v_dataset=live.v_dataset,
                        frozen_pc_model_epoch=registration["model_epoch"], sampler_validator=validator,
                        rules_profile_json=rules_snapshot["path"], directory=registration["output_directory"], execute=True,
                        output_guard=write_guard)
                    semantic_receipt = exported["provenance"]
                check_original()
                controls = _export_control_inventories(semantic_receipt, registration,
                    until_ns=args.execution_until_ns, clock=clock, output_guard=write_guard)
                check_original()
                live.check_original_binding(load, prepared, args.execution_until_ns)
                result = {"schema": RESULT, "status": "exported", "kind": registration["kind"],
                    "actual_export_executed": True, "semantic_receipt": semantic_receipt, "control_inventory_seals": controls,
                    "frozen_pc_sha256": live.optimizers.frozen_pc_sha256, "model_epoch": registration["model_epoch"],
                    "actual_updates": dict(live.ledger.completed), "checkpoint_sha256": registration["checkpoint"]["sha256"]}
        except BaseException as error:
            result, exit_code = {"schema": RESULT, "status": "failed", "kind": registration["kind"],
                "error_type": type(error).__name__, "error": str(error)[:4096], "actual_export_executed": False,
                "partial_artifacts_preserved": True, "automatic_retry": False}, 1
        live = producer.last_state
        artifact_pins, active_source = [], None
        try:
            if Path(registration["output_directory"]).exists():
                artifact_pins = _actual_export_files(registration["output_directory"], until_ns=args.whole_until_ns, clock=clock)
            active_source = _active_source_readback(until_ns=args.whole_until_ns, clock=clock)
            active_source["source_components"] = source_pins(EXPORT_COMPONENTS)
            # runpy's __main__ is an actual executed source module too.
            if not any(module["module"] == "__main__" for module in active_source["source_modules"]):
                active_source["source_modules"].append({"module": "__main__", **_regular_readback(__file__, maximum=4 * 1024**2,
                    until_ns=args.whole_until_ns, clock=clock)})
            if active_source["source_components"] != registration["export_source_components"]:
                raise ValueError("actual terminal exporter source differs")
            _staging_bytes(load)
        except BaseException as error:
            result.update(status="artifact_source_readback_failed", readback_error_type=type(error).__name__, readback_error=str(error)[:4096])
            exit_code = 1
        result.update(execution_id=startup["execution_id"], registration_sha256=startup["registration_sha256"],
            original_training_execution_id=registration["original_training_execution_id"],
            original_training_registration_sha256=registration["original_training_registration_sha256"],
            original_trainer_receipt=registration["trainer_receipt"], original_checkpoint=registration["checkpoint"],
            rules_profile_snapshot=rules_snapshot,
            directory=registration["output_directory"], artifact_pins=artifact_pins, active_source=active_source,
            interpreter_readback=interpreter, runtime_readback=None if live is None else live.runtime_readback,
            physical_completion="not_started" if live is None else live.physical_completion,
            optimizer_updates_executed=0, neural_forwards_executed=0, native_game_authority_issued=False,
            original_process_authority_issued=False, numeric_status="not_run", cuda_status="not_run", arena_accepted=False,
            parent_must_verify_process_and_artifacts=True, original_execution_until_ns=args.execution_until_ns,
            original_whole_until_ns=args.whole_until_ns)
        if result["physical_completion"] == "unknown_quarantined":
            result["status"], exit_code = "physical_completion_unknown", 1
        receipt = _terminal_receipt(load, result)
        print(json.dumps({"schema": RESULT, "status": result["status"], "receipt": receipt,
            "actual_export_executed": result["actual_export_executed"], "physical_completion": result["physical_completion"],
            "optimizer_updates_executed": 0, "arena_accepted": False}, sort_keys=True, allow_nan=False))
        return exit_code
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
