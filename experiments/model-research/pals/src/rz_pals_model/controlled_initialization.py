"""Explicit new untrained baseline registration with paired parameter evidence.

Never relabel an existing diagnostic checkpoint. The frozen initializer builds
the legacy banks before drawing every V2 bank in a fixed order; this tool
records the actual name/shape/FP32-bit comparisons for all four profiles.
Initialization is opt-in. The scheduler must provide its NN slot, resource
limits and physical timeout; this helper checks wall/output bounds by stage.
"""
import argparse
from collections import OrderedDict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

from .config import ModelConfig, PROFILES, matmul_flops

SCHEMA = "rz-pals-controlled-untrained-baselines/1"
INITIALIZATION_RECIPE = "legacy-banks-first;all-v2-banks-fixed-order-then-profile-selection;torch2.8.0-cpu;seeded-cpu-generator;fp32"
MAX_REGISTRATION_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 512 * 1024 * 1024
MAX_WALL_SECONDS = 300


def initialization_plan(seed=23):
    """Pure preparation. No dependency import, model construction or file write."""
    if type(seed) is not int or not 0 <= seed < 1 << 63:
        raise ValueError("controlled initializer seed must be a nonnegative 63-bit integer")
    return {"schema": SCHEMA, "phase": "prepared", "seed": seed, "recipe": INITIALIZATION_RECIPE,
            "new_checkpoints_required": True, "existing_diagnostic_relabel_allowed": False,
            "cost_comparison_shape": {"batch": 1, "role": "proposer", "records": 3, "candidates": 7,
                                      "divergences": 0, "full_line_physical_plies": 256},
            "runtime_latency": "not_measured", "parameter_memory": "recorded_after_explicit_initialization",
            "profiles": [{"profile": profile, "config": ModelConfig.for_profile(profile).to_dict(),
                          "full_line": ModelConfig.for_profile(profile).full_line,
                          "interaction_head": ModelConfig.for_profile(profile).interaction_head,
                          "declared_shape_flops": matmul_flops(ModelConfig.for_profile(profile), "proposer", 3, 7)}
                         for profile in PROFILES],
            "optimizer_steps": 0, "nn_forwards": 0, "arena_baseline_candidate": True,
            "arena_eligible": False, "eligibility": "requires_final_compiled_export_and_native_arena_registration",
            "diagnostic_only": False, "strength_accepted": False}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def compare_parameter_inventories(inventories):
    """Pure comparison of complete producer inventories, not model execution.

    Matching names must have the same shapes/dtypes and exact LE FP32 digest.
    This tests both the four-way common core and every shared added component.
    Heads replaced by differently named components remain explicit per profile.
    """
    if type(inventories) is not dict or set(inventories) != set(PROFILES):
        raise ValueError("controlled registration requires all four explicit profiles")
    indexed = {}
    for profile in PROFILES:
        entries = inventories[profile]
        if type(entries) is not list or not 1 <= len(entries) <= 512:
            raise ValueError("invalid complete parameter inventory size")
        values = {}
        for entry in entries:
            if type(entry) is not dict or set(entry) != {"name", "shape", "dtype", "bytes", "sha256"}:
                raise ValueError("invalid parameter inventory fields")
            name, shape, digest = entry["name"], entry["shape"], entry["sha256"]
            if (type(name) is not str or not name or len(name) > 256 or name in values
                    or type(shape) is not list or len(shape) > 8
                    or any(type(size) is not int or not 1 <= size <= 1048576 for size in shape)
                    or entry["dtype"] != "fp32_le" or type(entry["bytes"]) is not int
                    or entry["bytes"] != 4 * math.prod(shape)
                    or type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("parameter inventory shape/name/bytes/FP32 digest mismatch")
            values[name] = entry
        indexed[profile] = values
    core_names = sorted(set.intersection(*(set(values) for values in indexed.values())))
    if not core_names:
        raise ValueError("paired profiles must have a nonempty common parameter core")
    pairs = []
    for left, right in itertools.combinations(PROFILES, 2):
        names = sorted(indexed[left].keys() & indexed[right].keys())
        for name in names:
            if indexed[left][name] != indexed[right][name]:
                raise ValueError("paired common parameter bits/shape differ: " + name)
        common = [indexed[left][name] for name in names]
        pairs.append({"left": left, "right": right, "common_parameter_tensors": len(common),
                      "common_parameter_values": sum(entry["bytes"] for entry in common) // 4,
                      "common_parameter_bytes": sum(entry["bytes"] for entry in common),
                      "common_inventory_sha256": hashlib.sha256(_canonical(common)).hexdigest(), "equal_fp32_bits": True})
    core = [indexed[PROFILES[0]][name] for name in core_names]
    return {"common_core": core, "common_core_bytes": sum(entry["bytes"] for entry in core),
            "common_core_inventory_sha256": hashlib.sha256(_canonical(core)).hexdigest(), "pairs": pairs,
            "per_profile": {profile: {"parameter_tensors": len(indexed[profile]),
                                      "parameter_values": sum(entry["bytes"] for entry in indexed[profile].values()) // 4,
                                      "total_parameter_bytes": sum(entry["bytes"] for entry in indexed[profile].values()),
                                      "outside_common_core": sorted(indexed[profile].keys() - set(core_names))}
                            for profile in PROFILES}}


def _parameter_inventory(checkpoint, expected_profile, expected_seed, expected_parameter_count):
    """Read the newly generated checkpoint tensors; never run a model forward."""
    import numpy as np
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig.for_profile(expected_profile)
    if (type(payload) is not dict or payload.get("schema") != config.model_semantics
            or payload.get("config") != config.to_dict() or type(payload.get("seed")) is not int
            or payload["seed"] != expected_seed or type(payload.get("training_steps")) is not int
            or payload["training_steps"] != 0 or payload.get("trained") is not False
            or type(payload.get("state_dict")) not in (dict, OrderedDict)):
        raise ValueError("new controlled checkpoint configuration/seed/training provenance mismatch")
    result = []
    for name, tensor in sorted(payload["state_dict"].items()):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("controlled initializer requires finite CPU FP32 tensors")
        raw = np.ascontiguousarray(tensor.numpy(), dtype="<f4").tobytes()
        result.append({"name": name, "shape": list(tensor.shape), "dtype": "fp32_le",
                       "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    if sum(entry["bytes"] for entry in result) // 4 != expected_parameter_count:
        raise ValueError("new controlled checkpoint parameter inventory is incomplete")
    return result


def register_controlled_baselines(directory, seed=23, *, max_seconds=MAX_WALL_SECONDS):
    """Generate a fresh four-profile W0 set only under a scheduled CPU window.

    The original checkpoint schemas and weights-only loader remain unchanged.
    A separate receipt declares this new intended baseline and every comparison;
    it grants no final export, native/arena admission or learned-strength claim.
    """
    plan = initialization_plan(seed)
    if type(max_seconds) is not int or not 1 <= max_seconds <= MAX_WALL_SECONDS:
        raise ValueError("controlled initializer wall limit must be in 1..300 seconds")
    destination = Path(directory)
    if not destination.is_absolute() or destination.exists():
        raise ValueError("controlled baseline requires a new absolute output directory")
    source_root = Path(__file__).resolve().parent
    source_names = ("config.py", "model.py", "artifacts.py", "controlled_initialization.py")
    def source_hashes():
        return {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in source_names}
    before = source_hashes()
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("controlled initializer wall budget exceeded")
    import numpy as np
    import torch
    if torch.__version__ != "2.8.0+cpu" or np.__version__ != "2.2.6":
        raise ValueError("controlled initializer requires registered Torch2.8.0+cpu/NumPy2.2.6")
    from .artifacts import atomic_json, digest_file, initialize_checkpoint, output_directory
    original_threads, original_dtype = torch.get_num_threads(), torch.get_default_dtype()
    original_rng = torch.get_rng_state().clone()
    output = output_directory(destination)
    records, inventories = [], {}
    try:
        torch.set_num_threads(2)
        torch.set_default_dtype(torch.float32)
        with torch.device("cpu"):
            for profile in PROFILES:
                deadline()
                child = output / profile
                metadata = initialize_checkpoint(child, seed, profile)
                checkpoint = child / metadata["checkpoint"]
                inventories[profile] = _parameter_inventory(checkpoint, profile, seed, metadata["parameter_count"])
                if digest_file(checkpoint) != metadata["checkpoint_sha256"]:
                    raise ValueError("new controlled checkpoint identity changed")
                records.append({"profile": profile, "checkpoint": profile + "/" + checkpoint.name,
                                "checkpoint_sha256": metadata["checkpoint_sha256"], "checkpoint_bytes": checkpoint.stat().st_size,
                                "metadata_sha256": digest_file(child / "checkpoint.json"), "config": metadata["config"],
                                "seed": seed, "parameter_count": metadata["parameter_count"], "trained": False, "training_steps": 0})
                deadline()
                if sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) > MAX_OUTPUT_BYTES:
                    raise ValueError("controlled initializer output budget exceeded")
        comparison = compare_parameter_inventories(inventories)
    finally:
        torch.set_num_threads(original_threads)
        torch.set_default_dtype(original_dtype)
    after = source_hashes()
    if before != after or not torch.equal(original_rng, torch.get_rng_state()):
        raise ValueError("controlled initializer did not preserve source identity/caller CPU RNG")
    receipt = {**plan, "phase": "initialized_parameter_bits_compared", "new_checkpoints": records,
               "parameter_inventories": inventories, "comparison": comparison,
               "source_hashes_before": before, "source_hashes_after": after, "sources_unchanged": True,
               "caller_cpu_rng_preserved": True, "caller_threads_preserved": torch.get_num_threads() == original_threads,
               "caller_default_dtype_preserved": torch.get_default_dtype() == original_dtype,
               "elapsed_seconds": time.monotonic() - started, "wall_limit_seconds": max_seconds,
               "output_limit_bytes": MAX_OUTPUT_BYTES, "old_diagnostic_assets_modified": False,
               "torch_version": torch.__version__, "numpy_version": np.__version__, "precision": "fp32", "tf32": False}
    if len(_canonical(receipt)) > MAX_REGISTRATION_BYTES:
        raise ValueError("controlled initializer declaration exceeds its receipt limit")
    deadline()
    atomic_json(output / "controlled_initialization.json", receipt)
    return {"schema": SCHEMA, "phase": receipt["phase"], "seed": seed, "profiles": list(PROFILES),
            "receipt_sha256": digest_file(output / "controlled_initialization.json"), "common_fp32_parameters_equal": True,
            "optimizer_steps": 0, "nn_forwards": 0, "arena_eligible": False, "arena_baseline_candidate": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepared = commands.add_parser("plan")
    prepared.add_argument("--seed", type=int, default=23)
    register = commands.add_parser("register-new-baselines")
    register.add_argument("--output", required=True)
    register.add_argument("--seed", type=int, default=23)
    register.add_argument("--max-seconds", type=int, default=MAX_WALL_SECONDS)
    args = parser.parse_args(argv)
    result = initialization_plan(args.seed) if args.command == "plan" else register_controlled_baselines(args.output, args.seed, max_seconds=args.max_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
