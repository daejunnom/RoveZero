#!/usr/bin/env python3
"""Finite local V4 preparation, native model validation and 120+1 pilot.

No compilation, package installation, remote GPU creation or automatic retry.
Planning is the default. --run consumes the independently pinned executable
under existing inherited CPU/cgroup limits and preserves an outside-Git receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
from managed_process import run_group
from managed_storage import checked

GIB = 1024 ** 3
SCHEMA = "rz-pals-followup-local-wrapper/1"
PROFILES = ("legacy_summary_v1", "full_line_v2", "interaction_head_v2", "full_line_interaction_v2")


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("an independently registered lowercase SHA256 is required")
    return value


def regular(path, cap):
    if not path.is_absolute():
        raise ValueError("absolute paths are required")
    path = checked(path)
    if not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError("bounded regular file required")
    return path


def pinned(path, digest, cap):
    path = regular(path, cap)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    if hasher.hexdigest() != sha(digest):
        raise ValueError("actual pinned file bytes differ from registration")
    return path


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def bounded_json(path):
    path = regular(path, 256 * 1024)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def outside_git(path):
    if not path.is_absolute():
        raise ValueError("absolute output required")
    path = checked(path)
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            raise ValueError("generated output must remain outside Git")
    return path


def check_resources(cpus):
    """Read actual inherited policy; install no global or per-host setting."""
    if sys.platform != "linux" or len(cpus) != 2 or len(set(cpus)) != 2:
        raise ValueError("execution requires Linux and exactly two registered CPUs")
    if sorted(os.sched_getaffinity(0)) != sorted(cpus):
        raise ValueError("actual inherited affinity differs from the two registered CPUs")
    membership = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    relative = next((line[3:] for line in membership.splitlines() if line.startswith("0::")), None)
    if relative is None or not relative.startswith("/") or any(p in (".", "..") for p in relative.split("/")):
        raise ValueError("actual cgroup-v2 membership required")
    root = Path("/sys/fs/cgroup") / relative.lstrip("/")
    limits = {"memory.high": 6 * GIB, "memory.max": 12 * GIB, "memory.swap.max": 0}
    for name, value in limits.items():
        if (root / name).read_text(encoding="ascii").strip() != str(value):
            raise ValueError("inherited memory.high/max/swap differs from 6GiB/12GiB/0")
    return {"cpu_affinity": sorted(cpus), **limits, "method": "actual inherited affinity and cgroup-v2 reads"}


def arena_plan(args, binary):
    lock_path = pinned(args.lock, args.lock_sha256, 256 * 1024)
    lock = bounded_json(lock_path)
    if (lock.get("domain") != "rz-pals-arena-launch-v4/1"
            or lock.get("input", {}).get("domain") != "rz-pals-arena-launch-v4/1"):
        raise ValueError("an actual locked V4 arena envelope is required")
    sha(lock["sha256"])
    value = lock["input"]["semantic_lock"]["manifest"]
    pilot = value["pilot"]
    if (value.get("schema_version") != 4 or value.get("contract_revision") != "pals/0.2"
            or value.get("purpose") != "arena_pilot" or value.get("training_executed") is not False
            or pilot.get("games") != 2 or pilot.get("base_ms") != 120000 or pilot.get("increment_ms") != 1000
            or not 1 <= pilot.get("max_plies", 0) <= 256
            or not 1 <= pilot.get("wall_time_max_ms", 0) <= 900000
            or not 1 <= pilot.get("cleanup_max_ms", 0) <= 30000):
        raise ValueError("V4 pilot is frozen to exchanged 120+1 games, max256 ply, 15min+30s")
    cpus = sorted(args.cpu_affinity)
    if len(value["resources"]) != 2:
        raise ValueError("exactly two exchanged-game resource declarations required")
    for resource in value["resources"]:
        if (sorted(resource["cpu_affinity"]) != cpus or resource["cpu_threads"] != 2
                or resource["memory_high_bytes"] != 6 * GIB or resource["memory_max_bytes"] != 12 * GIB
                or resource["swap_max_bytes"] != 0):
            raise ValueError("V4 resource declaration differs from finite local CPU2/6GiB/12GiB policy")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.label):
        raise ValueError("bounded fresh attempt label required")
    assets = checked(args.asset_root)
    output = outside_git(args.output_root)
    if not assets.is_absolute() or not assets.is_dir() or not output.is_dir() or (output / args.label).exists():
        raise ValueError("existing absolute asset/output roots and fresh attempt label required")
    return [str(binary), args.stage, str(lock_path), str(assets), str(output), args.label], {
        "lock_file_sha256": args.lock_sha256, "arena_launch_sha256": lock["sha256"],
        "pilot": pilot, "resource_declarations_checked": True, "native_policy_admission": "Rust V4 owner before spawn",
    }


def model_plan(args, binary):
    export = pinned(args.export, args.export_sha256, 256 * 1024)
    descriptor = bounded_json(export)
    profile = descriptor.get("config", {}).get("profile", "legacy_summary_v1")
    if profile != args.model_profile:
        raise ValueError("actual export profile differs from explicit registered profile")
    if profile != "legacy_summary_v1" and (
            descriptor.get("schema") != "rovezero.pals-model.v2"
            or descriptor.get("model_semantics") != "rovezero.pals-model-semantics.v2"
            or descriptor.get("encoding_schema") != "rovezero.pals-board-records.v2"):
        raise ValueError("new model semantics and encoding must be actual V2")
    fixtures = pinned(args.fixtures, args.fixtures_sha256, 32 * 1024 * 1024)
    runtime = pinned(args.runtime, args.runtime_sha256, 1024 * 1024 * 1024)
    report = outside_git(args.model_report)
    cache = outside_git(args.runtime_cache)
    if report.exists() or not report.parent.is_dir():
        raise ValueError("new model report under existing outside-Git parent required")
    command = [str(binary), str(export), args.export_sha256, str(runtime), args.runtime_sha256,
               str(fixtures), str(report), args.provider, str(cache)]
    if args.provider != "cpu":
        if args.cuda_bundle is None or args.cuda_bundle_sha256 is None:
            raise ValueError("CUDA validation needs an independently pinned actual runtime bundle")
        bundle = pinned(args.cuda_bundle, args.cuda_bundle_sha256, 256 * 1024)
        command.append(str(bundle))
    elif args.cuda_bundle is not None or args.cuda_bundle_sha256 is not None:
        raise ValueError("CPU validation cannot silently inherit CUDA assets")
    return command, {"model_profile": profile, "export_sha256": args.export_sha256,
                     "fixture_sha256": args.fixtures_sha256, "provider": args.provider,
                     "numeric_reference_only": True, "strength_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cpu-affinity", nargs=2, type=int, required=True)
    parser.add_argument("--seconds", type=int, default=960)
    parser.add_argument("--run", action="store_true")
    commands = parser.add_subparsers(dest="mode", required=True)
    arena = commands.add_parser("arena")
    arena.add_argument("--stage", choices=("prepare", "execute"), default="prepare")
    arena.add_argument("--lock", type=Path, required=True)
    arena.add_argument("--lock-sha256", required=True)
    arena.add_argument("--asset-root", type=Path, required=True)
    arena.add_argument("--output-root", type=Path, required=True)
    arena.add_argument("--label", required=True)
    model = commands.add_parser("model-check")
    model.add_argument("--model-profile", choices=PROFILES, default="full_line_interaction_v2")
    model.add_argument("--export", type=Path, required=True)
    model.add_argument("--export-sha256", required=True)
    model.add_argument("--fixtures", type=Path, required=True)
    model.add_argument("--fixtures-sha256", required=True)
    model.add_argument("--runtime", type=Path, required=True)
    model.add_argument("--runtime-sha256", required=True)
    model.add_argument("--runtime-cache", type=Path, required=True)
    model.add_argument("--model-report", type=Path, required=True)
    model.add_argument("--provider", choices=("cpu", "cuda", "cuda-device"), default="cpu")
    model.add_argument("--cuda-bundle", type=Path)
    model.add_argument("--cuda-bundle-sha256")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600 or any(cpu < 0 for cpu in args.cpu_affinity) or len(set(args.cpu_affinity)) != 2:
        parser.error("finite 1..3600s window and two distinct nonnegative CPU IDs required")
    receipt_path = outside_git(args.receipt)
    if receipt_path.exists() or not receipt_path.parent.is_dir():
        parser.error("receipt must be a fresh file under existing outside-Git parent")
    binary = pinned(args.binary, args.binary_sha256, 256 * 1024 * 1024)
    command, details = arena_plan(args, binary) if args.mode == "arena" else model_plan(args, binary)
    receipt = {"schema": SCHEMA, "mode": args.mode, "planning_only": not args.run,
               "command": command, "binary_sha256": args.binary_sha256, "window_seconds": args.seconds,
               "gpu_validation_window_seconds_max": 3600, "resource_policy": {"cpu_affinity": sorted(args.cpu_affinity),
                   "memory_high_bytes": 6 * GIB, "memory_max_bytes": 12 * GIB, "swap_max_bytes": 0},
               "actual_resources": "unknown", "training_executed": False, "strength_eligible": False,
               "automatic_retry": False, "tree_gone": "unknown", "exit_code": None, **details}
    started = time.monotonic()
    if args.run:
        receipt["actual_resources"] = check_resources(args.cpu_affinity)
        environment = os.environ.copy()
        environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"})
        code, gone, failure = run_group(command, binary.parent, environment, args.seconds,
                                       lambda: check_resources(args.cpu_affinity))
        receipt.update(exit_code=code, tree_gone=gone, supervisor_error=failure,
                       status="completed" if code == 0 and gone else "failed_ownership_preserved")
    else:
        receipt["status"] = "prepared_no_child_spawned"
    receipt["elapsed_seconds"] = time.monotonic() - started
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"wrapper_receipt={receipt_path} planning_only={not args.run} training_executed=false strength_eligible=false")
    if args.run and receipt["tree_gone"] is not True:
        return 125
    return receipt["exit_code"] or 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"pals_followup_wrapper_error={error}; no automatic retry; existing artifacts/ownership retained", file=sys.stderr)
        raise SystemExit(125)
