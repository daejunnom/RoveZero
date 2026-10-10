#!/usr/bin/env python3
"""Bounded CPU-only PALS initialization/export validation; no training.

Run under run_managed.py so the dependency environment and pip scratch are
owned, bounded, and removed after all children exit. Preserve only the explicit
output directory outside the checkout. GPU validation is a different stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
from managed_storage import checked


def run(command, *, cwd, environment, seconds=300):
    started = time.monotonic()
    subprocess.run(command, cwd=cwd, env=environment, check=True, timeout=seconds)
    return time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runtime-store", type=Path,
                        help="External immutable content-addressed CPU runtime asset store")
    parser.add_argument("--source-commit", help="Coordinator-verified HEAD for a cross-OS worktree")
    parser.add_argument("--source-dirty", action="store_true")
    parser.add_argument("--layout", choices=("separate_pc", "shared_pc_if"), default="separate_pc")
    parser.add_argument("--rules-profile-json", type=Path,
                        help="Registered descriptor emitted by the Rust Rules encoder")
    parser.add_argument("--dataset-run", type=Path,
                        help="Registered own-data collection directory for no-step preparation")
    parser.add_argument("--dataset-receipt-sha256")
    parser.add_argument("--dataset-encoder-source-sha256")
    parser.add_argument("--verifier-cpu-binary", type=Path,
                        help="Registered own Rust training-private CPU task dispatcher")
    parser.add_argument("--verifier-cpu-binary-sha256")
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**63 - 1:
        parser.error("seed must fit nonnegative signed int64")
    if args.source_commit and not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("source commit must be a full lowercase SHA-1")
    dataset_args = (args.dataset_run, args.dataset_receipt_sha256,
                    args.dataset_encoder_source_sha256)
    if any(dataset_args) and not all(dataset_args):
        parser.error("dataset run, receipt digest, and encoder source digest must be supplied together")
    for digest in dataset_args[1:]:
        if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
            parser.error("dataset identity requires lowercase SHA-256")
    verifier_args = (args.verifier_cpu_binary, args.verifier_cpu_binary_sha256)
    if any(verifier_args) and (not all(verifier_args) or not all(dataset_args)):
        parser.error("private verifier requires binary/digest and a registered own dataset")
    if args.verifier_cpu_binary_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.verifier_cpu_binary_sha256):
        parser.error("private verifier binary requires lowercase SHA-256")
    source = Path(__file__).resolve().parent.parent
    output = checked(args.output)
    runtime_store = checked(args.runtime_store) if args.runtime_store else None
    rules_profile = checked(args.rules_profile_json) if args.rules_profile_json else None
    dataset_run = checked(args.dataset_run) if args.dataset_run else None
    verifier_binary = checked(args.verifier_cpu_binary) if args.verifier_cpu_binary else None
    if verifier_binary and (not verifier_binary.is_file() or not 1 <= verifier_binary.stat().st_size <= 1024**3):
        parser.error("private verifier dispatcher must be a bounded regular executable")
    if args.layout == "shared_pc_if" and rules_profile is None:
        parser.error("shared_pc_if requires a registered Rules descriptor")
    if rules_profile and (not rules_profile.is_file() or rules_profile.stat().st_size > 64 * 1024):
        parser.error("Rules descriptor must be a bounded regular file")
    if runtime_store and (runtime_store.is_relative_to(source) or source.is_relative_to(runtime_store)):
        parser.error("runtime store must be outside the checkout")
    if output.is_relative_to(source) or source.is_relative_to(output) or output.exists():
        parser.error("output must be a fresh directory outside the checkout")
    scratch = os.environ.get("TMPDIR")
    if not scratch or os.environ.get("CARGO_TARGET_DIR") is None:
        parser.error("run through scripts/run_managed.py with managed scratch")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "rovezero.pals-cpu-model-validation.v1", "status": "running",
               "training_executed": False, "training_steps": 0,
               "provider": "cpu", "gpu_validation": "not_run", "stages": {}}
    receipt["seed"] = args.seed
    receipt["layout"] = args.layout
    if rules_profile:
        receipt["rules_profile"] = {"sha256": hashlib.sha256(rules_profile.read_bytes()).hexdigest(),
                                    "bytes": rules_profile.stat().st_size}
    if dataset_run:
        receipt["preparation_dataset"] = {
            "receipt_sha256": args.dataset_receipt_sha256,
            "encoder_source_sha256": args.dataset_encoder_source_sha256,
            "optimizer_updates": 0,
        }
    if args.source_commit:
        receipt["source_commit"] = args.source_commit
        receipt["source_dirty"] = args.source_dirty
        receipt["source_commit_method"] = "coordinator_registered_cross_os_worktree"
    else:
        receipt["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        receipt["source_dirty"] = subprocess.run(
            ["git", "diff", "--quiet", "--no-ext-diff"], cwd=source, check=False).returncode != 0
        receipt["source_commit_method"] = "local_git"
    package = source / "experiments/model-research/pals"
    source_files = [Path(__file__).resolve(), source / "crates/rz-eval/src/pals_model.rs",
                    source / "crates/rz-eval/Cargo.toml",
                    source / "crates/rz-eval/examples/device_packing_graph_check.rs",
                    source / "crates/rz-eval/src/pals_onnx/device_packing_admission.rs",
                    source / "crates/rz-eval/src/pals_onnx/device_packing_graph.rs",
                    source / "crates/rz-eval/src/pals_onnx.rs"]
    source_files.extend(sorted((package / "src/rz_pals_model").glob("*.py")))
    source_files.extend(sorted((package / "tests").glob("*.py")))
    receipt["source_files"] = {str(p.relative_to(source)).replace("\\", "/"):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    receipt["versions"] = {"torch": "2.8.0", "numpy": "2.2.6", "onnx": "1.19.0",
                           "onnxruntime": "1.22.0"}
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_CACHE_DIR": "1",
                        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    try:
        with tempfile.TemporaryDirectory(prefix="pals-validation-", dir=scratch) as temporary:
            venv = Path(temporary) / "venv"
            run([sys.executable, "-m", "venv", "--copies", str(venv)],
                cwd=source, environment=environment)
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            stages = receipt["stages"]
            stages["torch_install_seconds"] = run([
                str(python), "-m", "pip", "install", "torch==2.8.0",
                "--index-url", "https://download.pytorch.org/whl/cpu"],
                cwd=source, environment=environment, seconds=600)
            stages["reference_install_seconds"] = run([
                str(python), "-m", "pip", "install", "numpy==2.2.6", "onnx==1.19.0",
                "onnxruntime==1.22.0"], cwd=source, environment=environment)
            environment["PYTHONPATH"] = str(package / "src")
            stages["model_tests_seconds"] = run([
                str(python), "-m", "unittest", "discover", "-s", str(package / "tests"), "-v"],
                cwd=source, environment=environment, seconds=600)
            # This separate static gate consumes the actual deterministic Python
            # artifact in Rust. It creates no ORT Session, provider or Run.
            packing = output / "device-packing"
            stages["device_packing_export_seconds"] = run([
                str(python), "-c",
                "from rz_pals_model.device_packing_artifacts import export_device_packing_artifact\n"
                "import sys\nexport_device_packing_artifact(sys.argv[1])\n",
                str(packing)], cwd=source, environment=environment)
            packing_files = {
                name: packing / name
                for name in ("device-packing.json", "device_public_pack.onnx")}
            packing_pins = {}
            for name, asset_path in packing_files.items():
                # Producer already applies its fixed file bounds; this consumer
                # independently checks the actual files before hashing them.
                cap = 512 * 1024 if name.endswith(".json") else 2 * 1024 * 1024
                size = asset_path.stat().st_size
                if asset_path.is_symlink() or not asset_path.is_file() or not 1 <= size <= cap:
                    raise OSError("generated packing artifact violates static file bounds")
                with asset_path.open("rb") as handle:
                    sha = hashlib.file_digest(handle, "sha256").hexdigest()
                packing_pins[name] = {"bytes": size, "sha256": sha}
            stages["device_packing_rust_body_seconds"] = run([
                "cargo", "run", "--locked", "-p", "rz-eval", "--features", "onnx,contracts",
                "--example", "device_packing_graph_check", "--quiet", "--",
                str(packing_files["device-packing.json"]),
                str(packing_files["device_public_pack.onnx"]),
                packing_pins["device-packing.json"]["sha256"],
                packing_pins["device_public_pack.onnx"]["sha256"]],
                cwd=source, environment=environment, seconds=300)
            receipt["device_packing_static_interop"] = {
                "assets": packing_pins, "scope": "python_artifact_rust_fixed_body_check_only",
                "rust_body_exit_status": 0, "native_validation": "not_performed",
                "provider_run_fence_authority": False,
                "registration_authenticity_authority": False}
            cli = [str(python), "-m", "rz_pals_model.cli"]
            initialization = output / "initialization"
            export = output / "export-pc"
            stages["initialize_seconds"] = run(cli + ["init", "--seed", str(args.seed),
                "--output", str(initialization)], cwd=source, environment=environment)
            checkpoint = initialization / "untrained.pt"
            if dataset_run:
                stages["preparation_check_seconds"] = run([
                    str(python), "-m", "rz_pals_model.preparation_check",
                    "--collection", str(dataset_run),
                    "--receipt-sha256", args.dataset_receipt_sha256,
                    "--encoder-source-sha256", args.dataset_encoder_source_sha256,
                    "--checkpoint", str(checkpoint),
                    "--output", str(output / "preparation-check.json"),
                    "--max-records", "256", "--max-wall-time-ms", "120000",
                    "--max-output-bytes", "1048576", "--threads", "2",
                    "--batch-size", "1", "--require-masked-value",
                ], cwd=source, environment=environment)
                if verifier_binary:
                    with checkpoint.open("rb") as stream:
                        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
                    stages["private_verifier_seconds"] = run([
                        str(python), "-B", "-m", "rz_pals_model.verifier_producer",
                        "--collection", str(dataset_run),
                        "--receipt-sha256", args.dataset_receipt_sha256,
                        "--encoder-source-sha256", args.dataset_encoder_source_sha256,
                        "--checkpoint", str(checkpoint), "--checkpoint-sha256", checkpoint_sha,
                        "--cpu-binary", str(verifier_binary),
                        "--cpu-binary-sha256", args.verifier_cpu_binary_sha256,
                        "--output", str(output / "private-verifier"),
                        "--allowed-task", "resume_task", "--max-games", "1", "--max-steps", "1",
                        "--max-nodes", "8192", "--max-wall-time-ms", "120000",
                        "--max-output-bytes", "4194304", "--max-forward-flops", "10000000000",
                        "--baseline-depth", "1", "--requested-depth", "2",
                        "--max-nodes-per-check", "4096", "--max-task-wall-time-ms", "5000",
                        "--max-cpu-output-bytes", "65536",
                    ], cwd=source, environment=environment, seconds=150)
            export_command = cli + ["export", "--checkpoint", str(checkpoint),
                "--output", str(export), "--layout", args.layout]
            if rules_profile:
                export_command += ["--rules-profile-json", str(rules_profile)]
            stages["export_seconds"] = run(export_command, cwd=source, environment=environment)
            stages["numeric_seconds"] = run(cli + ["numeric-check", "--checkpoint",
                str(checkpoint), "--export", str(export)], cwd=source, environment=environment)
            stages["rust_fixtures_seconds"] = run(cli + ["rust-fixtures", "--checkpoint",
                str(checkpoint), "--export", str(export), "--output", str(output / "rust-fixtures.json")],
                cwd=source, environment=environment)
            # Preserve one CPU reference runtime for the subsequent Rust reader
            # validation. This is a single pinned asset, not an arena-per-run copy.
            runtime_path = subprocess.check_output([str(python), "-c",
                "import onnxruntime,pathlib; p=pathlib.Path(onnxruntime.__file__).parent/'capi'; "
                "print(next(p.glob('libonnxruntime.so.1.22.0')))"],
                cwd=source, env=environment, text=True).strip()
            runtime_sha256 = hashlib.sha256(Path(runtime_path).read_bytes()).hexdigest()
            runtime_dir = (runtime_store / runtime_sha256) if runtime_store else output / "native-runtime"
            runtime_dir = checked(runtime_dir)
            runtime_dir.mkdir(parents=True, exist_ok=runtime_store is not None)
            runtime_asset = checked(runtime_dir / "libonnxruntime.so.1.22.0")
            if runtime_asset.exists():
                if hashlib.sha256(runtime_asset.read_bytes()).hexdigest() != runtime_sha256:
                    raise OSError("immutable CPU runtime store digest mismatch")
            else:
                # Exclusive create prevents replacing an already registered runtime.
                with open(runtime_path, "rb") as origin, runtime_asset.open("xb") as destination:
                    shutil.copyfileobj(origin, destination, 1024 * 1024)
            receipt["runtime"] = {"path": str(runtime_asset), "sha256": runtime_sha256,
                                  "bytes": runtime_asset.stat().st_size}
            if any(hashlib.sha256((source / relative).read_bytes()).hexdigest() != digest
                   for relative, digest in receipt["source_files"].items()):
                raise OSError("validation source changed during the registered run")
            if rules_profile and hashlib.sha256(rules_profile.read_bytes()).hexdigest() != receipt["rules_profile"]["sha256"]:
                raise OSError("registered Rules descriptor changed during validation")
            receipt["status"] = "success"
    except (OSError, subprocess.SubprocessError) as error:
        receipt["status"] = "failed"
        receipt["failure_type"] = type(error).__name__
        raise
    finally:
        receipt["total_seconds"] = time.monotonic() - started
        (output / "validation.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
