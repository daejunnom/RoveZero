#!/usr/bin/env python3
"""Bounded CPU-only PALS initialization/export validation; no training.

Run under run_managed.py so the dependency environment and pip scratch are
owned, bounded, and removed after all children exit. Preserve only the explicit
output directory outside the checkout. GPU validation is a different stage.
"""
from __future__ import annotations

import argparse
import json
import os
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
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**63 - 1:
        parser.error("seed must fit nonnegative signed int64")
    source = Path(__file__).resolve().parent.parent
    output = checked(args.output)
    if output.is_relative_to(source) or source.is_relative_to(output) or output.exists():
        parser.error("output must be a fresh directory outside the checkout")
    scratch = os.environ.get("TMPDIR")
    if not scratch or os.environ.get("CARGO_TARGET_DIR") is None:
        parser.error("run through scripts/run_managed.py with managed scratch")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "rovezero.pals-cpu-model-validation.v1", "status": "running",
               "training_executed": False, "training_steps": 0,
               "provider": "cpu", "gpu_validation": "not_run", "stages": {}}
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
            package = source / "experiments/model-research/pals"
            stages["reference_install_seconds"] = run([
                str(python), "-m", "pip", "install", "numpy==2.2.6", "onnx==1.19.0",
                "onnxruntime==1.22.0"], cwd=source, environment=environment)
            environment["PYTHONPATH"] = str(package / "src")
            stages["model_tests_seconds"] = run([
                str(python), "-m", "unittest", "discover", "-s", str(package / "tests"), "-v"],
                cwd=source, environment=environment)
            cli = [str(python), "-m", "rz_pals_model.cli"]
            initialization = output / "initialization"
            export = output / "export-pc"
            stages["initialize_seconds"] = run(cli + ["init", "--seed", str(args.seed),
                "--output", str(initialization)], cwd=source, environment=environment)
            checkpoint = initialization / "untrained.pt"
            stages["export_seconds"] = run(cli + ["export", "--checkpoint", str(checkpoint),
                "--output", str(export)], cwd=source, environment=environment)
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
            runtime_dir = output / "native-runtime"
            runtime_dir.mkdir(exist_ok=False)
            shutil.copyfile(runtime_path, runtime_dir / "libonnxruntime.so.1.22.0")
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
