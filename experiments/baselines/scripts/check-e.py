#!/usr/bin/env python3
"""Capture original output from bounded E01/E02 tool checks outside the checkout.

This script runs only when explicitly invoked. Reports contain command output and
metadata, including local paths; later sanitized exports must preserve these
originals. Dependency sources, build products and binaries are not copied into
the report directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone


TIMEOUT_SECONDS = 90
REAP_TIMEOUT_SECONDS = 5
MAX_LOCK_BYTES = 4 * 1024 * 1024
LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
HEAD_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
CHECKOUT = Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def planned_checks(cargo: str) -> list[tuple[str, list[str]]]:
    e01 = ["--manifest-path", "crates/rz-experiments/Cargo.toml"]
    arena = ["--manifest-path", "crates/rz-arena/Cargo.toml"]
    return [
        ("e01-fmt", [cargo, "fmt", *e01, "--all", "--", "--check"]),
        ("e01-test", [cargo, "test", *e01, "--all-targets", "--locked"]),
        (
            "e01-clippy",
            [cargo, "clippy", *e01, "--all-targets", "--locked", "--", "-D", "warnings"],
        ),
        ("arena-fmt", [cargo, "fmt", *arena, "--all", "--", "--check"]),
        (
            "arena-test",
            [cargo, "test", *arena, "--all-features", "--all-targets", "--locked"],
        ),
        (
            "arena-clippy",
            [
                cargo,
                "clippy",
                *arena,
                "--all-features",
                "--all-targets",
                "--locked",
                "--",
                "-D",
                "warnings",
            ],
        ),
        (
            "e01-msrv-1.85.0",
            [cargo, "+1.85.0", "check", *e01, "--all-targets", "--locked"],
        ),
        (
            "arena-msrv-1.90.0",
            [
                cargo,
                "+1.90.0",
                "check",
                *arena,
                "--all-features",
                "--all-targets",
                "--locked",
            ],
        ),
    ]


def version_commands(cargo: str) -> list[tuple[str, list[str]]]:
    cargo_path = Path(cargo)
    rustc_name = "rustc.exe" if cargo_path.name.lower().endswith(".exe") else "rustc"
    rustc = str(cargo_path.with_name(rustc_name))
    return [
        ("rustc-default-version", [rustc, "--version", "--verbose"]),
        ("rustc-1.85.0-version", [rustc, "+1.85.0", "--version", "--verbose"]),
        ("rustc-1.90.0-version", [rustc, "+1.90.0", "--version", "--verbose"]),
    ]


def snapshot_locks(directory: Path, phase: str) -> dict[str, dict[str, object]]:
    if phase not in ("start", "end"):
        raise ValueError("Cargo.lock snapshot phase must be start or end")
    target = directory / "inputs" / phase
    target.mkdir(parents=True)
    snapshots: dict[str, dict[str, object]] = {}
    try:
        for crate in ("rz-experiments", "rz-arena"):
            relative_source = Path("crates") / crate / "Cargo.lock"
            source = CHECKOUT / relative_source
            if not stat.S_ISREG(source.lstat().st_mode):
                raise ValueError(f"{relative_source} must be a regular file, not a symlink")
            with source.open("rb") as original:
                contents = original.read(MAX_LOCK_BYTES + 1)
            if len(contents) > MAX_LOCK_BYTES:
                raise ValueError(f"{relative_source} exceeds the snapshot byte bound")
            copied = target / f"{crate}.Cargo.lock"
            with copied.open("xb") as output:
                output.write(contents)
            snapshots[crate] = {
                "source": str(relative_source),
                "file": str(copied.relative_to(directory)),
                "sha256": hashlib.sha256(contents).hexdigest(),
                "bytes": len(contents),
            }
    except (OSError, ValueError) as error:
        write_json_new(
            directory / f"inputs-{phase}-failure.json",
            {"phase": phase, "error": str(error), "completed_snapshots": snapshots},
        )
        raise
    write_json_new(directory / f"inputs-{phase}.json", snapshots)
    return snapshots


def stop_process(process: subprocess.Popen[bytes]) -> str | None:
    """Kill the POSIX command group, then make a bounded attempt to reap it."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    except OSError as error:
        return f"cannot terminate command: {error}"
    try:
        process.wait(timeout=REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return "command could not be reaped within five seconds after termination"
    return None


def capture_command(
    directory: Path,
    name: str,
    argv: list[str],
    environment: dict[str, str],
    source_head: str | None,
) -> dict[str, object]:
    started = time.monotonic()
    record: dict[str, object] = {
        "name": name,
        "argv": argv,
        "cwd": str(CHECKOUT),
        "source_git_head": source_head,
        "started_utc": utc_now(),
        "timeout_seconds": TIMEOUT_SECONDS,
        "stdout": f"{name}.stdout.txt",
        "stderr": f"{name}.stderr.txt",
        "output_capture": "original_unmodified_bytes",
        "exit_code": None,
        "executed": False,
        "timed_out": False,
        "interrupted": False,
        "termination_scope": "process_group" if os.name == "posix" else "process",
    }
    with (directory / str(record["stdout"])).open("xb") as stdout:
        with (directory / str(record["stderr"])).open("xb") as stderr:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=CHECKOUT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=os.name == "posix",
                )
            except (OSError, ValueError) as error:
                record["status"] = "start_failed"
                record["error"] = str(error)
            else:
                record["executed"] = True
                try:
                    process.wait(timeout=TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    record["timed_out"] = True
                    record["status"] = "timed_out"
                    cleanup_error = stop_process(process)
                    if cleanup_error is not None:
                        record["cleanup_error"] = cleanup_error
                except KeyboardInterrupt:
                    record["interrupted"] = True
                    record["status"] = "interrupted"
                    cleanup_error = stop_process(process)
                    if cleanup_error is not None:
                        record["cleanup_error"] = cleanup_error
                else:
                    record["status"] = "passed" if process.returncode == 0 else "failed"
                record["exit_code"] = process.returncode
    record["elapsed_seconds"] = round(time.monotonic() - started, 6)
    record["finished_utc"] = utc_now()
    write_json_new(directory / f"{name}.json", record)
    return record


def prepare_output(artifact_root: str, label: str) -> tuple[Path, Path]:
    if LABEL_PATTERN.fullmatch(label) is None:
        raise ValueError("--label must be 1-80 ASCII letters/digits/._- and start with a letter/digit")
    root = Path(artifact_root).expanduser().resolve()
    if is_within(root, CHECKOUT):
        raise ValueError("--artifact-root must be outside the checkout")
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    target = root / "build"
    for directory in (reports, target):
        if directory.is_symlink() or not is_within(directory.resolve(), root):
            raise ValueError("report/build directories must remain within the artifact root")
        directory.mkdir(exist_ok=True)
    report = reports / label
    # No reuse, truncation, removal or automatic cleanup of previous reports.
    report.mkdir()
    return report, target


def run_checks(arguments: argparse.Namespace) -> int:
    report, target = prepare_output(arguments.artifact_root, arguments.label)
    checks = planned_checks(arguments.cargo)
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(target)
    common = {
        "schema": "rovezero-e-checks-v1",
        "label": arguments.label,
        "evidence_kind": "actual_tool_execution",
        "test_scope": "CPU tests including explicitly synthetic fixtures",
        "report_contents": [
            "original_stdout", "original_stderr", "command_metadata", "Cargo.lock_input_snapshots"
        ],
        "dependency_source_or_binary_payloads_in_report": False,
        "cargo_target_dir": str(target),
        "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json_new(
        report / "checks-planned.json",
        {
            **common,
            "created_utc": utc_now(),
            "checks": [{"name": name, "argv": argv} for name, argv in checks],
        },
    )
    start_locks = snapshot_locks(report, "start")
    source = capture_command(
        report, "source-head", ["git", "rev-parse", "--verify", "HEAD"], environment, None
    )
    source_head = None
    if source["status"] == "passed":
        raw_head = (report / "source-head.stdout.txt").read_bytes()
        try:
            candidate = raw_head.decode("ascii").strip() if len(raw_head) <= 128 else ""
        except UnicodeDecodeError:
            candidate = ""
        if HEAD_PATTERN.fullmatch(candidate) is not None:
            source_head = candidate
    results: list[dict[str, object]] = []
    versions: list[dict[str, object]] = []
    if source_head is not None:
        for name, argv in version_commands(arguments.cargo):
            version = capture_command(report, name, argv, environment, source_head)
            versions.append(version)
            if version["status"] != "passed":
                break
    versions_passed = len(versions) == 3 and all(version["status"] == "passed" for version in versions)
    if source_head is not None and versions_passed:
        for name, argv in checks:
            result = capture_command(report, name, argv, environment, source_head)
            results.append(result)
            print(f"{name}: {result['status']}", flush=True)
            if result["status"] != "passed":
                break
    end_locks = snapshot_locks(report, "end")
    locks_unchanged = all(
        start_locks[crate]["sha256"] == end_locks[crate]["sha256"]
        for crate in start_locks
    )
    completed = source_head is not None and len(results) == len(checks)
    passed = completed and versions_passed and locks_unchanged and all(result["status"] == "passed" for result in results)
    write_json_new(
        report / "checks-result.json",
        {
            **common,
            "source_git_head": source_head,
            "source_head_capture": source,
            "tool_versions": versions,
            "cargo_locks": {"start": start_locks, "end": end_locks, "unchanged": locks_unchanged},
            "status": "passed" if passed else "failed",
            "planned_checks": len(checks),
            "recorded_checks": len(results),
            "passed_checks": sum(result["status"] == "passed" for result in results),
            "checks": results,
            "finished_utc": utc_now(),
        },
    )
    print(f"original evidence: {report}", flush=True)
    if passed:
        return 0
    failed = results[-1] if results else (versions[-1] if versions else source)
    if failed.get("interrupted"):
        return 130
    if failed.get("timed_out"):
        return 124
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--cargo", default="cargo", help="Cargo/rustup proxy executable")
    parser.add_argument("--label", required=True, help="New unique report directory label")
    arguments = parser.parse_args()
    try:
        return run_checks(arguments)
    except (OSError, ValueError) as error:
        print(f"check-e: {error}; any existing partial report is preserved", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
