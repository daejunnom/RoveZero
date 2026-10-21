#!/usr/bin/env python3
"""Run builds/tests with bounded owned cache and post-run scratch cleanup."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from managed_storage import ManagedBuild, StorageError, canonical_root, tree_size
from managed_process import run_group
from managed_outputs import sample_output_budget


def _emit_storage_failure_observation(error):
    """Additional bounded JSON only; original failure/cleanup owns the result."""
    try:
        observation = getattr(error, "managed_output_identity_observation", None)
        if (type(observation) is not dict
                or observation.get("schema") != "rovezero.managed-output-identity-observation.v1"):
            return
        raw = json.dumps(observation, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(raw.encode("utf-8")) + 1 <= 32 << 10:
            print(raw, file=sys.stderr)
    except Exception:
        # A closed diagnostic pipe or encoding failure must not replace the
        # original StorageError, supervisor status or uncertain lease state.
        pass


def main(*, output_preservation=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--purpose", help="Stable diagnostic purpose slot; never a timestamp or PID")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.timeout <= 3600:
        parser.error("a command and timeout of 1..3600 seconds are required")
    # Windows CreateProcess does not perform POSIX PATH lookup for argv[0].
    program = shutil.which(command[0])
    if program is None or Path(program).suffix.lower() in {".bat", ".cmd"}:
        parser.error("command must resolve to an executable, not a shell batch file")
    command[0] = program
    budget_failure = None
    try:
        manager = ManagedBuild(canonical_root(), args.source_root, purpose=args.purpose,
                               output_preservation=output_preservation)
        manager.acquire()
        try:
            environment = manager.environment()
            if output_preservation is not None:
                manager.sample_output_budget()  # Must succeed before any payload child is launched.
        except StorageError:
            manager.finish(exit_code=125, tree_gone=True)  # No child was launched.
            raise

        def check_budget():
            nonlocal budget_failure
            try:
                if output_preservation is None:
                    sample_output_budget()  # Preserve the original default monkeypatch/consumer seam.
                else:
                    manager.sample_output_budget()
            except StorageError as exc:
                # run_group preserves the original failure string while it
                # drains children. Keep this original instance for diagnostics
                # after that same cleanup; do not alter its type or message.
                budget_failure = exc
                raise
            if tree_size(manager.root / "slots", allow_disappearing=True) > manager.total_limit:
                raise StorageError("live managed build catalog exceeds 12 GiB")

        code, gone, failure = run_group(command, manager.source, environment, args.timeout, check_budget)
        receipt = manager.finish(exit_code=code, tree_gone=gone, supervisor_error=failure)
        print("managed_storage=" + json.dumps(receipt, sort_keys=True), file=sys.stderr)
        if budget_failure is not None:
            _emit_storage_failure_observation(budget_failure)
        return code
    except (StorageError, OSError, RuntimeError) as exc:
        print(f"managed_storage_error={exc}; uncertain artifacts/leases are preserved", file=sys.stderr)
        if budget_failure is not None and budget_failure is not exc:
            _emit_storage_failure_observation(budget_failure)
        _emit_storage_failure_observation(exc)
        return 125


if __name__ == "__main__":
    sys.exit(main())
