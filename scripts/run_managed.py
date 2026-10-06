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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--timeout", type=int, default=1200)
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
    manager = ManagedBuild(canonical_root(), args.source_root)
    try:
        manager.acquire()
        try:
            environment = manager.environment()
        except StorageError:
            manager.finish(exit_code=125, tree_gone=True)  # No child was launched.
            raise

        def check_budget():
            if tree_size(manager.root / "slots", allow_disappearing=True) > manager.total_limit:
                raise StorageError("live managed build catalog exceeds 12 GiB")

        code, gone, failure = run_group(command, manager.source, environment, args.timeout, check_budget)
        receipt = manager.finish(exit_code=code, tree_gone=gone, supervisor_error=failure)
        print("managed_storage=" + json.dumps(receipt, sort_keys=True), file=sys.stderr)
        return code
    except (StorageError, OSError, RuntimeError) as exc:
        print(f"managed_storage_error={exc}; uncertain artifacts/leases are preserved", file=sys.stderr)
        return 125


if __name__ == "__main__":
    sys.exit(main())

