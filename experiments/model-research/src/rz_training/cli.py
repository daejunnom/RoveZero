"""CLI for explicitly scoped CPU fixture lifecycle checks."""

import argparse
import json
import os
from pathlib import Path

from rz_data.cli import _source_root
from rz_data.errors import DataError
from rz_data.io import read_json, write_run

from .checkpoint import seal
from .trainer import train_fixture, verify_export


def main(argv=None):
    parser = argparse.ArgumentParser(description="RoveZero internal CPU fixture training; no production model adapter")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="validate locked inputs, train or resume a CPU fixture")
    for name in ("recipe", "manifest", "records", "split-plan", "features"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--resume", type=Path)
    run.add_argument("--stop-after", type=int, help="pause after this invocation's completed steps; recipe remains locked")
    check = commands.add_parser("verify-export", help="re-import native fixture weights and compare saved outputs")
    check.add_argument("--export", dest="export_path", type=Path, required=True)
    for command in (run, check):
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    phase = "input"
    try:
        source_root = _source_root()
        if args.command == "run":
            recipe = read_json(args.recipe, max_bytes=65536)
            phase = "training"
            output, receipt = train_fixture(recipe, args.manifest, args.records, args.split_plan, args.features,
                                            output_root=args.output_root, run_id=args.run_id, source_root=source_root,
                                            resume_path=args.resume, stop_after=args.stop_after)
            print(json.dumps({"run": str(output), "status": receipt["status"], "steps": receipt["actual"]["steps"],
                              "execution_scope": "cpu_fixture", "execution_ready": False, "digest": receipt["digest"]}))
            return 1 if receipt["status"] in ("failed", "canceled", "time_budget_reached") else 0
        value = read_json(args.export_path, max_bytes=16777216)
        report = seal(verify_export(value), max_bytes=65536)
        phase = "output"
        output = write_run(args.output_root, args.run_id, {"verification.json": report}, source_root=source_root, max_bytes=65536)
        print(json.dumps({"run": str(output), "probes": report["probes"], "max_absolute_error": report["max_absolute_error"],
                          "engine_compatibility": "not_run", "digest": report["digest"]}))
        return 0
    except DataError as exc:
        print(json.dumps({"error": exc.as_dict()}))
        return 2
    except OSError as exc:
        print(json.dumps({"error": {"code": "IoFailure", "context": phase, "type": type(exc).__name__,
                                    "errno": exc.errno, "message": os.strerror(exc.errno) if exc.errno else "I/O failed"}}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
