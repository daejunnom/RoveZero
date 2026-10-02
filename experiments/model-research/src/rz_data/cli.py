"""Command line boundary for source splits and TASK-F01 structural audits."""

import argparse
import os
from pathlib import Path
import sys

from .audit import audit_dataset
from .errors import DataError
from .io import Limits, read_json, write_run
from .schema import choice, obj
from .serialization import canonical_bytes
from .splits import make_split_plan


def _source_root() -> Path:
    # Installed packages resolve against the invoked checkout, not site-packages.
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for path in (start, *start.parents):
            if (path / "AGENTS.md").is_file() and (path / "docs" / "TRAINING-PLAN.md").is_file():
                return path
    raise DataError("SourceRootUnknown", "output", "run from the RoveZero checkout to validate output location")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F01 source grouping and structural data audits (CPU only)")
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("split", help="freeze source groups before row generation")
    split.add_argument("--sources", type=Path, required=True)
    split.add_argument("--seed", required=True)
    split.add_argument("--ratios", type=int, nargs=3, default=(8, 1, 1), metavar=("TRAIN", "VALIDATION", "HOLDOUT"))
    validate = sub.add_parser("validate", help="audit rows against their immutable split plan")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--records", type=Path, required=True)
    validate.add_argument("--split-plan", type=Path, required=True)
    for command in (split, validate):
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--max-records", type=int, default=10000)
        command.add_argument("--max-record-bytes", type=int, default=262144)
        command.add_argument("--max-file-bytes", type=int, default=67108864)
        command.add_argument("--max-output-bytes", type=int, default=67108864)
    args = parser.parse_args(argv)
    phase = "limits"
    try:
        limits = Limits(args.max_records, args.max_record_bytes, args.max_file_bytes, args.max_output_bytes)
        if args.command == "split":
            phase = "read_sources"
            sources = read_json(args.sources, max_bytes=limits.max_file_bytes)
            obj(sources, "sources", ("schema_version", "sources"))
            choice(sources["schema_version"], "sources.schema_version", (1,))
            if not isinstance(sources["sources"], list) or len(sources["sources"]) > limits.max_records:
                raise DataError("SourceLimit", "sources", "source count exceeds configured record limit")
            phase = "split_sources"
            plan = make_split_plan(sources["sources"], seed=args.seed, ratios=tuple(args.ratios))
            if not plan["sources"]:
                raise DataError("EmptySources", "sources", "a split plan requires source games")
            phase = "write_split_plan"
            write_run(args.output_root, args.run_id, {"split-plan.json": plan}, source_root=_source_root(), max_bytes=limits.max_output_bytes)
            summary = {"split_plan_digest": plan["digest"], "source_games": len(plan["sources"]), "run_id": args.run_id}
            status = 0
        else:
            phase = "read_manifest"
            manifest = read_json(args.manifest, max_bytes=limits.max_file_bytes)
            phase = "read_split_plan"
            plan = read_json(args.split_plan, max_bytes=limits.max_file_bytes)
            phase = "audit_records"
            report = audit_dataset(manifest, args.records, plan, limits)
            phase = "write_audit"
            write_run(args.output_root, args.run_id, {"audit.json": report}, source_root=_source_root(), max_bytes=limits.max_output_bytes)
            summary = {"audit_digest": report["digest"], "structural_audit_passed": report["structural_audit_passed"],
                       "execution_ready": False, "counts": report["counts"], "run_id": args.run_id}
            status = 0 if report["structural_audit_passed"] else 1
        print(canonical_bytes(summary).decode("utf-8"))
        return status
    except DataError as exc:
        print(canonical_bytes({"error": exc.as_dict(), "execution_ready": False}).decode("utf-8"), file=sys.stderr)
        return 2
    except OSError as exc:
        # Keep private filesystem paths out of shared summaries.
        error = {"code": "IoFailure", "context": phase, "type": type(exc).__name__,
                 "errno": exc.errno, "message": os.strerror(exc.errno) if exc.errno is not None else "input/output failed"}
        print(canonical_bytes({"error": error, "execution_ready": False}).decode("utf-8"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
