#!/usr/bin/env python3
"""Explicit bounded inventories and exact-owner recovery; never age-only delete."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from managed_storage import StorageError, checked, recover_build, tree_size
from managed_outputs import recover_output


def inventory(roots):
    selected = [checked(path) for path in roots]
    if len(selected) > 16 or any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                               for i, a in enumerate(selected) for b in selected[i + 1:]):
        raise StorageError("at most 16 disjoint explicit roots required")
    rows = []
    for root in selected:
        if not root.is_dir():
            raise StorageError("inventory requires an existing explicit root")
        children = list(root.iterdir())
        if len(children) > 1024:
            raise StorageError("inventory child bound exceeded")
        rows.append({"root": str(root), "length_bytes": tree_size(root),
            "children": [{"name": path.name, "length_bytes": tree_size(path),
                          "classification": "unclassified_preserved"} for path in sorted(children)]})
    return {"schema": "rovezero.storage-inventory.v1", "roots": rows,
            "sum_root_length_bytes": sum(row["length_bytes"] for row in rows),
            "cross_root_hardlink_deduplication": "not_performed",
            "whole_C_drive_or_VHD_physical_usage": "unknown", "deletion_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    scan = commands.add_parser("inventory")
    scan.add_argument("--root", type=Path, action="append", required=True)
    build = commands.add_parser("recover-build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--owner-sha256", required=True)
    output = commands.add_parser("recover-output")
    output.add_argument("--catalog", type=Path, required=True)
    output.add_argument("--key", required=True)
    output.add_argument("--record-sha256", required=True)
    args = parser.parse_args()
    if args.operation == "inventory":
        result = inventory(args.root)
    elif args.operation == "recover-build":
        result = recover_build(args.root, args.owner_sha256)
    else:
        result = recover_output(args.catalog, args.key, args.record_sha256)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, StorageError) as error:
        print(f"storage_maintenance_error={error}; unknown artifacts and leases preserved", file=sys.stderr)
        raise SystemExit(125)
