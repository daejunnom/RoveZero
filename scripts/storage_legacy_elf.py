"""Bounded, explicit maintenance of unregistered legacy ELF leaves only.

The plan is current input, never a lease, adoption receipt or saved audit
authority. Audit is the default. Apply holds the original files, directories,
canonical build guard and output catalog guard until the operation ends.
The current visible PID namespace census has a non-atomic interval before
unlink; root-owned callers must exclude concurrent non-cooperating writers.
Logical removed bytes do not prove physical space or VHD space returned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

sys.dont_write_bytecode = True

from managed_storage import StorageError, canonical_root, read_record_snapshot
from managed_outputs import SCHEMA as OUTPUT_SCHEMA, MAX_RUNS, canonical_catalog
from storage_guard import CatalogGuard
import storage_artifact_review as review
import storage_code_retirement as core

PLAN_SCHEMA = "rovezero.legacy-elf-maintenance-plan.v1"
PROVENANCE_SCHEMA = "rovent.legacy-elf-summary/1"
PLAN_CAP = 128 << 10
PROVENANCE_CAP = 16 << 10
MAX_FILES = 8
ELF_CAP = 256 << 20
JOURNAL_CAP = 256 << 10
# Closed product names, rather than arbitrary regular files or suffix matches.
ELF_NAMES = frozenset(("rz-rovent-native", "rz-rovent-collect",
                       "rz-rovent-utility-trainer", "rz-rovent-trainer"))
ROOT_NAME = "rovent-four-mode-functional-20261018-001"
OBSOLETE_SOURCE = "90ebebc4d32b9dbaed1125085ba9ef54b4a839f6"
DUPLICATE_SOURCE = "e3b02d202b17fb1238a86cdaec2667872f90668e"
JOURNAL_DIRECTORY_CAP = 16 << 20
LIMITS = [*core._LIMITS, "unregistered_legacy_leaves_only",
          "catalog_lock_serializes_cooperating_producers_only",
          "obsolete_provenance_is_not_byte_exact_restoration"]


class LegacyElfUnsupported(StorageError):
    code = "LegacyElfUnsupported"


def _need(condition, detail):
    core._need(condition, detail)


def _pair(value):
    _need(type(value) is list and len(value) == 2
          and all(type(part) is int and 0 <= part < 1 << 128 for part in value),
          "exact current device/inode pair required")
    return tuple(value)


def _text(value, maximum=4096):
    _need(type(value) is str and 0 < len(value.encode("utf-8")) <= maximum
          and "\x00" not in value, "bounded original provenance text required")
    return value


def _raw_pin(value):
    return {name: value[name] for name in ("path", "bytes", "sha256")}


def _fd_pin(value):
    review.fields(value, ("path", "bytes", "sha256", "identity", "mtime_ns"))
    review.raw_pin(_raw_pin(value), ELF_CAP)
    core._path(value["path"])
    _pair(value["identity"])
    _need(type(value["mtime_ns"]) is int and 0 <= value["mtime_ns"] < 1 << 128,
          "exact current ELF mtime required")
    return value


def _overlap(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


def _allowed(root, item):
    """Closed build-specific paths; not a general regular-file deletion tool."""
    path = core._path(item["candidate"]["path"])
    _need(path.is_relative_to(root), "legacy candidate must be below declared ROOT")
    relative = path.relative_to(root)
    if item["mode"] == "obsolete":
        _need(relative.parent == Path("inputs/original-binaries-build-141")
              and relative.name in ELF_NAMES,
              "only explicit original build-141 product ELF basenames may be obsolete")
        _need(item["keeper"] is None and item["provenance"] is not None,
              "obsolete ELF requires original build provenance and no duplicate keeper")
    else:
        # Exact supported 202/209 path pairs are deliberately enumerated here.
        # An unknown layout stays unsupported until its explicit source scope
        # is reviewed; a filename substring is never deletion authorization.
        keeper = core._path(item["keeper"]["path"]) if item["keeper"] is not None else None
        pair = (relative.as_posix(), keeper.relative_to(root).as_posix()
                if keeper is not None and keeper.is_relative_to(root) else "")
        _need(pair in DUPLICATE_PAIRS and item["provenance"] is not None,
              "only the explicit legacy build-202 to build-209 duplicate pair is supported")
    return path


DUPLICATE_PAIRS = frozenset(("inputs/binaries/" + name + "-stripped-202",
                             "inputs/binaries/" + name + "-stripped-209")
                            for name in ELF_NAMES)


def _read_plan(inputs, plan_pin):
    stream = inputs.open(plan_pin, PLAN_CAP)
    plan = review.decode(stream.read(PLAN_CAP + 1))
    review.fields(plan, ("schema", "root", "catalog", "journal", "files"))
    _need(plan["schema"] == PLAN_SCHEMA, "closed legacy ELF maintenance plan required")
    for name in ("root", "catalog"):
        review.fields(plan[name], ("path", "identity"))
        core._path(plan[name]["path"])
        _pair(plan[name]["identity"])
    root, catalog = (core._path(plan[name]["path"]) for name in ("root", "catalog"))
    _need(root.name == ROOT_NAME and root.parent.name == "validation-assets",
          "only the explicitly approved validation-assets ROOT is supported")
    _need(catalog == core._path(canonical_catalog()), "current canonical output catalog required")
    review.fields(plan["journal"], ("path", "reserve_bytes", "directory_limit_bytes"))
    journal_path = core._path(plan["journal"]["path"])
    _need(journal_path.name == "legacy-elf-20261022-001" and journal_path.parent.name == "maintenance",
          "explicit original external legacy maintenance journal leaf required")
    reserve, limit = (plan["journal"][name] for name in ("reserve_bytes", "directory_limit_bytes"))
    _need(type(reserve) is int and 16384 <= reserve <= JOURNAL_CAP
          and type(limit) is int and reserve <= limit <= JOURNAL_DIRECTORY_CAP,
          "bounded external journal reserve and directory budget required")
    _need(not _overlap(root, journal_path) and not _overlap(catalog, journal_path),
          "legacy journal must be outside ROOT and output catalog")
    files = plan["files"]
    _need(type(files) is list and 0 < len(files) <= MAX_FILES,
          "finite explicit legacy ELF leaf selection required")
    candidates, keepers, provenance = [], [], []
    for item in files:
        review.fields(item, ("mode", "candidate", "keeper", "provenance"))
        _need(item["mode"] in ("duplicate", "obsolete"), "closed legacy ELF maintenance mode required")
        _fd_pin(item["candidate"])
        if item["keeper"] is not None:
            _fd_pin(item["keeper"])
            keepers.append(core._path(item["keeper"]["path"]))
        if item["provenance"] is not None:
            review.raw_pin(item["provenance"], PROVENANCE_CAP)
            provenance.append(core._path(item["provenance"]["path"]))
        candidates.append(_allowed(root, item))
    _need(len(set(candidates)) == len(candidates), "duplicate selected legacy leaf forbidden")
    _need(not set(candidates).intersection(keepers + provenance + [journal_path, core._path(plan_pin["path"])]),
          "selected legacy ELF overlaps keeper, provenance, plan or journal")
    return plan, root, catalog, candidates


def _open_elf(inputs, pin):
    path = core._path(pin["path"])
    inputs.directory(path.parent, controlled=True)
    stream = inputs.open(_raw_pin(pin), ELF_CAP)
    before = inputs.files[path]["before"]
    _need(core._inode(before) == _pair(pin["identity"]) and before.st_mtime_ns == pin["mtime_ns"],
          "expected current ELF FD identity/mtime changed")
    _need(stream.read(4) == b"\x7fELF", "legacy maintenance accepts actual ELF magic only")
    return path


def _same_bytes(inputs, left, right):
    _need(left != right, "duplicate keeper must be a different path")
    a, b = inputs.files[left], inputs.files[right]
    _need(core._inode(a["before"]) != core._inode(b["before"])
          and a["pin"]["bytes"] == b["pin"]["bytes"]
          and a["pin"]["sha256"] == b["pin"]["sha256"],
          "duplicate keeper must have exactly identical ELF bytes and a separate inode")
    offset = 0
    while offset < a["pin"]["bytes"]:
        inputs.tick()
        size = min(65536, a["pin"]["bytes"] - offset)
        left_raw = os.pread(a["stream"].fileno(), size, offset)
        right_raw = os.pread(b["stream"].fileno(), size, offset)
        _need(len(left_raw) == size and left_raw == right_raw,
              "stripped or different ELF bytes cannot be treated as a duplicate")
        offset += size
    inputs.check_file(left)
    inputs.check_file(right)


def _provenance(inputs, pin, item, root):
    raw = inputs.open(pin, PROVENANCE_CAP).read(PROVENANCE_CAP + 1)
    value = review.decode(raw)
    review.fields(value, ("schema", "authority", "current_required_inputs_retired", "duplicate_pairs",
                         "gpu_execution", "not_authority_for_registered_outputs_or_unknown_physical_material",
                         "obsolete_build", "old_results_promoted_to_current_acceptance", "optimizer_updates",
                         "provenance", "purpose"))
    _need(value["schema"] == PROVENANCE_SCHEMA
          and value["purpose"] == "user_authorized_obsolete_code_and_exact_duplicate_retirement"
          and value["current_required_inputs_retired"] is False and value["gpu_execution"] is False
          and value["old_results_promoted_to_current_acceptance"] is False
          and type(value["optimizer_updates"]) is int and value["optimizer_updates"] == 0
          and value["not_authority_for_registered_outputs_or_unknown_physical_material"] is True,
          "original bounded legacy provenance summary and non-authority flags required")
    build = value["obsolete_build"]
    review.fields(build, ("artifact_retention", "build_argv", "exact_byte_rebuild_guaranteed", "files",
                          "original_build_target", "original_resource_policy", "rebuild_description",
                          "retained_stripped_derivatives", "source", "status", "toolchain"))
    _need(build["source"] == OBSOLETE_SOURCE and build["status"] == "built_not_execution_admitted"
          and build["exact_byte_rebuild_guaranteed"] is False,
          "original build-141 source and no exact byte reconstruction claim required")
    for name in ("artifact_retention", "original_build_target", "rebuild_description"):
        _text(build[name])
    argv = build["build_argv"]
    _need(type(argv) is list and 0 < len(argv) <= 8
          and all(type(command) is list and 0 < len(command) <= 128
                  and all(type(part) is str and 0 < len(part) <= 4096 for part in command)
                  and command[:2] == ["cargo", "build"] and "--locked" in command for command in argv),
          "original bounded locked Cargo build argv required")
    toolchain = build["toolchain"]
    _need(type(toolchain) is dict and {"rustc", "cargo", "executables", "sysroot"} <= set(toolchain),
          "original Rust/Cargo/toolchain pins required")
    for name in ("rustc", "cargo"):
        _need(type(toolchain[name]) is dict and {"argv", "output"} <= set(toolchain[name]),
              "original toolchain command/output required")
        _text(toolchain[name]["output"])
    executables = toolchain["executables"]
    _need(type(executables) is list and len(executables) == 2,
          "original rustc and cargo executable pins required")
    for executable in executables:
        review.raw_pin(executable, ELF_CAP)  # Preserve pins only; never open toolchains.
    originals = build["files"]
    _need(type(originals) is list and len(originals) == 4, "four explicit original build-141 ELF pins required")
    original_by_path = {}
    for original in originals:
        review.fields(original, ("path", "bytes", "sha256", "source", "dependency_file"))
        review.raw_pin(_raw_pin(original), ELF_CAP)
        path = core._path(original["path"])
        _need(path.parent == root / "inputs/original-binaries-build-141" and path.name in ELF_NAMES
              and original["source"] == OBSOLETE_SOURCE and path not in original_by_path,
              "summary original ELF roster differs from approved build-141 scope")
        review.raw_pin(original["dependency_file"], 1 << 20)
        original_by_path[path] = original
    derivatives = build["retained_stripped_derivatives"]
    _need(type(derivatives) is list and len(derivatives) == 4,
          "four preserved strip-142 derivative pins required")
    derivative_paths = set()
    for derivative in derivatives:
        _need(type(derivative) is dict and {"path", "bytes", "sha256", "original", "source", "transformation"} <= set(derivative),
              "preserved strip-142 provenance link required")
        review.raw_pin(_raw_pin(derivative), ELF_CAP)
        path = core._path(derivative["path"])
        name = path.name.removesuffix("-stripped-142")
        _need(name in ELF_NAMES and path == root / "inputs/binaries" / (name + "-stripped-142")
              and derivative["original"] == original_by_path[root / "inputs/original-binaries-build-141" / name]
              and derivative["source"] == OBSOLETE_SOURCE and path not in derivative_paths,
              "preserved strip-142 derivative does not bind original ELF")
        derivative_paths.add(path)
    report_pins = value["provenance"]
    expected_reports = {"native-functional-build-141.json": (37859, "fd6965310452eff2f9dcc788db2ca2cee8159d6883e6e29ce88bfad863c3c278"),
                        "functional-binaries-stripped-142.json": (5039, "8feb25522192539d9468211b2249e98f86f0ea35e9bc1b6798f56bf7c35ca4c0"),
                        "functional-binaries-stripped-202.json": (5039, "96b273e952690ce91d17d2e3a9bb61b52db9e5474bc89ae781227a0a657748d1"),
                        "functional-binaries-stripped-209.json": (5039, "4daf0ddd417df6c044951d009ea68044b4309a8c46a5342f599d3dd68a857b07")}
    _need(type(report_pins) is dict and set(report_pins) == set(expected_reports),
          "exact original report provenance roster required")
    for name, report in report_pins.items():
        review.fields(report, ("path", "bytes", "sha256"))
        _text(report["path"])
        review.no_secret(report["path"])
        _need(report["path"].replace("\\", "/").rsplit("/", 1)[-1] == name
              and (report["bytes"], report["sha256"]) == expected_reports[name],
              "original immutable report pin differs")
    pairs = value["duplicate_pairs"]
    _need(type(pairs) is list and len(pairs) == 4, "four explicit duplicate provenance pairs required")
    seen = set()
    for pair in pairs:
        review.fields(pair, ("candidate", "keeper", "source"))
        for name in ("candidate", "keeper"):
            review.raw_pin(pair[name], ELF_CAP)
        candidate, keeper = (core._path(pair[name]["path"]) for name in ("candidate", "keeper"))
        _need(candidate.is_relative_to(root) and keeper.is_relative_to(root)
              and (candidate.relative_to(root).as_posix(), keeper.relative_to(root).as_posix()) in DUPLICATE_PAIRS
              and candidate not in seen and pair["source"] == DUPLICATE_SOURCE,
              "summary duplicate pair differs from approved build-202/209 scope")
        seen.add(candidate)
    candidate = item["candidate"]
    if item["mode"] == "obsolete":
        original = original_by_path.get(core._path(candidate["path"]))
        _need(original is not None and _raw_pin(original) == _raw_pin(candidate),
              "obsolete summary does not bind the exact current ELF bytes")
    else:
        _need(any(pair["candidate"] == _raw_pin(candidate) and pair["keeper"] == _raw_pin(item["keeper"])
                  for pair in pairs), "duplicate summary does not bind current candidate and keeper")
    return {"pin": pin, "raw_utf8": raw.decode("utf-8", errors="strict"),
            "byte_exact_restoration_attested": False}


def _roster(inputs, catalog):
    """Read raw rows only for DENY scopes; no closure or registry authority."""
    fd = inputs.directory(catalog, controlled=True)
    names = []
    with os.scandir(fd) as entries:
        for entry in entries:
            inputs.tick()
            _need(len(names) < MAX_RUNS + 1, "bounded output catalog roster required")
            names.append(entry.name)
    _need(len(set(names)) == len(names), "unique output catalog roster required")
    rows = []
    for name in sorted(names):
        if name == "catalog.lock":
            continue
        _need(re.fullmatch(r"[0-9a-f]{64}\.json", name) is not None,
              "unrecognized catalog entry; preserve legacy ELF")
        path = catalog / name
        original = read_record_snapshot(path, schema=OUTPUT_SCHEMA)
        row = original[0]
        _need(type(row.get("output")) is str, "registered output absolute scope required")
        target = core._path(row["output"])
        key = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()
        _need(name == key + ".json" and row.get("key") == key,
              "registered output path/key malformed; preserve legacy ELF")
        _need("identity" in row, "registered output identity field required")
        if row["identity"] is None:
            _need(row.get("deferred") is True, "only an explicitly deferred row may lack identity")
        else:
            _pair(row["identity"])  # Shape only; never claim old/current identity continuity.
        inputs.open({"path": str(path), "bytes": original[2][4], "sha256": original[1]}, 8192)
        rows.append((path, original, target))
    return rows


def _mounts(inputs):
    meter = {"bytes": 0}
    raw = core._proc_read(Path("/proc/self/mountinfo"), 1 << 20, inputs.tick, meter)
    table = {}
    for line in raw.decode("utf-8", errors="strict").splitlines():
        values = line.split()
        _need(len(values) >= 10 and "-" in values and values[0].isdigit()
              and len(table) < 16384, "bounded current mount table required")
        mount_id = int(values[0])
        _need(mount_id not in table, "unique current mount IDs required")
        def unescape(value):
            return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)
        table[mount_id] = (values[2], Path(unescape(values[3])), Path(unescape(values[4])))
    return raw, table


def _physical(inputs, path, table):
    existing = path
    held_file = None
    while True:
        inputs.tick()
        try:
            info = existing.lstat()
        except FileNotFoundError:
            _need(existing.parent != existing, "existing physical scope ancestor required")
            existing = existing.parent
            continue
        if stat.S_ISREG(info.st_mode) and existing in inputs.files:
            held_file = inputs.files[existing]["stream"].fileno()
        else:
            _need(stat.S_ISDIR(info.st_mode), "physical scope requires an original held directory/file")
        break
    fd = inputs.directory(existing) if held_file is None else held_file
    meter = {"bytes": 0}
    raw = core._proc_read(Path("/proc/self/fdinfo") / str(fd), 4096, inputs.tick, meter)
    ids = [line.split(b":", 1)[1].strip() for line in raw.splitlines() if line.startswith(b"mnt_id:")]
    _need(len(ids) == 1 and ids[0].isdigit() and int(ids[0]) in table,
          "actual held scope FD mount ID required")
    device, root, point = table[int(ids[0])]
    _need(path.is_relative_to(point), "current mount coordinate mismatch")
    if held_file is None:
        inputs.check_directory(existing)
    else:
        inputs.check_file(existing)
    return device, root / path.relative_to(point)


def _scope_check(inputs, selected_paths, protected):
    before, table = _mounts(inputs)
    for selected in selected_paths:
        dev, coordinate = _physical(inputs, selected, table)
        for target in protected:
            _need(not _overlap(selected, target), "legacy selection overlaps registered/protected path scope")
            other_dev, other_coordinate = _physical(inputs, target, table)
            _need(dev != other_dev or not _overlap(coordinate, other_coordinate),
                  "legacy selection aliases registered/protected physical scope")
    _need(_mounts(inputs)[0] == before, "current physical scope mount table changed")


def _journal_budget(inputs, value, written=0):
    path = core._path(value["path"])
    measured = core._tree_bytes(path.parent, inputs)
    _need(measured + max(0, value["reserve_bytes"] - written) <= value["directory_limit_bytes"],
          "external legacy journal directory budget exhausted")
    return measured


def maintain_legacy_elf(plan_pin, *, apply=False, seconds=300):
    """Fresh audit by default; explicit apply performs one bounded attempt."""
    if sys.platform != "linux":
        raise LegacyElfUnsupported("legacy ELF maintenance requires native Linux FD/proc observations")
    _need(type(apply) is bool, "explicit boolean legacy maintenance apply required")
    _need(os.geteuid() == 0, "legacy ELF maintenance requires UID 0 for a complete current census")
    inputs = core._Inputs(seconds)
    guards, journal = [], None
    first = result = None
    retired, removed, last_attempt, journal_path = [], 0, None, None
    evidence = {"schema": "rovezero.legacy-elf-maintenance-failure.v1", "plan": plan_pin,
                "status": "FAILED_OR_PARTIAL_PRESERVED", "retired_files": retired,
                "logical_removed_bytes": 0, "journal": None, "journal_path": None,
                "last_attempt": None, "partial": False, "physical_space_reclaimed": "unknown",
                "vhd_return": "unknown", "physical_completion_attested": False,
                "observation_limits": list(LIMITS)}
    try:
        plan, root, catalog, candidates = _read_plan(inputs, plan_pin)
        build = core._path(canonical_root())
        for directory in (root, build, catalog):
            inputs.directory(directory, controlled=True)
        _need(root != build and not root.is_relative_to(build), "legacy payload may not be a managed build slot")
        for name, path in (("root", root), ("catalog", catalog)):
            _need(core._inode(inputs.directories[path][1]) == _pair(plan[name]["identity"]),
                  "declared ROOT/catalog current directory identity changed")
        for directory in (build, catalog):
            guard = CatalogGuard(directory / "catalog.lock")
            guards.append(guard)
            guard.acquire()
        def no_owner():
            try:
                os.stat("owner.json", dir_fd=inputs.directories[build][0], follow_symlinks=False)
            except FileNotFoundError:
                return
            raise StorageError("canonical build owner is present; legacy maintenance is forbidden")
        no_owner()
        rows = _roster(inputs, catalog)
        selected_parents = sorted({path.parent for path in candidates}, key=str)
        registry = catalog.with_name(catalog.name + ".interrupted-unbound-custody-v1")
        protected = [catalog, build, registry, *(target for _, _, target in rows)]
        try:
            registry_before = core._file_identity(registry.lstat())
        except FileNotFoundError:
            registry_before = None
        journal_path = core._path(plan["journal"]["path"])
        evidence["journal_path"] = str(journal_path)
        # A journal may not write into any protected existing row or registry.
        _scope_check(inputs, [journal_path.parent], [root, *protected])
        provenance, duplicates = {}, []
        for item, path in zip(plan["files"], candidates):
            _open_elf(inputs, item["candidate"])
            inputs.candidates.add(path)
            if item["mode"] == "duplicate":
                keeper = _open_elf(inputs, item["keeper"])
                _need(keeper not in candidates, "duplicate keeper must be outside the selected deletion leaves")
                _same_bytes(inputs, path, keeper)
                duplicates.append((path, keeper))
            summary_path = core._path(item["provenance"]["path"])
            _need(not _overlap(summary_path, root) and not _overlap(summary_path, catalog),
                  "provenance keeper must be external to ROOT and catalog")
            summary = _provenance(inputs, item["provenance"], item, root)
            provenance[str(summary_path)] = summary
        _scope_check(inputs, candidates, protected)
        _scope_check(inputs, candidates, [core._path(value["pin"]["path"]).parent
                                         for value in provenance.values()])
        target_inodes = {core._inode(inputs.files[path]["before"]) for path in candidates}
        _need(len(target_inodes) == len(candidates), "selected legacy ELF inode aliases forbidden")
        _need(all(core._inode(item["before"]) not in target_inodes
                  for path, item in inputs.files.items() if path not in inputs.candidates),
              "legacy selected inode overlaps keeper or control")
        _need(all(core._inode(os.fstat(guard.stream.fileno())) not in target_inodes for guard in guards),
              "selected legacy ELF overlaps actual catalog guard inode")
        targets = target_inodes | {core._inode(inputs.directories[path][1]) for path in selected_parents}
        def recheck():
            inputs.check()
            for held_guard in guards:
                held_guard.assert_held()
            no_owner()
            _need(_roster(inputs, catalog) == rows, "registered catalog raw row or roster changed")
            try:
                registry_now = core._file_identity(registry.lstat())
            except FileNotFoundError:
                registry_now = None
            _need(registry_now == registry_before, "special custody registry root changed; no interpretation or adoption")
            _scope_check(inputs, [path for path in candidates if path not in inputs.deleted], protected)
            for summary in provenance.values():
                inputs.hash_file(core._path(summary["pin"]["path"]))
            if journal is not None:
                journal.check()
        def census():
            excluded = inputs.fds() | {guard.stream.fileno() for guard in guards}
            if journal is not None:
                excluded.add(journal.fd)
            return core._process_census(targets, excluded, inputs.tick)
        recheck()
        observed = census()
        recheck()
        header = {"schema": "rovezero.legacy-elf-maintenance-journal.v1", "event": "preflight_complete",
                  "plan": plan_pin, "files": plan["files"], "provenance": list(provenance.values()),
                  "root": plan["root"], "catalog": plan["catalog"], "census": observed,
                  "observations_only": True, "registered_records_changed": False,
                  "physical_completion_attested": False, "observation_limits": list(LIMITS)}
        _need(len(core._encode(header)) + len(candidates) * 1024 + 16384 <= plan["journal"]["reserve_bytes"],
              "legacy journal reserve too small for complete provenance and partial evidence")
        _journal_budget(inputs, plan["journal"])
        if apply:
            journal = core._Journal(inputs, journal_path.parent, journal_path.name,
                                    plan["journal"]["reserve_bytes"])
            journal.append(header)
            os.fsync(inputs.directory(journal_path.parent))
            for index, path in enumerate(candidates):
                recheck()
                _journal_budget(inputs, plan["journal"], journal.written)
                last_attempt = {"index": index, "path": str(path), "unlink_attempted": False,
                                "unlink_returned": False, "directory_fsynced": False}
                journal.append({"event": "unlink_intent", "index": index})
                recheck()
                inputs.hash_file(path)
                for candidate, keeper in duplicates:
                    if candidate == path:
                        inputs.hash_file(keeper)
                        _same_bytes(inputs, candidate, keeper)
                census()
                recheck()
                last_attempt["unlink_attempted"] = True
                os.unlink(path.name, dir_fd=inputs.directories[path.parent][0])
                last_attempt["unlink_returned"] = True
                inputs.deleted.add(path)
                retired.append(str(path))
                removed += plan["files"][index]["candidate"]["bytes"]
                os.fsync(inputs.directories[path.parent][0])
                last_attempt["directory_fsynced"] = True
                inputs.check_file(path)
                journal.append({"event": "unlinked_directory_fsynced", "index": index,
                                "logical_removed_bytes": plan["files"][index]["candidate"]["bytes"]})
            recheck()
            journal.append({"event": "complete", "retired_files": len(retired),
                            "logical_removed_bytes": removed})
        result = {"schema": "rovezero.legacy-elf-maintenance-result.v1",
                  "status": "LEGACY_ELF_RETIRED" if apply else "LEGACY_ELF_AUDITED",
                  "plan": plan_pin, "deletion_performed": bool(retired),
                  "retired_files": list(retired), "logical_removed_bytes": removed,
                  "journal": journal.pin() if journal is not None else None,
                  "journal_path": str(journal_path), "provenance": list(provenance.values()),
                  "census": observed, "registered_records_changed": False,
                  "owner_lease_or_adoption_issued": False, "reservations_changed": False,
                  "physical_completion_attested": False, "physical_space_reclaimed": "unknown",
                  "vhd_return": "unknown", "byte_exact_obsolete_restoration_attested": False,
                  "observation_limits": list(LIMITS)}
    except BaseException as error:
        first = error
        if journal is not None:
            try:
                journal.append({"event": "failed_or_partial_preserved", "type": type(error).__name__[:128],
                                "detail": core._describe(error), "retired_files": len(retired),
                                "logical_removed_bytes": removed, "last_attempt": last_attempt})
            except BaseException as secondary:
                core._note(first, "legacy partial journal append/fsync failed: " + core._describe(secondary))
    finally:
        evidence.update(logical_removed_bytes=removed, last_attempt=last_attempt,
                        partial=bool(retired) or bool(last_attempt and last_attempt["unlink_attempted"]))
        if journal is not None:
            try:
                evidence["journal"] = journal.pin()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    core._note(first, "legacy journal pin failed: " + core._describe(error))
            try:
                journal.close()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    core._note(first, "legacy journal close failed: " + core._describe(error))
        first = inputs.close(first)
        for guard in reversed(guards):
            try:
                guard.release()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    core._note(first, "legacy guard release failed: " + core._describe(error))
    if first is not None:
        try:
            first.legacy_elf_maintenance = evidence
        except BaseException:
            pass
        core._note(first, "legacy attempt preserved; journal=" + str(evidence["journal_path"])
                   + "; logical_removed_bytes=" + str(removed) + "; no automatic retry")
        raise first
    return result


def audit_legacy_elf(plan_pin, *, seconds=300):
    return maintain_legacy_elf(plan_pin, seconds=seconds)


def retire_legacy_elf(plan_pin, *, seconds=300):
    return maintain_legacy_elf(plan_pin, apply=True, seconds=seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-bytes", type=int, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        result = maintain_legacy_elf({"path": args.plan, "bytes": args.plan_bytes,
                                     "sha256": args.plan_sha256}, apply=args.apply, seconds=args.seconds)
    except BaseException as error:
        try:
            evidence = getattr(error, "legacy_elf_maintenance", None)
            if evidence is not None:
                print("legacy_elf_maintenance_partial=" + core._encode(evidence).decode("ascii").rstrip(),
                      file=sys.stderr)
            print("legacy_elf_maintenance_error=" + core._describe(error), file=sys.stderr)
        except BaseException:
            pass
        return 125
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
