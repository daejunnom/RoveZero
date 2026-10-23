"""Explicit one-row bound-output maintenance with the original source7 view.

This is separate from legacy/sealed/unbound recovery. Historical tips remain
history; no new transition, owner, output lease or physical fence is issued.
An independently managed research ZIP must exist before this API is called.
The caller retains that archive under its own registered storage lifecycle.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
import zipfile

import managed_output_quarantine as custody
import managed_output_unattributed_preservation as preservation
import managed_outputs as outputs
from managed_storage import GiB, StorageError, checked, read_record_snapshot, write_record
from storage_guard import CatalogGuard, lifetime, reboot_proves_dead

ARCHIVE_SCHEMA = "rovezero.bound-output-recovery-archive.v1"
RECEIPT_SCHEMA = "rovezero.bound-output-recovery.v1"
ORIGINAL_MEMBER = "original/catalog-row.json"
MARKER = ".rovezero-output.json"


def _need(value, message):
    if value is not True:
        raise StorageError(message)


def _remaining(until):
    remaining = until - time.monotonic()
    _need(remaining > 0, "original bound recovery deadline expired")
    return remaining


def _eligible(row, key, digest, expected, target, now):
    _need(digest == expected, "original bound catalog raw digest changed")
    identity = row.get("identity")
    _need(row.get("schema") == outputs.SCHEMA and row.get("key") == key
        and row.get("output") == str(target) and row.get("state") == "active"
        and row.get("deferred") is True and type(identity) is list and len(identity) == 2
        and all(type(item) is int for item in identity) and identity[0] >= 0 and identity[1] > 0,
        "only the exact active sealed deferred row is eligible")
    _need(type(row.get("pid")) is int and row["pid"] > 0
        and type(row.get("token")) is str and bool(row["token"])
        and row.get("outer_supervisor") is None,
        "original independent producer attribution required; supervised rows are separate")
    _need(type(row.get("reserved_bytes")) is int and type(row.get("catalog_limit_bytes")) is int
        and 0 < row["reserved_bytes"] <= row["catalog_limit_bytes"] <= 18 * GiB,
        "original positive reservation and catalog limit required")
    _need("tree_gone" not in row or row["tree_gone"] is False,
        "bound recovery cannot assert original tree completion")
    _need(row.get("physical_completion") in (None, "unknown", "not_inferred",
        "not_inferred_from_output_lifecycle", "unknown_unchanged"),
        "bound recovery cannot adopt physical completion")
    # Same-boot PID absence/reuse is deliberately not queried. Only actual
    # same-host/platform reboot proves that original processes cannot survive.
    _need(reboot_proves_dead(row.get("lifetime"), now),
        "actual same-host real reboot required; same-boot process absence is insufficient")


def _snapshots(rows, until):
    snapshots = {}
    for path, row, _ in rows:
        _remaining(until)
        snapshot = read_record_snapshot(path, schema=outputs.SCHEMA)
        info = path.lstat()
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            and info.st_uid == os.geteuid() and snapshot[0] == row
            and row["key"] not in snapshots, "original catalog raw/owner/roster changed")
        snapshots[row["key"]] = snapshot
    return snapshots


def _marker(held, tree, original_raw):
    held.tick()
    try:
        info = os.stat(MARKER, dir_fd=tree.root, follow_symlinks=False)
    except FileNotFoundError:
        return {"state": "absent"}
    _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and MARKER in tree.files,
        "existing deferred marker must remain an original plain file")
    fd, before = tree.files[MARKER]
    _need(custody._stamp(info) == before and held.bytes(fd, before[6]) == original_raw,
        "existing marker differs from original catalog raw")
    return {"state": "present", "member": "corpus/" + MARKER}


def _archive(held, archive_pin, row_pin, row, original_raw, tree, marker):
    """Verify every original byte, never accept a manifest's success claim."""
    fd, before = held.open_file(Path(archive_pin["path"]), preservation.MAX_ARCHIVE,
        expected=archive_pin)
    _need(held.digest(fd, archive_pin["bytes"]) == archive_pin["sha256"],
        "independent research archive raw SHA changed")
    preservation._zip_metadata_preflight(held, fd, archive_pin["bytes"])
    with os.fdopen(os.dup(fd), "rb") as stream, zipfile.ZipFile(stream) as archive:
        members, expanded = {}, 0
        for item in archive.infolist():
            held.tick()
            name = preservation._member(item.filename)
            kind = stat.S_IFMT(item.external_attr >> 16)
            _need(item.orig_filename == item.filename
                and name not in members and not item.is_dir() and not item.flag_bits & 1
                and item.create_system in (0, 3) and not item.external_attr & 0x10
                and ((item.create_system == 3 and kind in (0, stat.S_IFREG))
                     or (item.create_system == 0 and kind == 0)),
                "plain unique unencrypted research ZIP members required")
            members[name] = item
            expanded += item.file_size
            _need(len(members) <= preservation.MAX_ENTRIES + 1
                and expanded <= row["reserved_bytes"] + preservation.MAX_MANIFEST + preservation.MAX_METADATA,
                "bounded complete research archive required")
        _need("inventory.json" in members
            and members["inventory.json"].file_size <= preservation.MAX_MANIFEST,
            "bounded independent recovery inventory required")
        document = preservation._decode(archive.read(members["inventory.json"]))
        preservation._fields(document, ("schema", "original_catalog_pin", "original_catalog_member",
            "output", "marker", "directories", "files"))
        directories = sorted(item["name"] for item in tree.entries if stat.S_ISDIR(item["stat"][2]))
        _need(document["schema"] == ARCHIVE_SCHEMA and document["original_catalog_pin"] == row_pin
            and document["original_catalog_member"] == ORIGINAL_MEMBER
            and document["output"] == str(tree.target) and document["marker"] == marker
            and document["directories"] == directories,
            "archive original row/marker/directory join changed")
        _need(ORIGINAL_MEMBER in members
            and members[ORIGINAL_MEMBER].file_size == len(original_raw)
            and archive.read(members[ORIGINAL_MEMBER]) == original_raw,
            "independent original catalog raw bytes differ")
        files = document["files"]
        _need(type(files) is list and len(files) == len(tree.files),
            "every actual bound output file must be independently preserved")
        listed, relatives, hashes = {"inventory.json", ORIGINAL_MEMBER}, [], {}
        for item in files:
            held.tick()
            preservation._fields(item, ("relative", "member", "bytes", "sha256"))
            relative, member = preservation._member(item["relative"]), preservation._member(item["member"])
            _need(relative in tree.files and member == "corpus/" + relative
                and member in members and member not in listed,
                "archive corpus member missing/duplicated")
            source, original = tree.files[relative]
            _need(type(item["bytes"]) is int and item["bytes"] == original[6] == members[member].file_size
                and type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
                "archive file length/hash declaration changed")
            digest, count = hashlib.sha256(), 0
            with archive.open(members[member]) as part:
                while True:
                    held.tick()
                    raw = part.read(65536)
                    if not raw:
                        break
                    count += len(raw)
                    _need(count <= original[6], "archive member exceeded original bound")
                    digest.update(raw)
            _need(count == original[6] and held.digest(source, original[6])
                == digest.hexdigest() == item["sha256"], "archive and actual bound output bytes differ")
            relatives.append(relative)
            hashes[relative] = item["sha256"]
            listed.add(member)
        _need(relatives == sorted(tree.files) and set(members) == listed,
            "complete sorted archive roster required; extra members forbidden")
    _need(custody._stamp(os.fstat(fd)) == before, "independent archive FD changed")
    return fd, before, hashes


def _scope(rows, key, tree, archive_path, catalog, selection, until):
    protected = [catalog, custody.registry_path(catalog), *(Path(pin["path"]) for pin in selection.sources.values()),
        *(Path(pin["path"]) for pin in selection.tips),
        *(Path(item["research_archive"]["path"]) for item in selection.rows)]
    protected += [target for _, row, target in rows if row["key"] != key]
    target_ids = {tuple(item["stat"][:2]) for item in tree.entries}
    before, table = preservation._mount_table()
    target_scope = preservation._physical_scope(tree.target, table)
    archive_scope = preservation._physical_scope(archive_path, table)
    def overlaps(a, b):
        return a.is_relative_to(b) or b.is_relative_to(a)
    def aliases(a, b):
        return a[0] == b[0] and overlaps(a[1], b[1])
    _need(not overlaps(tree.target, archive_path) and not aliases(target_scope, archive_scope),
        "research archive must be independent of original output")
    for other in protected:
        _remaining(until)
        _need(not overlaps(tree.target, other) and not overlaps(archive_path, other),
            "recovery output/archive overlaps protected catalog, source, history or output")
        other_scope = preservation._physical_scope(other, table)
        _need(not aliases(target_scope, other_scope) and not aliases(archive_scope, other_scope),
            "recovery output/archive aliases protected evidence")
        try:
            info = other.lstat()
        except FileNotFoundError:
            continue
        _need(tuple(custody._stamp(info)[:2]) not in target_ids,
            "bound output inode aliases protected evidence")
    after, _ = preservation._mount_table()
    _need(before == after, "bound recovery mount table changed")


def _close(resources, primary):
    first = primary
    for resource in reversed(resources):
        try:
            resource.close(first)
        except BaseException as error:
            if first is None:
                first = error
            else:
                _note(first, "bound recovery descriptor cleanup also failed: ", error)
    if primary is None and first is not None:
        raise first


def _note(primary, message, secondary=None):
    # Only diagnostics while an original failure already exists are isolated.
    try:
        primary.add_note(message + (repr(secondary) if secondary is not None else ""))
    except BaseException:
        pass


def _failure(primary, receipt, status, **fields):
    try:
        receipt.update(status=status, **fields)
        primary.bound_recovery_receipt = copy.deepcopy(receipt)
    except BaseException:
        pass


@contextmanager
def _gate(catalog, receipt):
    primary = None
    try:
        with custody.original_custody_gate(catalog) as guard:
            try:
                yield guard
            except BaseException as error:
                primary = error
                raise
    except BaseException as error:
        if primary is not None and error is not primary:
            _note(primary, "original catalog gate cleanup/diagnostic also failed: ", error)
            raise primary
        if primary is None:
            _failure(error, receipt, "catalog_gate_failure_preserved", error_type=type(error).__name__)
        raise


def recover_bound_output_after_reboot(catalog, key, expected_sha256, *,
        unattributed_selection, research_archive_pin,
        root_exclusive_writer=False, apply=False, maximum_seconds=30):
    """Validate/dry-run or preserve precisely one independent sealed row.

    Only the catalog row is atomically replaced. Its original failure fields,
    reservation, target, marker and all payload remain unchanged. An existing
    marker is preserved as historical raw, not rewritten to the new metadata.
    After any commit-stage error inspect both canonical row and original ZIP;
    the exception carries bound_recovery_receipt and no rollback is attempted.
    """
    _need(root_exclusive_writer is True and type(apply) is bool,
        "explicit exclusive Root maintenance and exact apply boolean required")
    _need(type(maximum_seconds) in (int, float) and math.isfinite(maximum_seconds)
        and 0 < maximum_seconds <= 30, "one finite bound recovery window <=30s required")
    _need(type(key) is str and re.fullmatch(r"[0-9a-f]{64}", key) is not None
        and type(expected_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
        "exact original bound catalog key and raw SHA required")
    _need(type(unattributed_selection) is preservation.UnattributedSelection,
        "canonical typed source7 selection required; JSON grants no authority")
    archive_pin = preservation._pin(research_archive_pin, preservation.MAX_ARCHIVE)
    _need(sys.platform == "linux", "bound custody recovery requires original Linux observations")
    until = time.monotonic() + min(maximum_seconds, unattributed_selection.maximum_seconds)
    if unattributed_selection._bound_deadline is not None:
        until = min(until, unattributed_selection.original_hard_until)
    catalog = checked(Path(catalog))
    _need(catalog == checked(outputs.canonical_catalog()), "original canonical source7 catalog required")
    info, lock_info = catalog.lstat(), (catalog / "catalog.lock").lstat()
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
        and not stat.S_IMODE(info.st_mode) & 0o022
        and stat.S_ISREG(lock_info.st_mode) and lock_info.st_uid == os.geteuid()
        and lock_info.st_nlink == 1, "existing owner-controlled original catalog/gate required")
    receipt = {"schema": RECEIPT_SCHEMA, "status": "not_committed", "applied": False,
        "original_catalog_key": key, "original_catalog_sha256": expected_sha256,
        "research_archive": copy.deepcopy(archive_pin), "policy_seconds": maximum_seconds,
        "selected_history_modified": False, "new_owner_or_lease_issued": False,
        "physical_completion": "unknown_unchanged", "payload_modified": False,
        "catalog_write_attempted": False, "catalog_write_returned": False,
        "canonical_readback_verified": False, "performance_evidence": False}
    with _gate(catalog, receipt) as guard:
        resources = []
        try:
            _need(type(guard) is CatalogGuard and guard.path == catalog / "catalog.lock",
                "actual original canonical catalog guard required")
            guard.assert_held()
            _need(custody._did(catalog.lstat()) == custody._did(info)
                and custody._stamp(os.fstat(guard.stream.fileno())) == custody._stamp(lock_info),
                "original catalog/gate changed before acquisition")
            view = preservation.observe_catalog(catalog, unattributed_selection, guard, hard_until=until)
            resources.append(view)
            _need(type(view) is preservation.CatalogView and view.guard is guard
                and view.catalog == catalog and view.selection is unattributed_selection,
                "same actual source7 CatalogView/selection/guard required")
            _need(key not in view.keys, "selected/history original rows are immutable")
            rows = view.rows()
            snapshots = _snapshots(rows, until)
            _need(key in snapshots, "exact original bound row not found")
            row, digest, record_stamp = snapshots[key]
            target, now = checked(Path(row["output"])), lifetime()
            _eligible(row, key, digest, expected_sha256, target, now)
            row_path = catalog / (key + ".json")
            row_held = preservation._Held(_remaining(until), until)
            resources.append(row_held)
            row_pin = {"path": str(row_path), "bytes": row_path.lstat().st_size, "sha256": digest}
            original_raw = row_held.read(row_pin, preservation.MAX_METADATA, root_owned=True)
            _need(preservation._decode(original_raw) == row
                and read_record_snapshot(row_path, schema=outputs.SCHEMA) == (row, digest, record_stamp),
                "held original bound row raw/FD changed")
            held = preservation._Held(_remaining(until), until)
            resources.append(held)
            catalog_fd = held.directory(catalog)
            tree = preservation._Tree(held, target, row["reserved_bytes"])
            _need(list(tree.identity[:2]) == row["identity"], "original sealed inode changed")
            for fd in (tree.parent, tree.root):
                entry = os.fstat(fd)
                _need(entry.st_uid == os.geteuid() and not stat.S_IMODE(entry.st_mode) & 0o077,
                    "private owner-controlled original parent/target required")
            _need(all(entry["stat"][3] == os.geteuid() for entry in tree.entries),
                "bound output file/directory owner changed")
            mounts = custody._mounts(held, target.parent, target, tree.identity[0])
            marker = _marker(held, tree, original_raw)
            _scope(rows, key, tree, Path(archive_pin["path"]), catalog, unattributed_selection, until)
            archive_fd, archive_stamp, payload_hashes = _archive(held, archive_pin,
                row_pin, row, original_raw, tree, marker)
            receipt.update(original_catalog_pin=row_pin, original_catalog_stat=list(record_stamp),
                original_output_identity=list(row["identity"]), marker=copy.deepcopy(marker),
                archive_all_original_bytes_verified=True)

            def recheck(*, replacement=None):
                _remaining(until)
                guard.assert_held()
                _need(lifetime() == now, "bound recovery observer lifetime changed")
                _need(custody._did(os.fstat(catalog_fd)) == custody._did(info)
                    == custody._did(checked(catalog).lstat()), "original catalog FD/name changed")
                held.check()
                _need(custody._mounts(held, target.parent, target, tree.identity[0]) == mounts,
                    "bound recovery namespace/mount alias changed")
                _need(_marker(held, tree, original_raw) == marker, "original marker state changed")
                for relative, digest_value in payload_hashes.items():
                    fd, original = tree.files[relative]
                    _need(held.digest(fd, original[6]) == digest_value,
                        "original output bytes changed before/after commit")
                _need(held.digest(archive_fd, archive_pin["bytes"]) == archive_pin["sha256"]
                    and custody._stamp(os.fstat(archive_fd)) == archive_stamp,
                    "verified independent archive changed before/after commit")
                current_rows, used = view.checked_usage()
                current = _snapshots(current_rows, until)
                _need(set(current) == set(snapshots), "original catalog row roster changed")
                replacement_held = None
                for other_key, snapshot in snapshots.items():
                    if replacement is not None and other_key == key:
                        encoded = (json.dumps(replacement, sort_keys=True) + "\n").encode()
                        _need(current[key][0] == replacement
                            and current[key][1] == hashlib.sha256(encoded).hexdigest(),
                            "bound recovery canonical replacement readback changed")
                        replacement_held = preservation._Held(_remaining(until), until)
                        resources.append(replacement_held)
                        replacement_pin = {"path": str(row_path), "bytes": len(encoded),
                            "sha256": hashlib.sha256(encoded).hexdigest()}
                        _need(replacement_held.read(replacement_pin, preservation.MAX_METADATA,
                            root_owned=True) == encoded, "new replacement raw FD differs")
                    else:
                        _need(current[other_key] == snapshot, "original catalog raw/hash/stat changed")
                if replacement is None:
                    row_held.check()
                    _eligible(row, key, digest, expected_sha256, target, lifetime())
                _scope(current_rows, key, tree, Path(archive_pin["path"]), catalog,
                    unattributed_selection, until)
                _need(used + archive_pin["bytes"] <= min(row["catalog_limit_bytes"], 18 * GiB)
                    and tree.payload <= row["reserved_bytes"],
                    "original aggregate catalog/archive/reservation budget exceeded")
                held.check()
                view.check()
                if replacement_held is not None:
                    replacement_held.check()
                _remaining(until)
                guard.assert_held()
                return used + archive_pin["bytes"]

            proposed = dict(row, state="interrupted_retained", recovery_reason="verified_same_host_reboot_source7",
                recovered_ns=time.time_ns(), recovery_source_sha256=expected_sha256,
                bound_recovery={"schema": RECEIPT_SCHEMA, "research_archive": copy.deepcopy(archive_pin),
                    "marker": copy.deepcopy(marker), "physical_completion": "unknown_unchanged",
                    "old_owner_reissued": False, "original_failure_preserved": True})
            encoded = (json.dumps(proposed, sort_keys=True) + "\n").encode()
            _need(len(encoded) <= preservation.MAX_METADATA, "recovered bound row exceeds original record limit")
            used = recheck()
            receipt["catalog_plus_archive_charge_bytes"] = used
            if not apply:
                receipt.update(status="eligible_but_not_applied", row=copy.deepcopy(row))
                return receipt
            receipt.update(catalog_write_attempted=True, applied=None)
            write_record(row_path, proposed)
            receipt["catalog_write_returned"] = True
            # Atomic replacement intentionally unlinks the old original row's
            # name. Keep its FD for close only; never call row_held.check again.
            recheck(replacement=proposed)
            receipt.update(status="interrupted_retained", applied=True,
                canonical_readback_verified=True, row=proposed)
            return receipt
        except BaseException as error:
            _failure(error, receipt, ("commit_effects_unknown_preserved" if receipt["catalog_write_attempted"]
                else "failed_before_catalog_commit"), error_type=type(error).__name__)
            try:
                _note(error, "bound recovery write_attempted=" + str(receipt["catalog_write_attempted"])
                    + "; write_returned=" + str(receipt["catalog_write_returned"])
                    + "; inspect canonical row and independent archive; no rollback/retry")
            except BaseException:
                pass
            raise
        finally:
            try:
                _close(resources, sys.exc_info()[1])
            except BaseException as error:
                _failure(error, receipt, "descriptor_cleanup_failed_preserved", cleanup_error_type=type(error).__name__)
                raise
