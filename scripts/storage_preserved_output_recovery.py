"""Opt-in reboot recovery of one exact deferred row beside preserved Source7.

Audit is the default. This never acquires a build lease, deletes cache, adopts
an output, issues closure, or rewrites preserved history. Original row/owner
bytes must already have independent pinned copies. A returned observation is
not permission for a later call: every apply repeats the live original gates.
Only the catalog row is atomically replaced; post-write failure is partial and
must be preserved for explicit review, with no automatic retry or rollback.
OS guards serialize cooperating writers. The typed selection's exclusive-root
caller condition also covers the final name check/atomic replacement interval.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

sys.dont_write_bytecode = True

import managed_storage as storage
import managed_outputs as outputs
import managed_output_unattributed_preservation as policy
from storage_guard import CatalogGuard, lifetime, reboot_proves_dead

ARTIFACT_CAP = 16 << 10
METADATA_CAP = 8192
SCHEMA = "rovezero.preserved-output-recovery.v1"


class PreservedOutputRecoveryUnsupported(storage.StorageError):
    code = "PreservedOutputRecoveryUnsupported"


def _need(condition, text):
    if condition is not True:
        raise storage.StorageError(text)


def _hex(value, length=64):
    _need(type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None,
          "exact lowercase digest/key required")


def _path(value):
    _need(isinstance(value, (str, Path)), "explicit absolute maintenance path required")
    path = Path(value)
    _need(path.is_absolute() and ".." not in path.parts and len(str(path).encode()) <= 4096,
          "bounded nonescaping absolute maintenance path required")
    for part in path.parts:
        name = part.lower()
        _need(not (name.startswith(".env") or name.endswith((".key", ".pem"))
                  or any(tag in name for tag in ("credential", "service-account", "service_account",
                                                "id_rsa", "id_ed25519", "api_key", "api-key"))),
              "secret paths are never recovery inputs")
    _need(storage.checked(path) == path, "maintenance path aliases/links are forbidden")
    return path


def _note(first, text):
    try:
        first.add_note(text[:1024])
    except BaseException:
        pass


def _overlap(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


def _pair(value):
    _need(type(value) is list and len(value) == 2
          and all(type(part) is int and 0 <= part < 1 << 128 for part in value),
          "exact recorded device/inode pair required")
    return tuple(value)


def _identity(info):
    return info.st_dev, info.st_ino


def _directory(held, path, *, private=False):
    fd = held.directory(path)
    info = os.fstat(fd)
    mask = 0o077 if private else 0o022
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and info.st_mode & mask == 0,
          "original root-owned private/control directory required")
    _need(not private or stat.S_IMODE(info.st_mode) == 0o700,
          "original deferred output must retain its exact private 0700 mode")
    return fd


def _raw(held, path, schema):
    snapshot = storage.read_record_snapshot(path, schema=schema)
    pin = {"path": str(path), "bytes": snapshot[2][4], "sha256": snapshot[1]}
    raw = held.read(pin, METADATA_CAP, root_owned=True)
    _need(policy._decode(raw) == snapshot[0]
          and storage.read_record_snapshot(path, schema=schema) == snapshot,
          "original metadata raw/FD/name changed")
    fd, before, _ = held.blobs[-1]
    return snapshot, raw, fd, before


def _build_identity(owner, root):
    required = {"schema", "token", "pid", "source_id", "source", "lifetime", "root_identity"}
    _need(type(owner) is dict and required <= set(owner) <= required | {"purpose"}
          and owner["schema"] == storage.SCHEMA, "exact original build owner schema required")
    _hex(owner["token"], 32)
    _hex(owner["source_id"], 24)
    _need(type(owner["pid"]) is int and owner["pid"] > 0, "original positive build owner PID required")
    source = _path(owner["source"])
    purpose = owner.get("purpose")
    _need("purpose" not in owner or (type(purpose) is str and re.fullmatch(r"[a-z0-9_-]{1,64}", purpose) is not None),
          "original build purpose malformed")
    identity = os.path.normcase(str(source)) + ("\0" + purpose if purpose else "")
    _need(hashlib.sha256(identity.encode()).hexdigest()[:24] == owner["source_id"],
          "original build source/slot key differs")
    _need(_pair(owner["root_identity"]) == _identity(root.stat()), "original build root inode differs")
    return source, root / "slots" / owner["source_id"]


def _scopes(held, root, target, catalog, selection, rows, copies, view, source, row_source):
    """Current physical coordinates are DENY only, never old custody proof."""
    held.tick()
    protected = [catalog, policy.custody.registry_path(catalog), source, row_source]
    for selected in selection.rows:
        protected.extend(_path(selected[name]["path"]) for name in ("catalog", "marker", "research_archive"))
        protected.append(_path(selected["marker"]["path"]).parent)
    protected.extend(_path(pin["path"]) for pin in (*selection.tips, *selection.sources.values()))
    # The actual history reader already holds every old proof/receipt/source
    # blob. Their locations remain exclusions too; tips alone do not cover
    # source bodies preserved separately from the registry directory.
    protected.extend(_path(path) for path, _, _, _ in view.history.held.blobs)
    protected.extend(copies)
    before, table = policy._mount_table()
    def disjoint(left, right):
        held.tick()
        _need(not _overlap(left, right), "recovery scope overlaps preserved/control path")
        device, coordinate = policy._physical_scope(left, table)
        other_device, other_coordinate = policy._physical_scope(right, table)
        _need(device != other_device or not _overlap(coordinate, other_coordinate),
              "recovery scope aliases preserved/control physical scope")
    disjoint(root, target)
    for path in protected:
        disjoint(root, path)
        disjoint(target, path)
    for _, row, output in rows:
        disjoint(root, output)
        if row["output"] != str(target):
            disjoint(target, output)
    for copied in copies:
        disjoint(copied, catalog)
        disjoint(copied, root)
        disjoint(copied, target)
    _need(policy._mount_table()[0] == before, "recovery current mount scope changed")
    held.tick()


def _artifact(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("ascii")
    _need(len(raw) <= ARTIFACT_CAP, "bounded recovery observation required")
    return value


def _same_rows(left, right):
    # Atomic replacement may change filesystem iteration order. Compare the
    # exact keyed roster, without treating that incidental order as custody.
    return (len(left) == len(right)
            and {str(path): (row, target) for path, row, target in left}
            == {str(path): (row, target) for path, row, target in right})


def _check_source_pins(view, selection):
    """Recheck only the seven tiny held source blobs, not the corpus inventory."""
    for pin in selection.sources.values():
        matches = [(fd, before) for fd, before, actual in view.held.blobs if actual == pin]
        _need(len(matches) == 1, "exact held current Source7 FD required")
        fd, before = matches[0]
        path = Path(pin["path"])
        named = os.stat(path.name, dir_fd=view.held.file_parents[fd], follow_symlinks=False)
        _need(policy.custody._stamp(os.fstat(fd)) == before == policy.custody._stamp(named)
              and hashlib.sha256(view.held.bytes(fd, pin["bytes"])).hexdigest() == pin["sha256"]
              and policy.custody._stamp(os.fstat(fd)) == before,
              "held current Source7 bytes/name changed before recovery commit")


def recover_preserved_output(build_root, catalog, expected_build_owner_sha256, key,
                             expected_row_sha256, selection, *, preserved_owner_pin,
                             preserved_row_pin, apply=False):
    """Inspect by default; explicit apply replaces only the exact deferred row."""
    if sys.platform != "linux":
        raise PreservedOutputRecoveryUnsupported("preserved output recovery requires native Linux")
    _need(os.geteuid() == 0, "preserved output recovery requires Linux UID 0")
    _need(type(apply) is bool, "explicit boolean recovery apply required")
    _need(type(selection) is policy.UnattributedSelection and selection._bound_deadline is not None,
          "exact typed Source7 selection with original bound deadline required")
    _need(type(selection.maximum_seconds) in (int, float) and math.isfinite(selection.maximum_seconds)
          and 0 < selection.maximum_seconds <= 30, "original <=30s observation required")
    for digest in (expected_build_owner_sha256, key, expected_row_sha256):
        _hex(digest)
    root, catalog = _path(build_root), _path(catalog)
    _need(root == _path(storage.canonical_root()) and catalog == _path(outputs.canonical_catalog()),
          "actual canonical build root and output catalog required")
    copies = [_path(pin["path"]) for pin in (preserved_owner_pin, preserved_row_pin)]
    policy._pin(preserved_owner_pin, METADATA_CAP)
    policy._pin(preserved_row_pin, METADATA_CAP)
    for item in selection.rows:
        for pin in item.values():
            _path(pin["path"])
    for pin in (*selection.tips, *selection.sources.values()):
        _path(pin["path"])
    _need(copies[0] != copies[1], "independent original owner and row preservation files required")
    guards, held, row_held, view = [], None, None, None
    first = result = None
    stage, attempted, returned, committed_sha = "not_started", False, False, None
    failure = {"schema": SCHEMA, "status": "FAILED_OR_PARTIAL_PRESERVED", "key": key,
               "original_row_sha256": expected_row_sha256,
               "original_build_owner_sha256": expected_build_owner_sha256,
               "catalog_write_attempted": False, "catalog_write_returned": False,
               "recovered_row_sha256": None, "automatic_retry_authorized": False,
               "owner_reissued": False, "adoption_performed": False, "closure_issued": False,
               "original_creation_process_attested": False, "physical_completion_attested": False}
    try:
        stage = "holding_original_build_guard"
        for directory in (root, catalog):
            guard = CatalogGuard(directory / "catalog.lock")
            guards.append(guard)
            guard.acquire()
        held = policy._Held(selection.maximum_seconds, selection.original_hard_until)
        row_held = policy._Held(selection.maximum_seconds, held.until)
        now = lifetime()
        root_fd = _directory(held, root)
        catalog_fd = _directory(held, catalog)
        stage = "checking_original_owner_and_slot"
        owner_snapshot, owner_raw, owner_fd, _ = _raw(held, root / "owner.json", storage.SCHEMA)
        owner = owner_snapshot[0]
        _need(owner_snapshot[1] == expected_build_owner_sha256, "original build owner digest changed")
        source, slot = _build_identity(owner, root)
        held.directory(source)
        _need(reboot_proves_dead(owner.get("lifetime"), now), "same-host build owner reboot proof required")
        _directory(held, slot)
        slot_snapshot, _, _, _ = _raw(held, slot / "record.json", storage.SCHEMA)
        slot_record = slot_snapshot[0]
        _need({name: value for name, value in slot_record.items() if name not in ("state", "finished_ns")} == owner
              and slot_record.get("state") in ("active", "complete")
              and type(slot_record.get("finished_ns")) is int and slot_record["finished_ns"] >= 0,
              "original slot and exact build owner do not match")
        stage = "checking_exact_original_deferred_output"
        row_path = catalog / (key + ".json")
        original, row_raw, row_fd, row_before = _raw(row_held, row_path, outputs.SCHEMA)
        row = original[0]
        _need(original[1] == expected_row_sha256, "original deferred output row digest changed")
        target = _path(row.get("output"))
        _need(row.get("key") == key and hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest() == key,
              "exact original output catalog key/path required")
        _need(row.get("state") == "active" and row.get("deferred") is True and row.get("identity") is not None,
              "only one original active identity-bound deferred output is recoverable")
        target_identity = _pair(row["identity"])
        row_source = _path(row.get("source"))
        held.directory(row_source)
        outer = row.get("outer_supervisor")
        if outer is None:
            outer_binding = "absent_original"
        else:
            _need(outer == {"root": str(root), "owner": owner,
                        "scope": "inherited_outer_process_group_or_job"} and row_source == source,
                  "exact original outer build owner/source/scope required when binding exists")
            outer_binding = "matched_original_record"
        _need(reboot_proves_dead(row.get("lifetime"), now), "same-host original output reboot proof required")
        if outer is not None:
            _need(all(row["lifetime"][name] == owner["lifetime"][name] for name in ("host", "boot", "platform")),
                  "bound original output and outer supervisor must share their original host/boot/platform")
        _directory(held, target.parent)
        target_fd = _directory(held, target, private=True)
        _need(_identity(os.fstat(target_fd)) == target_identity, "original deferred output directory identity differs")
        stage = "verifying_independent_original_raw_preservation"
        for pin, raw, source_fd in ((preserved_owner_pin, owner_raw, owner_fd),
                                    (preserved_row_pin, row_raw, row_fd)):
            copied = held.read(pin, METADATA_CAP, root_owned=True)
            copied_fd = held.blobs[-1][0]
            _need(copied == raw and _identity(os.fstat(copied_fd)) != _identity(os.fstat(source_fd)),
                  "independent preservation FD must hold exactly the original raw bytes")
        _need(_identity(os.fstat(held.blobs[-1][0])) != _identity(os.fstat(held.blobs[-2][0])),
              "owner and row preservation inode aliases forbidden")
        stage = "observing_current_preserved_source7_history"
        view = policy.observe_catalog(catalog, selection, guards[1], hard_until=held.until)
        _need(key not in view.keys, "selected or historical preserved output can never be recovered")
        # observe_catalog already completed the initial full held check and
        # charge. Read its current metadata roster without repeating that full
        # walk; the same view is checked completely before and after commit.
        rows, usage = policy._rows(catalog, view.details), None
        snapshots = {str(path): storage.read_record_snapshot(path, schema=outputs.SCHEMA)
                     for path, _, _ in rows}
        _need(snapshots.get(str(row_path)) == original, "original deferred row changed during preserved observation")
        def recheck(*, after_write=False):
            held.check()
            for guard in guards:
                guard.assert_held()
            _need(lifetime() == now and reboot_proves_dead(owner.get("lifetime"), now)
                  and reboot_proves_dead(row.get("lifetime"), now), "original reboot/observer lifetime changed")
            _need(storage.read_record_snapshot(root / "owner.json") == owner_snapshot
                  and storage.read_record_snapshot(slot / "record.json") == slot_snapshot,
                  "original build owner/slot raw changed before recovery commit")
            _need(_path(owner["source"]) == source and _path(row["source"]) == row_source,
                  "original build/row source paths changed before recovery commit")
            _need(_identity(os.fstat(root_fd)) == _pair(owner["root_identity"])
                  and _identity(os.fstat(target_fd)) == target_identity,
                  "original build/output held directory identity changed")
            if not after_write:
                row_held.check()
                _need(storage.read_record_snapshot(row_path, schema=outputs.SCHEMA) == original,
                      "original row raw/stamp changed before recovery commit")
            else:
                # Atomic replacement intentionally detaches this original FD.
                # Its original content remains held; do not require old nlink
                # or ctime to equal a new catalog name's identity.
                current = os.fstat(row_fd)
                _need((current.st_dev, current.st_ino, current.st_mode, current.st_uid, current.st_gid,
                       current.st_size, current.st_mtime_ns) ==
                      (row_before[0], row_before[1], row_before[2], row_before[3], row_before[4],
                       row_before[6], row_before[7]) and current.st_nlink == 0
                      and hashlib.sha256(row_held.bytes(row_fd, len(row_raw))).hexdigest() == original[1],
                      "held original row bytes changed after atomic replacement")
                for directory in row_held.directories:
                    row_held.directory(directory)
            _scopes(held, root, target, catalog, selection, rows, copies, view, source, row_source)
            for path, _, _ in rows:
                if path != row_path:
                    _need(storage.read_record_snapshot(path, schema=outputs.SCHEMA) == snapshots[str(path)],
                          "another original catalog row changed")
            _check_source_pins(view, selection)
            held.tick()
            row_held.tick()
            guards[0].assert_held()
            guards[1].assert_held()
        stage = "final_original_gate_checks"
        recheck()
        proposed = dict(row, state="interrupted_retained", recovery_reason="verified_same_host_reboot",
                        recovered_ns=time.time_ns(), recovery_source_sha256=expected_row_sha256)
        _need(len((json.dumps(proposed, sort_keys=True) + "\n").encode()) <= METADATA_CAP,
              "recovered original catalog record exceeds existing bound")
        # A final complete same-view check precedes the sole mutation. It never
        # refreshes the 30s window or turns an archived observation into a gate.
        final_rows, usage = view.checked_usage()
        _need(_same_rows(final_rows, rows), "original catalog roster changed before recovery commit")
        recheck()
        if apply:
            stage, attempted = "catalog_write_pending", True
            storage.write_record(row_path, proposed)
            returned = True
            stage = "catalog_written_readback_pending"
            committed = storage.read_record_snapshot(row_path, schema=outputs.SCHEMA)
            _need(committed[0] == proposed, "recovered catalog row readback differs; preserve partial write")
            committed_sha = committed[1]
            recheck(after_write=True)
            # Existing held Source7/history/archive checks remain live after
            # publication too; failure cannot undo or acknowledge the row write.
            post_rows, usage = view.checked_usage()
            expected_rows = [(path, proposed if path == row_path else item, output)
                             for path, item, output in rows]
            _need(_same_rows(post_rows, expected_rows), "catalog changed after deferred row publication")
            _need(storage.read_record_snapshot(row_path, schema=outputs.SCHEMA) == committed,
                  "recovered row raw/stamp changed after publication")
            recheck(after_write=True)
        stage = "complete"
        result = _artifact({"schema": SCHEMA, "status": "INTERRUPTED_RETAINED" if apply else "ELIGIBLE_READ_ONLY",
            "key": key, "original_row_sha256": expected_row_sha256,
            "original_build_owner_sha256": expected_build_owner_sha256, "recovered_row_sha256": committed_sha,
            "catalog_row_changed": apply, "deletion_performed": False, "preserved_catalog_rows_changed": False,
            "build_owner_slot_or_cache_changed": False, "marker_or_custody_changed": False,
            "original_raw_copies_verified": True, "original_creation_process_attested": False,
            "original_row_outer_binding": outer_binding, "row_build_relationship_attested": False,
            "owner_reissued": False, "adoption_performed": False, "closure_issued": False,
            "original_failure_and_physical_fields_preserved": True,
            "physical_completion_attested": False, "automatic_retry_authorized": False,
            "conservative_usage_bytes": usage, "observation_refresh": False})
    except BaseException as error:
        first = error
    finally:
        for resource in (view, row_held, held):
            if resource is not None:
                try:
                    resource.close(first)
                except BaseException as error:
                    if first is None:
                        first = error
                    else:
                        _note(first, "secondary recovery observation close failed: " + type(error).__name__[:128])
        for guard in reversed(guards):
            try:
                guard.release()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "secondary recovery guard release failed: " + type(error).__name__[:128])
    if first is not None:
        failure.update(stage=stage, catalog_write_attempted=attempted,
                       catalog_write_returned=returned, recovered_row_sha256=committed_sha)
        try:
            first.preserved_output_recovery = _artifact(failure)
        except BaseException:
            pass
        _note(first, "preserved output recovery stopped; catalog_write_attempted=" + str(attempted)
              + "; no automatic retry, rollback, closure or owner reissue")
        raise first
    return result


def inspect_preserved_output_recovery(build_root, catalog, expected_build_owner_sha256, key,
                                      expected_row_sha256, selection, *, preserved_owner_pin,
                                      preserved_row_pin):
    return recover_preserved_output(build_root, catalog, expected_build_owner_sha256, key,
        expected_row_sha256, selection, preserved_owner_pin=preserved_owner_pin,
        preserved_row_pin=preserved_row_pin)
