"""Bounded, registered run outputs; unique evidence is never an eviction cache.

The catalog serializes producers and reserves their maximum output before they
start. Retained results count toward admission even after expiry. Expiry means
review/archival is due, not permission to delete evidence or mapped GPU inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import uuid

from managed_storage import (GiB, StorageError, canonical_root, checked, tree_size,
                             write_record, read_record_snapshot)
from storage_guard import CatalogGuard, lifetime, reboot_proves_dead

SCHEMA = "rovezero.managed-output.v1"
MAX_RUNS = 1024


class OutputBudgetError(StorageError):
    pass


def canonical_catalog():
    return canonical_root().parent.parent / "storage" / "outputs-v1"


def _read(path):
    return read_record_snapshot(path, schema=SCHEMA)[0]


def identity(path):
    info = checked(path).stat()
    return [info.st_dev, info.st_ino]


def _catalog_rows(catalog):
    rows = []
    entries = list(catalog.iterdir())
    if len(entries) > MAX_RUNS + 1:
        raise StorageError("output catalog entry bound exceeded; archive review required")
    for path in entries:
        if path.name == "catalog.lock":
            continue
        row = _read(path)
        target = checked(Path(row["output"]))
        key = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()
        pending = row.get("deferred") is True and row.get("identity") is None
        if (path.name != key + ".json" or row.get("key") != key
                or row.get("state") not in {"active", "retained", "interrupted_retained"}
                or (pending and row["state"] != "active" and target.exists())
                or (not pending and row.get("identity") != identity(target))
                or (not row.get("deferred") and _read(target / ".rovezero-output.json") != row)):
            raise StorageError("output custody changed or missing; preserve catalog")
        rows.append((path, row, target))
    return rows


class _CreatedOutputDirectory:
    """In-process custody from exclusive mkdir; never reconstructed from a pathname."""
    def __init__(self, parent_fd, parent_identity, name, owner):
        self.parent_fd, self.fd = parent_fd, None
        self.parent_identity, self.name = parent_identity, name
        self.identity, self.directory_identity = None, None
        self.owner, self.sealed = owner, False

    @staticmethod
    def directory_identity_of(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)

    def check(self, output):
        parent = self.directory_identity_of(os.fstat(self.parent_fd))
        named_parent = self.directory_identity_of(checked(output.parent).lstat())
        if parent != self.parent_identity or named_parent != parent:
            raise StorageError("created output parent changed; preserve evidence")
        if self.fd is not None:
            opened = os.fstat(self.fd)
            named = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
            if (self.directory_identity_of(opened) != self.directory_identity
                    or self.directory_identity_of(named) != self.directory_identity
                    or not stat.S_ISDIR(opened.st_mode) or opened.st_nlink == 0):
                raise StorageError("created output directory changed; preserve evidence")

    def close(self):
        first = None
        for name in ("fd", "parent_fd"):
            fd = getattr(self, name)
            setattr(self, name, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as error:
                    if first is None:
                        first = error
        if first is not None:
            raise first


class OutputLease:
    def __init__(self, output, source, *, catalog=None, reserve=8 * GiB,
                 total_limit=18 * GiB, retention_days=14, deferred=False, outer_supervised=False,
                 quarantine_receipts=None):
        if quarantine_receipts is not None:
            if not isinstance(quarantine_receipts, (tuple, list)) or len(quarantine_receipts) > MAX_RUNS:
                raise StorageError("bounded explicit custody receipt pins required")
            if outer_supervised:
                raise StorageError("outer finalizer does not carry explicit custody receipt pins")
        self.output, self.source = checked(output), checked(source)
        self.catalog = checked(catalog or canonical_catalog())
        if quarantine_receipts is not None:
            from managed_output_quarantine import registry_path
            registry = registry_path(self.catalog)
            if self.output.is_relative_to(registry) or registry.is_relative_to(self.output):
                raise StorageError("output overlaps custody registry; no producer admission")
        for other in (self.source, self.catalog):
            if self.output.is_relative_to(other) or other.is_relative_to(self.output):
                raise StorageError("output, checkout and catalog must be disjoint")
        if (not 0 < reserve <= total_limit <= 18 * GiB
                or not 1 <= retention_days <= 365):
            raise StorageError("finite output reservation and retention required")
        self.reserve, self.total_limit, self.days = reserve, total_limit, retention_days
        self.deferred = deferred
        self.quarantine_receipts = None if quarantine_receipts is None else tuple(dict(pin) for pin in quarantine_receipts)
        self._quarantine_view = None
        self.key = hashlib.sha256(os.path.normcase(str(self.output)).encode()).hexdigest()
        self.guard = CatalogGuard(self.catalog / "catalog.lock")
        self.record = None
        self._created_directory = None
        self._directory_binding_error = None
        self._directory_binding_stage = "not_started"
        self.outer = None
        if outer_supervised:
            from managed_storage import read_record
            parent_root = checked(Path(os.environ["RZ_MANAGED_BUILD_ROOT"]))
            parent = read_record(parent_root / "owner.json")
            if parent.get("token") != os.environ.get("RZ_MANAGED_BUILD_TOKEN") or parent["source"] != str(self.source):
                raise StorageError("original outer managed supervisor binding missing")
            self.outer = {"root": str(parent_root), "owner": parent,
                          "scope": "inherited_outer_process_group_or_job"}

    def _rows(self):
        rows = _catalog_rows(self.catalog)
        if self.quarantine_receipts is not None:
            from managed_output_quarantine import validate_quarantines
            previous, self._quarantine_view = self._quarantine_view, None
            try:
                self._quarantine_view = validate_quarantines(self.catalog, rows, self.quarantine_receipts, self.guard)
            finally:
                if previous is not None:
                    previous.close(sys.exc_info()[1])
        return rows

    def _check_quarantine_view(self):
        if self.quarantine_receipts is not None:
            if self._quarantine_view is None:
                raise StorageError("explicit custody validation has not completed")
            self._quarantine_view.check()

    def _close_quarantine_view(self, primary=None):
        view, self._quarantine_view = getattr(self, "_quarantine_view", None), None
        if view is not None:
            view.close(primary)

    def _release_guard(self, primary=None):
        try:
            self.guard.release()
        except BaseException:
            if self.quarantine_receipts is None or primary is None:
                raise
            if hasattr(primary, "add_note"):
                primary.add_note("custody output catalog gate release also failed")

    def _output_usage(self, rows, *, allow_disappearing=False):
        if self._quarantine_view is not None:
            return self._quarantine_view.usage(rows)
        return sum(tree_size(target, allow_disappearing=allow_disappearing) for _, _, target in rows)

    def acquire(self):
        self.catalog.mkdir(parents=True, exist_ok=True)
        self.guard.acquire()
        try:
            rows = self._rows()
            custody_keys = self._quarantine_view.keys if self._quarantine_view is not None else frozenset()
            if any(row["state"] == "active" and row["key"] not in custody_keys for _, row, _ in rows):
                raise StorageError("interrupted output requires explicit recovery")
            if len(rows) >= MAX_RUNS:
                raise StorageError("output catalog full; archive review required")
            if any(self.output.is_relative_to(target) or target.is_relative_to(self.output)
                   for _, _, target in rows):
                raise StorageError("overlapping output registration")
            used = self._output_usage(rows)
            if any(row["key"] not in custody_keys and tree_size(target) > row["reserved_bytes"] for _, row, target in rows):
                raise StorageError("previous output exceeds its reservation; archive review required")
            if used + self.reserve > self.total_limit:
                raise StorageError("retained results plus reservation exceed output catalog budget")
            if self.output.exists():
                raise StorageError("only fresh output can be registered; no historical adoption")
            if not self.deferred:
                self.output.mkdir(parents=True)
            now = time.time_ns()
            self.record = {"schema": SCHEMA, "key": self.key, "output": str(self.output),
                "identity": None if self.deferred else identity(self.output), "token": uuid.uuid4().hex,
                "deferred": self.deferred,
                "outer_supervisor": self.outer,
                "source": str(self.source), "pid": os.getpid(), "lifetime": lifetime(),
                "state": "active", "created_ns": now,
                "review_after_ns": now + self.days * 86400 * 10**9,
                "reserved_bytes": self.reserve, "catalog_limit_bytes": self.total_limit,
                "classification": "retained_evidence", "automatic_expiry_deletion": False}
            self._write()
            return self
        except BaseException as error:
            try:
                self._close_quarantine_view(error)
            finally:
                self._release_guard(error)
            raise

    def create_deferred_directory(self):
        """Create and seal a fresh Linux directory before allowing producer work.

        The acquired original producer must exclusively control the parent name.
        No existing path/FD is adopted. A crash between mkdir and catalog fsync
        remains unrecoverable by the unchanged identity-null recovery guard.
        Publication failure is sticky: preserve the path and primary error, and
        do not start payload/children or call success finish. No rollback occurs.
        """
        self.guard.assert_held()
        if self.record is None or not self.deferred:
            raise StorageError("an acquired deferred output lease is required")
        if self._created_directory is not None or self._directory_binding_error is not None:
            raise StorageError("created directory binding already attempted; preserve evidence")
        try:
            self._directory_binding_stage = "validating_original_reservation"
            if (sys.platform != "linux" or self.record.get("state") != "active"
                    or self.record.get("identity") is not None
                    or self.record.get("deferred") is not True
                    or self.record.get("key") != self.key
                    or self.record.get("output") != str(self.output)
                    or self.record.get("source") != str(self.source)
                    or self.record.get("pid") != os.getpid()
                    or self.record.get("lifetime") != lifetime()):
                raise StorageError("original active Linux deferred producer required")
            path = self.catalog / (self.key + ".json")
            original = read_record_snapshot(path, schema=SCHEMA)
            if original[0] != self.record:
                raise StorageError("output owner changed; preserve all records")
            self.check_budget()
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            owner = (self.record["token"], self.record["pid"], dict(self.record["lifetime"]))
            cap = _CreatedOutputDirectory(None, None, self.output.name, owner)
            self._created_directory = cap
            # Transfer the descriptor before fstat/identity setup can fail.
            cap.parent_fd = os.open(checked(self.output.parent), flags)
            parent_fd = cap.parent_fd
            parent = os.fstat(parent_fd)
            cap.parent_identity = cap.directory_identity_of(parent)
            if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.geteuid()
                    or stat.S_IMODE(parent.st_mode) & 0o022):
                raise StorageError("producer-owned parent without group/other writers required")
            cap.check(self.output)
            try:
                os.stat(cap.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise StorageError("only exclusive fresh directory creation is allowed")
            self._directory_binding_stage = "mkdir_pending"
            os.mkdir(cap.name, 0o700, dir_fd=parent_fd)
            self._directory_binding_stage = "directory_created_unsealed"
            cap.fd = os.open(cap.name, flags, dir_fd=parent_fd)
            directory = os.fstat(cap.fd)
            if (not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.geteuid()
                    or stat.S_IMODE(directory.st_mode) & 0o077):
                raise StorageError("private original directory descriptor required")
            cap.directory_identity = cap.directory_identity_of(directory)
            cap.identity = [directory.st_dev, directory.st_ino]
            cap.check(self.output)
            os.fsync(cap.fd)
            os.fsync(cap.parent_fd)
            self.guard.assert_held()
            if (read_record_snapshot(path, schema=SCHEMA) != original
                    or owner != (self.record["token"], os.getpid(), lifetime())):
                raise StorageError("original producer changed before directory seal")
            cap.check(self.output)
            proposed = dict(self.record, identity=cap.identity)
            # The deferred row stays outside the payload tree; its actual bytes
            # still obey the existing record bound before any metadata write.
            if len((json.dumps(proposed, sort_keys=True) + "\n").encode()) > 8192:
                raise StorageError("ownership record exceeds bound")
            self._check_quarantine_view()
            self._directory_binding_stage = "catalog_write_pending"
            write_record(path, proposed)
            self._directory_binding_stage = "catalog_written_readback_pending"
            actual, digest, _ = read_record_snapshot(path, schema=SCHEMA)
            if actual != proposed:
                raise StorageError("created output catalog readback changed")
            self.guard.assert_held()
            cap.check(self.output)
            self.record, cap.sealed = proposed, True
            self._directory_binding_sha256 = digest
            self.check_budget()
            self._directory_binding_stage = "sealed"
            return self.output
        except BaseException as error:
            self._directory_binding_error = error
            raise

    def _check_created_directory(self):
        if self._directory_binding_error is not None:
            raise StorageError("directory binding failed; no finish or further work is allowed") from self._directory_binding_error
        cap = self._created_directory
        if cap is not None:
            if (not cap.sealed or self.record["identity"] != cap.identity
                    or cap.owner != (self.record["token"], os.getpid(), lifetime())):
                raise StorageError("original created directory lease changed")
            cap.check(self.output)

    def _close_created_directory(self, primary=None):
        cap = getattr(self, "_created_directory", None)
        if cap is not None:
            try:
                cap.close()
            except OSError as error:
                self._directory_close_error = error
                if primary is None:
                    raise
                if hasattr(primary, "add_note"):
                    primary.add_note("created directory descriptor close also failed")

    def __del__(self):
        try:
            self._close_created_directory()
        except BaseException:
            pass  # Descriptor release only; no pathname mutation or row finish.
        try:
            self._close_quarantine_view()
        except BaseException:
            pass

    def _write(self):
        self.guard.assert_held()
        self._check_created_directory()
        self._check_quarantine_view()
        if not self.deferred:
            write_record(self.output / ".rovezero-output.json", self.record)
        write_record(self.catalog / (self.key + ".json"), self.record)
        if self.quarantine_receipts is not None:
            self._check_quarantine_view()  # A late error keeps written metadata; no payload permission.

    def check_budget(self):
        self.guard.assert_held()
        self._check_created_directory()
        if self.record is None:
            raise StorageError("output lease not acquired")
        if (_read(self.catalog / (self.key + ".json")) != self.record
                or (not self.deferred and _read(self.output / ".rovezero-output.json") != self.record)):
            raise StorageError("output owner changed; preserve all records")
        rows = self._rows()
        actual = self._output_usage(rows, allow_disappearing=True)
        if tree_size(self.output, allow_disappearing=True) > self.reserve or actual > self.total_limit:
            raise OutputBudgetError("managed output byte budget exceeded; evidence retained")
        self._check_created_directory()
        self._check_quarantine_view()

    def finish(self, *, status, tree_gone):
        # No file cleanup is implied by this method. In particular OS exit is
        # never promoted to a provider's physical completion fence.
        error = None
        try:
            self.check_budget()
        except OutputBudgetError as exc:
            error = exc
        self.record = {**self.record, "state": "retained" if tree_gone is True else "active",
                       "status": "failed_output_budget" if error else str(status), "producer_status": str(status),
                       "tree_gone": tree_gone, "finished_ns": time.time_ns(),
                       "budget_error": str(error) if error else None,
                       "physical_completion": "not_inferred_from_output_lifecycle"}
        if self.deferred and self._created_directory is None and self.output.exists():
            self.record["identity"] = identity(self.output)
        try:
            self._write()
            try:
                self.check_budget()  # Terminal metadata itself consumes bytes.
            except OutputBudgetError as late_error:
                error = late_error
                self.record.update(status="failed_output_budget", budget_error=str(late_error))
                self._write()
        finally:
            try:
                self._close_created_directory(sys.exc_info()[1] or error)
            finally:
                try:
                    self._close_quarantine_view(sys.exc_info()[1] or error)
                finally:
                    self._release_guard(sys.exc_info()[1] or error)
        if error:
            raise error
        return self.record


def _refuse_custody_registry(catalog):
    # Deny only: legacy recovery never interprets a receipt or adopts custody.
    registry = catalog.with_name(catalog.name + ".interrupted-unbound-custody-v1")
    try:
        registry.lstat()  # A broken symlink/special/empty/partial registry also blocks.
    except FileNotFoundError:
        return
    raise StorageError("custody registry present; legacy output recovery is forbidden")


def recover_output(catalog, key, expected_sha256):
    catalog = checked(catalog)
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise StorageError("exact output catalog key required")
    with CatalogGuard(catalog / "catalog.lock") as guard:
        guard.assert_held()
        _refuse_custody_registry(catalog)
        path = catalog / (key + ".json")
        row, digest, stamp = read_record_snapshot(path, schema=SCHEMA)
        if digest != expected_sha256:
            raise StorageError("output lease digest changed")
        target = checked(Path(row["output"]))
        if (row.get("state") != "active"
                or (row.get("identity") is None and target.exists())
                or (row.get("identity") is not None and row["identity"] != identity(target))
                or (not row.get("deferred") and _read(target / ".rovezero-output.json") != row)
                or not reboot_proves_dead(row.get("lifetime"), lifetime())):
            raise StorageError("interrupted output termination not proven")
        guard.assert_held()
        _refuse_custody_registry(catalog)
        if (read_record_snapshot(path, schema=SCHEMA) != (row, digest, stamp)
                or (row.get("identity") is not None and row["identity"] != identity(target))):
            raise StorageError("output identity changed before recovery commit")
        guard.assert_held()
        _refuse_custody_registry(catalog)
        row.update(state="interrupted_retained", recovery_reason="verified_same_host_reboot",
                   recovered_ns=time.time_ns(), recovery_source_sha256=expected_sha256)
        if not row.get("deferred"):
            write_record(target / ".rovezero-output.json", row)
        write_record(path, row)
        return row


def finalize_supervised_outputs(root, owner, catalog=None):
    """Called only by the original supervisor after its owned tree is gone."""
    catalog = checked(catalog or canonical_catalog())
    if not catalog.exists():
        return
    with CatalogGuard(catalog / "catalog.lock") as guard:
        sample_output_budget(catalog)
        entries = list(catalog.iterdir())
        if len(entries) > MAX_RUNS + 1:
            raise StorageError("output catalog bound exceeded")
        for path in entries:
            if path.name == "catalog.lock":
                continue
            row = _read(path)
            expected = {"root": str(root), "owner": owner,
                        "scope": "inherited_outer_process_group_or_job"}
            if row.get("outer_supervisor") != expected or row.get("state") != "active":
                continue
            target = checked(Path(row["output"]))
            if row.get("identity") != identity(target) or _read(target / ".rovezero-output.json") != row:
                raise StorageError("supervised output custody changed; preserve it")
            row.update(state="retained", tree_gone=True, closure="original_outer_supervisor",
                       physical_completion="not_inferred_from_output_lifecycle")
            guard.assert_held()
            write_record(target / ".rovezero-output.json", row)
            write_record(path, row)
        sample_output_budget(catalog)


def sample_output_budget(catalog=None, *, quarantine_receipts=None, quarantine_guard=None):
    """Sampled length cap, never a quota or physical completion witness.

    V1 calls remain read-only without acquiring the producer's held gate. Aware
    calls use an explicit original guard (live owner: prefer check_budget); only
    independent sampling may omit it and acquire that gate. Busy/errors propagate.
    """
    catalog = checked(catalog or canonical_catalog())
    if quarantine_receipts is not None:
        from managed_output_quarantine import original_custody_gate, validate_quarantines
        if quarantine_guard is None:
            with original_custody_gate(catalog) as held:
                return sample_output_budget(catalog, quarantine_receipts=quarantine_receipts, quarantine_guard=held)
        view = validate_quarantines(catalog, _catalog_rows(catalog), quarantine_receipts, quarantine_guard)
        try:
            total = view.usage(_catalog_rows(catalog))
            view.check()
            return total
        finally:
            view.close(sys.exc_info()[1])
    if quarantine_guard is not None:
        raise StorageError("custody guard without explicit receipt pins is not a typed selection")
    if not catalog.exists():
        return 0
    total = 0
    entries = list(catalog.iterdir())
    if len(entries) > MAX_RUNS + 1:
        raise StorageError("output catalog entry bound exceeded")
    for path in entries:
        if path.name == "catalog.lock":
            continue
        row = _read(path)
        target = checked(Path(row["output"]))
        if row.get("identity") is not None and row["identity"] != identity(target):
            raise StorageError("live output identity changed")
        amount = tree_size(target, allow_disappearing=True)
        if amount > row["reserved_bytes"]:
            raise StorageError("managed output exceeds reservation")
        total += amount
    if total > 18 * GiB:
        raise StorageError("retained output catalog exceeds 18GiB")
    return total


def publish_runtime(source, store, *, limit=4 * GiB):
    """One immutable CPU DSO per digest, with bounded aggregate admission.

    This is the source-library store, not ORT's extracted loader cache. No
    model, result or original runtime is evicted to make room.
    """
    source, store = checked(source), checked(store)
    if store == source or store.is_relative_to(source) or source.is_relative_to(store):
        raise StorageError("source runtime and shared store must be disjoint")
    store.mkdir(parents=True, exist_ok=True)
    with CatalogGuard(store / "catalog.lock") as guard:
        before = source.stat()
        if not source.is_file() or not 0 < before.st_size <= GiB:
            raise StorageError("bounded runtime file required")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest = digest.hexdigest()
        folder = checked(store / digest)
        target = checked(folder / "libonnxruntime.so.1.22.0")
        if target.exists():
            target_info = target.stat()
            if (not stat.S_ISREG(target_info.st_mode) or target_info.st_nlink != 1
                    or target_info.st_size != before.st_size):
                raise StorageError("immutable runtime store type or size mismatch")
            actual = hashlib.sha256()
            with target.open("rb") as stream:
                remaining = before.st_size
                while chunk := stream.read(min(1024 * 1024, remaining + 1)):
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise StorageError("runtime store grew while reading")
                    actual.update(chunk)
            if actual.hexdigest() != digest or target.stat().st_size != before.st_size:
                raise StorageError("immutable runtime store digest mismatch")
        else:
            if tree_size(store) + before.st_size > limit:
                raise StorageError("shared source runtime store budget exhausted; preserve assets")
            guard.assert_held()
            folder.mkdir(exist_ok=True)
            # An interrupted exclusive copy stays visible as invalid; never
            # repair/overwrite it under an already registered identity.
            with source.open("rb") as origin, target.open("xb") as destination:
                copied = hashlib.sha256()
                while chunk := origin.read(1024 * 1024):
                    destination.write(chunk)
                    copied.update(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if copied.hexdigest() != digest:
                raise StorageError("runtime source changed during copy")
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise StorageError("runtime source identity changed")
        if tree_size(store) > limit:
            raise StorageError("shared source runtime store budget exceeded")
        guard.assert_held()
        return target, digest
