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


class OutputLease:
    def __init__(self, output, source, *, catalog=None, reserve=8 * GiB,
                 total_limit=18 * GiB, retention_days=14, deferred=False, outer_supervised=False):
        self.output, self.source = checked(output), checked(source)
        self.catalog = checked(catalog or canonical_catalog())
        for other in (self.source, self.catalog):
            if self.output.is_relative_to(other) or other.is_relative_to(self.output):
                raise StorageError("output, checkout and catalog must be disjoint")
        if (not 0 < reserve <= total_limit <= 18 * GiB
                or not 1 <= retention_days <= 365):
            raise StorageError("finite output reservation and retention required")
        self.reserve, self.total_limit, self.days = reserve, total_limit, retention_days
        self.deferred = deferred
        self.key = hashlib.sha256(os.path.normcase(str(self.output)).encode()).hexdigest()
        self.guard = CatalogGuard(self.catalog / "catalog.lock")
        self.record = None
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
        rows = []
        entries = list(self.catalog.iterdir())
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

    def acquire(self):
        self.catalog.mkdir(parents=True, exist_ok=True)
        self.guard.acquire()
        try:
            rows = self._rows()
            if any(row["state"] == "active" for _, row, _ in rows):
                raise StorageError("interrupted output requires explicit recovery")
            if len(rows) >= MAX_RUNS:
                raise StorageError("output catalog full; archive review required")
            if any(self.output.is_relative_to(target) or target.is_relative_to(self.output)
                   for _, _, target in rows):
                raise StorageError("overlapping output registration")
            used = sum(tree_size(target) for _, _, target in rows)
            if any(tree_size(target) > row["reserved_bytes"] for _, row, target in rows):
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
        except BaseException:
            self.guard.release()
            raise

    def _write(self):
        self.guard.assert_held()
        if not self.deferred:
            write_record(self.output / ".rovezero-output.json", self.record)
        write_record(self.catalog / (self.key + ".json"), self.record)

    def check_budget(self):
        self.guard.assert_held()
        if self.record is None:
            raise StorageError("output lease not acquired")
        if (_read(self.catalog / (self.key + ".json")) != self.record
                or (not self.deferred and _read(self.output / ".rovezero-output.json") != self.record)):
            raise StorageError("output owner changed; preserve all records")
        rows = self._rows()
        actual = sum(tree_size(path, allow_disappearing=True) for _, _, path in rows)
        if tree_size(self.output, allow_disappearing=True) > self.reserve or actual > self.total_limit:
            raise OutputBudgetError("managed output byte budget exceeded; evidence retained")

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
        if self.deferred and self.output.exists():
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
            self.guard.release()
        if error:
            raise error
        return self.record


def recover_output(catalog, key, expected_sha256):
    catalog = checked(catalog)
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise StorageError("exact output catalog key required")
    with CatalogGuard(catalog / "catalog.lock") as guard:
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
        if (read_record_snapshot(path, schema=SCHEMA) != (row, digest, stamp)
                or (row.get("identity") is not None and row["identity"] != identity(target))):
            raise StorageError("output identity changed before recovery commit")
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


def sample_output_budget(catalog=None):
    """Read-only live supervision; does not acquire the producer's held gate.

    Atomic catalog records identify fresh outputs. Final custody validation and
    mutation still require the gate. This is a sampled length cap, not a quota.
    """
    catalog = checked(catalog or canonical_catalog())
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
