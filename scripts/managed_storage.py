"""Owned, bounded regenerable build storage; no historical-tree adoption."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import uuid
from pathlib import Path

GiB = 1024 ** 3
SCHEMA = "rovezero.managed-build.v1"
MAX_ENTRIES = 100_000


class StorageError(RuntimeError):
    pass


def checked(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.name.lower().startswith(".env") or part.name.lower() in {
            "credentials", "service-account.json", "id_rsa", "id_ed25519",
        }:
            raise StorageError("credential path is not a managed storage target")
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise StorageError("managed path contains a symlink or junction")
    # Windows AppContainer aliases can name the same inode without a reparse
    # attribute. Normalize their physical path before containment comparisons.
    return Path(os.path.realpath(path))


def tree_size(path: Path, *, allow_disappearing=False) -> int:
    checked(path)
    if not path.exists():
        return 0
    total = 0
    pending = [path]
    seen = set()
    entries = 0
    while pending:
        current = pending.pop()
        try:
            checked(current)
            info = current.lstat()
            entries += 1
            if entries > MAX_ENTRIES:
                raise StorageError("managed tree entry bound exceeded")
            if stat.S_ISDIR(info.st_mode):
                with os.scandir(current) as children:
                    for item in children:
                        if entries + len(pending) >= MAX_ENTRIES:
                            raise StorageError("managed tree entry bound exceeded")
                        pending.append(Path(item.path))
            elif stat.S_ISREG(info.st_mode):
                identity = (info.st_dev, info.st_ino)
                if identity not in seen:
                    total += info.st_size
                    seen.add(identity)
            else:
                raise StorageError("unsupported managed tree entry")
        except FileNotFoundError:
            if not allow_disappearing:
                raise
    return total


def remove_tree(path: Path, parent: Path) -> None:
    path = checked(path)
    parent = checked(parent)
    if path.parent != parent:
        raise StorageError("deletion must name a direct owned child")
    tree_size(path)  # Preflight the whole tree before any deletion.
    if not path.exists():
        return

    def writable_remove(func, name, _error):
        # Only owned, preflighted files; do not follow a changed link.
        target = checked(Path(name))
        if not target.is_relative_to(path):
            raise StorageError("cleanup escaped owned tree")
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        func(name)
    shutil.rmtree(path, onerror=writable_remove)


def write_record(path: Path, value: dict) -> None:
    checked(path)
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode()
    if len(encoded) > 8192:
        raise StorageError("ownership record exceeds bound")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_record(path: Path) -> dict:
    checked(path)
    if not path.is_file() or path.stat().st_size > 8192:
        raise StorageError("ownership record absent or too large")
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, OSError) as exc:
        raise StorageError("ownership record is invalid; preserve it") from exc
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise StorageError("unrecognized ownership record; preserve it")
    return value


def canonical_root() -> Path:
    if os.environ.get("RUNNER_TEMP"):
        base = Path(os.environ["RUNNER_TEMP"]) / "RoveZero" / "build"
    elif os.name == "nt":
        if not os.environ.get("APPDATA"):
            raise StorageError("APPDATA is unavailable")
        base = Path(os.environ["APPDATA"]) / "RoveZero" / "build"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "rovezero" / "build"
    return checked(base / "managed-v1")


class ManagedBuild:
    """One catalog owner serializes reuse/eviction and owns one fixed scratch slot.

    Interrupted/unknown leases are deliberately not stolen. Recovery is manual,
    after checking the original process tree, paths and source ownership.
    """
    def __init__(self, root: Path, source: Path, *, slot_limit=8 * GiB,
                 total_limit=12 * GiB, slots_limit=4):
        self.root = checked(root)
        self.source = Path(os.path.realpath(source))
        if self.root.is_relative_to(self.source) or self.source.is_relative_to(self.root):
            raise StorageError("build storage must be outside the checkout")
        if min(slot_limit, total_limit, slots_limit) <= 0 or slot_limit > total_limit:
            raise StorageError("invalid storage budget")
        self.slot_limit = slot_limit
        self.total_limit = total_limit
        self.slots_limit = slots_limit
        identity = os.path.normcase(str(self.source))
        self.source_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.slot = self.root / "slots" / self.source_id
        self.target = self.slot / "cargo-target"
        self.scratch = self.slot / "tmp"
        self.lock = self.root / "owner.json"
        self.token = uuid.uuid4().hex
        self.acquired = False
        self.finished = False
        self.record = {}
        self.before = 0

    def _identity(self) -> dict:
        return {"schema": SCHEMA, "token": self.token, "pid": os.getpid(),
                "source_id": self.source_id, "source": str(self.source)}

    def _assert_owner(self) -> None:
        if read_record(self.lock) != self._identity():
            raise StorageError("catalog owner changed; preserve all artifacts")
        if self.slot.exists():
            record = read_record(self.slot / "record.json")
            if record != self.record:
                raise StorageError("slot ownership record changed; preserve it")

    def _records(self) -> list[tuple[Path, dict]]:
        catalog = self.root / "slots"
        rows = []
        for item in catalog.iterdir():
            checked(item)
            if not item.is_dir():
                raise StorageError("unknown catalog entry; preserve it")
            record = read_record(item / "record.json")
            if record.get("source_id") != item.name or set(record) != {
                "schema", "source_id", "source", "token", "pid", "state", "finished_ns",
            }:
                raise StorageError("unrecognized slot identity; preserve it")
            if (not isinstance(record["source"], str) or not Path(record["source"]).is_absolute()
                    or hashlib.sha256(os.path.normcase(record["source"]).encode()).hexdigest()[:24] != item.name
                    or record["state"] not in {"active", "complete"}
                    or not isinstance(record["finished_ns"], int) or record["finished_ns"] < 0
                    or not isinstance(record["pid"], int) or record["pid"] <= 0
                    or not isinstance(record["token"], str) or len(record["token"]) != 32
                    or any(c not in "0123456789abcdef" for c in record["token"])):
                raise StorageError("invalid slot provenance; preserve it")
            allowed = {"record.json", "last-run.json", "cargo-target", "tmp"}
            if any(p.name not in allowed for p in item.iterdir()):
                raise StorageError("unknown slot content; preserve it")
            rows.append((item, record))
        return rows

    def _prune(self) -> None:
        rows = self._records()
        for slot, record in rows:
            if record["state"] == "active" and slot != self.slot:
                raise StorageError("interrupted or active slot requires owner-aware maintenance")
        completed = sorted(
            ((slot, record) for slot, record in rows
             if slot != self.slot and record["state"] == "complete"),
            key=lambda row: row[1]["finished_ns"],
        )
        total = sum(tree_size(slot) for slot, _ in rows)
        count = len(rows)
        while completed and (total > self.total_limit or count > self.slots_limit):
            slot, expected = completed.pop(0)
            if read_record(slot / "record.json") != expected:
                raise StorageError("eviction ownership changed")
            amount = tree_size(slot)
            remove_tree(slot, slot.parent)
            total -= amount
            count -= 1
        if total > self.total_limit or count > self.slots_limit:
            raise StorageError("storage budget cannot be met without deleting active/unknown data")

    def acquire(self) -> "ManagedBuild":
        checked(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        if any(p.name not in {"owner.json", "slots"} for p in self.root.iterdir()):
            raise StorageError("unknown build root content; preserve it")
        try:
            with self.lock.open("xb") as stream:
                stream.write((json.dumps(self._identity(), sort_keys=True) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise StorageError("managed build owner exists; do not steal a live/interrupted lease") from exc
        self.acquired = True
        try:
            (self.root / "slots").mkdir(exist_ok=True)
            if self.slot.exists():
                self._records()  # Validate ownership before resetting this slot.
                previous = read_record(self.slot / "record.json")
                if previous["source"] != str(self.source) or previous["state"] != "complete":
                    raise StorageError("source mismatch or interrupted slot; preserve it")
                if tree_size(self.slot) > self.slot_limit:
                    remove_tree(self.slot / "cargo-target", self.slot)
                remove_tree(self.scratch, self.slot)
            else:
                self._prune()
                self.slot.mkdir()
            self.record = {**self._identity(), "state": "active", "finished_ns": 0}
            write_record(self.slot / "record.json", self.record)
            self.target.mkdir(exist_ok=True)
            self.scratch.mkdir()
            self._prune()
            self.before = tree_size(self.slot)
            return self
        except BaseException:
            # A failed admission may expose an interrupted existing slot; do not
            # delete that slot. Release only this invocation's verified lock.
            if read_record(self.lock) == self._identity():
                self.lock.unlink()
            self.acquired = False
            raise

    def environment(self) -> dict:
        self._assert_owner()
        env = os.environ.copy()
        expected = str(self.target)
        if env.get("CARGO_TARGET_DIR") and checked(Path(env["CARGO_TARGET_DIR"])) != self.target:
            raise StorageError("CARGO_TARGET_DIR conflicts with the managed target")
        env.update(CARGO_TARGET_DIR=expected, CARGO_INCREMENTAL="0",
                   TMPDIR=str(self.scratch), TMP=str(self.scratch),
                   TEMP=str(self.scratch), PYTHONDONTWRITEBYTECODE="1")
        return env

    def finish(self, *, exit_code: int, tree_gone: bool, supervisor_error=None) -> dict:
        self._assert_owner()
        if not tree_gone:
            # Keep lease and inputs: uncertain cleanup is not ordinary success.
            raise StorageError("process cleanup unverified; slot and lease retained")
        observed = tree_size(self.slot)
        remove_tree(self.scratch, self.slot)
        reset = tree_size(self.slot) > self.slot_limit
        if reset:
            remove_tree(self.target, self.slot)
        self.record = {**self.record, "state": "complete", "finished_ns": time.time_ns()}
        write_record(self.slot / "record.json", self.record)
        result = {"schema": SCHEMA, "source_id": self.source_id,
                  "command_exit_code": exit_code, "tree_cleanup_verified": tree_gone,
                  "supervisor_error": supervisor_error,
                  "pre_run_bytes": self.before, "observed_end_bytes": observed,
                  "temporary_removed": True, "post_run_budget_reset": reset,
                  "retained_slot_bytes": tree_size(self.slot),
                  "slot_limit_bytes": self.slot_limit,
                  "catalog_limit_bytes": self.total_limit, "maximum_slots": self.slots_limit}
        write_record(self.slot / "last-run.json", result)
        self._prune()
        self._assert_owner()
        self.lock.unlink()
        self.finished = True
        return result

