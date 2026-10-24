"""One-shot retirement of exact duplicate ELF leaves in an already closed run.

The authority is the live held catalog gate, unchanged terminal row/marker,
original-owner death, external held keepers and a current Linux /proc census.
An old audit JSON is never authority. No lease, owner, expiry, reservation or
custody record is created, adopted or changed. Only the journal and selected
leaf names are written. A failure retains its journal and first exception.

Linux offers no atomic 'not mapped/open anywhere and unlink this exact inode'
operation. The census is bounded to the explicitly checked current proc/PID
namespace. Another process may open/map a leaf after its last observation; a
same-UID writer may race the final name check. Held parent directories and
O_NOFOLLOW prevent following a substituted link, but do not erase that final
check/unlink interval. These limits are reported, never physical-completion or
disk/VHD space-return proof.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

import storage_artifact_review as review
from managed_storage import GiB, StorageError, checked, read_record_snapshot
from managed_outputs import _catalog_rows, _refuse_custody_registry
from storage_guard import CatalogGuard, lifetime, owner_proven_dead

_JOURNAL_CAP = 256 << 10
_MAX_PROC_PIDS = 65536
_MAX_PROC_FDS = 131072
_MAX_PROC_BYTES = 64 << 20
_MAX_TREE_ENTRIES = 100000
_LIMITS = ["current_visible_proc_pid_namespace_only", "census_is_not_atomic_with_unlink",
           "same_uid_name_race_after_final_check", "no_global_physical_completion_proof"]


class CodeCacheRetirementUnsupported(StorageError):
    code = "CodeCacheRetirementUnsupported"


def _need(condition, message):
    if condition is not True:
        raise StorageError(message)


def _path(value):
    review.no_secret(value)
    path = Path(value)
    _need(path.is_absolute() and str(path) == str(value) and len(os.fsencode(path)) <= 4096
          and ".." not in path.parts, "bounded canonical absolute retirement path required")
    _need(checked(path) == path, "retirement path alias or link is forbidden")
    return path


def _inode(info):
    return info.st_dev, info.st_ino


def _directory_identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _file_identity(info):
    return (_inode(info), info.st_mode, info.st_uid, info.st_gid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _note(error, text):
    try:
        error.add_note(text[:2048])
    except BaseException:
        pass


def _describe(error):
    try:
        return str(error)[:1024]
    except BaseException:
        return "<" + type(error).__name__[:128] + ": detail unavailable>"


class _Inputs:
    """Original component-walk directory FDs and independently held files."""
    def __init__(self, seconds):
        _need(type(seconds) is int and 1 <= seconds <= 600,
              "finite <=600s retirement budget required")
        self.until = time.monotonic() + seconds
        self.directories = {}
        self.files = {}
        self.candidates = set()
        self.deleted = set()

    def tick(self):
        _need(time.monotonic() < self.until, "retirement time budget exceeded; originals preserved")

    def directory(self, path, *, controlled=False):
        self.tick()
        path = _path(path)
        if path not in self.directories:
            parent = None if path == Path("/") else self.directory(path.parent)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            fd = os.open(str(path) if parent is None else path.name, flags, dir_fd=parent)
            # Register before any fallible check so cleanup retains this FD.
            self.directories[path] = [fd, None]
            self.directories[path][1] = os.fstat(fd)
        fd, before = self.directories[path]
        self.check_directory(path)
        if controlled:
            _need(before.st_uid == os.geteuid() and before.st_mode & 0o022 == 0,
                  "retirement directory must be owner-controlled and not group/world writable")
        return fd

    def check_directory(self, path):
        self.tick()
        fd, before = self.directories[path]
        opened, named = os.fstat(fd), path.lstat()
        _need(stat.S_ISDIR(opened.st_mode) and opened.st_nlink > 0
              and _directory_identity(before) == _directory_identity(opened) == _directory_identity(named),
              "held retirement directory name/inode changed")

    def open(self, pin, maximum):
        self.tick()
        review.raw_pin(pin, maximum)
        path = _path(pin["path"])
        if path in self.files:
            item = self.files[path]
            _need(item["pin"] == pin, "one held file cannot have replacement byte pins")
        else:
            parent = self.directory(path.parent)
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
            try:
                stream = os.fdopen(fd, "rb")
            except BaseException as error:
                try:
                    os.close(fd)
                except BaseException as cleanup:
                    _note(error, "untransferred retirement FD close also failed: " + _describe(cleanup))
                raise
            item = {"path": path, "stream": stream, "before": None, "pin": dict(pin)}
            self.files[path] = item
            before = os.fstat(stream.fileno())
            item["before"] = before
            _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                  and before.st_uid == os.geteuid() and before.st_mode & 0o022 == 0
                  and 0 < before.st_size <= maximum,
                  "retirement FD/type/identity or writable file is forbidden")
            self.hash_file(path)
        self.check_file(path)
        item["stream"].seek(0)
        return item["stream"]

    def hash_file(self, path):
        self.check_file(path)
        item = self.files[path]
        stream, pin = item["stream"], item["pin"]
        digest, offset = hashlib.sha256(), 0
        while offset < pin["bytes"]:
            self.tick()
            raw = os.pread(stream.fileno(), min(65536, pin["bytes"] - offset), offset)
            _need(bool(raw), "retirement held file truncated during hash")
            digest.update(raw)
            offset += len(raw)
        _need(not os.pread(stream.fileno(), 1, offset)
              and (offset, digest.hexdigest()) == (pin["bytes"], pin["sha256"]),
              "retirement original file bytes/hash changed")
        self.check_file(path)

    def check_file(self, path):
        self.tick()
        item = self.files[path]
        before = item["before"]
        _need(before is not None, "retirement file admission did not complete")
        opened = os.fstat(item["stream"].fileno())
        if path in self.deleted:
            _need(_inode(opened) == _inode(before) and opened.st_nlink == 0
                  and opened.st_size == before.st_size and opened.st_mtime_ns == before.st_mtime_ns,
                  "unlinked original ELF still has a name or its held bytes changed")
            try:
                os.stat(path.name, dir_fd=self.directories[path.parent][0], follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise StorageError("unlinked candidate name reappeared; preserve its replacement")
        else:
            named = os.stat(path.name, dir_fd=self.directories[path.parent][0], follow_symlinks=False)
            _need(_file_identity(before) == _file_identity(opened) == _file_identity(named),
                  "held retirement file name/inode/mtime changed")

    def check(self):
        for path in self.directories:
            self.check_directory(path)
        for path in self.files:
            self.check_file(path)

    def fds(self):
        return ({item["stream"].fileno() for item in self.files.values()}
                | {item[0] for item in self.directories.values()})

    def close(self, first=None):
        for item in reversed(list(self.files.values())):
            try:
                item["stream"].close()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "retirement file FD close also failed: " + _describe(error))
        for item in reversed(list(self.directories.values())):
            fd, item[0] = item[0], None
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException as error:
                    if first is None:
                        first = error
                    else:
                        _note(first, "retirement directory FD close also failed: " + _describe(error))
        return first


def _proc_read(path, maximum, tick, meter):
    tick()
    stream, first, blocks, count = None, None, [], 0
    try:
        stream = open(path, "rb", buffering=0)
        while True:
            tick()
            raw = stream.read(min(65536, maximum + 1 - count))
            if not raw:
                break
            count += len(raw)
            meter["bytes"] += len(raw)
            _need(count <= maximum, "current /proc census read bound exceeded")
            _need(meter["bytes"] <= _MAX_PROC_BYTES, "aggregate /proc census byte bound exceeded")
            blocks.append(raw)
    except BaseException as error:
        first = error
    finally:
        if stream is not None:
            try:
                stream.close()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "current proc census FD close also failed: " + _describe(error))
    if first is not None:
        raise first
    tick()
    return b"".join(blocks)


def _proc_namespace(tick, meter):
    raw = _proc_read("/proc/self/mountinfo", 1 << 20, tick, meter)
    mounts = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 10 and fields[4] == b"/proc":
            split = fields.index(b"-") if b"-" in fields else -1
            _need(split > 5 and len(fields) > split + 3 and fields[split + 1] == b"proc",
                  "current proc mount identity is unknown")
            options = fields[5].split(b",") + fields[split + 3].split(b",")
            _need(all(not item.startswith(b"hidepid=") or item == b"hidepid=0" for item in options),
                  "current proc census has hidden or unknown process visibility")
            mounts.append(fields)
    _need(len(mounts) == 1, "one explicit current proc mount required")
    own, init = os.stat("/proc/self/ns/pid"), os.stat("/proc/1/ns/pid")
    _need(_inode(own) == _inode(init), "current proc/PID namespace is ambiguous")
    status = _proc_read("/proc/self/status", 65536, tick, meter)
    identifiers = [line.split()[1:] for line in status.splitlines() if line.startswith(b"NSpid:")]
    _need(identifiers == [[str(os.getpid()).encode()]],
          "current proc namespace visibility is unknown or nested")
    return {"device": own.st_dev, "inode": own.st_ino,
            "scope": "explicit_current_visible_proc_pid_namespace",
            "host_initial_namespace_attested": False}


def _process_census(targets, excluded_fds, tick):
    """Fail closed on use or inaccessible observations; never skip a whole PID."""
    meter = {"bytes": 0, "pids": 0, "fds": 0}
    try:
        namespace = _proc_namespace(tick, meter)
        with os.scandir("/proc") as entries:
            pids = []
            for entry in entries:
                tick()
                if entry.name.isascii() and entry.name.isdecimal():
                    pids.append(entry.name)
                    _need(len(pids) <= _MAX_PROC_PIDS, "current proc PID bound exceeded")
        for pid in pids:
            tick()
            root = Path("/proc") / pid
            try:
                before = _proc_read(root / "stat", 8192, tick, meter)
                birth = before.rsplit(b")", 1)[1].split()[19]
                for kind in ("exe", "cwd"):
                    try:
                        observed = os.stat(root / kind)
                    except OSError as error:
                        if error.errno not in (errno.ENOENT, errno.ESRCH):
                            raise
                    else:
                        _need(_inode(observed) not in targets, "code cache is in use by current proc " + kind)
                with os.scandir(root / "fd") as entries:
                    for entry in entries:
                        tick()
                        meter["fds"] += 1
                        _need(meter["fds"] <= _MAX_PROC_FDS, "current proc FD bound exceeded")
                        # Only our actual retained FD numbers are excluded.
                        if int(pid) == os.getpid() and entry.name.isdecimal() and int(entry.name) in excluded_fds:
                            continue
                        try:
                            observed = os.stat(entry.path)
                        except OSError as error:
                            if error.errno not in (errno.ENOENT, errno.ESRCH):
                                raise
                        else:
                            _need(_inode(observed) not in targets, "code cache is in use by current proc FD")
                maps = _proc_read(root / "maps", 4 << 20, tick, meter)
                for line in maps.splitlines():
                    tick()
                    fields = line.split(None, 5)
                    _need(len(fields) >= 5, "current proc maps format is unknown")
                    major, minor = fields[3].split(b":")
                    observed = (os.makedev(int(major, 16), int(minor, 16)), int(fields[4]))
                    _need(observed not in targets, "code cache is in use by current proc maps")
                after = _proc_read(root / "stat", 8192, tick, meter)
                _need(after.rsplit(b")", 1)[1].split()[19] == birth,
                      "current proc PID identity changed during census")
                meter["pids"] += 1
            except OSError as error:
                if error.errno not in (errno.ENOENT, errno.ESRCH):
                    raise
                # Only a genuinely gone PID can make a failed census harmless.
                try:
                    root.stat()
                except FileNotFoundError:
                    continue
                raise StorageError("current proc census incomplete for a still-visible PID") from error
        _need(_proc_namespace(tick, meter) == namespace, "current proc namespace changed")
        return {**meter, "namespace": namespace, "target_inodes": len(targets),
                "own_held_fd_numbers_only_excluded": True, "complete": True,
                "limits": list(_LIMITS)}
    except (OSError, ValueError, IndexError) as error:
        raise StorageError("current proc census visibility/permission/format unknown; preserve code cache") from error


def _tree_bytes(root, inputs):
    # Metadata only, under original nofollow directory FDs. In particular a
    # lstat/scandir pathname race cannot walk an unrelated payload or secret.
    seen, meter = set(), {"bytes": 0, "entries": 0}

    def walk(fd, path, depth):
        inputs.tick()
        _need(depth <= 128, "retirement budget directory depth exceeded")
        original = _directory_identity(os.fstat(fd))
        first, children = None, None
        try:
            children = os.scandir(fd)
            for child in children:
                inputs.tick()
                child_path = path / child.name
                review.no_secret(child_path)
                meter["entries"] += 1
                _need(meter["entries"] <= _MAX_TREE_ENTRIES, "retirement output budget entry bound exceeded")
                info = os.stat(child.name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child_fd = os.open(child.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                       dir_fd=fd)
                    child_error = None
                    try:
                        _need(_directory_identity(os.fstat(child_fd)) == _directory_identity(info),
                              "retirement budget child directory changed")
                        walk(child_fd, child_path, depth + 1)
                    except BaseException as error:
                        child_error = error
                    finally:
                        try:
                            os.close(child_fd)
                        except BaseException as error:
                            if child_error is None:
                                child_error = error
                            else:
                                _note(child_error, "retirement budget directory close also failed: " + _describe(error))
                    if child_error is not None:
                        raise child_error
                    _need(_directory_identity(os.stat(child.name, dir_fd=fd, follow_symlinks=False))
                          == _directory_identity(info), "retirement budget directory name changed")
                elif stat.S_ISREG(info.st_mode):
                    if _inode(info) not in seen:
                        seen.add(_inode(info))
                        meter["bytes"] += info.st_size
                elif stat.S_ISLNK(info.st_mode):
                    meter["bytes"] += info.st_size  # Metadata counted; never followed.
                else:
                    raise StorageError("unknown output inode type in retirement budget")
            _need(_directory_identity(os.fstat(fd)) == original,
                  "retirement budget original directory changed")
        except BaseException as error:
            first = error
        finally:
            if children is not None:
                try:
                    children.close()
                except BaseException as error:
                    if first is None:
                        first = error
                    else:
                        _note(first, "retirement budget iterator close also failed: " + _describe(error))
        if first is not None:
            raise first

    walk(inputs.directory(root), root, 0)
    return meter["bytes"]


def _budget(rows, selected, reserved, written, inputs):
    total, extra = 0, max(0, reserved - written)
    for _, row, output in rows:
        inputs.tick()
        _need(type(row.get("reserved_bytes")) is int and 0 < row["reserved_bytes"] <= 18 * GiB
              and type(row.get("catalog_limit_bytes")) is int
              and row["reserved_bytes"] <= row["catalog_limit_bytes"] <= 18 * GiB,
              "original output reservation and total limit required")
        amount = _tree_bytes(output, inputs)
        if row["key"] == selected:
            amount += extra
        _need(amount <= row["reserved_bytes"], "retirement journal exceeds original output reservation")
        total += amount
    _need(all(total <= row["catalog_limit_bytes"] for _, row, _ in rows),
          "retirement journal exceeds original aggregate output budget")
    return total


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


class _Journal:
    def __init__(self, inputs, output, name, reserved):
        self.path, self.parent = output / name, inputs.directory(output, controlled=True)
        self.fd, self.written, self.reserved, self.digest = None, 0, reserved, hashlib.sha256()
        self.fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                          0o600, dir_fd=self.parent)
        try:
            self.before = self.current = os.fstat(self.fd)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                _note(error, "untransferred retirement journal FD close also failed: " + _describe(cleanup))
            raise

    def check(self):
        opened = os.fstat(self.fd)
        named = os.stat(self.path.name, dir_fd=self.parent, follow_symlinks=False)
        _need(stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1
              and _inode(opened) == _inode(self.before) == _inode(named)
              and _directory_identity(opened) == _directory_identity(self.before)
              and _file_identity(opened) == _file_identity(named) == _file_identity(self.current)
              and opened.st_size == self.written,
              "original retirement journal name/inode changed")
        digest, offset = hashlib.sha256(), 0
        while offset < self.written:
            raw = os.pread(self.fd, min(65536, self.written - offset), offset)
            _need(bool(raw), "retirement journal readback truncated")
            digest.update(raw)
            offset += len(raw)
        _need(not os.pread(self.fd, 1, offset) and digest.hexdigest() == self.digest.hexdigest()
              and _file_identity(os.fstat(self.fd)) == _file_identity(self.current),
              "retirement journal held readback hash changed")

    def append(self, event):
        self.check()
        raw = _encode(event)
        _need(self.written + len(raw) <= self.reserved <= _JOURNAL_CAP,
              "retirement journal reservation exhausted")
        offset = 0
        while offset < len(raw):
            written = os.write(self.fd, raw[offset:])
            _need(type(written) is int and written > 0, "retirement journal write made no progress")
            self.digest.update(raw[offset:offset + written])
            self.written += written
            offset += written
        self.current = os.fstat(self.fd)
        os.fsync(self.fd)
        self.check()

    def pin(self):
        self.check()
        return {"path": str(self.path), "bytes": self.written, "sha256": self.digest.hexdigest()}

    def close(self):
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)


def retire_code_cache(plan_pin, *, seconds=300):
    """Apply one exact closed audit plan. A failure is never automatically retried."""
    if sys.platform != "linux":
        raise CodeCacheRetirementUnsupported("code-cache retirement is supported only by native Linux observations")
    inputs = _Inputs(seconds)
    guard = journal = None
    first = result = None
    retired, removed, last_attempt, census = [], 0, None, None
    evidence = {"schema": "rovezero.code-cache-retirement-failure.v1",
                "status": "FAILED_OR_PARTIAL_PRESERVED", "plan": plan_pin,
                "retired_files": retired, "logical_removed_bytes": 0, "partial": False,
                "journal": None, "journal_path": None, "last_attempt": None,
                "physical_completion_attested": False, "physical_space_reclaimed": "unknown",
                "vhd_return": "unknown", "observation_limits": list(_LIMITS)}
    try:
        plan, catalog, code, expected, archive_member = review.read_code_plan(inputs, plan_pin)
        catalog = _path(catalog)
        inputs.directory(catalog, controlled=True)
        guard = CatalogGuard(catalog / "catalog.lock")
        guard.acquire()
        _refuse_custody_registry(catalog)
        rows = _catalog_rows(catalog)
        _need(not any(row["state"] == "active" for _, row, _ in rows),
              "another active catalog row blocks code retirement")
        matches = [entry for entry in rows if entry[1]["key"] == plan["key"]]
        _need(len(matches) == 1, "exact original retirement output row missing")
        record_path, row, output = matches[0]
        output = _path(output)
        inputs.directory(output, controlled=True)
        original = read_record_snapshot(record_path, schema="rovezero.managed-output.v1")
        _need(original[1] == plan["record_sha256"] and original[0] == row,
              "original output row raw pin changed before retirement")
        _need(row["state"] == "retained" and row.get("tree_gone") is True
              and row.get("budget_error") is None and row.get("deferred") is False,
              "only original completed nondeferred budget-clean output may be retired")
        observer = lifetime()
        _need(owner_proven_dead(row.get("pid"), row.get("lifetime"), observer),
              "original owner death not proven; no adoption")
        snapshots = {str(path): read_record_snapshot(path, schema="rovezero.managed-output.v1")
                     for path, _, _ in rows}
        for path in (record_path, output / ".rovezero-output.json"):
            inputs.open({"path": str(path), "bytes": path.lstat().st_size,
                         "sha256": plan["record_sha256"]}, 8192)
        keepers = [_path(plan[name]["path"]) for name in ("artifact", "research")]
        for keeper in keepers:
            _need(not keeper.is_relative_to(catalog) and not catalog.is_relative_to(keeper)
                  and not keeper.is_relative_to(output) and not output.is_relative_to(keeper),
                  "preservation keeper archive must be outside selected output and catalog")
        verified_research = review.research(inputs, plan["research"])
        candidates = []
        for target in code:
            path = _path(output / target["relative"])
            _need(path.is_relative_to(output), "candidate escapes original output")
            inputs.directory(path.parent, controlled=True)
            stream = inputs.open({"path": str(path), "bytes": target["bytes"], "sha256": target["sha256"]}, review.ELF_CAP)
            _need(stream.read(4) == b"\x7fELF", "candidate must contain actual ELF bytes")
            inputs.candidates.add(path)
            candidates.append(path)
        review.canonical_code(inputs, plan["artifact"], archive_member, expected)
        target_inodes = {_inode(inputs.files[path]["before"]) for path in candidates}
        _need(len(target_inodes) == len(candidates), "candidate ELF inode aliases are forbidden")
        _need(all(_inode(item["before"]) not in target_inodes
                  for path, item in inputs.files.items() if path not in inputs.candidates),
              "keeper/control and selected candidate inode overlap is forbidden")
        # Directories protect future use through an existing output cwd/FD too.
        targets = target_inodes | {_inode(inputs.directories[path][1])
                                   for path in {output, *(candidate.parent for candidate in candidates)}}

        def recheck():
            inputs.check()
            guard.assert_held()
            _refuse_custody_registry(catalog)
            current = _catalog_rows(catalog)
            _need(current == rows and all(read_record_snapshot(path, schema="rovezero.managed-output.v1")
                      == snapshots[str(path)] for path, _, _ in current),
                  "original catalog row/marker roster changed during retirement")
            _need(lifetime() == observer
                  and owner_proven_dead(row.get("pid"), row.get("lifetime"), observer),
                  "original owner death or observer lifetime changed")
            if journal is not None:
                journal.check()

        recheck()
        census = _process_census(targets, inputs.fds() | {guard.stream.fileno()}, inputs.tick)
        recheck()
        header = {"schema": "rovezero.code-cache-retirement-journal.v1", "event": "preflight_complete",
                  "plan": plan_pin, "record_sha256": plan["record_sha256"], "key": plan["key"],
                  "artifact": plan["artifact"], "research": plan["research"], "code": code,
                  "candidate_inodes": [list(_inode(inputs.files[path]["before"])) for path in candidates],
                  "census": census, "physical_completion_attested": False,
                  "observation_limits": list(_LIMITS)}
        reserved = len(_encode(header)) + len(code) * 1024 + 16384
        _need(reserved <= _JOURNAL_CAP, "bounded retirement journal plan required")
        _budget(rows, plan["key"], reserved, 0, inputs)
        name = ".rovezero-code-retirement-" + plan_pin["sha256"] + ".jsonl"
        evidence["journal_path"] = str(output / name)
        journal = _Journal(inputs, output, name, reserved)
        journal.append(header)
        os.fsync(inputs.directory(output))  # Durable O_EXCL name before any unlink.
        for index, path in enumerate(candidates):
            recheck()
            _budget(rows, plan["key"], reserved, journal.written, inputs)
            last_attempt = {"index": index, "relative": code[index]["relative"],
                            "unlink_attempted": False, "unlink_returned": False, "directory_fsynced": False}
            journal.append({"event": "unlink_intent", "index": index})
            # Final original FD/name/keeper/catalog checks follow the durable
            # intent. No archive reopen, owner substitution or budget renewal.
            recheck()
            inputs.hash_file(path)
            _process_census(targets, inputs.fds() | {guard.stream.fileno(), journal.fd}, inputs.tick)
            recheck()
            last_attempt["unlink_attempted"] = True
            os.unlink(path.name, dir_fd=inputs.directories[path.parent][0])
            last_attempt["unlink_returned"] = True
            inputs.deleted.add(path)
            retired.append(code[index]["relative"])
            removed += code[index]["bytes"]
            os.fsync(inputs.directories[path.parent][0])
            last_attempt["directory_fsynced"] = True
            inputs.check_file(path)
            journal.append({"event": "unlinked_directory_fsynced", "index": index,
                            "logical_removed_bytes": code[index]["bytes"]})
        recheck()
        journal.append({"event": "complete", "retired_files": len(retired), "logical_removed_bytes": removed})
        result = {"schema": "rovezero.code-cache-retirement-result.v1", "status": "CODE_CACHE_RETIRED",
                  "plan": plan_pin, "retired_files": list(retired), "retired_bytes": removed,
                  "logical_removed_bytes": removed, "journal": journal.pin(),
                  "research": verified_research, "deletion_performed": bool(retired),
                  "original_row_unchanged": True, "reservations_changed": False,
                  "expiry_changed": False, "physical_completion_attested": False,
                  "physical_space_reclaimed": "unknown", "vhd_return": "unknown",
                  "census": census, "observation_limits": list(_LIMITS)}
    except BaseException as error:
        first = error
        if journal is not None:
            try:
                journal.append({"event": "failed_or_partial_preserved", "type": type(error).__name__[:128],
                                "detail": _describe(error), "retired_files": len(retired),
                                "logical_removed_bytes": removed, "last_attempt": last_attempt})
            except BaseException as secondary:
                _note(first, "partial retirement journal append/fsync also failed: " + _describe(secondary))
    finally:
        evidence.update(logical_removed_bytes=removed, partial=bool(retired) or bool(last_attempt and last_attempt["unlink_attempted"]),
                        last_attempt=last_attempt)
        if journal is not None:
            try:
                evidence["journal"] = journal.pin()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "retirement journal pin also unavailable: " + _describe(error))
            try:
                journal.close()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "retirement journal FD close also failed: " + _describe(error))
        first = inputs.close(first)
        if guard is not None:
            try:
                guard.release()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    _note(first, "original catalog guard release also failed: " + _describe(error))
    if first is not None:
        try:
            first.code_cache_retirement = evidence
        except BaseException:
            pass
        _note(first, "retirement evidence preserved: journal=" + str(evidence["journal_path"])
              + "; logical_removed_bytes=" + str(removed) + "; no automatic retry or owner reissue")
        raise first
    return result
