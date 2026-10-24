#!/usr/bin/env python3
"""Pin public control files before a separate finite unit's first observer.

This explicit setup profile replaces no Source9/17 predicate. On success execv
preserves the original caller argv and PID; its caller and external finite unit
continue to own supervision, cancellation, normal finish and namespace disposal.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import sys
import time


SCHEMA = "rovent.control-input-snapshot/1"
PROFILE = "systemd-private-tmp-file-bind-before-observer-v1"
RECEIPT_SCHEMA = "rovent.control-input-snapshot-setup/1"
LIMIT = 16 << 20
FILE_LIMIT = 1 << 20
FILE_COUNT = 128
CHUNK = 65536
SOURCE9 = frozenset(("storage_guard.py", "managed_storage.py", "managed_outputs.py",
    "managed_output_quarantine.py", "managed_output_quarantine_transition.py",
    "managed_output_preservation_session.py", "managed_output_unattributed_preservation.py",
    "managed_process.py", "run_managed.py"))
REPORT_SUFFIXES = {"report_python": ".py", "report_rust": ".rs", "report_json": ".json",
    "report_stdout": ".stdout", "report_stderr": ".stderr", "report_historical_log": ".log"}
DENIED_PARTS = frozenset(("models", "checkpoints", "datasets", "node_modules", "build", "dist",
    "dist-server", "coverage", ".cache", "outputs-v1", "catalog", "catalog-v1"))
DENIED_NAMES = frozenset(("owner.json", "record.json", "catalog.json", "journal.json",
    ".rovezero-output.json", ".rovezero-preservation-session.jsonl", "id_rsa", "id_dsa",
    "id_ecdsa", "id_ed25519", "authorized_keys", "known_hosts"))
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC = 1, 2, 4, 8
MS_REMOUNT, MS_BIND = 32, 4096
UMOUNT_NOFOLLOW = 8
SAFE_FLAGS = MS_NOSUID | MS_NODEV | MS_NOEXEC
READONLY_OPTIONS = frozenset(("ro", "nosuid", "nodev", "noexec"))


class SnapshotError(ValueError):
    pass


def need(value, detail):
    if not value:
        raise SnapshotError(detail)


def canonical_path(value):
    need(type(value) is str and 1 < len(value) <= 4096 and not any(ord(c) < 32 for c in value)
        and "\\" not in value, "plain absolute public input path required")
    path = PurePosixPath(value)
    need(path.is_absolute() and str(path) == value and ".." not in path.parts,
        "canonical POSIX path required")
    for part in path.parts:
        low = part.casefold()
        need(not low.startswith(".env") and low not in DENIED_NAMES and low not in DENIED_PARTS
            and not any(word in low for word in ("credential", "service-account", "service_account",
                "api-key", "api_key", "ssh-key"))
            and not low.endswith((".pem", ".key", ".p12", ".pfx")), "secret/heavy/storage input forbidden")
    return path


def raw_pin(value, *, allow_empty=False):
    need(type(value) is dict and set(value) == {"path", "bytes", "sha256"}, "exact raw3 required")
    canonical_path(value["path"])
    need(type(value["bytes"]) is int and (0 if allow_empty else 1) <= value["bytes"] <= FILE_LIMIT
        and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
        "bounded literal raw3 required")
    if value["bytes"] == 0:
        need(value["sha256"] == hashlib.sha256(b"").hexdigest(), "empty stream pin mismatch")
    return dict(value)


def strict_json(raw):
    need(type(raw) is bytes and 0 < len(raw) <= FILE_LIMIT, "bounded public manifest bytes required")
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate manifest field")
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(SnapshotError("nonfinite manifest")))


@dataclass(frozen=True)
class Input:
    kind: str
    path: str
    bytes: int
    sha256: str

    def pin(self):
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class Plan:
    report_root: str
    tooling_root: str
    setup_receipt_path: str
    setup_seconds: int
    inputs: tuple[Input, ...]
    caller_argv: tuple[str, ...]


def flag_pin(argv, name):
    need(argv.count(name) == 1, "original caller flag must occur exactly once: " + name)
    index = argv.index(name)
    need(index + 3 < len(argv), "original flag raw3 missing")
    try:
        size = int(argv[index + 2])
    except ValueError as error:
        raise SnapshotError("original flag size is not decimal") from error
    need(str(size) == argv[index + 2], "canonical original flag size required")
    return raw_pin({"path": argv[index + 1], "bytes": size, "sha256": argv[index + 3]})


def validate_manifest(value, literal_argv, *, manifest_path, wrapper_path):
    need(type(value) is dict and set(value) == {"schema", "profile", "report_root", "tooling_root",
        "maximum_bytes", "setup_seconds", "inputs", "caller_argv", "setup_receipt_path"},
        "closed snapshot manifest required")
    need(value["schema"] == SCHEMA and value["profile"] == PROFILE
        and type(value["maximum_bytes"]) is int and value["maximum_bytes"] == LIMIT
        and type(value["setup_seconds"]) is int and 1 <= value["setup_seconds"] <= 120,
        "explicit16MiB finite setup profile required")
    report = canonical_path(value["report_root"])
    tooling = canonical_path(value["tooling_root"])
    need(report != tooling and report not in tooling.parents and tooling not in report.parents,
        "disjoint declared report/tooling roots required")
    need(canonical_path(manifest_path).parent == report and PurePosixPath(manifest_path).suffix == ".json",
        "manifest must be an explicit report-root JSON")
    receipt = canonical_path(value["setup_receipt_path"])
    need(receipt.parent == report and receipt.suffix == ".json" and str(receipt) != manifest_path,
        "new separate report-root setup receipt required")
    need(type(value["inputs"]) is list and 1 <= len(value["inputs"]) <= FILE_COUNT,
        "explicit bounded allowlist required; no recursive collection")
    inputs, paths, total = [], set(), 0
    for row in value["inputs"]:
        need(type(row) is dict and set(row) == {"kind", "pin"} and type(row["kind"]) is str,
            "input kind and original raw3 required")
        kind = row["kind"]
        pin = raw_pin(row["pin"], allow_empty=kind in ("report_stdout", "report_stderr"))
        path = PurePosixPath(pin["path"])
        if kind in REPORT_SUFFIXES:
            need(path.parent == report and path.suffix == REPORT_SUFFIXES[kind]
                and not path.name.startswith("."), "only explicit public report-root controls allowed")
        elif kind == "tooling_git_pointer":
            need(path == tooling / ".git" and pin["bytes"] <= 4096,
                "only the one bounded public worktree Git pointer allowed")
        elif kind == "tooling_source9":
            need(path.parent == tooling / "scripts" and path.name in SOURCE9 and pin["bytes"] <= 128 << 10,
                "only exact original tooling9 source files allowed")
        else:
            raise SnapshotError("unsupported input kind")
        need(pin["path"] not in paths and pin["path"] not in (manifest_path, wrapper_path, str(receipt)),
            "duplicate/overlapping bootstrap, receipt or input path")
        need(all(path not in PurePosixPath(p).parents and PurePosixPath(p) not in path.parents for p in paths),
            "input parent/child overlap")
        paths.add(pin["path"])
        total += pin["bytes"]
        need(total <= LIMIT, "declared snapshot exceeds16MiB")
        inputs.append(Input(kind, **pin))
    argv = value["caller_argv"]
    need(type(argv) is list and 5 <= len(argv) <= 128
        and all(type(v) is str and v and not any(ord(c) < 32 for c in v) for v in argv)
        and sum(map(len, argv)) <= 32768 and list(literal_argv) == argv,
        "literal CLI original caller argv must exactly match manifest")
    need(argv[0] in ("/usr/bin/python3", "/usr/bin/python3.12") and argv[1:4] == ["-I", "-S", "-B"],
        "explicit isolated original Python caller required; no executable alias rewrite")
    by_path = {item.path: item for item in inputs}
    own = flag_pin(argv, "--self-pin")
    request = flag_pin(argv, "--request-pin")
    need(own["path"] == argv[4] and own["path"] in by_path
        and by_path[own["path"]].kind == "report_python" and own == by_path[own["path"]].pin()
        and request["path"] in by_path and by_path[request["path"]].kind == "report_json"
        and request == by_path[request["path"]].pin(), "same original caller/request raw3 and source required")
    need(argv.count("--result") == 1 and argv.index("--result") + 1 < len(argv), "one original result destination required")
    result = canonical_path(argv[argv.index("--result") + 1])
    need(result.parent == report and result.suffix == ".json" and str(result) not in paths
        and str(result) not in (manifest_path, str(receipt), wrapper_path), "result must remain a separate writable destination")
    return Plan(str(report), str(tooling), str(receipt), value["setup_seconds"], tuple(inputs), tuple(argv))


def unescape_mount(value):
    def replace(match):
        return chr(int(match.group(1), 8))
    decoded = re.sub(r"\\(040|011|012|134)", replace, value)
    need("\\" not in decoded, "unsupported mountinfo escape")
    return decoded


def mount_rows(raw):
    need(type(raw) is str and 0 < len(raw) <= 4 << 20, "bounded mountinfo required")
    rows = []
    for line in raw.splitlines():
        fields = line.split()
        need("-" in fields, "mountinfo separator missing")
        split = fields.index("-")
        need(split >= 6 and len(fields) >= split + 4, "malformed mountinfo")
        rows.append({"id": int(fields[0]), "device": fields[2], "root": unescape_mount(fields[3]),
            "target": unescape_mount(fields[4]), "options": frozenset(fields[5].split(",")),
            "optional": tuple(fields[6:split]), "fstype": fields[split + 1]})
    need(len({row["id"] for row in rows}) == len(rows), "duplicate mount identity")
    return rows


def isolation_proof(*, uid, gid, cap_effective, current_namespace, init_namespace, rows, tmp_info):
    need(uid == gid == 0 and type(cap_effective) is int and cap_effective & (1 << 21),
        "root and effective CAP_SYS_ADMIN required")
    need(type(current_namespace) is str and type(init_namespace) is str
        and re.fullmatch(r"mnt:\[\d+\]", current_namespace)
        and re.fullmatch(r"mnt:\[\d+\]", init_namespace)
        and current_namespace != init_namespace, "a separate mount namespace is required")
    need(not any(any(v.startswith(("shared:", "master:", "propagate_from:")) for v in row["optional"])
        for row in rows), "mount propagation must already be private")
    tmp = [row for row in rows if row["target"] == "/tmp"]
    need(len(tmp) == 1 and re.fullmatch(r"/(?:var/)?tmp/systemd-private-[A-Za-z0-9_.:-]+/tmp", tmp[0]["root"])
        and "rw" in tmp[0]["options"], "explicit systemd PrivateTmp bind geometry required")
    need(stat.S_ISDIR(tmp_info.st_mode) and tmp_info.st_uid == tmp_info.st_gid == 0
        and stat.S_IMODE(tmp_info.st_mode) == 0o1777, "root-owned private /tmp1777 required")
    return {"mount_namespace": current_namespace, "init_namespace_different": True,
        "private_tmp_mount_id": tmp[0]["id"], "private_tmp_root": tmp[0]["root"],
        "propagation_private": True, "cap_sys_admin_observed": True}


def stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
        info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def directory_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid, info.st_gid)


def parent_identity(info):
    # Own mkdir/rmdir can change a directory's nlink/mtime/ctime. Its retained
    # inode, type/mode and owner remain authoritative for parent traversal.
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)


class DirectoryCalls:
    """Only directory traversal syscalls; pure fixtures supply a fake object."""
    def open_root(self):
        return os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)

    def open_directory(self, parent, name):
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)

    def root_stat(self):
        return os.stat("/", follow_symlinks=False)

    def stat_at(self, parent, name):
        return os.stat(name, dir_fd=parent, follow_symlinks=False)

    def fstat(self, fd):
        return os.fstat(fd)

    def close(self, fd):
        os.close(fd)


class ParentChain:
    """Keep every actual directory FD; do not follow a later parent spelling."""
    def __init__(self, path, calls=None):
        self.path, self.calls = PurePosixPath(path), DirectoryCalls() if calls is None else calls
        self.rows, self.closed, self.close_errors = [], False, []
        try:
            fd = self.calls.open_root()
            self.rows.append((fd, None, None, None))  # Transfer before fstat.
            before = self.calls.fstat(fd)
            need(stat.S_ISDIR(before.st_mode), "root anchor must be a directory")
            self.rows[-1] = (fd, None, None, parent_identity(before))
            for name in self.path.parts[1:]:
                parent = self.rows[-1][0]
                fd = self.calls.open_directory(parent, name)
                self.rows.append((fd, parent, name, None))
                before = self.calls.fstat(fd)
                need(stat.S_ISDIR(before.st_mode)
                    and parent_identity(before) == parent_identity(self.calls.stat_at(parent, name)),
                    "plain directory FD/name anchor required")
                self.rows[-1] = (fd, parent, name, parent_identity(before))
            self.recheck()
        except BaseException as primary:
            for error in self.close():
                primary.add_note(error)
            raise

    @property
    def fd(self):
        need(not self.closed and self.rows, "live actual parent DirFD required")
        return self.rows[-1][0]

    def recheck(self):
        need(not self.closed, "parent custody is closed")
        for fd, parent, name, expected in self.rows:
            current = self.calls.root_stat() if parent is None else self.calls.stat_at(parent, name)
            need(stat.S_ISDIR(current.st_mode) and expected == parent_identity(current)
                == parent_identity(self.calls.fstat(fd)), "parent directory FD/name changed or became symlink")

    def close(self):
        if self.closed:
            return list(self.close_errors)
        self.closed = True
        rows, self.rows = self.rows, []
        for fd, _, _, _ in reversed(rows):
            try:
                self.calls.close(fd)
            except BaseException as error:
                self.close_errors.append("parent DirFD.close unknown: " + repr(error))
        return list(self.close_errors)


def require_receipt_parent(path, parent):
    need(type(parent) is ParentChain and parent.path == PurePosixPath(path).parent,
        "receipt must use its own exact held parent capability")
    parent.recheck()


class NativeOps:
    """Small libc mount boundary and owned setup FDs; no caller supervision."""
    def __init__(self):
        self.started = time.monotonic()
        self.namespace = None
        self.private_dir = None
        self.mounts = []
        self.fds = []
        self.anchors = []
        self.parents = []
        self.sources = {}
        self.tmp_parent = None
        self.tmp_name = None
        self.tmpfs_fd = None
        self.fd_close_errors = []
        self.cancel_requested = None
        self.until = self.started + 120
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_ulong, ctypes.c_void_p]
        self.libc.mount.restype = ctypes.c_int
        self.libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.libc.umount2.restype = ctypes.c_int

    def tick(self):
        need(self.cancel_requested is None, "setup cancellation requested: " + str(self.cancel_requested))
        need(self.until is None or time.monotonic() < self.until, "original finite setup deadline expired")
        if self.namespace is not None:
            need(os.readlink("/proc/self/ns/mnt") == self.namespace, "setup mount namespace changed")

    def rows(self):
        with open("/proc/self/mountinfo", "r", encoding="utf-8") as stream:
            return mount_rows(stream.read((4 << 20) + 1))

    def environment(self):
        need(sys.platform == "linux" and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,
            "isolated Linux finite-unit wrapper with -I -S -B required")
        with open("/proc/self/status", "r", encoding="ascii") as stream:
            status = stream.read(65537)
        need(len(status) <= 65536, "bounded self status required")
        cap = [line.split()[1] for line in status.splitlines() if line.startswith("CapEff:")]
        need(len(cap) == 1, "effective capabilities unavailable")
        proof = isolation_proof(uid=os.geteuid(), gid=os.getegid(), cap_effective=int(cap[0], 16),
            current_namespace=os.readlink("/proc/self/ns/mnt"), init_namespace=os.readlink("/proc/1/ns/mnt"),
            rows=self.rows(), tmp_info=os.lstat("/tmp"))
        self.namespace = proof["mount_namespace"]
        return proof

    def _mount(self, source, target, fstype, flags, data=None):
        encode = lambda item: None if item is None else os.fsencode(item)
        if self.libc.mount(encode(source), encode(target), encode(fstype), flags, encode(data)) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(target))

    def _row(self, target):
        rows = [row for row in self.rows() if row["target"] == str(target)]
        need(len(rows) == 1, "one actual setup mount required: " + str(target))
        return rows[0]

    def _row_fd(self, fd):
        with open("/proc/self/fdinfo/" + str(fd), "r", encoding="ascii") as stream:
            raw = stream.read(4097)
        need(len(raw) <= 4096, "bounded own FD mount observation required")
        identities = [int(line.split()[1]) for line in raw.splitlines() if line.startswith("mnt_id:")]
        need(len(identities) == 1, "own FD mount ID unavailable")
        rows = [row for row in self.rows() if row["id"] == identities[0]]
        need(len(rows) == 1, "held FD must belong to one observed mount")
        return rows[0]

    def _anchor(self, fd):
        self.anchors.append(fd)
        return fd

    def parent(self, path):
        parent = ParentChain(path)
        self.parents.append(parent)
        return parent

    def make_tmpfs(self):
        self.tick()
        self.tmp_parent = self.parent("/tmp")
        self.tmp_name = "rovezero-control-snapshot-" + os.urandom(16).hex()
        self.tmp_parent.recheck()
        os.mkdir(self.tmp_name, 0o700, dir_fd=self.tmp_parent.fd)
        directory = "/tmp/" + self.tmp_name
        self.private_dir = directory
        original = self._anchor(os.open(self.tmp_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self.tmp_parent.fd))
        self.tmp_parent.recheck()
        need(directory_identity(os.fstat(original)) == directory_identity(os.stat(self.tmp_name,
            dir_fd=self.tmp_parent.fd, follow_symlinks=False)), "own new tmp directory FD/name changed")
        self._mount("tmpfs", "/proc/self/fd/" + str(original), "tmpfs", SAFE_FLAGS,
            "size=" + str(LIMIT) + ",mode=0700")
        self.mounts.append({"target": directory, "id": None, "fd": None})
        self.tmp_parent.recheck()
        self.tmpfs_fd = self._anchor(os.open(self.tmp_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self.tmp_parent.fd))
        self.mounts[-1]["fd"] = self.tmpfs_fd
        row = self._row_fd(self.tmpfs_fd)
        self.mounts[-1]["id"] = row["id"]
        need(row["fstype"] == "tmpfs" and {"rw", "nosuid", "nodev", "noexec"} <= row["options"],
            "bounded noexec/nodev/nosuid tmpfs required")
        size = os.fstatvfs(self.tmpfs_fd)
        info = os.fstat(self.tmpfs_fd)
        need(0 < size.f_frsize * size.f_blocks <= LIMIT and stat.S_ISDIR(info.st_mode)
            and info.st_uid == info.st_gid == 0 and stat.S_IMODE(info.st_mode) == 0o700,
            "actual private tmpfs capacity/owner/mode required")
        return directory

    def _close(self, fd):
        if fd in self.fds:
            self.fds.remove(fd)
        try:
            os.close(fd)
        except BaseException as error:
            self.fd_close_errors.append("setup FD.close unknown: " + repr(error))
            raise

    def open_source(self, pin):
        self.tick()
        path = Path(pin["path"])
        parent = self.parent(str(path.parent))
        parent.recheck()
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent.fd)
        self.fds.append(fd)
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size == pin["bytes"]
            and stamp(before) == stamp(os.stat(path.name, dir_fd=parent.fd, follow_symlinks=False)),
            "original bounded regular FD/name required")
        chunks, offset, digest = [], 0, hashlib.sha256()
        while offset < pin["bytes"]:
            self.tick()
            raw = os.pread(fd, min(CHUNK, pin["bytes"] - offset), offset)
            need(bool(raw), "original control early EOF")
            chunks.append(raw)
            digest.update(raw)
            offset += len(raw)
        need(os.pread(fd, 1, offset) == b"" and digest.hexdigest() == pin["sha256"], "original control raw3 mismatch")
        item = {"fd": fd, "path": path, "name": path.name, "parent": parent,
            "stamp": stamp(before), "raw": b"".join(chunks)}
        self.recheck_source(item)
        self.sources[str(path)] = item
        return item

    def recheck_source(self, item):
        self.tick()
        item["parent"].recheck()
        need(item["stamp"] == stamp(os.fstat(item["fd"])) == stamp(os.stat(item["name"],
            dir_fd=item["parent"].fd, follow_symlinks=False)),
            "original control FD/name changed before binding")

    def copy_source(self, directory, index, item, pin):
        self.tick()
        need(directory == self.private_dir and self.tmpfs_fd is not None, "actual own tmpfs DirFD required")
        name = "input-%03d" % index
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600, dir_fd=self.tmpfs_fd)
        self.fds.append(fd)
        raw, offset = item["raw"], 0
        while offset < len(raw):
            self.tick()
            written = os.write(fd, raw[offset:offset + CHUNK])
            need(written > 0, "snapshot write made no progress")
            offset += written
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        before = os.fstat(fd)
        need(before.st_uid == before.st_gid == 0 and before.st_nlink == 1 and before.st_size == pin["bytes"]
            and stat.S_IMODE(before.st_mode) == 0o444 and stamp(before) == stamp(os.stat(name,
                dir_fd=self.tmpfs_fd, follow_symlinks=False)), "private copy identity mismatch")
        digest, offset = hashlib.sha256(), 0
        while offset < pin["bytes"]:
            self.tick()
            part = os.pread(fd, min(CHUNK, pin["bytes"] - offset), offset)
            need(bool(part), "same copy FD readback early EOF")
            digest.update(part)
            offset += len(part)
        need(os.pread(fd, 1, offset) == b"" and digest.hexdigest() == pin["sha256"]
            and stamp(before) == stamp(os.fstat(fd)) == stamp(os.stat(name, dir_fd=self.tmpfs_fd,
                follow_symlinks=False)), "independent copy raw3 readback failed")
        self.recheck_source(item)
        self._close(fd)  # No writable copy FD survives readonly remount.
        item["raw"] = None
        return {"path": directory + "/" + name, "name": name, "stamp": stamp(before)}

    def close_source(self, item):
        self._close(item["fd"])

    def bind_readonly(self, copied, target):
        self.tick()
        need(target in self.sources, "same actually held original target required")
        original = self.sources[target]
        self.recheck_source(original)
        need(not any(row["target"] == target for row in self.rows()), "pre-existing target mount is unsupported")
        copy_fd = os.open(copied["name"], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.tmpfs_fd)
        self.fds.append(copy_fd)
        need(stamp(os.fstat(copy_fd)) == copied["stamp"], "same copy FD required before mount")
        self._mount("/proc/self/fd/" + str(copy_fd), "/proc/self/fd/" + str(original["fd"]), None, MS_BIND)
        self.mounts.append({"target": target, "id": None, "fd": None, "original": original, "copied": copied})
        original["parent"].recheck()
        bound = self._anchor(os.open(original["name"], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=original["parent"].fd))
        self.mounts[-1]["fd"] = bound
        row = self._row_fd(bound)
        need(row["target"] == target and row["fstype"] == "tmpfs" and stamp(os.fstat(bound)) == copied["stamp"],
            "held original target did not expose the actual copy mount")
        self.mounts[-1]["id"] = row["id"]
        self._mount(None, "/proc/self/fd/" + str(bound), None, MS_BIND | MS_REMOUNT | MS_RDONLY | SAFE_FLAGS)
        self._close(copy_fd)
        return self.verify_binding(copied, target)

    def verify_binding(self, copied, target):
        self.tick()
        records = [entry for entry in self.mounts if entry["target"] == target]
        need(len(records) == 1 and records[0]["fd"] is not None, "own actual file mount FD required")
        record = records[0]
        original = record["original"]
        original["parent"].recheck()
        row = self._row_fd(record["fd"])
        need(row["fstype"] == "tmpfs" and READONLY_OPTIONS <= row["options"]
            and row["id"] == record["id"] and row["target"] == target
            and copied["stamp"] == stamp(os.fstat(record["fd"]))
            == stamp(os.stat(original["name"], dir_fd=original["parent"].fd, follow_symlinks=False))
            == stamp(os.stat(copied["name"], dir_fd=self.tmpfs_fd, follow_symlinks=False)),
            "actual readonly file mapping must expose the exact copied inode")
        return {"logical_path": target, "copied_identity": list(copied["stamp"]),
            "mount_id": row["id"], "readonly": True, "noexec": True, "nodev": True, "nosuid": True}

    def seal_tmpfs(self, directory):
        self.tick()
        need(directory == self.private_dir and self.tmpfs_fd is not None, "actual tmpfs FD required")
        self.tmp_parent.recheck()
        self._mount(None, "/proc/self/fd/" + str(self.tmpfs_fd), None, MS_REMOUNT | MS_RDONLY | SAFE_FLAGS)
        row = self._row_fd(self.tmpfs_fd)
        need(row["fstype"] == "tmpfs" and READONLY_OPTIONS <= row["options"], "entire private snapshot must be readonly")
        return {"path": directory, "mount_id": row["id"], "readonly": True,
            "maximum_bytes": LIMIT, "noexec": True, "nodev": True, "nosuid": True}

    def close_setup_fds(self):
        for fd in list(reversed(self.fds)):
            try:
                self._close(fd)
            except BaseException:
                pass  # _close preserved unknown and consumed this integer once.
        return list(self.fd_close_errors)

    def cleanup(self):
        # No force/lazy unmount and no foreign mount adoption. The external
        # finite namespace remains responsible for unknown residual mounts.
        errors = self.close_setup_fds()
        for row in reversed(self.mounts):
            try:
                need(os.readlink("/proc/self/ns/mnt") == self.namespace, "cleanup namespace changed")
                need(row["fd"] is not None and row["id"] is not None, "own mount FD identity is unknown")
                if "original" in row:
                    row["original"]["parent"].recheck()
                    need(stamp(os.fstat(row["fd"])) == row["copied"]["stamp"], "own bound inode changed")
                else:
                    self.tmp_parent.recheck()
                actual = self._row_fd(row["fd"])
                need(actual["id"] == row["id"], "own mount identity changed")
                if "original" in row:
                    parent, name = row["original"]["parent"], row["original"]["name"]
                    need(stamp(os.stat(name, dir_fd=parent.fd, follow_symlinks=False)) == row["copied"]["stamp"],
                        "own mount is no longer at its held parent/name")
                else:
                    parent, name = self.tmp_parent, self.tmp_name
                    need(directory_identity(os.stat(name, dir_fd=parent.fd, follow_symlinks=False))
                        == directory_identity(os.fstat(row["fd"])), "own tmpfs parent/name changed")
                descriptor = row["fd"]
                row["fd"] = None  # Never retry a close-unknown descriptor.
                self.anchors.remove(descriptor)
                os.close(descriptor)
                parent.recheck()
                target = "/proc/self/fd/" + str(parent.fd) + "/" + name
                if self.libc.umount2(os.fsencode(target), UMOUNT_NOFOLLOW) != 0:
                    error = ctypes.get_errno()
                    raise OSError(error, os.strerror(error), row["target"])
                parent.recheck()
                need(not any(current["id"] == row["id"] for current in self.rows()), "own mount ID remains after unmount")
            except BaseException as error:
                errors.append("own setup unmount: " + repr(error))
        if self.private_dir is not None:
            try:
                need(not any(row["target"] == self.private_dir for row in self.rows()), "snapshot mount remains")
                self.tmp_parent.recheck()
                os.rmdir(self.tmp_name, dir_fd=self.tmp_parent.fd)
            except BaseException as error:
                errors.append("own setup directory.rmdir: " + repr(error))
        for fd in reversed(self.anchors):
            try:
                os.close(fd)
            except BaseException as error:
                errors.append("mount anchor FD.close unknown: " + repr(error))
        self.anchors.clear()
        for parent in reversed(self.parents):
            errors.extend(parent.close())
        return {"errors": errors, "setup_scope_closed": not errors,
            "product_physical_completion_attested": False}


def prepare_snapshot(plan, ops):
    """Source-owned setup sequence; injectable syscall object for pure fixtures."""
    # Bootstrap and setup share the original start; no clock is renewed after
    # reading the manifest or opening an input.
    ops.until = min(ops.until, ops.started + plan.setup_seconds)
    sources, copies, observations = [], [], []
    try:
        environment = ops.environment()
        directory = ops.make_tmpfs()
        for index, entry in enumerate(plan.inputs):
            ops.tick()
            original = ops.open_source(entry.pin())
            sources.append(original)
            copied = ops.copy_source(directory, index, original, entry.pin())
            copies.append((entry, copied))
            observations.append({"kind": entry.kind, "original_raw_pin": entry.pin(),
                "original_identity_before_binding": list(original["stamp"]),
                "copied_identity": list(copied["stamp"]), "independent_raw_readback": True})
        for original in sources:
            ops.recheck_source(original)
        bindings = [ops.bind_readonly(copied, entry.path) for entry, copied in copies]
        sealed = ops.seal_tmpfs(directory)
        for entry, copied in copies:
            ops.verify_binding(copied, entry.path)
        for original in sources:
            ops.close_source(original)
        sources.clear()
        close_errors = ops.close_setup_fds()
        need(not close_errors, "setup FD close failed: " + repr(close_errors))
        ops.tick()
        return {"schema": RECEIPT_SCHEMA, "profile": PROFILE, "setup_complete": True,
            "environment": environment, "inputs": observations, "readonly_bindings": bindings,
            "snapshot": sealed, "caller_argv": list(plan.caller_argv),
            "exec_intent": "pending_original_execv_not_observed_by_this_receipt",
            "normal_finish_or_product_authority_issued": False, "physical_completion_attested": False,
            "success_cleanup": "delegated_to_original_finite_unit_namespace_disposal_and_external_readback"}
    except BaseException as primary:
        try:
            cleanup = ops.cleanup()
        except BaseException as secondary:
            cleanup = {"errors": [repr(secondary)], "setup_scope_closed": False,
                "product_physical_completion_attested": False}
        primary.snapshot_cleanup = cleanup
        primary.snapshot_partial_inputs = observations
        raise


def write_receipt(path, body, *, parent, previous=None):
    """Own new report only; an exec failure may update this same owned receipt."""
    raw = (json.dumps(body, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
    need(len(raw) <= FILE_LIMIT, "bounded setup receipt required")
    require_receipt_parent(path, parent)
    name = PurePosixPath(path).name
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(name, flags | os.O_NONBLOCK | (os.O_CREAT | os.O_EXCL if previous is None else 0),
        0o600, dir_fd=parent.fd)
    primary = None
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid == before.st_gid == 0
            and stamp(before) == stamp(os.stat(name, dir_fd=parent.fd, follow_symlinks=False)),
            "own setup receipt FD/name required")
        require_receipt_parent(path, parent)
        if previous is not None:
            need(stamp(before) == previous[0] and os.pread(fd, before.st_size + 1, 0) == previous[1],
                "only this wrapper's unchanged new receipt may be updated")
            os.ftruncate(fd, 0)
        offset = 0
        while offset < len(raw):
            count = os.pwrite(fd, raw[offset:], offset)
            need(count > 0, "receipt write made no progress")
            offset += count
        os.fsync(fd)
        after = os.fstat(fd)
        need(os.pread(fd, len(raw) + 1, 0) == raw and stamp(after) == stamp(os.stat(name,
            dir_fd=parent.fd, follow_symlinks=False)),
            "own setup receipt readback changed")
        require_receipt_parent(path, parent)
        os.fsync(parent.fd)
        return stamp(after), raw
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            os.close(fd)
        except BaseException as secondary:
            if primary is None:
                raise
            primary.add_note("setup receipt FD.close secondary: " + repr(secondary))


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    need(len(values) >= 7 and values[5] == "--", "five positional bootstrap values, literal -- and original caller argv required")
    manifest_pin = raw_pin({"path": values[0], "bytes": int(values[1]), "sha256": values[2]})
    wrapper_path = str(Path(__file__).absolute())
    self_pin = raw_pin({"path": wrapper_path, "bytes": int(values[3]), "sha256": values[4]})
    ops, plan, receipt_owned, receipt_parent = NativeOps(), None, None, None
    receipt = {"schema": RECEIPT_SCHEMA, "profile": PROFILE, "setup_complete": False,
        "normal_finish_or_product_authority_issued": False, "physical_completion_attested": False}
    old_handlers = {}
    def interrupted(number, _frame):
        ops.cancel_requested = number  # Denial only; never interrupt mount bookkeeping.
    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            old_handlers[number] = signal.signal(number, interrupted)
        ops.environment()  # Fail closed before input opens or setup writes.
        wrapper = ops.open_source(self_pin)
        manifest = ops.open_source(manifest_pin)
        plan = validate_manifest(strict_json(manifest["raw"]), values[6:],
            manifest_path=manifest_pin["path"], wrapper_path=wrapper_path)
        receipt_parent = ParentChain(str(PurePosixPath(plan.setup_receipt_path).parent))
        ops.recheck_source(wrapper)
        ops.recheck_source(manifest)
        ops.close_source(wrapper)
        ops.close_source(manifest)
        require_receipt_parent(plan.setup_receipt_path, receipt_parent)
        try:
            os.stat(PurePosixPath(plan.setup_receipt_path).name, dir_fd=receipt_parent.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise SnapshotError("setup receipt destination already exists")
        receipt = prepare_snapshot(plan, ops)
        receipt["wrapper_source_pin"] = self_pin
        receipt["manifest_source_pin"] = manifest_pin
        receipt["held_parent_and_mount_anchor_fds"] = "CLOEXEC_until_actual_execv_or_setup_failure_cleanup"
        receipt_owned = write_receipt(plan.setup_receipt_path, receipt, parent=receipt_parent)
        ops.tick()
        require_receipt_parent(plan.setup_receipt_path, receipt_parent)
        for number, previous in old_handlers.items():
            signal.signal(number, previous)
        old_handlers.clear()
        os.execv(plan.caller_argv[0], list(plan.caller_argv))
        raise SnapshotError("execv returned unexpectedly")
    except BaseException as primary:
        cleanup = getattr(primary, "snapshot_cleanup", None)
        if cleanup is None:
            try:
                cleanup = ops.cleanup()
            except BaseException as secondary:
                cleanup = {"errors": [repr(secondary)], "setup_scope_closed": False,
                    "product_physical_completion_attested": False}
        receipt.update({"setup_or_exec_failed": True, "first_error": repr(primary),
            "cleanup": cleanup, "exec_returned_successfully": False,
            "partial_input_observations": getattr(primary, "snapshot_partial_inputs", [])})
        if plan is not None:
            try:
                need(receipt_parent is not None, "actual held receipt parent required")
                write_receipt(plan.setup_receipt_path, receipt, parent=receipt_parent, previous=receipt_owned)
            except BaseException as secondary:
                primary.add_note("separate setup receipt secondary: " + repr(secondary))
        print("CONTROL_INPUT_SNAPSHOT_SETUP_FAILED " + repr(primary), file=sys.stderr)
        for note in getattr(primary, "__notes__", ()):
            print(note, file=sys.stderr)
        return 125
    finally:
        if receipt_parent is not None:
            for error in receipt_parent.close():
                print("CONTROL_INPUT_SNAPSHOT_RECEIPT_CLOSE_UNKNOWN " + error, file=sys.stderr)
        for number, previous in old_handlers.items():
            signal.signal(number, previous)


if __name__ == "__main__":
    raise SystemExit(main())
