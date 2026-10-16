"""G0 자식의 용량 관측 전용 CUDA tree. Rust FD 권한을 복원하지 않는다.

부모가 SHA로 봉인한 등록에만 연결한다. 일반 training scanner는 이 wrapper를
선택하지 않는다. payload를 읽거나 hash/load하지 않고 한 scan 동안 원 FD를 보유한다.
"""
from contextlib import ExitStack, contextmanager
import copy
import os
from pathlib import Path
import stat

from .generation_data import fields, uint
from .generation_training_cli import (
    _safe_absolute, _staging_bytes as _generic_staging_bytes,
    _output_credit as _generic_output_credit, _terminal_receipt as _generic_terminal_receipt,
)

SCHEMA = "rz-pals-prepared-cuda-artifact-tree-observation/1"
_ORT = ("libonnxruntime.so.1.22.0", "libonnxruntime_providers_shared.so", "libonnxruntime_providers_cuda.so")
_NVIDIA = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcufft.so.11",
    "libcurand.so.10", "libnvrtc.so.12", "libnvrtc-builtins.so.12.8", "libnvJitLink.so.12",
    "libcudnn.so.9", "libcudnn_adv.so.9", "libcudnn_cnn.so.9", "libcudnn_ops.so.9",
    "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_graph.so.9", "libcudnn_heuristic.so.9")
_REGULAR = frozenset((*_ORT, *_NVIDIA))
_ALIASES = {name + ".8.0": name for name in _NVIDIA if name.startswith("libcudnn")}
_FILE_FIELDS = ("filename", "device", "inode", "bytes", "mode", "nlink", "mtime_ns", "ctime_ns")


def validate_prepared_cuda_tree(value, staging_root):
    """순수 선언 검사. 이 자료는 Rust 소유자나 provider 성공을 발급하지 않는다."""
    fields(value, ("schema", "parent", "directory", "files", "aliases"), "G0 prepared CUDA tree observation")
    if value["schema"] != SCHEMA:
        raise ValueError("G0 prepared CUDA observation domain mismatch")
    root = _safe_absolute(str(staging_root))
    for name in ("parent", "directory"):
        item = fields(value[name], ("path", "device", "inode", "mode"), "G0 prepared CUDA directory observation")
        path = _safe_absolute(item["path"])
        uint(item["device"]); uint(item["inode"]); uint(item["mode"], 0o7777)
        if not item["inode"] or path == root or not path.is_relative_to(root):
            raise ValueError("G0 prepared CUDA tree must be a strict managed subtree")
    parent, directory = value["parent"], value["directory"]
    if (Path(directory["path"]).parent != Path(parent["path"])
            or parent["mode"] & 0o077 or directory["mode"] != 0o500
            or (parent["device"], parent["inode"]) == (directory["device"], directory["inode"])):
        raise ValueError("G0 prepared CUDA parent/entry identity or readonly mode differs")
    if type(value["files"]) is not list or len(value["files"]) != 19:
        raise ValueError("G0 prepared CUDA tree needs its exact nineteen regular observations")
    if type(value["aliases"]) is not list or len(value["aliases"]) not in (0, 8):
        raise ValueError("G0 prepared CUDA aliases must retain closed V1 or complete V2 scope")
    names, identities, pairs = set(), set(), {}
    for alias, rows in ((False, value["files"]), (True, value["aliases"])):
        for row in rows:
            fields(row, (*_FILE_FIELDS, "target") if alias else _FILE_FIELDS, "G0 prepared CUDA inode observation")
            name = row["filename"]
            if type(name) is not str or name not in (_ALIASES if alias else _REGULAR) or name in names:
                raise ValueError("G0 prepared CUDA filename is undeclared or duplicated")
            names.add(name)
            for key in ("device", "inode", "bytes", "mode", "nlink", "mtime_ns", "ctime_ns"):
                uint(row[key], 0o7777 if key == "mode" else 2**64 - 1)
            identity = row["device"], row["inode"]
            if not row["inode"] or identity in identities or row["nlink"] != 1:
                raise ValueError("G0 prepared CUDA observations contain an inode alias")
            identities.add(identity)
            if alias:
                if row["target"] != _ALIASES[name] or row["mode"] != 0o777 or row["bytes"] != len(row["target"].encode()):
                    raise ValueError("G0 prepared CUDA alias has a nonexact relative target")
                pairs[name] = row["target"]
            elif row["mode"] != 0o400 or not 0 < row["bytes"] <= 1024**3:
                raise ValueError("G0 prepared CUDA regular must be readonly and bounded")
    if {row["filename"] for row in value["files"]} != _REGULAR or (pairs and pairs != _ALIASES):
        raise ValueError("G0 prepared CUDA closed roster differs")
    if sum(row["bytes"] for row in value["files"]) > 4 * 1024**3:
        raise ValueError("G0 prepared CUDA payload exceeds original bundle limit")
    return copy.deepcopy(value)


def _matches(row, info, *, alias=False):
    return ((stat.S_ISLNK(info.st_mode) if alias else stat.S_ISREG(info.st_mode))
        and (info.st_dev, info.st_ino, info.st_size, stat.S_IMODE(info.st_mode), info.st_nlink,
             info.st_mtime_ns, info.st_ctime_ns)
        == tuple(row[key] for key in ("device", "inode", "bytes", "mode", "nlink", "mtime_ns", "ctime_ns")))


def _directory_matches(row, info):
    return stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino, stat.S_IMODE(info.st_mode)) == (row["device"], row["inode"], row["mode"])


def _open_directory(path, stack):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags); stack.callback(os.close, fd)
    for part in path.parts[1:]:
        fd = os.open(part, flags, dir_fd=fd); stack.callback(os.close, fd)
    return fd


class _Observation:
    def __init__(self, declaration, stack):
        if not hasattr(os, "O_PATH"):
            raise ValueError("G0 prepared CUDA observation requires Linux O_PATH")
        self.value = declaration
        self.path = Path(declaration["directory"]["path"])
        self.parent = _open_directory(Path(declaration["parent"]["path"]), stack)
        self.directory = os.open(self.path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.parent)
        stack.callback(os.close, self.directory)
        self.rows = []
        for alias, rows in ((False, declaration["files"]), (True, declaration["aliases"])):
            for row in rows:
                fd = os.open(row["filename"], os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.directory)
                stack.callback(os.close, fd)
                self.rows.append((row, fd, alias))
        self.verify()

    def verify(self):
        for key, fd in (("parent", self.parent), ("directory", self.directory)):
            row = self.value[key]
            with ExitStack() as check_stack:
                named = _open_directory(Path(row["path"]), check_stack)
                if not _directory_matches(row, os.fstat(fd)) or not _directory_matches(row, os.fstat(named)):
                    raise ValueError("G0 prepared CUDA original directory binding changed")
        names = set()
        fresh = os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.directory)
        try:
            with os.scandir(fresh) as entries:
                for entry in entries:
                    if len(names) >= len(self.rows) or entry.name in names:
                        raise ValueError("G0 prepared CUDA entry has an extra inode")
                    names.add(entry.name)
        finally:
            os.close(fresh)
        if names != {row["filename"] for row, _, _ in self.rows}:
            raise ValueError("G0 prepared CUDA entry has a missing or extra inode")
        identities, total = [], 0
        for row, fd, alias in self.rows:
            retained, named = os.fstat(fd), os.stat(row["filename"], dir_fd=self.directory, follow_symlinks=False)
            if not _matches(row, retained, alias=alias) or not _matches(row, named, alias=alias):
                raise ValueError("G0 prepared CUDA original file or alias metadata changed")
            if alias and os.readlink(row["filename"], dir_fd=self.directory) != row["target"]:
                raise ValueError("G0 prepared CUDA relative alias target changed")
            total += retained.st_size
            identities.append((retained.st_dev, retained.st_ino))
        return {"bytes": total, "entries": len(self.rows), "identities": identities}

    def inspect(self, path, info):
        if path != self.path:
            return None
        if not _directory_matches(self.value["directory"], info):
            raise ValueError("G0 prepared CUDA delegated root inode changed")
        return self.verify()

    def finish(self, matches):
        if matches != 1:
            raise ValueError("G0 prepared CUDA subtree was not observed exactly once")
        self.verify()


@contextmanager
def _observer(registration):
    if "prepared_cuda_tree" not in registration:
        yield None
        return
    declaration = validate_prepared_cuda_tree(registration["prepared_cuda_tree"], registration["staging_root"])
    with ExitStack() as stack:
        yield _Observation(declaration, stack)


def staging_bytes(registration):
    with _observer(registration) as tree:
        return _generic_staging_bytes(registration) if tree is None else _generic_staging_bytes(registration, delegated_tree=tree)


def output_credit(registration, *, reserved_bytes=0):
    with _observer(registration) as tree:
        return _generic_output_credit(registration, reserved_bytes=reserved_bytes) if tree is None else _generic_output_credit(registration, reserved_bytes=reserved_bytes, delegated_tree=tree)


def terminal_receipt(registration, result):
    with _observer(registration) as tree:
        return _generic_terminal_receipt(registration, result) if tree is None else _generic_terminal_receipt(registration, result, delegated_tree=tree)
