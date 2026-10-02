"""Bounded readers and immutable output runs outside the source repository."""

from dataclasses import dataclass
from pathlib import Path
import re

from .errors import DataError
from .serialization import canonical_bytes, loads


@dataclass(frozen=True)
class Limits:
    max_records: int = 10000
    max_record_bytes: int = 262144
    max_file_bytes: int = 67108864

    def __post_init__(self) -> None:
        for field in ("max_records", "max_record_bytes", "max_file_bytes"):
            if type(getattr(self, field)) is not int or not 1 <= getattr(self, field) <= 2**31 - 1:
                raise DataError("InvalidLimit", field, "limit must be a positive bounded integer")


def read_json(path: Path, *, max_bytes: int) -> object:
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise DataError("FileLimit", "input", "JSON file exceeds byte limit")
    return loads(data)


def read_jsonl(path: Path, limits: Limits, *, hasher=None):
    """Yield line number and value or contextual rejection; never skip silently."""
    total = 0
    with path.open("rb") as stream:
        for line_number in range(1, limits.max_records + 2):
            data = stream.readline(limits.max_record_bytes + 1)
            if not data:
                return
            if hasher is not None:
                hasher.update(data)
            total += len(data)
            if total > limits.max_file_bytes:
                raise DataError("FileLimit", "records", "record stream exceeds total byte limit")
            if line_number > limits.max_records:
                raise DataError("RecordLimit", "records", "record stream exceeds record count limit")
            if len(data) > limits.max_record_bytes:
                raise DataError("RecordByteLimit", f"line:{line_number}", "record exceeds byte limit")
            try:
                yield line_number, loads(data), None
            except DataError as exc:
                yield line_number, None, DataError(exc.code, f"line:{line_number}/{exc.context}", exc.message)


def write_run(output_root: Path, run_id: str, files: dict[str, object], *, source_root: Path) -> Path:
    """Create a fresh run directory, rejecting overwrite and repository outputs."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise DataError("InvalidRunId", "output", "run ID must be a simple name, max 80 characters")
    root = output_root.resolve()
    source = source_root.resolve()
    if root == source or root.is_relative_to(source):
        raise DataError("OutputInSource", "output", "generated reports belong outside the source repository")
    encoded = {}
    for name, value in files.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.json", name):
            raise DataError("InvalidOutputName", "output", "expected a simple JSON filename")
        encoded[name] = canonical_bytes(value) + b"\n"
    root.mkdir(parents=True, exist_ok=True)
    run = root / run_id
    try:
        run.mkdir()
    except FileExistsError as exc:
        raise DataError("RunExists", "output", "immutable run already exists; choose a new run ID") from exc
    try:
        for name, data in encoded.items():
            with (run / name).open("xb") as stream:
                stream.write(data)
    except OSError:
        # Keep a visibly incomplete run for diagnosis; never reuse this ID.
        raise
    return run
