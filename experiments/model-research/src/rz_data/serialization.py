"""Strict and deterministic JSON for manifests, records and audit receipts."""

import hashlib
import json

from .errors import DataError


def canonical_bytes(value: object, *, max_bytes: int | None = None) -> bytes:
    try:
        if max_bytes is None:
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8")
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False)
        output = bytearray()
        for chunk in encoder.iterencode(value):
            encoded = chunk.encode("utf-8")
            if len(output) + len(encoded) > max_bytes:
                raise DataError("OutputByteLimit", "output", "JSON report exceeds byte limit")
            output.extend(encoded)
        return bytes(output)
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, DataError):
            raise
        code = "NonFinite" if str(exc).startswith("Out of range float values") else "InvalidJson"
        raise DataError(code, "json", str(exc)) from exc


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError("DuplicateKey", "json", f"duplicate key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise DataError("NonFinite", "json", f"non-finite number: {value}")


def loads(data: str | bytes) -> object:
    try:
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
        canonical_bytes(value)  # also catches numeric overflow such as 1e999
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, DataError):
            raise
        raise DataError("InvalidJson", "json", str(exc)) from exc
