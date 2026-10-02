"""Bounded JSON artifact receipts and replayable sampling without pickle.

This is the F-owned persistence boundary. Model and optimizer contents are
validated by the trainer; no runtime registry handles are restored here.
"""

import hashlib
import importlib
from pathlib import Path
import random
import re

from rz_data.errors import DataError
from rz_data.serialization import canonical_bytes, digest, loads


DEFAULT_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_SAMPLER_SIZE = 10000
MAX_EPOCH = 1000000


def _integer(value: object, minimum: int, maximum: int, context: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise DataError("InvalidInteger", context, "integer is outside the supported finite range")
    return value


def _bounded(value: object, max_bytes: int, context: str) -> bytes:
    _integer(max_bytes, 1, 2**31 - 1, context + ".max_bytes")
    try:
        return canonical_bytes(value, max_bytes=max_bytes)
    except DataError as exc:
        raise DataError(exc.code, context + "." + exc.context, exc.message) from exc


def seal(payload: dict, *, max_bytes: int) -> dict:
    """Return a new receipt, reserving space for its digest before acceptance."""
    if type(payload) is not dict:
        raise DataError("InvalidArtifact", "artifact", "payload must be an object")
    if "digest" in payload:
        raise DataError("ExistingDigest", "artifact.digest", "payload must not contain a digest")
    encoded = _bounded(payload, max_bytes, "artifact")
    # Snapshot JSON contents so later optimizer updates cannot alter this receipt.
    sealed = {**loads(encoded), "digest": hashlib.sha256(encoded).hexdigest()}
    _bounded(sealed, max_bytes, "artifact")
    return sealed


def verify(value: object, *, artifact_kind: str, provenance: dict | None = None,
           max_bytes: int = DEFAULT_ARTIFACT_BYTES) -> dict:
    """Verify an envelope and return its payload, leaving semantic checks to F."""
    _bounded(value, max_bytes, "artifact")
    if type(value) is not dict:
        raise DataError("InvalidArtifact", "artifact", "receipt must be an object")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise DataError("UnsupportedArtifactSchema", "artifact.schema_version", "only exact integer schema version 1 is supported")
    if type(artifact_kind) is not str or not artifact_kind:
        raise DataError("InvalidArtifactKind", "artifact_kind", "expected a nonempty artifact kind")
    if type(value.get("artifact_kind")) is not str or value["artifact_kind"] != artifact_kind:
        raise DataError("ArtifactKindMismatch", "artifact.artifact_kind", "artifact kind differs from the requested kind")
    checksum = value.get("digest")
    if type(checksum) is not str or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise DataError("InvalidDigest", "artifact.digest", "expected lowercase SHA256")
    payload = {key: item for key, item in value.items() if key != "digest"}
    encoded = _bounded(payload, max_bytes, "artifact")
    if hashlib.sha256(encoded).hexdigest() != checksum:
        raise DataError("ArtifactDigestMismatch", "artifact.digest", "artifact content differs from its receipt digest")
    if provenance is not None:
        if type(provenance) is not dict:
            raise DataError("InvalidProvenance", "provenance", "expected an object")
        expected = _bounded(provenance, max_bytes, "provenance")
        actual = _bounded(value.get("provenance"), max_bytes, "artifact.provenance")
        if type(value.get("provenance")) is not dict or actual != expected:
            raise DataError("ProvenanceMismatch", "artifact.provenance", "artifact belongs to different source/configuration/data")
    return loads(encoded)


def code_digest() -> str:
    """Identify installed F Python source by package-relative names and raw bytes."""
    sources = []
    for package_name in ("rz_data", "rz_training"):
        package = importlib.import_module(package_name)
        root = Path(package.__file__).resolve().parent
        for source in sorted(root.rglob("*.py")):
            sources.append({
                "path": package_name + "/" + source.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            })
    return digest(sorted(sources, key=lambda item: item["path"]))


class Sampler:
    """Deterministic shuffled epochs with a fully JSON-safe RNG continuation."""

    def __init__(self, size: int, seed: int):
        self.size = _integer(size, 1, MAX_SAMPLER_SIZE, "sampler.size")
        _integer(seed, 0, 2**32 - 1, "sampler.seed")
        self.epoch = 0
        self.cursor = 0
        self.order = list(range(size))
        self._rng = random.Random(seed)
        self._rng.shuffle(self.order)

    def draw(self, count: int) -> list[int]:
        count = _integer(count, 1, 256, "sampler.count")
        wraps = (self.cursor + count - 1) // self.size
        if self.epoch + wraps > MAX_EPOCH:
            raise DataError("SamplerEpochLimit", "sampler.epoch", "draw would exceed the finite epoch limit")
        output = []
        while len(output) < count:
            if self.cursor == self.size:
                self.epoch += 1
                self.cursor = 0
                self._rng.shuffle(self.order)
            take = min(count - len(output), self.size - self.cursor)
            output.extend(self.order[self.cursor:self.cursor + take])
            self.cursor += take
        return output

    def state_dict(self) -> dict:
        version, internal, gaussian = self._rng.getstate()
        return {
            "size": self.size,
            "epoch": self.epoch,
            "order": list(self.order),
            "cursor": self.cursor,
            "rng_state": [version, list(internal), gaussian],
        }

    def load_state_dict(self, state: object) -> None:
        """Validate every field before changing either the sampler or its RNG."""
        fields = {"size", "epoch", "order", "cursor", "rng_state"}
        if type(state) is not dict or set(state) != fields:
            raise DataError("InvalidSamplerState", "sampler", "state must contain exactly the declared fields")
        if type(state["size"]) is not int or state["size"] != self.size:
            raise DataError("SamplerSizeMismatch", "sampler.size", "state belongs to a different dataset size")
        epoch = _integer(state["epoch"], 0, MAX_EPOCH, "sampler.epoch")
        cursor = _integer(state["cursor"], 0, self.size, "sampler.cursor")
        order = state["order"]
        if (type(order) is not list or len(order) != self.size
                or any(type(item) is not int or not 0 <= item < self.size for item in order)
                or len(set(order)) != self.size):
            raise DataError("InvalidSamplerOrder", "sampler.order", "expected an exact permutation of dataset indices")
        rng = state["rng_state"]
        if (type(rng) is not list or len(rng) != 3
                or type(rng[0]) is not int or rng[0] != 3
                or type(rng[1]) is not list or len(rng[1]) != 625
                or rng[2] is not None):
            raise DataError("InvalidRngState", "sampler.rng_state", "expected JSON Random version 3 state with no gaussian cache")
        if (any(type(item) is not int or not 0 <= item <= 2**32 - 1 for item in rng[1][:-1])
                or type(rng[1][-1]) is not int or not 0 <= rng[1][-1] <= 624):
            raise DataError("InvalidRngState", "sampler.rng_state", "expected 624 uint32 words and index in [0, 624]")
        restored = random.Random()
        try:
            restored.setstate((3, tuple(rng[1]), None))
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataError("InvalidRngState", "sampler.rng_state", "Random rejected its state") from exc
        self.epoch = epoch
        self.cursor = cursor
        self.order = list(order)
        self._rng = restored
