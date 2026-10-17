"""Independent strict decoding of actual Rust utility execution /2 DTO bytes.

No /1 fixture is converted or imported. Metadata decoding cannot establish a
historical corpus family, Rules replay, known target, frozen tensors or live
training authority. The caller reuses the existing bounded metadata runner.
"""
from __future__ import annotations

import importlib.abc
import json
from pathlib import Path
import sys

MAX_INPUT_BYTES = 65536
FIXTURE_DOMAIN = "rz-rovent-utility-metadata-v2-cross-language-fixture/1"
RESULT_DOMAIN = "rz-rovent-utility-metadata-v2-cross-language-result/1"
ANCESTOR_WIRE_DOMAIN = "rz-rovent-utility-ancestor-metadata-wire/1"
NEURAL_ROOTS = frozenset(("torch", "numpy", "onnx", "onnxruntime", "ort", "tensorflow", "jax", "cupy"))


class _NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in NEURAL_ROOTS:
            raise ImportError("neural imports are forbidden in the utility /2 metadata decoder")
        return None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate utility /2 metadata object field")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("nonfinite utility /2 metadata number")


def main():
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise RuntimeError("utility /2 decoder requires -I -S -B isolation")
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("utility /2 decoder began with a neural runtime")
    sys.meta_path.insert(0, _NoNeuralImports())
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise ValueError("finite nonempty utility /2 metadata fixture required")
    fixture = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    from rz_pals_model.rovent_training_bootstrap import _digest, _fields, _sha
    from rz_pals_model.rovent_utility_training_bootstrap import (REGISTRATION_V2_DOMAIN,
        validate_utility_registration_v2, validate_registration)

    _fields(fixture, ("schema", "registration", "expected_registration_sha256", "expected_ancestor_kinds",
        "expected_ancestor_metadata_sha256"), "actual Rust utility /2 fixture")
    if fixture["schema"] != FIXTURE_DOMAIN:
        raise ValueError("explicit Rust utility /2 fixture domain required")
    registration = validate_utility_registration_v2(fixture["registration"])
    # Test the original decoder directly. This does not rewrite the /2 request
    # as a /1 request or erase its mixed ancestry to make a common validator fit.
    try:
        validate_registration(registration)
    except ValueError:
        pass
    else:
        raise ValueError("original utility /1 decoder silently accepted a /2 declaration")
    actual_sha = _digest(REGISTRATION_V2_DOMAIN, registration)
    ancestors = registration["ancestor_datasets"]
    kinds = [ancestor["kind"] for ancestor in ancestors]
    ancestor_shas = [_digest(ANCESTOR_WIRE_DOMAIN, ancestor) for ancestor in ancestors]
    if actual_sha != _sha(fixture["expected_registration_sha256"]):
        raise ValueError("Rust and independent Python utility /2 canonical registration digest differ")
    if kinds != fixture["expected_ancestor_kinds"] or ancestor_shas != fixture["expected_ancestor_metadata_sha256"]:
        raise ValueError("utility /2 changed original ordered ancestry tags or complete payload bytes")
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("utility /2 decoder imported a neural runtime")
    result = {"schema": RESULT_DOMAIN, "status": "accepted_metadata_only", "registration_sha256": actual_sha,
        "ancestor_kinds": kinds, "ancestor_metadata_sha256": ancestor_shas, "original_v1_decoder_rejected": True,
        "historical_family_coverage_verified": False, "performed_optimizer_updates": 0,
        "past_native_authority_reissued": False, "execution_authorized": False, "neural_imports": False}
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.stderr.write((type(error).__name__ + ": " + str(error))[:4096] + "\n")
        raise SystemExit(1)
