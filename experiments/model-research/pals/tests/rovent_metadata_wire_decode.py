"""Decode actual Rust-serialized Rovent metadata with neural imports blocked.

This isolated, finite test entry does not open a corpus/checkpoint, create a
model, issue process authority, or dispatch a learner. Its caller is the Rust
metadata wire test, rather than a second manually copied Python fixture.
"""
from __future__ import annotations

import importlib.abc
import json
from pathlib import Path
import sys

MAX_INPUT_BYTES = 16 << 20
FIXTURE_DOMAIN = "rz-rovent-metadata-cross-language-fixture/1"
RESULT_DOMAIN = "rz-rovent-metadata-cross-language-result/1"
NEURAL_ROOTS = frozenset(("torch", "numpy", "onnx", "onnxruntime", "tensorflow", "jax"))


class _NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in NEURAL_ROOTS:
            raise ImportError("neural imports are forbidden in the metadata wire test")
        return None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate metadata object field")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("nonfinite metadata number")


def main():
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise RuntimeError("metadata decoder requires -I -S -B isolation")
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("metadata test began with an imported neural runtime")
    sys.meta_path.insert(0, _NoNeuralImports())
    # The registered source tree is the only extra import location. The test
    # starts with -I -S -B, so site hooks and dependency .pth files do not run.
    source = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(source))
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise ValueError("finite nonempty metadata fixture input required")
    fixture = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique_object,
                         parse_constant=_nonfinite)
    from rz_pals_model.generation_data import fields, sha
    from rz_pals_model.rovent_generation import SeriesRecipe
    from rz_pals_model.rovent_training_bootstrap import validate_recovered_registration
    from rz_pals_model.rovent_training_window_v2 import validate_window_spec, training_profile_digest
    from rz_pals_model.rovent_recovered_training_window import validate_checkpoint_publication_metadata

    fields(fixture, ("schema", "registration", "window", "checkpoint_publication", "recipe",
                     "expected_profile_sha256"), "actual Rust metadata fixture")
    if fixture["schema"] != FIXTURE_DOMAIN:
        raise ValueError("explicit Rust metadata fixture domain required")
    registration = validate_recovered_registration(fixture["registration"])
    recipe = SeriesRecipe(fixture["recipe"])
    window = validate_window_spec(recipe, fixture["window"])
    publication = validate_checkpoint_publication_metadata(
        fixture["checkpoint_publication"], original_window=window)
    if (registration["trainer_source_sha256"] != window["trainer_source_sha256"]
            or registration["trainer_python_executable"]["sha256"] != window["trainer_python_executable_sha256"]
            or registration["trainer_dependency_set_sha256"] != window["trainer_dependency_set_sha256"]
            or publication["recipe_sha256"] != recipe.sha256):
        raise ValueError("Rust metadata collection/trainer/profile identities changed")
    actual = training_profile_digest(window)
    if actual != sha(fixture["expected_profile_sha256"]):
        raise ValueError("Rust and Python fixed-profile canonical digest differ")
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("metadata decoder imported a neural runtime")
    result = {"schema": RESULT_DOMAIN, "status": "accepted", "profile_sha256": actual,
              "past_native_authority_reissued": False, "execution_authorized": False,
              "neural_imports": False}
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.stderr.write((type(error).__name__ + ": " + str(error))[:4096] + "\n")
        raise SystemExit(1)
