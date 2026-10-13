"""Standard-library metadata checks; never execute a model or optimizer."""
import builtins
import contextlib
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


def registration(module, tmp_path, *, kind="frozen_pc_collection"):
    # This creates declarations only: these paths are not opened, populated or
    # converted into a live export authority by any test.
    def pin(path, symbol):
        return {"path": str(path), "sha256": symbol * 64, "bytes": 32}

    from rz_pals_model.config import ModelConfig
    config = ModelConfig.for_profile("virtual_review_v1").to_dict()
    collection = kind == "frozen_pc_collection"
    roles = ["public", "shared_pc"] + (["validator"] if collection else [])
    names = {"public": "public_memory.onnx", "shared_pc": "shared_pc_if.onnx", "validator": "role_validator.onnx"}
    graphs = [dict(pin(tmp_path / "export" / names[role], "4"), role=role, file=names[role]) for role in roles]
    return {
        "schema": module.REGISTRATION_DOMAIN, "numerical_id": "reference-1",
        "template_startup": pin(tmp_path / "input" / "template.json", "1"),
        "case_plan": pin(tmp_path / "input" / "case-plan.json", "2"),
        "checkpoint": pin(tmp_path / "input" / "actual.pt", "3"),
        "export_manifest": pin(tmp_path / "export" / "export.json", "5"),
        "export_graphs": graphs,
        "source_components": dict.fromkeys(module.SOURCE_COMPONENTS, "6" * 64),
        "frozen_pc_sha256": "7" * 64, "model_epoch": 2,
        "generation_export_semantic": {
            "kind": kind, "config": config, "checkpoint_sha256": "3" * 64,
            "trained": True, "training_steps": 4000 if collection else 5000,
            "validator_present": collection, "model_epoch": 2, "frozen_pc_sha256": "7" * 64,
        },
        "staging_root": str(tmp_path), "reference_output_root": str(tmp_path / "reference"),
        "receipt_path": str(tmp_path / "terminal.json"),
        "max_wall_ns": 60 * 10 ** 9, "max_output_bytes": 512 * 1024 ** 2,
    }


def startup(module, value, tmp_path):
    return {
        "schema": module.STARTUP_DOMAIN, "execution_id": "reference.execution.1",
        "registration_sha256": module.numerical_registration_digest(value), "registration": value,
        "original_started_ns": 100 * 10 ** 9,
        "execution_until_ns": 150 * 10 ** 9, "whole_until_ns": 180 * 10 ** 9,
        "dependency_directories": [{"path": str(tmp_path / "runtime"), "device": 3, "inode": 11}],
        "python_runtime": {"version": "3.12.3", "executable": {
            "path": str(tmp_path / "runtime" / "python"), "sha256": "8" * 64, "bytes": 32,
        }},
        "runtime_module_pins": [{"path": str(tmp_path / "runtime" / "torch-init.py"),
                                 "sha256": "9" * 64, "bytes": 32}],
    }


def validate_startup(module, value, **overrides):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    args = {
        "expected_raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": raw,
        "execution_until_ns": 150 * 10 ** 9, "whole_until_ns": 180 * 10 ** 9,
        "now_ns": 110 * 10 ** 9,
    }
    args.update(overrides)
    return module.validate_startup(value, **args)


class GenerationNumericalMetadataTests(unittest.TestCase):
    def setUp(self):
        original_import = builtins.__import__

        def metadata_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime", "scipy", "transformers"}:
                raise AssertionError("pure numerical metadata imported a neural framework: " + name)
            level = kwargs.get("level", args[3] if len(args) > 3 else 0)
            if name == "rz_pals_model.model" or (level and name == "model"):
                raise AssertionError("pure numerical metadata imported the model")
            return original_import(name, *args, **kwargs)

        guard = patch("builtins.__import__", metadata_import)
        guard.start()
        self.addCleanup(guard.stop)
        temporary = tempfile.TemporaryDirectory(prefix="rz-numerical-metadata-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        name = "rz_pals_model.generation_numerical_cli"
        missing = object()
        previous_module = sys.modules.pop(name, missing)
        package = sys.modules.get("rz_pals_model")
        previous_attribute = getattr(package, "generation_numerical_cli", missing) if package is not None else missing

        def restore_module():
            if previous_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module
            current_package = sys.modules.get("rz_pals_model")
            if current_package is not None:
                if previous_attribute is missing:
                    if hasattr(current_package, "generation_numerical_cli"):
                        delattr(current_package, "generation_numerical_cli")
                else:
                    setattr(current_package, "generation_numerical_cli", previous_attribute)

        self.addCleanup(restore_module)
        self.cli = importlib.import_module(name)

        def forbidden_execution(*_args, **_kwargs):
            raise AssertionError("pure numerical metadata entered an execution path")

        for helper in ("_install_original_dependencies", "_load_original_cpu_model", "_validate_plan_and_manifest", "_runtime_readbacks"):
            blocker = patch.object(self.cli, helper, forbidden_execution)
            blocker.start()
            self.addCleanup(blocker.stop)
        for owner_name in ("rz_pals_model.generation_training", "rz_pals_model.generation_training_cli"):
            owner = sys.modules.get(owner_name)
            if owner is not None:
                blocker = patch.object(owner, "GenerationOptimizers", forbidden_execution)
                blocker.start()
                self.addCleanup(blocker.stop)

    def test_default_main_is_metadata_only_with_execution_and_import_sentinels(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.cli.main([]), 0)
        self.assertEqual(json.loads(output.getvalue()), {
            "schema": self.cli.RESULT_DOMAIN, "status": "metadata_only",
            "actual_reference_executed": False, "optimizer_updates_executed": 0,
            "native_numeric_authority_issued": False,
        })


    def test_frozen_and_g1_metadata_are_valid_declarations_not_live_authority(self):
        numerical = self.cli
        tmp_path = self.root
        for kind in ("frozen_pc_collection", "g1_actual_generation"):
            value = registration(numerical, tmp_path, kind=kind)
            observed = numerical.validate_registration(value)
            self.assertEqual(observed, value)
            self.assertIsNot(observed, value)
            self.assertIs(type(observed), dict)
            self.assertNotIn("native_numeric_authority_issued", observed)
            self.assertNotIn("passed", observed)
            observed["generation_export_semantic"]["config"]["profile"] = "changed-declaration"
            self.assertEqual(value["generation_export_semantic"]["config"]["profile"], "virtual_review_v1")


    def test_bad_artifact_pin_and_json_authority_claim_are_rejected(self):
        numerical = self.cli
        tmp_path = self.root
        value = registration(numerical, tmp_path)
        bad_digest = copy.deepcopy(value)
        bad_digest["checkpoint"]["sha256"] = "not-a-sha256"
        extra_pin_authority = copy.deepcopy(value)
        extra_pin_authority["case_plan"]["native_authority"] = True
        root_authority = copy.deepcopy(value)
        root_authority["native_numeric_authority_issued"] = True
        for invalid in (bad_digest, extra_pin_authority, root_authority):
            with self.assertRaises(ValueError):
                numerical.validate_registration(invalid)


    def test_unsupported_kind_and_wrong_update_or_validator_boundary_are_rejected(self):
        numerical = self.cli
        tmp_path = self.root
        frozen = registration(numerical, tmp_path)
        g1 = registration(numerical, tmp_path, kind="g1_actual_generation")
        unknown_kind = copy.deepcopy(frozen)
        unknown_kind["generation_export_semantic"]["kind"] = "diagnostic_smoke"
        wrong_frozen = copy.deepcopy(frozen)
        wrong_frozen["generation_export_semantic"]["training_steps"] = 5000
        wrong_g1 = copy.deepcopy(g1)
        wrong_g1["generation_export_semantic"]["training_steps"] = 4000
        g1_with_validator = copy.deepcopy(g1)
        g1_with_validator["generation_export_semantic"]["validator_present"] = True
        for invalid in (unknown_kind, wrong_frozen, wrong_g1, g1_with_validator):
            with self.assertRaises(ValueError):
                numerical.validate_registration(invalid)


    def test_duplicate_graph_role_and_changed_source_roster_are_rejected(self):
        numerical = self.cli
        tmp_path = self.root
        value = registration(numerical, tmp_path)
        duplicated = copy.deepcopy(value)
        duplicated["export_graphs"][1] = copy.deepcopy(duplicated["export_graphs"][0])
        missing_source = copy.deepcopy(value)
        del missing_source["source_components"]["model_reference"]
        extra_source = copy.deepcopy(value)
        extra_source["source_components"]["metadata_authority"] = "a" * 64
        for invalid in (duplicated, missing_source, extra_source):
            with self.assertRaises(ValueError):
                numerical.validate_registration(invalid)


    def test_original_startup_requires_unique_runtime_pins_and_dependencies(self):
        numerical = self.cli
        tmp_path = self.root
        value = startup(numerical, registration(numerical, tmp_path), tmp_path)
        self.assertEqual(validate_startup(numerical, value), value["registration"])
        duplicate_runtime = copy.deepcopy(value)
        duplicate_runtime["runtime_module_pins"].append(copy.deepcopy(duplicate_runtime["runtime_module_pins"][0]))
        duplicate_dependency = copy.deepcopy(value)
        duplicate_dependency["dependency_directories"].append(copy.deepcopy(duplicate_dependency["dependency_directories"][0]))
        for invalid in (duplicate_runtime, duplicate_dependency):
            with self.assertRaises(ValueError):
                validate_startup(numerical, invalid)


    def test_original_until_cannot_be_omitted_renewed_or_extended_beyond_window(self):
        numerical = self.cli
        tmp_path = self.root
        value = startup(numerical, registration(numerical, tmp_path), tmp_path)
        missing_until = copy.deepcopy(value)
        del missing_until["execution_until_ns"]
        renewed = copy.deepcopy(value)
        renewed["execution_until_ns"] += 1
        beyond_window = copy.deepcopy(value)
        beyond_window["registration"]["max_wall_ns"] = 40 * 10 ** 9
        beyond_window["registration_sha256"] = numerical.numerical_registration_digest(beyond_window["registration"])
        for invalid in (missing_until, renewed, beyond_window):
            with self.assertRaises(ValueError):
                validate_startup(numerical, invalid)
        with self.assertRaises(ValueError):
            validate_startup(numerical, value, now_ns=value["execution_until_ns"])


    def test_raw_pin_and_unregistered_startup_authority_are_rejected(self):
        numerical = self.cli
        tmp_path = self.root
        value = startup(numerical, registration(numerical, tmp_path), tmp_path)
        with self.assertRaises(ValueError):
            validate_startup(numerical, value, expected_raw_sha256="0" * 64)
        unsupported_authority = copy.deepcopy(value)
        unsupported_authority["actual_export_authority"] = {"reconstruct_from_json": True}
        with self.assertRaises(ValueError):
            validate_startup(numerical, unsupported_authority)


if __name__ == "__main__":
    unittest.main()
