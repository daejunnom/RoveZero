"""Original G0 metadata/authority checks; frameworks and execution are blocked."""
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


def registration(module, root, phase="prepare"):
    def pin(path, symbol):
        return {"path": str(path), "sha256": symbol * 64, "bytes": 32}
    from rz_pals_model.initialization_evidence import SOURCE_SCHEMA, SOURCE_FILES
    initial, export = root / "initial", root / "export"
    inputs = {"rules_profile": pin(root / "input" / "rules.json", "1"),
              "case_plan_template": pin(root / "input" / "template.json", "2")}
    if phase == "reference":
        inputs = {"case_plan": pin(root / "input" / "actual-plan.json", "2"),
            "checkpoint": pin(initial / "untrained.pt", "3"),
            "checkpoint_metadata": pin(initial / "checkpoint.json", "4"),
            "initialization_evidence": pin(initial / module.EVIDENCE_NAME, "5"),
            "export_manifest": pin(export / "export.json", "6"),
            "export_graphs": [dict(pin(export / file, "7"), role=role, file=file)
                              for role, file in module.GRAPH_FILES.items()],
            "cuda_control_inventory": pin(export / "cuda-control-inventory.json", "8")}
    return {"schema": module.REGISTRATION_DOMAIN, "numerical_id": "g0-reference-1", "phase": phase,
        "source_components": dict.fromkeys(module.SOURCE_COMPONENTS, "a" * 64),
        "source_commit": "b" * 40,
        "initialization_sources": {"schema": SOURCE_SCHEMA, "source_commit": "b" * 40,
                                   "pins": dict.fromkeys(SOURCE_FILES, "c" * 64)},
        "staging_root": str(root), "initialization_directory": str(initial),
        "export_directory": str(export), "reference_output_root": str(root / "reference"),
        "receipt_path": str(root / (phase + ".json")), "max_wall_ns": 60 * 10 ** 9,
        "max_output_bytes": 3 * 1024 ** 3, "inputs": inputs}


def startup(module, value, root):
    return {"schema": module.STARTUP_DOMAIN, "execution_id": "g0.reference.1", "phase": value["phase"],
        "registration_sha256": module.registration_digest(value), "registration": value,
        "original_started_ns": 100 * 10 ** 9, "execution_until_ns": 150 * 10 ** 9,
        "whole_until_ns": 180 * 10 ** 9,
        "dependency_directories": [{"path": str(root / "runtime"), "device": 3, "inode": 11}],
        "python_runtime": {"version": "3.12.3", "executable": {
            "path": str(root / "runtime" / "python"), "sha256": "d" * 64, "bytes": 32}},
        "runtime_module_pins": [{"path": str(root / "runtime" / "torch-init.py"),
                                 "sha256": "e" * 64, "bytes": 32}]}


def validate_startup(module, value, **overrides):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    args = {"expected_raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": raw,
        "execution_until_ns": 150 * 10 ** 9, "whole_until_ns": 180 * 10 ** 9,
        "now_ns": 110 * 10 ** 9}
    args.update(overrides)
    return module.validate_startup(value, **args)


class G0NumericalMetadataTests(unittest.TestCase):
    def setUp(self):
        original_import = builtins.__import__
        def metadata_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime", "scipy", "transformers"}:
                raise AssertionError("G0 metadata imported a neural framework: " + name)
            level = kwargs.get("level", args[3] if len(args) > 3 else 0)
            if name == "rz_pals_model.model" or (level and name == "model"):
                raise AssertionError("G0 metadata imported the model")
            return original_import(name, *args, **kwargs)
        guard = patch("builtins.__import__", metadata_import)
        guard.start()
        self.addCleanup(guard.stop)
        temporary = tempfile.TemporaryDirectory(prefix="rz-g0-numerical-metadata-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        name, missing = "rz_pals_model.g0_numerical_cli", object()
        previous = sys.modules.pop(name, missing)
        package = sys.modules.get("rz_pals_model")
        old_attribute = getattr(package, "g0_numerical_cli", missing) if package is not None else missing
        def restore_module():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
            current = sys.modules.get("rz_pals_model")
            if current is not None:
                if old_attribute is missing:
                    if hasattr(current, "g0_numerical_cli"):
                        delattr(current, "g0_numerical_cli")
                else:
                    setattr(current, "g0_numerical_cli", old_attribute)
        self.addCleanup(restore_module)
        self.cli = importlib.import_module(name)
        def execution_forbidden(*_args, **_kwargs):
            raise AssertionError("pure G0 metadata entered an actual execution path")
        for helper in ("_install_original_dependencies", "_prepare", "_reference", "_load_g0_reference",
                       "_original_inputs", "_artifact_readbacks", "_source_readback", "_runtime_readbacks"):
            blocker = patch.object(self.cli, helper, execution_forbidden)
            blocker.start()
            self.addCleanup(blocker.stop)
        for owner_name in ("rz_pals_model.generation_training", "rz_pals_model.generation_training_cli"):
            owner = sys.modules.get(owner_name)
            if owner is not None:
                blocker = patch.object(owner, "GenerationOptimizers", execution_forbidden)
                blocker.start()
                self.addCleanup(blocker.stop)

    def test_default_command_is_metadata_only_and_cannot_issue_any_native_authority(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.cli.main([]), 0)
        self.assertEqual(json.loads(output.getvalue()), {
            "schema": self.cli.RESULT_DOMAIN, "status": "metadata_only",
            "actual_initialization_executed": False, "actual_export_executed": False,
            "actual_reference_executed": False, "optimizer_updates_executed": 0,
            "native_numeric_authority_issued": False, "native_game_authority_issued": False})

    def test_child_flag_without_execute_does_not_dispatch(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.cli.main(["--original-g0-child"]), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "metadata_only")

    def test_execute_requires_explicit_original_child_before_any_input_or_import(self):
        with self.assertRaises(ValueError):
            self.cli.main(["--execute", "--startup-bytes", "1"], stdin=io.BytesIO(b"{}"))

    def test_both_phase_declarations_are_copies_not_live_producer_capabilities(self):
        for phase in self.cli.PHASES:
            value = registration(self.cli, self.root, phase)
            observed = self.cli.validate_registration(value)
            self.assertEqual(observed, value)
            self.assertIsNot(observed, value)
            self.assertIs(type(observed), dict)
            self.assertNotIn("native_numeric_authority_issued", observed)
            observed["source_components"]["model_reference"] = "f" * 64
            self.assertEqual(value["source_components"]["model_reference"], "a" * 64)

    def test_serialized_authority_and_trained_phase_cannot_be_added(self):
        value = registration(self.cli, self.root)
        claims = ("actual_initializer", "original_owner", "native_numeric_authority_issued", "trained")
        for claim in claims:
            invalid = copy.deepcopy(value)
            invalid[claim] = True
            with self.assertRaises(ValueError):
                self.cli.validate_registration(invalid)
        for phase in ("frozen_pc_collection", "g1_actual_generation", "diagnostic", "train"):
            invalid = copy.deepcopy(value)
            invalid["phase"] = phase
            with self.assertRaises(ValueError):
                self.cli.validate_registration(invalid)

    def test_initializer_source_domain_commit_and_fixed_roster_are_joined(self):
        value = registration(self.cli, self.root)
        invalids = []
        for key, replacement in (("schema", "rz-pals-legacy-initialization-sources/1"), ("source_commit", "f" * 40)):
            invalid = copy.deepcopy(value)
            invalid["initialization_sources"][key] = replacement
            invalids.append(invalid)
        missing = copy.deepcopy(value)
        del missing["initialization_sources"]["pins"]["initialization_evidence.py"]
        invalids.append(missing)
        malformed = copy.deepcopy(value)
        malformed["source_commit"] = "short-sha"
        invalids.append(malformed)
        for invalid in invalids:
            with self.assertRaises(ValueError):
                self.cli.validate_registration(invalid)

    def test_unknown_or_missing_numerical_source_and_boolean_limits_are_rejected(self):
        value = registration(self.cli, self.root)
        missing, extra = copy.deepcopy(value), copy.deepcopy(value)
        del missing["source_components"]["nn_runtime"]
        extra["source_components"]["json_authority"] = "f" * 64
        invalids = [missing, extra]
        for key, replacement in (("max_wall_ns", True), ("max_wall_ns", 0), ("max_output_bytes", True),
                                 ("max_output_bytes", self.cli.MAX_OUTPUT_BYTES + 1),
                                 ("max_output_bytes", self.cli.MAX_JSON_BYTES)):
            invalid = copy.deepcopy(value)
            invalid[key] = replacement
            invalids.append(invalid)
        for invalid in invalids:
            with self.assertRaises(ValueError):
                self.cli.validate_registration(invalid)

    def test_output_nesting_alias_and_input_overwrite_are_rejected(self):
        value = registration(self.cli, self.root)
        invalids = []
        for key, replacement in (("export_directory", value["initialization_directory"]),
                ("reference_output_root", str(Path(value["export_directory"]) / "nested")),
                ("receipt_path", str(Path(value["initialization_directory"]) / "terminal.json")),
                ("initialization_directory", str(self.root.parent / "foreign"))):
            invalid = copy.deepcopy(value)
            invalid[key] = replacement
            invalids.append(invalid)
        overwrite = copy.deepcopy(value)
        overwrite["inputs"]["case_plan_template"]["path"] = str(Path(value["export_directory"]) / "input.json")
        invalids.append(overwrite)
        alias = copy.deepcopy(value)
        alias["inputs"]["case_plan_template"] = copy.deepcopy(alias["inputs"]["rules_profile"])
        invalids.append(alias)
        for invalid in invalids:
            with self.assertRaises(ValueError):
                self.cli.validate_registration(invalid)

    def test_secrets_parent_traversal_and_fake_artifact_claims_are_rejected_without_reads(self):
        value = registration(self.cli, self.root)
        for path in (self.root / ".env", self.root / "client_secret.json", self.root / "ssh-key.key", self.root / "child" / ".." / "rules.json"):
            invalid = copy.deepcopy(value)
            invalid["inputs"]["rules_profile"]["path"] = str(path)
            with self.assertRaises(ValueError):
                self.cli.validate_registration(invalid)
        invalid = copy.deepcopy(value)
        invalid["inputs"]["rules_profile"]["native_owner"] = True
        with self.assertRaises(ValueError):
            self.cli.validate_registration(invalid)

    def test_reference_requires_original_fresh_names_and_complete_collection_graphs(self):
        value = registration(self.cli, self.root, "reference")
        invalids = []
        for name, replacement in (("checkpoint", self.root / "initial" / "frozen-4000.pt"),
                ("checkpoint_metadata", self.root / "foreign" / "checkpoint.json"),
                ("initialization_evidence", self.root / "initial" / "legacy-initialization-observation.json"),
                ("cuda_control_inventory", self.root / "foreign" / "cuda-control-inventory.json")):
            invalid = copy.deepcopy(value)
            invalid["inputs"][name]["path"] = str(replacement)
            invalids.append(invalid)
        missing_v = copy.deepcopy(value)
        missing_v["inputs"]["export_graphs"].pop()
        invalids.append(missing_v)
        duplicate = copy.deepcopy(value)
        duplicate["inputs"]["export_graphs"][1] = copy.deepcopy(duplicate["inputs"]["export_graphs"][0])
        invalids.append(duplicate)
        for invalid in invalids:
            with self.assertRaises(ValueError):
                self.cli.validate_registration(invalid)

    def test_valid_original_startup_is_metadata_not_a_reconstructed_g0_owner(self):
        for phase in self.cli.PHASES:
            value = startup(self.cli, registration(self.cli, self.root, phase), self.root)
            self.assertEqual(validate_startup(self.cli, value), value["registration"])

    def test_startup_rejects_wrong_phase_digest_duplicate_pins_and_serialized_authority(self):
        value = startup(self.cli, registration(self.cli, self.root), self.root)
        invalids = []
        wrong_phase = copy.deepcopy(value)
        wrong_phase["phase"] = "reference"
        invalids.append(wrong_phase)
        wrong_digest = copy.deepcopy(value)
        wrong_digest["registration_sha256"] = "0" * 64
        invalids.append(wrong_digest)
        extra = copy.deepcopy(value)
        extra["original_initializer_capture"] = {"from_json": True}
        invalids.append(extra)
        for key in ("dependency_directories", "runtime_module_pins"):
            invalid = copy.deepcopy(value)
            invalid[key].append(copy.deepcopy(invalid[key][0]))
            invalids.append(invalid)
        for invalid in invalids:
            with self.assertRaises(ValueError):
                validate_startup(self.cli, invalid)

    def test_original_until_cannot_be_renewed_extended_or_expired(self):
        value = startup(self.cli, registration(self.cli, self.root), self.root)
        missing = copy.deepcopy(value)
        del missing["execution_until_ns"]
        renewed = copy.deepcopy(value)
        renewed["execution_until_ns"] += 1
        over_window = copy.deepcopy(value)
        over_window["registration"]["max_wall_ns"] = 40 * 10 ** 9
        over_window["registration_sha256"] = self.cli.registration_digest(over_window["registration"])
        for invalid in (missing, renewed, over_window):
            with self.assertRaises(ValueError):
                validate_startup(self.cli, invalid)
        with self.assertRaises(ValueError):
            validate_startup(self.cli, value, now_ns=value["execution_until_ns"])
        with self.assertRaises(ValueError):
            validate_startup(self.cli, value, whole_until_ns=181 * 10 ** 9)

    def test_original_raw_hash_and_byte_credit_are_required(self):
        value = startup(self.cli, registration(self.cli, self.root), self.root)
        with self.assertRaises(ValueError):
            validate_startup(self.cli, value, expected_raw_sha256="0" * 64)
        with self.assertRaises(ValueError):
            validate_startup(self.cli, value, raw_bytes=b"{}")
        with self.assertRaises(ValueError):
            validate_startup(self.cli, value, raw_bytes=b"")

    def test_parent_retained_empty_output_is_accepted_without_creating_or_replacing_it(self):
        directory = self.root / "parent-held-output"
        directory.mkdir()
        before = directory.stat()
        self.assertEqual(self.cli._require_empty_output_directory(directory), directory)
        after = directory.stat()
        self.assertEqual((before.st_dev, before.st_ino), (after.st_dev, after.st_ino))
        self.assertEqual(list(directory.iterdir()), [])

    def test_missing_output_and_previous_partial_are_preserved_and_rejected(self):
        missing = self.root / "missing-output"
        with self.assertRaises(FileExistsError):
            self.cli._require_empty_output_directory(missing)
        self.assertFalse(missing.exists())
        directory = self.root / "old-output"
        directory.mkdir()
        artifact = directory / "failure.partial"
        artifact.write_bytes(b"retained failed output")
        with self.assertRaises(FileExistsError):
            self.cli._require_empty_output_directory(directory)
        self.assertEqual(artifact.read_bytes(), b"retained failed output")


if __name__ == "__main__":
    unittest.main()
