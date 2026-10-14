"""Pure declarations and in-memory fake dispatch only; no NN/native acceptance."""
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
import types
import unittest
from unittest.mock import patch


def registration(module, root, phase="export"):
    def pin(path, symbol):
        return {"path": str(path), "sha256": symbol * 64, "bytes": 32}
    from rz_pals_model.initialization_evidence import SOURCE_SCHEMA, SOURCE_FILES
    stage, initial = root / "deployment", root / "collection-initial"
    export = stage / "export"
    inputs = {"checkpoint": pin(initial / "untrained.pt", "3"),
              "checkpoint_metadata": pin(initial / "checkpoint.json", "4"),
              "initialization_evidence": pin(initial / module.EVIDENCE_NAME, "5")}
    if phase == "export":
        inputs["rules_profile"] = pin(root / "input" / "rules.json", "1")
    else:
        inputs.update(case_plan=pin(root / "input" / "actual-plan.json", "2"),
            export_manifest=pin(export / "export.json", "6"),
            export_graphs=[dict(pin(export / filename, "7"), role=role, file=filename)
                           for role, filename in module.GRAPH_FILES.items()],
            cuda_control_inventory=pin(export / "cuda-control-inventory.json", "8"))
    return {"schema": module.REGISTRATION_DOMAIN, "numerical_id": "g0-deployment-1", "phase": phase,
        "source_components": dict.fromkeys(module.SOURCE_COMPONENTS, "a" * 64), "source_commit": "b" * 40,
        "initialization_sources": {"schema": SOURCE_SCHEMA, "source_commit": "b" * 40,
                                   "pins": dict.fromkeys(SOURCE_FILES, "c" * 64)},
        "staging_root": str(stage), "initialization_directory": str(initial), "export_directory": str(export),
        "reference_output_root": str(stage / "reference"), "receipt_path": str(stage / (phase + ".json")),
        "max_wall_ns": 60 * 10 ** 9, "max_output_bytes": 3 * 1024 ** 3, "inputs": inputs}


def startup(module, value, root):
    return {"schema": module.STARTUP_DOMAIN, "execution_id": "g0.deployment.1", "phase": value["phase"],
        "registration_sha256": module.registration_digest(value), "registration": value,
        "original_started_ns": 100 * 10 ** 9, "execution_until_ns": 150 * 10 ** 9, "whole_until_ns": 180 * 10 ** 9,
        "dependency_directories": [{"path": str(root / "runtime"), "device": 3, "inode": 11}],
        "python_runtime": {"version": "3.12.13", "executable": {
            "path": str(root / "runtime" / "python"), "sha256": "d" * 64, "bytes": 32}},
        "runtime_module_pins": [{"path": str(root / "runtime" / "torch-init.py"), "sha256": "e" * 64, "bytes": 32}]}


def validate_startup(module, value, **overrides):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    options = dict(expected_raw_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=raw,
                   execution_until_ns=150 * 10 ** 9, whole_until_ns=180 * 10 ** 9, now_ns=110 * 10 ** 9)
    options.update(overrides)
    return module.validate_startup(value, **options)


def pc_bindings():
    from rz_pals_model.virtual_review import registered_encoding_sha256
    return {"model_profile": "virtual_review_v1", "model_schema": "rovezero.pals-model.v3",
        "model_semantics": "rovezero.pals-model-semantics.v3", "encoding_schema": "rovezero.pals-virtual-review.v1",
        "model_semantic_sha256": "01" * 32, "encoding_sha256": registered_encoding_sha256(), "model_epoch": 1,
        "weights_sha256": "03" * 32, "export_manifest_sha256": "04" * 32,
        "graphs": {role: "05" * 32 for role in ("public", "shared_pc")}, "runtime_library_sha256": "06" * 32,
        "cuda_libraries": [], "cuda_manifest_sha256": None, "backend": "ort_cpu", "precision": "fp32", "tf32": False,
        "source_components": {name: "09" * 32 for name in ("model_native", "model_encoding", "nn_runtime", "model_reference", "model_export")}}


class DeploymentPureTests(unittest.TestCase):
    def setUp(self):
        original_import = builtins.__import__
        def no_neural_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime", "scipy", "transformers"}:
                raise AssertionError("pure Deployment check imported framework: " + name)
            level = kwargs.get("level", args[3] if len(args) > 3 else 0)
            if name == "rz_pals_model.model" or (level and name == "model"):
                raise AssertionError("pure Deployment check imported model")
            return original_import(name, *args, **kwargs)
        guard = patch("builtins.__import__", no_neural_import); guard.start(); self.addCleanup(guard.stop)
        temporary = tempfile.TemporaryDirectory(prefix="rz-deployment-pure-")
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name).resolve()
        self.cli = importlib.import_module("rz_pals_model.g0_deployment_numerical_cli")

    def test_default_and_child_flag_are_metadata_only(self):
        for args in ([], ["--original-g0-child"]):
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(self.cli.main(args), 0)
            observed = json.loads(stream.getvalue())
            self.assertEqual(observed["status"], "metadata_only")
            for name in ("actual_initialization_executed", "actual_export_executed", "actual_reference_executed",
                         "native_numeric_authority_issued", "native_game_authority_issued"):
                self.assertIs(observed[name], False)
            self.assertEqual(observed["optimizer_updates_executed"], 0)

    def test_execute_without_original_owner_transport_is_refused(self):
        with self.assertRaises(ValueError):
            self.cli.main(["--execute", "--startup-bytes", "1"], stdin=io.BytesIO(b"{}"))

    def test_both_declarations_are_copies_not_capabilities(self):
        for phase in self.cli.PHASES:
            row = registration(self.cli, self.root, phase)
            value = self.cli.validate_registration(row)
            self.assertEqual(value, row); self.assertIsNot(value, row)
            value["source_components"]["model_export"] = "f" * 64
            self.assertEqual(row["source_components"]["model_export"], "a" * 64)

    def test_original_initialization_can_share_managed_root(self):
        row = registration(self.cli, self.root)
        initial = Path(row["staging_root"]) / "original-initial"
        row["initialization_directory"] = str(initial)
        for name, file in (("checkpoint", "untrained.pt"), ("checkpoint_metadata", "checkpoint.json"),
                           ("initialization_evidence", self.cli.EVIDENCE_NAME)):
            row["inputs"][name]["path"] = str(initial / file)
        self.cli.validate_registration(row)

    def test_collection_domain_prepare_phase_or_serialized_owner_is_not_deployment(self):
        for change in (lambda r: r.update(schema="rz-pals-g0-numerical-registration/1"),
                       lambda r: r.update(phase="prepare"), lambda r: r.update(original_owner=True),
                       lambda r: r.update(native_numeric_authority_issued=True)):
            row = registration(self.cli, self.root); change(row)
            with self.assertRaises(ValueError):
                self.cli.validate_registration(row)

    def test_initialization_and_generated_namespace_overlap_is_refused(self):
        row = registration(self.cli, self.root)
        for key in ("export_directory", "reference_output_root"):
            bad = copy.deepcopy(row); bad[key] = row["initialization_directory"]
            with self.assertRaises(ValueError):
                self.cli.validate_registration(bad)

    def test_original_checkpoint_path_cannot_be_replaced_by_new_export(self):
        row = registration(self.cli, self.root)
        row["inputs"]["checkpoint"]["path"] = str(Path(row["export_directory"]) / "untrained.pt")
        with self.assertRaises(ValueError):
            self.cli.validate_registration(row)

    def test_reference_requires_exact_new_two_graph_roster(self):
        original = registration(self.cli, self.root, "reference")
        for change in (lambda r: r["inputs"]["export_graphs"].reverse(),
                       lambda r: r["inputs"]["export_graphs"].append(dict(r["inputs"]["export_graphs"][0], role="validator", file="role_validator.onnx")),
                       lambda r: r["inputs"]["export_graphs"][0].update(file="role_validator.onnx"),
                       lambda r: r["inputs"]["export_manifest"].update(path=str(self.root / "collection-export.json"))):
            row = copy.deepcopy(original); change(row)
            with self.assertRaises(ValueError):
                self.cli.validate_registration(row)

    def test_startup_joins_original_phase_bytes_and_deadlines(self):
        row = startup(self.cli, registration(self.cli, self.root), self.root)
        self.assertEqual(validate_startup(self.cli, row), row["registration"])
        with self.assertRaises(ValueError):
            validate_startup(self.cli, row, expected_raw_sha256="0" * 64)
        with self.assertRaises(ValueError):
            validate_startup(self.cli, row, now_ns=150 * 10 ** 9)
        with self.assertRaises(ValueError):
            validate_startup(self.cli, row, whole_until_ns=181 * 10 ** 9)

    def test_startup_cannot_be_collection_or_duplicate_runtime_directory(self):
        original = startup(self.cli, registration(self.cli, self.root), self.root)
        for change in (lambda r: r.update(schema="rz-pals-original-g0-numerical-startup/1"),
                       lambda r: r["dependency_directories"].append(dict(r["dependency_directories"][0])),
                       lambda r: r.update(phase="reference"), lambda r: r["runtime_module_pins"].append(dict(r["runtime_module_pins"][0]))):
            row = copy.deepcopy(original); change(row)
            with self.assertRaises(ValueError):
                validate_startup(self.cli, row)

    def test_checkpoint_header_does_not_accept_trained_or_diagnostic_or_optimizer(self):
        metadata = {"config": {"profile": "virtual_review_v1"}, "schema": "rovezero.pals-model-semantics.v3"}
        original = {"config": metadata["config"], "schema": metadata["schema"], "seed": 23,
                    "training_steps": 0, "trained": False, "state_dict": {}}
        self.cli.validate_original_checkpoint_header(original, metadata)
        for change in (lambda r: r.update(trained=True), lambda r: r.update(training_steps=1),
                       lambda r: r.update(training_steps=False), lambda r: r.update(seed=31),
                       lambda r: r.update(schema="diagnostic"), lambda r: r.update(optimizer={})):
            row = copy.deepcopy(original); change(row)
            with self.assertRaises(ValueError):
                self.cli.validate_original_checkpoint_header(row, metadata)

    def test_role_declaration_cannot_relabel_collection_or_keep_v_bank(self):
        original = {"roles": ["proposer", "critic"], "validator_present": False,
                    "graphs": [{"role": role} for role in self.cli.GRAPH_FILES],
                    "initializer_banks": {role: {} for role in self.cli.GRAPH_FILES}}
        self.cli.validate_vfree_manifest_roles(original)
        for change in (lambda r: r.update(validator_present=True), lambda r: r["roles"].append("validator"),
                       lambda r: r["graphs"].append({"role": "validator"}),
                       lambda r: r["initializer_banks"].update(validator={})):
            row = copy.deepcopy(original); change(row)
            with self.assertRaises(ValueError):
                self.cli.validate_vfree_manifest_roles(row)

    def test_real_pure_pc_factory_has_12_groups_100_cases_and_no_v(self):
        from rz_pals_model.numeric_capture import make_case_plan
        bindings = pc_bindings(); plan = make_case_plan(bindings)
        inputs = {"checkpoint": {"sha256": bindings["weights_sha256"]},
                  "export_manifest": {"sha256": bindings["export_manifest_sha256"]},
                  "export_graphs": [{"role": role, "sha256": value} for role, value in bindings["graphs"].items()]}
        manifest = {"graphs": inputs["export_graphs"]}
        self.assertEqual(self.cli.validate_reference_plan(plan, inputs, manifest, bindings["source_components"]), 100)
        self.assertEqual(len(plan["groups"]), 12)
        self.assertEqual({group["role"] for group in plan["groups"]}, {"proposer", "critic"})
        collection_bindings = copy.deepcopy(bindings)
        collection_bindings["graphs"]["validator"] = "07" * 32
        collection = make_case_plan(collection_bindings)
        self.assertEqual([group for group in collection["groups"] if group["role"] != "validator"], plan["groups"])
        incomplete = copy.deepcopy(plan); incomplete["groups"].pop()
        with self.assertRaises(ValueError):
            self.cli.validate_reference_plan(incomplete, inputs, manifest, bindings["source_components"])
        mismatch = copy.deepcopy(inputs); mismatch["checkpoint"]["sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            self.cli.validate_reference_plan(plan, mismatch, manifest, bindings["source_components"])

    def test_export_dispatch_is_separate_vfree_existing_api_and_never_initializer(self):
        from rz_pals_model import artifacts, virtual_control_inventory
        row = registration(self.cli, self.root)
        outputs = {"export_manifest": {"path": str(Path(row["export_directory"]) / "export.json"), "sha256": "6" * 64, "bytes": 32},
                   "export_graphs": [], "cuda_control_inventory": {"path": str(Path(row["export_directory"]) / "cuda-control-inventory.json"), "sha256": "8" * 64, "bytes": 32}}
        manifest = {"roles": ["proposer", "critic"], "validator_present": False,
                    "graphs": [{"role": role} for role in self.cli.GRAPH_FILES],
                    "initializer_banks": {role: {} for role in self.cli.GRAPH_FILES}}
        control = {"manifest_sha256": "6" * 64, "scope": virtual_control_inventory.SCOPE}
        observed = {"actual_export_executed": False, "export_dispatch_started": False}
        calls = []
        def fake_export(model, metadata, output, *, include_validator, rules_profile_json, output_guard):
            self.assertIs(include_validator, False); calls.append("export"); output_guard(32); return manifest
        def fake_inventory(*args, **kwargs):
            calls.append("inventory"); kwargs["output_guard"](32)
            return {"inventory_sha256": "8" * 64, "inventory_bytes": 32,
                    "actual_provider_placement": "not_observed", "cpu_allowlist": "not_created"}
        def registered(module, name, file):
            self.assertNotIn("initialize", name)
            return fake_export if name == "export_virtual_checkpoint" else fake_inventory
        with contextlib.ExitStack() as stack:
            for name, value in (("_registered_function", registered), ("_require_empty_output_directory", lambda path: Path(path)),
                    ("_original_metadata", lambda *a, **k: {}), ("_load_original_g0_model", lambda *a, **k: object()),
                    ("_model_bits_digest", lambda *a, **k: "a" * 64), ("_json_pin", lambda pin, *a, **k: control if pin["sha256"] == "8" * 64 else {}),
                    ("_pin", lambda *a, **k: outputs["export_manifest"]), ("_export_outputs", lambda *a, **k: outputs),
                    ("_output_credit", lambda *a, **k: 1 << 30), ("_validate_vfree_manifest", lambda manifest, metadata: self.cli.validate_vfree_manifest_roles(manifest))):
                stack.enter_context(patch.object(self.cli, name, value))
            result = self.cli._export(row, until_ns=60 * 10 ** 9, clock=lambda: 0,
                                      check_original=lambda: None, execution_observation=observed)
        self.assertEqual(calls, ["export", "inventory"])
        self.assertEqual(result["status"], "exported")
        self.assertIs(result["original_model_bits_preserved"], True)
        self.assertEqual(set(result["outputs"]), {"export_manifest", "export_graphs", "cuda_control_inventory"})
        self.assertIs(observed["actual_export_executed"], True)
        self.assertNotIn("native_numeric_authority_issued", result)

    def test_reference_uses_borrowed_pc_view_and_preserves_unknown_failed_work(self):
        from rz_pals_model import numeric_capture
        row = registration(self.cli, self.root, "reference")
        for fail in (False, True):
            view = types.SimpleNamespace(experts={"proposer": object(), "critic": object()})
            class Owner:
                def virtual_review_pc_export_view(self):
                    return view
            observation = {"reference_dispatch_started": False, "reference_neural_forwards_executed": 0,
                           "actual_reference_executed": False}
            def capture(model, metadata, directory, plan, output, **kwargs):
                self.assertIs(model, view)
                self.assertEqual(kwargs["checkpoint_path"], row["inputs"]["checkpoint"]["path"])
                self.assertIs(kwargs["execute"], True)
                if fail:
                    raise ValueError("mock capture failed after dispatch")
                return {"cases": 100, "groups": 12}
            with contextlib.ExitStack() as stack:
                for name, value in (("_registered_function", lambda *a: capture),
                        ("_require_empty_output_directory", lambda path: Path(path)),
                        ("_validate_reference", lambda *a, **k: ({}, {})),
                        ("_load_original_g0_model", lambda *a, **k: Owner()),
                        ("_model_bits_digest", lambda *a, **k: "a" * 64),
                        ("_output_credit", lambda *a, **k: 1 << 30)):
                    stack.enter_context(patch.object(self.cli, name, value))
                if fail:
                    with self.assertRaises(ValueError):
                        self.cli._reference(row, until_ns=60 * 10 ** 9, clock=lambda: 0,
                                            check_original=lambda: None, execution_observation=observation)
                    self.assertIsNone(observation["reference_neural_forwards_executed"])
                    self.assertIs(observation["actual_reference_executed"], False)
                else:
                    result = self.cli._reference(row, until_ns=60 * 10 ** 9, clock=lambda: 0,
                                                  check_original=lambda: None, execution_observation=observation)
                    self.assertEqual(result["status"], "captured")
                    self.assertEqual(observation["reference_neural_forwards_executed"], 100)
                    self.assertIs(result["original_model_bits_preserved"], True)
            self.assertIs(observation["reference_dispatch_started"], True)

    def test_failed_child_preserves_typed_failure_and_never_issues_authority(self):
        row = startup(self.cli, registration(self.cli, self.root), self.root)
        raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        terminal = []
        def fail_dependency(*a, **k):
            raise ValueError("mock original runtime binding refused")
        def record(registration, result):
            terminal.append(copy.deepcopy(result))
            return {"path": registration["receipt_path"], "sha256": "a" * 64, "bytes": 32}
        args = ["--execute", "--original-g0-child", "--startup-sha256", hashlib.sha256(raw).hexdigest(),
                "--startup-bytes", str(len(raw)), "--execution-until-ns", str(150 * 10 ** 9),
                "--whole-until-ns", str(180 * 10 ** 9)]
        with contextlib.ExitStack() as stack:
            for name, value in (("os", types.SimpleNamespace(name="posix")),
                    ("_output_credit", lambda *a, **k: 1 << 30), ("_install_original_dependencies", fail_dependency),
                    ("_artifact_readbacks", lambda *a, **k: []), ("_original_inputs", lambda *a, **k: {}),
                    ("_source_readback", lambda *a, **k: {}), ("_runtime_readbacks", lambda *a, **k: []),
                    ("_staging_bytes", lambda *a, **k: 0), ("_terminal_receipt", record)):
                stack.enter_context(patch.object(self.cli, name, value))
            stack.enter_context(patch.object(self.cli.signal, "signal", return_value=None))
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                code = self.cli.main(args, clock=lambda: 110 * 10 ** 9, stdin=io.BytesIO(raw))
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stream.getvalue())["status"], "failed")
        self.assertEqual(terminal[0]["error_type"], "ValueError")
        self.assertEqual(terminal[0]["optimizer_updates_executed"], 0)
        self.assertEqual(terminal[0]["reference_neural_forwards_executed"], 0)
        self.assertIs(terminal[0]["partial_output_retained_for_parent"], True)
        for name in ("actual_initialization_executed", "actual_export_executed", "actual_reference_executed",
                     "native_numeric_authority_issued", "native_game_authority_issued", "original_process_authority_issued",
                     "arena_accepted"):
            self.assertIs(terminal[0][name], False)


if __name__ == "__main__":
    unittest.main()
