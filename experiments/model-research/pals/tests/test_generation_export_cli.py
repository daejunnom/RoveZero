"""Original export metadata checks; optimizer/NN/export execution is blocked."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from test_generation_contract import SourceOnlyTests
from test_generation_training_cli import registration as training_registration
from rz_pals_model import generation_export_cli as cli
from rz_pals_model.generation_training_cli import registration_digest


def registration(root, kind="frozen_pc_collection"):
    training = training_registration(root)
    if kind == "g1_actual_generation":
        training.update(phase="v", frozen_pc_sha256="e" * 64, model_epoch=2)
        training["datasets"]["v"] = copy.deepcopy(training["datasets"]["pc"])
        training["checkpoint"].update(domain=cli.CHECKPOINT_DOMAIN, mode="resume")
    pin = lambda path: {"path": str(root / path), "sha256": "a" * 64, "bytes": 1}
    return {"schema": cli.REGISTRATION_DOMAIN, "export_id": "export-original-1", "kind": kind,
        "training_registration": training, "original_training_registration_sha256": registration_digest(training),
        "trainer_receipt": pin("terminal.json"), "original_training_execution_id": "trainer-original-16",
        "checkpoint": pin("checkpoints/generation-4000.pt"), "rules_profile": pin("rules.json"),
        "export_source_components": {name: "c" * 64 for name in cli.EXPORT_COMPONENTS},
        "model_epoch": 2, "index_root": str(root / "export-index"), "output_directory": str(root / "export"),
        "receipt_path": str(root / "export-terminal.json"), "max_wall_ns": cli.MAX_WALL_NS,
        "max_output_bytes": training["max_output_bytes"]}


def terminal(value):
    counts = {"proposer": 2000, "critic": 2000, "validator": 0 if value["kind"] == "frozen_pc_collection" else 1000}
    status = "await_actual_v_collection" if value["kind"] == "frozen_pc_collection" else "generation_updates_complete"
    return {"execution_id": value["original_training_execution_id"], "registration_sha256": value["original_training_registration_sha256"],
        "physical_completion": "confirmed", "native_game_authority_issued": False, "original_process_authority_issued": False,
        "parent_must_verify_process_and_artifacts": True, "durable_checkpoint_boundary": True, "status": status,
        "checkpoint_pins": [value["checkpoint"]], "window": {"status": status, "completed_updates": counts,
            "ledger": {"completed": counts, "pending_microbatches": 0,
                "spec": {"model_sha256": value["training_registration"]["model_sha256"]}}, "checkpoints": [value["checkpoint"]["path"]]}}


class ExportRegistrationTests(SourceOnlyTests):
    def test_default_is_metadata_only_and_does_not_execute_export(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main([]), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["status"], "metadata_only")
        self.assertFalse(value["actual_export_executed"])
        self.assertEqual(value["optimizer_updates_executed"], 0)

    def test_frozen_and_g1_registration_keep_original_training_seals(self):
        with tempfile.TemporaryDirectory() as directory:
            for kind in cli.KINDS:
                value = registration(Path(directory).resolve(), kind)
                validated = cli.validate_export_registration(value)
                self.assertEqual(validated, value)
                self.assertIsNot(validated, value)
                loaded = cli.loading_registration(value)
                self.assertEqual(loaded["checkpoint"]["domain"], cli.CHECKPOINT_DOMAIN)
                self.assertEqual(loaded["checkpoint"]["mode"], "resume")
                self.assertEqual(loaded["datasets"], value["training_registration"]["datasets"])
                self.assertEqual(loaded["source_components"], value["training_registration"]["source_components"])

    def test_json_cannot_claim_native_or_gpu_export_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            value = registration(Path(directory).resolve())
            for extra in ("native_chain_verified", "arena_eligible", "cuda_status"):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    cli.validate_export_registration(dict(value, **{extra: True}))
            changed = copy.deepcopy(value)
            changed["export_source_components"]["trainer"] = "f" * 64
            with self.assertRaises(ValueError):
                cli.validate_export_registration(changed)

    def test_registered_phase_outputs_and_caps_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            value = registration(Path(directory).resolve())
            for name, change in (("kind", "g1_actual_generation"), ("max_wall_ns", 0),
                    ("max_wall_ns", cli.MAX_WALL_NS + 1), ("model_epoch", True),
                    ("receipt_path", value["training_registration"]["receipt_path"]),
                    ("max_output_bytes", value["max_output_bytes"] - 1)):
                changed = copy.deepcopy(value)
                changed[name] = change
                with self.subTest(name=name, change=change), self.assertRaises(ValueError):
                    cli.validate_export_registration(changed)

    def test_original_terminal_requires_exact_clean_last_checkpoint_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            for kind in cli.KINDS:
                value = registration(Path(directory).resolve(), kind)
                original = terminal(value)
                self.assertEqual(cli.validate_original_terminal(original, value), original["window"]["completed_updates"])
                for name, change in (("physical_completion", "unknown_quarantined"),
                        ("native_game_authority_issued", True), ("durable_checkpoint_boundary", False),
                        ("execution_id", "foreign-original")):
                    with self.subTest(kind=kind, name=name), self.assertRaises(ValueError):
                        cli.validate_original_terminal(dict(original, **{name: change}), value)
                bad = copy.deepcopy(original)
                bad["window"]["ledger"]["pending_microbatches"] = 1
                with self.assertRaises(ValueError):
                    cli.validate_original_terminal(bad, value)
                bad = copy.deepcopy(original)
                bad["window"]["checkpoints"] = [str(Path(directory).resolve() / "not-latest.pt")]
                with self.assertRaises(ValueError):
                    cli.validate_original_terminal(bad, value)

    def test_execute_invalid_startup_rejects_before_neural_import(self):
        with self.assertRaises(ValueError):
            cli.main(["--execute", "--original-export-child", "--startup-sha256", "f" * 64,
                "--startup-bytes", "2", "--execution-until-ns", "110", "--whole-until-ns", "120"],
                stdin=io.BytesIO(b"{}"), clock=lambda: 11)

    def test_metadata_writers_check_exact_byte_credit_before_any_output(self):
        from rz_pals_model.generation_export import _atomic_json
        from rz_pals_model.artifacts import atomic_json
        value = {"input": "한국어", "unchanged": True}
        expected = len((json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
        with tempfile.TemporaryDirectory() as directory:
            for number, writer in enumerate((_atomic_json, atomic_json)):
                path = Path(directory).resolve() / (str(number) + ".json")
                observed = []
                def reject(size):
                    observed.append(size)
                    raise ValueError("registered disk credit exhausted")
                with self.assertRaises(ValueError):
                    writer(path, value, output_guard=reject)
                self.assertEqual(observed, [expected])
                self.assertFalse(path.exists())
                self.assertFalse(path.with_name(path.name + ".partial").exists())

    def test_serialized_mock_graph_checks_credit_before_partial_write(self):
        from rz_pals_model.onnx_virtual_review import _save
        class MetadataDocument:
            def SerializeToString(self):
                return b"source-only-mock-serialization"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            observed = []
            def reject(size):
                observed.append(size)
                raise ValueError("registered disk credit exhausted")
            with patch("rz_pals_model.artifacts.output_directory", return_value=root):
                with self.assertRaises(ValueError):
                    _save(MetadataDocument(), {}, root / "mock.onnx", output_guard=reject)
            self.assertEqual(observed, [len(b"source-only-mock-serialization")])
            self.assertEqual(list(root.iterdir()), [])

    def test_inventory_original_deadline_rejects_before_file_read(self):
        from rz_pals_model import virtual_control_inventory as inventory
        with patch.object(inventory.time, "monotonic_ns", return_value=100):
            with patch.object(inventory.legacy, "_external", side_effect=AssertionError("read before original deadline validation")):
                for until in (99, 100, True, 100.5):
                    with self.subTest(until=until), self.assertRaisesRegex(ValueError, "original_inventory_deadline"):
                        inventory.produce_virtual_control_inventory("not-read", "f" * 64, "not-written",
                            opt_in=True, max_seconds=30, until_ns=until)
                with self.assertRaisesRegex(ValueError, "finite_metadata_wall"):
                    inventory.produce_virtual_control_inventory("not-read", "f" * 64, "not-written",
                        opt_in=True, max_seconds=61, until_ns=1000)

    def test_actual_inventory_byte_guard_denies_before_exclusive_write(self):
        from rz_pals_model import virtual_control_inventory as inventory
        value = {"schema": inventory.SCHEMA, "total_nodes": 1,
            "actual_provider_placement": "not_observed", "cpu_allowlist": "not_created"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "inventory.json"
            observed = []
            def reject(size):
                observed.append(size)
                raise ValueError("registered actual export disk credit exhausted")
            with self.assertRaises(ValueError):
                inventory._write_exclusive(path, value, output_guard=reject)
            self.assertFalse(path.exists())
            self.assertEqual(observed, [len((json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode())])
