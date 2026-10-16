"""Metadata-only CLI checks; no neural/model/optimizer execution is allowed."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_generation_contract import SourceOnlyTests
from rz_pals_model import generation_training_cli as cli
from rz_pals_model.config import MODEL_SEMANTICS_V3


def registration(root):
    pin = lambda name: {"path": str(root / name), "sha256": "a" * 64, "bytes": 1}
    corpus = {"manifest": pin("corpus.json"), "manifest_sha256": "b" * 64, "corpus_root": str(root / "corpus")}
    return {"schema": cli.REGISTRATION_DOMAIN, "window_id": "metadata-only-window",
        "spec": pin("spec.json"), "phase": "pc", "datasets": {"pc": corpus, "v": None},
        "checkpoint": {"artifact": pin("g0.pt"), "domain": MODEL_SEMANTICS_V3, "mode": "initialize"},
        "source_components": {name: "c" * 64 for name in cli.SOURCE_COMPONENTS},
        "runtime_sha256": "d" * 64, "model_sha256": "a" * 64, "frozen_pc_sha256": None,
        "model_epoch": None, "training_runtime": {"torch": "2.8.0", "cuda": "12.8", "cudnn": 90100, "device_name": "NVIDIA L4"},
        "staging_root": str(root), "index_root": str(root / "index"), "checkpoint_directory": str(root / "checkpoints"),
        "receipt_path": str(root / "terminal.json"),
        "max_wall_ns": cli.MAX_WALL_NS, "max_updates": 250, "max_output_bytes": 18 * 1024**3}


class RegistrationTests(SourceOnlyTests):
    def test_strict_registration_does_not_import_neural_frameworks(self):
        with tempfile.TemporaryDirectory() as directory:
            value = registration(Path(directory).resolve())
            snapshot = cli.validate_registration(value)
            self.assertEqual(snapshot, value)
            self.assertIsNot(snapshot, value)
            self.assertEqual(len(cli.registration_digest(value)), 64)

    def test_execute_without_live_owner_rejects_before_reading_any_file(self):
        with patch.object(Path, "open", side_effect=AssertionError("must not read")):
            with self.assertRaises(cli.NativeTrainingProducerUnavailable):
                cli.main(["--execute", "--execution-registration", "/missing/not-read.json", "--registration-sha256", "f" * 64])

    def test_dictionary_cannot_supply_native_producer_authority(self):
        with self.assertRaises(cli.NativeTrainingProducerUnavailable):
            cli.execute_registered_window({}, {}, {"native_chain_verified": True}, until_ns=10)

    def test_metadata_default_forwards_legacy_arguments_without_dispatch(self):
        args = ["--spec", "x", "--dataset", "y", "--phase", "pc"]
        with patch("rz_pals_model.generation_training.main", return_value=19) as metadata:
            self.assertEqual(cli.main(args), 19)
            metadata.assert_called_once_with(args)

    def test_unknown_or_inconsistent_recipe_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = registration(Path(directory).resolve())
            for key, value in (("max_updates", 251), ("max_wall_ns", cli.MAX_WALL_NS + 1),
                    ("max_wall_ns", 0), ("max_output_bytes", 18 * 1024**3 + 1), ("max_updates", True)):
                changed = copy.deepcopy(base)
                changed[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    cli.validate_registration(changed)
            changed = copy.deepcopy(base)
            changed["native_intake_verified"] = True
            with self.assertRaises(ValueError):
                cli.validate_registration(changed)

    def test_v_requires_separate_corpus_freeze_epoch_and_actual_resume_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            value = registration(Path(directory).resolve())
            value["phase"] = "v"
            with self.assertRaises(ValueError):
                cli.validate_registration(value)
            value["datasets"]["v"] = copy.deepcopy(value["datasets"]["pc"])
            value["checkpoint"]["mode"] = "resume"
            value["checkpoint"]["domain"] = cli.CHECKPOINT_DOMAIN
            value["frozen_pc_sha256"], value["model_epoch"] = "e" * 64, 1
            self.assertEqual(cli.validate_registration(value)["phase"], "v")
            value["checkpoint"]["domain"] = "diagnostic"
            with self.assertRaises(ValueError):
                cli.validate_registration(value)

    def test_secret_relative_and_noncanonical_paths_are_rejected(self):
        for path in ("relative.json", "/private/.env", "/private/ssh-key.key", "/private/service_account.json", "/private/api_key.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                cli._safe_absolute(path)

    def test_pinned_json_uses_exact_read_bytes_and_original_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "metadata.json"
            raw = json.dumps({"metadata_only": True}).encode()
            path.write_bytes(raw)
            pin = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            self.assertEqual(cli._pinned_bytes(pin, maximum=1024, until_ns=100, clock=lambda: 0), raw)
            with self.assertRaises(ValueError):
                cli._pinned_bytes(pin, maximum=1024, until_ns=100, clock=lambda: 100)
            changed = dict(pin, sha256="0" * 64)
            with self.assertRaises(ValueError):
                cli._pinned_bytes(changed, maximum=1024, until_ns=100, clock=lambda: 0)

    def test_original_child_startup_preserves_deadlines_and_separate_origin_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            value = registration(Path(directory).resolve())
            startup = {"schema": cli.STARTUP_DOMAIN, "execution_id": "original-training-1",
                "registration_sha256": cli.registration_digest(value), "registration": value,
                "original_started_ns": 10, "execution_until_ns": 110, "whole_until_ns": 120,
                "dataset_pin_scope": "external_original_archive_pin_native_chain_readback_only",
                "dependency_directories": [{"path": str(Path(directory).resolve() / "site-packages"), "device": 1, "inode": 2}],
                "python_runtime": {"version": "3.11.8", "executable": {"path": str(Path(directory).resolve() / "python"), "sha256": "f" * 64, "bytes": 1}}}
            raw = json.dumps(startup).encode()
            kwargs = {"expected_raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": raw,
                "execution_until_ns": 110, "whole_until_ns": 120, "now_ns": 11}
            self.assertEqual(cli.validate_startup(startup, **kwargs), value)
            for name, changed in (("execution_until_ns", 111), ("whole_until_ns", 121), ("now_ns", 110)):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    cli.validate_startup(startup, **dict(kwargs, **{name: changed}))
            with self.assertRaises(ValueError):
                cli.validate_startup(dict(startup, native_game_authority=True), **kwargs)
            for dependencies in ([], startup["dependency_directories"] * 2,
                    [dict(startup["dependency_directories"][0], inode=True)],
                    [dict(startup["dependency_directories"][0], path="/private/.env")]):
                changed = dict(startup, dependency_directories=dependencies)
                changed_raw = json.dumps(changed).encode()
                with self.subTest(dependencies=dependencies), self.assertRaises(ValueError):
                    cli.validate_startup(changed, **dict(kwargs, raw_bytes=changed_raw,
                        expected_raw_sha256=hashlib.sha256(changed_raw).hexdigest()))

    def test_original_child_invalid_startup_cannot_import_a_neural_framework(self):
        argv = ["--execute", "--original-trainer-child", "--startup-sha256", "f" * 64,
            "--startup-bytes", "2", "--execution-until-ns", "110", "--whole-until-ns", "120"]
        with self.assertRaises(ValueError):
            cli.original_trainer_child_main(argv, stdin=io.BytesIO(b"{}"), clock=lambda: 11)

    def test_frozen_pc_export_requires_explicit_execution_before_import(self):
        with self.assertRaises(ValueError):
            cli.export_frozen_pc_collection({}, "/unused.pt", "f" * 64, None, None, None,
                sampler_validator=None, model_epoch=1, directory="/unused", rules_profile_json=None)

    def test_checkpoint_writer_checks_credit_before_writing(self):
        from rz_pals_model.generation_training import _BoundedCheckpointWriter
        stream = io.BytesIO()
        writer = _BoundedCheckpointWriter(stream, 3)
        self.assertEqual(writer.write(b"abc"), 3)
        with self.assertRaises(ValueError):
            writer.write(b"d")
        self.assertEqual(stream.getvalue(), b"abc")

    def test_staging_output_guard_reserves_terminal_receipt_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            value = registration(root)
            (root / "source-only.bin").write_bytes(b"abc")
            self.assertEqual(cli._staging_bytes(value), 3)
            self.assertEqual(cli._output_credit(value), value["max_output_bytes"] - cli.MAX_JSON_BYTES - 3)

    def test_actual_runtime_source_byte_readback_is_bounded_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "owned-source.py"
            path.write_bytes(b"# source-only\n")
            actual = cli._regular_readback(path, maximum=128, until_ns=100, clock=lambda: 0)
            self.assertEqual(actual["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(actual["bytes"], path.stat().st_size)
            with self.assertRaises(ValueError):
                cli._regular_readback(path, maximum=1, until_ns=100, clock=lambda: 0)
            with self.assertRaises(ValueError):
                cli._regular_readback(path, maximum=128, until_ns=100, clock=lambda: 100)

    def test_frozen_collection_boundary_is_exact_pc4000_not_full_generation(self):
        from rz_pals_model.generation_training import generation_spec
        spec = generation_spec(generation_id="metadata-frozen-boundary", source_sha256="1" * 64,
            dataset_sha256="2" * 64, split_sha256="3" * 64, model_sha256="4" * 64, runtime_sha256="5" * 64)
        ledger = cli.GenerationLedger(spec, completed={"proposer": 2000, "critic": 2000, "validator": 0},
            dispatched={"proposer": 2000, "critic": 2000, "validator": 0}, phase="v", next_role="validator")
        self.assertIs(cli.validate_frozen_pc_boundary(ledger, model_epoch=1), ledger)
        with self.assertRaises(ValueError):
            cli.validate_frozen_pc_boundary(ledger, model_epoch=0)
        complete = cli.GenerationLedger(spec, completed={"proposer": 2000, "critic": 2000, "validator": 1000},
            dispatched={"proposer": 2000, "critic": 2000, "validator": 1000}, phase="complete", next_role="validator")
        with self.assertRaises(ValueError):
            cli.validate_frozen_pc_boundary(complete, model_epoch=1)


if __name__ == "__main__":
    unittest.main()
