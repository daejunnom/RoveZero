"""Strict preparation consumption fixtures, with no loaded model or optimizer.

Synthetic sealed files exercise the real frozen admission boundary. Numeric
forward shapes come from the existing tiny stand-in; this is no collector,
training, native-runtime, GPU or performance acceptance evidence.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rz_pals_model import preparation_check as preparation
from rz_pals_model.training import load_frozen_collected_dataset
from test_current_consumers import FrozenNumericFixture
from test_training import (fixture, frozen_fixture_bytes, native_sidecar, sha,
                           write_fixture_collection, write_frozen_fixture_collection)


class FrozenPreparationTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(torch.optim, "AdamW", side_effect=AssertionError("optimizer creation forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def configuration(self, root, *, kinds=("cpu",), legacy=False):
        collection = root / "collection"
        collection.mkdir()
        registrations = []
        if legacy:
            row = fixture()
            receipt = write_fixture_collection(collection, row, native_sidecar(row))
        else:
            frozen = write_frozen_fixture_collection(collection, kinds=kinds)
            receipt = frozen["expected_receipt_sha256"]
            registrations = frozen["producer_registrations"]
        checkpoint = root / "numeric-fixture.pt"
        checkpoint.write_bytes(b"numeric-checkpoint-fixture")
        checkpoint.with_name("checkpoint.json").write_bytes(b"{}")
        metadata = {"trained": False, "training_steps": 0,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        arguments = dict(collection=collection, receipt_sha256=receipt, checkpoint=checkpoint,
                         output=root / "report.json", max_records=16, max_wall_time_ms=10000,
                         max_output_bytes=262144, max_input_bytes=1048576, batch_size=2, threads=1)
        if legacy:
            arguments["encoder_source_sha256"] = sha("fixture-encoder-source")
        else:
            registration_set = root / "registered-producers.json"
            declared = [{"pin": copy.deepcopy(item["pin"]),
                         "registration_path": item["registration_path"].relative_to(root).as_posix(),
                         "checked_source_path": item["checked_source_path"].relative_to(root).as_posix()}
                        for item in registrations]
            registration_set.write_bytes(frozen_fixture_bytes({"version": preparation.REGISTRATION_SET_SCHEMA,
                                                                "producer_registrations": declared}))
            arguments.update(producer_registration_set=registration_set,
                             producer_registration_set_sha256=hashlib.sha256(registration_set.read_bytes()).hexdigest())
        return arguments, metadata, registrations

    def run_fixture(self, arguments, metadata, *, on_forward=None):
        admitted = []

        def load_strict(*args, **kwargs):
            data = load_frozen_collected_dataset(*args, **kwargs)
            admitted.append(data)
            return data

        model = FrozenNumericFixture(None if on_forward is None else lambda: on_forward(admitted[0]))
        with patch.object(preparation, "load_frozen_collected_dataset", side_effect=load_strict), \
             patch.object(preparation, "load_checkpoint", return_value=(model, metadata)):
            result = preparation.run_preparation_check(**arguments)
        return result, json.loads(arguments["output"].read_bytes()), admitted

    def test_strict_cpu_and_two_native_producers_are_consumed(self):
        for kinds in (("cpu",), ("native", "native")):
            with self.subTest(kinds=kinds), tempfile.TemporaryDirectory() as temporary:
                arguments, metadata, _ = self.configuration(Path(temporary), kinds=kinds)
                with patch.object(preparation, "load_collected_dataset", side_effect=AssertionError("strict may not retry legacy")):
                    result, report, admitted = self.run_fixture(arguments, metadata)
                self.assertEqual(result["records_consumed"], len(kinds))
                self.assertEqual(report["encoder_source_sha256"], None)
                self.assertEqual(report["optimizer_steps"], 0)
                self.assertFalse(report["optimizer_created"])
                self.assertFalse(report["training_executed"])
                self.assertFalse(report["gpu_executed"])
                admission = report["producer_admission"]
                self.assertEqual(admission["producers"], len(kinds))
                self.assertEqual(admission["registration_set"]["sha256"], arguments["producer_registration_set_sha256"])
                self.assertEqual(admission["scope"], "frozen_collected_cpu_forward_preparation")
                self.assertEqual(admission["metadata_audit"]["scope"], "metadata_only")
                self.assertTrue(admission["frozen_admission_unchanged"])
                self.assertTrue(admission["current_view_unchanged"])
                self.assertEqual(admission["frozen_admission_sha256"], preparation._canonical(
                    "rz-pals-preparation-frozen-admission/1", admitted[0].frozen_admission))
                self.assertEqual(report["target_masks"]["masked_wdl_rows"], len(kinds))

    def test_strict_missing_required_and_tampered_source_fail_without_legacy(self):
        for action in ("missing", "source", "set-sha", "epoch"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as temporary:
                arguments, metadata, registrations = self.configuration(Path(temporary))
                if action == "missing":
                    (arguments["collection"] / "producer-prepared.jsonl").unlink()
                elif action == "source":
                    registrations[0]["checked_source_path"].write_bytes(b"[]")
                elif action == "set-sha":
                    arguments["producer_registration_set_sha256"] = sha("wrong-independent-set")
                else:
                    path = arguments["producer_registration_set"]
                    value = json.loads(path.read_bytes())
                    value["producer_registrations"][0]["pin"]["frozen_epoch"] += 1
                    path.write_bytes(frozen_fixture_bytes(value))
                    arguments["producer_registration_set_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                with patch.object(preparation, "load_collected_dataset", side_effect=AssertionError("strict may not retry legacy")), \
                     patch.object(preparation, "load_checkpoint", side_effect=AssertionError("failed admission may not load model")):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        preparation.run_preparation_check(**arguments)
                self.assertFalse(arguments["output"].exists())

    def test_strict_half_selection_and_legacy_encoder_pin_are_rejected(self):
        for absent in ("producer_registration_set", "producer_registration_set_sha256", "legacy-encoder"):
            with self.subTest(absent=absent), tempfile.TemporaryDirectory() as temporary:
                arguments, _, _ = self.configuration(Path(temporary))
                if absent == "legacy-encoder":
                    arguments["encoder_source_sha256"] = sha("fixture-encoder-source")
                else:
                    arguments.pop(absent)
                with patch.object(preparation, "load_frozen_collected_dataset", side_effect=AssertionError("invalid selection must not load")), \
                     patch.object(preparation, "load_collected_dataset", side_effect=AssertionError("invalid selection must not fall back")):
                    with self.assertRaisesRegex(ValueError, "strict preparation"):
                        preparation.run_preparation_check(**arguments)

    def test_strict_budget_counts_set_checkpoint_and_external_producer_source_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments, metadata, registrations = self.configuration(Path(temporary), kinds=("native", "native"))
            collection = arguments["collection"]
            receipt = json.loads((collection / "receipt.json").read_bytes())
            paths = {arguments["checkpoint"], arguments["checkpoint"].with_name("checkpoint.json"),
                     arguments["producer_registration_set"], collection / "receipt.json"}
            paths.update(collection / name for name in receipt["artifacts"])
            paths.update(entry[key] for entry in registrations for key in ("registration_path", "checked_source_path"))
            total = sum(path.stat().st_size for path in paths)
            arguments["max_input_bytes"] = total - 1
            with patch.object(preparation, "load_collected_dataset", side_effect=AssertionError("strict may not retry legacy")), \
                 patch.object(preparation, "load_checkpoint", side_effect=AssertionError("budget failure may not load model")):
                with self.assertRaisesRegex(ValueError, "budget"):
                    preparation.run_preparation_check(**arguments)
            arguments["max_input_bytes"] = total
            _, report, _ = self.run_fixture(arguments, metadata)
            self.assertEqual(report["resources"]["input_bytes_preflight"], total)

    def test_strict_rechecks_raw_current_and_admission_after_numeric_forward(self):
        def mutate(data, kind):
            if kind == "raw":
                data.records[0]["verifier_private"] = {"fixture-mutation": 1}
            elif kind == "current":
                data._current_indices = frozenset()
            else:
                data.frozen_admission["metadata_audit"]["capture_sha256"] = sha("mutated-after-admission")

        for kind in ("raw", "current", "admission"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                arguments, metadata, _ = self.configuration(Path(temporary))
                with self.assertRaisesRegex(ValueError, "changed"):
                    self.run_fixture(arguments, metadata, on_forward=lambda data: mutate(data, kind))
                self.assertFalse(arguments["output"].exists())

    def test_strict_refuses_secret_named_registration_paths_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments, _, _ = self.configuration(Path(temporary))
            path = arguments["producer_registration_set"]
            value = json.loads(path.read_bytes())
            value["producer_registrations"][0]["checked_source_path"] = "credential-source.json"
            path.write_bytes(frozen_fixture_bytes(value))
            arguments["producer_registration_set_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(preparation, "load_frozen_collected_dataset", side_effect=AssertionError("secret path must not reach loader")):
                with self.assertRaisesRegex(ValueError, "secret"):
                    preparation.run_preparation_check(**arguments)

    def test_default_legacy_call_preserves_report_shape_and_encoder_requirement(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments, metadata, _ = self.configuration(Path(temporary), legacy=True)
            receipt_before = (arguments["collection"] / "receipt.json").read_bytes()
            with patch.object(preparation, "load_frozen_collected_dataset", side_effect=AssertionError("legacy must keep old loader")):
                result, report, admitted = self.run_fixture(arguments, metadata)
            self.assertEqual(result["records_consumed"], 1)
            self.assertEqual(admitted, [])
            self.assertNotIn("producer_admission", report)
            self.assertEqual(report["schema"], preparation.SCHEMA)
            self.assertEqual(report["encoder_source_sha256"], sha("fixture-encoder-source"))
            self.assertEqual((arguments["collection"] / "receipt.json").read_bytes(), receipt_before)
            arguments.pop("encoder_source_sha256")
            with self.assertRaisesRegex(ValueError, "legacy preparation requires"):
                preparation.run_preparation_check(**arguments)


if __name__ == "__main__":
    unittest.main()
