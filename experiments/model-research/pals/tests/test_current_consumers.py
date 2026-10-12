"""Current-view consumer fixtures; no real model, Rust child or optimizer runs.

The production preparation/producer boundaries run against a tiny numeric
stand-in. Files and labels are contract fixtures, not collected chess evidence.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rz_pals_model import preparation_check as preparation
from rz_pals_model import verifier_producer as producer
from rz_pals_model.config import ModelConfig, TASKS
from rz_pals_model.training import _canonical, load_collected_dataset
from test_training import (REGISTRY, fixture, label_chain_rows, native_sidecar,
                           sha, write_fixture_collection)


class FrozenNumericFixture(nn.Module):
    """Preserve declared output shapes without loading or running a model."""
    def __init__(self, on_forward=None):
        super().__init__()
        self.config = ModelConfig()
        self.fixed = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.experts = nn.ModuleDict({role: nn.Identity() for role in ("proposer", "critic", "validator")})
        self.on_forward = on_forward

    def public_encoder(self, board, metadata, records, record_mask):
        if self.on_forward is not None:
            self.on_forward()
        batch = board.shape[0]
        return (torch.zeros((batch, 1, 384), dtype=torch.float32), torch.zeros((batch, 1, 384), dtype=torch.float32),
                torch.ones((batch, 1), dtype=torch.bool))

    def role_graph(self, role):
        def forward(keys, values, mask, candidates, candidate_mask, divergences, divergence_mask, query):
            batch = candidates.shape[0]
            result = (torch.zeros((batch, candidates.shape[1]), dtype=torch.float32), torch.zeros((batch, 3), dtype=torch.float32),
                      torch.zeros((batch, 16, 384), dtype=torch.float32))
            if role != "proposer":
                result += (torch.zeros((batch, len(TASKS) if role == "validator" else divergences.shape[1]), dtype=torch.float32),)
            return result
        return forward


def history_collection(directory, rows):
    """Extend the existing single-row collector fixture with bounded history."""
    write_fixture_collection(directory, rows[0], native_sidecar(rows[0]))
    receipt = json.loads((directory / "receipt.json").read_bytes())
    by_input = {row["input"]["sha256"]: row for row in rows}
    files = {"records.jsonl": rows,
             "native-inputs.jsonl": [native_sidecar(row) for row in by_input.values()],
             "source-registry.jsonl": [REGISTRY],
             "split.jsonl": [{"games": {row["input"]["snapshot"]["game_id"]: "train" for row in rows}}]}
    for name, values in files.items():
        raw = b"".join((json.dumps(value, separators=(",", ":")) + "\n").encode() for value in values)
        (directory / name).write_bytes(raw)
        receipt["artifacts"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    receipt["audit"]["records"] = len(rows)
    raw = json.dumps(receipt).encode()
    (directory / "receipt.json").write_bytes(raw)
    receipt_sha = hashlib.sha256(raw).hexdigest()
    admitted = load_collected_dataset(directory, expected_receipt_sha256=receipt_sha,
                                      expected_encoder_source_sha256=sha("fixture-encoder-source"))
    return admitted, receipt_sha


def legacy_six_rows():
    return [fixture(role, game=f"legacy-{role}-{index}")
            for role in ("proposer", "critic") for index in range(3)]


class CurrentConsumerTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer updates forbidden"))
        guard.start()
        self.addCleanup(guard.stop)
        child = patch.object(producer.subprocess, "Popen", side_effect=AssertionError("Rust children forbidden in numeric fixture"))
        child.start()
        self.addCleanup(child.stop)

    def preparation_fixture(self, root, rows, *, on_forward=None):
        collection = root / "collection"
        collection.mkdir()
        data, receipt_sha = history_collection(collection, rows)
        checkpoint = root / "numeric-fixture.pt"
        checkpoint.write_bytes(b"numeric-checkpoint-fixture")
        checkpoint.with_name("checkpoint.json").write_text("{}", encoding="utf-8")
        metadata = {"trained": False, "training_steps": 0,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        model = FrozenNumericFixture(None if on_forward is None else lambda: on_forward(data))
        arguments = dict(collection=collection, receipt_sha256=receipt_sha,
                         encoder_source_sha256=sha("fixture-encoder-source"), checkpoint=checkpoint,
                         output=root / "report.json", max_records=16, max_wall_time_ms=10000,
                         max_output_bytes=262144, max_input_bytes=1048576, batch_size=2, threads=1)
        return data, model, metadata, arguments

    def run_preparation_fixture(self, data, model, metadata, arguments):
        with patch.object(preparation, "load_collected_dataset", return_value=data), \
             patch.object(preparation, "load_checkpoint", return_value=(model, metadata)):
            result = preparation.run_preparation_check(**arguments)
        report = json.loads(arguments["output"].read_bytes())
        return result, report

    def test_preparation_consumes_leaves_and_preserves_entire_raw_history(self):
        rows = label_chain_rows() + [fixture("critic", "unlabeled-critic")]
        with tempfile.TemporaryDirectory() as temporary:
            data, model, metadata, arguments = self.preparation_fixture(Path(temporary), rows)
            before = copy.deepcopy(data.records)
            result, report = self.run_preparation_fixture(data, model, metadata, arguments)
            self.assertEqual(result["records_consumed"], 2)
            self.assertEqual((report["records"], report["current_records"], report["records_consumed"]), (4, 2, 2))
            self.assertEqual(report["role_counts"], {"proposer": 1, "critic": 1})
            self.assertEqual(report["current_view_sha256"], data.current_view.sha256)
            self.assertEqual(report["raw_records_sha256"], _canonical("rz-pals-preparation-immutable-records/1", before))
            self.assertTrue(report["raw_records_unchanged"])
            self.assertEqual({index for batch in report["batches"] for index in batch["indices"]}, {2, 3})
            self.assertEqual(report["target_masks"]["wdl_rows"], 1)
            self.assertEqual(report["target_masks"]["masked_wdl_rows"], 1)
            self.assertEqual(data.records, before)

    def test_preparation_preserves_legacy_six_rows_and_unknown_value_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, model, metadata, arguments = self.preparation_fixture(Path(temporary), legacy_six_rows())
            arguments["require_masked_value"] = True
            result, report = self.run_preparation_fixture(data, model, metadata, arguments)
            self.assertEqual(result["records_consumed"], 6)
            self.assertEqual((report["records"], report["current_records"]), (6, 6))
            self.assertEqual(report["role_counts"], {"proposer": 3, "critic": 3})
            self.assertEqual(report["target_masks"]["wdl_rows"], 0)
            self.assertEqual(report["target_masks"]["masked_wdl_rows"], 6)

    def test_preparation_rejects_mutation_of_unselected_historical_row(self):
        def mutate(data):
            data.records[0]["verifier_private"] = {"injected": True}

        with tempfile.TemporaryDirectory() as temporary:
            data, model, metadata, arguments = self.preparation_fixture(Path(temporary), label_chain_rows(), on_forward=mutate)
            with self.assertRaisesRegex(ValueError, "immutable raw dataset history"), \
                 patch.object(preparation, "load_collected_dataset", return_value=data), \
                 patch.object(preparation, "load_checkpoint", return_value=(model, metadata)):
                preparation.run_preparation_check(**arguments)
            self.assertFalse(arguments["output"].exists())

    def producer_fixture(self, root, rows, *, on_forward=None, max_steps=16):
        collection = root / "collection"
        collection.mkdir()
        data, receipt_sha = history_collection(collection, rows)
        checkpoint = root / "numeric-fixture.pt"
        checkpoint.write_bytes(b"numeric-checkpoint-fixture")
        checkpoint.with_name("checkpoint.json").write_text("{}", encoding="utf-8")
        binary = root / "numeric-cpu-fixture.bin"
        binary.write_bytes(b"numeric-cpu-fixture-never-executed")
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        metadata = {"trained": False, "training_steps": 0, "checkpoint_sha256": checkpoint_sha}
        model = FrozenNumericFixture(None if on_forward is None else lambda: on_forward(data))
        arguments = dict(collection=collection, receipt_sha256=receipt_sha,
                         encoder_source_sha256=sha("fixture-encoder-source"), checkpoint=checkpoint,
                         checkpoint_sha256=checkpoint_sha, cpu_binary=binary,
                         cpu_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), output=root / "private-bank",
                         max_games=16, max_steps=max_steps, max_nodes=1000, max_wall_time_ms=10000,
                         max_output_bytes=1048576, max_forward_flops=100000000000,
                         max_input_bytes=1048576, max_nodes_per_check=10, max_task_wall_time_ms=1000,
                         max_cpu_output_bytes=1024, allowed_tasks=("defer",))
        return data, model, metadata, arguments

    def deferred_fixture(self, binary, request, limits, *, capture, pipe_reservation):
        result = {name: request[name] for name in ("schema", "task", "context_sha256", "cpu_binary_sha256",
                                                  "rules_state_sha256", "rules_history_sha256", "branch_sha256")}
        result.update(board_fen=request["expected_board_fen"], status="deferred", reason=None,
                      baseline=None, after=None, nodes=0, elapsed_ms=0, resume_kind=None,
                      product_verifier_enabled=False, deadline_exceeded=False)
        raw = producer._json(result) + b"\n"
        pipe_reservation.consume(len(raw))
        pipe_reservation.release()
        capture.update(stdout=raw, stderr=b"", spawned=False, reaped=True, exit_code=0)
        return result

    def run_producer_fixture(self, data, model, metadata, arguments):
        with patch.object(producer, "load_collected_dataset", return_value=data), \
             patch.object(producer, "load_checkpoint", return_value=(model, metadata)), \
             patch.object(producer, "run_cpu_bridge", side_effect=self.deferred_fixture) as bridge, \
             patch.object(producer, "verifier_input", wraps=producer.verifier_input) as prepare:
            result = producer.run_verifier_producer(**arguments)
        receipt = json.loads((arguments["output"] / "receipt.json").read_bytes())
        selected = [call.args[0] for call in prepare.call_args_list[:result["counts"]["selections"]]]
        return result, receipt, bridge.call_count, selected

    def test_producer_runs_one_current_parent_and_preserves_raw_parent_count_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, model, metadata, arguments = self.producer_fixture(Path(temporary), label_chain_rows())
            before = copy.deepcopy(data.records)
            result, receipt, dispatches, selected = self.run_producer_fixture(data, model, metadata, arguments)
            self.assertEqual((dispatches, result["counts"]["selections"], result["reloaded_records"]), (1, 1, 1))
            self.assertEqual((receipt["parent_records_available"], receipt["current_parent_records_available"]), (3, 1))
            self.assertEqual(receipt["parent_records_sha256"], producer._hash("rz-pals-private-immutable-parents/1", before))
            self.assertEqual(receipt["parent_current_view_sha256"], data.current_view.sha256)
            self.assertEqual(selected[0]["future_label"]["observed_sequence"], 9)
            self.assertEqual(data.records, before)
            self.assertTrue(result["private_bank_reloaded"])

    def test_producer_step_limit_applies_to_current_ordinal_not_raw_leaf_index(self):
        for rows in (label_chain_rows(), list(reversed(label_chain_rows()))):
            with self.subTest(order=[row["future_label"] is None for row in rows]), tempfile.TemporaryDirectory() as temporary:
                data, model, metadata, arguments = self.producer_fixture(Path(temporary), rows, max_steps=1)
                result, receipt, dispatches, selected = self.run_producer_fixture(data, model, metadata, arguments)
                self.assertEqual((dispatches, result["counts"]["selections"]), (1, 1))
                self.assertEqual(selected[0]["future_label"]["observed_sequence"], 9)
                self.assertEqual(receipt["resources"]["usage"]["steps"], 1)

    def test_producer_preserves_legacy_six_distinct_parent_selections(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, model, metadata, arguments = self.producer_fixture(Path(temporary), legacy_six_rows())
            result, receipt, dispatches, selected = self.run_producer_fixture(data, model, metadata, arguments)
            self.assertEqual((dispatches, result["counts"]["selections"], result["reloaded_records"]), (6, 6, 6))
            self.assertEqual((receipt["parent_records_available"], receipt["current_parent_records_available"]), (6, 6))
            self.assertEqual({row["input"]["sha256"] for row in selected}, {row["input"]["sha256"] for row in data.records})

    def test_producer_rejects_historical_mutation_with_failure_receipt(self):
        def mutate(data):
            data.records[0]["verifier_private"] = {"injected": True}

        with tempfile.TemporaryDirectory() as temporary:
            data, model, metadata, arguments = self.producer_fixture(Path(temporary), label_chain_rows(), on_forward=mutate)
            with self.assertRaisesRegex(ValueError, "immutable raw dataset history"), \
                 patch.object(producer, "load_collected_dataset", return_value=data), \
                 patch.object(producer, "load_checkpoint", return_value=(model, metadata)), \
                 patch.object(producer, "run_cpu_bridge", side_effect=self.deferred_fixture):
                producer.run_verifier_producer(**arguments)
            receipt = json.loads((arguments["output"] / "receipt.json").read_bytes())
            self.assertFalse(receipt["complete"])
            self.assertEqual(receipt["failure"]["type"], "ValueError")
            self.assertIn("immutable raw dataset history", receipt["failure"]["message"])
            self.assertEqual(receipt["counts"]["selections"], 1)


if __name__ == "__main__":
    unittest.main()
