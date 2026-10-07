"""Bounded preparation checks. No optimizer update or training is performed.

The small in-memory rows are contract fixtures, not collected chess evidence.
Real collection consumption is exposed separately by load_collected_dataset.
"""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from rz_pals_model.config import ModelConfig, TASKS
from rz_pals_model.training import (BudgetLedger, DivergenceContext, EncodedSnapshot, ResumableSampler,
                                   TaskContext, ValidatedDataset, _fen_board, _canonical, capture_rng,
                                   _label_digest_value, _private_fp32_bits, current_label_view, label_digest,
                                   encoded_from_sidecar, load_collected_dataset, load_frozen_collected_dataset,
                                   load_preparation_checkpoint, masked_losses,
                                   prepare_adamw, restore_rng, save_preparation_checkpoint, seal_snapshot,
                                   validate_recipe)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def move(source, destination, promotion=0):
    return source | destination << 6 | promotion << 12


CPU = sha("fixture-owned-cpu")
SOURCE = {"kind": "own_cpu", "cpu_binary_sha256": CPU,
          "evaluator_configuration_sha256": sha("fixture-evaluator"), "model_weights_sha256": None}
REGISTRY = {"cpu_binary_sha256": [CPU], "input_sources": [SOURCE]}


def fixture(role="proposer", game="fixture-game", *, terminal=False):
    s = {"game_id": game, "opening_id": "opening-" + game, "line_genealogy_id": "line-" + game,
         "position_command": "position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
         "board_fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "actual_history": [],
         "rules_state_sha256": sha("state-" + game), "rules_history_sha256": sha("history-" + game),
         "transposition_sha256": sha("transposition-" + game), "encoding_sha256": sha("fixture-encoding"),
         "source": copy.deepcopy(SOURCE), "frozen_epoch": 0, "input_revision": 1, "capture_sequence": 4,
         "white_to_move": True, "role": role, "legal_moves": [] if terminal else [move(12, 20), move(12, 28)],
         "public_records": [{"observation_sha256": sha("observation-" + game), "situation_revision": 1}]}
    return {"input": {"snapshot": s, "sha256": seal_snapshot(s)}, "future_label": None, "verifier_private": None}


def outcome_label(row, *, policy=True, outcome="draw", ending="repetition"):
    return {"observed_sequence": 8, "provenance": {"source": "actual_game", "result": {
        "game_id": row["input"]["snapshot"]["game_id"], "outcome": outcome, "ending": ending,
        "raw_evidence_sha256": sha("fixture-outcome")}},
        "policy": {"moves": list(row["input"]["snapshot"]["legal_moves"]), "probabilities": [0.25, 0.75]} if policy else None,
        "value_wdl": None if outcome == "unknown" else [0.0, 1.0, 0.0], "white_to_move": True,
        "counterexample": None, "verifier_tasks": None, "supersedes_label_sha256": None}


def encoding(row, *, divergences=(), task_queries=()):
    s = row["input"]["snapshot"]
    return EncodedSnapshot(row["input"]["sha256"], s["encoding_sha256"], _fen_board(s["board_fen"]),
                           tuple([0.0] * 16), tuple((v["observation_sha256"], v["situation_revision"], tuple([0.0] * 16)) for v in s["public_records"]),
                           tuple([0.0] * 16), tuple(divergences), tuple(task_queries))


def label_chain_rows():
    raw = fixture()
    masked, resolved = copy.deepcopy(raw), copy.deepcopy(raw)
    masked["future_label"] = outcome_label(masked, policy=False, outcome="unknown", ending="unresolved")
    resolved["future_label"] = outcome_label(resolved)
    resolved["future_label"].update(observed_sequence=9, supersedes_label_sha256=label_digest(masked))
    return [raw, masked, resolved]


def dataset(rows, encodings=None, assignments=None):
    if encodings is None:
        encodings = {r["input"]["sha256"]: encoding(r) for r in rows}
    split = {"games": assignments or {r["input"]["snapshot"]["game_id"]: "train" for r in rows}}
    return ValidatedDataset(rows, split, REGISTRY, encodings)


def native_sidecar(row):
    s = row["input"]["snapshot"]
    e = encoding(row)
    tensor = {"role": s["role"], "board": list(e.board), "metadata": list(e.metadata),
              "records": [{"record_id": i + 1, "revision": v[1], "critical": False, "features": list(v[2])} for i, v in enumerate(e.public_records)],
              "required_critical_records": [], "candidates": [{"from": v & 63, "to": v >> 6 & 63, "promotion": v >> 12} for v in s["legal_moves"]],
              "divergence_features": [], "query": list(e.query), "situation_revision": s["input_revision"],
              "history_digest": list(bytes.fromhex(s["rules_history_sha256"])), "model_epoch": [0] * 32}
    tensor["metadata"][0] = 1e-6  # Exact lexical tensor bytes survive the adapter.
    result = {"version": "rz-pals-native-input-sidecar/1", "input_sha256": e.input_sha256,
              "encoding_sha256": e.encoding_sha256, "encoder_source_sha256": sha("fixture-encoder-source"),
              "model_epoch_kind": "encoding_only_zero", "canonical_tensor_sha256": sha("fixture-native-key"),
              "tensor_json": json.dumps(tensor, separators=(",", ":")), "tensor_sha256": "", "record_sources": copy.deepcopy(s["public_records"]), "sha256": ""}
    return reseal_sidecar(result)


def reseal_sidecar(value):
    value["tensor_sha256"] = sha(value["tensor_json"])
    names = ("version", "input_sha256", "encoding_sha256", "encoder_source_sha256", "model_epoch_kind", "canonical_tensor_sha256", "tensor_json", "tensor_sha256", "record_sources")
    value["sha256"] = hashlib.sha256(json.dumps([value[n] for n in names], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return value


def write_fixture_collection(directory, row, sidecar, *, version="rz-pals-own-collector/1"):
    files = {"records.jsonl": row, "native-inputs.jsonl": sidecar, "source-registry.jsonl": REGISTRY,
             "split.jsonl": {"games": {row["input"]["snapshot"]["game_id"]: "train"}}}
    artifacts = {}
    for name, value in files.items():
        data = (json.dumps(value, separators=(",", ":")) + "\n").encode()
        (Path(directory) / name).write_bytes(data)
        artifacts[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    receipt = {"version": version, "complete": True, "failure": None, "actual_training_executed": False,
               "external_teacher_used": False, "source": {"encoder_source_sha256": sha("fixture-encoder-source"), "model_epoch_kind": "encoding_only_zero", "model_epoch": [0] * 32},
               "audit": {"records": 1, "canonical_dataset_sha256": sha("fixture-dataset"), "canonical_split_sha256": sha("fixture-split")}, "artifacts": artifacts}
    data = json.dumps(receipt).encode()
    (Path(directory) / "receipt.json").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def frozen_fixture_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def write_frozen_fixture_collection(directory, *, kinds=("cpu",)):
    """Synthetic DG06 artifact contracts; no collected/loaded runtime claim."""
    from rz_pals_model import frozen_producer as metadata
    root = Path(directory)
    rows, sidecars, inputs, lineage, journals, pins, registrations, descriptions, events = [], [], [], [], [], [], [], [], []
    for index, kind in enumerate(kinds):
        producer_id = "fixture-producer-" + str(index)
        row = fixture(game="frozen-fixture-game")
        source = copy.deepcopy(SOURCE)
        epoch, frozen_epoch, native = [0] * 32, 0, None
        configuration = {"value_identity": {"semantics": "bootstrap-material-pst-v1", "weights_sha256": None,
                                             "training": {"kind": "bootstrap"}}}
        if kind == "native":
            weights, config_sha = sha("fixture-native-weights-" + str(index)), sha("fixture-native-config-" + str(index))
            source = {"kind": "own_pals", "model_configuration_sha256": config_sha, "model_weights_sha256": weights}
            epoch, frozen_epoch, configuration = list(bytes.fromhex(weights)), index + 1, {"fixture_model": index + 1}
            graph = {"role": "proposer", "sha256": sha("fixture-graph-" + str(index)), "serialized_bytes": 32}
            export_sha, runtime_sha = sha("fixture-export-" + str(index)), sha("fixture-runtime-" + str(index))
            native_registry = {"version": "rz-pals-native-collection-registry/1", "collector_binary_sha256": CPU,
                               "model_configuration": configuration, "model_configuration_sha256": config_sha,
                               "checkpoint_sha256": weights, "export_manifest_sha256": export_sha,
                               "graphs": [graph], "runtime_sha256": runtime_sha,
                               "encoding_sha256": row["input"]["snapshot"]["encoding_sha256"],
                               "encoder_source_sha256": sha("fixture-encoder-source"), "frozen_epoch": frozen_epoch,
                               "provider": "cpu", "training_state": "untrained"}
            native = {"independent_registry": native_registry,
                      "loaded_source": {"checkpoint_sha256": epoch, "frozen_epoch": frozen_epoch,
                                        "export_manifest_sha256": list(bytes.fromhex(export_sha)),
                                        "model_configuration": configuration, "trained": False,
                                        "execution": {"runtime_sha256": list(bytes.fromhex(runtime_sha)), "provider": "cpu"}},
                      "graphs": [graph], "provider": "cpu", "precision": "fp32", "training_state": "Untrained",
                      "actual_training_executed": False}
        row["input"]["snapshot"].update(source=source, frozen_epoch=frozen_epoch)
        row["input"]["sha256"] = seal_snapshot(row["input"]["snapshot"])
        description = {"mode": "fixture-" + kind, "source": source, "cpu_profile_sha256": SOURCE["evaluator_configuration_sha256"],
                       "implementation_sha256": CPU, "encoding_sha256": row["input"]["snapshot"]["encoding_sha256"],
                       "encoder_source_sha256": sha("fixture-encoder-source"), "configuration": configuration,
                       "model_epoch": epoch, "model_epoch_kind": "frozen_model_epoch" if kind == "native" else "encoding_only_zero",
                       "frozen_epoch": frozen_epoch}
        if native is not None:
            description["native"] = native
        checked_bytes = frozen_fixture_bytes(["rz-pals-collector-checked-source/1", description])
        policy = {"kind": "native_exact", "encoding_sha256": description["encoding_sha256"],
                  "encoder_source_sha256": description["encoder_source_sha256"],
                  "native_model_epoch": {"kind": description["model_epoch_kind"]}}
        if kind == "native":
            policy["native_model_epoch"]["sha256"] = source["model_weights_sha256"]
        registration = {"version": "rz-pals-collector-producer-registration/1", "producer_id": producer_id,
                        "source": source, "frozen_epoch": frozen_epoch, "encoding_policy": policy,
                        "checked_source_sha256": hashlib.sha256(checked_bytes).hexdigest()}
        registration_bytes = frozen_fixture_bytes(registration)
        pin = {"game_id": row["input"]["snapshot"]["game_id"], "producer_id": producer_id,
               "registration_sha256": hashlib.sha256(registration_bytes).hexdigest(),
               "source": source, "frozen_epoch": frozen_epoch, "encoding_policy": policy}
        registration_path = root / ("producer-registration.json" if index == 0 else "fixture-registration-" + str(index) + ".json")
        checked_path = root / ("producer-source.json" if index == 0 else "fixture-source-" + str(index) + ".json")
        registration_path.write_bytes(registration_bytes)
        checked_path.write_bytes(checked_bytes)
        registrations.append({"pin": pin, "registration_path": registration_path, "checked_source_path": checked_path})
        descriptions.append(description)
        pins.append(pin)
        rows.append(row)
        sidecar = native_sidecar(row)
        if kind == "native":
            tensor = json.loads(sidecar["tensor_json"])
            tensor["model_epoch"] = epoch
            sidecar.update(model_epoch_kind="frozen_model_epoch", tensor_json=json.dumps(tensor, separators=(",", ":")))
            reseal_sidecar(sidecar)
        sidecars.append(sidecar)
        inputs.append(row["input"])
        branch = {"input_sha256": row["input"]["sha256"], "game_id": pin["game_id"],
                  "actual_played_history": row["input"]["snapshot"]["actual_history"], "actual_outcome_eligible": True}
        request = [1, index + 1] if kind == "native" else None
        if request is not None:
            branch.update(process_epoch=request[0], request_sequence=request[1])
            events.append({"domain": "rz-pals-native-call-event/1", "game_id": pin["game_id"],
                           "input_sha256": row["input"]["sha256"], "process_epoch": request[0],
                           "request_sequence": request[1], "stage": "prepared"})
        lineage.append(branch)
        body = {"version": "rz-pals-collector-prepared-producer/1", "producer_id": producer_id,
                "registration_sha256": pin["registration_sha256"], "roster_sha256": "",
                "checked_source_sha256": registration["checked_source_sha256"], "game_id": pin["game_id"],
                "input_sha256": row["input"]["sha256"], "capture_sequence": row["input"]["snapshot"]["capture_sequence"],
                "input_json": {}, "tensor_sidecar_json": {}, "lineage_json": {}, "native_request": request,
                "learning_input": True,
                "publication": "seal-before-submit; prepaid-drain-after-search" if kind == "native" else "append-before-analysis; sync-at-receipt-close"}
        for name, value in (("input_json", row["input"]), ("tensor_sidecar_json", sidecar), ("lineage_json", branch)):
            raw = frozen_fixture_bytes(value)
            body[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        journals.append({"prepared": body, "sha256": ""})
    registry = {"cpu_binary_sha256": [CPU], "input_sources": [pin["source"] for pin in pins]}
    roster = metadata.seal_roster({"version": metadata.ROSTER_DOMAIN, "game_producers": pins})
    bindings = []
    for journal, pin in zip(journals, pins):
        body = journal["prepared"]
        body["roster_sha256"] = roster["sha256"]
        journal["sha256"] = hashlib.sha256(frozen_fixture_bytes([body["version"], body])).hexdigest()
        bindings.append({"input_sha256": body["input_sha256"], "game_id": pin["game_id"], "producer_id": pin["producer_id"],
                         "capture_sequence": body["capture_sequence"], "capture_evidence_sha256": journal["sha256"]})
    capture = metadata.seal_capture({"version": metadata.CAPTURE_DOMAIN, "bindings": bindings})
    capture_bytes = frozen_fixture_bytes(capture)
    raw_rows = rows
    split = {"games": {"frozen-fixture-game": "train"}}
    checked = ValidatedDataset(raw_rows, split, registry, {})
    envelope = metadata.seal_envelope({"version": metadata.ENVELOPE_DOMAIN, "roster_sha256": roster["sha256"],
                                      "owned_sources_sha256": metadata.owned_sources_digest(registry),
                                      "raw_dataset_sha256": sha("fixture-raw-rust-audit"), "split_sha256": sha("fixture-split-rust-audit"),
                                      "current_view_sha256": checked.current_view.sha256, "raw_records": len(raw_rows),
                                      "unique_inputs": len(bindings), "capture_sha256": capture["sha256"],
                                      "capture_artifact": {"bytes": len(capture_bytes), "sha256": hashlib.sha256(capture_bytes).hexdigest()}})
    jsonl = {"records.jsonl": raw_rows, "native-inputs.jsonl": sidecars, "source-registry.jsonl": [registry],
             "split.jsonl": [split], "inputs.jsonl": inputs, "input-lineage.jsonl": lineage, "producer-prepared.jsonl": journals}
    if events:
        jsonl["native-events.jsonl"] = events
    artifacts = {}
    for name, values in jsonl.items():
        (root / name).write_bytes(b"".join(frozen_fixture_bytes(value) + b"\n" for value in values))
    producer_audit = {"scope": "metadata_only", "raw_records": len(raw_rows), "unique_inputs": len(bindings),
                      "roster_sha256": roster["sha256"], "envelope_sha256": envelope["sha256"], "capture_sha256": capture["sha256"],
                      "native_exact_metadata_inputs": len(bindings), "requires_derived_adapter": []}
    for name, value in (("producer-roster.json", roster), ("producer-captures.json", capture),
                        ("producer-envelope.json", envelope), ("producer-audit.json", producer_audit)):
        (root / name).write_bytes(frozen_fixture_bytes(value))
    for name in (*jsonl, "producer-registration.json", "producer-source.json", "producer-roster.json",
                 "producer-captures.json", "producer-envelope.json", "producer-audit.json"):
        data = (root / name).read_bytes()
        artifacts[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    receipt = {"version": "rz-pals-own-collector/1", "complete": True, "failure": None, "actual_training_executed": False,
               "external_teacher_used": False, "source": descriptions[0], "artifacts": artifacts,
               "audit": {"records": len(raw_rows), "canonical_dataset_sha256": envelope["envelope"]["raw_dataset_sha256"],
                         "canonical_split_sha256": envelope["envelope"]["split_sha256"]}}
    (root / "receipt.json").write_bytes(frozen_fixture_bytes(receipt))
    return {"expected_receipt_sha256": hashlib.sha256((root / "receipt.json").read_bytes()).hexdigest(),
            "producer_registrations": registrations}


def repin_frozen_fixture_artifact(directory, configuration, name):
    """Re-pin outer bytes to exercise a deeper contract failure in synthetic data."""
    root = Path(directory)
    receipt = json.loads((root / "receipt.json").read_bytes())
    raw = (root / name).read_bytes()
    receipt["artifacts"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    encoded = frozen_fixture_bytes(receipt)
    (root / "receipt.json").write_bytes(encoded)
    configuration["expected_receipt_sha256"] = hashlib.sha256(encoded).hexdigest()


def reseal_frozen_fixture_journals(directory, configuration, journals):
    from rz_pals_model import frozen_producer as metadata
    root = Path(directory)
    evidence = {}
    for journal in journals:
        body = journal["prepared"]
        journal["sha256"] = hashlib.sha256(frozen_fixture_bytes([body["version"], body])).hexdigest()
        evidence[body["input_sha256"]] = journal["sha256"]
    (root / "producer-prepared.jsonl").write_bytes(b"".join(frozen_fixture_bytes(value) + b"\n" for value in journals))
    repin_frozen_fixture_artifact(directory, configuration, "producer-prepared.jsonl")
    capture_path = root / "producer-captures.json"
    capture = json.loads(capture_path.read_bytes())["capture"]
    for binding in capture["bindings"]:
        binding["capture_evidence_sha256"] = evidence[binding["input_sha256"]]
    sealed = metadata.seal_capture(capture)
    capture_path.write_bytes(frozen_fixture_bytes(sealed))
    repin_frozen_fixture_artifact(directory, configuration, "producer-captures.json")
    envelope_path = root / "producer-envelope.json"
    envelope = json.loads(envelope_path.read_bytes())["envelope"]
    envelope.update(capture_sha256=sealed["sha256"],
                    capture_artifact={"bytes": capture_path.stat().st_size, "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest()})
    sealed_envelope = metadata.seal_envelope(envelope)
    envelope_path.write_bytes(frozen_fixture_bytes(sealed_envelope))
    repin_frozen_fixture_artifact(directory, configuration, "producer-envelope.json")
    audit_path = root / "producer-audit.json"
    audit = json.loads(audit_path.read_bytes())
    audit.update(capture_sha256=sealed["sha256"], envelope_sha256=sealed_envelope["sha256"])
    audit_path.write_bytes(frozen_fixture_bytes(audit))
    repin_frozen_fixture_artifact(directory, configuration, "producer-audit.json")


class CollectionReadProbe:
    def __init__(self, stream, reads, before_read=None, max_chunk=None):
        self.stream, self.reads, self.before_read = stream, reads, before_read
        self.max_chunk = max_chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def fileno(self):
        return self.stream.fileno()

    def read(self, size):
        if self.before_read is not None:
            self.before_read()
        data = self.stream.read(size if self.max_chunk is None else min(size, self.max_chunk))
        self.reads.append((size, len(data)))
        return data


def outputs(batch, *, policy=None, extra=None):
    b = batch.inputs.board.shape[0]
    policy = torch.zeros_like(batch.policy, requires_grad=True) if policy is None else policy
    wdl = torch.zeros_like(batch.wdl, requires_grad=True)
    latent = torch.zeros((b, 16, 384), dtype=torch.float32)
    result = (policy, wdl, latent)
    if batch.role != "proposer":
        extra = torch.zeros_like(batch.task if batch.role == "validator" else batch.divergence, requires_grad=True) if extra is None else extra
        result += (extra,)
    return result


def recipe(phase="pc_bootstrap"):
    return {"optimizer": "adamw", "learning_rate": 0.001, "betas": [0.9, 0.99], "epsilon": 1e-8,
            "weight_decay": 0.01, "gradient_accumulation_steps": 2, "phase": phase,
            "role_order": ["proposer", "critic"] + (["verifier"] if phase == "pcv_preparation" else []),
            "verifier_freeze": {"encoder": True, "public_reader": True, "move_embedding": True} if phase == "pcv_preparation" else None,
            "model": {"width": 384, "latent_slots": 16, "recurrent_blocks": 2, "recurrent_iterations": 2,
                      "query_heads": 6, "kv_heads": 2, "head_dimension": 64, "ffn_width": 1024, "fma_flops": 2,
                      "counted_operator_sha256": sha("fixture-declared-cost-not-measured"), "uncounted_operators": ["non-matrix-ops"],
                      "forward_flops_per_sample": 100, "backward_flops_per_sample": 200},
            "budget": {"max_flops": 100000, "max_steps": 10, "max_games": 10, "max_wall_time_ms": 60000,
                       "max_cpu_nodes": 10000, "max_cpu_time_ms": 10000, "max_compute_units_milli": 1000,
                       "max_spend_usd_micros": 1000, "max_output_bytes": 1000000},
            "implementation_sha256": sha("fixture-implementation"), "dataset_sha256": sha("fixture-dataset"), "split_sha256": sha("fixture-split")}


class TinyResponsibilityModel(nn.Module):
    """Small responsibility/resume fixture; not the neural architecture."""
    def __init__(self):
        super().__init__()
        self.config = ModelConfig()
        self.public_encoder = nn.Linear(2, 2)
        self.reader_blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.move_embedding = nn.Linear(2, 2)
        self.experts = nn.ModuleDict((role, nn.Linear(2, 2)) for role in ("proposer", "critic", "validator"))


class TrainingPreparationTests(unittest.TestCase):
    def setUp(self):
        self.rng = capture_rng()
        self.guard = patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer steps are forbidden in preparation tests"))
        self.guard.start()

    def tearDown(self):
        self.guard.stop()
        restore_rng(self.rng)

    def test_inputs_sealed_order_and_missing_feature_fail_closed(self):
        row = fixture()
        valid = dataset([row])
        batch = valid.collate([0], "proposer")
        self.assertEqual(batch.inputs.candidates.tolist(), [[[12, 20, 0], [12, 28, 0]]])
        self.assertFalse(batch.policy_mask.any())
        self.assertFalse(batch.wdl_mask.any())
        with self.assertRaises(ValueError):
            dataset([row], {}).collate([0], "proposer")
        changed = copy.deepcopy(row)
        changed["input"]["snapshot"]["legal_moves"].reverse()
        with self.assertRaises(ValueError):
            dataset([changed])
        malformed = replace(encoding(row), metadata=(1.0,))
        with self.assertRaises(ValueError):
            dataset([row], {row["input"]["sha256"]: malformed}).collate([0], "proposer")

    def test_holdout_leak_and_unregistered_source_rejected(self):
        a, b = fixture(game="a"), fixture(game="b")
        b["input"]["snapshot"]["transposition_sha256"] = a["input"]["snapshot"]["transposition_sha256"]
        b["input"]["sha256"] = seal_snapshot(b["input"]["snapshot"])
        with self.assertRaises(ValueError):
            dataset([a, b], assignments={"a": "train", "b": "holdout"})
        a["input"]["snapshot"]["source"]["cpu_binary_sha256"] = sha("spoofed")
        a["input"]["sha256"] = seal_snapshot(a["input"]["snapshot"])
        with self.assertRaises(ValueError):
            dataset([a])
        b = fixture(game="b")
        with self.assertRaises(ValueError):
            dataset([b], assignments={"b": "holdout"}).collate([0], "proposer")

    def test_actual_unknown_viewpoint_and_strict_boolean_failures(self):
        row = fixture()
        row["future_label"] = outcome_label(row, outcome="unknown", ending="user_stop")
        self.assertFalse(dataset([row]).collate([0], "proposer").wdl_mask.any())
        row["future_label"]["value_wdl"] = [0.0, 1.0, 0.0]
        with self.assertRaises(ValueError):
            dataset([row])
        row["future_label"] = outcome_label(row, outcome="white_win", ending="checkmate")
        row["future_label"]["value_wdl"] = [True, False, False]
        with self.assertRaises(ValueError):
            dataset([row])
        row["future_label"]["value_wdl"] = [0.0, 0.0, 1.0]
        with self.assertRaises(ValueError):
            dataset([row])

    def test_label_digest_shared_utf8_binary64_vector_and_input_binding(self):
        row = fixture()
        label = outcome_label(row)
        label["observed_sequence"] = 9007199254740993
        label["policy"] = {"moves": [1292, 1804], "probabilities": [0.1, 0.9]}
        label["provenance"]["result"].update(game_id="경기", raw_evidence_sha256="b" * 64)
        canonical = '["rz-pals-label/1",{"input_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","label":{"counterexample":null,"observed_sequence":9007199254740993,"policy":{"moves":[1292,1804],"probabilities":["3fb999999999999a","3feccccccccccccd"]},"provenance":{"result":{"ending":"repetition","game_id":"경기","outcome":"draw","raw_evidence_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"source":"actual_game"},"supersedes_label_sha256":null,"value_wdl":["0000000000000000","3ff0000000000000","0000000000000000"],"verifier_tasks":null,"white_to_move":true}}]'
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.assertEqual(_label_digest_value("a" * 64, label), expected)
        label["value_wdl"] = [0, 1, 0]
        reordered = json.loads(json.dumps(label, sort_keys=True, ensure_ascii=False))
        self.assertEqual(_label_digest_value("a" * 64, reordered), expected)
        self.assertNotEqual(_label_digest_value("c" * 64, label), expected)
        label["value_wdl"][0] = -0.0
        self.assertNotEqual(_label_digest_value("a" * 64, label), expected)
        self.assertIsNone(label_digest(row))
        row["future_label"] = outcome_label(row)
        original = label_digest(row)
        row["future_label"]["value_wdl"] = [0, 1, 0]
        self.assertEqual(label_digest(row), original)
        row["future_label"]["value_wdl"][0] = True
        with self.assertRaises(ValueError):
            label_digest(row)

    def test_label_digest_rejects_noninteger_policy_moves(self):
        for kind in ("float", "bool"):
            row = fixture()
            if kind == "bool":
                row["input"]["snapshot"]["legal_moves"] = [move(1, 0)]
                row["input"]["sha256"] = seal_snapshot(row["input"]["snapshot"])
            row["future_label"] = outcome_label(row)
            policy = row["future_label"]["policy"]
            policy["moves"] = [float(v) for v in policy["moves"]] if kind == "float" else [True]
            if kind == "bool":
                policy["probabilities"] = [1.0]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                label_digest(row)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                dataset([row])

    def test_current_label_view_preserves_raw_history_and_only_collates_leaf(self):
        rows = label_chain_rows()
        original = copy.deepcopy(rows)
        admitted = dataset(rows)
        self.assertEqual(len(admitted.records), 3)
        self.assertEqual(admitted.records, original)
        self.assertEqual(admitted.indices("proposer"), [2])
        batch = admitted.collate([2], "proposer")
        self.assertTrue(batch.policy_mask.all())
        self.assertTrue(batch.wdl_mask.all())
        for old_index in (0, 1):
            with self.assertRaisesRegex(ValueError, "historical or superseded"):
                admitted.collate([old_index], "proposer")
        shuffled = dataset([rows[2], rows[0], rows[1]])
        self.assertEqual(shuffled.indices("proposer"), [0])
        self.assertEqual(shuffled.current_view.sha256, admitted.current_view.sha256)
        self.assertEqual(dataset(rows[:1]).indices("proposer"), [0])
        self.assertEqual(dataset(rows[1:2]).indices("proposer"), [0])
        # A new masked whole label does not inherit prior resolved policy/WDL.
        masked_latest = copy.deepcopy(rows[1])
        masked_latest["future_label"].update(observed_sequence=10, supersedes_label_sha256=label_digest(rows[2]))
        latest = dataset(rows + [masked_latest])
        self.assertEqual(latest.indices("proposer"), [3])
        latest_batch = latest.collate([3], "proposer")
        self.assertFalse(latest_batch.policy_mask.any())
        self.assertFalse(latest_batch.wdl_mask.any())

    def test_label_chain_rejects_missing_duplicate_fork_roots_and_foreign_input(self):
        rows = label_chain_rows()
        invalid = [[rows[0], rows[2]], rows + [copy.deepcopy(rows[1])], [rows[0], copy.deepcopy(rows[0])]]
        fork = copy.deepcopy(rows[2])
        fork["future_label"]["observed_sequence"] = 10
        invalid.append(rows + [fork])
        roots = copy.deepcopy(rows)
        roots[2]["future_label"]["supersedes_label_sha256"] = None
        invalid.append(roots)
        foreign = fixture(game="foreign")
        foreign["future_label"] = outcome_label(foreign)
        foreign["future_label"].update(observed_sequence=9, supersedes_label_sha256=label_digest(rows[1]))
        invalid.append([rows[1], foreign])
        for case in invalid:
            with self.subTest(rows=len(case)), self.assertRaises(ValueError):
                dataset(case)
        for sequence in (7, 8):
            invalid_sequence = copy.deepcopy(rows)
            invalid_sequence[2]["future_label"]["observed_sequence"] = sequence
            with self.subTest(sequence=sequence), self.assertRaisesRegex(ValueError, "strictly increase"):
                dataset(invalid_sequence)
        # Synthetic graph IDs exercise cycles; constructing SHA fixed points
        # for self-containing content-addressed predecessor IDs is not claimed.
        cyclic = copy.deepcopy(rows[1:])
        ids = [sha("synthetic-cycle-a"), sha("synthetic-cycle-b")]
        cyclic[0]["future_label"]["supersedes_label_sha256"] = ids[1]
        cyclic[1]["future_label"]["supersedes_label_sha256"] = ids[0]
        with patch("rz_pals_model.training.label_digest", side_effect=ids), self.assertRaisesRegex(ValueError, "cyclic"):
            current_label_view(cyclic)

    def test_current_label_view_rejects_raw_history_mutation_after_admission(self):
        for mutation in ("predecessor", "append", "remove"):
            admitted = dataset(label_chain_rows())
            if mutation == "predecessor":
                admitted.records[1]["future_label"]["observed_sequence"] = 7
            elif mutation == "append":
                admitted.records.append(fixture(game="added"))
            else:
                admitted.records.pop(0)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "raw dataset history"):
                admitted.indices("proposer")
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "raw dataset history"):
                admitted.collate([2], "proposer")

    def test_label_chain_requires_exact_verifier_private_fp32_context(self):
        root = fixture("verifier")
        root["verifier_private"] = {"task_kind": "resume_task", "control_sha256": sha("control"), "private_latent": [0.0]}
        root["future_label"] = outcome_label(root, policy=False, outcome="unknown", ending="unresolved")
        successor = copy.deepcopy(root)
        successor["future_label"].update(observed_sequence=9, supersedes_label_sha256=label_digest(root))
        self.assertEqual(dataset([root, successor]).indices("verifier"), [1])
        for change in ("signed_zero", "control", "task", "absence"):
            changed = copy.deepcopy(successor)
            if change == "signed_zero":
                changed["verifier_private"]["private_latent"][0] = -0.0
            elif change == "control":
                changed["verifier_private"]["control_sha256"] = sha("other-control")
            elif change == "task":
                changed["verifier_private"]["task_kind"] = "defer"
            else:
                changed["verifier_private"] = None
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "different V private contexts"):
                dataset([root, changed])
        private_only = copy.deepcopy(root)
        private_only["future_label"] = None
        before = dataset([private_only]).current_view.sha256
        private_only["verifier_private"]["private_latent"][0] = -0.0
        self.assertNotEqual(before, dataset([private_only]).current_view.sha256)
        private_only["verifier_private"]["private_latent"][0] = 1e100
        with self.assertRaises(ValueError):
            dataset([private_only])

    def test_verifier_private_accepts_rust_fp32_max_shortest_json_and_rejects_overflow(self):
        row = fixture("verifier")
        row["verifier_private"] = {"task_kind": "resume_task", "control_sha256": sha("control"),
                                   "private_latent": [3.4028235e38, -3.4028235e38]}
        self.assertEqual(_private_fp32_bits(3.4028235e38), "7f7fffff")
        self.assertEqual(_private_fp32_bits(-3.4028235e38), "ff7fffff")
        shortest = dataset([row]).current_view.sha256
        row["verifier_private"]["private_latent"] = [float(np.finfo(np.float32).max), -float(np.finfo(np.float32).max)]
        self.assertEqual(dataset([row]).current_view.sha256, shortest)
        for value in (1e100, -1e100, float("inf"), float("nan"), True, 10**400):
            row["verifier_private"]["private_latent"] = [value]
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                dataset([row])

    def test_rules_terminal_checkmate_requires_captured_side_loss_or_mask(self):
        # Reuse rz-position's independently checked black mate and its color
        # exchange plus 180-degree board rotation for the white mate.
        for white_to_move, fen, winner, wrong_winner in (
                (True, "8/8/8/8/8/2k5/1q6/K7 w - - 0 1", "black_win", "white_win"),
                (False, "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", "white_win", "black_win")):
            with self.subTest(white_to_move=white_to_move):
                row = fixture(terminal=True)
                snapshot = row["input"]["snapshot"]
                snapshot.update(board_fen=fen, position_command="position fen " + fen,
                                white_to_move=white_to_move, rules_state_sha256=sha(fen),
                                transposition_sha256=sha(fen))
                row["input"]["sha256"] = seal_snapshot(snapshot)
                label = outcome_label(row, policy=False, outcome=winner, ending="checkmate")
                label["white_to_move"] = white_to_move
                label["provenance"].update(source="rules_terminal", rules_state_sha256=snapshot["rules_state_sha256"])
                row["future_label"] = label
                label["value_wdl"] = [0.0, 0.0, 1.0]
                batch = dataset([row]).collate([0], "proposer")
                self.assertTrue(batch.wdl_mask.all())
                torch.testing.assert_close(batch.wdl, torch.tensor([[0.0, 0.0, 1.0]]))
                label["value_wdl"] = None
                self.assertFalse(dataset([row]).collate([0], "proposer").wdl_mask.any())
                label["value_wdl"] = [1.0, 0.0, 0.0]
                with self.assertRaises(ValueError):
                    dataset([row])
                label["provenance"]["result"]["outcome"] = wrong_winner
                for value in (None, [1.0, 0.0, 0.0]):
                    label["value_wdl"] = value
                    with self.assertRaisesRegex(ValueError, "captured checkmate must defeat the side to move"):
                        dataset([row])

    def test_actual_game_future_checkmate_can_be_won_by_captured_side(self):
        for white_to_move, winner in ((True, "white_win"), (False, "black_win")):
            with self.subTest(white_to_move=white_to_move):
                row = fixture()
                snapshot = row["input"]["snapshot"]
                if not white_to_move:
                    snapshot["board_fen"] = snapshot["board_fen"].replace(" w ", " b ")
                    snapshot["position_command"] = "position fen " + snapshot["board_fen"]
                    snapshot["legal_moves"] = [move(60, 52), move(60, 53)]
                snapshot["white_to_move"] = white_to_move
                row["input"]["sha256"] = seal_snapshot(snapshot)
                label = outcome_label(row, policy=False, outcome=winner, ending="checkmate")
                label.update(white_to_move=white_to_move, value_wdl=[1.0, 0.0, 0.0])
                row["future_label"] = label
                batch = dataset([row]).collate([0], "proposer")
                self.assertTrue(batch.wdl_mask.all())
                torch.testing.assert_close(batch.wdl, torch.tensor([[1.0, 0.0, 0.0]]))

    def test_masked_losses_do_not_dilute_known_row_and_prepare_gradients(self):
        known, unknown = fixture(game="known"), fixture(game="unknown")
        known["future_label"] = outcome_label(known)
        single = dataset([known]).collate([0], "proposer")
        mixed = dataset([known, unknown]).collate([0, 1], "proposer")
        a, b = outputs(single), outputs(mixed)
        la, lb = masked_losses(a, single), masked_losses(b, mixed)
        torch.testing.assert_close(la["total"], lb["total"])
        lb["total"].backward()
        self.assertTrue(torch.all(b[0].grad[1] == 0))
        self.assertTrue(torch.all(b[1].grad[1] == 0))
        self.assertGreater(float(b[0].grad[0].abs().sum()), 0)

    def test_terminal_value_only_and_all_masked_finite_zero(self):
        row = fixture(terminal=True)
        row["future_label"] = outcome_label(row, policy=False)
        batch = dataset([row]).collate([0], "proposer")
        result = masked_losses(outputs(batch), batch)
        self.assertEqual(float(result["policy"]), 0)
        self.assertAlmostEqual(float(result["wdl"]), np.log(3), places=6)
        row["future_label"] = None
        other = fixture(game="other-terminal", terminal=True)
        batch = dataset([row, other]).collate([0, 1], "proposer")
        out = outputs(batch, policy=torch.full_like(batch.policy, 3e38, requires_grad=True))
        total = masked_losses(out, batch)["total"]
        total.backward()
        self.assertEqual(float(total), 0)
        self.assertTrue(torch.all(out[0].grad == 0))

    def test_true_padding_mask_excludes_extreme_inactive_logits(self):
        row = fixture()
        row["input"]["snapshot"]["legal_moves"] = [move(12, 20)]
        row["input"]["sha256"] = seal_snapshot(row["input"]["snapshot"])
        row["future_label"] = outcome_label(row)
        row["future_label"]["policy"]["probabilities"] = [1.0]
        other = fixture(game="other")
        batch = dataset([row, other]).collate([0, 1], "proposer")
        logits = torch.tensor([[-1e10, 0.0], [0.0, 0.0]], requires_grad=True)
        self.assertEqual(float(masked_losses(outputs(batch, policy=logits), batch)["policy"]), 0)
        logits = logits.detach().clone()
        logits[1, 1] = torch.nan
        with self.assertRaises(ValueError):
            masked_losses(outputs(batch, policy=logits), batch)

    def test_critic_conditional_target_mapping_and_unknown_mask(self):
        row = fixture("critic")
        context = DivergenceContext(sha("challenged"), 3, 1)
        label = outcome_label(row)
        label["counterexample"] = {"challenged_line_sha256": context.challenged_line_sha256, "divergence_ply": 3,
                                    "response_line": [move(12, 20)], "repair_line": [move(12, 28)],
                                    "validity": "supported_after_repair", "input_revision": 1,
                                    "legality_evidence_sha256": sha("rules-legal-fixture")}
        row["future_label"] = label
        e = encoding(row, divergences=[(context, tuple([0.0] * 8))])
        for validity, expected in (("supported_after_repair", 1), ("refuted_by_repair", 0), ("disputed", None), ("not_examined", None)):
            row["future_label"]["counterexample"]["validity"] = validity
            batch = dataset([row], {e.input_sha256: e}).collate([0], "critic")
            self.assertEqual(bool(batch.divergence_mask.any()), expected is not None)
            if expected is not None:
                self.assertEqual(float(batch.divergence[0, 0]), expected)
                self.assertAlmostEqual(float(masked_losses(outputs(batch), batch)["divergence"]), np.log(2), places=6)
        row["future_label"]["counterexample"]["validity"] = "supported_after_repair"
        with self.assertRaises(ValueError):
            dataset([row]).collate([0], "critic")
        row["future_label"]["counterexample"]["repair_line"] = None
        with self.assertRaises(ValueError):
            dataset([row])

    def test_direct_critic_target_corruption_rejected(self):
        batch = dataset([fixture("critic")]).collate([0], "critic")
        corrupted = replace(batch, divergence=torch.full_like(batch.divergence, 2.0))
        with self.assertRaises(ValueError):
            masked_losses(outputs(corrupted), corrupted)
        corrupted = replace(batch, wdl=torch.zeros((1, 4), dtype=torch.float32))
        with self.assertRaises(ValueError):
            masked_losses(outputs(corrupted), corrupted)

    def test_v_seven_tasks_grouped_by_full_context_not_rank_index(self):
        row = fixture("verifier")
        a, b = TaskContext(sha("branch-a"), sha("profile"), 1), TaskContext(sha("branch-b"), sha("profile"), 2)
        row["future_label"] = outcome_label(row, policy=False)
        row["future_label"]["verifier_tasks"] = []
        for context, name, rank in ((a, "attack_repair", 1), (b, "defer", 0), (a, "defend_response", 0), (a, "resume_task", None)):
            row["future_label"]["verifier_tasks"].append({"task": name, **context.__dict__, "preference_rank": rank,
                                                        "information_gain_evidence_sha256": sha("gain") if rank is not None else None})
        e = encoding(row, task_queries=[(a, tuple([1.0] * 16)), (b, tuple([2.0] * 16))])
        data = dataset([row], {e.input_sha256: e})
        first = data.collate([0], "verifier", task_contexts=[a])
        second = data.collate([0], "verifier", task_contexts=[b])
        self.assertEqual(first.role, "validator")
        self.assertEqual(first.task_mask.tolist(), [[True, True, False, False, False, False, False]])
        torch.testing.assert_close(first.task[0, :2], torch.softmax(torch.tensor([0.0, -1.0]), 0))
        self.assertEqual(second.task.tolist(), [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertEqual(float(masked_losses(outputs(second), second)["task"]), 0)
        with self.assertRaises(ValueError):
            data.collate([0], "verifier")
        row["future_label"]["verifier_tasks"].append(copy.deepcopy(row["future_label"]["verifier_tasks"][0]))
        with self.assertRaises(ValueError):
            dataset([row], {e.input_sha256: e})

    def test_v_unresolved_has_no_uniform_label_and_private_pc_rejected(self):
        row = fixture("verifier")
        context = TaskContext(sha("branch"), sha("profile"), 0)
        row["future_label"] = outcome_label(row, policy=False)
        row["future_label"]["verifier_tasks"] = [{"task": "defer", **context.__dict__, "preference_rank": None, "information_gain_evidence_sha256": None}]
        e = encoding(row, task_queries=[(context, tuple([0.0] * 16))])
        batch = dataset([row], {e.input_sha256: e}).collate([0], "verifier", task_contexts=[context])
        self.assertFalse(batch.task_mask.any())
        self.assertEqual(float(masked_losses(outputs(batch), batch)["total"]), 0)
        row = fixture()
        row["verifier_private"] = {"task_kind": "defer", "control_sha256": sha("control"), "private_latent": [0.0]}
        with self.assertRaises(ValueError):
            dataset([row])

    def test_optimizer_dedup_private_freeze_and_role_phase_admission(self):
        model = TinyResponsibilityModel()
        # An alias with the same owner is admitted once.
        model.reader_blocks.append(model.reader_blocks[0])
        prepared = prepare_adamw(model, recipe(), "proposer")
        members = [p for g in prepared.optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(members), len({id(p) for p in members}))
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        v = prepare_adamw(model, recipe("pcv_preparation"), "verifier")
        self.assertTrue(all(v.observed_freeze.values()))
        self.assertTrue(all(name.startswith("experts.validator.") for name in v.parameter_names))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(p.requires_grad == name.startswith("experts.validator.") for name, p in model.named_parameters()))
        self.assertEqual(v.optimizer.state, {})
        with self.assertRaises(ValueError):
            prepare_adamw(model, recipe(), "verifier")
        model.unowned = nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError):
            prepare_adamw(model, recipe(), "critic")

    def test_sampler_preserves_next_batch_epoch_and_rejects_rehashed_corruption(self):
        sampler = ResumableSampler(tuple(range(9)), 51, epoch=4, role="critic")
        sampler.next_batch(3)
        restored = ResumableSampler.restore(sampler.state(), indices=tuple(range(9)), role="critic")
        self.assertEqual(sampler.next_batch(4), restored.next_batch(4))
        self.assertEqual(restored.epoch, 4)
        for permutation in ([], [float(v) for v in sampler.permutation], [0] * 9):
            state = sampler.state()
            state["permutation"] = permutation
            state["order_sha256"] = _canonical("rz-pals-sampler-order/1", permutation)
            with self.assertRaises(ValueError):
                ResumableSampler.restore(state, indices=tuple(range(9)), role="critic")

    def test_ledger_projected_cost_overflow_time_cancel_and_zero_steps(self):
        now = [10.0]
        ledger = BudgetLedger(recipe(), clock=lambda: now[0])
        usage = ledger.admit_batch(2, backward=True)
        self.assertEqual((usage["forward_flops"], usage["backward_flops"]), (200, 400))
        ledger.admit_batch(1, validation=True)
        ledger.admit_batch(1, recompute=True)
        self.assertEqual(ledger.usage["steps"], 0)
        for increments in ({"steps": 1}, {"forward_flops": 2**64}, {"output_bytes": 1000001}, {"spend_usd_micros": 1001}):
            with self.assertRaises(ValueError):
                ledger.admit(**increments)
        ledger.cancel()
        with self.assertRaises(ValueError):
            ledger.admit_batch(1)
        now[0] = 71.0
        with self.assertRaises(ValueError):
            ledger.snapshot_usage()

    def test_recipe_all_finite_axes_and_strict_freeze_flags(self):
        value = recipe("pcv_preparation")
        validate_recipe(value)
        value["verifier_freeze"]["encoder"] = 1
        with self.assertRaises(ValueError):
            validate_recipe(value)
        for key in recipe()["budget"]:
            value = recipe()
            value["budget"][key] = 0
            with self.assertRaises(ValueError):
                validate_recipe(value)

    def test_complete_checkpoint_restores_model_rng_optimizer_and_next_batch(self):
        model, spec = TinyResponsibilityModel(), recipe()
        prepared = prepare_adamw(model, spec, "proposer")
        sampler = ResumableSampler(tuple(range(7)), 29, epoch=3)
        sampler.next_batch(2)
        ledger = BudgetLedger(spec)
        ledger.admit_batch(2, backward=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix="pals-preparation-") as temporary:
            receipt = save_preparation_checkpoint(Path(temporary) / "checkpoint.pt", model, prepared, spec, sampler, ledger)
            expected_rng = (random.random(), np.random.random(), torch.rand(3))
            expected_batch = sampler.next_batch(3)
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1)
            random.seed(899)
            np.random.seed(899)
            torch.random.default_generator.manual_seed(899)
            restored, next_sampler, next_ledger = load_preparation_checkpoint(receipt["path"], receipt["sha256"], model, spec,
                                                                             usage_receipt=receipt, indices=tuple(range(7)), role="proposer")
            self.assertEqual(next_sampler.next_batch(3), expected_batch)
            self.assertEqual(next_sampler.epoch, 3)
            self.assertEqual(restored.optimizer.state, {})
            self.assertEqual((random.random(), np.random.random()), expected_rng[:2])
            torch.testing.assert_close(torch.rand(3), expected_rng[2])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name])
            self.assertEqual(next_ledger.usage["output_bytes"], receipt["bytes"])
            self.assertEqual(next_ledger.usage["steps"], 0)
            self.assertFalse(receipt["training_executed"])

    def test_failed_resume_preserves_caller_gradients_trainability_and_rng(self):
        model, spec = TinyResponsibilityModel(), recipe()
        prepared = prepare_adamw(model, spec, "proposer")
        with tempfile.TemporaryDirectory(prefix="pals-preparation-") as temporary:
            path = Path(temporary) / "checkpoint.pt"
            receipt = save_preparation_checkpoint(path, model, prepared, spec, ResumableSampler((0, 1), 3), BudgetLedger(spec))
            payload = torch.load(path, weights_only=True)
            payload["parameter_names"].append("unowned")
            corrupt = Path(temporary) / "corrupt.pt"
            torch.save(payload, corrupt)
            before = {name: (p.requires_grad, torch.ones_like(p)) for name, p in model.named_parameters()}
            for name, p in model.named_parameters():
                p.grad = before[name][1].clone()
            rng = capture_rng()
            digest = hashlib.sha256(corrupt.read_bytes()).hexdigest()
            receipt = {**receipt, "sha256": digest, "bytes": corrupt.stat().st_size}
            with self.assertRaises(ValueError):
                load_preparation_checkpoint(corrupt, digest, model, spec, usage_receipt=receipt, indices=(0, 1), role="proposer")
            for name, p in model.named_parameters():
                self.assertEqual(p.requires_grad, before[name][0])
                torch.testing.assert_close(p.grad, before[name][1])
            torch.testing.assert_close(capture_rng()["torch_cpu"], rng["torch_cpu"])

    def test_checkpoint_refuses_pending_gradient_and_cancellation_persists(self):
        model, spec = TinyResponsibilityModel(), recipe()
        prepared = prepare_adamw(model, spec, "critic")
        sampler = ResumableSampler((0, 1), 1, role="critic")
        ledger = BudgetLedger(spec)
        with tempfile.TemporaryDirectory(prefix="pals-preparation-") as temporary:
            path = Path(temporary) / "checkpoint.pt"
            next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
            with self.assertRaises(ValueError):
                save_preparation_checkpoint(path, model, prepared, spec, sampler, ledger)
            for p in model.parameters():
                p.grad = None
            ledger.cancel()
            receipt = save_preparation_checkpoint(path, model, prepared, spec, sampler, ledger)
            _, _, restored = load_preparation_checkpoint(path, receipt["sha256"], model, spec, usage_receipt=receipt, indices=(0, 1), role="critic")
            with self.assertRaises(ValueError):
                restored.admit_batch(1)

    def test_save_rejects_modified_optimizer_options_without_issuing_artifact(self):
        model, spec = TinyResponsibilityModel(), recipe()
        prepared = prepare_adamw(model, spec, "proposer")
        prepared.optimizer.param_groups[0]["weight_decay"] = 0.5
        with tempfile.TemporaryDirectory(prefix="pals-preparation-") as temporary:
            path = Path(temporary) / "checkpoint.pt"
            with self.assertRaises(ValueError):
                save_preparation_checkpoint(path, model, prepared, spec, ResumableSampler((0, 1), 1), BudgetLedger(spec))
            self.assertFalse(path.exists())

    def test_native_sidecar_exact_bytes_source_and_revision_validation(self):
        row = fixture()
        sidecar = native_sidecar(row)
        actual = encoded_from_sidecar(sidecar, row, expected_encoder_source_sha256=sha("fixture-encoder-source"))
        self.assertAlmostEqual(actual.metadata[0], 1e-6)
        with self.assertRaises(ValueError):
            encoded_from_sidecar(sidecar, row, expected_encoder_source_sha256=sha("wrong-source"))
        tensor = json.loads(sidecar["tensor_json"])
        tensor["situation_revision"] = True
        sidecar["tensor_json"] = json.dumps(tensor)
        reseal_sidecar(sidecar)
        with self.assertRaises(ValueError):
            encoded_from_sidecar(sidecar, row, expected_encoder_source_sha256=sha("fixture-encoder-source"))

    def test_collection_loader_fixed_assets_and_unknown_version_fail_closed(self):
        row, expected_encoder = fixture(), sha("fixture-encoder-source")
        with tempfile.TemporaryDirectory(prefix="pals-collection-fixture-") as temporary:
            expected_receipt = write_fixture_collection(temporary, row, native_sidecar(row))
            loaded = load_collected_dataset(temporary, expected_receipt_sha256=expected_receipt,
                                             expected_encoder_source_sha256=expected_encoder)
            self.assertEqual(loaded.indices("proposer"), [0])
            loaded.collate([0], "proposer").inputs.validate()
            (Path(temporary) / "records.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_collected_dataset(temporary, expected_receipt_sha256=expected_receipt,
                                         expected_encoder_source_sha256=expected_encoder)
            expected_receipt = write_fixture_collection(temporary, row, native_sidecar(row), version="future/99")
            with self.assertRaises(ValueError):
                load_collected_dataset(temporary, expected_receipt_sha256=expected_receipt,
                                         expected_encoder_source_sha256=expected_encoder)

    def test_frozen_loader_cpu_and_native_actual_artifacts_enter_cpu_collate(self):
        for kinds in (("cpu",), ("native",), ("native", "native")):
            with self.subTest(kinds=kinds), tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
                configuration = write_frozen_fixture_collection(temporary, kinds=kinds)
                loaded = load_frozen_collected_dataset(temporary, **configuration)
                self.assertEqual(loaded.indices("proposer"), list(loaded.current_view.current_indices))
                batch = loaded.collate(loaded.indices("proposer"), "proposer")
                self.assertEqual(len(batch.input_sha256), len(kinds))
                self.assertEqual(batch.inputs.board.device.type, "cpu")
                self.assertFalse(batch.wdl_mask.any())
                self.assertFalse(batch.policy_mask.any())
                self.assertEqual(loaded.frozen_admission["current_view_sha256"], loaded.current_view.sha256)
                self.assertEqual(len(loaded.frozen_admission["inputs"]), len(kinds))
                self.assertIn("producer-prepared.jsonl", loaded.frozen_admission["artifacts"])

    def test_frozen_loader_same_game_two_models_use_input_binding_not_turn(self):
        with tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
            configuration = write_frozen_fixture_collection(temporary, kinds=("native", "native"))
            loaded = load_frozen_collected_dataset(temporary, **configuration)
            snapshots = [row["input"]["snapshot"] for row in loaded.records]
            self.assertEqual(snapshots[0]["game_id"], snapshots[1]["game_id"])
            self.assertEqual(snapshots[0]["white_to_move"], snapshots[1]["white_to_move"])
            self.assertNotEqual(snapshots[0]["source"]["model_weights_sha256"], snapshots[1]["source"]["model_weights_sha256"])
            self.assertEqual(len(loaded.collate(loaded.indices("proposer"), "proposer").input_sha256), 2)

    def test_frozen_loader_missing_required_artifacts_never_retries_legacy(self):
        required = ("producer-registration.json", "producer-source.json", "producer-roster.json",
                    "producer-captures.json", "producer-envelope.json", "producer-audit.json",
                    "producer-prepared.jsonl", "input-lineage.jsonl")
        for name in required:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
                configuration = write_frozen_fixture_collection(temporary)
                (Path(temporary) / name).unlink()
                with patch("rz_pals_model.training.load_collected_dataset", side_effect=AssertionError("legacy retry prohibited")):
                    with self.assertRaises((ValueError, OSError)):
                        load_frozen_collected_dataset(temporary, **configuration)

    def test_frozen_loader_receipt_failure_and_required_byte_tamper_fail_closed(self):
        for name in ("records.jsonl", "native-inputs.jsonl", "source-registry.jsonl", "split.jsonl",
                     "producer-registration.json", "producer-source.json", "producer-roster.json",
                     "producer-captures.json", "producer-envelope.json", "producer-audit.json", "producer-prepared.jsonl"):
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
                configuration = write_frozen_fixture_collection(temporary)
                with (Path(temporary) / name).open("ab") as stream:
                    stream.write(b" ")
                with self.assertRaises(ValueError):
                    load_frozen_collected_dataset(temporary, **configuration)
        with tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
            configuration = write_frozen_fixture_collection(temporary)
            path = Path(temporary) / "receipt.json"
            receipt = json.loads(path.read_bytes())
            receipt.update(complete=False, failure={"stage": "prepared"})
            path.write_bytes(frozen_fixture_bytes(receipt))
            configuration["expected_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                load_frozen_collected_dataset(temporary, **configuration)

    def test_frozen_loader_epoch_and_encoder_follow_each_input_producer(self):
        for field in ("model_epoch", "encoder_source_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
                configuration = write_frozen_fixture_collection(temporary, kinds=("native", "native"))
                path = Path(temporary) / "native-inputs.jsonl"
                values = [json.loads(line) for line in path.read_bytes().splitlines()]
                if field == "model_epoch":
                    tensor = json.loads(values[1]["tensor_json"])
                    tensor["model_epoch"] = json.loads(values[0]["tensor_json"])["model_epoch"]
                    values[1]["tensor_json"] = json.dumps(tensor, separators=(",", ":"))
                else:
                    values[1][field] = sha("another-encoder")
                reseal_sidecar(values[1])
                path.write_bytes(b"".join(frozen_fixture_bytes(value) + b"\n" for value in values))
                repin_frozen_fixture_artifact(temporary, configuration, "native-inputs.jsonl")
                # Also rebind exact row bytes in the synthetic journal; the
                # producer epoch/encoder policy must still reject the change.
                journal_path = Path(temporary) / "producer-prepared.jsonl"
                journals = [json.loads(line) for line in journal_path.read_bytes().splitlines()]
                exact = frozen_fixture_bytes(values[1])
                body = journals[1]["prepared"]
                body["tensor_sidecar_json"] = {"bytes": len(exact), "sha256": hashlib.sha256(exact).hexdigest()}
                reseal_frozen_fixture_journals(temporary, configuration, journals)
                with self.assertRaisesRegex(ValueError, "epoch|encoder"):
                    load_frozen_collected_dataset(temporary, **configuration)

    def test_frozen_loader_actual_evidence_and_native_request_attribution(self):
        for field, value in (("checked_source_sha256", sha("wrong-checked-source")),
                             ("native_request", [1, 99]), ("learning_input", False)):
            with self.subTest(field=field), tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
                configuration = write_frozen_fixture_collection(temporary, kinds=("native",))
                path = Path(temporary) / "producer-prepared.jsonl"
                journal = json.loads(path.read_bytes())
                journal["prepared"][field] = value
                reseal_frozen_fixture_journals(temporary, configuration, [journal])
                with self.assertRaises(ValueError):
                    load_frozen_collected_dataset(temporary, **configuration)
        with tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
            configuration = write_frozen_fixture_collection(temporary, kinds=("native",))
            (Path(temporary) / "native-events.jsonl").write_bytes(frozen_fixture_bytes({"stage": "prepared"}) + b"\n")
            repin_frozen_fixture_artifact(temporary, configuration, "native-events.jsonl")
            with self.assertRaisesRegex(ValueError, "actual prepared observer event"):
                load_frozen_collected_dataset(temporary, **configuration)

    def test_frozen_loader_independent_registration_and_source_facts_are_required(self):
        for part in ("registration", "source"):
            with self.subTest(part=part), tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
                configuration = write_frozen_fixture_collection(temporary)
                name = "producer-registration.json" if part == "registration" else "producer-source.json"
                path = Path(temporary) / name
                value = json.loads(path.read_bytes())
                if part == "registration":
                    value["producer_id"] = "unregistered-producer"
                else:
                    value[1]["implementation_sha256"] = sha("another-actual-binary")
                path.write_bytes(frozen_fixture_bytes(value))
                repin_frozen_fixture_artifact(temporary, configuration, name)
                with self.assertRaisesRegex(ValueError, "registration bytes|actual checked source"):
                    load_frozen_collected_dataset(temporary, **configuration)

    def test_frozen_loader_aggregate_read_budget_covers_external_registration_bytes(self):
        with tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
            configuration = write_frozen_fixture_collection(temporary, kinds=("native", "native"))
            total = sum(path.stat().st_size for path in Path(temporary).iterdir() if path.is_file())
            loaded = load_frozen_collected_dataset(temporary, max_input_bytes=total, **configuration)
            self.assertEqual(len(loaded.indices("proposer")), 2)
            with self.assertRaisesRegex(ValueError, "allocation budget"):
                load_frozen_collected_dataset(temporary, max_input_bytes=total - 1, **configuration)

    def test_frozen_loader_per_input_row_byte_cap_precedes_json_parse(self):
        with tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
            configuration = write_frozen_fixture_collection(temporary)
            path = Path(temporary) / "native-inputs.jsonl"
            path.write_bytes(path.read_bytes().rstrip(b"\n") + b" " * (2 * 1024 * 1024 + 1) + b"\n")
            repin_frozen_fixture_artifact(temporary, configuration, "native-inputs.jsonl")
            with self.assertRaisesRegex(ValueError, "JSONL record/line bound"):
                load_frozen_collected_dataset(temporary, **configuration)

    def test_frozen_admission_mutation_is_checked_by_indices_and_collate(self):
        for mutation in ("admission", "source", "receipt", "encoding"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
                configuration = write_frozen_fixture_collection(temporary)
                loaded = load_frozen_collected_dataset(temporary, **configuration)
                selected = loaded.indices("proposer")
                if mutation == "admission":
                    loaded.frozen_admission["metadata_audit"]["capture_sha256"] = sha("tampered-capture")
                elif mutation == "source":
                    loaded.owned_sources["cpu_binary_sha256"] = [sha("changed-authority")]
                elif mutation == "receipt":
                    loaded.collection_receipt["audit"]["records"] += 1
                else:
                    key = next(iter(loaded.encodings))
                    loaded.encodings[key] = replace(loaded.encodings[key], query=tuple([1.0] * 16))
                with self.assertRaises(ValueError):
                    loaded.indices("proposer")
                with self.assertRaises(ValueError):
                    loaded.collate(selected, "proposer")

    def test_frozen_loader_refuses_private_derived_policy_without_adapter(self):
        with tempfile.TemporaryDirectory(prefix="pals-frozen-fixture-") as temporary:
            configuration = write_frozen_fixture_collection(temporary, kinds=("native",))
            configuration["producer_registrations"][0]["pin"]["encoding_policy"] = {
                "kind": "private_checked_derived_query", "encoding_schema_sha256": sha("private-schema"),
                "private_encoder_source_sha256": sha("private-source"), "parent_encoding_sha256": sha("parent-encoding"),
                "parent_encoder_source_sha256": sha("parent-source")}
            with self.assertRaisesRegex(ValueError, "explicit checked derived adapter"):
                load_frozen_collected_dataset(temporary, **configuration)

    def test_collection_loader_reads_only_remaining_total_byte_budget_plus_one(self):
        row, reads = fixture(), []
        original_fdopen = os.fdopen
        with tempfile.TemporaryDirectory(prefix="pals-collection-fixture-") as temporary:
            expected_receipt = write_fixture_collection(temporary, row, native_sidecar(row))
            names = ("receipt.json", "records.jsonl", "native-inputs.jsonl", "source-registry.jsonl", "split.jsonl")
            total = sum((Path(temporary) / name).stat().st_size for name in names)
            with patch("rz_pals_model.training.os.fdopen", side_effect=lambda fd, mode, **options: CollectionReadProbe(original_fdopen(fd, mode, **options), reads)):
                loaded = load_collected_dataset(temporary, expected_receipt_sha256=expected_receipt,
                                                expected_encoder_source_sha256=sha("fixture-encoder-source"),
                                                max_input_bytes=total)
            self.assertEqual(loaded.indices("proposer"), [0])
            self.assertGreaterEqual(len(reads), len(names))
            remaining = total
            for cap, actual in reads:
                self.assertLessEqual(cap, 65536)
                self.assertLessEqual(cap, remaining + 1)
                self.assertLessEqual(actual, cap)
                self.assertLessEqual(actual, remaining)
                remaining -= actual
            self.assertEqual(remaining, 0)

    def test_collection_loader_stable_short_reads_preserve_bounded_admission(self):
        row, reads = fixture(), []
        original_fdopen = os.fdopen
        with tempfile.TemporaryDirectory(prefix="pals-collection-fixture-") as temporary:
            expected_receipt = write_fixture_collection(temporary, row, native_sidecar(row))
            names = ("receipt.json", "records.jsonl", "native-inputs.jsonl", "source-registry.jsonl", "split.jsonl")
            total = sum((Path(temporary) / name).stat().st_size for name in names)
            with patch("rz_pals_model.training.os.fdopen", side_effect=lambda fd, mode, **options: CollectionReadProbe(original_fdopen(fd, mode, **options), reads, max_chunk=17)):
                loaded = load_collected_dataset(temporary, expected_receipt_sha256=expected_receipt,
                                                expected_encoder_source_sha256=sha("fixture-encoder-source"),
                                                max_input_bytes=total)
            self.assertEqual(loaded.indices("proposer"), [0])
            self.assertEqual(sum(actual for _, actual in reads), total)
            self.assertTrue(any(0 < actual < cap for cap, actual in reads))
            remaining = total
            for cap, actual in reads:
                self.assertLessEqual(cap, min(65536, remaining + 1))
                self.assertLessEqual(actual, min(17, cap))
                remaining -= actual
            self.assertEqual(remaining, 0)

    def test_collection_loader_growth_during_read_is_bounded_before_rejection(self):
        row, reads = fixture(), []
        original_fdopen = os.fdopen
        with tempfile.TemporaryDirectory(prefix="pals-collection-fixture-") as temporary:
            expected_receipt = write_fixture_collection(temporary, row, native_sidecar(row))
            path = Path(temporary) / "receipt.json"
            maximum = path.stat().st_size
            def grow():
                with path.open("ab") as stream:
                    stream.write(b" " * 4096)
            with patch("rz_pals_model.training.os.fdopen", side_effect=lambda fd, mode, **options: CollectionReadProbe(original_fdopen(fd, mode, **options), reads, grow)):
                with self.assertRaisesRegex(ValueError, "changed or exceeded allocation budget"):
                    load_collected_dataset(temporary, expected_receipt_sha256=expected_receipt,
                                           expected_encoder_source_sha256=sha("fixture-encoder-source"),
                                           max_input_bytes=maximum)
            self.assertEqual(reads, [(maximum + 1, maximum + 1)])

    def test_collection_loader_rejects_initial_oversize_and_nonregular_without_open(self):
        row = fixture()
        with tempfile.TemporaryDirectory(prefix="pals-collection-fixture-") as temporary:
            expected_receipt = write_fixture_collection(temporary, row, native_sidecar(row))
            path = Path(temporary) / "receipt.json"
            with patch("rz_pals_model.training.os.open") as opened:
                with self.assertRaisesRegex(ValueError, "allocation budget exceeded"):
                    load_collected_dataset(temporary, expected_receipt_sha256=expected_receipt,
                                           expected_encoder_source_sha256=sha("fixture-encoder-source"),
                                           max_input_bytes=path.stat().st_size - 1)
                opened.assert_not_called()
            path.unlink()
            path.mkdir()
            with patch("rz_pals_model.training.os.open") as opened:
                with self.assertRaisesRegex(ValueError, "regular file"):
                    load_collected_dataset(temporary, expected_receipt_sha256=expected_receipt,
                                           expected_encoder_source_sha256=sha("fixture-encoder-source"))
                opened.assert_not_called()

    def test_collection_loader_rejects_same_size_path_replacement_before_open(self):
        row, reads = fixture(), []
        original_open, original_fdopen = os.open, os.fdopen
        with tempfile.TemporaryDirectory(prefix="pals-collection-fixture-") as temporary:
            expected_receipt = write_fixture_collection(temporary, row, native_sidecar(row))
            path = Path(temporary) / "receipt.json"
            replacement = Path(temporary) / "replacement.json"
            replacement.write_bytes(path.read_bytes())
            def replace_before_open(opened_path, flags):
                os.replace(replacement, path)
                return original_open(opened_path, flags)
            with patch("rz_pals_model.training.os.open", side_effect=replace_before_open), patch("rz_pals_model.training.os.fdopen", side_effect=lambda fd, mode, **options: CollectionReadProbe(original_fdopen(fd, mode, **options), reads)):
                with self.assertRaisesRegex(ValueError, "changed before bounded read"):
                    load_collected_dataset(temporary, expected_receipt_sha256=expected_receipt,
                                           expected_encoder_source_sha256=sha("fixture-encoder-source"))
            self.assertEqual(reads, [])


if __name__ == "__main__":
    unittest.main()
