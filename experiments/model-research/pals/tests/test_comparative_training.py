"""Synthetic receipt/mask/numeric wiring, not observed Rust/checker proof.

No child, chess game, real checkpoint/model, optimizer or training executes in
these fixtures. The independent pins/launch assurances below are deliberately
synthetic; passing them proves only the adapter's conditional contract checks.
"""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rz_pals_model import comparative as metadata
from rz_pals_model import comparative_training as candidate
from rz_pals_model import frozen_producer as frozen
from rz_pals_model.preparation_check import _parameter_digest
from rz_pals_model.training import (label_digest, load_frozen_collected_dataset,
                                   ValidatedDataset)
from test_current_consumers import FrozenNumericFixture
from test_training import (fixture, frozen_fixture_bytes as raw, sha,
                           outcome_label, repin_frozen_fixture_artifact,
                           write_frozen_fixture_collection)


class CandidateFixture:
    def __init__(self, root, *, roles=("proposer", "critic"), history=False):
        self.root = Path(root)
        iterator = iter(roles)

        def role_row(*args, **kwargs):
            return fixture(next(iterator), game=kwargs["game"])

        with patch("test_training.fixture", side_effect=role_row):
            self.parent_options = write_frozen_fixture_collection(root, kinds=("cpu",) * len(roles))
        if history:
            rows = [json.loads(line) for line in (self.root / "records.jsonl").read_bytes().splitlines()]
            labeled = copy.deepcopy(rows[0])
            labeled["future_label"] = outcome_label(labeled, policy=False, outcome="unknown", ending="unresolved")
            rows.insert(0, labeled)  # Deliberately nonchronological raw order.
            (self.root / "records.jsonl").write_bytes(b"".join(raw(row) + b"\n" for row in rows))
            registry = json.loads((self.root / "source-registry.jsonl").read_bytes())
            split = json.loads((self.root / "split.jsonl").read_bytes())
            view = ValidatedDataset(rows, split, registry, {}).current_view
            repin_frozen_fixture_artifact(root, self.parent_options, "records.jsonl")
            envelope = json.loads((self.root / "producer-envelope.json").read_bytes())["envelope"]
            envelope.update(raw_records=len(rows), current_view_sha256=view.sha256)
            sealed = frozen.seal_envelope(envelope)
            (self.root / "producer-envelope.json").write_bytes(raw(sealed))
            repin_frozen_fixture_artifact(root, self.parent_options, "producer-envelope.json")
            audit = json.loads((self.root / "producer-audit.json").read_bytes())
            audit.update(raw_records=len(rows), envelope_sha256=sealed["sha256"])
            (self.root / "producer-audit.json").write_bytes(raw(audit))
            repin_frozen_fixture_artifact(root, self.parent_options, "producer-audit.json")
            receipt = json.loads((self.root / "receipt.json").read_bytes())
            receipt["audit"]["records"] = len(rows)
            (self.root / "receipt.json").write_bytes(raw(receipt))
            self.parent_options["expected_receipt_sha256"] = candidate.byte_pin(raw(receipt))["sha256"]
        self.parents = load_frozen_collected_dataset(root, **self.parent_options)
        names = ("receipt.json", "producer-roster.json", "producer-envelope.json", "producer-prepared.jsonl",
                 "inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl")
        self.parent_artifacts = {name: (self.root / name).read_bytes() for name in names}
        admission = self.parents.frozen_admission
        parent = {"receipt_artifact": admission["receipt"], "raw_dataset_sha256": admission["raw_dataset_sha256"],
                  "split_sha256": admission["split_sha256"], "current_view_sha256": self.parents.current_view.sha256,
                  "producer_roster_sha256": json.loads(self.parent_artifacts["producer-roster.json"])["sha256"],
                  "producer_envelope_sha256": json.loads(self.parent_artifacts["producer-envelope.json"])["sha256"]}
        self.binary = b"synthetic fixture binary image; not a launched executable"
        profile = candidate.profile_description(2, 16, 4)
        facts = {"schema": candidate.SOURCE_SCHEMA, "profile": profile, "profile_sha256": candidate.wire_digest(profile),
                 "search_version": candidate.SEARCH_VERSION, "search_conditions": candidate.search_conditions(2, 16, 4),
                 "value_identity": {"semantics": candidate.SEMANTICS, "weights_sha256": None, "training": {"kind": "bootstrap"}},
                 "search_implementation_sha256": sha("synthetic independently registered source"),
                 "value_semantics_sha256": sha("synthetic registered semantics")}
        checker_source = {"kind": "own_cpu", "cpu_binary_sha256": candidate.byte_pin(self.binary)["sha256"],
                          "evaluator_configuration_sha256": candidate.wire_digest([candidate.SOURCE_SCHEMA, facts]),
                          "model_weights_sha256": None}
        self.source = raw({**facts, "source": checker_source})
        checker = {"source": checker_source, "search_implementation_sha256": facts["search_implementation_sha256"],
                   "value_semantics_sha256": facts["value_semantics_sha256"], "profile": candidate.PROFILE,
                   "search_conditions": facts["search_conditions"], "horizon": 2, "node_budget": 100000,
                   "perspective": "captured_side_to_move"}
        self.registration = raw({"schema": candidate.REGISTRATION_SCHEMA, "checker": checker,
                                 "source_artifact": candidate.byte_pin(self.source), "binary_artifact": candidate.byte_pin(self.binary),
                                 "profile_sha256": facts["profile_sha256"], "tt_entries": 16, "quiescence_ply": 4,
                                 "platform": "windows", "binary_pin_scope": "current_exe_path_hash"})
        checker["registration_sha256"] = candidate.byte_pin(self.registration)["sha256"]
        self.criterion = raw({"schema": candidate.CRITERION_SCHEMA, "recipe_id": "후보 ordinal 비교",
                              "method": "finite_depth_raw_ordinal", "minimum_margin": 1, "score_limit": 20000,
                              "mate_threshold": 29000, "score_scope": "completed_iteration", "completion": "depth_limit",
                              "perspective": "captured_side_to_move"})
        prepared, pairs = [], []
        journals = {journal["prepared"]["input_sha256"]: journal for journal in
                    (json.loads(line) for line in self.parent_artifacts["producer-prepared.jsonl"].splitlines())}
        for index in self.parents.current_view.current_indices:
            row, snapshot = self.parents.records[index], self.parents.records[index]["input"]["snapshot"]
            identity, journal = row["input"]["sha256"], journals[row["input"]["sha256"]]
            current = {"input_sha256": identity, "label_sha256": label_digest(row),
                       **{name: snapshot[name] for name in ("game_id", "role", "rules_state_sha256", "rules_history_sha256",
                          "encoding_sha256", "source", "frozen_epoch", "input_revision", "legal_moves")},
                       "side_to_move": "white" if snapshot["white_to_move"] else "black"}
            prepared.append({"current": current, "producer_id": journal["prepared"]["producer_id"],
                             "producer_registration_sha256": journal["prepared"]["registration_sha256"],
                             "capture_evidence_sha256": journal["sha256"],
                             **{name: journal["prepared"][name] for name in ("input_json", "tensor_sidecar_json", "lineage_json")}})
            checks = [{"candidate": move, "task_id": f"{snapshot['role']}-{slot}",
                       "conditions": metadata.task_conditions(current, checker), "restriction": {"kind": "candidate_only", "root_moves": [move]},
                       "completion": "completed", "completed_depth": 2, "evidence_id": f"{snapshot['role']}-{slot}",
                       "evidence_artifact": {"bytes": 1, "sha256": sha("placeholder")}} for slot, move in enumerate(snapshot["legal_moves"][:2])]
            pairs.append({"pair_id": snapshot["role"], "input_sha256": identity,
                          "kind": "proposer_candidates" if snapshot["role"] == "proposer" else "critic_responses",
                          "candidates": snapshot["legal_moves"][:2], "checks": checks, "preference": "left"})
        self.body = metadata.normalize_overlay({"version": metadata.COMPARATIVE_DOMAIN, "parent": parent,
                                               "criterion": {"recipe_id": "후보 ordinal 비교", "recipe_sha256": candidate.byte_pin(self.criterion)["sha256"]},
                                               "checker": checker, "prepared_inputs": prepared, "pairs": pairs})
        self.plan = raw({"schema": candidate.PLAN_SCHEMA, **{name: self.body[name] for name in ("parent", "criterion", "prepared_inputs")},
                         "checker_namespace_sha256": metadata.checker_namespace(checker),
                         "pairs": [{name: pair[name] for name in ("pair_id", "input_sha256", "kind", "candidates")}
                                   | {"task_ids": [check["task_id"] for check in pair["checks"]]} for pair in self.body["pairs"]]})
        self.executions, self.execution_pins = {}, {}
        for pair in self.body["pairs"]:
            row = next(row for row in self.parents.records if row["input"]["sha256"] == pair["input_sha256"])
            snapshot = row["input"]["snapshot"]
            for slot, check in enumerate(pair["checks"]):
                task, move = check["task_id"], check["candidate"]
                request = {"schema": candidate.REQUEST_SCHEMA, "task_id": task, "captured_input_sha256": pair["input_sha256"],
                           "checker_namespace_sha256": metadata.checker_namespace(checker),
                           "before_result_anchor_sha256": candidate.byte_pin(self.plan)["sha256"],
                           **{name: snapshot[name] for name in ("frozen_epoch", "input_revision", "position_command", "rules_state_sha256",
                              "rules_history_sha256", "white_to_move")}, "expected_board_fen": snapshot["board_fen"],
                           "expected_legal_moves": snapshot["legal_moves"], "candidate": move,
                           "cpu_binary_sha256": checker_source["cpu_binary_sha256"], "cpu_profile_sha256": facts["profile_sha256"],
                           "horizon": 2, "node_budget": 100000, "tt_entries": 16, "quiescence_ply": 4,
                           "max_wall_time_ms": 10000, "max_output_bytes": 65536}
                request["context_sha256"] = candidate.wire_digest([candidate.REQUEST_SCHEMA, request])
                conditions = {"schema": candidate.CONDITIONS_SCHEMA,
                              **{name: request[name] for name in ("rules_state_sha256", "rules_history_sha256", "white_to_move", "horizon",
                                 "node_budget", "tt_entries", "quiescence_ply")}, "board_fen": snapshot["board_fen"],
                              "legal_moves": snapshot["legal_moves"], "legal_order_sha256": candidate.wire_digest([candidate.LEGAL_DOMAIN, snapshot["legal_moves"]]),
                              "history_completeness": "unknown_prefix", "profile_sha256": facts["profile_sha256"], "profile": candidate.PROFILE,
                              "value_identity": facts["value_identity"], "search_version": candidate.SEARCH_VERSION,
                              "search_conditions": facts["search_conditions"], "root_moves": [move], "restriction": "candidate_only",
                              "perspective": "captured_side_to_move", "resource_policy": {"max_wall_time_ms": 10000, "max_checks": 1,
                                                                                        "search_deadline_reserve_ms": 1000}}
                report = {"conditions_sha256": candidate.wire_digest(conditions), "conditions": conditions,
                          "raw_score": 90 if slot == 0 else 20, "best_move": move, "pv": [move],
                          "score_scope": "completed_iteration", "completion": "depth_limit", "terminal_reason": None,
                          "completed_depth": 2, "requested_depth": 2, "nodes": 42, "quiescence_nodes": 8, "tt_hits": 2,
                          "elapsed_ms": 5, "reused_completed_depth": 0, "root_restricted": True,
                          "score_provenance": candidate.SEMANTICS, "pv_rules_validated": True}
                receipt = {name: request[name] for name in ("schema", "task_id", "captured_input_sha256", "checker_namespace_sha256",
                           "before_result_anchor_sha256", "frozen_epoch", "input_revision", "context_sha256", "cpu_binary_sha256")}
                receipt.update(binary_pin_scope="current_exe_path_hash", caller_registration_scope="declared_not_independently_verified",
                               status="completed", cpu_calls=1, fresh_engine=True, elapsed_ms=8, deadline_exceeded=False,
                               product_verifier_enabled=False, report=report)
                self.executions[task] = {"request": raw(request), "receipt": raw(receipt), "stderr": b"", "launch_observation": b""}
                self.observe(task)
        self.refresh()

    def observe(self, task, **changes):
        execution = self.executions[task]
        pins = {name: None if execution[name] is None else candidate.byte_pin(execution[name]) for name in ("request", "receipt", "stderr")}
        observation = {"schema": candidate.LAUNCH_SCHEMA, "task_id": task,
                       "registration_sha256": candidate.byte_pin(self.registration)["sha256"],
                       "binary_sha256": candidate.byte_pin(self.binary)["sha256"], "platform": "windows", "binary_pin_scope": "current_exe_path_hash",
                       "before_result_anchor_sha256": candidate.byte_pin(self.plan)["sha256"], **pins,
                       "assurance_scope": "independently_pinned_caller_observation", "anchor_durable_before_spawn": True,
                       "criterion_fixed_before_spawn": True, "spawned": True, "reaped": True, "exit_code": 0, "elapsed_ms": 12, "timed_out": False}
        observation.update(changes)
        execution["launch_observation"] = raw(observation)
        self.execution_pins[task] = {**pins, "launch_observation": candidate.byte_pin(execution["launch_observation"])}
        for pair in self.body["pairs"]:
            for check in pair["checks"]:
                if check["task_id"] == task:
                    check["evidence_artifact"] = pins["receipt"]
                    if pins["receipt"] is None:
                        check.update(evidence_id=None, completion="missing", completed_depth=0)

    def refresh(self):
        self.overlay = raw(metadata.seal_overlay(self.body))
        self.expected_metadata = {name: copy.deepcopy(self.body[name]) for name in ("parent", "criterion", "checker", "prepared_inputs")}
        self.pins = {"overlay": candidate.byte_pin(self.overlay), "criterion": candidate.byte_pin(self.criterion),
                     "registration": candidate.byte_pin(self.registration), "source": candidate.byte_pin(self.source),
                     "binary": candidate.byte_pin(self.binary), "plan": candidate.byte_pin(self.plan),
                     "executions": copy.deepcopy(self.execution_pins)}

    def arguments(self):
        return {"overlay_bytes": self.overlay, "expected_metadata_pins": self.expected_metadata,
                "criterion_bytes": self.criterion, "registration_bytes": self.registration, "source_bytes": self.source,
                "binary_bytes": self.binary, "plan_bytes": self.plan, "executions": self.executions, "independent_pins": self.pins}

    def admit(self):
        return candidate.admit_candidate_pairs(parents=self.parents, parent_artifacts=self.parent_artifacts, **self.arguments())

    def persist(self):
        """Explicit synthetic bank buffers; this creates no execution evidence."""
        directory = self.root / "candidate-fixture-bank"
        directory.mkdir()
        assets = {"overlay": self.overlay, "criterion": self.criterion, "registration": self.registration,
                  "source": self.source, "binary": self.binary, "plan": self.plan}
        manifest = {"schema": candidate.BANK_SCHEMA, "files": {}, "executions": {}}
        for name, value in assets.items():
            filename = name + ".fixture"
            manifest["files"][name] = filename
            (directory / filename).write_bytes(value)
        for task, buffers in self.executions.items():
            manifest["executions"][task] = {}
            for name, value in buffers.items():
                filename = task + "-" + name + ".fixture" if value is not None else None
                manifest["executions"][task][name] = filename
                if filename is not None:
                    (directory / filename).write_bytes(value)
        encoded = raw(manifest)
        (directory / "candidate-bank.json").write_bytes(encoded)
        return directory, candidate.byte_pin(encoded)["sha256"]

    def change_receipt(self, task, mutate):
        value = json.loads(self.executions[task]["receipt"])
        mutate(value)
        value["report"]["conditions_sha256"] = candidate.wire_digest(value["report"]["conditions"])
        self.executions[task]["receipt"] = raw(value)
        self.observe(task)
        self.refresh()


class ComparativeReceiptTests(unittest.TestCase):
    def test_wire_canonical_preserves_bool_signed_units_and_u64_without_float(self):
        value = {"recipe": "후보", "epoch": 9007199254740993, "raw_score": -20, "fresh": True}
        self.assertEqual(candidate.canonical_wire(value), '{"epoch":9007199254740993,"fresh":true,"raw_score":-20,"recipe":"후보"}'.encode())
        for bad in (1.0, float("inf"), 2**64, -(2**63)-1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                candidate.canonical_wire(bad)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            candidate._json(b'{"schema":"one","schema":"two"}')

    def test_saved_reloaded_positive_pc_wiring_keeps_raw_history_and_metadata_only(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root, history=True)
            checked = fixture.admit()
            self.assertEqual(len(checked.parents.records), 3)
            self.assertEqual(len(checked.parents.current_view.current_indices), 2)
            self.assertEqual({target.role for target in checked.targets}, {"proposer", "critic"})
            self.assertTrue(all(target.mask and target.sign == 1 for target in checked.targets))
            self.assertEqual(checked.admission["metadata_audit"]["scope"], "metadata_only")
            self.assertIs(checked.admission["metadata_audit"]["requires_actual_checker_admission"], True)
            self.assertFalse(checked.admission["actual_training_executed"])
            self.assertFalse(checked.admission["critic_repair_validity_admitted"])
            # Actual serialized bank buffers are read again; no generated target
            # file or generic metadata acceptance substitutes for these bytes.
            bank_directory, bank_sha = fixture.persist()
            reloaded = candidate.load_persisted_comparative_pairs(root, bank_directory, expected_bank_sha256=bank_sha,
                                                                  strict_parent_options=fixture.parent_options,
                                                                  expected_metadata_pins=fixture.expected_metadata, independent_pins=fixture.pins)
            self.assertEqual(checked._identity, reloaded._identity)
            for pair in fixture.body["pairs"]:
                requests = [json.loads(fixture.executions[check["task_id"]]["request"]) for check in pair["checks"]]
                self.assertNotEqual(requests[0]["context_sha256"], requests[1]["context_sha256"])
                self.assertNotEqual(requests[0]["candidate"], requests[1]["candidate"])
                self.assertEqual(requests[0]["expected_legal_moves"], requests[1]["expected_legal_moves"])
            model = FrozenNumericFixture().eval()
            model.requires_grad_(False)
            before = _parameter_digest(model)
            with patch("torch.optim.AdamW", side_effect=AssertionError("optimizer forbidden")), patch.object(
                    torch.Tensor, "backward", side_effect=AssertionError("backward forbidden")):
                result = candidate.frozen_pairwise_preparation(model, reloaded, expected_parameter_sha256=before,
                                                               checkpoint_sha256=sha("synthetic checkpoint pin"))
            self.assertEqual(result["known_pairs"], {"proposer": 1, "critic": 1})
            for value in result["pairwise_losses"].values():
                self.assertAlmostEqual(value, math.log(2), places=6)
            self.assertEqual(result["parameter_sha256_after"], before)
            self.assertEqual(result["consumed_pair_ids"], ["proposer", "critic"])
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            self.assertTrue(all(row["future_label"] is None or row["future_label"]["policy"] is None for row in reloaded.parents.records))

    def test_actual_byte_and_independent_pre_result_pin_mismatches_reject(self):
        for asset in ("binary", "criterion", "registration", "source", "plan"):
            with self.subTest(asset=asset), tempfile.TemporaryDirectory() as root:
                fixture = CandidateFixture(root)
                arguments = fixture.arguments()
                arguments[asset + "_bytes"] += b" "
                with self.assertRaisesRegex(ValueError, "independent pin"):
                    candidate.admit_candidate_pairs(parents=fixture.parents, parent_artifacts=fixture.parent_artifacts, **arguments)
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root)
            fixture.observe("proposer-0", anchor_durable_before_spawn=False)
            fixture.refresh()
            with self.assertRaisesRegex(ValueError, "pre-result"):
                fixture.admit()

    def test_namespace_epoch_profile_pv_nodes_and_nested_bool_mismatches_reject(self):
        mutations = [lambda value: value.update(checker_namespace_sha256=sha("other")),
                     lambda value: value.update(input_revision=value["input_revision"] + 1),
                     lambda value: value["report"]["conditions"].update(profile="other-profile"),
                     lambda value: value["report"].update(pv_rules_validated=False),
                     lambda value: value["report"].update(pv=[1804]),
                     lambda value: value["report"].update(nodes=100001),
                     lambda value: value["report"]["conditions"]["resource_policy"].update(max_checks=True),
                     lambda value: value["report"].update(score_provenance="other-score"),
                     lambda value: value["report"].update(completed_depth=1)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as root:
                fixture = CandidateFixture(root)
                fixture.change_receipt("proposer-0", mutate)
                with self.assertRaises(ValueError):
                    fixture.admit()

    def test_total_cli_elapsed_must_fit_independent_caller_launch_elapsed(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root, roles=("proposer",))
            # Re-pin every actual serialized buffer so only total elapsed is
            # inconsistent: original wall=10000, receipt=9999, search=5,
            # independently declared caller launch=12. Search-only comparison
            # would accept this synthetic positive receipt incorrectly.
            fixture.change_receipt("proposer-0", lambda value: value.update(elapsed_ms=9999))
            with self.assertRaisesRegex(ValueError, "total CLI receipt work"):
                fixture.admit()
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root, roles=("proposer",))
            fixture.change_receipt("proposer-0", lambda value: value.update(elapsed_ms=12))
            self.assertTrue(fixture.admit().targets[0].mask)  # Inclusive bound.

    def test_partial_canceled_missing_failed_terminal_mate_and_tie_stay_masked(self):
        for mode in ("partial", "canceled", "missing", "failed", "terminal", "mate", "tie"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                fixture = CandidateFixture(root, roles=("proposer",))
                pair = fixture.body["pairs"][0]
                pair["preference"] = "masked"
                if mode == "missing":
                    fixture.executions["proposer-0"]["receipt"] = None
                    fixture.observe("proposer-0")
                    fixture.refresh()
                elif mode == "failed":
                    fixture.executions["proposer-0"]["receipt"] = b'{"failed":"raw unknown work"}'
                    fixture.executions["proposer-0"]["stderr"] = b'{"known_nodes":42}'
                    fixture.observe("proposer-0", exit_code=1)
                    pair["checks"][0].update(completion="unknown", completed_depth=0)
                    fixture.refresh()
                else:
                    def change(value):
                        report = value["report"]
                        if mode in ("partial", "canceled"):
                            value["status"] = mode
                            report.update(completion="node_limit" if mode == "partial" else "canceled", completed_depth=1)
                            pair["checks"][0].update(completion="partial" if mode == "partial" else "cancelled", completed_depth=1)
                        elif mode == "terminal":
                            report.update(completion="rules_terminal", score_scope="rules_terminal", terminal_reason="Checkmate")
                        elif mode == "mate":
                            report["raw_score"] = 29000
                        else:
                            report["raw_score"] = 20
                    fixture.change_receipt("proposer-0", change)
                checked = fixture.admit()
                self.assertFalse(checked.targets[0].mask)
                self.assertEqual(checked.targets[0].sign, 0)
                self.assertTrue(checked.admission["preserved_executions"])
                self.assertEqual(checked.raw_executions(), fixture.executions)
                batch = checked.collate([0], role="proposer")
                self.assertEqual(float(candidate.pairwise_softplus_loss(torch.zeros((1, 2)), batch)), 0)
                model = FrozenNumericFixture().eval().requires_grad_(False)
                with self.assertRaisesRegex(ValueError, "all-masked"):
                    candidate.frozen_pairwise_preparation(model, checked, expected_parameter_sha256=_parameter_digest(model),
                                                           checkpoint_sha256=sha("fixture checkpoint"), required_roles=("proposer",))

    def test_weaker_platform_guarantee_and_unregistered_receipts_do_not_promote(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root)
            fixture.change_receipt("proposer-0", lambda value: value.update(binary_pin_scope="linux_loaded_executable_inode"))
            with self.assertRaisesRegex(ValueError, "binary"):
                fixture.admit()
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root)
            fixture.observe("proposer-0", assurance_scope="self_reported")
            fixture.refresh()
            with self.assertRaisesRegex(ValueError, "caller-observed"):
                fixture.admit()
            fixture.parents.frozen_admission = None
            with self.assertRaisesRegex(ValueError, "strict frozen parent"):
                fixture.admit()

    def test_stale_current_label_parent_or_target_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root, history=True)
            fixture.body["prepared_inputs"][0]["current"]["label_sha256"] = sha("stale current label")
            fixture.refresh()
            with self.assertRaisesRegex(ValueError, "current label"):
                fixture.admit()
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root)
            checked = fixture.admit()
            checked.targets = (replace(checked.targets[0], sign=-1), *checked.targets[1:])
            with self.assertRaisesRegex(ValueError, "changed"):
                checked.verify()
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root)
            checked = fixture.admit()
            checked.parents.records[0]["input"]["snapshot"]["input_revision"] += 1
            with self.assertRaisesRegex(ValueError, "immutable raw"):
                checked.verify()

    def test_pairwise_softplus_orientation_finite_and_no_autograd(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root, roles=("proposer",))
            checked = fixture.admit()
            batch = checked.collate([0], role="proposer")
            self.assertAlmostEqual(float(candidate.pairwise_softplus_loss(torch.tensor([[2.0, 0.0]]), batch)),
                                   math.log1p(math.exp(-2)), places=6)
            for logits in (torch.tensor([[2., 0.]], requires_grad=True), torch.tensor([[float("nan"), 0.]]),
                           torch.zeros((1, 3)), torch.zeros((1, 2), dtype=torch.float64)):
                with self.assertRaises(ValueError):
                    candidate.pairwise_softplus_loss(logits, batch)
            with self.assertRaisesRegex(ValueError, "mask/sign"):
                candidate.pairwise_softplus_loss(torch.zeros((1, 2)), replace(batch, signs=(0,)))
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root, roles=("proposer",))
            fixture.body["pairs"][0]["preference"] = "right"
            fixture.change_receipt("proposer-0", lambda value: value["report"].update(raw_score=10))
            checked = fixture.admit()
            self.assertEqual(checked.targets[0].sign, -1)
            self.assertAlmostEqual(float(candidate.pairwise_softplus_loss(torch.tensor([[0.0, 2.0]]), checked.collate([0], role="proposer"))),
                                   math.log1p(math.exp(-2)), places=6)

    def test_fixed_criterion_and_declared_preferences_cannot_override_real_units(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root)
            fixture.body["pairs"][0]["preference"] = "right"
            fixture.refresh()
            with self.assertRaisesRegex(ValueError, "ordinal criterion"):
                fixture.admit()
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root)
            changed = json.loads(fixture.criterion)
            changed["method"] = "cp_to_wdl"
            fixture.criterion = raw(changed)
            fixture.refresh()
            with self.assertRaisesRegex(ValueError, "unsupported fixed"):
                fixture.admit()

    def test_generic_metadata_cannot_construct_checked_pair_bank(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root)
            with self.assertRaisesRegex(ValueError, "checked adapter factory"):
                candidate.CheckedComparativePairs(fixture.parents, (), {"scope": "metadata_only"}, {})

    def test_persisted_bank_pin_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = CandidateFixture(root)
            directory, bank_sha = fixture.persist()
            options = {"strict_parent_options": fixture.parent_options, "expected_metadata_pins": fixture.expected_metadata,
                       "independent_pins": fixture.pins}
            with self.assertRaisesRegex(ValueError, "independent pin"):
                candidate.load_persisted_comparative_pairs(root, directory, expected_bank_sha256=sha("other bank"), **options)
            manifest = json.loads((directory / "candidate-bank.json").read_bytes())
            manifest["files"]["binary"] = "../elsewhere"
            changed = raw(manifest)
            (directory / "candidate-bank.json").write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "direct children"):
                candidate.load_persisted_comparative_pairs(root, directory, expected_bank_sha256=candidate.byte_pin(changed)["sha256"], **options)


if __name__ == "__main__":
    unittest.main()
