"""Synthetic raw-byte causal/ordinal wiring, never observed Rust/native proof.

Real strict loader and semantic/candidate admission factories are used. The
registered source, caller launch observations, physical raw heads and Rules
descriptors are finite synthetic fixtures. No process, chess replay, actual
checkpoint, learned repair, GPU, backward or optimizer executes here.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from rz_pals_model import comparative as metadata
from rz_pals_model import comparative_training as candidate
from rz_pals_model import frozen_producer as frozen
from rz_pals_model import native_divergence as native
from rz_pals_model import repair_context as repair
from rz_pals_model import semantic_verifier as semantic
from rz_pals_model import training
from rz_pals_model.preparation_check import _parameter_digest
from test_comparative_training import CandidateFixture
from test_current_consumers import FrozenNumericFixture
from test_native_divergence import NativeDivergenceFixture
from test_semantic_verifier import SemanticFixture, tokens
import test_training as fixtures


def raw(value):
    return fixtures.frozen_fixture_bytes(value)


def lines(values):
    return b"".join(raw(value) + b"\n" for value in values)


class RepairFixture:
    """Same registered producer, two CURRENT ordinary P inputs, no aux D row."""
    def __init__(self, directory):
        self.root = Path(directory)
        seed_root = self.root / "synthetic-native-seed"
        seed_root.mkdir()
        seed = NativeDivergenceFixture(seed_root)
        self.collection = seed_root
        self.base, self.options = seed.base, seed.options
        self.source_raw, self.export_raw, self.registration_raw = seed.source_raw, seed.export_raw, seed.registration_raw
        self.source = json.loads(self.source_raw)[1]
        self.rows = [copy.deepcopy(seed.parent_row), copy.deepcopy(seed.parent_row)]
        base, target = (row["input"]["snapshot"] for row in self.rows)
        base.update(public_records=[], input_revision=1, capture_sequence=1)
        target.update(public_records=[], input_revision=3, capture_sequence=9,
                      position_command=base["position_command"] + " moves e2e4 e8d7",
                      board_fen="8/3k4/8/8/4P3/8/8/4K3 w - - 1 2",
                      rules_state_sha256=fixtures.sha("synthetic repair Rules state"),
                      rules_history_sha256=fixtures.sha("synthetic repair Rules history"),
                      transposition_sha256=fixtures.sha("synthetic repair transposition"),
                      legal_moves=[fixtures.move(4, 12), fixtures.move(4, 3), fixtures.move(4, 5)])
        self.lineages = [
            {"input_sha256": "", "game_id": base["game_id"], "process_epoch": 1, "request_sequence": 1,
             "native_query_kind": "Propose", "actual_played_history": [], "virtual_prefix": [], "proposal": [],
             "counterexample": None, "divergence_plies": [], "actual_outcome_eligible": True,
             "counterfactual_wdl": "masked", "training_admission": "ordinary_role"},
            {"input_sha256": "", "game_id": base["game_id"], "process_epoch": 1, "request_sequence": 8,
             "native_query_kind": "Repair", "actual_played_history": [],
             "virtual_prefix": [fixtures.move(12, 28), fixtures.move(60, 51)],
             "proposal": [fixtures.move(12, 28), fixtures.move(60, 52), fixtures.move(4, 12), fixtures.move(52, 44)],
             "counterexample": [fixtures.move(12, 28), fixtures.move(60, 51), fixtures.move(4, 12)],
             "divergence_plies": [], "actual_outcome_eligible": False, "counterfactual_wdl": "masked",
             "training_admission": "ordinary_role"}]
        self.sidecars = [fixtures.native_sidecar(row) for row in self.rows]
        self.tensors = [json.loads(sidecar["tensor_json"]) for sidecar in self.sidecars]
        self.events, self.outputs = [], []
        for index, lineage in enumerate(self.lineages):
            details = ({"prepared_before_submit": True, "native_query_kind": lineage["native_query_kind"], "producer_metadata_admitted": True},
                       {"success": True, "logical_acceptance_inferred": False}, {"search_consumed": False}, {"search_consumed": True})
            self.events.extend({"domain": "rz-pals-native-call-event/1", "game_id": base["game_id"],
                                "process_epoch": lineage["process_epoch"], "request_sequence": lineage["request_sequence"],
                                "input_sha256": "", "stage": stage, "observer_elapsed_us": index * 10 + slot + 1,
                                "detail": details[slot]}
                               for slot, stage in enumerate(("prepared", "physically_completed", "delivered", "search_consumed")))
            self.outputs.append({"domain": "rz-pals-native-physical-raw/1", "process_epoch": lineage["process_epoch"],
                                 "request_sequence": lineage["request_sequence"], "input_sha256": "",
                                 "physical_completion_confirmed": True, "success": True,
                                 "raw": {"representation": "f32_ieee754_bits", "candidate_logits_bits": [0] * len(self.tensors[index]["candidates"]),
                                         "wdl_logits_bits": [0] * 3, "divergence_logits_bits": None, "task_logits_bits": None,
                                         "private_latent_bits": [0] * 6144, "prediction_is_future_label": False}})
        self.query_override = {}
        self.output_count = 2
        self.rebuild()

    def rebuild(self):
        """Re-pin synthetic source bytes to reach individual negative gates."""
        self.journals = []
        roster = json.loads(self.base["producer-roster.json"])
        template = json.loads(self.base["producer-prepared.jsonl"])
        for index, (row, sidecar, tensor, lineage) in enumerate(zip(self.rows, self.sidecars, self.tensors, self.lineages)):
            snapshot = row["input"]["snapshot"]
            row["input"]["sha256"] = training.seal_snapshot(snapshot)
            identity = row["input"]["sha256"]
            lineage["input_sha256"] = identity
            prefix, proposal, counter = lineage["virtual_prefix"], lineage["proposal"], lineage["counterexample"]
            query = [0.0] * 16
            query[:9] = [0 if index == 0 else 2, len(prefix) / 256, len(proposal) / 256, int(counter is not None),
                         0 if counter is None else len(counter) / 256, len(snapshot["legal_moves"]) / 256,
                         snapshot["input_revision"], 9.25, int(snapshot["white_to_move"])]
            for offset, movement in ((9, prefix[-1] if prefix else None), (12, proposal[0] if proposal else None)):
                if movement is not None:
                    source, target, promotion = training.move_components(movement)
                    query[offset:offset + 3] = [source / 63, target / 63, promotion / 4]
            query[15] = 1
            if index in self.query_override:
                for slot, value in self.query_override[index].items():
                    query[slot] = value
            tensor.update(board=list(training._fen_board(snapshot["board_fen"])), records=[], required_critical_records=[],
                          candidates=[dict(zip(("from", "to", "promotion"), training.move_components(move))) for move in snapshot["legal_moves"]],
                          query=query, model_epoch=list(bytes.fromhex(snapshot["source"]["model_weights_sha256"])),
                          situation_revision=snapshot["input_revision"], history_digest=list(bytes.fromhex(snapshot["rules_history_sha256"])))
            sidecar.update(input_sha256=identity, record_sources=[], model_epoch_kind="frozen_model_epoch",
                           tensor_json=json.dumps(tensor, separators=(",", ":"), allow_nan=False))
            fixtures.reseal_sidecar(sidecar)
            journal = copy.deepcopy(template)
            journal["prepared"].update(input_sha256=identity, capture_sequence=snapshot["capture_sequence"],
                native_request=[lineage["process_epoch"], lineage["request_sequence"]], input_json=semantic.byte_pin(raw(row["input"])),
                tensor_sidecar_json=semantic.byte_pin(raw(sidecar)), lineage_json=semantic.byte_pin(raw(lineage)), learning_input=True)
            journal["sha256"] = training._sorted_canonical(training.PREPARED_PRODUCER_DOMAIN, journal["prepared"])
            self.journals.append(journal)
            for event in self.events:
                if event["request_sequence"] == (1 if index == 0 else 8):
                    event.update(input_sha256=identity, process_epoch=lineage["process_epoch"], request_sequence=lineage["request_sequence"])
            self.outputs[index].update(input_sha256=identity, process_epoch=lineage["process_epoch"], request_sequence=lineage["request_sequence"])
        bindings = [{"input_sha256": journal["prepared"]["input_sha256"], "game_id": journal["prepared"]["game_id"],
                     "producer_id": journal["prepared"]["producer_id"], "capture_sequence": journal["prepared"]["capture_sequence"],
                     "capture_evidence_sha256": journal["sha256"]} for journal in self.journals]
        capture = frozen.seal_capture({"version": frozen.CAPTURE_DOMAIN, "bindings": bindings})
        registry = json.loads(self.base["source-registry.jsonl"])
        split = json.loads(self.base["split.jsonl"])
        view = training.ValidatedDataset(self.rows, split, registry, {}).current_view
        envelope = json.loads(self.base["producer-envelope.json"])["envelope"]
        envelope.update(raw_records=len(self.rows), unique_inputs=2, current_view_sha256=view.sha256,
                        capture_sha256=capture["sha256"], capture_artifact=semantic.byte_pin(raw(capture)))
        envelope = frozen.seal_envelope(envelope)
        audit = json.loads(self.base["producer-audit.json"])
        audit.update(raw_records=len(self.rows), unique_inputs=2, native_exact_metadata_inputs=2,
                     envelope_sha256=envelope["sha256"], capture_sha256=capture["sha256"])
        assets = dict(self.base)
        assets.update({"records.jsonl": lines(self.rows), "inputs.jsonl": lines([row["input"] for row in self.rows]),
                       "native-inputs.jsonl": lines(self.sidecars), "input-lineage.jsonl": lines(self.lineages),
                       "producer-prepared.jsonl": lines(self.journals), "producer-captures.json": raw(capture),
                       "producer-envelope.json": raw(envelope), "producer-audit.json": raw(audit),
                       "public-record-sources.jsonl": b"", "native-events.jsonl": lines(self.events),
                       "native-raw-outputs.jsonl": lines(self.outputs[:self.output_count])})
        receipt = json.loads(self.base["receipt.json"])
        receipt["audit"]["records"] = len(self.rows)
        receipt["native_finish"] = {"receipt": {"physical_shutdown_confirmed": True, "native_buffers_released": True,
            "quarantined": False, "physical_runs_in_flight": 0, "observer_failures": 0},
            "finish_error": None, "_collection_failure": None, "collection_accepted": True}
        for name, value in assets.items():
            if name != "receipt.json":
                receipt["artifacts"][name] = semantic.byte_pin(value)
        assets["receipt.json"] = raw(receipt)
        for name, value in assets.items():
            (self.collection / name).write_bytes(value)
        self.options["expected_receipt_sha256"] = semantic.byte_pin(assets["receipt.json"])["sha256"]
        self.parents = training.load_frozen_collected_dataset(self.collection, **self.options)
        self.root_index = next(index for index in self.parents.current_view.current_indices
                               if self.parents.records[index]["input"]["sha256"] == self.rows[0]["input"]["sha256"])
        self.repair_index = next(index for index in self.parents.current_view.current_indices
                                 if self.parents.records[index]["input"]["sha256"] == self.rows[1]["input"]["sha256"])
        self.artifacts = {name: assets[name] for name in repair._NAMES}
        launch = {"schema": native.LAUNCH_SCHEMA, "receipt_artifact": semantic.byte_pin(assets["receipt.json"]),
                  "checked_source_artifact": semantic.byte_pin(self.source_raw), "collector_binary_sha256": self.source["implementation_sha256"],
                  "assurance_scope": "independently_pinned_caller_collection_observation", "spawned": True, "reaped": True,
                  "exit_code": 0, "timed_out": False}
        self.launch_raw = raw(launch)
        self._rules_checks()
        self.context_and_pins()

    def _rules_checks(self):
        self.semantic_fixtures = []
        for name in ("prefix", "proposal"):
            folder = self.root / ("synthetic-semantic-" + name)
            folder.mkdir(exist_ok=True)
            check = SemanticFixture(folder)
            check.parents, check.index = self.parents, self.root_index
            check.snapshot = copy.deepcopy(self.rows[0]["input"]["snapshot"])
            check.common.update(parent_input_sha256=self.rows[0]["input"]["sha256"], current_view_sha256=self.parents.current_view.sha256,
                                question="continuation_challenge", prefix=self.lineages[1]["virtual_prefix"] if name == "prefix" else [],
                                claimed_line=self.lineages[1]["counterexample"][len(self.lineages[1]["virtual_prefix"]):] if name == "prefix" else self.lineages[1]["proposal"],
                                allowed_tasks=["attack_repair"] if name == "prefix" else ["resume_task"])
            root = check.receipt["root"]
            self._descriptor(root, self.rows[0]["input"]["snapshot"], "root_legal", 1)
            target = copy.deepcopy(root)
            if name == "prefix":
                self._descriptor(target, self.rows[1]["input"]["snapshot"], "target_legal", 3)
            else:
                target["legal_tokens"] = tokens(target["legal_moves"], "target_legal", target["legal_moves"])
            check.receipt["target"] = target
            final = copy.deepcopy(target)
            final.update(board_fen="8/3k4/8/8/4P3/8/4K3/8 b - - 2 2" if name == "prefix" else "8/8/4k3/8/4P3/8/4K3/8 w - - 3 3",
                         side_to_move="black" if name == "prefix" else "white",
                         board64_piece_codes=list(training._fen_board("8/3k4/8/8/4P3/8/4K3/8 b - - 2 2" if name == "prefix" else "8/8/4k3/8/4P3/8/4K3/8 w - - 3 3")),
                         rules_state_sha256=fixtures.sha("synthetic " + name + " claim state"),
                         rules_history_sha256=fixtures.sha("synthetic " + name + " claim history"),
                         known_history_positions=4 if name == "prefix" else 5)
            final["legal_tokens"] = tokens(final["legal_moves"], "claim_end_legal", final["legal_moves"])
            check.receipt["claimed_line"] = {"status": "legal_continuation", "legality_verified": True, "claim_truth": "unknown",
                "movements": tokens(check.common["claimed_line"], "claimed_continuation"), "restriction_checked": False, "final_state": final}
            check.reseal()
            self.semantic_fixtures.append(check)
        self.prefix_check, self.proposal_check = [check.admit() for check in self.semantic_fixtures]

    def _descriptor(self, descriptor, snapshot, kind, positions):
        descriptor.update(**{key: copy.deepcopy(snapshot[key]) for key in ("rules_state_sha256", "rules_history_sha256", "board_fen", "legal_moves")},
                          side_to_move="white" if snapshot["white_to_move"] else "black",
                          board64_piece_codes=list(training._fen_board(snapshot["board_fen"])), known_history_positions=positions)
        descriptor["legal_order_sha256"] = semantic.digest(semantic.MOVE_DOMAIN, descriptor["legal_moves"])
        descriptor["legal_tokens"] = tokens(descriptor["legal_moves"], kind, descriptor["legal_moves"])

    def context_and_pins(self):
        lineage = self.lineages[1]
        body = {"schema": repair.CONTEXT_SCHEMA, "provenance_mode": "checked_existing_prepared_lineage",
                "parent_input_sha256": self.rows[0]["input"]["sha256"], "parent_label_sha256": training.label_digest(self.rows[0]),
                "repair_input_sha256": self.rows[1]["input"]["sha256"], "repair_label_sha256": training.label_digest(self.rows[1]),
                "current_view_sha256": self.parents.current_view.sha256, "frozen_admission_sha256": self.parents._frozen_admission_identity,
                "collection_receipt_sha256": semantic.byte_pin(self.artifacts["receipt.json"])["sha256"],
                "root_prepared_sha256": self.journals[0]["sha256"], "repair_prepared_sha256": self.journals[1]["sha256"],
                "root_native_request": self.journals[0]["prepared"]["native_request"], "repair_native_request": self.journals[1]["prepared"]["native_request"],
                "divergence_ply": len(lineage["virtual_prefix"]) - 1, "virtual_prefix": lineage["virtual_prefix"],
                "proposal": lineage["proposal"], "counterexample": lineage["counterexample"],
                "rules_prefix_sha256": self.prefix_check.sha256, "rules_proposal_sha256": self.proposal_check.sha256}
        self.context_raw = raw({"context": body, "sha256": semantic.digest(repair.CONTEXT_SCHEMA, body)})
        self.pins = {"artifacts": {name: semantic.byte_pin(value) for name, value in self.artifacts.items()},
                     "registration": semantic.byte_pin(self.registration_raw), "checked_source": semantic.byte_pin(self.source_raw),
                     "export_manifest": semantic.byte_pin(self.export_raw), "launch": semantic.byte_pin(self.launch_raw),
                     "context": semantic.byte_pin(self.context_raw), "parent_input_sha256": self.rows[0]["input"]["sha256"],
                     "repair_input_sha256": self.rows[1]["input"]["sha256"], "current_view_sha256": self.parents.current_view.sha256,
                     "frozen_admission_sha256": self.parents._frozen_admission_identity,
                     "rules_prefix_sha256": self.prefix_check.sha256, "rules_proposal_sha256": self.proposal_check.sha256}

    def arguments(self):
        return {"parents": self.parents, "root_index": self.root_index, "repair_index": self.repair_index,
                "artifacts": self.artifacts, "registration_bytes": self.registration_raw, "checked_source_bytes": self.source_raw,
                "export_manifest_bytes": self.export_raw, "launch_bytes": self.launch_raw, "context_bytes": self.context_raw,
                "prefix_rules_check": self.prefix_check, "proposal_rules_check": self.proposal_check, "expected_pins": self.pins}

    def admit(self):
        return repair.admit_repair_context(**self.arguments())

    def candidate_bank(self):
        """Existing real adapter, independent checker; synthetic CPU receipts."""
        folder = self.root / "synthetic-candidate-bank"
        folder.mkdir(exist_ok=True)
        bank = CandidateFixture(folder, roles=("proposer",))
        bank.parents = self.parents
        bank.parent_artifacts = {name: self.artifacts[name] for name in candidate_parent_names()}
        admission = self.parents.frozen_admission
        bank.body["parent"] = {"receipt_artifact": admission["receipt"], "raw_dataset_sha256": admission["raw_dataset_sha256"],
            "split_sha256": admission["split_sha256"], "current_view_sha256": self.parents.current_view.sha256,
            "producer_roster_sha256": json.loads(self.artifacts["producer-roster.json"])["sha256"],
            "producer_envelope_sha256": json.loads(self.artifacts["producer-envelope.json"])["sha256"]}
        row = self.parents.records[self.repair_index]
        snapshot, journal = row["input"]["snapshot"], self.journals[1]
        current = {"input_sha256": row["input"]["sha256"], "label_sha256": training.label_digest(row),
            **{key: snapshot[key] for key in ("game_id", "role", "rules_state_sha256", "rules_history_sha256", "encoding_sha256",
                                             "source", "frozen_epoch", "input_revision", "legal_moves")},
            "side_to_move": "white" if snapshot["white_to_move"] else "black"}
        bank.body["prepared_inputs"] = [{"current": current, "producer_id": journal["prepared"]["producer_id"],
            "producer_registration_sha256": journal["prepared"]["registration_sha256"], "capture_evidence_sha256": journal["sha256"],
            **{key: journal["prepared"][key] for key in ("input_json", "tensor_sidecar_json", "lineage_json")}}]
        pair = bank.body["pairs"][0]
        pair.update(input_sha256=row["input"]["sha256"], candidates=snapshot["legal_moves"][:2])
        for slot, check in enumerate(pair["checks"]):
            movement = pair["candidates"][slot]
            check.update(candidate=movement, conditions=metadata.task_conditions(current, bank.body["checker"]),
                         restriction={"kind": "candidate_only", "root_moves": [movement]})
        bank.plan = raw({"schema": candidate.PLAN_SCHEMA, **{key: bank.body[key] for key in ("parent", "criterion", "prepared_inputs")},
                         "checker_namespace_sha256": metadata.checker_namespace(bank.body["checker"]),
                         "pairs": [{key: pair[key] for key in ("pair_id", "input_sha256", "kind", "candidates")}
                                   | {"task_ids": [check["task_id"] for check in pair["checks"]]}]})
        for check in pair["checks"]:
            task, movement = check["task_id"], check["candidate"]
            execution = bank.executions[task]
            request, receipt = json.loads(execution["request"]), json.loads(execution["receipt"])
            request.update(captured_input_sha256=row["input"]["sha256"], before_result_anchor_sha256=semantic.byte_pin(bank.plan)["sha256"],
                **{key: snapshot[key] for key in ("frozen_epoch", "input_revision", "position_command", "rules_state_sha256", "rules_history_sha256", "white_to_move")},
                expected_board_fen=snapshot["board_fen"], expected_legal_moves=snapshot["legal_moves"], candidate=movement)
            request["context_sha256"] = candidate.wire_digest([candidate.REQUEST_SCHEMA, {key: value for key, value in request.items() if key != "context_sha256"}])
            for key in ("captured_input_sha256", "before_result_anchor_sha256", "frozen_epoch", "input_revision", "context_sha256"):
                receipt[key] = request[key]
            conditions = receipt["report"]["conditions"]
            conditions.update(**{key: request[key] for key in ("rules_state_sha256", "rules_history_sha256", "white_to_move")},
                              board_fen=snapshot["board_fen"], legal_moves=snapshot["legal_moves"], root_moves=[movement],
                              legal_order_sha256=candidate.wire_digest([candidate.LEGAL_DOMAIN, snapshot["legal_moves"]]))
            receipt["report"].update(conditions_sha256=candidate.wire_digest(conditions), best_move=movement, pv=[movement])
            execution.update(request=raw(request), receipt=raw(receipt))
            bank.observe(task)
        bank.refresh()
        return bank


def candidate_parent_names():
    return ("receipt.json", "producer-roster.json", "producer-envelope.json", "producer-prepared.jsonl", "inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl")


class RepairContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = RepairFixture(self.temp.name)

    def test_real_factories_bind_ordinary_initial_repair_without_validity_or_wdl(self):
        checked = self.fixture.admit()
        admission = checked.admission()
        self.assertTrue(admission["native_repair_search_consumed"])
        self.assertTrue(admission["proposed_hypothesis_only"])
        self.assertEqual(admission["provenance_mode"], "checked_existing_prepared_lineage")
        for name in ("strategic_repair_validity_admitted", "counterexample_validity_admitted", "counterfactual_wdl_admitted", "divergence_ranking_admitted", "actual_training_executed"):
            self.assertFalse(admission[name])
        self.assertEqual(checked.context()["repair_input_sha256"], self.fixture.rows[1]["input"]["sha256"])
        self.assertEqual(len(self.fixture.parents.records), 2)
        self.assertEqual(set(self.fixture.parents.current_view.current_indices), {0, 1})

    def test_factory_type_and_mutable_return_cannot_inject_a_repair_proof(self):
        with self.assertRaisesRegex(ValueError, "unchecked"):
            repair.CheckedRepairContext()
        checked = self.fixture.admit()
        with self.assertRaises(AttributeError):
            checked._repair_index = 0
        view = checked.context()
        view["virtual_prefix"][:] = []
        self.assertEqual(len(checked.context()["virtual_prefix"]), 2)
        args = self.fixture.arguments()
        args["prefix_rules_check"] = lambda: True
        with self.assertRaisesRegex(ValueError, "capabilities"):
            repair.admit_repair_context(**args)

    def test_independent_raw_asset_pin_mismatch_is_rejected(self):
        args = self.fixture.arguments()
        args["artifacts"] = {**self.fixture.artifacts, "native-raw-outputs.jsonl": self.fixture.artifacts["native-raw-outputs.jsonl"] + b" "}
        with self.assertRaisesRegex(ValueError, "pin"):
            repair.admit_repair_context(**args)

    def test_auxiliary_or_reply_lineage_cannot_become_ordinary_repair(self):
        self.fixture.lineages[1]["training_admission"] = "deferred_divergence_head"
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "ordinary.*lineage"):
            self.fixture.admit()

    def test_physical_only_or_logically_rejected_hypothesis_is_not_repair_admission(self):
        self.fixture.events[-1].update(stage="logically_rejected", detail={"reason": "synthetic stale", "search_consumed": False})
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "search-consumed"):
            self.fixture.admit()

    def test_nonfinite_native_heads_are_rejected(self):
        self.fixture.outputs[1]["raw"]["candidate_logits_bits"][0] = 0x7F800000
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.fixture.admit()

    def test_missing_selected_physical_raw_is_not_empty_success(self):
        self.fixture.output_count = 1
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "missing/ambiguous.*physical raw"):
            self.fixture.admit()

    def test_failed_physical_output_is_not_a_legal_hypothesis_proof(self):
        self.fixture.outputs[1]["success"] = False
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "successful known"):
            self.fixture.admit()

    def test_tensor_query_kind_is_not_inferred_from_role_or_teacher(self):
        self.fixture.query_override[1] = {0: 0}
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "query features"):
            self.fixture.admit()

    def test_same_root_semantic_prefix_must_reach_exact_repair_state_and_legal_order(self):
        check = self.fixture.semantic_fixtures[0]
        check.receipt["target"]["rules_state_sha256"] = fixtures.sha("wrong synthetic target")
        check.reseal()
        self.fixture.prefix_check = check.admit()
        self.fixture.context_and_pins()
        with self.assertRaisesRegex(ValueError, "does not reach"):
            self.fixture.admit()

    def test_counterexample_prefix_cannot_be_the_original_proposal_response(self):
        self.fixture.lineages[1]["proposal"][1] = self.fixture.lineages[1]["virtual_prefix"][1]
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "opponent counterexample"):
            self.fixture.admit()

    def test_root_request_and_capture_chronology_is_preserved(self):
        self.fixture.rows[0]["input"]["snapshot"]["capture_sequence"] = 10
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "ordered.*captures"):
            self.fixture.admit()

    def test_sealed_context_and_parent_mutation_fail_closed(self):
        checked = self.fixture.admit()
        self.fixture.parents.records[self.fixture.repair_index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaises(ValueError):
            checked.verify()

    def test_separately_sealed_context_must_use_exact_current_label_anchor(self):
        value = json.loads(self.fixture.context_raw)
        value["context"]["repair_label_sha256"] = fixtures.sha("unrelated future label")
        value["sha256"] = semantic.digest(repair.CONTEXT_SCHEMA, value["context"])
        self.fixture.context_raw = raw(value)
        self.fixture.pins["context"] = semantic.byte_pin(self.fixture.context_raw)
        with self.assertRaisesRegex(ValueError, "current labels"):
            self.fixture.admit()

    def test_next_move_pair_reuses_actual_candidate_receipts_and_frozen_loss(self):
        causal = self.fixture.admit()
        bank = self.fixture.candidate_bank()
        for check in bank.body["pairs"][0]["checks"]:
            execution = bank.executions[check["task_id"]]
            request, receipt = json.loads(execution["request"]), json.loads(execution["receipt"])
            self.assertEqual(check["restriction"], {"kind": "candidate_only", "root_moves": [check["candidate"]]})
            self.assertEqual(request["candidate"], check["candidate"])
            self.assertEqual(receipt["report"]["conditions"]["root_moves"], check["restriction"]["root_moves"])
            self.assertEqual(request["captured_input_sha256"], causal.context()["repair_input_sha256"])
        admitted = bank.admit()
        checked = repair.bind_repair_candidate_pairs(causal, admitted, indices=[0])
        self.assertEqual(checked.targets[0].input_sha256, causal.context()["repair_input_sha256"])
        self.assertTrue(checked.targets[0].mask)
        batch = checked.collate([0])
        loss = candidate.pairwise_softplus_loss(torch.zeros((1, 3), dtype=torch.float32), batch)
        self.assertGreater(float(loss), 0)
        model = FrozenNumericFixture().eval().requires_grad_(False)
        report = repair.frozen_repair_preparation(model, checked, expected_parameter_sha256=_parameter_digest(model), checkpoint_sha256=fixtures.sha("synthetic reload observation"))
        self.assertEqual(report["known_pairs"], 1)
        self.assertGreater(report["pairwise_loss"], 0)
        self.assertEqual(report["parameter_sha256_before"], report["parameter_sha256_after"])
        self.assertFalse(report["strategic_repair_validity_admitted"])
        self.assertFalse(report["actual_training_executed"])

    def test_partial_candidate_pair_retains_mask_and_rejects_positive_preparation(self):
        bank = self.fixture.candidate_bank()
        pair = bank.body["pairs"][0]
        pair["preference"] = "masked"
        task = pair["checks"][0]["task_id"]
        value = json.loads(bank.executions[task]["receipt"])
        value["status"] = "partial"
        value["report"].update(completion="node_limit", completed_depth=1)
        bank.executions[task]["receipt"] = raw(value)
        pair["checks"][0].update(completion="partial", completed_depth=1)
        bank.observe(task)
        bank.refresh()
        checked = repair.bind_repair_candidate_pairs(self.fixture.admit(), bank.admit(), indices=[0])
        self.assertFalse(checked.targets[0].mask)
        self.assertEqual(checked.targets[0].sign, 0)
        model = FrozenNumericFixture().eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "all-masked"):
            repair.frozen_repair_preparation(model, checked, expected_parameter_sha256=_parameter_digest(model), checkpoint_sha256=fixtures.sha("synthetic checkpoint"))

    def test_unrelated_current_ordinary_pair_cannot_be_relabelled_repair(self):
        folder = self.fixture.root / "unrelated-synthetic-candidate"
        folder.mkdir()
        bank = CandidateFixture(folder, roles=("proposer",))
        with self.assertRaisesRegex(ValueError, "parent"):
            repair.bind_repair_candidate_pairs(self.fixture.admit(), bank.admit(), indices=[0])

    def test_mutation_during_frozen_forward_invalidates_causal_preparation(self):
        bank = self.fixture.candidate_bank()
        checked = repair.bind_repair_candidate_pairs(self.fixture.admit(), bank.admit(), indices=[0])

        def mutate():
            self.fixture.parents.records[self.fixture.repair_index]["input"]["snapshot"]["input_revision"] += 1

        model = FrozenNumericFixture(on_forward=mutate).eval().requires_grad_(False)
        with self.assertRaises(ValueError):
            repair.frozen_repair_preparation(model, checked, expected_parameter_sha256=_parameter_digest(model), checkpoint_sha256=fixtures.sha("synthetic checkpoint"))

    def test_forward_cannot_enable_training_even_when_parameter_bytes_are_unchanged(self):
        bank = self.fixture.candidate_bank()
        checked = repair.bind_repair_candidate_pairs(self.fixture.admit(), bank.admit(), indices=[0])
        model = FrozenNumericFixture().eval().requires_grad_(False)
        before = _parameter_digest(model)
        model.on_forward = lambda: model.train()
        with self.assertRaisesRegex(ValueError, "caller-frozen"):
            repair.frozen_repair_preparation(model, checked, expected_parameter_sha256=before, checkpoint_sha256=fixtures.sha("synthetic checkpoint"))
        self.assertEqual(before, _parameter_digest(model))
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_forward_cannot_enable_autograd_even_when_parameter_bytes_are_unchanged(self):
        bank = self.fixture.candidate_bank()
        checked = repair.bind_repair_candidate_pairs(self.fixture.admit(), bank.admit(), indices=[0])
        model = FrozenNumericFixture().eval().requires_grad_(False)
        before = _parameter_digest(model)
        model.on_forward = lambda: model.fixed.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "caller-frozen"):
            repair.frozen_repair_preparation(model, checked, expected_parameter_sha256=before, checkpoint_sha256=fixtures.sha("synthetic checkpoint"))
        self.assertEqual(before, _parameter_digest(model))
        self.assertTrue(model.fixed.requires_grad)
        self.assertIsNone(model.fixed.grad)


if __name__ == "__main__":
    unittest.main()
