"""Finite synthetic DG02 raw-byte, context, scope and CPU collator fixtures.

No Rust child, real native model, chess Rules replay, GPU or training executes.
Registered producer/launch and prefix descriptors below are synthetic caller
facts. Tests prove conditional admission checks only; never before-dispatch
collection history, ranking validity, repair utility or learned strength.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch

from rz_pals_model import native_divergence as divergence
from rz_pals_model import semantic_verifier as semantic
from rz_pals_model import training
import test_training as fixtures
from test_semantic_verifier import SemanticFixture, tokens


def raw(value):
    return fixtures.frozen_fixture_bytes(value)


def lines(values):
    return b"".join(raw(value) + b"\n" for value in values)


class NativeDivergenceFixture:
    """One true ordinary parent and a separately journaled false-learning D.

    The auxiliary deliberately differs in role, capture sequence, revision and
    selected public records. Neither synthetic source nor caller observation
    proves an actual Rust/native process; they exercise the declared trust seam.
    """
    def __init__(self, directory, *, captured=True, critic=True, layout="separate_pc", role_batching="one_scalar_role_per_physical_batch",
                 manifest_semantic=None, loaded_adapter_sha=None, loaded_encoding_sha=None):
        self.root, self.captured = Path(directory), captured
        original_bytes = fixtures.frozen_fixture_bytes
        original_fixture = fixtures.fixture
        export = []
        self.rules_semantic = fixtures.sha("synthetic exact Rules field semantics")

        def native_row(*args, **kwargs):
            row = original_fixture(*args, **kwargs)
            row["input"]["snapshot"]["encoding_sha256"] = hashlib.sha256(
                divergence.PALS_ENCODING_SCHEMA + bytes.fromhex(self.rules_semantic)).hexdigest()
            row["input"]["sha256"] = training.seal_snapshot(row["input"]["snapshot"])
            return row

        def with_critic(value):
            if critic and type(value) is list and len(value) == 2 and value[0] == training.CHECKED_SOURCE_DOMAIN:
                native = value[1].get("native")
                if native is not None:
                    graph_roles = ("public", "shared_pc") if layout == "shared_pc_if" else ("public", "proposer", "critic")
                    graphs = [{"role": role, "sha256": fixtures.sha("fixture " + role + " graph"), "serialized_bytes": 32} for role in graph_roles]
                    native["graphs"] = graphs
                    native["independent_registry"]["graphs"] = copy.deepcopy(graphs)
            if type(value) is list and len(value) == 2 and value[0] == training.CHECKED_SOURCE_DOMAIN and not export:
                description = value[1]
                native = description["native"]
                manifest = {"schema": "rovezero.pals-model.v2" if layout == "shared_pc_if" else "rovezero.pals-model.v1",
                            "layout": layout, "config": description["configuration"],
                            "checkpoint_sha256": description["source"]["model_weights_sha256"], "trained": False, "training_steps": 0,
                            "roles": ["proposer", "critic"] if critic else ["proposer"], "validator_present": False,
                            "rules_input_profile": "rz-pals-rules-fields-v1", "rules_input_semantic_sha256": manifest_semantic or self.rules_semantic,
                            "rules_encoder_source_sha256": fixtures.sha("synthetic export-era encoder source; differs from actual collector"),
                            "graphs": [{"role": graph["role"], "sha256": graph["sha256"]} for graph in native["graphs"]]}
                if layout == "shared_pc_if":
                    manifest.update(model_semantics="rovezero.pals-model.v1", layout_revision=1, role_batching=role_batching)
                    manifest["graphs"][1].update(inputs=[{"name": "role_is_critic", "dtype": "BOOL", "shape": []},
                                                         {"name": "divergence_features", "dtype": "FLOAT", "shape": ["batch", "divergences", 8]}],
                                                 outputs=[{"name": "divergence_logits", "dtype": "FLOAT", "shape": ["batch", "divergences"]}])
                manifest_raw = original_bytes(manifest)
                export.append(manifest_raw)
                manifest_sha = divergence.byte_pin(manifest_raw)["sha256"]
                native["independent_registry"]["export_manifest_sha256"] = manifest_sha
                native["loaded_source"]["export_manifest_sha256"] = list(bytes.fromhex(manifest_sha))
                native["loaded_source"]["encoding_semantic_sha256"] = list(bytes.fromhex(loaded_encoding_sha or description["encoding_sha256"]))
                native["loaded_source"]["adapter_source_sha256"] = list(bytes.fromhex(loaded_adapter_sha or description["encoder_source_sha256"]))
            return original_bytes(value)

        with patch.object(fixtures, "frozen_fixture_bytes", side_effect=with_critic), patch.object(fixtures, "fixture", side_effect=native_row):
            self.options = fixtures.write_frozen_fixture_collection(self.root, kinds=("native",))
        self.base = {path.name: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
        self.source_raw = self.base["producer-source.json"]
        self.export_raw = export[0]
        self.registration_raw = self.base["producer-registration.json"]
        self.source = json.loads(self.source_raw)[1]
        self.parent_row = json.loads(self.base["records.jsonl"])
        self.input = copy.deepcopy(self.parent_row["input"])
        aux = self.input["snapshot"]
        aux.update(role="critic", input_revision=9, capture_sequence=11)
        self.public = {"domain": "rz-pals-native-public-source/1", "game_id": aux["game_id"], "record_index": 1,
                       "revision": 8, "origin_state_id": 999, "origin_state_id_is_advisory": True,
                       "origin_rules_state_sha256": None, "origin_rules_identity_observation": "unknown",
                       "kind": "PrincipalVariation", "line": [fixtures.move(12, 28)], "value": 12,
                       "completed_depth": 2, "scope": "Raw", "white_score_perspective": True, "critical": True,
                       "source_cpu_profile_sha256": self.source["cpu_profile_sha256"]}
        aux["public_records"] = [{"observation_sha256": divergence.byte_pin(raw(self.public))["sha256"], "situation_revision": 8}]
        self.input["sha256"] = training.seal_snapshot(aux)
        row = {"input": self.input, "future_label": None, "verifier_private": None}
        self.sidecar = fixtures.native_sidecar(row)
        self.tensor = json.loads(self.sidecar["tensor_json"])
        self.tensor.update(candidates=[], required_critical_records=[1], query=[0.125] * 16,
                           divergence_features=[[3.25, -0.0, 0.25, 0.5, 0.75, 1., 2., 3.],
                                                [1.25, 0., -0.25, -0.5, -0.75, -1., -2., -3.]],
                           model_epoch=list(bytes.fromhex(aux["source"]["model_weights_sha256"])))
        self.tensor["records"][0].update(critical=True, features=[0.0625] * 16)
        self.sidecar["model_epoch_kind"] = "frozen_model_epoch"
        self.lineage = {"input_sha256": self.input["sha256"], "game_id": aux["game_id"],
                        "process_epoch": 1, "request_sequence": 10, "native_query_kind": "Divergence",
                        "actual_played_history": [], "virtual_prefix": [],
                        "proposal": [fixtures.move(12, 28), fixtures.move(60, 52), fixtures.move(4, 12), fixtures.move(52, 44)],
                        "counterexample": None, "divergence_plies": [3, 1], "actual_outcome_eligible": False,
                        "counterfactual_wdl": "masked", "training_admission": "deferred_divergence_head"}
        self.context = {"version": divergence.CONTEXT_VERSION, "input_sha256": self.input["sha256"],
                        "native_request": [1, 10], "tensor_sidecar_sha256": "0" * 64, "captured_input_revision": 9,
                        "proposal_move16": list(self.lineage["proposal"]), "challenged_line_sha256": "0" * 64,
                        "divergence_sites": [{"slot": slot, "divergence_ply": ply,
                                              "prefix_rules_state_sha256": fixtures.sha("synthetic prefix state " + str(ply)),
                                              "prefix_rules_history_sha256": fixtures.sha("synthetic prefix history " + str(ply))}
                                             for slot, ply in enumerate(self.lineage["divergence_plies"])]}
        self.journal = copy.deepcopy(json.loads(self.base["producer-prepared.jsonl"]))
        self.journal["prepared"].update(input_sha256=self.input["sha256"], capture_sequence=9, native_request=[1, 10], learning_input=False)
        details = ({"prepared_before_submit": True, "native_query_kind": "Divergence", "producer_metadata_admitted": True},
                   {"success": True, "logical_acceptance_inferred": False}, {"search_consumed": False},
                   {"search_consumed": True})
        self.events = [{"domain": "rz-pals-native-call-event/1", "game_id": aux["game_id"], "process_epoch": 1,
                        "request_sequence": 10, "input_sha256": self.input["sha256"], "stage": stage,
                        "observer_elapsed_us": index + 1, "detail": details[index]}
                       for index, stage in enumerate(("prepared", "physically_completed", "delivered", "search_consumed"))]
        self.output = {"domain": "rz-pals-native-physical-raw/1", "process_epoch": 1, "request_sequence": 10,
                       "input_sha256": self.input["sha256"], "physical_completion_confirmed": True, "success": True,
                       "raw": {"representation": "f32_ieee754_bits", "candidate_logits_bits": [],
                               "wdl_logits_bits": [0, 0, 0], "divergence_logits_bits": [0, 0], "task_logits_bits": None,
                               "private_latent_bits": [0] * 6144, "prediction_is_future_label": False}}
        self.finish = {"receipt": {"physical_shutdown_confirmed": True, "native_buffers_released": True,
                                  "quarantined": False, "physical_runs_in_flight": 0, "observer_failures": 0},
                       "finish_error": None, "_collection_failure": None, "collection_accepted": True}
        self.checks = ()
        self.rebuild()

    def rebuild(self, *, reseal_context=True, public_rows=None):
        """Re-pin synthetic collection bytes to reach deeper negative gates."""
        self.input["sha256"] = training.seal_snapshot(self.input["snapshot"])
        identity = self.input["sha256"]
        self.sidecar.update(input_sha256=identity, record_sources=copy.deepcopy(self.input["snapshot"]["public_records"]),
                            tensor_json=json.dumps(self.tensor, separators=(",", ":"), allow_nan=False))
        fixtures.reseal_sidecar(self.sidecar)
        self.lineage["input_sha256"] = identity
        self.context.update(input_sha256=identity, tensor_sidecar_sha256=self.sidecar["sha256"],
                            captured_input_revision=self.input["snapshot"]["input_revision"], proposal_move16=list(self.lineage["proposal"]))
        if reseal_context:
            self.context["challenged_line_sha256"] = hashlib.sha256(raw([
                divergence.CHALLENGED_DOMAIN, self.input["snapshot"]["rules_state_sha256"], self.lineage["proposal"]])).hexdigest()
        body = {key: value for key, value in self.context.items() if key != "sha256"}
        self.context["sha256"] = divergence.digest(divergence.CONTEXT_VERSION, body)
        if self.captured:
            self.lineage["native_divergence_context_sha256"] = self.context["sha256"]
        else:
            self.lineage.pop("native_divergence_context_sha256", None)
        self.journal["prepared"].update(input_sha256=identity, capture_sequence=self.input["snapshot"]["capture_sequence"],
                                        input_json=divergence.byte_pin(raw(self.input)), tensor_sidecar_json=divergence.byte_pin(raw(self.sidecar)),
                                        lineage_json=divergence.byte_pin(raw(self.lineage)))
        self.journal["sha256"] = training._sorted_canonical(training.PREPARED_PRODUCER_DOMAIN, self.journal["prepared"])
        for event in self.events:
            event["input_sha256"] = identity
        self.output["input_sha256"] = identity
        assets = dict(self.base)
        assets.update({"native-divergence-inputs.jsonl": lines([self.input]), "native-divergence-sidecars.jsonl": lines([self.sidecar]),
                       "input-lineage.jsonl": self.base["input-lineage.jsonl"] + lines([self.lineage]),
                       "producer-prepared.jsonl": self.base["producer-prepared.jsonl"] + lines([self.journal]),
                       "public-record-sources.jsonl": lines([self.public] if public_rows is None else public_rows),
                       "native-events.jsonl": self.base["native-events.jsonl"] + lines(self.events),
                       "native-raw-outputs.jsonl": lines([self.output])})
        if self.captured:
            assets["native-divergence-contexts.jsonl"] = lines([self.context])
        receipt = json.loads(self.base["receipt.json"])
        receipt["native_finish"] = copy.deepcopy(self.finish)
        for name, value in assets.items():
            if name != "receipt.json":
                receipt["artifacts"][name] = divergence.byte_pin(value)
        assets["receipt.json"] = raw(receipt)
        for name, value in assets.items():
            (self.root / name).write_bytes(value)
        self.options["expected_receipt_sha256"] = divergence.byte_pin(assets["receipt.json"])["sha256"]
        self.parents = training.load_frozen_collected_dataset(self.root, **self.options)
        self.index = self.parents.current_view.current_indices[0]
        names = (*divergence._BASE_NAMES, *(('native-divergence-contexts.jsonl',) if self.captured else ()))
        self.artifacts = {name: assets[name] for name in names}
        self.context_raw = raw(self.context)
        self.launch = {"schema": divergence.LAUNCH_SCHEMA, "receipt_artifact": divergence.byte_pin(assets["receipt.json"]),
                       "checked_source_artifact": divergence.byte_pin(self.source_raw),
                       "collector_binary_sha256": self.source["implementation_sha256"],
                       "assurance_scope": "independently_pinned_caller_collection_observation",
                       "spawned": True, "reaped": True, "exit_code": 0, "timed_out": False}
        self.launch_raw = raw(self.launch)
        self.pins = {"artifacts": {name: divergence.byte_pin(value) for name, value in self.artifacts.items()},
                     "registration": divergence.byte_pin(self.registration_raw), "checked_source": divergence.byte_pin(self.source_raw),
                     "export_manifest": divergence.byte_pin(self.export_raw),
                     "launch": divergence.byte_pin(self.launch_raw), "context": divergence.byte_pin(self.context_raw),
                     "rules_checks": [value.sha256 for value in self.checks],
                     "parent_input_sha256": self.parents.records[self.index]["input"]["sha256"],
                     "current_view_sha256": self.parents.current_view.sha256,
                     "frozen_admission_sha256": self.parents._frozen_admission_identity, "auxiliary_input_sha256": identity}

    def admit(self):
        return divergence.admit_native_divergence(parents=self.parents, parent_index=self.index, artifacts=self.artifacts,
            registration_bytes=self.registration_raw, checked_source_bytes=self.source_raw, export_manifest_bytes=self.export_raw, launch_bytes=self.launch_raw,
            context_bytes=self.context_raw, expected_pins=self.pins,
            provenance_mode="captured_before_dispatch" if self.captured else "derived_from_legacy_prepared", prefix_rules_checks=self.checks)

    def prefix_checks(self):
        """Actual semantic factory; synthetic pinned Rules/launch facts only."""
        checks = []
        for slot, ply in enumerate(self.lineage["divergence_plies"]):
            directory = self.root / ("synthetic-prefix-" + str(slot))
            directory.mkdir(exist_ok=True)
            fixture = SemanticFixture(directory)
            fixture.parents, fixture.index = self.parents, self.index
            fixture.snapshot = copy.deepcopy(self.parents.records[self.index]["input"]["snapshot"])
            fixture.common.update(parent_input_sha256=self.pins["parent_input_sha256"], current_view_sha256=self.parents.current_view.sha256,
                                  prefix=self.lineage["proposal"][:ply], question="continuation_challenge", allowed_tasks=["attack_repair"])
            root = fixture.receipt["root"]
            for key in ("rules_state_sha256", "rules_history_sha256", "board_fen", "legal_moves"):
                root[key] = copy.deepcopy(fixture.snapshot[key])
            root["board64_piece_codes"] = list(self.parents.encodings[self.pins["parent_input_sha256"]].board)
            root["legal_order_sha256"] = semantic.digest(semantic.MOVE_DOMAIN, root["legal_moves"])
            root["legal_tokens"] = tokens(root["legal_moves"], "root_legal", root["legal_moves"])
            target = copy.deepcopy(root)
            target.update(side_to_move="black", rules_state_sha256=self.context["divergence_sites"][slot]["prefix_rules_state_sha256"],
                          rules_history_sha256=self.context["divergence_sites"][slot]["prefix_rules_history_sha256"],
                          known_history_positions=ply + 1)
            target["legal_tokens"] = tokens(target["legal_moves"], "target_legal", target["legal_moves"])
            fixture.receipt["target"] = target
            fixture.reseal()
            checks.append(fixture.admit())
        self.checks = tuple(checks)
        self.pins["rules_checks"] = [value.sha256 for value in self.checks]


class NativeDivergenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = NativeDivergenceFixture(self.temp.name)

    def test_auxiliary_has_independent_identity_and_all_targets_stay_masked(self):
        bank = self.fixture
        before = copy.deepcopy(bank.parents.records), bank.parents.current_view
        checked = bank.admit()
        self.assertIs(checked.verify(), checked)
        self.assertNotEqual(bank.pins["parent_input_sha256"], bank.pins["auxiliary_input_sha256"])
        self.assertEqual((bank.parents.records, bank.parents.current_view), before)
        self.assertFalse(checked.admission["auxiliary_has_current_label"])
        self.assertFalse(checked.admission["producer_journal_learning_input"])
        self.assertFalse(checked.admission["same_parent_public_records_required"])
        batch = divergence.collate_native_divergences([checked])
        self.assertEqual(batch.role, "critic")
        self.assertEqual(batch.inputs.candidates.shape, (1, 1, 3))
        self.assertFalse(batch.inputs.candidate_mask.any())
        self.assertTrue(batch.inputs.divergence_mask.all())
        for mask in (batch.policy_mask, batch.wdl_mask, batch.task_mask, batch.divergence_mask):
            self.assertFalse(mask.any())
        self.assertEqual(batch.inputs.divergence_features[0, :, 0].tolist(), [3.25, 1.25])
        self.assertEqual(batch.inputs.query[0].tolist(), bank.tensor["query"])
        self.assertEqual(batch.inputs.records[0, 0].tolist(), [0.0625] * 16)
        self.assertEqual(batch.inputs.board.device.type, "cpu")
        self.assertEqual(batch.policy.dtype, torch.float32)

    def test_explicit_slot_order_prefix_identity_and_revision_are_preserved(self):
        checked = self.fixture.admit()
        encoded = checked.encoded_snapshot()
        self.assertEqual([context.divergence_ply for context, _ in encoded.divergences], [3, 1])
        self.assertEqual([context.input_revision for context, _ in encoded.divergences], [9, 9])
        self.assertEqual(encoded.divergences[0][0].challenged_line_sha256, self.fixture.context["challenged_line_sha256"])
        copied = checked.context()
        copied["divergence_sites"][0]["divergence_ply"] = 99
        self.assertEqual(checked.context()["divergence_sites"][0]["divergence_ply"], 3)
        copied = checked.admission
        copied["all_target_masks_false"] = False
        self.assertTrue(checked.admission["all_target_masks_false"])
        with self.assertRaises(ValueError):
            divergence.CheckedNativeDivergence()
        with self.assertRaises(AttributeError):
            checked._identity = "0" * 64

    def test_original_ordinary_decoder_and_collator_have_no_auxiliary_admission(self):
        row = {"input": self.fixture.input, "future_label": None, "verifier_private": None}
        with self.assertRaises(ValueError):
            training.encoded_from_sidecar(self.fixture.sidecar, row,
                expected_encoder_source_sha256=self.fixture.source["encoder_source_sha256"],
                expected_model_epoch=self.fixture.source["source"]["model_weights_sha256"])
        self.assertNotIn(self.fixture.input["sha256"], self.fixture.parents.encodings)
        with self.assertRaises(ValueError):
            divergence.collate_native_divergences([row])

    def test_parent_raw_or_current_pin_mutation_is_not_repaired(self):
        checked = self.fixture.admit()
        self.fixture.parents.records[self.fixture.index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaisesRegex(ValueError, "immutable raw"):
            checked.verify()

    def test_source_registration_and_collection_launch_are_independently_required(self):
        self.fixture.launch["collector_binary_sha256"] = fixtures.sha("unregistered collector")
        self.fixture.launch_raw = raw(self.fixture.launch)
        self.fixture.pins["launch"] = divergence.byte_pin(self.fixture.launch_raw)
        with self.assertRaisesRegex(ValueError, "launch observation"):
            self.fixture.admit()
        self.fixture.rebuild()
        self.fixture.registration_raw += b" "
        self.fixture.pins["registration"] = divergence.byte_pin(self.fixture.registration_raw)
        with self.assertRaisesRegex(ValueError, "independently registered"):
            self.fixture.admit()

    def test_unregistered_critic_graph_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            bank = NativeDivergenceFixture(root, critic=False)
            with self.assertRaisesRegex(ValueError, "critic graph"):
                bank.admit()

    def test_explicit_shared_pc_critic_route_is_supported_but_mixed_role_route_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            bank = NativeDivergenceFixture(root, layout="shared_pc_if")
            checked = bank.admit()
            self.assertEqual(checked.admission["registered_critic_graph_route"], "shared_pc_if")
            self.assertNotEqual(checked.admission["rules_input_semantic_sha256"], checked.admission["encoding_sha256"])
            self.assertNotEqual(checked.admission["export_declared_encoder_source_sha256"], checked.admission["actual_encoder_source_sha256"])
            self.assertEqual(checked.admission["encoding_sha256"], hashlib.sha256(
                divergence.PALS_ENCODING_SCHEMA + bytes.fromhex(checked.admission["rules_input_semantic_sha256"])).hexdigest())
            self.assertFalse(divergence.collate_native_divergences([checked]).divergence_mask.any())
        with tempfile.TemporaryDirectory() as root:
            bank = NativeDivergenceFixture(root, layout="shared_pc_if", role_batching="mixed_roles_per_batch")
            with self.assertRaisesRegex(ValueError, "explicit Critic graph route"):
                bank.admit()

    def test_real_rules_semantic_mismatch_and_actual_loaded_adapter_mismatch_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            bank = NativeDivergenceFixture(root, layout="shared_pc_if", manifest_semantic=fixtures.sha("another feature meaning"))
            with self.assertRaisesRegex(ValueError, "encoding namespace"):
                bank.admit()
        with tempfile.TemporaryDirectory() as root:
            bank = NativeDivergenceFixture(root, loaded_adapter_sha=fixtures.sha("different actual loaded adapter"))
            with self.assertRaisesRegex(ValueError, "actual loaded adapter/encoding"):
                bank.admit()
        with tempfile.TemporaryDirectory() as root:
            bank = NativeDivergenceFixture(root, loaded_encoding_sha=fixtures.sha("different actual loaded encoding"))
            with self.assertRaisesRegex(ValueError, "actual loaded adapter/encoding"):
                bank.admit()

    def test_self_resealed_sidecar_without_original_journal_and_receipt_is_refused(self):
        bank = self.fixture
        changed = copy.deepcopy(bank.sidecar)
        tensor = json.loads(changed["tensor_json"])
        tensor["divergence_features"][0][0] = 99.
        changed["tensor_json"] = json.dumps(tensor, separators=(",", ":"))
        fixtures.reseal_sidecar(changed)
        bank.artifacts["native-divergence-sidecars.jsonl"] = lines([changed])
        bank.pins["artifacts"]["native-divergence-sidecars.jsonl"] = divergence.byte_pin(bank.artifacts["native-divergence-sidecars.jsonl"])
        with self.assertRaisesRegex(ValueError, "receipt pin"):
            bank.admit()

    def test_same_board_different_root_history_is_not_current_parent_authority(self):
        bank = self.fixture
        bank.input["snapshot"]["rules_history_sha256"] = fixtures.sha("another auxiliary history")
        bank.tensor["history_digest"] = list(bytes.fromhex(bank.input["snapshot"]["rules_history_sha256"]))
        bank.rebuild()
        with self.assertRaisesRegex(ValueError, "same-root"):
            bank.admit()

    def test_wrong_slot_duplicate_ply_and_boolean_request_fail_closed(self):
        bank = self.fixture
        bank.context["divergence_sites"][0]["slot"] = 1
        bank.rebuild()
        with self.assertRaisesRegex(ValueError, "slot/ply"):
            bank.admit()
        bank.context["divergence_sites"][0]["slot"] = 0
        bank.lineage["divergence_plies"] = [1, 1]
        bank.rebuild()
        with self.assertRaisesRegex(ValueError, "duplicate divergence"):
            bank.admit()
        bank.lineage["divergence_plies"] = [3, 1]
        bank.context["native_request"][0] = True
        bank.rebuild()
        with self.assertRaisesRegex(ValueError, "integer bound"):
            bank.admit()

    def test_exact_challenged_domain_and_promotion_binding(self):
        bank = self.fixture
        bank.context["challenged_line_sha256"] = divergence.digest(divergence.CHALLENGED_DOMAIN, [
            bank.input["snapshot"]["rules_state_sha256"], bank.lineage["proposal"]])
        bank.rebuild(reseal_context=False)
        with self.assertRaisesRegex(ValueError, "challenged line"):
            bank.admit()
        bank.context["proposal_move16"][0] |= 1 << 12
        bank.context_raw = raw({**bank.context, "sha256": divergence.digest(divergence.CONTEXT_VERSION,
                                {key: value for key, value in bank.context.items() if key != "sha256"})})
        bank.pins["context"] = divergence.byte_pin(bank.context_raw)
        with self.assertRaisesRegex(ValueError, "proposal|binding"):
            bank.admit()

    def test_cleanup_unknown_or_failed_physical_result_is_refused(self):
        bank = self.fixture
        bank.finish["receipt"]["native_buffers_released"] = False
        bank.rebuild()
        with self.assertRaisesRegex(ValueError, "physical cleanup"):
            bank.admit()
        bank.finish["receipt"]["native_buffers_released"] = True
        bank.events[1]["detail"]["success"] = False
        bank.rebuild()
        with self.assertRaisesRegex(ValueError, "physical/delivered"):
            bank.admit()

    def test_nonfinite_raw_predictions_never_become_targets(self):
        bank = self.fixture
        bank.output["raw"]["divergence_logits_bits"][0] = 0x7FC00000
        bank.rebuild()
        with self.assertRaisesRegex(ValueError, "nonfinite FP32"):
            bank.admit()
        bank.output["raw"]["divergence_logits_bits"][0] = 0x42C60000  # 99 is still not supervision.
        bank.rebuild()
        batch = divergence.collate_native_divergences([bank.admit()])
        self.assertFalse(batch.divergence_mask.any())
        self.assertEqual(batch.divergence.tolist(), [[0., 0.]])
        self.assertEqual(batch.inputs.divergence_features[0, 0, 0].item(), 3.25)

    def test_maximum_fp32_decimal_roundtrips_and_overflow_is_rejected(self):
        bank = self.fixture
        bank.tensor["divergence_features"][0][0] = 3.4028235e38
        bank.rebuild()
        batch = divergence.collate_native_divergences([bank.admit()])
        self.assertEqual(struct.pack("<f", batch.inputs.divergence_features[0, 0, 0].item()), b"\xff\xff\x7f\x7f")
        self.assertEqual(struct.pack("<f", batch.inputs.divergence_features[0, 0, 1].item()), b"\x00\x00\x00\x80")
        bank.tensor["divergence_features"][0][0] = 3.5e38
        bank.rebuild()
        with self.assertRaisesRegex(ValueError, "finite FP32"):
            bank.admit()

    def test_repeated_identical_public_sources_are_one_exact_pinned_observation(self):
        bank = self.fixture
        bank.rebuild(public_rows=[bank.public, bank.public])
        checked = bank.admit()
        self.assertFalse(checked.admission["auxiliary_has_current_label"])
        self.assertTrue(checked.admission["all_target_masks_false"])

    def test_same_public_id_with_different_bytes_cannot_replace_the_pinned_source(self):
        bank = self.fixture
        changed = copy.deepcopy(bank.public)
        changed["revision"] += 1
        bank.rebuild(public_rows=[changed, changed])
        with self.assertRaisesRegex(ValueError, "public record source"):
            bank.admit()

    def test_newer_public_source_does_not_erase_the_older_pinned_bytes(self):
        bank = self.fixture
        changed = copy.deepcopy(bank.public)
        changed["revision"] += 1
        bank.rebuild(public_rows=[changed, bank.public, changed, bank.public])
        self.assertFalse(bank.admit().admission["auxiliary_has_current_label"])

    def test_public_advisory_id_does_not_supply_rules_authority(self):
        bank = self.fixture
        bank.public["origin_rules_state_sha256"] = bank.input["snapshot"]["rules_state_sha256"]
        bank.input["snapshot"]["public_records"][0]["observation_sha256"] = divergence.byte_pin(raw(bank.public))["sha256"]
        bank.rebuild()
        with self.assertRaisesRegex(ValueError, "public raw source"):
            bank.admit()

    def test_cpu_batch_budget_and_game_split_are_checked_before_allocation(self):
        checked = self.fixture.admit()
        with patch.object(divergence.torch, "tensor", side_effect=AssertionError("must reject before allocation")):
            with self.assertRaisesRegex(ValueError, "tensor budget"):
                divergence.collate_native_divergences([checked], max_tensor_bytes=1)
        with self.assertRaisesRegex(ValueError, "parent split"):
            divergence.collate_native_divergences([checked], split="validation")
        with self.assertRaisesRegex(ValueError, "batch 1..4"):
            divergence.collate_native_divergences([checked] * 5)
        batch = divergence.collate_native_divergences([checked] * 4)
        self.assertEqual(batch.inputs.board.shape, (4, 64))
        self.assertFalse(batch.policy_mask.any())

    def test_common_collection_facts_do_not_admit_an_individual_auxiliary(self):
        bank = self.fixture
        facts = divergence.verify_native_collection_authority(parent=bank.parents, parent_index=bank.index,
            artifacts=bank.artifacts, registration_bytes=bank.registration_raw, checked_source_bytes=bank.source_raw,
            export_manifest_bytes=bank.export_raw, launch_bytes=bank.launch_raw,
            expected_pins={name: bank.pins[name] for name in divergence._COMMON_PINS})
        self.assertEqual(json.loads(facts.source_json), bank.source)
        self.assertEqual(facts.parent_input_sha256, bank.pins["parent_input_sha256"])
        self.assertIn("no_individual_input", facts.scope)
        with self.assertRaises(AttributeError):
            facts.producer_json = b"{}"
        bank.context["divergence_sites"][0]["slot"] = 99
        bank.rebuild()
        with self.assertRaisesRegex(ValueError, "slot/ply"):
            bank.admit()

    def test_empty_selected_public_records_use_padding_without_parent_features(self):
        bank = self.fixture
        bank.input["snapshot"]["public_records"] = []
        bank.tensor.update(records=[], required_critical_records=[])
        bank.rebuild()
        batch = divergence.collate_native_divergences([bank.admit()])
        self.assertEqual(batch.inputs.records.shape, (1, 1, 16))
        self.assertFalse(batch.inputs.record_mask.any())
        self.assertTrue((batch.inputs.records == 0).all())
        self.assertTrue(batch.inputs.divergence_mask.all())
        self.assertFalse(batch.divergence_mask.any())

    def test_legacy_context_requires_real_capability_factory_and_keeps_derivation_scope(self):
        with tempfile.TemporaryDirectory() as root:
            bank = NativeDivergenceFixture(root, captured=False)
            with self.assertRaisesRegex(ValueError, "each legacy divergence"):
                bank.admit()
            bank.prefix_checks()
            derived = divergence.derive_legacy_context(parents=bank.parents, parent_index=bank.index,
                auxiliary_input_bytes=raw(bank.input), sidecar_bytes=raw(bank.sidecar), lineage_bytes=raw(bank.lineage),
                prefix_rules_checks=bank.checks)
            self.assertEqual(json.loads(derived), bank.context)
            bank.context_raw = derived
            bank.pins["context"] = divergence.byte_pin(derived)
            checked = bank.admit()
            self.assertEqual(checked.admission["provenance_mode"], "derived_from_legacy_prepared")
            self.assertFalse(checked.admission["auxiliary_has_current_label"])
            batch = divergence.collate_native_divergences([checked])
            self.assertFalse(batch.divergence_mask.any())
            with self.assertRaisesRegex(ValueError, "exact immutable checked Rules"):
                bank.checks = (object(), object())
                bank.admit()


if __name__ == "__main__":
    unittest.main()
