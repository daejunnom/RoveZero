"""Finite synthetic admission and frozen CPU tensor fixtures, never Rust proof.

The strict parent loader reads synthetic pinned artifacts. Registration/launch
observations and Rules descriptors below are deliberately synthetic. These
tests establish conditional byte contracts, semantic consumption and numeric
wiring, not chess replay, actual child launch, learned utility or strength.
No optimizer, backward, model export, GPU or product V is involved.
"""
import copy
from dataclasses import replace
import hashlib
import json
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from rz_pals_model import semantic_verifier as semantic
from rz_pals_model.model import initialize
from rz_pals_model.preparation_check import _parameter_digest
from rz_pals_model.training import move_components
from test_comparative_training import CandidateFixture
from test_training import move, sha


def tokens(moves, kind, legal=None):
    result = []
    for slot, movement in enumerate(moves):
        source, target, promotion = move_components(movement)
        uci = chr(97 + source % 8) + str(1 + source // 8) + chr(97 + target % 8) + str(1 + target // 8)
        if promotion:
            uci += "qrbn"[promotion - 1]
        result.append({"token_kind": kind, "slot": slot, "legal_order_slot": None if legal is None else legal.index(movement),
                       "move16": movement, "from": source, "to": target, "promotion": semantic.PROMOTIONS[promotion], "uci": uci})
    return result


class SemanticFixture:
    """Shared actual-adapter fixture for DG03; caller facts are synthetic."""
    def __init__(self, root):
        bank = CandidateFixture(root, roles=("proposer",))
        self.parents = bank.parents
        self.index = self.parents.current_view.current_indices[0]
        row = self.parents.records[self.index]
        snapshot = row["input"]["snapshot"]
        self.snapshot = copy.deepcopy(snapshot)
        self.raws = {"source": b"// synthetic registered semantic Rust source; not a launched binary"}
        self.registration = {"schema": semantic.REGISTRATION_SCHEMA, "semantic_schema": semantic.SEMANTIC_SCHEMA,
                             "implementation": semantic.IMPLEMENTATION, "source_artifact": semantic.byte_pin(self.raws["source"]),
                             "binary_sha256": sha("fixture semantic executable"), "rules_version": "rz-position/0.1.0",
                             "platform": "windows", "accepted_binary_pin_scope": "current_exe_path_hash",
                             "assurance_scope": "independently_registered_caller_source"}
        self.common = {"schema": semantic.COMMON_SCHEMA, "parent_input_sha256": row["input"]["sha256"],
                       "current_view_sha256": self.parents.current_view.sha256, "question": "unrestricted_recheck",
                       "prefix": [], "root_moves": [], "claimed_line": [], "allowed_tasks": ["resume_task", "defer"],
                       "baseline_depth": 2, "requested_depth": 4, "max_nodes_per_check": 100000,
                       "max_wall_time_ms": 10000, "budget_bucket": 1, "cpu_profile_sha256": sha("fixture fixed H4 profile"),
                       "known_completed_depth": 2}
        root_descriptor = {"schema": semantic.DESCRIPTOR_SCHEMA, "rules_version": self.registration["rules_version"],
                           "rules_variant": "standard_chess", "rules_state_sha256": snapshot["rules_state_sha256"],
                           "rules_history_sha256": snapshot["rules_history_sha256"], "board_fen": snapshot["board_fen"],
                           "side_to_move": "white", "board64_piece_codes": list(self.parents.encodings[row["input"]["sha256"]].board),
                           "piece_code_semantics": semantic.PIECES, "pals_rules_encoding": "rz-pals-rules-fields-v1",
                           "castling_rights": 0, "castling_bit_semantics": semantic.CASTLING, "en_passant_square": None,
                           "halfmove_clock": 0, "fullmove_number": 1, "in_check": False,
                           "history_completeness": "unknown_prefix", "history_origin": "fen", "known_history_positions": 1,
                           "known_repetition_count": 1, "repetition_history_complete": False, "repetition_scope": semantic.REPETITION,
                           "legal_moves": list(snapshot["legal_moves"]), "legal_order_sha256": semantic.digest(semantic.MOVE_DOMAIN, snapshot["legal_moves"]),
                           "legal_tokens": tokens(snapshot["legal_moves"], "root_legal", snapshot["legal_moves"]),
                           "play_status": "ongoing", "terminal_reason": None, "terminal_winner": None, "terminal_source": None}
        target = copy.deepcopy(root_descriptor)
        target["legal_tokens"] = tokens(snapshot["legal_moves"], "target_legal", snapshot["legal_moves"])
        self.receipt = {"schema": semantic.SEMANTIC_SCHEMA, "implementation": semantic.IMPLEMENTATION,
                        "context_sha256": "0" * 64, "current_binary_sha256": self.registration["binary_sha256"],
                        "binary_pin_scope": self.registration["accepted_binary_pin_scope"], "parent_input_sha256": row["input"]["sha256"],
                        "before_result_anchor_sha256": "0" * 64, "caller_declaration_scope": semantic.DECLARATION_SCOPE,
                        "meaning_scope": semantic.MEANING_SCOPE, "question": self.common["question"], "root": root_descriptor,
                        "prefix": [], "target": target, "root_restriction": None,
                        "claimed_line": {"status": "no_claim", "legality_verified": None, "claim_truth": "unknown",
                                         "movements": [], "restriction_checked": False, "final_state": None},
                        "branch_meaning_sha256": "0" * 64,
                        "resource_policy": {"max_wall_time_ms": 10000, "max_output_bytes": 65536,
                                            "max_prefix_plies": 64, "max_claim_plies": 64, "max_root_moves": 256, "cpu_checks": 0},
                        "cpu_checks": 0, "search_executed": False, "model_executed": False, "training_target_created": False,
                        "product_verifier_enabled": False, "elapsed_ms": 8, "deadline_exceeded": False}
        self.reseal()
        self.checked = self.admit()

    def reseal(self):
        self.raws["registration"] = semantic.canonical(self.registration)
        self.raws["common_query"] = semantic.canonical(self.common)
        self.before = {"schema": semantic.BEFORE_SCHEMA, "parent_input_sha256": self.common["parent_input_sha256"],
                       "current_view_sha256": self.common["current_view_sha256"], "common_query_sha256": semantic.byte_pin(self.raws["common_query"])["sha256"],
                       "registration_sha256": semantic.byte_pin(self.raws["registration"])["sha256"]}
        self.raws["before_result"] = semantic.canonical(self.before)
        snapshot = self.snapshot
        self.request = {"schema": semantic.SEMANTIC_SCHEMA, "question": self.common["question"],
                        "parent_input_sha256": self.common["parent_input_sha256"],
                        "before_result_anchor_sha256": semantic.byte_pin(self.raws["before_result"])["sha256"],
                        "position_command": snapshot["position_command"], "expected_board_fen": snapshot["board_fen"],
                        "rules_state_sha256": snapshot["rules_state_sha256"], "rules_history_sha256": snapshot["rules_history_sha256"],
                        "prefix": list(self.common["prefix"]), "root_moves": list(self.common["root_moves"]),
                        "claimed_line": list(self.common["claimed_line"]), "current_binary_sha256": self.registration["binary_sha256"],
                        "max_wall_time_ms": self.common["max_wall_time_ms"], "max_output_bytes": 65536}
        self.request["context_sha256"] = semantic.digest(semantic.SEMANTIC_SCHEMA, self.request)
        self.raws["request"] = semantic.canonical(self.request)
        for key in ("question", "context_sha256", "parent_input_sha256", "before_result_anchor_sha256", "current_binary_sha256"):
            self.receipt[key] = self.request[key]
        self.receipt["prefix"] = tokens(self.common["prefix"], "prefix")
        self.receipt["resource_policy"]["max_wall_time_ms"] = self.common["max_wall_time_ms"]
        self.receipt["branch_meaning_sha256"] = semantic.digest(semantic.MEANING_DOMAIN, {
            key: self.receipt[key] for key in ("question", "root", "prefix", "target", "root_restriction", "claimed_line")})
        self.raws["receipt"] = semantic.canonical(self.receipt) + b"\n"
        launch = {"schema": semantic.LAUNCH_SCHEMA, "registration_sha256": semantic.byte_pin(self.raws["registration"])["sha256"],
                  "request": semantic.byte_pin(self.raws["request"]), "receipt": semantic.byte_pin(self.raws["receipt"]),
                  "binary_sha256": self.registration["binary_sha256"], "platform": self.registration["platform"],
                  "binary_pin_scope": self.registration["accepted_binary_pin_scope"],
                  "before_result_anchor_sha256": self.request["before_result_anchor_sha256"],
                  "assurance_scope": "independently_pinned_caller_observation", "anchor_durable_before_spawn": True,
                  "spawned": True, "reaped": True, "exit_code": 0, "elapsed_ms": 12, "timed_out": False}
        self.raws["launch_observation"] = semantic.canonical(launch)
        self.pins = {name: semantic.byte_pin(raw) for name, raw in self.raws.items()}
        self.pins.update(parent_input_sha256=self.common["parent_input_sha256"], current_view_sha256=self.parents.current_view.sha256,
                         frozen_admission_sha256=self.parents._frozen_admission_identity,
                         encoding_sha256=self.parents.encodings[self.common["parent_input_sha256"]].encoding_sha256)

    def admit(self):
        return semantic.admit_semantic_input(parent=self.parents, parent_index=self.index,
                   **{name + "_bytes": self.raws[name] for name in ("request", "receipt", "source", "registration", "before_result", "launch_observation", "common_query")},
                   expected_pins=self.pins)

    def restrict(self, order):
        self.common.update(question="restricted_response", root_moves=list(order), allowed_tasks=["widen_responses", "defer"])
        # Strict subset enables Widen; Defer requires empty controls, so only
        # Widen is actually eligible. This fixture may instead use nonempty
        # prefix to enable defend_response when all target roots are chosen.
        if len(order) >= len(self.snapshot["legal_moves"]):
            self.common.update(prefix=[self.snapshot["legal_moves"][0]], allowed_tasks=["defend_response"])
        legal = self.receipt["target"]["legal_moves"]
        effective = [value for value in legal if value in order]
        self.receipt["root_restriction"] = {"declared_order": tokens(order, "root_restriction", legal),
                                            "effective_legal_order": tokens(effective, "effective_root_legal", legal),
                                            "declared_order_sha256": semantic.digest(semantic.MOVE_DOMAIN, list(order)),
                                            "effective_order_sha256": semantic.digest(semantic.MOVE_DOMAIN, effective),
                                            "scope": "exact_target_root_only;request_order_preserved;effective_order_is_rules_order"}
        self.reseal()

    def claim(self, movements):
        self.common["claimed_line"] = list(movements)
        end = copy.deepcopy(self.receipt["target"])
        end["rules_state_sha256"] = sha("synthetic claim-end state " + str(movements))
        end["rules_history_sha256"] = sha("synthetic claim-end history " + str(movements))
        end["legal_tokens"] = tokens(end["legal_moves"], "claim_end_legal", end["legal_moves"])
        self.receipt["claimed_line"] = {"status": "legal_continuation", "legality_verified": True, "claim_truth": "unknown",
                                        "movements": tokens(movements, "claimed_continuation"),
                                        "restriction_checked": bool(self.common["root_moves"]), "final_state": end}
        self.reseal()


def frozen_wrapper(model):
    observation = semantic.canonical({"schema": semantic.PARAMETER_SCHEMA, "checkpoint_sha256": sha("synthetic frozen checkpoint bytes"),
                                      "parameter_sha256": _parameter_digest(model),
                                      "assurance_scope": "independently_pinned_caller_parameter_observation"})
    return semantic.FrozenSemanticVerifier(model, parameter_observation_bytes=observation,
                                          expected_observation_sha256=hashlib.sha256(observation).hexdigest())


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = SemanticFixture(self.temp.name)

    def test_actual_adapter_current_legacy_derivation_and_copy_views(self):
        checked = self.fixture.checked
        self.assertIs(checked.verify(), checked)
        self.assertEqual(checked.verify_parent(self.fixture.parents), self.fixture.index)
        common = checked.common_query()
        self.assertEqual(common["question"], "unrestricted_recheck")
        self.assertEqual(common["query"][:7], [0., 0., 0., 0., 1., 0., 1.])
        self.assertEqual(common["query"][-2:], [1., 0.])
        self.assertEqual(checked.encoded_snapshot().board, self.fixture.parents.encodings[common["parent_input_sha256"]].board)
        self.assertIsNone(checked.derived_input()["future_label"])
        common["query"][0] = 99
        self.assertEqual(checked.common_query()["query"][0], 0.)
        self.assertEqual(len(checked.sha256), 64)
        with self.assertRaises(ValueError):
            semantic.CheckedSemanticInput()
        with self.assertRaises(AttributeError):
            checked._identity = "0" * 64

    def test_receipt_self_reseal_without_independent_pin_is_refused(self):
        value = json.loads(self.fixture.raws["receipt"])
        value["target"]["known_history_positions"] = 3
        value["branch_meaning_sha256"] = semantic.digest(semantic.MEANING_DOMAIN, {
            key: value[key] for key in ("question", "root", "prefix", "target", "root_restriction", "claimed_line")})
        self.fixture.raws["receipt"] = semantic.canonical(value)
        with self.assertRaisesRegex(ValueError, "independent expected pin"):
            self.fixture.admit()

    def test_current_raw_parent_mutation_rechecked_at_each_use(self):
        self.fixture.parents.records[self.fixture.index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaisesRegex(ValueError, "immutable raw"):
            self.fixture.checked.verify()

    def test_missing_history_or_mismatched_move_tokens_rejected(self):
        del self.fixture.receipt["target"]["known_repetition_count"]
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "exact semantic fields"):
            self.fixture.admit()

    def test_empty_prefix_must_match_full_state_history_and_order(self):
        self.fixture.receipt["target"]["known_history_positions"] = 3
        self.fixture.receipt["target"]["known_repetition_count"] = 2
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "empty prefix"):
            self.fixture.admit()

    def test_equal_board_different_known_history_is_not_reused(self):
        self.fixture.common.update(prefix=[move(12, 20)], question="continuation_challenge", allowed_tasks=["attack_repair"])
        target = self.fixture.receipt["target"]
        target.update(rules_history_sha256=sha("synthetic distinct known history"), known_history_positions=3, known_repetition_count=2)
        self.fixture.reseal()
        features = semantic.semantic_features(self.fixture.admit())
        self.assertEqual(features.states[0].board, features.states[1].board)
        self.assertIsNot(features.states[0], features.states[1])
        self.assertNotEqual(features.states[0].history, features.states[1].history)

    def test_high_counter_bits_and_ep_presence_are_real_scalar_features(self):
        descriptor = copy.deepcopy(self.fixture.receipt["root"])
        descriptor.update(halfmove_clock=65536, fullmove_number=2**32 - 1, en_passant_square=0)
        features = semantic.prepare_state_features(descriptor)
        self.assertEqual(features.fen[5:8], (1., 0., 0.))
        self.assertEqual(features.fen[8:12], (0., 1 / 65535, 1., 1.))
        self.assertNotEqual(features.fen, semantic.prepare_state_features(self.fixture.receipt["root"]).fen)

    def test_no_claim_unknown_mask_and_matched_claim_end(self):
        features = semantic.semantic_features(self.fixture.checked)
        self.assertIsNone(features.states[2])
        self.assertIs(features.states[0], features.states[1])
        self.assertEqual(features.claim_mode, 0)
        self.fixture.claim([move(12, 20)])
        admitted = self.fixture.admit()
        features = semantic.semantic_features(admitted)
        self.assertIsNotNone(features.states[2])
        self.assertEqual(features.claim_mode, 1)
        self.fixture.receipt["claimed_line"]["claim_truth"] = "proven"
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "tactical truth"):
            self.fixture.admit()

    def test_move_components_promotion_and_order_validate(self):
        legal = self.fixture.snapshot["legal_moves"]
        self.fixture.restrict(legal)
        checked = self.fixture.admit()
        self.assertEqual(semantic.semantic_features(checked).restriction, tuple(tuple(move_components(value)) for value in legal))
        self.fixture.receipt["root_restriction"]["declared_order"][0]["promotion"] = "knight"
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "Move16 meaning"):
            self.fixture.admit()

    def test_independent_launch_elapsed_cannot_be_smaller_than_receipt(self):
        launch = json.loads(self.fixture.raws["launch_observation"])
        launch["elapsed_ms"] = 7
        self.fixture.raws["launch_observation"] = semantic.canonical(launch)
        self.fixture.pins["launch_observation"] = semantic.byte_pin(self.fixture.raws["launch_observation"])
        with self.assertRaises(ValueError):
            self.fixture.admit()

    def test_module_argument_comparison_cannot_claim_cli_image_observation(self):
        self.fixture.receipt["binary_pin_scope"] = semantic.BINARY_SCOPE
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "Rules-only"):
            self.fixture.admit()

    def test_no_cpu_model_target_and_unknown_history(self):
        self.assertEqual(self.fixture.checked.rules_receipt()["cpu_checks"], 0)
        features = semantic.semantic_features(self.fixture.checked)
        self.assertEqual(features.states[0].history[:2], (0., 1.))
        self.fixture.receipt["search_executed"] = True
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "Rules-only"):
            self.fixture.admit()

    def test_resource_policy_boolean_cannot_alias_integer_count(self):
        self.fixture.receipt["resource_policy"]["cpu_checks"] = False
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "resource policy mismatch"):
            self.fixture.admit()


class FrozenNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = initialize(31)
        for parameter in cls.model.parameters():
            parameter.requires_grad_(False)
        cls.wrapper = frozen_wrapper(cls.model)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = SemanticFixture(self.temp.name)

    def test_tiled_kv_matches_small_fulltoken_reference_and_padding(self):
        features = semantic.semantic_features(self.fixture.checked)
        # Independent small 19-token oracle, including inactive claim padding.
        descriptors = (tuple(semantic.token_descriptors(features))[:19],)
        payload, mask = semantic._tile(self.model, descriptors)
        reference = torch.zeros_like(payload)
        expected_mask = torch.zeros_like(mask)
        with torch.no_grad():
            for slot, (segment, position, kind, value) in enumerate(descriptors[0]):
                if value is None:
                    continue
                expected_mask[0, slot] = True
                vector = torch.zeros(384)
                vector[position] = 1.
                vector[256 + segment] = 1.
                if kind == "question":
                    question, prior_depth = value
                    vector[304 + question] += 1.
                    vector[310] += prior_depth / 64
                elif kind == "claim_mode":
                    vector[307 + value] += 1.
                elif kind == "board":
                    vector += self.model.public_encoder.piece_embedding(torch.tensor(value)) + self.model.public_encoder.square_embedding[position]
                reference[0, slot] = vector
            self.assertTrue(torch.equal(mask, expected_mask))
            self.assertTrue(torch.equal(payload, reference))
            tiled = torch.cat([self.model.public_encoder.memory_key(payload[:, at:at + 7]) for at in range(0, 19, 7)], dim=1)
            whole = self.model.public_encoder.memory_key(reference)
            self.assertTrue(torch.allclose(tiled, whole, atol=2e-6, rtol=2e-6))
        empty_claim = tuple(semantic.token_descriptors(features))[674:1010]
        payload, mask = semantic._tile(self.model, (empty_claim[:64],))
        self.assertFalse(bool(mask.any()))
        self.assertEqual(float(payload.abs().sum()), 0.)

    def test_order_promotion_question_and_history_affect_actual_tensors(self):
        base = semantic.semantic_features(self.fixture.checked)
        moves = ((48, 56, 1), (49, 57, 4))
        variants = (replace(base, prefix=moves), replace(base, prefix=tuple(reversed(moves))),
                    replace(base, prefix=((48, 56, 2), moves[1])), replace(base, question=1))
        def payload(feature, start):
            descriptors = tuple(semantic.token_descriptors(feature))[start:start + 2]
            return semantic._tile(self.model, (descriptors,))[0]
        self.assertFalse(torch.equal(payload(variants[0], 1010), payload(variants[1], 1010)))
        self.assertFalse(torch.equal(payload(variants[0], 1010), payload(variants[2], 1010)))
        self.assertFalse(torch.equal(payload(base, 0), payload(variants[3], 0)))
        self.assertFalse(torch.equal(payload(base, 0), payload(replace(base, known_completed_depth=3), 0)))
        changed = replace(base.states[0], history=(1., 0., 2 / 65535, 0.))
        changed_features = replace(base, states=(changed, base.states[1], None))
        self.assertFalse(torch.equal(payload(base, 78), payload(changed_features, 78)))

    def test_checked_budget_refuses_before_any_tensor_or_model_call(self):
        self.assertLessEqual(semantic.reservation_bytes(4, 194), semantic.MAX_SEMANTIC_BYTES)
        with patch.object(semantic, "_tensor_inputs", side_effect=AssertionError("allocation happened")):
            with self.assertRaisesRegex(ValueError, "reservation denied"):
                self.wrapper([self.fixture.checked], deadline=time.monotonic() + 30, byte_limit=1)
        with self.assertRaises(ValueError):
            semantic.reservation_bytes(5, 194)
        with self.assertRaises(TimeoutError):
            self.wrapper([self.fixture.checked], deadline=time.monotonic() - 1)

    def test_cpu_autocast_refused_before_tensor_or_model_execution(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with patch.object(semantic, "_tensor_inputs", side_effect=AssertionError("allocation happened")):
                with self.assertRaisesRegex(ValueError, "autocast"):
                    self.wrapper([self.fixture.checked], deadline=time.monotonic() + 30)

    def test_frozen_forward_seven_head_and_public_parameters_unchanged(self):
        before = _parameter_digest(self.model)
        names = tuple(self.model.state_dict())
        public_inputs = semantic._tensor_inputs([self.fixture.checked])
        with torch.no_grad():
            public_before = tuple(value.clone() for value in self.model.public_encoder(*public_inputs.public_args()))
        logits, latent, audit = self.wrapper([self.fixture.checked], deadline=time.monotonic() + 120)
        self.assertEqual(tuple(logits.shape), (1, 7))
        self.assertEqual(tuple(latent.shape), (1, 16, 384))
        self.assertFalse(logits.requires_grad)
        self.assertFalse(audit["actual_training_executed"])
        self.assertFalse(audit["product_verifier_enabled"])
        self.assertEqual(audit["private_tokens"], 1394)
        self.assertEqual(dict(self.wrapper.state_dict()), {})
        self.assertEqual(tuple(self.wrapper.parameters()), ())
        self.assertEqual(tuple(self.model.state_dict()), names)
        self.assertEqual(_parameter_digest(self.model), before)
        with torch.no_grad():
            public_after = self.model.public_encoder(*public_inputs.public_args())
        for old, new in zip(public_before, public_after):
            self.assertTrue(torch.equal(old, new))
        # Same parent/public/query/budget, only the actual semantic question
        # changes. This tests consumption by the existing V head, not quality.
        self.fixture.common["question"] = "continuation_challenge"
        self.fixture.reseal()
        alternative = self.fixture.admit()
        self.assertEqual(alternative.common_query()["query"], self.fixture.checked.common_query()["query"])
        other_logits, _, _ = self.wrapper([alternative], deadline=time.monotonic() + 120)
        self.assertFalse(torch.equal(logits, other_logits))

    def test_missing_validator_and_parameter_change_are_refused(self):
        observation = semantic.canonical({"schema": semantic.PARAMETER_SCHEMA, "checkpoint_sha256": sha("fixture checkpoint"),
                                          "parameter_sha256": "0" * 64,
                                          "assurance_scope": "independently_pinned_caller_parameter_observation"})
        # No deepcopy/second full model is needed to test the missing expert.
        expert = self.model.experts.pop("validator")
        try:
            with self.assertRaisesRegex(ValueError, "validator required"):
                semantic.FrozenSemanticVerifier(self.model, parameter_observation_bytes=observation,
                                                expected_observation_sha256=semantic.byte_pin(observation)["sha256"])
        finally:
            self.model.experts["validator"] = expert
        with self.assertRaisesRegex(ValueError, "parameter bytes changed"):
            semantic.FrozenSemanticVerifier(self.model, parameter_observation_bytes=observation,
                                            expected_observation_sha256=semantic.byte_pin(observation)["sha256"])


if __name__ == "__main__":
    unittest.main()
