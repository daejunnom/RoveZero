"""Focused frozen action numeric fixtures, not live CPU/witness/training proof.

QueryFixture uses the existing strict parent and semantic factories with
synthetic caller/Rules/source/launch observations. One test below runs the
actual frozen CPU model when the central executor runs this module; other
mocked forward callbacks exercise refusal/lifetime boundaries only. No checker
child, utility criterion, target, backward, optimizer or product V is created.
"""

from dataclasses import replace
import itertools
import math
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from rz_pals_model import semantic_verifier as semantic
from rz_pals_model import strategic_verifier_query as query
from rz_pals_model import strategic_verifier_scoring as scoring
from rz_pals_model import verifier_producer as legacy
from rz_pals_model.config import ModelConfig
from rz_pals_model.model import initialize
from rz_pals_model.training import _parameter_digest
from test_strategic_verifier_query import QueryFixture


def frozen_wrapper(model):
    observation = query.canonical({"schema": semantic.PARAMETER_SCHEMA,
        "checkpoint_sha256": "9" * 64, "parameter_sha256": _parameter_digest(model),
        "assurance_scope": "independently_pinned_caller_parameter_observation"})
    return scoring.FrozenStrategicVerifierScorer(model, parameter_observation_bytes=observation,
                                               expected_observation_sha256=query.byte_pin(observation)["sha256"])


def numeric_stub(checked_inputs, features, *, deadline):
    # Synthetic numeric seam: no actual validator invocation in these tests.
    return torch.tensor([feature.task / 8 for feature in features], dtype=torch.float32)


class StrategicScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = initialize(83)
        for parameter in cls.model.parameters():
            parameter.requires_grad_(False)
        cls.wrapper = frozen_wrapper(cls.model)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = QueryFixture(self.temp.name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    def checked(self, actions=None):
        if actions is not None:
            self.f.actions = actions
            self.f.refresh()
        return self.f.admit()

    def run_stub(self, checked=None, **kwargs):
        checked = self.checked() if checked is None else checked
        with patch.object(self.wrapper, "_forward_batch", side_effect=numeric_stub):
            return self.wrapper(checked, deadline=time.monotonic() + 60, **kwargs)

    def header(self, feature):
        return scoring._tile(self.model, ((('action', feature),),))[0]

    def test_wrapper_has_no_registered_parameters_or_state(self):
        self.assertEqual(dict(self.wrapper.named_parameters()), {})
        self.assertEqual(dict(self.wrapper.named_buffers()), {})
        self.assertEqual(dict(self.wrapper.state_dict()), {})
        self.assertIs(self.wrapper._model, self.model)
        self.assertIs(self.wrapper._guard._model, self.model)
        self.assertFalse(hasattr(self.wrapper, "collate"))
        self.assertFalse(hasattr(self.wrapper, "select_action"))

    def test_actual_frozen_forward_finite_all_actions_without_update(self):
        checked = self.checked()
        before = _parameter_digest(self.model)
        raw_before = checked.raw_assets()
        result, audit = self.wrapper(checked, deadline=time.monotonic() + 120)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad_fn)
        self.assertEqual(_parameter_digest(self.model), before)
        self.assertEqual(checked.raw_assets(), raw_before)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(not module.training for module in self.model.modules()))
        self.assertEqual(audit["actions"], 2)
        self.assertEqual(audit["action_tasks"], ("resume_task", "defer"))
        self.assertEqual(audit["private_tokens"], 4483)
        self.assertEqual(audit["parameter_sha256"], before)
        self.assertEqual(audit["legacy_query_seed"], "all_zero")
        for key in ("utility_authority", "target_authority", "training_authority", "learned_utility_claim",
                    "action_execution_authority", "strategic_strength_claim", "past_input_equivalence_claim",
                    "product_verifier_enabled", "actual_training_executed", "backward_executed", "private_warm_continuation",
                    "source_authority", "build_authority", "binary_loaded_image_verified_here"):
            self.assertIs(audit[key], False)

    def test_legacy_query_seed_zero_preserves_original_sealed_public_input(self):
        checked = self.checked()
        semantics = checked.action_semantic_inputs()
        snapshots = tuple(value.encoded_snapshot() for value in semantics)
        original = semantic._tensor_inputs(semantics)
        actual = scoring._tensor_inputs_zero_seed(semantics)
        self.assertTrue(bool((actual.query == 0).all()))
        self.assertTrue(bool((original.query != 0).any()))
        for name in ("board", "metadata", "records", "record_mask", "candidates", "candidate_mask"):
            self.assertTrue(torch.equal(getattr(actual, name), getattr(original, name)))
        self.assertEqual(tuple(value.encoded_snapshot() for value in semantics), snapshots)

    def test_layout_extent_padding_and_real_action_header(self):
        feature = self.checked().features()[0]
        descriptors = tuple(scoring.token_descriptors(feature))
        self.assertEqual(len(descriptors), scoring.PRIVATE_TOKENS)
        self.assertEqual(descriptors[semantic.TOKEN_COUNT][0], "action")
        tile, mask = scoring._tile(self.model, (descriptors[semantic.TOKEN_COUNT:semantic.TOKEN_COUNT + 3],))
        self.assertEqual(mask.tolist(), [[True, False, False]])
        self.assertEqual(float(tile[0, 0, 340]), 1.)
        self.assertEqual(float(tile[0, 0, 4]), 1.)
        self.assertEqual(float(tile[0, 0, 21]), 1.)
        self.assertAlmostEqual(float(tile[0, 0, 32]), 2 / 64)
        self.assertAlmostEqual(float(tile[0, 0, 33]), 4 / 64)
        self.assertEqual(float(tile[0, 1:].abs().sum()), 0.)

    def test_actual_options_change_numeric_payload(self):
        self.f.profiles.append(self.f.profile(tt=32, q=5))
        checked = self.checked([self.f.action("resume_task"), self.f.action("resume_task", 1)])
        first, second = checked.features()
        self.assertFalse(torch.equal(self.header(first), self.header(second)))
        self.assertAlmostEqual(float(self.header(second)[0, 0, 36]), math.log2(33) / 21)
        self.assertAlmostEqual(float(self.header(second)[0, 0, 37]), 5 / 32)

    def test_actual_node_allowance_changes_feature_without_slot_feature(self):
        checked = self.checked([self.f.action("resume_task", nodes=100000), self.f.action("resume_task", nodes=99999)])
        first, second = checked.features()
        self.assertFalse(torch.equal(self.header(first), self.header(second)))
        self.assertNotEqual(first.numeric_controls()[2], second.numeric_controls()[2])

    def test_catalogue_recipe_slot_and_bucket_never_enter_payload(self):
        first = self.checked([self.f.action("resume_task", bucket=1)])
        first_features = first.features()[0]
        first_raw = first.raw_assets()["catalogue"]
        self.f.actions = [self.f.action("resume_task", bucket=2)]
        self.f.refresh()
        self.f.catalogue["recipe_id"] = "별도 선언 이름"
        self.f.catalogue_raw = query.canonical(self.f.catalogue)
        self.f.before["catalogue"] = query.byte_pin(self.f.catalogue_raw)
        self.f.before_raw = query.canonical(self.f.before)
        self.f.pins["catalogue"] = query.byte_pin(self.f.catalogue_raw)
        self.f.pins["before_result"] = query.byte_pin(self.f.before_raw)
        second = self.f.admit()
        self.assertNotEqual(first_raw, second.raw_assets()["catalogue"])
        self.assertEqual(first_features, second.features()[0])
        self.assertTrue(torch.equal(self.header(first_features), self.header(second.features()[0])))

    def test_profile_name_is_admission_metadata_not_numeric_feature(self):
        first = self.checked([self.f.action("resume_task")]).features()[0]
        self.f.profiles = [self.f.profile(legacy.RECHECK_PROFILE)]
        second = self.checked([self.f.action("resume_task")]).features()[0]
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(self.header(first), self.header(second)))

    def test_defer_cpu_options_are_not_executed_features(self):
        first = self.checked([self.f.action("defer", nodes=100000, wall=10000)]).features()[0]
        self.f.profiles = [self.f.profile(tt=32, q=6)]
        second = self.checked([self.f.action("defer", nodes=99999, wall=9999)]).features()[0]
        self.assertFalse(first.cpu_check)
        self.assertEqual(first.node_budget, 0)
        self.assertEqual(first.numeric_controls(), (0.,) * 7)
        self.assertTrue(torch.equal(self.header(first), self.header(second)))

    def test_prior_unknown_optional_costs_and_order_are_explicit(self):
        branch = self.checked().features()[0]
        prior = query.PriorFeatures(0, 4, "missing", False, (), (), (), None, None, None, 0)
        payload, mask = scoring._tile(self.model, ((('prior', (0, prior)), ('prior', (1, prior)), ('prior', (2, None))),))
        self.assertEqual(mask.tolist(), [[True, True, False]])
        self.assertEqual(float(payload[0, 0, 14]), 1.)
        self.assertEqual(float(payload[0, 0, 15]), 0.)
        self.assertEqual(float(payload[0, 0, 50] + payload[0, 0, 52] + payload[0, 0, 54]), 0.)
        self.assertEqual(float(payload[0, 0, 320]), 1.)
        self.assertEqual(float(payload[0, 1, 321]), 1.)
        self.assertEqual(float(payload[0, 2].abs().sum()), 0.)
        # Arbitrary ordinal is not a feature. Observed order determines layout.
        first = replace(branch, prior=(prior,))
        second = replace(branch, prior=(replace(prior, ordinal=9),))
        a = tuple(itertools.islice(scoring.token_descriptors(first), semantic.TOKEN_COUNT + 1, semantic.TOKEN_COUNT + 2))
        b = tuple(itertools.islice(scoring.token_descriptors(second), semantic.TOKEN_COUNT + 1, semantic.TOKEN_COUNT + 2))
        self.assertTrue(torch.equal(scoring._tile(self.model, (a,))[0], scoring._tile(self.model, (b,))[0]))

    def test_actual_partial_prior_uses_existing_factory_and_stays_unknown(self):
        self.f.add_prior(state="partial", partial=True)
        self.f.refresh()
        feature = self.f.admit().features()[0]
        prior = feature.prior[0]
        self.assertFalse(prior.known)
        self.assertEqual(prior.state, "partial")
        self.assertIsNotNone(prior.reported_nodes)
        payload, mask = scoring._tile(self.model, ((('prior', (0, prior)),),))
        self.assertTrue(bool(mask[0, 0]))
        self.assertEqual(float(payload[0, 0, 15]), 0.)
        self.assertEqual(float(payload[0, 0, 50]), 1.)

    def test_pv_actual_move_embedding_order_and_promotion(self):
        move = (48, 56, 1)
        descriptor = ('pv', (2, 0, 0, move))
        payload, mask = scoring._tile(self.model, ((descriptor, ('pv', (2, 0, 1, None))),))
        with torch.no_grad():
            expected = self.model.move_embedding(torch.tensor([[move]], dtype=torch.int64))[0, 0].clone()
            expected[0] += 1.
            expected[273] += 1.
            expected[322] += 1.
            expected[342] += 1.
        self.assertTrue(torch.allclose(payload[0, 0], expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(mask.tolist(), [[True, False]])
        self.assertEqual(float(payload[0, 1].abs().sum()), 0.)
        changed = scoring._tile(self.model, ((('pv', (2, 0, 0, (48, 56, 4))),),))[0]
        later = scoring._tile(self.model, ((('pv', (2, 0, 1, move)),),))[0]
        self.assertFalse(torch.equal(payload[:, :1], changed))
        self.assertFalse(torch.equal(payload[:, :1], later))

    def test_whole_catalogue_reserved_before_first_batch(self):
        checked = self.checked([self.f.action("resume_task", nodes=100000 + index) for index in range(8)])
        plan = scoring.reservation_plan(checked, config=self.model.config, owner_byte_limit=scoring.MAX_OWNER_BYTES)
        self.assertEqual(plan.actions, 8)
        self.assertEqual(tuple(value.actions for value in plan.batches), (4, 4))
        self.assertEqual(plan.owner_bytes, sum(value.owner_bytes for value in plan.batches) + 32)
        self.assertEqual(plan.matmul_flops, sum(value.matmul_flops for value in plan.batches))
        calls = []
        def observed(inputs, features, *, deadline):
            calls.append(len(features))
            self.assertEqual(plan.actions, 8)
            return numeric_stub(inputs, features, deadline=deadline)
        with patch.object(self.wrapper, "_forward_batch", side_effect=observed):
            result, audit = self.wrapper(checked, deadline=time.monotonic() + 60, owner_byte_limit=scoring.MAX_OWNER_BYTES)
        self.assertEqual(calls, [4, 4])
        self.assertEqual(result.shape, (8,))
        self.assertEqual(audit["owner_reserved_bytes"], plan.owner_bytes)
        self.assertGreater(dict(plan.batches[0].components)["cross_attention_repeated_gqa_kv"], 0)

    def test_byte_and_flop_reservation_fail_before_tensor_or_nn_allocation(self):
        checked = self.checked()
        for kwargs in ({"owner_byte_limit": 1}, {"matmul_flop_limit": 1}):
            with self.subTest(kwargs=kwargs), patch.object(scoring, "_tensor_inputs_zero_seed", side_effect=AssertionError("no tensor")), \
                    patch.object(self.wrapper, "_forward_batch", side_effect=AssertionError("no NN")):
                with self.assertRaisesRegex(ValueError, "reservation denied"):
                    self.wrapper(checked, deadline=time.monotonic() + 60, **kwargs)

    def test_batch_and_credit_exact_types_and_fixed_config(self):
        checked = self.checked()
        for kwargs in ({"batch_size": 0}, {"batch_size": 5}, {"batch_size": True},
                       {"owner_byte_limit": True}, {"matmul_flop_limit": math.inf}):
            with self.subTest(kwargs=kwargs), patch.object(self.wrapper, "_forward_batch", side_effect=AssertionError("no NN")):
                with self.assertRaises(ValueError):
                    self.wrapper(checked, deadline=time.monotonic() + 60, **kwargs)
        with self.assertRaises(ValueError):
            scoring.reservation_plan(checked, config=replace(ModelConfig(), iterations=3))

    def test_actual_task_head_gather_uses_task_not_catalogue_order(self):
        checked = self.checked()
        features = checked.features()
        inputs = checked.action_semantic_inputs()
        logits = torch.arange(14, dtype=torch.float32).reshape(2, 7)
        public = (torch.zeros(2, 2, 67, 64), torch.zeros(2, 2, 67, 64), torch.ones(2, 67, dtype=torch.bool))
        outputs = (torch.zeros(2, 1), torch.zeros(2, 3), torch.zeros(2, 16, 384), logits)
        with patch.object(self.model.public_encoder, "forward", return_value=public), \
                patch.object(scoring, "_private_memory", return_value=public), \
                patch.object(self.model.experts["validator"], "forward", return_value=outputs):
            result = self.wrapper._forward_batch(inputs, features, deadline=time.monotonic() + 60)
        self.assertEqual(result.tolist(), [4., 13.])

    def test_bad_deadline_refused_before_query_view_or_tensor_or_nn(self):
        for value in (math.nan, math.inf, -math.inf, True, "tomorrow", None, time.monotonic() - 1):
            with self.subTest(value=value), patch.object(scoring, "reservation_plan", side_effect=AssertionError("no query view")), \
                    patch.object(self.wrapper, "_forward_batch", side_effect=AssertionError("no NN")):
                with self.assertRaises(TimeoutError):
                    self.wrapper(object(), deadline=value)

    def test_original_deadline_post_forward_expiry_refuses_second_batch(self):
        checked = self.checked([self.f.action("resume_task", nodes=100000 + index) for index in range(5)])
        now = [100.]
        calls = []
        def late(inputs, features, *, deadline):
            calls.append(deadline)
            now[0] = 102.
            return numeric_stub(inputs, features, deadline=deadline)
        with patch.object(scoring.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(self.wrapper, "_forward_batch", side_effect=late):
            with self.assertRaises(TimeoutError):
                self.wrapper(checked, deadline=101., owner_byte_limit=scoring.MAX_OWNER_BYTES)
        self.assertEqual(calls, [101.])

    def test_unchecked_query_and_mutated_current_parent_refused(self):
        with patch.object(self.wrapper, "_forward_batch", side_effect=AssertionError("no NN")):
            with self.assertRaisesRegex(ValueError, "exact CheckedStrategicQuery"):
                self.wrapper(object(), deadline=time.monotonic() + 60)
            checked = self.checked()
            self.f.parents.records[self.f.index]["input"]["snapshot"]["input_revision"] += 1
            with self.assertRaises(ValueError):
                self.wrapper(checked, deadline=time.monotonic() + 60)

    def test_parent_mutation_during_forward_refused(self):
        checked = self.checked()
        def mutate(inputs, features, *, deadline):
            self.f.parents.records[self.f.index]["input"]["snapshot"]["input_revision"] += 1
            return numeric_stub(inputs, features, deadline=deadline)
        with patch.object(self.wrapper, "_forward_batch", side_effect=mutate):
            with self.assertRaises(ValueError):
                self.wrapper(checked, deadline=time.monotonic() + 60)

    def test_forward_changes_eval_or_requires_grad_flags_refused(self):
        checked = self.checked()
        first_parameter = next(self.model.parameters())
        mutations = (lambda: self.model.train(), lambda: self.model.public_encoder.train(),
                     lambda: first_parameter.requires_grad_(True))
        for mutation in mutations:
            def mutate(inputs, features, *, deadline):
                mutation()
                return numeric_stub(inputs, features, deadline=deadline)
            try:
                with patch.object(self.wrapper, "_forward_batch", side_effect=mutate), self.assertRaises(ValueError):
                    self.wrapper(checked, deadline=time.monotonic() + 60)
            finally:
                self.model.eval()
                first_parameter.requires_grad_(False)

    def test_forward_changes_parameter_or_grad_bytes_refused(self):
        checked = self.checked()
        parameter = next(self.model.parameters())
        backup = parameter.detach().clone()
        for kind in ("parameter", "gradient"):
            def mutate(inputs, features, *, deadline):
                if kind == "parameter":
                    parameter.add_(1.)
                else:
                    parameter.grad = torch.zeros_like(parameter)
                return numeric_stub(inputs, features, deadline=deadline)
            try:
                with patch.object(self.wrapper, "_forward_batch", side_effect=mutate), self.assertRaises(ValueError):
                    self.wrapper(checked, deadline=time.monotonic() + 60)
            finally:
                with torch.no_grad():
                    parameter.copy_(backup)
                parameter.grad = None

    def test_forward_changes_module_config_refused(self):
        checked = self.checked()
        module = self.model.experts["validator"]
        original = module.config
        def mutate(inputs, features, *, deadline):
            module.config = replace(original, iterations=3)
            return numeric_stub(inputs, features, deadline=deadline)
        try:
            with patch.object(self.wrapper, "_forward_batch", side_effect=mutate), self.assertRaises(ValueError):
                self.wrapper(checked, deadline=time.monotonic() + 60)
        finally:
            module.config = original

    def test_bad_score_shape_dtype_nonfinite_or_grad_refused(self):
        checked = self.checked()
        bad = (torch.tensor([math.nan, 0.]), torch.zeros(2, dtype=torch.float64), torch.zeros(3),
               torch.zeros(2, requires_grad=True))
        for value in bad:
            with self.subTest(shape=value.shape, dtype=value.dtype), \
                    patch.object(self.wrapper, "_forward_batch", return_value=value), self.assertRaises(ValueError):
                self.wrapper(checked, deadline=time.monotonic() + 60)

    def test_active_autocast_refused_before_nn(self):
        checked = self.checked()
        with torch.autocast("cpu", dtype=torch.bfloat16), \
                patch.object(self.wrapper, "_forward_batch", side_effect=AssertionError("no NN")):
            with self.assertRaisesRegex(ValueError, "autocast"):
                self.wrapper(checked, deadline=time.monotonic() + 60)

    def test_changed_tensor_defaults_refused_before_host_or_nn(self):
        checked = self.checked()
        for function, value in (("get_default_dtype", torch.float64), ("get_default_device", torch.device("cuda"))):
            with self.subTest(function=function), patch.object(torch, function, return_value=value), \
                    patch.object(scoring, "_tensor_inputs_zero_seed", side_effect=AssertionError("no host tensor")), \
                    patch.object(self.wrapper, "_forward_batch", side_effect=AssertionError("no NN")):
                with self.assertRaisesRegex(ValueError, "CPU FP32 tensor defaults"):
                    self.wrapper(checked, deadline=time.monotonic() + 60)

    def test_small_private_memory_matches_direct_projection_without_dense_bank(self):
        feature = self.checked().features()[0]
        # Actual small tiled projection oracle, independent of long layout and
        # not an admission or real action outcome. Query branch remains sealed.
        tokens = tuple(itertools.islice(scoring.token_descriptors(feature), semantic.TOKEN_COUNT, semantic.TOKEN_COUNT + 4))
        payload, mask = scoring._tile(self.model, (tokens,))
        with torch.no_grad():
            direct = self.model.public_encoder.memory_key(payload)
            tiled = torch.cat([self.model.public_encoder.memory_key(payload[:, at:at + 2]) for at in (0, 2)], dim=1)
        self.assertTrue(torch.equal(mask, torch.tensor([[True, False, False, False]])))
        self.assertTrue(torch.allclose(tiled, direct, atol=2e-6, rtol=2e-6))


if __name__ == "__main__":
    unittest.main()
