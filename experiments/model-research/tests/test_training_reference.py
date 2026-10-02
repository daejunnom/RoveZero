"""Analytic-gradient, finite-boundary and atomic-optimizer fixture checks."""

from copy import deepcopy
import math
import random
import unittest

from rz_data.errors import DataError
from rz_training.reference import LinearFixture


MOVES = ["a7a8q", "a7a8r", "a7a8b"]
BATCH = [
    {"features": [0.7, -0.4], "policy": [0.2, 0.3, 0.5], "wdl": [0.6, 0.1, 0.3]},
    {"features": [-0.2, 0.9], "policy": [0.55, 0.35, 0.1], "wdl": [0.15, 0.25, 0.6]},
]


def tensor_entries(state):
    for key, tensor in state.items():
        if key.endswith("_weight"):
            for row, values in enumerate(tensor):
                for column, value in enumerate(values):
                    yield key, row, column, value
        else:
            for column, value in enumerate(tensor):
                yield key, None, column, value


def replace_entry(state, key, row, column, value):
    if row is None:
        state[key][column] = value
    else:
        state[key][row][column] = value


class ReferenceTests(unittest.TestCase):
    def test_seed_reproducibility_and_normalized_probabilities(self):
        random.seed(90210)
        global_state = random.getstate()
        first = LinearFixture(2, MOVES, seed=31)
        second = LinearFixture(2, MOVES, seed=31)
        third = LinearFixture(2, MOVES, seed=32)
        self.assertEqual(first.state_dict(), second.state_dict())
        self.assertNotEqual(first.state_dict(), third.state_dict())
        self.assertEqual(random.getstate(), global_state)
        self.assertEqual(first.descriptor(), {
            "adapter_id": "linear-policy-wdl-fixture-v1", "feature_width": 2,
            "policy_moves": MOVES, "precision": "float64",
        })
        prediction = first.predict([0.2, -0.7])
        for output in prediction.values():
            self.assertEqual(len(output), 3)
            self.assertTrue(all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in output))
            self.assertAlmostEqual(math.fsum(output), 1.0, places=15)

    def test_finite_difference_gradient_with_soft_targets_and_batch_average(self):
        model = LinearFixture(2, MOVES, seed=7)
        baseline = model.state_dict()
        batch_original = deepcopy(BATCH)
        loss, analytic = model.loss_and_grad(BATCH, policy_weight=0.7, wdl_weight=1.3)
        self.assertGreater(loss, 0.0)
        epsilon = 1e-6
        for key, row, column, value in tensor_entries(baseline):
            with self.subTest(key=key, row=row, column=column):
                positive, negative = deepcopy(baseline), deepcopy(baseline)
                replace_entry(positive, key, row, column, value + epsilon)
                replace_entry(negative, key, row, column, value - epsilon)
                model.load_state_dict(positive)
                plus, _ = model.loss_and_grad(BATCH, 0.7, 1.3)
                model.load_state_dict(negative)
                minus, _ = model.loss_and_grad(BATCH, 0.7, 1.3)
                expected = analytic[key][column] if row is None else analytic[key][row][column]
                self.assertAlmostEqual((plus - minus) / (2.0 * epsilon), expected, delta=2e-9)
        model.load_state_dict(baseline)
        single_losses = [model.loss_and_grad([sample], 0.7, 1.3)[0] for sample in BATCH]
        self.assertAlmostEqual(loss, math.fsum(single_losses) / len(BATCH), places=14)
        self.assertEqual(BATCH, batch_original)

    def test_sgd_momentum_and_freeze_preserve_input_state(self):
        model = LinearFixture(2, MOVES, seed=7)
        initial = model.state_dict()
        optimizer = model.zero_optimizer_state()
        original_optimizer = deepcopy(optimizer)
        grads = model.zero_optimizer_state()["velocity"]
        for key, row, column, _ in list(tensor_entries(grads)):
            replace_entry(grads, key, row, column, 0.5)
        original_grads = deepcopy(grads)
        first_optimizer = model.apply_gradient(grads, optimizer, learning_rate=0.1, momentum=0.9)
        first_weights = model.state_dict()
        for key, row, column, value in tensor_entries(initial):
            actual = first_weights[key][column] if row is None else first_weights[key][row][column]
            self.assertAlmostEqual(actual, value - 0.05, places=15)
        second_optimizer = model.apply_gradient(grads, first_optimizer, learning_rate=0.1, momentum=0.9)
        second_weights = model.state_dict()
        for key, row, column, value in tensor_entries(initial):
            actual = second_weights[key][column] if row is None else second_weights[key][row][column]
            velocity = second_optimizer["velocity"][key][column] if row is None else second_optimizer["velocity"][key][row][column]
            self.assertAlmostEqual(velocity, 0.95, places=15)
            self.assertAlmostEqual(actual, value - 0.145, places=15)
        frozen_optimizer = model.apply_gradient(grads, second_optimizer, learning_rate=0.1, momentum=0.9, freeze=True)
        self.assertEqual(frozen_optimizer, second_optimizer)
        self.assertIsNot(frozen_optimizer, second_optimizer)
        self.assertEqual(model.state_dict(), second_weights)
        self.assertEqual(optimizer, original_optimizer)
        self.assertEqual(grads, original_grads)

    def test_gradient_step_changes_predictions_and_lowers_fixture_loss(self):
        model = LinearFixture(2, MOVES, seed=7)
        before = model.predict(BATCH[0]["features"])
        loss, grads = model.loss_and_grad(BATCH, 1.0, 1.0)
        model.apply_gradient(grads, model.zero_optimizer_state(), learning_rate=0.05, momentum=0.0)
        updated_loss, _ = model.loss_and_grad(BATCH, 1.0, 1.0)
        self.assertLess(updated_loss, loss)
        self.assertNotEqual(model.predict(BATCH[0]["features"]), before)

    def test_export_import_predictions_identical_and_state_not_aliased(self):
        original = LinearFixture(2, MOVES, seed=7)
        imported = LinearFixture(2, MOVES, seed=100)
        state = original.state_dict()
        imported.load_state_dict(state)
        self.assertEqual(original.predict([0.8, -0.6]), imported.predict([0.8, -0.6]))
        state["policy_bias"][0] = 100.0
        self.assertEqual(original.predict([0.8, -0.6]), imported.predict([0.8, -0.6]))
        descriptor = imported.descriptor()
        descriptor["policy_moves"].reverse()
        self.assertEqual(imported.descriptor()["policy_moves"], MOVES)

    def test_stable_softmax_and_cross_entropy_do_not_log_underflowed_probabilities(self):
        model = LinearFixture(1, ["m0", "m1"], seed=7)
        state = model.state_dict()
        state["policy_weight"] = [[0.0, 0.0]]
        state["policy_bias"] = [-1000.0, 1000.0]
        model.load_state_dict(state)
        self.assertEqual(model.predict([1.0])["policy"], [0.0, 1.0])
        sample = {"features": [1.0], "policy": [1.0, 0.0], "wdl": [1.0, 0.0, 0.0]}
        loss, gradient = model.loss_and_grad([sample], 1.0, 0.0)
        self.assertEqual(loss, 2000.0)
        self.assertEqual(gradient["policy_bias"], [-1.0, 1.0])
        self.assertEqual(gradient["wdl_bias"], [0.0, 0.0, 0.0])

    def test_invalid_initialization_dimensions_move_ids_and_seed(self):
        for width, moves, seed in ((True, MOVES, 1), (0, MOVES, 1), (65, MOVES, 1),
                                   (2, [], 1), (2, ["m"] * 513, 1), (2, ["m", "m"], 1),
                                   (2, [True], 1), (2, [""], 1), (2, ["x" * 2049], 1),
                                   (2, MOVES, True), (2, MOVES, "1")):
            with self.subTest(width=width, seed=seed), self.assertRaises(DataError):
                LinearFixture(width, moves, seed)
        self.assertEqual(LinearFixture(64, [str(index) for index in range(512)], 0).descriptor()["feature_width"], 64)

    def test_load_state_shape_key_and_nonfinite_rejections_are_atomic(self):
        model = LinearFixture(2, MOVES, seed=7)
        original = model.state_dict()
        invalid_states = []
        extra = deepcopy(original)
        extra["unknown"] = []
        invalid_states.append(extra)
        missing = deepcopy(original)
        del missing["wdl_bias"]
        invalid_states.append(missing)
        wrong_row = deepcopy(original)
        wrong_row["policy_weight"][1].pop()
        invalid_states.append(wrong_row)
        for bad in (True, float("nan"), float("inf"), float("-inf"), "0", 10**1000):
            state = deepcopy(original)
            state["wdl_bias"][2] = bad
            invalid_states.append(state)
        for state in invalid_states:
            with self.subTest(state=repr(state)[:100]), self.assertRaises(DataError):
                model.load_state_dict(state)
            self.assertEqual(model.state_dict(), original)

    def test_external_batch_target_feature_and_weight_errors(self):
        model = LinearFixture(2, MOVES, seed=7)
        for invalid_features in ([True, 0.0], [float("nan"), 0.0], [0.0], (0.0, 0.0)):
            with self.subTest(features=invalid_features), self.assertRaises(DataError):
                model.predict(invalid_features)
        invalid_batches = [[], BATCH * 2049, [None]]
        for field, value in (("features", [1.0]), ("policy", None), ("wdl", [0.1, 0.1, 0.1]),
                             ("policy", [True, 0.0, 0.0]), ("policy", [-0.1, 0.5, 0.6]),
                             ("wdl", [float("inf"), 0.0, 0.0])):
            sample = deepcopy(BATCH[0])
            sample[field] = value
            invalid_batches.append([sample])
        for batch in invalid_batches:
            with self.subTest(batch=repr(batch)[:80]), self.assertRaises(DataError):
                model.loss_and_grad(batch, 1.0, 1.0)
        for bad in (True, -1.0, float("inf"), float("nan")):
            with self.subTest(weight=bad), self.assertRaises(DataError):
                model.loss_and_grad(BATCH, bad, 1.0)

    def test_optimizer_rejections_including_freeze_and_late_overflow_are_atomic(self):
        model = LinearFixture(2, MOVES, seed=7)
        state = model.state_dict()
        grads = model.zero_optimizer_state()["velocity"]
        optimizer = model.zero_optimizer_state()
        bad_optimizer = deepcopy(optimizer)
        bad_optimizer["velocity"]["wdl_bias"][2] = float("inf")
        with self.assertRaises(DataError):
            model.apply_gradient(grads, bad_optimizer, learning_rate=0.1, momentum=0.9, freeze=True)
        bad_optimizer = deepcopy(optimizer)
        bad_optimizer["velocity"]["wdl_weight"].pop()
        with self.assertRaises(DataError):
            model.apply_gradient(grads, bad_optimizer, learning_rate=0.1, momentum=0.9, freeze=True)
        for options in ({"learning_rate": True, "momentum": 0.0},
                        {"learning_rate": 0.0, "momentum": 0.0},
                        {"learning_rate": 0.1, "momentum": 1.0},
                        {"learning_rate": 0.1, "momentum": -0.1},
                        {"learning_rate": 0.1, "momentum": True},
                        {"learning_rate": 0.1, "momentum": 0.0, "freeze": 1}):
            with self.subTest(options=options), self.assertRaises(DataError):
                model.apply_gradient(grads, optimizer, **options)
        grads["wdl_bias"][2] = 1e308
        optimizer_original = deepcopy(optimizer)
        with self.assertRaises(DataError) as caught:
            model.apply_gradient(grads, optimizer, learning_rate=1e308, momentum=0.0)
        self.assertEqual(caught.exception.code, "NonFinite")
        self.assertEqual(model.state_dict(), state)
        self.assertEqual(optimizer, optimizer_original)


if __name__ == "__main__":
    unittest.main()
