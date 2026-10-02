"""A CPU float64 linear policy/WDL adapter for synthetic gradient fixtures.

This adapter exercises training, export and optimizer bookkeeping. Its features
are supplied synthetic numbers; it does not encode chess or implement Maia.
"""

import math
import random

from rz_data.errors import DataError


ADAPTER_ID = "linear-policy-wdl-fixture-v1"
MAX_BATCH = 4096
TENSOR_KEYS = ("policy_weight", "policy_bias", "wdl_weight", "wdl_bias")


def _number(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise DataError("InvalidType", context, "expected a float64-compatible number, not bool")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise DataError("NonFinite", context, "number cannot be represented as finite float64") from exc
    if not math.isfinite(converted):
        raise DataError("NonFinite", context, "number must be finite float64")
    return converted


def _sum(values: list[float], context: str) -> float:
    try:
        return _number(math.fsum(values), context)
    except (OverflowError, ValueError) as exc:
        raise DataError("NonFinite", context, "float64 arithmetic overflow") from exc


def _fields(value: object, keys: tuple[str, ...], context: str) -> dict:
    if not isinstance(value, dict):
        raise DataError("InvalidType", context, "expected an object")
    if set(value) != set(keys):
        raise DataError("InvalidFields", context, "object fields must exactly match the adapter schema")
    return value


def _vector(value: object, width: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != width:
        raise DataError("InvalidShape", context, f"expected a list of {width} numbers")
    return [_number(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _target(value: object, width: int, context: str) -> list[float]:
    target = _vector(value, width, context)
    if any(item < 0.0 or item > 1.0 for item in target):
        raise DataError("OutOfRange", context, "soft targets must be probabilities in [0, 1]")
    if not math.isclose(_sum(target, context), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise DataError("InvalidNormalization", context, "soft targets must sum to one")
    return target


def _softmax(logits: list[float]) -> tuple[list[float], float, float]:
    maximum = max(logits)
    exponentials = [math.exp(item - maximum) for item in logits]
    denominator = _sum(exponentials, "softmax")
    return [item / denominator for item in exponentials], maximum, math.log(denominator)


class LinearFixture:
    """Two independent linear heads with SGD momentum and strict state imports."""

    def __init__(self, feature_width: int, policy_moves: list[str], seed: int):
        if type(feature_width) is not int or not 1 <= feature_width <= 64:
            raise DataError("InvalidDimension", "feature_width", "expected an integer in [1, 64]")
        if not isinstance(policy_moves, list) or not 1 <= len(policy_moves) <= 512:
            raise DataError("InvalidDimension", "policy_moves", "expected between 1 and 512 move identities")
        if any(not isinstance(move, str) or not 1 <= len(move) <= 2048 or not move.strip()
               for move in policy_moves):
            raise DataError("InvalidMoveIdentity", "policy_moves", "expected nonempty strings, max 2048 characters")
        if len(set(policy_moves)) != len(policy_moves):
            raise DataError("DuplicateMove", "policy_moves", "policy move identities must be unique and ordered")
        if (feature_width + 1) * (len(policy_moves) + 3) > 65536:
            raise DataError("TensorLimit", "adapter", "adapter tensor element count exceeds 65536")
        if type(seed) is not int:
            raise DataError("InvalidSeed", "seed", "seed must be an integer, not bool")
        self._width = feature_width
        self._moves = policy_moves.copy()
        generator = random.Random(seed)
        scale = 1.0 / math.sqrt(feature_width)
        self._state = {
            "policy_weight": [[generator.uniform(-scale, scale) for _ in policy_moves] for _ in range(feature_width)],
            "policy_bias": [0.0] * len(policy_moves),
            "wdl_weight": [[generator.uniform(-scale, scale) for _ in range(3)] for _ in range(feature_width)],
            "wdl_bias": [0.0] * 3,
        }

    def descriptor(self) -> dict:
        return {
            "adapter_id": ADAPTER_ID, "feature_width": self._width,
            "policy_moves": self._moves.copy(), "precision": "float64",
        }

    def _checked_state(self, state: object, context: str) -> dict:
        state = _fields(state, TENSOR_KEYS, context)
        checked = {}
        for head, columns in (("policy", len(self._moves)), ("wdl", 3)):
            weight, bias = f"{head}_weight", f"{head}_bias"
            matrix = state[weight]
            if not isinstance(matrix, list) or len(matrix) != self._width:
                raise DataError("InvalidShape", f"{context}.{weight}", f"expected {self._width} rows")
            checked[weight] = [_vector(row, columns, f"{context}.{weight}[{index}]")
                               for index, row in enumerate(matrix)]
            checked[bias] = _vector(state[bias], columns, f"{context}.{bias}")
        return checked

    def state_dict(self) -> dict:
        return self._checked_state(self._state, "state")

    def load_state_dict(self, state: dict) -> None:
        checked = self._checked_state(state, "state")
        self._state = checked

    def zero_optimizer_state(self) -> dict:
        return {"velocity": {
            "policy_weight": [[0.0] * len(self._moves) for _ in range(self._width)],
            "policy_bias": [0.0] * len(self._moves),
            "wdl_weight": [[0.0] * 3 for _ in range(self._width)],
            "wdl_bias": [0.0] * 3,
        }}

    def _logits(self, features: list[float], head: str, columns: int) -> list[float]:
        return [_sum(
            [self._state[f"{head}_bias"][column]]
            + [_number(feature * self._state[f"{head}_weight"][row][column], f"{head}.logits")
               for row, feature in enumerate(features)],
            f"{head}.logits",
        ) for column in range(columns)]

    def predict(self, features: list[float]) -> dict:
        checked = _vector(features, self._width, "features")
        policy, _, _ = _softmax(self._logits(checked, "policy", len(self._moves)))
        wdl, _, _ = _softmax(self._logits(checked, "wdl", 3))
        return {"policy": policy, "wdl": wdl}

    def loss_and_grad(self, batch: list[dict], policy_weight: float, wdl_weight: float) -> tuple[float, dict]:
        if not isinstance(batch, list) or not 1 <= len(batch) <= MAX_BATCH:
            raise DataError("InvalidBatch", "batch", f"expected a nonempty list, max {MAX_BATCH} samples")
        weights = {"policy": _number(policy_weight, "policy_weight"), "wdl": _number(wdl_weight, "wdl_weight")}
        if any(value < 0.0 for value in weights.values()):
            raise DataError("OutOfRange", "loss_weights", "loss weights must be nonnegative")
        grads = self.zero_optimizer_state()["velocity"]
        loss_terms = []
        for index, sample in enumerate(batch):
            context = f"batch[{index}]"
            sample = _fields(sample, ("features", "policy", "wdl"), context)
            features = _vector(sample["features"], self._width, context + ".features")
            targets = {"policy": _target(sample["policy"], len(self._moves), context + ".policy"),
                       "wdl": _target(sample["wdl"], 3, context + ".wdl")}
            for head, columns in (("policy", len(self._moves)), ("wdl", 3)):
                if weights[head] == 0.0:
                    continue
                logits = self._logits(features, head, columns)
                probabilities, maximum, log_denominator = _softmax(logits)
                target = targets[head]
                scale = weights[head] / len(batch)
                head_loss = _sum([_number(value * ((maximum - score) + log_denominator), context + ".loss")
                                  for value, score in zip(target, logits) if value != 0.0], context + ".loss")
                loss_terms.append(_number(head_loss * scale, context + ".weighted_loss"))
                target_total = _sum(target, context + ".target")
                for column, (probability, value) in enumerate(zip(probabilities, target)):
                    delta = _number((probability * target_total - value) * scale, context + ".gradient")
                    bias = f"{head}_bias"
                    grads[bias][column] = _number(grads[bias][column] + delta, context + ".gradient")
                    for row, feature in enumerate(features):
                        weight = f"{head}_weight"
                        grads[weight][row][column] = _number(
                            grads[weight][row][column] + feature * delta, context + ".gradient")
        return _sum(loss_terms, "loss"), self._checked_state(grads, "gradients")

    def apply_gradient(self, grads: dict, optimizer_state: dict, *,
                       learning_rate: float, momentum: float, freeze: bool = False) -> dict:
        gradients = self._checked_state(grads, "gradients")
        optimizer = _fields(optimizer_state, ("velocity",), "optimizer_state")
        velocity = self._checked_state(optimizer["velocity"], "optimizer_state.velocity")
        rate, coefficient = _number(learning_rate, "learning_rate"), _number(momentum, "momentum")
        if rate <= 0.0:
            raise DataError("OutOfRange", "learning_rate", "learning rate must be positive")
        if not 0.0 <= coefficient < 1.0:
            raise DataError("OutOfRange", "momentum", "momentum must be in [0, 1)")
        if type(freeze) is not bool:
            raise DataError("InvalidType", "freeze", "freeze must be bool")
        if freeze:
            return {"velocity": velocity}
        proposed_state = self.state_dict()
        proposed_velocity = self.zero_optimizer_state()["velocity"]

        def update(value: float, gradient: float, previous: float) -> tuple[float, float]:
            current_velocity = _number(coefficient * previous + gradient, "optimizer_state.velocity")
            current_value = _number(value - rate * current_velocity, "updated_state")
            return current_value, current_velocity

        for key in TENSOR_KEYS:
            if key.endswith("_weight"):
                for row in range(self._width):
                    for column in range(len(gradients[key][row])):
                        proposed_state[key][row][column], proposed_velocity[key][row][column] = update(
                            self._state[key][row][column], gradients[key][row][column], velocity[key][row][column])
            else:
                for column in range(len(gradients[key])):
                    proposed_state[key][column], proposed_velocity[key][column] = update(
                        self._state[key][column], gradients[key][column], velocity[key][column])
        # Commit after every tensor and every intermediate update is finite.
        checked_state = self._checked_state(proposed_state, "updated_state")
        checked_velocity = self._checked_state(proposed_velocity, "optimizer_state.velocity")
        self._state = checked_state
        return {"velocity": checked_velocity}
