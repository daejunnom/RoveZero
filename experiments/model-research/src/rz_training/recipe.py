"""Strict, immutable scope and resource declarations for internal training checks."""

import math
import re

from rz_data.errors import DataError
from rz_data.serialization import canonical_bytes


ADAPTER_ID = "linear-policy-wdl-fixture-v1"


def fields(value, names, context):
    if type(value) is not dict or set(value) != set(names):
        raise DataError("InvalidFields", context, "object must contain exactly the declared fields")
    return value


def integer(value, low, high, context):
    if type(value) is not int or not low <= value <= high:
        raise DataError("InvalidInteger", context, "integer is outside the supported finite range")
    return value


def number(value, low, high, context):
    if type(value) not in (int, float):
        raise DataError("InvalidNumber", context, "expected a finite number")
    try:
        valid = math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    if not valid:
        raise DataError("InvalidNumber", context, "number is outside the supported finite range")
    return value


def sha256(value, context):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DataError("InvalidDigest", context, "expected lowercase SHA256")
    return value


def validate_recipe(value):
    canonical_bytes(value, max_bytes=65536)
    fields(value, ("schema_version", "task_id", "execution_scope", "execution_ready",
                   "contract_reference", "dataset", "adapter", "backend", "precision",
                   "seed", "batch_size", "gradient_accumulation", "workers", "freeze",
                   "loss", "optimizer", "schedule", "checkpoint", "budget"), "recipe")
    integer(value["schema_version"], 1, 1, "recipe.schema_version")
    if (value["task_id"] != "TASK-F02" or value["execution_scope"] != "cpu_fixture"
            or value["execution_ready"] is not False):
        raise DataError("UnsupportedScope", "recipe", "only explicitly non-production CPU fixtures are supported")
    reference = fields(value["contract_reference"], ("revision", "wire_binding"), "recipe.contract_reference")
    if reference != {"revision": "0.1", "wire_binding": "not_run"}:
        raise DataError("UnsupportedContract", "recipe.contract_reference", "Rust 0.1 is a reference, not a Python wire binding")
    dataset = fields(value["dataset"], ("manifest_digest", "records_file_digest", "split_plan_digest",
                                        "features_file_digest"), "recipe.dataset")
    for key, item in dataset.items():
        sha256(item, "recipe.dataset." + key)
    adapter = fields(value["adapter"], ("id", "feature_width", "policy_moves", "initialization"), "recipe.adapter")
    if adapter["id"] != ADAPTER_ID or adapter["initialization"] != "seeded_fixture":
        raise DataError("UnsupportedAdapter", "recipe.adapter", "no production encoder/model fallback is available")
    integer(adapter["feature_width"], 1, 64, "recipe.adapter.feature_width")
    moves = adapter["policy_moves"]
    if type(moves) is not list or not 1 <= len(moves) <= 512:
        raise DataError("InvalidMoves", "recipe.adapter.policy_moves", "expected a bounded ordered UCI move list")
    for move in moves:
        if (type(move) is not str or re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move) is None
                or move[:2] == move[2:4]):
            raise DataError("InvalidMove", "recipe.adapter.policy_moves", "expected syntactically valid UCI moves")
    if len(set(moves)) != len(moves):
        raise DataError("DuplicateMove", "recipe.adapter.policy_moves", "ordered moves must be unique")
    if value["backend"] != "cpu" or value["precision"] != "float64":
        raise DataError("UnsupportedBackend", "recipe", "CPU float64 must be selected explicitly")
    integer(value["seed"], 0, 2**32 - 1, "recipe.seed")
    integer(value["batch_size"], 1, 256, "recipe.batch_size")
    integer(value["gradient_accumulation"], 1, 1, "recipe.gradient_accumulation")
    integer(value["workers"], 1, 1, "recipe.workers")
    if type(value["freeze"]) is not bool:
        raise DataError("InvalidBoolean", "recipe.freeze", "expected boolean")
    loss = fields(value["loss"], ("policy_weight", "teacher_wdl_weight", "policy_target", "value_target"), "recipe.loss")
    for key in ("policy_weight", "teacher_wdl_weight"):
        number(loss[key], 0, 100, "recipe.loss." + key)
    if loss["policy_weight"] + loss["teacher_wdl_weight"] <= 0:
        raise DataError("InvalidLoss", "recipe.loss", "at least one loss weight must be positive")
    if loss["policy_target"] != "declared_probabilities" or loss["value_target"] != "teacher_side_to_move_wdl":
        raise DataError("UnsupportedTarget", "recipe.loss", "visits, cp and game-result mixing require a separate explicit adapter")
    optimizer = fields(value["optimizer"], ("kind", "learning_rate", "momentum"), "recipe.optimizer")
    if optimizer["kind"] != "sgd":
        raise DataError("UnsupportedOptimizer", "recipe.optimizer", "only SGD with momentum is supported")
    number(optimizer["learning_rate"], 1e-12, 1, "recipe.optimizer.learning_rate")
    number(optimizer["momentum"], 0, 0.999, "recipe.optimizer.momentum")
    if value["schedule"] != {"kind": "constant"}:
        raise DataError("UnsupportedSchedule", "recipe.schedule", "only constant learning rate is supported")
    checkpoint = fields(value["checkpoint"], ("every_steps", "selection", "tie_break"), "recipe.checkpoint")
    integer(checkpoint["every_steps"], 1, 10000, "recipe.checkpoint.every_steps")
    if checkpoint["selection"] != "validation_loss" or checkpoint["tie_break"] != "earliest_step":
        raise DataError("UnsupportedSelection", "recipe.checkpoint", "selection uses validation only, with earliest-step ties")
    budget = fields(value["budget"], ("max_steps", "max_samples", "max_time_ms", "max_records",
                                      "max_input_bytes", "max_output_bytes"), "recipe.budget")
    for key, cap in (("max_steps", 10000), ("max_samples", 1000000), ("max_time_ms", 60000),
                     ("max_records", 10000), ("max_input_bytes", 16777216), ("max_output_bytes", 67108864)):
        integer(budget[key], 1, cap, "recipe.budget." + key)
    return value
