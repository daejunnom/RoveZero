"""Separate first-Repair-decision projection of checked suffix comparisons.

Only exact factory CheckedRepairContext/CheckedWholeLinePair capabilities are
accepted. The whole-line root is the CURRENT Repair state, not its old Propose
root; each forced line is a suffix replayed from that state. The unchanged CPU
wire keeps raw endpoint scores. This module grants only a next-move preference
conditioned on two fixed suffixes, never strategic validity, WDL, minimax or an
ordinary policy target. Equal first moves with different suffixes are masked.

Durable projection-before ordering and checkpoint reload are independently
pinned CALLER observations, not proof created by declarations/hashes. This
module launches nothing, loads no checkpoint and changes no dataset/seal/wire.
C divergence projection/witness APIs intentionally are not present here.
"""
import copy
from dataclasses import dataclass
import math
import time

from . import comparative_training as numeric
from . import semantic_verifier as rules
from . import whole_line_ordinal as ordinal
from .repair_context import CheckedRepairContext

SCOPE = "next_repair_move_whole_line_conditioned_surrogate"
BEFORE_SCHEMA = "rz-pals-repair-first-decision-before/1"
ORDER_SCHEMA = "rz-pals-repair-first-decision-caller-order/1"
ADMISSION_SCHEMA = "rz-pals-repair-first-decision-admission/1"
PARAMETER_SCHEMA = "rz-pals-repair-first-decision-frozen-parameter-observation/1"
PREPARATION_SCHEMA = "rz-pals-repair-first-decision-frozen-preparation/1"
MAX_BYTES = 16 << 20
MAX_TENSOR_BYTES = 1 << 20
_FACTORY = object()
_RAW_NAMES = ("ordinal_plan", "ordinal_criterion", "projection_before", "projection_order_observation")
_PIN_NAMES = (*_RAW_NAMES, "causal_admission_sha256", "whole_admission_sha256", "current_view_sha256", "frozen_admission_sha256")
canonical, byte_pin, digest = ordinal.canonical, ordinal.byte_pin, ordinal.digest


@dataclass(frozen=True)
class RepairFirstDecisionTarget:
    pair_id: str
    input_sha256: str
    row_index: int
    first_moves: tuple
    sign: int
    mask: bool
    reason: str
    split: str
    scope: str = SCOPE


def _validate(causal, whole, raws, pins):
    if type(causal) is not CheckedRepairContext or type(whole) is not ordinal.CheckedWholeLinePair:
        raise ValueError("exact factory-checked Repair and whole-line capabilities required")
    causal.verify()
    whole.verify()
    if (whole._parents is not causal._parents or whole._index != causal._repair_index or whole._anchor is not causal):
        raise ValueError("whole-line root must be the exact causal CURRENT Repair row/anchor object")
    rules._fields(pins, _PIN_NAMES)
    if any(type(raw) is not bytes for raw in raws.values()) or sum(map(len, raws.values())) > MAX_BYTES:
        raise ValueError("bounded immutable Repair projection raw bytes required")
    for name in _RAW_NAMES:
        ordinal._pin(raws[name], pins[name])
    parent, index = causal._parents, causal._repair_index
    if (pins["causal_admission_sha256"] != causal.sha256 or pins["whole_admission_sha256"] != whole.sha256
            or pins["current_view_sha256"] != parent.current_view.sha256
            or pins["frozen_admission_sha256"] != parent._frozen_admission_identity):
        raise ValueError("independent projection/current/frozen capability pins mismatch")
    row = parent.records[index]
    anchor = ordinal.input_anchor(parent, index, causal)
    whole_admission = whole.admission
    if anchor["kind"] != "native_initial_repair" or canonical(whole_admission["anchor"]) != canonical(anchor):
        raise ValueError("ordinary root/full proposal cannot become native_initial_repair suffix supervision")
    for name in ("ordinal_plan", "ordinal_criterion"):
        reference = whole_admission["plan_artifact" if name == "ordinal_plan" else "criterion_artifact"]
        if canonical(byte_pin(raws[name])) != canonical(reference):
            raise ValueError("whole-line original plan/criterion actual byte anchors mismatch")
    criterion = ordinal._criterion(raws["ordinal_criterion"])
    if criterion["direction"] != "maximize_root_surrogate":
        raise ValueError("first-Repair decision requires independently fixed maximize_root_surrogate criterion")
    plan = ordinal._json(raws["ordinal_plan"])
    if canonical(plan["anchor"]) != canonical(anchor):
        raise ValueError("whole-line plan root is not the current Repair state")
    context = causal.context()
    prefix_receipt = causal._checks[0].rules_receipt()
    # Rules fields are compared, not inferred from line strings or advisory IDs.
    executions = {task: dict(buffers) for task, buffers in whole._executions}
    tasks, first = [], []
    for planned in plan["lines"]:
        task, line = planned["task_id"], planned["line"]
        tasks.append(task)
        if (not line or len(line) != criterion["line_plies"] or line[0] not in anchor["legal_moves"]
                or planned["line_sha256"] != ordinal.ordered_line_sha256(anchor["rules_state_sha256"], anchor["rules_history_sha256"], line)):
            raise ValueError("actual Repair-root suffix/legal first-decision identity mismatch")
        first.append(line[0])
        request = ordinal._json(executions[task]["request"], 524288)
        if (request["position_command"] != anchor["position_command"]
                or request["root_rules_state_sha256"] != anchor["rules_state_sha256"]
                or request["root_rules_history_sha256"] != anchor["rules_history_sha256"]
                or canonical(request["line"]) != canonical(line)):
            raise ValueError("actual continuation request must replay suffix at this Repair root")
        if executions[task]["receipt"] is not None:
            receipt = ordinal._json(executions[task]["receipt"], criterion["max_output_bytes"])
            if not rules._same_state(receipt["root"], prefix_receipt["target"]):
                raise ValueError("actual whole-line root differs from causal prefix Rules target/full known history")
    before = rules._fields(ordinal._json(raws["projection_before"]), ("schema", "scope", "pair_id", "causal_admission_sha256",
        "repair_anchor", "causal_context_sha256", "ordinal_plan_artifact", "ordinal_criterion_artifact", "direction",
        "plan_root", "line_meaning", "ordered_task_ids", "first_moves"))
    expected_before = {"schema": BEFORE_SCHEMA, "scope": SCOPE, "pair_id": plan["pair_id"],
        "causal_admission_sha256": causal.sha256, "repair_anchor": anchor,
        "causal_context_sha256": causal.admission()["context_sha256"],
        "ordinal_plan_artifact": byte_pin(raws["ordinal_plan"]), "ordinal_criterion_artifact": byte_pin(raws["ordinal_criterion"]),
        "direction": "maximize_root_surrogate", "plan_root": "current_repair_state",
        "line_meaning": "suffix_from_current_repair_state", "ordered_task_ids": tasks, "first_moves": first}
    if canonical(before) != canonical(expected_before):
        raise ValueError("result-free first-decision before plan/indices/context/actual bytes mismatch")
    # The strict exact field set above cannot include a result whole SHA, score
    # or sign. Only the post-result admission below binds the whole capability.
    order = rules._fields(ordinal._json(raws["projection_order_observation"]), ("schema", "projection_before_artifact",
        "ordinal_plan_artifact", "ordinal_criterion_artifact", "causal_admission_sha256", "assurance_scope", "launches"))
    if (order["schema"] != ORDER_SCHEMA
            or order["assurance_scope"] != "independently_pinned_caller_durable_projection_order_observation"
            or canonical(order["projection_before_artifact"]) != canonical(byte_pin(raws["projection_before"]))
            or canonical(order["ordinal_plan_artifact"]) != canonical(byte_pin(raws["ordinal_plan"]))
            or canonical(order["ordinal_criterion_artifact"]) != canonical(byte_pin(raws["ordinal_criterion"]))
            or order["causal_admission_sha256"] != causal.sha256 or type(order["launches"]) is not list
            or len(order["launches"]) != 2):
        raise ValueError("actual caller projection-before publication/launch observation mismatch")
    for task, item in zip(tasks, order["launches"]):
        rules._fields(item, ("task_id", "launch_artifact", "projection_durable_before_spawn", "ordinal_plan_durable_before_spawn"))
        if (item["task_id"] != task or canonical(item["launch_artifact"]) != canonical(byte_pin(executions[task]["launch"]))
                or item["projection_durable_before_spawn"] is not True or item["ordinal_plan_durable_before_spawn"] is not True):
            raise ValueError("each actual whole-line launch needs independently observed durable projection ordering")
    outcome = whole.outcome
    if first[0] == first[1]:
        active, sign, reason = False, 0, "unsupported_same_first_move"
    else:
        active, sign, reason = outcome.mask, outcome.sign, outcome.reason
    if active != (sign in (-1, 1)):
        raise ValueError("checked first-decision ordinal mask/sign mismatch")
    target = RepairFirstDecisionTarget(plan["pair_id"], row["input"]["sha256"], index, tuple(first), sign, active, reason,
                                      parent.split[row["input"]["snapshot"]["game_id"]])
    admission = {"schema": ADMISSION_SCHEMA, "scope": SCOPE, "causal_admission_sha256": causal.sha256,
        "whole_admission_sha256": whole.sha256, "projection_before_artifact": byte_pin(raws["projection_before"]),
        "projection_order_artifact": byte_pin(raws["projection_order_observation"]),
        "repair_input_sha256": row["input"]["sha256"], "context_sha256": causal.admission()["context_sha256"],
        "virtual_prefix": context["virtual_prefix"], "counterexample": context["counterexample"],
        "known_first_decision_ordinal": target.mask, "reason": target.reason,
        "caller_assurance": "independently_pinned_caller_projection_order;whole_line_actual_execution_admission",
        "ordinary_policy_admitted": False, "strategic_repair_validity_admitted": False,
        "counterexample_validity_admitted": False, "wdl_admitted": False, "divergence_ranking_admitted": False,
        "minimax_admitted": False, "tactical_proof_admitted": False, "actual_training_executed": False}
    return target, admission


class CheckedRepairWholeLineProjection:
    """Factory-only separate suffix-conditioned first-decision authority."""
    __slots__ = ("_causal", "_whole", "_raws", "_pins", "_identity")

    def __init__(self, token=None, *, causal=None, whole=None, raws=None, pins=None):
        if token is not _FACTORY:
            raise ValueError("unchecked whole-line first-Repair projection constructor refused")
        for name, value in (("_causal", causal), ("_whole", whole), ("_raws", tuple(sorted(raws.items()))),
                            ("_pins", canonical(pins)), ("_identity", digest(ADMISSION_SCHEMA, pins))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("checked first-Repair projection is immutable")

    def _views(self):
        pins = ordinal._json(self._pins)
        if self._identity != digest(ADMISSION_SCHEMA, pins):
            raise ValueError("first-Repair projection identity changed")
        return _validate(self._causal, self._whole, dict(self._raws), pins)

    def verify(self):
        self._views()
        return self

    @property
    def sha256(self):
        self.verify()
        return self._identity

    @property
    def target(self):
        return self._views()[0]

    @property
    def admission(self):
        return copy.deepcopy(self._views()[1])

    def collate(self, *, split="train", max_tensor_bytes=MAX_TENSOR_BYTES):
        target = self.target
        if split not in ("train", "validation", "holdout") or target.split != split:
            raise ValueError("first-Repair projection cannot cross immutable game split")
        if target.first_moves[0] == target.first_moves[1]:
            raise ValueError("same first move needs a separate suffix scorer; no duplicate-logit collation")
        parent = self._causal._parents
        encoded = parent.encodings[target.input_sha256]
        legal = parent.records[target.row_index]["input"]["snapshot"]["legal_moves"]
        # Conservative ordinary tensors/targets plus temporary row copies;
        # model/public memory/private forward workspace are separate budgets.
        needed = 4096 + 2 * (max(1, len(encoded.public_records)) * (16 * 4 + 1)
            + max(1, len(legal)) * (3 * 8 + 1 + 4) + max(1, len(encoded.divergences)) * (8 * 4 + 1 + 4 + 1))
        rules._int(max_tensor_bytes, 1, MAX_TENSOR_BYTES)
        if needed > max_tensor_bytes:
            raise ValueError("first-Repair collation byte allowance before tensor allocation")
        base = parent.collate([target.row_index], "proposer", split=split, device="cpu")
        return numeric.ComparativeBatch(base, (0,), (tuple(legal.index(move) for move in target.first_moves),),
                                        (target.sign,), (target.mask,), (target.pair_id,))


def admit_repair_first_move_projection(causal, whole, *, ordinal_plan_bytes, ordinal_criterion_bytes,
        projection_before_bytes, projection_order_observation_bytes, expected_pins):
    raws = dict(ordinal_plan=ordinal_plan_bytes, ordinal_criterion=ordinal_criterion_bytes,
                projection_before=projection_before_bytes, projection_order_observation=projection_order_observation_bytes)
    pins = ordinal._json(canonical(expected_pins))
    _validate(causal, whole, raws, pins)
    return CheckedRepairWholeLineProjection(_FACTORY, causal=causal, whole=whole, raws=raws, pins=pins)


def _deadline(deadline):
    if type(deadline) not in (float, int) or not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise ValueError("first-Repair original absolute preparation deadline expired/invalid")


def frozen_repair_projection_preparation(model, checked, *, parameter_observation_bytes, expected_observation_pin,
        deadline, split="train", max_tensor_bytes=MAX_TENSOR_BYTES):
    """Offline CPU FP32 frozen no-step seam; caller already reloaded checkpoint.

    The caller's original monotonic absolute deadline covers this admission,
    collation, forward/loss and all final checks. Reload and final publication
    remain caller-supervised. Checkpoint/source identity and actual tensor digest
    are checked; independently pinned reload observation is a caller trust seam,
    not a checkpoint deserializer or a replacement for the Rust product engine.
    """
    _deadline(deadline)
    started = time.monotonic()
    if type(checked) is not CheckedRepairWholeLineProjection:
        raise ValueError("exact checked first-Repair projection required")
    checked.verify()
    target = checked.target
    if not target.mask:
        raise ValueError("all-masked first-Repair projection cannot be positive frozen preparation")
    ordinal._pin(parameter_observation_bytes, expected_observation_pin)
    observation = rules._fields(ordinal._json(parameter_observation_bytes), ("schema", "checkpoint_sha256", "parameter_sha256",
        "frozen_epoch", "model_configuration", "assurance_scope"))
    for name in ("checkpoint_sha256", "parameter_sha256"):
        rules._sha(observation[name])
    snapshot = checked._causal._parents.records[target.row_index]["input"]["snapshot"]
    if (observation["schema"] != PARAMETER_SCHEMA
            or observation["assurance_scope"] != "independently_pinned_caller_checkpoint_reload_observation"
            or observation["checkpoint_sha256"] != snapshot["source"]["model_weights_sha256"]
            or type(observation["frozen_epoch"]) is not int or observation["frozen_epoch"] != snapshot["frozen_epoch"]):
        raise ValueError("registered current Repair checkpoint/epoch/reload observation mismatch")
    import torch
    from .config import ModelConfig
    from .preparation_check import _parameter_digest

    def frozen_digest():
        _deadline(deadline)
        if (any(module.training for module in model.modules())
                or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())
                or torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda")):
            raise ValueError("frozen eval/no-grad/no-autocast model required before and after Repair forward")
        configuration = model.config.to_dict()
        if (canonical(configuration) != canonical(ModelConfig().to_dict())
                or canonical(configuration) != canonical(observation["model_configuration"])):
            raise ValueError("actual frozen Repair model configuration differs from independent reload pin")
        value = _parameter_digest(model)  # Finite CPU FP32 parameters and buffers.
        if value != observation["parameter_sha256"]:
            raise ValueError("actual frozen Repair parameter bytes differ from independent reload pin")
        _deadline(deadline)
        return value

    before = frozen_digest()
    batch = checked.collate(split=split, max_tensor_bytes=max_tensor_bytes)
    _deadline(deadline)
    with torch.inference_mode():
        memory = model.public_encoder(*batch.base.inputs.public_args())
        outputs = model.role_graph("proposer")(*batch.base.inputs.role_args(memory))
        loss = numeric.pairwise_softplus_loss(outputs[0], batch)
        value = float(loss)
    _deadline(deadline)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("first-Repair frozen loss must be finite and nonzero")
    after = frozen_digest()
    checked.verify()
    projection_sha, causal_sha, whole_sha = checked.sha256, checked._causal.sha256, checked._whole.sha256
    _deadline(deadline)
    report = {"schema": PREPARATION_SCHEMA, "scope": SCOPE, "projection_admission_sha256": projection_sha,
        "causal_admission_sha256": causal_sha, "whole_admission_sha256": whole_sha,
        "parameter_observation_artifact": byte_pin(parameter_observation_bytes), "checkpoint_sha256": observation["checkpoint_sha256"],
        "parameter_sha256_before": before, "parameter_sha256_after": after, "known_pairs": 1,
        "consumed_pair_ids": [target.pair_id], "pairwise_loss": value,
        "observed_pre_return_elapsed_ms": int((time.monotonic() - started) * 1000),
        "elapsed_scope": "offline_admission_collation_forward_loss_final_checks;caller_reload_publication_supervised_separately",
        "ordinary_policy_admitted": False, "strategic_repair_validity_admitted": False,
        "counterexample_validity_admitted": False, "wdl_admitted": False, "divergence_ranking_admitted": False,
        "minimax_admitted": False, "tactical_proof_admitted": False, "actual_training_executed": False,
        "backward_executed": False, "optimizer_created": False, "gpu_executed": False}
    _deadline(deadline)
    return report
