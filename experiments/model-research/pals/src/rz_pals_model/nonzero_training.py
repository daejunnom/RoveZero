"""Finite CPU optimizer diagnostics in an explicit nonzero checkpoint domain.

The preparation checkpoint remains zero-step. This module proves a bounded
optimizer/resume lifecycle; its synthetic targets and resulting weights are
diagnostic artifacts, ineligible for arena/model promotion. It neither converts
CP into WDL nor supplies labels for an unknown result. Actual collected target
coverage is inspected separately through target_coverage.
"""
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile

import torch

from .artifacts import atomic_json, output_directory
from .config import ModelConfig, TASKS
from .training import (OptimizerPreparation, ResumableSampler, TrainingBatch,
                       _CollectionArtifactReader, _clone_state, _fields, _sha,
                       _identity as _identifier, _uint, _validate_rng, capture_rng, masked_losses,
                       prepare_nonzero_adamw, restore_rng)

CHECKPOINT_SCHEMA = "rz-pals-python-nonzero-checkpoint/1"
SPEC_SCHEMA = "rz-pals-python-nonzero-smoke-spec/1"
REPORT_SCHEMA = "rz-pals-python-nonzero-smoke-report/1"
ELIGIBILITY = "diagnostic_only_arena_ineligible"
MAX_UPDATES = 24
MAX_WALL_MS = 15 * 60 * 1000
MAX_OUTPUT_BYTES = 2 * 1024**3
_ACTUAL_TARGET_CAPABILITY = object()
USAGE_FIELDS = ("updates_dispatched", "updates_completed", "forward_calls",
                "backward_calls", "samples", "wall_time_ms", "output_bytes")
LOCKED_OPTIMIZER = {"name": "adamw", "learning_rate": 1e-4,
                    "betas": [0.9, 0.999], "epsilon": 1e-8,
                    "weight_decay": 0.01, "clip_norm": 1.0,
                    "schedule": "constant", "batch_size": 1,
                    "gradient_accumulation_steps": 1}


def _identity(domain, value):
    return hashlib.sha256(json.dumps([domain, value], sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def smoke_spec(config, *, implementation_sha256, dataset_sha256, split_sha256,
               source_kind="diagnostic_fixture", training_run_id="nonzero-contract-fixture"):
    config.validate()
    value = {"schema": SPEC_SCHEMA, "config": config.to_dict(),
             "optimizer": copy.deepcopy(LOCKED_OPTIMIZER), "device": "cpu", "precision": "fp32",
             "cpu_threads": 2, "max_updates": MAX_UPDATES, "max_wall_time_ms": MAX_WALL_MS,
             "cleanup_time_ms": 30000, "max_output_bytes": MAX_OUTPUT_BYTES,
             "implementation_sha256": implementation_sha256, "dataset_sha256": dataset_sha256,
             "split_sha256": split_sha256, "source_kind": source_kind, "eligibility": ELIGIBILITY}
    value["training_run_id"] = training_run_id
    validate_spec(value)
    return value


def validate_spec(spec):
    _fields(spec, ("schema", "config", "optimizer", "device", "precision", "cpu_threads",
                   "max_updates", "max_wall_time_ms", "cleanup_time_ms", "max_output_bytes",
                   "implementation_sha256", "dataset_sha256", "split_sha256", "source_kind",
                   "eligibility", "training_run_id"), "nonzero smoke spec")
    if (spec["schema"] != SPEC_SCHEMA or not state_equal(spec["optimizer"], LOCKED_OPTIMIZER)
            or spec["device"] != "cpu" or spec["precision"] != "fp32"
            or type(spec["cpu_threads"]) is not int or spec["cpu_threads"] != 2
            or spec["source_kind"] not in ("diagnostic_fixture", "strict_collector")
            or spec["eligibility"] != ELIGIBILITY):
        raise ValueError("unsupported nonzero optimizer/device/source/eligibility")
    ModelConfig(**spec["config"]).validate()
    _identifier(spec["training_run_id"])
    for name in ("implementation_sha256", "dataset_sha256", "split_sha256"):
        _sha(spec[name])
    for name, maximum in (("max_updates", MAX_UPDATES), ("max_wall_time_ms", MAX_WALL_MS),
                          ("cleanup_time_ms", 30000), ("max_output_bytes", MAX_OUTPUT_BYTES)):
        if not 1 <= _uint(spec[name], maximum):
            raise ValueError("nonzero smoke finite limit required")


def spec_sha256(spec):
    validate_spec(spec)
    return _identity(SPEC_SCHEMA, spec)


class TrainingLedger:
    """One global finite allowance, including updates before a save/resume.

    A dispatched update consumes the update allowance even if it fails. The
    completed counter records only actual successful optimizer.step calls.
    This ledger does not claim measured FLOPs or GPU execution.
    """
    def __init__(self, spec, usage=None, *, clock=time.monotonic):
        validate_spec(spec)
        self.spec = copy.deepcopy(spec)
        self.usage = dict.fromkeys(USAGE_FIELDS, 0) if usage is None else dict(
            _fields(usage, USAGE_FIELDS, "nonzero usage"))
        self.clock, self.started = clock, clock()
        self.initial_wall_ms = self.usage["wall_time_ms"]
        self.canceled = False
        self.progress_path = None
        self._check(self.usage)

    def _check(self, usage):
        for value in _fields(usage, USAGE_FIELDS, "nonzero usage").values():
            _uint(value)
        if (usage["updates_completed"] > usage["updates_dispatched"]
                or usage["updates_dispatched"] > self.spec["max_updates"]
                or usage["output_bytes"] > self.spec["max_output_bytes"]
                or usage["wall_time_ms"] > self.spec["max_wall_time_ms"]
                or usage["samples"] != usage["forward_calls"]
                or usage["backward_calls"] > usage["forward_calls"]):
            raise ValueError("nonzero smoke budget/counter boundary exceeded")

    def snapshot_usage(self):
        elapsed = self.clock() - self.started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("nonzero monotonic clock moved backwards/nonfinite")
        self.usage["wall_time_ms"] = max(self.usage["wall_time_ms"],
                                         self.initial_wall_ms + math.ceil(1000 * elapsed))
        self._check(self.usage)
        return dict(self.usage)

    def reserve(self, *, forward=False, backward=False, update=False, output_bytes=0):
        if self.canceled:
            raise ValueError("nonzero smoke canceled; further work refused")
        value = self.snapshot_usage()
        if forward:
            value["forward_calls"] += 1
            value["samples"] += 1
        if backward:
            value["backward_calls"] += 1
        if update:
            value["updates_dispatched"] += 1
        value["output_bytes"] += _uint(output_bytes)
        self._check(value)
        self.usage = value

    def persist_progress(self, boundary, role, steps):
        """Durable global allowance before dispatch and after actual completion.

        If a process is killed between these boundaries, dispatched usage is
        the conservative consumed allowance. A new run may not reset it.
        """
        if self.progress_path is None:
            return
        if boundary not in ("optimizer_dispatched", "optimizer_completed"):
            raise ValueError("nonzero progress requires an exact optimizer boundary")
        report = {"schema": "rz-pals-python-nonzero-progress/1", "training_run_id": self.spec["training_run_id"],
                  "spec_sha256": spec_sha256(self.spec), "boundary": boundary, "role": role,
                  "training_steps": steps, "usage": {}, "canceled": self.canceled, "arena_eligible": False}
        receipt_bytes = 0
        for _ in range(8):
            report["usage"] = self.snapshot_usage()
            report["usage"]["output_bytes"] += receipt_bytes
            self._check(report["usage"])
            next_size = len(_json_bytes(report))
            if receipt_bytes == next_size:
                break
            receipt_bytes = next_size
        else:
            raise ValueError("nonzero progress receipt size did not stabilize")
        # Each boundary is an immutable registered artifact. Preserve both
        # pre-dispatch and completed evidence without overwriting either.
        path = self.progress_path.with_name(self.progress_path.stem +
                    f"-{self.usage['updates_dispatched']:02d}-{boundary}" + self.progress_path.suffix)
        if path.exists():
            raise FileExistsError("nonzero progress boundary already registered")
        self.reserve(output_bytes=receipt_bytes)
        atomic_json(path, report)


@dataclass
class NonzeroState:
    preparation: OptimizerPreparation
    scheduler: object
    steps: int = 0


def _new_state(model, role):
    prepared = prepare_nonzero_adamw(model, role)
    scheduler = torch.optim.lr_scheduler.LambdaLR(prepared.optimizer, lr_lambda=lambda _: 1.0)
    return NonzeroState(prepared, scheduler)


def _cpu_model(model):
    model.config.validate()
    if any(value.device.type != "cpu" or value.dtype != torch.float32
           or not bool(torch.all(torch.isfinite(value))) for value in model.state_dict().values()):
        raise ValueError("nonzero smoke requires finite CPU FP32 model state")


def initialize_training(model, role, spec):
    validate_spec(spec)
    _cpu_model(model)
    if model.config.to_dict() != spec["config"]:
        raise ValueError("nonzero model configuration differs from locked spec")
    # Responsibility/admission errors are rejected on a scratch object before
    # changing caller gradients or trainability.
    _new_state(copy.deepcopy(model), role)
    return _new_state(model, role)


def state_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    return left == right


def _state_digest(value):
    """Exact typed recursive state digest; not torch.save ZIP lexical identity."""
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            descriptor = ["tensor", str(item.dtype), list(item.shape)]
            result.update(json.dumps(descriptor, separators=(",", ":")).encode())
            result.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b"dict")
            for key in sorted(item, key=lambda key: (str(type(key)), str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update((type(item).__name__ + str(len(item))).encode())
            for child in item:
                visit(child)
        else:
            result.update(json.dumps([type(item).__name__, item], allow_nan=False,
                                     separators=(",", ":")).encode())
        result.update(b";")
    visit(value)
    return result.hexdigest()


def comparison_snapshot(model, state, sampler):
    """Use the same owned typed snapshot on both sides of resume comparison.

    state_dict returns OrderedDict while the checkpoint clone is a plain dict.
    Normalize both through the existing clone before strict value comparison;
    no tensor dtype, shape, value or optimizer setting tolerance is introduced.
    """
    return {"model": _clone_state(model.state_dict()),
            "optimizer": _clone_state(state.preparation.optimizer.state_dict()),
            "scheduler": _clone_state(state.scheduler.state_dict()),
            "rng": capture_rng(), "sampler": sampler.state()}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_bounded_report(path, report, ledger):
    size = 0
    for _ in range(8):
        report["usage"] = ledger.snapshot_usage()
        report["usage"]["output_bytes"] += size
        ledger._check(report["usage"])
        next_size = len(_json_bytes(report))
        if next_size == size:
            break
        size = next_size
    else:
        raise ValueError("nonzero report size did not stabilize")
    ledger.reserve(output_bytes=size)
    atomic_json(path, report)
    ledger.snapshot_usage()


def _check_optimizer(model, state):
    """Validate names, freeze, group settings, actual moments and scalar steps."""
    prepared = state.preparation
    if (prepared.phase != "bounded_nonzero_smoke" or not 0 <= _uint(state.steps, MAX_UPDATES)
            or any(parameter.grad is not None for parameter in model.parameters())):
        raise ValueError("nonzero checkpoint requires a complete no-gradient update boundary")
    expected = _new_state(copy.deepcopy(model), prepared.role)
    if prepared.parameter_names != expected.preparation.parameter_names or not state_equal(prepared.observed_freeze, expected.preparation.observed_freeze):
        raise ValueError("nonzero parameter names/freeze differ from selected responsibility")
    actual_active = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if actual_active != prepared.parameter_names:
        raise ValueError("nonzero actual freeze flags differ from selected responsibility")
    by_object = {id(parameter): (name, parameter) for name, parameter in model.named_parameters()}
    flat = [parameter for group in prepared.optimizer.param_groups for parameter in group["params"]]
    if len({id(value) for value in flat}) != len(flat) or {id(value) for value in flat} != {
            id(parameter) for name, parameter in model.named_parameters() if name in actual_active}:
        raise ValueError("nonzero optimizer actual parameter membership mismatch")
    actual, wanted = prepared.optimizer.state_dict(), expected.preparation.optimizer.state_dict()
    if not state_equal(actual["param_groups"], wanted["param_groups"]):
        raise ValueError("nonzero optimizer groups/options/order differ from locked AdamW")
    tensors = {slot: by_object[id(parameter)][1]
               for group, encoded in zip(prepared.optimizer.param_groups, actual["param_groups"])
               for slot, parameter in zip(encoded["params"], group["params"])}
    if not isinstance(actual["state"], dict) or set(actual["state"]) - set(tensors):
        raise ValueError("nonzero optimizer moment membership mismatch")
    if any(type(slot) is not int for slot in actual["state"]):
        raise ValueError("nonzero optimizer parameter slots must be exact integers")
    if state.steps == 0 and actual["state"]:
        raise ValueError("zero actual step cannot carry AdamW moments")
    if state.steps > 0 and not actual["state"]:
        raise ValueError("nonzero actual step requires AdamW moments")
    observed_steps = []
    for slot, values in actual["state"].items():
        _fields(values, ("step", "exp_avg", "exp_avg_sq"), "actual AdamW moments")
        step = values["step"]
        if (not isinstance(step, torch.Tensor) or step.shape != () or step.dtype != torch.float32
                or step.device.type != "cpu" or not bool(torch.isfinite(step))
                or float(step) != int(float(step)) or not 1 <= int(float(step)) <= state.steps):
            raise ValueError("invalid actual AdamW scalar step")
        observed_steps.append(int(float(step)))
        for name in ("exp_avg", "exp_avg_sq"):
            value = values[name]
            if (not isinstance(value, torch.Tensor) or value.shape != tensors[slot].shape
                    or value.dtype != torch.float32 or value.device.type != "cpu"
                    or not bool(torch.all(torch.isfinite(value)))
                    or (name == "exp_avg_sq" and bool(torch.any(value < 0)))):
                raise ValueError("nonzero optimizer moment shape/dtype/value mismatch")
    if state.steps and max(observed_steps) != state.steps:
        raise ValueError("nonzero declared step exceeds actual AdamW moments")
    expected.scheduler.last_epoch = state.steps
    expected.scheduler._step_count = state.steps + 1
    if not state_equal(state.scheduler.state_dict(), expected.scheduler.state_dict()):
        raise ValueError("nonzero constant scheduler state/step differs from optimizer")
    return actual


def valid_target_counts(batch):
    if not isinstance(batch, TrainingBatch):
        raise ValueError("nonzero update requires validated TrainingBatch")
    rows = batch.inputs.board.shape[0]
    slots = batch.inputs.divergence_mask.shape[1]
    masks = ((batch.policy_mask, (rows,)), (batch.wdl_mask, (rows,)),
             (batch.divergence_mask, (rows, slots)), (batch.task_mask, (rows, len(TASKS))))
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.shape != shape
           for value, shape in masks):
        raise ValueError("nonzero loss masks require exact bool head shapes")
    if batch.role == "proposer":
        return {"policy": int(batch.policy_mask.sum()), "wdl": int(batch.wdl_mask.sum())}
    if batch.role == "critic":
        return {"policy": int(batch.policy_mask.sum()), "wdl": int(batch.wdl_mask.sum()),
                "divergence": int(batch.divergence_mask.sum())}
    if batch.role == "validator":
        return {"task": int(batch.task_mask.sum())}
    raise ValueError("nonzero unknown hard-routed role")


def run_validated_update(model, state, dataset, index, ledger, *, task_context=None):
    """A non-diagnostic target must come through the strict current collator.

    No fabricated TrainingBatch can enroll itself as collected training data.
    Holdout, stale rows, unknown masks, unregistered profile and absent V
    context are rejected by the same existing dataset boundaries.
    """
    from .training import ValidatedDataset
    if (type(dataset) is not ValidatedDataset or dataset.frozen_admission is None
            or ledger.spec["source_kind"] != "strict_collector"):
        raise ValueError("actual update requires a strict frozen dataset and explicit actual spec")
    dataset._verify_raw_integrity()
    if (dataset.frozen_admission["raw_dataset_sha256"] != ledger.spec["dataset_sha256"]
            or dataset.frozen_admission["split_sha256"] != ledger.spec["split_sha256"]):
        raise ValueError("actual update dataset/split differs from independently locked spec")
    role = state.preparation.role
    batch = dataset.collate([index], role, split="train", device="cpu",
                            task_contexts=[task_context] if role == "verifier" else None)
    result = train_one_update(model, state, batch, ledger, _target_capability=_ACTUAL_TARGET_CAPABILITY)
    dataset._verify_raw_integrity()
    result["actual_input_sha256"] = batch.input_sha256[0]
    result["current_view_sha256"] = dataset.current_view.sha256
    result["source_kind"] = "strict_collector"
    return result


def train_one_update(model, state, batch, ledger, *, _target_capability=None):
    """One actual update; an all-masked sample is rejected before backward/step."""
    if ledger.spec["source_kind"] == "strict_collector" and _target_capability is not _ACTUAL_TARGET_CAPABILITY:
        raise ValueError("actual collected targets require run_validated_update; no diagnostic fallback")
    if ledger.spec["optimizer"] != LOCKED_OPTIMIZER or batch.inputs.board.shape[0] != 1:
        raise ValueError("nonzero smoke requires locked optimizer and batch one")
    if batch.role != {"proposer": "proposer", "critic": "critic", "verifier": "validator"}.get(state.preparation.role):
        raise ValueError("nonzero batch hard route differs from optimizer responsibility")
    if batch.model_profile != getattr(model.config, "profile", "legacy_summary_v1"):
        raise ValueError("nonzero input model/encoding profile differs from model")
    batch.inputs.validate(model.config)
    if any(value.device.type != "cpu" for value in batch.inputs.__dict__.values() if value is not None):
        raise ValueError("nonzero smoke input requires explicitly selected CPU; no provider fallback")
    counts = valid_target_counts(batch)
    if not any(counts.values()):
        raise ValueError("nonzero update refused: all loss masks are unknown/ineligible")
    _cpu_model(model)
    _check_optimizer(model, state)
    frozen = {name: value.detach().clone() for name, value in model.named_parameters()
              if not value.requires_grad}
    optimizer = state.preparation.optimizer
    ledger.reserve(forward=True)
    try:
        model.train()
        memory = model.public_encoder(*batch.inputs.public_args(model.config))
        outputs = model.role_graph(batch.role)(*batch.inputs.role_args(memory, model.config))
        losses = masked_losses(outputs, batch)
        if any(value.shape != () or not bool(torch.isfinite(value)) for value in losses.values()):
            raise ValueError("nonzero forward has nonscalar/nonfinite loss")
        ledger.reserve(backward=True)
        losses["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or any(not bool(torch.all(torch.isfinite(value))) for value in gradients):
            raise ValueError("nonzero backward has missing/nonfinite gradients")
        if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
            raise ValueError("nonzero backward reached a frozen parameter group")
        norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters()
                                             if parameter.requires_grad], 1.0, error_if_nonfinite=True)
        ledger.reserve(update=True)
        ledger.persist_progress("optimizer_dispatched", state.preparation.role, state.steps)
        optimizer.step()
        ledger.usage["updates_completed"] += 1
        state.steps += 1
        ledger.persist_progress("optimizer_completed", state.preparation.role, state.steps)
        state.scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        _cpu_model(model)
        if any(not torch.equal(value, dict(model.named_parameters())[name]) for name, value in frozen.items()):
            raise ValueError("nonzero actual frozen parameter group changed")
        ledger.snapshot_usage()
        return {"step": state.steps, "valid_targets": counts,
                "losses": {name: float(value.detach()) for name, value in losses.items()},
                "gradient_norm_before_clip": float(norm), "frozen_groups_unchanged": True,
                "finite_loss": True, "finite_gradients": True, "actual_optimizer_update": True}
    except BaseException:
        ledger.canceled = True
        raise


def save_nonzero_checkpoint(path, model, state, sampler, ledger):
    """Atomic nonzero state with actual moments, scheduler, RNG and sampler."""
    _cpu_model(model)
    if state.steps == 0:
        raise ValueError("nonzero checkpoint requires at least one actual optimizer update")
    if sampler.role != state.preparation.role or model.config.to_dict() != ledger.spec["config"]:
        raise ValueError("nonzero checkpoint sampler/model/spec mismatch")
    optimizer = _check_optimizer(model, state)
    usage = ledger.snapshot_usage()
    if usage["updates_completed"] < state.steps:
        raise ValueError("nonzero checkpoint lost actual global update usage")
    tensor_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    tensor_bytes += sum(value.numel() * value.element_size() for moments in optimizer["state"].values()
                        for value in moments.values())
    if tensor_bytes > ledger.spec["max_output_bytes"] - usage["output_bytes"]:
        raise ValueError("nonzero checkpoint tensor allocation exceeds remaining output allowance")
    payload = {"schema": CHECKPOINT_SCHEMA, "spec": copy.deepcopy(ledger.spec),
               "spec_sha256": spec_sha256(ledger.spec), "config": model.config.to_dict(),
               "role": state.preparation.role, "phase": state.preparation.phase,
               "parameter_names": list(state.preparation.parameter_names),
               "observed_freeze": dict(state.preparation.observed_freeze),
               "requires_grad": {name: parameter.requires_grad for name, parameter in model.named_parameters()},
               "model": _clone_state(model.state_dict()), "optimizer": _clone_state(optimizer),
               "scheduler": _clone_state(state.scheduler.state_dict()), "rng": capture_rng(),
               "sampler": sampler.state(), "usage": usage, "canceled": ledger.canceled,
               "training_steps": state.steps, "training_executed": True,
               "eligibility": ELIGIBILITY, "arena_eligible": False}
    size = 0
    for _ in range(8):
        payload["usage"] = ledger.snapshot_usage()
        payload["usage"]["output_bytes"] += size
        ledger._check(payload["usage"])
        stream = io.BytesIO()
        torch.save(payload, stream)
        data = stream.getvalue()
        if len(data) == size:
            break
        size = len(data)
    else:
        raise ValueError("nonzero checkpoint serialization size did not stabilize")
    from .preparation_check import _registered_path
    path = _registered_path(path)
    if path.suffix != ".pt":
        raise ValueError("nonzero checkpoint must use .pt")
    output_directory(path.parent)
    temporary = path.with_name(path.name + ".partial")
    receipt_path = path.with_name(path.stem + "-receipt.json")
    if path.exists() or temporary.exists() or receipt_path.exists() or receipt_path.with_name(receipt_path.name + ".partial").exists():
        raise FileExistsError("nonzero checkpoint/incomplete artifact already exists")
    ledger.reserve(output_bytes=len(data))
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    finalized = ledger.snapshot_usage()
    receipt = {"schema": CHECKPOINT_SCHEMA, "path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "spec_sha256": payload["spec_sha256"], "role": payload["role"],
            "training_steps": state.steps, "training_executed": True, "usage": finalized,
            "canceled": ledger.canceled, "eligibility": ELIGIBILITY, "arena_eligible": False,
            "receipt_bytes": 0}
    receipt_size = 0
    for _ in range(8):
        receipt["receipt_bytes"] = receipt_size
        receipt["usage"] = ledger.snapshot_usage()
        receipt["usage"]["output_bytes"] += receipt_size
        ledger._check(receipt["usage"])
        next_size = len(_json_bytes(receipt))
        if next_size == receipt_size:
            break
        receipt_size = next_size
    else:
        raise ValueError("nonzero final usage receipt size did not stabilize")
    ledger.reserve(output_bytes=receipt_size)
    atomic_json(receipt_path, receipt)
    # The original run ledger continues through this receipt's own write cost.
    ledger.snapshot_usage()
    return receipt


def load_nonzero_checkpoint(path, expected_sha256, model, spec, *, usage_receipt, indices, role):
    """Reject every inconsistent state before mutating caller model/RNG.

    Receipt counters close file/hash time beyond the immutable snapshot. All
    archive allocation, parameter, optimizer, scheduler, sampler, freeze and
    RNG checks run against a scratch model before the commit boundary.
    """
    validate_spec(spec)
    _cpu_model(model)
    _sha(expected_sha256)
    from .preparation_check import _registered_path
    path = _registered_path(path)
    if path.suffix != ".pt":
        raise ValueError("nonzero checkpoint must use .pt")
    raw = _CollectionArtifactReader(min(spec["max_output_bytes"], 1024**3)).read(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("nonzero checkpoint bytes differ from independently expected SHA")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if (not entries or len(entries) > 65536 or len({entry.filename for entry in entries}) != len(entries)
                or any(entry.compress_type != zipfile.ZIP_STORED for entry in entries)
                or sum(entry.file_size for entry in entries) > spec["max_output_bytes"]):
            raise ValueError("nonzero checkpoint archive allocation/identity exceeds admission")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    _fields(payload, ("schema", "spec", "spec_sha256", "config", "role", "phase", "parameter_names",
                      "observed_freeze", "requires_grad", "model", "optimizer", "scheduler", "rng",
                      "sampler", "usage", "canceled", "training_steps", "training_executed",
                      "eligibility", "arena_eligible"), "nonzero complete checkpoint")
    if (payload["schema"] != CHECKPOINT_SCHEMA or payload["spec"] != spec
            or payload["spec_sha256"] != spec_sha256(spec) or payload["config"] != model.config.to_dict()
            or payload["config"] != spec["config"] or payload["role"] != role
            or payload["phase"] != "bounded_nonzero_smoke" or payload["training_executed"] is not True
            or type(payload["canceled"]) is not bool or payload["eligibility"] != ELIGIBILITY
            or payload["arena_eligible"] is not False or not 1 <= _uint(payload["training_steps"], MAX_UPDATES)):
        raise ValueError("nonzero checkpoint domain/config/role/step/eligibility mismatch")
    _fields(usage_receipt, ("schema", "path", "sha256", "bytes", "spec_sha256", "role", "training_steps",
                           "training_executed", "usage", "canceled", "eligibility", "arena_eligible", "receipt_bytes"),
            "nonzero final usage receipt")
    if (usage_receipt["schema"] != CHECKPOINT_SCHEMA or usage_receipt["sha256"] != expected_sha256
            or usage_receipt["bytes"] != len(raw) or usage_receipt["spec_sha256"] != payload["spec_sha256"]
            or usage_receipt["role"] != role or usage_receipt["training_steps"] != payload["training_steps"]
            or usage_receipt["training_executed"] is not True or usage_receipt["canceled"] is not payload["canceled"]
            or usage_receipt["eligibility"] != ELIGIBILITY or usage_receipt["arena_eligible"] is not False):
        raise ValueError("nonzero final usage receipt identity mismatch")
    persisted = _fields(payload["usage"], USAGE_FIELDS, "nonzero snapshot usage")
    finalized = _fields(usage_receipt["usage"], USAGE_FIELDS, "nonzero final usage")
    if (any(finalized[name] != persisted[name] for name in USAGE_FIELDS if name not in ("wall_time_ms", "output_bytes"))
            or _uint(finalized["output_bytes"]) != _uint(persisted["output_bytes"]) + _uint(usage_receipt["receipt_bytes"])
            or usage_receipt["receipt_bytes"] != len(_json_bytes(usage_receipt))
            or _uint(finalized["wall_time_ms"]) < _uint(persisted["wall_time_ms"])):
        raise ValueError("nonzero receipt lost elapsed time or changed non-I/O counters")
    ledger = TrainingLedger(spec, finalized)
    if ledger.usage["updates_completed"] < payload["training_steps"]:
        raise ValueError("nonzero resume lost actual global update usage")
    sampler = ResumableSampler.restore(payload["sampler"], indices=indices, role=role)
    current = model.state_dict()
    if (not isinstance(payload["model"], dict) or set(payload["model"]) != set(current)
            or any(not isinstance(value, torch.Tensor) or value.shape != current[name].shape
                   or value.dtype != current[name].dtype or not bool(torch.all(torch.isfinite(value)))
                   for name, value in payload["model"].items())):
        raise ValueError("nonzero model parameter shape/dtype/value mismatch")
    scratch = copy.deepcopy(model)
    scratch.load_state_dict(payload["model"], strict=True)
    checked = _new_state(scratch, role)
    checked.steps = payload["training_steps"]
    if (list(checked.preparation.parameter_names) != payload["parameter_names"]
            or not state_equal(checked.preparation.observed_freeze, payload["observed_freeze"])
            or not state_equal({name: parameter.requires_grad for name, parameter in scratch.named_parameters()}, payload["requires_grad"])):
        raise ValueError("nonzero checkpoint names/freeze responsibility mismatch")
    # load_state_dict can silently coerce tensor dtype/settings; compare raw
    # groups first, and validate raw moment shape/value before accepting casts.
    optimizer = _fields(payload["optimizer"], ("state", "param_groups"), "nonzero optimizer")
    if not state_equal(optimizer["param_groups"], checked.preparation.optimizer.state_dict()["param_groups"]):
        raise ValueError("nonzero raw optimizer group/settings mismatch")
    for moments in optimizer["state"].values():
        _fields(moments, ("step", "exp_avg", "exp_avg_sq"), "raw actual AdamW moments")
        if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
               or not bool(torch.all(torch.isfinite(value))) for value in moments.values()):
            raise ValueError("nonzero raw optimizer moment dtype/value mismatch")
    checked.preparation.optimizer.load_state_dict(optimizer)
    checked.scheduler.load_state_dict(payload["scheduler"])
    _check_optimizer(scratch, checked)
    _validate_rng(payload["rng"])
    # All external input has passed. Known shapes/membership guarantee these
    # deterministic in-memory copies cannot introduce an input rejection.
    state = _new_state(model, role)
    state.preparation.optimizer.load_state_dict(optimizer)
    state.scheduler.load_state_dict(payload["scheduler"])
    model.load_state_dict(payload["model"], strict=True)
    state.steps = payload["training_steps"]
    restore_rng(payload["rng"])
    ledger.canceled = payload["canceled"]
    return state, sampler, ledger


def _diagnostic_batch(model, role, index):
    from .model import fixture_input
    profile = getattr(model.config, "profile", "legacy_summary_v1")
    inputs = fixture_input(records=2, candidates=4, divergences=2, batch=1, profile=profile)
    # P includes both proposal and Repair query invocations. These are numeric
    # diagnostic inputs, not collected or Rules-certified Repair trajectories.
    inputs.query[0, 0] = float(index % 2 == 0)
    policy = torch.tensor([[0.6, 0.25, 0.1, 0.05]], dtype=torch.float32)
    wdl = torch.tensor([[0.3, 0.5, 0.2]], dtype=torch.float32)
    divergence = torch.tensor([[float(index % 2), float((index + 1) % 2)]], dtype=torch.float32)
    task = torch.zeros((1, len(TASKS)), dtype=torch.float32)
    task_mask = torch.zeros_like(task, dtype=torch.bool)
    for name, probability in (("resume_task", 0.6), ("cross_profile_recheck", 0.3), ("defer", 0.1)):
        task[0, TASKS.index(name)] = probability
        task_mask[0, TASKS.index(name)] = role == "verifier"
    if role == "verifier":
        policy.zero_()
        wdl.zero_()
    return TrainingBatch("validator" if role == "verifier" else role, inputs,
                         (_identity("diagnostic-row", [role, index]),), policy,
                         torch.tensor([role != "verifier"]), wdl, torch.tensor([role != "verifier"]),
                         divergence, torch.tensor([[role == "critic", role == "critic"]]), task, task_mask, profile)


def add_train_step_smoke_parser(subparsers):
    parser = subparsers.add_parser("train-step-smoke", help="bounded 24-update CPU diagnostic; arena ineligible")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--profile", default="full_line_interaction_v2", choices=(
        "legacy_summary_v1", "full_line_v2", "interaction_head_v2", "full_line_interaction_v2"))
    parser.add_argument("--coverage-collection")
    parser.add_argument("--coverage-receipt-sha256")
    parser.add_argument("--producer-registration-set")
    parser.add_argument("--producer-registration-set-sha256")
    return parser


def _actual_coverage(args, ledger):
    names = ("coverage_collection", "coverage_receipt_sha256", "producer_registration_set",
             "producer_registration_set_sha256")
    selected = [getattr(args, name, None) for name in names]
    if not any(selected):
        return {"status": "not_supplied", "source_kind": "actual_collector",
                "actual_target_coverage_claimed": False, "actual_target_updates_executed": 0,
                "current_rows": {"observation": "unknown", "reason": "actual V2 collector data not supplied"},
                "supported_private_v_utility": {"observation": "unknown", "reason": "actual V2 collector data not supplied"}}
    if not all(selected):
        raise ValueError("actual coverage requires collection, receipt SHA and independent registration set/SHA")
    from .preparation_check import _registered_path, _registration_set
    from .target_coverage import target_coverage_report
    from .training import load_frozen_collected_dataset
    collection = _registered_path(args.coverage_collection)
    registrations, pin = _registration_set(args.producer_registration_set,
                                           args.producer_registration_set_sha256, 4 << 20)
    dataset = load_frozen_collected_dataset(collection, expected_receipt_sha256=args.coverage_receipt_sha256,
                                            producer_registrations=registrations, max_input_bytes=64 << 20)
    ledger.snapshot_usage()
    report = target_coverage_report(dataset, source_kind="actual_collector")
    report["registration_set_pin"] = pin
    ledger.snapshot_usage()
    return report


def run_train_step_smoke(args):
    """Exactly continuous4 vs 2+save+resume2 for P/C/V: 24 actual updates.

    Run under the caller's process watchdog (15 min, cleanup 30 sec, CPU 2).
    No retry or smaller model/target fallback is performed after a failure.
    """
    started = time.monotonic()
    _uint(args.seed, 2**63 - 1)
    from .model import initialize
    config = ModelConfig.for_profile(args.profile)
    import uuid
    module_root = Path(__file__).parent
    sources = {name: hashlib.sha256((module_root / name).read_bytes()).hexdigest() for name in (
        "nonzero_training.py", "training.py", "target_coverage.py", "config.py", "model.py")}
    spec = smoke_spec(config, implementation_sha256=_identity("nonzero-source-inputs", sources),
                      dataset_sha256=_identity("nonzero-diagnostic-dataset", [1, 2, 4, list(TASKS)]),
                      split_sha256=_identity("nonzero-diagnostic-split", "diagnostic_train_only"),
                      training_run_id=str(uuid.uuid4()))
    ledger = TrainingLedger(spec)
    ledger.started = started
    output = output_directory(args.output)
    ledger.reserve(output_bytes=(output / "pals-artifact-owner.json").stat().st_size)
    ledger.progress_path = output / "nonzero-smoke-progress.json"
    report = {"schema": REPORT_SCHEMA, "spec": spec, "spec_sha256": spec_sha256(spec),
              "status": "running", "diagnostic_target_source": "synthetic_numeric_fixture_only",
              "training_run_id": spec["training_run_id"], "checkpoint_domain": CHECKPOINT_SCHEMA,
              "source_inputs_sha256": sources,
              "eligibility": ELIGIBILITY, "arena_eligible": False, "roles": {},
              "actual_training_executed": False, "full_learning_claimed": False,
              "learned_verifier_strength_claimed": False, "gpu_executed": False}
    original_rng, original_threads = capture_rng(), torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        report["actual_target_coverage"] = _actual_coverage(args, ledger)
        for role in ("proposer", "critic", "verifier"):
            initial = initialize(args.seed, config)
            start_rng = capture_rng()
            continuous = copy.deepcopy(initial)
            continuous_state = initialize_training(continuous, role, spec)
            continuous_sampler = ResumableSampler(tuple(range(4)), args.seed, role=role)
            continuous_steps, split_steps = [], []
            for _ in range(4):
                index = continuous_sampler.next_batch(1)[0]
                continuous_steps.append(train_one_update(continuous, continuous_state,
                                                          _diagnostic_batch(continuous, role, index), ledger))
            reference = comparison_snapshot(continuous, continuous_state, continuous_sampler)
            del continuous, continuous_state, continuous_sampler
            restore_rng(start_rng)
            split_model = initial
            split_state = initialize_training(split_model, role, spec)
            split_sampler = ResumableSampler(tuple(range(4)), args.seed, role=role)
            for _ in range(2):
                index = split_sampler.next_batch(1)[0]
                split_steps.append(train_one_update(split_model, split_state,
                                                    _diagnostic_batch(split_model, role, index), ledger))
            receipt = save_nonzero_checkpoint(output / (role + "-step2.pt"), split_model, split_state,
                                              split_sampler, ledger)
            resumed_model = initialize(args.seed + 1, config)
            resumed, sampler, resumed_ledger = load_nonzero_checkpoint(
                receipt["path"], receipt["sha256"], resumed_model, spec,
                usage_receipt=receipt, indices=tuple(range(4)), role=role)
            # The global run ledger also charges checkpoint reload/validation;
            # do not reset its original wall clock by using a new role ledger.
            if any(resumed_ledger.usage[name] != ledger.usage[name] for name in USAGE_FIELDS if name != "wall_time_ms"):
                raise ValueError("nonzero resume changed global usage counters")
            del split_model, split_state, split_sampler, resumed_ledger
            for _ in range(2):
                index = sampler.next_batch(1)[0]
                split_steps.append(train_one_update(resumed_model, resumed,
                                                    _diagnostic_batch(resumed_model, role, index), ledger))
            candidate = comparison_snapshot(resumed_model, resumed, sampler)
            match = {name: state_equal(reference[name], candidate[name]) for name in reference}
            # Retain observed components before any mismatch raises. Failure
            # evidence must not erase completed work or which comparison failed.
            report["roles"][role] = {"actual_updates": 8, "continuous_steps": continuous_steps,
                                      "resumed_steps": split_steps, "state_matches": match,
                                      "reference_state_sha256": {name: _state_digest(value) for name, value in reference.items()},
                                      "candidate_state_sha256": {name: _state_digest(value) for name, value in candidate.items()},
                                      "step2_checkpoint_sha256": receipt["sha256"],
                                      "comparison_status": "passed" if all(match.values()) else "failed",
                                      "freeze": resumed.preparation.observed_freeze,
                                      "parameter_names": list(resumed.preparation.parameter_names)}
            if not all(match.values()):
                raise ValueError("continuous/resume model/optimizer/scheduler/RNG/sampler mismatch")
            final_receipt = save_nonzero_checkpoint(output / (role + "-step4.pt"), resumed_model,
                                                    resumed, sampler, ledger)
            report["roles"][role].update({"actual_updates": 8, "continuous_steps": continuous_steps,
                                      "resumed_steps": split_steps, "state_matches": match,
                                      "final_state_sha256": {name: _state_digest(value) for name, value in candidate.items()},
                                      "checkpoint_sha256": final_receipt["sha256"],
                                      "training_provenance": {"checkpoint_domain": CHECKPOINT_SCHEMA,
                                                              "training_run_id": spec["training_run_id"],
                                                              "completed_updates": 4,
                                                              "target_coverage": "diagnostic_fixture",
                                                              "target_coverage_sha256": spec["dataset_sha256"],
                                                              "arena_eligible": False},
                                      "repair_numeric_inputs_included": role == "proposer",
                                      "freeze": resumed.preparation.observed_freeze,
                                      "parameter_names": list(resumed.preparation.parameter_names)})
            report["actual_training_executed"] = True
            del resumed_model, resumed, sampler, reference, candidate
        if ledger.usage["updates_completed"] != 24 or ledger.usage["updates_dispatched"] != 24:
            raise ValueError("nonzero smoke did not complete exactly the approved 24 actual updates")
        report["status"] = "passed_bounded_optimizer_resume_diagnostic"
        report["usage"] = ledger.snapshot_usage()
    except BaseException as error:
        ledger.canceled = True
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        report["actual_training_executed"] = ledger.usage["updates_completed"] > 0
        report["usage"] = dict(ledger.usage)
        raise
    finally:
        restore_rng(original_rng)
        torch.set_num_threads(original_threads)
        restored_rng = capture_rng()
        report["caller_rng_state_sha256"] = _state_digest(original_rng)
        report["restored_caller_rng_state_sha256"] = _state_digest(restored_rng)
        report["caller_rng_preserved"] = state_equal(original_rng, restored_rng)
        report["caller_thread_count_preserved"] = torch.get_num_threads() == original_threads
        if report["status"] == "failed":
            # Failure evidence stays distinct from a success receipt. Cleanup
            # restores caller state but cannot authorize any optimizer retry.
            failure_bytes = 0
            for _ in range(8):
                report["usage"] = dict(ledger.usage)
                report["usage"]["output_bytes"] += failure_bytes
                next_size = len(_json_bytes(report))
                if next_size == failure_bytes:
                    break
                failure_bytes = next_size
            if failure_bytes <= spec["max_output_bytes"] - ledger.usage["output_bytes"]:
                ledger.usage["output_bytes"] += failure_bytes
                atomic_json(output / "nonzero-smoke-failure.json", report)
    if not report["caller_rng_preserved"] or not report["caller_thread_count_preserved"]:
        raise ValueError("nonzero smoke failed to preserve caller RNG/thread state")
    _write_bounded_report(output / "nonzero-smoke-report.json", report, ledger)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_train_step_smoke_parser(commands)
    result = run_train_step_smoke(parser.parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
