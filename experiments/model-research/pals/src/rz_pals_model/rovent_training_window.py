"""Bounded Rovent learner dispatch, separate from the closed PALS G0 trainer.

Import, recipe validation and execute=False admission use the standard library.
The execute=True path is source only until its original host/producer authority
and target device have been independently accepted. This module never obtains
execution authority from serialized corpus claims or preparation receipts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

from .generation_data import digest, fields, identifier, sha, uint
from .generation_training import OPTIMIZER, _typed_equal, loss_sums
from .rovent_generation import SeriesRecipe
from .rovent_tensor_training import (BUCKETS, MAX_BYTES, ROLES, OwnedTensorState,
    _RNG_LOCK, _load_pinned_tensor_file, _restore_scratch, _owner_component, _runtime,
    _validate_json_state, _nnuec_tensor_training_state, save_tensor_checkpoint, tensor_payload)
from .generation_export import COST_FIELDS
from .rovent_nnuec_training_state import nnuec_is_unmodified, nnuec_known_updates

WINDOW_DOMAIN = "rz-rovent-train-window/1"
SAMPLER_DOMAIN = "rz-rovent-native-stage-sampler/1"
JOURNAL_DOMAIN = "rz-rovent-original-learner-journal/1"
HEADS = ("policy", "wdl", "divergence", "task", "nnuec_wdl")


def retained_seal_set(values, kind):
    """Compact RPC proof; full source arrays remain in checkpoint/custody."""
    if kind not in ("input", "origin") or type(values) is not list or len(values) > 262144:
        raise ValueError("bounded typed retained seal set required")
    ordered = sorted(sha(value) for value in values)
    if len(set(ordered)) != len(ordered):
        raise ValueError("retained seal set contains duplicate origins/inputs")
    result = hashlib.sha256(b"rz-rovent-training-seal-set/1\0" + kind.encode("ascii") + b"\0" + len(ordered).to_bytes(8, "little"))
    for value in ordered:
        result.update(bytes.fromhex(value))
    return {"sha256": result.hexdigest(), "count": len(ordered)}


class OriginalTrainingAuthorityRequired(ValueError):
    """Raw files are valid evidence, but are not an original process lease."""


class OriginalTrainerAuthority(ABC):
    """Opaque source adapter implemented by the original execution owner.

    There is deliberately no from_dict, callback factory or authorized flag.
    A concrete adapter must bind its original process lease, registered source,
    runtime and native producer to these operations. A JSON reader cannot mint
    this owner or turn its execution_authorized=False report into permission.
    """
    @abstractmethod
    def begin_window(self, owner, intake, spec, admission):
        """Return an opaque live lease, rejecting stale/foreign producers."""

    @abstractmethod
    def consume_sample(self, lease, sample):
        """Verify the original sealed input/head/NNUEC submission provenance."""

    @abstractmethod
    def completed_update(self, lease, event, owner):
        """Publish original learner completion and return its retained pin."""

    @abstractmethod
    def checkpoint_published(self, lease, checkpoint, owner):
        """Join the full tensor byte pin to the original update journal."""

    @abstractmethod
    def end_window(self, lease, *, status, owner):
        """Close admission and drain the original owner; never refund unknown work."""

    @abstractmethod
    def validate_learned_export(self, owner, intake, checkpoint):
        """Recover original source/update/head lineage, not serialized counters alone."""

    @abstractmethod
    def validate_checkpoint_state(self, owner, intake, checkpoint):
        """Recover original nonzero resume and post-snapshot cost receipts."""


@dataclass(frozen=True)
class LearnedExportRequest:
    """Live dependencies only; never a JSON execution capability."""
    native_intake: object
    window_spec: dict
    execution_authority: OriginalTrainerAuthority


@dataclass(frozen=True)
class StagedLearnerResume:
    """Checked scratch ownership, published only after a final state check."""
    owner: OwnedTensorState
    sampler_validator: object
    original_admission: dict
    state_sha256: str

    def publish(self, slot, *, execute=False):
        runtime = _runtime(execute)
        current = tensor_payload(self.owner, sampler_validator=self.sampler_validator, execute=True)
        if runtime[5](current) != self.state_sha256:
            raise ValueError("staged resumed learner changed after original source admission")
        return slot.publish(self.owner, sampler_validator=self.sampler_validator, execute=True)


def validate_window_spec(recipe, value):
    fields(value, ("schema", "window_id", "recipe_sha256", "stage_id", "source_sha256", "runtime_sha256",
        "model_semantics_sha256", "nnuec_collection_model_sha256", "nnuec_semantics",
        "model_epoch", "context_epoch", "phase", "frozen_pc_sha256", "device", "precision", "tf32",
        "deterministic", "cpu_threads", "interop_threads", "microbatch", "accumulation", "optimizer",
        "sampling_seed", "training_seed", "loss_weights", "nnuec_loss", "max_updates", "max_groups", "max_samples",
        "max_wall_ns", "max_output_bytes", "checkpoint_interval"), "Rovent training window")
    if type(recipe) is not SeriesRecipe or value["schema"] != WINDOW_DOMAIN:
        raise ValueError("explicit Rovent training window domain required")
    identifier(value["window_id"])
    recipe.stage(value["stage_id"])
    registered = recipe.to_dict()
    if (value["recipe_sha256"] != recipe.sha256 or value["source_sha256"] != registered["source_sha256"]
            or value["model_semantics_sha256"] != registered["model_semantics_sha256"]):
        raise ValueError("training window changed registered recipe/source/model semantics")
    sha(value["runtime_sha256"])
    sha(value["nnuec_collection_model_sha256"])
    identifier(value["nnuec_semantics"])
    for name in ("model_epoch", "context_epoch"):
        uint(value[name])
    if value["phase"] == "pc":
        if value["frozen_pc_sha256"] is not None:
            raise ValueError("P/C learning cannot borrow a frozen V collection identity")
    elif value["phase"] == "v":
        sha(value["frozen_pc_sha256"])
    else:
        raise ValueError("explicit P/C or newly frozen V collection phase required")
    device = value["device"]
    if type(device) is not str or not (device == "cpu" or device.startswith("cuda:") and device[5:].isascii()
            and device[5:].isdigit() and str(int(device[5:])) == device[5:] and int(device[5:]) < 16):
        raise ValueError("explicit canonical CPU/CUDA device required; no silent fallback")
    if (value["precision"] != "fp32" or value["tf32"] is not False or value["deterministic"] is not True
            or type(value["microbatch"]) is not int or value["microbatch"] != 1
            or type(value["accumulation"]) is not int or value["accumulation"] != 8
            or not _typed_equal(value["optimizer"], OPTIMIZER)):
        raise ValueError("registered FP32/deterministic microbatch1 accumulation8 AdamW recipe required")
    for name in ("cpu_threads", "interop_threads"):
        if not 1 <= uint(value[name], 64):
            raise ValueError("explicit bounded host thread counts required")
    uint(value["sampling_seed"], 2**32 - 1)
    uint(value["training_seed"], 2**32 - 1)
    fields(value["loss_weights"], HEADS[:-1], "explicit P/C/V loss weights")
    for weight in value["loss_weights"].values():
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError("finite nonnegative explicitly registered loss weights required")
    nnuec = fields(value["nnuec_loss"], ("loss", "target", "normalization", "weight"), "explicit NNUEC learning recipe")
    if (nnuec["loss"] != "wdl_cross_entropy" or nnuec["target"] != "same_exact_rules_anchor"
            or nnuec["normalization"] != "valid_rows" or type(nnuec["weight"]) not in (int, float)
            or not math.isfinite(nnuec["weight"]) or nnuec["weight"] <= 0):
        raise ValueError("NNUEC loss/Rules target/normalization/positive weight must be explicitly resolved")
    for name, maximum in (("max_updates", 65536), ("max_groups", 65536), ("max_samples", 524288),
                          ("max_wall_ns", 24 * 3600 * 10**9), ("max_output_bytes", MAX_BYTES),
                          ("checkpoint_interval", 250)):
        if not 1 <= uint(value[name], maximum):
            raise ValueError("finite positive window work/time/output/checkpoint bound required")
    if value["max_samples"] < 8 or value["max_groups"] < value["max_updates"]:
        raise ValueError("window cannot fit its explicit group/sample budget")
    return value


def training_profile_digest(spec):
    """Window renewal may change limits; model/training semantics stay fixed."""
    renewal = {"window_id", "max_updates", "max_groups", "max_samples", "max_wall_ns", "max_output_bytes", "checkpoint_interval",
               "phase", "model_epoch", "frozen_pc_sha256", "nnuec_collection_model_sha256"}
    return digest("rz-rovent-fixed-training-profile/1", {name: value for name, value in spec.items() if name not in renewal})


class StageSampler:
    """Native-row tokens plus exact MT19937 cursor; never an input cache."""
    def __init__(self, rows, *, role, seed, pool_sha256, state=None):
        if role not in ROLES or not 1 <= len(rows) <= 262144:
            raise ValueError("bounded nonempty native role pool required")
        self.rows, self.role, self.seed = rows, role, uint(seed, 2**32 - 1)
        self.pool_sha256 = sha(pool_sha256)
        self.epoch, self.cursor, self.samples = 0, 0, 0
        if state is not None:
            fields(state, ("schema", "role", "seed", "pool_sha256", "epoch", "cursor", "samples", "permutation"), "native sampler state")
            if (state["schema"] != SAMPLER_DOMAIN or state["role"] != role or state["seed"] != seed
                    or state["pool_sha256"] != pool_sha256):
                raise ValueError("native sampler changed role/seed/pinned pool")
            self.epoch, self.cursor, self.samples = uint(state["epoch"]), uint(state["cursor"], len(rows)), uint(state["samples"])
            if self.samples != self.epoch * len(rows) + self.cursor:
                raise ValueError("native sampler lost its exact consumed position")
        self.permutation = self._order()
        if state is not None and not _typed_equal(state["permutation"], self.permutation):
            raise ValueError("native sampler seeded permutation changed")

    def _order(self):
        result = list(range(len(self.rows)))
        random.Random(digest(SAMPLER_DOMAIN, [self.seed, self.role, self.pool_sha256, self.epoch])).shuffle(result)
        return result

    def take(self, count):
        result = []
        for _ in range(uint(count, 8)):
            if self.cursor == len(self.permutation):
                self.epoch, self.cursor = self.epoch + 1, 0
                self.permutation = self._order()
            result.append(self.rows[self.permutation[self.cursor]])
            self.cursor += 1
            self.samples += 1
        return result

    def state(self):
        return {"schema": SAMPLER_DOMAIN, "role": self.role, "seed": self.seed,
            "pool_sha256": self.pool_sha256, "epoch": self.epoch, "cursor": self.cursor,
            "samples": self.samples, "permutation": list(self.permutation)}


def _read_admission(owner, intake, spec):
    from .rovent_corpus import RoventTrainingCorpus
    if type(intake) is not RoventTrainingCorpus:
        raise ValueError("original teacher-aware native corpus owner required; legacy/metadata receipts are not an intake")
    admission = intake.admit_stage(owner.recipe, owner.stage_id, spec["model_epoch"], spec["context_epoch"],
        list(owner.inherited_input_sha256), list(owner.inherited_origin_sha256))
    for name, expected in (("recipe_sha256", owner.recipe.sha256), ("stage_id", owner.stage_id),
            ("source_sha256", spec["source_sha256"]), ("runtime_sha256", spec["runtime_sha256"]),
            ("model_semantics_sha256", spec["model_semantics_sha256"]),
            ("model_epoch", spec["model_epoch"]), ("context_epoch", spec["context_epoch"])):
        if not _typed_equal(admission.get(name), expected):
            raise ValueError("native admission differs from registered window: " + name)
    sha(admission["corpus_sha256"])
    if admission.get("execution_authorized") is not False:
        raise ValueError("corpus report cannot issue original learner execution authority")
    return admission


def _journal(owner, spec, admission):
    value = owner.samplers
    fields(value, ("schema", "recipe_sha256", "stage_id", "profile_sha256", "phase_inputs",
        "updates", "head_counts", "frozen_pc_sha256", "resume_cost_receipts"), "original learner journal")
    fixed = {"schema": JOURNAL_DOMAIN, "recipe_sha256": owner.recipe.sha256, "stage_id": owner.stage_id,
        "profile_sha256": training_profile_digest(spec)}
    if any(not _typed_equal(value.get(name), item) for name, item in fixed.items()):
        raise ValueError("learner journal changed its pinned corpus/profile/epoch")
    fields(value["head_counts"], HEADS, "cumulative valid head/pair counts")
    for count in value["head_counts"].values():
        uint(count)
    if type(value["updates"]) is not list or len(value["updates"]) != sum(owner.completed.values()):
        raise ValueError("learner update journal differs from completed optimizer boundaries")
    phases = value["phase_inputs"]
    if type(phases) is not dict or not phases or set(phases) - {"pc", "v"} or spec["phase"] not in phases:
        raise ValueError("fresh phase source must be explicitly prepared before dispatch")
    for source in phases.values():
        fields(source, ("corpus_sha256", "nnuec_model_sha256", "model_epoch", "context_epoch", "samplers"), "retained phase collection")
        sha(source["corpus_sha256"])
        sha(source["nnuec_model_sha256"])
        uint(source["model_epoch"])
        uint(source["context_epoch"])
        if type(source["samplers"]) is not dict:
            raise ValueError("phase sampler owner map required")
    source = phases[spec["phase"]]
    if any(not _typed_equal(source[name], expected) for name, expected in
           (("corpus_sha256", admission["corpus_sha256"]), ("nnuec_model_sha256", spec["nnuec_collection_model_sha256"]),
            ("model_epoch", spec["model_epoch"]), ("context_epoch", spec["context_epoch"]))):
        raise ValueError("phase collection changed its original model/context/corpus identity")
    counted, heads = dict.fromkeys(ROLES, 0), dict.fromkeys(HEADS, 0)
    retained_inputs, retained_origins = set(owner.inherited_input_sha256), set(owner.inherited_origin_sha256)
    for event in value["updates"]:
        fields(event, ("schema", "window_id", "stage_id", "role", "completed_role_updates", "profile_sha256",
            "corpus_sha256", "phase", "sample_origins", "sample_inputs", "head_counts",
            "actual_model_state_sha256", "original_completion"), "original learner update event")
        role = event["role"]
        if role not in ROLES:
            raise ValueError("unknown original learner update role")
        counted[role] += 1
        phase = "v" if role == "validator" else "pc"
        if (event["schema"] != "rz-rovent-original-learner-update/1" or event["stage_id"] != owner.stage_id
                or event["completed_role_updates"] != counted[role] or event["phase"] != phase
                or event["profile_sha256"] != fixed["profile_sha256"]
                or event["corpus_sha256"] != phases[phase]["corpus_sha256"]):
            raise ValueError("learner update journal changed original order/source/phase")
        identifier(event["window_id"])
        sha(event["actual_model_state_sha256"])
        for name, allowed in (("sample_origins", retained_origins), ("sample_inputs", retained_inputs)):
            values = event[name]
            if type(values) is not list or len(values) != 8 or any(sha(item) not in allowed for item in values):
                raise ValueError("learner journal erased original consumed input/origin seals")
        fields(event["head_counts"], HEADS, "actual update head counts")
        for head, count in event["head_counts"].items():
            heads[head] += uint(count, 4096)
        receipt = fields(event["original_completion"], ("sha256", "bytes"), "original learner completion pin")
        sha(receipt["sha256"])
        if not uint(receipt["bytes"]):
            raise ValueError("empty original learner completion receipt")
    if counted != owner.completed or heads != value["head_counts"]:
        raise ValueError("original learner journal update/head totals disagree")
    if value["frozen_pc_sha256"] is not None:
        sha(value["frozen_pc_sha256"])
    if spec["phase"] == "v" and value["frozen_pc_sha256"] != spec["frozen_pc_sha256"]:
        raise ValueError("V phase does not retain the original frozen P/C checkpoint identity")
    recovered = value["resume_cost_receipts"]
    if type(recovered) is not list or len(recovered) > 4096:
        raise ValueError("bounded original checkpoint cost recovery chain required")
    for item in recovered:
        fields(item, ("checkpoint_sha256", "original_receipt_sha256", "cost_delta"), "original resume cost receipt")
        sha(item["checkpoint_sha256"])
        sha(item["original_receipt_sha256"])
        fields(item["cost_delta"], COST_FIELDS, "original post-snapshot cost delta")
        for count in item["cost_delta"].values():
            uint(count)
    return value


def initial_training_journal(recipe, spec, admission):
    """Plain initial state only; this function issues no native authority."""
    validate_window_spec(recipe, spec)
    sha(admission["corpus_sha256"])
    return {"schema": JOURNAL_DOMAIN, "recipe_sha256": recipe.sha256, "stage_id": spec["stage_id"],
        "profile_sha256": training_profile_digest(spec), "phase_inputs": {spec["phase"]: {
            "corpus_sha256": admission["corpus_sha256"], "model_epoch": spec["model_epoch"],
            "nnuec_model_sha256": spec["nnuec_collection_model_sha256"], "context_epoch": spec["context_epoch"], "samplers": {}}},
        "updates": [], "head_counts": dict.fromkeys(HEADS, 0), "frozen_pc_sha256": spec["frozen_pc_sha256"],
        "resume_cost_receipts": []}


def prepare_frozen_v_journal(owner, native_intake, window_spec):
    """Retain P/C history and bind a separately collected frozen V corpus.

    This is plain state preparation, not an execution lease or a new freeze.
    The original host authority still verifies the actual immutable P/C state.
    """
    spec = validate_window_spec(owner.recipe, window_spec)
    stage = owner.recipe.stage(owner.stage_id)
    if (spec["phase"] != "v" or not stage.role_updates[2] or owner.completed["validator"]
            or any(owner.completed[role] != limit for role, limit in zip(ROLES[:2], stage.role_updates[:2]))):
        raise ValueError("V collection requires complete P/C and zero current-stage V updates")
    journal = copy.deepcopy(owner.samplers)
    if journal["profile_sha256"] != training_profile_digest(spec) or journal["frozen_pc_sha256"] != spec["frozen_pc_sha256"]:
        raise ValueError("new V source cannot change the frozen P/C model/training profile")
    admission = _read_admission(owner, native_intake, spec)
    if "v" in journal["phase_inputs"] or spec["model_epoch"] <= journal["phase_inputs"]["pc"]["model_epoch"]:
        raise ValueError("V collection must be new and use a later frozen P/C epoch")
    journal["phase_inputs"]["v"] = {"corpus_sha256": admission["corpus_sha256"],
        "nnuec_model_sha256": spec["nnuec_collection_model_sha256"], "model_epoch": spec["model_epoch"],
        "context_epoch": spec["context_epoch"], "samplers": {}}
    return journal


def _valid_counts(samples, role):
    counts = dict.fromkeys(HEADS, 0)
    for sample in samples:
        targets = fields(sample["targets"], HEADS[:-1], "native normalized target heads")
        if role == "validator":
            if targets["task"] is not None:
                counts["task"] += len(targets["task"]["pairs"])
        else:
            counts["policy"] += targets["policy"] is not None
            counts["wdl"] += targets["wdl"] is not None
            counts["divergence"] += role == "critic" and targets["divergence"] is not None
            counts["nnuec_wdl"] += sample["nnuec_wdl"] is not None
    return counts


def _frozen_pc(owner, state_digest):
    return state_digest({name: value.detach().cpu().clone() for name, value in owner.model.named_parameters()
                         if _owner_component(name) != "v_private"})


def _runtime_profile(owner, spec, torch):
    if (torch.get_num_threads() != spec["cpu_threads"] or torch.get_num_interop_threads() != spec["interop_threads"]
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or not torch.are_deterministic_algorithms_enabled()):
        raise ValueError("actual host/TF32/deterministic runtime differs from registered window")
    expected = torch.device(spec["device"])
    if any(parameter.device != expected for parameter in owner.model.parameters()):
        raise ValueError("actual learner device differs; CPU/CUDA fallback is forbidden")
    if expected.type == "cuda" and tuple(owner.cuda_devices) != (expected.index,):
        raise ValueError("actual CUDA RNG owner differs from registered learner device")
    if expected.type == "cpu" and owner.cuda_devices:
        raise ValueError("CPU window cannot retain undeclared CUDA RNG ownership")
    if owner.nnuec_metadata["model_semantics"] != spec["nnuec_semantics"]:
        raise ValueError("actual NNUEC semantics differ from the registered collection namespace")


def _stage_rovent_checkpoint(path, expected_sha256, template, recipe, membership, native_intake, window_spec, *,
                         execution_authority=None, max_file_bytes, purpose, execute=False):
    """Preflight full checkpoint+journal into a new owner; never mutate caller.

    The original process authority recovers cost spent after the last tensor
    snapshot (its own write/drain) from retained receipts. A resumed owner must
    register a fresh full checkpoint before learned export; its old byte pin
    continues to identify the unmodified source, not the recovered state.
    """
    spec = copy.deepcopy(validate_window_spec(recipe, window_spec))
    if purpose not in ("resume", "export"):
        raise ValueError("explicit raw-export or training-resume checkpoint purpose required")
    sha(expected_sha256)
    maximum = uint(max_file_bytes, MAX_BYTES)
    if not maximum:
        raise ValueError("positive original checkpoint input byte credit required")
    if execute is False:
        return {"schema": "rz-rovent-learner-" + purpose + "-preparation/1", "recipe_sha256": recipe.sha256,
            "stage_id": spec["stage_id"], "checkpoint_sha256": expected_sha256,
            "actual_tensor_bytes_loaded": False, "caller_state_mutated": False, "execution_authorized": False}
    if execute is not True or not isinstance(execution_authority, OriginalTrainerAuthority):
        raise OriginalTrainingAuthorityRequired("original host authority required for learned checkpoint resume")
    runtime = _runtime(True)
    payload, size = _load_pinned_tensor_file(path, expected_sha256, maximum, runtime[0])
    if type(payload) is not dict or not {"samplers", "ledger", "inherited_input_sha256", "inherited_origin_sha256"} <= payload.keys():
        raise ValueError("complete original tensor journal/ledger required before resume")
    _validate_json_state(payload["samplers"])
    probe = SimpleNamespace(recipe=recipe, stage_id=spec["stage_id"], samplers=payload["samplers"],
        completed=payload["ledger"]["completed"], inherited_input_sha256=payload["inherited_input_sha256"],
        inherited_origin_sha256=payload["inherited_origin_sha256"])
    admission = _read_admission(probe, native_intake, spec)
    def validate_sampler(value):
        candidate = copy.copy(probe)
        candidate.samplers = value
        journal = _journal(candidate, spec, admission)
        for role, state in journal["phase_inputs"][spec["phase"]]["samplers"].items():
            rows = native_intake.rows(role, split="train", optimization=True)
            pool = digest("rz-rovent-native-role-pool/1", {"corpus_sha256": admission["corpus_sha256"], "role": role,
                "inputs": [(sample["input_sha256"], sample["origin_sha256"]) for sample in
                           (native_intake.training_sample(row) for row in rows)]})
            StageSampler(rows, role=role, seed=spec["sampling_seed"], pool_sha256=pool, state=state)
    with _RNG_LOCK:
        original_rng = runtime[3](tuple(payload.get("cuda_devices", ())))
        try:
            staged = _restore_scratch(payload, template, recipe, membership, sampler_validator=validate_sampler, runtime=runtime)
            if staged.stage_id != spec["stage_id"]:
                raise ValueError("learned resume cannot change stage or teacher branch")
            pin = {"sha256": expected_sha256, "bytes": size}
            original = execution_authority.validate_checkpoint_state(staged, native_intake, pin)
            fields(original, ("schema", "recipe_sha256", "stage_id", "checkpoint_sha256", "training_profile_sha256",
                "original_receipt_sha256", "original_update_chain_sha256", "completed_updates", "retained_input_set_sha256",
                "retained_input_count", "retained_origin_set_sha256", "retained_origin_count",
                "post_snapshot_cost", "original_native_lineage_verified"), "original resume admission")
            input_set = retained_seal_set(staged.inherited_input_sha256, "input")
            origin_set = retained_seal_set(staged.inherited_origin_sha256, "origin")
            fixed = {"schema": "rz-rovent-original-learner-resume-admission/1", "recipe_sha256": recipe.sha256,
                "stage_id": staged.stage_id, "checkpoint_sha256": expected_sha256, "training_profile_sha256": training_profile_digest(spec),
                "completed_updates": dict(staged.completed), "retained_input_set_sha256": input_set["sha256"],
                "retained_input_count": input_set["count"], "retained_origin_set_sha256": origin_set["sha256"],
                "retained_origin_count": origin_set["count"], "original_native_lineage_verified": True}
            if any(not _typed_equal(original[name], item) for name, item in fixed.items()):
                raise ValueError("original checkpoint recovery changed model/update/input ancestry")
            sha(original["original_receipt_sha256"])
            sha(original["original_update_chain_sha256"])
            fields(original["post_snapshot_cost"], COST_FIELDS, "original post-snapshot cost recovery")
            for count in original["post_snapshot_cost"].values():
                uint(count)
            if purpose == "resume":
                for name, count in original["post_snapshot_cost"].items():
                    staged.cost[name] = uint(staged.cost[name] + count)
                staged.samplers["resume_cost_receipts"].append({"checkpoint_sha256": expected_sha256,
                    "original_receipt_sha256": original["original_receipt_sha256"], "cost_delta": copy.deepcopy(original["post_snapshot_cost"])})
            staged.source_checkpoint_sha256, staged.source_checkpoint_path = expected_sha256, str(path)
            staged.source_model_state_sha256 = runtime[5](payload["model"])
            staged.source_nnuec_metadata_sha256 = digest("rz-rovent-original-nnuec-metadata/1", payload["nnuec_metadata"])
            staged.source_payload_state_sha256 = runtime[5](payload)
            probe.samplers = staged.samplers
            current = tensor_payload(staged, sampler_validator=validate_sampler, execute=True)
            return StagedLearnerResume(staged, validate_sampler, copy.deepcopy(original), runtime[5](current))
        finally:
            runtime[4](original_rng, tuple(payload.get("cuda_devices", ())))


def stage_rovent_resume(path, expected_sha256, template, recipe, membership, native_intake, window_spec, *,
                        execution_authority=None, max_file_bytes, execute=False):
    """Resume training state, retaining original post-snapshot cost receipts."""
    return _stage_rovent_checkpoint(path, expected_sha256, template, recipe, membership, native_intake, window_spec,
        execution_authority=execution_authority, max_file_bytes=max_file_bytes, purpose="resume", execute=execute)


def stage_rovent_export(path, expected_sha256, template, recipe, membership, native_intake, window_spec, *,
                        execution_authority=None, max_file_bytes, execute=False):
    """Own exact raw export state; retain post-snapshot expense beside it.

    This purpose does not append a sampler receipt or modify ledger/RNG/model
    state. Original training drain expense stays in original_admission, ready
    for the export receipt; it cannot be silently discarded or counted as a
    fresh optimizer update. Training must use stage_rovent_resume instead.
    """
    return _stage_rovent_checkpoint(path, expected_sha256, template, recipe, membership, native_intake, window_spec,
        execution_authority=execution_authority, max_file_bytes=max_file_bytes, purpose="export", execute=execute)


def _source_fork_journal(value, *, recipe, source_stage, completed):
    """Shape checks only; exact source bytes/state require the original owner.

    The target window cannot reconstruct the source loss recipe or corpus
    epochs. Those remain opaque in the retained source profile and are joined
    to the previous original window by validate_checkpoint_state.
    """
    fields(value, ("schema", "recipe_sha256", "stage_id", "profile_sha256", "phase_inputs",
        "updates", "head_counts", "frozen_pc_sha256", "resume_cost_receipts"), "source fork journal")
    if (value["schema"] != JOURNAL_DOMAIN or value["recipe_sha256"] != recipe.sha256
            or value["stage_id"] != source_stage or type(value["updates"]) is not list
            or len(value["updates"]) != sum(completed.values())):
        raise ValueError("fork source journal changed its original stage/update boundaries")
    sha(value["profile_sha256"])
    _validate_json_state(value)


def stage_rovent_fork(path, expected_sha256, template, recipe, membership, native_intake, window_spec, *,
                      execution_authority=None, max_file_bytes, execute=False):
    """Verify the original completed parent, then fork weights with reset state.

    Native source custody approves the parent checkpoint before any reset. Its
    post-snapshot expense is retained as ancestor cost. A new child journal
    binds only the separately collected target corpus; no old targets/optimizer
    moments are relabelled as a teacher-specific child.
    """
    from .rovent_tensor_training import stage_tensor_fork
    spec = copy.deepcopy(validate_window_spec(recipe, window_spec))
    target = recipe.stage(spec["stage_id"])
    if target.parent_stage_id is None:
        raise ValueError("common initialization cannot be represented as a fork")
    sha(expected_sha256)
    maximum = uint(max_file_bytes, MAX_BYTES)
    if not maximum:
        raise ValueError("positive original parent checkpoint byte credit required")
    if execute is False:
        return {"schema": "rz-rovent-learner-fork-preparation/1", "recipe_sha256": recipe.sha256,
            "source_stage_id": target.parent_stage_id, "stage_id": target.stage_id,
            "checkpoint_sha256": expected_sha256, "actual_tensor_bytes_loaded": False,
            "reset_optimizer_state": True, "execution_authorized": False}
    if execute is not True or not isinstance(execution_authority, OriginalTrainerAuthority):
        raise OriginalTrainingAuthorityRequired("original parent checkpoint authority required before a fork")
    runtime = _runtime(True)
    payload, size = _load_pinned_tensor_file(path, expected_sha256, maximum, runtime[0])
    if type(payload) is not dict or not {"samplers", "ledger", "stage_id"} <= payload.keys():
        raise ValueError("complete original source tensor journal required before fork")
    if payload["stage_id"] != target.parent_stage_id:
        raise ValueError("fork source is not the declared parent stage")
    source_counts = dict(zip(ROLES, recipe.stage(target.parent_stage_id).role_updates))
    if payload["ledger"]["completed"] != source_counts:
        raise ValueError("fork requires all declared source-stage optimizer updates")
    def validate_source_sampler(value):
        _source_fork_journal(value, recipe=recipe, source_stage=target.parent_stage_id, completed=source_counts)
    parent = _restore_scratch(payload, template, recipe, membership,
        sampler_validator=validate_source_sampler, runtime=runtime)
    pin = {"sha256": expected_sha256, "bytes": size}
    original = execution_authority.validate_checkpoint_state(parent, native_intake, pin)
    fields(original, ("schema", "recipe_sha256", "stage_id", "checkpoint_sha256", "training_profile_sha256",
        "original_receipt_sha256", "original_update_chain_sha256", "completed_updates", "retained_input_set_sha256",
        "retained_input_count", "retained_origin_set_sha256", "retained_origin_count", "post_snapshot_cost",
        "original_native_lineage_verified"), "original parent fork admission")
    input_set, origin_set = (retained_seal_set(parent.inherited_input_sha256, "input"),
                             retained_seal_set(parent.inherited_origin_sha256, "origin"))
    fixed = {"schema": "rz-rovent-original-learner-resume-admission/1", "recipe_sha256": recipe.sha256,
        "stage_id": parent.stage_id, "checkpoint_sha256": expected_sha256,
        "training_profile_sha256": parent.samplers["profile_sha256"], "completed_updates": source_counts,
        "retained_input_set_sha256": input_set["sha256"], "retained_input_count": input_set["count"],
        "retained_origin_set_sha256": origin_set["sha256"], "retained_origin_count": origin_set["count"],
        "original_native_lineage_verified": True}
    if any(not _typed_equal(original[name], expected) for name, expected in fixed.items()):
        raise ValueError("original fork proof changed source profile/update/input ancestry")
    sha(original["original_receipt_sha256"])
    sha(original["original_update_chain_sha256"])
    fields(original["post_snapshot_cost"], COST_FIELDS, "source fork post-snapshot cost")
    for value in original["post_snapshot_cost"].values():
        uint(value)
    probe = SimpleNamespace(recipe=recipe, stage_id=target.stage_id,
        inherited_input_sha256=list(parent.inherited_input_sha256),
        inherited_origin_sha256=list(parent.inherited_origin_sha256), completed=dict.fromkeys(ROLES, 0))
    admission = _read_admission(probe, native_intake, spec)
    journal = initial_training_journal(recipe, spec, admission)
    probe.samplers = journal
    def validate_target_sampler(value):
        candidate = copy.copy(probe)
        candidate.samplers = value
        _journal(candidate, spec, admission)
    child = stage_tensor_fork(path, expected_sha256, template, recipe, membership,
        source_sampler_validator=validate_source_sampler, target_sampler_validator=validate_target_sampler,
        target_stage_id=target.stage_id, reset_seed=spec["training_seed"], reset_samplers=journal,
        reset_bucket=8, max_file_bytes=maximum, execute=True)
    for name, value in original["post_snapshot_cost"].items():
        child.ancestor_cost[name] = uint(child.ancestor_cost[name] + value)
    current = tensor_payload(child, sampler_validator=validate_target_sampler, execute=True)
    return StagedLearnerResume(child, validate_target_sampler, copy.deepcopy(original), runtime[5](current))


def admit_learned_export(owner, checkpoint, request):
    """Recover raw/native lineage through the original host before export.

    The checkpoint still proves tensor lifecycle by itself. This additional
    admission is obtained from the live original authority, never loaded as a
    serialized eligibility flag or used to waive numerical/GPU acceptance.
    """
    if type(request) is not LearnedExportRequest or not isinstance(request.execution_authority, OriginalTrainerAuthority):
        raise OriginalTrainingAuthorityRequired("live original trainer authority required for learned export")
    spec = validate_window_spec(owner.recipe, request.window_spec)
    admission = _read_admission(owner, request.native_intake, spec)
    journal = _journal(owner, spec, admission)
    expected = dict(zip(ROLES, owner.recipe.stage(owner.stage_id).role_updates))
    if owner.completed != expected or owner.pending_microbatches:
        raise ValueError("learned export requires every declared current-stage optimizer boundary")
    totals = {role: uint(owner.completed[role] + owner.ancestor_updates[role]) for role in ROLES}
    if not any(totals.values()):
        raise ValueError("no actual learned ancestry to admit")
    source = request.execution_authority.validate_learned_export(owner, request.native_intake, checkpoint)
    fields(source, ("schema", "recipe_sha256", "stage_id", "checkpoint_sha256", "source_sha256", "runtime_sha256",
        "model_semantics_sha256", "training_profile_sha256", "original_update_chain_sha256",
        "retained_input_set_sha256", "retained_input_count", "retained_origin_set_sha256", "retained_origin_count",
        "lineage_teacher_actor_ids", "completed_updates",
        "pc_training_steps", "nnuec_training", "post_snapshot_cost", "original_native_lineage_verified"), "original learned export admission")
    input_set = retained_seal_set(owner.inherited_input_sha256, "input")
    origin_set = retained_seal_set(owner.inherited_origin_sha256, "origin")
    fixed = {"schema": "rz-rovent-original-learned-export-admission/1", "recipe_sha256": owner.recipe.sha256,
        "stage_id": owner.stage_id, "checkpoint_sha256": checkpoint["sha256"], "source_sha256": spec["source_sha256"],
        "runtime_sha256": spec["runtime_sha256"], "model_semantics_sha256": spec["model_semantics_sha256"],
        "training_profile_sha256": training_profile_digest(spec), "completed_updates": totals,
        "lineage_teacher_actor_ids": list(owner.recipe.inherited_teachers(owner.stage_id)),
        "retained_input_set_sha256": input_set["sha256"], "retained_input_count": input_set["count"],
        "retained_origin_set_sha256": origin_set["sha256"], "retained_origin_count": origin_set["count"],
        "pc_training_steps": totals["proposer"] + totals["critic"], "original_native_lineage_verified": True}
    if any(not _typed_equal(source[name], value) for name, value in fixed.items()):
        raise ValueError("original learned admission changed actual source/ancestry/update/input lineage")
    sha(source["original_update_chain_sha256"])
    fields(source["post_snapshot_cost"], COST_FIELDS, "original training post-snapshot export cost")
    for count in source["post_snapshot_cost"].values():
        uint(count)
    nnuec = source["nnuec_training"]
    nnuec_state = _nnuec_tensor_training_state(owner)
    if type(nnuec) is not dict:
        raise ValueError("explicit original NNUEC training provenance required")
    if nnuec.get("kind") == "untrained":
        fields(nnuec, ("kind",), "untrained NNUEC declaration")
        if (not nnuec_is_unmodified(nnuec_state) or nnuec_state["source_declaration"] != {"kind": "untrained"}
                or journal["head_counts"]["nnuec_wdl"]
                or any(count for name, count in owner.parameter_updates.items() if name.startswith("nnuec."))):
            raise ValueError("updated NNUEC parameters cannot be declared untrained")
    else:
        fields(nnuec, ("kind", "run_id", "steps", "dataset_sha256"), "original NNUEC learned declaration")
        known_updates = nnuec_known_updates(nnuec_state)
        if (nnuec["kind"] != "learned" or not 1 <= uint(nnuec["steps"], fixed["pc_training_steps"])
                or known_updates is None or nnuec["steps"] != known_updates):
            raise ValueError("NNUEC original update count differs from P/C learning ancestry")
        identifier(nnuec["run_id"])
        sha(nnuec["dataset_sha256"])
    return copy.deepcopy(source)


def train_rovent_window(owner, native_intake, window_spec, checkpoint_directory, *,
                         execution_authority=None, cancelled=lambda: False,
                         execution_until_ns=None, execute=False):
    """Validate only by default; explicit execution is one finite owned window.

    The caller must install its initial/fork sampler journal before dispatch;
    resume must retain it exactly. No fixed first-generation spec is submitted
    to the legacy trainer, no all-masked group earns an update, and any partial
    optimizer/physical failure quarantines the owner instead of retrying it.
    """
    call_started = time.monotonic_ns()
    if type(owner) is not OwnedTensorState or owner.quarantined:
        raise ValueError("original nonquarantined Rovent tensor owner required")
    spec = copy.deepcopy(validate_window_spec(owner.recipe, window_spec))
    if owner.stage_id != spec["stage_id"]:
        raise ValueError("window cannot change an owned checkpoint stage")
    admission = _read_admission(owner, native_intake, spec)
    journal = _journal(owner, spec, admission)
    if execute is not True:
        if execute is not False:
            raise ValueError("explicit bool execution selector required")
        return {"schema": WINDOW_DOMAIN, "recipe_sha256": owner.recipe.sha256, "stage_id": owner.stage_id,
            "admission": admission, "training_profile_sha256": training_profile_digest(spec),
            "actual_training_executed": False, "optimizer_updates": 0, "execution_authorized": False,
            "arena_eligible": False}
    if not isinstance(execution_authority, OriginalTrainerAuthority):
        raise OriginalTrainingAuthorityRequired("original host/producer training authority required; JSON admission is insufficient")
    if not callable(cancelled):
        raise ValueError("original cancellation observation callable required")
    if execution_until_ns is not None:
        uint(execution_until_ns)
        if time.monotonic_ns() >= execution_until_ns:
            raise TimeoutError("original execution deadline expired before tensor dispatch")
    runtime = _runtime(True)
    torch, _, _, capture_rng, restore_rng, state_digest, _ = runtime
    _runtime_profile(owner, spec, torch)
    directory = Path(checkpoint_directory)
    if not directory.is_absolute():
        raise ValueError("explicit outside-source checkpoint directory required")
    stage = owner.recipe.stage(owner.stage_id)
    available = dict(zip(ROLES, stage.role_updates))
    roles = tuple(role for role in (ROLES[:2] if spec["phase"] == "pc" else ("validator",)) if available[role])
    if not roles:
        raise ValueError("registered stage has no updates in this collection phase")
    if spec["phase"] == "v" and any(owner.completed[role] != available[role] for role in ROLES[:2]):
        raise ValueError("V window cannot start before frozen P/C completion")
    phase_state = journal["phase_inputs"][spec["phase"]]
    streams, samplers = {}, {}
    for role in roles:
        streams[role] = native_intake.rows(role, split="train", optimization=True)
        pool = digest("rz-rovent-native-role-pool/1", {"corpus_sha256": admission["corpus_sha256"], "role": role,
            "inputs": [(sample["input_sha256"], sample["origin_sha256"]) for sample in
                       (native_intake.training_sample(row) for row in streams[role])]})
        samplers[role] = StageSampler(streams[role], role=role, seed=spec["sampling_seed"], pool_sha256=pool,
            state=phase_state["samplers"].get(role))
    if set(phase_state["samplers"]) - set(roles):
        raise ValueError("learner journal contains a foreign role sampler")
    def validate_sampler(value):
        prior = owner.samplers
        try:
            owner.samplers = value
            checked = _journal(owner, spec, admission)
            for role, state in checked["phase_inputs"][spec["phase"]]["samplers"].items():
                StageSampler(streams[role], role=role, seed=spec["sampling_seed"],
                    pool_sha256=samplers[role].pool_sha256, state=state)
        finally:
            owner.samplers = prior
    with _RNG_LOCK:
        tensor_payload(owner, sampler_validator=validate_sampler, execute=True)
        original_rng = capture_rng(owner.cuda_devices)
        lease, status, checkpoints = None, "failed", []
        started, groups, consumed_samples = call_started, 0, 0
        wall_charged_through = started
        def charge_wall():
            nonlocal wall_charged_through
            now = time.monotonic_ns()
            if now < wall_charged_through:
                raise ValueError("original learner monotonic clock moved backwards")
            owner.cost["wall_time_ns"] = uint(owner.cost["wall_time_ns"] + now - wall_charged_through)
            wall_charged_through = now
        start_updates = sum(owner.completed.values())
        start_output = owner.cost["output_bytes"]
        owner.physical_completion = "unknown"
        def check_admission():
            if cancelled():
                raise InterruptedError("original learner cancellation closes further admission")
            if execution_until_ns is not None and time.monotonic_ns() >= execution_until_ns:
                raise TimeoutError("original learner deadline closes further admission")
        try:
            check_admission()
            lease = execution_authority.begin_window(owner, native_intake, spec, admission)
            if lease is None or type(lease) in (dict, list, tuple, str, int, float, bool):
                raise OriginalTrainingAuthorityRequired("original process returned no opaque live learner lease")
            restore_rng(owner.rng, owner.cuda_devices)
            owner.model.train()
            while sum(owner.completed.values()) - start_updates < spec["max_updates"]:
                check_admission()
                if time.monotonic_ns() - started >= spec["max_wall_ns"]:
                    status = "window_expired_at_boundary"
                    break
                if all(owner.completed[role] == available[role] for role in roles):
                    if spec["phase"] == "pc" and available["validator"]:
                        journal["frozen_pc_sha256"] = _frozen_pc(owner, state_digest)
                        status = "await_new_frozen_v_corpus"
                    else:
                        status = "declared_stage_updates_complete"
                    break
                if groups >= spec["max_groups"] or consumed_samples + 8 > spec["max_samples"]:
                    status = "sample_group_budget_exhausted"
                    break
                pending = [role for role in roles if owner.completed[role] < available[role]]
                role = min(pending, key=lambda item: (owner.completed[item], ROLES.index(item)))
                if role == "validator":
                    if journal["frozen_pc_sha256"] != _frozen_pc(owner, state_digest):
                        raise ValueError("shared/P/C/NNUEC changed after the V freeze")
                owner.activate_role(role)
                tokens = samplers[role].take(8)
                samples = [native_intake.training_sample(token) for token in tokens]
                phase_state["samplers"] = {name: sampler.state() for name, sampler in samplers.items()}
                groups += 1
                consumed_samples += len(samples)
                for sample in samples:
                    execution_authority.consume_sample(lease, sample)
                counts = _valid_counts(samples, role)
                weights = {**spec["loss_weights"], "nnuec_wdl": spec["nnuec_loss"]["weight"]}
                if not any(count and weights[head] for head, count in counts.items()):
                    owner.cost["masked_groups"] += 1
                    continue
                owner.dispatched[role] += 1
                update_started = time.monotonic_ns()
                active = {name: parameter for name, parameter in owner.model.named_parameters() if parameter.requires_grad}
                optimizer_ids = sorted({owner.recipe.to_dict()["parameter_ownership"][_owner_component(name)] for name in active})
                for identity in optimizer_ids:
                    owner.optimizers[identity].zero_grad(set_to_none=True)
                try:
                    from .generation_training import compact_collate_and_forward
                    from .virtual_review import validate_payload
                    for sample in samples:
                        check_admission()
                        # Pass only the submitted immutable input to the model;
                        # targets and future witnesses never enter its encoder.
                        batch, outputs = compact_collate_and_forward(owner.model, role, copy.deepcopy(sample["input"]))
                        owner.cost["forward_calls"] += 1
                        owner.cost["samples"] += 1
                        sums = loss_sums(outputs, batch, sample["targets"], role, divergence_index=sample["divergence_index"])
                        if role != "validator" and sample["nnuec_wdl"] is not None:
                            if sample["nnuec_input"] is None:
                                raise ValueError("NNUEC target has no original same-anchor Rules input")
                            if (sample["nnuec_model_sha256"] != spec["nnuec_collection_model_sha256"]
                                    or sample["nnuec_semantics"] != spec["nnuec_semantics"]):
                                raise ValueError("NNUEC training row changed its original collection model namespace")
                            nnuec = owner.model.nnuec(copy.deepcopy(sample["nnuec_input"]))
                            owner.cost["forward_calls"] += 1
                            logits = nnuec.wdl_logits
                            target = torch.tensor(sample["nnuec_wdl"], dtype=torch.float32, device=logits.device)
                            if logits.shape != target.shape or logits.shape != (3,) or not bool(torch.all(torch.isfinite(logits))):
                                raise ValueError("NNUEC same-anchor WDL shape/dtype/finite check failed")
                            sums["nnuec_wdl"] = -(target * torch.nn.functional.log_softmax(logits, dim=-1)).sum()
                        else:
                            sums["nnuec_wdl"] = outputs[0].sum() * 0.0
                        loss = sum((sums[head] * weights[head] / counts[head] for head in HEADS if counts[head]), outputs[0].sum() * 0.0)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError("nonfinite original learner loss")
                        loss.backward()
                        owner.cost["backward_calls"] += 1
                        owner.pending_microbatches += 1
                        owner.bucket = validate_payload(sample["input"]["encoded_model_input"])["line_bucket"]
                    if any(parameter.grad is not None and not bool(torch.all(torch.isfinite(parameter.grad))) for parameter in active.values()):
                        raise ValueError("nonfinite original learner gradient")
                    if not any(parameter.grad is not None for parameter in active.values()):
                        raise ValueError("valid supervised heads produced no owned parameter gradient")
                    torch.nn.utils.clip_grad_norm_(list(active.values()), OPTIMIZER["clip_norm"], error_if_nonfinite=True)
                    check_admission()
                    # Multiple declared owners are explicit (e.g. NNUEC's own
                    # moments). Failure after one step is uncertain and is
                    # quarantined; it cannot be atomically refunded/retried.
                    for identity in optimizer_ids:
                        owner.optimizers[identity].step()
                        owner.schedulers[identity].step()
                    for name, parameter in active.items():
                        if parameter.grad is not None:
                            owner.parameter_updates[name] += 1
                    for identity in optimizer_ids:
                        owner.optimizers[identity].zero_grad(set_to_none=True)
                    owner.completed[role] += 1
                    owner.pending_microbatches = 0
                    charge_wall()
                    owner.rng = capture_rng(owner.cuda_devices)
                    event = {"schema": "rz-rovent-original-learner-update/1", "window_id": spec["window_id"],
                        "stage_id": owner.stage_id, "role": role, "completed_role_updates": owner.completed[role],
                        "profile_sha256": training_profile_digest(spec), "corpus_sha256": admission["corpus_sha256"],
                        "phase": spec["phase"],
                        "sample_origins": [sample["origin_sha256"] for sample in samples],
                        "sample_inputs": [sample["input_sha256"] for sample in samples], "head_counts": counts,
                        "actual_model_state_sha256": state_digest(runtime[1](owner.model.state_dict()))}
                    original = execution_authority.completed_update(lease, event, owner)
                    fields(original, ("sha256", "bytes"), "original learner update receipt pin")
                    sha(original["sha256"])
                    if not uint(original["bytes"]):
                        raise ValueError("original learner completion receipt is empty")
                    owner.cost["output_bytes"] = uint(owner.cost["output_bytes"] + original["bytes"])
                    journal["updates"].append({**event, "original_completion": copy.deepcopy(original)})
                    for head, count in counts.items():
                        journal["head_counts"][head] = uint(journal["head_counts"][head] + count)
                    owner.inherited_input_sha256 = sorted(set(owner.inherited_input_sha256) | {sample["input_sha256"] for sample in samples})
                    owner.inherited_origin_sha256 = sorted(set(owner.inherited_origin_sha256) | {sample["origin_sha256"] for sample in samples})
                    if role == "validator" and journal["frozen_pc_sha256"] != _frozen_pc(owner, state_digest):
                        raise ValueError("V update mutated frozen shared/P/C/NNUEC")
                except BaseException:
                    if owner.pending_microbatches:
                        owner.cost["partial_accumulation_ns"] += time.monotonic_ns() - update_started
                    raise
                if sum(owner.completed.values()) % spec["checkpoint_interval"] == 0:
                    charge_wall()
                    owner.rng = capture_rng(owner.cuda_devices)
                    remaining = spec["max_output_bytes"] - (owner.cost["output_bytes"] - start_output)
                    pin = save_tensor_checkpoint(owner, directory / (spec["window_id"] + "-" + str(sum(owner.completed.values())) + ".pt"),
                        sampler_validator=validate_sampler, max_file_bytes=remaining, execute=True)
                    execution_authority.checkpoint_published(lease, pin, owner)
                    checkpoints.append(pin)
            else:
                status = "window_update_budget_complete"
            if spec["phase"] == "pc" and all(owner.completed[role] == available[role] for role in ROLES[:2]) and available["validator"]:
                journal["frozen_pc_sha256"] = _frozen_pc(owner, state_digest)
                status = "await_new_frozen_v_corpus"
            charge_wall()
            owner.rng = capture_rng(owner.cuda_devices)
            tensor_payload(owner, sampler_validator=validate_sampler, execute=True)
            remaining = spec["max_output_bytes"] - (owner.cost["output_bytes"] - start_output)
            final = save_tensor_checkpoint(owner, directory / (spec["window_id"] + "-final.pt"),
                sampler_validator=validate_sampler, max_file_bytes=remaining, execute=True)
            execution_authority.checkpoint_published(lease, final, owner)
            checkpoints.append(final)
        except BaseException:
            owner.quarantined = True
            raise
        finally:
            cleanup_error = None
            try:
                # The sole physical owner keeps tensors, optimizer workspace
                # and CUDA context alive through this fence. A failed fence
                # leaves the owner quarantined and cannot be exported/reused.
                for device in owner.cuda_devices:
                    torch.cuda.synchronize(device)
                owner.physical_completion = "confirmed"
            except BaseException as error:
                owner.physical_completion = "unknown"
                owner.quarantined = True
                cleanup_error = error
            try:
                charge_wall()
            except BaseException as error:
                owner.quarantined = True
                if cleanup_error is None:
                    cleanup_error = error
            try:
                owner.rng = capture_rng(owner.cuda_devices)
                if lease is not None:
                    execution_authority.end_window(lease, status=status, owner=owner)
            except BaseException as error:
                owner.quarantined = True
                if cleanup_error is None:
                    cleanup_error = error
            finally:
                try:
                    charge_wall()
                finally:
                    restore_rng(original_rng, owner.cuda_devices)
            if cleanup_error is not None:
                raise cleanup_error
    return {"schema": WINDOW_DOMAIN, "window_id": spec["window_id"], "status": status,
        "recipe_sha256": owner.recipe.sha256, "stage_id": owner.stage_id,
        "training_profile_sha256": training_profile_digest(spec), "admission": admission,
        "completed_updates": dict(owner.completed), "actual_optimizer_updates": sum(owner.completed.values()) - start_updates,
        "groups": groups, "samples_consumed": consumed_samples, "cost": dict(owner.cost),
        "checkpoint_cost_scope": "through_snapshot_before_its_own_write_and_final_drain",
        "physical_completion": owner.physical_completion,
        "post_snapshot_cost_retained_by_original_owner": True,
        "checkpoints": checkpoints, "actual_training_executed": sum(owner.completed.values()) > start_updates,
        "arena_eligible": False}
