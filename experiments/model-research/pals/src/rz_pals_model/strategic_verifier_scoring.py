"""Frozen numeric connection for the separately admitted private V query/2.

This offline Python seam executes the existing frozen validator on explicit
action and prior-observation features. It has no learned utility, action
execution, witness, target, loss, export, checkpoint, warm-state or product V
authority. Legacy query16 remains sealed in CheckedStrategicQuery; its numeric
seed is zero here. Existing P/C graphs and model parameters are unchanged.

Resource reservations cover this seam's planned CPU tensors, hashing workspace
and conservative attention/activation temporaries for the tracked fixed model.
They are not a process RSS or BLAS allocator limit. Original caller-owned model
weights, immutable admitted assets and the Python/Torch runtime are excluded.
Deadline checks are cooperative before/after kernels; no kernel preemption or
new time allowance is claimed. A late result is refused.
"""

from dataclasses import dataclass
import itertools
import math
import time

import torch
from torch import nn

from .config import ModelConfig, TASKS, matmul_flops
from .model import PalsModel, TensorInput
from . import semantic_verifier as semantic
from . import strategic_verifier_query as query


SCORER_DOMAIN = "rz-pals-private-v-frozen-action-scoring/1"
LAYOUT_DOMAIN = "rz-pals-private-v-action-memory/1"
SCOPE = "frozen_action_numeric_connection_only"
MAX_ACTIONS = 8
MAX_PRIOR = 16
MAX_PV = 96  # registered horizon64 + quiescence32, without invented truncation
TILE = 64
PRIVATE_TOKENS = semantic.TOKEN_COUNT + 1 + MAX_PRIOR * (1 + 2 * MAX_PV)
MAX_OWNER_BYTES = 512 * 1024 * 1024
DEFAULT_OWNER_BYTES = 256 * 1024 * 1024
MAX_MATMUL_FLOPS = 128_000_000_000
_STATES = ("completed", "deferred", "unavailable", "partial", "canceled", "failed", "missing")
_COMPLETIONS = ("depth_limit", "node_limit", "deadline", "canceled", "quiescence_limit", "rules_terminal")
_FEATURE_SCOPE = "checked_rules_branch_actual_action_controls_ordered_prior_known_unknown_cost_and_pv_only"


def _uint(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("action scorer bounded exact integer required")
    return value


def _deadline(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError("action scorer original caller absolute deadline expired")


def _config(config):
    if type(config) is not ModelConfig:
        raise ValueError("action scorer exact fixed ModelConfig required")
    expected = ModelConfig().to_dict()
    actual = config.to_dict()
    if actual != expected or any(type(value) is not int for value in actual.values()):
        raise ValueError("action scorer fixed configuration changed")
    config.validate()
    return query.canonical(actual)


def _feature(feature):
    """Sanity checks for immutable typed views, not another admission factory."""
    if type(feature) is not query.ActionFeatures or type(feature.branch) is not semantic.SemanticFeatures:
        raise ValueError("action scorer exact typed action/branch features required")
    _uint(feature.task, 0, len(TASKS) - 1)
    if type(feature.profile) is not query.ProfileFeatures:
        raise ValueError("action scorer typed registered profile required")
    profile = feature.profile
    if (any(type(value) is not int or value != 0 for value in
            (profile.search_kind, profile.value_kind, profile.ordering_kind))
            or profile.selective_reductions is not False):
        raise ValueError("action scorer supports registered legacy own bootstrap profile only")
    _uint(profile.tt_entries, 0, 1048576)
    _uint(profile.quiescence_ply, 0, 32)
    _uint(profile.requested_depth, 1, 64)
    _uint(feature.baseline_depth, 1, profile.requested_depth)
    _uint(feature.node_budget, 0 if feature.task == 6 else 1, 2**32 - 1)
    if feature.task == 6 and feature.node_budget != 0:
        raise ValueError("Defer has no CPU node reservation")
    _uint(feature.wall_ms, 1, 300000)
    _uint(feature.output_bytes, 1024, 1 << 20)
    if type(feature.prior) is not tuple or len(feature.prior) > MAX_PRIOR:
        raise ValueError("action scorer ordered prior extent")
    for prior in feature.prior:
        if type(prior) is not query.PriorFeatures:
            raise ValueError("action scorer exact typed prior features required")
        _uint(prior.task, 0, len(TASKS) - 1)
        if prior.state not in _STATES or type(prior.known) is not bool:
            raise ValueError("action scorer prior known/unknown state")
        if (type(prior.completed_depths) is not tuple or type(prior.completion) is not tuple
                or type(prior.pv) is not tuple or not len(prior.completed_depths) == len(prior.completion) == len(prior.pv) <= 2):
            raise ValueError("action scorer retained report extent")
        for depth, completion, pv in zip(prior.completed_depths, prior.completion, prior.pv):
            _uint(depth, 0, 64)
            if completion not in _COMPLETIONS or type(pv) is not tuple or len(pv) > MAX_PV:
                raise ValueError("action scorer completion/PV extent")
            for move in pv:
                if type(move) is not tuple or len(move) != 3:
                    raise ValueError("action scorer actual move components required")
                _uint(move[0], 0, 63)
                _uint(move[1], 0, 63)
                _uint(move[2], 0, 4)
        for value, maximum in ((prior.reported_nodes, 2 * (2**32 - 1)),
                               (prior.reported_elapsed_ms, 2**32 - 1), (prior.caller_elapsed_ms, 2**32 - 1)):
            if value is not None:
                _uint(value, 0, maximum)
        _uint(prior.raw_output_bytes, 0, query.MAX_RAW_BYTES)
    return feature


def token_descriptors(feature):
    """New fixed layout over actual typed features; padding has no basis.

    Catalogue slots, hashes, profile names, legacy bucket and prior.ordinal
    are never embedded. Prior block position means validated observation order,
    not an arbitrary ordinal/name feature. No CPU raw score is a reward feature.
    """
    _feature(feature)
    for descriptor in semantic.token_descriptors(feature.branch):
        yield "semantic", descriptor
    yield "action", feature
    for order in range(MAX_PRIOR):
        prior = feature.prior[order] if order < len(feature.prior) else None
        yield "prior", (order, prior)
        for report in range(2):
            moves = prior.pv[report] if prior is not None and report < len(prior.pv) else ()
            for position in range(MAX_PV):
                yield "pv", (order, report, position, moves[position] if position < len(moves) else None)


def _tile(model, descriptors):
    if type(descriptors) not in (tuple, list) or not 1 <= len(descriptors) <= 4:
        raise ValueError("action tile batch extent")
    length = len(descriptors[0])
    if not 1 <= length <= TILE or any(len(row) != length for row in descriptors):
        raise ValueError("action tile length extent")
    base = []
    for row in descriptors:
        values = []
        for kind, value in row:
            if kind == "semantic":
                values.append(value)
            elif kind == "pv":
                _, report, position, move = value
                values.append((17 + report, position, "move", move))
            elif kind in ("action", "prior"):
                values.append((0, 0, "question", None))
            else:
                raise ValueError("action tile kind")
        base.append(tuple(values))
    payload, mask = semantic._tile(model, base)
    for row, tokens in enumerate(descriptors):
        for slot, (kind, value) in enumerate(tokens):
            if kind == "action":
                feature = _feature(value)
                mask[row, slot] = True
                payload[row, slot, 340] = 1.0
                payload[row, slot, feature.task] = 1.0
                payload[row, slot, 21] = float(feature.cpu_check)
                if feature.cpu_check:
                    # The first profile supports one closed algorithm/value/
                    # ordering vocabulary; no profile name/digest is encoded.
                    payload[row, slot, 22] = 1.0
                    payload[row, slot, 24] = 1.0
                    payload[row, slot, 26] = 1.0
                    payload[row, slot, 32:39] = torch.tensor(feature.numeric_controls(), dtype=torch.float32, device="cpu")
            elif kind == "prior":
                order, prior = value
                if prior is None:
                    continue
                mask[row, slot] = True
                payload[row, slot, 341] = 1.0
                payload[row, slot, 320 + order] = 1.0
                payload[row, slot, prior.task] = 1.0
                payload[row, slot, 8 + _STATES.index(prior.state)] = 1.0
                payload[row, slot, 15] = float(prior.known)
                for report, (depth, completion) in enumerate(zip(prior.completed_depths, prior.completion)):
                    payload[row, slot, 32 + report] = depth / 64
                    payload[row, slot, 34 + report] = 1.0
                    payload[row, slot, 36 + 6 * report + _COMPLETIONS.index(completion)] = 1.0
                for coordinate, present, cost, scale in ((49, 50, prior.reported_nodes, 64),
                        (51, 52, prior.reported_elapsed_ms, 32), (53, 54, prior.caller_elapsed_ms, 32)):
                    if cost is not None:
                        payload[row, slot, coordinate] = math.log2(cost + 1) / scale
                        payload[row, slot, present] = 1.0
                payload[row, slot, 55] = prior.raw_output_bytes / query.MAX_RAW_BYTES
            elif kind == "pv" and value[3] is not None:
                payload[row, slot, 320 + value[0]] += 1.0
                payload[row, slot, 342] += 1.0
    return payload, mask


def _tensor_inputs_zero_seed(checked_inputs):
    # The original encoded query16 and input SHA are neither rewritten nor
    # relabeled. Only this new offline numeric invocation has a zero seed.
    original = semantic._tensor_inputs(checked_inputs)
    inputs = TensorInput(original.board, original.metadata, original.records, original.record_mask,
                         original.candidates, original.candidate_mask, original.divergence_features,
                         original.divergence_mask, torch.zeros_like(original.query))
    inputs.validate()
    return inputs


@dataclass(frozen=True)
class BatchReservation:
    actions: int
    records: int
    candidates: int
    public_tokens: int
    owner_bytes: int
    matmul_flops: int
    components: tuple


@dataclass(frozen=True)
class CatalogueReservation:
    batches: tuple
    owner_bytes: int
    peak_batch_bytes: int
    matmul_flops: int
    actions: int


def _batch_reservation(config, batch, records, candidates):
    """Conservative sum of owned tensors/temporaries, including repeated GQA.

    Sequential block scratch is charged once per physical invocation, at a
    conservative simultaneous upper bound. Whole-catalogue reservation sums
    these batch bounds even though batches execute sequentially. No unexecuted
    action or masked padding is dropped from this accounting.
    """
    _config(config)
    _uint(batch, 1, 4)
    _uint(records, 1, config.max_records)
    _uint(candidates, 1, config.max_candidates)
    w, kv, h, n, qh, hd = config.width, config.kv_heads * config.head_dimension, config.ffn_width, config.latent_slots, config.query_heads, config.head_dimension
    s, total = 66 + records, 66 + records + PRIVATE_TOKENS
    components = {
        "typed_host_inputs": batch * (64 * 8 + 16 * 4 + records * 16 * 4 + records + candidates * 3 * 8 + candidates + 8 * 4 + 1 + 2 * 16 * 4),
        "public_memory_and_projected_kv": batch * (4 * s * w * 4 + 2 * s * kv * 4),
        "board_record_attention_and_ffn_scratch": batch * (
            16 * (66 + 4 * records) * w * 4 + 8 * (66 + 4 * records) * h * 4
            + 4 * qh * (66 * 66 + records * 4 * 4) * 4 + qh * (66 * 66 + records * 4 * 4)),
        "combined_private_kv_and_mask": batch * (2 * total * kv * 4 + total),
        "tile_embedding_projection_and_indices": batch * TILE * (16 * w * 4 + 4 * kv * 4 + 6 * 8 + 2),
        "cross_attention_repeated_gqa_kv": batch * 2 * qh * total * hd * 4,
        "cross_attention_scores_masks_softmax_products": batch * (4 * qh * n * total * 4 + qh * n * total + 12 * n * w * 4),
        "private_self_attention_and_ffn_scratch": batch * (16 * n * w * 4 + 8 * n * h * 4 + 4 * qh * n * n * 4 + qh * n * n),
        "candidate_embedding_heads_and_returned_scores": batch * (10 * candidates * w * 4 + candidates * 3 * 8 + (candidates + len(TASKS) + 3) * 4 + 2 * n * w * 4),
        # Largest fixed MoveEmbedding projection is W x 3W; hashing and finite
        # checks may hold two byte views/copies. Preloaded weights are excluded.
        "parameter_digest_and_verification_workspace": 2 * 3 * w * w * 4 + 4 * query.MAX_RAW_BYTES,
        "additional_tensor_workspace": 2 * 1024 * 1024,
    }
    base = matmul_flops(config, "validator", records, candidates, batch=batch)["cold_forward_matmul_flops"]
    extra = batch * (4 * PRIVATE_TOKENS * w * kv
                     + 4 * n * PRIVATE_TOKENS * w * config.recurrent_blocks * config.iterations
                     + 6 * PRIVATE_TOKENS * w * w)
    return BatchReservation(batch, records, candidates, s, sum(components.values()), base + extra, tuple(sorted(components.items())))


def reservation_plan(checked_query, *, config, batch_size=4, owner_byte_limit=DEFAULT_OWNER_BYTES,
                     matmul_flop_limit=MAX_MATMUL_FLOPS):
    """Reserve every action before collating tensors or invoking the model."""
    _config(config)
    _uint(batch_size, 1, 4)
    _uint(owner_byte_limit, 1, MAX_OWNER_BYTES)
    _uint(matmul_flop_limit, 1, MAX_MATMUL_FLOPS)
    if type(checked_query) is not query.CheckedStrategicQuery:
        raise ValueError("action scorer exact CheckedStrategicQuery required")
    checked_query.verify()
    features = checked_query.features()
    checked_inputs = checked_query.action_semantic_inputs()
    if not 1 <= len(features) == len(checked_inputs) <= MAX_ACTIONS:
        raise ValueError("action scorer complete catalogue extent")
    for feature in features:
        _feature(feature)
    batches = []
    for start in range(0, len(features), batch_size):
        inputs = checked_inputs[start:start + batch_size]
        records = max(1, max(len(value.encoded_snapshot().public_records) for value in inputs))
        candidates = max(1, max(len(value.rules_receipt()["root"]["legal_moves"]) for value in inputs))
        batches.append(_batch_reservation(config, len(inputs), records, candidates))
    needed = sum(value.owner_bytes for value in batches) + len(features) * 4
    flops = sum(value.matmul_flops for value in batches)
    if needed > owner_byte_limit:
        raise ValueError("whole action catalogue owner byte reservation denied before allocation")
    if flops > matmul_flop_limit:
        raise ValueError("whole action catalogue matrix FLOPs reservation denied before execution")
    return CatalogueReservation(tuple(batches), needed, max(value.owner_bytes for value in batches), flops, len(features))


def _private_memory(model, public, features, *, deadline, public_tokens):
    key, value, public_mask = public
    batch = len(features)
    if (type(key) is not torch.Tensor or type(value) is not torch.Tensor or type(public_mask) is not torch.Tensor
            or key.shape != (batch, 2, public_tokens, 64) or value.shape != key.shape
            or public_mask.shape != (batch, public_tokens) or key.dtype != torch.float32 or value.dtype != torch.float32
            or public_mask.dtype != torch.bool or any(tensor.device.type != "cpu" for tensor in (key, value, public_mask))
            or key.requires_grad or value.requires_grad or not torch.all(torch.isfinite(key)) or not torch.all(torch.isfinite(value))):
        raise ValueError("action scorer public memory disagrees with reserved fixed shape")
    _deadline(deadline)
    combined_key = torch.empty((batch, 2, public_tokens + PRIVATE_TOKENS, 64), dtype=torch.float32, device="cpu")
    combined_value = torch.empty_like(combined_key)
    combined_mask = torch.zeros((batch, public_tokens + PRIVATE_TOKENS), dtype=torch.bool, device="cpu")
    combined_key[:, :, :public_tokens].copy_(key)
    combined_value[:, :, :public_tokens].copy_(value)
    combined_mask[:, :public_tokens].copy_(public_mask)
    streams = [iter(token_descriptors(feature)) for feature in features]
    for start in range(0, PRIVATE_TOKENS, TILE):
        _deadline(deadline)
        length = min(TILE, PRIVATE_TOKENS - start)
        descriptors = [tuple(itertools.islice(stream, length)) for stream in streams]
        if any(len(row) != length for row in descriptors):
            raise ValueError("action private layout did not produce all reserved tokens")
        payload, mask = _tile(model, descriptors)
        for target, projection in ((combined_key, model.public_encoder.memory_key), (combined_value, model.public_encoder.memory_value)):
            projected = projection(payload).view(batch, length, 2, 64).transpose(1, 2)
            target[:, :, public_tokens + start:public_tokens + start + length].copy_(projected)
            del projected
        combined_mask[:, public_tokens + start:public_tokens + start + length].copy_(mask)
        del payload, mask, descriptors
    if any(next(stream, None) is not None for stream in streams):
        raise ValueError("action private layout exceeded reservation")
    _deadline(deadline)
    return combined_key, combined_value, combined_mask


def _scores(value, count):
    if (type(value) is not torch.Tensor or value.shape != (count,) or value.dtype != torch.float32
            or value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None
            or not torch.all(torch.isfinite(value))):
        raise ValueError("finite frozen CPU FP32 per-action scores required")
    return value


class FrozenStrategicVerifierScorer(nn.Module):
    """No registered parameters/state, and no replacement model/head API.

    The caller first reloads/pins/fixes eval and gradient flags of the existing
    full PalsModel. Parameter observation is caller-conditional evidence, not
    training or checkpoint-provenance proof. Returned scores alone never admit
    utility or an action outcome. This wrapper deliberately has no selector.
    """
    def __init__(self, model, *, parameter_observation_bytes, expected_observation_sha256):
        super().__init__()
        if type(model) is not PalsModel:
            raise ValueError("action scorer exact existing PalsModel required")
        configuration = _config(model.config)
        guard = semantic.FrozenSemanticVerifier(model, parameter_observation_bytes=parameter_observation_bytes,
                                                expected_observation_sha256=expected_observation_sha256)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_guard", guard)
        self._configuration = configuration
        self._check_model()

    def _check_model(self):
        # Existing semantic collation uses the Torch defaults. Refuse changed
        # defaults BEFORE it can allocate elsewhere or cast a sealed FP32 row.
        if torch.get_default_dtype() != torch.float32 or torch.get_default_device().type != "cpu":
            raise ValueError("action scorer requires explicit CPU FP32 tensor defaults")
        if _config(self._model.config) != self._configuration:
            raise ValueError("action scorer model configuration changed")
        for module in self._model.modules():
            if hasattr(module, "config") and _config(module.config) != self._configuration:
                raise ValueError("action scorer existing module configuration changed")
        self._guard._check_model()

    def _forward_batch(self, checked_inputs, features, *, deadline):
        inputs = _tensor_inputs_zero_seed(checked_inputs)
        _deadline(deadline)
        public = self._model.public_encoder(*inputs.public_args())
        key, value, mask = _private_memory(self._model, public, features, deadline=deadline,
                                          public_tokens=66 + inputs.records.shape[1])
        _deadline(deadline)
        outputs = self._model.experts["validator"](
            self._model.reader_blocks, self._model.move_embedding, key, value, mask,
            inputs.candidates, inputs.candidate_mask, inputs.divergence_features, inputs.divergence_mask, inputs.query)
        logits, latent = outputs[3], outputs[2]
        batch = len(features)
        if (logits.shape != (batch, len(TASKS)) or latent.shape != (batch, 16, 384)
                or logits.dtype != torch.float32 or latent.dtype != torch.float32
                or logits.device.type != "cpu" or latent.device.type != "cpu"
                or logits.requires_grad or latent.requires_grad or logits.grad_fn is not None or latent.grad_fn is not None
                or not torch.all(torch.isfinite(logits)) or not torch.all(torch.isfinite(latent))):
            raise ValueError("finite existing validator task logits/private latent required")
        # Task codes choose the actual task head AFTER forward; they are real
        # action meanings. Catalogue indices only restore return order.
        rows = torch.arange(batch, dtype=torch.int64, device="cpu")
        tasks = torch.tensor([feature.task for feature in features], dtype=torch.int64, device="cpu")
        return logits[rows, tasks]

    def forward(self, checked_query, *, deadline, batch_size=4, owner_byte_limit=DEFAULT_OWNER_BYTES,
                matmul_flop_limit=MAX_MATMUL_FLOPS):
        _deadline(deadline)  # invalid/expired deadline is refused before views/I/O/NN
        plan = reservation_plan(checked_query, config=self._model.config, batch_size=batch_size,
                                owner_byte_limit=owner_byte_limit, matmul_flop_limit=matmul_flop_limit)
        _deadline(deadline)
        features = checked_query.features()
        checked_inputs = checked_query.action_semantic_inputs()
        identity = checked_query.sha256
        _deadline(deadline)
        self._check_model()
        _deadline(deadline)
        results = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                _deadline(deadline)
                checked_query.verify()
                _deadline(deadline)
                self._check_model()
                _deadline(deadline)
                batch_features = features[start:start + batch_size]
                value = self._forward_batch(checked_inputs[start:start + batch_size], batch_features, deadline=deadline)
                _scores(value, len(batch_features))
                _deadline(deadline)
                self._check_model()
                checked_query.verify()
                _deadline(deadline)
                results.append(value)
            scores = _scores(torch.cat(results), len(features))
        self._check_model()
        checked_query.verify()
        if checked_query.sha256 != identity:
            raise ValueError("action scorer checked query identity changed")
        _deadline(deadline)
        audit = {"schema": SCORER_DOMAIN, "scope": SCOPE, "layout_domain": LAYOUT_DOMAIN,
            "query_sha256": identity, "semantic_input_sha256": tuple(value.sha256 for value in checked_inputs),
            "model_config_sha256": query.byte_pin(self._configuration)["sha256"],
            "parameter_sha256": self._guard._parameter_sha, "parameter_observation_sha256": self._guard._observation_sha,
            "action_tasks": tuple(TASKS[value.task] for value in features), "actions": len(features),
            "private_tokens": PRIVATE_TOKENS, "tile_tokens": TILE, "legacy_query_seed": "all_zero",
            "feature_scope": _FEATURE_SCOPE, "hash_name_slot_bucket_features": False,
            "owner_reserved_bytes": plan.owner_bytes, "peak_batch_reserved_bytes": plan.peak_batch_bytes,
            "owner_byte_limit": owner_byte_limit, "matrix_flops_reserved": plan.matmul_flops,
            "matrix_flop_limit": matmul_flop_limit, "matrix_scope": "analytic_matrix_products_fma_2_with_conservative_all_private_token_move_projection_charge",
            "matrix_excluded": ("bias", "norm", "softmax", "silu", "elementwise", "pooling", "lookup", "transfer", "cpu_search"),
            "batches": tuple({"actions": batch.actions, "physical_records": batch.records, "physical_candidates": batch.candidates,
                              "public_tokens": batch.public_tokens, "owner_bytes": batch.owner_bytes,
                              "matrix_flops": batch.matmul_flops, "components": dict(batch.components)} for batch in plan.batches),
            "allocation_scope": "planned_tensor_and_verification_workspace;not_process_rss_or_backend_allocator_limit",
            "deadline_scope": "original_absolute_cooperative_before_after;no_kernel_preemption",
            "caller_parameter_assurance": "independently_pinned_caller_parameter_observation",
            "source_authority": False, "build_authority": False, "binary_loaded_image_verified_here": False,
            "action_scoring_executed": True, "action_execution_authority": False, "utility_authority": False,
            "target_authority": False, "training_authority": False, "learned_utility_claim": False,
            "past_input_equivalence_claim": False, "strategic_strength_claim": False,
            "product_verifier_enabled": False, "private_warm_continuation": False,
            "actual_training_executed": False, "backward_executed": False, "optimizer_steps": 0}
        self._check_model()
        checked_query.verify()
        _deadline(deadline)
        return scores, audit
