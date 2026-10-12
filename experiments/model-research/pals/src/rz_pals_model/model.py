"""Shared public memory and hard-routed private experts, with real neural math.

No optimizer or training loop exists here. P/C forward cannot access V private
parameters. Board and independent record encoders are role-neutral. Decoder
self-attention recomputes latent K/V on each recurrent invocation; only immutable
public-memory K/V may be retained. All first-profile arithmetic is FP32.
"""
import copy
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from .config import INTERACTION_WIDTH, LINE_WIDTH, MAX_LINE_PLIES, ModelConfig, ROLES, TASKS


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value):
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, value):
        return self.down(torch.nn.functional.silu(self.gate(value)) * self.up(value))


def attend(query, key, value, key_mask, config):
    """Standard ONNX ops; no CUDA-specific fused operator or CPU fallback."""
    key = key.repeat_interleave(config.query_heads // config.kv_heads, dim=1)
    value = value.repeat_interleave(config.query_heads // config.kv_heads, dim=1)
    scores = torch.matmul(query, key.transpose(-1, -2)) * (config.head_dimension ** -0.5)
    scores = scores.masked_fill(~key_mask[:, None, None, :], -1e9)
    return torch.matmul(torch.softmax(scores, dim=-1), value)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.width, config.width, bias=False)
        self.key = nn.Linear(config.width, config.kv_heads * config.head_dimension, bias=False)
        self.value = nn.Linear(config.width, config.kv_heads * config.head_dimension, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)

    def project_memory(self, memory):
        b, s, _ = memory.shape
        return (self.key(memory).reshape(b, s, self.config.kv_heads, self.config.head_dimension).transpose(1, 2),
                self.value(memory).reshape(b, s, self.config.kv_heads, self.config.head_dimension).transpose(1, 2))

    def forward(self, latent, key, value, mask):
        b, n, _ = latent.shape
        query = self.query(latent).reshape(b, n, self.config.query_heads, self.config.head_dimension).transpose(1, 2)
        result = attend(query, key, value, mask, self.config).transpose(1, 2).reshape(b, n, self.config.width)
        return self.output(result)

    def self_attention(self, tokens, mask):
        key, value = self.project_memory(tokens)
        return self(tokens, key, value, mask)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.width)
        self.ffn = SwiGLU(config.width, config.ffn_width)

    def forward(self, value, mask):
        value = value + self.attention.self_attention(self.attention_norm(value), mask)
        return value + self.ffn(self.ffn_norm(value))


class PublicEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.piece_embedding = nn.Embedding(13, config.width)
        self.square_embedding = nn.Parameter(torch.randn(64, config.width) * 0.02)
        self.metadata_projection = nn.Linear(16, 2 * config.width, bias=False)
        self.board_blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.board_blocks))
        # Four small structured field groups are contextualized only within one
        # immutable record. Changing a different record cannot change this token.
        self.record_projection = nn.Linear(4, config.width, bias=False)
        self.record_field_embedding = nn.Parameter(torch.randn(4, config.width) * 0.02)
        self.record_blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.record_blocks))
        self.memory_key = nn.Linear(config.width, config.kv_heads * config.head_dimension, bias=False)
        self.memory_value = nn.Linear(config.width, config.kv_heads * config.head_dimension, bias=False)

    def encode_records(self, records, record_line_tokens=None, record_line_mask=None):
        b, r, _ = records.shape
        fields = self.record_projection(records.reshape(b * r, 4, 4)) + self.record_field_embedding
        mask = torch.ones((b * r, 4), dtype=torch.bool, device=records.device)
        for block in self.record_blocks:
            fields = block(fields, mask)
        result = fields.mean(dim=1).reshape(b, r, self.config.width)
        if self.config.full_line:
            if record_line_tokens is None or record_line_mask is None:
                raise ValueError("full-line record tensors are required")
            result = result + self.record_line_projection(self.line_encoder(record_line_tokens, record_line_mask))
        return result

    def forward(self, board, metadata, records, record_mask, record_line_tokens=None, record_line_mask=None):
        b = board.shape[0]
        square = self.piece_embedding(board) + self.square_embedding
        meta = self.metadata_projection(metadata).reshape(b, 2, self.config.width)
        board_tokens = torch.cat((square, meta), dim=1)
        board_mask = torch.ones((b, 66), dtype=torch.bool, device=board.device)
        for block in self.board_blocks:
            board_tokens = block(board_tokens, board_mask)
        record_tokens = self.encode_records(records, record_line_tokens, record_line_mask)
        memory = torch.cat((board_tokens, record_tokens), dim=1)
        mask = torch.cat((board_mask, record_mask), dim=1)
        key = self.memory_key(memory).reshape(b, -1, self.config.kv_heads, self.config.head_dimension).transpose(1, 2)
        value = self.memory_value(memory).reshape(b, -1, self.config.kv_heads, self.config.head_dimension).transpose(1, 2)
        return key, value, mask


class ReaderBlock(nn.Module):
    """Shared cross/latent attention; private FFN is supplied by selected expert."""
    def __init__(self, config):
        super().__init__()
        self.cross_norm = RMSNorm(config.width)
        self.cross = Attention(config)
        # Only query/output of cross are used: the single public projection bank
        # is owned by PublicEncoder, so there are no role/layer copies of it.
        del self.cross.key
        del self.cross.value
        self.self_norm = RMSNorm(config.width)
        self.self_attn = Attention(config)

    def forward(self, latent, key, value, mask):
        latent = latent + self.cross(self.cross_norm(latent), key, value, mask)
        latent_mask = torch.ones(latent.shape[:2], dtype=torch.bool, device=latent.device)
        return latent + self.self_attn.self_attention(self.self_norm(latent), latent_mask)


class MoveEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.from_square = nn.Embedding(64, config.width)
        self.to_square = nn.Embedding(64, config.width)
        self.promotion = nn.Embedding(5, config.width)
        self.projection = nn.Linear(3 * config.width, config.width, bias=False)

    def forward(self, candidates):
        return self.projection(torch.cat((self.from_square(candidates[:, :, 0]),
                                         self.to_square(candidates[:, :, 1]),
                                         self.promotion(candidates[:, :, 2])), dim=-1))


class FullLineEncoder(nn.Module):
    """Independent ordered move/ply encoder shared by record and query lines.

    Mask before every lookup/convolution and after each residual keeps padded
    move values from affecting a live neighbor. Empty lines pool to exact zero.
    Relationships, global IDs, revisions and hashes are never inputs here.
    """
    def __init__(self):
        super().__init__()
        self.moves = MoveEmbedding(SimpleNamespace(width=LINE_WIDTH))
        self.ply_embedding = nn.Embedding(MAX_LINE_PLIES, LINE_WIDTH)
        self.temporal_blocks = nn.ModuleList(nn.Conv1d(LINE_WIDTH, LINE_WIDTH, 3, padding=dilation,
                                                      dilation=dilation, bias=False) for dilation in (1, 2))
        self.pool_attention = nn.Linear(LINE_WIDTH, 1, bias=False)

    def forward(self, tokens, mask):
        groups = tokens.shape[:-2]
        moves = tokens.reshape(-1, MAX_LINE_PLIES, 3)
        active = mask.reshape(-1, MAX_LINE_PLIES)
        safe_moves = moves.masked_fill(~active[:, :, None], 0)
        value = self.moves(safe_moves) + self.ply_embedding(torch.arange(MAX_LINE_PLIES, device=tokens.device))[None, :, :]
        value = value * active[:, :, None]
        for temporal in self.temporal_blocks:
            value = (value + torch.nn.functional.silu(temporal(value.transpose(1, 2)).transpose(1, 2))) * active[:, :, None]
        scores = self.pool_attention(value).squeeze(-1).masked_fill(~active, -1e9)
        weights = torch.softmax(scores, dim=-1) * active
        return (value * weights[:, :, None]).sum(dim=1).reshape(*groups, LINE_WIDTH)


class VirtualReviewLineEncoder(FullLineEncoder):
    """One dynamic-L bank; the caller admits the six physical length buckets."""
    def forward(self, tokens, mask):
        groups, length = tokens.shape[:-2], tokens.shape[-2]
        moves = tokens.reshape(-1, length, 3)
        active = mask.reshape(-1, length)
        safe_moves = moves.masked_fill(~active[:, :, None], 0)
        value = self.moves(safe_moves) + self.ply_embedding(torch.arange(length, device=tokens.device))[None, :, :]
        value = value * active[:, :, None]
        for temporal in self.temporal_blocks:
            value = (value + torch.nn.functional.silu(temporal(value.transpose(1, 2)).transpose(1, 2))) * active[:, :, None]
        weights = torch.softmax(self.pool_attention(value).squeeze(-1).masked_fill(~active, -1e9), dim=-1) * active
        return (value * weights[:, :, None]).sum(dim=1).reshape(*groups, LINE_WIDTH)


class VirtualReviewPublicEncoder(nn.Module):
    """Shared board bank for up to four exact anchors and independent pages.

    Public K/V remains root-66 + selected records. Anchor contexts are an
    explicit additional public result for query conditioning. Relationships
    never enter an independently reusable record page.
    """
    def __init__(self, original, config, line_encoder, line_projection):
        super().__init__()
        self.config = config
        for name in ("piece_embedding", "square_embedding", "metadata_projection", "board_blocks",
                     "record_field_embedding", "record_blocks", "memory_key", "memory_value"):
            setattr(self, name, getattr(original, name))
        self.record_projection = nn.Linear(2, config.width, bias=False)
        self.anchor_record_projection = nn.Linear(config.width, config.width, bias=False)
        self.line_encoder, self.record_line_projection = line_encoder, line_projection

    def encode_records(self, records, anchor_context, record_anchor, record_mask, record_line_tokens, record_line_mask):
        b, r, _ = records.shape
        fields = self.record_projection(records.reshape(b * r, 4, 2)) + self.record_field_embedding
        field_mask = torch.ones((b * r, 4), dtype=torch.bool, device=records.device)
        for block in self.record_blocks:
            fields = block(fields, field_mask)
        safe_anchor = record_anchor.masked_fill(~record_mask, 0)
        anchored = torch.gather(anchor_context, 1, safe_anchor[:, :, None].expand(b, r, self.config.width))
        result = fields.mean(dim=1).reshape(b, r, self.config.width)
        result = result + self.record_line_projection(self.line_encoder(record_line_tokens, record_line_mask))
        result = result + self.anchor_record_projection(anchored)
        return result * record_mask[:, :, None]

    def forward(self, anchor_board, anchor_metadata, anchor_mask, records, record_anchor,
                record_line_tokens, record_line_mask, record_mask):
        b, a, _ = anchor_board.shape
        safe_board = anchor_board.masked_fill(~anchor_mask[:, :, None], 0)
        safe_metadata = anchor_metadata.masked_fill(~anchor_mask[:, :, None], 0)
        square = self.piece_embedding(safe_board.reshape(b * a, 64)) + self.square_embedding
        meta = self.metadata_projection(safe_metadata.reshape(b * a, 16)).reshape(b * a, 2, self.config.width)
        board_tokens = torch.cat((square, meta), dim=1)
        board_mask = torch.ones((b * a, 66), dtype=torch.bool, device=anchor_board.device)
        for block in self.board_blocks:
            board_tokens = block(board_tokens, board_mask)
        board_tokens = board_tokens.reshape(b, a, 66, self.config.width)
        anchor_context = board_tokens.mean(dim=2) * anchor_mask[:, :, None]
        root_tokens = board_tokens[:, 0]
        record_tokens = self.encode_records(records, anchor_context, record_anchor, record_mask, record_line_tokens, record_line_mask)
        memory = torch.cat((root_tokens, record_tokens), dim=1)
        mask = torch.cat((torch.ones((b, 66), dtype=torch.bool, device=anchor_board.device), record_mask), dim=1)
        key = self.memory_key(memory).reshape(b, -1, self.config.kv_heads, self.config.head_dimension).transpose(1, 2)
        value = self.memory_value(memory).reshape(b, -1, self.config.kv_heads, self.config.head_dimension).transpose(1, 2)
        return key, value, mask, anchor_context


class CandidateInteractionHead(nn.Module):
    """Nonlinear candidate/context joint score permits from-dependent ranking."""
    def __init__(self, width):
        super().__init__()
        self.hidden = nn.Linear(2 * width, INTERACTION_WIDTH)
        self.output = nn.Linear(INTERACTION_WIDTH, 1, bias=False)

    def forward(self, candidates, context):
        context = context[:, None, :].expand(-1, candidates.shape[1], -1)
        return self.output(torch.nn.functional.silu(self.hidden(torch.cat((candidates, context), dim=-1))))


def relation_features(key, value, memory_mask, relations):
    """Gather local record relationships from independent public K/V only."""
    own = torch.cat((key[:, :, 66:, :], value[:, :, 66:, :]), dim=1).transpose(1, 2).flatten(2)
    batch, records, width = own.shape
    active_records = memory_mask[:, 66:]
    related = []
    for axis in range(2):
        index = relations[:, :, axis]
        selected = torch.gather(own, 1, index.clamp_min(0)[:, :, None].expand(batch, records, width))
        related.append(selected * (index >= 0)[:, :, None])
    active = active_records & (relations >= 0).any(dim=-1)
    return torch.cat((own, *related), dim=-1), active


class RoleExpert(nn.Module):
    def __init__(self, role, config):
        super().__init__()
        self.role, self.config = role, config
        self.initial_latent = nn.Parameter(torch.randn(config.latent_slots, config.width) * 0.02)
        self.query_projection = nn.Linear(16, config.width, bias=False)
        self.ffn_norms = nn.ModuleList(RMSNorm(config.width) for _ in range(config.recurrent_blocks))
        self.ffns = nn.ModuleList(SwiGLU(config.width, config.ffn_width) for _ in range(config.recurrent_blocks))
        self.policy_head = nn.Linear(config.width, 1, bias=False)
        self.wdl_head = nn.Linear(config.width, 3, bias=False)
        if role == "critic":
            self.divergence_projection = nn.Linear(8, config.width, bias=False)
            self.divergence_head = nn.Linear(config.width, 1, bias=False)
        if role == "validator":
            self.task_head = nn.Linear(config.width, len(TASKS), bias=False)

    def initial(self, query, query_line_features=None, graph_relations=None, graph_relation_mask=None, query_anchor_features=None):
        conditioning = self.query_projection(query)
        if self.config.full_line or self.config.virtual_review:
            if query_line_features is None or graph_relations is None or graph_relation_mask is None:
                raise ValueError("full-line private query/relationship tensors are required")
            conditioning = conditioning + self.query_line_projection(query_line_features.flatten(1))
            relations = self.relation_output(torch.nn.functional.silu(self.relation_hidden(graph_relations)))
            relations = (relations * graph_relation_mask[:, :, None]).sum(dim=1) / graph_relation_mask.sum(dim=1, keepdim=True).clamp_min(1)
            conditioning = conditioning + relations
        if self.config.virtual_review:
            if query_anchor_features is None:
                raise ValueError("virtual review query anchor contexts are required")
            conditioning = conditioning + self.query_anchor_projection(query_anchor_features.flatten(1))
        return self.initial_latent[None, :, :] + conditioning[:, None, :]

    def candidate_logits(self, move_embedding, candidates, context):
        embedded = move_embedding(candidates)
        if self.config.interaction_head:
            return self.interaction_policy_head(embedded, context).squeeze(-1)
        return self.policy_head(embedded * context[:, None, :]).squeeze(-1)

    def forward(self, reader_blocks, move_embedding, key, value, mask, candidates, candidate_mask,
                divergence_features, divergence_mask, query, query_line_features=None,
                graph_relations=None, graph_relation_mask=None, query_anchor_features=None):
        latent = self.initial(query, query_line_features, graph_relations, graph_relation_mask, query_anchor_features)
        for _ in range(self.config.iterations):
            for index, reader in enumerate(reader_blocks):
                latent = reader(latent, key, value, mask)
                latent = latent + self.ffns[index](self.ffn_norms[index](latent))
        context = latent.mean(dim=1)
        candidate_logits = self.candidate_logits(move_embedding, candidates, context)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1e9)
        outputs = (candidate_logits, self.wdl_head(context), latent)
        if self.role == "critic":
            divergences = self.divergence_head(self.divergence_projection(divergence_features) * context[:, None, :]).squeeze(-1)
            outputs += (divergences.masked_fill(~divergence_mask, -1e9),)
        if self.role == "validator":
            outputs += (self.task_head(context),)
        return outputs

    def forward_from_initial(self, reader_blocks, move_embedding, key, value, mask,
                             candidates, candidate_mask, divergence_features,
                             divergence_mask, initial_latent):
        """Parameter-free approximate entry point; seed is the whole start state.

        The legacy forward above is intentionally unchanged, including its
        exported execution path. Keep this recurrent/head body numerically
        checked against that path, rather than silently rewriting old exports.
        Query has already conditioned an accepted final seed and is not added
        again. Private latent K/V is recomputed by every existing reader call.
        """
        latent = initial_latent
        for _ in range(self.config.iterations):
            for index, reader in enumerate(reader_blocks):
                latent = reader(latent, key, value, mask)
                latent = latent + self.ffns[index](self.ffn_norms[index](latent))
        context = latent.mean(dim=1)
        candidate_logits = self.candidate_logits(move_embedding, candidates, context)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1e9)
        outputs = (candidate_logits, self.wdl_head(context), latent)
        if self.role == "critic":
            divergences = self.divergence_head(self.divergence_projection(divergence_features) * context[:, None, :]).squeeze(-1)
            outputs += (divergences.masked_fill(~divergence_mask, -1e9),)
        if self.role == "validator":
            outputs += (self.task_head(context),)
        return outputs


class RoleGraph(nn.Module):
    """An exported graph contains exactly one private expert, never a router."""
    def __init__(self, reader_blocks, move_embedding, expert, line_encoder=None):
        super().__init__()
        self.reader_blocks, self.move_embedding, self.expert = reader_blocks, move_embedding, expert
        self.line_encoder = line_encoder

    def forward(self, key, value, mask, candidates, candidate_mask, divergence_features, divergence_mask, query,
                query_line_tokens=None, query_line_mask=None, record_relations=None):
        extra = ()
        if self.expert.config.full_line:
            if query_line_tokens is None or query_line_mask is None or record_relations is None:
                raise ValueError("full-line private graph inputs are required")
            features, active = relation_features(key, value, mask, record_relations)
            extra = (self.line_encoder(query_line_tokens, query_line_mask), features, active)
        return self.expert(self.reader_blocks, self.move_embedding, key, value, mask, candidates,
                           candidate_mask, divergence_features, divergence_mask, query, *extra)


class WarmRoleGraph(RoleGraph):
    """CPU reference for the separate approximate P/C artifact domain.

    This wrapper is not traced through torch.onnx.export: onnx_warm constructs
    a real ONNX If. The scalar mode applies to the complete physical batch;
    there is no per-row router. Shape/dtype/finite checks apply in both modes.
    Caller seed acceptance/Rules context is certified by the Rust owner, not
    by this numerical wrapper. There is no implicit seed-zero => Fresh rule.
    """
    def __init__(self, reader_blocks, move_embedding, expert, line_encoder=None):
        if expert.role not in ("proposer", "critic"):
            raise ValueError("private warm graph is P/C-only; validator is unsupported")
        super().__init__(reader_blocks, move_embedding, expert, line_encoder)

    def forward(self, key, value, mask, candidates, candidate_mask,
                divergence_features, divergence_mask, query, *arguments):
        config = self.expert.config
        config.validate()
        expected = 5 if config.full_line else 2
        if len(arguments) != expected:
            raise ValueError("private warm profile input count mismatch")
        initial_latent, warm_start = arguments[-2:]
        line_arguments = arguments[:-2]
        batch = key.shape[0] if key.ndim == 4 else 0
        if not isinstance(initial_latent, torch.Tensor) or batch < 1 or initial_latent.shape != (batch, config.latent_slots, config.width):
            raise ValueError("private warm initial latent shape must be [batch,16,384]")
        if initial_latent.dtype != torch.float32 or not torch.all(torch.isfinite(initial_latent)):
            raise ValueError("private warm seed requires finite FP32")
        if not isinstance(warm_start, torch.Tensor) or warm_start.shape != () or warm_start.dtype != torch.bool:
            raise ValueError("private warm mode must be one scalar bool tensor")
        if initial_latent.device != key.device or warm_start.device != key.device:
            raise ValueError("private warm seed/mode is on a different device")
        if not bool(warm_start.item()):
            return super().forward(key, value, mask, candidates, candidate_mask,
                                   divergence_features, divergence_mask, query, *line_arguments)
        return self.expert.forward_from_initial(
            self.reader_blocks, self.move_embedding, key, value, mask, candidates,
            candidate_mask, divergence_features, divergence_mask, initial_latent)


class VirtualReviewRoleGraph(RoleGraph):
    def forward(self, key, value, mask, candidates, candidate_mask, divergence_features,
                divergence_mask, query, anchor_context, query_anchor, query_line_tokens,
                query_line_mask, record_relations):
        graph_relations, graph_active = relation_features(key, value, mask, record_relations)
        b = key.shape[0]
        anchor_features = torch.gather(anchor_context, 1, query_anchor[:, :, None].expand(b, 3, self.expert.config.width))
        return self.expert(self.reader_blocks, self.move_embedding, key, value, mask, candidates,
                           candidate_mask, divergence_features, divergence_mask, query,
                           self.line_encoder(query_line_tokens, query_line_mask), graph_relations,
                           graph_active, anchor_features)


class VirtualReviewWarmRoleGraph(VirtualReviewRoleGraph):
    """Separate approximate entry; accepted seed lifetime is Rust-owned."""
    def forward(self, key, value, mask, candidates, candidate_mask, divergence_features,
                divergence_mask, query, anchor_context, query_anchor, query_line_tokens,
                query_line_mask, record_relations, initial_latent, warm_start):
        config = self.expert.config
        if self.expert.role not in ("proposer", "critic"):
            raise ValueError("V and value use Fresh only")
        if initial_latent.shape != (key.shape[0], config.latent_slots, config.width) or initial_latent.dtype != torch.float32 or not torch.all(torch.isfinite(initial_latent)):
            raise ValueError("virtual Warm accepted seed shape/finite FP32")
        if warm_start.shape != () or warm_start.dtype != torch.bool or initial_latent.device != key.device or warm_start.device != key.device:
            raise ValueError("virtual Warm scalar mode/device")
        if not bool(warm_start.item()):
            return super().forward(key, value, mask, candidates, candidate_mask, divergence_features, divergence_mask,
                                   query, anchor_context, query_anchor, query_line_tokens, query_line_mask, record_relations)
        if torch.any(query[:, 4] != 0):
            raise ValueError("Fresh value query cannot consume approximate seed")
        return self.expert.forward_from_initial(self.reader_blocks, self.move_embedding, key, value, mask, candidates,
                                               candidate_mask, divergence_features, divergence_mask, initial_latent)


class PalsModel(nn.Module):
    def __init__(self, config=ModelConfig(), roles=ROLES):
        super().__init__()
        config.validate()
        if not roles or any(role not in ROLES for role in roles) or len(set(roles)) != len(roles):
            raise ValueError("invalid hard-routed role set")
        self.config = config
        self.public_encoder = PublicEncoder(config)
        self.reader_blocks = nn.ModuleList(ReaderBlock(config) for _ in range(config.recurrent_blocks))
        self.move_embedding = MoveEmbedding(config)
        self.experts = nn.ModuleDict((role, RoleExpert(role, config)) for role in roles)
        if config.profile != "legacy_summary_v1":
            # Build the unchanged legacy bank first, then draw all new V2 banks
            # in one fixed order. Selecting an ablation changes registered
            # components, never the seeded common/input/head parameter values.
            line_encoder = VirtualReviewLineEncoder() if config.virtual_review else FullLineEncoder()
            record_projection = nn.Linear(LINE_WIDTH, config.width, bias=False)
            if config.full_line:
                self.public_encoder.line_encoder = line_encoder
                self.public_encoder.record_line_projection = record_projection
            elif config.virtual_review:
                self.public_encoder = VirtualReviewPublicEncoder(self.public_encoder, config, line_encoder, record_projection)
            for expert in self.experts.values():
                interaction = CandidateInteractionHead(config.width)
                query_projection = nn.Linear(3 * LINE_WIDTH, config.width, bias=False)
                relation_hidden = nn.Linear(3 * 2 * config.kv_heads * config.head_dimension, INTERACTION_WIDTH)
                relation_output = nn.Linear(INTERACTION_WIDTH, config.width, bias=False)
                if config.interaction_head:
                    expert.interaction_policy_head = interaction
                    del expert.policy_head
                if config.full_line or config.virtual_review:
                    expert.query_line_projection = query_projection
                    expert.relation_hidden = relation_hidden
                    expert.relation_output = relation_output
                if config.virtual_review:
                    expert.query_projection = nn.Linear(8, config.width, bias=False)
                    expert.query_anchor_projection = nn.Linear(3 * config.width, config.width, bias=False)

    def role_graph(self, role):
        if role not in self.experts:
            raise ValueError("role absent from frozen model")
        if self.config.virtual_review:
            return VirtualReviewRoleGraph(self.reader_blocks, self.move_embedding, self.experts[role], self.public_encoder.line_encoder)
        return RoleGraph(self.reader_blocks, self.move_embedding, self.experts[role],
                         self.public_encoder.line_encoder if self.config.full_line else None)

    def warm_role_graph(self, role):
        if role not in self.experts:
            raise ValueError("role absent from frozen model")
        if self.config.virtual_review:
            return VirtualReviewWarmRoleGraph(self.reader_blocks, self.move_embedding, self.experts[role], self.public_encoder.line_encoder)
        return WarmRoleGraph(self.reader_blocks, self.move_embedding, self.experts[role],
                             self.public_encoder.line_encoder if self.config.full_line else None)

    def without_validator(self):
        result = copy.deepcopy(self)
        if "validator" in result.experts:
            del result.experts["validator"]
        return result

    def virtual_review_pc_export_view(self):
        """Borrow frozen P/C banks for serialization without copying V weights.

        This read-only export view shares parameter storage with its owner and
        must not outlive or mutate that owner. It creates neither a neural
        module bank nor a session and draws no RNG. The graph builders still
        audit every serialized initializer against the explicit P/C allowlist.
        Historical ``without_validator`` keeps its independent-copy behavior.
        """
        if not self.config.virtual_review or self.training or not {"proposer", "critic"} <= set(self.experts):
            raise ValueError("V3 P/C export view requires an eval-mode virtual model")
        result = copy.copy(self)
        # nn.Module.__setattr__ updates _modules. Give this wrapper its own map
        # before replacing experts so the source collection model retains V.
        result._modules = self._modules.copy()
        result.experts = nn.ModuleDict((role, self.experts[role]) for role in ("proposer", "critic"))
        return result


def initialize(seed, config=ModelConfig()):
    """Preserves caller RNG; random initialization carries no learned claims."""
    if not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also resets CUDA/MPS/XPU generators, which this
        # CPU-only fork does not save. Touch only the scoped CPU generator.
        torch.random.default_generator.manual_seed(seed)
        return PalsModel(config).eval()


@dataclass(frozen=True)
class TensorInput:
    board: torch.Tensor
    metadata: torch.Tensor
    records: torch.Tensor
    record_mask: torch.Tensor
    candidates: torch.Tensor
    candidate_mask: torch.Tensor
    divergence_features: torch.Tensor
    divergence_mask: torch.Tensor
    query: torch.Tensor
    record_line_tokens: torch.Tensor = None
    record_line_mask: torch.Tensor = None
    query_line_tokens: torch.Tensor = None
    query_line_mask: torch.Tensor = None
    record_relations: torch.Tensor = None

    def validate(self, config=ModelConfig()):
        config.validate()
        b = self.board.shape[0] if self.board.ndim == 2 else 0
        if b < 1 or self.board.shape != (b, 64) or self.board.dtype != torch.int64 or torch.any((self.board < 0) | (self.board > 12)):
            raise ValueError("invalid board shape/dtype/piece code")
        if self.metadata.shape != (b, 16) or self.query.shape != (b, 16):
            raise ValueError("metadata/query shape")
        r = self.records.shape[1] if self.records.ndim == 3 else 0
        c = self.candidates.shape[1] if self.candidates.ndim == 3 else 0
        d = self.divergence_features.shape[1] if self.divergence_features.ndim == 3 else 0
        if not 1 <= r <= config.max_records or self.records.shape != (b, r, 16) or self.record_mask.shape != (b, r):
            raise ValueError("record shape/capacity; an empty set uses one explicitly masked slot")
        if not 1 <= c <= config.max_candidates or self.candidates.shape != (b, c, 3) or self.candidate_mask.shape != (b, c):
            raise ValueError("candidate shape/capacity")
        if self.candidates.dtype != torch.int64 or torch.any((self.candidates[:, :, :2] < 0) | (self.candidates[:, :, :2] > 63)) or torch.any((self.candidates[:, :, 2] < 0) | (self.candidates[:, :, 2] > 4)):
            raise ValueError("candidate move code")
        if torch.any(self.candidate_mask & (self.candidates[:, :, 0] == self.candidates[:, :, 1])):
            raise ValueError("active candidate has identical from/to")
        if not 1 <= d <= config.max_divergences or self.divergence_features.shape != (b, d, 8) or self.divergence_mask.shape != (b, d):
            raise ValueError("divergence shape/capacity")
        for value in (self.metadata, self.records, self.divergence_features, self.query):
            if value.dtype != torch.float32 or not torch.all(torch.isfinite(value)):
                raise ValueError("input requires finite FP32")
        for mask in (self.record_mask, self.candidate_mask, self.divergence_mask):
            if mask.dtype != torch.bool:
                raise ValueError("mask dtype must be bool")
        full = (self.record_line_tokens, self.record_line_mask, self.query_line_tokens,
                self.query_line_mask, self.record_relations)
        if any(value is not None for value in full) and not all(isinstance(value, torch.Tensor) for value in full):
            raise ValueError("partial full-line payload")
        if not config.full_line and any(value is not None for value in full):
            raise ValueError("profile rejects unexpected full-line payload")
        if config.full_line and any(value is None for value in full):
            raise ValueError("full-line profile requires explicit complete line payload")
        if config.full_line and (torch.any(self.records[:, :, 6] != 0) or torch.any(self.query[:, 6:8] != 0)):
            raise ValueError("full-line profile forbids identifier/revision/deadline semantic features")
        if all(value is not None for value in full):
            self._validate_lines(self.record_line_tokens, self.record_line_mask, (b, r), "record")
            self._validate_lines(self.query_line_tokens, self.query_line_mask, (b, 3), "query")
            if torch.any(self.record_line_mask & ~self.record_mask[:, :, None]):
                raise ValueError("inactive record carries live line tokens")
            relations = self.record_relations
            if relations.dtype != torch.int64 or relations.shape != (b, r, 2) or torch.any((relations < -1) | (relations >= r)):
                raise ValueError("record local relationship shape/index")
            if torch.any((relations >= 0) & ~self.record_mask[:, :, None]):
                raise ValueError("inactive record carries relationships")
            # Admission runs outside exported arithmetic; inspect each bounded
            # DAG in Python so cycles cannot leak into the private role graph.
            for row in range(b):
                refs = relations[row].tolist()
                active = self.record_mask[row].tolist()
                visiting, complete = set(), set()
                def visit(index):
                    if index in visiting: raise ValueError("record relationship cycle/self reference")
                    if index in complete: return
                    visiting.add(index)
                    for parent in refs[index]:
                        if parent >= 0:
                            if not active[parent]: raise ValueError("relationship references masked record")
                            visit(parent)
                    visiting.remove(index)
                    complete.add(index)
                for index in range(r):
                    if active[index]: visit(index)
        devices = {value.device for value in self.__dict__.values() if value is not None}
        if len(devices) != 1:
            raise ValueError("input tensors are split across devices")

    @staticmethod
    def _validate_lines(tokens, mask, groups, label):
        if tokens.dtype != torch.int64 or tokens.shape != (*groups, MAX_LINE_PLIES, 3) or mask.dtype != torch.bool or mask.shape != (*groups, MAX_LINE_PLIES):
            raise ValueError(label + " line shape/dtype")
        if torch.any(mask[..., 1:] & ~mask[..., :-1]):
            raise ValueError(label + " line mask must be a contiguous prefix")
        active = tokens[mask]
        if torch.any((active[:, :2] < 0) | (active[:, :2] > 63)) or torch.any((active[:, 2] < 0) | (active[:, 2] > 4)) or torch.any(active[:, 0] == active[:, 1]):
            raise ValueError(label + " line move code")

    def public_args(self, config=None):
        args = self.board, self.metadata, self.records, self.record_mask
        if (config.full_line if config is not None else self.record_line_tokens is not None):
            return (*args, self.record_line_tokens, self.record_line_mask)
        return args

    def role_args(self, memory, config=None):
        args = (*memory, self.candidates, self.candidate_mask, self.divergence_features, self.divergence_mask, self.query)
        if (config.full_line if config is not None else self.query_line_tokens is not None):
            return (*args, self.query_line_tokens, self.query_line_mask, self.record_relations)
        return args


@dataclass(frozen=True)
class VirtualReviewTensorInput:
    anchor_board: torch.Tensor
    anchor_metadata: torch.Tensor
    anchor_mask: torch.Tensor
    records: torch.Tensor
    record_anchor: torch.Tensor
    record_line_tokens: torch.Tensor
    record_line_mask: torch.Tensor
    record_mask: torch.Tensor
    candidates: torch.Tensor
    candidate_mask: torch.Tensor
    divergence_features: torch.Tensor
    divergence_mask: torch.Tensor
    query: torch.Tensor
    query_anchor: torch.Tensor
    query_line_tokens: torch.Tensor
    query_line_mask: torch.Tensor
    record_relations: torch.Tensor

    @classmethod
    def from_payload(cls, value, device="cpu"):
        from .virtual_review import prepare_payload
        prepared = prepare_payload(value)
        integer = {"anchor_board", "record_anchor", "record_line_tokens", "candidates", "query_anchor", "query_line_tokens", "record_relations"}
        boolean = {"anchor_mask", "record_line_mask", "record_mask", "candidate_mask", "divergence_mask", "query_line_mask"}
        return cls(**{name: torch.tensor(data, dtype=torch.int64 if name in integer else torch.bool if name in boolean else torch.float32, device=device)
                      for name, data in prepared["feed"].items()})

    def validate(self, config):
        from .virtual_review import BUCKETS
        config.validate()
        if not config.virtual_review:
            raise ValueError("V3 tensors require virtual_review_v1 profile")
        b = self.anchor_board.shape[0] if self.anchor_board.ndim == 3 else 0
        a = self.anchor_board.shape[1] if b else 0
        r = self.records.shape[1] if self.records.ndim == 3 else 0
        c = self.candidates.shape[1] if self.candidates.ndim == 3 else 0
        d = self.divergence_features.shape[1] if self.divergence_features.ndim == 3 else 0
        length = self.record_line_tokens.shape[-2] if self.record_line_tokens.ndim == 4 else 0
        shapes = {"anchor_board": (b, a, 64), "anchor_metadata": (b, a, 16), "anchor_mask": (b, a),
                  "records": (b, r, 8), "record_anchor": (b, r), "record_line_tokens": (b, r, length, 3),
                  "record_line_mask": (b, r, length), "record_mask": (b, r), "candidates": (b, c, 3),
                  "candidate_mask": (b, c), "divergence_features": (b, d, 8), "divergence_mask": (b, d),
                  "query": (b, 8), "query_anchor": (b, 3), "query_line_tokens": (b, 3, length, 3),
                  "query_line_mask": (b, 3, length), "record_relations": (b, r, 2)}
        if not (b >= 1 and 1 <= a <= 4 and 1 <= r <= 64 and 1 <= c <= 256 and 1 <= d <= 128 and length in BUCKETS):
            raise ValueError("V3 tensor physical capacity/registered L bucket")
        integer = {"anchor_board", "record_anchor", "record_line_tokens", "candidates", "query_anchor", "query_line_tokens", "record_relations"}
        boolean = {"anchor_mask", "record_line_mask", "record_mask", "candidate_mask", "divergence_mask", "query_line_mask"}
        for name, expected in shapes.items():
            tensor = getattr(self, name)
            dtype = torch.int64 if name in integer else torch.bool if name in boolean else torch.float32
            if tensor.shape != expected or tensor.dtype != dtype:
                raise ValueError("V3 tensor shape/dtype: " + name)
            if dtype == torch.float32 and not torch.isfinite(tensor).all():
                raise ValueError("V3 finite FP32: " + name)
        if len({x.device for x in self.__dict__.values()}) != 1:
            raise ValueError("V3 tensors are split across devices")
        if torch.any((self.anchor_board < 0) | (self.anchor_board > 12)) or not self.anchor_mask[:, 0].all():
            raise ValueError("V3 root anchor/piece codes")
        if torch.any(self.anchor_mask[:, 1:] & ~self.anchor_mask[:, :-1]):
            raise ValueError("V3 anchor mask is not contiguous")
        for refs, active in ((self.record_anchor, self.record_mask), (self.query_anchor, torch.ones_like(self.query_anchor, dtype=torch.bool))):
            if torch.any((refs < 0) | (refs >= a)) or torch.any(active & ~torch.gather(self.anchor_mask, 1, refs)):
                raise ValueError("V3 local anchor references masked/missing anchor")
        for tokens, mask in ((self.record_line_tokens, self.record_line_mask), (self.query_line_tokens, self.query_line_mask), (self.candidates, self.candidate_mask)):
            moves = tokens[mask]
            if torch.any((moves[:, :2] < 0) | (moves[:, :2] >= 64)) or torch.any(moves[:, 0] == moves[:, 1]) or torch.any((moves[:, 2] < 0) | (moves[:, 2] > 4)):
                raise ValueError("V3 active Move16 components")
        # Candidate embeddings read all physical slots before head masking.
        if torch.any((self.candidates[..., :2] < 0) | (self.candidates[..., :2] >= 64)) or torch.any((self.candidates[..., 2] < 0) | (self.candidates[..., 2] > 4)):
            raise ValueError("V3 physical candidate lookup range")
        for mask in (self.record_line_mask, self.query_line_mask):
            if torch.any(mask[..., 1:] & ~mask[..., :-1]):
                raise ValueError("V3 line mask must be a contiguous prefix")
        if torch.any(self.record_line_mask & ~self.record_mask[:, :, None]) or torch.any(self.record_relations[~self.record_mask] != -1):
            raise ValueError("V3 inactive record carries line/relation")
        if torch.any((self.record_relations < -1) | (self.record_relations >= r)):
            raise ValueError("V3 relationship index")
        for row in range(b):
            refs, active = self.record_relations[row].tolist(), self.record_mask[row].tolist()
            visiting, complete = set(), set()
            def visit(slot):
                if slot in visiting: raise ValueError("V3 relationship cycle")
                if slot in complete: return
                visiting.add(slot)
                for target in refs[slot]:
                    if target >= 0:
                        if not active[target]: raise ValueError("V3 relationship references inactive record")
                        visit(target)
                visiting.remove(slot)
                complete.add(slot)
            for slot in range(r):
                if active[slot]: visit(slot)

    def public_args(self, config=None):
        return (self.anchor_board, self.anchor_metadata, self.anchor_mask, self.records, self.record_anchor,
                self.record_line_tokens, self.record_line_mask, self.record_mask)

    def role_args(self, memory, config=None):
        return (*memory[:3], self.candidates, self.candidate_mask, self.divergence_features, self.divergence_mask,
                self.query, memory[3], self.query_anchor, self.query_line_tokens, self.query_line_mask, self.record_relations)


def fixture_input(records=3, candidates=4, divergences=2, batch=1, profile="legacy_summary_v1"):
    """Numeric fixture only, not Rules-certified chess or training data."""
    if profile == "virtual_review_v1":
        return virtual_review_fixture_input(records, candidates, divergences, batch)
    if not 1 <= records <= 128 or not 1 <= candidates <= 256 or not 1 <= divergences <= 128 or batch < 1:
        raise ValueError("fixture shape exceeds bounded profile")
    board = (torch.arange(64, dtype=torch.int64) % 13)[None, :].repeat(batch, 1)
    moves = torch.tensor([(i % 64, (i + 8) % 64, i % 5) for i in range(candidates)], dtype=torch.int64)[None, :, :].repeat(batch, 1, 1)
    config = ModelConfig.for_profile(profile)
    result = TensorInput(board, torch.arange(16, dtype=torch.float32)[None, :].repeat(batch, 1) / 16,
                       torch.arange(batch * records * 16, dtype=torch.float32).reshape(batch, records, 16) / 127,
                       torch.ones((batch, records), dtype=torch.bool), moves,
                       torch.ones((batch, candidates), dtype=torch.bool),
                       torch.arange(batch * divergences * 8, dtype=torch.float32).reshape(batch, divergences, 8) / 31,
                       torch.ones((batch, divergences), dtype=torch.bool), torch.zeros((batch, 16), dtype=torch.float32))
    if config.full_line:
        result.records[:, :, 6] = 0
        result.query[:, 6:8] = 0
        lines = torch.zeros((batch, records, MAX_LINE_PLIES, 3), dtype=torch.int64)
        masks = torch.zeros((batch, records, MAX_LINE_PLIES), dtype=torch.bool)
        query_lines = torch.zeros((batch, 3, MAX_LINE_PLIES, 3), dtype=torch.int64)
        query_masks = torch.zeros((batch, 3, MAX_LINE_PLIES), dtype=torch.bool)
        for index in range(records):
            for ply in range(4 + index % 3):
                lines[:, index, ply] = torch.tensor([(index + ply * 7) % 64, (index + ply * 7 + 8) % 64, ply % 5])
                masks[:, index, ply] = True
        for index in range(3):
            for ply in range(index + 2):
                query_lines[:, index, ply] = torch.tensor([(index + ply * 9) % 64, (index + ply * 9 + 16) % 64, ply % 5])
                query_masks[:, index, ply] = True
        relations = torch.full((batch, records, 2), -1, dtype=torch.int64)
        if records > 1: relations[:, 1:, 0] = torch.arange(records - 1)
        result = TensorInput(**{**result.__dict__, "record_line_tokens": lines, "record_line_mask": masks,
                               "query_line_tokens": query_lines, "query_line_mask": query_masks,
                               "record_relations": relations})
    return result


def virtual_review_fixture_input(records=3, candidates=4, divergences=2, batch=1, *, bucket=8, anchors=2):
    """Numeric-only fixture; never a Rules certificate or a training sample."""
    from .virtual_review import BUCKETS, INPUT_SCHEMA
    if not 1 <= records <= 64 or not 1 <= candidates <= 256 or not 1 <= divergences <= 128 or type(batch) is not int or batch < 1 or bucket not in BUCKETS or not 1 <= anchors <= 4:
        raise ValueError("virtual fixture capacity")
    value = {"schema": INPUT_SCHEMA, "role": "critic", "anchors": [{"board": [i % 13 for i in range(64)], "metadata": [i / 16 for i in range(16)]} for _ in range(anchors)],
             "records": [{"kind": "proposal", "perspective": 1, "completed_iteration": False, "rules_terminal": False,
                          "line": [[i % 64, (i + 8) % 64, i % 5]], "anchor": i % anchors, "parent": i - 1 if i else None, "supersedes": None} for i in range(records)],
             "query": {"purpose": "reply_policy", "perspective": -1, "prefix": [], "proposal": [[12, 28, 0]], "counter": [[52, 36, 0]], "counter_present":True, "anchors": [0, 0, anchors - 1]},
             "candidates": [[i % 64, (i + 8) % 64, i % 5] for i in range(candidates)], "divergences": [[i / 31 for i in range(8)] for _ in range(divergences)], "token_budget": 512}
    result = VirtualReviewTensorInput.from_payload(value)
    tensors = {}
    for name, tensor in result.__dict__.items():
        if name in ("record_line_tokens", "query_line_tokens"):
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, bucket - tensor.shape[-2]))
        if name in ("record_line_mask", "query_line_mask"):
            tensor = torch.nn.functional.pad(tensor, (0, bucket - tensor.shape[-1]))
        tensors[name] = tensor.repeat(batch, *(1 for _ in range(tensor.ndim - 1)))
    return VirtualReviewTensorInput(**tensors)
