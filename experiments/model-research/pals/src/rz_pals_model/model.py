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

    def initial(self, query, query_line_features=None, graph_relations=None, graph_relation_mask=None):
        conditioning = self.query_projection(query)
        if self.config.full_line:
            if query_line_features is None or graph_relations is None or graph_relation_mask is None:
                raise ValueError("full-line private query/relationship tensors are required")
            conditioning = conditioning + self.query_line_projection(query_line_features.flatten(1))
            relations = self.relation_output(torch.nn.functional.silu(self.relation_hidden(graph_relations)))
            relations = (relations * graph_relation_mask[:, :, None]).sum(dim=1) / graph_relation_mask.sum(dim=1, keepdim=True).clamp_min(1)
            conditioning = conditioning + relations
        return self.initial_latent[None, :, :] + conditioning[:, None, :]

    def candidate_logits(self, move_embedding, candidates, context):
        embedded = move_embedding(candidates)
        if self.config.interaction_head:
            return self.interaction_policy_head(embedded, context).squeeze(-1)
        return self.policy_head(embedded * context[:, None, :]).squeeze(-1)

    def forward(self, reader_blocks, move_embedding, key, value, mask, candidates, candidate_mask,
                divergence_features, divergence_mask, query, query_line_features=None,
                graph_relations=None, graph_relation_mask=None):
        latent = self.initial(query, query_line_features, graph_relations, graph_relation_mask)
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
            line_encoder = FullLineEncoder()
            record_projection = nn.Linear(LINE_WIDTH, config.width, bias=False)
            if config.full_line:
                self.public_encoder.line_encoder = line_encoder
                self.public_encoder.record_line_projection = record_projection
            for expert in self.experts.values():
                interaction = CandidateInteractionHead(config.width)
                query_projection = nn.Linear(3 * LINE_WIDTH, config.width, bias=False)
                relation_hidden = nn.Linear(3 * 2 * config.kv_heads * config.head_dimension, INTERACTION_WIDTH)
                relation_output = nn.Linear(INTERACTION_WIDTH, config.width, bias=False)
                if config.interaction_head:
                    expert.interaction_policy_head = interaction
                    del expert.policy_head
                if config.full_line:
                    expert.query_line_projection = query_projection
                    expert.relation_hidden = relation_hidden
                    expert.relation_output = relation_output

    def role_graph(self, role):
        if role not in self.experts:
            raise ValueError("role absent from frozen model")
        return RoleGraph(self.reader_blocks, self.move_embedding, self.experts[role],
                         self.public_encoder.line_encoder if self.config.full_line else None)

    def warm_role_graph(self, role):
        if role not in self.experts:
            raise ValueError("role absent from frozen model")
        return WarmRoleGraph(self.reader_blocks, self.move_embedding, self.experts[role],
                             self.public_encoder.line_encoder if self.config.full_line else None)

    def without_validator(self):
        result = copy.deepcopy(self)
        if "validator" in result.experts:
            del result.experts["validator"]
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


def fixture_input(records=3, candidates=4, divergences=2, batch=1, profile="legacy_summary_v1"):
    """Numeric fixture only, not Rules-certified chess or training data."""
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
