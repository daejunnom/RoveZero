"""Shared public memory and hard-routed private experts, with real neural math.

No optimizer or training loop exists here. P/C forward cannot access V private
parameters. Board and independent record encoders are role-neutral. Decoder
self-attention recomputes latent K/V on each recurrent invocation; only immutable
public-memory K/V may be retained. All first-profile arithmetic is FP32.
"""
import copy
from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig, ROLES, TASKS


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

    def encode_records(self, records):
        b, r, _ = records.shape
        fields = self.record_projection(records.reshape(b * r, 4, 4)) + self.record_field_embedding
        mask = torch.ones((b * r, 4), dtype=torch.bool, device=records.device)
        for block in self.record_blocks:
            fields = block(fields, mask)
        return fields.mean(dim=1).reshape(b, r, self.config.width)

    def forward(self, board, metadata, records, record_mask):
        b = board.shape[0]
        square = self.piece_embedding(board) + self.square_embedding
        meta = self.metadata_projection(metadata).reshape(b, 2, self.config.width)
        board_tokens = torch.cat((square, meta), dim=1)
        board_mask = torch.ones((b, 66), dtype=torch.bool, device=board.device)
        for block in self.board_blocks:
            board_tokens = block(board_tokens, board_mask)
        record_tokens = self.encode_records(records)
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

    def forward(self, reader_blocks, move_embedding, key, value, mask, candidates, candidate_mask,
                divergence_features, divergence_mask, query):
        latent = self.initial_latent[None, :, :] + self.query_projection(query)[:, None, :]
        for _ in range(self.config.iterations):
            for index, reader in enumerate(reader_blocks):
                latent = reader(latent, key, value, mask)
                latent = latent + self.ffns[index](self.ffn_norms[index](latent))
        context = latent.mean(dim=1)
        candidate_logits = self.policy_head(move_embedding(candidates) * context[:, None, :]).squeeze(-1)
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
    def __init__(self, reader_blocks, move_embedding, expert):
        super().__init__()
        self.reader_blocks, self.move_embedding, self.expert = reader_blocks, move_embedding, expert

    def forward(self, key, value, mask, candidates, candidate_mask, divergence_features, divergence_mask, query):
        return self.expert(self.reader_blocks, self.move_embedding, key, value, mask, candidates,
                           candidate_mask, divergence_features, divergence_mask, query)


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

    def role_graph(self, role):
        if role not in self.experts:
            raise ValueError("role absent from frozen model")
        return RoleGraph(self.reader_blocks, self.move_embedding, self.experts[role])

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
        torch.manual_seed(seed)
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
        devices = {value.device for value in self.__dict__.values()}
        if len(devices) != 1:
            raise ValueError("input tensors are split across devices")

    def public_args(self):
        return self.board, self.metadata, self.records, self.record_mask

    def role_args(self, memory):
        return (*memory, self.candidates, self.candidate_mask, self.divergence_features, self.divergence_mask, self.query)


def fixture_input(records=3, candidates=4, divergences=2, batch=1):
    """Numeric fixture only, not Rules-certified chess or training data."""
    if not 1 <= records <= 128 or not 1 <= candidates <= 256 or not 1 <= divergences <= 128 or batch < 1:
        raise ValueError("fixture shape exceeds bounded profile")
    board = (torch.arange(64, dtype=torch.int64) % 13)[None, :].repeat(batch, 1)
    moves = torch.tensor([(i % 64, (i + 8) % 64, i % 5) for i in range(candidates)], dtype=torch.int64)[None, :, :].repeat(batch, 1, 1)
    return TensorInput(board, torch.arange(16, dtype=torch.float32)[None, :].repeat(batch, 1) / 16,
                       torch.arange(batch * records * 16, dtype=torch.float32).reshape(batch, records, 16) / 127,
                       torch.ones((batch, records), dtype=torch.bool), moves,
                       torch.ones((batch, candidates), dtype=torch.bool),
                       torch.arange(batch * divergences * 8, dtype=torch.float32).reshape(batch, divergences, 8) / 31,
                       torch.ones((batch, divergences), dtype=torch.bool), torch.zeros((batch, 16), dtype=torch.float32))
