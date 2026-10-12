"""Dependency-free configuration and declared-shape matrix FLOPs accounting."""
from dataclasses import asdict, dataclass

SCHEMA = "rovezero.pals-model.v1"
MODEL_SEMANTICS_V2 = "rovezero.pals-model-semantics.v2"
ENCODING_V2 = "rovezero.pals-board-records.v2"
PROFILES = ("legacy_summary_v1", "full_line_v2", "interaction_head_v2", "full_line_interaction_v2")
DEFAULT_INIT_PROFILE = "full_line_interaction_v2"
MAX_LINE_PLIES = 256
LINE_WIDTH = 64
INTERACTION_WIDTH = 128
ROLES = ("proposer", "critic", "validator")
TASKS = ("defend_response", "attack_repair", "widen_responses", "lower_selectivity", "resume_task", "cross_profile_recheck", "defer")


@dataclass(frozen=True)
class ModelConfig:
    width: int = 384
    query_heads: int = 6
    kv_heads: int = 2
    head_dimension: int = 64
    board_blocks: int = 2
    record_blocks: int = 1
    record_fields: int = 4
    latent_slots: int = 16
    recurrent_blocks: int = 2
    iterations: int = 2
    ffn_width: int = 1024
    max_records: int = 128
    max_candidates: int = 256
    max_divergences: int = 128
    profile: str = "legacy_summary_v1"

    def validate(self):
        if any(type(value) is not int for key, value in asdict(self).items() if key != "profile") or self.profile not in PROFILES or self != ModelConfig(profile=self.profile):
            raise ValueError("requires registered width384 configuration/profile; no automatic resizing")

    @classmethod
    def baseline(cls):
        """The original configuration and serialization remain LegacySummaryV1."""
        return cls()

    @classmethod
    def for_profile(cls, profile):
        result = cls(profile=profile)
        result.validate()
        return result

    @property
    def full_line(self):
        return self.profile in ("full_line_v2", "full_line_interaction_v2")

    @property
    def interaction_head(self):
        return self.profile in ("interaction_head_v2", "full_line_interaction_v2")

    @property
    def model_semantics(self):
        return SCHEMA if self.profile == "legacy_summary_v1" else MODEL_SEMANTICS_V2

    @property
    def encoding(self):
        return "rovezero.pals-board-records.v1" if self.profile == "legacy_summary_v1" else ENCODING_V2

    def to_dict(self):
        result = asdict(self)
        if self.profile == "legacy_summary_v1":
            del result["profile"]
        return result


def public_tensor_names(config):
    names = ["board", "metadata", "records", "record_mask"]
    if config.full_line:
        names += ["record_line_tokens", "record_line_mask"]
    return names


def role_tensor_names(config):
    """Complete Torch/feed argument order; unused heads may prune ONNX inputs."""
    names = ["memory_key", "memory_value", "memory_mask", "candidates", "candidate_mask",
             "divergence_features", "divergence_mask", "query"]
    if config.full_line:
        names += ["query_line_tokens", "query_line_mask", "record_relations"]
    return names


def matmul_flops(config, role, records, candidates, divergences=0, batch=1):
    """FMA=2 analytic GEMM and attention products, not a runtime/VRAM claim.

    Bias, RMSNorm, softmax, activation, mask, pooling, lookup, data movement and
    CPU search are excluded explicitly. Backward is not measured or claimed.
    Cold public memory and one role invocation are separate to avoid recounting
    reusable memory at every P/C transition.
    """
    config.validate()
    if role not in ROLES or not 0 <= records <= config.max_records or not 0 <= candidates <= config.max_candidates or not 0 <= divergences <= config.max_divergences or batch < 1:
        raise ValueError("FLOPs shape/role exceeds registered limits")
    # Zero semantic items require a masked physical padding slot in ONNX. Its
    # matrix products still run and must not disappear from compute accounting.
    physical_records, physical_candidates, physical_divergences = max(records, 1), max(candidates, 1), max(divergences, 1)
    w, kv, h, n = config.width, config.kv_heads * config.head_dimension, config.ffn_width, config.latent_slots
    s = 66 + physical_records
    self_attn = lambda tokens: 2 * tokens * w * (2 * w + 2 * kv) + 4 * tokens * tokens * w
    ffn = lambda tokens: 6 * tokens * w * h
    public = 2 * 16 * 2 * w + config.board_blocks * (self_attn(66) + ffn(66))
    public += physical_records * (2 * 16 * w + config.record_blocks * (self_attn(4) + ffn(4))) + 4 * s * w * kv
    cross = 4 * n * w * w + 4 * n * s * w
    private = 2 * 16 * w + config.recurrent_blocks * config.iterations * (cross + self_attn(n) + ffn(n))
    private += 2 * physical_candidates * 3 * w * w + 2 * physical_candidates * w + 2 * w * 3
    if role == "critic":
        private += 2 * physical_divergences * 8 * w + 2 * physical_divergences * w
    if role == "validator":
        private += 2 * w * len(TASKS)
    if config.full_line:
        # Registered physical 256-ply tensors execute both temporal Conv blocks
        # even when semantically empty; embedding/pooling stays excluded.
        line = 2 * MAX_LINE_PLIES * (3 * LINE_WIDTH * LINE_WIDTH + 2 * 3 * LINE_WIDTH * LINE_WIDTH + LINE_WIDTH)
        public += physical_records * (line + 2 * LINE_WIDTH * w)
        private += 3 * line + 2 * 3 * LINE_WIDTH * w
        private += physical_records * (2 * 3 * 2 * kv * INTERACTION_WIDTH + 2 * INTERACTION_WIDTH * w)
    if config.interaction_head:
        private += physical_candidates * (2 * 2 * w * INTERACTION_WIDTH + 2 * INTERACTION_WIDTH - 2 * w)
    return {"method": "analytic_matrix_products_fma_2", "batch": batch, "role": role,
            "physical_records": physical_records, "physical_candidates": physical_candidates, "physical_divergences": physical_divergences if role == "critic" else 0,
            "public_encode_matmul_flops": batch * public,
            "role_forward_matmul_flops": batch * private,
            "cold_forward_matmul_flops": batch * (public + private),
            "backward_flops": "not_measured", "optimizer_steps": 0,
            "excluded_operations": ["bias", "rmsnorm", "softmax", "silu", "elementwise", "pooling", "embedding_lookup", "transfer", "cpu_search"]}
