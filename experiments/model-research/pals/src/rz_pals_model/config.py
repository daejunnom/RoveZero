"""Dependency-free configuration and declared-shape matrix FLOPs accounting."""
from dataclasses import asdict, dataclass

SCHEMA = "rovezero.pals-model.v1"
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

    def validate(self):
        if self != ModelConfig():
            raise ValueError("v1 requires registered width384 configuration; no automatic resizing")

    def to_dict(self):
        return asdict(self)


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
    return {"method": "analytic_matrix_products_fma_2", "batch": batch, "role": role,
            "physical_records": physical_records, "physical_candidates": physical_candidates, "physical_divergences": physical_divergences if role == "critic" else 0,
            "public_encode_matmul_flops": batch * public,
            "role_forward_matmul_flops": batch * private,
            "cold_forward_matmul_flops": batch * (public + private),
            "backward_flops": "not_measured", "optimizer_steps": 0,
            "excluded_operations": ["bias", "rmsnorm", "softmax", "silu", "elementwise", "pooling", "embedding_lookup", "transfer", "cpu_search"]}
