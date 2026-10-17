"""Explicit R auxiliary recipe/sampler, separate from P/C role samples.

This module is stdlib-only. R observations never create a proposer/critic
update, native capability, optimizer or V task label. A live trainer may use
this source only alongside an independently valid P/C update boundary.
"""
from __future__ import annotations

import copy
import math
import random

from .generation_data import digest, fields, identifier, sha, uint

RECIPE_DOMAIN = "rz-rovent-r-auxiliary-recipe/1"
SAMPLER_DOMAIN = "rz-rovent-r-auxiliary-sampler/1"
POOL_DOMAIN = "rz-rovent-r-auxiliary-pool/1"
UPDATE_DOMAIN = "rz-rovent-r-auxiliary-update/1"
R_SOURCE_DOMAIN = "rz-rovent-r-readout-record/1"
MAX_ROWS = 262144
MAX_AUX_ROWS_PER_UPDATE = 64


def validate_auxiliary_recipe(value, *, recipe=None):
    """Validate an explicitly present auxiliary object; None stays None.

    The window codec retains the distinction between an absent field and an
    explicit null. This function never adds a field or re-signs either form.
    """
    if value is None:
        return None
    fields(value, ("schema", "manifest_file_sha256", "index_file_sha256", "sampling_seed", "rows_per_pc_update",
        "weight", "normalization", "optimizer_owner", "supervision_source"), "explicit R auxiliary recipe")
    if (value["schema"] != RECIPE_DOMAIN or value["normalization"] != "valid_exact_trajectory_rows"
            or value["supervision_source"] != "r_auxiliary_only_preserve_pc_evidence"):
        raise ValueError("explicit exact-trajectory R auxiliary recipe required")
    sha(value["manifest_file_sha256"])
    sha(value["index_file_sha256"])
    uint(value["sampling_seed"], 2**32 - 1)
    if not 1 <= uint(value["rows_per_pc_update"], MAX_AUX_ROWS_PER_UPDATE):
        raise ValueError("R auxiliary rows per valid P/C update must be bounded 1..64")
    weight = value["weight"]
    try:
        finite = type(weight) in (int, float) and math.isfinite(weight)
    except OverflowError:
        finite = False
    if not finite or weight <= 0:
        raise ValueError("finite explicitly positive R auxiliary loss weight required")
    identifier(value["optimizer_owner"])
    if recipe is not None:
        owners = recipe.to_dict()["parameter_ownership"]
        if (owners["nnuec"] != value["optimizer_owner"]
                or owners["nnuec"] in {owners[name] for name in ("pc_shared", "p_private", "c_private", "v_private")}):
            raise ValueError("R auxiliary requires its registered exclusive NNUEC optimizer owner")
    return value


def validate_auxiliary_sample(value):
    fields(value, ("input_sha256", "origin_sha256", "record_sha256", "nnuec_input", "nnuec_model_sha256",
        "nnuec_weights_file_sha256", "nnuec_semantics", "nnuec_wdl", "retirement_sha256", "wdl_raw_sha256", "split"),
        "separate original R auxiliary sample")
    for name in ("input_sha256", "origin_sha256", "record_sha256", "nnuec_model_sha256", "nnuec_weights_file_sha256",
                 "retirement_sha256"):
        sha(value[name])
    identifier(value["nnuec_semantics"])
    if value["split"] != "train" or type(value["nnuec_input"]) is not dict:
        raise ValueError("only original train-split R reference input may enter auxiliary optimization")
    target = value["nnuec_wdl"]
    if target is None:
        if value["wdl_raw_sha256"] is not None:
            sha(value["wdl_raw_sha256"])
    else:
        if (type(target) is not list or len(target) != 3
                or any(type(item) not in (float, int) or type(item) is bool or item not in (0, 1) for item in target)
                or sum(target) != 1):
            raise ValueError("R auxiliary target must be an actual one-hot Rules outcome, never CP/mate")
        sha(value["wdl_raw_sha256"])
    return value


def auxiliary_pool_digest(canonical_index_sha256, samples):
    """Seal the ordered eligible pool, retaining the original index identity."""
    sha(canonical_index_sha256)
    if type(samples) not in (list, tuple) or not 1 <= len(samples) <= MAX_ROWS:
        raise ValueError("bounded nonempty original R auxiliary pool required")
    projected, records, origins = [], set(), set()
    for sample in samples:
        validate_auxiliary_sample(sample)
        record, origin = sample["record_sha256"], sample["origin_sha256"]
        if record in records or origin in origins:
            raise ValueError("duplicate R occurrence in auxiliary pool")
        records.add(record)
        origins.add(origin)
        projected.append({"source_domain": R_SOURCE_DOMAIN,
            **{name: sample[name] for name in ("input_sha256", "origin_sha256", "record_sha256")}})
    return digest(POOL_DOMAIN, {"index_sha256": canonical_index_sha256, "occurrences": projected})


class RAuxiliarySampler:
    """A separate seeded R cursor; it never impersonates a P/C role sampler."""
    def __init__(self, rows, *, seed, pool_sha256, state=None):
        if type(rows) not in (tuple, list) or not 1 <= len(rows) <= MAX_ROWS:
            raise ValueError("bounded nonempty original R row tokens required")
        self.rows = tuple(rows)
        self.seed, self.pool_sha256 = uint(seed, 2**32 - 1), sha(pool_sha256)
        self.epoch, self.cursor, self.samples = 0, 0, 0
        if state is not None:
            fields(state, ("schema", "seed", "pool_sha256", "epoch", "cursor", "samples", "permutation"), "R auxiliary sampler state")
            if state["schema"] != SAMPLER_DOMAIN or state["seed"] != seed or state["pool_sha256"] != pool_sha256:
                raise ValueError("R auxiliary sampler changed source or seed")
            self.epoch, self.cursor, self.samples = uint(state["epoch"]), uint(state["cursor"], len(rows)), uint(state["samples"])
            if self.samples != self.epoch * len(rows) + self.cursor:
                raise ValueError("R auxiliary sampler lost exact consumed position")
        self.permutation = self._order()
        if state is not None and (type(state["permutation"]) is not list
                or any(type(index) is not int for index in state["permutation"])
                or state["permutation"] != self.permutation):
            raise ValueError("R auxiliary seeded permutation differs from checkpoint")

    def _order(self):
        permutation = list(range(len(self.rows)))
        random.Random(digest(SAMPLER_DOMAIN, [self.seed, self.pool_sha256, self.epoch])).shuffle(permutation)
        return permutation

    def take(self, count):
        count = uint(count, MAX_AUX_ROWS_PER_UPDATE)
        result = []
        for _ in range(count):
            if self.cursor == len(self.rows):
                self.epoch, self.cursor = self.epoch + 1, 0
                self.permutation = self._order()
            result.append(self.rows[self.permutation[self.cursor]])
            self.cursor += 1
            self.samples += 1
        return result

    def state(self):
        return {"schema": SAMPLER_DOMAIN, "seed": self.seed, "pool_sha256": self.pool_sha256,
            "epoch": self.epoch, "cursor": self.cursor, "samples": self.samples, "permutation": list(self.permutation)}


def update_projection(auxiliary_recipe, samples, *, pc_valid_rows):
    """Project original R seals only after a real P/C target is available.

    The projection is evidence to join with the parent's actual update owner,
    not a completed update or training authorization. Unknown R targets remain
    observations and never advance NNUEC moments on their own.
    """
    registration = validate_auxiliary_recipe(auxiliary_recipe)
    if registration is None:
        raise ValueError("disabled R auxiliary stream cannot publish consumed rows")
    if not 1 <= uint(pc_valid_rows, 4096):
        raise ValueError("R auxiliary stream cannot create an update without a valid P/C target")
    if type(samples) not in (tuple, list) or len(samples) != registration["rows_per_pc_update"]:
        raise ValueError("R auxiliary sample count differs from the registered P/C boundary")
    rows, valid, unique = [], 0, set()
    for sample in samples:
        validate_auxiliary_sample(sample)
        target_known = sample["nnuec_wdl"] is not None
        valid += target_known
        if target_known:
            unique.add(sample["input_sha256"])
        rows.append({name: copy.deepcopy(sample[name]) for name in ("input_sha256", "origin_sha256", "record_sha256",
            "retirement_sha256", "wdl_raw_sha256")})
    return {"schema": UPDATE_DOMAIN, "manifest_file_sha256": registration["manifest_file_sha256"],
        "index_file_sha256": registration["index_file_sha256"], "optimizer_owner": registration["optimizer_owner"],
        "occurrences": rows, "valid_rows": valid, "unique_valid_inputs": len(unique),
        "pc_valid_rows": pc_valid_rows, "normalization": registration["normalization"],
        "supervision_source": registration["supervision_source"],
        "pc_nnuec_evidence_preserved_not_optimized": True,
        "execution_authorized": False}
