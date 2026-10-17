"""Exact raw /3 submission metadata, distinct from legacy raw /1 and /2.

Rules replay, loaded native ownership and numerical re-encoding belong to the
Rust owner. These bounded joins retain the original submitted counter mapping;
they cannot create an old native lease or insert a future counter into input.
"""
from __future__ import annotations

from .generation_data import fields, sha, uint, _moves
from .rovent_native_v2 import _COMMON, _submission, _root_admission, validate_position_replay
from .rovent_submitted_input import validate_input

NATIVE_DOMAIN = "rz-rovent-raw-record/3"
REVIEW_DOMAIN = "rz-rovent-submitted-review-evidence/1"


def validate_submitted_record_metadata(snapshot, value):
    validate_input(snapshot)
    records = snapshot["selected_view"]["records"]
    if type(value) is not list or len(value) != len(records) or len(value) > 64:
        raise ValueError("raw /3 original selected metadata allocation differs")
    for index, metadata in enumerate(value):
        fields(metadata, ("local_index", "record_revision", "parent_revision", "supersedes_revision", "origin_state", "source_state",
            "source_revision", "source_line_offset", "original_first_move"), "raw /3 submitted selected-record metadata")
        if uint(metadata["local_index"], 63) != index:
            raise ValueError("raw /3 original selected metadata order differs")
        for key in ("record_revision", "origin_state", "source_state", "source_revision"):
            uint(metadata[key])
        uint(metadata["source_line_offset"], 65535)
        if metadata["source_line_offset"] + len(records[index]["moves"]) > 256:
            raise ValueError("raw /3 selected suffix exceeds its complete line bound")
        if metadata["original_first_move"] is not None:
            _moves([metadata["original_first_move"]], 1)
    # Validate every entry before following original relation references.
    for index, metadata in enumerate(value):
        for relation in ("parent", "supersedes"):
            local = records[index][relation]
            revision = metadata[relation + "_revision"]
            if (local is None) != (revision is None):
                raise ValueError("raw /3 selected relation revision nullness differs")
            if local is not None:
                uint(revision)
                if revision != value[local]["record_revision"]:
                    raise ValueError("raw /3 selected relation does not retain its original revision")
    for mapping in snapshot["attack_mapping"]:
        local = mapping["selected_counter_local_index"]
        if local is not None and value[local]["source_revision"] != mapping["selected_counter_revision"]:
            raise ValueError("raw /3 selected counter revision differs from its submitted source")
    return value


def validate_submitted_review(record, value):
    """Join original view metadata without recomputing Rules or view hashes."""
    fields(value, ("schema", "sequence", "generation", "root_state", "policy", "input_sha256", "view_seal_sha256",
        "record_scope_sha256", "source_anchors", "record_metadata"), "raw /3 original submitted ReviewView evidence")
    if value["schema"] != REVIEW_DOMAIN:
        raise ValueError("raw /3 original submitted ReviewView evidence domain required")
    snapshot, native = record["input"], record["native"]
    ticket = native["raw_submission"]["ticket"]
    for name in ("sequence", "generation", "root_state"):
        uint(value[name])
    for name in ("input_sha256", "view_seal_sha256", "record_scope_sha256"):
        sha(value[name])
    if value["sequence"] != ticket["view_sequence"] or value["view_seal_sha256"] != ticket["view_digest"]:
        raise ValueError("raw /3 original submitted view sequence/seal differs from its frozen ticket")
    root = native["raw_retirement"].get("root_admission")
    if type(root) is not dict or "store_root_state" not in root:
        raise ValueError("raw /3 submitted view requires its original Store root admission")
    if (value["generation"] != uint(native["root_generation"])
            or value["root_state"] != uint(root["store_root_state"])):
        raise ValueError("raw /3 submitted Store generation/root differs from its original root admission")
    view = snapshot["selected_view"]
    policy = fields(value["policy"], ("mode", "token_budget", "max_records", "max_anchors", "required_bundle"),
        "raw /3 submitted review selection policy")
    if (policy["mode"] not in ("current_root_only", "anchored_compact")
            or uint(policy["token_budget"], 512) not in (128, 256, 512) or policy["token_budget"] != view["token_budget"]
            or not 1 <= uint(policy["max_records"], 64) or len(view["records"]) > policy["max_records"]
            or not 1 <= uint(policy["max_anchors"], 4) or len(view["anchors"]) > policy["max_anchors"]
            or policy["mode"] == "current_root_only" and policy["max_anchors"] != 1):
        raise ValueError("raw /3 submitted policy changed original selected allocations")
    if policy["required_bundle"] is not None:
        uint(policy["required_bundle"])
    anchors = value["source_anchors"]
    if type(anchors) is not list or len(anchors) != len(view["records"]):
        raise ValueError("raw /3 selected original source replay coverage differs")
    for anchor in anchors:
        validate_position_replay(anchor)
    validate_submitted_record_metadata(snapshot, value["record_metadata"])
    return value


def validate_native_v3(manifest, record):
    """Consume only actual raw /3 bodies; never retag them as a /2 receipt."""
    from .rovent_corpus_codec import SUBMITTED_CODEC, validate_record_codec
    if validate_record_codec(manifest, record) is not SUBMITTED_CODEC:
        raise ValueError("raw /3 validator requires the explicit submitted corpus family")
    validate_input(record["input"])
    native = record["native"]
    extra = {
        "raw_submission": ("model_sha256", "encoding_sha256", "input_digest", "view_digest", "rules_replay",
            "submitted_review"),
        "raw_completion": ("submitted_sha256", "execution_epoch", "execution_sequence", "completed_nn_inputs",
            "physical_confirmed", "owner_run_invoked", "measurement_complete", "physical_fresh", "output_digest",
            "owner_service_elapsed_ns", "output"),
        "raw_retirement": ("submitted_sha256", "completed_sha256", "output_digest", "accepted_retired",
            "acceptance_kind", "root_admission", "root_admission_sha256"),
    }
    for name, keys in extra.items():
        raw = fields(native[name], (*_COMMON, *keys), "raw /3 original " + name)
        if raw["schema"] != NATIVE_DOMAIN:
            raise ValueError("mixed raw versions rejected; explicit raw /3 required")
        for key in ("sequence", "process_epoch", "game_generation", "root_generation", "request_sequence"):
            uint(raw[key])
    for key in ("execution_epoch", "execution_sequence", "completed_nn_inputs"):
        uint(native["raw_completion"][key])
    elapsed = native["raw_completion"]["owner_service_elapsed_ns"]
    if elapsed is not None:
        uint(elapsed)
    retired = native["raw_retirement"]
    if (retired["acceptance_kind"] not in ("accepted_role_step", "root_ranking_published")
            or retired["acceptance_kind"] == "root_ranking_published" and record["role"] != "proposer"):
        raise ValueError("raw /3 original acceptance kind/role differs")
    from .rovent_corpus import _native_common
    _native_common(manifest, record, NATIVE_DOMAIN)
    _submission(record, native["raw_submission"]["rules_replay"])
    _root_admission(manifest, record, retired)
    validate_submitted_review(record, native["raw_submission"]["submitted_review"])
    return native
