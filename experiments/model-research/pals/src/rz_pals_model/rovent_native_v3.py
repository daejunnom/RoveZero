"""Exact raw /3 submission metadata, distinct from legacy raw /1 and /2.

Rules replay, loaded native ownership and numerical re-encoding belong to the
Rust owner. These bounded joins retain the original submitted counter mapping;
they cannot create an old native lease or insert a future counter into input.
"""
from __future__ import annotations

from .generation_data import fields, uint, _moves
from .rovent_native_v2 import _COMMON, _submission, _root_admission

NATIVE_DOMAIN = "rz-rovent-raw-record/3"


def validate_submitted_record_metadata(snapshot, value):
    records = snapshot["selected_view"]["records"]
    if type(value) is not list or len(value) != len(records) or len(value) > 64:
        raise ValueError("raw /3 original selected metadata allocation differs")
    for index, metadata in enumerate(value):
        fields(metadata, ("local_index", "record_revision", "parent_revision", "supersedes_revision", "source_state",
            "source_revision", "source_line_offset", "original_first_move"), "raw /3 submitted selected-record metadata")
        if uint(metadata["local_index"], 63) != index:
            raise ValueError("raw /3 original selected metadata order differs")
        for key in ("record_revision", "source_state", "source_revision"):
            uint(metadata[key])
        uint(metadata["source_line_offset"], 65535)
        if metadata["source_line_offset"] + len(records[index]["moves"]) > 256:
            raise ValueError("raw /3 selected suffix exceeds its complete line bound")
        if metadata["original_first_move"] is not None:
            _moves([metadata["original_first_move"]], 1)
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


def validate_native_v3(manifest, record):
    """Consume only actual raw /3 bodies; never retag them as a /2 receipt."""
    native = record["native"]
    extra = {
        "raw_submission": ("model_sha256", "encoding_sha256", "input_digest", "view_digest", "rules_replay",
            "submitted_record_metadata"),
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
    validate_submitted_record_metadata(record["input"], native["raw_submission"]["submitted_record_metadata"])
    _root_admission(manifest, record, retired)
    return native
