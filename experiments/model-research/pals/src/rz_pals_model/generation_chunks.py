"""Pinned JSONL generation corpus with bounded rows and a disposable disk index.

These are logical byte/entry limits, not a promise about Python process RSS.
No neural framework is imported, no whole-corpus fallback exists, and closing
the reader removes only its owned transient index, never collected evidence.
"""
import copy
import hashlib
import json
import math
import os
import sqlite3
import stat
import struct
import tempfile
from collections import Counter, OrderedDict
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from .generation_data import (ACTUAL_SOURCE, GenerationDataset, admit_phase, digest, fields,
                              identifier, sha, strict_json, trajectory_digest, uint, validate_record,
                              _count_target_coverage, _admit_coverage, validate_evidence_receipt, _moves, TASKS,
                              TEACHER_INPUT_BINDING_DOMAIN)

MANIFEST_DOMAIN = "rz-pals-generation-chunk-manifest/1"
MAX_ROW_BYTES = 1024 * 1024
MAX_CHUNK_BYTES = 8 * 1024 * 1024
MAX_CHUNK_ROWS = 128
MAX_CORPUS_BYTES = 18 * 1024**3
MAX_INDEX_BYTES = 512 * 1024**2
MAX_RECORDS = 262144
MAX_EVIDENCE = 1048576
MAX_TRAJECTORIES = 8192
MAX_CHUNKS = 32768
MAX_CACHE_BYTES = 16 * 1024**2
SPLITS = ("train", "validation", "holdout")
KINDS = ("inputs", "evidence", "trajectories", "records")
NATIVE_DOMAINS = ("rz-pals-generation-submission-evidence/1", "rz-pals-generation-native-completion-evidence/1",
                  "rz-pals-generation-native-retirement-evidence/1")
ACTION_AUTHORITY_DOMAIN = "rz-pals-generation-action-authority-evidence/1"
HEAD_EVIDENCE_DOMAIN = "rz-pals-generation-head-evidence/1"
HEADS = ("policy", "wdl", "divergence", "task")
RAW_ACTION_COST_SCOPE = "input_prepare_to_actual_root_physical_fence"
EXCLUSIVE_ACTION_COST_SCOPE = "exclusive_v_decision_and_single_action_with_required_drain"
ACTION_GROUP_DOMAIN = "rz-pals-generation-frozen-action-group-evidence/1"


def _frozen_action_group(value):
    """Check a preregistered alias without changing any original receipt.

    This shape/digest check does not grant supervision. Each original member
    still needs its accepted Fresh native chain and private frozen cost proof.
    """
    fields(value, ("schema", "input_sha256", "group_sha256", "semantic_input_sha256", "frozen_context_sha256",
                   "frozen_pc_epoch", "frozen_pc_weights_sha256", "registered_actions", "action_schedule_sha256",
                   "selection_provenance", "members"), "frozen actual action group")
    if (value["schema"] != ACTION_GROUP_DOMAIN
            or value["selection_provenance"] != "registered-supported-enumeration-v1"):
        raise ValueError("actual action group needs preregistered supported enumeration")
    for name in ("input_sha256", "group_sha256", "semantic_input_sha256", "frozen_context_sha256",
                 "frozen_pc_weights_sha256", "action_schedule_sha256"):
        sha(value[name])
    uint(value["frozen_pc_epoch"])
    actions, members = value["registered_actions"], value["members"]
    if (type(actions) is not list or not 1 <= len(actions) <= len(TASKS)
            or any(type(index) is not int or not 0 <= index < len(TASKS) for index in actions)
            or actions != sorted(set(actions)) or type(members) is not list or len(members) != len(actions)):
        raise ValueError("actual action group registration/order/capacity mismatch")
    schedule = digest("rz-pals-generation-action-enumeration/1", actions)
    group = digest("rz-pals-generation-frozen-action-group/1", [value["semantic_input_sha256"],
                   value["frozen_context_sha256"], value["frozen_pc_epoch"], value["frozen_pc_weights_sha256"], actions, schedule])
    if schedule != value["action_schedule_sha256"] or group != value["group_sha256"]:
        raise ValueError("actual action group preregistration digest mismatch")
    seen_inputs, seen_causal, seen_authority = set(), set(), set()
    for index, member in zip(actions, members):
        fields(member, ("action", "index", "actual_input_sha256", "causal_receipt_sha256", "action_authority_sha256"),
               "frozen original action member")
        identities = [sha(member[name]) for name in ("actual_input_sha256", "causal_receipt_sha256", "action_authority_sha256")]
        if (type(member["index"]) is not int or member["index"] != index or member["action"] != TASKS[index]
                or any(identity in seen for identity, seen in zip(identities, (seen_inputs, seen_causal, seen_authority)))):
            raise ValueError("duplicate/reordered/renamed frozen original action member")
        for identity, seen in zip(identities, (seen_inputs, seen_causal, seen_authority)):
            seen.add(identity)
    if value["input_sha256"] not in seen_inputs:
        raise ValueError("actual action group representative is not an original member")
    return value


def _head_receipt(value):
    fields(value, ("schema", "input_sha256", "head", "target"), "streamed future head evidence")
    if (value["schema"] != HEAD_EVIDENCE_DOMAIN or value["head"] not in HEADS
            or type(value["target"]) is not dict or "raw_evidence_sha256" in value["target"]):
        raise ValueError("streamed future head domain/content mismatch")
    sha(value["input_sha256"])
    return value


def _action_authority(value):
    """Validate opaque actual-owner proof shape without admitting its target."""
    fields(value, ("schema", "input_sha256", "generation_view_sha256", "ticket", "action", "execution_id",
                   "native_view_content_sha256", "actual_native_input_sha256", "action_support_mask",
                   "input_store_observation_sequence", "original_first_move", "wall_cost_ns", "root_native_cpu_spent",
                   "training_wall_cost_ns", "wall_cost_scope", "measured_wall_cost_scope",
                   "frozen_context_sha256", "frozen_pc_epoch", "frozen_pc_weights_sha256",
                   "root_completed_nn_inputs", "executions", "role_requests", "witnesses", "causal_receipt_sha256"),
           "actual action authority")
    if value["schema"] != ACTION_AUTHORITY_DOMAIN or value["action"] not in TASKS:
        raise ValueError("actual action authority domain/action mismatch")
    _native_ticket(value["ticket"])
    for name in ("input_sha256", "generation_view_sha256", "native_view_content_sha256", "actual_native_input_sha256",
                 "causal_receipt_sha256"):
        sha(value[name])
    identifier(value["execution_id"])
    for name in ("input_store_observation_sequence", "wall_cost_ns", "root_native_cpu_spent", "root_completed_nn_inputs"):
        uint(value[name])
    training_cost = value["training_wall_cost_ns"]
    if value["frozen_context_sha256"] is not None:
        sha(value["frozen_context_sha256"])
    if value["frozen_pc_epoch"] is None or value["frozen_pc_weights_sha256"] is None:
        if value["frozen_pc_epoch"] is not None or value["frozen_pc_weights_sha256"] is not None:
            raise ValueError("actual action freeze requires a complete epoch/weights pair")
    else:
        uint(value["frozen_pc_epoch"])
        sha(value["frozen_pc_weights_sha256"])
    if training_cost is not None:
        uint(training_cost)
        if any(value[name] is None for name in ("frozen_context_sha256", "frozen_pc_epoch", "frozen_pc_weights_sha256")):
            raise ValueError("known exclusive action cost requires its private frozen context and preregistered PC state")
    if (value["measured_wall_cost_scope"] != RAW_ACTION_COST_SCOPE
            or value["wall_cost_scope"] != (RAW_ACTION_COST_SCOPE if training_cost is None else EXCLUSIVE_ACTION_COST_SCOPE)):
        raise ValueError("actual action cost needs its private exclusive scope or remains unknown for training")
    _moves([value["original_first_move"]])
    mask = value["action_support_mask"]
    if type(mask) is not list or len(mask) != len(TASKS) or any(type(bit) is not bool for bit in mask) or not mask[TASKS.index(value["action"])]:
        raise ValueError("actual action authority lacks original supported action mask")
    count = 0
    for name in ("executions", "role_requests"):
        identities = value[name]
        if type(identities) is not list or len(identities) > 4096:
            raise ValueError("actual action execution list capacity")
        seen = set()
        for identity in identities:
            fields(identity, ("epoch", "sequence"), "actual action execution identity")
            identity_pair = (uint(identity["epoch"]), uint(identity["sequence"]))
            if identity_pair in seen or identity["epoch"] != value["ticket"]["epoch"]:
                raise ValueError("duplicate/foreign epoch actual action execution")
            seen.add(identity_pair)
        count += len(identities)
    if count > 4096 or value["action"] != "defer" and not count:
        raise ValueError("completed actual action requires bounded actual execution work")
    witnesses = value["witnesses"]
    if type(witnesses) is not list or len(witnesses) > 64 or value["action"] == "defer" and witnesses:
        raise ValueError("actual action witness capacity or witness-free Defer mismatch")
    seen = set()
    for witness in witnesses:
        fields(witness, ("witness_id", "kind", "observed_sequence", "store_observation_sequence", "store_generation",
                         "packed_line", "semantic_identity_sha256", "store_commit_sha256"), "actual semantic witness")
        identity = sha(witness["semantic_identity_sha256"])
        sha(witness["store_commit_sha256"])
        uint(witness["observed_sequence"])
        uint(witness["store_generation"])
        ordinal = uint(witness["store_observation_sequence"])
        line = _moves(witness["packed_line"])
        if (witness["kind"] not in ("counter", "repair") or witness["witness_id"] != "witness-" + identity
                or identity in seen or ordinal < value["input_store_observation_sequence"]
                or not line or line[0] != value["original_first_move"]):
            raise ValueError("old/duplicate/misidentified actual semantic witness")
        seen.add(identity)
    return value


def _action_receipt(receipts, causal_sha256):
    if hasattr(receipts, "action_authority"):
        return receipts.action_authority(causal_sha256)
    # Pure bounded dictionaries are used by independent contract tests only;
    # actual corpora query the indexed causal key, never scan full evidence.
    found = [value for value in receipts.values() if value.get("schema") == ACTION_AUTHORITY_DOMAIN
             and value.get("causal_receipt_sha256") == causal_sha256]
    if len(found) != 1:
        raise ValueError("actual V outcome requires exactly one opaque action authority")
    return found[0]


def _native_one_action_authority(record, submitted, outcome, authority):
    snapshot, ticket = record["input"], submitted["ticket"]
    authority = _action_authority(authority)
    expected_execution = f"validator-action-{ticket['epoch']}-{ticket['request_sequence']}-{TASKS.index(outcome['action'])}"
    if (snapshot["role"] != "validator" or authority["input_sha256"] != record["input_sha256"]
                or authority["generation_view_sha256"] != snapshot["view_sha256"] or authority["ticket"] != ticket
                or authority["native_view_content_sha256"] != submitted["source_view_content_sha256"]
                or authority["actual_native_input_sha256"] != ticket["representation_input_sha256"]
                or authority["action_support_mask"] != snapshot["action_support_mask"]
                or authority["action"] != outcome["action"] or authority["execution_id"] != outcome["execution_id"]
                or outcome["execution_id"] != expected_execution or outcome["completed"] is not True
                or authority["training_wall_cost_ns"] != outcome["wall_cost_ns"]):
        raise ValueError("actual V action authority differs from original native input/outcome")
    if authority["training_wall_cost_ns"] is not None and authority["frozen_pc_epoch"] != snapshot["model_epoch"]:
        raise ValueError("actual V frozen epoch differs from its original input")
    raw = authority["witnesses"]
    if len(raw) != len(outcome["new_witnesses"]):
        raise ValueError("actual V action changed its unique witness set")
    root_semantic = bytes.fromhex(snapshot["selected_view"]["anchors"][0]["rules_state_sha256"])
    for witness, raw_witness in zip(outcome["new_witnesses"], raw):
        if (any(witness[name] != raw_witness[name] for name in ("witness_id", "kind", "observed_sequence"))
                or witness["observed_sequence"] <= snapshot["capture_sequence"]):
            raise ValueError("actual witness chronological publication differs from opaque Store proof")
        # Independent framing check of the owner-issued identity, not a
        # duplicate Rules implementation or a Store ordinal -> clock conversion.
        hasher = hashlib.sha256(b"rz-pals-action-unique-witness/1\0" + root_semantic
                                    + bytes((0 if raw_witness["kind"] == "counter" else 1,)))
        for move in raw_witness["packed_line"]:
            hasher.update(move.to_bytes(2, "little"))
        if hasher.hexdigest() != raw_witness["semantic_identity_sha256"]:
            raise ValueError("actual witness identity does not bind exact root/kind/full line")


def _native_action_authorities(record, submitted, receipts):
    target = record["future_targets"]["task"]
    if target is None:
        return
    group = receipts.action_group(record["input_sha256"]) if hasattr(receipts, "action_group") else None
    if group is not None:
        _native_frozen_action_group(record, submitted, receipts, group)
        return
    frozen = None
    for outcome in target["outcomes"]:
        authority = _action_authority(_action_receipt(receipts, outcome["causal_receipt_sha256"]))
        _native_one_action_authority(record, submitted, outcome, authority)
        if authority["training_wall_cost_ns"] is not None:
            observed = (authority["frozen_context_sha256"], authority["frozen_pc_epoch"], authority["frozen_pc_weights_sha256"])
            if frozen is not None and frozen != observed:
                raise ValueError("actual V outcomes changed their private frozen comparison context")
            if hasattr(receipts, "owner") and observed[2] != receipts.owner._manifest["producer"]["model_sha256"]:
                raise ValueError("actual V private PC weights differ from the registered frozen producer")
            frozen = observed


def _native_frozen_action_group(record, submitted, receipts, group):
    group = _frozen_action_group(group)
    snapshot, target = record["input"], record["future_targets"]["task"]
    indices = group["registered_actions"]
    if (snapshot["role"] != "validator" or group["input_sha256"] != record["input_sha256"]
            or [TASKS.index(outcome["action"]) for outcome in target["outcomes"]] != indices
            or any(not snapshot["action_support_mask"][index] for index in indices)
            or group["semantic_input_sha256"] != semantic_sample_seal(record, submitted)
            or group["frozen_pc_epoch"] != snapshot["model_epoch"]
            or group["frozen_pc_weights_sha256"] != receipts.owner._manifest["producer"]["model_sha256"]):
        raise ValueError("frozen action group differs from its preregistered original semantic input/PC state")
    for member, outcome in zip(group["members"], target["outcomes"]):
        original = receipts.source_record(member["actual_input_sha256"])
        native = _native_chain(original, receipts, receipts.accepted_retirement(member["actual_input_sha256"]),
                               with_details=True, validate_actions=False)
        source, origin = original["input"], native["submission"]
        if (native["physical_fresh"] is not True or native["semantic_sample_sha256"] != group["semantic_input_sha256"]
                or source["role"] != "validator" or source["model_epoch"] != group["frozen_pc_epoch"]
                or origin["ticket"]["frozen_epoch"] != group["frozen_pc_epoch"]
                or origin["native_backend_sha256"] != submitted["native_backend_sha256"]
                or any(source[name] != snapshot[name] for name in ("game_id", "opening_id", "context_epoch", "rules_anchor", "action_support_mask"))
                or source["selected_view"]["anchors"][0] != snapshot["selected_view"]["anchors"][0]
                or target["observed_sequence"] <= source["capture_sequence"]
                or member["causal_receipt_sha256"] != outcome["causal_receipt_sha256"]):
            raise ValueError("frozen action group member changed model/tensor/history/context or actual origin")
        authority = _action_authority(receipts.get(member["action_authority_sha256"]))
        if (authority["causal_receipt_sha256"] != member["causal_receipt_sha256"]
                or authority["frozen_context_sha256"] != group["frozen_context_sha256"]
                or authority["frozen_pc_epoch"] != group["frozen_pc_epoch"]
                or authority["frozen_pc_weights_sha256"] != group["frozen_pc_weights_sha256"]
                or authority["training_wall_cost_ns"] is None):
            raise ValueError("frozen action group lacks an original private exclusive frozen cost proof")
        _native_one_action_authority(original, origin, outcome, authority)


def _native_ticket(ticket):
    """Validate the same bounded causal identity for accepted and failed work."""
    fields(ticket, ("epoch", "request_sequence", "game", "root", "question_sha256", "source_view_sha256", "view_sequence",
                    "context_policy_sha256", "representation_input_sha256", "representation_history_sha256",
                    "representation_model_sha256", "representation_encoding_sha256", "frozen_epoch", "record_revision",
                    "deadline_tick"), "native causal ticket")
    for name in ("epoch", "request_sequence", "game", "root", "view_sequence", "frozen_epoch", "record_revision", "deadline_tick"):
        uint(ticket[name])
    for name in ("question_sha256", "source_view_sha256", "context_policy_sha256", "representation_input_sha256",
                 "representation_history_sha256", "representation_model_sha256", "representation_encoding_sha256"):
        sha(ticket[name])
    return ticket


def _native_retirement(retired):
    """Retain only a complete, unaccepted pre-submission null tuple as raw data."""
    fields(retired, ("schema", "input_sha256", "submission_sha256", "completion_sha256", "ticket", "accepted"), "native retirement")
    if retired["schema"] != NATIVE_DOMAINS[2] or type(retired["accepted"]) is not bool:
        raise ValueError("native retirement domain/acceptance type mismatch")
    _native_ticket(retired["ticket"])
    identities = (retired["input_sha256"], retired["submission_sha256"], retired["completion_sha256"])
    if any(identity is None for identity in identities):
        if all(identity is None for identity in identities) and retired["accepted"] is False:
            return False
        raise ValueError("native retirement requires a complete unaccepted null tuple or complete receipt identities")
    for identity in identities:
        sha(identity)
    return True


def semantic_sample_seal(record, submission):
    """Identify actual model input independently of publication/request IDs.

    This pure identity does not admit evidence: the caller must first validate
    the native submission/completion/retirement chain for the publication row.
    """
    validate_record(record)
    snapshot = record["input"]
    ticket = _native_ticket(submission["ticket"])
    return digest("rz-pals-generation-semantic-sample/1", [
        snapshot["role"], sha(submission["native_model_epoch_sha256"]),
        sha(submission["native_input_sha256"]), sha(ticket["representation_history_sha256"]),
        snapshot["action_support_mask"] if snapshot["role"] == "validator" else None,
    ])


def _native_chain(record, receipts, retirement_sha256, *, with_details=False, validate_actions=True):
    """Join original submission, physical completion and accepted retirement."""
    input_seal = record["input_sha256"]
    retired = receipts.get(retirement_sha256)
    if not _native_retirement(retired):
        raise ValueError("pre-submission native retirement cannot admit a generation row")
    completed = receipts.get(sha(retired["completion_sha256"]))
    fields(completed, ("schema", "input_sha256", "submission_sha256", "ticket", "physical_completion", "execution",
                       "completed_nn_inputs", "measurement_complete", "physical_fresh", "invocation_key_sha256",
                       "accepted_seed_sha256", "output", "admissible_actual_nn"), "native completion")
    submitted = receipts.get(sha(retired["submission_sha256"]))
    fields(submitted, ("schema", "input_sha256", "ticket", "source_view_seal_sha256", "physical_nn_admitted",
                       "native_input_sha256", "native_model_epoch_sha256", "native_backend_sha256",
                       "native_anchor_history_sha256", "generation_view_sha256", "game_id", "opening_id", "role",
                       "model_epoch", "context_epoch", "capture_sequence", "worker_id", "job_id", "slice_id",
                       "actual_root_anchor", "query_anchor", "source_view_content_sha256", "native_endpoint_history_sha256"), "native submission")
    for receipt, domain in zip((submitted, completed, retired), NATIVE_DOMAINS):
        if receipt["schema"] != domain or receipt["input_sha256"] != input_seal:
            raise ValueError("native receipt has another input/domain")
    if (submitted["physical_nn_admitted"] is not False or completed["submission_sha256"] != retired["submission_sha256"]
            or submitted["ticket"] != completed["ticket"] or completed["ticket"] != retired["ticket"]):
        raise ValueError("native submission/completion/retirement ticket chain mismatch")
    ticket = _native_ticket(submitted["ticket"])
    for name in ("source_view_seal_sha256", "native_input_sha256", "native_model_epoch_sha256", "native_backend_sha256",
                 "generation_view_sha256", "source_view_content_sha256", "native_endpoint_history_sha256"):
        sha(submitted[name])
    history = submitted["native_anchor_history_sha256"]
    if type(history) is not list or len(history) != len(record["input"]["selected_view"]["anchors"]):
        raise ValueError("native submission anchor history allocation mismatch")
    for identity in history:
        sha(identity)
    snapshot = record["input"]
    for name in ("game_id", "opening_id", "role", "model_epoch", "context_epoch", "capture_sequence", "worker_id", "job_id", "slice_id"):
        if type(submitted[name]) is not type(snapshot[name]) or submitted[name] != snapshot[name]:
            raise ValueError("native submission changed original source context")
    from .virtual_review import input_digest
    if (ticket["source_view_sha256"] != submitted["source_view_seal_sha256"]
            or ticket["representation_model_sha256"] != submitted["native_model_epoch_sha256"]
            or ticket["question_sha256"] != record["input"]["question_id"]
            or ticket["context_policy_sha256"] != snapshot["selected_view"]["context_policy_sha256"]
            or ticket["representation_history_sha256"] != submitted["native_endpoint_history_sha256"]
            or submitted["generation_view_sha256"] != record["input"]["view_sha256"]
            or submitted["actual_root_anchor"] != snapshot["selected_view"]["anchors"][0]
            or submitted["query_anchor"] != snapshot["rules_anchor"]
            or submitted["native_input_sha256"] != input_digest(record["input"]["encoded_model_input"])):
        raise ValueError("native representation differs from its original submission")
    fields(completed["execution"], ("epoch", "sequence"), "known physical execution")
    for name in ("epoch", "sequence"):
        uint(completed["execution"][name])
    if completed["execution"]["epoch"] != ticket["epoch"]:
        raise ValueError("physical execution belongs to another request epoch")
    if (completed["physical_completion"] != "confirmed" or completed["measurement_complete"] is not True
            or completed["admissible_actual_nn"] is not True or not uint(completed["completed_nn_inputs"])
            or type(completed["physical_fresh"]) is not bool or type(completed["output"]) is not dict
            or not completed["output"] or retired["accepted"] is not True):
        raise ValueError("actual generation input lacks accepted confirmed physical NN evidence")
    sha(completed["invocation_key_sha256"])
    if completed["physical_fresh"]:
        if completed["accepted_seed_sha256"] is not None:
            raise ValueError("Fresh physical invocation cannot claim a Warm seed")
    else:
        sha(completed["accepted_seed_sha256"])
    _native_output(snapshot, completed["output"])
    if validate_actions:
        _native_action_authorities(record, submitted, receipts)
    semantic_seal = semantic_sample_seal(record, submitted)
    if with_details:
        return {"semantic_sample_sha256": semantic_seal, "physical_fresh": completed["physical_fresh"], "submission": submitted}
    return semantic_seal


def _native_output(snapshot, output):
    role = snapshot["role"]
    expected = {"proposer": ("proposal", ("kind", "candidate_logits_f32", "wdl_f32")),
                "critic": ("counterexample", ("kind", "candidate_logits_f32", "divergence_logits_f32", "wdl_f32")),
                "validator": ("task_ranking", ("kind", "task_logits_f32"))}[role]
    fields(output, expected[1], "actual native role output")
    if output["kind"] != expected[0]:
        raise ValueError("actual native output role mismatch")
    def values(name, count):
        raw = output[name]
        if type(raw) is not list or len(raw) != count:
            raise ValueError("actual native output allocation mismatch")
        decoded = []
        for value in raw:
            if type(value) is not str or len(value) != 8 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("actual native output needs canonical f32 bit strings")
            number = struct.unpack(">f", bytes.fromhex(value))[0]
            if not math.isfinite(number):
                raise ValueError("actual native output is nonfinite")
            decoded.append(number)
        return decoded
    if role == "validator":
        values("task_logits_f32", 7)
    else:
        values("candidate_logits_f32", len(snapshot["legal_moves"]))
        if role == "critic":
            values("divergence_logits_f32", len(snapshot["divergence_record_ids"]))
        wdl = values("wdl_f32", 3)
        if any(v < 0 or v > 1 for v in wdl) or abs(sum(wdl) - 1) > 1e-4:
            raise ValueError("actual native WDL is not a normalized probability")


def _safe_path(root, relative):
    if type(relative) is not str or "\\" in relative:
        raise ValueError("chunk path must be relative POSIX JSONL")
    parts = PurePosixPath(relative)
    if (parts.is_absolute() or not parts.parts or any(p in ("", ".", "..") for p in parts.parts)
            or str(parts) != relative or parts.suffix != ".jsonl" or ":" in relative):
        raise ValueError("chunk path escapes registered corpus")
    target = root.joinpath(*parts.parts)
    # Walk only the declared root and descendants, including the final file.
    # A parent-chain loop can reach root.parent before checking target itself.
    current = root
    for depth in range(len(parts.parts) + 1):
        if current.is_symlink() or current.exists() and getattr(current.stat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("linked/reparse generation chunk is forbidden")
        if depth < len(parts.parts):
            current = current / parts.parts[depth]
    resolved = target.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("chunk outside immutable corpus root")
    return target


def validate_manifest(manifest):
    fields(manifest, ("schema", "source_kind", "producer", "opening_splits", "chunks", "total_bytes", "total_records"), "chunk manifest")
    if manifest["schema"] != MANIFEST_DOMAIN or manifest["source_kind"] != ACTUAL_SOURCE:
        raise ValueError("unknown generation chunk origin")
    producer, splits = manifest["producer"], manifest["opening_splits"]
    fields(producer, ("schema", "source_kind", "source_sha256", "model_sha256", "runtime_sha256", "teacher_binary_sha256",
                      "teacher_profile_sha256", "opening_split_sha256", "registered_before_outcomes", "actual_game_ids"), "chunk producer authority")
    if (producer["schema"] != "rz-pals-actual-collection-authority/2" or producer["source_kind"] != ACTUAL_SOURCE
            or producer["registered_before_outcomes"] is not True):
        raise ValueError("actual chunk producer authority required")
    for name in ("source_sha256", "model_sha256", "runtime_sha256", "teacher_binary_sha256", "teacher_profile_sha256", "opening_split_sha256"):
        sha(producer[name])
    if type(splits) is not dict or not 1 <= len(splits) <= 8192 or any(v not in SPLITS for v in splits.values()):
        raise ValueError("bounded preregistered opening splits required")
    for key in splits:
        identifier(key)
    if producer["opening_split_sha256"] != digest("rz-pals-opening-split/1", splits):
        raise ValueError("outcome-selected or changed opening split")
    games = producer["actual_game_ids"]
    if type(games) is not list or len(games) > MAX_TRAJECTORIES or len({identifier(g) for g in games}) != len(games):
        raise ValueError("bounded unique actual game publications required")
    chunks = manifest["chunks"]
    if type(chunks) is not list or not 1 <= len(chunks) <= MAX_CHUNKS:
        raise ValueError("bounded nonempty chunk manifest required")
    total_bytes, total_records, paths = 0, 0, set()
    for index, chunk in enumerate(chunks):
        fields(chunk, ("index", "kind", "path", "sha256", "bytes", "rows"), "generation chunk")
        if uint(chunk["index"]) != index or chunk["kind"] not in KINDS:
            raise ValueError("unknown or reordered chunk")
        identifier(chunk["path"])
        if chunk["path"] in paths:
            raise ValueError("duplicate chunk path")
        paths.add(chunk["path"])
        sha(chunk["sha256"])
        if not 1 <= uint(chunk["bytes"], MAX_CHUNK_BYTES) or not 1 <= uint(chunk["rows"], MAX_CHUNK_ROWS):
            raise ValueError("empty or oversized generation chunk")
        total_bytes += chunk["bytes"]
        total_records += chunk["rows"] if chunk["kind"] == "records" else 0
    if (uint(manifest["total_bytes"], MAX_CORPUS_BYTES) != total_bytes
            or uint(manifest["total_records"], MAX_RECORDS) != total_records):
        raise ValueError("generation corpus byte/record totals mismatch")
    return digest(MANIFEST_DOMAIN, manifest)


class _Lookup:
    def __init__(self, owner, table, value_field):
        self.owner, self.table, self.value_field = owner, table, value_field

    def __contains__(self, key):
        return self.owner._db.execute(f"SELECT 1 FROM {self.table} WHERE seal=?", (key,)).fetchone() is not None

    def get(self, key, default=None):
        found = self.owner._db.execute(f"SELECT chunk,offset,length FROM {self.table} WHERE seal=?", (key,)).fetchone()
        if found is None:
            return default
        row = self.owner._read_row(*found)
        if row["sha256"] != key:
            raise ValueError("sealed authority/evidence index mismatch")
        return row[self.value_field]

    def action_authority(self, causal_sha256):
        found = self.owner._db.execute("SELECT seal FROM action_authority WHERE causal_seal=?", (sha(causal_sha256),)).fetchone()
        if found is None:
            raise ValueError("actual V outcome lacks opaque action authority")
        return self.get(found[0])

    def head_evidence(self, input_sha256, head):
        found = self.owner._db.execute("SELECT seal FROM heads WHERE input_seal=? AND head=?", (sha(input_sha256), head)).fetchone()
        return None if found is None else (found[0], self.get(found[0]))

    def teacher_input_binding(self, input_sha256, policy_head_evidence_sha256):
        """Resolve one exact teacher/native input binding, never scan evidence.

        The primary key keeps duplicate semantic bindings out of the bounded
        index. Revalidation also binds the decoded immutable receipt to both
        lookup keys before the consumer joins its native retirement chain.
        """
        keys = (sha(input_sha256), sha(policy_head_evidence_sha256))
        found = self.owner._db.execute(
            "SELECT seal FROM teacher_inputs WHERE input_seal=? AND head_seal=? LIMIT 2", keys).fetchmany(2)
        if len(found) != 1:
            raise ValueError("teacher policy needs exactly one sealed input binding")
        seal = found[0][0]
        receipt = validate_evidence_receipt(seal, self.get(seal))
        if (receipt["schema"] != TEACHER_INPUT_BINDING_DOMAIN
                or (receipt["input_sha256"], receipt["policy_head_evidence_sha256"]) != keys):
            raise ValueError("teacher input binding index/raw authority mismatch")
        return seal, receipt

    def action_group(self, input_sha256):
        found = self.owner._db.execute("SELECT seal FROM action_groups WHERE input_seal=?", (sha(input_sha256),)).fetchone()
        return None if found is None else self.get(found[0])

    def action_source_input(self, input_sha256, outcome):
        group = self.action_group(input_sha256)
        if group is None:
            return input_sha256
        _frozen_action_group(group)
        found = [member for member in group["members"] if member["action"] == outcome["action"]
                 and member["causal_receipt_sha256"] == outcome["causal_receipt_sha256"]]
        if len(found) != 1:
            raise ValueError("task target is not an original preregistered frozen-group member")
        return found[0]["actual_input_sha256"]

    def accepted_retirement(self, input_sha256):
        accepted = []
        for (seal,) in self.owner._db.execute("SELECT seal FROM native WHERE input_seal=? AND domain=?",
                                              (sha(input_sha256), NATIVE_DOMAINS[2])):
            receipt = self.get(seal)
            if receipt.get("accepted") is True:
                accepted.append(seal)
                if len(accepted) > 1:
                    raise ValueError("duplicate accepted native chains for one original action input")
        if len(accepted) != 1:
            raise ValueError("original action input has no accepted native retirement")
        return accepted[0]

    def source_record(self, input_sha256):
        """Read one original publication, independent of chunk emission order."""
        found = self.owner._db.execute("SELECT chunk,offset,length,row_seal FROM raw_records WHERE input_seal=?",
                                       (sha(input_sha256),)).fetchone()
        if found is None:
            raise ValueError("action group refers to an unpublished original input")
        row = self.owner._read_row(*found[:3])
        if digest("rz-pals-generation-chunk-row/1", row) != found[3] or row["input_sha256"] != input_sha256:
            raise ValueError("immutable original action source row changed")
        return self.owner._hydrate_record(row)


class ChunkRows(Sequence):
    """Immutable row references; random access materializes at most one row."""
    def __init__(self, owner, split):
        self.owner, self.split = owner, split
        self.count = owner._db.execute("SELECT COUNT(*) FROM selected WHERE split=?", (split,)).fetchone()[0]

    def __len__(self):
        self.owner._assert_open()
        return self.count

    def __getitem__(self, index):
        if type(index) is not int:
            raise TypeError("bounded generation row access requires one integer, not a whole-corpus slice")
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.owner._check_index()
        reference = self.owner._db.execute("SELECT chunk,offset,length,row_seal FROM selected WHERE split=? AND ordinal=?", (self.split, index)).fetchone()
        result = self.owner._read_row(*reference[:3])
        if digest("rz-pals-generation-chunk-row/1", result) != reference[3]:
            raise ValueError("immutable generation row changed")
        return self.owner._hydrate_record(result)

    def __iter__(self):
        self.owner._check_index()
        references = self.owner._db.execute("SELECT chunk,offset,length,row_seal FROM selected WHERE split=? ORDER BY ordinal", (self.split,))
        active, stream = None, None
        try:
            for chunk, offset, length, row_seal in references:
                if chunk != active:
                    if stream is not None:
                        stream.close()
                    stream = self.owner._open_verified(chunk)
                    active = chunk
                stream.seek(offset)
                raw = stream.read(length)
                result = strict_json(raw.decode("utf-8", "strict"))
                if digest("rz-pals-generation-chunk-row/1", result) != row_seal:
                    raise ValueError("immutable streamed generation row changed")
                yield self.owner._hydrate_record(result)
        finally:
            if stream is not None:
                stream.close()

    def role_pool(self, role):
        """Only bounded indices and input identities, never all input payloads."""
        self.owner._check_index()
        found = self.owner._db.execute("SELECT ordinal,input_seal FROM selected WHERE split=? AND role=? ORDER BY ordinal", (self.split, role))
        indices, hasher, first = [], hashlib.sha256(), True
        # Canonical [domain,[manifest, [input seals...]]] without a seal list copy.
        hasher.update(b"[" + json.dumps("rz-pals-generation-chunk-sampler-pool/1").encode()
                      + b",[" + json.dumps(self.owner._seal).encode() + b",[")
        for ordinal, seal in found:
            indices.append(ordinal)
            hasher.update((b"" if first else b",") + json.dumps(seal).encode())
            first = False
        hasher.update(b"]]]")
        return indices, hasher.hexdigest()


class ChunkGenerationDataset:
    """Global admission over pinned chunks using bounded disposable SQLite.

An external manifest pin is mandatory. Receipt claims still require recovery
from the actual native producer; hashes alone do not prove an executed game.
The index is regenerated in an explicitly supplied transient directory.
"""
    def __init__(self, root, manifest, *, expected_manifest_sha256, index_root, max_index_bytes=MAX_INDEX_BYTES):
        self._closed, self._sealed, self._temporary = False, False, None
        self._cache, self._cache_bytes = OrderedDict(), 0
        self._seal = validate_manifest(manifest)  # Reject capacity before copying caller data.
        self._manifest = copy.deepcopy(manifest)
        if validate_manifest(self._manifest) != self._seal:
            raise ValueError("manifest changed while forming immutable snapshot")
        if sha(expected_manifest_sha256) != self._seal:
            raise ValueError("generation manifest differs from externally registered pin")
        self._root = Path(root).resolve(strict=True)
        if Path(root).is_symlink() or not self._root.is_dir():
            raise ValueError("explicit unlinked generation corpus directory required")
        self._paths = [_safe_path(self._root, c["path"]) for c in self._manifest["chunks"]]
        self._identities = [None] * len(self._paths)
        self._max_index_bytes = uint(max_index_bytes, MAX_INDEX_BYTES)
        if self._max_index_bytes < 65536:
            raise ValueError("generation index capacity is too small")
        parent = Path(index_root).resolve(strict=True)
        if not parent.is_dir() or Path(index_root).is_symlink():
            raise ValueError("explicit unlinked transient index directory required")
        self._index_parent = parent
        try:
            self._temporary = tempfile.TemporaryDirectory(prefix="rz-generation-index-", dir=parent)
            self._index_path = Path(self._temporary.name) / "verified.sqlite3"
            self._db = sqlite3.connect(self._index_path)
            self._db.execute("PRAGMA journal_mode=OFF")
            self._db.execute("PRAGMA temp_store=FILE")
            self._db.execute("PRAGMA cache_size=-2048")
            self._db.execute(f"PRAGMA max_page_count={self._max_index_bytes // 4096}")
            self._build_index()
            self._db.commit()
            self._db.execute("PRAGMA query_only=ON")
            self._index_seal = self._index_digest()
            self._index_stamp = self._file_stamp(self._index_path)
            self._sealed = True
        except sqlite3.Error as error:
            self.close()
            raise ValueError("GenerationIndexCapacityOrIntegrity: bounded SQLite index refused data") from error
        except Exception:
            self.close()
            raise

    def _assert_open(self):
        if self._closed:
            raise ValueError("generation chunk dataset is closed")

    def _open_verified(self, index):
        self._assert_open()
        chunk, path = self._manifest["chunks"][index], self._paths[index]
        if _safe_path(self._root, chunk["path"]) != path:
            raise ValueError("immutable chunk path changed")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        stream = os.fdopen(os.open(path, flags), "rb")
        try:
            info = os.fstat(stream.fileno())
            identity = (info.st_dev, info.st_ino, info.st_size)
            if not stat.S_ISREG(info.st_mode) or info.st_size != chunk["bytes"] or info.st_nlink != 1:
                raise ValueError("chunk type/size/link identity mismatch")
            if self._identities[index] is not None and self._identities[index] != identity:
                raise ValueError("immutable chunk was replaced")
            hasher, total = hashlib.sha256(), 0
            while True:
                block = stream.read(65536)
                if not block:
                    break
                total += len(block)
                if total > MAX_CHUNK_BYTES:
                    raise ValueError("chunk byte capacity exceeded")
                hasher.update(block)
            if total != chunk["bytes"] or hasher.hexdigest() != chunk["sha256"]:
                raise ValueError("generation chunk byte hash mismatch")
            current = path.stat()
            if (current.st_dev, current.st_ino, current.st_size) != identity:
                raise ValueError("chunk changed during verified open")
            self._identities[index] = identity
            stream.seek(0)
            return stream
        except Exception:
            stream.close()
            raise

    def _chunk_rows(self, index):
        with self._open_verified(index) as stream:
            count = 0
            while True:
                offset = stream.tell()
                raw = stream.readline(MAX_ROW_BYTES + 1)
                if not raw:
                    break
                if len(raw) > MAX_ROW_BYTES or not raw.endswith(b"\n"):
                    raise ValueError("bounded complete UTF-8 JSONL row required")
                count += 1
                if count > MAX_CHUNK_ROWS:
                    raise ValueError("chunk row capacity exceeded")
                yield offset, len(raw), strict_json(raw.decode("utf-8", "strict"))
            if count != self._manifest["chunks"][index]["rows"]:
                raise ValueError("chunk row count mismatch")

    def _read_row(self, chunk, offset, length):
        uint(length, MAX_ROW_BYTES)
        # Cached bytes are an immutable verified snapshot, not parsed corpus
        # objects. Full source hashes are checked again by sha256/admission;
        # callers never receive or mutate the cache backing this row.
        path = _safe_path(self._root, self._manifest["chunks"][chunk]["path"])
        info = path.stat()
        stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        cached = self._cache.pop(chunk, None)
        if cached is not None and cached[0] != stamp:
            self._cache_bytes -= len(cached[1])
            cached = None
        if cached is None:
            with self._open_verified(chunk) as stream:
                content = stream.read(MAX_CHUNK_BYTES + 1)
            if len(content) != self._manifest["chunks"][chunk]["bytes"] or hashlib.sha256(content).hexdigest() != self._manifest["chunks"][chunk]["sha256"]:
                raise ValueError("chunk changed while forming immutable cache")
            while self._cache and self._cache_bytes + len(content) > MAX_CACHE_BYTES:
                _, evicted = self._cache.popitem(last=False)
                self._cache_bytes -= len(evicted[1])
            cached = (stamp, content)
            self._cache_bytes += len(content)
        self._cache[chunk] = cached
        raw = cached[1][offset:offset + length]
        if len(raw) != length or not raw.endswith(b"\n"):
            raise ValueError("sealed row offset/length mismatch")
        return strict_json(raw.decode("utf-8", "strict"))

    def _index_digest(self):
        size = self._index_path.stat().st_size
        if size > self._max_index_bytes or size + self._manifest["total_bytes"] > MAX_CORPUS_BYTES:
            raise ValueError("GenerationIndexCapacity: corpus plus transient index exceeds storage budget")
        hasher = hashlib.sha256()
        with self._index_path.open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                hasher.update(block)
        return hasher.hexdigest()

    @staticmethod
    def _file_stamp(path):
        info = path.stat()
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _check_index(self):
        self._assert_open()
        # This private query-only index has one owner and is never externally
        # supplied. Any observed replacement/write triggers full verification;
        # corpus identity/admission still rehash every source chunk's bytes.
        if self._sealed and self._file_stamp(self._index_path) != self._index_stamp:
            raise ValueError("immutable generation manifest/index changed")

    def _build_index(self):
        self._db.executescript("""
            CREATE TABLE inputs(seal TEXT PRIMARY KEY);
            CREATE TABLE evidence(seal TEXT PRIMARY KEY,chunk INTEGER,offset INTEGER,length INTEGER);
            CREATE TABLE evidence_occurrences(seal TEXT NOT NULL,chunk INTEGER NOT NULL,offset INTEGER NOT NULL,
                                              length INTEGER NOT NULL,PRIMARY KEY(chunk,offset));
            CREATE INDEX evidence_occurrences_by_seal ON evidence_occurrences(seal,chunk,offset);
            CREATE TABLE trajectories(seal TEXT PRIMARY KEY,chunk INTEGER,offset INTEGER,length INTEGER);
            CREATE TABLE native(input_seal TEXT,domain TEXT,seal TEXT,PRIMARY KEY(input_seal,domain,seal));
            CREATE TABLE action_authority(causal_seal TEXT PRIMARY KEY,seal TEXT UNIQUE);
            CREATE TABLE action_groups(input_seal TEXT PRIMARY KEY,group_seal TEXT UNIQUE,seal TEXT UNIQUE);
            CREATE TABLE heads(input_seal TEXT,head TEXT,seal TEXT UNIQUE,PRIMARY KEY(input_seal,head));
            CREATE TABLE teacher_inputs(input_seal TEXT NOT NULL,head_seal TEXT NOT NULL,seal TEXT NOT NULL UNIQUE,
                                        PRIMARY KEY(input_seal,head_seal));
            CREATE TABLE raw_records(input_seal TEXT PRIMARY KEY,chunk INTEGER,offset INTEGER,length INTEGER,row_seal TEXT);
            CREATE TABLE records(position INTEGER PRIMARY KEY,chunk INTEGER,offset INTEGER,length INTEGER,row_seal TEXT,
                                 input_seal TEXT UNIQUE,semantic_seal TEXT,valid_target INTEGER,physical_fresh INTEGER,role TEXT,split TEXT,
                                 excluded INTEGER DEFAULT 0);
            CREATE INDEX by_semantic_choice ON records(semantic_seal,excluded,physical_fresh,valid_target,position);
            CREATE TABLE domains(kind TEXT,identity TEXT,mask INTEGER,PRIMARY KEY(kind,identity));
            CREATE TABLE record_domains(position INTEGER,kind TEXT,identity TEXT);
            CREATE INDEX by_record_domain ON record_domains(position);
            CREATE TABLE selected(split TEXT,ordinal INTEGER,chunk INTEGER,offset INTEGER,length INTEGER,row_seal TEXT,
                                  input_seal TEXT,role TEXT,PRIMARY KEY(split,ordinal));
        """)
        counts = {kind: 0 for kind in KINDS}
        # All authorities are indexed before any supervised record is joined.
        for chunk in self._manifest["chunks"]:
            kind, index = chunk["kind"], chunk["index"]
            if kind == "records":
                continue
            for offset, length, row in self._chunk_rows(index):
                counts[kind] += 1
                cap = {"inputs": MAX_RECORDS, "evidence": MAX_EVIDENCE, "trajectories": MAX_TRAJECTORIES}[kind]
                if counts[kind] > cap:
                    raise ValueError("generation authority entry capacity exceeded")
                if kind == "inputs":
                    fields(row, ("input_sha256",), "published chunk input")
                    self._db.execute("INSERT INTO inputs VALUES(?)", (sha(row["input_sha256"]),))
                else:
                    name = "receipt" if kind == "evidence" else "trajectory"
                    fields(row, ("sha256", name), "chunk authority entry")
                    seal, value = sha(row["sha256"]), row[name]
                    if kind == "evidence":
                        try:
                            validate_evidence_receipt(seal, value)
                        except ValueError as error:
                            raise ValueError("chunk full receipt seal/domain mismatch") from error
                        # Every original occurrence is checked, including a
                        # null retirement repeated after a fresh game resets
                        # its native epoch. A dictionary definition is not a
                        # second authority or a second accepted completion.
                        has_native_input = False
                        if value["schema"] in NATIVE_DOMAINS:
                            _native_ticket(value.get("ticket"))
                            has_native_input = value["schema"] != NATIVE_DOMAINS[2] or _native_retirement(value)
                            if has_native_input:
                                sha(value.get("input_sha256"))
                        elif value["schema"] == ACTION_AUTHORITY_DOMAIN:
                            _action_authority(value)
                        elif value["schema"] == ACTION_GROUP_DOMAIN:
                            _frozen_action_group(value)
                        elif value["schema"] == HEAD_EVIDENCE_DOMAIN:
                            _head_receipt(value)

                        first = self._db.execute("SELECT chunk,offset,length FROM evidence WHERE seal=?", (seal,)).fetchone()
                        if first is not None:
                            original = self._read_row(*first)
                            if (original["sha256"] != seal or
                                    self._canonical_receipt(original["receipt"]) != self._canonical_receipt(value)):
                                raise ValueError("conflicting canonical receipt content for identical evidence seal")
                        self._db.execute("INSERT INTO evidence_occurrences VALUES(?,?,?,?)", (seal, index, offset, length))
                        if first is not None:
                            # Do not reregister native/head/teacher/action
                            # causal keys for one identical sealed definition.
                            # Raw count and byte caps above still count this
                            # occurrence, and its immutable location is kept.
                            continue
                        if value["schema"] in NATIVE_DOMAINS and has_native_input:
                            self._db.execute("INSERT INTO native VALUES(?,?,?)", (value["input_sha256"], value["schema"], seal))
                        elif value["schema"] == ACTION_AUTHORITY_DOMAIN:
                            self._db.execute("INSERT INTO action_authority VALUES(?,?)", (value["causal_receipt_sha256"], seal))
                        elif value["schema"] == ACTION_GROUP_DOMAIN:
                            self._db.execute("INSERT INTO action_groups VALUES(?,?,?)",
                                             (value["input_sha256"], value["group_sha256"], seal))
                        elif value["schema"] == HEAD_EVIDENCE_DOMAIN:
                            self._db.execute("INSERT INTO heads VALUES(?,?,?)", (value["input_sha256"], value["head"], seal))
                        elif value["schema"] == TEACHER_INPUT_BINDING_DOMAIN:
                            self._db.execute("INSERT INTO teacher_inputs VALUES(?,?,?)",
                                             (sha(value["input_sha256"]), sha(value["policy_head_evidence_sha256"]), seal))
                    elif trajectory_digest(value) != seal:
                        raise ValueError("chunk trajectory seal mismatch")
                    self._db.execute(f"INSERT INTO {kind} VALUES(?,?,?,?)", (seal, index, offset, length))
        producer, splits = self._manifest["producer"], self._manifest["opening_splits"]
        published, receipts = _Lookup(self, "inputs", None), _Lookup(self, "evidence", "receipt")
        trajectories, games = _Lookup(self, "trajectories", "trajectory"), set(producer["actual_game_ids"])
        # Index only bounded immutable locations before joining any action
        # group. A later-emitted member is never mistaken for an unknown input,
        # and neither complete games nor model payloads are kept in RAM.
        raw_count = 0
        for chunk in self._manifest["chunks"]:
            if chunk["kind"] != "records":
                continue
            for offset, length, record in self._chunk_rows(chunk["index"]):
                raw_count += 1
                if raw_count > MAX_RECORDS:
                    raise ValueError("generation record entry capacity exceeded")
                validate_record(record)
                row_seal = digest("rz-pals-generation-chunk-row/1", record)
                found = self._db.execute("SELECT row_seal FROM raw_records WHERE input_seal=?", (record["input_sha256"],)).fetchone()
                if found:
                    if found[0] != row_seal:
                        raise ValueError("conflicting targets/provenance for identical generation input seal")
                else:
                    self._db.execute("INSERT INTO raw_records VALUES(?,?,?,?,?)",
                                     (record["input_sha256"], chunk["index"], offset, length, row_seal))
        if raw_count != self._manifest["total_records"]:
            raise ValueError("manifest record count differs from parsed corpus")
        for (seal,) in self._db.execute("SELECT seal FROM action_groups"):
            group = _frozen_action_group(receipts.get(seal))
            for member in group["members"]:
                origin = member["actual_input_sha256"]
                if (origin not in published
                        or not self._db.execute("SELECT 1 FROM raw_records WHERE input_seal=?", (origin,)).fetchone()):
                    raise ValueError("frozen action group refers to an unpublished original input")
        duplicate_rows, position = 0, 0
        for chunk in self._manifest["chunks"]:
            if chunk["kind"] != "records":
                continue
            for offset, length, record in self._chunk_rows(chunk["index"]):
                counts["records"] += 1
                # Keep the original immutable publication bytes as the row
                # authority; late labels are a separate sealed head receipt.
                row_seal = digest("rz-pals-generation-chunk-row/1", record)
                record = self._hydrate_record(record)
                GenerationDataset._validate_published_record(record, producer, published, games, trajectories, receipts)
                accepted = []
                for (retirement_sha256,) in self._db.execute("SELECT seal FROM native WHERE input_seal=? AND domain=?", (record["input_sha256"], NATIVE_DOMAINS[2])):
                    if receipts.get(retirement_sha256).get("accepted") is True:
                        accepted.append(retirement_sha256)
                        if len(accepted) > 1:
                            raise ValueError("duplicate accepted native chains for one input seal")
                if len(accepted) != 1:
                    raise ValueError("published input has no accepted native retirement")
                chain = _native_chain(record, receipts, accepted[0], with_details=True)
                semantic_seal, physical_fresh = chain["semantic_sample_sha256"], chain["physical_fresh"]
                targets = validate_record(record)
                snapshot = record["input"]
                valid_target = (bool(targets["task"] and targets["task"]["pairs"])
                                if snapshot["role"] == "validator"
                                else any(targets[head] is not None for head in ("policy", "wdl", "divergence")))
                if snapshot["opening_id"] not in splits:
                    raise ValueError("unregistered opening in generation chunks")
                found = self._db.execute("SELECT row_seal FROM records WHERE input_seal=?", (record["input_sha256"],)).fetchone()
                if found:
                    if found[0] != row_seal:
                        raise ValueError("conflicting targets/provenance for identical generation input seal")
                    duplicate_rows += 1
                    continue
                split = splits[snapshot["opening_id"]]
                self._db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,0)",
                                 (position, chunk["index"], offset, length, row_seal, record["input_sha256"],
                                  semantic_seal, int(valid_target), int(physical_fresh), snapshot["role"], split))
                for kind, identity in (("genealogy", snapshot["line_genealogy_id"]), ("transposition", snapshot["transposition_sha256"]),
                                       ("rules_state", snapshot["rules_anchor"]["rules_state_sha256"]), ("game", snapshot["game_id"]),
                                       ("sample", semantic_seal)):
                    bit = 1 << SPLITS.index(split)
                    self._db.execute("INSERT INTO domains VALUES(?,?,?) ON CONFLICT(kind,identity) DO UPDATE SET mask=mask|excluded.mask", (kind, identity, bit))
                    self._db.execute("INSERT INTO record_domains VALUES(?,?,?)", (position, kind, identity))
                position += 1
        if counts["records"] != self._manifest["total_records"]:
            raise ValueError("manifest record count differs from parsed corpus")
        if self._db.execute("SELECT COUNT(*) FROM inputs WHERE seal NOT IN (SELECT input_seal FROM records)").fetchone()[0]:
            raise ValueError("published input seal absent from collected record corpus")
        # Incremental empty-table indexes and a correlated EXISTS avoid a
        # corpus-sized sort/IN-result temp file outside max_page_count.
        self._db.execute("UPDATE records SET excluded=1 WHERE EXISTS (SELECT 1 FROM record_domains rd JOIN domains d USING(kind,identity) WHERE rd.position=records.position AND d.mask NOT IN (1,2,4))")
        self._exclusions = {"cross_split_records_excluded": self._db.execute("SELECT COUNT(*) FROM records WHERE excluded=1").fetchone()[0],
                           "cross_split_identities": self._db.execute("SELECT COUNT(*) FROM domains WHERE mask NOT IN (1,2,4)").fetchone()[0],
                           "duplicate_inputs_excluded": duplicate_rows,
                           "warm_seed_inputs_excluded": self._db.execute("SELECT COUNT(*) FROM records WHERE physical_fresh=0").fetchone()[0]}
        for split in SPLITS:
            # Only confirmed Fresh rows enter actual training; Warm source,
            # output and seeds remain raw evidence and are never rewritten.
            # Only target availability is compared among Fresh repetitions.
            # Prefer earliest valid, or earliest masked if every one is masked.
            # Raw publication rows stay in records and original chunks.
            # Two indexed EXISTS probes avoid GROUP BY or a temp sorting pool.
            source = self._db.execute("""
                SELECT r.chunk,r.offset,r.length,r.row_seal,r.input_seal,r.role
                FROM records r NOT INDEXED WHERE r.excluded=0 AND r.physical_fresh=1 AND r.split=?
                  AND (r.valid_target=1 OR NOT EXISTS (
                    SELECT 1 FROM records v INDEXED BY by_semantic_choice WHERE v.semantic_seal=r.semantic_seal
                      AND v.excluded=0 AND v.physical_fresh=1 AND v.valid_target=1))
                  AND NOT EXISTS (
                    SELECT 1 FROM records earlier INDEXED BY by_semantic_choice WHERE earlier.semantic_seal=r.semantic_seal
                      AND earlier.excluded=0 AND earlier.physical_fresh=1 AND earlier.valid_target=r.valid_target
                      AND earlier.position<r.position)
                ORDER BY r.position
            """, (split,))
            for ordinal, reference in enumerate(source):
                self._db.execute("INSERT INTO selected VALUES(?,?,?,?,?,?,?,?)", (split, ordinal, *reference))
        self._exclusions["duplicate_semantic_inputs_excluded"] = (
            self._db.execute("SELECT COUNT(*) FROM records WHERE excluded=0 AND physical_fresh=1").fetchone()[0]
            - self._db.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
        )

    def _hydrate_record(self, record):
        """Bind one late head per input without rewriting its submitted input.

        The raw JSONL row stays immutable. Only this newly decoded one-row
        object receives labels, and the completed manifest pins every receipt.
        Entire games' model inputs are never buffered waiting for final WDL.
        """
        validate_record(record)
        receipts = _Lookup(self, "evidence", "receipt")
        targets = dict(record["future_targets"])
        for head in HEADS:
            found = receipts.head_evidence(record["input_sha256"], head)
            if found is None:
                continue
            seal, receipt = found
            _head_receipt(receipt)
            if receipt["input_sha256"] != record["input_sha256"] or receipt["head"] != head:
                raise ValueError("streamed future head index/input mismatch")
            target = dict(receipt["target"], raw_evidence_sha256=seal)
            if uint(target.get("observed_sequence")) <= record["input"]["capture_sequence"]:
                raise ValueError("actual streamed future head must follow its immutable input event")
            if targets[head] is not None and targets[head] != target:
                raise ValueError("immutable published target conflicts with later head evidence")
            targets[head] = target
        record["future_targets"] = targets
        validate_record(record)
        return record

    def rows(self, split, *, optimization=False):
        self._check_index()
        if split not in SPLITS or optimization and split != "train":
            raise ValueError("holdout/validation corpus cannot be an optimization pool")
        return ChunkRows(self, split)

    @staticmethod
    def _canonical_receipt(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    def evidence_occurrences(self, evidence_sha256):
        """Stream original immutable receipt locations, not new authorities.

        Repeated definitions retain their chunk pin, offset and exact raw
        length. Only the evidence dictionary reuses a canonical definition;
        published input/trajectory keys and record admission are unchanged.
        """
        self._check_index()
        seal = sha(evidence_sha256)
        for chunk, offset, length in self._db.execute(
                "SELECT chunk,offset,length FROM evidence_occurrences WHERE seal=? ORDER BY chunk,offset", (seal,)):
            self._check_index()
            yield {"sha256": seal, "chunk": chunk, "path": self._manifest["chunks"][chunk]["path"],
                   "chunk_sha256": self._manifest["chunks"][chunk]["sha256"], "offset": offset, "length": length}

    def admit(self, phase):
        # SQL selected already removes semantic repeats and publication conflicts;
        # no corpus-sized Python set or full record collection is needed here.
        self.sha256
        if (self._exclusions["warm_seed_inputs_excluded"]
                and not self._db.execute("SELECT 1 FROM records WHERE physical_fresh=1 LIMIT 1").fetchone()):
            raise ValueError("UnsupportedWarmTrainingNamespace: eligible_fresh_inputs=0")
        counts = Counter()
        for record in self.rows("train", optimization=True):
            _count_target_coverage(counts, record, validate_record(record))
        return _admit_coverage(dict(counts), phase)

    @property
    def exclusions(self):
        self._check_index()
        return dict(self._exclusions)

    @property
    def exclusion_reasons(self):
        self._check_index()
        return {"warm_seed_inputs_excluded": "UnsupportedWarmTrainingNamespace"}

    @property
    def sha256(self):
        self._check_index()
        if validate_manifest(self._manifest) != self._seal or self._index_digest() != self._index_seal:
            raise ValueError("immutable generation manifest/index changed")
        # A dataset identity never survives unnoticed source chunk changes.
        for index in range(len(self._paths)):
            with self._open_verified(index):
                pass
        return self._seal

    @property
    def authority(self):
        self._check_index()
        return copy.deepcopy(self._manifest["producer"])

    @property
    def opening_splits(self):
        self._check_index()
        return copy.deepcopy(self._manifest["opening_splits"])

    def close(self):
        try:
            if not self._closed:
                self._closed = True
                if hasattr(self, "_db"):
                    self._db.close()
            if self._temporary is not None:
                target = Path(self._temporary.name)
                if target.exists() and (target.is_symlink() or not target.resolve(strict=True).is_relative_to(self._index_parent)
                                         or getattr(target.stat(), "st_file_attributes", 0) & 0x400):
                    raise ValueError("owned transient index cleanup target changed; quarantined")
                self._temporary.cleanup()
                self._temporary = None
                self.cleanup_failure = None
        except OSError as error:
            self.cleanup_failure = {"error_type": type(error).__name__, "owned_index_root": self._temporary.name}
            raise ValueError("GenerationIndexCleanupFailed: " + self._temporary.name) from error
        finally:
            self._cache.clear()
            self._cache_bytes = 0

    def __enter__(self):
        self._assert_open()
        return self

    def __exit__(self, *_):
        self.close()
