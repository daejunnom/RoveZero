"""Strict historical utility cohort bodies, without original trial authority.

The file owner supplies retained immutable raw bytes. Native seals, raw file
pins and canonical metadata hashes deliberately keep separate domains here.
This consumer never reconstructs a ready offer, physical fence, independent
trial, Rules owner or training permission from public JSON. The Rust parent
must separately compare the original source projection and replay Rules.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

from .generation_data import fields, sha, uint
from .rovent_utility_cohort_files import (
    COST_SCOPE, HistoricalUtilityCohortFiles, MAX_MATERIALIZATION_BYTES,
)
from .rovent_utility_training import (
    COHORT_PREPARATION_DOMAIN, COST_OBJECTIVE, validate_training_preparation,
)
from . import rovent_utility_root_local as structure


PROFILE_DOMAIN = "rz-rovent-utility-comparison-cohort/1"
TRIAL_DOMAIN = "rz-rovent-utility-cohort-trial/1"
MATERIALIZATION_DOMAIN = "rz-rovent-utility-cohort-materialization/1"
PAIR_DOMAIN = "rz-rovent-utility-cohort-pair/1"
COST_DOMAIN = "rz-rovent-utility-cohort-cost/1"
MAX_PROFILE_BYTES = 64 << 10
MAX_WINDOW_NS = 86_400_000_000_000
_PROFILE_FIELDS = (
    "input_lock_sha256", "source_sha256", "frozen_nnuec_model_sha256",
    "frozen_nnuec_namespace_sha256", "frozen_pc_model_sha256",
    "frozen_pc_namespace_sha256", "mode", "requested_recipients",
    "prelude_ns", "action_ns", "cleanup_ns", "logical_budget_sha256",
    "ready_policy_sha256", "cache_policy_sha256", "cost_class_sha256",
    "resource_plan_sha256", "resource_epoch", "usage", "cost_scope",
)
_SOURCE_FIELDS = (
    "input_lock_sha256", "source_sha256", "frozen_nnuec_model_sha256",
    "frozen_nnuec_namespace_sha256", "frozen_pc_model_sha256",
    "frozen_pc_namespace_sha256", "resource_plan_sha256", "logical_budget_sha256",
)
_FROZEN_FIELDS = (
    "input_sha256", "target_sha256", "context_sha256", "target_record_ids",
    "input_bytes", "root_rules_sha256", "original_first_move", "first_trial_sequence",
)
_TRIAL_FIELDS = (
    "schema", "cohort_id", "trial_sequence", "recipient", "profile",
    "profile_sha256", "source", "frozen_context", "materialized_sha256",
    "materialized_json", "historical_unbound_input", "unresolved_source",
    "cost", "resolved", "comparable", "unknown_reason", "target_known_mask",
)
_IO_FIELDS = (
    "clock_epoch", "started_tick_ns", "finished_tick_ns", "elapsed_ns",
    "written_bytes", "readback_bytes", "synced_files", "synced_directories",
)
_COST_FIELDS = (
    "domain", "scope", "objective", "cost_class_sha256", "cost_epoch_sha256",
    "resource_plan_sha256", "resource_epoch", "started_tick", "root_fence_tick",
    "actor_release_tick", "physical_pin_zero_tick", "whole_wall_ns",
    "original_deadline_tick", "cohort_action_cutoff_tick", "cohort_cleanup_cutoff_tick",
    "baseline_counters", "final_counters", "native_credit_baseline",
    "native_credit_final", "cpu_thread_time_ns", "energy", "complete_scope",
)
_COUNTER_FIELDS = (
    "native_preparations", "submitted_role_tickets", "physical_role_completions",
    "completed_nn_inputs", "nn_input_count_unknown", "consumed_role_outputs",
    "accepted_nn_inputs", "accepted_nn_input_count_unknown", "consumed_nn_inputs",
    "consumed_nn_input_count_unknown", "zero_nn_outputs_consumed", "unused_role_results",
    "canceled_preparations", "cpu_slice_receipts", "cpu_progress_publications",
    "cpu_yields", "cpu_yield_requests", "cpu_yield_acknowledgements",
    "cpu_natural_completions", "validator_ticket_admissions", "foreign_reports",
    "foreign_reported_nodes", "foreign_node_count_unknown", "foreign_yield_requests",
    "foreign_yield_acknowledgements", "rejected_events", "notification_waits",
)
_RECIPIENTS = ("proposer", "critic", "researcher")


class UnsupportedCohortPayload(ValueError):
    """An unregistered original proof body stays unknown, never a target."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"utility cohort exact boolean required: {label}")
    return value


def _text(value, maximum=4096):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > maximum
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError("utility cohort bounded original metadata text required")
    return value


def _positive(value, maximum=2**64 - 1):
    if not uint(value, maximum):
        raise ValueError("utility cohort positive bounded integer required")
    return value


def _array(value, maximum):
    if type(value) is not list or len(value) > maximum:
        raise ValueError("utility cohort bounded original array required")
    return value


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("utility cohort original JSON contains duplicate keys")
        value[key] = item
    return value


def _nonfinite(_value):
    raise ValueError("utility cohort nonfinite JSON constant rejected")


def _decode(raw, maximum):
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise ValueError("utility cohort bounded immutable raw bytes required")
    try:
        value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_object,
            parse_constant=_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("utility cohort original UTF8 JSON required") from error
    remaining = [maximum]

    def walk(item, depth=0):
        remaining[0] -= 1
        if depth > 64 or remaining[0] < 0:
            raise ValueError("utility cohort original JSON structure bound")
        if type(item) is dict:
            for nested in item.values():
                walk(nested, depth + 1)
        elif type(item) is list:
            for nested in item:
                walk(nested, depth + 1)
        elif type(item) not in (str, int, bool, type(None)):
            # Floating input spellings belong in reference_json, not a second
            # serialization of the outer source/clock/retirement metadata.
            raise ValueError("utility cohort outer metadata requires integer/string wire")
    walk(value)
    if type(value) is not dict:
        raise ValueError("utility cohort original JSON object required")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _native_digest(domain, raw):
    return hashlib.sha256(domain.encode("ascii") + b"\0"
        + len(raw).to_bytes(8, "little") + raw).hexdigest()


def _value_digest(domain, value):
    return _native_digest(domain, _canonical(value))


def validate_cohort_profile(value):
    """Check the exact declared profile; this does not observe ready trials."""
    fields(value, ("domain", "input"), "utility cohort profile")
    if value["domain"] != PROFILE_DOMAIN:
        raise ValueError("utility cohort explicit profile domain required")
    body = fields(value["input"], _PROFILE_FIELDS, "utility cohort profile input")
    for name in _PROFILE_FIELDS:
        if name.endswith("_sha256"):
            sha(body[name])
    if body["mode"] not in ("cpu_pc", "cpu_pcr"):
        raise ValueError("utility cohort CPU-only mode required")
    recipients = _array(body["requested_recipients"], 3)
    if (not recipients or any(type(item) is not str or item not in _RECIPIENTS for item in recipients)
            or len(set(recipients)) != len(recipients)
            or body["mode"] == "cpu_pc" and "researcher" in recipients):
        raise ValueError("utility cohort unique ordered supported recipients required")
    total = sum(_positive(body[name]) for name in ("prelude_ns", "action_ns", "cleanup_ns"))
    if total > MAX_WINDOW_NS:
        raise ValueError("utility cohort declared window exceeds 24 hours")
    _positive(body["resource_epoch"])
    if body["usage"] not in ("validation_only", "collection_candidate") or body["cost_scope"] != COST_SCOPE:
        raise ValueError("utility cohort explicit usage/cost scope required")
    if len(_canonical(value)) > MAX_PROFILE_BYTES:
        raise ValueError("utility cohort canonical profile exceeds 64KiB")
    return copy.deepcopy(value)


def cohort_profile_sha256(profile):
    """Return the declared profile's domain seal, distinct from raw SHA."""
    profile = validate_cohort_profile(profile)
    # This profile domain deliberately has NO length prefix, unlike native
    # trial/materialization/pair seals. A raw file SHA is different again.
    return hashlib.sha256(PROFILE_DOMAIN.encode("ascii") + b"\0" + _canonical(profile)).hexdigest()


def _source(value, profile):
    fields(value, _SOURCE_FIELDS, "utility cohort original source pins")
    for name in _SOURCE_FIELDS:
        if sha(value[name]) != profile["input"][name]:
            raise ValueError("utility cohort original source/profile pin differs")


def _frozen(value):
    fields(value, _FROZEN_FIELDS, "utility cohort frozen decision")
    for name in ("input_sha256", "target_sha256", "context_sha256", "root_rules_sha256"):
        sha(value[name])
    identifiers = _array(value["target_record_ids"], 64)
    for item in identifiers:
        uint(item, 63)
    if not identifiers or identifiers != sorted(set(identifiers)):
        raise ValueError("utility cohort frozen original ordered target IDs required")
    _positive(value["input_bytes"], 8 << 20)
    uint(value["original_first_move"], 65535)
    _positive(value["first_trial_sequence"], 3)


def _io(value):
    fields(value, _IO_FIELDS, "utility cohort original trial ACK collection I/O")
    for item in value.values():
        uint(item)
    if value["finished_tick_ns"] - value["started_tick_ns"] != value["elapsed_ns"]:
        raise ValueError("utility cohort original trial ACK I/O clock differs")
    return value


def _resource_observations(value):
    fields(value, ("native", "own"), "utility cohort actual resource observations")
    methods = ("LinuxProcThreadStatus", "LinuxProcProcessStatus", "LinuxSysfsSiblings",
        "LinuxCgroupV2Direct", "LinuxCgroupV2VisibleAncestors", "WindowsProcessAndThreadGroupAffinity",
        "WindowsLogicalProcessorInformationEx", "WindowsImmediateJobObject", "WorkerReadyBoundary",
        "OrtAppliedSessionSettings")
    unknowns = ("UnsupportedPlatform", "ReadUnavailable", "InvalidFormat", "BudgetExceeded",
        "NamespaceOrMountAmbiguous", "ChangedDuringProbe", "HiddenAncestors",
        "WindowsJobHierarchyUnavailable", "WindowsGroupAffinityAmbiguous", "UnobservedThread",
        "UnobservedOrtSettings")

    def one(observed):
        if observed is None:
            return False
        if type(observed) is dict and set(observed) == {"error"}:
            _text(observed["error"])
            return False
        fields(observed, ("actor", "thread_id_present", "affinity", "probe_method", "unknown_reason"),
            "utility cohort actual thread observation")
        actor = _text(observed["actor"])
        if actor not in ("Manager", "NativeCaller") and re.fullmatch(
                r"(?:OwnWorker|ExternalEngine|RuntimeHelper)\([0-9]+\)", actor) is None:
            raise ValueError("utility cohort original CPU actor differs")
        present = _boolean(observed["thread_id_present"], "original thread ID observed")
        if observed["affinity"] is None:
            if observed["probe_method"] is not None or observed["unknown_reason"] not in unknowns:
                raise ValueError("utility cohort unknown resource readback differs")
            return False
        affinity = _array(observed["affinity"], 65536)
        for cpu in affinity:
            if type(cpu) is not list or len(cpu) != 2:
                raise ValueError("utility cohort original CPU affinity pair required")
            uint(cpu[0], 65535)
            uint(cpu[1], 2**32 - 1)
        if observed["unknown_reason"] is not None or observed["probe_method"] not in methods:
            raise ValueError("utility cohort known resource readback differs")
        return present and bool(affinity)

    native_known = one(value["native"])
    own = _array(value["own"], 4)
    own_known = [one(item) for item in own]
    return native_known and bool(own_known) and all(own_known)


def _prelude(value, *, profile, root, authority):
    """Check the actual ready-group preimage; no ready offer is issued here."""
    fields(value, ("captured_tick", "native_credit", "role_credit", "counters", "offers",
        "ready_group_sha256", "actual_resource_observations", "native_thread_readback", "own_thread_readbacks"),
        "utility cohort original ready prelude")
    captured = uint(value["captured_tick"])
    if captured >= root["deadline_tick"]:
        raise ValueError("utility cohort ready prelude exceeds original deadline")
    credit = fields(value["native_credit"], ("limit", "spent", "reserved", "highest_job_sequence", "highest_execution_sequence"),
        "utility cohort original native credit prelude")
    for item in credit.values():
        uint(item)
    if credit["spent"] + credit["reserved"] > credit["limit"]:
        raise ValueError("utility cohort original native prelude credit exceeds limit")
    roles = fields(value["role_credit"], ("spent", "reserved"), "utility cohort original role prelude credit")
    for item in roles.values():
        uint(item)
    counters = fields(value["counters"], _COUNTER_FIELDS, "utility cohort original prelude counters")
    for item in counters.values():
        uint(item)
    resource_known = _resource_observations(value["actual_resource_observations"])
    for name in ("native_thread_readback", "own_thread_readbacks"):
        _text(value[name], MAX_MATERIALIZATION_BYTES)
    offers = _array(value["offers"], 3)
    if not offers:
        raise ValueError("utility cohort original nonempty ready offers required")
    stages = {
        "proposer": ("ProposerRanking", "ProposalContinuation", "Repair", "ProposalValueFresh", "CounterValueFresh",
            "RepairValueFresh", "FreshRepairedEndpoint", "FreshCounterEndpoint", "RecheckRepairedValueFresh"),
        "critic": ("CriticDivergence", "CriticReply", "CounterContinuation", "CriticRecheck"),
        "researcher": ("CpuValidation", "CpuDiscovery", "CounterValidation", "RepairValidation", "RecheckCounterValidation"),
    }
    seen, ready = set(), []
    for offer in offers:
        fields(offer, ("frame_sequence", "recipient", "stage", "original_first_move", "question_sha256", "view",
            "full_view_sha256", "public_context_sha256", "semantic_sha256", "budget", "deadline_clock", "deadline_tick"),
            "utility cohort original ready offer")
        _positive(offer["frame_sequence"])
        route = offer["recipient"]
        if type(route) is not str or route not in _RECIPIENTS or route in seen or offer["stage"] not in stages[route]:
            raise ValueError("utility cohort original ready recipient/stage differs")
        if profile["input"]["mode"] == "cpu_pc" and route == "researcher":
            raise ValueError("utility cohort CPU P-C prelude cannot own native R offer")
        seen.add(route)
        uint(offer["original_first_move"], 65535)
        for name in ("question_sha256", "full_view_sha256", "public_context_sha256", "semantic_sha256"):
            sha(offer[name])
        structure._view(offer["view"], authority)
        budget = fields(offer["budget"], ("role_submissions_remaining", "original_cpu_credit", "original_store_byte_limit"),
            "utility cohort original ready offer budget")
        for item in budget.values():
            uint(item)
        if (uint(offer["deadline_clock"]) != root["deadline_clock"]
                or uint(offer["deadline_tick"]) != root["deadline_tick"]):
            raise ValueError("utility cohort original ready offer deadline differs")
        ready.append((4 + _RECIPIENTS.index(route), offer["semantic_sha256"]))
    preimage = b"rz-rovent-utility-cohort-ready-offer-group/1\0" + len(ready).to_bytes(8, "little")
    for route, semantic in sorted(ready):
        preimage += bytes([route]) + bytes.fromhex(semantic)
    preimage += b"".join(item.to_bytes(8, "little") for item in (
        credit["limit"], credit["spent"], credit["reserved"], roles["spent"], roles["reserved"]))
    # Wait-loop iterations are preserved as original provenance, but only
    # this explicitly documented comparison-allocation digest excludes them.
    allocation_counters = {name: item for name, item in counters.items() if name != "notification_waits"}
    preimage += bytes.fromhex(_value_digest("rz-rovent-utility-cohort-prelude-counters/1", allocation_counters))
    preimage += bytes.fromhex(_value_digest("rz-rovent-utility-cohort-actual-resource-observations/1",
        value["actual_resource_observations"]))
    if hashlib.sha256(preimage).hexdigest() != sha(value["ready_group_sha256"]):
        raise ValueError("utility cohort original actual ready-group preimage differs")
    return resource_known


def _cost(value, profile):
    fields(value, _COST_FIELDS, "utility cohort original whole cost")
    declared = profile["input"]
    if (value["domain"] != COST_DOMAIN or value["scope"] != COST_SCOPE
            or value["objective"] != COST_OBJECTIVE
            or sha(value["cost_class_sha256"]) != declared["cost_class_sha256"]
            or sha(value["resource_plan_sha256"]) != declared["resource_plan_sha256"]
            or _positive(value["resource_epoch"]) != declared["resource_epoch"]):
        raise ValueError("utility cohort original whole cost profile differs")
    if value["cost_epoch_sha256"] is not None:
        sha(value["cost_epoch_sha256"])
    _boolean(value["complete_scope"], "complete whole cost scope")
    ticks = tuple(uint(value[name]) for name in (
        "started_tick", "root_fence_tick", "actor_release_tick", "physical_pin_zero_tick"))
    if (ticks != tuple(sorted(ticks))
            or uint(value["whole_wall_ns"]) != ticks[-1] - ticks[0]
            or uint(value["cohort_action_cutoff_tick"]) != ticks[0] + declared["action_ns"]
            or uint(value["cohort_cleanup_cutoff_tick"]) != value["cohort_action_cutoff_tick"] + declared["cleanup_ns"]):
        raise ValueError("utility cohort original whole cost clock/cutoff differs")
    uint(value["original_deadline_tick"])
    for name in ("cpu_thread_time_ns", "energy"):
        if value[name] is not None:
            uint(value[name])
    for name in ("baseline_counters", "final_counters"):
        fields(value[name], _COUNTER_FIELDS, "utility cohort original work counters")
        for item in value[name].values():
            uint(item)
    if any(value["final_counters"][name] < value["baseline_counters"][name] for name in _COUNTER_FIELDS):
        raise ValueError("utility cohort original whole cost counter reversal")
    for name in ("native_credit_baseline", "native_credit_final"):
        credit = _array(value[name], 2)
        if len(credit) != 2:
            raise ValueError("utility cohort native credit tuple required")
        for item in credit:
            uint(item)
    if value["native_credit_final"][0] < value["native_credit_baseline"][0]:
        raise ValueError("utility cohort native credit consumed total reversal")


def _trial_metadata(files, profile):
    """Check closed original clock/materialization linkage, not proof meaning."""
    publication = files.publication()
    profile_sha = cohort_profile_sha256(profile)
    materializations = {int(pin["path"].split("/")[1][:6]): pin for pin in publication["materializations"]}
    observed = []
    frozen = None
    for ordinal, pin in enumerate(publication["trials"]):
        body = _decode(files.raw_file(pin["path"]), MAX_MATERIALIZATION_BYTES)
        fields(body, _TRIAL_FIELDS, "utility cohort original trial checkpoint")
        if (body["schema"] != TRIAL_DOMAIN or body["cohort_id"] != publication["cohort_id"]
                or _positive(body["trial_sequence"], 3) != ordinal + 1
                or body["profile"] != profile or sha(body["profile_sha256"]) != profile_sha
                or body["recipient"] not in profile["input"]["requested_recipients"]):
            raise ValueError("utility cohort original trial profile/order/recipient differs")
        _source(body["source"], profile)
        _frozen(body["frozen_context"])
        if frozen is None:
            frozen = body["frozen_context"]
        elif any(body["frozen_context"][name] != frozen[name] for name in _FROZEN_FIELDS[:-1]):
            raise ValueError("utility cohort original frozen decision differs between trials")
        if body["frozen_context"]["first_trial_sequence"] > ordinal + 1:
            raise ValueError("utility cohort frozen first trial chronology differs")
        for name in ("resolved", "comparable"):
            _boolean(body[name], name)
        known = _array(body["target_known_mask"], 7)
        if len(known) != 7 or any(type(item) is not bool or item for item in known):
            raise ValueError("utility cohort trial cannot issue a known pair mask")
        reason = body["unknown_reason"]
        if reason is not None and (type(reason) is not str or not reason or len(reason.encode("utf-8")) > 256):
            raise ValueError("utility cohort bounded original unknown reason required")
        if body["comparable"] != (reason is None):
            raise ValueError("utility cohort original comparable/unknown observation differs")
        _cost(body["cost"], profile)
        materialized_pin = materializations.get(ordinal)
        if body["materialized_json"] is None:
            if body["materialized_sha256"] is not None or materialized_pin is not None:
                raise ValueError("utility cohort missing original materialization cannot be synthesized")
        else:
            if type(body["materialized_json"]) is not str or materialized_pin is None:
                raise ValueError("utility cohort original embedded materialization required")
            try:
                original = body["materialized_json"].encode("utf-8", "strict")
            except UnicodeEncodeError as error:
                raise ValueError("utility cohort original materialization UTF8 required") from error
            if (original != files.raw_file(materialized_pin["path"])
                    or sha(body["materialized_sha256"]) != _native_digest(MATERIALIZATION_DOMAIN, original)):
                raise ValueError("utility cohort embedded original materialization raw/native seal differs")
            materialized = _decode(original, MAX_MATERIALIZATION_BYTES)
            fields(materialized, ("schema", "cohort_id", "trial_sequence", "recipient", "profile_sha256",
                "source", "original_root", "prelude", "input", "action", "semantic"),
                "utility cohort original materialization")
            if (materialized["schema"] != MATERIALIZATION_DOMAIN
                    or any(materialized[name] != body[name] for name in (
                        "cohort_id", "trial_sequence", "recipient", "profile_sha256", "source"))):
                raise ValueError("utility cohort original materialization/trial linkage differs")
        observed.append(body)
    return tuple(observed)


def validate_trial_acknowledgements(files):
    """Join ACK4/IO8 with original checkpoint raw pins and native trial seals.

    This deliberately returns detached observations, not live FD-bearing ACKs.
    It does not infer physical pin-zero or independent trial authority.
    """
    if type(files) is not HistoricalUtilityCohortFiles:
        raise ValueError("utility cohort original pinned file owner required")
    publication, ack = files.publication(), files.durable_ack()
    acknowledgements = _array(ack["trial_acknowledgements"], 3)
    if len(acknowledgements) != len(publication["trials"]):
        raise ValueError("utility cohort original trial ACK complete coverage required")
    materialized = {int(pin["path"].split("/")[1][:6]): pin for pin in publication["materializations"]}
    previous_finish = None
    result = []
    for ordinal, (item, pin) in enumerate(zip(acknowledgements, publication["trials"])):
        fields(item, ("trial_sequence", "native_trial_sha256", "checkpoint", "collection_io"),
            "utility cohort original trial ACK")
        fields(item["checkpoint"], ("path", "sha256", "bytes"), "utility cohort original trial ACK pin")
        if (uint(item["trial_sequence"], 3) != ordinal + 1
                or item["checkpoint"] != pin or type(item["checkpoint"]["path"]) is not str):
            raise ValueError("utility cohort original trial ACK checkpoint/order differs")
        sha(item["checkpoint"]["sha256"])
        _positive(item["checkpoint"]["bytes"], MAX_MATERIALIZATION_BYTES)
        raw = files.raw_file(pin["path"])
        body = _decode(raw, MAX_MATERIALIZATION_BYTES)
        if (body.get("schema") != TRIAL_DOMAIN or type(body.get("trial_sequence")) is not int
                or body["trial_sequence"] != ordinal + 1 or body.get("cohort_id") != publication["cohort_id"]
                or sha(item["native_trial_sha256"]) != _native_digest(TRIAL_DOMAIN, raw)):
            raise ValueError("utility cohort original trial ACK native body seal differs")
        interval = _io(item["collection_io"])
        materialized_bytes = materialized.get(ordinal, {}).get("bytes", 0)
        if (interval["clock_epoch"] != ack["collection_io"]["clock_epoch"]
                or interval["finished_tick_ns"] > ack["collection_io"]["started_tick_ns"]
                or previous_finish is not None and previous_finish > interval["started_tick_ns"]
                or interval["written_bytes"] != materialized_bytes + pin["bytes"]
                or interval["readback_bytes"] != materialized_bytes + 2 * pin["bytes"]
                or interval["synced_files"] != 2 + int(materialized_bytes > 0)
                or interval["synced_directories"] != 3 * (1 + int(materialized_bytes > 0))):
            raise ValueError("utility cohort original trial ACK I/O accounting differs")
        previous_finish = interval["finished_tick_ns"]
        result.append(copy.deepcopy(item))
    return tuple(result)


def validate_cohort_payloads(files, *, preparation):
    """Consume only the independently registered cohort/PREP2 path.

    The source enrollment and original conditional witness proof are still
    being connected in their Rust producer. Until their closed body is
    registered, preserving a typed unknown is required; no report or known
    comparison is fabricated from raw pins, ticks or root-local evidence.
    """
    checked = validate_training_preparation(preparation)
    if checked["schema"] != COHORT_PREPARATION_DOMAIN:
        raise ValueError("utility cohort requires explicit preparation/2")
    if type(files) is not HistoricalUtilityCohortFiles:
        raise ValueError("utility cohort original pinned file owner required")
    profile = validate_cohort_profile(_decode(files.raw_file("cohort-profile.json"), MAX_PROFILE_BYTES))
    validate_trial_acknowledgements(files)
    _trial_metadata(files, profile)
    # A separately reviewed source enrollment is intentionally not inferred
    # from a detached profile or an old root-local action's provenance.
    raise UnsupportedCohortPayload("cohort_source_enrollment_not_yet_registered")
