"""Strict historical-byte handoff for the separate fresh replay constructor.

This adapter reuses Query/2, prepared action/1 and semantic/1 admission. It
requires the caller's seven ORIGINAL semantic buffers and independent pins;
serializing rules_receipt() cannot recover those originals. The same absolute
deadline and current strict parent are checked on every use. No I/O, Rust wire
codec, launcher, Rules replay, model operation or execution authority is added.

ReplayOriginals.registration is a NEW ReplayConsumerRegistration, not the old
semantic registration. Only prepared_action and semantic_receipt are available
here. New replay registration/build/source/binary/factory pins remain external.
The returned historical bindings are a partial constructor handoff, never a
complete ReplayExpectedPins or a 2N-to-3N resource conversion.

12 MiB originals, 64 KiB pin metadata and 8 MiB adapter serialization scratch
are this adapter's handoff ledger only. Existing capability resident storage,
their validators' workspaces, Python object overhead and caller RSS are outside
that ledger. Repeated references are charged separately; no buffer is trimmed.
Caller image-scope registration remains conditional assurance, not an observed
loaded executable. Utility/target/training groups remain zero.
"""

import hashlib
import math
import time

from . import semantic_verifier as semantic
from . import strategic_cpu_action as action
from . import strategic_verifier_query as query
from .training import ValidatedDataset


SCHEMA = "rz-pals-strict-replay-caller-bindings/1"
MAX_RAW_BYTES = 12 * 1024 * 1024
MAX_PIN_METADATA_BYTES = 64 * 1024
MAX_SCRATCH_BYTES = 8 * 1024 * 1024
SEMANTIC_RAW_NAMES = ("request", "receipt", "source", "registration", "before_result",
                      "launch_observation", "common_query")
PARENT_PIN_NAMES = ("parent_input_sha256", "current_view_sha256", "frozen_admission_sha256", "encoding_sha256")
_SCOPES = {"linux_loaded_executable_inode": "LinuxLoadedExecutableInode",
           "current_exe_path_hash": "CurrentExePathHash"}
_EXTERNAL = ("replay_consumer_registration_original_bytes_and_independent_pin",
             "new_replay_binary_and_source_build_registration",
             "engine_wrapper_replay_provider_factory_registered_artifacts",
             "complete_replay_expected_pins", "replay_preparation_mode_config_resources",
             "actual_loaded_process_and_final_closure_observation")
_FACTORY = object()
_DENIED = ("independent_registration_created", "loaded_image_observed", "rust_outer_wire_created",
           "rust_query_revalidated", "rust_semantic_capability_revalidated", "actual_dispatch_observed",
           "native_action_causal_bridge_observed", "whole_action_cost_observed", "final_search_closure_observed",
           "utility_authority", "target_authority", "training_authority", "model_executed",
           "product_verifier_enabled", "backward_executed", "optimizer_created", "gpu_used")


class ReplayCallerRefusal(ValueError):
    """A refused preparation, not a masked or failed execution observation."""
    def __init__(self, code, stage):
        self.code, self.stage = code, stage
        super().__init__("strict replay caller refusal: " + code + " at " + stage)


def _refuse(code, stage):
    raise ReplayCallerRefusal(code, stage)


def _deadline(value, stage):
    if type(value) not in (int, float):
        _refuse("invalid_absolute_deadline", stage)
    try:
        finite = math.isfinite(value)
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        _refuse("invalid_absolute_deadline", stage)
    if time.monotonic() >= value:
        _refuse("expired_original_deadline", stage)


def _fields(value, names, stage):
    # Never materialize a caller's arbitrary key set before checking its extent.
    if type(value) is not dict or len(value) != len(names) or any(name not in value for name in names):
        _refuse("closed_pin_metadata_required", stage)


def _sha(value, stage):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _refuse("lowercase_sha256_required", stage)


def _raw(raw, maximum, stage):
    if type(raw) is not bytes or not 1 <= len(raw) <= maximum:
        _refuse("original_immutable_bytes_required", stage)
    return len(raw)


def _pins(pins, raws, deadline):
    _fields(pins, (*SEMANTIC_RAW_NAMES, *PARENT_PIN_NAMES), "semantic_pins")
    for name, raw in zip(SEMANTIC_RAW_NAMES, raws):
        pin = pins[name]
        _fields(pin, ("bytes", "sha256"), "semantic_pin_" + name)
        if type(pin["bytes"]) is not int or not 1 <= pin["bytes"] <= semantic.MAX_INPUT_BYTES:
            _refuse("strict_pin_byte_extent", name)
        _sha(pin["sha256"], name)
        if pin["bytes"] != len(raw):
            _refuse("independent_original_pin_mismatch", name)
    for name in PARENT_PIN_NAMES:
        _sha(pins[name], name)
    # Finish every closed metadata/extent check before hashing originals. A
    # same-length mutation must refuse before getters, canonical copies or the
    # existing semantic factory; its later pin audit is not a preflight guard.
    for name, raw in zip(SEMANTIC_RAW_NAMES, raws):
        _deadline(deadline, "original_hash_" + name)
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        _deadline(deadline, "original_hash_" + name + "_end")
        if actual_sha256 != pins[name]["sha256"]:
            _refuse("independent_original_pin_mismatch", name)


def _cap_extents(checked, prepared):
    # Read only immutable backing extents before getters clone metadata. The
    # existing factories still own all meaning and integrity validation below.
    raw_total = _raw(checked._catalogue, query.MAX_RAW_BYTES, "query_catalogue")
    raw_total += _raw(checked._before, query.MAX_RAW_BYTES, "query_before")
    metadata = _raw(checked._pins, MAX_PIN_METADATA_BYTES, "query_pin_metadata")
    if type(checked._profiles) is not tuple or not 1 <= len(checked._profiles) <= query.MAX_ACTIONS:
        _refuse("query_backing_extent", "profiles")
    if type(checked._prior) is not tuple or len(checked._prior) > query.MAX_PRIOR:
        _refuse("query_backing_extent", "prior")
    for bundles, names in ((checked._profiles, ("registration", "source", "capabilities")),
                           (checked._prior, ("observation", "request", "receipt", "stdout", "stderr", "launch", "semantic_input"))):
        for bundle in bundles:
            if type(bundle) is not tuple or len(bundle) != len(names):
                _refuse("query_backing_extent", "bundle")
            seen = set()
            for entry in bundle:
                if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not str or entry[0] not in names or entry[0] in seen:
                    _refuse("query_backing_extent", "bundle_fields")
                name, value = entry
                seen.add(name)
                if name == "semantic_input" or (name in ("receipt", "launch") and value is None):
                    continue
                if name in ("stdout", "stderr") and type(value) is bytes and not value:
                    continue
                raw_total += _raw(value, query.MAX_RAW_BYTES, "query_" + name)
    raw_total += _raw(prepared._request, action.MAX_CPU_REQUEST_BYTES, "prepared_cpu_request")
    raw_total += _raw(prepared._envelope, action.MAX_CPU_REQUEST_BYTES, "prepared_envelope")
    metadata += _raw(prepared._pin_raw, MAX_PIN_METADATA_BYTES, "prepared_pin_metadata")
    return raw_total, metadata


def _preflight(parents, parent_index, checked, index, prepared, raws, pins, scope, deadline):
    _deadline(deadline, "preflight")
    if type(raws) is not tuple or len(raws) != len(SEMANTIC_RAW_NAMES):
        _refuse("seven_original_buffers_required", "semantic_originals")
    semantic_total = sum(_raw(raw, semantic.MAX_INPUT_BYTES, name) for name, raw in zip(SEMANTIC_RAW_NAMES, raws))
    if semantic_total > semantic.MAX_INPUT_BYTES:
        _refuse("semantic_original_aggregate_extent", "semantic_originals")
    _pins(pins, raws, deadline)
    if type(scope) is not str or scope not in _SCOPES:
        _refuse("unsupported_independent_image_scope", "scope")
    if type(checked) is not query.CheckedStrategicQuery or type(prepared) is not action.PreparedStrategicCpuAction:
        _refuse("exact_checked_capabilities_required", "capabilities")
    if type(parents) is not ValidatedDataset or type(parent_index) is not int or type(index) is not int:
        _refuse("exact_parent_and_indices_required", "parent")
    if not 0 <= index < query.MAX_ACTIONS or parent_index < 0:
        _refuse("strict_selected_index", "selection")
    try:
        query_parent, query_index = checked._parent, checked._index
        prepared_query, prepared_index = prepared._query, prepared._index
    except AttributeError as error:
        raise ReplayCallerRefusal("initialized_factory_capabilities_required", "capabilities") from error
    if query_parent is not parents or query_index != parent_index:
        _refuse("query_parent_object_mismatch", "parent")
    if prepared_query is not checked or prepared_index != index:
        _refuse("prepared_selected_query_mismatch", "selection")
    raw_credit, metadata_credit = _checked_call(deadline, "capability_extents", _cap_extents, checked, prepared)
    raw_credit += semantic_total
    if raw_credit > MAX_RAW_BYTES:
        _refuse("handoff_raw_credit_exceeded", "preflight")
    # Exact nested pins have bounded scalar sizes before canonical() is called.
    # Charge even repeated references at each occurrence, with no alias credit.
    available_metadata = MAX_PIN_METADATA_BYTES - metadata_credit
    if available_metadata < 0:
        _refuse("handoff_pin_credit_exceeded", "preflight")
    try:
        query._tree(pins, byte_credit=[available_metadata])
    except ValueError as error:
        raise ReplayCallerRefusal("handoff_pin_credit_exceeded", "preflight") from error
    # Own getter/serialization buffers are reserved separately from resident
    # capabilities and their existing validator workspaces; this is not RSS.
    scratch = 2 * available_metadata + len(checked._catalogue) + len(prepared._envelope) + 2 * semantic_total
    if scratch > MAX_SCRATCH_BYTES:
        _refuse("handoff_scratch_credit_exceeded", "preflight")
    _deadline(deadline, "preflight_end")
    return raw_credit, metadata_credit, scratch, available_metadata


def _checked_call(deadline, stage, fn, *args, **kwargs):
    _deadline(deadline, stage)
    try:
        result = fn(*args, **kwargs)
    except ReplayCallerRefusal:
        raise
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, TimeoutError) as error:
        raise ReplayCallerRefusal("existing_strict_admission_refused", stage) from error
    _deadline(deadline, stage + "_end")
    return result


def _validate(parents, parent_index, checked, index, prepared, raws, pins, scope, deadline):
    credit = _preflight(parents, parent_index, checked, index, prepared, raws, pins, scope, deadline)
    pins_raw = _checked_call(deadline, "pin_copy", query.canonical, pins, max_bytes=credit[3])
    detached_pins = _checked_call(deadline, "pin_parse", query._parse, pins_raw)
    _checked_call(deadline, "query_verify", checked.verify)
    assets = _checked_call(deadline, "query_assets", checked.raw_assets)
    catalogue = _checked_call(deadline, "catalogue", checked.catalogue)
    semantics = _checked_call(deadline, "semantic_slots", checked.action_semantic_inputs)
    if index >= len(catalogue["actions"]) or index >= len(semantics):
        _refuse("strict_selected_index", "selection")
    selected = semantics[index]
    if type(selected) is not semantic.CheckedSemanticInput:
        _refuse("exact_selected_semantic_required", "semantic_slot")
    if selected._parent is not parents or selected._index != parent_index:
        _refuse("selected_semantic_parent_object_mismatch", "semantic_slot")
    if catalogue["actions"][index]["task"] != "defend_response":
        _refuse("unsupported_replay_task", "selection")
    _checked_call(deadline, "prepared_verify", prepared.verify)
    prepared_assets = _checked_call(deadline, "prepared_assets", prepared.raw_assets)
    envelope = _checked_call(deadline, "prepared_envelope", prepared.envelope)
    query_sha = _checked_call(deadline, "query_identity", lambda: checked.sha256)
    if (envelope["query_sha256"] != query_sha or envelope["action"] != catalogue["actions"][index]
            or envelope["catalogue_artifact"] != query.byte_pin(assets["catalogue"])
            or envelope["before_result_artifact"] != query.byte_pin(assets["before_result"])
            or envelope["cpu_request_raw"].encode("utf-8") != prepared_assets["cpu_request"]
            or envelope["cpu_request_artifact"] != prepared_assets["expected_cpu_request_pin"]
            or query.byte_pin(prepared_assets["cpu_request"]) != prepared_assets["expected_cpu_request_pin"]):
        _refuse("prepared_original_binding_mismatch", "prepared")
    rebound = _checked_call(deadline, "original_semantic_readmission", semantic.admit_semantic_input,
        parent=parents, parent_index=parent_index, expected_pins=detached_pins,
        **{name + "_bytes": raw for name, raw in zip(SEMANTIC_RAW_NAMES, raws)})
    selected_sha = _checked_call(deadline, "selected_semantic_identity", lambda: selected.sha256)
    rebound_sha = _checked_call(deadline, "rebound_semantic_identity", lambda: rebound.sha256)
    if (selected_sha != rebound_sha or selected_sha != catalogue["actions"][index]["semantic_input_sha256"]
            or selected_sha != assets["expected_pins"]["action_semantic_sha256"][index]):
        _refuse("selected_original_semantic_mismatch", "semantic_slot")
    if _checked_call(deadline, "semantic_parent", rebound.verify_parent, parents) != parent_index:
        _refuse("selected_parent_index_mismatch", "parent")
    parent_pins = _checked_call(deadline, "semantic_parent_pins", rebound.parent_pins)
    if parent_pins != assets["expected_pins"]["parent"] or any(detached_pins[name] != parent_pins[name] for name in PARENT_PIN_NAMES):
        _refuse("current_frozen_encoding_mismatch", "parent")
    common = _checked_call(deadline, "semantic_common", rebound.common_query)
    registration = _checked_call(deadline, "original_registration", semantic._parse, raws[3])
    if registration["accepted_binary_pin_scope"] != scope:
        _refuse("independent_image_scope_mismatch", "scope")
    binding = {"query_sha256": query_sha, "action_index": index, "semantic_input_sha256": selected_sha,
        "prepared_action_artifact": query.byte_pin(prepared_assets["envelope"]),
        "semantic_receipt_artifact": query.byte_pin(raws[1]), "parent": parent_pins,
        "semantic_context_sha256": common["semantic_context_sha256"],
        "branch_meaning_sha256": common["branch_meaning_sha256"],
        "before_result_anchor_sha256": common["before_result_anchor_sha256"],
        "semantic_binary_sha256": registration["binary_sha256"], "accepted_semantic_binary_pin_scope": scope}
    remaining_metadata = MAX_PIN_METADATA_BYTES - credit[1] - len(pins_raw)
    remaining_scratch = MAX_SCRATCH_BYTES - credit[2]
    if min(remaining_metadata, remaining_scratch) < 0:
        _refuse("handoff_metadata_or_scratch_credit_exceeded", "binding_identity")
    identity_raw = _checked_call(deadline, "binding_identity", query.canonical,
        [SCHEMA, {"semantic_pins": detached_pins, "bindings": binding}],
        max_bytes=min(remaining_metadata, remaining_scratch))
    metadata_used = credit[1] + len(pins_raw) + len(identity_raw)
    scratch_used = credit[2] + len(identity_raw)
    if metadata_used > MAX_PIN_METADATA_BYTES or scratch_used > MAX_SCRATCH_BYTES:
        _refuse("handoff_metadata_or_scratch_credit_exceeded", "binding_identity")
    _checked_call(deadline, "final_query_verify", checked.verify)
    _checked_call(deadline, "final_prepared_verify", prepared.verify)
    _checked_call(deadline, "final_semantic_verify", rebound.verify)
    _deadline(deadline, "handoff_end")
    return binding, pins_raw, identity_raw, _ledger(credit[0], metadata_used, scratch_used)


def _ledger(raw, metadata, scratch):
    return {"original_raw_bytes": raw, "pin_metadata_bytes": metadata, "adapter_scratch_reserved_bytes": scratch,
        "limits": {"original_raw_bytes": MAX_RAW_BYTES, "pin_metadata_bytes": MAX_PIN_METADATA_BYTES,
                   "adapter_scratch_bytes": MAX_SCRATCH_BYTES}, "scope": "own_handoff_ledger_only_not_resident_or_rss_cap"}


class CheckedReplayCallerBindings:
    """Factory-only historical handoff; every getter uses the original expiry."""
    __slots__ = ("_parents", "_parent_index", "_query", "_action_index", "_prepared", "_raws", "_pins_raw",
                 "_scope", "_deadline", "_identity")

    def __init__(self, token=None, *, parents=None, parent_index=None, checked=None, index=None,
                 prepared=None, raws=None, pins_raw=None, scope=None, deadline=None, identity=None):
        if token is not _FACTORY:
            _refuse("factory_only_checked_bindings", "constructor")
        for name, value in (("_parents", parents), ("_parent_index", parent_index), ("_query", checked),
                            ("_action_index", index), ("_prepared", prepared), ("_raws", raws), ("_pins_raw", pins_raw),
                            ("_scope", scope), ("_deadline", deadline), ("_identity", identity)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("checked replay caller bindings are immutable")

    def _views(self):
        _deadline(self._deadline, "verify_start")
        _raw(self._pins_raw, MAX_PIN_METADATA_BYTES, "owned_pin_metadata")
        pins = _checked_call(self._deadline, "owned_pin_parse", query._parse, self._pins_raw)
        values = _validate(self._parents, self._parent_index, self._query, self._action_index, self._prepared,
                           self._raws, pins, self._scope, self._deadline)
        if hashlib.sha256(values[2]).hexdigest() != self._identity:
            _refuse("handoff_identity_changed", "verify_end")
        _deadline(self._deadline, "verify_end")
        return values

    def verify(self):
        self._views()
        return self

    @property
    def sha256(self):
        self.verify()
        return self._identity

    def raw_assets(self):
        values = self._views()
        result = {"semantic_originals": dict(zip(SEMANTIC_RAW_NAMES, self._raws)),
                  "expected_semantic_pins": query._parse(values[1]),
                  "prepared_action": self._prepared._envelope, "cpu_request": self._prepared._request,
                  "expected_cpu_request_pin": query._parse(self._prepared._pin_raw)}
        _deadline(self._deadline, "raw_assets_end")
        return result

    def constructor_inputs(self):
        """Partial Rust argument mapping only; no ReplayOriginals.registration."""
        values = self._views()
        result = {"originals_available": {"prepared_action": self._prepared._envelope, "semantic_receipt": self._raws[1]},
                  "original_semantic_registration": self._raws[3], "checked_historical_bindings": values[0],
                  "semantic_receipt_producer_scope_variant": _SCOPES[self._scope],
                  "required_external": tuple(_EXTERNAL), "complete_replay_expected_pins": False}
        _deadline(self._deadline, "constructor_inputs_end")
        return result

    def audit(self):
        values = self._views()
        result = {"schema": SCHEMA, "scope": "strict_historical_caller_handoff_only", "prepared": True,
                "ledger": values[3], "original_absolute_deadline": self._deadline,
                "semantic_caller_assurance_scope": semantic.ASSURANCE_SCOPE,
                "actual_utility_groups": 0, "required_external": tuple(_EXTERNAL),
                **{name: False for name in _DENIED}}
        _deadline(self._deadline, "audit_end")
        return result


def prepare_replay_caller_bindings(*, parents, parent_index, checked_query, action_index, prepared_action,
                                   semantic_raws, expected_semantic_pins, expected_accepted_binary_pin_scope, deadline):
    values = _validate(parents, parent_index, checked_query, action_index, prepared_action, semantic_raws,
                       expected_semantic_pins, expected_accepted_binary_pin_scope, deadline)
    _deadline(deadline, "factory_end")
    return CheckedReplayCallerBindings(_FACTORY, parents=parents, parent_index=parent_index, checked=checked_query,
        index=action_index, prepared=prepared_action, raws=semantic_raws, pins_raw=values[1],
        scope=expected_accepted_binary_pin_scope, deadline=deadline, identity=hashlib.sha256(values[2]).hexdigest())
