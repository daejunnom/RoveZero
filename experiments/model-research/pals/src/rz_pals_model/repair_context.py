"""Conditional causal admission for a captured first native P Repair decision.

The existing strict ordinary loader owns raw/current/producer/encoding checks.
This adapter adds actual prepared-lineage, physical-output and search-consumed
observations plus registered Rust Rules-only prefix/claim capabilities. Its
context is derived from existing prepared bytes, never retroactively called a
before-dispatch descriptor. Caller launch/registration pins remain conditional
assurance. Native predictions and a legal proposed repair are not repair truth.

Only the existing owned-CPU ordinal comparison of two legal NEXT moves at the
exact ordinary Repair input may be bound. No continuation strength, C pointer
ranking, counterexample validity, WDL, training or product V is admitted.
"""
import copy
import math
import struct
import time

from . import comparative_training as candidate
from . import training
from .native_divergence import verify_native_collection_authority
from .semantic_verifier import CheckedSemanticInput, byte_pin, canonical, digest, _fields, _int, _parse, _sha

CONTEXT_SCHEMA = "rz-pals-native-initial-repair-context/1"
ADMISSION_SCHEMA = "rz-pals-native-initial-repair-admission/1"
PAIR_SCHEMA = "rz-pals-native-next-repair-candidate-pairs/1"
PREPARATION_SCHEMA = "rz-pals-native-next-repair-preparation/1"
SCOPE = "checked_existing_prepared_lineage;first_repair_decision_only;next_move_ordinal_only;no_strategic_repair_or_wdl"
MAX_BYTES = 128 * 1024 * 1024
_FACTORY = object()
_NAMES = ("receipt.json", "producer-roster.json", "producer-envelope.json", "producer-prepared.jsonl",
          "inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl", "public-record-sources.jsonl",
          "native-events.jsonl", "native-raw-outputs.jsonl")
_AUTH_PINS = ("artifacts", "registration", "checked_source", "export_manifest", "launch", "parent_input_sha256",
              "current_view_sha256", "frozen_admission_sha256")
_LINEAGE = ("input_sha256", "game_id", "process_epoch", "request_sequence", "native_query_kind", "actual_played_history",
            "virtual_prefix", "proposal", "counterexample", "divergence_plies", "actual_outcome_eligible",
            "counterfactual_wdl", "training_admission")


def _rows(raw):
    rows = training._collection_jsonl(raw, max_rows=65536 * 8)
    if any(type(value) is not dict for _, value in rows):
        raise ValueError("actual native object rows required")
    return rows


def _one(rows, predicate, name):
    found = [(raw, value) for raw, value in rows if predicate(value)]
    if len(found) != 1:
        raise ValueError("missing/ambiguous actual Repair " + name)
    return found[0]


def _moves(values, *, minimum=0, maximum=64):
    if type(values) is not list or not minimum <= len(values) <= maximum:
        raise ValueError("bounded exact Repair move sequence required")
    for movement in values:
        _int(movement, 0, 65535)
        training.move_components(movement)
    return values


def _pin(raw, expected):
    _fields(expected, ("bytes", "sha256"))
    _int(expected["bytes"], 0, MAX_BYTES)
    _sha(expected["sha256"])
    if type(raw) is not bytes or byte_pin(raw) != expected:
        raise ValueError("actual Repair bytes differ from independent pin")


def _fp32(value):
    if type(value) not in (float, int):
        raise ValueError("native query must contain finite FP32 values")
    try:
        encoded = struct.pack("<f", value)
    except (OverflowError, struct.error) as error:
        raise ValueError("native query must contain finite FP32 values") from error
    if not math.isfinite(struct.unpack("<f", encoded)[0]):
        raise ValueError("native query must contain finite FP32 values")
    return encoded


def _native_query(tensor, snapshot, lineage, kind):
    query = tensor["query"]
    if type(query) is not list or len(query) != 16 or tensor["divergence_features"] != []:
        raise ValueError("ordinary captured native role query required")
    prefix, proposal, counter = lineage["virtual_prefix"], lineage["proposal"], lineage["counterexample"]
    expected = [0.0] * 16
    expected[:7] = [2 if kind == "Repair" else 0, len(prefix) / 256, len(proposal) / 256,
                    int(counter is not None), 0 if counter is None else len(counter) / 256,
                    len(snapshot["legal_moves"]) / 256, snapshot["input_revision"]]
    # Preserve the captured remaining clock feature. It is never recomputed.
    if struct.unpack("<f", _fp32(query[7]))[0] < 0:
        raise ValueError("negative captured native remaining allowance")
    expected[7] = query[7]
    expected[8], expected[15] = int(snapshot["white_to_move"]), 1
    for offset, movement in ((9, prefix[-1] if prefix else None), (12, proposal[0] if proposal else None)):
        if movement is not None:
            source, target, promotion = training.move_components(movement)
            expected[offset:offset + 3] = [source / 63, target / 63, promotion / 4]
    if any(_fp32(actual) != _fp32(want) for actual, want in zip(query, expected)):
        raise ValueError("native query features disagree with exact Repair lineage")


def _events_and_output(artifacts, frozen, lineage, tensor, kind):
    identity, snapshot = frozen["sha256"], frozen["snapshot"]
    key = (identity, lineage["process_epoch"], lineage["request_sequence"])
    matches = lambda value: (value.get("input_sha256"), value.get("process_epoch"), value.get("request_sequence")) == key
    events = [value for _, value in _rows(artifacts["native-events.jsonl"]) if matches(value)]
    for event in events:
        _fields(event, ("domain", "game_id", "process_epoch", "request_sequence", "input_sha256", "stage", "observer_elapsed_us", "detail"))
        if event["domain"] != "rz-pals-native-call-event/1" or event["game_id"] != snapshot["game_id"]:
            raise ValueError("native Repair event attribution")
        for name in ("process_epoch", "request_sequence", "observer_elapsed_us"):
            _int(event[name], 0, 2**64 - 1)
    if [event["stage"] for event in events] != ["prepared", "physically_completed", "delivered", "search_consumed"]:
        raise ValueError("Repair requires known physical, delivered and search-consumed events")
    for event, detail in zip(events, (
            {"prepared_before_submit": True, "native_query_kind": kind, "producer_metadata_admitted": True},
            {"success": True, "logical_acceptance_inferred": False},
            {"search_consumed": False}, {"search_consumed": True})):
        if canonical(event["detail"]) != canonical(detail):
            raise ValueError("Repair event cannot infer logical acceptance from physical completion")
    elapsed = [event["observer_elapsed_us"] for event in events]
    if elapsed != sorted(elapsed):
        raise ValueError("Repair native event chronology")
    _, output = _one(_rows(artifacts["native-raw-outputs.jsonl"]), matches, "physical raw output")
    _fields(output, ("domain", "process_epoch", "request_sequence", "input_sha256", "physical_completion_confirmed", "success", "raw"))
    for name in ("process_epoch", "request_sequence"):
        _int(output[name], 0, 2**64 - 1)
    _sha(output["input_sha256"])
    if (output["domain"] != "rz-pals-native-physical-raw/1" or output["physical_completion_confirmed"] is not True
            or output["success"] is not True):
        raise ValueError("Repair successful known native physical raw output required")
    raw = _fields(output["raw"], ("representation", "candidate_logits_bits", "wdl_logits_bits", "divergence_logits_bits",
                                  "task_logits_bits", "private_latent_bits", "prediction_is_future_label"))
    if (raw["representation"] != "f32_ieee754_bits" or raw["prediction_is_future_label"] is not False
            or raw["divergence_logits_bits"] is not None or raw["task_logits_bits"] is not None):
        raise ValueError("ordinary P predictions must remain predictions")
    for name, count in (("candidate_logits_bits", len(tensor["candidates"])), ("wdl_logits_bits", 3), ("private_latent_bits", 6144)):
        if type(raw[name]) is not list or len(raw[name]) != count:
            raise ValueError("Repair native raw head shape")
        for bits in raw[name]:
            _int(bits, 0, 2**32 - 1)
            if not math.isfinite(struct.unpack("<f", struct.pack("<I", bits))[0]):
                raise ValueError("Repair native raw head nonfinite")
    return elapsed


def _ordinary(parents, index, artifacts, source, kind):
    _int(index, 0, len(parents.records) - 1)
    if index not in parents.current_view.current_indices:
        raise ValueError("Repair anchors must be actual current ordinary rows")
    row = parents.records[index]
    frozen, snapshot = row["input"], row["input"]["snapshot"]
    identity = frozen["sha256"]
    if snapshot["role"] != "proposer" or snapshot["source"]["kind"] != "own_pals":
        raise ValueError("current owned native P input required")
    input_raw, actual = _one(_rows(artifacts["inputs.jsonl"]), lambda value: value.get("sha256") == identity, "ordinary input")
    sidecar_raw, sidecar = _one(_rows(artifacts["native-inputs.jsonl"]), lambda value: value.get("input_sha256") == identity, "ordinary sidecar")
    lineage_raw, lineage = _one(_rows(artifacts["input-lineage.jsonl"]), lambda value: value.get("input_sha256") == identity, "ordinary lineage")
    _fields(lineage, _LINEAGE)
    for name in ("process_epoch", "request_sequence"):
        _int(lineage[name], 0, 2**64 - 1)
    _moves(lineage["actual_played_history"], maximum=65536)
    _moves(lineage["virtual_prefix"])
    _moves(lineage["proposal"])
    if lineage["counterexample"] is not None:
        _moves(lineage["counterexample"])
    if (actual != frozen or lineage["game_id"] != snapshot["game_id"] or lineage["native_query_kind"] != kind
            or lineage["actual_played_history"] != snapshot["actual_history"] or lineage["divergence_plies"] != []
            or lineage["training_admission"] != "ordinary_role" or lineage["counterfactual_wdl"] != "masked"
            or lineage["actual_outcome_eligible"] is not (kind == "Propose")):
        raise ValueError("exact native ordinary Propose/Repair lineage required")
    admitted = next((entry for entry in parents.frozen_admission["inputs"] if entry["binding"]["input_sha256"] == identity), None)
    if admitted is None:
        raise ValueError("ordinary input lacks strict actual producer binding")
    _, journal = _one(_rows(artifacts["producer-prepared.jsonl"]),
                      lambda value: type(value.get("prepared")) is dict and value["prepared"].get("input_sha256") == identity, "true prepared journal")
    _fields(journal, ("prepared", "sha256"))
    prepared = journal["prepared"]
    if (journal["sha256"] != admitted["prepared_evidence_sha256"]
            or journal["sha256"] != training._sorted_canonical(training.PREPARED_PRODUCER_DOMAIN, prepared)
            or prepared["learning_input"] is not True
            or prepared["native_request"] != [lineage["process_epoch"], lineage["request_sequence"]]
            or any(prepared[name] != byte_pin(raw) for name, raw in (("input_json", input_raw), ("tensor_sidecar_json", sidecar_raw), ("lineage_json", lineage_raw)))):
        raise ValueError("actual ordinary Repair prepared byte anchors required")
    policy = admitted["producer"]["encoding_policy"]
    encoded = training.encoded_from_sidecar(sidecar, row, expected_encoder_source_sha256=policy["encoder_source_sha256"],
                                           expected_model_epoch=policy["native_model_epoch"]["sha256"])
    if encoded != parents.encodings[identity]:
        raise ValueError("ordinary Repair must use its own captured encoding")
    tensor = _parse(sidecar["tensor_json"].encode(), MAX_BYTES)
    _native_query(tensor, snapshot, lineage, kind)
    sources = _rows(artifacts["public-record-sources.jsonl"]) if artifacts["public-record-sources.jsonl"] else []
    for token, observation in zip(tensor["records"], snapshot["public_records"]):
        _, public = _one(
            [(raw, value) for raw, value in sources if byte_pin(raw)["sha256"] == observation["observation_sha256"]], lambda _: True, "public source")
        _fields(public, ("domain", "game_id", "record_index", "revision", "origin_state_id", "origin_state_id_is_advisory",
                         "origin_rules_state_sha256", "origin_rules_identity_observation", "kind", "line", "value",
                         "completed_depth", "scope", "white_score_perspective", "critical", "source_cpu_profile_sha256"))
        if (public.get("domain") != "rz-pals-native-public-source/1" or public.get("game_id") != snapshot["game_id"]
                or public.get("record_index") != token["record_id"] or public.get("revision") != token["revision"]
                or public.get("critical") is not token["critical"] or public.get("source_cpu_profile_sha256") != source["cpu_profile_sha256"]
                or public["origin_state_id_is_advisory"] is not True or public["origin_rules_state_sha256"] is not None
                or public["origin_rules_identity_observation"] != "unknown"):
            raise ValueError("Repair selected public source attribution")
        # Public origin IDs/CP values are advisory. They are never Rules truth.
    elapsed = _events_and_output(artifacts, frozen, lineage, tensor, kind)
    return row, lineage, journal, admitted["producer"], elapsed


def _validate(parents, root_index, repair_index, artifacts, raws, checks, pins):
    _fields(artifacts, _NAMES)
    _fields(pins, (*_AUTH_PINS, "context", "repair_input_sha256", "rules_prefix_sha256", "rules_proposal_sha256"))
    if any(type(raw) is not bytes for raw in (*artifacts.values(), *raws)) or sum(map(len, (*artifacts.values(), *raws))) > MAX_BYTES:
        raise ValueError("aggregate immutable Repair byte budget")
    registration, source_raw, export_raw, launch, context_raw = raws
    authority = verify_native_collection_authority(parent=parents, parent_index=root_index, artifacts=artifacts,
        registration_bytes=registration, checked_source_bytes=source_raw, export_manifest_bytes=export_raw,
        launch_bytes=launch, expected_pins={name: pins[name] for name in _AUTH_PINS})
    repair_pins = {name: pins[name] for name in _AUTH_PINS}
    repair_pins["parent_input_sha256"] = _sha(pins["repair_input_sha256"])
    repair_authority = verify_native_collection_authority(parent=parents, parent_index=repair_index, artifacts=artifacts,
        registration_bytes=registration, checked_source_bytes=source_raw, export_manifest_bytes=export_raw,
        launch_bytes=launch, expected_pins=repair_pins)
    if authority.producer_json != repair_authority.producer_json:
        raise ValueError("causal Repair must retain its root producer; game roster may contain other models")
    source = _parse(authority.source_json)
    root, root_lineage, root_journal, _, root_elapsed = _ordinary(parents, root_index, artifacts, source, "Propose")
    repair, lineage, journal, _, repair_elapsed = _ordinary(parents, repair_index, artifacts, source, "Repair")
    base, target = root["input"]["snapshot"], repair["input"]["snapshot"]
    if (root_index == repair_index or root_lineage["virtual_prefix"] or root_lineage["proposal"] or root_lineage["counterexample"] is not None
            or any(canonical(base[key]) != canonical(target[key]) for key in ("game_id", "opening_id", "line_genealogy_id", "actual_history", "source", "frozen_epoch", "encoding_sha256"))
            or base["white_to_move"] is not target["white_to_move"] or base["input_revision"] > target["input_revision"]
            or base["capture_sequence"] >= target["capture_sequence"]
            or root_lineage["process_epoch"] != lineage["process_epoch"] or root_lineage["request_sequence"] >= lineage["request_sequence"]
            or root_elapsed[-1] > repair_elapsed[0]):
        raise ValueError("same actual root/history/producer and ordered initial Repair causal captures required")
    prefix, proposal, counter = lineage["virtual_prefix"], lineage["proposal"], lineage["counterexample"]
    _moves(prefix, minimum=2)
    _moves(proposal, minimum=len(prefix))
    _moves(counter, minimum=len(prefix))
    divergence = len(prefix) - 1
    if (divergence % 2 != 1 or counter[:len(prefix)] != prefix or proposal[:divergence] != prefix[:divergence]
            or proposal[divergence] == prefix[divergence] or len(target["legal_moves"]) < 2):
        raise ValueError("first opponent counterexample response and P Repair decision required")
    prefix_check, proposal_check = checks
    if any(type(check) is not CheckedSemanticInput for check in checks):
        raise ValueError("exact checked Rust semantic prefix/proposal capabilities required")
    for check, name in zip(checks, ("rules_prefix_sha256", "rules_proposal_sha256")):
        if check.verify_parent(parents) != root_index or check.sha256 != _sha(pins[name]):
            raise ValueError("Repair Rules capability lacks the same strict current root")
    common, rules = prefix_check.common_query(), prefix_check.rules_receipt()
    if (common["question"] != "continuation_challenge" or common["prefix"] != prefix or common["root_moves"]
            or common["claimed_line"] != counter[len(prefix):] or rules["target"]["play_status"] != "ongoing"
            or any(rules["target"][key] != target[key] for key in ("rules_state_sha256", "rules_history_sha256", "board_fen", "legal_moves"))
            or rules["target"]["side_to_move"] != ("white" if target["white_to_move"] else "black")
            or tuple(rules["target"]["board64_piece_codes"]) != parents.encodings[repair["input"]["sha256"]].board):
        raise ValueError("actual Rules-checked counterexample prefix does not reach this Repair input")
    original = proposal_check.common_query()
    if (original["question"] != "continuation_challenge" or original["prefix"] or original["root_moves"]
            or original["claimed_line"] != proposal):
        raise ValueError("root-anchored original proposal lacks actual Rules-only legality check")
    _pin(context_raw, pins["context"])
    context = _fields(_parse(context_raw), ("context", "sha256"))
    body = {"schema": CONTEXT_SCHEMA, "provenance_mode": "checked_existing_prepared_lineage",
            "parent_input_sha256": root["input"]["sha256"], "parent_label_sha256": training.label_digest(root),
            "repair_input_sha256": repair["input"]["sha256"], "repair_label_sha256": training.label_digest(repair),
            "current_view_sha256": parents.current_view.sha256, "frozen_admission_sha256": parents._frozen_admission_identity,
            "collection_receipt_sha256": authority.collection_receipt_sha256,
            "root_prepared_sha256": root_journal["sha256"], "repair_prepared_sha256": journal["sha256"],
            "root_native_request": root_journal["prepared"]["native_request"], "repair_native_request": journal["prepared"]["native_request"],
            "divergence_ply": divergence, "virtual_prefix": prefix, "proposal": proposal, "counterexample": counter,
            "rules_prefix_sha256": prefix_check.sha256, "rules_proposal_sha256": proposal_check.sha256}
    if canonical(context["context"]) != canonical(body) or context["sha256"] != digest(CONTEXT_SCHEMA, body):
        raise ValueError("derived Repair context must match current labels and actual prepared lineage")
    admission = {"schema": ADMISSION_SCHEMA, "scope": SCOPE, **{key: value for key, value in body.items() if key != "schema"},
                 "context_sha256": context["sha256"], "caller_assurance": "independently_pinned_caller_collection_observation;Rules_only_registered_receipts",
                 "registered_graph_route": authority.registered_critic_graph_route,
                 "rules_input_semantic_sha256": authority.rules_input_semantic_sha256,
                 "encoding_sha256": authority.encoding_sha256,
                 "export_declared_encoder_source_sha256": authority.export_declared_encoder_source_sha256,
                 "actual_encoder_source_sha256": authority.actual_encoder_source_sha256,
                 "native_repair_search_consumed": True, "proposed_hypothesis_only": True,
                 "strategic_repair_validity_admitted": False, "counterexample_validity_admitted": False,
                 "counterfactual_wdl_admitted": False, "divergence_ranking_admitted": False,
                 "actual_training_executed": False}
    return body, admission


class CheckedRepairContext:
    """Factory-only immutable causal capability, not a repaired-line proof."""
    __slots__ = ("_parents", "_root_index", "_repair_index", "_artifacts", "_raws", "_checks", "_pins", "_identity")

    def __init__(self, token=None, *, parents=None, root_index=None, repair_index=None, artifacts=None, raws=(), checks=(), pins=None):
        if token is not _FACTORY:
            raise ValueError("unchecked Repair context constructor refused")
        for name, value in (("_parents", parents), ("_root_index", root_index), ("_repair_index", repair_index),
                            ("_artifacts", tuple(sorted(artifacts.items()))), ("_raws", tuple(raws)), ("_checks", tuple(checks)),
                            ("_pins", canonical(pins)), ("_identity", digest(ADMISSION_SCHEMA, pins))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("checked Repair context is immutable")

    def _view(self):
        pins = _parse(self._pins)
        if digest(ADMISSION_SCHEMA, pins) != self._identity:
            raise ValueError("Repair context identity changed")
        return _validate(self._parents, self._root_index, self._repair_index, dict(self._artifacts), self._raws, self._checks, pins)

    def verify(self):
        self._view()
        return self

    @property
    def sha256(self):
        self.verify()
        return self._identity

    def context(self):
        return copy.deepcopy(self._view()[0])

    def admission(self):
        return copy.deepcopy(self._view()[1])


def admit_repair_context(*, parents, root_index, repair_index, artifacts, registration_bytes, checked_source_bytes,
                         export_manifest_bytes, launch_bytes, context_bytes, prefix_rules_check, proposal_rules_check, expected_pins):
    """Add selected causal checks to an already strict-loaded ordinary dataset.

    Context bytes are a separately sealed post-collection description of exact
    prior prepared lineage. They neither change the old seals nor assert earlier
    publication. Two exact semantic capabilities validate root proposal legality
    and the counterexample prefix/suffix; no Python Rules implementation is used.
    """
    raws = (registration_bytes, checked_source_bytes, export_manifest_bytes, launch_bytes, context_bytes)
    checks = (prefix_rules_check, proposal_rules_check)
    pins = _parse(canonical(expected_pins))
    _validate(parents, root_index, repair_index, artifacts, raws, checks, pins)
    return CheckedRepairContext(_FACTORY, parents=parents, root_index=root_index, repair_index=repair_index,
                                artifacts=artifacts, raws=raws, checks=checks, pins=pins)


class CheckedRepairCandidatePairs:
    """Only existing checked ordinal targets at this exact current Repair row."""
    __slots__ = ("_causal", "_candidate", "_indices", "_identity")

    def __init__(self, token=None, *, causal=None, checked=None, indices=()):
        if token is not _FACTORY:
            raise ValueError("unchecked Repair pair constructor refused")
        for name, value in (("_causal", causal), ("_candidate", checked), ("_indices", tuple(indices)),
                            ("_identity", digest(PAIR_SCHEMA, [causal.sha256, checked._identity, list(indices)]))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("checked Repair candidate binding is immutable")

    def verify(self):
        self._causal.verify()
        self._candidate.verify()
        if (self._candidate.parents is not self._causal._parents
                or self._identity != digest(PAIR_SCHEMA, [self._causal.sha256, self._candidate._identity, list(self._indices)])):
            raise ValueError("Repair pair parent/capability identity changed")
        identity = self._causal.context()["repair_input_sha256"]
        for index in self._indices:
            target = self._candidate.targets[index]
            if target.input_sha256 != identity or target.row_index != self._causal._repair_index or target.role != "proposer":
                raise ValueError("next-Repair pair must use its exact ordinary current P input")
        return self

    @property
    def targets(self):
        self.verify()
        return tuple(self._candidate.targets[index] for index in self._indices)

    @property
    def sha256(self):
        self.verify()
        return self._identity

    def collate(self, indices, *, split="train"):
        self.verify()
        if type(indices) not in (list, tuple) or not 1 <= len(indices) <= 256 or len(set(indices)) != len(indices):
            raise ValueError("finite unique next-Repair pair batch required")
        chosen = [self._indices[_int(index, 0, len(self._indices) - 1)] for index in indices]
        return self._candidate.collate(chosen, role="proposer", split=split)


def bind_repair_candidate_pairs(causal, checked, *, indices):
    """Filter an actual admitted bank; never make masked evidence positive."""
    if type(causal) is not CheckedRepairContext or type(checked) is not candidate.CheckedComparativePairs:
        raise ValueError("exact factory-checked causal and candidate capabilities required")
    if type(indices) not in (tuple, list) or not 1 <= len(indices) <= 4096 or len(set(indices)) != len(indices):
        raise ValueError("finite unique selected next-Repair pair indices required")
    for index in indices:
        _int(index, 0, len(checked.targets) - 1)
    result = CheckedRepairCandidatePairs(_FACTORY, causal=causal, checked=checked, indices=indices)
    return result.verify()


def frozen_repair_preparation(model, checked, *, expected_parameter_sha256, checkpoint_sha256,
                              split="train", max_wall_time_ms=60000):
    """Separate frozen next-move ordinal loss; no strategic repair target.

    Caller supplies an actually reloaded, registered frozen model. Existing
    ordinary collation and candidate softplus are reused without new targets.
    All-masked banks remain ineligible for positive no-step preparation.
    """
    import torch
    from .preparation_check import _parameter_digest
    deadline = time.monotonic() + _int(max_wall_time_ms, 1, 300000) / 1000
    _sha(expected_parameter_sha256)
    _sha(checkpoint_sha256)
    if type(checked) is not CheckedRepairCandidatePairs or split not in ("train", "validation", "holdout"):
        raise ValueError("checked next-Repair bank and split required")

    def frozen_digest():
        # Tensor bytes do not encode training/requires_grad/autocast flags.
        # Recheck this complete admission at both sides of actual forwarding.
        if (any(module.training for module in model.modules()) or torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda")
                or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())):
            raise ValueError("caller-frozen FP32 no-autocast eval model without gradients required")
        return _parameter_digest(model)  # Rechecks finite CPU FP32 state too.

    before = frozen_digest()
    if before != expected_parameter_sha256:
        raise ValueError("actual Repair model parameters differ from independent pin")
    targets = checked.targets
    selected = [index for index, target in enumerate(targets) if target.split == split]
    known = sum(targets[index].mask for index in selected)
    if not known:
        raise ValueError("all-masked or missing next-Repair ordinal targets")
    total, consumed = 0.0, []
    with torch.inference_mode():
        for offset in range(0, len(selected), 256):
            if time.monotonic() >= deadline:
                raise ValueError("next-Repair original preparation wall expired")
            batch = checked.collate(selected[offset:offset + 256], split=split)
            memory = model.public_encoder(*batch.base.inputs.public_args())
            outputs = model.role_graph("proposer")(*batch.base.inputs.role_args(memory))
            loss = candidate.pairwise_softplus_loss(outputs[0], batch)
            total += float(loss) * sum(batch.mask)
            consumed.extend(batch.pair_ids)
    value = total / known
    if not 0 < value < float("inf"):
        raise ValueError("next-Repair preparation needs finite nonzero ordinal loss")
    after = frozen_digest()
    checked.verify()
    if before != after or any(parameter.grad is not None for parameter in model.parameters()) or time.monotonic() >= deadline:
        raise ValueError("next-Repair preparation parameter/gradient/wall invariant")
    return {"schema": PREPARATION_SCHEMA, "scope": SCOPE, "causal_admission_sha256": checked._causal.sha256,
            "pair_admission_sha256": checked.sha256, "checkpoint_sha256": checkpoint_sha256,
            "parameter_sha256_before": before, "parameter_sha256_after": after, "consumed_pair_ids": consumed,
            "known_pairs": known, "pairwise_loss": value, "strategic_repair_validity_admitted": False,
            "counterexample_validity_admitted": False, "counterfactual_wdl_admitted": False,
            "divergence_ranking_admitted": False, "actual_training_executed": False,
            "backward_executed": False, "optimizer_created": False, "gpu_executed": False}
