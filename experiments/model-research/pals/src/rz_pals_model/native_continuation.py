"""Bounded consumer of the original multi-Reply continuation/2 observation.

This checks byte custody and cross-journal attribution AFTER the independent
frozen-dataset loader has checked Rules inputs, producer registrations and split
leakage. It neither creates that capability nor replaces its checks. A sealed
JSON observation is not independent execution, Rules, utility or target proof.
No models, subprocesses, optimizer, legacy witness or CPU receipt are created.
The stdlib-only boundary lets receipt inspection run without loading Torch.
"""
import copy
import hashlib
import json


TRACE_ARTIFACT = "native-continuation-traces.jsonl"
TRACE_DOMAIN = "rz-pals-native-post-repair-continuation/2"
AUDIT_SCHEMA = "rz-pals-native-continuation-collection-observation/1"
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_ROW_BYTES = 2 * 1024 * 1024
MAX_ROWS = 65536 * 8
MAX_ATTEMPTS = 128
U64_MAX = 2**64 - 1
POLICY = {
    "version": "pals-post-repair-continuation/1",
    "policy": "actual_opponent_continuation_v1",
    "search_identity": "pals-restricted-refinement-post-repair-continuation/1",
    "conditions_sha256": list(bytes.fromhex(
        "876c7104c28183131243101df86c386865bf903f26545edde01d48de4403d4d1")),
}
REQUIRED_ARTIFACTS = (
    TRACE_ARTIFACT, "inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl",
    "producer-prepared.jsonl", "native-events.jsonl", "native-raw-outputs.jsonl",
    "records.jsonl", "split.jsonl",
)
DENIED_AUTHORITIES = {
    "independent_native_execution_authority": False,
    "rules_proof_admitted": False,
    "ordinary_policy_target_admitted": False,
    "wdl_admitted": False,
    "whole_action_cost_authority": False,
    "utility_authority": False,
    "training_target_authority": False,
    "actual_training_executed": False,
    "product_authority": False,
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("continuation lowercase SHA256 identity required")
    return value


def _uint(value, maximum=U64_MAX):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("continuation finite integer bound")
    return value


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("continuation exact fields required")
    return value


def _pin(raw):
    if type(raw) is not bytes:
        raise ValueError("continuation immutable original bytes required")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _parse(raw):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_ROW_BYTES:
        raise ValueError("continuation bounded JSON bytes required")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("continuation duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _rows(raw):
    if type(raw) is not bytes or len(raw) > MAX_INPUT_BYTES:
        raise ValueError("continuation bounded original journal required")
    if not raw:
        return []
    if not raw.endswith(b"\n") or raw.count(b"\n") > MAX_ROWS:
        raise ValueError("continuation exact newline/row bound")
    return [(line, _parse(line)) for line in raw.splitlines()]


def _moves(value, minimum=0, maximum=64):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise ValueError("continuation bounded Move16 list required")
    for movement in value:
        _uint(movement, 65535)
        if movement >> 12 > 4 or movement & 63 == (movement >> 6) & 63:
            raise ValueError("continuation invalid Move16 code")
    return value


def _request(value):
    if type(value) is not list or len(value) != 2:
        raise ValueError("continuation native RequestId pair required")
    for number in value:
        _uint(number)
    return tuple(value)


def _identity(value):
    _fields(value, ("game_generation", "search_generation", "root", "root_revision",
                    "repair_record_revision", "repaired_line"))
    _fields(value["root"], ("slot", "generation"))
    for name, number in value.items():
        if name != "root":
            _uint(number)
    for number in value["root"].values():
        _uint(number)
    return value


def _context(value, identity, purpose, prefix):
    _fields(value, ("game_generation", "search_generation", "situation", "state", "focus", "purpose", "prefix",
                    "focus_sha256", "prefix_sha256", "proposal_sha256", "refutation_sha256", "divergence_sha256",
                    "public_revision", "situation_revision"))
    _fields(value["situation"], ("slot", "generation"))
    for name in ("game_generation", "search_generation", "state", "focus", "public_revision", "situation_revision"):
        _uint(value[name])
    for number in value["situation"].values():
        _uint(number)
    for name in ("focus_sha256", "prefix_sha256", "proposal_sha256", "divergence_sha256"):
        _sha(value[name])
    if value["refutation_sha256"] is not None:
        _sha(value["refutation_sha256"])
    _moves(value["prefix"])
    if (value["game_generation"] != identity["game_generation"]
            or value["search_generation"] != identity["search_generation"]
            or value["purpose"] != purpose or value["prefix"] != prefix):
        raise ValueError("continuation logical generation/purpose/prefix changed")
    return value


def _index(rows, key):
    result = {}
    for raw, value in rows:
        if type(value) is not dict:
            raise ValueError("continuation journal object required")
        identity = _sha(value.get(key))
        result.setdefault(identity, []).append((raw, value))
    return result


def _one(index, identity, name):
    found = index.get(identity, [])
    if len(found) != 1:
        raise ValueError("continuation requires one original " + name)
    return found[0]


def _native_index(rows):
    by_request, by_input = {}, {}
    for _, row in rows:
        if type(row) is not dict:
            raise ValueError("continuation native journal object required")
        request = _request([row.get("process_epoch"), row.get("request_sequence")])
        input_sha = _sha(row.get("input_sha256"))
        by_request.setdefault(request, []).append(row)
        by_input.setdefault(input_sha, set()).add(request)
    return by_request, by_input


def _finite_bits(value, size):
    if type(value) is not list or len(value) != size:
        raise ValueError("continuation raw head shape mismatch")
    for bits in value:
        _uint(bits, 0xffffffff)
        if bits & 0x7f800000 == 0x7f800000:
            raise ValueError("continuation nonfinite raw f32")


def _ranking(raw, legal, kind):
    _fields(raw, ("representation", "candidate_logits_bits", "wdl_logits_bits", "divergence_logits_bits",
                  "task_logits_bits", "private_latent_bits", "prediction_is_future_label"))
    _moves(legal, 1, 256)
    if len(set(legal)) != len(legal):
        raise ValueError("continuation duplicate ordered candidate")
    if (raw["representation"] != "f32_ieee754_bits" or raw["prediction_is_future_label"] is not False
            or raw["task_logits_bits"] is not None
            or raw["divergence_logits_bits"] != ([] if kind == "Reply" else None)):
        raise ValueError("continuation raw role/head/label meaning changed")
    _finite_bits(raw["candidate_logits_bits"], len(legal))
    _finite_bits(raw["wdl_logits_bits"], 3)
    _finite_bits(raw["private_latent_bits"], 6144)
    def order(bits):
        return (~bits & 0xffffffff) if bits & 0x80000000 else bits ^ 0x80000000
    return [legal[index] for index in sorted(range(len(legal)),
            key=lambda index: (-order(raw["candidate_logits_bits"][index]), index))]


def _lifecycle(events, outputs, request, input_sha, game, legal, kind):
    matching = events[0].get(request, [])
    matching_outputs = outputs[0].get(request, [])
    if any(index[1].get(input_sha, set()) - {request} for index in (events, outputs)):
        raise ValueError("continuation input attributed to another native request")
    stages, elapsed, unknown_rejection = [], [], False
    for event in matching:
        _fields(event, ("domain", "game_id", "process_epoch", "request_sequence", "input_sha256", "stage", "observer_elapsed_us", "detail"))
        if (event["domain"] != "rz-pals-native-call-event/1" or event["input_sha256"] != input_sha
                or event["game_id"] != game):
            raise ValueError("continuation native event attribution mismatch")
        elapsed.append(_uint(event["observer_elapsed_us"]))
        stages.append(event["stage"])
        detail = event["detail"]
        if event["stage"] == "prepared":
            if _canonical(detail) != _canonical({"prepared_before_submit": True, "native_query_kind": kind, "producer_metadata_admitted": True}):
                raise ValueError("continuation actual prepared producer event required")
        elif event["stage"] == "physically_completed":
            _fields(detail, ("success", "logical_acceptance_inferred"))
            if type(detail["success"]) is not bool or detail["logical_acceptance_inferred"] is not False:
                raise ValueError("continuation completion cannot infer consumption")
        elif event["stage"] == "delivered":
            if _canonical(detail) != _canonical({"search_consumed": False}):
                raise ValueError("continuation delivery cannot infer consumption")
        elif event["stage"] == "search_consumed":
            if _canonical(detail) != _canonical({"search_consumed": True}):
                raise ValueError("continuation explicit search consumption required")
        elif event["stage"] == "logically_rejected":
            _fields(detail, ("reason", "search_consumed"))
            if detail["search_consumed"] is not False or type(detail["reason"]) is not str:
                raise ValueError("continuation rejection cannot infer consumption")
            unknown_rejection = detail["reason"] == "PhysicalCompletionUnknown"
        else:
            raise ValueError("continuation unknown native lifecycle stage")
    valid = ([], ["prepared"], ["prepared", "logically_rejected"],
             ["prepared", "physically_completed"], ["prepared", "physically_completed", "logically_rejected"],
             ["prepared", "physically_completed", "delivered"],
             ["prepared", "physically_completed", "delivered", "logically_rejected"],
             ["prepared", "physically_completed", "delivered", "search_consumed"])
    if stages not in valid or elapsed != sorted(elapsed) or len(matching_outputs) > 1:
        raise ValueError("continuation duplicate/reordered native lifecycle")
    physical = "physically_completed" in stages
    if physical and unknown_rejection:
        raise ValueError("continuation known physical completion cannot become unknown")
    if physical != bool(matching_outputs):
        raise ValueError("continuation physical event/raw journal mismatch")
    ranking = None
    if matching_outputs:
        output = _fields(matching_outputs[0], ("domain", "process_epoch", "request_sequence", "input_sha256", "physical_completion_confirmed", "success", "raw"))
        event_success = matching[stages.index("physically_completed")]["detail"]["success"]
        if (output["domain"] != "rz-pals-native-physical-raw/1" or output["input_sha256"] != input_sha
                or type(output["success"]) is not bool or type(output["physical_completion_confirmed"]) is not bool
                or output["success"] is not event_success):
            raise ValueError("continuation physical raw attribution/success changed")
        if not output["success"] or not output["physical_completion_confirmed"]:
            if "delivered" in stages or "search_consumed" in stages:
                raise ValueError("continuation failed/unknown physical result consumed")
        else:
            ranking = _ranking(output["raw"], legal, kind)
    return {"physical": physical and matching_outputs[0]["physical_completion_confirmed"],
            "physical_unknown": unknown_rejection or physical and not matching_outputs[0]["physical_completion_confirmed"],
            "delivered": "delivered" in stages,
            "consumed": "search_consumed" in stages, "rejected": "logically_rejected" in stages,
            "ranking": ranking, "elapsed_us": elapsed}


def _call(call, *, indices, events, outputs, game, identity, prefix, proposal, refutation, kind, source, journals):
    request = _request([call["process_epoch"], call["request_sequence"]])
    input_sha = _sha(call["input_sha256"])
    (_, frozen), (_, sidecar), (_, lineage) = [
        _one(indices[name], input_sha, name)[0:2]
        for name in ("inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl")]
    for name, anchor in (("inputs.jsonl", "input_row_sha256"), ("native-inputs.jsonl", "sidecar_row_sha256"),
                         ("input-lineage.jsonl", "lineage_row_sha256")):
        if _pin(_one(indices[name], input_sha, name)[0])["sha256"] != _sha(call[anchor]):
            raise ValueError("continuation descriptor differs from original row bytes")
    _fields(frozen, ("snapshot", "sha256"))
    records = indices["records"].get(input_sha, [])
    if not records or any(record["input"] != frozen for record in records):
        raise ValueError("continuation learning call lacks matching strict dataset records")
    snapshot = frozen["snapshot"]
    context = _context(call["logical_context"], identity, kind + "Policy", prefix)
    if (snapshot.get("game_id") != game or snapshot.get("role") != ("critic" if kind == "Reply" else "proposer")
            or snapshot.get("input_revision") != context["public_revision"]
            or snapshot.get("rules_state_sha256") != call["position_rules_state_sha256"]
            or snapshot.get("source") != source["source"] or snapshot.get("frozen_epoch") != source["frozen_epoch"]
            or snapshot.get("encoding_sha256") != source["encoding_sha256"]):
        raise ValueError("continuation frozen input role/source/state/revision mismatch")
    if (lineage.get("input_sha256") != input_sha or lineage.get("game_id") != game
            or _request([lineage.get("process_epoch"), lineage.get("request_sequence")]) != request
            or lineage.get("native_query_kind") != kind or lineage.get("virtual_prefix") != prefix
            or lineage.get("proposal") != proposal or lineage.get("counterexample") != refutation
            or lineage.get("actual_played_history") != snapshot.get("actual_history")
            or lineage.get("actual_outcome_eligible") is not False
            or lineage.get("counterfactual_wdl") != "masked" or lineage.get("training_admission") != "ordinary_role"):
        raise ValueError("continuation original conditional lineage changed or leaked actual WDL")
    if (sidecar.get("version") != "rz-pals-native-input-sidecar/1" or sidecar.get("sha256") != call["sidecar_sha256"]
            or sidecar.get("canonical_tensor_sha256") != call["canonical_tensor_sha256"]
            or sidecar.get("encoder_source_sha256") != source["encoder_source_sha256"]):
        raise ValueError("continuation sidecar/encoder identity changed")
    tensor_raw = sidecar["tensor_json"].encode("utf-8")
    if _pin(tensor_raw)["sha256"] != sidecar["tensor_sha256"]:
        raise ValueError("continuation actual tensor string bytes changed")
    tensor = _parse(tensor_raw)
    _fields(tensor, ("role", "board", "metadata", "records", "required_critical_records", "candidates", "divergence_features",
                    "query", "situation_revision", "history_digest", "model_epoch"))
    legal = _moves(snapshot["legal_moves"], 1, 256)
    candidates = [{"from": move & 63, "to": (move >> 6) & 63, "promotion": move >> 12} for move in legal]
    if (tensor["role"] != snapshot["role"] or tensor["situation_revision"] != snapshot["input_revision"]
            or tensor["candidates"] != candidates or tensor["divergence_features"] != []
            or type(tensor["history_digest"]) is not list or len(tensor["history_digest"]) != 32
            or bytes(_uint(b, 255) for b in tensor["history_digest"]).hex() != snapshot["rules_history_sha256"]):
        raise ValueError("continuation actual tensor role/history/candidate order changed")
    journal = _one(journals, input_sha, "prepared producer journal")[1]
    prepared = journal["prepared"]
    if (journal["sha256"] != _pin(_canonical(["rz-pals-collector-prepared-producer/1", prepared]))["sha256"]
            or prepared["native_request"] != list(request) or prepared["input_sha256"] != input_sha
            or prepared["game_id"] != game or prepared["learning_input"] is not True
            or prepared["publication"] != "seal-before-submit; prepaid-drain-after-search"
            or any(prepared[key] != _pin(_one(indices[name], input_sha, name)[0]) for name, key in
                   (("inputs.jsonl", "input_json"), ("native-inputs.jsonl", "tensor_sidecar_json"), ("input-lineage.jsonl", "lineage_json")))):
        raise ValueError("continuation original before-submit producer journal changed")
    lifecycle = _lifecycle(events, outputs, request, input_sha, game, legal, kind)
    return request, snapshot, lifecycle


def audit_native_continuation_collection(*, receipt_bytes, artifact_bytes, expected_receipt_sha256,
                                         policy_registration_bytes, expected_policy_registration_sha256,
                                         max_input_bytes=MAX_INPUT_BYTES):
    """Return conditional raw-observation facts, never a checked native authority.

    Receipt and policy pins must come from prior owner registration. Every raw
    artifact is already named by the original receipt; no inventory is rewritten.
    Missing completion remains unresolved, while contradictory evidence is an
    error. Callers must separately use the strict frozen dataset admission.
    """
    _uint(max_input_bytes, MAX_INPUT_BYTES)
    if (max_input_bytes == 0 or type(artifact_bytes) is not dict
            or any(type(raw) is not bytes for raw in artifact_bytes.values())
            or sum(map(len, artifact_bytes.values())) + len(receipt_bytes) + len(policy_registration_bytes) > max_input_bytes):
        raise ValueError("continuation aggregate original byte budget")
    if _pin(receipt_bytes)["sha256"] != _sha(expected_receipt_sha256):
        raise ValueError("continuation original receipt differs from independent pin")
    receipt = _parse(receipt_bytes)
    if (receipt.get("version") != "rz-pals-own-collector/1" or type(receipt.get("artifacts")) is not dict
            or type(receipt.get("complete")) is not bool or receipt.get("actual_training_executed") is not False
            or receipt.get("external_teacher_used") is not False):
        raise ValueError("continuation original owned collection receipt required")
    if len(policy_registration_bytes) > 32 * 1024 or _pin(policy_registration_bytes)["sha256"] != _sha(expected_policy_registration_sha256):
        raise ValueError("continuation independent policy registration pin mismatch")
    registration = _fields(_parse(policy_registration_bytes), ("version", "base_registry_canonical_sha256", "collector_binary_sha256", "search_policy"))
    source = receipt["source"]
    source_sha = _pin(_canonical(["rz-pals-collector-checked-source/1", source]))["sha256"]
    native = source.get("native", {})
    if (registration["version"] != "rz-pals-native-continuation-registration/1" or registration["search_policy"] != POLICY
            or registration["collector_binary_sha256"] != source["implementation_sha256"]
            or native.get("refinement_registration_sha256") != expected_policy_registration_sha256
            or native.get("refinement_registration") != registration or native.get("pals_search_policy") != POLICY):
        raise ValueError("continuation registered policy/source/binary mismatch")
    _sha(registration["base_registry_canonical_sha256"])
    for name in REQUIRED_ARTIFACTS:
        if name not in receipt["artifacts"] or name not in artifact_bytes:
            raise ValueError("continuation artifact absent from original receipt: " + name)
    for name, raw in artifact_bytes.items():
        if name not in receipt["artifacts"] or _pin(raw) != receipt["artifacts"][name]:
            raise ValueError("continuation artifact differs from original receipt: " + name)
    parsed = {name: _rows(artifact_bytes[name]) for name in REQUIRED_ARTIFACTS}
    indices = {name: _index(parsed[name], "sha256" if name == "inputs.jsonl" else "input_sha256")
               for name in ("inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl")}
    journals = {}
    for raw, row in parsed["producer-prepared.jsonl"]:
        _fields(row, ("prepared", "sha256"))
        if type(row["prepared"]) is not dict:
            raise ValueError("continuation prepared journal object required")
        journals.setdefault(_sha(row["prepared"].get("input_sha256")), []).append((raw, row))
    if len(parsed["split.jsonl"]) != 1:
        raise ValueError("continuation one original split declaration required")
    split = parsed["split.jsonl"][0][1].get("games")
    if type(split) is not dict or any(value not in ("train", "validation", "holdout") for value in split.values()):
        raise ValueError("continuation original split assignment required")
    events, outputs = _native_index(parsed["native-events.jsonl"]), _native_index(parsed["native-raw-outputs.jsonl"])
    records = {}
    for _, record in parsed["records.jsonl"]:
        if type(record) is not dict or type(record.get("input")) is not dict:
            raise ValueError("continuation original learning record input required")
        records.setdefault(_sha(record["input"].get("sha256")), []).append(record)
    indices["records"] = records
    groups, order = {}, []
    for _, row in parsed[TRACE_ARTIFACT]:
        _fields(row, ("domain", "stage", "game_id", "identity", "descriptor_sha256", "payload_sha256", "observer_elapsed_us", "data"))
        identity = _identity(row["identity"])
        if row["domain"] != TRACE_DOMAIN or row["stage"] not in ("prepared", "reply_bound", "finished"):
            raise ValueError("continuation separate v2 domain/stage required")
        _uint(row["observer_elapsed_us"])
        descriptor = _sha(row["descriptor_sha256"])
        expected = _pin(_canonical([TRACE_DOMAIN, row["stage"], identity, descriptor, row["data"]]))["sha256"]
        if expected != _sha(row["payload_sha256"]):
            raise ValueError("continuation original trace payload seal mismatch")
        key = (row["game_id"], _canonical(identity))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    if len(groups) > MAX_ATTEMPTS:
        raise ValueError("continuation attempt budget")
    observations, requests_seen = [], set()
    for key in order:
        group = groups[key]
        observations.append(_attempt(group, source_sha=source_sha, policy_sha=expected_policy_registration_sha256,
                                     source=source, indices=indices, journals=journals, split=split,
                                     records=records, events=events, outputs=outputs, requests_seen=requests_seen))
    return {"schema": AUDIT_SCHEMA, "scope": "conditional_original_multi_reply_observation_only",
            "receipt": _pin(receipt_bytes), "policy_registration": _pin(policy_registration_bytes),
            "artifacts": {name: _pin(artifact_bytes[name]) for name in REQUIRED_ARTIFACTS},
            "original_receipt_complete": receipt["complete"], "original_failure": receipt.get("failure"),
            "attempts": observations, "attempt_count": len(observations),
            "multi_reply_attempts_observed": sum(len(row["reply_calls"]) > 1 for row in observations),
            "reply_requests_observed": sum(len(row["reply_calls"]) for row in observations),
            "physical_nn_rows": None, "physical_nn_rows_observation": "unknown;call callbacks are not dispatcher input rows",
            **DENIED_AUTHORITIES}


def _attempt(group, *, source_sha, policy_sha, source, indices, journals, split, records, events, outputs, requests_seen):
    stages = [row["stage"] for row in group]
    if (stages[0] != "prepared" or any(stage != "reply_bound" for stage in stages[1:-1])
            or stages[-1] not in ("reply_bound", "finished", "prepared")
            or len({row["descriptor_sha256"] for row in group}) != 1
            or [row["observer_elapsed_us"] for row in group] != sorted(row["observer_elapsed_us"] for row in group)):
        raise ValueError("continuation attempt chronology/descriptor drift")
    prepared, identity, game = group[0]["data"], group[0]["identity"], group[0]["game_id"]
    _sha(prepared["root_rules_state_sha256"])
    _sha(prepared["anchor_rules_state_sha256"])
    _uint(prepared["engine_deadline_tick"])
    if group[0]["descriptor_sha256"] != _pin(_canonical([TRACE_DOMAIN, "prepared-descriptor", identity, prepared]))["sha256"]:
        raise ValueError("continuation prepared descriptor seal mismatch")
    repaired, refutation = _moves(prepared["repaired"], 3), _moves(prepared["refutation"], 3)
    anchor = _uint(prepared["anchor_ply"], len(repaired) - 1)
    initial = _context(prepared["initial_reply_context"], identity, "ReplyPolicy", repaired[:anchor])
    # Repair identifies a historical record; its single verification can
    # advance the current public view once before Reply without changing R.
    repair_revision = identity["repair_record_revision"]
    allowed_public_revisions = (repair_revision,)
    if repair_revision < U64_MAX:
        allowed_public_revisions += (repair_revision + 1,)
    if (anchor == 0 or initial["public_revision"] not in allowed_public_revisions
            or prepared["policy"] != POLICY or prepared["engine_observer_version"] != "pals-post-repair-continuation-observer/2"
            or prepared["checked_source_sha256"] != source_sha or prepared["refinement_registration_sha256"] != policy_sha
            or prepared["prepared_before_reply_submit"] is not True
            or prepared["maximum_reply_calls"] != len(repaired) - anchor
            or prepared["initial_selection_rule"] != "ranked_unexamined_different_else_ranked_different/1"):
        raise ValueError("continuation actual post-Repair anchor/revision/source changed")
    if game not in split:
        raise ValueError("continuation game missing from original split")
    parent = _fields(prepared["parent_repair_chain"], ("initial_prefix", "initial_prefix_len", "full_repaired_line", "calls", "journal_observation"))
    start = _uint(parent["initial_prefix_len"], len(repaired) - 1)
    if (start < 2 or parent["initial_prefix"] != repaired[:start] or parent["full_repaired_line"] != repaired
            or repaired[:start] != refutation[:start] or type(parent["calls"]) is not list
            or len(parent["calls"]) != len(repaired) - start):
        raise ValueError("continuation complete original Repair suffix required")
    previous, previous_consumed, ancestry, common = None, None, [], None
    for ply, call in enumerate(parent["calls"], start):
        context = _context(call["logical_context"], identity, "RepairPolicy", repaired[:ply])
        request, snapshot, life = _call(call, indices=indices, journals=journals, events=events, outputs=outputs,
                                      game=game, identity=identity, prefix=repaired[:ply], proposal=call["proposal"],
                                      refutation=refutation, kind="Repair", source=source)
        if (call["prefix_len"] != ply or call["prefix"] != repaired[:ply] or call["chosen_move"] != repaired[ply]
                or call["counterexample"] != refutation or context["public_revision"] + 1 != identity["repair_record_revision"]
                or call["logical_context_sha256"] != _pin(_canonical(context))["sha256"]
                or not life["consumed"] or life["ranking"][0] != repaired[ply]
                or life["elapsed_us"][-1] > group[0]["observer_elapsed_us"]
                or previous_consumed is not None and previous_consumed > life["elapsed_us"][0]
                or previous is not None and (request[0] != previous[0] or request[1] <= previous[1])):
            raise ValueError("continuation parent Repair acceptance/selection/order changed")
        root_facts = [snapshot[name] for name in ("game_id", "opening_id", "line_genealogy_id", "actual_history", "source", "frozen_epoch", "encoding_sha256")]
        if common is not None and common != root_facts:
            raise ValueError("continuation Repair root/history/model drift")
        common, previous, previous_consumed = root_facts, request, life["elapsed_us"][-1]
        ancestry.append({"native_request": list(request), "input_sha256": call["input_sha256"], "chosen_move": repaired[ply]})
    bound = [row for row in group[1:] if row["stage"] == "reply_bound"]
    if len(bound) > len(repaired) - anchor:
        raise ValueError("continuation Reply call bound exceeded")
    finished = group[-1]["data"] if stages[-1] == "finished" else None
    counter = _moves(finished["counterline"]) if finished is not None else []
    reply_calls, ranks, earlier_payload = [], [], None
    for ordinal, row in enumerate(bound):
        call = row["data"]
        prefix = call["logical_context"]["prefix"]
        _moves(prefix)
        if (len(prefix) != anchor + ordinal or prefix[:anchor] != repaired[:anchor]
                or call["ordinal"] != ordinal or call["prepared_payload_sha256"] != group[0]["payload_sha256"]
                or call["previous_bound_payload_sha256"] != earlier_payload or call["bound_before_submit"] is not True
                or any(call[name] is not False for name in ("physical_completion_observed", "delivery_observed", "search_consumption_observed"))):
            raise ValueError("continuation bound call chain/prefix/pre-submit changed")
        request, snapshot, life = _call(call, indices=indices, journals=journals, events=events, outputs=outputs,
                                      game=game, identity=identity, prefix=prefix, proposal=repaired,
                                      refutation=refutation, kind="Reply", source=source)
        root_facts = [snapshot[name] for name in ("game_id", "opening_id", "line_genealogy_id", "actual_history", "source", "frozen_epoch", "encoding_sha256")]
        if (root_facts != common or request in requests_seen or request[0] != previous[0] or request[1] <= previous[1]
                or call["logical_context"]["public_revision"] != initial["public_revision"]
                or life["elapsed_us"] and life["elapsed_us"][0] < row["observer_elapsed_us"]
                or ordinal == 0 and (call["logical_context"] != initial
                                    or call["position_rules_state_sha256"] != prepared["anchor_rules_state_sha256"]
                                    or repaired[anchor] not in snapshot["legal_moves"])):
            raise ValueError("continuation duplicate/reordered/stale Reply context")
        if ordinal > 0:
            prior = reply_calls[-1]
            wanted = ranks[0] if ordinal == 1 else prior["selected_ranked_first"]
            if (not prior["search_consumed"] or prefix[:-1] != prior["prefix"] or prefix[-1] != wanted
                    or prior["event_elapsed_us"][-1] > row["observer_elapsed_us"]):
                raise ValueError("continuation next prefix lacks preceding consumed actual output")
        if ordinal == 0 and life["ranking"] is not None:
            examined = _moves(prepared["examined_responses"], 0, 256)
            if len(set(examined)) != len(examined) or any(move not in snapshot["legal_moves"] for move in examined):
                raise ValueError("continuation examined anchor responses changed")
            original = repaired[anchor]
            alternatives = [move for move in life["ranking"] if move != original]
            unexamined = [move for move in alternatives if move not in examined]
            ranks.append(next(iter(unexamined or alternatives), None))
        previous, earlier_payload = request, row["payload_sha256"]
        requests_seen.add(request)
        reply_calls.append({"native_request": list(request), "input_sha256": call["input_sha256"],
                            "prefix": prefix, "physical_completion_observed": life["physical"],
                            "delivered": life["delivered"], "search_consumed": life["consumed"],
                            "rejected": life["rejected"], "physical_unknown": life["physical_unknown"],
                            "event_elapsed_us": life["elapsed_us"],
                            "selected_ranked_first": life["ranking"][0] if life["ranking"] else None})
        for record in records.get(call["input_sha256"], []):
            label = record.get("future_label")
            if label is not None and label.get("provenance", {}).get("source") == "actual_game" and label.get("value_wdl") is not None:
                raise ValueError("continuation conditional Reply leaked actual-game value")
    if finished is not None:
        for name in ("prepared_accepted", "initial_reply_attempted", "initial_reply_accepted", "initial_selection_checked",
                     "counterline_completed", "full_suffix_replayed", "comparable", "cancelled_at_observer", "deadline_expired_at_observer"):
            if type(finished[name]) is not bool:
                raise ValueError("continuation finished actual boolean observation required")
        if (finished["prepared_payload_sha256"] != group[0]["payload_sha256"] or finished["prepared_accepted"] is not True
                or finished["full_suffix_replayed"] is not False or len(finished["calls"]) != len(bound)
                or bound and not finished["initial_reply_attempted"]
                or len(counter) > len(repaired) or len(counter) > anchor + len(bound)
                or finished["engine_deadline_tick"] != prepared["engine_deadline_tick"]):
            raise ValueError("continuation finished changed scope/call coverage/deadline")
        for ordinal, (summary, observed, row) in enumerate(zip(finished["calls"], reply_calls, bound)):
            if (summary["ordinal"] != ordinal or [summary["process_epoch"], summary["request_sequence"]] != observed["native_request"]
                    or summary["input_sha256"] != observed["input_sha256"] or summary["bound_payload_sha256"] != row["payload_sha256"]
                    or summary["physical"] is not observed["physical_completion_observed"]
                    or summary["delivered"] is not observed["delivered"] or summary["search_consumed"] is not observed["search_consumed"]
                    or summary["rejected"] is not observed["rejected"] or summary["physical_unknown"] is not observed["physical_unknown"]
                    or observed["search_consumed"] and (summary["accepted_context_checked"] is not True
                        or summary["producer_metadata_admitted"] is not True
                        or observed["event_elapsed_us"][-1] > group[-1]["observer_elapsed_us"])):
                raise ValueError("continuation finished summaries contradict original lifecycle")
            ply = anchor + ordinal
            if len(counter) > ply:
                wanted = ranks[0] if ordinal == 0 else observed["selected_ranked_first"]
                if not observed["search_consumed"] or observed["prefix"] != counter[:ply] or counter[ply] != wanted:
                    raise ValueError("continuation counterline differs from actual consumed raw policy")
            elif finished["counterline_completed"] or finished["publication"] is not None:
                raise ValueError("continuation publication exceeds consumed call coverage")
        expected = ranks[0] if ranks and finished["initial_reply_accepted"] else None
        if (finished["expected_initial_response"] != expected or finished["selected_response"] is not None and finished["selected_response"] != expected
                or finished["initial_reply_accepted"] and (not reply_calls or not reply_calls[0]["search_consumed"])
                or finished["selected_response"] is None and (counter or finished["counterline_completed"] or finished["publication"] is not None)
                or finished["selected_response"] is not None and (len(counter) <= anchor or counter[:anchor] != repaired[:anchor]
                    or counter[anchor] != finished["selected_response"] or not finished["initial_reply_accepted"])
                or finished["disposition"] == "NoAlternativeResponse" and (not finished["initial_reply_accepted"] or expected is not None)
                or finished["initial_selection_checked"] != (finished["initial_reply_accepted"]
                    and (finished["selected_response"] is not None or finished["disposition"] == "NoAlternativeResponse"))
                or finished["counterline_completed"] != (finished["selected_response"] is not None and len(counter) == len(repaired))
                or finished["publication"] is not None and (not finished["counterline_completed"] or not finished["comparable"])):
            raise ValueError("continuation finished initial selection/completion changed")
    return {"game_id": game, "split": split[game], "identity": copy.deepcopy(identity),
            "descriptor_sha256": group[0]["descriptor_sha256"], "parent_repair_calls": ancestry,
            "reply_calls": reply_calls, "finished_observed": finished is not None,
            "status": "closed_raw_observation" if finished is not None else "unresolved",
            "counterline_completed_observed": finished["counterline_completed"] if finished is not None else False,
            "cancelled_at_observer": finished.get("cancelled_at_observer") if finished is not None else None,
            "original_error": finished.get("original_error") if finished is not None else None,
            "observer_elapsed_us": [row["observer_elapsed_us"] for row in group], **DENIED_AUTHORITIES}
