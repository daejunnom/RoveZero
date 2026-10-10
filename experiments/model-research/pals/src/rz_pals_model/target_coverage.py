"""Actual target masks and reasons, separate from optimizer diagnostic fixtures.

This is an audit of the already admitted current label view. It never fills an
unknown label, scores a game, treats CP as an outcome, or performs a forward or
optimizer operation. Current collector rows, ordinal pair capabilities and V
utility capabilities retain their separate target domains.
"""
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import json

from .training import (TaskContext, ValidatedDataset, _fields, _sha,
                       _identity, _collection_jsonl, _sorted_canonical, _uint, label_digest, move_components,
                       _validate_encoding, _encoding_identity, _validate_full_line)

SCHEMA = "rz-pals-current-target-coverage/1"
CHECKER_NAMESPACE_SCHEMA = "rz-pals-verifier-checker-value-namespace/2"
HEADS = {"proposer": ("policy", "wdl"), "critic": ("policy", "wdl", "divergence"),
         "verifier": ("task",)}
FULL_LINE_ARTIFACT = "native-full-line-contexts.v2.jsonl"
COVERAGE_ARTIFACT = "target-coverage.v2.jsonl"


def _sealed_struct(value):
    actual = _sha(value["sha256"])
    candidate = copy.deepcopy(value)
    candidate["sha256"] = ""
    expected = hashlib.sha256(json.dumps(candidate, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    if actual != expected:
        raise ValueError("V2 collector companion canonical seal mismatch")


def inspect_full_line_contexts(raw, *, encoded, exact_parents):
    """Pinned actual companion → exact sidecar/request/profile/namespace audit.

    exact_parents comes from the loader's independently checked producer and
    prepared journal, never from a companion enrolling itself. Extra observed
    nonlearning contexts remain raw evidence; only exact matched current input
    bindings can support the training encoded view.
    """
    fields = ("domain", "input_sha256", "sidecar_sha256", "canonical_tensor_sha256", "process_epoch",
              "request_sequence", "model_profile", "model_semantics", "encoding_profile", "records",
              "raw_record_relationships", "relationship_links_omitted",
              "query_prefix", "query_proposal", "query_counter", "namespace", "checker_identity",
              "checker_profile_sha256", "task_utility_observations", "task_utility_mask_reason", "sha256")
    parsed, selected = [], {}
    for _, value in _collection_jsonl(raw, max_rows=65536 * 8):
        value = _fields(value, fields, "V2 actual full-line context")
        _sealed_struct(value)
        if value["domain"] != "rz-pals-native-full-line-context/1":
            raise ValueError("V2 actual full-line context domain")
        for name in ("input_sha256", "sidecar_sha256", "canonical_tensor_sha256", "checker_profile_sha256"):
            _sha(value[name])
        for name in ("process_epoch", "request_sequence"):
            _uint(value[name])
        from .config import ModelConfig
        config = ModelConfig.for_profile(value["model_profile"])
        if (not config.full_line or value["model_semantics"] != config.model_semantics
                or value["encoding_profile"] != config.encoding):
            raise ValueError("V2 actual context model/encoding profile mismatch")
        if not isinstance(value["records"], list) or len(value["records"]) > 128:
            raise ValueError("V2 actual context record allocation bound")
        def unpack(line):
            if not isinstance(line, list) or len(line) > 256:
                raise ValueError("V2 actual context line allocation bound")
            return [dict(zip(("from", "to", "promotion"), move_components(move))) for move in line]
        decoded_records = []
        for record in value["records"]:
            record = _fields(record, ("moves", "parent", "supersedes", "parent_required", "supersedes_required"),
                             "V2 actual packed full-line record")
            decoded_records.append({**record, "moves": unpack(record["moves"])})
        decoded = {"records": decoded_records, **{name: unpack(value[name]) for name in (
            "query_prefix", "query_proposal", "query_counter")}}
        _validate_full_line(decoded, len(decoded_records))
        relationships = value["raw_record_relationships"]
        if not isinstance(relationships, list) or len(relationships) != len(decoded_records):
            raise ValueError("V2 raw relationship order/extent differs from selected records")
        ids, omitted = set(), 0
        for relationship in relationships:
            _fields(relationship, ("record_id", "revision", "parent_revision", "supersedes_revision",
                                   "parent_omitted", "supersedes_omitted"), "V2 raw record relationship")
            record_id = _uint(relationship["record_id"])
            if not record_id or record_id in ids:
                raise ValueError("V2 raw relationship source IDs must be unique and nonzero")
            ids.add(record_id)
            _uint(relationship["revision"])
            for name in ("parent", "supersedes"):
                if relationship[name + "_revision"] is not None:
                    _uint(relationship[name + "_revision"])
                if type(relationship[name + "_omitted"]) is not bool:
                    raise ValueError("V2 raw relationship omitted flag must be boolean")
        for line, relationship in zip(decoded_records, relationships):
            for name in ("parent", "supersedes"):
                raw_revision, local = relationship[name + "_revision"], line[name]
                expected_omitted = raw_revision is not None and local is None
                if relationship[name + "_omitted"] != expected_omitted:
                    raise ValueError("V2 raw relationship omitted flag differs from actual projection")
                matches = [i for i, selected_record in enumerate(relationships)
                           if selected_record["revision"] == raw_revision] if raw_revision is not None else []
                if local is not None and (matches != [local] or relationships[local]["revision"] != raw_revision):
                    raise ValueError("V2 local relationship differs from unique selected raw revision")
                if expected_omitted and (matches or line[name + "_required"]):
                    raise ValueError("V2 omitted relationship must be absent and optional")
                omitted += int(expected_omitted)
        observed_omitted = _fields(value["relationship_links_omitted"], ("observation", "value"),
                                   "V2 actual omitted relationship count")
        if (observed_omitted["observation"] != "observed" or type(observed_omitted["value"]) is not int
                or observed_omitted["value"] != omitted):
            raise ValueError("V2 omitted relationship count differs from actual flags")
        observation = _fields(value["task_utility_observations"], ("observation", "value"), "actual P/C utility observation")
        if observation["observation"] != "observed" or type(observation["value"]) is not int or observation["value"] != 0:
            raise ValueError("P/C capture cannot invent private V utility evidence")
        _identity(value["task_utility_mask_reason"])
        namespace = value["namespace"]
        if not isinstance(namespace, dict):
            raise ValueError("V2 actual checker namespace must be explicit")
        kind = namespace.get("value_namespace")
        names = {"own_raw": ("value_namespace", "checker_profile_sha256"),
                 "fresh_model_wdl": ("value_namespace", "model_identity", "model_epoch_sha256", "encoding_sha256",
                                     "checker_kind", "checker_profile_sha256"),
                 "unknown": ("value_namespace", "reason")}.get(kind)
        if names is None:
            raise ValueError("V2 actual context unsupported checker value namespace")
        _fields(namespace, names, "actual checker value namespace")
        if kind == "unknown":
            _identity(namespace["reason"])
        else:
            if _sha(namespace["checker_profile_sha256"]) != value["checker_profile_sha256"]:
                raise ValueError("V2 actual checker profile differs across namespace/context")
        if kind == "fresh_model_wdl":
            _identity(namespace["model_identity"])
            _sha(namespace["model_epoch_sha256"])
            _sha(namespace["encoding_sha256"])
            if namespace["checker_kind"] not in ("own", "external_uci"):
                raise ValueError("V2 actual WDL unsupported checker kind")
        identity = value["input_sha256"]
        if identity in encoded and encoded[identity].full_line is not None:
            parent = exact_parents[identity]
            sidecar, source, request = parent["sidecar"], parent["source"], parent["request"]
            if (value["sidecar_sha256"] != sidecar["sha256"]
                    or value["canonical_tensor_sha256"] != sidecar["canonical_tensor_sha256"]
                    or request != [value["process_epoch"], value["request_sequence"]]):
                # A repeated but unselected raw invocation cannot enroll the
                # exact current prepared input simply by sharing its SHA.
                parsed.append(value)
                continue
            if identity in selected:
                raise ValueError("duplicate actual full-line companion for exact prepared input")
            if value["model_profile"] != encoded[identity].model_profile:
                raise ValueError("V2 companion profile differs from admitted sidecar")
            full = encoded[identity].full_line
            def packed(tokens):
                return [token["from"] | token["to"] << 6 | token["promotion"] << 12 for token in tokens]
            wanted_records = [{"moves": packed(record["moves"]), "parent": record["parent"],
                               "supersedes": record["supersedes"], "parent_required": record.get("parent_required", False),
                               "supersedes_required": record.get("supersedes_required", False)} for record in full["records"]]
            if (value["records"] != wanted_records or any(value[name] != packed(full[name])
                    for name in ("query_prefix", "query_proposal", "query_counter"))):
                raise ValueError("V2 companion exact packed moves/relationships differ from sidecar")
            tensor_records = json.loads(sidecar["tensor_json"])["records"]
            record_sources = sidecar["record_sources"]
            if (len(relationships) != len(tensor_records) or len(relationships) != len(record_sources)
                    or any(relationship["record_id"] != token["record_id"]
                           or relationship["revision"] != token["revision"]
                           or relationship["revision"] != record_source["situation_revision"]
                           for relationship, token, record_source in zip(relationships, tensor_records, record_sources))):
                raise ValueError("V2 raw relationship provenance differs from exact sidecar sources")
            native = source.get("native") or {}
            if (value["checker_profile_sha256"] != source["cpu_profile_sha256"]
                    or value["checker_identity"] != native.get("checker_identity")):
                raise ValueError("V2 companion checker identity differs from independent source")
            resolver, checker = native.get("resolver_policy"), native.get("checker_kind")
            if kind == "own_raw" and (resolver != "pals-cpu-raw-restricted/0.1" or checker != "own"):
                raise ValueError("V2 own raw namespace requires actual own checker/resolver")
            if kind == "fresh_model_wdl" and (resolver != "pals-model-wdl-restricted/0.1"
                    or checker != namespace["checker_kind"] or native.get("actual_model_identity") != namespace["model_identity"]
                    or bytes(source["model_epoch"]).hex() != namespace["model_epoch_sha256"]
                    or source["encoding_sha256"] != namespace["encoding_sha256"]):
                raise ValueError("V2 fresh WDL namespace differs from actual model/checker/encoding source")
            selected[identity] = value
        parsed.append(value)
    required = {identity for identity, value in encoded.items() if value.full_line is not None}
    if set(selected) != required:
        raise ValueError("V2 frozen data lacks exact actual full-line context companion")
    return {"domain": "rz-pals-full-line-context-admission/1", "raw_observed_contexts": len(parsed),
            "matched_inputs": {identity: {"context_sha256": value["sha256"], "namespace": value["namespace"],
                                         "checker_profile_sha256": value["checker_profile_sha256"],
                                         "raw_record_relationships": value["raw_record_relationships"],
                                         "relationship_links_omitted": value["relationship_links_omitted"],
                                         "supported_private_v_utility": {"observation": "observed", "value": 0},
                                         "utility_mask_reason": value["task_utility_mask_reason"]}
                               for identity, value in sorted(selected.items())},
            "learned_scorer_support_admitted": False, "hashes_are_features": False}


def inspect_collector_target_coverage(raw, *, dataset):
    """Cross-check the collector's observed counts against current raw labels."""
    rows = _collection_jsonl(raw, max_rows=1)
    if len(rows) != 1:
        raise ValueError("one complete actual V2 coverage receipt required")
    value = _fields(rows[0][1], ("domain", "source_kind", "model_profile", "current_inputs", "policy_targets",
                                "actual_outcome_targets", "repair_supported", "repair_refuted",
                                "verifier_supported_utility", "masked_reasons", "provenance", "scan_complete",
                                "target_admission", "sha256"), "actual collector target coverage")
    _sealed_struct(value)
    if (value["domain"] != "rz-pals-target-coverage-v2/1" or value["source_kind"] != "actual_collector"
            or value["scan_complete"] is not True):
        raise ValueError("actual coverage domain/source/scan scope")
    from .config import ModelConfig
    ModelConfig.for_profile(value["model_profile"])
    expected = dict.fromkeys(("current_inputs", "policy_targets", "actual_outcome_targets",
                             "repair_supported", "repair_refuted", "verifier_supported_utility"), 0)
    expected["current_inputs"] = len(dataset.current_view.current_indices)
    provenance = {}
    reasons = Counter()
    for index in dataset.current_view.current_indices:
        row = dataset.records[index]
        label = row["future_label"]
        reasons["private_V_utility_context_not_observed"] += 1
        if label is None:
            reasons["future_label_absent"] += 1
            continue
        if label["policy"] is None:
            reasons["policy_target_absent"] += 1
        if label["value_wdl"] is None:
            reasons["actual_outcome_unobserved_or_virtual_branch"] += 1
        expected["policy_targets"] += int(label["policy"] is not None)
        expected["actual_outcome_targets"] += int(label["value_wdl"] is not None)
        validity = label["counterexample"]["validity"] if label["counterexample"] else None
        expected["repair_supported"] += int(validity == "supported_after_repair")
        expected["repair_refuted"] += int(validity == "refuted_by_repair")
        if validity in (None, "not_examined", "disputed"):
            reasons[{None: "repair_target_absent", "not_examined": "repair_not_examined",
                     "disputed": "repair_disputed"}[validity]] += 1
        if label["verifier_tasks"] is not None:
            reasons["V_context_and_namespace_support_not_observed_by_PC_collector"] += 1
        provenance[row["input"]["sha256"]] = {"input_sha256": row["input"]["sha256"], "label_sha256": label_digest(row),
                                               "observed_sequence": label["observed_sequence"], "source": label["provenance"],
                                               "supersedes_label_sha256": label["supersedes_label_sha256"]}
    for name, count in expected.items():
        observed = _fields(value[name], ("observation", "value"), "actual observed coverage count")
        if observed["observation"] != "observed" or type(observed["value"]) is not int or observed["value"] != count:
            raise ValueError("actual collector coverage count differs from admitted current labels")
    if (not isinstance(value["provenance"], list) or len(value["provenance"]) != len(provenance)
            or any(not isinstance(item, dict) for item in value["provenance"])
            or {item.get("input_sha256"): item for item in value["provenance"]} != provenance):
        raise ValueError("actual collector coverage provenance differs from current labels")
    if not isinstance(value["masked_reasons"], dict) or len(value["masked_reasons"]) > 64:
        raise ValueError("actual collector mask reason extent")
    for reason, count in value["masked_reasons"].items():
        _identity(reason)
        _uint(count, 65536 * 4)
    if value["masked_reasons"] != dict(reasons):
        raise ValueError("actual collector mask reasons differ from current label observations")
    if not isinstance(value["target_admission"], str) or not value["target_admission"]:
        raise ValueError("actual collector target admission scope missing")
    return {"domain": value["domain"], "coverage_sha256": value["sha256"],
            "observed_counts": expected, "verifier_supported_utility": {"observation": "observed", "value": 0},
            "masked_reasons": value["masked_reasons"], "target_admission": value["target_admission"]}


def _reason(row, head, context):
    label = row["future_label"]
    if label is None:
        return "future_label_missing"
    if head == "policy":
        return "policy_target_missing" if label["policy"] is None else "known_owned_policy"
    if head == "wdl":
        if label["value_wdl"] is not None:
            return "known_actual_outcome_wdl"
        provenance = label["provenance"]
        if provenance["source"] == "owned_cpu":
            return "owned_cpu_score_has_no_actual_outcome_wdl"
        result = provenance["result"]
        return ("unknown_outcome_" + result["ending"]) if result["outcome"] == "unknown" else "actual_outcome_wdl_not_supplied"
    if head == "divergence":
        value = label["counterexample"]
        if value is None:
            return "repair_validity_target_missing"
        return {"not_examined": "repair_validity_not_examined", "disputed": "repair_validity_disputed",
                "supported_after_repair": "known_supported_after_repair",
                "refuted_by_repair": "known_refuted_by_repair"}[value["validity"]]
    if context is None:
        return "explicit_task_context_not_selected"
    tasks = label["verifier_tasks"] or []
    matching = [value for value in tasks if TaskContext.from_target(value) == context]
    if not matching:
        return "task_target_missing_for_exact_context"
    if not any(value["preference_rank"] is not None for value in matching):
        return "exact_context_tasks_unranked"
    return "known_ranked_tasks_in_exact_context"


def target_coverage_report(dataset, *, source_kind, split=None, task_contexts=None,
                           max_rows=65536, max_report_rows=4096):
    """Count masks that the validated collator actually exposes, without NN.

    task_contexts maps current row indices to an explicit selected TaskContext.
    Unselected V contexts stay masked; no context is inferred from a hash, row
    number or the first encoded query. Actual collector coverage requires the
    frozen producer admission. In-memory contract fixtures have their own tag.
    """
    if type(dataset) is not ValidatedDataset or source_kind not in ("actual_collector", "contract_fixture"):
        raise ValueError("target coverage requires exact validated dataset and source kind")
    if split not in (None, "train", "validation", "holdout") or not 1 <= _uint(max_rows, 65536):
        raise ValueError("target coverage split/row bound")
    _uint(max_report_rows, 65536)
    dataset._verify_raw_integrity()
    if source_kind == "actual_collector" and dataset.frozen_admission is None:
        raise ValueError("actual target coverage requires checked frozen collector admission")
    contexts = {} if task_contexts is None else dict(task_contexts)
    for index, context in contexts.items():
        _uint(index, len(dataset.records) - 1)
        if index not in dataset.current_view.current_indices or not isinstance(context, TaskContext):
            raise ValueError("coverage task context must identify an admitted current row")
        context.validate()
        if dataset.records[index]["input"]["snapshot"]["role"] != "verifier":
            raise ValueError("coverage V context cannot enter P/C rows")
    current = [index for index in dataset.current_view.current_indices
               if split is None or dataset.split[dataset.records[index]["input"]["snapshot"]["game_id"]] == split]
    if len(current) > max_rows:
        raise ValueError("target coverage current row budget exceeded")
    totals = {role: {"current_rows": 0, "rows_with_valid_target": 0, "trainable_indices": [],
                     "valid_target_counts": dict.fromkeys(heads, 0),
                     "mask_reasons": {head: Counter() for head in heads}}
              for role, heads in HEADS.items()}
    rows = []
    for index in current:
        row = dataset.records[index]
        snapshot = row["input"]["snapshot"]
        role, assignment = snapshot["role"], dataset.split[snapshot["game_id"]]
        context = contexts.get(index)
        encoded = dataset.encodings.get(row["input"]["sha256"])
        if encoded is None:
            raise ValueError("target coverage lacks exact captured encoded input")
        _validate_encoding(encoded)
        if _encoding_identity(encoded) != dataset._encoding_identities.get(row["input"]["sha256"]):
            raise ValueError("target coverage encoded input changed after admission")
        if role == "verifier" and context is None:
            counts = {"task": 0}
        else:
            batch = dataset.collate([index], role, split=assignment,
                                    task_contexts=[context] if role == "verifier" else None, device="cpu")
            counts = {"policy": int(batch.policy_mask.sum()), "wdl": int(batch.wdl_mask.sum())}
            if role == "critic":
                counts["divergence"] = int(batch.divergence_mask.sum())
            if role == "verifier":
                counts = {"task": int(batch.task_mask.sum())}
        selected = {head: counts[head] for head in HEADS[role]}
        reasons = {head: _reason(row, head, context) for head in HEADS[role]}
        total = totals[role]
        total["current_rows"] += 1
        valid = any(selected.values())
        total["rows_with_valid_target"] += int(valid)
        if valid and assignment == "train":
            total["trainable_indices"].append(index)
        for head, count in selected.items():
            total["valid_target_counts"][head] += count
            if count == 0:
                total["mask_reasons"][head][reasons[head]] += 1
        if len(rows) < max_report_rows:
            rows.append({"index": index, "role": role, "split": assignment,
                         "input_sha256": row["input"]["sha256"], "label_sha256": label_digest(row),
                         "target_counts": selected, "target_reasons": reasons,
                         "selected_task_context": asdict(context) if context else None,
                         "input_profile": dataset.encodings[row["input"]["sha256"]].model_profile,
                         "full_line_encoding_admitted": dataset.encodings[row["input"]["sha256"]].full_line is not None})
    for total in totals.values():
        total["mask_reasons"] = {head: dict(sorted(counts.items())) for head, counts in total["mask_reasons"].items()}
    dataset._verify_raw_integrity()
    report = {"schema": SCHEMA, "source_kind": source_kind, "split": split,
              "current_view_sha256": dataset.current_view.sha256,
              "frozen_admission_sha256": dataset._frozen_admission_identity,
              "current_rows": len(current), "historical_rows": len(dataset.records),
              "roles": totals, "rows": rows, "report_rows_omitted": max(0, len(current) - len(rows)),
              "actual_target_coverage_claimed": source_kind == "actual_collector",
              "diagnostic_fixture_counts_included": False, "targets_changed": False,
              "cp_to_wdl_conversion": False, "actual_training_executed": False,
              "arena_eligible": False,
              "verifier_utility_scope": "base_rank_targets_only_separate_checked_utility_not_inferred"}
    report["coverage_sha256"] = _sorted_canonical(SCHEMA, report)
    return report


def checked_pair_coverage(*, comparative=None, utility=()):
    """Audit existing capability masks independently from ordinary target heads."""
    result = {"comparative": {"valid_pairs": 0, "masked_pairs": 0, "mask_reasons": {}},
              "verifier_utility": {"valid_pairs": 0, "masked_pairs": 0, "mask_reasons": {}},
              "actual_training_executed": False, "targets_changed": False}
    if comparative is not None:
        from .comparative_training import CheckedComparativePairs
        if type(comparative) is not CheckedComparativePairs:
            raise ValueError("comparative coverage requires checked actual receipt capability")
        comparative.verify()
        reasons = Counter(target.reason for target in comparative.targets if not target.mask)
        result["comparative"] = {"valid_pairs": sum(target.mask for target in comparative.targets),
                                  "masked_pairs": sum(not target.mask for target in comparative.targets),
                                  "mask_reasons": dict(sorted(reasons.items()))}
    from .verifier_utility import CheckedVerifierUtilityPairs
    if not isinstance(utility, (tuple, list)) or len(utility) > 4096:
        raise ValueError("bounded utility capability set required")
    reasons = Counter()
    for checked in utility:
        if type(checked) is not CheckedVerifierUtilityPairs:
            raise ValueError("utility coverage requires checked independently observed receipt capability")
        checked.verify()
        target = checked.target
        result["verifier_utility"]["valid_pairs"] += int(target.mask)
        result["verifier_utility"]["masked_pairs"] += int(not target.mask)
        if not target.mask:
            reasons[target.reason] += 1
    result["verifier_utility"]["mask_reasons"] = dict(sorted(reasons.items()))
    return result


def checker_value_namespace(value):
    """Explicit V2 data namespace; identities are never encoded as features.

    Own raw values require the own checker. Fresh model WDL supports either an
    own or external checker when the exact model/encoding identity matches.
    Admission of receipt execution and action utility remains a separate gate.
    Legacy strategic scorer support stays explicitly reported as unsupported
    for these new namespaces until an action-query/receipt adapter is supplied.
    """
    value = _fields(value, ("schema", "checker_kind", "checker_binary_sha256", "profile_sha256",
                            "search_implementation_sha256", "value_kind", "value_semantics_sha256",
                            "model_configuration_sha256", "model_weights_sha256", "encoding_sha256"),
                    "V checker value namespace")
    if value["schema"] != CHECKER_NAMESPACE_SCHEMA or value["checker_kind"] not in ("own", "external_uci"):
        raise ValueError("unsupported V checker namespace")
    for name in ("checker_binary_sha256", "profile_sha256", "search_implementation_sha256", "value_semantics_sha256"):
        _sha(value[name])
    if value["value_kind"] == "own_raw":
        if value["checker_kind"] != "own" or any(value[name] is not None for name in (
                "model_configuration_sha256", "model_weights_sha256", "encoding_sha256")):
            raise ValueError("own raw V value namespace requires own checker and no neural namespace")
    elif value["value_kind"] == "fresh_model_wdl":
        for name in ("model_configuration_sha256", "model_weights_sha256", "encoding_sha256"):
            _sha(value[name])
    else:
        raise ValueError("CP/mate/external scores cannot enter own raw or WDL namespace")
    return {"namespace": value, "namespace_sha256": _sorted_canonical(CHECKER_NAMESPACE_SCHEMA, value),
            "legacy_scorer_compatible": False, "receipt_execution_admitted": False,
            "learned_utility_admitted": False, "hashes_are_features": False}
