"""Consume F01 audit inputs without implementing Rules or the C encoder."""

import hashlib
from pathlib import Path

from rz_data.audit import audit_dataset
from rz_data.errors import DataError
from rz_data.io import Limits, read_json, read_jsonl
from rz_data.serialization import digest

from .recipe import ADAPTER_ID, fields, integer, number, validate_recipe


def load_fixture_data(recipe, manifest_path: Path, records_path: Path,
                      plan_path: Path, features_path: Path):
    validate_recipe(recipe)
    budget = recipe["budget"]
    limits = Limits(max_records=budget["max_records"], max_record_bytes=min(262144, budget["max_input_bytes"]),
                    max_file_bytes=budget["max_input_bytes"], max_output_bytes=budget["max_output_bytes"])
    manifest = read_json(manifest_path, max_bytes=limits.max_file_bytes)
    plan = read_json(plan_path, max_bytes=limits.max_file_bytes)
    report = audit_dataset(manifest, records_path, plan, limits)
    if manifest["kind"] != "fixture":
        raise DataError("UnsupportedScope", "manifest.kind", "real teacher/self-play data requires verified A/C bindings")
    if not report["structural_audit_passed"]:
        raise DataError("DatasetAuditFailed", "dataset", "F01 structural or leakage audit failed")
    expected = recipe["dataset"]
    observed = {"manifest_digest": report["manifest_digest"], "records_file_digest": report["records_file_digest"],
                "split_plan_digest": report["split_plan_digest"]}
    # Hash the exact bounded bytes that are parsed; do not re-open for a digest.
    with features_path.open("rb") as stream:
        raw = stream.read(limits.max_file_bytes + 1)
    if len(raw) > limits.max_file_bytes:
        raise DataError("FileLimit", "features", "feature fixture exceeds input byte limit")
    observed["features_file_digest"] = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise DataError("IdentityMismatch", "recipe.dataset", "locked dataset digests do not match supplied inputs")
    from rz_data.serialization import loads
    feature_file = fields(loads(raw), ("schema_version", "dataset_id", "adapter_id", "rows"), "features")
    integer(feature_file["schema_version"], 1, 1, "features.schema_version")
    if feature_file["dataset_id"] != manifest["dataset_id"] or feature_file["adapter_id"] != ADAPTER_ID:
        raise DataError("IdentityMismatch", "features", "feature dataset/adapter does not match recipe")
    rows = feature_file["rows"]
    if type(rows) is not list or len(rows) > limits.max_records:
        raise DataError("RecordLimit", "features.rows", "expected bounded feature rows")
    features = {}
    for row in rows:
        fields(row, ("record_id", "input_digest", "values"), "features.row")
        identity = row["record_id"]
        if type(identity) is not str or not 1 <= len(identity) <= 2048 or identity in features:
            raise DataError("InvalidIdentity", "features.record_id", "feature record IDs must be nonempty and unique")
        if type(row["values"]) is not list or len(row["values"]) != recipe["adapter"]["feature_width"]:
            raise DataError("InvalidShape", "features.values", "feature width does not match adapter")
        for value in row["values"]:
            number(value, -1000, 1000, "features.values")
        features[identity] = row
    train, validation, exclusions, identities = [], [], [], set()
    second_hash = hashlib.sha256()
    for _, record, error in read_jsonl(records_path, limits, hasher=second_hash):
        if error is not None:
            raise error
        identity = record["record_id"]
        identities.add(identity)
        feature = features.get(identity)
        if feature is None or feature["input_digest"] != record["input_identity"]["digest"]:
            raise DataError("IdentityMismatch", "features", "each row must reference the declared full input identity")
        if record["split"] == "holdout" or record["label"]["status"] != "complete":
            exclusions.append({"record_id": identity, "split": record["split"], "reason":
                               "HoldoutNotUsed" if record["split"] == "holdout" else "LabelNotComplete"})
            continue
        label = record["label"]
        policy, value = label["policy"], label["value"]
        if (policy is None or value is None or policy["kind"] != "probabilities"
                or value["kind"] != "wdl" or value["viewpoint"] != "side_to_move"):
            raise DataError("UnsupportedTarget", "label", "fixture adapter requires explicit policy probabilities and teacher WDL")
        if policy["moves"] != recipe["adapter"]["policy_moves"]:
            raise DataError("IdentityMismatch", "label.policy.moves", "fixture adapter uses one locked legal move order")
        sample = {"record_id": identity, "features": feature["values"], "policy": policy["values"], "wdl": value["value"],
                  "failure_groups": record["failure_groups"]}
        (train if record["split"] == "train" else validation).append(sample)
    if second_hash.hexdigest() != observed["records_file_digest"]:
        raise DataError("IdentityMismatch", "records", "records changed between audit and training read")
    if set(features) != identities:
        raise DataError("IdentityMismatch", "features", "feature rows must exactly cover the audited record set")
    if not train or not validation:
        raise DataError("EmptySplit", "dataset", "nonempty train and validation splits are required")
    # Canonical record order plus the saved sampler RNG defines reproducible batches.
    train.sort(key=lambda row: row["record_id"])
    validation.sort(key=lambda row: row["record_id"])
    return {"train": train, "validation": validation, "exclusions": exclusions,
            "audit_digest": report["digest"], "dataset_digests": observed,
            "split_counts": {"train": len(train), "validation": len(validation),
                             "holdout": sum(item["reason"] == "HoldoutNotUsed" for item in exclusions)},
            "feature_semantics": "authored numeric fixture; no chess encoder executed",
            "data_digest": digest({"train": train, "validation": validation, "exclusions": exclusions})}
