"""Structural dataset audits with explicit exclusions and missing engine evidence."""

import hashlib
from pathlib import Path

from .errors import DataError
from .io import Limits, read_jsonl
from .schema import validate_manifest, validate_record
from .serialization import digest
from .splits import audit_leakage, validate_split_plan


def audit_dataset(manifest: dict, records_path: Path, plan: dict, limits: Limits) -> dict:
    manifest = validate_manifest(manifest)
    plan = validate_split_plan(plan)
    if len(plan["sources"]) > limits.max_records:
        raise DataError("SourceLimit", "split_plan.sources", "source count exceeds configured record limit")
    accepted, rejected, excluded = [], [], []
    input_hash = hashlib.sha256()
    seen = 0
    for line, record, parse_error in read_jsonl(records_path, limits, hasher=input_hash):
        seen += 1
        identity = {key: record.get(key) for key in ("record_id", "game_id")
                    if isinstance(record, dict) and isinstance(record.get(key), str)}
        try:
            if parse_error is not None:
                raise parse_error
            validate_record(record, manifest)
        except DataError as exc:
            rejected.append({"line": line, **identity, "error": exc.as_dict()})
            continue
        accepted.append(record)
        if record["label"]["status"] != "complete":
            excluded.append({"line": line, **identity, "reason": "LabelNotComplete",
                             "status": record["label"]["status"],
                             "failure_reason": record["label"]["failure_reason"]})
    leakage = audit_leakage(accepted, plan)
    errors = list(leakage["errors"])
    for scope in ("source", "teacher"):
        if manifest[scope]["rights"]["status"] != "confirmed":
            errors.append({"code": "RightsUnverified", "context": "manifest." + scope,
                           "message": "source or teacher rights are unresolved"})
    if seen == 0:
        errors.append({"code": "EmptyDataset", "context": "records", "message": "no records supplied"})
    if seen > manifest["teacher"]["budget"]["max_positions"]:
        errors.append({"code": "BudgetExceeded", "context": "records",
                       "message": "record count exceeds declared teacher position budget"})
    report = {
        "schema_version": 1,
        "task_id": "TASK-F01",
        "dataset_id": manifest["dataset_id"],
        "source_kind": manifest["kind"],
        "manifest_digest": digest(manifest),
        "records_file_digest": input_hash.hexdigest(),
        "split_plan_digest": plan["digest"],
        "engine_contract_revision": manifest["engine_contract_revision"],
        "limits": vars(limits),
        "structural_audit_passed": not rejected and not errors,
        "execution_ready": False,
        "training_eligible_records": 0,
        "counts": {"input_records": seen, "structurally_valid_records": len(accepted),
                   "rejected_records": len(rejected), "excluded_labels": len(excluded),
                   "complete_label_candidates": len(accepted) - len(excluded)},
        "errors": errors,
        "rejections": rejected,
        "exclusions": excluded,
        "leakage": leakage,
        "coverage": {
            "engine_contract_binding": "not_run",
            "state_restore": "not_run",
            "independent_legal_moves": "not_run",
            "actual_encoding_identity": "not_run",
            "augmentation_rules_equivalence": "not_run",
            "teacher_label_quality": "not_run",
            "artifact_rights": "manifest_declarations_only",
        },
        "limitations": manifest["leakage_policy"],
        "blocking_reasons": ["I/A/C bindings and independent state/encoding evidence are unavailable"],
    }
    report["digest"] = digest(report)
    return report
