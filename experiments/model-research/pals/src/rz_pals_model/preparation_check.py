"""Bounded collected-data → frozen P/C forward → loss preparation check.

This command creates no optimizer, performs no backward pass and updates no
weights. It consumes every admitted collected P/C row exactly once, including
its immutable split. Targets and numeric loss values are preparation evidence,
not training, chess strength, backward-FLOPs or GPU acceptance evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .artifacts import atomic_json, load_checkpoint, output_directory
from .config import SCHEMA as MODEL_SCHEMA
from .training import load_collected_dataset, masked_losses

SCHEMA = "rz-pals-collected-preparation-check/1"


def _positive(value, maximum, description):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{description} must be a positive bounded integer")


def _parameter_digest(model):
    """Exact parameters/buffers and their names, shapes and FP32 byte content."""
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if value.device.type != "cpu" or value.dtype != torch.float32 or not torch.all(torch.isfinite(value)):
            raise ValueError("frozen first-profile check requires finite CPU FP32 model state")
        descriptor = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        raw = value.detach().contiguous().numpy().tobytes()
        result.update(len(descriptor).to_bytes(8, "little"))
        result.update(descriptor)
        result.update(len(raw).to_bytes(8, "little"))
        result.update(raw)
    return result.hexdigest()


def _run_preparation_check(*, collection, receipt_sha256, encoder_source_sha256, checkpoint, output,
                          max_records, max_wall_time_ms, max_output_bytes, max_input_bytes=64 * 1024 * 1024,
                          batch_size=1, threads=1, require_masked_value=False, _started):
    _positive(max_records, 65536, "record limit")
    _positive(max_wall_time_ms, 300000, "wall time limit")
    _positive(max_output_bytes, 16 * 1024 * 1024, "report output limit")
    _positive(max_input_bytes, 1024 * 1024 * 1024, "input byte limit")
    _positive(batch_size, 256, "batch size")
    _positive(threads, 32, "CPU thread limit")
    if type(require_masked_value) is not bool:
        raise ValueError("masked-value expectation must be explicit boolean")
    started = _started

    def admit_time():
        elapsed = time.monotonic() - started
        if elapsed < 0 or not math.isfinite(elapsed) or elapsed * 1000 > max_wall_time_ms:
            raise ValueError("collected preparation check exceeded finite wall time")
        return math.ceil(elapsed * 1000)

    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.suffix != ".pt":
        raise ValueError("frozen check checkpoint must use .pt")
    checkpoint_metadata = checkpoint.with_name("checkpoint.json")
    model_input_bytes = 0
    for path in (checkpoint, checkpoint_metadata):
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint and provenance metadata must be regular files")
        model_input_bytes += path.stat().st_size
    if model_input_bytes >= max_input_bytes:
        raise ValueError("checkpoint and metadata exceed registered input byte limit")
    collection_input_bytes = sum((Path(collection) / name).stat().st_size for name in (
        "receipt.json", "records.jsonl", "native-inputs.jsonl", "source-registry.jsonl", "split.jsonl"))
    if model_input_bytes + collection_input_bytes > max_input_bytes:
        raise ValueError("collection plus checkpoint exceed registered input byte limit")
    data = load_collected_dataset(collection, expected_receipt_sha256=receipt_sha256,
                                  expected_encoder_source_sha256=encoder_source_sha256,
                                  max_input_bytes=max_input_bytes - model_input_bytes)
    if not 1 <= len(data.records) <= max_records:
        raise ValueError("collection rows exceed registered check limit")
    if any(row["input"]["snapshot"]["role"] not in ("proposer", "critic") for row in data.records):
        raise ValueError("this collected frozen check accepts P/C rows only")
    admit_time()
    model, metadata = load_checkpoint(checkpoint)
    if metadata.get("trained") is not False or metadata.get("training_steps") != 0:
        raise ValueError("this preparation check requires an untrained zero-step checkpoint")
    # Operate on a local P/C model view. No V private module reaches forward.
    if "validator" in model.experts:
        del model.experts["validator"]
    if tuple(model.experts) != ("proposer", "critic"):
        raise ValueError("checkpoint must contain both frozen P/C experts")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    before = _parameter_digest(model)
    original_threads = torch.get_num_threads()
    consumed, observations = set(), []
    counts = {"proposer": 0, "critic": 0}
    masks = {"policy_rows": 0, "wdl_rows": 0, "masked_wdl_rows": 0, "divergence_slots": 0}
    output_bytes = 0
    try:
        torch.set_num_threads(threads)
        with torch.inference_mode():
            for role in ("proposer", "critic"):
                for split in ("train", "validation", "holdout"):
                    indices = data.indices(role, split)
                    for offset in range(0, len(indices), batch_size):
                        admit_time()
                        selected = indices[offset:offset + batch_size]
                        if any(index in consumed for index in selected):
                            raise ValueError("collected row would be consumed more than once")
                        batch = data.collate(selected, role, split=split, device="cpu")
                        memory = model.public_encoder(*batch.inputs.public_args())
                        actual = model.role_graph(batch.role)(*batch.inputs.role_args(memory))
                        loss = masked_losses(actual, batch)
                        if any(value.requires_grad for value in actual) or any(value.requires_grad for value in loss.values()):
                            raise ValueError("frozen preparation unexpectedly produced autograd work")
                        if any(not torch.all(torch.isfinite(value)) for value in (*memory[:2], *actual, *loss.values())):
                            raise ValueError("collected preparation produced nonfinite neural values")
                        if require_masked_value and batch.wdl_mask.any():
                            raise ValueError("registered unknown-outcome collection supplied a value target")
                        policy_rows = int(batch.policy_mask.sum())
                        wdl_rows = int(batch.wdl_mask.sum())
                        divergence_slots = int(batch.divergence_mask.sum())
                        counts[role] += len(selected)
                        masks["policy_rows"] += policy_rows
                        masks["wdl_rows"] += wdl_rows
                        masks["masked_wdl_rows"] += len(selected) - wdl_rows
                        masks["divergence_slots"] += divergence_slots
                        observations.append({"indices": selected, "role": role, "split": split,
                                             "input_sha256": list(batch.input_sha256), "rows": len(selected),
                                             "policy_rows": policy_rows, "wdl_rows": wdl_rows,
                                             "masked_wdl_rows": len(selected) - wdl_rows,
                                             "divergence_slots": divergence_slots,
                                             "candidate_slots": int(batch.inputs.candidate_mask.sum()),
                                             "public_record_slots": int(batch.inputs.record_mask.sum()),
                                             "loss": {name: float(value) for name, value in loss.items()}})
                        consumed.update(selected)
                        admit_time()
        if consumed != set(range(len(data.records))):
            raise ValueError("not all collected records were consumed exactly once")
        after = _parameter_digest(model)
        if before != after or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("frozen check changed parameters or created gradients")
        elapsed = admit_time()
        report = {"schema": SCHEMA, "status": "checks_passed_before_report_write", "training_executed": False, "backward_executed": False,
                  "optimizer_created": False, "optimizer_steps": 0, "gpu_executed": False,
                  "provider": "pytorch_cpu_fp32", "trained": False, "training_steps": 0,
                  "checkpoint_sha256": metadata["checkpoint_sha256"], "collection_receipt_sha256": receipt_sha256,
                  "encoder_source_sha256": encoder_source_sha256, "records": len(data.records),
                  "records_consumed": len(consumed), "consumed_exactly_once": True, "role_counts": counts,
                  "target_masks": masks, "require_masked_value": require_masked_value,
                  "parameters_sha256_before": before, "parameters_sha256_after": after,
                  "parameters_unchanged": True, "validator_present": False, "batches": observations,
                  "resources": {"cpu_threads": threads, "batch_size": batch_size, "max_records": max_records,
                                "max_input_bytes": max_input_bytes, "input_bytes_preflight": model_input_bytes + collection_input_bytes,
                                "max_wall_time_ms": max_wall_time_ms,
                                "max_output_bytes": max_output_bytes, "wall_time_ms_before_report_write": elapsed},
                  "flops": {"status": "not_measured", "backward": "not_executed"}}
        # Keep the known report owner marker within the same output byte bound.
        output = Path(output).expanduser().resolve()
        if output.suffix != ".json":
            raise ValueError("preparation check report must use .json")
        encoded = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        marker = output.parent / "pals-artifact-owner.json"
        marker_bytes = marker.stat().st_size if marker.exists() else len((json.dumps({"schema": MODEL_SCHEMA, "owner": "rz-pals-model"}, sort_keys=True, indent=2) + "\n").encode())
        output_bytes = len(encoded) + marker_bytes
        if output_bytes > max_output_bytes:
            raise ValueError("preparation report exceeds registered output byte limit")
        admit_time()
        directory = output_directory(output.parent)
        atomic_json(output, report)
        admit_time()
        return {"status": "passed", "report": str(output), "records_consumed": len(consumed),
                "masked_wdl_rows": masks["masked_wdl_rows"], "parameters_unchanged": True,
                "optimizer_steps": 0, "output_bytes_including_owner_marker": output_bytes,
                "wall_time_ms": admit_time()}
    finally:
        torch.set_num_threads(original_threads)


def run_preparation_check(*, threads=1, **configuration):
    """Apply CPU admission before loading/creating/checking any neural tensor."""
    _positive(threads, 32, "CPU thread limit")
    started = time.monotonic()
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        result = _run_preparation_check(threads=threads, _started=started, **configuration)
    finally:
        torch.set_num_threads(original_threads)
    elapsed = math.ceil((time.monotonic() - started) * 1000)
    if elapsed < 0 or elapsed > configuration["max_wall_time_ms"]:
        raise ValueError("collected preparation check including cleanup exceeded wall time")
    result["wall_time_ms"] = elapsed
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--encoder-source-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, required=True)
    parser.add_argument("--max-wall-time-ms", type=int, required=True)
    parser.add_argument("--max-output-bytes", type=int, required=True)
    parser.add_argument("--max-input-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--require-masked-value", action="store_true")
    args = parser.parse_args(argv)
    result = run_preparation_check(collection=args.collection, receipt_sha256=args.receipt_sha256,
                                   encoder_source_sha256=args.encoder_source_sha256, checkpoint=args.checkpoint,
                                   output=args.output, max_records=args.max_records, max_wall_time_ms=args.max_wall_time_ms,
                                   max_output_bytes=args.max_output_bytes, max_input_bytes=args.max_input_bytes,
                                   batch_size=args.batch_size, threads=args.threads, require_masked_value=args.require_masked_value)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
