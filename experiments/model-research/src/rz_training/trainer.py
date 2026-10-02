"""Finite CPU fixture training, validation selection, resume and native export."""

import math
import os
from pathlib import Path
import re
import time

from rz_data.errors import DataError
from rz_data.io import read_json, write_run
from rz_data.serialization import canonical_bytes, digest

from .checkpoint import Sampler, code_digest, seal, verify
from .data import load_fixture_data
from .recipe import ADAPTER_ID, fields, integer, number, sha256, validate_recipe
from .reference import LinearFixture


class RunWriter:
    """Append immutable JSON artifacts to a fresh run, with a total storage cap."""

    def __init__(self, output_root, run_id, source_root, recipe):
        self.limit = recipe["budget"]["max_output_bytes"]
        self.run = write_run(output_root, run_id, {"recipe.json": recipe}, source_root=source_root, max_bytes=self.limit)
        self.used = (self.run / "recipe.json").stat().st_size

    def write(self, name, value):
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.json", name):
            raise DataError("InvalidOutputName", "output", "expected a simple JSON filename")
        data = canonical_bytes(value, max_bytes=self.limit - self.used - 1) + b"\n"
        temporary = self.run / ("." + name + ".partial")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
            # Publish an exclusive complete artifact; interrupted writes cannot
            # leave a valid-looking partial checkpoint under its final name.
            os.link(temporary, self.run / name)
        finally:
            temporary.unlink(missing_ok=True)
        self.used += len(data)


def _evaluate(model, samples, recipe):
    """Only callers with train/validation samples can invoke this path."""
    loss = 0.0
    for start in range(0, len(samples), 256):
        batch = _targets(samples[start:start + 256])
        batch_loss, _ = model.loss_and_grad(batch, recipe["loss"]["policy_weight"], recipe["loss"]["teacher_wdl_weight"])
        loss += batch_loss * len(batch)
    loss /= len(samples)
    if not math.isfinite(loss):
        raise DataError("NumericalFailure", "validation", "validation loss is nonfinite")
    return loss


def _targets(samples):
    return [{key: row[key] for key in ("features", "policy", "wdl")} for row in samples]


def _checkpoint_payload(provenance, model, optimizer, sampler, progress, best, history):
    return {"schema_version": 1, "artifact_kind": "training_checkpoint", "provenance": provenance,
            "model": model.state_dict(), "optimizer": optimizer, "sampler": sampler.state_dict(),
            "progress": progress, "best": best, "history": history}


def _restore(value, provenance, recipe, model, sampler, baseline_loss):
    payload = verify(value, artifact_kind="training_checkpoint", provenance=provenance,
                     max_bytes=recipe["budget"]["max_output_bytes"])
    fields(payload, ("schema_version", "artifact_kind", "provenance", "model", "optimizer", "sampler",
                     "progress", "best", "history"), "checkpoint")
    progress = fields(payload["progress"], ("steps", "samples", "elapsed_ms"), "checkpoint.progress")
    integer(progress["steps"], 0, recipe["budget"]["max_steps"], "checkpoint.steps")
    integer(progress["samples"], 0, recipe["budget"]["max_samples"], "checkpoint.samples")
    number(progress["elapsed_ms"], 0, 2**63 - 1, "checkpoint.elapsed_ms")
    if not progress["steps"] <= progress["samples"] <= progress["steps"] * recipe["batch_size"]:
        raise DataError("InvalidProgress", "checkpoint.progress", "sample/step accounting does not match recipe")
    model.load_state_dict(payload["model"])
    # Freeze still validates the complete optimizer state, without changing it.
    model.apply_gradient(model.zero_optimizer_state()["velocity"], payload["optimizer"],
                         learning_rate=recipe["optimizer"]["learning_rate"], momentum=recipe["optimizer"]["momentum"], freeze=True)
    sampler.load_state_dict(payload["sampler"])
    best = fields(payload["best"], ("step", "validation_loss", "model"), "checkpoint.best")
    integer(best["step"], 0, progress["steps"], "checkpoint.best.step")
    number(best["validation_loss"], 0, 1e12, "checkpoint.best.validation_loss")
    clone = LinearFixture(recipe["adapter"]["feature_width"], recipe["adapter"]["policy_moves"], recipe["seed"])
    clone.load_state_dict(best["model"])
    history = payload["history"]
    if type(history) is not list or len(history) != progress["steps"]:
        raise DataError("InvalidProgress", "checkpoint.history", "one metric per completed step is required")
    samples = 0
    for step, entry in enumerate(history, 1):
        fields(entry, ("step", "batch_samples", "train_loss", "validation_loss"), "checkpoint.history")
        integer(entry["step"], step, step, "checkpoint.history.step")
        integer(entry["batch_samples"], 1, recipe["batch_size"], "checkpoint.history.batch_samples")
        number(entry["train_loss"], 0, 1e12, "checkpoint.history.train_loss")
        number(entry["validation_loss"], 0, 1e12, "checkpoint.history.validation_loss")
        samples += entry["batch_samples"]
    if samples != progress["samples"]:
        raise DataError("InvalidProgress", "checkpoint.history", "sample ledger differs from cumulative progress")
    if sampler.epoch * sampler.size + sampler.cursor != samples:
        raise DataError("InvalidProgress", "checkpoint.sampler", "sampler cursor differs from cumulative samples")
    candidates = [(0, baseline_loss)] + [(entry["step"], entry["validation_loss"]) for entry in history]
    expected_step, expected_loss = min(candidates, key=lambda pair: (pair[1], pair[0]))
    if best["step"] != expected_step or best["validation_loss"] != expected_loss:
        raise DataError("InvalidSelection", "checkpoint.best", "selected step differs from earliest validation minimum")
    return payload["optimizer"], progress, best, history


def train_fixture(recipe, manifest_path, records_path, plan_path, features_path, *,
                  output_root: Path, run_id: str, source_root: Path, resume_path=None, stop_after=None):
    invocation_start = time.monotonic_ns()
    validate_recipe(recipe)
    if stop_after is not None:
        integer(stop_after, 1, recipe["budget"]["max_steps"], "stop_after")
    data = load_fixture_data(recipe, manifest_path, records_path, plan_path, features_path)
    model = LinearFixture(recipe["adapter"]["feature_width"], recipe["adapter"]["policy_moves"], recipe["seed"])
    initial_weights_digest = digest(model.state_dict())
    sampler = Sampler(len(data["train"]), recipe["seed"])
    provenance = {"recipe_digest": digest(recipe), "dataset_digests": data["dataset_digests"],
                  "audit_digest": data["audit_digest"], "data_digest": data["data_digest"],
                  "implementation_digest": code_digest(), "adapter": model.descriptor(),
                  "execution_scope": "cpu_fixture", "engine_contract_binding": "not_run"}
    progress = {"steps": 0, "samples": 0, "elapsed_ms": 0.0}
    optimizer = model.zero_optimizer_state()
    history = []
    best = {"step": 0, "validation_loss": _evaluate(model, data["validation"], recipe), "model": model.state_dict()}
    parent_digest = None
    if resume_path is not None:
        saved = read_json(resume_path, max_bytes=recipe["budget"]["max_output_bytes"])
        optimizer, progress, best, history = _restore(saved, provenance, recipe, model, sampler, best["validation_loss"])
        parent_digest = saved["digest"]
        # Detect incorrect selected weights/loss rather than trusting a self-hashed claim.
        selected = LinearFixture(recipe["adapter"]["feature_width"], recipe["adapter"]["policy_moves"], recipe["seed"])
        selected.load_state_dict(best["model"])
        if _evaluate(selected, data["validation"], recipe) != best["validation_loss"]:
            raise DataError("IdentityMismatch", "checkpoint.best", "selected model does not reproduce its validation metric")
        if history and _evaluate(model, data["validation"], recipe) != history[-1]["validation_loss"]:
            raise DataError("IdentityMismatch", "checkpoint.model", "current model does not reproduce its last validation metric")
    # Reject a storage budget too small for the declared worst-case checkpoint ledger before updating weights.
    checkpoint_size = len(canonical_bytes(_checkpoint_payload(provenance, model, optimizer, sampler, progress, best, history)))
    tensor_elements = (recipe["adapter"]["feature_width"] + 1) * (len(recipe["adapter"]["policy_moves"]) + 3)
    # Each float64 JSON number needs at most 32 bytes, including a separator.
    # Current/best weights and momentum can grow longer than initialization.
    checkpoint_size += 3 * tensor_elements * 32
    count = recipe["budget"]["max_steps"] // recipe["checkpoint"]["every_steps"] + 2
    estimate = (checkpoint_size + recipe["budget"]["max_steps"] * 256 + 1024) * count
    probe_size = 8 * (recipe["adapter"]["feature_width"] + len(recipe["adapter"]["policy_moves"]) + 3) * 32
    estimate += len(canonical_bytes(recipe)) + checkpoint_size * 2 + probe_size + 32768
    estimate += len(canonical_bytes(data["exclusions"]))
    if estimate > recipe["budget"]["max_output_bytes"]:
        raise DataError("OutputLimit", "recipe.budget", "output budget cannot hold the declared checkpoint and metric ledger")
    writer = RunWriter(output_root, run_id, source_root, recipe)
    prior_ms, first_step, start = progress["elapsed_ms"], progress["steps"], invocation_start
    budget, status, failure, latest = recipe["budget"], "running", None, None

    def elapsed():
        return prior_ms + (time.monotonic_ns() - start) / 1_000_000

    def save_checkpoint():
        progress["elapsed_ms"] = elapsed()
        value = seal(_checkpoint_payload(provenance, model, optimizer, sampler, progress.copy(), best, history),
                     max_bytes=budget["max_output_bytes"])
        name = f"checkpoint-{progress['steps']:06d}.json"
        writer.write(name, value)
        return {"file": name, "digest": value["digest"]}

    try:
        while progress["steps"] < budget["max_steps"]:
            if progress["samples"] >= budget["max_samples"]:
                status = "sample_budget_reached"
                break
            if elapsed() >= budget["max_time_ms"]:
                status = "time_budget_reached"
                break
            if stop_after is not None and progress["steps"] - first_step >= stop_after:
                status = "paused"
                break
            # A step commits model, optimizer, sampler and metrics together.
            # Cancellation or numerical failure during validation rolls it back.
            rollback = (model.state_dict(), optimizer, sampler.state_dict(), progress.copy(), len(history), best)
            try:
                count_samples = min(recipe["batch_size"], budget["max_samples"] - progress["samples"])
                batch = [data["train"][index] for index in sampler.draw(count_samples)]
                train_loss, grads = model.loss_and_grad(_targets(batch), recipe["loss"]["policy_weight"], recipe["loss"]["teacher_wdl_weight"])
                proposed_optimizer = model.apply_gradient(grads, optimizer, learning_rate=recipe["optimizer"]["learning_rate"],
                                                          momentum=recipe["optimizer"]["momentum"], freeze=recipe["freeze"])
                validation_loss = _evaluate(model, data["validation"], recipe)
                optimizer = proposed_optimizer
                progress["steps"] += 1
                progress["samples"] += len(batch)
                history.append({"step": progress["steps"], "batch_samples": len(batch), "train_loss": train_loss,
                                "validation_loss": validation_loss})
                if validation_loss < best["validation_loss"]:
                    best = {"step": progress["steps"], "validation_loss": validation_loss, "model": model.state_dict()}
            except (DataError, KeyboardInterrupt):
                model.load_state_dict(rollback[0])
                optimizer = rollback[1]
                sampler.load_state_dict(rollback[2])
                progress = rollback[3]
                del history[rollback[4]:]
                best = rollback[5]
                raise
            if progress["steps"] % recipe["checkpoint"]["every_steps"] == 0:
                latest = save_checkpoint()
        else:
            status = "step_budget_reached"
        if elapsed() >= budget["max_time_ms"]:
            status = "time_budget_reached"
    except KeyboardInterrupt:
        status = "canceled"
    except DataError as exc:
        status, failure = "failed", exc.as_dict()
    if status != "failed" and (latest is None or latest["file"] != f"checkpoint-{progress['steps']:06d}.json"):
        latest = save_checkpoint()
    export = None
    try:
        if status not in ("failed", "canceled", "time_budget_reached"):
            selected = LinearFixture(recipe["adapter"]["feature_width"], recipe["adapter"]["policy_moves"], recipe["seed"])
            selected.load_state_dict(best["model"])
            probes = [{"record_id": row["record_id"], "features": row["features"], "expected": selected.predict(row["features"])}
                      for row in data["validation"][:8]]
            export = seal({"schema_version": 1, "artifact_kind": "fixture_reference_export", "provenance": provenance,
                           "descriptor": selected.descriptor(), "selected_step": best["step"],
                           "validation_loss": best["validation_loss"], "model": selected.state_dict(),
                           "checkpoint_digest": latest["digest"], "probes": probes,
                           "engine_compatibility": "not_run", "precision_conversion": "none_float64_fixture"},
                          max_bytes=budget["max_output_bytes"])
            verify_export(export)
            if elapsed() >= budget["max_time_ms"]:
                status, export = "time_budget_reached", None
            else:
                writer.write("export.json", export)
    except KeyboardInterrupt:
        status, export = "canceled", None
    except DataError as exc:
        status, failure, export = "failed", exc.as_dict(), None
    # The final continuation captures export/verification costs as well. Numbered
    # checkpoints preserve their earlier capture time and are never overwritten.
    if status != "failed":
        progress["elapsed_ms"] = elapsed()
        continuation = seal(_checkpoint_payload(provenance, model, optimizer, sampler, progress.copy(), best, history),
                            max_bytes=budget["max_output_bytes"])
        writer.write("resume-checkpoint.json", continuation)
        latest = {"file": "resume-checkpoint.json", "digest": continuation["digest"]}
    progress["elapsed_ms"] = elapsed()
    if status not in ("failed", "canceled") and progress["elapsed_ms"] >= budget["max_time_ms"]:
        status = "time_budget_reached"
    receipt = seal({"schema_version": 1, "artifact_kind": "training_receipt", "provenance": provenance,
                    "status": status, "execution_ready": False, "actual": progress, "failure": failure,
                    "parent_checkpoint_digest": parent_digest, "checkpoint": latest,
                    "initial_weights_digest": initial_weights_digest, "final_weights_digest": digest(model.state_dict()),
                    "weights_changed": initial_weights_digest != digest(model.state_dict()),
                    "selection": {"split": "validation", "step": best["step"], "loss": best["validation_loss"]},
                    "split_counts": data["split_counts"], "exclusions": data["exclusions"], "history": history,
                    "export_digest": None if export is None else export["digest"],
                    "coverage": {"cpu_fixture_gradient": "completed" if progress["steps"] else "not_run",
                                 "native_fixture_export": "completed" if export else "not_run",
                                 "holdout_forward_pass": "not_run", "rules_encoder_binding": "not_run",
                                 "maia_fine_tuning": "not_run", "engine_inference": "not_run",
                                 "gpu_training": "not_run", "paired_arena": "not_run"},
                    "time_limit_policy": "cooperative checks; checkpoints include costs through capture; final continuation includes export"},
                   max_bytes=budget["max_output_bytes"])
    writer.write("receipt.json", receipt)
    return writer.run, receipt


def verify_export(value):
    payload = verify(value, artifact_kind="fixture_reference_export")
    fields(payload, ("schema_version", "artifact_kind", "provenance", "descriptor", "selected_step",
                     "validation_loss", "model", "checkpoint_digest", "probes", "engine_compatibility",
                     "precision_conversion"), "export")
    descriptor = fields(payload["descriptor"], ("adapter_id", "feature_width", "policy_moves", "precision"), "export.descriptor")
    if (descriptor["adapter_id"] != ADAPTER_ID or descriptor["precision"] != "float64"
            or payload["engine_compatibility"] != "not_run"
            or payload["precision_conversion"] != "none_float64_fixture"):
        raise DataError("UnsupportedExport", "export", "only the native CPU fixture export is supported")
    model = LinearFixture(descriptor["feature_width"], descriptor["policy_moves"], 0)
    provenance = fields(payload["provenance"], ("recipe_digest", "dataset_digests", "audit_digest", "data_digest",
                                               "implementation_digest", "adapter", "execution_scope",
                                               "engine_contract_binding"), "export.provenance")
    for key in ("recipe_digest", "audit_digest", "data_digest", "implementation_digest"):
        sha256(provenance[key], "export.provenance." + key)
    dataset = fields(provenance["dataset_digests"], ("manifest_digest", "records_file_digest", "split_plan_digest",
                                                  "features_file_digest"), "export.provenance.dataset_digests")
    for key, item in dataset.items():
        sha256(item, "export.provenance.dataset_digests." + key)
    if (provenance["execution_scope"] != "cpu_fixture" or provenance["engine_contract_binding"] != "not_run"
            or canonical_bytes(provenance["adapter"]) != canonical_bytes(descriptor)
            or provenance["implementation_digest"] != code_digest()):
        raise DataError("ProvenanceMismatch", "export.provenance", "export scope, adapter or source differs from the verifier")
    integer(payload["selected_step"], 0, 10000, "export.selected_step")
    number(payload["validation_loss"], 0, 1e12, "export.validation_loss")
    sha256(payload["checkpoint_digest"], "export.checkpoint_digest")
    model.load_state_dict(payload["model"])
    probes = payload["probes"]
    if type(probes) is not list or not 1 <= len(probes) <= 8:
        raise DataError("InvalidProbes", "export.probes", "expected one to eight validation probes")
    seen = set()
    for probe in probes:
        fields(probe, ("record_id", "features", "expected"), "export.probe")
        identity = probe["record_id"]
        if type(identity) is not str or not 1 <= len(identity) <= 2048 or identity in seen:
            raise DataError("InvalidIdentity", "export.probe.record_id", "expected unique bounded validation record IDs")
        seen.add(identity)
        expected = fields(probe["expected"], ("policy", "wdl"), "export.probe.expected")
        for key, size in (("policy", len(descriptor["policy_moves"])), ("wdl", 3)):
            vector = expected[key]
            if type(vector) is not list or len(vector) != size:
                raise DataError("InvalidShape", "export.probe.expected", "probe shape differs from the descriptor")
            for item in vector:
                number(item, 0, 1, "export.probe.expected")
            if not math.isclose(math.fsum(vector), 1, rel_tol=0, abs_tol=1e-12):
                raise DataError("InvalidNormalization", "export.probe.expected", "probe probabilities must sum to one")
        actual = model.predict(probe["features"])
        if actual != probe["expected"]:
            raise DataError("ExportMismatch", "export.probes", "imported weights do not exactly reproduce saved float64 outputs")
    return {"schema_version": 1, "artifact_kind": "export_verification", "export_digest": value["digest"],
            "execution_scope": "cpu_fixture", "probes": len(probes), "max_absolute_error": 0.0,
            "source_implementation_match": True, "provenance_check": "syntax_and_source_only",
            "dataset_reaudit": "not_run", "checkpoint_selection_reaudit": "not_run",
            "engine_compatibility": "not_run", "gpu_inference": "not_run"}
