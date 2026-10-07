"""Meaningful end-to-end fixture invariants, including interrupted optimization."""

import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from rz_data.errors import DataError
from rz_data.serialization import canonical_bytes, digest, loads
from rz_training.checkpoint import seal
from rz_training.cli import main
from rz_training.data import load_fixture_data
from rz_training.recipe import validate_recipe
from rz_training.reference import LinearFixture
from rz_training.trainer import train_fixture, verify_export


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE.parents[1]
FIXTURE = PACKAGE / "fixtures" / "training"


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rz-f02-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {}
        for key, name in (("manifest", "manifest.json"), ("records", "records.jsonl"),
                          ("plan", "split-plan.json"), ("features", "features.json")):
            path = self.root / name
            path.write_bytes((FIXTURE / name).read_bytes())
            self.paths[key] = path
        self.recipe = loads((FIXTURE / "recipe.json").read_bytes())

    def run_training(self, name, *, recipe=None, **kwargs):
        return train_fixture(recipe or self.recipe, self.paths["manifest"], self.paths["records"],
                             self.paths["plan"], self.paths["features"], output_root=self.root / "runs",
                             run_id=name, source_root=SOURCE, **kwargs)

    def read(self, path):
        return loads(path.read_bytes())

    def reseal(self, value):
        return seal({key: item for key, item in value.items() if key != "digest"}, max_bytes=1048576)

    def test_gradients_change_weights_and_export_outputs_reproduce(self):
        run, receipt = self.run_training("full")
        checkpoint = self.read(run / "checkpoint-000008.json")
        initial = LinearFixture(2, self.recipe["adapter"]["policy_moves"], 17)
        self.assertNotEqual(initial.state_dict(), checkpoint["model"])
        self.assertTrue(receipt["weights_changed"])
        self.assertEqual(receipt["actual"]["samples"], 16)
        self.assertLess(receipt["history"][-1]["validation_loss"], receipt["history"][0]["validation_loss"])
        model = LinearFixture(2, self.recipe["adapter"]["policy_moves"], 17)
        model.load_state_dict(checkpoint["model"])
        self.assertNotEqual(initial.predict([1.0, 0.0]), model.predict([1.0, 0.0]))
        result = verify_export(self.read(run / "export.json"))
        self.assertEqual(result["max_absolute_error"], 0)
        self.assertEqual(result["engine_compatibility"], "not_run")

    def test_resume_matches_continuous_model_optimizer_rng_history_and_selection(self):
        continuous, _ = self.run_training("continuous")
        paused, pause_receipt = self.run_training("paused", stop_after=3)
        self.assertEqual(pause_receipt["status"], "paused")
        resume_path = paused / "checkpoint-000003.json"
        # A tiny fixture may fit within one Windows monotonic clock tick. Test
        # elapsed-time accumulation with a deterministic clock, not CPU speed.
        ticks = iter(range(0, 1_000_000_000, 1_000_000))
        with patch("rz_training.trainer.time.monotonic_ns", side_effect=lambda: next(ticks)):
            resumed, resume_receipt = self.run_training("resumed", resume_path=resume_path)
        left, right = self.read(continuous / "checkpoint-000008.json"), self.read(resumed / "checkpoint-000008.json")
        for field in ("provenance", "model", "optimizer", "sampler", "history", "best"):
            self.assertEqual(left[field], right[field], field)
        self.assertEqual(left["progress"]["steps"], right["progress"]["steps"])
        self.assertEqual(left["progress"]["samples"], right["progress"]["samples"])
        self.assertGreater(right["progress"]["elapsed_ms"], self.read(resume_path)["progress"]["elapsed_ms"])
        self.assertEqual(resume_receipt["parent_checkpoint_digest"], self.read(resume_path)["digest"])

    def test_frozen_recipe_keeps_weights_outputs_and_optimizer_unchanged(self):
        recipe = copy.deepcopy(self.recipe)
        recipe["freeze"] = True
        run, receipt = self.run_training("frozen", recipe=recipe)
        checkpoint = self.read(run / "checkpoint-000008.json")
        initial = LinearFixture(2, recipe["adapter"]["policy_moves"], recipe["seed"])
        self.assertEqual(checkpoint["model"], initial.state_dict())
        self.assertEqual(checkpoint["optimizer"], initial.zero_optimizer_state())
        self.assertFalse(receipt["weights_changed"])
        self.assertEqual(receipt["selection"]["step"], 0)
        exported = self.read(run / "export.json")
        for probe in exported["probes"]:
            self.assertEqual(probe["expected"], initial.predict(probe["features"]))

    def test_holdout_change_never_changes_training_or_selection(self):
        first, receipt = self.run_training("original")
        feature_file = self.read(self.paths["features"])
        feature_file["rows"][1]["values"] = [1000, -1000]
        self.paths["features"].write_bytes(canonical_bytes(feature_file) + b"\n")
        self.recipe["dataset"]["features_file_digest"] = hashlib.sha256(self.paths["features"].read_bytes()).hexdigest()
        second, modified = self.run_training("changed-holdout")
        self.assertEqual(receipt["history"], modified["history"])
        self.assertEqual(receipt["selection"], modified["selection"])
        self.assertEqual(receipt["final_weights_digest"], modified["final_weights_digest"])
        self.assertEqual(modified["coverage"]["holdout_forward_pass"], "not_run")
        self.assertNotIn("train-fixture-row-1", [p["record_id"] for p in self.read(second / "export.json")["probes"]])

    def test_locked_recipe_and_data_changes_reject_resume(self):
        paused, _ = self.run_training("paused", stop_after=2)
        self.recipe["optimizer"]["learning_rate"] = 0.2
        with self.assertRaisesRegex(DataError, "different source/configuration/data"):
            self.run_training("changed-recipe", resume_path=paused / "checkpoint-000002.json")
        self.assertFalse((self.root / "runs" / "changed-recipe").exists())
        self.paths["features"].write_bytes(self.paths["features"].read_bytes() + b" ")
        with self.assertRaises(DataError) as error:
            self.run_training("changed-data")
        self.assertEqual(error.exception.code, "IdentityMismatch")

    def test_sample_budget_is_cumulative_and_final_partial_batch_is_recorded(self):
        recipe = copy.deepcopy(self.recipe)
        recipe["budget"]["max_samples"] = 3
        run, receipt = self.run_training("samples", recipe=recipe)
        self.assertEqual(receipt["status"], "sample_budget_reached")
        self.assertEqual(receipt["actual"]["samples"], 3)
        self.assertEqual([entry["batch_samples"] for entry in receipt["history"]], [2, 1])
        _, resumed = self.run_training("samples-resumed", recipe=recipe, resume_path=run / "checkpoint-000002.json")
        self.assertEqual(resumed["actual"]["steps"], 2)
        self.assertEqual(resumed["actual"]["samples"], 3)

    def test_time_budget_resume_cannot_reset_elapsed_time(self):
        paused, _ = self.run_training("paused", stop_after=2)
        checkpoint = self.read(paused / "checkpoint-000002.json")
        checkpoint["progress"]["elapsed_ms"] = 10000
        checkpoint_path = self.root / "expired.json"
        checkpoint_path.write_bytes(canonical_bytes(self.reseal(checkpoint)))
        run, receipt = self.run_training("expired", resume_path=checkpoint_path)
        self.assertEqual(receipt["status"], "time_budget_reached")
        self.assertEqual(receipt["actual"]["steps"], 2)
        self.assertFalse((run / "export.json").exists())

    def test_cancellation_during_validation_rolls_back_entire_step_and_can_resume(self):
        from rz_training import trainer
        original = trainer._evaluate
        calls = 0

        def interrupt(model, samples, recipe):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return original(model, samples, recipe)

        with patch.object(trainer, "_evaluate", side_effect=interrupt):
            canceled, receipt = self.run_training("canceled")
        self.assertEqual(receipt["status"], "canceled")
        self.assertEqual(receipt["actual"]["steps"], 0)
        self.assertEqual(receipt["actual"]["samples"], 0)
        self.assertFalse(receipt["weights_changed"])
        resumed, _ = self.run_training("resumed", resume_path=canceled / "resume-checkpoint.json")
        continuous, _ = self.run_training("continuous")
        self.assertEqual(self.read(resumed / "checkpoint-000008.json")["model"],
                         self.read(continuous / "checkpoint-000008.json")["model"])

    def test_cancellation_after_sampler_draw_rolls_back_cursor_and_rng(self):
        from rz_training.checkpoint import Sampler
        draw = Sampler.draw

        def interrupt(sampler, count):
            draw(sampler, count)
            raise KeyboardInterrupt

        with patch.object(Sampler, "draw", new=interrupt):
            run, receipt = self.run_training("canceled-draw")
        checkpoint = self.read(run / "resume-checkpoint.json")
        self.assertEqual(receipt["status"], "canceled")
        self.assertEqual(checkpoint["sampler"], Sampler(3, 17).state_dict())
        self.run_training("resumed-draw", resume_path=run / "resume-checkpoint.json")

    def test_cancellation_after_checkpoint_publication_does_not_retry_existing_file(self):
        from rz_training.trainer import RunWriter
        write = RunWriter.write
        interrupted = False

        def interrupt(writer, name, value):
            nonlocal interrupted
            write(writer, name, value)
            if name.startswith("checkpoint-") and not interrupted:
                interrupted = True
                raise KeyboardInterrupt

        with patch.object(RunWriter, "write", new=interrupt):
            run, receipt = self.run_training("canceled-checkpoint")
        self.assertEqual(receipt["status"], "canceled")
        self.assertTrue((run / "checkpoint-000002.json").is_file())
        self.assertEqual(receipt["checkpoint"]["file"], "resume-checkpoint.json")
        self.run_training("resumed-checkpoint", resume_path=run / "resume-checkpoint.json")

    def test_export_cost_exceeding_time_budget_is_not_success_and_is_saved_for_resume(self):
        from rz_training import trainer
        clock = 0
        original = trainer.verify_export

        def costly_export(value):
            nonlocal clock
            result = original(value)
            clock = 20_000_000_000
            return result

        with patch.object(trainer.time, "monotonic_ns", side_effect=lambda: clock), \
                patch.object(trainer, "verify_export", side_effect=costly_export):
            run, receipt = self.run_training("expensive-export")
        self.assertEqual(receipt["status"], "time_budget_reached")
        self.assertEqual(receipt["actual"]["elapsed_ms"], 20000)
        self.assertFalse((run / "export.json").exists())
        self.assertEqual(self.read(run / "resume-checkpoint.json")["progress"]["elapsed_ms"], 20000)
        _, resumed = self.run_training("expired-export-resume", resume_path=run / "resume-checkpoint.json")
        self.assertEqual(resumed["status"], "time_budget_reached")

    def test_cancellation_during_export_preserves_continuation_and_canceled_receipt(self):
        with patch("rz_training.trainer.verify_export", side_effect=KeyboardInterrupt):
            run, receipt = self.run_training("canceled-export")
        self.assertEqual(receipt["status"], "canceled")
        self.assertEqual(receipt["actual"]["steps"], 8)
        self.assertFalse((run / "export.json").exists())
        self.assertTrue((run / "resume-checkpoint.json").is_file())
        _, resumed = self.run_training("resume-export", resume_path=run / "resume-checkpoint.json")
        self.assertEqual(resumed["status"], "step_budget_reached")

    def test_failure_preserves_last_checkpoint_and_typed_receipt(self):
        from rz_training import trainer
        original = trainer._evaluate
        calls = 0

        def fail(model, samples, recipe):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise DataError("NumericalFailure", "fixture", "injected failure")
            return original(model, samples, recipe)

        with patch.object(trainer, "_evaluate", side_effect=fail):
            run, receipt = self.run_training("failed")
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["actual"]["steps"], 2)
        self.assertEqual(receipt["failure"]["code"], "NumericalFailure")
        self.assertEqual(receipt["checkpoint"]["file"], "checkpoint-000002.json")
        self.assertFalse((run / "export.json").exists())

    def test_output_budget_preflight_and_immutable_run(self):
        recipe = copy.deepcopy(self.recipe)
        recipe["budget"]["max_output_bytes"] = 4096
        with self.assertRaises(DataError):
            self.run_training("small", recipe=recipe)
        self.assertFalse((self.root / "runs" / "small").exists())
        run, _ = self.run_training("immutable")
        original = (run / "receipt.json").read_bytes()
        with self.assertRaises(DataError) as error:
            self.run_training("immutable")
        self.assertEqual(error.exception.code, "RunExists")
        self.assertEqual((run / "receipt.json").read_bytes(), original)

    def test_receipt_exclusion_ledger_is_reserved_before_training(self):
        from rz_training import trainer
        original = trainer.load_fixture_data

        def large_ledger(*args, **kwargs):
            data = original(*args, **kwargs)
            data["exclusions"] = [{"record_id": "x" * 2000, "split": "holdout", "reason": "HoldoutNotUsed"}] * 60
            return data

        recipe = copy.deepcopy(self.recipe)
        recipe["budget"]["max_output_bytes"] = 150000
        with patch.object(trainer, "load_fixture_data", side_effect=large_ledger):
            with self.assertRaises(DataError) as error:
                self.run_training("large-ledger", recipe=recipe)
        self.assertEqual(error.exception.code, "OutputLimit")
        self.assertFalse((self.root / "runs" / "large-ledger").exists())

    def test_incomplete_labels_are_excluded_and_unsupported_targets_are_rejected(self):
        rows = [loads(line) for line in self.paths["records"].read_bytes().splitlines()]
        rows[2]["label"].update(status="partial", policy=None, value=None, failure_reason="fixture interruption")
        raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
        self.paths["records"].write_bytes(raw)
        self.recipe["dataset"]["records_file_digest"] = hashlib.sha256(raw).hexdigest()
        data = load_fixture_data(self.recipe, self.paths["manifest"], self.paths["records"], self.paths["plan"], self.paths["features"])
        self.assertEqual(len(data["train"]), 2)
        self.assertIn({"record_id": "train-fixture-row-2", "split": "train", "reason": "LabelNotComplete"}, data["exclusions"])
        rows[4]["label"]["value"] = {"kind": "centipawn", "value": 10, "viewpoint": "side_to_move", "scale": "centipawn"}
        raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
        self.paths["records"].write_bytes(raw)
        self.recipe["dataset"]["records_file_digest"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(DataError):
            load_fixture_data(self.recipe, self.paths["manifest"], self.paths["records"], self.paths["plan"], self.paths["features"])

    def test_semantic_checkpoint_tampering_rejects_even_after_rehash(self):
        paused, _ = self.run_training("paused", stop_after=2)
        original = self.read(paused / "checkpoint-000002.json")
        for kind in ("cursor", "selection", "optimizer", "steps", "source"):
            value = copy.deepcopy(original)
            if kind == "cursor":
                value["sampler"]["cursor"] = 0
            elif kind == "selection":
                value["best"]["step"] = 0
            elif kind == "optimizer":
                value["optimizer"]["velocity"]["policy_bias"] = []
            elif kind == "steps":
                value["progress"]["steps"] = True
            else:
                value["provenance"]["implementation_digest"] = "0" * 64
            path = self.root / (kind + ".json")
            path.write_bytes(canonical_bytes(self.reseal(value)))
            with self.subTest(kind=kind), self.assertRaises(DataError):
                self.run_training("tampered-" + kind, resume_path=path)

    def test_semantic_export_tampering_rejects_after_rehash(self):
        run, _ = self.run_training("export")
        original = self.read(run / "export.json")
        for key, replacement in (("selected_step", -1), ("validation_loss", -1),
                                 ("checkpoint_digest", "unrelated"), ("provenance", {"execution_scope": "production"})):
            value = copy.deepcopy(original)
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(DataError):
                verify_export(self.reseal(value))
        value = copy.deepcopy(original)
        value["model"]["policy_bias"][0] += 0.1
        with self.assertRaises(DataError):
            verify_export(self.reseal(value))

    def test_strict_recipe_rejects_unsupported_backend_target_and_bool_budgets(self):
        for key, value in (("backend", "cuda"), ("execution_ready", True), ("gradient_accumulation", 2),
                           ("workers", 2), ("schedule", {"kind": "adaptive"})):
            recipe = copy.deepcopy(self.recipe)
            recipe[key] = value
            with self.subTest(key=key), self.assertRaises(DataError):
                validate_recipe(recipe)
        recipe = copy.deepcopy(self.recipe)
        recipe["budget"]["max_steps"] = True
        with self.assertRaises(DataError):
            validate_recipe(recipe)

    def test_cli_run_and_verify_export_and_typed_io_failure(self):
        recipe_path = self.root / "recipe.json"
        recipe_path.write_bytes(canonical_bytes(self.recipe))
        args = ["run", "--recipe", str(recipe_path), "--manifest", str(self.paths["manifest"]),
                "--records", str(self.paths["records"]), "--split-plan", str(self.paths["plan"]),
                "--features", str(self.paths["features"]), "--output-root", str(self.root / "runs"), "--run-id", "cli"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
            self.assertEqual(main(["verify-export", "--export", str(self.root / "runs" / "cli" / "export.json"),
                                   "--output-root", str(self.root / "runs"), "--run-id", "verified"]), 0)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["verify-export", "--export", str(self.root / "missing-private-file"),
                                   "--output-root", str(self.root / "runs"), "--run-id", "missing"]), 2)
        self.assertEqual(loads(output.getvalue())["error"]["code"], "IoFailure")
        self.assertNotIn("missing-private-file", output.getvalue())


if __name__ == "__main__":
    unittest.main()
