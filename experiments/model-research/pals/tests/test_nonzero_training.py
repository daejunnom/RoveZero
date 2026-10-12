"""Nonzero checkpoint/admission contracts without an actual optimizer update.

Manufactured tiny AdamW moments are serialization fixtures, not evidence of
training. The single scheduled CLI smoke owns the positive 24-update budget.
Every test patches AdamW.step to fail, so running this suite consumes zero
actual optimizer updates and performs no neural forward/backward.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rz_pals_model.config import ModelConfig
from rz_pals_model.nonzero_training import (CHECKPOINT_SCHEMA, TrainingLedger, _check_optimizer,
                                          _diagnostic_batch, _json_bytes, comparison_snapshot, initialize_training,
                                          load_nonzero_checkpoint, save_nonzero_checkpoint,
                                          smoke_spec, state_equal, train_one_update)
from rz_pals_model.training import ResumableSampler, capture_rng, restore_rng
from test_training import TinyResponsibilityModel, sha


def spec():
    return smoke_spec(ModelConfig(), implementation_sha256=sha("nonzero-contract-code"),
                      dataset_sha256=sha("synthetic-checkpoint-moments-only"),
                      split_sha256=sha("diagnostic-split"))


def manufactured_state(model, role="proposer"):
    """No optimizer.step: manufacture moments for loader shape/state testing."""
    state = initialize_training(model, role, spec())
    for group in state.preparation.optimizer.param_groups:
        for parameter in group["params"]:
            state.preparation.optimizer.state[parameter] = {
                "step": torch.tensor(2.0), "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.ones_like(parameter)}
    state.steps = 2
    state.scheduler.last_epoch = 2
    state.scheduler._step_count = 3
    return state


def ledger():
    value = TrainingLedger(spec())
    value.usage.update(updates_dispatched=2, updates_completed=2, forward_calls=2, backward_calls=2, samples=2)
    return value


def repin(path, receipt, mutate):
    payload = torch.load(path, weights_only=True, map_location="cpu")
    mutate(payload)
    target = path.with_name("corrupt.pt")
    torch.save(payload, target)
    changed = copy.deepcopy(receipt)
    changed.update(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                   bytes=target.stat().st_size)
    size = 0
    for _ in range(8):
        changed["receipt_bytes"] = size
        changed["usage"]["output_bytes"] = payload["usage"]["output_bytes"] + size
        next_size = len(_json_bytes(changed))
        if next_size == size:
            break
        size = next_size
    return target, changed


class NonzeroCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.rng = capture_rng()
        self.guard = patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("no actual optimizer update in contract tests"))
        self.guard.start()

    def tearDown(self):
        self.guard.stop()
        restore_rng(self.rng)

    def test_optimizer_role_membership_and_all_selected_decay_are_locked(self):
        for role, active in (("proposer", "proposer"), ("critic", "critic"), ("verifier", "validator")):
            model = TinyResponsibilityModel()
            state = initialize_training(model, role, spec())
            names = state.preparation.parameter_names
            self.assertTrue(any(name.startswith("experts." + active + ".") for name in names))
            self.assertEqual(any(name.startswith("public_encoder.") for name in names), role != "verifier")
            self.assertTrue(all(group["weight_decay"] == 0.01 for group in state.preparation.optimizer.param_groups))
            self.assertEqual(state.preparation.optimizer.state, {})
            _check_optimizer(model, state)

    def test_zero_step_domain_is_rejected_before_artifact_creation(self):
        model = TinyResponsibilityModel()
        state = initialize_training(model, "proposer", spec())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "zero.pt"
            with self.assertRaisesRegex(ValueError, "at least one actual"):
                save_nonzero_checkpoint(path, model, state, ResumableSampler((0, 1), 3), TrainingLedger(spec()))
            self.assertFalse(path.exists())

    def test_comparison_snapshot_uses_equal_model_map_types_and_exact_tensors(self):
        source = TinyResponsibilityModel()
        source_state = manufactured_state(source)
        source_sampler = ResumableSampler((0, 1, 2, 3), 17)
        destination = copy.deepcopy(source)
        destination_state = manufactured_state(destination)
        destination_sampler = ResumableSampler((0, 1, 2, 3), 17)
        reference = comparison_snapshot(source, source_state, source_sampler)
        candidate = comparison_snapshot(destination, destination_state, destination_sampler)
        self.assertIs(type(reference["model"]), dict)
        self.assertIs(type(candidate["model"]), dict)
        self.assertTrue(state_equal(reference, candidate))
        first = next(iter(candidate["model"]))
        candidate["model"][first].view(-1)[0] += 1
        self.assertFalse(state_equal(reference["model"], candidate["model"]))
        self.assertTrue(state_equal(source.state_dict(), destination.state_dict()))

    def test_manufactured_moment_roundtrip_restores_all_state_without_step(self):
        for role in ("proposer", "critic", "verifier"):
            source = TinyResponsibilityModel()
            state = manufactured_state(source, role)
            sampler = ResumableSampler((0, 1, 2, 3), 17, role=role)
            sampler.next_batch(1)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "state.pt"
                receipt = save_nonzero_checkpoint(path, source, state, sampler, ledger())
                self.assertEqual(receipt["schema"], CHECKPOINT_SCHEMA)
                self.assertFalse(receipt["arena_eligible"])
                self.assertEqual(receipt["receipt_bytes"], path.with_name("state-receipt.json").stat().st_size)
                caller = TinyResponsibilityModel()
                before_rng = torch.load(path, weights_only=True)["rng"]
                restored, selected, usage = load_nonzero_checkpoint(path, receipt["sha256"], caller, spec(),
                                                                  usage_receipt=receipt, indices=(0, 1, 2, 3), role=role)
                self.assertTrue(state_equal(source.state_dict(), caller.state_dict()))
                self.assertTrue(state_equal(state.preparation.optimizer.state_dict(), restored.preparation.optimizer.state_dict()))
                self.assertTrue(state_equal(state.scheduler.state_dict(), restored.scheduler.state_dict()))
                self.assertTrue(state_equal(before_rng, capture_rng()))
                self.assertEqual(sampler.state(), selected.state())
                self.assertEqual(restored.steps, 2)
                self.assertEqual(usage.usage["updates_completed"], 2)

    def test_corrupt_resume_atomically_preserves_parameters_gradients_freeze_rng(self):
        def bad_moment(payload):
            next(iter(payload["optimizer"]["state"].values()))["exp_avg_sq"] = torch.tensor([-1.0])
        def bad_dtype(payload):
            item = next(iter(payload["optimizer"]["state"].values()))
            item["exp_avg"] = item["exp_avg"].double()
        def bad_step(payload):
            next(iter(payload["optimizer"]["state"].values()))["step"] = torch.tensor(float("nan"))
        mutations = {
            "preparation domain": lambda payload: payload.update(schema="rz-pals-python-preparation-checkpoint/1"),
            "missing actual moments": lambda payload: payload["optimizer"].update(state={}),
            "raw moment shape": bad_moment, "raw moment dtype": bad_dtype, "nonfinite step": bad_step,
            "constant scheduler": lambda payload: payload["scheduler"].update(last_epoch=7),
            "actual freeze": lambda payload: payload["requires_grad"].update({next(iter(payload["requires_grad"])): False}),
            "parameter names": lambda payload: payload["parameter_names"].append("unknown.parameter"),
            "sampler cursor": lambda payload: payload["sampler"].update(cursor=9),
            "full rng": lambda payload: payload["rng"].update(torch_cpu=torch.zeros(2, dtype=torch.uint8)),
            "training steps": lambda payload: payload.update(training_steps=0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                source = TinyResponsibilityModel()
                state = manufactured_state(source)
                path = Path(temporary) / "state.pt"
                receipt = save_nonzero_checkpoint(path, source, state, ResumableSampler((0, 1), 17), ledger())
                corrupt, repinned = repin(path, receipt, mutate)
                caller = TinyResponsibilityModel()
                for index, parameter in enumerate(caller.parameters()):
                    parameter.requires_grad_(index % 2 == 0)
                    parameter.grad = torch.full_like(parameter, 0.75)
                before_model = copy.deepcopy(caller.state_dict())
                before_grad = [parameter.grad.clone() for parameter in caller.parameters()]
                before_freeze = [parameter.requires_grad for parameter in caller.parameters()]
                before_rng = capture_rng()
                with self.assertRaises((ValueError, RuntimeError)):
                    load_nonzero_checkpoint(corrupt, repinned["sha256"], caller, spec(),
                                            usage_receipt=repinned, indices=(0, 1), role="proposer")
                self.assertTrue(state_equal(before_model, caller.state_dict()))
                self.assertTrue(all(torch.equal(a, parameter.grad) for a, parameter in zip(before_grad, caller.parameters())))
                self.assertEqual(before_freeze, [parameter.requires_grad for parameter in caller.parameters()])
                self.assertTrue(state_equal(before_rng, capture_rng()))

    def test_all_masked_batch_refused_before_any_nn_backward_or_step(self):
        model = TinyResponsibilityModel()
        state = initialize_training(model, "proposer", spec())
        batch = _diagnostic_batch(model, "proposer", 0)
        batch = replace(batch, policy_mask=torch.tensor([False]), wdl_mask=torch.tensor([False]))
        usage = TrainingLedger(spec())
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "all loss masks"), patch.object(model.public_encoder, "forward", side_effect=AssertionError("forward forbidden")):
            train_one_update(model, state, batch, usage)
        self.assertEqual(usage.usage["updates_dispatched"], 0)
        self.assertEqual(usage.usage["forward_calls"], 0)
        self.assertTrue(state_equal(before, model.state_dict()))

    def test_global_update_usage_cannot_reset_or_exceed_cap(self):
        value = ledger()
        value.usage["updates_dispatched"] = 24
        with self.assertRaisesRegex(ValueError, "budget/counter"):
            value.reserve(update=True)
        invalid = spec()
        invalid["optimizer"]["batch_size"] = True
        with self.assertRaisesRegex(ValueError, "unsupported nonzero"):
            TrainingLedger(invalid)

    def test_progress_boundaries_are_immutable_and_preserve_consumed_allowance(self):
        value = TrainingLedger(spec())
        with tempfile.TemporaryDirectory() as temporary:
            value.progress_path = Path(temporary) / "progress.json"
            value.usage.update(forward_calls=1, backward_calls=1, samples=1)
            value.reserve(update=True)
            value.persist_progress("optimizer_dispatched", "proposer", 0)
            dispatched = value.progress_path.with_name("progress-01-optimizer_dispatched.json")
            original = dispatched.read_bytes()
            # Only counters are manufactured; AdamW.step remains forbidden.
            value.usage["updates_completed"] = 1
            value.persist_progress("optimizer_completed", "proposer", 1)
            completed = value.progress_path.with_name("progress-01-optimizer_completed.json")
            self.assertEqual(original, dispatched.read_bytes())
            self.assertEqual(json.loads(original)["usage"]["updates_dispatched"], 1)
            self.assertEqual(json.loads(completed.read_bytes())["usage"]["updates_completed"], 1)
            self.assertEqual(value.usage["output_bytes"], dispatched.stat().st_size + completed.stat().st_size)
            with self.assertRaisesRegex(FileExistsError, "already registered"):
                value.persist_progress("optimizer_completed", "proposer", 1)

    def test_fabricated_batch_cannot_claim_actual_collector_targets(self):
        actual = spec()
        actual["source_kind"] = "strict_collector"
        model = TinyResponsibilityModel()
        state = initialize_training(model, "proposer", actual)
        usage = TrainingLedger(actual)
        with self.assertRaisesRegex(ValueError, "run_validated_update"):
            train_one_update(model, state, _diagnostic_batch(model, "proposer", 0), usage)
        self.assertEqual(usage.usage["updates_dispatched"], 0)
        self.assertEqual(usage.usage["forward_calls"], 0)

    def test_malformed_mask_refused_before_nn_dispatch(self):
        model = TinyResponsibilityModel()
        state = initialize_training(model, "proposer", spec())
        batch = _diagnostic_batch(model, "proposer", 0)
        batch = replace(batch, policy_mask=torch.tensor([1.0]))
        usage = TrainingLedger(spec())
        with self.assertRaisesRegex(ValueError, "exact bool"), patch.object(model.public_encoder, "forward", side_effect=AssertionError("forward forbidden")):
            train_one_update(model, state, batch, usage)
        self.assertEqual(usage.usage["forward_calls"], 0)


if __name__ == "__main__":
    unittest.main()
