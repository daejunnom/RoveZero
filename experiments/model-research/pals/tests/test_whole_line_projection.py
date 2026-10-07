"""Synthetic strict-factory Repair suffix wiring, never actual process proof.

The existing native/current/semantic/whole-line factories are exercised with
independently pinned synthetic bytes. Repeated move values and declared Rules
history are fixture descriptors, not Rust chess replay. The tiny frozen numeric
stand-in loads no checkpoint and proves no learned repair or strategic truth.
"""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from rz_pals_model import whole_line_ordinal as ordinal
from rz_pals_model import whole_line_projection as projection
from rz_pals_model.config import ModelConfig
from rz_pals_model.preparation_check import _parameter_digest
from test_current_consumers import FrozenNumericFixture
from test_repair_context import RepairFixture
from test_semantic_verifier import tokens
from test_training import sha
from test_whole_line_ordinal import WholeLineFixture


class ProjectionFixture:
    """Real admission factories with explicit synthetic caller/Rules facts."""
    def __init__(self, directory, *, direction="maximize_root_surrogate", complete_history=True):
        self.root = Path(directory)
        repair_dir = self.root / "synthetic-causal-repair"
        repair_dir.mkdir()
        self.repair = RepairFixture(repair_dir)
        # Coherent full-known-history descriptors are needed at BOTH factory
        # gates. These are intentionally synthetic, not a startpos replay.
        if complete_history:
            for check in self.repair.semantic_fixtures:
                for descriptor in (check.receipt["root"], check.receipt["target"], check.receipt["claimed_line"]["final_state"]):
                    descriptor.update(history_completeness="complete", history_origin="start_position",
                                      repetition_history_complete=True)
                check.reseal()
        self.repair.prefix_check, self.repair.proposal_check = [check.admit() for check in self.repair.semantic_fixtures]
        self.repair.context_and_pins()
        self.causal = self.repair.admit()
        self.parents, self.index = self.repair.parents, self.repair.repair_index
        self.lines = self.make_whole_fixture(self.index, direction=direction)
        self.lines.anchor = ordinal.input_anchor(self.parents, self.index, self.causal)
        self.lines.plan["anchor"] = copy.deepcopy(self.lines.anchor)
        self.rebind_plan()
        if not complete_history:
            for task in ("left", "right"):
                def unknown(receipt):
                    for descriptor in (receipt["root"], receipt["endpoint"]):
                        descriptor.update(history_completeness="unknown_prefix", history_origin="fen",
                                          repetition_history_complete=False)
                self.lines.change_receipt(task, unknown)
        self.refresh()

    def make_whole_fixture(self, index, *, direction="maximize_root_surrogate"):
        prefix_receipt = self.causal._checks[0].rules_receipt()
        descriptor = copy.deepcopy(prefix_receipt["target" if index == self.index else "root"])
        descriptor["legal_tokens"] = tokens(descriptor["legal_moves"], "root_legal", descriptor["legal_moves"])
        source = SimpleNamespace(parents=self.parents, index=index,
            registration={"rules_version": descriptor["rules_version"]}, receipt={"root": descriptor})
        # Only reuse the fixture's source descriptor constructor. No capability
        # factory, verify(), current selector or production validator is mocked.
        with patch("test_whole_line_ordinal.SemanticFixture", return_value=source):
            return WholeLineFixture(self.root, direction=direction)

    def rebind_plan(self):
        fixture = self.lines
        fixture.raws["plan"] = ordinal.canonical(fixture.plan)
        for planned in fixture.plan["lines"]:
            task = planned["task_id"]
            def request_update(request, planned=planned):
                request.update(before_result_anchor_sha256=ordinal.byte_pin(fixture.raws["plan"])["sha256"])
                request.update({name: copy.deepcopy(planned[name]) for name in ("line", "line_sha256", "expected_endpoint_board_fen",
                    "endpoint_rules_state_sha256", "endpoint_rules_history_sha256", "expected_endpoint_legal_moves", "endpoint_side_to_move")})
            fixture.change_request(task, request_update)
            request = json.loads(fixture.executions[task]["request"])
            fixture.requests[task] = request
            fixture.change_receipt(task, lambda receipt, request=request: receipt.update(
                before_result_anchor_sha256=request["before_result_anchor_sha256"], context_sha256=request["context_sha256"],
                line=request["line"], line_sha256=request["line_sha256"]), recompute_conditions=True)
        fixture.refresh_pins()

    def refresh(self):
        self.whole = self.lines.admit(query_anchor=self.causal)
        anchor = ordinal.input_anchor(self.parents, self.index, self.causal)
        tasks = [line["task_id"] for line in self.lines.plan["lines"]]
        self.before = {"schema": projection.BEFORE_SCHEMA, "scope": projection.SCOPE,
            "pair_id": self.lines.plan["pair_id"], "causal_admission_sha256": self.causal.sha256,
            "repair_anchor": anchor, "causal_context_sha256": self.causal.admission()["context_sha256"],
            "ordinal_plan_artifact": ordinal.byte_pin(self.lines.raws["plan"]),
            "ordinal_criterion_artifact": ordinal.byte_pin(self.lines.raws["criterion"]), "direction": "maximize_root_surrogate",
            "plan_root": "current_repair_state", "line_meaning": "suffix_from_current_repair_state",
            "ordered_task_ids": tasks, "first_moves": [line["line"][0] for line in self.lines.plan["lines"]]}
        self.raws = {"ordinal_plan": self.lines.raws["plan"], "ordinal_criterion": self.lines.raws["criterion"],
                     "projection_before": ordinal.canonical(self.before)}
        self.order = {"schema": projection.ORDER_SCHEMA, "projection_before_artifact": ordinal.byte_pin(self.raws["projection_before"]),
            "ordinal_plan_artifact": ordinal.byte_pin(self.raws["ordinal_plan"]),
            "ordinal_criterion_artifact": ordinal.byte_pin(self.raws["ordinal_criterion"]), "causal_admission_sha256": self.causal.sha256,
            "assurance_scope": "independently_pinned_caller_durable_projection_order_observation",
            "launches": [{"task_id": task, "launch_artifact": ordinal.byte_pin(self.lines.executions[task]["launch"]),
                          "projection_durable_before_spawn": True, "ordinal_plan_durable_before_spawn": True} for task in tasks]}
        self.raws["projection_order_observation"] = ordinal.canonical(self.order)
        self.refresh_pins()

    def refresh_pins(self):
        self.pins = {name: ordinal.byte_pin(value) for name, value in self.raws.items()}
        self.pins.update(causal_admission_sha256=self.causal.sha256, whole_admission_sha256=self.whole.sha256,
                         current_view_sha256=self.parents.current_view.sha256,
                         frozen_admission_sha256=self.parents._frozen_admission_identity)

    def change_before(self, mutate):
        mutate(self.before)
        self.raws["projection_before"] = ordinal.canonical(self.before)
        self.order["projection_before_artifact"] = ordinal.byte_pin(self.raws["projection_before"])
        self.raws["projection_order_observation"] = ordinal.canonical(self.order)
        self.refresh_pins()

    def change_order(self, mutate):
        mutate(self.order)
        self.raws["projection_order_observation"] = ordinal.canonical(self.order)
        self.refresh_pins()

    def same_first_different_suffix(self):
        left, right = self.lines.plan["lines"]
        right["line"] = [left["line"][0], right["line"][1]]
        right["line_sha256"] = ordinal.ordered_line_sha256(self.lines.anchor["rules_state_sha256"],
            self.lines.anchor["rules_history_sha256"], right["line"])
        self.rebind_plan()
        self.refresh()

    def terminal(self):
        self.lines.plan["lines"][0]["expected_endpoint_legal_moves"] = []
        self.rebind_plan()
        def endpoint_terminal(receipt):
            receipt["endpoint"].update(play_status="rules_terminal", terminal_reason="checkmate", terminal_winner="black",
                terminal_source="rz-position-rules", legal_moves=[], legal_tokens=[],
                legal_order_sha256=ordinal.rules.digest(ordinal.rules.MOVE_DOMAIN, []), in_check=True)
            receipt.update(status="rules_terminal", cpu_calls=0, fresh_engine=False, report=None)
        self.lines.change_receipt("left", endpoint_terminal, recompute_conditions=True)
        self.refresh()

    def admit(self, **changes):
        arguments = {name + "_bytes": value for name, value in self.raws.items()}
        arguments.update(causal=self.causal, whole=self.whole, expected_pins=self.pins)
        arguments.update(changes)
        return projection.admit_repair_first_move_projection(**arguments)

    def parameter_observation(self, model):
        snapshot = self.parents.records[self.index]["input"]["snapshot"]
        return ordinal.canonical({"schema": projection.PARAMETER_SCHEMA,
            "checkpoint_sha256": snapshot["source"]["model_weights_sha256"], "parameter_sha256": _parameter_digest(model),
            "frozen_epoch": snapshot["frozen_epoch"], "model_configuration": model.config.to_dict(),
            "assurance_scope": "independently_pinned_caller_checkpoint_reload_observation"})

    def prepare(self, model, *, checked=None, observation=None, **changes):
        observation = observation if observation is not None else self.parameter_observation(model)
        arguments = dict(parameter_observation_bytes=observation, expected_observation_pin=ordinal.byte_pin(observation), deadline=50.0)
        arguments.update(changes)
        with patch("rz_pals_model.whole_line_projection.time.monotonic", return_value=1.0):
            return projection.frozen_repair_projection_preparation(model, checked or self.admit(), **arguments)


class WholeLineProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = ProjectionFixture(self.temp.name)

    def test_exact_current_repair_suffix_scope_without_ordinary_or_truth_authority(self):
        checked = self.fixture.admit()
        self.assertIs(checked.verify(), checked)
        self.assertTrue(checked.target.mask)
        self.assertEqual(checked.target.sign, 1)
        self.assertEqual(checked.target.row_index, self.fixture.index)
        self.assertEqual(checked.target.input_sha256, self.fixture.repair.rows[1]["input"]["sha256"])
        self.assertEqual(checked.target.scope, "next_repair_move_whole_line_conditioned_surrogate")
        self.assertEqual(self.fixture.whole.outcome.raw_endpoint_scores, (90, 20))
        for field in ("ordinary_policy_admitted", "strategic_repair_validity_admitted", "counterexample_validity_admitted",
                      "wdl_admitted", "divergence_ranking_admitted", "minimax_admitted", "tactical_proof_admitted", "actual_training_executed"):
            self.assertIs(checked.admission[field], False)

    def test_actual74_like_propose_root_full_line_cannot_become_repair_positive(self):
        old = self.fixture.make_whole_fixture(self.fixture.repair.root_index).admit()
        self.assertEqual(old.admission["anchor"]["kind"], "ordinary_current")
        self.assertTrue(old.outcome.mask)
        pins = copy.deepcopy(self.fixture.pins)
        pins["whole_admission_sha256"] = old.sha256
        with self.assertRaisesRegex(ValueError, "CURRENT Repair row/anchor object"):
            self.fixture.admit(whole=old, expected_pins=pins)

    def test_ordinary_anchor_even_at_same_repair_row_is_not_causal_projection(self):
        fixture = self.fixture.lines
        fixture.anchor = ordinal.input_anchor(self.fixture.parents, self.fixture.index)
        fixture.plan["anchor"] = copy.deepcopy(fixture.anchor)
        self.fixture.rebind_plan()
        ordinary = fixture.admit()
        pins = copy.deepcopy(self.fixture.pins)
        pins["whole_admission_sha256"] = ordinary.sha256
        with self.assertRaisesRegex(ValueError, "CURRENT Repair row/anchor object"):
            self.fixture.admit(whole=ordinary, expected_pins=pins)

    def test_equal_causal_digest_from_different_object_is_not_dispatch_anchor(self):
        other = self.fixture.repair.admit()
        self.assertEqual(other.sha256, self.fixture.causal.sha256)
        self.assertIsNot(other, self.fixture.causal)
        with self.assertRaisesRegex(ValueError, "CURRENT Repair row/anchor object"):
            self.fixture.admit(causal=other)

    def test_unchecked_type_and_mutable_getter_cannot_inject_projection_authority(self):
        with self.assertRaisesRegex(ValueError, "unchecked"):
            projection.CheckedRepairWholeLineProjection()
        with self.assertRaisesRegex(ValueError, "exact factory"):
            self.fixture.admit(causal=lambda: self.fixture.causal)
        checked = self.fixture.admit()
        with self.assertRaises(AttributeError):
            checked._whole = None
        with self.assertRaises(AttributeError):
            checked.target.sign = -1
        copied = checked.admission
        copied["virtual_prefix"].clear()
        self.assertTrue(checked.admission["virtual_prefix"])

    def test_parent_epoch_revision_and_public_raw_are_rechecked_at_use(self):
        checked = self.fixture.admit()
        self.fixture.parents.records[self.fixture.index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaises(ValueError):
            checked.verify()

    def test_full_known_history_descriptor_must_match_causal_prefix_target(self):
        self.fixture.lines.change_receipt("left", lambda receipt: receipt["root"].update(known_repetition_count=2))
        self.fixture.refresh()
        self.assertTrue(self.fixture.whole.outcome.mask)
        with self.assertRaisesRegex(ValueError, "causal prefix Rules target/full known history"):
            self.fixture.admit()

    def test_minimize_c_direction_cannot_become_first_repair_preference(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProjectionFixture(directory, direction="minimize_root_surrogate")
            self.assertTrue(fixture.whole.outcome.mask)
            with self.assertRaisesRegex(ValueError, "maximize_root_surrogate"):
                fixture.admit()

    def test_projection_before_excludes_after_result_sha_score_and_sign(self):
        original = copy.deepcopy(self.fixture.before)
        for key, value in (("whole_admission_sha256", self.fixture.whole.sha256), ("score", 90), ("sign", 1)):
            with self.subTest(field=key):
                self.fixture.before = copy.deepcopy(original)
                self.fixture.change_before(lambda before: before.update({key: value}))
                with self.assertRaises(ValueError):
                    self.fixture.admit()

    def test_before_task_index_first_move_and_current_root_meaning_are_fixed(self):
        original = copy.deepcopy(self.fixture.before)
        mutations = (lambda before: before["ordered_task_ids"].reverse(), lambda before: before["first_moves"].reverse(),
                     lambda before: before.update(plan_root="old_propose_root"),
                     lambda before: before.update(line_meaning="full_root_proposal_counter"),
                     lambda before: before.update(causal_context_sha256=sha("other counterexample prefix")))
        for mutation in mutations:
            self.fixture.before = copy.deepcopy(original)
            self.fixture.change_before(mutation)
            with self.assertRaisesRegex(ValueError, "before plan/indices/context/actual bytes"):
                self.fixture.admit()

    def test_independent_plan_criterion_capability_and_current_pins_required(self):
        for name in projection._PIN_NAMES:
            with self.subTest(pin=name):
                pins = copy.deepcopy(self.fixture.pins)
                if type(pins[name]) is dict:
                    pins[name]["sha256"] = "0" * 64
                else:
                    pins[name] = "0" * 64
                with self.assertRaises(ValueError):
                    self.fixture.admit(expected_pins=pins)
        replacement = ordinal.canonical({**self.fixture.lines.criterion, "recipe_id": "changed after execution"})
        pins = copy.deepcopy(self.fixture.pins)
        pins["ordinal_criterion"] = ordinal.byte_pin(replacement)
        with self.assertRaisesRegex(ValueError, "original plan/criterion actual byte anchors"):
            self.fixture.admit(ordinal_criterion_bytes=replacement, expected_pins=pins)

    def test_durable_before_order_and_actual_launch_pin_required_even_when_masked(self):
        original = copy.deepcopy(self.fixture.order)
        for masked in (False, True):
            if masked:
                self.fixture.lines.change_receipt("left", lambda receipt: receipt["report"].update(raw_score=20))
                self.fixture.refresh()
                original = copy.deepcopy(self.fixture.order)
            mutations = (lambda order: order["launches"][0].update(projection_durable_before_spawn=False),
                         lambda order: order["launches"][0].update(ordinal_plan_durable_before_spawn=False),
                         lambda order: order["launches"].reverse(),
                         lambda order: order["launches"][0]["launch_artifact"].update(sha256="0" * 64))
            for mutate in mutations:
                with self.subTest(masked=masked):
                    self.fixture.order = copy.deepcopy(original)
                    self.fixture.change_order(mutate)
                    with self.assertRaisesRegex(ValueError, "durable projection ordering"):
                        self.fixture.admit()

    def test_same_first_move_different_suffix_is_unsupported_not_duplicate_logit_loss(self):
        self.fixture.same_first_different_suffix()
        self.assertNotEqual(*[line["line"] for line in self.fixture.lines.plan["lines"]])
        self.assertTrue(self.fixture.whole.outcome.mask)
        checked = self.fixture.admit()
        self.assertFalse(checked.target.mask)
        self.assertEqual(checked.target.sign, 0)
        self.assertEqual(checked.target.reason, "unsupported_same_first_move")
        with self.assertRaisesRegex(ValueError, "separate suffix scorer"):
            checked.collate()
        model = FrozenNumericFixture(on_forward=lambda: self.fail("all-masked forward executed")).eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "all-masked"):
            self.fixture.prepare(model, checked=checked)

    def test_partial_canceled_missing_and_score_masks_are_preserved(self):
        original = self.fixture.lines.executions["left"]["receipt"]
        for status, completion, reason in (("partial", "node_limit", "partial"), ("partial", "deadline", "partial"),
                                           ("canceled", "canceled", "canceled")):
            self.fixture.lines.executions["left"]["receipt"] = original
            self.fixture.lines.change_receipt("left", lambda receipt: (receipt.update(status=status),
                receipt["report"].update(completion=completion, completed_depth=1)))
            self.fixture.refresh()
            self.assertFalse(self.fixture.admit().target.mask)
            self.assertEqual(self.fixture.admit().target.reason, reason)
        for score, reason in ((29000, "mate_band"), (20001, "out_of_range"), (20, "tie_or_below_margin")):
            self.fixture.lines.executions["left"]["receipt"] = original
            self.fixture.lines.change_receipt("left", lambda receipt: receipt["report"].update(raw_score=score))
            self.fixture.refresh()
            self.assertEqual(self.fixture.admit().target.reason, reason)
            self.assertEqual(self.fixture.admit().target.sign, 0)
        self.fixture.lines.executions["left"]["receipt"] = None
        self.fixture.lines.observe("left")
        self.fixture.lines.refresh_pins()
        self.fixture.refresh()
        self.assertEqual(self.fixture.admit().target.reason, "missing")

    def test_terminal_endpoint_is_masked_without_fabricating_cpu_work(self):
        self.fixture.terminal()
        checked = self.fixture.admit()
        self.assertEqual(checked.target.reason, "terminal")
        self.assertFalse(checked.target.mask)
        self.assertIsNone(self.fixture.whole.outcome.raw_endpoint_scores[0])

    def test_consistent_unknown_prefix_is_masked_not_global_history_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProjectionFixture(directory, complete_history=False)
            checked = fixture.admit()
            self.assertEqual(checked.target.reason, "unknown_history_prefix")
            self.assertFalse(checked.target.mask)
            model = FrozenNumericFixture(on_forward=lambda: self.fail("unknown-history forward executed")).eval().requires_grad_(False)
            with self.assertRaisesRegex(ValueError, "all-masked"):
                fixture.prepare(model, checked=checked)

    def test_distinct_first_legal_indices_collate_without_mutating_ordinary_targets(self):
        checked = self.fixture.admit()
        batch = checked.collate()
        self.assertEqual(batch.candidate_slots, ((0, 1),))
        self.assertEqual(batch.base.input_sha256, (checked.target.input_sha256,))
        self.assertEqual(batch.base.role, "proposer")
        ordinary = self.fixture.parents.collate([self.fixture.index], "proposer", split="train", device="cpu")
        for name in ("policy", "policy_mask", "wdl", "wdl_mask", "divergence", "divergence_mask", "task", "task_mask"):
            self.assertTrue(torch.equal(getattr(batch.base, name), getattr(ordinary, name)))
        with torch.inference_mode():
            value = projection.numeric.pairwise_softplus_loss(torch.zeros_like(batch.base.policy), batch)
        self.assertAlmostEqual(float(value), math.log(2), places=6)
        self.assertFalse(value.requires_grad)

    def test_split_and_preallocation_byte_budget_refuse_before_collation(self):
        checked = self.fixture.admit()
        with patch.object(self.fixture.parents, "collate", side_effect=AssertionError("tensor allocation started")):
            with self.assertRaisesRegex(ValueError, "byte allowance"):
                checked.collate(max_tensor_bytes=1)
            with self.assertRaisesRegex(ValueError, "immutable game split"):
                checked.collate(split="holdout")

    def test_frozen_numeric_no_step_loss_checks_checkpoint_epoch_and_parameter_bytes(self):
        model = FrozenNumericFixture().eval().requires_grad_(False)
        before = _parameter_digest(model)
        report = self.fixture.prepare(model)
        self.assertAlmostEqual(report["pairwise_loss"], math.log(2), places=6)
        self.assertEqual(report["known_pairs"], 1)
        self.assertEqual(report["parameter_sha256_before"], before)
        self.assertEqual(report["parameter_sha256_after"], before)
        self.assertEqual(report["scope"], projection.SCOPE)
        for name in ("ordinary_policy_admitted", "strategic_repair_validity_admitted", "counterexample_validity_admitted",
                     "wdl_admitted", "divergence_ranking_admitted", "minimax_admitted", "tactical_proof_admitted",
                     "actual_training_executed", "backward_executed", "optimizer_created", "gpu_executed"):
            self.assertIs(report[name], False)
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()))

    def test_checkpoint_epoch_configuration_and_parameter_pin_mismatch_fail(self):
        model = FrozenNumericFixture().eval().requires_grad_(False)
        original = json.loads(self.fixture.parameter_observation(model))
        mutations = (lambda value: value.update(checkpoint_sha256=sha("different checkpoint")),
                     lambda value: value.update(parameter_sha256=sha("different frozen bytes")),
                     lambda value: value.update(frozen_epoch=value["frozen_epoch"] + 1),
                     lambda value: value["model_configuration"].update(iterations=3))
        for mutation in mutations:
            value = copy.deepcopy(original)
            mutation(value)
            with self.assertRaises(ValueError):
                self.fixture.prepare(model, observation=ordinal.canonical(value))
        raw = ordinal.canonical(original)
        with self.assertRaises(ValueError):
            self.fixture.prepare(model, observation=raw, expected_observation_pin={"bytes": len(raw), "sha256": "0" * 64})

    def test_partial_or_tie_allmasked_refuses_before_model_forward(self):
        self.fixture.lines.change_receipt("left", lambda receipt: receipt["report"].update(raw_score=20))
        self.fixture.refresh()
        model = FrozenNumericFixture(on_forward=lambda: self.fail("all-masked forward executed")).eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "all-masked"):
            self.fixture.prepare(model)

    def test_training_requires_grad_and_pending_grad_refused_before_forward(self):
        for mutate in (lambda model: model.train(), lambda model: model.fixed.requires_grad_(True),
                       lambda model: setattr(model.fixed, "grad", torch.ones_like(model.fixed))):
            model = FrozenNumericFixture(on_forward=lambda: self.fail("unfrozen forward executed")).eval().requires_grad_(False)
            observation = self.fixture.parameter_observation(model)
            mutate(model)
            with self.assertRaisesRegex(ValueError, "frozen eval/no-grad/no-autocast"):
                self.fixture.prepare(model, observation=observation)

    def test_finite_cpu_fp32_parameter_state_is_required_before_forward(self):
        for nonfinite in (False, True):
            model = FrozenNumericFixture(on_forward=lambda: self.fail("invalid FP32 forward executed")).eval().requires_grad_(False)
            observation = self.fixture.parameter_observation(model)
            if nonfinite:
                with torch.no_grad():
                    model.fixed.fill_(float("nan"))
            else:
                model.half()
            with self.assertRaisesRegex(ValueError, "finite CPU FP32"):
                self.fixture.prepare(model, observation=observation)

    def test_eval_requires_grad_and_grad_mutation_after_forward_are_rechecked(self):
        changes = (lambda model: model.train(), lambda model: model.fixed.requires_grad_(True),
                   lambda model: setattr(model.fixed, "grad", torch.ones_like(model.fixed)),
                   lambda model: model.experts["proposer"].train())
        for mutate in changes:
            model = FrozenNumericFixture().eval().requires_grad_(False)
            model.on_forward = lambda model=model, mutate=mutate: mutate(model)
            before = _parameter_digest(model)
            with self.assertRaisesRegex(ValueError, "frozen eval/no-grad/no-autocast"):
                self.fixture.prepare(model)
            self.assertEqual(_parameter_digest(model), before)

    def test_frozen_parameter_configuration_and_parent_mutation_after_forward_fail(self):
        def change_parameter(model):
            model.fixed.add_(1)
        mutations = (change_parameter, lambda model: setattr(model, "config", ModelConfig(iterations=3)),
                     lambda model: self.fixture.parents.records[self.fixture.index]["input"]["snapshot"].update(input_revision=4))
        for mutate in mutations:
            model = FrozenNumericFixture().eval().requires_grad_(False)
            model.on_forward = lambda model=model, mutate=mutate: mutate(model)
            with self.assertRaises(ValueError):
                self.fixture.prepare(model)

    def test_cpu_and_cuda_autocast_before_or_after_forward_are_refused(self):
        for device in ("cpu", "cuda"):
            for after in (False, True):
                with self.subTest(device=device, after=after):
                    active = {"value": not after}
                    model = FrozenNumericFixture(on_forward=lambda: active.update(value=True)).eval().requires_grad_(False)
                    # Probe APIs only; this fixture starts no autocast/GPU work.
                    with patch.object(torch, "is_autocast_enabled", side_effect=lambda name: name == device and active["value"]):
                        with self.assertRaisesRegex(ValueError, "frozen eval/no-grad/no-autocast"):
                            self.fixture.prepare(model)

    def test_original_absolute_deadline_before_and_after_forward_is_fail_closed(self):
        model = FrozenNumericFixture(on_forward=lambda: self.fail("expired forward executed")).eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "absolute preparation deadline"):
            self.fixture.prepare(model, deadline=0.5)
        clock = {"value": 1.0}
        model = FrozenNumericFixture(on_forward=lambda: clock.update(value=51.0)).eval().requires_grad_(False)
        observation = self.fixture.parameter_observation(model)
        with patch("rz_pals_model.whole_line_projection.time.monotonic", side_effect=lambda: clock["value"]):
            with self.assertRaisesRegex(ValueError, "absolute preparation deadline"):
                projection.frozen_repair_projection_preparation(model, self.fixture.admit(), parameter_observation_bytes=observation,
                    expected_observation_pin=ordinal.byte_pin(observation), deadline=50.0)


if __name__ == "__main__":
    unittest.main()
