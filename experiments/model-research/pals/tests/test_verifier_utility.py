"""Synthetic caller/CLI byte contracts and semantic no-step numeric wiring.

All registrations, launch observations and Rust receipts are synthetic here.
The real strict parent/semantic factories run when root executes these tests;
this establishes conditional admission, never an actual CPU child/Rules proof,
global novelty, learned utility, gameplay or strength evidence.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rz_pals_model import semantic_verifier as semantic
from rz_pals_model import verifier_producer as legacy
from rz_pals_model import verifier_utility as utility
from rz_pals_model.config import TASKS
from rz_pals_model.model import initialize
from rz_pals_model.preparation_check import _parameter_digest
from rz_pals_model.training import label_digest
from test_semantic_verifier import SemanticFixture
from test_training import sha


class UtilityFixture:
    """Actual admission factories, explicitly synthetic independent evidence."""
    def __init__(self, root, *, known=True):
        self.semantic = SemanticFixture(root)
        self.parents = self.semantic.parents
        row = self.parents.records[self.semantic.index]
        self.snapshot = row["input"]["snapshot"]
        profile = legacy.cpu_profile(16, 4, 4)
        self.semantic.common.update(cpu_profile_sha256=legacy.profile_sha256(profile), known_completed_depth=2 if known else 0)
        self.semantic.reseal()
        self.semantic.checked = self.semantic.admit()
        common = self.semantic.checked.common_query()
        names = ("receipt.json", "producer-roster.json", "producer-envelope.json", "producer-prepared.jsonl",
                 "inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl")
        self.parent_artifacts = {name: (Path(root) / name).read_bytes() for name in names}
        admission = self.parents.frozen_admission
        parent = {"receipt_artifact": admission["receipt"], "raw_dataset_sha256": admission["raw_dataset_sha256"],
                  "split_sha256": admission["split_sha256"], "current_view_sha256": self.parents.current_view.sha256,
                  "producer_roster_sha256": json.loads(self.parent_artifacts["producer-roster.json"])["sha256"],
                  "producer_envelope_sha256": json.loads(self.parent_artifacts["producer-envelope.json"])["sha256"]}
        current = {"input_sha256": row["input"]["sha256"], "label_sha256": label_digest(row),
                   **{name: self.snapshot[name] for name in ("game_id", "role", "rules_state_sha256", "rules_history_sha256",
                       "encoding_sha256", "source", "frozen_epoch", "input_revision", "legal_moves")},
                   "side_to_move": "white" if self.snapshot["white_to_move"] else "black"}
        identity = {"semantics": legacy.CPU_VALUE, "weights_sha256": None, "training": {"kind": "bootstrap"}}
        search = legacy.CPU_CONDITIONS + f";profile={legacy.PROFILE};max_depth=4;q_plies=4;tt_entries=16"
        self.raws = {"binary": b"synthetic independently registered utility executable bytes; no child runs"}
        self.source = {"schema": utility.SOURCE_SCHEMA, "profile": profile, "profile_sha256": legacy.profile_sha256(profile),
                       "search_version": legacy.CPU_SEARCH, "search_conditions": search, "value_identity": identity,
                       "search_implementation_sha256": sha("synthetic source build pin"), "value_semantics_sha256": sha("synthetic semantics source pin")}
        self.capabilities = {"schema": legacy.CPU_SCHEMA, "cpu_binary_sha256": utility.byte_pin(self.raws["binary"])["sha256"],
                             "training_private_only": True, "product_verifier_enabled": False,
                             "actual_training_executed": False, "backward_executed": False, "optimizer_created": False,
                             "external_teacher_used": False, "gpu_used": False, "cpu_search": legacy.CPU_SEARCH,
                             "value_identity": identity, "profile": legacy.PROFILE, "max_request_bytes": 512 * 1024,
                             "max_response_bytes": 1024 * 1024, "max_wall_time_ms": 300000, "max_checks": 2,
                             "max_nodes_per_check": (1 << 32) - 1, "max_depth": 64, "max_prefix_plies": 64,
                             "max_root_moves": 256, "selective_reductions": False,
                             "resume_kind": "completed_iteration_same_invocation_only",
                             "tasks": {"defend_response": "explicit_legal_prefix_and_restricted_response",
                                       "attack_repair": "explicit_legal_prefix", "widen_responses": "strict_legal_root_subset_to_unrestricted",
                                       "lower_selectivity": "unavailable_reductions_already_disabled",
                                       "resume_task": "conditional_owned_completed_iteration_token",
                                       "cross_profile_recheck": "fresh_own_independent_profile", "defer": "no_cpu_check"}}
        self.raws.update(source=utility.canonical(self.source), capabilities=utility.canonical(self.capabilities))
        self.registration = {"schema": utility.REGISTRATION_SCHEMA,
                             **{name + "_artifact": utility.byte_pin(self.raws[name]) for name in ("source", "binary", "capabilities")},
                             "profile": profile, "profile_sha256": self.source["profile_sha256"],
                             "recheck_profile_sha256": legacy.profile_sha256(legacy.cpu_profile(16, 4, 4, legacy.RECHECK_PROFILE)),
                             "platform": "windows", "binary_pin_scope": "current_exe_path_hash",
                             **{name: self.source[name] for name in ("search_implementation_sha256", "value_semantics_sha256")}}
        policy = {"max_wall_time_ms": 10000, "max_nodes_per_check": 100000, "max_checks": 2, "search_deadline_reserve_ms": 1000}
        self.conditions = {"schema": "rz-pals-private-cpu-conditions/1",
                           **{name: self.snapshot[name] for name in ("rules_state_sha256", "rules_history_sha256", "board_fen", "white_to_move", "legal_moves")},
                           "profile_sha256": self.source["profile_sha256"], "value_identity": identity, "search_version": legacy.CPU_SEARCH,
                           "search_conditions": search, "quiescence_ply": 4, "resource_policy": policy,
                           "root_moves": None, "root_order": self.snapshot["legal_moves"], "root_selection": "unrestricted",
                           "root_order_sha256": legacy._hash("rz-pals-private-cpu-root-order/1", self.snapshot["legal_moves"])}
        self.criterion = {"schema": utility.CRITERION_SCHEMA, "recipe_id": "고정-의무-coverage-fixture", "scope": utility.SCOPE,
                          "question_id": utility.QUESTION, "parent": parent, "current": current,
                          "semantic_input_sha256": common["derived_input_sha256"], "semantic_context_sha256": common["semantic_context_sha256"],
                          "branch_sha256": legacy._hash("rz-pals-private-cpu-branch/1", {"parent_input_sha256": current["input_sha256"], "prefix": [], "root_moves": []}),
                          "profile": profile, "obligation": {"conditions": self.conditions, "baseline_depth": 2, "required_horizon": 4,
                                                             "score_scope": "completed_iteration", "completion": "depth_limit"},
                          "budget_bucket": 1, "eligible_tasks": list(utility.ACTIONS), "max_output_bytes": 65536}
        self.requests = {action: self.request(action) for action in utility.ACTIONS}
        self.responses = {action: self.response(self.requests[action]) for action in utility.ACTIONS}
        self.known_requests = {"prior-resume": self.request("resume_task")} if known else {}
        self.known_responses = {"prior-resume": self.response(self.known_requests["prior-resume"], after_depth=2, completion="node_limit")} if known else {}
        self.launch_changes = {}
        self.knowledge = {"schema": utility.KNOWLEDGE_SCHEMA, "scope": utility.SCOPE, "current_input_sha256": current["input_sha256"],
                          "current_view_sha256": self.parents.current_view.sha256, "branch_sha256": self.criterion["branch_sha256"],
                          "execution_ids": list(self.known_requests)}
        self.plan = {"schema": utility.PLAN_SCHEMA, "scope": utility.SCOPE, "pair_id": "synthetic-utility-pair",
                     "parent": parent, "current_input_sha256": current["input_sha256"],
                     "semantic_input_sha256": common["derived_input_sha256"], "semantic_context_sha256": common["semantic_context_sha256"],
                     "common_query_sha256": common["common_query_sha256"], "requests": {}}
        self.reseal()

    def request(self, task):
        value = {"schema": legacy.CPU_SCHEMA, "task": task, "parent_input_sha256": self.criterion["current"]["input_sha256"],
                 "position_command": self.snapshot["position_command"], "expected_board_fen": self.snapshot["board_fen"],
                 **{name: self.snapshot[name] for name in ("rules_state_sha256", "rules_history_sha256")},
                 "cpu_binary_sha256": utility.byte_pin(self.raws["binary"])["sha256"], "branch_sha256": self.criterion["branch_sha256"],
                 "prefix": [], "root_moves": [], "baseline_depth": 2, "requested_depth": 4,
                 "max_nodes_per_check": 100000, "max_wall_time_ms": 10000, "max_output_bytes": 65536, "tt_entries": 16,
                 "quiescence_ply": 4, "cpu_profile_sha256": self.registration["profile_sha256"],
                 "recheck_profile_sha256": self.registration["recheck_profile_sha256"]}
        value["context_sha256"] = utility.digest([legacy.CPU_SCHEMA, value])
        return value

    def report(self, depth, requested, *, reused=0, completion="depth_limit"):
        return {"profile_sha256": self.registration["profile_sha256"], "conditions": copy.deepcopy(self.conditions),
                "conditions_sha256": utility.digest(self.conditions), "score_scope": "completed_iteration" if depth else "frontier_only",
                "completed_depth": depth, "requested_depth": requested, "nodes": 10, "quiescence_nodes": 2,
                "completion": completion, "root_restricted": False, "raw_score": -17, "pv": self.snapshot["legal_moves"][:1],
                "white_to_move": self.snapshot["white_to_move"], "elapsed_ms": 2, "reused_completed_depth": reused,
                "score_provenance": legacy.CPU_VALUE, "pv_rules_validated": True}

    def response(self, req, *, after_depth=4, completion="depth_limit"):
        defer = req["task"] == "defer"
        return {"schema": req["schema"], "task": req["task"],
                **{name: req[name] for name in ("context_sha256", "cpu_binary_sha256", "rules_state_sha256", "rules_history_sha256", "branch_sha256")},
                "board_fen": req["expected_board_fen"], "status": "deferred" if defer else "observed",
                "reason": "explicit_defer_no_cpu_check" if defer else None,
                "baseline": None if defer else self.report(2, 2),
                "after": None if defer else self.report(after_depth, 4, reused=2, completion=completion),
                "nodes": 0 if defer else 20, "elapsed_ms": 8,
                "resume_kind": None if defer else "completed_iteration", "product_verifier_enabled": False, "deadline_exceeded": False}

    def execution(self, execution_id, req, response, plan_sha):
        buffers = {"request": utility.canonical(req), "receipt": None if response is None else utility.canonical(response) + b"\n", "stderr": b""}
        launch = {"schema": utility.LAUNCH_SCHEMA, "execution_id": execution_id,
                  "registration_sha256": utility.byte_pin(self.raws["registration"])["sha256"],
                  "binary_sha256": utility.byte_pin(self.raws["binary"])["sha256"], "platform": "windows",
                  "binary_pin_scope": "current_exe_path_hash", "before_result_plan_sha256": plan_sha,
                  **{name: None if value is None else utility.byte_pin(value) for name, value in buffers.items()},
                  "assurance_scope": "independently_pinned_caller_observation", "anchor_durable_before_spawn": True,
                  "criterion_fixed_before_spawn": True, "spawned": True, "reaped": True, "pipes_finished": True, "exit_code": 0,
                  "elapsed_ms": 12, "timed_out": False, "canceled": False,
                  **{name + suffix: utility.byte_pin(self.raws[name]) for name in ("source", "binary") for suffix in ("_pre_artifact", "_post_artifact")},
                  "process_supervision_scope": "windows_direct_child_only_no_job_object", "owned_group_absent": None}
        launch.update(self.launch_changes.get(execution_id, {}))
        buffers["launch_observation"] = utility.canonical(launch)
        return buffers

    def reseal(self):
        for name in ("source", "capabilities", "registration", "criterion", "knowledge"):
            self.raws[name] = utility.canonical(getattr(self, name))
        for action, req in self.requests.items():
            req["context_sha256"] = utility.digest([legacy.CPU_SCHEMA, {key: value for key, value in req.items() if key != "context_sha256"}])
            if self.responses[action] is not None:
                self.responses[action]["context_sha256"] = req["context_sha256"]
        self.plan.update(**{name + "_artifact": utility.byte_pin(self.raws[name]) for name in ("criterion", "knowledge", "registration")})
        self.plan["requests"] = {action: {"execution_id": "current-" + action, "request_artifact": utility.byte_pin(utility.canonical(req))}
                                 for action, req in self.requests.items()}
        self.raws["plan"] = utility.canonical(self.plan)
        self.executions = {action: self.execution("current-" + action, req, self.responses[action], utility.byte_pin(self.raws["plan"])["sha256"])
                           for action, req in self.requests.items()}
        self.known_executions = {key: self.execution(key, req, self.known_responses[key], sha("independently pinned synthetic prior plan"))
                                for key, req in self.known_requests.items()}
        pins = lambda group: {key: {name: None if raw is None else utility.byte_pin(raw) for name, raw in buffers.items()} for key, buffers in group.items()}
        self.pins = {**{name: utility.byte_pin(raw) for name, raw in self.raws.items()},
                     "executions": pins(self.executions), "known_executions": pins(self.known_executions)}

    def admit(self, **overrides):
        arguments = {"parents": self.parents, "parent_artifacts": self.parent_artifacts,
                     **{name + "_bytes": raw for name, raw in self.raws.items()},
                     "executions": self.executions, "known_executions": self.known_executions,
                     "independent_pins": self.pins, "checked_common_query": self.semantic.checked}
        arguments.update(overrides)
        return utility.admit_fixed_obligation_utility(**arguments)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = UtilityFixture(self.temp.name)

    def test_positive_requires_actual_factories_and_scope_is_snapshot_relative(self):
        checked = self.fixture.admit()
        checked.verify()
        self.assertTrue(checked.target.mask)
        self.assertEqual((checked.target.resume_coverage, checked.target.defer_coverage, checked.target.sign), (1, 0, 1))
        self.assertIs(checked.checked_semantic, self.fixture.semantic.checked)
        self.assertEqual(checked.admission["scope"], utility.SCOPE)
        self.assertEqual(checked.admission["known_completed_depth"], 2)
        for name in ("known_snapshot_is_global_knowledge", "wdl_inferred", "rules_terminal_proof_inferred", "legacy_rank_modified",
                     "product_verifier_enabled", "actual_training_executed"):
            self.assertFalse(checked.admission[name])
        batch = checked.collate()
        self.assertEqual(batch.task_slots, (TASKS.index("resume_task"), TASKS.index("defer")))
        self.assertTrue(torch.equal(batch.base.inputs.query[0], torch.tensor(self.fixture.semantic.checked.encoded_snapshot().query, dtype=torch.float32)))
        self.assertFalse(bool(batch.base.task_mask.any()))

    def test_missing_or_injected_semantic_capability_cannot_open_positive(self):
        checked = self.fixture.admit(checked_common_query=None)
        self.assertFalse(checked.target.mask)
        self.assertEqual(checked.target.reason, "requires_checked_semantic_input")
        with self.assertRaises(ValueError):
            checked.collate()
        with self.assertRaises(ValueError):
            self.fixture.admit(checked_common_query=object())
        with self.assertRaises(ValueError):
            utility.CheckedVerifierUtilityPairs(parents=None, parent_artifacts={}, raw={}, executions={}, known={}, semantic=None,
                                                 target=None, admission={})

    def test_already_covered_partial_prior_iteration_refuses_novelty(self):
        self.fixture.known_responses["prior-resume"]["after"]["completed_depth"] = 4
        # The prior request later hit its node limit, but H4 is already known.
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "already covered"):
            self.fixture.admit()

    def test_prior_depth_must_equal_semantic_before_result_declaration(self):
        self.fixture.known_responses["prior-resume"]["after"]["completed_depth"] = 3
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "knowledge mismatch"):
            self.fixture.admit()

    def test_empty_prior_snapshot_does_not_claim_global_knowledge(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = UtilityFixture(root, known=False)
            checked = fixture.admit()
            self.assertTrue(checked.target.mask)
            self.assertTrue(checked.admission["known_snapshot_empty"])
            self.assertEqual(checked.admission["known_completed_depth"], 0)
            self.assertFalse(checked.admission["known_snapshot_is_global_knowledge"])

    def test_mixed_profile_configuration_root_order_and_request_are_refused(self):
        original = copy.deepcopy(self.fixture.requests["defer"])
        for key, value in (("cpu_profile_sha256", sha("mixed profile")), ("requested_depth", 3), ("max_nodes_per_check", 99999),
                           ("max_output_bytes", 32768), ("root_moves", self.fixture.snapshot["legal_moves"][:1]), ("baseline_depth", True)):
            with self.subTest(key=key):
                self.fixture.requests["defer"] = {**original, key: value}
                self.fixture.reseal()
                with self.assertRaisesRegex(ValueError, "request differs"):
                    self.fixture.admit()
        self.fixture.requests["defer"] = original
        after = self.fixture.responses["resume_task"]["after"]
        after["conditions"]["legal_moves"] = list(reversed(after["conditions"]["legal_moves"]))
        after["conditions_sha256"] = utility.digest(after["conditions"])
        self.fixture.reseal()
        with self.assertRaises(ValueError):
            self.fixture.admit()

    def test_raw_pins_current_leaf_and_registered_source_are_independent(self):
        pins = copy.deepcopy(self.fixture.pins)
        pins["binary"]["sha256"] = sha("not the actual binary bytes")
        with self.assertRaises(ValueError):
            self.fixture.admit(independent_pins=pins)
        self.fixture.criterion["current"]["label_sha256"] = sha("superseded label")
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "current parent identity"):
            self.fixture.admit()

    def test_cli_self_image_pin_and_owned_resume_are_required(self):
        original = copy.deepcopy(self.fixture.capabilities)
        for variant in ("static_only", "wrong_image"):
            with self.subTest(variant=variant):
                self.fixture.capabilities = copy.deepcopy(original)
                if variant == "static_only":
                    del self.fixture.capabilities["cpu_binary_sha256"]
                else:
                    self.fixture.capabilities["cpu_binary_sha256"] = sha("different executable")
                self.fixture.registration["capabilities_artifact"] = utility.byte_pin(utility.canonical(self.fixture.capabilities))
                self.fixture.reseal()
                with self.assertRaises(ValueError):
                    self.fixture.admit()
        self.fixture.capabilities = original
        self.fixture.registration["capabilities_artifact"] = utility.byte_pin(utility.canonical(original))
        self.fixture.responses["resume_task"]["after"]["reused_completed_depth"] = 0
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "reuse"):
            self.fixture.admit()

    def test_launch_total_elapsed_and_no_check_defer_accounting(self):
        self.fixture.responses["resume_task"]["elapsed_ms"] = 9999
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "elapsed exceeds"):
            self.fixture.admit()
        self.fixture.responses["resume_task"]["elapsed_ms"] = 8
        for report in (self.fixture.responses["resume_task"]["baseline"], self.fixture.responses["resume_task"]["after"]):
            report["elapsed_ms"] = 6
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "elapsed exceeds"):
            self.fixture.admit()
        for report in (self.fixture.responses["resume_task"]["baseline"], self.fixture.responses["resume_task"]["after"]):
            report["elapsed_ms"] = 2
        self.fixture.responses["defer"]["nodes"] = 1
        self.fixture.reseal()
        with self.assertRaises(ValueError):
            self.fixture.admit()

    def test_partial_cancel_missing_and_failed_launch_preserve_raw_and_mask(self):
        for variant in ("partial", "canceled", "terminal", "missing", "nonzero", "unstable_source", "not_durable"):
            with self.subTest(variant=variant):
                self.fixture.responses["resume_task"] = self.fixture.response(self.fixture.requests["resume_task"])
                self.fixture.launch_changes = {}
                if variant in ("partial", "canceled"):
                    self.fixture.responses["resume_task"]["after"].update(completed_depth=2, completion="node_limit" if variant == "partial" else "canceled")
                elif variant == "terminal":
                    response = self.fixture.responses["resume_task"]
                    response.update(status="unavailable", reason="no_completed_iteration_resume_token", after=None, nodes=10, resume_kind=None)
                    response["baseline"].update(score_scope="rules_terminal", completed_depth=0, completion="rules_terminal",
                                                 pv=[], score_provenance="rz-position/rules-terminal")
                elif variant == "missing":
                    self.fixture.responses["resume_task"] = None
                else:
                    key, value = {"nonzero": ("exit_code", 4), "unstable_source": ("source_post_artifact", {"bytes": 1, "sha256": sha("other source")}),
                                  "not_durable": ("anchor_durable_before_spawn", False)}[variant]
                    self.fixture.launch_changes["current-resume_task"] = {key: value}
                self.fixture.reseal()
                checked = self.fixture.admit()
                self.assertFalse(checked.target.mask)
                self.assertIsNone(checked.target.resume_coverage)
                self.assertEqual(checked.raw_executions()["current"], self.fixture.executions)

    def test_prior_missing_or_duplicate_unknown_cannot_make_positive(self):
        self.fixture.known_responses["prior-resume"] = None
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "prior observation is unknown"):
            self.fixture.admit()
        self.fixture.knowledge["execution_ids"] *= 2
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "duplicates"):
            self.fixture.admit()

    def test_mutation_guards_raw_admission_and_parent(self):
        checked = self.fixture.admit()
        with self.assertRaises(AttributeError):
            checked.target = replace(checked.target, sign=0, mask=False)
        returned = checked.raw_executions()
        returned["current"]["defer"]["receipt"] = b"changed copy"
        checked.verify()
        checked.admission["known_completed_depth"] = 63
        with self.assertRaisesRegex(ValueError, "changed"):
            checked.verify()
        checked = self.fixture.admit()
        checked._executions["defer"]["receipt"] += b" "
        with self.assertRaisesRegex(ValueError, "changed"):
            checked.verify()
        checked = self.fixture.admit()
        self.fixture.parents.records[self.fixture.semantic.index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaises(ValueError):
            checked.verify()

    def test_boolean_u64_duplicate_json_and_unregistered_capabilities_rejected(self):
        raw = self.fixture.raws["criterion"].replace(b'"schema":', b'"schema":"duplicate","schema":', 1)
        pins = copy.deepcopy(self.fixture.pins)
        pins["criterion"] = utility.byte_pin(raw)
        with self.assertRaises(ValueError):
            self.fixture.admit(criterion_bytes=raw, independent_pins=pins)
        self.fixture.capabilities["product_verifier_enabled"] = True
        self.fixture.registration["capabilities_artifact"] = utility.byte_pin(utility.canonical(self.fixture.capabilities))
        self.fixture.reseal()
        with self.assertRaisesRegex(ValueError, "capability/private"):
            self.fixture.admit()


class FrozenNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = initialize(37)
        cls.model.eval()
        for parameter in cls.model.parameters():
            parameter.requires_grad_(False)
        cls.parameter_sha = _parameter_digest(cls.model)
        cls.checkpoint_sha = sha("synthetic caller-reloaded checkpoint")
        cls.observation = semantic.canonical({"schema": semantic.PARAMETER_SCHEMA, "checkpoint_sha256": cls.checkpoint_sha,
                                              "parameter_sha256": cls.parameter_sha,
                                              "assurance_scope": "independently_pinned_caller_parameter_observation"})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = UtilityFixture(self.temp.name)
        self.options = {"parameter_observation_bytes": self.observation,
                        "expected_observation_sha256": utility.byte_pin(self.observation)["sha256"],
                        "expected_parameter_sha256": self.parameter_sha, "checkpoint_sha256": self.checkpoint_sha,
                        "max_wall_time_ms": 120000}

    def test_softplus_slots_mask_and_no_grad(self):
        batch = self.fixture.admit().collate()
        logits = torch.zeros((1, 7), dtype=torch.float32)
        logits[0, batch.task_slots[0]], logits[0, batch.task_slots[1]] = 2., -1.
        self.assertAlmostEqual(float(utility.task_pairwise_softplus_loss(logits, batch)), float(torch.nn.functional.softplus(torch.tensor(-3.))), places=7)
        self.assertEqual(float(utility.task_pairwise_softplus_loss(logits, replace(batch, sign=0, mask=False))), 0.)
        with self.assertRaises(ValueError):
            utility.task_pairwise_softplus_loss(logits.requires_grad_(True), batch)

    def test_frozen_semantic_logits_drive_finite_nonzero_loss_without_step(self):
        checked = self.fixture.admit()
        actual_forward = semantic.FrozenSemanticVerifier.forward
        seen = []
        def traced(wrapper, inputs, **kwargs):
            self.assertEqual(inputs, [checked.checked_semantic])
            result = actual_forward(wrapper, inputs, **kwargs)
            seen.append(result)
            return result
        with patch.object(semantic.FrozenSemanticVerifier, "forward", traced), \
                patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer forbidden")), \
                patch.object(self.model, "role_graph", side_effect=AssertionError("legacy public-only path forbidden")):
            report = utility.frozen_verifier_utility_preparation(self.model, checked, **self.options)
        self.assertEqual(len(seen), 1)
        expected = float(utility.task_pairwise_softplus_loss(seen[0][0], checked.collate()))
        self.assertEqual(report["task_pairwise_loss"], expected)
        self.assertGreater(expected, 0.)
        self.assertEqual(report["semantic_forward"]["input_sha256"], (checked.checked_semantic.sha256,))
        self.assertEqual(report["parameter_sha256_before"], report["parameter_sha256_after"])
        self.assertEqual(_parameter_digest(self.model), self.parameter_sha)
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in self.model.parameters()))
        for name in ("actual_training_executed", "backward_executed", "optimizer_created", "gpu_executed", "product_verifier_enabled"):
            self.assertFalse(report[name])

    def test_all_masked_parameter_observation_and_preallocation_limit_refused(self):
        masked = self.fixture.admit(checked_common_query=None)
        with patch.object(semantic.FrozenSemanticVerifier, "forward", side_effect=AssertionError("masked model executed")):
            with self.assertRaisesRegex(ValueError, "all-masked"):
                utility.frozen_verifier_utility_preparation(self.model, masked, **self.options)
        options = {**self.options, "checkpoint_sha256": sha("wrong independent checkpoint")}
        with self.assertRaisesRegex(ValueError, "checkpoint/parameter"):
            utility.frozen_verifier_utility_preparation(self.model, self.fixture.admit(), **options)
        with self.assertRaisesRegex(ValueError, "reservation denied"):
            utility.frozen_verifier_utility_preparation(self.model, self.fixture.admit(), **self.options, byte_limit=1)


if __name__ == "__main__":
    unittest.main()
