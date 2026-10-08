"""Coverage feedback byte/chronology and optional CPU numeric fixtures.

All child/loaded-image/source observations below are explicitly SYNTHETIC.
Real strict parent and semantic factories are reused. The end-to-end fixture
can execute the existing frozen CPU V twice when root runs it, but no Rust
child, Rules replay, trained utility, strategic feedback or product V is
proved. No optimizer/backward/GPU/checkpoint export is used.
"""
import copy
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from rz_pals_model import semantic_verifier as semantic
from rz_pals_model import verifier_feedback as feedback
from rz_pals_model import verifier_producer as legacy
from rz_pals_model.config import ModelConfig, TASKS, matmul_flops
from rz_pals_model.model import initialize
from rz_pals_model.preparation_check import _parameter_digest
from test_verifier_utility import UtilityFixture


def raw(value):
    return semantic.canonical(value)


def synthetic_process(binary_pin, source_pin, *, elapsed=12):
    """Pinned synthetic caller observations, never actual launch evidence."""
    def facts(pin, scope="caller_registered_path_bytes"):
        return {"status": "checked", "artifact": copy.deepcopy(pin), "device": "1", "inode": "2",
                "mtime_ns": "3", "ctime_ns": "4", "scope": scope}
    return {"schema": feedback.PROCESS_SCHEMA, "pid": 123, "spawned": True, "reaped": True, "pipes_finished": True,
            "owned_group_absent": True, "exit_code": 0, "failure": None, "cleanup_error": None,
            "elapsed_ms": elapsed, "original_deadline_met": True, "loaded_image_before_stdin": True,
            "loaded_image_sha256": binary_pin["sha256"], "process_supervision_scope": "posix_owned_process_group",
            "loaded_executable": facts(binary_pin, "caller_child_loaded_inode"),
            "binary_before": facts(binary_pin), "binary_after": facts(binary_pin),
            "source_before": facts(source_pin), "source_after": facts(source_pin),
            "binary_path_stable": True, "source_path_stable": True}


class FeedbackFixture:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "parents").mkdir()
        self.base = UtilityFixture(self.root / "parents")
        self.parents, self.index = self.base.parents, self.base.semantic.index
        self.identity = self.parents.records[self.index]["input"]["sha256"]
        self.snapshot = self.parents.records[self.index]["input"]["snapshot"]
        self.stage = {"allowed_tasks": ["resume_task", "cross_profile_recheck", "defer"], "baseline_depth": 2,
            "requested_depth": 4, "max_nodes_per_check": 100000, "max_wall_time_ms": 10000, "tt_entries": 16, "quiescence_ply": 4}
        self.limits = {"max_games": 1, "max_steps": 2, "max_nodes": 400000, "max_wall_time_ms": 60000,
            "max_output_bytes": feedback.MAX_BYTES, "max_forward_flops": 10**12,
            "max_semantic_bytes": semantic.MAX_SEMANTIC_BYTES, "cleanup_reserve_ms": 1000,
            "max_child_output_bytes": 65536, "max_input_bytes": 128 << 20}
        self.plan = {"schema": feedback.PLAN_SCHEMA, "scope": feedback.SCOPE, "recipe_id": "사전-두-round-coverage",
            "parent": feedback._parent_pins(self.parents), "parent_inputs": [self.identity],
            "limits": self.limits, "stages": [copy.deepcopy(self.stage), copy.deepcopy(self.stage)]}
        self.binary_pin = semantic.byte_pin(self.base.raws["binary"])
        self.base.registration.update(platform="linux", binary_pin_scope="linux_loaded_executable_inode")
        self.base.raws["registration"] = raw(self.base.registration)
        sem = self.base.semantic
        sem.registration.update(platform="linux", accepted_binary_pin_scope="linux_loaded_executable_inode",
                                binary_sha256=self.binary_pin["sha256"])
        sem.receipt["binary_pin_scope"] = "linux_loaded_executable_inode"
        sem.reseal()
        self.assets = {name: self.base.raws[name] for name in ("registration", "source", "capabilities", "binary")}
        self.assets.update(semantic_registration=sem.raws["registration"], semantic_source=sem.raws["source"])
        asset_root = self.root / "registered"
        asset_root.mkdir()
        self.paths = {}
        for name, value in self.assets.items():
            path = asset_root / (name + ".bin")
            path.write_bytes(value)
            self.paths[name] = path
        self.pins = {name: semantic.byte_pin(value) for name, value in self.assets.items()}

    def request(self, task="resume_task", stage=None):
        result = feedback._request(self.snapshot, self.identity, stage or self.stage, task, self.binary_pin["sha256"])
        result["max_output_bytes"] = self.limits["max_child_output_bytes"]
        result["context_sha256"] = legacy._hash(legacy.CPU_SCHEMA, result)
        return result

    def response(self, request, *, partial=False):
        result = self.base.response(request)
        if request["task"] != "defer":
            for after, report in ((False, result["baseline"]), (True, result["after"])):
                conditions = report["conditions"]
                conditions["resource_policy"] = {"max_wall_time_ms": request["max_wall_time_ms"],
                    "max_nodes_per_check": request["max_nodes_per_check"], "max_checks": 2,
                    "search_deadline_reserve_ms": min(request["max_wall_time_ms"] // 10, 1000)}
                independent = after and request["task"] == "cross_profile_recheck"
                profile_name = legacy.RECHECK_PROFILE if independent else legacy.PROFILE
                profile_pin = request["recheck_profile_sha256"] if independent else request["cpu_profile_sha256"]
                report["profile_sha256"] = conditions["profile_sha256"] = profile_pin
                conditions["search_conditions"] = legacy.CPU_CONDITIONS + f";profile={profile_name};max_depth=4;q_plies=4;tt_entries=16"
                report["conditions_sha256"] = semantic.byte_pin(raw(conditions))["sha256"]
                report["requested_depth"] = request["requested_depth"] if after else request["baseline_depth"]
                report["reused_completed_depth"] = 2 if after and request["task"] == "resume_task" else 0
            result["resume_kind"] = "completed_iteration" if request["task"] == "resume_task" else None
            if partial:
                result["after"].update(completion="node_limit", completed_depth=3)
        return result

    def observation(self, task="resume_task", *, stage=None, partial=False):
        request = self.request(task, stage)
        response = self.response(request, partial=partial)
        process = synthetic_process(self.binary_pin, self.pins["source"])
        request_raw, receipt_raw, process_raw = raw(request) + b"\n", raw(response) + b"\n", raw(process)
        gain = feedback._cpu_observation(request_raw, receipt_raw, process_raw, self.snapshot["legal_moves"])
        return request, response, process, request_raw, receipt_raw, process_raw, gain

    def append(self, ledger, **kwargs):
        _, _, _, request_raw, receipt_raw, process_raw, gain = self.observation(**kwargs)
        return ledger.append(input_sha256=self.identity, request_raw=request_raw, receipt_raw=receipt_raw,
            process_raw=process_raw, legal_order=self.snapshot["legal_moves"], gain=gain)

    def query(self, ledger, number=0, remaining=2, stage=None):
        return feedback.next_common_query(self.parents, self.index, stage or self.stage, ledger,
            round_index=number, maximum_rounds=2, remaining_global_steps=remaining, binary_sha256=self.binary_pin["sha256"])

    def capture(self, binary, arguments, request_raw, *, assets, budget):
        """Source fixture seam only; never invoke subprocess or claim a child."""
        request = legacy._parse(request_raw)
        if arguments == ["--prepare-semantic"]:
            stem = sorted(self.bank.glob("*-common_query.json"))[-1]
            self.base.semantic.common = legacy._parse(stem.read_bytes())
            self.base.semantic.reseal()
            self.test_requests.append(copy.deepcopy(self.base.semantic.common))
            if request != legacy._parse(self.base.semantic.raws["request"]):
                raise AssertionError("fixture semantic request must match producer exactly")
            result = self.base.semantic.raws["receipt"]
        else:
            result = raw(self.response(request)) + b"\n"
        return result, b"", raw(synthetic_process(assets["binary_pin"], assets["source_pin"]))


class CoverageFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = FeedbackFixture(self.temp.name)

    def test_completed_iteration_updates_next_checked_header_and_keeps_past_input(self):
        f, ledger = self.fixture, feedback.FeedbackLedger()
        first = f.query(ledger)
        original = copy.deepcopy(f.parents.records)
        sem = f.base.semantic
        sem.common = first
        sem.reseal()
        historical = sem.admit()
        old_sha = historical.sha256
        f.append(ledger)
        second = f.query(ledger, number=1, remaining=1)
        self.assertEqual((first["known_completed_depth"], second["known_completed_depth"]), (0, 4))
        self.assertNotIn("resume_task", second["allowed_tasks"])
        self.assertIn("cross_profile_recheck", second["allowed_tasks"])
        sem.common = second
        sem.reseal()
        updated = sem.admit()
        self.assertNotEqual(old_sha, updated.sha256)
        self.assertEqual(historical.sha256, old_sha)
        self.assertEqual(historical.common_query()["known_completed_depth"], 0)
        self.assertEqual(semantic.semantic_features(updated).known_completed_depth, 4)
        self.assertEqual(len(updated.encoded_snapshot().query), 16)
        self.assertEqual(f.parents.records, original)

    def test_bucket_accounts_for_actual_global_remaining_slots(self):
        f, ledger = self.fixture, feedback.FeedbackLedger()
        self.assertEqual(f.query(ledger)["budget_bucket"], 16)
        self.assertEqual(f.query(ledger, remaining=1)["budget_bucket"], 8)
        self.assertEqual(f.query(ledger, number=1, remaining=1)["budget_bucket"], 8)
        self.assertIsNone(f.query(ledger, remaining=0))
        with self.assertRaises(ValueError):
            f.query(object())

    def test_partial_result_preserves_raw_and_only_completed_baseline_coverage(self):
        f, ledger = self.fixture, feedback.FeedbackLedger()
        event = legacy._parse(f.append(ledger, partial=True))
        next_query = f.query(ledger, number=1, remaining=1)
        self.assertEqual(next_query["known_completed_depth"], 2)
        self.assertIn("resume_task", next_query["allowed_tasks"])
        response = legacy._parse(bytes.fromhex(event["receipt_hex"]))
        self.assertEqual(response["after"]["completion"], "node_limit")
        self.assertIsNone(event["preference_rank"])
        self.assertFalse(event["wdl_inferred"])

    def test_different_nodes_wall_legal_order_or_profile_do_not_share_coverage(self):
        f, ledger = self.fixture, feedback.FeedbackLedger()
        f.append(ledger)
        for key, value in (("max_nodes_per_check", 99999), ("max_wall_time_ms", 9999)):
            stage = dict(f.stage, **{key: value})
            self.assertEqual(f.query(ledger, number=1, remaining=1, stage=stage)["known_completed_depth"], 0)
        request = f.request()
        self.assertEqual(ledger.known_depth(request, request["recheck_profile_sha256"], f.snapshot["legal_moves"]), 0)
        self.assertEqual(ledger.known_depth(request, request["cpu_profile_sha256"], list(reversed(f.snapshot["legal_moves"]))), 0)

    def test_cross_profile_observation_stays_independent_without_strategy_reward(self):
        f, ledger = self.fixture, feedback.FeedbackLedger()
        f.append(ledger, task="cross_profile_recheck")
        req = f.request()
        self.assertEqual(ledger.known_depth(req, req["cpu_profile_sha256"], f.snapshot["legal_moves"]), 2)
        self.assertEqual(ledger.known_depth(req, req["recheck_profile_sha256"], f.snapshot["legal_moves"]), 4)
        query = f.query(ledger, number=1, remaining=1)
        self.assertIn("resume_task", query["allowed_tasks"])
        self.assertNotIn("cross_profile_recheck", query["allowed_tasks"])

    def test_defer_preserves_no_check_zero_coverage(self):
        f, ledger = self.fixture, feedback.FeedbackLedger()
        event = legacy._parse(f.append(ledger, task="defer"))
        self.assertEqual(f.query(ledger, number=1, remaining=1)["known_completed_depth"], 0)
        self.assertEqual(event["observed_information"]["status"], "deferred")
        self.assertIsNone(event["preference_rank"])
        self.assertFalse(event["cpu_stack_continued_across_children"])

    def test_canceled_partial_and_missing_receipt_do_not_invent_coverage_or_label(self):
        f = self.fixture
        request, response, _, request_raw, _, process_raw, _ = f.observation()
        response["after"].update(completion="canceled", completed_depth=2)
        receipt_raw = raw(response)
        gain = feedback._cpu_observation(request_raw, receipt_raw, process_raw, f.snapshot["legal_moves"])
        ledger = feedback.FeedbackLedger()
        ledger.append(input_sha256=f.identity, request_raw=request_raw, receipt_raw=receipt_raw,
            process_raw=process_raw, legal_order=f.snapshot["legal_moves"], gain=gain)
        self.assertEqual(f.query(ledger, number=1, remaining=1)["known_completed_depth"], 2)
        self.assertIsNone(legacy.future_label(f.snapshot, legacy.TaskContext(request["branch_sha256"], request["cpu_profile_sha256"], 8),
            "resume_task", request, response, ledger.sha256))
        with self.assertRaises(ValueError):
            feedback._cpu_observation(request_raw, b"", process_raw, f.snapshot["legal_moves"])

    def test_mutated_historical_parent_cannot_feed_a_new_query(self):
        f = self.fixture
        f.parents.records[f.index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaises(ValueError):
            f.query(feedback.FeedbackLedger())

    def test_raw_score_mate_band_is_retained_but_never_rank_or_wdl(self):
        f = self.fixture
        req, response, process, request_raw, _, process_raw, _ = f.observation()
        response["after"]["raw_score"] = -30000
        receipt_raw = raw(response)
        gain = feedback._cpu_observation(request_raw, receipt_raw, process_raw, f.snapshot["legal_moves"])
        ledger = feedback.FeedbackLedger()
        event = legacy._parse(ledger.append(input_sha256=f.identity, request_raw=request_raw, receipt_raw=receipt_raw,
            process_raw=process_raw, legal_order=f.snapshot["legal_moves"], gain=gain))
        self.assertFalse(gain["wdl_inferred"])
        self.assertFalse(event["strategic_truth_admitted"])
        self.assertIsNone(event["preference_rank"])

    def test_history_mutation_and_forged_chain_rejected(self):
        f, ledger = self.fixture, feedback.FeedbackLedger()
        f.append(ledger)
        event = legacy._parse(ledger._entries[0])
        event["ordinal"] = True
        ledger._entries = (raw(event),)
        ledger._seal = semantic.digest(feedback.EVIDENCE_SCHEMA, [event])
        with self.assertRaises(ValueError):
            ledger.verify()
        ledger = feedback.FeedbackLedger()
        f.append(ledger)
        ledger._entries = (ledger._entries[0] + b" ",)
        # Byte-identical history is required even when a JSON body is unchanged.
        with self.assertRaises(ValueError):
            ledger.verify()

    def test_completion_declaration_cannot_replace_process_identity_cleanup(self):
        f = self.fixture
        req, response, process, request_raw, receipt_raw, _, gain = f.observation()
        for change in ({"cleanup_error": "selector-close failed"}, {"reaped": False}, {"owned_group_absent": False},
                       {"loaded_image_before_stdin": False}, {"exit_code": True}, {"elapsed_ms": 7}):
            altered = dict(process, **change)
            with self.assertRaises(ValueError):
                feedback._cpu_observation(request_raw, receipt_raw, raw(altered), f.snapshot["legal_moves"])

    def test_changed_source_path_stamp_and_wrong_actual_root_order_rejected(self):
        f = self.fixture
        _, response, process, request_raw, receipt_raw, process_raw, _ = f.observation()
        changed = copy.deepcopy(process)
        changed["source_after"]["inode"] = "99"
        with self.assertRaises(ValueError):
            feedback._cpu_observation(request_raw, receipt_raw, raw(changed), f.snapshot["legal_moves"])
        with self.assertRaises(ValueError):
            feedback._cpu_observation(request_raw, receipt_raw, process_raw, list(reversed(f.snapshot["legal_moves"])))

    def test_prepared_cpu_request_context_and_boolean_budget_rejected(self):
        f = self.fixture
        req, _, _, request_raw, receipt_raw, process_raw, _ = f.observation()
        for change in ({"context_sha256": "1" * 64}, {"max_nodes_per_check": True}):
            with self.assertRaises(ValueError):
                feedback._cpu_observation(raw(dict(req, **change)), receipt_raw, process_raw, f.snapshot["legal_moves"])

    def test_plan_rejects_profile_drift_unknown_task_and_duplicate_parent(self):
        f = self.fixture
        for mutate in (lambda p: p["stages"][1].update(quiescence_ply=5),
                       lambda p: p["stages"][0].update(allowed_tasks=["attack_repair"]),
                       lambda p: p["parent_inputs"].append(f.identity),
                       lambda p: p["limits"].update(max_steps=True)):
            plan = copy.deepcopy(f.plan)
            mutate(plan)
            value = raw(plan)
            with self.assertRaises(ValueError):
                feedback._plan(value, f.parents, semantic.byte_pin(value))

    def test_quota_reservation_atomic_and_partial_output_keeps_charge(self):
        f = self.fixture
        with patch.object(feedback.time, "monotonic", return_value=100):
            budget = feedback.FeedbackBudget(f.limits, 160)
            before = dict(budget.used)
            self.assertIsNone(budget.reserve("game", nodes=400001, flops=1, output_bytes=1, semantic_bytes=1))
            self.assertEqual(budget.used, before)
            credit = budget.reserve("game", nodes=200000, flops=1, output_bytes=100, semantic_bytes=100)
            credit.take(30)
            credit.release()
            self.assertEqual(budget.used["output_bytes"], before["output_bytes"] + 30)
            self.assertEqual(budget.used["steps"], 1)

    def test_insufficient_original_task_wall_stops_without_silent_lower_scope(self):
        f = self.fixture
        with patch.object(feedback.time, "monotonic", return_value=100):
            budget = feedback.FeedbackBudget(f.limits, 120)
            self.assertTrue(budget.can_dispatch(10000))
            self.assertFalse(budget.can_dispatch(10000, calls=2))
        with patch.object(feedback.time, "monotonic", return_value=119):
            with self.assertRaises(TimeoutError):
                budget.check()

    def test_cancel_file_existence_only_and_secret_path_refused(self):
        f = self.fixture
        cancel = f.root / "cancel.request"
        cancel.write_text("This fixture content must not be read.")
        with patch.object(feedback.time, "monotonic", return_value=100), patch.object(Path, "read_bytes", side_effect=AssertionError("contents unread")):
            with self.assertRaises(InterruptedError):
                feedback.FeedbackBudget(f.limits, 160, cancel)
            with self.assertRaises(ValueError):
                feedback.FeedbackBudget(f.limits, 160, f.root / ".env")

    def test_aggregate_input_pin_cost_denied_before_source_read(self):
        f = self.fixture
        with patch.object(legacy, "_file", side_effect=AssertionError("must not allocate/read")):
            with self.assertRaises(ValueError):
                feedback._assets(f.paths, f.pins, f.stage, time.monotonic() + 1, 1)

    def test_semantic_matrix_flops_bound_includes_extra_private_tokens(self):
        config = ModelConfig()
        base = matmul_flops(config, "validator", 0, 2)["cold_forward_matmul_flops"]
        upper = feedback.forward_flops_upper(config, 0, 2)
        self.assertGreater(upper, base)
        self.assertEqual(type(upper), int)

    def test_two_cold_forwards_feedback_and_reload_without_training_admission(self):
        f = self.fixture
        model = initialize(29).eval()
        # Zero test task logits yield vocabulary-order ACTION selection. The
        # second input masks the already covered action, not a learned rank.
        with torch.no_grad():
            model.experts["validator"].task_head.weight.zero_()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        parameter_sha = _parameter_digest(model)
        observation = raw({"schema": semantic.PARAMETER_SCHEMA, "checkpoint_sha256": "2" * 64,
            "parameter_sha256": parameter_sha, "assurance_scope": "independently_pinned_caller_parameter_observation"})
        f.bank, f.test_requests = f.root / "feedback-bank", []
        plan_raw = raw(f.plan)
        deadline = time.monotonic() + 60
        with patch.object(feedback, "_capture", side_effect=f.capture), patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("no step")):
            result = feedback.run_verifier_feedback(parents=f.parents, model=model, plan_bytes=plan_raw,
                expected_plan_pin=semantic.byte_pin(plan_raw), registered_paths=f.paths, expected_registered_pins=f.pins,
                parameter_observation_bytes=observation, expected_parameter_observation_pin=semantic.byte_pin(observation),
                output=f.bank, deadline=deadline)
        self.assertEqual(result["counts"]["verifier_cold_forwards"], 2)
        self.assertEqual(result["counts"]["feedback_transitions"], 1)
        self.assertEqual([q["known_completed_depth"] for q in f.test_requests], [0, 4])
        self.assertFalse(result["private_warm_supported"])
        self.assertFalse(result["full_strategic_V_feedback_complete"])
        self.assertFalse(result["positive_training_target"])
        self.assertEqual(result["training_steps"], 0)
        self.assertEqual(_parameter_digest(model), parameter_sha)
        bank = feedback.load_verifier_feedback_bank(f.bank, parents=f.parents,
            expected_receipt_pin=semantic.byte_pin((f.bank / "receipt.json").read_bytes()), expected_registered_pins=f.pins,
            expected_parameter_observation_pin=semantic.byte_pin(observation), expected_producer_source_sha256=feedback._implementation_sha(),
            registered_paths=f.paths, deadline=deadline)
        summary = bank.summary()
        self.assertEqual(summary["counts"]["feedback_transitions"], 1)
        self.assertFalse(summary["ordinary_admission"])
        # Exact raw history remains authority; a bank mutation cannot be hidden
        # by changing the query declaration to a larger coverage number.
        raws = dict(bank._raws)
        name = next(name for name in raws if name.endswith("-feedback-evidence.json"))
        raws[name] += b" "
        bad = feedback.CheckedFeedbackHistory(bank._parent, tuple(sorted(raws.items())), bank._receipt_pin,
            bank._registered_pins, bank._parameter_pin, bank._source_sha256)
        with self.assertRaises(ValueError):
            bad.verify()


if __name__ == "__main__":
    unittest.main()
