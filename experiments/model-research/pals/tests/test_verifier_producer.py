"""Numeric/contract-only fixtures; no chess observations or optimizer updates."""
import copy
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from rz_pals_model import verifier_producer as producer
from rz_pals_model.config import TASKS
from rz_pals_model.training import (EncodedSnapshot, TaskContext, ValidatedDataset,
                                    masked_losses, seal_snapshot)

SHA = "a" * 64
MOVES = [7 | (6 << 6), 7 | (14 << 6)]
FEN = "k7/8/8/8/8/8/8/7K w - - 0 1"


def request(task="resume_task", roots=None):
    configuration = producer.cpu_profile(16, 2, 8)
    value = {"schema": producer.CPU_SCHEMA, "task": task, "parent_input_sha256": SHA,
             "position_command": "position fen " + FEN, "expected_board_fen": FEN,
             "rules_state_sha256": SHA, "rules_history_sha256": SHA,
             "cpu_binary_sha256": SHA, "branch_sha256": SHA, "context_sha256": SHA,
             "prefix": [], "root_moves": roots or [], "baseline_depth": 1, "requested_depth": 2,
             "max_nodes_per_check": 100, "max_wall_time_ms": 1000, "max_output_bytes": 32768,
             "tt_entries": 16, "quiescence_ply": 8,
             "cpu_profile_sha256": producer.profile_sha256(configuration),
             "recheck_profile_sha256": producer.profile_sha256(producer.cpu_profile(16, 2, 8, producer.RECHECK_PROFILE))}
    return value


def report(req, depth, *, after=False, reused=0, score=30000):
    profile = req["recheck_profile_sha256"] if after and req["task"] == "cross_profile_recheck" else req["cpu_profile_sha256"]
    restricted = req["task"] == "defend_response" or (req["task"] == "widen_responses" and not after)
    roots = req["root_moves"] if restricted else None
    order = [move for move in MOVES if roots is None or move in roots]
    conditions = {"schema": "rz-pals-private-cpu-conditions/1", "rules_state_sha256": SHA,
                  "rules_history_sha256": SHA, "board_fen": FEN, "profile_sha256": profile,
                  "value_identity": {"semantics": producer.CPU_VALUE, "weights_sha256": None, "training": {"kind": "bootstrap"}},
                  "search_version": producer.CPU_SEARCH, "quiescence_ply": 8, "root_moves": roots,
                  "search_conditions": producer.CPU_CONDITIONS +
                  f";profile={producer.RECHECK_PROFILE if after and req['task'] == 'cross_profile_recheck' else producer.PROFILE};max_depth=2;q_plies=8;tt_entries=16",
                  "root_selection": "explicit_registered_controls" if restricted else "unrestricted",
                  "resource_policy": {"max_wall_time_ms": 1000, "max_nodes_per_check": 100,
                                      "max_checks": 2, "search_deadline_reserve_ms": 100},
                  "white_to_move": True, "legal_moves": MOVES, "root_order": order,
                  "root_order_sha256": producer._hash("rz-pals-private-cpu-root-order/1", order)}
    return {"profile_sha256": profile, "conditions": conditions,
            "conditions_sha256": hashlib.sha256(producer._json(conditions)).hexdigest(),
            "score_scope": "completed_iteration" if depth else "frontier_only", "completed_depth": depth,
            "requested_depth": 2 if after else 1, "nodes": 20 if after else 10, "quiescence_nodes": 2,
            "completion": "depth_limit" if depth else "node_limit", "root_restricted": restricted,
            "raw_score": score, "pv": MOVES[:1], "white_to_move": True, "elapsed_ms": 2,
            "reused_completed_depth": reused, "score_provenance": producer.CPU_VALUE, "pv_rules_validated": True}


def response(req):
    before = report(req, 1)
    after = report(req, 2, after=True, reused=1 if req["task"] == "resume_task" else 0)
    return {"schema": req["schema"], "task": req["task"], "context_sha256": req["context_sha256"],
            "cpu_binary_sha256": req["cpu_binary_sha256"], "rules_state_sha256": req["rules_state_sha256"],
            "rules_history_sha256": req["rules_history_sha256"], "board_fen": req["expected_board_fen"],
            "branch_sha256": req["branch_sha256"], "status": "observed", "reason": None,
            "baseline": before, "after": after, "nodes": 30, "elapsed_ms": 4,
            "resume_kind": "completed_iteration" if req["task"] == "resume_task" else None,
            "product_verifier_enabled": False, "deadline_exceeded": False}


def see_request(task="resume_task"):
    value = request(task)
    value.update(schema=producer.SEE_CPU_SCHEMA, ordering_policy=producer.SEE_ORDERING)
    for key, profile in (("cpu_profile_sha256", producer.PROFILE), ("recheck_profile_sha256", producer.RECHECK_PROFILE)):
        value[key] = producer.profile_sha256(producer.cpu_profile_with_ordering(
            16, 2, 8, profile, ordering_policy=producer.SEE_ORDERING))
    value["branch_sha256"] = producer._hash("rz-pals-private-cpu-branch/1",
        {"parent_input_sha256": value["parent_input_sha256"], "prefix": [], "root_moves": []})
    value.pop("context_sha256")
    value["context_sha256"] = producer._hash(producer.SEE_CPU_SCHEMA, value)
    return value


def see_response(req):
    value = response(req)
    for key in ("baseline", "after"):
        conditions = value[key]["conditions"]
        profile = producer.RECHECK_PROFILE if key == "after" and req["task"] == "cross_profile_recheck" else producer.PROFILE
        conditions.update(schema=producer.SEE_CONDITIONS_SCHEMA, search_version=producer.SEE_CPU_SEARCH,
                          search_conditions=producer.cpu_conditions_with_ordering(16, 2, 8, profile,
                                                   ordering_policy=producer.SEE_ORDERING))
        value[key]["conditions_sha256"] = hashlib.sha256(producer._json(conditions)).hexdigest()
    return value


def parent():
    snapshot = {"game_id": "numeric-fixture-game", "opening_id": "numeric-fixture-opening",
                "line_genealogy_id": "numeric-fixture-line", "position_command": "position fen " + FEN,
                "board_fen": FEN, "actual_history": [], "rules_state_sha256": SHA,
                "rules_history_sha256": SHA, "transposition_sha256": SHA, "encoding_sha256": SHA,
                "source": {"kind": "own_cpu", "cpu_binary_sha256": SHA,
                           "evaluator_configuration_sha256": SHA, "model_weights_sha256": None},
                "frozen_epoch": 0, "input_revision": 0, "capture_sequence": 5,
                "white_to_move": True, "role": "proposer", "legal_moves": MOVES, "public_records": []}
    frozen = {"snapshot": snapshot, "sha256": seal_snapshot(snapshot)}
    board = [0] * 64
    board[7], board[56] = 6, 12
    encoding = EncodedSnapshot(frozen["sha256"], SHA, tuple(board), (0.0,) * 16, (), (0.0,) * 16)
    return {"input": frozen, "future_label": None, "verifier_private": None}, encoding


def cpu_storage_fixture():
    original, encoding = parent()
    context, req = TaskContext(SHA, SHA, 0), request()
    source = {"kind": "own_pals", "model_configuration_sha256": "b" * 64, "model_weights_sha256": "c" * 64}
    row, _ = producer.verifier_input(original, encoding, source, context, (0.0,) * 16, "d" * 64, 1)
    actual = response(req)
    evidence = {"input_sha256": row["input"]["sha256"], "context": producer.asdict(context),
                "request": req, "response": actual, "observed_information": producer.observed_gain(req, actual)}
    evidence_sha = producer._hash(producer.GAIN_SCHEMA, evidence)
    label = producer.future_label(row["input"]["snapshot"], context, "resume_task", req, actual, evidence_sha)
    private = {"task_kind": "resume_task", "control_sha256": req["context_sha256"], "private_latent": [0.25]}
    artifacts = (
        ("cpu-evidence.jsonl", {"evidence_sha256": evidence_sha, **evidence}),
        ("future-labels.jsonl", {"input_sha256": row["input"]["sha256"], "future_label": label,
                                 "observed_information_evidence_sha256": evidence_sha}),
        ("private-records.jsonl", {"record": dict(row, future_label=label, verifier_private=private),
                                  "evidence_sha256": evidence_sha, "task_context": producer.asdict(context),
                                  "loss": {"task": 0.0}, "comparative_training_target": False}),
    )
    return row, context, req, private, artifacts, producer._json(actual) + b"\n"


class VerifierProducerTests(unittest.TestCase):
    def setUp(self):
        self.step_guard = patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer updates forbidden"))
        self.step_guard.start()
        self.addCleanup(self.step_guard.stop)

    def test_see_helpers_preserve_legacy_profile_bytes_and_value_semantics(self):
        legacy = producer.cpu_profile(16, 2, 8)
        self.assertEqual(producer._json(legacy), producer._json(producer.cpu_profile_with_ordering(16, 2, 8)))
        self.assertEqual(legacy, {"domain": producer.CPU_SCHEMA, "search": producer.CPU_SEARCH,
            "evaluator": producer.CPU_VALUE, "profile": producer.PROFILE, "tt_entries": 16,
            "max_depth": 2, "quiescence_ply": 8, "selective_reductions": False})
        selected = producer.cpu_profile_with_ordering(16, 2, 8, ordering_policy=producer.SEE_ORDERING)
        self.assertNotEqual(producer.profile_sha256(legacy), producer.profile_sha256(selected))
        self.assertEqual(selected["domain"], producer.SEE_CPU_SCHEMA)
        self.assertEqual(selected["ordering_policy"], producer.SEE_ORDERING_IDENTITY)
        self.assertEqual(selected["evaluator"], legacy["evaluator"])
        self.assertNotIn("ordering_policy", request())
        self.assertEqual(producer.cpu_bridge_arguments("/owned/cpu", request()), ["/owned/cpu"])

    def test_see_bridge_literal_argv_requires_sealed_matching_request_before_spawn(self):
        selected = see_request()
        self.assertEqual(producer.cpu_bridge_arguments("/owned/cpu", selected, ordering_policy=producer.SEE_ORDERING),
                         ["/owned/cpu", "--cpu-ordering=legal-see-v1"])
        limits = producer.Limits(max_games=1, max_steps=1, max_nodes=1000, max_wall_time_ms=10000,
                                 max_output_bytes=65536, max_forward_flops=1000)
        with patch.object(producer.subprocess, "Popen", side_effect=AssertionError("must not spawn")) as popen:
            with self.assertRaises(producer.CpuBridgeFailure):
                producer.run_cpu_bridge("/owned/cpu", selected, limits)
            with self.assertRaises(producer.CpuBridgeFailure):
                producer.run_cpu_bridge("/owned/cpu", request(), limits, ordering_policy=producer.SEE_ORDERING)
            popen.assert_not_called()
        for key, changed in (("ordering_policy", None), ("ordering_policy", "unknown"),
                             ("context_sha256", SHA), ("cpu_profile_sha256", request()["cpu_profile_sha256"]),
                             ("schema", producer.CPU_SCHEMA)):
            bad = dict(selected, **{key: changed})
            with self.assertRaises(ValueError):
                producer.cpu_bridge_arguments("/owned/cpu", bad, ordering_policy=producer.SEE_ORDERING)

    def test_see_progress_keeps_raw_evidence_and_masks_comparative_targets(self):
        req, actual = see_request(), see_response(see_request())
        producer.report_with_ordering(actual["baseline"], req, req["cpu_profile_sha256"], ordering_policy=producer.SEE_ORDERING)
        gain = producer.observed_gain_with_ordering(req, actual, ordering_policy=producer.SEE_ORDERING)
        self.assertEqual(gain["new_completed_depth"], 1)
        self.assertFalse(gain["comparative_preference_available"])
        self.assertIsNone(gain["preference_rank"])
        self.assertFalse(gain["wdl_inferred"])
        self.assertEqual(actual["baseline"]["raw_score"], 30000)
        with self.assertRaises(ValueError):
            producer._report(actual["baseline"], req, req["cpu_profile_sha256"])
        with self.assertRaises(ValueError):
            producer.observed_gain(req, actual)
        snapshot = parent()[0]["input"]["snapshot"]
        context = TaskContext(req["branch_sha256"], req["cpu_profile_sha256"], 0)
        label = producer.future_label(snapshot, context, req["task"], req, actual, SHA)
        self.assertIsNone(label["policy"])
        self.assertIsNone(label["value_wdl"])
        self.assertTrue(all(t["preference_rank"] is None for t in label["verifier_tasks"]))

    def test_see_full_conditions_resealed_mismatch_is_rejected(self):
        req, actual = see_request(), see_response(see_request())
        changes = (("schema", "rz-pals-private-cpu-conditions/1"),
                   ("search_version", producer.CPU_SEARCH),
                   ("search_conditions", actual["baseline"]["conditions"]["search_conditions"].replace("search+qsearch+exchange", "search+qsearch")),
                   ("value_identity", {"semantics": "other", "weights_sha256": None, "training": {"kind": "bootstrap"}}))
        for field, value in changes:
            bad = copy.deepcopy(actual["baseline"])
            bad["conditions"][field] = value
            bad["conditions_sha256"] = hashlib.sha256(producer._json(bad["conditions"])).hexdigest()
            with self.assertRaises(ValueError):
                producer.report_with_ordering(bad, req, req["cpu_profile_sha256"], ordering_policy=producer.SEE_ORDERING)
        bad = copy.deepcopy(actual["baseline"])
        bad["nodes"] = 101
        with self.assertRaises(ValueError):
            producer.report_with_ordering(bad, req, req["cpu_profile_sha256"], ordering_policy=producer.SEE_ORDERING)

    def test_see_partial_and_canceled_reports_do_not_invent_completion_or_rank(self):
        req, actual = see_request(), see_response(see_request())
        for completion in ("node_limit", "deadline", "canceled", "quiescence_limit"):
            partial = copy.deepcopy(actual)
            partial["after"].update(completion=completion, completed_depth=1)
            gain = producer.observed_gain_with_ordering(req, partial, ordering_policy=producer.SEE_ORDERING)
            self.assertFalse(gain["actual_question_complete"])
            self.assertEqual(gain["new_completed_depth"], 0)
            self.assertIsNone(gain["preference_rank"])
        deferred_req = see_request("defer")
        deferred = dict(see_response(deferred_req), status="deferred", baseline=None, after=None, nodes=0, resume_kind=None)
        gain = producer.observed_gain_with_ordering(deferred_req, deferred, ordering_policy=producer.SEE_ORDERING)
        self.assertEqual(gain["status"], "deferred")
        self.assertFalse(gain["baseline_observed"])

    def test_see_cross_profile_is_fresh_observation_with_same_ordering(self):
        req = see_request("cross_profile_recheck")
        actual = see_response(req)
        gain = producer.observed_gain_with_ordering(req, actual, ordering_policy=producer.SEE_ORDERING)
        self.assertTrue(gain["new_completed_profile_observation"])
        self.assertFalse(gain["same_conditions"])
        self.assertEqual(gain["new_completed_depth"], 0)
        self.assertIsNone(gain["preference_rank"])
        self.assertEqual(actual["after"]["reused_completed_depth"], 0)
        self.assertEqual(actual["after"]["conditions"]["search_version"], producer.SEE_CPU_SEARCH)

    def test_declared_query_uses_pre_result_controls_only(self):
        row, _ = parent()
        mask, reasons = producer.eligible_tasks(TASKS, {"prefix": [], "root_moves": []}, MOVES)
        self.assertFalse(mask[TASKS.index("widen_responses")])
        self.assertFalse(mask[TASKS.index("lower_selectivity")])
        self.assertEqual(reasons["lower_selectivity"], "reductions_already_disabled")
        query = producer.private_query(mask, row["input"]["snapshot"], baseline_depth=1, requested_depth=2,
                                       max_nodes_per_check=100, max_wall_time_ms=1000, budget_bucket=4)
        self.assertEqual(len(query), len(producer.QUERY_FIELDS))
        self.assertEqual(query[-2:], (1.0, 0.0))

    def test_widen_requires_actual_strict_pre_result_subset(self):
        with self.assertRaises(ValueError):
            producer.eligible_tasks(("widen_responses",), {"prefix": [], "root_moves": MOVES}, MOVES)
        mask, _ = producer.eligible_tasks(("widen_responses",), {"prefix": [], "root_moves": MOVES[:1]}, MOVES)
        self.assertTrue(mask[2])
        self.assertFalse(mask[4])
        with self.assertRaises(ValueError):
            producer.eligible_tasks(TASKS, {"prefix": [], "root_moves": [MOVES[0], MOVES[0]]}, MOVES)

    def test_selection_masks_unavailable_and_rejects_nonfinite(self):
        mask = [False] * 7
        mask[4] = True
        self.assertEqual(producer.select_task(torch.tensor([[9., 9., 9., 9., -2., 9., 9.]]), mask), "resume_task")
        with self.assertRaises(ValueError):
            producer.select_task(torch.tensor([[0., 0., 0., 0., float("nan"), 0., 0.]]), mask)

    def test_fresh_verifier_source_and_private_query_seal_preserve_parent(self):
        original, encoding = parent()
        saved = copy.deepcopy(original)
        source = {"kind": "own_pals", "model_configuration_sha256": "b" * 64, "model_weights_sha256": "c" * 64}
        context = TaskContext(SHA, SHA, 4)
        query = (1.0,) * 16
        first, first_encoding = producer.verifier_input(original, encoding, source, context, query,
                                                        producer._hash(producer.QUERY_SCHEMA, {"query": query}), 1)
        second, _ = producer.verifier_input(original, encoding, source, context, (0.0,) * 16,
                                           producer._hash(producer.QUERY_SCHEMA, {"query": (0.0,) * 16}), 1)
        self.assertEqual(original, saved)
        self.assertEqual(first["input"]["snapshot"]["source"], source)
        self.assertNotEqual(first["input"]["sha256"], second["input"]["sha256"])
        payload = producer._encoded_json(first_encoding, original["input"]["sha256"], context, {"query": query})
        seal = payload.pop("payload_sha256")
        self.assertEqual(seal, producer._hash(producer.QUERY_SCHEMA, payload))

    def test_gain_counts_new_scope_not_raw_score_or_nodes(self):
        req = request()
        actual = response(req)
        actual["after"]["raw_score"] = -30000
        gain = producer.observed_gain(req, actual)
        self.assertEqual(gain["new_completed_depth"], 1)
        self.assertFalse(gain["all_defenses_mate_certified"])
        self.assertFalse(gain["wdl_inferred"])
        self.assertIsNone(gain["preference_rank"])

    def test_resume_cannot_be_fresh_analyze_or_wrong_conditions(self):
        req = request()
        for mutate in (lambda x: x["after"].update(reused_completed_depth=0),
                       lambda x: x.update(resume_kind=None),
                       lambda x: x["after"].update(conditions_sha256="b" * 64),
                       lambda x: x["after"].update(score_provenance="rz-fake")):
            actual = response(req)
            mutate(actual)
            with self.assertRaises(ValueError):
                producer.observed_gain(req, actual)

    def test_cross_profile_is_new_conditional_observation_not_depth_gain(self):
        req = request("cross_profile_recheck")
        gain = producer.observed_gain(req, response(req))
        self.assertFalse(gain["same_conditions"])
        self.assertEqual(gain["new_completed_depth"], 0)
        self.assertTrue(gain["new_completed_profile_observation"])

    def test_widen_preserves_restricted_and_unrestricted_scopes(self):
        req = request("widen_responses", MOVES[:1])
        gain = producer.observed_gain(req, response(req))
        self.assertFalse(gain["same_conditions"])
        self.assertEqual(gain["new_completed_depth"], 0)
        self.assertTrue(gain["new_unrestricted_completed_observation"])
        self.assertEqual((gain["root_choices_before"], gain["root_choices_after"]), (1, 2))

    def test_defer_and_unavailable_cannot_invent_results(self):
        req = request("defer")
        actual = response(req)
        with self.assertRaises(ValueError):
            producer.observed_gain(req, actual)
        actual.update(status="deferred", baseline=None, after=None, nodes=0, resume_kind=None)
        self.assertEqual(producer.observed_gain(req, actual)["status"], "deferred")
        req = request()
        actual = response(req)
        actual.update(status="unavailable", reason="no_completed_iteration_token", baseline=report(req, 0),
                      after=None, nodes=10, resume_kind=None)
        gain = producer.observed_gain(req, actual)
        self.assertTrue(gain["baseline_observed"])
        self.assertIsNone(producer.future_label(parent()[0]["input"]["snapshot"], TaskContext(SHA, SHA, 0), "resume_task", req, actual, SHA))

    def test_single_dispatch_all_ranks_unknown_and_loss_masked(self):
        original, encoding = parent()
        req, context = request(), TaskContext(SHA, SHA, 0)
        source = {"kind": "own_pals", "model_configuration_sha256": "b" * 64, "model_weights_sha256": "c" * 64}
        row, encoded = producer.verifier_input(original, encoding, source, context, (0.0,) * 16, "d" * 64, 1)
        row["future_label"] = producer.future_label(row["input"]["snapshot"], context, "resume_task", req, response(req), SHA)
        self.assertTrue(all(v["preference_rank"] is None for v in row["future_label"]["verifier_tasks"]))
        self.assertIsNone(row["future_label"]["value_wdl"])
        data = ValidatedDataset([row], {"games": {"numeric-fixture-game": "train"}},
                                {"cpu_binary_sha256": [SHA], "input_sources": [source]}, {encoded.input_sha256: encoded})
        batch = data.collate([0], "verifier", task_contexts=[context])
        outputs = (torch.zeros((1, 2)), torch.zeros((1, 3)), torch.zeros((1, 16, 384)), torch.zeros((1, 7)))
        losses = masked_losses(outputs, batch)
        self.assertFalse(batch.task_mask.any())
        self.assertTrue(all(float(value) == 0 for value in losses.values()))

    def test_all_limits_cancel_and_step_bounds(self):
        now = [10.0]
        limits = producer.Limits(max_games=1, max_steps=1, max_nodes=10, max_wall_time_ms=1000,
                                 max_output_bytes=32768, max_forward_flops=100, clock=lambda: now[0])
        limits.step("game")
        with self.assertRaises(ValueError):
            limits.step("game")
        with self.assertRaises(ValueError):
            limits.charge("nodes", 11)
        now[0] = 11.0
        with self.assertRaises(TimeoutError):
            limits.check()
        with tempfile.TemporaryDirectory() as temporary:
            cancel = Path(temporary) / "cancel.flag"
            limits = producer.Limits(max_games=1, max_steps=1, max_nodes=10, max_wall_time_ms=1000,
                                     max_output_bytes=32768, max_forward_flops=100, cancel_file=cancel)
            cancel.touch()
            with self.assertRaises(InterruptedError):
                limits.check()

    def test_secret_and_any_git_output_paths_fail_before_read_or_write(self):
        with self.assertRaises(ValueError):
            producer._path(".env")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            limits = producer.Limits(max_games=1, max_steps=1, max_nodes=10, max_wall_time_ms=1000,
                                     max_output_bytes=32768, max_forward_flops=100)
            with self.assertRaises(ValueError):
                producer.PrivateBank(root / "new-bank", limits)
            self.assertFalse((root / "new-bank").exists())

    def test_partial_bank_is_retained_and_marked_unusable(self):
        with tempfile.TemporaryDirectory() as temporary:
            limits = producer.Limits(max_games=1, max_steps=1, max_nodes=10, max_wall_time_ms=1000,
                                     max_output_bytes=32768, max_forward_flops=100)
            bank = producer.PrivateBank(Path(temporary) / "private", limits)
            bank.append("inputs.jsonl", {"numeric_only": True})
            receipt = bank.finish({"complete": False, "failure": {"type": "test"}})
            self.assertFalse(receipt["complete"])
            self.assertTrue((bank.root / "inputs.jsonl").is_file())
            actual = producer._parse((bank.root / "receipt.json").read_bytes())
            self.assertEqual(actual["optimizer_steps"], 0)
            self.assertFalse(actual["product_verifier_enabled"])

    def test_cpu_dispatch_output_and_nodes_admission_is_atomic(self):
        row, context, req, private, _, _ = cpu_storage_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            limits = producer.Limits(max_games=1, max_steps=1, max_nodes=1000, max_wall_time_ms=10000,
                                     max_output_bytes=65536, max_forward_flops=1000)
            bank = producer.PrivateBank(Path(temporary) / "admission", limits)
            before = dict(limits.usage)
            with self.assertRaisesRegex(ValueError, "resource limit: output_bytes"):
                bank.reserve_cpu(req, row, private, context)
            self.assertEqual(limits.usage, before)
            self.assertFalse(any(bank.root.iterdir()))

    def test_successful_cpu_capture_survives_each_post_result_storage_failure(self):
        row, context, req, private, artifacts, raw = cpu_storage_fixture()
        sealed = copy.deepcopy(row)
        for failed_name in (name for name, _ in artifacts):
            with self.subTest(stage=failed_name), tempfile.TemporaryDirectory() as temporary:
                limits = producer.Limits(max_games=1, max_steps=1, max_nodes=1000, max_wall_time_ms=10000,
                                         max_output_bytes=262144, max_forward_flops=1000)
                bank = producer.PrivateBank(Path(temporary) / "write-failure", limits)
                bank.append("inputs.jsonl", row["input"])
                inputs_before = (bank.root / "inputs.jsonl").read_bytes()
                evidence, recovery, pipe = bank.reserve_cpu(req, row, private, context)
                pipe.consume(len(raw))
                pipe.release()
                limits.usage["nodes"] -= 2 * req["max_nodes_per_check"] - 30
                # Fill all unreserved bytes: recovery must use its own credit,
                # even when the ordinary run output quota has no headroom.
                limits.charge("output_bytes", limits.maximum["output_bytes"] - limits.usage["output_bytes"])
                primary = OSError("injected post-result write failure")
                real_open = Path.open

                def fail_artifact(path, *args, **kwargs):
                    if path == bank.root / failed_name:
                        raise primary
                    return real_open(path, *args, **kwargs)

                counts = {"selections": 1, "cpu_dispatches": 1, "cpu_checks_observed": 2, "future_labels": 0}
                capture = {"stdout": raw, "stderr": b"", "spawned": True, "exit_code": 0, "reaped": True}
                with patch.object(Path, "open", new=fail_artifact), self.assertRaises(OSError) as raised:
                    try:
                        for name, value in artifacts:
                            bank.append(name, value, reservation=evidence)
                    except OSError as error:
                        producer._record_producer_failure(bank, error, counts=counts,
                                                          checkpoint_sha256=SHA, cpu_binary_sha256=SHA,
                                                          capture=capture, reservations=(evidence, recovery, pipe), stage=failed_name)
                        raise
                self.assertIs(raised.exception, primary)
                receipt = producer._parse((bank.root / "receipt.json").read_bytes())
                self.assertFalse(receipt["complete"])
                self.assertEqual(receipt["failure"]["type"], "OSError")
                self.assertEqual(receipt["failure"]["stage"], failed_name)
                self.assertTrue(receipt["failure"]["cpu_raw_output_preserved"])
                self.assertEqual(receipt["failure"]["cpu_observation"]["exit_code"], 0)
                self.assertEqual(receipt["failure"]["cpu_observation"]["stdout_sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(receipt["counts"], counts)
                self.assertEqual((bank.root / "failed-cpu-stdout.bin").read_bytes(), raw)
                self.assertEqual((bank.root / "inputs.jsonl").read_bytes(), inputs_before)
                self.assertEqual(row, sealed)
                self.assertEqual(limits.usage["nodes"], 30)
                self.assertLessEqual(limits.usage["output_bytes"], limits.maximum["output_bytes"])
                self.assertLessEqual(sum(path.stat().st_size for path in bank.root.iterdir()), limits.maximum["output_bytes"])

    def test_post_result_quota_failure_preserves_raw_with_reserved_credit(self):
        row, context, req, private, _, raw = cpu_storage_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            limits = producer.Limits(max_games=1, max_steps=1, max_nodes=1000, max_wall_time_ms=10000,
                                     max_output_bytes=262144, max_forward_flops=1000)
            bank = producer.PrivateBank(Path(temporary) / "quota-failure", limits)
            evidence, recovery, pipe = bank.reserve_cpu(req, row, private, context)
            pipe.consume(len(raw))
            pipe.release()
            capture = {"stdout": raw, "stderr": b"", "spawned": True, "exit_code": 0, "reaped": True}
            with self.assertRaisesRegex(ValueError, "sealed output reservation") as raised:
                try:
                    bank.append("cpu-evidence.jsonl", {"oversized": "x" * evidence.bounds["cpu-evidence.jsonl"]},
                                reservation=evidence)
                except ValueError as error:
                    producer._record_producer_failure(bank, error, counts={"cpu_dispatches": 1},
                                                      checkpoint_sha256=SHA, cpu_binary_sha256=SHA,
                                                      capture=capture, reservations=(evidence, recovery, pipe), stage="cpu-evidence.jsonl")
                    raise
            receipt = producer._parse((bank.root / "receipt.json").read_bytes())
            self.assertEqual(receipt["failure"]["message"], str(raised.exception))
            self.assertTrue(receipt["failure"]["cpu_raw_output_preserved"])
            self.assertEqual((bank.root / "failed-cpu-stdout.bin").read_bytes(), raw)
            self.assertLessEqual(limits.usage["output_bytes"], limits.maximum["output_bytes"])

    def test_raw_and_receipt_retention_failures_do_not_replace_primary_error(self):
        row, context, req, private, _, raw = cpu_storage_fixture()
        for failed_stage in ("raw", "receipt"):
            with self.subTest(stage=failed_stage), tempfile.TemporaryDirectory() as temporary:
                limits = producer.Limits(max_games=1, max_steps=1, max_nodes=1000, max_wall_time_ms=10000,
                                         max_output_bytes=262144, max_forward_flops=1000)
                bank = producer.PrivateBank(Path(temporary) / "retention-failure", limits)
                reservations = bank.reserve_cpu(req, row, private, context)
                capture = {"stdout": raw, "stderr": b"diagnostic", "spawned": True, "exit_code": 1, "reaped": True}
                primary = ValueError("original post-result failure")
                real_failed_bytes = bank.failed_cpu_bytes

                def fail_stdout(name, content, **kwargs):
                    if name == "failed-cpu-stdout.bin":
                        raise OSError("injected raw retention failure")
                    return real_failed_bytes(name, content, **kwargs)

                target = patch.object(bank, "failed_cpu_bytes", side_effect=fail_stdout) if failed_stage == "raw" else patch.object(bank, "finish", side_effect=OSError("injected receipt failure"))
                with target, self.assertRaises(ValueError) as raised:
                    try:
                        raise primary
                    except ValueError as error:
                        producer._record_producer_failure(bank, error, counts={"cpu_dispatches": 1},
                                                          checkpoint_sha256=SHA, cpu_binary_sha256=SHA,
                                                          capture=capture, reservations=reservations, stage="private-records.jsonl")
                        raise
                self.assertIs(raised.exception, primary)
                self.assertLessEqual(limits.usage["output_bytes"], limits.maximum["output_bytes"])
                if failed_stage == "raw":
                    receipt = producer._parse((bank.root / "receipt.json").read_bytes())
                    self.assertFalse(receipt["failure"]["cpu_raw_output_preserved"])
                    self.assertEqual(receipt["failure"]["retention_failures"][0]["stream"], "stdout")
                    self.assertEqual((bank.root / "failed-cpu-stderr.bin").read_bytes(), b"diagnostic")
                else:
                    self.assertEqual(primary.preservation_failures[0]["stage"], "failure_receipt")
                    self.assertEqual(primary.producer_failure["counts"], {"cpu_dispatches": 1})
                    self.assertFalse(primary.producer_failure["complete"])
                    self.assertEqual((bank.root / "failed-cpu-stdout.bin").read_bytes(), raw)

    def test_see_bank_reload_requires_independent_selected_policy_and_preserves_mask(self):
        original, native = parent()
        parents = ValidatedDataset([original], {"games": {"numeric-fixture-game": "train"}},
                                   {"cpu_binary_sha256": [SHA], "input_sources": [original["input"]["snapshot"]["source"]]},
                                   {native.input_sha256: native})
        req = see_request()
        req["parent_input_sha256"] = native.input_sha256
        req["branch_sha256"] = producer._hash("rz-pals-private-cpu-branch/1",
                                {"parent_input_sha256": native.input_sha256, "prefix": [], "root_moves": []})
        req.pop("context_sha256")
        req["context_sha256"] = producer._hash(producer.SEE_CPU_SCHEMA, req)
        context = TaskContext(req["branch_sha256"], req["cpu_profile_sha256"], 0)
        source = {"kind": "own_pals", "model_configuration_sha256": "b" * 64, "model_weights_sha256": "c" * 64}
        declaration = {"schema": producer.QUERY_SCHEMA, "model_configuration_sha256": "b" * 64,
                       "private_encoder_source_sha256": "d" * 64, "cpu_ordering_policy": producer.SEE_ORDERING_IDENTITY}
        schema_sha = producer._hash(producer.QUERY_SCHEMA, declaration)
        mask, _ = producer.eligible_tasks(("resume_task",), {"prefix": [], "root_moves": []}, MOVES)
        query = producer.private_query(mask, original["input"]["snapshot"], baseline_depth=1, requested_depth=2,
                                       max_nodes_per_check=100, max_wall_time_ms=1000, budget_bucket=0)
        derivation = {"encoding_schema_sha256": schema_sha, "parent_input_sha256": native.input_sha256,
                      "context": producer.asdict(context), "prefix": [], "root_moves": [],
                      "allowed_tasks": ["resume_task"], "eligible_tasks": mask, "query": list(query),
                      "baseline_depth": 1, "requested_depth": 2, "max_nodes_per_check": 100,
                      "max_task_wall_time_ms": 1000}
        row, encoding = producer.verifier_input(original, native, source, context, query,
                                                producer._hash(producer.QUERY_SCHEMA, derivation), 1)
        actual = see_response(req)
        evidence = {"input_sha256": row["input"]["sha256"], "context": producer.asdict(context), "request": req,
                    "response": actual, "observed_information": producer.observed_gain_with_ordering(req, actual, ordering_policy=producer.SEE_ORDERING)}
        evidence_sha = producer._hash(producer.GAIN_SCHEMA, evidence)
        row["future_label"] = producer.future_label(row["input"]["snapshot"], context, "resume_task", req, actual, evidence_sha)
        row["verifier_private"] = {"task_kind": "resume_task", "control_sha256": req["context_sha256"], "private_latent": [0.25]}
        with tempfile.TemporaryDirectory() as temporary:
            limits = producer.Limits(max_games=1, max_steps=1, max_nodes=1000, max_wall_time_ms=10000,
                                     max_output_bytes=65536, max_forward_flops=1000)
            bank = producer.PrivateBank(Path(temporary) / "see-reload", limits)
            bank.append("inputs.jsonl", {"parent_input_sha256": native.input_sha256, "input": row["input"],
                                          "context": producer.asdict(context), "split": "train"})
            bank.append("encodings.jsonl", producer._encoded_json(encoding, native.input_sha256, context, derivation))
            bank.append("decisions.jsonl", {"input_sha256": row["input"]["sha256"], "request": req,
                                            "task": "resume_task", "verifier_private": row["verifier_private"]})
            bank.append("cpu-evidence.jsonl", {"evidence_sha256": evidence_sha, **evidence})
            bank.append("future-labels.jsonl", {"input_sha256": row["input"]["sha256"], "future_label": row["future_label"],
                                               "observed_information_evidence_sha256": evidence_sha})
            bank.append("private-records.jsonl", {"record": row, "evidence_sha256": evidence_sha})
            receipt = bank.finish({"complete": True, "failure": None, "status": "checks_passed_before_receipt_write",
                                   "counts": {"selections": 1}, "native_verifier_encoded": False, "trained": False,
                                   "cpu_ordering_policy": producer.SEE_ORDERING, "checkpoint_sha256": "c" * 64,
                                   "no_comparative_training_target": True, "query_encoding": declaration,
                                   "encoding_schema_sha256": schema_sha,
                                   "source_registry": {"cpu_binary_sha256": [SHA], "input_sources": [source]}})
            args = {"expected_receipt_sha256": receipt["receipt_sha256"], "parents": parents,
                    "expected_checkpoint_sha256": "c" * 64, "expected_cpu_binary_sha256": SHA,
                    "expected_private_encoder_source_sha256": "d" * 64}
            with self.assertRaises(ValueError):
                producer.load_private_verifier_bank(bank.root, **args)
            loaded = producer.load_private_verifier_bank(bank.root, **args, ordering_policy=producer.SEE_ORDERING)
            self.assertEqual(len(loaded.records), 1)
            self.assertFalse(loaded.collate([0], "verifier", task_contexts=[context]).task_mask.any())
            self.assertIsNone(loaded.records[0]["future_label"]["policy"])
            self.assertIsNone(loaded.records[0]["future_label"]["value_wdl"])
            with (bank.root / "cpu-evidence.jsonl").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaises(ValueError):
                producer.load_private_verifier_bank(bank.root, **args, ordering_policy=producer.SEE_ORDERING)

    def test_private_bank_reload_checks_query_context_native_parent_and_future_label(self):
        original, native = parent()
        parents = ValidatedDataset([original], {"games": {"numeric-fixture-game": "train"}},
                                   {"cpu_binary_sha256": [SHA], "input_sources": [original["input"]["snapshot"]["source"]]},
                                   {native.input_sha256: native})
        req = request()
        req["parent_input_sha256"] = native.input_sha256
        req["branch_sha256"] = producer._hash("rz-pals-private-cpu-branch/1",
                                              {"parent_input_sha256": native.input_sha256, "prefix": [], "root_moves": []})
        req.pop("context_sha256")
        req["context_sha256"] = producer._hash(producer.CPU_SCHEMA, req)
        context = TaskContext(req["branch_sha256"], req["cpu_profile_sha256"], 0)
        source = {"kind": "own_pals", "model_configuration_sha256": "b" * 64, "model_weights_sha256": "c" * 64}
        declaration = {"schema": producer.QUERY_SCHEMA, "model_configuration_sha256": "b" * 64,
                       "private_encoder_source_sha256": "d" * 64}
        schema_sha = producer._hash(producer.QUERY_SCHEMA, declaration)
        mask, _ = producer.eligible_tasks(("resume_task",), {"prefix": [], "root_moves": []}, MOVES)
        query = producer.private_query(mask, original["input"]["snapshot"], baseline_depth=1, requested_depth=2,
                                       max_nodes_per_check=100, max_wall_time_ms=1000, budget_bucket=0)
        derivation = {"encoding_schema_sha256": schema_sha, "parent_input_sha256": native.input_sha256,
                      "context": producer.asdict(context), "prefix": [], "root_moves": [],
                      "allowed_tasks": ["resume_task"], "eligible_tasks": mask, "query": list(query),
                      "baseline_depth": 1, "requested_depth": 2, "max_nodes_per_check": 100,
                      "max_task_wall_time_ms": 1000}
        row, encoding = producer.verifier_input(original, native, source, context, query,
                                                producer._hash(producer.QUERY_SCHEMA, derivation), 1)
        actual = response(req)
        evidence = {"input_sha256": row["input"]["sha256"], "context": producer.asdict(context),
                    "request": req, "response": actual, "observed_information": producer.observed_gain(req, actual)}
        evidence_sha = producer._hash(producer.GAIN_SCHEMA, evidence)
        row["future_label"] = producer.future_label(row["input"]["snapshot"], context, "resume_task", req, actual, evidence_sha)
        row["verifier_private"] = {"task_kind": "resume_task", "control_sha256": req["context_sha256"], "private_latent": [0.25]}
        with tempfile.TemporaryDirectory() as temporary:
            limits = producer.Limits(max_games=1, max_steps=1, max_nodes=1000, max_wall_time_ms=10000,
                                     max_output_bytes=65536, max_forward_flops=1000)
            bank = producer.PrivateBank(Path(temporary) / "reload", limits)
            bank.append("inputs.jsonl", {"parent_input_sha256": native.input_sha256, "input": row["input"],
                                          "context": producer.asdict(context), "split": "train"})
            bank.append("encodings.jsonl", producer._encoded_json(encoding, native.input_sha256, context, derivation))
            bank.append("decisions.jsonl", {"input_sha256": row["input"]["sha256"], "request": req,
                                             "task": "resume_task", "verifier_private": row["verifier_private"]})
            bank.append("cpu-evidence.jsonl", {"evidence_sha256": evidence_sha, **evidence})
            bank.append("future-labels.jsonl", {"input_sha256": row["input"]["sha256"], "future_label": row["future_label"],
                                                 "observed_information_evidence_sha256": evidence_sha})
            bank.append("private-records.jsonl", {"record": row, "evidence_sha256": evidence_sha})
            receipt = bank.finish({"complete": True, "failure": None, "status": "checks_passed_before_receipt_write",
                                   "counts": {"selections": 1}, "native_verifier_encoded": False, "trained": False,
                                   "checkpoint_sha256": "c" * 64, "no_comparative_training_target": True,
                                   "query_encoding": declaration, "encoding_schema_sha256": schema_sha,
                                   "source_registry": {"cpu_binary_sha256": [SHA], "input_sources": [source]}})
            arguments = {"expected_receipt_sha256": receipt["receipt_sha256"], "parents": parents,
                         "expected_checkpoint_sha256": "c" * 64, "expected_cpu_binary_sha256": SHA,
                         "expected_private_encoder_source_sha256": "d" * 64}
            loaded = producer.load_private_verifier_bank(bank.root, **arguments)
            self.assertEqual(len(loaded.records), 1)
            self.assertFalse(loaded.collate([0], "verifier", task_contexts=[context]).task_mask.any())
            with self.assertRaises(ValueError):
                producer.load_private_verifier_bank(bank.root, **dict(arguments, expected_private_encoder_source_sha256="e" * 64))
            with (bank.root / "encodings.jsonl").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaises(ValueError):
                producer.load_private_verifier_bank(bank.root, **arguments)


if __name__ == "__main__":
    unittest.main()
