"""No Torch, model, optimizer or training execution; adversarial data contracts."""
import copy
import importlib.abc
import math
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural/optimizer execution forbidden in generation source tests")


_import_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _import_guard)
try:
    from rz_pals_model import generation_data as data
    from rz_pals_model import generation_training as train
finally:
    sys.meta_path.remove(_import_guard)


class SourceOnlyTests(unittest.TestCase):
    def setUp(self):
        self.import_guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.import_guard)

    def tearDown(self):
        sys.meta_path.remove(self.import_guard)


def row(role="proposer"):
    """A schema-only adversarial fixture; never admitted as real producer data."""
    anchor = {"rules_state_sha256": "a" * 64, "rules_history_sha256": "b" * 64,
              "board_fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "white_to_move": True}
    view = {"context_policy_sha256": "c" * 64, "anchors": [anchor], "records": [],
            "query": {"purpose": "propose_policy", "anchors": [0, 0, 0], "prefix": [], "proposal": [], "counter": [], "counter_present": False, "white_to_move": True},
            "token_budget": 512, "length_bucket": 8}
    snapshot = {"game_id": "g1", "opening_id": "o1", "line_genealogy_id": "l1", "transposition_sha256": "d" * 64,
                "role": role, "model_epoch": 0, "context_epoch": 0, "capture_sequence": 10,
                "question_id": "q1", "worker_id": "w1", "job_id": "j1", "slice_id": "s1",
                "rules_anchor": anchor, "actual_history": [64], "selected_view": view,
                "view_sha256": data.view_digest(view), "legal_moves": [64, 128], "action_support_mask": [True] * 7, "divergence_record_ids": [],
                "encoded_model_input": {"schema": "rovezero.pals-virtual-review-input.v1", "role": role,
                    "anchors": [{"board": [0] * 64, "metadata": [0.0] * 16}], "records": [],
                    "query": {"purpose": "propose_policy", "perspective": 1, "prefix": [], "proposal": [], "counter": [], "counter_present": False, "anchors": [0, 0, 0]},
                    "candidates": [[0, 1, 0], [0, 2, 0]], "divergences": [], "token_budget": 512}}
    return {"schema": data.DATA_DOMAIN, "source_kind": data.ACTUAL_SOURCE, "eligibility": "actual_generation_candidate",
            "input": snapshot, "input_sha256": data.seal_input(snapshot),
            "future_targets": dict.fromkeys(("policy", "wdl", "divergence", "task"), None)}


def label_base(record):
    s = record["input"]
    return {"observed_sequence": 20, "question_id": s["question_id"], "view_sha256": s["view_sha256"], "raw_evidence_sha256": "e" * 64}


def policy(record):
    return {**label_base(record), "source": "stockfish19_completed_multipv", "completed": True,
            "stale": False, "anchor": copy.deepcopy(record["input"]["rules_anchor"]), "ranked_moves": [128, 64],
            "teacher_binary_sha256": "6" * 64, "teacher_profile_sha256": "7" * 64, "go_sequence": 1, "completed_depth": 1}


def wdl(record):
    return {**label_base(record), "source": "rules_selfplay_trajectory", "game_id": "g1",
            "anchor": copy.deepcopy(record["input"]["rules_anchor"]), "trajectory_membership_sha256": "f" * 64,
            "rules_result_sha256": "a" * 64, "ending": "checkmate", "outcome": "white_win"}


def outcome(action, witnesses, cost):
    return {"action": action, "execution_id": "ex-" + action, "completed": True, "wall_cost_ns": cost,
            "new_witnesses": [{"witness_id": witness, "kind": "counter", "observed_sequence": 15,
                               "raw_evidence_sha256": "9" * 64} for witness in witnesses], "causal_receipt_sha256": "8" * 64}


def tasks(record, outcomes):
    known = data.derive_task_targets(record["input"], outcomes)["target_known_mask"]
    return {**label_base(record), "source": "actual_action_witness", "outcomes": outcomes, "target_known_mask": known}


def spec():
    return train.generation_spec(generation_id="g0", source_sha256="1" * 64, dataset_sha256="2" * 64,
                                 split_sha256="3" * 64, model_sha256="4" * 64, runtime_sha256="5" * 64)


def schema_dataset(records):
    """Only low-count authority-contract fixtures; always fail real admission."""
    splits = {r["input"]["opening_id"]: "train" for r in records}
    producer = {"schema": "rz-pals-actual-collection-authority/1", "source_kind": data.ACTUAL_SOURCE,
                "source_sha256": "1" * 64, "model_sha256": "4" * 64, "runtime_sha256": "5" * 64,
                "teacher_binary_sha256": "6" * 64, "teacher_profile_sha256": "7" * 64,
                "opening_split_sha256": data.digest("rz-pals-opening-split/1", splits), "registered_before_outcomes": True,
                "input_seals": [r["input_sha256"] for r in records], "actual_game_ids": [r["input"]["game_id"] for r in records]}
    receipts = {}
    result = {"schema": "rz-pals-rules-outcome-evidence/1", "game_id": "g1", "ending": "checkmate", "outcome": "white_win", "rules_terminal": True}
    result_sha256 = data.digest(result["schema"], result)
    receipts[result_sha256] = result
    authority = {"game_id": "g1", "ending": "checkmate", "outcome": "white_win",
                 "anchors": [records[0]["input"]["rules_anchor"]], "result_evidence_sha256": result_sha256}
    trajectory_sha256 = data.trajectory_digest(authority)
    for r in records:
        for head, target in r["future_targets"].items():
            if target is None:
                continue
            if head == "wdl":
                target["rules_result_sha256"] = result_sha256
                target["trajectory_membership_sha256"] = trajectory_sha256
            receipt = {"schema": "rz-pals-generation-head-evidence/1", "input_sha256": r["input_sha256"],
                       "head": head, "target": {k: v for k, v in target.items() if k != "raw_evidence_sha256"}}
            identity = data.digest(receipt["schema"], receipt)
            target["raw_evidence_sha256"] = identity
            receipts[identity] = receipt
    trajectories = {trajectory_sha256: authority}
    return data.GenerationDataset(records, producer=producer, trajectories=trajectories, opening_splits=splits, allowed_evidence=receipts)


class DataTests(SourceOnlyTests):
    def test_policy_wdl_have_independent_provenance_and_legal_order(self):
        r = row()
        r["future_targets"].update(policy=policy(r), wdl=wdl(r))
        result = data.validate_record(r)
        self.assertAlmostEqual(result["policy"][0], 1 / (1 + math.e))
        self.assertEqual(result["wdl"], [1.0, 0.0, 0.0])
        self.assertEqual(r["input_sha256"], data.seal_input(r["input"]))

    def test_future_target_cannot_mutate_sealed_input(self):
        r = row()
        r["input"]["actual_history"].append(192)
        with self.assertRaisesRegex(ValueError, "seal"):
            data.validate_record(r)

    def test_history_is_not_counted_as_review_tokens(self):
        r = row()
        r["input"]["actual_history"] = [64] * 1000
        r["input_sha256"] = data.seal_input(r["input"])
        self.assertIsNotNone(data.validate_record(r))
        r["input"]["selected_view"]["actual_history"] = []
        with self.assertRaisesRegex(ValueError, "fields"):
            data.seal_input(r["input"])

    def test_duplicate_query_tokens_are_charged(self):
        r = row()
        v = r["input"]["selected_view"]
        v["query"]["prefix"] = [64] * 256
        v["query"]["proposal"] = [64] * 256
        v["query"]["counter"] = [64]
        v["length_bucket"] = 256
        with self.assertRaisesRegex(ValueError, "token"):
            data.view_digest(v)

    def test_padding_bucket_is_minimal(self):
        r = row()
        r["input"]["selected_view"]["length_bucket"] = 256
        with self.assertRaisesRegex(ValueError, "bucket"):
            data.seal_input(r["input"])

    def test_future_public_record_refused(self):
        r = row()
        v = r["input"]["selected_view"]
        v["records"] = [{"record_id": "r1", "kind": "counter", "anchor": 0, "moves": [64],
                         "parent": None, "supersedes": None, "observed_sequence": 11, "perspective": 1,
                         "completed_iteration": True, "rules_terminal": False}]
        r["input"]["view_sha256"] = data.view_digest(v)
        with self.assertRaisesRegex(ValueError, "future"):
            data.seal_input(r["input"])

    def test_stale_incomplete_illegal_teacher_refused(self):
        for changes in ({"stale": True}, {"completed": False}, {"ranked_moves": [192]}, {"ranked_moves": [64, 64]}):
            r = row()
            r["future_targets"]["policy"] = policy(r)
            r["future_targets"]["policy"].update(changes)
            with self.assertRaises(ValueError):
                data.validate_record(r)

    def test_wdl_does_not_use_cp_crash_or_virtual_endpoint(self):
        for changes in ({"source": "cpu_cp"}, {"ending": "engine_crash"}, {"ending": "time_forfeit"},
                        {"anchor": {**row()["input"]["rules_anchor"], "rules_history_sha256": "0" * 64}}):
            r = row()
            r["future_targets"]["wdl"] = wdl(r)
            r["future_targets"]["wdl"].update(changes)
            with self.assertRaises(ValueError):
                data.validate_record(r)

    def test_view_question_identity_cannot_be_relabelled(self):
        for field in ("view_sha256", "question_id"):
            r = row()
            r["future_targets"]["policy"] = policy(r)
            r["future_targets"]["policy"][field] = "0" * 64
            with self.assertRaisesRegex(ValueError, "question/view"):
                data.validate_record(r)

    def test_diagnostic_boolean_or_string_flip_insufficient(self):
        r = row()
        r["eligibility"] = "diagnostic_only_arena_ineligible"
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            data.validate_record(r)
        r = row()
        with self.assertRaisesRegex(ValueError, "authority"):
            data.GenerationDataset([r], producer={}, trajectories={}, opening_splits={"o1": "train"}, allowed_evidence=[])

    def test_pareto_pair_requires_new_valid_witness_and_known_cost(self):
        r = row("validator")
        r["future_targets"]["task"] = tasks(r, [outcome("attack_repair", ["new1"], 100), outcome("defer", [], 200)])
        result = data.validate_record(r)["task"]
        self.assertEqual(result["pairs"], [(1, 6)])
        self.assertNotEqual(result["target_known_mask"], r["input"]["action_support_mask"])
        r["future_targets"]["task"]["outcomes"][0]["wall_cost_ns"] = None
        r["future_targets"]["task"]["target_known_mask"] = [False] * 7
        self.assertEqual(data.validate_record(r)["task"]["pairs"], [])

    def test_pareto_tradeoff_and_zero_evidence_are_masked(self):
        for outcomes in ([outcome("attack_repair", ["new1"], 200), outcome("defer", [], 100)],
                         [outcome("attack_repair", [], 200), outcome("defer", [], 100)]):
            r = row("validator")
            r["future_targets"]["task"] = tasks(r, outcomes)
            self.assertEqual(data.validate_record(r)["task"]["pairs"], [])

    def test_unsupported_action_is_not_known(self):
        r = row("validator")
        r["input"]["action_support_mask"][1] = False
        r["input_sha256"] = data.seal_input(r["input"])
        r["future_targets"]["task"] = {**label_base(r), "source": "actual_action_witness",
            "outcomes": [outcome("attack_repair", ["new1"], 100)], "target_known_mask": [False] * 7}
        with self.assertRaisesRegex(ValueError, "unsupported"):
            data.validate_record(r)

    def test_duplicate_old_witness_refused(self):
        r = row("validator")
        o = outcome("attack_repair", ["new1", "new1"], 100)
        r["future_targets"]["task"] = {**label_base(r), "source": "actual_action_witness", "outcomes": [o], "target_known_mask": [False] * 7}
        with self.assertRaisesRegex(ValueError, "duplicate/old"):
            data.validate_record(r)
        o["new_witnesses"] = o["new_witnesses"][:1]
        o["new_witnesses"][0]["observed_sequence"] = 10
        with self.assertRaisesRegex(ValueError, "duplicate/old"):
            data.validate_record(r)

    def test_duplicate_rows_do_not_fill_sample_quota(self):
        r = row()
        r["future_targets"]["policy"] = policy(r)
        observed = data.coverage([r, r])
        self.assertEqual(observed["proposer_valid_samples"], 1)
        self.assertEqual(observed["duplicate_inputs_excluded"], 1)
        with self.assertRaisesRegex(ValueError, "InsufficientValidTargets"):
            data.admit_phase([r], "pc")

    def test_cross_split_state_leak_is_excluded_symmetrically(self):
        a, b = row(), row()
        b["input"].update(game_id="g2", opening_id="o2", line_genealogy_id="l2")
        b["input_sha256"] = data.seal_input(b["input"])
        parts, report = data.split_records([a, b], {"o1": "train", "o2": "holdout"})
        self.assertFalse(any(parts.values()))
        self.assertEqual(report["cross_split_records_excluded"], 2)

    def test_duplicate_json_keys_and_nonfinite_are_rejected(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                data.strict_json(text)

    def test_authority_trajectory_join_and_immutable_train_split(self):
        r = row()
        r["future_targets"]["wdl"] = wdl(r)
        ds = schema_dataset([r])
        copied = ds.rows("train")
        copied[0]["input"]["game_id"] = "changed"
        self.assertEqual(ds.rows("train")[0]["input"]["game_id"], "g1")
        with self.assertRaisesRegex(ValueError, "split"):
            ds.rows("holdout", optimization=True)
        ds._records["train"][0]["future_targets"]["wdl"]["outcome"] = "black_win"
        with self.assertRaisesRegex(ValueError, "immutable"):
            ds.rows("train")

    def test_authority_does_not_broadcast_wdl_to_virtual_anchor(self):
        r = row()
        r["future_targets"]["wdl"] = wdl(r)
        ds = schema_dataset([r])
        trajectories = copy.deepcopy(ds._authority)
        next(iter(trajectories.values()))["anchors"] = []
        with self.assertRaisesRegex(ValueError, "trajectory"):
            data.GenerationDataset([r], producer=ds.authority, trajectories=trajectories, opening_splits={"o1": "train"}, allowed_evidence=ds._evidence)

    def test_full_raw_receipt_rejects_changed_policy_with_same_sha(self):
        r = row()
        r["future_targets"]["policy"] = policy(r)
        ds = schema_dataset([r])
        r["future_targets"]["policy"]["ranked_moves"].reverse()
        with self.assertRaisesRegex(ValueError, "full head evidence"):
            data.GenerationDataset([r], producer=ds.authority, trajectories={}, opening_splits={"o1": "train"}, allowed_evidence=ds._evidence)

    def test_exact_rules_state_leak_survives_transposition_renaming(self):
        a, b = row(), row()
        b["input"].update(game_id="g2", opening_id="o2", line_genealogy_id="l2", transposition_sha256="0" * 64)
        b["input_sha256"] = data.seal_input(b["input"])
        parts, report = data.split_records([a, b], {"o1": "train", "o2": "holdout"})
        self.assertFalse(any(parts.values()))
        self.assertEqual(report["cross_split_records_excluded"], 2)

    def test_known_mask_cannot_claim_unsupported_or_unknown_targets(self):
        r = row("validator")
        r["future_targets"]["task"] = tasks(r, [outcome("defer", [], 1)])
        r["future_targets"]["task"]["target_known_mask"][6] = True
        with self.assertRaisesRegex(ValueError, "target-known"):
            data.validate_record(r)

    def test_encoded_payload_does_not_reorder_legal_candidates(self):
        r = row()
        r["input"]["encoded_model_input"]["candidates"].reverse()
        with self.assertRaisesRegex(ValueError, "candidate"):
            data.seal_input(r["input"])


class TrainingSourceTests(SourceOnlyTests):
    def test_exact_registered_recipe_and_locked_role_limits(self):
        s = spec()
        self.assertEqual(s["role_updates"], {"proposer": 2000, "critic": 2000, "validator": 1000})
        for field, value in (("accumulation", 1), ("device", "cpu"), ("tf32", True)):
            bad = copy.deepcopy(s)
            bad[field] = value
            with self.assertRaises(ValueError):
                train.validate_spec(bad)

    def test_execution_is_refused_before_neural_import(self):
        with self.assertRaisesRegex(ValueError, "authorization"):
            train.GenerationOptimizers(None, spec())
        with self.assertRaisesRegex(ValueError, "disabled"):
            train.train_accumulation(None, None, None, [], None)

    def test_role_alternation_dispatch_completion_and_checkpoint_boundary(self):
        ledger = train.GenerationLedger(spec())
        with self.assertRaises(ValueError):
            ledger.dispatch_update("critic")
        for _ in range(125):
            for role in ("proposer", "critic"):
                ledger.dispatch_update(role)
                ledger.pending_microbatches = 8
                ledger.complete_update(role)
        self.assertTrue(ledger.checkpoint_boundary())
        ledger.dispatch_update("proposer")
        ledger.pending_microbatches = 1
        self.assertFalse(ledger.checkpoint_boundary())
        ledger.discard_partial(1000, 200)
        self.assertEqual(ledger.cost["partial_accumulation_ns"], 1000)
        self.assertEqual(ledger.dispatched["proposer"], 126)
        self.assertEqual(ledger.completed["proposer"], 125)

    def test_v_requires_completed_frozen_pc_phase(self):
        with self.assertRaises(ValueError):
            train.GenerationLedger(spec(), phase="v", next_role="validator")
        ledger = train.GenerationLedger(spec(), completed={"proposer": 2000, "critic": 2000, "validator": 0},
                                        dispatched={"proposer": 2000, "critic": 2000, "validator": 0}, phase="v", next_role="validator")
        self.assertTrue(ledger.checkpoint_boundary())

    def test_terminal_update_counts_cannot_resume_previous_phase(self):
        with self.assertRaisesRegex(ValueError, "alternation"):
            train.GenerationLedger(spec(), completed={"proposer": 2000, "critic": 2000, "validator": 0},
                                   dispatched={"proposer": 2000, "critic": 2000, "validator": 0})
        with self.assertRaisesRegex(ValueError, "terminal"):
            train.GenerationLedger(spec(), completed={"proposer": 2000, "critic": 2000, "validator": 1000},
                                   dispatched={"proposer": 2000, "critic": 2000, "validator": 1000}, phase="v", next_role="validator")

    def test_allmasked_has_no_optimizer_update_or_dispatch(self):
        records = [row() for _ in range(8)]
        ledger = train.GenerationLedger(spec())
        result = train.train_accumulation(None, None, ledger, records, None, execute=True)
        self.assertFalse(result["optimizer_updated"])
        self.assertEqual(sum(ledger.dispatched.values()), 0)

    def test_head_valid_counts_are_across_whole_group(self):
        records = [row() for _ in range(8)]
        records[0]["future_targets"]["policy"] = policy(records[0])
        records[1]["future_targets"]["wdl"] = wdl(records[1])
        self.assertEqual(train.head_counts(records, "proposer"), {"policy": 1, "wdl": 1, "divergence": 0, "task": 0})

    def test_unmapped_parameters_are_rejected(self):
        self.assertEqual(train.parameter_owner("public_encoder.foo"), "shared")
        self.assertEqual(train.parameter_owner("experts.critic.foo"), "critic")
        with self.assertRaises(ValueError):
            train.parameter_owner("utils.weight")

    def test_generation_window_disabled_before_authority_or_model_loading(self):
        with self.assertRaisesRegex(ValueError, "authorization"):
            train.train_generation_window(None, None, None, checkpoint_directory="unused")

    def test_actual_admission_refuses_raw_records(self):
        with self.assertRaisesRegex(ValueError, "independently admitted"):
            train.generation_admission(spec(), [row()], "pc")

    def test_checkpoint_phase_rejects_pc_freeze_before_restore(self):
        ledger = train.GenerationLedger(spec())
        with self.assertRaisesRegex(ValueError, "already be frozen"):
            train.validate_checkpoint_phase(ledger, "a" * 64)
        train.validate_checkpoint_phase(ledger, None)
        v = train.GenerationLedger(spec(), completed={"proposer": 2000, "critic": 2000, "validator": 0},
                                   dispatched={"proposer": 2000, "critic": 2000, "validator": 0}, phase="v", next_role="validator")
        with self.assertRaisesRegex(ValueError, "freeze evidence"):
            train.validate_checkpoint_phase(v, None)

    def test_sampler_resume_preserves_cross_epoch_sequence(self):
        records = []
        for i in range(3):
            r = row()
            r["input"]["question_id"] = f"q{i}"
            r["input_sha256"] = data.seal_input(r["input"])
            records.append(r)
        a = train.GenerationSampler(records, "proposer")
        a.next_accumulation()
        state = a.state()
        b = train.GenerationSampler(records, "proposer", state=state)
        self.assertEqual(a.next_accumulation(), b.next_accumulation())
        self.assertEqual(a.state(), b.state())
        state["samples"] += 1
        with self.assertRaisesRegex(ValueError, "cumulative"):
            train.GenerationSampler(records, "proposer", state=state)


if __name__ == "__main__":
    unittest.main()
