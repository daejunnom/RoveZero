"""Opaque action authority contract checks, not actual collection evidence.

Every fixture has one input and at most two actions. It cannot fill admission
quotas and executes no Rules, neural model, optimizer, training or export.
"""
import builtins
import copy
import hashlib
import importlib.abc
import sys
import tempfile
import unittest
from pathlib import Path


FORBIDDEN_IMPORTS = frozenset(("torch", "numpy", "onnx", "onnxruntime"))


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN_IMPORTS:
            raise AssertionError("neural import forbidden in action authority tests")


def forbid_imports():
    finder, previous = NeuralImportsForbidden(), builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in FORBIDDEN_IMPORTS:
            raise AssertionError("neural import forbidden in action authority tests")
        return previous(name, *args, **kwargs)

    sys.meta_path.insert(0, finder)
    builtins.__import__ = guarded

    def restore():
        builtins.__import__ = previous
        sys.meta_path.remove(finder)

    return restore


_restore = forbid_imports()
try:
    from test_generation_contract import row, tasks
    from test_generation_chunks import add_native_chain, write_corpus
    from rz_pals_model import generation_chunks as chunks
    from rz_pals_model import generation_data as data
finally:
    _restore()


def witness_identity(root_sha, kind, line):
    framing = (b"rz-pals-action-unique-witness/1\0" + bytes.fromhex(root_sha)
               + bytes((0 if kind == "counter" else 1,)))
    return hashlib.sha256(framing + b"".join(move.to_bytes(2, "little") for move in line)).hexdigest()


def put(receipts, receipt):
    seal = data.digest(receipt["schema"], receipt)
    receipts[seal] = receipt
    return seal


def authority_fixture(witness_kind="counter", *, shared_cost=False):
    """One schema-only V input; synthetic hashes are not owner-issued proof."""
    record = row("validator")
    snapshot = record["input"]
    snapshot["question_id"] = hashlib.sha256(b"contract-only-validator-question").hexdigest()
    snapshot["selected_view"]["query"]["purpose"] = "task_policy"
    snapshot["encoded_model_input"]["query"]["purpose"] = "task_policy"
    snapshot["view_sha256"] = data.view_digest(snapshot["selected_view"])
    record["input_sha256"] = data.seal_input(snapshot)
    receipts = {}
    submitted, completed, retired = add_native_chain(record, receipts)
    receipts.clear()
    completed["output"] = {"kind": "task_ranking", "task_logits_f32": ["00000000"] * 7}
    completed["submission_sha256"] = put(receipts, submitted)
    retired["completion_sha256"] = put(receipts, completed)
    retirement_sha = put(receipts, retired)
    outcomes, authorities = [], []
    for index, (action, cost, kind) in enumerate((("defend_response", 10, witness_kind),
                                                 ("attack_repair", 30, None))):
        raw_witnesses, public_witnesses = [], []
        if kind is not None:
            line = [64, 128]
            identity = witness_identity(snapshot["selected_view"]["anchors"][0]["rules_state_sha256"], kind, line)
            public = {"witness_id": "witness-" + identity, "kind": kind, "observed_sequence": 20}
            witness_sha = put(receipts, {"schema": "rz-pals-generation-witness-evidence/1",
                                       "input_sha256": record["input_sha256"], "witness": copy.deepcopy(public),
                                       "rules_validated": True})
            public_witnesses.append(dict(public, raw_evidence_sha256=witness_sha))
            # Store ordinals and collector chronology deliberately differ.
            raw_witnesses.append(dict(public, store_observation_sequence=4, store_generation=0,
                                      packed_line=line, semantic_identity_sha256=identity,
                                      store_commit_sha256="f" * 64))
        execution_id = f"validator-action-0-1-{data.TASKS.index(action)}"
        training_cost = None if shared_cost else cost
        outcome = {"action": action, "execution_id": execution_id, "completed": True,
                   "wall_cost_ns": training_cost, "new_witnesses": public_witnesses}
        causal_sha = put(receipts, {"schema": "rz-pals-generation-causal-evidence/1",
                                  "input_sha256": record["input_sha256"], "outcome": copy.deepcopy(outcome)})
        outcome["causal_receipt_sha256"] = causal_sha
        outcomes.append(outcome)
        authority = {"schema": chunks.ACTION_AUTHORITY_DOMAIN, "input_sha256": record["input_sha256"],
                     "generation_view_sha256": snapshot["view_sha256"], "ticket": copy.deepcopy(submitted["ticket"]),
                     "action": action, "execution_id": execution_id,
                     "native_view_content_sha256": submitted["source_view_content_sha256"],
                     "actual_native_input_sha256": submitted["ticket"]["representation_input_sha256"],
                     "action_support_mask": list(snapshot["action_support_mask"]),
                     "input_store_observation_sequence": 2, "original_first_move": 64,
                     # Measured root and exclusive action have different starts;
                     # neither is required to be larger than the other.
                     "wall_cost_ns": 5 if index == 0 else 90, "training_wall_cost_ns": training_cost,
                     "wall_cost_scope": (chunks.RAW_ACTION_COST_SCOPE if shared_cost else chunks.EXCLUSIVE_ACTION_COST_SCOPE),
                     "measured_wall_cost_scope": chunks.RAW_ACTION_COST_SCOPE,
                     "frozen_context_sha256": None if shared_cost else "6" * 64,
                     "frozen_pc_epoch": None if shared_cost else snapshot["model_epoch"],
                     "frozen_pc_weights_sha256": None if shared_cost else "4" * 64,
                     "root_native_cpu_spent": 16, "root_completed_nn_inputs": 2,
                     "executions": [{"epoch": 0, "sequence": 11 + index}], "role_requests": [],
                     "witnesses": raw_witnesses, "causal_receipt_sha256": causal_sha}
        put(receipts, authority)
        authorities.append(authority)
    target = tasks(record, outcomes)
    target["raw_evidence_sha256"] = put(receipts, {"schema": chunks.HEAD_EVIDENCE_DOMAIN,
        "input_sha256": record["input_sha256"], "head": "task",
        "target": {key: copy.deepcopy(value) for key, value in target.items() if key != "raw_evidence_sha256"}})
    record["future_targets"]["task"] = target
    return record, submitted, receipts, authorities, retirement_sha


def reseal(receipts, receipt):
    """Replace just a synthetic receipt pin after a deliberate negative edit."""
    for key, value in list(receipts.items()):
        if value is receipt:
            del receipts[key]
    return put(receipts, receipt)


def replace_task_head(record, receipts):
    target = record["future_targets"]["task"]
    del receipts[target["raw_evidence_sha256"]]
    target["raw_evidence_sha256"] = put(receipts, {"schema": chunks.HEAD_EVIDENCE_DOMAIN,
        "input_sha256": record["input_sha256"], "head": "task",
        "target": {key: copy.deepcopy(value) for key, value in target.items() if key != "raw_evidence_sha256"}})


class ActionAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(forbid_imports())
        self.temporary = tempfile.TemporaryDirectory(prefix="rz-action-authority-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus, self.index_root = self.root / "corpus", self.root / "index"
        self.corpus.mkdir()
        self.index_root.mkdir()
        self.record, self.submitted, self.receipts, self.authorities, self.retirement_sha = authority_fixture()

    def check_join(self):
        chunks._native_action_authorities(self.record, self.submitted, self.receipts)

    def opened(self, manifest):
        return chunks.ChunkGenerationDataset(self.corpus, manifest,
            expected_manifest_sha256=chunks.validate_manifest(manifest), index_root=self.index_root)

    def corpus_manifest(self, *, late=False):
        record = copy.deepcopy(self.record)
        if late:
            record["future_targets"]["task"] = None
        return write_corpus(self.corpus, [record], evidence=self.receipts, native=False)

    def test_native_three_chain_and_all_head_proofs_join(self):
        self.assertNotEqual(self.submitted["native_input_sha256"], self.submitted["ticket"]["representation_input_sha256"])
        self.assertEqual(data.validate_record(self.record)["task"]["pairs"], [(0, 1)])
        chunks._native_chain(self.record, self.receipts, self.retirement_sha)
        for authority in self.authorities:
            self.assertIs(chunks._action_authority(authority), authority)

    def test_counter_and_repair_identity_fixed_framing_vectors(self):
        expected = {"counter": "54c2998008b1f2cf2de0e82bdf6844001f488852beb951e5c516b9fb0d33c0ad",
                    "repair": "12d21cddcfdc00ed3435ca8bc9f2f7c992f828e35103a4542faffdf21f635200"}
        for kind, seal in expected.items():
            with self.subTest(kind=kind):
                self.assertEqual(witness_identity("a" * 64, kind, [64, 128]), seal)
                record, submitted, receipts, _, retired = authority_fixture(kind)
                chunks._native_chain(record, receipts, retired)
                self.assertEqual(record["future_targets"]["task"]["outcomes"][0]["new_witnesses"][0]["witness_id"],
                                 "witness-" + seal)

    def test_store_ordinal_is_not_collector_clock_and_equal_watermark_is_valid(self):
        authority = self.authorities[0]
        self.assertLess(authority["witnesses"][0]["store_observation_sequence"], self.record["input"]["capture_sequence"])
        self.check_join()
        authority["witnesses"][0]["store_observation_sequence"] = authority["input_store_observation_sequence"]
        self.check_join()

    def test_sql_authority_lookup_and_late_task_head_hydrate_one_row(self):
        manifest = self.corpus_manifest(late=True)
        raw_record = (self.corpus / manifest["chunks"][-1]["path"]).read_bytes()
        with self.opened(manifest) as dataset:
            decoded = dataset.rows("train")[0]
            self.assertEqual(decoded, self.record)
            self.assertEqual(list(dataset.rows("train")), [self.record])
            self.assertEqual(dataset._db.execute("SELECT COUNT(*) FROM action_authority").fetchone()[0], 2)
            lookup = chunks._Lookup(dataset, "evidence", "receipt")
            self.assertEqual(lookup.action_authority(self.authorities[0]["causal_receipt_sha256"]), self.authorities[0])
            with self.assertRaisesRegex(ValueError, "InsufficientValidTargets"):
                dataset.admit("v")
        self.assertEqual((self.corpus / manifest["chunks"][-1]["path"]).read_bytes(), raw_record)
        self.assertEqual(list(self.index_root.iterdir()), [])

    def test_sql_missing_authority_refused_even_with_causal_and_witness_receipts(self):
        del self.receipts[data.digest(chunks.ACTION_AUTHORITY_DOMAIN, self.authorities[0])]
        with self.assertRaisesRegex(ValueError, "opaque action authority"):
            self.opened(self.corpus_manifest(late=True))
        self.assertEqual(list(self.index_root.iterdir()), [])

    def test_sql_duplicate_causal_authority_refused(self):
        duplicate = copy.deepcopy(self.authorities[0])
        duplicate["root_native_cpu_spent"] += 1
        put(self.receipts, duplicate)
        with self.assertRaisesRegex(ValueError, "GenerationIndexCapacityOrIntegrity"):
            self.opened(self.corpus_manifest())
        self.assertEqual(list(self.index_root.iterdir()), [])

    def test_sql_duplicate_late_head_refused_with_two_valid_receipt_pins(self):
        target = self.record["future_targets"]["task"]
        duplicate = copy.deepcopy(self.receipts[target["raw_evidence_sha256"]])
        duplicate["target"]["observed_sequence"] += 1
        put(self.receipts, duplicate)
        with self.assertRaisesRegex(ValueError, "GenerationIndexCapacityOrIntegrity"):
            self.opened(self.corpus_manifest(late=True))

    def test_unaccepted_native_retirement_cannot_authorize_action(self):
        retirement = self.receipts[self.retirement_sha]
        retirement["accepted"] = False
        self.retirement_sha = reseal(self.receipts, retirement)
        with self.assertRaisesRegex(ValueError, "accepted confirmed physical NN evidence"):
            chunks._native_chain(self.record, self.receipts, self.retirement_sha)

    def test_rehashed_completion_with_different_ticket_is_not_original_work(self):
        retired = self.receipts[self.retirement_sha]
        completed = copy.deepcopy(self.receipts[retired["completion_sha256"]])
        completed["ticket"]["request_sequence"] += 1
        retired["completion_sha256"] = put(self.receipts, completed)
        self.retirement_sha = reseal(self.receipts, retired)
        with self.assertRaisesRegex(ValueError, "ticket"):
            chunks._native_chain(self.record, self.receipts, self.retirement_sha)

    def test_authority_absent_or_ambiguous_in_bounded_dictionary(self):
        authority = self.authorities[0]
        original = self.receipts.pop(data.digest(chunks.ACTION_AUTHORITY_DOMAIN, authority))
        with self.assertRaisesRegex(ValueError, "exactly one opaque"):
            self.check_join()
        put(self.receipts, original)
        duplicate = copy.deepcopy(original)
        duplicate["wall_cost_ns"] += 1
        put(self.receipts, duplicate)
        with self.assertRaisesRegex(ValueError, "exactly one opaque"):
            self.check_join()

    def test_changed_ticket_fields_do_not_authorize_original_input(self):
        for name, changed in (("request_sequence", 2), ("root", 2), ("question_sha256", "1" * 64),
                              ("representation_history_sha256", "1" * 64)):
            with self.subTest(field=name):
                authority = self.authorities[0]
                previous = authority["ticket"][name]
                authority["ticket"][name] = changed
                with self.assertRaisesRegex(ValueError, "original native input/outcome"):
                    self.check_join()
                authority["ticket"][name] = previous

    def test_wrong_input_view_and_native_view_content_refused(self):
        for name in ("input_sha256", "generation_view_sha256", "native_view_content_sha256"):
            with self.subTest(field=name):
                authority = self.authorities[0]
                previous = authority[name]
                authority[name] = "1" * 64
                with self.assertRaisesRegex(ValueError, "original native input/outcome"):
                    self.check_join()
                authority[name] = previous

    def test_actual_native_input_cannot_alias_raw_tensor_submission_digest(self):
        self.authorities[0]["actual_native_input_sha256"] = self.submitted["native_input_sha256"]
        with self.assertRaisesRegex(ValueError, "original native input/outcome"):
            self.check_join()

    def test_mask_mismatch_and_unsupported_selected_action_refused(self):
        self.authorities[0]["action_support_mask"][6] = False
        with self.assertRaisesRegex(ValueError, "original native input/outcome"):
            self.check_join()
        self.authorities[0]["action_support_mask"][0] = False
        with self.assertRaisesRegex(ValueError, "supported action mask"):
            self.check_join()

    def test_execution_id_must_bind_epoch_request_and_action(self):
        for value in ("arbitrary-action", "validator-action-0-2-0", "validator-action-0-1-1"):
            with self.subTest(value=value):
                self.authorities[0]["execution_id"] = value
                self.record["future_targets"]["task"]["outcomes"][0]["execution_id"] = value
                with self.assertRaisesRegex(ValueError, "original native input/outcome"):
                    self.check_join()

    def test_training_action_wall_cost_must_match_causal_outcome(self):
        self.authorities[0]["training_wall_cost_ns"] = 1
        with self.assertRaisesRegex(ValueError, "original native input/outcome"):
            self.check_join()

    def test_raw_root_cost_and_exclusive_training_cost_can_differ_in_either_direction(self):
        self.assertLess(self.authorities[0]["wall_cost_ns"], self.authorities[0]["training_wall_cost_ns"])
        self.assertGreater(self.authorities[1]["wall_cost_ns"], self.authorities[1]["training_wall_cost_ns"])
        self.check_join()
        with self.opened(self.corpus_manifest(late=True)) as dataset:
            self.assertEqual(dataset.rows("train")[0], self.record)

    def test_shared_unknown_cost_preserves_raw_witnesses_without_training_pairs(self):
        self.record, self.submitted, self.receipts, self.authorities, self.retirement_sha = authority_fixture(shared_cost=True)
        known = data.validate_record(self.record)["task"]
        self.assertEqual(known["target_known_mask"], [False] * 7)
        self.assertEqual(known["pairs"], [])
        self.assertEqual(len(self.record["future_targets"]["task"]["outcomes"][0]["new_witnesses"]), 1)
        for authority, outcome in zip(self.authorities, self.record["future_targets"]["task"]["outcomes"]):
            self.assertIsNone(outcome["wall_cost_ns"])
            self.assertIsNone(authority["training_wall_cost_ns"])
            self.assertGreater(authority["wall_cost_ns"], 0)
            self.assertEqual(authority["wall_cost_scope"], "input_prepare_to_actual_root_physical_fence")
        chunks._native_chain(self.record, self.receipts, self.retirement_sha)
        with self.opened(self.corpus_manifest(late=True)) as dataset:
            self.assertEqual(dataset.rows("train")[0], self.record)
            self.assertEqual(list(dataset.rows("train")), [self.record])
            self.assertEqual(dataset._db.execute("SELECT valid_target FROM records").fetchone()[0], 0)
            with self.assertRaisesRegex(ValueError, "InsufficientValidTargets"):
                dataset.admit("v")

    def test_unknown_cost_action_duplicate_is_rejected_before_supervision_filtering(self):
        first = copy.deepcopy(self.record["future_targets"]["task"]["outcomes"][0])
        first["wall_cost_ns"] = None
        baseline = data.derive_task_targets(self.record["input"], [first])
        self.assertEqual(baseline["pairs"], [])
        self.assertEqual(baseline["target_known_mask"], [False] * 7)
        for next_cost in (None, 10):
            with self.subTest(second_cost=next_cost):
                duplicate = copy.deepcopy(first)
                duplicate["wall_cost_ns"] = next_cost
                observed = [first, duplicate]
                before = copy.deepcopy(observed)
                with self.assertRaisesRegex(ValueError, "duplicate action target"):
                    data.derive_task_targets(self.record["input"], observed)
                self.assertEqual(observed, before)

    def test_incomplete_action_duplicate_is_rejected_before_supervision_filtering(self):
        first = copy.deepcopy(self.record["future_targets"]["task"]["outcomes"][0])
        first["completed"] = False
        baseline = data.derive_task_targets(self.record["input"], [first])
        self.assertEqual(baseline["pairs"], [])
        self.assertEqual(baseline["target_known_mask"], [False] * 7)
        for next_completed in (False, True):
            with self.subTest(second_completed=next_completed):
                duplicate = copy.deepcopy(first)
                duplicate["completed"] = next_completed
                observed = [first, duplicate]
                before = copy.deepcopy(observed)
                with self.assertRaisesRegex(ValueError, "duplicate action target"):
                    data.derive_task_targets(self.record["input"], observed)
                self.assertEqual(observed, before)

    def test_unknown_cost_cannot_alias_known_causal_cost_or_reverse(self):
        authority = self.authorities[0]
        authority["training_wall_cost_ns"] = None
        authority["wall_cost_scope"] = chunks.RAW_ACTION_COST_SCOPE
        with self.assertRaisesRegex(ValueError, "original native input/outcome"):
            self.check_join()
        authority["training_wall_cost_ns"] = 10
        authority["wall_cost_scope"] = chunks.EXCLUSIVE_ACTION_COST_SCOPE
        self.record["future_targets"]["task"]["outcomes"][0]["wall_cost_ns"] = None
        with self.assertRaisesRegex(ValueError, "original native input/outcome"):
            self.check_join()

    def test_nullable_training_cost_requires_exact_integer_type(self):
        for cost in (True, False, -1, 1.0, "10", {}, []):
            with self.subTest(cost=cost):
                authority = copy.deepcopy(self.authorities[0])
                authority["training_wall_cost_ns"] = cost
                with self.assertRaises(ValueError):
                    chunks._action_authority(authority)

    def test_training_and_measured_scope_must_match_their_private_semantics(self):
        mutations = (
            {"wall_cost_scope": chunks.RAW_ACTION_COST_SCOPE},
            {"training_wall_cost_ns": None},
            {"training_wall_cost_ns": None, "wall_cost_scope": "unknown"},
            {"training_wall_cost_ns": None, "wall_cost_scope": chunks.EXCLUSIVE_ACTION_COST_SCOPE},
            {"measured_wall_cost_scope": chunks.EXCLUSIVE_ACTION_COST_SCOPE},
            {"measured_wall_cost_scope": None},
        )
        for changed in mutations:
            with self.subTest(changed=changed):
                authority = copy.deepcopy(self.authorities[0])
                authority.update(changed)
                with self.assertRaisesRegex(ValueError, "private exclusive scope"):
                    chunks._action_authority(authority)

    def test_known_cost_requires_complete_private_frozen_context_and_pc_state(self):
        for name in ("frozen_context_sha256", "frozen_pc_epoch", "frozen_pc_weights_sha256"):
            with self.subTest(field=name):
                authority = copy.deepcopy(self.authorities[0])
                authority[name] = None
                with self.assertRaisesRegex(ValueError, "complete epoch/weights|private frozen context"):
                    chunks._action_authority(authority)

    def test_nullable_pc_pair_and_frozen_epoch_reject_aliases(self):
        for changed in ({"frozen_pc_epoch": None}, {"frozen_pc_weights_sha256": None},
                        {"frozen_pc_epoch": True}, {"frozen_pc_epoch": 0.0},
                        {"frozen_pc_epoch": -1}, {"frozen_pc_weights_sha256": "unknown"},
                        {"frozen_context_sha256": "unknown"}):
            with self.subTest(changed=changed):
                authority = copy.deepcopy(self.authorities[0])
                authority.update(changed)
                with self.assertRaises(ValueError):
                    chunks._action_authority(authority)

    def test_known_cost_pc_epoch_must_match_original_input_model_epoch(self):
        self.authorities[0]["frozen_pc_epoch"] = 1
        with self.assertRaisesRegex(ValueError, "frozen epoch differs"):
            self.check_join()

    def test_same_input_known_outcomes_cannot_compare_different_private_contexts(self):
        self.authorities[1]["frozen_context_sha256"] = "7" * 64
        with self.assertRaisesRegex(ValueError, "private frozen comparison context"):
            self.check_join()

    def test_same_input_known_outcomes_cannot_compare_different_pc_weights(self):
        self.authorities[1]["frozen_pc_weights_sha256"] = "7" * 64
        with self.assertRaisesRegex(ValueError, "private frozen comparison context"):
            self.check_join()

    def test_same_input_pc_weights_must_match_actual_registered_producer(self):
        for authority in self.authorities:
            authority["frozen_pc_weights_sha256"] = "7" * 64
            reseal(self.receipts, authority)
        with self.assertRaisesRegex(ValueError, "private PC weights differ"):
            self.opened(self.corpus_manifest())

    def test_unknown_outcome_keeps_raw_evidence_without_entering_frozen_comparison(self):
        target = self.record["future_targets"]["task"]
        outcome = target["outcomes"][1]
        outcome["wall_cost_ns"] = None
        causal = self.receipts[outcome["causal_receipt_sha256"]]
        causal["outcome"]["wall_cost_ns"] = None
        outcome["causal_receipt_sha256"] = reseal(self.receipts, causal)
        authority = self.authorities[1]
        authority.update(training_wall_cost_ns=None, wall_cost_scope=chunks.RAW_ACTION_COST_SCOPE,
                         frozen_context_sha256=None, frozen_pc_epoch=None, frozen_pc_weights_sha256=None,
                         causal_receipt_sha256=outcome["causal_receipt_sha256"])
        reseal(self.receipts, authority)
        target["target_known_mask"] = data.derive_task_targets(self.record["input"], target["outcomes"])["target_known_mask"]
        replace_task_head(self.record, self.receipts)
        self.check_join()
        self.assertEqual(data.validate_record(self.record)["task"]["pairs"], [])
        with self.opened(self.corpus_manifest(late=True)) as dataset:
            self.assertEqual(dataset.rows("train")[0], self.record)
            with self.assertRaisesRegex(ValueError, "InsufficientValidTargets"):
                dataset.admit("v")

    def test_uncompleted_action_cannot_use_completed_authority(self):
        self.record["future_targets"]["task"]["outcomes"][0]["completed"] = False
        with self.assertRaisesRegex(ValueError, "original native input/outcome"):
            self.check_join()

    def test_old_store_witness_refused_before_semantic_join(self):
        self.authorities[0]["witnesses"][0]["store_observation_sequence"] = 1
        with self.assertRaisesRegex(ValueError, "old/duplicate/misidentified"):
            self.check_join()

    def test_chronology_cannot_be_replaced_by_store_ordinal(self):
        self.authorities[0]["witnesses"][0]["observed_sequence"] = 4
        with self.assertRaisesRegex(ValueError, "chronological publication"):
            self.check_join()

    def test_semantic_identity_binds_full_line_kind_and_exact_root(self):
        witness = self.authorities[0]["witnesses"][0]
        for field, changed in (("packed_line", [64, 192]), ("kind", "repair")):
            with self.subTest(field=field):
                previous = witness[field]
                witness[field] = changed
                if field == "kind":
                    self.record["future_targets"]["task"]["outcomes"][0]["new_witnesses"][0]["kind"] = changed
                with self.assertRaisesRegex(ValueError, "exact root/kind/full line"):
                    self.check_join()
                witness[field] = previous
                self.record["future_targets"]["task"]["outcomes"][0]["new_witnesses"][0]["kind"] = "counter"
        self.record["input"]["selected_view"]["anchors"][0]["rules_state_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "exact root/kind/full line"):
            self.check_join()

    def test_forged_semantic_hash_with_consistent_public_id_still_refused(self):
        raw = self.authorities[0]["witnesses"][0]
        raw["semantic_identity_sha256"] = "1" * 64
        raw["witness_id"] = "witness-" + "1" * 64
        self.record["future_targets"]["task"]["outcomes"][0]["new_witnesses"][0]["witness_id"] = raw["witness_id"]
        with self.assertRaisesRegex(ValueError, "exact root/kind/full line"):
            self.check_join()

    def test_arbitrary_witness_id_and_changed_first_move_refused(self):
        witness = self.authorities[0]["witnesses"][0]
        witness["witness_id"] = "arbitrary-id"
        with self.assertRaisesRegex(ValueError, "misidentified"):
            self.check_join()
        witness["witness_id"] = "witness-" + witness["semantic_identity_sha256"]
        witness["packed_line"][0] = 128
        with self.assertRaisesRegex(ValueError, "misidentified"):
            self.check_join()

    def test_original_first_move_cannot_change_without_witness_line(self):
        self.authorities[0]["original_first_move"] = 128
        with self.assertRaisesRegex(ValueError, "misidentified"):
            self.check_join()

    def test_repeated_semantic_identity_within_action_refused(self):
        self.authorities[0]["witnesses"].append(copy.deepcopy(self.authorities[0]["witnesses"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check_join()

    def test_public_witness_set_cannot_add_or_remove_opaque_witness(self):
        self.record["future_targets"]["task"]["outcomes"][0]["new_witnesses"] = []
        with self.assertRaisesRegex(ValueError, "changed its unique witness set"):
            self.check_join()

    def test_non_defer_requires_actual_execution_identity(self):
        authority = self.authorities[0]
        authority["executions"] = []
        authority["role_requests"] = []
        authority["root_native_cpu_spent"] = authority["root_completed_nn_inputs"] = 0
        with self.assertRaisesRegex(ValueError, "actual execution work"):
            self.check_join()

    def test_role_execution_can_supply_work_without_native_cpu_execution(self):
        authority = self.authorities[0]
        authority["executions"] = []
        authority["role_requests"] = [{"epoch": 0, "sequence": 15}]
        self.check_join()

    def test_duplicate_foreign_and_excessive_execution_identities_refused(self):
        for identities in (([{"epoch": 0, "sequence": 1}] * 2),
                           [{"epoch": 1, "sequence": 1}],
                           [{"epoch": 0, "sequence": i} for i in range(4097)]):
            with self.subTest(size=len(identities)):
                self.authorities[0]["executions"] = identities
                with self.assertRaises(ValueError):
                    self.check_join()

    def test_execution_and_role_request_combined_capacity_is_bounded(self):
        authority = self.authorities[0]
        authority["executions"] = [{"epoch": 0, "sequence": i} for i in range(4096)]
        authority["role_requests"] = [{"epoch": 0, "sequence": 5000}]
        with self.assertRaisesRegex(ValueError, "bounded actual execution work"):
            self.check_join()

    def test_defer_can_be_witness_free_but_cannot_claim_new_witness(self):
        authority = copy.deepcopy(self.authorities[0])
        authority.update(action="defer", execution_id="validator-action-0-1-6", executions=[], role_requests=[])
        with self.assertRaisesRegex(ValueError, "witness-free Defer"):
            chunks._action_authority(authority)
        authority["witnesses"] = []
        chunks._action_authority(authority)

    def test_nonvalidator_input_cannot_consume_task_authority(self):
        self.record["input"]["role"] = "proposer"
        with self.assertRaisesRegex(ValueError, "original native input/outcome"):
            self.check_join()

    def test_strict_authority_fields_types_and_domain(self):
        for mutate in (lambda a: a.update(extra=True), lambda a: a.pop("root_native_cpu_spent"),
                       lambda a: a.update(schema="rz-pals-generation-causal-evidence/1"),
                       lambda a: a.update(wall_cost_ns=True),
                       lambda a: a.update(input_store_observation_sequence=-1),
                       lambda a: a.update(root_completed_nn_inputs=1.0),
                       lambda a: a.update(action_support_mask=[1] * 7),
                       lambda a: a["witnesses"][0].update(store_generation=True),
                       lambda a: a["witnesses"][0].update(store_commit_sha256="unknown")):
            with self.subTest(mutation=mutate):
                authority = copy.deepcopy(self.authorities[0])
                mutate(authority)
                with self.assertRaises(ValueError):
                    chunks._action_authority(authority)

    def test_sql_semantic_forgery_with_valid_receipt_hash_is_rejected(self):
        self.authorities[0]["witnesses"][0]["packed_line"][1] = 192
        reseal(self.receipts, self.authorities[0])
        with self.assertRaisesRegex(ValueError, "exact root/kind/full line"):
            self.opened(self.corpus_manifest(late=True))

    def test_sql_raw_receipt_hash_mismatch_is_rejected_before_authority_join(self):
        self.authorities[0]["wall_cost_ns"] = 1
        with self.assertRaisesRegex(ValueError, "digest|seal|hash"):
            self.opened(self.corpus_manifest())

    def test_task_head_causal_and_rules_witness_receipts_are_all_required(self):
        target = self.record["future_targets"]["task"]
        outcome = target["outcomes"][0]
        pins = (target["raw_evidence_sha256"], outcome["causal_receipt_sha256"],
                outcome["new_witnesses"][0]["raw_evidence_sha256"])
        for seal in pins:
            with self.subTest(removed=seal):
                receipt = self.receipts.pop(seal)
                with self.assertRaisesRegex(ValueError, "recovered|evidence"):
                    self.opened(self.corpus_manifest())
                self.receipts[seal] = receipt

    def test_rehashed_causal_receipt_cannot_change_full_wall_cost(self):
        target = self.record["future_targets"]["task"]
        outcome = target["outcomes"][0]
        causal = self.receipts[outcome["causal_receipt_sha256"]]
        causal["outcome"]["wall_cost_ns"] = 1
        outcome["causal_receipt_sha256"] = reseal(self.receipts, causal)
        self.authorities[0]["causal_receipt_sha256"] = outcome["causal_receipt_sha256"]
        reseal(self.receipts, self.authorities[0])
        replace_task_head(self.record, self.receipts)
        with self.assertRaisesRegex(ValueError, "recovered causal execution evidence"):
            self.opened(self.corpus_manifest(late=True))

    def test_rehashed_witness_receipt_cannot_remove_rules_validation(self):
        outcome = self.record["future_targets"]["task"]["outcomes"][0]
        public = outcome["new_witnesses"][0]
        witness = self.receipts[public["raw_evidence_sha256"]]
        witness["rules_validated"] = False
        public["raw_evidence_sha256"] = reseal(self.receipts, witness)
        causal = self.receipts[outcome["causal_receipt_sha256"]]
        causal["outcome"]["new_witnesses"] = copy.deepcopy(outcome["new_witnesses"])
        outcome["causal_receipt_sha256"] = reseal(self.receipts, causal)
        self.authorities[0]["causal_receipt_sha256"] = outcome["causal_receipt_sha256"]
        reseal(self.receipts, self.authorities[0])
        replace_task_head(self.record, self.receipts)
        with self.assertRaisesRegex(ValueError, "recovered Rules validation evidence"):
            self.opened(self.corpus_manifest(late=True))

    def test_late_head_conflict_cannot_replace_published_action_outcome(self):
        self.record["future_targets"]["task"]["outcomes"][0]["wall_cost_ns"] = 1
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.opened(self.corpus_manifest())

    def test_native_validator_output_rejects_wrong_head_width_and_nonfinite(self):
        output = {"kind": "task_ranking", "task_logits_f32": ["00000000"] * 7}
        chunks._native_output(self.record["input"], output)
        for invalid in ({"kind": "proposal", "task_logits_f32": ["00000000"] * 7},
                        {"kind": "task_ranking", "task_logits_f32": ["00000000"] * 6},
                        {"kind": "task_ranking", "task_logits_f32": ["7fc00000"] * 7}):
            with self.subTest(output=invalid):
                with self.assertRaises(ValueError):
                    chunks._native_output(self.record["input"], invalid)

    def test_import_guard_blocks_frameworks_even_if_cached(self):
        for name in FORBIDDEN_IMPORTS:
            with self.subTest(module=name):
                with self.assertRaisesRegex(AssertionError, "neural import forbidden"):
                    builtins.__import__(name)


if __name__ == "__main__":
    unittest.main()
