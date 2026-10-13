"""Frozen action group shape scaffolding; never actual producer evidence.

Two original inputs and two actions are enough to exercise the contracts.
Synthetic pins do not prove native execution, grant alias admission or fill
real generation quotas. No model/optimizer/export execution is permitted.
"""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from test_generation_action_authority import forbid_imports

_restore = forbid_imports()
try:
    from test_generation_action_authority import authority_fixture, put, reseal
    from test_generation_chunks import write_corpus
    from test_generation_contract import tasks
    from rz_pals_model import generation_chunks as chunks
    from rz_pals_model import generation_data as data
    from rz_pals_model.virtual_review import input_digest
finally:
    _restore()


GROUP_DOMAIN = "rz-pals-generation-frozen-action-group-evidence/1"
GROUP_DIGEST_DOMAIN = "rz-pals-generation-frozen-action-group/1"
SCHEDULE_DOMAIN = "rz-pals-generation-action-enumeration/1"


def seal_group(group):
    group["action_schedule_sha256"] = data.digest(SCHEDULE_DOMAIN, group["registered_actions"])
    group["group_sha256"] = data.digest(GROUP_DIGEST_DOMAIN, [group["semantic_input_sha256"],
        group["frozen_context_sha256"], group["frozen_pc_epoch"], group["frozen_pc_weights_sha256"],
        group["registered_actions"], group["action_schedule_sha256"]])
    return group


def original_action_fixture(index, *, tensor_marker=None, native_model_sha=None, history_sha=None,
                            backend_sha=None, capture_sequence=None):
    """Build one immutable schema input and its own native/causal receipts.

    Request IDs are distinct; submitted model tensor, history, model and V
    support are identical. Labels stay outside the submitted raw record.
    """
    base, submitted, prior, authorities, retirement_sha = authority_fixture()
    snapshot = base["input"]
    snapshot.update(question_id=hashlib.sha256(f"contract-only-action-member-{index}".encode()).hexdigest(),
                    capture_sequence=10 + index if capture_sequence is None else capture_sequence,
                    job_id=f"group-job-{index}", slice_id=f"group-slice-{index}",
                    action_support_mask=[True, True, False, False, False, False, False])
    if tensor_marker is not None:
        snapshot["encoded_model_input"]["anchors"][0]["metadata"][0] = tensor_marker
    outcome = copy.deepcopy(base["future_targets"]["task"]["outcomes"][index])
    base["future_targets"]["task"] = None
    base["input_sha256"] = data.seal_input(snapshot)
    ticket = copy.deepcopy(submitted["ticket"])
    ticket.update(question_sha256=snapshot["question_id"], request_sequence=index + 1)
    if native_model_sha is not None:
        ticket["representation_model_sha256"] = native_model_sha
    if history_sha is not None:
        ticket["representation_history_sha256"] = history_sha
    submitted = copy.deepcopy(submitted)
    submitted.update(input_sha256=base["input_sha256"], ticket=ticket,
                     capture_sequence=snapshot["capture_sequence"], job_id=snapshot["job_id"], slice_id=snapshot["slice_id"])
    submitted["native_input_sha256"] = input_digest(snapshot["encoded_model_input"])
    if native_model_sha is not None:
        submitted["native_model_epoch_sha256"] = native_model_sha
    if history_sha is not None:
        submitted["native_endpoint_history_sha256"] = history_sha
    if backend_sha is not None:
        submitted["native_backend_sha256"] = backend_sha
    receipts = {}
    submitted_sha = put(receipts, submitted)
    old_retired = prior[retirement_sha]
    completed = copy.deepcopy(prior[old_retired["completion_sha256"]])
    completed.update(input_sha256=base["input_sha256"], submission_sha256=submitted_sha,
                     ticket=copy.deepcopy(ticket), execution={"epoch": 0, "sequence": index + 1},
                     invocation_key_sha256=hashlib.sha256(f"contract-only-invocation-{index}".encode()).hexdigest())
    completed_sha = put(receipts, completed)
    retired = copy.deepcopy(old_retired)
    retired.update(input_sha256=base["input_sha256"], submission_sha256=submitted_sha,
                   completion_sha256=completed_sha, ticket=copy.deepcopy(ticket))
    retired_sha = put(receipts, retired)
    outcome["execution_id"] = f"validator-action-0-{index + 1}-{index}"
    for witness in outcome["new_witnesses"]:
        raw = copy.deepcopy(prior[witness["raw_evidence_sha256"]])
        raw["input_sha256"] = base["input_sha256"]
        witness["raw_evidence_sha256"] = put(receipts, raw)
    outcome["causal_receipt_sha256"] = put(receipts, {
        "schema": "rz-pals-generation-causal-evidence/1", "input_sha256": base["input_sha256"],
        "outcome": {key: copy.deepcopy(value) for key, value in outcome.items() if key != "causal_receipt_sha256"}})
    authority = copy.deepcopy(authorities[index])
    authority.update(input_sha256=base["input_sha256"], ticket=copy.deepcopy(ticket),
                     action_support_mask=list(snapshot["action_support_mask"]), execution_id=outcome["execution_id"],
                     causal_receipt_sha256=outcome["causal_receipt_sha256"])
    authority_sha = put(receipts, authority)
    return base, submitted, receipts, outcome, authority, authority_sha, retired_sha


def frozen_group_fixture():
    """Metadata only: original records retain their own native identities."""
    originals = [original_action_fixture(index) for index in (0, 1)]
    registered = [0, 1]
    group = {"schema": GROUP_DOMAIN, "input_sha256": originals[0][0]["input_sha256"],
             "group_sha256": "0" * 64,
             "semantic_input_sha256": chunks.semantic_sample_seal(originals[0][0], originals[0][1]),
             "frozen_context_sha256": "6" * 64, "frozen_pc_epoch": 0,
             "frozen_pc_weights_sha256": "4" * 64, "registered_actions": registered,
             "action_schedule_sha256": "0" * 64,
             "selection_provenance": "registered-supported-enumeration-v1",
             "members": [{"action": data.TASKS[index], "index": index,
                          "actual_input_sha256": original[0]["input_sha256"],
                          "causal_receipt_sha256": original[3]["causal_receipt_sha256"],
                          "action_authority_sha256": original[5]}
                         for index, original in zip(registered, originals)]}
    seal_group(group)
    target = tasks(originals[0][0], [copy.deepcopy(original[3]) for original in originals])
    head = {"schema": chunks.HEAD_EVIDENCE_DOMAIN, "input_sha256": group["input_sha256"], "head": "task",
            "target": {key: copy.deepcopy(value) for key, value in target.items() if key != "raw_evidence_sha256"}}
    target["raw_evidence_sha256"] = data.digest(head["schema"], head)
    return group, originals, target, head


class FrozenActionGroupShapeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(forbid_imports())
        self.group, self.originals, self.target, self.head = frozen_group_fixture()

    def test_exact_shape_and_digests_do_not_mutate_original_inputs_or_receipts(self):
        before_group, before_originals = copy.deepcopy(self.group), copy.deepcopy(self.originals)
        self.assertIs(chunks._frozen_action_group(self.group), self.group)
        self.assertEqual(self.group, before_group)
        self.assertEqual(self.originals, before_originals)
        self.assertEqual(self.group["action_schedule_sha256"], data.digest(SCHEDULE_DOMAIN, [0, 1]))
        self.assertEqual(self.target["target_known_mask"], [True, True, False, False, False, False, False])

    def test_original_records_keep_distinct_fresh_native_and_causal_chains(self):
        semantic = []
        for record, submitted, receipts, outcome, authority, authority_sha, retirement_sha in self.originals:
            self.assertIsNone(record["future_targets"]["task"])
            details = chunks._native_chain(record, receipts, retirement_sha, with_details=True)
            self.assertTrue(details["physical_fresh"])
            semantic.append(details["semantic_sample_sha256"])
            self.assertEqual(receipts[authority_sha], authority)
            self.assertEqual(authority["training_wall_cost_ns"], outcome["wall_cost_ns"])
            self.assertEqual(authority["wall_cost_scope"], chunks.EXCLUSIVE_ACTION_COST_SCOPE)
            self.assertEqual(authority["actual_native_input_sha256"], submitted["ticket"]["representation_input_sha256"])
        self.assertEqual(semantic, [self.group["semantic_input_sha256"]] * 2)
        self.assertNotEqual(self.originals[0][0]["input_sha256"], self.originals[1][0]["input_sha256"])
        self.assertNotEqual(self.originals[0][1]["ticket"], self.originals[1][1]["ticket"])

    def test_group_target_copies_outcomes_without_rebinding_original_proofs(self):
        original = copy.deepcopy(self.originals)
        for copied, member, source in zip(self.target["outcomes"], self.group["members"], self.originals):
            self.assertEqual(copied, source[3])
            self.assertIsNot(copied, source[3])
            self.assertEqual(copied["causal_receipt_sha256"], member["causal_receipt_sha256"])
        self.target["outcomes"][0]["wall_cost_ns"] = 999
        self.assertEqual(self.originals, original)

    def test_required_group_fields_and_unknown_fields_are_strict(self):
        for name in self.group:
            with self.subTest(missing=name):
                value = copy.deepcopy(self.group)
                del value[name]
                with self.assertRaises(ValueError):
                    chunks._frozen_action_group(value)
        value = copy.deepcopy(self.group)
        value["eligible"] = True
        with self.assertRaises(ValueError):
            chunks._frozen_action_group(value)

    def test_domain_and_selection_provenance_aliases_are_rejected(self):
        for name, value in (("schema", "rz-pals-generation-causal-evidence/1"),
                            ("selection_provenance", "posthoc-best-pair"),
                            ("selection_provenance", None)):
            with self.subTest(field=name, value=value):
                group = copy.deepcopy(self.group)
                group[name] = value
                with self.assertRaises(ValueError):
                    chunks._frozen_action_group(group)

    def test_hash_fields_require_canonical_sha256(self):
        names = ("input_sha256", "group_sha256", "semantic_input_sha256", "frozen_context_sha256",
                 "frozen_pc_weights_sha256", "action_schedule_sha256")
        for name in names:
            with self.subTest(field=name):
                group = copy.deepcopy(self.group)
                group[name] = "unknown"
                with self.assertRaises(ValueError):
                    chunks._frozen_action_group(group)

    def test_frozen_epoch_rejects_boolean_float_negative_and_string(self):
        for value in (True, 0.0, -1, "0"):
            with self.subTest(value=value):
                group = copy.deepcopy(self.group)
                group["frozen_pc_epoch"] = value
                with self.assertRaises(ValueError):
                    chunks._frozen_action_group(group)

    def test_registration_requires_ascending_unique_supported_index_shape(self):
        for actions in ([], [1, 0], [0, 0], [True, 1], [0, 7], [0, -1], [0, 1.0], "01", tuple((0, 1))):
            with self.subTest(actions=actions):
                group = copy.deepcopy(self.group)
                group["registered_actions"] = actions
                with self.assertRaises(ValueError):
                    chunks._frozen_action_group(group)

    def test_schedule_pin_and_group_pin_are_independently_required(self):
        for name in ("action_schedule_sha256", "group_sha256"):
            with self.subTest(pin=name):
                group = copy.deepcopy(self.group)
                group[name] = "1" * 64
                with self.assertRaisesRegex(ValueError, "preregistration digest"):
                    chunks._frozen_action_group(group)

    def test_each_frozen_semantic_field_changes_group_digest(self):
        for name, value in (("semantic_input_sha256", "1" * 64), ("frozen_context_sha256", "1" * 64),
                            ("frozen_pc_epoch", 1), ("frozen_pc_weights_sha256", "1" * 64)):
            with self.subTest(field=name):
                group = copy.deepcopy(self.group)
                group[name] = value
                with self.assertRaisesRegex(ValueError, "preregistration digest"):
                    chunks._frozen_action_group(group)
                seal_group(group)
                self.assertNotEqual(group["group_sha256"], self.group["group_sha256"])
                chunks._frozen_action_group(group)  # Shape is not actual proof admission.

    def test_member_count_matches_registered_schedule(self):
        for members in (self.group["members"][:1], self.group["members"] * 2, None, tuple(self.group["members"])):
            with self.subTest(members=members):
                group = copy.deepcopy(self.group)
                group["members"] = members
                with self.assertRaises(ValueError):
                    chunks._frozen_action_group(group)

    def test_member_exact_fields_action_and_index_are_not_aliases(self):
        for changed in ({"action": "attack_repair"}, {"index": 1}, {"index": False},
                        {"index": 0.0}, {"extra": True}):
            with self.subTest(changed=changed):
                group = copy.deepcopy(self.group)
                group["members"][0].update(changed)
                with self.assertRaises(ValueError):
                    chunks._frozen_action_group(group)
        for field in self.group["members"][0]:
            with self.subTest(missing=field):
                group = copy.deepcopy(self.group)
                del group["members"][0][field]
                with self.assertRaises(ValueError):
                    chunks._frozen_action_group(group)

    def test_member_input_causal_and_authority_each_require_unique_canonical_pin(self):
        for name in ("actual_input_sha256", "causal_receipt_sha256", "action_authority_sha256"):
            with self.subTest(field=name):
                group = copy.deepcopy(self.group)
                group["members"][1][name] = group["members"][0][name]
                with self.assertRaisesRegex(ValueError, "duplicate/reordered/renamed"):
                    chunks._frozen_action_group(group)
                group["members"][1][name] = "bad"
                with self.assertRaises(ValueError):
                    chunks._frozen_action_group(group)

    def test_member_order_cannot_be_changed_even_with_valid_group_hash(self):
        group = copy.deepcopy(self.group)
        group["members"].reverse()
        self.assertEqual(group["group_sha256"], self.group["group_sha256"])
        with self.assertRaisesRegex(ValueError, "reordered"):
            chunks._frozen_action_group(group)

    def test_representative_must_be_one_original_member(self):
        group = copy.deepcopy(self.group)
        group["input_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "representative"):
            chunks._frozen_action_group(group)
        group["input_sha256"] = group["members"][1]["actual_input_sha256"]
        chunks._frozen_action_group(group)


class FrozenActionGroupNativeTests(unittest.TestCase):
    """Low-count SQLite contract fixtures, not native collection acceptance."""
    def setUp(self):
        self.addCleanup(forbid_imports())
        self.temporary = tempfile.TemporaryDirectory(prefix="rz-frozen-action-group-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus, self.index_root = self.root / "corpus", self.root / "index"
        self.corpus.mkdir()
        self.index_root.mkdir()
        self.group, self.originals, self.target, self.head = frozen_group_fixture()

    def refresh_head(self):
        self.target["target_known_mask"] = data.derive_task_targets(self.originals[0][0]["input"], self.target["outcomes"])["target_known_mask"]
        self.head["target"] = {key: copy.deepcopy(value) for key, value in self.target.items() if key != "raw_evidence_sha256"}
        self.target["raw_evidence_sha256"] = data.digest(self.head["schema"], self.head)

    def replace_member(self, index, original):
        self.originals[index] = original
        self.group["members"][index].update(actual_input_sha256=original[0]["input_sha256"],
            causal_receipt_sha256=original[3]["causal_receipt_sha256"], action_authority_sha256=original[5])
        self.target["outcomes"][index] = copy.deepcopy(original[3])
        self.refresh_head()

    def reseal_authority(self, index):
        original = list(self.originals[index])
        original[5] = reseal(original[2], original[4])
        self.originals[index] = tuple(original)
        self.group["members"][index]["action_authority_sha256"] = original[5]

    def rewrite_completion(self, index, changes):
        original = list(self.originals[index])
        receipts = original[2]
        retired = receipts[original[6]]
        completed = receipts[retired["completion_sha256"]]
        completed.update(changes)
        retired["completion_sha256"] = reseal(receipts, completed)
        original[6] = reseal(receipts, retired)
        self.originals[index] = tuple(original)

    def manifest(self, *, group_present=True, head_present=True, reverse_records=False, extra_receipts=()):
        receipts = {}
        for original in self.originals:
            receipts.update(original[2])
        if group_present:
            put(receipts, self.group)
        if head_present:
            put(receipts, self.head)
        for receipt in extra_receipts:
            put(receipts, receipt)
        records = [original[0] for original in self.originals]
        if reverse_records:
            records.reverse()
        return write_corpus(self.corpus, records, evidence=receipts, native=False)

    def opened(self, manifest):
        return chunks.ChunkGenerationDataset(self.corpus, manifest,
            expected_manifest_sha256=chunks.validate_manifest(manifest), index_root=self.index_root)

    def test_sql_alias_preserves_all_original_fresh_inputs_and_proofs(self):
        manifest = self.manifest()
        raw = {entry["path"]: (self.corpus / entry["path"]).read_bytes() for entry in manifest["chunks"]}
        with self.opened(manifest) as dataset:
            rows = dataset.rows("train")
            self.assertEqual(len(rows), 1)  # One supervised semantic group, two original publications.
            representative = rows[0]
            self.assertEqual(representative["input"], self.originals[0][0]["input"])
            self.assertEqual(representative["future_targets"]["task"], self.target)
            self.assertEqual(dataset._db.execute("SELECT COUNT(*) FROM raw_records").fetchone()[0], 2)
            self.assertEqual(dataset._db.execute("SELECT COUNT(*) FROM native").fetchone()[0], 6)
            lookup = chunks._Lookup(dataset, "evidence", "receipt")
            self.assertEqual(lookup.action_group(representative["input_sha256"]), self.group)
            for original, member in zip(self.originals, self.group["members"]):
                self.assertEqual(lookup.get(member["action_authority_sha256"]), original[4])
                self.assertEqual(lookup.action_source_input(representative["input_sha256"], original[3]), original[0]["input_sha256"])
                details = chunks._native_chain(original[0], lookup, lookup.accepted_retirement(original[0]["input_sha256"]),
                                              with_details=True, validate_actions=False)
                self.assertTrue(details["physical_fresh"])
                self.assertEqual(details["submission"], original[1])
            with self.assertRaisesRegex(ValueError, "InsufficientValidTargets"):
                dataset.admit("v")
        self.assertEqual(list(self.index_root.iterdir()), [])
        for relative, original_bytes in raw.items():
            self.assertEqual((self.corpus / relative).read_bytes(), original_bytes)

    def test_alias_recovers_member_emitted_after_base_without_whole_corpus_buffer(self):
        with self.opened(self.manifest(reverse_records=True)) as dataset:
            representative = dataset.rows("train")[0]
            self.assertEqual(representative["input_sha256"], self.group["input_sha256"])
            self.assertEqual(representative["future_targets"]["task"], self.target)
            self.assertFalse(hasattr(dataset, "_records"))

    def test_group_absence_cannot_rebind_other_input_causal_receipt(self):
        with self.assertRaisesRegex(ValueError, "recovered causal execution"):
            self.opened(self.manifest(group_present=False))

    def test_duplicate_groups_for_one_representative_are_rejected_with_valid_pins(self):
        duplicate = copy.deepcopy(self.group)
        duplicate["frozen_context_sha256"] = "7" * 64
        seal_group(duplicate)
        chunks._frozen_action_group(duplicate)
        with self.assertRaisesRegex(ValueError, "GenerationIndexCapacityOrIntegrity"):
            self.opened(self.manifest(extra_receipts=(duplicate,)))

    def test_legacy_dictionary_join_does_not_grant_group_alias(self):
        receipts = {}
        for original in self.originals:
            receipts.update(original[2])
        put(receipts, self.group)
        put(receipts, self.head)
        with self.assertRaisesRegex(ValueError, "recovered causal execution"):
            data.GenerationDataset._join_evidence(self.group["input_sha256"], "task", self.target, receipts)

    def test_changed_group_frozen_context_with_valid_group_hash_is_rejected(self):
        self.group["frozen_context_sha256"] = "7" * 64
        seal_group(self.group)
        with self.assertRaisesRegex(ValueError, "private exclusive frozen cost proof"):
            self.opened(self.manifest())

    def test_member_private_context_cannot_differ_from_group(self):
        self.originals[1][4]["frozen_context_sha256"] = "7" * 64
        self.reseal_authority(1)
        with self.assertRaisesRegex(ValueError, "private exclusive frozen cost proof"):
            self.opened(self.manifest())

    def test_group_pc_weights_must_match_registered_producer_model(self):
        self.group["frozen_pc_weights_sha256"] = "7" * 64
        seal_group(self.group)
        with self.assertRaisesRegex(ValueError, "preregistered original semantic input/PC state"):
            self.opened(self.manifest())

    def test_member_pc_weights_cannot_differ_from_frozen_group(self):
        self.originals[1][4]["frozen_pc_weights_sha256"] = "7" * 64
        self.reseal_authority(1)
        with self.assertRaisesRegex(ValueError, "private exclusive frozen cost proof"):
            self.opened(self.manifest())

    def test_group_and_member_pc_epochs_must_match_original_model_epoch(self):
        for location in ("group", "member"):
            with self.subTest(location=location):
                if location == "group":
                    self.group["frozen_pc_epoch"] = 1
                    seal_group(self.group)
                else:
                    self.group["frozen_pc_epoch"] = 0
                    seal_group(self.group)
                    self.originals[1][4]["frozen_pc_epoch"] = 1
                    self.reseal_authority(1)
                with self.assertRaisesRegex(ValueError, "PC state|private exclusive frozen cost proof"):
                    self.opened(self.manifest())

    def test_warm_original_member_cannot_supply_fresh_group_supervision(self):
        self.rewrite_completion(1, {"physical_fresh": False, "accepted_seed_sha256": "e" * 64})
        with self.assertRaisesRegex(ValueError, "changed model/tensor/history/context"):
            self.opened(self.manifest())

    def test_unpublished_member_input_is_rejected_without_receipt_rewriting(self):
        self.group["members"][1]["actual_input_sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            self.opened(self.manifest())

    def test_raw_group_unknown_member_is_rejected_even_without_supervised_head(self):
        self.group["members"][1]["actual_input_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "unpublished original input"):
            self.opened(self.manifest(head_present=False))

    def test_complete_raw_group_without_head_preserves_originals_without_supervision(self):
        with self.opened(self.manifest(head_present=False)) as dataset:
            self.assertEqual(dataset._db.execute("SELECT COUNT(*) FROM raw_records").fetchone()[0], 2)
            self.assertTrue(all(row["future_targets"]["task"] is None for row in dataset.rows("train")))
            self.assertEqual(dataset._db.execute("SELECT SUM(valid_target) FROM records").fetchone()[0], 0)
            with self.assertRaisesRegex(ValueError, "InsufficientValidTargets"):
                dataset.admit("v")

    def test_original_authority_cannot_be_reticketed_for_group_base(self):
        self.originals[1][4]["ticket"] = copy.deepcopy(self.originals[0][1]["ticket"])
        self.reseal_authority(1)
        with self.assertRaisesRegex(ValueError, "original native input/outcome"):
            self.opened(self.manifest())

    def test_rehashed_causal_input_rewrite_cannot_replace_original_member(self):
        original = list(self.originals[1])
        causal = original[2][original[3]["causal_receipt_sha256"]]
        causal["input_sha256"] = self.group["input_sha256"]
        original[3]["causal_receipt_sha256"] = reseal(original[2], causal)
        original[4]["causal_receipt_sha256"] = original[3]["causal_receipt_sha256"]
        original[5] = reseal(original[2], original[4])
        self.replace_member(1, tuple(original))
        with self.assertRaisesRegex(ValueError, "recovered causal execution"):
            self.opened(self.manifest())

    def test_reordered_members_cannot_change_preregistered_execution_order(self):
        self.group["members"].reverse()
        with self.assertRaisesRegex(ValueError, "reordered"):
            self.opened(self.manifest())

    def test_unknown_training_cost_remains_raw_but_cannot_enter_frozen_alias(self):
        authority = self.originals[1][4]
        authority.update(training_wall_cost_ns=None, wall_cost_scope=chunks.RAW_ACTION_COST_SCOPE,
                         frozen_context_sha256=None, frozen_pc_epoch=None, frozen_pc_weights_sha256=None)
        self.reseal_authority(1)
        with self.assertRaisesRegex(ValueError, "private exclusive frozen cost proof"):
            self.opened(self.manifest())

    def test_model_tensor_history_and_backend_differences_are_not_aliases(self):
        changes = ({"tensor_marker": 0.25}, {"native_model_sha": "7" * 64},
                   {"history_sha": "7" * 64}, {"backend_sha": "7" * 64})
        for change in changes:
            with self.subTest(change=change):
                original = original_action_fixture(1, **change)
                details = chunks._native_chain(original[0], original[2], original[6],
                                              with_details=True, validate_actions=False)
                self.assertTrue(details["physical_fresh"])  # Valid original; the group rejects its different meaning.
                self.replace_member(1, original)
                with self.assertRaisesRegex(ValueError, "changed model/tensor/history/context"):
                    self.opened(self.manifest())

    def test_group_head_must_follow_every_original_capture_clock(self):
        self.replace_member(1, original_action_fixture(1, capture_sequence=self.target["observed_sequence"]))
        with self.assertRaisesRegex(ValueError, "capture|chronolog|publication|follow|actual origin"):
            self.opened(self.manifest())

    def test_group_head_cannot_precede_its_witness_clock(self):
        self.target["observed_sequence"] = 19
        self.refresh_head()
        with self.assertRaisesRegex(ValueError, "witness|chronolog|publication"):
            self.opened(self.manifest())


if __name__ == "__main__":
    unittest.main()
