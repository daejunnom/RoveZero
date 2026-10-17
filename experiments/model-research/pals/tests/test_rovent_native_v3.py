"""Synthetic submitted-review wire only; no Rules/native/neural authority."""
import copy
import importlib.abc
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("raw /3 contracts cannot import a neural runtime")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import generation_data as data
    from rz_pals_model import rovent_native_v2 as legacy
    from rz_pals_model import rovent_native_v3 as submitted
    from rz_pals_model import rovent_submitted_input as inputs
    from test_rovent_historical_corpus import synthetic_v2, h
    from test_rovent_submitted_input import fixture as input_fixture
finally:
    sys.meta_path.remove(_guard)


def metadata_fixture():
    return {"local_index": 0, "record_revision": 23, "parent_revision": None,
        "supersedes_revision": None, "origin_state": 17, "source_state": 18,
        "source_revision": 23, "source_line_offset": 0, "original_first_move": 192}


def native_fixture():
    """Construct a new synthetic wire; never convert a real legacy corpus."""
    _recipe, manifest, record, _lock, _binding = synthetic_v2()
    manifest["schema"] = "rz-rovent-corpus-manifest/2"
    record["schema"] = "rz-rovent-training-record/2"
    snapshot = record["input"]
    snapshot.pop("divergence_record_ids")
    snapshot.update(schema=inputs.INPUT_DOMAIN, attack_mapping=[], action_support_mask=[False] * 7)
    record["input_sha256"] = inputs.seal_input(snapshot)
    native = record["native"]
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        native[name].update(schema=submitted.NATIVE_DOMAIN, input_sha256=record["input_sha256"])
    ticket = native["raw_submission"]["ticket"]
    native["raw_submission"]["submitted_review"] = {"schema": submitted.REVIEW_DOMAIN,
        "sequence": ticket["view_sequence"], "generation": native["root_generation"],
        "root_state": native["raw_retirement"]["root_admission"]["store_root_state"],
        "policy": {"mode": "anchored_compact", "token_budget": snapshot["selected_view"]["token_budget"],
            "max_records": 64, "max_anchors": 4, "required_bundle": None},
        "input_sha256": h("ReviewView semantic input distinct from numeric native input"),
        "view_seal_sha256": ticket["view_digest"], "record_scope_sha256": h("record scope"),
        "source_anchors": [], "record_metadata": []}
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        if name != "raw_submission":
            native[name]["submitted_sha256"] = native["raw_submission_sha256"]
        if name == "raw_retirement":
            native[name]["completed_sha256"] = native["raw_completion_sha256"]
        native[name + "_sha256"] = data.digest(submitted.NATIVE_DOMAIN, native[name])
    return manifest, record


class SubmittedNativeTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_new_raw_wire_keeps_numeric_and_semantic_view_identities_separate(self):
        manifest, record = native_fixture()
        before = copy.deepcopy(record)
        self.assertIs(submitted.validate_native_v3(manifest, record), record["native"])
        self.assertEqual(record, before)
        review = record["native"]["raw_submission"]["submitted_review"]
        self.assertNotEqual(review["input_sha256"], record["native"]["input_digest"])

    def test_new_and_legacy_raw_parser_reject_each_others_domains(self):
        manifest, record = native_fixture()
        with self.assertRaises(ValueError):
            legacy.validate_native_v2(manifest, record)
        _recipe, old_manifest, old_record, _lock, _binding = synthetic_v2()
        before = copy.deepcopy(old_record)
        legacy.validate_native_v2(old_manifest, old_record)
        with self.assertRaises(ValueError):
            submitted.validate_native_v3(old_manifest, old_record)
        self.assertEqual(old_record, before)

    def test_mixed_original_raw_versions_are_not_retagged(self):
        manifest, record = native_fixture()
        for name in ("raw_submission", "raw_completion", "raw_retirement"):
            bad = copy.deepcopy(record)
            bad["native"][name]["schema"] = legacy.NATIVE_DOMAIN
            with self.subTest(name=name), self.assertRaises(ValueError):
                submitted.validate_native_v3(manifest, bad)

    def test_submitted_review_sequence_and_view_seal_join_the_original_ticket(self):
        _manifest, record = native_fixture()
        value = record["native"]["raw_submission"]["submitted_review"]
        for change in ({"sequence": value["sequence"] + 1}, {"view_seal_sha256": h("other view")}):
            with self.subTest(change=tuple(change)), self.assertRaisesRegex(ValueError, "frozen ticket"):
                submitted.validate_submitted_review(record, {**value, **change})

    def test_submitted_review_shape_policy_and_source_coverage_are_closed(self):
        _manifest, record = native_fixture()
        value = record["native"]["raw_submission"]["submitted_review"]
        for mutate in (lambda item: item.update(extra=True),
                       lambda item: item["policy"].update(max_records=0),
                       lambda item: item["policy"].update(mode="current_root_only", max_anchors=4),
                       lambda item: item.update(source_anchors=[{}])):
            bad = copy.deepcopy(value)
            mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                submitted.validate_submitted_review(record, bad)

    def test_selected_counter_revision_uses_original_metadata_not_record_id_text(self):
        snapshot = input_fixture(selected_counter=True)
        metadata = [metadata_fixture()]
        before = copy.deepcopy(snapshot)
        submitted.validate_submitted_record_metadata(snapshot, metadata)
        self.assertEqual(snapshot, before)
        snapshot["selected_view"]["records"][0]["record_id"] = "arbitrary-record-id-without-a-revision"
        snapshot["view_sha256"] = data.view_digest(snapshot["selected_view"])
        submitted.validate_submitted_record_metadata(snapshot, metadata)
        with self.assertRaisesRegex(ValueError, "submitted source"):
            submitted.validate_submitted_record_metadata(snapshot, [{**metadata[0], "source_revision": 24}])

    def test_selected_metadata_order_shape_and_original_suffix_are_bounded(self):
        snapshot = input_fixture(selected_counter=True)
        for change in ({"local_index": 1}, {"source_line_offset": 256},
                       {"record_revision": True}, {"original_first_move": 0}, {"future_counter": True}):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                submitted.validate_submitted_record_metadata(snapshot, [{**metadata_fixture(), **change}])

    def test_future_unselected_counter_is_not_inserted_into_original_mapping(self):
        snapshot = input_fixture()
        before = copy.deepcopy(snapshot)
        submitted.validate_submitted_record_metadata(snapshot, [])
        self.assertEqual(snapshot, before)
        self.assertIsNone(snapshot["attack_mapping"][0]["selected_counter_local_index"])
        with self.assertRaises(ValueError):
            submitted.validate_submitted_record_metadata(snapshot, [metadata_fixture()])

    def test_submitted_store_root_is_distinct_from_contract_state_owner(self):
        _manifest, record = native_fixture()
        review = record["native"]["raw_submission"]["submitted_review"]
        for change in ({"generation": review["generation"] + 1}, {"root_state": review["root_state"] + 1}):
            with self.subTest(change=tuple(change)), self.assertRaisesRegex(ValueError, "Store generation/root"):
                submitted.validate_submitted_review(record, {**review, **change})


if __name__ == "__main__":
    unittest.main()
