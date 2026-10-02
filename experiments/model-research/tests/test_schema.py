"""Envelope and target regressions; synthetic moves are not Rules evidence."""

from copy import deepcopy
from pathlib import Path
import unittest

from rz_data.errors import DataError
from rz_data.schema import validate_manifest, validate_record
from rz_data.serialization import digest, loads


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_fixture = loads((FIXTURES / "manifest.json").read_bytes())
        cls.record_fixture = loads((FIXTURES / "record.json").read_bytes())

    def setUp(self):
        self.manifest = deepcopy(self.manifest_fixture)
        self.record = deepcopy(self.record_fixture)

    def assert_invalid_manifest(self, manifest, code):
        with self.assertRaises(DataError) as caught:
            validate_manifest(manifest)
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(caught.exception.context.startswith("manifest"))

    def assert_invalid_record(self, record, code):
        with self.assertRaises(DataError) as caught:
            validate_record(record, self.manifest)
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(caught.exception.context.startswith("record"))

    def test_validators_return_original_objects_without_repair(self):
        original_manifest = deepcopy(self.manifest)
        original_record = deepcopy(self.record)
        self.assertIs(validate_manifest(self.manifest), self.manifest)
        self.assertIs(validate_record(self.record, self.manifest), self.record)
        self.assertEqual(self.manifest, original_manifest)
        self.assertEqual(self.record, original_record)
        self.assertFalse(self.manifest["execution_ready"])
        self.assertIsNone(self.manifest["engine_contract_revision"])

    def test_unknown_or_bool_schema_versions_are_rejected(self):
        for version in (2, True, "1"):
            with self.subTest(version=version):
                manifest = deepcopy(self.manifest)
                manifest["schema_version"] = version
                self.assert_invalid_manifest(manifest, "InvalidEnum")
                record = deepcopy(self.record)
                record["schema_version"] = version
                self.assert_invalid_record(record, "InvalidEnum")

    def test_no_value_can_enable_execution_without_engine_bindings(self):
        for ready in (True, 0, None, "false"):
            with self.subTest(ready=ready):
                manifest = deepcopy(self.manifest)
                manifest["execution_ready"] = ready
                self.assert_invalid_manifest(manifest, "EngineBindingUnavailable")

    def test_confirmed_rights_require_both_license_and_evidence(self):
        for owner in ("source", "teacher"):
            for field in ("license", "reference"):
                with self.subTest(owner=owner, field=field):
                    manifest = deepcopy(self.manifest)
                    manifest[owner]["rights"][field] = None
                    self.assert_invalid_manifest(manifest, "MissingRightsEvidence")

    def test_budgets_and_tolerance_cannot_accept_bool(self):
        for field in self.manifest["teacher"]["budget"]:
            with self.subTest(field=field):
                manifest = deepcopy(self.manifest)
                manifest["teacher"]["budget"][field] = True
                self.assert_invalid_manifest(manifest, "InvalidInteger")
        self.manifest["probability_tolerance"] = True
        self.assert_invalid_manifest(self.manifest, "NonFinite")

    def test_record_encoding_must_match_declared_manifest(self):
        for field in ("encoding_id", "encoding_version"):
            with self.subTest(field=field):
                record = deepcopy(self.record)
                record["input_identity"][field] = "another-encoding"
                self.assert_invalid_record(record, "EncodingMismatch")

    def test_four_promotion_identities_and_targets_are_preserved(self):
        expected = {
            "a7a8q": 0.4, "a7a8r": 0.3,
            "a7a8b": 0.2, "a7a8n": 0.1,
        }
        validated = validate_record(self.record, self.manifest)
        self.assertEqual(set(validated["state"]["legal_moves"]), set(expected))
        policy = validated["label"]["policy"]
        self.assertEqual(dict(zip(policy["moves"], policy["values"])), expected)
        duplicate = deepcopy(self.record)
        duplicate["state"]["legal_moves"][-1] = "a7a8q"
        duplicate["state"]["legal_moves_digest"] = digest(duplicate["state"]["legal_moves"])
        self.assert_invalid_record(duplicate, "DuplicateMove")

    def test_legal_order_requires_new_digest_and_matching_policy_permutation(self):
        original_policy = deepcopy(self.record["label"]["policy"])
        legal = list(reversed(self.record["state"]["legal_moves"]))
        self.record["state"]["legal_moves"] = legal
        self.assert_invalid_record(self.record, "LegalOrderMismatch")
        self.record["state"]["legal_moves_digest"] = digest(legal)
        self.assert_invalid_record(self.record, "PolicyMoveMismatch")
        policy = self.record["label"]["policy"]
        by_move = dict(zip(original_policy["moves"], original_policy["values"]))
        policy["moves"] = list(legal)
        policy["values"] = [by_move[move] for move in legal]
        original_value = deepcopy(self.record["label"]["value"])
        validate_record(self.record, self.manifest)
        self.assertEqual(dict(zip(policy["moves"], policy["values"])), by_move)
        self.assertEqual(self.record["label"]["value"], original_value)

    def test_unknown_fen_prefix_cannot_be_claimed_complete(self):
        self.record["state"]["history_completeness"] = "complete"
        self.assert_invalid_record(self.record, "UnknownHistory")

    def test_nonfinite_targets_are_rejected_without_clipping(self):
        for target in ("policy", "value"):
            for invalid in (float("nan"), float("inf"), float("-inf"), True):
                with self.subTest(target=target, invalid=invalid):
                    record = deepcopy(self.record)
                    field = "values" if target == "policy" else "value"
                    record["label"][target][field][0] = invalid
                    self.assert_invalid_record(record, "NonFinite")

    def test_policy_and_wdl_probabilities_require_range_shape_and_sum(self):
        for target, field, bad_values, code in (
            ("policy", "values", [0.4, 0.3, 0.2, 0.2], "InvalidNormalization"),
            ("policy", "values", [-0.1, 0.3, 0.3, 0.5], "OutOfRange"),
            ("policy", "values", [0.5, 0.5], "InvalidShape"),
            ("value", "value", [0.6, 0.3, 0.2], "InvalidNormalization"),
            ("value", "value", [1.1, 0.0, -0.1], "OutOfRange"),
            ("value", "value", [0.5, 0.5], "InvalidShape"),
        ):
            with self.subTest(target=target, values=bad_values):
                record = deepcopy(self.record)
                record["label"][target][field] = bad_values
                self.assert_invalid_record(record, code)

    def test_visits_require_integer_positive_total_and_search_provenance(self):
        policy = self.record["label"]["policy"]
        policy.update(kind="visits", source="search", normalization="visits_unscaled",
                      values=[4, 3, 2, 1])
        self.assertIs(validate_record(self.record, self.manifest), self.record)
        for values, code in (([True, 3, 2, 1], "InvalidInteger"),
                             ([0, 0, 0, 0], "EmptyVisits")):
            with self.subTest(values=values):
                record = deepcopy(self.record)
                record["label"]["policy"]["values"] = values
                self.assert_invalid_record(record, code)
        policy["source"] = "raw_nn"
        self.assert_invalid_record(self.record, "InvalidEnum")

    def test_teacher_value_requires_explicit_viewpoint_and_scale(self):
        for field, invalid in (("viewpoint", "white"), ("scale", "centipawn")):
            with self.subTest(field=field):
                record = deepcopy(self.record)
                record["label"]["value"][field] = invalid
                self.assert_invalid_record(record, "InvalidEnum")

    def test_centipawns_remain_a_distinct_score_without_conversion(self):
        target = self.record["label"]["value"]
        target.update(kind="centipawn", scale="centipawn", value=325)
        self.assertIs(validate_record(self.record, self.manifest), self.record)
        self.assertEqual(target["value"], 325)
        self.assertEqual(target["kind"], "centipawn")
        target.update(kind="q", scale="wdl_difference")
        self.assert_invalid_record(self.record, "OutOfRange")

    def test_failed_partial_or_canceled_labels_cannot_fabricate_targets(self):
        for status in ("failed", "partial", "canceled"):
            for retained_target in ("policy", "value"):
                with self.subTest(status=status, retained_target=retained_target):
                    record = deepcopy(self.record)
                    label = record["label"]
                    label.update(status=status, failure_reason="analysis did not finish")
                    label["value" if retained_target == "policy" else "policy"] = None
                    self.assert_invalid_record(record, "InvalidFailureLabel")
            record = deepcopy(self.record)
            record["label"].update(status=status, failure_reason="analysis did not finish",
                                   policy=None, value=None)
            self.assertIs(validate_record(record, self.manifest), record)

    def test_complete_label_needs_a_target_and_no_failure_reason(self):
        self.record["label"].update(policy=None, value=None)
        self.assert_invalid_record(self.record, "MissingTarget")
        record = deepcopy(self.record_fixture)
        record["label"]["failure_reason"] = "crash"
        self.assert_invalid_record(record, "CompletionMismatch")

    def test_incomplete_game_keeps_teacher_wdl_separate_without_outcome_target(self):
        teacher_wdl = deepcopy(self.record["label"]["value"]["value"])
        validate_record(self.record, self.manifest)
        self.assertEqual(self.record["label"]["value"]["value"], teacher_wdl)
        self.assertIsNone(self.record["game_result"]["wdl"])
        self.record["game_result"]["wdl"] = [0, 1, 0]
        self.assert_invalid_record(self.record, "IncompleteResult")

    def test_actual_game_outcome_is_one_hot_and_not_teacher_distribution(self):
        self.record["game_result"].update(status="rule_terminal", wdl=[0, 0, 1],
                                          termination_reason="synthetic checkmate declaration")
        validate_record(self.record, self.manifest)
        self.assertEqual(self.record["label"]["value"]["value"], [0.6, 0.2, 0.2])
        self.record["game_result"]["wdl"] = [0.6, 0.2, 0.2]
        self.assert_invalid_record(self.record, "InvalidGameOutcome")

    def test_complete_label_cannot_exceed_declared_cost_budgets(self):
        for cost, maximum in (("time_ms", "max_time_ms_per_position"),
                              ("nodes", "max_nodes_per_position"),
                              ("peak_memory_bytes", "max_memory_bytes")):
            with self.subTest(cost=cost):
                record = deepcopy(self.record)
                record["label"]["actual_cost"][cost] = self.manifest["teacher"]["budget"][maximum] + 1
                self.assert_invalid_record(record, "BudgetExceeded")

    def test_terminal_empty_legal_accepts_value_only_and_rejects_policy(self):
        self.record["state"].update(play_status="terminal", legal_moves=[],
                                     legal_moves_digest=digest([]),
                                     termination_reason="synthetic stalemate declaration")
        self.record["label"]["policy"] = None
        self.assertIs(validate_record(self.record, self.manifest), self.record)
        self.record["label"]["policy"] = deepcopy(self.record_fixture["label"]["policy"])
        self.record["label"]["policy"].update(moves=[], values=[])
        self.assert_invalid_record(self.record, "EmptyPolicy")

    def test_ongoing_empty_legal_is_rejected(self):
        self.record["state"].update(legal_moves=[], legal_moves_digest=digest([]))
        self.record["label"]["policy"] = None
        self.assert_invalid_record(self.record, "TerminalMismatch")


if __name__ == "__main__":
    unittest.main()
