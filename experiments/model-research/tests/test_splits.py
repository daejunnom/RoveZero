"""Source grouping and leakage safety checks with independently varied inputs."""

import copy
import unittest

from rz_data.errors import DataError
from rz_data.serialization import digest
from rz_data.splits import audit_leakage, make_split_plan, validate_split_plan


def source(game_id, opening=None, lineage=None, seed_group=None):
    return {
        "game_id": game_id, "opening_family_id": opening,
        "lineage_id": lineage, "seed_group_id": seed_group,
    }


def record(record_id, source_row, plan, *, input_key=None, board_key=None):
    return {
        "record_id": record_id, **source_row,
        "split": plan["assignments"][source_row["game_id"]],
        "parent_record_id": None, "augmentation_id": None,
        "input_identity": {
            "digest": digest(input_key or record_id),
            "encoding_id": "fixture-encoding", "encoding_version": "1",
        },
        "state": {
            "board_digest": digest(board_key or record_id),
            "state_digest": digest([record_id, "full-state"]),
            "initial_fen": "fixture", "moves": [], "history_completeness": "complete",
        },
        "failure_groups": ["unique-defense"],
    }


def multi_split_fixture():
    sources = [source(f"game-{index:03}") for index in range(100)]
    plan = make_split_plan(sources, seed="audit-fixture", ratios=(1, 1, 1))
    selected = {}
    for row in sources:
        selected.setdefault(plan["assignments"][row["game_id"]], row)
    assert set(selected) == {"train", "validation", "holdout"}
    return plan, selected


def codes(report, field="errors"):
    return {item["code"] for item in report[field]}


class SplitPlanTests(unittest.TestCase):
    def test_order_independent_connected_groups_and_namespace(self):
        rows = [
            source("a", opening="open"), source("b", opening="open", lineage="line"),
            source("c", lineage="line", seed_group="seed"), source("d", seed_group="seed"),
            source("e", opening="line"), source("f", lineage="open"),
        ]
        original = copy.deepcopy(rows)
        plan = make_split_plan(rows, seed="lock", ratios=(1, 1, 1))
        self.assertEqual(plan, make_split_plan(list(reversed(rows)), seed="lock", ratios=(1, 1, 1)))
        self.assertEqual(rows, original)
        self.assertEqual(len({plan["assignments"][game] for game in "abcd"}), 1)
        # A single component contributes a single hash draw, independent of roots.
        component_slot = int(digest({"seed": "lock", "component": list("abcd")}), 16) % 3
        expected = ("train", "validation", "holdout")[component_slot]
        self.assertEqual(plan["assignments"]["a"], expected)
        isolated = make_split_plan([rows[4]], seed="lock", ratios=(1, 1, 1))
        self.assertEqual(plan["assignments"]["e"], isolated["assignments"]["e"])
        self.assertNotEqual(plan["assignments"]["a"], plan["assignments"]["e"])
        isolated = make_split_plan([rows[5]], seed="lock", ratios=(1, 1, 1))
        self.assertEqual(plan["assignments"]["f"], isolated["assignments"]["f"])
        self.assertNotEqual(plan["assignments"]["a"], plan["assignments"]["f"])
        self.assertEqual(validate_split_plan(plan), plan)

    def test_plan_rejects_tampering_even_with_recomputed_digest(self):
        plan = make_split_plan([source("a")], seed="lock")
        changed = copy.deepcopy(plan)
        changed["assignments"]["a"] = "holdout" if plan["assignments"]["a"] != "holdout" else "train"
        changed["digest"] = digest({key: value for key, value in changed.items() if key != "digest"})
        with self.assertRaises(DataError) as caught:
            validate_split_plan(changed)
        self.assertEqual(caught.exception.code, "SplitPlanTampered")
        changed = copy.deepcopy(plan)
        changed["schema_version"] = 2
        with self.assertRaises(DataError) as caught:
            validate_split_plan(changed)
        self.assertEqual(caught.exception.code, "UnsupportedSchema")
        changed["schema_version"] = True
        with self.assertRaises(DataError):
            validate_split_plan(changed)

    def test_rejects_invalid_sources_and_weights(self):
        for weights in ((True, 1, 1), (1, 0, 1), (1, -1, 1), (1.0, 1, 1), (1, 1), (1, float("inf"), 1)):
            with self.subTest(weights=weights), self.assertRaises(DataError):
                make_split_plan([source("a")], seed="lock", ratios=weights)
        for rows in ([source("a"), source("a")], [source("")], [{"game_id": "a"}],
                     [source("a", opening="")], [source("a", lineage=["unhashable"])]):
            with self.subTest(rows=rows), self.assertRaises(DataError):
                make_split_plan(rows, seed="lock")

    def test_empty_sources_are_reproducible_and_no_fake_rows_are_added(self):
        plan = make_split_plan([], seed="empty")
        self.assertEqual(plan["assignments"], {})
        self.assertEqual(validate_split_plan(plan), plan)
        report = audit_leakage([], plan)
        self.assertEqual(report["counts"]["records"], 0)
        self.assertEqual(report["errors"], [])


class LeakageTests(unittest.TestCase):
    def test_unknown_game_and_erased_provenance_are_errors(self):
        plan, rows = multi_split_fixture()
        item = record("r", rows["train"], plan)
        item["game_id"] = "unlisted"
        del item["lineage_id"]
        report = audit_leakage([item], plan)
        self.assertIn("UnknownGame", codes(report))
        self.assertIn("MissingGroupField", codes(report))
        self.assertIn("lineage_id", report["ledger"][0])

    def test_invalid_field_types_are_typed_findings_not_type_errors(self):
        plan, rows = multi_split_fixture()
        item = record("r", rows["train"], plan)
        item.update(game_id=["unhashable"], split=["train"], lineage_id={"bad": "group"},
                    parent_record_id={"bad": "parent"}, failure_groups=[{"bad": "group"}])
        item["input_identity"]["encoding_version"] = ["1"]
        report = audit_leakage([item], plan)
        self.assertTrue({"UnknownGame", "InvalidSplit", "InvalidGroupId", "InvalidFailureGroups",
                         "InvalidInputIdentity", "MissingParent"}.issubset(codes(report)))
        with self.assertRaises(DataError):
            audit_leakage({}, plan)

    def test_exact_duplicate_full_inputs_across_splits(self):
        plan, rows = multi_split_fixture()
        records = [record("r-a", rows["train"], plan, input_key="same"),
                   record("r-b", rows["holdout"], plan, input_key="same")]
        original = copy.deepcopy(records)
        report = audit_leakage(records, plan)
        self.assertEqual(codes(report), {"ExactInputAcrossSplits"})
        self.assertEqual(report["errors"][0]["record_ids"], ["r-a", "r-b"])
        self.assertEqual(records, original)
        self.assertEqual(report["failure_groups"]["train"], {"unique-defense": 1})
        self.assertEqual(report["failure_groups"]["holdout"], {"unique-defense": 1})
        self.assertEqual(report["coverage"]["arbitrary_near_positions"], "not_run")
        self.assertEqual(report["coverage"]["pretraining_overlap"], "unknown")

    def test_same_board_distinct_histories_are_conservative_risk(self):
        plan, rows = multi_split_fixture()
        records = [record("r-a", rows["train"], plan, input_key="history-a", board_key="board"),
                   record("r-b", rows["holdout"], plan, input_key="history-b", board_key="board")]
        records[1]["state"]["moves"] = ["g1f3", "g8f6", "f3g1", "f6g8"]
        report = audit_leakage(records, plan)
        self.assertNotIn("ExactInputAcrossSplits", codes(report))
        self.assertEqual(codes(report, "warnings"), {"PositionOverlapRisk"})
        self.assertEqual(report["warnings"][0]["context"], "board_digest")
        records[1]["input_identity"]["digest"] = records[0]["input_identity"]["digest"]
        records[1]["input_identity"]["encoding_version"] = "2"
        report = audit_leakage(records, plan)
        self.assertNotIn("ExactInputAcrossSplits", codes(report))
        self.assertIn("PositionOverlapRisk", codes(report, "warnings"))

    def test_metadata_and_assignment_changes_are_reported(self):
        rows = [source("a", opening="same"), source("b", opening="same")]
        plan = make_split_plan(rows, seed="lock")
        records = [record("a-r", rows[0], plan), record("b-r", rows[1], plan)]
        records[1]["split"] = "holdout" if records[0]["split"] != "holdout" else "train"
        records[1]["lineage_id"] = "fabricated"
        report = audit_leakage(records, plan)
        self.assertIn("SplitAssignmentMismatch", codes(report))
        self.assertIn("GroupMetadataMismatch", codes(report))
        self.assertIn("SourceGroupAcrossSplits", codes(report))

    def test_augmentation_keeps_lineage_and_parent_split(self):
        rows = [source("a", lineage="original-family"), source("b", lineage="original-family")]
        plan = make_split_plan(rows, seed="lock")
        parent = record("original", rows[0], plan)
        derived = record("derived", rows[1], plan)
        derived.update(parent_record_id="original", augmentation_id="safe-transform-v1")
        self.assertEqual(audit_leakage([parent, derived], plan)["errors"], [])
        derived["split"] = "holdout" if parent["split"] != "holdout" else "train"
        report = audit_leakage([parent, derived], plan)
        self.assertIn("ParentSplitMismatch", codes(report))
        self.assertIn("SourceGroupAcrossSplits", codes(report))
        derived["parent_record_id"] = None
        self.assertIn("MissingAugmentationParent", codes(audit_leakage([parent, derived], plan)))

    def test_disconnected_parent_sources_are_detected_even_when_splits_match(self):
        plan, rows = multi_split_fixture()
        train_sources = [row for row in plan["sources"] if plan["assignments"][row["game_id"]] == "train"]
        parent = record("original", train_sources[0], plan)
        derived = record("derived", train_sources[1], plan)
        derived.update(parent_record_id="original", augmentation_id="transform-v1")
        self.assertIn("DisconnectedProvenance", codes(audit_leakage([parent, derived], plan)))

    def test_missing_self_and_cyclic_parent_graphs(self):
        row = source("a")
        plan = make_split_plan([row], seed="lock")
        a, b, c = [record(key, row, plan) for key in ("a", "b", "c")]
        a["parent_record_id"], b["parent_record_id"] = "b", "a"
        c["parent_record_id"] = "missing"
        report = audit_leakage([a, b, c], plan)
        self.assertIn("ParentCycle", codes(report))
        self.assertIn("MissingParent", codes(report))
        cycle = next(item for item in report["errors"] if item["code"] == "ParentCycle")
        self.assertEqual(cycle["record_ids"], ["a", "b"])
        c["parent_record_id"] = "c"
        self.assertIn("SelfParent", codes(audit_leakage([c], plan)))

    def test_long_parent_chain_and_bucket_output_are_linear(self):
        plan, rows = multi_split_fixture()
        records = [record(f"r-{index:04}", rows["train"], plan, input_key="duplicate") for index in range(1200)]
        for previous, current in zip(records, records[1:]):
            current["parent_record_id"] = previous["record_id"]
        records.append(record("holdout-r", rows["holdout"], plan, input_key="duplicate"))
        report = audit_leakage(records, plan)
        self.assertEqual(codes(report), {"ExactInputAcrossSplits"})
        self.assertEqual(len(report["errors"]), 1)
        self.assertEqual(len(report["errors"][0]["record_ids"]), len(records))
        self.assertEqual(len(report["ledger"]), len(records))


if __name__ == "__main__":
    unittest.main()
