"""Synthetic /3 report wiring, never a native owner or actual clock proof."""
import copy
import tempfile
import unittest

from rz_pals_model import strategic_verifier_query as query
import test_strategic_verifier_query as fixtures


def report(base):
    assets = base.raw_assets()
    pins = assets["expected_pins"]
    semantic = base.action_semantic_inputs()[0]
    receipt = semantic.rules_receipt()
    prior = query._parse(assets["before_result"])["prior_ledger_sha256"]
    value = {"schema": query.REPAIR_QUERY_REPORT_SCHEMA,
        "assurance_scope": query.REPAIR_QUERY_REPORT_SCOPE, "parent": pins["parent"],
        "source_binding": {"query_sha256": base.sha256, "catalogue_artifact": pins["catalogue"],
            "before_result_artifact": pins["before_result"], "prior_ledger_sha256": prior,
            "semantic_input_sha256": semantic.sha256, "semantic_context_sha256": receipt["context_sha256"],
            "semantic_branch_meaning_sha256": receipt["branch_meaning_sha256"],
            "semantic_before_result_anchor_sha256": receipt["before_result_anchor_sha256"],
            "cpu_request_artifact": query.byte_pin(b"synthetic original request")},
        "query_sha256": "0" * 64, "prior_ledger_before_sha256": prior,
        "prior_ledger_sha256": "0" * 64, "prior_ordinal": 0,
        "catalogue": pins["catalogue"], "source_input": query.byte_pin(b"synthetic replay input"),
        "source_output": query.byte_pin(b"synthetic /4 output"), "conditional_fact_sha256": "1" * 64,
        "clock_scope": "before_selection_callback", "selection_interval_ns": [0, 17],
        "reported_work_counts": [21, 4],
        "reported_work_counts_scope": "child_report_correlated_to_independent_native_not_same_physical_execution",
        "elapsed_through_admission_ns": 91, "ledger_event_elapsed_ns": 87, "whole_budget_ns": 1000,
        "native_execution_is_same_physical_child": False, "query2_action_semantics_revalidated": False,
        "utility_authority": False, "target_authority": False, "training_authority": False,
        "product_authority": False}
    return seal(value)


def seal(value):
    source = value["source_binding"]
    event = {"ordinal": value["prior_ordinal"], "source_query": source["query_sha256"],
        "previous_ledger": value["prior_ledger_before_sha256"], "source_input": value["source_input"],
        "source_output": value["source_output"], "conditional_fact_sha256": value["conditional_fact_sha256"],
        "ledger_event_elapsed_ns": value["ledger_event_elapsed_ns"]}
    value["prior_ledger_sha256"] = query._repair_episode_digest("rz-pals-actual-repair-prior-ledger/3", event)
    identity = {"schema": query.REPAIR_QUERY_REPORT_SCHEMA, "parent": value["parent"],
        "source_query_sha256": source["query_sha256"], "prior_ledger_sha256": value["prior_ledger_sha256"],
        "decision_ordinal": value["prior_ordinal"] + 1, "catalogue": value["catalogue"]}
    value["query_sha256"] = query._repair_episode_digest(query.REPAIR_QUERY_REPORT_SCHEMA, identity)
    return value


class RepairQueryReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = fixtures.QueryFixture(self.directory.name)
        self.base = self.fixture.admit()
        self.value = report(self.base)

    def admit(self, value=None, expected=None):
        raw = query.canonical(self.value if value is None else value)
        return query.admit_repair_query_report(original_query=self.base, report_bytes=raw,
            expected_report_pin=query.byte_pin(raw) if expected is None else expected)

    def test_report_preserves_query2_and_cannot_import_native_or_legacy_receipt(self):
        before = self.base.sha256
        checked = self.admit()
        self.assertEqual(checked.sha256, self.value["query_sha256"])
        self.assertEqual(checked.original_action_features(), self.base.features())
        self.assertEqual(self.base.sha256, before)
        self.assertFalse(checked.audit()["live_native_capability"])
        self.assertFalse(checked.audit()["legacy_cpu_receipt_created"])
        changed = checked.report()
        changed["utility_authority"] = True
        self.assertFalse(checked.report()["utility_authority"])
        with self.assertRaises(AttributeError):
            checked._raw = b"{}"
        with self.assertRaises(ValueError):
            query.CheckedRepairQueryReport()

    def test_report_rejects_pin_parent_binding_prior_order_expiry_and_authority(self):
        with self.assertRaises(ValueError):
            self.admit(expected=query.byte_pin(b"another observation"))
        for change in (lambda v: v["parent"].update(encoding_sha256="2"*64),
                lambda v: v["source_binding"].update(query_sha256="2"*64),
                lambda v: v["source_binding"].update(semantic_context_sha256="2"*64),
                lambda v: v.update(prior_ledger_before_sha256="2"*64),
                lambda v: v.update(prior_ordinal=1),
                lambda v: v.update(elapsed_through_admission_ns=1000),
                lambda v: v.update(ledger_event_elapsed_ns=92),
                lambda v: v.update(selection_interval_ns=[0,92]),
                lambda v: v.update(native_execution_is_same_physical_child=True),
                lambda v: v.update(utility_authority=True),
                lambda v: v.update(training_authority=True),
                lambda v: v.update(reported_work_counts=[True,4]),
                lambda v: v.update(extra="unknown field")):
            value = copy.deepcopy(self.value)
            change(value)
            seal(value)
            with self.assertRaises(ValueError):
                self.admit(value)

    def test_report_identity_and_unknown_work_remain_separate(self):
        value = copy.deepcopy(self.value)
        value["source_output"] = query.byte_pin(b"changed output")
        with self.assertRaises(ValueError):
            self.admit(value)
        value = copy.deepcopy(self.value)
        value.update(clock_scope="preparation_only", selection_interval_ns=None,
            reported_work_counts=None)
        checked = self.admit(value)
        self.assertIsNone(checked.report()["reported_work_counts"])
        self.assertEqual(checked.report()["clock_scope"], "preparation_only")
        self.assertFalse(checked.audit()["runtime_clock_proof"])

    def test_same_semantic_can_be_reused_with_distinct_output_allowances(self):
        spec, semantic = self.fixture.actions[0]
        extra = dict(spec, max_output_bytes=spec["max_output_bytes"] // 2)
        self.fixture.actions.append((extra, semantic))
        self.fixture.refresh()
        self.base = self.fixture.admit()
        self.value = report(self.base)
        checked = self.admit()
        self.assertEqual(len(checked.original_action_features()), 3)

    def test_captured_lane_is_distinct_report_without_child_execution_authority(self):
        value = copy.deepcopy(self.value)
        value["assurance_scope"] = query.CAPTURED_REPAIR_QUERY_REPORT_SCOPE
        with self.assertRaises(ValueError):
            self.admit(value)
        value["reported_work_counts_scope"] = "child_report_separate_from_captured_nn_and_independent_cpu_executions"
        checked = self.admit(value)
        self.assertEqual(checked.report()["assurance_scope"], query.CAPTURED_REPAIR_QUERY_REPORT_SCOPE)
        self.assertFalse(checked.audit()["live_native_capability"])
        self.assertFalse(checked.audit()["runtime_clock_proof"])


if __name__ == "__main__":
    unittest.main()
