"""Pure synthetic /2 audit inspection, never live action/utility authority.

These tests construct reports, not Rust capabilities, process/native owners,
model execution, losses or training. A numerical claim stays a report even when
its own accounting is consistent and its claimed preference is Pareto-valid.
"""
import copy
import unittest

from rz_pals_model import strategic_verifier_query as query
from rz_pals_model import strategic_verifier_utility as utility


def synthetic_parent():
    return {name: "8" * 64 for name in utility._REPORT_PARENT_FIELDS}


def synthetic_binding():
    return dict({name: "9" * 64 for name in utility._REPORT_BINDING_SHA_FIELDS},
                **{name: query.byte_pin(b"synthetic bound asset") for name in utility._REPORT_BINDING_PIN_FIELDS})


def synthetic_action(index=0, wall=1000):
    return {"slot": index, "task": "defend_response", "semantic_input_sha256": "1" * 64,
        "profile_registration": 0, "baseline_depth": 1, "requested_depth": 2, "max_nodes_per_check": 100,
        "max_wall_time_ms": wall, "max_output_bytes": 4096, "budget_bucket": 1}


def synthetic_whole_report():
    return {"schema": utility.WHOLE_COST_REPORT_SCHEMA, "policy": utility._COVERAGE_POLICY,
        "query_sha256": "1" * 64, "parent": synthetic_parent(), "source_binding": synthetic_binding(),
        "source_input": query.byte_pin(b"synthetic input"), "source_output": query.byte_pin(b"synthetic output"),
        "conditional_fact_sha256": "2" * 64, "selected_original_action_index": 0,
        "selected_action": synthetic_action(),
        "whole_elapsed_ns": 100, "whole_cost": {"whole_elapsed_ns": 100, "cpu_nodes": 80, "physical_nn_inputs": 6},
        "initial_selection": {"start_ns": 0, "end_ns": 5}, "next_selection": {"start_ns": 90, "end_ns": 100},
        "independent_native_work_interval": {"start_ns": 40, "end_ns": 80}, "source_work": [40, 3],
        "independent_native_work": [40, 3], "work_counter_scope": utility._COUNTER_SCOPE,
        "source_native_ids_equated_to_witness": False, "whole_causal_elapsed_observed": True,
        "preparation_transport_check_intervals_ns": {"preparation_and_preflight": 10, "supervisor_setup": 2,
            "launch_work_and_wait": 20, "drain": 2, "postflight": 2, "after_capture_before_check": 2,
            "result_check": 40, "check_to_next_query_admission": 10},
        "logical_phase_elapsed_ns": {"initial_v_or_coverage_selection": 5, "preparation": None,
            "nn": None, "cpu": None, "transfer": None, "check": None, "next_selection": 10},
        "logical_phase_scope": "combined_parent_intervals_observed;unseparated_subphases_unknown;no_additive_double_count",
        "neural_v_executed": False, "neural_v_skipped_by_policy": True, "selector_cpu_search_executed": False,
        "selector_nn_inputs": 0, "next_selection_cpu_search_executed": False, "next_selection_nn_inputs": 0,
        "next_choice": "defer", "whole_work_counts_known": True, "utility_authority": False,
        "target_authority": False, "training_authority": False, "learned_utility_claim": False,
        "product_verifier_enabled": False, "optimizer_steps": 0}


def synthetic_pair_report():
    return {"schema": utility.CONDITIONAL_COST_REPORT_SCHEMA,
        "policy": "same_conditional_raw_endpoint_whole_cost_dominance/2",
        "scope": "conditional_cost_observation_only;not_paper_reward_or_learned_utility",
        "parent": synthetic_parent(), "source_query_sha256": "1" * 64, "prior_ledger_before_sha256": "2" * 64,
        "source_inputs": [query.byte_pin(b"synthetic left"), query.byte_pin(b"synthetic right")],
        "selected_original_action_indices": [0, 1], "source_actions": [synthetic_action(), synthetic_action(1, 1100)],
        "whole_costs": [{"whole_elapsed_ns": 90, "cpu_nodes": 80, "physical_nn_inputs": 6},
                        {"whole_elapsed_ns": 100, "cpu_nodes": 80, "physical_nn_inputs": 6}],
        "work_counter_scope": utility._COUNTER_SCOPE, "preference": "left", "mask": True,
        "reason": "same_conditional_fact_strict_whole_cost_dominance", "actual_utility_groups": 1,
        "utility_observation_admitted": True, "target_authority": False, "training_authority": False,
        "neural_v_executed": False, "learned_utility_claim": False, "strategic_refutation_admitted": False,
        "rules_proof_admitted": False, "depth_or_cpu_agreement_reward": False, "no_counterexample_reward": False,
        "paper_reward_claim": False, "product_verifier_enabled": False, "optimizer_steps": 0}


def synthetic_captured_report(*, pair=False):
    value = synthetic_pair_report() if pair else synthetic_whole_report()
    value["schema"] = (utility.CAPTURED_CONDITIONAL_COST_REPORT_SCHEMA if pair
                       else utility.CAPTURED_WHOLE_COST_REPORT_SCHEMA)
    value["work_counter_scope"] = utility._CAPTURED_COUNTER_SCOPE
    value.update(source_child_work_provenance=utility._SOURCE_REPORT_SCOPE,
                 source_child_physical_work_attested=False, source_cpu_scalar_score_correlated=None)
    if pair:
        value["conditional_fact_scope"] = utility._FRESH_FACT_SCOPE
    else:
        value["independent_witness_work_interval"] = value.pop("independent_native_work_interval")
        value["actual_independent_work_scope"] = utility._INDEPENDENT_WORK_SCOPE
        # Different independent workloads are charged without inventing equality
        # with the separately retained child work report.
        value["independent_native_work"] = [50, 2]
        value["whole_cost"].update(cpu_nodes=90, physical_nn_inputs=5)
    return value


class CoverageCostReportTests(unittest.TestCase):
    def inspect(self, value, *, pair=False):
        raw = query.canonical(value)
        inspect = utility.inspect_conditional_cost_pair_report if pair else utility.inspect_coverage_whole_cost_report
        return inspect(report_bytes=raw, expected_report_pin=query.byte_pin(raw))

    def assert_report_only(self, result):
        self.assertEqual(result["actual_utility_groups"], 0)
        for name in ("utility_authority", "target_authority", "training_authority",
                     "native_capability_imported", "physical_execution_attested_here"):
            self.assertIs(result[name], False)

    def test_source_and_independent_work_both_charged_but_not_same_physical_execution(self):
        value = synthetic_whole_report()
        result = self.inspect(value)
        self.assert_report_only(result)
        self.assertEqual(result["reported"]["whole_cost"]["physical_nn_inputs"], 6)
        result["reported"]["whole_cost"]["cpu_nodes"] = 0
        self.assertEqual(value["whole_cost"]["cpu_nodes"], 80)
        for source, native, claimed in ((None, [40, 3], [80, 6]), ([40, 3], [41, 3], [81, 6]),
                                       ([40, 3], [40, 3], [40, 3])):
            changed = copy.deepcopy(value)
            changed.update(source_work=source, independent_native_work=native)
            changed["whole_cost"].update(cpu_nodes=claimed[0], physical_nn_inputs=claimed[1])
            with self.subTest(source=source, native=native), self.assertRaises(ValueError):
                self.inspect(changed)

    def test_unobserved_counts_and_subphases_stay_unknown(self):
        value = synthetic_whole_report()
        value.update(source_work=None, whole_cost=None, whole_work_counts_known=False)
        result = self.inspect(value)
        self.assert_report_only(result)
        self.assertIsNone(result["reported"]["whole_cost"])
        self.assertIsNone(result["reported"]["logical_phase_elapsed_ns"]["nn"])
        value["logical_phase_elapsed_ns"]["nn"] = 0
        with self.assertRaisesRegex(ValueError, "unseparated"):
            self.inspect(value)

    def test_whole_report_refuses_bool_counts_late_clock_authority_and_raw_mutation(self):
        for key, replacement in (("source_native_ids_equated_to_witness", True), ("utility_authority", True),
                                 ("neural_v_executed", True), ("selector_nn_inputs", True),
                                 ("whole_work_counts_known", 1)):
            value = synthetic_whole_report()
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.inspect(value)
        value = synthetic_whole_report()
        value["whole_cost"]["cpu_nodes"] = True
        with self.assertRaises(ValueError):
            self.inspect(value)
        value = synthetic_whole_report()
        value["logical_phase_elapsed_ns"]["initial_v_or_coverage_selection"] = True
        value["initial_selection"]["end_ns"] = 1
        with self.assertRaises(ValueError):
            self.inspect(value)
        value = synthetic_whole_report()
        value["next_selection"]["end_ns"] = 101
        with self.assertRaises(ValueError):
            self.inspect(value)
        raw = query.canonical(synthetic_whole_report())
        with self.assertRaisesRegex(ValueError, "independent"):
            utility.inspect_coverage_whole_cost_report(report_bytes=raw, expected_report_pin=query.byte_pin(raw + b" "))

    def test_report_identity_is_closed_without_becoming_live_authority(self):
        for name in ("parent", "source_binding"):
            value = synthetic_whole_report()
            value[name] = {}
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.inspect(value)
        value = synthetic_pair_report()
        value["parent"]["caller_capability"] = True
        with self.assertRaises(ValueError):
            self.inspect(value, pair=True)

    def test_registered_coverage_policy_stays_untrained_and_same_action_variance_masked(self):
        value = synthetic_whole_report()
        value["policy"] = utility._REGISTERED_COVERAGE_POLICY
        self.assert_report_only(self.inspect(value))
        value["selected_action"]["slot"] = 1
        with self.assertRaises(ValueError):
            self.inspect(value)
        value = synthetic_pair_report()
        value["source_actions"][1] = synthetic_action(1)
        value["source_actions"][1]["budget_bucket"] = 7
        value["source_actions"][1]["profile_registration"] = 3
        with self.assertRaisesRegex(ValueError, "repetition"):
            self.inspect(value, pair=True)
        value.update(mask=False, actual_utility_groups=0, utility_observation_admitted=False,
                     preference=None, reason="same_action_repetition_is_variance_observation_only")
        self.assert_report_only(self.inspect(value, pair=True))

    def test_captured_input_scope_separately_charges_different_independent_work(self):
        value = synthetic_captured_report()
        result = self.inspect(value)
        self.assert_report_only(result)
        self.assertEqual(result["reported"]["whole_cost"]["cpu_nodes"], 90)
        self.assertEqual(result["reported"]["whole_cost"]["physical_nn_inputs"], 5)
        for name, changed in (("source_child_physical_work_attested", True),
                              ("source_cpu_scalar_score_correlated", True), ("work_counter_scope", utility._COUNTER_SCOPE)):
            report = copy.deepcopy(value)
            report[name] = changed
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.inspect(report)
        value.update(source_work=None, whole_cost=None, whole_work_counts_known=False)
        self.assertIsNone(self.inspect(value)["reported"]["whole_cost"])

    def test_captured_fresh_cpu_fact_report_is_not_child_scalar_or_utility_authority(self):
        value = synthetic_captured_report(pair=True)
        self.assert_report_only(self.inspect(value, pair=True))
        value["source_cpu_scalar_score_correlated"] = 0
        with self.assertRaises(ValueError):
            self.inspect(value, pair=True)

    def test_consistent_reported_preference_does_not_import_utility_target(self):
        value = synthetic_pair_report()
        result = self.inspect(value, pair=True)
        self.assert_report_only(result)
        self.assertEqual(result["reported"]["actual_utility_groups"], 1)
        self.assertEqual(result["reported"]["preference"], "left")

    def test_crossing_tie_unknown_or_masked_cost_cannot_claim_preference(self):
        for left in (None, {"whole_elapsed_ns": 100, "cpu_nodes": 80, "physical_nn_inputs": 6},
                     {"whole_elapsed_ns": 90, "cpu_nodes": 81, "physical_nn_inputs": 6}):
            value = synthetic_pair_report()
            value["whole_costs"][0] = left
            with self.subTest(left=left), self.assertRaises(ValueError):
                self.inspect(value, pair=True)
        value = synthetic_pair_report()
        value.update(mask=False, actual_utility_groups=0, utility_observation_admitted=False)
        with self.assertRaisesRegex(ValueError, "masked"):
            self.inspect(value, pair=True)
        value.update(preference=None, reason="whole_work_counter_unknown", whole_costs=[None, None])
        self.assert_report_only(self.inspect(value, pair=True))


if __name__ == "__main__":
    unittest.main()
