"""Authored parser fixtures only; no NN library, model load or device access."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pals_gqa_cost as cost


def node(name, op, inputs, output=None):
    return {"name": name, "op": op, "inputs": inputs, "outputs": [output or name + "_out"]}


def fixture(provider="CPUExecutionProvider"):
    nodes = []
    events = []
    def event(name, op, duration, shape=None):
        args = {"op_name": op, "provider": provider}
        if shape is not None:
            args["output_type_shape"] = [{"float": shape}]
        return {"cat": "Node", "ph": "X", "name": name + "_kernel_time", "dur": duration, "args": args}
    for kind in ("key", "value"):
        nodes += [node(kind + "_unsq", "Unsqueeze", ["memory_" + kind, "axes"]),
                  node(kind + "_expand", "Expand", [kind + "_unsq_out", "repeat_shape"]),
                  node(kind + "_reshape", "Reshape", [kind + "_expand_out", "flat_shape"])]
        for suffix, op, duration, shape in (("unsq", "Unsqueeze", 2, [1, 2, 1, 69, 64]),
                                          ("expand", "Expand", 3, [1, 2, 3, 69, 64]),
                                          ("reshape", "Reshape", 5, [1, 6, 69, 64])):
            events.append(event(kind + "_" + suffix, op, duration, shape))
    nodes += [node("key_transpose", "Transpose", ["key_reshape_out"]),
              node("scores", "MatMul", ["query", "key_transpose_out"]),
              node("scale", "Mul", ["scores_out", "scale_constant"]),
              node("weights", "Softmax", ["scale_out"]),
              node("attended", "MatMul", ["weights_out", "value_reshape_out"]),
              node("proposer_ffn", "MatMul", ["attended_out", "ffn_weight"])]
    events += [event("scores", "MatMul", 11), event("weights", "Softmax", 13),
               event("attended", "MatMul", 17), event("proposer_ffn", "MatMul", 19)]
    inventory = {"graphs": [{"role": "shared_pc", "sha256": "a" * 64,
                              "inventory": {"path": "root", "nodes": nodes, "subgraphs": []}}]}
    context = {"graph": "shared_pc", "graph_sha256": "a" * 64, "role": "proposer", "batch": 1,
               "records": 3, "memory_tokens": 69}
    return inventory, events, context


class GqaCostTests(unittest.TestCase):
    def test_exact_name_dataflow_shapes_and_cpu_diagnostic_scope(self):
        report = cost.analyze(*fixture())
        self.assertEqual(report["matching_summary"], {"matched": 2, "unknown": 0, "fused": 0})
        self.assertEqual(report["attributed_role"], "proposer")
        self.assertEqual(report["attention_score_value_matmul_and_softmax"]["duration_sum_us"], 41)
        for row in report["repeat_kv_groups"]:
            self.assertEqual(row["repeat_path_duration_us"], 10)
            self.assertEqual(row["provider_scope"], "cpu_diagnostic_events")
            self.assertEqual(row["dimensions"]["logical_expanded_fp32_bytes"], 1 * 6 * 69 * 64 * 4)
            self.assertEqual(row["dimensions"]["physical_copy_or_allocation_bytes"], "unknown")
        self.assertFalse(report["active_engine_bottleneck_claim"])

    def test_cuda_profile_duration_is_not_isolated_gpu_stopwatch(self):
        report = cost.analyze(*fixture("CUDAExecutionProvider"))
        self.assertTrue(all(row["provider_scope"] == "cuda_profile_events" for row in report["repeat_kv_groups"]))
        self.assertFalse(report["parser_gpu_execution"])
        self.assertEqual(report["isolated_gpu_repeat_time"], "unknown")
        self.assertIn("not_isolated_CUDA_kernel_stopwatch", report["duration_scope"])

    def test_unrelated_expand_opcode_never_counts_as_kv_repetition(self):
        inventory, events, context = fixture()
        inventory["graphs"][0]["inventory"]["nodes"] += [
            node("context_unsq", "Unsqueeze", ["candidate_context", "axes"]),
            node("context_expand", "Expand", ["context_unsq_out", "candidate_shape"]),
            node("context_reshape", "Reshape", ["context_expand_out", "candidate_flat"]),
            node("context_product", "MatMul", ["candidate", "context_reshape_out"])]
        self.assertEqual(len(cost.analyze(inventory, events, context)["repeat_kv_groups"]), 2)

    def test_missing_or_changed_named_node_keeps_full_cost_unknown(self):
        for mutation in ("missing", "wrong_op", "missing_duration"):
            inventory, events, context = fixture()
            target = next(event for event in events if event["name"] == "key_expand_kernel_time")
            if mutation == "missing":
                events.remove(target)
            elif mutation == "wrong_op":
                target["args"]["op_name"] = "Tile"
            else:
                del target["dur"]
            report = cost.analyze(inventory, events, context)
            key = next(row for row in report["repeat_kv_groups"] if row["kind"] == "key")
            self.assertEqual(key["status"], "unknown")
            self.assertIsNone(key["repeat_path_duration_us"])

    def test_wrong_or_unobserved_shapes_do_not_impute_expanded_bytes(self):
        for shape in ([1, 4, 69, 64], [2, 6, 69, 64], None):
            inventory, events, context = fixture()
            target = next(event for event in events if event["name"] == "key_reshape_kernel_time")
            target["args"]["output_type_shape"] = [{"float": shape}]
            key = next(row for row in cost.analyze(inventory, events, context)["repeat_kv_groups"] if row["kind"] == "key")
            self.assertEqual(key["dimensions"]["status"], "unknown")
            self.assertNotIn("logical_expanded_fp32_bytes", key["dimensions"])

    def test_explicit_fused_attention_is_combined_time_not_repeat_time(self):
        inventory, _, context = fixture()
        inventory["graphs"][0]["inventory"]["nodes"].append(node("fused_attention", "GroupedQueryAttention", ["query", "memory_key", "memory_value"]))
        events = [{"cat": "Node", "name": "fused_attention_kernel_time", "dur": 99,
                   "args": {"op_name": "GroupedQueryAttention", "provider": "CUDAExecutionProvider"}}]
        report = cost.analyze(inventory, events, context)
        self.assertEqual(report["matching_summary"], {"matched": 0, "unknown": 2, "fused": 1})
        fused = report["explicit_fused_attention"][0]
        self.assertEqual(fused["combined_attention_duration"]["duration_sum_us"], 99)
        self.assertIsNone(fused["repeat_kv_duration_us"])
        self.assertIsNone(fused["logical_expanded_bytes"])

    def test_runtime_fused_name_without_inventory_binding_remains_unattributed(self):
        inventory, _, context = fixture()
        events = [{"cat": "Node", "name": "new_runtime_fused_kernel_time", "dur": 50,
                   "args": {"op_name": "FusedMatMul", "provider": "CUDAExecutionProvider"}}]
        report = cost.analyze(inventory, events, context)
        self.assertFalse(report["explicit_fused_attention"])
        self.assertTrue(report["unattributed_runtime_nodes"][0]["possible_fused"])

    def test_profile_spanning_both_roles_has_no_per_role_attribution(self):
        inventory, events, context = fixture()
        inventory["graphs"][0]["inventory"]["nodes"].append(node("critic_ffn", "MatMul", ["attended_out", "critic_weight"]))
        events.append({"cat": "Node", "name": "critic_ffn_kernel_time", "dur": 1,
                       "args": {"op_name": "MatMul", "provider": "CPUExecutionProvider"}})
        report = cost.analyze(inventory, events, context)
        self.assertEqual(report["observed_private_roles"], ["critic", "proposer"])
        self.assertEqual(report["attributed_role"], "unknown")

    def test_profile_window_counts_and_duration_range_are_not_per_call_time(self):
        inventory, events, context = fixture()
        second = copy.deepcopy(events)
        for event in second:
            event["dur"] *= 2
        report = cost.analyze(inventory, events + second, context)
        for row in report["repeat_kv_groups"]:
            self.assertEqual(row["repeat_path_duration_us"], 30)
            self.assertTrue(row["node_event_counts_equal"])
            self.assertEqual(set(row["observed_node_event_counts"].values()), {2})
            self.assertEqual(row["observed_event_duration"]["known_event_duration_min_us"], 2)
            self.assertEqual(row["observed_event_duration"]["known_event_duration_max_us"], 10)
            self.assertIn("not_per_invocation_wall_time", row["duration_aggregation"])

    def test_partial_profile_window_node_counts_keep_path_duration_unknown(self):
        inventory, events, context = fixture()
        events.append(copy.deepcopy(events[0]))
        key = next(row for row in cost.analyze(inventory, events, context)["repeat_kv_groups"] if row["kind"] == "key")
        self.assertEqual(key["status"], "unknown")
        self.assertFalse(key["node_event_counts_equal"])
        self.assertIsNone(key["repeat_path_duration_us"])

    def test_bounded_dataflow_and_ambiguous_attention_are_fail_closed(self):
        inventory, events, context = fixture()
        with patch.object(cost, "MAX_DATAFLOW_STEPS", 1), self.assertRaises(ValueError):
            cost.analyze(inventory, events, context)
        inventory["graphs"][0]["inventory"]["nodes"].append(node("other_weights", "Softmax", ["scale_out"]))
        report = cost.analyze(inventory, events, context)
        self.assertEqual([row["kind"] for row in report["repeat_kv_groups"]], ["value"])

    def test_observed_role_mismatch_is_explicit_and_malformed_scope_is_rejected(self):
        inventory, events, context = fixture()
        context["role"] = "critic"
        report = cost.analyze(inventory, events, context)
        self.assertEqual(report["attributed_role"], "proposer")
        self.assertFalse(report["role_context_matches_observation"])
        inventory["graphs"].append(None)
        with self.assertRaises(ValueError):
            cost.analyze(inventory, events, context)
        inventory, events, context = fixture()
        scope = inventory["graphs"][0]["inventory"]
        scope["subgraphs"] = [{"path": scope["path"], "nodes": [], "subgraphs": []}]
        with self.assertRaises(ValueError):
            cost.analyze(inventory, events, context)

    def test_bad_duration_provider_and_context_cannot_be_success(self):
        for duration in (-1, True, float("nan")):
            inventory, events, context = fixture()
            events[0]["dur"] = duration
            with self.assertRaises(ValueError):
                cost.analyze(inventory, events, context)
        inventory, events, context = fixture("unknown_provider")
        self.assertEqual(cost.analyze(inventory, events, context)["matching_summary"]["matched"], 0)
        for field, value in (("batch", True), ("records", 129), ("memory_tokens", 70), ("graph_sha256", "b" * 64)):
            malformed = {**context, field: value}
            with self.assertRaises(ValueError):
                cost.analyze(inventory, events, malformed)

    def test_duplicate_names_json_keys_nonfinite_and_deep_scopes_are_rejected(self):
        for raw in (b'{"nodes":1,"nodes":2}', b'{"duration":NaN}'):
            with self.assertRaises(ValueError):
                cost.decode(raw)
        inventory, events, context = fixture()
        scope = inventory["graphs"][0]["inventory"]
        scope["nodes"].append(copy.deepcopy(scope["nodes"][0]))
        with self.assertRaises(ValueError):
            cost.analyze(inventory, events, context)
        inventory, events, context = fixture()
        scope = inventory["graphs"][0]["inventory"]
        for index in range(10):
            child = {"path": str(index), "nodes": [], "subgraphs": []}
            scope["subgraphs"] = [child]
            scope = child
        with self.assertRaises(ValueError):
            cost.analyze(inventory, events, context)

    def test_input_budget_is_combined_and_does_not_read_large_second_file(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory = Path(directory) / "inventory.json"
            profile = Path(directory) / "profile.json"
            inventory.write_bytes(b'{"a":1}')
            profile.write_bytes(b'{"b":2}')
            with patch.object(cost, "MAX_INPUT_BYTES", 10), self.assertRaises(ValueError):
                cost.read_inputs(inventory, profile)

    def test_cli_explicit_inputs_and_new_report_preserve_pins_and_scope(self):
        inventory, events, context = fixture()
        with tempfile.TemporaryDirectory() as directory:
            paths = {name: Path(directory) / (name + ".json") for name in ("inventory", "profile", "report")}
            paths["inventory"].write_text(json.dumps(inventory), encoding="utf-8")
            paths["profile"].write_text(json.dumps(events), encoding="utf-8")
            arguments = ["--inventory", str(paths["inventory"]), "--profile", str(paths["profile"]),
                         "--output", str(paths["report"])]
            for name, value in context.items():
                arguments += ["--" + name.replace("_", "-"), str(value)]
            self.assertEqual(cost.main(arguments), 0)
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["context"], context)
            self.assertEqual(report["inputs"]["inventory"]["sha256"], hashlib.sha256(paths["inventory"].read_bytes()).hexdigest())
            self.assertFalse(report["parser_nn_execution"])
            self.assertFalse(report["parser_gpu_execution"])
            with self.assertRaises(ValueError):
                cost.main(arguments)


if __name__ == "__main__":
    unittest.main()
