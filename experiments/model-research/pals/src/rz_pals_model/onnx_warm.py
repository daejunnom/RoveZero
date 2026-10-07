"""Explicit approximate private seed entry point, separate from legacy exports.

The checked legacy P/C graph is the template. Only its initial-latent selector
is wrapped in another ONNX If: Fresh executes the original initialization;
Warm takes the full accepted final latent without adding query a second time.
The unchanged readers recompute private self-attention K/V. This serialized
graph cannot certify Rules eligibility, accepted output, or physical completion;
those remain the Rust owner's responsibilities.
"""
import copy
import hashlib

from .onnx_shared import build_shared_pc_graph

WARM_ARTIFACT_SCHEMA = "rovezero.pals-private-warm.v1"
WARM_LAYOUT = "shared_pc_if_approx_warm_v1"
WARM_LAYOUT_REVISION = 1
WARM_GRAPH_SEMANTICS = (
    "rz-pals-private-native-warm/1;accepted-final-latent;no-query-double-add;"
    "fixed-2-iterations;private-self-kv-recomputed;fp32;pc-only"
)
PRIVATE_SEED_POLICY = (
    "rz-pals-private-approx-warm/1;fp32-bits;same-exact-rules-context;"
    "role-isolated;accepted-seed-only;no-exact-cache"
)
WARM_MODE_NODE = "private_warm_mode"


def _branches(node):
    result = {attribute.name: attribute.g for attribute in node.attribute
              if attribute.name in ("then_branch", "else_branch")}
    if set(result) != {"then_branch", "else_branch"} or len(node.attribute) != 2:
        raise ValueError("warm If requires exactly two branches")
    return result


def _walk(graph):
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.name in ("then_branch", "else_branch"):
                yield from _walk(attribute.g)


def _initial_route(document):
    routes = [(index, node) for index, node in enumerate(document.graph.node)
              if node.op_type == "If" and node.name.startswith("route_private_initial_")]
    if len(routes) != 1:
        raise ValueError("legacy private initial route is not unique")
    return routes[0]


def build_warm_pc_graph(model):
    """Return a new opt-in graph; never mutate legacy graph/model/checkpoint."""
    from onnx import TensorProto, checker, helper
    legacy, legacy_evidence = build_shared_pc_graph(model)
    document = copy.deepcopy(legacy)
    index, initial = _initial_route(document)
    selected = initial.output[0]
    fresh_name, warm_name = selected + "__fresh_branch", selected + "__warm_branch"
    fresh = copy.deepcopy(initial)
    fresh.output[0] = fresh_name
    shape = ["batch", model.config.latent_slots, model.config.width]
    fresh_branch = helper.make_graph(
        [fresh], "private_fresh_initial", [],
        [helper.make_tensor_value_info(fresh_name, TensorProto.FLOAT, shape)])
    warm_branch = helper.make_graph(
        [helper.make_node("Identity", ["initial_latent"], [warm_name],
                          name="consume_complete_private_seed")],
        "private_approx_warm_initial", [],
        [helper.make_tensor_value_info(warm_name, TensorProto.FLOAT, shape)])
    document.graph.node[index].CopyFrom(helper.make_node(
        "If", ["warm_start"], [selected], name=WARM_MODE_NODE,
        then_branch=warm_branch, else_branch=fresh_branch))
    document.graph.input.extend([
        helper.make_tensor_value_info("initial_latent", TensorProto.FLOAT, shape),
        helper.make_tensor_value_info("warm_start", TensorProto.BOOL, []),
    ])
    document.graph.name = WARM_LAYOUT
    checker.check_model(document, full_check=True)
    return document, audit_warm_pc_graph(document, legacy, legacy_evidence)


def audit_warm_pc_graph(document, legacy, legacy_evidence):
    """Strict template equivalence plus recursive role/seed routing audit.

    Comparing complete initializer and node protobufs prevents approximate
    warm selection from silently moving, copying, or changing shared weights.
    The legacy six-route auditor remains unchanged and owns the base audit.
    This proves serialized ownership, never ORT prepacking/device residency.
    """
    from onnx import TensorProto, helper
    index, initial = _initial_route(legacy)
    if legacy_evidence.get("if_routes") != 6 or legacy_evidence.get("branch_local_initializers") != 0:
        raise ValueError("warm template lacks the checked legacy route evidence")
    if document.graph.name != WARM_LAYOUT or len(document.graph.node) != len(legacy.graph.node):
        raise ValueError("warm graph layout/node count mismatch")
    for field in ("functions", "opset_import", "training_info"):
        if [value.SerializeToString() for value in getattr(document, field)] != [value.SerializeToString() for value in getattr(legacy, field)]:
            raise ValueError("warm graph changed legacy function/opset/training domain")
    if document.ir_version != legacy.ir_version:
        raise ValueError("warm graph changed legacy IR version")
    for field in ("sparse_initializer", "value_info", "quantization_annotation"):
        if [value.SerializeToString() for value in getattr(document.graph, field)] != [value.SerializeToString() for value in getattr(legacy.graph, field)]:
            raise ValueError("warm graph changed legacy sparse bank/type annotation")
    for node_index, (actual, expected) in enumerate(zip(document.graph.node, legacy.graph.node)):
        if node_index == index:
            continue
        if actual.SerializeToString() != expected.SerializeToString():
            raise ValueError("warm export changed legacy recurrent/head operation")
    actual_bank = [value.SerializeToString() for value in document.graph.initializer]
    legacy_bank = [value.SerializeToString() for value in legacy.graph.initializer]
    if actual_bank != legacy_bank:
        raise ValueError("warm initializer bank must remain single-owned and unchanged")
    if any("validator" in node.name or any("validator" in value for value in node.input)
           for node in _walk(document.graph)):
        raise ValueError("warm graph is V-free")
    expected_inputs = [value.SerializeToString() for value in legacy.graph.input] + [
        helper.make_tensor_value_info("initial_latent", TensorProto.FLOAT, ["batch", 16, 384]).SerializeToString(),
        helper.make_tensor_value_info("warm_start", TensorProto.BOOL, []).SerializeToString(),
    ]
    if [value.SerializeToString() for value in document.graph.input] != expected_inputs:
        raise ValueError("warm graph input shape/dtype/mode mismatch")
    if [value.SerializeToString() for value in document.graph.output] != [value.SerializeToString() for value in legacy.graph.output]:
        raise ValueError("warm graph changed legacy output meaning")
    mode = document.graph.node[index]
    selected = initial.output[0]
    if mode.name != WARM_MODE_NODE or mode.op_type != "If" or mode.domain or getattr(mode, "overload", "") or list(mode.input) != ["warm_start"] or list(mode.output) != [selected]:
        raise ValueError("warm mode requires hard scalar If routing")
    branches = _branches(mode)
    warm, fresh = branches["then_branch"], branches["else_branch"]
    if warm.input or fresh.input or warm.initializer or fresh.initializer or warm.sparse_initializer or fresh.sparse_initializer or len(warm.node) != 1 or len(fresh.node) != 1:
        raise ValueError("warm mode branch contains copied bank or extra computation")
    seed_node = warm.node[0]
    if seed_node.op_type != "Identity" or seed_node.domain or getattr(seed_node, "overload", "") or list(seed_node.input) != ["initial_latent"] or list(seed_node.output) != [selected + "__warm_branch"] or seed_node.attribute:
        raise ValueError("warm branch must consume the whole seed without query addition")
    expected_initial = copy.deepcopy(initial)
    expected_initial.output[0] = selected + "__fresh_branch"
    if fresh.node[0].SerializeToString() != expected_initial.SerializeToString():
        raise ValueError("Fresh branch changed legacy initialization")
    for branch, suffix in ((warm, "__warm_branch"), (fresh, "__fresh_branch")):
        expected = helper.make_tensor_value_info(selected + suffix, TensorProto.FLOAT, ["batch", 16, 384])
        if len(branch.output) != 1 or branch.output[0].SerializeToString() != expected.SerializeToString():
            raise ValueError("warm branch result shape mismatch")
    role_routes = [node for node in _walk(document.graph)
                   if node.op_type == "If" and list(node.input) == ["role_is_critic"]]
    all_routes = [node for node in _walk(document.graph) if node.op_type == "If"]
    if len(role_routes) != 6 or len(all_routes) != 7:
        raise ValueError("warm recursive role/mode route count mismatch")
    for route in all_routes:
        if any(branch.initializer or branch.sparse_initializer for branch in _branches(route).values()):
            raise ValueError("warm branch-local initializer copies are forbidden")
    return {
        "scope": "onnx_serialized_initializers_only",
        "layout_revision": WARM_LAYOUT_REVISION,
        "legacy_graph_sha256": hashlib.sha256(legacy.SerializeToString()).hexdigest(),
        "legacy_ownership": copy.deepcopy(legacy_evidence),
        "recursive_role_if_routes": 6, "warm_mode_if_routes": 1,
        "branch_local_initializers": 0, "seed_values_per_batch_row": 6144,
        "fresh_initialization": "unchanged_legacy_role_if",
        "warm_initialization": "complete_final_latent_identity_no_query_addition",
        "private_self_kv": "recomputed_by_unchanged_readers",
        "accepted_seed_and_rules_context": "requires_native_owner_validation",
        "inactive_private_execution": "requires_runtime_profile",
        "ort_prepack_copies": "unknown", "device_residency_sharing": "unknown",
    }
