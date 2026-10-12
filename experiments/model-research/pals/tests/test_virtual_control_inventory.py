"""V3 metadata-only fixtures; Torch/ONNX/NumPy/ORT import and NN are blocked."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rz_pals_model import virtual_control_inventory as inventory


def value(name, dtype=1, shape=("batch", 16, 384)):
    dims = []
    for item in shape:
        symbolic = isinstance(item, str)
        dims.append(NS(dim_param=item if symbolic else "", dim_value=0 if symbolic else item,
            HasField=lambda field, symbolic=symbolic: field == ("dim_param" if symbolic else "dim_value")))
    tensor = NS(elem_type=dtype, shape=NS(dim=dims), HasField=lambda field: field == "shape")
    return NS(name=name, type=NS(tensor_type=tensor, HasField=lambda field: field == "tensor_type"))


class TensorMetadataOnly:
    def __init__(self, name="index", dtype=7, dims=()):
        self.name, self.data_type, self.dims = name, dtype, dims
        self.external_data, self.data_location = [], 0

    @property
    def raw_data(self):
        raise AssertionError("Tensor contents are outside metadata-only inspection")


def node(name, op, inputs, outputs, attrs=()):
    return NS(name=name, op_type=op, domain="", overload="", input=list(inputs), output=list(outputs), attribute=list(attrs))


def graph(nodes=(), inputs=(), outputs=(), initializers=(), name="fixture"):
    return NS(name=name, node=list(nodes), input=list(inputs), output=list(outputs),
              initializer=list(initializers), sparse_initializer=[])


def branch(name, body):
    return NS(name=name, type=5, g=body)


def route(kind, serial, output=None):
    name = "route_private_" + kind + "_" + str(serial)
    branches = []
    for key, expert in (("then_branch", "critic"), ("else_branch", "proposer")):
        result = name + "_" + expert
        body = graph([node(result + "_nn", "MatMul", ["query", "experts." + expert + ".weight"], [result])],
                     outputs=[value(result)])
        branches.append(branch(key, body))
    return node(name, "If", ["role_is_critic"], [output or "selected_" + str(serial)], branches)


def private_graph(warm=False):
    identity = node("output_is_critic", "Identity", ["role_is_critic"], ["is_critic"])
    suffix = [route("ffn", serial) for serial in range(2, 6)] + [route("heads", 6), identity]
    inputs = [value("role_is_critic", 9, ()), value("query")]
    weights = [TensorMetadataOnly("experts." + expert + ".weight", 1, (2, 2)) for expert in ("critic", "proposer")]
    if not warm:
        nodes = [route("initial", 1), *suffix]
    else:
        selected = "selected_private_initial_7"
        seed = graph([node("consume_complete_virtual_private_seed", "Identity", ["initial_latent"], [selected + "__warm_branch"])],
                     outputs=[value(selected + "__warm_branch")])
        prefix = [node("prepare_" + field, "Identity", ["query"], [field]) for field in
                  ("query_line_features", "relation_features", "relation_active_float", "query_anchor_features")]
        fresh = graph([*prefix, route("initial", 1, selected + "__fresh_branch")],
                      outputs=[value(selected + "__fresh_branch")])
        mode = node(inventory.WARM_MODE_NODE, "If", ["warm_start"], [selected],
                    [branch("then_branch", seed), branch("else_branch", fresh)])
        nodes = [mode, *suffix]
        inputs += [value("initial_latent"), value("warm_start", 9, ())]
    return graph(nodes, inputs, [value("is_critic", 9, ())], weights,
                 inventory.WARM_LAYOUT if warm else inventory.SHARED_LAYOUT)


def manifest(warm=False, validator=False):
    from rz_pals_model.artifacts import VIRTUAL_RULES_BOUNDS, VIRTUAL_RULES_FEATURES
    from rz_pals_model.config import ModelConfig, TASKS
    from rz_pals_model.virtual_review import RECORD_FEATURES, QUERY_FEATURES
    config = ModelConfig.for_profile("virtual_review_v1")
    profile, base = "rz-pals-rules-virtual-review-v1", "fixture_rules_semantics"
    metadata = ["metadata_" + str(i) for i in range(16)]
    divergence = ["divergence_" + str(i) for i in range(8)]
    fields = [profile, base, *metadata, *RECORD_FEATURES, *QUERY_FEATURES, *divergence, *VIRTUAL_RULES_FEATURES]
    digest = hashlib.sha256()
    for field in fields:
        encoded = field.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little")); digest.update(encoded)
    declaration = {"schema": "rovezero.pals-rules-descriptor.v3", "rules_input_profile": profile,
        "config": config.to_dict(), "metadata_features": metadata, "divergence_features": divergence,
        "record_features": list(RECORD_FEATURES), "query_features": list(QUERY_FEATURES),
        "virtual_review_features": VIRTUAL_RULES_FEATURES, "bounds": VIRTUAL_RULES_BOUNDS,
        "rules_base_semantics": base, "semantic_fields": fields,
        "semantic_digest_algorithm": "sha256_u64le_length_prefixed_utf8_fields",
        "rules_input_semantic_sha256": digest.hexdigest(), "rules_encoder_source_sha256": "11" * 32,
        "learned_input_compatibility": "unverified_declaration_only"}
    graph_roles = ["public", "shared_warm"] if warm else ["public", "shared_pc"] + (["validator"] if validator else [])
    from rz_pals_model.onnx_virtual_review import virtual_io_spec
    spec = virtual_io_spec(config)
    result = {"schema": "rovezero.pals-model.v3", "config": config.to_dict(),
        "model_semantics": config.model_semantics, "encoding_schema": config.encoding,
        "model_profile": config.profile, "checkpoint_sha256": "22" * 32,
        "trained": False, "training_steps": 0, "roles": ["proposer", "critic"] + (["validator"] if validator else []),
        "validator_present": validator, "precision": "fp32", "tf32": False,
        "layout": inventory.WARM_LAYOUT if warm else inventory.SHARED_LAYOUT, "layout_revision": 1,
        "task_names": list(TASKS), "physical_line_buckets": [8,16,32,64,128,256],
        "session_policy": "one_dynamic_L_graph_no_bucket_weight_copies",
        "rules_input_declaration": declaration, "rules_input_semantic_sha256": digest.hexdigest(),
        "rules_encoder_source_sha256": "11" * 32,
        "graphs": [{"role": role, "file": inventory.FILES[role], "sha256": "33" * 32, "opset": 17,
                    **{key: [{"name": name, "dtype": dtype, "shape": list(shape)} for name, dtype, shape in spec[role][key]]
                       for key in ("inputs", "outputs")}} for role in graph_roles]}
    if warm:
        result.update(approximate=True, warm_seed_policy="accepted_final_latent_same_model_role_rules_query_epoch_physical_fence")
    return result


def document(body, role, descriptor):
    keys = ("schema", "layout", "model_semantics", "encoding_schema", "model_profile", "checkpoint_sha256",
            "rules_input_profile", "rules_input_semantic_sha256", "rules_encoder_source_sha256")
    props = {key: descriptor[key] for key in keys if key in descriptor}; props["role"] = role
    return NS(graph=body, functions=[], training_info=[], ir_version=10,
              opset_import=[NS(domain="", version=17)], metadata_props=[NS(key=key, value=item) for key, item in props.items()])


class VirtualInventoryTests(unittest.TestCase):
    def setUp(self):
        self.neural_imports = []
        original = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
                self.neural_imports.append(name)
                raise AssertionError("metadata fixture attempted neural import")
            return original(name, *args, **kwargs)
        self.blocker = patch("builtins.__import__", side_effect=blocked); self.blocker.start()

    def tearDown(self):
        self.blocker.stop(); self.assertEqual(self.neural_imports, [])

    def scan(self, body, role="shared_pc", warm=False):
        return inventory._scope(body, role, {}, inventory._State(role, warm, time.monotonic() + 5))

    def test_opt_in_and_finite_wall_required_before_any_file_read(self):
        with patch.object(inventory.legacy, "_external", side_effect=AssertionError("path inspected before admission")):
            with self.assertRaisesRegex(ValueError, "opt_in"):
                inventory.produce_virtual_control_inventory("not_read", "pin", "not_written")
            for wall in (0, 61, True, 1.5):
                with self.assertRaisesRegex(ValueError, "finite_metadata_wall"):
                    inventory.produce_virtual_control_inventory("not_read", "pin", "not_written", opt_in=True, max_seconds=wall)

    def test_schema_and_exact_v3_profile_and_role_sets(self):
        for warm, validator in ((False, False), (False, True), (True, False)):
            original = manifest(warm, validator)
            actual, observed = inventory._manifest(json.dumps(original).encode())
            self.assertEqual(actual, original); self.assertEqual(observed, warm)
            for mutate in (lambda m: m.update(schema="rovezero.pals-model.v2"),
                           lambda m: m.update(graphs=[None]),
                           lambda m: m["config"].update(max_records=128),
                           lambda m: m["graphs"][0].update(file="arbitrary.onnx"),
                           lambda m: m["graphs"].reverse()):
                bad = copy.deepcopy(original); mutate(bad)
                with self.assertRaises(ValueError): inventory._manifest(json.dumps(bad).encode())
        for raw in (b'{"schema":1,"schema":2}', b'{"schema":NaN}'):
            with self.assertRaises(ValueError): inventory._manifest(raw)
        self.assertEqual(inventory.legacy.SCHEMA, "rovezero.pals-static-control-inventory.v2")
        self.assertEqual(inventory.legacy.MAX_NODES, 5000)
        self.assertEqual(inventory.legacy.MAX_INVENTORY_BYTES, 1048576)
        self.assertEqual((inventory.MAX_NODES, inventory.MAX_GRAPH_NODES, inventory.MAX_INVENTORY_BYTES), (12000,5000,4194304))
        self.assertEqual(inventory._admission_limits(), {"max_manifest_bytes":131072,"max_graph_bytes":268435456,
            "max_inventory_bytes":4194304,"max_nodes":12000,"max_graph_nodes":5000,
            "max_initializers_per_scope":5000,"max_attributes_per_node":32,"max_recursive_depth":32,"max_path_bytes":4096})

    def test_fresh_six_routes_and_role_output_identity(self):
        body = private_graph(); result = self.scan(body)
        inventory._validate_routes("shared_pc", result, False)
        self.assertEqual(len(result["subgraphs"]), 12)
        self.assertTrue(all(child["nodes"][0]["actual_provider"] == "unknown" for child in result["subgraphs"]))
        for mutate in (lambda g: g.node.pop(2), lambda g: setattr(g.node[-1], "name", "another_identity"),
                       lambda g: setattr(g.node[0].attribute[0].g.node[0], "input", ["query", "experts.proposer.weight"]),
                       lambda g: g.node[0].attribute[0].g.initializer.append(TensorMetadataOnly("local"))):
            bad = private_graph(); mutate(bad)
            with self.assertRaises(ValueError): inventory._validate_routes("shared_pc", self.scan(bad), False)

    def test_warm_complete_fresh_prefix_and_seed_only_identity(self):
        body = private_graph(True); result = self.scan(body, "shared_warm", True)
        inventory._validate_routes("shared_warm", result, True)
        self.assertEqual(result["nodes"][0]["input_origins"], [inventory.WARM])
        self.assertEqual(result["nodes"][0]["static_annotation"], inventory.WARM_ANNOTATION)
        seed = result["subgraphs"][0]; fresh = result["subgraphs"][1]
        self.assertEqual(len(seed["nodes"]), 1); self.assertEqual(len(fresh["subgraphs"]), 2)
        for mutate in (lambda g: setattr(g.node[0], "name", "arbitrary_mode"),
                       lambda g: setattr(g.node[0].attribute[0].g.node[0], "name", "wrong_seed"),
                       lambda g: g.node[0].attribute[0].g.node.append(node("extra_seed_math", "Add", ["initial_latent", "initial_latent"], ["changed"])),
                       lambda g: g.node[0].attribute[1].g.node.pop(0),
                       lambda g: g.node[0].attribute[1].g.input.append(value("local")),
                       lambda g: setattr(g.node[0].attribute[0].g.output[0], "name", "wrong_output")):
            bad = private_graph(True); mutate(bad)
            with self.assertRaises(ValueError): inventory._validate_routes("shared_warm", self.scan(bad, "shared_warm", True), True)
        with self.assertRaises(ValueError): self.scan(body, "shared_warm", False)

    def test_no_range_or_float_slice_cpu_annotation(self):
        body = graph([node("shape", "Shape", ["query"], ["dimensions"]),
                      node("shape_gather", "Gather", ["dimensions", "index"], ["L"]),
                      node("float_ply_slice", "Slice", ["ply_embedding", "index", "L"], ["ply"]),
                      node("dynamic_range", "Range", ["index", "L", "step"], ["range"])],
                     [value("query")], initializers=[TensorMetadataOnly(), TensorMetadataOnly("step"), TensorMetadataOnly("ply_embedding", 1, (256,64))])
        result = self.scan(body, "public")
        self.assertEqual(result["nodes"][1]["static_annotation"], "shape_metadata_chain_candidate")
        for item in result["nodes"][2:]: self.assertEqual(item["static_annotation"], "none")
        self.assertEqual(result["nodes"][2]["input_origins"][0], inventory.DATA)
        self.assertTrue(all(item["actual_provider"] == "unknown" for item in result["nodes"]))

    def test_integral_metadata_size_and_external_storage(self):
        body = graph(initializers=[TensorMetadataOnly("scalar"), TensorMetadataOnly("too_many", 7, (65,)), TensorMetadataOnly("float_weight", 1, (2,2))])
        result = self.scan(body, "public")
        self.assertEqual([row["name"] for row in result["integral_initializers"]], ["scalar"])
        body.initializer[0].external_data = [NS(key="location", value="not_read")]
        with self.assertRaises(ValueError): self.scan(body, "public")

    def test_late_values_aliases_names_and_unregistered_bool_do_not_gain_control(self):
        body = private_graph(); body.node[0].attribute[0].g.node[0].input = ["later_shape", "experts.critic.weight"]
        body.node.append(node("late_shape", "Shape", ["query"], ["later_shape"]))
        result = self.scan(body)
        self.assertEqual(result["subgraphs"][0]["nodes"][0]["input_origins"][0], inventory.DATA)
        for mutate in (lambda g: setattr(g.node[-1], "output", ["query"]),
                       lambda g: setattr(g.node[-1], "name", g.node[0].name),
                       lambda g: setattr(g.input[0], "name", "arbitrary_bool"),
                       lambda g: setattr(g.input[0], "type", value("role_is_critic", 9, (1,)).type)):
            bad = private_graph(); mutate(bad)
            with self.assertRaises(ValueError): self.scan(bad)

    def test_node_graph_total_and_wall_limits(self):
        body = graph([node("one", "Identity", ["query"], ["out"])], [value("query")])
        state = inventory._State("public", False, time.monotonic() + 5); state.count = inventory.MAX_NODES
        with self.assertRaises(ValueError): inventory._scope(body, "public", {}, state)
        state = inventory._State("public", False, time.monotonic() + 5); state.graph_count = inventory.MAX_GRAPH_NODES
        with self.assertRaises(ValueError): inventory._scope(body, "public", {}, state)
        with self.assertRaisesRegex(ValueError, "wall_deadline"):
            inventory._scope(body, "public", {}, inventory._State("public", False, time.monotonic() - 1))

    def test_document_pins_exact_graph_name_abi_and_private_bank(self):
        source = manifest(); entry = source["graphs"][0]
        reverse = {name: number for number, name in inventory.legacy.DTYPES.items()}
        body = graph(inputs=[value(row["name"], reverse[row["dtype"]], row["shape"]) for row in entry["inputs"]],
                     outputs=[value(row["name"], reverse[row["dtype"]], row["shape"]) for row in entry["outputs"]],
                     name=inventory.PUBLIC_LAYOUT)
        record = document(body, "public", source); inventory._check_document(record, entry, source)
        body.name = "old_public_graph"
        with self.assertRaises(ValueError): inventory._check_document(record, entry, source)
        body.name = inventory.PUBLIC_LAYOUT; body.initializer.append(TensorMetadataOnly("experts.validator.weight", 1, (2,2)))
        with self.assertRaises(ValueError): inventory._check_document(record, entry, source)

    def test_weight_ledger_exact_shape_pin_and_v_private_direct_dependency(self):
        parameter = "experts.validator.task_head.weight"; initializer = parameter + ".linear_transpose"
        row = {"parameter": parameter, "initializer": initializer, "shape": [384,7], "bytes":384*7*4,
               "sha256":"aa"*32, "source_parameter_sha256":"bb"*32}
        body = graph([node("v_task", "MatMul", ["pooled",initializer], ["task_logits"])],
                     initializers=[TensorMetadataOnly(initializer,1,(384,7))])
        record = NS(graph=body); pinned = {"initializer_banks":{"validator":{"parameters":[row]}}}
        bindings = inventory._weight_bindings(record,"validator",pinned)
        self.assertEqual(bindings[0]["dtype"], "FLOAT")
        scope = self.scan(body,"validator")
        deps = inventory._private_dependencies(scope, bindings)
        self.assertEqual(deps, [{"scope_path":"validator","node_index":0,"node_name":"v_task","op":"MatMul",
                                "initializer":initializer,"parameter":parameter}])
        for mutate in (lambda p: p["initializer_banks"]["validator"]["parameters"][0].update(bytes=1),
                       lambda p: p["initializer_banks"]["validator"]["parameters"][0].update(shape=[7,384]),
                       lambda p: p["initializer_banks"]["validator"]["parameters"][0].update(sha256="unknown"),
                       lambda p: p["initializer_banks"]["validator"]["parameters"].append(copy.deepcopy(row))):
            bad = copy.deepcopy(pinned); mutate(bad)
            with self.assertRaises(ValueError): inventory._weight_bindings(record,"validator",bad)
        with self.assertRaises(ValueError): inventory._weight_bindings(record,"public",{"initializer_banks":{"public":{"parameters":[row]}}})
        body.initializer.append(TensorMetadataOnly("constant_3",1,()))
        self.assertEqual(inventory._weight_bindings(record,"validator",pinned),bindings)
        for name, shape in (("reader_blocks.0.query_projection.weight",(2,2)),("unregistered.weight",(2,2)),("constant_4",(2,2))):
            body.initializer.append(TensorMetadataOnly(name,1,shape))
            with self.assertRaisesRegex(ValueError,"missing_from_export_ledger"):
                inventory._weight_bindings(record,"validator",pinned)
            body.initializer.pop()

    def test_graph_byte_mismatch_precedes_onnx_import_and_output(self):
        with tempfile.TemporaryDirectory(prefix="rz-v3-control-") as directory:
            root = Path(directory); original = manifest()
            for entry in original["graphs"]: (root / entry["file"]).write_bytes(b"source-only-fake-protobuf")
            raw = json.dumps(original).encode(); source = root / "export.json"; source.write_bytes(raw)
            output = root / "inventory.json"
            with self.assertRaisesRegex(ValueError, "sha256_differs"):
                inventory.produce_virtual_control_inventory(source, hashlib.sha256(raw).hexdigest(), output, opt_in=True)
            self.assertFalse(output.exists())

    def test_exclusive_bounded_write_never_overwrites_or_claims_provider(self):
        with tempfile.TemporaryDirectory(prefix="rz-v3-control-") as directory:
            output = Path(directory) / "inventory.json"
            original = {"schema":inventory.SCHEMA,"total_nodes":1,"actual_provider_placement":"not_observed","cpu_allowlist":"not_created"}
            receipt = inventory._write_exclusive(output, original)
            saved = output.read_bytes()
            self.assertEqual(receipt["inventory_sha256"],hashlib.sha256(saved).hexdigest())
            self.assertEqual(receipt["cpu_allowlist"],"not_created")
            with self.assertRaises(ValueError): inventory._write_exclusive(output,original)
            self.assertEqual(output.read_bytes(),saved)
            with patch.object(inventory,"MAX_INVENTORY_BYTES",1):
                with self.assertRaisesRegex(ValueError,"output_bound"):
                    inventory._write_exclusive(Path(directory)/"not_created.json",original)


if __name__ == "__main__":
    unittest.main()
