"""Bounded stdlib/fake-protobuf metadata tests: no ONNX or NN imports/forward."""
import builtins
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rz_pals_model import cuda_control_inventory as inventory


def value(name, dtype=1, shape=("batch", 16)):
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
        raise AssertionError("Tensor values were read")


def node(name, op, inputs, outputs, attrs=()):
    return NS(name=name, op_type=op, domain="", overload="", input=list(inputs), output=list(outputs), attribute=list(attrs))


def graph(nodes, inputs=(), outputs=(), *, initializers=(), name="shared_pc_if"):
    return NS(name=name, node=list(nodes), input=list(inputs), output=list(outputs),
              initializer=list(initializers), sparse_initializer=[])


def branch_attr(name, branch):
    return NS(name=name, type=5, g=branch)


def role_graph():
    critic = graph([node("critic_nn", "Gemm", ["query"], ["critic_latent"])], outputs=[value("critic_latent")])
    proposer = graph([node("proposer_nn", "Gemm", ["query"], ["proposer_latent"])], outputs=[value("proposer_latent")])
    route = node("route_private_initial_9", "If", ["role_is_critic"], ["selected_private_initial_10"],
                 [branch_attr("then_branch", critic), branch_attr("else_branch", proposer)])
    return graph([node("shape", "Shape", ["query"], ["dims"]),
                  node("gather", "Gather", ["dims", "index"], ["batch_dim"]), route,
                  node("output_is_critic", "Identity", ["role_is_critic"], ["is_critic"])],
                 [value("role_is_critic", 9, ()), value("query")], [value("is_critic", 9, ())],
                 initializers=[TensorMetadataOnly(), TensorMetadataOnly("weight", 1, (16, 16))])


def manifest(warm=False):
    return {"schema": inventory.CUDA_WARM_SCHEMA if warm else "rovezero.pals-model.v2",
            "layout": inventory.CUDA_WARM_LAYOUT if warm else "shared_pc_if", "layout_revision": 2 if warm else 1,
            "roles": ["proposer", "critic"], "validator_present": False,
            "execution_domain": "cuda_device_io_binding_v2", "model_profile": "full_line_interaction_v2",
            "precision": "fp32", "tf32": False,
            "graphs": [{"role": role, "file": filename, "sha256": "11" * 32, "opset": 17, "inputs": [], "outputs": []}
                       for role, filename in [("public", "public_memory.onnx"), ("shared_pc", (inventory.CUDA_WARM_LAYOUT if warm else "shared_pc_if") + ".onnx")]]}


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.neural_imports = []
        original = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
                self.neural_imports.append(name)
                raise AssertionError("neural import in metadata fixture")
            return original(name, *args, **kwargs)

        self.blocker = patch("builtins.__import__", side_effect=blocked)
        self.blocker.start()

    def tearDown(self):
        self.blocker.stop()
        self.assertEqual(self.neural_imports, [])

    def scan(self, document, *, warm=False):
        return inventory._scope(document, "shared_pc", {}, inventory._State(warm, time.monotonic() + 5))

    def test_metadata_and_integral_provenance_do_not_read_float_tensor_values(self):
        result = self.scan(role_graph())
        self.assertEqual(result["initializer_count"], 2)
        self.assertEqual(len(result["integral_initializers"]), 1)
        self.assertEqual(result["nodes"][0]["static_annotation"], "shape_metadata_candidate")
        self.assertEqual(result["nodes"][1]["input_origins"], ["shape_metadata", "small_integral_constant"])
        self.assertEqual(result["nodes"][1]["static_annotation"], "shape_metadata_chain_candidate")
        self.assertEqual(result["subgraphs"][0]["nodes"][0]["static_annotation"], "major_nn_requires_cuda")
        self.assertEqual(result["subgraphs"][0]["nodes"][0]["actual_provider"], "unknown")

    def test_role_condition_is_scalar_bool_and_arbitrary_bool_if_is_refused(self):
        for change in ("vector", "float", "another_bool"):
            document = role_graph()
            if change == "vector":
                document.input[0] = value("role_is_critic", 9, (1,))
            elif change == "float":
                document.input[0] = value("role_is_critic", 1, ())
            else:
                document.input.append(value("another_bool", 9, ()))
                document.node[2].input = ["another_bool"]
            with self.assertRaises(ValueError):
                self.scan(document)

    def test_cuda_warm_mode_is_explicit_and_unregistered_warm_if_is_refused(self):
        legacy = role_graph()
        fresh = graph([legacy.node[2]], outputs=[value("selected_private_initial_10", 1, ("batch", 16, 384))])
        seed = graph([node("consume_complete_private_seed", "Identity", ["initial_latent"], ["seed_out"])],
                     outputs=[value("seed_out", 1, ("batch", 16, 384))])
        mode = node("private_warm_mode", "If", ["warm_start"], ["selected"],
                    [branch_attr("then_branch", seed), branch_attr("else_branch", fresh)])
        document = graph([mode], [*legacy.input, value("initial_latent", 1, ("batch", 16, 384)), value("warm_start", 9, ())])
        result = self.scan(document, warm=True)
        self.assertEqual(result["nodes"][0]["input_origins"], ["cuda_warm_mode_control"])
        self.assertEqual(result["nodes"][0]["static_annotation"], "cuda_warm_if_dispatch_candidate")
        self.assertEqual(result["subgraphs"][0]["nodes"][0]["static_annotation"], "none")
        self.assertEqual(result["subgraphs"][1]["nodes"][0]["input_origins"], ["role_control"])
        with self.assertRaises(ValueError):
            self.scan(document)
        mode.name = "another_mode"
        with self.assertRaises(ValueError):
            self.scan(document, warm=True)

    def test_if_branch_inherits_only_values_available_at_its_parent_node(self):
        document = role_graph()
        document.node[2].attribute[0].g.node[0].input = ["late_dimensions"]
        document.node.append(node("late_shape", "Shape", ["query"], ["late_dimensions"]))
        result = self.scan(document)
        self.assertEqual(result["subgraphs"][0]["nodes"][0]["input_origins"], ["model_data_or_unknown"])

    def test_node_names_aliases_and_initializer_shadows_are_refused(self):
        document = role_graph()
        document.node[1].name = document.node[0].name
        with self.assertRaises(ValueError):
            self.scan(document)
        document = role_graph()
        document.node[0].output = ["query"]
        with self.assertRaises(ValueError):
            self.scan(document)
        document = role_graph()
        document.initializer.append(TensorMetadataOnly("query"))
        with self.assertRaises(ValueError):
            self.scan(document)

    def test_unbounded_constant_is_data_and_cannot_become_shape_cpu_permission(self):
        document = role_graph()
        document.initializer[0].dims = (65,)
        result = self.scan(document)
        self.assertEqual(result["integral_initializers"], [])
        self.assertEqual(result["nodes"][1]["input_origins"], ["shape_metadata", "model_data_or_unknown"])
        self.assertEqual(result["nodes"][1]["static_annotation"], "none")

    def test_constant_tensor_seed_is_metadata_only_and_external_storage_is_refused(self):
        constant = node("constant", "Constant", [], ["index"],
                        [NS(name="value", type=4, t=TensorMetadataOnly())])
        document = graph([constant])
        result = self.scan(document)
        self.assertEqual(result["nodes"][0]["constant_integral_seeds"][0]["source"], "constant_node_tensor_metadata")
        constant.attribute[0].t.external_data = [NS(key="location", value="unopened")]
        with self.assertRaises(ValueError):
            self.scan(document)

    def test_recursion_node_and_scope_path_limits_are_finite(self):
        with patch.object(inventory, "MAX_RECURSIVE_DEPTH", 0):
            with self.assertRaises(ValueError):
                self.scan(role_graph())
        with patch.object(inventory, "MAX_NODES", 1):
            with self.assertRaises(ValueError):
                self.scan(role_graph())
        with patch.object(inventory, "MAX_PATH_BYTES", 1):
            with self.assertRaises(ValueError):
                self.scan(role_graph())

    def test_manifest_duplicate_nonfinite_and_unregistered_layout_are_refused(self):
        self.assertFalse(inventory._manifest(json.dumps(manifest()).encode())[1])
        self.assertTrue(inventory._manifest(json.dumps(manifest(True)).encode())[1])
        for raw in (b'{"schema":1,"schema":2}', b'{"schema":NaN}', b'[]'):
            with self.assertRaises(ValueError):
                inventory._manifest(raw)
        altered = manifest(True)
        altered["layout"] = "other_warm"
        with self.assertRaises(ValueError):
            inventory._manifest(json.dumps(altered).encode())

    def test_bad_raw_graph_pin_fails_before_onnx_import_and_creates_no_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            record = manifest(True)
            for entry in record["graphs"]:
                (root / entry["file"]).write_bytes(b"fake-pinned-graph-not-parsed")
            source = root / "manifest.json"
            raw = json.dumps(record).encode()
            source.write_bytes(raw)
            output = root / "inventory.json"
            with self.assertRaises(ValueError):
                inventory.produce_cuda_control_inventory(source, hashlib.sha256(raw).hexdigest(), output)
            self.assertFalse(output.exists())

    def test_raw_pin_size_and_input_symlink_are_checked_without_onnx(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "graph.onnx"
            source.write_bytes(b"bounded-bytes")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(inventory._read_pinned(source, digest, 32), b"bounded-bytes")
            for pin, limit in (("00" * 32, 32), (digest, 1)):
                with self.assertRaises(ValueError):
                    inventory._read_pinned(source, pin, limit)
            linked = root / "linked.onnx"
            try:
                linked.symlink_to(source)
            except OSError:
                return
            with self.assertRaises(ValueError):
                inventory._read_pinned(linked, digest, 32)

    def test_output_bound_and_exclusive_write_preserve_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "inventory.json"
            record = {"schema": inventory.SCHEMA, "total_nodes": 1}
            with patch.object(inventory, "MAX_INVENTORY_BYTES", 1):
                with self.assertRaises(ValueError):
                    inventory._write_exclusive(output, record)
            self.assertFalse(output.exists())
            receipt = inventory._write_exclusive(output, record)
            original = output.read_bytes()
            self.assertEqual(receipt["inventory_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(receipt["actual_provider_placement"], "not_observed")
            with self.assertRaises(ValueError):
                inventory._write_exclusive(output, record)
            self.assertEqual(output.read_bytes(), original)

    def test_graph_name_abi_and_model_metadata_must_match_manifest_pins(self):
        record = manifest()
        record.pop("execution_domain")
        record.pop("model_profile")
        document = NS(graph=role_graph(), functions=[], training_info=[], ir_version=10,
                      opset_import=[NS(domain="", version=17)], metadata_props=[
                          NS(key="schema", value=record["schema"]), NS(key="layout", value=record["layout"]),
                          NS(key="role", value="shared_pc")])
        entry = record["graphs"][1]
        entry["inputs"], entry["outputs"] = inventory._descriptors(document.graph.input), inventory._descriptors(document.graph.output)
        inventory._check_document(document, entry, record)
        document.graph.name = "unregistered_graph_name"
        with self.assertRaises(ValueError):
            inventory._check_document(document, entry, record)
        document.graph.name = record["layout"]
        document.metadata_props[-1].value = "critic"
        with self.assertRaises(ValueError):
            inventory._check_document(document, entry, record)

    def test_deadline_rank_and_unsupported_dtype_are_refused(self):
        with self.assertRaises(ValueError):
            inventory._scope(role_graph(), "shared_pc", {}, inventory._State(False, 0))
        with self.assertRaises(ValueError):
            inventory._descriptors([value("too_many_dimensions", shape=(1,) * 9)])
        with self.assertRaises(ValueError):
            inventory._descriptors([value("float16", 10)])
        with self.assertRaises(ValueError):
            inventory.produce_cuda_control_inventory(Path("/unused"), "00" * 32, Path("/unused-output"), max_seconds=0)


if __name__ == "__main__":
    unittest.main()
