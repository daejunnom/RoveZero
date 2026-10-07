"""Opset-17 P/C graph with one shared reader bank and hard ONNX If routing.

This is an export representation of the existing Torch forward, not a different
model. Shared attention and move embedding run outside If. Private initial
state, FFNs and heads run only inside the selected branch. No Where-based router
or both-role forward is used. Serialized sharing does not prove ORT prepacking
or device allocation sharing; those require native execution evidence.
"""
import hashlib

ARTIFACT_SCHEMA = "rovezero.pals-model.v2"
SHARED_LAYOUT = "shared_pc_if"
SHARED_LAYOUT_REVISION = 1


class _Bank:
    def __init__(self, model):
        import numpy as np
        from onnx import numpy_helper
        self.np, self.numpy_helper = np, numpy_helper
        self.state = model.state_dict()
        self.initializers, self.values, self.weights = [], {}, []
        self.serial = 0

    def unique(self, stem):
        self.serial += 1
        return f"{stem}_{self.serial}"

    def constant(self, value, dtype, stem="constant"):
        array = self.np.asarray(value, dtype=dtype)
        key = (array.dtype.str, array.shape, array.tobytes())
        if key not in self.values:
            name = self.unique(stem)
            self.values[key] = name
            self.initializers.append(self.numpy_helper.from_array(array, name))
        return self.values[key]

    def weight(self, parameter, transpose=False):
        name = parameter + (".linear_transpose" if transpose else "")
        if name in self.values:
            return self.values[name]
        source = self.state[parameter].detach().cpu().numpy()
        array = self.np.ascontiguousarray(source.T if transpose else source)
        self.values[name] = name
        self.initializers.append(self.numpy_helper.from_array(array, name))
        self.weights.append({"parameter": parameter, "initializer": name,
                             "bytes": int(array.nbytes),
                             "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                             "source_parameter_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                             "shape": list(array.shape)})
        return name


class _Graph:
    def __init__(self, bank, config, name):
        self.bank, self.config, self.name, self.nodes = bank, config, name, []

    def op(self, operation, *inputs, stem=None, **attributes):
        from onnx import helper
        name = self.bank.unique(self.name + "_" + (stem or operation.lower()))
        self.nodes.append(helper.make_node(operation, list(inputs), [name], name=name, **attributes))
        return name

    def ints(self, values):
        return self.bank.constant(values, self.bank.np.int64)

    def floats(self, values):
        return self.bank.constant(values, self.bank.np.float32)

    def unsqueeze(self, value, axes):
        return self.op("Unsqueeze", value, self.ints(axes))

    def linear(self, value, parameter):
        return self.op("MatMul", value, self.bank.weight(parameter, transpose=True))

    def norm(self, value, parameter):
        mean = self.op("ReduceMean", self.op("Mul", value, value), axes=[-1], keepdims=1)
        inverse = self.op("Reciprocal", self.op("Sqrt", self.op("Add", mean, self.floats(1e-6))))
        return self.op("Mul", self.op("Mul", value, inverse), self.bank.weight(parameter))

    def dimension(self, value, axis):
        return self.op("Gather", self.op("Shape", value), self.ints([axis]), axis=0)

    def shape(self, *parts):
        return self.op("Concat", *parts, axis=0)

    def kv(self, value, batch, tokens):
        shaped = self.op("Reshape", value, self.shape(batch, tokens, self.ints([self.config.kv_heads, self.config.head_dimension])))
        return self.op("Transpose", shaped, perm=[0, 2, 1, 3])

    def repeat_kv(self, value, batch, tokens):
        groups = self.config.query_heads // self.config.kv_heads
        expanded = self.op("Expand", self.unsqueeze(value, [2]),
                           self.shape(batch, self.ints([self.config.kv_heads, groups]), tokens, self.ints([self.config.head_dimension])))
        return self.op("Reshape", expanded,
                       self.shape(batch, self.ints([self.config.query_heads]), tokens, self.ints([self.config.head_dimension])))

    def attention(self, latent, key, value, mask, prefix):
        batch, tokens = self.dimension(latent, 0), self.dimension(latent, 1)
        query = self.linear(latent, prefix + ".query.weight")
        query = self.op("Transpose", self.op("Reshape", query,
                        self.shape(batch, tokens, self.ints([self.config.query_heads, self.config.head_dimension]))), perm=[0, 2, 1, 3])
        memory_tokens = self.dimension(key, 2)
        key, value = self.repeat_kv(key, batch, memory_tokens), self.repeat_kv(value, batch, memory_tokens)
        scores = self.op("Mul", self.op("MatMul", query, self.op("Transpose", key, perm=[0, 1, 3, 2])),
                         self.floats(self.config.head_dimension ** -0.5))
        if mask is not None:
            # This Where applies the attention mask only. It is not role routing.
            scores = self.op("Where", self.unsqueeze(mask, [1, 2]), scores, self.floats(-1e9))
        attended = self.op("MatMul", self.op("Softmax", scores, axis=-1), value)
        attended = self.op("Reshape", self.op("Transpose", attended, perm=[0, 2, 1, 3]),
                           self.shape(batch, tokens, self.ints([self.config.width])))
        return self.linear(attended, prefix + ".output.weight")

    def reader(self, latent, index):
        prefix = f"reader_blocks.{index}"
        normalized = self.norm(latent, prefix + ".cross_norm.weight")
        latent = self.op("Add", latent, self.attention(normalized, "memory_key", "memory_value", "memory_mask", prefix + ".cross"))
        normalized = self.norm(latent, prefix + ".self_norm.weight")
        batch, tokens = self.dimension(normalized, 0), self.dimension(normalized, 1)
        key = self.kv(self.linear(normalized, prefix + ".self_attn.key.weight"), batch, tokens)
        value = self.kv(self.linear(normalized, prefix + ".self_attn.value.weight"), batch, tokens)
        return self.op("Add", latent, self.attention(normalized, key, value, None, prefix + ".self_attn"))

    def private_initial(self, role):
        prefix = f"experts.{role}"
        latent = self.unsqueeze(self.bank.weight(prefix + ".initial_latent"), [0])
        query = self.unsqueeze(self.linear("query", prefix + ".query_projection.weight"), [1])
        return self.op("Add", latent, query)

    def private_ffn(self, latent, role, index):
        prefix = f"experts.{role}"
        value = self.norm(latent, prefix + f".ffn_norms.{index}.weight")
        gate = self.linear(value, prefix + f".ffns.{index}.gate.weight")
        gate = self.op("Mul", gate, self.op("Sigmoid", gate))
        value = self.op("Mul", gate, self.linear(value, prefix + f".ffns.{index}.up.weight"))
        return self.op("Add", latent, self.linear(value, prefix + f".ffns.{index}.down.weight"))

    def candidate_embedding(self):
        values = []
        for index, name in enumerate(("from_square", "to_square", "promotion")):
            codes = self.op("Gather", "candidates", self.ints(index), axis=2)
            values.append(self.op("Gather", self.bank.weight(f"move_embedding.{name}.weight"), codes, axis=0))
        return self.linear(self.op("Concat", *values, axis=-1), "move_embedding.projection.weight")

    def private_heads(self, latent, embedded, role):
        prefix = f"experts.{role}"
        context = self.op("ReduceMean", latent, axes=[1], keepdims=0)
        expanded = self.unsqueeze(context, [1])
        logits = self.op("Squeeze", self.linear(self.op("Mul", embedded, expanded), prefix + ".policy_head.weight"), self.ints([-1]))
        logits = self.op("Where", "candidate_mask", logits, self.floats(-1e9))
        wdl = self.linear(context, prefix + ".wdl_head.weight")
        if role == "critic":
            divergences = self.linear("divergence_features", prefix + ".divergence_projection.weight")
            divergences = self.op("Squeeze", self.linear(self.op("Mul", divergences, expanded), prefix + ".divergence_head.weight"), self.ints([-1]))
            divergences = self.op("Where", "divergence_mask", divergences, self.floats(-1e9))
        else:
            # Inactive C projection/head is absent from this branch entirely.
            divergences = self.op("ConstantOfShape", self.op("Shape", "divergence_mask"))
        return logits, wdl, divergences

    def branch(self, values, shapes, suffix):
        from onnx import TensorProto, helper
        outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in zip(values, shapes)]
        return helper.make_graph(self.nodes, self.name + suffix, [], outputs)

    def routed(self, method, shapes, *arguments):
        from onnx import helper
        branches = {}
        for role in ("critic", "proposer"):
            branch = _Graph(self.bank, self.config, role)
            values = getattr(branch, method)(*arguments, role) if method == "private_heads" else (
                branch.private_ffn(arguments[0], role, arguments[1]) if method == "private_ffn" else branch.private_initial(role))
            if not isinstance(values, tuple): values = (values,)
            branches[role] = branch.branch(values, shapes, method)
        names = [self.bank.unique("selected_" + method) for _ in shapes]
        self.nodes.append(helper.make_node("If", ["role_is_critic"], names,
                          name=self.bank.unique("route_" + method), then_branch=branches["critic"], else_branch=branches["proposer"]))
        return names[0] if len(names) == 1 else tuple(names)


def build_shared_pc_graph(model):
    """Return the checked, two-role graph and serialized-ownership evidence."""
    from onnx import TensorProto, checker, helper
    model.config.validate()
    if set(model.experts) != {"proposer", "critic"}:
        raise ValueError("shared_pc_if requires exactly the V-free P/C model")
    config = model.config
    bank = _Bank(model)
    graph = _Graph(bank, config, SHARED_LAYOUT)
    latent_shape = ["batch", config.latent_slots, config.width]
    latent = graph.routed("private_initial", [latent_shape])
    for _ in range(config.iterations):
        for index in range(config.recurrent_blocks):
            latent = graph.reader(latent, index)
            latent = graph.routed("private_ffn", [latent_shape], latent, index)
    embedded = graph.candidate_embedding()
    logits, wdl, divergences = graph.routed("private_heads", [["batch", "candidates"], ["batch", 3], ["batch", "divergences"]], latent, embedded)
    for value, name in ((logits, "candidate_logits"), (wdl, "wdl_logits"), (latent, "private_latent"), (divergences, "divergence_logits"), ("role_is_critic", "is_critic")):
        graph.nodes.append(helper.make_node("Identity", [value], [name], name="output_" + name))
    descriptors = [("role_is_critic", TensorProto.BOOL, []),
                   ("memory_key", TensorProto.FLOAT, ["batch", config.kv_heads, "memory_tokens", config.head_dimension]),
                   ("memory_value", TensorProto.FLOAT, ["batch", config.kv_heads, "memory_tokens", config.head_dimension]),
                   ("memory_mask", TensorProto.BOOL, ["batch", "memory_tokens"]),
                   ("candidates", TensorProto.INT64, ["batch", "candidates", 3]),
                   ("candidate_mask", TensorProto.BOOL, ["batch", "candidates"]),
                   ("divergence_features", TensorProto.FLOAT, ["batch", "divergences", 8]),
                   ("divergence_mask", TensorProto.BOOL, ["batch", "divergences"]),
                   ("query", TensorProto.FLOAT, ["batch", 16])]
    outputs = [("candidate_logits", TensorProto.FLOAT, ["batch", "candidates"]),
               ("wdl_logits", TensorProto.FLOAT, ["batch", 3]), ("private_latent", TensorProto.FLOAT, latent_shape),
               ("divergence_logits", TensorProto.FLOAT, ["batch", "divergences"]), ("is_critic", TensorProto.BOOL, [])]
    document = helper.make_model(helper.make_graph(graph.nodes, SHARED_LAYOUT,
                    [helper.make_tensor_value_info(*value) for value in descriptors],
                    [helper.make_tensor_value_info(*value) for value in outputs], bank.initializers),
                    producer_name="rz-pals-model", opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    checker.check_model(document, full_check=True)
    evidence = audit_shared_pc_graph(document, bank.weights)
    return document, evidence


def audit_shared_pc_graph(document, weights):
    """Fail closed on accidental shared copies or eager private computation."""
    shared = [value for value in weights if value["parameter"].startswith(("reader_blocks.", "move_embedding."))]
    names = [value["initializer"] for value in shared]
    outer_names = [value.name for value in document.graph.initializer]
    if len(names) != len(set(names)) or any(outer_names.count(name) != 1 for name in names):
        raise ValueError("reader/move initializer bank is not single-owned")
    if any("validator" in name for name in outer_names):
        raise ValueError("V-free graph contains validator weights")
    routes = [node for node in document.graph.node if node.op_type == "If"]
    if len(routes) != 6 or any(list(node.input) != ["role_is_critic"] for node in routes):
        raise ValueError("hard-route count/condition mismatch")
    branch_reports = []
    for node in document.graph.node:
        if any(value.startswith("experts.") for value in node.input):
            raise ValueError("private weights are used eagerly outside If")
        for attribute in node.attribute:
            if attribute.name not in ("then_branch", "else_branch"): continue
            branch = attribute.g
            if branch.initializer:
                raise ValueError("branch-local initializer copies are forbidden")
            for inner in branch.node:
                if any(value in names for value in inner.input):
                    raise ValueError("shared attention/embedding operation moved into role branch")
            branch_reports.append({"route": node.name, "branch": attribute.name, "nodes": len(branch.node)})
    return {"scope": "onnx_serialized_initializers_only", "layout_revision": SHARED_LAYOUT_REVISION,
            "shared_parameters": shared, "unique_shared_parameters": len(shared),
            "shared_weight_bytes": sum(value["bytes"] for value in shared),
            "if_routes": len(routes), "branch_local_initializers": 0,
            "branch_operations": branch_reports, "inactive_private_execution": "requires_runtime_profile",
            "ort_prepack_copies": "unknown", "device_residency_sharing": "unknown"}
