"""Explicit opset-17 graphs for the registered virtual_review_v1 profile.

One public graph uses a shared board bank for every anchor and one dynamic-L
line bank. The P/C graph retains hard role routing and never executes both
private experts. Building these graphs reads a caller-owned parameter bank;
it does not run a Torch/ORT neural forward or an optimizer. Serialized
ownership is evidence about the ONNX artifact only, not device residency.
Legacy graph builders and their registered shape/digest domains are untouched.
"""
import copy
import hashlib
import os
from pathlib import Path

from .config import LINE_WIDTH, TASKS, VIRTUAL_REVIEW_PROFILE
from .onnx_shared import _Bank, _Graph, audit_shared_pc_graph

ARTIFACT_SCHEMA = "rovezero.pals-model.v3"
PUBLIC_LAYOUT = "virtual_review_public_v1"
SHARED_LAYOUT = "virtual_review_shared_pc_if_v1"
VALIDATOR_LAYOUT = "virtual_review_validator_fresh_v1"
WARM_LAYOUT = "virtual_review_shared_pc_if_approx_warm_v1"
WARM_MODE_NODE = "private_virtual_warm_mode"
LAYOUT_REVISION = 1
LINE_BUCKETS = (8, 16, 32, 64, 128, 256)
MAX_ANCHORS = 4
MAX_RECORDS = 64


def virtual_io_spec(config=None):
    """Return dependency-free declared I/O; actual values need owner validation.

    B, A, R, and L are physical axes. A and R must be positive; zero semantic
    records use a masked padding slot. The native owner validates allowed L
    buckets and all anchor/relation/move references before graph dispatch.
    A dynamic dimension declaration alone does not enforce those limits.
    """
    if config is not None:
        _validate_config(config)
    public_inputs = (
        ("anchor_board", "INT64", ("batch", "anchors", 64)),
        ("anchor_metadata", "FLOAT", ("batch", "anchors", 16)),
        ("anchor_mask", "BOOL", ("batch", "anchors")),
        ("records", "FLOAT", ("batch", "records", 8)),
        ("record_anchor", "INT64", ("batch", "records")),
        ("record_line_tokens", "INT64", ("batch", "records", "line_bucket", 3)),
        ("record_line_mask", "BOOL", ("batch", "records", "line_bucket")),
        ("record_mask", "BOOL", ("batch", "records")),
    )
    public_outputs = (
        ("memory_key", "FLOAT", ("batch", 2, "memory_tokens", 64)),
        ("memory_value", "FLOAT", ("batch", 2, "memory_tokens", 64)),
        ("memory_mask", "BOOL", ("batch", "memory_tokens")),
        ("anchor_context", "FLOAT", ("batch", "anchors", 384)),
    )
    role_inputs = (
        ("role_is_critic", "BOOL", ()),
        *public_outputs[:3],
        ("candidates", "INT64", ("batch", "candidates", 3)),
        ("candidate_mask", "BOOL", ("batch", "candidates")),
        ("divergence_features", "FLOAT", ("batch", "divergences", 8)),
        ("divergence_mask", "BOOL", ("batch", "divergences")),
        ("query", "FLOAT", ("batch", 8)),
        public_outputs[3],
        ("query_anchor", "INT64", ("batch", 3)),
        ("query_line_tokens", "INT64", ("batch", 3, "line_bucket", 3)),
        ("query_line_mask", "BOOL", ("batch", 3, "line_bucket")),
        ("record_relations", "INT64", ("batch", "records", 2)),
    )
    role_outputs = (
        ("candidate_logits", "FLOAT", ("batch", "candidates")),
        ("wdl_logits", "FLOAT", ("batch", 3)),
        ("private_latent", "FLOAT", ("batch", 16, 384)),
        ("divergence_logits", "FLOAT", ("batch", "divergences")),
        ("is_critic", "BOOL", ()),
    )
    validator_outputs = (*role_outputs[:3], ("task_logits", "FLOAT", ("batch", len(TASKS))))
    warm_inputs = (*role_inputs,
                   ("initial_latent", "FLOAT", ("batch", 16, 384)),
                   ("warm_start", "BOOL", ()))
    return {
        "schema": ARTIFACT_SCHEMA, "model_profile": VIRTUAL_REVIEW_PROFILE,
        "public": {"inputs": public_inputs, "outputs": public_outputs},
        "shared_pc": {"inputs": role_inputs, "outputs": role_outputs},
        "validator": {"inputs": role_inputs[1:], "outputs": validator_outputs},
        "shared_warm": {"inputs": warm_inputs, "outputs": role_outputs},
        "physical_limits": {"anchors": (1, MAX_ANCHORS), "records": (1, MAX_RECORDS),
                            "line_buckets": LINE_BUCKETS, "root_anchor": 0},
        "required_owner_checks": (
            "root_anchor_active", "active_anchor_references", "local_record_relations",
            "ordered_legal_candidates", "finite_features", "move_and_promotion_codes",
            "approved_line_bucket", "model_context_and_physical_lifetime",
        ),
    }


def _validate_config(config):
    config.validate()
    if config.profile != VIRTUAL_REVIEW_PROFILE or config.max_records != MAX_RECORDS:
        raise ValueError("virtual graph requires the registered virtual_review_v1 configuration")


class _VirtualGraph(_Graph):
    def line_encoder(self, tokens, mask):
        """The existing temporal/pooling math with a physical dynamic-L axis."""
        from onnx import TensorProto
        prefix = "public_encoder.line_encoder"
        batch, groups, length = (self.dimension(tokens, axis) for axis in (0, 1, 2))
        moves = self.op("Reshape", tokens, self.shape(self.ints([-1]), length, self.ints([3])))
        active = self.op("Reshape", mask, self.shape(self.ints([-1]), length))
        safe = self.op("Where", self.unsqueeze(active, [-1]), moves, self.ints(0))
        fields = [self.op("Gather", self.bank.weight(prefix + ".moves." + name + ".weight"),
                          self.op("Gather", safe, self.ints(index), axis=2), axis=0)
                  for index, name in enumerate(("from_square", "to_square", "promotion"))]
        value = self.linear(self.op("Concat", *fields, axis=-1), prefix + ".moves.projection.weight")
        positions = self.op("Slice", self.bank.weight(prefix + ".ply_embedding.weight"),
                            self.ints([0]), length, self.ints([0]))
        value = self.op("Add", value, self.unsqueeze(positions, [0]))
        active_float = self.op("Cast", active, to=TensorProto.FLOAT)
        expanded_mask = self.unsqueeze(active_float, [-1])
        value = self.op("Mul", value, expanded_mask)
        for index, dilation in enumerate((1, 2)):
            temporal = self.op("Conv", self.op("Transpose", value, perm=[0, 2, 1]),
                               self.bank.weight(prefix + f".temporal_blocks.{index}.weight"),
                               kernel_shape=[3], dilations=[dilation],
                               pads=[dilation, dilation], strides=[1])
            temporal = self.op("Transpose", temporal, perm=[0, 2, 1])
            value = self.op("Mul", self.op("Add", value, self.silu(temporal)), expanded_mask)
        scores = self.op("Squeeze", self.linear(value, prefix + ".pool_attention.weight"), self.ints([-1]))
        weights = self.op("Mul", self.op("Softmax", self.op("Where", active, scores, self.floats(-1e9)), axis=-1), active_float)
        pooled = self.op("ReduceSum", self.op("Mul", value, self.unsqueeze(weights, [-1])), self.ints([1]), keepdims=0)
        return self.op("Reshape", pooled, self.shape(batch, groups, self.ints([LINE_WIDTH])))

    def encoder_block(self, value, mask, prefix):
        normalized = self.norm(value, prefix + ".attention_norm.weight")
        batch, tokens = self.dimension(normalized, 0), self.dimension(normalized, 1)
        key = self.kv(self.linear(normalized, prefix + ".attention.key.weight"), batch, tokens)
        projected = self.kv(self.linear(normalized, prefix + ".attention.value.weight"), batch, tokens)
        value = self.op("Add", value, self.attention(normalized, key, projected, mask, prefix + ".attention"))
        normalized = self.norm(value, prefix + ".ffn_norm.weight")
        hidden = self.op("Mul", self.silu(self.linear(normalized, prefix + ".ffn.gate.weight")),
                         self.linear(normalized, prefix + ".ffn.up.weight"))
        return self.op("Add", value, self.linear(hidden, prefix + ".ffn.down.weight"))

    def boolean_ones(self, shape):
        return self.op("Expand", self.bank.constant(True, self.bank.np.bool_), shape)

    def public_memory(self):
        from onnx import TensorProto
        prefix = "public_encoder"
        batch, anchors = self.dimension("anchor_board", 0), self.dimension("anchor_board", 1)
        records = self.dimension("records", 1)
        safe_board = self.op("Where", self.unsqueeze("anchor_mask", [-1]), "anchor_board", self.ints(0))
        safe_metadata = self.op("Where", self.unsqueeze("anchor_mask", [-1]), "anchor_metadata", self.floats(0))
        board = self.op("Reshape", safe_board, self.ints([-1, 64]))
        metadata = self.op("Reshape", safe_metadata, self.ints([-1, 16]))
        square = self.op("Add", self.op("Gather", self.bank.weight(prefix + ".piece_embedding.weight"), board, axis=0),
                         self.bank.weight(prefix + ".square_embedding"))
        meta = self.op("Reshape", self.linear(metadata, prefix + ".metadata_projection.weight"),
                       self.ints([-1, 2, self.config.width]))
        board_tokens = self.op("Concat", square, meta, axis=1)
        board_mask = self.boolean_ones(self.shape(self.dimension(board_tokens, 0), self.ints([66])))
        for index in range(self.config.board_blocks):
            board_tokens = self.encoder_block(board_tokens, board_mask, prefix + f".board_blocks.{index}")
        all_boards = self.op("Reshape", board_tokens, self.shape(batch, anchors, self.ints([66, self.config.width])))
        anchor_context = self.op("Mul", self.op("ReduceMean", all_boards, axes=[2], keepdims=0),
                                 self.unsqueeze(self.op("Cast", "anchor_mask", to=TensorProto.FLOAT), [-1]))
        root_tokens = self.op("Gather", all_boards, self.ints(0), axis=1)

        fields = self.op("Reshape", "records", self.ints([-1, 4, 2]))
        fields = self.op("Add", self.linear(fields, prefix + ".record_projection.weight"),
                         self.bank.weight(prefix + ".record_field_embedding"))
        field_mask = self.boolean_ones(self.shape(self.dimension(fields, 0), self.ints([4])))
        for index in range(self.config.record_blocks):
            fields = self.encoder_block(fields, field_mask, prefix + f".record_blocks.{index}")
        tokens = self.op("Reshape", self.op("ReduceMean", fields, axes=[1], keepdims=0),
                         self.shape(batch, records, self.ints([self.config.width])))
        lines = self.line_encoder("record_line_tokens", "record_line_mask")
        tokens = self.op("Add", tokens, self.linear(lines, prefix + ".record_line_projection.weight"))
        # Only padded record references are replaced. Active invalid references
        # remain a dispatch validation error, never clamped to another anchor.
        refs = self.op("Where", "record_mask", "record_anchor", self.ints(0))
        refs = self.op("Expand", self.unsqueeze(refs, [-1]), self.op("Shape", tokens))
        contexts = self.op("GatherElements", anchor_context, refs, axis=1)
        tokens = self.op("Add", tokens, self.linear(contexts, prefix + ".anchor_record_projection.weight"))
        tokens = self.op("Mul", tokens, self.unsqueeze(self.op("Cast", "record_mask", to=TensorProto.FLOAT), [-1]))
        memory = self.op("Concat", root_tokens, tokens, axis=1)
        root_mask = self.boolean_ones(self.shape(batch, self.ints([66])))
        mask = self.op("Concat", root_mask, "record_mask", axis=1)
        memory_tokens = self.dimension(memory, 1)
        key = self.kv(self.linear(memory, prefix + ".memory_key.weight"), batch, memory_tokens)
        value = self.kv(self.linear(memory, prefix + ".memory_value.weight"), batch, memory_tokens)
        return key, value, mask, anchor_context

    def virtual_query_inputs(self):
        from onnx import helper
        # The existing relationship graph consumes local record references and
        # the independent public K/V. The overridden line encoder uses dynamic L.
        self.full_line_inputs()
        batch = self.dimension("query_anchor", 0)
        refs = self.op("Expand", self.unsqueeze("query_anchor", [-1]),
                       self.shape(batch, self.ints([3, self.config.width])))
        contexts = self.op("GatherElements", "anchor_context", refs, axis=1)
        contexts = self.op("Reshape", contexts, self.shape(batch, self.ints([3 * self.config.width])))
        self.nodes.append(helper.make_node("Identity", [contexts], ["query_anchor_features"],
                                           name=self.name + "_identity_query_anchor_features"))

    def private_initial(self, role):
        prefix = f"experts.{role}"
        query = self.linear("query", prefix + ".query_projection.weight")
        query = self.op("Add", query, self.linear("query_line_features", prefix + ".query_line_projection.weight"))
        related = self.linear(self.silu(self.biased_linear("relation_features", prefix + ".relation_hidden")),
                              prefix + ".relation_output.weight")
        related = self.op("Mul", related, self.unsqueeze("relation_active_float", [-1]))
        summed = self.op("ReduceSum", related, self.ints([1]), keepdims=0)
        count = self.op("ReduceSum", "relation_active_float", self.ints([1]), keepdims=1)
        query = self.op("Add", query, self.op("Div", summed, self.op("Max", count, self.floats(1))))
        query = self.op("Add", query, self.linear("query_anchor_features", prefix + ".query_anchor_projection.weight"))
        return self.op("Add", self.unsqueeze(self.bank.weight(prefix + ".initial_latent"), [0]), self.unsqueeze(query, [1]))

    def routed_initial(self, latent_shape):
        from onnx import helper
        branches = {}
        for role in ("critic", "proposer"):
            branch = _VirtualGraph(self.bank, self.config, role)
            branches[role] = branch.branch((branch.private_initial(role),), (latent_shape,), "private_initial")
        selected = self.bank.unique("selected_private_initial")
        self.nodes.append(helper.make_node("If", ["role_is_critic"], [selected],
                          name=self.bank.unique("route_private_initial"),
                          then_branch=branches["critic"], else_branch=branches["proposer"]))
        return selected

    def validator_heads(self, latent, embedded):
        """Fixed V expert; no P/C private weights or unused divergence branch."""
        prefix = "experts.validator"
        context = self.op("ReduceMean", latent, axes=[1], keepdims=0)
        expanded = self.op("Expand", self.unsqueeze(context, [1]), self.op("Shape", embedded))
        hidden = self.silu(self.biased_linear(self.op("Concat", embedded, expanded, axis=-1),
                                              prefix + ".interaction_policy_head.hidden"))
        logits = self.op("Squeeze", self.linear(hidden, prefix + ".interaction_policy_head.output.weight"), self.ints([-1]))
        logits = self.op("Where", "candidate_mask", logits, self.floats(-1e9))
        wdl = self.linear(context, prefix + ".wdl_head.weight")
        tasks = self.linear(context, prefix + ".task_head.weight")
        return logits, wdl, tasks


def _document(graph, bank, inputs, outputs):
    from onnx import TensorProto, checker, helper
    info = lambda entry: helper.make_tensor_value_info(entry[0], getattr(TensorProto, entry[1]), entry[2])
    document = helper.make_model(helper.make_graph(graph.nodes, graph.name,
                    [info(value) for value in inputs], [info(value) for value in outputs], bank.initializers),
                    producer_name="rz-pals-model", opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    helper.set_model_props(document, {"schema": ARTIFACT_SCHEMA, "model_profile": VIRTUAL_REVIEW_PROFILE,
                                     "model_semantics": graph.config.model_semantics,
                                     "encoding_schema": graph.config.encoding, "layout": graph.name,
                                     "layout_revision": str(LAYOUT_REVISION), "precision": "fp32",
                                     "physical_line_buckets": ",".join(map(str, LINE_BUCKETS))})
    checker.check_model(document, full_check=True)
    return document


def _identities(graph, values, names):
    from onnx import helper
    for value, name in zip(values, names):
        graph.nodes.append(helper.make_node("Identity", [value], [name], name="output_" + name))


def build_virtual_public_graph(model):
    """Build public K/V and anchor contexts without a neural forward."""
    _validate_config(model.config)
    bank = _Bank(model)
    graph = _VirtualGraph(bank, model.config, PUBLIC_LAYOUT)
    outputs = virtual_io_spec(model.config)["public"]
    _identities(graph, graph.public_memory(), [value[0] for value in outputs["outputs"]])
    document = _document(graph, bank, outputs["inputs"], outputs["outputs"])
    names = [value["initializer"] for value in bank.weights]
    if len(names) != len(set(names)) or any(not value["parameter"].startswith("public_encoder.") for value in bank.weights):
        raise ValueError("public anchor/record graph must own one role-neutral parameter bank")
    evidence = {"scope": "onnx_serialized_initializers_only", "layout_revision": LAYOUT_REVISION,
                "parameters": bank.weights, "unique_parameters": len(bank.weights),
                "weight_bytes": sum(value["bytes"] for value in bank.weights),
                "dynamic_line_axis": True, "allowed_line_buckets": LINE_BUCKETS,
                "shared_anchor_encoder_bank": True, "record_encoding_before_relation_mixing": True,
                "ort_prepack_copies": "unknown", "device_residency_sharing": "unknown"}
    return document, evidence


def build_virtual_shared_pc_graph(model):
    """Build V-free Fresh P/C with six hard role routes and one reader bank."""
    _validate_config(model.config)
    if set(model.experts) != {"proposer", "critic"}:
        raise ValueError("virtual shared P/C graph requires exactly the V-free P/C model")
    bank = _Bank(model)
    graph = _VirtualGraph(bank, model.config, SHARED_LAYOUT)
    spec = virtual_io_spec(model.config)["shared_pc"]
    latent_shape = ["batch", model.config.latent_slots, model.config.width]
    graph.virtual_query_inputs()
    latent = graph.routed_initial(latent_shape)
    for _ in range(model.config.iterations):
        for index in range(model.config.recurrent_blocks):
            latent = graph.reader(latent, index)
            latent = graph.routed("private_ffn", [latent_shape], latent, index)
    embedded = graph.candidate_embedding()
    logits, wdl, divergences = graph.routed("private_heads", [["batch", "candidates"], ["batch", 3], ["batch", "divergences"]], latent, embedded)
    _identities(graph, (logits, wdl, latent, divergences, "role_is_critic"), [value[0] for value in spec["outputs"]])
    document = _document(graph, bank, spec["inputs"], spec["outputs"])
    evidence = audit_shared_pc_graph(document, bank.weights)
    evidence.update({"parameters": bank.weights,
                     "layout_revision": LAYOUT_REVISION, "model_profile": VIRTUAL_REVIEW_PROFILE,
                     "dynamic_line_axis": True, "allowed_line_buckets": LINE_BUCKETS,
                     "anchor_query_conditioning": "all_three_exact_anchor_refs_including_empty_lines"})
    return document, evidence


def build_virtual_validator_graph(model):
    """Build a Fresh V graph from the full collection model's shared bank.

    The caller may pass a model containing P/C/V. Only explicitly selected
    shared and V-private parameters are serialized; unrelated experts are not
    copied into the graph. V has no Warm or role-routing capability.
    """
    _validate_config(model.config)
    if "validator" not in model.experts:
        raise ValueError("virtual validator graph requires the collection V expert")
    bank = _Bank(model)
    graph = _VirtualGraph(bank, model.config, VALIDATOR_LAYOUT)
    spec = virtual_io_spec(model.config)["validator"]
    graph.virtual_query_inputs()
    latent = graph.private_initial("validator")
    for _ in range(model.config.iterations):
        for index in range(model.config.recurrent_blocks):
            latent = graph.reader(latent, index)
            latent = graph.private_ffn(latent, "validator", index)
    logits, wdl, tasks = graph.validator_heads(latent, graph.candidate_embedding())
    _identities(graph, (logits, wdl, latent, tasks), [value[0] for value in spec["outputs"]])
    document = _document(graph, bank, spec["inputs"], spec["outputs"])
    shared_prefixes = ("reader_blocks.", "move_embedding.", "public_encoder.line_encoder.")
    if any(not value["parameter"].startswith((*shared_prefixes, "experts.validator.")) for value in bank.weights):
        raise ValueError("validator graph contains an unrelated private/board parameter")
    if any("experts.proposer." in value.name or "experts.critic." in value.name for value in document.graph.initializer):
        raise ValueError("validator graph must not contain P/C private initializer banks")
    names = [value["initializer"] for value in bank.weights]
    if len(names) != len(set(names)) or any(node.op_type == "If" for node in document.graph.node):
        raise ValueError("Fresh validator requires single-owned banks without role/Warm routing")
    evidence = {"scope": "onnx_serialized_initializers_only", "layout_revision": LAYOUT_REVISION,
                "model_profile": VIRTUAL_REVIEW_PROFILE, "role": "validator", "fresh_only": True,
                "parameters": bank.weights, "unique_parameters": len(bank.weights),
                "weight_bytes": sum(value["bytes"] for value in bank.weights),
                "pc_private_initializers": 0, "if_routes": 0,
                "shared_parameters": [value for value in bank.weights if value["parameter"].startswith(shared_prefixes)],
                "shared_bank_scope": "single_graph_serialized_bank_only_not_cross_session_device_sharing",
                "dynamic_line_axis": True, "allowed_line_buckets": LINE_BUCKETS,
                "ort_prepack_copies": "unknown", "device_residency_sharing": "unknown"}
    return document, evidence


def _warm_branches(node):
    branches = {attribute.name: attribute.g for attribute in node.attribute
                if attribute.name in ("then_branch", "else_branch")}
    if set(branches) != {"then_branch", "else_branch"} or len(node.attribute) != 2:
        raise ValueError("virtual Warm mode requires exactly two If branches")
    return branches


def _walk_nodes(graph):
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.name in ("then_branch", "else_branch"):
                yield from _walk_nodes(attribute.g)


def _virtual_initial_route(document):
    initial = [(index, node) for index, node in enumerate(document.graph.node)
               if node.op_type == "If" and node.name.startswith("route_private_initial_")]
    if len(initial) != 1:
        raise ValueError("virtual Fresh private initialization route is not unique")
    return initial[0]


def build_virtual_warm_shared_pc_graph(model):
    """Create an approximate P/C seed entry with no query preparation on Warm.

    All Fresh query-line encoding, relation gathering, anchor conditioning and
    role initialization are inside the Fresh If branch. The Warm branch only
    consumes the complete accepted final latent. Unchanged reader operations
    recompute private self-attention K/V. This graph cannot admit a seed; the
    Rust owner supplies accepted-role/Rules/context and physical-fence proof.
    """
    from onnx import TensorProto, checker, helper
    fresh, fresh_evidence = build_virtual_shared_pc_graph(model)
    document = copy.deepcopy(fresh)
    index, initial = _virtual_initial_route(fresh)
    selected = initial.output[0]
    fresh_name, seed_name = selected + "__fresh_branch", selected + "__warm_branch"
    fresh_nodes = copy.deepcopy(list(fresh.graph.node[:index + 1]))
    fresh_nodes[-1].output[0] = fresh_name
    shape = ["batch", model.config.latent_slots, model.config.width]
    fresh_branch = helper.make_graph(fresh_nodes, "virtual_complete_fresh_initial", [],
                                    [helper.make_tensor_value_info(fresh_name, TensorProto.FLOAT, shape)])
    seed_branch = helper.make_graph(
        [helper.make_node("Identity", ["initial_latent"], [seed_name], name="consume_complete_virtual_private_seed")],
        "virtual_accepted_seed_initial", [],
        [helper.make_tensor_value_info(seed_name, TensorProto.FLOAT, shape)])
    mode = helper.make_node("If", ["warm_start"], [selected], name=WARM_MODE_NODE,
                            then_branch=seed_branch, else_branch=fresh_branch)
    suffix = copy.deepcopy(list(fresh.graph.node[index + 1:]))
    del document.graph.node[:]
    document.graph.node.extend([mode, *suffix])
    document.graph.input.extend([
        helper.make_tensor_value_info("initial_latent", TensorProto.FLOAT, shape),
        helper.make_tensor_value_info("warm_start", TensorProto.BOOL, []),
    ])
    document.graph.name = WARM_LAYOUT
    properties = {value.key: value.value for value in document.metadata_props}
    properties.update({"layout": WARM_LAYOUT,
                       "private_seed": "accepted_final_latent_same_role_rules_query_epoch_physical_fence",
                       "fresh_query_preparation": "inside_fresh_branch_only",
                       "warm_mode": "approximate_complete_seed_no_query_double_add"})
    helper.set_model_props(document, properties)
    checker.check_model(document, full_check=True)
    return document, audit_virtual_warm_shared_pc_graph(document, fresh, fresh_evidence)


def audit_virtual_warm_shared_pc_graph(document, fresh, fresh_evidence):
    """Compare full initializer bytes and every Fresh/recurrent node protobuf."""
    from onnx import TensorProto, helper
    if fresh_evidence.get("if_routes") != 6 or fresh_evidence.get("branch_local_initializers") != 0:
        raise ValueError("virtual Warm template lacks the audited Fresh P/C bank")
    if fresh.graph.name != SHARED_LAYOUT or document.graph.name != WARM_LAYOUT:
        raise ValueError("unregistered virtual Fresh/Warm layout")
    index, initial = _virtual_initial_route(fresh)
    if len(document.graph.node) != len(fresh.graph.node) - index:
        raise ValueError("Warm must replace only the complete Fresh initialization prefix")
    for field in ("functions", "opset_import", "training_info"):
        if [value.SerializeToString() for value in getattr(document, field)] != [value.SerializeToString() for value in getattr(fresh, field)]:
            raise ValueError("virtual Warm changed a function/opset/training domain")
    if document.ir_version != fresh.ir_version:
        raise ValueError("virtual Warm changed IR version")
    for field in ("initializer", "sparse_initializer", "value_info", "quantization_annotation"):
        if [value.SerializeToString() for value in getattr(document.graph, field)] != [value.SerializeToString() for value in getattr(fresh.graph, field)]:
            raise ValueError("virtual Warm changed the serialized bank/type annotations")
    actual_suffix = [node.SerializeToString() for node in document.graph.node[1:]]
    expected_suffix = [node.SerializeToString() for node in fresh.graph.node[index + 1:]]
    if actual_suffix != expected_suffix:
        raise ValueError("virtual Warm changed recurrent/head operations")
    shape = ["batch", 16, 384]
    expected_inputs = [value.SerializeToString() for value in fresh.graph.input] + [
        helper.make_tensor_value_info("initial_latent", TensorProto.FLOAT, shape).SerializeToString(),
        helper.make_tensor_value_info("warm_start", TensorProto.BOOL, []).SerializeToString(),
    ]
    if [value.SerializeToString() for value in document.graph.input] != expected_inputs:
        raise ValueError("virtual Warm input dtype/shape/mode mismatch")
    if [value.SerializeToString() for value in document.graph.output] != [value.SerializeToString() for value in fresh.graph.output]:
        raise ValueError("virtual Warm changed output meaning")
    mode = document.graph.node[0]
    selected = initial.output[0]
    if (mode.name != WARM_MODE_NODE or mode.op_type != "If" or mode.domain or getattr(mode, "overload", "")
            or list(mode.input) != ["warm_start"] or list(mode.output) != [selected]):
        raise ValueError("virtual Warm requires one scalar hard If seed selector")
    branches = _warm_branches(mode)
    seed, initial_branch = branches["then_branch"], branches["else_branch"]
    for branch in branches.values():
        if branch.input or branch.initializer or branch.sparse_initializer:
            raise ValueError("virtual Warm branch-local inputs/bank copies are forbidden")
    seed_name, fresh_name = selected + "__warm_branch", selected + "__fresh_branch"
    expected_seed = helper.make_node("Identity", ["initial_latent"], [seed_name], name="consume_complete_virtual_private_seed")
    if len(seed.node) != 1 or seed.node[0].SerializeToString() != expected_seed.SerializeToString():
        raise ValueError("Warm seed branch must contain only complete latent Identity")
    expected_initial = copy.deepcopy(list(fresh.graph.node[:index + 1]))
    expected_initial[-1].output[0] = fresh_name
    if [node.SerializeToString() for node in initial_branch.node] != [node.SerializeToString() for node in expected_initial]:
        raise ValueError("Fresh branch changed query/anchor/relation/role initialization")
    for branch, output in ((seed, seed_name), (initial_branch, fresh_name)):
        expected = helper.make_tensor_value_info(output, TensorProto.FLOAT, shape)
        if len(branch.output) != 1 or branch.output[0].SerializeToString() != expected.SerializeToString():
            raise ValueError("virtual Warm branch output shape mismatch")
    all_nodes = list(_walk_nodes(document.graph))
    roles = [node for node in all_nodes if node.op_type == "If" and list(node.input) == ["role_is_critic"]]
    modes = [node for node in all_nodes if node.op_type == "If" and list(node.input) == ["warm_start"]]
    routes = [node for node in all_nodes if node.op_type == "If"]
    if len(roles) != 6 or len(modes) != 1 or len(routes) != 7:
        raise ValueError("virtual Warm recursive role/mode route count mismatch")
    for route in routes:
        if any(branch.initializer or branch.sparse_initializer for branch in _warm_branches(route).values()):
            raise ValueError("virtual Warm nested branch initializer copies are forbidden")
    if any("validator" in value.name for value in document.graph.initializer):
        raise ValueError("virtual Warm graph must remain V-free")
    return {"scope": "onnx_serialized_initializers_only", "layout_revision": LAYOUT_REVISION,
            "model_profile": VIRTUAL_REVIEW_PROFILE,
            "fresh_graph_sha256": hashlib.sha256(fresh.SerializeToString()).hexdigest(),
            "parameters": copy.deepcopy(fresh_evidence["parameters"]),
            "fresh_ownership": copy.deepcopy(fresh_evidence),
            "recursive_role_if_routes": 6, "warm_mode_if_routes": 1,
            "branch_local_initializers": 0, "seed_values_per_batch_row": 6144,
            "fresh_initialization": "unchanged_complete_query_anchor_relation_role_prefix",
            "warm_initialization": "complete_final_latent_identity_no_query_preparation",
            "private_self_kv": "recomputed_by_unchanged_readers",
            "accepted_seed_and_rules_context": "requires_native_owner_validation",
            "inactive_private_execution": "requires_runtime_profile",
            "ort_prepack_copies": "unknown", "device_residency_sharing": "unknown"}


def _save(document, evidence, destination, *, output_guard=None):
    """Publish a complete artifact exclusively; retain failed partial writes.

    A link-free native rename provides atomic no-replace publication.
    The caller owns the managed output root and cleanup of preserved partials.
    """
    from .artifacts import output_directory, publish_noreplace
    destination = Path(destination).expanduser().resolve()
    destination = output_directory(destination.parent, output_guard=output_guard) / destination.name
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists():
        raise FileExistsError(f"graph already exists: {destination.name}")
    raw = document.SerializeToString()
    if output_guard is not None:
        output_guard(len(raw))
    with partial.open("xb") as stream:
        written = stream.write(raw)
        if written != len(raw):
            raise OSError("incomplete serialized graph write")
        stream.flush()
        os.fsync(stream.fileno())
    publish_noreplace(partial, destination)
    return {"file": destination.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "evidence": evidence, "numeric_status": "not_run", "cuda_status": "not_run"}


def export_virtual_public_graph(model, destination):
    document, evidence = build_virtual_public_graph(model)
    return _save(document, evidence, destination)


def export_virtual_shared_pc_graph(model, destination):
    document, evidence = build_virtual_shared_pc_graph(model)
    return _save(document, evidence, destination)


def export_virtual_validator_graph(model, destination):
    document, evidence = build_virtual_validator_graph(model)
    return _save(document, evidence, destination)


def export_virtual_warm_shared_pc_graph(model, destination):
    document, evidence = build_virtual_warm_shared_pc_graph(model)
    return _save(document, evidence, destination)
