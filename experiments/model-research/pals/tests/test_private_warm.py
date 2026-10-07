"""Separate approximate warm tensor checks, never native/game/training claims."""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rz_pals_model.config import ModelConfig, SCHEMA, TASKS
from rz_pals_model.artifacts import validate_export_manifest
from rz_pals_model.onnx_warm import (PRIVATE_SEED_POLICY, WARM_ARTIFACT_SCHEMA,
                                    WARM_GRAPH_SEMANTICS, WARM_LAYOUT,
                                    WARM_LAYOUT_REVISION, audit_warm_pc_graph,
                                    build_warm_pc_graph)
from rz_pals_model.warm_artifacts import (WARM_GRAPH_FILE, _expected_private_inputs,
                                        _expected_private_outputs, _feed,
                                        _reference, _compare, _cpu_session,
                                        _document_from_immutable_bytes, _read_immutable_bytes,
                                        _read_warm_manifest, MAX_WARM_MANIFEST_BYTES,
                                        export_warm_checkpoint, validate_warm_feed,
                                        validate_warm_export_manifest, verify_warm_cpu_routes)

try:
    import numpy as np
    import torch
    from rz_pals_model.model import fixture_input, initialize
except ImportError:
    torch = None


def _rules_declaration():
    declaration = {
        "schema": "rovezero.pals-rules-descriptor.v1",
        "rules_input_profile": "rz-pals-rules-fields-v1",
        "rules_encoder_source_sha256": "11" * 32,
        "config": ModelConfig().to_dict(),
        "learned_input_compatibility": "unverified_declaration_only",
        "rules_base_semantics": "unit_source_declaration_not_chess_fixture",
        "semantic_digest_algorithm": "sha256_u64le_length_prefixed_utf8_fields",
    }
    fields = [declaration["rules_input_profile"], declaration["rules_base_semantics"]]
    for key, count in (("metadata_features", 16), ("record_features", 16),
                       ("query_features", 16), ("divergence_features", 8)):
        declaration[key] = [key + str(index) for index in range(count)]
        fields += declaration[key]
    declaration["semantic_fields"] = fields
    semantic = hashlib.sha256()
    for field in fields:
        value = field.encode("utf-8")
        semantic.update(len(value).to_bytes(8, "little"))
        semantic.update(value)
    declaration["rules_input_semantic_sha256"] = semantic.hexdigest()
    return declaration


def _manifest():
    declaration = _rules_declaration()
    canonical = json.dumps(declaration, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    metadata = {"config": ModelConfig().to_dict(), "checkpoint_sha256": "00" * 32,
                "training_steps": 0, "trained": False}
    public_inputs = [
        {"name": "board", "dtype": "INT64", "shape": ["batch", 64]},
        {"name": "metadata", "dtype": "FLOAT", "shape": ["batch", 16]},
        {"name": "records", "dtype": "FLOAT", "shape": ["batch", "records", 16]},
        {"name": "record_mask", "dtype": "BOOL", "shape": ["batch", "records"]},
    ]
    public_outputs = [
        {"name": "memory_key", "dtype": "FLOAT", "shape": ["batch", 2, "memory_tokens", 64]},
        {"name": "memory_value", "dtype": "FLOAT", "shape": ["batch", 2, "memory_tokens", 64]},
        {"name": "memory_mask", "dtype": "BOOL", "shape": ["batch", "memory_tokens"]},
    ]
    manifest = {
        "schema": WARM_ARTIFACT_SCHEMA, "model_semantics": SCHEMA,
        "layout": WARM_LAYOUT, "layout_revision": WARM_LAYOUT_REVISION,
        **metadata, "precision": "fp32", "tf32": False, "approximate": True,
        "roles": ["proposer", "critic"], "validator_present": False, "task_names": list(TASKS),
        "warm_graph_semantics": WARM_GRAPH_SEMANTICS, "private_seed_policy": PRIVATE_SEED_POLICY,
        "native_seed_owner_support": "not_registered_by_python_export",
        "batch_mode": "one_scalar_role_and_one_scalar_mode_per_physical_batch",
        "rules_input_profile": declaration["rules_input_profile"],
        "rules_input_semantic_sha256": declaration["rules_input_semantic_sha256"],
        "rules_encoder_source_sha256": declaration["rules_encoder_source_sha256"],
        "rules_profile_descriptor_sha256": hashlib.sha256(canonical).hexdigest(),
        "rules_profile_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "rules_input_declaration": declaration, "learned_input_compatibility": declaration["learned_input_compatibility"],
        "graphs": [
            {"file": "public_memory.onnx", "role": "public", "sha256": "22" * 32,
             "opset": 17, "inputs": public_inputs, "outputs": public_outputs},
            {"file": WARM_GRAPH_FILE, "role": "shared_pc_warm", "sha256": "33" * 32,
             "opset": 17, "inputs": _expected_private_inputs(), "outputs": _expected_private_outputs()},
        ],
    }
    return manifest, metadata


class WarmRegistrationTests(unittest.TestCase):
    def test_bounded_regular_file_pin_and_manifest_duplicate_rejection(self):
        with tempfile.TemporaryDirectory(prefix="rovezero-pals-warm-pin-") as directory:
            path = Path(directory) / "manifest.json"
            original = b'{"schema":"original"}'
            path.write_bytes(original)
            pinned, provenance = _read_immutable_bytes(path, 64, hashlib.sha256(original).hexdigest())
            replacement = Path(directory) / "replacement.json"
            replacement.write_bytes(b'{"schema":"replacement"}')
            replacement.replace(path)
            self.assertEqual(pinned, original)
            self.assertEqual(provenance["sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(provenance["bytes"], len(original))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                _read_immutable_bytes(path, 64, provenance["sha256"])
            for contents in (b'{"x":1,"x":2}', b'{"x":{"y":1,"y":2}}', b'{"x":NaN}', b'[]'):
                path.write_bytes(contents)
                with self.subTest(contents=contents), self.assertRaises(ValueError):
                    _read_warm_manifest(path)
            path.write_bytes(b" " * (MAX_WARM_MANIFEST_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "bounded"):
                _read_warm_manifest(path)
            with self.assertRaisesRegex(ValueError, "regular file"):
                _read_immutable_bytes(Path(directory), 64)

    def test_domain_is_explicit_and_old_parser_cannot_silently_accept_it(self):
        manifest, metadata = _manifest()
        self.assertEqual(validate_warm_export_manifest(manifest, metadata), ["proposer", "critic"])
        with self.assertRaisesRegex(ValueError, "schema/config/checkpoint"):
            validate_export_manifest(manifest, metadata)
        for key, value in (("schema", SCHEMA), ("layout", "shared_pc_if"),
                           ("approximate", False), ("warm_graph_semantics", "unknown"),
                           ("private_seed_policy", "fresh_cache"), ("tf32", True),
                           ("native_seed_owner_support", "supported")):
            changed = copy.deepcopy(manifest)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_warm_export_manifest(changed, metadata)

    def test_shape_mode_role_training_and_rules_declaration_are_bound(self):
        manifest, metadata = _manifest()
        for name, mutate in (
            ("seed_shape", lambda m: m["graphs"][1]["inputs"][-2].update(shape=["batch", 15, 384])),
            ("seed_dtype", lambda m: m["graphs"][1]["inputs"][-2].update(dtype="DOUBLE")),
            ("mode_shape", lambda m: m["graphs"][1]["inputs"][-1].update(shape=["batch"])),
            ("V", lambda m: m.update(validator_present=True, roles=["proposer", "critic", "validator"])),
            ("missing_private", lambda m: m["graphs"].pop()),
            ("trained", lambda m: m.update(trained=True)),
            ("false_training_steps", lambda m: m.update(training_steps=False)),
            ("rules_meaning", lambda m: m["rules_input_declaration"]["record_features"].__setitem__(0, "changed")),
            ("rules_hash", lambda m: m.update(rules_input_semantic_sha256="44" * 32)),
        ):
            changed = copy.deepcopy(manifest)
            mutate(changed)
            with self.subTest(case=name), self.assertRaises(ValueError):
                validate_warm_export_manifest(changed, metadata)


@unittest.skipUnless(torch is not None, "torch absent: warm neural checks not executed")
class WarmNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = initialize(19)
        cls.free = cls.model.without_validator()
        cls.data = fixture_input()
        cls.data.validate()
        with torch.no_grad():
            cls.memory = cls.free.public_encoder(*cls.data.public_args())

    def assertOutputsEqual(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))

    def test_fresh_is_exact_legacy_and_seed_independent_without_checkpoint_change(self):
        before = {key: (tuple(value.shape), hashlib.sha256(value.detach().numpy().tobytes()).hexdigest())
                  for key, value in self.free.state_dict().items()}
        parameters = sum(p.numel() for p in self.free.parameters())
        seed = torch.arange(6144, dtype=torch.float32).reshape(1, 16, 384) / 6144
        with torch.no_grad():
            for role in ("proposer", "critic"):
                old = self.free.role_graph(role)(*self.data.role_args(self.memory))
                wrapper = self.free.warm_role_graph(role)
                self.assertOutputsEqual(old, wrapper(*self.data.role_args(self.memory), seed, torch.tensor(False)))
                self.assertOutputsEqual(old, wrapper(*self.data.role_args(self.memory), -seed, torch.tensor(False)))
        after = {key: (tuple(value.shape), hashlib.sha256(value.detach().numpy().tobytes()).hexdigest())
                 for key, value in self.free.state_dict().items()}
        self.assertEqual(before, after)
        self.assertEqual(parameters, sum(p.numel() for p in self.free.parameters()))

    def test_parameter_free_body_matches_original_initialization_for_both_roles(self):
        with torch.no_grad():
            data = dataclasses.replace(self.data, query=torch.arange(16, dtype=torch.float32).reshape(1, 16) / 17)
            for role in ("proposer", "critic"):
                expert = self.free.experts[role]
                original_initial = expert.initial_latent[None, :, :] + expert.query_projection(data.query)[:, None, :]
                expected = self.free.role_graph(role)(*data.role_args(self.memory))
                actual = self.free.warm_role_graph(role)(*data.role_args(self.memory), original_initial, torch.tensor(True))
                self.assertOutputsEqual(expected, actual)

    def test_warm_consumes_full_seed_and_does_not_double_add_query(self):
        with torch.no_grad():
            for role in ("proposer", "critic"):
                wrapper = self.free.warm_role_graph(role)
                seed = self.free.role_graph(role)(*self.data.role_args(self.memory))[2].clone()
                actual = wrapper(*self.data.role_args(self.memory), seed, torch.tensor(True))
                changed_query = dataclasses.replace(self.data, query=torch.ones_like(self.data.query))
                # Tensor math only: the Rust owner rejects changing the sealed
                # logical query/context when admitting a previously accepted seed.
                self.assertOutputsEqual(actual, wrapper(*changed_query.role_args(self.memory), seed, torch.tensor(True)))
                for index in (0, 3071, 6143):
                    changed = seed.clone()
                    changed.reshape(-1)[index] += 0.25
                    output = wrapper(*self.data.role_args(self.memory), changed, torch.tensor(True))
                    self.assertFalse(torch.equal(actual[2], output[2]), (role, index))
                self.assertTrue(torch.equal(seed, self.free.role_graph(role)(*self.data.role_args(self.memory))[2]))

    def test_role_order_preserves_seed_and_memory_and_never_adds_v(self):
        with torch.no_grad():
            seed = torch.zeros((1, 16, 384), dtype=torch.float32)
            saved = tuple(value.clone() for value in self.memory)
            p = self.free.warm_role_graph("proposer")
            a = p(*self.data.role_args(self.memory), seed, torch.tensor(True))
            self.free.warm_role_graph("critic")(*self.data.role_args(self.memory), seed, torch.tensor(True))
            self.assertOutputsEqual(a, p(*self.data.role_args(self.memory), seed, torch.tensor(True)))
            for before, after in zip(saved, self.memory):
                self.assertTrue(torch.equal(before, after))
            self.assertTrue(torch.count_nonzero(seed) == 0)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.model.warm_role_graph("validator")
        with self.assertRaisesRegex(ValueError, "absent"):
            self.free.warm_role_graph("validator")

    def test_invalid_seed_mode_finite_and_truncation_fail_in_both_modes(self):
        wrapper = self.free.warm_role_graph("proposer")
        seed = torch.zeros((1, 16, 384), dtype=torch.float32)
        invalid = [seed[:, :15], seed[:, :, :383], seed.repeat(2, 1, 1), seed.to(torch.float64)]
        for bad in (float("nan"), float("inf"), -float("inf")):
            changed = seed.clone()
            changed.reshape(-1)[6143] = bad
            invalid.append(changed)
        for value in invalid:
            for mode in (False, True):
                with self.subTest(shape=tuple(value.shape), mode=mode), self.assertRaises(ValueError):
                    wrapper(*self.data.role_args(self.memory), value, torch.tensor(mode))
        for mode in (True, torch.tensor([True]), torch.tensor(1), torch.tensor(1.0)):
            with self.assertRaisesRegex(ValueError, "scalar bool"):
                wrapper(*self.data.role_args(self.memory), seed, mode)

    def test_numpy_preflight_preserves_bits_and_rejects_mode_shape_and_nonfinite(self):
        seed = np.zeros((1, 16, 384), dtype=np.float32)
        seed.view(np.uint32).reshape(-1)[0] = 0x80000000
        seed.view(np.uint32).reshape(-1)[6143] = 1
        memory = [value.detach().numpy() for value in self.memory]
        feed = _feed(self.data, memory, seed, "proposer", True)
        before = seed.tobytes()
        validate_warm_feed(feed)
        self.assertEqual(seed.tobytes(), before)
        self.assertEqual(feed["initial_latent"].tobytes(), before)
        changes = {"warm_start": np.asarray([True]), "role_is_critic": np.asarray(1),
                   "initial_latent": seed.astype(np.float64), "query": np.full((1, 16), np.nan, dtype=np.float32)}
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_warm_feed({**feed, key: value})
        with self.assertRaises(ValueError):
            validate_warm_feed({key: value for key, value in feed.items() if key != "initial_latent"})
        with self.assertRaises(ValueError):
            validate_warm_feed(feed, max_batch=0)

    def test_new_graph_has_recursive_if_and_unchanged_legacy_serialization(self):
        import onnx
        from rz_pals_model.onnx_shared import build_shared_pc_graph
        original, legacy_evidence = build_shared_pc_graph(self.free)
        before = original.SerializeToString()
        warm, evidence = build_warm_pc_graph(self.free)
        again, _ = build_shared_pc_graph(self.free)
        self.assertEqual(before, again.SerializeToString())
        self.assertEqual(evidence["recursive_role_if_routes"], 6)
        self.assertEqual(evidence["warm_mode_if_routes"], 1)
        self.assertEqual(evidence["seed_values_per_batch_row"], 6144)
        self.assertEqual(evidence["legacy_ownership"], legacy_evidence)
        self.assertEqual([v.SerializeToString() for v in warm.graph.initializer],
                         [v.SerializeToString() for v in original.graph.initializer])
        changed = onnx.ModelProto()
        changed.CopyFrom(warm)
        changed.graph.initializer.append(warm.graph.initializer[0])
        with self.assertRaisesRegex(ValueError, "bank"):
            audit_warm_pc_graph(changed, original, legacy_evidence)
        changed.CopyFrom(warm)
        mode = next(node for node in changed.graph.node if node.name == "private_warm_mode")
        mode.op_type = "Where"
        with self.assertRaisesRegex(ValueError, "scalar If"):
            audit_warm_pc_graph(changed, original, legacy_evidence)
        changed.CopyFrom(warm)
        changed.graph.input[-1].type.tensor_type.elem_type = onnx.TensorProto.INT64
        with self.assertRaisesRegex(ValueError, "shape/dtype/mode"):
            audit_warm_pc_graph(changed, original, legacy_evidence)
        with self.assertRaisesRegex(ValueError, "V-free"):
            build_warm_pc_graph(self.model)

    def test_closed_template_rejects_hidden_banks_functions_opsets_and_custom_mode_ops(self):
        import onnx
        from rz_pals_model.onnx_shared import build_shared_pc_graph
        original, evidence = build_shared_pc_graph(self.free)
        warm, _ = build_warm_pc_graph(self.free)
        changed = onnx.ModelProto()
        for field in ("functions", "opset_import", "training_info"):
            changed.CopyFrom(warm)
            getattr(changed, field).add()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "function/opset/training"):
                audit_warm_pc_graph(changed, original, evidence)
        changed.CopyFrom(warm)
        changed.graph.sparse_initializer.add().values.name = "hidden_sparse_weight"
        with self.assertRaisesRegex(ValueError, "sparse bank"):
            audit_warm_pc_graph(changed, original, evidence)
        for target in ("mode", "seed"):
            changed.CopyFrom(warm)
            mode = next(node for node in changed.graph.node if node.name == "private_warm_mode")
            branch = next(attribute.g for attribute in mode.attribute if attribute.name == "then_branch")
            node = mode if target == "mode" else branch.node[0]
            node.domain = "custom_execution"
            with self.subTest(target=target), self.assertRaises(ValueError):
                audit_warm_pc_graph(changed, original, evidence)
            if hasattr(node, "overload"):
                node.domain = ""
                node.overload = "custom_overload"
                with self.subTest(target=target, overload=True), self.assertRaises(ValueError):
                    audit_warm_pc_graph(changed, original, evidence)
        changed.CopyFrom(warm)
        mode = next(node for node in changed.graph.node if node.name == "private_warm_mode")
        next(attribute.g for attribute in mode.attribute if attribute.name == "then_branch").sparse_initializer.add()
        with self.assertRaisesRegex(ValueError, "copied bank"):
            audit_warm_pc_graph(changed, original, evidence)

    def test_replaced_graph_path_cannot_change_audit_session_or_cpu_profile_bytes(self):
        document, _ = build_warm_pc_graph(self.free)
        serialized = document.SerializeToString()
        expected_sha = hashlib.sha256(serialized).hexdigest()
        memory = [value.detach().numpy() for value in self.memory]
        seed = np.zeros((1, 16, 384), dtype=np.float32)
        feed = _feed(self.data, memory, seed, "proposer", False)
        with tempfile.TemporaryDirectory(prefix="rovezero-pals-warm-execution-pin-") as directory:
            path = Path(directory) / "registered.onnx"
            path.write_bytes(serialized)
            pinned, provenance = _read_immutable_bytes(path, len(serialized), expected_sha)
            replacement = Path(directory) / "replacement.onnx"
            replacement.write_bytes(b"invalid mutable replacement must never execute")
            replacement.replace(path)
            self.assertEqual(_document_from_immutable_bytes(pinned).SerializeToString(), serialized)
            session = _cpu_session(pinned)
            _compare(_reference(self.free, self.data, self.memory, "proposer", seed, False),
                     session.run(None, feed), feed["candidate_mask"], {})
            routes = verify_warm_cpu_routes(pinned, feed)
            self.assertEqual(len(routes["cases"]), 4)
            self.assertTrue(all(case["graph_sha256"] == provenance["sha256"] for case in routes["cases"]))
            self.assertTrue(all(case["execution_source"] == "same_verified_immutable_bytes" for case in routes["cases"]))

    def test_cpu_ort_fresh_legacy_warm_torch_and_all_seed_regions(self):
        import onnxruntime as ort
        from rz_pals_model.onnx_shared import build_shared_pc_graph
        self.assertEqual(ort.__version__, "1.22.0", "different ORT is not silent acceptance")
        document, _ = build_warm_pc_graph(self.free)
        legacy_document, _ = build_shared_pc_graph(self.free)
        warm, legacy = _cpu_session(document.SerializeToString()), _cpu_session(legacy_document.SerializeToString())
        maxima = {}
        memory = [value.detach().numpy() for value in self.memory]
        for role in ("proposer", "critic"):
            seed = np.zeros((1, 16, 384), dtype=np.float32)
            feed = _feed(self.data, memory, seed, role, False)
            fresh = warm.run(None, feed)
            old = legacy.run(None, {name: value for name, value in feed.items() if name not in ("initial_latent", "warm_start")})
            reference = _reference(self.free, self.data, self.memory, role, seed, False)
            _compare(reference, old, feed["candidate_mask"], maxima)
            _compare(old, fresh, feed["candidate_mask"], maxima)
            seed = np.ascontiguousarray(reference[2])
            ignored = warm.run(None, _feed(self.data, memory, seed, role, False))
            for a, b in zip(fresh, ignored):
                self.assertTrue(np.array_equal(a, b))
            output = warm.run(None, _feed(self.data, memory, seed, role, True))
            _compare(_reference(self.free, self.data, self.memory, role, seed, True), output, feed["candidate_mask"], maxima)
            for index in (0, 3071, 6143):
                changed = seed.copy()
                changed.reshape(-1)[index] += np.float32(0.25)
                actual = warm.run(None, _feed(self.data, memory, changed, role, True))
                _compare(_reference(self.free, self.data, self.memory, role, changed, True), actual, feed["candidate_mask"], maxima)
                self.assertFalse(np.array_equal(output[2], actual[2]))
        self.assertLessEqual(maxima["policy"], 1e-4)
        self.assertLessEqual(maxima["wdl"], 1e-4)

    def test_export_failure_preserves_legacy_paths_autograd_and_external_output(self):
        metadata = {"checkpoint_sha256": "00" * 32, "config": ModelConfig().to_dict(),
                    "trained": False, "training_steps": 0}
        declaration = _rules_declaration()
        with tempfile.TemporaryDirectory(prefix="rovezero-pals-private-warm-export-") as directory:
            path = Path(directory) / "rules.json"
            path.write_text(json.dumps(declaration), encoding="utf-8")
            with torch.enable_grad(), patch("rz_pals_model.warm_artifacts.load_checkpoint", return_value=(self.model, metadata)), patch("torch.onnx.export", side_effect=RuntimeError("injected new warm export failure")):
                with self.assertRaisesRegex(RuntimeError, "injected new warm"):
                    export_warm_checkpoint("unused", Path(directory) / "export", path)
                self.assertTrue(torch.is_grad_enabled())
                self.assertFalse((Path(directory) / "export" / "warm_export.json").exists())


if __name__ == "__main__":
    unittest.main()
