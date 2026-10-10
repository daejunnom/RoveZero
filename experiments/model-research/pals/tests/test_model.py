import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rz_pals_model.config import ModelConfig, TASKS, matmul_flops
from rz_pals_model.artifacts import annotate_public_memory_shapes, atomic_json, digest_file, export_checkpoint, output_directory, read_rules_profile, validate_export_manifest

try:
    import torch
    from rz_pals_model.model import fixture_input, initialize
except ImportError:
    torch = None


class ConfigTests(unittest.TestCase):
    def test_no_silent_configuration_or_capacity_change(self):
        with self.assertRaises(ValueError): dataclasses.replace(ModelConfig(), width=256).validate()
        with self.assertRaises(ValueError): matmul_flops(ModelConfig(), "proposer", 129, 1)
        p = matmul_flops(ModelConfig(), "proposer", 0, 20)
        c = matmul_flops(ModelConfig(), "critic", 128, 20, 16)
        self.assertGreater(c["public_encode_matmul_flops"], p["public_encode_matmul_flops"])
        self.assertGreater(c["role_forward_matmul_flops"], p["role_forward_matmul_flops"])
        self.assertEqual(c["optimizer_steps"], 0)

    def test_task_indices_and_empty_padding_are_explicit(self):
        self.assertEqual(TASKS, ("defend_response", "attack_repair", "widen_responses", "lower_selectivity", "resume_task", "cross_profile_recheck", "defer"))
        empty = matmul_flops(ModelConfig(), "critic", 0, 0, 0)
        self.assertEqual(empty["physical_records"], 1)
        self.assertEqual(empty["physical_candidates"], 1)
        self.assertEqual(empty["physical_divergences"], 1)
        self.assertEqual(empty["cold_forward_matmul_flops"], matmul_flops(ModelConfig(), "critic", 1, 1, 1)["cold_forward_matmul_flops"])


class ArtifactTests(unittest.TestCase):
    def test_rules_declaration_digest_binds_order_and_all_field_meanings(self):
        declaration = {"schema": "rovezero.pals-rules-descriptor.v1", "rules_input_profile": "rz-pals-rules-fields-v1",
                       "rules_encoder_source_sha256": "00" * 32, "config": ModelConfig().to_dict(),
                       "learned_input_compatibility": "unverified_declaration_only", "rules_base_semantics": "explicit_base_meaning",
                       "semantic_digest_algorithm": "sha256_u64le_length_prefixed_utf8_fields"}
        fields = [declaration["rules_input_profile"], declaration["rules_base_semantics"]]
        for key, count in (("metadata_features", 16), ("record_features", 16), ("query_features", 16), ("divergence_features", 8)):
            declaration[key] = [key + str(index) for index in range(count)]
            fields += declaration[key]
        declaration["semantic_fields"] = fields
        digest = hashlib.sha256()
        for field in fields:
            value = field.encode("utf-8")
            digest.update(len(value).to_bytes(8, "little")); digest.update(value)
        declaration["rules_input_semantic_sha256"] = digest.hexdigest()
        with tempfile.TemporaryDirectory(prefix="rovezero-pals-profile-test-") as directory:
            path = Path(directory) / "rules.json"
            path.write_text(json.dumps(declaration), encoding="utf-8")
            verified = read_rules_profile(path, ModelConfig())
            self.assertEqual(verified["rules_input_semantic_sha256"], digest.hexdigest())
            declaration["metadata_features"][0] = "changed_meaning"
            path.write_text(json.dumps(declaration), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonicalization"):
                read_rules_profile(path, ModelConfig())

    def test_public_only_or_missing_private_roles_cannot_pass_numeric_registration(self):
        metadata = {"checkpoint_sha256": "00" * 32, "config": ModelConfig().to_dict(), "trained": False, "training_steps": 0}
        manifest = {"schema": "rovezero.pals-model.v1", **metadata, "validator_present": False,
                    "roles": [], "graphs": [{"role": "public"}], "task_names": list(TASKS)}
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            validate_export_manifest(manifest, metadata)
        manifest["roles"] = ["proposer", "critic"]
        with self.assertRaisesRegex(ValueError, "graph count"):
            validate_export_manifest(manifest, metadata)

    def test_source_output_and_registered_overwrite_are_rejected(self):
        repository = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
        with self.assertRaises(ValueError): output_directory(repository / "models")
        with tempfile.TemporaryDirectory(prefix="rovezero-pals-test-") as temporary:
            output = output_directory(temporary)
            path = output / "receipt.json"
            atomic_json(path, {"status": "untrained", "training_steps": 0})
            digest = digest_file(path)
            self.assertEqual(len(digest), 64)
            with self.assertRaises(FileExistsError): atomic_json(path, {"status": "overwritten"})
            self.assertEqual(digest_file(path), digest)
            self.assertEqual(json.loads(path.read_text())["training_steps"], 0)


@unittest.skipUnless(torch is not None, "torch absent: real neural checks not executed")
class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = initialize(19)
        cls.data = fixture_input()
        cls.data.validate()

    def test_seed_is_deterministic_and_does_not_mutate_caller_rng(self):
        before = torch.get_rng_state().clone()
        with patch("torch.manual_seed", side_effect=AssertionError("global device RNG must not be reseeded")):
            same = initialize(19)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for key, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, same.state_dict()[key]), key)

    def test_public_fixed_kv_axes_are_annotated_without_freezing_batch_or_tokens(self):
        import onnx
        outputs = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT,
                   ["batch", "transpose_heads", "memory_tokens", "transpose_dimension"])
                   for name in ("memory_key", "memory_value")]
        document = onnx.helper.make_model(onnx.helper.make_graph([], "shape_metadata", [], outputs))
        annotate_public_memory_shapes(document, ModelConfig())
        for output in document.graph.output:
            dimensions = output.type.tensor_type.shape.dim
            self.assertEqual(dimensions[0].dim_param, "batch")
            self.assertEqual(dimensions[1].dim_value, 2)
            self.assertEqual(dimensions[2].dim_param, "memory_tokens")
            self.assertEqual(dimensions[3].dim_value, 64)
            self.assertFalse(dimensions[1].dim_param)
            self.assertFalse(dimensions[3].dim_param)
        document.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 3
        with self.assertRaisesRegex(ValueError, "contradicts"):
            annotate_public_memory_shapes(document, ModelConfig())

    def test_shared_pc_graph_has_one_outer_bank_and_real_if_private_routes(self):
        import onnx
        from rz_pals_model.onnx_shared import build_shared_pc_graph, audit_shared_pc_graph
        document, evidence = build_shared_pc_graph(self.model.without_validator())
        self.assertEqual(evidence["if_routes"], 6)
        self.assertEqual(evidence["shared_weight_bytes"], 7484928)
        self.assertEqual(evidence["branch_local_initializers"], 0)
        self.assertEqual([value.name for value in document.graph.input][0], "role_is_critic")
        self.assertEqual(len(document.graph.input[0].type.tensor_type.shape.dim), 0)
        self.assertEqual(document.graph.output[2].type.tensor_type.shape.dim[1].dim_value, 16)
        self.assertEqual(document.graph.output[2].type.tensor_type.shape.dim[2].dim_value, 384)
        bad = onnx.ModelProto()
        bad.CopyFrom(document)
        shared = next(value for value in bad.graph.initializer if value.name.startswith("reader_blocks."))
        bad.graph.initializer.append(shared)
        with self.assertRaisesRegex(ValueError, "single-owned"):
            audit_shared_pc_graph(bad, evidence["shared_parameters"])
        with self.assertRaisesRegex(ValueError, "V-free"):
            build_shared_pc_graph(self.model)

    def test_export_failure_preserves_caller_autograd_and_backend_flags(self):
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with tempfile.TemporaryDirectory(prefix="rovezero-pals-export-test-") as temporary:
            with torch.enable_grad(), patch("rz_pals_model.artifacts.load_checkpoint", return_value=(self.model, {"checkpoint_sha256": "00" * 32, "trained": False, "training_steps": 0})), patch("torch.onnx.export", side_effect=RuntimeError("injected export failure")):
                with self.assertRaisesRegex(RuntimeError, "injected export failure"):
                    export_checkpoint("unused-checkpoint", temporary)
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(flags, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))

    def test_v_free_equals_pc_and_excludes_private_v(self):
        free = self.model.without_validator()
        self.assertNotIn("validator", free.experts)
        self.assertFalse(any("validator" in key for key in free.state_dict()))
        with torch.no_grad():
            memory = self.model.public_encoder(*self.data.public_args())
            for role in ("proposer", "critic"):
                a = self.model.role_graph(role)(*self.data.role_args(memory))
                b = free.role_graph(role)(*self.data.role_args(memory))
                for expected, actual in zip(a, b): self.assertTrue(torch.equal(expected, actual))
            with self.assertRaises(ValueError): free.role_graph("validator")

    def test_private_v_mutation_cannot_change_pc_outputs(self):
        with torch.no_grad():
            memory = self.model.public_encoder(*self.data.public_args())
            before = self.model.role_graph("proposer")(*self.data.role_args(memory))
            validator = self.model.experts["validator"]
            original = {k: v.clone() for k, v in validator.state_dict().items()}
            try:
                for parameter in validator.parameters(): parameter.add_(100)
                after = self.model.role_graph("proposer")(*self.data.role_args(memory))
                for a, b in zip(before, after): self.assertTrue(torch.equal(a, b))
            finally:
                validator.load_state_dict(original)

    def test_role_order_candidates_and_mask_preserve_memory(self):
        with torch.no_grad():
            memory = self.model.public_encoder(*self.data.public_args())
            saved = tuple(value.clone() for value in memory)
            a = self.model.role_graph("proposer")(*self.data.role_args(memory))
            self.model.role_graph("critic")(*self.data.role_args(memory))
            b = self.model.role_graph("proposer")(*self.data.role_args(memory))
            for expected, actual in zip(a, b): self.assertTrue(torch.equal(expected, actual))
            for expected, actual in zip(saved, memory): self.assertTrue(torch.equal(expected, actual))
            permutation = torch.tensor([3, 0, 2, 1])
            permuted = dataclasses.replace(self.data, candidates=self.data.candidates[:, permutation, :], candidate_mask=self.data.candidate_mask[:, permutation])
            result = self.model.role_graph("proposer")(*permuted.role_args(memory))
            self.assertTrue(torch.equal(a[0][:, permutation], result[0]))
            self.assertTrue(torch.equal(a[1], result[1]))
            masked = dataclasses.replace(self.data, candidate_mask=torch.tensor([[True, True, False, True]]))
            result = self.model.role_graph("proposer")(*masked.role_args(memory))
            self.assertEqual(result[0][0, 2].item(), -1e9)
            self.assertTrue(torch.isfinite(result[2]).all())

    def test_records_are_independent_and_missing_shape_is_rejected(self):
        with torch.no_grad():
            a = self.model.public_encoder.encode_records(self.data.records)
            records = self.data.records.clone()
            records[:, 1, :] += 33
            b = self.model.public_encoder.encode_records(records)
            self.assertTrue(torch.equal(a[:, 0, :], b[:, 0, :]))
            self.assertFalse(torch.equal(a[:, 1, :], b[:, 1, :]))
        invalid = dataclasses.replace(self.data, query=torch.full((1, 16), float("nan")))
        with self.assertRaises(ValueError): invalid.validate()
        invalid = dataclasses.replace(self.data, board=self.data.board.float())
        with self.assertRaises(ValueError): invalid.validate()


if __name__ == "__main__": unittest.main()
