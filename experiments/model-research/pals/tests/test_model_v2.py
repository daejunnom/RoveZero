"""V2 admission, ordered-line semantics and independent private graph math."""
import copy
import dataclasses
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from rz_pals_model.artifacts import (FULL_LINE_RULES_BOUNDS, FULL_LINE_RULES_FEATURES,
                                    initialize_checkpoint, read_rules_profile, validate_rules_declaration)
from rz_pals_model.config import (DEFAULT_INIT_PROFILE, ENCODING_V2, MODEL_SEMANTICS_V2,
                                 ModelConfig, PROFILES, public_tensor_names, role_tensor_names)
from rz_pals_model.warm_artifacts import (warm_export_domain, _expected_private_inputs,
                                        _expected_private_outputs, _expected_public_inputs,
                                        validate_warm_export_manifest)

try:
    import torch
    from rz_pals_model.model import (CandidateInteractionHead, MoveEmbedding, fixture_input,
                                    initialize, relation_features)
except ImportError:
    torch = None


def rules_declaration(config):
    value = {"schema": "rovezero.pals-rules-descriptor.v2" if config.full_line else "rovezero.pals-rules-descriptor.v1",
             "rules_input_profile": "rz-pals-rules-full-line-v2" if config.full_line else "rz-pals-rules-fields-v1",
             "config": config.to_dict(), "rules_encoder_source_sha256": "00" * 32,
             "rules_base_semantics": "explicit_rules_features", "learned_input_compatibility": "unverified_declaration_only",
             "semantic_digest_algorithm": "sha256_u64le_length_prefixed_utf8_fields"}
    fields = [value["rules_input_profile"], value["rules_base_semantics"]]
    for name, count in (("metadata_features", 16), ("record_features", 16), ("query_features", 16), ("divergence_features", 8)):
        value[name] = [name + str(index) for index in range(count)]
        fields += value[name]
    if config.full_line:
        value["full_line_features"] = list(FULL_LINE_RULES_FEATURES)
        value["bounds"] = dict(FULL_LINE_RULES_BOUNDS)
        fields += value["full_line_features"]
    value["semantic_fields"] = fields
    digest = hashlib.sha256()
    for field in fields:
        encoded = field.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    value["rules_input_semantic_sha256"] = digest.hexdigest()
    return value


def warm_manifest(config, cuda=False):
    domain = warm_export_domain(config, cuda)
    declaration = rules_declaration(config)
    canonical = json.dumps(declaration, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    metadata = {"config": config.to_dict(), "checkpoint_sha256": "11" * 32, "trained": False, "training_steps": 0}
    value = {**metadata, **{key: domain[key] for key in ("schema", "layout", "layout_revision", "warm_graph_semantics", "private_seed_policy")},
             "model_semantics": config.model_semantics, "model_profile": config.profile, "encoding_schema": config.encoding,
             "approximate": True, "precision": "fp32", "tf32": False, "native_seed_owner_support": "not_registered_by_python_export",
             "batch_mode": "one_scalar_role_and_one_scalar_mode_per_physical_batch", "validator_present": False,
             "roles": ["proposer", "critic"], "task_names": ["defend_response", "attack_repair", "widen_responses", "lower_selectivity", "resume_task", "cross_profile_recheck", "defer"],
             "rules_input_profile": declaration["rules_input_profile"], "rules_input_declaration": declaration,
             "rules_input_semantic_sha256": declaration["rules_input_semantic_sha256"], "rules_encoder_source_sha256": "00" * 32,
             "rules_profile_descriptor_sha256": "22" * 32, "rules_profile_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
             "learned_input_compatibility": "unverified_declaration_only",
             "graphs": [{"role": "public", "file": "public_memory.onnx", "sha256": "33" * 32, "opset": 17,
                         "inputs": _expected_public_inputs(config), "outputs": [
                             {"name": "memory_key", "dtype": "FLOAT", "shape": ["batch", 2, "memory_tokens", 64]},
                             {"name": "memory_value", "dtype": "FLOAT", "shape": ["batch", 2, "memory_tokens", 64]},
                             {"name": "memory_mask", "dtype": "BOOL", "shape": ["batch", "memory_tokens"]}]},
                        {"role": domain["private_role"], "file": domain["graph_file"], "sha256": "44" * 32, "opset": 17,
                         "inputs": _expected_private_inputs(config), "outputs": _expected_private_outputs()}]}
    if cuda:
        value.update({key: domain[key] for key in ("execution_domain", "query_semantics")})
    return value, metadata


class ProfileTests(unittest.TestCase):
    def test_legacy_config_serialization_and_new_init_default(self):
        expected = {"width": 384, "query_heads": 6, "kv_heads": 2, "head_dimension": 64, "board_blocks": 2,
                    "record_blocks": 1, "record_fields": 4, "latent_slots": 16, "recurrent_blocks": 2, "iterations": 2,
                    "ffn_width": 1024, "max_records": 128, "max_candidates": 256, "max_divergences": 128}
        self.assertEqual(ModelConfig.baseline().to_dict(), expected)
        self.assertEqual(ModelConfig().profile, "legacy_summary_v1")
        self.assertEqual(inspect.signature(initialize_checkpoint).parameters["profile"].default, DEFAULT_INIT_PROFILE)
        for profile in PROFILES:
            config = ModelConfig.for_profile(profile)
            self.assertEqual(config.full_line, profile in ("full_line_v2", "full_line_interaction_v2"))
            self.assertEqual(config.interaction_head, profile in ("interaction_head_v2", "full_line_interaction_v2"))
            if profile != "legacy_summary_v1":
                self.assertEqual(config.to_dict()["profile"], profile)
                self.assertEqual(config.model_semantics, MODEL_SEMANTICS_V2)
                self.assertEqual(config.encoding, ENCODING_V2)
        with self.assertRaises(ValueError): ModelConfig.for_profile("unregistered")
        with self.assertRaises(ValueError): ModelConfig(width=64, profile=DEFAULT_INIT_PROFILE).validate()
        with self.assertRaises(ValueError): ModelConfig(record_blocks=True).validate()

    def test_rules_v2_extension_is_explicit_and_legacy_digest_domain_unchanged(self):
        for profile in PROFILES:
            config = ModelConfig.for_profile(profile)
            declaration = rules_declaration(config)
            self.assertEqual(len(declaration["semantic_fields"]), 67 if config.full_line else 58)
            self.assertEqual(validate_rules_declaration(declaration, config), declaration["rules_input_semantic_sha256"])
            with tempfile.TemporaryDirectory(prefix="rovezero-pals-v2-descriptor-") as directory:
                path = Path(directory) / "rules.json"
                path.write_text(json.dumps(declaration), encoding="utf-8")
                self.assertEqual(read_rules_profile(path, config)["rules_input_semantic_sha256"], declaration["rules_input_semantic_sha256"])
            if config.full_line:
                declaration["bounds"]["max_line_plies"] = 255
                with self.assertRaisesRegex(ValueError, "bounds"): validate_rules_declaration(declaration, config)

    def test_host_and_cuda_warm_v2_identity_and_shapes(self):
        config = ModelConfig.for_profile(DEFAULT_INIT_PROFILE)
        for cuda in (False, True):
            manifest, metadata = warm_manifest(config, cuda)
            self.assertEqual(validate_warm_export_manifest(manifest, metadata), ["proposer", "critic"])
            invalid = copy.deepcopy(manifest)
            next(value for value in invalid["graphs"][1]["inputs"] if value["name"] == "query_line_tokens")["shape"][2] = 255
            with self.assertRaisesRegex(ValueError, "tensor shape"): validate_warm_export_manifest(invalid, metadata)
            invalid = copy.deepcopy(manifest)
            invalid["model_profile"] = "full_line_v2"
            with self.assertRaisesRegex(ValueError, "identity"): validate_warm_export_manifest(invalid, metadata)
        with self.assertRaisesRegex(ValueError, "full-line"): warm_export_domain(ModelConfig(), cuda=True)


@unittest.skipUnless(torch is not None, "Torch absent: V2 neural checks not executed")
class V2NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = ModelConfig.for_profile(DEFAULT_INIT_PROFILE)
        cls.model = initialize(23, cls.config).without_validator()
        cls.data = fixture_input(profile=cls.config.profile)
        cls.data.validate(cls.config)

    def test_middle_line_changes_independent_record_but_padding_does_not(self):
        data = self.data
        with torch.no_grad():
            before = self.model.public_encoder.encode_records(data.records, data.record_line_tokens, data.record_line_mask)
            changed = data.record_line_tokens.clone()
            changed[:, 0, 1, 0] = 31
            after = self.model.public_encoder.encode_records(data.records, changed, data.record_line_mask)
            self.assertFalse(torch.equal(before[:, 0], after[:, 0]))
            self.assertTrue(torch.equal(before[:, 1:], after[:, 1:]))
            changed = data.record_line_tokens.clone()
            changed[~data.record_line_mask] = 2**62
            padded = dataclasses.replace(data, record_line_tokens=changed)
            padded.validate(self.config)
            actual = self.model.public_encoder.encode_records(data.records, changed, data.record_line_mask)
            self.assertTrue(torch.equal(before, actual))
            empty = self.model.public_encoder.line_encoder(changed, torch.zeros_like(data.record_line_mask))
            self.assertTrue(torch.equal(empty, torch.zeros_like(empty)))

    def test_relationships_and_middle_query_change_private_only(self):
        with torch.no_grad():
            memory = self.model.public_encoder(*self.data.public_args(self.config))
            before = self.model.role_graph("proposer")(*self.data.role_args(memory, self.config))
            refs = self.data.record_relations.clone()
            refs[:, 2] = torch.tensor([-1, 0])
            changed = dataclasses.replace(self.data, record_relations=refs)
            changed.validate(self.config)
            other_memory = self.model.public_encoder(*changed.public_args(self.config))
            for a, b in zip(memory, other_memory): self.assertTrue(torch.equal(a, b))
            after = self.model.role_graph("proposer")(*changed.role_args(memory, self.config))
            self.assertFalse(torch.equal(before[2], after[2]))
            lines = self.data.query_line_tokens.clone()
            lines[:, 1, 1, 0] = 31
            changed = dataclasses.replace(self.data, query_line_tokens=lines)
            changed.validate(self.config)
            after = self.model.role_graph("proposer")(*changed.role_args(memory, self.config))
            self.assertFalse(torch.equal(before[2], after[2]))

    def test_missing_partial_sparse_and_cyclic_payloads_rejected(self):
        with self.assertRaisesRegex(ValueError, "partial"): dataclasses.replace(self.data, query_line_mask=None).validate(self.config)
        with self.assertRaisesRegex(ValueError, "unexpected"): self.data.validate(ModelConfig())
        mask = self.data.query_line_mask.clone()
        mask[:, 0, 1] = False
        mask[:, 0, 2] = True
        with self.assertRaisesRegex(ValueError, "contiguous"): dataclasses.replace(self.data, query_line_mask=mask).validate(self.config)
        refs = self.data.record_relations.clone()
        refs[:, 0, 1] = 2
        with self.assertRaisesRegex(ValueError, "cycle"): dataclasses.replace(self.data, record_relations=refs).validate(self.config)

    def test_paired_profiles_keep_same_seed_common_and_added_banks(self):
        baseline = initialize(23, ModelConfig())
        input_only = initialize(23, ModelConfig.for_profile("full_line_v2"))
        head_only = initialize(23, ModelConfig.for_profile("interaction_head_v2"))
        both = initialize(23, self.config)
        states = [value.state_dict() for value in (baseline, input_only, head_only, both)]
        for name in set(states[0]) & set(states[1]) & set(states[2]) & set(states[3]):
            for state in states[1:]: self.assertTrue(torch.equal(states[0][name], state[name]), name)
        for name in set(states[1]) & set(states[3]):
            self.assertTrue(torch.equal(states[1][name], states[3][name]), name)
        for name in set(states[2]) & set(states[3]):
            self.assertTrue(torch.equal(states[2][name], states[3][name]), name)
        self.assertIn("experts.proposer.policy_head.weight", states[1])
        self.assertNotIn("experts.proposer.policy_head.weight", states[3])

    def test_nonlinear_head_can_reverse_destination_ranking_by_from_square(self):
        moves = MoveEmbedding(SimpleNamespace(width=2))
        head = CandidateInteractionHead(2)
        with torch.no_grad():
            for parameter in (*moves.parameters(), *head.parameters()): parameter.zero_()
            moves.from_square.weight[0, 0] = -4
            moves.from_square.weight[1, 0] = 4
            moves.to_square.weight[8, 1] = -0.5
            moves.to_square.weight[9, 1] = 0.5
            moves.projection.weight[0, 0] = 1
            moves.projection.weight[1, 3] = 1
            head.hidden.weight[0, 0:2] = 1
            head.output.weight[0, 0] = 1
            candidates = torch.tensor([[[0, 8, 0], [0, 9, 0], [1, 8, 0], [1, 9, 0]]])
            logits = head(moves(candidates), torch.zeros((1, 2))).flatten()
            self.assertGreater(float(logits[0]), float(logits[1]))
            self.assertLess(float(logits[2]), float(logits[3]))

    def test_warm_seed_is_complete_and_fresh_ignores_seed(self):
        with torch.no_grad():
            memory = self.model.public_encoder(*self.data.public_args(self.config))
            role = self.model.role_graph("proposer")
            fresh = role(*self.data.role_args(memory, self.config))
            wrapper = self.model.warm_role_graph("proposer")
            seed = fresh[2].clone()
            args = self.data.role_args(memory, self.config)
            actual = wrapper(*args, seed, torch.tensor(False))
            for a, b in zip(fresh, actual): self.assertTrue(torch.equal(a, b))
            warm = wrapper(*args, seed, torch.tensor(True))
            query = self.data.query.clone() + 0.25
            lines = self.data.query_line_tokens.clone()
            lines[:, 1, 1, 0] = 31
            changed = dataclasses.replace(self.data, query=query, query_line_tokens=lines)
            other = wrapper(*changed.role_args(memory, self.config), seed, torch.tensor(True))
            for a, b in zip(warm, other): self.assertTrue(torch.equal(a, b))

    @unittest.skipUnless(importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None,
                         "ONNX/ORT absent: independent V2 private numerical check not executed")
    def test_shared_and_warm_onnx_match_independent_torch_for_fresh_and_seeded(self):
        import numpy as np
        import onnxruntime as ort
        from rz_pals_model.onnx_shared import build_shared_pc_graph
        from rz_pals_model.onnx_warm import build_warm_pc_graph
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        documents = {"fresh": build_shared_pc_graph(self.model)[0], "warm": build_warm_pc_graph(self.model)[0]}
        sessions = {name: ort.InferenceSession(document.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
                    for name, document in documents.items()}
        with torch.no_grad():
            memory = self.model.public_encoder(*self.data.public_args(self.config))
            base = dict(zip(role_tensor_names(self.config), [value.numpy() for value in self.data.role_args(memory, self.config)]))
            for role in ("proposer", "critic"):
                reference = self.model.role_graph(role)(*self.data.role_args(memory, self.config))
                seed = reference[2].clone()
                for mode in (False, True):
                    expected = self.model.warm_role_graph(role)(*self.data.role_args(memory, self.config), seed, torch.tensor(mode))
                    feed = {**base, "role_is_critic": np.asarray(role == "critic"), "initial_latent": seed.numpy(), "warm_start": np.asarray(mode)}
                    actual = sessions["warm"].run(None, feed)
                    for a, b in zip(expected, actual): np.testing.assert_allclose(b, a.numpy(), atol=1e-4, rtol=1e-3)
                feed = {**base, "role_is_critic": np.asarray(role == "critic")}
                actual = sessions["fresh"].run(None, feed)
                for a, b in zip(reference, actual): np.testing.assert_allclose(b, a.numpy(), atol=1e-4, rtol=1e-3)

    @unittest.skipUnless(importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None,
                         "ONNX/ORT absent: paired ablation graph math not executed")
    def test_input_only_and_head_only_onnx_use_their_registered_tensor_contracts(self):
        import numpy as np
        import onnxruntime as ort
        from rz_pals_model.onnx_shared import build_shared_pc_graph
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        for profile in ("full_line_v2", "interaction_head_v2"):
            config = ModelConfig.for_profile(profile)
            model = initialize(23, config).without_validator()
            data = fixture_input(profile=profile)
            data.validate(config)
            document, _ = build_shared_pc_graph(model)
            session = ort.InferenceSession(document.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
            self.assertEqual({value.name for value in session.get_inputs()}, {"role_is_critic", *role_tensor_names(config)})
            with torch.no_grad():
                memory = model.public_encoder(*data.public_args(config))
                feed = dict(zip(role_tensor_names(config), [value.numpy() for value in data.role_args(memory, config)]))
                for role in ("proposer", "critic"):
                    expected = model.role_graph(role)(*data.role_args(memory, config))
                    actual = session.run(None, {**feed, "role_is_critic": np.asarray(role == "critic")})
                    for a, b in zip(expected, actual): np.testing.assert_allclose(b, a.numpy(), atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
