import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from rz_pals_model.config import ModelConfig, TASKS, matmul_flops
from rz_pals_model.artifacts import atomic_json, digest_file, output_directory

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
        same = initialize(19)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for key, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, same.state_dict()[key]), key)

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
