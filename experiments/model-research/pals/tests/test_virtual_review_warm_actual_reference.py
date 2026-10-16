"""원 Warm 실행 key·seed 연결의 CPU 검사. 실제 NN/수명 인수가 아니다."""
import copy
import importlib.abc
import struct
import sys
import unittest


class DenyNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split(".")[0] in ("torch", "numpy", "onnx", "onnxruntime")
                or fullname in ("rz_pals_model.model", "rz_pals_model.artifacts")
                or "optimizer" in fullname):
            raise AssertionError("CPU key test attempted neural/optimizer import: " + fullname)
        return None


GUARD = DenyNeuralImports()
sys.meta_path.insert(0, GUARD)
try:
    from rz_pals_model import virtual_review_warm_actual_reference as reference
    from rz_pals_model.virtual_review import INPUT_SCHEMA
finally:
    sys.meta_path.remove(GUARD)


def raw_fixture(role="proposer", seed=None):
    """수치·식별 전용 합성 fixture이며 Rules/NN 실행 증거가 아니다."""
    movement = {"from": 12, "to": 28, "promotion": 0}
    bits = [0x80000000] + [0x3f800000] * 6143
    data, latent = reference._bits(bits)
    identity = {"model_epoch": "01" * 32, "rules_history": "03" * 32,
                "backend": "02" * 32, "anchor_history_seals": ["03" * 32], "fp32": True}
    value = {
        "domain": reference.DOMAIN, "scope": reference.SCOPE,
        "case": "synthetic-fresh" if seed is None else "synthetic-warm",
        "physical_completion_known": True, "numerical_acceptance": False,
        "physical_fresh": seed is None, "accepted_seed_digest": seed, "actual_seed": None,
        "identity": identity, "bindings": {"checkpoint_sha256": identity["model_epoch"]},
        "line_bucket": 8,
        "input": {"schema": INPUT_SCHEMA, "role": role,
                  "anchors": [{"board": [0] * 64, "metadata": [-0.0] + [0.0] * 15}],
                  "records": [], "query": {
                      "purpose": "propose_policy" if role == "proposer" else "reply_policy",
                      "perspective": -1, "prefix": [movement], "proposal": [], "counter": [],
                      "counter_present": False, "anchors": [0, 0, 0]},
                  "candidates": [movement], "divergences": [], "token_budget": 512},
        "ticket": {"role": role.title(), "line_bucket": 8, "process_epoch": 7,
                   "request_sequence": 1 if seed is None else 2,
                   "view_sequence": 1 if seed is None else 2, "game": 1, "root": 1,
                   "authority_implementation": "04" * 32, "context_policy": "05" * 32,
                   "root_state": {"owner": 1, "revision": 0, "semantic": "06" * 32},
                   "representation": {"precision": "fp32", "model": identity["model_epoch"]}},
        "execution": {"epoch": 7, "sequence": 1 if seed is None else 2},
        "final_latent_bits": bits, "final_latent_sha256": latent,
        "raw_output": {"candidate_logits": [0.0], "wdl_logits": [0.0] * 3,
                       "private_latent": list(struct.unpack("<6144f", data)),
                       "divergence_logits": None if role == "proposer" else [], "task_logits": None},
        "completed_nn_inputs": 2,
        "owner_accounting": {"public_cache_hits": 0, "public_cache_misses": 1,
                             "public_nn_runs_attempted": 1, "public_nn_runs_completed": 1,
                             "public_nn_runs_failed_known": 0, "role_nn_runs_attempted": 1,
                             "role_nn_runs_completed": 1, "role_nn_runs_failed_known": 0,
                             "validator_nn_runs_attempted": 0, "validator_nn_runs_completed": 0,
                             "validator_nn_runs_failed_known": 0},
    }
    keys = reference._canonical_keys(value)
    value["ticket"]["representation"]["input"] = keys["fresh_key"]
    value["invocation_key"] = keys["fresh_key"] if seed is None else keys["warm_key"]
    return value


def original_chain(role="proposer"):
    fresh = raw_fixture(role)
    predecessor = {"raw": fresh,
                   "raw_pin": {"file": "synthetic-fresh.raw.json", "bytes": 1, "sha256": "07" * 32},
                   "retirement_pin": {"file": "synthetic-fresh.accepted.json", "bytes": 1, "sha256": "08" * 32}}
    warm = raw_fixture(role, fresh["final_latent_sha256"])
    warm["actual_seed"] = {"case": fresh["case"], "ticket": copy.deepcopy(fresh["ticket"]),
                           "raw": copy.deepcopy(predecessor["raw_pin"]),
                           "retirement": copy.deepcopy(predecessor["retirement_pin"]),
                           "latent_bits": list(fresh["final_latent_bits"]),
                           "latent_sha256": fresh["final_latent_sha256"]}
    return fresh, warm, predecessor


class ActualWarmCanonicalKeyTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, GUARD)

    def tearDown(self):
        sys.meta_path.remove(GUARD)

    def test_rust_golden_preserves_negative_zero_and_signed_perspective(self):
        fresh, warm, _ = original_chain()
        self.assertEqual(fresh["final_latent_sha256"],
                         "c6c71f043f19cd8d880510cb2416396e8e59a83070608e1a9e5a59ffaba5c159")
        self.assertEqual(fresh["invocation_key"],
                         "b038382273247dd0eec43bb7b86d5a57335bdbedac6ee7c696b3b0a2bb9cf278")
        self.assertEqual(warm["invocation_key"],
                         "f8a64059ec1d4391366b17d90c28e2dcdc1a5f5146ec9c5bfe26c08add7782b6")
        self.assertNotEqual(fresh["invocation_key"], warm["invocation_key"])

    def test_exact_fresh_and_retired_seed_chain_for_both_roles(self):
        for role in ("proposer", "critic"):
            with self.subTest(role=role):
                fresh, warm, predecessor = original_chain(role)
                self.assertIsNone(reference._validate_chain(fresh, None))
                self.assertEqual(reference._validate_chain(warm, predecessor), fresh["final_latent_bits"])

    def test_warm_rejects_fresh_alias_or_arbitrary_invocation_key(self):
        _, warm, _ = original_chain()
        for key in (warm["ticket"]["representation"]["input"], "00" * 32, "ab" * 32, None):
            with self.subTest(key=key):
                altered = copy.deepcopy(warm)
                altered["invocation_key"] = key
                with self.assertRaisesRegex(ValueError, "physical invocation"):
                    reference._validate_raw(altered)

    def test_fresh_rejects_warm_alias_or_wrong_prepared_input(self):
        fresh, warm, _ = original_chain()
        for field, value in (("invocation_key", warm["invocation_key"]), ("prepared", "aa" * 32)):
            altered = copy.deepcopy(fresh)
            if field == "prepared":
                altered["ticket"]["representation"]["input"] = value
            else:
                altered[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "physical invocation"):
                reference._validate_raw(altered)

    def test_seed_or_original_identity_mutation_invalidates_the_key(self):
        _, warm, _ = original_chain()
        for field in ("seed", "backend", "history", "anchor", "candidate", "perspective", "negative_zero"):
            altered = copy.deepcopy(warm)
            if field == "seed":
                altered["accepted_seed_digest"] = "ab" * 32
            elif field == "backend":
                altered["identity"]["backend"] = "ab" * 32
            elif field == "history":
                altered["identity"]["rules_history"] = "ab" * 32
                altered["identity"]["anchor_history_seals"][0] = "ab" * 32
            elif field == "anchor":
                altered["input"]["anchors"][0]["board"][0] = 1
            elif field == "candidate":
                altered["input"]["candidates"][0]["to"] = 20
            elif field == "perspective":
                altered["input"]["query"]["perspective"] = 1
            else:
                altered["input"]["anchors"][0]["metadata"][0] = 0.0
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "physical invocation"):
                reference._validate_raw(altered)

    def test_warm_requires_actual_seed_and_eligible_query(self):
        fresh, warm, predecessor = original_chain()
        self.assertIsNone(reference._validate_chain(fresh, None))
        self.assertEqual(reference._validate_chain(warm, predecessor), fresh["final_latent_bits"])
        cases = (
            ("missing_seed", "^original physical invocation identity differs$"),
            ("value_fresh", "^canonical ApproxWarm requires an eligible P/C query and actual seed digest$"),
            ("bool_fresh", "^original physical invocation identity differs$"),
            ("empty_history", "^canonical execution requires every ordered original identity digest$"),
            ("foreign_history", "^canonical execution requires every ordered original identity digest$"),
        )
        for field, rejection in cases:
            altered = copy.deepcopy(warm)
            if field == "missing_seed":
                altered["accepted_seed_digest"] = None
            elif field == "value_fresh":
                altered["input"]["query"]["purpose"] = "value_fresh"
            elif field == "bool_fresh":
                altered["physical_fresh"] = 0
            elif field == "empty_history":
                altered["identity"]["anchor_history_seals"] = []
            else:
                altered["identity"]["anchor_history_seals"][0] = "ab" * 32
            before = copy.deepcopy(altered)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, rejection):
                    reference._validate_raw(altered)
                self.assertEqual(altered, before)

    def test_retirement_seed_and_authority_cannot_be_replaced(self):
        fresh, warm, predecessor = original_chain()
        self.assertIsNone(reference._validate_chain(fresh, None))
        self.assertEqual(reference._validate_chain(warm, predecessor), fresh["final_latent_bits"])
        cases = (
            ("seed_bit", "^Warm seed differs from the exact original accepted predecessor$"),
            ("retirement", "^Warm seed differs from the exact original accepted predecessor$"),
            ("root", "^seed crossed original authority/role/context: root$"),
            ("request", "^spent request/view sequence reused$"),
            ("view", "^spent request/view sequence reused$"),
        )
        for field, rejection in cases:
            altered = copy.deepcopy(warm)
            if field == "seed_bit":
                altered["actual_seed"]["latent_bits"][1] ^= 1
            elif field == "retirement":
                altered["actual_seed"]["retirement"]["sha256"] = "ab" * 32
            elif field == "root":
                altered["ticket"]["root"] = 2
            elif field == "request":
                altered["ticket"]["request_sequence"] = 1
            else:
                altered["ticket"]["view_sequence"] = 1
            before = copy.deepcopy(altered)
            original_predecessor = copy.deepcopy(predecessor)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, rejection):
                    reference._validate_chain(altered, predecessor)
                self.assertEqual(altered, before)
                self.assertEqual(predecessor, original_predecessor)

    def test_unknown_completion_or_fake_nn_counters_never_become_a_key_pass(self):
        fresh, warm, predecessor = original_chain()
        self.assertIsNone(reference._validate_chain(fresh, None))
        self.assertEqual(reference._validate_chain(warm, predecessor), fresh["final_latent_bits"])
        cases = (
            ("unknown", "^original separate physical raw domain required$"),
            ("completed", "^original completed P/C-only owner accounting differs$"),
            ("attempted", "^original completed P/C-only owner accounting differs$"),
            ("validator", "^original completed P/C-only owner accounting differs$"),
            ("bool_counter", "^original completed P/C-only owner accounting differs$"),
        )
        for field, rejection in cases:
            altered = copy.deepcopy(warm)
            if field == "unknown":
                altered["physical_completion_known"] = None
            elif field == "completed":
                altered["completed_nn_inputs"] = 0
            elif field == "attempted":
                altered["owner_accounting"]["role_nn_runs_attempted"] = 0
            elif field == "validator":
                altered["owner_accounting"]["validator_nn_runs_completed"] = 1
            else:
                altered["owner_accounting"]["role_nn_runs_completed"] = True
            before = copy.deepcopy(altered)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, rejection):
                    reference._validate_raw(altered)
                self.assertEqual(altered, before)


if __name__ == "__main__":
    unittest.main()
