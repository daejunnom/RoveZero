"""Artifact integrity and continuous/resumed sampling contract regressions."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rz_data.errors import DataError
from rz_data.serialization import canonical_bytes, digest, loads
from rz_training.checkpoint import Sampler, code_digest, seal, verify


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "schema_version": 1, "artifact_kind": "checkpoint",
            "provenance": {"code": "fixture-code", "seed": 7},
            "step": 12, "parameters": [0.25, -0.1],
        }

    def test_sealed_roundtrip_preserves_original_and_checks_provenance(self):
        original = deepcopy(self.payload)
        receipt = seal(self.payload, max_bytes=4096)
        self.assertEqual(self.payload, original)
        self.assertNotIn("digest", self.payload)
        self.assertEqual(receipt["digest"], hashlib.sha256(canonical_bytes(self.payload)).hexdigest())
        loaded = loads(canonical_bytes(receipt))
        verified = verify(loaded, artifact_kind="checkpoint", provenance=self.payload["provenance"])
        self.assertEqual(verified, original)
        self.assertNotIn("digest", verified)
        self.assertIn("digest", loaded)

    def test_tampered_payload_and_wrong_provenance_are_rejected(self):
        receipt = seal(self.payload, max_bytes=4096)
        receipt["parameters"][0] = 0.5
        with self.assertRaises(DataError) as caught:
            verify(receipt, artifact_kind="checkpoint")
        self.assertEqual(caught.exception.code, "ArtifactDigestMismatch")
        receipt = seal(self.payload, max_bytes=4096)
        with self.assertRaises(DataError) as caught:
            verify(receipt, artifact_kind="checkpoint", provenance={"code": "new-source", "seed": 7})
        self.assertEqual(caught.exception.code, "ProvenanceMismatch")

    def test_receipt_and_verified_payload_do_not_alias_live_trainer_state(self):
        receipt = seal(self.payload, max_bytes=4096)
        expected_parameters = list(self.payload["parameters"])
        self.payload["parameters"][0] = 0.75
        verified = verify(receipt, artifact_kind="checkpoint")
        self.assertEqual(verified["parameters"], expected_parameters)
        verified["parameters"][0] = 0.5
        self.assertEqual(verify(receipt, artifact_kind="checkpoint")["parameters"], expected_parameters)

    def test_provenance_requires_exact_json_types_and_keys(self):
        self.payload["provenance"] = {"step": 1}
        receipt = seal(self.payload, max_bytes=4096)
        for expected in ({"step": True}, {"step": 1, "extra": None}, {}):
            with self.subTest(expected=expected), self.assertRaises(DataError):
                verify(receipt, artifact_kind="checkpoint", provenance=expected)

    def test_artifact_kind_schema_and_digest_syntax_are_checked(self):
        for field, invalid in (("schema_version", True), ("schema_version", 2),
                               ("schema_version", "1"), ("artifact_kind", "export")):
            with self.subTest(field=field, invalid=invalid):
                payload = deepcopy(self.payload)
                payload[field] = invalid
                with self.assertRaises(DataError):
                    verify(seal(payload, max_bytes=4096), artifact_kind="checkpoint")
        for invalid in (None, "0" * 63, "A" * 64, True):
            with self.subTest(digest=invalid):
                receipt = seal(self.payload, max_bytes=4096)
                receipt["digest"] = invalid
                with self.assertRaises(DataError):
                    verify(receipt, artifact_kind="checkpoint")

    def test_missing_schema_fields_and_nonobjects_are_typed_rejections(self):
        for missing in ("schema_version", "artifact_kind"):
            payload = deepcopy(self.payload)
            del payload[missing]
            with self.subTest(missing=missing), self.assertRaises(DataError):
                verify(seal(payload, max_bytes=4096), artifact_kind="checkpoint")
        for invalid in (None, [], "checkpoint", 1):
            with self.subTest(invalid=invalid), self.assertRaises(DataError):
                verify(invalid, artifact_kind="checkpoint")
            with self.subTest(seal=invalid), self.assertRaises(DataError):
                seal(invalid, max_bytes=4096)

    def test_seal_rejects_an_existing_digest_and_nonfinite_contents(self):
        self.payload["digest"] = "0" * 64
        with self.assertRaises(DataError) as caught:
            seal(self.payload, max_bytes=4096)
        self.assertEqual(caught.exception.code, "ExistingDigest")
        del self.payload["digest"]
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                payload = deepcopy(self.payload)
                payload["parameters"][0] = invalid
                with self.assertRaises(DataError):
                    seal(payload, max_bytes=4096)
                payload["digest"] = "0" * 64
                with self.assertRaises(DataError):
                    verify(payload, artifact_kind="checkpoint")

    def test_final_receipt_bytes_and_invalid_limits_are_bounded(self):
        payload_size = len(canonical_bytes(self.payload))
        with self.assertRaises(DataError) as caught:
            seal(self.payload, max_bytes=payload_size)
        self.assertEqual(caught.exception.code, "OutputByteLimit")
        receipt = seal(self.payload, max_bytes=4096)
        size = len(canonical_bytes(receipt))
        self.assertEqual(seal(self.payload, max_bytes=size), receipt)
        self.assertEqual(verify(receipt, artifact_kind="checkpoint", max_bytes=size), self.payload)
        with self.assertRaises(DataError):
            verify(receipt, artifact_kind="checkpoint", max_bytes=size - 1)
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(max_bytes=invalid), self.assertRaises(DataError):
                seal(self.payload, max_bytes=invalid)

    def test_code_digest_uses_package_relative_python_raw_bytes_only(self):
        with tempfile.TemporaryDirectory(prefix="rz-code-fixture-") as directory:
            root = Path(directory)
            packages = {}
            expected = []
            for package_name in ("rz_data", "rz_training"):
                package_root = root / package_name
                package_root.mkdir()
                package = type("Package", (), {"__file__": str(package_root / "__init__.py")})
                packages[package_name] = package
                for name, raw in (("z.py", b"z = 1\r\n"), ("__init__.py", b"# fixture\n")):
                    (package_root / name).write_bytes(raw)
                    expected.append({"path": package_name + "/" + name,
                                     "sha256": hashlib.sha256(raw).hexdigest()})
                (package_root / "private.txt").write_text("excluded", encoding="utf-8")
            with patch("rz_training.checkpoint.importlib.import_module", side_effect=packages.__getitem__):
                first = code_digest()
                self.assertEqual(first, digest(sorted(expected, key=lambda item: item["path"])))
                (root / "rz_data" / "private.txt").write_text("changed", encoding="utf-8")
                self.assertEqual(code_digest(), first)
                (root / "rz_training" / "z.py").write_bytes(b"z = 1\n")
                self.assertNotEqual(code_digest(), first)


class SamplerTests(unittest.TestCase):
    def test_initial_epoch_is_a_saved_permutation_with_json_rng(self):
        sampler = Sampler(9, 7)
        state = sampler.state_dict()
        self.assertEqual(state["epoch"], 0)
        self.assertEqual(state["cursor"], 0)
        self.assertEqual(sorted(state["order"]), list(range(9)))
        self.assertEqual(state["rng_state"][0], 3)
        self.assertEqual(len(state["rng_state"][1]), 625)
        self.assertIsNone(state["rng_state"][2])
        self.assertEqual(loads(canonical_bytes(state)), state)

    def test_split_draws_match_continuous_draw_values_and_saved_state(self):
        for size in (1, 3, 9, 64):
            for first, second in ((1, 1), (3, 7), (17, 31), (64, 192)):
                with self.subTest(size=size, first=first, second=second):
                    split = Sampler(size, 19)
                    continuous = Sampler(size, 19)
                    indices = split.draw(first) + split.draw(second)
                    self.assertEqual(indices, continuous.draw(first + second))
                    self.assertEqual(split.state_dict(), continuous.state_dict())

    def test_consumed_epoch_boundary_resumes_before_the_next_shuffle(self):
        sampler = Sampler(7, 5)
        expected_order = sampler.state_dict()["order"]
        self.assertEqual(sampler.draw(7), expected_order)
        state = sampler.state_dict()
        self.assertEqual((state["epoch"], state["cursor"]), (0, 7))
        resumed = Sampler(7, 999)
        resumed.load_state_dict(loads(canonical_bytes(state)))
        self.assertEqual(resumed.draw(21), sampler.draw(21))
        self.assertEqual(resumed.state_dict(), sampler.state_dict())

    def test_partial_epoch_resume_and_state_copies_are_independent(self):
        sampler = Sampler(13, 17)
        sampler.draw(5)
        state = sampler.state_dict()
        resumed = Sampler(13, 4)
        resumed.load_state_dict(state)
        self.assertEqual(resumed.draw(64), sampler.draw(64))
        state["order"].reverse()
        state["rng_state"][1][0] = 0
        self.assertEqual(resumed.state_dict(), sampler.state_dict())

    def test_constructor_and_draw_reject_bool_invalid_count_and_size(self):
        for size, seed in ((True, 1), (0, 1), (-1, 1), (10001, 1), (2, True), (2, -1)):
            with self.subTest(size=size, seed=seed), self.assertRaises(DataError):
                Sampler(size, seed)
        sampler = Sampler(3, 1)
        for count in (True, 0, -1, 257, 1.5):
            state = sampler.state_dict()
            with self.subTest(count=count), self.assertRaises(DataError):
                sampler.draw(count)
            self.assertEqual(sampler.state_dict(), state)

    def test_invalid_state_fields_and_permutations_are_rejected_atomically(self):
        sampler = Sampler(4, 8)
        good = sampler.state_dict()
        invalid_states = []
        for key, value in (("size", True), ("size", 3), ("epoch", True),
                           ("epoch", 1000001), ("cursor", True), ("cursor", 5),
                           ("order", [0, 1, 2, 2]), ("order", [0, 1, 2, 4]),
                           ("order", [True, 1, 2, 3]), ("order", [0, 1, 2])):
            state = deepcopy(good)
            state[key] = value
            invalid_states.append(state)
        missing = deepcopy(good)
        del missing["epoch"]
        invalid_states.extend([missing, {**good, "extra": None}, None])
        for state in invalid_states:
            with self.subTest(state=state), self.assertRaises(DataError):
                sampler.load_state_dict(state)
            self.assertEqual(sampler.state_dict(), good)

    def test_unsafe_rng_shape_version_words_index_and_gaussian_are_rejected(self):
        sampler = Sampler(4, 8)
        good = sampler.state_dict()
        rngs = []
        for field, value in ((0, True), (0, 2), (1, []), (2, 0.0), (2, float("nan"))):
            rng = deepcopy(good["rng_state"])
            rng[field] = value
            rngs.append(rng)
        for index, value in ((0, True), (0, -1), (0, 2**32), (-1, True), (-1, 625)):
            rng = deepcopy(good["rng_state"])
            rng[1][index] = value
            rngs.append(rng)
        rngs.extend([None, tuple(good["rng_state"]), good["rng_state"] + [None]])
        for rng in rngs:
            state = deepcopy(good)
            state["rng_state"] = rng
            with self.subTest(rng_type=type(rng).__name__), self.assertRaises(DataError):
                sampler.load_state_dict(state)
            self.assertEqual(sampler.state_dict(), good)

    def test_epoch_limit_preserves_state_on_rejection(self):
        sampler = Sampler(3, 8)
        state = sampler.state_dict()
        state.update(epoch=1000000, cursor=2)
        sampler.load_state_dict(state)
        self.assertEqual(len(sampler.draw(1)), 1)
        before = sampler.state_dict()
        with self.assertRaises(DataError) as caught:
            sampler.draw(1)
        self.assertEqual(caught.exception.code, "SamplerEpochLimit")
        self.assertEqual(sampler.state_dict(), before)


if __name__ == "__main__":
    unittest.main()
