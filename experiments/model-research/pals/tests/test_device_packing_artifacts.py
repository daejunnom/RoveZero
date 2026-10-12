"""CPU routing/manifest evidence only; no CUDA owner, native or model claims."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest

from rz_pals_model import device_packing_artifacts as packing

try:
    import numpy as np
except ImportError:
    np = None
try:
    import onnx
except ImportError:
    onnx = None
try:
    import onnxruntime as ort
except ImportError:
    ort = None


BOARD = "11" * 32
MODEL = "22" * 32
PUBLIC = "33" * 32
GENERATION = 7


def owned(array):
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def block(rows=1, board=BOARD, offset=0, zero_pad=False):
    tokens = packing.BOARD_TOKENS + rows
    shape = (1, 2, tokens, 64)
    key = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + offset
    value = -key - np.float32(0.5)
    # A routing operation must preserve FP32 bits, including signed zero.
    key[0, 0, 0, 0] = np.float32(-0.0)
    features = np.arange(rows * 16, dtype=np.float32).reshape(rows, 16) + offset + 1
    if zero_pad:
        features[:] = 0
    return packing.CpuProjectionBlock(board, MODEL, PUBLIC, GENERATION,
                                      owned(key), owned(value), owned(features),
                                      (False,) if zero_pad else (True,) * rows)


def prepare(base, references=(), **changes):
    options = {"board_identity": BOARD, "model_identity": MODEL,
               "public_graph_identity": PUBLIC, "game_generation": GENERATION,
               "record_count": len(references),
               "expected_projection_keys": tuple(packing.projection_key(r.block, r.record_index)
                                                   for r in references)}
    options.update(changes)
    return packing.prepare_cpu_inputs(base, references, **options)


def change_feed(prepared, name, value):
    feeds = dict(prepared.feeds)
    feeds[name] = value
    return replace(prepared, feeds=MappingProxyType(feeds))


@unittest.skipUnless(np is not None, "NumPy is required for actual CPU input checks")
class InputWitnessTests(unittest.TestCase):
    def test_exact_digest_types_reject_custom_equality_in_prepare_and_revalidation(self):
        class EqualDigest:
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        base = block(zero_pad=True)
        empty = prepare(base)
        for field in ("model_identity", "public_graph_identity"):
            bad_base = replace(base, **{field: EqualDigest()})
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare(bad_base)
            forged = replace(empty, base=bad_base, owners=(bad_base,))
            direct = packing.PreparedPackingInputs(**vars(forged))
            for witness in (forged, direct):
                with self.assertRaises(ValueError):
                    packing.validate_prepared_inputs(witness)
        source = block()
        refs = (packing.RecordReference(source, 0),)
        with self.assertRaises(ValueError):
            prepare(source, refs, expected_projection_keys=(EqualDigest(),))
        real = prepare(source, refs)
        forged = replace(real, expected_projection_keys=(EqualDigest(),))
        direct = packing.PreparedPackingInputs(**vars(forged))
        for witness in (forged, direct):
            with self.assertRaises(ValueError):
                packing.validate_prepared_inputs(witness)

    def test_unique_full_owners_ordered_duplicates_and_strong_aliases(self):
        base, source = block(), block(3, board="44" * 32, offset=100)
        refs = (packing.RecordReference(source, 2), packing.RecordReference(source, 0),
                packing.RecordReference(source, 2))
        prepared = prepare(base, refs)
        self.assertEqual(len(prepared.feeds), 387)
        self.assertEqual(prepared.owners, (base, source))
        self.assertIs(prepared.feeds["record_key_000"], source.key)
        self.assertIs(prepared.feeds["record_key_002"], source.key)
        self.assertIs(prepared.feeds["record_key_003"], base.key)
        self.assertEqual([int(prepared.feeds[f"record_offset_{i:03}"][0]) for i in range(3)],
                         [68, 66, 68])
        expected = sum(a.nbytes for b in (base, source)
                       for a in (b.key, b.value, b.record_features))
        expected += 129 * 8 + 69
        self.assertEqual(prepared.unique_owned_payload_bytes, expected)
        self.assertIs(packing.validate_prepared_inputs(prepared), prepared)
        self.assertTrue(prepared.current_mask.all())

    def test_zero_records_require_actual_zero_feature_false_mask_projection(self):
        base = block(zero_pad=True)
        prepared = prepare(base)
        self.assertEqual(prepared.current_mask.shape, (1, 67))
        self.assertFalse(bool(prepared.current_mask[0, 66]))
        self.assertNotEqual(float(base.key[0, 0, 66, 1]), 0.0)
        with self.assertRaisesRegex(ValueError, "zero records"):
            prepare(block())
        with self.assertRaisesRegex(ValueError, "zero records"):
            prepare(replace(base, record_mask=(True,)))
        features = base.record_features.copy()
        features[0, 0] = np.float32(-0.0)
        features.setflags(write=False)
        with self.assertRaisesRegex(ValueError, "zero records"):
            prepare(replace(base, record_features=features))

    def test_identity_and_count_rejections(self):
        base, source = block(), block(2)
        refs = (packing.RecordReference(source, 0),)
        for field, value in (("board_identity", "55" * 32),
                             ("model_identity", "66" * 32),
                             ("public_graph_identity", "77" * 32),
                             ("game_generation", 8), ("record_count", True),
                             ("record_count", 129), ("record_count", 0),
                             ("expected_projection_keys", ("00" * 32,))):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                prepare(base, refs, **{field: value})
        for index in (-1, 2, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                packing.prepare_cpu_inputs(base, (packing.RecordReference(source, index),),
                    board_identity=BOARD, model_identity=MODEL, public_graph_identity=PUBLIC,
                    game_generation=GENERATION, record_count=1,
                    expected_projection_keys=("00" * 32,))
        with self.assertRaises(ValueError):
            prepare(base, (packing.RecordReference(replace(source, record_mask=(False, True)), 0),))

    def test_actual_array_dtype_shape_backing_finiteness_and_budget(self):
        base = block(zero_pad=True)
        nonfinite = base.key.copy()
        nonfinite[0, 0, 0, 0] = np.nan
        nonfinite.setflags(write=False)
        bigger = owned(np.zeros((1, 2, 68, 64), dtype=np.float32))
        candidates = [owned(base.key.astype(np.float64)), base.key.copy(),
                      base.key[:, :, :, :], bigger[:, :, :67, :], nonfinite,
                      owned(np.zeros((1, 2, 66, 64), dtype=np.float32))]
        for key in candidates:
            with self.subTest(dtype=str(key.dtype), shape=key.shape), self.assertRaises(ValueError):
                prepare(replace(base, key=key))
        with self.assertRaisesRegex(ValueError, "budget"):
            prepare(base, maximum_owned_payload_bytes=1)

    def test_revalidation_rejects_alias_control_mask_or_accounting_drift(self):
        source = block(2)
        prepared = prepare(source, (packing.RecordReference(source, 1),))
        candidates = [change_feed(prepared, "record_key_000", owned(source.key)),
                      change_feed(prepared, "board_value", owned(source.value)),
                      change_feed(prepared, "record_offset_000", owned(np.array([66], dtype=np.int64))),
                      change_feed(prepared, "record_offset_000", owned(np.array([67], dtype=np.int32))),
                      change_feed(prepared, "record_stop", owned(np.array([194], dtype=np.int64))),
                      replace(prepared, current_mask=owned(np.zeros((1, 67), dtype=np.bool_))),
                      replace(prepared, feeds=dict(prepared.feeds)),
                      change_feed(prepared, "unrecognized_port", owned(np.array([1], dtype=np.int64))),
                      replace(prepared, unique_owned_payload_bytes=1),
                      replace(prepared, owners=())]
        for index, candidate in enumerate(candidates):
            with self.subTest(index=index), self.assertRaises(ValueError):
                packing.validate_prepared_inputs(candidate)


@unittest.skipUnless(onnx is not None, "ONNX is required for actual artifact validation")
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = packing.graph_bytes()
        cls.manifest = packing.manifest_for_graph(cls.raw)

    def test_fixed_weight_free_interface_and_arithmetic_from_node_shapes(self):
        document = onnx.load_model_from_string(self.raw)
        self.assertEqual(len(document.graph.input), 387)
        self.assertEqual(len(document.graph.output), 3)
        self.assertEqual(len(document.graph.node), 275)
        self.assertEqual(len(document.graph.initializer), 5)
        self.assertTrue(all(t.data_type == onnx.TensorProto.INT64 for t in document.graph.initializer))
        self.assertFalse(document.functions)
        self.assertEqual(self.manifest["learned_weights"], {"count": 0, "bytes": 0})
        self.assertFalse(self.manifest["scope"]["native_connected"])
        ledger = self.manifest["intermediate_arithmetic"]
        self.assertEqual(ledger["node_output_count"], len(document.graph.node))
        self.assertEqual({row["output"] for row in ledger["node_outputs"]},
                         {name for node in document.graph.node for name in node.output})
        # Independently check FLOAT/Bools/INT64 casts at maximum 194 tokens.
        per_kv_layout = (66 + 128 + 194 + 194) * 2 * 64 * 4
        per_finite = 194 * 2 * 64 * (1 + 1 + 1 + 8) + 8 + 1
        self.assertEqual(ledger["maximum_sum_payload_bytes"], 2 * (per_kv_layout + per_finite) + 1)
        self.assertIn("not_peak", ledger["meaning"])
        self.assertEqual(ledger["allocator_workspace_and_session_bytes"], "unknown")
        packing.read_manifest_bytes(json.dumps(self.manifest).encode(), self.raw)

    def test_graph_and_strict_manifest_reject_foreign_or_retyped_inputs(self):
        document = onnx.load_model_from_string(self.raw)
        document.graph.initializer.append(onnx.helper.make_tensor("foreign_weight", onnx.TensorProto.FLOAT, [1], [1.0]))
        foreign = document.SerializeToString(deterministic=True)
        with self.assertRaisesRegex(ValueError, "exact weight-free"):
            packing.manifest_for_graph(foreign)
        mutations = [lambda m: m.update(extra="unrecognized"),
                     lambda m: m.update(layout_revision=True),
                     lambda m: m["graph"]["inputs"][0].update(shape=tuple(m["graph"]["inputs"][0]["shape"])),
                     lambda m: m["learned_weights"].update(count=1),
                     lambda m: m["graph"].update(sha256="00" * 32),
                     lambda m: m["graph"]["inputs"][0].update(dtype="FLOAT16"),
                     lambda m: m["bounds"].update(record_capacity=129),
                     lambda m: m["scope"].update(native_connected=True),
                     lambda m: m["intermediate_arithmetic"].update(maximum_sum_payload_bytes=0)]
        for mutation in mutations:
            changed = copy.deepcopy(self.manifest)
            mutation(changed)
            with self.assertRaises(ValueError):
                packing.validate_manifest(changed, self.raw)
        for raw in (b'{"schema":"x","schema":"y"}', b'{"value":NaN}', b"[]"):
            with self.assertRaises(ValueError):
                packing.read_manifest_bytes(raw, self.raw)
        with self.assertRaises(ValueError):
            packing.read_manifest_bytes(b" " * (packing.MAX_MANIFEST_BYTES + 1), self.raw)

    def test_export_is_outside_source_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-pack"
            manifest = packing.export_device_packing_artifact(output)
            raw = (output / packing.GRAPH_FILE).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["graph"]["sha256"])
            packing.read_manifest_bytes((output / packing.MANIFEST_FILE).read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                packing.export_device_packing_artifact(output)
        with self.assertRaisesRegex(ValueError, "source checkout"):
            packing.export_device_packing_artifact(Path(packing.__file__).resolve().parent / "forbidden-generated")
        with self.assertRaisesRegex(ValueError, "absolute"):
            packing.export_device_packing_artifact("relative-generated")


@unittest.skipUnless(np is not None and onnx is not None and ort is not None,
                     "NumPy, ONNX and ORT are required for actual CPU numeric checks")
class CpuNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        cls.session = ort.InferenceSession(packing.graph_bytes(), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        if cls.session.get_providers() != ["CPUExecutionProvider"]:
            raise AssertionError("focused numeric test requires exact CPU provider")

    def assert_routes(self, prepared):
        packing.validate_prepared_inputs(prepared)
        key, value, finite = self.session.run(None, dict(prepared.feeds))
        for kind, actual in (("key", key), ("value", value)):
            board = getattr(prepared.base, kind)[:, :, :66, :]
            if prepared.record_count:
                records = [getattr(r.block, kind)[:, :, 66 + r.record_index:67 + r.record_index, :]
                           for r in prepared.references]
            else:
                records = [getattr(prepared.base, kind)[:, :, 66:67, :]]
            expected = np.concatenate([board] + records, axis=2)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        self.assertEqual(np.asarray(finite).shape, ())
        self.assertTrue(bool(finite))
        self.assertEqual(key.shape[2], 66 + max(prepared.record_count, 1))
        return key, value

    def test_zero_uses_actual_nonzero_projection_and_false_current_mask(self):
        prepared = prepare(block(zero_pad=True))
        key, _ = self.assert_routes(prepared)
        self.assertNotEqual(float(key[0, 0, 66, 1]), 0)
        self.assertFalse(bool(prepared.current_mask[0, 66]))

    def test_mixed_blocks_current_order_and_duplicate_occurrences(self):
        base, a, b = block(2), block(4, board="44" * 32, offset=200), block(3, board="55" * 32, offset=500)
        refs = (packing.RecordReference(b, 2), packing.RecordReference(a, 0),
                packing.RecordReference(b, 2), packing.RecordReference(base, 1))
        self.assert_routes(prepare(base, refs))

    def test_capacity_128_and_shared_alias_backing(self):
        base = block(128)
        refs = tuple(packing.RecordReference(base, 127 - i) for i in range(128))
        prepared = prepare(base, refs)
        self.assertEqual(len(prepared.owners), 1)
        self.assertEqual(self.assert_routes(prepared)[0].shape, (1, 2, 194, 64))

    def test_finite_flag_is_for_visible_join_only_and_input_boundary_is_stricter(self):
        base = block(2)
        prepared = prepare(base, (packing.RecordReference(base, 1),))
        active_nan = base.key.copy()
        active_nan[0, 1, 67, 3] = np.nan
        feeds = dict(prepared.feeds)
        feeds["record_key_000"] = active_nan
        self.assertFalse(bool(self.session.run(None, feeds)[2]))
        active_inf = base.value.copy()
        active_inf[0, 0, 0, 2] = np.inf
        feeds = dict(prepared.feeds)
        feeds["board_value"] = active_inf
        self.assertFalse(bool(self.session.run(None, feeds)[2]))
        feeds = dict(prepared.feeds)
        feeds["record_key_127"] = np.full(base.key.shape, np.inf, dtype=np.float32)
        self.assertTrue(bool(self.session.run(None, feeds)[2]))
        active_nan.setflags(write=False)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            prepare(replace(base, key=active_nan), (packing.RecordReference(base, 1),))


if __name__ == "__main__":
    unittest.main()
