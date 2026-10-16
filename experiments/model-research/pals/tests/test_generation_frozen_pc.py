"""Pure frozen-bank contract checks; no tensor/model/export execution.

The metadata fixtures contain arbitrary hashes, not checkpoint or ONNX bytes.
Their successful parse establishes schema agreement only. They cannot issue a
content proof, satisfy actual generation quotas, or arm a native capability.
"""
import builtins
import copy
import hashlib
import importlib.abc
import io
import json
import math
import struct
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from types import ModuleType
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural imports forbidden in frozen P/C pure tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import generation_frozen_pc as frozen
    from rz_pals_model import generation_data as data
    from rz_pals_model.config import ModelConfig, MODEL_SEMANTICS_V3
    from rz_pals_model.virtual_review import registered_encoding_sha256
finally:
    sys.meta_path.remove(_guard)


def metadata_sha(label):
    """An arbitrary fixture identifier, never a tensor content digest."""
    return hashlib.sha256(("schema-fixture-only:" + label).encode("ascii")).hexdigest()


def request_fixture():
    return {"schema": frozen.REQUEST_DOMAIN, "source_sha256": metadata_sha("source"),
        "model_epoch": 1, "config": ModelConfig.for_profile("virtual_review_v1").to_dict(),
        "checkpoint": {"file": "g0/checkpoint.pt", "sha256": metadata_sha("checkpoint"),
                       "bytes": 64 * 1024**2, "domain": MODEL_SEMANTICS_V3},
        "export_manifest": {"file": "g0/export.json", "sha256": metadata_sha("export-manifest"), "bytes": 4096},
        "graphs": [{"role": role, "file": "g0/" + file,
                    "sha256": metadata_sha(role), "bytes": 64 * 1024**2}
                   for role, file in (("public", "public_memory.onnx"),
                                      ("shared_pc", "shared_pc_if.onnx"))],
        "verifier_sources": [{"file": name, "sha256": metadata_sha(name), "bytes": 1024}
                             for name in frozen.SOURCE_NAMES],
        "expected_frozen_pc_sha256": metadata_sha("named-parameter-bank"),
        "limits": dict(frozen.LIMITS)}


def proof_fixture(request=None):
    """Complete metadata shapes only; no decoded bytes or actual issuer proof."""
    request = copy.deepcopy(request or request_fixture())
    inventory, graphs, mapping = [], [], []
    specs = frozen.registered_pc_parameter_spec(ModelConfig(**request["config"]))
    for spec in specs:
        inventory.append({key: copy.deepcopy(spec[key])
                          for key in ("name", "shape", "dtype", "owner", "semantics")} |
                         {"bytes": 4 * math.prod(spec["shape"]),
                          "sha256": metadata_sha("parameter:" + spec["name"])})
    indexed = {item["name"]: item for item in inventory}
    for pin in request["graphs"]:
        initializers = []
        for spec in specs:
            if pin["role"] not in spec["graphs"]:
                continue
            transpose = spec["transform"] == "transpose_2d"
            name = spec["name"] + (".linear_transpose" if transpose else "")
            source_sha = indexed[spec["name"]]["sha256"]
            target_sha = metadata_sha("transpose:" + spec["name"]) if transpose else source_sha
            initializers.append({"name": name, "dtype": "FLOAT",
                "shape": list(reversed(spec["shape"])) if transpose else spec["shape"].copy(),
                "bytes": indexed[spec["name"]]["bytes"], "sha256": target_sha})
            mapping.append({"graph_role": pin["role"], "parameter": spec["name"],
                "initializer": name, "transform": spec["transform"],
                "source_sha256": source_sha, "initializer_sha256": target_sha})
        initializers.sort(key=lambda item: item["name"])
        graphs.append(dict(pin, structure_sha256=metadata_sha("structure:" + pin["role"]),
                           initializers=initializers, initializer_inventory_sha256=data.digest(
                               "rz-pals-onnx-readback-initializers/1", initializers)))
    mapping.sort(key=lambda item: (item["graph_role"], item["parameter"]))
    return {"schema": frozen.PROOF_DOMAIN, "request": request,
        "request_sha256": frozen.request_digest(request), "scope": frozen.SCOPE,
        "checkpoint_scope": "own_initialization_tensor_content_only" if request["checkpoint"]["domain"] == MODEL_SEMANTICS_V3 else "actual_pc_frozen_tensor_content_only",
        "frozen_pc_parameter_sha256": request["expected_frozen_pc_sha256"] or metadata_sha("bank"),
        "parameter_inventory": inventory,
        "parameter_inventory_sha256": data.digest(frozen.PARAMETER_DOMAIN, inventory),
        "graphs": graphs, "mapping": mapping,
        "mapping_sha256": data.digest(frozen.MAPPING_DOMAIN, mapping),
        "encoding_registry_sha256": registered_encoding_sha256(),
        "verification": dict(frozen.VERIFICATION)}


def reseal_inventory(proof):
    proof["parameter_inventory_sha256"] = data.digest(frozen.PARAMETER_DOMAIN, proof["parameter_inventory"])


def reseal_graph(graph):
    graph["initializer_inventory_sha256"] = data.digest("rz-pals-onnx-readback-initializers/1", graph["initializers"])


def reseal_mapping(proof):
    proof["mapping_sha256"] = data.digest(frozen.MAPPING_DOMAIN, proof["mapping"])


def reseal_request(proof):
    proof["request_sha256"] = frozen.request_digest(proof["request"])


def stored_zip(entries):
    """Small owned ZIP strings, not a Torch checkpoint or pickle payload."""
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, contents in entries:
                archive.writestr(name, contents)
    stream.seek(0)
    return stream


class PureTests(unittest.TestCase):
    def setUp(self):
        guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)
        original_import = builtins.__import__

        def forbidden_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
                raise AssertionError("neural imports forbidden even when cached")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = forbidden_import
        self.addCleanup(setattr, builtins, "__import__", original_import)

    def validate_proof(self, proof, **kwargs):
        return frozen.validate_frozen_pc_proof(proof,
            expected_request_sha256=proof["request_sha256"], **kwargs)


class RegisteredBankTests(PureTests):
    def test_unique_count_occurrences_and_bytes_are_source_arithmetic(self):
        specs = frozen.registered_pc_parameter_spec()
        self.assertEqual(len(specs), 104)
        self.assertEqual(len({item["name"] for item in specs}), 104)
        self.assertEqual(sum(len(item["graphs"]) for item in specs), 112)
        self.assertEqual(sum(4 * math.prod(item["shape"]) for item in specs), 52948992)
        self.assertEqual([item["name"] for item in specs], sorted(item["name"] for item in specs))
        self.assertEqual({item["owner"] for item in specs}, {"shared", "proposer", "critic"})
        self.assertFalse(any("validator" in item["name"] for item in specs))

    def test_only_shared_line_bank_has_two_graph_occurrences(self):
        repeated = [item for item in frozen.registered_pc_parameter_spec() if len(item["graphs"]) == 2]
        self.assertEqual(len(repeated), 8)
        self.assertTrue(all(item["name"].startswith("public_encoder.line_encoder.") for item in repeated))
        self.assertTrue(all(item["graphs"] == ["public", "shared_pc"] for item in repeated))

    def test_transform_does_not_change_parameter_ownership_or_byte_count(self):
        for item in frozen.registered_pc_parameter_spec():
            with self.subTest(name=item["name"]):
                self.assertEqual(item["dtype"], "fp32_le")
                self.assertEqual(item["semantics"], MODEL_SEMANTICS_V3 + ":" + item["name"])
                if item["transform"] == "transpose_2d":
                    self.assertEqual(len(item["shape"]), 2)
                    self.assertEqual(math.prod(item["shape"]), math.prod(reversed(item["shape"])))
                else:
                    self.assertEqual(item["transform"], "identity")

    def test_legacy_or_resized_config_cannot_become_registered_pc_bank(self):
        for config in (ModelConfig.baseline(), ModelConfig.for_profile("full_line_interaction_v2"),
                       ModelConfig(profile="virtual_review_v1", max_records=128)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                frozen.registered_pc_parameter_spec(config)

    def test_returned_specs_do_not_mutate_future_registration(self):
        first = frozen.registered_pc_parameter_spec()
        first[0]["shape"].append(1)
        first[0]["graphs"].clear()
        self.assertNotEqual(first, frozen.registered_pc_parameter_spec())


class RequestTests(PureTests):
    def test_request_parser_is_schema_only_and_defensive(self):
        request = request_fixture()
        result = frozen.validate_request(request)
        self.assertEqual(result, request)
        result["graphs"][0]["sha256"] = "a" * 64
        self.assertNotEqual(result, request)
        self.assertEqual(frozen.request_digest(request), data.digest(frozen.REQUEST_DOMAIN, request))

    def test_strict_request_domain_and_field_set(self):
        for mutate in (lambda p: p.update(schema="rz-pals-frozen-pc-content-request/0"),
                       lambda p: p.update(extra=True), lambda p: p.pop("source_sha256")):
            request = request_fixture()
            mutate(request)
            with self.assertRaises(ValueError):
                frozen.validate_request(request)

    def test_source_epoch_and_expected_bank_hash_types(self):
        cases = (("source_sha256", "A" * 64), ("source_sha256", "f" * 63),
                 ("source_sha256", None), ("model_epoch", True), ("model_epoch", -1),
                 ("model_epoch", 1.0), ("expected_frozen_pc_sha256", "Z" * 64))
        for key, value in cases:
            request = request_fixture()
            request[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                frozen.validate_request(request)

    def test_optional_expected_parameter_pin_is_explicit_null(self):
        request = request_fixture()
        request["expected_frozen_pc_sha256"] = None
        self.assertIsNone(frozen.validate_request(request)["expected_frozen_pc_sha256"])

    def test_registered_config_rejects_extra_missing_boolean_and_wrong_profile(self):
        for mutation in (lambda c: c.update(width=True), lambda c: c.update(extra=1),
                         lambda c: c.pop("max_records"), lambda c: c.update(profile="legacy_summary_v1")):
            request = request_fixture()
            mutation(request["config"])
            with self.assertRaises((ValueError, TypeError)):
                frozen.validate_request(request)

    def test_each_limit_is_positive_integer_and_bounded(self):
        for name, maximum in frozen.LIMITS.items():
            for value in (0, -1, True, 1.5, maximum + 1):
                request = request_fixture()
                request["limits"][name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    frozen.validate_request(request)

    def test_limit_fields_are_exact(self):
        for mutation in (lambda p: p.pop("wall_seconds"), lambda p: p.update(other=1)):
            request = request_fixture()
            mutation(request["limits"])
            with self.assertRaises(ValueError):
                frozen.validate_request(request)

    def test_artifact_pin_path_rejects_escape_hidden_absolute_and_noncanonical(self):
        for file in ("../checkpoint.pt", "/checkpoint.pt", "C:/checkpoint.pt", "g0\\checkpoint.pt",
                     ".cache/checkpoint.pt", "g0/.private/checkpoint.pt", "g0//checkpoint.pt",
                     "g0/./checkpoint.pt", "line\ncheckpoint.pt", "nul\x00checkpoint.pt",
                     "tab\tcheckpoint.pt", "delete\x7fcheckpoint.pt", "", "a" * 257 + ".pt"):
            request = request_fixture()
            request["checkpoint"]["file"] = file
            with self.subTest(file=file), self.assertRaises(ValueError):
                frozen.validate_request(request)

    def test_checkpoint_pin_requires_explicit_domain_and_pt(self):
        for domain, file in (("diagnostic-checkpoint/1", "g0/checkpoint.pt"),
                             (MODEL_SEMANTICS_V3, "g0/checkpoint.json")):
            request = request_fixture()
            request["checkpoint"].update(domain=domain, file=file)
            with self.assertRaises(ValueError):
                frozen.validate_request(request)
        request = request_fixture()
        request["checkpoint"]["domain"] = frozen.CHECKPOINT_DOMAIN
        self.assertEqual(frozen.validate_request(request), request)

    def test_export_manifest_pin_is_required_and_independent(self):
        for mutation in (lambda r: r.pop("export_manifest"),
                         lambda r: r["export_manifest"].update(sha256="BAD"),
                         lambda r: r["export_manifest"].update(bytes=0),
                         lambda r: r["export_manifest"].update(bytes=True),
                         lambda r: r["export_manifest"].update(file="g0/.private.json"),
                         lambda r: r["export_manifest"].update(file="g0/not-export.json"),
                         lambda r: r["export_manifest"].update(extra=True)):
            request = request_fixture()
            mutation(request)
            with self.assertRaises(ValueError):
                frozen.validate_request(request)

    def test_empty_oversize_or_boolean_artifact_pin_rejected(self):
        for path in (("checkpoint",), ("graphs", 0), ("verifier_sources", 0)):
            for value in (0, True, 1.0):
                request = request_fixture()
                target = request[path[0]] if len(path) == 1 else request[path[0]][path[1]]
                target["bytes"] = value
                with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                    frozen.validate_request(request)
        request = request_fixture()
        request["checkpoint"]["bytes"] = request["limits"]["checkpoint_bytes"] + 1
        with self.assertRaises(ValueError):
            frozen.validate_request(request)

    def test_graph_pair_role_order_filename_and_count_are_fixed(self):
        for mutate in (lambda r: r["graphs"].reverse(), lambda r: r["graphs"].pop(),
                       lambda r: r["graphs"][0].update(role="validator"),
                       lambda r: r["graphs"][1].update(file="g0/shared_pc_warm.onnx")):
            request = request_fixture()
            mutate(request)
            with self.assertRaises(ValueError):
                frozen.validate_request(request)

    def test_verifier_source_list_is_complete_ordered_unique_and_pinned(self):
        for mutate in (lambda r: r["verifier_sources"].reverse(),
                       lambda r: r["verifier_sources"].pop(),
                       lambda r: r["verifier_sources"].__setitem__(1, copy.deepcopy(r["verifier_sources"][0])),
                       lambda r: r["verifier_sources"][0].update(sha256="BAD"),
                       lambda r: r["verifier_sources"][0].update(bytes=2 * 1024**2 + 1)):
            request = request_fixture()
            mutate(request)
            with self.assertRaises(ValueError):
                frozen.validate_request(request)


class MetadataProofTests(PureTests):
    def test_complete_metadata_fixture_parses_but_not_native_or_arena_acceptance(self):
        proof = proof_fixture()
        result = self.validate_proof(proof, expected_parameter_sha256=proof["frozen_pc_parameter_sha256"])
        self.assertEqual(result, proof)
        self.assertFalse(result["verification"]["optimizer_content_verified"])
        self.assertFalse(result["verification"]["numerical_acceptance"])
        self.assertFalse(result["verification"]["native_loaded_assets_verified"])
        self.assertFalse(result["verification"]["arena_eligible"])
        result["parameter_inventory"][0]["shape"].append(99)
        self.assertNotEqual(result, proof)

    def test_proof_domain_scope_field_set_and_request_pin_are_strict(self):
        for key, value in (("schema", "rz-pals-frozen-pc-content-proof/0"), ("scope", "trained"),
                           ("request_sha256", metadata_sha("different-request")), ("extra", True)):
            proof = proof_fixture()
            proof[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate_proof(proof)
        proof = proof_fixture()
        proof.pop("mapping")
        with self.assertRaises(ValueError):
            self.validate_proof(proof)

    def test_original_registered_source_and_graph_pins_cannot_drift(self):
        for mutate in (lambda p: p["request"].update(source_sha256=metadata_sha("another-source")),
                       lambda p: p["request"]["verifier_sources"][0].update(sha256=metadata_sha("edited-source")),
                       lambda p: p["request"]["export_manifest"].update(sha256=metadata_sha("edited-manifest")),
                       lambda p: p["request"]["graphs"][0].update(sha256=metadata_sha("edited-graph"))):
            proof = proof_fixture()
            mutate(proof)
            with self.assertRaises(ValueError):
                self.validate_proof(proof)

    def test_checkpoint_scope_is_not_interchangeable(self):
        for domain, correct, wrong in ((MODEL_SEMANTICS_V3, "own_initialization_tensor_content_only", "actual_pc_frozen_tensor_content_only"),
                                      (frozen.CHECKPOINT_DOMAIN, "actual_pc_frozen_tensor_content_only", "own_initialization_tensor_content_only")):
            request = request_fixture()
            request["checkpoint"]["domain"] = domain
            proof = proof_fixture(request)
            self.assertEqual(self.validate_proof(proof)["checkpoint_scope"], correct)
            proof["checkpoint_scope"] = wrong
            with self.assertRaises(ValueError):
                self.validate_proof(proof)

    def test_verification_flags_have_exact_bool_types_and_cannot_promote(self):
        for name, expected in frozen.VERIFICATION.items():
            for invalid in (not expected, int(expected)):
                proof = proof_fixture()
                proof["verification"][name] = invalid
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    self.validate_proof(proof)
        proof = proof_fixture()
        proof["verification"]["extra"] = False
        with self.assertRaises(ValueError):
            self.validate_proof(proof)

    def test_named_parameter_pin_joins_request_and_external_pin(self):
        proof = proof_fixture()
        for invalid in (metadata_sha("other-bank"), None, "A" * 64):
            value = copy.deepcopy(proof)
            value["frozen_pc_parameter_sha256"] = invalid
            with self.assertRaises(ValueError):
                self.validate_proof(value)
        with self.assertRaises(ValueError):
            self.validate_proof(proof, expected_parameter_sha256=metadata_sha("external-other-bank"))

    def test_exact_encoding_registry_is_required(self):
        proof = proof_fixture()
        proof["encoding_registry_sha256"] = metadata_sha("other-abi")
        with self.assertRaises(ValueError):
            self.validate_proof(proof)

    def test_parameter_inventory_missing_extra_duplicate_or_reordered_rejected(self):
        for mutate in (lambda p: p["parameter_inventory"].pop(),
                       lambda p: p["parameter_inventory"].append(copy.deepcopy(p["parameter_inventory"][0])),
                       lambda p: p["parameter_inventory"].reverse(),
                       lambda p: p["parameter_inventory"][0].update(name="experts.validator.initial_latent")):
            proof = proof_fixture()
            mutate(proof)
            reseal_inventory(proof)
            with self.assertRaises(ValueError):
                self.validate_proof(proof)

    def test_parameter_shape_dtype_owner_meaning_and_bytes_are_exact(self):
        for key, invalid in (("shape", [1]), ("dtype", "fp16_le"), ("owner", "validator"),
                             ("semantics", "same-looking-name"), ("bytes", 1),
                             ("bytes", True), ("sha256", "BAD")):
            proof = proof_fixture()
            proof["parameter_inventory"][0][key] = invalid
            reseal_inventory(proof)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate_proof(proof)

    def test_parameter_shape_boolean_and_tuple_do_not_match_integer_list(self):
        for invalid in ([True, 384], (16, 384)):
            proof = proof_fixture()
            proof["parameter_inventory"][0]["shape"] = invalid
            reseal_inventory(proof)
            with self.assertRaises(ValueError):
                self.validate_proof(proof)

    def test_inventory_digest_and_aggregate_tensor_budget(self):
        proof = proof_fixture()
        proof["parameter_inventory_sha256"] = metadata_sha("different-inventory")
        with self.assertRaises(ValueError):
            self.validate_proof(proof)
        proof = proof_fixture()
        proof["request"]["limits"]["tensor_bytes"] = 52948991
        reseal_request(proof)
        with self.assertRaises(ValueError):
            self.validate_proof(proof)

    def test_graph_file_pin_order_role_and_size_join_exactly(self):
        for mutate in (lambda p: p["graphs"].reverse(), lambda p: p["graphs"].pop(),
                       lambda p: p["graphs"][0].update(sha256=metadata_sha("different-file")),
                       lambda p: p["graphs"][0].update(bytes=64 * 1024**2 + 1),
                       lambda p: p["graphs"][1].update(role="validator"),
                       lambda p: p["graphs"][0].update(file="other/public_memory.onnx"),
                       lambda p: p["graphs"][0].update(structure_sha256="BAD")):
            proof = proof_fixture()
            mutate(proof)
            with self.assertRaises(ValueError):
                self.validate_proof(proof)

    def test_initializer_inventory_duplicates_order_digest_or_empty_rejected(self):
        for mutate in (lambda g: g["initializers"].reverse(),
                       lambda g: g["initializers"].append(copy.deepcopy(g["initializers"][0])),
                       lambda g: g["initializers"].clear()):
            proof = proof_fixture()
            mutate(proof["graphs"][0])
            reseal_graph(proof["graphs"][0])
            with self.assertRaises(ValueError):
                self.validate_proof(proof)
        proof = proof_fixture()
        proof["graphs"][0]["initializer_inventory_sha256"] = metadata_sha("other-initializers")
        with self.assertRaises(ValueError):
            self.validate_proof(proof)

    def test_initializer_dtype_shape_byte_size_and_sha_rejected(self):
        for key, invalid in (("dtype", "FLOAT16"), ("shape", [True]), ("shape", [0]),
                             ("shape", [1048577]), ("shape", [1] * 9), ("shape", (1,)),
                             ("bytes", 1), ("bytes", True), ("sha256", "BAD")):
            proof = proof_fixture()
            proof["graphs"][0]["initializers"][0][key] = invalid
            reseal_graph(proof["graphs"][0])
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate_proof(proof)

    def test_initializer_aggregate_bytes_cannot_exceed_pinned_graph_bytes(self):
        proof = proof_fixture()
        graph = proof["graphs"][0]
        graph["initializers"].append({"name": "z_metadata_only_constant", "dtype": "FLOAT",
            "shape": [1048576, 16], "bytes": 64 * 1024**2,
            "sha256": metadata_sha("aggregate-over-pinned-size")})
        graph["initializers"].sort(key=lambda item: item["name"])
        reseal_graph(graph)
        self.assertGreater(sum(item["bytes"] for item in graph["initializers"]), graph["bytes"])
        with self.assertRaises(ValueError):
            self.validate_proof(proof)

    def test_mapping_missing_extra_duplicate_reordered_and_wrong_graph_rejected(self):
        for mutate in (lambda p: p["mapping"].pop(),
                       lambda p: p["mapping"].append(copy.deepcopy(p["mapping"][0])),
                       lambda p: p["mapping"].reverse(),
                       lambda p: p["mapping"][0].update(graph_role="validator")):
            proof = proof_fixture()
            mutate(proof)
            reseal_mapping(proof)
            with self.assertRaises(ValueError):
                self.validate_proof(proof)

    def test_mapping_transform_copied_name_and_source_sha_join_exactly(self):
        for key, invalid in (("transform", "transpose_2d_twice"),
                             ("initializer", "looks_similar.weight"),
                             ("source_sha256", metadata_sha("other-original-bytes")),
                             ("initializer_sha256", metadata_sha("other-copied-bytes"))):
            proof = proof_fixture()
            proof["mapping"][0][key] = invalid
            reseal_mapping(proof)
            with self.assertRaises(ValueError):
                self.validate_proof(proof)

    def test_mapping_digest_is_required_even_when_entries_match(self):
        proof = proof_fixture()
        proof["mapping_sha256"] = metadata_sha("other-mapping")
        with self.assertRaises(ValueError):
            self.validate_proof(proof)

    def test_transpose_initializer_has_reversed_shape_and_distinct_namespace(self):
        proof = proof_fixture()
        by_name = {item["name"]: item for item in proof["parameter_inventory"]}
        entry = next(item for item in proof["mapping"] if item["transform"] == "transpose_2d"
                     and by_name[item["parameter"]]["shape"][0] != by_name[item["parameter"]]["shape"][1])
        target = next(item for graph in proof["graphs"] if graph["role"] == entry["graph_role"]
                      for item in graph["initializers"] if item["name"] == entry["initializer"])
        source = next(item for item in proof["parameter_inventory"] if item["name"] == entry["parameter"])
        self.assertEqual(target["shape"], list(reversed(source["shape"])))
        self.assertTrue(entry["initializer"].endswith(".linear_transpose"))
        target["shape"] = source["shape"].copy()
        reseal_graph(next(graph for graph in proof["graphs"] if graph["role"] == entry["graph_role"]))
        with self.assertRaises(ValueError):
            self.validate_proof(proof)

    def test_identity_initializer_cannot_change_bits_even_with_resealed_metadata(self):
        proof = proof_fixture()
        entry = next(item for item in proof["mapping"] if item["transform"] == "identity")
        graph = next(graph for graph in proof["graphs"] if graph["role"] == entry["graph_role"])
        target = next(item for item in graph["initializers"] if item["name"] == entry["initializer"])
        target["sha256"] = entry["initializer_sha256"] = metadata_sha("changed-identity")
        reseal_graph(graph)
        reseal_mapping(proof)
        with self.assertRaises(ValueError):
            self.validate_proof(proof)

    def test_public_and_shared_copies_of_line_bank_must_both_join_original(self):
        proof = proof_fixture()
        name = "public_encoder.line_encoder.moves.from_square.weight"
        entries = [entry for entry in proof["mapping"] if entry["parameter"] == name]
        self.assertEqual([entry["graph_role"] for entry in entries], ["public", "shared_pc"])
        self.assertEqual(entries[0]["source_sha256"], entries[1]["source_sha256"])
        entry = entries[1]
        graph = next(graph for graph in proof["graphs"] if graph["role"] == "shared_pc")
        target = next(item for item in graph["initializers"] if item["name"] == entry["initializer"])
        target["sha256"] = entry["initializer_sha256"] = metadata_sha("only-second-copy-changed")
        reseal_graph(graph)
        reseal_mapping(proof)
        with self.assertRaises(ValueError):
            self.validate_proof(proof)


class ExecutionBoundaryTests(PureTests):
    def test_every_non_true_execution_gate_rejects_before_request_or_filesystem(self):
        with mock.patch.object(frozen, "validate_request", side_effect=AssertionError("request touched")) as validation, \
             mock.patch.object(frozen, "_hash_file", side_effect=AssertionError("file touched")) as hashing, \
             mock.patch.object(frozen, "Path", side_effect=AssertionError("path touched")) as path:
            for execute in (False, None, 0, 1, "True"):
                with self.subTest(execute=execute), self.assertRaises(ValueError):
                    frozen.verify_frozen_pc_content(None, artifact_root="unused", output_directory="unused", execute=execute)
            validation.assert_not_called()
            hashing.assert_not_called()
            path.assert_not_called()

    def test_source_pin_mismatch_rejects_before_artifact_or_neural_access(self):
        request = request_fixture()
        with mock.patch.object(frozen, "_owned_file", return_value=Path("fixture-source.py")) as owned, \
             mock.patch.object(frozen, "_hash_file", return_value=(metadata_sha("changed-source"), 1024)) as hashing, \
             mock.patch.object(frozen.tempfile, "TemporaryFile", side_effect=AssertionError("checkpoint touched")) as snapshot, \
             mock.patch.object(frozen, "_atomic_json", side_effect=AssertionError("proof published")) as publication:
            with self.assertRaisesRegex(ValueError, "source changed"):
                frozen.verify_frozen_pc_content(request, artifact_root="unused", output_directory="unused", execute=True)
            self.assertEqual(owned.call_args.args[1], frozen.SOURCE_NAMES[0])
            hashing.assert_called_once()
            snapshot.assert_not_called()
            publication.assert_not_called()

    def test_last_verifier_source_is_checked_before_any_checkpoint_decode(self):
        request = request_fixture()
        pins = {pin["file"]: pin for pin in request["verifier_sources"]}

        def source_hash(path, maximum, **kwargs):
            pin = pins[path.name]
            return (metadata_sha("last-source-changed"), pin["bytes"]) if path.name == frozen.SOURCE_NAMES[-1] else (pin["sha256"], pin["bytes"])

        with mock.patch.object(frozen, "_owned_file", side_effect=lambda root, relative: Path(relative)) as owned, \
             mock.patch.object(frozen, "_hash_file", side_effect=source_hash) as hashing, \
             mock.patch.object(frozen, "_verify_active_source") as active_source, \
             mock.patch.object(frozen.tempfile, "TemporaryFile", side_effect=AssertionError("checkpoint touched")) as snapshot:
            with self.assertRaisesRegex(ValueError, "source changed"):
                frozen.verify_frozen_pc_content(request, artifact_root="unused", output_directory="unused", execute=True)
            self.assertEqual([call.args[1] for call in owned.call_args_list], list(frozen.SOURCE_NAMES))
            self.assertEqual(hashing.call_count, len(frozen.SOURCE_NAMES))
            self.assertEqual(active_source.call_count, len(frozen.SOURCE_NAMES) - 1)
            snapshot.assert_not_called()

    def test_validated_request_does_not_read_files_or_import_frameworks(self):
        with mock.patch.object(frozen, "_hash_file", side_effect=AssertionError("metadata read files")), \
             mock.patch.object(frozen, "_owned_file", side_effect=AssertionError("metadata read assets")):
            self.assertEqual(frozen.validate_request(request_fixture()), request_fixture())
            self.assertEqual(self.validate_proof(proof_fixture())["schema"], frozen.PROOF_DOMAIN)

    def test_external_json_pin_is_raw_bytes_and_does_not_issue_content_proof(self):
        proof = proof_fixture()
        raw = json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with tempfile.TemporaryDirectory(prefix="rz-frozen-contract-only-") as directory:
            path = Path(directory) / frozen.FILENAME
            path.write_bytes(raw)
            before = path.read_bytes()
            accepted = frozen.read_frozen_pc_proof(path, expected_file_sha256=hashlib.sha256(raw).hexdigest(),
                expected_request_sha256=proof["request_sha256"],
                expected_parameter_sha256=proof["frozen_pc_parameter_sha256"])
            self.assertEqual(accepted, proof)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(accepted["verification"]["native_loaded_assets_verified"])
            with self.assertRaises(ValueError):
                frozen.read_frozen_pc_proof(path, expected_file_sha256=metadata_sha("different-raw-json"),
                    expected_request_sha256=proof["request_sha256"])

    def test_external_reader_requires_registered_filename_before_file_read(self):
        with mock.patch.object(frozen, "_pinned_json", side_effect=AssertionError("file read")) as read:
            with self.assertRaises(ValueError):
                frozen.read_frozen_pc_proof("not-the-proof.json", expected_file_sha256="a" * 64,
                    expected_request_sha256="b" * 64)
            read.assert_not_called()

    def test_external_reader_enforces_registered_proof_byte_cap(self):
        request = request_fixture()
        request["limits"]["proof_bytes"] = 1024
        proof = proof_fixture(request)
        raw = json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertGreater(len(raw), request["limits"]["proof_bytes"])
        self.assertLess(len(raw), frozen.LIMITS["proof_bytes"])
        with tempfile.TemporaryDirectory(prefix="rz-frozen-cap-only-") as directory:
            path = Path(directory) / frozen.FILENAME
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                frozen.read_frozen_pc_proof(path, expected_file_sha256=hashlib.sha256(raw).hexdigest(),
                    expected_request_sha256=proof["request_sha256"])


class CheckpointZipPreflightTests(PureTests):
    def test_small_stored_string_zip_is_rewound_without_decoding(self):
        stream = stored_zip([("fixture/data", b"schema-only"), ("fixture/version", b"1")])
        stream.seek(7)
        frozen._zip_preflight(stream, 12)
        self.assertEqual(stream.tell(), 0)

    def test_malformed_empty_and_duplicate_archive_rejected(self):
        with self.assertRaises(zipfile.BadZipFile):
            frozen._zip_preflight(io.BytesIO(b"not-a-checkpoint-or-zip"), 32)
        for entries in ([], [("fixture/data", b"1"), ("fixture/data", b"2")]):
            with self.assertRaises(ValueError):
                frozen._zip_preflight(stored_zip(entries), 32)

    def test_decoded_storage_aggregate_cap_is_enforced(self):
        stream = stored_zip([("fixture/a", b"12345"), ("fixture/b", b"67890")])
        with self.assertRaisesRegex(ValueError, "explicit cap"):
            frozen._zip_preflight(stream, 9)

    def test_compressed_archive_is_rejected_before_decoder(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("fixture/data", "schema-only")
        stream.seek(0)
        with self.assertRaisesRegex(ValueError, "compressed"):
            frozen._zip_preflight(stream, 1024)

    def test_absolute_and_parent_escape_storage_names_are_rejected(self):
        for name in ("/data", "../data", "fixture/../data"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "escaping"):
                frozen._zip_preflight(stored_zip([(name, b"1")]), 32)

    def test_noncanonical_relative_and_drive_zip_names_are_rejected(self):
        for name in ("root/./data", "root//data", "C:/outside", "root/line\nname"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                frozen._zip_preflight(stored_zip([(name, b"1")]), 32)

    def test_stored_zip_cannot_understate_decoded_storage_size(self):
        stream = stored_zip([("fixture/data", b"schema-only")])
        raw = bytearray(stream.getvalue())
        start = raw.index(b"PK\x01\x02")
        struct.pack_into("<I", raw, start + 24, 0)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entry = archive.infolist()[0]
            self.assertEqual(entry.file_size, 0)
            self.assertGreater(entry.compress_size, 0)
        with self.assertRaises(ValueError):
            frozen._zip_preflight(io.BytesIO(raw), 1)

    def test_encryption_bit_is_rejected_without_reading_payload(self):
        stream = stored_zip([("fixture/data", b"schema-only")])
        raw = bytearray(stream.getvalue())
        for signature, flags_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
            start = raw.index(signature)
            flags = struct.unpack_from("<H", raw, start + flags_offset)[0]
            struct.pack_into("<H", raw, start + flags_offset, flags | 1)
        with self.assertRaisesRegex(ValueError, "encrypted"):
            frozen._zip_preflight(io.BytesIO(raw), 1024)

    def test_entry_count_cap_is_independent_of_small_payload(self):
        stream = stored_zip([(f"fixture/{i}", b"") for i in range(8193)])
        with self.assertRaisesRegex(ValueError, "bounded unique"):
            frozen._zip_preflight(stream, 1)


class ActiveSourceBindingTests(PureTests):
    SOURCE = b'''CAPS = {"bytes": 1024 ** 2}
SCALAR = 3
def check(value=3, *, strict=True):
    return value
class Slot:
    width: int = 384
    @staticmethod
    def first(value=2):
        return value
    @classmethod
    def second(cls, value=4):
        return value
    @property
    def ready(self):
        return True
'''

    def source_module(self, source=None, *, filename_alias=False):
        """Compile only owned trivial definitions; no models or library code."""
        source = self.SOURCE if source is None else source
        directory = tempfile.TemporaryDirectory(prefix="rz-frozen-source-only-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "pure_source_fixture.py"
        name = "_rz_frozen_pc_pure_source_fixture"
        module = ModuleType(name)
        module.__file__ = str(path)
        filename = str(path.parent) + "/./" + path.name if filename_alias else str(path)
        exec(compile(source, filename, "exec", dont_inherit=True, optimize=sys.flags.optimize), module.__dict__)
        self.assertNotIn(name, sys.modules)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name)
        return name, module, path, source

    def test_matching_loaded_code_literal_defaults_and_declared_values(self):
        name, module, path, source = self.source_module()
        self.assertEqual(module.SCALAR, 3)
        frozen._verify_active_source(name, source, path)

    def test_unloaded_module_does_not_import_or_execute_source(self):
        frozen._verify_active_source("_rz_absent_contract_source", b"import torch\nraise AssertionError('not executed')\n", Path("unused.py"))

    def test_new_unloaded_import_rejected_without_executing_source(self):
        name, module, path, source = self.source_module()
        with self.assertRaisesRegex(ValueError, "imported module alias"):
            frozen._verify_active_source(name, source + b"import torch\nraise AssertionError('not executed')\n", path)

    def test_active_module_path_must_be_registered_source_root(self):
        name, module, path, source = self.source_module()
        module.__file__ = str(path.parent / "other_source.py")
        with self.assertRaisesRegex(ValueError, "another source root"):
            frozen._verify_active_source(name, source, path)

    def test_equivalent_loaded_filename_spelling_is_not_a_different_source(self):
        name, module, path, source = self.source_module(filename_alias=True)
        self.assertNotEqual(module.check.__code__.co_filename, str(path))
        self.assertEqual(Path(module.check.__code__.co_filename).resolve(), path.resolve())
        frozen._verify_active_source(name, source, path)

    def test_loaded_code_filename_from_other_root_is_rejected(self):
        name, module, path, source = self.source_module()
        module.check.__code__ = module.check.__code__.replace(co_filename=str(path.parent / "other_source.py"))
        with self.assertRaisesRegex(ValueError, "code differs"):
            frozen._verify_active_source(name, source, path)

    def test_loaded_function_cannot_lie_about_different_disk_code(self):
        name, module, path, source = self.source_module()
        with self.assertRaisesRegex(ValueError, "code differs"):
            frozen._verify_active_source(name, source.replace(b"return value", b"return value + 1", 1), path)

    def test_default_value_mutation_rejected_even_when_code_is_same(self):
        name, module, path, source = self.source_module()
        module.check.__defaults__ = (4,)
        with self.assertRaisesRegex(ValueError, "default differs"):
            frozen._verify_active_source(name, source, path)

    def test_keyword_default_bool_cannot_be_relabelled_integer(self):
        name, module, path, source = self.source_module()
        module.check.__kwdefaults__["strict"] = 1
        with self.assertRaisesRegex(ValueError, "keyword default differs"):
            frozen._verify_active_source(name, source, path)

    def test_global_arithmetic_literal_and_nested_type_are_pinned(self):
        name, module, path, source = self.source_module()
        module.CAPS["bytes"] = float(1024**2)
        with self.assertRaisesRegex(ValueError, "declaration differs"):
            frozen._verify_active_source(name, source, path)

    def test_executing_stale_alias_rejected_after_module_attribute_replacement(self):
        name, module, path, source = self.source_module()
        old_code = module.check.__code__
        replacement = source.replace(b"return value", b"return value + 1", 1)
        exec(compile(replacement, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize), module.__dict__)
        # Current module attrs alone now agree with replacement. The actual
        # retained entry frame must still prove which code was executing.
        frozen._verify_active_source(name, replacement, path)
        with self.assertRaisesRegex(ValueError, "executing verifier alias"):
            frozen._verify_active_source(name, replacement, path,
                                        executing_code=old_code, executing_name="check")
        frozen._verify_active_source(name, replacement, path,
                                    executing_code=module.check.__code__, executing_name="check")

    def test_named_default_and_imported_module_attribute_are_not_skipped(self):
        source = b"import time\nLIMIT = 1024 ** 2\ndef check(limit=LIMIT, *, clock=time.monotonic):\n    return limit\n"
        name, module, path, source = self.source_module(source)
        frozen._verify_active_source(name, source, path)
        module.check.__defaults__ = (1024,)
        with self.assertRaisesRegex(ValueError, "default differs"):
            frozen._verify_active_source(name, source, path)
        module.check.__defaults__ = (1024 ** 2,)
        module.check.__kwdefaults__["clock"] = module.time.time
        with self.assertRaisesRegex(ValueError, "keyword default differs"):
            frozen._verify_active_source(name, source, path)

    def test_imported_module_alias_cannot_point_at_another_module(self):
        source = b"import time\ndef check(clock=time.monotonic):\n    return clock\n"
        name, module, path, source = self.source_module(source)
        module.time = ModuleType("time")
        with self.assertRaisesRegex(ValueError, "imported module alias"):
            frozen._verify_active_source(name, source, path)

    def test_unregistered_constructor_default_fails_closed_without_execution(self):
        name, module, path, source = self.source_module(b"def check(value=list()):\n    return value\n")
        with self.assertRaisesRegex(ValueError, "unsupported source-owned"):
            frozen._verify_active_source(name, source, path)

    def test_config_default_compares_dataclass_values_without_calling_constructor(self):
        source = b"from rz_pals_model.config import ModelConfig\ndef check(config=ModelConfig()):\n    return config\n"
        name, module, path, source = self.source_module(source)
        with mock.patch.object(module.ModelConfig, "__init__", side_effect=AssertionError("constructor called")):
            frozen._verify_active_source(name, source, path)
        object.__setattr__(module.check.__defaults__[0], "width", 385)
        with self.assertRaisesRegex(ValueError, "config default differs"):
            frozen._verify_active_source(name, source, path)

    def test_generated_dataclass_init_defaults_must_match_source_descriptor(self):
        source = b"from dataclasses import dataclass, field\n@dataclass\nclass Slot:\n    width: int = 384\n    items: list = field(default_factory=lambda: [])\n"
        name, module, path, source = self.source_module(source)
        frozen._verify_active_source(name, source, path)
        original = module.Slot.__init__.__defaults__
        module.Slot.__init__.__defaults__ = (768, original[1])
        self.assertEqual(module.Slot.width, 384)
        self.assertEqual(module.Slot.__dataclass_fields__["width"].default, 384)
        with self.assertRaisesRegex(ValueError, "init defaults differ"):
            frozen._verify_active_source(name, source, path)

    def test_generated_dataclass_init_factory_closure_must_match_descriptor(self):
        source = b"from dataclasses import dataclass, field\n@dataclass\nclass Slot:\n    width: int = 384\n    items: list = field(default_factory=lambda: [])\n"
        name, module, path, source = self.source_module(source)
        frozen._verify_active_source(name, source, path)
        init = module.Slot.__init__
        cells = dict(zip(init.__code__.co_freevars, init.__closure__))
        factory_cell = next(cells[name] for name in ("_dflt_items", "__dataclass_dflt_items__") if name in cells)
        factory_cell.cell_contents = lambda: [1]
        with self.assertRaisesRegex(ValueError, "factory connection differs"):
            frozen._verify_active_source(name, source, path)

    def test_current_loaded_pure_module_source_matches_without_neural_import(self):
        root = Path(frozen.__file__).parent
        checked = []
        for filename in frozen.SOURCE_NAMES:
            module_name = frozen.__package__ + "." + Path(filename).stem
            if module_name in sys.modules and filename not in ("model.py", "nonzero_training.py", "training.py"):
                frozen._verify_active_source(module_name, (root / filename).read_bytes(), root / filename)
                checked.append(filename)
        self.assertIn("generation_frozen_pc.py", checked)
        self.assertIn("config.py", checked)

    def test_declared_class_default_and_static_method_defaults_are_pinned(self):
        name, module, path, source = self.source_module()
        module.Slot.width = 384.0
        with self.assertRaisesRegex(ValueError, "dataclass default differs"):
            frozen._verify_active_source(name, source, path)
        module.Slot.width = 384
        module.Slot.__dict__["first"].__func__.__defaults__ = (3,)
        with self.assertRaisesRegex(ValueError, "default differs"):
            frozen._verify_active_source(name, source, path)


if __name__ == "__main__":
    unittest.main()
