"""Pure adversarial initialization contracts. Never execute Torch or a model."""
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("model/NN/optimizer imports forbidden in initialization contract tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import initialization_evidence as init
    from rz_pals_model import artifacts
    from rz_pals_model.config import ModelConfig, VIRTUAL_REVIEW_PROFILE
finally:
    sys.meta_path.remove(_guard)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(name="weight", bits="a" * 64, semantics=None):
    row = {"name": name, "shape": [2], "dtype": "fp32_le", "bytes": 8, "sha256": bits}
    if semantics is not None:
        row["semantics"] = semantics
    return row


def evidence():
    """Schema-only adversarial fixture, not a Torch checkpoint or live capture."""
    metadata = {"schema": init.MODEL_SEMANTICS_V3, "checkpoint": "untrained.pt", "checkpoint_sha256": "a" * 64,
        "config": ModelConfig.for_profile(VIRTUAL_REVIEW_PROFILE).to_dict(), "seed": 23,
        "trained": False, "training_steps": 0, "external_weights": False, "external_teacher": False,
        "precision": "fp32", "tf32": False, "torch_version": "2.8.0+cpu", "parameter_count": 2,
        "roles": list(init.ROLES), "status": "random_initialization_only"}
    return {"schema": init.EVIDENCE_SCHEMA, "generation_id": "g0", "initialization_id": "contract-fixture-only",
        "model": {"schema": init.MODEL_SCHEMA_V3, "semantics": init.MODEL_SEMANTICS_V3,
            "encoding": init.ENCODING_V3, "profile": VIRTUAL_REVIEW_PROFILE, "config": copy.deepcopy(metadata["config"]),
            "roles": list(init.ROLES), "seed": 23},
        "source": {"schema": init.SOURCE_SCHEMA, "source_commit": "1" * 40, "pins": {name: "2" * 64 for name in init.SOURCE_FILES}},
        "checkpoint": {"name": "untrained.pt", "bytes": 8, "sha256": "a" * 64},
        "metadata": {"name": "checkpoint.json", "bytes": 1024, "sha256": "b" * 64},
        "metadata_fields": metadata, "checkpoint_inventory": [inventory()],
        "initialization": {"mode": "fresh_random", "producer": init.PRODUCER,
            "invocation_status": "captured_registered_initializer_return",
            "inventory_producer": "rz_pals_model.controlled_initialization._parameter_inventory",
            "allowlist": {}, "copied": [], "ignored_source": [], "ignored_target": ["weight"],
            "partial_initialization": "requires_separate_actual_producer_registration"},
        "software": {"pytorch_version": "2.8.0+cpu", "python_version": "3.12.3", "python_implementation": "CPython",
            "torch_git_version": None, "torch_cuda_build": None, "torch_thread_settings": {"intra": 2, "inter": 1},
            "deterministic_algorithms": False, "tf32_matmul": False, "tf32_cudnn": True,
            "observation": "already_loaded_torch_after_registered_initializer", "device_capability": "not_probed"},
        "claims": init._claims(), "acceptance": init._acceptance()}


def legacy_evidence():
    """A separate V2 observation fixture; cannot be admitted as G0/V3."""
    result = evidence()
    result["schema"] = init.LEGACY_EVIDENCE_SCHEMA
    result["generation_id"] = "legacy_baseline"
    result["source"]["schema"] = init.LEGACY_SOURCE_SCHEMA
    result["model"] = init._model_registration(legacy=True)
    result["metadata_fields"]["schema"] = result["model"]["semantics"]
    result["metadata_fields"]["config"] = copy.deepcopy(result["model"]["config"])
    result["acceptance"] = init._acceptance(legacy=True)
    return result


def owned(directory):
    (directory / "pals-artifact-owner.json").write_text(json.dumps({"schema": init.OWNER_SCHEMA, "owner": "rz-pals-model"}), encoding="utf-8")


def store_observation_fixture(directory, *, legacy=False):
    """Small fake bytes test joins only; no tensor/producer authority is issued."""
    owned(directory)
    value = legacy_evidence() if legacy else evidence()
    raw = b"schema-only bytes, deliberately not a Torch checkpoint"
    (directory / "untrained.pt").write_bytes(raw)
    value["checkpoint"] = {"name": "untrained.pt", "bytes": len(raw), "sha256": digest(raw)}
    value["metadata_fields"]["checkpoint_sha256"] = digest(raw)
    metadata = json.dumps(value["metadata_fields"], sort_keys=True).encode("utf-8")
    (directory / "checkpoint.json").write_bytes(metadata)
    value["metadata"] = {"name": "checkpoint.json", "bytes": len(metadata), "sha256": digest(metadata)}
    name = init.LEGACY_EVIDENCE_NAME if legacy else init.EVIDENCE_NAME
    (directory / name).write_bytes(json.dumps(value, sort_keys=True).encode("utf-8"))
    return value


class InitializationContracts(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_schema_separates_initializer_from_numeric_export_and_arena(self):
        value = init.validate_initialization_evidence(evidence())
        self.assertEqual(value["acceptance"]["independent_numeric_validation"], "pending")
        self.assertEqual(value["acceptance"]["arena_eligibility"], "not_granted")

    def test_legacy_v2_observation_retains_exact_domains_and_cannot_be_g0(self):
        value = init.validate_legacy_initialization_observation(legacy_evidence())
        self.assertEqual(value["model"]["schema"], "rovezero.pals-model.v2")
        self.assertEqual(value["model"]["semantics"], "rovezero.pals-model-semantics.v2")
        self.assertEqual(value["model"]["encoding"], "rovezero.pals-board-records.v2")
        self.assertEqual(value["model"]["config"]["max_records"], 128)
        self.assertEqual(value["acceptance"]["capability_scope"], "legacy_initializer_observation_only")
        with self.assertRaises(ValueError):
            init.validate_initialization_evidence(value)
        with self.assertRaises(ValueError):
            init.validate_legacy_initialization_observation(evidence())

    def test_legacy_reader_preserves_observation_scope_and_named_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            value = store_observation_fixture(directory, legacy=True)
            result = init.read_legacy_initialization_observation(directory / init.LEGACY_EVIDENCE_NAME)
            self.assertEqual(result["evidence"], value)
            self.assertEqual(result["execution_capability"], "not_issued")
            self.assertEqual(result["evidence_file"]["name"], init.LEGACY_EVIDENCE_NAME)
            with self.assertRaises(ValueError):
                init.read_initialization_evidence(directory / init.LEGACY_EVIDENCE_NAME)

    def test_legacy_replaced_producer_rejected_before_torch_import(self):
        registration = {"schema": init.LEGACY_SOURCE_SCHEMA, "source_commit": "a" * 40, "pins": init.observe_initialization_sources()}
        fake = mock.Mock(side_effect=AssertionError("must not invoke a replacement initializer"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(artifacts, "initialize_checkpoint", fake):
            with self.assertRaises(ValueError):
                init.initialize_legacy_with_observation(tmp, registration, initialization_id="legacy-rejection-only")
        fake.assert_not_called()

    def test_no_json_or_boolean_can_create_or_publish_capture(self):
        for value in (True, False, {}, evidence(), None):
            with self.subTest(value=type(value)):
                with self.assertRaises(TypeError):
                    init._publish_capture(value)
                with self.assertRaises(TypeError):
                    init._InitializationCapture(value, Path("."), evidence())

    def test_model_metadata_and_counter_types_are_strict(self):
        changes = [("metadata_fields", "training_steps", False), ("metadata_fields", "seed", True),
                   ("metadata_fields", "trained", 0), ("metadata_fields", "parameter_count", True),
                   ("claims", "nn_forwards", False), ("claims", "diagnostic", 0),
                   ("model", "seed", True), ("software", "deterministic_algorithms", 0)]
        for group, key, replacement in changes:
            value = evidence()
            value[group][key] = replacement
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                init.validate_initialization_evidence(value)
        for group in ("model", "metadata_fields"):
            value = evidence()
            value[group]["config"]["iterations"] = True
            with self.subTest(group=group), self.assertRaises(ValueError):
                init.validate_initialization_evidence(value)

    def test_old_semantics_other_roles_weights_or_training_rejected(self):
        mutations = [("metadata_fields", "schema", "rovezero.pals-model.v1"),
                     ("metadata_fields", "training_steps", 1), ("metadata_fields", "external_weights", True),
                     ("metadata_fields", "roles", ["proposer", "critic"]),
                     ("model", "encoding", "rovezero.pals-board-records.v2"),
                     ("model", "seed", 24), ("model", "roles", ["critic", "proposer", "validator"])]
        for group, key, replacement in mutations:
            value = evidence()
            value[group][key] = replacement
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                init.validate_initialization_evidence(value)

    def test_actual_readback_cannot_be_replaced_by_requested_or_unknown_software(self):
        for key, replacement in (("pytorch_version", "2.7.0"), ("observation", "requested_version"),
                                 ("device_capability", "cuda_available"), ("torch_thread_settings", {"intra": True, "inter": 1})):
            value = evidence()
            value["software"][key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                init.validate_initialization_evidence(value)

    def test_acceptance_claims_and_unknown_fields_rejected(self):
        for mutation in (lambda x: x.update(actual=True),
                         lambda x: x["acceptance"].update(arena_eligibility="accepted"),
                         lambda x: x["initialization"].update(mode="partial_copy"),
                         lambda x: x["source"]["pins"].update({".env": "a" * 64})):
            value = evidence()
            mutation(value)
            with self.assertRaises(ValueError):
                init.validate_initialization_evidence(value)

    def test_checkpoint_and_metadata_byte_identity_must_join(self):
        value = evidence()
        value["checkpoint"]["sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            init.validate_initialization_evidence(value)

    def test_inventory_rejects_duplicate_missing_wrong_dtype_and_bool_dimensions(self):
        for inventory_values in ([inventory(), inventory()], [],
                                 [{**inventory(), "dtype": "fp16_le"}],
                                 [{**inventory(), "shape": [True, 2]}],
                                 [{**inventory(), "bytes": 4}]):
            with self.subTest(inventory=inventory_values), self.assertRaises(ValueError):
                init.validate_parameter_inventory(inventory_values, 2)
        with self.assertRaises(ValueError):
            init.validate_parameter_inventory([inventory()], 3)

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in (b'{"seed":23,"seed":24}', b'{"value":NaN}', b'{"value":Infinity}'):
            with self.assertRaises(ValueError):
                init._parse_json(raw, 1024)

    def test_reader_issues_observation_only_even_for_self_declared_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            value = store_observation_fixture(directory)
            result = init.read_initialization_evidence(directory / init.EVIDENCE_NAME)
            self.assertEqual(result["validation"], "schema_and_named_file_identity")
            self.assertEqual(result["execution_capability"], "not_issued")
            self.assertEqual(result["evidence"], value)
            with self.assertRaises(TypeError):
                init._publish_capture(result)

    def test_reader_detects_asset_changes_and_schema_only_is_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            store_observation_fixture(directory)
            (directory / "untrained.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                init.read_initialization_evidence(directory / init.EVIDENCE_NAME)
            result = init.read_initialization_evidence(directory / init.EVIDENCE_NAME, join_files=False)
            self.assertEqual(result["validation"], "schema_only")
            self.assertEqual(result["execution_capability"], "not_issued")

    def test_reader_brackets_metadata_and_evidence_changes_during_join(self):
        for changing_name in ("checkpoint.json", init.EVIDENCE_NAME):
            with self.subTest(name=changing_name), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                store_observation_fixture(directory)
                original = init._read_small
                def changed(path, maximum):
                    result = original(path, maximum)
                    if Path(path).name == changing_name:
                        # Same parsed mapping, different bytes: an exact SHA
                        # join must still reject the observation.
                        Path(path).write_text(json.dumps(result, indent=1), encoding="utf-8")
                    return result
                with mock.patch.object(init, "_read_small", side_effect=changed), self.assertRaises(ValueError):
                    init.read_initialization_evidence(directory / init.EVIDENCE_NAME)

    def test_reader_rejects_metadata_change_wrong_owner_and_wrong_name(self):
        for mode in ("metadata", "owner", "name"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                store_observation_fixture(directory)
                path = directory / init.EVIDENCE_NAME
                if mode == "metadata":
                    (directory / "checkpoint.json").write_bytes(b'{}')
                elif mode == "owner":
                    (directory / "pals-artifact-owner.json").write_bytes(b'{}')
                else:
                    path = directory / "other.json"
                    path.write_bytes(b'{}')
                with self.assertRaises(ValueError):
                    init.read_initialization_evidence(path)

    def test_byte_publication_is_atomic_immutable_and_outside_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            owned(directory)
            target = init._atomic_publish_bytes(directory, b'{"contract_only":true}\n')
            self.assertEqual(target.read_bytes(), b'{"contract_only":true}\n')
            with self.assertRaises(FileExistsError):
                init._atomic_publish_bytes(directory, b'{"changed":true}\n')
            self.assertEqual(target.read_bytes(), b'{"contract_only":true}\n')
            self.assertFalse((directory / (init.EVIDENCE_NAME + ".partial")).exists())
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            owned(directory)
            (directory / ".git").mkdir()
            with self.assertRaises(ValueError):
                init._atomic_publish_bytes(directory, b'{}')

    def test_atomic_writer_allows_only_the_two_registered_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            owned(directory)
            with self.assertRaises(ValueError):
                init._atomic_publish_bytes(directory, b'{}', name="../outside.json")
            target = init._atomic_publish_bytes(directory, b'{"legacy_contract_only":true}', name=init.LEGACY_EVIDENCE_NAME)
            self.assertEqual(target.name, init.LEGACY_EVIDENCE_NAME)

    def test_existing_partial_preserved_and_no_final_evidence_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            owned(directory)
            partial = directory / (init.EVIDENCE_NAME + ".partial")
            partial.write_bytes(b"prior interrupted observation")
            with self.assertRaises(FileExistsError):
                init._atomic_publish_bytes(directory, b'{}')
            self.assertEqual(partial.read_bytes(), b"prior interrupted observation")
            self.assertFalse((directory / init.EVIDENCE_NAME).exists())

    def test_publish_race_never_overwrites_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            owned(directory)
            original_link = init.os.link
            def race(source, target):
                Path(target).write_bytes(b"other publisher won")
                return original_link(source, target)
            with mock.patch.object(init.os, "link", side_effect=race), self.assertRaises(FileExistsError):
                init._atomic_publish_bytes(directory, b'{"contract_only":true}')
            self.assertEqual((directory / init.EVIDENCE_NAME).read_bytes(), b"other publisher won")
            self.assertTrue((directory / (init.EVIDENCE_NAME + ".partial")).exists())

    def test_source_pins_use_only_fixed_registered_python_files(self):
        observed = init.observe_initialization_sources()
        self.assertEqual(set(observed), set(init.SOURCE_FILES))
        registration = {"schema": init.SOURCE_SCHEMA, "source_commit": "a" * 40, "pins": observed}
        init._join_sources(registration)
        registration["pins"]["config.py"] = "0" * 64
        with self.assertRaises(ValueError):
            init._join_sources(registration)

    def test_actual_initializer_rejects_replaced_producer_before_any_torch_import(self):
        registration = {"schema": init.SOURCE_SCHEMA, "source_commit": "a" * 40, "pins": init.observe_initialization_sources()}
        fake = mock.Mock(side_effect=AssertionError("must never invoke a replacement initializer"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(artifacts, "initialize_checkpoint", fake):
            with self.assertRaises(ValueError):
                init.initialize_g0_with_evidence(tmp, registration, initialization_id="pure-rejection-only")
        fake.assert_not_called()

    def test_initializer_refuses_old_assets_and_invalid_windows_before_execution(self):
        registration = {"schema": init.SOURCE_SCHEMA, "source_commit": "a" * 40, "pins": init.observe_initialization_sources()}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "old-checkpoint.pt").write_bytes(b"preserve historical fixture")
            with self.assertRaises(FileExistsError):
                init.initialize_g0_with_evidence(tmp, registration, initialization_id="pure-rejection-only")
        for window in (True, 0, -1, 3601):
            with self.assertRaises(ValueError):
                init.initialize_g0_with_evidence("unused", registration, initialization_id="pure-rejection-only", max_seconds=window)

    def test_symlink_destination_is_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "actual"
            target.mkdir()
            owned(target)
            link = root / "link"
            try:
                link.symlink_to(target, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("this host cannot create a test symlink")
            with self.assertRaises(ValueError):
                init._atomic_publish_bytes(link, b'{}')


class PartialCopyContracts(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)
        self.source = [inventory("copied", semantics="registered.core"), inventory("source_only", bits="b" * 64, semantics="source.private")]
        self.target = [inventory("copied", semantics="registered.core"), inventory("target_only", bits="c" * 64, semantics="target.private")]
        self.copy = {"schema": "rz-pals-actual-partial-initialization/1", "allowlist": {"copied": "registered.core"},
            "copied": [{"name": "copied", "shape": [2], "dtype": "fp32_le", "semantics": "registered.core",
                "source_sha256": "a" * 64, "target_sha256": "a" * 64, "operation": "copied_exact_fp32_bits"}],
            "ignored_source": ["source_only"], "ignored_target": ["target_only"], "source_profile": "full_line_interaction_v2",
            "target_profile": VIRTUAL_REVIEW_PROFILE, "learned_transfer_claim": False, "automatic_asset_conversion": False,
            "nn_forwards": 0, "optimizer_steps": 0}

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_exact_actual_copy_trace_is_purely_validated_without_publication_authority(self):
        self.assertEqual(init.validate_partial_copy_evidence(self.copy, self.source, self.target), self.copy)
        with self.assertRaises(TypeError):
            init._publish_capture(self.copy)

    def test_prepared_plan_or_missing_ignored_names_cannot_be_actual_copy(self):
        for mutation in (lambda x: x["copied"][0].update(operation="copy_after_complete_prevalidation"),
                         lambda x: x.update(ignored_source=[]), lambda x: x.update(ignored_target=[]),
                         lambda x: x.update(learned_transfer_claim=True), lambda x: x.update(optimizer_steps=False)):
            value = copy.deepcopy(self.copy)
            mutation(value)
            with self.assertRaises(ValueError):
                init.validate_partial_copy_evidence(value, self.source, self.target)

    def test_partial_copy_requires_complete_membership_shape_dtype_meaning_and_bits(self):
        for key, replacement in (("shape", [1]), ("dtype", "fp16_le"), ("semantics", "different.meaning"), ("sha256", "d" * 64)):
            target = copy.deepcopy(self.target)
            target[0][key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                init.validate_partial_copy_evidence(self.copy, self.source, target)
        with self.assertRaises(ValueError):
            init.validate_partial_copy_evidence(self.copy, self.source, self.target[1:])


if __name__ == "__main__":
    unittest.main()
