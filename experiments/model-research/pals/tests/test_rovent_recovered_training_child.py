"""Recovered entry source contracts; no model/optimizer or learner execution."""
import copy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_rovent_training import NeuralImportsForbidden, membership_fixture, recipe_fixture
from test_rovent_training_bootstrap import recovered_fixture
from test_rovent_training_window_v2 import spec_fixture

_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_recovered_training_child as child
finally:
    sys.meta_path.remove(_guard)


def rules_fixture():
    from rz_pals_model.artifacts import VIRTUAL_RULES_BOUNDS, VIRTUAL_RULES_FEATURES
    from rz_pals_model.config import ModelConfig
    from rz_pals_model.virtual_review import RECORD_FEATURES, QUERY_FEATURES
    profile, base = "rz-pals-rules-virtual-review-v1", "fixture_contract_not_Rules_execution"
    metadata = ["metadata" + str(i) for i in range(16)]
    divergence = ["divergence" + str(i) for i in range(8)]
    semantic_fields = [profile, base, *metadata, *RECORD_FEATURES, *QUERY_FEATURES, *divergence, *VIRTUAL_RULES_FEATURES]
    semantic = hashlib.sha256()
    for field in semantic_fields:
        raw = field.encode("utf-8")
        semantic.update(len(raw).to_bytes(8, "little"))
        semantic.update(raw)
    return {"schema": "rovezero.pals-rules-descriptor.v3", "rules_input_profile": profile,
        "config": ModelConfig.for_profile("virtual_review_v1").to_dict(), "rules_base_semantics": base,
        "metadata_features": metadata, "divergence_features": divergence,
        "record_features": list(RECORD_FEATURES), "query_features": list(QUERY_FEATURES),
        "virtual_review_features": VIRTUAL_RULES_FEATURES, "bounds": VIRTUAL_RULES_BOUNDS,
        "semantic_fields": semantic_fields, "semantic_digest_algorithm": "sha256_u64le_length_prefixed_utf8_fields",
        "rules_input_semantic_sha256": semantic.hexdigest(), "rules_encoder_source_sha256": "7" * 64,
        "learned_input_compatibility": "unverified_declaration_only"}


def metadata_fixture():
    membership = membership_fixture()
    recipe = recipe_fixture(membership)
    spec = spec_fixture(recipe)
    startup = recovered_fixture()
    registration = startup["registration"]
    dataset = registration["current_dataset"]
    dataset.update(recipe_sha256=recipe.sha256, stage_id=spec["stage_id"],
        collection_source_sha256=spec["collection_source_sha256"],
        collection_runtime_library_sha256=spec["collection_runtime_library_sha256"],
        model_semantics_sha256=spec["collection_model_semantics_sha256"],
        model_epoch=spec["collection_model_epoch"], context_epoch=spec["collection_context_epoch"])
    startup["window_id"] = spec["window_id"]
    spec.update(trainer_source_sha256=registration["trainer_source_sha256"],
        trainer_python_executable_sha256=registration["trainer_python_executable"]["sha256"],
        trainer_dependency_set_sha256=registration["trainer_dependency_set_sha256"])
    payloads = {registration["recipe"]["path"]: recipe.to_dict(),
        registration["membership"]["path"]: membership, registration["window_spec"]["path"]: spec,
        registration["rules_profile"]["path"]: rules_fixture()}
    class Reads:
        def json(self, pin, *, maximum):
            return copy.deepcopy(payloads[pin["path"]])
    return startup, spec, payloads, Reads()


class RecoveredChildContractTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_startup_read_requires_exact_eof_and_bounded_integer(self):
        self.assertEqual(child._read_startup(io.BytesIO(b"{}"), 2), b"{}")
        for payload, count in ((b"{}x", 2), (b"{", 2), (b"", 0), (b"{}", True)):
            with self.subTest(payload=payload, count=count), self.assertRaisesRegex(ValueError, "startup"):
                child._read_startup(io.BytesIO(payload), count)

    def test_no_execute_fails_before_startup_and_runtime(self):
        with self.assertRaisesRegex(ValueError, "explicitly authorize"):
            child.recovered_rovent_child_main(["--startup-sha256", "1" * 64, "--startup-bytes", "2",
                "--execution-until-ns", "100", "--whole-until-ns", "200"], stdin=io.BytesIO(b"{}"))

    def test_metadata_keeps_collection_ort_and_trainer_identities_distinct(self):
        startup, spec, _payloads, reads = metadata_fixture()
        recipe, membership, selected, _rules, binding = child._metadata_inputs(startup, reads)
        self.assertEqual(selected, spec)
        self.assertEqual(selected["training_model_epoch"], 5)
        self.assertEqual(selected["collection_model_epoch"], 2)
        self.assertNotEqual(selected["collection_runtime_library_sha256"], selected["trainer_python_executable_sha256"])
        self.assertEqual(recipe.sha256, selected["recipe_sha256"])
        self.assertEqual(membership, membership_fixture())
        self.assertFalse(binding["execution_authorized"])
        self.assertFalse(binding["past_native_authority_reissued"])

    def test_old_window_codec_and_runtime_alias_are_not_converted(self):
        startup, spec, payloads, reads = metadata_fixture()
        selected = payloads[startup["registration"]["window_spec"]["path"]]
        selected["schema"] = "rz-rovent-train-window/1"
        with self.assertRaisesRegex(ValueError, "legacy conversion"):
            child._metadata_inputs(startup, reads)
        selected["schema"] = spec["schema"] = "rz-rovent-train-window/2"
        selected["runtime_sha256"] = selected.pop("collection_runtime_library_sha256")
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            child._metadata_inputs(startup, reads)

    def test_metadata_rejects_python_pin_relabelled_as_collection_ort(self):
        startup, spec, _payloads, reads = metadata_fixture()
        startup["registration"]["current_dataset"]["collection_runtime_library_sha256"] = spec["trainer_python_executable_sha256"]
        with self.assertRaisesRegex(ValueError, "identity namespace"):
            child._metadata_inputs(startup, reads)

    def test_missing_imported_torch_endpoint_pin_fails_without_read(self):
        class Reads:
            def verify(self, *_args, **_kwargs):
                raise AssertionError("unregistered endpoint must fail before any read")
        fake = SimpleNamespace(__file__="/explicit/not-a-runtime/torch.py", _C=None)
        with self.assertRaisesRegex(ValueError, "runtime pin missing"):
            child._verify_imported_runtime({"runtime_module_pins": []}, Reads(), fake)

    def test_history_report_copies_recorded_evidence_without_reissuing_authority(self):
        startup = recovered_fixture()
        result = child._history_scope(startup)
        self.assertFalse(result["past_native_authority_reissued"])
        self.assertFalse(result["past_physical_completion_reissued"])
        result["collection_admission"]["execution_authorized"] = True
        self.assertFalse(startup["collection_admission"]["execution_authorized"])

    def test_export_request_has_a_separate_domain_and_direct_new_output_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            startup = {"source_root": str(root / "source"), "registration": {"export_request": "fixture-only-pin",
                "output_root": str(root), "checkpoint_directory": str(root / "checkpoints"), "max_output_bytes": 8192}}
            request = {"schema": "rz-rovent-recovered-export-request/1", "kind": "deployment",
                "directory": str(root / "deployment"), "max_output_bytes": 4096}
            class Reads:
                def json(self, _pin, *, maximum):
                    return copy.deepcopy(request)
            self.assertEqual(child._export_request(startup, Reads()), request)
            request["schema"] = "rz-rovent-original-export-request/1"
            with self.assertRaisesRegex(ValueError, "new recovered"):
                child._export_request(startup, Reads())
            request["schema"] = "rz-rovent-recovered-export-request/1"
            request["directory"] = str(root / "nested" / "deployment")
            with self.assertRaisesRegex(ValueError, "new recovered"):
                child._export_request(startup, Reads())
            self.assertEqual(list(root.iterdir()), [])

    def test_incomplete_historical_graph_fails_before_loading_rows(self):
        from rz_pals_model.rovent_historical_corpus import HistoricalRoventCorpus
        startup = recovered_fixture()
        startup["registration"]["ancestor_datasets"] = [{"fixture": "incomplete"}]
        with patch.object(child, "_publication_metadata", return_value=({"canonical_index_sha256": "a" * 64}, ["b" * 64])), \
                patch.object(HistoricalRoventCorpus, "from_registration", side_effect=AssertionError("no unresolved row loading")):
            with self.assertRaisesRegex(ValueError, "cyclic, foreign, or incomplete"):
                child._intake(startup, object())

    def test_unrelated_registered_ancestor_is_not_silently_added_to_current_lineage(self):
        from rz_pals_model.rovent_historical_corpus import HistoricalRoventCorpus
        startup = recovered_fixture()
        startup["registration"]["ancestor_datasets"] = [{"fixture": "ancestor"}]
        fake_owner = SimpleNamespace(corpus=SimpleNamespace(index_semantic_sha256="a" * 64))
        with patch.object(child, "_publication_metadata", side_effect=[({"canonical_index_sha256": "a" * 64}, []), ({}, [])]), \
                patch.object(HistoricalRoventCorpus, "from_registration", return_value=fake_owner):
            with self.assertRaisesRegex(ValueError, "complete current dataset ancestry"):
                child._intake(startup, object())


if __name__ == "__main__":
    unittest.main()
