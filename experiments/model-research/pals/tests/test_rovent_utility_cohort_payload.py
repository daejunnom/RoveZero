"""Strict cohort profile/ACK observations, never a fabricated trial/target.

The file owner below comes from actual tiny temporary-file pin readback. All
payloads remain synthetic test data; the unimplemented source/witness/final
closure boundaries must not become known targets merely by recomputing hashes.
Profile, native byte framing and typed ACK expectations follow the Rust source
independently of the new payload reader's helpers. No model is initialized.
"""
import copy
from contextlib import contextmanager
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime", "tensorflow", "jax"}
                or fullname in {"rz_pals_model.model", "rz_pals_model.training", "rz_pals_model.nonzero_training",
                    "rz_pals_model.nnuec_trainable", "rz_pals_model.nnuec_utility_trainable"}):
            raise AssertionError("cohort payload tests cannot load a model or tensor runtime")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_utility_cohort_payload as reader
    from rz_pals_model import rovent_utility_cohort_files as files_reader
    from rz_pals_model import rovent_utility_training as training
    from test_rovent_utility_cohort_files import publication_fixture, write_fixture, file_pin, payloads, raw, budget
    from test_rovent_utility_cohort_training import fixture as preparation_fixture
finally:
    sys.meta_path.remove(_guard)


PROFILE_DOMAIN = "rz-rovent-utility-comparison-cohort/1"
TRIAL_DOMAIN = "rz-rovent-utility-cohort-trial/1"
COHORT_SCOPE = "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1"
PUBLICATION_FILE = "utility-cohort-publication.json"
ACK_SCHEMA = "rz-rovent-utility-cohort-durable-ack/1"
PROFILE_INPUT_FIELDS = ("input_lock_sha256", "source_sha256", "frozen_nnuec_model_sha256", "frozen_nnuec_namespace_sha256",
    "frozen_pc_model_sha256", "frozen_pc_namespace_sha256", "mode", "requested_recipients", "prelude_ns", "action_ns",
    "cleanup_ns", "logical_budget_sha256", "ready_policy_sha256", "cache_policy_sha256", "cost_class_sha256",
    "resource_plan_sha256", "resource_epoch", "usage", "cost_scope")


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def native_bytes_sha(domain, body):
    # UCI trial/materialization/pair bytes use a distinct length-framed ABI.
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + len(body).to_bytes(8, "little") + body).hexdigest()


def preparation():
    recipe, registration, spec = preparation_fixture()
    return training.prepare_cohort_training(recipe, registration, spec)


def profile_fixture(prepared):
    source = prepared["registration"]
    return {"domain": PROFILE_DOMAIN, "input": {
        "input_lock_sha256": h("synthetic original startup lock"), "source_sha256": source["collection_source_sha256"],
        "frozen_nnuec_model_sha256": source["frozen_base"]["weights_sha256"], "frozen_nnuec_namespace_sha256": h("synthetic NNUEC namespace"),
        "frozen_pc_model_sha256": h("synthetic P/C model"), "frozen_pc_namespace_sha256": h("synthetic P/C namespace"),
        "mode": "cpu_pc", "requested_recipients": ["proposer", "critic"], "prelude_ns": 1000000,
        "action_ns": 2000000, "cleanup_ns": 3000000, "logical_budget_sha256": h("synthetic logical budget"),
        "ready_policy_sha256": h("synthetic ready policy"), "cache_policy_sha256": h("synthetic cache policy"),
        "cost_class_sha256": h("synthetic cohort cost class"), "resource_plan_sha256": source["resource_profile_sha256"],
        "resource_epoch": 1, "usage": "collection_candidate", "cost_scope": COHORT_SCOPE}}


def typed_ack(publication, publication_body, blobs):
    """Trial ACK/4, collection ACK/6 and IO/8, never a live FD ACK."""
    publication_pin = file_pin(PUBLICATION_FILE, publication_body)
    checkpoints = []
    for ordinal, trial in enumerate(publication["trials"]):
        materialized = next((pin["bytes"] for pin in publication["materializations"]
            if pin["path"] == f"materializations/{ordinal:06d}.json"), 0)
        start = 10 + ordinal * 10
        checkpoints.append({"trial_sequence": ordinal + 1,
            "native_trial_sha256": native_bytes_sha(TRIAL_DOMAIN, blobs[trial["path"]]), "checkpoint": copy.deepcopy(trial),
            "collection_io": {"clock_epoch": 7, "started_tick_ns": start, "finished_tick_ns": start + 2,
                "elapsed_ns": 2, "written_bytes": materialized + trial["bytes"],
                "readback_bytes": materialized + 2 * trial["bytes"], "synced_files": 2 + int(materialized > 0),
                "synced_directories": 3 * (1 + int(materialized > 0))}})
    readback = publication["total_payload_bytes"] + len(publication_body)
    earlier = sum(pin["bytes"] for pin in (*publication["trials"], *publication["materializations"]))
    return {"schema": ACK_SCHEMA, "cohort_id": publication["cohort_id"], "cost_scope": COHORT_SCOPE,
        "publication": publication_pin, "collection_io": {"clock_epoch": 7, "started_tick_ns": 100,
            "finished_tick_ns": 103, "elapsed_ns": 3, "written_bytes": readback - earlier, "readback_bytes": readback,
            "synced_files": len(payloads(publication)) + 1, "synced_directories": 4},
        "trial_acknowledgements": checkpoints}


@contextmanager
def pinned_files(prepared, *, change_profile=None, change_ack=None,
                 change_trial_body=None, change_trial_raw=None, trial_count=2,
                 materialization_ordinals=(0, 1), pair_count=1):
    """Integrity-valid files with intentionally incomplete historical bodies.

    The minimal trial dictionaries expose only schema/cohort/sequence for the
    ACK-specific parser. All remaining bodies are opaque fixture text. None
    represent a resolved action, witness, materialization or final fence.
    """
    publication, blobs = publication_fixture(trial_count=trial_count,
        materialization_ordinals=materialization_ordinals, pair_count=pair_count)
    profile = profile_fixture(prepared)
    if change_profile is not None:
        change_profile(profile)
    blobs[publication["cohort_profile"]["path"]] = raw(profile)
    for ordinal, pin in enumerate(publication["trials"]):
        body = {"schema": TRIAL_DOMAIN, "cohort_id": publication["cohort_id"], "trial_sequence": ordinal + 1}
        if change_trial_body is not None:
            change_trial_body(body, ordinal)
        original = raw(body)
        if change_trial_raw is not None:
            original = change_trial_raw(original, ordinal)
        blobs[pin["path"]] = original
    for pin in payloads(publication):
        pin.update(file_pin(pin["path"], blobs[pin["path"]]))
    publication["total_payload_bytes"] = sum(pin["bytes"] for pin in payloads(publication))
    publication_body = raw(publication)
    ack = typed_ack(publication, publication_body, blobs)
    if change_ack is not None:
        change_ack(ack)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        reference, _publication, _ack, _blobs = write_fixture(root, publication=publication, blobs=blobs,
            publication_body=publication_body, ack=ack, ack_body=raw(ack))
        owner = files_reader.read_utility_cohort_files(reference, read_accounting=budget())
        yield owner, profile, ack


class UtilityCohortPayloadStrictTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def strict_rejection(self, files, prepared):
        with self.assertRaises(ValueError) as rejected:
            reader.validate_cohort_payloads(files, preparation=prepared)
        self.assertNotIsInstance(rejected.exception, reader.UnsupportedCohortPayload)

    def test_profile_declaration_accepts_exact_cpu_modes_and_detaches_metadata(self):
        for mode, recipients in (("cpu_pc", ["proposer", "critic"]),
                ("cpu_pcr", ["researcher", "critic", "proposer"])):
            value = profile_fixture(preparation())
            value["input"].update(mode=mode, requested_recipients=recipients)
            self.assertEqual(set(value["input"]), set(PROFILE_INPUT_FIELDS))
            with self.subTest(mode=mode):
                checked = reader.validate_cohort_profile(value)
                self.assertEqual(checked, value)
                checked["input"]["requested_recipients"].clear()
                self.assertEqual(value["input"]["requested_recipients"], recipients)
        # Zero is valid for some observed counters, but not declared windows.
        value = profile_fixture(preparation())
        value["input"].update(prelude_ns=1, action_ns=86400000000000 - 2, cleanup_ns=1)
        self.assertEqual(reader.validate_cohort_profile(value), value)

    def test_preparation_is_validated_before_any_files_or_legacy_conversion(self):
        class NoFiles:
            def __getattr__(self, _name):
                raise AssertionError("invalid preparation cannot inspect a payload")
        spec = preparation_fixture()[2]
        from test_rovent_utility_training import registration
        legacy = training.prepare_training(training.make_recipe(spec["optimizer_owner"]), registration(spec), spec)
        for prepared in (None, {}, legacy, {**preparation(), "performed_optimizer_updates": 1}):
            with self.subTest(schema=prepared.get("schema") if type(prepared) is dict else None):
                self.strict_rejection(NoFiles(), prepared)

    def test_profile_seal_is_canonical_domain_nul_without_native_length_or_raw_sha(self):
        profile = profile_fixture(preparation())
        original = raw(profile)
        compact = json.dumps(profile, sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(",", ":")).encode("utf-8")
        expected = hashlib.sha256(PROFILE_DOMAIN.encode("ascii") + b"\0" + compact).hexdigest()
        self.assertEqual(reader.cohort_profile_sha256(profile), expected)
        self.assertNotEqual(expected, hashlib.sha256(original).hexdigest())
        self.assertNotEqual(expected, native_bytes_sha(PROFILE_DOMAIN, compact))
        reversed_profile = {"input": dict(reversed(tuple(profile["input"].items()))), "domain": PROFILE_DOMAIN}
        self.assertEqual(reader.cohort_profile_sha256(reversed_profile), expected)
        changed = copy.deepcopy(profile)
        changed["input"]["requested_recipients"].reverse()
        self.assertNotEqual(reader.cohort_profile_sha256(changed), expected)

    def test_preparation_two_cannot_accept_a_duck_typed_or_success_flag_file_owner(self):
        class MetadataOnlyFiles:
            publication = lambda self: {"claimed_payloads_verified": True}
            raw_file = lambda self, _name: b"{}"
            durable_ack = lambda self: {"claimed_physical_complete": True}
        for value in (None, {}, MetadataOnlyFiles()):
            with self.subTest(kind=type(value).__name__):
                self.strict_rejection(value, preparation())

    def test_profile_roster_domain_and_unknown_fields_fail_before_incomplete_sources(self):
        changes = ((lambda value: value.update(domain="rz-rovent-utility-root-local-action/1"), "explicit profile domain"),
            (lambda value: value.update(extra="not a cohort declaration"), "profile fields mismatch"),
            (lambda value: value["input"].pop("resource_epoch"), "profile input fields mismatch"),
            (lambda value: value["input"].update(observed_available_recipients=["proposer"]), "profile input fields mismatch"))
        for index, (change, message) in enumerate(changes):
            value = profile_fixture(preparation())
            change(value)
            with self.subTest(public_profile=index), self.assertRaisesRegex(ValueError, message):
                reader.validate_cohort_profile(value)
            with self.subTest(case=index), pinned_files(preparation(), change_profile=change) as (files, _profile, _ack):
                self.strict_rejection(files, preparation())

    def test_cpu_modes_requested_order_and_exact_integer_windows_are_strict(self):
        changes = ((lambda value: value["input"].update(mode="hybrid_pc"), "CPU-only mode"),
            (lambda value: value["input"].update(requested_recipients=["proposer", "researcher"]), "unique ordered supported recipients"),
            (lambda value: value["input"].update(requested_recipients=["proposer", "proposer"]), "unique ordered supported recipients"),
            (lambda value: value["input"].update(requested_recipients=[]), "unique ordered supported recipients"),
            (lambda value: value["input"].update(prelude_ns=0), "positive bounded integer"),
            (lambda value: value["input"].update(action_ns=True), "invalid bounded unsigned integer"),
            (lambda value: value["input"].update(cleanup_ns=1.0), "invalid bounded unsigned integer"),
            (lambda value: value["input"].update(action_ns=2**64), "invalid bounded unsigned integer"),
            (lambda value: value["input"].update(action_ns=86400000000000), "declared window exceeds 24 hours"),
            (lambda value: value["input"].update(resource_epoch=0), "positive bounded integer"))
        for index, (change, message) in enumerate(changes):
            value = profile_fixture(preparation())
            change(value)
            with self.subTest(case=index), self.assertRaisesRegex(ValueError, message):
                reader.validate_cohort_profile(value)

    def test_collection_candidate_does_not_allow_root_local_or_unknown_cost_policies(self):
        changes = ((lambda value: value["input"].update(cost_scope="exclusive_submitted_global_target_route_and_descendants_with_required_drain"), "explicit usage/cost scope"),
            (lambda value: value["input"].update(usage="training_authorized"), "explicit usage/cost scope"),
            (lambda value: value["input"].update(frozen_nnuec_model_sha256="ab" * 33), "invalid SHA-256"),
            (lambda value: value["input"].update(source_sha256="AB" * 32), "invalid SHA-256"))
        for index, (change, message) in enumerate(changes):
            value = profile_fixture(preparation())
            change(value)
            with self.subTest(case=index), self.assertRaisesRegex(ValueError, message):
                reader.validate_cohort_profile(value)

    def test_trial_ack_returns_only_detached_data_and_respects_sparse_materializations(self):
        for count, materializations, pair_count in ((1, (), 0), (2, (0, 1), 1), (3, (0, 2), 3)):
            with self.subTest(trials=count), pinned_files(preparation(), trial_count=count,
                    materialization_ordinals=materializations, pair_count=pair_count) as (files, _profile, ack):
                checked = reader.validate_trial_acknowledgements(files)
                self.assertIs(type(checked), tuple)
                self.assertEqual(checked, tuple(ack["trial_acknowledgements"]))
                self.assertFalse(files.observation()["physical_completion_verified"])
                self.assertFalse(files.observation()["training_target_authorized"])
                checked[0]["collection_io"]["written_bytes"] = -1
                self.assertEqual(reader.validate_trial_acknowledgements(files), tuple(ack["trial_acknowledgements"]))

    def test_ack_minimal_body_acceptance_cannot_be_promoted_to_a_full_payload_target(self):
        with pinned_files(preparation()) as (files, _profile, _ack):
            self.assertEqual(len(reader.validate_trial_acknowledgements(files)), 2)
            with self.assertRaisesRegex(ValueError, "utility cohort original trial checkpoint fields mismatch"):
                reader.validate_cohort_payloads(files, preparation=preparation())

    def test_trial_ack_requires_original_file_owner_not_a_detached_flag(self):
        class MetadataOnlyAck:
            def publication(self):
                raise AssertionError("duck typed file owner must not be inspected")
        for value in (None, {}, MetadataOnlyAck()):
            with self.subTest(kind=type(value).__name__), self.assertRaisesRegex(ValueError, "original pinned file owner"):
                reader.validate_trial_acknowledgements(value)

    def test_trial_ack_body_is_closed_and_cannot_invent_complete_or_free_io(self):
        prefix = "utility cohort original trial ACK"
        changes = ((lambda value: value["trial_acknowledgements"].pop(), prefix + " complete coverage"),
            (lambda value: value["trial_acknowledgements"][0].update(physical_completion_verified=True), prefix + " fields"),
            (lambda value: value["trial_acknowledgements"][0].pop("checkpoint"), prefix + " fields"),
            (lambda value: value["trial_acknowledgements"][0].update(trial_sequence=2), prefix + " checkpoint/order"),
            (lambda value: value["trial_acknowledgements"][0].update(trial_sequence=True), "invalid bounded unsigned integer"),
            (lambda value: value["trial_acknowledgements"][0].update(native_trial_sha256="f" * 64), prefix + " native body seal"),
            (lambda value: value["trial_acknowledgements"][0]["checkpoint"].update(extra=True), prefix + " pin fields"),
            (lambda value: value["trial_acknowledgements"][0]["checkpoint"].update(bytes=1), prefix + " checkpoint/order"),
            (lambda value: value["trial_acknowledgements"][0]["checkpoint"].update(sha256="0" * 64), prefix + " checkpoint/order"),
            (lambda value: value["trial_acknowledgements"][0]["checkpoint"].update(bytes=True), prefix + " checkpoint/order"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(written_bytes=0), prefix + " I/O accounting"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(readback_bytes=1), prefix + " I/O accounting"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(synced_files=True), "invalid bounded unsigned integer"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(readback_bytes=2**64), "invalid bounded unsigned integer"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(clock_epoch=8), prefix + " I/O accounting"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(elapsed_ns=3), prefix + " I/O clock"),
            (lambda value: value["trial_acknowledgements"][0]["collection_io"].update(physical_zero_tick=12), prefix + " collection I/O fields"),
            (lambda value: value["trial_acknowledgements"][1]["collection_io"].update(started_tick_ns=0, finished_tick_ns=2), prefix + " I/O accounting"),
            (lambda value: value["trial_acknowledgements"][1]["collection_io"].update(started_tick_ns=100, finished_tick_ns=102), prefix + " I/O accounting"))
        for index, (change, message) in enumerate(changes):
            with self.subTest(case=index), pinned_files(preparation(), change_ack=change) as (files, _profile, _ack):
                with self.assertRaisesRegex(ValueError, message):
                    reader.validate_trial_acknowledgements(files)

    def test_trial_ack_native_seal_uses_original_raw_bytes_not_file_or_canonical_sha(self):
        def wrong_seal(kind):
            def change(ack):
                first = ack["trial_acknowledgements"][0]
                if kind == "file":
                    first["native_trial_sha256"] = first["checkpoint"]["sha256"]
                else:
                    body = {"schema": TRIAL_DOMAIN, "cohort_id": ack["cohort_id"], "trial_sequence": 1}
                    compact = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    if kind == "canonical":
                        first["native_trial_sha256"] = native_bytes_sha(TRIAL_DOMAIN, compact)
                    else:
                        original = raw(body)
                        first["native_trial_sha256"] = hashlib.sha256(TRIAL_DOMAIN.encode("ascii") + b"\0" + original).hexdigest()
            return change
        for kind in ("file", "canonical", "missing length"):
            with self.subTest(kind=kind), pinned_files(preparation(), change_ack=wrong_seal(kind)) as (files, _profile, _ack):
                with self.assertRaisesRegex(ValueError, "utility cohort original trial ACK native body seal"):
                    reader.validate_trial_acknowledgements(files)

    def test_trial_ack_rejects_body_order_or_domain_even_when_raw_pins_are_recomputed(self):
        changes = (lambda body, ordinal: body.update(schema="rz-rovent-utility-root-local-action/1") if ordinal == 0 else None,
            lambda body, ordinal: body.update(trial_sequence=True) if ordinal == 0 else None,
            lambda body, ordinal: body.update(trial_sequence=2) if ordinal == 0 else None,
            lambda body, ordinal: body.update(cohort_id="f" * 64) if ordinal == 0 else None)
        for index, change in enumerate(changes):
            with self.subTest(case=index), pinned_files(preparation(), change_trial_body=change) as (files, _profile, _ack):
                with self.assertRaisesRegex(ValueError, "utility cohort original trial ACK native body seal"):
                    reader.validate_trial_acknowledgements(files)

    def test_trial_ack_rejects_duplicate_json_keys_and_non_utf8_original_bytes(self):
        changes = (lambda original, ordinal: b'{"schema":"other",' + original.lstrip()[1:] if ordinal == 0 else original,
            lambda original, ordinal: b"\xff" + original if ordinal == 0 else original)
        for index, change in enumerate(changes):
            message = "duplicate keys" if index == 0 else "original UTF8 JSON"
            with self.subTest(case=index), pinned_files(preparation(), change_trial_raw=change) as (files, _profile, _ack):
                with self.assertRaisesRegex(ValueError, message):
                    reader.validate_trial_acknowledgements(files)


if __name__ == "__main__":
    unittest.main()
