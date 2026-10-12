"""Stdlib-only declaration fixtures, not live engines or training admission."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from rz_pals_model import frozen_producer as frozen


def raw(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def source(name="model-a"):
    return {"kind": "own_pals", "model_configuration_sha256": digest(name), "model_weights_sha256": digest("weights-" + name)}


def cpu_source():
    return {"kind": "own_cpu", "cpu_binary_sha256": digest("fixture-cpu"),
            "evaluator_configuration_sha256": digest("fixture-cpu-config"), "model_weights_sha256": None}


def row(input_source, *, sequence=8, epoch=1, encoding=None, role="proposer", game="game"):
    snapshot = {"game_id": game, "opening_id": "opening-" + game, "line_genealogy_id": "line-" + game,
                "position_command": "position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
                "board_fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "actual_history": [],
                "rules_state_sha256": digest("state-" + game), "rules_history_sha256": digest("history"),
                "transposition_sha256": digest("transposition-" + game), "encoding_sha256": encoding or digest("encoding"),
                "source": input_source, "frozen_epoch": epoch, "input_revision": 1, "capture_sequence": sequence,
                "white_to_move": True, "role": role, "legal_moves": [1292, 1804], "public_records": []}
    # The input's old seal is only fixture setup; shared metadata domains never
    # replace that domain or hash raw labels using a guessed Rust float format.
    identity = digest(json.dumps(["rz-pals-data/2", snapshot], ensure_ascii=False, separators=(",", ":")).encode())
    return {"input": {"snapshot": snapshot, "sha256": identity}, "future_label": None, "verifier_private": None}


class Fixture:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)
        unique = {value["input"]["sha256"]: value for value in self.rows}
        sources = [value["input"]["snapshot"]["source"] for value in unique.values()]
        self.registry = {"cpu_binary_sha256": sorted({value["cpu_binary_sha256"] for value in sources if value["kind"] == "own_cpu"}),
                         "input_sources": list({raw(value): value for value in sources}.values())}
        self.pins, bindings = [], []
        for index, value in enumerate(unique.values()):
            s = value["input"]["snapshot"]
            native_epoch = {"kind": "encoding_only_zero"} if s["source"]["kind"] == "own_cpu" else {"kind": "frozen_model_epoch", "sha256": s["source"]["model_weights_sha256"]}
            producer_id = f"producer-{index}"
            self.pins.append({"game_id": s["game_id"], "producer_id": producer_id, "registration_sha256": digest(producer_id),
                              "source": s["source"], "frozen_epoch": s["frozen_epoch"],
                              "encoding_policy": {"kind": "native_exact", "encoding_sha256": s["encoding_sha256"],
                                                  "encoder_source_sha256": digest("encoder"), "native_model_epoch": native_epoch}})
            bindings.append({"input_sha256": value["input"]["sha256"], "game_id": s["game_id"], "producer_id": producer_id,
                             "capture_sequence": s["capture_sequence"], "capture_evidence_sha256": digest("capture-evidence")})
        self.roster = frozen.seal_roster({"version": frozen.ROSTER_DOMAIN, "game_producers": self.pins})
        self.capture = frozen.seal_capture({"version": frozen.CAPTURE_DOMAIN, "bindings": bindings})
        # This pin stands for the existing base/DG05 audit, supplied separately.
        # No fixture here claims to have run that audit or a live producer.
        self.checked_view = digest("independently-checked-current-view-fixture")

    def arguments(self):
        records = b"".join(raw(value) + b"\n" for value in self.rows)
        registry = raw(self.registry) + b"\n"
        audit = {"records": len(self.rows), "canonical_dataset_sha256": digest("independently-pinned-rust-raw-seal"),
                 "canonical_split_sha256": digest("independently-pinned-rust-split-seal")}
        receipt = raw({"complete": True, "failure": None, "audit": audit,
                       "artifacts": {"records.jsonl": {"bytes": len(records), "sha256": digest(records)},
                                     "source-registry.jsonl": {"bytes": len(registry), "sha256": digest(registry)}}})
        capture = raw(self.capture)
        envelope = frozen.seal_envelope({"version": frozen.ENVELOPE_DOMAIN, "roster_sha256": self.roster["sha256"],
                                         "owned_sources_sha256": frozen.owned_sources_digest(self.registry),
                                         "raw_dataset_sha256": audit["canonical_dataset_sha256"], "split_sha256": audit["canonical_split_sha256"],
                                         "current_view_sha256": self.checked_view, "raw_records": len(self.rows),
                                         "unique_inputs": len(self.capture["capture"]["bindings"]), "capture_sha256": self.capture["sha256"],
                                         "capture_artifact": {"bytes": len(capture), "sha256": digest(capture)}})
        return {"roster_bytes": raw(self.roster), "envelope_bytes": raw(envelope), "capture_bytes": capture,
                "independently_registered": self.pins, "source_registry_bytes": registry,
                "raw_receipt_bytes": receipt, "records_bytes": records, "expected_raw_receipt_sha256": digest(receipt),
                "checked_current_view_sha256": self.checked_view}


class FrozenProducerMetadataTests(unittest.TestCase):
    def test_shared_float_free_canonical_vector(self):
        path = Path(__file__).resolve().parents[4] / "crates/rz-experiments/src/pals_data/frozen_producer_vector.fixture"
        vector = json.loads(path.read_text(encoding="utf-8"))
        roster = frozen.seal_roster(vector["roster_body"])
        self.assertEqual(frozen.canonical_metadata(frozen.ROSTER_DOMAIN, roster["roster"]), vector["canonical_roster"].encode("utf-8"))
        self.assertEqual(roster["sha256"], digest(vector["canonical_roster"]))
        self.assertEqual(roster["roster"]["game_producers"][0]["frozen_epoch"], 9007199254740993)
        self.assertEqual(frozen.load_roster(raw(roster)), roster)

    def test_two_models_one_game_and_cpu_absent_weights_are_metadata_only(self):
        fixture = Fixture([row(source("a")), row(source("b"), sequence=9, epoch=9)])
        audit = frozen.audit_metadata(**fixture.arguments())
        self.assertEqual((audit["scope"], audit["unique_inputs"], audit["native_exact_metadata_inputs"]), ("metadata_only", 2, 2))
        self.assertNotIn("training_admitted", audit)
        self.assertNotIn("live_producer_verified", audit)
        fixture = Fixture([row(cpu_source(), epoch=0)])
        self.assertIsNone(fixture.roster["roster"]["game_producers"][0]["source"]["model_weights_sha256"])
        self.assertEqual(frozen.audit_metadata(**fixture.arguments())["unique_inputs"], 1)

    def test_same_producer_source_epoch_and_native_encoding_drift(self):
        for second in (row(source("b"), sequence=9), row(source("a"), sequence=9, epoch=2),
                       row(source("a"), sequence=9, encoding=digest("changed-encoding"))):
            with self.subTest(snapshot=second["input"]["snapshot"]):
                fixture = Fixture([row(source("a")), second])
                body = copy.deepcopy(fixture.capture["capture"])
                for binding in body["bindings"]:
                    binding["producer_id"] = "producer-0"
                fixture.capture = frozen.seal_capture(body)
                with self.assertRaisesRegex(ValueError, "drift"):
                    frozen.audit_metadata(**fixture.arguments())

    def test_one_registered_producer_can_capture_multiple_stable_inputs(self):
        fixture = Fixture([row(source("a")), row(source("a"), sequence=9)])
        body = copy.deepcopy(fixture.capture["capture"])
        for binding in body["bindings"]:
            binding["producer_id"] = "producer-0"
        fixture.capture = frozen.seal_capture(body)
        fixture.pins = fixture.pins[:1]
        fixture.roster = frozen.seal_roster({"version": frozen.ROSTER_DOMAIN, "game_producers": fixture.pins})
        self.assertEqual(frozen.audit_metadata(**fixture.arguments())["unique_inputs"], 2)

    def test_registration_pin_and_owned_registry_cannot_be_enrolled_by_row(self):
        fixture = Fixture([row(source())])
        arguments = fixture.arguments()
        arguments["independently_registered"] = copy.deepcopy(fixture.pins)
        arguments["independently_registered"][0]["registration_sha256"] = digest("changed-registration")
        with self.assertRaisesRegex(ValueError, "independent registration"):
            frozen.audit_metadata(**arguments)
        fixture.registry["input_sources"] = [source("other")]
        with self.assertRaisesRegex(ValueError, "owned source"):
            frozen.audit_metadata(**fixture.arguments())

    def test_whole_history_count_preserved_with_one_input_binding(self):
        original = row(cpu_source(), epoch=0)
        first, latest = copy.deepcopy(original), copy.deepcopy(original)
        first["future_label"] = {"observed_sequence": 9, "provenance": {"source": "actual_game", "result": {
            "game_id": "game", "outcome": "unknown", "ending": "user_stop", "raw_evidence_sha256": digest("game-evidence")}},
            "policy": None, "value_wdl": None, "white_to_move": True, "counterexample": None, "verifier_tasks": None, "supersedes_label_sha256": None}
        latest["future_label"] = copy.deepcopy(first["future_label"])
        latest["future_label"].update(observed_sequence=10, supersedes_label_sha256=digest(raw(["rz-pals-label/1", {
            "input_sha256": original["input"]["sha256"], "label": first["future_label"]}])))
        fixture = Fixture([original, first, latest])
        before = copy.deepcopy(fixture.rows)
        audit = frozen.audit_metadata(**fixture.arguments())
        self.assertEqual((audit["raw_records"], audit["unique_inputs"]), (3, 1))
        self.assertEqual(fixture.rows, before)
        arguments = fixture.arguments()
        arguments["records_bytes"] = arguments["records_bytes"].replace(b"user_stop", b"unresolved")
        with self.assertRaisesRegex(ValueError, "actual raw artifact"):
            frozen.audit_metadata(**arguments)

    def test_missing_duplicate_unobserved_and_changed_capture_bindings(self):
        fixture = Fixture([row(source("a")), row(source("b"), sequence=9)])
        body = copy.deepcopy(fixture.capture["capture"])
        body["bindings"].append(copy.deepcopy(body["bindings"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate input"):
            frozen.seal_capture(body)
        body = copy.deepcopy(fixture.capture["capture"])
        body["bindings"].pop()
        fixture.capture = frozen.seal_capture(body)
        with self.assertRaisesRegex(ValueError, "missing input"):
            frozen.audit_metadata(**fixture.arguments())
        fixture = Fixture([row(source("a")), row(source("b"), sequence=9)])
        body = copy.deepcopy(fixture.capture["capture"])
        body["bindings"][0]["input_sha256"] = digest("unobserved")
        fixture.capture = frozen.seal_capture(body)
        with self.assertRaisesRegex(ValueError, "missing input"):
            frozen.audit_metadata(**fixture.arguments())
        fixture = Fixture([row(source())])
        arguments = fixture.arguments()
        arguments["capture_bytes"] += b" "
        with self.assertRaisesRegex(ValueError, "capture byte/seal"):
            frozen.audit_metadata(**arguments)

    def test_private_derived_encoding_always_requires_existing_adapter(self):
        original = row(source("v"), role="verifier", encoding=digest("query-specific-encoding"))
        original["verifier_private"] = {"task_kind": "resume_task", "control_sha256": digest("private-control"),
                                        "private_latent": [0.0, -0.0]}
        fixture = Fixture([original])
        with self.assertRaisesRegex(ValueError, "private verifier row requires derived"):
            frozen.audit_metadata(**fixture.arguments())
        policy = {"kind": "private_checked_derived_query", "encoding_schema_sha256": digest("schema"),
                  "private_encoder_source_sha256": digest("private-encoder"), "parent_encoding_sha256": digest("native-encoding"),
                  "parent_encoder_source_sha256": digest("native-encoder")}
        fixture.pins[0]["encoding_policy"] = policy
        fixture.roster = frozen.seal_roster({"version": frozen.ROSTER_DOMAIN, "game_producers": fixture.pins})
        audit = frozen.audit_metadata(**fixture.arguments())
        self.assertEqual(audit["native_exact_metadata_inputs"], 0)
        self.assertEqual(len(audit["requires_derived_adapter"]), 1)
        pending = audit["requires_derived_adapter"][0]
        self.assertNotEqual(pending["derived_encoding_sha256"], pending["encoding_schema_sha256"])
        self.assertEqual(audit["scope"], "metadata_only")

    def test_strict_u64_duplicate_keys_unknown_fields_and_float_rejection(self):
        vector = Path(__file__).resolve().parents[4] / "crates/rz-experiments/src/pals_data/frozen_producer_vector.fixture"
        body = json.loads(vector.read_text(encoding="utf-8"))["roster_body"]
        valid = raw(frozen.seal_roster(body))
        for invalid in (b"true", b"1.0", b"18446744073709551616", b"-1"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                frozen.load_roster(valid.replace(b"9007199254740993", invalid))
        duplicate = valid.replace('"game_id":"경기"'.encode(), '"game_id":"경기","game_id":"경기"'.encode())
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            frozen.load_roster(duplicate)
        body["unknown"] = None
        with self.assertRaises(ValueError):
            frozen.seal_roster(body)
        with self.assertRaisesRegex(ValueError, "float"):
            frozen.canonical_metadata(frozen.ROSTER_DOMAIN, {"value": 0.1})

    def test_raw_receipt_current_view_registry_seals_and_aggregate_byte_cap(self):
        fixture = Fixture([row(cpu_source(), epoch=0)])
        for change in ("receipt", "view", "registry", "budget"):
            arguments = fixture.arguments()
            if change == "receipt":
                arguments["expected_raw_receipt_sha256"] = digest("other-receipt")
            elif change == "view":
                arguments["checked_current_view_sha256"] = digest("other-view")
            elif change == "registry":
                arguments["source_registry_bytes"] += b" "
            else:
                arguments["max_input_bytes"] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                frozen.audit_metadata(**arguments)

    def test_malformed_receipt_asset_map_is_rejected_even_with_independent_pin(self):
        fixture = Fixture([row(cpu_source(), epoch=0)])
        for invalid in (None, [], "artifacts"):
            arguments = fixture.arguments()
            receipt = json.loads(arguments["raw_receipt_bytes"])
            receipt["artifacts"] = invalid
            arguments["raw_receipt_bytes"] = raw(receipt)
            arguments["expected_raw_receipt_sha256"] = digest(arguments["raw_receipt_bytes"])
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "artifacts must be an object"):
                frozen.audit_metadata(**arguments)

    def test_roster_order_is_independent_and_game_specific_identity_is_allowed(self):
        fixture = Fixture([row(source("a"), game="game-a"), row(source("b"), game="game-b")])
        body = copy.deepcopy(fixture.roster["roster"])
        for pin in body["game_producers"]:
            pin["producer_id"] = "same-producer"
        roster = frozen.seal_roster(body)
        body["game_producers"].reverse()
        self.assertEqual(frozen.seal_roster(body), roster)
        body["game_producers"].append(copy.deepcopy(body["game_producers"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate game/producer"):
            frozen.seal_roster(body)


if __name__ == "__main__":
    unittest.main()
