"""Stdlib-only comparative declarations, never actual checker admission."""
import copy
import hashlib
import json
import unittest

from rz_pals_model import comparative as comparative


# Rust comparative.rs tests assert this same literal. No DG05 selector, base
# receipt or live checker is executed by this canonical protocol fixture.
CANONICAL_VECTOR = '["rz-pals-comparative-overlay/1",{"criterion":{"recipe_id":"후보 비교","recipe_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"frozen_epoch":9007199254740993,"legal_moves":[1292,1804],"optional":null}]'
CHECKER_VECTOR = '["rz-pals-comparative-checker/1",{"horizon":2,"node_budget":9007199254740993,"perspective":"captured_side_to_move","profile":"fixture-depth-profile","registration_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","search_conditions":"공통 조건; roots excluded","search_implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","source":{"cpu_binary_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","evaluator_configuration_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","kind":"own_cpu","model_weights_sha256":null},"value_semantics_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"}]'


def raw(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def pin(value):
    return {"bytes": len(value), "sha256": digest(value)}


def cpu_source():
    return {"kind": "own_cpu", "cpu_binary_sha256": digest("fixture-cpu-binary"),
            "evaluator_configuration_sha256": digest("fixture-cpu-configuration"), "model_weights_sha256": None}


class Fixture:
    def __init__(self, role="proposer"):
        # Current anchors and view pin stand for an existing strict consumer's
        # output. These fixture declarations cannot prove that consumer ran.
        current = {"input_sha256": digest("fixture-input-" + role), "label_sha256": None, "game_id": "game",
                   "role": role, "rules_state_sha256": digest("state"), "rules_history_sha256": digest("history"),
                   "encoding_sha256": digest("encoding"), "source": cpu_source(), "frozen_epoch": 0,
                   "input_revision": 3, "side_to_move": "white", "legal_moves": [1292, 1804, 259]}
        snapshot = {name: current[name] for name in ("game_id", "role", "rules_state_sha256", "rules_history_sha256",
                                                    "encoding_sha256", "source", "frozen_epoch", "input_revision", "legal_moves")}
        snapshot.update({"opening_id": "opening", "line_genealogy_id": "line",
                         "position_command": "position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
                         "board_fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "white_to_move": True,
                         "actual_history": [], "transposition_sha256": digest("transposition"),
                         "capture_sequence": 8, "public_records": []})
        input_json = raw({"snapshot": snapshot, "sha256": current["input_sha256"]})
        tensor_json, lineage_json = raw({"fixture": "tensor sidecar"}), raw({"fixture": "lineage"})
        self.receipt = raw({"fixture": "independently pinned base receipt"})
        checker = {"registration_sha256": digest("checker-registration"), "source": cpu_source(),
                   "search_implementation_sha256": digest("search-implementation"), "value_semantics_sha256": digest("value-semantics"),
                   "profile": "fixture-depth-profile", "search_conditions": "fixture common conditions; roots excluded",
                   "horizon": 2, "node_budget": 9007199254740993, "perspective": "captured_side_to_move"}
        conditions = comparative.task_conditions(current, checker)
        candidates = [1292, 1804]
        checks = [{"candidate": candidate, "task_id": f"task-{candidate}", "conditions": copy.deepcopy(conditions),
                   "restriction": {"kind": "candidate_only", "root_moves": [candidate]}, "completion": "completed",
                   "completed_depth": 2, "evidence_id": f"evidence-{candidate}",
                   "evidence_artifact": pin(f"fixture-evidence-{candidate}".encode())} for candidate in candidates]
        self.body = {"version": comparative.COMPARATIVE_DOMAIN,
                     "parent": {"receipt_artifact": pin(self.receipt), "raw_dataset_sha256": digest("raw"), "split_sha256": digest("split"),
                                "current_view_sha256": digest("independently checked current view"),
                                "producer_roster_sha256": digest("roster"), "producer_envelope_sha256": digest("envelope")},
                     "criterion": {"recipe_id": "후보 비교", "recipe_sha256": digest("fixed-recipe")}, "checker": checker,
                     "prepared_inputs": [{"current": current, "producer_id": "input-producer",
                                          "producer_registration_sha256": digest("producer-registration"),
                                          "capture_evidence_sha256": digest("prepared-journal-entry"),
                                          "input_json": pin(input_json), "tensor_sidecar_json": pin(tensor_json), "lineage_json": pin(lineage_json)}],
                     "pairs": [{"pair_id": "pair", "input_sha256": current["input_sha256"],
                                "kind": "proposer_candidates" if role == "proposer" else "critic_responses",
                                "candidates": candidates, "checks": checks, "preference": "left"}]}
        self.checked = [copy.deepcopy(current)]
        self.actual = [{"input_sha256": current["input_sha256"], "input_json": input_json,
                        "tensor_sidecar_json": tensor_json, "lineage_json": lineage_json}]
        self.expected = {name: copy.deepcopy(self.body[name]) for name in ("parent", "criterion", "checker", "prepared_inputs")}

    def arguments(self, body=None):
        return {"overlay_bytes": raw(comparative.seal_overlay(self.body if body is None else body)),
                "expected_pins": copy.deepcopy(self.expected), "checked_current_anchors": copy.deepcopy(self.checked),
                "independent_current_view_sha256": self.expected["parent"]["current_view_sha256"],
                "actual_parent_receipt": self.receipt, "actual_prepared": copy.deepcopy(self.actual)}


class ComparativeMetadataTests(unittest.TestCase):
    def test_shared_literal_sorted_utf8_and_exact_u64_canonical_vector(self):
        value = {"optional": None, "legal_moves": [1292, 1804], "frozen_epoch": 9007199254740993,
                 "criterion": {"recipe_sha256": "a" * 64, "recipe_id": "후보 비교"}}
        self.assertEqual(comparative.canonical_metadata(comparative.COMPARATIVE_DOMAIN, value), CANONICAL_VECTOR.encode("utf-8"))
        self.assertEqual(comparative.metadata_digest(comparative.COMPARATIVE_DOMAIN, value), digest(CANONICAL_VECTOR))
        for bad in (True, False, -1, 1.0, 2**64, float("inf")):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                comparative.canonical_metadata(comparative.COMPARATIVE_DOMAIN, bad)
        checker = {"registration_sha256": "a" * 64,
                   "source": {"kind": "own_cpu", "cpu_binary_sha256": "c" * 64,
                              "evaluator_configuration_sha256": "d" * 64, "model_weights_sha256": None},
                   "search_implementation_sha256": "b" * 64, "value_semantics_sha256": "e" * 64,
                   "profile": "fixture-depth-profile", "search_conditions": "공통 조건; roots excluded",
                   "horizon": 2, "node_budget": 9007199254740993, "perspective": "captured_side_to_move"}
        self.assertEqual(comparative.canonical_metadata(comparative.CHECKER_DOMAIN, checker), CHECKER_VECTOR.encode("utf-8"))
        self.assertEqual(comparative.checker_namespace(checker), digest(CHECKER_VECTOR))

    def test_proposer_and_critic_metadata_never_admit_positive_training(self):
        for role in ("proposer", "critic"):
            fixture = Fixture(role)
            overlay = comparative.seal_overlay(fixture.body)
            self.assertEqual(comparative.load_overlay(raw(overlay)), overlay)
            audit = comparative.audit_metadata(**fixture.arguments())
            self.assertEqual((audit["scope"], audit["declared_preferences"], audit["masked_pairs"]), ("metadata_only", 1, 0))
            self.assertIs(audit["requires_actual_checker_admission"], True)
            self.assertNotIn("training_admitted", audit)
            self.assertNotIn("checker_verified", audit)
            checks = overlay["overlay"]["pairs"][0]["checks"]
            self.assertNotEqual(checks[0]["task_id"], checks[1]["task_id"])
            self.assertNotEqual(checks[0]["restriction"], checks[1]["restriction"])
            self.assertEqual(checks[0]["conditions"], checks[1]["conditions"])
            self.assertIsNone(overlay["overlay"]["prepared_inputs"][0]["current"]["label_sha256"])

    def test_unresolved_canceled_partial_unknown_missing_pairs_remain_masked(self):
        fixture = Fixture("critic")
        for status in ("partial", "cancelled", "unknown", "missing"):
            body = copy.deepcopy(fixture.body)
            check = body["pairs"][0]["checks"][0]
            check["completion"] = status
            if status == "missing":
                check.update(completed_depth=0, evidence_id=None, evidence_artifact=None)
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "remain masked"):
                comparative.seal_overlay(body)
            body["pairs"][0]["preference"] = "masked"
            audit = comparative.audit_metadata(**fixture.arguments(body))
            self.assertEqual((audit["declared_preferences"], audit["masked_pairs"]), (0, 1))
            self.assertIs(audit["requires_actual_checker_admission"], True)

    def test_namespace_profile_horizon_perspective_state_epoch_revision_and_masks(self):
        fixture = Fixture()
        for name, value in (("checker_namespace_sha256", digest("other")), ("profile", "other"),
                            ("search_conditions", "changed"), ("horizon", 3), ("node_budget", 9007199254740994),
                            ("perspective", "white"), ("rules_state_sha256", digest("other")),
                            ("rules_history_sha256", digest("other")), ("frozen_epoch", 1), ("input_revision", 4)):
            body = copy.deepcopy(fixture.body)
            body["pairs"][0]["checks"][0]["conditions"][name] = value
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "conditions"):
                comparative.seal_overlay(body)
        for change in (lambda c: c.update(restriction={"kind": "candidate_only", "root_moves": [1292, 1804]}),
                       lambda c: c.update(completed_depth=1), lambda c: c.update(evidence_id=None, evidence_artifact=None)):
            body = copy.deepcopy(fixture.body)
            change(body["pairs"][0]["checks"][0])
            with self.assertRaises(ValueError):
                comparative.seal_overlay(body)
        for name in ("task_id", "evidence_id"):
            body = copy.deepcopy(fixture.body)
            body["pairs"][0]["checks"][1][name] = body["pairs"][0]["checks"][0][name]
            with self.assertRaises(ValueError):
                comparative.seal_overlay(body)

    def test_illegal_duplicate_reordered_pairs_and_auxiliary_queries(self):
        fixture = Fixture()
        for candidates in ([1292, 65535], [1292, 1292], [1804, 1292]):
            body = copy.deepcopy(fixture.body)
            body["pairs"][0]["candidates"] = candidates
            with self.assertRaises(ValueError):
                comparative.seal_overlay(body)
        body = copy.deepcopy(fixture.body)
        pair = copy.deepcopy(body["pairs"][0]); pair["pair_id"] = "another"
        body["pairs"].append(pair)
        with self.assertRaisesRegex(ValueError, "duplicate.*candidate"):
            comparative.seal_overlay(body)
        for kind in ("critic_responses", "critic_divergences", "divergence"):
            body = copy.deepcopy(fixture.body); body["pairs"][0]["kind"] = kind
            with self.assertRaisesRegex(ValueError, "auxiliary adapter"):
                comparative.seal_overlay(body)
        self.assertEqual(comparative.query_boundary("critic_divergences"), "requires_auxiliary_adapter")

    def test_independent_pins_actual_bytes_and_current_label_mismatch(self):
        fixture = Fixture()
        changes = (("criterion", "recipe_sha256"), ("parent", "producer_envelope_sha256"),
                   ("parent", "raw_dataset_sha256"), ("parent", "split_sha256"),
                   ("parent", "producer_roster_sha256"), ("checker", "registration_sha256"))
        for section, name in changes:
            args = fixture.arguments(); args["expected_pins"][section][name] = digest("other")
            with self.subTest(pin=name), self.assertRaises(ValueError):
                comparative.audit_metadata(**args)
        args = fixture.arguments(); args["expected_pins"]["prepared_inputs"][0]["producer_registration_sha256"] = digest("other")
        with self.assertRaises(ValueError):
            comparative.audit_metadata(**args)
        for artifact in ("input_json", "tensor_sidecar_json", "lineage_json"):
            args = fixture.arguments(); args["actual_prepared"][0][artifact] += b"changed"
            with self.subTest(artifact=artifact), self.assertRaises(ValueError):
                comparative.audit_metadata(**args)
        args = fixture.arguments(); args["actual_parent_receipt"] += b"changed"
        with self.assertRaises(ValueError):
            comparative.audit_metadata(**args)
        args = fixture.arguments(); args["independent_current_view_sha256"] = digest("stale")
        with self.assertRaisesRegex(ValueError, "current view"):
            comparative.audit_metadata(**args)
        body = copy.deepcopy(fixture.body); body["prepared_inputs"][0]["current"]["label_sha256"] = digest("old label")
        args = fixture.arguments(body); args["expected_pins"]["prepared_inputs"] = body["prepared_inputs"]
        with self.assertRaisesRegex(ValueError, "current label/input"):
            comparative.audit_metadata(**args)

    def test_supplied_current_anchor_sanity_does_not_claim_chain_validation(self):
        fixture = Fixture()
        fixture.body["prepared_inputs"][0]["current"]["label_sha256"] = digest("checked current leaf")
        fixture.checked = [copy.deepcopy(fixture.body["prepared_inputs"][0]["current"])]
        fixture.expected["prepared_inputs"] = copy.deepcopy(fixture.body["prepared_inputs"])
        audit = comparative.audit_metadata(**fixture.arguments())
        self.assertNotIn("chain_validated", audit)
        self.assertNotIn("base_admitted", audit)
        args = fixture.arguments(); args["checked_current_anchors"][0]["label_sha256"] = digest("old leaf")
        with self.assertRaisesRegex(ValueError, "current label/input"):
            comparative.audit_metadata(**args)
        args = fixture.arguments(); args["checked_current_anchors"].append(copy.deepcopy(args["checked_current_anchors"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate.*current"):
            comparative.audit_metadata(**args)

    def test_duplicate_unknown_fields_and_strict_u64_reject_before_seal_check(self):
        fixture = Fixture()
        serialized = raw(comparative.seal_overlay(fixture.body))
        bads = (serialized.replace(b'"version":', b'"version":"duplicate","version":', 1),
                serialized.replace(b'"overlay":', b'"unknown":0,"overlay":', 1))
        for bad in bads:
            with self.assertRaises(ValueError):
                comparative.load_overlay(bad)
        for bad in (b"9007199254740993.0", b"true", b"18446744073709551616", b"-1"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                comparative.load_overlay(serialized.replace(b"9007199254740993", bad))
        # A nested True must not equal integer 1 in equality-based pin checks.
        body = copy.deepcopy(fixture.body); body["pairs"][0]["checks"][0]["conditions"]["frozen_epoch"] = False
        with self.assertRaises(ValueError):
            comparative.seal_overlay(body)


if __name__ == "__main__":
    unittest.main()
