"""Mask, full-line and namespace contracts; no neural/optimizer execution."""
import copy
from dataclasses import asdict, replace
import hashlib
import json
import unittest

from rz_pals_model.config import ModelConfig
from rz_pals_model.target_coverage import (CHECKER_NAMESPACE_SCHEMA, checker_value_namespace,
                                         inspect_full_line_contexts, target_coverage_report)
from rz_pals_model.training import (DivergenceContext, TaskContext, _canonical, _encoding_identity,
                                   encoded_from_sidecar)
from test_training import dataset, encoding, fixture, move, native_sidecar, outcome_label, sha


def full_line():
    return {"records": [{"moves": [{"from": 12, "to": 28, "promotion": 0}],
                          "parent": None, "supersedes": None}],
            "query_prefix": [], "query_proposal": [{"from": 4, "to": 5, "promotion": 0}],
            "query_counter": [{"from": 60, "to": 59, "promotion": 0}]}


def sidecar_v2(row):
    sidecar = native_sidecar(row)
    tensor = json.loads(sidecar["tensor_json"])
    tensor["full_line"] = full_line()
    sidecar.update(version="rz-pals-native-input-sidecar/2", model_profile="full_line_interaction_v2",
                   encoding_profile="rovezero.pals-board-records.v2", tensor_json=json.dumps(tensor, separators=(",", ":")))
    sidecar["tensor_sha256"] = sha(sidecar["tensor_json"])
    names = ("version", "input_sha256", "encoding_sha256", "encoder_source_sha256", "model_epoch_kind",
             "canonical_tensor_sha256", "tensor_json", "tensor_sha256", "record_sources", "model_profile", "encoding_profile")
    sidecar["sha256"] = hashlib.sha256(json.dumps([sidecar[name] for name in names], sort_keys=True,
                                                 ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return sidecar


def seal_struct(value):
    value["sha256"] = ""
    value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                                separators=(",", ":")).encode()).hexdigest()
    return value


class TargetCoverageTests(unittest.TestCase):
    def test_unknown_outcome_remains_masked_but_owned_policy_is_counted(self):
        row = fixture()
        row["future_label"] = outcome_label(row, outcome="unknown", ending="ply_limit")
        admitted = dataset([row])
        before = copy.deepcopy(admitted.records)
        report = target_coverage_report(admitted, source_kind="contract_fixture")
        self.assertEqual(report["roles"]["proposer"]["valid_target_counts"], {"policy": 1, "wdl": 0})
        self.assertEqual(report["roles"]["proposer"]["mask_reasons"]["wdl"], {"unknown_outcome_ply_limit": 1})
        self.assertFalse(report["actual_target_coverage_claimed"])
        self.assertFalse(report["diagnostic_fixture_counts_included"])
        self.assertEqual(before, admitted.records)
        with self.assertRaisesRegex(ValueError, "checked frozen"):
            target_coverage_report(admitted, source_kind="actual_collector")

    def test_disputed_unexamined_repair_labels_cannot_become_binary_targets(self):
        for validity in ("disputed", "not_examined", "supported_after_repair", "refuted_by_repair"):
            row = fixture("critic")
            context = DivergenceContext(sha("challenged"), 3, 1)
            label = outcome_label(row, policy=False, outcome="unknown", ending="unresolved")
            label["counterexample"] = {"challenged_line_sha256": context.challenged_line_sha256,
                                        "divergence_ply": 3, "response_line": [move(12, 20)],
                                        "repair_line": [move(12, 28)], "validity": validity,
                                        "input_revision": 1, "legality_evidence_sha256": sha("fixture-legality")}
            row["future_label"] = label
            e = encoding(row, divergences=[(context, tuple([0.0] * 8))])
            report = target_coverage_report(dataset([row], {e.input_sha256: e}), source_kind="contract_fixture")
            expected = int(validity in ("supported_after_repair", "refuted_by_repair"))
            self.assertEqual(report["roles"]["critic"]["valid_target_counts"]["divergence"], expected)
            self.assertEqual(report["roles"]["critic"]["rows_with_valid_target"], expected)

    def test_v_targets_require_explicit_exact_context_and_observed_rank(self):
        row = fixture("verifier")
        context = TaskContext(sha("branch"), sha("profile"), 1)
        row["future_label"] = outcome_label(row, policy=False)
        row["future_label"]["verifier_tasks"] = [
            {"task": "resume_task", **asdict(context), "preference_rank": 0, "information_gain_evidence_sha256": sha("gain")},
            {"task": "defer", **asdict(context), "preference_rank": None, "information_gain_evidence_sha256": None}]
        e = encoding(row, task_queries=[(context, tuple([0.0] * 16))])
        admitted = dataset([row], {e.input_sha256: e})
        unresolved = target_coverage_report(admitted, source_kind="contract_fixture")
        selected = target_coverage_report(admitted, source_kind="contract_fixture", task_contexts={0: context})
        self.assertEqual(unresolved["roles"]["verifier"]["valid_target_counts"]["task"], 0)
        self.assertEqual(selected["roles"]["verifier"]["valid_target_counts"]["task"], 1)
        self.assertEqual(unresolved["roles"]["verifier"]["mask_reasons"]["task"], {"explicit_task_context_not_selected": 1})

    def test_legacy_encoding_identity_and_v2_exact_tensor_conversion(self):
        row = fixture()
        legacy = encoding(row)
        old = asdict(legacy)
        del old["model_profile"]
        del old["full_line"]
        self.assertEqual(_encoding_identity(legacy), _canonical("rz-pals-python-immutable-encoding/1", old))
        sidecar = sidecar_v2(row)
        e = encoded_from_sidecar(sidecar, row, expected_encoder_source_sha256=sha("fixture-encoder-source"))
        self.assertEqual(e.full_line, full_line())
        admitted = dataset([row], {e.input_sha256: e})
        batch = admitted.collate([0], "proposer")
        self.assertEqual(batch.model_profile, "full_line_interaction_v2")
        self.assertEqual(batch.inputs.record_line_tokens[0, 0, 0].tolist(), [12, 28, 0])
        self.assertEqual(batch.inputs.record_line_mask.sum().item(), 1)
        self.assertEqual(batch.inputs.record_relations.tolist(), [[[-1, -1]]])
        batch.inputs.validate(ModelConfig.for_profile(batch.model_profile))
        report = target_coverage_report(admitted, source_kind="contract_fixture")
        self.assertTrue(report["rows"][0]["full_line_encoding_admitted"])
        self.assertEqual(report["rows"][0]["input_profile"], batch.model_profile)

    def test_v2_missing_required_relation_or_cycle_rejected_without_truncation(self):
        row = fixture()
        e = replace(encoding(row), model_profile="full_line_v2", full_line=full_line())
        for relation in ({"parent": None, "parent_required": True}, {"parent": 0}):
            modified = copy.deepcopy(e.full_line)
            modified["records"][0].update(relation)
            with self.assertRaises(ValueError):
                dataset([row], {e.input_sha256: replace(e, full_line=modified)})

    def test_checker_value_namespace_separates_own_raw_and_model_wdl(self):
        value = {"schema": CHECKER_NAMESPACE_SCHEMA, "checker_kind": "own", "checker_binary_sha256": sha("checker"),
                 "profile_sha256": sha("profile"), "search_implementation_sha256": sha("search"),
                 "value_kind": "own_raw", "value_semantics_sha256": sha("raw-value"),
                 "model_configuration_sha256": None, "model_weights_sha256": None, "encoding_sha256": None}
        own = checker_value_namespace(value)
        self.assertFalse(own["learned_utility_admitted"])
        with self.assertRaisesRegex(ValueError, "own checker"):
            checker_value_namespace({**value, "checker_kind": "external_uci"})
        neural = {**value, "checker_kind": "external_uci", "value_kind": "fresh_model_wdl",
                  "model_configuration_sha256": sha("model"), "model_weights_sha256": sha("weights"),
                  "encoding_sha256": sha("encoding")}
        result = checker_value_namespace(neural)
        self.assertFalse(result["legacy_scorer_compatible"])
        with self.assertRaisesRegex(ValueError, "CP/mate"):
            checker_value_namespace({**neural, "value_kind": "external_cp"})

    def test_actual_full_line_context_requires_exact_sidecar_request_and_paths(self):
        row = fixture()
        sidecar = sidecar_v2(row)
        e = encoded_from_sidecar(sidecar, row, expected_encoder_source_sha256=sha("fixture-encoder-source"))
        source = {"cpu_profile_sha256": sha("profile"), "native": {"checker_identity": {"fixture": True}}}
        context = {"domain": "rz-pals-native-full-line-context/1", "input_sha256": e.input_sha256,
                   "sidecar_sha256": sidecar["sha256"], "canonical_tensor_sha256": sidecar["canonical_tensor_sha256"],
                   "process_epoch": 7, "request_sequence": 9, "model_profile": e.model_profile,
                   "model_semantics": "rovezero.pals-model-semantics.v2", "encoding_profile": "rovezero.pals-board-records.v2",
                   "records": [{"moves": [move(12, 28)], "parent": None, "supersedes": None,
                                "parent_required": False, "supersedes_required": False}],
                   "raw_record_relationships": [{"record_id": 1, "revision": 1, "parent_revision": None,
                                                 "supersedes_revision": None, "parent_omitted": False,
                                                 "supersedes_omitted": False}],
                   "relationship_links_omitted": {"observation": "observed", "value": 0},
                   "query_prefix": [], "query_proposal": [move(4, 5)], "query_counter": [move(60, 59)],
                   "namespace": {"value_namespace": "unknown", "reason": "fixture-no-actual-checker"},
                   "checker_identity": source["native"]["checker_identity"], "checker_profile_sha256": sha("profile"),
                   "task_utility_observations": {"observation": "observed", "value": 0},
                   "task_utility_mask_reason": "P/C did not submit V", "sha256": ""}
        parent = {e.input_sha256: {"sidecar": sidecar, "source": source, "request": [7, 9]}}
        raw = json.dumps(seal_struct(context)).encode() + b"\n"
        audit = inspect_full_line_contexts(raw, encoded={e.input_sha256: e}, exact_parents=parent)
        self.assertEqual(audit["raw_observed_contexts"], 1)
        self.assertFalse(audit["learned_scorer_support_admitted"])
        omitted = copy.deepcopy(context)
        omitted["raw_record_relationships"][0].update(parent_revision=99, parent_omitted=True)
        omitted["relationship_links_omitted"]["value"] = 1
        audit = inspect_full_line_contexts(json.dumps(seal_struct(omitted)).encode() + b"\n",
                                          encoded={e.input_sha256: e}, exact_parents=parent)
        self.assertEqual(audit["matched_inputs"][e.input_sha256]["relationship_links_omitted"]["value"], 1)
        for changed, message in ((copy.deepcopy(omitted), "count differs"),
                                 (copy.deepcopy(context), "provenance differs"),
                                 (copy.deepcopy(context), "flag differs")):
            if message == "count differs":
                changed["relationship_links_omitted"]["value"] = 0
            elif message == "provenance differs":
                changed["raw_record_relationships"][0]["revision"] = 2
            else:
                changed["raw_record_relationships"][0]["parent_revision"] = 99
            with self.assertRaisesRegex(ValueError, message):
                inspect_full_line_contexts(json.dumps(seal_struct(changed)).encode() + b"\n",
                                           encoded={e.input_sha256: e}, exact_parents=parent)
        changed = copy.deepcopy(context)
        changed["query_counter"] = [move(60, 58)]
        with self.assertRaisesRegex(ValueError, "packed moves"):
            inspect_full_line_contexts(json.dumps(seal_struct(changed)).encode() + b"\n",
                                       encoded={e.input_sha256: e}, exact_parents=parent)
        changed = copy.deepcopy(context)
        changed["request_sequence"] = 10
        with self.assertRaisesRegex(ValueError, "lacks exact"):
            inspect_full_line_contexts(json.dumps(seal_struct(changed)).encode() + b"\n",
                                       encoded={e.input_sha256: e}, exact_parents=parent)


if __name__ == "__main__":
    unittest.main()
