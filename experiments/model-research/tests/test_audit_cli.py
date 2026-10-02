from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from rz_data.audit import audit_dataset
from rz_data.cli import main
from rz_data.errors import DataError
from rz_data.io import Limits, read_json, read_jsonl, write_run
from rz_data.serialization import canonical_bytes, loads
from rz_data.splits import make_split_plan

PACKAGE = Path(__file__).resolve().parents[1]
FIXTURES = PACKAGE / "fixtures"
SOURCE_ROOT = PACKAGE.parents[1]


def fixture(name):
    return loads((FIXTURES / name).read_bytes())


class AuditCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="rz-f01-tests-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = fixture("manifest.json")
        self.record = fixture("record.json")
        self.sources = fixture("sources.json")
        self.plan = make_split_plan(self.sources["sources"], seed="fixture-demo")
        self.record["split"] = self.plan["assignments"][self.record["game_id"]]
        self.rows = self.root / "rows.jsonl"

    def write_rows(self, records):
        self.rows.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in records))

    def audit(self, records):
        self.write_rows(records)
        return audit_dataset(self.manifest, self.rows, self.plan, Limits())

    def test_success_receipt_hashes_exact_bytes_and_keeps_bindings_unrun(self):
        report = self.audit([self.record])
        self.assertTrue(report["structural_audit_passed"])
        self.assertEqual(report["records_file_digest"], hashlib.sha256(self.rows.read_bytes()).hexdigest())
        self.assertFalse(report["execution_ready"])
        self.assertEqual(report["training_eligible_records"], 0)
        self.assertEqual(report["coverage"]["state_restore"], "not_run")
        self.assertEqual(report["counts"]["complete_label_candidates"], 1)

    def test_failed_labels_are_excluded_without_synthetic_targets(self):
        self.record["label"].update(status="failed", failure_reason="teacher timeout", policy=None, value=None)
        report = self.audit([self.record])
        self.assertTrue(report["structural_audit_passed"])
        self.assertEqual(report["counts"]["excluded_labels"], 1)
        self.assertEqual(report["counts"]["complete_label_candidates"], 0)
        self.assertEqual(report["exclusions"][0]["game_id"], self.record["game_id"])
        self.assertIsNone(self.record["label"]["value"])

    def test_invalid_labels_retain_source_ids_and_reason(self):
        self.record["label"]["policy"]["values"] = [0, 0, 0, 0]
        report = self.audit([self.record])
        self.assertFalse(report["structural_audit_passed"])
        self.assertEqual(report["counts"]["rejected_records"], 1)
        self.assertEqual(report["rejections"][0]["record_id"], self.record["record_id"])
        self.assertEqual(report["rejections"][0]["lineage_id"], self.record["lineage_id"])
        self.assertEqual(report["rejections"][0]["opening_family_id"], self.record["opening_family_id"])
        self.assertEqual(report["coverage"]["rejected_records_outside_leakage_audit"], 1)
        self.assertEqual(report["coverage"]["leakage_scope"], "structurally_valid_records_only")
        self.assertEqual(report["rejections"][0]["error"]["code"], "InvalidNormalization")

    def test_json_rejections_are_not_silently_dropped(self):
        self.rows.write_bytes(canonical_bytes(self.record) + b'\n{"x":1,"x":2}\n\n')
        report = audit_dataset(self.manifest, self.rows, self.plan, Limits())
        self.assertFalse(report["structural_audit_passed"])
        self.assertEqual(report["counts"]["input_records"], 3)
        self.assertEqual([r["line"] for r in report["rejections"]], [2, 3])

    def test_unconfirmed_rights_block_structural_audit(self):
        self.manifest["source"]["rights"]["status"] = "unverified"
        report = self.audit([self.record])
        self.assertFalse(report["structural_audit_passed"])
        self.assertIn("RightsUnverified", [e["code"] for e in report["errors"]])

    def test_record_budget_and_empty_input_are_visible(self):
        report = self.audit([])
        self.assertIn("EmptyDataset", [e["code"] for e in report["errors"]])
        self.manifest["teacher"]["budget"]["max_positions"] = 1
        other = deepcopy(self.record)
        other["record_id"] = "fixture-row-2"
        report = self.audit([self.record, other])
        self.assertIn("BudgetExceeded", [e["code"] for e in report["errors"]])

    def test_cross_split_duplicate_blocks_audit(self):
        other = deepcopy(self.record)
        other.update(record_id="row-other", game_id="game-other", opening_family_id=None,
                     lineage_id=None, seed_group_id=None)
        sources = self.sources["sources"] + [{key: other[key] for key in self.sources["sources"][0]}]
        for seed in range(100):
            plan = make_split_plan(sources, seed=str(seed))
            if len(set(plan["assignments"].values())) > 1:
                self.plan = plan
                break
        else:
            self.fail("fixture seed search found no distinct splits")
        for row in (self.record, other):
            row["split"] = self.plan["assignments"][row["game_id"]]
        report = self.audit([self.record, other])
        self.assertFalse(report["structural_audit_passed"])
        self.assertIn("ExactInputAcrossSplits", [e["code"] for e in report["errors"]])

    def test_stream_record_line_and_file_limits_abort(self):
        self.write_rows([self.record, self.record])
        for limits in (Limits(max_records=1), Limits(max_record_bytes=10), Limits(max_file_bytes=10)):
            with self.subTest(limits=limits), self.assertRaises(DataError):
                list(read_jsonl(self.rows, limits))
        with self.assertRaises(DataError):
            read_json(self.rows, max_bytes=10)
        with self.assertRaises(DataError):
            Limits(max_records=True)

    def test_run_outputs_are_immutable_and_outside_source(self):
        run = write_run(self.root / "outputs", "audit-one", {"audit.json": {"ok": True}}, source_root=SOURCE_ROOT)
        self.assertEqual(loads((run / "audit.json").read_bytes()), {"ok": True})
        with self.assertRaises(DataError) as caught:
            write_run(self.root / "outputs", "audit-one", {"audit.json": {}}, source_root=SOURCE_ROOT)
        self.assertEqual(caught.exception.code, "RunExists")
        with self.assertRaises(DataError) as caught:
            write_run(PACKAGE / "outputs", "bad", {}, source_root=SOURCE_ROOT)
        self.assertEqual(caught.exception.code, "OutputInSource")
        with self.assertRaises(DataError):
            write_run(self.root, "../escape", {}, source_root=SOURCE_ROOT)

    def test_source_symlink_output_is_rejected(self):
        link = self.root / "source-link"
        link.symlink_to(SOURCE_ROOT, target_is_directory=True)
        with self.assertRaises(DataError) as caught:
            write_run(link / "out", "bad", {}, source_root=SOURCE_ROOT)
        self.assertEqual(caught.exception.code, "OutputInSource")

    def call(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main(args)
        return status, out.getvalue(), err.getvalue()

    def test_cli_split_then_validate_and_no_overwrite(self):
        sources = self.root / "sources.json"
        sources.write_bytes(canonical_bytes(self.sources))
        common = ["--output-root", str(self.root / "outputs")]
        status, out, err = self.call(["split", "--sources", str(sources), "--seed", "fixture-demo", "--run-id", "plan", *common])
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(loads(out)["split_plan_digest"], self.plan["digest"])
        manifest = self.root / "manifest.json"
        manifest.write_bytes(canonical_bytes(self.manifest))
        self.write_rows([self.record])
        args = ["validate", "--manifest", str(manifest), "--records", str(self.rows), "--split-plan",
                str(self.root / "outputs" / "plan" / "split-plan.json"), "--run-id", "audit", *common]
        status, out, err = self.call(args)
        self.assertEqual((status, err), (0, ""))
        self.assertFalse(loads(out)["execution_ready"])
        status, out, err = self.call(args)
        self.assertEqual(status, 2)
        self.assertEqual(loads(err)["error"]["code"], "RunExists")

    def test_cli_unsafe_or_invalid_inputs_return_typed_errors(self):
        source = self.root / "sources.json"
        source.write_text('{"schema_version":1,"sources":[]}', encoding="utf-8")
        status, _, err = self.call(["split", "--sources", str(source), "--seed", "x",
                                   "--output-root", str(self.root / "out"), "--run-id", "run"])
        self.assertEqual(status, 2)
        self.assertEqual(loads(err)["error"]["code"], "EmptySources")

    def test_declared_contract_revision_cannot_enable_training(self):
        self.manifest["engine_contract_revision"] = "unverified-claim-v999"
        report = self.audit([self.record])
        self.assertFalse(report["execution_ready"])
        self.assertEqual(report["coverage"]["engine_contract_binding"], "not_run")

    def test_report_size_budget_rejects_before_creating_run(self):
        with self.assertRaises(DataError) as caught:
            write_run(self.root / "outputs", "too-big", {"audit.json": {"data": "x" * 1000}},
                      source_root=SOURCE_ROOT, max_bytes=50)
        self.assertEqual(caught.exception.code, "OutputByteLimit")
        self.assertFalse((self.root / "outputs" / "too-big").exists())
        self.write_rows([self.record])
        with self.assertRaises(DataError) as caught:
            audit_dataset(self.manifest, self.rows, self.plan, Limits(max_output_bytes=50))
        self.assertEqual(caught.exception.code, "OutputByteLimit")

    def test_python_manifest_boundary_rejects_nested_nonfinite_options(self):
        self.manifest["teacher"]["options"] = {"nested": {"bad": float("nan")}}
        with self.assertRaises(DataError):
            self.audit([self.record])

    def test_invalid_identity_values_have_bounded_digest_ledger(self):
        self.record["lineage_id"] = "x" * 50000
        report = self.audit([self.record])
        saved = report["rejections"][0]["lineage_id"]
        self.assertEqual(len(saved["invalid_field_digest"]), 64)
        self.assertLess(len(canonical_bytes(report)), 10000)

    def test_cli_io_errors_preserve_cause_and_stage_without_paths(self):
        missing = self.root / "private-missing-sources.json"
        status, _, err = self.call(["split", "--sources", str(missing), "--seed", "x",
                                   "--output-root", str(self.root / "out"), "--run-id", "run"])
        self.assertEqual(status, 2)
        error = loads(err)["error"]
        self.assertEqual(error["type"], "FileNotFoundError")
        self.assertEqual(error["errno"], 2)
        self.assertEqual(error["context"], "read_sources")
        self.assertNotIn(str(missing), err)


if __name__ == "__main__":
    unittest.main()
