"""Accounting failures must not turn repeated/unsafe runs into improvements."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("memory_evidence", Path(__file__).parents[1] / "memory_evidence.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Accounting(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0

    def pin(self, value):
        self.serial += 1
        name = f"evidence-{self.serial}.json"
        payload = json.dumps(value).encode()
        (self.root / name).write_bytes(payload)
        return dict(path=name, sha256=hashlib.sha256(payload).hexdigest())

    def measurement(self, seconds, peak, **changes):
        value = dict(accepted=True, exit_code=0, conditioning=False,
            forced_cgroup_cleanup=False, remaining_owned_processes=[], source_commit="source0",
            whole_wall_seconds=seconds, fixed_work={"completed": 16, "consumed": 16},
            resources={"memory.events": "high 0\nmax 0\noom 0\noom_kill 0\n", "memory.peak": str(peak)})
        value.update(changes)
        return dict(layout="native", result=self.pin(value), receipt=self.pin(dict(
            integration_checks_passed=True, cleanup_verified=True, unresolved_owner_retained=False,
            incomplete_games=0, provider_sessions=[{}, {}, {}, {}])))

    def pair(self, identifier, a=None, b=None, option="reclaim"):
        return dict(id=identifier, baseline=a or self.measurement(10, 100), variant=b or self.measurement(11, 70),
            environment_epoch="fixture", compatibility=dict(option=option, workload="three-ply",
                model="model", runtime="runtime", precision="fp32", batch=1, resources="fixed",
                fixed_work="16", time_scope="whole", peak_kind="cgroup_memory_peak_bytes", other_options="off"))

    def aggregate(self, pairs, **extra):
        return module.aggregate(self.root, dict(schema_version=1, comparisons=pairs, **extra))

    def test_totals_retain_epochs_and_compute_mean_peak(self):
        pair = self.pair("p2", self.measurement(20, 200, source_commit="source1"), self.measurement(18, 160, source_commit="source1"))
        result = self.aggregate([self.pair("p1"), pair])["series"][0]
        self.assertEqual(len(result["epochs"]), 2)
        total = result["cumulative"]
        self.assertEqual(total["baseline"]["time_seconds"], 30)
        self.assertEqual(total["baseline"]["peak_observation_sum_bytes"], 300)
        self.assertEqual(total["baseline"]["mean_run_peak_bytes"], 150)
        self.assertAlmostEqual(total["cumulative_time_ratio"], 29 / 30)
        self.assertAlmostEqual(total["cumulative_peak_ratio"], 230 / 300)

    def test_identical_comparison_is_counted_once(self):
        pair = self.pair("p1")
        result = self.aggregate([pair, copy.deepcopy(pair)])
        self.assertEqual(result["duplicate_comparisons_skipped"], 1)
        self.assertEqual(result["unique_compared_physical_runs"], 2)

    def test_conflicting_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting comparison"):
            self.aggregate([self.pair("same"), self.pair("same", b=self.measurement(9, 80))])

    def test_renaming_pair_cannot_double_count_run(self):
        pair = self.pair("p1")
        duplicate = copy.deepcopy(pair)
        duplicate["id"] = "p2"
        with self.assertRaisesRegex(ValueError, "physical run reused"):
            self.aggregate([pair, duplicate])

    def test_shared_control_across_options_has_one_physical_identity(self):
        control = self.measurement(10, 100)
        result = self.aggregate([self.pair("r", a=control), self.pair("i", a=control, b=self.measurement(8, 90), option="io")])
        self.assertEqual(len(result["series"]), 2)
        self.assertEqual(result["unique_compared_physical_runs"], 3)
        self.assertEqual(result["shared_control_occurrences"], 1)

    def test_tampered_raw_evidence_is_rejected(self):
        pair = self.pair("p1")
        (self.root / pair["baseline"]["result"]["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.aggregate([pair])

    def test_failed_conditioning_forced_and_missing_exit_rejected(self):
        for changes in (dict(accepted=False), dict(conditioning=True), dict(forced_cgroup_cleanup=True),
                        dict(exit_code=None), dict(remaining_owned_processes=["42"])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.aggregate([self.pair("p", a=self.measurement(10, 100, **changes))])

    def test_oom_and_missing_peak_cannot_be_averaged_away(self):
        for resources in ({"memory.events": "oom 1\n", "memory.peak": "100"},
                          {"memory.events": "oom 0\n", "memory.peak": "0"}):
            with self.subTest(resources=resources), self.assertRaises(ValueError):
                self.aggregate([self.pair("p", a=self.measurement(10, 100, resources=resources))])

    def test_workload_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "completed work"):
            self.aggregate([self.pair("p", b=self.measurement(10, 80, fixed_work={"completed": 17}))])

    def test_peak_definition_is_not_substituted(self):
        pair = self.pair("p")
        pair["compatibility"]["peak_kind"] = "process_vm_hwm_bytes"
        with self.assertRaisesRegex(ValueError, "registered whole-group peak"):
            self.aggregate([pair])

    def test_nonfinite_time_rejected(self):
        for value in (float("nan"), float("inf"), 0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.aggregate([self.pair("p", a=self.measurement(value, 100))])

    def test_excluded_failure_retained_without_affecting_totals(self):
        ref = self.pin(dict(accepted=False, stage="before_gpu"))
        result = self.aggregate([self.pair("p")], excluded=[dict(id="failed", reason="before GPU", evidence=[ref])])
        self.assertEqual(len(result["excluded"]), 1)
        self.assertEqual(result["series"][0]["cumulative"]["pairs"], 1)

    def test_duplicate_json_keys_and_secret_names_rejected(self):
        path = self.root / "ambiguous.json"
        path.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            module.read_json(path)
        with self.assertRaisesRegex(ValueError, "non-secret JSON"):
            module.read_json(self.root / "service-account.json")


if __name__ == "__main__":
    unittest.main()
