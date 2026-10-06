"""Accounting failures must not turn repeated/unsafe runs into improvements."""
import copy
import hashlib
import importlib.util
import json
import os
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

    def test_cuda_cpu_arena_requires_actual_explicit_option_and_no_fallback(self):
        result = self.pin(dict(accepted=True, exit_code=0, conditioning=False,
            forced_cgroup_cleanup=False, remaining_after_cleanup=[], source_commit="source0",
            elapsed_s=10, cgroup={"memory.events":"max 0\noom 0\noom_kill 0\n"}))
        report = dict(accepted=True, failure=None, kind="b1_cuda_cpu_arena_fixed_work",
            disable_cuda_cpu_arena=True, cpu_ep_fallback=False, reuse_buffers=False,
            batches=[1], warmup_runs=3, measured_runs=20,
            samples=[dict(completed_nn_items=20, physical_completion_confirmed=True)],
            memory={"self_vm_hwm":"100 kB"}, input_sha256="input", output_sha256="output",
            model_sha256="model", export_manifest_sha256="manifest", runtime_bundle_sha256="bundle",
            precision="fp32", history_fill="no", tf32=False, cache=False, dedup=False,
            io_binding=False, cuda_graph=False)
        item = dict(layout="inference", result=result, report=self.pin(report),
            reuse_buffers=False, disable_cuda_cpu_arena=True)
        self.assertEqual(module.observation(self.root, item, "process_vm_hwm_bytes")["peak_bytes"], 102400)
        for changes in ({"disable_cuda_cpu_arena":False}, {"cpu_ep_fallback":True}, {"reuse_buffers":True}):
            invalid = dict(item, report=self.pin(dict(report, **changes)))
            with self.assertRaises(ValueError):
                module.observation(self.root, invalid, "process_vm_hwm_bytes")
        for value in (False, 1, None):
            invalid = dict(item, disable_cuda_cpu_arena=value)
            with self.assertRaises(ValueError):
                module.observation(self.root, invalid, "process_vm_hwm_bytes")

    def test_identical_comparison_is_counted_once(self):
        pair = self.pair("p1")
        result = self.aggregate([pair, copy.deepcopy(pair)])
        self.assertEqual(result["duplicate_comparisons_skipped"], 1)
        self.assertEqual(result["unique_compared_physical_runs"], 2)

    def test_owned_ort_checks_actual_provenance_retention_and_independent_flags(self):
        result=self.pin(dict(accepted=True,exit_code=0,conditioning=False,
            forced_cgroup_cleanup=False,remaining_after_cleanup=[],source_commit="source0",
            elapsed_s=10,cgroup={"memory.events":"max 0\noom 0\noom_kill 0\n"}))
        derived=self.pin(dict(schema=1,source_onnx_sha256="model",source_export_manifest_sha256="manifest",
            ort_sha256="derived",ort_bytes=80,runtime_core_sha256="core",runtime_bundle_sha256="bundle",
            runtime_version="1.22.0",optimization_level=1,provider="cuda",precision="fp32",tf32=False,redistribution_ready=False))
        report=dict(accepted=True,failure=None,kind="b1_owned_ort_fixed_work",zero_copy_ort=True,
            cpu_ep_fallback=False,reuse_buffers=False,disable_cuda_cpu_arena=False,
            batches=[1],warmup_runs=3,measured_runs=20,
            samples=[dict(completed_nn_items=20,physical_completion_confirmed=True)],memory={"self_vm_hwm":"100 kB"},
            input_sha256="input",output_sha256="output",model_sha256="model",export_manifest_sha256="manifest",
            runtime_bundle_sha256="bundle",precision="fp32",history_fill="no",tf32=False,cache=False,dedup=False,
            io_binding=False,cuda_graph=False,serialized_model_sha256="derived",derived_manifest_sha256=derived["sha256"],retained_model_bytes=80)
        item=dict(layout="inference",result=result,report=self.pin(report),reuse_buffers=False,zero_copy_ort=True,derived_manifest=derived)
        self.assertEqual(module.observation(self.root,item,"process_vm_hwm_bytes")["peak_bytes"],102400)
        for changes in ({"zero_copy_ort":False},{"cpu_ep_fallback":True},{"disable_cuda_cpu_arena":True},
            {"reuse_buffers":True},{"retained_model_bytes":0},{"serialized_model_sha256":"other"},
            {"derived_manifest_sha256":"other"},{"model_sha256":"other"}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                module.observation(self.root,dict(item,report=self.pin(dict(report,**changes))),"process_vm_hwm_bytes")

    def test_ort_comparison_rejects_two_baselines_and_reversed_arms(self):
        for a,b in ((False,False),(True,False),(True,True)):
            pair=self.pair("ort",option="owned-ort-flatbuffer")
            pair["baseline"].update(layout="inference",zero_copy_ort=a)
            pair["variant"].update(layout="inference",zero_copy_ort=b)
            with self.assertRaisesRegex(ValueError,"requires original baseline and direct variant"):
                self.aggregate([pair])

    def test_arena_comparison_cannot_label_two_baselines_or_reversed_options(self):
        for baseline, variant in ((False, False), (True, False), (True, True)):
            pair = self.pair("arena", option="cuda-cpu-arena-off")
            pair["baseline"].update(layout="inference", disable_cuda_cpu_arena=baseline)
            pair["variant"].update(layout="inference", disable_cuda_cpu_arena=variant)
            with self.assertRaisesRegex(ValueError, "requires baseline on and variant off"):
                self.aggregate([pair])

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

    def test_legacy_source_requires_pinned_registration_and_cannot_conflict(self):
        pair = self.pair("p", a=self.measurement(10, 100, source_commit=None))
        with self.assertRaisesRegex(ValueError, "source registration required"):
            self.aggregate([pair])
        pair["baseline"]["registration"] = self.pin(dict(source_commit="source0"))
        self.assertEqual(len(self.aggregate([pair])["series"][0]["epochs"]), 1)
        pair["variant"]["registration"] = self.pin(dict(source_commit="source1"))
        with self.assertRaisesRegex(ValueError, "raw source differs"):
            self.aggregate([pair])

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

    def test_repeated_exclusion_is_preserved_once_and_conflicts_rejected(self):
        item = dict(id="failure", reason="failed", evidence=[self.pin(dict(accepted=False))])
        result = self.aggregate([self.pair("p")], excluded=[item, copy.deepcopy(item)])
        self.assertEqual(len(result["excluded"]), 1)
        conflict = copy.deepcopy(item)
        conflict["reason"] = "different failure"
        with self.assertRaisesRegex(ValueError, "conflicting exclusion"):
            self.aggregate([self.pair("p2")], excluded=[item, conflict])

    def test_duplicate_json_keys_and_secret_names_rejected(self):
        path = self.root / "ambiguous.json"
        path.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            module.read_json(path)
        with self.assertRaisesRegex(ValueError, "non-secret JSON"):
            module.read_json(self.root / "service-account.json")

    def test_non_regular_json_cannot_block_the_reader(self):
        directory = self.root / "directory.json"
        directory.mkdir()
        with self.assertRaisesRegex(ValueError, "regular JSON"):
            module.read_json(directory)
        if hasattr(os, "mkfifo"):
            fifo = self.root / "pipe.json"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "regular JSON"):
                module.read_json(fifo)

    def test_included_ledgers_preserve_relative_evidence_and_deduplicate_comparisons(self):
        pair = self.pair("p")
        ledger = dict(schema_version=1, comparisons=[pair])
        first = self.root / "first.json"
        second = self.root / "second.json"
        first.write_text(json.dumps(ledger))
        second.write_text(json.dumps(ledger))
        combined, inputs = module.combine_ledgers([first, second])
        self.assertEqual(len(inputs), 2)
        self.assertTrue(Path(combined["comparisons"][0]["baseline"]["result"]["path"]).is_absolute())
        result = module.aggregate(self.root / "unrelated", combined)
        self.assertEqual(result["duplicate_comparisons_skipped"], 1)
        self.assertEqual(result["series"][0]["cumulative"]["pairs"], 1)


if __name__ == "__main__":
    unittest.main()
