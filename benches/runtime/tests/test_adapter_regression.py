"""Fail-closed comparison/accumulation checks; no GPU performance test in CI."""
import copy
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter_regression import pair_result, resource_affinity, run, summarize
from bounded_local import OwnedRun, resolve_affinity


def observation(t=100., p=1000):
    return {"accepted": True, "work":{"simulations":8192,"bestmoves":["e2e4","a2a4"]},
            "capture":{"exit_code":0,"forced_cleanup":False,"cleanup_error":None,
                       "remaining_owned_processes":[],"whole_wall_seconds":t,
                       "resources":{"memory.peak":str(p)}}}


class AdapterRegressionTests(unittest.TestCase):
    def test_checkpoint_keeps_historical_peak_separate_from_point_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner=OwnedRun.__new__(OwnedRun)
            owner.group=Path(temporary)
            owner.started=time.monotonic()
            owner.deadline=owner.started+30
            files={"memory.peak":"5000\n", "memory.current":"1000\n",
                   "memory.stat":"anon 700\nfile 200\nkernel 100\n",
                   "cpu.stat":"usage_usec 20\n"}
            for name,value in files.items():
                (owner.group/name).write_text(value)
            checkpoint=owner.resource_checkpoint("model_ready")
            self.assertEqual(checkpoint["resources"],files)
            self.assertIn("not_atomic_or_peak_components",checkpoint["scope"])
            self.assertIn("historical",checkpoint["peak_scope"])
            self.assertTrue(all((owner.group/name).read_text()==value for name,value in files.items()))
            self.assertGreaterEqual(checkpoint["read_wall_seconds"],0)
            for stage in ("", "../foreign", "a"*65, None):
                with self.assertRaises(ValueError):
                    owner.resource_checkpoint(stage)

    def test_supplemental_checkpoint_read_failure_is_visible_without_resetting_peak(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner=OwnedRun.__new__(OwnedRun)
            owner.group=Path(temporary)
            owner.started=time.monotonic()
            owner.deadline=owner.started+30
            (owner.group/"memory.peak").write_text("5000\n")
            checkpoint=owner.resource_checkpoint("model_ready")
            self.assertIn("FileNotFoundError",checkpoint["error"])
            self.assertNotIn("resources",checkpoint)
            self.assertEqual((owner.group/"memory.peak").read_text(),"5000\n")

    def test_invalid_series_budget_cannot_create_output_or_start_a_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"comparison"
            with patch("adapter_regression.prepare_cache") as prepare, \
                    patch("adapter_regression.run_once") as execute:
                for budget in (True,209,3601,3600.0,"3600"):
                    with self.assertRaises(ValueError):
                        run({},output,budget)
                self.assertFalse(output.exists())
                prepare.assert_not_called()
                execute.assert_not_called()

    def test_AA_global_drift_holds_before_AB_even_when_each_AA_pair_passes(self):
        source={"source_commit":"a"*40,"source_patch":None,
                "binary_sha256":"b"*64,"expected_profile":{}}
        manifest={"baseline":source,"candidate":source,
                  "resource_notes":{},"runtime_cache_preparation":{}}
        # Each pair has zero local spread, but the six A runs drift by 8%.
        observations=[observation(t) for t in (100.,100.,104.,104.,108.,108.)]
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"comparison"
            with patch("adapter_regression.resource_affinity",return_value=[0,2]), \
                    patch("adapter_regression.prepare_cache",return_value={"status":"passed"}), \
                    patch("adapter_regression.run_once",side_effect=observations) as execute, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(manifest,output,3550),2)
                self.assertEqual(execute.call_count,6)
                self.assertTrue(all(call.args[0] is source and call.args[3]==[0,2]
                                    for call in execute.call_args_list))
            import json
            registration=json.loads((output/"registration.json").read_text())
            summary=json.loads((output/"summary.json").read_text())
            self.assertEqual(registration["overall_seconds"],3550)
            self.assertEqual(summary["status"],"hold")
            self.assertEqual(summary["pair_count"],3)
            self.assertEqual(summary["AB_totals"]["A_time"],0)
            self.assertIn("AA overall spread",summary["error"])

    def test_declared_cpu_ids_are_applied_without_an_unavailable_fallback(self):
        self.assertEqual(resolve_affinity([0,2],{0,1,2,3}),[0,2])
        self.assertEqual(resolve_affinity(None,{0,1,2,3}),[0,1])
        for requested in ([0,0],[0,9],[False,1],[0],[0,1,2],"0,2"):
            with self.assertRaises(ValueError):
                resolve_affinity(requested,{0,1,2,3})

    def test_registration_cannot_misstate_applied_cpu_memory_or_input_policy(self):
        notes=dict(cpu_affinity=[0,2],memory_high_GiB=6,memory_max_GiB=12,swap=0,
                   precision="fp32",tf32=False,history_fill="no",batch=1)
        self.assertEqual(resource_affinity(notes,{0,1,2,3}),[0,2])
        for key,value in (("cpu_affinity",[2,0]),("memory_high_GiB",7),("memory_max_GiB",13),
                          ("swap",1),("precision","fp16"),("tf32",True),("history_fill","repeat"),
                          ("batch",4),("batch",True)):
            changed={**notes,key:value}
            with self.assertRaises(ValueError):
                resource_affinity(changed,{0,1,2,3})

    def test_both_time_and_peak_are_per_pair_gates(self):
        self.assertTrue(pair_result(observation(), observation(105.,1050), "AB")["passed"])
        self.assertFalse(pair_result(observation(), observation(105.01,800), "AB")["passed"])
        self.assertFalse(pair_result(observation(), observation(80.,1051), "AB")["passed"])

    def test_AA_noise_is_symmetric_and_never_an_optimization(self):
        self.assertFalse(pair_result(observation(), observation(90.,900), "AA")["passed"])
        self.assertTrue(pair_result(observation(), observation(100.,1000), "AA")["passed"])

    def test_missing_peak_failure_or_different_work_is_rejected(self):
        for mutation in (lambda o:o.update(accepted=False),
                         lambda o:o["capture"].update(forced_cleanup=True),
                         lambda o:o["capture"]["resources"].update({"memory.peak":"0"}),
                         lambda o:o["work"].update(simulations=8191)):
            changed=observation()
            mutation(changed)
            with self.assertRaises(ValueError):
                pair_result(observation(), changed, "AB")

    def test_successful_diagnostic_is_rejected_from_both_comparison_arms(self):
        for fields in ({"diagnostic_only":True}, {"performance_measurement":False},
                       {"diagnostic_only":1}, {"performance_measurement":1}):
            diagnostic={**observation(), **fields}
            for left,right in ((diagnostic,observation()),(observation(),diagnostic)):
                with self.subTest(fields=fields), self.assertRaisesRegex(ValueError,"diagnostic evidence"):
                    pair_result(left,right,"AB")

    def test_sum_never_overrides_failed_pair_or_missing_runs(self):
        aa=pair_result(observation(),observation(),"AA")
        ab=pair_result(observation(),observation(99.,990),"AB")
        pairs=[copy.deepcopy(aa) for _ in range(3)]+[copy.deepcopy(ab) for _ in range(5)]
        self.assertEqual(summarize(pairs)["status"],"passed")
        self.assertEqual(summarize(pairs)["AB_totals"]["A_time"],500.)
        pairs[-1]=pair_result(observation(),observation(106.,1000),"AB")
        self.assertLess(summarize(pairs)["AB_time_ratio_of_sums"],1.05)
        self.assertEqual(summarize(pairs)["status"],"hold")
        self.assertEqual(summarize(pairs[:-1])["status"],"hold")


if __name__ == "__main__":
    unittest.main()
