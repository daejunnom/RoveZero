"""Fail-closed comparison/accumulation checks; no GPU performance test in CI."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter_regression import pair_result, summarize


def observation(t=100., p=1000):
    return {"accepted": True, "work":{"simulations":8192,"bestmoves":["e2e4","a2a4"]},
            "capture":{"exit_code":0,"forced_cleanup":False,"cleanup_error":None,
                       "remaining_owned_processes":[],"whole_wall_seconds":t,
                       "resources":{"memory.peak":str(p)}}}


class AdapterRegressionTests(unittest.TestCase):
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
