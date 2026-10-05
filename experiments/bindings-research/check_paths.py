"""Correctness only; does not aggregate or report performance measurements."""
import concurrent.futures
import json
import os
import subprocess
import unittest

from cases import CASES
import rz_bindings_poc

CLI = os.environ["RZ_BINDINGS_CLI"]

class PathChecks(unittest.TestCase):
    def test_all_three_paths_and_repeat_concurrency(self):
        process = subprocess.Popen([CLI], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertTrue(json.loads(process.stdout.readline())["ready"])
            for position in CASES.values():
                expected = rz_bindings_poc.search_once(position, 8)["result"]
                direct = subprocess.run([CLI, "--direct", position, "1", "8"],
                                        check=True, text=True, capture_output=True, timeout=10)
                self.assertEqual(expected, json.loads(direct.stdout)["measurement"]["result"])
                for _ in range(2):
                    process.stdin.write(json.dumps({"position_command":position,"simulation_limit":8}) + "\n")
                    process.stdin.flush()
                    self.assertEqual(expected, json.loads(process.stdout.readline())["measurement"]["result"])
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    outputs = list(pool.map(lambda _: rz_bindings_poc.search_once(position, 8)["result"], range(2)))
                self.assertEqual(outputs, [expected, expected])
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
            process.stdout.close()
        self.assertEqual(process.returncode, 0)

    def test_invalid_inputs_and_budget_are_explicit(self):
        for position, limit in [("go nodes 1",8),("position startpos moves e2e5",8),
                                ("position fen broken",8),("position startpos",0),("position startpos",129)]:
            with self.assertRaises(ValueError):
                rz_bindings_poc.search_once(position, limit)

if __name__ == "__main__":
    unittest.main()
