"""Controller/fault tests. The subprocess is a protocol simulator, not an engine."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

SPEC = importlib.util.spec_from_file_location("paired_search", Path(__file__).parents[1] / "paired_search.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class PairedSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        variant = {"binary": sys.executable, "binary_sha256": runner.sha256(sys.executable),
                   "source_commit": 'a' * 40, "build_features": [], "compiler": "test-controller",
                   "args": ["--cpu-mock"]}
        return {"schema": runner.SCHEMA, "provider": "cpu_mock", "seed": 71, "profile_mode": "off", "primary_metric": "position_bestmove_ns",
                "witness_sha256": 'b' * 64, "timeout_ms": 1500, "max_rss_mib": 64,
                "max_throttled_usec": 10**12, "run_wall_limit_ms": 10000,
                "max_run_bytes": 16 * 1024 * 1024, "cpu_affinity": [],
                "baseline": variant, "candidate": copy.deepcopy(variant),
                "resource_notes": {"host_exclusive": "unknown"},
                "fixtures": [{"name": "start", "position": "position startpos",
                              "go": "go nodes 1", "legal_moves": ["a2a3"], "terminal": False}]}

    def process(self, response, quit_response="", diagnostic=""):
        # Intentionally bypass manifest argv validation for the isolated controller test.
        code = "\n".join([
            "import sys", "for line in sys.stdin:", " command = line.strip()",
            " if command == 'uci': print('uciok', flush=True)",
            " elif command == 'isready': print('readyok', flush=True)",
            " elif command.startswith('go '):",
            f"  print({response!r}, flush=True)" if response else "  pass",
            f"  print({diagnostic!r}, file=sys.stderr, flush=True)" if diagnostic else "  pass",
            " elif command == 'quit':",
            f"  print({quit_response!r}, flush=True)" if quit_response else "  pass", "  break",
        ])
        manifest = self.manifest()
        manifest["baseline"]["args"] = ["-u", "-c", code]
        return runner.run_process(manifest, "baseline", manifest["fixtures"][0], self.root / "run")

    def test_locked_manifest_hash_and_injection_guards(self):
        manifest = self.manifest()
        runner.validate(manifest)
        for field, value in [("binary_sha256", '0' * 64), ("source_commit", "short"), ("args", ["--onnx-cpu"])]:
            bad = copy.deepcopy(manifest)
            bad["baseline"][field] = value
            with self.assertRaises(ValueError):
                runner.validate(bad)
        manifest["fixtures"][0]["position"] = "position startpos\nquit"
        with self.assertRaises(ValueError):
            runner.validate(manifest)

    def test_duplicate_json_keys_are_rejected(self):
        path = self.root / "manifest.json"
        path.write_text('{"schema":1,"schema":2}')
        with self.assertRaises(ValueError):
            runner.read_json(path)

    def test_serial_orders_are_repeatable_and_balanced(self):
        patterns = runner.order(71, 2, 4)
        self.assertEqual(patterns, runner.order(71, 2, 4))
        self.assertEqual(patterns.count("ABBA"), 2)
        self.assertEqual(patterns.count("BAAB"), 2)
        self.assertTrue(all(p.count('A') == p.count('B') == 2 for p in runner.order(71, 1, 3)))

    def test_legal_completion_uses_real_wall_and_does_not_invent_nodes(self):
        result = self.process("bestmove a2a3")
        self.assertEqual(result["classification"], "diagnostic")
        self.assertGreater(result["go_bestmove_ns"], 0)
        self.assertIsNone(result["reported_nodes"])
        self.assertFalse(result["formal_acceptance"])

    def test_illegal_output_is_preserved_as_product_failure(self):
        result = self.process("bestmove a7a6")
        self.assertEqual(result["classification"], "product_failure")
        self.assertIn("bestmove differs", result["error"])
        self.assertIn("bestmove a7a6", (self.root / "run/stdout.log").read_text())

    def test_late_duplicate_cannot_hide_after_exit(self):
        result = self.process("bestmove a2a3", "bestmove a2a3")
        self.assertEqual(result["classification"], "product_failure")
        self.assertIn("duplicate bestmove", result["error"])

    def test_diagnostic_input_failure_cannot_be_counted_as_success(self):
        result = self.process("bestmove a2a3", diagnostic="PositionRejected: bad trace")
        self.assertEqual(result["classification"], "product_failure")
        self.assertIn("diagnostic stderr", result["error"])

    def test_timeout_kills_only_child_and_retains_output(self):
        began = time.monotonic()
        result = self.process("")
        self.assertEqual(result["classification"], "product_failure")
        self.assertIn("timeout", result["error"])
        self.assertLess(time.monotonic() - began, 5)
        self.assertIsNotNone(result["exit_code"])
        self.assertTrue((self.root / "run/stdout.log").exists())

    def test_reader_rejects_unbounded_line(self):
        reader = runner.Reader(io.BytesIO(b'x' * (runner.MAX_LINE + 1) + b'\n'), self.root / "log", True)
        reader.thread.join(timeout=2)
        self.assertFalse(reader.thread.is_alive())
        self.assertIn("line exceeds cap", reader.error)

    def test_native_receipt_binding_and_drain_are_checked(self):
        variant = self.manifest()["baseline"]
        variant["expected_profile"] = {"backend_sha256": 'a' * 64,
            "model_manifest_sha256": 'b' * 64, "encoding_manifest_sha256": 'c' * 64}
        slot = self.root / "native/slot"
        slot.mkdir(parents=True)
        profile = {"provider": "cpu", "precision": "fp32", "max_batch_items": 1,
            "intra_threads": 1, "max_workers": 1, "full_steps": 1, "min_steps": 1,
            "max_steps": 1, "require_full": True, "contract_major": 0, "contract_minor": 1,
            "backend_sha256": 'a' * 64, "model": {"manifest_sha256": 'b' * 64},
            "encoding": {"manifest_sha256": 'c' * 64}}
        startup = {"schema_version": 1, "kind": "startup", "loaded": True, "process_id": 17,
            "process_run_id": "slot", "executable": {"sha256": variant["binary_sha256"]}, "profile": profile}
        termination = {"schema_version": 1, "kind": "termination", "process_run_id": "slot",
            "startup": startup, "run_succeeded": True, "physical_drain": "confirmed",
            "actual_cpu_inference_observed": False, "report": {"origin": "cpu_onnx",
                "backend_sha256": 'a' * 64, "model_manifest_sha256": 'b' * 64}}
        (slot / "native-cpu-startup.v1.json").write_text(json.dumps(startup))
        path = slot / "native-cpu-termination.v1.json"
        path.write_text(json.dumps(termination))
        result = runner.native_receipts(self.root, variant, "onnx_cpu", 17)
        self.assertFalse(result["actual_inference_observed"])
        with self.assertRaises(ValueError):
            runner.native_receipts(self.root, variant, "onnx_cpu", 18)
        termination["physical_drain"] = "unconfirmed"
        path.write_text(json.dumps(termination))
        with self.assertRaises(ValueError):
            runner.native_receipts(self.root, variant, "onnx_cpu", 17)

    def test_cuda_requires_executed_nodes_and_verified_mapping(self):
        # A CPU startup cannot satisfy the CUDA receipt path.
        variant = self.manifest()["baseline"]
        variant["expected_profile"] = {"backend_sha256": 'a' * 64,
            "model_manifest_sha256": 'b' * 64, "encoding_manifest_sha256": 'c' * 64}
        slot = self.root / "native/slot"
        slot.mkdir(parents=True)
        startup = {"schema_version": 1, "kind": "startup", "loaded": True, "process_id": 17,
            "process_run_id": "slot", "executable": {"sha256": variant["binary_sha256"]},
            "profile": {"provider": "cuda", "precision": "fp32", "max_batch_items": 1,
                "intra_threads": 1, "max_workers": 1, "full_steps": 1, "min_steps": 1,
                "max_steps": 1, "require_full": True, "contract_major": 0, "contract_minor": 1,
                "runtime_mapping_verified": True, "executed_cuda_nodes": 0}}
        (slot / "native-cuda-startup.v1.json").write_text(json.dumps(startup))
        (slot / "native-cuda-termination.v1.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "execution/mapping"):
            runner.native_receipts(self.root, variant, "onnx_cuda", 17)


if __name__ == "__main__":
    unittest.main()
