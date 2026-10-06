"""Prior success for a different model/runtime/opponent must not authorize play."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_model_pair import gate_identities


def fixture():
    artifacts={k:{"sha256":k} for k in ("onnx","ort_library","export_manifest")}
    runtime={"expected_backend_sha256":"backend","expected_encoding_sha256":"encoding"}
    launch={"profile":{"runtime":runtime},"cuda_bundle":{"canonical_sha256":"bundle"},
            "artifacts":[{"role":k,"artifact":v} for k,v in artifacts.items()]}
    rove={"tool":{"binary":{"sha256":"engine"},"source_commit":"source"},
          "launch":{"provider":"lc0_cuda","declaration":launch}}
    external={"family":"stockfish","version":"19","expected_uci_name":"Stockfish 19",
              "arguments":[],"requested_options":{"Threads":"2"},"source":{"commit":"sf-source"},
              "binary":{"sha256":"sf-binary","bytes":100},"assets":[]}
    declaration={"evaluation_policy":"shared-runtime-warm-readhash","rove_tree_max_edges":262144,"engines":[{"endpoint":"rove_zero","configuration":rove},
                            {"endpoint":"external_uci","configuration":external}],
                 "resources":{"affinity":[0,1],"memory_high_bytes":6*1024**3,
                              "memory_max_bytes":12*1024**3,"swap_max_bytes":0}}
    regression={"cache_preparation":{"status":"passed","report":{"canonical_sha256":"bundle","native_code_loaded":False,"GPU_used":False,"files_modified":False}},"tree_max_edges":262144,"candidate":{"binary_sha256":"engine","source_commit":"source",
                 "expected_profile":{"backend_sha256":"backend","encoding_manifest_sha256":"encoding",
                                     "model_manifest_sha256":"export_manifest"}},
                "resource_notes":{"cpu_affinity":[0,1],"memory_high_GiB":6,"memory_max_GiB":12,"swap":0}}
    raw={"onnx_sha256":"onnx","input_f32_bytes_equal":True,"backend_shutdown_completed":True,
         "backend_sha256":"backend","runtime_sha256":"ort_library","cuda_bundle_sha256":"bundle",
         "provider":"cuda","precision":"fp32","tf32":False}
    rules={"identity":{"onnx_sha256":"onnx","runtime_sha256":"ort_library","runtime_bundle_sha256":"bundle"},
           "provider":"cuda","precision":"fp32"}
    preflight={"endpoint":copy.deepcopy(external),"two_fresh_processes":True,
               "quit_and_owned_group_cleanup":True,"stop_legal_bestmove":True,"supported_requested_options":True}
    return declaration,{"adapter_regression":[regression],"numerical":[raw,rules],"external_preflight":[preflight]}


class ModelPairGateTests(unittest.TestCase):
    def test_registered_model_runtime_and_opponent_match(self):
        gate_identities(*fixture())

    def test_default_regression_does_not_admit_a_native_environment_experiment(self):
        declaration,reports=fixture()
        declaration["engines"][0]["configuration"]["environment"]={"variables":{"MALLOC_ARENA_MAX":"2"}}
        with self.assertRaises(ValueError):
            gate_identities(declaration,reports)

    def test_success_for_different_inputs_does_not_admit_play(self):
        mutations=[
            lambda r:r["adapter_regression"][0]["candidate"].update(binary_sha256="other-engine"),
            lambda r:r["adapter_regression"][0].update(tree_max_edges=100000),
            lambda r:r["adapter_regression"][0]["cache_preparation"].update(status="failed"),
            lambda r:r["adapter_regression"][0]["resource_notes"].update(memory_max_GiB=24),
            lambda r:r["numerical"][0].update(onnx_sha256="other-model"),
            lambda r:r["numerical"][0].update(runtime_sha256="other-ORT"),
            lambda r:r["numerical"][1]["identity"].update(runtime_bundle_sha256="other-CUDA"),
            lambda r:r["external_preflight"][0]["endpoint"]["binary"].update(sha256="other-Stockfish"),
            lambda r:r["external_preflight"][0]["endpoint"]["requested_options"].update(Threads="8"),
            lambda r:r["external_preflight"][0]["endpoint"].update(environment={"variables":{"RZ_PUBLIC_FIXTURE":"other"}}),
        ]
        for mutate in mutations:
            declaration,reports=fixture()
            mutate(reports)
            with self.assertRaises(ValueError):
                gate_identities(declaration,reports)


if __name__=="__main__":
    unittest.main()
