"""Source/DTO capture checks only; no model, neural library or inference."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest.mock import patch


def blocked_import(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"} or name == "model" or name.endswith(".model"):
        raise AssertionError("neural/model import is forbidden in source-only capture tests")
    return original_import(name, *args, **kwargs)


original_import = builtins.__import__
with patch("builtins.__import__", side_effect=blocked_import):
    from rz_pals_model import numeric_capture as capture


def bindings(collection=False, cuda=False):
    from rz_pals_model.virtual_review import registered_encoding_sha256
    roles = ["public", "shared_pc"] + (["validator"] if collection else [])
    return {"model_profile":"virtual_review_v1", "model_schema":"rovezero.pals-model.v3",
        "model_semantics":"rovezero.pals-model-semantics.v3", "encoding_schema":"rovezero.pals-virtual-review.v1",
        "model_semantic_sha256":"01"*32, "encoding_sha256":registered_encoding_sha256(), "model_epoch":0,
        "weights_sha256":"03"*32, "export_manifest_sha256":"04"*32,
        "graphs":{role:"05"*32 for role in roles}, "runtime_library_sha256":"06"*32,
        "cuda_libraries":["07"*32] if cuda else [], "cuda_manifest_sha256":"08"*32 if cuda else None,
        "backend":"ort_cuda" if cuda else "ort_cpu", "precision":"fp32", "tf32":False,
        "source_components":{name:"09"*32 for name in ("model_native","model_encoding","nn_runtime","model_reference","model_export")}}


def first_payload():
    return capture.python_input(capture.make_case_plan(bindings())["groups"][0]["cases"][0]["input"]["input"])


def tensor(name, dtype, shape, raw):
    return {"name":name, "dtype":dtype, "shape":shape, "bytes":list(raw)}


class NumericCaptureTests(unittest.TestCase):
    def setUp(self):
        self.blocker = patch("builtins.__import__", side_effect=blocked_import)
        self.blocker.start()

    def tearDown(self):
        self.blocker.stop()

    def test_binding_rosters_domains_and_dtype_are_exact_and_copied(self):
        for collection, cuda in ((False,False),(True,False),(False,True)):
            original = bindings(collection,cuda)
            result = capture.validate_bindings(original)
            self.assertEqual(result,original)
            result["graphs"]["public"] = "aa"*32
            self.assertNotEqual(result["graphs"],original["graphs"])
        for change in (lambda b: b.update(tf32=True), lambda b: b.update(precision="fp16"),
                       lambda b: b.update(model_epoch=True), lambda b: b.update(model_epoch=2**64),
                       lambda b: b["graphs"].update(shared_warm="01"*32),
                       lambda b: b["source_components"].update(extra_source="01"*32),
                       lambda b: b["source_components"].pop("model_reference"),
                       lambda b: b.update(encoding_sha256="02"*32),
                       lambda b: b.update(cuda_libraries=["01"*32]),
                       lambda b: b.update(weights_sha256="not_a_pin")):
            bad = bindings(); change(bad)
            with self.assertRaises(ValueError): capture.validate_bindings(bad)
        for libraries in (["unhashed-library.so"], ["07"*32,"07"*32], []):
            bad = bindings(cuda=True); bad["cuda_libraries"] = libraries
            with self.assertRaises(ValueError): capture.validate_bindings(bad)

    def test_deterministic_case_plan_has_exact_registered_math_coverage(self):
        for collection,total,roles in ((False,100,{"proposer","critic"}),(True,150,{"proposer","critic","validator"})):
            plan = capture.make_case_plan(bindings(collection))
            self.assertEqual(capture.validate_case_plan(plan),total)
            self.assertEqual(plan,capture.make_case_plan(bindings(collection)))
            self.assertEqual({group["role"] for group in plan["groups"]},roles)
            self.assertEqual({group["line_bucket"] for group in plan["groups"]},{8,16,32,64,128,256})
            self.assertEqual(plan["invocation"],"fresh"); self.assertEqual(plan["batch"],1)
            ids = []
            for group in plan["groups"]:
                canonical = [case for case in group["cases"] if ".L" in case["id"]]
                self.assertEqual(len(canonical),8)
                coverage = {(len(case["input"]["input"]["anchors"]), case["input"]["input"]["anchors"][0]["metadata"][0]) for case in canonical}
                self.assertEqual(coverage,{(anchor,float(white)) for anchor in range(1,5) for white in (0,1)})
                for case in canonical:
                    payload = case["input"]["input"]
                    self.assertEqual(payload["query"]["anchors"],[0,len(payload["anchors"])-1,0])
                    self.assertEqual(len(payload["query"]["proposal"]),group["line_bucket"])
                    self.assertEqual([move["promotion"] for move in payload["candidates"]],[1,2,3,4])
                if group["line_bucket"] == 8:
                    empty = next(case for case in group["cases"] if case["id"].endswith(".empty"))
                    reversed_case = next(case for case in group["cases"] if case["id"].endswith(".order_reversed"))
                    self.assertEqual(empty["input"]["input"]["records"],[])
                    self.assertEqual(reversed_case["input"]["input"]["candidates"],list(reversed(canonical[0]["input"]["input"]["candidates"])))
                ids.extend(case["id"] for case in group["cases"])
            self.assertEqual(len(ids),len(set(ids)))
            self.assertNotIn("passed",plan); self.assertNotIn("physical_completion_known",plan)

    def test_typed_move_codec_is_an_independent_copy_and_roundtrips(self):
        original = first_payload(); typed = capture.rust_input(original)
        self.assertEqual(set(typed["candidates"][0]),{"from","to","promotion"})
        self.assertEqual(capture.python_input(typed),original)
        typed["candidates"][0]["from"] = 47
        self.assertEqual(original["candidates"][0][0],48)
        for malformed in ({"from":0,"to":1}, {"from":0,"to":1,"promotion":0,"extra":1}):
            bad = capture.rust_input(original); bad["candidates"][0] = malformed
            with self.assertRaises(ValueError): capture.python_input(bad)
        bad = capture.rust_input(original); bad["query"]["proposal"][0]["promotion"] = 5
        with self.assertRaises(ValueError): capture.python_input(bad)

    def test_case_plan_rejects_schema_identity_role_bucket_and_duplicate_cases(self):
        original = capture.make_case_plan(bindings())
        for change in (lambda p: p.update(schema_version=4),lambda p: p.update(invocation="approx_warm_seeded"),
                       lambda p: p.update(schema_version=5.0),lambda p: p.update(batch=True),
                       lambda p: p.update(batch=2), lambda p: p["groups"][0].update(line_bucket=16),
                       lambda p: p["groups"][0].update(line_bucket=8.0),
                       lambda p: p["groups"][0].update(id="x"*128),
                       lambda p: p["groups"][1].update(id=p["groups"][0]["id"]),
                       lambda p: p["groups"][0].update(role="validator"),
                       lambda p: p["groups"][0]["cases"][0]["input"]["input"].update(candidates=[]),
                       lambda p: p["groups"][0]["cases"][0]["input"]["input"]["anchors"][0]["metadata"].__setitem__(0,0.5),
                       lambda p: p["groups"][0]["cases"][1].update(id=p["groups"][0]["cases"][0]["id"]),
                       lambda p: p["groups"][0]["cases"][0]["input"].update(model_input_kind="legacy_full_line")):
            bad = copy.deepcopy(original); change(bad)
            with self.assertRaises(ValueError): capture.validate_case_plan(bad)

    def test_actual_prepared_source_bytes_shapes_dtype_and_padding(self):
        payload = first_payload(); rows = capture.captured_tensors(payload)
        self.assertEqual([row["name"] for row in rows],sorted(row["name"] for row in rows))
        by_name = {row["name"]:row for row in rows}
        self.assertEqual(by_name["anchor_board"]["shape"],[1,1,64])
        self.assertEqual(by_name["anchor_metadata"]["shape"],[1,1,16])
        self.assertEqual(by_name["query_line_tokens"]["shape"],[1,3,8,3])
        self.assertEqual(by_name["query_line_tokens"]["dtype"],"i64_le")
        self.assertEqual(by_name["record_relations"]["bytes"],list(struct.pack("<qq",-1,-1)))
        self.assertEqual(by_name["query_line_mask"]["bytes"],[0]*8+[1]*8+[0]*8)
        self.assertEqual(by_name["record_line_mask"]["bytes"],[1]*7+[0])
        self.assertEqual(by_name["divergence_mask"]["bytes"],[0])
        payload["records"] = []
        empty = {row["name"]:row for row in capture.captured_tensors(payload)}
        self.assertEqual(empty["record_mask"]["bytes"],[0])
        self.assertEqual(empty["record_line_tokens"]["shape"],[1,1,8,3])
        self.assertTrue(all(byte == 0 for byte in empty["record_line_tokens"]["bytes"]))

    def test_tensor_digest_matches_explicit_rust_u64le_framing_and_role(self):
        raw = struct.pack("<ff",1.,-0.)
        tensors = [tensor("query","f32_le",[1,2],raw)]
        frame = lambda data: struct.pack("<Q",len(data))+data
        preimage = (frame(b"rz-pals-captured-tensor-bytes-v5/1")+frame(b"proposer")+frame(b"query")+frame(b"f32")
                    +struct.pack("<QQQ",2,1,2)+frame(raw))
        expected = hashlib.sha256(preimage).hexdigest()
        self.assertEqual(capture.tensor_digest(tensors,"proposer"),expected)
        self.assertNotEqual(capture.tensor_digest(tensors,"critic"),expected)
        positive_zero = [tensor("query","f32_le",[1,2],struct.pack("<ff",1.,0.))]
        self.assertNotEqual(capture.tensor_digest(positive_zero,"proposer"),expected)
        with self.assertRaises(ValueError): capture.tensor_digest(tensors,"unknown_role")

    def test_tensor_roster_refuses_nonfinite_masks_ragged_or_ambiguous_bytes(self):
        valid = [tensor("a","bool_u8",[1],[1]),tensor("b","i64_le",[1],struct.pack("<q",-1))]
        self.assertEqual(len(capture.tensor_digest(valid,"validator")),64)
        for bad in ([tensor("a","bool_u8",[1],[2])], [tensor("a","f32_le",[1],struct.pack("<f",float("nan")))],
                    [tensor("a","f32_le",[1],struct.pack("<f",float("inf")))], [tensor("a","i64_le",[2],b"\0"*8)],
                    [tensor("a","bool_u8",[0],b"")], [tensor("a","bool_u8",[1],[True])],
                    list(reversed(valid)), valid+[copy.deepcopy(valid[1])]):
            with self.assertRaises(ValueError): capture.tensor_digest(bad,"validator")
        with self.assertRaises(ValueError): capture._shape_flat([[1],[1,2]])

    def test_source_gate_rejects_before_any_model_or_export_access(self):
        class ForbiddenModel:
            @property
            def training(self):
                raise AssertionError("model inspected before immutable source admission")
        plan = capture.make_case_plan(bindings())
        actual = capture.source_pins()
        plan["bindings"]["source_components"].update(actual)
        with self.assertRaisesRegex(ValueError,"execute=True"):
            capture.capture_reference(ForbiddenModel(),{},"not_read",plan,"not_written",checkpoint_path="not_read",
                expected_reference_source_sha256=actual["model_reference"])
        with self.assertRaisesRegex(ValueError,"source differs"):
            capture.capture_reference(ForbiddenModel(),{},"not_read",plan,"not_written",checkpoint_path="not_read",
                expected_reference_source_sha256="ff"*32,execute=True)
        with patch.object(capture,"source_pins",return_value=actual):
            for seconds in (0,3601,True):
                with self.assertRaisesRegex(ValueError,"finite reference"):
                    capture.capture_reference(ForbiddenModel(),{},"not_read",plan,"not_written",checkpoint_path="not_read",
                        expected_reference_source_sha256=actual["model_reference"],max_seconds=seconds,execute=True)

    def test_source_named_byte_framing_reads_sources_without_model_import(self):
        actual = capture.source_pins()
        folder = Path(capture.__file__).parent
        def reference(domain, names):
            framed = []
            for item in [domain.encode(),*[part for name in sorted(names) for part in (name.encode(),(folder/name).read_bytes())]]:
                framed.append(struct.pack("<Q",len(item))+item)
            return hashlib.sha256(b"".join(framed)).hexdigest()
        self.assertEqual(actual["model_reference"],reference("rz-pals-numeric-reference-source-v5/1",capture._REFERENCE_FILES))
        self.assertEqual(actual["model_export"],reference("rz-pals-numeric-export-source-v5/1",capture._EXPORT_FILES))
        self.assertNotEqual(actual["model_reference"],hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest())

    def test_registered_encoding_vocabulary_and_digest_match_rust_source(self):
        from rz_pals_model.virtual_review import ENCODING_REGISTRY_FIELDS, registered_encoding_sha256
        root = Path(__file__).parents[4]
        source = (root/"crates/rz-eval/src/pals_model/virtual_review.rs").read_text(encoding="utf-8")
        match = re.search(r"pub const ENCODING_REGISTRY_FIELDS: &\[&str\] = &\[(.*?)\n\];",source,re.S)
        self.assertIsNotNone(match)
        declared = tuple(json.loads(raw) for raw in re.findall(r'^\s*("[^"\n]*"),\s*$',match.group(1),re.M))
        self.assertEqual(len(declared),33)
        self.assertEqual(tuple(ENCODING_REGISTRY_FIELDS),declared)
        framed = b"".join(struct.pack("<Q",len(field.encode()))+field.encode() for field in declared)
        self.assertEqual(registered_encoding_sha256(),hashlib.sha256(framed).hexdigest())

    def test_regular_file_size_and_checkpoint_pin_reject_before_decode(self):
        with tempfile.TemporaryDirectory(prefix="rz-numeric-capture-") as directory:
            root = Path(directory); checkpoint = root/"fixture.pt"; checkpoint.write_bytes(b"pure-source-fixture-not-a-checkpoint")
            with self.assertRaisesRegex(ValueError,"bounded regular"):
                capture._regular_read(checkpoint,1)
            self.assertEqual(capture._regular_read(checkpoint,100),checkpoint.read_bytes())
            with self.assertRaisesRegex(ValueError,"checkpoint bytes differ"):
                capture._verify_model_checkpoint(object(),checkpoint,"00"*32,float("inf"),None)
            with self.assertRaises(TimeoutError):
                capture._verify_model_checkpoint(object(),checkpoint,"00"*32,0,None)
            with self.assertRaisesRegex(ValueError,".pt checkpoint"):
                capture._verify_model_checkpoint(object(),root/"not_read.bin","00"*32,float("inf"),None)
        self.assertEqual(capture.MAX_MANIFEST_BYTES,256*1024)

    def test_exclusive_publication_pins_exact_bytes_and_preserves_existing(self):
        with tempfile.TemporaryDirectory(prefix="rz-numeric-capture-") as directory:
            output = Path(directory)/"case-plan.json"
            receipt = capture.write_case_plan(bindings(),output)
            raw = output.read_bytes()
            self.assertEqual(receipt["sha256"],hashlib.sha256(raw).hexdigest())
            self.assertEqual(receipt["bytes"],len(raw))
            self.assertEqual(capture.validate_case_plan(json.loads(raw)),100)
            self.assertFalse(output.with_suffix(".json.partial").exists())
            with self.assertRaises(FileExistsError): capture.write_case_plan(bindings(),output)
            self.assertEqual(output.read_bytes(),raw)
            self.assertEqual(json.loads((Path(directory)/"pals-artifact-owner.json").read_text())["owner"],"rz-pals-model")

    def test_file_and_total_output_limits_fail_without_publication(self):
        with tempfile.TemporaryDirectory(prefix="rz-numeric-capture-") as directory:
            output = Path(directory)/"never-created.json"
            with patch.object(capture,"MAX_FILE_BYTES",1):
                with self.assertRaisesRegex(ValueError,"output bound"): capture._publish({"fixture":1},output,0)
            with self.assertRaisesRegex(ValueError,"output bound"):
                capture._publish({"fixture":1},output,capture.MAX_TOTAL_BYTES)
            self.assertFalse(output.exists()); self.assertFalse(output.with_suffix(".json.partial").exists())


if __name__ == "__main__":
    unittest.main()
