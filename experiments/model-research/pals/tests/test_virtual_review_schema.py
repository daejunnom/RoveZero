"""Pure schema/packing checks: NN, Torch, ORT and optimizer imports forbidden."""
import ast
import copy
import hashlib
import importlib.abc
from pathlib import Path
import sys
import unittest


class DenyNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "numpy", "onnx", "onnxruntime") or fullname == "rz_pals_model.model":
            raise AssertionError("pure schema test attempted neural import: " + fullname)
        return None


GUARD = DenyNeuralImports()
sys.meta_path.insert(0, GUARD)
try:
    from rz_pals_model.config import ENCODING_V3, MODEL_SEMANTICS_V3, ModelConfig, PROFILES, public_tensor_names, role_tensor_names, matmul_flops
    from rz_pals_model.virtual_review import (INPUT_SCHEMA, BUCKETS, RECORD_FEATURES, QUERY_FEATURES,
        controlled_copy_plan, execution_keys, independent_record_key, input_digest, length_bucket,
        prepare_payload, validate_payload)
finally:
    sys.meta_path.remove(GUARD)


def payload():
    return {"schema": INPUT_SCHEMA, "role": "proposer", "anchors": [{"board": [0] * 64, "metadata": [0.] * 16}],
            "records": [{"kind": "proposal", "perspective": 1, "completed_iteration": False, "rules_terminal": False,
                         "line": [[12, 28, 0]], "anchor": 0, "parent": None, "supersedes": None}],
            "query": {"purpose": "propose_policy", "perspective": 1, "prefix": [], "proposal": [[12, 28, 0]], "counter": [], "counter_present":False, "anchors": [0, 0, 0]},
            "candidates": [[12, 28, 0]], "divergences": [], "token_budget": 512}


class VirtualReviewSchemaTests(unittest.TestCase):
    def test_tensor_encoding_registry_is_separately_pinned(self):
        from rz_pals_model.virtual_review import ENCODING_REGISTRY_FIELDS, registered_encoding_sha256
        self.assertEqual(len(ENCODING_REGISTRY_FIELDS),33)
        self.assertEqual(registered_encoding_sha256(), "57f53319114c3da14612f5d4dc99ff1a05fa115c471a5ed764d02b7fce0f5d0f")
        # These fields pin physical ABI/projection. Rules feature semantics and
        # actual source files remain independent identifiers.
        self.assertIn("counter_present=explicit_bool", "\n".join(ENCODING_REGISTRY_FIELDS))
    def setUp(self):
        sys.meta_path.insert(0,GUARD)

    def tearDown(self):
        sys.meta_path.remove(GUARD)
    def test_all_buckets_and_capacity(self):
        for length, bucket in ((0,8),(8,8),(9,16),(17,32),(33,64),(65,128),(129,256),(256,256)):
            self.assertEqual(length_bucket(length), bucket)
        for length in (-1, 257, True):
            with self.assertRaises(ValueError): length_bucket(length)
        for bucket in BUCKETS:
            value = payload(); value["records"][0]["line"] *= bucket
            result = prepare_payload(value)
            self.assertEqual(result["admission"]["line_bucket"], bucket)
            self.assertEqual(len(result["feed"]["record_line_tokens"][0][0]), bucket)

    def test_actual_query_duplicates_count(self):
        value = payload(); value["token_budget"] = 128
        value["records"][0]["line"] *= 64
        value["query"]["proposal"] *= 64
        self.assertEqual(validate_payload(value)["token_count"], 128)
        value["query"]["prefix"] = [[12, 28, 0]]
        with self.assertRaisesRegex(ValueError, "ViewCapacity"): validate_payload(value)

    def test_explicit_empty_counter_has_presence_without_extra_tokens(self):
        value=payload();absent=input_digest(value);tokens=validate_payload(value)["token_count"]
        value["query"]["counter_present"]=True
        self.assertEqual(validate_payload(value)["token_count"],tokens)
        self.assertEqual(prepare_payload(value)["feed"]["query"][0][7],1.)
        self.assertNotEqual(input_digest(value),absent)
        value["query"]["counter_present"]=False;value["query"]["counter"]=[[52,36,0]]
        with self.assertRaisesRegex(ValueError,"presence"):validate_payload(value)

    def test_anchors_relationships_and_semantic_axes(self):
        value = payload(); value["anchors"].append({"board": [1] * 64, "metadata": [0.] * 16})
        value["records"][0]["anchor"] = 1
        self.assertEqual(prepare_payload(value)["feed"]["record_anchor"], [[1]])
        value["query"]["anchors"][0] = 2
        with self.assertRaisesRegex(ValueError, "anchor"): validate_payload(value)
        value = payload(); value["records"].append(copy.deepcopy(value["records"][0]))
        value["records"][0]["parent"] = 1; value["records"][1]["supersedes"] = 0
        with self.assertRaisesRegex(ValueError, "cycle"): validate_payload(value)
        self.assertEqual(len(RECORD_FEATURES), 8); self.assertEqual(len(QUERY_FEATURES), 8)
        value = payload(); value["records"][0]["depth"] = 12
        with self.assertRaisesRegex(ValueError, "fields"): validate_payload(value)

    def test_empty_physical_padding_and_caller_copy(self):
        value = payload(); original = copy.deepcopy(value)
        prepared = prepare_payload(value)
        prepared["feed"]["anchor_board"][0][0][0] = 12
        self.assertEqual(value, original)
        value["records"] = []; value["candidates"] = []
        feed = prepare_payload(value)["feed"]
        self.assertEqual(feed["record_mask"], [[False]])
        self.assertEqual(feed["record_relations"], [[[-1,-1]]])
        self.assertEqual(feed["candidate_mask"], [[False]])
        self.assertFalse(any(feed["record_line_mask"][0][0]))

    def test_candidate_promotions_order_and_anchor_change_digest(self):
        value = payload(); value["candidates"] = [[48,56,p] for p in (1,2,3,4)]
        original = input_digest(value)
        value["candidates"].reverse(); self.assertNotEqual(input_digest(value), original)
        value["candidates"].append(value["candidates"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"): validate_payload(value)
        value = payload(); original = input_digest(value)
        value["anchors"][0]["metadata"][0] = 1
        self.assertNotEqual(input_digest(value), original)
        # Independent Rust golden has exactly the same B1 typed tensor bytes.
        self.assertEqual(input_digest(payload()), "34f13fc24725fc4eb338044a502d5da37534a0eb9044194baf4c50a1534d5e57")

    def test_record_page_excludes_relationship_mixing(self):
        value = payload(); value["records"].append(copy.deepcopy(value["records"][0]))
        original = independent_record_key(value, 0)
        value["records"][0]["parent"] = 1
        self.assertEqual(independent_record_key(value, 0), original)
        self.assertNotEqual(input_digest(value), input_digest({**value, "records": payload()["records"]}))
        value["records"][1]["line"].append([52,36,0])
        self.assertEqual(independent_record_key(value, 0), original)

    def test_fresh_and_warm_domains_and_context(self):
        value = payload(); args = dict(model_epoch="11"*32,rules_history="22"*32,anchor_history_seals=["22"*32],backend="ort-cuda",precision="fp32",seed_digest="33"*32)
        keys = execution_keys(value, **args)
        self.assertNotEqual(keys["fresh_key"], keys["warm_invocation_key"])
        value["records"][0]["line"].append([52,36,0])
        revised = execution_keys(value, **args)
        self.assertEqual(keys["context_key"], revised["context_key"])
        self.assertNotEqual(keys["fresh_key"], revised["fresh_key"])
        value["query"]["proposal"].append([52,36,0])
        self.assertNotEqual(keys["context_key"], execution_keys(value, **args)["context_key"])
        value["query"]["purpose"]="value_fresh"
        with self.assertRaisesRegex(ValueError,"Fresh"): execution_keys(value,**args)

    def test_explicit_controlled_copy_inventory(self):
        source = {"reader.weight": {"shape": [384,384], "dtype": "fp32_le", "semantics": "unchanged_reader", "sha256": "aa"*32}}
        target = copy.deepcopy(source)
        result = controlled_copy_plan(source,target,{"reader.weight": "unchanged_reader"})
        self.assertEqual(result["copies_executed"],0); self.assertFalse(result["learned_transfer_claim"])
        target["reader.weight"]["shape"] = [384,128]
        with self.assertRaisesRegex(ValueError, "shape"): controlled_copy_plan(source,target,{"reader.weight": "unchanged_reader"})

    def test_registered_configuration_preserves_old_profiles(self):
        config = ModelConfig.for_profile("virtual_review_v1")
        self.assertEqual(config.max_records,64)
        self.assertEqual(config.model_semantics,MODEL_SEMANTICS_V3)
        self.assertEqual(config.encoding,ENCODING_V3)
        self.assertEqual(len(PROFILES),4)
        self.assertNotIn("profile", ModelConfig.baseline().to_dict())
        self.assertEqual(ModelConfig.baseline().max_records,128)
        self.assertEqual(public_tensor_names(config)[0],"anchor_board")
        self.assertIn("anchor_context",role_tensor_names(config))
        with self.assertRaises(ValueError): ModelConfig(profile="virtual_review_v1").validate()

    def test_analytic_work_requires_explicit_physical_bucket(self):
        config=ModelConfig.for_profile("virtual_review_v1")
        with self.assertRaises(ValueError):matmul_flops(config,"proposer",3,7)
        result=matmul_flops(config,"proposer",3,7,line_bucket=8,anchors=2)
        self.assertEqual(result["public_encode_matmul_flops"],911_496_192)
        self.assertEqual(result["role_forward_matmul_flops"],258_700_288)

    def test_finite_f64_not_representable_as_f32_is_rejected(self):
        value=payload();value["anchors"][0]["metadata"][0]=1e300
        with self.assertRaisesRegex(ValueError,"finite"):validate_payload(value)

    def test_sources_parse_without_execution(self):
        root = Path(__file__).parents[1] / "src" / "rz_pals_model"
        for name in ("model.py", "config.py", "virtual_review.py", "onnx_virtual_review.py", "artifacts.py", "controlled_initialization.py"):
            ast.parse((root/name).read_text(encoding="utf-8"), filename=name)

    def test_v3_rules_descriptor_and_export_schema_are_separate(self):
        from rz_pals_model.artifacts import (VIRTUAL_RULES_FEATURES,VIRTUAL_RULES_BOUNDS,
            validate_virtual_rules_declaration,validate_virtual_export_manifest,validate_virtual_warm_export_manifest)
        from rz_pals_model.onnx_virtual_review import virtual_io_spec,SHARED_LAYOUT,LAYOUT_REVISION,WARM_LAYOUT
        config=ModelConfig.for_profile("virtual_review_v1")
        descriptor={"schema":"rovezero.pals-rules-descriptor.v3","rules_input_profile":"rz-pals-rules-virtual-review-v1",
            "config":config.to_dict(),"rules_base_semantics":"explicit_rules_features","metadata_features":[f"metadata{i}" for i in range(16)],
            "record_features":list(RECORD_FEATURES),"query_features":list(QUERY_FEATURES),"divergence_features":[f"divergence{i}" for i in range(8)],
            "virtual_review_features":list(VIRTUAL_RULES_FEATURES),"bounds":copy.deepcopy(VIRTUAL_RULES_BOUNDS),
            "rules_encoder_source_sha256":"11"*32,"learned_input_compatibility":"unverified_declaration_only",
            "semantic_digest_algorithm":"sha256_u64le_length_prefixed_utf8_fields"}
        fields=[descriptor["rules_input_profile"],descriptor["rules_base_semantics"]]+descriptor["metadata_features"]+list(RECORD_FEATURES)+list(QUERY_FEATURES)+descriptor["divergence_features"]+list(VIRTUAL_RULES_FEATURES)
        digest=hashlib.sha256()
        for field in fields:
            encoded=field.encode();digest.update(len(encoded).to_bytes(8,"little"));digest.update(encoded)
        descriptor["semantic_fields"]=fields;descriptor["rules_input_semantic_sha256"]=digest.hexdigest()
        self.assertEqual(len(fields),56)
        self.assertEqual(validate_virtual_rules_declaration(descriptor,config),digest.hexdigest())
        spec=virtual_io_spec(config)
        graphs=[]
        for role,filename in (("public","public_memory.onnx"),("shared_pc","shared_pc_if.onnx"),("validator","role_validator.onnx")):
            graphs.append({"role":role,"file":filename,"sha256":"22"*32,"opset":17,
                **{key:[{"name":name,"dtype":dtype,"shape":list(shape)} for name,dtype,shape in spec[role][key]] for key in ("inputs","outputs")}})
        metadata={"config":config.to_dict(),"checkpoint_sha256":"33"*32,"trained":False,"training_steps":0}
        manifest={**metadata,"schema":"rovezero.pals-model.v3","model_semantics":config.model_semantics,"encoding_schema":config.encoding,
            "model_profile":config.profile,"layout":SHARED_LAYOUT,"layout_revision":LAYOUT_REVISION,"roles":["proposer","critic","validator"],"validator_present":True,
            "task_names":["defend_response","attack_repair","widen_responses","lower_selectivity","resume_task","cross_profile_recheck","defer"],
            "graphs":graphs,"rules_input_declaration":descriptor,"rules_input_semantic_sha256":digest.hexdigest(),"rules_encoder_source_sha256":"11"*32,
            "physical_line_buckets":[8,16,32,64,128,256],"session_policy":"one_dynamic_L_graph_no_bucket_weight_copies","precision":"fp32","tf32":False}
        self.assertEqual(validate_virtual_export_manifest(manifest,metadata),["proposer","critic","validator"])
        wrong=copy.deepcopy(manifest);wrong["graphs"][0]["inputs"][5]["shape"][2]=256
        with self.assertRaisesRegex(ValueError,"dynamic"):validate_virtual_export_manifest(wrong,metadata)
        wrong=copy.deepcopy(manifest);wrong["rules_input_declaration"]["bounds"]["max_records"]=128
        with self.assertRaisesRegex(ValueError,"bounds"):validate_virtual_export_manifest(wrong,metadata)
        warm=copy.deepcopy(manifest)
        warm.update({"layout":WARM_LAYOUT,"roles":["proposer","critic"],"validator_present":False,"approximate":True,
            "warm_seed_policy":"accepted_final_latent_same_model_role_rules_query_epoch_physical_fence"})
        warm["graphs"]=warm["graphs"][:2]
        warm["graphs"][1]={"role":"shared_warm","file":"shared_pc_if_approx_warm.onnx","sha256":"44"*32,"opset":17,
            **{key:[{"name":name,"dtype":dtype,"shape":list(shape)} for name,dtype,shape in spec["shared_warm"][key]] for key in ("inputs","outputs")}}
        self.assertEqual(validate_virtual_warm_export_manifest(warm,metadata),["proposer","critic"])
        with self.assertRaisesRegex(ValueError,"semantic"):validate_virtual_export_manifest(warm,metadata)
        wrong=copy.deepcopy(warm);wrong["graphs"][1]["inputs"][-1]["shape"]=["batch"]
        with self.assertRaisesRegex(ValueError,"dynamic"):validate_virtual_warm_export_manifest(wrong,metadata)


if __name__ == "__main__": unittest.main()
