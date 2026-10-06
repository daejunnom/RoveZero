#!/usr/bin/env python3
"""Run one V2 external pair only after pinned numerical and regression gates."""
import argparse
import json
from pathlib import Path
from bounded_local import OwnedRun, outside_git, put
from paired_search import read_json, require, sha256


def gate_identities(declaration, reports):
    """A passed report from another executable/model/opponent cannot admit play."""
    rove=[e["configuration"] for e in declaration["engines"] if e["endpoint"]=="rove_zero"]
    external=[e["configuration"] for e in declaration["engines"] if e["endpoint"]=="external_uci"]
    require(len(rove)==len(external)==1,"one RoveZero and external endpoint required")
    rove,external=rove[0],external[0]
    launch=rove["launch"]
    require(launch["provider"]=="lc0_cuda","this pilot requires the registered BT4 CUDA B1 recipe")
    launch=launch["declaration"]
    profile=launch["profile"]["runtime"]
    artifacts={a["role"]:a["artifact"] for a in launch["artifacts"]}
    regression=reports["adapter_regression"][0]
    cache=regression["cache_preparation"]
    require(cache["status"]=="passed" and cache["report"]["canonical_sha256"]==launch["cuda_bundle"]["canonical_sha256"] and
            not cache["report"]["native_code_loaded"] and not cache["report"]["GPU_used"] and not cache["report"]["files_modified"],
            "registered shared runtime read/hash preparation incomplete")
    require("shared-runtime-warm-readhash" in declaration["evaluation_policy"],"external pilot cache condition differs")
    require(regression["tree_max_edges"]==declaration["rove_tree_max_edges"],
            "adapter regression used a different tree envelope")
    tested=regression["candidate"]
    require(tested["binary_sha256"]==rove["tool"]["binary"]["sha256"] and
            tested["source_commit"]==rove["tool"]["source_commit"] and
            tested["expected_profile"]["backend_sha256"]==profile["expected_backend_sha256"] and
            tested["expected_profile"]["encoding_manifest_sha256"]==profile["expected_encoding_sha256"] and
            tested["expected_profile"]["model_manifest_sha256"]==artifacts["export_manifest"]["sha256"],
            "adapter regression used a different executable or semantic profile")
    resources=regression["resource_notes"]
    policy=declaration["resources"]
    require(resources["cpu_affinity"]==policy["affinity"] and
            resources["memory_high_GiB"]*1024**3==policy["memory_high_bytes"] and
            resources["memory_max_GiB"]*1024**3==policy["memory_max_bytes"] and
            resources["swap"]==policy["swap_max_bytes"],"regression resource policy differs")
    matched_raw=matched_rules=0
    for report in reports["numerical"]:
        if report.get("onnx_sha256")==artifacts["onnx"]["sha256"]:
            require(report["input_f32_bytes_equal"] and report["backend_shutdown_completed"] and
                    report["backend_sha256"]==profile["expected_backend_sha256"] and
                    report["runtime_sha256"]==artifacts["ort_library"]["sha256"] and
                    report["cuda_bundle_sha256"]==launch["cuda_bundle"]["canonical_sha256"] and
                    report["provider"]=="cuda" and report["precision"]=="fp32" and not report["tf32"],
                    "raw numerical gate used different runtime/model semantics")
            matched_raw+=1
        identity=report.get("identity",{})
        if identity.get("onnx_sha256")==artifacts["onnx"]["sha256"]:
            require(identity["runtime_sha256"]==artifacts["ort_library"]["sha256"] and
                    identity["runtime_bundle_sha256"]==launch["cuda_bundle"]["canonical_sha256"] and
                    report["provider"]=="cuda" and report["precision"]=="fp32",
                    "Rules numerical gate used different runtime/model semantics")
            matched_rules+=1
    require(matched_raw==matched_rules==1,"one raw and Rules numerical gate for the played model required")
    preflight=reports["external_preflight"][0]
    probed=preflight["endpoint"]
    for key in ("family","version","expected_uci_name","arguments","requested_options","source"):
        require(probed[key]==external[key],"external preflight differs: "+key)
    for key in ("sha256","bytes"):
        require(probed["binary"][key]==external["binary"][key],"external preflight binary differs")
    require(probed["assets"]==external["assets"] and preflight["two_fresh_processes"] and
            preflight["quit_and_owned_group_cleanup"] and preflight["stop_legal_bestmove"] and
            preflight["supported_requested_options"],"external lifecycle preflight incomplete")


def run(registration, output):
    require(set(registration)=={"binary","binary_sha256","lock","lock_sha256","asset_root","gate_reports"},"pair registration fields differ")
    binary=outside_git(registration["binary"])
    lock=outside_git(registration["lock"])
    require(binary.is_file() and binary.stat().st_size<=64*1024**2 and sha256(binary)==registration["binary_sha256"] and sha256(lock)==registration["lock_sha256"],"V2 executable/lock differs")
    envelope=read_json(lock,1024**2)
    require(envelope["lock_version"]==2 and envelope["domain"]=="rz-e01-model-endpoints-v2" and envelope["execution_ready"] is False,"separate V2 lock required")
    declaration=envelope["input"]
    require(declaration["comparison"]=="external_engine" and declaration["seed"]==1 and declaration["max_plies"]==256 and declaration["clock"]=={"base_ms":120000,"increment_ms":1000},"first external pilot conditions differ")
    require(declaration["white_order"]==["baseline","candidate"] and declaration["opening"]["initial"]=="startpos" and not declaration["opening"]["moves"],"paired complete standard start required")
    gates=registration["gate_reports"]
    require(set(gates)=={"numerical","adapter_regression","external_preflight"},"all independent gates required")
    reports={}
    for gate, refs in gates.items():
        require(isinstance(refs,list) and len(refs)==(4 if gate=="numerical" else 1),"four numerical and one regression/preflight report required")
        require(len({ref["sha256"] for ref in refs})==len(refs),"duplicate evidence cannot satisfy a gate")
        reports[gate]=[]
        for ref in refs:
            path=outside_git(ref["path"])
            report=read_json(path,1024**2)
            require(sha256(path)==ref["sha256"] and report.get("status")=="passed","prior gate did not pass: "+gate)
            reports[gate].append(report)
    gate_identities(declaration,reports)
    asset_root=Path(registration["asset_root"])
    require(asset_root.is_absolute() and asset_root.is_dir() and not asset_root.is_symlink(),"absolute asset root required")
    owner=OwnedRun(outside_git(output),wall=900,address_space=declaration["budget"]["address_space_per_process_bytes"])
    result={"status":"failed","strength_eligible":False,"lock_sha256":registration["lock_sha256"]}
    try:
        put(owner.directory/"registration.json",registration)
        arena=owner.directory/"arena"
        arena.mkdir()
        owner.spawn([str(binary),"execute",str(lock),str(asset_root),str(arena),"stockfish19-pilot"])
        require(owner.wait_exit()==0,"V2 pair command failed; PGN/failure evidence retained")
        files=list(arena.glob("*/model-endpoints-pair-receipt.v2.json"))
        require(len(files)==1,"one V2 pair receipt required")
        receipt=read_json(files[0],128*1024)
        require(receipt["integration_checks_passed"] and receipt["cleanup_verified"] and receipt["scored_games"]==2 and len(receipt["provider_sessions"])==2 and len(receipt["external_execution"])==1,"external pair did not meet independent acceptance gates")
        result.update(status="passed",receipt=str(files[0]),receipt_sha256=sha256(files[0]),scored_games=2)
    except (OSError,ValueError,KeyError,TypeError,TimeoutError) as exc:
        result["error"]=str(exc)
    finally:
        result["capture"]=owner.finish()
        c=result["capture"]
        events=dict(line.split() for line in c["resources"]["memory.events"].splitlines())
        if c["exit_code"]!=0 or c["forced_cleanup"] or c["cleanup_error"] or c["remaining_owned_processes"] or int(events["oom"]) or int(events["oom_kill"]):
            result["status"]="failed"
            result.setdefault("error","owned lifecycle cleanup gate failed")
        put(owner.directory/"result.json",result)
    print(json.dumps(result),flush=True)
    return 0 if result["status"]=="passed" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    raise SystemExit(run(read_json(args.registration,1024**2),args.output))
