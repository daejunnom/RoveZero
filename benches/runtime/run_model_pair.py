#!/usr/bin/env python3
"""Run one V2 external pair only after pinned numerical and regression gates."""
import argparse
import json
from pathlib import Path
from bounded_local import OwnedRun, outside_git, put
from paired_search import read_json, require, sha256


def run(registration, output):
    require(set(registration)=={"binary","binary_sha256","lock","lock_sha256","asset_root","gate_reports"},"pair registration fields differ")
    binary=outside_git(registration["binary"])
    lock=outside_git(registration["lock"])
    require(binary.is_file() and binary.stat().st_size<=64*1024**2 and sha256(binary)==registration["binary_sha256"] and sha256(lock)==registration["lock_sha256"],"V2 executable/lock differs")
    envelope=read_json(lock,1024**2)
    require(envelope["lock_version"]==2 and envelope["domain"]=="rz-e01-model-endpoints-v2" and envelope["execution_ready"] is False,"separate V2 lock required")
    declaration=envelope["input"]
    require(declaration["comparison"]=="external_engine" and declaration["seed"]==1 and declaration["max_plies"]==256 and declaration["clock"]=={"base_ms":120000,"increment_ms":1000,"ponder":False},"first external pilot conditions differ")
    require(declaration["white_order"]==["baseline","candidate"] and declaration["opening"]["initial"]=="startpos" and not declaration["opening"]["moves"],"paired complete standard start required")
    gates=registration["gate_reports"]
    require(set(gates)=={"numerical","adapter_regression","external_preflight"},"all independent gates required")
    for gate, refs in gates.items():
        require(isinstance(refs,list) and 1<=len(refs)<=8,"finite gate reports required")
        for ref in refs:
            path=outside_git(ref["path"])
            require(sha256(path)==ref["sha256"] and read_json(path,1024**2).get("status")=="passed","prior gate did not pass: "+gate)
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
        if c["exit_code"]!=0 or c["forced_cleanup"] or c["cleanup_error"] or c["remaining_owned_processes"]:
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
