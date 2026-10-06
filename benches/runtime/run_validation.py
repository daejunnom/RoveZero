#!/usr/bin/env python3
"""Finite pinned validation command. No automatic fallback, benchmark or retry."""
import argparse
import json
import re
import time
from pathlib import Path
from bounded_local import OwnedRun, outside_git, put
from paired_search import read_json, require, sha256


def execute(registration, output):
    require(set(registration)=={"source_commit","binary","binary_sha256","arguments","report","wall_seconds","assets","scope"},"validation registration fields differ")
    require(re.fullmatch(r"[0-9a-f]{40}",registration["source_commit"]) is not None,"full source SHA required")
    require(1<=registration["wall_seconds"]<=900,"finite validation wall required")
    require(isinstance(registration["arguments"],list) and len(registration["arguments"])<=32 and all(isinstance(x,str) and len(x)<=4096 and not any(c in x for c in "\r\n\0") for x in registration["arguments"]),"finite argument list required")
    binary=outside_git(registration["binary"])
    require(binary.is_file() and binary.stat().st_size<=64*1024**2 and sha256(binary)==registration["binary_sha256"],"validation binary differs")
    require(1<=len(registration["assets"])<=32,"finite asset list required")
    for artifact in registration["assets"]:
        path=Path(artifact["path"])
        require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.stat().st_size==artifact["bytes"] and 0<artifact["bytes"]<=2*1024**3 and sha256(path)==artifact["sha256"],"validation asset differs")
    report=Path(registration["report"])
    require(not report.is_absolute() and report.parts and all(p not in {".",".."} for p in report.parts),"relative owned report required")
    output=outside_git(output)
    owner=OwnedRun(output,wall=registration["wall_seconds"])
    result={"status":"failed","source_commit":registration["source_commit"],"scope":registration["scope"],"started_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    try:
        put(output/"registration.json",registration)
        args=[x.replace("{{output}}",str(output)) for x in registration["arguments"]]
        require(not any("{{" in x for x in args),"unknown validation token")
        owner.spawn([str(binary),*args])
        require(owner.wait_exit()==0,"validation command failed")
        evidence=read_json(output/report,1024**2)
        require(evidence.get("status")=="passed","validation report did not pass")
        result.update(status="passed",report_sha256=sha256(output/report))
    except (OSError,ValueError,KeyError,TypeError,TimeoutError) as exc:
        result["error"]=str(exc)
    finally:
        result["capture"]=owner.finish()
        c=result["capture"]
        events=dict(line.split() for line in c["resources"]["memory.events"].splitlines())
        if c["exit_code"]!=0 or c["forced_cleanup"] or c["cleanup_error"] or c["remaining_owned_processes"] or int(events["oom"]) or int(events["oom_kill"]):
            result["status"]="failed"
            result.setdefault("error","process/resource/cleanup gate failed")
        put(output/"result.json",result)
    print(json.dumps(result),flush=True)
    return 0 if result["status"]=="passed" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    raise SystemExit(execute(read_json(args.registration,1024**2),args.output))
