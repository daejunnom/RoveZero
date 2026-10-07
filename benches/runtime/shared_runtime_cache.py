#!/usr/bin/env python3
"""Read/hash an existing immutable runtime cache; never load, repair or evict it."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from bounded_local import outside_git, put
from paired_search import read_json, require

SCHEMA = "rz-shared-runtime-cache-prepare/1"
MAX_BYTES = 8*1024**3


def validate(spec):
    require(set(spec)=={"schema","cache_directory","canonical_sha256","files"} and spec["schema"]==SCHEMA,
            "runtime cache preparation declaration differs")
    require(re.fullmatch(r"[0-9a-f]{64}",spec["canonical_sha256"]) is not None,"runtime cache identity invalid")
    root=outside_git(spec["cache_directory"])
    require(root.name=="cuda-"+spec["canonical_sha256"] and root.is_dir(),"existing registered CUDA cache required")
    require(isinstance(spec["files"],list) and len(spec["files"])==19,"exactly 19 declared runtime files required")
    names=set()
    total=0
    for file in spec["files"]:
        require(set(file)=={"filename","bytes","sha256"} and
                re.fullmatch(r"lib[a-zA-Z0-9_.+-]+\.so(?:\.[0-9.]+)?",file["filename"]) is not None and
                file["filename"] not in names and type(file["bytes"]) is int and 0<file["bytes"]<=MAX_BYTES and
                re.fullmatch(r"[0-9a-f]{64}",file["sha256"]) is not None,"invalid runtime cache member")
        names.add(file["filename"]);total+=file["bytes"]
    require(total<=MAX_BYTES,"shared runtime exceeds finite read budget")
    require(not(stat.S_IMODE(root.stat().st_mode)&0o222),"runtime cache is not sealed")
    require({entry.name for entry in root.iterdir()}==names,"runtime cache membership differs")
    return root,total


def prepare(spec, report, wall=60):
    if os.name!="posix" or not hasattr(os,"O_NOFOLLOW"):
        raise ValueError("Linux no-follow file admission required")
    started=time.monotonic()
    root,total=validate(spec)
    witnesses=[]
    for file in sorted(spec["files"],key=lambda f:f["filename"]):
        if time.monotonic()-started>=wall:raise TimeoutError("cache preparation wall limit exceeded")
        path=root/file["filename"]
        descriptor=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
        try:
            before=os.fstat(descriptor)
            require(stat.S_ISREG(before.st_mode) and before.st_size==file["bytes"] and
                    not(stat.S_IMODE(before.st_mode)&0o222),"runtime cache file is not a sealed regular file")
            digest=hashlib.sha256();read_bytes=0
            while True:
                if time.monotonic()-started>=wall:raise TimeoutError("cache preparation wall limit exceeded")
                block=os.read(descriptor,1024**2)
                if not block:break
                read_bytes+=len(block)
                require(read_bytes<=file["bytes"],"runtime file grew during preparation")
                digest.update(block)
            after=os.fstat(descriptor);named=path.lstat()
            identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
            require(identity(before)==identity(after)==identity(named) and read_bytes==file["bytes"] and
                    digest.hexdigest()==file["sha256"],"runtime cache identity changed")
            witnesses.append(dict(filename=file["filename"],bytes=read_bytes,sha256=digest.hexdigest(),
                                  device=before.st_dev,inode=before.st_ino))
        finally:
            os.close(descriptor)
    result=dict(schema=SCHEMA,status="passed",wall_seconds=time.monotonic()-started,
                canonical_sha256=spec["canonical_sha256"],shared_file_bytes=total,files=witnesses,
                native_code_loaded=False,GPU_used=False,files_modified=False,
                page_residency="unknown_read_completed_not_locked",
                accounting_scope="preparation_is_separate_from_fresh_engine_cgroup_peak_not_whole_host_peak")
    put(outside_git(report),result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec",type=Path);parser.add_argument("report",type=Path)
    args=parser.parse_args()
    result=prepare(read_json(args.spec,64*1024),args.report)
    print(json.dumps(dict(status=result["status"],shared_file_bytes=result["shared_file_bytes"])),flush=True)

