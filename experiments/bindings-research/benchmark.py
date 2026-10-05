"""USER-RUN timing; supervisor must supply CPU2 / 512MiB / 120s cgroup.

Rust-body timing uses the existing diagnostic pump, not native UCI throughput.
Startup/import, per-call transport and the Rust body remain separate fields.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cases import CASES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if sys.version_info[:3] != (3,12,3):
        raise RuntimeError("PoC requires exactly Python 3.12.3")
    if not args.cli.is_absolute() or not args.report.is_absolute():
        raise ValueError("absolute paths required")
    if any((p / ".git").exists() for p in args.report.parent.resolve().parents):
        raise ValueError("report must stay outside Git")
    samples, cold = [], []
    report = {"schema_version":1,"accepted":False,"cpu_only_mock":True,
              "claim":"call_overhead_only_not_native_search_speed_or_strength",
              "limits":{"cpu":2,"memory_max_bytes":512*1024**2,"wall_s":120},
              "cli_sha256":hashlib.file_digest(args.cli.open("rb"),"sha256").hexdigest(),
              "cold":cold,"samples":samples}
    # Exclusive report creation precedes workload; failures retain partial data.
    with args.report.open("x", encoding="utf-8") as saved:
        process = None
        try:
            for path in ["rust_direct","rust_cli","python_binding"]:
                for run in range(3):
                    start = time.perf_counter_ns()
                    if path == "rust_direct":
                        output = subprocess.run([str(args.cli),"--direct",CASES["startpos"],"1","128"],
                            text=True,capture_output=True,check=True,timeout=10)
                    elif path == "rust_cli":
                        output = subprocess.run([str(args.cli)], input=json.dumps({"position_command":CASES["startpos"]})+"\n",
                            text=True,capture_output=True,check=True,timeout=10)
                    else:
                        output = subprocess.run([sys.executable,"-c",
                            "import json,time; s=time.perf_counter_ns(); import rz_bindings_poc; i=time.perf_counter_ns()-s; r=rz_bindings_poc.search_once('position startpos'); print(json.dumps({'import_ns':i,'measurement':r}))"],
                            text=True,capture_output=True,check=True,timeout=10)
                    cold.append({"path":path,"run":run,"process_wall_ns":time.perf_counter_ns()-start,
                                 "response":json.loads(output.stdout.splitlines()[-1])})
            started = time.perf_counter_ns()
            import rz_bindings_poc
            report["persistent_binding_import_ns"] = time.perf_counter_ns()-started
            started = time.perf_counter_ns()
            process = subprocess.Popen([str(args.cli)], stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            if not json.loads(process.stdout.readline()).get("ready"):
                raise RuntimeError("CLI not ready")
            report["persistent_cli_ready_ns"] = time.perf_counter_ns()-started
            for name, position in CASES.items():
                direct = subprocess.run([str(args.cli),"--direct",position,"20","128"],
                    check=True,text=True,capture_output=True,timeout=30)
                expected = None
                for index, line in enumerate(direct.stdout.splitlines()):
                    response=json.loads(line)
                    if not response["ok"]: raise RuntimeError(response["error"])
                    measurement=response["measurement"]
                    if expected is None: expected=measurement["result"]
                    if measurement["result"] != expected: raise RuntimeError("direct nondeterminism")
                    samples.append({"path":"rust_direct","case":name,"run":index,
                        "measurement":measurement,"full_call_ns":measurement["rust_call_ns"],
                        "transport_ns":"none_in_process; JSON sink excluded"})
                if index != 19: raise RuntimeError("direct sample count differs")
                for path in ["rust_cli","python_binding"]:
                    for index in range(20):
                        start=time.perf_counter_ns()
                        if path=="rust_cli":
                            serialize=time.perf_counter_ns()
                            payload=json.dumps({"position_command":position})+"\n"
                            serialization_ns=time.perf_counter_ns()-serialize
                            process.stdin.write(payload); process.stdin.flush()
                            response=json.loads(process.stdout.readline())
                            if not response["ok"]: raise RuntimeError(response["error"])
                            measurement=response["measurement"]
                        else:
                            serialization_ns=0
                            measurement=rz_bindings_poc.search_once(position)
                        full=time.perf_counter_ns()-start
                        if measurement["result"] != expected: raise RuntimeError("path result mismatch")
                        samples.append({"path":path,"case":name,"run":index,
                            "measurement":measurement,"full_call_ns":full,"serialization_ns":serialization_ns})
            process.stdin.close(); process.wait(timeout=5)
            if process.returncode: raise RuntimeError("CLI failed on exit")
            report["accepted"]=True
        except BaseException as error:
            report["failure"]=str(error)
            raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate(); process.wait(timeout=5)
            json.dump(report,saved,ensure_ascii=False,indent=2); saved.write("\n")

if __name__ == "__main__": main()
