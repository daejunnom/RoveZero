"""CPU-only hard-route acceptance, separate from speed/device claims."""
import hashlib
import json
from pathlib import Path
import tempfile


def verify_cpu_hard_routes(graph_path, feed):
    """ORT profiling proves only the selected private branch executes on CPU.

    Optimizations are disabled for this execution witness so node provenance is
    observable. Production optimized execution and device residency need their
    own native acceptance. The raw tiny profiles are temporary; node evidence
    and profile digests remain in the returned report.
    """
    import numpy as np
    import onnxruntime as ort
    if ort.__version__ != "1.22.0":
        raise ValueError("hard-route witness requires pinned ORT1.22.0")
    reports = []
    with tempfile.TemporaryDirectory(prefix="rovezero-pals-route-") as directory:
        for role in ("proposer", "critic"):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(directory) / role)
            session = ort.InferenceSession(str(graph_path), sess_options=options, providers=["CPUExecutionProvider"])
            current = {**feed, "role_is_critic": np.asarray(role == "critic", dtype=np.bool_)}
            if {value.name for value in session.get_inputs()} != set(current):
                raise ValueError("hard-route witness input/profile tensor names mismatch")
            # V2 adds explicit bounded line/mask/local-ref tensors; route
            # profiling consumes those same tensors and never synthesizes an
            # old summary input for a registered full-line graph.
            result = session.run(None, current)
            profile = Path(session.end_profiling())
            encoded = profile.read_bytes()
            events = json.loads(encoded)
            selected = "critic_" if role == "critic" else "proposer_"
            inactive = "proposer_" if role == "critic" else "critic_"
            nodes = [event.get("name", "") for event in events
                     if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")]
            selected_nodes = [name for name in nodes if name.startswith(selected)]
            inactive_nodes = [name for name in nodes if name.startswith(inactive)]
            shared_nodes = [name for name in nodes if name.startswith("shared_pc_if_")]
            if not selected_nodes or inactive_nodes or not shared_nodes:
                raise ValueError("CPU If profile does not prove selected-only private execution")
            if result[4].shape != () or result[4].dtype != np.bool_ or bool(result[4]) != (role == "critic"):
                raise ValueError("hard-route witness role tag mismatch")
            reports.append({"role": role, "selected_private_nodes": selected_nodes,
                            "inactive_private_nodes": inactive_nodes, "shared_nodes_executed": len(shared_nodes),
                            "profile_sha256": hashlib.sha256(encoded).hexdigest(),
                            "profile_raw_retained": False, "provider": "CPUExecutionProvider",
                            "graph_optimization": "disabled_for_routing_witness"})
            del session
    return {"status": "passed", "cases": reports, "cuda": "not_run",
            "production_optimized_routing": "requires_native_check", "device_residency_sharing": "unknown"}
