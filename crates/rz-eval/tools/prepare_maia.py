#!/usr/bin/env python3
"""Local, bounded conversion with the external pinned GPL LC0 tool.

Requires onnx==1.18.0. No downloads; no LC0 implementation is vendored/linked.
The original and derived weights retain upstream GPL terms and stay external.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE_SHA = "e2f565f42d7cd9f122557e6dc4eb84e5bbaedceda1d404dc485d3611c7c97a12"
PROTO_SHA = "e8fe5a7f35594d4190c440a2716fda57ba9d071b26175d509476be7a4fda052a"
LC0_COMMIT = "fd71a2d921b689c5f479d3227c3806c8e272d9c5"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args):
    import onnx

    if onnx.__version__ != "1.18.0":
        raise ValueError("use pinned onnx==1.18.0")
    source, tool, checkout, out = [p.resolve() for p in
                                   (args.source, args.lc0, args.lc0_source, args.output_dir)]
    if source.stat().st_size != 1262607 or digest(source) != SOURCE_SHA:
        raise ValueError("selected compressed Maia source identity differs")
    with gzip.open(source, "rb") as stream:
        pb = stream.read(1738565)
    if len(pb) != 1738564 or hashlib.sha256(pb).hexdigest() != PROTO_SHA:
        raise ValueError("selected protobuf source identity differs")
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True)
    if commit != LC0_COMMIT or dirty:
        raise ValueError("converter checkout must be the clean pinned LC0 commit")
    out.mkdir(parents=True, exist_ok=True)
    model = out / "maia-1900.onnx"
    if model.exists() or (out / "manifest.json").exists():
        raise ValueError("use a new output directory; existing artifacts are preserved")
    command = [str(tool), "leela2onnx", f"--input={source}", f"--output={model}",
               "--onnx-data-type=f32", "--onnx-opset=17", "--onnx-batch-size=-1"]
    completed = subprocess.run(command, capture_output=True, timeout=60, check=True)
    (out / "conversion.log").write_bytes(completed.stdout + completed.stderr)
    if not 0 < model.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError("converted model exceeds the selected asset bound")
    graph = onnx.load(model, load_external_data=False)
    onnx.checker.check_model(graph)
    if [(v.domain, v.version) for v in graph.opset_import] != [("", 17)]:
        raise ValueError("unexpected opset")
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise ValueError("external tensor files are unsupported")
    def shape(value):
        tensor = value.type.tensor_type
        if tensor.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("expected FP32 interface")
        return [d.dim_value if d.HasField("dim_value") else -1 for d in tensor.shape.dim]
    inputs = {v.name: shape(v) for v in graph.graph.input}
    outputs = {v.name: shape(v) for v in graph.graph.output}
    if inputs != {"/input/planes": [-1, 112, 8, 8]} or outputs != {
            "/output/policy": [-1, 1858], "/output/wdl": [-1, 3]}:
        raise ValueError("unexpected model interface")
    producers = {name: node for node in graph.graph.node for name in node.output}
    if producers["/output/wdl"].op_type != "Softmax":
        raise ValueError("WDL must already be probabilities")
    if producers["/output/policy"].op_type == "Softmax":
        raise ValueError("policy must remain logits")
    manifest = dict(schema=1, source_gzip_sha256=SOURCE_SHA, source_protobuf_sha256=PROTO_SHA,
                    onnx_sha256=digest(model), onnx_bytes=model.stat().st_size,
                    converter_commit=commit, converter_binary_sha256=digest(tool), command=command,
                    opset=17, dtype="float32", input_name="/input/planes",
                    policy_name="/output/policy", wdl_name="/output/wdl",
                    weights_license="GPL-3.0", redistribution_ready=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "RIGHTS.txt").write_text(
        "External Maia1 weights and derived ONNX: upstream GPL-3.0, not RoveZero MIT.\n"
        "Source: https://github.com/CSSLab/maia-chess/releases/tag/v1.0\n"
        "License: https://github.com/CSSLab/maia-chess/blob/37de81e2bef89336e03266b3b5f7e1155ba68f5d/LICENSE\n"
        "Weights scope: https://github.com/CSSLab/maia-chess/issues/8#issuecomment-744732973\n"
        "Local FP32 ONNX conversion only. Redistribution/source obligations are not certified.\n"
        f"Converter source and notices: https://github.com/LeelaChessZero/lc0/tree/{commit}\n")
    print(json.dumps({"manifest": str(out / "manifest.json"), "onnx_sha256": manifest["onnx_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--lc0", required=True, type=Path)
    parser.add_argument("--lc0-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    prepare(parser.parse_args())
