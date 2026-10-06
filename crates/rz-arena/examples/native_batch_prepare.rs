//! Offline V4 batch or V5 B1 memory declarations from existing pinned V3 inputs. No model load,
//! inference, runner or GPU query. Actual inputs are reverified by arena launch.
use rz_eval::{
    asset,
    onnx::{BackendConfig, Provider, declared_backend_identity},
};
use rz_experiments::*;
use std::{error::Error, fs::OpenOptions, io::Write, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(args.len() == 4 || args.len() == 5 && args[4] == "--b1-memory-aa") {
        return Err("usage: native_batch_prepare V3_INPUT ENGINE_BINARY SOURCE_SHA NEW_INPUT [--b1-memory-aa]".into());
    }
    let memory_aa = args.len() == 5;
    if args[2].len() != 40 || !args[2].bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("exact source SHA required".into());
    }
    let source: serde_json::Value =
        serde_json::from_slice(&asset::read_bounded(Path::new(&args[0]), 64 * 1024)?)?;
    let original: CudaSearchPilotPairSpecV3 = serde_json::from_value(source.clone())?;
    original.validate()?;
    let mut input = source;
    input["schema_version"] = if memory_aa { 5 } else { 4 }.into();
    input["pilot"] = serde_json::Value::Null;
    let run_id = if memory_aa {
        "bt4-b1-clock-memory-aa-v5"
    } else {
        "bt4-batch-b1-b4-pilot-v4"
    };
    input["run_id"] = run_id.into();
    input["pair_id"] = format!("{run_id}-00").into();
    if memory_aa {
        input["purpose"] = "cuda_nn_integration".into();
    }
    input["clock"] = serde_json::json!({"base_ms":120_000,"increment_ms":1_000});
    input["timeouts"]["runtime_ms"] = 900_000.into();
    input["timeouts"]["shutdown_ms"] = 30_000.into();
    let binary = Path::new(&args[1]).canonicalize()?;
    let mut file = std::fs::File::open(&binary)?;
    let mut digest = sha2::Sha256::new();
    use sha2::Digest;
    let mut buffer = [0; 64 * 1024];
    use std::io::Read;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let binary_hash = format!("{:x}", digest.finalize());
    let binary_bytes = file.metadata()?.len();
    for engine in input["engines"].as_array_mut().ok_or("engines missing")? {
        let width = if memory_aa || engine["role"] == "baseline" {
            1
        } else {
            4
        };
        engine["source_commit"] = args[2].clone().into();
        let artifacts = engine["artifacts"]
            .as_array_mut()
            .ok_or("artifacts missing")?;
        for binding in artifacts.iter_mut().filter(|a| a["role"] == "binary") {
            binding["artifact"]["path"] = binary
                .to_str()
                .ok_or("binary UTF8")?
                .trim_start_matches('/')
                .into();
            binding["artifact"]["sha256"] = binary_hash.clone().into();
            binding["artifact"]["bytes"] = binary_bytes.into();
            binding["artifact"]["source"] =
                format!("https://github.com/daejunnom/RoveZero/tree/{}", args[2]).into();
        }
        let artifact_hash = |role: &str| -> Result<[u8; 32], Box<dyn Error>> {
            let binding = artifacts
                .iter()
                .find(|a| a["role"] == role)
                .ok_or("artifact role missing")?;
            Ok(asset::parse_sha256(
                binding["artifact"]["sha256"]
                    .as_str()
                    .ok_or("hash missing")?,
            )?)
        };
        let ort = artifact_hash("ort_library")?;
        let model = artifact_hash("export_manifest")?;
        let bundle = asset::parse_sha256(
            engine["cuda_bundle"]["canonical_sha256"]
                .as_str()
                .ok_or("bundle missing")?,
        )?;
        let mut config = BackendConfig::cpu();
        config.max_batch = width;
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 3 * 1024 * 1024 * 1024,
        };
        let backend = declared_backend_identity(ort, model, Some(bundle), &config);
        let mut runtime = engine["profile"]["runtime"].clone();
        runtime["batch_size"] = width.into();
        runtime["expected_backend_sha256"] = backend
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
            .into();
        let mut search = engine["profile"]["search"].clone();
        search["final_selection"] = "visits".into();
        engine["profile"] = if memory_aa {
            serde_json::json!({"model":"bt4_it332","runtime":runtime,"search":search})
        } else {
            serde_json::json!({"runtime":runtime,"search":search,"max_batch_wait_us":if width>1 {200} else {0}})
        };
        engine["engine_id"] = if memory_aa {
            format!(
                "RoveZero-BT4-FP32-PUCT-visits-B1-{}",
                engine["role"].as_str().ok_or("role missing")?
            )
        } else {
            format!("RoveZero-BT4-FP32-PUCT-visits-B{width}")
        }
        .into();
    }
    let reserve = 4 * (64 * 1024 * 1024 + 4 * 1024 * 1024 + 768 * 1024);
    let old = input["budget"]["max_runtime_bytes"]
        .as_u64()
        .ok_or("runtime budget missing")?;
    input["budget"]["max_runtime_bytes"] = if memory_aa { old } else { old.max(reserve) }.into();
    let total = ["max_input_bytes", "max_output_bytes", "max_runtime_bytes"]
        .iter()
        .try_fold(0u64, |sum, key| {
            sum.checked_add(input["budget"][key].as_u64().ok_or("budget missing")?)
                .ok_or("budget overflow")
        })?;
    input["budget"]["max_artifact_bytes"] = total.into();
    let bytes = if memory_aa {
        let spec: CudaB1ClockPairSpecV5 = serde_json::from_value(input)?;
        spec.validate()?;
        serde_json::to_vec_pretty(&spec)?
    } else {
        let spec: CudaBatchPilotPairSpecV4 = serde_json::from_value(input)?;
        spec.validate()?;
        serde_json::to_vec_pretty(&spec)?
    };
    let output = Path::new(&args[3]);
    if !output.is_absolute()
        || output
            .parent()
            .ok_or("parent missing")?
            .canonicalize()?
            .ancestors()
            .any(|p| p.join(".git").exists())
    {
        return Err("fresh absolute output outside Git required".into());
    }
    let mut saved = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    saved.write_all(&bytes)?;
    saved.sync_all()?;
    Ok(())
}
