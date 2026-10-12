//! Host evidence and bounded ponder lifecycle audit for the opt-in match policy.
use crate::ArenaError;
#[cfg(any(target_os = "linux", test))]
use rz_experiments::GpuCapacity;
use rz_experiments::{MatchExecutionV1, MatchHardware};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct MatchHardwareProbe {
    pub hardware: MatchHardware,
    pub gpu_inventory_known: bool,
    pub gpu_probe_error: Option<String>,
}

#[cfg(any(target_os = "linux", test))]
const GPU_INVENTORY_MAX_BYTES: usize = 8192;

/// Parse only the bounded, unquoted three-column renderer requested below.
/// A successful process exit alone is not evidence of a usable inventory.
#[cfg(any(target_os = "linux", test))]
fn parse_gpu_inventory(bytes: &[u8]) -> Result<Vec<GpuCapacity>, ArenaError> {
    if bytes.len() > GPU_INVENTORY_MAX_BYTES {
        return Err(ArenaError::Budget(
            "GPU inventory exceeds byte bound".into(),
        ));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| ArenaError::Invalid("invalid GPU inventory encoding".into()))?;
    let mut gpus: Vec<GpuCapacity> = Vec::new();
    for line in text.lines() {
        if gpus.len() >= 16 || line.len() > 512 {
            return Err(ArenaError::Budget(
                "GPU inventory row bound exceeded".into(),
            ));
        }
        let mut fields = line.split(',');
        let id = fields.next().unwrap_or_default().trim_matches(' ');
        let model = fields
            .next()
            .ok_or_else(|| ArenaError::Invalid("invalid GPU inventory row".into()))?
            .trim_matches(' ');
        let memory = fields
            .next()
            .ok_or_else(|| ArenaError::Invalid("invalid GPU inventory row".into()))?
            .trim_matches(' ');
        if fields.next().is_some()
            || id.is_empty()
            || id.len() > 80
            || !id.as_bytes()[0].is_ascii_alphanumeric()
            || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            || model.is_empty()
            || model.len() > 256
            || !model
                .bytes()
                .all(|b| (b.is_ascii_graphic() || b == b' ') && b != b'"')
        {
            return Err(ArenaError::Invalid("invalid GPU inventory identity".into()));
        }
        // memory.total with nounits is an integral MiB count. Reject signs,
        // floating/non-finite values, units and overflow rather than coercing.
        let memory = (!memory.is_empty() && memory.bytes().all(|b| b.is_ascii_digit()))
            .then(|| memory.parse::<u64>().ok())
            .flatten()
            .and_then(|m| m.checked_mul(1024 * 1024))
            .filter(|m| *m > 0)
            .ok_or_else(|| ArenaError::Invalid("invalid GPU capacity".into()))?;
        if gpus.iter().any(|gpu| gpu.id == id) {
            return Err(ArenaError::Invalid(
                "duplicate GPU inventory identity".into(),
            ));
        }
        gpus.push(GpuCapacity {
            id: id.into(),
            model: Some(model.into()),
            vram_bytes: memory,
        });
    }
    if gpus.is_empty() {
        return Err(ArenaError::Invalid("GPU inventory is empty".into()));
    }
    Ok(gpus)
}
#[cfg(not(target_os = "linux"))]
pub fn probe_match_hardware() -> Result<MatchHardwareProbe, ArenaError> {
    Err(ArenaError::Invalid(
        "automatic match placement currently requires Linux".into(),
    ))
}
#[cfg(target_os = "linux")]
pub fn probe_match_hardware() -> Result<MatchHardwareProbe, ArenaError> {
    use nix::{sched::sched_getaffinity, unistd::Pid};
    use std::{fs, path::Path};
    let affinity =
        sched_getaffinity(Pid::from_raw(0)).map_err(|e| ArenaError::Io(e.to_string()))?;
    let mut cores = BTreeMap::<(u32, u32), Vec<u32>>::new();
    for cpu in 0..4096 {
        if !affinity.is_set(cpu).unwrap_or(false) {
            continue;
        }
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
        let read = |name| -> Result<u32, ArenaError> {
            fs::read_to_string(format!("{path}/{name}"))
                .map_err(|e| ArenaError::Io(e.to_string()))?
                .trim()
                .parse()
                .map_err(|_| ArenaError::Invalid("invalid physical CPU topology".into()))
        };
        cores
            .entry((read("physical_package_id")?, read("core_id")?))
            .or_default()
            .push(cpu as u32);
    }
    let program = Path::new("/usr/bin/nvidia-smi");
    let mut gpus = Vec::new();
    let mut gpu_probe_error = None;
    let known = if program.exists() {
        let file = fs::File::open(program).map_err(|e| ArenaError::Io(e.to_string()))?;
        let output = crate::supervise(
            &file,
            &[
                "--query-gpu=uuid,name,memory.total".into(),
                "--format=csv,noheader,nounits".into(),
            ],
            Path::new("/tmp"),
            crate::ProcessLimits {
                wall_ms: 3000,
                shutdown_grace_ms: 500,
                max_output_bytes: GPU_INVENTORY_MAX_BYTES as u64,
                max_child_processes: 3,
            },
            None,
        )?;
        if output.receipt.group_cleanup != crate::CleanupStatus::Gone
            || output.pending_child.is_some()
        {
            return Err(ArenaError::HardwareProbe {
                cause: "GPU inventory probe cleanup is unconfirmed".into(),
                process: Box::new(output),
            });
        }
        if output.receipt.exit_code != Some(0) || !output.receipt.errors.is_empty() {
            gpu_probe_error = Some(format!(
                "GPU inventory unavailable: exit={:?}, stop={:?}, errors={:?}",
                output.receipt.exit_code, output.receipt.stop, output.receipt.errors
            ));
            false
        } else {
            gpus = parse_gpu_inventory(&output.stdout)?;
            true
        }
    } else {
        false
    };
    Ok(MatchHardwareProbe {
        hardware: MatchHardware {
            cpu_cores: cores.into_values().collect(),
            gpus,
        },
        gpu_inventory_known: known,
        gpu_probe_error,
    })
}
pub fn validate_match_host(execution: &MatchExecutionV1) -> Result<(), ArenaError> {
    let probe = probe_match_hardware()?;
    validate_match_hardware(&execution.hardware, &probe)
}

fn validate_match_hardware(
    declared_hardware: &MatchHardware,
    probe: &MatchHardwareProbe,
) -> Result<(), ArenaError> {
    // A declared inventory may select a subset, but cannot split one physical core
    // into several apparently isolated cores or claim a device/capacity not present.
    let mut owners = BTreeMap::new();
    for (group, declared) in declared_hardware.cpu_cores.iter().enumerate() {
        let actual = probe
            .hardware
            .cpu_cores
            .iter()
            .position(|core| declared.iter().all(|cpu| core.contains(cpu)))
            .ok_or_else(|| {
                ArenaError::Invalid("declared CPUs do not share one available physical core".into())
            })?;
        if owners.insert(actual, group).is_some() {
            return Err(ArenaError::Invalid(
                "declared inventory splits SMT siblings".into(),
            ));
        }
    }
    for gpu in &declared_hardware.gpus {
        if !probe.gpu_inventory_known
            || !probe.hardware.gpus.iter().any(|g| {
                g.id == gpu.id
                    && match gpu.model.as_ref() {
                        // A named physical device declares its full capacity,
                        // not an engine reservation. Never let a larger device
                        // masquerade as a smaller controlled-comparison device.
                        Some(model) => {
                            g.model.as_ref() == Some(model) && g.vram_bytes == gpu.vram_bytes
                        }
                        // Preserve historical unnamed inventory semantics.
                        None => g.vram_bytes >= gpu.vram_bytes,
                    }
            })
        {
            return Err(ArenaError::Invalid(
                "GPU inventory/model/capacity is unavailable or differs; no engine started".into(),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct PonderProtocolAudit {
    pub started: u32,
    pub hits: u32,
    pub stopped: u32,
}
pub fn audit_ponder_protocol(
    trace: &[u8],
    engines: [&str; 2],
) -> Result<PonderProtocolAudit, ArenaError> {
    let text = std::str::from_utf8(trace)
        .map_err(|_| ArenaError::Invalid("ponder trace is not UTF-8".into()))?;
    if text.len() > 64 * 1024 * 1024 {
        return Err(ArenaError::Budget("ponder trace exceeds byte bound".into()));
    }
    let mut pending = [false; 2];
    let mut policies = [0u32; 2];
    let mut audit = PonderProtocolAudit {
        started: 0,
        hits: 0,
        stopped: 0,
    };
    for (index, line) in text.lines().enumerate() {
        if index >= 131_072 || line.len() > 4096 {
            return Err(ArenaError::Budget(
                "ponder trace line budget exceeded".into(),
            ));
        }
        // Only the pinned runner renderer is evidence. Engine stdout/stderr
        // payloads cannot manufacture policy or drain witnesses.
        if !line.starts_with("[TRACE ] [") || !line.contains("RZ_PONDER_") {
            continue;
        }
        let message = crate::native_exit::cuda_exit_trace_message(line)
            .ok_or_else(|| ArenaError::Invalid("ponder trace renderer differs".into()))?;
        let Some(record) = message.strip_prefix("RZ_PONDER_") else {
            continue;
        };
        if record.starts_with("FAILURE_V1 ") {
            return Err(ArenaError::Invalid("ponder cleanup failed".into()));
        }
        let fields: BTreeMap<_, _> = record
            .split_whitespace()
            .skip(1)
            .filter_map(|s| s.split_once('='))
            .collect();
        let name = fields
            .get("engine")
            .ok_or_else(|| ArenaError::Invalid("ponder engine identity missing".into()))?;
        let index = engines
            .iter()
            .position(|e| e == name)
            .ok_or_else(|| ArenaError::Invalid("foreign ponder engine".into()))?;
        if record.starts_with("POLICY_V1 ") {
            if fields.len() != 2
                || record.split_whitespace().count() != 3
                || fields.get("enabled") != Some(&"true")
            {
                return Err(ArenaError::Invalid("ponder policy differs".into()));
            }
            policies[index] += 1;
            continue;
        }
        if !record.starts_with("V1 ")
            || fields.len() != 3
            || record.split_whitespace().count() != 4
            || fields.get("valid") != Some(&"true")
        {
            return Err(ArenaError::Invalid("ponder transition failed".into()));
        }
        match fields.get("event").copied() {
            Some("start") if !pending[index] => {
                pending[index] = true;
                audit.started += 1;
            }
            Some("hit") if pending[index] => {
                pending[index] = false;
                audit.hits += 1;
            }
            Some("stop") if pending[index] => {
                pending[index] = false;
                audit.stopped += 1;
            }
            _ => {
                return Err(ArenaError::Invalid(
                    "duplicate/stale ponder transition".into(),
                ));
            }
        }
    }
    if policies != [2, 2] || pending.iter().any(|v| *v) {
        return Err(ArenaError::Invalid(
            "two-game ponder policy/drain witnesses incomplete".into(),
        ));
    }
    Ok(audit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hardware(model: Option<&str>, vram_bytes: u64) -> MatchHardware {
        MatchHardware {
            cpu_cores: vec![vec![0], vec![2, 3]],
            gpus: vec![GpuCapacity {
                id: "GPU-a".into(),
                model: model.map(str::to_owned),
                vram_bytes,
            }],
        }
    }

    fn probe() -> MatchHardwareProbe {
        MatchHardwareProbe {
            hardware: hardware(Some("NVIDIA L4"), 24 << 30),
            gpu_inventory_known: true,
            gpu_probe_error: None,
        }
    }

    #[test]
    fn gpu_inventory_preserves_actual_model_uuid_and_integral_mib_capacity() {
        let inventory = parse_gpu_inventory(
            b"GPU-a, NVIDIA L4, 23034\r\nGPU-b, NVIDIA GeForce RTX 4090, 24564\r\n",
        )
        .unwrap();
        assert_eq!(inventory.len(), 2);
        assert_eq!(inventory[0].id, "GPU-a");
        assert_eq!(inventory[0].model.as_deref(), Some("NVIDIA L4"));
        assert_eq!(inventory[0].vram_bytes, 23034 * 1024 * 1024);
        assert_eq!(
            inventory[1].model.as_deref(),
            Some("NVIDIA GeForce RTX 4090")
        );
        assert_eq!(inventory[1].vram_bytes, 24564 * 1024 * 1024);
    }

    #[test]
    fn gpu_inventory_rejects_malformed_empty_and_duplicate_rows() {
        for text in [
            "",
            "GPU-a, 24564\n",
            "GPU-a, NVIDIA L4, 24564, extra\n",
            "GPU-a, NVIDIA L4, 24564,\n",
            "GPU-a, NVIDIA L4, 24564\n\n",
            ", NVIDIA L4, 24564\n",
            "GPU-a, , 24564\n",
            "GPU-a, \"NVIDIA L4\", 24564\n",
            "GPU a, NVIDIA L4, 24564\n",
            "GPU-a, NVIDIA\tL4, 24564\n",
            "GPU-a, NVIDIA\0L4, 24564\n",
            "GPU-a, NVIDIA L4, 24564\nGPU-a, NVIDIA L4, 24564\n",
        ] {
            assert!(parse_gpu_inventory(text.as_bytes()).is_err(), "{text:?}");
        }
        assert!(parse_gpu_inventory(b"GPU-a, NVIDIA \xff, 24564\n").is_err());
    }

    #[test]
    fn gpu_inventory_rejects_non_integral_non_finite_zero_and_overflow_capacity() {
        for capacity in [
            "",
            "0",
            "-1",
            "+1",
            "1.5",
            "NaN",
            "inf",
            "Infinity",
            "1e4",
            "24564 MiB",
            "18446744073709551615",
            "18446744073709551616",
            "17592186044416",
        ] {
            let row = format!("GPU-a, NVIDIA L4, {capacity}\n");
            assert!(parse_gpu_inventory(row.as_bytes()).is_err(), "{capacity}");
        }
    }

    #[test]
    fn gpu_inventory_limits_bytes_rows_and_identity_lengths_before_acceptance() {
        let rows: String = (0..16)
            .map(|id| format!("GPU-{id}, NVIDIA L4, 24564\n"))
            .collect();
        assert_eq!(parse_gpu_inventory(rows.as_bytes()).unwrap().len(), 16);
        assert!(
            parse_gpu_inventory(format!("{rows}GPU-16, NVIDIA L4, 24564\n").as_bytes()).is_err()
        );
        assert!(parse_gpu_inventory(&vec![b'x'; GPU_INVENTORY_MAX_BYTES + 1]).is_err());
        assert!(
            parse_gpu_inventory(format!("{}, NVIDIA L4, 24564\n", "G".repeat(81)).as_bytes())
                .is_err()
        );
        assert!(
            parse_gpu_inventory(format!("GPU-a, {}, 24564\n", "N".repeat(257)).as_bytes()).is_err()
        );
        assert!(
            parse_gpu_inventory(format!("GPU-a, {}, 24564\n", "N".repeat(513)).as_bytes()).is_err()
        );
    }

    #[test]
    fn declared_gpu_model_requires_the_observed_model_for_the_same_device() {
        let declared = hardware(Some("NVIDIA L4"), 24 << 30);
        assert!(validate_match_hardware(&declared, &probe()).is_ok());
        assert!(validate_match_hardware(&hardware(Some("NVIDIA L4"), 16 << 30), &probe()).is_err());
        assert!(validate_match_hardware(&hardware(Some("NVIDIA L4"), 32 << 30), &probe()).is_err());
        let mut actual = probe();
        actual.hardware.gpus[0].vram_bytes = 25 << 30;
        assert!(validate_match_hardware(&declared, &actual).is_err());
        actual.hardware.gpus[0].vram_bytes = 23 << 30;
        assert!(validate_match_hardware(&declared, &actual).is_err());
        actual.hardware.gpus[0].vram_bytes = 24 << 30;
        actual.hardware.gpus[0].model = Some("NVIDIA GeForce RTX 4090".into());
        assert!(validate_match_hardware(&declared, &actual).is_err());
        actual.hardware.gpus[0].model = Some("NVIDIA l4".into());
        assert!(validate_match_hardware(&declared, &actual).is_err());
        actual.hardware.gpus[0].model = None;
        assert!(validate_match_hardware(&declared, &actual).is_err());
        actual.hardware.gpus[0].model = Some("NVIDIA L4".into());
        actual.hardware.gpus[0].id = "GPU-b".into();
        assert!(validate_match_hardware(&declared, &actual).is_err());
    }

    #[test]
    fn missing_unknown_or_insufficient_gpu_inventory_never_trusts_declaration() {
        let declared = hardware(Some("NVIDIA L4"), 24 << 30);
        let mut actual = probe();
        actual.gpu_inventory_known = false;
        assert!(validate_match_hardware(&declared, &actual).is_err());
        actual.gpu_inventory_known = true;
        actual.hardware.gpus[0].vram_bytes = 23 << 30;
        assert!(validate_match_hardware(&declared, &actual).is_err());
        actual.hardware.gpus.clear();
        assert!(validate_match_hardware(&declared, &actual).is_err());
    }

    #[test]
    fn historical_model_omission_preserves_uuid_capacity_and_cpu_checks() {
        let mut declared = hardware(None, 16 << 30);
        let mut actual = probe();
        assert!(validate_match_hardware(&declared, &actual).is_ok());
        actual.hardware.gpus[0].model = None;
        assert!(validate_match_hardware(&declared, &actual).is_ok());
        actual.hardware.gpus[0].vram_bytes = 15 << 30;
        assert!(validate_match_hardware(&declared, &actual).is_err());
        actual.hardware.gpus[0].vram_bytes = 24 << 30;
        actual.hardware.gpus[0].id = "GPU-b".into();
        assert!(validate_match_hardware(&declared, &actual).is_err());
        declared.gpus.clear();
        actual.gpu_inventory_known = false;
        assert!(validate_match_hardware(&declared, &actual).is_ok());
        declared.cpu_cores = vec![vec![2], vec![3]];
        assert!(validate_match_hardware(&declared, &actual).is_err());
        declared.cpu_cores = vec![vec![9]];
        assert!(validate_match_hardware(&declared, &actual).is_err());
    }
}
