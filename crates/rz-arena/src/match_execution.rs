//! Host evidence and bounded ponder lifecycle audit for the opt-in match policy.
use crate::ArenaError;
use rz_experiments::{MatchExecutionV1, MatchHardware};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct MatchHardwareProbe {
    pub hardware: MatchHardware,
    pub gpu_inventory_known: bool,
    pub gpu_probe_error: Option<String>,
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
                "--query-gpu=uuid,memory.total".into(),
                "--format=csv,noheader,nounits".into(),
            ],
            Path::new("/tmp"),
            crate::ProcessLimits {
                wall_ms: 3000,
                shutdown_grace_ms: 500,
                max_output_bytes: 8192,
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
            let text = std::str::from_utf8(&output.stdout)
                .map_err(|_| ArenaError::Invalid("invalid GPU inventory encoding".into()))?;
            for line in text.lines() {
                let (id, memory) = line
                    .split_once(',')
                    .ok_or_else(|| ArenaError::Invalid("invalid GPU inventory row".into()))?;
                let memory = memory
                    .trim()
                    .parse::<u64>()
                    .ok()
                    .and_then(|m| m.checked_mul(1024 * 1024))
                    .filter(|m| *m > 0)
                    .ok_or_else(|| ArenaError::Invalid("invalid GPU capacity".into()))?;
                gpus.push(rz_experiments::GpuCapacity {
                    id: id.trim().into(),
                    vram_bytes: memory,
                });
            }
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
    // A declared inventory may select a subset, but cannot split one physical core
    // into several apparently isolated cores or claim a device/capacity not present.
    let mut owners = BTreeMap::new();
    for (group, declared) in execution.hardware.cpu_cores.iter().enumerate() {
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
    for gpu in &execution.hardware.gpus {
        if !probe.gpu_inventory_known
            || !probe
                .hardware
                .gpus
                .iter()
                .any(|g| g.id == gpu.id && g.vram_bytes >= gpu.vram_bytes)
        {
            return Err(ArenaError::Invalid(
                "GPU inventory/capacity is unavailable or differs; no engine started".into(),
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
    for line in text.lines() {
        if line.contains("RZ_PONDER_FAILURE_V1") {
            return Err(ArenaError::Invalid("ponder cleanup failed".into()));
        }
        let Some((_, record)) = line.split_once("RZ_PONDER_") else {
            continue;
        };
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
            if fields.get("enabled") != Some(&"true") {
                return Err(ArenaError::Invalid("ponder policy differs".into()));
            }
            policies[index] += 1;
            continue;
        }
        if !record.starts_with("V1 ") || fields.get("valid") != Some(&"true") {
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
