//! Locked simultaneous-engine placement. Weights are a heuristic, not a speed estimate.
use crate::{ArtifactRef, ManifestError, NativeEngineRole};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EngineComputeKind { Cpu, Gpu, Hybrid }
impl EngineComputeKind {
    fn cpu_weight(self) -> u32 { match self { Self::Cpu => 4, Self::Hybrid => 2, Self::Gpu => 1 } }
    fn gpu_weight(self) -> u32 { match self { Self::Cpu => 0, Self::Hybrid => 2, Self::Gpu => 4 } }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResourceSharing { Isolated, Shared }
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GpuCapacity {
    /// Stable CUDA UUID preferred; numeric IDs must use the same host enumeration.
    pub id: String,
    pub vram_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MatchHardware {
    /// Each entry is one physical core's available SMT siblings. Never split in isolated mode.
    pub cpu_cores: Vec<Vec<u32>>,
    pub gpus: Vec<GpuCapacity>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EngineResourceRequest {
    pub role: NativeEngineRole,
    pub kind: EngineComputeKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_ids: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_ids: Option<Vec<String>>,
    /// Per assigned device reservation. A declaration, not a CUDA allocator cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_memory_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_weight: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_weight: Option<u32>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MatchExecutionV1 {
    pub ponder: bool,
    pub sharing: ResourceSharing,
    pub hardware: MatchHardware,
    pub engines: [EngineResourceRequest; 2],
    /// Pinned model-pair binary exposing engine-exec; no shell/taskset dependency.
    pub executor: ArtifactRef,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct EngineResourceAllocation {
    pub role: NativeEngineRole,
    pub kind: EngineComputeKind,
    pub cpu_ids: Vec<u32>,
    pub threads: u32,
    pub gpu_ids: Vec<String>,
    pub gpu_memory_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct MatchResourcePlan {
    pub policy_version: u32,
    pub sharing: ResourceSharing,
    pub runner_cpu_ids: Vec<u32>,
    pub engines: [EngineResourceAllocation; 2],
    pub shared_gpu_compute: bool,
}
fn check(ok: bool, reason: &str) -> Result<(), ManifestError> {
    if ok { Ok(()) } else { Err(ManifestError::Integrity(format!("match resources: {reason}"))) }
}
impl MatchExecutionV1 {
    pub fn plan(&self) -> Result<MatchResourcePlan, ManifestError> {
        self.executor.validate()?;
        let cores = &self.hardware.cpu_cores;
        check((3..=256).contains(&cores.len()), "need a runner core and at least one core per engine")?;
        let all: BTreeSet<_> = cores.iter().flatten().copied().collect();
        check(cores.iter().all(|c| !c.is_empty() && c.len() <= 8 && c.iter().all(|id| *id < 4096))
            && all.len() == cores.iter().map(Vec::len).sum::<usize>(), "invalid/duplicate CPU topology")?;
        check(self.hardware.gpus.len() <= 16 && self.hardware.gpus.iter().all(|g|
            !g.id.is_empty() && g.id.len() <= 80 && g.id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            && g.vram_bytes > 0)
            && self.hardware.gpus.iter().map(|g| &g.id).collect::<BTreeSet<_>>().len() == self.hardware.gpus.len(), "invalid GPU inventory")?;
        check(self.engines[0].role != self.engines[1].role, "duplicate roles")?;
        for r in &self.engines {
            check(r.cpu_weight.is_none_or(|w| (1..=64).contains(&w))
                && r.gpu_weight.is_none_or(|w| (1..=64).contains(&w)), "invalid weight")?;
        }
        // Reserve a whole unused core for runner I/O; an explicit engine pin is never stolen.
        let runner = cores.iter().find(|core| !self.engines.iter().any(|r|
            r.cpu_ids.as_ref().is_some_and(|ids| core.iter().any(|id| ids.contains(id)))))
            .ok_or_else(|| ManifestError::Integrity("match resources: no runner core left".into()))?.clone();
        let mut free: Vec<_> = cores.iter().filter(|c| **c != runner).cloned().collect();
        let mut cpus: [Vec<u32>; 2] = [vec![], vec![]];
        for (i, r) in self.engines.iter().enumerate() {
            if let Some(ids) = &r.cpu_ids {
                check(!ids.is_empty() && ids.iter().collect::<BTreeSet<_>>().len() == ids.len()
                    && ids.iter().all(|id| all.contains(id) && !runner.contains(id)), "invalid explicit CPU pin")?;
                check(cores.iter().all(|core| !core.iter().any(|id| ids.contains(id)) || core.iter().all(|id| ids.contains(id))), "explicit pin splits SMT siblings")?;
                cpus[i] = ids.clone();
                free.retain(|core| !core.iter().any(|id| ids.contains(id)));
            }
        }
        if self.sharing == ResourceSharing::Isolated {
            check(!cpus[0].iter().any(|id| cpus[1].contains(id)), "isolated CPU overlap")?;
        }
        let automatic: Vec<_> = (0..2).filter(|i| self.engines[*i].cpu_ids.is_none()).collect();
        check(free.len() >= automatic.len(), "no core for automatic engine")?;
        if !automatic.is_empty() {
            let mut counts = [0u32; 2];
            for i in &automatic { cpus[*i].extend(free.remove(0)); counts[*i] = 1; }
            for core in free {
                let i = *automatic.iter().min_by_key(|i| {
                    let w = self.engines[**i].cpu_weight.unwrap_or(self.engines[**i].kind.cpu_weight());
                    (counts[**i] * 65536 / w, **i)
                }).unwrap();
                cpus[i].extend(core); counts[i] += 1;
            }
        }
        let mut gpu_ids: [Vec<String>; 2] = [vec![], vec![]];
        for (i, r) in self.engines.iter().enumerate() {
            if let Some(ids) = &r.gpu_ids {
                check(ids.iter().collect::<BTreeSet<_>>().len() == ids.len()
                    && ids.iter().all(|id| self.hardware.gpus.iter().any(|g| &g.id == id)), "unknown/duplicate explicit GPU")?;
                gpu_ids[i] = ids.clone();
            }
            check(r.kind != EngineComputeKind::Cpu || (gpu_ids[i].is_empty() && r.gpu_memory_bytes.unwrap_or(0) == 0), "CPU engine requests GPU resources")?;
        }
        let mut gpu_order: Vec<_> = self.hardware.gpus.iter().collect();
        gpu_order.sort_by_key(|g| (std::cmp::Reverse(g.vram_bytes), &g.id));
        // Give the more GPU-oriented engine first choice; CPU weighting is deliberately different.
        let mut order = [0usize, 1];
        order.sort_by_key(|i| std::cmp::Reverse(self.engines[*i].gpu_weight.unwrap_or(self.engines[*i].kind.gpu_weight())));
        for i in order {
            let r = &self.engines[i];
            if r.kind != EngineComputeKind::Cpu && r.gpu_ids.is_none() {
                let g = gpu_order.iter().find(|g| self.sharing == ResourceSharing::Shared
                    || !gpu_ids.iter().any(|ids| ids.contains(&g.id)))
                    .ok_or_else(|| ManifestError::Integrity("match resources: insufficient GPUs; choose shared explicitly or add devices".into()))?;
                gpu_ids[i].push(g.id.clone());
            }
            check(r.kind == EngineComputeKind::Cpu || !gpu_ids[i].is_empty(), "GPU/Hybrid engine needs a GPU")?;
        }
        let overlap = gpu_ids[0].iter().any(|id| gpu_ids[1].contains(id));
        check(self.sharing == ResourceSharing::Shared || !overlap, "isolated GPU overlap")?;
        let mut memory = [0u64; 2];
        for (i, r) in self.engines.iter().enumerate() {
            if r.kind != EngineComputeKind::Cpu {
                memory[i] = if let Some(bytes) = r.gpu_memory_bytes { bytes } else {
                    gpu_ids[i].iter().map(|id| {
                        let cap = self.hardware.gpus.iter().find(|g| &g.id == id).unwrap().vram_bytes;
                        let other = 1-i;
                        if !gpu_ids[other].contains(id) { cap }
                        else if let Some(reserved) = self.engines[other].gpu_memory_bytes { cap.saturating_sub(reserved) }
                        else {
                            let w = u64::from(r.gpu_weight.unwrap_or(r.kind.gpu_weight()));
                            let ow = u64::from(self.engines[other].gpu_weight.unwrap_or(self.engines[other].kind.gpu_weight()));
                            cap / (w+ow) * w
                        }
                    }).min().unwrap()
                };
                check(memory[i] > 0, "zero GPU reservation")?;
            }
        }
        for gpu in &self.hardware.gpus {
            let total = (0..2).filter(|i| gpu_ids[*i].contains(&gpu.id)).try_fold(0u64, |n,i| n.checked_add(memory[i]));
            check(total.is_some_and(|n| n <= gpu.vram_bytes), "GPU memory reservations exceed capacity")?;
        }
        let allocations: Vec<_> = (0..2).map(|i| {
            cpus[i].sort_unstable();
            let r = &self.engines[i];
            let threads = r.threads.unwrap_or(match r.kind { EngineComputeKind::Cpu => cpus[i].len() as u32, EngineComputeKind::Hybrid => 2.min(cpus[i].len() as u32), EngineComputeKind::Gpu => 1 });
            EngineResourceAllocation { role:r.role, kind:r.kind, cpu_ids:cpus[i].clone(), threads, gpu_ids:gpu_ids[i].clone(), gpu_memory_bytes:memory[i] }
        }).collect();
        check(allocations.iter().all(|a| a.threads > 0 && a.threads as usize <= a.cpu_ids.len()), "thread count exceeds placement")?;
        if self.sharing == ResourceSharing::Shared {
            let used: BTreeSet<_> = allocations.iter().flat_map(|a| &a.cpu_ids).collect();
            check(allocations.iter().map(|a| a.threads as usize).sum::<usize>() <= used.len(), "shared CPU thread reservations oversubscribe cores")?;
        }
        Ok(MatchResourcePlan { policy_version:1, sharing:self.sharing, runner_cpu_ids:runner, engines:allocations.try_into().unwrap(), shared_gpu_compute:overlap })
    }
}
