//! Small read-only OS observations. No model, search, scheduler, or settings.
//!
//! Native calls are confined to Windows kernel32 and keep requested resource
//! declarations out of the returned facts. No affinity or quota is installed.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::{observe_windows_cpu, observe_windows_thread_affinity};

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::{observe_linux_thread_affinity, LinuxThreadAffinity};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlatformCpuError {
    NativeCall {
        function: &'static str,
        os_code: u32,
    },
    InvalidNativeBuffer,
    NativeBufferBudget,
    AmbiguousProcessGroups,
    UnsupportedArchitecture,
    UnsupportedJobRatePolicy,
    InvalidText,
    ReadUnavailable,
    ChangedDuringProbe,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowsDirectCpuQuota {
    Unlimited,
    HardCap { rate_per_10000: u16 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowsCpuSnapshot {
    pub process_id: u32,
    pub caller_thread_id: u64,
    pub identity_stable: bool,
    pub process_group_masks: Result<Vec<(u16, u64)>, PlatformCpuError>,
    pub caller_group_mask: Result<(u16, u64), PlatformCpuError>,
    pub physical_core_group_masks: Result<Vec<Vec<(u16, u64)>>, PlatformCpuError>,
    pub immediate_job_cpu_quota: Result<WindowsDirectCpuQuota, PlatformCpuError>,
    /// QueryInformationJobObject(NULL) reads only the immediate Job Object.
    pub effective_job_hierarchy_known: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowsThreadAffinity {
    pub process_id: u32,
    pub thread_id: u64,
    pub group_mask: (u16, u64),
    pub identity_stable: bool,
}
