//! Auditable kernel32 CPU-query boundary. All handles are current-process/thread
//! pseudo handles, and all calls are read-only. Variable native buffers are
//! bounded, aligned, length checked, then decoded without pointer casts.
use crate::{PlatformCpuError, WindowsCpuSnapshot, WindowsDirectCpuQuota, WindowsThreadAffinity};
use std::ffi::c_void;

type Handle = *mut c_void;
const RELATION_PROCESSOR_CORE: u32 = 0;
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
const MAX_NATIVE_BYTES: usize = 512 * 1024;
const MAX_GROUPS: usize = 64;
const MAX_CORES: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GroupAffinity {
    mask: usize,
    group: u16,
    reserved: [u16; 3],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct JobCpuRate {
    flags: u32,
    rate: u32,
}

#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> Handle;
    fn GetCurrentThread() -> Handle;
    fn GetCurrentProcessId() -> u32;
    fn GetCurrentThreadId() -> u32;
    fn GetActiveProcessorGroupCount() -> u16;
    fn GetLastError() -> u32;
    fn GetProcessAffinityMask(
        process: Handle,
        process_mask: *mut usize,
        system_mask: *mut usize,
    ) -> i32;
    fn GetProcessGroupAffinity(process: Handle, group_count: *mut u16, groups: *mut u16) -> i32;
    fn GetThreadGroupAffinity(thread: Handle, affinity: *mut GroupAffinity) -> i32;
    fn GetLogicalProcessorInformationEx(
        relationship: u32,
        buffer: *mut c_void,
        bytes: *mut u32,
    ) -> i32;
    fn IsProcessInJob(process: Handle, job: Handle, result: *mut i32) -> i32;
    fn QueryInformationJobObject(
        job: Handle,
        info_class: u32,
        info: *mut c_void,
        info_bytes: u32,
        returned: *mut u32,
    ) -> i32;
}

fn native_error(function: &'static str) -> PlatformCpuError {
    // SAFETY: GetLastError reads the calling thread's last error with no pointer.
    PlatformCpuError::NativeCall {
        function,
        os_code: unsafe { GetLastError() },
    }
}

fn process_affinity(process: Handle) -> Result<Vec<(u16, u64)>, PlatformCpuError> {
    // SAFETY: a scalar read-only system query. On multi-group Windows 11 the
    // affinity APIs expose primary-group masks without proving a complete
    // eligible set. This implementation does not install group restrictions.
    if unsafe { GetActiveProcessorGroupCount() } != 1 {
        return Err(PlatformCpuError::AmbiguousProcessGroups);
    }
    let mut count = MAX_GROUPS as u16;
    let mut groups = [0u16; MAX_GROUPS];
    // SAFETY: count is initialized to the length of groups; all buffers outlive
    // the query. The current-process pseudo handle remains valid throughout.
    if unsafe { GetProcessGroupAffinity(process, &mut count, groups.as_mut_ptr()) } == 0 {
        return Err(native_error("GetProcessGroupAffinity"));
    }
    // Windows 11 can span multiple processor groups. GetProcessAffinityMask
    // cannot supply complete per-group masks, so never invent that readback.
    if count != 1 || groups[0] >= MAX_GROUPS as u16 {
        return Err(PlatformCpuError::AmbiguousProcessGroups);
    }
    let mut process_mask = 0usize;
    let mut system_mask = 0usize;
    // SAFETY: two initialized usize out-pointers have the API's ULONG_PTR size.
    if unsafe { GetProcessAffinityMask(process, &mut process_mask, &mut system_mask) } == 0 {
        return Err(native_error("GetProcessAffinityMask"));
    }
    if process_mask == 0 || process_mask & !system_mask != 0 {
        return Err(PlatformCpuError::InvalidNativeBuffer);
    }
    Ok(vec![(groups[0], process_mask as u64)])
}

fn thread_affinity(thread: Handle) -> Result<(u16, u64), PlatformCpuError> {
    // SAFETY: scalar system-group count. A primary-group-only readback must
    // never be represented as complete thread affinity on a multi-group host.
    if unsafe { GetActiveProcessorGroupCount() } != 1 {
        return Err(PlatformCpuError::AmbiguousProcessGroups);
    }
    let mut value = GroupAffinity::default();
    // SAFETY: value is repr(C) with KAFFINITY/WORD/WORD[3] layout; initialized
    // storage is writable for exactly sizeof(GROUP_AFFINITY).
    if unsafe { GetThreadGroupAffinity(thread, &mut value) } == 0 {
        return Err(native_error("GetThreadGroupAffinity"));
    }
    if value.mask == 0 || value.group >= MAX_GROUPS as u16 {
        return Err(PlatformCpuError::InvalidNativeBuffer);
    }
    Ok((value.group, value.mask as u64))
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16, PlatformCpuError> {
    let raw = bytes
        .get(offset..offset + 2)
        .ok_or(PlatformCpuError::InvalidNativeBuffer)?;
    Ok(u16::from_ne_bytes([raw[0], raw[1]]))
}
fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, PlatformCpuError> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or(PlatformCpuError::InvalidNativeBuffer)?;
    Ok(u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]))
}
fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, PlatformCpuError> {
    let raw = bytes
        .get(offset..offset + 8)
        .ok_or(PlatformCpuError::InvalidNativeBuffer)?;
    Ok(u64::from_ne_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
}

fn decode_cores(bytes: &[u8]) -> Result<Vec<Vec<(u16, u64)>>, PlatformCpuError> {
    let mut result = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        if result.len() >= MAX_CORES {
            return Err(PlatformCpuError::NativeBufferBudget);
        }
        let relationship = u32_at(bytes, cursor)?;
        let size = u32_at(bytes, cursor + 4)? as usize;
        if relationship != RELATION_PROCESSOR_CORE || size < 48 || size % 8 != 0 {
            return Err(PlatformCpuError::InvalidNativeBuffer);
        }
        let row = bytes
            .get(
                cursor
                    ..cursor
                        .checked_add(size)
                        .ok_or(PlatformCpuError::InvalidNativeBuffer)?,
            )
            .ok_or(PlatformCpuError::InvalidNativeBuffer)?;
        // Header 8 bytes + PROCESSOR_RELATIONSHIP's flags/efficiency/reserved/
        // GroupCount 24 bytes; 64-bit GROUP_AFFINITY entries are 16 bytes each.
        let count = u16_at(row, 30)? as usize;
        // RelationProcessorCore has exactly one group mask in this API.
        if count != 1 || 32 + count * 16 != size {
            return Err(PlatformCpuError::InvalidNativeBuffer);
        }
        let mut groups = Vec::with_capacity(count);
        for group_index in 0..count {
            let offset = 32 + group_index * 16;
            let mask = u64_at(row, offset)?;
            let group = u16_at(row, offset + 8)?;
            if mask == 0
                || group >= MAX_GROUPS as u16
                || groups.iter().any(|(old, _)| *old == group)
            {
                return Err(PlatformCpuError::InvalidNativeBuffer);
            }
            groups.push((group, mask));
        }
        result.push(groups);
        cursor += size;
    }
    if result.is_empty() {
        return Err(PlatformCpuError::InvalidNativeBuffer);
    }
    Ok(result)
}

fn physical_cores() -> Result<Vec<Vec<(u16, u64)>>, PlatformCpuError> {
    if std::mem::size_of::<usize>() != 8 {
        return Err(PlatformCpuError::UnsupportedArchitecture);
    }
    let mut bytes = 0u32;
    // SAFETY: NULL with a zero length is the documented required-size query.
    if unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut bytes)
    } != 0
        || unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER
    {
        return Err(native_error("GetLogicalProcessorInformationEx:size"));
    }
    if bytes == 0 || bytes as usize > MAX_NATIVE_BYTES {
        return Err(PlatformCpuError::NativeBufferBudget);
    }
    let capacity = (bytes as usize).div_ceil(8);
    let mut storage = vec![0u64; capacity];
    let reserved_bytes = bytes;
    // SAFETY: storage is 8-byte aligned, initialized and at least bytes long;
    // the API can write only the supplied byte length and returns actual size.
    if unsafe {
        GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE,
            storage.as_mut_ptr().cast(),
            &mut bytes,
        )
    } == 0
    {
        return Err(native_error("GetLogicalProcessorInformationEx:data"));
    }
    if bytes == 0 || bytes > reserved_bytes {
        return Err(PlatformCpuError::InvalidNativeBuffer);
    }
    // SAFETY: the initialized u64 storage lives through decode_cores; the
    // returned length was checked against the original allocated byte extent.
    let raw = unsafe { std::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), bytes as usize) };
    decode_cores(raw)
}

fn immediate_job_quota(process: Handle) -> Result<WindowsDirectCpuQuota, PlatformCpuError> {
    let mut in_job = 0i32;
    // SAFETY: read-only query to an initialized BOOL out-pointer; NULL queries
    // whether this current-process handle belongs to any Job Object.
    if unsafe { IsProcessInJob(process, std::ptr::null_mut(), &mut in_job) } == 0 {
        return Err(native_error("IsProcessInJob"));
    }
    if in_job == 0 {
        return Ok(WindowsDirectCpuQuota::Unlimited);
    }
    let mut rate = JobCpuRate::default();
    let mut returned = 0u32;
    // SAFETY: class 15's repr(C) pair of DWORDs is 8 bytes. NULL reads the
    // calling process's immediate Job, not a claimed effective ancestor limit.
    if unsafe {
        QueryInformationJobObject(
            std::ptr::null_mut(),
            15,
            (&mut rate as *mut JobCpuRate).cast(),
            8,
            &mut returned,
        )
    } == 0
    {
        return Err(native_error("QueryInformationJobObject:cpu_rate"));
    }
    if returned != 8 {
        return Err(PlatformCpuError::InvalidNativeBuffer);
    }
    if rate.flags == 0 {
        return Ok(WindowsDirectCpuQuota::Unlimited);
    }
    if rate.flags & !0x1f == 0
        && rate.flags & 1 != 0
        && rate.flags & 4 != 0
        && rate.flags & (2 | 16) == 0
        && (1..=10000).contains(&rate.rate)
    {
        return Ok(WindowsDirectCpuQuota::HardCap {
            rate_per_10000: rate.rate as u16,
        });
    }
    Err(PlatformCpuError::UnsupportedJobRatePolicy)
}

pub fn observe_windows_cpu() -> WindowsCpuSnapshot {
    // SAFETY: these return current-process/thread pseudo handles and scalar
    // identities; no opened handle requires closing and no settings change.
    let (process, thread, process_id, thread_id) = unsafe {
        (
            GetCurrentProcess(),
            GetCurrentThread(),
            GetCurrentProcessId(),
            GetCurrentThreadId(),
        )
    };
    let process_before = process_affinity(process);
    let thread_before = thread_affinity(thread);
    let cores = physical_cores();
    let quota = immediate_job_quota(process);
    let process_after = process_affinity(process);
    let thread_after = thread_affinity(thread);
    let cores_after = physical_cores();
    let quota_after = immediate_job_quota(process);
    // SAFETY: scalar identity reads, used only to bracket the sequential probe.
    let ids_unchanged =
        unsafe { GetCurrentProcessId() == process_id && GetCurrentThreadId() == thread_id };
    WindowsCpuSnapshot {
        process_id,
        caller_thread_id: u64::from(thread_id),
        identity_stable: ids_unchanged
            && process_before == process_after
            && thread_before == thread_after
            && cores == cores_after
            && quota == quota_after,
        process_group_masks: process_before,
        caller_group_mask: thread_before,
        physical_core_group_masks: cores,
        immediate_job_cpu_quota: quota,
        effective_job_hierarchy_known: false,
    }
}

/// Each worker calls this itself at its ready boundary. No borrowed cross-
/// thread HANDLE can substitute for the worker's actual current affinity.
pub fn observe_windows_thread_affinity() -> Result<WindowsThreadAffinity, PlatformCpuError> {
    // SAFETY: scalar identities and the current-thread pseudo handle have no
    // lifetime beyond this synchronous query and require no CloseHandle.
    let (thread, process_id, thread_id) = unsafe {
        (
            GetCurrentThread(),
            GetCurrentProcessId(),
            GetCurrentThreadId(),
        )
    };
    let before = thread_affinity(thread)?;
    let after = thread_affinity(thread)?;
    // SAFETY: scalar identity reads bracket this current-thread observation.
    if before != after
        || unsafe { GetCurrentProcessId() != process_id || GetCurrentThreadId() != thread_id }
    {
        return Err(PlatformCpuError::ChangedDuringProbe);
    }
    Ok(WindowsThreadAffinity {
        process_id,
        thread_id: u64::from(thread_id),
        group_mask: before,
        identity_stable: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(group: u16, mask: u64) -> Vec<u8> {
        let mut bytes = vec![0; 48];
        bytes[4..8].copy_from_slice(&48u32.to_ne_bytes());
        bytes[30..32].copy_from_slice(&1u16.to_ne_bytes());
        bytes[32..40].copy_from_slice(&mask.to_ne_bytes());
        bytes[40..42].copy_from_slice(&group.to_ne_bytes());
        bytes
    }
    #[test]
    fn native_core_decoder_bounds_every_row_and_group() {
        let mut bytes = row(0, 3);
        bytes.extend(row(1, 12));
        assert_eq!(
            decode_cores(&bytes).unwrap(),
            vec![vec![(0, 3)], vec![(1, 12)]]
        );
        for raw in [vec![], vec![0; 7], row(0, 0), row(64, 1)] {
            assert!(decode_cores(&raw).is_err());
        }
        let mut truncated = row(0, 1);
        truncated.pop();
        assert!(decode_cores(&truncated).is_err());
        let mut wrong = row(0, 1);
        wrong[30..32].copy_from_slice(&2u16.to_ne_bytes());
        assert!(decode_cores(&wrong).is_err());
    }
    #[test]
    fn actual_current_process_probe_does_not_claim_nested_job_enforcement() {
        let observed = observe_windows_cpu();
        assert_eq!(observed.process_id, std::process::id());
        assert!(observed.caller_thread_id > 0);
        assert!(!observed.effective_job_hierarchy_known);
        if let Ok(cores) = observed.physical_core_group_masks {
            assert!(!cores.is_empty());
            assert!(cores
                .iter()
                .all(|groups| groups.iter().all(|(_, mask)| *mask != 0)));
        }
    }
}
