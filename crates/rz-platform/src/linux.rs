//! Current-thread observations and the explicit bounded libc affinity boundary.
use crate::placement::{ThreadAffinityAccess, ThreadAffinityMask, ThreadAffinitySnapshot};
use crate::{LinuxThreadAffinity, PlatformCpuError};
use std::collections::BTreeSet;
use std::io::Read;

const MAX_AFFINITY_CPUS: usize = 4096;
const MASK_WORDS: usize = MAX_AFFINITY_CPUS / usize::BITS as usize;

extern "C" {
    fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut usize) -> i32;
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const usize) -> i32;
}

fn affinity_native_error(function: &'static str) -> PlatformCpuError {
    match std::io::Error::last_os_error().raw_os_error() {
        Some(os_code) => PlatformCpuError::NativeCall {
            function,
            os_code: os_code as u32,
        },
        None => PlatformCpuError::ReadUnavailable,
    }
}
fn native_current_mask() -> Result<Vec<u16>, PlatformCpuError> {
    let mut words = [0usize; MASK_WORDS];
    // SAFETY: pid=0 addresses only the calling thread. The initialized native
    // unsigned-long array is aligned and writable for exactly cpusetsize bytes.
    // The libc interface returns 0, unlike the raw syscall's byte-count return.
    if unsafe { sched_getaffinity(0, std::mem::size_of_val(&words), words.as_mut_ptr()) } != 0 {
        return Err(affinity_native_error("sched_getaffinity"));
    }
    let mut cpus = Vec::new();
    for cpu in 0..MAX_AFFINITY_CPUS {
        if words[cpu / usize::BITS as usize] & (1usize << (cpu % usize::BITS as usize)) != 0 {
            cpus.push(cpu as u16);
        }
    }
    if cpus.is_empty() {
        return Err(PlatformCpuError::InvalidNativeBuffer);
    }
    Ok(cpus)
}

pub(crate) struct NativeThreadAffinity;
impl ThreadAffinityAccess for NativeThreadAffinity {
    fn accepts(&self, mask: &ThreadAffinityMask) -> bool {
        matches!(mask, ThreadAffinityMask::Linux { .. })
    }
    fn observe(&mut self) -> Result<ThreadAffinitySnapshot, PlatformCpuError> {
        // Reuse the existing bounded procfs identity/readback and require the
        // independent native mask to agree between both stable observations.
        let before = observe_linux_thread_affinity()?;
        let cpus = native_current_mask()?;
        let after = observe_linux_thread_affinity()?;
        if before != after || before.cpu_ids != cpus {
            return Err(PlatformCpuError::ChangedDuringProbe);
        }
        Ok(ThreadAffinitySnapshot::Linux(before))
    }
    fn set(
        &mut self,
        mask: &ThreadAffinityMask,
    ) -> Result<Option<ThreadAffinityMask>, PlatformCpuError> {
        let ThreadAffinityMask::Linux { cpu_ids } = mask else {
            return Err(PlatformCpuError::UnsupportedArchitecture);
        };
        if cpu_ids.is_empty() || cpu_ids.len() > MAX_AFFINITY_CPUS {
            return Err(PlatformCpuError::InvalidNativeBuffer);
        }
        let mut words = [0usize; MASK_WORDS];
        for &cpu in cpu_ids {
            let cpu = usize::from(cpu);
            if cpu >= MAX_AFFINITY_CPUS {
                return Err(PlatformCpuError::NativeBufferBudget);
            }
            words[cpu / usize::BITS as usize] |= 1usize << (cpu % usize::BITS as usize);
        }
        // SAFETY: pid=0 is the calling thread, never the process leader or a
        // foreign tid. The native unsigned-long bitset has the requested size
        // and remains alive for this synchronous libc call. No pointer escapes.
        if unsafe { sched_setaffinity(0, std::mem::size_of_val(&words), words.as_ptr()) } != 0 {
            return Err(affinity_native_error("sched_setaffinity"));
        }
        Ok(None)
    }
}

fn read_status() -> Result<String, PlatformCpuError> {
    let file = std::fs::File::open("/proc/thread-self/status")
        .map_err(|_| PlatformCpuError::ReadUnavailable)?;
    let mut bytes = Vec::new();
    file.take(64 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| PlatformCpuError::ReadUnavailable)?;
    if bytes.len() > 64 * 1024 {
        return Err(PlatformCpuError::NativeBufferBudget);
    }
    String::from_utf8(bytes).map_err(|_| PlatformCpuError::InvalidText)
}

fn parse_status(raw: &str) -> Result<(u64, Vec<u16>), PlatformCpuError> {
    let one = |key: &str| -> Result<&str, PlatformCpuError> {
        let mut rows = raw.lines().filter_map(|line| line.strip_prefix(key));
        let row = rows.next().ok_or(PlatformCpuError::InvalidText)?.trim();
        if rows.next().is_some() {
            return Err(PlatformCpuError::InvalidText);
        }
        Ok(row)
    };
    let thread_id = one("Pid:")?
        .parse::<u64>()
        .map_err(|_| PlatformCpuError::InvalidText)?;
    if thread_id == 0 {
        return Err(PlatformCpuError::InvalidText);
    }
    let mut cpus = BTreeSet::new();
    let number = |s: &str| -> Result<u16, PlatformCpuError> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(PlatformCpuError::InvalidText);
        }
        let n = s
            .parse::<u16>()
            .map_err(|_| PlatformCpuError::InvalidText)?;
        if n >= 4096 {
            return Err(PlatformCpuError::NativeBufferBudget);
        }
        Ok(n)
    };
    for range in one("Cpus_allowed_list:")?.split(',') {
        let (lo, hi) = if let Some((a, b)) = range.split_once('-') {
            (number(a)?, number(b)?)
        } else {
            let n = number(range)?;
            (n, n)
        };
        if lo > hi {
            return Err(PlatformCpuError::InvalidText);
        }
        for cpu in lo..=hi {
            if !cpus.insert(cpu) {
                return Err(PlatformCpuError::InvalidText);
            }
        }
    }
    if cpus.is_empty() {
        return Err(PlatformCpuError::InvalidText);
    }
    Ok((thread_id, cpus.into_iter().collect()))
}

pub fn observe_linux_thread_affinity() -> Result<LinuxThreadAffinity, PlatformCpuError> {
    let before = parse_status(&read_status()?)?;
    let after = parse_status(&read_status()?)?;
    if before != after {
        return Err(PlatformCpuError::ChangedDuringProbe);
    }
    Ok(LinuxThreadAffinity {
        process_id: std::process::id(),
        thread_id: before.0,
        cpu_ids: before.1,
        identity_stable: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn current_thread_parser_preserves_large_cpu_ids_and_rejects_ambiguity() {
        assert_eq!(
            parse_status("Pid:\t3\nCpus_allowed_list:\t1,64-65").unwrap(),
            (3, vec![1, 64, 65])
        );
        for raw in [
            "Pid:3\nPid:4\nCpus_allowed_list:0",
            "Pid:0\nCpus_allowed_list:0",
            "Pid:3\nCpus_allowed_list:0,0",
            "Pid:3\nCpus_allowed_list:4096",
        ] {
            assert!(parse_status(raw).is_err());
        }
    }
}
