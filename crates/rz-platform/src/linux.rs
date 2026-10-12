//! Current thread affinity through fixed, bounded procfs metadata reads.
use crate::PlatformCpuError;
use std::collections::BTreeSet;
use std::io::Read;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinuxThreadAffinity {
    pub process_id: u32,
    pub thread_id: u64,
    pub cpu_ids: Vec<u16>,
    pub identity_stable: bool,
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
