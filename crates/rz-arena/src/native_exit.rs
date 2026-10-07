//! Pinned Fastchess process exit and logger codecs shared by CPU/CUDA endpoints.
use crate::ArenaError;
/// Verify the fixed Linux Fastchess renderer, not an engine's self-report.
/// The caller supplies four identities from validated startup records and the
/// supervisor-owned, bounded stdout whose ArtifactRef is already retained.
/// This pinned source logs raw wait status: only decimal `0` proves exit zero.
/// Missing records (including earlier reaping), malformed records or renderer
/// changes are unsupported evidence and fail closed, never inferred as success.
pub fn validate_native_cuda_process_exit_trace(
    stdout: &[u8],
    expected_pids: &[u32],
) -> Result<(), ArenaError> {
    if expected_pids.len() != 4 {
        return Err(ArenaError::Integrity(
            "CUDA exit trace requires exactly four distinct native PIDs".into(),
        ));
    }
    validate_endpoint_exit_trace(stdout, expected_pids, 0).map(|_| ())
}

pub(crate) fn validate_endpoint_exit_trace(
    stdout: &[u8],
    expected_pids: &[u32],
    external_count: usize,
) -> Result<Vec<u32>, ArenaError> {
    const MAX_BYTES: usize = 64 * 1024 * 1024;
    const MAX_LINES: usize = 131_072;
    const MAX_LINE_BYTES: usize = 4096;
    let fail = |reason: &str| ArenaError::Integrity(reason.into());
    if !(1..=4).contains(&expected_pids.len())
        || expected_pids.len() + external_count != 4
        || expected_pids
            .iter()
            .any(|pid| *pid == 0 || *pid > i32::MAX as u32)
        || expected_pids
            .iter()
            .enumerate()
            .any(|(index, pid)| expected_pids[..index].contains(pid))
    {
        return Err(fail(
            "CUDA exit trace requires exactly four distinct native PIDs",
        ));
    }
    if stdout.is_empty() || stdout.len() > MAX_BYTES || !stdout.ends_with(b"\n") {
        return Err(fail(
            "CUDA exit trace is empty, over budget or unterminated",
        ));
    }
    let text = std::str::from_utf8(stdout).map_err(|_| fail("CUDA runner stdout is not UTF-8"))?;
    let mut seen = vec![false; expected_pids.len()];
    let mut external = Vec::new();
    for (index, line) in text.split_terminator('\n').enumerate() {
        if index >= MAX_LINES || line.len() > MAX_LINE_BYTES {
            return Err(fail("CUDA runner trace line budget exceeded"));
        }
        // Logger::readFromEngine adds its own anchored [Engine] prefix to each
        // protocol line; a quoted TRACE-looking payload cannot supply evidence.
        if !line.starts_with("[TRACE")
            || !(line.contains("Process with pid")
                || line.contains("Force terminating process with pid"))
        {
            continue;
        }
        if !line.is_ascii() {
            return Err(fail("CUDA process exit renderer is not ASCII"));
        }
        let message = cuda_exit_trace_message(line)
            .ok_or_else(|| fail("CUDA process exit renderer is malformed"))?;
        if message.starts_with("Force terminating process with pid:") {
            return Err(fail("CUDA native process required force termination"));
        }
        let (pid_text, status) = message
            .strip_prefix("Process with pid: ")
            .and_then(|value| value.split_once(" terminated with status: "))
            .ok_or_else(|| fail("CUDA native process exit record is malformed"))?;
        let pid = pid_text
            .parse::<u32>()
            .map_err(|_| fail("CUDA native process exit PID is malformed"))?;
        if pid_text != pid.to_string() {
            return Err(fail(
                "CUDA native process exit PID is not canonical decimal",
            ));
        }
        if status != "0" || pid == 0 || pid > i32::MAX as u32 {
            return Err(fail("engine exit is nonzero or malformed"));
        }
        let Some(slot) = expected_pids.iter().position(|expected| *expected == pid) else {
            if external.len() >= external_count || external.contains(&pid) {
                return Err(fail("foreign or duplicate external engine PID"));
            }
            external.push(pid);
            continue;
        };
        if seen[slot] {
            return Err(fail("CUDA native process exit evidence is duplicated"));
        }
        if status != "0" {
            return Err(fail(
                "CUDA native process exit status is nonzero or malformed",
            ));
        }
        seen[slot] = true;
    }
    if seen.iter().any(|present| !present) {
        return Err(fail("CUDA native process exit evidence is incomplete"));
    }
    if external.len() != external_count {
        return Err(fail("external engine exit evidence incomplete"));
    }
    Ok(external)
}

pub(crate) fn cuda_exit_trace_message(line: &str) -> Option<&str> {
    native_runner_message(line, "[TRACE ] [", true)
}

pub(crate) fn native_runner_message<'a>(
    line: &'a str,
    prefix: &str,
    require_thread: bool,
) -> Option<&'a str> {
    // Pinned logger.hpp:157: [label left-width6] [time width15]
    // <thread right-width20> fastchess --- message. TRACE_THREAD is nonempty.
    let (time, tail) = line.strip_prefix(prefix)?.split_once("] <")?;
    let time = time.as_bytes();
    if time.len() != 15
        || time[2] != b':'
        || time[5] != b':'
        || time[8] != b'.'
        || time
            .iter()
            .enumerate()
            .any(|(index, byte)| !matches!(index, 2 | 5 | 8) && !byte.is_ascii_digit())
        || (time[0] - b'0') * 10 + time[1] - b'0' > 23
        || (time[3] - b'0') * 10 + time[4] - b'0' > 59
        || (time[6] - b'0') * 10 + time[7] - b'0' > 59
    {
        return None;
    }
    let (thread, message) = tail.split_once("> fastchess --- ")?;
    let digits = thread.trim_start_matches(' ');
    if thread.len() != 20 {
        return None;
    }
    if digits.is_empty() {
        if require_thread {
            return None;
        }
    } else if digits.starts_with('0')
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
        || digits.parse::<u64>().is_err()
    {
        return None;
    }
    Some(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn heterogeneous_pair_requires_two_native_and_two_external_zero_exits() {
        let trace=[101,102,201,202].into_iter().map(|pid|format!("[TRACE ] [10:51:44.181984] <     137433152747200> fastchess --- Process with pid: {pid} terminated with status: 0\n")).collect::<String>();
        assert_eq!(
            validate_endpoint_exit_trace(trace.as_bytes(), &[101, 201], 2).unwrap(),
            [102, 202]
        );
        assert!(validate_endpoint_exit_trace(trace.as_bytes(), &[101, 201], 0).is_err());
        assert!(
            validate_endpoint_exit_trace(
                trace.replace("pid: 202", "pid: 102").as_bytes(),
                &[101, 201],
                2
            )
            .is_err()
        );
        assert!(
            validate_endpoint_exit_trace(
                trace
                    .replace(
                        "202 terminated with status: 0",
                        "202 terminated with status: 256"
                    )
                    .as_bytes(),
                &[101, 201],
                2
            )
            .is_err()
        );
    }
}
