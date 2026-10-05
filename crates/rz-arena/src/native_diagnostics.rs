//! Opt-in execution timing only; these observations never grant acceptance.
//! RZ_ARENA_PHASE_TRACE=1 writes at most 64 short JSON lines to stderr.

use serde::Serialize;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(any(target_os = "linux", test))]
const LINE_CAP: usize = 16 * 1024;
const EVENT_CAP: u64 = 64;

struct Trace {
    start: Instant,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    sequence: u64,
    #[cfg(target_os = "linux")]
    logs: LogProbe,
    #[cfg(target_os = "linux")]
    stdout_bytes: u64,
    #[cfg(target_os = "linux")]
    stderr_bytes: u64,
    #[cfg(target_os = "linux")]
    drain_batches: u64,
    #[cfg(target_os = "linux")]
    last_stdout_ns: Option<u64>,
    #[cfg(target_os = "linux")]
    last_stderr_ns: Option<u64>,
}

static TRACE: OnceLock<Option<Trace>> = OnceLock::new();

fn trace() -> Option<&'static Trace> {
    TRACE
        .get_or_init(|| {
            (std::env::var_os("RZ_ARENA_PHASE_TRACE").as_deref() == Some(std::ffi::OsStr::new("1")))
                .then(|| Trace {
                    start: Instant::now(),
                    state: Mutex::new(State::default()),
                })
        })
        .as_ref()
}

#[derive(Default, Serialize)]
struct Detail {
    #[serde(skip_serializing_if = "Option::is_none")]
    game: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    engine_slot: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stdout_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stderr_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    drain_batches: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_stdout_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_stderr_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    oversized_lines: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    readiness: Option<Readiness>,
}

fn emit(trace: &Trace, state: &mut State, phase: &'static str, detail: Detail) {
    if state.sequence >= EVENT_CAP {
        return;
    }
    state.sequence += 1;
    let value = serde_json::json!({
        "schema_version":1, "sequence":state.sequence, "phase":phase,
        "elapsed_ns":u64::try_from(trace.start.elapsed().as_nanos()).unwrap_or(u64::MAX),
        "utc_unix_ms":SystemTime::now().duration_since(UNIX_EPOCH).ok()
            .and_then(|d| u64::try_from(d.as_millis()).ok()), "detail":detail,
    });
    // Best effort diagnostics must not change runner acceptance on sink failure.
    let _ = writeln!(std::io::stderr().lock(), "native_phase={value}");
}

/// Opt-in lifecycle marker on stderr. No raw input paths or log text are emitted.
pub fn emit_native_phase(phase: &'static str) {
    if let Some(trace) = trace()
        && let Ok(mut state) = trace.state.lock()
    {
        emit(trace, &mut state, phase, Detail::default());
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn observe_stream(stdout: bool, bytes: &[u8]) {
    let Some(trace) = trace() else { return };
    let Ok(mut state) = trace.state.lock() else {
        return;
    };
    let now = u64::try_from(trace.start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    state.drain_batches = state.drain_batches.saturating_add(1);
    let first = if stdout {
        let first = state.stdout_bytes == 0;
        state.stdout_bytes = state.stdout_bytes.saturating_add(bytes.len() as u64);
        state.last_stdout_ns = Some(now);
        first
    } else {
        let first = state.stderr_bytes == 0;
        state.stderr_bytes = state.stderr_bytes.saturating_add(bytes.len() as u64);
        state.last_stderr_ns = Some(now);
        first
    };
    if first {
        emit(
            trace,
            &mut state,
            if stdout {
                "runner_stdout_first"
            } else {
                "runner_stderr_first"
            },
            Detail::default(),
        );
    }
    if stdout {
        for (phase, game, engine_slot) in state.logs.feed(bytes, now) {
            emit(
                trace,
                &mut state,
                phase,
                Detail {
                    game: Some(game),
                    engine_slot,
                    ..Detail::default()
                },
            );
        }
        for (game, readiness) in std::mem::take(&mut state.logs.completed) {
            emit(
                trace,
                &mut state,
                "readiness_summary",
                Detail {
                    game: Some(game),
                    readiness: Some(readiness),
                    ..Detail::default()
                },
            );
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn finish_logs() {
    if let Some(trace) = trace()
        && let Ok(mut state) = trace.state.lock()
    {
        let detail = Detail {
            stdout_bytes: Some(state.stdout_bytes),
            stderr_bytes: Some(state.stderr_bytes),
            drain_batches: Some(state.drain_batches),
            last_stdout_ns: state.last_stdout_ns,
            last_stderr_ns: state.last_stderr_ns,
            oversized_lines: Some(state.logs.oversized_lines),
            ..Detail::default()
        };
        emit(trace, &mut state, "runner_log_summary", detail);
    }
}

#[cfg(any(target_os = "linux", test))]
type LogEvent = (&'static str, u8, Option<u8>);

#[cfg(any(target_os = "linux", test))]
#[derive(Default)]
struct LogProbe {
    partial: Vec<u8>,
    dropping: bool,
    oversized_lines: u64,
    started: [bool; 2],
    game: Option<Game>,
    completed: Vec<(u8, Readiness)>,
}

#[derive(Default, Serialize)]
struct Readiness {
    calls: [u64; 2],
    capture_wait_ns: [u64; 2],
    max_capture_wait_ns: [u64; 2],
    unmatched_replies: [u64; 2],
    overlapping_requests: [u64; 2],
}

#[cfg(any(target_os = "linux", test))]
struct Game {
    number: u8,
    names: [String; 2],
    seen: [[bool; 4]; 2],
    first_go: bool,
    pending_ready: [Option<u64>; 2],
    readiness: Readiness,
}

#[cfg(any(target_os = "linux", test))]
impl LogProbe {
    fn feed(&mut self, bytes: &[u8], now_ns: u64) -> Vec<LogEvent> {
        let mut events = Vec::new();
        for &byte in bytes {
            if byte == b'\n' {
                if !self.dropping {
                    let line = std::mem::take(&mut self.partial);
                    if let Ok(line) = std::str::from_utf8(&line) {
                        self.line(line.trim_end_matches('\r'), now_ns, &mut events);
                    }
                    self.partial = line;
                }
                self.partial.clear();
                self.dropping = false;
            } else if !self.dropping {
                if self.partial.len() == LINE_CAP {
                    self.oversized_lines = self.oversized_lines.saturating_add(1);
                    self.partial.clear();
                    self.dropping = true;
                } else {
                    self.partial.push(byte);
                }
            }
        }
        events
    }

    fn line(&mut self, line: &str, now_ns: u64, events: &mut Vec<LogEvent>) {
        if let Some((_, suffix)) = line.split_once("fastchess --- Game ")
            && let Some(pair) = suffix.strip_suffix(" starting")
            && let Some((number, names)) = pair.split_once(" between ")
            && let Ok(number @ 1..=2) = number.parse::<u8>()
            && !self.started[number as usize - 1]
            && let Some((white, black)) = names.split_once(" and ")
            && !white.is_empty()
            && !black.is_empty()
            && white != black
            && white.len() <= 256
            && black.len() <= 256
        {
            self.started[number as usize - 1] = true;
            self.game = Some(Game {
                number,
                names: [white.into(), black.into()],
                seen: [[false; 4]; 2],
                first_go: false,
                pending_ready: [None; 2],
                readiness: Readiness::default(),
            });
            events.push(("game_started", number, None));
            return;
        }
        let Some(game) = self.game.as_mut() else {
            return;
        };
        for (kind, (suffix, phase)) in [
            (" <--- uci", "engine_uci_sent"),
            (" ---> uciok", "engine_uciok"),
            (" <--- isready", "engine_isready_sent"),
            (" ---> readyok", "engine_ready"),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(before) = line.strip_suffix(suffix) {
                for slot in 0..2 {
                    if before
                        .strip_suffix(&game.names[slot])
                        .and_then(|prefix| prefix.strip_suffix(' '))
                        .is_none()
                    {
                        continue;
                    }
                    if kind == 2 {
                        if game.pending_ready[slot].replace(now_ns).is_some() {
                            game.readiness.overlapping_requests[slot] =
                                game.readiness.overlapping_requests[slot].saturating_add(1);
                        }
                    } else if kind == 3 {
                        if let Some(sent) = game.pending_ready[slot].take() {
                            let delay = now_ns.saturating_sub(sent);
                            game.readiness.calls[slot] =
                                game.readiness.calls[slot].saturating_add(1);
                            game.readiness.capture_wait_ns[slot] =
                                game.readiness.capture_wait_ns[slot].saturating_add(delay);
                            game.readiness.max_capture_wait_ns[slot] =
                                game.readiness.max_capture_wait_ns[slot].max(delay);
                        } else {
                            game.readiness.unmatched_replies[slot] =
                                game.readiness.unmatched_replies[slot].saturating_add(1);
                        }
                    }
                    if !game.seen[slot][kind] {
                        game.seen[slot][kind] = true;
                        events.push((phase, game.number, Some(slot as u8)));
                        if kind == 3 && game.seen.iter().all(|s| s[3]) {
                            events.push(("both_engines_ready", game.number, None));
                        }
                    }
                }
            }
        }
        if line.contains(" <--- go ") && !game.first_go {
            game.first_go = true;
            events.push(("first_go", game.number, None));
        }
        if let Some((_, suffix)) = line.split_once("fastchess --- Game ")
            && suffix
                .strip_suffix(" finished")
                .and_then(|s| s.split_once(" between "))
                .and_then(|(n, _)| n.parse::<u8>().ok())
                == Some(game.number)
        {
            events.push(("game_finished", game.number, None));
            let game = self.game.take().expect("matched current game");
            self.completed.push((game.number, game.readiness));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ready_roundtrips_are_aggregated_without_emitting_every_ping() {
        let mut probe = LogProbe::default();
        probe.feed(
            b"fastchess --- Game 1 between A and B starting\n[time] A <--- isready\n",
            10,
        );
        probe.feed(b" A <--- isready\n", 20);
        probe.feed(b" A ---> readyok\n", 50);
        probe.feed(b" A <--- isready\n", 100);
        let events = probe.feed(
            b" A ---> readyok\nfastchess --- Game 1 between A and B finished\n",
            200,
        );
        assert!(events.contains(&("game_finished", 1, None)));
        let (game, ready) = probe.completed.pop().unwrap();
        assert_eq!(game, 1);
        assert_eq!(ready.calls, [2, 0]);
        assert_eq!(ready.capture_wait_ns, [130, 0]);
        assert_eq!(ready.max_capture_wait_ns, [100, 0]);
        assert_eq!(ready.overlapping_requests, [1, 0]);
    }

    #[test]
    fn fragmented_lines_keep_two_games_and_distinct_readiness_without_duplicate_events() {
        let mut probe = LogProbe::default();
        assert!(
            probe
                .feed(b"[time] fastchess --- Game 1 between A and B sta", 0)
                .is_empty()
        );
        let events = probe.feed(b"rting\n[time] A <--- uci\n[time] A ---> uciok\n[time] A <--- isready\n[time] A ---> readyok\n[time] A ---> readyok\n[time] B ---> readyok\n[time] A <--- go wtime 1\n[time] fastchess --- Game 1 between A and B finished\n[time] fastchess --- Game 2 between B and A starting\n[time] B ---> readyok\n", 0);
        assert_eq!(
            events
                .iter()
                .filter(|(p, _, _)| *p == "engine_ready")
                .count(),
            3
        );
        assert_eq!(
            events
                .iter()
                .filter(|(p, _, _)| *p == "both_engines_ready")
                .count(),
            1
        );
        assert!(events.contains(&("game_finished", 1, None)));
        assert!(events.contains(&("engine_ready", 2, Some(0))));
        assert!(
            probe
                .feed(b"[time] fastchess --- Game 3 between A and B starting\n", 0)
                .is_empty()
        );
    }

    #[test]
    fn oversized_or_non_utf8_lines_cannot_grow_state_or_invent_readiness() {
        let mut probe = LogProbe::default();
        probe.feed(b"fastchess --- Game 1 between A and B starting\n", 0);
        assert!(probe.feed(&vec![b'x'; LINE_CAP * 3], 0).is_empty());
        assert!(probe.partial.len() <= LINE_CAP);
        assert_eq!(probe.oversized_lines, 1);
        assert!(
            probe
                .feed(b" A ---> readyok\n\xff A ---> readyok\n", 0)
                .is_empty()
        );
        assert_eq!(
            probe.feed(b"[time] A ---> readyok\n", 0),
            vec![("engine_ready", 1, Some(0))]
        );
    }
}
