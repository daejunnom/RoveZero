//! Optional synthetic UCI protocol fixture, never a product chess engine.
//! The only supported positions are exact prefixes of one declared move script.
//! There is no board model, legality checker, search, clock manager or neural net.

use std::env;
use std::ffi::OsString;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;

const MAX_LINE_BYTES: usize = 8192;
const MAX_COMMANDS: usize = 10_000;
const SCRIPT: [&str; 7] = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Mode {
    #[default]
    Normal,
    Crash,
    Illegal,
    Timeout,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, &'static str> {
        match value {
            "normal" => Ok(Self::Normal),
            "crash" => Ok(Self::Crash),
            "illegal" => Ok(Self::Illegal),
            "timeout" => Ok(Self::Timeout),
            _ => Err("mode must be exactly normal, crash, illegal or timeout"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Crash => "crash",
            Self::Illegal => "illegal",
            Self::Timeout => "timeout",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Termination {
    Complete,
    InjectedCrash,
}

#[derive(Default)]
struct FixtureState {
    mode: Mode,
    prefix_len: usize,
    pending_go: bool,
    seed: u64,
}

impl FixtureState {
    fn newgame(&mut self) {
        // UCI configuration survives ucinewgame; all game-scoped state resets.
        self.prefix_len = 0;
        self.pending_go = false;
    }

    fn next_move(&self) -> &'static str {
        // 0000 is only the sentinel after the complete declared script. It does
        // not assert a terminal-rule result for an arbitrary chess position.
        SCRIPT.get(self.prefix_len).copied().unwrap_or("0000")
    }
}

fn main() -> ExitCode {
    let mode = match parse_mode(&env::args_os().skip(1).collect::<Vec<_>>()) {
        Ok(mode) => mode,
        Err(error) => {
            let _ = line(&mut io::stderr().lock(), &error);
            return ExitCode::from(2);
        }
    };
    let result = run(
        mode,
        &mut io::stdin().lock(),
        &mut io::stdout().lock(),
        &mut io::stderr().lock(),
    );
    match result {
        Ok(Termination::Complete) => ExitCode::SUCCESS,
        Ok(Termination::InjectedCrash) => ExitCode::from(17),
        Err(error) => {
            let _ = line(
                &mut io::stderr().lock(),
                &format!("synthetic fixture error: {error}"),
            );
            ExitCode::from(2)
        }
    }
}

fn parse_mode(args: &[OsString]) -> Result<Mode, String> {
    if args.is_empty() {
        return Ok(Mode::Normal);
    }
    if args.len() == 2 && args[0] == "--mode" {
        return args[1]
            .to_str()
            .ok_or_else(|| "--mode must be valid UTF-8".to_string())
            .and_then(|value| Mode::parse(value).map_err(str::to_string));
    }
    Err("Usage: rz-arena-fixture-engine [--mode normal|crash|illegal|timeout]".to_string())
}

fn run<R: BufRead, W: Write, D: Write>(
    mode: Mode,
    input: &mut R,
    output: &mut W,
    diagnostics: &mut D,
) -> Result<Termination, String> {
    line(
        diagnostics,
        &format!(
            "SYNTHETIC_UCI_FIXTURE mode={} script=scholars_mate_v1 no_nn=true max_line_bytes={MAX_LINE_BYTES} max_commands={MAX_COMMANDS}",
            mode.name()
        ),
    )?;
    let mut state = FixtureState {
        mode,
        ..FixtureState::default()
    };
    for _ in 0..MAX_COMMANDS {
        let Some(command) = read_bounded_line(input)? else {
            return Ok(Termination::Complete);
        };
        let words: Vec<&str> = command.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        match words.as_slice() {
            ["uci"] => {
                for response in [
                    "id name RoveZero SYNTHETIC ScholarMate fixture (no NN, no search)",
                    "id author RoveZero synthetic test fixtures",
                    "option name Ponder type check default false",
                    "option name Threads type spin default 1 min 1 max 1",
                    "option name Seed type string default 0",
                ] {
                    line(output, response)?;
                }
                line(
                    output,
                    &format!(
                        "option name FixtureMode type combo default {} var normal var crash var illegal var timeout",
                        mode.name()
                    ),
                )?;
                line(output, "uciok")?;
            }
            ["isready"] => {
                let applied = format!(
                    "synthetic applied Ponder=false Threads=1 Seed={} mode={}",
                    state.seed,
                    state.mode.name()
                );
                line(diagnostics, &applied)?;
                line(output, &format!("info string {applied}"))?;
                line(output, "readyok")?;
            }
            ["ucinewgame"] => {
                state.newgame();
                line(
                    diagnostics,
                    "synthetic newgame: prefix and pending go reset",
                )?;
            }
            ["setoption", ..] => {
                apply_option(&words, &mut state)?;
                line(
                    diagnostics,
                    &format!(
                        "synthetic applied Ponder=false Threads=1 Seed={} mode={}",
                        state.seed,
                        state.mode.name()
                    ),
                )?;
            }
            ["position", ..] => {
                let prefix_len = parse_position(&words)?;
                state.prefix_len = prefix_len;
                state.pending_go = false;
                line(
                    diagnostics,
                    &format!(
                        "synthetic position startpos prefix_len={prefix_len} moves={}",
                        SCRIPT[..prefix_len].join(" ")
                    ),
                )?;
            }
            ["go", ..] => {
                validate_go(&words)?;
                line(
                    diagnostics,
                    &format!(
                        "synthetic go mode={} prefix_len={} clock_arguments_ignored=true",
                        state.mode.name(),
                        state.prefix_len
                    ),
                )?;
                match state.mode {
                    Mode::Normal => line(output, &format!("bestmove {}", state.next_move()))?,
                    Mode::Crash => {
                        line(diagnostics, "synthetic injected crash: exit 17")?;
                        return Ok(Termination::InjectedCrash);
                    }
                    Mode::Illegal => line(output, "bestmove a1a8")?,
                    // No busy loop or sleep. The external runner owns the finite
                    // timeout and process termination; stop/quit remain usable.
                    Mode::Timeout => state.pending_go = true,
                }
            }
            ["stop"] => {
                if state.pending_go {
                    state.pending_go = false;
                    line(output, &format!("bestmove {}", state.next_move()))?;
                }
            }
            ["debug", "on" | "off"] => {
                line(
                    diagnostics,
                    "synthetic debug command accepted; tracing is always on",
                )?;
            }
            ["quit"] => return Ok(Termination::Complete),
            _ => return Err("unsupported synthetic UCI fixture command".to_string()),
        }
    }
    Err(format!(
        "synthetic fixture exceeded {MAX_COMMANDS} input lines"
    ))
}

fn apply_option(words: &[&str], state: &mut FixtureState) -> Result<(), String> {
    match words {
        ["setoption", "name", "Ponder", "value", "false"]
        | ["setoption", "name", "Threads", "value", "1"] => Ok(()),
        ["setoption", "name", "Seed", "value", value] => {
            state.seed = value
                .parse::<u64>()
                .map_err(|_| "synthetic Seed option must be a u64".to_string())?;
            Ok(())
        }
        ["setoption", "name", "FixtureMode", "value", value] => {
            state.mode = Mode::parse(value).map_err(str::to_string)?;
            Ok(())
        }
        _ => Err(
            "synthetic fixture supports only Ponder=false, Threads=1, a u64 Seed and canonical FixtureMode values".to_string(),
        ),
    }
}

fn parse_position(words: &[&str]) -> Result<usize, String> {
    let moves = match words {
        ["position", "startpos"] => &[][..],
        ["position", "startpos", "moves", moves @ ..] => moves,
        _ => {
            return Err(
                "synthetic fixture accepts only position startpos with a script prefix".to_string(),
            );
        }
    };
    if moves.len() > SCRIPT.len()
        || moves
            .iter()
            .zip(SCRIPT)
            .any(|(actual, expected)| *actual != expected)
    {
        return Err("position is outside the declared synthetic ScholarMate script".to_string());
    }
    Ok(moves.len())
}

fn validate_go(words: &[&str]) -> Result<(), String> {
    let mut index = 1;
    while index < words.len() {
        match words[index] {
            "infinite" => index += 1,
            "wtime" | "btime" | "winc" | "binc" | "movestogo" | "depth" | "nodes" | "mate"
            | "movetime" => {
                words
                    .get(index + 1)
                    .and_then(|value| value.parse::<u64>().ok())
                    .ok_or_else(|| "synthetic go numeric option requires a u64".to_string())?;
                index += 2;
            }
            _ => {
                return Err(
                    "unsupported synthetic go option (ponder/searchmoves are disabled)".to_string(),
                );
            }
        }
    }
    Ok(())
}

fn read_bounded_line<R: BufRead>(input: &mut R) -> Result<Option<String>, String> {
    let mut bytes = Vec::new();
    loop {
        let available = input
            .fill_buf()
            .map_err(|error| format!("cannot read UCI input: {error}"))?;
        if available.is_empty() {
            if bytes.is_empty() {
                return Ok(None);
            }
            break;
        }
        let count = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        if count > MAX_LINE_BYTES - bytes.len() {
            return Err(format!(
                "synthetic UCI input line exceeds {MAX_LINE_BYTES} bytes"
            ));
        }
        bytes.extend_from_slice(&available[..count]);
        let complete = available[count - 1] == b'\n';
        input.consume(count);
        if complete {
            break;
        }
    }
    let text =
        String::from_utf8(bytes).map_err(|_| "synthetic UCI input is not UTF-8".to_string())?;
    Ok(Some(text.trim_end_matches(['\r', '\n']).to_string()))
}

fn line<W: Write>(output: &mut W, text: &str) -> Result<(), String> {
    writeln!(output, "{text}")
        .and_then(|()| output.flush())
        .map_err(|error| format!("cannot write synthetic UCI output: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn session(mode: Mode, commands: &str) -> (Result<Termination, String>, String, String) {
        let mut input = Cursor::new(commands.as_bytes());
        let mut output = Vec::new();
        let mut diagnostics = Vec::new();
        let result = run(mode, &mut input, &mut output, &mut diagnostics);
        (
            result,
            String::from_utf8(output).unwrap(),
            String::from_utf8(diagnostics).unwrap(),
        )
    }

    #[test]
    fn normal_session_returns_the_declared_script_and_observed_options() {
        let mut commands = "uci\nsetoption name Ponder value false\nsetoption name Threads value 1\nsetoption name Seed value 42\nisready\n".to_string();
        for prefix_len in 0..SCRIPT.len() {
            commands.push_str(&format!(
                "position startpos moves {}\ngo wtime 1000 btime 1000 winc 0 binc 0\n",
                SCRIPT[..prefix_len].join(" ")
            ));
        }
        commands.push_str("quit\n");
        let (result, output, diagnostics) = session(Mode::Normal, &commands);
        assert_eq!(result.unwrap(), Termination::Complete);
        assert!(output.contains("SYNTHETIC"));
        assert!(output.contains("uciok\n"));
        assert!(output.contains("Seed=42 mode=normal\nreadyok\n"));
        let moves: Vec<_> = output
            .lines()
            .filter_map(|line| line.strip_prefix("bestmove "))
            .collect();
        assert_eq!(moves, SCRIPT);
        assert!(diagnostics.contains("clock_arguments_ignored=true"));
    }

    #[test]
    fn cli_initial_mode_is_the_advertised_uci_combo_default() {
        assert_eq!(parse_mode(&[]).unwrap(), Mode::Normal);
        for mode in [Mode::Normal, Mode::Crash, Mode::Illegal, Mode::Timeout] {
            let initial = parse_mode(&["--mode".into(), mode.name().into()]).unwrap();
            assert_eq!(initial, mode);
            assert_eq!(Mode::parse(mode.name()).unwrap(), mode);
            let (result, output, _) = session(initial, "uci\nisready\nquit\n");
            assert_eq!(result.unwrap(), Termination::Complete);
            assert!(output.contains(&format!(
                "option name FixtureMode type combo default {} var normal var crash var illegal var timeout\n",
                mode.name()
            )));
            assert!(output.contains(&format!("mode={}\nreadyok\n", mode.name())));
        }
    }

    #[test]
    fn runtime_mode_option_controls_go_and_records_the_applied_configuration() {
        for mode in [Mode::Normal, Mode::Crash, Mode::Illegal, Mode::Timeout] {
            let initial = if mode == Mode::Normal {
                Mode::Illegal
            } else {
                Mode::Normal
            };
            let commands = format!(
                "setoption name FixtureMode value {}\nisready\nposition startpos\ngo\nisready\nstop\nquit\n",
                mode.name()
            );
            let (result, output, diagnostics) = session(initial, &commands);
            assert!(output.contains(&format!("mode={}\nreadyok\n", mode.name())));
            assert!(diagnostics.contains(&format!(
                "synthetic applied Ponder=false Threads=1 Seed=0 mode={}\n",
                mode.name()
            )));
            assert!(diagnostics.contains(&format!("synthetic go mode={} ", mode.name())));
            let moves: Vec<_> = output
                .lines()
                .filter(|line| line.starts_with("bestmove "))
                .collect();
            match mode {
                Mode::Crash => {
                    assert_eq!(result.unwrap(), Termination::InjectedCrash);
                    assert!(moves.is_empty());
                }
                Mode::Illegal => {
                    assert_eq!(result.unwrap(), Termination::Complete);
                    assert_eq!(moves, ["bestmove a1a8"]);
                }
                Mode::Normal | Mode::Timeout => {
                    assert_eq!(result.unwrap(), Termination::Complete);
                    assert_eq!(moves, ["bestmove e2e4"]);
                    let bestmove = output.find("bestmove ").unwrap();
                    let ready_after_go = output.rfind("readyok\n").unwrap();
                    if mode == Mode::Timeout {
                        assert!(
                            ready_after_go < bestmove,
                            "timeout withholds reply until stop"
                        );
                    } else {
                        assert!(
                            bestmove < ready_after_go,
                            "normal replies to go immediately"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn position_parser_rejects_every_unregistered_variation() {
        assert_eq!(parse_position(&["position", "startpos"]).unwrap(), 0);
        assert_eq!(
            parse_position(&["position", "startpos", "moves", "e2e4", "e7e5"]).unwrap(),
            2
        );
        assert!(parse_position(&["position", "fen", "8/8/8/8/8/8/8/8"]).is_err());
        assert!(parse_position(&["position", "startpos", "moves", "d2d4"]).is_err());
        assert!(parse_position(&["position", "startpos", "moves", "e7e5", "e2e4"]).is_err());
    }

    #[test]
    fn newgame_resets_position_and_pending_work_but_preserves_seed_and_mode() {
        let (result, output, _) = session(
            Mode::Normal,
            "setoption name Seed value 42\nposition startpos moves e2e4 e7e5\nucinewgame\nisready\ngo\nquit\n",
        );
        assert!(result.is_ok());
        assert!(output.contains("Seed=42"));
        assert!(output.contains("bestmove e2e4\n"));
        let (result, output, _) = session(
            Mode::Timeout,
            "position startpos moves e2e4 e7e5\ngo\nucinewgame\nstop\nquit\n",
        );
        assert!(result.is_ok());
        assert!(!output.contains("bestmove"));
        let (result, output, _) = session(
            Mode::Normal,
            "setoption name Seed value 42\nsetoption name FixtureMode value illegal\nposition startpos moves e2e4 e7e5\nucinewgame\nisready\ngo\nquit\n",
        );
        assert_eq!(result.unwrap(), Termination::Complete);
        assert!(output.contains("Seed=42 mode=illegal\nreadyok\n"));
        assert!(output.contains("bestmove a1a8\n"));
        let (result, output, _) = session(
            Mode::Normal,
            "setoption name FixtureMode value timeout\nposition startpos moves e2e4 e7e5\ngo\nucinewgame\nisready\nstop\ngo\nstop\nquit\n",
        );
        assert_eq!(result.unwrap(), Termination::Complete);
        assert!(output.contains("mode=timeout\nreadyok\n"));
        assert_eq!(output.matches("bestmove ").count(), 1);
        assert!(output.contains("bestmove e2e4\n"));
    }

    #[test]
    fn fault_modes_are_explicit_and_stop_can_end_a_withheld_reply() {
        let (result, output, _) = session(Mode::Crash, "position startpos\ngo\n");
        assert_eq!(result.unwrap(), Termination::InjectedCrash);
        assert!(!output.contains("bestmove"));
        let (result, output, _) = session(Mode::Illegal, "position startpos\ngo\nquit\n");
        assert!(result.is_ok());
        assert_eq!(output, "bestmove a1a8\n");
        let (result, output, _) = session(
            Mode::Timeout,
            "position startpos moves e2e4 e7e5\ngo\nisready\nstop\nquit\n",
        );
        assert!(result.is_ok());
        assert_eq!(
            output
                .lines()
                .filter(|line| line.starts_with("bestmove "))
                .collect::<Vec<_>>(),
            ["bestmove d1h5"]
        );
    }

    #[test]
    fn unsupported_options_and_go_inputs_fail_without_a_reply() {
        for command in [
            "setoption name Ponder value true\n",
            "setoption name Threads value 2\n",
            "setoption name Seed value 18446744073709551616\n",
            "go ponder\n",
            "go movetime bad\n",
        ] {
            let (result, output, _) = session(Mode::Normal, command);
            assert!(result.is_err());
            assert!(output.is_empty());
        }
        assert!(parse_mode(&["--mode".into(), "random".into()]).is_err());
        assert!(parse_mode(&["--mode".into(), "normal".into(), "extra".into()]).is_err());
    }

    #[test]
    fn invalid_runtime_mode_is_static_and_leaves_the_previous_configuration_intact() {
        let mut state = FixtureState {
            mode: Mode::Illegal,
            prefix_len: 2,
            pending_go: true,
            seed: 42,
        };
        for value in ["NORMAL", "Normal", "private-option-marker", "", "timeout=1"] {
            let rejected = apply_option(
                &["setoption", "name", "FixtureMode", "value", value],
                &mut state,
            )
            .unwrap_err();
            assert_eq!(
                rejected,
                "mode must be exactly normal, crash, illegal or timeout"
            );
            assert_eq!(state.mode, Mode::Illegal);
            assert_eq!(state.prefix_len, 2);
            assert!(state.pending_go);
            assert_eq!(state.seed, 42);
        }
        let (result, output, diagnostics) = session(
            Mode::Normal,
            "setoption name FixtureMode value private-option-marker\ngo\n",
        );
        assert_eq!(
            result.unwrap_err(),
            "mode must be exactly normal, crash, illegal or timeout"
        );
        assert!(output.is_empty());
        assert!(!diagnostics.contains("private-option-marker"));
        assert!(!diagnostics.contains("synthetic applied"));
    }

    #[test]
    fn line_reader_is_bounded_and_strict_utf8() {
        let mut too_long = Cursor::new(vec![b'x'; MAX_LINE_BYTES + 1]);
        assert!(read_bounded_line(&mut too_long).is_err());
        let mut invalid = Cursor::new([0xff, b'\n']);
        assert!(read_bounded_line(&mut invalid).is_err());
        let mut normal = Cursor::new(b"uci\r\nquit");
        assert_eq!(
            read_bounded_line(&mut normal).unwrap().as_deref(),
            Some("uci")
        );
        assert_eq!(
            read_bounded_line(&mut normal).unwrap().as_deref(),
            Some("quit")
        );
        assert_eq!(read_bounded_line(&mut normal).unwrap(), None);
    }
}
