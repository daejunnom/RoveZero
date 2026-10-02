use rz_arena::{
    ArenaPlan, FASTCHESS_SOURCE_COMMIT, FASTCHESS_SOURCE_URL, FASTCHESS_VERSION,
    FastchessInvocation, PlanLimits, build_fastchess_invocation,
};
use rz_experiments::{ClockSpec, EngineKind, RunManifest};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;

const FIXTURE: &str = include_str!("../../../experiments/baselines/fixtures/e01-input.json");

fn fixture() -> RunManifest {
    let mut input = RunManifest::from_json(FIXTURE).unwrap();
    input.contract_revision = Some("0.1".into());
    input.protocol.runner.source_url = FASTCHESS_SOURCE_URL.into();
    input.protocol.runner.source_commit = FASTCHESS_SOURCE_COMMIT.into();
    input.protocol.runner.version = FASTCHESS_VERSION.into();
    input.protocol.max_plies = 6;
    input.budget.max_child_processes = 3;
    input
}

fn plan(input: RunManifest) -> ArenaPlan {
    ArenaPlan::build(
        &input.lock().expect("structurally valid test input"),
        PlanLimits {
            max_pairs: 8,
            max_json_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap()
}

// The invocation builder preserves native path bytes on every host. These
// paths are argument fixtures only; the Linux runner's /proc handles are not
// opened by this test. Keep intentionally relative invalid cases relative.
fn native_path(path: &str) -> PathBuf {
    #[cfg(windows)]
    if path.starts_with('/') {
        return PathBuf::from(format!("C:{path}"));
    }
    PathBuf::from(path)
}

fn commands() -> BTreeMap<String, PathBuf> {
    BTreeMap::from([
        ("candidate-fixture".into(), native_path("/proc/123/fd/10")),
        ("baseline-fixture".into(), native_path("/proc/123/fd/11")),
    ])
}

fn invoke(input: RunManifest) -> Result<FastchessInvocation, rz_arena::ArenaError> {
    let planned = plan(input);
    build_fastchess_invocation(
        &planned,
        &planned.pairs()[0].id,
        &commands(),
        &native_path("/fixture/opening.pgn"),
        &native_path("/fixture/pair.pgn"),
    )
}

fn value<'a>(args: &'a [OsString], flag: &str) -> &'a std::ffi::OsStr {
    let index = args
        .iter()
        .position(|argument| argument == flag)
        .expect("flag present");
    &args[index + 1]
}

#[test]
fn constructs_the_verified_single_pair_interface_and_accounts_for_book_prefix() {
    // Actual pinned-runner smoke: two book plies + maxmoves=2 produced six
    // total plies, with the same prefix in both color-swapped games.
    let mut input = fixture();
    input.lifecycle.handshake_timeout_ms = 1100;
    input.lifecycle.drain_timeout_ms = 1200;
    input.lifecycle.timeout_grace_ms = 7;
    let invocation = invoke(input).unwrap();
    let args = &invocation.args;
    assert_eq!(value(args, "-rounds"), "1");
    assert_eq!(value(args, "-games"), "2");
    assert_eq!(value(args, "-concurrency"), "1");
    assert_eq!(value(args, "-maxmoves"), "2");
    assert_eq!(value(args, "-startup-ms"), "1100");
    assert_eq!(value(args, "-ping-ms"), "1100");
    assert_eq!(value(args, "-ucinewgame-ms"), "1200");
    // Check the pinned Linux logging profile's arguments. The real runner's
    // ability to capture UCI trace is a separate execution check.
    assert_eq!(value(args, "-log"), "file=/proc/self/fd/1");
    let log_index = args.iter().position(|argument| argument == "-log").unwrap();
    assert_eq!(
        &args[log_index + 1..log_index + 6],
        &[
            OsString::from("file=/proc/self/fd/1"),
            "append=true".into(),
            "level=trace".into(),
            "realtime=true".into(),
            "engine=true".into(),
        ]
    );
    for expected in [
        "-repeat",
        "-strict",
        "format=pgn",
        "order=sequential",
        "plies=2",
        "start=1",
        "tc=1",
        "timemargin=7",
        "restart=on",
        "proto=uci",
        "notation=uci",
        "append=false",
    ] {
        assert!(
            args.iter().any(|argument| argument == expected),
            "missing {expected}"
        );
    }
    for forbidden in [
        "-reverse",
        "-noswap",
        "-recover",
        "-draw",
        "-resign",
        "-sprt",
        "-tb",
        "-force-concurrency",
        "-debug",
    ] {
        assert!(
            !args.iter().any(|argument| argument == forbidden),
            "unexpected {forbidden}"
        );
    }
    assert!(
        invocation
            .limitations
            .iter()
            .any(|text| text.contains("execution_ready=false"))
    );
    assert!(
        invocation
            .limitations
            .iter()
            .any(|text| text.contains("clock boundary is unverified"))
    );
    assert!(
        invocation
            .limitations
            .iter()
            .any(|text| text.contains("engine seeds are not applied"))
    );
}

#[test]
fn engine_group_order_matches_each_declared_first_game_in_both_plan_parities() {
    for seed in [0, 1] {
        let mut input = fixture();
        input.plan.seed = seed;
        let planned = plan(input);
        let pair = &planned.pairs()[0];
        let first = &pair.games[pair.execution_order[0]];
        let invocation = build_fastchess_invocation(
            &planned,
            &pair.id,
            &commands(),
            &native_path("/fixture/opening.pgn"),
            &native_path("/fixture/pair.pgn"),
        )
        .unwrap();
        let names: Vec<_> = invocation
            .args
            .iter()
            .filter_map(|argument| argument.to_str()?.strip_prefix("name="))
            .collect();
        assert_eq!(
            names,
            [first.white_engine.as_str(), first.black_engine.as_str()]
        );
        let command_values: Vec<_> = invocation
            .args
            .iter()
            .filter_map(|argument| argument.to_str()?.strip_prefix("cmd="))
            .collect();
        let engine_paths = commands();
        assert_eq!(
            command_values[0],
            engine_paths[&first.white_engine].to_str().unwrap()
        );
        assert_eq!(
            command_values[1],
            engine_paths[&first.black_engine].to_str().unwrap()
        );
    }
}

#[test]
fn preserves_requested_options_and_does_not_invent_hash_or_seed_application() {
    let invocation = invoke(fixture()).unwrap();
    assert_eq!(
        invocation
            .args
            .iter()
            .filter(|argument| *argument == "option.Threads=1")
            .count(),
        2
    );
    assert_eq!(
        invocation
            .args
            .iter()
            .filter(|argument| *argument == "option.Ponder=false")
            .count(),
        2
    );
    assert!(!invocation.args.iter().any(|argument| {
        argument.to_str().is_some_and(|text| {
            text.starts_with("option.Hash=") || text.starts_with("option.Seed=")
        })
    }));
    let mut input = fixture();
    input.engines[0]
        .requested_options
        .insert("Hash".into(), "1".into());
    let invocation = invoke(input).unwrap();
    assert_eq!(
        invocation
            .args
            .iter()
            .filter(|argument| *argument == "option.Hash=1")
            .count(),
        1
    );
}

#[test]
fn rejects_unsupported_clock_increment_and_fractional_second_conversion() {
    for clock in [
        ClockSpec::T1 {
            base_ms: 1000,
            increment_ms: 1,
        },
        ClockSpec::T1 {
            base_ms: 1001,
            increment_ms: 0,
        },
        ClockSpec::T1 {
            base_ms: 86_401_000,
            increment_ms: 0,
        },
        ClockSpec::T2 { movetime_ms: 100 },
        ClockSpec::T3 {
            unique_evaluation_budget: 1,
        },
    ] {
        let mut input = fixture();
        input.clock = clock;
        assert!(invoke(input).is_err());
    }
}

#[test]
fn rejects_limits_that_cannot_represent_the_full_opening_plus_searched_plies() {
    for maximum in [1, 2, 3, 5, 4096] {
        let mut input = fixture();
        input.protocol.max_plies = maximum;
        assert!(invoke(input).is_err());
    }
    let mut input = fixture();
    input.input.openings[0].moves.clear();
    input.protocol.max_plies = 2;
    let invocation = invoke(input).unwrap();
    assert!(invocation.args.contains(&"plies=0".into()));
    assert_eq!(value(&invocation.args, "-maxmoves"), "1");
}

#[test]
fn rejects_unknown_or_dirty_runner_profiles_and_contract_versions() {
    for variant in 0..6 {
        let mut input = fixture();
        match variant {
            0 => input.protocol.runner.version = "fastchess alpha 1.8.1".into(),
            1 => input.protocol.runner.source_commit = "a".repeat(40),
            2 => input.protocol.runner.source_url = "https://github.com/other/fastchess".into(),
            3 => {
                input.protocol.runner.dirty = true;
                input.protocol.runner.dirty_patch = Some(input.protocol.runner.binary.clone());
            }
            4 => input.contract_revision = None,
            5 => input.contract_revision = Some("0.2".into()),
            _ => unreachable!(),
        }
        assert!(invoke(input).is_err());
    }
}

#[test]
fn rejects_nonfixture_engines_weights_and_insufficient_process_capacity() {
    for kind in [EngineKind::RoveZero, EngineKind::Lc0] {
        let mut input = fixture();
        input.engines[0].kind = kind;
        assert!(invoke(input).is_err());
    }
    let mut weighted = fixture();
    weighted.engines[0].weight = Some(weighted.engines[0].tool.binary.clone());
    assert!(invoke(weighted).is_err());
    let mut input = fixture();
    input.budget.max_child_processes = 2;
    assert!(invoke(input).is_err());
    let mut input = fixture();
    input.budget.workers = 2;
    input.budget.max_child_processes = 8;
    assert!(invoke(input).is_err());
}

#[test]
fn optional_fixture_mode_accepts_only_the_four_literals_behind_fixture_guards() {
    assert!(
        !invoke(fixture())
            .unwrap()
            .args
            .iter()
            .any(|argument| argument
                .as_encoded_bytes()
                .starts_with(b"option.FixtureMode=")),
        "the mode option remains optional"
    );
    for mode in ["normal", "crash", "illegal", "timeout"] {
        let mut input = fixture();
        input.engines[0]
            .requested_options
            .insert("FixtureMode".into(), mode.into());
        let invocation = invoke(input).unwrap();
        let expected = OsString::from(format!("option.FixtureMode={mode}"));
        assert_eq!(
            invocation
                .args
                .iter()
                .filter(|argument| *argument == &expected)
                .count(),
            1
        );
        for kind in [EngineKind::RoveZero, EngineKind::Lc0] {
            let mut input = fixture();
            input.engines[0].kind = kind;
            input.engines[0]
                .requested_options
                .insert("FixtureMode".into(), mode.into());
            assert!(
                invoke(input).is_err(),
                "product kind {kind:?} cannot consume FixtureMode={mode}"
            );
        }
    }
    for value in ["Normal", "CRASH", "random", "normal crash", "", "--illegal"] {
        let mut input = fixture();
        input.engines[0]
            .requested_options
            .insert("FixtureMode".into(), value.into());
        assert!(
            invoke(input).is_err(),
            "accepted noncanonical mode {value:?}"
        );
    }
    let mut input = fixture();
    input.engines[0]
        .requested_options
        .insert("fixtureMode".into(), "normal".into());
    assert!(invoke(input).is_err());
}

#[test]
fn rejects_engine_option_token_injection_unknown_rng_options_and_noncanonical_values() {
    for (name, value) in [
        ("Hash", "1 2"),
        ("Hash", "1\u{a0}2"),
        ("Hash", ""),
        ("Hash", "0"),
        ("Hash", "001"),
        ("Hash", "1025"),
        ("Hash", "-games=1000"),
        ("Hash Size", "1"),
        ("Seed", "7"),
        ("Threads", "2"),
    ] {
        let mut input = fixture();
        input.engines[0]
            .requested_options
            .insert(name.into(), value.into());
        assert!(invoke(input).is_err(), "accepted {name}={value}");
    }
    let mut input = fixture();
    input.engines[0].requested_options.remove("Threads");
    input.engines[0]
        .requested_options
        .insert("threads".into(), "1".into());
    assert!(invoke(input).is_err());
}

#[test]
fn rejects_path_token_injection_alias_output_epd_and_unknown_engine_commands() {
    let planned = plan(fixture());
    let id = &planned.pairs()[0].id;
    for (opening, output) in [
        ("relative/opening.pgn", "/fixture/output.pgn"),
        ("/fixture/with space.pgn", "/fixture/output.pgn"),
        ("/fixture/opening=evil.pgn", "/fixture/output.pgn"),
        ("/fixture/opening.pgn", "/fixture/output\n.pgn"),
        ("/fixture/opening.pgn", "/fixture/opening.pgn"),
        ("/fixture/opening.epd", "/fixture/output.pgn"),
    ] {
        assert!(
            build_fastchess_invocation(
                &planned,
                id,
                &commands(),
                &native_path(opening),
                &native_path(output)
            )
            .is_err()
        );
    }
    for path in [
        "engine",
        "/fixture/engine option.Hash=1",
        "/fixture/engine'",
        "/fixture/engine\\",
    ] {
        let mut engines = commands();
        engines.insert("candidate-fixture".into(), native_path(path));
        assert!(
            build_fastchess_invocation(
                &planned,
                id,
                &engines,
                &native_path("/fixture/input.pgn"),
                &native_path("/fixture/output.pgn")
            )
            .is_err()
        );
    }
    let mut engines = commands();
    engines.insert("extra-engine".into(), native_path("/fixture/extra"));
    assert!(
        build_fastchess_invocation(
            &planned,
            id,
            &engines,
            &native_path("/fixture/input.pgn"),
            &native_path("/fixture/output.pgn")
        )
        .is_err()
    );
    assert!(
        build_fastchess_invocation(
            &planned,
            "missing-pair",
            &commands(),
            &native_path("/fixture/input.pgn"),
            &native_path("/fixture/output.pgn")
        )
        .is_err()
    );
}

#[test]
fn rejects_unrepresentable_runner_deadlines_instead_of_wrapping_cpp_durations() {
    let mut input = fixture();
    input.lifecycle.handshake_timeout_ms = u64::MAX;
    assert!(invoke(input).is_err());
    let mut input = fixture();
    input.lifecycle.drain_timeout_ms = u64::MAX;
    assert!(invoke(input).is_err());
    let mut input = fixture();
    input.lifecycle.timeout_grace_ms = i32::MAX as u64 + 1;
    assert!(invoke(input).is_err());
}

#[cfg(unix)]
#[test]
fn constructs_native_non_utf8_engine_paths_without_lossy_conversion() {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    let planned = plan(fixture());
    let mut engines = commands();
    engines.insert(
        "candidate-fixture".into(),
        PathBuf::from(OsString::from_vec(b"/fixture/engine-\xff".to_vec())),
    );
    let invocation = build_fastchess_invocation(
        &planned,
        &planned.pairs()[0].id,
        &engines,
        &native_path("/fixture/opening.pgn"),
        &native_path("/fixture/result.pgn"),
    )
    .unwrap();
    assert!(
        invocation
            .args
            .iter()
            .any(|argument| argument.as_bytes() == b"cmd=/fixture/engine-\xff")
    );
}

#[cfg(feature = "test-fixtures")]
mod fixture_mode_binary {
    //! Actual fixture binary checks, not runner/NN/strength acceptance.
    use std::{
        cell::RefCell,
        io::{self, Read, Write},
        process::{Child, Command, Output, Stdio},
        rc::Rc,
        sync::mpsc::{self, Receiver},
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };

    const WAIT: Duration = Duration::from_secs(5);
    const MAX_OUTPUT: usize = 16 * 1024;

    #[derive(Debug, Default)]
    struct CleanupFailures {
        kill: Option<io::Error>,
        poll: Option<io::Error>,
        reap_timed_out: bool,
        readers_timed_out: bool,
        reader_panicked: bool,
    }
    impl CleanupFailures {
        fn is_empty(&self) -> bool {
            self.kill.is_none()
                && self.poll.is_none()
                && !self.reap_timed_out
                && !self.readers_timed_out
                && !self.reader_panicked
        }
    }

    struct Attempt {
        child: Child,
        readers: Vec<JoinHandle<()>>,
        // The caller retains owned cleanup failures after this guard is dropped.
        // This local receipt is never shared with the pipe producer threads.
        cleanup: Rc<RefCell<CleanupFailures>>,
    }
    impl Drop for Attempt {
        fn drop(&mut self) {
            let mut cleanup = self.cleanup.borrow_mut();
            if !matches!(self.child.try_wait(), Ok(Some(_))) {
                cleanup.kill = self.child.kill().err();
            }
            let until = Instant::now() + Duration::from_secs(1);
            loop {
                match self.child.try_wait() {
                    Ok(Some(_)) => break,
                    Err(error) => {
                        cleanup.poll = Some(error);
                        break;
                    }
                    Ok(None) if Instant::now() >= until => {
                        cleanup.reap_timed_out = true;
                        break;
                    }
                    Ok(None) => thread::sleep(Duration::from_millis(10)),
                }
            }
            let until = Instant::now() + Duration::from_secs(1);
            while self.readers.iter().any(|reader| !reader.is_finished()) && Instant::now() < until
            {
                thread::sleep(Duration::from_millis(10));
            }
            cleanup.readers_timed_out = self.readers.iter().any(|reader| !reader.is_finished());
            for reader in self.readers.drain(..) {
                if reader.is_finished() {
                    cleanup.reader_panicked |= reader.join().is_err();
                }
            }
        }
    }
    fn capture(
        reader: impl Read + Send + 'static,
    ) -> (Receiver<io::Result<Vec<u8>>>, JoinHandle<()>) {
        let (send, recv) = mpsc::sync_channel(1);
        let reader = thread::spawn(move || {
            let mut bytes = Vec::new();
            let result = reader
                .take((MAX_OUTPUT + 1) as u64)
                .read_to_end(&mut bytes)
                .map(|_| bytes);
            let _ = send.send(result);
        });
        (recv, reader)
    }
    fn run(arguments: &[&str], input: &str) -> Output {
        assert!(
            input.len() <= 1024,
            "one bounded fixture input fits the process pipe"
        );
        let child = Command::new(env!("CARGO_BIN_EXE_rz-arena-fixture-engine"))
            .args(arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn the explicitly enabled Cargo fixture binary");
        let cleanup = Rc::new(RefCell::new(CleanupFailures::default()));
        let mut attempt = Attempt {
            child,
            readers: Vec::new(),
            cleanup: Rc::clone(&cleanup),
        };
        let (stdout, out) = capture(attempt.child.stdout.take().unwrap());
        let (stderr, err) = capture(attempt.child.stderr.take().unwrap());
        attempt.readers = vec![out, err];
        attempt
            .child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .expect("write bounded fixture command sequence and close stdin");
        let until = Instant::now() + WAIT;
        let status = loop {
            if let Some(status) = attempt.child.try_wait().expect("poll fixture exit") {
                break status;
            }
            assert!(
                Instant::now() < until,
                "fixture exceeded its five-second external test deadline"
            );
            thread::sleep(Duration::from_millis(10));
        };
        let stdout = stdout
            .recv_timeout(WAIT)
            .expect("bounded fixture stdout EOF")
            .expect("read fixture stdout");
        let stderr = stderr
            .recv_timeout(WAIT)
            .expect("bounded fixture stderr EOF")
            .expect("read fixture stderr");
        assert!(
            stdout.len() <= MAX_OUTPUT && stderr.len() <= MAX_OUTPUT,
            "bounded fixture process evidence"
        );
        let output = Output {
            status,
            stdout,
            stderr,
        };
        drop(attempt);
        let cleanup = cleanup.borrow();
        assert!(
            cleanup.is_empty(),
            "fixture cleanup failures retained: {cleanup:?}"
        );
        output
    }
    fn streams(output: &Output) -> (String, String) {
        (
            String::from_utf8(output.stdout.clone()).unwrap(),
            String::from_utf8(output.stderr.clone()).unwrap(),
        )
    }

    #[test]
    fn crash_mode_option_changes_the_actual_process_exit_after_newgame() {
        let result = run(
            &[],
            "uci\nsetoption name FixtureMode value crash\nucinewgame\nisready\nposition startpos\ngo\n",
        );
        assert_eq!(result.status.code(), Some(17));
        let (protocol, diagnostics) = streams(&result);
        assert!(protocol.contains("option name FixtureMode type combo default normal var normal var crash var illegal var timeout\n"));
        assert!(protocol.contains("mode=crash\nreadyok\n"));
        assert!(!protocol.contains("bestmove"));
        assert!(
            diagnostics.contains("synthetic applied Ponder=false Threads=1 Seed=0 mode=crash\n")
        );
        assert!(diagnostics.contains("synthetic injected crash: exit 17"));
    }

    #[test]
    fn illegal_mode_option_is_applied_and_survives_newgame() {
        let result = run(
            &[],
            "uci\nsetoption name FixtureMode value illegal\nposition startpos moves e2e4 e7e5\ngo\nucinewgame\nisready\nposition startpos\ngo\nquit\n",
        );
        assert!(result.status.success());
        let (protocol, diagnostics) = streams(&result);
        assert_eq!(
            protocol
                .lines()
                .filter(|line| line.starts_with("bestmove "))
                .collect::<Vec<_>>(),
            ["bestmove a1a8", "bestmove a1a8"]
        );
        assert!(protocol.contains("mode=illegal\nreadyok\n"));
        assert!(diagnostics.contains("synthetic go mode=illegal prefix_len=0"));
    }

    #[test]
    fn timeout_mode_withholds_reply_until_stop_and_newgame_keeps_the_mode() {
        let result = run(
            &[],
            "uci\nsetoption name FixtureMode value timeout\nposition startpos moves e2e4 e7e5\ngo\nisready\nstop\nucinewgame\nisready\ngo\nisready\nquit\n",
        );
        assert!(result.status.success());
        let (protocol, diagnostics) = streams(&result);
        let lines: Vec<_> = protocol.lines().collect();
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.starts_with("bestmove "))
                .copied()
                .collect::<Vec<_>>(),
            ["bestmove d1h5"]
        );
        let stopped = lines
            .iter()
            .position(|line| *line == "bestmove d1h5")
            .unwrap();
        let ready = lines.iter().position(|line| *line == "readyok").unwrap();
        assert!(
            ready < stopped,
            "withheld go must permit readyok before its stop reply"
        );
        assert_eq!(
            protocol
                .lines()
                .filter(|line| line.contains("mode=timeout"))
                .count(),
            3
        );
        assert!(diagnostics.contains("synthetic go mode=timeout prefix_len=0"));
    }

    #[test]
    fn fixture_cli_initial_mode_is_advertised_and_invalid_mode_values_fail_closed() {
        let result = run(
            &["--mode", "illegal"],
            "uci\nisready\nposition startpos\ngo\nquit\n",
        );
        assert!(result.status.success());
        let (protocol, _) = streams(&result);
        assert!(protocol.contains("option name FixtureMode type combo default illegal var normal var crash var illegal var timeout\n"));
        assert!(protocol.contains("bestmove a1a8\n"));
        for (value, expected) in [
            (
                "Normal",
                "mode must be exactly normal, crash, illegal or timeout",
            ),
            (
                "CRASH",
                "mode must be exactly normal, crash, illegal or timeout",
            ),
            (
                "random",
                "mode must be exactly normal, crash, illegal or timeout",
            ),
            ("normal extra", "canonical FixtureMode values"),
        ] {
            let result = run(
                &[],
                &format!("setoption name FixtureMode value {value}\nposition startpos\ngo\n"),
            );
            assert_eq!(result.status.code(), Some(2));
            let (protocol, diagnostics) = streams(&result);
            assert!(
                protocol.is_empty(),
                "invalid option cannot launch a go or emit success"
            );
            assert!(diagnostics.contains(expected));
        }
    }
}
