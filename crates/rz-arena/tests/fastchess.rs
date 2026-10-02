use rz_arena::{
    ArenaPlan, FASTCHESS_SOURCE_COMMIT, FASTCHESS_SOURCE_URL, FASTCHESS_VERSION,
    FastchessInvocation, PlanLimits, build_fastchess_invocation,
};
use rz_experiments::{ClockSpec, EngineKind, RunManifest};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

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

fn commands() -> BTreeMap<String, PathBuf> {
    BTreeMap::from([
        ("candidate-fixture".into(), PathBuf::from("/proc/123/fd/10")),
        ("baseline-fixture".into(), PathBuf::from("/proc/123/fd/11")),
    ])
}

fn invoke(input: RunManifest) -> Result<FastchessInvocation, rz_arena::ArenaError> {
    let planned = plan(input);
    build_fastchess_invocation(
        &planned,
        &planned.pairs()[0].id,
        &commands(),
        Path::new("/fixture/opening.pgn"),
        Path::new("/fixture/pair.pgn"),
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
    // Verified against the pinned native runner with a subprocess stdout pipe:
    // this sink captured setoption/ucinewgame/full position and UCI replies.
    // Its -debug compatibility parser explicitly rejects that alternative.
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
            Path::new("/fixture/opening.pgn"),
            Path::new("/fixture/pair.pgn"),
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
    for maximum in [1, 2, 3, 5] {
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
                Path::new(opening),
                Path::new(output)
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
        engines.insert("candidate-fixture".into(), path.into());
        assert!(
            build_fastchess_invocation(
                &planned,
                id,
                &engines,
                Path::new("/fixture/input.pgn"),
                Path::new("/fixture/output.pgn")
            )
            .is_err()
        );
    }
    let mut engines = commands();
    engines.insert("extra-engine".into(), "/fixture/extra".into());
    assert!(
        build_fastchess_invocation(
            &planned,
            id,
            &engines,
            Path::new("/fixture/input.pgn"),
            Path::new("/fixture/output.pgn")
        )
        .is_err()
    );
    assert!(
        build_fastchess_invocation(
            &planned,
            "missing-pair",
            &commands(),
            Path::new("/fixture/input.pgn"),
            Path::new("/fixture/output.pgn")
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
        Path::new("/fixture/opening.pgn"),
        Path::new("/fixture/result.pgn"),
    )
    .unwrap();
    assert!(
        invocation
            .args
            .iter()
            .any(|argument| argument.as_bytes() == b"cmd=/fixture/engine-\xff")
    );
}
