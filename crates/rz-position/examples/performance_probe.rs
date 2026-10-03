//! Bounded CPU diagnostic for A experiments. Timings are process-local samples;
//! compare fresh processes serially; this is not an end-to-end search benchmark.
#![forbid(unsafe_code)]

use std::{hint::black_box, time::Instant};

use rz_contracts::OwnerId;
use rz_position::{
    contracts::{ContractPosition, ContractState},
    PlayStatus, Position,
};

const TRACE: &str = include_str!("../tests/fixtures/performance_trace.txt");

fn fixtures() -> Vec<(String, Position)> {
    let trace: Vec<_> = TRACE.split_whitespace().collect();
    assert_eq!(trace.len(), 256);
    let mut result = Vec::new();
    let mut position = Position::startpos();
    for ply in 0..=trace.len() {
        if [0, 16, 64, 128, 256].contains(&ply) {
            assert!(matches!(
                position.classify_position().unwrap().play_status,
                PlayStatus::Ongoing
            ));
            result.push((format!("trace-{ply}"), position.clone()));
        }
        if let Some(mv) = trace.get(ply) {
            position.make_uci(mv).unwrap();
        }
    }
    result
}

fn owner(sequence: &mut u64, position: Position) -> ContractPosition {
    *sequence = sequence.checked_add(1).expect("bounded owner sequence");
    ContractPosition::new(OwnerId(*sequence), position)
}

fn emit(name: &str, view: &ContractState) {
    let rules = view.rules();
    let state = rules.snapshot();
    println!(
        "STATE {name} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
        view.snapshot().identity().semantic,
        view.legal_moves().order(),
        view.legal_moves().moves(),
        rules.classification(),
        view.snapshot().classification(),
        (
            state.history_origin(),
            state.history_completeness(),
            state.revision()
        ),
        view.terminal_wdl(),
    );
    for (index, item) in state.known_history().enumerate() {
        println!("HISTORY {name} {index} {}", item.to_fen());
    }
}

fn witness() {
    let mut fixtures = fixtures();
    for (name, fen) in [
        ("unknown", Position::startpos().to_fen()),
        ("castling", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 99 1".into()),
        ("ep", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2".into()),
        ("ep-pin", "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 2".into()),
        ("promotion", "1r2k3/P7/8/8/8/8/7p/R3K3 w Q - 99 1".into()),
        ("mate", "7k/6Q1/6K1/8/8/8/8/8 b - - 150 76".into()),
        (
            "max-counter",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 4294967295 4294967295".into(),
        ),
    ] {
        fixtures.push((name.into(), Position::from_fen(&fen).unwrap()));
    }
    let mut repeated = Position::startpos();
    for index in 0..16 {
        repeated
            .make_uci(["g1f3", "g8f6", "f3g1", "f6g8"][index % 4])
            .unwrap();
        if index == 6 || index == 15 {
            fixtures.push((format!("repeat-{}", index + 1), repeated.clone()));
        }
    }
    let mut sequence = 1_000;
    let mut states = 0;
    let mut children = 0;
    for (name, position) in fixtures {
        let live = owner(&mut sequence, position);
        let before = live.position().snapshot();
        let view = match live.export() {
            Ok(view) => view,
            Err(error) => {
                println!("ERROR {name} {error:?}");
                assert!(live.position().snapshot().same_state(&before));
                continue;
            }
        };
        emit(&name, &view);
        states += 1;
        let again = live.export().unwrap();
        emit(&format!("{name}/again"), &again);
        states += 1;
        assert!(live.position().snapshot().same_state(&before));
        if !matches!(
            view.rules().classification().play_status,
            PlayStatus::Ongoing
        ) {
            continue;
        }
        for mv in view.legal_moves().moves() {
            sequence = sequence.checked_add(1).unwrap();
            let child = live.fork_from_view(OwnerId(sequence), &view, *mv).unwrap();
            let child_view = child.export().unwrap();
            emit(&format!("{name}/{mv:?}"), &child_view);
            states += 1;
            children += 1;
            assert!(live.position().snapshot().same_state(&before));

            let mut mutable = owner(&mut sequence, live.position().clone());
            let mutable_before = mutable.position().snapshot();
            let mutable_view = mutable.export().unwrap();
            let undo = mutable.make_from_view(&mutable_view, *mv).unwrap();
            let made = mutable.export().unwrap();
            assert_eq!(
                made.snapshot().identity().semantic,
                child_view.snapshot().identity().semantic
            );
            assert_eq!(made.legal_moves().moves(), child_view.legal_moves().moves());
            emit(&format!("{name}/{mv:?}/made"), &made);
            states += 1;
            mutable.unmake(undo).unwrap();
            assert!(mutable.position().snapshot().same_state(&mutable_before));
            let restored = mutable.export().unwrap();
            assert_eq!(
                restored.snapshot().identity().semantic,
                view.snapshot().identity().semantic
            );
            assert_eq!(restored.legal_moves().moves(), view.legal_moves().moves());
            emit(&format!("{name}/{mv:?}/restored"), &restored);
            states += 1;
            assert!(mutable.make_from_view(&mutable_view, *mv).is_err());
        }
    }
    println!("SUMMARY states={states} children={children}");
}

fn measure<T>(iterations: u32, mut operation: impl FnMut() -> T) -> f64 {
    for _ in 0..20 {
        black_box(operation());
    }
    let started = Instant::now();
    for _ in 0..iterations {
        black_box(operation());
    }
    started.elapsed().as_nanos() as f64 / f64::from(iterations)
}

fn timing(iterations: u32) {
    let mut sequence = 1_000;
    println!("fixture,history_len,legal_count,operation,iterations,ns_per_call");
    for (name, position) in fixtures() {
        let history = position.known_history_len();
        let legal = position.legal_moves().len();
        let live = owner(&mut sequence, position);
        // Setup and a first export are outside the timed warm-export region.
        let root = live.export().unwrap();
        let mv = root.legal_moves().moves()[0];
        let classified = measure(iterations, || live.position().classify_position().unwrap());
        println!("{name},{history},{legal},classify,{iterations},{classified:.3}");
        // Every cold sample has a fresh nonreused owner and empty digest memo.
        let cold = measure(iterations, || {
            owner(&mut sequence, live.position().clone())
                .export()
                .unwrap()
        });
        println!("{name},{history},{legal},cold-owner-export,{iterations},{cold:.3}");
        let warm = measure(iterations, || live.export().unwrap());
        println!("{name},{history},{legal},warm-export,{iterations},{warm:.3}");
        let fork = measure(iterations, || {
            sequence = sequence.checked_add(1).unwrap();
            live.fork_from_view(OwnerId(sequence), &root, mv)
                .unwrap()
                .export()
                .unwrap()
        });
        println!("{name},{history},{legal},fork-export,{iterations},{fork:.3}");
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [mode] if mode == "witness" => witness(),
        [mode, count] if mode == "time" => {
            let count: u32 = count.parse().expect("iteration count");
            assert!((1..=10_000).contains(&count), "bounded iteration count");
            timing(count);
        }
        _ => panic!("usage: performance_probe witness | time ITERATIONS(1..=10000)"),
    }
}
