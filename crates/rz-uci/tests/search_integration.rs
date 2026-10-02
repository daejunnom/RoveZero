//! Hand-authored tree oracle: this fixture exercises B boundaries, not chess rules/perft.
use std::sync::mpsc::sync_channel;
use std::time::{Duration, Instant};

use rz_search::time::TimeBudgetConfig;
use rz_search::{
    TreeLimits,
    driver::{CheckedPosition, Evaluation, Evaluator, SearchControl, SearchStatus, run_search},
};
use rz_uci::bridge::{BuildSearchSettings, SearchBinding, SideToMove};
use rz_uci::{
    Effect, EngineIdentity, Event, ParserLimits, PositionBase, PositionPort, PositionSpec,
    PreparedPosition, SearchCompletion, Session, handle_event, serve_events_with_handler,
};

#[derive(Clone, Debug)]
struct FixturePosition {
    side: SideToMove,
    selected: Option<bool>,
}

impl FixturePosition {
    fn root_moves(&self) -> [&'static str; 2] {
        match self.side {
            SideToMove::White => ["e2e4", "d2d4"],
            SideToMove::Black => ["e7e5", "d7d5"],
        }
    }
}

impl CheckedPosition for FixturePosition {
    type Move = &'static str;
    type Error = &'static str;
    fn classify(&self) -> Result<Option<f64>, Self::Error> {
        // The independent artificial oracle makes the first child lose, the second win.
        Ok(self.selected.map(|first| if first { -1.0 } else { 1.0 }))
    }
    fn legal_moves(&self) -> Result<Vec<Self::Move>, Self::Error> {
        Ok(if self.selected.is_some() {
            vec![]
        } else {
            self.root_moves().to_vec()
        })
    }
    fn play(&self, mv: &Self::Move) -> Result<Self, Self::Error> {
        if self.selected.is_some() {
            return Err("fixture terminal cannot transition");
        }
        let [first, second] = self.root_moves();
        let selected = if *mv == first {
            true
        } else if *mv == second {
            false
        } else {
            return Err("fixture illegal move");
        };
        Ok(Self {
            side: self.side,
            selected: Some(selected),
        })
    }
}

struct FixturePort;
impl PositionPort for FixturePort {
    type Snapshot = FixturePosition;
    type Error = &'static str;
    fn prepare(
        &self,
        spec: &PositionSpec,
    ) -> Result<PreparedPosition<Self::Snapshot>, Self::Error> {
        let side = match &spec.base {
            PositionBase::StartPos => SideToMove::White,
            PositionBase::Fen(fen) if fen.split_ascii_whitespace().nth(1) == Some("b") => {
                SideToMove::Black
            }
            PositionBase::Fen(_) => SideToMove::White,
        };
        let mut snapshot = FixturePosition {
            side,
            selected: None,
        };
        for text in &spec.moves {
            let legal = snapshot.legal_moves()?;
            let mv = legal
                .into_iter()
                .find(|mv| *mv == text)
                .ok_or("fixture trace rejected")?;
            snapshot = snapshot.play(&mv)?;
        }
        let exact_terminal = snapshot.classify()?.is_some();
        let legal_moves = snapshot
            .legal_moves()?
            .into_iter()
            .map(str::to_owned)
            .collect();
        Ok(PreparedPosition {
            snapshot,
            legal_moves,
            exact_terminal,
        })
    }
}

struct Oracle {
    calls: usize,
}
impl Evaluator<FixturePosition> for Oracle {
    type Error = &'static str;
    fn evaluate(
        &mut self,
        position: &FixturePosition,
        legal: &[&'static str],
        _: &SearchControl,
    ) -> Result<Evaluation, Self::Error> {
        assert!(
            position.selected.is_none(),
            "exact terminals must bypass evaluator"
        );
        assert_eq!(legal.len(), 2);
        self.calls += 1;
        Ok(Evaluation {
            priors: vec![0.5, 0.5],
            wdl: [0.0, 1.0, 0.0],
        })
    }
}

fn session() -> Session<FixturePort> {
    Session::new(
        FixturePort,
        EngineIdentity {
            name: "B fixture".into(),
            author: "RoveZero".into(),
        },
        vec![],
        ParserLimits::default(),
    )
    .unwrap()
}

fn settings() -> BuildSearchSettings {
    BuildSearchSettings {
        time_config: TimeBudgetConfig::default(),
        max_simulations: 64,
        untimed_limit: Duration::from_secs(2),
    }
}

fn start(session: &mut Session<FixturePort>, command: &str) -> (FixturePosition, SearchBinding) {
    let tick = Instant::now();
    let result = session.handle_line(command);
    assert!(result.accepted, "{:?}", result.diagnostics);
    let Effect::Start {
        ticket,
        snapshot,
        limits,
        ..
    } = result.effects.into_iter().last().expect("start effect")
    else {
        panic!("Start")
    };
    let binding = SearchBinding::new(ticket, &limits, snapshot.side, tick, settings()).unwrap();
    (snapshot, binding)
}

#[test]
fn uci_to_puct_to_single_bestmove_with_terminal_bypass() {
    let mut session = session();
    assert_eq!(session.handle_line("uci").protocol.last().unwrap(), "uciok");
    let (snapshot, binding) = start(&mut session, "go nodes 2");
    assert_eq!(session.handle_line("isready").protocol, ["readyok"]);
    let mut oracle = Oracle { calls: 0 };
    let result = run_search(
        &snapshot,
        &mut oracle,
        binding.control(),
        TreeLimits::default(),
    );
    assert!(matches!(result.status, SearchStatus::Completed));
    assert_eq!(result.best_move, Some("e2e4"));
    assert_eq!(result.counters.root_initializations, 1);
    assert_eq!(result.counters.accepted_backups, 2);
    assert_eq!(result.root_stats[0].1.visits, 2);
    assert_eq!(result.root_stats[0].1.value_sum, 2.0);
    assert_eq!(oracle.calls, 1);
    let completion = SearchCompletion::Completed {
        bestmove: result.best_move.map(str::to_owned),
    };
    assert_eq!(
        binding
            .complete(&mut session, completion.clone(), Instant::now())
            .protocol,
        ["bestmove e2e4"]
    );
    assert!(
        binding
            .complete(&mut session, completion, Instant::now())
            .protocol
            .is_empty()
    );
    assert!(session.handle_line("stop").protocol.is_empty());
}

#[test]
fn exact_deadline_closes_search_gate_and_keeps_previous_valid_move() {
    let mut session = session();
    let (snapshot, binding) = start(&mut session, "go movetime 1000");
    let early = Instant::now();
    assert!(binding.progress(&mut session, "d2d4", early).accepted);
    let result = binding.complete(
        &mut session,
        SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
        binding.budget().hard_deadline,
    );
    assert_eq!(result.protocol, ["bestmove d2d4"]);
    assert!(!result.accepted);
    let mut oracle = Oracle { calls: 0 };
    let stopped = run_search(
        &snapshot,
        &mut oracle,
        binding.control(),
        TreeLimits::default(),
    );
    assert_eq!(oracle.calls, 0);
    assert_eq!(stopped.counters.accepted_backups, 0);
    assert!(matches!(stopped.status, SearchStatus::Stopped(_)));
}

#[test]
fn replaced_position_keeps_new_search_independent_of_old_callbacks() {
    let mut session = session();
    let (_, old) = start(&mut session, "go nodes 2");
    let replacement = session
        .handle_line("position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    assert!(replacement.accepted);
    assert!(matches!(replacement.effects[0], Effect::Cancel { .. }));
    old.cancel();
    let (snapshot, new) = start(
        &mut session,
        "go wtime 60000 btime 1000 winc 10000 binc 100 movestogo 1 nodes 2",
    );
    assert_eq!(new.budget().allocation, Duration::from_millis(1000));
    let late = old.complete(
        &mut session,
        SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
        old.budget().hard_deadline,
    );
    assert!(!late.accepted);
    assert!(late.protocol.is_empty());
    assert!(
        !new.control()
            .cancellation
            .load(std::sync::atomic::Ordering::Acquire)
    );
    let mut oracle = Oracle { calls: 0 };
    let result = run_search(&snapshot, &mut oracle, new.control(), TreeLimits::default());
    assert_eq!(
        new.complete(
            &mut session,
            SearchCompletion::Completed {
                bestmove: result.best_move.map(str::to_owned)
            },
            Instant::now()
        )
        .protocol,
        ["bestmove e7e5"]
    );
}

#[test]
fn invalid_trace_preserves_running_root_then_stop_closes_its_worker() {
    let mut session = session();
    let (snapshot, binding) = start(&mut session, "go nodes 2");
    let invalid = session.handle_line("position startpos moves e2e4 a7a6");
    assert!(!invalid.accepted);
    assert!(invalid.effects.is_empty());
    assert_eq!(session.active_ticket().as_ref(), Some(binding.ticket()));
    binding.progress(&mut session, "d2d4", Instant::now());
    let stop = session.handle_line("stop");
    // A bootstrap processes this cancellation before writing stop.protocol.
    assert!(matches!(stop.effects[0], Effect::Cancel { .. }));
    binding.cancel();
    assert_eq!(stop.protocol, ["bestmove d2d4"]);
    let mut oracle = Oracle { calls: 0 };
    let result = run_search(
        &snapshot,
        &mut oracle,
        binding.control(),
        TreeLimits::default(),
    );
    assert_eq!(result.counters.accepted_backups, 0);
    assert_eq!(oracle.calls, 0);
    assert!(
        binding
            .complete(
                &mut session,
                SearchCompletion::Completed {
                    bestmove: Some("e2e4".into())
                },
                Instant::now()
            )
            .protocol
            .is_empty()
    );
}

#[test]
fn event_owner_rechecks_queued_completion_at_deadline_before_output() {
    let mut session = session();
    let (_, binding) = start(&mut session, "go movetime 1000");
    binding.progress(&mut session, "d2d4", Instant::now());
    let (sender, events) = sync_channel(4);
    // Completion is queued before the timer; the dispatch tick reaches the strict
    // boundary. Producer ordering must not let that completion replace d2d4.
    sender
        .send(Event::Complete {
            ticket: binding.ticket().clone(),
            completion: SearchCompletion::Completed {
                bestmove: Some("e2e4".into()),
            },
        })
        .unwrap();
    sender
        .send(Event::Deadline(binding.ticket().clone()))
        .unwrap();
    sender.send(Event::Line("isready".into())).unwrap();
    sender.send(Event::Line("quit".into())).unwrap();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut canceled = 0;
    serve_events_with_handler(
        &mut session,
        &events,
        &mut stdout,
        &mut stderr,
        |session, event| match event {
            Event::Complete { ticket, completion } if &ticket == binding.ticket() => {
                binding.complete(session, completion, binding.budget().hard_deadline)
            }
            Event::Deadline(ticket) if &ticket == binding.ticket() => {
                binding.expire(session, binding.budget().hard_deadline)
            }
            other => handle_event(session, other),
        },
        |effect| {
            if let Effect::Cancel { ticket, .. } = effect {
                assert_eq!(&ticket, binding.ticket());
                binding.cancel();
                canceled += 1;
            }
            Ok::<_, &'static str>(())
        },
    )
    .unwrap();
    assert_eq!(
        String::from_utf8(stdout).unwrap(),
        "bestmove d2d4\nreadyok\n"
    );
    assert!(String::from_utf8(stderr).unwrap().contains("SearchExpired"));
    assert_eq!(canceled, 1);
    assert!(
        binding
            .control()
            .cancellation
            .load(std::sync::atomic::Ordering::Acquire)
    );
}

#[test]
fn event_owner_keeps_infinite_resource_expiry_silent_until_stop() {
    let mut session = session();
    let (_, binding) = start(&mut session, "go infinite");
    binding.progress(&mut session, "d2d4", Instant::now());
    let (sender, events) = sync_channel(3);
    sender
        .send(Event::Deadline(binding.ticket().clone()))
        .unwrap();
    sender.send(Event::Line("stop".into())).unwrap();
    sender.send(Event::Line("quit".into())).unwrap();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    serve_events_with_handler(
        &mut session,
        &events,
        &mut stdout,
        &mut stderr,
        |session, event| match event {
            Event::Deadline(ticket) if &ticket == binding.ticket() => {
                let out = binding.expire(session, binding.budget().hard_deadline);
                assert!(out.protocol.is_empty());
                assert_eq!(session.active_ticket().as_ref(), Some(binding.ticket()));
                out
            }
            other => handle_event(session, other),
        },
        |effect| {
            if let Effect::Cancel { .. } = effect {
                binding.cancel();
            }
            Ok::<_, &'static str>(())
        },
    )
    .unwrap();
    assert_eq!(String::from_utf8(stdout).unwrap(), "bestmove d2d4\n");
    assert!(
        String::from_utf8(stderr)
            .unwrap()
            .contains("ResourceWallLimit")
    );
}
