//! Independent two-ply artificial tree. This is B contract wiring, not chess/perft.
use std::collections::VecDeque;
use std::sync::{Arc, mpsc::sync_channel};
use std::time::{Duration, Instant};

use rz_contracts::*;
use rz_search::TreeLimits;
use rz_search::contract_time::InstantClock;
use rz_search::contracts::{
    ContractPosition, ContractPumpEvent, ContractSearch, ContractSearchConfig,
    ContractSearchStatus, IdAllocator,
};
use rz_search::time::TimeBudgetConfig;
use rz_uci::bridge::BuildSearchSettings;
use rz_uci::contracts::ContractSessionOwner;
use rz_uci::{
    Effect, EngineIdentity, Event, ParserLimits, PositionBase, PositionPort, PositionSpec,
    PreparedPosition, SearchCompletion, ServeFailure, Session, serve_events_with_handler,
};

const EPOCH: ProcessEpoch = ProcessEpoch(77);

#[derive(Debug)]
struct State(u8);

#[derive(Clone)]
struct Position {
    snapshot: PositionSnapshot<State>,
    legal: LegalMoveView,
}

fn digest(tag: u8) -> Digest {
    // Synthetic fixture identity; no real encoder/hash attestation is claimed.
    Digest([tag; 32])
}

fn movement(ply: u8) -> Move {
    let (from, to) = if ply == 0 { (12, 28) } else { (52, 36) };
    Move::new(
        Square::try_new(from).unwrap(),
        Square::try_new(to).unwrap(),
        None,
    )
    .unwrap()
}

impl Position {
    fn new(ply: u8) -> Self {
        let state = StateIdentity {
            owner: OwnerId(11),
            revision: StateRevision(u64::from(ply)),
            semantic: digest(ply),
        };
        let side = if ply == 1 { Color::Black } else { Color::White };
        let play_status = if ply == 2 {
            PlayStatus::Terminal {
                reason: TerminalReason::Checkmate,
                winner: Some(Color::Black),
            }
        } else {
            PlayStatus::Ongoing
        };
        let classification = PositionClassification {
            play_status,
            claims: Arc::from([]),
            history: HistoryCompleteness::Complete,
            rules_profile: digest(90),
        };
        Self {
            snapshot: PositionSnapshot::try_new(state, Arc::new(State(ply)), side, classification)
                .unwrap(),
            legal: LegalMoveView::try_new(
                state,
                LegalOrderIdentity(digest(ply + 10)),
                if ply < 2 { vec![movement(ply)] } else { vec![] },
                2,
            )
            .unwrap(),
        }
    }
}

impl ContractPosition for Position {
    type State = State;
    fn snapshot(&self) -> &PositionSnapshot<State> {
        &self.snapshot
    }
    fn legal(&self) -> &LegalMoveView {
        &self.legal
    }
    fn play(&self, mv: &Move) -> Result<Self, ContractError> {
        let ply = self.snapshot.state().0;
        if ply >= 2 || *mv != movement(ply) {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Contract,
                "artificial transition rejected",
            ));
        }
        Ok(Self::new(ply + 1))
    }
    fn validate_authority(&self) -> Result<(), ContractError> {
        let ply = self.snapshot.state().0;
        if self.snapshot.identity().revision.0 != u64::from(ply)
            || self.legal.state() != self.snapshot.identity()
        {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Contract,
                "artificial authority differs",
            ));
        }
        Ok(())
    }
}

struct Port;
impl PositionPort for Port {
    type Snapshot = Position;
    type Error = ContractError;
    fn prepare(&self, spec: &PositionSpec) -> Result<PreparedPosition<Position>, ContractError> {
        if !matches!(spec.base, PositionBase::StartPos) {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Contract,
                "fixture supports startpos only",
            ));
        }
        let mut position = Position::new(0);
        for text in &spec.moves {
            let expected = match position.snapshot.state().0 {
                0 => "e2e4",
                1 => "e7e5",
                _ => "",
            };
            if text != expected {
                return Err(ContractError::new(
                    ErrorCode::InvalidInput,
                    Stage::Contract,
                    "artificial trace rejected",
                ));
            }
            position = position.play(&movement(position.snapshot.state().0))?;
        }
        let legal_moves = match position.snapshot.state().0 {
            0 => vec!["e2e4".to_owned()],
            1 => vec!["e7e5".to_owned()],
            _ => vec![],
        };
        let exact_terminal = position.snapshot.classification().play_status != PlayStatus::Ongoing;
        Ok(PreparedPosition {
            snapshot: position,
            legal_moves,
            exact_terminal,
        })
    }
}

#[derive(Default)]
struct Runtime {
    results: VecDeque<EvalResult>,
    requests: Vec<Arc<EvalRequest<State>>>,
    canceled: Vec<RequestId>,
}

impl Evaluator<State> for Runtime {
    fn submit(&mut self, request: Arc<EvalRequest<State>>) -> Result<(), ContractError> {
        // These are finalized, owned mock results, not raw device callbacks.
        let ply = request.position().state().0;
        assert!(ply < 2, "exact terminal must bypass evaluation");
        let wdl = if ply == 1 {
            Wdl::try_new(0.75, 0.25, 0.0, 1e-5).unwrap()
        } else {
            Wdl::try_new(0.0, 1.0, 0.0, 1e-5).unwrap()
        };
        self.results.push_back(EvalResult::Completed(EvalOutput {
            context: request.context(),
            legal: request.legal().clone(),
            policy: LegalPolicy::try_new(vec![1.0], 1e-5).unwrap(),
            wdl,
            viewpoint: Viewpoint::SideToMove,
            actual: ActualCompute {
                precision: PrecisionProfile::Fp32,
                steps: 1,
                full: true,
                backend: request.context().backend,
                execution: Some(ExecutionId::new(EPOCH, request.context().request.sequence)),
                provenance: CacheProvenance::Computed,
            },
        }));
        self.requests.push(request);
        Ok(())
    }
    fn poll(&mut self) -> Option<EvalResult> {
        self.results.pop_front()
    }
    fn cancel(&mut self, request: RequestId) -> Result<(), ContractError> {
        // Logical cancellation leaves queued owned results with the runtime.
        self.canceled.push(request);
        Ok(())
    }
}

fn setup() -> (
    Session<Port>,
    ContractSessionOwner,
    Arc<ModelDescriptor>,
    Arc<IdAllocator>,
) {
    let model = Arc::new(
        ModelDescriptor::try_new(
            ModelHandle {
                owner: OwnerId(22),
                slot: 0,
                generation: SlotGeneration(1),
                manifest: digest(50),
            },
            EncodingDescriptor {
                handle: EncodingHandle {
                    owner: OwnerId(33),
                    slot: 0,
                    generation: SlotGeneration(1),
                    manifest: digest(60),
                },
                history_length: 2,
                action_map: digest(61),
                history_policy: digest(62),
            },
            vec![PrecisionProfile::Fp32],
            1,
            1,
        )
        .unwrap(),
    );
    let scope = AcceptanceScope {
        game: GameGeneration(0),
        root: RootGeneration(0),
        model: model.handle(),
        encoding: model.encoding().handle,
        backend: digest(70),
    };
    let session = Session::new(
        Port,
        EngineIdentity {
            name: "B typed fixture".into(),
            author: "RoveZero".into(),
        },
        vec![],
        ParserLimits::default(),
    )
    .unwrap();
    let owner = ContractSessionOwner::new(
        scope,
        EPOCH,
        InstantClock::new(ClockDomain(EPOCH), Instant::now()),
        BuildSearchSettings {
            time_config: TimeBudgetConfig::default(),
            max_simulations: 8,
            untimed_limit: Duration::from_secs(10),
        },
    )
    .unwrap();
    (session, owner, model, Arc::new(IdAllocator::new(EPOCH)))
}

fn start(
    session: &mut Session<Port>,
    owner: &mut ContractSessionOwner,
    model: &Arc<ModelDescriptor>,
    ids: &Arc<IdAllocator>,
    command: &str,
) -> ContractSearch<Position> {
    let out = owner.handle_line(session, command, |p| Ok(p.snapshot.side_to_move()));
    assert!(
        out.session.accepted,
        "{:?} {:?}",
        out.errors, out.session.diagnostics
    );
    let Effect::Start { snapshot, .. } = out.session.effects.into_iter().last().unwrap() else {
        panic!("Start")
    };
    let config = ContractSearchConfig {
        scope: owner.scope(),
        model: Arc::clone(model),
        precision: PrecisionProfile::Fp32,
        compute: ComputeBudget {
            min_steps: 1,
            max_steps: 1,
            require_full: true,
        },
        bytes: ByteBudget {
            host: 4096,
            device: 0,
            pinned: 0,
        },
        policy_tolerance: 1e-5,
        wdl_tolerance: 1e-5,
        cancellation: owner.cancellation().unwrap().token(),
        deadlines: *owner.deadlines().unwrap(),
        ids: Arc::clone(ids),
        max_simulations: owner.control().unwrap().max_simulations,
        tree_limits: TreeLimits::default(),
    };
    ContractSearch::new(snapshot, config).unwrap()
}

fn pump(
    search: &mut ContractSearch<Position>,
    runtime: &mut Runtime,
    owner: &ContractSessionOwner,
) -> ContractPumpEvent {
    search.pump(
        runtime,
        owner.clock(),
        || owner.scope(),
        |p, _encoding| Ok(EvalInputKey(digest(p.snapshot.state().0 + 80))),
    )
}

#[test]
fn typed_uci_async_evaluation_two_ply_backup_and_single_bestmove() {
    let (mut session, mut owner, model, ids) = setup();
    let mut search = start(&mut session, &mut owner, &model, &ids, "go nodes 2");
    let ticket = owner.active_ticket().unwrap().clone();
    let mut runtime = Runtime::default();
    for _ in 0..12 {
        pump(&mut search, &mut runtime, &owner);
        if search.is_finished() {
            break;
        }
    }
    let outcome = search.outcome();
    assert!(matches!(outcome.status, ContractSearchStatus::Completed));
    assert_eq!(outcome.counters.root_initializations, 1);
    assert_eq!(outcome.counters.accepted_backups, 2);
    assert_eq!(outcome.counters.reservations_released, 3);
    assert_eq!(outcome.root_stats[0].1.visits, 2);
    assert_eq!(outcome.root_stats[0].1.value_sum, -1.75);
    assert_eq!(runtime.requests.len(), 2);
    assert_ne!(
        runtime.requests[0].context().selection,
        runtime.requests[1].context().selection
    );
    let event = Event::Complete {
        ticket,
        completion: SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
    };
    assert_eq!(
        owner
            .handle_event(&mut session, event.clone())
            .session
            .protocol,
        ["bestmove e2e4"]
    );
    assert!(
        owner
            .handle_event(&mut session, event)
            .session
            .protocol
            .is_empty()
    );
}

#[test]
fn uci_stop_closes_common_request_before_output_and_late_owned_result() {
    let (mut session, mut owner, model, ids) = setup();
    let mut search = start(&mut session, &mut owner, &model, &ids, "go nodes 2");
    let ticket = owner.active_ticket().unwrap().clone();
    let mut runtime = Runtime::default();
    assert!(matches!(
        pump(&mut search, &mut runtime, &owner),
        ContractPumpEvent::Submitted { .. }
    ));
    let request = Arc::clone(&runtime.requests[0]);
    let out = owner.handle_line(&mut session, "stop", |p| Ok(p.snapshot.side_to_move()));
    assert!(request.cancel_token().is_canceled());
    assert_eq!(out.session.protocol, ["bestmove e2e4"]);
    assert!(matches!(
        pump(&mut search, &mut runtime, &owner),
        ContractPumpEvent::Finished
    ));
    assert_eq!(runtime.canceled, [request.context().request]);
    assert_eq!(
        runtime.results.len(),
        1,
        "logical cancel must not steal runtime-owned results"
    );
    assert_eq!(search.outcome().counters.reservations_released, 1);
    assert_eq!(search.outcome().counters.root_initializations, 0);
    assert!(
        owner
            .handle_event(
                &mut session,
                Event::Complete {
                    ticket,
                    completion: SearchCompletion::Completed {
                        bestmove: Some("e2e4".into())
                    },
                }
            )
            .session
            .protocol
            .is_empty()
    );
}

#[test]
fn newgame_invalidates_old_request_and_reuses_global_allocator_without_ids_repeating() {
    let (mut session, mut owner, model, ids) = setup();
    let mut old = start(&mut session, &mut owner, &model, &ids, "go nodes 2");
    let old_ticket = owner.active_ticket().unwrap().clone();
    let mut old_runtime = Runtime::default();
    pump(&mut old, &mut old_runtime, &owner);
    let old_context = old_runtime.requests[0].context();
    let reset = owner.handle_line(&mut session, "ucinewgame", |p| {
        Ok(p.snapshot.side_to_move())
    });
    assert!(
        reset
            .session
            .effects
            .iter()
            .any(|e| matches!(e, Effect::NewGame))
    );
    assert_eq!(owner.scope().game, GameGeneration(1));
    assert_eq!(owner.scope().model, model.handle());
    let mut new = start(&mut session, &mut owner, &model, &ids, "go nodes 1");
    let mut runtime = Runtime::default();
    pump(&mut new, &mut runtime, &owner);
    let context = runtime.requests[0].context();
    assert!(context.request.sequence > old_context.request.sequence);
    assert!(context.selection.sequence > old_context.selection.sequence);
    assert_eq!(context.request.epoch, old_context.request.epoch);
    assert_ne!(context.game, old_context.game);
    assert_ne!(context.root, old_context.root);
    let late = owner.handle_event(
        &mut session,
        Event::Complete {
            ticket: old_ticket,
            completion: SearchCompletion::Completed {
                bestmove: Some("e2e4".into()),
            },
        },
    );
    assert!(late.session.protocol.is_empty());
    assert!(!runtime.requests[0].cancel_token().is_canceled());
    pump(&mut old, &mut old_runtime, &owner);
    assert_eq!(old.outcome().counters.accepted_backups, 0);
}

#[test]
fn owner_deadline_ends_unresponsive_evaluation_without_backup_or_duplicate_output() {
    let (mut session, mut owner, model, ids) = setup();
    let mut search = start(&mut session, &mut owner, &model, &ids, "go movetime 1000");
    let ticket = owner.active_ticket().unwrap().clone();
    let hard = owner.active_binding().unwrap().budget().hard_deadline;
    let mut runtime = Runtime::default();
    pump(&mut search, &mut runtime, &owner);
    runtime.results.clear();
    assert!(matches!(
        pump(&mut search, &mut runtime, &owner),
        ContractPumpEvent::Waiting
    ));
    let expiry = owner.handle_event_at(&mut session, Event::Deadline(ticket.clone()), hard);
    assert_eq!(expiry.session.protocol, ["bestmove e2e4"]);
    assert!(expiry.errors.iter().any(|e| e.code == ErrorCode::Expired));
    pump(&mut search, &mut runtime, &owner);
    assert!(search.is_finished());
    assert_eq!(runtime.canceled.len(), 1);
    assert_eq!(search.outcome().counters.root_initializations, 0);
    assert_eq!(search.outcome().counters.reservations_released, 1);
    assert!(
        owner
            .handle_event_at(&mut session, Event::Deadline(ticket), hard)
            .session
            .protocol
            .is_empty()
    );
}

#[test]
fn transport_failure_closes_common_admission_even_when_cancel_dispatch_fails() {
    struct BrokenTransport;
    impl std::io::Write for BrokenTransport {
        fn write(&mut self, _bytes: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "fixture closed output",
            ))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let (mut session, mut owner, model, ids) = setup();
    let mut search = start(&mut session, &mut owner, &model, &ids, "go nodes 2");
    let mut runtime = Runtime::default();
    pump(&mut search, &mut runtime, &owner);
    let token = runtime.requests[0].cancel_token().clone();
    let (sender, receiver) = sync_channel(1);
    sender.send(Event::Line("isready".into())).unwrap();
    let mut stdout = BrokenTransport;
    let mut diagnostics = Vec::new();
    let mut shutdowns = 0;
    let failure = serve_events_with_handler(
        &mut session,
        &receiver,
        &mut stdout,
        &mut diagnostics,
        |session, event| match event {
            Event::Line(line) => {
                owner
                    .handle_line(session, &line, |p| Ok(p.snapshot.side_to_move()))
                    .session
            }
            other => owner.handle_event(session, other).session,
        },
        |effect| match effect {
            Effect::Cancel { .. } => {
                assert!(
                    token.is_canceled(),
                    "admission must close before fallible runtime cancel"
                );
                Err("fixture cancel dispatch failed")
            }
            Effect::Shutdown => {
                shutdowns += 1;
                Ok(())
            }
            _ => Ok(()),
        },
    )
    .unwrap_err();
    assert!(
        matches!(failure.failure, ServeFailure::Io(ref e) if e.kind() == std::io::ErrorKind::BrokenPipe)
    );
    assert_eq!(failure.cleanup_failures, ["fixture cancel dispatch failed"]);
    assert_eq!(shutdowns, 1);
    assert!(session.is_closed());
    assert!(token.is_canceled());
    assert!(owner.active_ticket().is_none());
    pump(&mut search, &mut runtime, &owner);
    assert_eq!(search.outcome().counters.root_initializations, 0);
    assert_eq!(search.outcome().counters.reservations_released, 1);
}
