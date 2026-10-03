//! OPT-00: bounded Rules/C encoding/D runtime/B search correctness witness.
//! Uses synthetic model heads and a manual clock; never a performance sample.
#![forbid(unsafe_code)]

use rz_contracts::*;
use rz_encoding::classical::HistoryFill;
use rz_eval::{
    RawOutput,
    contracts::{HOST_BYTES_PER_ITEM, MaiaBinding, encoding_manifest},
    mock::{Callback, Limits as MockLimits, ScriptIdentity, ScriptedBackend, Step},
    rules_projection::ClassicalProjection,
    runtime_bridge::{MockTicket, ScriptedRuntimeBackend},
};
use rz_position::{PositionLimits, contracts::RulesState};
use rz_runtime::{
    Clock, DrainState, Limits, Resources,
    contracts::{ContractEvaluator, ContractsAdapter, SharedScope},
};
use rz_search::{
    TreeLimits,
    contract_time::ContractDeadlines,
    contracts::{
        ContractPosition, ContractPumpEvent, ContractSearch, ContractSearchConfig,
        ContractSearchFailure, ContractSearchOutcome, ContractSearchStatus, IdAllocator,
    },
};
use rz_uci::{
    PositionBase, PositionPort, PositionSpec,
    engine::{OwnerRegistry, RulesSearchPosition, RulesUciPort},
};
use std::{
    cell::{Cell, RefCell},
    io::{self, BufWriter, Write},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

const EPOCH: ProcessEpoch = ProcessEpoch(71);
const HARD_TICK: u64 = 1_000_000;
const MAX_OUTPUT: usize = 64 * 1024 * 1024;
const TRACE: &str = include_str!("../../rz-position/tests/fixtures/performance_trace.txt");
type Runtime = ContractEvaluator<RulesState, ScriptedRuntimeBackend<ManualClock>, ManualClock>;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Default)]
struct ManualClock(Arc<AtomicU64>);
impl ManualClock {
    fn advance(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl Clock for ManualClock {
    type Tick = MonotonicTick;
    fn now(&self) -> MonotonicTick {
        MonotonicTick(self.0.load(Ordering::SeqCst))
    }
    fn elapsed(&self, earlier: MonotonicTick, later: MonotonicTick) -> Duration {
        Duration::from_nanos(later.0.saturating_sub(earlier.0))
    }
}
impl rz_runtime::contracts::ContractClock for ManualClock {
    fn domain(&self) -> ClockDomain {
        ClockDomain(EPOCH)
    }
}
impl rz_search::contract_time::ContractClock for ManualClock {
    fn domain(&self) -> ClockDomain {
        ClockDomain(EPOCH)
    }
    fn now(&self) -> std::result::Result<MonotonicTick, ContractError> {
        Ok(Clock::now(self))
    }
}

#[derive(Clone)]
struct Leaf {
    position: RulesSearchPosition,
    path: Vec<Move>,
}
#[derive(Clone)]
struct TrackedPosition {
    leaf: Leaf,
    last: Rc<RefCell<Leaf>>,
}
impl ContractPosition for TrackedPosition {
    type State = RulesState;
    fn snapshot(&self) -> &PositionSnapshot<RulesState> {
        self.leaf.position.snapshot()
    }
    fn legal(&self) -> &LegalMoveView {
        self.leaf.position.legal()
    }
    fn validate_authority(&self) -> std::result::Result<(), ContractError> {
        self.leaf.position.validate_authority()
    }
    fn play(&self, movement: &Move) -> std::result::Result<Self, ContractError> {
        let position = self.leaf.position.play(movement)?;
        let mut path = self.leaf.path.clone();
        path.push(*movement);
        let leaf = Leaf { position, path };
        self.last.replace(leaf.clone());
        Ok(Self {
            leaf,
            last: Rc::clone(&self.last),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Scenario {
    Normal,
    Cancel,
    Expired,
    Stale,
    NodeLimit,
    BadInput,
    BadOutput,
}
impl Scenario {
    const GUARDS: [Self; 6] = [
        Self::Cancel,
        Self::Expired,
        Self::Stale,
        Self::NodeLimit,
        Self::BadInput,
        Self::BadOutput,
    ];
}

struct CappedWriter<W> {
    inner: W,
    written: usize,
}
impl<W: Write> Write for CappedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_OUTPUT.saturating_sub(self.written) {
            return Err(io::Error::other("witness exceeded 64 MiB output limit"));
        }
        let written = self.inner.write(bytes)?;
        self.written += written;
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn emit_leaf(writer: &mut impl Write, tag: &str, leaf: &Leaf) -> io::Result<()> {
    let view = leaf.position.state();
    let state = view.rules().snapshot();
    // Do not print process-issued owner IDs. Fresh/stale authority is tested separately.
    writeln!(
        writer,
        "LEAF {tag} path={:?} semantic={:?} revision={:?} order={:?} moves={:?} rules={:?} shared={:?} origin={:?} completeness={:?}",
        leaf.path,
        view.snapshot().identity().semantic,
        view.snapshot().identity().revision,
        view.legal_moves().order(),
        view.legal_moves().moves(),
        view.rules().classification(),
        view.snapshot().classification(),
        state.history_origin(),
        state.history_completeness()
    )?;
    for (index, state) in state.known_history().enumerate() {
        writeln!(writer, "HISTORY {tag} {index} {}", state.to_fen())?;
    }
    Ok(())
}

struct WitnessRuntime<'a, W> {
    runtime: Runtime,
    projection: ClassicalProjection,
    last: Rc<RefCell<Leaf>>,
    writer: &'a mut W,
    write_failed: Rc<Cell<bool>>,
    delivered: Option<(RequestId, f64)>,
}
impl<W: Write> WitnessRuntime<'_, W> {
    fn preparation(&mut self, request: Arc<EvalRequest<RulesState>>) -> Result<()> {
        let sequence = request.context().request.sequence;
        let leaf = self.last.borrow();
        if leaf.position.snapshot().identity() != request.position().identity() {
            return Err("witness leaf and actual request differ".into());
        }
        emit_leaf(self.writer, &format!("request-{sequence}"), &leaf)?;
        let prepared = self.projection.prepare(Arc::clone(&request))?;
        let projection = self.projection.project(request.position().state())?;
        writeln!(
            self.writer,
            "PREPARE {sequence} input={:?} indices={:?} projection={:?}",
            request.context().input,
            prepared.indices(),
            projection.input()
        )?;
        write!(self.writer, "TENSOR-LE {sequence} ")?;
        for value in prepared.encoded().values() {
            for byte in value.to_le_bytes() {
                write!(self.writer, "{byte:02x}")?;
            }
        }
        writeln!(self.writer)?;
        Ok(())
    }
}
impl<W: Write> Evaluator<RulesState> for WitnessRuntime<'_, W> {
    fn submit(
        &mut self,
        request: Arc<EvalRequest<RulesState>>,
    ) -> std::result::Result<(), ContractError> {
        if let Err(error) = self.preparation(Arc::clone(&request)) {
            if let Some(error) = error.downcast_ref::<ContractError>() {
                return Err(*error);
            }
            self.write_failed.set(true);
            return Err(ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Admission,
                "witness write or leaf binding failed",
            ));
        }
        self.runtime.submit(request)
    }
    fn poll(&mut self) -> Option<EvalResult> {
        let result = self.runtime.poll()?;
        let emitted = match &result {
            EvalResult::Completed(output) => {
                let sequence = output.context.request.sequence;
                let wdl = output.wdl.probabilities();
                let normalized = output.wdl.normalized();
                self.delivered = Some((output.context.request, normalized[0] - normalized[2]));
                writeln!(
                    self.writer,
                    "OUTPUT {sequence} policy-bits={:?} wdl-bits={:?} actual={:?}",
                    output
                        .policy
                        .probabilities()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    wdl.map(f32::to_bits),
                    output.actual
                )
            }
            EvalResult::Failed(failure) => writeln!(
                self.writer,
                "RESULT failed {} {:?}",
                failure.error, failure.recovery
            ),
            EvalResult::Canceled(_) => writeln!(self.writer, "RESULT canceled"),
            EvalResult::Expired(_) => writeln!(self.writer, "RESULT expired"),
            EvalResult::Stale(_) => writeln!(self.writer, "RESULT stale"),
        };
        if emitted.is_err() {
            self.write_failed.set(true);
        }
        Some(result)
    }
    fn cancel(&mut self, request: RequestId) -> std::result::Result<(), ContractError> {
        self.runtime.cancel(request)
    }
}

fn fixtures() -> Vec<(String, PositionSpec)> {
    let trace: Vec<_> = TRACE.split_whitespace().map(str::to_owned).collect();
    assert_eq!(trace.len(), 256);
    let mut fixtures: Vec<_> = [0, 16, 64, 128, 256]
        .into_iter()
        .map(|ply| {
            (
                format!("trace-{ply}"),
                PositionSpec {
                    base: PositionBase::StartPos,
                    moves: trace[..ply].to_vec(),
                },
            )
        })
        .collect();
    for (name, fen) in [
        (
            "unknown",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "kiwipete",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ),
        ("castling-claim", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 99 1"),
        ("ep", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"),
        ("ep-pin", "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"),
        ("promotion-claim", "1r2k3/P7/8/8/8/8/7p/R3K3 w Q - 99 1"),
        ("mate-75", "7k/6Q1/6K1/8/8/8/8/8 b - - 150 76"),
        (
            "counter-overflow",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 4294967295 4294967295",
        ),
    ] {
        fixtures.push((
            name.into(),
            PositionSpec {
                base: PositionBase::Fen(fen.into()),
                moves: vec![],
            },
        ));
    }
    for ply in [7, 8, 16] {
        fixtures.push((
            format!("repeat-{ply}"),
            PositionSpec {
                base: PositionBase::StartPos,
                moves: (0..ply)
                    .map(|i| ["g1f3", "g8f6", "f3g1", "f6g8"][i % 4].to_owned())
                    .collect(),
            },
        ));
    }
    fixtures
}

fn script(visits: usize, scenario: Scenario) -> Result<ScriptedBackend<MockTicket>> {
    let mut steps = Vec::new();
    for sequence in 0..=visits {
        let mut logits: Vec<f32> = (0..rz_encoding::POLICY_SIZE)
            .map(|i| (((i * 17 + sequence * 7) % 29) as f32 - 14.0) / 16.0)
            .collect();
        if scenario == Scenario::BadOutput {
            logits[0] = f32::NAN;
        }
        let raw = RawOutput {
            policy_logits: logits,
            wdl: match sequence % 3 {
                0 => vec![0.5, 0.25, 0.25],
                1 => vec![0.25, 0.5, 0.25],
                _ => vec![0.25, 0.25, 0.5],
            },
        };
        steps.push(Step {
            callbacks: vec![Callback {
                after: Duration::from_nanos(1),
                reply: Ok(raw),
            }],
            device_complete_after: Duration::from_nanos(2),
            cancel_ack_after: Some(Duration::ZERO),
        });
    }
    Ok(ScriptedBackend::new(
        ScriptIdentity {
            name: "rz-opt00-synthetic-heads/1".into(),
            seed: 71,
        },
        steps,
        MockLimits {
            max_steps: visits + 1,
            max_in_flight: 1,
            max_pending_events: 4,
            max_script_values: (visits + 1) * (rz_encoding::POLICY_SIZE + 3),
        },
    )?)
}

fn failure_text(failure: &ContractSearchFailure) -> String {
    match failure {
        ContractSearchFailure::Boundary(error) => format!("boundary {error}"),
        ContractSearchFailure::Tree(error) => format!("tree {error:?}"),
        ContractSearchFailure::Evaluation(failure) => {
            format!("evaluation {} {:?}", failure.error, failure.recovery)
        }
        ContractSearchFailure::Cleanup {
            primary,
            runtime,
            tree,
        } => format!("cleanup {} {runtime:?} {tree:?}", failure_text(primary)),
    }
}
fn status_text(status: &ContractSearchStatus) -> String {
    match status {
        ContractSearchStatus::Failed(failure) => failure_text(failure),
        other => format!("{other:?}"),
    }
}

fn run_case(
    writer: &mut impl Write,
    name: &str,
    spec: &PositionSpec,
    visits: usize,
    fill: HistoryFill,
    scenario: Scenario,
) -> Result<Option<ContractSearchOutcome>> {
    writeln!(
        writer,
        "CASE {name} fill={fill:?} scenario={scenario:?} visits={visits}"
    )?;
    let port = RulesUciPort::new(
        Arc::new(OwnerRegistry::default()),
        PositionLimits::default(),
    );
    let prepared = match port.prepare(spec) {
        Ok(prepared) => prepared,
        Err(error) if name == "counter-overflow" && error.code == ErrorCode::ResourceExhausted => {
            writeln!(writer, "PREPARE-ERROR {error}")?;
            return Ok(None);
        }
        Err(error) => return Err(error.into()),
    };
    let leaf = Leaf {
        position: prepared.snapshot,
        path: vec![],
    };
    emit_leaf(writer, "root", &leaf)?;
    let last = Rc::new(RefCell::new(leaf.clone()));
    let root = TrackedPosition {
        leaf,
        last: Rc::clone(&last),
    };
    let clock = ManualClock::default();
    let script = script(visits, scenario)?;
    let backend_id = Digest(script.identity_digest());
    let binding = MaiaBinding::new(
        ModelHandle {
            owner: OwnerId(900),
            slot: 0,
            generation: SlotGeneration(1),
            manifest: Digest(rz_eval::asset::sha256(
                b"rz-opt00-synthetic-heads/1;not-neural",
            )),
        },
        EncodingHandle {
            owner: OwnerId(901),
            slot: 0,
            generation: SlotGeneration(1),
            manifest: encoding_manifest(fill),
        },
        fill,
        backend_id,
        1,
    )?;
    let projection = ClassicalProjection::new(binding);
    let scope = AcceptanceScope {
        game: GameGeneration(1),
        root: RootGeneration(1),
        model: projection.model().handle(),
        encoding: projection.model().encoding().handle,
        backend: backend_id,
    };
    let shared = SharedScope::new(scope);
    let backend = ScriptedRuntimeBackend::new(
        script,
        clock.clone(),
        projection.clone(),
        Arc::clone(projection.model()),
        backend_id,
    )?;
    let controller = backend.controller();
    let adapter = ContractsAdapter::new(shared.clone(), clock.clone(), 1)?;
    let runtime = ContractEvaluator::new(
        adapter,
        backend,
        Limits {
            max_requests: 1,
            max_batch_items: 1,
            max_executions: 1,
            max_batch_wait: Duration::ZERO,
            max_queue_age: Duration::from_secs(1),
            deadline_reserve: Duration::ZERO,
            memory: Resources {
                host_bytes: HOST_BYTES_PER_ITEM * 2,
                device_bytes: 0,
                pinned_bytes: 0,
            },
        },
        256,
    )?;
    let write_failed = Rc::new(Cell::new(false));
    let mut runtime = WitnessRuntime {
        runtime,
        projection: projection.clone(),
        last: Rc::clone(&last),
        writer,
        write_failed: Rc::clone(&write_failed),
        delivered: None,
    };
    let deadline = Deadline {
        clock: ClockDomain(EPOCH),
        at: MonotonicTick(HARD_TICK),
    };
    let cancel = CancelToken::new();
    let tree_limits = TreeLimits {
        max_nodes: if scenario == Scenario::NodeLimit {
            1
        } else {
            1024
        },
        max_edges: 16384,
        max_depth: 32,
        max_legal_moves: 512,
        probability_tolerance: 1e-9,
    };
    let mut search = ContractSearch::new(
        root.clone(),
        ContractSearchConfig {
            scope,
            model: Arc::clone(projection.model()),
            precision: PrecisionProfile::Fp32,
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            bytes: ByteBudget {
                host: HOST_BYTES_PER_ITEM,
                device: 0,
                pinned: 0,
            },
            policy_tolerance: 1e-5,
            wdl_tolerance: 1e-5,
            cancellation: cancel.clone(),
            deadlines: ContractDeadlines {
                soft: deadline,
                admission: deadline,
                hard: deadline,
                output: deadline,
            },
            ids: Arc::new(IdAllocator::new(EPOCH)),
            max_simulations: visits as u64,
            tree_limits,
        },
    )?;
    let started = Instant::now();
    for _ in 0..(visits * 16 + 64) {
        if search.is_finished() {
            break;
        }
        if started.elapsed() > Duration::from_secs(30) || write_failed.get() {
            return Err("witness wall/output bound exceeded".into());
        }
        last.replace(root.leaf.clone());
        let event = search.pump(
            &mut runtime,
            &clock,
            || shared.get(),
            |position, encoding| {
                if encoding != projection.model().encoding() {
                    return Err(ContractError::new(
                        ErrorCode::IdentityMismatch,
                        Stage::Admission,
                        "encoder changed",
                    ));
                }
                if scenario == Scenario::BadInput {
                    return Ok(EvalInputKey(Digest([0; 32])));
                }
                projection.input_key(position.snapshot().state(), position.legal().moves())
            },
        );
        match event {
            ContractPumpEvent::Submitted { request, selection } => {
                writeln!(
                    runtime.writer,
                    "SUBMITTED request={} selection={}",
                    request.sequence, selection.sequence
                )?;
                match scenario {
                    Scenario::Cancel => cancel.cancel(),
                    Scenario::Expired => clock.0.store(HARD_TICK, Ordering::SeqCst),
                    Scenario::Stale => shared.update(AcceptanceScope {
                        root: RootGeneration(2),
                        ..scope
                    }),
                    _ => {}
                }
            }
            ContractPumpEvent::Accepted {
                request,
                selection,
                traversed_edges,
                evaluation,
            } => {
                let value = if let Some(request) = request {
                    let (delivered, value) = runtime
                        .delivered
                        .take()
                        .ok_or("accepted without delivered output")?;
                    if request != delivered {
                        return Err("accepted request mismatch".into());
                    }
                    value
                } else {
                    let leaf = last.borrow();
                    emit_leaf(
                        runtime.writer,
                        &format!("terminal-{}", selection.sequence),
                        &leaf,
                    )?;
                    let wdl = leaf
                        .position
                        .state()
                        .terminal_wdl()
                        .ok_or("accepted terminal lacks Rules WDL")?
                        .probabilities();
                    f64::from(wdl[0]) - f64::from(wdl[2])
                };
                writeln!(
                    runtime.writer,
                    "ACCEPTED selection={} request={:?} depth={traversed_edges} leaf-value-bits={} evaluation={}",
                    selection.sequence,
                    request.map(|r| r.sequence),
                    value.to_bits(),
                    evaluation.is_some()
                )?;
            }
            ContractPumpEvent::Waiting => {}
            ContractPumpEvent::Finished => {}
            other => return Err(format!("unexpected pump event {other:?}").into()),
        }
        clock.advance();
    }
    if !search.is_finished() {
        return Err("witness pump bound exceeded".into());
    }
    let outcome = search.outcome();
    runtime.runtime.begin_shutdown(Deadline {
        at: MonotonicTick(Clock::now(&clock).0 + 1000),
        ..deadline
    })?;
    for _ in 0..64 {
        controller.pump()?;
        runtime.runtime.pump();
        if runtime.runtime.poll().is_none() && runtime.runtime.drain_state() == DrainState::Drained
        {
            break;
        }
        clock.advance();
    }
    let snapshot = runtime.runtime.shutdown_snapshot();
    let script_status = controller.script_status()?;
    if snapshot.drain != DrainState::Drained
        || snapshot.state.reserved != Resources::default()
        || snapshot.state.logical_requests != 0
        || snapshot.state.reserved_requests != 0
        || snapshot.state.executions != 0
        || script_status.physical_leases != 0
    {
        return Err(format!("witness did not drain: {snapshot:?} {script_status:?}").into());
    }
    writeln!(
        runtime.writer,
        "FINAL status={} best={:?} fallback={} counters={:?} metrics={:?} policy={:?}",
        status_text(&outcome.status),
        outcome.best_move,
        outcome.fallback_used,
        outcome.counters,
        outcome.metrics,
        outcome.policy_identity
    )?;
    for (movement, stats) in &outcome.root_stats {
        writeln!(
            runtime.writer,
            "ROOT {movement:?} prior-bits={} visits={} value-sum-bits={}",
            stats.prior.to_bits(),
            stats.visits,
            stats.value_sum.to_bits()
        )?;
    }
    writeln!(runtime.writer, "DRAIN {snapshot:?} {script_status:?}")?;
    if write_failed.get() {
        return Err("witness write failed".into());
    }
    Ok(Some(outcome))
}

fn options(args: Vec<String>) -> Result<(usize, String, HistoryFill)> {
    let mut visits = 16;
    let mut fixture = "all".to_owned();
    let mut fill = HistoryFill::No;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or("expected --visits N / --fixture NAME / --history-fill no|always")?;
        match arg.as_str() {
            "--visits" => visits = value.parse()?,
            "--fixture" => fixture = value,
            "--history-fill" => {
                fill = match value.as_str() {
                    "no" => HistoryFill::No,
                    "always" => HistoryFill::RepeatOldest,
                    _ => return Err("unsupported history fill".into()),
                }
            }
            _ => return Err("unsupported witness argument".into()),
        }
    }
    if !(1..=64).contains(&visits)
        || (fixture != "all" && !fixtures().iter().any(|(name, _)| *name == fixture))
    {
        return Err("visits must be 1..64 and fixture must be known".into());
    }
    Ok((visits, fixture, fill))
}

fn run() -> Result<()> {
    let (visits, filter, fill) = options(std::env::args().skip(1).collect())?;
    let mut writer = CappedWriter {
        inner: BufWriter::new(io::stdout().lock()),
        written: 0,
    };
    writeln!(
        writer,
        "rz-opt00-witness/1;clock=manual;heads=synthetic;not-a-performance-sample;fp32;B1"
    )?;
    for (name, spec) in fixtures() {
        if filter == "all" || filter == name {
            run_case(&mut writer, &name, &spec, visits, fill, Scenario::Normal)?;
        }
    }
    if filter == "all" {
        let spec = PositionSpec {
            base: PositionBase::StartPos,
            moves: vec![],
        };
        for scenario in Scenario::GUARDS {
            run_case(&mut writer, "guard-startpos", &spec, 4, fill, scenario)?;
        }
    }
    writer.flush()?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("optimization witness: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn start() -> PositionSpec {
        PositionSpec {
            base: PositionBase::StartPos,
            moves: vec![],
        }
    }
    #[test]
    fn real_rules_encoder_runtime_and_search_commit_and_drain() {
        let mut bytes = Vec::new();
        let outcome = run_case(
            &mut bytes,
            "start",
            &start(),
            4,
            HistoryFill::No,
            Scenario::Normal,
        )
        .unwrap()
        .unwrap();
        assert!(matches!(outcome.status, ContractSearchStatus::Completed));
        assert_eq!(outcome.counters.root_initializations, 1);
        assert_eq!(outcome.counters.completed_visits, 4);
        assert_eq!(outcome.counters.accepted_backups, 4);
        assert_eq!(
            outcome
                .root_stats
                .iter()
                .map(|(_, stats)| stats.visits)
                .sum::<u64>(),
            4
        );
        let witness = String::from_utf8(bytes).unwrap();
        assert_eq!(
            witness
                .lines()
                .filter(|line| line.starts_with("ACCEPTED "))
                .count(),
            5
        );
        for tensor in witness
            .lines()
            .filter_map(|line| line.strip_prefix("TENSOR-LE "))
        {
            assert_eq!(tensor.split_once(' ').unwrap().1.len(), 112 * 8 * 8 * 4 * 2);
        }
    }
    #[test]
    fn guard_failures_release_reservations_and_cannot_initialize_root() {
        for scenario in Scenario::GUARDS {
            let outcome = run_case(
                &mut io::sink(),
                "guard",
                &start(),
                4,
                HistoryFill::No,
                scenario,
            )
            .unwrap()
            .unwrap();
            assert_eq!(outcome.counters.completed_visits, 0, "{scenario:?}");
            assert_eq!(outcome.counters.accepted_backups, 0, "{scenario:?}");
            assert_eq!(
                outcome.counters.root_initializations,
                u64::from(scenario == Scenario::NodeLimit)
            );
            assert!(!matches!(
                outcome.status,
                ContractSearchStatus::Running | ContractSearchStatus::Completed
            ));
        }
    }
    #[test]
    fn witness_and_cli_have_finite_limits() {
        assert!(options(vec!["--visits".into(), "65".into()]).is_err());
        assert!(options(vec!["--fixture".into(), "unknown-name".into()]).is_err());
        assert!(options(vec!["--visits".into()]).is_err());
        let mut writer = CappedWriter {
            inner: io::sink(),
            written: MAX_OUTPUT,
        };
        assert!(writer.write_all(b"x").is_err());
    }
}
