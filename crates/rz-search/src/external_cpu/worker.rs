//! Single native ELF owner behind the shared bounded mailbox. The coordinator
//! submits immutable jobs and never drains external pipes synchronously.

use super::*;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::{
    CpuJobId, CpuWorkerId, ExternalCpuJobEnvelope, ExternalSliceGrant,
};
use rz_contracts::{ClockDomain, Digest, ExecutionId, MonotonicTick};
use rz_runtime::CompletionSignal;
use rz_runtime::adaptive::{ControlMailbox, ControlSnapshot, MailboxError, WorkerMailbox};
use sha2::Digest as _;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalWorkerScope {
    pub job: CpuJobId,
    pub execution: ExecutionId,
    pub owner: CpuWorkerId,
}

#[derive(Clone, Debug)]
pub struct ExternalJob {
    pub envelope: ExternalCpuJobEnvelope,
    pub grant: ExternalSliceGrant,
    pub execution: ExecutionId,
    pub owner: CpuWorkerId,
    pub position: Position,
    pub root_moves: Option<Vec<BoardMove>>,
    pub max_depth: u16,
    pub policy: ExternalSlicePolicy,
    pub control: ControlMailbox<ExternalWorkerScope>,
}

impl ExternalJob {
    pub fn scope(&self) -> ExternalWorkerScope {
        ExternalWorkerScope {
            job: self.envelope.id,
            execution: self.execution,
            owner: self.owner,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExternalWorkerProgress {
    pub scope: ExternalWorkerScope,
    pub opinion: ExternalPolicyOpinion,
    pub reported_nodes: Option<u64>,
    pub strict_native_nodes: Option<u64>,
    pub control_sequence: u64,
    /// A foreign depth/PV tuple cannot certify native completed iteration.
    pub completed_iteration_observed: Option<bool>,
}

/// A coherent foreign info tuple and the actual private owner's clocks. The
/// tuple remains raw selective-engine feedback, not a completed native depth.
#[derive(Clone, Debug)]
pub struct ExternalWorkerProgressObservation {
    original: ExternalWorkerProgress,
    published_at: Instant,
    go_issued_at: Option<Instant>,
    quantum_deadline: Option<Instant>,
    stop_sent_at: Option<Instant>,
    coherent_tuple_advanced: bool,
    coherent_tuple_interval: Option<Duration>,
}

impl ExternalWorkerProgressObservation {
    pub fn progress(&self) -> &ExternalWorkerProgress {
        &self.original
    }
    pub fn published_at(&self) -> Instant {
        self.published_at
    }
    pub fn go_issued_at(&self) -> Option<Instant> {
        self.go_issued_at
    }
    pub fn quantum_deadline(&self) -> Option<Instant> {
        self.quantum_deadline
    }
    pub fn stop_sent_at(&self) -> Option<Instant> {
        self.stop_sent_at
    }
    pub fn is_new_coherent_tuple(&self) -> bool {
        self.coherent_tuple_advanced
    }
    /// Requires two distinct legal tuples from this same actual go. The first
    /// tuple and repeated identical tuples never create an interval sample.
    pub fn completed_coherent_tuple_interval(&self) -> Option<Duration> {
        self.coherent_tuple_interval
    }
}

/// Owner-observed protocol timing, separate from public receipt/codec fields.
/// A ready barrier does not certify applied options or completed minimax work.
#[derive(Clone, Copy, Debug)]
pub struct ExternalWorkerTiming {
    admitted_at: Instant,
    run_started_at: Instant,
    go_issued_at: Option<Instant>,
    quantum_deadline: Option<Instant>,
    stop_sent_at: Option<Instant>,
    bestmove_drained_at: Option<Instant>,
    ready_completed_at: Option<Instant>,
    analyze_returned_at: Instant,
    report_ready_at: Instant,
}

impl ExternalWorkerTiming {
    pub fn admitted_at(&self) -> Instant {
        self.admitted_at
    }
    pub fn run_started_at(&self) -> Instant {
        self.run_started_at
    }
    pub fn go_issued_at(&self) -> Option<Instant> {
        self.go_issued_at
    }
    pub fn quantum_deadline(&self) -> Option<Instant> {
        self.quantum_deadline
    }
    pub fn stop_sent_at(&self) -> Option<Instant> {
        self.stop_sent_at
    }
    pub fn bestmove_drained_at(&self) -> Option<Instant> {
        self.bestmove_drained_at
    }
    pub fn ready_completed_at(&self) -> Option<Instant> {
        self.ready_completed_at
    }
    pub fn analyze_returned_at(&self) -> Instant {
        self.analyze_returned_at
    }
    pub fn report_ready_at(&self) -> Instant {
        self.report_ready_at
    }
    pub fn queue_wait(&self) -> Duration {
        self.run_started_at.duration_since(self.admitted_at)
    }
    pub fn stop_drain(&self) -> Option<Duration> {
        self.ready_completed_at?
            .checked_duration_since(self.stop_sent_at?)
    }
    pub fn return_processing(&self) -> Duration {
        self.report_ready_at
            .duration_since(self.analyze_returned_at)
    }
}

#[derive(Clone, Debug)]
pub struct ExternalWorkerReport {
    pub job: ExternalCpuJobEnvelope,
    pub grant: ExternalSliceGrant,
    pub scope: ExternalWorkerScope,
    pub result: Result<ExternalCooperativeReport, CheckerError>,
    pub attempt: Option<CheckerAttempt>,
    pub controls: ControlSnapshot,
    /// No external exact-stack token is emitted on Yield.
    pub exact_resume_supported: bool,
    pub go_issue: ExternalGoIssue,
}

#[derive(Clone, Debug)]
pub struct ExternalWorkerStartup {
    pub result: Result<ExternalUciIdentity, CheckerError>,
    pub attempt: Option<CheckerAttempt>,
}

/// Evidence issued by a live native owner only after its successful startup
/// and this exact mailbox admission. It certifies submission, not `go`, a
/// complete teacher opinion, an applied option readback or a Rules outcome.
/// Public Job/Report fixtures cannot construct this private-field token.
#[derive(Clone, Debug)]
pub struct ExternalTeacherDispatch {
    job: ExternalCpuJobEnvelope,
    grant: ExternalSliceGrant,
    scope: ExternalWorkerScope,
    process: ExternalProcessIdentity,
    observed_uci: ExternalUciIdentity,
    evaluation_namespace: Digest,
}

impl ExternalTeacherDispatch {
    pub fn job(&self) -> ExternalCpuJobEnvelope {
        self.job
    }
    pub fn grant(&self) -> ExternalSliceGrant {
        self.grant
    }
    pub fn scope(&self) -> ExternalWorkerScope {
        self.scope
    }
    pub fn process_identity(&self) -> ExternalProcessIdentity {
        self.process
    }
    pub fn observed_uci(&self) -> &ExternalUciIdentity {
        &self.observed_uci
    }
    pub fn evaluation_namespace(&self) -> Digest {
        self.evaluation_namespace
    }
    /// Comparison preserves original question and grant. It never verifies a
    /// caller report or permits a replacement go to reuse this authority.
    pub fn matches_job(&self, job: &ExternalJob) -> bool {
        self.job == job.envelope && self.grant == job.grant && self.scope == job.scope()
    }
}

/// Immutable terminal packet minted inside the private native worker. A
/// success flag on a public `ExternalWorkerReport` is deliberately insufficient
/// to issue one. Failed, canceled and pre-go jobs retain a terminal packet too;
/// the packet's actual report must still pass the target's completion policy.
#[derive(Clone, Debug)]
pub struct ExternalWorkerCompletion {
    report: ExternalWorkerReport,
    process: Option<ExternalProcessIdentity>,
    evaluation_namespace: Digest,
    timing: Option<ExternalWorkerTiming>,
}

impl ExternalWorkerCompletion {
    pub fn report(&self) -> &ExternalWorkerReport {
        &self.report
    }
    pub fn process_identity(&self) -> Option<ExternalProcessIdentity> {
        self.process
    }
    pub fn evaluation_namespace(&self) -> Digest {
        self.evaluation_namespace
    }
    pub fn timing(&self) -> Option<ExternalWorkerTiming> {
        self.timing
    }
    /// Submission/failure join only. A failed report has no successful UCI
    /// tuple to compare and does not acquire readiness or target authority.
    /// Missing physical owner identity stays unjoined/unknown.
    pub fn matches_submission_scope(&self, submitted: &ExternalTeacherDispatch) -> bool {
        self.report.job == submitted.job
            && self.report.grant == submitted.grant
            && self.report.scope == submitted.scope
            && self.process == Some(submitted.process)
            && self.evaluation_namespace == submitted.evaluation_namespace
    }
    pub fn matches_dispatch(&self, submitted: &ExternalTeacherDispatch) -> bool {
        self.matches_submission_scope(submitted)
            && self.report.result.as_ref().is_ok_and(|actual| {
                actual.report.observed_uci == submitted.observed_uci
                    && external_evaluation_namespace(&actual.report.identity)
                        .is_ok_and(|namespace| namespace == submitted.evaluation_namespace)
            })
    }
    pub fn into_report(self) -> ExternalWorkerReport {
        self.report
    }
}

struct TimedExternalJob {
    original: ExternalJob,
    admitted_at: Instant,
    observed_progress: bool,
}

type Mailbox =
    WorkerMailbox<TimedExternalJob, ExternalWorkerCompletion, ExternalWorkerProgressObservation>;

/// Issued only after the private owner observed a new ready barrier and all
/// accepted old-root terminal permits were consumed. A live owner needs no
/// stdout/stderr EOF and this is never a quit/join receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalRootFence {
    authority: SearchAuthority,
    owner: CpuWorkerId,
    clock: ClockDomain,
    sequence: u64,
    execution_high_water: u64,
    job_high_water: u64,
    control_high_water: u64,
    progress_high_water: u64,
    native_request_high_water: u64,
    game_reset_high_water: u64,
    source: Digest,
    process: CheckerShutdown,
    readiness_observed: bool,
}

impl ExternalRootFence {
    pub fn authority(self) -> SearchAuthority {
        self.authority
    }
    pub fn owner(self) -> CpuWorkerId {
        self.owner
    }
    pub fn process_identity(self) -> Option<ExternalProcessIdentity> {
        self.process.process_identity
    }
    pub fn execution_high_water(self) -> u64 {
        self.execution_high_water
    }
    pub fn job_high_water(self) -> u64 {
        self.job_high_water
    }
    pub fn control_high_water(self) -> u64 {
        self.control_high_water
    }
    pub fn progress_high_water(self) -> u64 {
        self.progress_high_water
    }
    pub fn native_request_high_water(self) -> u64 {
        self.native_request_high_water
    }
    pub fn game_reset_high_water(self) -> u64 {
        self.game_reset_high_water
    }
    pub fn source(self) -> Digest {
        self.source
    }
    pub fn permits_reuse(self) -> bool {
        self.readiness_observed
            && self.process.process_identity.is_some()
            && !self.process.quit_sent
            && !self.process.exit_observed
            && !self.process.cleanup_complete
            && !self.process.quarantined
            && !self.process.ownership_lost
    }
}

/// Actual game reset by the same retained owner, never a caller metadata arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalNewGameFence {
    previous: ExternalRootFence,
    next: SearchAuthority,
    sequence: u64,
    game_reset_high_water: u64,
    process: CheckerShutdown,
    readiness_observed: bool,
}
impl ExternalNewGameFence {
    pub fn previous_authority(self) -> SearchAuthority {
        self.previous.authority
    }
    pub fn authority(self) -> SearchAuthority {
        self.next
    }
    pub fn owner(self) -> CpuWorkerId {
        self.previous.owner
    }
    pub fn source(self) -> Digest {
        self.previous.source
    }
    pub fn process_identity(self) -> Option<ExternalProcessIdentity> {
        self.process.process_identity
    }
    pub fn game_reset_high_water(self) -> u64 {
        self.game_reset_high_water
    }
    pub fn permits_reuse(self) -> bool {
        self.readiness_observed
            && self.previous.permits_reuse()
            && self.process.process_identity == self.previous.process.process_identity
            && !self.process.quit_sent
            && !self.process.exit_observed
            && !self.process.cleanup_complete
            && !self.process.quarantined
            && !self.process.ownership_lost
    }
}

#[derive(Clone, Copy)]
struct RootRequest {
    authority: SearchAuthority,
    sequence: u64,
    until: Instant,
}

#[derive(Clone, Copy)]
struct GameRequest {
    previous: ExternalRootFence,
    next: SearchAuthority,
    sequence: u64,
    until: Instant,
}

#[derive(Default)]
struct RootLifecycle {
    authority: Option<SearchAuthority>,
    request: Option<RootRequest>,
    result: Option<Result<ExternalRootFence, CheckerError>>,
    game_request: Option<GameRequest>,
    game_result: Option<Result<ExternalNewGameFence, CheckerError>>,
    game_sequence: u64,
    fence_sequence: u64,
    execution_high_water: u64,
    last_job: Option<ExternalCpuJobEnvelope>,
    control_high_water: u64,
}

fn physical_fence(
    shutdown_deadline: &Mutex<Option<Instant>>,
    roots: &Mutex<RootLifecycle>,
) -> Option<Instant> {
    let shutdown = *shutdown_deadline.lock().unwrap_or_else(|p| p.into_inner());
    let root = roots
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .request
        .map(|r| r.until);
    match (shutdown, root) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn game_physical_fence(
    shutdown_deadline: &Mutex<Option<Instant>>,
    roots: &Mutex<RootLifecycle>,
) -> Option<Instant> {
    let shutdown = *shutdown_deadline.lock().unwrap_or_else(|p| p.into_inner());
    // A completed root's historical deadline is not the new game's wall.
    let game = roots
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .game_request
        .map(|r| r.until);
    match (shutdown, game) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// One explicit same-process monotonic domain. It is supplied by the real
/// coordinator owner and is never reconstructed from UTC or another process.
#[derive(Clone, Copy, Debug)]
pub struct ExternalWorkerClock {
    pub domain: ClockDomain,
    pub origin: Instant,
}

impl ExternalWorkerClock {
    fn now(self) -> Result<MonotonicTick, CheckerError> {
        let elapsed =
            Instant::now()
                .checked_duration_since(self.origin)
                .ok_or(CheckerError::Invalid(
                    "external clock origin is in the future",
                ))?;
        Ok(MonotonicTick(u64::try_from(elapsed.as_nanos()).map_err(
            |_| CheckerError::Invalid("external clock tick overflow"),
        )?))
    }

    fn deadline(self, job: &ExternalCpuJobEnvelope) -> Result<Instant, CheckerError> {
        job.validate(self.domain, self.now()?)
            .map_err(|_| CheckerError::Invalid("external job clock/authority/deadline"))?;
        self.origin
            .checked_add(Duration::from_nanos(job.hard_deadline.at.0))
            .ok_or(CheckerError::Invalid(
                "external job Instant deadline overflow",
            ))
    }
}

pub fn external_root_restriction_digest(roots: Option<&[BoardMove]>) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-external-root-restriction/1\0");
    hash.update([u8::from(roots.is_some())]);
    if let Some(roots) = roots {
        hash.update((roots.len() as u64).to_le_bytes());
        for mv in roots {
            let text = mv.to_string();
            hash.update((text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        }
    }
    Digest(hash.finalize().into())
}

pub fn external_evaluation_namespace(
    identity: &ExternalCheckerIdentity,
) -> Result<Digest, CheckerError> {
    identity.validate()?;
    let bytes = serde_json::to_vec(identity)
        .map_err(|_| CheckerError::Invalid("external identity serialization"))?;
    let mut hash = Sha256::new();
    hash.update(b"rz-external-evaluation-namespace/1\0");
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
    Ok(Digest(hash.finalize().into()))
}

pub fn external_search_conditions(
    config: &ExternalCpuConfig,
    max_depth: u16,
    policy: ExternalSlicePolicy,
) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-external-cooperative-search/1\0");
    hash.update(EXTERNAL_UCI_CHECKER_VERSION.as_bytes());
    hash.update(EXTERNAL_COOPERATIVE_CHECKER_VERSION.as_bytes());
    hash.update(max_depth.to_le_bytes());
    hash.update((config.max_prefix_plies as u64).to_le_bytes());
    hash.update(config.max_depth.to_le_bytes());
    for duration in [
        config.handshake_timeout,
        config.max_task_wall_time,
        config.stop_grace,
        config.shutdown_grace,
    ] {
        hash.update(duration.as_nanos().to_le_bytes());
    }
    hash.update((config.max_output_bytes as u64).to_le_bytes());
    hash.update((config.max_line_bytes as u64).to_le_bytes());
    // Quantum and this invocation's reported-node hint live in SliceGrant.
    // Changing them cannot silently replace an immutable question namespace.
    hash.update(policy.stop_drain_reserve.as_nanos().to_le_bytes());
    hash.update(policy.multipv.to_le_bytes());
    Digest(hash.finalize().into())
}

/// State/question evidence is deliberately separate from native node credit.
/// The producer calls this after Rules exported the exact position/history.
pub fn external_question_digest(
    state: rz_contracts::StateIdentity,
    roots: Digest,
    evaluation: Digest,
    conditions: Digest,
) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-external-job-question/1\0");
    hash.update(state.owner.0.to_le_bytes());
    hash.update(state.revision.0.to_le_bytes());
    for digest in [state.semantic, roots, evaluation, conditions] {
        hash.update(digest.0);
    }
    Digest(hash.finalize().into())
}

fn validate_job(
    job: &ExternalJob,
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    config: &ExternalCpuConfig,
) -> Result<Instant, CheckerError> {
    let deadline = clock.deadline(&job.envelope)?;
    job.grant
        .validate(&job.envelope, clock.domain, clock.now()?)
        .map_err(|_| CheckerError::Invalid("external slice grant/deadline mismatch"))?;
    if job.owner != owner
        || job.execution.epoch != job.envelope.id.authority.epoch
        || job.execution.sequence == 0
        || *job.control.scope() != job.scope()
        || job.control.snapshot().terminal
        || job.grant.execution != job.execution
        || job.grant.owner != job.owner
        || job.grant.soft_quantum != job.policy.quantum
        || job.grant.requested_reported_node_hint != job.policy.requested_reported_nodes
        || job.max_depth == 0
        || job.max_depth > config.max_depth
    {
        return Err(CheckerError::Invalid(
            "external job owner/execution/control/depth seal",
        ));
    }
    // Rules, not a caller digest, attests the complete board/history payload.
    let exported = rz_position::contracts::ContractPosition::new(
        job.envelope.state_identity.owner,
        job.position.clone(),
    )
    .export()
    .map_err(|_| CheckerError::Invalid("external Rules state export"))?;
    if exported.snapshot().identity() != job.envelope.state_identity {
        return Err(CheckerError::Invalid(
            "external job state/history seal mismatch",
        ));
    }
    let roots = external_root_restriction_digest(job.root_moves.as_deref());
    let evaluation = external_evaluation_namespace(&config.identity)?;
    let conditions = external_search_conditions(config, job.max_depth, job.policy);
    if roots != job.envelope.root_restriction
        || evaluation != job.envelope.evaluation_namespace
        || conditions != job.envelope.search_conditions
        || external_question_digest(job.envelope.state_identity, roots, evaluation, conditions)
            != job.envelope.question
    {
        return Err(CheckerError::Invalid(
            "external job immutable question namespace mismatch",
        ));
    }
    let legal = job.position.ordered_legal_moves();
    if let Some(roots) = &job.root_moves {
        if roots.is_empty()
            || roots.len() > 256
            || roots
                .iter()
                .enumerate()
                .any(|(at, mv)| !legal.moves().contains(mv) || roots[..at].contains(mv))
        {
            return Err(CheckerError::Invalid("external job legal root mask"));
        }
    }
    job.policy.validate(
        job.root_moves
            .as_ref()
            .map_or(legal.moves().len(), Vec::len),
    )?;
    if config.stop_grace != job.policy.stop_drain_reserve
        || deadline.saturating_duration_since(Instant::now())
            < job.policy.quantum + job.policy.stop_drain_reserve
    {
        return Err(CheckerError::Invalid(
            "external job quantum/reserve admission",
        ));
    }
    Ok(deadline)
}

fn validate_submission(
    job: &ExternalJob,
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    config: &ExternalCpuConfig,
) -> Result<(), CheckerError> {
    let deadline = clock.deadline(&job.envelope)?;
    job.grant
        .validate(&job.envelope, clock.domain, clock.now()?)
        .map_err(|_| CheckerError::Invalid("external submitted slice grant mismatch"))?;
    if job.owner != owner
        || job.execution.epoch != job.envelope.id.authority.epoch
        || job.execution.sequence == 0
        || *job.control.scope() != job.scope()
        || job.control.snapshot().terminal
        || job.grant.execution != job.execution
        || job.grant.owner != job.owner
        || job.grant.soft_quantum != job.policy.quantum
        || job.grant.requested_reported_node_hint != job.policy.requested_reported_nodes
        || job.max_depth == 0
        || job.max_depth > config.max_depth
        || job
            .root_moves
            .as_ref()
            .is_some_and(|roots| roots.is_empty() || roots.len() > 256 || roots.capacity() > 256)
    {
        return Err(CheckerError::Invalid(
            "external submitted job owner/scope/depth/root bound",
        ));
    }
    // The private worker verifies legal candidates and Rules/input digests.
    // Admission does not encode or regenerate the full Rules history.
    job.policy
        .validate(job.root_moves.as_ref().map_or(256, Vec::len))?;
    if config.stop_grace != job.policy.stop_drain_reserve
        || deadline.saturating_duration_since(Instant::now())
            < job.policy.quantum + job.policy.stop_drain_reserve
    {
        return Err(CheckerError::Invalid(
            "external submitted quantum/reserve bound",
        ));
    }
    Ok(())
}

struct ExternalOwnerObservations {
    timing: ExternalWorkerTiming,
    last_coherent_tuple: Option<ExternalPolicyOpinion>,
    last_coherent_published_at: Option<Instant>,
}

impl ExternalOwnerObservations {
    fn new(admitted_at: Instant, run_started_at: Instant) -> Self {
        Self {
            timing: ExternalWorkerTiming {
                admitted_at,
                run_started_at,
                go_issued_at: None,
                quantum_deadline: None,
                stop_sent_at: None,
                bestmove_drained_at: None,
                ready_completed_at: None,
                analyze_returned_at: run_started_at,
                report_ready_at: run_started_at,
            },
            last_coherent_tuple: None,
            last_coherent_published_at: None,
        }
    }
}

struct WorkerControl {
    scope: ExternalWorkerScope,
    controls: ControlMailbox<ExternalWorkerScope>,
    mailbox: Mailbox,
    progress_sequence: Arc<AtomicU64>,
    shutdown_deadline: Arc<Mutex<Option<Instant>>>,
    roots: Arc<Mutex<RootLifecycle>>,
    observations: Arc<Mutex<ExternalOwnerObservations>>,
    observed_progress: bool,
}

impl ExternalControl for WorkerControl {
    fn observe(&self) -> ExternalControlObservation {
        let snapshot = self.controls.snapshot();
        if snapshot.terminal {
            ExternalControlObservation {
                sequence: 0,
                action: ExternalControlAction::Revoked,
            }
        } else if snapshot.is_cancelled() {
            ExternalControlObservation {
                sequence: snapshot.cancel_sequence,
                action: ExternalControlAction::Cancel,
            }
        } else if self
            .roots
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .request
            .is_some()
        {
            ExternalControlObservation {
                sequence: 0,
                action: ExternalControlAction::Revoked,
            }
        } else if let Some(sequence) = snapshot.pending_yield() {
            ExternalControlObservation {
                sequence,
                action: ExternalControlAction::Yield,
            }
        } else {
            ExternalControlObservation {
                sequence: 0,
                action: ExternalControlAction::Continue,
            }
        }
    }

    fn physical_fence_deadline(&self) -> Option<Instant> {
        physical_fence(&self.shutdown_deadline, &self.roots)
    }

    fn observe_owner_phase(&self, phase: ExternalOwnerPhase) {
        let mut observations = self
            .observations
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let timing = &mut observations.timing;
        match phase {
            ExternalOwnerPhase::GoIssued {
                at,
                quantum_deadline,
            } => {
                if timing.go_issued_at.is_none() {
                    timing.go_issued_at = Some(at);
                    timing.quantum_deadline = Some(quantum_deadline);
                }
            }
            ExternalOwnerPhase::StopSent { at } => {
                timing.stop_sent_at.get_or_insert(at);
            }
            ExternalOwnerPhase::BestMoveDrained { at } => {
                timing.bestmove_drained_at.get_or_insert(at);
            }
            ExternalOwnerPhase::ReadyBarrierCompleted { at } => {
                timing.ready_completed_at.get_or_insert(at);
            }
        }
    }

    fn publish_progress(&self, opinion: &ExternalPolicyOpinion, reported_nodes: Option<u64>) {
        let pending_publish = self.controls.snapshot().pending_publish();
        if !self.observed_progress && pending_publish.is_none() {
            return;
        }
        let published_at = Instant::now();
        let (go_issued_at, quantum_deadline, stop_sent_at, advanced, interval) = {
            let mut observations = self
                .observations
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            // Equality uses the whole coherent source tuple, not an increased
            // reported-node count or a repeated public-control request.
            let advanced = observations.last_coherent_tuple.as_ref() != Some(opinion);
            let active_go = observations.timing.go_issued_at.is_some()
                && observations.timing.stop_sent_at.is_none();
            let interval = if advanced && active_go {
                observations
                    .last_coherent_published_at
                    .and_then(|previous| published_at.checked_duration_since(previous))
            } else {
                None
            };
            if advanced && active_go {
                observations.last_coherent_tuple = Some(opinion.clone());
                observations.last_coherent_published_at = Some(published_at);
            }
            (
                observations.timing.go_issued_at,
                observations.timing.quantum_deadline,
                observations.timing.stop_sent_at,
                advanced && active_go,
                interval,
            )
        };
        // Legacy explicit publication remains available, including its real
        // ack. Observed admission alone enables new coherent-tuple feedback.
        if pending_publish.is_none() && !advanced {
            return;
        }
        let sequence =
            self.progress_sequence
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |previous| {
                    previous.checked_add(1)
                });
        let Ok(previous) = sequence else {
            return;
        };
        let progress = ExternalWorkerProgress {
            scope: self.scope,
            opinion: opinion.clone(),
            reported_nodes,
            strict_native_nodes: None,
            control_sequence: pending_publish.unwrap_or(0),
            completed_iteration_observed: None,
        };
        if self
            .mailbox
            .publish_progress(
                previous + 1,
                ExternalWorkerProgressObservation {
                    original: progress,
                    published_at,
                    go_issued_at,
                    quantum_deadline,
                    stop_sent_at,
                    coherent_tuple_advanced: advanced,
                    coherent_tuple_interval: interval,
                },
            )
            .is_ok()
        {
            if let Some(sequence) = pending_publish {
                let _ = self.controls.acknowledge_publish_progress(sequence);
            }
        }
    }
}

pub struct ExternalCpuWorker {
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    config: ExternalCpuConfig,
    mailbox: Mailbox,
    shutdown: Arc<AtomicBool>,
    shutdown_deadline: Arc<Mutex<Option<Instant>>>,
    startup: Arc<Mutex<Option<ExternalWorkerStartup>>>,
    roots: Arc<Mutex<RootLifecycle>>,
    thread: Option<JoinHandle<Result<CheckerShutdown, CheckerError>>>,
    joined: Option<Result<CheckerShutdown, CheckerError>>,
    last_execution: u64,
}

impl ExternalCpuWorker {
    /// Creates one private Rust worker; native ELF start/handshake occurs there.
    /// The caller registers CPU quota/affinity separately; none is silently set.
    pub fn spawn(
        config: ExternalCpuConfig,
        owner: CpuWorkerId,
        clock: ExternalWorkerClock,
        startup_deadline: Instant,
    ) -> Result<Self, CheckerError> {
        Self::spawn_internal(config, owner, clock, startup_deadline, None)
    }

    /// V5 private snapshots reach the actual native start as held executable
    /// and cwd descriptors. The worker still verifies executable bytes/ELF and
    /// obtains its own process/readiness evidence; pins alone are not authority.
    pub fn spawn_pinned(
        config: ExternalCpuConfig,
        owner: CpuWorkerId,
        clock: ExternalWorkerClock,
        startup_deadline: Instant,
        launch_pin: ExternalLaunchPin,
    ) -> Result<Self, CheckerError> {
        Self::spawn_internal(config, owner, clock, startup_deadline, Some(launch_pin))
    }

    fn spawn_internal(
        config: ExternalCpuConfig,
        owner: CpuWorkerId,
        clock: ExternalWorkerClock,
        startup_deadline: Instant,
        launch_pin: Option<ExternalLaunchPin>,
    ) -> Result<Self, CheckerError> {
        config.validate()?;
        if cfg!(not(target_os = "linux")) {
            return Err(CheckerError::Unsupported(
                "external UCI worker requires Linux native ELF",
            ));
        }
        clock.now()?;
        let signal = CompletionSignal::default();
        let mailbox = Mailbox::new(8, 16, signal)
            .map_err(|_| CheckerError::Invalid("external worker mailbox capacity"))?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_deadline = Arc::new(Mutex::new(None));
        let startup = Arc::new(Mutex::new(None));
        let roots = Arc::new(Mutex::new(RootLifecycle::default()));
        let cloned_mailbox = mailbox.clone();
        let cloned_shutdown = shutdown.clone();
        let cloned_deadline = shutdown_deadline.clone();
        let cloned_config = config.clone();
        let cloned_startup = startup.clone();
        let cloned_roots = roots.clone();
        let thread = std::thread::Builder::new()
            .name(format!("rz-external-cpu-{}", owner.0))
            .spawn(move || {
                run_worker(
                    cloned_config,
                    owner,
                    clock,
                    startup_deadline,
                    cloned_mailbox,
                    cloned_shutdown,
                    cloned_deadline,
                    cloned_startup,
                    cloned_roots,
                    launch_pin,
                )
            })
            .map_err(|_| error("worker", "thread_spawn"))?;
        Ok(Self {
            owner,
            clock,
            config,
            mailbox,
            shutdown,
            shutdown_deadline,
            startup,
            roots,
            thread: Some(thread),
            joined: None,
            last_execution: 0,
        })
    }

    pub fn signal(&self) -> CompletionSignal {
        self.mailbox.signal()
    }
    pub fn startup_report(&self) -> Option<ExternalWorkerStartup> {
        self.startup
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }
    /// Only an actual successful native startup followed by successful bounded
    /// mailbox admission grants teacher submission authority. This performs no
    /// forward/pipe work on the manager and does not claim go was issued.
    pub fn submit_attested(
        &mut self,
        job: ExternalJob,
    ) -> Result<ExternalTeacherDispatch, CheckerError> {
        self.submit_attested_inner(job, false)
    }
    /// Opt in to immediate distinct coherent-tuple observations. Submission
    /// still issues only the original dispatch authority, never a go/target.
    pub fn submit_observed(
        &mut self,
        job: ExternalJob,
    ) -> Result<ExternalTeacherDispatch, CheckerError> {
        self.submit_attested_inner(job, true)
    }
    fn submit_attested_inner(
        &mut self,
        job: ExternalJob,
        observed_progress: bool,
    ) -> Result<ExternalTeacherDispatch, CheckerError> {
        let startup = self
            .startup_report()
            .ok_or_else(|| error("teacher_admission", "actual_startup_missing"))?;
        let observed_uci = startup
            .result
            .map_err(|_| error("teacher_admission", "actual_startup_failed"))?;
        let process_state = startup
            .attempt
            .and_then(|attempt| attempt.external)
            .map(|attempt| attempt.process)
            .ok_or_else(|| error("teacher_admission", "actual_process_evidence_missing"))?;
        let process = process_state
            .process_identity
            .filter(|_| {
                !process_state.quit_sent
                    && !process_state.exit_observed
                    && !process_state.cleanup_complete
                    && !process_state.quarantined
                    && !process_state.ownership_lost
            })
            .ok_or_else(|| error("teacher_admission", "actual_process_not_reusable"))?;
        let evaluation_namespace = external_evaluation_namespace(&self.config.identity)?;
        if job.envelope.evaluation_namespace != evaluation_namespace {
            return Err(error("teacher_admission", "configured_namespace_mismatch"));
        }
        let submitted = ExternalTeacherDispatch {
            job: job.envelope,
            grant: job.grant,
            scope: job.scope(),
            process,
            observed_uci,
            evaluation_namespace,
        };
        self.submit_inner(job, observed_progress)?;
        Ok(submitted)
    }
    pub fn submit(&mut self, job: ExternalJob) -> Result<(), CheckerError> {
        self.submit_inner(job, false)
    }
    fn submit_inner(
        &mut self,
        job: ExternalJob,
        observed_progress: bool,
    ) -> Result<(), CheckerError> {
        validate_submission(&job, self.owner, self.clock, &self.config)?;
        if self.shutdown.load(Ordering::Acquire)
            || self.thread.as_ref().is_none_or(JoinHandle::is_finished)
            || job.execution.sequence <= self.last_execution
        {
            return Err(error("worker", "closed_or_reused_execution"));
        }
        let sequence = job.execution.sequence;
        let envelope = job.envelope;
        let control_sequence = job.grant.control_sequence;
        let mut roots = self.roots.lock().unwrap_or_else(|p| p.into_inner());
        if roots.request.is_some()
            || roots
                .authority
                .is_some_and(|authority| authority != job.envelope.id.authority)
            || roots.last_job.is_some_and(|previous| {
                previous.id.sequence > job.envelope.id.sequence
                    || (previous.id.sequence == job.envelope.id.sequence
                        && previous != job.envelope)
            })
        {
            return Err(error("admission", "external_root_or_job_revoked"));
        }
        self.mailbox
            .admit(
                TimedExternalJob {
                    original: job,
                    admitted_at: Instant::now(),
                    observed_progress,
                },
                2,
            )
            .map_err(|rejected| match rejected.reason {
                MailboxError::QueueFull | MailboxError::ReliableFull => {
                    error("admission", "external_worker_capacity")
                }
                _ => error("admission", "external_worker_closed"),
            })?;
        self.last_execution = sequence;
        roots.authority = Some(envelope.id.authority);
        roots.last_job = Some(envelope);
        roots.execution_high_water = sequence;
        roots.control_high_water = roots.control_high_water.max(control_sequence);
        Ok(())
    }

    pub fn configuration_matches(&self, config: &ExternalCpuConfig, owner: CpuWorkerId) -> bool {
        self.owner == owner && self.config == *config
    }
    pub fn clock_matches(&self, domain: ClockDomain, origin: Instant) -> bool {
        self.clock.domain == domain && self.clock.origin == origin
    }

    /// Root revocation is independent of process shutdown and bypasses the
    /// ordinary full command queue. Accepted old jobs retain terminal slots.
    pub fn request_root_quiesce(
        &self,
        authority: SearchAuthority,
        until: Instant,
    ) -> Result<(), CheckerError> {
        if self.shutdown.load(Ordering::Acquire)
            || self.thread.as_ref().is_none_or(JoinHandle::is_finished)
        {
            return Err(error("root_quiesce", "worker_closed"));
        }
        let mut roots = self.roots.lock().unwrap_or_else(|p| p.into_inner());
        if authority.epoch != self.clock.domain.0
            || roots.authority.is_some_and(|current| current != authority)
            || roots.game_request.is_some()
        {
            return Err(error("root_quiesce", "authority_mismatch"));
        }
        if let Some(request) = roots.request {
            if request.authority != authority {
                return Err(error("root_quiesce", "another_root_pending"));
            }
            if roots.result.is_none() {
                roots.request = Some(RootRequest {
                    until: request.until.min(until),
                    ..request
                });
            }
        } else {
            roots.fence_sequence =
                roots
                    .fence_sequence
                    .checked_add(1)
                    .ok_or(CheckerError::Invalid(
                        "external root fence sequence overflow",
                    ))?;
            roots.authority = Some(authority);
            roots.request = Some(RootRequest {
                authority,
                sequence: roots.fence_sequence,
                until,
            });
            roots.result = None;
        }
        drop(roots);
        self.mailbox.signal().notify();
        Ok(())
    }

    /// Register a finite game reset after a real root fence. The worker sends
    /// ucinewgame and observes readyok; a Store metadata mirror must only arm
    /// from the returned opaque evidence, never from this request alone.
    pub fn request_new_game(
        &self,
        next: SearchAuthority,
        until: Instant,
    ) -> Result<(), CheckerError> {
        if self.shutdown.load(Ordering::Acquire)
            || self.thread.as_ref().is_none_or(JoinHandle::is_finished)
        {
            return Err(error("new_game", "worker_closed"));
        }
        let mut roots = self.roots.lock().unwrap_or_else(|p| p.into_inner());
        let previous = roots
            .result
            .as_ref()
            .and_then(|r| r.as_ref().ok())
            .copied()
            .ok_or_else(|| error("new_game", "actual_root_fence_missing"))?;
        let occupancy = self.mailbox.occupancy();
        if !previous.permits_reuse()
            || previous.owner != self.owner
            || previous.clock != self.clock.domain
            || roots.authority != Some(previous.authority)
            || next.epoch != previous.authority.epoch
            || next.implementation != previous.authority.implementation
            || next.game.0 <= previous.authority.game.0
            || next.root.0 <= previous.authority.root.0
            || occupancy.commands != 0
            || occupancy.reliable_queued != 0
            || occupancy.reliable_reserved != 0
            || occupancy.closed
        {
            return Err(error("new_game", "fence_or_authority_mismatch"));
        }
        if let Some(request) = roots.game_request {
            if request.previous != previous || request.next != next {
                return Err(error("new_game", "another_game_reset_pending"));
            }
            if roots.game_result.is_none() {
                roots.game_request = Some(GameRequest {
                    until: request.until.min(until),
                    ..request
                });
            }
        } else {
            roots.game_sequence =
                roots
                    .game_sequence
                    .checked_add(1)
                    .ok_or(CheckerError::Invalid(
                        "external game fence sequence overflow",
                    ))?;
            roots.game_request = Some(GameRequest {
                previous,
                next,
                sequence: roots.game_sequence,
                until,
            });
            roots.game_result = None;
        }
        drop(roots);
        self.mailbox.signal().notify();
        Ok(())
    }

    pub fn new_game_fence(&self) -> Option<Result<ExternalNewGameFence, CheckerError>> {
        self.roots
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .game_result
            .clone()
    }

    /// No caller can mint this readiness fence. Failure keeps the same process
    /// owner retained for explicit shutdown instead of silently spawning one.
    pub fn root_fence(&self) -> Option<Result<ExternalRootFence, CheckerError>> {
        self.roots
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .result
            .clone()
    }

    /// Reopens only an actually fenced owner. The original process, native
    /// request IDs and Job/Execution/control/progress high-water are preserved.
    /// A different game requires its actual ucinewgame/readyok proof before
    /// reopening; a root change within the game skips reset and preserves TT.
    pub fn replace_root(&self, next: SearchAuthority) -> Result<(), CheckerError> {
        if self.shutdown.load(Ordering::Acquire)
            || self.thread.as_ref().is_none_or(JoinHandle::is_finished)
        {
            return Err(error("root_reuse", "worker_closed"));
        }
        if next.epoch != self.clock.domain.0 {
            return Err(error("root_reuse", "clock_epoch_mismatch"));
        }
        let mut roots = self.roots.lock().unwrap_or_else(|p| p.into_inner());
        let Some(previous) = roots.authority else {
            roots.authority = Some(next);
            return Ok(());
        };
        if previous == next && roots.request.is_none() {
            return Ok(());
        }
        let fence = roots
            .result
            .as_ref()
            .and_then(|result| result.as_ref().ok())
            .ok_or_else(|| error("root_reuse", "actual_root_fence_missing"))?;
        let occupancy = self.mailbox.occupancy();
        let game_matches = if next.game == previous.game {
            roots.game_request.is_none()
        } else {
            roots
                .game_result
                .as_ref()
                .and_then(|r| r.as_ref().ok())
                .is_some_and(|game| {
                    game.permits_reuse()
                        && game.previous == *fence
                        && game.next == next
                        && roots
                            .game_request
                            .is_some_and(|request| request.sequence == game.sequence)
                })
        };
        if !fence.permits_reuse()
            || fence.authority != previous
            || fence.owner != self.owner
            || fence.clock != self.clock.domain
            || fence.execution_high_water != self.last_execution
            || roots
                .request
                .is_none_or(|request| request.sequence != fence.sequence)
            || next.epoch != previous.epoch
            || next.implementation != previous.implementation
            || next.game.0 < previous.game.0
            || next.root.0 <= previous.root.0
            || occupancy.commands != 0
            || occupancy.reliable_queued != 0
            || occupancy.reliable_reserved != 0
            || occupancy.progress_present
            || occupancy.closed
            || !game_matches
        {
            return Err(error("root_reuse", "fence_or_authority_mismatch"));
        }
        roots.authority = Some(next);
        roots.request = None;
        roots.result = None;
        roots.game_request = None;
        roots.game_result = None;
        drop(roots);
        self.mailbox.signal().notify();
        Ok(())
    }
    pub fn take_report(&self) -> Option<ExternalWorkerReport> {
        self.take_report_attested()
            .map(ExternalWorkerCompletion::into_report)
    }
    /// Preserve the private owner boundary for collector consumers. The legacy
    /// getter above deliberately discards this authority when returning a raw
    /// public report; callers cannot re-wrap that report into a completion.
    pub fn take_report_attested(&self) -> Option<ExternalWorkerCompletion> {
        self.mailbox.take_reliable()
    }
    pub fn take_progress(&self) -> Option<(u64, ExternalWorkerProgress)> {
        self.take_progress_observed()
            .map(|(sequence, observation)| (sequence, observation.original))
    }
    pub fn take_progress_observed(&self) -> Option<(u64, ExternalWorkerProgressObservation)> {
        self.mailbox.take_progress()
    }

    /// Admission closes first. Active go observes this atomic independently of
    /// the full command queue; queued accepted jobs retain terminal permits.
    pub fn request_shutdown(&self, deadline: Instant) {
        self.mailbox.close();
        let mut saved = self
            .shutdown_deadline
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        *saved = Some(saved.map_or(deadline, |old| old.min(deadline)));
        self.shutdown.store(true, Ordering::Release);
        self.mailbox.signal().notify();
    }

    /// Never starts a replacement owner on observation timeout. The caller
    /// retains this handle and waits on its CompletionSignal until joined.
    pub fn try_join(&mut self) -> Option<Result<CheckerShutdown, CheckerError>> {
        if self.thread.as_ref().is_some_and(JoinHandle::is_finished) {
            let result = self
                .thread
                .take()
                .expect("checked worker handle")
                .join()
                .unwrap_or_else(|_| Err(error("worker", "thread_panicked")));
            self.joined = Some(result.clone());
            Some(result)
        } else {
            None
        }
    }

    /// Returns the same immutable actual join result, including failure. This
    /// observation never issues another quit, joins twice or fabricates pipe
    /// completion from a boolean held by a caller.
    pub fn shutdown_snapshot(&self) -> Option<Result<CheckerShutdown, CheckerError>> {
        self.joined.clone()
    }
}

impl Drop for ExternalCpuWorker {
    fn drop(&mut self) {
        // Explicit shutdown/join is required for acceptance; Drop only revokes
        // admission and requests bounded owned process cleanup as a fallback.
        self.request_shutdown(Instant::now() + self.config.shutdown_grace);
    }
}

// The single thread entry transfers the original clock, admission, root-fence
// and shutdown handles once. Keep those independent authorities explicit at
// this initialization boundary rather than merging their lifecycle scopes.
#[allow(clippy::too_many_arguments)]
fn run_worker(
    config: ExternalCpuConfig,
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    startup_deadline: Instant,
    mailbox: Mailbox,
    shutdown: Arc<AtomicBool>,
    shutdown_deadline: Arc<Mutex<Option<Instant>>>,
    startup_report: Arc<Mutex<Option<ExternalWorkerStartup>>>,
    roots: Arc<Mutex<RootLifecycle>>,
    launch_pin: Option<ExternalLaunchPin>,
) -> Result<CheckerShutdown, CheckerError> {
    let mut checker = ExternalUciCpuChecker::create(config.clone())?;
    checker.launch_pin = launch_pin;
    let evaluation_namespace = external_evaluation_namespace(&config.identity)?;
    // Jobs may already be admitted while the private worker is handshaking.
    // Preserve the startup failure and their terminal obligations if that
    // boundary panics; no replacement process or successful fence is inferred.
    let startup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let read_fence = || physical_fence(&shutdown_deadline, &roots);
        checker.start_with_cleanup_fence(startup_deadline, &shutdown, Some(&read_fence))
    }))
    .unwrap_or_else(|_| {
        checker.failed = true;
        Err(error("startup", "worker_startup_panicked"))
    });
    *startup_report
        .lock()
        .unwrap_or_else(|poison| poison.into_inner()) = Some(ExternalWorkerStartup {
        result: startup.clone().and_then(|()| {
            checker
                .startup_uci()
                .ok_or_else(|| error("startup", "handshake_identity_missing"))
        }),
        attempt: checker.last_attempt().cloned(),
    });
    mailbox.signal().notify();
    if startup.is_err() {
        // Startup has failed: revoke admission but preserve any already
        // accepted commands for explicit NotIssued terminal reports below.
        let mut fence = shutdown_deadline
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        *fence = Some(fence.map_or(startup_deadline, |old| old.min(startup_deadline)));
        shutdown.store(true, Ordering::Release);
        mailbox.close();
    }
    let progress_sequence = Arc::new(AtomicU64::new(0));
    let mut last_authority: Option<rz_contracts::pals::SearchAuthority> = None;
    let mut game_reset_high_water = 0_u64;
    loop {
        let observed = mailbox.signal().version();
        if let Some(mut admitted) = mailbox.take_command() {
            // Retain a second reliable slot outside the fallible task closure.
            // A panic cannot erase this execution's terminal obligation.
            let mut supervisor = admitted
                .reliable
                .split_off(1)
                .map_err(|_| error("worker", "failure_permit_split"))?;
            let admitted_at = admitted.command.admitted_at;
            let observed_progress = admitted.command.observed_progress;
            let job = admitted.command.original;
            let scope = job.scope();
            let observations = Arc::new(Mutex::new(ExternalOwnerObservations::new(
                admitted_at,
                Instant::now(),
            )));
            if startup.is_ok() {
                checker.reset_attempt();
                checker.go_issue = ExternalGoIssue::NotIssued;
            }
            let operation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if shutdown.load(Ordering::Acquire) {
                    Err(error("admission", "external_worker_shutdown_before_go"))
                } else if roots
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .request
                    .is_some()
                {
                    Err(error("admission", "external_root_revoked_before_go"))
                } else if let Err(failure) = &startup {
                    Err(failure.clone())
                } else {
                    validate_job(&job, owner, clock, &config).and_then(|deadline| {
                        if let Some(previous) = last_authority {
                            if previous.epoch != job.envelope.id.authority.epoch
                                || previous.game > job.envelope.id.authority.game
                                || (previous.game == job.envelope.id.authority.game
                                    && previous.root > job.envelope.id.authority.root)
                            {
                                return Err(error("admission", "old_game_or_root_job"));
                            }
                            if previous.game != job.envelope.id.authority.game {
                                return Err(error("new_game", "actual_game_reset_missing"));
                            }
                        }
                        last_authority = Some(job.envelope.id.authority);
                        let control = WorkerControl {
                            scope,
                            controls: job.control.clone(),
                            mailbox: mailbox.clone(),
                            progress_sequence: progress_sequence.clone(),
                            shutdown_deadline: shutdown_deadline.clone(),
                            roots: roots.clone(),
                            observations: observations.clone(),
                            observed_progress,
                        };
                        checker.analyze_cooperative(
                            &job.position,
                            job.root_moves.as_deref(),
                            CpuLimits {
                                max_depth: job.max_depth,
                                max_nodes: job.policy.requested_reported_nodes,
                                deadline: Some(deadline),
                            },
                            &shutdown,
                            &control,
                            job.policy,
                        )
                    })
                }
            }));
            let analyze_returned_at = Instant::now();
            let (result, panicked) = match operation {
                Ok(result) => (result, false),
                Err(_) => {
                    checker.failed = true;
                    // The accepted envelope was admission-validated before
                    // entering this worker. Retain its original physical wall
                    // even if the operation failed by panic rather than Err.
                    if let Some(original) = clock
                        .origin
                        .checked_add(Duration::from_nanos(job.envelope.hard_deadline.at.0))
                    {
                        let mut fence = shutdown_deadline
                            .lock()
                            .unwrap_or_else(|poison| poison.into_inner());
                        *fence = Some(fence.map_or(original, |old| old.min(original)));
                    }
                    shutdown.store(true, Ordering::Release);
                    mailbox.close();
                    (Err(error("worker", "task_panicked")), true)
                }
            };
            if let Ok(report) = &result {
                if report.readiness_barrier_completed {
                    if let Some(control) = report.stop_control {
                        match control.action {
                            ExternalControlAction::Yield => {
                                let _ = job.control.acknowledge_yield(control.sequence);
                            }
                            ExternalControlAction::Cancel => {
                                let _ = job.control.acknowledge_cancel(control.sequence);
                            }
                            ExternalControlAction::Continue | ExternalControlAction::Revoked => {}
                        }
                    }
                }
            }
            let controls = job.control.seal_terminal();
            {
                let mut root = roots.lock().unwrap_or_else(|p| p.into_inner());
                root.control_high_water = root
                    .control_high_water
                    .max(controls.cancel_sequence)
                    .max(controls.yield_sequence)
                    .max(controls.publish_sequence);
            }
            let report = ExternalWorkerReport {
                job: job.envelope,
                grant: job.grant,
                scope,
                result,
                attempt: checker.last_attempt().cloned(),
                controls,
                exact_resume_supported: false,
                go_issue: checker.go_issue,
            };
            let completion = ExternalWorkerCompletion {
                report,
                process: checker
                    .owner
                    .as_ref()
                    .and_then(|owned| owned.evidence().process_identity),
                evaluation_namespace,
                timing: {
                    let mut observed = observations
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner());
                    observed.timing.analyze_returned_at = analyze_returned_at;
                    observed.timing.report_ready_at = Instant::now();
                    Some(observed.timing)
                },
            };
            // Admission reserved the normal terminal and its failure backup
            // atomically; only one terminal fact is published for this scope.
            let published = if panicked {
                supervisor.publish(completion)
            } else {
                admitted.reliable.publish(completion)
            };
            if published.is_err() {
                return Err(error("worker", "reserved_terminal_publication_failed"));
            }
            continue;
        }
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        let request = {
            let root = roots.lock().unwrap_or_else(|p| p.into_inner());
            root.result.is_none().then_some(root.request).flatten()
        };
        if let Some(request) = request {
            // Reliable terminals remain owned until the manager consumes
            // them, but that must not leave root closure pending forever.
            // Expiry publishes failure without dropping those obligations or
            // claiming a physical fence; the process remains for shutdown.
            let expired = {
                let mut root = roots.lock().unwrap_or_else(|p| p.into_inner());
                let expired = root.result.is_none()
                    && root.request.is_some_and(|current| {
                        current.authority == request.authority
                            && current.sequence == request.sequence
                            && Instant::now() >= current.until
                    });
                if expired {
                    root.result =
                        Some(Err(error("root_quiesce", "deadline_before_terminal_drain")));
                }
                expired
            };
            if expired {
                mailbox.signal().notify();
                continue;
            }
            let occupancy = mailbox.occupancy();
            if occupancy.commands == 0
                && occupancy.reliable_queued == 0
                && occupancy.reliable_reserved == 0
            {
                // Coalesced progress has no terminal obligation and cannot be
                // consumed as a new-root observation after this fence.
                mailbox.take_progress();
                let read_fence = || physical_fence(&shutdown_deadline, &roots);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    checker.quiesce_root(request.until, &shutdown, &read_fence)
                }))
                .unwrap_or_else(|_| {
                    checker.failed = true;
                    Err(error("root_quiesce", "worker_quiesce_panicked"))
                });
                let mut root = roots.lock().unwrap_or_else(|p| p.into_inner());
                // request_root_quiesce only shortens this same request. It
                // cannot replace authority or consume a fence during the call.
                let result = result.and_then(|process| {
                    let current = root
                        .request
                        .ok_or_else(|| error("root_quiesce", "request_lost"))?;
                    if current.authority != request.authority
                        || current.sequence != request.sequence
                        || Instant::now() >= current.until
                    {
                        return Err(error("root_quiesce", "late_or_replaced_ready"));
                    }
                    Ok(ExternalRootFence {
                        authority: request.authority,
                        owner,
                        clock: clock.domain,
                        sequence: request.sequence,
                        execution_high_water: root.execution_high_water,
                        job_high_water: root.last_job.map_or(0, |job| job.id.sequence),
                        control_high_water: root.control_high_water,
                        progress_high_water: progress_sequence.load(Ordering::Acquire),
                        native_request_high_water: checker.request_id,
                        game_reset_high_water,
                        source: external_evaluation_namespace(&config.identity)?,
                        process,
                        readiness_observed: true,
                    })
                });
                // Even a root without a submitted Job participates in game
                // authority. A later game's first go must reset this owner.
                if result.is_ok() {
                    last_authority = Some(request.authority);
                }
                root.result = Some(result);
                drop(root);
                mailbox.signal().notify();
                continue;
            }
        }
        let game_request = {
            let root = roots.lock().unwrap_or_else(|p| p.into_inner());
            root.game_result
                .is_none()
                .then_some(root.game_request)
                .flatten()
        };
        if let Some(request) = game_request {
            let expired = {
                let mut root = roots.lock().unwrap_or_else(|p| p.into_inner());
                let expired = root.game_result.is_none()
                    && root.game_request.is_some_and(|current| {
                        current.previous == request.previous
                            && current.next == request.next
                            && current.sequence == request.sequence
                            && Instant::now() >= current.until
                    });
                if expired {
                    root.game_result = Some(Err(error("new_game", "deadline_before_owner_idle")));
                }
                expired
            };
            if expired {
                mailbox.signal().notify();
                continue;
            }
            let occupancy = mailbox.occupancy();
            if occupancy.commands == 0
                && occupancy.reliable_queued == 0
                && occupancy.reliable_reserved == 0
            {
                let read_fence = || game_physical_fence(&shutdown_deadline, &roots);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let next_count =
                        game_reset_high_water
                            .checked_add(1)
                            .ok_or(CheckerError::Invalid(
                                "external game reset counter overflow",
                            ))?;
                    checker.reset_game_with_fence(request.until, &shutdown, Some(&read_fence))?;
                    let process = checker.owner()?.evidence();
                    game_reset_high_water = next_count;
                    Ok((process, next_count))
                }))
                .unwrap_or_else(|_| Err(error("new_game", "worker_game_reset_panicked")));
                let mut root = roots.lock().unwrap_or_else(|p| p.into_inner());
                let result = result.and_then(|(process, count)| {
                    let current = root
                        .game_request
                        .ok_or_else(|| error("new_game", "request_lost"))?;
                    if current.next != request.next
                        || current.sequence != request.sequence
                        || current.previous != request.previous
                        || Instant::now() >= current.until
                    {
                        return Err(error("new_game", "late_or_replaced_ready"));
                    }
                    let fence = ExternalNewGameFence {
                        previous: request.previous,
                        next: request.next,
                        sequence: request.sequence,
                        game_reset_high_water: count,
                        process,
                        readiness_observed: true,
                    };
                    if !fence.permits_reuse() {
                        return Err(error("new_game", "owner_not_reusable"));
                    }
                    Ok(fence)
                });
                if result.is_ok() {
                    last_authority = Some(request.next);
                } else {
                    checker.failed = true;
                }
                root.game_result = Some(result);
                drop(root);
                mailbox.signal().notify();
                continue;
            }
        }
        mailbox.wait_changed(observed, Duration::from_millis(20));
    }
    let deadline = shutdown_deadline
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .unwrap_or_else(|| Instant::now() + config.shutdown_grace);
    checker.shutdown(deadline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_restriction_order_and_none_are_not_interchangeable() {
        let a = BoardMove::from_uci("e2e4").unwrap();
        let b = BoardMove::from_uci("d2d4").unwrap();
        assert_ne!(
            external_root_restriction_digest(None),
            external_root_restriction_digest(Some(&[]))
        );
        assert_ne!(
            external_root_restriction_digest(Some(&[a, b])),
            external_root_restriction_digest(Some(&[b, a]))
        );
    }

    #[test]
    fn pending_control_is_not_prior_process_stop_evidence() {
        use rz_contracts::pals::SearchAuthority;
        use rz_contracts::{GameGeneration, ProcessEpoch, RootGeneration};
        let scope = ExternalWorkerScope {
            job: CpuJobId {
                authority: SearchAuthority {
                    epoch: ProcessEpoch(1),
                    game: GameGeneration(1),
                    root: RootGeneration(1),
                    implementation: Digest([1; 32]),
                },
                sequence: 1,
            },
            execution: ExecutionId::new(ProcessEpoch(1), 1),
            owner: CpuWorkerId(0),
        };
        let mailbox = Mailbox::new(8, 16, CompletionSignal::default()).unwrap();
        let controls = ControlMailbox::new(scope, mailbox.signal());
        let now = Instant::now();
        let reader = WorkerControl {
            scope,
            controls: controls.clone(),
            mailbox,
            progress_sequence: Arc::new(AtomicU64::new(0)),
            shutdown_deadline: Arc::new(Mutex::new(None)),
            roots: Arc::new(Mutex::new(RootLifecycle::default())),
            observations: Arc::new(Mutex::new(ExternalOwnerObservations::new(now, now))),
            observed_progress: false,
        };
        controls.request_yield(1).unwrap();
        assert_eq!(reader.observe().action, ExternalControlAction::Yield);
        assert_eq!(controls.snapshot().yield_ack, 0);
        controls.request_cancel(2).unwrap();
        assert_eq!(reader.observe().action, ExternalControlAction::Cancel);
        assert_eq!(reader.observe().sequence, 2);
        controls.seal_terminal();
        assert_eq!(reader.observe().action, ExternalControlAction::Revoked);
        assert_eq!(reader.observe().sequence, 0);
    }

    fn job_fixture() -> (ExternalJob, ExternalCpuConfig, ExternalWorkerClock) {
        use rz_contracts::pals::SearchAuthority;
        use rz_contracts::{Deadline, GameGeneration, OwnerId, ProcessEpoch, RootGeneration};
        let owner = CpuWorkerId(0);
        let epoch = ProcessEpoch(7);
        let clock = ExternalWorkerClock {
            domain: ClockDomain(epoch),
            origin: Instant::now(),
        };
        let identity = ExternalCheckerIdentity {
            adapter_semantics: EXTERNAL_UCI_SCORE_SEMANTICS.to_owned(),
            binary_sha256: "0".repeat(64),
            launch_arguments_sha256: arguments_sha256(&[]),
            declared_name: "fixture".to_owned(),
            declared_version: "fixture/1".to_owned(),
            declared_source: "RoveZero source fixture".to_owned(),
            declared_license: "MIT".to_owned(),
            options: BTreeMap::new(),
            assets: Vec::new(),
            model_metadata: ExternalModelMetadata {
                weights_sha256: None,
                training: ExternalTrainingKnowledge::Unknown,
                declared_rights: None,
                precision: None,
            },
        };
        let config = ExternalCpuConfig {
            program: PathBuf::from("/fixture/uci"),
            arguments: Vec::new(),
            working_directory: PathBuf::from("/fixture"),
            identity,
            expected_uci_name: None,
            max_depth: 8,
            max_prefix_plies: 256,
            handshake_timeout: Duration::from_secs(1),
            max_task_wall_time: Duration::from_secs(2),
            stop_grace: EXTERNAL_STOP_DRAIN_RESERVE,
            shutdown_grace: Duration::from_secs(1),
            max_output_bytes: 65536,
            max_line_bytes: 4096,
        };
        let position = Position::startpos();
        let state = rz_position::contracts::ContractPosition::new(OwnerId(47), position.clone())
            .export()
            .unwrap();
        let state_identity = state.snapshot().identity();
        let policy = ExternalSlicePolicy::default();
        let root_restriction = external_root_restriction_digest(None);
        let evaluation_namespace = external_evaluation_namespace(&config.identity).unwrap();
        let search_conditions = external_search_conditions(&config, 8, policy);
        let id = CpuJobId {
            authority: SearchAuthority {
                epoch,
                game: GameGeneration(2),
                root: RootGeneration(3),
                implementation: Digest([1; 32]),
            },
            sequence: 1,
        };
        let execution = ExecutionId::new(epoch, 1);
        let envelope = ExternalCpuJobEnvelope {
            id,
            state_identity,
            root_restriction,
            evaluation_namespace,
            search_conditions,
            question: external_question_digest(
                state_identity,
                root_restriction,
                evaluation_namespace,
                search_conditions,
            ),
            requested_reported_node_hint: policy.requested_reported_nodes,
            hard_deadline: Deadline {
                clock: clock.domain,
                at: MonotonicTick(2_000_000_000),
            },
        };
        let control = ControlMailbox::new(
            ExternalWorkerScope {
                job: id,
                execution,
                owner,
            },
            CompletionSignal::default(),
        );
        (
            ExternalJob {
                envelope,
                grant: ExternalSliceGrant {
                    job: id,
                    execution,
                    owner,
                    requested_reported_node_hint: policy.requested_reported_nodes,
                    soft_quantum: policy.quantum,
                    control_sequence: 1,
                },
                execution,
                owner,
                position,
                root_moves: None,
                max_depth: 8,
                policy,
                control,
            },
            config,
            clock,
        )
    }

    #[test]
    fn worker_checks_exact_rules_history_question_owner_and_original_clock() {
        let (job, config, clock) = job_fixture();
        assert!(validate_job(&job, job.owner, clock, &config).is_ok());
        let mut changed = job.clone();
        changed
            .position
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        assert!(validate_job(&changed, job.owner, clock, &config).is_err());
        let mut changed = job.clone();
        changed.envelope.question = Digest([8; 32]);
        assert!(validate_job(&changed, job.owner, clock, &config).is_err());
        let mut changed = job.clone();
        changed.envelope.id.authority.game.0 += 1;
        assert!(validate_job(&changed, job.owner, clock, &config).is_err());
        assert!(validate_job(&job, CpuWorkerId(1), clock, &config).is_err());
        let mut other_clock = clock;
        other_clock.domain.0.0 += 1;
        assert!(validate_job(&job, job.owner, other_clock, &config).is_err());
        job.control.seal_terminal();
        assert!(validate_submission(&job, job.owner, clock, &config).is_err());
        assert!(validate_job(&job, job.owner, clock, &config).is_err());
    }

    #[test]
    fn external_namespace_records_runtime_conditions_without_native_credit_conversion() {
        let (job, mut config, clock) = job_fixture();
        config.max_task_wall_time += Duration::from_millis(1);
        assert!(validate_job(&job, job.owner, clock, &config).is_err());
        let (mut job, config, clock) = job_fixture();
        job.policy.requested_reported_nodes = 1;
        assert_eq!(
            external_search_conditions(&config, 8, job.policy),
            job.envelope.search_conditions
        );
        job.envelope.search_conditions = external_search_conditions(&config, 8, job.policy);
        job.envelope.question = external_question_digest(
            job.envelope.state_identity,
            job.envelope.root_restriction,
            job.envelope.evaluation_namespace,
            job.envelope.search_conditions,
        );
        job.grant.requested_reported_node_hint = 1;
        // Typed external contracts have no native reservation field.
        assert_eq!(job.envelope.requested_reported_node_hint, 65_536);
        assert!(validate_job(&job, job.owner, clock, &config).is_ok());
        job.policy.quantum = Duration::from_millis(200);
        job.grant.soft_quantum = job.policy.quantum;
        assert_eq!(
            external_search_conditions(&config, 8, job.policy),
            job.envelope.search_conditions
        );
        assert!(validate_job(&job, job.owner, clock, &config).is_ok());
    }
}
