//! Private own-CPU worker and manager-owned exact native credit.
//!
//! A job fixes Rules/history, restriction, evaluator, conditions and deadline.
//! A grant changes occupancy only. Tokens never leave this worker and actual
//! invocation deltas, rather than progress snapshots, settle reservations.

use crate::contract_time::{ContractClock, InstantClock};
use crate::cpu::{
    CpuCompletion, CpuControlMailbox, CpuControlledReport, CpuEngine, CpuError,
    CpuIterationProgress, CpuLimits, CpuMailboxScope, CpuResumePolicy, CpuResumeToken,
    CpuSliceControl, CpuWork,
};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuJobId, CpuWorkerId, SliceGrant};
use rz_contracts::{Digest, ExecutionId};
use rz_position::contracts::ContractPosition;
use rz_position::{BoardMove, Position};
use rz_runtime::CompletionSignal;
use rz_runtime::adaptive::{
    ControlSnapshot, CpuActor, CpuResourceError, CpuThreadReadback, MailboxError, ReliablePermit,
    WorkerMailbox, observe_current_cpu_thread,
};
use sha2::{Digest as _, Sha256};
use std::collections::HashMap;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub const CPU_WORKER_COMMAND_CAPACITY: usize = 8;
pub const CPU_JOB_CAPACITY: usize = 64;
pub const CPU_FIXED_SLICE_NODES: u64 = 4096;
pub const CPU_FIXED_SOFT_LEASE: Duration = Duration::from_millis(8);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuWorkerError {
    Invalid(&'static str),
    Capacity,
    UnknownJob,
    AlreadyReserved,
    Duplicate,
    UnknownUsage,
    Quarantined,
    Disconnected,
    Cpu(CpuError),
}
impl From<CpuError> for CpuWorkerError {
    fn from(error: CpuError) -> Self {
        Self::Cpu(error)
    }
}

/// Only a worker's definite pre-enqueue rejection can create this receipt.
/// A missing/failed response from an admitted execution cannot release credit.
#[derive(Clone, Debug)]
pub struct CpuAdmissionRejection {
    grant: SliceGrant,
}

pub struct CpuSubmissionRejected {
    reason: CpuWorkerError,
    proof: CpuAdmissionRejection,
    job: Option<CpuJob>,
}
impl std::fmt::Debug for CpuSubmissionRejected {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CpuSubmissionRejected")
            .field("reason", &self.reason)
            .field("proof", &self.proof)
            .field("job_retained", &self.job.is_some())
            .finish()
    }
}
impl CpuSubmissionRejected {
    pub fn reason(&self) -> &CpuWorkerError {
        &self.reason
    }
    pub fn proof(&self) -> &CpuAdmissionRejection {
        &self.proof
    }
    pub fn into_parts(self) -> (CpuWorkerError, CpuAdmissionRejection, Option<CpuJob>) {
        (self.reason, self.proof, self.job)
    }
    fn from_command(reason: CpuWorkerError, command: CpuCommand) -> Self {
        let (grant, job) = match command {
            CpuCommand::Begin(job, grant) => (grant, Some(job)),
            CpuCommand::Resume(grant) => (grant, None),
        };
        Self {
            reason,
            proof: CpuAdmissionRejection { grant },
            job,
        }
    }
}

pub fn cpu_root_restriction_digest(moves: Option<&[BoardMove]>) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-cpu-root-restriction/1");
    hash.update([u8::from(moves.is_some())]);
    if let Some(moves) = moves {
        hash.update((moves.len() as u64).to_be_bytes());
        for movement in moves {
            let encoded = rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(*movement).expect("Rules board move is representable"),
            );
            hash.update(encoded.bits().to_be_bytes());
        }
    }
    Digest(hash.finalize().into())
}

pub fn cpu_evaluation_namespace_digest(engine: &CpuEngine) -> Result<Digest, CpuWorkerError> {
    let bytes = serde_json::to_vec(engine.value_identity())
        .map_err(|_| CpuWorkerError::Invalid("CPU value identity serialization"))?;
    let mut hash = Sha256::new();
    hash.update(b"rz-cpu-evaluation-namespace/1");
    hash.update(bytes);
    Ok(Digest(hash.finalize().into()))
}

pub fn cpu_search_conditions_digest(engine: &CpuEngine) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-cpu-search-conditions/1");
    hash.update(engine.search_conditions().as_bytes());
    Digest(hash.finalize().into())
}

pub fn compiled_cooperative_cpu_owner_sha256() -> [u8; 32] {
    let mut hash = Sha256::new();
    for bytes in [
        b"rz-cooperative-cpu-owner/1".as_slice(),
        include_bytes!("cpu.rs").as_slice(),
        include_bytes!("cpu/stack.rs").as_slice(),
        include_bytes!("cpu_worker.rs").as_slice(),
    ] {
        hash.update((bytes.len() as u64).to_be_bytes());
        hash.update(bytes);
    }
    hash.finalize().into()
}

pub struct CpuJob {
    envelope: CpuJobEnvelope,
    position: Position,
    root_moves: Option<Vec<BoardMove>>,
    target_depth: u16,
    hard_deadline: Instant,
    clock: InstantClock,
    pub mailbox: Arc<CpuControlMailbox>,
}

impl CpuJob {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        envelope: CpuJobEnvelope,
        position: ContractPosition,
        root_moves: Option<Vec<BoardMove>>,
        target_depth: u16,
        clock: &InstantClock,
        engine: &CpuEngine,
        owner: CpuWorkerId,
        signal: CompletionSignal,
    ) -> Result<Self, CpuWorkerError> {
        if engine.resume_policy() != CpuResumePolicy::PausedStack {
            return Err(CpuWorkerError::Invalid("CPU cooperative policy"));
        }
        Self::from_profile(
            envelope,
            position,
            root_moves,
            target_depth,
            clock,
            engine.config().max_depth,
            cpu_evaluation_namespace_digest(engine)?,
            cpu_search_conditions_digest(engine),
            owner,
            signal,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn from_profile(
        envelope: CpuJobEnvelope,
        position: ContractPosition,
        root_moves: Option<Vec<BoardMove>>,
        target_depth: u16,
        clock: &InstantClock,
        max_depth: u16,
        evaluation_namespace: Digest,
        search_conditions: Digest,
        owner: CpuWorkerId,
        signal: CompletionSignal,
    ) -> Result<Self, CpuWorkerError> {
        let state = position
            .export()
            .map_err(|_| CpuWorkerError::Invalid("Rules export failed"))?;
        if state.snapshot().identity() != envelope.state_identity {
            return Err(CpuWorkerError::Invalid("job Rules identity mismatch"));
        }
        if envelope.total_native_node_credit == 0
            || envelope.id.sequence == 0
            || envelope.id.authority.epoch != envelope.hard_deadline.clock.0
        {
            return Err(CpuWorkerError::Invalid(
                "job credit or authority is invalid",
            ));
        }
        envelope
            .hard_deadline
            .accepts(
                clock.domain(),
                clock
                    .now()
                    .map_err(|_| CpuWorkerError::Invalid("CPU clock observation failed"))?,
            )
            .map_err(|_| CpuWorkerError::Invalid("CPU job deadline is not live"))?;
        let hard_deadline = clock
            .instant_at(envelope.hard_deadline.at)
            .map_err(|_| CpuWorkerError::Invalid("CPU deadline is not representable"))?;
        if target_depth == 0 || target_depth > max_depth {
            return Err(CpuWorkerError::Invalid("CPU cooperative depth or policy"));
        }
        if envelope.root_restriction != cpu_root_restriction_digest(root_moves.as_deref())
            || envelope.evaluation_namespace != evaluation_namespace
            || envelope.search_conditions != search_conditions
        {
            return Err(CpuWorkerError::Invalid("CPU immutable namespace mismatch"));
        }
        if let Some(moves) = &root_moves {
            let legal = state.legal_moves().moves();
            if moves.is_empty()
                || moves.len() > 256
                || moves.iter().enumerate().any(|(i, movement)| {
                    moves[..i].contains(movement)
                        || !legal
                            .contains(&rz_contracts::Move::try_from(*movement).expect("Rules move"))
                })
            {
                return Err(CpuWorkerError::Invalid("CPU root restriction is illegal"));
            }
        }
        let mailbox = Arc::new(CpuControlMailbox::new(
            CpuMailboxScope {
                job: envelope.id,
                owner,
            },
            signal,
        ));
        Ok(Self {
            envelope,
            position: position.into_position(),
            root_moves,
            target_depth,
            hard_deadline,
            clock: *clock,
            mailbox,
        })
    }
    pub fn envelope(&self) -> CpuJobEnvelope {
        self.envelope
    }
}

#[derive(Clone, Debug)]
pub struct CpuSliceReport {
    pub grant: SliceGrant,
    /// None is unknown and must quarantine the outstanding reservation.
    pub actual_work: Option<CpuWork>,
    pub result: Result<CpuControlledReport, CpuError>,
    /// Total Job work including earlier slices; progress nodes are invocation-local.
    /// An admission rejection did not inspect or execute a continuation, so
    /// its cumulative snapshot is not settlement evidence; the known delta is zero.
    pub cumulative_nodes: u64,
    pub progress_sequence: u64,
    pub continuation_retained: bool,
    pub terminal: bool,
    pub admission_rejected: bool,
    pub controls: Option<ControlSnapshot>,
}

#[derive(Clone, Debug)]
pub struct CpuJobTermination {
    pub job: CpuJobId,
    pub owner: CpuWorkerId,
    pub last_execution: ExecutionId,
    pub completion: CpuCompletion,
    pub controls: ControlSnapshot,
    pub retained_bytes: usize,
    /// Owner shutdown is distinct from a sequenced job cancel request.
    pub shutdown: bool,
}

#[derive(Clone, Debug)]
pub enum CpuWorkerEvent {
    Slice(CpuSliceReport),
    JobTerminated(CpuJobTermination),
}

/// Reliable receipts are returned before disposing the closed owner. A joined
/// thread alone is not proof that every admitted execution has settled work.
#[derive(Debug)]
pub struct CpuWorkerShutdown {
    pub owner: CpuWorkerId,
    pub events: Vec<CpuWorkerEvent>,
    pub last_progress: Option<CpuWorkerProgress>,
    pub thread_joined: bool,
    pub thread_panicked: bool,
    pub mailbox: rz_runtime::adaptive::MailboxOccupancy,
    /// Ready-boundary observation, distinct from the declared engine profile.
    /// None is not observed; Err retains a startup probe/role failure and never
    /// attests a complete inventory or an applied resource setting.
    pub ready_readback: Option<Result<CpuThreadReadback, CpuResourceError>>,
}

struct CreditAccount {
    envelope: CpuJobEnvelope,
    spent: u64,
    reservation: Option<SliceGrant>,
    last_settled_sequence: u64,
    last_started_sequence: u64,
    last_control_sequence: u64,
    terminal: bool,
    quarantine: bool,
    owner: Option<CpuWorkerId>,
}

/// Root-scoped single manager writer. Unknown failure retains reservation;
/// snapshots do not settle credit. Completed receipts are externally retained.
pub struct CpuCreditLedger {
    jobs: HashMap<CpuJobId, CreditAccount>,
    authority: SearchAuthority,
    clock: InstantClock,
    global_limit: u64,
    global_spent: u64,
    highest_job_sequence: u64,
    highest_execution_sequence: u64,
}
impl CpuCreditLedger {
    pub fn new(
        authority: SearchAuthority,
        global_limit: u64,
        clock: InstantClock,
    ) -> Result<Self, CpuWorkerError> {
        if global_limit == 0 || clock.domain().0 != authority.epoch {
            return Err(CpuWorkerError::Invalid(
                "CPU ledger root, clock or credit mismatch",
            ));
        }
        Ok(Self {
            jobs: HashMap::new(),
            authority,
            clock,
            global_limit,
            global_spent: 0,
            highest_job_sequence: 0,
            highest_execution_sequence: 0,
        })
    }
    pub fn register(&mut self, envelope: CpuJobEnvelope) -> Result<(), CpuWorkerError> {
        if envelope.id.authority != self.authority
            || envelope.id.sequence <= self.highest_job_sequence
        {
            return Err(CpuWorkerError::Duplicate);
        }
        envelope
            .validate(
                self.clock.domain(),
                self.clock
                    .now()
                    .map_err(|_| CpuWorkerError::Invalid("CPU ledger clock failed"))?,
            )
            .map_err(|_| CpuWorkerError::Invalid("CPU job deadline or authority invalid"))?;
        if envelope.total_native_node_credit > self.global_limit {
            return Err(CpuWorkerError::Invalid(
                "CPU job exceeds global native credit",
            ));
        }
        self.jobs.retain(|_, account| {
            !account.terminal || account.quarantine || account.reservation.is_some()
        });
        if self.jobs.len() >= CPU_JOB_CAPACITY {
            return Err(CpuWorkerError::Capacity);
        }
        self.highest_job_sequence = envelope.id.sequence;
        self.jobs.insert(
            envelope.id,
            CreditAccount {
                envelope,
                spent: 0,
                reservation: None,
                last_settled_sequence: 0,
                last_started_sequence: 0,
                last_control_sequence: 0,
                terminal: false,
                quarantine: false,
                owner: None,
            },
        );
        Ok(())
    }
    pub fn reserve(&mut self, grant: SliceGrant) -> Result<(), CpuWorkerError> {
        if grant.job.authority != self.authority
            || grant.execution.sequence <= self.highest_execution_sequence
        {
            return Err(CpuWorkerError::Duplicate);
        }
        if self.jobs.iter().any(|(id, account)| {
            *id != grant.job && !account.terminal && account.owner == Some(grant.owner)
        }) {
            return Err(CpuWorkerError::Invalid(
                "CPU owner already holds another primary job",
            ));
        }
        let (_, outstanding) = self.global_usage()?;
        if self
            .global_spent
            .checked_add(outstanding)
            .and_then(|sum| sum.checked_add(grant.node_credit))
            .is_none_or(|sum| sum > self.global_limit)
        {
            return Err(CpuWorkerError::Invalid(
                "CPU grant exceeds global native credit",
            ));
        }
        let account = self
            .jobs
            .get_mut(&grant.job)
            .ok_or(CpuWorkerError::UnknownJob)?;
        if account.quarantine {
            return Err(CpuWorkerError::Quarantined);
        }
        if account.terminal {
            return Err(CpuWorkerError::Invalid("CPU job has terminated"));
        }
        if account.reservation.is_some() {
            return Err(CpuWorkerError::AlreadyReserved);
        }
        if account.owner.is_some_and(|owner| owner != grant.owner) {
            return Err(CpuWorkerError::Invalid(
                "CPU primary job cannot move between owners",
            ));
        }
        grant
            .validate(
                &account.envelope,
                self.clock.domain(),
                self.clock
                    .now()
                    .map_err(|_| CpuWorkerError::Invalid("CPU ledger clock failed"))?,
            )
            .map_err(|_| {
                CpuWorkerError::Invalid("CPU slice lease or immutable deadline invalid")
            })?;
        if grant.control_sequence <= account.last_control_sequence {
            return Err(CpuWorkerError::Invalid("CPU control sequence is stale"));
        }
        if account
            .spent
            .checked_add(grant.node_credit)
            .is_none_or(|sum| sum > account.envelope.total_native_node_credit)
        {
            return Err(CpuWorkerError::Invalid(
                "CPU slice exceeds remaining exact credit",
            ));
        }
        account.owner = Some(grant.owner);
        account.reservation = Some(grant);
        self.highest_execution_sequence = grant.execution.sequence;
        Ok(())
    }
    pub fn settle(&mut self, report: &CpuSliceReport) -> Result<(), CpuWorkerError> {
        let account = self
            .jobs
            .get_mut(&report.grant.job)
            .ok_or(CpuWorkerError::UnknownJob)?;
        if account.quarantine {
            return Err(CpuWorkerError::Quarantined);
        }
        if account.reservation.as_ref() != Some(&report.grant) {
            return Err(CpuWorkerError::Duplicate);
        }
        let Some(work) = report.actual_work else {
            account.quarantine = true;
            return Err(CpuWorkerError::UnknownUsage);
        };
        let invalid_result = match &report.result {
            Ok(value) => {
                value.report.nodes != work.nodes
                    || value.report.quiescence_nodes != work.quiescence_nodes
                    || value.report.tt_hits != work.tt_hits
                    || value.report.resume.is_some()
                    || (value.report.completion == CpuCompletion::Yielded
                        && (report.terminal || !report.continuation_retained))
                    || (value.report.completion != CpuCompletion::Yielded
                        && (!report.terminal || report.continuation_retained))
            }
            Err(_) => !report.admission_rejected && !report.terminal,
        };
        let spent = account.spent.checked_add(work.nodes);
        let global_spent = self.global_spent.checked_add(work.nodes);
        if work.nodes > report.grant.node_credit
            || spent.is_none_or(|sum| sum > account.envelope.total_native_node_credit)
            || (!report.admission_rejected && spent != Some(report.cumulative_nodes))
            || global_spent.is_none_or(|sum| sum > self.global_limit)
            || invalid_result
            || (report.terminal && report.continuation_retained)
            || (report.admission_rejected
                && (work != CpuWork::default() || report.result.is_ok() || report.terminal))
        {
            account.quarantine = true;
            return Err(CpuWorkerError::Invalid(
                "CPU actual usage or completion violates reservation",
            ));
        }
        account.spent = spent.expect("checked native credit");
        self.global_spent = global_spent.expect("checked global native credit");
        account.last_settled_sequence = report.grant.execution.sequence;
        if !report.admission_rejected {
            account.last_started_sequence = report.grant.execution.sequence;
        }
        account.last_control_sequence = report.grant.control_sequence;
        account.reservation = None;
        // A definite first-Begin rejection never acquired a private stack.
        // Rejected Resume keeps the existing same-owner continuation authority.
        if report.admission_rejected && account.last_started_sequence == 0 {
            account.owner = None;
        }
        account.terminal |= report.terminal;
        Ok(())
    }
    /// Roll back only a command proved not to have entered its worker queue.
    /// Execution high water remains advanced, so the rejected ID cannot replay.
    pub fn release_unadmitted(
        &mut self,
        rejection: &CpuAdmissionRejection,
    ) -> Result<(), CpuWorkerError> {
        let account = self
            .jobs
            .get_mut(&rejection.grant.job)
            .ok_or(CpuWorkerError::UnknownJob)?;
        if account.quarantine {
            return Err(CpuWorkerError::Quarantined);
        }
        if account.reservation.as_ref() != Some(&rejection.grant) {
            return Err(CpuWorkerError::Duplicate);
        }
        account.last_settled_sequence = rejection.grant.execution.sequence;
        account.last_control_sequence = rejection.grant.control_sequence;
        account.reservation = None;
        if account.last_started_sequence == 0 {
            account.owner = None;
        }
        Ok(())
    }
    /// Abandon a queued-but-never-started question. An admitted reservation or
    /// an owned continuation requires its worker's actual terminal receipt.
    pub fn retire_unstarted(&mut self, job: CpuJobId) -> Result<(), CpuWorkerError> {
        let account = self.jobs.get_mut(&job).ok_or(CpuWorkerError::UnknownJob)?;
        if account.quarantine {
            return Err(CpuWorkerError::Quarantined);
        }
        if account.terminal {
            return Err(CpuWorkerError::Duplicate);
        }
        if account.reservation.is_some() {
            return Err(CpuWorkerError::AlreadyReserved);
        }
        if account.last_started_sequence != 0 || account.owner.is_some() || account.spent != 0 {
            return Err(CpuWorkerError::Invalid(
                "CPU started Job needs owner termination",
            ));
        }
        account.terminal = true;
        Ok(())
    }
    pub fn accept_job_termination(
        &mut self,
        termination: &CpuJobTermination,
    ) -> Result<(), CpuWorkerError> {
        let account = self
            .jobs
            .get_mut(&termination.job)
            .ok_or(CpuWorkerError::UnknownJob)?;
        if account.quarantine {
            return Err(CpuWorkerError::Quarantined);
        }
        if account.terminal {
            return Err(CpuWorkerError::Duplicate);
        }
        if account.owner != Some(termination.owner)
            || account.last_started_sequence != termination.last_execution.sequence
            || termination.last_execution.epoch != self.authority.epoch
            || termination.retained_bytes != 0
            || !termination.controls.terminal
            || !matches!(
                termination.completion,
                CpuCompletion::Canceled | CpuCompletion::Deadline
            )
            || (termination.completion == CpuCompletion::Canceled
                && termination.controls.cancel_ack == 0
                && !termination.shutdown)
        {
            return Err(CpuWorkerError::Invalid(
                "CPU paused termination is not attested",
            ));
        }
        account.terminal = true;
        Ok(())
    }
    pub fn global_usage(&self) -> Result<(u64, u64), CpuWorkerError> {
        let outstanding = self
            .jobs
            .values()
            .try_fold(0_u64, |sum, account| {
                sum.checked_add(
                    account
                        .reservation
                        .as_ref()
                        .map_or(0, |grant| grant.node_credit),
                )
            })
            .ok_or(CpuWorkerError::Invalid(
                "CPU outstanding reservations overflow",
            ))?;
        Ok((self.global_spent, outstanding))
    }
    pub fn quarantine(&mut self, job: CpuJobId) -> Result<(), CpuWorkerError> {
        self.jobs
            .get_mut(&job)
            .ok_or(CpuWorkerError::UnknownJob)?
            .quarantine = true;
        Ok(())
    }
    pub fn usage(&self, job: CpuJobId) -> Result<(u64, u64, bool), CpuWorkerError> {
        let account = self.jobs.get(&job).ok_or(CpuWorkerError::UnknownJob)?;
        Ok((
            account.spent,
            account
                .reservation
                .as_ref()
                .map_or(0, |grant| grant.node_credit),
            account.quarantine,
        ))
    }
}

enum CpuCommand {
    Begin(CpuJob, SliceGrant),
    Resume(SliceGrant),
}
struct ActiveJob {
    job: CpuJob,
    token: Option<CpuResumeToken>,
    spent: u64,
    last_sequence: u64,
    last_control_sequence: u64,
    terminal_permit: Option<ReliablePermit<CpuCommand, CpuWorkerEvent, CpuWorkerProgress>>,
}
#[derive(Clone, Debug)]
pub struct CpuWorkerProgress {
    pub job: CpuJobId,
    pub execution: ExecutionId,
    pub owner: CpuWorkerId,
    pub sequence: u64,
    /// Coherent iteration result; nodes count only this invocation, not the Job.
    pub iteration: CpuIterationProgress,
}

/// Commands and terminal slots share an admission count of eight. Each command
/// reserves its report seat before enqueueing, so terminal send never depends
/// on an unavailable general queue slot. Progress alone is coalesced.
type CpuWorkerMailbox = WorkerMailbox<CpuCommand, CpuWorkerEvent, CpuWorkerProgress>;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuWorkerProfile {
    pub owner: CpuWorkerId,
    pub max_depth: u16,
    pub evaluation_namespace: Digest,
    pub search_conditions: Digest,
}
pub struct CpuWorker {
    owner: CpuWorkerId,
    max_depth: u16,
    evaluation_namespace: Digest,
    search_conditions: Digest,
    mailbox: CpuWorkerMailbox,
    shutdown: Arc<AtomicBool>,
    ready_readback: Arc<OnceLock<Result<CpuThreadReadback, CpuResourceError>>>,
    pub signal: CompletionSignal,
    join: Option<JoinHandle<()>>,
}

impl CpuWorker {
    pub fn spawn(owner: CpuWorkerId, engine: CpuEngine) -> Result<Self, CpuWorkerError> {
        Self::spawn_notified(owner, engine, CompletionSignal::default())
    }
    /// Pool/native services can share one manager wake without creating a
    /// relay thread. Each Job still has its own immutable control mailbox.
    pub fn spawn_notified(
        owner: CpuWorkerId,
        mut engine: CpuEngine,
        signal: CompletionSignal,
    ) -> Result<Self, CpuWorkerError> {
        if engine.resume_policy() != CpuResumePolicy::PausedStack {
            return Err(CpuWorkerError::Invalid("CPU worker requires PausedStack"));
        }
        let max_depth = engine.config().max_depth;
        let evaluation_namespace = cpu_evaluation_namespace_digest(&engine)?;
        let search_conditions = cpu_search_conditions_digest(&engine);
        let shutdown = Arc::new(AtomicBool::new(false));
        let mailbox = CpuWorkerMailbox::new(
            CPU_WORKER_COMMAND_CAPACITY,
            CPU_WORKER_COMMAND_CAPACITY * 2,
            signal.clone(),
        )
        .map_err(|_| CpuWorkerError::Invalid("CPU mailbox capacity"))?;
        let worker_shutdown = shutdown.clone();
        let worker_mailbox = mailbox.clone();
        let ready_readback = Arc::new(OnceLock::new());
        let worker_readback = ready_readback.clone();
        let join = thread::Builder::new()
            .name(format!("rz-cpu-{}", owner.0))
            .spawn(move || {
                // Observe this actual OS thread once, before accepting search
                // work. A failed probe remains explicit and does not install
                // affinity or promote SharedControl to DedicatedControl.
                let observed = u8::try_from(owner.0)
                    .map_err(|_| CpuResourceError::InvalidDeclaration)
                    .and_then(|id| observe_current_cpu_thread(CpuActor::OwnWorker(id)));
                let _ = worker_readback.set(observed);
                worker_mailbox.signal().notify();
                let mut active: Option<ActiveJob> = None;
                let mut progress_sequence = 0u64;
                loop {
                    let observed = worker_mailbox.signal().version();
                    // A paused owner keeps a reliable terminal seat. Atomic
                    // cancellation is serviced even when normal queues are full.
                    if let Some(task) = active.as_mut() {
                        let snapshot = task.job.mailbox.snapshot();
                        let reason = if snapshot.is_cancelled() {
                            Some(CpuCompletion::Canceled)
                        } else if Instant::now() >= task.job.hard_deadline {
                            Some(CpuCompletion::Deadline)
                        } else if worker_shutdown.load(Ordering::Acquire) {
                            Some(CpuCompletion::Canceled)
                        } else {
                            None
                        };
                        if let Some(completion) = reason {
                            engine.discard_paused_stack();
                            if snapshot.cancel_sequence > snapshot.cancel_ack {
                                let _ = task
                                    .job
                                    .mailbox
                                    .acknowledge_cancel(snapshot.cancel_sequence);
                            }
                            let controls = task.job.mailbox.seal_terminal();
                            if let Some(mut permit) = task.terminal_permit.take() {
                                let termination = CpuJobTermination {
                                    job: task.job.envelope.id,
                                    owner,
                                    last_execution: ExecutionId::new(
                                        task.job.envelope.id.authority.epoch,
                                        task.last_sequence,
                                    ),
                                    completion,
                                    controls,
                                    retained_bytes: engine.paused_stack_bytes().unwrap_or(0),
                                    shutdown: worker_shutdown.load(Ordering::Acquire),
                                };
                                let _ = permit.publish(CpuWorkerEvent::JobTerminated(termination));
                            }
                            active = None;
                        }
                    }
                    let mut command = match worker_mailbox.take_command() {
                        Some(command) => command,
                        None if worker_shutdown.load(Ordering::Acquire) => break,
                        None => {
                            worker_mailbox.wait_changed(observed, Duration::from_millis(10));
                            continue;
                        }
                    };
                    let mut rejected_controls = None;
                    let (grant, accepted_command) = match command.command {
                        CpuCommand::Begin(job, grant) => {
                            let fresh_eligible = !worker_shutdown.load(Ordering::Acquire)
                                && grant.job == job.envelope.id
                                && grant.owner == owner
                                && job.mailbox.scope()
                                    == &CpuMailboxScope {
                                        job: grant.job,
                                        owner,
                                    }
                                && job.envelope.evaluation_namespace
                                    == cpu_evaluation_namespace_digest(&engine)
                                        .unwrap_or(Digest([0; 32]))
                                && job.envelope.search_conditions
                                    == cpu_search_conditions_digest(&engine)
                                && job.target_depth <= engine.config().max_depth
                                && job.clock.now().ok().is_some_and(|now| {
                                    grant
                                        .validate(&job.envelope, job.clock.domain(), now)
                                        .is_ok()
                                });
                            if active.is_none() && fresh_eligible {
                                active = Some(ActiveJob {
                                    job,
                                    token: None,
                                    spent: 0,
                                    last_sequence: 0,
                                    last_control_sequence: 0,
                                    terminal_permit: None,
                                });
                                (grant, true)
                            } else {
                                // This owned Begin never ran. Closing its
                                // question admission is distinct from an
                                // execution terminal or a retained stack ack.
                                // A malformed duplicate must not seal the live
                                // primary when it aliases that exact mailbox.
                                let aliases_primary = active.as_ref().is_some_and(|task| {
                                    Arc::ptr_eq(&task.job.mailbox, &job.mailbox)
                                });
                                if !aliases_primary {
                                    rejected_controls = Some(job.mailbox.seal_terminal());
                                }
                                (grant, false)
                            }
                        }
                        CpuCommand::Resume(grant) => (grant, true),
                    };
                    let mut report = CpuSliceReport {
                        grant: grant.clone(),
                        actual_work: Some(CpuWork::default()),
                        result: Err(CpuError::InvalidLimits(
                            "slice does not name the live private job",
                        )),
                        cumulative_nodes: 0,
                        progress_sequence,
                        continuation_retained: false,
                        terminal: false,
                        admission_rejected: true,
                        controls: rejected_controls,
                    };
                    let eligible = !worker_shutdown.load(Ordering::Acquire)
                        && accepted_command
                        && active.as_ref().is_some_and(|task| {
                            grant.job == task.job.envelope.id
                                && grant.owner == owner
                                && grant.execution.epoch == grant.job.authority.epoch
                                && grant.execution.sequence > task.last_sequence
                                && grant.node_credit > 0
                                && !grant.soft_lease.is_zero()
                                && grant.control_sequence > task.last_control_sequence
                                && task.job.clock.now().ok().is_some_and(|now| {
                                    grant
                                        .validate(&task.job.envelope, task.job.clock.domain(), now)
                                        .is_ok()
                                })
                                && task
                                    .spent
                                    .checked_add(grant.node_credit)
                                    .is_some_and(|sum| {
                                        sum <= task.job.envelope.total_native_node_credit
                                    })
                        });
                    if eligible {
                        let task = active.as_mut().expect("checked private job");
                        // Starting the next slice consumes the previous pause
                        // authority; its unused terminal storage can be released.
                        task.terminal_permit.take();
                        let lease = Instant::now().checked_add(grant.soft_lease);
                        let mut invoked = false;
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            CpuSliceControl::new(grant.node_credit, lease, task.job.mailbox.clone())
                                .and_then(|control| {
                                    invoked = true;
                                    engine.analyze_controlled(
                                        &task.job.position,
                                        task.job.root_moves.as_deref(),
                                        CpuLimits {
                                            max_depth: task.job.target_depth,
                                            max_nodes: task.job.envelope.total_native_node_credit
                                                - task.spent,
                                            deadline: Some(task.job.hard_deadline),
                                        },
                                        &worker_shutdown,
                                        &control,
                                        task.token.as_ref(),
                                        |iteration| {
                                            if iteration.completed_depth == 0 {
                                                return;
                                            }
                                            progress_sequence = progress_sequence.saturating_add(1);
                                            let _ = worker_mailbox.publish_progress(
                                                progress_sequence,
                                                CpuWorkerProgress {
                                                    job: grant.job,
                                                    execution: grant.execution,
                                                    owner,
                                                    sequence: progress_sequence,
                                                    iteration,
                                                },
                                            );
                                        },
                                    )
                                })
                        }));
                        let panicked = result.is_err();
                        let mut result = result.unwrap_or_else(|_| {
                            Err(CpuError::Unsupported("CPU worker panicked; usage unknown"))
                        });
                        // Preserve the existing OwnedCpuChecker post-call
                        // frozen namespace guard in this private execution path.
                        if result.is_ok() {
                            if let Err(error) = engine.validate_frozen_value_namespace() {
                                result = Err(error);
                            }
                        }
                        report.actual_work = if panicked {
                            None
                        } else if invoked {
                            engine.last_attempt_work()
                        } else {
                            Some(CpuWork::default())
                        };
                        if let Some(work) = report.actual_work {
                            if let Some(spent) = task.spent.checked_add(work.nodes) {
                                task.spent = spent;
                            } else {
                                report.actual_work = None;
                                result = Err(CpuError::Unsupported(
                                    "CPU cumulative native work overflow; usage unknown",
                                ));
                            }
                        }
                        report.cumulative_nodes = task.spent;
                        task.last_sequence = grant.execution.sequence;
                        task.last_control_sequence = grant.control_sequence;
                        task.token = result
                            .as_ref()
                            .ok()
                            .filter(|value| value.report.completion == CpuCompletion::Yielded)
                            .and_then(|value| value.report.resume.clone());
                        // Only this private worker retains the opaque engine
                        // owner. The manager receives liveness, not the token.
                        if let Ok(value) = &mut result {
                            value.report.resume = None;
                        }
                        report.continuation_retained = task
                            .token
                            .as_ref()
                            .is_some_and(|token| engine.token_is_current(token));
                        report.terminal = match &result {
                            Ok(value) => value.report.completion != CpuCompletion::Yielded,
                            Err(_) => true,
                        };
                        report.admission_rejected = false;
                        report.result = result;
                        report.progress_sequence = progress_sequence;
                        if report.terminal {
                            engine.discard_paused_stack();
                            let snapshot = task.job.mailbox.snapshot();
                            if snapshot.cancel_sequence > snapshot.cancel_ack {
                                let _ = task
                                    .job
                                    .mailbox
                                    .acknowledge_cancel(snapshot.cancel_sequence);
                            }
                            task.job.mailbox.seal_terminal();
                            report.controls = Some(task.job.mailbox.snapshot());
                            active = None;
                        } else {
                            report.controls = Some(task.job.mailbox.snapshot());
                        }
                        if panicked {
                            worker_shutdown.store(true, Ordering::Release);
                            worker_mailbox.close();
                        }
                    } else if let Some(task) = active
                        .as_ref()
                        .filter(|task| task.job.envelope.id == grant.job)
                    {
                        report.cumulative_nodes = task.spent;
                        if accepted_command && task.last_sequence == 0 {
                            // Shutdown can arrive between installing a fresh
                            // primary and its first invocation. No stack or
                            // work exists, so return the admission rejection
                            // with its actual closed question scope.
                            report.controls = Some(task.job.mailbox.seal_terminal());
                            active = None;
                        }
                    }
                    let retain_permit = !report.admission_rejected && report.continuation_retained;
                    if command
                        .reliable
                        .publish(CpuWorkerEvent::Slice(report))
                        .is_err()
                    {
                        break;
                    }
                    if retain_permit {
                        if let Some(task) = active.as_mut() {
                            task.terminal_permit = Some(command.reliable);
                        }
                    }
                }
                if let Some(task) = active {
                    task.job.mailbox.seal_terminal();
                }
                engine.discard_paused_stack();
                worker_mailbox.signal().notify();
            })
            .map_err(|_| CpuWorkerError::Disconnected)?;
        Ok(Self {
            owner,
            max_depth,
            evaluation_namespace,
            search_conditions,
            shutdown,
            ready_readback,
            mailbox,
            signal,
            join: Some(join),
        })
    }
    /// Build a question from the immutable worker profile without copying its
    /// engine/TT. Controls notify the same owner signal as command admission.
    pub fn prepare_job(
        &self,
        envelope: CpuJobEnvelope,
        position: ContractPosition,
        root_moves: Option<Vec<BoardMove>>,
        target_depth: u16,
        clock: &InstantClock,
    ) -> Result<CpuJob, CpuWorkerError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(CpuWorkerError::Disconnected);
        }
        CpuJob::from_profile(
            envelope,
            position,
            root_moves,
            target_depth,
            clock,
            self.max_depth,
            self.evaluation_namespace,
            self.search_conditions,
            self.owner,
            self.signal.clone(),
        )
    }
    pub fn profile(&self) -> CpuWorkerProfile {
        CpuWorkerProfile {
            owner: self.owner,
            max_depth: self.max_depth,
            evaluation_namespace: self.evaluation_namespace,
            search_conditions: self.search_conditions,
        }
    }
    /// Borrow the single real ready observation without blocking or copying
    /// CPU sets on manager ticks. A declaration is not an observed placement.
    pub fn ready_readback(&self) -> Option<&Result<CpuThreadReadback, CpuResourceError>> {
        self.ready_readback.get()
    }
    fn submit(&self, command: CpuCommand) -> Result<(), CpuSubmissionRejected> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(CpuSubmissionRejected::from_command(
                CpuWorkerError::Disconnected,
                command,
            ));
        }
        self.mailbox.admit(command, 2).map_err(|error| {
            let reason = match error.reason {
                MailboxError::Closed => CpuWorkerError::Disconnected,
                MailboxError::QueueFull | MailboxError::ReliableFull => CpuWorkerError::Capacity,
                _ => CpuWorkerError::Invalid("CPU mailbox admission failed"),
            };
            CpuSubmissionRejected::from_command(reason, error.value)
        })
    }
    pub fn begin(&self, job: CpuJob, grant: SliceGrant) -> Result<(), CpuWorkerError> {
        self.begin_retaining(job, grant)
            .map_err(|error| error.reason)
    }
    /// Preserve the immutable Job and an admission receipt on definite failure.
    pub fn begin_retaining(
        &self,
        job: CpuJob,
        grant: SliceGrant,
    ) -> Result<(), CpuSubmissionRejected> {
        if grant.owner != self.owner
            || grant.job != job.envelope.id
            || job.mailbox.scope()
                != &(CpuMailboxScope {
                    job: grant.job,
                    owner: self.owner,
                })
        {
            return Err(CpuSubmissionRejected::from_command(
                CpuWorkerError::Invalid("CPU job owner mismatch"),
                CpuCommand::Begin(job, grant),
            ));
        }
        self.submit(CpuCommand::Begin(job, grant))
    }
    pub fn resume(&self, grant: SliceGrant) -> Result<(), CpuWorkerError> {
        self.resume_attested(grant).map_err(|error| error.reason)
    }
    pub fn resume_attested(&self, grant: SliceGrant) -> Result<(), CpuSubmissionRejected> {
        self.submit(CpuCommand::Resume(grant))
    }
    pub fn try_event(&self) -> Result<Option<CpuWorkerEvent>, CpuWorkerError> {
        if let Some(report) = self.mailbox.take_reliable() {
            return Ok(Some(report));
        }
        if self.join.as_ref().is_some_and(JoinHandle::is_finished) {
            return Err(CpuWorkerError::Disconnected);
        }
        Ok(None)
    }
    pub fn latest_progress(&self) -> Option<CpuWorkerProgress> {
        self.mailbox.take_progress().map(|(_, progress)| progress)
    }
    pub fn occupancy(&self) -> rz_runtime::adaptive::MailboxOccupancy {
        self.mailbox.occupancy()
    }
    /// Request closure without waiting for CPU work or joining a running owner.
    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.mailbox.close();
        self.signal.notify();
    }
    /// Manager polling joins only an already-finished owner. All reliable
    /// receipts are returned before disposal; None means it is still running.
    /// During closure retain a finite polling tick: the final signal is a wake
    /// hint and can precede JoinHandle's finished flag by a scheduling boundary.
    pub fn try_join(&mut self) -> Result<Option<CpuWorkerShutdown>, CpuWorkerError> {
        let join = self.join.as_ref().ok_or(CpuWorkerError::Duplicate)?;
        if !join.is_finished() {
            return Ok(None);
        }
        let join = self.join.take().expect("finished owner retained");
        Ok(Some(self.collect_joined(join)))
    }
    fn collect_joined(&self, join: JoinHandle<()>) -> CpuWorkerShutdown {
        let thread_panicked = join.join().is_err();
        let mut events = Vec::new();
        while let Some(event) = self.mailbox.take_reliable() {
            events.push(event);
        }
        let last_progress = self.latest_progress();
        CpuWorkerShutdown {
            owner: self.owner,
            events,
            last_progress,
            thread_joined: true,
            thread_panicked,
            mailbox: self.mailbox.occupancy(),
            ready_readback: self.ready_readback().cloned(),
        }
    }
    /// Explicit blocking shutdown for a caller outside the manager event loop.
    pub fn shutdown(mut self) -> Result<CpuWorkerShutdown, CpuWorkerError> {
        self.request_shutdown();
        let join = self.join.take().ok_or(CpuWorkerError::Duplicate)?;
        Ok(self.collect_joined(join))
    }
}
impl Drop for CpuWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.mailbox.close();
        self.signal.notify();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

#[cfg(test)]
#[path = "cpu_worker_tests.rs"]
mod tests;
