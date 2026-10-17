//! Exclusive archive maintenance. The manager lends the complete owner, never
//! a cloned Store, and cannot perform semantic commits until its checked return.
//! Commands and completion notifications each have one slot. Cancellation is
//! separate from that queue; a timed-out owner is quarantined, never recycled.

use super::{AdaptiveEnginePort, PalsEngine, PalsError, RoleModel};
use crate::pals::store::{
    ArchiveIoBudget, ArchiveOwnerSnapshot, ArchiveStats, ForeignAppendPermit, ForeignGoIssuedProof,
    Observation, ProfiledForeignAppendPermit, ProfiledForeignGoIssuedProof, SituationId,
    StoreError, StorePins, StoreReadOwner,
};
use rz_contracts::pals::SearchAuthority;
use rz_position::Position;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MAX_IO_BYTES: u64 = 16 * 1024 * 1024;

/// Before construction, root maintenance may compact the engine arena. An
/// active port retains pending node references and permits Store-only reclaim.
// This is the one exclusive engine/port transfer in a one-slot worker, not a
// collection of variant values. Preserve rejected/quarantined owners by value
// without adding a heap allocation at a pressure-recovery boundary.
#[allow(clippy::large_enum_variant)]
pub enum ArchiveOwner<M: RoleModel> {
    Engine(PalsEngine<M>),
    Port(AdaptiveEnginePort<M>),
}
impl<M: RoleModel> ArchiveOwner<M> {
    fn engine(&self) -> &PalsEngine<M> {
        match self {
            Self::Engine(engine) => engine,
            Self::Port(port) => &port.engine,
        }
    }
    fn engine_mut(&mut self) -> &mut PalsEngine<M> {
        match self {
            Self::Engine(engine) => engine,
            Self::Port(port) => &mut port.engine,
        }
    }
}

// One command slot owns the original observation and its one-use proof. Keep
// them together by value; no additional box/clone is needed per archive dispatch.
#[allow(clippy::large_enum_variant)]
pub enum ArchiveOperation {
    /// Existing root preparation, including checked topology persistence and
    /// compaction. Only an Engine owner may use this pre-construction barrier.
    PrepareRoot { position: Position },
    /// Existing cold load and metadata verification, without model/CPU work.
    LoadRoot { position: Position },
    /// Stable numeric engine indices, including two pending immutable tickets.
    ReclaimHot { incoming_pins: StorePins },
    /// Read-only cold uniqueness scan. The exact raw append is reserved by an
    /// opaque one-use permit and remains a manager hot commit after return.
    PrepareForeign {
        observation: Observation,
        issued: ForeignGoIssuedProof,
    },
    /// Separate profiled query namespace; no legacy job or completed-depth
    /// evidence is manufactured when the original go omitted a limit.
    PrepareProfiledForeign {
        observation: Observation,
        issued: ProfiledForeignGoIssuedProof,
    },
    /// Official lifecycle reset on the exclusive worker. This also supports an
    /// explicitly declared archive-off profile; it never changes archive mode.
    NewGame { next_authority: SearchAuthority },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveOperationKind {
    PrepareRoot,
    LoadRoot,
    ReclaimHot,
    PrepareForeign,
    NewGame,
    PrepareProfiledForeign,
}
impl ArchiveOperation {
    fn kind(&self) -> ArchiveOperationKind {
        match self {
            Self::PrepareRoot { .. } => ArchiveOperationKind::PrepareRoot,
            Self::LoadRoot { .. } => ArchiveOperationKind::LoadRoot,
            Self::ReclaimHot { .. } => ArchiveOperationKind::ReclaimHot,
            Self::PrepareForeign { .. } => ArchiveOperationKind::PrepareForeign,
            Self::PrepareProfiledForeign { .. } => ArchiveOperationKind::PrepareProfiledForeign,
            Self::NewGame { .. } => ArchiveOperationKind::NewGame,
        }
    }
}
pub struct ArchiveWork {
    pub operation: ArchiveOperation,
    pub deadline: Instant,
    pub max_io_bytes: u64,
}

/// A small source-only change stamp for the manager's no-progress latch. It
/// includes counts and authority, never a cloned node/record/statistics graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveMaintenanceStamp {
    pub store_generation: u64,
    pub public_revision: u64,
    pub hot: ArchiveStats,
}
pub struct ArchiveMaintenanceProposal {
    pub incoming_pins: StorePins,
    pub max_io_bytes: u64,
    pub deadline: Instant,
    pub stamp: ArchiveMaintenanceStamp,
}
impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Actual remaining aggregate search allowance, not another 16MiB grant.
    pub fn archive_remaining_io(&self) -> u64 {
        self.engine.archive_remaining_io()
    }
    /// Call before a semantic commit, not on every manager statistics tick.
    /// The no-pressure path reads counters only. Active ports cannot compact
    /// their engine arena; the worker later builds the full resident pin union.
    pub fn archive_maintenance_proposal(
        &self,
    ) -> Result<Option<ArchiveMaintenanceProposal>, PalsError> {
        if !self.engine.archive_enabled() || !self.engine.semantic.stores.archive_pressure() {
            return Ok(None);
        }
        self.engine.ensure_archive_admission_open()?;
        if Instant::now() >= self.limits.deadline {
            return Err(StoreError::ArchiveDeadline.into());
        }
        let max_io_bytes = self.archive_remaining_io().min(MAX_IO_BYTES);
        if max_io_bytes == 0 {
            return Err(StoreError::ArchiveByteBudget.into());
        }
        let stores = &self.engine.semantic.stores;
        let root = self
            .engine
            .semantic
            .nodes
            .get(self.root)
            .ok_or(StoreError::InvalidHandle("adaptive archive root"))?;
        let situation = stores.situations.get(root.situation)?;
        if situation.state != root.state || stores.root() != Some(root.situation) {
            return Err(StoreError::StaleConsumer.into());
        }
        let mut pins = stores.review_pin_union()?;
        pins.states.insert(root.state);
        pins.situations.insert(root.situation);
        pins.lines.insert(situation.focus);
        merge_pins(&mut pins, &self.goal_local_protected_pins()?);
        merge_pins(&mut pins, &self.profiled_external_protected_pins()?);
        for admission in self.cpu_admissions.values() {
            if let Some(execution) = admission.execution {
                pins.executions.insert(execution);
                pins.states.insert(stores.tasks.get(execution)?.key.state);
            }
        }
        validate_pins(stores, &pins)?;
        Ok(Some(ArchiveMaintenanceProposal {
            incoming_pins: pins,
            max_io_bytes,
            deadline: self.limits.deadline,
            stamp: ArchiveMaintenanceStamp {
                store_generation: stores.generation(),
                public_revision: self.engine.semantic.revision,
                hot: stores.hot_stats(),
            },
        }))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveWorkerError {
    Busy,
    StaleTransfer,
    Closed,
    CompletionUnknown,
    SequenceExhausted,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveShutdownReceipt {
    pub admission_closed: bool,
    pub worker_joined: bool,
    /// Every accepted transfer's actual owner was received before joining.
    pub owners_returned: bool,
}
/// Rejection does not consume ownership; it returns the exact submitted value.
pub struct ArchiveRejected<M: RoleModel> {
    pub owner: ArchiveOwner<M>,
    pub error: PalsError,
}
/// A transfer is opaque and belongs to one worker. Cancel remains reachable
/// even while its bounded command queue is occupied.
pub struct ArchiveTransfer {
    nonce: u64,
    worker: Arc<()>,
    cancel: Arc<AtomicBool>,
}
impl ArchiveTransfer {
    pub fn nonce(&self) -> u64 {
        self.nonce
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
}

struct ArchiveSeal {
    authority: SearchAuthority,
    store_owner: StoreReadOwner,
    store_generation: u64,
    game_generation: Option<u64>,
    root: Option<SituationId>,
    revision: u64,
    node_handles: Vec<(crate::pals::store::StateId, SituationId)>,
    record_revisions: Vec<u64>,
    review_pins: StorePins,
    active_cpu_pins: StorePins,
    archive_owner: Option<ArchiveOwnerSnapshot>,
    registry_owner: Weak<()>,
    next_game: Option<SearchAuthority>,
    is_port: bool,
}

#[derive(Clone, Debug)]
pub struct ArchiveWorkReceipt {
    pub nonce: u64,
    pub authority: SearchAuthority,
    pub operation: ArchiveOperationKind,
    pub generation_before: u64,
    pub generation_after: u64,
    pub hot_before: ArchiveStats,
    pub hot_after: ArchiveStats,
    pub io_bytes: Option<u64>,
    pub elapsed: Duration,
    pub loaded_root: Option<usize>,
    pub archive_owner: Option<ArchiveOwnerSnapshot>,
    pub next_authority: Option<SearchAuthority>,
}
#[derive(Clone, Debug)]
pub enum ArchiveQuarantineReason {
    Operation(PalsError),
    Panic,
    Canceled,
    Deadline,
    InvalidReturn(&'static str),
}
/// Failed maintenance still owns all sessions, Store facts, pins and tokens.
/// Only explicit shutdown/recovery may take it; it is not an accepted return.
pub struct ArchiveQuarantine<M: RoleModel> {
    pub reason: ArchiveQuarantineReason,
    pub receipt: ArchiveWorkReceipt,
    owner: ArchiveOwner<M>,
    foreign_permit: Option<ForeignAppendPermit>,
    profiled_foreign_permit: Option<ProfiledForeignAppendPermit>,
}
impl<M: RoleModel> ArchiveQuarantine<M> {
    pub fn retains_foreign_permit(&self) -> bool {
        self.foreign_permit.is_some()
    }
    pub fn retains_profiled_foreign_permit(&self) -> bool {
        self.profiled_foreign_permit.is_some()
    }
    pub fn into_owner_for_shutdown(self) -> ArchiveOwner<M> {
        self.owner
    }
}

pub struct ArchiveCompletion<M: RoleModel> {
    owner: ArchiveOwner<M>,
    seal: ArchiveSeal,
    receipt: ArchiveWorkReceipt,
    result: Result<(), ArchiveQuarantineReason>,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
    foreign_permit: Option<ForeignAppendPermit>,
    profiled_foreign_permit: Option<ProfiledForeignAppendPermit>,
}
impl<M: RoleModel> ArchiveCompletion<M> {
    pub fn receipt(&self) -> &ArchiveWorkReceipt {
        &self.receipt
    }
    /// No clock/root authority is replaced by the worker's submitted copy.
    /// A late result is useful evidence but cannot resume semantic admission.
    // Invalid returns retain every engine/session/pin in the original owner;
    // quarantine must not depend on allocating a new boxed error.
    #[allow(clippy::result_large_err)]
    pub fn install(
        mut self,
        expected_authority: SearchAuthority,
        expected_nonce: u64,
    ) -> Result<(ArchiveOwner<M>, ArchiveWorkReceipt), ArchiveQuarantine<M>> {
        if self.receipt.operation == ArchiveOperationKind::PrepareForeign {
            self.result = Err(ArchiveQuarantineReason::InvalidReturn(
                "foreign scan requires install_with_foreign",
            ));
        }
        if self.result.is_ok()
            && self.receipt.operation == ArchiveOperationKind::PrepareProfiledForeign
        {
            self.result = Err(ArchiveQuarantineReason::InvalidReturn(
                "profiled foreign scan requires install_with_profiled_foreign",
            ));
        }
        self.install_checked(expected_authority, expected_nonce)
            .map(|(owner, receipt, _, _)| (owner, receipt))
    }
    // Same by-value quarantine as install, including the one-use foreign proof.
    #[allow(clippy::result_large_err)]
    pub fn install_with_foreign(
        mut self,
        expected_authority: SearchAuthority,
        expected_nonce: u64,
    ) -> Result<(ArchiveOwner<M>, ArchiveWorkReceipt, ForeignAppendPermit), ArchiveQuarantine<M>>
    {
        if self.receipt.operation != ArchiveOperationKind::PrepareForeign
            || self.foreign_permit.is_none()
        {
            self.result = Err(ArchiveQuarantineReason::InvalidReturn(
                "foreign permit missing",
            ));
        }
        self.install_checked(expected_authority, expected_nonce)
            .map(|(owner, receipt, permit, _)| {
                (owner, receipt, permit.expect("checked foreign permit"))
            })
    }

    /// Only this path returns the exact profiled permit. A wrong operation,
    /// late return or invalid owner keeps it in the original quarantine.
    #[allow(clippy::result_large_err)]
    pub fn install_with_profiled_foreign(
        mut self,
        expected_authority: SearchAuthority,
        expected_nonce: u64,
    ) -> Result<
        (
            ArchiveOwner<M>,
            ArchiveWorkReceipt,
            ProfiledForeignAppendPermit,
        ),
        ArchiveQuarantine<M>,
    > {
        if self.result.is_ok()
            && (self.receipt.operation != ArchiveOperationKind::PrepareProfiledForeign
                || self.profiled_foreign_permit.is_none())
        {
            self.result = Err(ArchiveQuarantineReason::InvalidReturn(
                "profiled foreign permit missing",
            ));
        }
        self.install_checked(expected_authority, expected_nonce)
            .map(|(owner, receipt, _, permit)| {
                (
                    owner,
                    receipt,
                    permit.expect("checked profiled foreign permit"),
                )
            })
    }
    // All install paths share this allocation-free owner rejection.
    #[allow(clippy::result_large_err)]
    fn install_checked(
        self,
        expected_authority: SearchAuthority,
        expected_nonce: u64,
    ) -> Result<
        (
            ArchiveOwner<M>,
            ArchiveWorkReceipt,
            Option<ForeignAppendPermit>,
            Option<ProfiledForeignAppendPermit>,
        ),
        ArchiveQuarantine<M>,
    > {
        let checked = self.result.clone().and_then(|()| {
            if expected_nonce != self.receipt.nonce || expected_authority != self.seal.authority {
                return Err(ArchiveQuarantineReason::InvalidReturn("transfer authority"));
            }
            if self.cancel.load(Ordering::Acquire) {
                return Err(ArchiveQuarantineReason::Canceled);
            }
            if Instant::now() >= self.deadline {
                return Err(ArchiveQuarantineReason::Deadline);
            }
            self.validate_return()
        });
        match checked {
            Ok(()) => Ok((
                self.owner,
                self.receipt,
                self.foreign_permit,
                self.profiled_foreign_permit,
            )),
            Err(reason) => Err(ArchiveQuarantine {
                reason,
                receipt: self.receipt,
                owner: self.owner,
                foreign_permit: self.foreign_permit,
                profiled_foreign_permit: self.profiled_foreign_permit,
            }),
        }
    }
    fn validate_return(&self) -> Result<(), ArchiveQuarantineReason> {
        let invalid = ArchiveQuarantineReason::InvalidReturn;
        let permit_shape_matches = match self.receipt.operation {
            ArchiveOperationKind::PrepareForeign => {
                self.foreign_permit.is_some() && self.profiled_foreign_permit.is_none()
            }
            ArchiveOperationKind::PrepareProfiledForeign => {
                self.profiled_foreign_permit.is_some() && self.foreign_permit.is_none()
            }
            _ => self.foreign_permit.is_none() && self.profiled_foreign_permit.is_none(),
        };
        if !permit_shape_matches {
            return Err(invalid("archive typed permit differs from operation"));
        }
        let engine = self.owner.engine();
        if self.receipt.operation == ArchiveOperationKind::NewGame {
            let next = self
                .seal
                .next_game
                .ok_or_else(|| invalid("newgame target missing"))?;
            if self.seal.is_port
                || !matches!(self.owner, ArchiveOwner::Engine(_))
                || !self
                    .seal
                    .registry_owner
                    .ptr_eq(&engine.adaptive_registry_identity())
                || engine.adaptive_new_game_scope() != Some(next)
                || engine.semantic.game_generation != Some(next.game.0)
                || engine.semantic.stores.generation() != 0
                || engine.semantic.stores.root().is_some()
                || self.seal.store_owner.matches(&engine.semantic.stores)
                || engine
                    .semantic
                    .stores
                    .tasks
                    .active_states()
                    .next()
                    .is_some()
                || engine
                    .semantic
                    .stores
                    .review_pin_count()
                    .map_err(|error| ArchiveQuarantineReason::Operation(error.into()))?
                    != 0
                || !engine.semantic.nodes.is_empty()
                || !engine.semantic.records.is_empty()
            {
                return Err(invalid("official newgame owner transition"));
            }
            let actual = engine
                .archive_owner_snapshot()
                .map_err(ArchiveQuarantineReason::Operation)?;
            match (self.seal.archive_owner, actual) {
                (Some(old), Some(new)) => {
                    let previous = engine
                        .semantic
                        .archive_state
                        .previous_closed_owner
                        .ok_or_else(|| invalid("previous archive close missing"))?;
                    if previous.owner_id != old.owner_id
                        || previous.root_sha256 != old.root_sha256
                        || previous.repository_sha256 != old.repository_sha256
                        || !previous.admission_closed
                        || new.owner_id == old.owner_id
                        || new.admission_closed
                    {
                        return Err(invalid("newgame archive owner handoff"));
                    }
                }
                (None, None) => {}
                _ => return Err(invalid("newgame changed declared archive mode")),
            }
            return Ok(());
        }
        if !self.seal.store_owner.matches(&engine.semantic.stores)
            || engine.semantic.game_generation != self.seal.game_generation
            || engine.semantic.revision != self.seal.revision
            || matches!(self.owner, ArchiveOwner::Port(_)) != self.seal.is_port
        {
            return Err(invalid("store owner, game or semantic revision"));
        }
        let generation = engine.semantic.stores.generation();
        if generation != self.seal.store_generation {
            // Root preparation alone may retire the previous exact root after
            // its attested metadata commit; it does not admit the next root.
            let retired = self.receipt.operation == ArchiveOperationKind::PrepareRoot
                && self.seal.root.is_some()
                && engine.semantic.stores.root().is_none()
                && self.seal.store_generation.checked_add(1) == Some(generation)
                && engine.last_engine_archive().is_some_and(|receipt| {
                    receipt
                        .actual_root()
                        .is_some_and(|(root, _, old_generation)| {
                            Some(root) == self.seal.root
                                && old_generation == self.seal.store_generation
                        })
                });
            if !retired {
                return Err(invalid("store generation"));
            }
        } else if engine.semantic.stores.root() != self.seal.root {
            return Err(invalid("store root"));
        }
        let actual = engine
            .archive_owner_snapshot()
            .map_err(ArchiveQuarantineReason::Operation)?
            .ok_or_else(|| invalid("archive owner missing"))?;
        let expected = self
            .seal
            .archive_owner
            .ok_or_else(|| invalid("submitted archive owner missing"))?;
        if actual.owner_id != expected.owner_id
            || actual.root_sha256 != expected.root_sha256
            || actual.repository_sha256 != expected.repository_sha256
            || actual.admission_closed
        {
            return Err(invalid("archive persistence owner"));
        }
        validate_pins(&engine.semantic.stores, &self.seal.review_pins)
            .map_err(|_| invalid("submitted review pins"))?;
        validate_pins(&engine.semantic.stores, &self.seal.active_cpu_pins)
            .map_err(|_| invalid("worker CPU task pins"))?;
        if matches!(
            self.receipt.operation,
            ArchiveOperationKind::ReclaimHot
                | ArchiveOperationKind::PrepareForeign
                | ArchiveOperationKind::PrepareProfiledForeign
        ) && (engine.semantic.nodes.len() != self.seal.node_handles.len()
            || engine
                .semantic
                .nodes
                .iter()
                .map(|node| (node.state, node.situation))
                .ne(self.seal.node_handles.iter().copied())
            || engine
                .semantic
                .records
                .iter()
                .map(|record| record.revision)
                .ne(self.seal.record_revisions.iter().copied()))
        {
            return Err(invalid("active engine topology"));
        }
        if let ArchiveOwner::Port(port) = &self.owner {
            if port.authority != self.seal.authority {
                return Err(invalid("active port authority"));
            }
        }
        Ok(())
    }
}

struct Command<M: RoleModel> {
    owner: ArchiveOwner<M>,
    seal: ArchiveSeal,
    nonce: u64,
    work: ArchiveWork,
    cancel: Arc<AtomicBool>,
}
pub struct ArchiveWorker<M: RoleModel + 'static> {
    identity: Arc<()>,
    command: Option<SyncSender<Command<M>>>,
    completion: Receiver<u64>,
    join: Option<JoinHandle<()>>,
    orphan: Arc<Mutex<Option<ArchiveCompletion<M>>>>,
    next_nonce: u64,
    pending: Option<(u64, Arc<AtomicBool>)>,
    shutdown_receipt: Option<ArchiveShutdownReceipt>,
    shutdown_unknown: bool,
}
impl<M: RoleModel + 'static> ArchiveWorker<M> {
    pub fn start() -> std::io::Result<Self> {
        Self::start_inner(None)
    }
    /// Connect the caller's existing completion signal without introducing a
    /// runtime dependency or dispatching any semantic work in this callback.
    pub fn start_with_notifier(
        notifier: impl Fn() + Send + Sync + 'static,
    ) -> std::io::Result<Self> {
        Self::start_inner(Some(Arc::new(notifier)))
    }
    fn start_inner(notifier: Option<Arc<dyn Fn() + Send + Sync>>) -> std::io::Result<Self> {
        let (command, receive_command) = mpsc::sync_channel::<Command<M>>(1);
        let (send_completion, completion) = mpsc::sync_channel(1);
        let orphan: Arc<Mutex<Option<ArchiveCompletion<M>>>> = Arc::new(Mutex::new(None));
        let thread_orphan = Arc::clone(&orphan);
        let join = thread::Builder::new()
            .name("rz-pals-archive-owner".into())
            .spawn(move || {
                while let Ok(command) = receive_command.recv() {
                    let completed = execute(command);
                    let nonce = completed.receipt.nonce;
                    // Ownership always resides in the shared one-owner slot.
                    // The channel carries only notification, so dropping its
                    // receiver cannot destroy a queued session/Store owner.
                    *thread_orphan.lock().unwrap_or_else(|p| p.into_inner()) = Some(completed);
                    if send_completion.send(nonce).is_err() {
                        break;
                    }
                    if let Some(notifier) = &notifier {
                        notifier();
                    }
                }
                // Wake the existing manager signal on idle admission close as
                // well. The manager still checks is_finished before joining;
                // this notification itself is not a completion attestation.
                if let Some(notifier) = &notifier {
                    notifier();
                }
            })?;
        Ok(Self {
            identity: Arc::new(()),
            command: Some(command),
            completion,
            join: Some(join),
            orphan,
            next_nonce: 1,
            pending: None,
            shutdown_receipt: None,
            shutdown_unknown: false,
        })
    }
    pub fn is_busy(&self) -> bool {
        self.pending.is_some()
    }
    // Busy, closed and failed sends must return the exact original owner. A
    // second heap allocation on rejection would undermine bounded recovery.
    #[allow(clippy::result_large_err)]
    pub fn submit(
        &mut self,
        owner: ArchiveOwner<M>,
        authority: SearchAuthority,
        work: ArchiveWork,
    ) -> Result<ArchiveTransfer, ArchiveRejected<M>> {
        let reject = |owner, detail| ArchiveRejected {
            owner,
            error: PalsError::Store(StoreError::InvalidConditions(detail)),
        };
        if self.pending.is_some() {
            return Err(reject(owner, "archive worker busy"));
        }
        let Some(command) = &self.command else {
            return Err(reject(owner, "archive worker closed"));
        };
        let seal = match seal_owner(&owner, authority, &work) {
            Ok(seal) => seal,
            Err(error) => return Err(ArchiveRejected { owner, error }),
        };
        let nonce = self.next_nonce;
        let Some(next) = nonce.checked_add(1) else {
            return Err(reject(owner, "archive transfer sequence exhausted"));
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let message = Command {
            owner,
            seal,
            nonce,
            work,
            cancel: Arc::clone(&cancel),
        };
        match command.try_send(message) {
            Ok(()) => {
                self.next_nonce = next;
                self.pending = Some((nonce, Arc::clone(&cancel)));
                Ok(ArchiveTransfer {
                    nonce,
                    worker: Arc::clone(&self.identity),
                    cancel,
                })
            }
            Err(TrySendError::Full(message)) => Err(reject(message.owner, "archive command full")),
            Err(TrySendError::Disconnected(message)) => {
                Err(reject(message.owner, "archive worker unavailable"))
            }
        }
    }
    /// Nonblocking manager poll. mpsc supplies the existing bounded completion
    /// notification, with no detached ownership or manager tick sleep.
    pub fn try_receive(
        &mut self,
        transfer: &ArchiveTransfer,
    ) -> Result<Option<ArchiveCompletion<M>>, ArchiveWorkerError> {
        if !Arc::ptr_eq(&transfer.worker, &self.identity)
            || self.pending.as_ref().map(|p| p.0) != Some(transfer.nonce)
        {
            return Err(ArchiveWorkerError::StaleTransfer);
        }
        match self.completion.try_recv() {
            Ok(nonce) => {
                if nonce != transfer.nonce {
                    return Err(ArchiveWorkerError::CompletionUnknown);
                }
                let completion = self
                    .orphan
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .take()
                    .ok_or(ArchiveWorkerError::CompletionUnknown)?;
                self.pending = None;
                Ok(Some(completion))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(ArchiveWorkerError::CompletionUnknown),
        }
    }
    /// Admission close and cooperative cancel only. Neither dropping the last
    /// sender nor setting cancel proves that archive I/O physically completed.
    pub fn request_shutdown(&mut self) {
        self.command.take();
        if let Some((_, cancel)) = &self.pending {
            cancel.store(true, Ordering::Release);
        }
    }
    /// A manager-safe poll: unfinished worker or unreturned owner yields None.
    /// join is called only after JoinHandle reports the thread actually ended.
    pub fn try_join(&mut self) -> Result<Option<ArchiveShutdownReceipt>, ArchiveWorkerError> {
        if self.shutdown_unknown {
            return Err(ArchiveWorkerError::CompletionUnknown);
        }
        if let Some(receipt) = self.shutdown_receipt {
            return Ok(Some(receipt));
        }
        if self.pending.is_some() || self.command.is_some() {
            return Ok(None);
        }
        let Some(join) = &self.join else {
            return Err(ArchiveWorkerError::CompletionUnknown);
        };
        if !join.is_finished() {
            return Ok(None);
        }
        let joined = self
            .join
            .take()
            .expect("checked archive worker handle")
            .join();
        if joined.is_err() {
            self.shutdown_unknown = true;
            return Err(ArchiveWorkerError::CompletionUnknown);
        }
        let receipt = ArchiveShutdownReceipt {
            admission_closed: true,
            worker_joined: true,
            owners_returned: true,
        };
        self.shutdown_receipt = Some(receipt);
        Ok(Some(receipt))
    }
    /// Blocking helper for tests/off-manager callers after ownership return.
    /// The manager uses request_shutdown/try_join; an idle thread may still need
    /// scheduling before this join completes. A pending owner is never joined.
    pub fn finish(&mut self) -> Result<(), ArchiveWorkerError> {
        if self.pending.is_some() {
            return Err(ArchiveWorkerError::Busy);
        }
        self.request_shutdown();
        if self.shutdown_unknown {
            return Err(ArchiveWorkerError::CompletionUnknown);
        }
        if self.join.take().is_some_and(|join| join.join().is_err()) {
            self.shutdown_unknown = true;
            return Err(ArchiveWorkerError::CompletionUnknown);
        }
        self.shutdown_receipt = Some(ArchiveShutdownReceipt {
            admission_closed: true,
            worker_joined: true,
            owners_returned: true,
        });
        Ok(())
    }
}
impl<M: RoleModel + 'static> Drop for ArchiveWorker<M> {
    fn drop(&mut self) {
        if let Some((_, cancel)) = &self.pending {
            cancel.store(true, Ordering::Release);
            // Dropping an unknown owner must not drop physical resources. This
            // bounded one-owner quarantine is deliberate and process-scoped;
            // normal callers receive/drain then finish and never take this path.
            std::mem::forget(Arc::clone(&self.orphan));
        }
        self.command.take();
    }
}

fn seal_owner<M: RoleModel>(
    owner: &ArchiveOwner<M>,
    authority: SearchAuthority,
    work: &ArchiveWork,
) -> Result<ArchiveSeal, PalsError> {
    let engine = owner.engine();
    if engine.semantic.game_generation != Some(authority.game.0)
        || engine.semantic.stores.generation() != authority.root.0
    {
        return Err(StoreError::StaleConsumer.into());
    }
    if work.max_io_bytes == 0 || work.max_io_bytes > MAX_IO_BYTES {
        return Err(StoreError::InvalidConditions("archive worker I/O allowance").into());
    }
    if Instant::now() >= work.deadline {
        return Err(StoreError::ArchiveDeadline.into());
    }
    if let ArchiveOperation::PrepareForeign { issued, .. } = &work.operation {
        if work.deadline > issued.retain_until() {
            return Err(StoreError::ArchiveDeadline.into());
        }
    }
    if let ArchiveOperation::PrepareProfiledForeign { issued, .. } = &work.operation {
        if work.deadline > issued.retain_until() {
            return Err(StoreError::ArchiveDeadline.into());
        }
    }
    if let ArchiveOwner::Port(port) = owner {
        if port.authority != authority || work.deadline > port.limits.deadline {
            return Err(StoreError::StaleConsumer.into());
        }
        if !matches!(
            work.operation,
            ArchiveOperation::ReclaimHot { .. }
                | ArchiveOperation::PrepareForeign { .. }
                | ArchiveOperation::PrepareProfiledForeign { .. }
        ) {
            return Err(
                StoreError::InvalidConditions("active port cannot compact engine nodes").into(),
            );
        }
        if work.max_io_bytes > engine.archive_remaining_io() {
            return Err(StoreError::ArchiveByteBudget.into());
        }
    }
    if matches!(work.operation, ArchiveOperation::PrepareRoot { .. })
        && work.max_io_bytes != MAX_IO_BYTES
    {
        return Err(StoreError::InvalidConditions(
            "root preparation uses registered 16MiB allowance",
        )
        .into());
    }
    if matches!(
        work.operation,
        ArchiveOperation::PrepareRoot { .. }
            | ArchiveOperation::LoadRoot { .. }
            | ArchiveOperation::NewGame { .. }
    ) && engine
        .semantic
        .stores
        .tasks
        .active_states()
        .next()
        .is_some()
    {
        return Err(
            StoreError::InvalidConditions("root maintenance requires drained CPU tasks").into(),
        );
    }
    engine.ensure_archive_admission_open()?;
    let archive_owner = engine.archive_owner_snapshot()?;
    if !matches!(work.operation, ArchiveOperation::NewGame { .. })
        && (archive_owner.is_none() || !engine.archive_enabled())
    {
        return Err(StoreError::ArchiveDisabled.into());
    }
    if archive_owner.is_some_and(|owner| owner.admission_closed) {
        return Err(StoreError::ArchiveDisabled.into());
    }
    let next_game = if let ArchiveOperation::NewGame { next_authority } = work.operation {
        if engine.adaptive_archive_authority(next_authority)? != authority {
            return Err(StoreError::StaleConsumer.into());
        }
        Some(next_authority)
    } else {
        None
    };
    if let ArchiveOperation::ReclaimHot { incoming_pins } = &work.operation {
        validate_pins(&engine.semantic.stores, incoming_pins)?;
    }
    let mut active_cpu_pins = StorePins::default();
    if let ArchiveOwner::Port(port) = owner {
        merge_pins(&mut active_cpu_pins, &port.goal_local_protected_pins()?);
        merge_pins(
            &mut active_cpu_pins,
            &port.profiled_external_protected_pins()?,
        );
        for admission in port.cpu_admissions.values() {
            active_cpu_pins.executions.extend(admission.execution);
            if let Some(execution) = admission.execution {
                active_cpu_pins
                    .states
                    .insert(engine.semantic.stores.tasks.get(execution)?.key.state);
            }
        }
        validate_pins(&engine.semantic.stores, &active_cpu_pins)?;
    }
    Ok(ArchiveSeal {
        authority,
        store_owner: engine.semantic.stores.read_owner(),
        store_generation: engine.semantic.stores.generation(),
        game_generation: engine.semantic.game_generation,
        root: engine.semantic.stores.root(),
        revision: engine.semantic.revision,
        node_handles: engine
            .semantic
            .nodes
            .iter()
            .map(|node| (node.state, node.situation))
            .collect(),
        record_revisions: engine
            .semantic
            .records
            .iter()
            .map(|record| record.revision)
            .collect(),
        review_pins: engine.semantic.stores.review_pin_union()?,
        active_cpu_pins,
        archive_owner,
        registry_owner: engine.adaptive_registry_identity(),
        next_game,
        is_port: matches!(owner, ArchiveOwner::Port(_)),
    })
}

fn validate_pins(
    stores: &crate::pals::store::PalsStores,
    pins: &StorePins,
) -> Result<(), StoreError> {
    for &id in &pins.states {
        stores.states.get(id)?;
    }
    for &id in &pins.lines {
        stores.lines.get(id)?;
    }
    for &id in &pins.situations {
        stores.situations.get(id)?;
    }
    for &id in &pins.observations {
        stores.observations.get(id)?;
    }
    for &id in &pins.executions {
        stores.tasks.get(id)?;
    }
    Ok(())
}
fn merge_pins(target: &mut StorePins, additional: &StorePins) {
    target.states.extend(&additional.states);
    target.lines.extend(&additional.lines);
    target.situations.extend(&additional.situations);
    target.observations.extend(&additional.observations);
    target.executions.extend(&additional.executions);
}

fn catch_archive_work(
    operation: impl FnOnce() -> Result<(), ArchiveQuarantineReason>,
) -> Result<(), ArchiveQuarantineReason> {
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or(Err(ArchiveQuarantineReason::Panic))
}

fn execute<M: RoleModel>(mut command: Command<M>) -> ArchiveCompletion<M> {
    let began = Instant::now();
    let before = command.owner.engine().stores().hot_stats();
    let mut budget = ArchiveIoBudget {
        deadline: command.work.deadline,
        max_bytes: command.work.max_io_bytes,
    };
    let kind = command.work.operation.kind();
    let operation = command.work.operation;
    let mut loaded_root = None;
    let mut foreign_permit = None;
    let mut profiled_foreign_permit = None;
    let result = catch_archive_work(|| {
        if command.cancel.load(Ordering::Acquire) {
            return Err(ArchiveQuarantineReason::Canceled);
        }
        if Instant::now() >= command.work.deadline {
            return Err(ArchiveQuarantineReason::Deadline);
        }
        let engine = command.owner.engine_mut();
        let operation: Result<(), PalsError> = (|| match operation {
            ArchiveOperation::PrepareRoot { position } => {
                let result =
                    engine.prepare_archive_root(&position, command.work.deadline, &command.cancel);
                budget.max_bytes = engine.archive_remaining_io();
                result
            }
            ArchiveOperation::LoadRoot { position } => engine
                .load_archived_root_with_budget(&position, &mut budget)
                .map(|root| {
                    loaded_root = Some(root);
                }),
            ArchiveOperation::ReclaimHot { incoming_pins } => {
                let mut pins = engine.resident_archive_pins()?;
                merge_pins(&mut pins, &incoming_pins);
                // Independent review leases are already protected by Store;
                // adding the submission union also covers release during I/O.
                merge_pins(&mut pins, &command.seal.review_pins);
                merge_pins(&mut pins, &command.seal.active_cpu_pins);
                engine.semantic.stores.set_archive_pins(pins)?;
                let initial = budget.max_bytes;
                let result = engine
                    .semantic
                    .stores
                    .archive_inactive_with_budget(&mut budget);
                if command.seal.is_port {
                    engine.charge_engine_archive_io(initial.saturating_sub(budget.max_bytes))?;
                }
                result?;
                Ok(())
            }
            ArchiveOperation::PrepareForeign {
                observation,
                issued,
            } => {
                let initial = budget.max_bytes;
                let prepared = engine.semantic.stores.prepare_foreign_append_permit(
                    observation,
                    issued,
                    &mut budget,
                );
                if command.seal.is_port {
                    engine.charge_engine_archive_io(initial.saturating_sub(budget.max_bytes))?;
                }
                foreign_permit = Some(prepared?);
                Ok(())
            }
            ArchiveOperation::PrepareProfiledForeign {
                observation,
                issued,
            } => {
                let initial = budget.max_bytes;
                let prepared = engine
                    .semantic
                    .stores
                    .prepare_profiled_foreign_append_permit(observation, issued, &mut budget)
                    .map(|permit| {
                        // Preserve the actual scan result even if subsequent I/O
                        // accounting fails and this owner must be quarantined.
                        profiled_foreign_permit = Some(permit);
                    });
                if command.seal.is_port {
                    engine.charge_engine_archive_io(initial.saturating_sub(budget.max_bytes))?;
                }
                prepared?;
                Ok(())
            }
            ArchiveOperation::NewGame { next_authority } => {
                engine.try_adaptive_new_game(next_authority, command.work.deadline, &command.cancel)
            }
        })();
        operation.map_err(ArchiveQuarantineReason::Operation)?;
        if command.cancel.load(Ordering::Acquire) {
            return Err(ArchiveQuarantineReason::Canceled);
        }
        if Instant::now() >= command.work.deadline {
            return Err(ArchiveQuarantineReason::Deadline);
        }
        Ok(())
    });
    let engine = command.owner.engine();
    let receipt = ArchiveWorkReceipt {
        nonce: command.nonce,
        authority: command.seal.authority,
        operation: kind,
        generation_before: command.seal.store_generation,
        generation_after: engine.semantic.stores.generation(),
        hot_before: before,
        hot_after: engine.semantic.stores.hot_stats(),
        io_bytes: if matches!(result, Err(ArchiveQuarantineReason::Panic))
            || (kind == ArchiveOperationKind::PrepareRoot && result.is_err())
            || kind == ArchiveOperationKind::NewGame
        {
            None
        } else {
            command.work.max_io_bytes.checked_sub(budget.max_bytes)
        },
        elapsed: began.elapsed(),
        loaded_root,
        archive_owner: engine.archive_owner_snapshot().ok().flatten(),
        next_authority: command.seal.next_game,
    };
    ArchiveCompletion {
        owner: command.owner,
        seal: command.seal,
        receipt,
        result,
        deadline: command.work.deadline,
        cancel: command.cancel,
        foreign_permit,
        profiled_foreign_permit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuConfig;
    use crate::pals::engine::{LegalOrderRoleMock, PalsConfig};
    use crate::pals::store::{ArchiveConfig, PalsStores, StoreLimits};
    use rz_contracts::{Digest, GameGeneration, ProcessEpoch, RootGeneration};
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;

    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            // This exact unique test directory was created by this fixture.
            // No cache, source, foreign generation or report root is traversed.
            if self.0.file_name().is_some_and(|name| {
                name.to_string_lossy()
                    .starts_with("adaptive-archive-worker-")
            }) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }
    fn fixture(archive: bool) -> (PalsEngine<LegalOrderRoleMock>, Scratch) {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let output = if cfg!(windows) {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("RUNNER_TEMP").map(PathBuf::from)
        }
        .unwrap_or_else(std::env::temp_dir);
        let root = output.join("RoveZero").join("tmp").join(format!(
            "adaptive-archive-worker-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        let mut engine = PalsEngine::new(
            PalsConfig {
                beam_width: 1,
                line_plies: 2,
                max_nodes: 257,
                max_records: 32,
                max_role_calls: 64,
                cpu_nodes_per_task: 256,
            },
            LegalOrderRoleMock,
            crate::cpu::CpuEngine::new(CpuConfig {
                tt_entries: 64,
                max_depth: 4,
                quiescence_ply: 4,
                ..CpuConfig::default()
            })
            .unwrap(),
        )
        .unwrap();
        engine.semantic.stores = PalsStores::new(StoreLimits {
            states: 8,
            situations: 8,
            line_chunks: 16,
            observations: 8,
            ..StoreLimits::default()
        });
        if archive {
            let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_owned();
            engine
                .enable_archive(ArchiveConfig::new(root.clone(), repository))
                .unwrap();
        }
        engine
            .semantic
            .stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        (engine, Scratch(root))
    }
    fn authority(engine: &PalsEngine<LegalOrderRoleMock>) -> SearchAuthority {
        SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(engine.semantic.game_generation.unwrap()),
            root: RootGeneration(engine.semantic.stores.generation()),
            implementation: Digest([7; 32]),
        }
    }
    fn populate(engine: &mut PalsEngine<LegalOrderRoleMock>) -> Vec<crate::pals::store::StateId> {
        (100..105)
            .map(|fullmove| {
                let position = Position::from_fen(&format!(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 {fullmove}"
                ))
                .unwrap();
                let situation = engine
                    .semantic
                    .stores
                    .insert_situation(position.snapshot())
                    .unwrap();
                engine
                    .semantic
                    .stores
                    .situations
                    .get(situation)
                    .unwrap()
                    .state
            })
            .collect()
    }
    fn work(bytes: u64) -> ArchiveWork {
        ArchiveWork {
            operation: ArchiveOperation::ReclaimHot {
                incoming_pins: StorePins::default(),
            },
            deadline: Instant::now() + Duration::from_secs(10),
            max_io_bytes: bytes,
        }
    }
    fn receive(
        worker: &mut ArchiveWorker<LegalOrderRoleMock>,
        transfer: &ArchiveTransfer,
    ) -> ArchiveCompletion<LegalOrderRoleMock> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(completion) = worker.try_receive(transfer).unwrap() {
                return completion;
            }
            assert!(Instant::now() < deadline, "bounded archive test completion");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn actual_archive_reclaim_returns_same_owner_and_preserves_submitted_pins() {
        let (mut engine, scratch) = fixture(true);
        let inactive = populate(&mut engine);
        let original_owner = engine.semantic.stores.read_owner();
        let mut review = StorePins::default();
        review.states.insert(inactive[1]);
        let lease = engine.semantic.stores.pin_review(review).unwrap();
        let mut incoming = StorePins::default();
        incoming.states.insert(inactive[0]);
        let authority = authority(&engine);
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(
                ArchiveOwner::Engine(engine),
                authority,
                ArchiveWork {
                    operation: ArchiveOperation::ReclaimHot {
                        incoming_pins: incoming,
                    },
                    ..work(MAX_IO_BYTES)
                },
            )
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let completion = receive(&mut worker, &transfer);
        let (owner, receipt) = completion
            .install(authority, transfer.nonce())
            .unwrap_or_else(|_| panic!("actual verified archive return rejected"));
        let ArchiveOwner::Engine(engine) = owner else {
            panic!("owner kind changed")
        };
        assert!(original_owner.matches(&engine.semantic.stores));
        assert!(lease.validates(&engine.semantic.stores));
        assert_eq!(receipt.generation_before, receipt.generation_after);
        assert!(receipt.io_bytes.is_some_and(|bytes| bytes > 0));
        assert!(receipt.hot_after.states < receipt.hot_before.states);
        assert!(engine.semantic.stores.states.get(inactive[0]).is_ok());
        assert!(engine.semantic.stores.states.get(inactive[1]).is_ok());
        assert!(matches!(
            engine.semantic.stores.states.get(inactive[2]),
            Err(StoreError::ColdRecord(_))
        ));
        assert_eq!(engine.semantic.revision, 0);
        assert!(engine.semantic.nodes.is_empty() && engine.semantic.records.is_empty());
        drop(lease);
        worker.finish().unwrap();
        drop(engine);
        drop(scratch);
    }

    #[test]
    fn archive_disabled_rejection_returns_original_store_without_fallback() {
        let (engine, scratch) = fixture(false);
        let source = engine.semantic.stores.read_owner();
        let authority = authority(&engine);
        let mut worker = ArchiveWorker::start().unwrap();
        let rejected =
            match worker.submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES)) {
                Err(rejected) => rejected,
                Ok(_) => panic!("disabled archive accepted"),
            };
        assert!(matches!(
            rejected.error,
            PalsError::Store(StoreError::ArchiveDisabled)
        ));
        assert!(source.matches(rejected.owner.engine().stores()));
        assert!(!worker.is_busy());
        worker.finish().unwrap();
        drop(rejected);
        drop(scratch);
    }

    #[test]
    fn failed_io_allowance_preserves_hot_facts_in_quarantine() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let before = engine.semantic.stores.hot_stats();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(1))
            .unwrap_or_else(|_| panic!("registered small allowance rejected"));
        let quarantine = match receive(&mut worker, &transfer).install(authority, transfer.nonce())
        {
            Err(quarantine) => quarantine,
            Ok(_) => panic!("impossible allowance treated as success"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::Operation(_)
        ));
        let owner = quarantine.into_owner_for_shutdown();
        assert!(source.matches(owner.engine().stores()));
        assert_eq!(owner.engine().stores().hot_stats(), before);
        worker.finish().unwrap();
        drop(owner);
        drop(scratch);
    }

    #[test]
    fn ordinary_archive_completion_cannot_issue_a_profiled_permit() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let quarantine = match receive(&mut worker, &transfer)
            .install_with_profiled_foreign(authority, transfer.nonce())
        {
            Err(quarantine) => quarantine,
            Ok(_) => panic!("ordinary archive work issued a profiled permit"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::InvalidReturn("profiled foreign permit missing")
        ));
        assert!(!quarantine.retains_foreign_permit());
        assert!(!quarantine.retains_profiled_foreign_permit());
        assert!(source.matches(quarantine.owner.engine().stores()));
        assert_eq!(
            quarantine.receipt.operation,
            ArchiveOperationKind::ReclaimHot
        );
        worker.finish().unwrap();
        drop(quarantine);
        drop(scratch);
    }

    #[test]
    fn profiled_install_preserves_the_original_archive_failure_metadata() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let before = engine.semantic.stores.hot_stats();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(1))
            .unwrap_or_else(|_| panic!("registered small allowance rejected"));
        let quarantine = match receive(&mut worker, &transfer)
            .install_with_profiled_foreign(authority, transfer.nonce())
        {
            Err(quarantine) => quarantine,
            Ok(_) => panic!("failed archive work issued a profiled permit"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::Operation(_)
        ));
        assert!(source.matches(quarantine.owner.engine().stores()));
        assert_eq!(quarantine.owner.engine().stores().hot_stats(), before);
        assert!(!quarantine.retains_profiled_foreign_permit());
        worker.finish().unwrap();
        drop(quarantine);
        drop(scratch);
    }

    #[test]
    fn wrong_nonce_is_quarantined_with_actual_owner() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let quarantine =
            match receive(&mut worker, &transfer).install(authority, transfer.nonce() + 1) {
                Err(value) => value,
                Ok(_) => panic!("foreign nonce accepted"),
            };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::InvalidReturn("transfer authority")
        ));
        let owner = quarantine.into_owner_for_shutdown();
        assert!(source.matches(owner.engine().stores()));
        worker.finish().unwrap();
        drop(owner);
        drop(scratch);
    }

    #[test]
    fn canceled_after_completion_cannot_reopen_admission() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let completion = receive(&mut worker, &transfer);
        transfer.cancel();
        let quarantine = match completion.install(authority, transfer.nonce()) {
            Err(value) => value,
            Ok(_) => panic!("canceled ownership accepted"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::Canceled
        ));
        assert!(quarantine.receipt.io_bytes.is_some_and(|bytes| bytes > 0));
        worker.finish().unwrap();
        drop(quarantine);
        drop(scratch);
    }

    #[test]
    fn expired_submission_does_not_consume_owner_or_start_work() {
        let (engine, scratch) = fixture(true);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let mut worker = ArchiveWorker::start().unwrap();
        let rejected = match worker.submit(
            ArchiveOwner::Engine(engine),
            authority,
            ArchiveWork {
                deadline: Instant::now(),
                ..work(MAX_IO_BYTES)
            },
        ) {
            Err(value) => value,
            Ok(_) => panic!("expired work submitted"),
        };
        assert!(matches!(
            rejected.error,
            PalsError::Store(StoreError::ArchiveDeadline)
        ));
        assert!(source.matches(rejected.owner.engine().stores()));
        assert!(!worker.is_busy());
        worker.finish().unwrap();
        drop(rejected);
        drop(scratch);
    }

    #[test]
    fn unrelated_worker_cannot_receive_transfer_and_busy_finish_is_nonblocking() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let mut worker = ArchiveWorker::start().unwrap();
        let mut other = ArchiveWorker::<LegalOrderRoleMock>::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        assert!(matches!(
            other.try_receive(&transfer),
            Err(ArchiveWorkerError::StaleTransfer)
        ));
        assert_eq!(worker.finish(), Err(ArchiveWorkerError::Busy));
        let completion = receive(&mut worker, &transfer);
        let (owner, _) = completion
            .install(authority, transfer.nonce())
            .unwrap_or_else(|_| panic!("fixture return rejected"));
        assert!(matches!(
            worker.try_receive(&transfer),
            Err(ArchiveWorkerError::StaleTransfer)
        ));
        worker.finish().unwrap();
        other.finish().unwrap();
        drop(owner);
        drop(scratch);
    }

    #[test]
    fn queued_notification_drop_preserves_actual_owner_in_one_owner_slot() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let mut worker = ArchiveWorker::start().unwrap();
        let retained = Arc::clone(&worker.orphan);
        let join = worker.join.take().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let deadline = Instant::now() + Duration::from_secs(5);
        while retained.lock().unwrap().is_none() {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        // The completion notification can already be queued; it owns no Store.
        drop(worker);
        join.join().unwrap();
        let completed = retained
            .lock()
            .unwrap()
            .take()
            .expect("actual owner retained");
        assert!(source.matches(completed.owner.engine().stores()));
        let quarantine = match completed.install(authority, transfer.nonce()) {
            Err(value) => value,
            Ok(_) => panic!("abandoned transfer accepted"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::Canceled
        ));
        drop(quarantine);
        drop(scratch);
    }

    #[test]
    fn panic_guard_keeps_borrowed_owner_and_hot_facts() {
        let (engine, scratch) = fixture(true);
        let mut owner = ArchiveOwner::Engine(engine);
        let source = owner.engine().stores().read_owner();
        let before = owner.engine().stores().hot_stats();
        let result = catch_archive_work(|| {
            let _exact_borrow = owner.engine_mut();
            panic!("bounded archive failure fixture");
        });
        assert!(matches!(result, Err(ArchiveQuarantineReason::Panic)));
        assert!(source.matches(owner.engine().stores()));
        assert_eq!(owner.engine().stores().hot_stats(), before);
        drop(owner);
        drop(scratch);
    }

    #[test]
    fn existing_notification_signal_is_called_after_completion_publication() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let notified = Arc::new(AtomicU64::new(0));
        let producer = Arc::clone(&notified);
        let mut worker = ArchiveWorker::start_with_notifier(move || {
            producer.fetch_add(1, Ordering::Release);
        })
        .unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let (owner, _) = receive(&mut worker, &transfer)
            .install(authority, transfer.nonce())
            .unwrap_or_else(|_| panic!("fixture return rejected"));
        worker.finish().unwrap();
        assert_eq!(notified.load(Ordering::Acquire), 2);
        drop(owner);
        drop(scratch);
    }

    #[test]
    fn root_maintenance_does_not_retire_worker_owned_paused_task() {
        use crate::pals::store::{TaskAdmission, TaskConsumer, TaskKey, TaskQuestion, TaskStatus};
        let (mut engine, scratch) = fixture(true);
        let root = engine.semantic.stores.root().unwrap();
        let situation = engine.semantic.stores.situations.get(root).unwrap();
        let key = TaskKey {
            state: situation.state,
            line: None,
            question: TaskQuestion::AnalyzePosition,
            root_moves: Vec::new(),
            model: 1,
            epoch: 0,
            value_identity: engine.researcher.cpu_registered_value.clone(),
            checker_identity: Some(engine.researcher.checker_registered_identity.clone()),
            cpu_condition: Some(engine.researcher.cpu_registered_condition.clone()),
            profile: 1,
            condition: 1,
            input_revision: 0,
            requested_depth: 1,
            node_budget: 32,
        };
        let consumer = TaskConsumer {
            id: 1,
            situation: root,
            revision: situation.revision,
            generation: engine.semantic.stores.generation(),
            deadline_tick: 100_000,
        };
        let TaskAdmission::Start(execution) = engine
            .semantic
            .stores
            .request_task(key, consumer, 0)
            .unwrap()
        else {
            panic!("fixture admission")
        };
        engine
            .semantic
            .stores
            .pause_task(execution, 17, None)
            .unwrap();
        let authority = authority(&engine);
        let mut worker = ArchiveWorker::start().unwrap();
        let rejected = match worker.submit(
            ArchiveOwner::Engine(engine),
            authority,
            ArchiveWork {
                operation: ArchiveOperation::PrepareRoot {
                    position: Position::startpos(),
                },
                ..work(MAX_IO_BYTES)
            },
        ) {
            Err(value) => value,
            Ok(_) => panic!("live paused task crossed compaction boundary"),
        };
        assert!(matches!(
            rejected.error,
            PalsError::Store(StoreError::InvalidConditions(
                "root maintenance requires drained CPU tasks"
            ))
        ));
        assert!(matches!(
            rejected
                .owner
                .engine()
                .stores()
                .tasks
                .get(execution)
                .unwrap()
                .status,
            TaskStatus::Paused {
                checkpoint: 17,
                evidence: None
            }
        ));
        worker.finish().unwrap();
        drop(rejected);
        drop(scratch);
    }

    #[test]
    fn shutdown_close_and_join_poll_return_only_real_join_receipt() {
        let mut worker = ArchiveWorker::<LegalOrderRoleMock>::start().unwrap();
        assert_eq!(worker.try_join().unwrap(), None);
        worker.request_shutdown();
        let deadline = Instant::now() + Duration::from_secs(5);
        let receipt = loop {
            if let Some(receipt) = worker.try_join().unwrap() {
                break receipt;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        };
        assert!(receipt.admission_closed && receipt.worker_joined && receipt.owners_returned);
        assert_eq!(worker.try_join().unwrap(), Some(receipt));
        worker.request_shutdown();
        assert_eq!(worker.try_join().unwrap(), Some(receipt));
    }

    #[test]
    fn shutdown_does_not_join_or_discard_unreturned_owner() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        worker.request_shutdown();
        assert_eq!(worker.try_join().unwrap(), None);
        let completion = receive(&mut worker, &transfer);
        let quarantine = match completion.install(authority, transfer.nonce()) {
            Err(value) => value,
            Ok(_) => panic!("shutdown canceled transfer admitted"),
        };
        let owner = quarantine.into_owner_for_shutdown();
        assert!(source.matches(owner.engine().stores()));
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if worker.try_join().unwrap().is_some() {
                break;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        drop(owner);
        drop(scratch);
    }
}
