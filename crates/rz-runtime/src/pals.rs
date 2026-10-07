//! Typed PALS scheduling without converting role tasks to position EvalOutput.
//!
//! The generic scheduler owns reservations and physical leases. These adapters
//! fence PALS identities at admission, physical completion, and final delivery.
//! A successful delivery still grants no authority to commit search backup.

use crate::contracts::ContractClock;
use crate::{
    Adapter, Backend, CompletionReceiver, DrainState, Limits, Resources, RuntimeFault, Scheduler,
    SchedulerObservations, ShutdownSnapshot, State, TerminalEvent,
};
use rz_contracts::pals::{
    ExecutionMode, RepresentationKey, Role, RoleOutput, RoleRequest, SearchAuthority,
    SituationHandle,
};
use rz_contracts::{
    ContractError, Deadline, Digest, ErrorCode, ExecutionId, MonotonicTick, PrecisionProfile,
    ProcessEpoch, RequestId, Stage, StateIdentity,
};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Deref;
use std::sync::{mpsc, Arc, OnceLock, RwLock};

/// One frozen model configuration selected before admitting role work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsScope {
    pub authority: SearchAuthority,
    pub model: Digest,
    pub encoding: Digest,
    pub precision: PrecisionProfile,
    pub frozen_epoch: u64,
    pub mode: ExecutionMode,
}

impl PalsScope {
    pub fn accepts<P>(&self, request: &RoleRequest<P>) -> bool {
        self.authority == request.authority
            && self.model == request.key.model
            && self.encoding == request.key.encoding
            && self.precision == request.key.precision
            && self.frozen_epoch == request.key.frozen_epoch
            && self.mode == request.mode
            && (self.mode != ExecutionMode::Deployment || request.role != Role::Validator)
    }
}

/// Updated by the search/model owner; updates do not revoke physical pins.
#[derive(Clone, Debug)]
pub struct SharedPalsScope(Arc<RwLock<PalsScope>>);

impl SharedPalsScope {
    pub fn new(scope: PalsScope) -> Self {
        Self(Arc::new(RwLock::new(scope)))
    }
    pub fn get(&self) -> Option<PalsScope> {
        self.0.read().ok().map(|scope| *scope)
    }
    pub fn update(&self, scope: PalsScope) -> Result<(), ContractError> {
        *self.0.write().map_err(|_| poisoned_scope())? = scope;
        Ok(())
    }
}

/// Situation slot generation and public-record revision remain independent of
/// the root generation. Reusing a slot or repairing its inputs revokes old work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepresentationScope {
    pub situation: SituationHandle,
    pub state: StateIdentity,
    pub key: RepresentationKey,
}

#[derive(Clone, Debug)]
pub struct SharedRepresentationScope(Arc<RwLock<RepresentationScope>>);

impl SharedRepresentationScope {
    pub fn new(scope: RepresentationScope) -> Self {
        Self(Arc::new(RwLock::new(scope)))
    }
    pub fn get(&self) -> Option<RepresentationScope> {
        self.0.read().ok().map(|scope| *scope)
    }
    pub fn update(&self, scope: RepresentationScope) -> Result<(), ContractError> {
        *self.0.write().map_err(|_| poisoned_scope())? = scope;
        Ok(())
    }
}

/// Immutable common request plus runtime-local physical binding. All clones
/// share the same OnceLock so a request can bind exactly one fresh execution.
pub struct RuntimeRoleRequest<P> {
    role: Arc<RoleRequest<P>>,
    execution: Arc<OnceLock<ExecutionId>>,
    representation: SharedRepresentationScope,
}

impl<P> Clone for RuntimeRoleRequest<P> {
    fn clone(&self) -> Self {
        Self {
            role: Arc::clone(&self.role),
            execution: Arc::clone(&self.execution),
            representation: self.representation.clone(),
        }
    }
}

impl<P> RuntimeRoleRequest<P> {
    /// The situation owner supplies a live scope shared by every task targeting
    /// the same representation; repairs and slot reuse update that scope.
    pub fn new(role: Arc<RoleRequest<P>>, representation: SharedRepresentationScope) -> Self {
        Self {
            role,
            execution: Arc::new(OnceLock::new()),
            representation,
        }
    }
    pub fn with_representation(mut self, scope: SharedRepresentationScope) -> Self {
        self.representation = scope;
        self
    }
    pub fn role(&self) -> &Arc<RoleRequest<P>> {
        &self.role
    }
    pub fn execution(&self) -> Option<ExecutionId> {
        self.execution.get().copied()
    }
    pub fn representation_scope(&self) -> &SharedRepresentationScope {
        &self.representation
    }
    fn representation_is_current(&self) -> bool {
        self.representation.get().is_some_and(|live| {
            live.situation == self.role.situation
                && live.state == self.role.state_identity
                && live.key == self.role.key
        })
    }
    fn context(&self) -> RoleCompletionContext {
        RoleCompletionContext {
            request: self.role.id,
            authority: self.role.authority,
            situation: self.role.situation,
            state: self.role.state_identity,
            key: self.role.key,
            execution: self.execution(),
        }
    }
}

/// RoleOutput's wire contract has no execution ID. The provider must echo the
/// freshly issued physical ID in this local envelope before it is accepted.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalRoleOutput {
    pub execution: ExecutionId,
    pub output: RoleOutput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoleCompletionContext {
    pub request: RequestId,
    pub authority: SearchAuthority,
    pub situation: SituationHandle,
    pub state: StateIdentity,
    pub key: RepresentationKey,
    pub execution: Option<ExecutionId>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PalsTerminal {
    Completed(PhysicalRoleOutput),
    Canceled(RoleCompletionContext),
    Expired(RoleCompletionContext),
    Stale(RoleCompletionContext),
    Failed {
        context: RoleCompletionContext,
        error: ContractError,
    },
}

/// First PALS path does not grant mixed-candidate or mixed-role batching. Every
/// actual input identity and the requested recurrent work must match.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsBatchKey {
    authority: SearchAuthority,
    representation: RepresentationKey,
    mode: ExecutionMode,
    recurrent_steps: u16,
    max_records: u16,
    divergence_count: u16,
    candidates: usize,
    records: usize,
}

/// Fresh IDs use high-water marks rather than an unbounded historical ID set.
pub struct PalsAdapter<P, C> {
    scope: SharedPalsScope,
    clock: C,
    epoch: ProcessEpoch,
    last_request: Option<u64>,
    last_execution: u64,
    _position: PhantomData<fn() -> P>,
}

impl<P, C: ContractClock> PalsAdapter<P, C> {
    pub fn new(scope: SharedPalsScope, clock: C) -> Result<Self, ContractError> {
        Self::with_execution_high_water(scope, clock, 0)
    }
    /// Recovery must register the former owner's high-water mark, never restart
    /// execution sequence zero while old receipts from the same epoch exist.
    pub fn with_execution_high_water(
        scope: SharedPalsScope,
        clock: C,
        last_execution: u64,
    ) -> Result<Self, ContractError> {
        Self::with_high_water(scope, clock, None, last_execution)
    }
    /// A replacement owner in the same process epoch restores both sequence
    /// high-water marks before accepting delayed or replayed logical requests.
    pub fn with_high_water(
        scope: SharedPalsScope,
        clock: C,
        last_request: Option<u64>,
        last_execution: u64,
    ) -> Result<Self, ContractError> {
        if scope
            .get()
            .is_none_or(|scope| scope.authority.epoch != clock.domain().0)
        {
            return Err(identity(
                Stage::Admission,
                "PALS scope has another clock epoch",
            ));
        }
        Ok(Self {
            scope,
            epoch: clock.domain().0,
            clock,
            last_request,
            last_execution,
            _position: PhantomData,
        })
    }
    pub fn scope(&self) -> &SharedPalsScope {
        &self.scope
    }
    fn current(&self, request: &RuntimeRoleRequest<P>) -> bool {
        self.scope
            .get()
            .is_some_and(|scope| scope.accepts(&request.role))
            && request.representation_is_current()
    }
    fn acceptance(&self, request: &RuntimeRoleRequest<P>) -> Result<(), ContractError> {
        request
            .role
            .validate(self.clock.domain(), self.clock.now())?;
        if !self.current(request) {
            return Err(ContractError::new(
                ErrorCode::Stale,
                Stage::Output,
                "PALS authority or representation is no longer current",
            ));
        }
        Ok(())
    }
}

impl<P: Send + Sync + 'static, C: ContractClock> Adapter for PalsAdapter<P, C> {
    type RequestId = RequestId;
    type ExecutionId = ExecutionId;
    type Tick = MonotonicTick;
    type Request = RuntimeRoleRequest<P>;
    type BatchKey = PalsBatchKey;
    type Output = PhysicalRoleOutput;
    type Error = ContractError;
    type TerminalResult = PalsTerminal;

    fn request_id(&self, request: &Self::Request) -> RequestId {
        request.role.id
    }
    fn deadline(&self, request: &Self::Request) -> MonotonicTick {
        request.role.deadline.at
    }
    fn batch_key(&self, request: &Self::Request) -> PalsBatchKey {
        PalsBatchKey {
            authority: request.role.authority,
            representation: request.role.key,
            mode: request.role.mode,
            recurrent_steps: request.role.recurrent_steps,
            max_records: request.role.max_records,
            divergence_count: request.role.divergence_count,
            candidates: request.role.candidates.len(),
            records: request.role.public_records.len(),
        }
    }
    fn resources(&self, request: &Self::Request) -> Resources {
        Resources {
            host_bytes: request.role.resources.host,
            device_bytes: request.role.resources.device,
            pinned_bytes: request.role.resources.pinned,
        }
    }
    fn validate_admission(&mut self, request: &Self::Request) -> Result<(), ContractError> {
        let id = request.role.id;
        if id.epoch != self.epoch || self.last_request.is_some_and(|last| id.sequence <= last) {
            return Err(identity(
                Stage::Admission,
                "PALS request sequence is reused or foreign",
            ));
        }
        // Refused submissions consume their ID as well, matching Scheduler's
        // exactly-once admission contract. A corrected retry uses a fresh ID.
        self.last_request = Some(id.sequence);
        if request.execution().is_some() {
            return Err(identity(
                Stage::Admission,
                "PALS request is already physically bound",
            ));
        }
        self.acceptance(request)?;
        if request.role.resources.host < minimum_output_bytes(&request.role) {
            return Err(ContractError::new(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "PALS host reservation cannot retain the required role output",
            ));
        }
        Ok(())
    }
    fn is_current(&self, request: &Self::Request) -> bool {
        self.current(request)
    }
    fn is_canceled(&self, request: &Self::Request) -> bool {
        request.role.cancel.is_canceled()
    }
    fn validate_output(
        &mut self,
        request: &Self::Request,
        output: &PhysicalRoleOutput,
    ) -> Result<(), ContractError> {
        self.acceptance(request)?;
        if request.execution() != Some(output.execution) {
            return Err(identity(
                Stage::Output,
                "PALS output has another physical execution",
            ));
        }
        output.output.validate_for(&request.role)?;
        if output_allocation_bytes(&output.output)
            .is_none_or(|bytes| bytes > request.role.resources.host)
        {
            return Err(ContractError::new(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "PALS output allocation exceeds its reserved host bytes",
            ));
        }
        Ok(())
    }
    fn bind_execution(&mut self, request: &Self::Request, id: &ExecutionId) {
        request
            .execution
            .set(*id)
            .expect("one physical execution per fresh PALS request");
    }
    fn validate_execution(
        &mut self,
        request: &Self::Request,
        expected: &ExecutionId,
        output: &PhysicalRoleOutput,
    ) -> Result<(), ContractError> {
        if output.execution != *expected {
            return Err(identity(
                Stage::Output,
                "PALS output does not echo launched execution",
            ));
        }
        self.validate_output(request, output)
    }
    fn allocate_execution_id(&mut self) -> Result<ExecutionId, ContractError> {
        self.last_execution = self.last_execution.checked_add(1).ok_or_else(|| {
            ContractError::new(
                ErrorCode::ResourceExhausted,
                Stage::Backend,
                "PALS physical execution sequence exhausted",
            )
        })?;
        Ok(ExecutionId::new(self.epoch, self.last_execution))
    }
    fn runtime_error(&self, fault: RuntimeFault) -> ContractError {
        fault_error(fault)
    }
    fn execution_error(&self, _: &Self::Request, error: &ContractError) -> ContractError {
        *error
    }
    fn terminal(
        &mut self,
        request: &Self::Request,
        event: TerminalEvent<PhysicalRoleOutput, ContractError>,
    ) -> PalsTerminal {
        let context = request.context();
        match event {
            TerminalEvent::Success(output) => PalsTerminal::Completed(output),
            TerminalEvent::Canceled => PalsTerminal::Canceled(context),
            TerminalEvent::Expired => PalsTerminal::Expired(context),
            TerminalEvent::Stale => PalsTerminal::Stale(context),
            TerminalEvent::Failure(error) => failed(context, error),
        }
    }
}

struct Pending<P> {
    request: Arc<RuntimeRoleRequest<P>>,
    receiver: CompletionReceiver<PalsTerminal>,
}

/// Single physical worker, bounded queue and bounded completed-result mailbox.
/// Provider leases obey Backend's physical-completion guarantee. On an unknown
/// completion, Scheduler retains the provider/context/inputs rather than
/// reporting successful drain or permitting reuse.
pub struct PalsRuntime<P, B, C>
where
    P: Send + Sync + 'static,
    C: ContractClock,
    B: Backend<PalsAdapter<P, C>>,
{
    scheduler: Scheduler<PalsAdapter<P, C>, B, C>,
    pending: VecDeque<Pending<P>>,
    scope: SharedPalsScope,
    clock: C,
    max_requests: usize,
    last_request: Option<u64>,
}

impl<P, B, C> PalsRuntime<P, B, C>
where
    P: Send + Sync + 'static,
    C: ContractClock,
    B: Backend<PalsAdapter<P, C>>,
{
    pub fn new(
        adapter: PalsAdapter<P, C>,
        backend: B,
        limits: Limits,
        metric_sample_capacity: usize,
    ) -> Result<Self, ContractError> {
        if limits.max_executions != 1 {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "PALS requires one physical worker and one simultaneous execution",
            ));
        }
        let scope = adapter.scope.clone();
        let clock = adapter.clock.clone();
        let last_request = adapter.last_request;
        let scheduler = Scheduler::new(
            adapter,
            backend,
            clock.clone(),
            limits,
            metric_sample_capacity,
        )
        .map_err(fault_error)?;
        Ok(Self {
            scheduler,
            pending: VecDeque::new(),
            scope,
            clock,
            max_requests: limits.max_requests,
            last_request,
        })
    }
    pub fn submit(&mut self, request: RuntimeRoleRequest<P>) -> Result<(), ContractError> {
        let id = request.role.id;
        if id.epoch != self.clock.domain().0
            || self.last_request.is_some_and(|last| id.sequence <= last)
        {
            return Err(identity(
                Stage::Admission,
                "PALS submitted request ID is reused or foreign",
            ));
        }
        self.last_request = Some(id.sequence);
        if self.pending.len() >= self.max_requests {
            return Err(fault_error(RuntimeFault::QueueFull));
        }
        let request = Arc::new(request);
        let receiver = self
            .scheduler
            .submit(request.as_ref().clone())
            .map_err(|error| error.error)?;
        self.pending.push_back(Pending { request, receiver });
        Ok(())
    }
    /// Validated after receipt transfer, so a completed-but-unread response is
    /// rejected if a repair, new root, deadline or cancellation intervened.
    pub fn poll(&mut self) -> Option<PalsTerminal> {
        self.scheduler.pump();
        for _ in 0..self.pending.len() {
            let pending = self.pending.pop_front().expect("bounded pending count");
            match pending.receiver.try_recv() {
                Ok(PalsTerminal::Completed(output)) => {
                    let checked = pending
                        .request
                        .role
                        .validate(self.clock.domain(), self.clock.now())
                        .and_then(|()| {
                            if !self
                                .scope
                                .get()
                                .is_some_and(|scope| scope.accepts(&pending.request.role))
                                || !pending.request.representation_is_current()
                            {
                                Err(ContractError::new(
                                    ErrorCode::Stale,
                                    Stage::Output,
                                    "PALS delivery has stale authority or representation",
                                ))
                            } else {
                                output.output.validate_for(&pending.request.role)
                            }
                        })
                        .and_then(|()| {
                            if pending.request.execution() == Some(output.execution) {
                                Ok(())
                            } else {
                                Err(identity(Stage::Output, "PALS delivery execution differs"))
                            }
                        });
                    return Some(match checked {
                        Ok(()) => PalsTerminal::Completed(output),
                        Err(error) => failed(pending.request.context(), error),
                    });
                }
                Ok(terminal) => return Some(terminal),
                Err(mpsc::TryRecvError::Empty) => self.pending.push_back(pending),
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Some(failed(
                        pending.request.context(),
                        ContractError::new(
                            ErrorCode::BackendFailure,
                            Stage::Output,
                            "PALS completion mailbox disconnected",
                        ),
                    ))
                }
            }
        }
        None
    }
    pub fn cancel(&mut self, id: RequestId) -> Result<(), ContractError> {
        if id.epoch != self.clock.domain().0 {
            return Err(identity(
                Stage::Admission,
                "PALS cancellation has another epoch",
            ));
        }
        if let Some(pending) = self
            .pending
            .iter()
            .find(|pending| pending.request.role.id == id)
        {
            pending.request.role.cancel.cancel();
        }
        self.scheduler.cancel(&id);
        Ok(())
    }
    pub fn pump(&mut self) {
        self.scheduler.pump();
    }
    pub fn state(&self) -> State {
        self.scheduler.state()
    }
    pub fn begin_shutdown(&mut self, deadline: Deadline) -> Result<(), ContractError> {
        if deadline.clock != self.clock.domain() {
            return Err(identity(
                Stage::Admission,
                "PALS drain deadline has another epoch",
            ));
        }
        self.scheduler.begin_shutdown(deadline.at);
        Ok(())
    }
    pub fn drain_state(&self) -> DrainState {
        self.scheduler.drain_state()
    }
    pub fn shutdown_snapshot(&self) -> ShutdownSnapshot {
        self.scheduler.shutdown_snapshot()
    }
    pub fn take_observations(&mut self) -> SchedulerObservations<PalsAdapter<P, C>> {
        self.scheduler.take_observations()
    }
}

/// A role-neutral encoder must compute every digest from its actual canonical
/// content. There is deliberately no conversion from a role-private key: shared
/// projection weights do not prove identical input/mask/positions or KV data.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RoleNeutralMemoryKey {
    pub input: Digest,
    pub history: Digest,
    pub candidates: Digest,
    pub mask_positions: Digest,
    pub model: Digest,
    pub encoding: Digest,
    pub precision: PrecisionProfile,
    pub frozen_epoch: u64,
    pub record_revision: u64,
}

/// Public pages have a distinct identity for exact whole-input ownership,
/// contextual board encodings, independent records, and lossy summaries.
/// A summary is never an exact-record cache hit even if its bytes match.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PublicPageKind {
    /// Existing exporters produce one owned K/V allocation for the whole input.
    WholeInput,
    /// All 66 board/metadata tokens are contextualized together. Changing any
    /// board input requires a different content identity for this entire page.
    ContextualBoard,
    ImmutableRecord {
        record_id: u64,
        revision: u64,
    },
    Summary {
        source_id: u64,
        revision: u64,
    },
}

/// The encoder supplies a digest of the actual prepared features, masks and
/// positions, not a FEN-only digest or a hash of another role's latent.
/// Generation fencing prevents a game reset from retaining old logical pages.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PublicPageKey {
    pub kind: PublicPageKind,
    pub content: Digest,
    pub model: Digest,
    pub encoding: Digest,
    pub precision: PrecisionProfile,
    pub frozen_epoch: u64,
    pub game_generation: u64,
}

/// Namespace separation prevents a private latent or role-specific KV from
/// aliasing public role-neutral memory, even if its numeric bytes happen to match.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MemoryKey {
    Neutral(RoleNeutralMemoryKey),
    PublicPage(PublicPageKey),
    Private(RepresentationKey),
}

#[derive(Debug)]
struct MemoryValue<T>(T);

/// Exposes a borrow and cloneable pin, never its owner Arc/Weak. This closes a
/// Weak::upgrade race between an LRU eviction decision and physical ownership.
#[derive(Debug)]
pub struct MemoryPin<T> {
    value: Arc<MemoryValue<T>>,
}
impl<T> Clone for MemoryPin<T> {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
        }
    }
}
impl<T> Deref for MemoryPin<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value.0
    }
}
impl<T> AsRef<T> for MemoryPin<T> {
    fn as_ref(&self) -> &T {
        &self.value.0
    }
}

struct MemoryEntry<T> {
    key: MemoryKey,
    value: Arc<MemoryValue<T>>,
    bytes: u64,
}

/// A scheduler-owned finite hot bank. Cloned Arc pins prevent eviction while a
/// request/lease consumes an entry. Raw evidence and CPU task completion records
/// have independent owners and survive this bank's representation eviction.
pub struct MemoryBank<T> {
    entries: VecDeque<MemoryEntry<T>>,
    max_entries: usize,
    max_bytes: u64,
    bytes: u64,
}

/// Host ownership evidence, not a measurement of native allocator or VRAM peak.
/// `pinned_entries` counts each retained allocation once; cloned pins do not
/// multiply its reservation. The caller must charge the complete backing owner,
/// including capacity, rather than a slice view that keeps a larger owner alive.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryBankSnapshot {
    pub entries: usize,
    pub reserved_bytes: u64,
    pub pinned_entries: usize,
    pub pinned_bytes: u64,
    pub max_entries: usize,
    pub max_bytes: u64,
}

impl<T> MemoryBank<T> {
    pub fn new(max_entries: usize, max_bytes: u64) -> Result<Self, ContractError> {
        if max_entries == 0 || max_bytes == 0 {
            return Err(fault_error(RuntimeFault::InvalidLimits));
        }
        Ok(Self {
            entries: VecDeque::new(),
            max_entries,
            max_bytes,
            bytes: 0,
        })
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn reserved_bytes(&self) -> u64 {
        self.bytes
    }
    pub fn snapshot(&self) -> MemoryBankSnapshot {
        let mut result = MemoryBankSnapshot {
            entries: self.entries.len(),
            reserved_bytes: self.bytes,
            max_entries: self.max_entries,
            max_bytes: self.max_bytes,
            ..MemoryBankSnapshot::default()
        };
        for entry in &self.entries {
            if Arc::strong_count(&entry.value) > 1 {
                result.pinned_entries += 1;
                // Every entry was admitted against max_bytes, so their sum
                // cannot overflow while the bank invariant holds.
                result.pinned_bytes += entry.bytes;
            }
        }
        result
    }
    /// A game reset or successful drain retires the bank only after all lease
    /// pins have been released. A refused clear changes no entry or reservation.
    /// Logical cancellation alone never grants permission to release a pin.
    pub fn clear(&mut self) -> Result<(), ContractError> {
        if self
            .entries
            .iter()
            .any(|entry| Arc::strong_count(&entry.value) > 1)
        {
            return Err(ContractError::new(
                ErrorCode::ResourceExhausted,
                Stage::Backend,
                "public memory remains physically pinned",
            ));
        }
        self.entries.clear();
        self.bytes = 0;
        Ok(())
    }
    /// Read-only diagnostics borrow the bank's owner and do not refresh LRU or
    /// create an external pin. Physical invocations must use `acquire` instead.
    pub fn get(&self, key: MemoryKey) -> Option<&T> {
        self.entries
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| &entry.value.0)
    }
    /// Acquiring a pin refreshes LRU. No visits, task completion or new neural
    /// execution are generated by a representation cache lookup.
    pub fn acquire(&mut self, key: MemoryKey) -> Option<MemoryPin<T>> {
        let index = self.entries.iter().position(|entry| entry.key == key)?;
        let entry = self.entries.remove(index)?;
        let value = MemoryPin {
            value: Arc::clone(&entry.value),
        };
        self.entries.push_back(entry);
        Some(value)
    }
    pub fn insert(
        &mut self,
        key: MemoryKey,
        value: T,
        bytes: u64,
    ) -> Result<MemoryPin<T>, ContractError> {
        if bytes == 0 || bytes > self.max_bytes {
            return Err(fault_error(RuntimeFault::ResourceLimit));
        }
        if self.entries.iter().any(|entry| entry.key == key) {
            // Existing exact identities are immutable. Caller acquires that
            // entry, or changes the model/record/input identity for new data.
            return Err(identity(Stage::Admission, "memory identity already exists"));
        }
        // Prove enough unpinned capacity before evicting anything. A refusal
        // preserves every existing entry rather than losing partial useful work.
        let mut future_bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| fault_error(RuntimeFault::ResourceOverflow))?;
        let mut future_len = self
            .entries
            .len()
            .checked_add(1)
            .ok_or_else(|| fault_error(RuntimeFault::ResourceOverflow))?;
        let mut evictions = Vec::new();
        for (index, entry) in self.entries.iter().enumerate() {
            if future_bytes <= self.max_bytes && future_len <= self.max_entries {
                break;
            }
            if Arc::strong_count(&entry.value) == 1 {
                future_bytes -= entry.bytes;
                future_len -= 1;
                evictions.push(index);
            }
        }
        if future_bytes > self.max_bytes || future_len > self.max_entries {
            return Err(fault_error(RuntimeFault::ResourceLimit));
        }
        for index in evictions.into_iter().rev() {
            let removed = self
                .entries
                .remove(index)
                .expect("selected eviction exists");
            self.bytes -= removed.bytes;
        }
        let value = Arc::new(MemoryValue(value));
        self.entries.push_back(MemoryEntry {
            key,
            value: Arc::clone(&value),
            bytes,
        });
        self.bytes += bytes;
        Ok(MemoryPin { value })
    }
}

fn failed(context: RoleCompletionContext, error: ContractError) -> PalsTerminal {
    match error.code {
        ErrorCode::Canceled => PalsTerminal::Canceled(context),
        ErrorCode::Expired => PalsTerminal::Expired(context),
        ErrorCode::Stale => PalsTerminal::Stale(context),
        _ => PalsTerminal::Failed { context, error },
    }
}
/// A necessary floor for the host-owned result, not a complete input/session
/// allocation estimate. Providers also declare every retained prepared buffer
/// and batch workspace through request/additional_resources reservations.
fn minimum_output_bytes<P>(request: &RoleRequest<P>) -> u64 {
    let floats = 16 * 384
        + match request.role {
            Role::Proposer => request.candidates.len(),
            Role::Critic => request.candidates.len() + usize::from(request.divergence_count),
            Role::Validator => 7,
        };
    (size_of::<PhysicalRoleOutput>() + floats * size_of::<f32>()) as u64
}

fn output_allocation_bytes(output: &RoleOutput) -> Option<u64> {
    use rz_contracts::pals::RolePayload;
    let head_capacity = match &output.payload {
        RolePayload::Proposal {
            candidate_logits, ..
        } => candidate_logits.capacity(),
        RolePayload::Counterexample {
            candidate_logits,
            divergence_logits,
            ..
        } => candidate_logits
            .capacity()
            .checked_add(divergence_logits.capacity())?,
        RolePayload::TaskRanking { task_logits } => task_logits.capacity(),
    };
    let bytes = output
        .private_latent
        .capacity()
        .checked_add(head_capacity)?
        .checked_mul(size_of::<f32>())?
        .checked_add(size_of::<PhysicalRoleOutput>())?;
    u64::try_from(bytes).ok()
}
fn identity(stage: Stage, detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::IdentityMismatch, stage, detail)
}
fn poisoned_scope() -> ContractError {
    ContractError::new(
        ErrorCode::BackendFailure,
        Stage::Contract,
        "PALS authority lock poisoned",
    )
}
fn fault_error(fault: RuntimeFault) -> ContractError {
    let (code, stage) = match fault {
        RuntimeFault::Canceled => (ErrorCode::Canceled, Stage::Admission),
        RuntimeFault::Expired | RuntimeFault::QueueAgeExceeded => {
            (ErrorCode::Expired, Stage::Admission)
        }
        RuntimeFault::Stale => (ErrorCode::Stale, Stage::Admission),
        RuntimeFault::DuplicateRequest => (ErrorCode::IdentityMismatch, Stage::Admission),
        RuntimeFault::InvalidCompletion => (ErrorCode::BackendFailure, Stage::Output),
        RuntimeFault::InvalidLimits => (ErrorCode::InvalidInput, Stage::Admission),
        RuntimeFault::Closed => (ErrorCode::BackendUnavailable, Stage::Admission),
        _ => (ErrorCode::ResourceExhausted, Stage::Admission),
    };
    ContractError::new(code, stage, "PALS runtime scheduling fault")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendResult, Clock};
    use rz_contracts::pals::{Move16, RolePayload, PALS_CONTRACT_REVISION};
    use rz_contracts::{
        ByteBudget, CancelToken, ClockDomain, GameGeneration, Move, OwnerId, RootGeneration,
        Square, StateRevision, Wdl,
    };
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::task::Poll;
    use std::time::Duration;

    const EPOCH: ProcessEpoch = ProcessEpoch(31);
    const DEADLINE: u64 = 1_000_000;

    #[derive(Clone, Default)]
    struct ManualClock(Arc<AtomicU64>);
    impl ManualClock {
        fn set(&self, tick: u64) {
            self.0.store(tick, Ordering::SeqCst);
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
    impl ContractClock for ManualClock {
        fn domain(&self) -> ClockDomain {
            ClockDomain(EPOCH)
        }
    }

    struct Input {
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Input {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[derive(Default)]
    struct Control {
        ready: AtomicBool,
        lease_drops: AtomicUsize,
        provider_drops: AtomicUsize,
        dispatches: AtomicUsize,
    }
    #[derive(Clone, Copy)]
    enum OutputMode {
        Valid,
        WrongShape,
        WrongRole,
        ExcessCapacity,
        OldExecution,
        MissingItem,
        DuplicateItem,
        Failure,
    }
    struct ControlledBackend {
        control: Arc<Control>,
        mode: OutputMode,
    }
    impl Drop for ControlledBackend {
        fn drop(&mut self) {
            self.control.provider_drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct Lease {
        requests: Vec<Arc<RuntimeRoleRequest<Input>>>,
        execution: ExecutionId,
        control: Arc<Control>,
    }
    impl Drop for Lease {
        fn drop(&mut self) {
            self.control.lease_drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    type TestAdapter = PalsAdapter<Input, ManualClock>;
    type Runtime = PalsRuntime<Input, ControlledBackend, ManualClock>;
    impl Backend<TestAdapter> for ControlledBackend {
        type Lease = Lease;
        fn additional_resources(&self, _: &[Arc<RuntimeRoleRequest<Input>>]) -> Resources {
            Resources {
                host_bytes: 3,
                device_bytes: 7,
                pinned_bytes: 2,
            }
        }
        fn dispatch(
            &mut self,
            execution: &ExecutionId,
            requests: &[Arc<RuntimeRoleRequest<Input>>],
        ) -> Result<Lease, ContractError> {
            self.control.dispatches.fetch_add(1, Ordering::SeqCst);
            Ok(Lease {
                requests: requests.to_vec(),
                execution: *execution,
                control: Arc::clone(&self.control),
            })
        }
        fn poll(&mut self, lease: &mut Lease) -> Poll<Vec<BackendResult<TestAdapter>>> {
            if !self.control.ready.load(Ordering::SeqCst) {
                return Poll::Pending;
            }
            let mut outputs: Vec<_> = lease
                .requests
                .iter()
                .map(|request| {
                    let role = request.role();
                    let mut output = valid_output(role);
                    match self.mode {
                        OutputMode::WrongShape => output.private_latent.pop().map(|_| ()).unwrap(),
                        OutputMode::WrongRole => {
                            output.payload = RolePayload::TaskRanking {
                                task_logits: vec![0.0],
                            }
                        }
                        OutputMode::ExcessCapacity => {
                            output.private_latent.reserve(100_000);
                        }
                        _ => (),
                    }
                    let execution = match self.mode {
                        OutputMode::OldExecution => {
                            ExecutionId::new(EPOCH, lease.execution.sequence.saturating_sub(1))
                        }
                        _ => lease.execution,
                    };
                    BackendResult {
                        request_id: role.id,
                        output: match self.mode {
                            OutputMode::Failure => Err(ContractError::new(
                                ErrorCode::BackendFailure,
                                Stage::Backend,
                                "fixture physical failure after completion",
                            )),
                            _ => Ok(PhysicalRoleOutput { execution, output }),
                        },
                    }
                })
                .collect();
            match self.mode {
                OutputMode::MissingItem => {
                    outputs.pop();
                }
                OutputMode::DuplicateItem if outputs.len() > 1 => {
                    outputs[1].request_id = outputs[0].request_id;
                }
                _ => (),
            }
            Poll::Ready(outputs)
        }
    }
    fn scope() -> PalsScope {
        PalsScope {
            authority: SearchAuthority {
                epoch: EPOCH,
                game: GameGeneration(4),
                root: RootGeneration(8),
                implementation: Digest([9; 32]),
            },
            model: Digest([1; 32]),
            encoding: Digest([2; 32]),
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 3,
            mode: ExecutionMode::Deployment,
        }
    }
    fn key(role: Role) -> RepresentationKey {
        RepresentationKey {
            input: Digest([3; 32]),
            history: Digest([4; 32]),
            candidates: Digest([5; 32]),
            mask_positions: Digest([6; 32]),
            model: scope().model,
            encoding: scope().encoding,
            precision: scope().precision,
            frozen_epoch: 3,
            record_revision: 7,
            role,
        }
    }
    fn request(sequence: u64, drops: Arc<AtomicUsize>) -> RuntimeRoleRequest<Input> {
        let role = Arc::new(RoleRequest {
            revision: PALS_CONTRACT_REVISION,
            id: RequestId::new(EPOCH, sequence),
            authority: scope().authority,
            situation: SituationHandle {
                slot: 1,
                generation: 4,
            },
            state_identity: StateIdentity {
                owner: OwnerId(8),
                revision: StateRevision(4),
                semantic: Digest([7; 32]),
            },
            state: Arc::new(Input { drops }),
            key: key(Role::Proposer),
            role: Role::Proposer,
            mode: ExecutionMode::Deployment,
            candidates: Arc::from([Move16::encode(
                Move::new(
                    Square::try_new(12).unwrap(),
                    Square::try_new(28).unwrap(),
                    None,
                )
                .unwrap(),
            )]),
            public_records: Arc::from([Digest([8; 32])]),
            max_records: 128,
            divergence_count: 0,
            recurrent_steps: 2,
            deadline: Deadline {
                clock: ClockDomain(EPOCH),
                at: MonotonicTick(DEADLINE),
            },
            cancel: CancelToken::new(),
            resources: ByteBudget {
                host: 65_536,
                device: 100,
                pinned: 20,
            },
        });
        let live = SharedRepresentationScope::new(RepresentationScope {
            situation: role.situation,
            state: role.state_identity,
            key: role.key,
        });
        RuntimeRoleRequest::new(role, live)
    }
    fn valid_output<P>(request: &RoleRequest<P>) -> RoleOutput {
        RoleOutput {
            id: request.id,
            authority: request.authority,
            key: request.key,
            payload: match request.role {
                Role::Proposer => RolePayload::Proposal {
                    candidate_logits: vec![0.25; request.candidates.len()],
                    wdl: Wdl::try_new(0.3, 0.4, 0.3, 1e-4).unwrap(),
                },
                Role::Critic => RolePayload::Counterexample {
                    candidate_logits: vec![0.25; request.candidates.len()],
                    divergence_logits: vec![0.0; usize::from(request.divergence_count)],
                    wdl: Wdl::try_new(0.3, 0.4, 0.3, 1e-4).unwrap(),
                },
                Role::Validator => RolePayload::TaskRanking {
                    task_logits: vec![0.0; 7],
                },
            },
            private_latent: vec![0.0; 16 * 384],
        }
    }
    fn request_as_role(
        sequence: u64,
        role: Role,
        divergence_count: u16,
        mode: ExecutionMode,
    ) -> RuntimeRoleRequest<Input> {
        let mut request = request(sequence, dropped());
        let common = Arc::get_mut(&mut request.role).unwrap();
        common.role = role;
        common.key.role = role;
        common.mode = mode;
        common.divergence_count = divergence_count;
        request
            .representation
            .update(representation(&request))
            .unwrap();
        request
    }
    fn limits() -> Limits {
        Limits {
            max_requests: 4,
            max_batch_items: 2,
            max_executions: 1,
            max_batch_wait: Duration::ZERO,
            max_queue_age: Duration::from_secs(1),
            deadline_reserve: Duration::from_nanos(20),
            memory: Resources {
                host_bytes: 1_048_576,
                device_bytes: 4096,
                pinned_bytes: 4096,
            },
        }
    }
    fn runtime(mode: OutputMode) -> (Runtime, ManualClock, SharedPalsScope, Arc<Control>) {
        let clock = ManualClock::default();
        let shared = SharedPalsScope::new(scope());
        let control = Arc::new(Control::default());
        let adapter = PalsAdapter::new(shared.clone(), clock.clone()).unwrap();
        let runtime = PalsRuntime::new(
            adapter,
            ControlledBackend {
                control: Arc::clone(&control),
                mode,
            },
            limits(),
            32,
        )
        .unwrap();
        (runtime, clock, shared, control)
    }
    fn dropped() -> Arc<AtomicUsize> {
        Arc::new(AtomicUsize::new(0))
    }
    fn representation(request: &RuntimeRoleRequest<Input>) -> RepresentationScope {
        RepresentationScope {
            situation: request.role.situation,
            state: request.role.state_identity,
            key: request.role.key,
        }
    }

    #[test]
    fn delayed_batch_completes_every_request_once_and_releases_inputs() {
        let (mut runtime, _, _, control) = runtime(OutputMode::Valid);
        let drops = dropped();
        runtime.submit(request(1, Arc::clone(&drops))).unwrap();
        runtime.submit(request(2, Arc::clone(&drops))).unwrap();
        assert!(runtime.poll().is_none());
        assert_eq!(runtime.state().executions, 1);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        control.ready.store(true, Ordering::SeqCst);
        assert!(matches!(runtime.poll(), Some(PalsTerminal::Completed(_))));
        assert_eq!(runtime.state().reserved_requests, 1);
        assert!(matches!(runtime.poll(), Some(PalsTerminal::Completed(_))));
        assert!(runtime.poll().is_none());
        assert_eq!(runtime.state().reserved_requests, 0);
        assert_eq!(control.dispatches.load(Ordering::SeqCst), 1);
        assert_eq!(control.lease_drops.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
    #[test]
    fn logical_cancel_retains_reservations_and_buffers_until_physical_ready() {
        let (mut runtime, _, _, control) = runtime(OutputMode::Valid);
        let drops = dropped();
        runtime.submit(request(1, Arc::clone(&drops))).unwrap();
        runtime.pump();
        runtime.cancel(RequestId::new(EPOCH, 1)).unwrap();
        assert!(matches!(runtime.poll(), Some(PalsTerminal::Canceled(_))));
        assert_eq!(runtime.state().logical_requests, 0);
        assert_eq!(runtime.state().reserved_requests, 1);
        assert_eq!(runtime.state().executions, 1);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(control.lease_drops.load(Ordering::SeqCst), 0);
        control.ready.store(true, Ordering::SeqCst);
        assert!(
            runtime.poll().is_none(),
            "late physical result must not become another completed response"
        );
        assert_eq!(runtime.state().reserved_requests, 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(control.lease_drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn root_replacement_discards_late_outputs_but_preserves_physical_pins() {
        let (mut runtime, _, shared, control) = runtime(OutputMode::Valid);
        let drops = dropped();
        runtime.submit(request(1, Arc::clone(&drops))).unwrap();
        runtime.pump();
        shared
            .update(PalsScope {
                authority: SearchAuthority {
                    root: RootGeneration(9),
                    ..scope().authority
                },
                ..scope()
            })
            .unwrap();
        assert!(matches!(runtime.poll(), Some(PalsTerminal::Stale(_))));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        control.ready.store(true, Ordering::SeqCst);
        assert!(runtime.poll().is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn completed_mailbox_is_rechecked_after_repair_or_root_change() {
        for change_root in [false, true] {
            let (mut runtime, _, shared, control) = runtime(OutputMode::Valid);
            let request = request(1, dropped());
            let live = SharedRepresentationScope::new(representation(&request));
            runtime
                .submit(request.with_representation(live.clone()))
                .unwrap();
            runtime.pump();
            control.ready.store(true, Ordering::SeqCst);
            runtime.pump(); // Physically complete, but not delivered to Search.
            assert_eq!(runtime.state().executions, 0);
            if change_root {
                shared
                    .update(PalsScope {
                        authority: SearchAuthority {
                            root: RootGeneration(10),
                            ..scope().authority
                        },
                        ..scope()
                    })
                    .unwrap();
            } else {
                let mut revised = live.get().unwrap();
                revised.key.record_revision += 1;
                live.update(revised).unwrap();
            }
            assert!(matches!(runtime.poll(), Some(PalsTerminal::Stale(_))));
            assert!(runtime.poll().is_none());
        }
    }
    #[test]
    fn completed_mailbox_is_rechecked_for_deadline_and_cancellation() {
        for expire in [false, true] {
            let (mut runtime, clock, _, control) = runtime(OutputMode::Valid);
            runtime.submit(request(1, dropped())).unwrap();
            runtime.pump();
            control.ready.store(true, Ordering::SeqCst);
            runtime.pump();
            if expire {
                clock.set(DEADLINE);
                assert!(matches!(runtime.poll(), Some(PalsTerminal::Expired(_))));
            } else {
                runtime.cancel(RequestId::new(EPOCH, 1)).unwrap();
                assert!(matches!(runtime.poll(), Some(PalsTerminal::Canceled(_))));
            }
        }
    }
    #[test]
    fn wrong_role_shape_or_old_execution_is_not_success() {
        for mode in [
            OutputMode::WrongShape,
            OutputMode::WrongRole,
            OutputMode::OldExecution,
            OutputMode::ExcessCapacity,
        ] {
            let (mut runtime, _, _, control) = runtime(mode);
            runtime.submit(request(1, dropped())).unwrap();
            runtime.pump();
            control.ready.store(true, Ordering::SeqCst);
            assert!(matches!(runtime.poll(), Some(PalsTerminal::Failed { .. })));
            assert_eq!(runtime.state().executions, 0);
            assert!(runtime.poll().is_none());
        }
    }
    #[test]
    fn missing_or_duplicate_batch_item_rejects_all_sibling_successes() {
        for mode in [OutputMode::MissingItem, OutputMode::DuplicateItem] {
            let (mut runtime, _, _, control) = runtime(mode);
            runtime.submit(request(1, dropped())).unwrap();
            runtime.submit(request(2, dropped())).unwrap();
            runtime.pump();
            control.ready.store(true, Ordering::SeqCst);
            for _ in 0..2 {
                assert!(matches!(runtime.poll(), Some(PalsTerminal::Failed { .. })));
            }
            assert!(runtime.poll().is_none());
        }
    }
    #[test]
    fn physical_failure_preserves_error_and_releases_after_ready() {
        let (mut runtime, _, _, control) = runtime(OutputMode::Failure);
        let drops = dropped();
        runtime.submit(request(1, Arc::clone(&drops))).unwrap();
        runtime.pump();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        control.ready.store(true, Ordering::SeqCst);
        let result = runtime.poll().unwrap();
        assert!(matches!(
            result,
            PalsTerminal::Failed {
                error: ContractError {
                    code: ErrorCode::BackendFailure,
                    stage: Stage::Backend,
                    ..
                },
                ..
            }
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn unknown_physical_completion_times_out_and_quarantines_live_owners() {
        let (mut runtime, clock, _, control) = runtime(OutputMode::Valid);
        let drops = dropped();
        runtime.submit(request(1, Arc::clone(&drops))).unwrap();
        runtime.pump();
        runtime
            .begin_shutdown(Deadline {
                clock: ClockDomain(EPOCH),
                at: MonotonicTick(10),
            })
            .unwrap();
        assert!(matches!(runtime.poll(), Some(PalsTerminal::Canceled(_))));
        clock.set(10);
        assert!(matches!(
            runtime.drain_state(),
            DrainState::TimedOut { executions: 1, .. }
        ));
        drop(runtime); // Unknown physical completion cannot safely release pins.
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(control.lease_drops.load(Ordering::SeqCst), 0);
        assert_eq!(control.provider_drops.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn timed_out_owner_can_continue_pumping_until_actual_completion() {
        let (mut runtime, clock, _, control) = runtime(OutputMode::Valid);
        let drops = dropped();
        runtime.submit(request(1, Arc::clone(&drops))).unwrap();
        runtime.pump();
        runtime
            .begin_shutdown(Deadline {
                clock: ClockDomain(EPOCH),
                at: MonotonicTick(10),
            })
            .unwrap();
        assert!(matches!(runtime.poll(), Some(PalsTerminal::Canceled(_))));
        clock.set(10);
        assert!(matches!(runtime.drain_state(), DrainState::TimedOut { .. }));
        control.ready.store(true, Ordering::SeqCst);
        runtime.pump();
        assert_eq!(runtime.drain_state(), DrainState::Drained);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn refused_requests_consume_ids_and_epoch_model_and_v_are_fenced() {
        let (mut runtime, _, _, _) = runtime(OutputMode::Valid);
        let mut invalid = request(1, dropped());
        Arc::get_mut(&mut invalid.role).unwrap().key.model = Digest([99; 32]);
        assert_eq!(runtime.submit(invalid).unwrap_err().code, ErrorCode::Stale);
        assert_eq!(
            runtime.submit(request(1, dropped())).unwrap_err().code,
            ErrorCode::IdentityMismatch
        );
        let mut invalid = request(2, dropped());
        let role = Arc::get_mut(&mut invalid.role).unwrap();
        role.role = Role::Validator;
        role.key.role = Role::Validator;
        assert!(runtime.submit(invalid).is_err());
        let mut invalid = request(3, dropped());
        Arc::get_mut(&mut invalid.role).unwrap().id.epoch = ProcessEpoch(32);
        assert_eq!(
            runtime.submit(invalid).unwrap_err().code,
            ErrorCode::IdentityMismatch
        );
        runtime.submit(request(4, dropped())).unwrap();
    }
    #[test]
    fn execution_sequence_is_fresh_and_cannot_wrap() {
        let shared = SharedPalsScope::new(scope());
        let mut adapter: TestAdapter =
            PalsAdapter::new(shared.clone(), ManualClock::default()).unwrap();
        assert_eq!(
            adapter.allocate_execution_id().unwrap(),
            ExecutionId::new(EPOCH, 1)
        );
        assert_eq!(
            adapter.allocate_execution_id().unwrap(),
            ExecutionId::new(EPOCH, 2)
        );
        let mut adapter: TestAdapter =
            PalsAdapter::with_execution_high_water(shared, ManualClock::default(), u64::MAX)
                .unwrap();
        assert_eq!(
            adapter.allocate_execution_id().unwrap_err().code,
            ErrorCode::ResourceExhausted
        );
    }
    #[test]
    fn latent_output_requires_a_real_host_reservation() {
        let (mut runtime, _, _, control) = runtime(OutputMode::Valid);
        let mut invalid = request(1, dropped());
        Arc::get_mut(&mut invalid.role).unwrap().resources.host = 24_000;
        assert_eq!(
            runtime.submit(invalid).unwrap_err().code,
            ErrorCode::ResourceExhausted
        );
        assert_eq!(control.dispatches.load(Ordering::SeqCst), 0);
        assert_eq!(runtime.state().reserved_requests, 0);
        assert_eq!(
            runtime.submit(request(1, dropped())).unwrap_err().code,
            ErrorCode::IdentityMismatch
        );
    }
    #[test]
    fn critic_batch_and_output_match_the_declared_divergence_count() {
        for divergence_count in [0, 1, 128] {
            let request =
                request_as_role(1, Role::Critic, divergence_count, ExecutionMode::Deployment);
            let mut adapter: TestAdapter =
                PalsAdapter::new(SharedPalsScope::new(scope()), ManualClock::default()).unwrap();
            adapter.validate_admission(&request).unwrap();
            let execution = adapter.allocate_execution_id().unwrap();
            adapter.bind_execution(&request, &execution);
            let mut output = PhysicalRoleOutput {
                execution,
                output: valid_output(request.role()),
            };
            adapter.validate_output(&request, &output).unwrap();
            if let RolePayload::Counterexample {
                divergence_logits, ..
            } = &mut output.output.payload
            {
                divergence_logits.push(0.0);
            }
            assert_eq!(
                adapter.validate_output(&request, &output).unwrap_err().code,
                ErrorCode::NumericalFailure
            );
        }
        let adapter: TestAdapter =
            PalsAdapter::new(SharedPalsScope::new(scope()), ManualClock::default()).unwrap();
        let a = request_as_role(1, Role::Critic, 1, ExecutionMode::Deployment);
        let b = request_as_role(2, Role::Critic, 2, ExecutionMode::Deployment);
        assert_ne!(adapter.batch_key(&a), adapter.batch_key(&b));
        assert_eq!(
            minimum_output_bytes(b.role()) - minimum_output_bytes(a.role()),
            4
        );
    }
    #[test]
    fn validator_has_exactly_seven_task_logits_and_no_divergence_head() {
        let request = request_as_role(1, Role::Validator, 0, ExecutionMode::Collection);
        let mut adapter: TestAdapter = PalsAdapter::new(
            SharedPalsScope::new(PalsScope {
                mode: ExecutionMode::Collection,
                ..scope()
            }),
            ManualClock::default(),
        )
        .unwrap();
        adapter.validate_admission(&request).unwrap();
        let execution = adapter.allocate_execution_id().unwrap();
        adapter.bind_execution(&request, &execution);
        let mut output = PhysicalRoleOutput {
            execution,
            output: valid_output(request.role()),
        };
        adapter.validate_output(&request, &output).unwrap();
        if let RolePayload::TaskRanking { task_logits } = &mut output.output.payload {
            task_logits.pop();
        }
        assert_eq!(
            adapter.validate_output(&request, &output).unwrap_err().code,
            ErrorCode::NumericalFailure
        );
        for (role, mode, divergence_count) in [
            (Role::Proposer, ExecutionMode::Deployment, 1),
            (Role::Validator, ExecutionMode::Collection, 1),
            (Role::Critic, ExecutionMode::Deployment, 129),
        ] {
            let invalid = request_as_role(2, role, divergence_count, mode);
            assert_eq!(
                invalid
                    .role
                    .validate(ClockDomain(EPOCH), MonotonicTick(0))
                    .unwrap_err()
                    .code,
                ErrorCode::InvalidInput
            );
        }
    }
    #[test]
    fn already_executed_request_clone_is_refused_before_a_new_dispatch() {
        let (mut first, _, _, first_control) = runtime(OutputMode::Valid);
        let request = request(1, dropped());
        let clone = request.clone();
        first.submit(request).unwrap();
        first.pump();
        first_control.ready.store(true, Ordering::SeqCst);
        assert!(matches!(first.poll(), Some(PalsTerminal::Completed(_))));
        let (mut second, _, _, second_control) = runtime(OutputMode::Valid);
        assert_eq!(
            second.submit(clone).unwrap_err().code,
            ErrorCode::IdentityMismatch
        );
        assert_eq!(second_control.dispatches.load(Ordering::SeqCst), 0);
        assert_eq!(second.state().reserved_requests, 0);
    }
    #[test]
    fn recovered_owner_restores_request_and_execution_high_water_marks() {
        let clock = ManualClock::default();
        let shared = SharedPalsScope::new(scope());
        let mut adapter: TestAdapter =
            PalsAdapter::with_high_water(shared, clock, Some(9), 12).unwrap();
        assert_eq!(
            adapter
                .validate_admission(&request(9, dropped()))
                .unwrap_err()
                .code,
            ErrorCode::IdentityMismatch
        );
        adapter.validate_admission(&request(10, dropped())).unwrap();
        assert_eq!(
            adapter.allocate_execution_id().unwrap(),
            ExecutionId::new(EPOCH, 13)
        );
    }
    #[test]
    fn situation_aba_generation_and_state_revision_reject_same_root_completion() {
        for change_slot in [false, true] {
            let (mut runtime, _, _, control) = runtime(OutputMode::Valid);
            let request = request(1, dropped());
            let live = request.representation_scope().clone();
            runtime.submit(request).unwrap();
            runtime.pump();
            let mut changed = live.get().unwrap();
            if change_slot {
                changed.situation.generation += 1;
            } else {
                changed.state.revision.0 += 1;
            }
            live.update(changed).unwrap();
            control.ready.store(true, Ordering::SeqCst);
            assert!(matches!(runtime.poll(), Some(PalsTerminal::Stale(_))));
        }
    }
    #[test]
    fn private_roles_and_canonical_masks_do_not_alias_neutral_memory() {
        let neutral = RoleNeutralMemoryKey {
            input: key(Role::Proposer).input,
            history: key(Role::Proposer).history,
            candidates: key(Role::Proposer).candidates,
            mask_positions: key(Role::Proposer).mask_positions,
            model: scope().model,
            encoding: scope().encoding,
            precision: scope().precision,
            frozen_epoch: 3,
            record_revision: 7,
        };
        let mut bank = MemoryBank::new(4, 40).unwrap();
        drop(
            bank.insert(MemoryKey::Neutral(neutral), vec![1], 10)
                .unwrap(),
        );
        drop(
            bank.insert(MemoryKey::Private(key(Role::Proposer)), vec![2], 10)
                .unwrap(),
        );
        drop(
            bank.insert(MemoryKey::Private(key(Role::Critic)), vec![3], 10)
                .unwrap(),
        );
        assert_eq!(
            bank.acquire(MemoryKey::Neutral(neutral)).unwrap().as_ref(),
            &[1]
        );
        assert_eq!(
            bank.acquire(MemoryKey::Private(key(Role::Proposer)))
                .unwrap()
                .as_ref(),
            &[2]
        );
        assert_eq!(
            bank.acquire(MemoryKey::Private(key(Role::Critic)))
                .unwrap()
                .as_ref(),
            &[3]
        );
        assert!(bank
            .acquire(MemoryKey::Neutral(RoleNeutralMemoryKey {
                mask_positions: Digest([44; 32]),
                ..neutral
            }))
            .is_none());
        assert!(bank
            .acquire(MemoryKey::Neutral(RoleNeutralMemoryKey {
                candidates: Digest([45; 32]),
                ..neutral
            }))
            .is_none());
        assert!(bank
            .acquire(MemoryKey::Neutral(RoleNeutralMemoryKey {
                record_revision: 8,
                ..neutral
            }))
            .is_none());
    }
    #[test]
    fn bank_refuses_pinned_eviction_and_reclaims_only_unpinned_representation() {
        let mut bank = MemoryBank::new(1, 10).unwrap();
        let first = MemoryKey::Private(key(Role::Proposer));
        let second = MemoryKey::Private(key(Role::Critic));
        let pin = bank.insert(first, vec![1], 10).unwrap();
        assert_eq!(
            bank.insert(second, vec![2], 10).unwrap_err().code,
            ErrorCode::ResourceExhausted
        );
        assert_eq!(bank.len(), 1);
        assert_eq!(bank.reserved_bytes(), 10);
        drop(pin);
        drop(bank.insert(second, vec![2], 10).unwrap());
        assert!(bank.acquire(first).is_none());
        assert!(bank.acquire(second).is_some());
        assert_eq!(bank.reserved_bytes(), 10);
    }

    fn public_page(kind: PublicPageKind) -> MemoryKey {
        MemoryKey::PublicPage(PublicPageKey {
            kind,
            content: Digest([17; 32]),
            model: scope().model,
            encoding: scope().encoding,
            precision: scope().precision,
            frozen_epoch: 3,
            game_generation: 4,
        })
    }

    #[test]
    fn public_page_reset_waits_for_every_physical_pin_without_partial_eviction() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut bank = MemoryBank::new(2, 30).unwrap();
        drop(
            bank.insert(
                public_page(PublicPageKind::ContextualBoard),
                Input {
                    drops: Arc::clone(&drops),
                },
                10,
            )
            .unwrap(),
        );
        let pin = bank
            .insert(
                public_page(PublicPageKind::WholeInput),
                Input {
                    drops: Arc::clone(&drops),
                },
                20,
            )
            .unwrap();
        let physical_lease_pin = pin.clone();
        assert_eq!(bank.snapshot().pinned_entries, 1);
        assert_eq!(bank.snapshot().pinned_bytes, 20);
        assert_eq!(bank.clear().unwrap_err().stage, Stage::Backend);
        assert_eq!(bank.len(), 2);
        assert_eq!(bank.reserved_bytes(), 30);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        // Retiring the logical request's pin does not retire a CUDA/worker pin.
        drop(pin);
        assert!(bank.clear().is_err());
        drop(physical_lease_pin);
        bank.clear().unwrap();
        assert_eq!(bank.snapshot().entries, 0);
        assert_eq!(bank.snapshot().reserved_bytes, 0);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        bank.clear().unwrap();
    }

    #[test]
    fn public_page_kind_generation_and_actual_content_are_separate_identities() {
        let mut bank = MemoryBank::new(4, 40).unwrap();
        let raw = public_page(PublicPageKind::ImmutableRecord {
            record_id: 8,
            revision: 2,
        });
        drop(bank.insert(raw, vec![1], 10).unwrap());
        assert!(bank
            .acquire(public_page(PublicPageKind::Summary {
                source_id: 8,
                revision: 2,
            }))
            .is_none());
        let MemoryKey::PublicPage(key) = raw else {
            unreachable!()
        };
        for changed in [
            PublicPageKey {
                content: Digest([18; 32]),
                ..key
            },
            PublicPageKey {
                game_generation: 5,
                ..key
            },
            PublicPageKey {
                frozen_epoch: 4,
                ..key
            },
            PublicPageKey {
                kind: PublicPageKind::ImmutableRecord {
                    record_id: 8,
                    revision: 3,
                },
                ..key
            },
        ] {
            assert!(bank.acquire(MemoryKey::PublicPage(changed)).is_none());
        }
        assert_eq!(bank.acquire(raw).unwrap().as_ref(), &[1]);
        assert_eq!(bank.reserved_bytes(), 10);
    }

    #[test]
    fn oversize_public_page_refusal_preserves_existing_page_and_reservation() {
        let mut bank = MemoryBank::new(1, 10).unwrap();
        let key = public_page(PublicPageKind::WholeInput);
        drop(bank.insert(key, vec![1], 10).unwrap());
        assert_eq!(
            bank.insert(public_page(PublicPageKind::ContextualBoard), vec![2], 11,)
                .unwrap_err()
                .code,
            ErrorCode::ResourceExhausted
        );
        assert_eq!(bank.acquire(key).unwrap().as_ref(), &[1]);
        assert_eq!(bank.snapshot().reserved_bytes, 10);
    }
}
