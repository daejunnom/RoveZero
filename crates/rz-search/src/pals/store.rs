//! Bounded, single-writer PALS situation and evidence storage.
//!
//! Exact Rules history and immutable completed observations outlive derived model
//! caches. A refutation concerns one continuation under its recorded conditions;
//! it never permanently marks the continuation's first move as losing. These
//! stores own no CPU/GPU execution lease: cancelling a consumer does not assert
//! that its physical work has finished.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rz_position::{
    BoardMove, Color, PieceKind, PlayStatus, Position, PositionSnapshot, RepetitionIdentity, Square,
};

macro_rules! index_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(pub usize);
    };
}
index_id!(StateId);
index_id!(LineId);
index_id!(ObservationId);
index_id!(ExecutionId);

/// Reusing a slot issues another generation, so stale IDs cannot name new work.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SituationId {
    pub slot: usize,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StoreError {
    Capacity(&'static str),
    InvalidHandle(&'static str),
    InvalidMove,
    InvalidEvidence(&'static str),
    InvalidConditions(&'static str),
    ExpiredConsumer,
    StaleConsumer,
    CancelledConsumer,
    AlreadyConsumed,
    NotCompleted,
    RevisionExhausted,
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PALS store: {self:?}")
    }
}
impl std::error::Error for StoreError {}

/// Counts include retained completed records. Capacity exhaustion is explicit;
/// completed evidence is never silently evicted to admit another branch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoreLimits {
    pub states: usize,
    pub retained_history_bytes: usize,
    pub line_chunks: usize,
    pub line_plies: usize,
    pub situations: usize,
    pub conclusions_per_situation: usize,
    pub observations: usize,
    pub dependency_edges: usize,
    pub executions: usize,
    pub consumers: usize,
    pub root_moves_per_task: usize,
    pub derived_entries: usize,
    pub derived_bytes: usize,
}

impl Default for StoreLimits {
    fn default() -> Self {
        Self {
            states: 8_192,
            retained_history_bytes: 64 * 1024 * 1024,
            line_chunks: 32_768,
            line_plies: 256,
            situations: 8_192,
            conclusions_per_situation: 256,
            observations: 65_536,
            dependency_edges: 131_072,
            executions: 32_768,
            consumers: 65_536,
            root_moves_per_task: 256,
            derived_entries: 4_096,
            derived_bytes: 8 * 1024 * 1024,
        }
    }
}

/// Coordinate move packing only; Rules still owns legality, castling and EP.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Move16(u16);

impl Move16 {
    pub fn pack(mv: BoardMove) -> Result<Self, StoreError> {
        if mv.from == mv.to {
            return Err(StoreError::InvalidMove);
        }
        let promotion = match mv.promotion {
            None => 0,
            Some(PieceKind::Queen) => 1,
            Some(PieceKind::Rook) => 2,
            Some(PieceKind::Bishop) => 3,
            Some(PieceKind::Knight) => 4,
            _ => return Err(StoreError::InvalidMove),
        };
        Ok(Self(
            mv.from.index() as u16 | ((mv.to.index() as u16) << 6) | (promotion << 12),
        ))
    }

    pub fn unpack(self) -> Result<BoardMove, StoreError> {
        if self.0 & 0x8000 != 0 {
            return Err(StoreError::InvalidMove);
        }
        let promotion = match (self.0 >> 12) & 7 {
            0 => None,
            1 => Some(PieceKind::Queen),
            2 => Some(PieceKind::Rook),
            3 => Some(PieceKind::Bishop),
            4 => Some(PieceKind::Knight),
            _ => return Err(StoreError::InvalidMove),
        };
        BoardMove::new(
            Square::new((self.0 & 63) as u8).map_err(|_| StoreError::InvalidMove)?,
            Square::new(((self.0 >> 6) & 63) as u8).map_err(|_| StoreError::InvalidMove)?,
            promotion,
        )
        .map_err(|_| StoreError::InvalidMove)
    }

    pub fn bits(self) -> u16 {
        self.0
    }
}

#[derive(Debug)]
pub struct StateStore {
    snapshots: Vec<PositionSnapshot>,
    index: HashMap<RepetitionIdentity, Vec<StateId>>,
    limit: usize,
    history_limit: usize,
    history_bytes: usize,
}

impl StateStore {
    pub fn new(limit: usize, retained_history_bytes: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            index: HashMap::new(),
            limit,
            history_limit: retained_history_bytes,
            history_bytes: 0,
        }
    }

    /// The repetition key only selects a bucket. Exact history equality decides
    /// reuse; same FEN with missing/different history receives another StateId.
    pub fn find(&self, snapshot: &PositionSnapshot) -> Option<StateId> {
        self.index
            .get(&snapshot.repetition_identity())?
            .iter()
            .copied()
            .find(|id| self.snapshots[id.0].same_state(snapshot))
    }

    pub fn insert(&mut self, snapshot: PositionSnapshot) -> Result<StateId, StoreError> {
        if let Some(id) = self.find(&snapshot) {
            return Ok(id);
        }
        if self.snapshots.len() >= self.limit {
            return Err(StoreError::Capacity("states"));
        }
        let charge = snapshot
            .retained_history_bytes()
            .ok_or(StoreError::Capacity("state history bytes"))?;
        let new_bytes = self
            .history_bytes
            .checked_add(charge)
            .filter(|bytes| *bytes <= self.history_limit)
            .ok_or(StoreError::Capacity("state history bytes"))?;
        let key = snapshot.repetition_identity();
        let id = StateId(self.snapshots.len());
        self.snapshots.push(snapshot);
        self.index.entry(key).or_default().push(id);
        self.history_bytes = new_bytes;
        Ok(id)
    }

    pub fn get(&self, id: StateId) -> Result<&PositionSnapshot, StoreError> {
        self.snapshots
            .get(id.0)
            .ok_or(StoreError::InvalidHandle("state"))
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    pub fn retained_history_bytes(&self) -> usize {
        self.history_bytes
    }
}

pub const MOVES_PER_CHUNK: usize = 16;

#[derive(Clone, Debug)]
pub struct LineChunk {
    pub parent: Option<LineId>,
    pub start_state: StateId,
    moves: [Move16; MOVES_PER_CHUNK],
    len: usize,
    plies: usize,
}

impl LineChunk {
    pub fn chunk_moves(&self) -> &[Move16] {
        &self.moves[..self.len]
    }

    pub fn plies(&self) -> usize {
        self.plies
    }
}

#[derive(Debug)]
pub struct LinePool {
    chunks: Vec<LineChunk>,
    roots: BTreeMap<StateId, LineId>,
    index: BTreeMap<(LineId, Vec<Move16>), LineId>,
    limit: usize,
    ply_limit: usize,
}

impl LinePool {
    pub fn new(limit: usize, ply_limit: usize) -> Self {
        Self {
            chunks: Vec::new(),
            roots: BTreeMap::new(),
            index: BTreeMap::new(),
            limit,
            ply_limit,
        }
    }

    pub fn root(&mut self, start_state: StateId) -> Result<LineId, StoreError> {
        if let Some(id) = self.roots.get(&start_state) {
            return Ok(*id);
        }
        if self.chunks.len() >= self.limit {
            return Err(StoreError::Capacity("line chunks"));
        }
        let id = LineId(self.chunks.len());
        self.chunks.push(LineChunk {
            parent: None,
            start_state,
            moves: [Move16(0); MOVES_PER_CHUNK],
            len: 0,
            plies: 0,
        });
        self.roots.insert(start_state, id);
        Ok(id)
    }

    /// Each append retains its immutable parent and adds chunks of <=16 moves.
    /// Repeating the same suffix from the same prefix reuses existing chunks.
    /// Preflight prevents a capacity error from inserting a partial suffix.
    pub fn append(&mut self, prefix: LineId, moves: &[BoardMove]) -> Result<LineId, StoreError> {
        let head = self.get(prefix)?;
        let plies = head
            .plies
            .checked_add(moves.len())
            .filter(|plies| *plies <= self.ply_limit)
            .ok_or(StoreError::Capacity("line plies"))?;
        if moves.is_empty() {
            return Ok(prefix);
        }
        let packed = moves
            .iter()
            .copied()
            .map(Move16::pack)
            .collect::<Result<Vec<_>, _>>()?;
        let mut existing = prefix;
        let mut missing = 0;
        let mut diverged = false;
        for chunk in packed.chunks(MOVES_PER_CHUNK) {
            let hit = if diverged {
                None
            } else {
                self.index.get(&(existing, chunk.to_vec()))
            };
            if let Some(id) = hit {
                existing = *id;
            } else {
                diverged = true;
                missing += 1;
            }
        }
        if self.chunks.len().saturating_add(missing) > self.limit {
            return Err(StoreError::Capacity("line chunks"));
        }
        let mut parent = prefix;
        let start_state = self.get(prefix)?.start_state;
        let mut accumulated = self.get(prefix)?.plies;
        for chunk in packed.chunks(MOVES_PER_CHUNK) {
            let key = (parent, chunk.to_vec());
            if let Some(id) = self.index.get(&key) {
                parent = *id;
                accumulated = self.chunks[parent.0].plies;
                continue;
            }
            let mut payload = [Move16(0); MOVES_PER_CHUNK];
            payload[..chunk.len()].copy_from_slice(chunk);
            accumulated += chunk.len();
            let id = LineId(self.chunks.len());
            self.chunks.push(LineChunk {
                parent: Some(parent),
                start_state,
                moves: payload,
                len: chunk.len(),
                plies: accumulated,
            });
            self.index.insert(key, id);
            parent = id;
        }
        debug_assert_eq!(self.get(parent)?.plies, plies);
        Ok(parent)
    }

    pub fn get(&self, id: LineId) -> Result<&LineChunk, StoreError> {
        self.chunks
            .get(id.0)
            .ok_or(StoreError::InvalidHandle("line"))
    }

    pub fn moves(&self, id: LineId) -> Result<Vec<BoardMove>, StoreError> {
        let mut chain = Vec::new();
        let mut at = Some(id);
        while let Some(id) = at {
            let node = self.get(id)?;
            chain.push(id);
            at = node.parent;
        }
        let mut result = Vec::with_capacity(self.get(id)?.plies);
        for id in chain.into_iter().rev() {
            for mv in self.get(id)?.chunk_moves() {
                result.push(mv.unpack()?);
            }
        }
        Ok(result)
    }

    pub fn first_move(&self, id: LineId) -> Result<Option<BoardMove>, StoreError> {
        let mut at = id;
        let mut first = None;
        loop {
            let node = self.get(at)?;
            if let Some(mv) = node.chunk_moves().first() {
                first = Some(mv.unpack()?);
            }
            match node.parent {
                Some(parent) => at = parent,
                None => return Ok(first),
            }
        }
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundKind {
    ExactWithinSearch,
    LowerWithinSearch,
    UpperWithinSearch,
}

/// A finite CPU search result is scoped by depth/profile/conditions. It does not
/// constitute a proof of the complete chess game, even when its TT says exact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceScope {
    RulesTerminal,
    DepthLimited {
        depth: u16,
        profile: u64,
        condition: u64,
    },
    Model {
        model: u64,
        encoding: u64,
        input: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RawScore {
    Unknown,
    Estimate {
        value: f32,
        perspective: Color,
    },
    Cpu {
        value: i32,
        perspective: Color,
        bound: BoundKind,
    },
    Wdl {
        win: f32,
        draw: f32,
        loss: f32,
        perspective: Color,
    },
    Terminal {
        winner: Option<Color>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationKind {
    Proposal,
    Refutation,
    Repair,
    CpuAnalysis,
    ExactTerminal,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub state: StateId,
    pub line: Option<LineId>,
    pub source: u64,
    pub epoch: u64,
    pub scope: EvidenceScope,
    pub score: RawScore,
    pub budget: u64,
    pub kind: ObservationKind,
    pub supersedes: Option<ObservationId>,
    pub execution: Option<ExecutionId>,
}

impl Observation {
    fn validate(&self) -> Result<(), StoreError> {
        match self.score {
            RawScore::Estimate { value, .. } if !value.is_finite() => {
                return Err(StoreError::InvalidEvidence("non-finite estimate"));
            }
            RawScore::Wdl {
                win, draw, loss, ..
            } if ![win, draw, loss]
                .into_iter()
                .all(|x| x.is_finite() && (0.0..=1.0).contains(&x))
                || (win + draw + loss - 1.0).abs() > 1e-4 =>
            {
                return Err(StoreError::InvalidEvidence("invalid WDL"));
            }
            RawScore::Terminal { .. } if self.scope != EvidenceScope::RulesTerminal => {
                return Err(StoreError::InvalidEvidence("unverified terminal claim"));
            }
            _ => {}
        }
        if self.scope == EvidenceScope::RulesTerminal
            && (!matches!(self.score, RawScore::Terminal { .. })
                || self.kind != ObservationKind::ExactTerminal)
        {
            return Err(StoreError::InvalidEvidence(
                "terminal scope needs Rules result",
            ));
        }
        if matches!(self.score, RawScore::Cpu { .. })
            && !matches!(self.scope, EvidenceScope::DepthLimited { .. })
        {
            return Err(StoreError::InvalidEvidence("CPU result needs search scope"));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ObservationStore {
    observations: Vec<Observation>,
    limit: usize,
}

impl ObservationStore {
    pub fn new(limit: usize) -> Self {
        Self {
            observations: Vec::new(),
            limit,
        }
    }

    pub fn append(&mut self, observation: Observation) -> Result<ObservationId, StoreError> {
        observation.validate()?;
        if let Some(previous) = observation.supersedes {
            self.get(previous)?;
        }
        if self.observations.len() >= self.limit {
            return Err(StoreError::Capacity("observations"));
        }
        let id = ObservationId(self.observations.len());
        self.observations.push(observation);
        Ok(id)
    }

    pub fn get(&self, id: ObservationId) -> Result<&Observation, StoreError> {
        self.observations
            .get(id.0)
            .ok_or(StoreError::InvalidHandle("observation"))
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinuationStatus {
    Unresolved,
    Supported,
    Refuted,
    RepairedBy(LineId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContinuationConclusion {
    pub status: ContinuationStatus,
    pub evidence: Option<ObservationId>,
    pub revision: u64,
}

/// Mutable interpretation; immutable raw observations remain independently readable.
#[derive(Clone, Debug, Default)]
pub struct ConclusionView {
    continuations: BTreeMap<LineId, ContinuationConclusion>,
}

impl ConclusionView {
    pub fn get(&self, line: LineId) -> Option<&ContinuationConclusion> {
        self.continuations.get(&line)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&LineId, &ContinuationConclusion)> {
        self.continuations.iter()
    }

    fn ensure_capacity(&self, lines: &[LineId], limit: usize) -> Result<(), StoreError> {
        let additional = lines
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter(|line| !self.continuations.contains_key(line))
            .count();
        if self.continuations.len().saturating_add(additional) > limit {
            return Err(StoreError::Capacity("situation conclusions"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Situation {
    pub state: StateId,
    pub focus: LineId,
    pub revision: u64,
    pub dirty: bool,
    pub conclusions: ConclusionView,
}

#[derive(Debug)]
struct SituationSlot {
    generation: u64,
    value: Option<Situation>,
}

#[derive(Debug)]
pub struct SituationArena {
    slots: Vec<SituationSlot>,
    limit: usize,
    live: usize,
}

impl SituationArena {
    pub fn new(limit: usize) -> Self {
        Self {
            slots: Vec::new(),
            limit,
            live: 0,
        }
    }

    pub fn insert(&mut self, state: StateId, focus: LineId) -> Result<SituationId, StoreError> {
        let value = Situation {
            state,
            focus,
            revision: 0,
            dirty: false,
            conclusions: ConclusionView::default(),
        };
        if let Some((slot, free)) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.value.is_none())
        {
            let id = SituationId {
                slot,
                generation: free.generation,
            };
            free.value = Some(value);
            self.live += 1;
            return Ok(id);
        }
        if self.slots.len() >= self.limit {
            return Err(StoreError::Capacity("situations"));
        }
        let id = SituationId {
            slot: self.slots.len(),
            generation: 0,
        };
        self.slots.push(SituationSlot {
            generation: 0,
            value: Some(value),
        });
        self.live += 1;
        Ok(id)
    }

    pub fn get(&self, id: SituationId) -> Result<&Situation, StoreError> {
        self.slots
            .get(id.slot)
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.value.as_ref())
            .ok_or(StoreError::InvalidHandle("situation generation"))
    }

    pub fn get_mut(&mut self, id: SituationId) -> Result<&mut Situation, StoreError> {
        self.slots
            .get_mut(id.slot)
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.value.as_mut())
            .ok_or(StoreError::InvalidHandle("situation generation"))
    }

    pub fn remove(&mut self, id: SituationId) -> Result<Situation, StoreError> {
        self.get(id)?;
        let slot = &mut self.slots[id.slot];
        let next = slot
            .generation
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        let value = slot
            .value
            .take()
            .ok_or(StoreError::InvalidHandle("situation"))?;
        slot.generation = next;
        self.live -= 1;
        Ok(value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (SituationId, &Situation)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.value.as_ref().map(|value| {
                (
                    SituationId {
                        slot: index,
                        generation: slot.generation,
                    },
                    value,
                )
            })
        })
    }

    pub fn len(&self) -> usize {
        self.live
    }
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
}

#[derive(Debug)]
pub struct DependencyIndex {
    dependents: BTreeMap<ObservationId, BTreeSet<SituationId>>,
    edges: usize,
    limit: usize,
}

impl DependencyIndex {
    pub fn new(limit: usize) -> Self {
        Self {
            dependents: BTreeMap::new(),
            edges: 0,
            limit,
        }
    }

    pub fn add(
        &mut self,
        observation: ObservationId,
        situation: SituationId,
    ) -> Result<(), StoreError> {
        if self
            .dependents
            .get(&observation)
            .is_some_and(|ids| ids.contains(&situation))
        {
            return Ok(());
        }
        if self.edges >= self.limit {
            return Err(StoreError::Capacity("dependency edges"));
        }
        self.dependents
            .entry(observation)
            .or_default()
            .insert(situation);
        self.edges += 1;
        Ok(())
    }

    pub fn affected(&self, observation: ObservationId) -> impl Iterator<Item = SituationId> + '_ {
        self.dependents
            .get(&observation)
            .into_iter()
            .flat_map(|ids| ids.iter().copied())
    }

    /// Only registered live dependents advance revision. Removed generations are
    /// ignored rather than dirtying the next occupant of the same arena slot.
    pub fn invalidate(
        &self,
        observation: ObservationId,
        arena: &mut SituationArena,
    ) -> Result<usize, StoreError> {
        let affected = self
            .affected(observation)
            .filter(|id| arena.get(*id).is_ok())
            .collect::<Vec<_>>();
        for id in &affected {
            arena
                .get(*id)?
                .revision
                .checked_add(1)
                .ok_or(StoreError::RevisionExhausted)?;
        }
        for id in &affected {
            let situation = arena.get_mut(*id)?;
            situation.revision += 1;
            situation.dirty = true;
        }
        Ok(affected.len())
    }

    pub fn len(&self) -> usize {
        self.edges
    }
    pub fn is_empty(&self) -> bool {
        self.edges == 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum TaskQuestion {
    AnalyzePosition,
    AnalyzeRootMoves,
    FindAlternative { divergence_ply: u16 },
    ModelProposal,
    ModelRefutation,
    ModelRepair,
}

/// Exact task reuse namespace. A different requested depth, input revision,
/// profile, history StateId, candidate order, or model epoch is another question.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct TaskKey {
    pub state: StateId,
    pub line: Option<LineId>,
    pub question: TaskQuestion,
    pub root_moves: Vec<Move16>,
    pub model: u64,
    pub epoch: u64,
    pub profile: u64,
    /// Issuer-owned identity for exact search conditions, not a truncated hash.
    /// Model/profile IDs likewise name immutable registered configurations.
    pub condition: u64,
    pub input_revision: u64,
    pub requested_depth: u16,
    pub node_budget: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskConsumer {
    pub id: u64,
    pub situation: SituationId,
    pub revision: u64,
    pub generation: u64,
    /// Monotonic caller clock units, not wall-clock timestamps.
    pub deadline_tick: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskStatus {
    InFlight,
    CancellationRequested,
    Paused {
        checkpoint: u64,
        evidence: Option<ObservationId>,
    },
    Completed(ObservationId),
    Failed,
}

#[derive(Clone, Debug)]
struct ConsumerRecord {
    consumer: TaskConsumer,
    cancelled: bool,
    consumed: bool,
}

#[derive(Clone, Debug)]
pub struct TaskRecord {
    pub key: TaskKey,
    pub status: TaskStatus,
    pub resumed_from: Option<ExecutionId>,
    consumers: Vec<ConsumerRecord>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskAdmission {
    Start(ExecutionId),
    Join(ExecutionId),
    Reuse {
        execution: ExecutionId,
        observation: ObservationId,
    },
    Resume {
        execution: ExecutionId,
        previous: ExecutionId,
        checkpoint: u64,
    },
}

#[derive(Debug)]
pub struct TaskTable {
    tasks: Vec<TaskRecord>,
    latest: BTreeMap<TaskKey, ExecutionId>,
    consumers: BTreeSet<u64>,
    execution_limit: usize,
    consumer_limit: usize,
    root_move_limit: usize,
}

impl TaskTable {
    pub fn new(executions: usize, consumers: usize, root_moves_per_task: usize) -> Self {
        Self {
            tasks: Vec::new(),
            latest: BTreeMap::new(),
            consumers: BTreeSet::new(),
            execution_limit: executions,
            consumer_limit: consumers,
            root_move_limit: root_moves_per_task,
        }
    }

    pub fn request(
        &mut self,
        key: TaskKey,
        consumer: TaskConsumer,
        now_tick: u64,
    ) -> Result<TaskAdmission, StoreError> {
        if consumer.deadline_tick <= now_tick {
            return Err(StoreError::ExpiredConsumer);
        }
        if key.root_moves.len() > self.root_move_limit
            || key.root_moves.capacity() > self.root_move_limit
        {
            return Err(StoreError::Capacity("task root moves"));
        }
        if matches!(
            key.question,
            TaskQuestion::AnalyzePosition
                | TaskQuestion::AnalyzeRootMoves
                | TaskQuestion::FindAlternative { .. }
        ) && (key.requested_depth == 0 || key.node_budget == 0)
        {
            return Err(StoreError::InvalidConditions(
                "CPU task requires a finite positive budget",
            ));
        }
        if self.consumers.contains(&consumer.id) {
            return Err(StoreError::InvalidConditions("duplicate consumer ID"));
        }
        if self.consumers.len() >= self.consumer_limit {
            return Err(StoreError::Capacity("task consumers"));
        }
        let reusable = self.latest.get(&key).copied().filter(|id| {
            matches!(
                self.tasks[id.0].status,
                TaskStatus::InFlight | TaskStatus::Completed(_)
            )
        });
        let (execution, admission) = if let Some(id) = reusable {
            let admission = match self.tasks[id.0].status {
                TaskStatus::Completed(observation) => TaskAdmission::Reuse {
                    execution: id,
                    observation,
                },
                _ => TaskAdmission::Join(id),
            };
            (id, admission)
        } else {
            if self.tasks.len() >= self.execution_limit {
                return Err(StoreError::Capacity("task executions"));
            }
            let id = ExecutionId(self.tasks.len());
            let paused =
                self.latest
                    .get(&key)
                    .and_then(|previous| match self.tasks[previous.0].status {
                        TaskStatus::Paused { checkpoint, .. } => Some((*previous, checkpoint)),
                        _ => None,
                    });
            self.tasks.push(TaskRecord {
                key: key.clone(),
                status: TaskStatus::InFlight,
                resumed_from: paused.map(|(previous, _)| previous),
                consumers: Vec::new(),
            });
            self.latest.insert(key, id);
            let admission = match paused {
                Some((previous, checkpoint)) => TaskAdmission::Resume {
                    execution: id,
                    previous,
                    checkpoint,
                },
                None => TaskAdmission::Start(id),
            };
            (id, admission)
        };
        self.tasks[execution.0].consumers.push(ConsumerRecord {
            consumer,
            cancelled: false,
            consumed: false,
        });
        self.consumers.insert(consumer.id);
        Ok(admission)
    }

    pub fn get(&self, id: ExecutionId) -> Result<&TaskRecord, StoreError> {
        self.tasks
            .get(id.0)
            .ok_or(StoreError::InvalidHandle("execution"))
    }

    /// Publication retains the completed observation even if every consumer has
    /// expired. It grants no old or new root permission to use that observation.
    pub fn complete(
        &mut self,
        execution: ExecutionId,
        observation: ObservationId,
    ) -> Result<(), StoreError> {
        let task = self
            .tasks
            .get_mut(execution.0)
            .ok_or(StoreError::InvalidHandle("execution"))?;
        if !matches!(
            task.status,
            TaskStatus::InFlight | TaskStatus::CancellationRequested
        ) {
            return Err(StoreError::InvalidConditions("execution already terminal"));
        }
        task.status = TaskStatus::Completed(observation);
        Ok(())
    }

    pub fn fail(&mut self, execution: ExecutionId) -> Result<(), StoreError> {
        let task = self
            .tasks
            .get_mut(execution.0)
            .ok_or(StoreError::InvalidHandle("execution"))?;
        if !matches!(
            task.status,
            TaskStatus::InFlight | TaskStatus::CancellationRequested
        ) {
            return Err(StoreError::InvalidConditions("execution already terminal"));
        }
        task.status = TaskStatus::Failed;
        Ok(())
    }

    /// The backend must first establish a supported, physically quiescent resume
    /// boundary. The opaque checkpoint is owned by that same backend/profile;
    /// this table does not copy a stack or transfer it to another CPU engine.
    pub fn pause(
        &mut self,
        execution: ExecutionId,
        checkpoint: u64,
        evidence: Option<ObservationId>,
    ) -> Result<(), StoreError> {
        let task = self
            .tasks
            .get_mut(execution.0)
            .ok_or(StoreError::InvalidHandle("execution"))?;
        if !matches!(
            task.status,
            TaskStatus::InFlight | TaskStatus::CancellationRequested
        ) {
            return Err(StoreError::InvalidConditions("execution already terminal"));
        }
        task.status = TaskStatus::Paused {
            checkpoint,
            evidence,
        };
        Ok(())
    }

    pub fn cancel_consumer(&mut self, execution: ExecutionId, id: u64) -> Result<(), StoreError> {
        let task = self
            .tasks
            .get_mut(execution.0)
            .ok_or(StoreError::InvalidHandle("execution"))?;
        let record = task
            .consumers
            .iter_mut()
            .find(|record| record.consumer.id == id)
            .ok_or(StoreError::InvalidHandle("consumer"))?;
        record.cancelled = true;
        if task.status == TaskStatus::InFlight
            && task.consumers.iter().all(|record| record.cancelled)
        {
            task.status = TaskStatus::CancellationRequested;
        }
        Ok(())
    }

    /// Exactly one consumption per live consumer. Completion reuse may add a new
    /// consumer, but cannot resurrect the deadline or generation of an old one.
    pub fn consume(
        &mut self,
        execution: ExecutionId,
        consumer_id: u64,
        now_tick: u64,
        current_generation: u64,
        arena: &SituationArena,
    ) -> Result<ObservationId, StoreError> {
        let task = self
            .tasks
            .get_mut(execution.0)
            .ok_or(StoreError::InvalidHandle("execution"))?;
        let observation = match task.status {
            TaskStatus::Completed(id) => id,
            _ => return Err(StoreError::NotCompleted),
        };
        let record = task
            .consumers
            .iter_mut()
            .find(|record| record.consumer.id == consumer_id)
            .ok_or(StoreError::InvalidHandle("consumer"))?;
        if record.consumed {
            return Err(StoreError::AlreadyConsumed);
        }
        if record.cancelled {
            return Err(StoreError::CancelledConsumer);
        }
        if record.consumer.deadline_tick <= now_tick {
            return Err(StoreError::ExpiredConsumer);
        }
        let situation = arena
            .get(record.consumer.situation)
            .map_err(|_| StoreError::StaleConsumer)?;
        if record.consumer.generation != current_generation
            || record.consumer.revision != situation.revision
            || task.key.state != situation.state
        {
            return Err(StoreError::StaleConsumer);
        }
        record.consumed = true;
        Ok(observation)
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
    pub fn consumer_count(&self) -> usize {
        self.consumers.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct DerivedCacheKey {
    pub state: StateId,
    pub line: Option<LineId>,
    pub model: u64,
    pub epoch: u64,
    pub encoding: u64,
    pub mask: u64,
    pub precision: u64,
}

/// Optional CPU representation bytes; physical GPU pages live in the runtime's
/// event/lease owner. Evicting these bytes never removes facts or completed tasks.
#[derive(Debug)]
pub struct DerivedCache {
    entries: BTreeMap<DerivedCacheKey, Vec<u8>>,
    entry_limit: usize,
    byte_limit: usize,
    bytes: usize,
}

impl DerivedCache {
    pub fn new(entries: usize, bytes: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            entry_limit: entries,
            byte_limit: bytes,
            bytes: 0,
        }
    }

    pub fn insert(&mut self, key: DerivedCacheKey, bytes: Vec<u8>) -> Result<(), StoreError> {
        let previous = self.entries.get(&key).map_or(0, Vec::capacity);
        let new_total = self
            .bytes
            .checked_sub(previous)
            .and_then(|value| value.checked_add(bytes.capacity()))
            .filter(|total| *total <= self.byte_limit)
            .ok_or(StoreError::Capacity("derived cache bytes"))?;
        if !self.entries.contains_key(&key) && self.entries.len() >= self.entry_limit {
            return Err(StoreError::Capacity("derived cache entries"));
        }
        self.entries.insert(key, bytes);
        self.bytes = new_total;
        Ok(())
    }

    pub fn get(&self, key: &DerivedCacheKey) -> Option<&[u8]> {
        self.entries.get(key).map(Vec::as_slice)
    }

    pub fn evict_all(&mut self) -> usize {
        let freed = self.bytes;
        self.entries.clear();
        self.bytes = 0;
        freed
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Single-writer facade validates cross-store handles before publishing a fact.
#[derive(Debug)]
pub struct PalsStores {
    pub states: StateStore,
    pub lines: LinePool,
    pub situations: SituationArena,
    pub observations: ObservationStore,
    pub dependencies: DependencyIndex,
    pub tasks: TaskTable,
    pub derived: DerivedCache,
    limits: StoreLimits,
    root: Option<SituationId>,
    generation: u64,
}

impl PalsStores {
    pub fn new(limits: StoreLimits) -> Self {
        Self {
            states: StateStore::new(limits.states, limits.retained_history_bytes),
            lines: LinePool::new(limits.line_chunks, limits.line_plies),
            situations: SituationArena::new(limits.situations),
            observations: ObservationStore::new(limits.observations),
            dependencies: DependencyIndex::new(limits.dependency_edges),
            tasks: TaskTable::new(
                limits.executions,
                limits.consumers,
                limits.root_moves_per_task,
            ),
            derived: DerivedCache::new(limits.derived_entries, limits.derived_bytes),
            limits,
            root: None,
            generation: 0,
        }
    }

    pub fn root(&self) -> Option<SituationId> {
        self.root
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn insert_situation(
        &mut self,
        snapshot: PositionSnapshot,
    ) -> Result<SituationId, StoreError> {
        if self.situations.len() >= self.limits.situations {
            return Err(StoreError::Capacity("situations"));
        }
        let existing = self.states.find(&snapshot);
        if (existing.is_none()
            || existing.is_some_and(|state| !self.lines.roots.contains_key(&state)))
            && self.lines.len() >= self.limits.line_chunks
        {
            return Err(StoreError::Capacity("line chunks"));
        }
        let state = self.states.insert(snapshot)?;
        let line = self.lines.root(state)?;
        self.situations.insert(state, line)
    }

    pub fn append_line(
        &mut self,
        prefix: LineId,
        checked_moves: &[BoardMove],
    ) -> Result<LineId, StoreError> {
        self.states.get(self.lines.get(prefix)?.start_state)?;
        self.lines.append(prefix, checked_moves)
    }

    pub fn append_observation(
        &mut self,
        observation: Observation,
    ) -> Result<ObservationId, StoreError> {
        if matches!(observation.score, RawScore::Terminal { .. }) {
            return Err(StoreError::InvalidEvidence(
                "Rules terminal must be minted from a checked Position",
            ));
        }
        self.validate_observation_handles(&observation)?;
        self.observations.append(observation)
    }

    fn validate_observation_handles(&self, observation: &Observation) -> Result<(), StoreError> {
        self.states.get(observation.state)?;
        if let Some(previous) = observation.supersedes {
            if self.observations.get(previous)?.state != observation.state {
                return Err(StoreError::InvalidEvidence(
                    "superseded observation belongs to another exact state",
                ));
            }
        }
        if let Some(line) = observation.line {
            if self.lines.get(line)?.start_state != observation.state {
                return Err(StoreError::InvalidEvidence("line starts at another state"));
            }
        }
        if let Some(execution) = observation.execution {
            let task = self.tasks.get(execution)?;
            if task.key.state != observation.state || task.key.line != observation.line {
                return Err(StoreError::InvalidEvidence(
                    "task and observation conditions differ",
                ));
            }
        }
        Ok(())
    }

    /// Only the Rules-owned automatic termination classifier can create exact
    /// terminal evidence through the product facade. Model/CPU reported mate and
    /// claimable draws cannot select this path by setting an enum field.
    pub fn append_rules_terminal(
        &mut self,
        position: &Position,
        source: u64,
        epoch: u64,
    ) -> Result<ObservationId, StoreError> {
        let legal = position.ordered_legal_moves();
        let winner = match position
            .play_status_from_view(&legal)
            .map_err(|_| StoreError::InvalidEvidence("Rules terminal validation failed"))?
        {
            PlayStatus::Terminal { winner, .. } => winner,
            PlayStatus::Ongoing => {
                return Err(StoreError::InvalidEvidence(
                    "Rules position is not terminal",
                ));
            }
        };
        if self.observations.len() >= self.limits.observations {
            return Err(StoreError::Capacity("observations"));
        }
        let state = self.states.insert(position.snapshot())?;
        let observation = Observation {
            state,
            line: None,
            source,
            epoch,
            scope: EvidenceScope::RulesTerminal,
            score: RawScore::Terminal { winner },
            budget: 0,
            kind: ObservationKind::ExactTerminal,
            supersedes: None,
            execution: None,
        };
        self.validate_observation_handles(&observation)?;
        self.observations.append(observation)
    }

    fn check_conclusion(
        &self,
        situation: SituationId,
        line: LineId,
        evidence: ObservationId,
    ) -> Result<(), StoreError> {
        let current = self.situations.get(situation)?;
        let observation = self.observations.get(evidence)?;
        if self.lines.get(line)?.start_state != current.state
            || observation.state != current.state
            || observation.line != Some(line)
        {
            return Err(StoreError::InvalidEvidence("conclusion conditions differ"));
        }
        Ok(())
    }

    pub fn refute_continuation(
        &mut self,
        situation: SituationId,
        line: LineId,
        evidence: ObservationId,
    ) -> Result<(), StoreError> {
        self.check_conclusion(situation, line, evidence)?;
        if self.observations.get(evidence)?.kind != ObservationKind::Refutation {
            return Err(StoreError::InvalidEvidence(
                "refutation requires refutation record",
            ));
        }
        let current = self.situations.get(situation)?;
        current
            .conclusions
            .ensure_capacity(&[line], self.limits.conclusions_per_situation)?;
        let revision = current
            .revision
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        self.dependencies.add(evidence, situation)?;
        let current = self.situations.get_mut(situation)?;
        current.revision = revision;
        current.dirty = true;
        current.conclusions.continuations.insert(
            line,
            ContinuationConclusion {
                status: ContinuationStatus::Refuted,
                evidence: Some(evidence),
                revision,
            },
        );
        Ok(())
    }

    pub fn repair(
        &mut self,
        situation: SituationId,
        refuted_line: LineId,
        repaired_line: LineId,
        evidence: ObservationId,
    ) -> Result<(), StoreError> {
        self.check_conclusion(situation, repaired_line, evidence)?;
        let observation = self.observations.get(evidence)?;
        if observation.kind != ObservationKind::Repair {
            return Err(StoreError::InvalidEvidence("repair requires repair record"));
        }
        let current = self.situations.get(situation)?;
        if self.lines.get(refuted_line)?.start_state != current.state {
            return Err(StoreError::InvalidEvidence(
                "repair starts at another state",
            ));
        }
        let previous = current
            .conclusions
            .get(refuted_line)
            .filter(|conclusion| conclusion.status == ContinuationStatus::Refuted)
            .ok_or(StoreError::InvalidConditions(
                "repair needs active refutation",
            ))?;
        if observation.supersedes != previous.evidence {
            return Err(StoreError::InvalidEvidence(
                "repair does not supersede active refutation",
            ));
        }
        if refuted_line == repaired_line {
            return Err(StoreError::InvalidConditions(
                "repair needs revised continuation",
            ));
        }
        current.conclusions.ensure_capacity(
            &[refuted_line, repaired_line],
            self.limits.conclusions_per_situation,
        )?;
        let revision = current
            .revision
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        self.dependencies.add(evidence, situation)?;
        let current = self.situations.get_mut(situation)?;
        current.revision = revision;
        current.dirty = true;
        current.conclusions.continuations.insert(
            refuted_line,
            ContinuationConclusion {
                status: ContinuationStatus::RepairedBy(repaired_line),
                evidence: Some(evidence),
                revision,
            },
        );
        current.conclusions.continuations.insert(
            repaired_line,
            ContinuationConclusion {
                status: ContinuationStatus::Supported,
                evidence: Some(evidence),
                revision,
            },
        );
        Ok(())
    }

    /// The caller supplies a state already produced by checked Rules moves. This
    /// method does not replay moves or create legality rules. An existing exact
    /// state situation is retained; every focus issues new acceptance authority.
    pub fn focus_actual_moves(
        &mut self,
        checked_state: PositionSnapshot,
    ) -> Result<SituationId, StoreError> {
        let next_generation = self
            .generation
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        let matching = self.states.find(&checked_state).and_then(|state| {
            self.situations
                .iter()
                .find(|(_, situation)| situation.state == state)
                .map(|(id, _)| id)
        });
        let root = match matching {
            Some(id) => id,
            None => self.insert_situation(checked_state)?,
        };
        self.root = Some(root);
        self.generation = next_generation;
        Ok(root)
    }

    pub fn request_task(
        &mut self,
        key: TaskKey,
        consumer: TaskConsumer,
        now_tick: u64,
    ) -> Result<TaskAdmission, StoreError> {
        self.states.get(key.state)?;
        if let Some(line) = key.line {
            if self.lines.get(line)?.start_state != key.state {
                return Err(StoreError::InvalidConditions("task line state"));
            }
        }
        let situation = self.situations.get(consumer.situation)?;
        if situation.state != key.state
            || situation.revision != consumer.revision
            || consumer.generation != self.generation
        {
            return Err(StoreError::StaleConsumer);
        }
        self.tasks.request(key, consumer, now_tick)
    }

    pub fn complete_task(
        &mut self,
        execution: ExecutionId,
        observation: ObservationId,
    ) -> Result<(), StoreError> {
        let task = self.tasks.get(execution)?;
        let evidence = self.observations.get(observation)?;
        if evidence.execution != Some(execution)
            || evidence.state != task.key.state
            || evidence.line != task.key.line
        {
            return Err(StoreError::InvalidEvidence(
                "completed task record mismatch",
            ));
        }
        if evidence.epoch != task.key.epoch || evidence.budget > task.key.node_budget {
            return Err(StoreError::InvalidEvidence(
                "completed task epoch or budget mismatch",
            ));
        }
        let cpu_question = matches!(
            task.key.question,
            TaskQuestion::AnalyzePosition
                | TaskQuestion::AnalyzeRootMoves
                | TaskQuestion::FindAlternative { .. }
        );
        if (cpu_question && matches!(evidence.scope, EvidenceScope::Model { .. }))
            || (!cpu_question && matches!(evidence.scope, EvidenceScope::DepthLimited { .. }))
        {
            return Err(StoreError::InvalidEvidence(
                "completed task backend scope mismatch",
            ));
        }
        match evidence.scope {
            EvidenceScope::DepthLimited {
                depth,
                profile,
                condition,
            } if depth < task.key.requested_depth
                || profile != task.key.profile
                || condition != task.key.condition =>
            {
                return Err(StoreError::InvalidEvidence(
                    "completed CPU scope does not satisfy task",
                ));
            }
            EvidenceScope::Model { model, input, .. }
                if model != task.key.model || input != task.key.input_revision =>
            {
                return Err(StoreError::InvalidEvidence(
                    "completed model scope does not satisfy task",
                ));
            }
            _ => {}
        }
        self.tasks.complete(execution, observation)
    }

    pub fn pause_task(
        &mut self,
        execution: ExecutionId,
        checkpoint: u64,
        partial: Option<ObservationId>,
    ) -> Result<(), StoreError> {
        let task = self.tasks.get(execution)?;
        if let Some(id) = partial {
            let evidence = self.observations.get(id)?;
            if evidence.execution != Some(execution)
                || evidence.state != task.key.state
                || evidence.line != task.key.line
            {
                return Err(StoreError::InvalidEvidence("paused task record mismatch"));
            }
        }
        self.tasks.pause(execution, checkpoint, partial)
    }

    pub fn consume_task(
        &mut self,
        execution: ExecutionId,
        consumer_id: u64,
        now_tick: u64,
    ) -> Result<ObservationId, StoreError> {
        self.tasks.consume(
            execution,
            consumer_id,
            now_tick,
            self.generation,
            &self.situations,
        )
    }

    pub fn evict_derived_cache(&mut self) -> usize {
        self.derived.evict_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_position::Position;

    fn moves(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|mv| BoardMove::from_uci(mv).unwrap())
            .collect()
    }

    fn observation(state: StateId, line: Option<LineId>, kind: ObservationKind) -> Observation {
        Observation {
            state,
            line,
            source: 1,
            epoch: 1,
            scope: EvidenceScope::DepthLimited {
                depth: 4,
                profile: 1,
                condition: 1,
            },
            score: RawScore::Unknown,
            budget: 128,
            kind,
            supersedes: None,
            execution: None,
        }
    }

    fn key(state: StateId) -> TaskKey {
        TaskKey {
            state,
            line: None,
            question: TaskQuestion::AnalyzePosition,
            root_moves: Vec::new(),
            model: 1,
            epoch: 1,
            profile: 1,
            condition: 1,
            input_revision: 0,
            requested_depth: 4,
            node_budget: 128,
        }
    }

    fn consumer(
        stores: &PalsStores,
        situation: SituationId,
        id: u64,
        deadline_tick: u64,
    ) -> TaskConsumer {
        TaskConsumer {
            id,
            situation,
            revision: stores.situations.get(situation).unwrap().revision,
            generation: stores.generation(),
            deadline_tick,
        }
    }

    #[test]
    fn move16_roundtrips_castling_ep_and_all_promotions_without_claiming_legality() {
        for text in ["e1g1", "e5d6", "a7a8q", "a7a8r", "a7a8b", "a7a8n"] {
            let mv = BoardMove::from_uci(text).unwrap();
            assert_eq!(Move16::pack(mv).unwrap().unpack().unwrap(), mv);
        }
        assert!(Move16(0xffff).unpack().is_err());
        assert!(Move16(0).unpack().is_err());
    }

    #[test]
    fn same_board_with_different_rule_history_is_not_the_same_state() {
        let mut played = Position::startpos();
        played
            .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8"])
            .unwrap();
        let missing_history = Position::from_fen(&played.to_fen()).unwrap();
        assert_eq!(played.to_fen(), missing_history.to_fen());
        let mut states = StateStore::new(4, 1024 * 1024);
        let a = states.insert(played.snapshot()).unwrap();
        let b = states.insert(missing_history.snapshot()).unwrap();
        assert_ne!(a, b);
        assert_eq!(states.insert(played.snapshot()).unwrap(), a);
        assert_eq!(states.len(), 2);
    }

    #[test]
    fn immutable_chunks_share_prefix_and_reject_partial_admission() {
        let mut pool = LinePool::new(4, 64);
        let root = pool.root(StateId(0)).unwrap();
        let prefix = pool.append(root, &moves("e2e4 e7e5")).unwrap();
        let old = pool.append(prefix, &moves("g1f3 b8c6")).unwrap();
        let repaired = pool.append(prefix, &moves("f1c4 g8f6")).unwrap();
        assert_eq!(pool.get(old).unwrap().parent, Some(prefix));
        assert_eq!(pool.get(repaired).unwrap().parent, Some(prefix));
        assert_eq!(pool.append(prefix, &moves("g1f3 b8c6")).unwrap(), old);
        let count = pool.len();
        assert!(pool.append(repaired, &moves("d2d3")).is_err());
        assert_eq!(pool.len(), count);
        assert_eq!(pool.moves(old).unwrap(), moves("e2e4 e7e5 g1f3 b8c6"));
    }

    #[test]
    fn refutation_then_repair_preserves_the_same_first_move() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let situation = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let current = stores.situations.get(situation).unwrap().clone();
        let old = stores
            .append_line(current.focus, &moves("e2e4 e7e5 g1f3"))
            .unwrap();
        let new = stores
            .append_line(current.focus, &moves("e2e4 c7c5 g1f3"))
            .unwrap();
        let refutation = stores
            .append_observation(observation(
                current.state,
                Some(old),
                ObservationKind::Refutation,
            ))
            .unwrap();
        let original_record = stores.observations.get(refutation).unwrap().clone();
        stores
            .refute_continuation(situation, old, refutation)
            .unwrap();
        let mut repair = observation(current.state, Some(new), ObservationKind::Repair);
        repair.supersedes = Some(refutation);
        let repair = stores.append_observation(repair).unwrap();
        stores.repair(situation, old, new, repair).unwrap();
        assert_eq!(
            stores.lines.first_move(old).unwrap(),
            stores.lines.first_move(new).unwrap()
        );
        let conclusions = &stores.situations.get(situation).unwrap().conclusions;
        assert_eq!(
            conclusions.get(old).unwrap().status,
            ContinuationStatus::RepairedBy(new)
        );
        assert_eq!(
            conclusions.get(new).unwrap().status,
            ContinuationStatus::Supported
        );
        assert_eq!(
            stores.observations.get(refutation).unwrap(),
            &original_record
        );
    }

    #[test]
    fn situation_generation_prevents_aba_and_stale_dependency_invalidation() {
        let mut arena = SituationArena::new(1);
        let a = arena.insert(StateId(0), LineId(0)).unwrap();
        let mut dependencies = DependencyIndex::new(4);
        dependencies.add(ObservationId(0), a).unwrap();
        arena.remove(a).unwrap();
        let b = arena.insert(StateId(1), LineId(1)).unwrap();
        assert_eq!(a.slot, b.slot);
        assert_ne!(a.generation, b.generation);
        assert!(arena.get(a).is_err());
        assert_eq!(
            dependencies
                .invalidate(ObservationId(0), &mut arena)
                .unwrap(),
            0
        );
        assert_eq!(arena.get(b).unwrap().revision, 0);
    }

    #[test]
    fn only_registered_dependents_advance_revision() {
        let mut arena = SituationArena::new(3);
        let affected = arena.insert(StateId(0), LineId(0)).unwrap();
        let unrelated = arena.insert(StateId(1), LineId(1)).unwrap();
        let mut dependencies = DependencyIndex::new(4);
        dependencies.add(ObservationId(0), affected).unwrap();
        dependencies.add(ObservationId(0), affected).unwrap();
        assert_eq!(dependencies.len(), 1);
        assert_eq!(
            dependencies
                .invalidate(ObservationId(0), &mut arena)
                .unwrap(),
            1
        );
        assert!(arena.get(affected).unwrap().dirty);
        assert_eq!(arena.get(affected).unwrap().revision, 1);
        assert!(!arena.get(unrelated).unwrap().dirty);
        assert_eq!(arena.get(unrelated).unwrap().revision, 0);
    }

    #[test]
    fn expired_consumer_cannot_pollute_new_root_but_new_consumer_can_reuse_completed_work() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let position = Position::startpos();
        let root = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let first = consumer(&stores, root, 1, 10);
        let execution = match stores.request_task(key(state), first, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("must start"),
        };
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.execution = Some(execution);
        let evidence = stores.append_observation(record).unwrap();
        stores.complete_task(execution, evidence).unwrap();
        assert_eq!(
            stores.consume_task(execution, 1, 10),
            Err(StoreError::ExpiredConsumer)
        );
        stores.focus_actual_moves(position.snapshot()).unwrap();
        assert_eq!(
            stores.consume_task(execution, 1, 5),
            Err(StoreError::StaleConsumer)
        );
        assert_eq!(stores.situations.get(root).unwrap().revision, 0);
        let next = consumer(&stores, root, 2, 30);
        assert_eq!(
            stores.request_task(key(state), next, 12).unwrap(),
            TaskAdmission::Reuse {
                execution,
                observation: evidence
            }
        );
        assert_eq!(stores.consume_task(execution, 2, 13).unwrap(), evidence);
        assert_eq!(
            stores.consume_task(execution, 2, 14),
            Err(StoreError::AlreadyConsumed)
        );
    }

    #[test]
    fn conditions_depth_history_candidate_order_and_consumers_are_independent() {
        let mut table = TaskTable::new(4, 5, 4);
        let first = TaskConsumer {
            id: 1,
            situation: SituationId {
                slot: 0,
                generation: 0,
            },
            revision: 0,
            generation: 1,
            deadline_tick: 20,
        };
        let a = match table.request(key(StateId(0)), first, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let mut second = first;
        second.id = 2;
        second.deadline_tick = 30;
        assert_eq!(
            table.request(key(StateId(0)), second, 1).unwrap(),
            TaskAdmission::Join(a)
        );
        let mut changed = key(StateId(0));
        changed.requested_depth = 10;
        let mut third = second;
        third.id = 3;
        assert!(matches!(
            table.request(changed, third, 1).unwrap(),
            TaskAdmission::Start(_)
        ));
        let mut reordered = key(StateId(0));
        reordered.root_moves = moves("e2e4 d2d4")
            .into_iter()
            .map(|mv| Move16::pack(mv).unwrap())
            .collect();
        let mut fourth = third;
        fourth.id = 4;
        let original = match table.request(reordered.clone(), fourth, 1).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        reordered.root_moves.reverse();
        let mut fifth = fourth;
        fifth.id = 5;
        assert_ne!(
            table.request(reordered, fifth, 1).unwrap(),
            TaskAdmission::Join(original)
        );
        table.cancel_consumer(a, 1).unwrap();
        assert_eq!(table.get(a).unwrap().status, TaskStatus::InFlight);
        table.cancel_consumer(a, 2).unwrap();
        assert_eq!(
            table.get(a).unwrap().status,
            TaskStatus::CancellationRequested
        );
    }

    #[test]
    fn completed_tasks_and_unknown_evidence_survive_derived_cache_eviction() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let requester = consumer(&stores, root, 1, 100);
        let execution = match stores.request_task(key(state), requester, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.execution = Some(execution);
        let evidence = stores.append_observation(record).unwrap();
        stores.complete_task(execution, evidence).unwrap();
        let cache_key = DerivedCacheKey {
            state,
            line: None,
            model: 1,
            epoch: 1,
            encoding: 1,
            mask: 1,
            precision: 32,
        };
        stores.derived.insert(cache_key, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(stores.evict_derived_cache(), 4);
        assert!(stores.derived.get(&cache_key).is_none());
        assert_eq!(
            stores.tasks.get(execution).unwrap().status,
            TaskStatus::Completed(evidence)
        );
        assert_eq!(
            stores.observations.get(evidence).unwrap().score,
            RawScore::Unknown
        );
        assert_eq!(stores.consume_task(execution, 1, 50).unwrap(), evidence);
    }

    #[test]
    fn bounded_pools_keep_records_on_admission_failure() {
        let limits = StoreLimits {
            states: 1,
            observations: 1,
            derived_entries: 1,
            derived_bytes: 2,
            ..StoreLimits::default()
        };
        let mut stores = PalsStores::new(limits);
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let id = stores
            .append_observation(observation(state, None, ObservationKind::CpuAnalysis))
            .unwrap();
        assert!(
            stores
                .append_observation(observation(state, None, ObservationKind::Proposal))
                .is_err()
        );
        assert_eq!(
            stores.observations.get(id).unwrap().score,
            RawScore::Unknown
        );
        let mut played = Position::startpos();
        played.make_uci("e2e4").unwrap();
        assert!(stores.focus_actual_moves(played.snapshot()).is_err());
        assert_eq!(stores.root(), Some(root));
        assert_eq!(stores.generation(), 1);
        let cache_key = DerivedCacheKey {
            state,
            line: None,
            model: 1,
            epoch: 1,
            encoding: 1,
            mask: 1,
            precision: 32,
        };
        stores.derived.insert(cache_key, vec![1, 2]).unwrap();
        assert!(stores.derived.insert(cache_key, vec![3, 4, 5]).is_err());
        assert_eq!(stores.derived.get(&cache_key), Some([1, 2].as_slice()));
    }

    #[test]
    fn finite_search_exact_bound_cannot_be_published_as_a_rules_terminal() {
        let mut observations = ObservationStore::new(2);
        let mut record = observation(StateId(0), None, ObservationKind::CpuAnalysis);
        record.score = RawScore::Cpu {
            value: 9_999,
            perspective: Color::White,
            bound: BoundKind::ExactWithinSearch,
        };
        assert!(observations.append(record.clone()).is_ok());
        record.score = RawScore::Terminal {
            winner: Some(Color::White),
        };
        assert_eq!(
            observations.append(record),
            Err(StoreError::InvalidEvidence("unverified terminal claim"))
        );
    }

    #[test]
    fn partial_depth_is_not_complete_and_resume_issues_a_new_execution() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let mut query = key(state);
        query.requested_depth = 10;
        let requester = consumer(&stores, root, 1, 100);
        let original = match stores.request_task(query.clone(), requester, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.execution = Some(original);
        let partial = stores.append_observation(record).unwrap();
        assert_eq!(
            stores.complete_task(original, partial),
            Err(StoreError::InvalidEvidence(
                "completed CPU scope does not satisfy task"
            ))
        );
        stores.pause_task(original, 7, Some(partial)).unwrap();
        stores.evict_derived_cache();
        let requester = consumer(&stores, root, 2, 100);
        let resumed = match stores.request_task(query.clone(), requester, 1).unwrap() {
            TaskAdmission::Resume {
                execution,
                previous,
                checkpoint,
            } => {
                assert_eq!(previous, original);
                assert_eq!(checkpoint, 7);
                execution
            }
            _ => panic!("must resume same exact task conditions"),
        };
        assert_ne!(original, resumed);
        assert_eq!(
            stores.tasks.get(resumed).unwrap().resumed_from,
            Some(original)
        );
        assert!(matches!(
            stores.tasks.get(original).unwrap().status,
            TaskStatus::Paused { .. }
        ));
        query.profile = 2;
        let requester = consumer(&stores, root, 3, 100);
        assert!(matches!(
            stores.request_task(query, requester, 1).unwrap(),
            TaskAdmission::Start(_)
        ));
        assert_eq!(
            stores.consume_task(original, 1, 2),
            Err(StoreError::NotCompleted)
        );
    }

    #[test]
    fn mismatched_profile_or_model_epoch_does_not_complete_matching_board_task() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let requester = consumer(&stores, root, 1, 100);
        let execution = match stores.request_task(key(state), requester, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.execution = Some(execution);
        record.scope = EvidenceScope::DepthLimited {
            depth: 4,
            profile: 2,
            condition: 1,
        };
        let mismatch = stores.append_observation(record.clone()).unwrap();
        assert_eq!(
            stores.complete_task(execution, mismatch),
            Err(StoreError::InvalidEvidence(
                "completed CPU scope does not satisfy task"
            ))
        );
        record.scope = EvidenceScope::DepthLimited {
            depth: 4,
            profile: 1,
            condition: 1,
        };
        record.epoch = 2;
        let mismatch = stores.append_observation(record).unwrap();
        assert_eq!(
            stores.complete_task(execution, mismatch),
            Err(StoreError::InvalidEvidence(
                "completed task epoch or budget mismatch"
            ))
        );
        assert_eq!(
            stores.tasks.get(execution).unwrap().status,
            TaskStatus::InFlight
        );
    }

    #[test]
    fn chunks_never_exceed_sixteen_and_reconstruction_has_exact_order() {
        let mut pool = LinePool::new(5, 64);
        let root = pool.root(StateId(0)).unwrap();
        let suffix = moves("e2e4 e7e5")
            .into_iter()
            .cycle()
            .take(33)
            .collect::<Vec<_>>();
        let line = pool.append(root, &suffix).unwrap();
        assert_eq!(pool.moves(line).unwrap(), suffix);
        assert_eq!(pool.get(line).unwrap().plies(), 33);
        let mut current = Some(line);
        while let Some(id) = current {
            let chunk = pool.get(id).unwrap();
            assert!(chunk.chunk_moves().len() <= MOVES_PER_CHUNK);
            current = chunk.parent;
        }
    }

    #[test]
    fn focused_exact_analysis_situation_is_retained_after_actual_move() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let parent = Position::startpos();
        let old_root = stores.focus_actual_moves(parent.snapshot()).unwrap();
        let child = parent
            .preview_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let analyzed = stores.insert_situation(child.snapshot()).unwrap();
        let state = stores.situations.get(analyzed).unwrap().state;
        let evidence = stores
            .append_observation(observation(state, None, ObservationKind::CpuAnalysis))
            .unwrap();
        assert_eq!(
            stores.focus_actual_moves(child.snapshot()).unwrap(),
            analyzed
        );
        assert_ne!(stores.root(), Some(old_root));
        assert_eq!(stores.observations.get(evidence).unwrap().state, state);
        assert!(stores.situations.get(old_root).is_ok());
        assert_eq!(stores.generation(), 2);
    }

    #[test]
    fn facade_mints_exact_terminal_only_from_rules_and_claimable_draw_stays_ongoing() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let ongoing = Position::startpos();
        let root = stores.focus_actual_moves(ongoing.snapshot()).unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let mut fabricated = observation(state, None, ObservationKind::ExactTerminal);
        fabricated.scope = EvidenceScope::RulesTerminal;
        fabricated.score = RawScore::Terminal {
            winner: Some(Color::White),
        };
        assert_eq!(
            stores.append_observation(fabricated),
            Err(StoreError::InvalidEvidence(
                "Rules terminal must be minted from a checked Position"
            ))
        );
        assert_eq!(
            stores.append_rules_terminal(&ongoing, 1, 1),
            Err(StoreError::InvalidEvidence(
                "Rules position is not terminal"
            ))
        );
        let claimable = Position::from_fen("7k/8/8/8/8/8/Q7/K7 w - - 100 1").unwrap();
        assert_eq!(
            stores.append_rules_terminal(&claimable, 1, 1),
            Err(StoreError::InvalidEvidence(
                "Rules position is not terminal"
            ))
        );
        let mate = Position::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let id = stores.append_rules_terminal(&mate, 1, 1).unwrap();
        assert_eq!(
            stores.observations.get(id).unwrap().score,
            RawScore::Terminal {
                winner: Some(Color::White)
            }
        );
        let stale = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let id = stores.append_rules_terminal(&stale, 1, 1).unwrap();
        assert_eq!(
            stores.observations.get(id).unwrap().score,
            RawScore::Terminal { winner: None }
        );
    }

    #[test]
    fn large_spare_owned_capacity_cannot_bypass_payload_limits() {
        let mut cache = DerivedCache::new(1, 2);
        let mut spare = Vec::with_capacity(512);
        spare.push(1);
        let key = DerivedCacheKey {
            state: StateId(0),
            line: None,
            model: 1,
            epoch: 1,
            encoding: 1,
            mask: 1,
            precision: 32,
        };
        assert_eq!(
            cache.insert(key, spare),
            Err(StoreError::Capacity("derived cache bytes"))
        );
        assert_eq!(cache.bytes(), 0);
        let mut table = TaskTable::new(1, 1, 4);
        let mut query = super::tests::key(StateId(0));
        query.root_moves = Vec::with_capacity(512);
        let consumer = TaskConsumer {
            id: 1,
            situation: SituationId {
                slot: 0,
                generation: 0,
            },
            revision: 0,
            generation: 1,
            deadline_tick: 20,
        };
        assert_eq!(
            table.request(query, consumer, 0),
            Err(StoreError::Capacity("task root moves"))
        );
        assert_eq!(table.len(), 0);
    }
}
