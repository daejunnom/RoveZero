//! Actual, pinned ReviewView -> relational C preparation. The manager admits
//! immutable custody only; Rules replay and NNUEC preparation run on the
//! explicitly created actor. This does not consume a view or select a policy.

use super::{Context, Edge, EdgeKind, LineKind, Model};
use crate::cpu_value::{CpuValueIdentity, nnuec::NnuecError};
use crate::pals::engine::RecordKind;
use crate::pals::review_context::{ReviewView, checked_rebase_offset};
use crate::pals::store::{PalsStores, StateId};
use rz_position::{MAX_UCI_REPLAY_PLIES, Position, PositionIdentity, PositionSnapshot};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparationError {
    StaleView,
    InvalidView(&'static str),
    Cancelled,
    Expired,
    Busy,
    WorkerStopped,
    WorkerPanicked,
    Neural(NnuecError),
}
impl From<NnuecError> for PreparationError {
    fn from(value: NnuecError) -> Self {
        Self::Neural(value)
    }
}
impl std::fmt::Display for PreparationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NNUEC review preparation: {self:?}")
    }
}
impl std::error::Error for PreparationError {}

/// Identifies the original submission, including its query. The public C
/// record scope is additional evidence, never a replacement for this seal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Provenance {
    pub root: StateId,
    pub generation: u64,
    pub sequence: u64,
    pub full_view_digest: [u8; 32],
    pub view_seal: [u8; 32],
    pub record_scope_digest: [u8; 32],
    pub policy_digest: [u8; 32],
    pub model: CpuValueIdentity,
}

/// Fields cannot be forged by callers. The view's actual hot pin remains
/// alive while queued, preparing, cached, or held by an accepted consumer.
pub struct Request {
    view: Arc<ReviewView>,
    sources: Vec<PositionSnapshot>,
    provenance: Provenance,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
}
impl Request {
    /// Bounded custody capture, with no encoder/forward/Rules replay. Source
    /// snapshots share immutable history rather than copying an actual game.
    pub fn admit(
        model: &Model,
        view: Arc<ReviewView>,
        stores: &PalsStores,
        expected_root: StateId,
        expected_generation: u64,
        deadline: Instant,
        cancel: Arc<AtomicBool>,
    ) -> Result<Self, PreparationError> {
        check_control(deadline, &cancel)?;
        if !view.validates(stores)
            || view.root_state() != expected_root
            || view.generation() != expected_generation
            || stores.generation() != expected_generation
        {
            return Err(PreparationError::StaleView);
        }
        if view.records().len() > 64
            || view.records().len() > model.base_checkpoint().config.max_lines
            || view.records_metadata().len() != view.records().len()
            || view.anchors().is_empty()
            || view.anchors().len() > 4
        {
            return Err(PreparationError::InvalidView(
                "selected view shape/capacity",
            ));
        }
        let root = stores
            .states
            .get(expected_root)
            .map_err(|_| PreparationError::StaleView)?;
        if view.anchors()[0].state != expected_root || !root.same_state(&view.anchors()[0].snapshot)
        {
            return Err(PreparationError::StaleView);
        }
        let mut sources = Vec::with_capacity(view.records().len());
        for metadata in view.records_metadata() {
            sources.push(
                stores
                    .states
                    .get(metadata.source_state)
                    .map_err(|_| PreparationError::StaleView)?
                    .clone(),
            );
        }
        let provenance = Provenance {
            root: expected_root,
            generation: expected_generation,
            sequence: view.sequence(),
            full_view_digest: view.digest(),
            view_seal: view.seal_digest(),
            record_scope_digest: view.record_scope_digest(),
            policy_digest: view.policy_digest(),
            model: model.identity().clone(),
        };
        Ok(Self {
            view,
            sources,
            provenance,
            deadline,
            cancel,
        })
    }
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

pub struct BoundNnuecReviewContext {
    context: Arc<Context>,
    view: Arc<ReviewView>,
    root: PositionIdentity,
    provenance: Provenance,
    reused_record_scope: bool,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
    invalidations: Arc<AtomicU64>,
    preparation_epoch: u64,
    stopping: Arc<AtomicBool>,
}
impl BoundNnuecReviewContext {
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }
    pub fn view(&self) -> &Arc<ReviewView> {
        &self.view
    }
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    pub fn reused_record_scope(&self) -> bool {
        self.reused_record_scope
    }
    /// Worker-side authority check without borrowing the single-writer store.
    /// Store/root/original-ticket admission remains required in validates().
    pub fn authority_is_live(&self) -> bool {
        !self.stopping.load(Ordering::Acquire)
            && check_control(self.deadline, &self.cancel).is_ok()
            && self.invalidations.load(Ordering::Acquire) == self.preparation_epoch
    }
    /// Acceptance must use the original ticket seal, never the latest view.
    /// Preparation does not call ReviewContextIndex::consume.
    pub fn validates(
        &self,
        stores: &PalsStores,
        expected_root: StateId,
        expected_generation: u64,
        original_view_seal: [u8; 32],
    ) -> bool {
        self.view.validates(stores)
            && self.authority_is_live()
            && self.provenance.root == expected_root
            && self.provenance.generation == expected_generation
            && self.provenance.view_seal == original_view_seal
            && self.view.seal_digest() == original_view_seal
            && stores
                .states
                .get(expected_root)
                .is_ok_and(|root| root.position_identity() == self.root)
    }
}

struct CachedScope {
    view: Arc<ReviewView>,
    digest: [u8; 32],
    context: Arc<Context>,
}
/// Owned exclusively by the actor below, with one bounded cached scope.
struct Factory {
    model: Arc<Model>,
    cached: Option<CachedScope>,
    stopping: Arc<AtomicBool>,
    invalidations: Arc<AtomicU64>,
    invalidation_ack: Arc<AtomicU64>,
    active_epoch: u64,
}
impl Factory {
    fn sync_cache_epoch(&mut self) {
        let current = self.invalidations.load(Ordering::Acquire);
        if self.active_epoch != current {
            self.cached = None;
            self.active_epoch = current;
        }
        self.invalidation_ack.store(current, Ordering::Release);
    }
    fn control(&self, request: &Request) -> Result<(), PreparationError> {
        if self.stopping.load(Ordering::Acquire)
            || self.active_epoch != self.invalidations.load(Ordering::Acquire)
        {
            return Err(PreparationError::Cancelled);
        }
        check_control(request.deadline, &request.cancel)
    }
    #[cfg(test)]
    fn prepare(&mut self, request: Request) -> Result<BoundNnuecReviewContext, PreparationError> {
        let epoch = self.invalidations.load(Ordering::Acquire);
        self.prepare_at(request, epoch)
    }
    fn prepare_at(
        &mut self,
        request: Request,
        epoch: u64,
    ) -> Result<BoundNnuecReviewContext, PreparationError> {
        self.sync_cache_epoch();
        if self.active_epoch != epoch {
            return Err(PreparationError::Cancelled);
        }
        self.control(&request)?;
        if &request.provenance.model != self.model.identity() {
            return Err(PreparationError::StaleView);
        }
        let view = &request.view;
        let cached = self.cached.as_ref().filter(|cached| {
            cached.digest == request.provenance.record_scope_digest
                && same_numeric_scope(&cached.view, view)
        });
        let reused_record_scope = cached.is_some();
        let context = if let Some(cached) = cached {
            Arc::clone(&cached.context)
        } else {
            let mut context = self.model.empty_context();
            for (index, ((record, metadata), source)) in view
                .records()
                .iter()
                .zip(view.records_metadata())
                .zip(&request.sources)
                .enumerate()
            {
                self.control(&request)?;
                let anchor = view
                    .anchors()
                    .get(usize::from(metadata.anchor))
                    .ok_or(PreparationError::InvalidView("record anchor"))?;
                if record.origin_state != anchor.state
                    || record.revision != metadata.source_revision
                {
                    return Err(PreparationError::InvalidView("record source provenance"));
                }
                let offset = usize::from(metadata.source_line_offset);
                if offset
                    .checked_add(record.line.len())
                    .is_none_or(|length| length > 256)
                {
                    return Err(PreparationError::InvalidView("original line capacity"));
                }
                let mut original = Vec::with_capacity(offset + record.line.len());
                if offset != 0 {
                    let actual = anchor
                        .snapshot
                        .uci_replay(MAX_UCI_REPLAY_PLIES)
                        .map_err(|_| PreparationError::InvalidView("rebase actual history"))?;
                    let start = actual
                        .moves
                        .len()
                        .checked_sub(offset)
                        .ok_or(PreparationError::InvalidView("rebase history offset"))?;
                    original.extend_from_slice(&actual.moves[start..]);
                }
                original.extend_from_slice(&record.line);
                if original.first().copied() != metadata.original_first_move
                    || checked_rebase_offset(source, &anchor.snapshot, &original)
                        .map_err(|_| PreparationError::InvalidView("Rules rebase"))?
                        != Some(offset)
                {
                    return Err(PreparationError::InvalidView(
                        "original bundle/rebase mismatch",
                    ));
                }
                let anchor_position = Position::from_snapshot(&anchor.snapshot, Default::default())
                    .map_err(|e| PreparationError::Neural(NnuecError::Rules(e.to_string())))?;
                let kind = match record.kind {
                    RecordKind::Proposal => LineKind::Proposal,
                    RecordKind::Counterexample => LineKind::Counterexample,
                    RecordKind::Repair => LineKind::Repair,
                    RecordKind::CpuVerification => LineKind::CpuVerification,
                };
                let line = self.model.encode_review_line(
                    &anchor_position,
                    &record.line,
                    kind,
                    metadata.original_first_move,
                )?;
                context = self.model.with_line(&context, index as u64, line)?;
            }
            for (index, metadata) in view.records_metadata().iter().enumerate() {
                self.control(&request)?;
                let record = &view.records()[index];
                for (source, revision, kind) in [
                    (metadata.parent, record.parent_revision, EdgeKind::Parent),
                    (
                        metadata.supersedes,
                        record.supersedes_revision,
                        EdgeKind::Supersedes,
                    ),
                ] {
                    match (source, revision) {
                        (None, None) => {}
                        (Some(source), Some(revision))
                            if usize::from(source) < index
                                && view.records()[usize::from(source)].revision == revision =>
                        {
                            context = self.model.with_edge(
                                &context,
                                Edge {
                                    source: u64::from(source),
                                    target: index as u64,
                                    kind,
                                },
                            )?;
                        }
                        _ => {
                            return Err(PreparationError::InvalidView("local relation provenance"));
                        }
                    }
                }
            }
            context
        };
        self.control(&request)?;
        self.cached = Some(CachedScope {
            view: Arc::clone(view),
            digest: request.provenance.record_scope_digest,
            context: Arc::clone(&context),
        });
        Ok(BoundNnuecReviewContext {
            root: view.anchors()[0].snapshot.position_identity(),
            context,
            view: request.view,
            provenance: request.provenance,
            reused_record_scope,
            deadline: request.deadline,
            cancel: request.cancel,
            invalidations: Arc::clone(&self.invalidations),
            preparation_epoch: epoch,
            stopping: Arc::clone(&self.stopping),
        })
    }
}
fn same_numeric_scope(first: &ReviewView, second: &ReviewView) -> bool {
    first.root_state() == second.root_state()
        && first.generation() == second.generation()
        && first.records_metadata() == second.records_metadata()
        && first.anchors().len() == second.anchors().len()
        && first
            .anchors()
            .iter()
            .zip(second.anchors())
            .all(|(a, b)| a.state == b.state && a.snapshot.same_state(&b.snapshot))
        && first.records().len() == second.records().len()
        && first.records().iter().zip(second.records()).all(|(a, b)| {
            a.kind == b.kind
                && a.line == b.line
                && a.revision == b.revision
                && a.parent_revision == b.parent_revision
                && a.supersedes_revision == b.supersedes_revision
                && a.origin_state == b.origin_state
        })
}
fn check_control(deadline: Instant, cancel: &AtomicBool) -> Result<(), PreparationError> {
    if cancel.load(Ordering::Acquire) {
        return Err(PreparationError::Cancelled);
    }
    if Instant::now() >= deadline {
        return Err(PreparationError::Expired);
    }
    Ok(())
}

pub struct Completion {
    pub provenance: Provenance,
    pub result: Result<BoundNnuecReviewContext, PreparationError>,
}
enum Command {
    Prepare(Request, u64),
    Wake,
}
/// One explicitly accounted preparation worker; no default background thread.
/// At most one queued/running/uncollected completion exists. The result channel
/// is logically bounded by that gate, and terminal results are never coalesced.
pub struct PreparationActor {
    sender: Option<SyncSender<Command>>,
    receiver: Receiver<Completion>,
    outstanding: Arc<AtomicUsize>,
    stopping: Arc<AtomicBool>,
    invalidations: Arc<AtomicU64>,
    invalidation_ack: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
}
impl PreparationActor {
    pub fn spawn(model: Arc<Model>) -> Result<Self, PreparationError> {
        let (sender, commands) = mpsc::sync_channel::<Command>(1);
        let (results, receiver) = mpsc::channel();
        let outstanding = Arc::new(AtomicUsize::new(0));
        let stopping = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&stopping);
        let invalidations = Arc::new(AtomicU64::new(0));
        let invalidation_ack = Arc::new(AtomicU64::new(0));
        let worker_invalidations = Arc::clone(&invalidations);
        let worker_ack = Arc::clone(&invalidation_ack);
        let worker = thread::Builder::new()
            .name("nnuec-review-prepare".into())
            .spawn(move || {
                let mut factory = Factory {
                    model,
                    cached: None,
                    stopping: Arc::clone(&stop),
                    invalidations: worker_invalidations,
                    invalidation_ack: worker_ack,
                    active_epoch: 0,
                };
                while let Ok(command) = commands.recv() {
                    factory.sync_cache_epoch();
                    let Command::Prepare(request, epoch) = command else {
                        continue;
                    };
                    let provenance = request.provenance.clone();
                    let result = if stop.load(Ordering::Acquire) {
                        Err(PreparationError::Cancelled)
                    } else {
                        factory.prepare_at(request, epoch)
                    };
                    factory.sync_cache_epoch();
                    if results.send(Completion { provenance, result }).is_err() {
                        break;
                    }
                }
            })
            .map_err(|_| PreparationError::WorkerStopped)?;
        Ok(Self {
            sender: Some(sender),
            receiver,
            outstanding,
            stopping,
            invalidations,
            invalidation_ack,
            worker: Some(worker),
        })
    }
    /// Failure returns ownership of the request and its pin to the caller.
    pub fn try_submit(&self, request: Request) -> Result<(), (PreparationError, Request)> {
        if self.stopping.load(Ordering::Acquire) {
            return Err((PreparationError::WorkerStopped, request));
        }
        if self
            .outstanding
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err((PreparationError::Busy, request));
        }
        match self
            .sender
            .as_ref()
            .expect("live actor sender")
            .try_send(Command::Prepare(
                request,
                self.invalidations.load(Ordering::Acquire),
            )) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.outstanding.store(0, Ordering::Release);
                match error {
                    TrySendError::Full(Command::Prepare(request, _)) => {
                        Err((PreparationError::Busy, request))
                    }
                    TrySendError::Disconnected(Command::Prepare(request, _)) => {
                        Err((PreparationError::WorkerStopped, request))
                    }
                    _ => unreachable!("only a prepare command was submitted"),
                }
            }
        }
    }
    pub fn try_complete(&self) -> Result<Option<Completion>, PreparationError> {
        match self.receiver.try_recv() {
            Ok(completion) => {
                self.outstanding.store(0, Ordering::Release);
                Ok(Some(completion))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(PreparationError::WorkerStopped),
        }
    }
    pub fn shutdown(mut self) -> Result<(), PreparationError> {
        self.stop_and_join()
    }
    /// Independent atomic control cannot be blocked by the ordinary request
    /// slot. It revokes old Bound acceptance immediately; the worker publishes
    /// a cache-release acknowledgement after observing it. Pending terminals
    /// remain available for collection and retain their original provenance.
    pub fn invalidate_cache(&self) -> Result<u64, PreparationError> {
        let previous = self
            .invalidations
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |epoch| {
                epoch.checked_add(1)
            })
            .map_err(|_| PreparationError::InvalidView("preparation epoch exhausted"))?;
        match self
            .sender
            .as_ref()
            .expect("live actor sender")
            .try_send(Command::Wake)
        {
            Ok(()) | Err(TrySendError::Full(_)) => Ok(previous + 1),
            Err(TrySendError::Disconnected(_)) => Err(PreparationError::WorkerStopped),
        }
    }
    pub fn cache_invalidated_through(&self) -> u64 {
        self.invalidation_ack.load(Ordering::Acquire)
    }
    fn stop_and_join(&mut self) -> Result<(), PreparationError> {
        self.stopping.store(true, Ordering::Release);
        self.sender.take();
        if self
            .worker
            .take()
            .is_some_and(|worker| worker.join().is_err())
        {
            return Err(PreparationError::WorkerPanicked);
        }
        Ok(())
    }
}
impl Drop for PreparationActor {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals::engine::{RoleQueryPurpose, RoleRecord};
    use crate::pals::review_context::{
        ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
    };
    use crate::pals::store::StoreLimits;
    use rz_position::{BoardMove, Color};
    use std::time::Duration;

    fn record(
        state: StateId,
        revision: u64,
        kind: RecordKind,
        text: &str,
        parent: Option<u64>,
    ) -> RoleRecord {
        RoleRecord {
            revision,
            parent_revision: parent,
            supersedes_revision: None,
            origin_state: state,
            kind,
            line: text
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect(),
            value: None,
            completed_depth: 0,
            score_scope: None,
            cpu_observation: None,
            perspective: Color::White,
            critical: true,
        }
    }
    fn query(purpose: RoleQueryPurpose) -> ReviewQuerySequences<'static> {
        ReviewQuerySequences {
            purpose,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            legal: &[],
            divergences: &[],
        }
    }
    fn setup() -> (PalsStores, ReviewContextIndex, StateId, u64) {
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let mut index = ReviewContextIndex::default();
        index
            .publish(
                &stores,
                &record(state, 1, RecordKind::Proposal, "e2e4", None),
                None,
            )
            .unwrap();
        let bundle = index
            .publish(
                &stores,
                &record(state, 2, RecordKind::Counterexample, "e2e4 e7e5", Some(1)),
                None,
            )
            .unwrap();
        (stores, index, state, bundle)
    }
    fn factory() -> Factory {
        Factory {
            model: Arc::new(Model::new(super::super::tests::checkpoint()).unwrap()),
            cached: None,
            stopping: Arc::new(AtomicBool::new(false)),
            invalidations: Arc::new(AtomicU64::new(0)),
            invalidation_ack: Arc::new(AtomicU64::new(0)),
            active_epoch: 0,
        }
    }
    fn request(
        factory: &Factory,
        view: Arc<ReviewView>,
        stores: &PalsStores,
        root: StateId,
    ) -> Request {
        Request::admit(
            &factory.model,
            view,
            stores,
            root,
            stores.generation(),
            Instant::now() + Duration::from_secs(10),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap()
    }

    #[test]
    fn actual_frozen_views_reuse_c_but_keep_query_seals_and_consumption_authority() {
        let (stores, mut index, state, bundle) = setup();
        let mut factory = factory();
        let (a, b) = index
            .select_frozen_pair(
                &stores,
                state,
                query(RoleQueryPurpose::ProposePolicy),
                query(RoleQueryPurpose::ValueFresh),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        assert_ne!(a.digest(), b.digest());
        let protected = index.bundle(bundle).unwrap().protected_consumptions();
        let prepared_a = factory
            .prepare(request(&factory, Arc::clone(&a), &stores, state))
            .unwrap();
        let prepared_b = factory
            .prepare(request(&factory, Arc::clone(&b), &stores, state))
            .unwrap();
        assert!(!prepared_a.reused_record_scope());
        assert!(prepared_b.reused_record_scope());
        assert!(Arc::ptr_eq(prepared_a.context(), prepared_b.context()));
        assert_ne!(
            prepared_a.provenance().full_view_digest,
            prepared_b.provenance().full_view_digest
        );
        assert!(prepared_a.validates(&stores, state, stores.generation(), a.seal_digest()));
        assert!(!prepared_a.validates(&stores, state, stores.generation(), b.seal_digest()));
        assert_eq!(
            index.bundle(bundle).unwrap().protected_consumptions(),
            protected
        );
        index.consume(&a, &stores).unwrap();
        assert_eq!(
            index.bundle(bundle).unwrap().protected_consumptions(),
            protected - 1
        );
    }

    #[test]
    fn actual_rules_rebase_preserves_original_bundle_move_and_empty_suffix_relation() {
        let (mut stores, mut index, _state, bundle) = setup();
        let mut actual = Position::startpos();
        actual
            .make_move("e2e4".parse::<BoardMove>().unwrap())
            .unwrap();
        let situation = stores.focus_actual_moves(actual.snapshot()).unwrap();
        let current = stores.situations.get(situation).unwrap().state;
        let view = Arc::new(
            index
                .select(
                    &stores,
                    current,
                    query(RoleQueryPurpose::ReplyPolicy),
                    ReviewSelectionPolicy {
                        required_bundle: Some(bundle),
                        ..ReviewSelectionPolicy::default()
                    },
                )
                .unwrap(),
        );
        assert!(view.records()[0].line.is_empty());
        assert_eq!(view.records_metadata()[0].source_line_offset, 1);
        let mut factory = factory();
        let bound = factory
            .prepare(request(&factory, Arc::clone(&view), &stores, current))
            .unwrap();
        assert_eq!(bound.context().line_count(), 2);
        assert_eq!(bound.context().edge_count(), 1);
        let reference = bound.context().reference_lines();
        assert_eq!(reference[0].moves.len(), 0);
        assert!(reference[0].original_first_move.is_some());
        assert_eq!(reference[1].moves.len(), 1);
        assert_eq!(
            reference[0]
                .original_first_move
                .as_ref()
                .unwrap()
                .from_square,
            12
        );
    }

    #[test]
    fn foreign_store_stale_root_cancel_and_expiry_reject_without_consumption() {
        let (mut stores, mut index, state, bundle) = setup();
        let factory = factory();
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let (foreign, _, _, _) = setup();
        assert!(
            Request::admit(
                &factory.model,
                Arc::clone(&view),
                &foreign,
                state,
                stores.generation(),
                Instant::now() + Duration::from_secs(1),
                Arc::new(AtomicBool::new(false))
            )
            .is_err()
        );
        let cancelled = request(&factory, Arc::clone(&view), &stores, state);
        cancelled.cancel.store(true, Ordering::Release);
        let mut preparing = factory;
        assert!(matches!(
            preparing.prepare(cancelled),
            Err(PreparationError::Cancelled)
        ));
        let mut expired = request(&preparing, Arc::clone(&view), &stores, state);
        expired.deadline = Instant::now() - Duration::from_secs(1);
        assert!(matches!(
            preparing.prepare(expired),
            Err(PreparationError::Expired)
        ));
        assert_eq!(index.bundle(bundle).unwrap().protected_consumptions(), 2);
        let next = Position::from_fen("8/8/8/8/8/8/4K3/7k w - - 0 1").unwrap();
        stores.focus_actual_moves(next.snapshot()).unwrap();
        assert!(
            Request::admit(
                &preparing.model,
                view,
                &stores,
                state,
                stores.generation(),
                Instant::now() + Duration::from_secs(1),
                Arc::new(AtomicBool::new(false))
            )
            .is_err()
        );
    }

    #[test]
    fn actor_retains_terminal_slot_until_collected_and_joins_on_shutdown() {
        let (stores, mut index, state, _) = setup();
        let model = Arc::new(Model::new(super::super::tests::checkpoint()).unwrap());
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let make_request = || {
            Request::admit(
                &model,
                Arc::clone(&view),
                &stores,
                state,
                stores.generation(),
                Instant::now() + Duration::from_secs(10),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap()
        };
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        actor
            .try_submit(make_request())
            .unwrap_or_else(|_| panic!("first admission"));
        assert!(matches!(
            actor.try_submit(make_request()),
            Err((PreparationError::Busy, _))
        ));
        let until = Instant::now() + Duration::from_secs(10);
        let completion = loop {
            if let Some(completion) = actor.try_complete().unwrap() {
                break completion;
            }
            assert!(
                Instant::now() < until,
                "actor did not publish terminal result"
            );
            thread::yield_now();
        };
        let bound = completion.result.unwrap();
        assert!(bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        actor.shutdown().unwrap();
    }

    #[test]
    fn late_cancel_expiry_and_cache_epoch_revoke_prepared_acceptance() {
        let (stores, mut index, state, _) = setup();
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let mut factory = factory();
        let request = request(&factory, Arc::clone(&view), &stores, state);
        let cancel = Arc::clone(&request.cancel);
        let mut bound = factory.prepare(request).unwrap();
        assert!(bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        cancel.store(true, Ordering::Release);
        assert!(!bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        cancel.store(false, Ordering::Release);
        bound.deadline = Instant::now() - Duration::from_secs(1);
        assert!(!bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        bound.deadline = Instant::now() + Duration::from_secs(1);
        factory.invalidations.fetch_add(1, Ordering::AcqRel);
        assert!(!bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        factory.sync_cache_epoch();
        assert!(factory.cached.is_none());
        assert_eq!(factory.invalidation_ack.load(Ordering::Acquire), 1);
    }

    #[test]
    fn independent_cache_invalidation_wakes_idle_actor_and_preserves_terminal_slot() {
        let (stores, mut index, state, _) = setup();
        let model = Arc::new(Model::new(super::super::tests::checkpoint()).unwrap());
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        let request = Request::admit(
            &model,
            Arc::clone(&view),
            &stores,
            state,
            stores.generation(),
            Instant::now() + Duration::from_secs(10),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        actor
            .try_submit(request)
            .unwrap_or_else(|_| panic!("first admission"));
        let until = Instant::now() + Duration::from_secs(10);
        let bound = loop {
            if let Some(completion) = actor.try_complete().unwrap() {
                break completion.result.unwrap();
            }
            assert!(Instant::now() < until);
            thread::yield_now();
        };
        let epoch = actor.invalidate_cache().unwrap();
        assert!(!bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        while actor.cache_invalidated_through() < epoch {
            assert!(Instant::now() < until);
            thread::yield_now();
        }
        assert!(actor.try_complete().unwrap().is_none());
        actor.shutdown().unwrap();
    }
}
