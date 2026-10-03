use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use crate::policy::{EdgeStats, PolicyIdentity, Puct, SelectionPolicy};

#[derive(Clone, Debug, PartialEq)]
pub enum SearchError {
    InvalidConfiguration(&'static str),
    InvalidStatistics,
    InvalidValue,
    InvalidPolicy,
    DuplicateMove,
    NoLegalEdges,
    Busy,
    Canceled,
    Expired,
    DepthLimit,
    NodeLimit,
    EdgeLimit,
    AllocationFailed,
    CounterOverflow,
    InvalidPolicySelection,
    InconsistentLeaf,
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SearchError {}

#[derive(Clone, Copy, Debug)]
pub struct TreeLimits {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub max_depth: usize,
    pub max_legal_moves: usize,
    pub probability_tolerance: f64,
}

impl Default for TreeLimits {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_edges: 1_000_000,
            max_depth: 256,
            max_legal_moves: 4096,
            probability_tolerance: 1e-9,
        }
    }
}

impl TreeLimits {
    fn validate(self) -> Result<(), SearchError> {
        if self.max_nodes == 0
            || self.max_edges == 0
            || self.max_depth == 0
            || self.max_legal_moves == 0
        {
            return Err(SearchError::InvalidConfiguration("zero tree limit"));
        }
        if !self.probability_tolerance.is_finite()
            || !(0.0..=1e-3).contains(&self.probability_tolerance)
        {
            return Err(SearchError::InvalidConfiguration("probability tolerance"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchCounters {
    /// Logical selections, including root preparation; never network executions.
    pub selections: u64,
    pub root_initializations: u64,
    pub accepted_backups: u64,
    pub completed_visits: u64,
    pub reservations_released: u64,
    pub rejected_results: u64,
}

fn increment(value: &mut u64) -> Result<(), SearchError> {
    *value = value.checked_add(1).ok_or(SearchError::CounterOverflow)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    Stale,
    AlreadyFinalized,
    Canceled,
    Expired,
    /// The consumer's live contract authority rejected the final commit.
    AdmissionClosed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion {
    Accepted { traversed_edges: usize },
    Rejected(Rejection),
}

/// B-local consume capability, not the shared contract's RequestId/SelectionId.
/// Adapters map this ticket to the published common request and generation fields.
#[derive(Clone, Debug)]
pub struct SelectionTicket {
    owner: Arc<()>,
    serial: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Leaf {
    Unexpanded,
    Terminal(f64),
}

#[derive(Clone, Debug)]
pub struct Selection<M> {
    pub ticket: SelectionTicket,
    /// Replay with checked Rules transitions to reconstruct this exact leaf and history.
    pub moves: Vec<M>,
    pub leaf: Leaf,
}

#[derive(Debug)]
struct Edge<M> {
    mv: M,
    stats: EdgeStats,
    child: Option<usize>,
}

#[derive(Debug)]
enum Node<M> {
    Unexpanded,
    Expanded(Vec<Edge<M>>),
    Terminal(f64),
}

struct Pending {
    ticket: SelectionTicket,
    path: Vec<(usize, usize)>,
    leaf: usize,
}

/// One-owner, one-selection reference tree. There is no board-only transposition table.
pub struct Tree<M, P = Puct> {
    nodes: Vec<Node<M>>,
    edge_count: usize,
    limits: TreeLimits,
    policy: P,
    owner: Arc<()>,
    next_serial: u64,
    pending: Option<Pending>,
    #[cfg(feature = "experimental-batch")]
    other_pending: Vec<Pending>,
    #[cfg(feature = "experimental-batch")]
    max_pending: usize,
    deadline: Option<Instant>,
    cancellation: Option<Arc<AtomicBool>>,
    counters: SearchCounters,
    #[cfg(feature = "experimental-search-buffers")]
    selection_stats: Vec<EdgeStats>,
    #[cfg(all(
        feature = "experimental-search-buffers",
        feature = "experimental-batch"
    ))]
    selection_indices: Vec<usize>,
}

impl<M: Clone + Eq> Tree<M, Puct> {
    pub fn baseline(limits: TreeLimits) -> Result<Self, SearchError> {
        Self::new(Puct::default(), limits)
    }
}

impl<M: Clone + Eq, P: SelectionPolicy> Tree<M, P> {
    pub fn new(policy: P, limits: TreeLimits) -> Result<Self, SearchError> {
        limits.validate()?;
        Ok(Self {
            nodes: vec![Node::Unexpanded],
            edge_count: 0,
            limits,
            policy,
            owner: Arc::new(()),
            next_serial: 0,
            pending: None,
            #[cfg(feature = "experimental-batch")]
            other_pending: Vec::new(),
            #[cfg(feature = "experimental-batch")]
            max_pending: 1,
            deadline: None,
            cancellation: None,
            counters: SearchCounters::default(),
            #[cfg(feature = "experimental-search-buffers")]
            selection_stats: Vec::new(),
            #[cfg(all(
                feature = "experimental-search-buffers",
                feature = "experimental-batch"
            ))]
            selection_indices: Vec::new(),
        })
    }

    pub fn policy_identity(&self) -> PolicyIdentity {
        self.policy.identity()
    }
    pub fn counters(&self) -> SearchCounters {
        self.counters
    }
    pub fn has_pending(&self) -> bool {
        self.pending_count() != 0
    }
    fn pending_count(&self) -> usize {
        let count = usize::from(self.pending.is_some());
        #[cfg(feature = "experimental-batch")]
        let count = count + self.other_pending.len();
        count
    }
    #[cfg(feature = "experimental-batch")]
    pub fn set_parallelism(&mut self, max_pending: usize) -> Result<(), SearchError> {
        if self.has_pending() {
            return Err(SearchError::Busy);
        }
        if !(1..=16).contains(&max_pending) {
            return Err(SearchError::InvalidConfiguration("parallelism 1..=16"));
        }
        self.other_pending
            .try_reserve(max_pending)
            .map_err(|_| SearchError::AllocationFailed)?;
        self.max_pending = max_pending;
        Ok(())
    }
    #[cfg(feature = "experimental-batch")]
    fn activate(&mut self, ticket: &SelectionTicket) {
        if !Arc::ptr_eq(&ticket.owner, &self.owner) {
            return;
        }
        if let Some(index) = self
            .other_pending
            .iter()
            .position(|p| p.ticket.serial == ticket.serial)
        {
            let selected = self.other_pending.swap_remove(index);
            if let Some(previous) = self.pending.replace(selected) {
                self.other_pending.push(previous);
            }
        }
    }

    pub fn set_deadline(&mut self, deadline: Instant) -> Result<(), SearchError> {
        if self.has_pending() {
            return Err(SearchError::Busy);
        }
        self.deadline = Some(deadline);
        Ok(())
    }

    pub fn set_cancellation(&mut self, cancellation: Arc<AtomicBool>) -> Result<(), SearchError> {
        if self.has_pending() {
            return Err(SearchError::Busy);
        }
        self.cancellation = Some(cancellation);
        Ok(())
    }

    /// Close old consume rights and start an empty root. External game/model IDs remain adapter-owned.
    pub fn reset(&mut self) -> Result<(), SearchError> {
        let mut counters = self.counters;
        if self.pending.is_some() {
            increment(&mut counters.reservations_released)?;
            increment(&mut counters.rejected_results)?;
        }
        #[cfg(feature = "experimental-batch")]
        for _ in &self.other_pending {
            increment(&mut counters.reservations_released)?;
            increment(&mut counters.rejected_results)?;
        }
        #[cfg(feature = "experimental-batch")]
        self.other_pending.clear();
        self.pending = None;
        self.owner = Arc::new(());
        self.next_serial = 0;
        self.nodes.clear();
        self.nodes.push(Node::Unexpanded);
        self.edge_count = 0;
        // The caller explicitly supplies a new session deadline after reset.
        self.deadline = None;
        self.cancellation = None;
        self.counters = counters;
        Ok(())
    }

    pub fn root_stats(&self) -> Vec<(M, EdgeStats)> {
        match &self.nodes[0] {
            Node::Expanded(edges) => edges.iter().map(|e| (e.mv.clone(), e.stats)).collect(),
            _ => Vec::new(),
        }
    }

    pub fn has_root_visits(&self) -> bool {
        matches!(&self.nodes[0], Node::Expanded(edges) if edges.iter().any(|edge| edge.stats.visits > 0))
    }

    /// None until root initialization or for an exact terminal root.
    pub fn best_move(&self) -> Option<&M> {
        let Node::Expanded(edges) = &self.nodes[0] else {
            return None;
        };
        let mut best = 0;
        for i in 1..edges.len() {
            let candidate = edges[i].stats;
            let current = edges[best].stats;
            if candidate.visits > current.visits
                || (candidate.visits == current.visits && candidate.q() > current.q())
            {
                best = i;
            }
        }
        Some(&edges[best].mv)
    }

    pub fn begin_selection(&mut self, now: Instant) -> Result<Selection<M>, SearchError> {
        #[cfg(feature = "experimental-batch")]
        {
            if self.pending_count() >= self.max_pending {
                return Err(SearchError::Busy);
            }
            if let Some(previous) = self.pending.take() {
                self.other_pending.push(previous);
            }
        }
        #[cfg(not(feature = "experimental-batch"))]
        if self.pending.is_some() {
            return Err(SearchError::Busy);
        }
        if self
            .cancellation
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire))
        {
            return Err(SearchError::Canceled);
        }
        if self.deadline.is_some_and(|deadline| now >= deadline) {
            return Err(SearchError::Expired);
        }
        let serial = self
            .next_serial
            .checked_add(1)
            .ok_or(SearchError::CounterOverflow)?;
        let mut counters = self.counters;
        increment(&mut counters.selections)?;
        let mut node = 0;
        let mut path = Vec::new();
        let mut moves = Vec::new();
        let leaf = loop {
            match &self.nodes[node] {
                Node::Unexpanded => {
                    #[cfg(feature = "experimental-batch")]
                    if self
                        .other_pending
                        .iter()
                        .any(|pending| pending.leaf == node)
                    {
                        return Err(SearchError::Busy);
                    }
                    break Leaf::Unexpanded;
                }
                Node::Terminal(value) => break Leaf::Terminal(*value),
                Node::Expanded(edges) => {
                    if path.len() >= self.limits.max_depth {
                        return Err(SearchError::DepthLimit);
                    }
                    #[cfg(feature = "experimental-batch")]
                    let serial = self.max_pending == 1;
                    #[cfg(not(feature = "experimental-batch"))]
                    let serial = true;
                    #[cfg(not(feature = "experimental-search-buffers"))]
                    let selected = if serial {
                        let stats: Vec<_> = edges.iter().map(|e| e.stats).collect();
                        self.policy.select(&stats)?
                    } else {
                        0
                    };
                    #[cfg(feature = "experimental-search-buffers")]
                    let selected = if serial {
                        self.selection_stats.clear();
                        self.selection_stats
                            .try_reserve(edges.len())
                            .map_err(|_| SearchError::AllocationFailed)?;
                        self.selection_stats
                            .extend(edges.iter().map(|edge| edge.stats));
                        self.policy.select(&self.selection_stats)?
                    } else {
                        0
                    };
                    #[cfg(feature = "experimental-batch")]
                    let selected = if self.max_pending > 1 {
                        #[cfg(not(feature = "experimental-search-buffers"))]
                        let (mut scratch_indices, mut scratch_stats) = (
                            Vec::with_capacity(edges.len()),
                            Vec::with_capacity(edges.len()),
                        );
                        #[cfg(not(feature = "experimental-search-buffers"))]
                        let (indices, stats) = (&mut scratch_indices, &mut scratch_stats);
                        #[cfg(feature = "experimental-search-buffers")]
                        let (indices, stats) = {
                            self.selection_indices.clear();
                            self.selection_stats.clear();
                            self.selection_indices
                                .try_reserve(edges.len())
                                .map_err(|_| SearchError::AllocationFailed)?;
                            self.selection_stats
                                .try_reserve(edges.len())
                                .map_err(|_| SearchError::AllocationFailed)?;
                            (&mut self.selection_indices, &mut self.selection_stats)
                        };
                        for (index, edge) in edges.iter().enumerate() {
                            if edge.child.is_some_and(|child| {
                                self.other_pending.iter().any(|p| p.leaf == child)
                                    && matches!(self.nodes[child], Node::Unexpanded)
                            }) {
                                continue;
                            }
                            let reserved = self
                                .other_pending
                                .iter()
                                .filter(|p| p.path.contains(&(node, index)))
                                .count() as u64;
                            let mut value = edge.stats;
                            value.visits = value
                                .visits
                                .checked_add(reserved)
                                .ok_or(SearchError::CounterOverflow)?;
                            value.value_sum -= reserved as f64;
                            indices.push(index);
                            stats.push(value);
                        }
                        if stats.is_empty() {
                            return Err(SearchError::Busy);
                        }
                        *indices
                            .get(self.policy.select(stats)?)
                            .ok_or(SearchError::InvalidPolicySelection)?
                    } else {
                        selected
                    };
                    let edge = edges
                        .get(selected)
                        .ok_or(SearchError::InvalidPolicySelection)?;
                    let mv = edge.mv.clone();
                    let existing_child = edge.child;
                    let child = match existing_child {
                        Some(child) => child,
                        None => {
                            if self.nodes.len() >= self.limits.max_nodes {
                                return Err(SearchError::NodeLimit);
                            }
                            self.nodes
                                .try_reserve(1)
                                .map_err(|_| SearchError::AllocationFailed)?;
                            let child = self.nodes.len();
                            self.nodes.push(Node::Unexpanded);
                            let Node::Expanded(edges) = &mut self.nodes[node] else {
                                unreachable!()
                            };
                            edges[selected].child = Some(child);
                            child
                        }
                    };
                    path.push((node, selected));
                    moves.push(mv);
                    node = child;
                }
            }
        };
        let ticket = SelectionTicket {
            owner: Arc::clone(&self.owner),
            serial,
        };
        self.pending = Some(Pending {
            ticket: ticket.clone(),
            path,
            leaf: node,
        });
        self.next_serial = serial;
        self.counters = counters;
        Ok(Selection {
            ticket,
            moves,
            leaf,
        })
    }

    fn ticket_rejection(&self, ticket: &SelectionTicket) -> Option<Rejection> {
        if !Arc::ptr_eq(&ticket.owner, &self.owner) {
            return Some(Rejection::Stale);
        }
        if self
            .pending
            .as_ref()
            .is_none_or(|pending| pending.ticket.serial != ticket.serial)
        {
            return Some(Rejection::AlreadyFinalized);
        }
        None
    }

    fn release(&mut self, rejection: Rejection) -> Result<Completion, SearchError> {
        let mut counters = self.counters;
        increment(&mut counters.reservations_released)?;
        increment(&mut counters.rejected_results)?;
        self.pending = None;
        self.counters = counters;
        Ok(Completion::Rejected(rejection))
    }

    fn check_acceptance(
        &mut self,
        ticket: &SelectionTicket,
        now: Instant,
    ) -> Result<Option<Completion>, SearchError> {
        #[cfg(feature = "experimental-batch")]
        self.activate(ticket);
        if let Some(rejection) = self.ticket_rejection(ticket) {
            return Ok(Some(Completion::Rejected(rejection)));
        }
        if self
            .cancellation
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire))
        {
            return self.release(Rejection::Canceled).map(Some);
        }
        if self.deadline.is_some_and(|deadline| now >= deadline) {
            return self.release(Rejection::Expired).map(Some);
        }
        Ok(None)
    }

    pub fn cancel(&mut self, ticket: &SelectionTicket) -> Result<Completion, SearchError> {
        #[cfg(feature = "experimental-batch")]
        self.activate(ticket);
        if let Some(rejection) = self.ticket_rejection(ticket) {
            return Ok(Completion::Rejected(rejection));
        }
        self.release(Rejection::Canceled)
    }

    fn validate_expansion(&self, moves: &[M], priors: &[f64]) -> Result<(), SearchError> {
        if moves.is_empty() {
            return Err(SearchError::NoLegalEdges);
        }
        if moves.len() != priors.len() {
            return Err(SearchError::InvalidPolicy);
        }
        if moves.len() > self.limits.max_legal_moves {
            return Err(SearchError::EdgeLimit);
        }
        let count = self
            .edge_count
            .checked_add(moves.len())
            .ok_or(SearchError::EdgeLimit)?;
        if count > self.limits.max_edges {
            return Err(SearchError::EdgeLimit);
        }
        for i in 0..moves.len() {
            if moves[..i].contains(&moves[i]) {
                return Err(SearchError::DuplicateMove);
            }
        }
        let mut sum = 0.0;
        for &prior in priors {
            if !prior.is_finite() || !(0.0..=1.0).contains(&prior) {
                return Err(SearchError::InvalidPolicy);
            }
            sum += prior;
        }
        if (sum - 1.0).abs() > self.limits.probability_tolerance {
            return Err(SearchError::InvalidPolicy);
        }
        Ok(())
    }

    /// The adapter supplies legal moves in the checked Rules order and v = W-L from the leaf's actual turn.
    pub fn accept_evaluation(
        &mut self,
        ticket: &SelectionTicket,
        moves: Vec<M>,
        priors: &[f64],
        value: f64,
        now: Instant,
    ) -> Result<Completion, SearchError> {
        self.accept_evaluation_with_guard(ticket, moves, priors, value, now, || true)
    }

    /// The contract adapter rechecks live generations, Rules authority, cancellation
    /// and its clock after all expansion/backup preparation. Rejection consumes the
    /// reservation once without publishing edges or visits. It does not drain a backend.
    pub fn accept_evaluation_with_guard(
        &mut self,
        ticket: &SelectionTicket,
        moves: Vec<M>,
        priors: &[f64],
        value: f64,
        now: Instant,
        admit: impl FnOnce() -> bool,
    ) -> Result<Completion, SearchError> {
        if let Some(completion) = self.check_acceptance(ticket, now)? {
            return Ok(completion);
        }
        let prepared = (|| {
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(SearchError::InvalidValue);
            }
            self.validate_expansion(&moves, priors)?;
            let pending = self.pending.as_ref().expect("checked live ticket");
            if !matches!(self.nodes[pending.leaf], Node::Unexpanded) {
                return Err(SearchError::InconsistentLeaf);
            }
            let backup = self.prepare_backup(value)?;
            let mut edges = Vec::new();
            edges
                .try_reserve_exact(moves.len())
                .map_err(|_| SearchError::AllocationFailed)?;
            for (mv, &prior) in moves.into_iter().zip(priors) {
                edges.push(Edge {
                    mv,
                    stats: EdgeStats {
                        prior,
                        visits: 0,
                        value_sum: 0.0,
                    },
                    child: None,
                });
            }
            Ok((edges, backup))
        })();
        match prepared {
            Ok((edges, backup)) => {
                let pending = self.pending.as_ref().expect("checked live ticket");
                let leaf = pending.leaf;
                let completion = self.commit_backup(backup, now, admit)?;
                if matches!(completion, Completion::Accepted { .. }) {
                    self.edge_count += edges.len();
                    self.nodes[leaf] = Node::Expanded(edges);
                }
                Ok(completion)
            }
            Err(error) => {
                self.release(Rejection::Canceled)?;
                Err(error)
            }
        }
    }

    /// Exact Rules outcome; bypasses the evaluator. The draw utility is fixed to zero in S0.
    pub fn accept_terminal(
        &mut self,
        ticket: &SelectionTicket,
        value: f64,
        now: Instant,
    ) -> Result<Completion, SearchError> {
        self.accept_terminal_with_guard(ticket, value, now, || true)
    }

    /// Exact terminal traversal still belongs to the current root and game.
    /// The final guard never creates an evaluation request for a terminal.
    pub fn accept_terminal_with_guard(
        &mut self,
        ticket: &SelectionTicket,
        value: f64,
        now: Instant,
        admit: impl FnOnce() -> bool,
    ) -> Result<Completion, SearchError> {
        if let Some(completion) = self.check_acceptance(ticket, now)? {
            return Ok(completion);
        }
        let prepared = (|| {
            if value != -1.0 && value != 0.0 && value != 1.0 {
                return Err(SearchError::InvalidValue);
            }
            let pending = self.pending.as_ref().expect("checked live ticket");
            match self.nodes[pending.leaf] {
                Node::Unexpanded => {}
                Node::Terminal(previous) if previous == value => {}
                _ => return Err(SearchError::InconsistentLeaf),
            }
            self.prepare_backup(value)
        })();
        match prepared {
            Ok(backup) => {
                let leaf = self.pending.as_ref().expect("checked live ticket").leaf;
                let completion = self.commit_backup(backup, now, admit)?;
                if matches!(completion, Completion::Accepted { .. }) {
                    self.nodes[leaf] = Node::Terminal(value);
                }
                Ok(completion)
            }
            Err(error) => {
                self.release(Rejection::Canceled)?;
                Err(error)
            }
        }
    }

    fn prepare_backup(
        &self,
        mut value: f64,
    ) -> Result<Vec<(usize, usize, EdgeStats)>, SearchError> {
        let pending = self.pending.as_ref().expect("checked live ticket");
        let mut updates = Vec::new();
        updates
            .try_reserve_exact(pending.path.len())
            .map_err(|_| SearchError::AllocationFailed)?;
        for &(node, edge) in pending.path.iter().rev() {
            value = -value;
            let Node::Expanded(edges) = &self.nodes[node] else {
                return Err(SearchError::InconsistentLeaf);
            };
            let mut stats = edges[edge].stats;
            stats.visits = stats
                .visits
                .checked_add(1)
                .ok_or(SearchError::CounterOverflow)?;
            stats.value_sum += value;
            if !stats.value_sum.is_finite() {
                return Err(SearchError::InvalidStatistics);
            }
            updates.push((node, edge, stats));
        }
        Ok(updates)
    }

    fn commit_backup(
        &mut self,
        updates: Vec<(usize, usize, EdgeStats)>,
        supplied_now: Instant,
        admit: impl FnOnce() -> bool,
    ) -> Result<Completion, SearchError> {
        let mut counters = self.counters;
        increment(&mut counters.reservations_released)?;
        if updates.is_empty() {
            increment(&mut counters.root_initializations)?;
        } else {
            increment(&mut counters.accepted_backups)?;
            increment(&mut counters.completed_visits)?;
        }
        // Validation, allocation and arithmetic are complete. This final owner check
        // is the backup linearization point; a provider's earlier timestamp cannot
        // authorize results that cross the deadline while being validated.
        let ticket = self
            .pending
            .as_ref()
            .expect("checked live ticket")
            .ticket
            .clone();
        if let Some(completion) =
            self.check_acceptance(&ticket, supplied_now.max(Instant::now()))?
        {
            return Ok(completion);
        }
        if !admit() {
            return self.release(Rejection::AdmissionClosed);
        }
        // The external guard can perform authority checks. A local deadline or
        // cancellation that changed during it must also leave the prepared tree untouched.
        if let Some(completion) =
            self.check_acceptance(&ticket, supplied_now.max(Instant::now()))?
        {
            return Ok(completion);
        }
        let count = updates.len();
        for (node, edge, stats) in updates {
            let Node::Expanded(edges) = &mut self.nodes[node] else {
                unreachable!()
            };
            edges[edge].stats = stats;
        }
        self.counters = counters;
        self.pending = None;
        Ok(Completion::Accepted {
            traversed_edges: count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn root(tree: &mut Tree<&'static str>, now: Instant, moves: Vec<&'static str>, priors: &[f64]) {
        let init = tree.begin_selection(now).unwrap();
        assert!(init.moves.is_empty());
        assert_eq!(
            tree.accept_evaluation(&init.ticket, moves, priors, 0.25, now)
                .unwrap(),
            Completion::Accepted { traversed_edges: 0 }
        );
        assert!(
            tree.root_stats()
                .iter()
                .all(|(_, stats)| stats.visits == 0 && stats.value_sum == 0.0)
        );
        assert_eq!(tree.counters().accepted_backups, 0);
    }

    #[test]
    fn one_ply_wdl_reference_values_and_duplicate_callback() {
        let now = Instant::now();
        // WDL scalar conversion is performed by the driver; these are independent W-L values.
        for (leaf, expected_parent) in [(-1.0, 1.0), (1.0, -1.0), (0.0, 0.0)] {
            let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
            root(&mut tree, now, vec!["a"], &[1.0]);
            let selection = tree.begin_selection(now).unwrap();
            assert_eq!(selection.moves, ["a"]);
            tree.accept_terminal(&selection.ticket, leaf, now).unwrap();
            let stats = tree.root_stats()[0].1;
            assert_eq!(stats.visits, 1);
            assert_eq!(stats.value_sum, expected_parent);
            assert_eq!(stats.q(), expected_parent);
            assert_eq!(
                tree.accept_terminal(&selection.ticket, leaf, now).unwrap(),
                Completion::Rejected(Rejection::AlreadyFinalized)
            );
            assert_eq!(tree.root_stats()[0].1, stats);
            assert_eq!(tree.counters().accepted_backups, 1);
            assert_eq!(tree.counters().reservations_released, 2); // root preparation + traversal
        }
    }

    #[test]
    fn two_ply_backup_alternates_each_actual_move() {
        let now = Instant::now();
        let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
        root(&mut tree, now, vec!["a"], &[1.0]);
        let child = tree.begin_selection(now).unwrap();
        tree.accept_evaluation(&child.ticket, vec!["b"], &[1.0], 0.0, now)
            .unwrap();
        let grandchild = tree.begin_selection(now).unwrap();
        assert_eq!(grandchild.moves, ["a", "b"]);
        // (.8,.15,.05) from the grandchild's turn: v=.75, child edge=-.75, root edge=+.75.
        tree.accept_evaluation(&grandchild.ticket, vec!["c"], &[1.0], 0.75, now)
            .unwrap();
        assert_eq!(tree.root_stats()[0].1.visits, 2);
        assert_eq!(tree.root_stats()[0].1.value_sum, 0.75);
        let Node::Expanded(edges) = &tree.nodes[1] else {
            panic!("expanded child")
        };
        assert_eq!(edges[0].stats.visits, 1);
        assert_eq!(edges[0].stats.value_sum, -0.75);
    }

    #[test]
    fn new_traversal_can_consume_the_same_exact_terminal_twice() {
        let now = Instant::now();
        let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
        root(&mut tree, now, vec!["a"], &[1.0]);
        let first = tree.begin_selection(now).unwrap();
        tree.accept_terminal(&first.ticket, -1.0, now).unwrap();
        let second = tree.begin_selection(now).unwrap();
        assert_eq!(second.leaf, Leaf::Terminal(-1.0));
        tree.accept_terminal(&second.ticket, -1.0, now).unwrap();
        assert_eq!(tree.root_stats()[0].1.visits, 2);
        assert_eq!(tree.root_stats()[0].1.value_sum, 2.0);
        assert_eq!(tree.counters().accepted_backups, 2);
    }

    #[test]
    fn cancel_and_exact_deadline_never_make_a_visit() {
        let now = Instant::now();
        let deadline = now + Duration::from_millis(10);
        let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
        tree.set_deadline(deadline).unwrap();
        root(&mut tree, now, vec!["a"], &[1.0]);
        let canceled = tree.begin_selection(now).unwrap();
        assert_eq!(tree.begin_selection(now).unwrap_err(), SearchError::Busy);
        assert_eq!(
            tree.cancel(&canceled.ticket).unwrap(),
            Completion::Rejected(Rejection::Canceled)
        );
        assert_eq!(
            tree.cancel(&canceled.ticket).unwrap(),
            Completion::Rejected(Rejection::AlreadyFinalized)
        );
        let late = tree
            .begin_selection(deadline - Duration::from_nanos(1))
            .unwrap();
        assert_eq!(
            tree.accept_terminal(&late.ticket, -1.0, deadline).unwrap(),
            Completion::Rejected(Rejection::Expired)
        );
        assert_eq!(tree.counters().accepted_backups, 0);
        assert_eq!(tree.counters().reservations_released, 3);
        assert_eq!(tree.root_stats()[0].1.visits, 0);
        assert_eq!(
            tree.begin_selection(deadline).unwrap_err(),
            SearchError::Expired
        );
    }

    #[test]
    fn reset_and_different_owner_reject_old_results() {
        let now = Instant::now();
        let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
        let old = tree.begin_selection(now).unwrap();
        tree.reset().unwrap();
        let current = tree.begin_selection(now).unwrap();
        assert_eq!(
            tree.accept_evaluation(&old.ticket, vec!["bad"], &[1.0], 0.0, now)
                .unwrap(),
            Completion::Rejected(Rejection::Stale)
        );
        assert!(tree.has_pending());
        let mut other: Tree<&str> = Tree::baseline(TreeLimits::default()).unwrap();
        let foreign = other.begin_selection(now).unwrap();
        assert_eq!(
            tree.accept_evaluation(&foreign.ticket, vec!["bad"], &[1.0], 0.0, now)
                .unwrap(),
            Completion::Rejected(Rejection::Stale)
        );
        tree.accept_evaluation(&current.ticket, vec!["ok"], &[1.0], 0.0, now)
            .unwrap();
        assert_eq!(tree.best_move(), Some(&"ok"));
    }

    #[test]
    fn invalid_expansion_releases_once_and_preserves_statistics() {
        let now = Instant::now();
        let bad = [
            (
                vec!["x", "x"],
                vec![0.5, 0.5],
                0.0,
                SearchError::DuplicateMove,
            ),
            (vec!["x"], vec![f64::NAN], 0.0, SearchError::InvalidPolicy),
            (vec!["x"], vec![0.5], 0.0, SearchError::InvalidPolicy),
            (vec!["x"], vec![], 0.0, SearchError::InvalidPolicy),
            (
                vec!["x"],
                vec![1.0],
                f64::INFINITY,
                SearchError::InvalidValue,
            ),
            (vec![], vec![], 0.0, SearchError::NoLegalEdges),
        ];
        for (moves, priors, value, expected) in bad {
            let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
            root(&mut tree, now, vec!["a"], &[1.0]);
            let selection = tree.begin_selection(now).unwrap();
            assert_eq!(
                tree.accept_evaluation(&selection.ticket, moves, &priors, value, now)
                    .unwrap_err(),
                expected
            );
            assert!(!tree.has_pending());
            assert_eq!(tree.counters().reservations_released, 2);
            assert_eq!(tree.root_stats()[0].1.visits, 0);
            tree.cancel(&selection.ticket).unwrap();
            assert_eq!(tree.counters().reservations_released, 2);
        }
    }

    #[test]
    fn exact_terminal_cannot_be_overwritten_by_neural_output() {
        let now = Instant::now();
        let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
        root(&mut tree, now, vec!["a"], &[1.0]);
        let first = tree.begin_selection(now).unwrap();
        tree.accept_terminal(&first.ticket, 0.0, now).unwrap();
        let second = tree.begin_selection(now).unwrap();
        assert_eq!(
            tree.accept_evaluation(&second.ticket, vec!["b"], &[1.0], 0.9, now),
            Err(SearchError::InconsistentLeaf)
        );
        assert_eq!(tree.root_stats()[0].1.visits, 1);
    }

    #[test]
    fn finite_node_edge_and_depth_limits() {
        let now = Instant::now();
        let mut tree = Tree::baseline(TreeLimits {
            max_nodes: 1,
            ..TreeLimits::default()
        })
        .unwrap();
        root(&mut tree, now, vec!["a"], &[1.0]);
        assert_eq!(
            tree.begin_selection(now).unwrap_err(),
            SearchError::NodeLimit
        );
        let mut tree = Tree::baseline(TreeLimits {
            max_edges: 1,
            ..TreeLimits::default()
        })
        .unwrap();
        let init = tree.begin_selection(now).unwrap();
        assert_eq!(
            tree.accept_evaluation(&init.ticket, vec!["a", "b"], &[0.5, 0.5], 0.0, now),
            Err(SearchError::EdgeLimit)
        );
        let mut tree = Tree::baseline(TreeLimits {
            max_depth: 1,
            ..TreeLimits::default()
        })
        .unwrap();
        root(&mut tree, now, vec!["a"], &[1.0]);
        let child = tree.begin_selection(now).unwrap();
        tree.accept_evaluation(&child.ticket, vec!["b"], &[1.0], 0.0, now)
            .unwrap();
        assert_eq!(
            tree.begin_selection(now).unwrap_err(),
            SearchError::DepthLimit
        );
    }

    #[test]
    fn best_move_uses_visits_then_q_then_legal_order() {
        let now = Instant::now();
        let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
        root(&mut tree, now, vec!["a", "b"], &[0.5, 0.5]);
        assert_eq!(tree.best_move(), Some(&"a"));
        // Hand-authored statistics, not computed using the backup implementation under test.
        let Node::Expanded(edges) = &mut tree.nodes[0] else {
            panic!("root")
        };
        edges[0].stats = EdgeStats {
            prior: 0.5,
            visits: 4,
            value_sum: 1.0,
        };
        edges[1].stats = EdgeStats {
            prior: 0.5,
            visits: 4,
            value_sum: 2.0,
        };
        assert_eq!(tree.best_move(), Some(&"b"));
        let Node::Expanded(edges) = &mut tree.nodes[0] else {
            panic!("root")
        };
        edges[0].stats.visits = 5;
        assert_eq!(tree.best_move(), Some(&"a"));
    }

    #[test]
    fn policy_can_be_replaced_without_owning_visit_accounting() {
        struct Last;
        impl SelectionPolicy for Last {
            fn identity(&self) -> PolicyIdentity {
                PolicyIdentity {
                    algorithm: "test-last",
                    revision: 1,
                    configuration: "fixture".into(),
                }
            }
            fn select(&self, edges: &[EdgeStats]) -> Result<usize, SearchError> {
                Ok(edges.len() - 1)
            }
        }
        let now = Instant::now();
        let mut tree = Tree::new(Last, TreeLimits::default()).unwrap();
        let init = tree.begin_selection(now).unwrap();
        tree.accept_evaluation(&init.ticket, vec!["a", "b"], &[0.9, 0.1], 0.0, now)
            .unwrap();
        let selected = tree.begin_selection(now).unwrap();
        assert_eq!(selected.moves, ["b"]);
        tree.accept_terminal(&selected.ticket, -1.0, now).unwrap();
        assert_eq!(tree.best_move(), Some(&"b"));
        assert_eq!(tree.policy_identity().algorithm, "test-last");
        assert_eq!(tree.counters().accepted_backups, 1);
    }

    #[test]
    fn validation_crossing_deadline_cannot_commit_a_visit_or_expansion() {
        #[derive(Clone, Debug)]
        struct SlowMove(u8);
        impl PartialEq for SlowMove {
            fn eq(&self, other: &Self) -> bool {
                std::thread::sleep(Duration::from_millis(30));
                self.0 == other.0
            }
        }
        impl Eq for SlowMove {}
        let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
        let start = Instant::now();
        let initial = tree.begin_selection(start).unwrap();
        tree.accept_evaluation(&initial.ticket, vec![SlowMove(0)], &[1.0], 0.0, start)
            .unwrap();
        let before = Instant::now();
        tree.set_deadline(before + Duration::from_millis(20))
            .unwrap();
        let child = tree.begin_selection(before).unwrap();
        let result = tree
            .accept_evaluation(
                &child.ticket,
                vec![SlowMove(1), SlowMove(2)],
                &[0.5, 0.5],
                -1.0,
                before,
            )
            .unwrap();
        assert_eq!(result, Completion::Rejected(Rejection::Expired));
        assert_eq!(tree.root_stats()[0].1.visits, 0);
        assert!(matches!(tree.nodes[1], Node::Unexpanded));
        assert_eq!(tree.counters().reservations_released, 2);
    }

    #[test]
    fn cancellation_during_validation_cannot_commit_a_visit() {
        #[derive(Clone, Debug)]
        struct CancelMove {
            value: u8,
            flag: Arc<AtomicBool>,
        }
        impl PartialEq for CancelMove {
            fn eq(&self, other: &Self) -> bool {
                self.flag.store(true, Ordering::Release);
                self.value == other.value
            }
        }
        impl Eq for CancelMove {}
        let flag = Arc::new(AtomicBool::new(false));
        let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
        tree.set_cancellation(Arc::clone(&flag)).unwrap();
        let now = Instant::now();
        let initial = tree.begin_selection(now).unwrap();
        tree.accept_evaluation(
            &initial.ticket,
            vec![CancelMove {
                value: 0,
                flag: Arc::clone(&flag),
            }],
            &[1.0],
            0.0,
            now,
        )
        .unwrap();
        let child = tree.begin_selection(now).unwrap();
        let legal = [1, 2]
            .map(|value| CancelMove {
                value,
                flag: Arc::clone(&flag),
            })
            .to_vec();
        assert_eq!(
            tree.accept_evaluation(&child.ticket, legal, &[0.5, 0.5], -1.0, now)
                .unwrap(),
            Completion::Rejected(Rejection::Canceled)
        );
        assert_eq!(tree.root_stats()[0].1.visits, 0);
        assert!(matches!(tree.nodes[1], Node::Unexpanded));
        assert_eq!(
            tree.cancel(&child.ticket).unwrap(),
            Completion::Rejected(Rejection::AlreadyFinalized)
        );
        assert_eq!(tree.counters().reservations_released, 2);
    }

    #[test]
    fn final_contract_guard_releases_once_without_publishing_expansion() {
        let now = Instant::now();
        let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
        root(&mut tree, now, vec!["a"], &[1.0]);
        let child = tree.begin_selection(now).unwrap();
        let mut called = 0;
        let completion = tree
            .accept_evaluation_with_guard(&child.ticket, vec!["b"], &[1.0], -0.75, now, || {
                called += 1;
                false
            })
            .unwrap();
        assert_eq!(called, 1);
        assert_eq!(completion, Completion::Rejected(Rejection::AdmissionClosed));
        assert_eq!(tree.root_stats()[0].1.visits, 0);
        assert!(matches!(tree.nodes[1], Node::Unexpanded));
        assert_eq!(tree.counters().reservations_released, 2);
        assert_eq!(tree.counters().accepted_backups, 0);
        assert_eq!(
            tree.cancel(&child.ticket).unwrap(),
            Completion::Rejected(Rejection::AlreadyFinalized)
        );
        let next = tree.begin_selection(now).unwrap();
        assert_eq!(next.leaf, Leaf::Unexpanded);
    }

    #[test]
    fn guarded_terminal_observes_local_cancellation_during_external_guard() {
        let now = Instant::now();
        let flag = Arc::new(AtomicBool::new(false));
        let mut tree = Tree::baseline(TreeLimits::default()).unwrap();
        tree.set_cancellation(Arc::clone(&flag)).unwrap();
        root(&mut tree, now, vec!["a"], &[1.0]);
        let child = tree.begin_selection(now).unwrap();
        let completion = tree
            .accept_terminal_with_guard(&child.ticket, -1.0, now, || {
                flag.store(true, Ordering::Release);
                true
            })
            .unwrap();
        assert_eq!(completion, Completion::Rejected(Rejection::Canceled));
        assert_eq!(tree.root_stats()[0].1.visits, 0);
        assert!(matches!(tree.nodes[1], Node::Unexpanded));
        assert_eq!(tree.counters().reservations_released, 2);
    }
}
