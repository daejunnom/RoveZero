//! 잠정 B 평가 연결 경계. 공통 계약이 게시되면 A/C/D adapter로 통일한다.
//!
//! 이 driver는 규칙이나 backend를 구현하지 않는다. CheckedPosition이 소유한 정확한
//! 상태를 매 selection마다 복원하고, terminal을 평가 요청 전에 처리한다. evaluator는
//! deadline/취소를 준수하고 물리 작업·buffer의 수명을 직접 소유해야 한다. 취소할 수
//! 없는 backend를 무기한 join하는 wrapper는 이 동기 경계의 올바른 구현이 아니다.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use crate::policy::{EdgeStats, PolicyIdentity, Puct, SelectionPolicy};
use crate::time::TimeBudget;
use crate::tree::{
    Completion, Leaf, SearchCounters, SearchError, SelectionTicket, Tree, TreeLimits,
};

/// Rules가 검증한 불변 상태만 이 경계에 전달한다. claim 가능성은 terminal이 아니다.
pub trait CheckedPosition: Clone {
    type Move: Clone + Eq;
    type Error;

    /// 정확한 side-to-move terminal utility: loss=-1, draw=0, win=1.
    fn classify(&self) -> Result<Option<f64>, Self::Error>;
    /// Rules의 중복 없는, 결정적인 합법 수 순서. backend action index와 구별한다.
    fn legal_moves(&self) -> Result<Vec<Self::Move>, Self::Error>;
    /// 원본을 변경하지 않는 checked transition. 불법 수는 반드시 실패한다.
    fn play(&self, chess_move: &Self::Move) -> Result<Self, Self::Error>;
}

pub trait Evaluator<P: CheckedPosition> {
    type Error;

    /// policy는 요청 legal 순서 그대로이며 WDL은 이 position의 실제 차례 관점이다.
    fn evaluate(
        &mut self,
        position: &P,
        legal: &[P::Move],
        control: &SearchControl,
    ) -> Result<Evaluation, Self::Error>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    pub priors: Vec<f64>,
    pub wdl: [f64; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationError {
    InvalidTolerance,
    PolicyLength,
    InvalidPolicyProbability,
    PolicySum,
    InvalidWdlProbability,
    WdlSum,
}

impl Evaluation {
    /// TreeLimits와 같은 확률 합 허용 오차를 사용하며 값을 정규화하지 않는다.
    pub fn validate(&self, legal_count: usize, tolerance: f64) -> Result<f64, EvaluationError> {
        if !tolerance.is_finite() || !(0.0..=1e-3).contains(&tolerance) {
            return Err(EvaluationError::InvalidTolerance);
        }
        if self.priors.len() != legal_count {
            return Err(EvaluationError::PolicyLength);
        }
        validate_probabilities(
            &self.priors,
            tolerance,
            EvaluationError::InvalidPolicyProbability,
            EvaluationError::PolicySum,
        )?;
        validate_probabilities(
            &self.wdl,
            tolerance,
            EvaluationError::InvalidWdlProbability,
            EvaluationError::WdlSum,
        )?;
        Ok(self.wdl[0] - self.wdl[2])
    }
}

fn validate_probabilities(
    values: &[f64],
    tolerance: f64,
    invalid: EvaluationError,
    sum_error: EvaluationError,
) -> Result<(), EvaluationError> {
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(invalid);
    }
    let sum: f64 = values.iter().sum();
    if !sum.is_finite() || (sum - 1.0).abs() > tolerance {
        return Err(sum_error);
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct SearchControl {
    pub cancellation: Arc<AtomicBool>,
    /// 결과 수락의 엄격한 경계. soft/admission 종료는 in-flight 결과를 무효화하지 않는다.
    pub deadline: Instant,
    pub soft_deadline: Instant,
    pub admission_deadline: Instant,
    /// root prior 초기화는 traversal이 아니므로 이 예산에 포함하지 않는다.
    pub max_simulations: u64,
}

impl SearchControl {
    pub fn new(deadline: Instant, max_simulations: u64) -> Self {
        Self {
            cancellation: Arc::new(AtomicBool::new(false)),
            deadline,
            soft_deadline: deadline,
            admission_deadline: deadline,
            max_simulations,
        }
    }

    pub fn from_budget(budget: &TimeBudget, max_simulations: u64) -> Self {
        Self {
            cancellation: Arc::new(AtomicBool::new(false)),
            deadline: budget.hard_deadline,
            soft_deadline: budget.soft_deadline,
            admission_deadline: budget.admission_deadline,
            max_simulations,
        }
    }

    pub fn cancel(&self) {
        self.cancellation.store(true, Ordering::Release);
    }

    pub fn stop_reason(&self, now: Instant) -> Option<StopReason> {
        if self.cancellation.load(Ordering::Acquire) {
            Some(StopReason::Canceled)
        } else if now >= self.deadline {
            Some(StopReason::Deadline)
        } else {
            None
        }
    }

    /// 새 selection과 평가 제출에만 적용한다. 이미 실행 중인 결과에는 hard만 적용한다.
    pub fn selection_stop_reason(&self, now: Instant) -> Option<StopReason> {
        self.stop_reason(now).or_else(|| {
            if now >= self.admission_deadline {
                Some(StopReason::AdmissionClosed)
            } else if now >= self.soft_deadline {
                Some(StopReason::SoftBudget)
            } else {
                None
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReason {
    Canceled,
    Deadline,
    SoftBudget,
    AdmissionClosed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionStage {
    Classification,
    LegalMoves,
    Transition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionViolation {
    InvalidTerminalUtility,
    NonterminalWithoutLegalMoves,
    DuplicateLegalMove,
    LegalMoveLimit,
    ClassificationChanged,
}

#[derive(Debug)]
pub enum SearchFailure<PositionError, EvaluatorError> {
    Position {
        stage: PositionStage,
        source: PositionError,
    },
    PositionContract(PositionViolation),
    Evaluation(EvaluatorError),
    InvalidEvaluation(EvaluationError),
    Tree(SearchError),
    /// 취소 정리 자체도 실패했으면 원래 오류와 정리 오류를 함께 보존한다.
    Cleanup {
        primary: Option<Box<Self>>,
        source: SearchError,
    },
    UnexpectedRejection,
}

#[derive(Debug)]
pub enum SearchStatus<PositionError, EvaluatorError> {
    Completed,
    Stopped(StopReason),
    Terminal(f64),
    Failed(SearchFailure<PositionError, EvaluatorError>),
}

#[derive(Debug)]
pub struct SearchOutcome<M, PositionError, EvaluatorError> {
    pub best_move: Option<M>,
    pub status: SearchStatus<PositionError, EvaluatorError>,
    pub fallback_used: bool,
    pub counters: SearchCounters,
    pub root_stats: Vec<(M, EdgeStats)>,
    /// backend 함수 호출 수다. cache hit/물리 network 실행 수라고 해석하지 않는다.
    pub evaluator_calls: u64,
    pub policy_identity: PolicyIdentity,
    /// evaluator 오류와 동시에 발생한 취소/만료도 원래 오류를 숨기지 않고 기록한다.
    pub observed_stop: Option<StopReason>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgressStage {
    BeforeSelection,
    EvaluationReady,
    Completed,
}

pub struct SearchProgress<M> {
    pub stage: ProgressStage,
    pub best_move: Option<M>,
    pub counters: SearchCounters,
    pub completed_simulations: u64,
    pub evaluator_calls: u64,
}

pub fn run_search<P, E>(
    position: &P,
    evaluator: &mut E,
    control: &SearchControl,
    limits: TreeLimits,
) -> SearchOutcome<P::Move, P::Error, E::Error>
where
    P: CheckedPosition,
    E: Evaluator<P>,
{
    run_search_with_policy_and_progress(
        position,
        evaluator,
        control,
        limits,
        Puct::default(),
        |_| {},
    )
}

pub fn run_search_with_progress<P, E, F>(
    position: &P,
    evaluator: &mut E,
    control: &SearchControl,
    limits: TreeLimits,
    progress: F,
) -> SearchOutcome<P::Move, P::Error, E::Error>
where
    P: CheckedPosition,
    E: Evaluator<P>,
    F: FnMut(&SearchProgress<P::Move>),
{
    run_search_with_policy_and_progress(
        position,
        evaluator,
        control,
        limits,
        Puct::default(),
        progress,
    )
}

pub fn run_search_with_policy<P, E, S>(
    position: &P,
    evaluator: &mut E,
    control: &SearchControl,
    limits: TreeLimits,
    policy: S,
) -> SearchOutcome<P::Move, P::Error, E::Error>
where
    P: CheckedPosition,
    E: Evaluator<P>,
    S: SelectionPolicy,
{
    run_search_with_policy_and_progress(position, evaluator, control, limits, policy, |_| {})
}

pub fn run_search_with_policy_and_progress<P, E, S, F>(
    position: &P,
    evaluator: &mut E,
    control: &SearchControl,
    limits: TreeLimits,
    policy: S,
    mut progress: F,
) -> SearchOutcome<P::Move, P::Error, E::Error>
where
    P: CheckedPosition,
    E: Evaluator<P>,
    S: SelectionPolicy,
    F: FnMut(&SearchProgress<P::Move>),
{
    let mut state = DriverState {
        tree: None,
        fallback: None,
        evaluator_calls: 0,
        completed_simulations: 0,
        policy_identity: policy.identity(),
    };
    // Even an already stopped search checks Rules once to obtain a legal fallback or terminal.
    match position.classify() {
        Err(source) => {
            return state.finish(
                SearchStatus::Failed(SearchFailure::Position {
                    stage: PositionStage::Classification,
                    source,
                }),
                control,
            );
        }
        Ok(Some(value)) if valid_terminal(value) => {
            return state.finish(SearchStatus::Terminal(value), control);
        }
        Ok(Some(_)) => {
            return state.finish(
                SearchStatus::Failed(SearchFailure::PositionContract(
                    PositionViolation::InvalidTerminalUtility,
                )),
                control,
            );
        }
        Ok(None) => {}
    }
    let root_moves = match position.legal_moves() {
        Ok(moves) => moves,
        Err(source) => {
            return state.finish(
                SearchStatus::Failed(SearchFailure::Position {
                    stage: PositionStage::LegalMoves,
                    source,
                }),
                control,
            );
        }
    };
    if root_moves.len() > limits.max_legal_moves {
        // A search resource ceiling does not make Rules' first move illegal. Verify that
        // move independently without scanning the over-limit array for policy expansion.
        if let Some(chess_move) = root_moves.first() {
            match position.play(chess_move) {
                Ok(_) => state.fallback = Some(chess_move.clone()),
                Err(source) => {
                    return state.finish(
                        SearchStatus::Failed(SearchFailure::Position {
                            stage: PositionStage::Transition,
                            source,
                        }),
                        control,
                    );
                }
            }
        }
        return state.finish(
            SearchStatus::Failed(SearchFailure::PositionContract(
                PositionViolation::LegalMoveLimit,
            )),
            control,
        );
    }
    if let Err(error) = checked_legal(&root_moves, limits.max_legal_moves) {
        return state.finish(
            SearchStatus::Failed(SearchFailure::PositionContract(error)),
            control,
        );
    }
    state.fallback = root_moves.first().cloned();
    state.tree = match Tree::new(policy, limits) {
        Ok(mut tree) => {
            if let Err(error) = tree.set_deadline(control.deadline) {
                return state.finish(SearchStatus::Failed(SearchFailure::Tree(error)), control);
            }
            if let Err(error) = tree.set_cancellation(Arc::clone(&control.cancellation)) {
                return state.finish(SearchStatus::Failed(SearchFailure::Tree(error)), control);
            }
            Some(tree)
        }
        Err(error) => {
            return state.finish(SearchStatus::Failed(SearchFailure::Tree(error)), control);
        }
    };

    loop {
        state.report(ProgressStage::BeforeSelection, &mut progress);
        if let Some(reason) = control.selection_stop_reason(Instant::now()) {
            return state.finish(SearchStatus::Stopped(reason), control);
        }
        if state.completed_simulations >= control.max_simulations {
            return state.finish(SearchStatus::Completed, control);
        }
        let selection = match state.tree_mut().begin_selection(Instant::now()) {
            Ok(selection) => selection,
            Err(SearchError::Expired) => {
                return state.finish(SearchStatus::Stopped(StopReason::Deadline), control);
            }
            Err(SearchError::Canceled) => {
                return state.finish(SearchStatus::Stopped(StopReason::Canceled), control);
            }
            Err(error) => {
                return state.finish(SearchStatus::Failed(SearchFailure::Tree(error)), control);
            }
        };
        let mut leaf_position = position.clone();
        for chess_move in &selection.moves {
            leaf_position = match leaf_position.play(chess_move) {
                Ok(next) => next,
                Err(source) => {
                    return state.fail_pending(
                        &selection.ticket,
                        SearchFailure::Position {
                            stage: PositionStage::Transition,
                            source,
                        },
                        control,
                    );
                }
            };
        }
        let terminal = match leaf_position.classify() {
            Ok(value) => value,
            Err(source) => {
                return state.fail_pending(
                    &selection.ticket,
                    SearchFailure::Position {
                        stage: PositionStage::Classification,
                        source,
                    },
                    control,
                );
            }
        };
        if terminal.is_some_and(|value| !valid_terminal(value)) {
            return state.fail_pending(
                &selection.ticket,
                SearchFailure::PositionContract(PositionViolation::InvalidTerminalUtility),
                control,
            );
        }
        if let Leaf::Terminal(expected) = selection.leaf {
            if terminal != Some(expected) {
                return state.fail_pending(
                    &selection.ticket,
                    SearchFailure::PositionContract(PositionViolation::ClassificationChanged),
                    control,
                );
            }
        }
        if let Some(reason) = control.stop_reason(Instant::now()) {
            return state.stop_pending(&selection.ticket, reason, control);
        }

        let completion = if let Some(value) = terminal {
            state
                .tree_mut()
                .accept_terminal(&selection.ticket, value, Instant::now())
        } else {
            let legal = if selection.moves.is_empty() {
                root_moves.clone()
            } else {
                match leaf_position.legal_moves() {
                    Ok(moves) => moves,
                    Err(source) => {
                        return state.fail_pending(
                            &selection.ticket,
                            SearchFailure::Position {
                                stage: PositionStage::LegalMoves,
                                source,
                            },
                            control,
                        );
                    }
                }
            };
            if let Err(error) = checked_legal(&legal, limits.max_legal_moves) {
                return state.fail_pending(
                    &selection.ticket,
                    SearchFailure::PositionContract(error),
                    control,
                );
            }
            if let Some(reason) = control.selection_stop_reason(Instant::now()) {
                return state.stop_pending(&selection.ticket, reason, control);
            }
            state.evaluator_calls = match state.evaluator_calls.checked_add(1) {
                Some(count) => count,
                None => {
                    return state.fail_pending(
                        &selection.ticket,
                        SearchFailure::Tree(SearchError::CounterOverflow),
                        control,
                    );
                }
            };
            let evaluation = match evaluator.evaluate(&leaf_position, &legal, control) {
                Ok(value) => value,
                Err(source) => {
                    return state.fail_pending(
                        &selection.ticket,
                        SearchFailure::Evaluation(source),
                        control,
                    );
                }
            };
            state.report(ProgressStage::EvaluationReady, &mut progress);
            if let Some(reason) = control.stop_reason(Instant::now()) {
                return state.stop_pending(&selection.ticket, reason, control);
            }
            let value = match evaluation.validate(legal.len(), limits.probability_tolerance) {
                Ok(value) => value,
                Err(error) => {
                    return state.fail_pending(
                        &selection.ticket,
                        SearchFailure::InvalidEvaluation(error),
                        control,
                    );
                }
            };
            state.tree_mut().accept_evaluation(
                &selection.ticket,
                legal,
                &evaluation.priors,
                value,
                Instant::now(),
            )
        };
        match completion {
            Ok(Completion::Accepted { traversed_edges }) => {
                if traversed_edges > 0 {
                    state.completed_simulations = match state.completed_simulations.checked_add(1) {
                        Some(count) => count,
                        None => {
                            return state.finish(
                                SearchStatus::Failed(SearchFailure::Tree(
                                    SearchError::CounterOverflow,
                                )),
                                control,
                            );
                        }
                    };
                }
            }
            Ok(Completion::Rejected(_)) => {
                if let Some(reason) = control.stop_reason(Instant::now()) {
                    return state.finish(SearchStatus::Stopped(reason), control);
                }
                return state.finish(
                    SearchStatus::Failed(SearchFailure::UnexpectedRejection),
                    control,
                );
            }
            Err(error) => {
                return state.fail_pending(&selection.ticket, SearchFailure::Tree(error), control);
            }
        }
        state.report(ProgressStage::Completed, &mut progress);
    }
}

fn valid_terminal(value: f64) -> bool {
    value == -1.0 || value == 0.0 || value == 1.0
}

fn checked_legal<M: Eq>(moves: &[M], maximum: usize) -> Result<(), PositionViolation> {
    if moves.is_empty() {
        return Err(PositionViolation::NonterminalWithoutLegalMoves);
    }
    if moves.len() > maximum {
        return Err(PositionViolation::LegalMoveLimit);
    }
    for (index, chess_move) in moves.iter().enumerate() {
        if moves[..index].contains(chess_move) {
            return Err(PositionViolation::DuplicateLegalMove);
        }
    }
    Ok(())
}

struct DriverState<M: Clone + Eq, S: SelectionPolicy> {
    tree: Option<Tree<M, S>>,
    fallback: Option<M>,
    evaluator_calls: u64,
    completed_simulations: u64,
    policy_identity: PolicyIdentity,
}

impl<M: Clone + Eq, S: SelectionPolicy> DriverState<M, S> {
    fn tree_mut(&mut self) -> &mut Tree<M, S> {
        // Established once before the selection loop; no external input controls this invariant.
        self.tree
            .as_mut()
            .expect("driver initializes tree before selection")
    }

    fn report<F: FnMut(&SearchProgress<M>)>(&self, stage: ProgressStage, callback: &mut F) {
        callback(&SearchProgress {
            stage,
            best_move: self
                .tree
                .as_ref()
                .and_then(|tree| tree.best_move().cloned())
                .or_else(|| self.fallback.clone()),
            counters: self
                .tree
                .as_ref()
                .map_or_else(SearchCounters::default, |tree| tree.counters()),
            completed_simulations: self.completed_simulations,
            evaluator_calls: self.evaluator_calls,
        });
    }

    fn finish<PE, EE>(
        self,
        status: SearchStatus<PE, EE>,
        control: &SearchControl,
    ) -> SearchOutcome<M, PE, EE> {
        let (root_stats, counters, visited_best) = match &self.tree {
            Some(tree) => {
                let stats = tree.root_stats();
                let visited = stats.iter().any(|(_, edge)| edge.visits > 0);
                (
                    stats,
                    tree.counters(),
                    visited.then(|| tree.best_move().cloned()).flatten(),
                )
            }
            None => (Vec::new(), SearchCounters::default(), None),
        };
        let fallback_used = visited_best.is_none() && self.fallback.is_some();
        SearchOutcome {
            best_move: visited_best.or(self.fallback),
            status,
            fallback_used,
            counters,
            root_stats,
            evaluator_calls: self.evaluator_calls,
            policy_identity: self.policy_identity,
            observed_stop: control.selection_stop_reason(Instant::now()),
        }
    }

    fn fail_pending<PE, EE>(
        mut self,
        ticket: &SelectionTicket,
        primary: SearchFailure<PE, EE>,
        control: &SearchControl,
    ) -> SearchOutcome<M, PE, EE> {
        let failure = match self.tree_mut().cancel(ticket) {
            Ok(_) => primary,
            Err(source) => SearchFailure::Cleanup {
                primary: Some(Box::new(primary)),
                source,
            },
        };
        self.finish(SearchStatus::Failed(failure), control)
    }

    fn stop_pending<PE, EE>(
        mut self,
        ticket: &SelectionTicket,
        reason: StopReason,
        control: &SearchControl,
    ) -> SearchOutcome<M, PE, EE> {
        let status = match self.tree_mut().cancel(ticket) {
            Ok(_) => SearchStatus::Stopped(reason),
            Err(source) => SearchStatus::Failed(SearchFailure::Cleanup {
                primary: None,
                source,
            }),
        };
        self.finish(status, control)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[derive(Clone)]
    struct Position {
        node: usize,
        fixture: Arc<Vec<Node>>,
    }

    struct Node {
        terminal: Option<f64>,
        edges: Vec<(&'static str, usize)>,
        evaluation: Result<Evaluation, &'static str>,
    }

    impl CheckedPosition for Position {
        type Move = &'static str;
        type Error = &'static str;

        fn classify(&self) -> Result<Option<f64>, Self::Error> {
            Ok(self.fixture[self.node].terminal)
        }

        fn legal_moves(&self) -> Result<Vec<Self::Move>, Self::Error> {
            Ok(self.fixture[self.node]
                .edges
                .iter()
                .map(|(chess_move, _)| *chess_move)
                .collect())
        }

        fn play(&self, chess_move: &Self::Move) -> Result<Self, Self::Error> {
            let next = self.fixture[self.node]
                .edges
                .iter()
                .find(|(candidate, _)| candidate == chess_move)
                .map(|(_, target)| *target)
                .ok_or("illegal fixture move")?;
            Ok(Self {
                node: next,
                fixture: Arc::clone(&self.fixture),
            })
        }
    }

    #[derive(Default)]
    struct FixtureEvaluator {
        calls: usize,
    }

    impl Evaluator<Position> for FixtureEvaluator {
        type Error = &'static str;

        fn evaluate(
            &mut self,
            position: &Position,
            _: &[&'static str],
            _: &SearchControl,
        ) -> Result<Evaluation, Self::Error> {
            self.calls += 1;
            position.fixture[position.node].evaluation.clone()
        }
    }

    fn evaluation(priors: &[f64], wdl: [f64; 3]) -> Result<Evaluation, &'static str> {
        Ok(Evaluation {
            priors: priors.to_vec(),
            wdl,
        })
    }

    fn ongoing(edges: &[(&'static str, usize)], wdl: [f64; 3]) -> Node {
        let prior = 1.0 / edges.len() as f64;
        Node {
            terminal: None,
            edges: edges.to_vec(),
            evaluation: evaluation(&vec![prior; edges.len()], wdl),
        }
    }

    fn terminal(value: f64) -> Node {
        Node {
            terminal: Some(value),
            edges: Vec::new(),
            evaluation: Err("terminal must not evaluate"),
        }
    }

    fn position(nodes: Vec<Node>) -> Position {
        Position {
            node: 0,
            fixture: Arc::new(nodes),
        }
    }

    fn control(simulations: u64) -> SearchControl {
        SearchControl::new(Instant::now() + Duration::from_secs(5), simulations)
    }

    #[test]
    fn independent_terminal_selection_vector_and_zero_visit_root_initialization() {
        // Equal priors select A first. A is a win for the child: parent -1.
        // B is a loss for the child: parent +1, and PUCT selects B again.
        let root = position(vec![
            ongoing(&[("A", 1), ("B", 2)], [0.2, 0.6, 0.2]),
            terminal(1.0),
            terminal(-1.0),
        ]);
        let mut evaluator = FixtureEvaluator::default();
        let result = run_search(&root, &mut evaluator, &control(3), TreeLimits::default());
        assert!(matches!(result.status, SearchStatus::Completed));
        assert_eq!(result.best_move, Some("B"));
        assert!(!result.fallback_used);
        assert_eq!(evaluator.calls, 1);
        assert_eq!(result.evaluator_calls, 1);
        assert_eq!(result.counters.root_initializations, 1);
        assert_eq!(result.counters.accepted_backups, 3);
        assert_eq!(result.counters.reservations_released, 4);
        assert_eq!(
            result.root_stats,
            vec![
                (
                    "A",
                    EdgeStats {
                        prior: 0.5,
                        visits: 1,
                        value_sum: -1.0
                    }
                ),
                (
                    "B",
                    EdgeStats {
                        prior: 0.5,
                        visits: 2,
                        value_sum: 2.0
                    }
                ),
            ]
        );
    }

    #[test]
    fn independent_one_ply_wdl_vectors() {
        for (wdl, expected) in [
            ([0.0, 0.0, 1.0], 1.0),
            ([1.0, 0.0, 0.0], -1.0),
            ([0.0, 1.0, 0.0], 0.0),
        ] {
            let root = position(vec![
                ongoing(&[("A", 1)], [0.0, 1.0, 0.0]),
                ongoing(&[("later", 2)], wdl),
                terminal(0.0),
            ]);
            let result = run_search(
                &root,
                &mut FixtureEvaluator::default(),
                &control(1),
                TreeLimits::default(),
            );
            assert!(matches!(result.status, SearchStatus::Completed));
            assert_eq!(result.root_stats[0].1.visits, 1);
            assert_eq!(result.root_stats[0].1.value_sum, expected);
            assert_eq!(result.counters.accepted_backups, 1);
            assert_eq!(result.evaluator_calls, 2);
        }
    }

    #[test]
    fn independent_two_ply_wdl_sign_restoration() {
        // First traversal backs up a draw at depth one. Second backs up +0.75
        // from depth two, so root N=2 and W=+0.75 rather than -0.75.
        let root = position(vec![
            ongoing(&[("A", 1)], [0.0, 1.0, 0.0]),
            ongoing(&[("B", 2)], [0.0, 1.0, 0.0]),
            ongoing(&[("C", 3)], [0.8, 0.15, 0.05]),
            terminal(0.0),
        ]);
        let result = run_search(
            &root,
            &mut FixtureEvaluator::default(),
            &control(2),
            TreeLimits::default(),
        );
        assert!(matches!(result.status, SearchStatus::Completed));
        assert_eq!(result.root_stats[0].1.visits, 2);
        assert!((result.root_stats[0].1.value_sum - 0.75).abs() < 1e-12);
        assert_eq!(result.counters.completed_visits, 2);
        assert_eq!(result.evaluator_calls, 3);
    }

    #[test]
    fn exact_root_terminal_and_empty_nonterminal_have_different_outcomes() {
        let mut evaluator = FixtureEvaluator::default();
        let result = run_search(
            &position(vec![terminal(-1.0)]),
            &mut evaluator,
            &control(10),
            TreeLimits::default(),
        );
        assert!(matches!(result.status, SearchStatus::Terminal(-1.0)));
        assert_eq!(result.best_move, None);
        assert!(!result.fallback_used);
        assert_eq!(evaluator.calls, 0);
        let invalid = position(vec![Node {
            terminal: None,
            edges: Vec::new(),
            evaluation: Err("must not evaluate"),
        }]);
        let result = run_search(
            &invalid,
            &mut evaluator,
            &control(10),
            TreeLimits::default(),
        );
        assert!(matches!(
            result.status,
            SearchStatus::Failed(SearchFailure::PositionContract(
                PositionViolation::NonterminalWithoutLegalMoves
            ))
        ));
        assert_eq!(result.best_move, None);
        assert_eq!(evaluator.calls, 0);
    }

    #[test]
    fn evaluator_failure_preserves_source_and_legal_fallback() {
        let mut bad_root = ongoing(&[("A", 1)], [0.0, 1.0, 0.0]);
        bad_root.evaluation = Err("injected backend failure");
        let root = position(vec![bad_root, terminal(0.0)]);
        let result = run_search(
            &root,
            &mut FixtureEvaluator::default(),
            &control(1),
            TreeLimits::default(),
        );
        assert!(matches!(
            result.status,
            SearchStatus::Failed(SearchFailure::Evaluation("injected backend failure"))
        ));
        assert_eq!(result.best_move, Some("A"));
        assert!(result.fallback_used);
        assert_eq!(result.counters.accepted_backups, 0);
        assert_eq!(result.counters.reservations_released, 1);
    }

    #[test]
    fn nan_bad_shape_and_non_normalized_evaluations_fail_without_backup() {
        let cases = [
            (
                Evaluation {
                    priors: vec![f64::NAN],
                    wdl: [0.0, 1.0, 0.0],
                },
                EvaluationError::InvalidPolicyProbability,
            ),
            (
                Evaluation {
                    priors: vec![1.0],
                    wdl: [f64::INFINITY, 0.0, 0.0],
                },
                EvaluationError::InvalidWdlProbability,
            ),
            (
                Evaluation {
                    priors: vec![],
                    wdl: [0.0, 1.0, 0.0],
                },
                EvaluationError::PolicyLength,
            ),
            (
                Evaluation {
                    priors: vec![0.9],
                    wdl: [0.0, 1.0, 0.0],
                },
                EvaluationError::PolicySum,
            ),
            (
                Evaluation {
                    priors: vec![1.0],
                    wdl: [0.2, 0.2, 0.2],
                },
                EvaluationError::WdlSum,
            ),
        ];
        for (evaluation, expected) in cases {
            let mut bad_root = ongoing(&[("A", 1)], [0.0, 1.0, 0.0]);
            bad_root.evaluation = Ok(evaluation);
            let result = run_search(
                &position(vec![bad_root, terminal(0.0)]),
                &mut FixtureEvaluator::default(),
                &control(1),
                TreeLimits::default(),
            );
            match result.status {
                SearchStatus::Failed(SearchFailure::InvalidEvaluation(actual)) => {
                    assert_eq!(actual, expected)
                }
                unexpected => panic!("expected numeric failure, got {unexpected:?}"),
            }
            assert!(result.fallback_used);
            assert_eq!(result.counters.root_initializations, 0);
            assert_eq!(result.counters.accepted_backups, 0);
            assert_eq!(result.counters.reservations_released, 1);
        }
    }

    #[test]
    fn callback_cancellation_after_child_result_releases_without_backup() {
        let root = position(vec![
            ongoing(&[("A", 1)], [0.0, 1.0, 0.0]),
            ongoing(&[("B", 2)], [1.0, 0.0, 0.0]),
            terminal(0.0),
        ]);
        let control = control(10);
        let result = run_search_with_progress(
            &root,
            &mut FixtureEvaluator::default(),
            &control,
            TreeLimits::default(),
            |progress| {
                if progress.stage == ProgressStage::EvaluationReady && progress.evaluator_calls == 2
                {
                    control.cancel();
                }
            },
        );
        assert!(matches!(
            result.status,
            SearchStatus::Stopped(StopReason::Canceled)
        ));
        assert_eq!(result.best_move, Some("A"));
        assert!(result.fallback_used);
        assert_eq!(result.counters.root_initializations, 1);
        assert_eq!(result.counters.accepted_backups, 0);
        assert_eq!(result.counters.reservations_released, 2);
        assert_eq!(result.root_stats[0].1.visits, 0);
    }

    #[test]
    fn cancellation_after_completed_visit_preserves_last_valid_move() {
        let root = position(vec![
            ongoing(&[("A", 1), ("B", 2)], [0.0, 1.0, 0.0]),
            terminal(-1.0),
            terminal(1.0),
        ]);
        let control = control(10);
        let result = run_search_with_progress(
            &root,
            &mut FixtureEvaluator::default(),
            &control,
            TreeLimits::default(),
            |progress| {
                if progress.stage == ProgressStage::Completed && progress.completed_simulations == 1
                {
                    control.cancel();
                }
            },
        );
        assert!(matches!(
            result.status,
            SearchStatus::Stopped(StopReason::Canceled)
        ));
        assert_eq!(result.best_move, Some("A"));
        assert!(!result.fallback_used);
        assert_eq!(result.counters.accepted_backups, 1);
    }

    #[test]
    fn expired_search_and_zero_simulation_budget_do_not_call_evaluator() {
        let root = position(vec![ongoing(&[("A", 1)], [0.0, 1.0, 0.0]), terminal(0.0)]);
        let expired = SearchControl::new(Instant::now(), 10);
        let result = run_search(
            &root,
            &mut FixtureEvaluator::default(),
            &expired,
            TreeLimits::default(),
        );
        assert!(matches!(
            result.status,
            SearchStatus::Stopped(StopReason::Deadline)
        ));
        assert!(result.fallback_used);
        assert_eq!(result.evaluator_calls, 0);
        let result = run_search(
            &root,
            &mut FixtureEvaluator::default(),
            &control(0),
            TreeLimits::default(),
        );
        assert!(matches!(result.status, SearchStatus::Completed));
        assert_eq!(result.evaluator_calls, 0);
        assert_eq!(result.best_move, Some("A"));
    }

    #[test]
    fn delayed_result_after_deadline_is_not_root_initialization() {
        struct Delayed;
        impl Evaluator<Position> for Delayed {
            type Error = &'static str;
            fn evaluate(
                &mut self,
                _: &Position,
                _: &[&'static str],
                control: &SearchControl,
            ) -> Result<Evaluation, Self::Error> {
                while Instant::now() < control.deadline {
                    std::thread::park_timeout(Duration::from_millis(1));
                }
                Ok(Evaluation {
                    priors: vec![1.0],
                    wdl: [0.0, 1.0, 0.0],
                })
            }
        }
        let root = position(vec![ongoing(&[("A", 1)], [0.0, 1.0, 0.0]), terminal(0.0)]);
        let control = SearchControl::new(Instant::now() + Duration::from_millis(20), 10);
        let result = run_search(&root, &mut Delayed, &control, TreeLimits::default());
        assert!(matches!(
            result.status,
            SearchStatus::Stopped(StopReason::Deadline)
        ));
        assert_eq!(result.best_move, Some("A"));
        assert!(result.fallback_used);
        assert_eq!(result.counters.root_initializations, 0);
        assert_eq!(result.counters.accepted_backups, 0);
        assert_eq!(result.counters.reservations_released, 1);
    }

    #[test]
    fn finite_depth_limit_returns_failure_and_last_valid_move() {
        let root = position(vec![
            ongoing(&[("A", 1)], [0.0, 1.0, 0.0]),
            ongoing(&[("B", 1)], [0.0, 1.0, 0.0]),
        ]);
        let limits = TreeLimits {
            max_depth: 1,
            ..TreeLimits::default()
        };
        let result = run_search(
            &root,
            &mut FixtureEvaluator::default(),
            &control(100),
            limits,
        );
        assert!(matches!(
            result.status,
            SearchStatus::Failed(SearchFailure::Tree(SearchError::DepthLimit))
        ));
        assert_eq!(result.best_move, Some("A"));
        assert!(!result.fallback_used);
        assert_eq!(result.counters.accepted_backups, 1);
    }

    #[test]
    fn root_legal_width_ceiling_retains_independently_checked_fallback() {
        let root = position(vec![
            ongoing(&[("A", 1), ("B", 2)], [0.0, 1.0, 0.0]),
            terminal(-1.0),
            terminal(1.0),
        ]);
        let limits = TreeLimits {
            max_legal_moves: 1,
            ..TreeLimits::default()
        };
        let result = run_search(&root, &mut FixtureEvaluator::default(), &control(1), limits);
        assert!(matches!(
            result.status,
            SearchStatus::Failed(SearchFailure::PositionContract(
                PositionViolation::LegalMoveLimit
            ))
        ));
        assert_eq!(result.best_move, Some("A"));
        assert!(result.fallback_used);
        assert_eq!(result.evaluator_calls, 0);
        assert_eq!(result.counters.selections, 0);
    }

    #[test]
    fn duplicate_legal_moves_and_invalid_terminal_are_explicit_errors() {
        let root = position(vec![
            ongoing(&[("A", 1), ("A", 1)], [0.0, 1.0, 0.0]),
            terminal(0.0),
        ]);
        let result = run_search(
            &root,
            &mut FixtureEvaluator::default(),
            &control(1),
            TreeLimits::default(),
        );
        assert!(matches!(
            result.status,
            SearchStatus::Failed(SearchFailure::PositionContract(
                PositionViolation::DuplicateLegalMove
            ))
        ));
        assert_eq!(result.best_move, None);
        let result = run_search(
            &position(vec![terminal(0.5)]),
            &mut FixtureEvaluator::default(),
            &control(1),
            TreeLimits::default(),
        );
        assert!(matches!(
            result.status,
            SearchStatus::Failed(SearchFailure::PositionContract(
                PositionViolation::InvalidTerminalUtility
            ))
        ));
    }

    #[test]
    fn policy_switch_changes_selection_and_records_identity() {
        struct LastLegal;
        impl SelectionPolicy for LastLegal {
            fn identity(&self) -> PolicyIdentity {
                PolicyIdentity {
                    algorithm: "fixture-last-legal",
                    revision: 7,
                    configuration: "independent-selection-fixture".into(),
                }
            }
            fn select(&self, edges: &[EdgeStats]) -> Result<usize, SearchError> {
                edges.len().checked_sub(1).ok_or(SearchError::NoLegalEdges)
            }
        }
        let root = position(vec![
            ongoing(&[("A", 1), ("B", 2)], [0.0, 1.0, 0.0]),
            terminal(-1.0),
            terminal(-1.0),
        ]);
        let baseline = run_search(
            &root,
            &mut FixtureEvaluator::default(),
            &control(1),
            TreeLimits::default(),
        );
        let alternative = run_search_with_policy(
            &root,
            &mut FixtureEvaluator::default(),
            &control(1),
            TreeLimits::default(),
            LastLegal,
        );
        assert_eq!(baseline.best_move, Some("A"));
        assert_eq!(alternative.best_move, Some("B"));
        assert_eq!(baseline.policy_identity.algorithm, "rz-puct");
        assert_eq!(alternative.policy_identity.algorithm, "fixture-last-legal");
        assert_eq!(alternative.policy_identity.revision, 7);
        assert_eq!(alternative.root_stats[0].1.visits, 0);
        assert_eq!(alternative.root_stats[1].1.visits, 1);
        assert_eq!(alternative.root_stats[1].1.value_sum, 1.0);
    }

    #[test]
    fn budget_projection_and_exclusive_admission_acceptance_boundaries() {
        let start = Instant::now();
        let budget = TimeBudget {
            start,
            soft_deadline: start + Duration::from_millis(50),
            admission_deadline: start + Duration::from_millis(100),
            hard_deadline: start + Duration::from_millis(150),
            output_deadline: start + Duration::from_millis(160),
            allocation: Duration::from_millis(160),
        };
        let control = SearchControl::from_budget(&budget, 17);
        assert_eq!(control.deadline, budget.hard_deadline);
        assert_eq!(control.soft_deadline, budget.soft_deadline);
        assert_eq!(control.admission_deadline, budget.admission_deadline);
        assert_eq!(control.max_simulations, 17);
        assert_eq!(
            control.selection_stop_reason(start + Duration::from_millis(49)),
            None
        );
        assert_eq!(
            control.selection_stop_reason(budget.soft_deadline),
            Some(StopReason::SoftBudget)
        );
        assert_eq!(
            control.selection_stop_reason(budget.admission_deadline),
            Some(StopReason::AdmissionClosed)
        );
        assert_eq!(control.stop_reason(budget.admission_deadline), None);
        assert_eq!(
            control.stop_reason(budget.hard_deadline),
            Some(StopReason::Deadline)
        );
        control.cancel();
        assert_eq!(control.stop_reason(start), Some(StopReason::Canceled));
    }

    #[test]
    fn in_flight_results_after_soft_and_admission_boundaries_are_still_accepted() {
        struct UntilBoundary {
            admission: bool,
        }
        impl Evaluator<Position> for UntilBoundary {
            type Error = &'static str;
            fn evaluate(
                &mut self,
                _: &Position,
                _: &[&'static str],
                control: &SearchControl,
            ) -> Result<Evaluation, Self::Error> {
                let boundary = if self.admission {
                    control.admission_deadline
                } else {
                    control.soft_deadline
                };
                while Instant::now() < boundary {
                    std::thread::park_timeout(Duration::from_millis(1));
                }
                Ok(Evaluation {
                    priors: vec![1.0],
                    wdl: [0.0, 1.0, 0.0],
                })
            }
        }
        for (admission, expected) in [
            (false, StopReason::SoftBudget),
            (true, StopReason::AdmissionClosed),
        ] {
            let start = Instant::now();
            let budget = TimeBudget {
                start,
                soft_deadline: start + Duration::from_millis(50),
                admission_deadline: start + Duration::from_millis(100),
                hard_deadline: start + Duration::from_secs(5),
                output_deadline: start + Duration::from_secs(6),
                allocation: Duration::from_secs(6),
            };
            let control = SearchControl::from_budget(&budget, 10);
            let root = position(vec![ongoing(&[("A", 1)], [0.0, 1.0, 0.0]), terminal(0.0)]);
            let result = run_search(
                &root,
                &mut UntilBoundary { admission },
                &control,
                TreeLimits::default(),
            );
            assert!(matches!(result.status, SearchStatus::Stopped(actual) if actual == expected));
            assert_eq!(result.evaluator_calls, 1);
            assert_eq!(result.counters.root_initializations, 1);
            assert_eq!(result.counters.accepted_backups, 0);
            assert_eq!(result.counters.reservations_released, 1);
            assert!(result.fallback_used);
        }
    }
}
