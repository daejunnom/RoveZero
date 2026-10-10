//! Native CPU checkpoint ownership at admission and logical cancellation.
//! Historical task evidence does not certify that a backend frame still lives.

use super::*;
use crate::cpu::CpuResumePolicy;
use std::collections::BTreeSet;

impl<M: RoleModel> PalsEngine<M> {
    fn cpu_stack_discard_required(&self) -> Result<bool, PalsError> {
        let policy = self.cpu.owned_resume_policy();
        let supports_discard = self.cpu.supports_paused_stack_discard();
        let stack_token = self
            .nodes
            .iter()
            .filter_map(|node| node.resume.as_ref())
            .any(CpuResumeToken::is_paused_stack);
        let observed_stack = self.cpu.paused_stack_snapshot().is_some_and(|snapshot| {
            snapshot.tokens_retained != 0 || snapshot.bytes_current != 0 || !snapshot.complete
        });
        // Disposal capability alone does not select or retain a stack: the
        // native CompletedIteration owner deliberately has no stack snapshot.
        // A custom capable owner with no disclosed policy remains unknown and
        // still needs the original disposal witness.
        let required = policy == Some(CpuResumePolicy::PausedStack)
            || stack_token
            || observed_stack
            || (supports_discard && policy.is_none());
        if required && !supports_discard {
            return Err(CheckerError::Unsupported("owned paused stack disposal").into());
        }
        Ok(required)
    }

    /// Before every own CPU admission, reconcile retained task metadata with
    /// the actual owner. A fresh analysis may have invalidated another Node's
    /// single live stack since the previous admission, even with archive off.
    /// This neither observes new work nor resets an attempt's measured work.
    pub(super) fn synchronize_cpu_checkpoints(&mut self) -> Result<(), PalsError> {
        if self.is_external() {
            return Ok(());
        }
        let discard_required = self.cpu_stack_discard_required()?;
        let mut current = BTreeSet::new();
        let mut stale = Vec::new();
        stale
            .try_reserve_exact(self.nodes.len())
            .map_err(|_| StoreError::Capacity("CPU checkpoint synchronization"))?;
        for (index, node) in self.nodes.iter().enumerate() {
            let Some(token) = node.resume.as_ref() else {
                continue;
            };
            let mut latest = None;
            if self.cpu.token_is_current(token) {
                for execution in self.stores.tasks.paused_executions_for_state(node.state)? {
                    let task = self.stores.tasks.get(execution)?;
                    if task.key.line.is_none()
                        && task.key.question == TaskQuestion::AnalyzePosition
                        && task.key.root_moves.is_empty()
                        && task.key.value_identity.as_ref() == self.cpu_registered_value.as_ref()
                        && task.key.cpu_condition.as_ref() == Some(&self.cpu_registered_condition)
                        && task
                            .key
                            .checker_identity
                            .as_ref()
                            .is_none_or(|identity| identity == &self.checker_registered_identity)
                        && matches!(task.status, TaskStatus::Paused { checkpoint, .. }
                            if checkpoint == index as u64)
                    {
                        latest =
                            Some(latest.map_or(execution, |old: ExecutionId| old.max(execution)));
                    }
                }
            }
            if let Some(execution) = latest {
                current.insert(execution);
            } else {
                stale.push(index);
            }
        }
        if current.is_empty() && discard_required {
            // Restricted discovery can leave frames without a resumable Node.
            // Such an orphan has no live task consumer to preserve.
            self.discard_cpu_stack_owner()?;
        }
        // Unknown or retained physical ownership above must not retire the
        // logical checkpoint. Store preflights before changing any statuses.
        self.stores.tasks.retire_paused_except(&current)?;
        for index in stale {
            self.nodes[index].resume = None;
        }
        Ok(())
    }

    /// Called at the logical canceled search boundary, including cancellation
    /// during progress or a model role after an earlier CPU pause. This never
    /// invokes checker new_game/shutdown or the model's physical lease owner.
    pub(super) fn discard_canceled_cpu_checkpoints(&mut self) -> Result<(), PalsError> {
        if self.is_external() {
            return Ok(());
        }
        if self.cpu_stack_discard_required()? {
            self.discard_cpu_stack_owner()?;
        }
        // Drop logical handles even if the finite Store retirement allocation
        // fails. The physical native frames have already been disposed above.
        for node in &mut self.nodes {
            node.resume = None;
        }
        self.stores.tasks.retire_paused_except(&BTreeSet::new())?;
        Ok(())
    }

    fn discard_cpu_stack_owner(&mut self) -> Result<(), PalsError> {
        self.cpu.discard_paused_stack();
        // An advertised custom implementation must actually invalidate known
        // frame owners; a false return alone proves no physical completion.
        if self
            .nodes
            .iter()
            .filter_map(|node| node.resume.as_ref())
            .any(|token| token.is_paused_stack() && self.cpu.token_is_current(token))
        {
            return Err(CheckerError::Invalid("paused stack disposal retained its owner").into());
        }
        let snapshot = self
            .cpu
            .paused_stack_snapshot()
            .ok_or(CheckerError::Unsupported(
                "paused stack disposal completion is unknown",
            ))?;
        if snapshot.tokens_retained != 0 || snapshot.bytes_current != 0 {
            return Err(CheckerError::Invalid("paused stack disposal retained its owner").into());
        }
        if !snapshot.complete {
            return Err(
                CheckerError::Invalid("paused stack disposal completion is unknown").into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{CpuCapabilities, CpuConfig, CpuWork};
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex};

    struct Trace {
        tokens: Mutex<Vec<CpuResumeToken>>,
        discards: AtomicUsize,
        supports_discard: AtomicBool,
        honors_discard: AtomicBool,
        snapshot_mode: AtomicUsize,
    }

    impl Default for Trace {
        fn default() -> Self {
            Self {
                tokens: Mutex::new(Vec::new()),
                discards: AtomicUsize::new(0),
                supports_discard: AtomicBool::new(true),
                honors_discard: AtomicBool::new(true),
                snapshot_mode: AtomicUsize::new(0),
            }
        }
    }

    struct RecordingCpu {
        inner: CpuEngine,
        trace: Arc<Trace>,
    }

    impl RecordingCpu {
        fn record(&self, result: Result<CpuReport, CpuError>) -> Result<CpuReport, CpuError> {
            if let Ok(report) = &result
                && let Some(token) = &report.resume
            {
                self.trace.tokens.lock().unwrap().push(token.clone());
            }
            result
        }
    }

    impl CpuSearcher for RecordingCpu {
        fn config(&self) -> &CpuConfig {
            self.inner.config()
        }
        fn value_identity(&self) -> &CpuValueIdentity {
            self.inner.value_identity()
        }
        fn search_identity(&self) -> &'static str {
            self.inner.search_identity()
        }
        fn search_conditions(&self) -> String {
            self.inner.search_conditions()
        }
        fn capabilities(&self) -> CpuCapabilities {
            CpuSearcher::capabilities(&self.inner)
        }
        fn resume_policy(&self) -> CpuResumePolicy {
            self.inner.resume_policy()
        }
        fn token_is_current(&self, token: &CpuResumeToken) -> bool {
            self.inner.token_is_current(token)
        }
        fn paused_stack_bytes(&self) -> Option<usize> {
            self.inner.paused_stack_bytes()
        }
        fn paused_stack_snapshot(&self) -> Option<crate::cpu::CpuPausedStackSnapshot> {
            let mut snapshot = self.inner.paused_stack_snapshot()?;
            match self.trace.snapshot_mode.load(Ordering::Acquire) {
                1 => None,
                2 => {
                    snapshot.complete = false;
                    Some(snapshot)
                }
                _ => Some(snapshot),
            }
        }
        fn supports_paused_stack_discard(&self) -> bool {
            self.trace.supports_discard.load(Ordering::Acquire)
        }
        fn discard_paused_stack(&mut self) -> bool {
            self.trace.discards.fetch_add(1, Ordering::Relaxed);
            self.trace.honors_discard.load(Ordering::Acquire) && self.inner.discard_paused_stack()
        }
        fn last_attempt_work(&self) -> Option<CpuWork> {
            self.inner.last_attempt_work()
        }
        fn clear(&mut self) {
            self.inner.clear();
        }
        fn analyze(
            &mut self,
            position: &Position,
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let result = self.inner.analyze(position, limits, cancel);
            self.record(result)
        }
        fn analyze_root_moves(
            &mut self,
            position: &Position,
            moves: &[BoardMove],
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let result = self
                .inner
                .analyze_root_moves(position, moves, limits, cancel);
            self.record(result)
        }
        fn analyze_divergence(
            &mut self,
            position: &Position,
            prefix: &[BoardMove],
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let result = self
                .inner
                .analyze_divergence(position, prefix, limits, cancel);
            self.record(result)
        }
        fn resume(
            &mut self,
            position: &Position,
            token: &CpuResumeToken,
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let result = self.inner.resume(position, token, limits, cancel);
            self.record(result)
        }
    }

    fn engine<M: RoleModel>(model: M) -> (PalsEngine<M>, Arc<Trace>) {
        engine_with_policy(model, CpuResumePolicy::PausedStack)
    }

    fn engine_with_policy<M: RoleModel>(
        model: M,
        policy: CpuResumePolicy,
    ) -> (PalsEngine<M>, Arc<Trace>) {
        let trace = Arc::new(Trace::default());
        let cpu = RecordingCpu {
            inner: CpuEngine::with_resume_policy(
                CpuConfig {
                    tt_entries: 64,
                    max_depth: 4,
                    quiescence_ply: 4,
                    ..CpuConfig::default()
                },
                policy,
            )
            .unwrap(),
            trace: trace.clone(),
        };
        // Exercise both Box<CpuSearcher> and OwnedCpuChecker forwarding.
        let engine = PalsEngine::new_with_boxed_cpu(
            PalsConfig {
                beam_width: 1,
                line_plies: 2,
                max_nodes: 257,
                max_records: 32,
                max_role_calls: 64,
                cpu_nodes_per_task: 1,
            },
            model,
            Box::new(cpu),
        )
        .unwrap();
        (engine, trace)
    }

    fn limits() -> PalsLimits {
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(20),
            max_rounds: 1,
            max_cpu_nodes: 32,
            cpu_depth: 2,
        }
    }

    fn root<M: RoleModel>(engine: &mut PalsEngine<M>, position: &Position) -> usize {
        engine
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        engine.intern(position.clone()).unwrap()
    }

    #[test]
    fn completed_iteration_without_stack_skips_disposal_witness_at_admission_and_cancel() {
        let (mut engine, trace) =
            engine_with_policy(LegalOrderRoleMock, CpuResumePolicy::CompletedIteration);
        assert!(engine.cpu.supports_paused_stack_discard());
        assert_eq!(engine.cpu.paused_stack_snapshot(), None);
        engine.synchronize_cpu_checkpoints().unwrap();
        let node = root(&mut engine, &Position::startpos());
        let cancel = AtomicBool::new(false);
        let limits = limits();
        let mut counters = PalsCounters::default();
        engine
            .verify_checked_cpu(node, &[], limits, &cancel, &mut counters)
            .unwrap();
        assert_eq!(counters.cpu_tasks_requested, 1);
        assert_eq!(counters.cpu_nodes, 1);
        let observation = engine.nodes[node]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap()
            .0;
        let raw = engine.stores.observations.get(observation).unwrap().clone();
        let work = engine.cpu.last_attempt().unwrap().work;
        let before = counters;
        engine.synchronize_cpu_checkpoints().unwrap();
        engine.discard_canceled_cpu_checkpoints().unwrap();
        assert_eq!(trace.discards.load(Ordering::Acquire), 0);
        assert_eq!(engine.cpu.paused_stack_snapshot(), None);
        assert_eq!(engine.cpu.last_attempt().unwrap().work, work);
        assert_eq!(engine.stores.observations.get(observation).unwrap(), &raw);
        assert_eq!(counters, before);
        engine.try_new_game(limits.deadline, &cancel).unwrap();
        engine.synchronize_cpu_checkpoints().unwrap();
        assert_eq!(trace.discards.load(Ordering::Acquire), 0);
        assert_eq!(engine.cpu.paused_stack_snapshot(), None);
    }

    #[test]
    fn archive_off_a_pause_b_fresh_a_same_key_starts_without_recharging_history() {
        let (mut engine, _) = engine(LegalOrderRoleMock);
        assert!(!engine.archive_enabled());
        let position = Position::startpos();
        let a = root(&mut engine, &position);
        let mut position_b = position.clone();
        position_b
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let b = engine.intern(position_b).unwrap();
        let cancel = AtomicBool::new(false);
        let limits = limits();
        let mut counters = PalsCounters::default();
        engine
            .verify_checked_cpu(a, &[], limits, &cancel, &mut counters)
            .unwrap();
        let token_a = engine.nodes[a].resume.clone().unwrap();
        let (old_observation, old_execution) = engine.nodes[a]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap();
        let old_fact = engine
            .stores
            .observations
            .get(old_observation)
            .unwrap()
            .clone();
        let old_key = engine.stores.tasks.get(old_execution).unwrap().key.clone();
        assert_eq!(old_fact.budget, 1);
        engine
            .verify_checked_cpu(b, &[], limits, &cancel, &mut counters)
            .unwrap();
        assert!(!engine.cpu.token_is_current(&token_a));
        // No explicit synchronization here: actual CPU admission must perform it.
        engine
            .verify_checked_cpu(a, &[], limits, &cancel, &mut counters)
            .unwrap();
        let (_, new_execution) = engine.nodes[a]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap();
        let new_task = engine.stores.tasks.get(new_execution).unwrap();
        assert_ne!(new_execution, old_execution);
        assert_eq!(new_task.key, old_key);
        assert!(new_task.resumed_from.is_none());
        assert!(
            engine
                .cpu
                .token_is_current(engine.nodes[a].resume.as_ref().unwrap())
        );
        assert!(
            matches!(engine.stores.tasks.get(old_execution).unwrap().status,
            TaskStatus::RetiredPaused { evidence: Some(id), .. } if id == old_observation)
        );
        assert_eq!(
            engine.stores.observations.get(old_observation).unwrap(),
            &old_fact
        );
        assert_eq!(counters.cpu_tasks_requested, 3);
        assert_eq!(counters.cpu_nodes, 3);
        let before = counters;
        let work = engine.cpu.last_attempt().unwrap().work;
        engine.synchronize_cpu_checkpoints().unwrap();
        assert!(engine.nodes[b].resume.is_none());
        assert_eq!(counters, before);
        assert_eq!(engine.cpu.last_attempt().unwrap().work, work);
    }

    #[test]
    fn current_a_resumes_once_and_only_latest_paused_execution_remains_live() {
        let (mut engine, _) = engine(LegalOrderRoleMock);
        let node = root(&mut engine, &Position::startpos());
        let mut counters = PalsCounters::default();
        let cancel = AtomicBool::new(false);
        let limits = limits();
        engine
            .verify_checked_cpu(node, &[], limits, &cancel, &mut counters)
            .unwrap();
        let first_token = engine.nodes[node].resume.clone().unwrap();
        let (observation, previous) = engine.nodes[node]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap();
        let fact = engine.stores.observations.get(observation).unwrap().clone();
        engine
            .verify_checked_cpu(node, &[], limits, &cancel, &mut counters)
            .unwrap();
        let (_, execution) = engine.nodes[node]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap();
        let token = engine.nodes[node].resume.as_ref().unwrap();
        assert!(!engine.cpu.token_is_current(&first_token));
        assert!(engine.cpu.token_is_current(token));
        assert_eq!(token.cumulative_work().unwrap().nodes, 2);
        assert_eq!(
            engine.stores.tasks.get(execution).unwrap().resumed_from,
            Some(previous)
        );
        engine.synchronize_cpu_checkpoints().unwrap();
        assert!(matches!(
            engine.stores.tasks.get(previous).unwrap().status,
            TaskStatus::RetiredPaused { .. }
        ));
        assert!(matches!(
            engine.stores.tasks.get(execution).unwrap().status,
            TaskStatus::Paused { .. }
        ));
        assert_eq!(engine.stores.observations.get(observation).unwrap(), &fact);
        assert_eq!(counters.cpu_nodes, 2);
    }

    fn assert_canceled_owner<M: RoleModel>(
        engine: &PalsEngine<M>,
        trace: &Trace,
        report: &PalsResult,
    ) {
        assert_eq!(report.completion, PalsCompletion::Canceled);
        assert!(report.counters.cpu_tasks_requested > 0);
        assert_eq!(
            report.counters.cpu_nodes,
            report.counters.cpu_tasks_requested
        );
        let tokens = trace.tokens.lock().unwrap();
        assert!(
            !tokens.is_empty(),
            "cancellation must follow an actual CPU pause"
        );
        assert!(
            tokens
                .iter()
                .all(|token| !engine.cpu.token_is_current(token))
        );
        assert!(engine.nodes.iter().all(|node| node.resume.is_none()));
        let mut retired = 0;
        for state in engine.nodes.iter().map(|node| node.state) {
            assert!(
                engine
                    .stores
                    .tasks
                    .paused_executions_for_state(state)
                    .unwrap()
                    .is_empty()
            );
        }
        for execution in 0..engine.stores.tasks.len() {
            if let TaskStatus::RetiredPaused {
                evidence: Some(observation),
                ..
            } = engine
                .stores
                .tasks
                .get(ExecutionId(execution))
                .unwrap()
                .status
            {
                retired += 1;
                let fact = engine.stores.observations.get(observation).unwrap();
                assert_eq!(fact.budget, 1);
                assert_eq!(fact.execution, Some(ExecutionId(execution)));
            }
        }
        assert!(retired > 0);
        assert_eq!(engine.cpu.last_attempt().unwrap().work.nodes, Some(1));
    }

    #[test]
    fn cancel_from_progress_after_cpu_pause_disposes_frames_and_keeps_raw_facts() {
        let (mut engine, trace) = engine(LegalOrderRoleMock);
        let cancel = AtomicBool::new(false);
        let report = engine
            .search_with_progress(&Position::startpos(), limits(), &cancel, |_| {
                cancel.store(true, Ordering::Release);
            })
            .unwrap();
        assert!(
            cancel.load(Ordering::Acquire),
            "progress must have run after verification"
        );
        assert_canceled_owner(&engine, &trace, &report);
        let work = engine.cpu.last_attempt().unwrap().work;
        let observations = engine.stores.observations.len();
        engine.discard_canceled_cpu_checkpoints().unwrap();
        assert_eq!(engine.cpu.last_attempt().unwrap().work, work);
        assert_eq!(engine.stores.observations.len(), observations);
        cancel.store(false, Ordering::Release);
        assert!(
            engine
                .search(&Position::startpos(), limits(), &cancel)
                .is_ok()
        );
    }

    struct CancelOnReply;

    impl RoleModel for CancelOnReply {
        fn identity(&self) -> &str {
            LegalOrderRoleMock.identity()
        }
        fn value_identity(&self) -> Option<&ModelValueIdentity> {
            LegalOrderRoleMock.value_identity()
        }
        fn evaluate_value(&mut self, query: RoleQuery<'_>) -> Result<ModelValueOutput, RoleError> {
            LegalOrderRoleMock.evaluate_value(query)
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            LegalOrderRoleMock.propose(query)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let cancel = query.cancel;
            let output = LegalOrderRoleMock::evaluate(query)?;
            cancel.store(true, Ordering::Release);
            Ok(output)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            LegalOrderRoleMock.repair(query)
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            LegalOrderRoleMock.divergences(query)
        }
    }

    #[test]
    fn cancel_in_critic_after_cpu_discovery_disposes_orphan_frames() {
        let (mut engine, trace) = engine(CancelOnReply);
        let cancel = AtomicBool::new(false);
        let report = engine
            .search(&Position::startpos(), limits(), &cancel)
            .unwrap();
        assert!(report.counters.critic_calls > 0);
        assert_canceled_owner(&engine, &trace, &report);
    }

    #[test]
    fn custom_paused_owner_must_explicitly_support_and_honor_disposal() {
        let (mut engine, trace) = engine(LegalOrderRoleMock);
        let node = root(&mut engine, &Position::startpos());
        let mut counters = PalsCounters::default();
        engine
            .verify_checked_cpu(node, &[], limits(), &AtomicBool::new(false), &mut counters)
            .unwrap();
        let token = engine.nodes[node].resume.clone().unwrap();
        let tasks = engine.stores.tasks.len();
        let facts = engine.stores.observations.len();
        trace.supports_discard.store(false, Ordering::Release);
        assert!(matches!(
            engine.synchronize_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Unsupported(
                "owned paused stack disposal"
            )))
        ));
        assert!(matches!(
            engine.discard_canceled_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Unsupported(
                "owned paused stack disposal"
            )))
        ));
        assert!(engine.cpu.token_is_current(&token));
        trace.supports_discard.store(true, Ordering::Release);
        trace.honors_discard.store(false, Ordering::Release);
        assert!(matches!(
            engine.discard_canceled_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Invalid(
                "paused stack disposal retained its owner"
            )))
        ));
        assert!(engine.cpu.token_is_current(&token));
        assert_eq!(engine.stores.tasks.len(), tasks);
        assert_eq!(engine.stores.observations.len(), facts);
        trace.honors_discard.store(true, Ordering::Release);
        engine.discard_canceled_cpu_checkpoints().unwrap();
        assert!(!engine.cpu.token_is_current(&token));
        assert!(engine.nodes[node].resume.is_none());
        assert_eq!(counters.cpu_nodes, 1);
    }

    #[test]
    fn orphan_frames_and_unknown_disposal_do_not_retire_paused_task_or_raw_evidence() {
        let (mut engine, trace) = engine(LegalOrderRoleMock);
        let node = root(&mut engine, &Position::startpos());
        let mut counters = PalsCounters::default();
        engine
            .verify_checked_cpu(node, &[], limits(), &AtomicBool::new(false), &mut counters)
            .unwrap();
        let (observation, execution) = engine.nodes[node]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap();
        let token = engine.nodes[node].resume.take().unwrap();
        let raw = engine.stores.observations.get(observation).unwrap().clone();
        assert!(engine.cpu.token_is_current(&token));
        // Actual owner survives even though no Node has a logical token.
        trace.honors_discard.store(false, Ordering::Release);
        assert!(matches!(
            engine.synchronize_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Invalid(
                "paused stack disposal retained its owner"
            )))
        ));
        assert!(matches!(
            engine.stores.tasks.get(execution).unwrap().status,
            TaskStatus::Paused { .. }
        ));
        assert!(engine.cpu.paused_stack_snapshot().unwrap().tokens_retained > 0);
        trace.honors_discard.store(true, Ordering::Release);
        trace.snapshot_mode.store(1, Ordering::Release);
        assert!(matches!(
            engine.synchronize_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Unsupported(
                "paused stack disposal completion is unknown"
            )))
        ));
        assert!(matches!(
            engine.stores.tasks.get(execution).unwrap().status,
            TaskStatus::Paused { .. }
        ));
        trace.snapshot_mode.store(2, Ordering::Release);
        assert!(matches!(
            engine.synchronize_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Invalid(
                "paused stack disposal completion is unknown"
            )))
        ));
        assert!(matches!(
            engine.stores.tasks.get(execution).unwrap().status,
            TaskStatus::Paused { .. }
        ));
        trace.snapshot_mode.store(0, Ordering::Release);
        engine.synchronize_cpu_checkpoints().unwrap();
        let actual = engine.cpu.paused_stack_snapshot().unwrap();
        assert!(actual.complete);
        assert_eq!((actual.tokens_retained, actual.bytes_current), (0, 0));
        assert!(matches!(engine.stores.tasks.get(execution).unwrap().status,
            TaskStatus::RetiredPaused { evidence: Some(id), .. } if id == observation));
        assert_eq!(engine.stores.observations.get(observation).unwrap(), &raw);
        assert_eq!(counters.cpu_nodes, 1);
    }
}
