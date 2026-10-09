//! Bounded post-Repair work with one frozen Fresh-WDL namespace.
//! CPU checker reports discover lines; they never supply this comparison.
use super::*;

pub(super) const MAX_REPAIRS_PER_FIRST: u8 = 3;
pub(super) const MAX_PENDING_REPAIRS: usize = 64;

#[derive(Clone)]
pub(super) struct RepairWork {
    pub root: usize,
    pub original_first: BoardMove,
    pub attack_ply: usize,
    pub repaired_line: LineId,
    pub repair_record_revision: u64,
    pub repaired_leaf: usize,
    pub repaired: Vec<BoardMove>,
    pub refutation: Vec<BoardMove>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuConfig;

    struct WdlFixture {
        identity: ModelValueIdentity,
        finished: Vec<(RecheckDisposition, bool, Option<u64>, Option<u64>)>,
        revisions: Vec<u64>,
    }
    impl WdlFixture {
        fn new() -> Self {
            Self {
                identity: ModelValueIdentity {
                    semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
                    model: "frozen-WDL-fixture/2".into(),
                    encoding: "exact-typed-mock/2".into(),
                    precision: "fp32".into(),
                    model_epoch: [3; 32],
                },
                finished: Vec::new(),
                revisions: Vec::new(),
            }
        }
    }
    impl RoleModel for WdlFixture {
        fn identity(&self) -> &str {
            &self.identity.model
        }
        fn value_identity(&self) -> Option<&ModelValueIdentity> {
            Some(&self.identity)
        }
        fn evaluate_value(&mut self, query: RoleQuery<'_>) -> Result<ModelValueOutput, RoleError> {
            query.check_control()?;
            self.revisions.push(query.revision);
            let mut digest = Sha256::new();
            digest.update(query.position.to_fen().as_bytes());
            digest.update(query.revision.to_le_bytes());
            for movement in query.prefix {
                digest.update(
                    Move16::pack(*movement)
                        .map_err(|_| RoleError::InvalidOutput)?
                        .bits()
                        .to_le_bytes(),
                );
            }
            let lower = query.prefix.last() == Some(&BoardMove::from_uci("e5d4").unwrap());
            Ok(ModelValueOutput {
                identity: self.identity.clone(),
                input_sha256: digest.finalize().into(),
                state: query.position.position_identity(),
                perspective: query.position.side_to_move(),
                wdl: if lower {
                    [0.1, 0.2, 0.7]
                } else {
                    [0.7, 0.2, 0.1]
                },
            })
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            LegalOrderRoleMock.propose(query)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let wanted = BoardMove::from_uci(match query.prefix.len() {
                4 => "d2d3",
                5 => "d7d6",
                _ => "e2e4",
            })
            .unwrap();
            let selected = query.legal.iter().position(|movement| *movement == wanted);
            let mut output = LegalOrderRoleMock.repair(query)?;
            if let Some(index) = selected {
                output.logits[index] = 100.0;
            }
            Ok(output)
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            LegalOrderRoleMock.divergences(query)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let wanted = BoardMove::from_uci(match query.prefix.len() {
                4 => "d2d4",
                5 => "e5d4",
                _ => "g8f6",
            })
            .unwrap();
            let selected = query.legal.iter().position(|m| *m == wanted);
            let mut output = LegalOrderRoleMock.reply(query)?;
            if let Some(index) = selected {
                output.logits[index] = 100.0;
            }
            Ok(output)
        }
        fn recheck_finished(&mut self, event: RecheckFinished<'_>) -> Result<(), RoleError> {
            let revision = |endpoint: &RecheckEndpoint<'_>| match &endpoint.evidence {
                RecheckEndpointEvidence::ModelWdl {
                    context_revision, ..
                } => Some(*context_revision),
                _ => None,
            };
            self.finished.push((
                event.disposition,
                event.comparable,
                revision(&event.repaired_endpoint),
                event.counter_endpoint.as_ref().and_then(revision),
            ));
            Ok(())
        }
    }
    fn engine(policy: PostRepairRecheckPolicy, resolver: ResolverPolicy) -> PalsEngine<WdlFixture> {
        PalsEngine::new_with_boxed_checker_and_policies(
            PalsConfig {
                line_plies: 6,
                ..PalsConfig::default()
            },
            WdlFixture::new(),
            Box::new(
                OwnedCpuChecker::new(Box::new(
                    CpuEngine::new(CpuConfig {
                        max_depth: 2,
                        tt_entries: 128,
                        ..CpuConfig::default()
                    })
                    .unwrap(),
                ))
                .unwrap(),
            ),
            resolver,
            policy,
        )
        .unwrap()
    }
    fn limits() -> PalsLimits {
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(10),
            max_rounds: 1,
            max_cpu_nodes: 100_000,
            cpu_depth: 1,
        }
    }
    fn line(text: &[&str]) -> Vec<BoardMove> {
        text.iter()
            .map(|m| BoardMove::from_uci(m).unwrap())
            .collect()
    }
    fn accepted_repair(
        engine: &mut PalsEngine<WdlFixture>,
        counters: &mut PalsCounters,
    ) -> RepairWork {
        let root_position = Position::startpos();
        engine
            .stores
            .focus_actual_moves(root_position.snapshot())
            .unwrap();
        let root = engine.intern(root_position).unwrap();
        let refutation = line(&["e2e4", "e7e5", "g1f3", "b8c6", "d2d3", "d7d6"]);
        let repaired = line(&["e2e4", "e7e5", "f1c4", "b8c6", "d2d3", "d7d6"]);
        engine
            .record(RecordKind::Counterexample, &refutation, None, 0, None, None)
            .unwrap();
        let mut leaf = root;
        for movement in &repaired {
            let mut state = engine.nodes[leaf].position.clone();
            state.make_move(*movement).unwrap();
            leaf = engine.connect(leaf, *movement, state, counters).unwrap();
        }
        let (repaired_line, _) = engine
            .record(RecordKind::Repair, &repaired, None, 0, None, None)
            .unwrap()
            .unwrap();
        RepairWork {
            root,
            original_first: repaired[0],
            attack_ply: 1,
            repaired_line,
            repair_record_revision: engine.revision,
            repaired_leaf: leaf,
            repaired,
            refutation,
        }
    }
    #[test]
    fn own_checker_and_model_resolver_are_independent_and_legacy_constructor_is_preserved() {
        let mut selected = engine(
            PostRepairRecheckPolicy::Disabled,
            ResolverPolicy::ModelWdlRestricted,
        );
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let report = selected.search(&position, limits(), &cancel).unwrap();
        assert_eq!(report.resolver_version, MODEL_WDL_RESOLVER_VERSION);
        assert!(matches!(report.checker_identity, CheckerIdentity::Owned(_)));
        assert!(report.counters.value_calls > 0);
        assert_eq!(report.score, None);
        assert_eq!(selected.search_identity(), PALS_FOLLOWUP_SEARCH_VERSION);
        let legacy = PalsEngine::new(
            PalsConfig::default(),
            LegalOrderRoleMock,
            CpuEngine::new(CpuConfig::default()).unwrap(),
        )
        .unwrap();
        assert_eq!(legacy.resolver_policy(), ResolverPolicy::OwnRawRestricted);
        assert_eq!(legacy.search_identity(), PALS_SEARCH_VERSION);
    }
    #[test]
    fn common_recheck_uses_actual_c_continuation_and_same_fresh_record_revision() {
        for resolver in [
            ResolverPolicy::OwnRawRestricted,
            ResolverPolicy::ModelWdlRestricted,
        ] {
            let mut selected = engine(PostRepairRecheckPolicy::FrozenModelWdlV2, resolver);
            let mut counters = PalsCounters::default();
            let work = accepted_repair(&mut selected, &mut counters);
            let cancel = AtomicBool::new(false);
            let frontier = selected
                .frozen_recheck_once(&work, limits(), &cancel, &mut counters, &mut |_| {})
                .unwrap()
                .unwrap();
            assert_eq!(
                frontier.counterline,
                line(&["e2e4", "e7e5", "f1c4", "g8f6", "d2d4", "e5d4"])
            );
            assert_eq!(counters.frozen_wdl_comparisons, 1);
            assert_eq!(counters.supported_refutations, 1);
            let final_event = selected.model.finished.last().unwrap();
            assert_eq!(
                final_event.0,
                RecheckDisposition::ConditionalRefutationPublished
            );
            assert!(final_event.1);
            assert!(final_event.2.is_some());
            assert_eq!(final_event.2, final_event.3);
            let revisions = &selected.model.revisions;
            assert_eq!(
                revisions[revisions.len() - 1],
                revisions[revisions.len() - 2]
            );
            let conclusion = selected
                .stores
                .situations
                .get(selected.nodes[work.root].situation)
                .unwrap()
                .conclusions
                .get(work.repaired_line)
                .unwrap();
            let observation = selected
                .stores
                .observations
                .get(conclusion.evidence.unwrap())
                .unwrap();
            assert!(matches!(observation.score, RawScore::ConditionalWdl { .. }));
            assert!(observation.value_identity.is_none());
            assert!(observation.model_value_identity.is_none());
            assert!(observation.model_value_input.is_none());
        }
    }
    #[test]
    fn iterative_queue_bounds_questions_and_never_changes_original_first_move() {
        let mut selected = engine(
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
            ResolverPolicy::ModelWdlRestricted,
        );
        let mut counters = PalsCounters::default();
        let work = accepted_repair(&mut selected, &mut counters);
        let cancel = AtomicBool::new(false);
        let budget = limits();
        selected
            .queue_or_recheck_frozen(work.clone(), budget, &cancel, &mut counters, &mut |_| {})
            .unwrap();
        selected
            .queue_or_recheck_frozen(work.clone(), budget, &cancel, &mut counters, &mut |_| {})
            .unwrap();
        assert_eq!(selected.repair_queue.len(), 1);
        assert_eq!(counters.repair_queue_deduplicated, 1);
        for _ in 0..MAX_REPAIRS_PER_FIRST {
            assert!(selected.admit_iterative_repair(work.original_first));
        }
        assert!(!selected.admit_iterative_repair(work.original_first));
        selected
            .drain_repair_queue(budget, &cancel, &mut counters, &mut |_| {})
            .unwrap();
        assert!(selected.repair_queue.is_empty());
        assert_eq!(selected.repair_counts[&work.original_first], 3);
        assert!(
            selected
                .records
                .iter()
                .filter(|r| r.kind == RecordKind::Repair)
                .all(|r| r.line.first() == Some(&work.original_first))
        );
        selected.new_game();
        assert!(selected.repair_counts.is_empty());
        assert!(selected.repair_questions.is_empty());
    }
    #[test]
    fn endpoint_perspective_and_missing_or_different_revision_never_mint_completion() {
        let mut selected = engine(
            PostRepairRecheckPolicy::FrozenModelWdlV2,
            ResolverPolicy::ModelWdlRestricted,
        );
        let position = Position::startpos();
        selected
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = selected.intern(position).unwrap();
        assert_eq!(selected.endpoint_expectation(root, Color::White), None);
        let mut counters = PalsCounters::default();
        selected
            .evaluate_model_value(root, &[], limits(), &AtomicBool::new(false), &mut counters)
            .unwrap();
        let white = selected.endpoint_expectation(root, Color::White).unwrap();
        assert_eq!(
            selected.endpoint_expectation(root, Color::Black),
            Some(-white)
        );
        let snapshot = selected.nodes[root].position.snapshot();
        let endpoint = PalsEngine::<WdlFixture>::frozen_endpoint(
            &selected.nodes[root],
            &selected.stores,
            &snapshot,
            9999,
        );
        assert!(matches!(
            endpoint.evidence,
            RecheckEndpointEvidence::ModelWdl {
                context_revision: 0,
                ..
            }
        ));
        selected.nodes[root].model_value_revision = Some(9999);
        let endpoint = PalsEngine::<WdlFixture>::frozen_endpoint(
            &selected.nodes[root],
            &selected.stores,
            &snapshot,
            9999,
        );
        assert!(matches!(
            endpoint.evidence,
            RecheckEndpointEvidence::InvalidProvenance { .. }
        ));
        selected.nodes[root].model_observation = None;
        let endpoint = PalsEngine::<WdlFixture>::frozen_endpoint(
            &selected.nodes[root],
            &selected.stores,
            &snapshot,
            0,
        );
        assert!(matches!(
            endpoint.evidence,
            RecheckEndpointEvidence::Unobserved
        ));
    }

    #[test]
    fn actual_iterative_repair_supersedes_conditional_refutation_with_fresh_pair() {
        let mut selected = engine(
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
            ResolverPolicy::ModelWdlRestricted,
        );
        let mut counters = PalsCounters::default();
        let work = accepted_repair(&mut selected, &mut counters);
        let cancel = AtomicBool::new(false);
        let budget = limits();
        selected
            .queue_or_recheck_frozen(work.clone(), budget, &cancel, &mut counters, &mut |_| {})
            .unwrap();
        selected
            .drain_repair_queue(budget, &cancel, &mut counters, &mut |_| {})
            .unwrap();
        assert!(counters.accepted_repair_outputs > 0);
        assert!(counters.supported_repairs > 0);
        assert!(selected.repair_counts[&work.original_first] <= MAX_REPAIRS_PER_FIRST);
        assert!(selected.repair_queue.is_empty());
        let conclusion = selected
            .stores
            .situations
            .get(selected.nodes[work.root].situation)
            .unwrap()
            .conclusions
            .get(work.repaired_line)
            .unwrap();
        let ContinuationStatus::RepairedBy(repaired_line) = conclusion.status else {
            panic!("an accepted new P line must replace its actual conditional refutation");
        };
        assert_eq!(
            selected.stores.lines.moves(repaired_line).unwrap().first(),
            Some(&work.original_first)
        );
        let evidence = selected
            .stores
            .observations
            .get(conclusion.evidence.unwrap())
            .unwrap();
        assert_eq!(evidence.kind, ObservationKind::Repair);
        let refutation = selected
            .stores
            .observations
            .get(evidence.supersedes.unwrap())
            .unwrap();
        assert_eq!(refutation.kind, ObservationKind::Refutation);
        assert_eq!(refutation.line, Some(work.repaired_line));
        let RawScore::ConditionalRepairWdl {
            before,
            after,
            context_revision,
            ..
        } = evidence.score
        else {
            panic!("model repair requires immutable paired raw observations");
        };
        for observation in [before, after] {
            assert!(
                matches!(selected.stores.observations.get(observation).unwrap().score,
                RawScore::ContextWdl { context_revision: actual, .. } if actual == context_revision)
            );
        }
    }
}

struct NewFrontier {
    prefix: Vec<BoardMove>,
    response_node: usize,
    counterline: Vec<BoardMove>,
    attack_ply: usize,
    counter_leaf: usize,
    publication: Option<ObservationId>,
}

impl<M: RoleModel> PalsEngine<M> {
    pub(super) fn admit_iterative_repair(&mut self, movement: BoardMove) -> bool {
        if self.post_repair_recheck != PostRepairRecheckPolicy::IterativeFrozenModelWdlV2 {
            return true;
        }
        let count = self.repair_counts.entry(movement).or_default();
        if *count >= MAX_REPAIRS_PER_FIRST {
            return false;
        }
        *count += 1;
        true
    }

    pub(super) fn queue_or_recheck_frozen(
        &mut self,
        work: RepairWork,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<(), PalsError> {
        if self.post_repair_recheck == PostRepairRecheckPolicy::IterativeFrozenModelWdlV2 {
            let question = (
                work.original_first,
                work.repair_record_revision,
                work.repaired.clone(),
            );
            if !self.repair_questions.insert(question) {
                counters.repair_queue_deduplicated += 1;
                return Ok(());
            }
            if self.repair_queue.len() >= MAX_PENDING_REPAIRS {
                return Err(PalsError::Capacity);
            }
            self.repair_queue.push_back(work);
            counters.repair_queue_enqueued += 1;
            counters.repair_queue_peak = counters.repair_queue_peak.max(self.repair_queue.len());
        } else {
            self.frozen_recheck_once(&work, limits, cancel, counters, progress)?;
        }
        Ok(())
    }

    pub(super) fn drain_repair_queue(
        &mut self,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<(), PalsError> {
        while self.stopped(limits, cancel).is_none() {
            let Some(work) = self.repair_queue.pop_front() else {
                break;
            };
            let Some(frontier) =
                self.frozen_recheck_once(&work, limits, cancel, counters, progress)?
            else {
                counters.repair_queue_without_new_evidence += 1;
                continue;
            };
            if !self.admit_iterative_repair(work.original_first) {
                continue;
            }
            if self.cpu_budget_used(counters) >= limits.max_cpu_nodes {
                break;
            }
            let before = counters.accepted_repair_outputs;
            let mut repaired = frontier.prefix.clone();
            let repaired_leaf = self.follow(
                frontier.response_node,
                &mut repaired,
                &work.repaired,
                Some(&frontier.counterline),
                Call::Repair,
                limits,
                cancel,
                counters,
            )?;
            if repaired.first() != Some(&work.original_first) {
                return Err(StoreError::InvalidEvidence("queued Repair changed first move").into());
            }
            counters.repairs += 1;
            let Some((repaired_line, _)) =
                self.record(RecordKind::Repair, &repaired, None, 0, None, None)?
            else {
                continue;
            };
            let revision = self.revision;
            self.verify(repaired_leaf, &repaired, limits, cancel, counters)?;
            self.publish_iterative_repair(
                &work,
                &frontier,
                repaired_line,
                repaired_leaf,
                &repaired,
                limits,
                cancel,
                counters,
            )?;
            self.publish_choice(work.root, limits, cancel, progress)?;
            if counters.accepted_repair_outputs > before && repaired != work.repaired {
                self.queue_or_recheck_frozen(
                    RepairWork {
                        root: work.root,
                        original_first: work.original_first,
                        attack_ply: frontier.attack_ply,
                        repaired_line,
                        repair_record_revision: revision,
                        repaired_leaf,
                        repaired,
                        refutation: frontier.counterline,
                    },
                    limits,
                    cancel,
                    counters,
                    progress,
                )?;
            } else {
                counters.repair_queue_without_new_evidence += 1;
            }
        }
        Ok(())
    }

    /// A new P line is not a repair proof by itself. Compare its endpoint with
    /// the actual C counter under one Fresh context, then supersede only the
    /// still-active conditional refutation. CPU CP/mate is never converted here.
    #[allow(clippy::too_many_arguments)]
    fn publish_iterative_repair(
        &mut self,
        work: &RepairWork,
        frontier: &NewFrontier,
        repaired_line: LineId,
        repaired_leaf: usize,
        repaired: &[BoardMove],
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<bool, PalsError> {
        let Some(previous) = frontier.publication else {
            return Ok(false);
        };
        if self.stopped(limits, cancel).is_some() || repaired_line == work.repaired_line {
            return Ok(false);
        }
        self.validate_checker_namespace()?;
        let root = &self.nodes[work.root];
        let root_state = root.state;
        let root_situation = root.situation;
        let root_color = root.position.side_to_move();
        let generation = self.stores.generation();
        let current = self.stores.situations.get(root_situation)?;
        let root_revision = current.revision;
        if self.stores.root() != Some(root_situation)
            || repaired.first() != Some(&work.original_first)
            || self.stores.lines.get(repaired_line)?.start_state != root_state
            || self.stores.lines.moves(repaired_line)? != repaired
            || current
                .conclusions
                .get(work.repaired_line)
                .is_none_or(|conclusion| {
                    conclusion.status != ContinuationStatus::Refuted
                        || conclusion.evidence != Some(previous)
                })
        {
            return Err(StoreError::StaleConsumer.into());
        }
        let context_revision = self.revision;
        let terminals = (
            self.nodes[frontier.counter_leaf].terminal,
            self.nodes[repaired_leaf].terminal,
        );
        match terminals {
            (None, None) => {
                self.evaluate_model_value(
                    frontier.counter_leaf,
                    &frontier.counterline,
                    limits,
                    cancel,
                    counters,
                )?;
                self.evaluate_model_value(repaired_leaf, repaired, limits, cancel, counters)?;
                if self.nodes[frontier.counter_leaf].model_value_revision != Some(context_revision)
                    || self.nodes[repaired_leaf].model_value_revision != Some(context_revision)
                {
                    counters.frozen_wdl_unresolved += 1;
                    return Ok(false);
                }
            }
            (Some(_), Some(_)) => {}
            _ => {
                counters.frozen_wdl_unresolved += 1;
                return Ok(false);
            }
        }
        if self.stopped(limits, cancel).is_some() {
            return Ok(false);
        }
        if self.revision != context_revision
            || self.stores.root() != Some(root_situation)
            || self.stores.generation() != generation
            || self.stores.situations.get(root_situation)?.revision != root_revision
        {
            return Err(StoreError::StaleConsumer.into());
        }
        self.validate_checker_namespace()?;
        let Some((before, after)) = self
            .endpoint_expectation(frontier.counter_leaf, root_color)
            .zip(self.endpoint_expectation(repaired_leaf, root_color))
        else {
            counters.frozen_wdl_unresolved += 1;
            return Ok(false);
        };
        counters.frozen_wdl_comparisons += 1;
        if after <= before {
            return Ok(false);
        }
        let score = if terminals.0.is_some() {
            // Both endpoints are exact Rules terminals. This derived line
            // conclusion remains conditional Model scope, not a mate proof.
            RawScore::Estimate {
                value: after,
                perspective: root_color,
            }
        } else {
            RawScore::ConditionalRepairWdl {
                expectation: after,
                perspective: root_color,
                before: self.nodes[frontier.counter_leaf]
                    .model_observation
                    .ok_or(RoleError::Unavailable)?,
                after: self.nodes[repaired_leaf]
                    .model_observation
                    .ok_or(RoleError::Unavailable)?,
                context_revision,
            }
        };
        let evidence = self.append_engine_observation(Observation {
            state: root_state,
            line: Some(repaired_line),
            source: stable_id(self.search_identity()),
            epoch: 0,
            scope: EvidenceScope::Model {
                model: stable_id(self.model.identity()),
                encoding: stable_id("pals-frozen-wdl-conditional-repair/2"),
                input: context_revision,
            },
            score,
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: None,
            cpu_pv: None,
            budget: 0,
            kind: ObservationKind::Repair,
            supersedes: Some(previous),
            execution: None,
        })?;
        self.stores
            .repair(root_situation, work.repaired_line, repaired_line, evidence)?;
        counters.supported_repairs += 1;
        Ok(true)
    }

    fn frozen_endpoint<'a>(
        node: &'a Node,
        stores: &'a PalsStores,
        snapshot: &'a PositionSnapshot,
        _revision: u64,
    ) -> RecheckEndpoint<'a> {
        let evidence = if let Some((reason, value)) = node.terminal {
            RecheckEndpointEvidence::RulesTerminal {
                reason,
                value,
                perspective: node.position.side_to_move(),
            }
        } else if let (Some(output), Some(id)) = (&node.model_value, node.model_observation) {
            match stores.observations.get(id) {
                Ok(observation)
                    if observation.state == node.state
                        && observation.model_value_identity.as_ref() == Some(&output.identity)
                        && observation.model_value_input == Some(output.input_sha256)
                        && matches!(observation.score,
                            RawScore::ContextWdl { win, draw, loss, perspective, context_revision }
                            if [win,draw,loss] == output.wdl
                                && perspective == output.perspective
                                && Some(context_revision) == node.model_value_revision) =>
                {
                    RecheckEndpointEvidence::ModelWdl {
                        observation_id: id,
                        observation,
                        output,
                        context_revision: node
                            .model_value_revision
                            .expect("validated raw revision"),
                    }
                }
                _ => RecheckEndpointEvidence::InvalidProvenance {
                    error: StoreError::InvalidEvidence("Fresh WDL observation mismatch"),
                },
            }
        } else {
            RecheckEndpointEvidence::Unobserved
        };
        RecheckEndpoint {
            state: node.state,
            situation: node.situation,
            snapshot,
            evidence,
        }
    }

    fn endpoint_expectation(&self, node: usize, root_color: Color) -> Option<f32> {
        let node = &self.nodes[node];
        let value = if let Some((_, value)) = node.terminal {
            value.signum() as f32
        } else {
            let output = node.model_value.as_ref()?;
            output
                .validate(&node.position, self.model_registered_value.as_ref()?)
                .ok()?;
            output.wdl[0] - output.wdl[2]
        };
        Some(if node.position.side_to_move() == root_color {
            value
        } else {
            -value
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn frozen_recheck_once(
        &mut self,
        work: &RepairWork,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<Option<NewFrontier>, PalsError> {
        if self.stopped(limits, cancel).is_some() {
            return Ok(None);
        }
        self.validate_checker_namespace()?;
        let root = self
            .nodes
            .get(work.root)
            .ok_or(StoreError::InvalidHandle("WDL root"))?;
        let root_state = root.state;
        let root_situation = root.situation;
        let root_color = root.position.side_to_move();
        let generation = self.stores.generation();
        let root_revision = self.stores.situations.get(root_situation)?.revision;
        if self.stores.root() != Some(root_situation)
            || work.repaired.first() != Some(&work.original_first)
            || work.attack_ply >= work.repaired.len()
            || work.attack_ply >= work.refutation.len()
            || work.repaired[..=work.attack_ply] != work.refutation[..=work.attack_ply]
            || self.stores.lines.get(work.repaired_line)?.start_state != root_state
            || self.stores.lines.moves(work.repaired_line)? != work.repaired
        {
            return Err(
                StoreError::InvalidEvidence("frozen recheck line authority mismatch").into(),
            );
        }
        let record = self
            .records
            .iter()
            .find(|record| {
                record.revision == work.repair_record_revision
                    && record.origin_state == root_state
                    && record.kind == RecordKind::Repair
                    && record.line == work.repaired
            })
            .cloned()
            .ok_or(StoreError::InvalidEvidence("WDL Repair record missing"))?;
        let mut path = Vec::with_capacity(work.repaired.len() + 1);
        path.push(work.root);
        for movement in &work.repaired {
            let parent = *path.last().ok_or(StoreError::InvalidHandle("WDL path"))?;
            path.push(
                self.nodes[parent]
                    .edges
                    .iter()
                    .find(|edge| edge.movement == *movement)
                    .ok_or(StoreError::InvalidHandle("WDL path edge"))?
                    .child,
            );
        }
        if path.last() != Some(&work.repaired_leaf) {
            return Err(StoreError::InvalidEvidence("WDL repaired endpoint mismatch").into());
        }
        let changed = (work.attack_ply + 1..work.repaired.len()).find(|&ply| {
            self.nodes[path[ply]].position.side_to_move() == root_color
                && work.refutation.get(ply) != Some(&work.repaired[ply])
        });
        let Some(anchor_ply) = changed.and_then(|ply| {
            (ply + 1..work.repaired.len())
                .find(|&index| self.nodes[path[index]].position.side_to_move() != root_color)
        }) else {
            counters.frozen_wdl_unresolved += 1;
            return Ok(None);
        };
        let anchor = path[anchor_ply];
        self.evaluate_model_value(work.repaired_leaf, &work.repaired, limits, cancel, counters)?;
        let reply_context = self.role_context(
            anchor,
            RoleQuestion {
                purpose: RoleQueryPurpose::ReplyPolicy,
                prefix: &work.repaired[..anchor_ply],
                proposal: &work.repaired,
                refutation: Some(&work.refutation),
                divergences: &[],
            },
        )?;
        let identity = RecheckIdentity {
            game_generation: self.game_generation.ok_or(PalsError::Capacity)?,
            search_generation: generation,
            root: root_situation,
            root_revision,
            repair_record_revision: work.repair_record_revision,
            repaired_line: work.repaired_line,
        };
        let root_snapshot = self.nodes[work.root].position.snapshot();
        let anchor_snapshot = self.nodes[anchor].position.snapshot();
        let repaired_snapshot = self.nodes[work.repaired_leaf].position.snapshot();
        let examined: Vec<_> = self.nodes[anchor]
            .edges
            .iter()
            .map(|edge| edge.movement)
            .collect();
        let deadline_tick = self.tick_at(limits.deadline);
        let previous = self
            .stores
            .situations
            .get(root_situation)?
            .conclusions
            .get(work.repaired_line)
            .and_then(|c| c.evidence);
        let mut trace = RecheckProgress::default();
        let prepared = self.model.recheck_prepared(RecheckPrepared {
            policy: self.post_repair_recheck,
            identity,
            root_state,
            root_snapshot: &root_snapshot,
            anchor_state: self.nodes[anchor].state,
            anchor_situation: self.nodes[anchor].situation,
            anchor_snapshot: &anchor_snapshot,
            anchor_ply,
            examined_responses: &examined,
            reply_context: &reply_context,
            repair_record: &record,
            repaired: &work.repaired,
            refutation: &work.refutation,
            repaired_endpoint: Self::frozen_endpoint(
                &self.nodes[work.repaired_leaf],
                &self.stores,
                &repaired_snapshot,
                self.revision,
            ),
            limits,
            deadline_tick,
            cancel,
            config: &self.config,
            checker_identity: &self.checker_registered_identity,
            cpu_condition: &self.cpu_registered_condition,
        });
        let mut new_frontier = None;
        let mut result = if let Err(error) = prepared {
            trace.disposition = RecheckDisposition::PreparedRejected;
            Err(PalsError::Role(error))
        } else {
            trace.prepared_accepted = true;
            (|| {
                let calls = counters.critic_calls;
                let accepted = counters.accepted_critic_outputs;
                let replies = self.ranked(
                    anchor,
                    Call::Reply,
                    &work.repaired[..anchor_ply],
                    &work.repaired,
                    Some(&work.refutation),
                    limits,
                    cancel,
                    counters,
                );
                trace.reply_call_attempted = counters.critic_calls > calls;
                trace.reply_accepted = counters.accepted_critic_outputs > accepted;
                let replies = replies?;
                let old = work.repaired[anchor_ply];
                let response = replies
                    .iter()
                    .copied()
                    .find(|m| *m != old && !examined.contains(m))
                    .or_else(|| replies.iter().copied().find(|m| *m != old));
                let Some(response) = response else {
                    trace.disposition = RecheckDisposition::NoAlternativeResponse;
                    return Ok(());
                };
                trace.selected_response = Some(response);
                trace
                    .counterline
                    .extend_from_slice(&work.repaired[..anchor_ply]);
                let mut state = self.nodes[anchor].position.clone();
                state.make_move(response)?;
                let response_node = self.connect(anchor, response, state, counters)?;
                trace.counterline.push(response);
                let repair_prefix = trace.counterline.clone();
                let mut leaf = response_node;
                trace.counter_leaf = Some(leaf);
                while trace.counterline.len() < work.repaired.len()
                    && self.nodes[leaf].terminal.is_none()
                    && self.stopped(limits, cancel).is_none()
                {
                    let ranked = self.ranked(
                        leaf,
                        Call::Reply,
                        &trace.counterline,
                        &work.repaired,
                        Some(&work.refutation),
                        limits,
                        cancel,
                        counters,
                    )?;
                    let Some(movement) = ranked.first().copied() else {
                        break;
                    };
                    let mut position = self.nodes[leaf].position.clone();
                    position.make_move(movement)?;
                    leaf = self.connect(leaf, movement, position, counters)?;
                    trace.counter_leaf = Some(leaf);
                    trace.counterline.push(movement);
                }
                trace.counterline_completed = trace.counterline.len() == work.repaired.len();
                if !trace.counterline_completed {
                    trace.disposition = RecheckDisposition::IncompleteCounterline;
                    return Ok(());
                }
                if trace.counterline == work.refutation || trace.counterline == work.repaired {
                    trace.disposition = RecheckDisposition::NoAlternativeResponse;
                    return Ok(());
                }
                counters.refutations += 1;
                self.record(
                    RecordKind::Counterexample,
                    &trace.counterline,
                    None,
                    0,
                    None,
                    None,
                )?;
                self.verify_checked_cpu(leaf, &trace.counterline, limits, cancel, counters)?;
                let comparison_revision = self.revision;
                // Both Fresh invocations see exactly the same public-record revision.
                // They have different target states, not a shared raw input/cache key.
                self.evaluate_model_value(
                    work.repaired_leaf,
                    &work.repaired,
                    limits,
                    cancel,
                    counters,
                )?;
                self.evaluate_model_value(leaf, &trace.counterline, limits, cancel, counters)?;
                if self.revision != comparison_revision
                    || self.stores.root() != Some(root_situation)
                    || self.stores.generation() != generation
                    || self.stores.situations.get(root_situation)?.revision != root_revision
                {
                    return Err(StoreError::StaleConsumer.into());
                }
                self.validate_checker_namespace()?;
                trace.comparable = match (
                    self.nodes[work.repaired_leaf].terminal,
                    self.nodes[leaf].terminal,
                ) {
                    (Some(_), Some(_)) => true,
                    (None, None) => {
                        self.nodes[work.repaired_leaf].model_value_revision
                            == Some(comparison_revision)
                            && self.nodes[leaf].model_value_revision == Some(comparison_revision)
                            && self.nodes[work.repaired_leaf]
                                .model_value
                                .as_ref()
                                .zip(self.nodes[leaf].model_value.as_ref())
                                .is_some_and(|(a, b)| {
                                    a.identity == b.identity
                                        && Some(&a.identity) == self.model_registered_value.as_ref()
                                })
                    }
                    _ => false,
                };
                new_frontier = Some(NewFrontier {
                    prefix: repair_prefix,
                    response_node,
                    counterline: trace.counterline.clone(),
                    attack_ply: anchor_ply,
                    counter_leaf: leaf,
                    publication: None,
                });
                if !trace.comparable {
                    counters.frozen_wdl_unresolved += 1;
                    trace.disposition = RecheckDisposition::IncomparableEvidence;
                    return Ok(());
                }
                counters.frozen_wdl_comparisons += 1;
                let Some((old, new)) = self
                    .endpoint_expectation(work.repaired_leaf, root_color)
                    .zip(self.endpoint_expectation(leaf, root_color))
                else {
                    counters.frozen_wdl_unresolved += 1;
                    trace.disposition = RecheckDisposition::MissingCompletedCounterValue;
                    return Ok(());
                };
                if new >= old {
                    trace.disposition = RecheckDisposition::CounterNotLower;
                    return Ok(());
                }
                self.publish_choice(work.root, limits, cancel, progress)?;
                let evidence = self.frozen_refutation_observation(work, leaf, new, previous)?;
                self.stores
                    .refute_continuation(root_situation, work.repaired_line, evidence)?;
                counters.supported_refutations += 1;
                trace.publication = Some(evidence);
                if let Some(frontier) = new_frontier.as_mut() {
                    frontier.publication = Some(evidence);
                }
                trace.disposition = RecheckDisposition::ConditionalRefutationPublished;
                Ok(())
            })()
        };
        let counter_snapshot = trace
            .counter_leaf
            .map(|node| self.nodes[node].position.snapshot());
        let publication = match trace
            .publication
            .map(|id| -> Result<_, PalsError> {
                Ok(RecheckPublication {
                    observation_id: id,
                    observation: self.stores.observations.get(id)?,
                    conclusion: *self
                        .stores
                        .situations
                        .get(root_situation)?
                        .conclusions
                        .get(work.repaired_line)
                        .ok_or(StoreError::InvalidEvidence("WDL publication absent"))?,
                })
            })
            .transpose()
        {
            Ok(publication) => publication,
            Err(error) => {
                if result.is_ok() {
                    result = Err(error);
                }
                None
            }
        };
        let finish = self.model.recheck_finished(RecheckFinished {
            policy: self.post_repair_recheck,
            identity,
            prepared_accepted: trace.prepared_accepted,
            reply_call_attempted: trace.reply_call_attempted,
            reply_accepted: trace.reply_accepted,
            reply_context: &reply_context,
            selected_response: trace.selected_response,
            counterline: &trace.counterline,
            full_suffix_replayed: false,
            counterline_completed: trace.counterline_completed,
            repaired_endpoint: Self::frozen_endpoint(
                &self.nodes[work.repaired_leaf],
                &self.stores,
                &repaired_snapshot,
                self.revision,
            ),
            counter_endpoint: trace.counter_leaf.zip(counter_snapshot.as_ref()).map(
                |(node, snapshot)| {
                    Self::frozen_endpoint(&self.nodes[node], &self.stores, snapshot, self.revision)
                },
            ),
            comparable: trace.comparable,
            publication,
            disposition: trace.disposition,
            original_error: result.as_ref().err(),
            limits,
            deadline_tick,
            cancel,
        });
        if let Err(error) = finish {
            self.last_recheck_observer_error
                .get_or_insert_with(|| error.clone());
            if result.is_ok() {
                return Err(error.into());
            }
        }
        result?;
        if let Some(stop) = self.stopped(limits, cancel) {
            return Err(match stop {
                PalsCompletion::Canceled => RoleError::Canceled,
                _ => RoleError::Deadline,
            }
            .into());
        }
        Ok(new_frontier)
    }

    fn frozen_refutation_observation(
        &mut self,
        work: &RepairWork,
        counter_leaf: usize,
        expectation: f32,
        supersedes: Option<ObservationId>,
    ) -> Result<ObservationId, PalsError> {
        let score = if self.nodes[counter_leaf].terminal.is_some() {
            RawScore::Estimate {
                value: expectation,
                perspective: self.nodes[work.root].position.side_to_move(),
            }
        } else {
            RawScore::ConditionalWdl {
                expectation,
                perspective: self.nodes[work.root].position.side_to_move(),
                repaired: self.nodes[work.repaired_leaf]
                    .model_observation
                    .ok_or(RoleError::Unavailable)?,
                counter: self.nodes[counter_leaf]
                    .model_observation
                    .ok_or(RoleError::Unavailable)?,
                context_revision: self.revision,
            }
        };
        Ok(self.append_engine_observation(Observation {
            state: self.nodes[work.root].state,
            line: Some(work.repaired_line),
            source: stable_id(self.search_identity()),
            epoch: 0,
            scope: EvidenceScope::Model {
                model: stable_id(self.model.identity()),
                encoding: stable_id("pals-frozen-wdl-conditional-line/2"),
                input: self.revision,
            },
            score,
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: None,
            cpu_pv: None,
            budget: 0,
            kind: ObservationKind::Refutation,
            supersedes,
            execution: None,
        })?)
    }
}
