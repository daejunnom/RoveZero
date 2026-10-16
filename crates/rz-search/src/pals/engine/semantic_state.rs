//! Common logical ownership, independent of an R searcher or worker.
//!
//! This first extraction owns admitted initialization. Legacy construction
//! explicitly moves its parts back into the unchanged flat engine layout;
//! subsequent owner/port extraction must not infer R-free support from this
//! initializer alone. No model, Store, or researcher is cloned here.

use super::*;

/// P/C graph, public-memory and role limits. There is no native-R task budget.
/// The caller retains its authoritative configuration; state does not keep a
/// second mutable copy of these fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsSemanticConfig {
    pub beam_width: usize,
    pub line_plies: usize,
    pub max_nodes: usize,
    pub max_records: usize,
    pub max_role_calls: u64,
}

impl From<&PalsConfig> for PalsSemanticConfig {
    fn from(config: &PalsConfig) -> Self {
        Self {
            beam_width: config.beam_width,
            line_plies: config.line_plies,
            max_nodes: config.max_nodes,
            max_records: config.max_records,
            max_role_calls: config.max_role_calls,
        }
    }
}

impl PalsSemanticConfig {
    pub fn validate(&self) -> Result<(), PalsError> {
        if self.beam_width == 0
            || self.beam_width > 16
            || self.line_plies == 0
            || self.line_plies > 16
            || self.max_nodes < 257
            || self.max_nodes > 65_536
            || self.max_records < 4
            || self.max_records > 128
            || self.max_role_calls == 0
            || self.max_role_calls > 1_000_000
        {
            return Err(PalsError::InvalidConfig);
        }
        Ok(())
    }

    pub(super) fn store_limits(&self) -> StoreLimits {
        StoreLimits {
            states: self.max_nodes,
            situations: self.max_nodes,
            line_chunks: self.max_nodes * 16,
            line_plies: self.line_plies,
            observations: self.max_nodes * 16,
            dependency_edges: self.max_nodes * 16,
            executions: self.max_nodes * 8,
            consumers: self.max_nodes * 16,
            ..StoreLimits::default()
        }
    }
}

/// Exact graph/public-memory ownership shared by future typed assemblies.
/// A state is not itself a search driver, researcher registration or receipt.
pub struct PalsSemanticState<M: RoleModel> {
    pub(super) model: M,
    pub(super) model_registered_value: Option<ModelValueIdentity>,
    pub(super) nodes: Vec<Node>,
    pub(super) records: Vec<RoleRecord>,
    pub(super) review_index: Option<ReviewContextIndex>,
    pub(super) review_policy: Option<ReviewSelectionPolicy>,
    pub(super) legacy_full_native_context: bool,
    pub(super) last_review_issue: Option<ReviewError>,
    pub(super) revision: u64,
    pub(super) game_generation: Option<u64>,
    pub(super) adaptive_root_registry: adaptive_root::AdaptiveRootRegistry,
    pub(super) stores: PalsStores,
    pub(super) clock_origin: Instant,
    pub(super) consumer_id: u64,
    pub(super) last_search_counters: Option<PalsCounters>,
    pub(super) last_recheck_observer_error: Option<RoleError>,
    pub(super) repair_queue: std::collections::VecDeque<frozen_recheck::RepairWork>,
    pub(super) repair_questions: HashSet<(BoardMove, u64, Vec<BoardMove>)>,
    pub(super) repair_counts: std::collections::HashMap<BoardMove, u8>,
    pub(super) archive_state: archive_lifecycle::EngineArchiveLifecycle,
    pub(super) new_game_transition_error: Option<PalsError>,
}

impl<M: RoleModel> PalsSemanticState<M> {
    /// Caller already admitted its model namespace and execution policy. This
    /// initializer allocates only the existing bounded semantic structures.
    pub(super) fn new_admitted(
        config: &PalsSemanticConfig,
        model: M,
        model_registered_value: Option<ModelValueIdentity>,
    ) -> Result<Self, PalsError> {
        config.validate()?;
        let mut nodes = Vec::new();
        nodes
            .try_reserve(config.max_nodes)
            .map_err(|_| PalsError::Capacity)?;
        let mut records = Vec::new();
        records
            .try_reserve(config.max_records)
            .map_err(|_| PalsError::Capacity)?;
        let stores = PalsStores::new(config.store_limits());
        Ok(Self {
            model,
            model_registered_value,
            nodes,
            records,
            review_index: None,
            review_policy: None,
            legacy_full_native_context: false,
            last_review_issue: None,
            revision: 0,
            game_generation: Some(0),
            adaptive_root_registry: adaptive_root::AdaptiveRootRegistry::default(),
            stores,
            clock_origin: Instant::now(),
            consumer_id: 0,
            last_search_counters: None,
            last_recheck_observer_error: None,
            repair_queue: std::collections::VecDeque::new(),
            repair_questions: HashSet::new(),
            repair_counts: std::collections::HashMap::new(),
            archive_state: archive_lifecycle::EngineArchiveLifecycle::default(),
            new_game_transition_error: None,
        })
    }
}

/// Actual registered legacy R owner. Construction requires the already
/// admitted checker; it never fabricates an absent checker identity.
pub struct RegisteredResearcher {
    pub(super) cpu: Box<dyn CpuChecker>,
    pub(super) checker_registered_identity: CheckerIdentity,
    pub(super) cpu_registered_value: Option<CpuValueIdentity>,
    pub(super) cpu_registered_condition: String,
    pub(super) checker_new_game_pending: bool,
    pub(super) last_external_attempt: Option<u64>,
    pub(super) external_attempts: Vec<CheckerAttempt>,
}

impl RegisteredResearcher {
    pub(super) fn from_admitted(
        cpu: Box<dyn CpuChecker>,
        checker_registered_identity: CheckerIdentity,
        cpu_registered_value: Option<CpuValueIdentity>,
        cpu_registered_condition: String,
    ) -> Self {
        Self {
            cpu,
            checker_registered_identity,
            cpu_registered_value,
            cpu_registered_condition,
            checker_new_game_pending: false,
            last_external_attempt: None,
            external_attempts: Vec::new(),
        }
    }
}

/// Typed absence, deliberately implementing neither CpuChecker nor CpuSearcher.
/// It owns no checker, TT, continuation, worker, identity or native-R budget.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NoResearcher;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct MoveOnlyModel(Arc<AtomicUsize>);
    impl Drop for MoveOnlyModel {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl RoleModel for MoveOnlyModel {
        fn identity(&self) -> &str {
            "semantic-owner-no-forward"
        }
        fn propose(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            panic!("admitted initialization must not run P");
        }
        fn reply(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            panic!("admitted initialization must not run C");
        }
        fn repair(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            panic!("admitted initialization must not run Repair");
        }
        fn divergences(&mut self, _: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            panic!("admitted initialization must not run divergence");
        }
    }

    #[test]
    fn admitted_semantic_owner_moves_nonclone_model_without_researcher_or_forward() {
        let legacy = PalsConfig {
            cpu_nodes_per_task: 0,
            ..PalsConfig::default()
        };
        assert!(legacy.validate().is_err());
        let common = PalsSemanticConfig::from(&legacy);
        common.validate().unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let state =
            PalsSemanticState::new_admitted(&common, MoveOnlyModel(drops.clone()), None).unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let PalsSemanticState { model, .. } = state;
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(model);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(std::mem::size_of::<NoResearcher>(), 0);
    }
}
