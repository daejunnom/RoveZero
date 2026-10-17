//! Concrete native product assembly. These constructors reuse the actual
//! selected-view ticket service and private CPU engines; they never substitute
//! the synchronous RoleModel compatibility driver for native inference.

use super::configuration::{ExecutionConfiguration, ResearchValueConfiguration};
use super::nnuec_session::{RoventNnuecSessionDriver, RoventNnuecSessionRejection};
use super::nnuec_source::LoadedRoventNnuecSource;
use super::pc_session::{RoventPcCollectionHooks, RoventPcSessionDriver};
use super::{PreparedRoventNative, RoventNativeBackend, RoventPinnedNativeSources};
use crate::EngineIdentity;
use crate::engine::{OwnerRegistry, ProcessClock};
use crate::pals_adaptive::{
    AdaptiveCheckerMetadata, AdaptiveSessionAdmissionRejection, AdaptiveSessionCollectionHooks,
    AdaptiveSessionDriver, AdaptiveSessionOptions, AdaptiveSessionSources, PersistentCpuOwner,
    RoventPcrSessionProfile,
};
use crate::rovent_root_admission::RoventRootAdmissionPolicy;
use crate::search_driver::{
    DriverPhysicalCompletion, PalsCheckerRegistration, SearchKind, SearchSessionContext,
    SearchSessionDriver, SearchSessionFailure, SearchSessionReport,
};
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_contracts::{ContractError, Digest};
use rz_position::{BoardMove, Position};
use rz_search::cpu::{CpuEngine, CpuResumePolicy};
use rz_search::cpu_value::BootstrapCpuValue;
use rz_search::cpu_worker::{
    CpuWorkerProfile, cpu_evaluation_namespace_digest, cpu_search_conditions_digest,
};
use rz_search::external_cpu::{CpuEndpointSelection, ExternalSlicePolicy};
use rz_search::pals::engine::adaptive_port::{
    AdaptivePortModel, AdaptivePortQuestion, AdaptiveRulesOwnerIssuer,
};
use rz_search::pals::engine::pc_port::PcPortQuestion;
use rz_search::pals::engine::{PalsConfig, PalsEngine, ResearcherSemanticProfile, ResolverPolicy};
use sha2::{Digest as _, Sha256};
use std::{
    num::NonZeroU16,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

type PcrDriver = AdaptiveSessionDriver<RoventNativeBackend<AdaptivePortQuestion>>;
enum ActualProductDriver {
    Pc(RoventPcSessionDriver),
    Pcr(PcrDriver),
    Nnuec(RoventNnuecSessionDriver),
}

/// Keeps the same original retained descriptors until actual whole-session
/// shutdown. Unknown drop quarantines both physical owners and their sources.
pub struct RoventNativeProductDriver {
    inner: Arc<ActualProductDriver>,
    sources: Arc<RoventPinnedNativeSources>,
    physically_closed: AtomicBool,
}
pub enum RoventProductStartupOwners {
    Pc(PreparedRoventNative<PcPortQuestion>),
    Pcr(PreparedRoventNative<AdaptivePortQuestion>),
    PcrRejected(AdaptiveSessionAdmissionRejection<RoventNativeBackend<AdaptivePortQuestion>>),
    PcrDriver(Box<PcrDriver>),
    NnuecRejected(Box<RoventNnuecSessionRejection>),
}
pub struct RoventProductStartupRejection {
    pub failure: SearchSessionFailure,
    pub owners: RoventProductStartupOwners,
    /// Same pins even when the rejected physical factory is retained.
    pub sources: Arc<RoventPinnedNativeSources>,
}

impl RoventNativeProductDriver {
    #[allow(clippy::result_large_err)]
    pub fn own_nnuec_pcr(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventProductStartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let driver =
            RoventNnuecSessionDriver::new(prepared, loaded, clock, owners, identity, root_policy)
                .map_err(|rejected| RoventProductStartupRejection {
                failure: rejected.failure.clone(),
                owners: RoventProductStartupOwners::NnuecRejected(Box::new(rejected)),
                sources: Arc::clone(&sources),
            })?;
        Ok(Self {
            inner: Arc::new(ActualProductDriver::Nnuec(driver)),
            sources,
            physically_closed: AtomicBool::new(false),
        })
    }
    #[allow(clippy::result_large_err)]
    pub fn pc(
        prepared: PreparedRoventNative<PcPortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventProductStartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let driver = RoventPcSessionDriver::new(prepared, clock, owners, identity, root_policy)
            .map_err(|rejected| RoventProductStartupRejection {
                failure: rejected.failure,
                owners: RoventProductStartupOwners::Pc(rejected.prepared),
                sources: Arc::clone(&sources),
            })?;
        Ok(Self {
            inner: Arc::new(ActualProductDriver::Pc(driver)),
            sources,
            physically_closed: AtomicBool::new(false),
        })
    }

    #[allow(clippy::result_large_err)]
    pub fn pc_with_collection_hooks(
        prepared: PreparedRoventNative<PcPortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        hooks: RoventPcCollectionHooks,
    ) -> Result<Self, RoventProductStartupRejection> {
        let sources = Arc::clone(prepared.sources());
        let driver = RoventPcSessionDriver::new_with_collection_hooks(
            prepared,
            clock,
            owners,
            identity,
            root_policy,
            hooks,
        )
        .map_err(|rejected| RoventProductStartupRejection {
            failure: rejected.failure,
            owners: RoventProductStartupOwners::Pc(rejected.prepared),
            sources: Arc::clone(&sources),
        })?;
        Ok(Self {
            inner: Arc::new(ActualProductDriver::Pc(driver)),
            sources,
            physically_closed: AtomicBool::new(false),
        })
    }

    /// Actual existing material/PST integer board evaluator. This is a distinct
    /// M01 baseline, never a placeholder described as NNUEC.
    #[allow(clippy::result_large_err)]
    pub fn own_board_pcr(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventProductStartupRejection> {
        Self::admit_own_board_pcr(prepared, clock, owners, identity, root_policy, None)
    }

    /// Collection preserves the actual loaded origin and reuses the same
    /// exclusive root owner. Upper callers retain independent teacher owners.
    #[allow(clippy::result_large_err)]
    pub fn own_board_pcr_with_collection_hooks(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        hooks: AdaptiveSessionCollectionHooks<RoventNativeBackend<AdaptivePortQuestion>>,
    ) -> Result<Self, RoventProductStartupRejection> {
        Self::admit_own_board_pcr(prepared, clock, owners, identity, root_policy, Some(hooks))
    }

    #[allow(clippy::result_large_err)]
    fn admit_own_board_pcr(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks: Option<
            AdaptiveSessionCollectionHooks<RoventNativeBackend<AdaptivePortQuestion>>,
        >,
    ) -> Result<Self, RoventProductStartupRejection> {
        let pins = Arc::clone(prepared.sources());
        let collection = match (pins.lock().manifest.purpose, collection_hooks) {
            (rz_experiments::RoventPurpose::Collection, Some(hooks))
                if hooks.root_admission.is_some() =>
            {
                match prepared.collection_identity() {
                    Ok(origin) => Some((origin, hooks)),
                    Err(error) => {
                        return Err(RoventProductStartupRejection {
                            failure: SearchSessionFailure::debug("RoventCollectionOrigin", &error),
                            owners: RoventProductStartupOwners::Pcr(prepared),
                            sources: pins,
                        });
                    }
                }
            }
            (rz_experiments::RoventPurpose::Collection, _) | (_, Some(_)) => {
                return Err(RoventProductStartupRejection {
                    failure: failure(
                        "RoventCollectionHooks",
                        "collection purpose and actual root observer must be selected together",
                    ),
                    owners: RoventProductStartupOwners::Pcr(prepared),
                    sources: pins,
                });
            }
            (_, None) => None,
        };

        let build = (|| -> Result<_, SearchSessionFailure> {
            let configuration = pins.configuration();
            let ExecutionConfiguration::OwnResearcher {
                event_order,
                queues,
                max_rounds,
                max_native_nodes,
                cpu_depth,
                cpu_nodes_per_task,
                cpu_plan,
                value,
                controller,
                reciprocal,
                peers,
            } = configuration.fields().execution
            else {
                return Err(failure(
                    "RoventPcrStartup",
                    "an actual own-R plan is required",
                ));
            };
            if !matches!(value, ResearchValueConfiguration::BaselineBoard)
                || pins.lock().manifest.engine.nnuec.is_some()
                || !prepared.snapshot().mode.has_native_r()
                || prepared.snapshot().process_epoch != clock.epoch().0
            {
                return Err(failure(
                    "RoventPcrValue",
                    "baseline board and NNUEC sources are distinct",
                ));
            }
            if cpu_plan.affinity.is_some() {
                return Err(failure(
                    "RoventPcrAffinityUnsupported",
                    "this engine-retaining session does not apply a per-worker affinity request",
                ));
            }
            let semantic = configuration.fields().semantic;
            let namespace = prepared
                .snapshot()
                .model_namespace()
                .map_err(|error| SearchSessionFailure::debug("RoventPcrNamespace", &error))?;
            let model = AdaptivePortModel::new_with_namespace(
                prepared.snapshot().role_model_identity(),
                Some(
                    prepared
                        .snapshot()
                        .model_value_identity()
                        .map_err(|error| SearchSessionFailure::debug("RoventPcrValue", &error))?,
                ),
                namespace.clone(),
            )
            .map_err(|error| SearchSessionFailure::debug("RoventPcrModel", &error))?;
            let control = controller
                .actual(*event_order)
                .map_err(|error| SearchSessionFailure::debug("RoventPcrController", &error))?;
            let mut cpu = Vec::new();
            cpu.try_reserve_exact(usize::from(control.max_workers))
                .map_err(|error| SearchSessionFailure::debug("RoventPcrCpuCapacity", &error))?;
            for slot in 0..u16::from(control.max_workers) {
                let evaluator = Arc::new(BootstrapCpuValue::default());
                let actual = cpu_plan
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrCpuPlan", &error))?;
                let mut engine = if cpu_plan.preserve_paused_side_scope {
                    CpuEngine::with_paused_side_scope(actual, evaluator, cpu_plan.ordering())
                } else {
                    CpuEngine::with_evaluator_ordering_and_resume_policy(
                        actual,
                        evaluator,
                        cpu_plan.ordering(),
                        CpuResumePolicy::PausedStack,
                    )
                }
                .map_err(|error| SearchSessionFailure::debug("RoventPcrCpu", &error))?;
                engine
                    .set_paused_stack_max_bytes(cpu_plan.paused_bytes_limit)
                    .map_err(|error| {
                        SearchSessionFailure::debug("RoventPcrPausedCapacity", &error)
                    })?;
                let profile = CpuWorkerProfile {
                    owner: CpuWorkerId(slot + 1),
                    max_depth: engine.config().max_depth,
                    evaluation_namespace: cpu_evaluation_namespace_digest(&engine).map_err(
                        |error| SearchSessionFailure::debug("RoventPcrCpuNamespace", &error),
                    )?,
                    search_conditions: cpu_search_conditions_digest(&engine),
                };
                cpu.push(PersistentCpuOwner { engine, profile });
            }
            let primary = cpu.first().ok_or_else(|| {
                failure("RoventPcrCpu", "controller has no actual researcher owner")
            })?;
            let metadata = AdaptiveCheckerMetadata::capture_owned(&primary.engine)
                .map_err(|error| SearchSessionFailure::debug("RoventPcrMetadata", &error))?;
            let mirror = metadata.clone().into_mirror();
            let resolver = ResolverPolicy::ModelWdlRestricted;
            let recheck = configuration.post_repair_policy_for(prepared.snapshot().mode);
            let registration =
                PalsCheckerRegistration::capture_with_resolver(&mirror, &model, resolver)?;
            let engine = PalsEngine::new_with_boxed_checker_and_rovent_profile(
                ResearcherSemanticProfile {
                    config: PalsConfig {
                        beam_width: semantic.beam_width,
                        line_plies: semantic.line_plies,
                        max_nodes: semantic.max_nodes,
                        max_records: semantic.max_records,
                        max_role_calls: semantic.max_role_calls,
                        cpu_nodes_per_task: *cpu_nodes_per_task,
                    },
                    maximum_root_prefix_plies: semantic.maximum_root_prefix_plies,
                },
                model,
                Box::new(mirror),
                resolver,
                recheck,
            )
            .map_err(|error| SearchSessionFailure::debug("RoventPcrSemantic", &error))?
            .with_selected_review_context(
                configuration
                    .fields()
                    .review
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrReview", &error))?,
                configuration
                    .fields()
                    .index
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrIndex", &error))?,
            )
            .map_err(|error| SearchSessionFailure::debug("RoventPcrReview", &error))?;
            if engine.post_repair_recheck_policy() != recheck {
                return Err(failure(
                    "RoventPcrRecheckPolicy",
                    "actual engine policy differs from its registered configuration",
                ));
            }
            let root_budget = Digest(
                rz_eval::asset::parse_sha256(
                    &pins.lock().manifest.coordinator.original_root_budget_sha256,
                )
                .map_err(|error| SearchSessionFailure::debug("RoventPcrBudget", &error))?,
            );
            let mut hash = Sha256::new();
            hash.update(b"rz-rovent-native-product/1\0");
            hash.update(include_bytes!("product.rs"));
            hash.update(super::compiled_bootstrap_source_digest());
            hash.update(
                configuration.configuration_digest().map_err(|error| {
                    SearchSessionFailure::debug("RoventPcrConfiguration", &error)
                })?,
            );
            let options = AdaptiveSessionOptions {
                implementation: Digest(hash.finalize().into()),
                identity,
                max_rounds: *max_rounds,
                max_cpu_nodes: *max_native_nodes,
                cpu_depth: *cpu_depth,
                token_budget: configuration.fields().review.token_budget,
                policy: event_order.actual(),
                endpoints: CpuEndpointSelection::deployment(),
                controller: control,
                device_digest: prepared.snapshot().backend,
                cpu_device_digest: Digest(Sha256::digest(b"native-own-board-host-cpu").into()),
                external_policy: ExternalSlicePolicy::default(),
            };
            let profile = RoventPcrSessionProfile {
                mode: prepared.snapshot().mode,
                maximum_goal_plies: NonZeroU16::new(
                    configuration.fields().frontier.maximum_goal_plies,
                )
                .ok_or_else(|| failure("RoventPcrGoal", "zero goal capability"))?,
                input_lock: prepared.snapshot().input_lock,
                root_budget,
                selective_policy: Digest(configuration.search_configuration_digest().map_err(
                    |error| SearchSessionFailure::debug("RoventPcrConfiguration", &error),
                )?),
                model_namespace: Arc::new(namespace),
                frontier_limits: configuration
                    .fields()
                    .frontier
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrFrontier", &error))?,
                reciprocal: reciprocal
                    .actual(prepared.snapshot().mode, root_budget)
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrReciprocal", &error))?,
                peers: peers
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrPeers", &error))?,
            };
            Ok((
                engine,
                cpu,
                metadata,
                registration,
                options,
                queues
                    .actual()
                    .map_err(|error| SearchSessionFailure::debug("RoventPcrQueues", &error))?,
                profile,
            ))
        })();
        let (engine, cpu, metadata, registration, options, limits, profile) = match build {
            Ok(built) => built,
            Err(failure) => {
                return Err(RoventProductStartupRejection {
                    failure,
                    owners: RoventProductStartupOwners::Pcr(prepared),
                    sources: pins,
                });
            }
        };
        let (native, _, _) = prepared.into_parts();
        let sources = AdaptiveSessionSources {
            native,
            engine,
            cpu,
            external: None,
            checker_metadata: metadata,
        };
        let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
        let mut driver = match collection {
            Some((origin, hooks)) => {
                AdaptiveSessionDriver::new_rovent_pcr_with_resource_limits_and_collection_hooks(
                    options,
                    limits,
                    sources,
                    clock,
                    issuer,
                    registration,
                    profile,
                    origin,
                    hooks,
                )
            }
            None => AdaptiveSessionDriver::new_rovent_pcr_with_resource_limits(
                options,
                limits,
                sources,
                clock,
                issuer,
                registration,
                profile,
            ),
        }
        .map_err(|rejected| RoventProductStartupRejection {
            failure: rejected.failure.clone(),
            owners: RoventProductStartupOwners::PcrRejected(rejected),
            sources: Arc::clone(&pins),
        })?;
        if let Err(failure) =
            driver.register_rovent_root_policy(Arc::new(pins.lock().clone()), root_policy)
        {
            return Err(RoventProductStartupRejection {
                failure,
                owners: RoventProductStartupOwners::PcrDriver(Box::new(driver)),
                sources: pins,
            });
        }
        Ok(Self {
            inner: Arc::new(ActualProductDriver::Pcr(driver)),
            sources: pins,
            physically_closed: AtomicBool::new(false),
        })
    }
    fn actual(&self) -> &dyn SearchSessionDriver {
        match self.inner.as_ref() {
            ActualProductDriver::Pc(driver) => driver,
            ActualProductDriver::Pcr(driver) => driver,
            ActualProductDriver::Nnuec(driver) => driver,
        }
    }
}
impl SearchSessionDriver for RoventNativeProductDriver {
    fn supports_ponder_clock(&self) -> bool {
        self.actual().supports_ponder_clock()
    }
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        self.actual().node_budget_semantics()
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        self.actual().validate_go(position, go)
    }
    fn kind(&self) -> SearchKind {
        SearchKind::Rovent
    }
    fn identity(&self) -> EngineIdentity {
        self.actual().identity()
    }
    fn implementation(&self) -> Digest {
        self.actual().implementation()
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.actual().search_goal_capability()
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.actual().reset_game()
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        let result = self.actual().finish_session(until)?;
        if result == Some(DriverPhysicalCompletion::Confirmed) {
            self.physically_closed.store(true, Ordering::Release);
        }
        Ok(result)
    }
    fn accept_uci_output(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.actual()
            .accept_uci_output(authority, bestmove, from_report)
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.actual().run(position, context, progress)
    }
}
impl Drop for RoventNativeProductDriver {
    fn drop(&mut self) {
        if !self.physically_closed.load(Ordering::Acquire) {
            std::mem::forget(Arc::clone(&self.inner));
            std::mem::forget(Arc::clone(&self.sources));
        }
    }
}
fn failure(code: &'static str, detail: &str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Confirmed,
        code,
        detail: detail.into(),
    }
}
