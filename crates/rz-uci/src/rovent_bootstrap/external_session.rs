//! Product assembly for a pinned external R and the original native P/C owner.
//! CPU_T has an independent Collection actor; no native R/PVS engine is created.

use super::configuration::ExecutionConfiguration;
use super::{
    PreparedRoventNative, RoventNativeBackend, RoventPinnedExternalRSource,
    RoventPinnedNativeSources,
};
use crate::EngineIdentity;
use crate::engine::{OwnerRegistry, ProcessClock};
use crate::pals_adaptive::{
    AdaptiveSessionCollectionHooks, AdaptiveSessionDriver, AdaptiveSessionOptions,
    AdaptiveSessionSources, RoventExternalSessionAdmissionRejection, RoventExternalSessionSource,
    RoventExternalSessionSources, RoventExternalSourcePreparationRejection,
    RoventPcrSessionProfile,
};
use crate::rovent_root_admission::RoventRootAdmissionPolicy;
use crate::search_driver::{PalsCheckerRegistration, SearchSessionFailure};
use rz_contracts::Digest;
use rz_runtime::adaptive::ControllerConfig;
use rz_search::external_cpu::{CpuEndpoint, CpuEndpointSelection};
use rz_search::pals::engine::adaptive_port::profiled_retirement_archive::RegisteredProfiledRetirementSink;
use rz_search::pals::engine::adaptive_port::{
    AdaptivePortModel, AdaptivePortQuestion, AdaptiveRulesOwnerIssuer,
};
use rz_search::pals::engine::{PalsConfig, PalsEngine, ResearcherSemanticProfile, ResolverPolicy};
use sha2::{Digest as _, Sha256};
use std::{num::NonZeroU16, sync::Arc};

pub type RoventExternalProductSession =
    AdaptiveSessionDriver<RoventNativeBackend<AdaptivePortQuestion>>;

/// Every rejected transition retains its original physical/native/FD owners.
pub enum RoventExternalProductStartupOwners {
    Prepared {
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        external: RoventPinnedExternalRSource,
    },
    SourcePreparation {
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        rejection: RoventExternalSourcePreparationRejection,
    },
    PreparedSource {
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        external: RoventExternalSessionSource,
    },
    Session(RoventExternalSessionAdmissionRejection<RoventNativeBackend<AdaptivePortQuestion>>),
    Driver(Box<RoventExternalProductSession>),
}

pub struct RoventExternalProductStartupRejection {
    pub failure: SearchSessionFailure,
    pub owners: RoventExternalProductStartupOwners,
    pub sources: Arc<RoventPinnedNativeSources>,
    /// Early failures retain the admitted managed-output capability here.
    /// A Session/Driver owner already retains it and therefore leaves None.
    pub retirement_sink: Option<RegisteredProfiledRetirementSink>,
}

/// Construct the shared session only. Its first live, original go creates the
/// external process from the retained FDs; a stopped-initial root cannot do so.
#[allow(clippy::too_many_arguments, clippy::result_large_err)]
pub fn external_researcher_session(
    prepared: PreparedRoventNative<AdaptivePortQuestion>,
    external: RoventPinnedExternalRSource,
    retirement_sink: RegisteredProfiledRetirementSink,
    clock: ProcessClock,
    owners: Arc<OwnerRegistry>,
    identity: EngineIdentity,
    root_policy: RoventRootAdmissionPolicy,
    hooks: Option<AdaptiveSessionCollectionHooks<RoventNativeBackend<AdaptivePortQuestion>>>,
) -> Result<RoventExternalProductSession, RoventExternalProductStartupRejection> {
    let pins = Arc::clone(prepared.sources());
    if external.input_lock_digest() != prepared.snapshot().input_lock
        || external.source_purpose() != rz_experiments::RoventExternalPurpose::ResearchChecker
        || !prepared.snapshot().mode.has_native_r()
        || prepared.snapshot().process_epoch != clock.epoch().0
        || !matches!(
            &pins.configuration().fields().execution,
            ExecutionConfiguration::ExternalResearcher { .. }
        )
    {
        return Err(RoventExternalProductStartupRejection {
            failure: failure(
                "RoventExternalProductSources",
                "original native/R profile source or mode differs",
            ),
            owners: RoventExternalProductStartupOwners::Prepared { prepared, external },
            sources: pins,
            retirement_sink: Some(retirement_sink),
        });
    }
    match (pins.lock().manifest.purpose, hooks) {
        (rz_experiments::RoventPurpose::Collection, _) => {
            return Err(RoventExternalProductStartupRejection {
                failure: failure(
                    "RoventExternalCollectionUnsupported",
                    "external R requires a separately admitted Collection provenance source",
                ),
                owners: RoventExternalProductStartupOwners::Prepared { prepared, external },
                sources: pins,
                retirement_sink: Some(retirement_sink),
            });
        }
        (_, Some(_)) => {
            return Err(RoventExternalProductStartupRejection {
                failure: failure(
                    "RoventExternalCollectionHooks",
                    "a non-Collection startup cannot consume Collection hooks",
                ),
                owners: RoventExternalProductStartupOwners::Prepared { prepared, external },
                sources: pins,
                retirement_sink: Some(retirement_sink),
            });
        }
        (_, None) => {}
    }
    let external = match RoventExternalSessionSource::from_pinned(external) {
        Ok(source) => source,
        Err(rejection) => {
            return Err(RoventExternalProductStartupRejection {
                failure: SearchSessionFailure::debug("RoventExternalDeclaration", &rejection.error),
                owners: RoventExternalProductStartupOwners::SourcePreparation {
                    prepared,
                    rejection,
                },
                sources: pins,
                retirement_sink: Some(retirement_sink),
            });
        }
    };
    let build = (|| {
        let configuration = pins.configuration();
        let ExecutionConfiguration::ExternalResearcher {
            event_order,
            queues,
            max_rounds,
            max_reported_nodes,
            cpu_depth,
            cpu_nodes_per_task,
            slice,
            peers,
            ..
        } = &configuration.fields().execution
        else {
            unreachable!("checked execution kind")
        };
        let semantic = &configuration.fields().semantic;
        let namespace = prepared
            .snapshot()
            .model_namespace()
            .map_err(|error| SearchSessionFailure::debug("RoventExternalNamespace", &error))?;
        let model = AdaptivePortModel::new_with_namespace(
            prepared.snapshot().role_model_identity(),
            Some(
                prepared
                    .snapshot()
                    .model_value_identity()
                    .map_err(|error| SearchSessionFailure::debug("RoventExternalValue", &error))?,
            ),
            namespace.clone(),
        )
        .map_err(|error| SearchSessionFailure::debug("RoventExternalModel", &error))?;
        let metadata = external.declaration().clone();
        let mirror = metadata.clone().into_mirror();
        let resolver = ResolverPolicy::ModelWdlRestricted;
        let recheck = configuration.post_repair_policy_for(prepared.snapshot().mode);
        let registration =
            PalsCheckerRegistration::capture_with_resolver(&mirror, &model, resolver)?;
        let engine = PalsEngine::new_with_boxed_profiled_researcher_and_rovent_profile(
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
            external.admission(),
        )
        .map_err(|error| SearchSessionFailure::debug("RoventExternalSemantic", &error))?
        .with_selected_review_context(
            configuration
                .fields()
                .review
                .actual()
                .map_err(|error| SearchSessionFailure::debug("RoventExternalReview", &error))?,
            configuration
                .fields()
                .index
                .actual()
                .map_err(|error| SearchSessionFailure::debug("RoventExternalIndex", &error))?,
        )
        .map_err(|error| SearchSessionFailure::debug("RoventExternalReview", &error))?;
        if engine.post_repair_recheck_policy() != recheck {
            return Err(failure(
                "RoventExternalRecheck",
                "effective recheck differs from registered configuration",
            ));
        }
        let root_budget = Digest(
            rz_eval::asset::parse_sha256(
                &pins.lock().manifest.coordinator.original_root_budget_sha256,
            )
            .map_err(|error| SearchSessionFailure::debug("RoventExternalBudget", &error))?,
        );
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-external-product/1\0");
        hash.update(include_bytes!("external_session.rs"));
        hash.update(super::compiled_bootstrap_source_digest());
        hash.update(
            configuration.configuration_digest().map_err(|error| {
                SearchSessionFailure::debug("RoventExternalConfiguration", &error)
            })?,
        );
        let options = AdaptiveSessionOptions {
            implementation: Digest(hash.finalize().into()),
            identity,
            max_rounds: *max_rounds,
            // This is the external reported-work ceiling. The separate typed
            // manager never creates or charges a strict native CPU ledger.
            max_cpu_nodes: *max_reported_nodes,
            cpu_depth: *cpu_depth,
            token_budget: configuration.fields().review.token_budget,
            policy: event_order.actual(),
            // The separately owned Collection CPU_T is never this R worker.
            endpoints: CpuEndpointSelection {
                cpu_r: CpuEndpoint::ExternalUci,
                ..CpuEndpointSelection::deployment()
            },
            // Unused native-controller declaration is explicitly zero. Only
            // the external typed manager admits this shape; old guards remain.
            controller: ControllerConfig {
                policy: event_order.actual(),
                initial_workers: 0,
                max_workers: 0,
                ..ControllerConfig::default()
            },
            device_digest: prepared.snapshot().backend,
            cpu_device_digest: Digest(Sha256::digest(b"profiled-external-r-host-process").into()),
            external_policy: slice
                .actual()
                .map_err(|error| SearchSessionFailure::debug("RoventExternalSlice", &error))?,
        };
        let profile = RoventPcrSessionProfile {
            mode: prepared.snapshot().mode,
            maximum_goal_plies: NonZeroU16::new(configuration.fields().frontier.maximum_goal_plies)
                .ok_or_else(|| failure("RoventExternalGoal", "zero registered goal capability"))?,
            input_lock: prepared.snapshot().input_lock,
            root_budget,
            selective_policy: Digest(configuration.search_configuration_digest().map_err(
                |error| SearchSessionFailure::debug("RoventExternalConfiguration", &error),
            )?),
            model_namespace: Arc::new(namespace),
            frontier_limits: configuration
                .fields()
                .frontier
                .actual()
                .map_err(|error| SearchSessionFailure::debug("RoventExternalFrontier", &error))?,
            // Shared profile layout only. The external attachment never
            // installs this own-R callback policy; actual source capabilities
            // keep external reciprocal callbacks typed unsupported.
            reciprocal: crate::pals_adaptive::RoventReciprocalPolicy {
                mode: prepared.snapshot().mode,
                root_budget,
                recipient: rz_contracts::rovent::SearchRecipient::Proposer,
                max_questions_per_job: 1,
                node_interval: 1,
                first_question_sequence: 1,
            },
            peers: peers
                .actual()
                .map_err(|error| SearchSessionFailure::debug("RoventExternalPeers", &error))?,
        };
        Ok((
            engine,
            metadata,
            registration,
            options,
            queues
                .actual()
                .map_err(|error| SearchSessionFailure::debug("RoventExternalQueues", &error))?,
            profile,
        ))
    })();
    let (engine, metadata, registration, options, limits, profile) = match build {
        Ok(built) => built,
        Err(failure) => {
            return Err(RoventExternalProductStartupRejection {
                failure,
                owners: RoventExternalProductStartupOwners::PreparedSource { prepared, external },
                sources: pins,
                retirement_sink: Some(retirement_sink),
            });
        }
    };
    let (native, _, _) = prepared.into_parts();
    let hooks = AdaptiveSessionCollectionHooks::default();
    let sources = RoventExternalSessionSources {
        common: AdaptiveSessionSources {
            native,
            engine,
            cpu: Vec::new(),
            external: None,
            checker_metadata: metadata,
        },
        external,
    };
    let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
    let mut driver = AdaptiveSessionDriver::new_rovent_external_with_resource_limits(
        options,
        limits,
        sources,
        clock,
        issuer,
        registration,
        profile,
        retirement_sink,
        None,
        hooks,
    )
    .map_err(|rejected| RoventExternalProductStartupRejection {
        failure: rejected.failure.clone(),
        owners: RoventExternalProductStartupOwners::Session(rejected),
        sources: Arc::clone(&pins),
        retirement_sink: None,
    })?;
    if let Err(failure) =
        driver.register_rovent_root_policy(Arc::new(pins.lock().clone()), root_policy)
    {
        return Err(RoventExternalProductStartupRejection {
            failure,
            owners: RoventExternalProductStartupOwners::Driver(Box::new(driver)),
            sources: pins,
            retirement_sink: None,
        });
    }
    Ok(driver)
}

fn failure(code: &'static str, detail: &'static str) -> SearchSessionFailure {
    SearchSessionFailure {
        code,
        detail: detail.into(),
        physical_completion: crate::search_driver::DriverPhysicalCompletion::Confirmed,
    }
}
