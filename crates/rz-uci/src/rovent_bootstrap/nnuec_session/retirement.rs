//! Return the same semantic owner only after actual NNUEC retirement.
//!
//! The parent first obtains its original manager's physical root receipt and
//! moves the resulting port into its same engine. Each step here crosses at
//! most one nonblocking actor boundary. The parent retains its cleanup deadline
//! and UCI gate; this helper never waits, joins an actor, or creates an owner.

use crate::pals_adaptive::{
    AdaptiveCheckerMetadata, AdaptiveManagerError, AdaptiveNnuecRetiredOwner,
    CompletedUtilityRouteAction, UtilityRouteFinalizationRejection, UtilityRoutePendingClosure,
};
use rz_contracts::Digest;
use rz_search::cpu_value::nnuec::v2::retirement::{NnuecCacheInvalidation, NnuecCacheReleaseAck};
use rz_search::cpu_value::nnuec::v2::review::{PreparationActor, PreparationError, Provenance};
use rz_search::cpu_worker::{
    CpuCreditLedger, CpuWorkerError, CpuWorkerProfile, DeferredNnuecRSource, NnuecEngineReleaseAck,
    NnuecEngineRetirementCompletion, NnuecEngineRetirementId, NnuecReadyRegistration,
    RecoveredNnuecEngine, ValueInstallReceipt,
};
use rz_search::pals::engine::adaptive_port::{
    AdaptivePortModel, AdaptivePortQuestion, AdaptivePreparingRootCustody, PreparingRootViewSeal,
};
use rz_search::pals::engine::preparing_root::{
    ReleasedResearcherReadback, ResearcherPreparingOwner, ResearcherRetirementRejection,
};
use rz_search::pals::engine::{PalsEngine, PalsError};
use rz_search::pals::review_context::ReviewView;
use rz_search::pals::rovent_engine::RootMoveDomain;
use rz_search::pals::store::StoreError;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

/// Original handoff pins, kept separately while its same ledger is in manager
/// custody. There is no metadata-only constructor for this transport.
#[must_use = "retain original NNUEC Ready/View custody through actual retirement"]
pub struct NnuecRootRetirementPins {
    source: DeferredNnuecRSource,
    pins: Option<OriginalRootPins>,
}

struct OriginalRootPins {
    ready: NnuecReadyRegistration,
    original_view: Arc<ReviewView>,
    original_review_seal: PreparingRootViewSeal,
    root_moves: Arc<RootMoveDomain>,
    metadata: AdaptiveCheckerMetadata,
    cancellation: Arc<AtomicBool>,
}

impl NnuecRootRetirementPins {
    /// Move every real handoff owner once. The original ledger and controls
    /// go directly to manager admission; the other original members stay here.
    pub fn from_custody(
        custody: AdaptivePreparingRootCustody,
        metadata: AdaptiveCheckerMetadata,
    ) -> (CpuCreditLedger, Digest, Instant, Self) {
        let AdaptivePreparingRootCustody {
            ledger,
            root_budget,
            cleanup_deadline,
            original_view,
            original_review_seal,
            ready,
            root_moves,
            source,
            cancellation,
            ..
        } = custody;
        (
            ledger,
            root_budget,
            cleanup_deadline,
            Self {
                source,
                pins: Some(OriginalRootPins {
                    ready,
                    original_view,
                    original_review_seal,
                    root_moves,
                    metadata,
                    cancellation,
                }),
            },
        )
    }

    /// Called only after matching actual engine and cache acknowledgements.
    fn release_after_acknowledgements(mut self) {
        if let Some(OriginalRootPins {
            ready,
            original_view,
            original_review_seal,
            root_moves,
            metadata,
            cancellation,
        }) = self.pins.take()
        {
            drop(ready);
            drop(original_review_seal);
            drop(original_view);
            drop(root_moves);
            drop(metadata);
            // This releases only the handoff's original Arc. The parent's
            // control remains the sole source of cancellation decisions.
            drop(cancellation);
        }
    }
}

impl Drop for NnuecRootRetirementPins {
    fn drop(&mut self) {
        // A discarded handoff is not evidence that its actor/evaluator is
        // gone. Retain the original pins unless the ack path released them.
        if let Some(pins) = self.pins.take() {
            std::mem::forget(pins);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NnuecRootRetirementPhase {
    ValidateOriginalOwners,
    SubmitEngine,
    AwaitEngine,
    InvalidateCache,
    AwaitCache,
    ReleaseOriginalPins,
    AwaitPinZero,
    FinalizeUtilityRoute,
    ReturnPreparingOwner,
}

#[derive(Debug)]
pub enum NnuecRootRetirementError {
    Preparation(PreparationError),
    Worker(CpuWorkerError),
    Store(StoreError),
    Semantic(PalsError),
    UtilityRoute(AdaptiveManagerError),
    Invalid(&'static str),
}

impl std::fmt::Display for NnuecRootRetirementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The original typed payload remains in the retained failure. This
        // projection reports the actual source rather than a generic release
        // error; it grants no release, recovery or physical completion proof.
        match self {
            Self::Preparation(error) => write!(f, "preparation: {error:?}"),
            Self::Worker(error) => write!(f, "worker: {error:?}"),
            Self::Store(error) => write!(f, "store: {error:?}"),
            Self::Semantic(error) => write!(f, "semantic: {error:?}"),
            Self::UtilityRoute(error) => write!(f, "utility route: {error:?}"),
            Self::Invalid(reason) => write!(f, "invalid retirement: {reason}"),
        }
    }
}

struct ExpectedEngineRelease {
    profile: CpuWorkerProfile,
    stack_owner: u64,
    binding: Digest,
    provenance: Provenance,
}

/// All actual owners and one-use cleanup observations remain together. A
/// missing local engine after submission means actor custody, not release.
struct RetirementOwners {
    actor: Option<PreparationActor>,
    engine: Option<PalsEngine<AdaptivePortModel>>,
    recovered: Option<RecoveredNnuecEngine>,
    last_installation: Option<ValueInstallReceipt>,
    pins: Option<NnuecRootRetirementPins>,
    source: DeferredNnuecRSource,
    expected: Option<ExpectedEngineRelease>,
    retirement: Option<NnuecEngineRetirementId>,
    engine_release: Option<NnuecEngineReleaseAck>,
    invalidation: Option<NnuecCacheInvalidation>,
    cache_release: Option<NnuecCacheReleaseAck>,
    unmatched_completion: Option<NnuecEngineRetirementCompletion>,
    utility_route_pending: Option<Box<UtilityRoutePendingClosure<AdaptivePortQuestion>>>,
    utility_route: Option<Box<CompletedUtilityRouteAction>>,
    utility_route_rejection: Option<Box<UtilityRouteFinalizationRejection<AdaptivePortQuestion>>>,
}

#[must_use = "a retirement failure still owns the same engine/actor custody"]
pub struct NnuecRootRetirementFailure {
    retirement: NnuecRootRetirement,
    error: NnuecRootRetirementError,
}

impl NnuecRootRetirementFailure {
    pub fn phase(&self) -> NnuecRootRetirementPhase {
        self.retirement.phase()
    }

    pub fn error(&self) -> &NnuecRootRetirementError {
        &self.error
    }
}

/// The same Store owner and actor, after both physical acknowledgements and
/// actual pin zero. This transport makes no claim about the root's result.
#[must_use = "retain the same semantic owner and preparation actor"]
pub struct NnuecRootRetired {
    pub owner: ResearcherPreparingOwner<AdaptivePortModel>,
    pub actor: PreparationActor,
    pub engine_release: NnuecEngineReleaseAck,
    pub cache_release: NnuecCacheReleaseAck,
    pub utility_route: Option<CompletedUtilityRouteAction>,
}

#[must_use = "pending and failure variants retain actual cleanup obligations"]
pub enum NnuecRootRetirementStep {
    Pending(NnuecRootRetirement),
    Ready(NnuecRootRetired),
    RetainedFailure(NnuecRootRetirementFailure),
}

#[must_use = "retain this original retirement or transfer its retained custody"]
pub struct NnuecRootRetirement {
    phase: NnuecRootRetirementPhase,
    owners: Option<RetirementOwners>,
}

impl NnuecRootRetirement {
    /// The parent supplies the exact joined worker owner, handoff pins and
    /// engine recovered after its actual manager/driver root fence.
    pub fn new(
        actor: PreparationActor,
        retired: AdaptiveNnuecRetiredOwner,
        pins: NnuecRootRetirementPins,
        engine: PalsEngine<AdaptivePortModel>,
    ) -> Self {
        let source = pins.source.clone();
        Self {
            phase: NnuecRootRetirementPhase::ValidateOriginalOwners,
            owners: Some(RetirementOwners {
                actor: Some(actor),
                engine: Some(engine),
                recovered: Some(retired.engine),
                last_installation: retired.last_installation,
                pins: Some(pins),
                source,
                expected: None,
                retirement: None,
                engine_release: None,
                invalidation: None,
                cache_release: None,
                unmatched_completion: None,
                utility_route_pending: None,
                utility_route: None,
                utility_route_rejection: None,
            }),
        }
    }

    pub fn phase(&self) -> NnuecRootRetirementPhase {
        self.phase
    }

    /// Move the manager's original unfinished whole-cost owner alongside the
    /// same actor/engine custody before retirement begins. A rejected attach
    /// returns that exact pending owner without finalizing or discarding it.
    #[allow(clippy::result_large_err)]
    pub fn attach_utility_route_pending(
        &mut self,
        pending: UtilityRoutePendingClosure<AdaptivePortQuestion>,
    ) -> Result<
        (),
        (
            NnuecRootRetirementError,
            UtilityRoutePendingClosure<AdaptivePortQuestion>,
        ),
    > {
        if self.phase != NnuecRootRetirementPhase::ValidateOriginalOwners {
            return Err((
                NnuecRootRetirementError::Invalid(
                    "utility route pending must attach before original retirement",
                ),
                pending,
            ));
        }
        let owners = self.owners_mut();
        if owners.utility_route_pending.is_some()
            || owners.utility_route.is_some()
            || owners.utility_route_rejection.is_some()
        {
            return Err((
                NnuecRootRetirementError::Invalid(
                    "original utility route retirement custody already attached",
                ),
                pending,
            ));
        }
        owners.utility_route_pending = Some(Box::new(pending));
        Ok(())
    }

    fn owners(&self) -> &RetirementOwners {
        self.owners.as_ref().expect("retirement retains its owners")
    }

    fn owners_mut(&mut self) -> &mut RetirementOwners {
        self.owners.as_mut().expect("retirement retains its owners")
    }

    fn fail(self, error: NnuecRootRetirementError) -> NnuecRootRetirementStep {
        NnuecRootRetirementStep::RetainedFailure(NnuecRootRetirementFailure {
            retirement: self,
            error,
        })
    }

    fn invalid(self, detail: &'static str) -> NnuecRootRetirementStep {
        self.fail(NnuecRootRetirementError::Invalid(detail))
    }

    /// One bounded step only. Busy/None and outstanding pins preserve the
    /// same custody for the parent's next gate/deadline check.
    pub fn step(mut self) -> NnuecRootRetirementStep {
        match self.phase {
            NnuecRootRetirementPhase::ValidateOriginalOwners => {
                let owners = self.owners();
                let (Some(recovered), Some(pins)) =
                    (owners.recovered.as_ref(), owners.pins.as_ref())
                else {
                    return self.invalid("original joined NNUEC engine or root pins absent");
                };
                let Some(original) = pins.pins.as_ref() else {
                    return self.invalid("original NNUEC Ready pins already released");
                };
                let ready = &original.ready;
                if !pins.source.same_authority(&owners.source)
                    || !ready.matches_source(&owners.source)
                    || !recovered.matches_source(&owners.source)
                    || ready.profile().owner != recovered.owner()
                    || ready.value_identity() != recovered.model_identity()
                    || ready.value_identity() != owners.source.model().identity()
                    || ready.provenance().root != recovered.provenance().root
                    || ready.provenance().generation != recovered.provenance().generation
                {
                    return self.invalid("joined NNUEC owner differs from original root/source");
                }
                let (profile, stack_owner, binding) = match owners.last_installation.as_ref() {
                    Some(receipt) => match receipt.result() {
                        Ok(readback) => (
                            readback.profile(),
                            readback.actual_stack_owner_id(),
                            receipt.binding_digest(),
                        ),
                        Err(error) => {
                            let error = error.clone();
                            return self.fail(NnuecRootRetirementError::Worker(error));
                        }
                    },
                    None => (
                        ready.profile(),
                        ready.actual_stack_owner_id(),
                        ready.binding_digest(),
                    ),
                };
                if profile != recovered.profile()
                    || profile.owner != ready.profile().owner
                    || profile.evaluation_namespace != binding
                    || stack_owner == 0
                    || stack_owner != recovered.actual_stack_owner_id()
                    || binding != recovered.binding_digest()
                {
                    return self.invalid("last actual NNUEC readback differs from joined owner");
                }
                let expected = ExpectedEngineRelease {
                    profile,
                    stack_owner,
                    binding,
                    provenance: recovered.provenance().clone(),
                };
                self.owners_mut().expected = Some(expected);
                self.phase = NnuecRootRetirementPhase::SubmitEngine;
            }
            NnuecRootRetirementPhase::SubmitEngine => {
                let owners = self.owners_mut();
                if owners.actor.is_none() || owners.recovered.is_none() {
                    return self.invalid("original actor or joined NNUEC engine absent");
                }
                let actor = owners.actor.as_ref().unwrap();
                let recovered = owners.recovered.take().unwrap();
                match actor.try_retire_engine(recovered) {
                    Ok(id) => {
                        owners.retirement = Some(id);
                        self.phase = NnuecRootRetirementPhase::AwaitEngine;
                    }
                    Err((PreparationError::Busy, recovered)) => {
                        owners.recovered = Some(recovered);
                    }
                    Err((error, recovered)) => {
                        owners.recovered = Some(recovered);
                        return self.fail(NnuecRootRetirementError::Preparation(error));
                    }
                }
            }
            NnuecRootRetirementPhase::AwaitEngine => {
                let owners = self.owners_mut();
                let (Some(actor), Some(retirement)) =
                    (owners.actor.as_ref(), owners.retirement.as_ref())
                else {
                    return self.invalid("same actor retirement request absent");
                };
                match actor.try_retired_engine() {
                    Ok(None) => {}
                    Err(error) => {
                        return self.fail(NnuecRootRetirementError::Preparation(error));
                    }
                    Ok(Some(completion)) => {
                        if completion.request_id() != retirement {
                            owners.unmatched_completion = Some(completion);
                            return self.invalid("actor returned a different retirement terminal");
                        }
                        match completion.into_result() {
                            Ok(release) => {
                                let matches = owners.expected.as_ref().is_some_and(|expected| {
                                    release.matches_source(&owners.source)
                                        && release.profile() == expected.profile
                                        && release.actual_stack_owner_id() == expected.stack_owner
                                        && release.binding_digest() == expected.binding
                                        && release.provenance() == &expected.provenance
                                        && release.model_identity()
                                            == owners.source.model().identity()
                                });
                                owners.engine_release = Some(release);
                                if !matches {
                                    return self.invalid(
                                        "actual engine release differs from submitted owner",
                                    );
                                }
                                self.phase = NnuecRootRetirementPhase::InvalidateCache;
                            }
                            Err((recovered, error)) => {
                                owners.recovered = Some(recovered);
                                return self.fail(NnuecRootRetirementError::Worker(error));
                            }
                        }
                    }
                }
            }
            NnuecRootRetirementPhase::InvalidateCache => {
                let owners = self.owners_mut();
                let Some(actor) = owners.actor.as_ref() else {
                    return self.invalid("same NNUEC cache actor absent");
                };
                match actor.invalidate_context_cache() {
                    Ok(invalidation) => {
                        owners.invalidation = Some(invalidation);
                        self.phase = NnuecRootRetirementPhase::AwaitCache;
                    }
                    Err(error) => {
                        return self.fail(NnuecRootRetirementError::Preparation(error));
                    }
                }
            }
            NnuecRootRetirementPhase::AwaitCache => {
                let owners = self.owners_mut();
                let (Some(actor), Some(invalidation)) =
                    (owners.actor.as_ref(), owners.invalidation.as_ref())
                else {
                    return self.invalid("same actor cache invalidation request absent");
                };
                match actor.try_context_cache_release(invalidation) {
                    Ok(None) => {}
                    Err(error) => {
                        return self.fail(NnuecRootRetirementError::Preparation(error));
                    }
                    Ok(Some(cache)) => {
                        let matches = owners
                            .engine_release
                            .as_ref()
                            .is_some_and(|release| release.matches_cache_release(&cache));
                        owners.cache_release = Some(cache);
                        if !matches {
                            return self
                                .invalid("actual cache release does not clear original evaluator");
                        }
                        self.phase = NnuecRootRetirementPhase::ReleaseOriginalPins;
                    }
                }
            }
            NnuecRootRetirementPhase::FinalizeUtilityRoute => {
                let owners = self.owners_mut();
                let (Some(release), Some(cache)) = (
                    owners.engine_release.as_ref(),
                    owners.cache_release.as_ref(),
                ) else {
                    return self
                        .invalid("utility finalization requires actual release acknowledgements");
                };
                if !release.matches_source(&owners.source)
                    || !release.matches_cache_release(cache)
                    || owners.pins.is_some()
                    || owners.utility_route.is_some()
                    || owners.utility_route_rejection.is_some()
                {
                    return self
                        .invalid("utility finalization differs from original retirement custody");
                }
                if let Some(pending) = owners.utility_route_pending.take() {
                    // Original handoff pins and actual Store pin zero precede
                    // this original pending owner's final cost boundary. The
                    // leaf observes that boundary; no elapsed duration or
                    // replacement cost proof is supplied here.
                    match (*pending).finish_with_nnuec_release(
                        &owners.source,
                        std::slice::from_ref(release),
                        cache,
                    ) {
                        Ok(completed) => owners.utility_route = Some(Box::new(completed)),
                        Err(rejection) => {
                            let error = rejection.error.clone();
                            // The whole rejected pending/capability/cost owner,
                            // including any leaf closure proof, remains beside
                            // the original actor, semantic engine and actual
                            // engine/cache release acknowledgements.
                            owners.utility_route_rejection = Some(Box::new(rejection));
                            return self.fail(NnuecRootRetirementError::UtilityRoute(error));
                        }
                    }
                }
                self.phase = NnuecRootRetirementPhase::ReturnPreparingOwner;
            }
            NnuecRootRetirementPhase::ReleaseOriginalPins => {
                let owners = self.owners_mut();
                let witnessed = match (
                    owners.engine_release.as_ref(),
                    owners.cache_release.as_ref(),
                ) {
                    (Some(release), Some(cache)) => {
                        release.matches_source(&owners.source)
                            && release.matches_cache_release(cache)
                    }
                    _ => false,
                };
                if !witnessed {
                    return self
                        .invalid("original pins require both actual release acknowledgements");
                }
                let Some(pins) = owners.pins.take() else {
                    return self.invalid("original NNUEC pins absent before acknowledged release");
                };
                pins.release_after_acknowledgements();
                self.phase = NnuecRootRetirementPhase::AwaitPinZero;
            }
            NnuecRootRetirementPhase::AwaitPinZero => {
                let Some(engine) = self.owners().engine.as_ref() else {
                    return self.invalid("same semantic engine absent before pin check");
                };
                match engine.stores().review_pin_count() {
                    Ok(0) => {
                        self.phase = if self.owners().utility_route_pending.is_some() {
                            NnuecRootRetirementPhase::FinalizeUtilityRoute
                        } else {
                            NnuecRootRetirementPhase::ReturnPreparingOwner
                        };
                    }
                    Ok(_) => {}
                    Err(error) => return self.fail(NnuecRootRetirementError::Store(error)),
                }
            }
            NnuecRootRetirementPhase::ReturnPreparingOwner => {
                let owners = self.owners_mut();
                if owners.engine.is_none()
                    || owners.engine_release.is_none()
                    || owners.cache_release.is_none()
                    || owners.actor.is_none()
                    || owners.pins.is_some()
                    || owners.recovered.is_some()
                    || owners.unmatched_completion.is_some()
                    || owners.utility_route_pending.is_some()
                    || owners.utility_route_rejection.is_some()
                {
                    return self.invalid("retirement owner or acknowledgement custody incomplete");
                }
                // Recheck after the parent's intervening gate call. The same
                // Store, not a new or metadata-only owner, must still be free.
                match owners.engine.as_ref().unwrap().stores().review_pin_count() {
                    Ok(0) => {}
                    Ok(_) => {
                        self.phase = NnuecRootRetirementPhase::AwaitPinZero;
                        return NnuecRootRetirementStep::Pending(self);
                    }
                    Err(error) => return self.fail(NnuecRootRetirementError::Store(error)),
                }
                let engine = owners.engine.take().unwrap();
                let release = owners.engine_release.take().unwrap();
                let cache = owners.cache_release.take().unwrap();
                let readback = match owners.last_installation.as_ref() {
                    Some(receipt) => ReleasedResearcherReadback::Installed(receipt),
                    None => ReleasedResearcherReadback::Initial,
                };
                match engine.into_preparing_owner(release, cache, readback) {
                    Ok(retired) => {
                        let mut owners = self.owners.take().unwrap();
                        let actor = owners.actor.take().unwrap();
                        return NnuecRootRetirementStep::Ready(NnuecRootRetired {
                            owner: retired.owner,
                            actor,
                            engine_release: retired.engine_release,
                            cache_release: retired.cache_release,
                            utility_route: owners.utility_route.take().map(|completed| *completed),
                        });
                    }
                    Err(ResearcherRetirementRejection {
                        engine,
                        engine_release,
                        cache_release,
                        error,
                    }) => {
                        owners.engine = Some(engine);
                        owners.engine_release = Some(engine_release);
                        owners.cache_release = Some(cache_release);
                        return self.fail(NnuecRootRetirementError::Semantic(error));
                    }
                }
            }
        }
        NnuecRootRetirementStep::Pending(self)
    }
}

impl Drop for NnuecRootRetirement {
    fn drop(&mut self) {
        // PreparationActor::drop joins its worker and RecoveredNnuecEngine's
        // normal destructor would release its engine on this caller. Neither
        // is an actual retirement observation. Preserve the whole original
        // custody if a pending/failure transport is accidentally discarded.
        if let Some(owners) = self.owners.take() {
            std::mem::forget(owners);
        }
    }
}
