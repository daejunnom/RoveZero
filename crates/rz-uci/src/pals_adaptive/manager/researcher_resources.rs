//! Actual researcher resource kinds. A profiled external owner has neither an
//! internal native credit ledger nor an adaptive native worker controller.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveResearcherSourceKind {
    Native,
    ProfiledExternal,
}

pub(super) struct NativeResearcherResources {
    pub ledger: CpuCreditLedger,
    pub controller: AdaptiveController,
}

pub(super) enum ResearcherResources {
    Native(NativeResearcherResources),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledExternal(Box<profiled_external_source::ProfiledExternalManagerSource>),
}

impl ResearcherResources {
    pub fn kind(&self) -> AdaptiveResearcherSourceKind {
        match self {
            Self::Native(_) => AdaptiveResearcherSourceKind::Native,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Self::ProfiledExternal(_) => AdaptiveResearcherSourceKind::ProfiledExternal,
        }
    }
    pub fn native(&self) -> Result<&NativeResearcherResources, AdaptiveManagerError> {
        match self {
            Self::Native(native) => Ok(native),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Self::ProfiledExternal(_) => Err(AdaptiveManagerError::Invalid(
                "profiled external R has no native ledger/controller",
            )),
        }
    }
    pub fn native_mut(&mut self) -> Result<&mut NativeResearcherResources, AdaptiveManagerError> {
        match self {
            Self::Native(native) => Ok(native),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Self::ProfiledExternal(_) => Err(AdaptiveManagerError::Invalid(
                "profiled external R has no native ledger/controller",
            )),
        }
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn profiled(
        &self,
    ) -> Result<&profiled_external_source::ProfiledExternalManagerSource, AdaptiveManagerError>
    {
        match self {
            Self::ProfiledExternal(source) => Ok(source),
            Self::Native(_) => Err(AdaptiveManagerError::Invalid(
                "native R has no profiled external owner",
            )),
        }
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn profiled_mut(
        &mut self,
    ) -> Result<&mut profiled_external_source::ProfiledExternalManagerSource, AdaptiveManagerError>
    {
        match self {
            Self::ProfiledExternal(source) => Ok(source),
            Self::Native(_) => Err(AdaptiveManagerError::Invalid(
                "native R has no profiled external owner",
            )),
        }
    }
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub fn researcher_source_kind(&self) -> AdaptiveResearcherSourceKind {
        self.researcher.kind()
    }
    pub(super) fn native_ledger(&self) -> Result<&CpuCreditLedger, AdaptiveManagerError> {
        Ok(&self.researcher.native()?.ledger)
    }
    pub(super) fn native_ledger_mut(
        &mut self,
    ) -> Result<&mut CpuCreditLedger, AdaptiveManagerError> {
        Ok(&mut self.researcher.native_mut()?.ledger)
    }
    pub(super) fn native_credit_outstanding(&self) -> Result<bool, AdaptiveManagerError> {
        match &self.researcher {
            ResearcherResources::Native(native) => Ok(native.ledger.global_usage()?.1 != 0),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            ResearcherResources::ProfiledExternal(_) => Ok(false),
        }
    }
    pub(super) fn profiled_owner_closed(&self) -> bool {
        match &self.researcher {
            ResearcherResources::Native(_) => true,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            ResearcherResources::ProfiledExternal(source) => match self.close_mode {
                AdaptiveCloseMode::RootQuiesce => source.root_drained(),
                AdaptiveCloseMode::ProcessShutdown => {
                    source.shutdown_snapshot().is_some_and(|result| {
                        result.as_ref().is_ok_and(|receipt| {
                            receipt.cleanup_complete
                                && receipt.exit_observed
                                && receipt.stdout_drained
                                && receipt.stderr_drained
                                && !receipt.quarantined
                                && !receipt.ownership_lost
                        })
                    })
                }
            },
        }
    }
    pub(super) fn profiled_busy(&self) -> bool {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.profiled_binding.is_some()
            || self.profiled_orphan.is_some()
            || self
                .deferred
                .iter()
                .any(|event| matches!(event, DeferredCommit::ProfiledExternal { .. }))
        {
            return true;
        }
        false
    }
    pub(super) fn profiled_frame_pending(&self, id: FrameId) -> bool {
        #[cfg(all(feature="pals-adaptive",feature="onnx-cpu"))]
        return self.profiled_binding.as_ref().is_some_and(|binding|binding.work.frame==id)
            || self.deferred.iter().any(|event|matches!(event,DeferredCommit::ProfiledExternal{binding,..} if binding.work.frame==id));
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        {
            let _ = id;
            false
        }
    }
    pub(super) fn profiled_dispatch_available(&self) -> bool {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        return self.researcher_source_kind() == AdaptiveResearcherSourceKind::ProfiledExternal
            && !self.profiled_busy()
            && self.researcher.profiled().is_ok_and(|source| {
                source.pending_dispatch().is_none()
                    && !source.is_quarantined()
                    && !source.root_drained()
            });
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        {
            false
        }
    }
}
