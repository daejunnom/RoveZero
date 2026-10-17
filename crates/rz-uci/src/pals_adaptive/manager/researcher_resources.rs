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
}
