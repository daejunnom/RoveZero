//! 원 worker가 발급한 물리 root source를 원 lock/live root와 대조한다.
//! 공개 profile·PID·fence 사본으로 이 결합을 재구성할 수 없다.
use super::*;
use crate::rovent_root_admission::RoventRootAdmission;
use rz_runtime::contracts::ContractClock;
use rz_search::external_cpu::{
    ProfiledExternalCurrentRootFence, ProfiledExternalResetProof, ProfiledExternalRootSource,
};
use std::sync::Arc;

/// 실제 source와 원 live RootAdmission Arc를 함께 보존한다. source getter는
/// 새 실행·ledger credit이나 외부 reported nodes의 native 환산을 발급하지 않는다.
#[derive(Clone)]
pub struct RoventAdmittedExternalRootSource {
    root: Arc<RoventRootAdmission>,
    source: ProfiledExternalRootSource,
}
impl RoventAdmittedExternalRootSource {
    pub fn root_admission(&self) -> &Arc<RoventRootAdmission> {
        &self.root
    }
    pub fn worker_source(&self) -> &ProfiledExternalRootSource {
        &self.source
    }
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.root, &other.root) && self.source.same_source(&other.source)
    }
    pub fn validate_current(&self) -> Result<(), CheckerError> {
        self.source.validate_current()?;
        if !self
            .root
            .original_context()
            .accepts()
            .map_err(|_| invalid("original external root unavailable"))?
        {
            return Err(invalid("original external root no longer admits work"));
        }
        Ok(())
    }
}
impl RoventExternalOwner {
    /// Startup 대기 중에는 None이다. 실제 worker source, 원 process/profile,
    /// root authority·clock·startup lock이 모두 같은 경우만 결합한다.
    pub fn frontier_root_source(
        &self,
    ) -> Result<Option<RoventAdmittedExternalRootSource>, CheckerError> {
        let root = self
            .actual_root_admission()
            .ok_or(invalid("actual root not admitted"))?;
        let Some(source) = self.worker.current_root_source()? else {
            return Ok(None);
        };
        let startup = self
            .verified_startup()?
            .ok_or(invalid("actual external startup missing"))?;
        let profile = self.launch.profile();
        let clock = source.clock();
        if profile.purpose != ExternalEnginePurpose::ResearchChecker
            || source.purpose() != profile.purpose
            || source.profile() != profile.engine
            || source.authority() != root.authority()
            || source.process_identity() != startup.process_identity()
            || clock.domain != root.runtime_clock().domain()
            || clock.origin != root.runtime_clock().origin()
            || self
                .launch
                .input_lock_sha256
                .as_deref()
                .map(digest)
                .transpose()?
                != Some(root.startup_lock_digest())
            || !self.worker.configuration_matches(
                &self.launch.configuration()?,
                profile.engine,
                profile.purpose,
                source.worker_owner(),
            )
        {
            return Err(invalid(
                "worker source differs from original launch/lock/root/clock",
            ));
        }
        let joined = RoventAdmittedExternalRootSource {
            root: Arc::clone(root),
            source,
        };
        joined.validate_current()?;
        Ok(Some(joined))
    }
    pub fn request_actual_root_quiesce(&self, until: Instant) -> Result<(), CheckerError> {
        let root = self
            .actual_root_admission()
            .ok_or(invalid("actual root not admitted"))?;
        let ceiling = root
            .runtime_clock()
            .origin()
            .checked_add(std::time::Duration::from_nanos(
                root.cleanup_deadline().at.0,
            ))
            .ok_or(invalid("external original cleanup ceiling overflow"))?;
        if until > ceiling {
            return Err(invalid("external quiesce exceeds original cleanup ceiling"));
        }
        self.worker.request_root_quiesce(root.authority(), until)
    }
    pub fn actual_root_quiesce_result(
        &self,
    ) -> Option<Result<ProfiledExternalCurrentRootFence, CheckerError>> {
        self.worker.root_quiesce_result()
    }
    pub fn initial_reset_proof(&self) -> Result<Option<ProfiledExternalResetProof>, CheckerError> {
        self.worker.initial_reset_proof()
    }
    pub fn current_reset_proof(&self) -> Option<ProfiledExternalResetProof> {
        self.worker.current_reset_proof()
    }
}
