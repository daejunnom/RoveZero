//! 실제 per-go root를 원 외부 process fence 뒤에만 등록한다.
//! 원 startup lock과 실제 go의 budget을 재라벨하지 않으며, 공개 JSON은
//! live root cap이나 worker dispatch/completion을 구성하지 못한다.
use super::*;
use crate::rovent_root_admission::RoventRootAdmission;
use rz_runtime::contracts::ContractClock;
use std::sync::Arc;

pub(super) struct ActualExternalRoot {
    admission: Arc<RoventRootAdmission>,
    fence_sequence: u64,
}

impl RoventExternalOwner {
    /// 최초 실제 root는 spawn 때의 원 authority와 같은 경우만 직접 붙인다.
    /// 이미 exact legacy job을 제출했거나 root를 바꾼 경우, 이 물리 owner가
    /// old terminal을 회수하고 ready/reset을 끝낸 원 fence가 필수다.
    pub fn admit_actual_root(
        &mut self,
        admission: Arc<RoventRootAdmission>,
    ) -> Result<(), CheckerError> {
        let expected_lock = self
            .launch
            .input_lock_sha256
            .as_deref()
            .ok_or(invalid("external original startup lock is missing"))?;
        if digest(expected_lock)? != admission.startup_lock_digest()
            || !admission
                .original_context()
                .accepts()
                .map_err(|_| invalid("actual external root authority unavailable"))?
            || !self.worker.clock_matches(ExternalWorkerClock {
                domain: admission.runtime_clock().domain(),
                origin: admission.runtime_clock().origin(),
            })
        {
            return Err(invalid(
                "actual external root differs from original startup/clock/authority",
            ));
        }
        let startup = self
            .verified_startup()?
            .ok_or(invalid("actual external startup is still pending"))?;
        if let Some(active) = &self.active_root {
            if Arc::ptr_eq(&active.admission, &admission) {
                return Ok(());
            }
            if active.admission.authority() == admission.authority() {
                return Err(invalid(
                    "same external root cannot replace its original live admission",
                ));
            }
        }
        let previous = self
            .active_root
            .as_ref()
            .map_or(self.initial_root, |root| root.admission.authority());
        let first = self.active_root.is_none()
            && !self.legacy_dispatched
            && admission.authority() == self.initial_root;
        let fence_sequence = if first {
            0
        } else {
            // Read the result from this retained physical owner. A caller may
            // not pass a fence copied from another external process.
            let fence = self
                .worker
                .root_change_result()
                .ok_or(invalid("actual external root change fence is pending"))??;
            if fence.previous() != previous
                || fence.next() != admission.authority()
                || !fence.permits_reuse()
                || fence.process_identity() != Some(startup.process_identity())
                || fence.sequence()
                    <= self
                        .active_root
                        .as_ref()
                        .map_or(0, |root| root.fence_sequence)
                || fence.new_game_reset() != (previous.game != admission.authority().game)
            {
                return Err(invalid(
                    "actual external root fence does not join this process/previous/next root",
                ));
            }
            fence.sequence()
        };
        self.active_root = Some(ActualExternalRoot {
            admission,
            fence_sequence,
        });
        Ok(())
    }

    pub fn actual_root_admission(&self) -> Option<&Arc<RoventRootAdmission>> {
        self.active_root.as_ref().map(|root| &root.admission)
    }

    pub fn submit_actual_root(
        &mut self,
        job: ProfiledExternalJob,
    ) -> Result<ProfiledExternalDispatch, CheckerError> {
        let root = self
            .active_root
            .as_ref()
            .ok_or(invalid("actual external root was not admitted"))?;
        if !root
            .admission
            .original_context()
            .accepts()
            .map_err(|_| invalid("actual external root authority unavailable"))?
            || root.admission.requested_nodes() == Some(0)
            || Some(job.mode) != self.launch.mode
            || job.question.id.authority != root.admission.authority()
            || job.question.root_budget != root.admission.root_budget_digest()
            || job.question.original_deadline.clock != root.admission.hard_deadline().clock
            || job.question.original_deadline.at > root.admission.hard_deadline().at
        {
            return Err(invalid(
                "actual external job differs from its original go/budget/deadline",
            ));
        }
        if let Some(first) = job.question.original_first_move {
            let first = rz_position::BoardMove::try_from(
                first
                    .decode()
                    .map_err(|_| invalid("external first move encoding"))?,
            )
            .map_err(|_| invalid("external first move Rules mapping"))?;
            if !root.admission.allowed_root_moves().contains(&first) {
                return Err(invalid(
                    "actual external first move is outside original searchmoves",
                ));
            }
        }
        self.launch
            .require_query(job.constraint, job.root_moves.is_some())?;
        self.worker.submit(job)
    }

    /// Original failed/expired terminals remain inspectable for cleanup. This
    /// source check does not make them accepted teacher labels or Rules proofs.
    pub fn validate_actual_completion(
        &self,
        completion: &ProfiledExternalCompletion,
        dispatch: &ProfiledExternalDispatch,
    ) -> Result<(), CheckerError> {
        let root = self
            .active_root
            .as_ref()
            .ok_or(invalid("actual external root was not admitted"))?;
        let original = dispatch.question();
        if original.id.authority != root.admission.authority()
            || original.root_budget != root.admission.root_budget_digest()
            || original.original_deadline.clock != root.admission.hard_deadline().clock
            || original.original_deadline.at > root.admission.hard_deadline().at
        {
            return Err(invalid(
                "external completion is from a different original go/root scope",
            ));
        }
        self.validate_completion_for_budget(
            completion,
            dispatch,
            root.admission.root_budget_digest(),
        )
    }
}
