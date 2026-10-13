//! Thin actual Search owner. Physical work remains on the manager's workers.
//! Archive maintenance transfers the complete port rather than cloning Store.
use super::*;
use crate::pals_native::ticketed::TicketExecutionEvidence;
use rz_contracts::pals::{RoleOutput, SearchAuthority};
use rz_contracts::pals_adaptive::{
    CpuJobEnvelope, ExternalCpuJobEnvelope, ExternalSliceGrant, ReviewViewSeal, RoleTicketEnvelope,
    SliceGrant,
};
use rz_contracts::{ClockDomain, Digest, MonotonicTick};
use rz_position::contracts::ContractState;
use rz_runtime::CompletionSignal;
use rz_search::cpu_worker::{
    CpuAdmissionRejection, CpuJobTermination, CpuSliceReport, CpuWorkerProfile, CpuWorkerProgress,
};
use rz_search::external_cpu::worker::{
    ExternalJob, ExternalTeacherDispatch, ExternalWorkerCompletion, ExternalWorkerReport,
};
use rz_search::external_cpu::{
    CpuEndpoint, CpuEndpointSelection, CpuExecutionUse, ExternalCpuConfig, ExternalSlicePolicy,
};
use rz_search::pals::coordinator::{
    CheckedExternalCpuBinding, ClosedFrame, CoordinatorWork, FrameBudget, FrameId, FrameInput,
    ProposerRankingPublished, StepExecution, ValidatorActionAdmission,
};
use rz_search::pals::engine::adaptive_port::archive_worker::{
    ArchiveMaintenanceStamp, ArchiveOperation, ArchiveOwner, ArchiveQuarantine, ArchiveTransfer,
    ArchiveWork, ArchiveWorkReceipt, ArchiveWorker,
};
use rz_search::pals::engine::adaptive_port::validator_port::{
    AdaptiveValidatorAction, AdaptiveValidatorActionPlan, AdaptiveValidatorQuestion,
    ValidatorActionId, ValidatorActionOutcome,
};
use rz_search::pals::engine::adaptive_port::{
    AdaptiveDeferredRepair, AdaptiveEnginePort, AdaptivePortMaterializedStep, AdaptivePortQuestion,
    AdaptivePortRoleEvidence, AdaptiveRootFanoutPolicy, AdaptiveRulesPairProof,
};
use rz_search::pals::engine::{PalsCompletion, PalsError, PalsResult, RoleModel};
use rz_search::pals::review_context::ReviewView;
use rz_search::pals::store::{ForeignAppendPermit, ObservationId, StorePins};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_RETAINED_QUESTIONS: usize = 64;
const MAX_ARCHIVE_RECEIPTS: usize = 64;

/// A coherent iteration publication is scoped to its original question and
/// physical owner. It is not CPU credit settlement or a new semantic witness.
struct PublishedCpuFeedback {
    frame: FrameId,
    view: ReviewViewSeal,
    public_context: Digest,
    job: CpuJobEnvelope,
    grant: SliceGrant,
    sequence: u64,
    progress: Option<CpuWorkerProgress>,
}

impl PublishedCpuFeedback {
    fn matches_work<Q>(&self, work: &CoordinatorWork<ReviewView, Q>) -> bool {
        self.frame == work.frame
            && self.job.question == work.input.question_digest
            && self.view == work.input.view_seal
            && self.public_context == work.input.public_context
            && self.job.hard_deadline == work.original_deadline
    }
    fn publish(&mut self, progress: &CpuWorkerProgress) -> Result<(), AdaptiveManagerError> {
        if progress.job != self.job.id
            || progress.execution != self.grant.execution
            || progress.owner != self.grant.owner
            || progress.sequence <= self.sequence
        {
            return Err(AdaptiveManagerError::Invalid(
                "original CPU progress authority",
            ));
        }
        self.sequence = progress.sequence;
        self.progress = Some(progress.clone());
        Ok(())
    }
}

/// The observer registers the original immutable input before native Execute.
/// The optional V question is the Search-issued support plan, not caller masks.
pub type AdaptiveCollectionDispatchObserver = Arc<
    dyn Fn(
            &RoleTicketEnvelope,
            &FrameInput<ReviewView, AdaptivePortQuestion>,
            Option<&AdaptiveValidatorQuestion>,
        ) -> Result<(), AdaptiveManagerError>
        + Send
        + Sync,
>;

/// Actual external owner evidence. The submitted view is immutable and remains
/// identified as the CPU question; it is never relabelled as a native output.
/// Collection explicitly joins an earlier sealed P/C input by exact endpoint.
// Keep the original bounded job and completion inline at the observer seam;
// boxing would add allocation around the sealed dispatch/receipt owners.
#[allow(clippy::large_enum_variant)]
pub enum AdaptiveExternalTeacherEvent {
    Submitted {
        input: FrameInput<ReviewView, AdaptivePortQuestion>,
        original: ExternalJob,
        dispatch: ExternalTeacherDispatch,
    },
    Completed {
        dispatch: ExternalTeacherDispatch,
        completion: ExternalWorkerCompletion,
        eligibility: AdaptiveTeacherTargetEligibility,
    },
}
pub type AdaptiveExternalTeacherObserver =
    Arc<dyn Fn(AdaptiveExternalTeacherEvent) -> Result<(), AdaptiveManagerError> + Send + Sync>;

pub struct AdaptivePortDriver<M: RoleModel + 'static> {
    // Keep actual and quarantined engine owners on the heap so moving this
    // lifecycle owner does not reserve several complete engines on the stack.
    port: Option<Box<AdaptiveEnginePort<M>>>,
    rules: Arc<ContractState>,
    authority: SearchAuthority,
    hard_deadline: Instant,
    archive: Option<ArchiveWorker<M>>,
    transfer: Option<ArchiveTransfer>,
    foreign_transfer_question: Option<Digest>,
    foreign_permit: Option<(Digest, ForeignAppendPermit)>,
    archive_receipts: VecDeque<ArchiveWorkReceipt>,
    quarantine: Option<Box<ArchiveQuarantine<M>>>,
    unexpected_archive_owner: Option<Box<ArchiveOwner<M>>>,
    closing: bool,
    archive_close_requested: bool,
    archive_joined: bool,
    external_policy: ExternalSlicePolicy,
    // A raw receipt is committed once. Building its next continuation may
    // require another owner turn and must not duplicate TaskTable settlement.
    cpu_committed: HashMap<Digest, (ObservationId, bool)>,
    cpu_feedback: HashMap<Digest, PublishedCpuFeedback>,
    foreign_committed: HashMap<Digest, (ObservationId, bool)>,
    validator_questions: HashMap<Digest, AdaptiveValidatorQuestion>,
    validator_seen: HashSet<Digest>,
    action_frames: HashMap<FrameId, ValidatorActionId>,
    action_outcomes: VecDeque<ValidatorActionOutcome>,
    collection_observer: Option<AdaptiveCollectionDispatchObserver>,
    external_teacher_observer: Option<AdaptiveExternalTeacherObserver>,
    maintenance_stamp: Option<ArchiveMaintenanceStamp>,
    registered_action: Option<AdaptiveValidatorAction>,
}
impl<M: RoleModel + 'static> AdaptivePortDriver<M> {
    /// `port` must already have its exact root and model/checker metadata.
    /// Creating the archive owner starts only a bounded private command loop.
    pub fn new(
        port: AdaptiveEnginePort<M>,
        hard_deadline: Instant,
        signal: Arc<CompletionSignal>,
        external_policy: ExternalSlicePolicy,
    ) -> Result<Self, AdaptiveManagerError> {
        Self::new_with_coordinator_policy(
            port,
            hard_deadline,
            signal,
            external_policy,
            rz_runtime::adaptive::CoordinatorPolicy::Fixed,
        )
    }
    pub fn new_with_coordinator_policy(
        port: AdaptiveEnginePort<M>,
        hard_deadline: Instant,
        signal: Arc<CompletionSignal>,
        external_policy: ExternalSlicePolicy,
        policy: rz_runtime::adaptive::CoordinatorPolicy,
    ) -> Result<Self, AdaptiveManagerError> {
        Self::new_with_coordinator_policy_retaining(
            port,
            hard_deadline,
            signal,
            external_policy,
            policy,
        )
        .map_err(|(_, error)| error)
    }
    // Failure returns the original port and its session/Store owners intact;
    // an error box would add allocation to this ownership-preserving rollback.
    #[allow(clippy::result_large_err)]
    pub fn new_with_coordinator_policy_retaining(
        mut port: AdaptiveEnginePort<M>,
        hard_deadline: Instant,
        signal: Arc<CompletionSignal>,
        external_policy: ExternalSlicePolicy,
        policy: rz_runtime::adaptive::CoordinatorPolicy,
    ) -> Result<Self, (AdaptiveEnginePort<M>, AdaptiveManagerError)> {
        if Instant::now() >= hard_deadline {
            return Err((port, AdaptiveManagerError::Deadline));
        }
        if let Err(error) = port.configure_root_fanout(match policy {
            rz_runtime::adaptive::CoordinatorPolicy::Adaptive => {
                AdaptiveRootFanoutPolicy::SimultaneousAdaptive
            }
            _ => AdaptiveRootFanoutPolicy::SequentialFixed,
        }) {
            return Err((port, semantic(error)));
        }
        let rules = port.root_rules();
        let authority = port.authority();
        let archive = if port.engine().archive_enabled() {
            match ArchiveWorker::start_with_notifier(move || signal.notify()) {
                Ok(worker) => Some(worker),
                Err(error) => {
                    return Err((
                        port,
                        AdaptiveManagerError::Semantic(format!("archive owner spawn: {error}")),
                    ));
                }
            }
        } else {
            None
        };
        let archive_joined = archive.is_none();
        Ok(Self {
            port: Some(Box::new(port)),
            rules,
            authority,
            hard_deadline,
            archive,
            transfer: None,
            foreign_transfer_question: None,
            foreign_permit: None,
            archive_receipts: VecDeque::with_capacity(MAX_ARCHIVE_RECEIPTS),
            quarantine: None,
            unexpected_archive_owner: None,
            closing: false,
            archive_close_requested: false,
            archive_joined,
            external_policy,
            cpu_committed: HashMap::new(),
            cpu_feedback: HashMap::new(),
            foreign_committed: HashMap::new(),
            validator_questions: HashMap::new(),
            validator_seen: HashSet::new(),
            action_frames: HashMap::new(),
            action_outcomes: VecDeque::new(),
            collection_observer: None,
            external_teacher_observer: None,
            maintenance_stamp: None,
            registered_action: None,
        })
    }
    pub fn port(&self) -> Option<&AdaptiveEnginePort<M>> {
        self.port.as_deref()
    }
    pub fn set_collection_dispatch_observer(
        &mut self,
        observer: AdaptiveCollectionDispatchObserver,
    ) -> Result<(), AdaptiveManagerError> {
        if self.collection_observer.is_some() || self.closing {
            return Err(AdaptiveManagerError::Invalid(
                "collection observer installation",
            ));
        }
        self.collection_observer = Some(observer);
        Ok(())
    }
    pub fn root_input(
        &mut self,
    ) -> Result<FrameInput<ReviewView, AdaptivePortQuestion>, AdaptiveManagerError> {
        self.hot()?.root_input().map_err(semantic)
    }
    pub fn set_external_teacher_observer(
        &mut self,
        observer: AdaptiveExternalTeacherObserver,
    ) -> Result<(), AdaptiveManagerError> {
        if self.external_teacher_observer.is_some() || self.closing {
            return Err(AdaptiveManagerError::Invalid(
                "teacher observer installation",
            ));
        }
        self.external_teacher_observer = Some(observer);
        Ok(())
    }
    pub fn prepare_root_rules(&mut self) -> Result<(), AdaptiveManagerError> {
        self.hot()?.prepare_root_rules().map_err(semantic)
    }
    pub fn root_is_terminal(&self) -> Result<bool, AdaptiveManagerError> {
        Ok(self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .root_is_terminal())
    }
    pub fn result(&self, completion: PalsCompletion) -> Result<PalsResult, AdaptiveManagerError> {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .result(completion)
            .map_err(semantic)
    }
    pub fn archive_receipts(&self) -> impl Iterator<Item = &ArchiveWorkReceipt> {
        self.archive_receipts.iter()
    }
    pub fn take_action_outcome(&mut self) -> Option<ValidatorActionOutcome> {
        self.action_outcomes.pop_front()
    }
    /// Inspect original semantic results before a collector's validation can
    /// consume them. A copied observation grants no action-completion authority.
    pub fn action_outcomes(&self) -> impl Iterator<Item = &ValidatorActionOutcome> {
        self.action_outcomes.iter()
    }
    /// Retained semantic data never grants a physical completion on its own.
    pub fn complete_action(
        &self,
        outcome: ValidatorActionOutcome,
        receipt: &AdaptiveShutdownReceipt,
    ) -> Result<CompletedValidatorAction, AdaptiveManagerError> {
        CompletedValidatorAction::after_manager_shutdown(outcome, receipt)
    }
    // A failed closure returns this complete lifecycle owner with every
    // session/worker/quarantine intact, without an extra error-box allocation.
    #[allow(clippy::result_large_err)]
    pub fn into_port(
        mut self,
        receipt: &AdaptiveShutdownReceipt,
    ) -> Result<AdaptiveEnginePort<M>, (Self, AdaptiveManagerError)> {
        if !receipt.permits_semantic_closure()
            || receipt.authority != self.authority
            || !self.archive_joined
            || self.transfer.is_some()
            || self.quarantine.is_some()
            || self.unexpected_archive_owner.is_some()
        {
            return Err((
                self,
                AdaptiveManagerError::Invalid("Search owner requires actual shutdown fence"),
            ));
        }
        match self.port.take() {
            Some(port) => Ok(*port),
            None => Err((self, AdaptiveManagerError::PhysicalUnknown)),
        }
    }
    fn hot(&mut self) -> Result<&mut AdaptiveEnginePort<M>, AdaptiveManagerError> {
        if self.quarantine.is_some() {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        self.port
            .as_deref_mut()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)
    }
    /// Called with an owned immutable view's actual incoming pins. Existing
    /// active pins remain sealed by ArchiveWorker; no all-Store clone occurs.
    pub fn request_reclaim(
        &mut self,
        incoming_pins: StorePins,
    ) -> Result<(), AdaptiveManagerError> {
        self.transfer(ArchiveOperation::ReclaimHot { incoming_pins }, None)
    }
    fn transfer(
        &mut self,
        operation: ArchiveOperation,
        question: Option<Digest>,
    ) -> Result<(), AdaptiveManagerError> {
        if self.archive_close_requested
            || self.transfer.is_some()
            || self.foreign_permit.is_some()
            || (self.closing && !matches!(&operation, ArchiveOperation::PrepareForeign { .. }))
        {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        let port = self
            .port
            .take()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?;
        let Some(worker) = &mut self.archive else {
            self.port = Some(port);
            return Err(AdaptiveManagerError::Invalid(
                "archive operation without declared archive owner",
            ));
        };
        let work = ArchiveWork {
            deadline: match &operation {
                ArchiveOperation::PrepareForeign { issued, .. } => {
                    self.hard_deadline.min(issued.retain_until())
                }
                _ => self.hard_deadline,
            },
            max_io_bytes: port.archive_remaining_io().min(16 * 1024 * 1024),
            operation,
        };
        match worker.submit(ArchiveOwner::Port(*port), self.authority, work) {
            Ok(transfer) => {
                self.transfer = Some(transfer);
                self.foreign_transfer_question = question;
                Ok(())
            }
            Err(rejected) => {
                if let ArchiveOwner::Port(port) = rejected.owner {
                    self.port = Some(Box::new(port));
                }
                Err(semantic(rejected.error))
            }
        }
    }
    fn foreign_raw(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &ExternalWorkerReport,
    ) -> Result<(ObservationId, bool), AdaptiveManagerError> {
        let key = work.input.question_digest;
        if let Some(committed) = self.foreign_committed.get(&key) {
            return Ok(*committed);
        }
        if self.foreign_committed.len() >= MAX_RETAINED_QUESTIONS {
            return Err(AdaptiveManagerError::Capacity(
                "foreign committed questions",
            ));
        }
        let permit = match self.foreign_permit.take() {
            Some((question, permit)) if question == key => Some(permit),
            Some(other) => {
                self.foreign_permit = Some(other);
                return Err(AdaptiveManagerError::AuxiliaryPending);
            }
            None => None,
        };
        let result = match permit {
            Some(permit) => self.hot()?.commit_external_permitted(work, receipt, permit),
            None => self.hot()?.commit_external_report(work, receipt),
        };
        match result {
            Ok(committed) => {
                self.foreign_committed.insert(key, committed);
                Ok(committed)
            }
            Err(PalsError::Store(rz_search::pals::store::StoreError::ArchiveBudgetRequired)) => {
                let publication = self
                    .hot()?
                    .prepare_external_publication(work, receipt)
                    .map_err(semantic)?;
                self.transfer(
                    ArchiveOperation::PrepareForeign {
                        observation: publication.observation,
                        issued: publication.issued,
                    },
                    Some(key),
                )?;
                Err(AdaptiveManagerError::AuxiliaryPending)
            }
            Err(error) => Err(semantic(error)),
        }
    }
    fn action_before(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        execution: StepExecution,
    ) -> Result<
        Option<rz_search::pals::engine::adaptive_port::validator_port::ActionCommitGuard>,
        AdaptiveManagerError,
    > {
        let Some(action) = self.action_frames.get(&work.frame).copied() else {
            return Ok(None);
        };
        self.hot()?
            .begin_validator_action_commit(action, work, execution)
            .map(Some)
            .map_err(semantic)
    }
    fn action_after(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        guard: Option<rz_search::pals::engine::adaptive_port::validator_port::ActionCommitGuard>,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(guard) = guard {
            self.hot()?
                .finish_validator_action_commit(guard, work)
                .map_err(semantic)?;
        }
        Ok(())
    }
}
fn semantic(error: PalsError) -> AdaptiveManagerError {
    AdaptiveManagerError::Semantic(error.to_string())
}
fn checked(step: AdaptivePortMaterializedStep) -> CheckedStep<AdaptivePortQuestion> {
    CheckedStep {
        evidence: step.evidence,
        transition: step.transition,
        fresh_pair: step.fresh_pair,
        rules_pair: step.rules_pair,
        queued_recheck: step.queued_recheck,
    }
}
impl<M: RoleModel + 'static> AdaptiveSearchDriver<AdaptivePortQuestion> for AdaptivePortDriver<M> {
    fn root_rules(&self) -> Arc<ContractState> {
        Arc::clone(&self.rules)
    }
    fn collection_before_dispatch(
        &mut self,
        ticket: &RoleTicketEnvelope,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(observer) = &self.collection_observer {
            let validator = self.validator_questions.get(&input.question_digest);
            if ticket.role == rz_contracts::pals::Role::Validator && validator.is_none() {
                return Err(AdaptiveManagerError::Invalid(
                    "V capture lacks actual prepared support question",
                ));
            }
            observer(ticket, input, validator)?;
        }
        Ok(())
    }
    fn poll_auxiliary(&mut self) -> Result<bool, AdaptiveManagerError> {
        let mut changed = false;
        if let (Some(worker), Some(transfer)) = (&mut self.archive, &self.transfer) {
            if let Some(completed) = worker.try_receive(transfer).map_err(|error| {
                AdaptiveManagerError::Semantic(format!("archive receive: {error:?}"))
            })? {
                let nonce = transfer.nonce();
                let foreign = self.foreign_transfer_question.take();
                self.transfer = None;
                let installed = if let Some(question) = foreign {
                    completed.install_with_foreign(self.authority, nonce).map(
                        |(owner, receipt, permit)| {
                            self.foreign_permit = Some((question, permit));
                            (owner, receipt)
                        },
                    )
                } else {
                    completed.install(self.authority, nonce)
                };
                match installed {
                    Ok((ArchiveOwner::Port(port), receipt)) => {
                        self.port = Some(Box::new(port));
                        if self.archive_receipts.len() == MAX_ARCHIVE_RECEIPTS {
                            self.archive_receipts.pop_front();
                        }
                        self.archive_receipts.push_back(receipt);
                    }
                    Ok((owner @ ArchiveOwner::Engine(_), _)) => {
                        self.unexpected_archive_owner = Some(Box::new(owner));
                        return Err(AdaptiveManagerError::Invalid(
                            "archive returned a different semantic owner kind",
                        ));
                    }
                    Err(quarantine) => {
                        self.quarantine = Some(Box::new(quarantine));
                        return Err(AdaptiveManagerError::PhysicalUnknown);
                    }
                }
                changed = true;
            }
        }
        if self.archive_close_requested && self.transfer.is_none() && !self.archive_joined {
            if let Some(worker) = &mut self.archive {
                worker.request_shutdown();
                if let Some(receipt) = worker.try_join().map_err(|error| {
                    AdaptiveManagerError::Semantic(format!("archive join: {error:?}"))
                })? {
                    if !receipt.admission_closed
                        || !receipt.worker_joined
                        || !receipt.owners_returned
                    {
                        return Err(AdaptiveManagerError::PhysicalUnknown);
                    }
                    self.archive_joined = true;
                    changed = true;
                }
            }
        }
        Ok(changed)
    }
    fn semantic_owner_available(&self) -> bool {
        self.port.is_some() && self.quarantine.is_none() && self.unexpected_archive_owner.is_none()
    }
    fn prepare_semantic_commit(&mut self) -> Result<bool, AdaptiveManagerError> {
        if !self.semantic_owner_available() {
            return Ok(false);
        }
        if self.closing || self.archive.is_none() {
            return Ok(true);
        }
        let proposal = self
            .hot()?
            .archive_maintenance_proposal()
            .map_err(semantic)?;
        let Some(proposal) = proposal else {
            return Ok(true);
        };
        if self.maintenance_stamp == Some(proposal.stamp) {
            // A fully pinned/no-change owner does not retry the same I/O each
            // manager tick. A real Store commit/revision changes the stamp.
            return Ok(true);
        }
        let stamp = proposal.stamp;
        self.transfer(
            ArchiveOperation::ReclaimHot {
                incoming_pins: proposal.incoming_pins,
            },
            None,
        )?;
        self.maintenance_stamp = Some(stamp);
        Ok(false)
    }
    fn begin_auxiliary_shutdown(&mut self, _: Instant) -> Result<(), AdaptiveManagerError> {
        self.closing = true;
        Ok(())
    }
    fn finish_auxiliary_shutdown(&mut self) -> Result<(), AdaptiveManagerError> {
        if self.transfer.is_some() || self.foreign_permit.is_some() {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        self.archive_close_requested = true;
        if let Some(worker) = &mut self.archive {
            worker.request_shutdown();
        }
        Ok(())
    }
    fn auxiliary_shutdown_confirmed(&self) -> bool {
        self.archive_joined
            && self.transfer.is_none()
            && self.port.is_some()
            && self.quarantine.is_none()
            && self.unexpected_archive_owner.is_none()
    }
    fn short_circuit(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<CheckedRulesStep<AdaptivePortQuestion>>, AdaptiveManagerError> {
        let Some(step) = self.hot()?.short_circuit(work).map_err(semantic)? else {
            return Ok(None);
        };
        let materialized = self
            .hot()?
            .materialize_step(step.checked)
            .map_err(semantic)?;
        Ok(Some(CheckedRulesStep {
            endpoint: step.endpoint,
            checked: checked(materialized),
        }))
    }
    fn ranking_published(
        &mut self,
        ranking: &ProposerRankingPublished,
    ) -> Result<Vec<CandidateFrame<AdaptivePortQuestion>>, AdaptiveManagerError> {
        self.hot()?
            .ranking_published(ranking)
            .map(|candidates| {
                candidates
                    .into_iter()
                    .map(|candidate| CandidateFrame {
                        first_move: candidate.first_move,
                        input: candidate.input,
                        budget: candidate.budget,
                        deferred: candidate.deferred,
                        initialized: candidate.initialized,
                    })
                    .collect()
            })
            .map_err(semantic)
    }
    fn root_candidate_admitted(
        &mut self,
        frame: FrameId,
        first: rz_contracts::pals::Move16,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .root_candidate_admitted(frame, first, input)
            .map_err(semantic)
    }
    fn next_root_round_input(
        &mut self,
    ) -> Result<Option<FrameInput<ReviewView, AdaptivePortQuestion>>, AdaptiveManagerError> {
        self.hot()?.next_root_round_input().map_err(semantic)
    }
    fn next_fixed_root_candidate(
        &mut self,
    ) -> Result<Option<CandidateFrame<AdaptivePortQuestion>>, AdaptiveManagerError> {
        self.hot()?
            .next_fixed_root_candidate()
            .map(|candidate| {
                candidate.map(|candidate| CandidateFrame {
                    first_move: candidate.first_move,
                    input: candidate.input,
                    budget: candidate.budget,
                    deferred: candidate.deferred,
                    initialized: candidate.initialized,
                })
            })
            .map_err(semantic)
    }
    fn next_fixed_iterative_repair(
        &mut self,
    ) -> Result<Option<AdaptiveDeferredRepair<AdaptivePortQuestion>>, AdaptiveManagerError> {
        self.hot()?.next_fixed_iterative_repair().map_err(semantic)
    }
    fn validate_role(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .validate_role(work, ticket, output)
            .map_err(semantic)
    }
    fn commit_role(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
    ) -> Result<CheckedStep<AdaptivePortQuestion>, AdaptiveManagerError> {
        let guard = self.action_before(work, StepExecution::Role(ticket.id))?;
        let physical = physical.map(|evidence| AdaptivePortRoleEvidence {
            execution: evidence.execution,
            completed_nn_inputs: evidence.completed_nn_inputs,
            measurement_complete: evidence.measurement_complete,
            physical_fresh: evidence.physical_fresh,
        });
        let step = self
            .hot()?
            .commit_role(work, ticket, output, physical)
            .map_err(semantic)?;
        // Selection of the next immutable view follows consumption of the
        // exact view submitted for this accepted native ticket.
        self.hot()?
            .role_consumed(ticket, &work.input.view)
            .map_err(semantic)?;
        self.action_after(work, guard)?;
        let materialized = self.hot()?.materialize_step(step).map_err(semantic)?;
        Ok(checked(materialized))
    }
    fn role_consumed(
        &mut self,
        ticket: &RoleTicketEnvelope,
        view: &Arc<ReviewView>,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?.role_consumed(ticket, view).map_err(semantic)
    }
    fn cpu_question(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        profiles: &[CpuWorkerProfile],
        external: Option<&ExternalCpuConfig>,
        selection: CpuEndpointSelection,
    ) -> Result<CpuQuestion, AdaptiveManagerError> {
        let endpoint = if selection.execution_use == CpuExecutionUse::Collection {
            selection.cpu_t
        } else {
            selection.cpu_r
        };
        match endpoint {
            CpuEndpoint::Own => {
                let question = self.hot()?.cpu_question(work, profiles).map_err(semantic)?;
                Ok(CpuQuestion::Own(OwnCpuQuestion {
                    position: question.position,
                    root_moves: question.root_moves,
                    target_depth: question.target_depth,
                    total_native_node_credit: question.total_native_node_credit,
                }))
            }
            CpuEndpoint::ExternalUci => {
                let config = external.ok_or(AdaptiveManagerError::Invalid(
                    "foreign endpoint lacks actual config",
                ))?;
                let policy = self.external_policy;
                let question = self
                    .hot()?
                    .external_question(work, config, policy)
                    .map_err(semantic)?;
                Ok(CpuQuestion::External(ForeignCpuQuestion {
                    position: question.position,
                    root_moves: question.root_moves,
                    target_depth: question.target_depth,
                    policy: question.policy,
                }))
            }
            CpuEndpoint::Disabled => {
                Err(AdaptiveManagerError::Invalid("no checker owner selected"))
            }
        }
    }
    fn cpu_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        grant: &SliceGrant,
    ) -> Result<(), AdaptiveManagerError> {
        let key = work.input.question_digest;
        if let Some(original) = self.cpu_feedback.get(&key) {
            if !original.matches_work(work)
                || original.job != *job
                || original.grant.owner != grant.owner
            {
                return Err(AdaptiveManagerError::Invalid("CPU progress resume context"));
            }
        } else if self.cpu_feedback.len() >= MAX_RETAINED_QUESTIONS {
            return Err(AdaptiveManagerError::Capacity("CPU progress questions"));
        }
        self.hot()?
            .cpu_admitted(work, job, grant)
            .map_err(semantic)?;
        match self.cpu_feedback.get_mut(&key) {
            Some(original) => {
                original.grant = *grant;
                original.progress = None;
            }
            None => {
                self.cpu_feedback.insert(
                    key,
                    PublishedCpuFeedback {
                        frame: work.frame,
                        view: work.input.view_seal,
                        public_context: work.input.public_context,
                        job: *job,
                        grant: *grant,
                        sequence: 0,
                        progress: None,
                    },
                );
            }
        }
        Ok(())
    }
    fn cpu_progress_for(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        progress: &CpuWorkerProgress,
    ) -> Result<(), AdaptiveManagerError> {
        let original = self
            .cpu_feedback
            .get_mut(&work.input.question_digest)
            .ok_or(AdaptiveManagerError::Invalid(
                "CPU progress lacks original admission",
            ))?;
        if self.closing || work.frame.authority != self.authority || !original.matches_work(work) {
            return Err(AdaptiveManagerError::Invalid(
                "CPU progress question context",
            ));
        }
        original.publish(progress)
    }
    fn predict_yield_feedback(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        costs: &rz_runtime::adaptive::CostEstimator,
        class: rz_runtime::adaptive::CostClass,
        since_publication: Duration,
        lease_remaining: Duration,
    ) -> Option<rz_runtime::adaptive::FeedbackWaitPrediction> {
        if self.closing || work.frame.authority != self.authority {
            return None;
        }
        let original = self.cpu_feedback.get(&work.input.question_digest)?;
        if !original.matches_work(work) {
            return None;
        }
        let progress = original.progress.as_ref()?;
        if !self
            .port
            .as_ref()?
            .published_cpu_continuation_eligible(work, progress)
        {
            return None;
        }
        // This only forecasts waiting until eligible feedback can be consumed.
        // The model uses original completed owner timing, never CPU agreement,
        // depth as utility, a fabricated witness, or paused residence time.
        costs.feedback_wait_prediction(class, since_publication, lease_remaining)
    }
    fn cpu_progress_request_eligible(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        grant: &SliceGrant,
    ) -> bool {
        !self.closing
            && work.frame.authority == self.authority
            && self
                .port
                .as_ref()
                .is_some_and(|port| port.cpu_progress_request_eligible(work, job, grant))
    }
    fn cpu_partial_continuation_eligible(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> bool {
        !self.closing
            && work.frame.authority == self.authority
            && self
                .port
                .as_ref()
                .is_some_and(|port| port.yielded_cpu_continuation_eligible(work, report))
    }
    fn external_feedback_eligible(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        progress: &rz_search::external_cpu::worker::ExternalWorkerProgress,
    ) -> bool {
        !self.closing
            && work.frame.authority == self.authority
            && self
                .port
                .as_ref()
                .is_some_and(|port| port.published_external_continuation_eligible(work, progress))
    }
    fn cpu_detached_slice_observed(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        // The original CPU question remains the consumer. No after_cpu or
        // next-frame transition is issued for later receipts from this owner.
        self.cpu_slice_observed(work, report)?;
        if report.terminal {
            self.cpu_feedback.remove(&work.input.question_digest);
        }
        Ok(())
    }
    fn cpu_slice_observed(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(original) = self.cpu_feedback.get_mut(&work.input.question_digest) {
            original.progress = None;
        }
        if self.closing {
            self.hot()?
                .observe_cpu_report(work, report)
                .map_err(semantic)
        } else {
            self.hot()?
                .commit_cpu_report(work, report)
                .map(|_| ())
                .map_err(semantic)
        }
    }
    fn cpu_rejected(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        proof: &CpuAdmissionRejection,
    ) -> Result<(), AdaptiveManagerError> {
        if !proof.is_verified_unqueued() {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        self.hot()?
            .reject_cpu_admission(work, job, proof)
            .map_err(semantic)?;
        self.cpu_feedback.remove(&work.input.question_digest);
        Ok(())
    }
    fn role_consumption_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        view: &ReviewView,
    ) -> Result<bool, AdaptiveManagerError> {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .role_consumption_recorded(ticket, view)
            .map_err(semantic)
    }
    fn cpu_token_discarded(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        termination: &CpuJobTermination,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .retire_cpu_token(work, termination)
            .map_err(semantic)?;
        self.cpu_feedback.remove(&work.input.question_digest);
        Ok(())
    }
    fn commit_cpu(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> Result<CheckedStep<AdaptivePortQuestion>, AdaptiveManagerError> {
        let key = work.input.question_digest;
        if let Some(original) = self.cpu_feedback.get_mut(&key) {
            original.progress = None;
        }
        let guard = self.action_before(
            work,
            StepExecution::Cpu {
                job: report.grant.job,
                execution: report.grant.execution,
            },
        )?;
        let actual = if let Some(committed) = self.cpu_committed.get(&key) {
            *committed
        } else {
            if self.cpu_committed.len() >= MAX_RETAINED_QUESTIONS {
                return Err(AdaptiveManagerError::Capacity("CPU committed questions"));
            }
            let committed = self
                .hot()?
                .commit_cpu_report(work, report)
                .map_err(semantic)?;
            self.cpu_committed.insert(key, committed);
            committed
        };
        let step = self
            .hot()?
            .after_cpu(work, actual.0, actual.1)
            .map_err(semantic)?;
        self.action_after(work, guard)?;
        self.cpu_committed.remove(&key);
        let materialized = self.hot()?.materialize_step(step).map_err(semantic)?;
        self.cpu_feedback.remove(&key);
        Ok(checked(materialized))
    }
    fn external_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &ExternalCpuJobEnvelope,
        grant: &ExternalSliceGrant,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<CheckedExternalCpuBinding, AdaptiveManagerError> {
        self.hot()?
            .external_admitted(work, job, grant, clock, now)
            .map_err(semantic)
    }
    fn external_rejected(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &ExternalCpuJobEnvelope,
        grant: &ExternalSliceGrant,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .external_rejected(work, job, grant)
            .map_err(semantic)
    }
    fn external_teacher_submitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        original: &ExternalJob,
        dispatch: &ExternalTeacherDispatch,
    ) -> Result<(), AdaptiveManagerError> {
        if !dispatch.matches_job(original)
            || original.envelope.id.authority != self.authority
            || work.input.view_seal.view_digest.0 != work.input.view.seal_digest()
            || work.input.view_seal.view_sequence != work.input.view.sequence()
        {
            return Err(AdaptiveManagerError::Invalid(
                "actual teacher submission scope",
            ));
        }
        if let Some(observer) = &self.external_teacher_observer {
            observer(AdaptiveExternalTeacherEvent::Submitted {
                input: work.input.clone(),
                original: original.clone(),
                dispatch: dispatch.clone(),
            })?;
        }
        Ok(())
    }
    fn external_teacher_completed(
        &mut self,
        dispatch: &ExternalTeacherDispatch,
        completion: &ExternalWorkerCompletion,
        eligibility: AdaptiveTeacherTargetEligibility,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(observer) = &self.external_teacher_observer {
            observer(AdaptiveExternalTeacherEvent::Completed {
                dispatch: dispatch.clone(),
                completion: completion.clone(),
                eligibility,
            })?;
        }
        Ok(())
    }
    fn external_report_observed(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &ExternalWorkerReport,
    ) -> Result<(), AdaptiveManagerError> {
        let key = work.input.question_digest;
        let permit = match self.foreign_permit.take() {
            Some((question, permit)) if question == key => Some(permit),
            Some(other) => {
                self.foreign_permit = Some(other);
                return Err(AdaptiveManagerError::AuxiliaryPending);
            }
            None => None,
        };
        let observed = match permit {
            Some(permit) => self.hot()?.observe_external_permitted(work, report, permit),
            None => self.hot()?.observe_external_report(work, report),
        };
        match observed {
            Ok(()) => Ok(()),
            Err(PalsError::Store(rz_search::pals::store::StoreError::ArchiveBudgetRequired)) => {
                let publication = self
                    .hot()?
                    .prepare_external_publication(work, report)
                    .map_err(semantic)?;
                self.transfer(
                    ArchiveOperation::PrepareForeign {
                        observation: publication.observation,
                        issued: publication.issued,
                    },
                    Some(key),
                )?;
                Err(AdaptiveManagerError::AuxiliaryPending)
            }
            Err(error) => Err(semantic(error)),
        }
    }
    fn commit_external(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &ExternalWorkerReport,
    ) -> Result<CheckedStep<AdaptivePortQuestion>, AdaptiveManagerError> {
        let guard = self.action_before(
            work,
            StepExecution::ExternalCpu {
                job: report.job.id,
                execution: report.grant.execution,
            },
        )?;
        let actual = self.foreign_raw(work, report)?;
        let step = self
            .hot()?
            .after_cpu(work, actual.0, actual.1)
            .map_err(semantic)?;
        self.action_after(work, guard)?;
        self.foreign_committed.remove(&work.input.question_digest);
        let materialized = self.hot()?.materialize_step(step).map_err(semantic)?;
        Ok(checked(materialized))
    }
    fn publish_closed(&mut self, closed: &ClosedFrame) -> Result<(), AdaptiveManagerError> {
        self.hot()?.publish_closed(closed).map_err(semantic)?;
        if let Some(action) = self.action_frames.remove(&closed.id) {
            if closed.unknown.is_none() {
                let outcome = self
                    .hot()?
                    .finish_validator_action_semantics(action, Some(closed))
                    .map_err(semantic)?;
                if self.action_outcomes.len() >= MAX_RETAINED_QUESTIONS {
                    return Err(AdaptiveManagerError::Capacity("V action outcomes"));
                }
                self.action_outcomes.push_back(outcome);
            }
        }
        Ok(())
    }
    fn validate_rules_pair(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        proof: &AdaptiveRulesPairProof,
    ) -> Result<(), AdaptiveManagerError> {
        proof.validate(work).map_err(semantic)
    }
    fn validator_input(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        profiles: &[CpuWorkerProfile],
        selection: CpuEndpointSelection,
        budget: FrameBudget,
    ) -> Result<Option<FrameInput<ReviewView, AdaptivePortQuestion>>, AdaptiveManagerError> {
        if !selection.load_v || selection.execution_use != CpuExecutionUse::Collection {
            return Ok(None);
        }
        let review_key = self.hot()?.validator_review_key(work).map_err(semantic)?;
        if self.validator_seen.contains(&review_key) {
            return Ok(None);
        }
        if self.validator_seen.len() >= MAX_RETAINED_QUESTIONS {
            return Err(AdaptiveManagerError::Capacity("V root contexts"));
        }
        let question = self
            .hot()?
            .prepare_validator_question(work, budget, profiles, selection)
            .map_err(semantic)?;
        let input = question.input().clone();
        self.validator_seen.insert(review_key);
        self.validator_questions
            .insert(input.question_digest, question);
        Ok(Some(input))
    }
    fn validate_validator(
        &self,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        let question = self.validator_questions.get(&input.question_digest).ok_or(
            AdaptiveManagerError::Invalid("V result has no original support question"),
        )?;
        if question.input().view_seal != input.view_seal
            || ticket.view != input.view_seal
            || ticket.question != input.question_digest
            || ticket.authority != self.authority
            || output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != ticket.representation
        {
            return Err(AdaptiveManagerError::Invalid(
                "V original support question or actual output",
            ));
        }
        if !matches!(
            output.payload,
            rz_contracts::pals::RolePayload::TaskRanking { .. }
        ) {
            return Err(AdaptiveManagerError::Invalid("V actual task head"));
        }
        Ok(())
    }
    fn validator_accepted(
        &mut self,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        proof: AcceptedValidator,
    ) -> Result<AdaptiveValidatorActionPlan, AdaptiveManagerError> {
        let question = self
            .validator_questions
            .remove(&input.question_digest)
            .ok_or(AdaptiveManagerError::Invalid(
                "accepted V lacks original support question",
            ))?;
        match self.registered_action {
            Some(action) => proof.plan_registered_action(self.hot()?, &question, action),
            None => proof.select(self.hot()?, &question),
        }
    }
    fn register_supported_validator_action(
        &mut self,
        action: AdaptiveValidatorAction,
    ) -> Result<(), AdaptiveManagerError> {
        if self.closing
            || self.registered_action.is_some()
            || !self.validator_seen.is_empty()
            || !self.action_frames.is_empty()
        {
            return Err(AdaptiveManagerError::Invalid(
                "registered action must precede V submission",
            ));
        }
        self.registered_action = Some(action);
        Ok(())
    }
    fn validator_action_admitted(
        &mut self,
        id: ValidatorActionId,
        admission: &ValidatorActionAdmission,
    ) -> Result<(), AdaptiveManagerError> {
        match admission {
            ValidatorActionAdmission::Frame(frame) | ValidatorActionAdmission::Resumed(frame) => {
                if self.action_frames.contains_key(frame) {
                    return Err(AdaptiveManagerError::Invalid(
                        "two action owners for one frame",
                    ));
                }
                self.hot()?
                    .bind_validator_action(id, *frame)
                    .map_err(semantic)?;
                self.action_frames.insert(*frame, id);
            }
            ValidatorActionAdmission::Defer => {
                let outcome = self
                    .hot()?
                    .finish_validator_action_semantics(id, None)
                    .map_err(semantic)?;
                if self.action_outcomes.len() >= MAX_RETAINED_QUESTIONS {
                    return Err(AdaptiveManagerError::Capacity("V action outcomes"));
                }
                self.action_outcomes.push_back(outcome);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod published_progress_tests {
    use super::*;
    use rz_contracts::pals_adaptive::{CpuJobId, CpuWorkerId};
    use rz_contracts::{
        Deadline, ExecutionId, GameGeneration, OwnerId, ProcessEpoch, RootGeneration,
        StateIdentity, StateRevision,
    };
    use rz_search::cpu::CpuIterationProgress;

    fn original() -> PublishedCpuFeedback {
        let authority = SearchAuthority {
            epoch: ProcessEpoch(3),
            game: GameGeneration(2),
            root: RootGeneration(4),
            implementation: Digest([1; 32]),
        };
        let state_identity = StateIdentity {
            owner: OwnerId(1),
            revision: StateRevision(5),
            semantic: Digest([2; 32]),
        };
        let job = CpuJobEnvelope {
            id: CpuJobId {
                authority,
                sequence: 1,
            },
            state_identity,
            question: Digest([3; 32]),
            root_restriction: Digest([4; 32]),
            evaluation_namespace: Digest([5; 32]),
            search_conditions: Digest([6; 32]),
            total_native_node_credit: 8192,
            hard_deadline: Deadline {
                clock: ClockDomain(authority.epoch),
                at: MonotonicTick(20_000_000),
            },
        };
        PublishedCpuFeedback {
            frame: FrameId {
                authority,
                sequence: 1,
            },
            view: ReviewViewSeal {
                authority,
                view_sequence: 1,
                context_policy: Digest([7; 32]),
                view_digest: Digest([8; 32]),
                root_state: state_identity,
                actual_tokens: 0,
                selected_records: 0,
                anchors: 1,
                length_bucket: 8,
            },
            public_context: Digest([9; 32]),
            job,
            grant: SliceGrant {
                job: job.id,
                execution: ExecutionId::new(authority.epoch, 1),
                owner: CpuWorkerId(0),
                node_credit: 4096,
                soft_lease: Duration::from_millis(8),
                control_sequence: 1,
            },
            sequence: 0,
            progress: None,
        }
    }
    #[test]
    fn published_progress_rejects_other_question_owner_and_old_slice() {
        let mut original = original();
        let progress = CpuWorkerProgress {
            job: original.job.id,
            execution: original.grant.execution,
            owner: original.grant.owner,
            sequence: 1,
            iteration: CpuIterationProgress {
                best_move: None,
                score: 100,
                completed_depth: 1,
                nodes: 4,
            },
        };
        for wrong in [
            CpuWorkerProgress {
                job: CpuJobId {
                    sequence: 2,
                    ..progress.job
                },
                ..progress.clone()
            },
            CpuWorkerProgress {
                owner: CpuWorkerId(1),
                ..progress.clone()
            },
            CpuWorkerProgress {
                execution: ExecutionId::new(progress.execution.epoch, 2),
                ..progress.clone()
            },
        ] {
            assert!(original.publish(&wrong).is_err());
            assert!(original.progress.is_none());
            assert_eq!(original.sequence, 0);
        }
        original.publish(&progress).unwrap();
        assert!(original.publish(&progress).is_err());
        assert_eq!(original.progress.as_ref().unwrap().iteration.nodes, 4);
        original.grant.execution.sequence = 2;
        original.progress = None;
        assert!(
            original
                .publish(&CpuWorkerProgress {
                    sequence: 2,
                    ..progress.clone()
                })
                .is_err()
        );
        original
            .publish(&CpuWorkerProgress {
                execution: original.grant.execution,
                sequence: 2,
                ..progress
            })
            .unwrap();
        // Publication did not change the immutable Job or settle any credit.
        assert_eq!(original.job.total_native_node_credit, 8192);
    }
}
