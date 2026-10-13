//! Actual collection-only Validator acceptance and action completion boundary.
//!
//! The task head is captured/accepted by the existing ticket registry. This
//! facade cannot accept caller replacement logits or a public `passed` flag.
//! Its action completion proof adds the manager's actual shutdown fence to
//! the Search owner's Store-backed semantic outcome before arena collection.
use super::{
    AdaptiveManagerCounters, AdaptiveManagerError, AdaptiveShutdownReceipt,
    ExclusiveActionClosureProof,
};
use rz_contracts::Digest;
use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
use rz_search::pals::coordinator::AcceptedValidatorPublished;
use rz_search::pals::engine::RoleModel;
use rz_search::pals::engine::adaptive_port::AdaptiveEnginePort;
use rz_search::pals::engine::adaptive_port::validator_port::{
    AdaptiveValidatorAction, AdaptiveValidatorActionPlan, AdaptiveValidatorQuestion,
    ValidatorActionId, ValidatorActionOutcome,
};
use rz_search::pals::review_context::ReviewView;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// An actual manager-controlled interval with exactly one admitted V action.
/// Ordinary whole-root completion cannot construct this evidence. Its cost
/// includes the V decision and required CPU/GPU/foreign drain; root P/setup
/// before arming are a separate observation and never silently included.
pub struct ExclusiveValidatorActionCost {
    source: RoleTicketEnvelope,
    action: ValidatorActionId,
    frozen_context: Digest,
    parent_question: Digest,
    parent_view: ReviewViewSeal,
    parent_input: Digest,
    parent_public_context: Digest,
    action_support_mask: [bool; 7],
    started: Instant,
    finished: Instant,
    baseline: AdaptiveManagerCounters,
    final_counters: AdaptiveManagerCounters,
}
impl ExclusiveValidatorActionCost {
    pub fn source(&self) -> &RoleTicketEnvelope {
        &self.source
    }
    pub fn action(&self) -> ValidatorActionId {
        self.action
    }
    pub fn frozen_context_digest(&self) -> Digest {
        self.frozen_context
    }
    pub fn parent_question(&self) -> Digest {
        self.parent_question
    }
    pub fn parent_view(&self) -> ReviewViewSeal {
        self.parent_view
    }
    pub fn parent_input(&self) -> Digest {
        self.parent_input
    }
    pub fn parent_public_context(&self) -> Digest {
        self.parent_public_context
    }
    pub fn action_support_mask(&self) -> [bool; 7] {
        self.action_support_mask
    }
    pub fn started(&self) -> Instant {
        self.started
    }
    pub fn finished(&self) -> Instant {
        self.finished
    }
    pub fn wall_cost(&self) -> Duration {
        self.finished.saturating_duration_since(self.started)
    }
    pub fn baseline_counters(&self) -> AdaptiveManagerCounters {
        self.baseline
    }
    pub fn final_counters(&self) -> AdaptiveManagerCounters {
        self.final_counters
    }
    pub fn cost_scope(&self) -> &'static str {
        "exclusive_v_decision_and_single_action_with_required_drain"
    }
}

/// Only the manager can wrap the registry's already opaque accepted proof.
/// Constructing an ordinary `RoleOutput` cannot construct this value.
pub struct AcceptedValidator {
    published: AcceptedValidatorPublished,
}
impl AcceptedValidator {
    pub(super) fn from_published(published: AcceptedValidatorPublished) -> Self {
        Self { published }
    }
    pub fn published(&self) -> &AcceptedValidatorPublished {
        &self.published
    }
    pub fn source(&self) -> &RoleTicketEnvelope {
        self.published.ticket()
    }
    pub fn view(&self) -> &Arc<ReviewView> {
        self.published.view()
    }
    pub fn task_logits(&self) -> &[f32; 7] {
        self.published.task_logits()
    }
    pub fn select<M: RoleModel>(
        &self,
        port: &mut AdaptiveEnginePort<M>,
        question: &AdaptiveValidatorQuestion,
    ) -> Result<AdaptiveValidatorActionPlan, AdaptiveManagerError> {
        port.select_validator_action(question, &self.published)
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))
    }
    /// The collection runner registers the supported-action sequence before
    /// isolated roots execute. It can select a supported action, never replace
    /// this accepted head, infer a witness, or grant a deployment V call.
    pub fn plan_registered_action<M: RoleModel>(
        &self,
        port: &mut AdaptiveEnginePort<M>,
        question: &AdaptiveValidatorQuestion,
        action: AdaptiveValidatorAction,
    ) -> Result<AdaptiveValidatorActionPlan, AdaptiveManagerError> {
        port.plan_validator_action(question, &self.published, action)
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))
    }
}

/// Arena may serialize a completed action only from this opaque value. A
/// Search semantic frame close, accepted V head or positive utility number
/// does not by itself prove physical completion of any owner.
pub struct CompletedValidatorAction {
    semantic: ValidatorActionOutcome,
    native_cpu_spent: u64,
    completed_nn_inputs: u64,
    full_action_wall_cost: Duration,
    exclusive_cost: Option<ExclusiveValidatorActionCost>,
}
impl CompletedValidatorAction {
    /// Whole-root fence is deliberately conservative. The actual manager
    /// owns the original receipt and issues this after all pending root work,
    /// CPU reservations, GPU fences and foreign drain are known. A persistent
    /// native owner needs its actual quiescence receipt; a process exit instead
    /// needs the actual owner join. Declaration alone satisfies neither.
    /// Per-action early fences need a separately attested owner receipt.
    pub(super) fn after_manager_shutdown(
        semantic: ValidatorActionOutcome,
        receipt: &AdaptiveShutdownReceipt,
    ) -> Result<Self, AdaptiveManagerError> {
        if !receipt.permits_semantic_closure()
            || receipt.authority != semantic.id().authority
            || semantic.source().authority != receipt.authority
            || semantic.source().id.epoch != receipt.authority.epoch
        {
            return Err(AdaptiveManagerError::Invalid(
                "action requires actual manager physical fence",
            ));
        }
        let full_action_wall_cost = semantic.physical_close_wall_cost();
        Ok(Self {
            semantic,
            full_action_wall_cost,
            native_cpu_spent: receipt.native_cpu_spent,
            completed_nn_inputs: receipt.counters.completed_nn_inputs,
            exclusive_cost: None,
        })
    }
    /// Only the actual manager may attach its gated one-action root fence.
    /// The proof keeps the original V ticket and rejects attaching another
    /// action's interval even when both executions used the same root.
    pub(super) fn after_exclusive_manager_shutdown(
        semantic: ValidatorActionOutcome,
        receipt: &AdaptiveShutdownReceipt,
        proof: &ExclusiveActionClosureProof,
    ) -> Result<Self, AdaptiveManagerError> {
        if !receipt.permits_root_reuse()
            || proof.authority() != receipt.authority
            || proof.action() != semantic.id()
            || proof.frozen_context_digest() != semantic.frozen_context_digest()
            || proof.source() != semantic.source()
            || proof.parent_question() != semantic.parent_question()
            || proof.parent_view() != semantic.parent_view()
            || proof.parent_input() != semantic.parent_input()
            || proof.parent_public_context() != semantic.parent_public_context()
            || proof.action_support_mask() != semantic.action_support_mask()
            || proof.finished() < proof.started()
            || proof.final_counters() != receipt.counters
        {
            return Err(AdaptiveManagerError::Invalid(
                "exclusive action requires original gated source and actual root fence",
            ));
        }
        let exclusive = ExclusiveValidatorActionCost {
            source: *proof.source(),
            action: proof.action(),
            frozen_context: proof.frozen_context_digest(),
            parent_question: proof.parent_question(),
            parent_view: proof.parent_view(),
            parent_input: proof.parent_input(),
            parent_public_context: proof.parent_public_context(),
            action_support_mask: proof.action_support_mask(),
            started: proof.started(),
            finished: proof.finished(),
            baseline: proof.baseline_counters(),
            final_counters: proof.final_counters(),
        };
        let mut completed = Self::after_manager_shutdown(semantic, receipt)?;
        completed.exclusive_cost = Some(exclusive);
        Ok(completed)
    }
    pub fn semantic(&self) -> &ValidatorActionOutcome {
        &self.semantic
    }
    pub fn actual_native_input(&self) -> Digest {
        self.semantic.source().representation.input
    }
    pub fn full_action_wall_cost(&self) -> Duration {
        self.full_action_wall_cost
    }
    /// Root fence may include wait/drain for other concurrent questions. It
    /// is total completion latency, not isolated action service time. Explicit
    /// action comparisons require controlled frozen-root executions.
    pub fn full_action_wall_cost_scope(&self) -> &'static str {
        "input_prepare_to_actual_root_physical_fence"
    }
    /// Only the exclusive manager issuer can provide this value. The raw
    /// whole-root interval remains useful evidence but is not a known cost
    /// for pairwise V targets because concurrent actions can overlap it.
    pub fn exclusive_action_cost(&self) -> Option<&ExclusiveValidatorActionCost> {
        self.exclusive_cost.as_ref()
    }
    pub fn training_wall_cost(&self) -> Option<Duration> {
        self.exclusive_action_cost()
            .map(ExclusiveValidatorActionCost::wall_cost)
    }
    /// Whole root counters are auxiliary evidence, not a per-action delta.
    pub fn root_native_cpu_spent(&self) -> u64 {
        self.native_cpu_spent
    }
    pub fn root_completed_nn_inputs(&self) -> u64 {
        self.completed_nn_inputs
    }
}

#[cfg(test)]
#[path = "validator_tests.rs"]
mod tests;
