//! Thin NoResearcher owner. The semantic port selects exact questions; the
//! role-only manager and native worker own submission, inference and fences.
use super::*;
use crate::pals_adaptive::{RoleOnlySearchDriver, RoleOnlyShutdownReceipt};
use rz_search::pals::coordinator::AcceptedRoleStep;
use rz_search::pals::engine::NoResearcher;
use rz_search::pals::engine::pc_port::{PcPortQuestion, PcPortResult, PcSemanticPort};

pub struct PcPortDriver<M: RoleModel> {
    port: PcSemanticPort<M>,
}
impl<M: RoleModel> PcPortDriver<M> {
    pub fn new(port: PcSemanticPort<M>) -> Self {
        Self { port }
    }
    pub fn port(&self) -> &PcSemanticPort<M> {
        &self.port
    }
    pub fn port_mut(&mut self) -> &mut PcSemanticPort<M> {
        &mut self.port
    }
    pub fn result(&self) -> PcPortResult {
        self.port.finish_materialized()
    }

    #[allow(clippy::result_large_err)]
    pub fn into_port(
        self,
        receipt: &RoleOnlyShutdownReceipt,
    ) -> Result<PcSemanticPort<M>, (Self, AdaptiveManagerError)> {
        if !receipt.permits_semantic_closure()
            || receipt.authority != self.port.authority()
            || receipt.mode != self.port.mode()
            || self.port.original_root_budget() != Some(receipt.root_budget)
        {
            return Err((self, AdaptiveManagerError::PhysicalUnknown));
        }
        Ok(self.port)
    }
}

impl<M: RoleModel> RoleOnlySearchDriver<PcPortQuestion> for PcPortDriver<M> {
    fn researcher(&self) -> &NoResearcher {
        self.port.researcher()
    }
    fn validate_registration(
        &self,
        authority: SearchAuthority,
        mode: rz_contracts::rovent::ExecutionMode,
        hard_deadline: Instant,
        root_budget: Digest,
    ) -> Result<(), AdaptiveManagerError> {
        if authority != self.port.authority()
            || mode != self.port.mode()
            || hard_deadline != self.port.hard_deadline()
            || self.port.original_root_budget() != Some(root_budget)
        {
            return Err(AdaptiveManagerError::Invalid(
                "P/C exact semantic registration",
            ));
        }
        Ok(())
    }
    fn next_role(
        &mut self,
    ) -> Result<Option<FrameInput<ReviewView, PcPortQuestion>>, AdaptiveManagerError> {
        self.port.next_role().map_err(semantic)
    }
    fn role_submitted(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), AdaptiveManagerError> {
        self.port.role_submitted(input, ticket).map_err(semantic)
    }
    fn validate_role(
        &self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .validate_role(input, ticket, output)
            .map_err(semantic)
    }
    fn consume_accepted(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        let physical = physical.map(|evidence| AdaptivePortRoleEvidence {
            execution: evidence.execution,
            completed_nn_inputs: evidence.completed_nn_inputs,
            measurement_complete: evidence.measurement_complete,
            physical_fresh: evidence.physical_fresh,
        });
        self.port
            .consume_accepted(input, accepted, output, physical)
            .map_err(semantic)
    }
    fn role_discarded(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: Option<&RoleTicketEnvelope>,
    ) -> Result<(), AdaptiveManagerError> {
        self.port.role_discarded(input, ticket).map_err(semantic)
    }
    fn is_complete(&self) -> bool {
        self.port.is_complete()
    }
    fn revoke(&mut self) -> Result<(), AdaptiveManagerError> {
        self.port.revoke();
        Ok(())
    }
    fn review_lease_admission(&self) -> Result<AdaptiveReviewLeaseAdmission, AdaptiveManagerError> {
        self.port.review_lease_admission().map_err(semantic)
    }
}

#[cfg(feature = "onnx-cpu")]
impl crate::pals_adaptive::NativeQuestionProjection for PcPortQuestion {
    fn project_native_question(
        &self,
    ) -> Result<
        crate::pals_native::ticketed::NativeTicketQuestion,
        rz_search::pals::coordinator::RoleFailure,
    > {
        Ok(crate::pals_native::ticketed::NativeTicketQuestion {
            position: self.position().clone(),
            context: self.context().clone(),
            deadline: self.deadline(),
            legacy_full_records: None,
        })
    }
    fn owned_host_bytes(&self) -> Option<u64> {
        PcPortQuestion::owned_host_bytes(self)
    }
}
