//! Fixed-size observations from the original single native owner. Public K/V
//! reuse, a whole role evaluation, owner acknowledgement and Search consumption
//! are separate scopes. Unknown commands never become completed-input zero.
use super::manager::AdaptiveManagerError;
use crate::pals_native::ticketed::{
    TicketExecutionEvidence, TicketNativeAccountingEvidence, TicketOwnerRunWitness,
};
use rz_contracts::pals::Role;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_search::pals::coordinator::PhysicalRoleCompletion;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "search-work-receipts", derive(serde::Serialize))]
pub struct AdaptiveNativeAccountingSnapshot {
    completed_service_commands: u64,
    backend_run_invoked_commands: u64,
    backend_run_skipped_commands: u64,
    unknown_native_accounting_commands: u64,
    unknown_evaluation_origin_commands: u64,
    physical_execution_tickets: u64,
    terminal_without_nn_dispatch: u64,
    public_cache_hits: u64,
    public_cache_misses: u64,
    public_nn_runs_attempted: u64,
    public_nn_runs_completed: u64,
    public_nn_runs_failed_known: u64,
    role_nn_runs_attempted: u64,
    role_nn_runs_completed: u64,
    role_nn_runs_failed_known: u64,
    validator_nn_runs_attempted: u64,
    validator_nn_runs_completed: u64,
    validator_nn_runs_failed_known: u64,
    owner_accepted_outputs: u64,
    owner_accepted_neural_outputs: u64,
    owner_accepted_origin_unknown: u64,
    search_consumed_outputs: u64,
    search_consumed_neural_outputs: u64,
    search_consumed_origin_unknown: u64,
}

macro_rules! getters {
    ($($field:ident),+ $(,)?) => {
        $(pub fn $field(&self) -> u64 { self.$field })+
    };
}
impl AdaptiveNativeAccountingSnapshot {
    getters!(
        completed_service_commands,
        backend_run_invoked_commands,
        backend_run_skipped_commands,
        unknown_native_accounting_commands,
        unknown_evaluation_origin_commands,
        physical_execution_tickets,
        terminal_without_nn_dispatch,
        public_cache_hits,
        public_cache_misses,
        public_nn_runs_attempted,
        public_nn_runs_completed,
        public_nn_runs_failed_known,
        role_nn_runs_attempted,
        role_nn_runs_completed,
        role_nn_runs_failed_known,
        validator_nn_runs_attempted,
        validator_nn_runs_completed,
        validator_nn_runs_failed_known,
        owner_accepted_outputs,
        owner_accepted_neural_outputs,
        owner_accepted_origin_unknown,
        search_consumed_outputs,
        search_consumed_neural_outputs,
        search_consumed_origin_unknown,
    );

    /// This is a complete partition only at the caller's actual settled root
    /// fence, after every mailbox dispatch has supplied one original completion.
    /// A zero public cache delta is not whole-evaluation cache evidence.
    pub fn accounting_complete_for_dispatches(&self, dispatched: u64) -> bool {
        self.completed_service_commands == dispatched
            && self.unknown_native_accounting_commands == 0
    }
    pub fn physical_partition_complete_for_dispatches(&self, dispatched: u64) -> bool {
        self.accounting_complete_for_dispatches(dispatched)
            && self.unknown_evaluation_origin_commands == 0
            && self
                .physical_execution_tickets
                .checked_add(self.terminal_without_nn_dispatch)
                == Some(dispatched)
    }
    pub fn validator_called_for_dispatches(&self, dispatched: u64) -> Option<bool> {
        self.accounting_complete_for_dispatches(dispatched)
            .then_some(self.validator_nn_runs_attempted > 0)
    }
    /// No full-result cache witness currently exists. Known zero here follows
    /// only when every actually acknowledged output includes its own completed
    /// role graph. Public K/V reuse never increments this evaluation count.
    pub fn cache_evaluations_accepted(&self) -> Option<u64> {
        (self.owner_accepted_origin_unknown == 0
            && self.owner_accepted_neural_outputs == self.owner_accepted_outputs)
            .then_some(0)
    }
    pub fn search_consumption_origin_known(&self) -> bool {
        self.search_consumed_origin_unknown == 0
            && self.search_consumed_neural_outputs == self.search_consumed_outputs
    }

    pub(super) fn observe_completion(
        &mut self,
        ticket: &RoleTicketEnvelope,
        owner_run: &TicketOwnerRunWitness,
        physical: PhysicalRoleCompletion,
        output_success: bool,
        evidence: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        if owner_run.ticket() != ticket {
            return Err(AdaptiveManagerError::Invalid(
                "native accounting owner-run witness belongs to another ticket",
            ));
        }
        self.observe_completion_facts(
            ticket.role,
            owner_run.backend_run_invoked(),
            physical,
            output_success,
            evidence,
        )
    }
    fn observe_completion_facts(
        &mut self,
        role: Role,
        backend_invoked: bool,
        physical: PhysicalRoleCompletion,
        output_success: bool,
        evidence: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        let mut next = *self;
        add(&mut next.completed_service_commands, 1)?;
        if backend_invoked {
            add(&mut next.backend_run_invoked_commands, 1)?;
        } else {
            add(&mut next.backend_run_skipped_commands, 1)?;
        }
        if !backend_invoked {
            // Actual service producer skipped backend.run, before any role Run.
            // A successful output or contradictory evidence invalidates that
            // partition rather than being relabelled as a cache hit.
            if physical == PhysicalRoleCompletion::Confirmed
                && !output_success
                && evidence.is_none()
            {
                add(&mut next.terminal_without_nn_dispatch, 1)?;
            } else {
                add(&mut next.unknown_native_accounting_commands, 1)?;
                add(&mut next.unknown_evaluation_origin_commands, 1)?;
            }
            *self = next;
            return Ok(());
        }
        let Some(actual) = known_owner_accounting(role, physical, evidence) else {
            add(&mut next.unknown_native_accounting_commands, 1)?;
            add(&mut next.unknown_evaluation_origin_commands, 1)?;
            *self = next;
            return Ok(());
        };
        next.add_owner_delta(&actual)?;
        let attempted = actual
            .public_nn_runs_attempted()
            .checked_add(actual.role_nn_runs_attempted())
            .ok_or(AdaptiveManagerError::Capacity("native attempted Run count"))?;
        if attempted > 0 {
            add(&mut next.physical_execution_tickets, 1)?;
        } else if !output_success {
            add(&mut next.terminal_without_nn_dispatch, 1)?;
        } else {
            // An authored or future zero-Run success lacks explicit full-result
            // cache provenance. Its native V delta can still be known, while
            // the physical/evaluation-origin partition remains unknown.
            add(&mut next.unknown_evaluation_origin_commands, 1)?;
        }
        *self = next;
        Ok(())
    }

    pub(super) fn observe_owner_acceptance(
        &mut self,
        role: Role,
        evidence: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        let mut next = *self;
        add(&mut next.owner_accepted_outputs, 1)?;
        if completed_role_origin(role, evidence) {
            add(&mut next.owner_accepted_neural_outputs, 1)?;
        } else {
            add(&mut next.owner_accepted_origin_unknown, 1)?;
        }
        *self = next;
        Ok(())
    }
    pub(super) fn observe_search_consumption(
        &mut self,
        role: Role,
        evidence: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        let mut next = *self;
        add(&mut next.search_consumed_outputs, 1)?;
        if completed_role_origin(role, evidence) {
            add(&mut next.search_consumed_neural_outputs, 1)?;
        } else {
            add(&mut next.search_consumed_origin_unknown, 1)?;
        }
        *self = next;
        Ok(())
    }
    // Only actual native bootstrap aggregates across roots; retain the focused
    // accounting unit test in builds without that optional consumer.
    #[cfg(any(test, all(feature = "pals-adaptive", feature = "onnx-cpu")))]
    pub(crate) fn checked_merge(&mut self, actual: Self) -> Result<(), AdaptiveManagerError> {
        let mut next = *self;
        macro_rules! merge {
            ($($field:ident),+ $(,)?) => { $(add(&mut next.$field,actual.$field)?;)+ };
        }
        merge!(
            completed_service_commands,
            backend_run_invoked_commands,
            backend_run_skipped_commands,
            unknown_native_accounting_commands,
            unknown_evaluation_origin_commands,
            physical_execution_tickets,
            terminal_without_nn_dispatch,
            public_cache_hits,
            public_cache_misses,
            public_nn_runs_attempted,
            public_nn_runs_completed,
            public_nn_runs_failed_known,
            role_nn_runs_attempted,
            role_nn_runs_completed,
            role_nn_runs_failed_known,
            validator_nn_runs_attempted,
            validator_nn_runs_completed,
            validator_nn_runs_failed_known,
            owner_accepted_outputs,
            owner_accepted_neural_outputs,
            owner_accepted_origin_unknown,
            search_consumed_outputs,
            search_consumed_neural_outputs,
            search_consumed_origin_unknown,
        );
        *self = next;
        Ok(())
    }
    fn add_owner_delta(
        &mut self,
        actual: &TicketNativeAccountingEvidence,
    ) -> Result<(), AdaptiveManagerError> {
        macro_rules! delta {
            ($($field:ident),+ $(,)?) => { $(add(&mut self.$field,actual.$field())?;)+ };
        }
        delta!(
            public_cache_hits,
            public_cache_misses,
            public_nn_runs_attempted,
            public_nn_runs_completed,
            public_nn_runs_failed_known,
            role_nn_runs_attempted,
            role_nn_runs_completed,
            role_nn_runs_failed_known,
            validator_nn_runs_attempted,
            validator_nn_runs_completed,
            validator_nn_runs_failed_known,
        );
        Ok(())
    }
}

fn add(total: &mut u64, amount: u64) -> Result<(), AdaptiveManagerError> {
    *total = total
        .checked_add(amount)
        .ok_or(AdaptiveManagerError::Capacity(
            "native accounting count overflow",
        ))?;
    Ok(())
}
fn known_owner_accounting(
    role: Role,
    physical: PhysicalRoleCompletion,
    evidence: Option<TicketExecutionEvidence>,
) -> Option<TicketNativeAccountingEvidence> {
    if physical != PhysicalRoleCompletion::Confirmed {
        return None;
    }
    let evidence = evidence.filter(|actual| actual.measurement_complete)?;
    let actual = evidence.accounting?;
    let input_count = evidence.completed_nn_inputs?;
    let public_attempted = actual.public_nn_runs_attempted();
    let role_attempted = actual.role_nn_runs_attempted();
    if public_attempted > 1
        || role_attempted > 1
        || actual
            .public_cache_hits()
            .checked_add(actual.public_cache_misses())?
            > 1
        || actual
            .public_nn_runs_completed()
            .checked_add(actual.public_nn_runs_failed_known())?
            != public_attempted
        || actual
            .role_nn_runs_completed()
            .checked_add(actual.role_nn_runs_failed_known())?
            != role_attempted
        || actual
            .public_nn_runs_completed()
            .checked_add(actual.role_nn_runs_completed())?
            != input_count
    {
        return None;
    }
    let validator = role == Role::Validator;
    if actual.validator_nn_runs_attempted() != if validator { role_attempted } else { 0 }
        || actual.validator_nn_runs_completed()
            != if validator {
                actual.role_nn_runs_completed()
            } else {
                0
            }
        || actual.validator_nn_runs_failed_known()
            != if validator {
                actual.role_nn_runs_failed_known()
            } else {
                0
            }
    {
        return None;
    }
    Some(actual)
}
fn completed_role_origin(role: Role, evidence: Option<TicketExecutionEvidence>) -> bool {
    known_owner_accounting(role, PhysicalRoleCompletion::Confirmed, evidence)
        .is_some_and(|actual| actual.role_nn_runs_completed() == 1)
}

#[cfg(test)]
#[path = "native_accounting/tests.rs"]
mod tests;
