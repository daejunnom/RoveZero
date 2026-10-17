//! Original consumed contextual Task provenance. It owns no ReviewView pin,
//! cannot submit/resume work, and survives the bounded admission's retirement
//! only beside its actual retained Node evidence.
use super::*;
use crate::cpu_worker::CpuWorkerProfile;
use crate::pals::store::StoreReadOwner;
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuWorkerId};

#[derive(Clone, Debug)]
pub(in crate::pals::engine) struct CpuEvidenceContextSeal {
    store: StoreReadOwner,
    generation: u64,
    root: StateId,
    authority: SearchAuthority,
    frame: FrameId,
    question: ContractDigest,
    task: TaskKey,
    execution: ExecutionId,
    observation: ObservationId,
    job: CpuJobEnvelope,
    owner: CpuWorkerId,
    profile: CpuWorkerProfile,
}

#[cfg(test)]
mod tests {
    use super::super::conditional_evidence::tests::{accepted_raw, port as contract_port};
    use super::*;
    use rz_contracts::pals_adaptive::CpuJobId;

    // Reuse the Store semantic fixture, which makes no physical search claim.
    // Actual Bound/worker minting is exercised by the UCI owner integration.
    fn consumed_context(
        port: &AdaptiveEnginePort<LegalOrderRoleMock>,
        node: usize,
    ) -> CpuEvidenceContextSeal {
        let evidence = port.engine.semantic.nodes[node].evidence.as_ref().unwrap();
        let (observation, execution) = evidence.provenance.unwrap();
        let task = port.engine.semantic.stores.tasks.get(execution).unwrap();
        assert!(matches!(task.status, TaskStatus::Completed(id) if id == observation));
        let profile = CpuWorkerProfile {
            owner: CpuWorkerId(1),
            max_depth: 8,
            evaluation_namespace: ContractDigest([31; 32]),
            search_conditions: ContractDigest([32; 32]),
        };
        let question = ContractDigest([33; 32]);
        CpuEvidenceContextSeal {
            store: port.engine.semantic.stores.read_owner(),
            generation: port.engine.semantic.stores.generation(),
            root: port.root_state_id(),
            authority: port.authority,
            frame: FrameId {
                authority: port.authority,
                sequence: 1,
            },
            question,
            task: task.key.clone(),
            execution,
            observation,
            job: CpuJobEnvelope {
                id: CpuJobId {
                    authority: port.authority,
                    sequence: 1,
                },
                state_identity: port.engine.semantic.nodes[node]
                    .position
                    .export()
                    .unwrap()
                    .snapshot()
                    .identity(),
                question,
                root_restriction: ContractDigest([0; 32]),
                evaluation_namespace: profile.evaluation_namespace,
                search_conditions: profile.search_conditions,
                total_native_node_credit: task.key.node_budget,
                hard_deadline: port.deadline,
            },
            owner: profile.owner,
            profile,
        }
    }

    #[test]
    fn retained_consumed_context_does_not_depend_on_active_admission() {
        let mut port = contract_port(Position::startpos());
        let movement = BoardMove::from_uci("e2e4").unwrap();
        accepted_raw(&mut port, &[movement], 14, 2);
        let node = port.replay_node(&[movement]).unwrap();
        let seal = consumed_context(&port, node);
        let task = seal.task.clone();
        let execution = seal.execution;
        let observation = seal.observation;
        port.engine.semantic.nodes[node]
            .evidence
            .as_mut()
            .unwrap()
            .context = Some(Arc::new(seal));
        port.cpu_admissions.clear();
        let evidence = port.engine.semantic.nodes[node].evidence.as_ref().unwrap();
        assert!(port.conditional_cpu_task_condition_matches(
            evidence,
            &task,
            execution,
            observation
        ));
        let foreign = contract_port(Position::startpos());
        assert!(!foreign.conditional_cpu_task_condition_matches(
            evidence,
            &task,
            execution,
            observation
        ));
        assert!(!port.conditional_cpu_task_condition_matches(
            evidence,
            &task,
            execution,
            ObservationId(observation.0 + 1)
        ));
        let mut other_task = task;
        other_task
            .cpu_condition
            .as_mut()
            .unwrap()
            .push_str(";nnuec-context-binding=another-view");
        assert!(!port.conditional_cpu_task_condition_matches(
            evidence,
            &other_task,
            execution,
            observation
        ));
    }
}

/// Live installation admission is checked before the Store consumes anything.
/// Finalizing that already consumed fact never reopens a canceled authority.
pub(super) struct PreparedCpuEvidenceContext {
    profile: CpuWorkerProfile,
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    pub(super) fn prepare_cpu_context_seal(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        admission: &CpuPortAdmission,
    ) -> Result<Option<PreparedCpuEvidenceContext>, PalsError> {
        if self.engine.owned_identity()?.semantics != crate::cpu_value::nnuec::v2::SEMANTICS {
            return Ok(None);
        }
        let profile = self
            .installed_cpu_binding(work)?
            .ok_or(StoreError::InvalidEvidence(
                "consumed CPU context has no original installation",
            ))?;
        if admission.owner != Some(profile.owner)
            || admission.profiles.as_slice() != [profile]
            || profile.evaluation_namespace != admission.evaluation_namespace
            || profile.search_conditions != admission.search_conditions
        {
            return Err(StoreError::InvalidEvidence(
                "CPU context differs from original installed profile",
            )
            .into());
        }
        Ok(Some(PreparedCpuEvidenceContext { profile }))
    }

    pub(super) fn consumed_cpu_context_seal(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        admission: &CpuPortAdmission,
        prepared: Option<PreparedCpuEvidenceContext>,
        execution: ExecutionId,
        observation: ObservationId,
    ) -> Result<Option<Arc<CpuEvidenceContextSeal>>, PalsError> {
        let Some(PreparedCpuEvidenceContext { profile }) = prepared else {
            return Ok(None);
        };
        let stored = self.cpu_admissions.get(&work.input.question_digest).ok_or(
            StoreError::InvalidEvidence("consumed CPU context lost original admission"),
        )?;
        let job = admission.job.ok_or(StoreError::InvalidEvidence(
            "consumed CPU context has no Job",
        ))?;
        let owner = admission.owner.ok_or(StoreError::InvalidEvidence(
            "consumed CPU context has no owner",
        ))?;
        if stored.consumed_observation != Some(observation)
            || stored
                .committed_execution
                .is_none_or(|(_, original, observed)| {
                    original != execution || observed != observation
                })
            || admission.execution != Some(execution)
            || profile.owner != owner
            || admission.profiles.as_slice() != [profile]
            || profile.evaluation_namespace != admission.evaluation_namespace
            || profile.search_conditions != admission.search_conditions
            || job.question != work.input.question_digest
            || job.id.authority != self.authority
            || job.evaluation_namespace != admission.evaluation_namespace
            || job.search_conditions != admission.search_conditions
            || self.engine.semantic.stores.tasks.get(execution)?.key != admission.task_key
            || self
                .engine
                .semantic
                .stores
                .observations
                .get(observation)?
                .execution
                != Some(execution)
        {
            return Err(StoreError::InvalidEvidence(
                "CPU context differs from original consumed task",
            )
            .into());
        }
        Ok(Some(Arc::new(CpuEvidenceContextSeal {
            store: self.engine.semantic.stores.read_owner(),
            generation: self.engine.semantic.stores.generation(),
            root: self.root_state_id(),
            authority: self.authority,
            frame: work.frame,
            question: work.input.question_digest,
            task: admission.task_key.clone(),
            execution,
            observation,
            job,
            owner,
            profile,
        })))
    }

    pub(super) fn conditional_cpu_task_condition_matches(
        &self,
        accepted: &CpuEvidence,
        task: &TaskKey,
        execution: ExecutionId,
        observation: ObservationId,
    ) -> bool {
        let Some(context) = &accepted.context else {
            return task.cpu_condition.as_ref() == Some(&self.engine.cpu_condition());
        };
        context.store.matches(&self.engine.semantic.stores)
            && context.generation == self.engine.semantic.stores.generation()
            && context.authority == self.authority
            && context.frame.authority == self.authority
            && context.root == self.root_state_id()
            && context.task == *task
            && context.execution == execution
            && context.observation == observation
            && context.job.id.authority == self.authority
            && context.job.question == context.question
            && context.job.evaluation_namespace == context.profile.evaluation_namespace
            && context.job.search_conditions == context.profile.search_conditions
            && context.profile.owner == context.owner
            && accepted.provenance == Some((observation, execution))
            && task.value_identity.as_ref() == Some(&accepted.value_identity)
    }
}
