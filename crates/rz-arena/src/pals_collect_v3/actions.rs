//! Causal action bookkeeping. Utility requires a new accepted store witness;
//! depth, model agreement, duplicate lines and witness-free Defer never reward V.
use super::*;
use rz_contracts::ExecutionId;
use rz_contracts::pals::SearchAuthority;
use rz_search::pals::store::{ObservationId, ObservationKind, PalsStores, RawScore};
use std::time::Instant;

/// A non-deserializable capability from the arena's accepted Store commit
/// bridge. A public ObservationId alone is not causal action evidence.
pub struct AcceptedActionWitnessV3 {
    input_sha256: String,
    action_execution: ExecutionId,
    authority: SearchAuthority,
    work_execution: ExecutionId,
    store_execution: rz_search::pals::store::ExecutionId,
    store_generation: u64,
    observation: ObservationId,
    accepted_sequence: u64,
}
impl AcceptedActionWitnessV3 {
    /// The actual integration consumer calls this after verified coordinator
    /// acceptance and Store commit with its exact old/new execution map.
    #[allow(dead_code)]
    #[allow(
        clippy::too_many_arguments,
        reason = "Each independently validated input, execution, authority and Store acceptance seal must remain explicit at the owner commit boundary"
    )]
    pub(crate) fn from_owner_commit(
        input_sha256: String,
        action_execution: ExecutionId,
        authority: SearchAuthority,
        work_execution: ExecutionId,
        store_execution: rz_search::pals::store::ExecutionId,
        store_generation: u64,
        observation: ObservationId,
        accepted_sequence: u64,
    ) -> Result<Self, ArenaError> {
        check(
            generation_sha(&input_sha256)
                && action_execution.epoch == authority.epoch
                && work_execution.epoch == authority.epoch
                && action_execution.sequence > 0
                && work_execution.sequence > 0
                && accepted_sequence > 0,
            "accepted action commit identity",
        )?;
        Ok(Self {
            input_sha256,
            action_execution,
            authority,
            work_execution,
            store_execution,
            store_generation,
            observation,
            accepted_sequence,
        })
    }
}
pub struct AcceptedActionCompletionV3 {
    input_sha256: String,
    execution: ExecutionId,
    authority: SearchAuthority,
}
impl AcceptedActionCompletionV3 {
    /// Worker Yield/partial completion cannot issue semantic action completion.
    #[allow(dead_code)]
    pub(crate) fn from_owner_completion(
        input_sha256: String,
        execution: ExecutionId,
        authority: SearchAuthority,
    ) -> Result<Self, ArenaError> {
        check(
            generation_sha(&input_sha256)
                && execution.epoch == authority.epoch
                && execution.sequence > 0,
            "accepted action completion identity",
        )?;
        Ok(Self {
            input_sha256,
            execution,
            authority,
        })
    }
}

pub struct ActionExecutionCaptureV3 {
    input_sha256: String,
    action: GenerationTaskV3,
    execution: ExecutionId,
    authority: SearchAuthority,
    started: Instant,
    hard_deadline: Instant,
    known_work_observed: bool,
    unknown_work: bool,
    work: BTreeSet<(u8, u64, u64)>,
    witnesses: Vec<GenerationWitnessV3>,
    witness_keys: BTreeSet<(u64, usize)>,
    witness_lines: BTreeSet<String>,
    store_generation: Option<u64>,
    observation_baseline: usize,
    cpu_bindings: BTreeMap<(u64, u64), rz_contracts::pals_adaptive::SliceGrant>,
    external_bindings: BTreeMap<
        (u64, u64),
        (
            rz_contracts::pals_adaptive::ExternalCpuJobEnvelope,
            rz_contracts::pals_adaptive::ExternalSliceGrant,
        ),
    >,
}
impl ActionExecutionCaptureV3 {
    pub fn begin(
        record: &GenerationRecordV3,
        action: GenerationTaskV3,
        execution: ExecutionId,
        authority: SearchAuthority,
        started: Instant,
        hard_deadline: Instant,
        stores: &PalsStores,
    ) -> Result<Self, ArenaError> {
        record.validate()?;
        check(
            record.input.role == GenerationRoleV3::Validator
                && record.input.action_support_mask[action.index()]
                && execution.epoch == authority.epoch
                && execution.sequence > 0
                && started < hard_deadline,
            "unsupported or wrong-owner actual action",
        )?;
        Ok(Self {
            input_sha256: record.input_sha256.clone(),
            action,
            execution,
            authority,
            started,
            hard_deadline,
            known_work_observed: false,
            unknown_work: false,
            work: BTreeSet::new(),
            witnesses: vec![],
            witness_keys: BTreeSet::new(),
            witness_lines: BTreeSet::new(),
            store_generation: Some(stores.generation()),
            observation_baseline: stores.observations.next_observation_sequence(),
            cpu_bindings: BTreeMap::new(),
            external_bindings: BTreeMap::new(),
        })
    }
    pub fn bind_cpu_before_dispatch(
        &mut self,
        job: &rz_contracts::pals_adaptive::CpuJobEnvelope,
        grant: rz_contracts::pals_adaptive::SliceGrant,
    ) -> Result<(), ArenaError> {
        let key = (grant.execution.epoch.0, grant.execution.sequence);
        check(
            job.id.authority == self.authority
                && grant.job == job.id
                && grant.execution.epoch == self.authority.epoch
                && self.cpu_bindings.len() < 64
                && !self.cpu_bindings.contains_key(&key)
                && !self.external_bindings.contains_key(&key),
            "action CPU binding scope/duplicate/capacity",
        )?;
        self.cpu_bindings.insert(key, grant);
        Ok(())
    }
    pub fn bind_external_before_dispatch(
        &mut self,
        job: &rz_search::external_cpu::worker::ExternalJob,
    ) -> Result<(), ArenaError> {
        let key = (job.execution.epoch.0, job.execution.sequence);
        check(
            job.envelope.id.authority == self.authority
                && job.grant.job == job.envelope.id
                && !job.control.snapshot().terminal
                && self.external_bindings.len() < 64
                && !self.cpu_bindings.contains_key(&key)
                && !self.external_bindings.contains_key(&key),
            "action external binding scope/duplicate/capacity",
        )?;
        self.external_bindings
            .insert(key, (job.envelope, job.grant));
        Ok(())
    }
    /// The real packet, not a caller's completed boolean, proves worker work.
    pub fn observe_cpu(
        &mut self,
        report: &rz_search::cpu_worker::CpuSliceReport,
    ) -> Result<(), ArenaError> {
        check(
            report.grant.job.authority == self.authority
                && self.cpu_bindings.get(&(
                    report.grant.execution.epoch.0,
                    report.grant.execution.sequence,
                )) == Some(&report.grant)
                && !report.admission_rejected
                && self.work.insert((
                    0,
                    report.grant.execution.epoch.0,
                    report.grant.execution.sequence,
                )),
            "CPU action scope/duplicate work",
        )?;
        if report.actual_work.is_none() {
            self.unknown_work = true;
            return Ok(());
        }
        // This flags an observed known worker occupancy, NOT completion of the
        // original semantic action. Yield may contribute a valid witness; the
        // separate opaque manager completion in finish is still mandatory.
        match &report.result {
            Ok(result)
                if matches!(
                    result.report.completion,
                    rz_search::cpu::CpuCompletion::Canceled
                        | rz_search::cpu::CpuCompletion::Deadline
                ) => {}
            Ok(_) => self.known_work_observed = true,
            Err(_) => self.unknown_work = true,
        }
        Ok(())
    }
    pub fn observe_external(
        &mut self,
        report: &rz_search::external_cpu::worker::ExternalWorkerReport,
    ) -> Result<(), ArenaError> {
        check(
            report.job.id.authority == self.authority
                && self.external_bindings.get(&(
                    report.scope.execution.epoch.0,
                    report.scope.execution.sequence,
                )) == Some(&(report.job, report.grant))
                && self.work.insert((
                    1,
                    report.scope.execution.epoch.0,
                    report.scope.execution.sequence,
                )),
            "external action scope/duplicate work",
        )?;
        match &report.result {
            Ok(actual)
                if actual.readiness_barrier_completed
                    && actual.go_issue == rz_search::external_cpu::ExternalGoIssue::Issued =>
            {
                self.known_work_observed = true
            }
            _ => self.unknown_work = true,
        }
        Ok(())
    }
    /// Only a store-owned conditional accepted Refutation/Repair may reward V.
    /// Raw CP/mate, unknown observations and neural opinion alone are excluded.
    pub fn observe_new_store_witness(
        &mut self,
        record: &GenerationRecordV3,
        stores: &PalsStores,
        commit: &AcceptedActionWitnessV3,
        evidence: &mut BTreeMap<String, serde_json::Value>,
    ) -> Result<(), ArenaError> {
        check(
            record.input_sha256 == self.input_sha256
                && commit.input_sha256 == self.input_sha256
                && commit.action_execution == self.execution
                && commit.authority == self.authority
                && commit.store_generation == stores.generation()
                && Some(stores.generation()) == self.store_generation
                && commit.observation.0 >= self.observation_baseline
                && self.work.iter().any(|(_, epoch, sequence)| {
                    (*epoch, *sequence)
                        == (
                            commit.work_execution.epoch.0,
                            commit.work_execution.sequence,
                        )
                })
                && commit.accepted_sequence > record.input.capture_sequence
                && self.witnesses.len() < 256
                && self
                    .witness_keys
                    .insert((stores.generation(), commit.observation.0)),
            "old/duplicate action witness or original input mismatch",
        )?;
        if let Some(generation) = self.store_generation {
            check(
                generation == stores.generation(),
                "new game cannot contribute an old action witness",
            )?;
        } else {
            self.store_generation = Some(stores.generation());
        }
        let observed = stores
            .observations
            .get(commit.observation)
            .map_err(|e| invalid(e.to_string()))?;
        check(
            observed.execution == Some(commit.store_execution),
            "accepted witness did not consume its mapped action execution",
        )?;
        let kind = match (observed.kind, observed.score) {
            (ObservationKind::Refutation, RawScore::ConditionalWdl { .. }) => {
                GenerationRecordKindV3::Counter
            }
            (ObservationKind::Repair, RawScore::ConditionalRepairWdl { .. }) => {
                GenerationRecordKindV3::Repair
            }
            // Exact Rules termination of a checked continuation is also evidence.
            (ObservationKind::Refutation, RawScore::Terminal { .. }) => {
                GenerationRecordKindV3::Counter
            }
            (ObservationKind::Repair, RawScore::Terminal { .. }) => GenerationRecordKindV3::Repair,
            _ => {
                return Err(invalid(
                    "raw/unknown/opinion observation is not a new conditional action witness",
                ));
            }
        };
        let line = observed
            .line
            .ok_or_else(|| invalid("accepted witness requires its Rules checked line"))?;
        let moves = stores
            .lines
            .moves(line)
            .map_err(|e| invalid(e.to_string()))?;
        let origin = stores
            .lines
            .get(line)
            .map_err(|e| invalid(e.to_string()))?
            .start_state;
        let snapshot = stores
            .states
            .get(origin)
            .map_err(|e| invalid(e.to_string()))?;
        let mut position =
            Position::from_snapshot(snapshot, rz_position::PositionLimits::default())
                .map_err(|e| invalid(e.to_string()))?;
        check(
            exact_rules_anchor(&position)? == record.input.rules_anchor,
            "witness line belongs to another actual root",
        )?;
        let packed = pack(&moves)?;
        let line_sha = generation_digest(
            "rz-pals-action-witness-line/1",
            &(kind, record.input.rules_anchor.clone(), &packed),
        )?;
        check(
            !record.input.selected_view.records.iter().any(|r| {
                record.input.selected_view.anchors.get(r.anchor as usize)
                    == Some(&record.input.rules_anchor)
                    && r.moves == packed
            }) && self.witness_lines.insert(line_sha.clone()),
            "rediscovered/visible line cannot reward V",
        )?;
        // Store lines include their own base state; use its full checked walk.
        for movement in &moves {
            position
                .make_move(*movement)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let witness_id = format!("witness-{line_sha}");
        check(
            !record
                .input
                .selected_view
                .records
                .iter()
                .any(|r| r.record_id == witness_id),
            "visible witness is not new",
        )?;
        let mut witness = GenerationWitnessV3 {
            witness_id,
            kind,
            observed_sequence: commit.accepted_sequence,
            raw_evidence_sha256: "0".repeat(64),
        };
        let mut content = serde_json::to_value(&witness).unwrap();
        content
            .as_object_mut()
            .unwrap()
            .remove("raw_evidence_sha256");
        let receipt = serde_json::json!({"schema":"rz-pals-generation-witness-evidence/1","input_sha256":self.input_sha256,"witness":content,"rules_validated":true});
        witness.raw_evidence_sha256 =
            generation_digest("rz-pals-generation-witness-evidence/1", &receipt)?;
        evidence.insert(witness.raw_evidence_sha256.clone(), receipt);
        self.witnesses.push(witness);
        Ok(())
    }
    /// Every stage is included in action wall time, from original dispatch until
    /// acceptance. Missing/unknown work remains incomplete; cost is never zeroed.
    pub fn finish(
        self,
        record: &GenerationRecordV3,
        finished: Instant,
        accepted: Option<&AcceptedActionCompletionV3>,
        evidence: &mut BTreeMap<String, serde_json::Value>,
    ) -> Result<GenerationActionOutcomeV3, ArenaError> {
        check(
            record.input_sha256 == self.input_sha256 && finished >= self.started,
            "action cost clock/input mismatch",
        )?;
        let wall_cost_ns = u64::try_from(finished.duration_since(self.started).as_nanos())
            .map_err(|_| invalid("action wall cost overflow"))?;
        let mut outcome = GenerationActionOutcomeV3 {
            action: self.action,
            execution_id: format!(
                "action-{}-{}",
                self.execution.epoch.0, self.execution.sequence
            ),
            completed: self.known_work_observed
                && !self.unknown_work
                && finished <= self.hard_deadline
                && accepted.is_some_and(|a| {
                    a.input_sha256 == self.input_sha256
                        && a.execution == self.execution
                        && a.authority == self.authority
                }),
            wall_cost_ns: Some(wall_cost_ns),
            new_witnesses: self.witnesses,
            causal_receipt_sha256: "0".repeat(64),
        };
        let mut content = serde_json::to_value(&outcome).unwrap();
        content
            .as_object_mut()
            .unwrap()
            .remove("causal_receipt_sha256");
        let receipt = serde_json::json!({"schema":"rz-pals-generation-causal-evidence/1","input_sha256":self.input_sha256,"outcome":content});
        outcome.causal_receipt_sha256 =
            generation_digest("rz-pals-generation-causal-evidence/1", &receipt)?;
        evidence.insert(outcome.causal_receipt_sha256.clone(), receipt);
        Ok(outcome)
    }
}
pub fn attach_actual_task_outcomes_v3(
    record: &mut GenerationRecordV3,
    outcomes: Vec<GenerationActionOutcomeV3>,
    observed_sequence: u64,
    evidence: &mut BTreeMap<String, serde_json::Value>,
) -> Result<(), ArenaError> {
    record.validate()?;
    check(
        record.input.role == GenerationRoleV3::Validator && record.future_targets.task.is_none(),
        "V task target role/duplicate",
    )?;
    let derived = generation_derive_tasks(&record.input, &outcomes)?;
    let mut target = GenerationTaskTargetV3 {
        observed_sequence,
        question_id: record.input.question_id.clone(),
        view_sha256: record.input.view_sha256.clone(),
        raw_evidence_sha256: "0".repeat(64),
        source: "actual_action_witness".into(),
        outcomes,
        target_known_mask: derived.target_known_mask,
    };
    target.raw_evidence_sha256 =
        seal_head_evidence(&record.input_sha256, "task", &target, evidence)?;
    let mut candidate = record.clone();
    candidate.future_targets.task = Some(target);
    validate_generation_head_evidence(&candidate, evidence)?;
    *record = candidate;
    Ok(())
}

/// Serialize an actual action only from the manager's opaque Store-commit and
/// physical-fence proof. A worker Yield, public completion boolean or V logits
/// cannot enter this API. Store ordinals remain verbatim raw provenance; the
/// label's chronological sequence comes from the collector event owner.
#[cfg(feature = "pals-collection")]
pub fn capture_completed_validator_action_v3(
    record: &GenerationRecordV3,
    completed: rz_uci::pals_adaptive::CompletedValidatorAction,
    chronology: &mut GenerationChronologyV3,
    evidence: &mut BTreeMap<String, serde_json::Value>,
) -> Result<GenerationActionOutcomeV3, ArenaError> {
    capture_validator_action_inner_v3(record, completed, chronology, evidence, None)
}

/// Frozen registration is supplied only by the isolated actual-action owner.
/// Ordinary selfplay never promotes its shared root latency into this path.
#[cfg(feature = "pals-collection")]
pub(super) fn capture_isolated_validator_action_v3(
    record: &GenerationRecordV3,
    completed: rz_uci::pals_adaptive::CompletedValidatorAction,
    chronology: &mut GenerationChronologyV3,
    evidence: &mut BTreeMap<String, serde_json::Value>,
    frozen: &super::action_driver::FrozenActionRegistrationV3,
) -> Result<GenerationActionOutcomeV3, ArenaError> {
    capture_validator_action_inner_v3(record, completed, chronology, evidence, Some(frozen))
}

#[cfg(feature = "pals-collection")]
fn capture_validator_action_inner_v3(
    record: &GenerationRecordV3,
    completed: rz_uci::pals_adaptive::CompletedValidatorAction,
    chronology: &mut GenerationChronologyV3,
    evidence: &mut BTreeMap<String, serde_json::Value>,
    frozen: Option<&super::action_driver::FrozenActionRegistrationV3>,
) -> Result<GenerationActionOutcomeV3, ArenaError> {
    use rz_search::pals::engine::adaptive_port::validator_port::{
        ActionWitnessKind, AdaptiveValidatorAction,
    };
    record.validate()?;
    let semantic = completed.semantic();
    let source = semantic.source();
    let ticket = super::capture::generation_role_ticket_json_v3(source);
    let native_view_content = hex(&semantic.input().0);
    check(
        record.input.role == GenerationRoleV3::Validator
            && record.input.question_id == hex(&source.question.0)
            && record.input.action_support_mask == semantic.action_support_mask()
            && source.role == rz_contracts::pals::Role::Validator
            && source.id.epoch == source.authority.epoch
            && semantic.id().validator_request == source.id
            && semantic.id().authority == source.authority
            && completed.actual_native_input() == source.representation.input,
        "completed V action does not own the original input/mask/ticket",
    )?;
    let mut submissions = evidence.iter().filter(|(_, raw)| {
        raw.get("schema").and_then(|v| v.as_str())
            == Some("rz-pals-generation-submission-evidence/1")
            && raw.get("input_sha256").and_then(|v| v.as_str())
                == Some(record.input_sha256.as_str())
    });
    let (submission_sha, submission) = submissions
        .next()
        .ok_or_else(|| invalid("V action requires actual submitted input receipt"))?;
    check(
        submissions.next().is_none()
            && generation_digest("rz-pals-generation-submission-evidence/1", submission)?
                == *submission_sha
            && submission.get("ticket") == Some(&ticket)
            && submission
                .get("generation_view_sha256")
                .and_then(|v| v.as_str())
                == Some(record.input.view_sha256.as_str())
            && submission
                .get("source_view_content_sha256")
                .and_then(|v| v.as_str())
                == Some(native_view_content.as_str()),
        "V action original source differs from native input publication",
    )?;
    let completion = evidence
        .iter()
        .find(|(_, raw)| {
            raw.get("schema").and_then(|v| v.as_str())
                == Some("rz-pals-generation-native-completion-evidence/1")
                && raw.get("submission_sha256").and_then(|v| v.as_str())
                    == Some(submission_sha.as_str())
        })
        .ok_or_else(|| invalid("V action input has no physical completion"))?;
    check(
        generation_digest(
            "rz-pals-generation-native-completion-evidence/1",
            completion.1,
        )? == *completion.0
            && completion.1.get("ticket") == Some(&ticket)
            && completion.1.get("input_sha256").and_then(|v| v.as_str())
                == Some(record.input_sha256.as_str())
            && completion
                .1
                .get("admissible_actual_nn")
                .and_then(|v| v.as_bool())
                == Some(true),
        "V action native completion differs or is not actual NN",
    )?;
    let accepted_logits = serde_json::json!(
        semantic
            .accepted_task_logits()
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>()
    );
    check(
        semantic
            .accepted_task_logits()
            .iter()
            .all(|value| value.is_finite())
            && completion
                .1
                .get("output")
                .and_then(|output| output.get("task_logits_f32"))
                == Some(&accepted_logits),
        "accepted V head differs from the original physically completed task output",
    )?;
    if frozen.is_some() {
        check(
            completion
                .1
                .get("physical_fresh")
                .and_then(|value| value.as_bool())
                == Some(true)
                && completion.1.get("accepted_seed_sha256") == Some(&serde_json::Value::Null),
            "controlled frozen action comparisons require actual Fresh V inference",
        )?;
    }
    check(
        evidence.iter().any(|(sha, raw)| {
            raw.get("schema").and_then(|v| v.as_str())
                == Some("rz-pals-generation-native-retirement-evidence/1")
                && raw.get("completion_sha256").and_then(|v| v.as_str())
                    == Some(completion.0.as_str())
                && raw.get("submission_sha256").and_then(|v| v.as_str())
                    == Some(submission_sha.as_str())
                && raw.get("input_sha256").and_then(|v| v.as_str())
                    == Some(record.input_sha256.as_str())
                && raw.get("ticket") == Some(&ticket)
                && raw.get("accepted").and_then(|v| v.as_bool()) == Some(true)
                && generation_digest("rz-pals-generation-native-retirement-evidence/1", raw)
                    .ok()
                    .as_ref()
                    == Some(sha)
        }),
        "V action requires original accepted native retirement",
    )?;
    let action = match semantic.id().action {
        AdaptiveValidatorAction::DefendResponse => GenerationTaskV3::DefendResponse,
        AdaptiveValidatorAction::AttackRepair => GenerationTaskV3::AttackRepair,
        AdaptiveValidatorAction::WidenResponses => GenerationTaskV3::WidenResponses,
        AdaptiveValidatorAction::LowerSelectivity => GenerationTaskV3::LowerSelectivity,
        AdaptiveValidatorAction::ResumeTask => GenerationTaskV3::ResumeTask,
        AdaptiveValidatorAction::CrossProfileRecheck => GenerationTaskV3::CrossProfileRecheck,
        AdaptiveValidatorAction::Defer => GenerationTaskV3::Defer,
    };
    check(
        record.input.action_support_mask[action.index()] && semantic.witnesses().len() <= 64,
        "unsupported V action or witness capacity",
    )?;
    let observed_sequence = chronology.label_after(&record.input)?;
    let execution_id = format!(
        "validator-action-{}-{}-{}",
        source.id.epoch.0,
        source.id.sequence,
        action.index()
    );
    let wall_cost_ns = u64::try_from(completed.full_action_wall_cost().as_nanos())
        .map_err(|_| invalid("full action wall cost overflow"))?;
    // The ordinary selfplay adapter has no preregistered frozen action group.
    // Even an exclusive root fence alone cannot grant a controlled comparison
    // cost. The isolated-action adapter binds that fence to frozen PC pins.
    let (training_wall_cost_ns, wall_cost_scope, frozen_context, frozen_epoch, frozen_weights) =
        match frozen {
            None => (
                None,
                completed.full_action_wall_cost_scope(),
                None,
                None,
                None,
            ),
            Some(registration) => {
                registration.verify_actual_completion(record, &completed)?;
                registration.verify_native_pins(record, evidence)?;
                let cost = completed.training_wall_cost().ok_or_else(|| {
                    invalid("registered action lacks actual exclusive fence cost")
                })?;
                (
                    Some(
                        u64::try_from(cost.as_nanos())
                            .map_err(|_| invalid("exclusive action cost overflow"))?,
                    ),
                    completed
                        .exclusive_action_cost()
                        .expect("training cost requires exclusive proof")
                        .cost_scope(),
                    Some(registration.frozen_context_sha256().to_owned()),
                    Some(registration.frozen_pc_epoch()),
                    Some(registration.frozen_pc_weights_sha256().to_owned()),
                )
            }
        };
    let mut receipts = BTreeMap::new();
    let mut new_witnesses = Vec::new();
    let mut raw_witnesses = Vec::new();
    let mut identities = BTreeSet::new();
    for actual in semantic.witnesses() {
        let identity = hex(&actual.witness_identity().0);
        check(
            actual.action() == semantic.id()
                && identities.insert(identity.clone())
                && actual.observation_sequence() >= semantic.input_observation_sequence()
                && actual.line().first().is_some_and(|first| {
                    Move16::pack(*first).ok().map(|mv| mv.bits())
                        == Some(semantic.original_first_move().bits())
                }),
            "V witness original first move/source ordinal/semantic identity mismatch",
        )?;
        let kind = match actual.kind() {
            ActionWitnessKind::Counter => GenerationRecordKindV3::Counter,
            ActionWitnessKind::Repair => GenerationRecordKindV3::Repair,
        };
        let mut witness = GenerationWitnessV3 {
            witness_id: format!("witness-{identity}"),
            kind,
            observed_sequence,
            raw_evidence_sha256: "0".repeat(64),
        };
        let mut content = serde_json::to_value(&witness).map_err(|e| invalid(e.to_string()))?;
        content
            .as_object_mut()
            .ok_or_else(|| invalid("witness shape"))?
            .remove("raw_evidence_sha256");
        let raw = serde_json::json!({"schema":"rz-pals-generation-witness-evidence/1","input_sha256":record.input_sha256,"witness":content,"rules_validated":true});
        witness.raw_evidence_sha256 =
            generation_digest("rz-pals-generation-witness-evidence/1", &raw)?;
        receipts.insert(witness.raw_evidence_sha256.clone(), raw);
        raw_witnesses.push(serde_json::json!({"witness_id":witness.witness_id,"kind":kind,"observed_sequence":observed_sequence,
            "store_observation_sequence":actual.observation_sequence(),"store_generation":actual.store_generation(),"packed_line":pack(actual.line())?,
            "semantic_identity_sha256":identity,"store_commit_sha256":hex(&actual.raw_evidence().0)}));
        new_witnesses.push(witness);
    }
    let mut outcome = GenerationActionOutcomeV3 {
        action,
        execution_id: execution_id.clone(),
        completed: true,
        // Shared-root completion latency is raw only. The manager's private
        // exclusive-action fence is the sole issuer of a known training cost.
        wall_cost_ns: training_wall_cost_ns,
        new_witnesses,
        causal_receipt_sha256: "0".repeat(64),
    };
    let mut content = serde_json::to_value(&outcome).map_err(|e| invalid(e.to_string()))?;
    content
        .as_object_mut()
        .ok_or_else(|| invalid("action outcome shape"))?
        .remove("causal_receipt_sha256");
    let raw = serde_json::json!({"schema":"rz-pals-generation-causal-evidence/1","input_sha256":record.input_sha256,"outcome":content});
    outcome.causal_receipt_sha256 =
        generation_digest("rz-pals-generation-causal-evidence/1", &raw)?;
    receipts.insert(outcome.causal_receipt_sha256.clone(), raw);
    let authority = serde_json::json!({"schema":"rz-pals-generation-action-authority-evidence/1","input_sha256":record.input_sha256,
        "generation_view_sha256":record.input.view_sha256,"ticket":ticket,"action":action,"execution_id":execution_id,
        "native_view_content_sha256":native_view_content,"actual_native_input_sha256":hex(&completed.actual_native_input().0),
        "action_support_mask":semantic.action_support_mask(),"input_store_observation_sequence":semantic.input_observation_sequence(),
        "original_first_move":semantic.original_first_move().bits(),"wall_cost_ns":wall_cost_ns,
        "training_wall_cost_ns":outcome.wall_cost_ns,"wall_cost_scope":wall_cost_scope,
        "measured_wall_cost_scope":completed.full_action_wall_cost_scope(),
        "frozen_context_sha256":frozen_context,
        "frozen_pc_epoch":frozen_epoch,
        "frozen_pc_weights_sha256":frozen_weights,
        "root_native_cpu_spent":completed.root_native_cpu_spent(),"root_completed_nn_inputs":completed.root_completed_nn_inputs(),
        "executions":semantic.executions().iter().map(|id|serde_json::json!({"epoch":id.epoch.0,"sequence":id.sequence})).collect::<Vec<_>>(),
        "role_requests":semantic.role_requests().iter().map(|id|serde_json::json!({"epoch":id.epoch.0,"sequence":id.sequence})).collect::<Vec<_>>(),
        "witnesses":raw_witnesses,"causal_receipt_sha256":outcome.causal_receipt_sha256});
    let sha = generation_digest("rz-pals-generation-action-authority-evidence/1", &authority)?;
    check(
        serde_json::to_vec(&authority)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            < GENERATION_MAX_ROW_BYTES,
        "actual action authority receipt row capacity",
    )?;
    receipts.insert(sha, authority);
    evidence.extend(receipts);
    Ok(outcome)
}
