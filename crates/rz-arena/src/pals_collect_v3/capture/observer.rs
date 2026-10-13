//! Optional collection-only observer. The default engine has no JSON capture,
//! input clone, archive write or training dependency on this path.
use super::*;
use rz_contracts::RequestId;
use rz_contracts::pals::{RoleOutput, RolePayload};
use rz_eval::pals_onnx::VirtualReviewCudaPlacementWitness;
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_uci::pals_native::ticketed::{
    NativeTicketQuestion, TicketExecutionEvidence, TicketPreparation,
    VirtualReviewCollectionObserver, VirtualReviewPhysicalInvocationEvidence,
    VirtualReviewPreparedTicket, exact_review_anchor,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};

struct Registered {
    ticket: RoleTicketEnvelope,
    context: GenerationCaptureContextV3,
    submitted: bool,
    completed: bool,
    retired: bool,
    input_sha256: Option<String>,
    submission_sha256: Option<String>,
    completion_sha256: Option<String>,
}
type Registrations = Arc<Mutex<HashMap<RequestId, Registered>>>;
pub struct GenerationCaptureRegistrationV3 {
    registrations: Registrations,
}
pub struct NativeGenerationObserverV3 {
    registrations: Registrations,
    inputs: mpsc::SyncSender<GenerationNativeCaptureEventV3>,
    terminals: mpsc::SyncSender<GenerationNativeCaptureEventV3>,
    audits: mpsc::SyncSender<GenerationNativeCaptureEventV3>,
    runtime_audited: bool,
}
pub struct GenerationCaptureReceiverV3 {
    registrations: Registrations,
    inputs: mpsc::Receiver<GenerationNativeCaptureEventV3>,
    terminals: mpsc::Receiver<GenerationNativeCaptureEventV3>,
    audits: mpsc::Receiver<GenerationNativeCaptureEventV3>,
}
#[derive(Clone, Debug)]
pub enum GenerationNativeCaptureEventV3 {
    RuntimeAudited {
        receipt: serde_json::Value,
    },
    Submitted {
        ticket: RoleTicketEnvelope,
        input: GenerationInputV3,
        input_sha256: String,
        receipt: serde_json::Value,
    },
    Completed {
        ticket: RoleTicketEnvelope,
        admissible_actual_nn: bool,
        receipt: serde_json::Value,
    },
    Retired {
        ticket: RoleTicketEnvelope,
        accepted: bool,
        receipt: serde_json::Value,
    },
}
pub fn generation_capture_channel_v3() -> (
    GenerationCaptureRegistrationV3,
    NativeGenerationObserverV3,
    GenerationCaptureReceiverV3,
) {
    let registrations = Arc::new(Mutex::new(HashMap::new()));
    let (input_tx, input_rx) = mpsc::sync_channel(16);
    let (terminal_tx, terminal_rx) = mpsc::sync_channel(32);
    let (audit_tx, audit_rx) = mpsc::sync_channel(1);
    (
        GenerationCaptureRegistrationV3 {
            registrations: registrations.clone(),
        },
        NativeGenerationObserverV3 {
            registrations: registrations.clone(),
            inputs: input_tx,
            terminals: terminal_tx,
            audits: audit_tx,
            runtime_audited: false,
        },
        GenerationCaptureReceiverV3 {
            registrations,
            inputs: input_rx,
            terminals: terminal_rx,
            audits: audit_rx,
        },
    )
}
impl GenerationCaptureRegistrationV3 {
    /// Called before actual ticket execution. This only reserves capture space;
    /// native physical completion and manager acceptance remain separate facts.
    pub fn register(
        &self,
        ticket: RoleTicketEnvelope,
        context: GenerationCaptureContextV3,
    ) -> Result<(), ArenaError> {
        for text in [
            &context.game_id,
            &context.opening_id,
            &context.line_genealogy_id,
        ] {
            check(generation_identifier(text), "capture context identifier")?;
        }
        context.work_scope.identifiers(&ticket)?;
        check(
            context.actual_history.len() <= 16_384,
            "capture actual history bound",
        )?;
        let root = exact_rules_anchor(&context.actual_position)?;
        check(
            root.rules_state_sha256 == hex(&ticket.view.root_state.semantic.0),
            "capture registration root Rules identity",
        )?;
        let mut map = self
            .registrations
            .lock()
            .map_err(|_| invalid("capture registry poisoned"))?;
        check(
            map.len() < 16 && !map.contains_key(&ticket.id),
            "capture registration capacity/duplicate",
        )?;
        map.insert(
            ticket.id,
            Registered {
                ticket,
                context,
                submitted: false,
                completed: false,
                retired: false,
                input_sha256: None,
                submission_sha256: None,
                completion_sha256: None,
            },
        );
        Ok(())
    }
}
impl GenerationCaptureReceiverV3 {
    /// Input precedes terminal consumption for the same ticket, even when the
    /// native worker completed before the manager next polled.
    pub fn try_event(&self) -> Result<Option<GenerationNativeCaptureEventV3>, ArenaError> {
        match self.inputs.try_recv() {
            Ok(e) => return Ok(Some(e)),
            Err(mpsc::TryRecvError::Disconnected) => {}
            Err(mpsc::TryRecvError::Empty) => {}
        }
        match self.terminals.try_recv() {
            Ok(e) => Ok(Some(e)),
            Err(mpsc::TryRecvError::Empty) => Ok(self.audits.try_recv().ok()),
            Err(mpsc::TryRecvError::Disconnected) => self.audits.try_recv().map(Some).or(Ok(None)), // disconnection alone never proves a fence
        }
    }
    /// Keep bounded reservations until the receiver persisted every terminal.
    /// Retirement alone cannot free capacity while reliable receipts pile up.
    pub fn acknowledge_persisted_retirement(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), ArenaError> {
        let mut map = self
            .registrations
            .lock()
            .map_err(|_| invalid("capture registry poisoned"))?;
        let row = map
            .get(&ticket.id)
            .ok_or_else(|| invalid("unknown collected ticket"))?;
        check(
            row.ticket == *ticket && row.retired,
            "capture retirement identity/order",
        )?;
        map.remove(&ticket.id);
        Ok(())
    }
}
fn failure(error: ArenaError) -> RoleFailure {
    RoleFailure::Backend(format!("collection capture: {error}"))
}
impl NativeGenerationObserverV3 {
    fn send_terminal(&self, event: GenerationNativeCaptureEventV3) -> Result<(), RoleFailure> {
        self.terminals.try_send(event).map_err(|_| {
            RoleFailure::Backend("bounded reliable capture receipt queue unavailable".into())
        })
    }
}
impl VirtualReviewCollectionObserver for NativeGenerationObserverV3 {
    fn runtime_audited(
        &mut self,
        witness: &VirtualReviewCudaPlacementWitness,
    ) -> Result<(), RoleFailure> {
        if self.runtime_audited {
            return Err(RoleFailure::InvalidOutput);
        }
        let witness_json =
            serde_json::to_string(witness).map_err(|e| RoleFailure::Backend(e.to_string()))?;
        if witness_json.len() > GENERATION_MAX_ROW_BYTES {
            return Err(RoleFailure::InvalidOutput);
        }
        let receipt = serde_json::json!({"schema":"rz-pals-generation-runtime-audit-evidence/1","witness_json":witness_json});
        if serde_json::to_vec(&receipt)
            .map_err(|e| RoleFailure::Backend(e.to_string()))?
            .len()
            + 1
            > GENERATION_MAX_ROW_BYTES
        {
            return Err(RoleFailure::InvalidOutput);
        }
        self.audits
            .try_send(GenerationNativeCaptureEventV3::RuntimeAudited { receipt })
            .map_err(|_| {
                RoleFailure::Backend("bounded runtime audit capture queue unavailable".into())
            })?;
        self.runtime_audited = true;
        Ok(())
    }
    fn submitted(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        prepared: &VirtualReviewPreparedTicket,
    ) -> Result<(), RoleFailure> {
        let context = {
            let map = self
                .registrations
                .lock()
                .map_err(|_| RoleFailure::InvalidOutput)?;
            let row = map.get(&ticket.id).ok_or(RoleFailure::InvalidOutput)?;
            if row.ticket != *ticket
                || row.submitted
                || prepared.representation != ticket.representation
            {
                return Err(RoleFailure::InvalidOutput);
            }
            row.context.clone()
        };
        for (anchor, encoded) in question.view.anchors().iter().zip(&prepared.input.anchors) {
            let exact =
                Position::from_snapshot(&anchor.snapshot, rz_position::PositionLimits::default())
                    .map_err(|_| RoleFailure::InvalidOutput)?;
            if exact_review_anchor(&exact)? != *encoded {
                return Err(RoleFailure::InvalidOutput);
            }
        }
        let input = capture_submitted_review_v3(ticket, &question.view, &prepared.input, &context)
            .map_err(failure)?;
        let input_sha256 = input.sha256().map_err(|e| failure(e.into()))?;
        let receipt = serde_json::json!({"schema":"rz-pals-generation-submission-evidence/1","input_sha256":input_sha256,"ticket":generation_role_ticket_json_v3(ticket),
            "generation_view_sha256":input.view_sha256,"game_id":input.game_id,"opening_id":input.opening_id,"role":input.role,"model_epoch":input.model_epoch,"context_epoch":input.context_epoch,"capture_sequence":input.capture_sequence,
            "worker_id":input.worker_id,"job_id":input.job_id,"slice_id":input.slice_id,"actual_root_anchor":input.selected_view.anchors[0],"query_anchor":input.rules_anchor,
            "source_view_content_sha256":hex(&question.view.digest()),"native_endpoint_history_sha256":hex(&ticket.representation.history.0),
            "source_view_seal_sha256":hex(&question.view.seal_digest()),"physical_nn_admitted":false,"native_input_sha256":hex(&prepared.input.prepare().map_err(|_|RoleFailure::InvalidOutput)?.input_digest(prepared.input.role)),
            "native_model_epoch_sha256":hex(&prepared.identity.model_epoch),"native_backend_sha256":hex(&prepared.identity.backend),"native_anchor_history_sha256":prepared.identity.anchor_history_seals.iter().map(|s|hex(s)).collect::<Vec<_>>()});
        let submission_sha256 =
            generation_digest("rz-pals-generation-submission-evidence/1", &receipt)
                .map_err(|e| failure(e.into()))?;
        self.inputs
            .try_send(GenerationNativeCaptureEventV3::Submitted {
                ticket: *ticket,
                input,
                input_sha256: input_sha256.clone(),
                receipt,
            })
            .map_err(|_| RoleFailure::Backend("bounded capture input queue unavailable".into()))?;
        let mut map = self
            .registrations
            .lock()
            .map_err(|_| RoleFailure::InvalidOutput)?;
        let registered = map.get_mut(&ticket.id).ok_or(RoleFailure::InvalidOutput)?;
        registered.submitted = true;
        registered.input_sha256 = Some(input_sha256);
        registered.submission_sha256 = Some(submission_sha256);
        Ok(())
    }
    fn completed(
        &mut self,
        ticket: &RoleTicketEnvelope,
        physical: PhysicalRoleCompletion,
        output: Option<&RoleOutput>,
        evidence: Option<TicketExecutionEvidence>,
        invocation: Option<VirtualReviewPhysicalInvocationEvidence>,
    ) -> Result<(), RoleFailure> {
        let (input_sha256, submission_sha256) = {
            let map = self
                .registrations
                .lock()
                .map_err(|_| RoleFailure::InvalidOutput)?;
            let row = map.get(&ticket.id).ok_or(RoleFailure::InvalidOutput)?;
            if row.ticket != *ticket || !row.submitted || row.completed {
                return Err(RoleFailure::InvalidOutput);
            }
            (
                row.input_sha256.clone().ok_or(RoleFailure::InvalidOutput)?,
                row.submission_sha256
                    .clone()
                    .ok_or(RoleFailure::InvalidOutput)?,
            )
        };
        if output.is_some_and(|o| {
            o.id != ticket.id || o.authority != ticket.authority || o.key != ticket.representation
        }) {
            return Err(RoleFailure::InvalidOutput);
        }
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|v| format!("{:08x}", v.to_bits()))
                .collect::<Vec<_>>()
        };
        let payload=output.map(|o|match &o.payload{RolePayload::Proposal{candidate_logits,wdl}=>serde_json::json!({"kind":"proposal","candidate_logits_f32":bits(candidate_logits),"wdl_f32":bits(&wdl.probabilities())}),RolePayload::Counterexample{candidate_logits,divergence_logits,wdl}=>serde_json::json!({"kind":"counterexample","candidate_logits_f32":bits(candidate_logits),"divergence_logits_f32":bits(divergence_logits),"wdl_f32":bits(&wdl.probabilities())}),RolePayload::TaskRanking{task_logits}=>serde_json::json!({"kind":"task_ranking","task_logits_f32":bits(task_logits)})});
        let admissible_actual_nn = physical == PhysicalRoleCompletion::Confirmed
            && output.is_some()
            && evidence.is_some_and(|e| {
                e.measurement_complete
                    && e.completed_nn_inputs.is_some_and(|n| n >= 1)
                    && e.execution.is_some()
            })
            && invocation.is_some_and(|i| {
                evidence.is_some_and(|e| {
                    e.physical_fresh == Some(i.physical_fresh)
                        && e.execution.is_some_and(|x| x.epoch == ticket.id.epoch)
                }) && (i.physical_fresh && i.accepted_seed_digest.is_none()
                    || !i.physical_fresh && i.accepted_seed_digest.is_some())
            });
        let receipt = serde_json::json!({"schema":"rz-pals-generation-native-completion-evidence/1","input_sha256":input_sha256,"submission_sha256":submission_sha256,"ticket":generation_role_ticket_json_v3(ticket),"physical_completion":if physical==PhysicalRoleCompletion::Confirmed{"confirmed"}else{"unknown"},
            "execution":evidence.and_then(|e|e.execution).map(|e|serde_json::json!({"epoch":e.epoch.0,"sequence":e.sequence})),"completed_nn_inputs":evidence.and_then(|e|e.completed_nn_inputs),"measurement_complete":evidence.map(|e|e.measurement_complete),"physical_fresh":evidence.and_then(|e|e.physical_fresh),
            "invocation_key_sha256":invocation.map(|i|hex(&i.invocation_key.0)),"accepted_seed_sha256":invocation.and_then(|i|i.accepted_seed_digest).map(|d|hex(&d.0)),"output":payload,"admissible_actual_nn":admissible_actual_nn});
        let completion_sha256 =
            generation_digest("rz-pals-generation-native-completion-evidence/1", &receipt)
                .map_err(|e| failure(e.into()))?;
        self.send_terminal(GenerationNativeCaptureEventV3::Completed {
            ticket: *ticket,
            admissible_actual_nn,
            receipt,
        })?;
        let mut map = self
            .registrations
            .lock()
            .map_err(|_| RoleFailure::InvalidOutput)?;
        let row = map.get_mut(&ticket.id).ok_or(RoleFailure::InvalidOutput)?;
        row.completed = true;
        row.completion_sha256 = Some(completion_sha256);
        Ok(())
    }
    fn retired(&mut self, ticket: &RoleTicketEnvelope, accepted: bool) -> Result<(), RoleFailure> {
        let (input_sha256, submission_sha256, completion_sha256) = {
            let map = self
                .registrations
                .lock()
                .map_err(|_| RoleFailure::InvalidOutput)?;
            let row = map.get(&ticket.id).ok_or(RoleFailure::InvalidOutput)?;
            if row.ticket != *ticket || row.retired || (accepted && !row.completed) {
                return Err(RoleFailure::InvalidOutput);
            }
            (
                row.input_sha256.clone(),
                row.submission_sha256.clone(),
                row.completion_sha256.clone(),
            )
        };
        let receipt = serde_json::json!({"schema":"rz-pals-generation-native-retirement-evidence/1","input_sha256":input_sha256,"submission_sha256":submission_sha256,"completion_sha256":completion_sha256,"ticket":generation_role_ticket_json_v3(ticket),"accepted":accepted});
        self.send_terminal(GenerationNativeCaptureEventV3::Retired {
            ticket: *ticket,
            accepted,
            receipt,
        })?;
        self.registrations
            .lock()
            .map_err(|_| RoleFailure::InvalidOutput)?
            .get_mut(&ticket.id)
            .ok_or(RoleFailure::InvalidOutput)?
            .retired = true;
        Ok(())
    }
}

/// A receiver joins three independent phases. Only original submitted input +
/// observed actual NN completion + manager/backend acceptance publish a row.
pub struct GenerationCaptureJoinV3 {
    pending: HashMap<RequestId, (RoleTicketEnvelope, Option<GenerationInputV3>, Option<bool>)>,
    receipts: Vec<serde_json::Value>,
}
impl GenerationCaptureJoinV3 {
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
    pub fn new() -> Self {
        Self {
            pending: HashMap::new(),
            receipts: vec![],
        }
    }
    pub fn apply(
        &mut self,
        event: GenerationNativeCaptureEventV3,
    ) -> Result<Option<GenerationRecordV3>, ArenaError> {
        check(
            self.receipts.len() < 64,
            "capture receipt persistence must drain before admitting more tickets",
        )?;
        match event {
            GenerationNativeCaptureEventV3::RuntimeAudited { receipt } => {
                self.receipts.push(receipt);
                Ok(None)
            }
            GenerationNativeCaptureEventV3::Submitted {
                ticket,
                input,
                input_sha256,
                receipt,
            } => {
                check(
                    self.pending.len() < 16
                        && !self.pending.contains_key(&ticket.id)
                        && input.sha256()? == input_sha256,
                    "capture join submitted identity/capacity",
                )?;
                self.receipts.push(receipt);
                self.pending.insert(ticket.id, (ticket, Some(input), None));
                Ok(None)
            }
            GenerationNativeCaptureEventV3::Completed {
                ticket,
                admissible_actual_nn,
                receipt,
            } => {
                let row = self
                    .pending
                    .get_mut(&ticket.id)
                    .ok_or_else(|| invalid("completion before capture"))?;
                check(
                    row.0 == ticket && row.2.is_none(),
                    "completion belongs to another submitted ticket",
                )?;
                row.2 = Some(admissible_actual_nn);
                self.receipts.push(receipt);
                Ok(None)
            }
            GenerationNativeCaptureEventV3::Retired {
                ticket,
                accepted,
                receipt,
            } => {
                if !self.pending.contains_key(&ticket.id) {
                    check(
                        !accepted
                            && receipt["input_sha256"].is_null()
                            && receipt["submission_sha256"].is_null()
                            && receipt["completion_sha256"].is_null(),
                        "unknown/accepted retirement before capture",
                    )?;
                    self.receipts.push(receipt);
                    return Ok(None);
                }
                let row = self.pending.get(&ticket.id).unwrap();
                check(
                    row.0 == ticket,
                    "retirement belongs to another submitted ticket",
                )?;
                let row = self.pending.remove(&ticket.id).unwrap();
                self.receipts.push(receipt);
                if !accepted || row.2 != Some(true) {
                    return Ok(None);
                }
                let input = row
                    .1
                    .ok_or_else(|| invalid("missing submitted capture input"))?;
                let record = GenerationRecordV3 {
                    schema: GENERATION_DATA_DOMAIN.into(),
                    source_kind: "actual_selfplay".into(),
                    eligibility: "actual_generation_candidate".into(),
                    input_sha256: input.sha256()?,
                    input,
                    future_targets: GenerationTargetsV3::default(),
                };
                record.validate()?;
                Ok(Some(record))
            }
        }
    }
    pub fn take_receipts(&mut self) -> Vec<serde_json::Value> {
        std::mem::take(&mut self.receipts)
    }
}
impl Default for GenerationCaptureJoinV3 {
    fn default() -> Self {
        Self::new()
    }
}
