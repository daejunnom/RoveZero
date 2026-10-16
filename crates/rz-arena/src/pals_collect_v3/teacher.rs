//! Actual CPU_T receipts. Native selected inputs are immutable; a later UCI
//! opinion is explicitly conditional on the exact Rules endpoint, not proof
//! that Stockfish read the selected neural review context.
use super::*;
use rz_search::external_cpu::worker::{
    ExternalJob, ExternalTeacherDispatch, ExternalWorkerCompletion, external_evaluation_namespace,
};
use rz_uci::pals_adaptive::{
    AdaptiveExternalTeacherEvent, AdaptiveExternalTeacherObserver, AdaptiveManagerError,
};
use rz_uci::pals_adaptive_bootstrap::external::ExternalBootstrapAdmissionV5;
use std::sync::{Arc, Mutex, mpsc};

const MAX_TEACHER_EVENTS: usize = 128;
const MAX_TEACHER_JOBS: usize = 64;

#[derive(Clone)]
struct TeacherPins {
    binary: String,
    profile: String,
    namespace: rz_contracts::Digest,
}
struct TeacherEvent {
    sequence: u64,
    pins: TeacherPins,
    event: AdaptiveExternalTeacherEvent,
}
struct SubmittedTeacher {
    sequence: u64,
    pins: TeacherPins,
    original: ExternalJob,
    dispatch: ExternalTeacherDispatch,
    submission_sha256: String,
}

/// Install this observer only after the actual pinned external profile was
/// admitted. Neither a text profile SHA nor an arbitrary public report can
/// construct an eligible teacher binding.
#[derive(Clone)]
pub struct GenerationTeacherObserverV3 {
    registered: Arc<Mutex<Option<TeacherPins>>>,
    expected: Arc<Mutex<Option<(String, String)>>>,
    hook: AdaptiveExternalTeacherObserver,
}
impl GenerationTeacherObserverV3 {
    pub(super) fn expect_source(&self, binary: &str, profile: &str) -> Result<(), ArenaError> {
        check(
            generation_sha(binary) && generation_sha(profile),
            "collection registered teacher source",
        )?;
        let mut expected = self
            .expected
            .lock()
            .map_err(|_| invalid("expected teacher pins poisoned"))?;
        check(
            expected.is_none(),
            "duplicate expected teacher registration",
        )?;
        *expected = Some((binary.into(), profile.into()));
        Ok(())
    }
    pub fn register_source(
        &self,
        admitted: &ExternalBootstrapAdmissionV5,
    ) -> Result<(), ArenaError> {
        check(
            admitted.selection().execution_use
                == rz_search::external_cpu::CpuExecutionUse::Collection
                && admitted.selection().cpu_t == rz_search::external_cpu::CpuEndpoint::ExternalUci,
            "teacher requires admitted Collection CPU_T endpoint",
        )?;
        let namespace = external_evaluation_namespace(&admitted.config().identity)
            .map_err(|e| invalid(format!("teacher namespace: {e:?}")))?;
        let pins = TeacherPins {
            binary: admitted.relocation().binary_sha256.clone(),
            profile: admitted.relocation().profile_file_sha256.clone(),
            namespace,
        };
        check(
            generation_sha(&pins.binary) && generation_sha(&pins.profile),
            "teacher profile pins",
        )?;
        let expected = self
            .expected
            .lock()
            .map_err(|_| invalid("expected teacher pins poisoned"))?;
        check(
            expected.as_ref().is_some_and(|(binary, profile)| {
                *binary == pins.binary && *profile == pins.profile
            }),
            "admitted teacher differs from registered generation producer",
        )?;
        let mut registered = self
            .registered
            .lock()
            .map_err(|_| invalid("teacher registration poisoned"))?;
        check(
            registered.is_none(),
            "duplicate teacher source registration",
        )?;
        *registered = Some(pins);
        Ok(())
    }
    pub fn observer(&self) -> AdaptiveExternalTeacherObserver {
        Arc::clone(&self.hook)
    }
}

pub(super) struct GenerationTeacherReceiverV3 {
    receiver: mpsc::Receiver<TeacherEvent>,
    pending: BTreeMap<(u64, u64), SubmittedTeacher>,
    terminal_seen: BTreeSet<(u64, u64)>,
    bindings: TeacherBindingRegistryV3,
}
pub(super) fn teacher_channel_v3(
    chronology: Arc<Mutex<GenerationChronologyV3>>,
) -> (GenerationTeacherObserverV3, GenerationTeacherReceiverV3) {
    let registered: Arc<Mutex<Option<TeacherPins>>> = Arc::new(Mutex::new(None));
    let expected = Arc::new(Mutex::new(None));
    let hook_registered = Arc::clone(&registered);
    let (sender, receiver) = mpsc::sync_channel(MAX_TEACHER_EVENTS);
    let hook: AdaptiveExternalTeacherObserver = Arc::new(move |event| {
        let pins = hook_registered
            .lock()
            .map_err(|_| AdaptiveManagerError::Invalid("teacher source registration poisoned"))?
            .clone()
            .ok_or(AdaptiveManagerError::Invalid("teacher source not admitted"))?;
        let dispatch = match &event {
            AdaptiveExternalTeacherEvent::Submitted { dispatch, .. }
            | AdaptiveExternalTeacherEvent::Completed { dispatch, .. } => dispatch,
        };
        if dispatch.evaluation_namespace() != pins.namespace {
            return Err(AdaptiveManagerError::Invalid(
                "teacher actual namespace differs from pinned admission",
            ));
        }
        let sequence = chronology
            .lock()
            .map_err(|_| AdaptiveManagerError::Invalid("teacher chronology poisoned"))?
            .next_event_sequence()
            .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
        sender
            .try_send(TeacherEvent {
                sequence,
                pins,
                event,
            })
            .map_err(|_| AdaptiveManagerError::Capacity("reliable teacher receipt queue"))
    });
    (
        GenerationTeacherObserverV3 {
            registered,
            expected,
            hook,
        },
        GenerationTeacherReceiverV3 {
            receiver,
            pending: BTreeMap::new(),
            terminal_seen: BTreeSet::new(),
            bindings: TeacherBindingRegistryV3::default(),
        },
    )
}
impl GenerationTeacherReceiverV3 {
    pub(super) fn pending_count(&self) -> usize {
        self.pending.len()
    }
    pub(super) fn reset_root(&mut self) -> Result<(), ArenaError> {
        check(
            self.pending.is_empty(),
            "teacher root changed before terminal receipts",
        )?;
        self.terminal_seen.clear();
        self.bindings = TeacherBindingRegistryV3::default();
        Ok(())
    }
    pub(super) fn poll(
        &mut self,
        records: &mut BTreeMap<String, GenerationRecordV3>,
        evidence: &mut BTreeMap<String, serde_json::Value>,
    ) -> Result<bool, ArenaError> {
        let mut changed = false;
        while let Ok(event) = self.receiver.try_recv() {
            changed = true;
            match event.event {
                AdaptiveExternalTeacherEvent::Submitted {
                    input,
                    original,
                    dispatch,
                } => {
                    let key = (
                        dispatch.grant().execution.epoch.0,
                        dispatch.grant().execution.sequence,
                    );
                    check(
                        self.pending.len() < MAX_TEACHER_JOBS
                            && !self.pending.contains_key(&key)
                            && !self.terminal_seen.contains(&key)
                            && dispatch.matches_job(&original)
                            && input.view_seal.view_digest.0 == input.view.seal_digest()
                            && original.envelope.id.authority == input.view_seal.authority,
                        "teacher submission identity/capacity",
                    )?;
                    let anchor = exact_rules_anchor(&original.position)?;
                    let process = dispatch.process_identity();
                    let raw = serde_json::json!({"schema":"rz-pals-generation-teacher-submission-evidence/1",
                        "observed_sequence":event.sequence,"execution_epoch":key.0,"execution_sequence":key.1,
                        "job_sequence":dispatch.job().id.sequence,"root_generation":dispatch.job().id.authority.root.0,
                        "source_cpu_question_sha256":hex(&input.question_digest.0),"source_cpu_view_sha256":hex(&input.view.seal_digest()),
                        "teacher_question_sha256":hex(&dispatch.job().question.0),"evaluation_namespace":hex(&dispatch.evaluation_namespace().0),
                        "teacher_binary_sha256":event.pins.binary,"teacher_profile_sha256":event.pins.profile,
                        "root_restriction_sha256":hex(&dispatch.job().root_restriction.0),
                        "search_conditions_sha256":hex(&dispatch.job().search_conditions.0),
                        "root_moves":original.root_moves.as_ref().map(|moves|moves.iter().map(|m|m.to_string()).collect::<Vec<_>>()),
                        "process":{"pid":process.pid,"process_group":process.process_group,"proc_start_ticks":process.proc_start_ticks},
                        "uci_name":dispatch.observed_uci().name,"uci_author":dispatch.observed_uci().author,
                        "rules_anchor":anchor,"interpretation":"conditional_exact_endpoint_opinion; neural_review_not_read_by_teacher"});
                    let submission_sha256 = insert_teacher_raw(evidence, raw)?;
                    self.pending.insert(
                        key,
                        SubmittedTeacher {
                            sequence: event.sequence,
                            pins: event.pins,
                            original,
                            dispatch,
                            submission_sha256,
                        },
                    );
                }
                AdaptiveExternalTeacherEvent::Completed {
                    dispatch,
                    completion,
                    eligibility,
                } => {
                    let key = (
                        dispatch.grant().execution.epoch.0,
                        dispatch.grant().execution.sequence,
                    );
                    check(
                        !self.terminal_seen.contains(&key),
                        "duplicate teacher terminal",
                    )?;
                    let submitted = self
                        .pending
                        .remove(&key)
                        .ok_or_else(|| invalid("teacher terminal before actual submission"))?;
                    check(
                        submitted.dispatch.job() == dispatch.job()
                            && submitted.dispatch.grant() == dispatch.grant()
                            && submitted.dispatch.process_identity() == dispatch.process_identity()
                            && submitted.pins.namespace == event.pins.namespace
                            && event.sequence > submitted.sequence
                            && eligibility.matches_submission(&submitted.dispatch),
                        "teacher terminal changed original source",
                    )?;
                    self.terminal_seen.insert(key);
                    let completion_sha256 = insert_teacher_raw(
                        evidence,
                        completion_receipt(
                            &submitted,
                            &completion,
                            eligibility.target_eligible(),
                            event.sequence,
                        ),
                    )?;
                    if !eligibility.target_eligible()
                        || !completion.matches_dispatch(&submitted.dispatch)
                    {
                        continue;
                    }
                    let anchor = exact_rules_anchor(&submitted.original.position)?;
                    for record in records.values_mut().filter(|r| {
                        r.input.role != GenerationRoleV3::Validator
                            && matches!(
                                r.input.selected_view.query.purpose.as_str(),
                                "propose_policy" | "reply_policy" | "repair_policy"
                            )
                            && r.input.capture_sequence < submitted.sequence
                            && r.input.rules_anchor == anchor
                            && r.input.worker_id == format!("native-owner-{}", key.0)
                            && r.future_targets.policy.is_none()
                    }) {
                        let binding = self.bindings.bind_actual_submission(
                            record,
                            &submitted.original,
                            &submitted.dispatch,
                        )?;
                        if attach_completed_teacher_policy_v3(
                            record,
                            &binding,
                            &submitted.original,
                            &completion,
                            event.sequence,
                            &submitted.pins.binary,
                            &submitted.pins.profile,
                            evidence,
                        )? {
                            let head = record
                                .future_targets
                                .policy
                                .as_ref()
                                .ok_or_else(|| invalid("attached teacher policy missing"))?;
                            insert_teacher_raw(
                                evidence,
                                serde_json::json!({"schema":"rz-pals-generation-teacher-input-binding-evidence/1",
                                "input_sha256":record.input_sha256,"policy_head_evidence_sha256":head.raw_evidence_sha256,
                                "teacher_submission_sha256":submitted.submission_sha256,"teacher_completion_sha256":completion_sha256,
                                "rules_anchor":record.input.rules_anchor,"observed_sequence":event.sequence,
                                "interpretation":"conditional_exact_endpoint_opinion"}),
                            )?;
                        }
                    }
                }
            }
        }
        Ok(changed)
    }
}
fn insert_teacher_raw(
    evidence: &mut BTreeMap<String, serde_json::Value>,
    raw: serde_json::Value,
) -> Result<String, ArenaError> {
    check(evidence.len() < 4096, "teacher root raw evidence capacity")?;
    let schema = raw
        .get("schema")
        .and_then(|v| v.as_str())
        .ok_or_else(|| invalid("teacher raw schema"))?;
    let sha = generation_digest(schema, &raw)?;
    check(
        !evidence.contains_key(&sha),
        "duplicate teacher raw evidence",
    )?;
    evidence.insert(sha.clone(), raw);
    Ok(sha)
}
fn completion_receipt(
    submitted: &SubmittedTeacher,
    completion: &ExternalWorkerCompletion,
    target_eligible: bool,
    sequence: u64,
) -> serde_json::Value {
    let report = completion.report();
    let result = match &report.result {
        Ok(actual) => serde_json::json!({"identity":actual.report.identity,
            "uci_name":actual.report.observed_uci.name,"completion":format!("{:?}",actual.report.completion),
            "go_sequence":actual.report.request_id,"go_issue":format!("{:?}",actual.go_issue),
            "best_move_move16":actual.report.best_move.and_then(|movement|Move16::pack(movement).ok()).map(|movement|movement.bits()),
            "readiness_barrier_completed":actual.readiness_barrier_completed,"policy_target_known":actual.policy_target_known,
            "requested_multipv":actual.requested_multipv,"option_application_observed":actual.option_application_observed,
            "reported_nodes":actual.report.work.nodes,"opinions":actual.opinions.iter().map(|o| serde_json::json!({
                "rank":o.rank,"depth":o.depth,"seldepth":o.seldepth,"score":format!("{:?}",o.score),
                "bound":format!("{:?}",o.bound),"reported_wdl":o.reported_wdl,"pv_move16":pack(&o.pv).ok(),
                "pv":o.pv.iter().map(|m|m.to_string()).collect::<Vec<_>>()})).collect::<Vec<_>>() }),
        Err(error) => serde_json::json!({"error":format!("{error:?}")}),
    };
    serde_json::json!({"schema":"rz-pals-generation-teacher-completion-evidence/1",
        "observed_sequence":sequence,"submission_observed_sequence":submitted.sequence,
        "execution_epoch":report.grant.execution.epoch.0,"execution_sequence":report.grant.execution.sequence,
        "job_sequence":report.job.id.sequence,"root_generation":report.job.id.authority.root.0,
        "source_scope_joined":completion.matches_submission_scope(&submitted.dispatch),
        "successful_owner_join":completion.matches_dispatch(&submitted.dispatch),
        "original_question_target_eligible":target_eligible,
        "cancel_sequence":report.controls.cancel_sequence,
        "evaluation_namespace":hex(&completion.evaluation_namespace().0),
        "physical_process_identity":completion.process_identity().map(|p|serde_json::json!({"pid":p.pid,"process_group":p.process_group,"proc_start_ticks":p.proc_start_ticks})),
        "teacher_binary_sha256":submitted.pins.binary,"teacher_profile_sha256":submitted.pins.profile,
        "exact_resume_supported":report.exact_resume_supported,"go_issue":format!("{:?}",report.go_issue),"result":result})
}
