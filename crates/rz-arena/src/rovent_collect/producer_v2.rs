//! submitted-input/1 전용 원 native 수집 소유자다. 원 ticket/완료/수락 처리는
//! 기존 유한 owner 경로를 공유하고 형식별 입력과 raw/3 봉인은 별도로 유지한다.
use super::*;
use rz_experiments::rovent_corpus_v2::{
    RoventCorpusManifestV2, RoventTargetsV2, RoventTrainingRecordV2,
};
use rz_experiments::rovent_submitted_input::{
    ROVENT_NATIVE_RAW_V3_DOMAIN, ROVENT_TRAINING_RECORD_V2_DOMAIN,
};

pub struct RoventInputCaptureReservationV2 {
    original: RoventInputCaptureReservation,
}
pub struct RoventCapturedInputV2 {
    original: RoventCapturedInput,
    input: rz_experiments::rovent_submitted_input::RoventSubmittedInputV1,
}
impl RoventInputCaptureReservationV2 {
    pub fn sequence(&self) -> u64 {
        self.original.sequence()
    }
    pub fn capture(
        self,
        ticket: RoleTicketEnvelope,
        view: &ReviewView,
        input: &PalsVirtualReviewInput,
        identity: &ReviewExecutionIdentity,
    ) -> Result<RoventCapturedInputV2, ArenaError> {
        input
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        check(
            input
                .fresh_key(identity)
                .map_err(|error| invalid(error.to_string()))?
                .0
                == ticket.representation.input.0,
            "raw3 captured input differs from original native representation",
        )?;
        let (captured, review) =
            capture_rovent_submitted_input(&ticket, view, input, &self.original.context)?;
        let input_sha256 = captured.sha256()?;
        let replay =
            RoventSubmittedRulesReplay::capture(&self.original.context.actual_position, view)?;
        let raw = json!({"schema":ROVENT_NATIVE_RAW_V3_DOMAIN,"kind":"submitted","sequence":self.original.context.capture_sequence,
            "process_epoch":ticket.id.epoch.0,"game_generation":ticket.authority.game.0,"root_generation":ticket.authority.root.0,
            "request_sequence":ticket.id.sequence,"head":role_name(ticket.role),"source_sha256":self.original.source_sha256,
            "model_sha256":hex(ticket.representation.model),"encoding_sha256":hex(ticket.representation.encoding),
            "input_digest":hex(ticket.representation.input),"view_digest":hex(ticket.view.view_digest),
            "input_sha256":input_sha256,"ticket":ticket_json(&ticket),"rules_replay":replay,"submitted_review":review});
        check(
            serde_json::to_vec(&raw)
                .map_err(|error| invalid(error.to_string()))?
                .len()
                <= GENERATION_MAX_ROW_BYTES,
            "raw3 original submission exceeds bounded row cap before native dispatch",
        )?;
        // 이 내부 projection은 물리 owner 처리의 공통 scalar만 전달한다.
        // legacy input/record로 검증·직렬화·게시하는 경로에 전달하지 않는다.
        let common = captured.common_validation_projection();
        Ok(RoventCapturedInputV2 {
            input: captured,
            original: RoventCapturedInput {
                manifest_sha256: self.original.manifest_sha256,
                pending: Pending {
                    ticket,
                    input: common,
                    input_sha256,
                    lineage: self.original.lineage,
                    submitted: self.original.context.capture_sequence,
                    raw_submission: raw,
                    completed: None,
                },
            },
        })
    }
}
impl RoventCapturedInputV2 {
    pub fn submitted_input_seal(&self) -> RoventSubmittedInputSeal {
        self.original.submitted_input_seal()
    }
}

pub struct RoventCorpusProducerV2 {
    manifest: RoventCorpusManifestV2,
    original: RoventCorpusProducer,
    inputs: BTreeMap<(u64, u64), rz_experiments::rovent_submitted_input::RoventSubmittedInputV1>,
}
impl RoventCorpusProducerV2 {
    /// 원 C row를 보존한 caller가 실제 후속 cycle witness를 나중에 붙인다.
    /// 수집기 순번과 manager event 순번을 합치거나 과거 입력을 수정하지 않는다.
    pub fn attach_pc_divergence(
        &mut self,
        row: &mut RoventRetiredRecordV2,
        witness: &rz_search::pals::engine::pc_port::PcDivergenceWitness,
    ) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.original.manifest_sha256,
            "divergence row belongs to another original corpus owner",
        )?;
        self.validate_admission(&row.admission)?;
        let observed_sequence = self.original.next()?;
        divergence_owner::attach_actual_pc_divergence(row, witness, observed_sequence)
    }
    #[cfg(feature = "pals-collection-onnx")]
    pub(super) fn original_teacher_producer(&self) -> &RoventCorpusProducer {
        &self.original
    }
    pub fn new(manifest: RoventCorpusManifestV2) -> Result<Self, ArenaError> {
        let manifest_sha = manifest.sha256()?;
        let mut original = RoventCorpusProducer::new(manifest.common_registration_projection())?;
        original.manifest_sha256 = manifest_sha;
        original.native_domain = ROVENT_NATIVE_RAW_V3_DOMAIN;
        Ok(Self {
            manifest,
            original,
            inputs: BTreeMap::new(),
        })
    }
    pub fn manifest(&self) -> &RoventCorpusManifestV2 {
        &self.manifest
    }
    pub(super) fn validate_admission(
        &self,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<(), ArenaError> {
        self.original.validate_admission(admission)
    }
    pub fn incomplete_capture_count(&self) -> usize {
        self.original.incomplete_capture_count()
    }
    pub fn submitted_input_seal(
        &self,
        id: RequestId,
    ) -> Result<RoventSubmittedInputSeal, ArenaError> {
        self.original.submitted_input_seal(id)
    }
    pub fn reserve_input(
        &mut self,
        context: GenerationCaptureContextV3,
        lineage: RoventLineage,
    ) -> Result<RoventInputCaptureReservationV2, ArenaError> {
        Ok(RoventInputCaptureReservationV2 {
            original: self.original.reserve_input(context, lineage)?,
        })
    }
    pub fn admit_captured_input(
        &mut self,
        captured: RoventCapturedInputV2,
    ) -> Result<(), ArenaError> {
        let ticket = captured.original.pending.ticket;
        let key = (ticket.id.epoch.0, ticket.id.sequence);
        check(
            !self.inputs.contains_key(&key) && self.inputs.len() < 16,
            "raw3 original input duplicate/cap",
        )?;
        self.original.admit_captured_input(captured.original)?;
        self.inputs.insert(key, captured.input);
        Ok(())
    }
    pub fn discard_reserved_input(&mut self, sequence: u64) -> bool {
        self.original.discard_reserved_input(sequence)
    }
    pub fn discard_unpublished(&mut self, id: RequestId) -> bool {
        self.inputs.remove(&(id.epoch.0, id.sequence));
        self.original.discard_unpublished(id)
    }
    pub fn capture_submission(
        &mut self,
        ticket: RoleTicketEnvelope,
        view: &ReviewView,
        input: &PalsVirtualReviewInput,
        identity: &ReviewExecutionIdentity,
        context: GenerationCaptureContextV3,
        lineage: RoventLineage,
    ) -> Result<(), ArenaError> {
        let reservation = self.reserve_input(context, lineage)?;
        let sequence = reservation.sequence();
        let result = reservation
            .capture(ticket, view, input, identity)
            .and_then(|input| self.admit_captured_input(input));
        if result.is_err() {
            self.discard_reserved_input(sequence);
        }
        result
    }
    pub fn observe_native_event<O: ProposerRankingOutput>(
        &mut self,
        event: &TicketNativeEvent<O>,
    ) -> Result<(), ArenaError> {
        self.original.observe_native_event(event)?;
        if let TicketNativeEvent::Retired {
            ticket,
            accepted: false,
            ..
        } = event
        {
            self.inputs.remove(&(ticket.id.epoch.0, ticket.id.sequence));
        }
        Ok(())
    }
    pub fn retire_accepted(
        &mut self,
        admission: &RoventCollectionRootAdmission,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<RoventRetiredRecordV2, ArenaError> {
        let original = self.original.retire_accepted(admission, accepted, output)?;
        self.into_retired(original)
    }
    pub fn retire_root_ranking(
        &mut self,
        accepted: &RoventRootRankingAcceptance,
        output: &RoleOutput,
    ) -> Result<RoventRetiredRecordV2, ArenaError> {
        let original = self.original.retire_root_ranking(accepted, output)?;
        self.into_retired(original)
    }
    fn into_retired(
        &mut self,
        original: RoventRetiredRecord,
    ) -> Result<RoventRetiredRecordV2, ArenaError> {
        let key = (original.ticket.id.epoch.0, original.ticket.id.sequence);
        let input = self
            .inputs
            .remove(&key)
            .ok_or_else(|| invalid("raw3 original input lost before retirement"))?;
        check(
            input.sha256()? == original.record.input_sha256,
            "raw3 retained input differs from original owner",
        )?;
        Ok(RoventRetiredRecordV2 {
            manifest_sha256: original.manifest_sha256,
            ticket: original.ticket,
            admission: original.admission,
            record: RoventTrainingRecordV2 {
                schema: ROVENT_TRAINING_RECORD_V2_DOMAIN.into(),
                role: original.record.role,
                split: original.record.split,
                input,
                input_sha256: original.record.input_sha256,
                origin_sha256: String::new(),
                record_sha256: String::new(),
                lineage: original.record.lineage,
                native: original.record.native,
                head_sources: BTreeMap::new(),
                targets: RoventTargetsV2::default(),
                divergence_index: None,
                nnuec_input: None,
                nnuec_input_sha256: None,
                nnuec_model_sha256: None,
                nnuec_semantics: None,
                nnuec_wdl: None,
            },
        })
    }
    pub fn publish(
        &self,
        mut row: RoventRetiredRecordV2,
    ) -> Result<PublishedRoventRecordV2, ArenaError> {
        check(
            row.manifest_sha256 == self.original.manifest_sha256,
            "raw3 retired row belongs to another owner",
        )?;
        self.validate_admission(&row.admission)?;
        check(
            row.record.native.raw_retirement["root_admission_sha256"] == row.admission.raw_sha256()
                && row.record.native.raw_retirement["root_admission"]
                    == *row.admission.raw_receipt(),
            "raw3 row lost original admission",
        )?;
        row.record.seal(&self.manifest)?;
        Ok(PublishedRoventRecordV2 {
            record: row.record,
            manifest_sha256: row.manifest_sha256,
            admission: row.admission,
        })
    }
    pub fn attach_actual_wdl(
        &mut self,
        row: &mut RoventRetiredRecordV2,
        outcome: &RoventRulesOutcome,
    ) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.original.manifest_sha256,
            "raw3 WDL row owner mismatch",
        )?;
        let input = &row.record.input;
        check(
            row.record.targets.wdl.is_none()
                && outcome.authority.game_id == input.game_id
                && outcome.authority.anchors.contains(&input.rules_anchor),
            "raw3 WDL is not an actual trajectory member",
        )?;
        let sequence = self.original.next()?;
        let raw = json!({"schema":HEAD_DOMAIN,"kind":"rules_wdl","input_sha256":row.record.input_sha256,
            "anchor":input.rules_anchor,"trajectory":outcome.authority,"rules_receipt":outcome.receipt,"rules_replay":outcome.replay});
        let raw_sha = generation_digest(HEAD_DOMAIN, &raw)?;
        let target = GenerationWdlTargetV3 {
            observed_sequence: sequence,
            question_id: input.question_id.clone(),
            view_sha256: input.view_sha256.clone(),
            raw_evidence_sha256: raw_sha.clone(),
            source: "rules_selfplay_trajectory".into(),
            game_id: input.game_id.clone(),
            anchor: input.rules_anchor.clone(),
            trajectory_membership_sha256: outcome.authority.sha256()?,
            rules_result_sha256: outcome.authority.result_evidence_sha256.clone(),
            ending: outcome.authority.ending,
            outcome: outcome.authority.outcome,
        };
        input.wdl_probabilities(&target)?;
        row.record.head_sources.insert(
            "wdl".into(),
            RoventHeadSource {
                actor_id: row.record.lineage.actor_id.clone(),
                source: target.source.clone(),
                binary_sha256: None,
                profile_sha256: None,
                raw_receipt: raw,
                raw_receipt_sha256: raw_sha,
                observed_sequence: sequence,
                anchor: input.rules_anchor.clone(),
            },
        );
        row.record.targets.wdl = Some(target);
        Ok(())
    }
}

impl RoventRetiredRecordV2 {
    pub fn input_sha256(&self) -> &str {
        &self.record.input_sha256
    }
    pub fn submitted_input_seal(&self) -> RoventSubmittedInputSeal {
        input_seal(
            &self.manifest_sha256,
            self.ticket,
            &self.record.input.common_validation_projection(),
            &self.record.input_sha256,
            &self.record.lineage,
        )
    }
}

#[cfg(feature = "pals-collection-onnx")]
impl RoventCorpusProducerV2 {
    /// 형식별 입력은 바꾸지 않고 원 dispatch/completion과 policy 계산만 공유한다.
    pub fn attach_collection_teacher_policy(
        &mut self,
        row: &mut RoventRetiredRecordV2,
        binding: RoventCollectionTeacherBinding,
        owner: &rz_uci::rovent_external_launch::RoventCollectionTeacherOwner,
        dispatch: &rz_search::external_cpu::collection_teacher::TeacherDispatch,
        completion: &rz_search::external_cpu::collection_teacher::TeacherCompletion,
    ) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.original.manifest_sha256,
            "raw3 teacher row belongs to another producer",
        )?;
        self.validate_admission(&row.admission)?;
        let mut projection = RoventRetiredRecord {
            record: historical_v2::common_scope_projection(&row.record),
            manifest_sha256: row.manifest_sha256.clone(),
            ticket: row.ticket,
            admission: row.admission.clone(),
        };
        self.original.attach_collection_teacher_policy(
            &mut projection,
            binding,
            owner,
            dispatch,
            completion,
        )?;
        row.record.targets.policy = projection.record.targets.policy;
        if let Some(source) = projection.record.head_sources.remove("policy") {
            row.record.head_sources.insert("policy".into(), source);
        }
        Ok(())
    }
}
