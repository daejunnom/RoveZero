//! Controlled V action collection over original accepted native publications.
//! A context is counted once; its supported actions run in separate roots in a
//! registered order. Native/input/causal receipts retain their original seals.
use super::*;
use rz_search::pals::engine::adaptive_port::validator_port::{
    AdaptiveValidatorQuestion, ValidatorActionSelection,
};
use rz_uci::pals_adaptive::CompletedValidatorAction;

const ACTION_GROUP_DOMAIN: &str = "rz-pals-generation-frozen-action-group-evidence/1";

/// Distinct parameter/export/native namespaces remain bound by the intake
/// owner. A caller may preserve an unverified declaration, but cannot mint a
/// successful frozen-weights authority with a boolean or arbitrary JSON.
#[derive(Clone)]
pub struct FrozenPcRegistrationV3 {
    epoch: u64,
    parameter_sha256: String,
    native_model_epoch_sha256: String,
    native_backend_sha256: String,
    teacher_binary_sha256: String,
    teacher_profile_sha256: String,
    verification_sha256: Option<String>,
}
impl FrozenPcRegistrationV3 {
    pub fn unverified(
        plan: &ActionOutcomeCollectionPlanV3,
        native_model_epoch_sha256: String,
        native_backend_sha256: String,
        teacher_binary_sha256: String,
        teacher_profile_sha256: String,
    ) -> Result<Self, ArenaError> {
        plan.validate()?;
        check(
            [
                &native_model_epoch_sha256,
                &native_backend_sha256,
                &teacher_binary_sha256,
                &teacher_profile_sha256,
            ]
            .into_iter()
            .all(|sha| generation_sha(sha)),
            "frozen PC source pin syntax",
        )?;
        Ok(Self {
            epoch: plan.frozen_pc_epoch,
            parameter_sha256: plan.frozen_pc_weights_sha256.clone(),
            native_model_epoch_sha256,
            native_backend_sha256,
            teacher_binary_sha256,
            teacher_profile_sha256,
            verification_sha256: None,
        })
    }
    /// The actual asset-intake owner calls this only after independently
    /// joining named shared/P/C tensors with the exported initializers and
    /// actual loaded graph pins. Metadata-only intake must not call it.
    #[allow(dead_code)]
    pub(crate) fn after_verified_intake(
        mut registered: Self,
        verification_sha256: String,
    ) -> Result<Self, ArenaError> {
        check(
            generation_sha(&verification_sha256),
            "frozen PC content verification receipt pin",
        )?;
        registered.verification_sha256 = Some(verification_sha256);
        Ok(registered)
    }
    pub fn content_verified(&self) -> bool {
        self.verification_sha256.is_some()
    }
    pub fn verification_sha256(&self) -> Option<&str> {
        self.verification_sha256.as_deref()
    }
}

/// This value copies an actual owner-issued immutable question before action
/// outcomes. It is not deserializable and cannot be made from a support mask.
pub struct FrozenActionRegistrationV3 {
    plan: ActionOutcomeCollectionPlanV3,
    pins: FrozenPcRegistrationV3,
    frozen_context: String,
    support: [bool; 7],
    actions: Vec<usize>,
    schedule: String,
    root: GenerationRulesAnchorV3,
}
impl FrozenActionRegistrationV3 {
    pub fn from_actual_question(
        plan: &ActionOutcomeCollectionPlanV3,
        question: &AdaptiveValidatorQuestion,
        pins: &FrozenPcRegistrationV3,
    ) -> Result<Self, ArenaError> {
        plan.validate()?;
        check(
            pins.content_verified()
                && pins.epoch == plan.frozen_pc_epoch
                && pins.parameter_sha256 == plan.frozen_pc_weights_sha256,
            "UnverifiedFrozenPc: named PC parameters and actual loaded exports must be verified before collection",
        )?;
        let support = question.support_mask();
        let actions = (0..7).filter(|index| support[*index]).collect::<Vec<_>>();
        check(
            !actions.is_empty(),
            "actual action context supports no registered action",
        )?;
        let view = question.input().view.as_ref();
        let anchor = view
            .anchors()
            .first()
            .ok_or_else(|| invalid("actual action view has no root anchor"))?;
        let root_position =
            Position::from_snapshot(&anchor.snapshot, rz_position::PositionLimits::default())
                .map_err(|e| invalid(e.to_string()))?;
        let root = exact_rules_anchor(&root_position)?;
        let frozen_context = hex(&question.frozen_context_digest().0);
        let schedule = generation_digest("rz-pals-generation-action-enumeration/1", &actions)?;
        Ok(Self {
            plan: plan.clone(),
            pins: pins.clone(),
            frozen_context,
            support,
            actions,
            schedule,
            root,
        })
    }
    pub fn actions(&self) -> &[usize] {
        &self.actions
    }
    pub fn action_support_mask(&self) -> [bool; 7] {
        self.support
    }
    pub fn frozen_context_sha256(&self) -> &str {
        &self.frozen_context
    }
    pub fn frozen_pc_epoch(&self) -> u64 {
        self.plan.frozen_pc_epoch
    }
    pub fn frozen_pc_weights_sha256(&self) -> &str {
        &self.plan.frozen_pc_weights_sha256
    }
    pub(super) fn verify_collection_source(
        &self,
        plan: &ActionOutcomeCollectionPlanV3,
        producer: &GenerationProducerAuthorityV3,
        root: &Position,
    ) -> Result<(), ArenaError> {
        check(
            self.plan.frozen_pc_epoch == plan.frozen_pc_epoch
                && self.plan.frozen_pc_weights_sha256 == plan.frozen_pc_weights_sha256
                && self.pins.content_verified()
                && self.pins.parameter_sha256 == producer.model_sha256
                && self.pins.teacher_binary_sha256 == producer.teacher_binary_sha256
                && self.pins.teacher_profile_sha256 == producer.teacher_profile_sha256
                && self.root == exact_rules_anchor(root)?,
            "actual frozen context differs from registered PC/teacher/source root",
        )
    }
    pub(super) fn verify_actual_completion(
        &self,
        record: &GenerationRecordV3,
        completed: &CompletedValidatorAction,
    ) -> Result<(), ArenaError> {
        let semantic = completed.semantic();
        check(
            record.input.role == GenerationRoleV3::Validator
                && record.future_targets.task.is_none()
                && record.input.model_epoch == self.plan.frozen_pc_epoch
                && semantic.source().representation.frozen_epoch == self.plan.frozen_pc_epoch
                && record.input.action_support_mask == self.support
                && semantic.action_support_mask() == self.support
                && semantic.selection_provenance()
                    == ValidatorActionSelection::RegisteredSupportedEnumerationV1
                && hex(&semantic.frozen_context_digest().0) == self.frozen_context
                && record.input.selected_view.anchors.first() == Some(&self.root)
                && completed.exclusive_action_cost().is_some()
                && completed.training_wall_cost().is_some(),
            "action differs from preregistered frozen PC/question or has no private exclusive fence",
        )
    }
    pub(super) fn verify_native_pins(
        &self,
        record: &GenerationRecordV3,
        evidence: &BTreeMap<String, serde_json::Value>,
    ) -> Result<(), ArenaError> {
        let submitted = original_submission(record, evidence)?;
        check(
            self.pins.content_verified()
                && submitted
                    .get("native_model_epoch_sha256")
                    .and_then(|v| v.as_str())
                    == Some(self.pins.native_model_epoch_sha256.as_str())
                && submitted
                    .get("native_backend_sha256")
                    .and_then(|v| v.as_str())
                    == Some(self.pins.native_backend_sha256.as_str()),
            "actual native owner differs from the content-verified frozen PC asset",
        )
    }
}

/// An original action row minted by the actual manager/capture integration.
/// Its private fields prevent a mock or caller JSON from claiming execution.
pub struct GenerationIsolatedActionV3 {
    pub(super) record: GenerationRecordV3,
    pub(super) evidence: BTreeMap<String, serde_json::Value>,
    pub(super) outcome: GenerationActionOutcomeV3,
    pub(super) root_authority: rz_contracts::pals::SearchAuthority,
    pub(super) additional_records: Vec<GenerationRecordV3>,
}
impl GenerationIsolatedActionV3 {
    pub fn record(&self) -> &GenerationRecordV3 {
        &self.record
    }
    pub fn outcome(&self) -> &GenerationActionOutcomeV3 {
        &self.outcome
    }
}

/// Owns at most seven rows and their bounded source evidence. Publication is
/// possible only after every preregistered action has closed its isolated root.
pub struct FrozenActionGroupV3 {
    registration: FrozenActionRegistrationV3,
    members: Vec<GenerationIsolatedActionV3>,
    semantic: Option<String>,
    authorities: BTreeSet<(u64, u64, u64)>,
    sealed: bool,
}
pub struct GenerationFrozenActionPublicationV3 {
    pub(super) records: Vec<GenerationRecordV3>,
    pub(super) evidence: BTreeMap<String, serde_json::Value>,
    pub group_sha256: String,
    pub member_count: usize,
    pub known_pair_count: usize,
}
impl GenerationFrozenActionPublicationV3 {
    pub fn records(&self) -> &[GenerationRecordV3] {
        &self.records
    }
    pub fn evidence(&self) -> &BTreeMap<String, serde_json::Value> {
        &self.evidence
    }
}
impl FrozenActionGroupV3 {
    pub fn begin(registration: FrozenActionRegistrationV3) -> Self {
        Self {
            registration,
            members: vec![],
            semantic: None,
            authorities: BTreeSet::new(),
            sealed: false,
        }
    }
    pub fn registration(&self) -> &FrozenActionRegistrationV3 {
        &self.registration
    }
    pub fn next_action(&self) -> Option<usize> {
        self.registration.actions.get(self.members.len()).copied()
    }
    #[allow(
        clippy::result_large_err,
        reason = "A rejected action returns its original evidence and process owner inline without allocating or discarding failure custody"
    )]
    pub fn push_actual(
        &mut self,
        member: GenerationIsolatedActionV3,
    ) -> Result<(), (ArenaError, GenerationIsolatedActionV3)> {
        let checked = (|| -> Result<String, ArenaError> {
            check(!self.sealed, "frozen group is already sealed")?;
            check(
                self.next_action() == Some(member.outcome.action.index()),
                "actual actions are missing, reordered or unregistered",
            )?;
            check(
                member.record.input.model_epoch == self.registration.frozen_pc_epoch()
                    && member.record.input.action_support_mask == self.registration.support,
                "isolated member changed the registered model or support",
            )?;
            let authority = member.root_authority;
            check(
                !self.authorities.contains(&(
                    authority.epoch.0,
                    authority.game.0,
                    authority.root.0,
                )),
                "a second action reused the same root instead of an isolated root",
            )?;
            let semantic = actual_semantic_input(&member.record, &member.evidence)?;
            if let Some(original) = &self.semantic {
                check(
                    *original == semantic,
                    "isolated action changed the frozen model/input/history/mask",
                )?;
                let first = &self.members[0];
                check(
                    first.record.input.game_id == member.record.input.game_id
                        && first.record.input.opening_id == member.record.input.opening_id
                        && first.record.input.rules_anchor == member.record.input.rules_anchor
                        && first.record.input.actual_history == member.record.input.actual_history
                        && first.record.input.context_epoch == member.record.input.context_epoch
                        && same_native_context(
                            &first.evidence,
                            &member.evidence,
                            &first.record.input_sha256,
                            &member.record.input_sha256,
                        )?,
                    "frozen action origins differ in anchor, actual history, context or backend",
                )?;
            }
            Ok(semantic)
        })();
        let semantic = match checked {
            Ok(semantic) => semantic,
            Err(error) => return Err((error, member)),
        };
        if self.semantic.is_none() {
            self.semantic = Some(semantic);
        }
        let authority = member.root_authority;
        self.authorities
            .insert((authority.epoch.0, authority.game.0, authority.root.0));
        self.members.push(member);
        Ok(())
    }
    pub fn finish(
        &mut self,
        chronology: &mut GenerationChronologyV3,
    ) -> Result<GenerationFrozenActionPublicationV3, ArenaError> {
        check(
            !self.sealed && self.members.len() == self.registration.actions.len(),
            "incomplete isolated action group",
        )?;
        let semantic = self
            .semantic
            .clone()
            .ok_or_else(|| invalid("group has no actual publication"))?;
        let group_sha = generation_digest(
            "rz-pals-generation-frozen-action-group/1",
            &serde_json::json!([
                semantic,
                self.registration.frozen_context,
                self.registration.plan.frozen_pc_epoch,
                self.registration.plan.frozen_pc_weights_sha256,
                self.registration.actions,
                self.registration.schedule
            ]),
        )?;
        let mut receipts = BTreeMap::new();
        let mut records = vec![];
        let mut members = vec![];
        let mut outcomes = vec![];
        for member in &self.members {
            let mut authorities = member.evidence.iter().filter(|(_, raw)| {
                raw.get("schema").and_then(|v| v.as_str())
                    == Some("rz-pals-generation-action-authority-evidence/1")
                    && raw.get("input_sha256").and_then(|v| v.as_str())
                        == Some(member.record.input_sha256.as_str())
                    && raw.get("causal_receipt_sha256").and_then(|v| v.as_str())
                        == Some(member.outcome.causal_receipt_sha256.as_str())
            });
            let (sha, authority) = authorities
                .next()
                .ok_or_else(|| invalid("actual action member lacks its opaque source authority"))?;
            check(
                authorities.next().is_none()
                    && generation_digest(
                        "rz-pals-generation-action-authority-evidence/1",
                        authority,
                    )? == *sha
                    && authority
                        .get("frozen_context_sha256")
                        .and_then(|v| v.as_str())
                        == Some(self.registration.frozen_context.as_str())
                    && authority.get("frozen_pc_epoch").and_then(|v| v.as_u64())
                        == Some(self.registration.plan.frozen_pc_epoch)
                    && authority
                        .get("frozen_pc_weights_sha256")
                        .and_then(|v| v.as_str())
                        == Some(self.registration.plan.frozen_pc_weights_sha256.as_str()),
                "actual action member authority changed frozen source",
            )?;
            members.push(serde_json::json!({"action":member.outcome.action,"index":member.outcome.action.index(),
                "actual_input_sha256":member.record.input_sha256,"causal_receipt_sha256":member.outcome.causal_receipt_sha256,"action_authority_sha256":sha}));
            outcomes.push(member.outcome.clone());
            records.push(member.record.clone());
            records.extend(member.additional_records.iter().cloned());
            for (sha, raw) in &member.evidence {
                if let Some(previous) = receipts.insert(sha.clone(), raw.clone()) {
                    check(previous == *raw, "group evidence seal collision")?;
                }
            }
        }
        let original = &records[0];
        let raw_group = serde_json::json!({"schema":ACTION_GROUP_DOMAIN,"input_sha256":original.input_sha256,"group_sha256":group_sha,
            "semantic_input_sha256":semantic,"frozen_context_sha256":self.registration.frozen_context,"frozen_pc_epoch":self.registration.plan.frozen_pc_epoch,
            "frozen_pc_weights_sha256":self.registration.plan.frozen_pc_weights_sha256,"registered_actions":self.registration.actions,
            "action_schedule_sha256":self.registration.schedule,"selection_provenance":"registered-supported-enumeration-v1","members":members});
        let group_receipt = generation_digest(ACTION_GROUP_DOMAIN, &raw_group)?;
        receipts.insert(group_receipt, raw_group);
        let derived = generation_derive_tasks(&original.input, &outcomes)?;
        let mut target = GenerationTaskTargetV3 {
            observed_sequence: chronology.label_after(&original.input)?,
            question_id: original.input.question_id.clone(),
            view_sha256: original.input.view_sha256.clone(),
            raw_evidence_sha256: "0".repeat(64),
            source: "actual_action_witness".into(),
            outcomes,
            target_known_mask: derived.target_known_mask,
        };
        target.raw_evidence_sha256 =
            seal_head_evidence(&original.input_sha256, "task", &target, &mut receipts)?;
        // Source records keep future task None. The pinned later head receipt
        // hydrates the representative at read time and resolves original aliases.
        let mut validated = original.clone();
        validated.future_targets.task = Some(target);
        validate_generation_head_evidence(&validated, &receipts)?;
        self.sealed = true;
        Ok(GenerationFrozenActionPublicationV3 {
            member_count: records.len(),
            records,
            evidence: receipts,
            group_sha256: group_sha,
            known_pair_count: derived.pairs.len(),
        })
    }
}
fn original_submission<'a>(
    record: &GenerationRecordV3,
    evidence: &'a BTreeMap<String, serde_json::Value>,
) -> Result<&'a serde_json::Value, ArenaError> {
    let mut rows = evidence.iter().filter(|(_, raw)| {
        raw.get("schema").and_then(|v| v.as_str())
            == Some("rz-pals-generation-submission-evidence/1")
            && raw.get("input_sha256").and_then(|v| v.as_str())
                == Some(record.input_sha256.as_str())
    });
    let (sha, row) = rows
        .next()
        .ok_or_else(|| invalid("missing native original action submission"))?;
    check(
        rows.next().is_none()
            && generation_digest("rz-pals-generation-submission-evidence/1", row)? == *sha,
        "ambiguous native action input publication",
    )?;
    Ok(row)
}
fn actual_semantic_input(
    record: &GenerationRecordV3,
    evidence: &BTreeMap<String, serde_json::Value>,
) -> Result<String, ArenaError> {
    let row = original_submission(record, evidence)?;
    let ticket = row
        .get("ticket")
        .ok_or_else(|| invalid("native action ticket absent"))?;
    Ok(generation_digest(
        "rz-pals-generation-semantic-sample/1",
        &serde_json::json!([
            record.input.role,
            row["native_model_epoch_sha256"],
            row["native_input_sha256"],
            ticket["representation_history_sha256"],
            record.input.action_support_mask
        ]),
    )?)
}
fn same_native_context(
    a: &BTreeMap<String, serde_json::Value>,
    b: &BTreeMap<String, serde_json::Value>,
    asha: &str,
    bsha: &str,
) -> Result<bool, ArenaError> {
    let find = |map: &BTreeMap<String, serde_json::Value>, seal: &str| {
        map.values()
            .find(|raw| {
                raw.get("schema").and_then(|v| v.as_str())
                    == Some("rz-pals-generation-submission-evidence/1")
                    && raw.get("input_sha256").and_then(|v| v.as_str()) == Some(seal)
            })
            .cloned()
    };
    let a = find(a, asha).ok_or_else(|| invalid("first original action submission absent"))?;
    let b = find(b, bsha).ok_or_else(|| invalid("next original action submission absent"))?;
    Ok([
        "native_model_epoch_sha256",
        "native_input_sha256",
        "native_backend_sha256",
        "native_endpoint_history_sha256",
        "actual_root_anchor",
        "query_anchor",
    ]
    .iter()
    .all(|key| a.get(*key) == b.get(*key)))
}
