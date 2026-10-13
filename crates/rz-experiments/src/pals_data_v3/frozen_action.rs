//! Frozen action alias codec. Shape/digest validation preserves the original
//! receipts; it is not an actual execution or asset-loading authority.
use super::*;
pub const GENERATION_ACTION_GROUP_DOMAIN: &str =
    "rz-pals-generation-frozen-action-group-evidence/1";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationFrozenActionMemberV3 {
    pub action: GenerationTaskV3,
    pub index: u8,
    pub actual_input_sha256: String,
    pub causal_receipt_sha256: String,
    pub action_authority_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationFrozenActionGroupReceiptV3 {
    pub schema: String,
    pub input_sha256: String,
    pub group_sha256: String,
    pub semantic_input_sha256: String,
    pub frozen_context_sha256: String,
    pub frozen_pc_epoch: u64,
    pub frozen_pc_weights_sha256: String,
    pub registered_actions: Vec<u8>,
    pub action_schedule_sha256: String,
    pub selection_provenance: String,
    pub members: Vec<GenerationFrozenActionMemberV3>,
}
impl GenerationFrozenActionGroupReceiptV3 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            self.schema == GENERATION_ACTION_GROUP_DOMAIN
                && self.selection_provenance == "registered-supported-enumeration-v1"
                && self.frozen_pc_epoch > 0
                && [
                    &self.input_sha256,
                    &self.group_sha256,
                    &self.semantic_input_sha256,
                    &self.frozen_context_sha256,
                    &self.frozen_pc_weights_sha256,
                    &self.action_schedule_sha256,
                ]
                .into_iter()
                .all(|sha| generation_sha(sha))
                && !self.registered_actions.is_empty()
                && self.registered_actions.len() <= 7
                && self.members.len() == self.registered_actions.len()
                && self.registered_actions.iter().all(|index| *index < 7)
                && self
                    .registered_actions
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
            "frozen action registration shape/order/domain",
        )?;
        let schedule = generation_digest(
            "rz-pals-generation-action-enumeration/1",
            &self.registered_actions,
        )?;
        ensure(
            schedule == self.action_schedule_sha256
                && self.group_sha256
                    == generation_digest(
                        "rz-pals-generation-frozen-action-group/1",
                        &serde_json::json!([
                            self.semantic_input_sha256,
                            self.frozen_context_sha256,
                            self.frozen_pc_epoch,
                            self.frozen_pc_weights_sha256,
                            self.registered_actions,
                            schedule
                        ]),
                    )?,
            "frozen action registration digest mismatch",
        )?;
        let mut inputs = BTreeSet::new();
        let mut causal = BTreeSet::new();
        let mut authority = BTreeSet::new();
        for (index, member) in self.registered_actions.iter().zip(&self.members) {
            ensure(
                member.index == *index
                    && member.action.index() == *index as usize
                    && generation_sha(&member.actual_input_sha256)
                    && generation_sha(&member.causal_receipt_sha256)
                    && generation_sha(&member.action_authority_sha256)
                    && inputs.insert(member.actual_input_sha256.as_str())
                    && causal.insert(member.causal_receipt_sha256.as_str())
                    && authority.insert(member.action_authority_sha256.as_str()),
                "frozen action member duplicate/order/source identity",
            )?;
        }
        ensure(
            inputs.contains(self.input_sha256.as_str()),
            "frozen representative is not an original member",
        )
    }
}
pub(super) fn causal_input_for_outcome<'a>(
    record: &'a GenerationRecordV3,
    outcome: &GenerationActionOutcomeV3,
    evidence: &BTreeMap<String, serde_json::Value>,
) -> Result<String, ManifestError> {
    let mut groups = evidence.iter().filter(|(_, raw)| {
        raw.get("schema").and_then(|v| v.as_str()) == Some(GENERATION_ACTION_GROUP_DOMAIN)
            && raw.get("input_sha256").and_then(|v| v.as_str())
                == Some(record.input_sha256.as_str())
    });
    let Some((sha, raw)) = groups.next() else {
        return Ok(record.input_sha256.clone());
    };
    ensure(
        groups.next().is_none() && generation_digest(GENERATION_ACTION_GROUP_DOMAIN, raw)? == *sha,
        "ambiguous/unsealed frozen alias",
    )?;
    let group: GenerationFrozenActionGroupReceiptV3 =
        serde_json::from_value(raw.clone()).map_err(|e| ManifestError::Integrity(e.to_string()))?;
    group.validate()?;
    ensure(
        record.input.role == GenerationRoleV3::Validator
            && group.frozen_pc_epoch == record.input.model_epoch
            && group.registered_actions
                == (0u8..7)
                    .filter(|index| record.input.action_support_mask[*index as usize])
                    .collect::<Vec<_>>(),
        "frozen alias role/model/supported registration differs",
    )?;
    let member = group
        .members
        .iter()
        .find(|member| {
            member.action == outcome.action
                && member.causal_receipt_sha256 == outcome.causal_receipt_sha256
        })
        .ok_or_else(|| {
            ManifestError::Integrity("outcome is not an original registered action member".into())
        })?;
    let raw_authority = evidence
        .get(&member.action_authority_sha256)
        .ok_or_else(|| ManifestError::Integrity("missing original member authority".into()))?;
    ensure(
        generation_digest(
            "rz-pals-generation-action-authority-evidence/1",
            raw_authority,
        )? == member.action_authority_sha256
            && raw_authority.get("schema").and_then(|v| v.as_str())
                == Some("rz-pals-generation-action-authority-evidence/1")
            && raw_authority.get("input_sha256").and_then(|v| v.as_str())
                == Some(member.actual_input_sha256.as_str())
            && raw_authority
                .get("causal_receipt_sha256")
                .and_then(|v| v.as_str())
                == Some(member.causal_receipt_sha256.as_str())
            && raw_authority
                .get("frozen_context_sha256")
                .and_then(|v| v.as_str())
                == Some(group.frozen_context_sha256.as_str())
            && raw_authority
                .get("frozen_pc_epoch")
                .and_then(|v| v.as_u64())
                == Some(group.frozen_pc_epoch)
            && raw_authority
                .get("frozen_pc_weights_sha256")
                .and_then(|v| v.as_str())
                == Some(group.frozen_pc_weights_sha256.as_str()),
        "frozen alias changes the original action authority",
    )?;
    Ok(member.actual_input_sha256.clone())
}
