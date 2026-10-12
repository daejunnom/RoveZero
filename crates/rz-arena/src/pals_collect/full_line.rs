//! Exact V2 capture and observed target coverage; neither is a training label.
use super::*;
#[cfg(feature = "pals-collection-onnx")]
use rz_eval::pals_model::PalsCandidateToken;
use rz_eval::pals_model::PalsModelProfile;

pub const CONTEXT_DOMAIN: &str = "rz-pals-native-full-line-context/1";
pub const CONTEXT_ARTIFACT: &str = "native-full-line-contexts.v2.jsonl";

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "observation", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsObservedCountV2 {
    Observed { value: u64 },
    Unknown { reason: String },
}

/// Selected producer namespace is metadata. It never turns foreign CP into
/// owned raw, or a neural estimate into Rules or an information-gain rank.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(
    tag = "value_namespace",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PalsCheckerValueNamespaceV2 {
    OwnRaw {
        checker_profile_sha256: String,
    },
    FreshModelWdl {
        model_identity: String,
        model_epoch_sha256: String,
        encoding_sha256: String,
        checker_kind: String,
        checker_profile_sha256: String,
    },
    Unknown {
        reason: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsFullLineRecordV2 {
    pub moves: Vec<u16>,
    pub parent: Option<usize>,
    pub supersedes: Option<usize>,
    pub parent_required: bool,
    pub supersedes_required: bool,
}
/// Raw source IDs/revisions are provenance only, outside the neural payload.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsRawRecordRelationshipV2 {
    pub record_id: u64,
    pub revision: u64,
    pub parent_revision: Option<u64>,
    pub supersedes_revision: Option<u64>,
    pub parent_omitted: bool,
    pub supersedes_omitted: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsFullLineContextV2 {
    pub domain: String,
    pub input_sha256: String,
    pub sidecar_sha256: String,
    pub canonical_tensor_sha256: String,
    pub process_epoch: u64,
    pub request_sequence: u64,
    pub model_profile: String,
    pub model_semantics: String,
    pub encoding_profile: String,
    pub records: Vec<PalsFullLineRecordV2>,
    pub raw_record_relationships: Vec<PalsRawRecordRelationshipV2>,
    pub relationship_links_omitted: PalsObservedCountV2,
    pub query_prefix: Vec<u16>,
    pub query_proposal: Vec<u16>,
    pub query_counter: Vec<u16>,
    pub namespace: PalsCheckerValueNamespaceV2,
    pub checker_identity: Option<serde_json::Value>,
    pub checker_profile_sha256: String,
    /// No private V task was submitted by this P/C capture.
    pub task_utility_observations: PalsObservedCountV2,
    pub task_utility_mask_reason: String,
    pub sha256: String,
}

#[cfg(feature = "pals-collection-onnx")]
fn moves(tokens: &[PalsCandidateToken]) -> Result<Vec<u16>, ArenaError> {
    tokens
        .iter()
        .map(|m| m.packed().map_err(|e| invalid(e.to_string())))
        .collect()
}

pub(super) fn model_config(
    source: &PalsCollectionSourceDescription,
) -> Result<PalsModelConfig, ArenaError> {
    match &source.source {
        PalsInputSource::OwnPals { .. } => {
            let config: PalsModelConfig = serde_json::from_value(source.configuration.clone())
                .map_err(|e| invalid(format!("registered model configuration: {e}")))?;
            config.validate().map_err(|e| invalid(e.to_string()))?;
            Ok(config)
        }
        PalsInputSource::OwnCpu { .. } => Ok(PalsModelConfig::baseline()),
    }
}

#[cfg(any(test, feature = "pals-collection-onnx"))]
fn raw_relationships(
    records: &[rz_eval::pals_model::PalsRecordToken],
    lines: &[rz_eval::pals_model::PalsRecordLine],
    raw_records: &[RoleRecord],
) -> Result<(Vec<PalsRawRecordRelationshipV2>, u64), ArenaError> {
    fn require(condition: bool, message: &str) -> Result<(), ArenaError> {
        if condition {
            Ok(())
        } else {
            Err(invalid(message))
        }
    }
    require(
        records.len() == lines.len(),
        "selected record/path count differs",
    )?;
    let mut revisions = BTreeSet::new();
    for record in records {
        require(
            revisions.insert(record.revision),
            "selected raw revision is ambiguous",
        )?;
    }
    let mut omitted = 0_u64;
    let relationships = records
        .iter()
        .zip(lines)
        .map(|(token, line)| {
            let index = token
                .record_id
                .checked_sub(1)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| invalid("full-line raw source ID invalid"))?;
            let raw = raw_records
                .get(index)
                .ok_or_else(|| invalid("full-line selected source absent from actual query"))?;
            require(
                raw.revision == token.revision,
                "full-line raw relationship revision differs",
            )?;
            for (revision, local, required) in [
                (raw.parent_revision, line.parent, line.parent_required),
                (
                    raw.supersedes_revision,
                    line.supersedes,
                    line.supersedes_required,
                ),
            ] {
                match local {
                    Some(local) => require(
                        records.get(local).map(|r| r.revision) == revision,
                        "full-line local relationship differs from actual raw relationship",
                    )?,
                    None => {
                        require(!required, "required relationship was omitted")?;
                        require(
                            revision.is_none_or(|r| !revisions.contains(&r)),
                            "selected actual relationship target was omitted",
                        )?;
                    }
                }
            }
            let parent_omitted = raw.parent_revision.is_some() && line.parent.is_none();
            let supersedes_omitted = raw.supersedes_revision.is_some() && line.supersedes.is_none();
            omitted += u64::from(parent_omitted) + u64::from(supersedes_omitted);
            Ok(PalsRawRecordRelationshipV2 {
                record_id: token.record_id,
                revision: raw.revision,
                parent_revision: raw.parent_revision,
                supersedes_revision: raw.supersedes_revision,
                parent_omitted,
                supersedes_omitted,
            })
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    Ok((relationships, omitted))
}

#[cfg(feature = "pals-collection-onnx")]
pub(super) fn prepare_context(
    input: &PalsFrozenInput,
    sidecar: &PalsNativeInputSidecar,
    prepared: &PalsModelInput,
    raw_records: &[rz_search::pals::engine::RoleRecord],
    source: &PalsCollectionSourceDescription,
    epoch: u64,
    sequence: u64,
) -> Result<Option<PalsFullLineContextV2>, ArenaError> {
    let config = model_config(source)?;
    if !config.profile.uses_full_line() {
        return Ok(None);
    }
    prepared
        .validate(&config)
        .map_err(|e| invalid(e.to_string()))?;
    let lines = prepared
        .full_line
        .as_ref()
        .ok_or_else(|| invalid("full-line context lacks actual payload"))?;
    let (raw_record_relationships, omitted) =
        raw_relationships(&prepared.records, &lines.records, raw_records)?;
    let native = source.native.as_ref();
    let resolver = native.and_then(|v| v["resolver_policy"].as_str());
    let checker_kind = native.and_then(|v| v["checker_kind"].as_str());
    let namespace = match (resolver, checker_kind) {
        (Some("pals-cpu-raw-restricted/0.1"), Some("own")) => PalsCheckerValueNamespaceV2::OwnRaw {
            checker_profile_sha256: source.cpu_profile_sha256.clone(),
        },
        (Some("pals-model-wdl-restricted/0.1"), Some(kind @ ("own" | "external_uci"))) => {
            let model_identity = native
                .and_then(|v| v["actual_model_identity"].as_str())
                .ok_or_else(|| invalid("model WDL context lacks actual model identity"))?;
            PalsCheckerValueNamespaceV2::FreshModelWdl {
                model_identity: model_identity.into(),
                model_epoch_sha256: hex(prepared.model_epoch),
                encoding_sha256: source.encoding_sha256.clone(),
                checker_kind: kind.into(),
                checker_profile_sha256: source.cpu_profile_sha256.clone(),
            }
        }
        _ => PalsCheckerValueNamespaceV2::Unknown {
            reason: "actual resolver/checker identity was not supplied by this producer".into(),
        },
    };
    let mut value = PalsFullLineContextV2 {
        domain: CONTEXT_DOMAIN.into(),
        input_sha256: input.sha256().into(),
        sidecar_sha256: sidecar.sha256.clone(),
        canonical_tensor_sha256: sidecar.canonical_tensor_sha256.clone(),
        process_epoch: epoch,
        request_sequence: sequence,
        model_profile: config.profile.as_str().into(),
        model_semantics: config.profile.model_semantics().into(),
        encoding_profile: config.profile.encoding_schema().into(),
        records: lines
            .records
            .iter()
            .map(|r| {
                Ok(PalsFullLineRecordV2 {
                    moves: moves(&r.moves)?,
                    parent: r.parent,
                    supersedes: r.supersedes,
                    parent_required: r.parent_required,
                    supersedes_required: r.supersedes_required,
                })
            })
            .collect::<Result<_, ArenaError>>()?,
        raw_record_relationships,
        relationship_links_omitted: PalsObservedCountV2::Observed { value: omitted },
        query_prefix: moves(&lines.query_prefix)?,
        query_proposal: moves(&lines.query_proposal)?,
        query_counter: moves(&lines.query_counter)?,
        namespace,
        checker_identity: native.and_then(|v| v.get("checker_identity")).cloned(),
        checker_profile_sha256: source.cpu_profile_sha256.clone(),
        task_utility_observations: PalsObservedCountV2::Observed { value: 0 },
        task_utility_mask_reason:
            "P/C role input; no private V task or measured information gain was submitted".into(),
        sha256: String::new(),
    };
    value.sha256 = canonical_sha256(&value)?;
    Ok(Some(value))
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsTargetCoverageReceiptV2 {
    pub domain: String,
    pub source_kind: String,
    pub model_profile: String,
    pub current_inputs: PalsObservedCountV2,
    pub policy_targets: PalsObservedCountV2,
    pub actual_outcome_targets: PalsObservedCountV2,
    pub repair_supported: PalsObservedCountV2,
    pub repair_refuted: PalsObservedCountV2,
    pub verifier_supported_utility: PalsObservedCountV2,
    pub masked_reasons: BTreeMap<String, u64>,
    pub provenance: Vec<serde_json::Value>,
    pub scan_complete: bool,
    pub target_admission: String,
    pub sha256: String,
}

pub(super) fn coverage(
    rows: &[PalsLearningRecord],
    profile: PalsModelProfile,
) -> Result<PalsTargetCoverageReceiptV2, ArenaError> {
    let mut latest = BTreeMap::<&str, &PalsLearningRecord>::new();
    for row in rows {
        row.validate()?;
        let key = row.input.sha256();
        if let Some(previous) = latest.get(key) {
            // Whole-label supersession only: never merge old target fields.
            if row.future_label.as_ref().is_none_or(|l| {
                previous
                    .future_label
                    .as_ref()
                    .is_some_and(|p| p.observed_sequence >= l.observed_sequence)
            }) {
                continue;
            }
        }
        latest.insert(key, row);
    }
    let (mut policy, mut outcome, mut supported, mut refuted) = (0, 0, 0, 0);
    let mut reasons = BTreeMap::new();
    let mut provenance = Vec::with_capacity(latest.len());
    for (&input_sha, row) in &latest {
        let mut mask = |reason: &str| {
            *reasons.entry(reason.to_owned()).or_insert(0) += 1;
        };
        if let Some(label) = &row.future_label {
            if label.policy.is_some() {
                policy += 1;
            } else {
                mask("policy_target_absent");
            }
            if label.value_wdl.is_some() {
                outcome += 1;
            } else {
                mask("actual_outcome_unobserved_or_virtual_branch");
            }
            match label.counterexample.as_ref().map(|c| c.validity) {
                Some(PalsConditionalValidity::SupportedAfterRepair) => supported += 1,
                Some(PalsConditionalValidity::RefutedByRepair) => refuted += 1,
                Some(PalsConditionalValidity::NotExamined) => mask("repair_not_examined"),
                Some(PalsConditionalValidity::Disputed) => mask("repair_disputed"),
                None => mask("repair_target_absent"),
            }
            if label.verifier_tasks.is_some() {
                mask("V_context_and_namespace_support_not_observed_by_PC_collector");
            }
            provenance.push(
                serde_json::json!({"input_sha256":input_sha,"label_sha256":row.label_digest()?,
                "observed_sequence":label.observed_sequence,"source":label.provenance,
                "supersedes_label_sha256":label.supersedes_label_sha256}),
            );
        } else {
            mask("future_label_absent");
        }
        mask("private_V_utility_context_not_observed");
    }
    let observed = |value| PalsObservedCountV2::Observed { value };
    let mut result = PalsTargetCoverageReceiptV2 {
        domain: "rz-pals-target-coverage-v2/1".into(), source_kind: "actual_collector".into(),
        model_profile: profile.as_str().into(), current_inputs: observed(latest.len() as u64),
        policy_targets: observed(policy), actual_outcome_targets: observed(outcome),
        repair_supported: observed(supported), repair_refuted: observed(refuted),
        verifier_supported_utility: observed(0), masked_reasons: reasons, provenance,
        scan_complete: true,
        target_admission: "observed current labels only; V remains masked until branch/profile/budget/rank/namespace evidence is independently supported".into(),
        sha256: String::new(),
    };
    result.sha256 = canonical_sha256(&result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_relationships_preserve_only_actual_omissions_and_reject_local_drift() {
        use rz_eval::pals_model::{PalsRecordLine, PalsRecordToken, RECORD_FEATURES};
        let raw = |revision, parent_revision| RoleRecord {
            revision,
            parent_revision,
            supersedes_revision: None,
            origin_state: StateId(0),
            kind: RecordKind::Repair,
            line: vec![],
            value: None,
            completed_depth: 0,
            score_scope: None,
            cpu_observation: None,
            perspective: Color::White,
            critical: false,
        };
        let records = vec![
            PalsRecordToken {
                record_id: 1,
                revision: 11,
                critical: false,
                features: [0.; RECORD_FEATURES],
            },
            PalsRecordToken {
                record_id: 2,
                revision: 22,
                critical: true,
                features: [0.; RECORD_FEATURES],
            },
        ];
        let source = vec![raw(11, Some(99)), raw(22, Some(11))];
        let mut lines = vec![
            PalsRecordLine {
                moves: vec![],
                parent: None,
                supersedes: None,
                parent_required: false,
                supersedes_required: false,
            },
            PalsRecordLine {
                moves: vec![],
                parent: Some(0),
                supersedes: None,
                parent_required: true,
                supersedes_required: false,
            },
        ];
        let (facts, omitted) = raw_relationships(&records, &lines, &source).unwrap();
        assert_eq!(omitted, 1);
        assert_eq!(facts[0].parent_revision, Some(99));
        assert!(facts[0].parent_omitted);
        assert!(!facts[1].parent_omitted);
        lines[1].parent = None;
        assert!(raw_relationships(&records, &lines, &source).is_err());
        lines[1].parent_required = false;
        assert!(raw_relationships(&records, &lines, &source).is_err());
        lines[1].parent = Some(1);
        assert!(raw_relationships(&records, &lines, &source).is_err());
        lines[1].parent = Some(0);
        let mut wrong_source = source;
        wrong_source[1].revision = 33;
        assert!(raw_relationships(&records, &lines, &wrong_source).is_err());
    }
    #[test]
    fn empty_actual_scan_records_observed_zero_without_labels() {
        let result = coverage(&[], PalsModelProfile::FullLineInteractionV2).unwrap();
        assert_eq!(
            result.current_inputs,
            PalsObservedCountV2::Observed { value: 0 }
        );
        assert_eq!(
            result.verifier_supported_utility,
            PalsObservedCountV2::Observed { value: 0 }
        );
        assert!(result.provenance.is_empty());
    }
    #[test]
    fn sidecar_profile_discriminates_v1_and_v2_without_reinterpreting_bytes() {
        let mut sidecar = PalsNativeInputSidecar {
            version: "rz-pals-native-input-sidecar/1".into(),
            input_sha256: "01".repeat(32),
            encoding_sha256: "02".repeat(32),
            encoder_source_sha256: "03".repeat(32),
            model_epoch_kind: "frozen_model_epoch".into(),
            canonical_tensor_sha256: "04".repeat(32),
            tensor_json: "{}".into(),
            tensor_sha256: "05".repeat(32),
            record_sources: vec![],
            model_profile: None,
            encoding_profile: None,
            sha256: String::new(),
        };
        let legacy = sidecar.digest().unwrap();
        sidecar.version = "rz-pals-native-input-sidecar/2".into();
        assert!(sidecar.digest().is_err());
        sidecar.model_profile = Some("full_line_interaction_v2".into());
        sidecar.encoding_profile = Some("rovezero.pals-board-records.v2".into());
        assert_ne!(legacy, sidecar.digest().unwrap());
    }
}
