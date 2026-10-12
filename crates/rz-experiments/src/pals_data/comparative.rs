//! Separate, float-free comparative metadata. This is not checker evidence,
//! Rules proof, a target producer, tensor collation or training admission.
//!
//! Existing input/raw/split/current/producer identities are never rewritten.
//! Independent pins must be supplied by their owners, not copied from this
//! overlay. Actual checker admission and the fixed recipe's utility semantics
//! remain mandatory follow-up work even for a declared completed preference.
use super::*;

pub const COMPARATIVE_DOMAIN: &str = "rz-pals-comparative-overlay/1";
pub const COMPARATIVE_CHECKER_DOMAIN: &str = "rz-pals-comparative-checker/1";
const MAX_PAIRS: usize = 4096;
const MAX_ACTUAL_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComparativeQueryBoundary {
    CandidatePairMetadataOnly,
    RequiresAuxiliaryAdapter,
}
pub fn query_boundary(kind: &str) -> Result<ComparativeQueryBoundary, ManifestError> {
    match kind {
        "proposer_candidates" | "critic_responses" => {
            Ok(ComparativeQueryBoundary::CandidatePairMetadataOnly)
        }
        "divergence" | "critic_divergences" => {
            Ok(ComparativeQueryBoundary::RequiresAuxiliaryAdapter)
        }
        _ => Err(ManifestError::Integrity(
            "unsupported comparative query".into(),
        )),
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparativeBytePin {
    pub bytes: u64,
    pub sha256: String,
}
impl ComparativeBytePin {
    fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            (1..=crate::MAX_MANIFEST_BYTES as u64).contains(&self.bytes) && sha(&self.sha256),
            "comparative actual byte pin",
        )
    }
    fn matches(&self, actual: &str) -> Result<(), ManifestError> {
        self.validate()?;
        ensure(
            self.bytes == actual.len() as u64 && self.sha256 == digest(actual.as_bytes()),
            "comparative actual prepared bytes mismatch",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparativeParentPins {
    pub receipt_artifact: ComparativeBytePin,
    pub raw_dataset_sha256: String,
    pub split_sha256: String,
    pub current_view_sha256: String,
    pub producer_roster_sha256: String,
    pub producer_envelope_sha256: String,
}
impl ComparativeParentPins {
    fn validate(&self) -> Result<(), ManifestError> {
        self.receipt_artifact.validate()?;
        ensure(
            [
                &self.raw_dataset_sha256,
                &self.split_sha256,
                &self.current_view_sha256,
                &self.producer_roster_sha256,
                &self.producer_envelope_sha256,
            ]
            .into_iter()
            .all(|value| sha(value)),
            "comparative parent pins",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparativeCriterionPin {
    /// An independently fixed recipe, not a rule inferred from raw CPU scores.
    pub recipe_id: String,
    pub recipe_sha256: String,
}
impl ComparativeCriterionPin {
    fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            identifier(&self.recipe_id) && sha(&self.recipe_sha256),
            "comparative fixed criterion",
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ComparativeSide {
    White,
    Black,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ComparativePerspective {
    CapturedSideToMove,
    White,
    Black,
}

/// Supplied from the existing checked current view. No new label selector is
/// implemented here. A None label remains a legitimate current unlabeled row.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparativeCurrentAnchor {
    pub input_sha256: String,
    pub label_sha256: Option<String>,
    pub game_id: String,
    pub role: PalsDataRole,
    pub rules_state_sha256: String,
    pub rules_history_sha256: String,
    pub encoding_sha256: String,
    pub source: PalsInputSource,
    pub frozen_epoch: u64,
    pub input_revision: u64,
    pub side_to_move: ComparativeSide,
    pub legal_moves: Vec<u16>,
}
impl ComparativeCurrentAnchor {
    pub fn from_current_record(record: &PalsLearningRecord) -> Result<Self, ManifestError> {
        record.validate()?;
        let snapshot = record.input.snapshot();
        let result = Self {
            input_sha256: record.input.sha256().into(),
            label_sha256: record.label_digest()?,
            game_id: snapshot.game_id.clone(),
            role: snapshot.role,
            rules_state_sha256: snapshot.rules_state_sha256.clone(),
            rules_history_sha256: snapshot.rules_history_sha256.clone(),
            encoding_sha256: snapshot.encoding_sha256.clone(),
            source: snapshot.source.clone(),
            frozen_epoch: snapshot.frozen_epoch,
            input_revision: snapshot.input_revision,
            side_to_move: if snapshot.white_to_move {
                ComparativeSide::White
            } else {
                ComparativeSide::Black
            },
            legal_moves: snapshot.legal_moves.clone(),
        };
        result.validate()?;
        Ok(result)
    }
    fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            sha(&self.input_sha256)
                && self.label_sha256.as_ref().is_none_or(|value| sha(value))
                && identifier(&self.game_id)
                && matches!(self.role, PalsDataRole::Proposer | PalsDataRole::Critic)
                && [
                    &self.rules_state_sha256,
                    &self.rules_history_sha256,
                    &self.encoding_sha256,
                ]
                .into_iter()
                .all(|value| sha(value))
                && (2..=256).contains(&self.legal_moves.len())
                && self.legal_moves.iter().collect::<BTreeSet<_>>().len() == self.legal_moves.len(),
            "comparative current P/C anchor or legal order",
        )?;
        self.source.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparativePreparedAnchor {
    pub current: ComparativeCurrentAnchor,
    pub producer_id: String,
    pub producer_registration_sha256: String,
    pub capture_evidence_sha256: String,
    pub input_json: ComparativeBytePin,
    pub tensor_sidecar_json: ComparativeBytePin,
    pub lineage_json: ComparativeBytePin,
}
impl ComparativePreparedAnchor {
    fn validate(&self) -> Result<(), ManifestError> {
        self.current.validate()?;
        ensure(
            identifier(&self.producer_id)
                && sha(&self.producer_registration_sha256)
                && sha(&self.capture_evidence_sha256),
            "comparative producer anchor",
        )?;
        self.input_json.validate()?;
        self.tensor_sidecar_json.validate()?;
        self.lineage_json.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparativeCheckerPin {
    pub registration_sha256: String,
    pub source: PalsInputSource,
    pub search_implementation_sha256: String,
    pub value_semantics_sha256: String,
    pub profile: String,
    /// Exact shared conditions, excluding the explicitly typed candidate mask.
    pub search_conditions: String,
    pub horizon: u16,
    pub node_budget: u64,
    pub perspective: ComparativePerspective,
}
impl ComparativeCheckerPin {
    fn validate(&self) -> Result<(), ManifestError> {
        self.source.validate()?;
        ensure(
            matches!(self.source, PalsInputSource::OwnCpu { .. })
                && [
                    &self.registration_sha256,
                    &self.search_implementation_sha256,
                    &self.value_semantics_sha256,
                ]
                .into_iter()
                .all(|value| sha(value))
                && identifier(&self.profile)
                && !self.search_conditions.is_empty()
                && self.search_conditions.len() <= 2048
                && !self.search_conditions.chars().any(char::is_control)
                && (1..=256).contains(&self.horizon)
                && self.node_budget > 0,
            "comparative shared checker namespace",
        )
    }
    pub fn namespace_sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        metadata_digest(COMPARATIVE_CHECKER_DOMAIN, self)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparativeTaskConditions {
    pub checker_namespace_sha256: String,
    pub rules_state_sha256: String,
    pub rules_history_sha256: String,
    pub frozen_epoch: u64,
    pub input_revision: u64,
    pub profile: String,
    pub search_conditions: String,
    pub horizon: u16,
    pub node_budget: u64,
    pub perspective: ComparativePerspective,
}
impl ComparativeTaskConditions {
    pub fn for_anchor(
        current: &ComparativeCurrentAnchor,
        checker: &ComparativeCheckerPin,
    ) -> Result<Self, ManifestError> {
        current.validate()?;
        Ok(Self {
            checker_namespace_sha256: checker.namespace_sha256()?,
            rules_state_sha256: current.rules_state_sha256.clone(),
            rules_history_sha256: current.rules_history_sha256.clone(),
            frozen_epoch: current.frozen_epoch,
            input_revision: current.input_revision,
            profile: checker.profile.clone(),
            search_conditions: checker.search_conditions.clone(),
            horizon: checker.horizon,
            node_budget: checker.node_budget,
            perspective: checker.perspective,
        })
    }
}

/// Only a single candidate restriction is supported. Full TaskKey equality is
/// deliberately not required: the two candidate root masks and task IDs differ.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComparativeRestriction {
    CandidateOnly { root_moves: Vec<u16> },
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ComparativeCompletion {
    Completed,
    Partial,
    Cancelled,
    Unknown,
    Missing,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparativeCandidateCheck {
    pub candidate: u16,
    pub task_id: String,
    pub conditions: ComparativeTaskConditions,
    pub restriction: ComparativeRestriction,
    pub completion: ComparativeCompletion,
    pub completed_depth: u16,
    pub evidence_id: Option<String>,
    /// Actual bytes must be checked by the future checker adapter. A declared
    /// pin, status or depth cannot prove that the requested check ran.
    pub evidence_artifact: Option<ComparativeBytePin>,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum ComparativePairKind {
    ProposerCandidates,
    CriticResponses,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ComparativePreference {
    Left,
    Right,
    Tie,
    Masked,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparativePair {
    pub pair_id: String,
    pub input_sha256: String,
    pub kind: ComparativePairKind,
    /// Two distinct moves, in the captured legal list's order. Continuations
    /// and divergence slots cannot masquerade as ordinary policy candidates.
    pub candidates: [u16; 2],
    pub checks: [ComparativeCandidateCheck; 2],
    pub preference: ComparativePreference,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparativeOverlayBody {
    pub version: String,
    pub parent: ComparativeParentPins,
    pub criterion: ComparativeCriterionPin,
    pub checker: ComparativeCheckerPin,
    pub prepared_inputs: Vec<ComparativePreparedAnchor>,
    pub pairs: Vec<ComparativePair>,
}
impl ComparativeOverlayBody {
    fn normalized(&self) -> Result<Self, ManifestError> {
        ensure(
            self.version == COMPARATIVE_DOMAIN
                && (1..=MAX_PAIRS).contains(&self.pairs.len())
                && (1..=MAX_PAIRS).contains(&self.prepared_inputs.len()),
            "comparative overlay version/extent",
        )?;
        self.parent.validate()?;
        self.criterion.validate()?;
        self.checker.validate()?;
        let mut result = self.clone();
        result
            .prepared_inputs
            .sort_by(|a, b| a.current.input_sha256.cmp(&b.current.input_sha256));
        result.pairs.sort_by(|a, b| a.pair_id.cmp(&b.pair_id));
        let mut anchors = BTreeMap::new();
        for anchor in &result.prepared_inputs {
            anchor.validate()?;
            ensure(
                anchors
                    .insert(anchor.current.input_sha256.as_str(), anchor)
                    .is_none(),
                "duplicate comparative prepared input",
            )?;
        }
        let mut pair_ids = BTreeSet::new();
        let mut candidate_pairs = BTreeSet::new();
        let mut used = BTreeSet::new();
        // Reusing an ID for a different candidate/task statement is ambiguous.
        let mut tasks = BTreeMap::new();
        let mut evidence = BTreeMap::new();
        for pair in &result.pairs {
            ensure(
                identifier(&pair.pair_id) && sha(&pair.input_sha256),
                "comparative pair identity",
            )?;
            ensure(
                pair_ids.insert(&pair.pair_id),
                "duplicate comparative pair ID",
            )?;
            let anchor = anchors.get(pair.input_sha256.as_str()).ok_or_else(|| {
                ManifestError::Integrity("missing comparative prepared input".into())
            })?;
            let current = &anchor.current;
            ensure(
                matches!(
                    (pair.kind, current.role),
                    (
                        ComparativePairKind::ProposerCandidates,
                        PalsDataRole::Proposer
                    ) | (ComparativePairKind::CriticResponses, PalsDataRole::Critic)
                ),
                "comparative pair role/auxiliary adapter required",
            )?;
            let positions: Vec<_> = pair
                .candidates
                .iter()
                .map(|candidate| {
                    current
                        .legal_moves
                        .iter()
                        .position(|legal| legal == candidate)
                })
                .collect();
            ensure(
                matches!((positions[0], positions[1]), (Some(a), Some(b)) if a < b),
                "comparative candidates duplicate/illegal/out of captured order",
            )?;
            ensure(
                candidate_pairs.insert((&pair.input_sha256, pair.kind, pair.candidates)),
                "duplicate comparative candidate pair",
            )?;
            let conditions = ComparativeTaskConditions::for_anchor(current, &result.checker)?;
            for (candidate, check) in pair.candidates.iter().zip(&pair.checks) {
                ensure(
                    check.candidate == *candidate
                        && identifier(&check.task_id)
                        && check.conditions == conditions
                        && matches!(&check.restriction,
                            ComparativeRestriction::CandidateOnly { root_moves } if root_moves.as_slice() == [*candidate])
                        && check.completed_depth <= 256,
                    "comparative checker conditions/candidate restriction mismatch",
                )?;
                ensure(
                    check.evidence_id.as_ref().is_none_or(|id| identifier(id))
                        && check.evidence_id.is_some() == check.evidence_artifact.is_some(),
                    "comparative evidence identity/pin mismatch",
                )?;
                if let Some(artifact) = &check.evidence_artifact {
                    artifact.validate()?;
                }
                if check.completion == ComparativeCompletion::Completed {
                    ensure(
                        check.completed_depth >= conditions.horizon
                            && check.evidence_artifact.is_some(),
                        "completed comparative check lacks horizon/evidence",
                    )?;
                }
                if check.completion == ComparativeCompletion::Missing {
                    ensure(
                        check.completed_depth == 0 && check.evidence_artifact.is_none(),
                        "missing comparative evidence has fabricated completion",
                    )?;
                }
                if let Some(prior) = tasks.insert(&check.task_id, (&pair.input_sha256, check)) {
                    ensure(
                        prior == (&pair.input_sha256, check),
                        "comparative task ID reused across statements",
                    )?;
                }
                if let Some(id) = &check.evidence_id {
                    if let Some(prior) = evidence.insert(id, &check.task_id) {
                        ensure(
                            prior == &check.task_id,
                            "comparative evidence ID reused by different tasks",
                        )?;
                    }
                }
            }
            ensure(
                pair.checks[0].task_id != pair.checks[1].task_id,
                "comparative candidate task IDs must differ",
            )?;
            ensure(
                pair.preference == ComparativePreference::Masked
                    || pair
                        .checks
                        .iter()
                        .all(|check| check.completion == ComparativeCompletion::Completed),
                "unresolved comparative pair must remain masked",
            )?;
            used.insert(pair.input_sha256.as_str());
        }
        ensure(
            used.len() == anchors.len(),
            "unused comparative prepared input",
        )?;
        Ok(result)
    }
}

/// New canonical metadata only: recursively sorted object keys, compact UTF-8,
/// array order retained, strict u64/string/null. No scores, floats or booleans.
pub fn canonical_comparative_metadata<T: Serialize>(
    domain: &str,
    value: &T,
) -> Result<Vec<u8>, ManifestError> {
    let value = serde_json::to_value(value).map_err(|e| ManifestError::Integrity(e.to_string()))?;
    fn validate(value: &serde_json::Value) -> Result<(), ManifestError> {
        match value {
            serde_json::Value::Null | serde_json::Value::String(_) => Ok(()),
            serde_json::Value::Number(value) => {
                ensure(value.as_u64().is_some(), "comparative canonical strict u64")
            }
            serde_json::Value::Array(values) => values.iter().try_for_each(validate),
            serde_json::Value::Object(values) => values.values().try_for_each(validate),
            _ => Err(ManifestError::Integrity(
                "comparative canonical float/bool unsupported".into(),
            )),
        }
    }
    ensure(identifier(domain), "comparative canonical domain")?;
    validate(&value)?;
    let bytes = serde_json::to_vec(&(domain, sorted_json(value)))
        .map_err(|e| ManifestError::Integrity(e.to_string()))?;
    ensure(
        bytes.len() <= crate::MAX_MANIFEST_BYTES,
        "comparative canonical byte limit",
    )?;
    Ok(bytes)
}
fn metadata_digest<T: Serialize>(domain: &str, value: &T) -> Result<String, ManifestError> {
    Ok(digest(&canonical_comparative_metadata(domain, value)?))
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrozenComparativeOverlay {
    overlay: ComparativeOverlayBody,
    sha256: String,
}
impl FrozenComparativeOverlay {
    pub fn seal(body: ComparativeOverlayBody) -> Result<Self, ManifestError> {
        let overlay = body.normalized()?;
        let sha256 = metadata_digest(COMPARATIVE_DOMAIN, &overlay)?;
        let result = Self { overlay, sha256 };
        ensure(
            serde_json::to_vec(&result)
                .map_err(|e| ManifestError::Integrity(e.to_string()))?
                .len()
                <= crate::MAX_MANIFEST_BYTES,
            "comparative serialized byte limit",
        )?;
        Ok(result)
    }
    pub fn body(&self) -> &ComparativeOverlayBody {
        &self.overlay
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let declared: Self = decode_json(input)?;
        let expected = Self::seal(declared.overlay)?;
        ensure(
            declared.sha256 == expected.sha256,
            "comparative overlay seal mismatch",
        )?;
        Ok(expected)
    }
    pub fn to_json(&self) -> Result<String, ManifestError> {
        ensure(
            self == &Self::seal(self.overlay.clone())?,
            "comparative overlay seal mismatch",
        )?;
        serde_json::to_string(self).map_err(|e| ManifestError::Integrity(e.to_string()))
    }
}

/// Independent expected pins. Shape or equality alone does not prove who
/// registered the source, performed a checker task or audited the base dataset.
#[derive(Clone, Debug)]
pub struct ComparativeExpectedPins {
    pub parent: ComparativeParentPins,
    pub criterion: ComparativeCriterionPin,
    pub checker: ComparativeCheckerPin,
    pub prepared_inputs: Vec<ComparativePreparedAnchor>,
}
/// Exact captured row bytes; these exclude the JSONL newline. Sidecar/lineage
/// byte matching does not replace the native tensor or checked query adapter.
pub struct ComparativePreparedBytes<'a> {
    pub input_sha256: &'a str,
    pub input_json: &'a str,
    pub tensor_sidecar_json: &'a str,
    pub lineage_json: &'a str,
}
#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
pub struct ComparativeMetadataAudit {
    pub scope: String,
    pub requires_actual_checker_admission: bool,
    pub overlay_sha256: String,
    pub current_view_sha256: String,
    pub prepared_inputs: u64,
    pub pairs: u64,
    /// Declarations only, never a count of admitted positive training targets.
    pub declared_preferences: u64,
    pub masked_pairs: u64,
}

pub fn audit_metadata(
    overlay: &FrozenComparativeOverlay,
    expected: &ComparativeExpectedPins,
    records: &[PalsLearningRecord],
    current_view: &PalsCurrentLabelView,
    actual_parent_receipt: &str,
    actual_prepared: &[ComparativePreparedBytes<'_>],
) -> Result<ComparativeMetadataAudit, ManifestError> {
    let overlay = FrozenComparativeOverlay::from_json(&overlay.to_json()?)?;
    let body = overlay.body();
    let mut expected_inputs = expected.prepared_inputs.clone();
    expected_inputs.sort_by(|a, b| a.current.input_sha256.cmp(&b.current.input_sha256));
    ensure(
        body.parent == expected.parent
            && body.criterion == expected.criterion
            && body.checker == expected.checker
            && body.prepared_inputs == expected_inputs,
        "comparative independent expected pins mismatch",
    )?;
    body.parent
        .receipt_artifact
        .matches(actual_parent_receipt)?;
    // Reuse the existing selector rather than implementing another DG05 chain.
    // This does not replace the independently completed owned-source/split audit.
    ensure(
        &current_label_view(records)? == current_view
            && current_view.sha256() == expected.parent.current_view_sha256,
        "comparative stale current view",
    )?;
    let selected: BTreeMap<_, _> = current_view
        .indices()
        .iter()
        .map(|&index| {
            let row = &records[index];
            (row.input.sha256(), row)
        })
        .collect();
    ensure(
        actual_prepared.len() == body.prepared_inputs.len(),
        "comparative actual prepared extent",
    )?;
    let mut actual = BTreeMap::new();
    let mut byte_total = 0usize;
    for rows in actual_prepared {
        byte_total = byte_total
            .checked_add(rows.input_json.len())
            .and_then(|n| n.checked_add(rows.tensor_sidecar_json.len()))
            .and_then(|n| n.checked_add(rows.lineage_json.len()))
            .ok_or_else(|| ManifestError::Integrity("comparative prepared byte overflow".into()))?;
        ensure(
            byte_total <= MAX_ACTUAL_BYTES,
            "comparative aggregate prepared byte limit",
        )?;
        ensure(
            sha(rows.input_sha256) && actual.insert(rows.input_sha256, rows).is_none(),
            "duplicate comparative actual prepared input",
        )?;
    }
    for anchor in &body.prepared_inputs {
        let identity = anchor.current.input_sha256.as_str();
        let row = selected.get(identity).ok_or_else(|| {
            ManifestError::Integrity("comparative input is not a selected current row".into())
        })?;
        ensure(
            ComparativeCurrentAnchor::from_current_record(row)? == anchor.current,
            "comparative current label/input anchor mismatch",
        )?;
        let rows = actual.get(identity).ok_or_else(|| {
            ManifestError::Integrity("missing comparative actual prepared bytes".into())
        })?;
        anchor.input_json.matches(rows.input_json)?;
        anchor
            .tensor_sidecar_json
            .matches(rows.tensor_sidecar_json)?;
        anchor.lineage_json.matches(rows.lineage_json)?;
        let captured: PalsFrozenInput = decode_json(rows.input_json)?;
        captured.verify()?;
        ensure(
            captured == row.input,
            "comparative prepared input belongs to a different current row",
        )?;
    }
    let declared_preferences = body
        .pairs
        .iter()
        .filter(|pair| pair.preference != ComparativePreference::Masked)
        .count() as u64;
    Ok(ComparativeMetadataAudit {
        scope: "metadata_only".into(),
        requires_actual_checker_admission: true,
        overlay_sha256: overlay.sha256().into(),
        current_view_sha256: current_view.sha256().into(),
        prepared_inputs: body.prepared_inputs.len() as u64,
        pairs: body.pairs.len() as u64,
        declared_preferences,
        masked_pairs: body.pairs.len() as u64 - declared_preferences,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Python test_comparative.py asserts this exact same literal. It tests the
    // canonical protocol, not a base/current/checker admission fixture.
    const CANONICAL_VECTOR: &str = r#"["rz-pals-comparative-overlay/1",{"criterion":{"recipe_id":"후보 비교","recipe_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"frozen_epoch":9007199254740993,"legal_moves":[1292,1804],"optional":null}]"#;
    const CHECKER_VECTOR: &str = r#"["rz-pals-comparative-checker/1",{"horizon":2,"node_budget":9007199254740993,"perspective":"captured_side_to_move","profile":"fixture-depth-profile","registration_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","search_conditions":"공통 조건; roots excluded","search_implementation_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","source":{"cpu_binary_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","evaluator_configuration_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","kind":"own_cpu","model_weights_sha256":null},"value_semantics_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"}]"#;

    fn hash(name: &str) -> String {
        digest(name.as_bytes())
    }
    fn pin(actual: &str) -> ComparativeBytePin {
        ComparativeBytePin {
            bytes: actual.len() as u64,
            sha256: hash(actual),
        }
    }
    fn cpu_source() -> PalsInputSource {
        PalsInputSource::OwnCpu {
            cpu_binary_sha256: hash("fixture-cpu-binary"),
            evaluator_configuration_sha256: hash("fixture-cpu-configuration"),
            model_weights_sha256: None,
        }
    }
    struct Fixture {
        body: ComparativeOverlayBody,
        records: Vec<PalsLearningRecord>,
        receipt: String,
        input_json: String,
        tensor_json: String,
        lineage_json: String,
    }
    impl Fixture {
        fn new(role: PalsDataRole) -> Self {
            let record = PalsLearningRecord {
                input: PalsInputSnapshot {
                    game_id: "game".into(),
                    opening_id: "opening".into(),
                    line_genealogy_id: "line".into(),
                    position_command: "position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1".into(),
                    board_fen: "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1".into(),
                    actual_history: vec![],
                    rules_state_sha256: hash("state"),
                    rules_history_sha256: hash("history"),
                    transposition_sha256: hash("transposition"),
                    encoding_sha256: hash("encoding"),
                    source: cpu_source(),
                    frozen_epoch: 0,
                    input_revision: 3,
                    capture_sequence: 8,
                    white_to_move: true,
                    role,
                    legal_moves: vec![1292, 1804, 259],
                    public_records: vec![],
                }
                .seal()
                .unwrap(),
                future_label: None,
                verifier_private: None,
            };
            let current = ComparativeCurrentAnchor::from_current_record(&record).unwrap();
            let input_json = serde_json::to_string(&record.input).unwrap();
            // Metadata fixtures only: no native tensor/checker was executed.
            let tensor_json: String = r#"{"fixture":"tensor sidecar"}"#.into();
            let lineage_json: String = r#"{"fixture":"lineage"}"#.into();
            let receipt: String = r#"{"fixture":"independently pinned base receipt"}"#.into();
            let checker = ComparativeCheckerPin {
                registration_sha256: hash("checker-registration"),
                source: cpu_source(),
                search_implementation_sha256: hash("search-implementation"),
                value_semantics_sha256: hash("value-semantics"),
                profile: "fixture-depth-profile".into(),
                search_conditions: "fixture common conditions; roots excluded".into(),
                horizon: 2,
                node_budget: 9007199254740993,
                perspective: ComparativePerspective::CapturedSideToMove,
            };
            let conditions = ComparativeTaskConditions::for_anchor(&current, &checker).unwrap();
            let candidates = [1292, 1804];
            let checks = candidates.map(|candidate| ComparativeCandidateCheck {
                candidate,
                task_id: format!("task-{candidate}"),
                conditions: conditions.clone(),
                restriction: ComparativeRestriction::CandidateOnly {
                    root_moves: vec![candidate],
                },
                completion: ComparativeCompletion::Completed,
                completed_depth: 2,
                evidence_id: Some(format!("evidence-{candidate}")),
                evidence_artifact: Some(pin(&format!("fixture-evidence-{candidate}"))),
            });
            let parent = ComparativeParentPins {
                receipt_artifact: pin(&receipt),
                raw_dataset_sha256: hash("raw"),
                split_sha256: hash("split"),
                current_view_sha256: current_label_view(std::slice::from_ref(&record))
                    .unwrap()
                    .sha256()
                    .into(),
                producer_roster_sha256: hash("roster"),
                producer_envelope_sha256: hash("envelope"),
            };
            let body = ComparativeOverlayBody {
                version: COMPARATIVE_DOMAIN.into(),
                parent,
                criterion: ComparativeCriterionPin {
                    recipe_id: "후보 비교".into(),
                    recipe_sha256: hash("fixed-recipe"),
                },
                checker,
                prepared_inputs: vec![ComparativePreparedAnchor {
                    current: current.clone(),
                    producer_id: "input-producer".into(),
                    producer_registration_sha256: hash("producer-registration"),
                    capture_evidence_sha256: hash("prepared-journal-entry"),
                    input_json: pin(&input_json),
                    tensor_sidecar_json: pin(&tensor_json),
                    lineage_json: pin(&lineage_json),
                }],
                pairs: vec![ComparativePair {
                    pair_id: "pair".into(),
                    input_sha256: current.input_sha256,
                    kind: if role == PalsDataRole::Proposer {
                        ComparativePairKind::ProposerCandidates
                    } else {
                        ComparativePairKind::CriticResponses
                    },
                    candidates,
                    checks,
                    preference: ComparativePreference::Left,
                }],
            };
            Self {
                body,
                records: vec![record],
                receipt,
                input_json,
                tensor_json,
                lineage_json,
            }
        }
        fn expected(&self) -> ComparativeExpectedPins {
            ComparativeExpectedPins {
                parent: self.body.parent.clone(),
                criterion: self.body.criterion.clone(),
                checker: self.body.checker.clone(),
                prepared_inputs: self.body.prepared_inputs.clone(),
            }
        }
        fn actual(&self) -> Vec<ComparativePreparedBytes<'_>> {
            vec![ComparativePreparedBytes {
                input_sha256: self.records[0].input.sha256(),
                input_json: &self.input_json,
                tensor_sidecar_json: &self.tensor_json,
                lineage_json: &self.lineage_json,
            }]
        }
        fn audit(
            &self,
            overlay: &FrozenComparativeOverlay,
            expected: &ComparativeExpectedPins,
        ) -> Result<ComparativeMetadataAudit, ManifestError> {
            audit_metadata(
                overlay,
                expected,
                &self.records,
                &current_label_view(&self.records)?,
                &self.receipt,
                &self.actual(),
            )
        }
    }

    #[test]
    fn shared_literal_sorted_utf8_and_exact_u64_canonical_vector() {
        let value = serde_json::json!({
            "optional": null, "legal_moves": [1292, 1804], "frozen_epoch": 9007199254740993u64,
            "criterion": {"recipe_sha256": "a".repeat(64), "recipe_id": "후보 비교"}
        });
        assert_eq!(
            canonical_comparative_metadata(COMPARATIVE_DOMAIN, &value).unwrap(),
            CANONICAL_VECTOR.as_bytes()
        );
        assert_eq!(
            metadata_digest(COMPARATIVE_DOMAIN, &value).unwrap(),
            hash(CANONICAL_VECTOR)
        );
        for bad in [
            serde_json::json!(true),
            serde_json::json!(-1),
            serde_json::json!(1.0),
        ] {
            assert!(canonical_comparative_metadata(COMPARATIVE_DOMAIN, &bad).is_err());
        }
        let checker = ComparativeCheckerPin {
            registration_sha256: "a".repeat(64),
            source: PalsInputSource::OwnCpu {
                cpu_binary_sha256: "c".repeat(64),
                evaluator_configuration_sha256: "d".repeat(64),
                model_weights_sha256: None,
            },
            search_implementation_sha256: "b".repeat(64),
            value_semantics_sha256: "e".repeat(64),
            profile: "fixture-depth-profile".into(),
            search_conditions: "공통 조건; roots excluded".into(),
            horizon: 2,
            node_budget: 9007199254740993,
            perspective: ComparativePerspective::CapturedSideToMove,
        };
        assert_eq!(
            canonical_comparative_metadata(COMPARATIVE_CHECKER_DOMAIN, &checker).unwrap(),
            CHECKER_VECTOR.as_bytes()
        );
        assert_eq!(checker.namespace_sha256().unwrap(), hash(CHECKER_VECTOR));
    }
    #[test]
    fn proposer_and_critic_metadata_never_admit_positive_training() {
        for role in [PalsDataRole::Proposer, PalsDataRole::Critic] {
            let fixture = Fixture::new(role);
            let overlay = FrozenComparativeOverlay::seal(fixture.body.clone()).unwrap();
            assert_eq!(
                FrozenComparativeOverlay::from_json(&overlay.to_json().unwrap()).unwrap(),
                overlay
            );
            let audit = fixture.audit(&overlay, &fixture.expected()).unwrap();
            assert_eq!(
                (
                    audit.scope.as_str(),
                    audit.declared_preferences,
                    audit.masked_pairs
                ),
                ("metadata_only", 1, 0)
            );
            assert!(audit.requires_actual_checker_admission);
            let json = serde_json::to_value(audit).unwrap();
            assert!(json.get("training_admitted").is_none());
            assert!(json.get("checker_verified").is_none());
            assert_ne!(
                overlay.body().pairs[0].checks[0].task_id,
                overlay.body().pairs[0].checks[1].task_id
            );
            assert_ne!(
                overlay.body().pairs[0].checks[0].restriction,
                overlay.body().pairs[0].checks[1].restriction
            );
            assert!(
                overlay.body().prepared_inputs[0]
                    .current
                    .label_sha256
                    .is_none()
            );
        }
    }
    #[test]
    fn unresolved_canceled_partial_unknown_missing_pairs_remain_masked() {
        for status in [
            ComparativeCompletion::Partial,
            ComparativeCompletion::Cancelled,
            ComparativeCompletion::Unknown,
            ComparativeCompletion::Missing,
        ] {
            let fixture = Fixture::new(PalsDataRole::Critic);
            let mut body = fixture.body.clone();
            body.pairs[0].checks[0].completion = status;
            if status == ComparativeCompletion::Missing {
                body.pairs[0].checks[0].completed_depth = 0;
                body.pairs[0].checks[0].evidence_artifact = None;
                body.pairs[0].checks[0].evidence_id = None;
            }
            assert!(FrozenComparativeOverlay::seal(body.clone()).is_err());
            body.pairs[0].preference = ComparativePreference::Masked;
            let overlay = FrozenComparativeOverlay::seal(body).unwrap();
            let audit = fixture.audit(&overlay, &fixture.expected()).unwrap();
            assert_eq!((audit.declared_preferences, audit.masked_pairs), (0, 1));
            assert!(audit.requires_actual_checker_admission);
        }
        let fixture = Fixture::new(PalsDataRole::Proposer);
        let mut body = fixture.body;
        body.pairs[0].checks[0].completed_depth = 1;
        assert!(FrozenComparativeOverlay::seal(body).is_err());
    }
    fn rejects(change: fn(&mut ComparativeOverlayBody)) {
        let mut body = Fixture::new(PalsDataRole::Proposer).body;
        change(&mut body);
        assert!(FrozenComparativeOverlay::seal(body).is_err());
    }
    #[test]
    fn rejects_namespace_profile_horizon_perspective_state_epoch_revision_and_masks() {
        rejects(|b| b.pairs[0].checks[0].conditions.checker_namespace_sha256 = hash("other"));
        rejects(|b| b.pairs[0].checks[0].conditions.profile = "other".into());
        rejects(|b| {
            b.pairs[0].checks[0]
                .conditions
                .search_conditions
                .push_str("changed")
        });
        rejects(|b| b.pairs[0].checks[0].conditions.horizon += 1);
        rejects(|b| b.pairs[0].checks[0].conditions.node_budget += 1);
        rejects(|b| b.pairs[0].checks[0].conditions.perspective = ComparativePerspective::White);
        rejects(|b| b.pairs[0].checks[0].conditions.rules_state_sha256 = hash("other"));
        rejects(|b| b.pairs[0].checks[0].conditions.rules_history_sha256 = hash("other"));
        rejects(|b| b.pairs[0].checks[0].conditions.frozen_epoch += 1);
        rejects(|b| b.pairs[0].checks[0].conditions.input_revision += 1);
        rejects(|b| {
            b.pairs[0].checks[0].restriction = ComparativeRestriction::CandidateOnly {
                root_moves: vec![1292, 1804],
            }
        });
        rejects(|b| b.pairs[0].checks[1].evidence_id = b.pairs[0].checks[0].evidence_id.clone());
        rejects(|b| b.pairs[0].checks[1].task_id = b.pairs[0].checks[0].task_id.clone());
    }
    #[test]
    fn rejects_illegal_duplicate_reordered_pairs_and_auxiliary_queries() {
        rejects(|b| b.pairs[0].candidates = [1292, 65535]);
        rejects(|b| b.pairs[0].candidates = [1292, 1292]);
        rejects(|b| b.pairs[0].candidates.reverse());
        rejects(|b| b.pairs[0].kind = ComparativePairKind::CriticResponses);
        rejects(|b| {
            let mut pair = b.pairs[0].clone();
            pair.pair_id = "other-pair".into();
            b.pairs.push(pair);
        });
        rejects(|b| b.prepared_inputs.push(b.prepared_inputs[0].clone()));
        assert_eq!(
            query_boundary("critic_divergences").unwrap(),
            ComparativeQueryBoundary::RequiresAuxiliaryAdapter
        );
        let fixture = Fixture::new(PalsDataRole::Critic);
        let json = FrozenComparativeOverlay::seal(fixture.body)
            .unwrap()
            .to_json()
            .unwrap()
            .replace("critic_responses", "critic_divergences");
        assert!(FrozenComparativeOverlay::from_json(&json).is_err());
    }
    #[test]
    fn independent_pins_actual_bytes_and_current_label_mismatch_are_rejected() {
        let fixture = Fixture::new(PalsDataRole::Proposer);
        let overlay = FrozenComparativeOverlay::seal(fixture.body.clone()).unwrap();
        let mut expected = fixture.expected();
        expected.criterion.recipe_sha256 = hash("other-recipe");
        assert!(fixture.audit(&overlay, &expected).is_err());
        expected = fixture.expected();
        expected.parent.producer_envelope_sha256 = hash("other-envelope");
        assert!(fixture.audit(&overlay, &expected).is_err());
        expected = fixture.expected();
        expected.checker.registration_sha256 = hash("other-checker");
        assert!(fixture.audit(&overlay, &expected).is_err());
        expected = fixture.expected();
        expected.prepared_inputs[0].producer_registration_sha256 = hash("other-producer");
        assert!(fixture.audit(&overlay, &expected).is_err());
        let mut bad_bytes = fixture.actual();
        bad_bytes[0].tensor_sidecar_json = "changed";
        assert!(
            audit_metadata(
                &overlay,
                &fixture.expected(),
                &fixture.records,
                &current_label_view(&fixture.records).unwrap(),
                &fixture.receipt,
                &bad_bytes
            )
            .is_err()
        );
        // Even resealed declarations and matching expected pins cannot claim a
        // historical label absent from the actual selected current row.
        let mut body = fixture.body.clone();
        body.prepared_inputs[0].current.label_sha256 = Some(hash("stale-label"));
        let stale = FrozenComparativeOverlay::seal(body.clone()).unwrap();
        expected = fixture.expected();
        expected.prepared_inputs = body.prepared_inputs;
        assert!(fixture.audit(&stale, &expected).is_err());
        assert!(
            audit_metadata(
                &overlay,
                &fixture.expected(),
                &fixture.records,
                &current_label_view(&fixture.records).unwrap(),
                "changed receipt",
                &fixture.actual()
            )
            .is_err()
        );
    }
    #[test]
    fn existing_label_chain_leaf_is_used_without_merging_old_labels() {
        let mut fixture = Fixture::new(PalsDataRole::Proposer);
        let original = fixture.records[0].clone();
        let make_label = |sequence, predecessor| PalsFutureLabel {
            observed_sequence: sequence,
            provenance: PalsTargetProvenance::OwnedCpu {
                engine_sha256: hash("fixture-cpu-binary"),
                profile_sha256: hash("profile"),
                task_sha256: hash("task"),
                completed_depth: 2,
                nodes: 3,
                raw_evidence_sha256: hash("raw-evidence"),
            },
            policy: None,
            value_wdl: None,
            white_to_move: true,
            counterexample: None,
            verifier_tasks: None,
            supersedes_label_sha256: predecessor,
        };
        let mut first = original.clone();
        first.future_label = Some(make_label(9, None));
        let mut leaf = original.clone();
        leaf.future_label = Some(make_label(10, first.label_digest().unwrap()));
        fixture.records = vec![leaf.clone(), original, first.clone()];
        fixture.body.parent.current_view_sha256 = current_label_view(&fixture.records)
            .unwrap()
            .sha256()
            .into();
        fixture.body.prepared_inputs[0].current =
            ComparativeCurrentAnchor::from_current_record(&leaf).unwrap();
        let overlay = FrozenComparativeOverlay::seal(fixture.body.clone()).unwrap();
        fixture.audit(&overlay, &fixture.expected()).unwrap();
        fixture.body.prepared_inputs[0].current.label_sha256 = first.label_digest().unwrap();
        let stale = FrozenComparativeOverlay::seal(fixture.body.clone()).unwrap();
        assert!(fixture.audit(&stale, &fixture.expected()).is_err());
        assert_eq!(fixture.records.len(), 3);
    }
    #[test]
    fn sealed_json_rejects_duplicate_keys_unknown_fields_float_bool_and_overflow_u64() {
        let fixture = Fixture::new(PalsDataRole::Proposer);
        let json = FrozenComparativeOverlay::seal(fixture.body)
            .unwrap()
            .to_json()
            .unwrap();
        assert!(
            FrozenComparativeOverlay::from_json(&json.replacen(
                "\"overlay\":",
                "\"unknown\":0,\"overlay\":",
                1
            ))
            .is_err()
        );
        assert!(
            FrozenComparativeOverlay::from_json(&json.replacen(
                "\"version\":",
                "\"version\":\"duplicate\",\"version\":",
                1
            ))
            .is_err()
        );
        for bad in ["9007199254740993.0", "true", "18446744073709551616", "-1"] {
            assert!(
                FrozenComparativeOverlay::from_json(&json.replace("9007199254740993", bad))
                    .is_err()
            );
        }
    }
}
