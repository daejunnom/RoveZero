//! Frozen producer metadata, separate from input/raw/current-view identities.
//!
//! Matching independently supplied pins is not proof of a live producer or
//! actual training admission. Derived private encodings always require their
//! existing query adapter; a well-formed schema SHA cannot replace that check.
use super::*;

pub const ROSTER_DOMAIN: &str = "rz-pals-producer-roster/1";
pub const CAPTURE_DOMAIN: &str = "rz-pals-producer-captures/1";
pub const ENVELOPE_DOMAIN: &str = "rz-pals-producer-envelope/1";
pub const OWNED_SOURCES_DOMAIN: &str = "rz-pals-owned-source-registry/1";

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeModelEpoch {
    EncodingOnlyZero,
    FrozenModelEpoch { sha256: String },
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProducerEncodingPolicy {
    NativeExact {
        encoding_sha256: String,
        encoder_source_sha256: String,
        native_model_epoch: NativeModelEpoch,
    },
    PrivateCheckedDerivedQuery {
        encoding_schema_sha256: String,
        private_encoder_source_sha256: String,
        parent_encoding_sha256: String,
        parent_encoder_source_sha256: String,
    },
}
impl ProducerEncodingPolicy {
    fn validate(&self, source: &PalsInputSource) -> Result<(), ManifestError> {
        match self {
            Self::NativeExact {
                encoding_sha256,
                encoder_source_sha256,
                native_model_epoch,
            } => {
                ensure(
                    sha(encoding_sha256) && sha(encoder_source_sha256),
                    "invalid native encoding pin",
                )?;
                match (source, native_model_epoch) {
                    (PalsInputSource::OwnCpu { .. }, NativeModelEpoch::EncodingOnlyZero) => Ok(()),
                    (
                        PalsInputSource::OwnPals { .. },
                        NativeModelEpoch::FrozenModelEpoch { sha256 },
                    ) => ensure(sha(sha256), "invalid independently registered native epoch"),
                    _ => Err(ManifestError::Integrity(
                        "native epoch/source kind mismatch".into(),
                    )),
                }
            }
            Self::PrivateCheckedDerivedQuery {
                encoding_schema_sha256,
                private_encoder_source_sha256,
                parent_encoding_sha256,
                parent_encoder_source_sha256,
            } => ensure(
                matches!(source, PalsInputSource::OwnPals { .. })
                    && [
                        encoding_schema_sha256,
                        private_encoder_source_sha256,
                        parent_encoding_sha256,
                        parent_encoder_source_sha256,
                    ]
                    .into_iter()
                    .all(|value| sha(value)),
                "invalid private derived encoding pin",
            ),
        }
    }
}

/// Supplied by an independent owner. Deserializing this declaration does not
/// establish that a binary, checkpoint, callback or runtime was actually used.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProducerRegistrationPin {
    pub game_id: String,
    pub producer_id: String,
    pub registration_sha256: String,
    pub source: PalsInputSource,
    pub frozen_epoch: u64,
    pub encoding_policy: ProducerEncodingPolicy,
}
impl ProducerRegistrationPin {
    fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            identifier(&self.game_id)
                && identifier(&self.producer_id)
                && sha(&self.registration_sha256),
            "invalid producer registration identity",
        )?;
        self.source.validate()?;
        self.encoding_policy.validate(&self.source)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProducerRosterBody {
    pub version: String,
    pub game_producers: Vec<ProducerRegistrationPin>,
}
impl ProducerRosterBody {
    fn normalized(&self) -> Result<Self, ManifestError> {
        ensure(
            self.version == ROSTER_DOMAIN
                && !self.game_producers.is_empty()
                && self.game_producers.len() <= MAX_RECORDS,
            "producer roster extent/version",
        )?;
        let mut result = self.clone();
        result
            .game_producers
            .sort_by(|a, b| (&a.game_id, &a.producer_id).cmp(&(&b.game_id, &b.producer_id)));
        let mut previous = None;
        let mut counts = BTreeMap::new();
        for pin in &result.game_producers {
            pin.validate()?;
            let key = (&pin.game_id, &pin.producer_id);
            ensure(
                previous != Some(key),
                "duplicate game/producer registration",
            )?;
            previous = Some(key);
            let count = counts.entry(&pin.game_id).or_insert(0usize);
            *count += 1;
            ensure(*count <= 64, "per-game producer limit")?;
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrozenProducerRoster {
    roster: ProducerRosterBody,
    sha256: String,
}
impl FrozenProducerRoster {
    pub fn seal(body: ProducerRosterBody) -> Result<Self, ManifestError> {
        let roster = body.normalized()?;
        let sha256 = metadata_digest(ROSTER_DOMAIN, &roster)?;
        Ok(Self { roster, sha256 })
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn body(&self) -> &ProducerRosterBody {
        &self.roster
    }
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        let expected = Self::seal(value.roster.clone())?;
        ensure(
            value.sha256 == expected.sha256,
            "producer roster seal mismatch",
        )?;
        Ok(expected)
    }
    pub fn to_json(&self) -> Result<String, ManifestError> {
        ensure(
            self.sha256 == Self::seal(self.roster.clone())?.sha256,
            "producer roster seal mismatch",
        )?;
        serde_json::to_string(self).map_err(|e| ManifestError::Integrity(e.to_string()))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InputProducerBinding {
    pub input_sha256: String,
    pub game_id: String,
    pub producer_id: String,
    pub capture_sequence: u64,
    pub capture_evidence_sha256: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProducerCaptureBody {
    pub version: String,
    pub bindings: Vec<InputProducerBinding>,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProducerCaptureArtifact {
    capture: ProducerCaptureBody,
    sha256: String,
}
impl ProducerCaptureArtifact {
    pub fn seal(mut capture: ProducerCaptureBody) -> Result<Self, ManifestError> {
        ensure(
            capture.version == CAPTURE_DOMAIN
                && !capture.bindings.is_empty()
                && capture.bindings.len() <= MAX_RECORDS,
            "producer capture extent/version",
        )?;
        capture
            .bindings
            .sort_by(|a, b| a.input_sha256.cmp(&b.input_sha256));
        let mut previous = None;
        for binding in &capture.bindings {
            ensure(
                sha(&binding.input_sha256)
                    && identifier(&binding.game_id)
                    && identifier(&binding.producer_id)
                    && sha(&binding.capture_evidence_sha256),
                "invalid producer capture binding",
            )?;
            ensure(
                previous != Some(&binding.input_sha256),
                "duplicate input producer binding",
            )?;
            previous = Some(&binding.input_sha256);
        }
        let sha256 = metadata_digest(CAPTURE_DOMAIN, &capture)?;
        Ok(Self { capture, sha256 })
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn body(&self) -> &ProducerCaptureBody {
        &self.capture
    }
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        let expected = Self::seal(value.capture.clone())?;
        ensure(
            value.sha256 == expected.sha256,
            "producer capture seal mismatch",
        )?;
        Ok(expected)
    }
    pub fn to_json(&self) -> Result<String, ManifestError> {
        ensure(
            self.sha256 == Self::seal(self.capture.clone())?.sha256,
            "producer capture seal mismatch",
        )?;
        serde_json::to_string(self).map_err(|e| ManifestError::Integrity(e.to_string()))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProducerCaptureBytePin {
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProducerEnvelopeBody {
    pub version: String,
    pub roster_sha256: String,
    pub owned_sources_sha256: String,
    pub raw_dataset_sha256: String,
    pub split_sha256: String,
    pub current_view_sha256: String,
    pub raw_records: u64,
    pub unique_inputs: u64,
    pub capture_sha256: String,
    pub capture_artifact: ProducerCaptureBytePin,
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrozenProducerEnvelope {
    envelope: ProducerEnvelopeBody,
    sha256: String,
}
impl FrozenProducerEnvelope {
    pub fn seal(envelope: ProducerEnvelopeBody) -> Result<Self, ManifestError> {
        ensure(
            envelope.version == ENVELOPE_DOMAIN
                && (1..=MAX_RECORDS as u64).contains(&envelope.raw_records)
                && (1..=envelope.raw_records).contains(&envelope.unique_inputs)
                && (1..=crate::MAX_MANIFEST_BYTES as u64)
                    .contains(&envelope.capture_artifact.bytes),
            "producer envelope extent/version",
        )?;
        ensure(
            [
                &envelope.roster_sha256,
                &envelope.owned_sources_sha256,
                &envelope.raw_dataset_sha256,
                &envelope.split_sha256,
                &envelope.current_view_sha256,
                &envelope.capture_sha256,
                &envelope.capture_artifact.sha256,
            ]
            .into_iter()
            .all(|value| sha(value)),
            "invalid producer envelope digest",
        )?;
        let sha256 = metadata_digest(ENVELOPE_DOMAIN, &envelope)?;
        Ok(Self { envelope, sha256 })
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn body(&self) -> &ProducerEnvelopeBody {
        &self.envelope
    }
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        let expected = Self::seal(value.envelope.clone())?;
        ensure(
            value.sha256 == expected.sha256,
            "producer envelope seal mismatch",
        )?;
        Ok(expected)
    }
    pub fn to_json(&self) -> Result<String, ManifestError> {
        ensure(
            self.sha256 == Self::seal(self.envelope.clone())?.sha256,
            "producer envelope seal mismatch",
        )?;
        serde_json::to_string(self).map_err(|e| ManifestError::Integrity(e.to_string()))
    }
}

fn metadata_digest<T: Serialize>(domain: &str, value: &T) -> Result<String, ManifestError> {
    let value = serde_json::to_value(value).map_err(|e| ManifestError::Integrity(e.to_string()))?;
    sorted_canonical_sha(domain, value)
}

/// New float-free registry identity. Existing registry artifact byte hashes and
/// raw/split/input/current-view domains retain their previous meanings.
pub fn owned_sources_digest(sources: &PalsOwnedSources) -> Result<String, ManifestError> {
    sources.validate()?;
    let mut entries = Vec::new();
    for source in &sources.input_sources {
        let value = sorted_json(
            serde_json::to_value(source).map_err(|e| ManifestError::Integrity(e.to_string()))?,
        );
        let key =
            serde_json::to_string(&value).map_err(|e| ManifestError::Integrity(e.to_string()))?;
        entries.push((key, value));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    sorted_canonical_sha(
        OWNED_SOURCES_DOMAIN,
        serde_json::json!({
            "cpu_binary_sha256": sources.cpu_binary_sha256,
            "input_sources": entries.into_iter().map(|(_, value)| value).collect::<Vec<_>>()
        }),
    )
}

#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
pub struct RequiresDerivedAdapter {
    pub input_sha256: String,
    pub game_id: String,
    pub producer_id: String,
    pub derived_encoding_sha256: String,
    pub encoding_schema_sha256: String,
    pub private_encoder_source_sha256: String,
    pub parent_encoding_sha256: String,
    pub parent_encoder_source_sha256: String,
}
#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
pub struct FrozenProducerMetadataAudit {
    pub scope: String,
    pub raw_records: u64,
    pub unique_inputs: u64,
    pub roster_sha256: String,
    pub envelope_sha256: String,
    pub capture_sha256: String,
    pub native_exact_metadata_inputs: u64,
    pub requires_derived_adapter: Vec<RequiresDerivedAdapter>,
}

/// Audit every historical row, not merely the current leaf. Independent pins
/// are required, but this only proves metadata consistency against those pins.
/// Native sidecar validation and live capture/runtime evidence are not supplied
/// by this function. No result from this API is an actual training admission.
pub fn audit_metadata(
    records: &[PalsLearningRecord],
    split: &PalsDatasetSplit,
    sources: &PalsOwnedSources,
    roster: &FrozenProducerRoster,
    envelope: &FrozenProducerEnvelope,
    capture_json: &str,
    independently_registered: &[ProducerRegistrationPin],
) -> Result<FrozenProducerMetadataAudit, ManifestError> {
    let roster = FrozenProducerRoster::from_json(&roster.to_json()?)?;
    let envelope = FrozenProducerEnvelope::from_json(&envelope.to_json()?)?;
    let capture = ProducerCaptureArtifact::from_json(capture_json)?;
    let pins = ProducerRosterBody {
        version: ROSTER_DOMAIN.into(),
        game_producers: independently_registered.to_vec(),
    }
    .normalized()?;
    let pins: BTreeMap<_, _> = pins
        .game_producers
        .iter()
        .map(|pin| ((&pin.game_id, &pin.producer_id), pin))
        .collect();
    let declarations: BTreeMap<_, _> = roster
        .body()
        .game_producers
        .iter()
        .map(|pin| ((&pin.game_id, &pin.producer_id), pin))
        .collect();
    for (key, declaration) in &declarations {
        ensure(
            pins.get(key) == Some(declaration),
            "producer declaration differs from independent registration",
        )?;
        ensure(
            sources.input_sources.contains(&declaration.source),
            "producer source absent from owned registry",
        )?;
    }
    let (raw, current) = sources.audit_with_current_view(records, split)?;
    let e = envelope.body();
    ensure(
        e.roster_sha256 == roster.sha256()
            && e.owned_sources_sha256 == owned_sources_digest(sources)?
            && e.raw_dataset_sha256 == raw.canonical_dataset_sha256
            && e.split_sha256 == raw.canonical_split_sha256
            && e.current_view_sha256 == current.sha256()
            && e.raw_records == raw.records,
        "producer envelope raw/current/registry/roster mismatch",
    )?;
    ensure(
        e.capture_sha256 == capture.sha256()
            && e.capture_artifact.bytes == capture_json.len() as u64
            && e.capture_artifact.sha256 == digest(capture_json.as_bytes()),
        "producer capture byte/seal mismatch",
    )?;
    let bindings: BTreeMap<_, _> = capture
        .body()
        .bindings
        .iter()
        .map(|binding| (binding.input_sha256.as_str(), binding))
        .collect();
    let mut seen = BTreeSet::new();
    let mut native_exact_metadata_inputs = 0;
    let mut requires_derived_adapter = Vec::new();
    for row in records {
        let input = row.input.sha256();
        let snapshot = row.input.snapshot();
        let binding = bindings
            .get(input)
            .ok_or_else(|| ManifestError::Integrity("missing input producer binding".into()))?;
        ensure(
            binding.game_id == snapshot.game_id
                && binding.capture_sequence == snapshot.capture_sequence,
            "producer binding game/capture mismatch",
        )?;
        let pin = declarations
            .get(&(&binding.game_id, &binding.producer_id))
            .ok_or_else(|| ManifestError::Integrity("unregistered input producer".into()))?;
        ensure(
            pin.source == snapshot.source && pin.frozen_epoch == snapshot.frozen_epoch,
            "producer source/frozen epoch drift",
        )?;
        match &pin.encoding_policy {
            ProducerEncodingPolicy::NativeExact {
                encoding_sha256, ..
            } => {
                ensure(
                    row.verifier_private.is_none(),
                    "private verifier row requires derived encoding policy",
                )?;
                ensure(
                    encoding_sha256 == &snapshot.encoding_sha256,
                    "producer native encoding drift",
                )?;
            }
            ProducerEncodingPolicy::PrivateCheckedDerivedQuery { .. } => ensure(
                snapshot.role == PalsDataRole::Verifier,
                "private derived producer requires verifier role",
            )?,
        }
        if seen.insert(input) {
            match &pin.encoding_policy {
                ProducerEncodingPolicy::NativeExact { .. } => native_exact_metadata_inputs += 1,
                ProducerEncodingPolicy::PrivateCheckedDerivedQuery {
                    encoding_schema_sha256,
                    private_encoder_source_sha256,
                    parent_encoding_sha256,
                    parent_encoder_source_sha256,
                } => requires_derived_adapter.push(RequiresDerivedAdapter {
                    input_sha256: input.into(),
                    game_id: binding.game_id.clone(),
                    producer_id: binding.producer_id.clone(),
                    derived_encoding_sha256: snapshot.encoding_sha256.clone(),
                    encoding_schema_sha256: encoding_schema_sha256.clone(),
                    private_encoder_source_sha256: private_encoder_source_sha256.clone(),
                    parent_encoding_sha256: parent_encoding_sha256.clone(),
                    parent_encoder_source_sha256: parent_encoder_source_sha256.clone(),
                }),
            }
        }
    }
    ensure(
        seen.len() == bindings.len() && e.unique_inputs == seen.len() as u64,
        "unobserved or missing producer bindings",
    )?;
    requires_derived_adapter.sort_by(|a, b| a.input_sha256.cmp(&b.input_sha256));
    Ok(FrozenProducerMetadataAudit {
        scope: "metadata_only".into(),
        raw_records: raw.records,
        unique_inputs: e.unique_inputs,
        roster_sha256: roster.sha256().into(),
        envelope_sha256: envelope.sha256().into(),
        capture_sha256: capture.sha256().into(),
        native_exact_metadata_inputs,
        requires_derived_adapter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str) -> PalsInputSource {
        PalsInputSource::OwnPals {
            model_configuration_sha256: digest(name.as_bytes()),
            model_weights_sha256: digest(format!("weights-{name}").as_bytes()),
        }
    }
    fn cpu() -> PalsInputSource {
        PalsInputSource::OwnCpu {
            cpu_binary_sha256: digest(b"cpu"),
            evaluator_configuration_sha256: digest(b"cpu-config"),
            model_weights_sha256: None,
        }
    }
    fn row(
        source: PalsInputSource,
        epoch: u64,
        sequence: u64,
        encoding: &str,
    ) -> PalsLearningRecord {
        PalsLearningRecord {
            input: PalsInputSnapshot {
                game_id: "game".into(),
                opening_id: "opening".into(),
                line_genealogy_id: "line".into(),
                position_command: "position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1".into(),
                board_fen: "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1".into(),
                actual_history: vec![],
                rules_state_sha256: digest(b"state"),
                rules_history_sha256: digest(b"history"),
                transposition_sha256: digest(b"transposition"),
                encoding_sha256: encoding.into(),
                source,
                frozen_epoch: epoch,
                input_revision: 1,
                capture_sequence: sequence,
                white_to_move: true,
                role: PalsDataRole::Proposer,
                legal_moves: vec![1292, 1804],
                public_records: vec![],
            }
            .seal()
            .unwrap(),
            future_label: None,
            verifier_private: None,
        }
    }
    struct Fixture {
        rows: Vec<PalsLearningRecord>,
        split: PalsDatasetSplit,
        sources: PalsOwnedSources,
        pins: Vec<ProducerRegistrationPin>,
        roster: FrozenProducerRoster,
        capture: ProducerCaptureArtifact,
    }
    impl Fixture {
        fn new(rows: Vec<PalsLearningRecord>) -> Self {
            let split = PalsDatasetSplit {
                games: BTreeMap::from([("game".into(), PalsSplit::Train)]),
            };
            let sources = PalsOwnedSources {
                cpu_binary_sha256: rows
                    .iter()
                    .filter_map(|row| match &row.input.snapshot().source {
                        PalsInputSource::OwnCpu {
                            cpu_binary_sha256, ..
                        } => Some(cpu_binary_sha256.clone()),
                        _ => None,
                    })
                    .collect(),
                input_sources: rows
                    .iter()
                    .map(|row| row.input.snapshot().source.clone())
                    .collect(),
            };
            let mut seen = BTreeSet::new();
            let mut pins = Vec::new();
            let mut bindings = Vec::new();
            for row in &rows {
                if !seen.insert(row.input.sha256()) {
                    continue;
                }
                let snapshot = row.input.snapshot();
                let producer_id = format!("producer-{}", pins.len());
                let native_model_epoch = match &snapshot.source {
                    PalsInputSource::OwnCpu { .. } => NativeModelEpoch::EncodingOnlyZero,
                    PalsInputSource::OwnPals {
                        model_weights_sha256,
                        ..
                    } => NativeModelEpoch::FrozenModelEpoch {
                        sha256: model_weights_sha256.clone(),
                    },
                };
                pins.push(ProducerRegistrationPin {
                    game_id: snapshot.game_id.clone(),
                    producer_id: producer_id.clone(),
                    registration_sha256: digest(producer_id.as_bytes()),
                    source: snapshot.source.clone(),
                    frozen_epoch: snapshot.frozen_epoch,
                    encoding_policy: ProducerEncodingPolicy::NativeExact {
                        encoding_sha256: snapshot.encoding_sha256.clone(),
                        encoder_source_sha256: digest(b"encoder"),
                        native_model_epoch,
                    },
                });
                bindings.push(InputProducerBinding {
                    input_sha256: row.input.sha256().into(),
                    game_id: snapshot.game_id.clone(),
                    producer_id,
                    capture_sequence: snapshot.capture_sequence,
                    capture_evidence_sha256: digest(b"capture-evidence"),
                });
            }
            let roster = FrozenProducerRoster::seal(ProducerRosterBody {
                version: ROSTER_DOMAIN.into(),
                game_producers: pins.clone(),
            })
            .unwrap();
            let capture = ProducerCaptureArtifact::seal(ProducerCaptureBody {
                version: CAPTURE_DOMAIN.into(),
                bindings,
            })
            .unwrap();
            Self {
                rows,
                split,
                sources,
                pins,
                roster,
                capture,
            }
        }
        fn envelope(&self, capture_json: &str) -> FrozenProducerEnvelope {
            let (raw, current) = self
                .sources
                .audit_with_current_view(&self.rows, &self.split)
                .unwrap();
            FrozenProducerEnvelope::seal(ProducerEnvelopeBody {
                version: ENVELOPE_DOMAIN.into(),
                roster_sha256: self.roster.sha256().into(),
                owned_sources_sha256: owned_sources_digest(&self.sources).unwrap(),
                raw_dataset_sha256: raw.canonical_dataset_sha256,
                split_sha256: raw.canonical_split_sha256,
                current_view_sha256: current.sha256().into(),
                raw_records: raw.records,
                unique_inputs: self.capture.body().bindings.len() as u64,
                capture_sha256: self.capture.sha256().into(),
                capture_artifact: ProducerCaptureBytePin {
                    bytes: capture_json.len() as u64,
                    sha256: digest(capture_json.as_bytes()),
                },
            })
            .unwrap()
        }
        fn audit(&self) -> Result<FrozenProducerMetadataAudit, ManifestError> {
            let raw = self.capture.to_json()?;
            audit_metadata(
                &self.rows,
                &self.split,
                &self.sources,
                &self.roster,
                &self.envelope(&raw),
                &raw,
                &self.pins,
            )
        }
    }

    #[test]
    fn shared_float_free_vector_and_strict_u64_duplicate_key_boundary() {
        let vector: serde_json::Value =
            decode_json(include_str!("frozen_producer_vector.fixture")).unwrap();
        let body: ProducerRosterBody =
            serde_json::from_value(vector["roster_body"].clone()).unwrap();
        let roster = FrozenProducerRoster::seal(body).unwrap();
        let canonical = vector["canonical_roster"].as_str().unwrap();
        let value = sorted_json(serde_json::to_value(roster.body()).unwrap());
        assert_eq!(
            serde_json::to_string(&(ROSTER_DOMAIN, value)).unwrap(),
            canonical
        );
        assert_eq!(roster.sha256(), digest(canonical.as_bytes()));
        let raw = roster.to_json().unwrap();
        for invalid in ["true", "1.0", "18446744073709551616", "-1"] {
            assert!(
                FrozenProducerRoster::from_json(&raw.replace("9007199254740993", invalid)).is_err()
            );
        }
        let duplicate = raw.replace(
            "\"game_id\":\"경기\"",
            "\"game_id\":\"경기\",\"game_id\":\"경기\"",
        );
        assert!(FrozenProducerRoster::from_json(&duplicate).is_err());
        assert_eq!(FrozenProducerRoster::from_json(&raw).unwrap(), roster);
    }
    #[test]
    fn two_models_in_one_game_and_weightless_cpu_are_metadata_only() {
        let encoding = digest(b"encoding");
        let fixture = Fixture::new(vec![
            row(model("a"), 1, 8, &encoding),
            row(model("b"), 9, 9, &encoding),
        ]);
        let audit = fixture.audit().unwrap();
        assert_eq!(audit.scope, "metadata_only");
        assert_eq!(audit.unique_inputs, 2);
        assert_eq!(audit.native_exact_metadata_inputs, 2);
        let fixture = Fixture::new(vec![row(cpu(), 0, 8, &encoding)]);
        assert!(matches!(
            &fixture.roster.body().game_producers[0].source,
            PalsInputSource::OwnCpu {
                model_weights_sha256: None,
                ..
            }
        ));
        assert_eq!(fixture.audit().unwrap().native_exact_metadata_inputs, 1);
    }
    #[test]
    fn same_producer_cannot_drift_source_epoch_or_native_encoding() {
        let encoding = digest(b"encoding");
        for next in [
            row(model("b"), 1, 9, &encoding),
            row(model("a"), 2, 9, &encoding),
            row(model("a"), 1, 9, &digest(b"changed-encoding")),
        ] {
            let mut fixture = Fixture::new(vec![row(model("a"), 1, 8, &encoding), next]);
            let mut capture = fixture.capture.body().clone();
            for binding in &mut capture.bindings {
                binding.producer_id = "producer-0".into();
            }
            fixture.capture = ProducerCaptureArtifact::seal(capture).unwrap();
            assert!(fixture.audit().is_err());
        }
    }
    #[test]
    fn one_registered_producer_can_capture_multiple_stable_inputs() {
        let encoding = digest(b"encoding");
        let mut fixture = Fixture::new(vec![
            row(model("a"), 1, 8, &encoding),
            row(model("a"), 1, 9, &encoding),
        ]);
        let mut capture = fixture.capture.body().clone();
        for binding in &mut capture.bindings {
            binding.producer_id = "producer-0".into();
        }
        fixture.capture = ProducerCaptureArtifact::seal(capture).unwrap();
        fixture.pins.truncate(1);
        fixture.roster = FrozenProducerRoster::seal(ProducerRosterBody {
            version: ROSTER_DOMAIN.into(),
            game_producers: fixture.pins.clone(),
        })
        .unwrap();
        assert_eq!(fixture.audit().unwrap().unique_inputs, 2);

        let mut missing = fixture.capture.body().clone();
        missing.bindings.pop();
        fixture.capture = ProducerCaptureArtifact::seal(missing).unwrap();
        assert!(fixture.audit().is_err());
    }
    #[test]
    fn roster_is_order_independent_and_freezes_each_game_producer_pair() {
        let fixture = Fixture::new(vec![
            row(model("a"), 1, 8, &digest(b"encoding")),
            row(model("b"), 9, 9, &digest(b"encoding")),
        ]);
        let mut body = fixture.roster.body().clone();
        body.game_producers[0].game_id = "game-a".into();
        body.game_producers[1].game_id = "game-b".into();
        for pin in &mut body.game_producers {
            pin.producer_id = "same-producer".into();
        }
        let roster = FrozenProducerRoster::seal(body.clone()).unwrap();
        body.game_producers.reverse();
        assert_eq!(FrozenProducerRoster::seal(body.clone()).unwrap(), roster);
        body.game_producers.push(body.game_producers[0].clone());
        assert!(FrozenProducerRoster::seal(body).is_err());
    }
    #[test]
    fn independent_registration_and_whole_history_seals_are_required() {
        let mut fixture = Fixture::new(vec![row(cpu(), 0, 8, &digest(b"encoding"))]);
        fixture.pins[0].registration_sha256 = digest(b"changed-registration");
        assert!(fixture.audit().is_err());
        fixture.pins = fixture.roster.body().game_producers.clone();
        let raw = fixture.capture.to_json().unwrap();
        let envelope = fixture.envelope(&raw);
        let mut changed = envelope.body().clone();
        changed.current_view_sha256 = digest(b"stale-current-view");
        let changed = FrozenProducerEnvelope::seal(changed).unwrap();
        assert!(
            audit_metadata(
                &fixture.rows,
                &fixture.split,
                &fixture.sources,
                &fixture.roster,
                &changed,
                &raw,
                &fixture.pins
            )
            .is_err()
        );
        assert!(
            audit_metadata(
                &fixture.rows,
                &fixture.split,
                &fixture.sources,
                &fixture.roster,
                &envelope,
                &(raw + " "),
                &fixture.pins
            )
            .is_err()
        );
        let mut duplicate = fixture.capture.body().clone();
        duplicate.bindings.push(duplicate.bindings[0].clone());
        assert!(ProducerCaptureArtifact::seal(duplicate).is_err());
    }
    #[test]
    fn raw_capture_first_label_and_supersession_keep_one_binding() {
        let original = row(cpu(), 0, 8, &digest(b"encoding"));
        let mut first = original.clone();
        first.future_label = Some(PalsFutureLabel {
            observed_sequence: 9,
            provenance: PalsTargetProvenance::ActualGame {
                result: PalsGameResult {
                    game_id: "game".into(),
                    outcome: PalsOutcome::Unknown,
                    ending: PalsGameEnd::UserStop,
                    raw_evidence_sha256: digest(b"game-evidence"),
                },
            },
            policy: None,
            value_wdl: None,
            white_to_move: true,
            counterexample: None,
            verifier_tasks: None,
            supersedes_label_sha256: None,
        });
        let mut next = first.clone();
        next.future_label.as_mut().unwrap().observed_sequence = 10;
        next.future_label.as_mut().unwrap().supersedes_label_sha256 = first.label_digest().unwrap();
        let fixture = Fixture::new(vec![original, first, next]);
        let audit = fixture.audit().unwrap();
        assert_eq!((audit.raw_records, audit.unique_inputs), (3, 1));
        assert_eq!(
            fixture
                .sources
                .current_training_view(&fixture.rows, &fixture.split)
                .unwrap()
                .indices(),
            &[2]
        );
    }
    #[test]
    fn private_query_schema_never_satisfies_derived_adapter_requirement() {
        let mut original = row(model("v"), 1, 8, &digest(b"per-query-encoding"));
        let mut snapshot = original.input.snapshot().clone();
        snapshot.role = PalsDataRole::Verifier;
        original.input = snapshot.seal().unwrap();
        original.verifier_private = Some(PalsVerifierPrivate {
            task_kind: "resume_task".into(),
            control_sha256: digest(b"private-control"),
            private_latent: vec![0.0, -0.0],
        });
        let mut fixture = Fixture::new(vec![original]);
        assert!(fixture.audit().is_err()); // The native declaration cannot hide private V data.
        fixture.pins[0].encoding_policy = ProducerEncodingPolicy::PrivateCheckedDerivedQuery {
            encoding_schema_sha256: digest(b"query-schema"),
            private_encoder_source_sha256: digest(b"private-encoder"),
            parent_encoding_sha256: digest(b"native-encoding"),
            parent_encoder_source_sha256: digest(b"native-encoder"),
        };
        fixture.roster = FrozenProducerRoster::seal(ProducerRosterBody {
            version: ROSTER_DOMAIN.into(),
            game_producers: fixture.pins.clone(),
        })
        .unwrap();
        let audit = fixture.audit().unwrap();
        assert_eq!(audit.scope, "metadata_only");
        assert_eq!(audit.native_exact_metadata_inputs, 0);
        assert_eq!(audit.requires_derived_adapter.len(), 1);
        assert_ne!(
            audit.requires_derived_adapter[0].derived_encoding_sha256,
            audit.requires_derived_adapter[0].encoding_schema_sha256
        );
    }
}
