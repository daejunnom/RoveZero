//! Optional collector-owned producer registration and prepared capture evidence.
//! The metadata module owns seals; this module connects checked driver owners,
//! actual serialized bytes and prepaid publication. It never admits training.
use super::*;
use rz_experiments::frozen_producer::{
    CAPTURE_DOMAIN, ENVELOPE_DOMAIN, FrozenProducerEnvelope, FrozenProducerRoster,
    InputProducerBinding, NativeModelEpoch, ProducerCaptureArtifact, ProducerCaptureBody,
    ProducerCaptureBytePin, ProducerEncodingPolicy, ProducerEnvelopeBody, ProducerRegistrationPin,
    ProducerRosterBody, ROSTER_DOMAIN, audit_metadata, owned_sources_digest,
};
use std::io::{Seek, SeekFrom};
use std::sync::{Arc, Mutex};

const REGISTRATION_DOMAIN: &str = "rz-pals-collector-producer-registration/1";
const SOURCE_DOMAIN: &str = "rz-pals-collector-checked-source/1";
const PREPARED_DOMAIN: &str = "rz-pals-collector-prepared-producer/1";
const MANIFEST_CAP: u64 = rz_experiments::MAX_MANIFEST_BYTES as u64;
const CLOSE_ALLOWANCE: u64 = 8 * 1024;
const FAILURE_ALLOWANCE: u64 = 8 * 1024;

/// Borrowed authority over the actual checked dispatch owner. Its private
/// constructor is used by the known CPU/native driver implementations.
/// Delegating a real owner's capability redirects strict dispatch to that
/// owner, rather than authorizing a custom wrapper to execute its own work.
pub struct CheckedProducerOwner<'a> {
    driver: &'a mut dyn PalsCollectionDriver,
    source: PalsCollectionSourceDescription,
}
impl<'a> CheckedProducerOwner<'a> {
    pub(super) fn new(
        driver: &'a mut dyn PalsCollectionDriver,
        source: PalsCollectionSourceDescription,
    ) -> Self {
        Self { driver, source }
    }
    pub(super) fn into_parts(
        self,
    ) -> (
        &'a mut dyn PalsCollectionDriver,
        PalsCollectionSourceDescription,
    ) {
        (self.driver, self.source)
    }
    pub(super) fn source(&self) -> &PalsCollectionSourceDescription {
        &self.source
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Registration {
    version: String,
    producer_id: String,
    source: PalsInputSource,
    frozen_epoch: u64,
    encoding_policy: ProducerEncodingPolicy,
    checked_source_sha256: String,
}

/// Independent prior declaration bytes are required. No row or teacher label
/// can enroll a producer. Limits reserve conservative close-stage output credit.
#[derive(Clone, Debug)]
pub struct PalsProducerCollectionConfig {
    registration: Registration,
    registration_bytes: Vec<u8>,
    registration_sha256: String,
    journal_bytes: u64,
    capture_bytes: u64,
}
impl PalsProducerCollectionConfig {
    pub fn from_registration_bytes(
        bytes: &[u8],
        expected_sha256: &str,
    ) -> Result<Self, ArenaError> {
        if bytes.is_empty()
            || bytes.len() > 256 * 1024
            || !valid_sha(expected_sha256)
            || sha_bytes(bytes) != expected_sha256
        {
            return Err(invalid(
                "producer registration bytes differ from independent pin",
            ));
        }
        let text = std::str::from_utf8(bytes).map_err(|e| invalid(e.to_string()))?;
        let registration: Registration = rz_experiments::decode_json(text)?;
        if registration.version != REGISTRATION_DOMAIN
            || !ident(&registration.producer_id)
            || !valid_sha(&registration.checked_source_sha256)
            || !matches!(
                registration.encoding_policy,
                ProducerEncodingPolicy::NativeExact { .. }
            )
        {
            return Err(invalid(
                "unsupported independent native/CPU producer registration",
            ));
        }
        FrozenProducerRoster::seal(ProducerRosterBody {
            version: ROSTER_DOMAIN.into(),
            game_producers: vec![ProducerRegistrationPin {
                game_id: "registration-validation".into(),
                producer_id: registration.producer_id.clone(),
                registration_sha256: expected_sha256.into(),
                source: registration.source.clone(),
                frozen_epoch: registration.frozen_epoch,
                encoding_policy: registration.encoding_policy.clone(),
            }],
        })?;
        Ok(Self {
            registration,
            registration_bytes: bytes.to_vec(),
            registration_sha256: expected_sha256.into(),
            journal_bytes: 512 * 1024,
            capture_bytes: 512 * 1024,
        })
    }
    pub fn read_pinned(path: &Path, expected_sha256: &str) -> Result<Self, ArenaError> {
        if !path.is_absolute()
            || path
                .file_name()
                .and_then(|v| v.to_str())
                .is_some_and(|v| v == ".env" || v.starts_with(".env."))
        {
            return Err(invalid(
                "producer registration requires an absolute public file",
            ));
        }
        let mut verified = VerifiedExecutableIdentity::open(path, 256 * 1024)?;
        if verified.sha256 != expected_sha256 {
            return Err(invalid(
                "producer registration differs from independent pin",
            ));
        }
        verified.file.seek(SeekFrom::Start(0)).map_err(io)?;
        let mut bytes = Vec::new();
        (&mut verified.file)
            .take(256 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(io)?;
        verified.validate_stability()?;
        Self::from_registration_bytes(&bytes, expected_sha256)
    }
    pub fn with_metadata_limits(
        mut self,
        journal_bytes: u64,
        capture_bytes: u64,
    ) -> Result<Self, ArenaError> {
        if !(CLOSE_ALLOWANCE..=MANIFEST_CAP - FAILURE_ALLOWANCE).contains(&journal_bytes)
            || !(CLOSE_ALLOWANCE..=MANIFEST_CAP).contains(&capture_bytes)
        {
            return Err(invalid(
                "producer metadata limits must be finite bounded byte credits",
            ));
        }
        self.journal_bytes = journal_bytes;
        self.capture_bytes = capture_bytes;
        Ok(self)
    }
}
fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
}
fn sha_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn sorted(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(fields) => serde_json::Value::Object(
            fields
                .into_iter()
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .map(|(k, v)| (k, sorted(v)))
                .collect(),
        ),
        serde_json::Value::Array(values) => {
            values.into_iter().map(sorted).collect::<Vec<_>>().into()
        }
        value => value,
    }
}
fn source_digest(source: &PalsCollectionSourceDescription) -> Result<String, ArenaError> {
    Ok(sha_bytes(&source_bytes(source)?))
}
fn source_bytes(source: &PalsCollectionSourceDescription) -> Result<Vec<u8>, ArenaError> {
    bounded_json(
        &(
            SOURCE_DOMAIN,
            sorted(serde_json::to_value(source).map_err(|e| invalid(e.to_string()))?),
        ),
        MAX_JSON_RECORD_BYTES,
        false,
    )
}
fn policy(source: &PalsCollectionSourceDescription) -> ProducerEncodingPolicy {
    ProducerEncodingPolicy::NativeExact {
        encoding_sha256: source.encoding_sha256.clone(),
        encoder_source_sha256: source.encoder_source_sha256.clone(),
        native_model_epoch: if source.model_epoch_kind == "encoding_only_zero" {
            NativeModelEpoch::EncodingOnlyZero
        } else {
            NativeModelEpoch::FrozenModelEpoch {
                sha256: hex(source.model_epoch),
            }
        },
    }
}

/// Read-only checked driver facts for an owner to register before a later run.
/// This declaration is not self-enrollment or evidence of training execution.
pub fn pals_producer_registration_description(
    driver: &mut dyn PalsCollectionDriver,
    producer_id: &str,
) -> Result<serde_json::Value, ArenaError> {
    let owner = driver
        .checked_producer_owner()?
        .ok_or_else(|| invalid("driver has no checked producer constructor"))?;
    let source = owner.source();
    checked_source(source)?;
    if !ident(producer_id) {
        return Err(invalid("invalid registered producer ID"));
    }
    serde_json::to_value(Registration {
        version: REGISTRATION_DOMAIN.into(),
        producer_id: producer_id.into(),
        source: source.source.clone(),
        frozen_epoch: source.frozen_epoch,
        encoding_policy: policy(source),
        checked_source_sha256: source_digest(source)?,
    })
    .map_err(|e| invalid(e.to_string()))
}

#[derive(Clone, Debug, Serialize)]
struct JsonBytePin {
    bytes: u64,
    sha256: String,
}
impl JsonBytePin {
    fn new(bytes: &[u8]) -> Self {
        Self {
            bytes: bytes.len() as u64,
            sha256: sha_bytes(bytes),
        }
    }
}
#[derive(Clone, Debug, Serialize)]
struct PreparedEvidence {
    version: String,
    producer_id: String,
    registration_sha256: String,
    roster_sha256: String,
    checked_source_sha256: String,
    game_id: String,
    input_sha256: String,
    capture_sequence: u64,
    /// Byte pins exclude the JSONL newline; no f32 reserialization is involved.
    input_json: JsonBytePin,
    tensor_sidecar_json: JsonBytePin,
    lineage_json: JsonBytePin,
    native_request: Option<(u64, u64)>,
    learning_input: bool,
    publication: String,
}
struct Journal {
    lines: Vec<Vec<u8>>,
    published: usize,
    bytes: u64,
    binding_bytes: u64,
    bindings: BTreeMap<String, InputProducerBinding>,
    failed: bool,
}
/// Only admission against independently pinned bytes and a checked driver can
/// construct this handle. Input owner is explicit, never inferred from color or teacher.
#[derive(Clone)]
pub struct RegisteredProducerHandle {
    registration: Registration,
    registration_sha256: String,
    roster: FrozenProducerRoster,
    checked_source_bytes: Vec<u8>,
    journal_limit: u64,
    capture_limit: u64,
    journal: Arc<Mutex<Journal>>,
}
impl RegisteredProducerHandle {
    #[cfg(all(test, feature = "pals-collection-onnx"))]
    pub(super) fn journal_snapshot(&self) -> Vec<Vec<u8>> {
        self.journal.lock().unwrap().lines.clone()
    }
    pub(super) fn admit(
        config: &PalsProducerCollectionConfig,
        games: &PalsCollectionConfig,
        actual: &PalsCollectionSourceDescription,
    ) -> Result<Self, ArenaError> {
        checked_source(actual)?;
        let declaration = &config.registration;
        if declaration.source != actual.source
            || declaration.frozen_epoch != actual.frozen_epoch
            || declaration.encoding_policy != policy(actual)
            || declaration.checked_source_sha256 != source_digest(actual)?
        {
            return Err(invalid(
                "registered producer differs from actual checked driver source",
            ));
        }
        let roster = FrozenProducerRoster::seal(ProducerRosterBody {
            version: ROSTER_DOMAIN.into(),
            game_producers: (0..games.games)
                .map(|n| ProducerRegistrationPin {
                    game_id: format!("{}-g{}", games.run_id, n + 1),
                    producer_id: declaration.producer_id.clone(),
                    registration_sha256: config.registration_sha256.clone(),
                    source: actual.source.clone(),
                    frozen_epoch: actual.frozen_epoch,
                    encoding_policy: policy(actual),
                })
                .collect(),
        })?;
        Ok(Self {
            registration: declaration.clone(),
            registration_sha256: config.registration_sha256.clone(),
            roster,
            checked_source_bytes: source_bytes(actual)?,
            journal_limit: config.journal_bytes,
            capture_limit: config.capture_bytes,
            journal: Arc::new(Mutex::new(Journal {
                lines: Vec::new(),
                published: 0,
                bytes: 0,
                binding_bytes: CLOSE_ALLOWANCE,
                bindings: BTreeMap::new(),
                failed: false,
            })),
        })
    }
    pub(super) fn verify_source(
        &self,
        source: &PalsCollectionSourceDescription,
    ) -> Result<(), ArenaError> {
        if self.registration.checked_source_sha256 != source_digest(source)? {
            return Err(invalid("checked producer source changed after admission"));
        }
        Ok(())
    }
    pub(super) fn reserve_bytes(&self) -> u64 {
        self.journal_limit + FAILURE_ALLOWANCE + self.capture_limit + CLOSE_ALLOWANCE
    }
    pub(super) fn capture(
        &self,
        input: &PalsFrozenInput,
        input_json: &[u8],
        sidecar_json: &[u8],
        lineage_json: &[u8],
        native_request: Option<(u64, u64)>,
        learning_input: bool,
    ) -> Result<(), ArenaError> {
        input.verify()?;
        let snapshot = input.snapshot();
        if snapshot.source != self.registration.source
            || snapshot.frozen_epoch != self.registration.frozen_epoch
            || !self
                .roster
                .body()
                .game_producers
                .iter()
                .any(|pin| pin.game_id == snapshot.game_id)
            || match &self.registration.encoding_policy {
                ProducerEncodingPolicy::NativeExact {
                    encoding_sha256, ..
                } => encoding_sha256 != &snapshot.encoding_sha256,
                _ => true,
            }
        {
            return Err(invalid(
                "prepared input differs from registered producer handle",
            ));
        }
        let evidence = PreparedEvidence {
            version: PREPARED_DOMAIN.into(),
            producer_id: self.registration.producer_id.clone(),
            registration_sha256: self.registration_sha256.clone(),
            roster_sha256: self.roster.sha256().into(),
            checked_source_sha256: self.registration.checked_source_sha256.clone(),
            game_id: snapshot.game_id.clone(),
            input_sha256: input.sha256().into(),
            capture_sequence: snapshot.capture_sequence,
            input_json: JsonBytePin::new(input_json),
            tensor_sidecar_json: JsonBytePin::new(sidecar_json),
            lineage_json: JsonBytePin::new(lineage_json),
            native_request,
            learning_input,
            publication: if native_request.is_some() {
                "seal-before-submit; prepaid-drain-after-search"
            } else {
                "append-before-analysis; sync-at-receipt-close"
            }
            .into(),
        };
        let evidence_sha = canonical_sha256(&(
            PREPARED_DOMAIN,
            sorted(serde_json::to_value(&evidence).map_err(|e| invalid(e.to_string()))?),
        ))?;
        let line = bounded_json(
            &serde_json::json!({"prepared":evidence,"sha256":evidence_sha}),
            MAX_JSON_RECORD_BYTES,
            false,
        )?;
        let binding = InputProducerBinding {
            input_sha256: input.sha256().into(),
            game_id: snapshot.game_id.clone(),
            producer_id: self.registration.producer_id.clone(),
            capture_sequence: snapshot.capture_sequence,
            capture_evidence_sha256: evidence_sha,
        };
        let binding_bytes = bounded_json(&binding, MAX_JSON_RECORD_BYTES, false)?.len() as u64 + 1;
        let mut journal = self
            .journal
            .lock()
            .map_err(|_| invalid("producer journal owner poisoned"))?;
        let next_bytes = journal.bytes + line.len() as u64 + 1;
        let next_binding_bytes =
            journal.binding_bytes + if learning_input { binding_bytes } else { 0 };
        if journal.failed
            || journal.lines.len() >= MAX_ROWS
            || next_bytes > self.journal_limit
            || next_binding_bytes > self.capture_limit
            || journal.bindings.contains_key(input.sha256())
        {
            if !journal.failed {
                let rejected = bounded_json(
                    &serde_json::json!({"prepared":evidence,"sha256":binding.capture_evidence_sha256,"stage":"prepared-rejected-before-submit",
                    "reason":"producer metadata quota or duplicate capture", "input_sha256":input.sha256(),
                    "producer_id":self.registration.producer_id}),
                    FAILURE_ALLOWANCE as usize - 1,
                    false,
                )?;
                journal.bytes += rejected.len() as u64 + 1;
                journal.lines.push(rejected);
            }
            journal.failed = true;
            return Err(ArenaError::Budget(
                "producer metadata credit rejected prepared input before dispatch".into(),
            ));
        }
        journal.bytes = next_bytes;
        journal.binding_bytes = next_binding_bytes;
        journal.lines.push(line);
        if learning_input {
            journal.bindings.insert(input.sha256().into(), binding);
        }
        Ok(())
    }
    pub(super) fn flush_journal(&self, output: &mut Output) -> Result<(), ArenaError> {
        let mut journal = self
            .journal
            .lock()
            .map_err(|_| invalid("producer journal owner poisoned"))?;
        while journal.published < journal.lines.len() {
            let line = &journal.lines[journal.published];
            output.write_prepaid("producer-prepared.jsonl", line, true)?;
            journal.published += 1;
        }
        Ok(())
    }
    pub(super) fn close(
        &self,
        output: &mut Output,
        records: &[PalsLearningRecord],
        split: &PalsDatasetSplit,
        sources: &PalsOwnedSources,
        complete: bool,
    ) -> Result<(), ArenaError> {
        // All admitted prepared events, including divergence and rejected calls,
        // survive independently of the learning-row projection and completion.
        self.flush_journal(output)?;
        let journal = self
            .journal
            .lock()
            .map_err(|_| invalid("producer journal owner poisoned"))?;
        if !complete || journal.failed {
            return Ok(());
        }
        let (raw, current) = sources.audit_with_current_view(records, split)?;
        let mut seen = BTreeSet::new();
        let mut bindings = Vec::new();
        for row in records {
            if seen.insert(row.input.sha256()) {
                bindings.push(
                    journal
                        .bindings
                        .get(row.input.sha256())
                        .ok_or_else(|| {
                            invalid("learning input has no actual prepared producer capture")
                        })?
                        .clone(),
                );
            }
        }
        let capture = ProducerCaptureArtifact::seal(ProducerCaptureBody {
            version: CAPTURE_DOMAIN.into(),
            bindings,
        })?;
        let capture_json = capture.to_json()?;
        if capture_json.len() as u64 > self.capture_limit {
            return Err(invalid("producer capture exceeds prepaid metadata limit"));
        }
        output.write_prepaid("producer-captures.json", capture_json.as_bytes(), false)?;
        let envelope = FrozenProducerEnvelope::seal(ProducerEnvelopeBody {
            version: ENVELOPE_DOMAIN.into(),
            roster_sha256: self.roster.sha256().into(),
            owned_sources_sha256: owned_sources_digest(sources)?,
            raw_dataset_sha256: raw.canonical_dataset_sha256,
            split_sha256: raw.canonical_split_sha256,
            current_view_sha256: current.sha256().into(),
            raw_records: raw.records,
            unique_inputs: seen.len() as u64,
            capture_sha256: capture.sha256().into(),
            capture_artifact: ProducerCaptureBytePin {
                bytes: capture_json.len() as u64,
                sha256: sha_bytes(capture_json.as_bytes()),
            },
        })?;
        // This remains the metadata-only audit; live prepared journal evidence
        // is separately preserved and does not imply a learner was run.
        let audit = audit_metadata(
            records,
            split,
            sources,
            &self.roster,
            &envelope,
            &capture_json,
            &self.roster.body().game_producers,
        )?;
        let envelope_json = envelope.to_json()?;
        let audit_json = bounded_json(&audit, CLOSE_ALLOWANCE as usize / 2, false)?;
        if envelope_json.len() + audit_json.len() > CLOSE_ALLOWANCE as usize {
            return Err(invalid(
                "producer envelope/audit exceeds prepaid close limit",
            ));
        }
        output.write_prepaid("producer-envelope.json", envelope_json.as_bytes(), false)?;
        output.write_prepaid("producer-audit.json", &audit_json, false)?;
        Ok(())
    }
    pub(super) fn start(
        &self,
        config: &PalsProducerCollectionConfig,
        output: &mut Output,
    ) -> Result<(), ArenaError> {
        output.reserve_producer(self.reserve_bytes())?;
        output.write(
            "producer-registration.json",
            &config.registration_bytes,
            false,
        )?;
        output.write("producer-source.json", &self.checked_source_bytes, false)?;
        let roster = self.roster.to_json()?;
        if roster.len() as u64 > MANIFEST_CAP {
            return Err(invalid("producer roster byte bound"));
        }
        output.write("producer-roster.json", roster.as_bytes(), false)?;
        Ok(())
    }
}
