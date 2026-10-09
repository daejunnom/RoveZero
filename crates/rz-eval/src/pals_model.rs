//! PALS neural model boundary. These are model tensors, not chess authority.
//!
//! The role scheduler supplies Rules-validated candidates. This module validates
//! bounded, finite model inputs and separates private role heads from public memory.
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

mod full_line;
pub use full_line::{PalsFullLineInput, PalsRecordLine, PreparedPalsFullLineTensors};

pub const PALS_MODEL_SCHEMA: &str = "rovezero.pals-model.v1";
pub const PALS_ENCODING_SCHEMA: &str = "rovezero.pals-board-records.v1";
/// Semantic version, distinct from the legacy v2 shared-P/C export layout.
pub const PALS_MODEL_SEMANTICS_V2: &str = "rovezero.pals-model-semantics.v2";
pub const PALS_ENCODING_SCHEMA_V2: &str = "rovezero.pals-board-records.v2";
pub const MAX_LINE_PLY: usize = 256;
pub const QUERY_LINE_COUNT: usize = 3;
pub const METADATA_FEATURES: usize = 16;
pub const RECORD_FEATURES: usize = 16;
/// Only the registered encoder's independent record projection, never the
/// full observation/task identity or an assertion of numerical equivalence.
pub const INDEPENDENT_RECORD_PROJECTION_SEMANTICS: &str =
    "rz-pals-independent-record-fp32/1;16-feature-bits;four-fields;no-record-position;no-cross-record-context";
pub const INDEPENDENT_RECORD_PROJECTION_SEMANTICS_V2: &str =
    "rz-pals-independent-record-fp32/2;16-feature-bits;full-ordered-move-line;masked-256-ply;no-record-position;no-relationships;no-cross-record-context";
pub const QUERY_FEATURES: usize = 16;
pub const DIVERGENCE_FEATURES: usize = 8;
/// Same ordered vocabulary as the versioned verifier-data contract.
pub const V_TASKS: usize = 7;
pub const V_TASK_NAMES: [&str; V_TASKS] = [
    "defend_response",
    "attack_repair",
    "widen_responses",
    "lower_selectivity",
    "resume_task",
    "cross_profile_recheck",
    "defer",
];

/// Both model evidence pins below use exactly this byte framing. Lengths are
/// UTF-8 byte counts, never character counts; there is no delimiter, terminator,
/// JSON serialization, newline normalization or implicit final frame.
pub const PALS_MODEL_PIN_FRAMING: &str =
    "sha256(concat(u64_le(name_utf8.len),name_utf8,u64_le(payload.len),payload));ordered-named-frames;no-normalization";
pub const PALS_REGISTERED_SEMANTICS_DOMAIN: &str = "rz-pals-registered-model-semantics/1";
/// Each config payload is one unsigned 64-bit little-endian integer. The values
/// come from `PalsModelConfig::for_profile`, never an unvalidated asset declaration.
pub const PALS_REGISTERED_CONFIG_FIELD_NAMES: [&str; 14] = [
    "config.width",
    "config.query_heads",
    "config.kv_heads",
    "config.head_dimension",
    "config.board_blocks",
    "config.record_blocks",
    "config.record_fields",
    "config.latent_slots",
    "config.recurrent_blocks",
    "config.iterations",
    "config.ffn_width",
    "config.max_records",
    "config.max_candidates",
    "config.max_divergences",
];
/// These exact UTF-8 definitions, including punctuation, are evidence inputs.
/// They describe the registered graph; the independent source pin additionally
/// identifies the five compiled source texts that implement its boundary.
pub const PALS_REGISTERED_BASE_DEFINITION: &str = concat!(
    "public:piece13x384+square64x384;metadata16->2x384-no-bias;66-board-tokens;",
    "board-2x(rmsnorm-eps1e-6,gqa-q6-kv2-d64,residual,rmsnorm,swiglu1024-no-bias,residual);",
    "record16->4x4;field4->384-no-bias+learned-field4x384;",
    "record-1x-encoder-block-per-record;mean4-fields;no-cross-record-context;",
    "public-k384->128-no-bias;public-v384->128-no-bias;board-before-records;bool-memory-mask;",
    "attention-scale64-pow-minus-half;repeat-kv-groups3;masked-softmax-last-axis;",
    "private:hard-routed-proposer-critic-validator;learned-initial16x384+query16->384-no-bias;",
    "2-iterations-of-2-reader-blocks;shared-cross-gqa-and-latent-self-gqa;private-rmsnorm-swiglu1024;",
    "context-mean16-latents;wdl384->3-no-bias;",
    "critic-divergence8->384-no-bias-times-context->1-no-bias;validator384->7-no-bias;",
    "validator-task-order:defend_response,attack_repair,widen_responses,lower_selectivity,resume_task,cross_profile_recheck,defer;",
    "candidate:from64x384,to64x384,promotion5x384;concat1152->384-no-bias;",
    "attention-and-output-inactive-mask-minus1e9;fp32;",
    "board-i64[B,64];metadata-f32[B,16];records-f32[B,R,16];record-mask-bool[B,R];",
    "candidates-i64[B,C,3];candidate-mask-bool[B,C];query-f32[B,16];",
    "critic-divergence-f32[B,D,8];critic-divergence-mask-bool[B,D];",
    "empty-record-candidate-divergence-physical-slot1-false-mask"
);
pub const PALS_REGISTERED_SUMMARY_DEFINITION: &str =
    "record16-and-query16-summary-only;no-line-tensors;no-local-relation-tensors";
pub const PALS_REGISTERED_FULL_LINE_DEFINITION: &str = concat!(
    "line:shared-record-and-query-encoder;max-ply256;",
    "from64x64,to64x64,promotion5x64;concat192->64-no-bias+ply256x64;",
    "masked-tokens-zero-before-lookup;mask-after-embedding-and-each-residual;",
    "2x-conv1d64->64-kernel3-dilation1,2-padding1,2-no-bias;residual-silu;",
    "pool64->1-no-bias;inactive-score-minus1e9;softmax-times-mask;weighted-sum;empty-line-exact-zero;",
    "record-line64->384-no-bias-added-after-independent-record-encoding-before-kv;",
    "query-prefix-proposal-counter-ordered;concat3x64->384-no-bias-added-to-query-conditioning;",
    "relations:local-parent-and-supersedes-i64;absent-minus1;range-self-cycle-and-required-ref-validation;",
    "independent-public-record-kv-before-private-relation-gather;own-kv256+parent-kv256+supersedes-kv256;",
    "absent-related-kv-zero;concat768->128-with-bias;silu;128->384-no-bias;",
    "active-record-and-any-relation-mask;masked-sum-divide-clamped-active-count;added-to-initial-conditioning;",
    "record-line-tokens-i64[B,R,256,3];record-line-mask-bool[B,R,256];",
    "query-line-tokens-i64[B,3,256,3];query-line-mask-bool[B,3,256];record-relations-i64[B,R,2];",
    "record-feature6-origin-state-id-zero;query-feature6-revision-and7-deadline-zero;",
    "global-ids-revisions-history-hashes-typed-identity-only;deadlines-execution-control-only"
);
pub const PALS_REGISTERED_LEGACY_HEAD_DEFINITION: &str =
    "candidate-embedding-times-context;384->1-no-bias;linear-scalar";
pub const PALS_REGISTERED_INTERACTION_HEAD_DEFINITION: &str =
    "concat(candidate-embedding384,context384);768->128-with-bias;silu;128->1-no-bias;scalar";

pub const PALS_COMPILED_MODEL_BOUNDARY_DOMAIN: &str = "rz-pals-compiled-model-boundary-sources/1";
/// Source names are repository-relative logical names, in hash order. Payloads
/// are the exact UTF-8 bytes captured by `include_str!` at compilation, including
/// their original line endings. The proof is limited to these model boundary
/// sources; it does not identify weights, an ONNX file, the whole binary, Rules,
/// a provider implementation or a successful model execution.
pub const PALS_COMPILED_MODEL_BOUNDARY_SOURCE_NAMES: [&str; 5] = [
    "crates/rz-eval/src/pals_model.rs",
    "crates/rz-eval/src/pals_model/full_line.rs",
    "experiments/model-research/pals/src/rz_pals_model/config.py",
    "experiments/model-research/pals/src/rz_pals_model/model.py",
    "experiments/model-research/pals/src/rz_pals_model/onnx_shared.py",
];

/// Hashes `domain` followed by the five named source payloads above using
/// `PALS_MODEL_PIN_FRAMING`. This is a separate observation pin: no model epoch,
/// private weight identity, input digest or cache namespace uses this getter.
pub fn compiled_model_boundary_sha256() -> [u8; 32] {
    let mut hash = Sha256::new();
    hash_model_pin_frame(
        &mut hash,
        "domain",
        PALS_COMPILED_MODEL_BOUNDARY_DOMAIN.as_bytes(),
    );
    let sources = [
        include_str!("pals_model.rs"),
        include_str!("pals_model/full_line.rs"),
        include_str!("../../../experiments/model-research/pals/src/rz_pals_model/config.py"),
        include_str!("../../../experiments/model-research/pals/src/rz_pals_model/model.py"),
        include_str!("../../../experiments/model-research/pals/src/rz_pals_model/onnx_shared.py"),
    ];
    for (name, source) in PALS_COMPILED_MODEL_BOUNDARY_SOURCE_NAMES
        .into_iter()
        .zip(sources)
    {
        hash_model_pin_frame(&mut hash, name, source.as_bytes());
    }
    hash.finalize().into()
}

fn hash_model_pin_frame(hash: &mut Sha256, name: &str, payload: &[u8]) {
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name.as_bytes());
    hash.update((payload.len() as u64).to_le_bytes());
    hash.update(payload);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsRole {
    Proposer,
    Critic,
    Validator,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsModelProfile {
    /// Missing profile fields in historical assets have precisely V1 semantics.
    #[default]
    LegacySummaryV1,
    FullLineV2,
    InteractionHeadV2,
    FullLineInteractionV2,
}
impl PalsModelProfile {
    pub const fn is_legacy(&self) -> bool {
        matches!(self, Self::LegacySummaryV1)
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LegacySummaryV1 => "legacy_summary_v1",
            Self::FullLineV2 => "full_line_v2",
            Self::InteractionHeadV2 => "interaction_head_v2",
            Self::FullLineInteractionV2 => "full_line_interaction_v2",
        }
    }
    pub const fn uses_full_line(self) -> bool {
        matches!(self, Self::FullLineV2 | Self::FullLineInteractionV2)
    }
    pub const fn uses_interaction_head(self) -> bool {
        matches!(self, Self::InteractionHeadV2 | Self::FullLineInteractionV2)
    }
    pub const fn model_semantics(self) -> &'static str {
        if self.is_legacy() {
            PALS_MODEL_SCHEMA
        } else {
            PALS_MODEL_SEMANTICS_V2
        }
    }
    pub const fn encoding_schema(self) -> &'static str {
        if self.is_legacy() {
            PALS_ENCODING_SCHEMA
        } else {
            PALS_ENCODING_SCHEMA_V2
        }
    }
    pub const fn independent_projection_semantics(self) -> &'static str {
        if self.uses_full_line() {
            INDEPENDENT_RECORD_PROJECTION_SEMANTICS_V2
        } else {
            INDEPENDENT_RECORD_PROJECTION_SEMANTICS
        }
    }
    /// Visits every registered semantic pin frame in canonical order without
    /// allocating. Text payloads are exact UTF-8. Config payloads are u64 LE.
    /// Independent producers can hash these frames with the documented framing.
    /// This API is separate from the historical config JSON and all cache keys.
    pub fn registered_semantics_frames(self, mut consume: impl FnMut(&'static str, &[u8])) {
        consume("domain", PALS_REGISTERED_SEMANTICS_DOMAIN.as_bytes());
        consume("model_profile", self.as_str().as_bytes());
        consume("model_semantics", self.model_semantics().as_bytes());
        consume("encoding_schema", self.encoding_schema().as_bytes());
        let config = PalsModelConfig::for_profile(self);
        let values = [
            config.width,
            config.query_heads,
            config.kv_heads,
            config.head_dimension,
            config.board_blocks,
            config.record_blocks,
            config.record_fields,
            config.latent_slots,
            config.recurrent_blocks,
            config.iterations,
            config.ffn_width,
            config.max_records,
            config.max_candidates,
            config.max_divergences,
        ];
        for (name, value) in PALS_REGISTERED_CONFIG_FIELD_NAMES.into_iter().zip(values) {
            consume(name, &(value as u64).to_le_bytes());
        }
        consume(
            "structure.public_and_private",
            PALS_REGISTERED_BASE_DEFINITION.as_bytes(),
        );
        consume(
            "structure.line_encoder",
            if self.uses_full_line() {
                PALS_REGISTERED_FULL_LINE_DEFINITION
            } else {
                PALS_REGISTERED_SUMMARY_DEFINITION
            }
            .as_bytes(),
        );
        consume(
            "structure.candidate_head",
            if self.uses_interaction_head() {
                PALS_REGISTERED_INTERACTION_HEAD_DEFINITION
            } else {
                PALS_REGISTERED_LEGACY_HEAD_DEFINITION
            }
            .as_bytes(),
        );
    }
    /// Identifies the registered canonical config and graph meaning for this
    /// profile. It is not an asset-provided SHA, weight identity, source proof or
    /// execution success, and is never substituted for existing V1 digests.
    pub fn registered_semantics_sha256(self) -> [u8; 32] {
        let mut hash = Sha256::new();
        self.registered_semantics_frames(|name, payload| {
            hash_model_pin_frame(&mut hash, name, payload)
        });
        hash.finalize().into()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PalsModelConfig {
    #[serde(default, skip_serializing_if = "PalsModelProfile::is_legacy")]
    pub profile: PalsModelProfile,
    pub width: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dimension: usize,
    pub board_blocks: usize,
    pub record_blocks: usize,
    pub record_fields: usize,
    pub latent_slots: usize,
    pub recurrent_blocks: usize,
    pub iterations: usize,
    pub ffn_width: usize,
    pub max_records: usize,
    pub max_candidates: usize,
    pub max_divergences: usize,
}
impl Default for PalsModelConfig {
    fn default() -> Self {
        Self::baseline()
    }
}
impl PalsModelConfig {
    pub const fn baseline() -> Self {
        Self {
            profile: PalsModelProfile::LegacySummaryV1,
            width: 384,
            query_heads: 6,
            kv_heads: 2,
            head_dimension: 64,
            board_blocks: 2,
            record_blocks: 1,
            record_fields: 4,
            latent_slots: 16,
            recurrent_blocks: 2,
            iterations: 2,
            ffn_width: 1024,
            max_records: 128,
            max_candidates: 256,
            max_divergences: 128,
        }
    }
    pub const fn for_profile(profile: PalsModelProfile) -> Self {
        Self {
            profile,
            ..Self::baseline()
        }
    }
    /// New model initialization selects this profile explicitly. `Default` and
    /// `baseline()` remain legacy for existing caller and asset compatibility.
    pub const fn full_line_interaction_v2() -> Self {
        Self::for_profile(PalsModelProfile::FullLineInteractionV2)
    }
    /// Every profile has one registered configuration; no silent OOM resizing.
    pub fn validate(&self) -> Result<(), PalsModelError> {
        if self != &Self::for_profile(self.profile) {
            return Err(PalsModelError::UnsupportedConfiguration);
        }
        Ok(())
    }
    pub fn latent_elements(&self) -> usize {
        self.latent_slots * self.width
    }
    pub fn public_memory_tokens(&self, records: usize) -> Result<usize, PalsModelError> {
        self.validate()?;
        if records > self.max_records {
            return Err(PalsModelError::Capacity("records"));
        }
        // The ONNX profile uses one masked padding slot for an empty record set.
        Ok(66 + records.max(1))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PalsRecordToken {
    pub record_id: u64,
    pub revision: u64,
    pub critical: bool,
    /// Structured numeric fields; Missing/NotExamined must be represented by the
    /// caller's explicit status field, never supplied as a fabricated result.
    pub features: [f32; RECORD_FEATURES],
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PalsCandidateToken {
    pub from: u8,
    pub to: u8,
    /// Common Move16 mapping: 0=none, 1=queen, 2=rook, 3=bishop, 4=knight.
    pub promotion: u8,
}
impl PalsCandidateToken {
    pub fn validate(self) -> Result<(), PalsModelError> {
        if self.from >= 64 || self.to >= 64 || self.from == self.to || self.promotion > 4 {
            return Err(PalsModelError::InvalidCandidate);
        }
        Ok(())
    }
    pub fn packed(self) -> Result<u16, PalsModelError> {
        self.validate()?;
        Ok(self.from as u16 | (self.to as u16) << 6 | (self.promotion as u16) << 12)
    }
    pub fn from_packed(value: u16) -> Result<Self, PalsModelError> {
        if value & 0x8000 != 0 {
            return Err(PalsModelError::InvalidCandidate);
        }
        let result = Self {
            from: (value & 63) as u8,
            to: ((value >> 6) & 63) as u8,
            promotion: ((value >> 12) & 7) as u8,
        };
        result.validate()?;
        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PalsModelInput {
    pub role: PalsRole,
    /// a1..h8; 0=empty, 1..6=white P/N/B/R/Q/K, 7..12=black.
    pub board: Vec<u8>,
    pub metadata: [f32; METADATA_FEATURES],
    pub records: Vec<PalsRecordToken>,
    /// Every required critical record ID is checked before model admission.
    pub required_critical_records: Vec<u64>,
    pub candidates: Vec<PalsCandidateToken>,
    pub divergence_features: Vec<[f32; DIVERGENCE_FEATURES]>,
    pub query: [f32; QUERY_FEATURES],
    pub situation_revision: u64,
    /// Exact Rules history identity. The model's compact features do not replace
    /// exact repetition/termination history; different histories cannot alias.
    pub history_digest: [u8; 32],
    pub model_epoch: [u8; 32],
    /// Explicit V2 inputs. Historical JSON omits this field and retains its
    /// original byte representation and canonical key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_line: Option<PalsFullLineInput>,
}
impl PalsModelInput {
    /// Physical projection identities leave the full canonical input unchanged.
    /// The empty view keeps its actual zero-feature, false-mask padding slot.
    pub fn independent_public_plan(
        &self,
        config: &PalsModelConfig,
    ) -> Result<IndependentPublicPlan, PalsModelError> {
        self.validate(config)?;
        let mut board = Sha256::new();
        if config.profile.is_legacy() {
            board.update(b"rz-pals-contextual-board-fp32/1");
        } else {
            board.update(b"rz-pals-contextual-board-fp32/2");
            board.update(config.profile.as_str().as_bytes());
            board.update(config.profile.encoding_schema().as_bytes());
        }
        board.update(self.model_epoch);
        board.update(&self.board);
        for value in self.metadata {
            board.update(value.to_bits().to_le_bytes());
        }
        let features: Vec<_> = if self.records.is_empty() {
            vec![[0.; RECORD_FEATURES]]
        } else {
            self.records.iter().map(|record| record.features).collect()
        };
        let record_contents = features
            .iter()
            .enumerate()
            .map(|(slot, values)| {
                let mut hash = Sha256::new();
                hash.update(config.profile.independent_projection_semantics().as_bytes());
                if !config.profile.is_legacy() {
                    hash.update(config.profile.as_str().as_bytes());
                    hash.update(config.profile.encoding_schema().as_bytes());
                }
                hash.update(self.model_epoch);
                for value in values {
                    hash.update(value.to_bits().to_le_bytes());
                }
                if let Some(lines) = &self.full_line {
                    full_line::hash_moves(
                        &mut hash,
                        lines.records.get(slot).map_or(&[], |r| r.moves.as_slice()),
                    );
                }
                hash.finalize().into()
            })
            .collect();
        Ok(IndependentPublicPlan {
            board_content: board.finalize().into(),
            record_contents,
            features,
            record_mask: (0..self.records.len().max(1))
                .map(|index| index < self.records.len())
                .collect(),
            record_lines: self.full_line.as_ref().map(|lines| {
                if lines.records.is_empty() {
                    vec![Vec::new()]
                } else {
                    lines.records.iter().map(|r| r.moves.clone()).collect()
                }
            }),
        })
    }
    pub fn validate(&self, config: &PalsModelConfig) -> Result<(), PalsModelError> {
        config.validate()?;
        if self.board.len() != 64 || self.board.iter().any(|v| *v > 12) {
            return Err(PalsModelError::Shape("board"));
        }
        if self.records.len() > config.max_records {
            return Err(PalsModelError::Capacity("records"));
        }
        if self.required_critical_records.len() > config.max_records {
            return Err(PalsModelError::Capacity("required critical records"));
        }
        if self.candidates.len() > config.max_candidates {
            return Err(PalsModelError::Capacity("candidates"));
        }
        if self.divergence_features.len() > config.max_divergences {
            return Err(PalsModelError::Capacity("divergences"));
        }
        if !self
            .metadata
            .iter()
            .chain(self.query.iter())
            .all(|v| v.is_finite())
        {
            return Err(PalsModelError::NonFinite("metadata/query"));
        }
        let mut records = BTreeSet::new();
        let mut critical = BTreeSet::new();
        for record in &self.records {
            if !records.insert(record.record_id) {
                return Err(PalsModelError::Duplicate("record"));
            }
            if record.critical {
                critical.insert(record.record_id);
            }
            if !record.features.iter().all(|v| v.is_finite()) {
                return Err(PalsModelError::NonFinite("record"));
            }
        }
        let required: BTreeSet<_> = self.required_critical_records.iter().copied().collect();
        if required.len() != self.required_critical_records.len() {
            return Err(PalsModelError::Duplicate("critical record"));
        }
        if !required.is_subset(&critical) {
            return Err(PalsModelError::MissingCriticalRecord);
        }
        let mut candidates = BTreeSet::new();
        for candidate in &self.candidates {
            candidate.validate()?;
            if !candidates.insert(*candidate) {
                return Err(PalsModelError::Duplicate("candidate"));
            }
        }
        if !self
            .divergence_features
            .iter()
            .flatten()
            .all(|v| v.is_finite())
        {
            return Err(PalsModelError::NonFinite("divergence"));
        }
        if self.role != PalsRole::Critic && !self.divergence_features.is_empty() {
            return Err(PalsModelError::PrivateRoleInput);
        }
        // V uses its private task head. No private V feature is present in this
        // public-memory input shape or admitted into P/C canonical inputs.
        if config.profile.uses_full_line()
            && (self.records.iter().any(|record| record.features[6] != 0.)
                || self.query[6..8].iter().any(|value| *value != 0.))
        {
            // Global state/revision identifiers are exact typed identity, and
            // the deadline belongs to execution control. V2 never treats their
            // floating-point conversions as learned semantic features.
            return Err(PalsModelError::NonSemanticMetadata);
        }
        match (&self.full_line, config.profile.uses_full_line()) {
            (Some(lines), true) => lines.validate(self.records.len())?,
            (None, false) => {}
            (None, true) => return Err(PalsModelError::Shape("required full_line payload")),
            (Some(_), false) => return Err(PalsModelError::UnexpectedFullLine),
        }
        Ok(())
    }
    /// Explicit little-endian f32 bytes preserve candidate order, masks, revision
    /// and model epoch. Negative zero remains a distinct actual tensor input.
    pub fn canonical_input_key(
        &self,
        config: &PalsModelConfig,
    ) -> Result<[u8; 32], PalsModelError> {
        self.validate(config)?;
        let mut hash = Sha256::new();
        hash.update(config.profile.model_semantics().as_bytes());
        hash.update(config.profile.encoding_schema().as_bytes());
        if !config.profile.is_legacy() {
            hash.update(config.profile.as_str().as_bytes());
        }
        hash.update([match self.role {
            PalsRole::Proposer => 0,
            PalsRole::Critic => 1,
            PalsRole::Validator => 2,
        }]);
        hash.update(self.model_epoch);
        hash.update(self.history_digest);
        hash.update(self.situation_revision.to_le_bytes());
        hash.update(&self.board);
        for value in self.metadata.iter().chain(self.query.iter()) {
            hash.update(value.to_bits().to_le_bytes());
        }
        hash.update((self.records.len() as u64).to_le_bytes());
        for record in &self.records {
            hash.update(record.record_id.to_le_bytes());
            hash.update(record.revision.to_le_bytes());
            hash.update([u8::from(record.critical)]);
            for value in record.features {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        // Critical requirements authorize context; their order is not a tensor.
        let mut required = self.required_critical_records.clone();
        required.sort_unstable();
        hash.update((required.len() as u64).to_le_bytes());
        for value in required {
            hash.update(value.to_le_bytes());
        }
        hash.update((self.candidates.len() as u64).to_le_bytes());
        for candidate in &self.candidates {
            hash.update(candidate.packed()?.to_le_bytes());
        }
        hash.update((self.divergence_features.len() as u64).to_le_bytes());
        for value in self.divergence_features.iter().flatten() {
            hash.update(value.to_bits().to_le_bytes());
        }
        if let Some(lines) = &self.full_line {
            lines.hash_records(&mut hash);
            lines.hash_query(&mut hash);
        }
        Ok(hash.finalize().into())
    }

    /// The public encoder is role-neutral and does not consume candidates,
    /// query, divergence or private latent. Sharing this key never authorizes
    /// sharing private outputs or reusing a different role's latent.
    pub fn public_memory_key(&self, config: &PalsModelConfig) -> Result<[u8; 32], PalsModelError> {
        self.validate(config)?;
        let mut hash = Sha256::new();
        hash.update(config.profile.model_semantics().as_bytes());
        hash.update(config.profile.encoding_schema().as_bytes());
        if !config.profile.is_legacy() {
            hash.update(config.profile.as_str().as_bytes());
        }
        hash.update(b"public-memory-fp32");
        hash.update(self.model_epoch);
        hash.update(self.history_digest);
        hash.update(self.situation_revision.to_le_bytes());
        hash.update(&self.board);
        for value in self.metadata {
            hash.update(value.to_bits().to_le_bytes());
        }
        hash.update((self.records.len() as u64).to_le_bytes());
        for record in &self.records {
            hash.update(record.record_id.to_le_bytes());
            hash.update(record.revision.to_le_bytes());
            hash.update([u8::from(record.critical)]);
            for value in record.features {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        // Whole-view caching preserves selected order, revision and local
        // relationships. Independent pages below intentionally omit references.
        if let Some(lines) = &self.full_line {
            lines.hash_records(&mut hash);
        }
        Ok(hash.finalize().into())
    }

    /// Move once into a physical lease. B=1 tensors retain semantic lengths so
    /// masked empty slots are stripped from outputs before contract validation.
    pub fn prepare_tensors(
        &self,
        config: &PalsModelConfig,
    ) -> Result<PreparedPalsTensors, PalsModelError> {
        self.validate(config)?;
        let records = self.records.len().max(1);
        let candidates = self.candidates.len().max(1);
        let divergences = self.divergence_features.len().max(1);
        let mut record_features = vec![0.; records * RECORD_FEATURES];
        for (slot, record) in self.records.iter().enumerate() {
            record_features[slot * RECORD_FEATURES..(slot + 1) * RECORD_FEATURES]
                .copy_from_slice(&record.features);
        }
        let mut candidate_codes = vec![0; candidates * 3];
        for (slot, candidate) in self.candidates.iter().enumerate() {
            candidate_codes[slot * 3..(slot + 1) * 3].copy_from_slice(&[
                candidate.from as i64,
                candidate.to as i64,
                candidate.promotion as i64,
            ]);
        }
        let mut divergence_features = vec![0.; divergences * DIVERGENCE_FEATURES];
        for (slot, features) in self.divergence_features.iter().enumerate() {
            divergence_features[slot * DIVERGENCE_FEATURES..(slot + 1) * DIVERGENCE_FEATURES]
                .copy_from_slice(features);
        }
        Ok(PreparedPalsTensors {
            role: self.role,
            input_key: self.canonical_input_key(config)?,
            public_memory_key: self.public_memory_key(config)?,
            board: self.board.iter().map(|v| *v as i64).collect(),
            metadata: self.metadata.to_vec(),
            records: record_features,
            record_mask: (0..records).map(|i| i < self.records.len()).collect(),
            candidates: candidate_codes,
            candidate_mask: (0..candidates).map(|i| i < self.candidates.len()).collect(),
            divergences: divergence_features,
            divergence_mask: (0..divergences)
                .map(|i| i < self.divergence_features.len())
                .collect(),
            query: self.query.to_vec(),
            semantic_candidates: self.candidates.len(),
            semantic_divergences: self.divergence_features.len(),
            full_line: self.full_line.as_ref().map(PalsFullLineInput::prepare),
        })
    }
}

/// Ordered full view, with separate allocation identities. Reordering/ID or
/// critical changes still change the existing full key, even if these exact
/// independent feature pages can be physically reused.
#[derive(Clone, Debug, PartialEq)]
pub struct IndependentPublicPlan {
    pub board_content: [u8; 32],
    pub record_contents: Vec<[u8; 32]>,
    pub features: Vec<[f32; RECORD_FEATURES]>,
    pub record_mask: Vec<bool>,
    /// Independent full sequences aligned with feature/mask slots. Local
    /// relation references never enter this projection or its physical key.
    pub record_lines: Option<Vec<Vec<PalsCandidateToken>>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedPalsTensors {
    pub role: PalsRole,
    pub input_key: [u8; 32],
    pub public_memory_key: [u8; 32],
    pub board: Vec<i64>,
    pub metadata: Vec<f32>,
    pub records: Vec<f32>,
    pub record_mask: Vec<bool>,
    pub candidates: Vec<i64>,
    pub candidate_mask: Vec<bool>,
    pub divergences: Vec<f32>,
    pub divergence_mask: Vec<bool>,
    pub query: Vec<f32>,
    pub semantic_candidates: usize,
    pub semantic_divergences: usize,
    pub full_line: Option<PreparedPalsFullLineTensors>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PalsRawOutput {
    pub candidate_logits: Vec<f32>,
    /// Win, draw, loss from the input position's side to move.
    pub wdl_logits: [f32; 3],
    pub divergence_logits: Option<Vec<f32>>,
    pub task_logits: Option<[f32; V_TASKS]>,
    pub private_latent: Vec<f32>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct PalsDecodedOutput {
    pub candidate_policy: Vec<f32>,
    pub wdl: [f32; 3],
    pub divergence_policy: Option<Vec<f32>>,
    pub task_policy: Option<[f32; V_TASKS]>,
}
impl PalsRawOutput {
    pub fn decode(
        &self,
        input: &PalsModelInput,
        config: &PalsModelConfig,
    ) -> Result<PalsDecodedOutput, PalsModelError> {
        input.validate(config)?;
        if self.candidate_logits.len() != input.candidates.len()
            || self.private_latent.len() != config.latent_elements()
        {
            return Err(PalsModelError::Shape("candidate/private_latent output"));
        }
        if !self.private_latent.iter().all(|v| v.is_finite()) {
            return Err(PalsModelError::NonFinite("private_latent"));
        }
        let divergence_policy = match (input.role, &self.divergence_logits) {
            (PalsRole::Critic, Some(logits)) if logits.len() == input.divergence_features.len() => {
                Some(softmax(logits)?)
            }
            (PalsRole::Critic, _) => return Err(PalsModelError::Shape("critic divergence output")),
            (_, None) => None,
            _ => return Err(PalsModelError::PrivateRoleOutput),
        };
        let task_policy = match (input.role, &self.task_logits) {
            (PalsRole::Validator, Some(logits)) => Some(
                softmax(logits)?
                    .try_into()
                    .map_err(|_| PalsModelError::Shape("task output"))?,
            ),
            (PalsRole::Validator, None) => {
                return Err(PalsModelError::Shape("validator task output"))
            }
            (_, None) => None,
            _ => return Err(PalsModelError::PrivateRoleOutput),
        };
        Ok(PalsDecodedOutput {
            candidate_policy: softmax(&self.candidate_logits)?,
            wdl: softmax(&self.wdl_logits)?
                .try_into()
                .map_err(|_| PalsModelError::Shape("wdl output"))?,
            divergence_policy,
            task_policy,
        })
    }
}
fn softmax(values: &[f32]) -> Result<Vec<f32>, PalsModelError> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    if !values.iter().all(|v| v.is_finite()) {
        return Err(PalsModelError::NonFinite("logits"));
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut result: Vec<_> = values.iter().map(|v| ((*v - max) as f64).exp()).collect();
    let sum: f64 = result.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(PalsModelError::NonFinite("softmax sum"));
    }
    Ok(result.drain(..).map(|v| (v / sum) as f32).collect())
}

/// Analytic matrix FLOPs, FMA=2. Activations, masking, softmax, normalization,
/// lookup, data movement and CPU search are reported separately, not fabricated.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PalsMatmulFlops {
    pub public_encode: u64,
    pub role_forward: u64,
}
impl PalsModelConfig {
    pub fn matmul_flops(
        &self,
        role: PalsRole,
        records: usize,
        candidates: usize,
        divergences: usize,
    ) -> Result<PalsMatmulFlops, PalsModelError> {
        self.validate()?;
        if records > self.max_records
            || candidates > self.max_candidates
            || divergences > self.max_divergences
        {
            return Err(PalsModelError::Capacity("FLOPs input"));
        }
        let w = self.width as u64;
        let kv = (self.kv_heads * self.head_dimension) as u64;
        let h = self.ffn_width as u64;
        let n = self.latent_slots as u64;
        // Charge physical padding work as well as semantically active tokens.
        let r = records.max(1) as u64;
        let s = 66 + r;
        let self_attn = |tokens: u64| 2 * tokens * w * (2 * w + 2 * kv) + 4 * tokens * tokens * w;
        let ffn = |tokens: u64| 6 * tokens * w * h;
        let mut public_encode = 2 * 16 * 2 * w
            + self.board_blocks as u64 * (self_attn(66) + ffn(66))
            + r * (2 * 16 * w + self.record_blocks as u64 * (self_attn(4) + ffn(4)))
            + 4 * s * w * kv;
        let cross = 4 * n * w * w + 4 * n * s * w;
        let iterations = (self.recurrent_blocks * self.iterations) as u64;
        let mut role_forward = 2 * 16 * w
            + iterations * (cross + self_attn(n) + ffn(n))
            + 2 * candidates.max(1) as u64 * 3 * w * w
            + 2 * candidates.max(1) as u64 * w
            + 2 * w * 3;
        if role == PalsRole::Critic {
            role_forward +=
                2 * divergences.max(1) as u64 * 8 * w + 2 * divergences.max(1) as u64 * w;
        }
        if role == PalsRole::Validator {
            role_forward += 2 * w * V_TASKS as u64;
        }
        if self.profile.uses_full_line() {
            let line_width = 64_u64;
            let interaction_width = 128_u64;
            // Charge the registered physical 256-ply tensor products,
            // including masked empty lines. Norm/pooling/masks are excluded.
            let line = 2
                * MAX_LINE_PLY as u64
                * (3 * line_width * line_width + 2 * 3 * line_width * line_width + line_width);
            public_encode += r * (line + 2 * line_width * w);
            role_forward += 3 * line
                + 2 * 3 * line_width * w
                + r * (2 * 3 * 2 * kv * interaction_width + 2 * interaction_width * w);
        }
        if self.profile.uses_interaction_head() {
            role_forward += candidates.max(1) as u64 * (2 * 2 * w * 128 + 2 * 128 - 2 * w);
        }
        Ok(PalsMatmulFlops {
            public_encode,
            role_forward,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PalsModelError {
    UnsupportedConfiguration,
    Capacity(&'static str),
    Shape(&'static str),
    NonFinite(&'static str),
    Duplicate(&'static str),
    InvalidCandidate,
    MissingCriticalRecord,
    PrivateRoleInput,
    PrivateRoleOutput,
    UnexpectedFullLine,
    InvalidRelationship,
    MissingRequiredRelationship,
    RelationshipCycle,
    NonSemanticMetadata,
}
impl std::fmt::Display for PalsModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PALS model boundary: {self:?}")
    }
}
impl std::error::Error for PalsModelError {}

#[cfg(test)]
mod tests {
    use super::*;
    fn input() -> PalsModelInput {
        PalsModelInput {
            role: PalsRole::Proposer,
            board: vec![0; 64],
            metadata: [0.; 16],
            records: vec![PalsRecordToken {
                record_id: 1,
                revision: 1,
                critical: true,
                features: [0.; 16],
            }],
            required_critical_records: vec![1],
            candidates: vec![
                PalsCandidateToken {
                    from: 12,
                    to: 28,
                    promotion: 0,
                },
                PalsCandidateToken {
                    from: 11,
                    to: 27,
                    promotion: 0,
                },
            ],
            divergence_features: vec![],
            query: [0.; 16],
            situation_revision: 1,
            history_digest: [9; 32],
            model_epoch: [7; 32],
            full_line: None,
        }
    }
    fn line(moves: &[(u8, u8, u8)]) -> Vec<PalsCandidateToken> {
        moves
            .iter()
            .map(|&(from, to, promotion)| PalsCandidateToken {
                from,
                to,
                promotion,
            })
            .collect()
    }
    fn v2_input() -> PalsModelInput {
        let mut input = input();
        input.records.push(PalsRecordToken {
            record_id: 2,
            revision: 1,
            critical: false,
            features: [0.; 16],
        });
        input.full_line = Some(PalsFullLineInput {
            records: vec![
                PalsRecordLine {
                    moves: line(&[(12, 28, 0), (52, 36, 0), (6, 21, 0)]),
                    parent: None,
                    supersedes: None,
                    parent_required: false,
                    supersedes_required: false,
                },
                PalsRecordLine {
                    moves: line(&[(11, 27, 0), (51, 35, 0), (1, 18, 0)]),
                    parent: Some(0),
                    supersedes: None,
                    parent_required: true,
                    supersedes_required: false,
                },
            ],
            query_prefix: line(&[(12, 28, 0), (52, 36, 0), (6, 21, 0)]),
            query_proposal: line(&[(11, 27, 0), (51, 35, 0), (1, 18, 0)]),
            query_counter: vec![],
        });
        input
    }
    #[test]
    fn legacy_json_and_registered_profile_domains_are_explicit() {
        let legacy = PalsModelConfig::baseline();
        let json = serde_json::to_string(&legacy).unwrap();
        assert!(!json.contains("profile"));
        assert_eq!(
            serde_json::from_str::<PalsModelConfig>(&json).unwrap(),
            legacy
        );
        assert_eq!(PalsModelConfig::default(), legacy);
        let old_input = serde_json::to_string(&input()).unwrap();
        assert!(!old_input.contains("full_line"));
        assert_eq!(
            serde_json::from_str::<PalsModelInput>(&old_input).unwrap(),
            input()
        );
        for profile in [
            PalsModelProfile::FullLineV2,
            PalsModelProfile::InteractionHeadV2,
            PalsModelProfile::FullLineInteractionV2,
        ] {
            let config = PalsModelConfig::for_profile(profile);
            config.validate().unwrap();
            assert!(serde_json::to_string(&config)
                .unwrap()
                .contains(profile.as_str()));
            assert!(input().validate(&config).is_err() == profile.uses_full_line());
        }
        let bad = json.trim_end_matches('}').to_owned() + ",\"profile\":\"unknown_v2\"}";
        assert!(serde_json::from_str::<PalsModelConfig>(&bad).is_err());
        assert_eq!(
            v2_input().validate(&legacy),
            Err(PalsModelError::UnexpectedFullLine)
        );
        // Frozen byte-contract digests of the pre-V2 fixture, independently
        // computed from its original little-endian V1 field ordering.
        let hex = |key: [u8; 32]| key.iter().map(|v| format!("{v:02x}")).collect::<String>();
        assert_eq!(
            hex(input().canonical_input_key(&legacy).unwrap()),
            "875b695a906826b469416043a69310753f6a5aa67932e304f73a5e8d4a051719"
        );
        assert_eq!(
            hex(input().public_memory_key(&legacy).unwrap()),
            "0920d857d2822a67289ca3facf3739c3e6c32e8595e9298f4e08ed18fcceb731"
        );
    }
    #[test]
    fn registered_semantics_pin_matches_independent_framed_reference() {
        // SHA-256 references computed by a dependency-free Python producer using
        // struct.pack('<Q', ...), the published definitions and canonical config.
        // These are separate observation pins, including for legacy, so the V1
        // byte-contract input/public digests above retain their original values.
        let references = [
            (
                PalsModelProfile::LegacySummaryV1,
                "ba0579e7e52458775a73e9fd8f47c69446266e362c7435e70bb1ede20dbce5a6",
            ),
            (
                PalsModelProfile::FullLineV2,
                "01882c814de556b5a9aed8069b82814c6596e4a7851c4b0021e01fac11039d7c",
            ),
            (
                PalsModelProfile::InteractionHeadV2,
                "62c59138292088104252ef3590df6304c32cfb2c5b6bc51ad4e10ad8c5fc5f28",
            ),
            (
                PalsModelProfile::FullLineInteractionV2,
                "7ff4791662ade6f7e6b8ed88629769f63756c0d33d357a7f414640f60b1a752b",
            ),
        ];
        let mut unique = BTreeSet::new();
        for (profile, expected) in references {
            let pin = profile.registered_semantics_sha256();
            assert_eq!(
                pin.iter()
                    .map(|value| format!("{value:02x}"))
                    .collect::<String>(),
                expected
            );
            assert!(unique.insert(pin));
            let mut frames = Vec::new();
            profile.registered_semantics_frames(|name, payload| {
                frames.push((name, payload.to_vec()));
            });
            assert_eq!(frames.len(), 4 + 14 + 3);
            assert_eq!(
                frames[0],
                (
                    "domain",
                    PALS_REGISTERED_SEMANTICS_DOMAIN.as_bytes().to_vec()
                )
            );
            assert_eq!(frames[1].1, profile.as_str().as_bytes());
            let registered_values = [384_u64, 6, 2, 64, 2, 1, 4, 16, 2, 2, 1024, 128, 256, 128];
            for ((name, bytes), (expected_name, value)) in frames[4..18].iter().zip(
                PALS_REGISTERED_CONFIG_FIELD_NAMES
                    .into_iter()
                    .zip(registered_values),
            ) {
                assert_eq!(*name, expected_name);
                assert_eq!(bytes.as_slice(), value.to_le_bytes());
            }
            assert_eq!(
                frames[19].1,
                if profile.uses_full_line() {
                    PALS_REGISTERED_FULL_LINE_DEFINITION
                } else {
                    PALS_REGISTERED_SUMMARY_DEFINITION
                }
                .as_bytes()
            );
            assert_eq!(
                frames[20].1,
                if profile.uses_interaction_head() {
                    PALS_REGISTERED_INTERACTION_HEAD_DEFINITION
                } else {
                    PALS_REGISTERED_LEGACY_HEAD_DEFINITION
                }
                .as_bytes()
            );
        }
    }
    #[test]
    fn model_pin_framing_binds_domain_names_lengths_and_order() {
        let digest = |frames: &[(&str, &[u8])]| -> [u8; 32] {
            let mut hash = Sha256::new();
            for (name, payload) in frames {
                hash_model_pin_frame(&mut hash, name, payload);
            }
            hash.finalize().into()
        };
        assert_ne!(digest(&[("ab", b"c")]), digest(&[("a", b"bc")]));
        assert_ne!(
            digest(&[("source", b"ab"), ("source", b"c")]),
            digest(&[("source", b"a"), ("source", b"bc")])
        );
        assert_ne!(
            digest(&[("a", b"1"), ("b", b"2")]),
            digest(&[("b", b"2"), ("a", b"1")])
        );
        assert_ne!(
            digest(&[("domain", PALS_REGISTERED_SEMANTICS_DOMAIN.as_bytes())]),
            digest(&[("domain", PALS_COMPILED_MODEL_BOUNDARY_DOMAIN.as_bytes())])
        );
        // Raw source framing intentionally distinguishes original line endings.
        assert_ne!(
            digest(&[("source", b"line\n")]),
            digest(&[("source", b"line\r\n")])
        );
    }
    #[test]
    fn full_middle_moves_and_query_lines_fence_actual_input_and_independent_pages() {
        let config = PalsModelConfig::full_line_interaction_v2();
        let original = v2_input();
        let mut changed = original.clone();
        changed.full_line.as_mut().unwrap().records[0].moves[1].to = 44;
        assert_ne!(
            original.canonical_input_key(&config).unwrap(),
            changed.canonical_input_key(&config).unwrap()
        );
        assert_ne!(
            original.public_memory_key(&config).unwrap(),
            changed.public_memory_key(&config).unwrap()
        );
        assert_ne!(
            original
                .independent_public_plan(&config)
                .unwrap()
                .record_contents,
            changed
                .independent_public_plan(&config)
                .unwrap()
                .record_contents
        );
        changed = original.clone();
        changed.full_line.as_mut().unwrap().query_proposal[1].to = 43;
        assert_ne!(
            original.canonical_input_key(&config).unwrap(),
            changed.canonical_input_key(&config).unwrap()
        );
        assert_eq!(
            original.public_memory_key(&config).unwrap(),
            changed.public_memory_key(&config).unwrap()
        );
        assert_eq!(
            original.independent_public_plan(&config).unwrap(),
            changed.independent_public_plan(&config).unwrap()
        );
        let full_line = PalsModelConfig::for_profile(PalsModelProfile::FullLineV2);
        assert_ne!(
            original.canonical_input_key(&config).unwrap(),
            original.canonical_input_key(&full_line).unwrap()
        );
        assert_ne!(
            original
                .independent_public_plan(&config)
                .unwrap()
                .record_contents,
            original
                .independent_public_plan(&full_line)
                .unwrap()
                .record_contents
        );
    }
    #[test]
    fn relationships_are_private_features_with_exact_whole_view_identity() {
        let config = PalsModelConfig::full_line_interaction_v2();
        let original = v2_input();
        let mut changed = original.clone();
        let record = &mut changed.full_line.as_mut().unwrap().records[1];
        record.parent = None;
        record.parent_required = false;
        record.supersedes = Some(0);
        assert_ne!(
            original.canonical_input_key(&config).unwrap(),
            changed.canonical_input_key(&config).unwrap()
        );
        assert_ne!(
            original.public_memory_key(&config).unwrap(),
            changed.public_memory_key(&config).unwrap()
        );
        assert_eq!(
            original.independent_public_plan(&config).unwrap(),
            changed.independent_public_plan(&config).unwrap()
        );
        assert_ne!(
            original
                .prepare_tensors(&config)
                .unwrap()
                .full_line
                .unwrap()
                .record_relations,
            changed
                .prepare_tensors(&config)
                .unwrap()
                .full_line
                .unwrap()
                .record_relations
        );
        changed = original.clone();
        changed.full_line.as_mut().unwrap().records[1].parent = None;
        assert_eq!(
            changed.validate(&config),
            Err(PalsModelError::MissingRequiredRelationship)
        );
        changed = original.clone();
        changed.full_line.as_mut().unwrap().records[1].parent = Some(2);
        assert_eq!(
            changed.validate(&config),
            Err(PalsModelError::InvalidRelationship)
        );
        changed.full_line.as_mut().unwrap().records[1].parent = Some(1);
        assert_eq!(
            changed.validate(&config),
            Err(PalsModelError::InvalidRelationship)
        );
        changed = original.clone();
        changed.full_line.as_mut().unwrap().records[0].supersedes = Some(1);
        assert_eq!(
            changed.validate(&config),
            Err(PalsModelError::RelationshipCycle)
        );
    }
    #[test]
    fn full_line_capacity_shapes_tokens_and_masked_padding_are_bounded() {
        let config = PalsModelConfig::full_line_interaction_v2();
        let original = v2_input();
        let prepared = original.prepare_tensors(&config).unwrap();
        let lines = prepared.full_line.unwrap();
        assert_eq!(lines.record_line_tokens.len(), 2 * MAX_LINE_PLY * 3);
        assert_eq!(lines.record_line_mask.len(), 2 * MAX_LINE_PLY);
        assert_eq!(lines.query_line_tokens.len(), 3 * MAX_LINE_PLY * 3);
        assert_eq!(lines.query_line_mask.len(), 3 * MAX_LINE_PLY);
        assert_eq!(lines.record_relations, vec![-1, -1, 0, -1]);
        assert_eq!(
            &lines.record_line_tokens[..9],
            &[12, 28, 0, 52, 36, 0, 6, 21, 0]
        );
        for slot in 0..2 {
            assert!(
                lines.record_line_mask[slot * MAX_LINE_PLY..slot * MAX_LINE_PLY + 3]
                    .iter()
                    .all(|m| *m)
            );
            assert!(
                lines.record_line_mask[slot * MAX_LINE_PLY + 3..(slot + 1) * MAX_LINE_PLY]
                    .iter()
                    .all(|m| !*m)
            );
            assert!(lines.record_line_tokens
                [(slot * MAX_LINE_PLY + 3) * 3..(slot + 1) * MAX_LINE_PLY * 3]
                .iter()
                .all(|v| *v == 0));
        }
        assert!(lines.query_line_mask[2 * MAX_LINE_PLY..]
            .iter()
            .all(|m| !*m));
        let mut changed = original.clone();
        changed.full_line.as_mut().unwrap().records.pop();
        assert_eq!(
            changed.validate(&config),
            Err(PalsModelError::Shape("full_line records"))
        );
        changed = original.clone();
        changed.full_line.as_mut().unwrap().query_counter =
            vec![original.candidates[0]; MAX_LINE_PLY + 1];
        assert_eq!(
            changed.validate(&config),
            Err(PalsModelError::Capacity("line ply"))
        );
        changed = original.clone();
        changed.full_line.as_mut().unwrap().records[0].moves[1].promotion = 5;
        assert_eq!(
            changed.validate(&config),
            Err(PalsModelError::InvalidCandidate)
        );
        changed = original.clone();
        changed.records.clear();
        changed.required_critical_records.clear();
        changed.full_line.as_mut().unwrap().records.clear();
        let empty = changed.prepare_tensors(&config).unwrap();
        assert_eq!(empty.record_mask, vec![false]);
        let empty_line = empty.full_line.unwrap();
        assert!(empty_line.record_line_tokens.iter().all(|v| *v == 0));
        assert!(empty_line.record_line_mask.iter().all(|v| !*v));
        assert_eq!(empty_line.record_relations, vec![-1, -1]);
    }
    #[test]
    fn full_line_rejects_global_id_revision_and_deadline_float_features() {
        let config = PalsModelConfig::full_line_interaction_v2();
        for field in 6..8 {
            let mut input = v2_input();
            input.query[field] = 1.;
            assert_eq!(
                input.validate(&config),
                Err(PalsModelError::NonSemanticMetadata)
            );
        }
        let mut input = v2_input();
        input.records[0].features[6] = 1.;
        assert_eq!(
            input.validate(&config),
            Err(PalsModelError::NonSemanticMetadata)
        );
        input.full_line = None;
        input.validate(&PalsModelConfig::baseline()).unwrap();
    }
    #[test]
    fn candidate_order_and_model_epoch_are_input_identity() {
        let config = PalsModelConfig::baseline();
        let a = input();
        let mut b = a.clone();
        b.candidates.reverse();
        assert_ne!(
            a.canonical_input_key(&config).unwrap(),
            b.canonical_input_key(&config).unwrap()
        );
        b = a.clone();
        b.model_epoch[0] += 1;
        assert_ne!(
            a.canonical_input_key(&config).unwrap(),
            b.canonical_input_key(&config).unwrap()
        );
    }
    #[test]
    fn independent_projection_keeps_full_identity_and_bit_exact_feature_namespace() {
        let config = PalsModelConfig::baseline();
        let original = input();
        let mut changed = original.clone();
        changed.records[0].record_id = 40;
        changed.records[0].revision += 1;
        changed.records[0].critical = false;
        changed.required_critical_records.clear();
        changed.history_digest[0] += 1;
        changed.situation_revision += 1;
        assert_ne!(
            original.canonical_input_key(&config).unwrap(),
            changed.canonical_input_key(&config).unwrap()
        );
        assert_ne!(
            original.public_memory_key(&config).unwrap(),
            changed.public_memory_key(&config).unwrap()
        );
        assert_eq!(
            original.independent_public_plan(&config).unwrap(),
            changed.independent_public_plan(&config).unwrap()
        );
        let mut negative_zero = original.clone();
        negative_zero.records[0].features[0] = -0.;
        assert_ne!(
            original
                .independent_public_plan(&config)
                .unwrap()
                .record_contents,
            negative_zero
                .independent_public_plan(&config)
                .unwrap()
                .record_contents
        );
        changed = original.clone();
        changed.metadata[2] = 1.;
        assert_ne!(
            original
                .independent_public_plan(&config)
                .unwrap()
                .board_content,
            changed
                .independent_public_plan(&config)
                .unwrap()
                .board_content
        );
        changed = original.clone();
        changed.model_epoch[0] += 1;
        assert_ne!(
            original
                .independent_public_plan(&config)
                .unwrap()
                .record_contents,
            changed
                .independent_public_plan(&config)
                .unwrap()
                .record_contents
        );
    }
    #[test]
    fn record_eviction_id_shift_and_reorder_only_change_the_full_view() {
        let config = PalsModelConfig::baseline();
        let mut original = input();
        original.required_critical_records.clear();
        original.records = (0..128)
            .map(|index| PalsRecordToken {
                record_id: index + 1,
                revision: index + 10,
                critical: false,
                features: [index as f32; 16],
            })
            .collect();
        let old = original.independent_public_plan(&config).unwrap();
        let mut next = original.clone();
        next.records.remove(0);
        for (index, record) in next.records.iter_mut().enumerate() {
            record.record_id = index as u64 + 1;
        }
        next.records.push(PalsRecordToken {
            record_id: 128,
            revision: 200,
            critical: false,
            features: [200.; 16],
        });
        let fresh = next.independent_public_plan(&config).unwrap();
        assert_eq!(&old.record_contents[1..], &fresh.record_contents[..127]);
        assert_ne!(
            original.canonical_input_key(&config).unwrap(),
            next.canonical_input_key(&config).unwrap()
        );
        next.records.reverse();
        assert_eq!(
            fresh.record_contents.into_iter().rev().collect::<Vec<_>>(),
            next.independent_public_plan(&config)
                .unwrap()
                .record_contents
        );
    }
    #[test]
    fn empty_record_view_keeps_zero_feature_projection_with_false_mask() {
        let config = PalsModelConfig::baseline();
        let present = input();
        let mut empty = present.clone();
        empty.records.clear();
        empty.required_critical_records.clear();
        let padding = empty.independent_public_plan(&config).unwrap();
        assert_eq!(padding.features, vec![[0.; 16]]);
        assert_eq!(padding.record_mask, vec![false]);
        assert_eq!(
            padding.record_contents,
            present
                .independent_public_plan(&config)
                .unwrap()
                .record_contents
        );
        assert_ne!(
            empty.canonical_input_key(&config).unwrap(),
            present.canonical_input_key(&config).unwrap()
        );
    }
    #[test]
    fn exact_history_separates_inputs_and_public_memory_is_role_neutral() {
        let c = PalsModelConfig::baseline();
        let a = input();
        let mut b = a.clone();
        b.role = PalsRole::Critic;
        b.divergence_features = vec![[0.; DIVERGENCE_FEATURES]];
        b.query[0] = 1.;
        b.candidates.reverse();
        assert_eq!(
            a.public_memory_key(&c).unwrap(),
            b.public_memory_key(&c).unwrap()
        );
        assert_ne!(
            a.canonical_input_key(&c).unwrap(),
            b.canonical_input_key(&c).unwrap()
        );
        b = a.clone();
        b.history_digest[0] += 1;
        assert_ne!(
            a.public_memory_key(&c).unwrap(),
            b.public_memory_key(&c).unwrap()
        );
    }
    #[test]
    fn critical_missing_overflow_and_nonfinite_are_errors() {
        let c = PalsModelConfig::baseline();
        let mut a = input();
        a.records.clear();
        assert_eq!(a.validate(&c), Err(PalsModelError::MissingCriticalRecord));
        a = input();
        a.query[0] = f32::NAN;
        assert!(matches!(a.validate(&c), Err(PalsModelError::NonFinite(_))));
        a = input();
        a.records = vec![a.records[0].clone(); 129];
        assert_eq!(a.validate(&c), Err(PalsModelError::Capacity("records")));
    }
    #[test]
    fn all_promotions_roundtrip_and_reserved_bits_reject() {
        for promotion in 1..=4 {
            let c = PalsCandidateToken {
                from: 48,
                to: 56,
                promotion,
            };
            assert_eq!(
                PalsCandidateToken::from_packed(c.packed().unwrap()).unwrap(),
                c
            );
        }
        assert_eq!(
            PalsCandidateToken::from_packed(0x8000),
            Err(PalsModelError::InvalidCandidate)
        );
    }
    #[test]
    fn decode_keeps_order_and_private_roles() {
        let c = PalsModelConfig::baseline();
        let a = input();
        let mut o = PalsRawOutput {
            candidate_logits: vec![1., -1.],
            wdl_logits: [2., 0., -2.],
            divergence_logits: None,
            task_logits: None,
            private_latent: vec![0.; c.latent_elements()],
        };
        let d = o.decode(&a, &c).unwrap();
        assert!(d.candidate_policy[0] > d.candidate_policy[1]);
        assert!((d.wdl.iter().sum::<f32>() - 1.).abs() < 1e-6);
        o.task_logits = Some([0.; V_TASKS]);
        assert_eq!(o.decode(&a, &c), Err(PalsModelError::PrivateRoleOutput));
    }
    #[test]
    fn flops_charge_recurrence_and_record_encoding() {
        let c = PalsModelConfig::baseline();
        let a = c.matmul_flops(PalsRole::Proposer, 0, 20, 0).unwrap();
        let b = c.matmul_flops(PalsRole::Critic, 128, 20, 16).unwrap();
        assert!(b.public_encode > a.public_encode);
        assert!(b.role_forward > a.role_forward);
    }
}
