//! PALS neural model boundary. These are model tensors, not chess authority.
//!
//! The role scheduler supplies Rules-validated candidates. This module validates
//! bounded, finite model inputs and separates private role heads from public memory.
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

pub const PALS_MODEL_SCHEMA: &str = "rovezero.pals-model.v1";
pub const PALS_ENCODING_SCHEMA: &str = "rovezero.pals-board-records.v1";
pub const METADATA_FEATURES: usize = 16;
pub const RECORD_FEATURES: usize = 16;
/// Only the registered encoder's independent record projection, never the
/// full observation/task identity or an assertion of numerical equivalence.
pub const INDEPENDENT_RECORD_PROJECTION_SEMANTICS: &str =
    "rz-pals-independent-record-fp32/1;16-feature-bits;four-fields;no-record-position;no-cross-record-context";
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsRole {
    Proposer,
    Critic,
    Validator,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PalsModelConfig {
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
    /// v1 is intentionally one registered configuration; no silent OOM resizing.
    pub fn validate(&self) -> Result<(), PalsModelError> {
        if self != &Self::baseline() {
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
        board.update(b"rz-pals-contextual-board-fp32/1");
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
            .map(|values| {
                let mut hash = Sha256::new();
                hash.update(INDEPENDENT_RECORD_PROJECTION_SEMANTICS.as_bytes());
                hash.update(self.model_epoch);
                for value in values {
                    hash.update(value.to_bits().to_le_bytes());
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
        hash.update(PALS_MODEL_SCHEMA.as_bytes());
        hash.update(PALS_ENCODING_SCHEMA.as_bytes());
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
        Ok(hash.finalize().into())
    }

    /// The public encoder is role-neutral and does not consume candidates,
    /// query, divergence or private latent. Sharing this key never authorizes
    /// sharing private outputs or reusing a different role's latent.
    pub fn public_memory_key(&self, config: &PalsModelConfig) -> Result<[u8; 32], PalsModelError> {
        self.validate(config)?;
        let mut hash = Sha256::new();
        hash.update(PALS_MODEL_SCHEMA.as_bytes());
        hash.update(PALS_ENCODING_SCHEMA.as_bytes());
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
        let public_encode = 2 * 16 * 2 * w
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
        }
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
