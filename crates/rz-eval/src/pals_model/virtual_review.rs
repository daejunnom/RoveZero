//! V3 selected-review tensors. Rules replay and store pins belong to Search.
//! No raw record scan, accumulated real-game moves, ID/revision feature or NN
//! session exists here. One graph accepts all registered physical L buckets.
use super::{PalsCandidateToken, PalsRole};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

pub const MODEL_SCHEMA: &str = "rovezero.pals-model.v3";
pub const MODEL_SEMANTICS: &str = "rovezero.pals-model-semantics.v3";
pub const ENCODING: &str = "rovezero.pals-virtual-review.v1";
pub const INPUT_SCHEMA: &str = "rovezero.pals-virtual-review-input.v1";
pub const PROFILE: &str = "virtual_review_v1";
pub const LENGTH_BUCKETS: [usize; 6] = [8, 16, 32, 64, 128, 256];
pub const MAX_RECORDS: usize = 64;
pub const MAX_ANCHORS: usize = 4;
/// V3 tensor/projection registry. This is independent from model parameter
/// semantics, the 56-field Rules descriptor and source implementation pins.
/// Python emits the same UTF-8 fields with u64-LE length framing. Historical
/// encoding/cache/fixture hashes are untouched.
pub const ENCODING_REGISTRY_FIELDS: &[&str] = &[
    "rz-pals-virtual-review-encoding-registry/1",
    "rovezero.pals-virtual-review.v1",
    "batch=1;records=0..64;anchors=1..4;root_anchor=0;candidates=0..256;divergences=0..128",
    "L=min_bucket(max_record_and_query_line_len);buckets=8,16,32,64,128,256;one_dynamic_graph",
    "token_budget=128|256|512;charge_records_and_prefix_proposal_counter_including_duplicates",
    "anchor_board:i64[B,A,64];piece_codes=0..12;a1_to_h8",
    "anchor_metadata:f32[B,A,16];Rules_metadata_meaning_is_separately_pinned",
    "anchor_mask:bool[B,A];all_typed_anchors_active;root0_required",
    "records:f32[B,max(R,1),8]",
    "record_features=kind.proposal,kind.counterexample,kind.repair,kind.cpu_verification,perspective,completed_iteration,rules_terminal,line_nonempty",
    "record_anchor:i64[B,max(R,1)];selected_anchor_local_index",
    "record_line_tokens:i64[B,max(R,1),L,3];from,to,promotion",
    "record_line_mask:bool[B,max(R,1),L];true_for_exact_line_prefix_only",
    "record_mask:bool[B,max(R,1)];true_for_actual_records_only",
    "record_relations:i64[B,max(R,1),2];parent,supersedes;absent=-1;local_DAG;no_self_edges",
    "query:f32[B,8]",
    "query_features=purpose.propose_policy,purpose.reply_policy,purpose.repair_policy,purpose.divergence_policy,purpose.value_fresh,purpose.task_policy,perspective,counter_present",
    "query_anchor:i64[B,3];prefix,proposal,counter_local_anchor_including_empty_sequences",
    "query_line_tokens:i64[B,3,L,3];prefix,proposal,counter;from,to,promotion",
    "query_line_mask:bool[B,3,L];true_for_exact_sequence_prefix_only",
    "candidates:i64[B,max(C,1),3];caller_legal_order;no_duplicates",
    "candidate_mask:bool[B,max(C,1)];true_for_exact_candidates_only",
    "divergence_features:f32[B,max(D,1),8];critic_only;raw_observation_vocabulary_separately_pinned",
    "divergence_mask:bool[B,max(D,1)];true_for_actual_critic_divergences_only",
    "moves=from0..63,to0..63,from!=to;promotion=0_none,1_queen,2_rook,3_bishop,4_knight",
    "perspective=+1_white|-1_black;counter_present=explicit_bool;true_empty_counter_is_present",
    "padding=zero_features_zero_moves_false_masks;record_relations=-1;masked_inputs_cannot_contribute",
    "FP32=finite_little_endian_preserve_signed_zero;I64=little_endian;BOOL=0|1",
    "role=proposer|critic|validator;private_PC_scalar_BOOL_routes_are_separate_from_request_tensors",
    "policy_projection=trim_to_exact_candidate_count_then_stable_softmax_in_caller_order;C0_returns_empty_policy",
    "wdl_projection=stable_softmax_of_win_draw_loss_logits;exact_query_position_side_to_move",
    "critic_projection=trim_to_exact_divergence_count;validator_task_logits=7;private_latent=16x384",
    "no_CP_depth_hash_stateID_revision_numeric_features;Rules_full_history_is_execution_seal_only",
];
pub fn registered_encoding_sha256() -> [u8; 32] {
    let mut hash = Sha256::new();
    for field in ENCODING_REGISTRY_FIELDS {
        hash.update((field.len() as u64).to_le_bytes());
        hash.update(field.as_bytes());
    }
    hash.finalize().into()
}
pub const RECORD_FEATURE_NAMES: [&str; 8] = [
    "kind.proposal",
    "kind.counterexample",
    "kind.repair",
    "kind.cpu_verification",
    "perspective",
    "completed_iteration",
    "rules_terminal",
    "line_nonempty",
];
pub const QUERY_FEATURE_NAMES: [&str; 8] = [
    "purpose.propose_policy",
    "purpose.reply_policy",
    "purpose.repair_policy",
    "purpose.divergence_policy",
    "purpose.value_fresh",
    "purpose.task_policy",
    "perspective",
    "counter_present",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewRecordKind {
    Proposal,
    Counterexample,
    Repair,
    CpuVerification,
}
impl ReviewRecordKind {
    fn index(self) -> usize {
        match self {
            Self::Proposal => 0,
            Self::Counterexample => 1,
            Self::Repair => 2,
            Self::CpuVerification => 3,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewQueryPurpose {
    ProposePolicy,
    ReplyPolicy,
    RepairPolicy,
    DivergencePolicy,
    ValueFresh,
    TaskPolicy,
}
impl ReviewQueryPurpose {
    fn index(self) -> usize {
        match self {
            Self::ProposePolicy => 0,
            Self::ReplyPolicy => 1,
            Self::RepairPolicy => 2,
            Self::DivergencePolicy => 3,
            Self::ValueFresh => 4,
            Self::TaskPolicy => 5,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewAnchorInput {
    /// Existing a1..h8 piece codes. Anchor zero is the current exact root.
    pub board: Vec<u8>,
    /// Existing Rules metadata; exact history remains an execution seal.
    pub metadata: [f32; 16],
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewRecordInput {
    pub kind: ReviewRecordKind,
    /// +1 white, -1 black. CP/depth/IDs are deliberately absent.
    pub perspective: i8,
    pub completed_iteration: bool,
    pub rules_terminal: bool,
    pub line: Vec<PalsCandidateToken>,
    pub anchor: u8,
    pub parent: Option<u16>,
    pub supersedes: Option<u16>,
}
impl ReviewRecordInput {
    pub fn features(&self) -> [f32; 8] {
        let mut result = [0.; 8];
        result[self.kind.index()] = 1.;
        result[4] = self.perspective as f32;
        result[5] = u8::from(self.completed_iteration) as f32;
        result[6] = u8::from(self.rules_terminal) as f32;
        result[7] = u8::from(!self.line.is_empty()) as f32;
        result
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewQueryInput {
    pub purpose: ReviewQueryPurpose,
    pub perspective: i8,
    pub prefix: Vec<PalsCandidateToken>,
    pub proposal: Vec<PalsCandidateToken>,
    pub counter: Vec<PalsCandidateToken>,
    /// Presence is distinct from line length: Some(empty terminal witness)
    /// must not be silently changed to an absent counterexample.
    pub counter_present: bool,
    /// One local anchor per sequence, including semantically empty sequences.
    pub anchors: [u8; 3],
}
impl ReviewQueryInput {
    pub fn features(&self) -> [f32; 8] {
        let mut result = [0.; 8];
        result[self.purpose.index()] = 1.;
        result[6] = self.perspective as f32;
        result[7] = u8::from(self.counter_present) as f32;
        result
    }
    fn lines(&self) -> [&[PalsCandidateToken]; 3] {
        [&self.prefix, &self.proposal, &self.counter]
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PalsVirtualReviewInput {
    pub schema: String,
    pub role: PalsRole,
    pub anchors: Vec<ReviewAnchorInput>,
    pub records: Vec<ReviewRecordInput>,
    pub query: ReviewQueryInput,
    pub candidates: Vec<PalsCandidateToken>,
    pub divergences: Vec<[f32; 8]>,
    pub token_budget: u16,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualReviewAdmission {
    pub token_count: usize,
    pub length_bucket: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReviewFreshKey(pub [u8; 32]);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReviewWarmContextKey(pub [u8; 32]);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReviewApproxWarmKey(pub [u8; 32]);
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewExecutionIdentity {
    pub model_epoch: [u8; 32],
    pub rules_history: [u8; 32],
    pub backend: [u8; 32],
    /// Complete exact Rules/history seal of every selected anchor, in local
    /// anchor order. Anchor zero must equal the independent current-root seal.
    pub anchor_history_seals: Vec<[u8; 32]>,
    /// Model V3 currently supports FP32. Unknown/automatic precision is rejected.
    pub fp32: bool,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VirtualReviewError {
    Schema,
    Capacity(&'static str),
    Shape(&'static str),
    InvalidMove,
    NonFinite(&'static str),
    Perspective,
    Anchor,
    Relationship,
    Cycle,
    DuplicateCandidate,
    PrivateRole,
    ViewCapacity,
    UnsupportedPrecision,
    FreshRequired,
}
impl std::fmt::Display for VirtualReviewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PALS virtual review boundary: {self:?}")
    }
}
impl std::error::Error for VirtualReviewError {}
pub fn length_bucket(length: usize) -> Result<usize, VirtualReviewError> {
    LENGTH_BUCKETS
        .into_iter()
        .find(|bucket| length <= *bucket)
        .ok_or(VirtualReviewError::Capacity("line ply"))
}
fn validate_moves(moves: &[PalsCandidateToken]) -> Result<(), VirtualReviewError> {
    if moves.len() > 256 {
        return Err(VirtualReviewError::Capacity("line ply"));
    }
    if moves.iter().any(|movement| movement.validate().is_err()) {
        return Err(VirtualReviewError::InvalidMove);
    }
    Ok(())
}
impl PalsVirtualReviewInput {
    fn execution_prefix(
        &self,
        identity: &ReviewExecutionIdentity,
        domain: &'static [u8],
    ) -> Result<Sha256, VirtualReviewError> {
        self.validate()?;
        if !identity.fp32 {
            return Err(VirtualReviewError::UnsupportedPrecision);
        }
        if identity.anchor_history_seals.len() != self.anchors.len()
            || identity.anchor_history_seals.first() != Some(&identity.rules_history)
        {
            return Err(VirtualReviewError::Anchor);
        }
        let mut hash = Sha256::new();
        hash.update(domain);
        hash.update(identity.model_epoch);
        hash.update(identity.rules_history);
        hash.update(identity.backend);
        hash.update(b"fp32");
        hash.update((identity.anchor_history_seals.len() as u64).to_le_bytes());
        for seal in &identity.anchor_history_seals {
            hash.update(seal);
        }
        hash.update([match self.role {
            PalsRole::Proposer => 0,
            PalsRole::Critic => 1,
            PalsRole::Validator => 2,
        }]);
        Ok(hash)
    }
    pub fn fresh_key(
        &self,
        identity: &ReviewExecutionIdentity,
    ) -> Result<ReviewFreshKey, VirtualReviewError> {
        let mut hash = self.execution_prefix(identity, b"rovezero.pals-review-fresh.v1\0")?;
        hash.update(self.prepare()?.input_digest(self.role));
        Ok(ReviewFreshKey(hash.finalize().into()))
    }
    /// Public record revision is the only omitted model context. Caller must
    /// still prove accepted same-owner seed authority and physical completion.
    pub fn warm_context_key(
        &self,
        identity: &ReviewExecutionIdentity,
    ) -> Result<ReviewWarmContextKey, VirtualReviewError> {
        if self.role == PalsRole::Validator || self.query.purpose == ReviewQueryPurpose::ValueFresh
        {
            return Err(VirtualReviewError::FreshRequired);
        }
        let mut hash = self.execution_prefix(identity, b"rovezero.pals-review-context.v1\0")?;
        hash.update([
            self.query.purpose.index() as u8,
            self.query.perspective as u8,
            u8::from(self.query.counter_present),
        ]);
        for (line, index) in self.query.lines().into_iter().zip(self.query.anchors) {
            hash.update((line.len() as u64).to_le_bytes());
            for movement in line {
                hash.update([movement.from, movement.to, movement.promotion]);
            }
            let anchor = &self.anchors[index as usize];
            hash.update(&anchor.board);
            for field in anchor.metadata {
                hash.update(field.to_le_bytes());
            }
        }
        hash.update((self.candidates.len() as u64).to_le_bytes());
        for movement in &self.candidates {
            hash.update([movement.from, movement.to, movement.promotion]);
        }
        hash.update((self.divergences.len() as u64).to_le_bytes());
        for field in self.divergences.iter().flatten() {
            hash.update(field.to_le_bytes());
        }
        Ok(ReviewWarmContextKey(hash.finalize().into()))
    }
    pub fn approximate_warm_key(
        &self,
        identity: &ReviewExecutionIdentity,
        accepted_seed_sha256: [u8; 32],
    ) -> Result<ReviewApproxWarmKey, VirtualReviewError> {
        let mut hash = self.execution_prefix(identity, b"rovezero.pals-review-approx-warm.v1\0")?;
        hash.update(self.warm_context_key(identity)?.0);
        hash.update(self.prepare()?.input_digest(self.role));
        hash.update(accepted_seed_sha256);
        Ok(ReviewApproxWarmKey(hash.finalize().into()))
    }
    pub fn validate(&self) -> Result<VirtualReviewAdmission, VirtualReviewError> {
        if self.schema != INPUT_SCHEMA {
            return Err(VirtualReviewError::Schema);
        }
        if self.anchors.is_empty() || self.anchors.len() > MAX_ANCHORS {
            return Err(VirtualReviewError::Capacity("anchors"));
        }
        if self.records.len() > MAX_RECORDS {
            return Err(VirtualReviewError::Capacity("records"));
        }
        for anchor in &self.anchors {
            if anchor.board.len() != 64 || anchor.board.iter().any(|piece| *piece > 12) {
                return Err(VirtualReviewError::Shape("anchor board"));
            }
            if anchor.metadata.iter().any(|value| !value.is_finite()) {
                return Err(VirtualReviewError::NonFinite("anchor metadata"));
            }
        }
        let mut token_count = 0;
        let mut maximum = 0;
        for (slot, record) in self.records.iter().enumerate() {
            if !matches!(record.perspective, -1 | 1) {
                return Err(VirtualReviewError::Perspective);
            }
            if record.anchor as usize >= self.anchors.len() {
                return Err(VirtualReviewError::Anchor);
            }
            validate_moves(&record.line)?;
            token_count += record.line.len();
            maximum = maximum.max(record.line.len());
            for target in [record.parent, record.supersedes].into_iter().flatten() {
                if target as usize >= self.records.len() || target as usize == slot {
                    return Err(VirtualReviewError::Relationship);
                }
            }
        }
        fn visit(
            slot: usize,
            records: &[ReviewRecordInput],
            marks: &mut [u8; MAX_RECORDS],
        ) -> Result<(), VirtualReviewError> {
            if marks[slot] == 1 {
                return Err(VirtualReviewError::Cycle);
            }
            if marks[slot] == 2 {
                return Ok(());
            }
            marks[slot] = 1;
            for target in [records[slot].parent, records[slot].supersedes]
                .into_iter()
                .flatten()
            {
                visit(target as usize, records, marks)?;
            }
            marks[slot] = 2;
            Ok(())
        }
        let mut marks = [0; MAX_RECORDS];
        for slot in 0..self.records.len() {
            visit(slot, &self.records, &mut marks)?;
        }
        if !matches!(self.query.perspective, -1 | 1) {
            return Err(VirtualReviewError::Perspective);
        }
        if !self.query.counter_present && !self.query.counter.is_empty() {
            return Err(VirtualReviewError::Shape(
                "absent counterexample carries moves",
            ));
        }
        if self
            .query
            .anchors
            .iter()
            .any(|index| *index as usize >= self.anchors.len())
        {
            return Err(VirtualReviewError::Anchor);
        }
        for line in self.query.lines() {
            validate_moves(line)?;
            // A query duplicate is a physical input and consumes tokens again.
            token_count += line.len();
            maximum = maximum.max(line.len());
        }
        if !matches!(self.token_budget, 128 | 256 | 512) || token_count > self.token_budget as usize
        {
            return Err(VirtualReviewError::ViewCapacity);
        }
        validate_moves(&self.candidates)?;
        let mut candidates = std::collections::BTreeSet::new();
        for movement in &self.candidates {
            if !candidates.insert(*movement) {
                return Err(VirtualReviewError::DuplicateCandidate);
            }
        }
        if self.divergences.len() > 128 {
            return Err(VirtualReviewError::Capacity("divergences"));
        }
        if self
            .divergences
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(VirtualReviewError::NonFinite("divergences"));
        }
        if self.role != PalsRole::Critic && !self.divergences.is_empty() {
            return Err(VirtualReviewError::PrivateRole);
        }
        Ok(VirtualReviewAdmission {
            token_count,
            length_bucket: length_bucket(maximum)?,
        })
    }
    pub fn prepare(&self) -> Result<PreparedVirtualReviewTensors, VirtualReviewError> {
        let admission = self.validate()?;
        let r = self.records.len().max(1);
        let c = self.candidates.len().max(1);
        let d = self.divergences.len().max(1);
        let l = admission.length_bucket;
        let mut out = PreparedVirtualReviewTensors {
            anchors: self.anchors.len(),
            records_count: r,
            candidates_count: c,
            divergences_count: d,
            length_bucket: l,
            token_count: admission.token_count,
            anchor_board: self
                .anchors
                .iter()
                .flat_map(|anchor| anchor.board.iter().map(|value| *value as i64))
                .collect(),
            anchor_metadata: self
                .anchors
                .iter()
                .flat_map(|anchor| anchor.metadata)
                .collect(),
            anchor_mask: vec![true; self.anchors.len()],
            records: vec![0.; r * 8],
            record_anchor: vec![0; r],
            record_mask: vec![false; r],
            record_line_tokens: vec![0; r * l * 3],
            record_line_mask: vec![false; r * l],
            record_relations: vec![-1; r * 2],
            query: self.query.features(),
            query_anchor: self.query.anchors.map(i64::from),
            query_line_tokens: vec![0; 3 * l * 3],
            query_line_mask: vec![false; 3 * l],
            candidates: vec![0; c * 3],
            candidate_mask: vec![false; c],
            divergence_features: vec![0.; d * 8],
            divergence_mask: vec![false; d],
        };
        for (slot, record) in self.records.iter().enumerate() {
            out.records[slot * 8..slot * 8 + 8].copy_from_slice(&record.features());
            out.record_anchor[slot] = record.anchor as i64;
            out.record_mask[slot] = true;
            write_moves(
                &record.line,
                slot,
                l,
                &mut out.record_line_tokens,
                &mut out.record_line_mask,
            );
            for (axis, target) in [record.parent, record.supersedes].into_iter().enumerate() {
                out.record_relations[slot * 2 + axis] = target.map_or(-1, i64::from);
            }
        }
        for (slot, line) in self.query.lines().into_iter().enumerate() {
            write_moves(
                line,
                slot,
                l,
                &mut out.query_line_tokens,
                &mut out.query_line_mask,
            );
        }
        for (slot, movement) in self.candidates.iter().enumerate() {
            out.candidates[slot * 3..slot * 3 + 3].copy_from_slice(&[
                movement.from as i64,
                movement.to as i64,
                movement.promotion as i64,
            ]);
            out.candidate_mask[slot] = true;
        }
        for (slot, features) in self.divergences.iter().enumerate() {
            out.divergence_features[slot * 8..slot * 8 + 8].copy_from_slice(features);
            out.divergence_mask[slot] = true;
        }
        Ok(out)
    }
    pub fn owned_bytes(&self) -> usize {
        self.anchors.capacity() * std::mem::size_of::<ReviewAnchorInput>()
            + self
                .anchors
                .iter()
                .map(|a| a.board.capacity())
                .sum::<usize>()
            + self.records.capacity() * std::mem::size_of::<ReviewRecordInput>()
            + self
                .records
                .iter()
                .map(|r| r.line.capacity())
                .sum::<usize>()
                * std::mem::size_of::<PalsCandidateToken>()
            + [
                self.query.prefix.capacity(),
                self.query.proposal.capacity(),
                self.query.counter.capacity(),
            ]
            .into_iter()
            .sum::<usize>()
                * std::mem::size_of::<PalsCandidateToken>()
            + self.candidates.capacity() * std::mem::size_of::<PalsCandidateToken>()
            + self.divergences.capacity() * 8 * 4
            + self.schema.capacity()
    }
}
fn write_moves(
    moves: &[PalsCandidateToken],
    slot: usize,
    length: usize,
    tokens: &mut [i64],
    mask: &mut [bool],
) {
    for (ply, movement) in moves.iter().enumerate() {
        let offset = (slot * length + ply) * 3;
        tokens[offset..offset + 3].copy_from_slice(&[
            movement.from as i64,
            movement.to as i64,
            movement.promotion as i64,
        ]);
        mask[slot * length + ply] = true;
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedVirtualReviewTensors {
    pub anchors: usize,
    pub records_count: usize,
    pub candidates_count: usize,
    pub divergences_count: usize,
    pub length_bucket: usize,
    pub token_count: usize,
    pub anchor_board: Vec<i64>,
    pub anchor_metadata: Vec<f32>,
    pub anchor_mask: Vec<bool>,
    pub records: Vec<f32>,
    pub record_anchor: Vec<i64>,
    pub record_mask: Vec<bool>,
    pub record_line_tokens: Vec<i64>,
    pub record_line_mask: Vec<bool>,
    pub record_relations: Vec<i64>,
    pub query: [f32; 8],
    pub query_anchor: [i64; 3],
    pub query_line_tokens: Vec<i64>,
    pub query_line_mask: Vec<bool>,
    pub candidates: Vec<i64>,
    pub candidate_mask: Vec<bool>,
    pub divergence_features: Vec<f32>,
    pub divergence_mask: Vec<bool>,
}
impl PreparedVirtualReviewTensors {
    /// Owned host capacity only; ORT prepack/device workspace are observations.
    pub fn owned_bytes(&self) -> usize {
        [
            self.anchor_board.capacity(),
            self.record_anchor.capacity(),
            self.record_line_tokens.capacity(),
            self.record_relations.capacity(),
            self.query_line_tokens.capacity(),
            self.candidates.capacity(),
        ]
        .into_iter()
        .sum::<usize>()
            * 8
            + [
                self.anchor_metadata.capacity(),
                self.records.capacity(),
                self.divergence_features.capacity(),
            ]
            .into_iter()
            .sum::<usize>()
                * 4
            + [
                self.anchor_mask.capacity(),
                self.record_mask.capacity(),
                self.record_line_mask.capacity(),
                self.query_line_mask.capacity(),
                self.candidate_mask.capacity(),
                self.divergence_mask.capacity(),
            ]
            .into_iter()
            .sum::<usize>()
            + (8 * 4)
            + (3 * 8)
    }
    /// Canonical actual typed tensor bytes, shared with Python's input_digest.
    pub fn input_digest(&self, role: PalsRole) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(ENCODING.as_bytes());
        hash.update([0]);
        let role = match role {
            PalsRole::Proposer => "proposer",
            PalsRole::Critic => "critic",
            PalsRole::Validator => "validator",
        };
        frame(&mut hash, role.as_bytes());
        let (a, r, c, d, l) = (
            self.anchors,
            self.records_count,
            self.candidates_count,
            self.divergences_count,
            self.length_bucket,
        );
        // Lexical tensor-name order is part of the V3 digest ABI.
        hash_i64(&mut hash, "anchor_board", &[1, a, 64], &self.anchor_board);
        hash_bool(&mut hash, "anchor_mask", &[1, a], &self.anchor_mask);
        hash_f32(
            &mut hash,
            "anchor_metadata",
            &[1, a, 16],
            &self.anchor_metadata,
        );
        hash_bool(&mut hash, "candidate_mask", &[1, c], &self.candidate_mask);
        hash_i64(&mut hash, "candidates", &[1, c, 3], &self.candidates);
        hash_f32(
            &mut hash,
            "divergence_features",
            &[1, d, 8],
            &self.divergence_features,
        );
        hash_bool(&mut hash, "divergence_mask", &[1, d], &self.divergence_mask);
        hash_f32(&mut hash, "query", &[1, 8], &self.query);
        hash_i64(&mut hash, "query_anchor", &[1, 3], &self.query_anchor);
        hash_bool(
            &mut hash,
            "query_line_mask",
            &[1, 3, l],
            &self.query_line_mask,
        );
        hash_i64(
            &mut hash,
            "query_line_tokens",
            &[1, 3, l, 3],
            &self.query_line_tokens,
        );
        hash_i64(&mut hash, "record_anchor", &[1, r], &self.record_anchor);
        hash_bool(
            &mut hash,
            "record_line_mask",
            &[1, r, l],
            &self.record_line_mask,
        );
        hash_i64(
            &mut hash,
            "record_line_tokens",
            &[1, r, l, 3],
            &self.record_line_tokens,
        );
        hash_bool(&mut hash, "record_mask", &[1, r], &self.record_mask);
        hash_i64(
            &mut hash,
            "record_relations",
            &[1, r, 2],
            &self.record_relations,
        );
        hash_f32(&mut hash, "records", &[1, r, 8], &self.records);
        hash.finalize().into()
    }
}
fn frame(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
fn tensor_header(hash: &mut Sha256, name: &str, dtype: &str, shape: &[usize]) {
    frame(hash, name.as_bytes());
    frame(hash, dtype.as_bytes());
    hash.update((shape.len() as u64).to_le_bytes());
    for dimension in shape {
        hash.update((*dimension as u64).to_le_bytes());
    }
}
fn hash_i64(hash: &mut Sha256, name: &str, shape: &[usize], values: &[i64]) {
    tensor_header(hash, name, "i64", shape);
    for value in values {
        hash.update(value.to_le_bytes());
    }
}
fn hash_f32(hash: &mut Sha256, name: &str, shape: &[usize], values: &[f32]) {
    tensor_header(hash, name, "f32", shape);
    for value in values {
        hash.update(value.to_le_bytes());
    }
}
fn hash_bool(hash: &mut Sha256, name: &str, shape: &[usize], values: &[bool]) {
    tensor_header(hash, name, "bool", shape);
    for value in values {
        hash.update([u8::from(*value)]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encoding_registry_has_independent_cross_language_pin() {
        let pin = registered_encoding_sha256()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            pin,
            "57f53319114c3da14612f5d4dc99ff1a05fa115c471a5ed764d02b7fce0f5d0f"
        );
        assert_eq!(ENCODING_REGISTRY_FIELDS.len(), 33);
        assert_ne!(
            registered_encoding_sha256(),
            super::super::PalsModelProfile::VirtualReviewV1.registered_semantics_sha256()
        );
    }
    fn movement() -> PalsCandidateToken {
        PalsCandidateToken {
            from: 12,
            to: 28,
            promotion: 0,
        }
    }
    fn input() -> PalsVirtualReviewInput {
        PalsVirtualReviewInput {
            schema: INPUT_SCHEMA.to_string(),
            role: PalsRole::Proposer,
            anchors: vec![ReviewAnchorInput {
                board: vec![0; 64],
                metadata: [0.; 16],
            }],
            records: vec![ReviewRecordInput {
                kind: ReviewRecordKind::Proposal,
                perspective: 1,
                completed_iteration: false,
                rules_terminal: false,
                line: vec![movement()],
                anchor: 0,
                parent: None,
                supersedes: None,
            }],
            query: ReviewQueryInput {
                purpose: ReviewQueryPurpose::ProposePolicy,
                perspective: 1,
                prefix: vec![],
                proposal: vec![movement()],
                counter: vec![],
                counter_present: false,
                anchors: [0; 3],
            },
            candidates: vec![movement()],
            divergences: vec![],
            token_budget: 512,
        }
    }
    #[test]
    fn buckets_and_duplicate_query_charge() {
        for (length, expected) in [
            (0, 8),
            (8, 8),
            (9, 16),
            (17, 32),
            (33, 64),
            (65, 128),
            (129, 256),
            (256, 256),
        ] {
            assert_eq!(length_bucket(length), Ok(expected));
        }
        assert!(length_bucket(257).is_err());
        let mut value = input();
        value.records[0].line = vec![movement(); 128];
        value.query.proposal = vec![movement(); 128];
        value.token_budget = 128;
        assert_eq!(value.validate(), Err(VirtualReviewError::ViewCapacity));
        value.token_budget = 256;
        assert_eq!(value.validate().unwrap().token_count, 256);
    }
    #[test]
    fn missing_anchor_cycle_private_and_candidate_boundary() {
        let mut value = input();
        value.query.anchors[2] = 1;
        assert_eq!(value.validate(), Err(VirtualReviewError::Anchor));
        value = input();
        let mut r = value.records[0].clone();
        r.parent = Some(0);
        value.records.push(r);
        value.records[0].parent = Some(1);
        assert_eq!(value.validate(), Err(VirtualReviewError::Cycle));
        value = input();
        value.divergences.push([0.; 8]);
        assert_eq!(value.validate(), Err(VirtualReviewError::PrivateRole));
        value = input();
        value.candidates.push(movement());
        assert_eq!(
            value.validate(),
            Err(VirtualReviewError::DuplicateCandidate)
        );
    }
    #[test]
    fn feature_axes_and_padding_and_order_are_semantic() {
        let value = input();
        let prepared = value.prepare().unwrap();
        assert_eq!(prepared.records_count, 1);
        assert_eq!(prepared.length_bucket, 8);
        assert_eq!(&prepared.records[..8], &[1., 0., 0., 0., 1., 0., 0., 1.]);
        assert_eq!(prepared.query, [1., 0., 0., 0., 0., 0., 1., 0.]);
        assert!(prepared.record_line_tokens[3..].iter().all(|v| *v == 0));
        let digest = prepared
            .input_digest(value.role)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest,
            "34f13fc24725fc4eb338044a502d5da37534a0eb9044194baf4c50a1534d5e57"
        );
        let mut other = value.clone();
        other.records[0].line.push(PalsCandidateToken {
            from: 52,
            to: 36,
            promotion: 0,
        });
        assert_ne!(
            prepared.input_digest(value.role),
            other.prepare().unwrap().input_digest(other.role)
        );
        other = value;
        other.records.clear();
        let padded = other.prepare().unwrap();
        assert_eq!(padded.record_mask, vec![false]);
        assert_eq!(padded.record_relations, vec![-1, -1]);
    }
}
