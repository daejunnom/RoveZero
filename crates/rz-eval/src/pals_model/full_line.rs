//! Ordered move sequences and local relationship admission for model V2.
//! These tokens describe supplied evidence; Rules remains the move authority.
use super::{PalsCandidateToken, PalsModelError, MAX_LINE_PLY, QUERY_LINE_COUNT};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PalsRecordLine {
    pub moves: Vec<PalsCandidateToken>,
    /// Index in the currently selected record list, never a global record ID.
    pub parent: Option<usize>,
    pub supersedes: Option<usize>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub parent_required: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub supersedes_required: bool,
}
fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PalsFullLineInput {
    /// Exactly aligned with PalsModelInput.records, including its order.
    pub records: Vec<PalsRecordLine>,
    pub query_prefix: Vec<PalsCandidateToken>,
    pub query_proposal: Vec<PalsCandidateToken>,
    pub query_counter: Vec<PalsCandidateToken>,
}
impl PalsFullLineInput {
    pub fn validate(&self, records: usize) -> Result<(), PalsModelError> {
        if self.records.len() != records {
            return Err(PalsModelError::Shape("full_line records"));
        }
        if records > 128 {
            return Err(PalsModelError::Capacity("full_line records"));
        }
        for (slot, record) in self.records.iter().enumerate() {
            validate_moves(&record.moves)?;
            if (record.parent_required && record.parent.is_none())
                || (record.supersedes_required && record.supersedes.is_none())
            {
                return Err(PalsModelError::MissingRequiredRelationship);
            }
            if [record.parent, record.supersedes]
                .into_iter()
                .flatten()
                .any(|target| target >= records || target == slot)
            {
                return Err(PalsModelError::InvalidRelationship);
            }
        }
        for line in [
            &self.query_prefix,
            &self.query_proposal,
            &self.query_counter,
        ] {
            validate_moves(line)?;
        }
        // At most 128 nodes: bounded depth and storage, and both parent and
        // supersedes edges participate. A cyclic evidence graph is rejected.
        fn visit(
            slot: usize,
            records: &[PalsRecordLine],
            marks: &mut [u8; 128],
        ) -> Result<(), PalsModelError> {
            if marks[slot] == 1 {
                return Err(PalsModelError::RelationshipCycle);
            }
            if marks[slot] == 2 {
                return Ok(());
            }
            marks[slot] = 1;
            for target in [records[slot].parent, records[slot].supersedes]
                .into_iter()
                .flatten()
            {
                visit(target, records, marks)?;
            }
            marks[slot] = 2;
            Ok(())
        }
        let mut marks = [0; 128];
        for slot in 0..records {
            visit(slot, &self.records, &mut marks)?;
        }
        Ok(())
    }
    pub(super) fn hash_records(&self, hash: &mut Sha256) {
        hash.update((self.records.len() as u64).to_le_bytes());
        for record in &self.records {
            hash_moves(hash, &record.moves);
            for target in [record.parent, record.supersedes] {
                hash.update(target.map_or(u64::MAX, |value| value as u64).to_le_bytes());
            }
            hash.update([
                u8::from(record.parent_required),
                u8::from(record.supersedes_required),
            ]);
        }
    }
    pub(super) fn hash_query(&self, hash: &mut Sha256) {
        for line in [
            &self.query_prefix,
            &self.query_proposal,
            &self.query_counter,
        ] {
            hash_moves(hash, line);
        }
    }
    /// Called after validation. Fixed maximum lengths eliminate shape-dependent
    /// temporal padding effects; inactive tokens are exactly zero and false.
    pub(super) fn prepare(&self) -> PreparedPalsFullLineTensors {
        let records = self.records.len().max(1);
        let mut record_line_tokens = vec![0; records * MAX_LINE_PLY * 3];
        let mut record_line_mask = vec![false; records * MAX_LINE_PLY];
        let mut record_relations = vec![-1; records * 2];
        for (slot, record) in self.records.iter().enumerate() {
            write_moves(
                &record.moves,
                slot,
                &mut record_line_tokens,
                &mut record_line_mask,
            );
            for (kind, target) in [record.parent, record.supersedes].into_iter().enumerate() {
                record_relations[slot * 2 + kind] = target.map_or(-1, |value| value as i64);
            }
        }
        let mut query_line_tokens = vec![0; QUERY_LINE_COUNT * MAX_LINE_PLY * 3];
        let mut query_line_mask = vec![false; QUERY_LINE_COUNT * MAX_LINE_PLY];
        for (slot, line) in [
            &self.query_prefix,
            &self.query_proposal,
            &self.query_counter,
        ]
        .into_iter()
        .enumerate()
        {
            write_moves(line, slot, &mut query_line_tokens, &mut query_line_mask);
        }
        PreparedPalsFullLineTensors {
            record_line_tokens,
            record_line_mask,
            query_line_tokens,
            query_line_mask,
            record_relations,
        }
    }
    /// Complete owned input capacities for physical lease admission. Does not
    /// estimate native allocator or device workspace usage.
    pub fn owned_bytes(&self) -> usize {
        self.records.capacity() * std::mem::size_of::<PalsRecordLine>()
            + self
                .records
                .iter()
                .map(|r| r.moves.capacity() * std::mem::size_of::<PalsCandidateToken>())
                .sum::<usize>()
            + [
                self.query_prefix.capacity(),
                self.query_proposal.capacity(),
                self.query_counter.capacity(),
            ]
            .into_iter()
            .sum::<usize>()
                * std::mem::size_of::<PalsCandidateToken>()
    }
}
fn validate_moves(moves: &[PalsCandidateToken]) -> Result<(), PalsModelError> {
    if moves.len() > MAX_LINE_PLY {
        return Err(PalsModelError::Capacity("line ply"));
    }
    for movement in moves {
        movement.validate()?;
    }
    Ok(())
}
pub(super) fn hash_moves(hash: &mut Sha256, moves: &[PalsCandidateToken]) {
    hash.update((moves.len() as u64).to_le_bytes());
    for movement in moves {
        // Validation occurs at the model boundary. Three typed bytes retain
        // order, every middle move and all four promotion choices.
        hash.update([movement.from, movement.to, movement.promotion]);
    }
}
pub(super) fn write_moves(
    moves: &[PalsCandidateToken],
    slot: usize,
    tokens: &mut [i64],
    mask: &mut [bool],
) {
    for (ply, movement) in moves.iter().enumerate() {
        let offset = (slot * MAX_LINE_PLY + ply) * 3;
        tokens[offset..offset + 3].copy_from_slice(&[
            movement.from as i64,
            movement.to as i64,
            movement.promotion as i64,
        ]);
        mask[slot * MAX_LINE_PLY + ply] = true;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedPalsFullLineTensors {
    pub record_line_tokens: Vec<i64>,
    pub record_line_mask: Vec<bool>,
    pub query_line_tokens: Vec<i64>,
    pub query_line_mask: Vec<bool>,
    pub record_relations: Vec<i64>,
}
impl PreparedPalsFullLineTensors {
    pub fn owned_bytes(&self) -> usize {
        (self.record_line_tokens.capacity()
            + self.query_line_tokens.capacity()
            + self.record_relations.capacity())
            * 8
            + self.record_line_mask.capacity()
            + self.query_line_mask.capacity()
    }
}
