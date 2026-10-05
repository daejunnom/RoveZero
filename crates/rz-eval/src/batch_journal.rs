//! Bounded opt-in experiment evidence; never evaluation/cache authority.
//! Compact rows share physical execution IDs without per-node heap strings.
use rz_contracts::*;
use serde::{Deserialize, Serialize};
pub const MAX_BATCH_REQUESTS: usize = 600_000;
pub const MAX_BATCH_RECEIPT_BYTES: u64 = 64 * 1024 * 1024;
/// request, selection, game, root, execution, physical result (0 unknown,
/// 1 completed, 2 failed), cancellation requested, logical outcome (0 pending,
/// 1 delivered, 2 canceled, 3 expired, 4 stale, 5 failed, 6 drain-discarded), consumed.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BatchRequestRow(
    pub u64,
    pub u64,
    pub u64,
    pub u64,
    pub Option<u64>,
    pub u8,
    pub bool,
    pub u8,
    pub bool,
);
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchJournalReceipt {
    pub schema_version: u32,
    pub max_requests: usize,
    pub max_batch: usize,
    /// Physical lease handoffs, not proof of kernel launch.
    pub physical_dispatches: u64,
    pub physical_completed: u64,
    pub physical_failed: u64,
    pub dispatched_batch_distribution: [u64; 16],
    pub completed_batch_distribution: [u64; 16],
    #[serde(deserialize_with = "bounded_rows")]
    pub rows: Vec<BatchRequestRow>,
}
fn bounded_rows<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<BatchRequestRow>, D::Error> {
    struct Rows;
    impl<'de> serde::de::Visitor<'de> for Rows {
        type Value = Vec<BatchRequestRow>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("bounded request row array")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut input: A,
        ) -> Result<Self::Value, A::Error> {
            let mut rows = Vec::new();
            while let Some(row) = input.next_element()? {
                if rows.len() == MAX_BATCH_REQUESTS {
                    return Err(serde::de::Error::custom("request row budget exceeded"));
                }
                if rows.len() == rows.capacity() {
                    rows.try_reserve(1024.min(MAX_BATCH_REQUESTS - rows.len()))
                        .map_err(|_| serde::de::Error::custom("request row allocation failed"))?;
                }
                rows.push(row);
            }
            Ok(rows)
        }
    }
    deserializer.deserialize_seq(Rows)
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BatchJournalAudit {
    pub max_pending: usize,
    pub physical_dispatches: u64,
    pub physical_completed: u64,
    pub completed_nn_items: u64,
    pub delivered_evaluations: u64,
    pub consumed_evaluations: u64,
    pub canceled_requests: u64,
    pub unused_completed_items: u64,
    pub completed_batch_distribution: [u64; 16],
}
/// Recompute all summary fields from the raw finite rows. Unknown physical or
/// logical outcomes are excluded from successful acceptance, never guessed.
pub fn audit_batch_journal(
    receipt: &BatchJournalReceipt,
    require_complete: bool,
) -> Result<BatchJournalAudit, ContractError> {
    use std::collections::BTreeMap;
    if receipt.schema_version != 1
        || receipt.max_requests != MAX_BATCH_REQUESTS
        || !(1..=16).contains(&receipt.max_batch)
        || receipt.rows.len() > MAX_BATCH_REQUESTS
    {
        return Err(invalid());
    }
    let mut executions = BTreeMap::<u64, (u64, u64, usize, u8)>::new();
    let mut previous = 0;
    let mut consumed = 0;
    let mut delivered = 0;
    let mut canceled = 0;
    for row in &receipt.rows {
        if row.0 <= previous
            || row.0 as usize > MAX_BATCH_REQUESTS
            || row.1 == 0
            || row.2 == 0
            || row.3 == 0
            || row.5 > 2
            || row.7 > 6
            || (require_complete && row.7 == 0)
        {
            return Err(invalid());
        }
        previous = row.0;
        if row.8 && (row.5 != 1 || row.7 != 1 || row.4.is_none()) {
            return Err(invalid());
        }
        consumed += u64::from(row.8);
        delivered += u64::from(row.7 == 1);
        canceled += u64::from(row.6 || row.7 == 2);
        if let Some(execution) = row.4 {
            if execution == 0 || (require_complete && row.5 == 0) {
                return Err(invalid());
            }
            let item = executions
                .entry(execution)
                .or_insert((row.2, row.3, 0, row.5));
            if (item.0, item.1, item.3) != (row.2, row.3, row.5) {
                return Err(invalid());
            }
            item.2 += 1;
            if item.2 > receipt.max_batch {
                return Err(invalid());
            }
        } else if row.5 != 0 || row.7 == 1 || row.8 {
            return Err(invalid());
        }
    }
    let mut dispatched = [0; 16];
    let mut completed = [0; 16];
    let mut failed = 0;
    for (_, _, width, status) in executions.values() {
        dispatched[width - 1] += 1;
        if *status == 1 {
            completed[width - 1] += 1;
        } else if *status == 2 {
            failed += 1;
        }
    }
    let physical_completed = completed.iter().sum();
    let nn_items = completed
        .iter()
        .enumerate()
        .map(|(i, n)| n * (i as u64 + 1))
        .sum::<u64>();
    if dispatched != receipt.dispatched_batch_distribution
        || completed != receipt.completed_batch_distribution
        || executions.len() as u64 != receipt.physical_dispatches
        || physical_completed != receipt.physical_completed
        || failed != receipt.physical_failed
        || consumed > nn_items
    {
        return Err(invalid());
    }
    Ok(BatchJournalAudit {
        max_pending: receipt.max_batch,
        physical_dispatches: receipt.physical_dispatches,
        physical_completed,
        completed_nn_items: nn_items,
        delivered_evaluations: delivered,
        consumed_evaluations: consumed,
        canceled_requests: canceled,
        unused_completed_items: nn_items - consumed,
        completed_batch_distribution: completed,
    })
}
pub(crate) struct BatchJournal {
    width: usize,
    rows: Vec<Option<BatchRequestRow>>,
    dispatched: [u64; 16],
    completed: [u64; 16],
    failed: u64,
}

fn invalid() -> ContractError {
    ContractError::new(
        ErrorCode::IdentityMismatch,
        Stage::Output,
        "batch experiment journal identity/state mismatch",
    )
}
impl BatchJournal {
    pub(crate) fn new(width: usize) -> Result<Self, ContractError> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(MAX_BATCH_REQUESTS).map_err(|_| {
            ContractError::new(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "bounded batch receipt allocation failed",
            )
        })?;
        Ok(Self {
            width,
            rows,
            dispatched: [0; 16],
            completed: [0; 16],
            failed: 0,
        })
    }
    pub(crate) fn register(&mut self, context: EvalContext) -> Result<(), ContractError> {
        let index = usize::try_from(context.request.sequence)
            .map_err(|_| invalid())?
            .checked_sub(1)
            .filter(|&i| i < MAX_BATCH_REQUESTS)
            .ok_or_else(|| {
                ContractError::new(
                    ErrorCode::ResourceExhausted,
                    Stage::Admission,
                    "batch receipt request budget exhausted",
                )
            })?;
        if self.rows.len() <= index {
            self.rows.resize_with(index + 1, || None);
        }
        if self.rows[index].is_some() {
            return Err(invalid());
        }
        self.rows[index] = Some(BatchRequestRow(
            context.request.sequence,
            context.selection.sequence,
            context.game.0,
            context.root.0,
            None,
            0,
            false,
            0,
            false,
        ));
        Ok(())
    }
    pub(crate) fn row(
        &mut self,
        request: RequestId,
    ) -> Result<&mut BatchRequestRow, ContractError> {
        let index = usize::try_from(request.sequence)
            .map_err(|_| invalid())?
            .checked_sub(1)
            .ok_or_else(invalid)?;
        self.rows
            .get_mut(index)
            .and_then(Option::as_mut)
            .ok_or_else(invalid)
    }
    pub(crate) fn dispatch(
        &mut self,
        contexts: &[EvalContext],
        execution: ExecutionId,
    ) -> Result<(), ContractError> {
        if contexts.is_empty() || contexts.len() > self.width || contexts.len() > 16 {
            return Err(invalid());
        }
        let first = contexts[0];
        for &context in contexts {
            if context.game != first.game
                || context.root != first.root
                || context.model != first.model
                || context.encoding != first.encoding
                || context.backend != first.backend
            {
                return Err(invalid());
            }
            let row = self.row(context.request)?;
            if row.4.is_some() || row.2 != context.game.0 || row.3 != context.root.0 {
                return Err(invalid());
            }
        }
        for &context in contexts {
            self.row(context.request)?.4 = Some(execution.sequence);
        }
        self.dispatched[contexts.len() - 1] += 1;
        Ok(())
    }
    pub(crate) fn undo_dispatch(&mut self, contexts: &[EvalContext]) {
        for &context in contexts {
            if let Ok(row) = self.row(context.request) {
                row.4 = None;
            }
        }
        self.dispatched[contexts.len() - 1] -= 1;
    }
    pub(crate) fn physical(
        &mut self,
        contexts: impl Iterator<Item = CompletionContext>,
        succeeded: bool,
    ) -> Result<(), ContractError> {
        let mut width = 0;
        for context in contexts {
            let row = self.row(context.request.request)?;
            if row.4 != context.execution.map(|e| e.sequence) || row.5 != 0 {
                return Err(invalid());
            }
            row.5 = if succeeded { 1 } else { 2 };
            width += 1;
        }
        if width == 0 || width > self.width {
            return Err(invalid());
        }
        if succeeded {
            self.completed[width - 1] += 1;
        } else {
            self.failed += 1;
        }
        Ok(())
    }
    pub(crate) fn consume(&mut self, context: CompletionContext) -> Result<(), ContractError> {
        let row = self.row(context.request.request)?;
        if row.4 != context.execution.map(|e| e.sequence) || row.5 != 1 || row.7 != 1 || row.8 {
            return Err(invalid());
        }
        row.8 = true;
        Ok(())
    }
    pub(crate) fn take(&mut self) -> BatchJournalReceipt {
        BatchJournalReceipt {
            schema_version: 1,
            max_requests: MAX_BATCH_REQUESTS,
            max_batch: self.width,
            physical_dispatches: self.dispatched.iter().sum(),
            physical_completed: self.completed.iter().sum(),
            physical_failed: self.failed,
            dispatched_batch_distribution: self.dispatched,
            completed_batch_distribution: self.completed,
            rows: std::mem::take(&mut self.rows)
                .into_iter()
                .flatten()
                .collect(),
        }
    }
    pub(crate) fn snapshot(&self) -> BatchJournalReceipt {
        BatchJournalReceipt {
            schema_version: 1,
            max_requests: MAX_BATCH_REQUESTS,
            max_batch: self.width,
            physical_dispatches: self.dispatched.iter().sum(),
            physical_completed: self.completed.iter().sum(),
            physical_failed: self.failed,
            dispatched_batch_distribution: self.dispatched,
            completed_batch_distribution: self.completed,
            rows: self.rows.iter().filter_map(Clone::clone).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> BatchJournalReceipt {
        let mut d = [0; 16];
        d[3] = 1;
        BatchJournalReceipt {
            schema_version: 1,
            max_requests: MAX_BATCH_REQUESTS,
            max_batch: 4,
            physical_dispatches: 1,
            physical_completed: 1,
            physical_failed: 0,
            dispatched_batch_distribution: d,
            completed_batch_distribution: d,
            rows: vec![
                BatchRequestRow(1, 1, 1, 1, Some(1), 1, false, 1, true),
                BatchRequestRow(2, 2, 1, 1, Some(1), 1, false, 1, true),
                BatchRequestRow(3, 3, 1, 1, Some(1), 1, true, 2, false),
                BatchRequestRow(4, 4, 1, 1, Some(1), 1, false, 6, false),
                BatchRequestRow(5, 5, 1, 1, None, 0, true, 2, false),
            ],
        }
    }
    #[test]
    fn physical_batch_is_one_execution_four_items_and_two_consumed() {
        let audit = audit_batch_journal(&fixture(), true).unwrap();
        assert_eq!(
            (
                audit.physical_completed,
                audit.completed_nn_items,
                audit.consumed_evaluations
            ),
            (1, 4, 2)
        );
        assert_eq!(audit.unused_completed_items, 2);
        assert_eq!(audit.canceled_requests, 2);
    }
    #[test]
    fn rejects_cross_root_duplicate_ids_wrong_distribution_and_unconfirmed_outputs() {
        let mut r = fixture();
        r.rows[1].3 = 2;
        assert!(audit_batch_journal(&r, true).is_err());
        let mut r = fixture();
        r.rows[1].0 = 1;
        assert!(audit_batch_journal(&r, true).is_err());
        let mut r = fixture();
        r.completed_batch_distribution[3] = 2;
        assert!(audit_batch_journal(&r, true).is_err());
        let mut r = fixture();
        for row in &mut r.rows[..4] {
            row.5 = 0;
        }
        assert!(audit_batch_journal(&r, true).is_err());
        let mut r = fixture();
        r.rows[2].8 = true;
        assert!(audit_batch_journal(&r, true).is_err());
    }
}
