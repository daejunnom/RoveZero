//! Exact persisted trace restoration issues a fresh Rules observer owner.
use super::*;

impl PositionSnapshot {
    /// The same conservative history charge used by live snapshots, available
    /// before allocating a decoded trace.
    pub fn retained_history_bytes_for_len(history_len: usize) -> Option<usize> {
        history_len
            .checked_mul(std::mem::size_of::<HistoryNode>() + 2 * std::mem::size_of::<usize>())
    }
    /// Restore the complete known trace through checked Rules moves. An imported
    /// FEN keeps its unknown prefix. The original observer revision is metadata;
    /// a fresh owner prevents old live views or undo proofs from being revived.
    pub fn from_uci_replay(
        trace: &UciReplay,
        revision: u64,
        limits: PositionLimits,
    ) -> Result<Self, PositionError> {
        if trace.moves.len() > MAX_UCI_REPLAY_PLIES
            || trace
                .moves
                .len()
                .checked_add(1)
                .is_none_or(|n| n > limits.max_history_positions)
        {
            return Err(PositionError::ResourceLimit("snapshot trace history"));
        }
        let mut position = match (trace.origin, trace.completeness) {
            (HistoryOrigin::StartPosition, HistoryCompleteness::Complete)
                if trace.start_fen == fen::START_FEN =>
            {
                Position::startpos_with_limits(limits)?
            }
            (HistoryOrigin::Fen, HistoryCompleteness::UnknownPrefix) => {
                Position::from_fen_with_limits(&trace.start_fen, limits)?
            }
            _ => return Err(PositionError::InvalidMove("snapshot trace origin")),
        };
        for movement in &trace.moves {
            position.make_move(*movement)?;
        }
        if revision < position.revision {
            return Err(PositionError::InvalidMove("snapshot trace revision"));
        }
        position.revision = revision;
        Ok(position.snapshot())
    }
}

impl Position {
    /// Create a new Rules mutation owner from an exact immutable Rules view.
    /// Old observer and undo authorities do not transfer to this owner.
    pub fn from_snapshot(
        snapshot: &PositionSnapshot,
        limits: PositionLimits,
    ) -> Result<Self, PositionError> {
        if snapshot.known_history_len() > limits.max_history_positions
            || limits.max_history_positions == 0
        {
            return Err(PositionError::ResourceLimit("snapshot history positions"));
        }
        if limits.max_perft_depth > MAX_PERFT_STACK_DEPTH {
            return Err(PositionError::ResourceLimit(
                "perft depth configuration exceeds 64",
            ));
        }
        Ok(Self {
            state: snapshot.identity.history.state.clone(),
            history: snapshot.identity.history.clone(),
            completeness: snapshot.identity.completeness,
            origin: snapshot.identity.origin,
            owner: Arc::new(()),
            revision: snapshot.revision,
            limits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_trace_roundtrip_keeps_history_revision_and_fresh_owner() {
        for imported in [false, true] {
            let mut position = if imported {
                Position::from_fen(fen::START_FEN).unwrap()
            } else {
                Position::startpos()
            };
            for text in [
                "g1f3", "g8f6", "f3g1", "f6g8", "e2e4", "a7a6", "e4e5", "d7d5",
            ] {
                position.make_move(text.parse().unwrap()).unwrap();
            }
            let undo = position.make_move("e5d6".parse().unwrap()).unwrap();
            position.unmake(undo).unwrap();
            let old = position.snapshot();
            let restored = PositionSnapshot::from_uci_replay(
                &old.uci_replay(MAX_UCI_REPLAY_PLIES).unwrap(),
                old.revision(),
                PositionLimits::default(),
            )
            .unwrap();
            assert!(restored.same_state(&old));
            assert_eq!(restored.revision(), old.revision());
            assert_eq!(restored.known_history_fens(), old.known_history_fens());
            assert!(!position.matches_snapshot(&restored));
            assert_eq!(restored.en_passant_target(), old.en_passant_target());
            let mut invalid = old.uci_replay(MAX_UCI_REPLAY_PLIES).unwrap();
            invalid.completeness = if imported {
                HistoryCompleteness::Complete
            } else {
                HistoryCompleteness::UnknownPrefix
            };
            assert!(
                PositionSnapshot::from_uci_replay(
                    &invalid,
                    old.revision(),
                    PositionLimits::default()
                )
                .is_err()
            );
        }
    }
}
