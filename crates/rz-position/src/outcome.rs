use crate::{
    BoardMove, Color, HistoryCompleteness, HistoryOrigin, PieceKind, Position, PositionError,
    Square,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Availability {
    Available,
    Unavailable,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalReason {
    Checkmate,
    Stalemate,
    DeadPosition,
    FivefoldRepetition,
    SeventyFiveMove,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlayStatus {
    Ongoing,
    Terminal {
        reason: TerminalReason,
        /// A checkmate has a winner; exact draws have none.
        winner: Option<Color>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimReason {
    ThreefoldRepetition,
    FiftyMove,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimEvidence {
    CurrentPosition,
    IntendedMove(BoardMove),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawClaim {
    pub reason: ClaimReason,
    pub evidence: ClaimEvidence,
    pub availability: Availability,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistoryEvidence {
    /// Model/game history completeness remains unknown after an imported FEN.
    pub completeness: HistoryCompleteness,
    pub origin: HistoryOrigin,
    pub known_repetition_count: usize,
    /// An irreversible boundary can make repetition evidence complete even when
    /// the earlier model/game history is still unknown.
    pub repetition_is_complete: bool,
    /// Incomplete history cannot prove that a missing fifth occurrence did not
    /// happen. Known five occurrences, however, are sufficient proof.
    pub fivefold_repetition: Availability,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositionClassification {
    pub play_status: PlayStatus,
    /// Current threshold evidence is always reported. Intended-move entries
    /// contain available claims and repetition claims limited by unknown
    /// history. `play_status` determines whether play has already ended and a
    /// claim can still be exercised.
    pub claim_availability: Vec<DrawClaim>,
    pub history_evidence: HistoryEvidence,
}

impl Position {
    /// Classify a checked state using FIDE claim and automatic-draw thresholds.
    ///
    /// Threefold repetition and 100 halfmoves are claim conditions, not exact
    /// terminal outcomes. A mate on the 150th halfmove takes precedence over
    /// the automatic 75-move draw. Unknown game history affects repetition
    /// evidence rather than unrelated mate, stalemate or counter evidence.
    ///
    /// Dead-position detection covers the proven material subset of bare kings,
    /// a single bishop or knight against a bare king, and positions containing
    /// only kings and bishops confined to one square color. Other positions are
    /// not certified dead by this implementation.
    ///
    /// Operational errors while checking intended moves are returned rather
    /// than silently dropping claim evidence.
    pub fn classify_position(&self) -> Result<PositionClassification, PositionError> {
        let known_repetitions = self.known_repetition_count();
        let repetition_complete = self.repetition_history_complete();
        let fivefold = repetition_availability(known_repetitions, 5, repetition_complete);
        let history_evidence = HistoryEvidence {
            completeness: self.history_completeness(),
            origin: self.history_origin(),
            known_repetition_count: known_repetitions,
            repetition_is_complete: repetition_complete,
            fivefold_repetition: fivefold,
        };
        let mut claim_availability = vec![
            DrawClaim {
                reason: ClaimReason::ThreefoldRepetition,
                evidence: ClaimEvidence::CurrentPosition,
                availability: repetition_availability(known_repetitions, 3, repetition_complete),
            },
            DrawClaim {
                reason: ClaimReason::FiftyMove,
                evidence: ClaimEvidence::CurrentPosition,
                availability: if self.halfmove_clock() >= 100 {
                    Availability::Available
                } else {
                    Availability::Unavailable
                },
            },
        ];

        let legal = self.legal_moves();
        let play_status = if legal.is_empty() {
            if self.in_check() {
                PlayStatus::Terminal {
                    reason: TerminalReason::Checkmate,
                    winner: Some(self.side_to_move().opposite()),
                }
            } else {
                draw(TerminalReason::Stalemate)
            }
        } else if has_proven_dead_material(self) {
            draw(TerminalReason::DeadPosition)
        } else if fivefold == Availability::Available {
            draw(TerminalReason::FivefoldRepetition)
        } else if self.halfmove_clock() >= 150 {
            draw(TerminalReason::SeventyFiveMove)
        } else {
            PlayStatus::Ongoing
        };

        if play_status == PlayStatus::Ongoing {
            for mv in legal {
                let child = self.preview_generated(mv)?;
                let repetition = repetition_availability(
                    child.known_repetition_count(),
                    3,
                    child.repetition_history_complete(),
                );
                if repetition != Availability::Unavailable {
                    claim_availability.push(DrawClaim {
                        reason: ClaimReason::ThreefoldRepetition,
                        evidence: ClaimEvidence::IntendedMove(mv),
                        availability: repetition,
                    });
                }
                if child.halfmove_clock() >= 100 {
                    claim_availability.push(DrawClaim {
                        reason: ClaimReason::FiftyMove,
                        evidence: ClaimEvidence::IntendedMove(mv),
                        availability: Availability::Available,
                    });
                }
            }
        }

        Ok(PositionClassification {
            play_status,
            claim_availability,
            history_evidence,
        })
    }
}

fn repetition_availability(known: usize, needed: usize, complete: bool) -> Availability {
    if known >= needed {
        Availability::Available
    } else if complete {
        Availability::Unavailable
    } else {
        Availability::Unknown
    }
}

fn draw(reason: TerminalReason) -> PlayStatus {
    PlayStatus::Terminal {
        reason,
        winner: None,
    }
}

fn has_proven_dead_material(position: &Position) -> bool {
    let mut minor_count = 0;
    let mut bishop_color = None;
    let mut has_knight = false;
    let mut bishops_on_both_colors = false;
    for index in 0..64 {
        let square = Square(index);
        let Some(piece) = position.piece_at(square) else {
            continue;
        };
        match piece.kind {
            PieceKind::King => {}
            PieceKind::Pawn | PieceKind::Rook | PieceKind::Queen => return false,
            PieceKind::Knight => {
                minor_count += 1;
                has_knight = true;
            }
            PieceKind::Bishop => {
                minor_count += 1;
                let color = (square.file() + square.rank()) % 2;
                if let Some(previous) = bishop_color {
                    bishops_on_both_colors |= previous != color;
                } else {
                    bishop_color = Some(color);
                }
            }
        }
    }
    minor_count <= 1 || (!has_knight && !bishops_on_both_colors)
}
