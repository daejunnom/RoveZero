use crate::{movegen::is_attacked, types::*};

pub(crate) const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub(crate) fn parse(fen: &str, limits: PositionLimits) -> Result<CoreState, PositionError> {
    if fen.len() > limits.max_fen_bytes {
        return Err(PositionError::ResourceLimit("FEN bytes"));
    }
    if !fen.is_ascii() {
        return Err(PositionError::InvalidFen("non-ASCII input"));
    }
    let fields: Vec<_> = fen.split_ascii_whitespace().collect();
    if fields.len() != 6 {
        return Err(PositionError::InvalidFen("expected six fields"));
    }
    let ranks: Vec<_> = fields[0].split('/').collect();
    if ranks.len() != 8 {
        return Err(PositionError::InvalidFen("expected eight ranks"));
    }
    let mut board = Board::empty();
    for (i, rank) in ranks.iter().enumerate() {
        let mut file = 0u8;
        let mut last_digit = false;
        for b in rank.bytes() {
            if (b'1'..=b'8').contains(&b) {
                if last_digit {
                    return Err(PositionError::InvalidFen("adjacent empty-square digits"));
                }
                file = file
                    .checked_add(b - b'0')
                    .ok_or(PositionError::InvalidFen("rank overflow"))?;
                last_digit = true;
            } else {
                if file >= 8 {
                    return Err(PositionError::InvalidFen("rank too long"));
                }
                let kind = match b.to_ascii_lowercase() {
                    b'p' => PieceKind::Pawn,
                    b'n' => PieceKind::Knight,
                    b'b' => PieceKind::Bishop,
                    b'r' => PieceKind::Rook,
                    b'q' => PieceKind::Queen,
                    b'k' => PieceKind::King,
                    _ => return Err(PositionError::InvalidFen("unknown piece")),
                };
                let color = if b.is_ascii_uppercase() {
                    Color::White
                } else {
                    Color::Black
                };
                board.set(
                    Square((7 - i as u8) * 8 + file),
                    Some(Piece { color, kind }),
                );
                file += 1;
                last_digit = false;
            }
            if file > 8 {
                return Err(PositionError::InvalidFen("rank too long"));
            }
        }
        if file != 8 {
            return Err(PositionError::InvalidFen(
                "rank has fewer than eight squares",
            ));
        }
    }
    for color in [Color::White, Color::Black] {
        if board.bitboards[color.index() * 6 + PieceKind::King as usize].count_ones() != 1 {
            return Err(PositionError::InvalidFen("expected one king per color"));
        }
        let pawns = board.bitboards[color.index() * 6];
        if pawns & 0xff000000000000ff != 0 {
            return Err(PositionError::InvalidFen("pawn on back rank"));
        }
        if pawns.count_ones() > 8
            || board.bitboards[color.index() * 6..color.index() * 6 + 6]
                .iter()
                .map(|x| x.count_ones())
                .sum::<u32>()
                > 16
        {
            return Err(PositionError::InvalidFen("too many pieces or pawns"));
        }
        // Each additional original-stock piece needs a missing pawn. Bishops
        // cannot change square color, so count their original stock separately.
        let count =
            |kind: PieceKind| board.bitboards[color.index() * 6 + kind as usize].count_ones();
        let bishops = board.bitboards[color.index() * 6 + PieceKind::Bishop as usize];
        let light = (bishops & 0x55aa55aa55aa55aa).count_ones();
        let dark = (bishops & !0x55aa55aa55aa55aa).count_ones();
        let minimum_promotions = count(PieceKind::Queen).saturating_sub(1)
            + count(PieceKind::Rook).saturating_sub(2)
            + count(PieceKind::Knight).saturating_sub(2)
            + light.saturating_sub(1)
            + dark.saturating_sub(1);
        if minimum_promotions > 8 - pawns.count_ones() {
            return Err(PositionError::InvalidFen(
                "promoted inventory lacks missing pawns",
            ));
        }
    }
    let side = match fields[1] {
        "w" => Color::White,
        "b" => Color::Black,
        _ => return Err(PositionError::InvalidFen("invalid side to move")),
    };
    let mut castling = 0u8;
    if fields[2] != "-" {
        for b in fields[2].bytes() {
            let (mask, color, rook) = match b {
                b'K' => (1, Color::White, 7),
                b'Q' => (2, Color::White, 0),
                b'k' => (4, Color::Black, 63),
                b'q' => (8, Color::Black, 56),
                _ => return Err(PositionError::InvalidFen("invalid castling rights")),
            };
            if castling & mask != 0 {
                return Err(PositionError::InvalidFen("duplicate castling right"));
            }
            let king = if color == Color::White { 4 } else { 60 };
            if board.get(Square(king))
                != Some(Piece {
                    color,
                    kind: PieceKind::King,
                })
                || board.get(Square(rook))
                    != Some(Piece {
                        color,
                        kind: PieceKind::Rook,
                    })
            {
                return Err(PositionError::InvalidFen(
                    "castling right lacks home king or rook",
                ));
            }
            castling |= mask;
        }
    }
    let counter = |s: &str| -> Result<u32, PositionError> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(PositionError::InvalidFen("invalid rule counter"));
        }
        s.parse()
            .map_err(|_| PositionError::InvalidFen("rule counter outside u32"))
    };
    let halfmove = counter(fields[4])?;
    let fullmove = counter(fields[5])?;
    if fullmove == 0 {
        return Err(PositionError::InvalidFen(
            "fullmove number must be positive",
        ));
    }
    let ep = if fields[3] == "-" {
        None
    } else {
        let square: Square = fields[3]
            .parse()
            .map_err(|_| PositionError::InvalidFen("invalid en-passant target"))?;
        let (rank, captured_offset, origin_offset) = if side == Color::White {
            (5, -1, 1)
        } else {
            (2, 1, -1)
        };
        if square.rank() != rank
            || board.get(square).is_some()
            || halfmove != 0
            || (side == Color::White && fullmove == 1)
        {
            return Err(PositionError::InvalidFen("inconsistent en-passant target"));
        }
        let captured = square.offset(0, captured_offset).expect("EP rank checked");
        let origin = square.offset(0, origin_offset).expect("EP rank checked");
        if board.get(captured)
            != Some(Piece {
                color: side.opposite(),
                kind: PieceKind::Pawn,
            })
            || board.get(origin).is_some()
        {
            return Err(PositionError::InvalidFen(
                "en-passant target lacks preceding double pawn push",
            ));
        }
        Some(square)
    };
    let state = CoreState {
        board,
        side,
        castling,
        ep,
        halfmove,
        fullmove,
    };
    if is_attacked(&state, state.board.king(side.opposite()), side) {
        return Err(PositionError::InvalidFen("non-moving king is in check"));
    }
    if crate::movegen::attackers(&state, state.board.king(side), side.opposite()).count_ones() > 2 {
        return Err(PositionError::InvalidFen(
            "more than two simultaneous checkers",
        ));
    }
    if let Some(target) = state.ep {
        // Raw EP asserts the immediately preceding double push. The opponent
        // could not already be in check in that predecessor with this mover.
        let back = if side == Color::White { -1 } else { 1 };
        let mut previous = state.clone();
        previous
            .board
            .set(target.offset(0, back).expect("checked EP rank"), None);
        previous.board.set(
            target.offset(0, -back).expect("checked EP rank"),
            Some(Piece {
                color: side.opposite(),
                kind: PieceKind::Pawn,
            }),
        );
        if is_attacked(&previous, previous.board.king(side), side.opposite()) {
            return Err(PositionError::InvalidFen(
                "en-passant predecessor has non-moving king in check",
            ));
        }
    }
    Ok(state)
}

fn format_board_fields(state: &CoreState, s: &mut String) {
    for rank in (0..8).rev() {
        let mut empty = 0u8;
        for file in 0..8 {
            if let Some(piece) = state.board.get(Square(rank * 8 + file)) {
                if empty > 0 {
                    s.push((b'0' + empty) as char);
                    empty = 0;
                }
                let c: char = match piece.kind {
                    PieceKind::Pawn => 'p',
                    PieceKind::Knight => 'n',
                    PieceKind::Bishop => 'b',
                    PieceKind::Rook => 'r',
                    PieceKind::Queen => 'q',
                    PieceKind::King => 'k',
                };
                s.push(if piece.color == Color::White {
                    c.to_ascii_uppercase()
                } else {
                    c
                });
            } else {
                empty += 1;
            }
        }
        if empty > 0 {
            s.push((b'0' + empty) as char);
        }
        if rank > 0 {
            s.push('/');
        }
    }
    s.push_str(if state.side == Color::White {
        " w "
    } else {
        " b "
    });
    if state.castling == 0 {
        s.push('-');
    } else {
        for (mask, c) in [(1, 'K'), (2, 'Q'), (4, 'k'), (8, 'q')] {
            if state.castling & mask != 0 {
                s.push(c);
            }
        }
    }
}

pub(crate) fn format(state: &CoreState) -> String {
    let mut s = String::new();
    format_board_fields(state, &mut s);
    s.push(' ');
    s.push_str(
        &state
            .ep
            .map(|x| x.to_string())
            .unwrap_or_else(|| "-".into()),
    );
    s.push_str(&format!(" {} {}", state.halfmove, state.fullmove));
    s
}

/// Reuse one buffer for canonical rule frames. The longest representation
/// is 64 pieces + seven '/' + side/rights/EP + two ten-digit u32 counters.
pub(crate) const MAX_CANONICAL_FEN_BYTES: usize = 103;

pub(crate) fn format_into(state: &CoreState, buffer: &mut String) {
    use std::fmt::Write;
    buffer.clear();
    format_board_fields(state, buffer);
    buffer.push(' ');
    if let Some(target) = state.ep {
        buffer.push(char::from(b'a' + target.file()));
        buffer.push(char::from(b'1' + target.rank()));
    } else {
        buffer.push('-');
    }
    write!(buffer, " {} {}", state.halfmove, state.fullmove)
        .expect("formatting into a String cannot fail");
}

#[cfg(all(test, feature = "experimental-history-digest"))]
mod tests {
    use super::*;

    #[test]
    fn reusable_fen_frames_match_owned_format_and_keep_their_capacity() {
        let mut positions = vec![crate::Position::startpos()];
        for input in [
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
            "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 2",
            "1r2k3/P7/8/8/8/8/7p/R3K3 w Q - 99 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 4294967295 4294967295",
        ] {
            positions.push(crate::Position::from_fen(input).unwrap());
        }
        let mut trace = crate::Position::startpos();
        for index in 0..128 {
            trace
                .make_uci(["g1f3", "g8f6", "f3g1", "f6g8"][index % 4])
                .unwrap();
        }
        positions.push(trace);
        let mut buffer = String::with_capacity(MAX_CANONICAL_FEN_BYTES);
        let capacity = buffer.capacity();
        buffer.push_str("old content must be cleared");
        for position in positions {
            for (state, _) in position.snapshot().history_states() {
                // Exercise reuse even after a preceding frame's different length.
                format_into(state, &mut buffer);
                assert_eq!(buffer, format(state));
                assert!(buffer.len() <= MAX_CANONICAL_FEN_BYTES);
                assert_eq!(buffer.capacity(), capacity);
                let reparsed = parse(&buffer, PositionLimits::default()).unwrap();
                assert_eq!(format(&reparsed), buffer);
            }
        }
    }
}
