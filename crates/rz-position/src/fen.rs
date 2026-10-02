use crate::{movegen::is_attacked, types::*};

pub(crate) const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub(crate) fn parse(fen: &str, limits: PositionLimits) -> Result<CoreState, PositionError> {
    if fen.len() > limits.max_fen_bytes { return Err(PositionError::ResourceLimit("FEN bytes")); }
    if !fen.is_ascii() { return Err(PositionError::InvalidFen("non-ASCII input")); }
    let fields: Vec<_> = fen.split_ascii_whitespace().collect();
    if fields.len() != 6 { return Err(PositionError::InvalidFen("expected six fields")); }
    let ranks: Vec<_> = fields[0].split('/').collect();
    if ranks.len() != 8 { return Err(PositionError::InvalidFen("expected eight ranks")); }
    let mut board = Board::empty();
    for (i, rank) in ranks.iter().enumerate() {
        let mut file = 0u8;
        let mut last_digit = false;
        for b in rank.bytes() {
            if (b'1'..=b'8').contains(&b) {
                if last_digit { return Err(PositionError::InvalidFen("adjacent empty-square digits")); }
                file = file.checked_add(b - b'0').ok_or(PositionError::InvalidFen("rank overflow"))?;
                last_digit = true;
            } else {
                if file >= 8 { return Err(PositionError::InvalidFen("rank too long")); }
                let kind = match b.to_ascii_lowercase() {
                    b'p' => PieceKind::Pawn, b'n' => PieceKind::Knight, b'b' => PieceKind::Bishop,
                    b'r' => PieceKind::Rook, b'q' => PieceKind::Queen, b'k' => PieceKind::King,
                    _ => return Err(PositionError::InvalidFen("unknown piece")),
                };
                let color = if b.is_ascii_uppercase() { Color::White } else { Color::Black };
                board.set(Square((7 - i as u8) * 8 + file), Some(Piece { color, kind }));
                file += 1;
                last_digit = false;
            }
            if file > 8 { return Err(PositionError::InvalidFen("rank too long")); }
        }
        if file != 8 { return Err(PositionError::InvalidFen("rank has fewer than eight squares")); }
    }
    for color in [Color::White, Color::Black] {
        if board.bitboards[color.index() * 6 + PieceKind::King as usize].count_ones() != 1 { return Err(PositionError::InvalidFen("expected one king per color")); }
        let pawns = board.bitboards[color.index() * 6];
        if pawns & 0xff000000000000ff != 0 { return Err(PositionError::InvalidFen("pawn on back rank")); }
        if pawns.count_ones() > 8 || board.bitboards[color.index() * 6..color.index() * 6 + 6].iter().map(|x| x.count_ones()).sum::<u32>() > 16 {
            return Err(PositionError::InvalidFen("too many pieces or pawns"));
        }
    }
    let side = match fields[1] { "w" => Color::White, "b" => Color::Black, _ => return Err(PositionError::InvalidFen("invalid side to move")) };
    let mut castling = 0u8;
    if fields[2] != "-" {
        for b in fields[2].bytes() {
            let (mask, color, rook) = match b { b'K' => (1, Color::White, 7), b'Q' => (2, Color::White, 0), b'k' => (4, Color::Black, 63), b'q' => (8, Color::Black, 56), _ => return Err(PositionError::InvalidFen("invalid castling rights")) };
            if castling & mask != 0 { return Err(PositionError::InvalidFen("duplicate castling right")); }
            let king = if color == Color::White { 4 } else { 60 };
            if board.get(Square(king)) != Some(Piece { color, kind: PieceKind::King }) || board.get(Square(rook)) != Some(Piece { color, kind: PieceKind::Rook }) {
                return Err(PositionError::InvalidFen("castling right lacks home king or rook"));
            }
            castling |= mask;
        }
    }
    let counter = |s: &str| -> Result<u32, PositionError> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) { return Err(PositionError::InvalidFen("invalid rule counter")); }
        s.parse().map_err(|_| PositionError::InvalidFen("rule counter outside u32"))
    };
    let halfmove = counter(fields[4])?;
    let fullmove = counter(fields[5])?;
    if fullmove == 0 { return Err(PositionError::InvalidFen("fullmove number must be positive")); }
    let ep = if fields[3] == "-" { None } else {
        let square: Square = fields[3].parse().map_err(|_| PositionError::InvalidFen("invalid en-passant target"))?;
        let (rank, captured_offset, origin_offset) = if side == Color::White { (5, -1, 1) } else { (2, 1, -1) };
        if square.rank() != rank || board.get(square).is_some() || halfmove != 0 || (side == Color::White && fullmove == 1) {
            return Err(PositionError::InvalidFen("inconsistent en-passant target"));
        }
        let captured = square.offset(0, captured_offset).expect("EP rank checked");
        let origin = square.offset(0, origin_offset).expect("EP rank checked");
        if board.get(captured) != Some(Piece { color: side.opposite(), kind: PieceKind::Pawn }) || board.get(origin).is_some() {
            return Err(PositionError::InvalidFen("en-passant target lacks preceding double pawn push"));
        }
        Some(square)
    };
    let state = CoreState { board, side, castling, ep, halfmove, fullmove };
    if is_attacked(&state, state.board.king(side.opposite()), side) { return Err(PositionError::InvalidFen("non-moving king is in check")); }
    Ok(state)
}

pub(crate) fn format(state: &CoreState) -> String {
    let mut s = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0u8;
        for file in 0..8 {
            if let Some(piece) = state.board.get(Square(rank * 8 + file)) {
                if empty > 0 { s.push((b'0' + empty) as char); empty = 0; }
                let c: char = match piece.kind { PieceKind::Pawn => 'p', PieceKind::Knight => 'n', PieceKind::Bishop => 'b', PieceKind::Rook => 'r', PieceKind::Queen => 'q', PieceKind::King => 'k' };
                s.push(if piece.color == Color::White { c.to_ascii_uppercase() } else { c });
            } else { empty += 1; }
        }
        if empty > 0 { s.push((b'0' + empty) as char); }
        if rank > 0 { s.push('/'); }
    }
    s.push_str(if state.side == Color::White { " w " } else { " b " });
    if state.castling == 0 { s.push('-'); } else {
        for (mask, c) in [(1, 'K'), (2, 'Q'), (4, 'k'), (8, 'q')] { if state.castling & mask != 0 { s.push(c); } }
    }
    s.push(' ');
    s.push_str(&state.ep.map(|x| x.to_string()).unwrap_or_else(|| "-".into()));
    s.push_str(&format!(" {} {}", state.halfmove, state.fullmove));
    s
}
