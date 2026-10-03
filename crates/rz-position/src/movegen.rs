use crate::types::*;

const KNIGHT: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const KING: [(i8, i8); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
const ROOK: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

pub(crate) fn is_attacked(state: &CoreState, square: Square, by: Color) -> bool {
    attackers(state, square, by) != 0
}

#[cfg(not(feature = "experimental-attack-tables"))]
pub(crate) fn attackers(state: &CoreState, square: Square, by: Color) -> u64 {
    let mut mask = 0;
    let pawn_back = if by == Color::White { -1 } else { 1 };
    for df in [-1, 1] {
        if let Some(s) = square.offset(df, pawn_back) {
            if state.board.get(s)
                == Some(Piece {
                    color: by,
                    kind: PieceKind::Pawn,
                })
            {
                mask |= 1u64 << s.0;
            }
        }
    }
    for (df, dr) in KNIGHT {
        if let Some(s) = square.offset(df, dr) {
            if state.board.get(s)
                == Some(Piece {
                    color: by,
                    kind: PieceKind::Knight,
                })
            {
                mask |= 1u64 << s.0;
            }
        }
    }
    for (df, dr) in KING {
        if let Some(s) = square.offset(df, dr) {
            if state.board.get(s)
                == Some(Piece {
                    color: by,
                    kind: PieceKind::King,
                })
            {
                mask |= 1u64 << s.0;
            }
        }
    }
    for (directions, slider) in [
        (&ROOK[..], PieceKind::Rook),
        (&BISHOP[..], PieceKind::Bishop),
    ] {
        for &(df, dr) in directions {
            let mut at = square.offset(df, dr);
            while let Some(s) = at {
                if let Some(piece) = state.board.get(s) {
                    if piece.color == by && (piece.kind == slider || piece.kind == PieceKind::Queen)
                    {
                        mask |= 1u64 << s.0;
                    }
                    break;
                }
                at = s.offset(df, dr);
            }
        }
    }
    mask
}

#[cfg(feature = "experimental-attack-tables")]
const fn leaps(directions: [(i8, i8); 8]) -> [u64; 64] {
    let mut table = [0; 64];
    let mut at = 0;
    while at < 64 {
        let mut i = 0;
        while i < 8 {
            let f = (at % 8) as i8 + directions[i].0;
            let r = (at / 8) as i8 + directions[i].1;
            if f >= 0 && f < 8 && r >= 0 && r < 8 {
                table[at] |= 1u64 << (r * 8 + f);
            }
            i += 1;
        }
        at += 1;
    }
    table
}
#[cfg(feature = "experimental-attack-tables")]
const KNIGHT_ATTACKS: [u64; 64] = leaps(KNIGHT);
#[cfg(feature = "experimental-attack-tables")]
const KING_ATTACKS: [u64; 64] = leaps(KING);
#[cfg(feature = "experimental-attack-tables")]
const fn rays() -> [[u64; 8]; 64] {
    let mut result = [[0; 8]; 64];
    let mut at = 0;
    while at < 64 {
        let mut i = 0;
        while i < 8 {
            let mut f = (at % 8) as i8 + KING[i].0;
            let mut r = (at / 8) as i8 + KING[i].1;
            while f >= 0 && f < 8 && r >= 0 && r < 8 {
                result[at][i] |= 1u64 << (r * 8 + f);
                f += KING[i].0;
                r += KING[i].1;
            }
            i += 1;
        }
        at += 1;
    }
    result
}
#[cfg(feature = "experimental-attack-tables")]
const RAYS: [[u64; 8]; 64] = rays();
#[cfg(feature = "experimental-attack-tables")]
pub(crate) fn attackers(state: &CoreState, square: Square, by: Color) -> u64 {
    let pieces = &state.board.bitboards;
    let base = by.index() * 6;
    let at = square.0 as usize;
    let mut pawn_sources = 0;
    let pawn_back = if by == Color::White { -1 } else { 1 };
    for df in [-1, 1] {
        if let Some(s) = square.offset(df, pawn_back) {
            pawn_sources |= 1u64 << s.0;
        }
    }
    let mut mask = pawn_sources & pieces[base]
        | KNIGHT_ATTACKS[at] & pieces[base + PieceKind::Knight as usize]
        | KING_ATTACKS[at] & pieces[base + PieceKind::King as usize];
    let occupied = pieces.iter().fold(0, |all, board| all | board);
    for (i, &(df, dr)) in KING.iter().enumerate() {
        let blockers = RAYS[at][i] & occupied;
        if blockers == 0 {
            continue;
        }
        let index = if df + dr * 8 > 0 {
            blockers.trailing_zeros()
        } else {
            63 - blockers.leading_zeros()
        };
        let slider = if df == 0 || dr == 0 {
            PieceKind::Rook
        } else {
            PieceKind::Bishop
        };
        mask |= (1u64 << index)
            & (pieces[base + slider as usize] | pieces[base + PieceKind::Queen as usize]);
    }
    mask
}

fn add(moves: &mut Vec<BoardMove>, from: Square, to: Square, promotion: bool) {
    if promotion {
        for kind in [
            PieceKind::Queen,
            PieceKind::Rook,
            PieceKind::Bishop,
            PieceKind::Knight,
        ] {
            moves.push(BoardMove {
                from,
                to,
                promotion: Some(kind),
            });
        }
    } else {
        moves.push(BoardMove {
            from,
            to,
            promotion: None,
        });
    }
}

fn pseudo(state: &CoreState) -> Vec<BoardMove> {
    let mut moves = Vec::with_capacity(64);
    for index in 0..64 {
        let from = Square(index);
        let Some(piece) = state.board.get(from) else {
            continue;
        };
        if piece.color != state.side {
            continue;
        }
        let can_land = |s: Square| match state.board.get(s) {
            None => true,
            Some(p) => p.color != piece.color && p.kind != PieceKind::King,
        };
        match piece.kind {
            PieceKind::Pawn => {
                let dr = if piece.color == Color::White { 1 } else { -1 };
                if let Some(to) = from.offset(0, dr) {
                    if state.board.get(to).is_none() {
                        add(&mut moves, from, to, to.rank() == 0 || to.rank() == 7);
                        if from.rank() == if piece.color == Color::White { 1 } else { 6 } {
                            let double = from.offset(0, dr * 2).expect("pawn on starting rank");
                            if state.board.get(double).is_none() {
                                add(&mut moves, from, double, false);
                            }
                        }
                    }
                }
                for df in [-1, 1] {
                    if let Some(to) = from.offset(df, dr) {
                        let capture = state
                            .board
                            .get(to)
                            .is_some_and(|p| p.color != piece.color && p.kind != PieceKind::King);
                        let ep = state.ep == Some(to)
                            && state.board.get(to).is_none()
                            && to.offset(0, -dr).is_some_and(|s| {
                                state.board.get(s)
                                    == Some(Piece {
                                        color: piece.color.opposite(),
                                        kind: PieceKind::Pawn,
                                    })
                            });
                        if capture || ep {
                            add(&mut moves, from, to, to.rank() == 0 || to.rank() == 7);
                        }
                    }
                }
            }
            PieceKind::Knight | PieceKind::King => {
                let directions = if piece.kind == PieceKind::Knight {
                    &KNIGHT
                } else {
                    &KING
                };
                for &(df, dr) in directions {
                    if let Some(to) = from.offset(df, dr) {
                        if can_land(to) {
                            add(&mut moves, from, to, false);
                        }
                    }
                }
                if piece.kind == PieceKind::King {
                    castles(state, from, &mut moves);
                }
            }
            PieceKind::Bishop | PieceKind::Rook | PieceKind::Queen => {
                let directions: &[(i8, i8)] = match piece.kind {
                    PieceKind::Bishop => &BISHOP,
                    PieceKind::Rook => &ROOK,
                    _ => &KING,
                };
                for &(df, dr) in directions {
                    let mut at = from.offset(df, dr);
                    while let Some(to) = at {
                        if can_land(to) {
                            add(&mut moves, from, to, false);
                        }
                        if state.board.get(to).is_some() {
                            break;
                        }
                        at = to.offset(df, dr);
                    }
                }
            }
        }
    }
    moves
}

fn castles(state: &CoreState, from: Square, moves: &mut Vec<BoardMove>) {
    let (base, flags) = if state.side == Color::White {
        (0, 1)
    } else {
        (56, 4)
    };
    if from != Square(base + 4) || is_attacked(state, from, state.side.opposite()) {
        return;
    }
    for (mask, rook, empty, transit, destination) in [
        (
            flags,
            base + 7,
            [Some(base + 5), Some(base + 6), None],
            base + 5,
            base + 6,
        ),
        (
            flags * 2,
            base,
            [Some(base + 1), Some(base + 2), Some(base + 3)],
            base + 3,
            base + 2,
        ),
    ] {
        if state.castling & mask == 0
            || state.board.get(Square(rook))
                != Some(Piece {
                    color: state.side,
                    kind: PieceKind::Rook,
                })
            || empty
                .iter()
                .flatten()
                .any(|&s| state.board.get(Square(s)).is_some())
        {
            continue;
        }
        let mut through = state.clone();
        through.board.set(from, None);
        through.board.set(
            Square(transit),
            Some(Piece {
                color: state.side,
                kind: PieceKind::King,
            }),
        );
        if !is_attacked(&through, Square(transit), state.side.opposite()) {
            add(moves, from, Square(destination), false);
        }
    }
}

fn precise_legal(state: &CoreState, mv: BoardMove) -> bool {
    let mut child = state.clone();
    apply(&mut child, mv);
    !is_attacked(&child, child.board.king(state.side), state.side.opposite())
}

#[cfg(feature = "experimental-pin-check")]
struct KingConstraints {
    pins: [u64; 64],
    evasion: u64,
}
#[cfg(feature = "experimental-pin-check")]
fn king_constraints(state: &CoreState) -> KingConstraints {
    let king = state.board.king(state.side);
    let checkers = attackers(state, king, state.side.opposite());
    let mut constraints = KingConstraints {
        pins: [u64::MAX; 64],
        evasion: u64::MAX,
    };
    if checkers.count_ones() > 1 {
        constraints.evasion = 0;
    } else if checkers != 0 {
        let checker = Square(checkers.trailing_zeros() as u8);
        constraints.evasion = checkers;
        let kind = state.board.get(checker).expect("checker inventory").kind;
        if matches!(kind, PieceKind::Bishop | PieceKind::Rook | PieceKind::Queen) {
            let df = (checker.file() as i8 - king.file() as i8).signum();
            let dr = (checker.rank() as i8 - king.rank() as i8).signum();
            let mut at = king.offset(df, dr);
            while let Some(s) = at {
                constraints.evasion |= 1u64 << s.0;
                if s == checker {
                    break;
                }
                at = s.offset(df, dr);
            }
        }
    }
    for (df, dr) in KING {
        let slider = if df == 0 || dr == 0 {
            PieceKind::Rook
        } else {
            PieceKind::Bishop
        };
        let mut candidate = None;
        let mut ray = 0;
        let mut at = king.offset(df, dr);
        while let Some(s) = at {
            ray |= 1u64 << s.0;
            if let Some(piece) = state.board.get(s) {
                if piece.color == state.side && candidate.is_none() {
                    candidate = Some(s);
                } else {
                    if piece.color != state.side
                        && (piece.kind == slider || piece.kind == PieceKind::Queen)
                    {
                        if let Some(pinned) = candidate {
                            constraints.pins[pinned.0 as usize] = ray;
                        }
                    }
                    break;
                }
            }
            at = s.offset(df, dr);
        }
    }
    constraints
}

pub(crate) fn legal(state: &CoreState) -> Vec<BoardMove> {
    #[cfg(feature = "experimental-pin-check")]
    let constraints = king_constraints(state);
    let mut moves: Vec<_> = pseudo(state)
        .into_iter()
        .filter(|&mv| {
            #[cfg(feature = "experimental-pin-check")]
            {
                let piece = state.board.get(mv.from).expect("pseudo inventory");
                // King/castling and EP may change attacks outside the destination;
                // retain the exact clone/apply/check path for those moves.
                let ep = piece.kind == PieceKind::Pawn
                    && state.ep == Some(mv.to)
                    && mv.from.file() != mv.to.file()
                    && state.board.get(mv.to).is_none();
                if piece.kind != PieceKind::King && !ep {
                    return (1u64 << mv.to.0)
                        & constraints.evasion
                        & constraints.pins[mv.from.0 as usize]
                        != 0;
                }
            }
            precise_legal(state, mv)
        })
        .collect();
    moves.sort_unstable_by_key(|mv| mv.sort_key());
    #[cfg(all(test, feature = "experimental-pin-check"))]
    {
        let mut reference: Vec<_> = pseudo(state)
            .into_iter()
            .filter(|&mv| precise_legal(state, mv))
            .collect();
        reference.sort_unstable_by_key(|mv| mv.sort_key());
        assert_eq!(moves, reference, "pin/check differs from full transition");
    }
    moves
}

/// Only called with a generated pseudo-legal move and a checked king inventory.
/// Saturated trial counters are never published: checked make validates counters.
pub(crate) fn apply(state: &mut CoreState, mv: BoardMove) -> bool {
    let piece = state
        .board
        .get(mv.from)
        .expect("generated move has a piece");
    let mut capture = state.board.get(mv.to).is_some();
    if piece.kind == PieceKind::Pawn
        && state.ep == Some(mv.to)
        && mv.from.file() != mv.to.file()
        && !capture
    {
        let dr = if piece.color == Color::White { -1 } else { 1 };
        state
            .board
            .set(mv.to.offset(0, dr).expect("EP target rank"), None);
        capture = true;
    }
    state.board.set(mv.from, None);
    state.board.set(
        mv.to,
        Some(Piece {
            color: piece.color,
            kind: mv.promotion.unwrap_or(piece.kind),
        }),
    );
    if piece.kind == PieceKind::King && mv.from.file().abs_diff(mv.to.file()) == 2 {
        let (rook_from, rook_to) = if mv.to.file() == 6 {
            (
                Square(mv.from.rank() * 8 + 7),
                Square(mv.from.rank() * 8 + 5),
            )
        } else {
            (Square(mv.from.rank() * 8), Square(mv.from.rank() * 8 + 3))
        };
        state.board.set(rook_from, None);
        state.board.set(
            rook_to,
            Some(Piece {
                color: piece.color,
                kind: PieceKind::Rook,
            }),
        );
    }
    if piece.kind == PieceKind::King {
        state.castling &= if piece.color == Color::White { !3 } else { !12 };
    }
    for s in [mv.from, mv.to] {
        state.castling &= match s.0 {
            0 => !2,
            7 => !1,
            56 => !8,
            63 => !4,
            _ => 255,
        };
    }
    state.ep = if piece.kind == PieceKind::Pawn && mv.from.rank().abs_diff(mv.to.rank()) == 2 {
        Some(Square((mv.from.0 + mv.to.0) / 2))
    } else {
        None
    };
    state.halfmove = if piece.kind == PieceKind::Pawn || capture {
        0
    } else {
        state.halfmove.saturating_add(1)
    };
    if state.side == Color::Black {
        state.fullmove = state.fullmove.saturating_add(1);
    }
    state.side = state.side.opposite();
    capture
}

pub(crate) fn legal_ep(state: &CoreState) -> Option<Square> {
    let ep = state.ep?;
    let back = if state.side == Color::White { -1 } else { 1 };
    for df in [-1, 1] {
        if let Some(from) = ep.offset(df, back) {
            if state.board.get(from)
                == Some(Piece {
                    color: state.side,
                    kind: PieceKind::Pawn,
                })
            {
                let mut child = state.clone();
                apply(
                    &mut child,
                    BoardMove {
                        from,
                        to: ep,
                        promotion: None,
                    },
                );
                if !is_attacked(&child, child.board.king(state.side), state.side.opposite()) {
                    return Some(ep);
                }
            }
        }
    }
    None
}
