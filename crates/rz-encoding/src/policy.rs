//! Classical LC0/Maia 1858-action format, generated from geometry.
//!
//! Enumerate from squares, then to squares in ascending a1=0 order, accepting
//! queen rays and knight displacements (1792 slots). Append rank-seven to
//! rank-eight pawn destinations by from file, to file, and Q/R/B (66 slots).
//! Knight promotions reuse their ordinary from/to slot. No external mapping
//! table is embedded, and geometric representability never establishes legality.

use crate::POLICY_SIZE;
use std::fmt;
use std::sync::OnceLock;

const BASE_SIZE: usize = 1792;
const INVALID: u16 = u16::MAX;

/// A model slot descriptor, not a Rules move. A base slot cannot distinguish a
/// knight promotion from an ordinary move without the checked legal move view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicySlot {
    pub from: u8,
    pub to: u8,
    pub promotion: Option<char>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError {
    InvalidSquare(u8),
    UnsupportedPromotion(char),
    InvalidPromotionGeometry,
    UnrepresentableMove,
    InvalidCastlingGeometry,
    InvalidIndex(usize),
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Maia policy encoding: {self:?}")
    }
}

impl std::error::Error for PolicyError {}

struct ActionMap {
    base: [u16; 64 * 64],
    slots: [PolicySlot; POLICY_SIZE],
}

fn action_map() -> &'static ActionMap {
    static MAP: OnceLock<ActionMap> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = ActionMap {
            base: [INVALID; 64 * 64],
            slots: [PolicySlot {
                from: 0,
                to: 0,
                promotion: None,
            }; POLICY_SIZE],
        };
        let mut index = 0;
        for from in 0u8..64 {
            for to in 0u8..64 {
                let df = (from % 8).abs_diff(to % 8);
                let dr = (from / 8).abs_diff(to / 8);
                let ray = df == 0 || dr == 0 || df == dr;
                let knight = (df == 1 && dr == 2) || (df == 2 && dr == 1);
                if from != to && (ray || knight) {
                    map.base[from as usize * 64 + to as usize] = index as u16;
                    map.slots[index] = PolicySlot {
                        from,
                        to,
                        promotion: None,
                    };
                    index += 1;
                }
            }
        }
        assert_eq!(index, BASE_SIZE, "classical geometric action count");
        for file in 0u8..8 {
            for destination in file.saturating_sub(1)..=(file + 1).min(7) {
                for piece in ['q', 'r', 'b'] {
                    map.slots[index] = PolicySlot {
                        from: 48 + file,
                        to: 56 + destination,
                        promotion: Some(piece),
                    };
                    index += 1;
                }
            }
        }
        assert_eq!(index, POLICY_SIZE, "classical complete action count");
        map
    })
}

/// Rotate ranks into the current side's viewpoint; files are unchanged.
pub fn canonical_square(square: u8, black_to_move: bool) -> Result<u8, PolicyError> {
    if square >= 64 {
        return Err(PolicyError::InvalidSquare(square));
    }
    Ok(if black_to_move { square ^ 56 } else { square })
}

/// Map an already-canonical model action. The caller supplies a checked legal
/// move and retains its promotion kind and request order separately.
pub fn index(from: u8, to: u8, promotion: Option<char>) -> Result<usize, PolicyError> {
    canonical_square(from, false)?;
    canonical_square(to, false)?;
    if let Some(piece) = promotion {
        let offset = match piece {
            'q' => Some(0),
            'r' => Some(1),
            'b' => Some(2),
            'n' => None,
            other => return Err(PolicyError::UnsupportedPromotion(other)),
        };
        if from / 8 != 6 || to / 8 != 7 || (from % 8).abs_diff(to % 8) > 1 {
            return Err(PolicyError::InvalidPromotionGeometry);
        }
        if let Some(piece_offset) = offset {
            let file = from as usize % 8;
            let before = if file == 0 { 0 } else { 2 + 3 * (file - 1) };
            let destination = to as usize % 8 - file.saturating_sub(1);
            return Ok(BASE_SIZE + (before + destination) * 3 + piece_offset);
        }
    }
    let result = action_map().base[from as usize * 64 + to as usize];
    if result == INVALID {
        Err(PolicyError::UnrepresentableMove)
    } else {
        Ok(result as usize)
    }
}

/// Standard-chess castling is encoded king-to-rook, not king-to-destination.
/// `king` and `rook` are already canonical squares; legality remains with Rules.
pub fn castling_index(king: u8, rook: u8) -> Result<usize, PolicyError> {
    canonical_square(king, false)?;
    canonical_square(rook, false)?;
    if king != 4 || (rook != 0 && rook != 7) {
        return Err(PolicyError::InvalidCastlingGeometry);
    }
    index(king, rook, None)
}

pub fn slot(index: usize) -> Result<PolicySlot, PolicyError> {
    action_map()
        .slots
        .get(index)
        .copied()
        .ok_or(PolicyError::InvalidIndex(index))
}

pub fn slots() -> &'static [PolicySlot; POLICY_SIZE] {
    &action_map().slots
}
