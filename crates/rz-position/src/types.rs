use std::{error::Error, fmt, str::FromStr};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Color { White, Black }
impl Color {
    pub fn opposite(self) -> Self { match self { Self::White => Self::Black, Self::Black => Self::White } }
    pub(crate) fn index(self) -> usize { self as usize }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum PieceKind { Pawn, Knight, Bishop, Rook, Queen, King }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Piece { pub color: Color, pub kind: PieceKind }
impl Piece { pub(crate) fn index(self) -> usize { self.color.index() * 6 + self.kind as usize } }

/// Rank-major coordinates: a1=0, h8=63.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Square(pub(crate) u8);
impl Square {
    pub fn new(index: u8) -> Result<Self, PositionError> {
        if index < 64 { Ok(Self(index)) } else { Err(PositionError::InvalidMove("square outside 0..64")) }
    }
    pub fn index(self) -> u8 { self.0 }
    pub fn file(self) -> u8 { self.0 % 8 }
    pub fn rank(self) -> u8 { self.0 / 8 }
    pub(crate) fn offset(self, df: i8, dr: i8) -> Option<Self> {
        let f = self.file() as i8 + df;
        let r = self.rank() as i8 + dr;
        if (0..8).contains(&f) && (0..8).contains(&r) { Some(Self((r * 8 + f) as u8)) } else { None }
    }
}
impl FromStr for Square {
    type Err = PositionError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let b = s.as_bytes();
        if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
            return Err(PositionError::InvalidMove("invalid coordinate"));
        }
        Ok(Self((b[1] - b'1') * 8 + b[0] - b'a'))
    }
}
impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", (b'a' + self.file()) as char, (b'1' + self.rank()) as char)
    }
}

/// Position-local coordinate input, not a declaration of the shared Move ABI.
/// A checked transition establishes legality; constructing this value does not.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BoardMove { pub from: Square, pub to: Square, pub promotion: Option<PieceKind> }
impl BoardMove {
    pub fn new(from: Square, to: Square, promotion: Option<PieceKind>) -> Result<Self, PositionError> {
        if from == to || matches!(promotion, Some(PieceKind::Pawn | PieceKind::King)) {
            return Err(PositionError::InvalidMove("invalid move or promotion"));
        }
        Ok(Self { from, to, promotion })
    }
    pub fn from_uci(s: &str) -> Result<Self, PositionError> { s.parse() }
    pub(crate) fn sort_key(self) -> (u8, u8, u8) {
        let p = match self.promotion { None => 0, Some(PieceKind::Queen) => 1, Some(PieceKind::Rook) => 2, Some(PieceKind::Bishop) => 3, Some(PieceKind::Knight) => 4, _ => 5 };
        (self.from.0, self.to.0, p)
    }
}
impl FromStr for BoardMove {
    type Err = PositionError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if !s.is_ascii() || !(s.len() == 4 || s.len() == 5) { return Err(PositionError::InvalidMove("expected four coordinates and optional promotion")); }
        let p = if s.len() == 5 { Some(match s.as_bytes()[4] { b'q' => PieceKind::Queen, b'r' => PieceKind::Rook, b'b' => PieceKind::Bishop, b'n' => PieceKind::Knight, _ => return Err(PositionError::InvalidMove("invalid promotion suffix")) }) } else { None };
        Self::new(s[0..2].parse()?, s[2..4].parse()?, p)
    }
}
impl fmt::Display for BoardMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from, self.to)?;
        if let Some(p) = self.promotion { f.write_str(match p { PieceKind::Queen => "q", PieceKind::Rook => "r", PieceKind::Bishop => "b", PieceKind::Knight => "n", _ => "?" })?; }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PositionError {
    InvalidFen(&'static str), InvalidMove(&'static str), IllegalMove,
    CounterOverflow, RevisionExhausted, ResourceLimit(&'static str),
    UndoMismatch, StaleView,
}
impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFen(s) => write!(f, "invalid FEN: {s}"), Self::InvalidMove(s) => write!(f, "invalid move: {s}"),
            Self::IllegalMove => f.write_str("move is not legal in this state"), Self::CounterOverflow => f.write_str("rule counter overflow"),
            Self::RevisionExhausted => f.write_str("position revision exhausted"), Self::ResourceLimit(s) => write!(f, "position resource limit: {s}"),
            Self::UndoMismatch => f.write_str("undo token does not belong to this position and child"), Self::StaleView => f.write_str("legal view is stale or belongs to another position"),
        }
    }
}
impl Error for PositionError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PositionLimits { pub max_fen_bytes: usize, pub max_history_positions: usize, pub max_perft_depth: u32, pub max_perft_nodes: u64 }
impl Default for PositionLimits {
    fn default() -> Self { Self { max_fen_bytes: 4096, max_history_positions: 4096, max_perft_depth: 8, max_perft_nodes: 50_000_000 } }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistoryCompleteness { Complete, UnknownPrefix }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistoryOrigin { StartPosition, Fen }

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Board { pub(crate) squares: [Option<Piece>; 64], pub(crate) bitboards: [u64; 12] }
impl Board {
    pub(crate) fn empty() -> Self { Self { squares: [None; 64], bitboards: [0; 12] } }
    pub(crate) fn get(&self, s: Square) -> Option<Piece> { self.squares[s.0 as usize] }
    pub(crate) fn set(&mut self, s: Square, p: Option<Piece>) {
        let mask = 1u64 << s.0;
        if let Some(old) = self.get(s) { self.bitboards[old.index()] &= !mask; }
        self.squares[s.0 as usize] = p;
        if let Some(new) = p { self.bitboards[new.index()] |= mask; }
    }
    pub(crate) fn king(&self, c: Color) -> Square {
        Square(self.bitboards[c.index() * 6 + PieceKind::King as usize].trailing_zeros() as u8)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CoreState {
    pub(crate) board: Board, pub(crate) side: Color, pub(crate) castling: u8,
    pub(crate) ep: Option<Square>, pub(crate) halfmove: u32, pub(crate) fullmove: u32,
}
