//! Classical 112-plane model projection from immutable, absolute bitboards.
//!
//! This module does not parse FEN, generate moves, classify positions, or invent
//! Rules history. The eventual A03 adapter supplies checked bitboards and the
//! repetition-plane value from known history. Padding is only a model policy.

use std::fmt;

pub const HISTORY_FRAMES: usize = 8;
pub const PLANES: usize = 112;
pub const INPUT_VALUES: usize = PLANES * 64;

/// Bit order: a1=0..h8=63. Color order: white, black. Piece order:
/// pawn, knight, bishop, rook, queen, king. This is a model projection, not State.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    pub pieces: [[u64; 6]; 2],
    pub repeated: bool,
    /// Original checked EP target in absolute coordinates, even if no capture
    /// is legal. Used only for the model's limited missing-history reconstruction.
    pub en_passant_target: Option<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoryFill {
    /// LC0 FillEmptyHistory::NO: zeros beyond known history, except one
    /// predecessor inferable from the oldest frame's EP target.
    No,
    /// LC0 FillEmptyHistory::ALWAYS: repeat the oldest frame (or its EP
    /// predecessor) into remaining model slots. Does not create real history.
    RepeatOldest,
}

impl HistoryFill {
    pub fn reference_option(self) -> &'static str {
        match self {
            Self::No => "no",
            Self::RepeatOldest => "always",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Input<'a> {
    /// Current first, then known predecessors. Supply at most the eight frames
    /// actually consumed by this model; retain full Rules history in its owner.
    pub history: &'a [Frame],
    pub black_to_move: bool,
    /// Absolute white queenside/kingside, black queenside/kingside rights.
    pub castling: [bool; 4],
    pub halfmove_clock: u32,
    pub history_fill: HistoryFill,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncodedInput {
    values: Vec<f32>,
    pub known_frames: usize,
    pub padded_frames: usize,
    pub inferred_ep_predecessor: bool,
    pub history_fill: HistoryFill,
}

impl EncodedInput {
    /// Contiguous NCHW values for one [112, 8, 8] sample.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassicalError {
    EmptyHistory,
    TooManyFrames(usize),
    OverlappingPieces { frame: usize },
    InvalidEnPassant { frame: usize },
    AllocationFailed,
}

impl fmt::Display for ClassicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "classical input encoding: {self:?}")
    }
}

impl std::error::Error for ClassicalError {}

pub fn encode(input: Input<'_>) -> Result<EncodedInput, ClassicalError> {
    if input.history.is_empty() {
        return Err(ClassicalError::EmptyHistory);
    }
    if input.history.len() > HISTORY_FRAMES {
        return Err(ClassicalError::TooManyFrames(input.history.len()));
    }
    for (index, frame) in input.history.iter().enumerate() {
        let mut occupied = 0;
        for &mask in frame.pieces.iter().flatten() {
            if occupied & mask != 0 {
                return Err(ClassicalError::OverlappingPieces { frame: index });
            }
            occupied |= mask;
        }
        if frame.en_passant_target.is_some() {
            ep_predecessor(*frame).ok_or(ClassicalError::InvalidEnPassant { frame: index })?;
        }
    }

    let mut values = Vec::new();
    values
        .try_reserve_exact(INPUT_VALUES)
        .map_err(|_| ClassicalError::AllocationFailed)?;
    values.resize(INPUT_VALUES, 0.0);
    for (index, frame) in input.history.iter().enumerate() {
        write_frame(&mut values, index, frame, input.black_to_move);
    }

    let oldest = *input
        .history
        .last()
        .expect("nonempty history checked above");
    let inferred = oldest.en_passant_target.is_some();
    let padding = if inferred {
        ep_predecessor(oldest).expect("EP projection checked above")
    } else {
        oldest
    };
    let remaining = HISTORY_FRAMES - input.history.len();
    let padded_frames = match input.history_fill {
        HistoryFill::No => usize::from(inferred).min(remaining),
        HistoryFill::RepeatOldest => remaining,
    };
    for index in input.history.len()..input.history.len() + padded_frames {
        write_frame(&mut values, index, &padding, input.black_to_move);
    }

    let own = if input.black_to_move { 2 } else { 0 };
    let other = 2 - own;
    for (plane, right) in [own, own + 1, other, other + 1].into_iter().enumerate() {
        if input.castling[right] {
            fill_plane(&mut values, 104 + plane, 1.0);
        }
    }
    fill_plane(&mut values, 108, f32::from(input.black_to_move));
    // Classical input uses the original ply counter, not a /100 normalization.
    fill_plane(&mut values, 109, input.halfmove_clock as f32);
    fill_plane(&mut values, 111, 1.0);
    Ok(EncodedInput {
        values,
        known_frames: input.history.len(),
        padded_frames,
        inferred_ep_predecessor: inferred && padded_frames != 0,
        history_fill: input.history_fill,
    })
}

fn write_frame(output: &mut [f32], index: usize, frame: &Frame, black: bool) {
    let own = usize::from(black);
    for (relative_color, absolute_color) in [own, 1 - own].into_iter().enumerate() {
        for (piece, &original_mask) in frame.pieces[absolute_color].iter().enumerate() {
            let mut mask = original_mask;
            while mask != 0 {
                let absolute = mask.trailing_zeros() as usize;
                let square = if black { absolute ^ 56 } else { absolute };
                let plane = index * 13 + relative_color * 6 + piece;
                output[plane * 64 + square] = 1.0;
                mask &= mask - 1;
            }
        }
    }
    if frame.repeated {
        fill_plane(output, index * 13 + 12, 1.0);
    }
}

fn fill_plane(output: &mut [f32], plane: usize, value: f32) {
    output[plane * 64..(plane + 1) * 64].fill(value);
}

/// Model-only inverse of the one double pawn push identified by an EP target.
/// No legal transition, repetition count, or Rules history is manufactured.
fn ep_predecessor(mut frame: Frame) -> Option<Frame> {
    let target = frame.en_passant_target?;
    let (color, current, previous) = match target / 8 {
        2 => (0, target + 8, target - 8),
        5 => (1, target - 8, target + 8),
        _ => return None,
    };
    let occupied = frame
        .pieces
        .iter()
        .flatten()
        .fold(0, |all, &mask| all | mask);
    let current_mask = 1u64 << current;
    let previous_mask = 1u64 << previous;
    if frame.pieces[color][0] & current_mask == 0
        || occupied & ((1u64 << target) | previous_mask) != 0
    {
        return None;
    }
    frame.pieces[color][0] = (frame.pieces[color][0] & !current_mask) | previous_mask;
    frame.en_passant_target = None;
    Some(frame)
}
