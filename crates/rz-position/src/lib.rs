//! Independent standard-chess state, checked transitions and immutable views.
//!
//! This crate owns concrete rule state. Shared evaluator identifiers and request
//! types remain owned by `rz-contracts`; no backend or search types live here.
//!
//! ```
//! use rz_position::{BoardMove, Position, PositionError};
//! let mut position = Position::startpos();
//! let before = position.snapshot();
//! let legal = position.ordered_legal_moves();
//! let undo = position.make_from_view(&legal, BoardMove::from_uci("e2e4")?)?;
//! assert_eq!(before.side_to_move(), rz_position::Color::White);
//! assert_eq!(position.side_to_move(), rz_position::Color::Black);
//! position.unmake(undo)?;
//! assert!(before.same_state(&position.snapshot()));
//! assert!(!position.matches_snapshot(&before)); // observer revision advances
//! # Ok::<(), PositionError>(())
//! ```
#![forbid(unsafe_code)]

mod fen;
mod movegen;
mod outcome;
mod position;
mod types;

pub use outcome::*;
pub use position::*;
pub use types::*;
