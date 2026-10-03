//! B-owned UCI text and session control. Rules and Runtime are injected adapters.
//!
//! This crate deliberately does not define shared evaluation IDs or implement
//! chess rules. A [`PositionPort`] must validate complete move traces and supply
//! the root's ordered legal moves before a [`Session`] can start searching.

#![forbid(unsafe_code)]

pub mod bootstrap;
pub mod bridge;
pub mod contracts;
pub mod engine;
#[cfg(feature = "onnx-cpu")]
pub mod native_attestation;
#[cfg(feature = "onnx-cpu")]
pub mod native_bootstrap;
#[cfg(feature = "onnx-cuda")]
pub mod native_cuda_attestation;
pub mod parser;
pub mod runner;
pub mod session;

pub use parser::{Command, GoLimits, ParseError, ParserLimits, PositionBase, PositionSpec, parse};
pub use runner::{
    Event, ServeError, ServeFailure, forward_lines, handle_event, serve_events,
    serve_events_with_handler,
};
pub use session::{
    CancelReason, Diagnostic, Effect, EngineIdentity, OptionKind, OptionSpec, OptionValue,
    PositionPort, PreparedPosition, SearchCompletion, SearchTicket, Session, SessionError,
    SessionResult,
};
