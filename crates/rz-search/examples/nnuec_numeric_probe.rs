//! Explicit CPU numeric research caller. It never initializes weights, trains,
//! probes directories or selects an engine. Both input files are caller-owned.
//! Run with CHECKPOINT_JSON REQUEST_JSON; stdout contains the independent
//! reference input and the Rust result for that exact admitted checkpoint.

use rz_position::{BoardMove, Position};
use rz_search::cpu_value::nnuec::{NNUEC_MAX_BYTES, NnuecModel};
use serde::Deserialize;
use serde_json::json;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

const REQUEST_MAX_BYTES: usize = 256 * 1024;
const PREFIX_MAX_PLIES: usize = 4096;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PositionInput {
    fen: String,
    moves: Vec<String>,
}

impl PositionInput {
    fn position(&self) -> Result<Position, String> {
        if self.fen.len() > 128 || self.moves.len() > PREFIX_MAX_PLIES {
            return Err("Rules anchor exceeds declared input bound".into());
        }
        let mut position = Position::from_fen(&self.fen).map_err(|e| e.to_string())?;
        for movement in &self.moves {
            position
                .make_move(parse_move(movement)?)
                .map_err(|e| e.to_string())?;
        }
        Ok(position)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextLineInput {
    id: u64,
    anchor: PositionInput,
    moves: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    context_lines: Vec<ContextLineInput>,
    local_anchor: PositionInput,
    local_moves: Vec<String>,
}

fn parse_move(value: &str) -> Result<BoardMove, String> {
    if !(4..=5).contains(&value.len()) {
        return Err("move must be a bounded UCI coordinate".into());
    }
    value.parse::<BoardMove>().map_err(|e| e.to_string())
}

fn bounded_file(path: &Path, maximum: usize) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("caller must name a regular input file".into());
    }
    if metadata.len() > maximum as u64 {
        return Err("input file exceeds the registered byte limit".into());
    }
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| e.to_string())?
        .take((maximum + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > maximum {
        return Err("input file grew beyond the registered byte limit".into());
    }
    Ok(bytes)
}

fn run() -> Result<(), String> {
    let arguments: Vec<_> = std::env::args_os().collect();
    if arguments.len() != 3 {
        return Err("usage: nnuec_numeric_probe CHECKPOINT_JSON REQUEST_JSON".into());
    }
    let checkpoint = bounded_file(Path::new(&arguments[1]), NNUEC_MAX_BYTES)?;
    let model = NnuecModel::from_reader(checkpoint.as_slice()).map_err(|e| e.to_string())?;
    let request = bounded_file(Path::new(&arguments[2]), REQUEST_MAX_BYTES)?;
    let request: Request = serde_json::from_slice(&request).map_err(|e| e.to_string())?;
    let config = &model.checkpoint().config;
    if request.schema != "rz-nnuec-rules-probe/1"
        || request.context_lines.len() > config.max_lines
        || request.local_moves.len() > config.max_plies
    {
        return Err("probe schema or configured input capacity mismatch".into());
    }
    let mut context = model.empty_context();
    let mut ids = std::collections::BTreeSet::new();
    for input in request.context_lines {
        if !ids.insert(input.id) || input.moves.len() > config.max_plies {
            return Err("duplicate context ID or line length beyond configured capacity".into());
        }
        let moves = input
            .moves
            .iter()
            .map(|m| parse_move(m))
            .collect::<Result<Vec<_>, _>>()?;
        let line = model
            .encode_line(&input.anchor.position()?, &moves)
            .map_err(|e| e.to_string())?;
        context = model
            .with_line(&context, input.id, line)
            .map_err(|e| e.to_string())?;
    }
    // Use a canonical rebuild for the independent full-reference comparison.
    // Incremental-vs-rebuild and undo are separate Rust contract checks.
    let context = model.rebuild_context(&context).map_err(|e| e.to_string())?;
    let mut local = model
        .local(&request.local_anchor.position()?)
        .map_err(|e| e.to_string())?;
    for movement in request.local_moves {
        model
            .append_local(&mut local, parse_move(&movement)?)
            .map_err(|e| e.to_string())?;
    }
    let board = model.board(local.position()).map_err(|e| e.to_string())?;
    let reference_input = model
        .reference_input(&context, &local, local.position())
        .map_err(|e| e.to_string())?;
    let output = model
        .forward(&board, &context, &local, local.position())
        .map_err(|e| e.to_string())?;
    let record = json!({
        "schema": "rz-nnuec-rules-probe-result/1",
        "scope": "cpu_numeric_only_not_training_or_strength_evidence",
        "training": model.identity().training,
        "reference_input": reference_input,
        "output": output,
    });
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &record).map_err(|e| e.to_string())?;
    stdout.write_all(b"\n").map_err(|e| e.to_string())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("NNUEC numeric probe failed: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
