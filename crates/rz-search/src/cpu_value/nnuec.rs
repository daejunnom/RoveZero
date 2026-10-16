//! CPU-only context-aware, efficiently updatable FP32 reference.
//!
//! B is an owned two-perspective board accumulator, C is an immutable shared
//! line set, and L is an owned, Rules-checked local suffix. No search, runtime,
//! Python, GPU, or global mutable context is required to use this module.
//! The dimensions are required experimental configuration, not adopted defaults.
//! Arithmetic uses separate scalar FP32 multiplication and addition. B/C delta
//! results are compared to full rebuild with abs 1e-4 + rel 1e-3; undo restores
//! the original FP32 bits. There is no clipping before an accumulator's sum.

use super::{
    CPU_FEATURE_COUNT, CpuTrainingState, CpuValueIdentity, metadata_features, piece_feature,
    validate_training_declaration,
};
use rz_position::{
    BoardMove, Color, PieceKind, Position, PositionIdentity, RuleMoveDelta, Square, UndoToken,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;

pub const NNUEC_SCHEMA: &str = "rz-nnuec-fp32-v1";
pub const NNUEC_SEMANTICS: &str = "board800-causal-line-bcl-fp32-v1";
pub const NNUEC_INPUT_SCHEMA: &str = "rz-nnuec-fp32-input/1";
pub const NNUEC_OUTPUT_SCHEMA: &str = "rz-nnuec-fp32-output/1";
pub const NNUEC_MAX_BYTES: usize = 32 * 1024 * 1024;
pub const NNUEC_OUTPUTS: usize = 11;
pub const NNUEC_ABSOLUTE_TOLERANCE: f32 = 1e-4;
pub const NNUEC_RELATIVE_TOLERANCE: f32 = 1e-3;
const PARAMETER_MAGNITUDE: f32 = 1000.0;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NnuecError {
    InvalidConfig(&'static str),
    InvalidCheckpoint(&'static str),
    Capacity(&'static str),
    StateMismatch,
    NonFinite,
    Rules(String),
    Io(String),
    Json(String),
}

impl std::fmt::Display for NnuecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(s) => write!(f, "NNUEC config: {s}"),
            Self::InvalidCheckpoint(s) => write!(f, "NNUEC checkpoint: {s}"),
            Self::Capacity(s) => write!(f, "NNUEC capacity: {s}"),
            Self::StateMismatch => f.write_str("NNUEC owner/model/Rules state mismatch"),
            Self::NonFinite => f.write_str("NNUEC FP32 operation produced a non-finite result"),
            Self::Rules(s) => write!(f, "NNUEC Rules transition: {s}"),
            Self::Io(s) => write!(f, "NNUEC checkpoint read: {s}"),
            Self::Json(s) => write!(f, "NNUEC JSON: {s}"),
        }
    }
}
impl std::error::Error for NnuecError {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnuecConfig {
    pub board_width: usize,
    pub line_width: usize,
    pub context_width: usize,
    pub local_width: usize,
    pub hidden_width: usize,
    pub causal_window: usize,
    pub max_lines: usize,
    pub max_plies: usize,
}

impl NnuecConfig {
    pub fn validate(&self) -> Result<(), NnuecError> {
        if [
            self.board_width,
            self.line_width,
            self.context_width,
            self.local_width,
        ]
        .into_iter()
        .any(|n| !(1..=256).contains(&n))
            || !(1..=64).contains(&self.hidden_width)
            || ![3, 5].contains(&self.causal_window)
            || !(1..=128).contains(&self.max_lines)
            || !(1..=256).contains(&self.max_plies)
        {
            return Err(NnuecError::InvalidConfig(
                "unsupported dimension or capacity",
            ));
        }
        Ok(())
    }

    pub fn input_width(&self) -> usize {
        2 * self.board_width + self.context_width + self.local_width
    }

    fn dimensions(&self) -> [usize; 8] {
        [
            self.board_width,
            self.line_width,
            self.context_width,
            self.local_width,
            self.hidden_width,
            self.causal_window,
            self.max_lines,
            self.max_plies,
        ]
    }
}

/// Flat row-major FP32 parameters. Embedding tables have one row per feature;
/// dense projections have one row per output. No implicit initializer exists.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnuecParameters {
    pub board_weights: Vec<f32>,
    pub board_bias: Vec<f32>,
    pub anchor_weights: Vec<f32>,
    pub anchor_bias: Vec<f32>,
    pub from_weights: Vec<f32>,
    pub to_weights: Vec<f32>,
    pub promotion_weights: Vec<f32>,
    pub ply_weights: Vec<f32>,
    pub token_bias: Vec<f32>,
    /// Diagonal causal convolution, lag 0 is the current token.
    pub causal_weights: Vec<f32>,
    pub causal_bias: Vec<f32>,
    pub context_projection: Vec<f32>,
    pub context_bias: Vec<f32>,
    pub local_projection: Vec<f32>,
    pub local_bias: Vec<f32>,
    pub hidden_weights: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    /// Rows: value, W/D/L, state candidate hint, retain/drop/revisit, P/C/R.
    pub output_weights: Vec<f32>,
    pub output_bias: Vec<f32>,
}

impl NnuecParameters {
    fn arrays(&self) -> [&Vec<f32>; 19] {
        [
            &self.board_weights,
            &self.board_bias,
            &self.anchor_weights,
            &self.anchor_bias,
            &self.from_weights,
            &self.to_weights,
            &self.promotion_weights,
            &self.ply_weights,
            &self.token_bias,
            &self.causal_weights,
            &self.causal_bias,
            &self.context_projection,
            &self.context_bias,
            &self.local_projection,
            &self.local_bias,
            &self.hidden_weights,
            &self.hidden_bias,
            &self.output_weights,
            &self.output_bias,
        ]
    }

    fn validate(&self, c: &NnuecConfig) -> Result<(), NnuecError> {
        let e = c.line_width;
        let expected = [
            CPU_FEATURE_COUNT * c.board_width,
            c.board_width,
            CPU_FEATURE_COUNT * e,
            e,
            64 * e,
            64 * e,
            5 * e,
            c.max_plies * e,
            e,
            c.causal_window * e,
            e,
            c.context_width * e,
            c.context_width,
            c.local_width * e,
            c.local_width,
            c.hidden_width * c.input_width(),
            c.hidden_width,
            NNUEC_OUTPUTS * c.hidden_width,
            NNUEC_OUTPUTS,
        ];
        let mut allocated = 0usize;
        for (values, expected) in self.arrays().into_iter().zip(expected) {
            if values.len() != expected || values.capacity() > expected.saturating_mul(2).max(4) {
                return Err(NnuecError::InvalidCheckpoint(
                    "parameter shape/allocation mismatch",
                ));
            }
            allocated = allocated
                .checked_add(values.capacity() * 4)
                .ok_or(NnuecError::Capacity("parameter bytes"))?;
            if values
                .iter()
                .any(|v| !v.is_finite() || v.abs() > PARAMETER_MAGNITUDE)
            {
                return Err(NnuecError::InvalidCheckpoint(
                    "non-finite/excessive parameter",
                ));
            }
        }
        if allocated > NNUEC_MAX_BYTES {
            return Err(NnuecError::Capacity("parameter bytes"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnuecCheckpoint {
    pub schema: String,
    pub semantics: String,
    pub config: NnuecConfig,
    pub training: CpuTrainingState,
    pub parameters: NnuecParameters,
    pub output_scale: f32,
}

impl NnuecCheckpoint {
    pub fn validate(&self) -> Result<(), NnuecError> {
        self.config.validate()?;
        if self.schema != NNUEC_SCHEMA
            || self.semantics != NNUEC_SEMANTICS
            || self.schema.capacity() > 256
            || self.semantics.capacity() > 256
            || matches!(self.training, CpuTrainingState::Bootstrap)
            || validate_training_declaration(&self.training).is_err()
            || !self.output_scale.is_finite()
            || self.output_scale <= 0.0
            || self.output_scale > 10_000.0
        {
            return Err(NnuecError::InvalidCheckpoint(
                "schema/semantics/training/scale",
            ));
        }
        self.parameters.validate(&self.config)
    }

    /// Binary domain, required dimensions (u32 LE), ordered FP32 parameter bits,
    /// then scale bits. JSON formatting and training declarations are excluded.
    pub fn weights_sha256(&self) -> Result<String, NnuecError> {
        self.validate()?;
        let mut hash = Sha256::new();
        hash.update(NNUEC_SCHEMA.as_bytes());
        hash.update([0]);
        hash.update(NNUEC_SEMANTICS.as_bytes());
        hash.update([0]);
        for n in self.config.dimensions() {
            hash.update((n as u32).to_le_bytes());
        }
        for values in self.parameters.arrays() {
            for v in values {
                hash.update(v.to_bits().to_le_bytes());
            }
        }
        hash.update(self.output_scale.to_bits().to_le_bytes());
        Ok(format!("{:x}", hash.finalize()))
    }
}

pub struct NnuecModel {
    checkpoint: Arc<NnuecCheckpoint>,
    identity: Arc<CpuValueIdentity>,
}

impl NnuecModel {
    pub fn new(checkpoint: NnuecCheckpoint) -> Result<Self, NnuecError> {
        let weights = checkpoint.weights_sha256()?;
        let identity = Arc::new(CpuValueIdentity {
            semantics: NNUEC_SEMANTICS.to_owned(),
            weights_sha256: Some(weights),
            training: checkpoint.training.clone(),
        });
        Ok(Self {
            checkpoint: Arc::new(checkpoint),
            identity,
        })
    }

    pub fn from_reader(reader: impl Read) -> Result<Self, NnuecError> {
        let mut bytes = Vec::new();
        reader
            .take((NNUEC_MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| NnuecError::Io(e.to_string()))?;
        if bytes.len() > NNUEC_MAX_BYTES {
            return Err(NnuecError::Capacity("checkpoint bytes"));
        }
        let checkpoint =
            serde_json::from_slice(&bytes).map_err(|e| NnuecError::Json(e.to_string()))?;
        Self::new(checkpoint)
    }

    pub fn identity(&self) -> &CpuValueIdentity {
        &self.identity
    }
    pub fn checkpoint(&self) -> &NnuecCheckpoint {
        &self.checkpoint
    }

    pub fn board(&self, position: &Position) -> Result<NnuecBoard, NnuecError> {
        let p = &self.checkpoint.parameters;
        let mut values = [p.board_bias.clone(), p.board_bias.clone()];
        for side in [Color::White, Color::Black] {
            for feature in board_features(position, side) {
                add_row(&mut values[side as usize], &p.board_weights, feature, 1.0)?;
            }
        }
        Ok(NnuecBoard {
            values,
            position: position.position_identity(),
            model: Arc::clone(&self.identity),
            stamp: Arc::new(()),
        })
    }

    pub fn apply_board_delta(
        &self,
        board: &mut NnuecBoard,
        delta: &RuleMoveDelta,
    ) -> Result<NnuecBoardUndo, NnuecError> {
        if board.model != self.identity || board.position != delta.before.position_identity() {
            return Err(NnuecError::StateMismatch);
        }
        if delta.removals.len() > 4 || delta.additions.len() > 4 {
            return Err(NnuecError::Capacity("Rules piece delta"));
        }
        let mut next = board.values.clone();
        for side in [Color::White, Color::Black] {
            let out = &mut next[side as usize];
            for r in &delta.removals {
                add_row(
                    out,
                    &self.checkpoint.parameters.board_weights,
                    piece_feature(r.piece, r.square, side),
                    -1.0,
                )?;
            }
            for a in &delta.additions {
                add_row(
                    out,
                    &self.checkpoint.parameters.board_weights,
                    piece_feature(a.piece, a.square, side),
                    1.0,
                )?;
            }
            let before = metadata_features(&delta.before, side);
            let after = metadata_features(&delta.after, side);
            for f in before.as_slice() {
                if !after.as_slice().contains(f) {
                    add_row(out, &self.checkpoint.parameters.board_weights, *f, -1.0)?;
                }
            }
            for f in after.as_slice() {
                if !before.as_slice().contains(f) {
                    add_row(out, &self.checkpoint.parameters.board_weights, *f, 1.0)?;
                }
            }
        }
        let after = delta.after.position_identity();
        let stamp = Arc::new(());
        let undo = NnuecBoardUndo {
            values: std::mem::replace(&mut board.values, next),
            before: board.position.clone(),
            after: after.clone(),
            model: Arc::clone(&self.identity),
            before_stamp: Arc::clone(&board.stamp),
            after_stamp: Arc::clone(&stamp),
        };
        board.position = after;
        board.stamp = stamp;
        Ok(undo)
    }

    pub fn restore_board(
        &self,
        board: &mut NnuecBoard,
        undo: NnuecBoardUndo,
        position: &Position,
    ) -> Result<(), NnuecError> {
        if board.model != self.identity
            || undo.model != self.identity
            || board.position != undo.after
            || position.position_identity() != undo.before
            || !Arc::ptr_eq(&board.stamp, &undo.after_stamp)
        {
            return Err(NnuecError::StateMismatch);
        }
        board.values = undo.values;
        board.position = undo.before;
        board.stamp = undo.before_stamp;
        Ok(())
    }

    pub fn local(&self, anchor: &Position) -> Result<NnuecLocal, NnuecError> {
        let side = anchor.side_to_move();
        let features = board_features(anchor, side);
        let mut anchor_values = self.checkpoint.parameters.anchor_bias.clone();
        for feature in &features {
            add_row(
                &mut anchor_values,
                &self.checkpoint.parameters.anchor_weights,
                *feature,
                1.0,
            )?;
        }
        Ok(NnuecLocal {
            work: anchor.clone(),
            anchor: anchor.position_identity(),
            perspective: side,
            anchor_features: features,
            anchor_values,
            tokens: Vec::new(),
            moves: Vec::new(),
            embedding: vec![0.0; self.checkpoint.config.line_width],
            projected: self.checkpoint.parameters.local_bias.clone(),
            model: Arc::clone(&self.identity),
            stamp: Arc::new(()),
        })
    }

    /// Appending validates a genuine Rules transition. Only a bounded causal
    /// window and local accumulator change; C and the global store are untouched.
    pub fn append_local(
        &self,
        local: &mut NnuecLocal,
        mv: BoardMove,
    ) -> Result<NnuecLocalUndo, NnuecError> {
        if local.model != self.identity {
            return Err(NnuecError::StateMismatch);
        }
        let c = &self.checkpoint.config;
        if local.moves.len() >= c.max_plies {
            return Err(NnuecError::Capacity("local plies"));
        }
        let token = self.token(
            &local.anchor_values,
            local.perspective,
            mv,
            local.moves.len(),
        )?;
        let mut embedding = local.embedding.clone();
        for (j, out) in embedding.iter_mut().enumerate() {
            let mut value = self.checkpoint.parameters.causal_bias[j];
            for lag in 0..c.causal_window {
                let input = if lag == 0 {
                    token[j]
                } else if lag <= local.tokens.len() {
                    local.tokens[local.tokens.len() - lag][j]
                } else {
                    continue;
                };
                value = add(
                    value,
                    mul(
                        input,
                        self.checkpoint.parameters.causal_weights[lag * c.line_width + j],
                    )?,
                )?;
            }
            *out = add(*out, value.max(0.0))?;
        }
        let projected = project(
            &embedding,
            &self.checkpoint.parameters.local_projection,
            &self.checkpoint.parameters.local_bias,
        )?;
        local
            .tokens
            .try_reserve(1)
            .map_err(|_| NnuecError::Capacity("local token allocation"))?;
        local
            .moves
            .try_reserve(1)
            .map_err(|_| NnuecError::Capacity("local move allocation"))?;
        let rules = local
            .work
            .make_move(mv)
            .map_err(|e| NnuecError::Rules(e.to_string()))?;
        let stamp = Arc::new(());
        let undo = NnuecLocalUndo {
            rules,
            previous_embedding: std::mem::replace(&mut local.embedding, embedding),
            previous_projected: std::mem::replace(&mut local.projected, projected),
            model: Arc::clone(&self.identity),
            before_stamp: Arc::clone(&local.stamp),
            after_stamp: Arc::clone(&stamp),
            expected_length: local.moves.len() + 1,
        };
        local.tokens.push(token);
        local.moves.push(mv);
        local.stamp = stamp;
        Ok(undo)
    }

    pub fn undo_local(
        &self,
        local: &mut NnuecLocal,
        undo: NnuecLocalUndo,
    ) -> Result<(), NnuecError> {
        if local.model != self.identity
            || undo.model != self.identity
            || local.moves.len() != undo.expected_length
            || !Arc::ptr_eq(&local.stamp, &undo.after_stamp)
        {
            return Err(NnuecError::StateMismatch);
        }
        local
            .work
            .unmake(undo.rules)
            .map_err(|e| NnuecError::Rules(e.to_string()))?;
        local.embedding = undo.previous_embedding;
        local.projected = undo.previous_projected;
        local.tokens.pop();
        local.moves.pop();
        local.stamp = undo.before_stamp;
        Ok(())
    }

    /// Re-anchoring or deleting a middle move requires a genuine rebuild.
    pub fn encode_line(
        &self,
        anchor: &Position,
        moves: &[BoardMove],
    ) -> Result<Arc<NnuecLine>, NnuecError> {
        if moves.len() > self.checkpoint.config.max_plies {
            return Err(NnuecError::Capacity("line plies"));
        }
        let mut local = self.local(anchor)?;
        for mv in moves {
            self.append_local(&mut local, *mv)?;
        }
        Ok(Arc::new(NnuecLine {
            model: Arc::clone(&self.identity),
            anchor: local.anchor,
            input: ReferenceLine {
                anchor_features: local.anchor_features,
                perspective: color_name(local.perspective).into(),
                moves: local
                    .moves
                    .iter()
                    .copied()
                    .map(ReferenceMove::from_board)
                    .collect(),
            },
            embedding: local.embedding,
        }))
    }

    fn token(
        &self,
        anchor: &[f32],
        side: Color,
        mv: BoardMove,
        ply: usize,
    ) -> Result<Vec<f32>, NnuecError> {
        let promotion = promotion_index(mv)?;
        let p = &self.checkpoint.parameters;
        let width = self.checkpoint.config.line_width;
        let from = oriented_square(mv.from, side);
        let to = oriented_square(mv.to, side);
        let mut token = p.token_bias.clone();
        for (j, out) in token.iter_mut().enumerate() {
            *out = add(*out, anchor[j])?;
            *out = add(*out, p.from_weights[from * width + j])?;
            *out = add(*out, p.to_weights[to * width + j])?;
            *out = add(*out, p.promotion_weights[promotion * width + j])?;
            *out = add(*out, p.ply_weights[ply * width + j])?;
        }
        Ok(token)
    }

    pub fn empty_context(&self) -> Arc<NnuecContext> {
        Arc::new(NnuecContext {
            model: Arc::clone(&self.identity),
            lines: BTreeMap::new(),
            values: self.checkpoint.parameters.context_bias.clone(),
        })
    }

    /// IDs are ownership keys only. They never enter an embedding or a head.
    /// Returning a new Arc keeps all existing workers' frozen C unchanged.
    pub fn with_line(
        &self,
        context: &Arc<NnuecContext>,
        id: u64,
        line: Arc<NnuecLine>,
    ) -> Result<Arc<NnuecContext>, NnuecError> {
        if context.model != self.identity || line.model != self.identity {
            return Err(NnuecError::StateMismatch);
        }
        if !context.lines.contains_key(&id)
            && context.lines.len() >= self.checkpoint.config.max_lines
        {
            return Err(NnuecError::Capacity("context lines"));
        }
        let mut values = context.values.clone();
        if let Some(old) = context.lines.get(&id) {
            self.add_line(&mut values, old, -1.0)?;
        }
        self.add_line(&mut values, &line, 1.0)?;
        let mut lines = context.lines.clone();
        lines.insert(id, line);
        Ok(Arc::new(NnuecContext {
            model: Arc::clone(&self.identity),
            lines,
            values,
        }))
    }

    pub fn without_line(
        &self,
        context: &Arc<NnuecContext>,
        id: u64,
    ) -> Result<Arc<NnuecContext>, NnuecError> {
        if context.model != self.identity {
            return Err(NnuecError::StateMismatch);
        }
        let old = context.lines.get(&id).ok_or(NnuecError::StateMismatch)?;
        let mut values = context.values.clone();
        self.add_line(&mut values, old, -1.0)?;
        let mut lines = context.lines.clone();
        lines.remove(&id);
        Ok(Arc::new(NnuecContext {
            model: Arc::clone(&self.identity),
            lines,
            values,
        }))
    }

    pub fn rebuild_context(&self, context: &NnuecContext) -> Result<Arc<NnuecContext>, NnuecError> {
        if context.model != self.identity {
            return Err(NnuecError::StateMismatch);
        }
        let mut values = self.checkpoint.parameters.context_bias.clone();
        for line in context.lines.values() {
            self.add_line(&mut values, line, 1.0)?;
        }
        Ok(Arc::new(NnuecContext {
            model: Arc::clone(&self.identity),
            lines: context.lines.clone(),
            values,
        }))
    }

    fn add_line(&self, values: &mut [f32], line: &NnuecLine, sign: f32) -> Result<(), NnuecError> {
        let contribution = project(
            &line.embedding,
            &self.checkpoint.parameters.context_projection,
            &vec![0.0; values.len()],
        )?;
        for (value, contribution) in values.iter_mut().zip(contribution) {
            *value = add(*value, mul(sign, contribution)?)?;
        }
        Ok(())
    }

    pub fn forward(
        &self,
        board: &NnuecBoard,
        context: &NnuecContext,
        local: &NnuecLocal,
        position: &Position,
    ) -> Result<NnuecOutput, NnuecError> {
        if board.model != self.identity
            || context.model != self.identity
            || local.model != self.identity
            || board.position != position.position_identity()
            || local.work.position_identity() != board.position
        {
            return Err(NnuecError::StateMismatch);
        }
        let side = position.side_to_move() as usize;
        let mut input = Vec::with_capacity(self.checkpoint.config.input_width());
        input.extend(
            board.values[side]
                .iter()
                .chain(&board.values[1 - side])
                .chain(&context.values)
                .chain(&local.projected)
                .map(|v| v.clamp(0.0, 1.0)),
        );
        let mut hidden = project(
            &input,
            &self.checkpoint.parameters.hidden_weights,
            &self.checkpoint.parameters.hidden_bias,
        )?;
        for v in &mut hidden {
            *v = v.max(0.0);
        }
        let raw = project(
            &hidden,
            &self.checkpoint.parameters.output_weights,
            &self.checkpoint.parameters.output_bias,
        )?;
        let raw_outputs: [f32; NNUEC_OUTPUTS] =
            raw.try_into().map_err(|_| NnuecError::StateMismatch)?;
        let value = mul(raw_outputs[0], self.checkpoint.output_scale)?;
        let wdl_logits = [raw_outputs[1], raw_outputs[2], raw_outputs[3]];
        let max = wdl_logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
        let mut wdl = wdl_logits.map(|v| (v - max).exp());
        let sum = add(add(wdl[0], wdl[1])?, wdl[2])?;
        for v in &mut wdl {
            *v /= sum;
            if !v.is_finite() {
                return Err(NnuecError::NonFinite);
            }
        }
        Ok(NnuecOutput {
            schema: NNUEC_OUTPUT_SCHEMA.into(),
            weights_sha256: self
                .identity
                .weights_sha256
                .clone()
                .expect("admitted weights"),
            board_accumulators: board.values.clone(),
            context_accumulator: context.values.clone(),
            local_accumulator: local.projected.clone(),
            hidden,
            raw_outputs,
            value,
            wdl_logits,
            wdl,
        })
    }

    pub fn reference_input(
        &self,
        context: &NnuecContext,
        local: &NnuecLocal,
        position: &Position,
    ) -> Result<NnuecReferenceInput, NnuecError> {
        if context.model != self.identity
            || local.model != self.identity
            || local.work.position_identity() != position.position_identity()
        {
            return Err(NnuecError::StateMismatch);
        }
        Ok(NnuecReferenceInput {
            schema: NNUEC_INPUT_SCHEMA.into(),
            board_features: [
                board_features(position, Color::White),
                board_features(position, Color::Black),
            ],
            side_to_move: color_name(position.side_to_move()).into(),
            context_lines: context.lines.values().map(|l| l.input.clone()).collect(),
            local_line: ReferenceLine {
                anchor_features: local.anchor_features.clone(),
                perspective: color_name(local.perspective).into(),
                moves: local
                    .moves
                    .iter()
                    .copied()
                    .map(ReferenceMove::from_board)
                    .collect(),
            },
        })
    }
}

/// Owned B. It is deliberately not Clone: undo authority belongs to this owner.
pub struct NnuecBoard {
    values: [Vec<f32>; 2],
    position: PositionIdentity,
    model: Arc<CpuValueIdentity>,
    stamp: Arc<()>,
}
impl NnuecBoard {
    pub fn values(&self) -> &[Vec<f32>; 2] {
        &self.values
    }
}
pub struct NnuecBoardUndo {
    values: [Vec<f32>; 2],
    before: PositionIdentity,
    after: PositionIdentity,
    model: Arc<CpuValueIdentity>,
    before_stamp: Arc<()>,
    after_stamp: Arc<()>,
}

pub struct NnuecLine {
    model: Arc<CpuValueIdentity>,
    anchor: PositionIdentity,
    input: ReferenceLine,
    embedding: Vec<f32>,
}
impl NnuecLine {
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }
    pub fn anchor(&self) -> &PositionIdentity {
        &self.anchor
    }
}
pub struct NnuecContext {
    model: Arc<CpuValueIdentity>,
    lines: BTreeMap<u64, Arc<NnuecLine>>,
    values: Vec<f32>,
}
impl NnuecContext {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn len(&self) -> usize {
        self.lines.len()
    }
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}
pub struct NnuecLocal {
    work: Position,
    anchor: PositionIdentity,
    perspective: Color,
    anchor_features: Vec<usize>,
    anchor_values: Vec<f32>,
    tokens: Vec<Vec<f32>>,
    moves: Vec<BoardMove>,
    embedding: Vec<f32>,
    projected: Vec<f32>,
    model: Arc<CpuValueIdentity>,
    stamp: Arc<()>,
}
impl NnuecLocal {
    pub fn values(&self) -> &[f32] {
        &self.projected
    }
    pub fn moves(&self) -> &[BoardMove] {
        &self.moves
    }
    pub fn position(&self) -> &Position {
        &self.work
    }
}
pub struct NnuecLocalUndo {
    rules: UndoToken,
    previous_embedding: Vec<f32>,
    previous_projected: Vec<f32>,
    model: Arc<CpuValueIdentity>,
    before_stamp: Arc<()>,
    after_stamp: Arc<()>,
    expected_length: usize,
}

/// The value is a nonterminal learned/frontier number, never a Rules mate proof.
/// Candidate/context/route rows are independent uncalibrated hints. Row 4 is a
/// state-level candidate hint; per-move action scoring is a separate consumer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnuecOutput {
    pub schema: String,
    pub weights_sha256: String,
    pub board_accumulators: [Vec<f32>; 2],
    pub context_accumulator: Vec<f32>,
    pub local_accumulator: Vec<f32>,
    pub hidden: Vec<f32>,
    pub raw_outputs: [f32; NNUEC_OUTPUTS],
    pub value: f32,
    pub wdl_logits: [f32; 3],
    pub wdl: [f32; 3],
}
impl NnuecOutput {
    pub fn frontier_score(&self) -> i32 {
        self.value.clamp(-20_000.0, 20_000.0).round() as i32
    }
    pub fn context_utility(&self) -> [f32; 3] {
        [
            self.raw_outputs[5],
            self.raw_outputs[6],
            self.raw_outputs[7],
        ]
    }
    pub fn receiver_utility(&self) -> [f32; 3] {
        [
            self.raw_outputs[8],
            self.raw_outputs[9],
            self.raw_outputs[10],
        ]
    }
}

/// Independent reference wire contains only model-consumed features. Exact
/// Rules/history identities remain in the Rust owner and are not numeric IDs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnuecReferenceInput {
    pub schema: String,
    pub board_features: [Vec<usize>; 2],
    pub side_to_move: String,
    pub context_lines: Vec<ReferenceLine>,
    pub local_line: ReferenceLine,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceLine {
    pub anchor_features: Vec<usize>,
    pub perspective: String,
    pub moves: Vec<ReferenceMove>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceMove {
    pub from_square: u8,
    pub to_square: u8,
    pub promotion: u8,
}
impl ReferenceMove {
    fn from_board(mv: BoardMove) -> Self {
        Self {
            from_square: mv.from.index(),
            to_square: mv.to.index(),
            promotion: promotion_index(mv).expect("Rules move") as u8,
        }
    }
}

fn board_features(position: &Position, side: Color) -> Vec<usize> {
    let snapshot = position.snapshot();
    let mut out = Vec::with_capacity(40);
    for index in 0..64 {
        let square = Square::new(index).expect("bounded square");
        if let Some(piece) = snapshot.piece_at(square) {
            out.push(piece_feature(piece, square, side));
        }
    }
    out.extend(metadata_features(&snapshot, side));
    out
}
fn color_name(side: Color) -> &'static str {
    if side == Color::White {
        "white"
    } else {
        "black"
    }
}
fn oriented_square(square: Square, side: Color) -> usize {
    usize::from(square.index() ^ if side == Color::Black { 56 } else { 0 })
}
fn promotion_index(mv: BoardMove) -> Result<usize, NnuecError> {
    match mv.promotion {
        None => Ok(0),
        Some(PieceKind::Queen) => Ok(1),
        Some(PieceKind::Rook) => Ok(2),
        Some(PieceKind::Bishop) => Ok(3),
        Some(PieceKind::Knight) => Ok(4),
        _ => Err(NnuecError::InvalidConfig("invalid promotion")),
    }
}
fn add(a: f32, b: f32) -> Result<f32, NnuecError> {
    let v = a + b;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(NnuecError::NonFinite)
    }
}
fn mul(a: f32, b: f32) -> Result<f32, NnuecError> {
    let v = a * b;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(NnuecError::NonFinite)
    }
}
fn add_row(out: &mut [f32], weights: &[f32], row: usize, sign: f32) -> Result<(), NnuecError> {
    let width = out.len();
    let weights = weights
        .get(row * width..(row + 1) * width)
        .ok_or(NnuecError::StateMismatch)?;
    for (out, weight) in out.iter_mut().zip(weights) {
        *out = add(*out, mul(sign, *weight)?)?;
    }
    Ok(())
}
fn project(input: &[f32], weights: &[f32], bias: &[f32]) -> Result<Vec<f32>, NnuecError> {
    if weights.len() != input.len() * bias.len() {
        return Err(NnuecError::StateMismatch);
    }
    let mut out = bias.to_vec();
    for (row, out) in out.iter_mut().enumerate() {
        for (input, weight) in input
            .iter()
            .zip(&weights[row * input.len()..(row + 1) * input.len()])
        {
            *out = add(*out, mul(*input, *weight)?)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint() -> NnuecCheckpoint {
        let config = NnuecConfig {
            board_width: 3,
            line_width: 2,
            context_width: 3,
            local_width: 2,
            hidden_width: 4,
            causal_window: 3,
            max_lines: 2,
            max_plies: 8,
        };
        let weights = |n: usize| (0..n).map(|i| ((i % 17) as f32 - 8.0) / 128.0).collect();
        let e = config.line_width;
        NnuecCheckpoint {
            schema: NNUEC_SCHEMA.into(),
            semantics: NNUEC_SEMANTICS.into(),
            training: CpuTrainingState::Untrained,
            output_scale: 1000.0,
            parameters: NnuecParameters {
                board_weights: weights(CPU_FEATURE_COUNT * config.board_width),
                board_bias: vec![0.125; config.board_width],
                anchor_weights: weights(CPU_FEATURE_COUNT * e),
                anchor_bias: vec![0.25; e],
                from_weights: (0..64 * e).map(|i| i as f32 / 128.0).collect(),
                to_weights: weights(64 * e),
                promotion_weights: weights(5 * e),
                ply_weights: weights(config.max_plies * e),
                token_bias: vec![0.5; e],
                causal_weights: (0..config.causal_window)
                    .flat_map(|lag| vec![(lag + 1) as f32; e])
                    .collect(),
                causal_bias: vec![0.125; e],
                context_projection: weights(config.context_width * e),
                context_bias: vec![0.0; config.context_width],
                local_projection: weights(config.local_width * e),
                local_bias: vec![0.0; config.local_width],
                hidden_weights: weights(config.hidden_width * config.input_width()),
                hidden_bias: vec![0.125; config.hidden_width],
                output_weights: weights(NNUEC_OUTPUTS * config.hidden_width),
                output_bias: weights(NNUEC_OUTPUTS),
            },
            config,
        }
    }

    fn moves(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect()
    }
    fn position(text: &str) -> Position {
        if let Some(fen) = text.strip_prefix("position fen ") {
            Position::from_fen(fen).unwrap()
        } else {
            let mut position = Position::startpos();
            if let Some(rest) = text.strip_prefix("position startpos moves ") {
                for mv in moves(rest) {
                    position.make_move(mv).unwrap();
                }
            } else {
                assert_eq!(text, "position startpos");
            }
            position
        }
    }
    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!(
                (*a - *b).abs() <= NNUEC_ABSOLUTE_TOLERANCE + NNUEC_RELATIVE_TOLERANCE * b.abs(),
                "{a} != {b}"
            );
        }
    }
    fn bits(a: &[f32]) -> Vec<u32> {
        a.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn required_config_parameters_and_weight_domain_are_validated() {
        let good = checkpoint();
        good.validate().unwrap();
        let mut changed = good.clone();
        changed.config.causal_window = 4;
        assert_eq!(
            changed.validate(),
            Err(NnuecError::InvalidConfig(
                "unsupported dimension or capacity"
            ))
        );
        changed = good.clone();
        changed.parameters.board_weights.pop();
        assert!(matches!(
            changed.validate(),
            Err(NnuecError::InvalidCheckpoint(
                "parameter shape/allocation mismatch"
            ))
        ));
        changed = good.clone();
        changed.parameters.output_bias[0] = f32::NAN;
        assert!(matches!(
            changed.validate(),
            Err(NnuecError::InvalidCheckpoint(
                "non-finite/excessive parameter"
            ))
        ));
        changed = good.clone();
        changed.training = CpuTrainingState::Bootstrap;
        assert!(matches!(
            changed.validate(),
            Err(NnuecError::InvalidCheckpoint(_))
        ));
        changed = good.clone();
        changed.parameters.output_bias[0] += 0.125;
        assert_ne!(
            changed.weights_sha256().unwrap(),
            good.weights_sha256().unwrap()
        );
        changed = good.clone();
        changed.training = CpuTrainingState::Learned {
            run_id: "declaration-only".into(),
            steps: 1,
            dataset_sha256: "a".repeat(64),
        };
        assert_eq!(
            changed.weights_sha256().unwrap(),
            good.weights_sha256().unwrap()
        );
        let bytes = serde_json::to_vec(&good).unwrap();
        let restored = NnuecModel::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(restored.identity().training, CpuTrainingState::Untrained);
    }

    #[test]
    fn board_rules_deltas_match_rebuild_and_undo_restores_original_bits() {
        let model = NnuecModel::new(checkpoint()).unwrap();
        for (fen, text) in [
            ("position startpos", "e2e4"),
            ("position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("position fen 4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"),
            ("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
            ("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8r"),
            ("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8b"),
            ("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
        ] {
            let mut position = position(fen);
            let mut board = model.board(&position).unwrap();
            let original = board.values.clone();
            let rules = position.make_move(text.parse().unwrap()).unwrap();
            let undo = model.apply_board_delta(&mut board, rules.delta()).unwrap();
            let rebuilt = model.board(&position).unwrap();
            close(&board.values[0], &rebuilt.values[0]);
            close(&board.values[1], &rebuilt.values[1]);
            let child = board.values.clone();
            assert_eq!(
                model.apply_board_delta(&mut board, rules.delta()).err(),
                Some(NnuecError::StateMismatch)
            );
            assert_eq!(bits(&board.values[0]), bits(&child[0]));
            position.unmake(rules).unwrap();
            model.restore_board(&mut board, undo, &position).unwrap();
            assert_eq!(bits(&board.values[0]), bits(&original[0]));
            assert_eq!(bits(&board.values[1]), bits(&original[1]));
        }
    }

    #[test]
    fn causal_lines_preserve_order_and_local_lifo_is_transactional() {
        let model = NnuecModel::new(checkpoint()).unwrap();
        let root = position("position startpos");
        let a = model
            .encode_line(&root, &moves("g1f3 g8f6 b1c3 b8c6"))
            .unwrap();
        let b = model
            .encode_line(&root, &moves("b1c3 b8c6 g1f3 g8f6"))
            .unwrap();
        assert_ne!(bits(a.embedding()), bits(b.embedding()));
        let mut local = model.local(&root).unwrap();
        let empty = bits(local.values());
        let empty_position = local.position().position_identity();
        let first = model
            .append_local(&mut local, "e2e4".parse().unwrap())
            .unwrap();
        let after_first = bits(local.values());
        assert!(matches!(
            model.append_local(&mut local, "e2e4".parse().unwrap()),
            Err(NnuecError::Rules(_))
        ));
        assert_eq!(bits(local.values()), after_first);
        assert_eq!(local.moves().len(), 1);
        let second = model
            .append_local(&mut local, "e7e5".parse().unwrap())
            .unwrap();
        model.undo_local(&mut local, second).unwrap();
        assert_eq!(bits(local.values()), after_first);
        model.undo_local(&mut local, first).unwrap();
        assert_eq!(bits(local.values()), empty);
        assert_eq!(local.position().position_identity(), empty_position);
        let rebuilt = model
            .encode_line(&root, &moves("g1f3 g8f6 b1c3 b8c6"))
            .unwrap();
        assert_eq!(bits(a.embedding()), bits(rebuilt.embedding()));
    }

    #[test]
    fn immutable_context_add_replace_delete_matches_rebuild_without_worker_copy() {
        let model = NnuecModel::new(checkpoint()).unwrap();
        let root = position("position startpos");
        let a = model.encode_line(&root, &moves("e2e4 e7e5")).unwrap();
        let b = model.encode_line(&root, &moves("d2d4 d7d5")).unwrap();
        let c0 = model.empty_context();
        let old = bits(c0.values());
        let c1 = model.with_line(&c0, 17, Arc::clone(&a)).unwrap();
        let c2 = model.with_line(&c1, 99, Arc::clone(&b)).unwrap();
        assert_eq!(bits(c0.values()), old);
        assert!(c0.is_empty());
        assert!(Arc::ptr_eq(
            c1.lines.get(&17).unwrap(),
            c2.lines.get(&17).unwrap()
        ));
        close(c2.values(), model.rebuild_context(&c2).unwrap().values());
        let c3 = model.with_line(&c2, 17, Arc::clone(&b)).unwrap();
        close(c3.values(), model.rebuild_context(&c3).unwrap().values());
        let c4 = model.without_line(&c3, 99).unwrap();
        close(c4.values(), model.rebuild_context(&c4).unwrap().values());
        let before = bits(c2.values());
        assert!(matches!(
            model.with_line(&c2, 100, a),
            Err(NnuecError::Capacity("context lines"))
        ));
        assert_eq!(bits(c2.values()), before);
        let mut foreign_checkpoint = checkpoint();
        foreign_checkpoint.parameters.output_bias[0] += 1.0;
        let foreign = NnuecModel::new(foreign_checkpoint).unwrap();
        assert!(matches!(
            foreign.with_line(&c1, 99, b),
            Err(NnuecError::StateMismatch)
        ));
    }

    #[test]
    fn empty_perspectives_and_four_promotions_remain_distinct_without_rules_authority() {
        let model = NnuecModel::new(checkpoint()).unwrap();
        let root = position("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        let embeddings: Vec<Vec<u32>> = ["a7a8q", "a7a8r", "a7a8b", "a7a8n"]
            .into_iter()
            .map(|s| {
                bits(
                    model
                        .encode_line(&root, &[s.parse().unwrap()])
                        .unwrap()
                        .embedding(),
                )
            })
            .collect();
        for (i, a) in embeddings.iter().enumerate() {
            for b in &embeddings[i + 1..] {
                assert_ne!(a, b);
            }
        }
        for root in [
            position("position startpos"),
            position("position startpos moves e2e4"),
        ] {
            let board = model.board(&root).unwrap();
            let c = model.empty_context();
            let l = model.local(&root).unwrap();
            let output = model.forward(&board, &c, &l, &root).unwrap();
            assert_eq!(output.schema, NNUEC_OUTPUT_SCHEMA);
            assert!((output.wdl.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!(output.wdl.iter().all(|v| *v >= 0.0));
            assert!((-20_000..=20_000).contains(&output.frontier_score()));
            let wire = model.reference_input(&c, &l, &root).unwrap();
            assert_eq!(wire.side_to_move, color_name(root.side_to_move()));
            assert!(wire.local_line.moves.is_empty());
        }
    }

    #[test]
    fn capacity_and_nonfinite_failures_preserve_local_and_context_semantics() {
        let mut tiny = checkpoint();
        tiny.config.max_plies = 1;
        tiny.parameters.ply_weights.truncate(tiny.config.line_width);
        tiny.parameters.ply_weights.shrink_to_fit();
        let model = NnuecModel::new(tiny).unwrap();
        let root = position("position startpos");
        let mut local = model.local(&root).unwrap();
        model
            .append_local(&mut local, "e2e4".parse().unwrap())
            .unwrap();
        let before = bits(local.values());
        let position_before = local.position().position_identity();
        assert!(matches!(
            model.append_local(&mut local, "e7e5".parse().unwrap()),
            Err(NnuecError::Capacity("local plies"))
        ));
        assert_eq!(bits(local.values()), before);
        assert_eq!(local.position().position_identity(), position_before);
        assert_eq!(add(f32::MAX, f32::MAX), Err(NnuecError::NonFinite));
        assert_eq!(mul(f32::MAX, 2.0), Err(NnuecError::NonFinite));
        // A rejected LIFO token cannot rewind a different local owner.
        let mut other = model.local(&root).unwrap();
        let token = model
            .append_local(&mut other, "e2e4".parse().unwrap())
            .unwrap();
        assert_eq!(
            model.undo_local(&mut local, token),
            Err(NnuecError::StateMismatch)
        );
        assert_eq!(bits(local.values()), before);
    }
}
