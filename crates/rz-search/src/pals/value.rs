//! PALS 모델 값의 고정 의미와 제한 게임 WDL 접점.
//!
//! 모델 WDL, 자체 CPU raw 값과 Rules 종료 사실은 서로 다른 타입이다. 여기의
//! 순수 helper는 CP 변환, 값의 평균, 전체 방어 수의 승패 증명을 수행하지 않는다.
//! 모델 identity와 입력 digest는 선언/식별 경계다. provider는 실제 선택한 모델과
//! 준비 입력에서 이를 생성하고, consumer는 고정 identity와 정확한 Rules 상태를
//! 대조해야 한다. `[u8; 32]` 형식 자체는 입력 bytes를 검증했다는 증거가 아니다.

use super::engine::RoleError;
use rz_position::{Color, Position, PositionIdentity};
use std::cmp::Ordering;

pub const MODEL_WDL_VALUE_SEMANTICS: &str = "rz-pals-context-wdl/1;side-to-move;restricted-model-estimate;actual-prepared-input;no-cp-calibration;no-rules-proof";
pub const MODEL_WDL_RESOLVER_VERSION: &str = "pals-model-wdl-restricted/0.1";
/// Integration policy: only Rules can certify a win. Accepted children use
/// side-to-move W-L, perspective reversal swaps W/L, and equal scores keep the
/// existing Rules order. Unknown is absent evidence, never a zero-score draw.
/// This declaration does not turn these helpers into a whole-game proof.
pub const MODEL_WDL_RESOLVER_SEMANTICS: &str = "root:exact-Rules-win-priority;child:accepted-registered-model-W-L-max;perspective:flip-W-L-preserve-D;unknown:None-not-zero;ties:Rules-order;namespace:model-estimate-only;owned-raw:separate;cp-conversion:none;averaging:none;all-defenses-proof:none";

const MAX_SEMANTICS_BYTES: usize = 256;
const MAX_MODEL_BYTES: usize = 1024;
const MAX_ENCODING_BYTES: usize = 256;
const MAX_PRECISION_BYTES: usize = 64;
const MAX_IDENTITY_CAPACITY: usize =
    MAX_SEMANTICS_BYTES + MAX_MODEL_BYTES + MAX_ENCODING_BYTES + MAX_PRECISION_BYTES;
const WDL_SUM_TOLERANCE: f32 = 1e-4;

/// Immutable selected-model namespace, independent of own CPU score semantics.
/// `model` is the actual RoleModel identity/manifest, not an invented model label.
/// `model_epoch` and an encoding digest identify the provider's actual frozen
/// configuration; validating their shape does not attest the corresponding data.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ModelValueIdentity {
    pub semantics: String,
    pub model: String,
    pub encoding: String,
    pub precision: String,
    pub model_epoch: [u8; 32],
}

impl ModelValueIdentity {
    pub fn validate(&self) -> Result<(), RoleError> {
        let mut capacity = 0usize;
        for (text, maximum) in [
            (&self.semantics, MAX_SEMANTICS_BYTES),
            (&self.model, MAX_MODEL_BYTES),
            (&self.encoding, MAX_ENCODING_BYTES),
            (&self.precision, MAX_PRECISION_BYTES),
        ] {
            if text.trim().is_empty()
                || text.len() > maximum
                || text.capacity() > maximum
                || text.chars().any(char::is_control)
            {
                return Err(RoleError::InvalidOutput);
            }
            capacity = capacity
                .checked_add(text.capacity())
                .ok_or(RoleError::InvalidOutput)?;
        }
        if capacity > MAX_IDENTITY_CAPACITY {
            return Err(RoleError::InvalidOutput);
        }
        Ok(())
    }
}

/// Model WDL for one exact Rules state/history in its side-to-move perspective.
/// `input_sha256` comes from actual provider preparation. The consumer must also
/// compare it with any separately registered input key before cache/task reuse;
/// this state's validator cannot reconstruct a model's private encoding.
#[derive(Clone, Debug)]
pub struct ModelValueOutput {
    pub identity: ModelValueIdentity,
    pub input_sha256: [u8; 32],
    pub state: PositionIdentity,
    pub perspective: Color,
    pub wdl: [f32; 3],
}

impl ModelValueOutput {
    pub fn validate(
        &self,
        position: &Position,
        expected_identity: &ModelValueIdentity,
    ) -> Result<(), RoleError> {
        expected_identity.validate()?;
        self.identity.validate()?;
        if &self.identity != expected_identity
            || self.state != position.position_identity()
            || self.perspective != position.side_to_move()
        {
            return Err(RoleError::InvalidOutput);
        }
        validate_wdl(self.wdl)
    }
}

/// These variants cannot be averaged or converted into each other implicitly.
/// Only the Rules owner may construct terminal facts for the resolved state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PalsResolvedValue {
    Unknown,
    OwnedRaw {
        value: i32,
        perspective: Color,
    },
    ModelWdl {
        wdl: [f32; 3],
        perspective: Color,
    },
    RulesTerminal {
        winner: Option<Color>,
        perspective: Color,
    },
    /// Derived only from a Rules-terminal leaf along examined continuations.
    /// Unexamined defenses remain unknown: this is not a whole-game outcome,
    /// calibrated neural distribution or an all-defenses proof.
    RestrictedRulesLine {
        winner: Option<Color>,
        perspective: Color,
    },
}

fn validate_wdl(wdl: [f32; 3]) -> Result<(), RoleError> {
    if wdl
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || (wdl.iter().sum::<f32>() - 1.0).abs() > WDL_SUM_TOLERANCE
    {
        return Err(RoleError::InvalidOutput);
    }
    Ok(())
}

/// Reverse perspective without CP mapping or changing the draw probability.
pub fn flip_model_wdl(wdl: [f32; 3]) -> Result<[f32; 3], RoleError> {
    validate_wdl(wdl)?;
    Ok([wdl[2], wdl[1], wdl[0]])
}

/// Compare only expected W-L in one registered model namespace and perspective.
/// Outputs may be for different child states. Each must already have passed
/// `ModelValueOutput::validate` against its own target Position. Keep those raw
/// outputs unchanged: parent preference reverses the child-side ordering, while
/// `flip_model_wdl` prepares separate derived `PalsResolvedValue::ModelWdl` values.
/// Equal expectations return Equal, including +/-0, so callers keep Rules order.
pub fn compare_model_wdl(
    expected_identity: &ModelValueIdentity,
    left: &ModelValueOutput,
    right: &ModelValueOutput,
) -> Result<Ordering, RoleError> {
    expected_identity.validate()?;
    left.identity.validate()?;
    right.identity.validate()?;
    if &left.identity != expected_identity
        || &right.identity != expected_identity
        || left.perspective != right.perspective
    {
        return Err(RoleError::InvalidOutput);
    }
    validate_wdl(left.wdl)?;
    validate_wdl(right.wdl)?;
    (left.wdl[0] - left.wdl[2])
        .partial_cmp(&(right.wdl[0] - right.wdl[2]))
        .ok_or(RoleError::InvalidOutput)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_position::BoardMove;

    fn identity() -> ModelValueIdentity {
        ModelValueIdentity {
            semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
            model: "mock-proposer-fixed-model-v1".into(),
            encoding: "mock-state-candidates-v1".into(),
            precision: "fp32".into(),
            model_epoch: [7; 32],
        }
    }

    fn output(position: &Position, wdl: [f32; 3]) -> ModelValueOutput {
        ModelValueOutput {
            identity: identity(),
            input_sha256: [9; 32],
            state: position.position_identity(),
            perspective: position.side_to_move(),
            wdl,
        }
    }

    #[test]
    fn flip_preserves_draw_and_is_an_involution() {
        let wdl = [0.65, 0.25, 0.10];
        let flipped = flip_model_wdl(wdl).unwrap();
        assert_eq!(flipped, [0.10, 0.25, 0.65]);
        assert_eq!(flip_model_wdl(flipped).unwrap(), wdl);
        assert!(flip_model_wdl([f32::NAN, 0.0, 1.0]).is_err());
    }

    #[test]
    fn expected_value_ties_keep_rules_order_and_do_not_average() {
        let position = Position::startpos();
        let left = output(&position, [0.5, 0.0, 0.5]);
        let right = output(&position, [0.25, 0.5, 0.25]);
        assert_eq!(
            compare_model_wdl(&identity(), &left, &right).unwrap(),
            Ordering::Equal
        );
        let positive = output(&position, [0.75, 0.0, 0.25]);
        assert_eq!(
            compare_model_wdl(&identity(), &positive, &right).unwrap(),
            Ordering::Greater
        );
        let negative_zero = output(&position, [-0.0, 1.0, 0.0]);
        let positive_zero = output(&position, [0.0, 1.0, 0.0]);
        assert_eq!(
            compare_model_wdl(&identity(), &negative_zero, &positive_zero).unwrap(),
            Ordering::Equal
        );
    }

    #[test]
    fn output_rejects_wrong_state_history_perspective_and_namespace() {
        let position = Position::startpos();
        let expected = identity();
        let accepted = output(&position, [0.3, 0.4, 0.3]);
        accepted.validate(&position, &expected).unwrap();
        let mut repeated = position.clone();
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            repeated
                .make_move(BoardMove::from_uci(uci).unwrap())
                .unwrap();
        }
        let historyless = Position::from_fen(&repeated.to_fen()).unwrap();
        let with_history = output(&repeated, [0.3, 0.4, 0.3]);
        assert!(with_history.validate(&historyless, &expected).is_err());
        assert!(accepted.validate(&repeated, &expected).is_err());
        let mut wrong = accepted.clone();
        wrong.perspective = Color::Black;
        assert!(wrong.validate(&position, &expected).is_err());
        wrong = accepted.clone();
        wrong.identity.model_epoch[0] ^= 1;
        assert!(wrong.validate(&position, &expected).is_err());
        assert!(compare_model_wdl(&expected, &accepted, &wrong).is_err());
        wrong = accepted.clone();
        wrong.identity.precision = "fp16".into();
        assert!(compare_model_wdl(&expected, &accepted, &wrong).is_err());
    }

    #[test]
    fn invalid_wdl_and_retained_namespace_capacity_are_rejected() {
        let position = Position::startpos();
        for wdl in [
            [0.1, 0.2, 0.3],
            [-0.1, 0.8, 0.3],
            [0.0, f32::INFINITY, 1.0],
            [f32::NAN, 0.0, 1.0],
        ] {
            assert!(
                output(&position, wdl)
                    .validate(&position, &identity())
                    .is_err()
            );
        }
        for (field, maximum) in [
            (0, MAX_SEMANTICS_BYTES),
            (1, MAX_MODEL_BYTES),
            (2, MAX_ENCODING_BYTES),
            (3, MAX_PRECISION_BYTES),
        ] {
            let mut oversized = identity();
            let mut text = String::with_capacity(maximum + 1);
            text.push('x');
            match field {
                0 => oversized.semantics = text,
                1 => oversized.model = text,
                2 => oversized.encoding = text,
                _ => oversized.precision = text,
            }
            assert!(oversized.validate().is_err());
        }
        let mut invalid = identity();
        invalid.model.push('\n');
        assert!(invalid.validate().is_err());
        invalid = identity();
        invalid.encoding = " ".into();
        assert!(invalid.validate().is_err());
    }
}
