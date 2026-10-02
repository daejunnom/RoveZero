//! B 담당의 임시 시간 정책. 공통 계약 revision 또는 정식 clock 설정이 아니다.
//!
//! 단조 clock의 `Instant`만 사용한다. UCI 연결부가 실제 차례의 clock/increment를
//! 고른 뒤 전달한다. 다음 착수 후 받을 increment를 현재 보유 시간으로 계산하지
//! 않는다. 시간 배분이 작으면 여유 시간을 억지로 확보하지 않고 탐색 예산을 0으로
//! 만들어 이미 검증한 합법 fallback을 즉시 반환할 수 있게 한다.
//!
//! 배분 전체의 끝은 `output_deadline`이다. 마지막 `output_margin`은 출력용으로
//! 남기고 그 앞을 `hard_deadline`으로 정한다. 결과 검증·수락은 반드시 이 시각보다
//! 일찍 끝나야 한다. 그 앞의 `drain_margin`은 취소 불가 실행을 마무리하는 비용이다.
//! `admission_deadline` 이후에는 새 요청을 제출하지 않는다. soft deadline은 정상
//! 탐색 종료의 목표이고, hard/admission 한도를 늘리는 권한을 주지 않는다.

use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

const BASIS_POINTS: u32 = 10_000;

/// 실제 차례의 clock 선택은 UCI 연결부 책임이다.
/// `go infinite`의 유한 자원·취소 정책은 이 시간 배분과 별도로 관리한다.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimeControl {
    /// 전체 자기 시간의 명시적 상한. clock 정책의 multiplier를 적용하지 않는다.
    MoveTime(Duration),
    Clock {
        remaining: Duration,
        increment: Duration,
        moves_to_go: Option<u32>,
    },
}

/// B 담당의 설정 제안이다. 실제 실행 manifest에서 값을 고정해야 한다.
///
/// 기본값: 남은 착수 수 30, increment 반영 비율 80%, hard 배분 배수 3,
/// 출력 여유 10ms, drain 여유 10ms. `moves_to_go`가 주어지면 30 대신 사용한다.
/// soft 전체 배분은 `remaining / moves + increment * share`이다. hard 전체
/// 배분은 soft의 배수와 현재 `remaining` 중 작은 값이다. 따라서 큰 increment도
/// 아직 받지 않은 시간을 빌려 쓰는 근거가 되지 않는다.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimeBudgetConfig {
    pub default_moves_to_go: u32,
    pub increment_share_basis_points: u16,
    pub hard_budget_multiplier: u32,
    pub output_margin: Duration,
    pub drain_margin: Duration,
}

impl Default for TimeBudgetConfig {
    fn default() -> Self {
        Self {
            default_moves_to_go: 30,
            increment_share_basis_points: 8_000,
            hard_budget_multiplier: 3,
            output_margin: Duration::from_millis(10),
            drain_margin: Duration::from_millis(10),
        }
    }
}

impl TimeBudgetConfig {
    pub fn validate(self) -> Result<(), TimeBudgetError> {
        if self.default_moves_to_go == 0 {
            return Err(TimeBudgetError::ZeroMoveHorizon);
        }
        if u32::from(self.increment_share_basis_points) > BASIS_POINTS {
            return Err(TimeBudgetError::InvalidIncrementShare(
                self.increment_share_basis_points,
            ));
        }
        if self.hard_budget_multiplier == 0 {
            return Err(TimeBudgetError::ZeroHardBudgetMultiplier);
        }
        self.output_margin
            .checked_add(self.drain_margin)
            .ok_or(TimeBudgetError::DurationOverflow("reserved margins"))?;
        Ok(())
    }
}

/// 배분의 끝과 요청 수명 경계. 모든 deadline 비교는 경계를 제외한다.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimeBudget {
    pub start: Instant,
    pub soft_deadline: Instant,
    pub admission_deadline: Instant,
    pub hard_deadline: Instant,
    pub output_deadline: Instant,
    /// 출력·drain 여유를 포함한 전체 배분. clock에서는 현재 보유 시간 이하이다.
    pub allocation: Duration,
}

impl TimeBudget {
    pub fn new(
        start: Instant,
        control: TimeControl,
        config: TimeBudgetConfig,
    ) -> Result<Self, TimeBudgetError> {
        config.validate()?;
        // 표현할 수 없는 외부 duration은 여유가 크다는 이유로 조용히 무시하지 않는다.
        deadline(start, config.output_margin, "output margin")?;
        deadline(start, config.drain_margin, "drain margin")?;

        let (soft_allocation, allocation) = match control {
            TimeControl::MoveTime(movetime) => {
                deadline(start, movetime, "movetime")?;
                (movetime, movetime)
            }
            TimeControl::Clock {
                remaining,
                increment,
                moves_to_go,
            } => {
                let horizon = moves_to_go.unwrap_or(config.default_moves_to_go);
                if horizon == 0 {
                    return Err(TimeBudgetError::ZeroMoveHorizon);
                }
                deadline(start, remaining, "remaining clock")?;
                deadline(start, increment, "increment")?;
                let increment_share = share_duration(
                    increment,
                    u32::from(config.increment_share_basis_points),
                )?;
                let soft = (remaining / horizon)
                    .checked_add(increment_share)
                    .ok_or(TimeBudgetError::DurationOverflow("soft allocation"))?;
                let hard = soft
                    .checked_mul(config.hard_budget_multiplier)
                    .ok_or(TimeBudgetError::DurationOverflow("hard allocation"))?;
                (soft.min(remaining), hard.min(remaining))
            }
        };

        // 각각 차감하므로 전체 배분보다 margin이 커도 0에서 끝나고 wrap하지 않는다.
        let result_window = allocation.saturating_sub(config.output_margin);
        let admission_window = result_window.saturating_sub(config.drain_margin);
        let soft_window = soft_allocation
            .saturating_sub(config.output_margin)
            .saturating_sub(config.drain_margin)
            .min(admission_window);
        Ok(Self {
            start,
            soft_deadline: deadline(start, soft_window, "soft deadline")?,
            admission_deadline: deadline(start, admission_window, "admission deadline")?,
            hard_deadline: deadline(start, result_window, "hard deadline")?,
            output_deadline: deadline(start, allocation, "output deadline")?,
            allocation,
        })
    }

    /// 새 평가 제출이 가능한 마지막 경계. soft 종료 여부는 탐색 정책이 별도 확인한다.
    pub fn can_admit(&self, now: Instant) -> bool {
        now < self.admission_deadline
    }

    /// 결과 검증·수락이 완료된 시각으로 검사해야 한다. enqueue 시각으로 대체하지 않는다.
    pub fn can_accept(&self, now: Instant) -> bool {
        now < self.hard_deadline
    }

    pub fn soft_expired(&self, now: Instant) -> bool {
        now >= self.soft_deadline
    }

    pub fn hard_expired(&self, now: Instant) -> bool {
        !self.can_accept(now)
    }

    pub fn remaining_admission_time(&self, now: Instant) -> Duration {
        self.admission_deadline.saturating_duration_since(now)
    }

    pub fn remaining_result_time(&self, now: Instant) -> Duration {
        self.hard_deadline.saturating_duration_since(now)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimeBudgetError {
    ZeroMoveHorizon,
    InvalidIncrementShare(u16),
    ZeroHardBudgetMultiplier,
    DurationOverflow(&'static str),
    DeadlineOverflow(&'static str),
}

impl fmt::Display for TimeBudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMoveHorizon => formatter.write_str("moves-to-go must be positive"),
            Self::InvalidIncrementShare(share) => {
                write!(formatter, "increment share {share} exceeds 10000 basis points")
            }
            Self::ZeroHardBudgetMultiplier => {
                formatter.write_str("hard budget multiplier must be positive")
            }
            Self::DurationOverflow(context) => {
                write!(formatter, "duration overflow while calculating {context}")
            }
            Self::DeadlineOverflow(context) => {
                write!(formatter, "monotonic deadline overflow for {context}")
            }
        }
    }
}

impl Error for TimeBudgetError {}

fn deadline(
    start: Instant,
    duration: Duration,
    context: &'static str,
) -> Result<Instant, TimeBudgetError> {
    start
        .checked_add(duration)
        .ok_or(TimeBudgetError::DeadlineOverflow(context))
}

fn share_duration(duration: Duration, share: u32) -> Result<Duration, TimeBudgetError> {
    // Duration::MAX nanoseconds * 10000 fits u128. Keep the check to make the bound explicit.
    let nanos = duration
        .as_nanos()
        .checked_mul(u128::from(share))
        .ok_or(TimeBudgetError::DurationOverflow("increment share"))?
        / u128::from(BASIS_POINTS);
    let seconds = u64::try_from(nanos / 1_000_000_000)
        .map_err(|_| TimeBudgetError::DurationOverflow("increment share"))?;
    let subsecond_nanos = (nanos % 1_000_000_000) as u32;
    Ok(Duration::new(seconds, subsecond_nanos))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn millis(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    fn elapsed(start: Instant, end: Instant) -> Duration {
        end.duration_since(start)
    }

    #[test]
    fn movetime_preserves_whole_limit_and_partitions_reserves() {
        let start = Instant::now();
        let budget = TimeBudget::new(
            start,
            TimeControl::MoveTime(millis(100)),
            TimeBudgetConfig::default(),
        )
        .unwrap();
        assert_eq!(budget.allocation, millis(100));
        assert_eq!(elapsed(start, budget.output_deadline), millis(100));
        assert_eq!(elapsed(start, budget.hard_deadline), millis(90));
        assert_eq!(elapsed(start, budget.admission_deadline), millis(80));
        assert_eq!(budget.soft_deadline, budget.admission_deadline);
    }

    #[test]
    fn exact_boundaries_do_not_admit_or_accept_work() {
        let start = Instant::now();
        let budget = TimeBudget::new(
            start,
            TimeControl::MoveTime(millis(100)),
            TimeBudgetConfig::default(),
        )
        .unwrap();
        assert!(budget.can_admit(start + millis(79)));
        assert!(!budget.can_admit(start + millis(80)));
        // A previously admitted request may finish in the reserved drain window.
        assert!(budget.can_accept(start + millis(89)));
        assert!(!budget.can_accept(start + millis(90)));
        assert!(budget.soft_expired(start + millis(80)));
        assert!(budget.hard_expired(start + millis(90)));
        assert_eq!(budget.remaining_result_time(start + millis(91)), Duration::ZERO);
    }

    #[test]
    fn zero_and_tiny_limits_leave_no_search_admission() {
        let start = Instant::now();
        for requested in [0, 1, 10, 20] {
            let budget = TimeBudget::new(
                start,
                TimeControl::MoveTime(millis(requested)),
                TimeBudgetConfig::default(),
            )
            .unwrap();
            assert_eq!(budget.admission_deadline, start);
            assert!(budget.soft_expired(start));
            assert!(!budget.can_admit(start));
            assert_eq!(budget.output_deadline, start + millis(requested));
        }
    }

    #[test]
    fn clock_soft_target_and_hard_limit_use_different_windows() {
        let start = Instant::now();
        let budget = TimeBudget::new(
            start,
            TimeControl::Clock {
                remaining: millis(30_000),
                increment: millis(1_000),
                moves_to_go: None,
            },
            TimeBudgetConfig::default(),
        )
        .unwrap();
        // Independently calculated: 30000/30 + 1000*0.8 = 1800; hard = 5400.
        assert_eq!(elapsed(start, budget.soft_deadline), millis(1_780));
        assert_eq!(elapsed(start, budget.admission_deadline), millis(5_380));
        assert_eq!(elapsed(start, budget.hard_deadline), millis(5_390));
        assert_eq!(budget.allocation, millis(5_400));
    }

    #[test]
    fn future_increment_cannot_exceed_current_remaining_clock() {
        let start = Instant::now();
        let budget = TimeBudget::new(
            start,
            TimeControl::Clock {
                remaining: millis(200),
                increment: Duration::from_secs(60),
                moves_to_go: Some(1),
            },
            TimeBudgetConfig::default(),
        )
        .unwrap();
        assert_eq!(budget.allocation, millis(200));
        assert_eq!(elapsed(start, budget.admission_deadline), millis(180));
        assert_eq!(budget.soft_deadline, budget.admission_deadline);
    }

    #[test]
    fn explicit_moves_to_go_overrides_default_horizon() {
        let start = Instant::now();
        let budget = TimeBudget::new(
            start,
            TimeControl::Clock {
                remaining: millis(1_000),
                increment: Duration::ZERO,
                moves_to_go: Some(10),
            },
            TimeBudgetConfig::default(),
        )
        .unwrap();
        assert_eq!(elapsed(start, budget.soft_deadline), millis(80));
        assert_eq!(budget.allocation, millis(300));
    }

    #[test]
    fn movetime_does_not_apply_clock_multiplier() {
        let start = Instant::now();
        let config = TimeBudgetConfig {
            hard_budget_multiplier: 100,
            ..TimeBudgetConfig::default()
        };
        let budget = TimeBudget::new(start, TimeControl::MoveTime(millis(100)), config).unwrap();
        assert_eq!(budget.allocation, millis(100));
    }

    #[test]
    fn invalid_horizons_and_config_are_errors() {
        let start = Instant::now();
        let control = TimeControl::Clock {
            remaining: millis(1_000),
            increment: Duration::ZERO,
            moves_to_go: Some(0),
        };
        assert_eq!(
            TimeBudget::new(start, control, TimeBudgetConfig::default()),
            Err(TimeBudgetError::ZeroMoveHorizon)
        );
        let invalid_configs = [
            (
                TimeBudgetConfig {
                    default_moves_to_go: 0,
                    ..TimeBudgetConfig::default()
                },
                TimeBudgetError::ZeroMoveHorizon,
            ),
            (
                TimeBudgetConfig {
                    increment_share_basis_points: 10_001,
                    ..TimeBudgetConfig::default()
                },
                TimeBudgetError::InvalidIncrementShare(10_001),
            ),
            (
                TimeBudgetConfig {
                    hard_budget_multiplier: 0,
                    ..TimeBudgetConfig::default()
                },
                TimeBudgetError::ZeroHardBudgetMultiplier,
            ),
        ];
        for (config, expected) in invalid_configs {
            assert_eq!(config.validate(), Err(expected));
        }
    }

    #[test]
    fn unrepresentable_duration_is_rejected_instead_of_wrapping() {
        let start = Instant::now();
        assert_eq!(
            TimeBudget::new(
                start,
                TimeControl::MoveTime(Duration::MAX),
                TimeBudgetConfig::default(),
            ),
            Err(TimeBudgetError::DeadlineOverflow("movetime"))
        );
        let config = TimeBudgetConfig {
            output_margin: Duration::MAX,
            drain_margin: Duration::from_nanos(1),
            ..TimeBudgetConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(TimeBudgetError::DurationOverflow("reserved margins"))
        );
    }

    #[test]
    fn oversized_hard_multiplier_is_checked_before_clock_cap() {
        let start = Instant::now();
        let control = TimeControl::Clock {
            remaining: Duration::from_secs(10_000_000_000),
            increment: Duration::ZERO,
            moves_to_go: Some(1),
        };
        let config = TimeBudgetConfig {
            hard_budget_multiplier: u32::MAX,
            ..TimeBudgetConfig::default()
        };
        assert_eq!(
            TimeBudget::new(start, control, config),
            Err(TimeBudgetError::DurationOverflow("hard allocation"))
        );
    }
}
