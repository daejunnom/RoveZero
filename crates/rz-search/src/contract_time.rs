//! 공통 0.1의 단조 ns clock/deadline과 B의 Instant 시간 배분을 연결한다.
//!
//! ClockDomain과 origin은 실제 process owner가 한 번 정해 같은 도메인으로
//! 공유해야 한다. origin은 `go` 수신 전에 확보한다. 준비 작업 이후 새 origin이나
//! 새 상대 예산을 만들면 이미 쓴 자기 시간을 빠뜨리므로 그렇게 변환하지 않는다.
//! UTC 및 Instant 내부 표현을 tick으로 복사하지 않고 origin 이후 elapsed ns만 쓴다.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use rz_contracts::{
    CancelToken, ClockDomain, ContractError, Deadline, ErrorCode, MonotonicTick, Stage,
};

use crate::driver::SearchControl;
use crate::time::TimeBudget;

/// 실제 owner의 단조 clock. 수동 CPU 검사 clock도 같은 domain 규칙을 지킨다.
pub trait ContractClock {
    fn domain(&self) -> ClockDomain;
    fn now(&self) -> Result<MonotonicTick, ContractError>;
}

/// 같은 process epoch 안의 하나의 Instant origin. domain 발급 권한은 caller 소유다.
#[derive(Clone, Copy, Debug)]
pub struct InstantClock {
    domain: ClockDomain,
    origin: Instant,
}

impl InstantClock {
    pub fn new(domain: ClockDomain, origin: Instant) -> Self {
        Self { domain, origin }
    }

    pub fn domain(&self) -> ClockDomain {
        self.domain
    }

    pub fn origin(&self) -> Instant {
        self.origin
    }

    pub fn tick_at(&self, instant: Instant) -> Result<MonotonicTick, ContractError> {
        let elapsed = instant.checked_duration_since(self.origin).ok_or_else(|| {
            ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "Instant precedes the owned clock origin",
            )
        })?;
        elapsed_tick(elapsed)
    }

    pub fn instant_at(&self, tick: MonotonicTick) -> Result<Instant, ContractError> {
        self.origin
            .checked_add(Duration::from_nanos(tick.0))
            .ok_or_else(|| {
                ContractError::new(
                    ErrorCode::ResourceExhausted,
                    Stage::Admission,
                    "monotonic tick exceeds the platform Instant range",
                )
            })
    }

    /// Construct a future deadline in this owned domain. Equality is expired.
    pub fn deadline_at(&self, instant: Instant) -> Result<Deadline, ContractError> {
        let deadline = Deadline {
            clock: self.domain,
            at: self.tick_at(instant)?,
        };
        deadline.accepts(self.domain, self.now()?)?;
        Ok(deadline)
    }
}

fn elapsed_tick(elapsed: Duration) -> Result<MonotonicTick, ContractError> {
    let nanos = u64::try_from(elapsed.as_nanos()).map_err(|_| {
        ContractError::new(
            ErrorCode::ResourceExhausted,
            Stage::Admission,
            "elapsed nanoseconds exceed the monotonic u64 tick range",
        )
    })?;
    Ok(MonotonicTick(nanos))
}

impl ContractClock for InstantClock {
    fn domain(&self) -> ClockDomain {
        self.domain
    }

    fn now(&self) -> Result<MonotonicTick, ContractError> {
        self.tick_at(Instant::now())
    }
}

/// Each deadline uses the same owned origin; hard is the strict result boundary.
/// Soft and admission may be in either order and may have passed while an
/// already-admitted evaluation is still valid before hard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContractDeadlines {
    pub soft: Deadline,
    pub admission: Deadline,
    pub hard: Deadline,
    pub output: Deadline,
}

impl ContractDeadlines {
    pub fn from_budget(clock: &InstantClock, budget: &TimeBudget) -> Result<Self, ContractError> {
        // The shared origin must precede go/start; resetting it after preparation
        // cannot silently exclude that preparation cost even if deadlines are future.
        clock.tick_at(budget.start)?;
        let convert = |instant| {
            Ok(Deadline {
                clock: clock.domain(),
                at: clock.tick_at(instant)?,
            })
        };
        let deadlines = Self {
            soft: convert(budget.soft_deadline)?,
            admission: convert(budget.admission_deadline)?,
            hard: convert(budget.hard_deadline)?,
            output: convert(budget.output_deadline)?,
        };
        deadlines.validate(clock)?;
        Ok(deadlines)
    }

    /// Validate metadata first, then check acceptance using a fresh owner tick.
    /// Caller must repeat this guard immediately before logical acceptance/backup.
    pub fn validate<C: ContractClock + ?Sized>(&self, clock: &C) -> Result<(), ContractError> {
        self.validate_metadata()?;
        let domain = clock.domain();
        if self.hard.clock != domain {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "search deadline does not belong to the current clock owner",
            ));
        }
        self.hard.accepts(domain, clock.now()?)
    }

    /// Constructors can check structure before a clock is available. This does
    /// not attest that any deadline remains current; admission repeats validate.
    pub fn validate_metadata(&self) -> Result<(), ContractError> {
        let domain = self.hard.clock;
        if [self.soft, self.admission, self.hard, self.output]
            .iter()
            .any(|deadline| deadline.clock != domain)
        {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "search deadlines belong to different clock domains",
            ));
        }
        if self.soft.at > self.hard.at
            || self.admission.at > self.hard.at
            || self.hard.at > self.output.at
        {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "search deadline boundaries are incoherent",
            ));
        }
        Ok(())
    }
}

/// Bridges the old SearchControl cancellation flag and the common logical token.
/// It does not own physical buffer release or guarantee that a GPU kernel stops.
///
/// `cancel()` closes both authorities. Direct cancellation through either exposed
/// clone is observed by `is_canceled()`, but their private Arc values are separate.
/// External common cancellation is authoritative for the async acceptance guard.
/// A legacy worker adapter must check this wrapper/common token; calling only the
/// old SearchControl::stop_reason cannot observe a directly canceled common clone.
#[derive(Clone, Debug)]
pub struct ContractCancellation {
    token: CancelToken,
    local: Arc<AtomicBool>,
}

impl ContractCancellation {
    pub fn from_control(control: &SearchControl) -> Self {
        let token = CancelToken::new();
        if control.cancellation.load(Ordering::Acquire) {
            token.cancel();
        }
        Self {
            token,
            local: control.cancellation.clone(),
        }
    }

    pub fn token(&self) -> CancelToken {
        self.token.clone()
    }

    pub fn cancel(&self) {
        self.local.store(true, Ordering::Release);
        self.token.cancel();
    }

    pub fn is_canceled(&self) -> bool {
        self.token.is_canceled() || self.local.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::StopReason;
    use rz_contracts::ProcessEpoch;

    #[derive(Clone, Copy)]
    struct ManualClock {
        domain: ClockDomain,
        tick: MonotonicTick,
    }

    impl ContractClock for ManualClock {
        fn domain(&self) -> ClockDomain {
            self.domain
        }
        fn now(&self) -> Result<MonotonicTick, ContractError> {
            Ok(self.tick)
        }
    }

    fn domain() -> ClockDomain {
        ClockDomain(ProcessEpoch(7))
    }

    fn deadlines(soft: u64, admission: u64, hard: u64, output: u64) -> ContractDeadlines {
        let deadline = |tick| Deadline {
            clock: domain(),
            at: MonotonicTick(tick),
        };
        ContractDeadlines {
            soft: deadline(soft),
            admission: deadline(admission),
            hard: deadline(hard),
            output: deadline(output),
        }
    }

    #[test]
    fn instant_tick_roundtrip_uses_exact_integral_nanoseconds() {
        let origin = Instant::now();
        let clock = InstantClock::new(domain(), origin);
        assert_eq!(clock.domain(), domain());
        assert_eq!(clock.origin(), origin);
        for nanos in [0, 1, 1_000_000_001, u64::MAX] {
            assert_eq!(
                elapsed_tick(Duration::from_nanos(nanos)).unwrap(),
                MonotonicTick(nanos)
            );
            if let Some(expected) = origin.checked_add(Duration::from_nanos(nanos)) {
                assert_eq!(clock.instant_at(MonotonicTick(nanos)).unwrap(), expected);
                assert_eq!(clock.tick_at(expected).unwrap(), MonotonicTick(nanos));
            } else {
                // The platform may have a smaller Instant range than u64 nanoseconds.
                assert_eq!(
                    clock.instant_at(MonotonicTick(nanos)).unwrap_err().code,
                    ErrorCode::ResourceExhausted
                );
            }
        }
    }

    #[test]
    fn before_origin_and_tick_overflow_are_explicit_errors() {
        let origin = Instant::now();
        let clock = InstantClock::new(domain(), origin);
        let before = origin.checked_sub(Duration::from_nanos(1)).unwrap();
        assert_eq!(
            clock.tick_at(before).unwrap_err().code,
            ErrorCode::InvalidInput
        );
        let beyond_tick = Duration::from_nanos(u64::MAX)
            .checked_add(Duration::from_nanos(1))
            .unwrap();
        assert_eq!(
            elapsed_tick(beyond_tick).unwrap_err().code,
            ErrorCode::ResourceExhausted
        );
        if let Some(beyond) = origin.checked_add(beyond_tick) {
            assert_eq!(
                clock.tick_at(beyond).unwrap_err().code,
                ErrorCode::ResourceExhausted
            );
        }
    }

    #[test]
    fn deadline_constructor_uses_owned_domain_and_rejects_current_or_past() {
        let origin = Instant::now();
        let clock = InstantClock::new(domain(), origin);
        let deadline = clock.deadline_at(origin + Duration::from_secs(60)).unwrap();
        assert_eq!(deadline.clock, domain());
        assert_eq!(deadline.at, MonotonicTick(60_000_000_000));
        assert_eq!(
            clock.deadline_at(origin).unwrap_err().code,
            ErrorCode::Expired
        );
        let future_origin = InstantClock::new(domain(), origin + Duration::from_secs(60));
        assert_eq!(
            future_origin.now().unwrap_err().code,
            ErrorCode::InvalidInput
        );
    }

    #[test]
    fn hard_acceptance_is_strict_and_soft_admission_can_already_have_passed() {
        let schedule = deadlines(10, 20, 30, 40);
        for (now, expected) in [
            (0, None),
            (29, None),
            (30, Some(ErrorCode::Expired)),
            (31, Some(ErrorCode::Expired)),
        ] {
            let clock = ManualClock {
                domain: domain(),
                tick: MonotonicTick(now),
            };
            assert_eq!(
                schedule.validate(&clock).err().map(|error| error.code),
                expected
            );
        }
        let clock = ManualClock {
            domain: domain(),
            tick: MonotonicTick(29),
        };
        assert!(deadlines(20, 10, 30, 30).validate(&clock).is_ok());
    }

    #[test]
    fn every_deadline_domain_and_each_order_boundary_is_checked() {
        let clock = ManualClock {
            domain: domain(),
            tick: MonotonicTick(0),
        };
        for index in 0..4 {
            let mut deadlines = deadlines(10, 20, 30, 40);
            let boundaries = [
                &mut deadlines.soft,
                &mut deadlines.admission,
                &mut deadlines.hard,
                &mut deadlines.output,
            ];
            boundaries.into_iter().nth(index).unwrap().clock = ClockDomain(ProcessEpoch(8));
            assert_eq!(
                deadlines.validate(&clock).unwrap_err().code,
                ErrorCode::InvalidInput
            );
        }
        for invalid in [
            deadlines(31, 20, 30, 40),
            deadlines(10, 31, 30, 40),
            deadlines(10, 20, 41, 40),
        ] {
            assert_eq!(
                invalid.validate(&clock).unwrap_err().code,
                ErrorCode::InvalidInput
            );
        }
        let other = ManualClock {
            domain: ClockDomain(ProcessEpoch(8)),
            ..clock
        };
        assert_eq!(
            deadlines(10, 20, 30, 40).validate(&other).unwrap_err().code,
            ErrorCode::InvalidInput
        );
    }

    #[test]
    fn budget_conversion_keeps_process_origin_and_preparation_time() {
        let origin = Instant::now();
        let clock = InstantClock::new(domain(), origin);
        let start = origin + Duration::from_secs(5);
        let budget = TimeBudget {
            start,
            soft_deadline: start + Duration::from_secs(1),
            admission_deadline: start + Duration::from_secs(2),
            hard_deadline: start + Duration::from_secs(3),
            output_deadline: start + Duration::from_secs(4),
            allocation: Duration::from_secs(4),
        };
        let converted = ContractDeadlines::from_budget(&clock, &budget).unwrap();
        assert_eq!(
            converted,
            deadlines(6_000_000_000, 7_000_000_000, 8_000_000_000, 9_000_000_000)
        );
        let reset_after_start = InstantClock::new(domain(), start + Duration::from_nanos(1));
        assert_eq!(
            ContractDeadlines::from_budget(&reset_after_start, &budget)
                .unwrap_err()
                .code,
            ErrorCode::InvalidInput
        );
    }

    #[test]
    fn expired_or_overflowing_budget_is_not_recreated_as_new_relative_time() {
        let now = Instant::now();
        let origin = now - Duration::from_secs(10);
        let clock = InstantClock::new(domain(), origin);
        let expired = TimeBudget {
            start: origin,
            soft_deadline: origin + Duration::from_secs(1),
            admission_deadline: origin + Duration::from_secs(2),
            hard_deadline: origin + Duration::from_secs(3),
            output_deadline: origin + Duration::from_secs(4),
            allocation: Duration::from_secs(4),
        };
        assert_eq!(
            ContractDeadlines::from_budget(&clock, &expired)
                .unwrap_err()
                .code,
            ErrorCode::Expired
        );
        let beyond_tick = Duration::from_nanos(u64::MAX)
            .checked_add(Duration::from_nanos(1))
            .unwrap();
        if let Some(too_late) = clock.origin().checked_add(beyond_tick) {
            let overflowing = TimeBudget {
                soft_deadline: too_late,
                admission_deadline: too_late,
                hard_deadline: too_late,
                output_deadline: too_late,
                ..expired
            };
            assert_eq!(
                ContractDeadlines::from_budget(&clock, &overflowing)
                    .unwrap_err()
                    .code,
                ErrorCode::ResourceExhausted
            );
        }
    }

    #[test]
    fn adapter_cancel_closes_both_and_existing_local_cancel_is_preserved() {
        let now = Instant::now();
        let control = SearchControl::new(now + Duration::from_secs(60), 10);
        let cancel = ContractCancellation::from_control(&control);
        let token = cancel.token();
        assert!(!cancel.is_canceled());
        cancel.cancel();
        cancel.cancel();
        assert!(cancel.is_canceled());
        assert!(token.is_canceled());
        assert_eq!(control.stop_reason(now), Some(StopReason::Canceled));
        let already_canceled = ContractCancellation::from_control(&control);
        assert!(already_canceled.is_canceled());
        assert!(already_canceled.token().is_canceled());
    }

    #[test]
    fn either_external_clone_is_authoritative_without_claiming_shared_private_arc() {
        let now = Instant::now();
        let control = SearchControl::new(now + Duration::from_secs(60), 10);
        let cancel = ContractCancellation::from_control(&control);
        let other_adapter = cancel.clone();
        cancel.token().cancel();
        assert!(cancel.is_canceled());
        assert!(other_adapter.is_canceled());
        // The legacy-only flag does not claim to observe the common private Arc.
        assert_eq!(control.stop_reason(now), None);
        let control = SearchControl::new(now + Duration::from_secs(60), 10);
        let cancel = ContractCancellation::from_control(&control);
        control.cancel();
        assert!(cancel.is_canceled());
        // A direct local cancellation is not a call to the adapter's cancel().
        assert!(!cancel.token().is_canceled());
    }
}
