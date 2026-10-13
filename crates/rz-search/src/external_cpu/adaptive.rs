//! 외부 CPU의 선언·지원 범위. CPU_T와 CPU_R 및 Collection/Deployment를
//! 따로 선택하며, UCI reported nodes를 native node credit으로 바꾸지 않는다.

use super::{CheckerError, ExternalCheckerIdentity};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

pub const EXTERNAL_MIN_QUANTUM: Duration = Duration::from_millis(100);
pub const EXTERNAL_STOP_DRAIN_RESERVE: Duration = Duration::from_millis(500);
pub const EXTERNAL_MAX_REPORTED_NODE_REQUEST: u64 = 65_536;
pub const EXTERNAL_COOPERATIVE_CHECKER_VERSION: &str = "rz-external-uci-cooperative/0.1";
pub const STOCKFISH19_SOURCE_COMMIT: &str = "edb0d9db6731067ec50ce619ff372b463bc4dd5d";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuExecutionUse {
    Collection,
    Deployment,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuEndpoint {
    Disabled,
    Own,
    ExternalUci,
}

/// Immutable selections; choosing a teacher does not select the arena opponent.
/// Deployment's V-free restriction is consumed by the model/resource producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuEndpointSelection {
    pub execution_use: CpuExecutionUse,
    pub cpu_t: CpuEndpoint,
    pub cpu_r: CpuEndpoint,
    pub load_v: bool,
}

impl CpuEndpointSelection {
    pub const fn collection() -> Self {
        Self {
            execution_use: CpuExecutionUse::Collection,
            cpu_t: CpuEndpoint::ExternalUci,
            cpu_r: CpuEndpoint::Own,
            load_v: true,
        }
    }

    pub const fn deployment() -> Self {
        Self {
            execution_use: CpuExecutionUse::Deployment,
            cpu_t: CpuEndpoint::Disabled,
            cpu_r: CpuEndpoint::Own,
            load_v: false,
        }
    }

    pub fn validate(self) -> Result<(), CheckerError> {
        if self.execution_use == CpuExecutionUse::Deployment
            && (self.load_v || self.cpu_t != CpuEndpoint::Disabled)
        {
            return Err(CheckerError::Invalid(
                "deployment must not create/load/call V or CPU_T",
            ));
        }
        if self.cpu_r == CpuEndpoint::Disabled {
            return Err(CheckerError::Unsupported("PALS CPU_R disabled"));
        }
        Ok(())
    }

    pub fn validate_generation_collection(self) -> Result<(), CheckerError> {
        self.validate()?;
        if self.execution_use != CpuExecutionUse::Collection
            || self.cpu_t != CpuEndpoint::ExternalUci
            || !self.load_v
        {
            return Err(CheckerError::Invalid(
                "generation collection requires full P/C/V and external CPU_T",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalCpuAction {
    Analyze,
    RestrictRoot,
    PublishProgress,
    YieldPartial,
    ResumeTask,
    TransferTtOwnership,
    ChangeSelectivity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalCpuActionMask(u8);

impl ExternalCpuActionMask {
    pub const UCI_STOP_DRAIN: Self = Self(0b000_1111);
    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn supports(self, action: ExternalCpuAction) -> bool {
        self.0 & (1 << action as u8) != 0
    }

    pub fn require(self, action: ExternalCpuAction) -> Result<(), CheckerError> {
        if self.supports(action) {
            Ok(())
        } else {
            Err(CheckerError::Unsupported(match action {
                ExternalCpuAction::ResumeTask => "external UCI exact stack resume",
                ExternalCpuAction::TransferTtOwnership => "external UCI TT ownership transfer",
                ExternalCpuAction::ChangeSelectivity => "external UCI selectivity control",
                _ => "external UCI action",
            }))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalSlicePolicy {
    pub quantum: Duration,
    pub stop_drain_reserve: Duration,
    /// A requested UCI limit; an external engine can overshoot or omit reports.
    pub requested_reported_nodes: u64,
    pub multipv: u16,
}

impl Default for ExternalSlicePolicy {
    fn default() -> Self {
        Self {
            quantum: EXTERNAL_MIN_QUANTUM,
            stop_drain_reserve: EXTERNAL_STOP_DRAIN_RESERVE,
            requested_reported_nodes: EXTERNAL_MAX_REPORTED_NODE_REQUEST,
            multipv: 1,
        }
    }
}

impl ExternalSlicePolicy {
    pub fn validate(self, legal_candidate_count: usize) -> Result<(), CheckerError> {
        if self.quantum < EXTERNAL_MIN_QUANTUM
            || self.quantum > Duration::from_secs(180)
            || self.stop_drain_reserve < EXTERNAL_STOP_DRAIN_RESERVE
            || self.stop_drain_reserve > Duration::from_secs(180)
            || self.requested_reported_nodes == 0
            || self.requested_reported_nodes > EXTERNAL_MAX_REPORTED_NODE_REQUEST
            || self.multipv == 0
            || self.multipv > 4
            || usize::from(self.multipv) > legal_candidate_count
        {
            return Err(CheckerError::Invalid(
                "external quantum/reserve/nodes/MultiPV profile",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalControlAction {
    Continue,
    Yield,
    Cancel,
    /// Scope revoked without an issued cancel sequence; it cannot mint an ack.
    Revoked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalControlObservation {
    pub sequence: u64,
    pub action: ExternalControlAction,
}

/// Actual native-owner protocol boundaries. These clocks do not certify a
/// native completed iteration, an applied option, or exact-stack resumption.
#[derive(Clone, Copy, Debug)]
pub enum ExternalOwnerPhase {
    GoIssued {
        at: Instant,
        quantum_deadline: Instant,
    },
    StopSent {
        at: Instant,
    },
    BestMoveDrained {
        at: Instant,
    },
    ReadyBarrierCompleted {
        at: Instant,
    },
}

/// Adapter for the coordinator's atomic mailbox. No second mailbox or runtime
/// dependency is hidden in search; observations must already be job-sealed.
pub trait ExternalControl: Send + Sync {
    fn observe(&self) -> ExternalControlObservation;
    fn physical_fence_deadline(&self) -> Option<std::time::Instant> {
        None
    }
    /// Optional owner timing only. The default preserves existing control
    /// consumers; progress/terminal authority still comes from their receipts.
    fn observe_owner_phase(&self, _phase: ExternalOwnerPhase) {}
    /// Optional consistent raw tuple publication; does not stop external go or
    /// assert that its selective iteration is independently complete.
    fn publish_progress(
        &self,
        _opinion: &super::ExternalPolicyOpinion,
        _reported_nodes: Option<u64>,
    ) {
    }
}

/// Full means strength limiting is off, not unbounded computation.
pub fn stockfish19_full_options(multipv: u16) -> Result<BTreeMap<String, String>, CheckerError> {
    if !(1..=4).contains(&multipv) {
        return Err(CheckerError::Invalid(
            "Stockfish conditional MultiPV must be 1..4",
        ));
    }
    Ok([
        ("Skill Level", "20".to_owned()),
        ("UCI_LimitStrength", "false".to_owned()),
        ("Threads", "1".to_owned()),
        ("Hash", "256".to_owned()),
        ("Ponder", "false".to_owned()),
        ("NumaPolicy", "none".to_owned()),
        ("MultiPV", multipv.to_string()),
        ("UCI_Chess960", "false".to_owned()),
        ("SyzygyPath", "".to_owned()),
        ("SyzygyProbeLimit", "0".to_owned()),
        ("nodestime", "0".to_owned()),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value))
    .collect())
}

/// Identity pin admission; executable bytes are independently streamed before
/// spawn by the existing native ELF owner. Readyok is not option readback.
pub fn validate_stockfish19_full_identity(
    identity: &ExternalCheckerIdentity,
) -> Result<(), CheckerError> {
    identity.validate()?;
    if identity.declared_version != "19"
        || !identity.declared_source.contains(STOCKFISH19_SOURCE_COMMIT)
        || !identity.declared_license.contains("GPL")
    {
        return Err(CheckerError::Invalid(
            "Stockfish 19 version/source/license pin",
        ));
    }
    let multipv = identity
        .options
        .get("MultiPV")
        .and_then(|text| text.parse().ok())
        .ok_or(CheckerError::Invalid("Stockfish MultiPV declaration"))?;
    for (option, value) in stockfish19_full_options(multipv)? {
        if identity.options.get(&option) != Some(&value) {
            return Err(CheckerError::Invalid(
                "Stockfish Full option profile mismatch",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_never_claim_exact_resume_or_tt_transfer() {
        let mask = ExternalCpuActionMask::UCI_STOP_DRAIN;
        assert!(mask.supports(ExternalCpuAction::YieldPartial));
        assert!(mask.require(ExternalCpuAction::ResumeTask).is_err());
        assert!(
            mask.require(ExternalCpuAction::TransferTtOwnership)
                .is_err()
        );
        assert!(mask.require(ExternalCpuAction::ChangeSelectivity).is_err());
    }

    #[test]
    fn collection_and_deployment_selections_are_independent_and_v_free() {
        assert!(CpuEndpointSelection::collection().validate().is_ok());
        let mut selection = CpuEndpointSelection::deployment();
        selection.cpu_r = CpuEndpoint::ExternalUci;
        assert!(selection.validate().is_ok());
        selection.load_v = true;
        assert!(selection.validate().is_err());
    }

    #[test]
    fn external_profile_enforces_quantum_reserve_reported_limit_and_candidate_count() {
        let policy = ExternalSlicePolicy::default();
        assert!(policy.validate(1).is_ok());
        assert!(
            ExternalSlicePolicy {
                multipv: 4,
                ..policy
            }
            .validate(3)
            .is_err()
        );
        assert!(
            ExternalSlicePolicy {
                quantum: Duration::from_millis(99),
                ..policy
            }
            .validate(20)
            .is_err()
        );
        assert!(
            ExternalSlicePolicy {
                requested_reported_nodes: 65_537,
                ..policy
            }
            .validate(20)
            .is_err()
        );
    }

    #[test]
    fn stockfish_full_options_do_not_mean_unbounded_search() {
        let options = stockfish19_full_options(4).unwrap();
        assert_eq!(options["Threads"], "1");
        assert_eq!(options["UCI_LimitStrength"], "false");
        assert_eq!(options["Skill Level"], "20");
        assert_eq!(options["SyzygyProbeLimit"], "0");
        assert!(stockfish19_full_options(5).is_err());
    }
}
