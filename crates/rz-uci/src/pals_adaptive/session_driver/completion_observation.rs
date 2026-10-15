//! Bounded final-return diagnostics. These observations grant no search,
//! output, reset, physical-fence, or acceptance authority.
use crate::search_driver::{DriverPhysicalCompletion, SearchSessionFailure};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::{Digest, ProcessEpoch};
use rz_search::pals::engine::PalsCompletion;

const FAILURE_DETAIL_BYTES_MAX: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveCompletionReturn {
    Ok,
    Err,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveCompletionFailure {
    pub code: &'static str,
    pub detail: String,
    pub detail_truncated: bool,
    pub physical_completion: DriverPhysicalCompletion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveCompletionObservation {
    pub authority: SearchAuthority,
    /// Copied only from the actual final PalsResult. A later driver error does
    /// not erase it or turn it into a successful return.
    pub semantic_completion: Option<PalsCompletion>,
    pub driver_return: AdaptiveCompletionReturn,
    pub failure: Option<AdaptiveCompletionFailure>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdaptiveCompletionReasonCounts {
    pub round_limit: u64,
    pub cpu_node_limit: u64,
    pub role_call_limit: u64,
    pub deadline: u64,
    pub canceled: u64,
    pub capacity: u64,
    pub terminal: u64,
}
impl AdaptiveCompletionReasonCounts {
    fn add(&mut self, completion: PalsCompletion) -> Option<()> {
        let count = match completion {
            PalsCompletion::RoundLimit => &mut self.round_limit,
            PalsCompletion::CpuNodeLimit => &mut self.cpu_node_limit,
            PalsCompletion::RoleCallLimit => &mut self.role_call_limit,
            PalsCompletion::Deadline => &mut self.deadline,
            PalsCompletion::Canceled => &mut self.canceled,
            PalsCompletion::Capacity => &mut self.capacity,
            PalsCompletion::Terminal => &mut self.terminal,
        };
        *count = count.checked_add(1)?;
        Some(())
    }
}

/// One last return plus fixed counters; no per-go, per-ticket or per-node
/// history. Null totals mean unavailable, never zero or an inferred reason.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveCompletionDiagnostic {
    pub process_epoch: ProcessEpoch,
    pub implementation: Digest,
    pub last: Option<AdaptiveCompletionObservation>,
    pub observed_driver_returns: Option<u64>,
    pub successful_driver_return_reason_counts: Option<AdaptiveCompletionReasonCounts>,
    pub semantic_observed_but_driver_failed: Option<u64>,
    pub unobserved_completion_returns: Option<u64>,
    pub unavailable_reason: Option<&'static str>,
}

pub(super) struct CompletionJournal {
    diagnostic: AdaptiveCompletionDiagnostic,
    last_authority: Option<SearchAuthority>,
}
impl CompletionJournal {
    pub(super) fn new(process_epoch: ProcessEpoch, implementation: Digest) -> Self {
        Self {
            diagnostic: AdaptiveCompletionDiagnostic {
                process_epoch,
                implementation,
                last: None,
                observed_driver_returns: Some(0),
                successful_driver_return_reason_counts: Some(Default::default()),
                semantic_observed_but_driver_failed: Some(0),
                unobserved_completion_returns: Some(0),
                unavailable_reason: None,
            },
            last_authority: None,
        }
    }
    pub(super) fn snapshot(&self) -> AdaptiveCompletionDiagnostic {
        self.diagnostic.clone()
    }
    fn invalidate(&mut self, reason: &'static str) {
        self.diagnostic.observed_driver_returns = None;
        self.diagnostic.successful_driver_return_reason_counts = None;
        self.diagnostic.semantic_observed_but_driver_failed = None;
        self.diagnostic.unobserved_completion_returns = None;
        self.diagnostic.unavailable_reason.get_or_insert(reason);
    }
    pub(super) fn record(
        &mut self,
        authority: SearchAuthority,
        semantic_completion: Option<PalsCompletion>,
        returned: Result<(), &SearchSessionFailure>,
    ) {
        let failure = returned.err().map(|error| {
            let detail = bounded(&error.detail, FAILURE_DETAIL_BYTES_MAX);
            AdaptiveCompletionFailure {
                code: error.code,
                detail: detail.to_owned(),
                detail_truncated: detail.len() != error.detail.len(),
                physical_completion: error.physical_completion,
            }
        });
        let driver_return = if failure.is_some() {
            AdaptiveCompletionReturn::Err
        } else {
            AdaptiveCompletionReturn::Ok
        };
        self.diagnostic.last = Some(AdaptiveCompletionObservation {
            authority,
            semantic_completion,
            driver_return,
            failure,
        });
        if authority.epoch != self.diagnostic.process_epoch
            || authority.implementation != self.diagnostic.implementation
            || authority.root.0 == 0
            || self
                .last_authority
                .is_some_and(|old| authority.game.0 < old.game.0 || authority.root.0 <= old.root.0)
        {
            self.invalidate("foreign_duplicate_or_out_of_order_authority");
            return;
        }
        self.last_authority = Some(authority);
        if self.diagnostic.unavailable_reason.is_some() {
            return;
        }
        // A checked copy makes overflow transactional. The last actual return
        // above survives even when cumulative totals become unavailable.
        let mut next = self.diagnostic.clone();
        let added = (|| {
            next.observed_driver_returns = Some(next.observed_driver_returns?.checked_add(1)?);
            match (semantic_completion, driver_return) {
                (Some(completion), AdaptiveCompletionReturn::Ok) => {
                    next.successful_driver_return_reason_counts
                        .as_mut()?
                        .add(completion)?;
                }
                (Some(_), AdaptiveCompletionReturn::Err) => {
                    next.semantic_observed_but_driver_failed =
                        Some(next.semantic_observed_but_driver_failed?.checked_add(1)?);
                }
                (None, _) => {
                    next.unobserved_completion_returns =
                        Some(next.unobserved_completion_returns?.checked_add(1)?);
                }
            }
            Some(())
        })();
        if added.is_some() {
            self.diagnostic = next;
        } else {
            self.invalidate("counter_overflow");
        }
    }
}

fn bounded(text: &str, max: usize) -> &str {
    let mut end = text.len().min(max);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[cfg(feature = "search-work-receipts")]
fn completion_name(value: PalsCompletion) -> &'static str {
    match value {
        PalsCompletion::RoundLimit => "round_limit",
        PalsCompletion::CpuNodeLimit => "cpu_node_limit",
        PalsCompletion::RoleCallLimit => "role_call_limit",
        PalsCompletion::Deadline => "deadline",
        PalsCompletion::Canceled => "canceled",
        PalsCompletion::Capacity => "capacity",
        PalsCompletion::Terminal => "terminal",
    }
}

#[cfg(feature = "search-work-receipts")]
impl AdaptiveCompletionDiagnostic {
    /// Independent raw diagnostic only. Existing Work and V1-V4 envelopes are
    /// unchanged; neither an Ok return nor a reason grants output acceptance.
    pub fn as_json(&self) -> serde_json::Value {
        let hex = |digest: Digest| {
            digest
                .0
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        let counts = self.successful_driver_return_reason_counts.map(|counts| {
            serde_json::json!({
                "round_limit":counts.round_limit,"cpu_node_limit":counts.cpu_node_limit,
                "role_call_limit":counts.role_call_limit,"deadline":counts.deadline,
                "canceled":counts.canceled,"capacity":counts.capacity,"terminal":counts.terminal
            })
        });
        serde_json::json!({
            "schema":"rz-pals-semantic-completion-diagnostic/1",
            "scope":"latest returned adaptive run's final semantic result and driver return; not an active-root observation",
            "status":if self.unavailable_reason.is_some() {"unavailable"} else if self.last.is_none() {"not_observed"} else {"observed"},
            "process_epoch":self.process_epoch.0,
            "implementation_sha256":hex(self.implementation),
            "last":self.last.as_ref().map(|last| serde_json::json!({
                "authority":{
                    "process_epoch":last.authority.epoch.0,
                    "game_generation":last.authority.game.0,
                    "root_generation":last.authority.root.0,
                    "implementation_sha256":hex(last.authority.implementation)
                },
                "semantic_completion":last.semantic_completion.map(completion_name),
                "semantic_observation_stage":if last.semantic_completion.is_some() {"final_semantic_result"} else {"not_observed"},
                "driver_return":match last.driver_return {AdaptiveCompletionReturn::Ok=>"ok",AdaptiveCompletionReturn::Err=>"err"},
                "failure_code":last.failure.as_ref().map(|failure| bounded(failure.code,128)),
                "failure_code_truncated":last.failure.as_ref().map(|failure| failure.code.len()>128),
                "failure_detail":last.failure.as_ref().map(|failure| failure.detail.as_str()),
                "failure_detail_truncated":last.failure.as_ref().map(|failure| failure.detail_truncated),
                "driver_physical_completion":last.failure.as_ref().map(|failure| match failure.physical_completion {
                    DriverPhysicalCompletion::Confirmed=>"confirmed",DriverPhysicalCompletion::Unknown=>"unknown"
                }),
                "physical_scope":"original driver failure classification only; success does not synthesize a physical fence"
            })),
            "observed_driver_returns":self.observed_driver_returns,
            "successful_driver_return_reason_counts":counts,
            "semantic_observed_but_driver_failed":self.semantic_observed_but_driver_failed,
            "unobserved_completion_returns":self.unobserved_completion_returns,
            "unavailable_reason":self.unavailable_reason,
            "grants_output_or_acceptance_authority":false
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::{GameGeneration, RootGeneration};

    fn authority(root: u64) -> SearchAuthority {
        SearchAuthority {
            epoch: ProcessEpoch(31),
            game: GameGeneration(0),
            root: RootGeneration(root),
            implementation: Digest([4; 32]),
        }
    }
    fn journal() -> CompletionJournal {
        CompletionJournal::new(ProcessEpoch(31), Digest([4; 32]))
    }
    #[test]
    fn final_reason_keeps_later_failure_and_next_root_has_no_stale_reason() {
        let mut journal = journal();
        let failure = SearchSessionFailure::physical_completion_unknown();
        journal.record(
            authority(1),
            Some(PalsCompletion::RoleCallLimit),
            Err(&failure),
        );
        let actual = journal.snapshot();
        assert_eq!(actual.semantic_observed_but_driver_failed, Some(1));
        assert_eq!(
            actual
                .successful_driver_return_reason_counts
                .unwrap()
                .role_call_limit,
            0
        );
        let last = actual.last.unwrap();
        assert_eq!(
            last.semantic_completion,
            Some(PalsCompletion::RoleCallLimit)
        );
        assert_eq!(last.driver_return, AdaptiveCompletionReturn::Err);
        assert_eq!(
            last.failure.unwrap().physical_completion,
            DriverPhysicalCompletion::Unknown
        );
        journal.record(authority(2), None, Err(&failure));
        let actual = journal.snapshot();
        assert_eq!(actual.last.unwrap().semantic_completion, None);
        assert_eq!(actual.observed_driver_returns, Some(2));
        assert_eq!(actual.unobserved_completion_returns, Some(1));
    }
    #[test]
    fn duplicate_reordered_and_foreign_scopes_make_totals_unavailable() {
        for rejected in [
            authority(1),
            authority(0),
            SearchAuthority {
                epoch: ProcessEpoch(32),
                ..authority(2)
            },
            SearchAuthority {
                implementation: Digest([8; 32]),
                ..authority(2)
            },
        ] {
            let mut journal = journal();
            journal.record(authority(1), Some(PalsCompletion::Terminal), Ok(()));
            journal.record(rejected, None, Ok(()));
            let actual = journal.snapshot();
            assert!(actual.unavailable_reason.is_some());
            assert_eq!(actual.observed_driver_returns, None);
            assert_eq!(actual.successful_driver_return_reason_counts, None);
            assert_eq!(actual.last.unwrap().semantic_completion, None);
        }
    }
    #[test]
    fn overflow_is_unavailable_and_utf8_failure_detail_is_bounded() {
        let mut journal = journal();
        journal.diagnostic.observed_driver_returns = Some(u64::MAX);
        let failure = SearchSessionFailure {
            code: "fixture",
            detail: "가".repeat(1000),
            physical_completion: DriverPhysicalCompletion::Confirmed,
        };
        journal.record(authority(1), Some(PalsCompletion::Capacity), Err(&failure));
        let actual = journal.snapshot();
        assert_eq!(actual.unavailable_reason, Some("counter_overflow"));
        assert_eq!(actual.observed_driver_returns, None);
        let failure = actual.last.unwrap().failure.unwrap();
        assert!(failure.detail.len() <= FAILURE_DETAIL_BYTES_MAX);
        assert!(failure.detail_truncated);
    }
    #[cfg(feature = "search-work-receipts")]
    #[test]
    fn seven_completion_names_are_exhaustive_and_diagnostic_never_grants_acceptance() {
        for (index, (reason, name)) in [
            (PalsCompletion::RoundLimit, "round_limit"),
            (PalsCompletion::CpuNodeLimit, "cpu_node_limit"),
            (PalsCompletion::RoleCallLimit, "role_call_limit"),
            (PalsCompletion::Deadline, "deadline"),
            (PalsCompletion::Canceled, "canceled"),
            (PalsCompletion::Capacity, "capacity"),
            (PalsCompletion::Terminal, "terminal"),
        ]
        .into_iter()
        .enumerate()
        {
            let mut journal = journal();
            journal.record(authority(index as u64 + 1), Some(reason), Ok(()));
            let raw = journal.snapshot().as_json();
            assert_eq!(raw["last"]["semantic_completion"], name);
            assert_eq!(raw["successful_driver_return_reason_counts"][name], 1);
            assert_eq!(raw["grants_output_or_acceptance_authority"], false);
            assert!(raw["last"]["driver_physical_completion"].is_null());
        }
    }
}
