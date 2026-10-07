//! Bounded helper observations, independent of Native NN completion.
//! Registration identifies declarations; a reported score or readyok does not
//! certify a Rules result or the actual application of every requested option.

use crate::search_driver::{PalsSessionDriver, SearchSessionFailure};
use rz_search::cpu_checker::{
    CheckerAttempt, CheckerIdentity, CheckerShutdown, ExternalBound, ExternalCompletion,
    ExternalRawScore,
};
use rz_search::pals::engine::RoleModel;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};

#[derive(Clone, Debug, Serialize)]
pub struct PalsCheckerProcessReceipt {
    schema_version: u32,
    domain: &'static str,
    profile_file_sha256: String,
    profile_canonical_sha256: String,
    registration_sha256: String,
    registration: Value,
    startup_handshake_completed: bool,
    observed_uci: Option<Value>,
    /// This startup barrier verified advertised types/ranges and sent options.
    /// UCI does not confirm each applied value; that observation stays unknown.
    requested_option_checks_completed: bool,
    applied_option_values: &'static str,
    latest_attempt: Option<Value>,
    shutdown: Option<Value>,
    cleanup_complete: bool,
    started_owner_exit_and_drains_confirmed: bool,
}

impl PalsCheckerProcessReceipt {
    pub fn observe<M: RoleModel + 'static>(
        driver: &PalsSessionDriver<M>,
        profile_file_sha256: &str,
        profile_canonical_sha256: &str,
    ) -> Result<Self, SearchSessionFailure> {
        for digest in [profile_file_sha256, profile_canonical_sha256] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(SearchSessionFailure::debug(
                    "PalsCheckerReceipt",
                    &"canonical profile digest required",
                ));
            }
        }
        let selected = driver.checker_registration();
        if !matches!(selected.identity, CheckerIdentity::ExternalUci(_)) {
            return Err(SearchSessionFailure::debug(
                "PalsCheckerReceipt",
                &"external helper observation cannot attest an own checker",
            ));
        }
        let caps = selected.capabilities;
        let model_value = selected.model_value.as_ref().map(|value| {
            json!({
                "semantics": value.semantics,
                "model": value.model,
                "encoding": value.encoding,
                "precision": value.precision,
                "model_epoch": value.model_epoch,
            })
        });
        let registration = json!({
            "identity": selected.identity,
            "conditions": selected.conditions,
            "capabilities": {
                "max_depth": caps.max_depth,
                "max_prefix_plies": caps.max_prefix_plies,
                "max_root_moves": caps.max_root_moves,
                "root_moves": caps.root_moves,
                "divergence": caps.divergence,
                "resume": caps.resume,
                "selective_search": caps.selective_search,
            },
            "role_model": selected.role_model,
            "model_value": model_value,
            "resolver_version": selected.resolver_version,
            "resolver_semantics": selected.resolver_semantics,
        });
        let mut digest = Sha256::new();
        digest.update(b"rz-pals-checker-registration/1\0");
        digest.update(serde_json::to_vec(&registration).map_err(|error| {
            SearchSessionFailure::debug("PalsCheckerRegistrationEncoding", &error)
        })?);
        let shutdown = driver.checker_shutdown()?;
        let started = driver.checker_started();
        Ok(Self {
            schema_version: 1,
            domain: "rz-pals-checker-process/1",
            profile_file_sha256: profile_file_sha256.into(),
            profile_canonical_sha256: profile_canonical_sha256.into(),
            registration_sha256: format!("{:x}", digest.finalize()),
            registration,
            startup_handshake_completed: started,
            observed_uci: driver
                .checker_startup_uci()?
                .map(|observed| json!({"name": observed.name, "author": observed.author})),
            requested_option_checks_completed: started,
            applied_option_values: "unknown",
            latest_attempt: driver
                .checker_last_attempt()?
                .as_ref()
                .map(attempt)
                .transpose()?,
            shutdown: shutdown.map(process),
            cleanup_complete: shutdown.is_some_and(|state| {
                state.cleanup_complete && !state.quarantined && !state.ownership_lost
            }),
            started_owner_exit_and_drains_confirmed: shutdown.is_some_and(|state| {
                state.cleanup_complete
                    && state.exit_observed
                    && state.stdout_drained
                    && state.stderr_drained
                    && !state.quarantined
                    && !state.ownership_lost
            }),
        })
    }

    pub(super) fn same_registration(&self, other: &Self) -> bool {
        self.profile_file_sha256 == other.profile_file_sha256
            && self.profile_canonical_sha256 == other.profile_canonical_sha256
            && self.registration_sha256 == other.registration_sha256
            && self.registration == other.registration
    }
    pub(super) fn cleanup_complete(&self) -> bool {
        self.startup_handshake_completed
            && self.cleanup_complete
            && self.started_owner_exit_and_drains_confirmed
    }
}

fn process(state: CheckerShutdown) -> Value {
    json!({
        "process_identity": state.process_identity.map(|identity| json!({
            "pid": identity.pid,
            "process_group": identity.process_group,
            "proc_start_ticks": identity.proc_start_ticks,
            "scope": "linux_spawn_observed_identity",
        })),
        "stop_sent": state.stop_sent,
        "quit_sent": state.quit_sent,
        "exit_observed": state.exit_observed,
        "stdout_drained": state.stdout_drained,
        "stderr_drained": state.stderr_drained,
        "exit_code": state.exit_code,
        "exit_signal": state.exit_signal,
        "cleanup_complete": state.cleanup_complete,
        "quarantined": state.quarantined,
        "ownership_lost": state.ownership_lost,
        "stdout_bytes": state.stdout_bytes,
        "stderr_bytes": state.stderr_bytes,
    })
}

fn attempt(observed: &CheckerAttempt) -> Result<Value, SearchSessionFailure> {
    let elapsed_us = u64::try_from(observed.elapsed.as_micros())
        .map_err(|error| SearchSessionFailure::debug("PalsCheckerElapsedOverflow", &error))?;
    let external = observed.external.as_ref().map(|external| {
        let report = external.partial_report.as_ref().map(|report| {
            let score = match report.score {
                ExternalRawScore::Centipawns(value) => json!({"kind": "foreign_cp", "value": value}),
                ExternalRawScore::MateMoves(value) => json!({"kind": "foreign_mate_moves", "value": value}),
                ExternalRawScore::Unknown => json!({"kind": "unknown", "value": null}),
            };
            json!({
                "identity": report.identity,
                "observed_uci": {"name": report.observed_uci.name, "author": report.observed_uci.author},
                "request_id": report.request_id,
                "best_move": report.best_move.map(|movement| movement.to_string()),
                "pv": report.pv.iter().map(|movement| movement.to_string()).collect::<Vec<_>>(),
                "score": score,
                "bound": match report.bound { ExternalBound::ExactReported => "exact_reported", ExternalBound::Lower => "lower", ExternalBound::Upper => "upper", ExternalBound::Unknown => "unknown" },
                "wdl_per_mille": report.wdl_per_mille,
                "perspective": format!("{:?}", report.perspective),
                "requested_depth": report.requested_depth,
                "reported_depth": report.reported_depth,
                "seldepth": report.seldepth,
                "root_restricted": report.root_restricted,
                "completion": match report.completion { ExternalCompletion::Pending => "pending", ExternalCompletion::BestMove => "best_move", ExternalCompletion::StoppedDeadline => "stopped_deadline", ExternalCompletion::StoppedCanceled => "stopped_canceled" },
                "nodes": report.work.nodes,
                "qnodes": report.work.qnodes,
                "tt_hits": report.work.tt_hits,
            })
        });
        json!({"request_id": external.request_id, "report": report, "process": process(external.process)})
    });
    Ok(json!({
        "elapsed_us": elapsed_us,
        "nodes": observed.work.nodes,
        "qnodes": observed.work.qnodes,
        "tt_hits": observed.work.tt_hits,
        "external": external,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_position::{BoardMove, Color};
    use rz_search::cpu_checker::{
        CheckerWork, ExternalAttemptEvidence, ExternalCheckerIdentity, ExternalCheckerReport,
        ExternalModelMetadata, ExternalTrainingKnowledge, ExternalUciIdentity,
    };
    use std::{collections::BTreeMap, time::Duration};

    #[test]
    fn unknown_work_and_live_idle_process_are_not_reported_as_completed_cleanup() {
        let value = attempt(&CheckerAttempt {
            work: CheckerWork::default(),
            elapsed: Duration::from_micros(1),
            external: Some(ExternalAttemptEvidence {
                request_id: 0,
                partial_report: None,
                process: CheckerShutdown::default(),
            }),
        })
        .unwrap();
        assert!(value["nodes"].is_null());
        assert!(value["external"]["report"].is_null());
        assert_eq!(value["external"]["process"]["exit_observed"], false);
        assert_eq!(value["external"]["process"]["cleanup_complete"], false);
        assert_eq!(value["external"]["process"]["quarantined"], false);
        assert!(value["external"]["process"]["process_identity"].is_null());
    }

    #[test]
    fn historical_linux_spawn_identity_is_separate_from_exit_and_drain_evidence() {
        let mut state = CheckerShutdown {
            process_identity: Some(rz_search::cpu_checker::ExternalProcessIdentity {
                pid: 123,
                process_group: 123,
                proc_start_ticks: 456,
            }),
            ..CheckerShutdown::default()
        };
        let active = process(state);
        assert_eq!(active["process_identity"]["pid"], 123);
        assert_eq!(active["process_identity"]["process_group"], 123);
        assert_eq!(active["process_identity"]["proc_start_ticks"], 456);
        assert_eq!(active["cleanup_complete"], false);
        state.exit_observed = true;
        state.stdout_drained = true;
        state.stderr_drained = true;
        state.cleanup_complete = true;
        let ended = process(state);
        assert_eq!(ended["process_identity"], active["process_identity"]);
        assert_eq!(ended["cleanup_complete"], true);
        assert!(process(CheckerShutdown::owned_no_process())["process_identity"].is_null());
    }

    #[test]
    fn foreign_cp_bound_and_stopped_report_preserve_raw_scope_without_wdl_conversion() {
        let work = CheckerWork {
            nodes: Some(0),
            qnodes: None,
            tt_hits: None,
        };
        let report = ExternalCheckerReport {
            identity: ExternalCheckerIdentity {
                adapter_semantics: rz_search::external_cpu::EXTERNAL_UCI_SCORE_SEMANTICS.into(),
                binary_sha256: "a".repeat(64),
                launch_arguments_sha256: "b".repeat(64),
                declared_name: "fixture".into(),
                declared_version: "fixture".into(),
                declared_source: "fixture".into(),
                declared_license: "fixture".into(),
                options: BTreeMap::new(),
                assets: Vec::new(),
                model_metadata: ExternalModelMetadata {
                    weights_sha256: None,
                    training: ExternalTrainingKnowledge::Unknown,
                    declared_rights: None,
                    precision: None,
                },
            },
            observed_uci: ExternalUciIdentity {
                name: "actual fixture response".into(),
                author: None,
            },
            request_id: 7,
            best_move: Some(BoardMove::from_uci("e2e4").unwrap()),
            pv: vec![BoardMove::from_uci("e2e4").unwrap()],
            score: ExternalRawScore::Centipawns(20_000),
            bound: ExternalBound::Lower,
            wdl_per_mille: None,
            perspective: Color::White,
            requested_depth: 9,
            reported_depth: Some(1),
            seldepth: None,
            root_restricted: true,
            completion: ExternalCompletion::StoppedDeadline,
            work,
            elapsed: Duration::from_micros(2),
        };
        let value = attempt(&CheckerAttempt {
            work,
            elapsed: report.elapsed,
            external: Some(ExternalAttemptEvidence {
                request_id: 7,
                partial_report: Some(report),
                process: CheckerShutdown::default(),
            }),
        })
        .unwrap();
        let report = &value["external"]["report"];
        assert_eq!(
            report["score"],
            json!({"kind": "foreign_cp", "value": 20_000})
        );
        assert_eq!(report["bound"], "lower");
        assert_eq!(report["completion"], "stopped_deadline");
        assert_eq!(report["requested_depth"], 9);
        assert_eq!(report["reported_depth"], 1);
        assert!(report["wdl_per_mille"].is_null());
        assert_eq!(report["nodes"], 0);
        assert!(report["qnodes"].is_null());
        assert!(report.get("rules_proof").is_none());
    }
}
