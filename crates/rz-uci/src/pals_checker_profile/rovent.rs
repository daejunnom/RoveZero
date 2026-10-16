//! Rovent용 새 엔진 프로필. 기존 Stockfish V1/V5 schema와 canonical bytes는
//! 그대로 두고, 엔진·실행 목적·query capability를 별도 선언에 고정한다.
//! 이 모듈의 admission은 프로세스를 시작하거나 자산을 내려받지 않는다.

use super::{
    BoundedJson, MAX_CHECKER_PROFILE_BYTES, UniqueJsonKeys, invalid, linux_path, sha, sorted_value,
};
use rz_search::cpu_checker::{CheckerError, ExternalCheckerIdentity};
use rz_search::external_cpu::ExternalCpuConfig;
use rz_search::external_cpu::{
    ExternalEngineProfile, ExternalEnginePurpose, ExternalGoConstraint, ExternalUciCpuChecker,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ROVENT_EXTERNAL_PROFILE_SCHEMA: &str = "rz-rovent-external-engine-profile/1";
pub const ROVENT_EXTERNAL_REGISTRATION_SCHEMA: &str = "rz-rovent-external-engine-registration/1";
const CANONICAL_DOMAIN: &str = "rz-rovent-external-engine-profile-canonical/1";
pub const ROVENT_EXTERNAL_CAPABILITY_DOMAIN: &str = "rz-rovent-external-capabilities/1";

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoventExternalProfile {
    pub schema: String,
    pub engine: ExternalEngineProfile,
    pub purpose: ExternalEnginePurpose,
    pub program: String,
    pub working_directory: String,
    pub arguments: Vec<String>,
    pub identity: ExternalCheckerIdentity,
    pub expected_uci_name: String,
    pub max_depth: u16,
    pub max_prefix_plies: usize,
    pub handshake_timeout_ms: u64,
    pub max_task_wall_time_ms: u64,
    pub stop_grace_ms: u64,
    pub shutdown_grace_ms: u64,
    pub max_output_bytes: usize,
    pub max_line_bytes: usize,
}

#[derive(Debug)]
pub struct LoadedRoventExternalProfile {
    profile: RoventExternalProfile,
    file_sha256: String,
    canonical_sha256: String,
    file_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct RoventExternalProfileRegistration<'a> {
    pub schema: &'static str,
    pub file_sha256: &'a str,
    pub canonical_sha256: &'a str,
    pub file_bytes: u64,
    pub declaration: &'a RoventExternalProfile,
    pub declaration_only: bool,
    pub process_start_performed: bool,
    pub options_application_observed: Option<bool>,
    pub model_loading_observed: Option<bool>,
}

impl RoventExternalProfile {
    /// Canonical source capability, independent of actual startup/ISA/options
    /// application. Profile canonical bytes separately bind requested options.
    pub fn capability_sha256(&self) -> Result<String, CheckerError> {
        self.validate()?;
        let caps = self.engine.capabilities();
        let bytes = serde_json::to_vec(&(
            ROVENT_EXTERNAL_CAPABILITY_DOMAIN,
            self.engine,
            self.purpose,
            self.engine.source_commit(),
            &self.expected_uci_name,
            rz_search::external_cpu::EXTERNAL_UCI_SCORE_SEMANTICS,
            (
                caps.searchmoves,
                caps.combined_depth_nodes_movetime,
                caps.ponder,
                caps.embedded_nnue,
                caps.separate_assets,
                caps.exact_stack_resume,
            ),
        ))
        .map_err(|_| invalid("cannot encode Rovent external capabilities"))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    /// Caller owns descriptor retention and bounded reads. Exact pinned bytes,
    /// duplicate keys, schema and all semantic bounds are checked before any
    /// executable/path access; legacy schemas are not automatically converted.
    pub fn from_verified_bytes(
        bytes: &[u8],
        expected_file_sha256: &str,
    ) -> Result<LoadedRoventExternalProfile, CheckerError> {
        if !sha(expected_file_sha256) || bytes.is_empty() || bytes.len() > MAX_CHECKER_PROFILE_BYTES
        {
            return Err(invalid("Rovent external profile byte/hash bounds"));
        }
        let file_sha256 = format!("{:x}", Sha256::digest(bytes));
        if file_sha256 != expected_file_sha256 {
            return Err(invalid("Rovent external profile file SHA256 mismatch"));
        }
        serde_json::from_slice::<UniqueJsonKeys>(bytes)
            .map_err(|_| invalid("invalid JSON or duplicate Rovent external profile key"))?;
        let profile: Self = serde_json::from_slice(bytes)
            .map_err(|_| invalid("invalid Rovent external profile JSON"))?;
        profile.validate()?;
        let canonical_sha256 = profile.canonical_digest()?;
        Ok(LoadedRoventExternalProfile {
            profile,
            file_sha256,
            canonical_sha256,
            file_bytes: bytes.len() as u64,
        })
    }

    pub fn validate(&self) -> Result<(), CheckerError> {
        if self.schema != ROVENT_EXTERNAL_PROFILE_SCHEMA {
            return Err(invalid("unsupported Rovent external profile schema"));
        }
        self.engine
            .validate_identity(self.purpose, &self.identity, &self.expected_uci_name)?;
        if !self.arguments.is_empty() {
            return Err(CheckerError::Unsupported(
                "embedded Rovent engine profile admits no engine argv",
            ));
        }
        if !linux_path(&self.program, false) || !linux_path(&self.working_directory, true) {
            return Err(invalid(
                "Rovent external profile requires absolute Linux paths",
            ));
        }
        if self.max_depth == 0 || self.max_depth > 64 || self.max_prefix_plies > 4096 {
            return Err(invalid("Rovent external profile depth/history bounds"));
        }
        if [
            self.handshake_timeout_ms,
            self.max_task_wall_time_ms,
            self.stop_grace_ms,
            self.shutdown_grace_ms,
        ]
        .into_iter()
        .any(|milliseconds| milliseconds == 0 || milliseconds > 180_000)
            || self.max_task_wall_time_ms <= self.stop_grace_ms
        {
            return Err(invalid("Rovent external profile finite task/stop budgets"));
        }
        if self.max_output_bytes == 0
            || self.max_output_bytes > 16 * 1024 * 1024
            || self.max_line_bytes == 0
            || self.max_line_bytes > 65_536
            || self.max_line_bytes > self.max_output_bytes
        {
            return Err(invalid("Rovent external profile output bounds"));
        }
        Ok(())
    }

    pub fn require_query(
        &self,
        constraint: ExternalGoConstraint,
        root_restricted: bool,
    ) -> Result<(), CheckerError> {
        self.validate()?;
        if self.purpose == ExternalEnginePurpose::ArenaOpponent {
            return Err(CheckerError::Unsupported(
                "arena profile cannot authorize a teacher/research query",
            ));
        }
        self.engine.require_query(constraint, root_restricted)?;
        if constraint
            .requested_depth()
            .is_some_and(|depth| depth > self.max_depth)
        {
            return Err(invalid("Rovent query depth exceeds declared profile cap"));
        }
        Ok(())
    }

    fn canonical_digest(&self) -> Result<String, CheckerError> {
        let value = sorted_value(
            serde_json::to_value(self)
                .map_err(|_| invalid("cannot encode Rovent external profile declaration"))?,
        );
        let mut output = BoundedJson { bytes: Vec::new() };
        serde_json::to_writer(&mut output, &(CANONICAL_DOMAIN, value))
            .map_err(|_| invalid("canonical Rovent external profile exceeds byte bound"))?;
        Ok(format!("{:x}", Sha256::digest(&output.bytes)))
    }
}

impl LoadedRoventExternalProfile {
    pub fn profile(&self) -> &RoventExternalProfile {
        &self.profile
    }

    pub fn registration(&self) -> RoventExternalProfileRegistration<'_> {
        RoventExternalProfileRegistration {
            schema: ROVENT_EXTERNAL_REGISTRATION_SCHEMA,
            file_sha256: &self.file_sha256,
            canonical_sha256: &self.canonical_sha256,
            file_bytes: self.file_bytes,
            declaration: &self.profile,
            declaration_only: true,
            process_start_performed: false,
            options_application_observed: None,
            model_loading_observed: None,
        }
    }

    /// Configuration only. No executable is read or process started here.
    pub fn configuration(&self) -> Result<ExternalCpuConfig, CheckerError> {
        self.profile.validate()?;
        #[cfg(target_os = "linux")]
        {
            use std::{path::PathBuf, time::Duration};
            let p = &self.profile;
            let config = ExternalCpuConfig {
                program: PathBuf::from(&p.program),
                arguments: Vec::new(),
                working_directory: PathBuf::from(&p.working_directory),
                identity: p.identity.clone(),
                expected_uci_name: Some(p.expected_uci_name.clone()),
                max_depth: p.max_depth,
                max_prefix_plies: p.max_prefix_plies,
                handshake_timeout: Duration::from_millis(p.handshake_timeout_ms),
                max_task_wall_time: Duration::from_millis(p.max_task_wall_time_ms),
                stop_grace: Duration::from_millis(p.stop_grace_ms),
                shutdown_grace: Duration::from_millis(p.shutdown_grace_ms),
                max_output_bytes: p.max_output_bytes,
                max_line_bytes: p.max_line_bytes,
            };
            config.validate()?;
            Ok(config)
        }
        #[cfg(not(target_os = "linux"))]
        Err(CheckerError::Unsupported(
            "Rovent external checker requires Linux",
        ))
    }

    /// Creates an unstarted checker using the existing native process owner.
    /// Arena execution belongs to its independent arena launch path.
    pub fn checker(&self) -> Result<ExternalUciCpuChecker, CheckerError> {
        self.profile.validate()?;
        if self.profile.purpose == ExternalEnginePurpose::ArenaOpponent {
            return Err(CheckerError::Unsupported(
                "arena opponent must use the arena process owner",
            ));
        }
        ExternalUciCpuChecker::create_profiled(
            self.configuration()?,
            self.profile.engine,
            self.profile.purpose,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_search::cpu_checker::{ExternalModelMetadata, ExternalTrainingKnowledge};
    use rz_search::external_cpu::{ExternalEngineResources, arguments_sha256};

    fn profile() -> RoventExternalProfile {
        let engine = ExternalEngineProfile::Reckless090EmbeddedNnue;
        let purpose = ExternalEnginePurpose::ResearchChecker;
        RoventExternalProfile {
            schema: ROVENT_EXTERNAL_PROFILE_SCHEMA.into(),
            engine,
            purpose,
            program: "/opt/external/reckless".into(),
            working_directory: "/opt/external".into(),
            arguments: Vec::new(),
            identity: ExternalCheckerIdentity {
                adapter_semantics: rz_search::external_cpu::EXTERNAL_UCI_SCORE_SEMANTICS.into(),
                binary_sha256: "a".repeat(64),
                launch_arguments_sha256: arguments_sha256(&[]),
                declared_name: engine.expected_uci_name().into(),
                declared_version: "0.9.0".into(),
                declared_source: format!(
                    "{}/tree/{}",
                    engine.source_repository(),
                    engine.source_commit()
                ),
                declared_license: engine.license().into(),
                options: engine
                    .options(purpose, ExternalEngineResources::default())
                    .unwrap(),
                assets: Vec::new(),
                model_metadata: ExternalModelMetadata {
                    weights_sha256: None,
                    training: ExternalTrainingKnowledge::Unknown,
                    declared_rights: None,
                    precision: None,
                },
            },
            expected_uci_name: engine.expected_uci_name().into(),
            max_depth: 64,
            max_prefix_plies: 4096,
            handshake_timeout_ms: 1000,
            max_task_wall_time_ms: 2000,
            stop_grace_ms: 500,
            shutdown_grace_ms: 500,
            max_output_bytes: 65536,
            max_line_bytes: 4096,
        }
    }

    #[test]
    fn new_profile_is_pinned_without_start_and_legacy_schema_does_not_convert() {
        let declaration = profile();
        let bytes = serde_json::to_vec(&declaration).unwrap();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let loaded = RoventExternalProfile::from_verified_bytes(&bytes, &hash).unwrap();
        let registration = loaded.registration();
        assert_eq!(loaded.profile(), &declaration);
        assert!(registration.declaration_only);
        assert!(!registration.process_start_performed);
        assert_eq!(registration.options_application_observed, None);
        assert_eq!(registration.model_loading_observed, None);
        assert!(
            super::super::PalsCheckerProfile::from_verified_adaptive_bytes(&bytes, &hash).is_err()
        );
        let mut legacy = declaration;
        legacy.schema = super::super::PALS_ADAPTIVE_CHECKER_PROFILE_V5_SCHEMA.into();
        assert_eq!(
            legacy.validate(),
            Err(invalid("unsupported Rovent external profile schema"))
        );
    }

    #[test]
    fn duplicate_keys_wrong_pin_and_undeclared_engine_options_are_rejected() {
        let bytes = serde_json::to_vec(&profile()).unwrap();
        assert!(RoventExternalProfile::from_verified_bytes(&bytes, &"b".repeat(64)).is_err());
        let mut duplicate = bytes.clone();
        duplicate.pop();
        duplicate.extend_from_slice(b",\"schema\":\"rz-rovent-external-engine-profile/1\"}");
        let hash = format!("{:x}", Sha256::digest(&duplicate));
        assert!(RoventExternalProfile::from_verified_bytes(&duplicate, &hash).is_err());
        let mut changed = profile();
        changed
            .identity
            .options
            .insert("Ponder".into(), "false".into());
        assert!(changed.validate().is_err());
    }

    #[test]
    fn fulllegal_single_limit_does_not_authorize_restricted_or_combined_queries() {
        let mut p = profile();
        let nodes = ExternalGoConstraint::Nodes {
            reported_nodes: 4096,
        };
        assert!(p.require_query(nodes, false).is_ok());
        assert_eq!(
            p.require_query(nodes, true),
            Err(CheckerError::Unsupported(
                "engine does not support searchmoves"
            ))
        );
        assert_eq!(
            p.require_query(
                ExternalGoConstraint::Combined {
                    depth: 4,
                    reported_nodes: 4096
                },
                false
            ),
            Err(CheckerError::Unsupported(
                "engine does not support combined go constraints"
            ))
        );
        p.purpose = ExternalEnginePurpose::ArenaOpponent;
        assert_eq!(
            p.require_query(nodes, false),
            Err(CheckerError::Unsupported(
                "arena profile cannot authorize a teacher/research query"
            ))
        );
    }
}
