//! PALS-native V3 and additive V4 process evidence, separate from LC0 V1.
//! All snapshots come from the native worker's actual counters and join fence.

#[cfg(feature = "onnx-cpu")]
use crate::pals_native::{NativeRoleReceipt, NativeStartupErrorKind};
use crate::{
    process_receipts::{ExecutableIdentityV1, ProcessReceiptError, ProcessReceiptWriter},
    search_driver::ProcessSearchWorkReceipt,
};
use serde::Serialize;
use std::path::Path;
pub mod checker;
pub mod followup;

pub const STARTUP_FILE: &str = "pals-native-startup.v3.json";
pub const TERMINATION_FILE: &str = "pals-native-termination.v3.json";
pub const STARTUP_DOMAIN: &str = "rz-pals-native-startup-v3/1";
pub const TERMINATION_DOMAIN: &str = "rz-pals-native-termination-v3/1";
pub const STARTUP_FILE_V4: &str = "pals-native-startup.v4.json";
pub const TERMINATION_FILE_V4: &str = "pals-native-termination.v4.json";
pub const STARTUP_DOMAIN_V4: &str = "rz-pals-native-startup-v4/1";
pub const TERMINATION_DOMAIN_V4: &str = "rz-pals-native-termination-v4/1";

/// Actual immutable source/driver selections. This identifies the lane and
/// loaded graph; it does not claim that a Repair, stack resume or Warm ran.
#[derive(Clone, Debug, Serialize)]
pub struct PalsFollowupMarkerV4 {
    pub model_identity: String,
    pub model_profile: Option<String>,
    pub model_semantics: String,
    pub model_semantics_sha256: Option<[u8; 32]>,
    pub model_implementation_sha256: Option<[u8; 32]>,
    pub encoding_schema: Option<String>,
    pub model_configuration: Option<rz_eval::pals_model::PalsModelConfig>,
    pub model_epoch: Option<[u8; 32]>,
    pub resolver_policy: String,
    pub cpu_resume: Option<String>,
    pub post_repair_recheck: String,
    pub policy_identity: String,
    pub policy_conditions_sha256: Option<[u8; 32]>,
    pub search_implementation_sha256: [u8; 32],
    pub resolver_implementation_sha256: [u8; 32],
    pub resolver_semantics_sha256: [u8; 32],
    pub cuda_warm: bool,
}
impl PalsFollowupMarkerV4 {
    pub fn capture_search<M: rz_search::pals::engine::RoleModel + 'static>(
        driver: &crate::search_driver::PalsSessionDriver<M>,
    ) -> Result<Self, crate::search_driver::SearchSessionFailure> {
        use rz_search::pals::engine::{PostRepairRecheckPolicy as Recheck, ResolverPolicy};
        let (resolver, recheck, identity, conditions) = driver.selected_policies()?;
        let registration = driver.checker_registration();
        Ok(Self {
            model_identity: registration.role_model.clone(),
            model_profile: None,
            model_semantics: registration.role_model.clone(),
            model_semantics_sha256: None,
            model_implementation_sha256: None,
            encoding_schema: None,
            model_configuration: None,
            model_epoch: None,
            resolver_policy: match resolver {
                ResolverPolicy::OwnRawRestricted => "own-raw-restricted",
                ResolverPolicy::ModelWdlRestricted => "model-wdl-restricted",
            }
            .into(),
            cpu_resume: registration.cpu_resume_policy.map(|policy| {
                match policy {
                    rz_search::cpu::CpuResumePolicy::CompletedIteration => "completed-iteration",
                    rz_search::cpu::CpuResumePolicy::PausedStack => "paused-stack",
                }
                .into()
            }),
            post_repair_recheck: match recheck {
                Recheck::Disabled => "disabled",
                Recheck::SameRepairedLineOnceV1 => "same-repaired-line-once-v1",
                Recheck::ActualOpponentContinuationV1 => "actual-opponent-continuation-v1",
                Recheck::FrozenModelWdlV2 => "frozen-model-wdl-v2",
                Recheck::IterativeFrozenModelWdlV2 => "iterative-frozen-model-wdl-v2",
            }
            .into(),
            policy_identity: identity.into(),
            policy_conditions_sha256: conditions
                .map(|value| rz_eval::asset::sha256(value.as_bytes())),
            cuda_warm: false,
            search_implementation_sha256:
                rz_search::pals::engine::compiled_search_implementation_sha256(),
            resolver_implementation_sha256:
                rz_search::pals::engine::compiled_resolver_implementation_sha256(),
            resolver_semantics_sha256: resolver.semantics_sha256(),
        })
    }
    #[cfg(feature = "onnx-cpu")]
    pub fn capture_native<M: rz_search::pals::engine::RoleModel + 'static>(
        source: &crate::pals_native::NativeRoleSourceIdentity,
        driver: &crate::search_driver::PalsSessionDriver<M>,
    ) -> Result<Self, crate::search_driver::SearchSessionFailure> {
        source.model_configuration.validate().map_err(|error| {
            crate::search_driver::SearchSessionFailure::debug(
                "PalsModelConfigurationObservation",
                &error,
            )
        })?;
        let mut marker = Self::capture_search(driver)?;
        let profile = source.model_configuration.profile;
        marker.model_profile = Some(profile.as_str().into());
        marker.model_semantics = profile.model_semantics().into();
        marker.model_semantics_sha256 = Some(profile.registered_semantics_sha256());
        marker.model_implementation_sha256 =
            Some(rz_eval::pals_model::compiled_model_boundary_sha256());
        marker.encoding_schema = Some(profile.encoding_schema().into());
        marker.model_configuration = Some(source.model_configuration.clone());
        marker.model_epoch = Some(source.checkpoint_sha256);
        marker.cuda_warm = source
            .execution
            .private_warm
            .as_ref()
            .is_some_and(|warm| warm.schema == "rovezero.pals-private-cuda-warm.v2");
        Ok(marker)
    }
}

#[cfg(feature = "onnx-cpu")]
#[derive(Clone, Debug, Serialize)]
pub struct PalsNativeReceiptV3 {
    pub schema_version: u32,
    pub domain: &'static str,
    pub endpoint_id: String,
    pub launch_sha256: String,
    pub process_id: u32,
    pub binary_sha256: String,
    pub runtime_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_bundle_sha256: Option<String>,
    pub provider: &'static str,
    pub precision: &'static str,
    pub service_exit_success: bool,
    /// An unsuccessful startup is evidence, never a readiness attestation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub startup_failure: Option<NativeStartupErrorKind>,
    pub native: NativeRoleReceipt,
    pub search_work: Option<ProcessSearchWorkReceipt>,
    /// Optional external helper evidence. Native NN and helper completion are
    /// independent owner fences; historical V3 receipts omit this member.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_checker: Option<checker::PalsCheckerProcessReceipt>,
}
#[cfg(feature = "onnx-cpu")]
#[derive(Clone, Debug, Serialize)]
pub struct PalsNativeReceiptV4 {
    #[serde(flatten)]
    pub evidence: PalsNativeReceiptV3,
    /// None preserves a failure before a search driver was constructed; no
    /// requested selection is promoted into an actual driver observation.
    pub v4: Option<PalsFollowupMarkerV4>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub followup_lifecycle: Option<followup::PalsFollowupLifecycleV4>,
}

/// Existing private root, fresh process slot, fixed names, bounded encoding,
/// no overwrite, no symlink follow, and file sync are shared with V1.
#[cfg(feature = "onnx-cpu")]
pub struct PalsReceiptWriter {
    writer: ProcessReceiptWriter,
    endpoint_id: String,
    launch_sha256: String,
    binary_sha256: String,
    runtime_sha256: String,
    startup_failure: Option<NativeStartupErrorKind>,
    cpu_checker: Option<checker::PalsCheckerProcessReceipt>,
    v4: bool,
    followup: Option<PalsFollowupMarkerV4>,
    followup_lifecycle: Option<followup::PalsFollowupLifecycleV4>,
}
#[cfg(feature = "onnx-cpu")]
impl PalsReceiptWriter {
    pub fn open(
        output_root: &Path,
        endpoint_id: &str,
        launch_sha256: &str,
        runtime_sha256: &str,
    ) -> Result<Self, ProcessReceiptError> {
        Self::open_selected(
            output_root,
            endpoint_id,
            launch_sha256,
            runtime_sha256,
            false,
            None,
        )
    }
    pub fn open_v4(
        output_root: &Path,
        endpoint_id: &str,
        launch_sha256: &str,
        runtime_sha256: &str,
        followup: Option<PalsFollowupMarkerV4>,
    ) -> Result<Self, ProcessReceiptError> {
        Self::open_selected(
            output_root,
            endpoint_id,
            launch_sha256,
            runtime_sha256,
            true,
            followup,
        )
    }
    fn open_selected(
        output_root: &Path,
        endpoint_id: &str,
        launch_sha256: &str,
        runtime_sha256: &str,
        v4: bool,
        followup: Option<PalsFollowupMarkerV4>,
    ) -> Result<Self, ProcessReceiptError> {
        if endpoint_id.is_empty()
            || endpoint_id.len() > 64
            || !endpoint_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err(ProcessReceiptError::boundary("PALS endpoint ID is invalid"));
        }
        let valid_digest = |text: &str| {
            text.len() == 64
                && text
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        };
        if !valid_digest(launch_sha256) || !valid_digest(runtime_sha256) {
            return Err(ProcessReceiptError::boundary(
                "PALS launch/runtime identity requires canonical SHA-256",
            ));
        }
        let binary = ExecutableIdentityV1::observe()?;
        let writer = ProcessReceiptWriter::open_named_root(
            output_root,
            if v4 { STARTUP_FILE_V4 } else { STARTUP_FILE },
            if v4 {
                TERMINATION_FILE_V4
            } else {
                TERMINATION_FILE
            },
        )?;
        Ok(Self {
            writer,
            endpoint_id: endpoint_id.into(),
            launch_sha256: launch_sha256.into(),
            binary_sha256: binary.sha256,
            runtime_sha256: runtime_sha256.into(),
            startup_failure: None,
            cpu_checker: None,
            v4,
            followup,
            followup_lifecycle: None,
        })
    }
    pub fn observe_checker(
        &mut self,
        observation: checker::PalsCheckerProcessReceipt,
    ) -> Result<(), ProcessReceiptError> {
        if self
            .cpu_checker
            .as_ref()
            .is_some_and(|previous| !previous.same_registration(&observation))
        {
            return Err(ProcessReceiptError::boundary(
                "PALS helper registration changed after startup observation",
            ));
        }
        self.cpu_checker = Some(observation);
        Ok(())
    }
    pub fn set_followup_lifecycle(
        &mut self,
        snapshot: Option<followup::PalsFollowupLifecycleV4>,
    ) -> Result<(), ProcessReceiptError> {
        if !self.v4 && snapshot.is_some() {
            return Err(ProcessReceiptError::boundary(
                "followup owner evidence requires the explicit V4 envelope",
            ));
        }
        self.followup_lifecycle = snapshot;
        Ok(())
    }
    fn envelope(
        &self,
        domain: &'static str,
        native: NativeRoleReceipt,
        service_exit_success: bool,
        search_work: Option<ProcessSearchWorkReceipt>,
    ) -> PalsNativeReceiptV3 {
        PalsNativeReceiptV3 {
            schema_version: 3,
            domain,
            endpoint_id: self.endpoint_id.clone(),
            launch_sha256: self.launch_sha256.clone(),
            process_id: std::process::id(),
            binary_sha256: self.binary_sha256.clone(),
            runtime_sha256: self.runtime_sha256.clone(),
            provider: native.execution.provider,
            runtime_bundle_sha256: native
                .execution
                .runtime_bundle_sha256
                .map(|digest| digest.iter().map(|byte| format!("{byte:02x}")).collect()),
            precision: "fp32",
            service_exit_success,
            startup_failure: self.startup_failure,
            native,
            search_work,
            cpu_checker: self.cpu_checker.clone(),
        }
    }
    pub fn startup(
        &mut self,
        native: NativeRoleReceipt,
        search_work: Option<ProcessSearchWorkReceipt>,
    ) -> Result<(), ProcessReceiptError> {
        if self.v4 && self.followup.is_none() {
            return Err(ProcessReceiptError::boundary(
                "PALS V4 startup requires actual loaded model and driver selections",
            ));
        }
        self.validate_execution(&native)?;
        self.validate_loading_mapping(&native, false)?;
        self.publish_startup(native, search_work)
    }
    /// The same identity and bounded, exclusive files preserve actual partial
    /// counters, timing and an unconfirmed fence. Missing ACKs stay missing;
    /// their absence must not replace the original preparation failure.
    pub fn failed_startup(
        &mut self,
        native: NativeRoleReceipt,
        primary: &rz_search::pals::engine::RoleError,
        search_work: Option<ProcessSearchWorkReceipt>,
    ) -> Result<(), ProcessReceiptError> {
        self.validate_execution(&native)?;
        self.startup_failure = Some(NativeStartupErrorKind::from(primary));
        self.publish_startup(native, search_work)
    }
    /// Publication also occurs for failed service/drain. The native snapshot
    /// preserves an unconfirmed fence instead of creating a success receipt.
    pub fn termination(
        &mut self,
        native: NativeRoleReceipt,
        service_exit_success: bool,
        search_work: Option<ProcessSearchWorkReceipt>,
    ) -> Result<(), ProcessReceiptError> {
        if self.v4 && service_exit_success && self.followup.is_none() {
            return Err(ProcessReceiptError::boundary(
                "successful PALS V4 termination requires actual driver source selections",
            ));
        }
        self.validate_execution(&native)?;
        if service_exit_success
            && self
                .cpu_checker
                .as_ref()
                .is_some_and(|checker| !checker.cleanup_complete())
        {
            return Err(ProcessReceiptError::boundary(
                "successful PALS service requires independent external helper cleanup",
            ));
        }
        // An unsuccessful service may legitimately lack the final observation.
        // Preserve its original failure and partially completed evidence.
        if service_exit_success {
            if self.startup_failure.is_some() {
                return Err(ProcessReceiptError::boundary(
                    "failed PALS startup cannot publish a successful service termination",
                ));
            }
            self.validate_loading_mapping(&native, true)?;
            crate::pals_native::validate_host_record_page_evidence(&native, true).map_err(|_| {
                ProcessReceiptError::boundary("successful PALS host record pages require actual final snapshot, declared bounds and physical join evidence")
            })?;
            let terminal = self
                .followup_lifecycle
                .as_ref()
                .and_then(|lifecycle| match &lifecycle.cuda_warm {
                    followup::NativeObservedV4::Observed { value, .. } => Some(value),
                    followup::NativeObservedV4::Unknown => None,
                });
            crate::pals_native::validate_private_warm_evidence_with_terminal_scope(&native, true,terminal).map_err(|_| {
                ProcessReceiptError::boundary("successful PALS private warm requires its actual capability, bounded seed ownership and known physical join")
            })?;
        }
        if service_exit_success
            && native.execution.provider == "cuda"
            && (native.final_runtime_mapping_confirmed != Some(true)
                || !native.startup_probe.as_ref().is_some_and(|probe| {
                    probe.completed_proposer_calls == 1
                        && probe.completed_critic_calls == 1
                        && probe.runtime_mapping_confirmed
                        && probe.cuda_placement_witness.is_some()
                        && probe.reset_completed
                }))
        {
            return Err(ProcessReceiptError::boundary(
                "successful PALS CUDA service requires actual placement and final origin ACKs",
            ));
        }
        let mut receipt = self.envelope(
            if self.v4 {
                TERMINATION_DOMAIN_V4
            } else {
                TERMINATION_DOMAIN
            },
            native,
            service_exit_success,
            search_work,
        );
        if self.v4 {
            receipt.schema_version = 4;
            self.writer.publish_termination(&PalsNativeReceiptV4 {
                evidence: receipt,
                v4: self.followup.clone(),
                followup_lifecycle: self.followup_lifecycle.clone(),
            })
        } else {
            self.writer.publish_termination(&receipt)
        }
    }
    fn publish_startup(
        &mut self,
        native: NativeRoleReceipt,
        search_work: Option<ProcessSearchWorkReceipt>,
    ) -> Result<(), ProcessReceiptError> {
        let mut receipt = self.envelope(
            if self.v4 {
                STARTUP_DOMAIN_V4
            } else {
                STARTUP_DOMAIN
            },
            native,
            false,
            search_work,
        );
        if self.v4 {
            receipt.schema_version = 4;
            self.writer.publish_startup(&PalsNativeReceiptV4 {
                evidence: receipt,
                v4: self.followup.clone(),
                followup_lifecycle: self.followup_lifecycle.clone(),
            })
        } else {
            self.writer.publish_startup(&receipt)
        }
    }
    fn validate_loading_mapping(
        &self,
        native: &NativeRoleReceipt,
        require_final: bool,
    ) -> Result<(), ProcessReceiptError> {
        if native.execution.cuda_loading_profile.is_none() {
            return Ok(());
        }
        let startup = native
            .startup_probe
            .as_ref()
            .and_then(|probe| probe.runtime_loading_mapping.as_ref())
            .ok_or_else(|| {
                ProcessReceiptError::boundary(
                    "selected PALS CUDA loading profile requires an actual startup mapping ACK",
                )
            })?;
        crate::pals_native::validate_runtime_loading_mapping(startup, &native.execution).map_err(
            |_| {
                ProcessReceiptError::boundary(
                    "PALS startup loading mapping differs from its pinned actual execution",
                )
            },
        )?;
        if require_final {
            let final_mapping = native.final_runtime_loading_mapping.as_ref().ok_or_else(|| {
                ProcessReceiptError::boundary(
                    "successful selected PALS CUDA loading profile requires a final mapping ACK",
                )
            })?;
            crate::pals_native::validate_runtime_loading_mapping(final_mapping, &native.execution)
                .map_err(|_| {
                    ProcessReceiptError::boundary(
                        "PALS final loading mapping differs from its pinned actual execution",
                    )
                })?;
        }
        Ok(())
    }
    fn validate_execution(&self, native: &NativeRoleReceipt) -> Result<(), ProcessReceiptError> {
        if let Some(marker) = &self.followup {
            let config = marker.model_configuration.as_ref().ok_or_else(|| {
                ProcessReceiptError::boundary("native V4 marker lacks actual model configuration")
            })?;
            config.validate().map_err(|_| {
                ProcessReceiptError::boundary("native V4 model configuration is unsupported")
            })?;
            if marker.model_epoch != Some(native.model_epoch)
                || marker.model_profile.as_deref() != Some(config.profile.as_str())
                || marker.model_semantics != config.profile.model_semantics()
                || marker.model_semantics_sha256
                    != Some(config.profile.registered_semantics_sha256())
                || marker.model_implementation_sha256
                    != Some(rz_eval::pals_model::compiled_model_boundary_sha256())
                || marker.encoding_schema.as_deref() != Some(config.profile.encoding_schema())
                || marker.cuda_warm
                    != native
                        .execution
                        .private_warm
                        .as_ref()
                        .is_some_and(|warm| warm.schema == "rovezero.pals-private-cuda-warm.v2")
            {
                return Err(ProcessReceiptError::boundary(
                    "native V4 marker differs from its actual loaded source",
                ));
            }
        }
        crate::pals_native::validate_host_record_page_evidence(native, false).map_err(|_| {
            ProcessReceiptError::boundary("PALS host record page declaration/observation differs from its actual execution identity or bounds")
        })?;
        crate::pals_native::validate_private_warm_evidence(native, false).map_err(|_| {
            ProcessReceiptError::boundary("PALS private warm declaration or actual seed observation differs from its selected namespace")
        })?;
        let runtime: String = native
            .execution
            .runtime_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        if runtime != self.runtime_sha256
            || !matches!(native.execution.provider, "cpu" | "cuda")
            || (native.execution.provider == "cuda")
                != native.execution.runtime_bundle_sha256.is_some()
            || (native.execution.provider == "cpu"
                && (native.execution.cuda_control_inventory_sha256.is_some()
                    || native.execution.cuda_loading_profile.is_some()))
        {
            return Err(ProcessReceiptError::boundary(
                "PALS actual provider/runtime pin differs from receipt identity",
            ));
        }
        // Missing observations are allowed only by the failed-startup/failed
        // termination paths. An observation that is actually present must
        // always agree with the owner identity, even on those failure paths.
        for mapping in native
            .startup_probe
            .as_ref()
            .and_then(|probe| probe.runtime_loading_mapping.as_ref())
            .into_iter()
            .chain(native.final_runtime_loading_mapping.as_ref())
        {
            crate::pals_native::validate_runtime_loading_mapping(mapping, &native.execution)
                .map_err(|_| {
                    ProcessReceiptError::boundary(
                        "PALS partial loading mapping differs from its pinned actual execution",
                    )
                })?;
        }
        if let Some(witness) = native
            .startup_probe
            .as_ref()
            .and_then(|probe| probe.cuda_placement_witness.as_ref())
        {
            if native.execution.provider != "cuda"
                || native.execution.cuda_control_inventory_sha256 != Some(witness.inventory_sha256)
                || native.export_manifest_sha256 != witness.manifest_sha256
                || native.execution.runtime_sha256 != witness.runtime_sha256
                || native.execution.runtime_bundle_sha256 != Some(witness.runtime_bundle_sha256)
            {
                return Err(ProcessReceiptError::boundary(
                    "PALS actual placement witness differs from its pinned execution identity",
                ));
            }
        }
        Ok(())
    }
}

pub const SEARCH_STARTUP_FILE: &str = "search-work-startup.v3.json";
pub const SEARCH_TERMINATION_FILE: &str = "search-work-termination.v3.json";
pub const SEARCH_STARTUP_FILE_V4: &str = "search-work-startup.v4.json";
pub const SEARCH_TERMINATION_FILE_V4: &str = "search-work-termination.v4.json";
#[derive(Clone, Debug, Serialize)]
pub struct SearchWorkReceiptV3 {
    pub schema_version: u32,
    pub domain: &'static str,
    pub endpoint_id: String,
    pub launch_sha256: String,
    pub process_id: u32,
    pub binary_sha256: String,
    pub service_exit_success: bool,
    pub search_work: Option<ProcessSearchWorkReceipt>,
}
#[derive(Clone, Debug, Serialize)]
pub struct SearchWorkReceiptV4 {
    #[serde(flatten)]
    pub evidence: SearchWorkReceiptV3,
    pub v4: Option<PalsFollowupMarkerV4>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub followup_lifecycle: Option<followup::PalsFollowupLifecycleV4>,
}
pub struct SearchWorkReceiptWriter {
    writer: ProcessReceiptWriter,
    endpoint_id: String,
    launch_sha256: String,
    binary_sha256: String,
    v4: bool,
    followup: Option<PalsFollowupMarkerV4>,
    followup_lifecycle: Option<followup::PalsFollowupLifecycleV4>,
}
impl SearchWorkReceiptWriter {
    pub fn open(
        root: &Path,
        endpoint_id: &str,
        launch_sha256: &str,
    ) -> Result<Self, ProcessReceiptError> {
        Self::open_selected(root, endpoint_id, launch_sha256, false, None)
    }
    pub fn open_v4(
        root: &Path,
        endpoint_id: &str,
        launch_sha256: &str,
        followup: PalsFollowupMarkerV4,
    ) -> Result<Self, ProcessReceiptError> {
        Self::open_selected(root, endpoint_id, launch_sha256, true, Some(followup))
    }
    /// Independent OwnCpu endpoint using the shared V4 envelope. A null PALS
    /// marker preserves that no PALS model or driver was actually constructed.
    pub fn open_v4_without_pals(
        root: &Path,
        endpoint_id: &str,
        launch_sha256: &str,
    ) -> Result<Self, ProcessReceiptError> {
        Self::open_selected(root, endpoint_id, launch_sha256, true, None)
    }
    fn open_selected(
        root: &Path,
        endpoint_id: &str,
        launch_sha256: &str,
        v4: bool,
        followup: Option<PalsFollowupMarkerV4>,
    ) -> Result<Self, ProcessReceiptError> {
        if endpoint_id.is_empty()
            || endpoint_id.len() > 64
            || !endpoint_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err(ProcessReceiptError::boundary(
                "search work endpoint ID is invalid",
            ));
        }
        if launch_sha256.len() != 64
            || !launch_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(ProcessReceiptError::boundary(
                "search work launch identity requires canonical SHA-256",
            ));
        }
        let binary_sha256 = ExecutableIdentityV1::observe()?.sha256;
        Ok(Self {
            writer: ProcessReceiptWriter::open_named_root(
                root,
                if v4 {
                    SEARCH_STARTUP_FILE_V4
                } else {
                    SEARCH_STARTUP_FILE
                },
                if v4 {
                    SEARCH_TERMINATION_FILE_V4
                } else {
                    SEARCH_TERMINATION_FILE
                },
            )?,
            endpoint_id: endpoint_id.into(),
            launch_sha256: launch_sha256.into(),
            binary_sha256,
            v4,
            followup,
            followup_lifecycle: None,
        })
    }
    pub fn set_followup_lifecycle(
        &mut self,
        snapshot: Option<followup::PalsFollowupLifecycleV4>,
    ) -> Result<(), ProcessReceiptError> {
        if !self.v4 && snapshot.is_some() {
            return Err(ProcessReceiptError::boundary(
                "followup owner evidence requires the explicit V4 envelope",
            ));
        }
        self.followup_lifecycle = snapshot;
        Ok(())
    }
    fn envelope(
        &self,
        domain: &'static str,
        work: Option<ProcessSearchWorkReceipt>,
        success: bool,
    ) -> SearchWorkReceiptV3 {
        SearchWorkReceiptV3 {
            schema_version: 3,
            domain,
            endpoint_id: self.endpoint_id.clone(),
            launch_sha256: self.launch_sha256.clone(),
            process_id: std::process::id(),
            binary_sha256: self.binary_sha256.clone(),
            service_exit_success: success,
            search_work: work,
        }
    }
    pub fn startup(
        &mut self,
        work: Option<ProcessSearchWorkReceipt>,
    ) -> Result<(), ProcessReceiptError> {
        let mut receipt = self.envelope(
            if self.v4 {
                "rz-pals-search-work-startup-v4/1"
            } else {
                "rz-pals-search-work-startup-v3/1"
            },
            work,
            false,
        );
        if self.v4 {
            receipt.schema_version = 4;
            self.writer.publish_startup(&SearchWorkReceiptV4 {
                evidence: receipt,
                v4: self.followup.clone(),
                followup_lifecycle: self.followup_lifecycle.clone(),
            })
        } else {
            self.writer.publish_startup(&receipt)
        }
    }
    pub fn termination(
        &mut self,
        work: Option<ProcessSearchWorkReceipt>,
        success: bool,
    ) -> Result<(), ProcessReceiptError> {
        let mut receipt = self.envelope(
            if self.v4 {
                "rz-pals-search-work-termination-v4/1"
            } else {
                "rz-pals-search-work-termination-v3/1"
            },
            work,
            success,
        );
        if self.v4 {
            receipt.schema_version = 4;
            self.writer.publish_termination(&SearchWorkReceiptV4 {
                evidence: receipt,
                v4: self.followup.clone(),
                followup_lifecycle: self.followup_lifecycle.clone(),
            })
        } else {
            self.writer.publish_termination(&receipt)
        }
    }
}

#[cfg(test)]
mod followup_tests {
    use super::*;
    use crate::{EngineIdentity, search_driver::PalsSessionDriver};
    use rz_search::{
        cpu::{CpuConfig, CpuResumePolicy},
        pals::engine::{
            LegalOrderRoleMock, PALS_FOLLOWUP_CONDITIONS, PALS_FOLLOWUP_SEARCH_VERSION,
            PALS_SEARCH_VERSION, PalsConfig, PostRepairRecheckPolicy, ResolverPolicy,
        },
    };

    fn identity() -> EngineIdentity {
        EngineIdentity {
            name: "PALS marker fixture".into(),
            author: "RoveZero".into(),
        }
    }

    #[test]
    fn marker_observes_selected_policy_and_keeps_mock_model_metadata_absent() {
        let legacy = PalsSessionDriver::new(
            PalsConfig::default(),
            LegalOrderRoleMock,
            CpuConfig::default(),
            1,
            100,
            1,
            identity(),
        )
        .unwrap();
        let legacy_marker = PalsFollowupMarkerV4::capture_search(&legacy).unwrap();
        assert_eq!(legacy_marker.policy_identity, PALS_SEARCH_VERSION);
        assert_eq!(legacy_marker.policy_conditions_sha256, None);
        let followup = PalsSessionDriver::new_with_policies(
            PalsConfig::default(),
            LegalOrderRoleMock,
            CpuConfig::default(),
            1,
            100,
            1,
            identity(),
            ResolverPolicy::OwnRawRestricted,
            PostRepairRecheckPolicy::Disabled,
            CpuResumePolicy::PausedStack,
        )
        .unwrap();
        let marker = PalsFollowupMarkerV4::capture_search(&followup).unwrap();
        assert_eq!(
            marker.model_identity,
            followup.checker_registration().role_model
        );
        assert_eq!(marker.policy_identity, PALS_FOLLOWUP_SEARCH_VERSION);
        assert_eq!(
            marker.policy_conditions_sha256,
            Some(rz_eval::asset::sha256(PALS_FOLLOWUP_CONDITIONS.as_bytes()))
        );
        assert_eq!(marker.resolver_policy, "own-raw-restricted");
        assert_eq!(marker.cpu_resume.as_deref(), Some("paused-stack"));
        assert_eq!(marker.post_repair_recheck, "disabled");
        assert!(
            marker.model_profile.is_none()
                && marker.model_configuration.is_none()
                && marker.model_semantics_sha256.is_none()
                && marker.model_implementation_sha256.is_none()
                && marker.encoding_schema.is_none()
                && marker.model_epoch.is_none()
                && !marker.cuda_warm
        );
    }

    #[test]
    fn v4_search_wrapper_uses_a_distinct_domain_without_changing_v3_fields() {
        let driver = PalsSessionDriver::new_with_policies(
            PalsConfig::default(),
            LegalOrderRoleMock,
            CpuConfig::default(),
            1,
            100,
            1,
            identity(),
            ResolverPolicy::OwnRawRestricted,
            PostRepairRecheckPolicy::Disabled,
            CpuResumePolicy::CompletedIteration,
        )
        .unwrap();
        let receipt = SearchWorkReceiptV3 {
            schema_version: 3,
            domain: "rz-pals-search-work-startup-v3/1",
            endpoint_id: "fixture".into(),
            launch_sha256: "01".repeat(32),
            process_id: 1,
            binary_sha256: "02".repeat(32),
            service_exit_success: false,
            search_work: None,
        };
        let legacy = serde_json::to_value(&receipt).unwrap();
        assert_eq!(legacy["schema_version"], 3);
        assert!(legacy.get("v4").is_none());
        let mut evidence = receipt;
        evidence.schema_version = 4;
        evidence.domain = "rz-pals-search-work-startup-v4/1";
        let own_cpu = serde_json::to_value(SearchWorkReceiptV4 {
            evidence: evidence.clone(),
            v4: None,
            followup_lifecycle: None,
        })
        .unwrap();
        assert_eq!(own_cpu["schema_version"], 4);
        assert!(own_cpu["v4"].is_null());
        let v4 = serde_json::to_value(SearchWorkReceiptV4 {
            evidence,
            v4: Some(PalsFollowupMarkerV4::capture_search(&driver).unwrap()),
            followup_lifecycle: None,
        })
        .unwrap();
        assert_eq!(v4["schema_version"], 4);
        assert_eq!(v4["domain"], "rz-pals-search-work-startup-v4/1");
        assert_eq!(v4["endpoint_id"], legacy["endpoint_id"]);
        assert_eq!(v4["v4"]["policy_identity"], PALS_FOLLOWUP_SEARCH_VERSION);
        assert!(v4["v4"]["model_configuration"].is_null());
        assert_ne!(SEARCH_STARTUP_FILE, SEARCH_STARTUP_FILE_V4);
        assert_ne!(SEARCH_TERMINATION_FILE, SEARCH_TERMINATION_FILE_V4);
    }
}
