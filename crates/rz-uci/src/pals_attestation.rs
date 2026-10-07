//! PALS-native V3 process evidence. This is not the LC0 V1 NN attestation.
//! All snapshots come from the native worker's actual counters and join fence.

#[cfg(feature = "onnx-cpu")]
use crate::pals_native::{NativeRoleReceipt, NativeStartupErrorKind};
use crate::{
    process_receipts::{ExecutableIdentityV1, ProcessReceiptError, ProcessReceiptWriter},
    search_driver::ProcessSearchWorkReceipt,
};
use serde::Serialize;
use std::path::Path;

pub const STARTUP_FILE: &str = "pals-native-startup.v3.json";
pub const TERMINATION_FILE: &str = "pals-native-termination.v3.json";
pub const STARTUP_DOMAIN: &str = "rz-pals-native-startup-v3/1";
pub const TERMINATION_DOMAIN: &str = "rz-pals-native-termination-v3/1";

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
}
#[cfg(feature = "onnx-cpu")]
impl PalsReceiptWriter {
    pub fn open(
        output_root: &Path,
        endpoint_id: &str,
        launch_sha256: &str,
        runtime_sha256: &str,
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
        let writer =
            ProcessReceiptWriter::open_named_root(output_root, STARTUP_FILE, TERMINATION_FILE)?;
        Ok(Self {
            writer,
            endpoint_id: endpoint_id.into(),
            launch_sha256: launch_sha256.into(),
            binary_sha256: binary.sha256,
            runtime_sha256: runtime_sha256.into(),
            startup_failure: None,
        })
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
        }
    }
    pub fn startup(
        &mut self,
        native: NativeRoleReceipt,
        search_work: Option<ProcessSearchWorkReceipt>,
    ) -> Result<(), ProcessReceiptError> {
        self.validate_execution(&native)?;
        self.validate_loading_mapping(&native, false)?;
        let receipt = self.envelope(STARTUP_DOMAIN, native, false, search_work);
        self.writer.publish_startup(&receipt)
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
        let receipt = self.envelope(STARTUP_DOMAIN, native, false, search_work);
        self.writer.publish_startup(&receipt)
    }
    /// Publication also occurs for failed service/drain. The native snapshot
    /// preserves an unconfirmed fence instead of creating a success receipt.
    pub fn termination(
        &mut self,
        native: NativeRoleReceipt,
        service_exit_success: bool,
        search_work: Option<ProcessSearchWorkReceipt>,
    ) -> Result<(), ProcessReceiptError> {
        self.validate_execution(&native)?;
        // An unsuccessful service may legitimately lack the final observation.
        // Preserve its original failure and partially completed evidence.
        if service_exit_success {
            if self.startup_failure.is_some() {
                return Err(ProcessReceiptError::boundary(
                    "failed PALS startup cannot publish a successful service termination",
                ));
            }
            self.validate_loading_mapping(&native, true)?;
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
        let receipt = self.envelope(
            TERMINATION_DOMAIN,
            native,
            service_exit_success,
            search_work,
        );
        self.writer.publish_termination(&receipt)
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
pub struct SearchWorkReceiptWriter {
    writer: ProcessReceiptWriter,
    endpoint_id: String,
    launch_sha256: String,
    binary_sha256: String,
}
impl SearchWorkReceiptWriter {
    pub fn open(
        root: &Path,
        endpoint_id: &str,
        launch_sha256: &str,
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
                SEARCH_STARTUP_FILE,
                SEARCH_TERMINATION_FILE,
            )?,
            endpoint_id: endpoint_id.into(),
            launch_sha256: launch_sha256.into(),
            binary_sha256,
        })
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
        let receipt = self.envelope("rz-pals-search-work-startup-v3/1", work, false);
        self.writer.publish_startup(&receipt)
    }
    pub fn termination(
        &mut self,
        work: Option<ProcessSearchWorkReceipt>,
        success: bool,
    ) -> Result<(), ProcessReceiptError> {
        let receipt = self.envelope("rz-pals-search-work-termination-v3/1", work, success);
        self.writer.publish_termination(&receipt)
    }
}
