//! PALS-native V3 process evidence. This is not the LC0 V1 NN attestation.
//! All snapshots come from the native worker's actual counters and join fence.

#[cfg(feature = "onnx-cpu")]
use crate::pals_native::NativeRoleReceipt;
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
    pub provider: &'static str,
    pub precision: &'static str,
    pub service_exit_success: bool,
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
            provider: "cpu",
            precision: "fp32",
            service_exit_success,
            native,
            search_work,
        }
    }
    pub fn startup(
        &mut self,
        native: NativeRoleReceipt,
        search_work: Option<ProcessSearchWorkReceipt>,
    ) -> Result<(), ProcessReceiptError> {
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
        let receipt = self.envelope(
            TERMINATION_DOMAIN,
            native,
            service_exit_success,
            search_work,
        );
        self.writer.publish_termination(&receipt)
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
