//! Static Python-to-Rust packing artifact interoperability check.
//!
//! Required features are `onnx,contracts`, but no ORT runtime/session/provider,
//! model, device allocation, Run or physical completion is requested here.
//! The two nonzero resource sentinels satisfy the declaration-only API; neither
//! is an observation of a session or metadata allocation. Matching caller pins
//! proves byte identity, not producer authenticity or native authorization.
//!
//! Usage: device_packing_graph_check ABS_MANIFEST ABS_GRAPH MANIFEST_SHA GRAPH_SHA
use rz_eval::pals_onnx::{
    load_checked_fixed_packing_graph, DevicePackingAssetError, DevicePackingResourceDeclaration,
    PackingGraphBodyError, PackingGraphInspectionBudget, PackingNativeVerification,
    DEVICE_PACKING_MAX_GRAPH_BYTES, DEVICE_PACKING_MAX_MANIFEST_BYTES,
    DEVICE_PACKING_MAX_NODE_PAYLOAD_SUM, PACKING_GRAPH_BODY_INSPECTOR_VERSION,
    PACKING_GRAPH_MAX_INSPECTION_HOST_BYTES, PACKING_GRAPH_MAX_WIRE_FIELDS,
};
use serde_json::{json, Value};
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Component, PathBuf};
use std::process::ExitCode;

const REPORT_SCHEMA: &str = "rz-pals-device-packing-graph-check/1";
const STATIC_SCOPE: &str = "registered_bytes_and_fixed_graph_body_only;no_native_execution";
const SESSION_SENTINEL_BYTES: u64 = 1;
const METADATA_SENTINEL_BYTES: u64 = 1;
const MAX_ARTIFACT_OWNER_BYTES: u64 =
    (DEVICE_PACKING_MAX_GRAPH_BYTES + DEVICE_PACKING_MAX_MANIFEST_BYTES) as u64 + 64 * 1024;
const MAX_DECLARED_HOST_BYTES: u64 = MAX_ARTIFACT_OWNER_BYTES + 64 * 1024;
const MAX_DECLARED_DEVICE_BYTES: u64 = DEVICE_PACKING_MAX_NODE_PAYLOAD_SUM + SESSION_SENTINEL_BYTES;

#[derive(Debug)]
enum CheckError {
    Arguments(&'static str),
    Assets(DevicePackingAssetError),
    GraphBody(PackingGraphBodyError),
    UnexpectedNativeScope,
    Json(serde_json::Error),
    Output(io::Error),
}

impl CheckError {
    fn stage(&self) -> &'static str {
        match self {
            Self::Arguments(_) => "arguments",
            Self::Assets(error) => error.stage(),
            Self::GraphBody(_) => "fixed_graph_body",
            Self::UnexpectedNativeScope => "scope",
            Self::Json(_) | Self::Output(_) => "report_delivery",
        }
    }

    fn detail(&self) -> String {
        let text = match self {
            Self::Arguments(reason) => (*reason).to_owned(),
            Self::Assets(error) => error.detail(),
            Self::Output(error) => error.to_string(),
            Self::GraphBody(error) => error.to_string(),
            Self::UnexpectedNativeScope => {
                "static inspector unexpectedly returned native scope".to_owned()
            }
            Self::Json(error) => error.to_string(),
        };
        text.chars().take(512).collect()
    }
}

type Result<T> = std::result::Result<T, CheckError>;

#[derive(Debug)]
struct Arguments {
    manifest: PathBuf,
    graph: PathBuf,
    manifest_sha256: [u8; 32],
    graph_sha256: [u8; 32],
}

fn absolute_path(value: OsString) -> Result<PathBuf> {
    let path = PathBuf::from(value);
    if !path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
    {
        return Err(CheckError::Arguments(
            "artifact paths must be absolute without parent traversal",
        ));
    }
    Ok(path)
}

fn lowercase_sha256(value: OsString) -> Result<[u8; 32]> {
    let text = value
        .into_string()
        .map_err(|_| CheckError::Arguments("SHA-256 must be lowercase ASCII hex"))?;
    let bytes = text.as_bytes();
    if bytes.len() != 64
        || !bytes
            .iter()
            .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(value))
    {
        return Err(CheckError::Arguments(
            "SHA-256 requires exactly 64 lowercase ASCII hex characters",
        ));
    }
    let mut digest = [0u8; 32];
    for (index, pair) in bytes.chunks_exact(2).enumerate() {
        let nibble = |value| match value {
            b'0'..=b'9' => value - b'0',
            b'a'..=b'f' => value - b'a' + 10,
            _ => 0, // The entire input was validated above.
        };
        digest[index] = (nibble(pair[0]) << 4) | nibble(pair[1]);
    }
    Ok(digest)
}

fn parse_arguments(values: impl IntoIterator<Item = OsString>) -> Result<Arguments> {
    let mut values = values.into_iter();
    let mut next = || {
        values.next().ok_or(CheckError::Arguments(
            "expected exactly four arguments: ABS_MANIFEST ABS_GRAPH MANIFEST_SHA GRAPH_SHA",
        ))
    };
    let manifest = absolute_path(next()?)?;
    let graph = absolute_path(next()?)?;
    let manifest_sha256 = lowercase_sha256(next()?)?;
    let graph_sha256 = lowercase_sha256(next()?)?;
    if values.next().is_some() {
        return Err(CheckError::Arguments(
            "expected exactly four positional arguments",
        ));
    }
    Ok(Arguments {
        manifest,
        graph,
        manifest_sha256,
        graph_sha256,
    })
}

fn hex(digest: [u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut value = String::with_capacity(64);
    for byte in digest {
        value.push(DIGITS[usize::from(byte >> 4)] as char);
        value.push(DIGITS[usize::from(byte & 15)] as char);
    }
    value
}

fn check(arguments: Arguments) -> Result<Value> {
    let declaration = DevicePackingResourceDeclaration {
        packing_session_bytes: Some(SESSION_SENTINEL_BYTES),
        additional_owner_metadata_host_bytes: Some(METADATA_SENTINEL_BYTES),
        max_owned_artifact_host_bytes: MAX_ARTIFACT_OWNER_BYTES,
        max_declared_host_bytes: MAX_DECLARED_HOST_BYTES,
        max_declared_device_bytes: MAX_DECLARED_DEVICE_BYTES,
    };
    // Only this static CLI supplies sentinel declarations. Product callers must
    // supply their own resources to the same guarded loader, never this policy.
    let checked = load_checked_fixed_packing_graph(
        &arguments.manifest,
        &arguments.graph,
        arguments.manifest_sha256,
        arguments.graph_sha256,
        declaration,
        PackingGraphInspectionBudget {
            max_inspection_host_bytes: PACKING_GRAPH_MAX_INSPECTION_HOST_BYTES,
            max_wire_fields: PACKING_GRAPH_MAX_WIRE_FIELDS,
        },
    )
    .map_err(CheckError::Assets)?;
    let registration = checked.registration();
    let byte_scope = format!("{:?}", checked.registered_artifacts().verification_scope());
    if checked.native_verification() != PackingNativeVerification::NotPerformed
        || checked.registered_artifacts().native_verification()
            != PackingNativeVerification::NotPerformed
    {
        return Err(CheckError::UnexpectedNativeScope);
    }
    let body = checked.inspection();
    let resources = checked.registered_artifacts().resources();
    let retained_host = checked
        .known_retained_host_bytes()
        .map_err(CheckError::GraphBody)?;
    Ok(json!({
        "schema": REPORT_SCHEMA,
        "status": "checked_static_graph_body",
        "scope": STATIC_SCOPE,
        "manifest": {"bytes": registration.manifest.bytes, "sha256": hex(registration.manifest.sha256)},
        "graph": {"bytes": body.graph_bytes, "sha256": hex(body.graph_sha256)},
        "byte_verification_scope": byte_scope,
        "graph_body": {
            "inspector_version": PACKING_GRAPH_BODY_INSPECTOR_VERSION,
            "verification_scope": format!("{:?}", checked.verification_scope()),
            "node_count": body.node_count, "input_count": body.input_count, "output_count": body.output_count,
            "value_info_count": body.value_info_count,
            "integer_control_initializer_count": body.integer_control_initializer_count,
            "wire_fields_and_packed_elements": body.wire_fields_and_packed_elements,
            "inspection_host_reserved_bytes": body.inspection_host_reserved_bytes
        },
        "limits": {"manifest_read_bytes": DEVICE_PACKING_MAX_MANIFEST_BYTES, "graph_read_bytes": DEVICE_PACKING_MAX_GRAPH_BYTES,
            "max_owned_artifact_host_bytes": MAX_ARTIFACT_OWNER_BYTES, "max_declared_host_bytes": MAX_DECLARED_HOST_BYTES,
            "max_declared_device_bytes": MAX_DECLARED_DEVICE_BYTES,
            "max_inspection_host_bytes": PACKING_GRAPH_MAX_INSPECTION_HOST_BYTES, "max_wire_fields": PACKING_GRAPH_MAX_WIRE_FIELDS},
        "resources": {
            "packing_session_sentinel_bytes": SESSION_SENTINEL_BYTES,
            "additional_owner_metadata_sentinel_bytes": METADATA_SENTINEL_BYTES,
            "sentinel_meaning": "nonzero_static_declaration_only;no_native_session_or_allocation_observed",
            "owned_artifact_host_bytes": resources.owned_artifact_host_bytes,
            "known_retained_host_bytes": retained_host,
            "controls_mask_and_finite_host_bytes": resources.controls_mask_and_finite_host_bytes,
            "maximum_node_output_payload_sum": resources.maximum_node_output_payload_sum,
            "declared_host_bytes": resources.declared_host_bytes, "declared_device_bytes": resources.declared_device_bytes,
            "accounting_scope": "known_backing_and_fixed_scratch_declarations_only;not_process_peak_OR_ORT_OR_CUDA_residency"
        },
        "read_buffer_owners_released_before_graph_inspection": true,
        "file_scope": "opened_regular_file_and_registered_content_hash;no_general_path_or_inode_certificate",
        "symlink_guard": "all_path_components_checked_before_and_after;Linux_final_NOFOLLOW_or_Windows_reparse_handle_check",
        "native_verification": "NotPerformed", "native_authorization": false,
        "byte_producer_authenticity": false, "ort_session_constructed": false,
        "provider_placement_checked": false, "numerical_results_checked": false,
        "run_performed": false, "physical_completion_observed": false, "cuda_observed": false
    }))
}

fn write_report(mut writer: impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut writer, value).map_err(CheckError::Json)?;
    writer.write_all(b"\n").map_err(CheckError::Output)?;
    writer.flush().map_err(CheckError::Output)
}

fn main() -> ExitCode {
    let result = parse_arguments(std::env::args_os().skip(1))
        .and_then(check)
        .and_then(|report| write_report(io::stdout().lock(), &report));
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let diagnostic = json!({"schema": REPORT_SCHEMA, "status": "error", "scope": STATIC_SCOPE,
                "stage": error.stage(), "detail": error.detail(), "native_verification": "NotPerformed",
                "native_authorization": false, "byte_producer_authenticity": false});
            let _ = write_report(io::stderr().lock(), &diagnostic);
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ABC_SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn positional_arguments() -> Vec<OsString> {
        let root = std::env::temp_dir();
        vec![
            root.join("device-packing.json").into_os_string(),
            root.join("device_public_pack.onnx").into_os_string(),
            ABC_SHA.into(),
            ABC_SHA.into(),
        ]
    }

    #[test]
    fn parser_requires_four_absolute_paths_and_exact_lowercase_pins() {
        let parsed = parse_arguments(positional_arguments()).unwrap();
        assert_eq!(hex(parsed.graph_sha256), ABC_SHA);
        assert_eq!(hex(parsed.manifest_sha256), ABC_SHA);
        for bad in [vec![], positional_arguments()[..3].to_vec()] {
            assert!(matches!(
                parse_arguments(bad),
                Err(CheckError::Arguments(_))
            ));
        }
        let mut extra = positional_arguments();
        extra.push("--run".into());
        assert!(matches!(
            parse_arguments(extra),
            Err(CheckError::Arguments(_))
        ));
        let mut relative = positional_arguments();
        relative[0] = "device-packing.json".into();
        assert!(matches!(
            parse_arguments(relative),
            Err(CheckError::Arguments(_))
        ));
        for invalid in [
            ABC_SHA.to_uppercase(),
            format!(" {ABC_SHA}"),
            "0".repeat(63),
            "g".repeat(64),
        ] {
            let mut arguments = positional_arguments();
            arguments[3] = invalid.into();
            assert!(matches!(
                parse_arguments(arguments),
                Err(CheckError::Arguments(_))
            ));
        }
    }

    #[test]
    fn report_delivery_error_is_not_success() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "synthetic closed report transport",
                ))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        assert!(write_report(Broken, &json!({"status": "checked_static_graph_body"})).is_err());
    }
}
