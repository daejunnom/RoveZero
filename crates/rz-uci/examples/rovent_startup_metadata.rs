//! Bounded, declaration-only metadata for an original Rovent startup JSON file.
//!
//! Usage: `rovent_startup_metadata --config=PATH.json [--backend-metadata=PATH.json]`
//! The existing startup codec owns validation, canonicalization and domain hashes.
//! This example never opens referenced assets or creates a native/runtime owner.

#![forbid(unsafe_code)]

#[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
fn main() -> std::process::ExitCode {
    use std::io::Write as _;
    let _ = std::io::stderr().lock().write_all(
        b"{\"schema\":\"rz-rovent-startup-metadata-error/1\",\"code\":\"unsupported_build_requires_pals_adaptive_and_onnx_cpu\",\"input_read\":false,\"actual_startup_ready\":false,\"native_authority\":false,\"training\":false}\n",
    );
    std::process::ExitCode::from(2)
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
fn main() -> std::process::ExitCode {
    metadata::main()
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod metadata {
    use rz_eval::error::BackendError;
    use rz_experiments::RoventDevice;
    use rz_uci::rovent_bootstrap::configuration::{
        LoadingProfile, ROVENT_STARTUP_CONFIGURATION_DOMAIN, StartupConfigurationDocument,
        v2::ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN, v3::ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN,
    };
    use rz_uci::rovent_bootstrap::configuration_digest_from_metadata;
    use serde::de::{MapAccess, SeqAccess, Visitor};
    use serde::{Deserialize, Deserializer, Serialize};
    use sha2::{Digest as _, Sha256};
    use std::collections::BTreeSet;
    use std::ffi::OsString;
    use std::fmt;
    use std::fs::File;
    use std::io::{self, Read, Write};
    use std::path::PathBuf;
    use std::process::ExitCode;

    const INPUT_MAX_BYTES: usize = 64 << 10;
    const OUTPUT_MAX_BYTES: usize = 256 << 10;
    // A failed/partial stdout write may still be followed by this one diagnostic.
    const DIAGNOSTIC_MAX_BYTES: usize = 4 << 10;
    const STDOUT_MAX_BYTES: usize = OUTPUT_MAX_BYTES - DIAGNOSTIC_MAX_BYTES;
    const ERROR_DETAIL_MAX_BYTES: usize = 512;

    #[derive(Debug)]
    enum Error {
        Arguments,
        InputKind,
        InputBudget,
        BackendDeclaration(&'static str),
        Io {
            stage: &'static str,
            cause: io::Error,
        },
        Json(serde_json::Error),
        Configuration(Box<BackendError>),
        CanonicalUtf8(std::str::Utf8Error),
        OutputEncoding(serde_json::Error),
        OutputBudget,
    }
    impl Error {
        fn code(&self) -> &'static str {
            match self {
                Self::Arguments => "invalid_arguments",
                Self::InputKind => "unsupported_input_file",
                Self::InputBudget => "configuration_byte_budget",
                Self::BackendDeclaration(_) => "backend_metadata_declaration_rejected",
                Self::Io { stage, .. } => stage,
                Self::Json(_) => "invalid_or_duplicate_json",
                Self::Configuration(_) => "startup_configuration_rejected",
                Self::CanonicalUtf8(_) => "canonical_utf8_rejected",
                Self::OutputEncoding(_) => "metadata_encoding_failed",
                Self::OutputBudget => "metadata_output_budget",
            }
        }
    }
    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Arguments => f.write_str("expected --config=PATH.json and optional --backend-metadata=PATH.json exactly once each"),
                Self::InputKind => f.write_str("input must be a regular non-symlink JSON file"),
                Self::InputBudget => f.write_str("each input must contain 1..=65536 bytes"),
                Self::BackendDeclaration(detail) => f.write_str(detail),
                Self::Io { stage, cause } => write!(f, "{stage}: {cause}"),
                Self::Json(cause) => write!(f, "JSON syntax or duplicate key: {cause}"),
                Self::Configuration(cause) => write!(f, "startup declaration: {cause}"),
                Self::CanonicalUtf8(cause) => write!(f, "canonical JSON UTF-8: {cause}"),
                Self::OutputEncoding(cause) => write!(f, "metadata JSON: {cause}"),
                Self::OutputBudget => f.write_str("metadata exceeds reserved stdout budget"),
            }
        }
    }
    fn io_error(stage: &'static str, cause: io::Error) -> Error {
        Error::Io { stage, cause }
    }
    fn configuration_error(cause: BackendError) -> Error {
        Error::Configuration(Box::new(cause))
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Arguments {
        configuration: PathBuf,
        backend_metadata: Option<PathBuf>,
    }
    fn json_path(text: &str) -> Result<PathBuf, Error> {
        if text.is_empty() {
            return Err(Error::Arguments);
        }
        let path = PathBuf::from(text);
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            return Err(Error::Arguments);
        }
        Ok(path)
    }
    fn parse_arguments(arguments: impl IntoIterator<Item = OsString>) -> Result<Arguments, Error> {
        let mut configuration = None;
        let mut backend_metadata = None;
        for argument in arguments {
            let text = argument.to_str().ok_or(Error::Arguments)?;
            if let Some(path) = text.strip_prefix("--config=") {
                if configuration.is_some() {
                    return Err(Error::Arguments);
                }
                configuration = Some(json_path(path)?);
            } else if let Some(path) = text.strip_prefix("--backend-metadata=") {
                if backend_metadata.is_some() {
                    return Err(Error::Arguments);
                }
                backend_metadata = Some(json_path(path)?);
            } else {
                return Err(Error::Arguments);
            }
        }
        Ok(Arguments {
            configuration: configuration.ok_or(Error::Arguments)?,
            backend_metadata,
        })
    }

    fn read_configuration(path: &std::path::Path) -> Result<Vec<u8>, Error> {
        let before = std::fs::symlink_metadata(path).map_err(|e| io_error("input_metadata", e))?;
        if !before.file_type().is_file() || before.file_type().is_symlink() {
            return Err(Error::InputKind);
        }
        if before.len() == 0 || before.len() > INPUT_MAX_BYTES as u64 {
            return Err(Error::InputBudget);
        }
        let file = File::open(path).map_err(|e| io_error("input_open", e))?;
        if !file
            .metadata()
            .map_err(|e| io_error("opened_input_metadata", e))?
            .is_file()
        {
            return Err(Error::InputKind);
        }
        let mut bytes = Vec::with_capacity(before.len() as usize);
        file.take((INPUT_MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| io_error("input_read", e))?;
        check_input_budget(&bytes)?;
        // The pin below identifies these actual read bytes, not an immutable
        // filesystem snapshot or the identity of any referenced native asset.
        Ok(bytes)
    }
    fn check_input_budget(bytes: &[u8]) -> Result<(), Error> {
        if bytes.is_empty() || bytes.len() > INPUT_MAX_BYTES {
            Err(Error::InputBudget)
        } else {
            Ok(())
        }
    }

    // This pass stores only keys of the current object. It adds no startup
    // encoder/schema; the existing typed codec still decides all fields/values.
    // Decoded key comparison also rejects escaped spellings of the same key.
    struct UniqueJson;
    impl<'de> Deserialize<'de> for UniqueJson {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            deserializer.deserialize_any(UniqueVisitor)
        }
    }
    struct UniqueVisitor;
    impl<'de> Visitor<'de> for UniqueVisitor {
        type Value = UniqueJson;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("JSON with unique object keys")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
            let mut keys = BTreeSet::new();
            while let Some(key) = access.next_key::<String>()? {
                if !keys.insert(key) {
                    return Err(serde::de::Error::custom("duplicate JSON object key"));
                }
                access.next_value::<UniqueJson>()?;
            }
            Ok(UniqueJson)
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
            while access.next_element::<UniqueJson>()?.is_some() {}
            Ok(UniqueJson)
        }
        fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
        fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
        fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
        fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
        fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
        fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            Ok(UniqueJson)
        }
    }

    #[derive(Serialize)]
    struct BytePin {
        bytes: usize,
        sha256: String,
    }
    fn hex(bytes: &[u8]) -> String {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut text = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            text.push(DIGITS[(byte >> 4) as usize] as char);
            text.push(DIGITS[(byte & 15) as usize] as char);
        }
        text
    }
    fn byte_pin(bytes: &[u8]) -> BytePin {
        BytePin {
            bytes: bytes.len(),
            sha256: hex(&Sha256::digest(bytes)),
        }
    }
    // Required tagged declarations distinguish explicit absence from omission.
    // No default, path scan or runtime hash is substituted for a missing value.
    #[derive(Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
    enum DeclaredDigest {
        Absent {},
        Sha256 { value: String },
    }
    impl DeclaredDigest {
        fn digest(&self) -> Result<Option<[u8; 32]>, Error> {
            match self {
                Self::Absent {} => Ok(None),
                Self::Sha256 { value } => declared_sha256(value).map(Some),
            }
        }
    }
    #[derive(Deserialize)]
    #[serde(tag = "kind", deny_unknown_fields)]
    enum DeclaredLoadingProfile {
        #[serde(rename = "absent")]
        Absent {},
        #[serde(rename = "eager_cuda12_cudnn9_v1")]
        EagerCuda12Cudnn9V1 {},
        #[serde(rename = "cu_dnn_shim_lazy_v1")]
        CuDnnShimLazyV1 {},
    }
    impl DeclaredLoadingProfile {
        fn selected(&self) -> Option<LoadingProfile> {
            match self {
                Self::Absent {} => None,
                Self::EagerCuda12Cudnn9V1 {} => Some(LoadingProfile::EagerCuda12Cudnn9V1),
                Self::CuDnnShimLazyV1 {} => Some(LoadingProfile::CuDnnShimLazyV1),
            }
        }
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct BackendInput {
        domain: String,
        device: RoventDevice,
        runtime_binary_sha256: String,
        runtime_bundle_sha256: DeclaredDigest,
        control_inventory_sha256: DeclaredDigest,
        loading_profile: DeclaredLoadingProfile,
    }
    fn declared_sha256(text: &str) -> Result<[u8; 32], Error> {
        if text.len() != 64
            || !text
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::BackendDeclaration(
                "SHA256 declarations require 64 lowercase hexadecimal characters",
            ));
        }
        let mut bytes = [0; 32];
        for (out, pair) in bytes.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
            let nibble = |b: u8| {
                if b.is_ascii_digit() {
                    b - b'0'
                } else {
                    b - b'a' + 10
                }
            };
            *out = (nibble(pair[0]) << 4) | nibble(pair[1]);
        }
        if bytes == [0; 32] {
            return Err(Error::BackendDeclaration(
                "an all-zero placeholder is not an independently registered runtime identity",
            ));
        }
        Ok(bytes)
    }
    #[derive(Serialize)]
    struct BackendMetadata {
        scope: &'static str,
        source_metadata: BytePin,
        backend_configuration_sha256: String,
        declared_runtime_binary_sha256: String,
        declared_runtime_bundle_sha256: Option<String>,
        declared_control_inventory_sha256: Option<String>,
        actual_runtime_identity_verified: bool,
        actual_provider_verified: bool,
    }
    fn prepare_backend(
        document: &StartupConfigurationDocument,
        raw: &[u8],
    ) -> Result<BackendMetadata, Error> {
        check_input_budget(raw)?;
        serde_json::from_slice::<UniqueJson>(raw).map_err(Error::Json)?;
        let input: BackendInput = serde_json::from_slice(raw).map_err(Error::Json)?;
        if input.domain != "rz-rovent-backend-metadata-input/1" {
            return Err(Error::BackendDeclaration(
                "unsupported backend metadata domain",
            ));
        }
        let native = document.fields().native;
        if input.loading_profile.selected() != native.loading_profile {
            return Err(Error::BackendDeclaration(
                "backend loading profile differs from the original startup document",
            ));
        }
        let binary = declared_sha256(&input.runtime_binary_sha256)?;
        let bundle = input.runtime_bundle_sha256.digest()?;
        let control = input.control_inventory_sha256.digest()?;
        // Presence means a declared explicit control selection only. It does
        // not establish inventory bytes, a loaded provider or GPU capability.
        let configuration = native
            .onnx(&input.device, control.is_some())
            .map_err(configuration_error)?;
        let digest = configuration_digest_from_metadata(
            configuration,
            native.loading_profile.map(LoadingProfile::actual),
            binary,
            bundle,
            control,
        )
        .map_err(configuration_error)?;
        Ok(BackendMetadata {
            scope: "independently_supplied_declarations_only",
            source_metadata: byte_pin(raw),
            backend_configuration_sha256: hex(&digest),
            declared_runtime_binary_sha256: hex(&binary),
            declared_runtime_bundle_sha256: bundle.map(|d| hex(&d)),
            declared_control_inventory_sha256: control.map(|d| hex(&d)),
            actual_runtime_identity_verified: false,
            actual_provider_verified: false,
        })
    }
    #[derive(Serialize)]
    struct Checks {
        unique_json_object_keys: &'static str,
        existing_closed_codec_and_declaration_constraints: &'static str,
        actual_lock_join: &'static str,
        compiled_native_capability_admission: &'static str,
        referenced_asset_identity: &'static str,
        backend_runtime_configuration_digest: &'static str,
        native_loading_provider_execution: &'static str,
        source_file_snapshot_stability: &'static str,
    }
    #[derive(Serialize)]
    struct Metadata<'a> {
        schema: &'static str,
        scope: &'static str,
        configuration_document_domain: &'static str,
        source_configuration: BytePin,
        canonical_configuration: BytePin,
        canonical_configuration_json: &'a str,
        configuration_sha256: String,
        search_configuration_sha256: String,
        nnuec_conditions_sha256: Option<String>,
        backend_metadata: Option<BackendMetadata>,
        checks: Checks,
        actual_startup_ready: bool,
        native_authority: bool,
        training: bool,
    }

    fn prepare_metadata(bytes: &[u8], backend_raw: Option<&[u8]>) -> Result<Vec<u8>, Error> {
        check_input_budget(bytes)?;
        serde_json::from_slice::<UniqueJson>(bytes).map_err(Error::Json)?;
        let document =
            StartupConfigurationDocument::from_json(bytes).map_err(configuration_error)?;
        document.validate().map_err(configuration_error)?;
        let canonical = document.canonical_bytes().map_err(configuration_error)?;
        let backend_metadata = backend_raw
            .map(|raw| prepare_backend(&document, raw))
            .transpose()?;
        let backend_scope = if backend_metadata.is_some() {
            "declaration_digest_only_actual_identity_pending"
        } else {
            "pending_independent_runtime_metadata"
        };
        let output = Metadata {
            schema: "rz-rovent-startup-metadata/1",
            scope: "startup_configuration_declarations_only",
            configuration_document_domain: match &document {
                StartupConfigurationDocument::V1(_) => ROVENT_STARTUP_CONFIGURATION_DOMAIN,
                StartupConfigurationDocument::V2(_) => ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN,
                StartupConfigurationDocument::V3(_) => ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN,
            },
            source_configuration: byte_pin(bytes),
            canonical_configuration: byte_pin(&canonical),
            canonical_configuration_json: std::str::from_utf8(&canonical)
                .map_err(Error::CanonicalUtf8)?,
            configuration_sha256: hex(&document
                .configuration_digest()
                .map_err(configuration_error)?),
            search_configuration_sha256: hex(&document
                .search_configuration_digest()
                .map_err(configuration_error)?),
            nnuec_conditions_sha256: document
                .nnuec_conditions_digest()
                .map_err(configuration_error)?
                .map(|d| hex(&d)),
            backend_metadata,
            checks: Checks {
                unique_json_object_keys: "passed",
                existing_closed_codec_and_declaration_constraints: "passed",
                actual_lock_join: "pending_external_inputs",
                compiled_native_capability_admission: "not_performed",
                referenced_asset_identity: "not_performed",
                backend_runtime_configuration_digest: backend_scope,
                native_loading_provider_execution: "not_performed",
                source_file_snapshot_stability: "not_claimed",
            },
            actual_startup_ready: false,
            native_authority: false,
            training: false,
        };
        let mut encoded = serde_json::to_vec(&output).map_err(Error::OutputEncoding)?;
        encoded.push(b'\n');
        enforce_stdout_budget(encoded)
    }
    fn enforce_stdout_budget(bytes: Vec<u8>) -> Result<Vec<u8>, Error> {
        if bytes.len() > STDOUT_MAX_BYTES {
            Err(Error::OutputBudget)
        } else {
            Ok(bytes)
        }
    }

    #[derive(Serialize)]
    struct Diagnostic<'a> {
        schema: &'static str,
        code: &'static str,
        detail_prefix: &'a str,
        detail_truncated: bool,
        metadata_stdout_complete: bool,
        actual_startup_ready: bool,
        native_authority: bool,
        training: bool,
    }
    fn diagnostic(error: &Error) -> Vec<u8> {
        // Causes remain owned by `error`; only the displayed prefix is bounded.
        let detail = error.to_string();
        let mut end = detail.len().min(ERROR_DETAIL_MAX_BYTES);
        while !detail.is_char_boundary(end) {
            end -= 1;
        }
        let value = Diagnostic {
            schema: "rz-rovent-startup-metadata-error/1",
            code: error.code(),
            detail_prefix: &detail[..end],
            detail_truncated: end != detail.len(),
            metadata_stdout_complete: false,
            actual_startup_ready: false,
            native_authority: false,
            training: false,
        };
        let fallback = b"{\"schema\":\"rz-rovent-startup-metadata-error/1\",\"code\":\"bounded_diagnostic_encoding_failed\",\"actual_startup_ready\":false,\"native_authority\":false,\"training\":false}\n";
        match serde_json::to_vec(&value) {
            Ok(mut bytes) if bytes.len() < DIAGNOSTIC_MAX_BYTES => {
                bytes.push(b'\n');
                bytes
            }
            _ => fallback.to_vec(),
        }
    }
    fn run() -> Result<(), Error> {
        let arguments = parse_arguments(std::env::args_os().skip(1))?;
        let input = read_configuration(&arguments.configuration)?;
        let backend = arguments
            .backend_metadata
            .as_deref()
            .map(read_configuration)
            .transpose()?;
        let output = prepare_metadata(&input, backend.as_deref())?;
        let mut stdout = io::stdout().lock();
        stdout
            .write_all(&output)
            .map_err(|e| io_error("stdout_write", e))?;
        stdout.flush().map_err(|e| io_error("stdout_flush", e))
    }
    pub(super) fn main() -> ExitCode {
        match run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = io::stderr().lock().write_all(&diagnostic(&error));
                ExitCode::from(2)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // Explicit synthetic constructor declarations, not a product asset/lock.
        fn fixture(domain: &str, version_fields: &str) -> Vec<u8> {
            r#"{
              "domain":"__DOMAIN__",
              "semantic":{"beam_width":4,"line_plies":4,"max_nodes":4096,"max_records":64,"max_role_calls":4096,"maximum_root_prefix_plies":256},
              "review":{"anchor_mode":"anchored_compact","token_budget":512,"max_records":64,"max_anchors":4},
              "index":{"active_bundles":256,"records_per_bundle":64},
              "frontier":{"maximum_goal_plies":256,"maximum_frames":64,"maximum_paths":4096,"maximum_mate_nodes":4096,"maximum_native_receipts":64},
              "native":{"cache_public_memory":true,"device_public_memory":false,"cudnn_conv_search":"legacy_implicit_exhaustive","loading_profile":null,"cuda_arena_bytes":null,"drain_ms":30000,"prepared_host_bytes_per_ticket":1048576,"output_host_bytes_per_ticket":1048576,"command_capacity":8},
              "execution":{"assembly":"roles_only","event_order":"fixed","role_tickets_max":16,"shared_nnuec_readout":null}__VERSION_FIELDS__
            }"#.replace("__DOMAIN__", domain).replace("__VERSION_FIELDS__", version_fields).into_bytes()
        }
        #[test]
        fn exact_arguments_reject_missing_extra_unknown_and_non_json() {
            assert!(parse_arguments(Vec::<OsString>::new()).is_err());
            for args in [
                vec!["--help"],
                vec!["--config="],
                vec!["--config=.env"],
                vec!["--config=a.json", "--other=1"],
                vec!["--config=a.json", "--config=b.json"],
                vec!["--backend-metadata=a.json"],
                vec![
                    "--config=a.json",
                    "--backend-metadata=a.json",
                    "--backend-metadata=b.json",
                ],
            ] {
                assert!(parse_arguments(args.into_iter().map(OsString::from)).is_err());
            }
            assert_eq!(
                parse_arguments([OsString::from("--config=선언.json")]).unwrap(),
                Arguments {
                    configuration: PathBuf::from("선언.json"),
                    backend_metadata: None
                }
            );
            assert_eq!(
                parse_arguments(
                    ["--backend-metadata=b.json", "--config=a.json"].map(OsString::from)
                )
                .unwrap(),
                Arguments {
                    configuration: PathBuf::from("a.json"),
                    backend_metadata: Some(PathBuf::from("b.json"))
                }
            );
        }
        #[test]
        fn byte_bounds_and_aggregate_output_reservation_are_explicit() {
            assert!(matches!(
                prepare_metadata(&[], None),
                Err(Error::InputBudget)
            ));
            assert!(matches!(
                prepare_metadata(&vec![b' '; INPUT_MAX_BYTES + 1], None),
                Err(Error::InputBudget)
            ));
            assert!(enforce_stdout_budget(vec![0; STDOUT_MAX_BYTES]).is_ok());
            assert!(matches!(
                enforce_stdout_budget(vec![0; STDOUT_MAX_BYTES + 1]),
                Err(Error::OutputBudget)
            ));
            assert_eq!(STDOUT_MAX_BYTES + DIAGNOSTIC_MAX_BYTES, OUTPUT_MAX_BYTES);
        }
        #[test]
        fn duplicate_keys_include_nested_and_escaped_spellings() {
            for raw in [
                br#"{"a":1,"\u0061":2}"#.as_slice(),
                br#"{"outer":[{"x":1,"x":2}]}"#.as_slice(),
            ] {
                assert!(serde_json::from_slice::<UniqueJson>(raw).is_err());
            }
            assert!(
                serde_json::from_slice::<UniqueJson>(br#"{"one":{"x":1},"two":{"x":2}}"#).is_ok()
            );
        }
        #[test]
        fn existing_codec_rejects_unknown_fields_invalid_values_and_extra_json() {
            let raw = String::from_utf8(fixture(ROVENT_STARTUP_CONFIGURATION_DOMAIN, "")).unwrap();
            for rejected in [
                raw.replace("\"max_nodes\":4096", "\"max_nodes\":4096,\"unknown\":1"),
                raw.replace("\"line_plies\":4", "\"line_plies\":0"),
                format!("{raw} {{}}"),
            ] {
                assert!(prepare_metadata(rejected.as_bytes(), None).is_err());
            }
        }
        #[derive(Deserialize)]
        struct Observed {
            configuration_document_domain: String,
            canonical_configuration_json: String,
            configuration_sha256: String,
            search_configuration_sha256: String,
            nnuec_conditions_sha256: Option<String>,
            actual_startup_ready: bool,
            native_authority: bool,
            training: bool,
        }
        #[test]
        fn all_versions_use_existing_digests_without_runtime_authority() {
            for (domain, fields) in [
                (ROVENT_STARTUP_CONFIGURATION_DOMAIN, ""),
                (
                    ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN,
                    r#", "post_repair":"disabled""#,
                ),
                (
                    ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN,
                    r#", "post_repair":"disabled", "external":{"kind":"absent"}"#,
                ),
            ] {
                let raw = fixture(domain, fields);
                let original = StartupConfigurationDocument::from_json(&raw).unwrap();
                let output = prepare_metadata(&raw, None).unwrap();
                let observed: Observed = serde_json::from_slice(&output).unwrap();
                assert_eq!(observed.configuration_document_domain, domain);
                assert_eq!(
                    observed.canonical_configuration_json.as_bytes(),
                    original.canonical_bytes().unwrap()
                );
                assert_eq!(
                    observed.configuration_sha256,
                    hex(&original.configuration_digest().unwrap())
                );
                assert_eq!(
                    observed.search_configuration_sha256,
                    hex(&original.search_configuration_digest().unwrap())
                );
                assert_eq!(
                    observed.nnuec_conditions_sha256,
                    original.nnuec_conditions_digest().unwrap().map(|d| hex(&d))
                );
                assert!(
                    !observed.actual_startup_ready
                        && !observed.native_authority
                        && !observed.training
                );
                assert!(output.len() <= STDOUT_MAX_BYTES);
            }
        }
        fn backend_fixture() -> Vec<u8> {
            // A declaration-only synthetic digest, not an actual native asset.
            format!(r#"{{"domain":"rz-rovent-backend-metadata-input/1","device":{{"device":"cpu"}},"runtime_binary_sha256":"{}","runtime_bundle_sha256":{{"kind":"absent"}},"control_inventory_sha256":{{"kind":"absent"}},"loading_profile":{{"kind":"absent"}}}}"#, "11".repeat(32)).into_bytes()
        }
        #[test]
        fn optional_backend_digest_reuses_native_typed_declaration_codec() {
            let startup = fixture(ROVENT_STARTUP_CONFIGURATION_DOMAIN, "");
            let document = StartupConfigurationDocument::from_json(&startup).unwrap();
            let raw = backend_fixture();
            let output = prepare_backend(&document, &raw).unwrap();
            let native = document.fields().native;
            let existing_config = native.onnx(&RoventDevice::Cpu, false).unwrap();
            let existing_digest =
                configuration_digest_from_metadata(existing_config, None, [0x11; 32], None, None)
                    .unwrap();
            assert_eq!(output.backend_configuration_sha256, hex(&existing_digest));
            assert_eq!(output.source_metadata.sha256, byte_pin(&raw).sha256);
            assert_eq!(output.source_metadata.bytes, raw.len());
            assert_eq!(output.scope, "independently_supplied_declarations_only");
            assert!(!output.actual_runtime_identity_verified && !output.actual_provider_verified);
            assert!(prepare_metadata(&startup, Some(&raw)).is_ok());
        }
        #[test]
        fn backend_requires_explicit_absence_and_matching_profile_and_device() {
            let startup = fixture(ROVENT_STARTUP_CONFIGURATION_DOMAIN, "");
            let document = StartupConfigurationDocument::from_json(&startup).unwrap();
            let raw = String::from_utf8(backend_fixture()).unwrap();
            for rejected in [
                raw.replace("\"runtime_bundle_sha256\":{\"kind\":\"absent\"},", ""),
                raw.replace("\"control_inventory_sha256\":{\"kind\":\"absent\"}", "\"control_inventory_sha256\":null"),
                raw.replace("\"loading_profile\":{\"kind\":\"absent\"}", "\"loading_profile\":{\"kind\":\"cu_dnn_shim_lazy_v1\"}"),
                raw.replace("\"device\":{\"device\":\"cpu\"}", "\"device\":{\"device\":\"cuda\",\"ordinal\":0,\"registered_device\":\"synthetic\"}"),
                raw.replace("\"kind\":\"absent\"", "\"kind\":\"absent\",\"extra\":1"),
                raw.replace("\"domain\":", "\"domain\":\"rz-rovent-backend-metadata-input/1\",\"domain\":"),
                raw.replace("rz-rovent-backend-metadata-input/1", "rz-rovent-backend-metadata-input/2"),
            ] {
                assert!(prepare_backend(&document, rejected.as_bytes()).is_err());
            }
            assert!(matches!(
                prepare_backend(&document, &vec![b' '; INPUT_MAX_BYTES + 1]),
                Err(Error::InputBudget)
            ));
            for rejected in [
                "0".repeat(64),
                "a".repeat(63),
                "A".repeat(64),
                "g".repeat(64),
            ] {
                assert!(declared_sha256(&rejected).is_err());
            }
        }
        #[test]
        fn diagnostics_are_bounded_and_cannot_promote_a_failed_output() {
            let error = Error::Io {
                stage: "stdout_flush",
                cause: io::Error::other("한".repeat(5000)),
            };
            let bytes = diagnostic(&error);
            assert!(bytes.len() <= DIAGNOSTIC_MAX_BYTES);
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(text.contains("\"detail_truncated\":true"));
            assert!(text.contains("\"metadata_stdout_complete\":false"));
            assert!(text.contains("\"native_authority\":false"));
        }
    }
}
