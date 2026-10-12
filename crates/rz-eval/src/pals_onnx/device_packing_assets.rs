//! Guarded file ownership for the fixed public packing graph's static proof.
//!
//! Caller SHA-256 pins establish byte identity, not producer authenticity. The
//! caller must supply every resource declaration and inspection budget; this
//! module supplies no default session, sentinel, provider or native Run authority.
//! Regular-file/ancestor checks and platform open guards are not a general path
//! certificate. Windows metadata equality is not globally unique file identity.
//! No ORT session, device allocation, provider audit, numerical Run or physical
//! completion is performed. These bounded reads are not a process-peak or hard
//! real-time filesystem latency guarantee.
use super::device_packing_admission::{
    DevicePackingAdmissionError, DevicePackingResourceDeclaration, PackingArtifactPart,
    PackingArtifactRegistration, RegisteredPackingArtifactBytes, RegisteredPackingBytePin,
    DEVICE_PACKING_MAX_GRAPH_BYTES, DEVICE_PACKING_MAX_MANIFEST_BYTES,
    DEVICE_PACKING_MAX_NODE_PAYLOAD_SUM,
};
use super::device_packing_graph::{
    CheckedFixedPackingGraph, PackingGraphBodyError, PackingGraphInspectionBudget,
    PACKING_GRAPH_MAX_INSPECTION_HOST_BYTES, PACKING_GRAPH_MAX_WIRE_FIELDS,
};
use sha2::{Digest as _, Sha256};
use std::collections::TryReserveError;
use std::fmt;
#[cfg(any(target_os = "linux", windows))]
use std::fs::OpenOptions;
use std::fs::{self, File, Metadata};
use std::io::{self, Read};
use std::mem::size_of;
use std::path::{Component, Path};

// The fixed declaration's 129 INT64 offsets/stop, 194 BOOL mask elements and
// one BOOL finite scalar. This lower bound performs no allocation observation;
// byte admission remains responsible for the actual owned Vec capacities.
const FIXED_CONTROL_MASK_FINITE_HOST_BYTES: u64 = 129 * 8 + 194 + 1;

#[derive(Debug)]
pub enum DevicePackingAssetError {
    InvalidPath {
        part: PackingArtifactPart,
        reason: &'static str,
    },
    Filesystem {
        part: PackingArtifactPart,
        stage: &'static str,
        cause: io::Error,
    },
    FileRejected {
        part: PackingArtifactPart,
        reason: &'static str,
    },
    Allocation {
        part: PackingArtifactPart,
        cause: TryReserveError,
    },
    BytePinMismatch {
        part: PackingArtifactPart,
    },
    ByteAdmission(DevicePackingAdmissionError),
    GraphBody(PackingGraphBodyError),
}

impl DevicePackingAssetError {
    pub fn stage(&self) -> &'static str {
        match self {
            Self::InvalidPath { .. } => "arguments",
            Self::Filesystem { stage, .. } => stage,
            Self::FileRejected { .. } | Self::Allocation { .. } => "file_admission",
            Self::BytePinMismatch { .. } => "byte_pin",
            Self::ByteAdmission(_) => "registered_byte_admission",
            Self::GraphBody(_) => "fixed_graph_body",
        }
    }

    /// Bounded diagnostic text; typed variants and Error::source retain cause.
    pub fn detail(&self) -> String {
        let text = match self {
            Self::InvalidPath { reason, .. } | Self::FileRejected { reason, .. } => {
                (*reason).to_owned()
            }
            Self::Filesystem { cause, .. } => cause.to_string(),
            Self::Allocation { .. } => "fallible artifact buffer reservation failed".to_owned(),
            Self::BytePinMismatch {
                part: PackingArtifactPart::Manifest,
            } => "manifest SHA-256 mismatch".to_owned(),
            Self::BytePinMismatch {
                part: PackingArtifactPart::Graph,
            } => "graph SHA-256 mismatch".to_owned(),
            Self::ByteAdmission(error) => error.to_string(),
            Self::GraphBody(error) => error.to_string(),
        };
        text.chars().take(512).collect()
    }
}

impl fmt::Display for DevicePackingAssetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "device packing asset {}: {}",
            self.stage(),
            self.detail()
        )
    }
}
impl std::error::Error for DevicePackingAssetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Filesystem { cause, .. } => Some(cause),
            Self::Allocation { cause, .. } => Some(cause),
            Self::ByteAdmission(error) => Some(error),
            Self::GraphBody(error) => Some(error),
            _ => None,
        }
    }
}
type Result<T> = std::result::Result<T, DevicePackingAssetError>;

/// Read one independently pinned manifest/graph pair and return static body proof.
///
/// Sizes are obtained from bounded regular-file reads and placed in the immutable
/// registration alongside the caller's SHA pins. The exact manifest must also
/// declare the graph's matching size/hash. Read Vecs are released before body
/// inspection; the returned proof retains only its admitted owned artifact pair.
/// Resources are caller declarations, never a measured session or device lease.
pub fn load_checked_fixed_packing_graph(
    manifest_path: &Path,
    graph_path: &Path,
    manifest_sha256: [u8; 32],
    graph_sha256: [u8; 32],
    declaration: DevicePackingResourceDeclaration,
    inspection_budget: PackingGraphInspectionBudget,
) -> Result<CheckedFixedPackingGraph> {
    validate_absolute_path(manifest_path, PackingArtifactPart::Manifest)?;
    validate_absolute_path(graph_path, PackingArtifactPart::Graph)?;
    // Reject invalid scalar declarations BEFORE metadata/open/read I/O. Exact
    // artifact Vec capacities and graph inspection backing are checked later by
    // their existing owners, rather than inferred from this minimum preflight.
    preflight_resources(declaration)?;
    preflight_inspection_budget(inspection_budget)?;
    let manifest = read_registered_file(
        manifest_path,
        DEVICE_PACKING_MAX_MANIFEST_BYTES,
        manifest_sha256,
        PackingArtifactPart::Manifest,
    )?;
    let graph = read_registered_file(
        graph_path,
        DEVICE_PACKING_MAX_GRAPH_BYTES,
        graph_sha256,
        PackingArtifactPart::Graph,
    )?;
    let registration = PackingArtifactRegistration {
        manifest: RegisteredPackingBytePin {
            bytes: manifest.len() as u64,
            sha256: manifest_sha256,
        },
        graph: RegisteredPackingBytePin {
            bytes: graph.len() as u64,
            sha256: graph_sha256,
        },
    };
    let registered =
        RegisteredPackingArtifactBytes::admit(&manifest, &graph, registration, declaration)
            .map_err(DevicePackingAssetError::ByteAdmission)?;
    drop(manifest);
    drop(graph);
    CheckedFixedPackingGraph::verify(registered, inspection_budget)
        .map_err(DevicePackingAssetError::GraphBody)
}

fn validate_absolute_path(path: &Path, part: PackingArtifactPart) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
    {
        return Err(DevicePackingAssetError::InvalidPath {
            part,
            reason: "artifact paths must be absolute without parent traversal",
        });
    }
    Ok(())
}

fn preflight_resources(declaration: DevicePackingResourceDeclaration) -> Result<()> {
    let required = |value: Option<u64>, name| {
        value
            .filter(|value| *value > 0)
            .ok_or(DevicePackingAssetError::ByteAdmission(
                DevicePackingAdmissionError::UnknownResource(name),
            ))
    };
    let session = required(declaration.packing_session_bytes, "packing_session_bytes")?;
    let metadata = required(
        declaration.additional_owner_metadata_host_bytes,
        "additional_owner_metadata_host_bytes",
    )?;
    let checked_sum = |left: u64, right: u64| {
        left.checked_add(right)
            .ok_or(DevicePackingAssetError::ByteAdmission(
                DevicePackingAdmissionError::Overflow,
            ))
    };
    let minimum_artifact = checked_sum(size_of::<RegisteredPackingArtifactBytes>() as u64, 2)?;
    let minimum_host = checked_sum(
        checked_sum(minimum_artifact, FIXED_CONTROL_MASK_FINITE_HOST_BYTES)?,
        metadata,
    )?;
    let minimum_device = checked_sum(DEVICE_PACKING_MAX_NODE_PAYLOAD_SUM, session)?;
    for (observed_minimum, maximum, name) in [
        (
            minimum_artifact,
            declaration.max_owned_artifact_host_bytes,
            "owned_artifact_host_bytes",
        ),
        (
            minimum_host,
            declaration.max_declared_host_bytes,
            "declared_host_bytes",
        ),
        (
            minimum_device,
            declaration.max_declared_device_bytes,
            "declared_device_bytes",
        ),
    ] {
        if observed_minimum > maximum {
            return Err(DevicePackingAssetError::ByteAdmission(
                DevicePackingAdmissionError::ResourceExceeded(name),
            ));
        }
    }
    Ok(())
}

fn preflight_inspection_budget(budget: PackingGraphInspectionBudget) -> Result<()> {
    if budget.max_inspection_host_bytes == 0
        || budget.max_inspection_host_bytes > PACKING_GRAPH_MAX_INSPECTION_HOST_BYTES
        || budget.max_wire_fields == 0
        || budget.max_wire_fields > PACKING_GRAPH_MAX_WIRE_FIELDS
    {
        return Err(DevicePackingAssetError::GraphBody(
            PackingGraphBodyError::InvalidBudget,
        ));
    }
    Ok(())
}

fn is_link_or_reparse(metadata: &Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn path_metadata(path: &Path, maximum: usize, part: PackingArtifactPart) -> Result<Metadata> {
    let mut selected = None;
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor).map_err(|cause| {
            DevicePackingAssetError::Filesystem {
                part,
                stage: "path_metadata",
                cause,
            }
        })?;
        if is_link_or_reparse(&metadata) {
            return Err(DevicePackingAssetError::FileRejected {
                part,
                reason: "symlink or reparse artifact path component",
            });
        }
        if ancestor == path {
            if !metadata.is_file() || metadata.len() == 0 || metadata.len() > maximum as u64 {
                return Err(DevicePackingAssetError::FileRejected {
                    part,
                    reason: "artifact must be a nonempty regular file within its fixed byte cap",
                });
            }
            selected = Some(metadata);
        } else if !metadata.is_dir() {
            return Err(DevicePackingAssetError::FileRejected {
                part,
                reason: "artifact ancestor must be a regular directory",
            });
        }
    }
    selected.ok_or(DevicePackingAssetError::FileRejected {
        part,
        reason: "artifact metadata missing",
    })
}

fn open_regular_candidate(path: &Path, part: PackingArtifactPart) -> Result<File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Linux O_NOFOLLOW | O_NONBLOCK, unchanged from the static CLI reader.
        OpenOptions::new()
            .read(true)
            .custom_flags(0x20000 | 0x800)
            .open(path)
            .map_err(|cause| DevicePackingAssetError::Filesystem {
                part,
                stage: "open",
                cause,
            })
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // OPEN_REPARSE_POINT and FILE_SHARE_READ; reject the opened reparse
        // handle, and disallow writer/delete sharing while this reader owns it.
        OpenOptions::new()
            .read(true)
            .custom_flags(0x00200000)
            .share_mode(0x00000001)
            .open(path)
            .map_err(|cause| DevicePackingAssetError::Filesystem {
                part,
                stage: "open",
                cause,
            })
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        let _ = path;
        Err(DevicePackingAssetError::FileRejected {
            part,
            reason: "guarded regular-file reader supports Linux and Windows only",
        })
    }
}

fn matching_metadata(before: &Metadata, after: &Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        before.dev() == after.dev()
            && before.ino() == after.ino()
            && before.len() == after.len()
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec()
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        // These fields are not a globally unique file-ID certificate.
        before.file_attributes() == after.file_attributes()
            && before.file_size() == after.file_size()
            && before.creation_time() == after.creation_time()
            && before.last_write_time() == after.last_write_time()
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (before, after);
        false
    }
}

fn read_registered_file(
    path: &Path,
    maximum: usize,
    expected_sha256: [u8; 32],
    part: PackingArtifactPart,
) -> Result<Vec<u8>> {
    let before = path_metadata(path, maximum, part)?;
    let mut file = open_regular_candidate(path, part)?;
    let opened = file
        .metadata()
        .map_err(|cause| DevicePackingAssetError::Filesystem {
            part,
            stage: "opened_metadata",
            cause,
        })?;
    if !opened.is_file() || is_link_or_reparse(&opened) || !matching_metadata(&before, &opened) {
        return Err(DevicePackingAssetError::FileRejected {
            part,
            reason: "opened artifact is not the prechecked regular file",
        });
    }
    let initial =
        usize::try_from(opened.len()).map_err(|_| DevicePackingAssetError::FileRejected {
            part,
            reason: "artifact length conversion overflow",
        })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(initial)
        .map_err(|cause| DevicePackingAssetError::Allocation { part, cause })?;
    let mut scratch = [0u8; 8192];
    loop {
        // Retain at most maximum bytes; one bounded nonretained probe catches
        // growth. Interrupted errors remain errors rather than retrying forever.
        let remaining =
            maximum
                .checked_sub(bytes.len())
                .ok_or(DevicePackingAssetError::FileRejected {
                    part,
                    reason: "artifact byte cap exceeded",
                })?;
        let probe = remaining
            .checked_add(1)
            .ok_or(DevicePackingAssetError::FileRejected {
                part,
                reason: "artifact probe bound overflow",
            })?;
        let wanted = scratch.len().min(probe);
        let count = file.read(&mut scratch[..wanted]).map_err(|cause| {
            DevicePackingAssetError::Filesystem {
                part,
                stage: "read",
                cause,
            }
        })?;
        if count == 0 {
            break;
        }
        if count > remaining {
            return Err(DevicePackingAssetError::FileRejected {
                part,
                reason: "artifact grew beyond its fixed byte cap",
            });
        }
        bytes
            .try_reserve_exact(count)
            .map_err(|cause| DevicePackingAssetError::Allocation { part, cause })?;
        bytes.extend_from_slice(&scratch[..count]);
    }
    let closed_read = file
        .metadata()
        .map_err(|cause| DevicePackingAssetError::Filesystem {
            part,
            stage: "read_metadata",
            cause,
        })?;
    let named_after = path_metadata(path, maximum, part)?;
    if bytes.is_empty()
        || bytes.len() != initial
        || !matching_metadata(&opened, &closed_read)
        || !matching_metadata(&closed_read, &named_after)
    {
        return Err(DevicePackingAssetError::FileRejected {
            part,
            reason: "artifact metadata or length changed during read",
        });
    }
    if <[u8; 32]>::from(Sha256::digest(&bytes)) != expected_sha256 {
        return Err(DevicePackingAssetError::BytePinMismatch { part });
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQUENCE: AtomicU64 = AtomicU64::new(1);
    const ABC_SHA: [u8; 32] = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad,
    ];

    struct OwnedFixtureDirectory(PathBuf);
    impl OwnedFixtureDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "rz-packing-assets-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            fs::write(&path, bytes).unwrap();
            path
        }
    }
    impl Drop for OwnedFixtureDirectory {
        fn drop(&mut self) {
            for name in ["abc", "empty", "large", "link"] {
                let _ = fs::remove_file(self.0.join(name));
            }
            let _ = fs::remove_dir(&self.0);
        }
    }

    #[test]
    fn regular_file_reader_checks_exact_cap_nonempty_and_independent_hash() {
        let owner = OwnedFixtureDirectory::new();
        let path = owner.file("abc", b"abc");
        assert_eq!(
            read_registered_file(&path, 3, ABC_SHA, PackingArtifactPart::Manifest)
                .unwrap()
                .as_slice(),
            b"abc"
        );
        assert!(matches!(
            read_registered_file(&path, 3, [0; 32], PackingArtifactPart::Graph),
            Err(DevicePackingAssetError::BytePinMismatch {
                part: PackingArtifactPart::Graph
            })
        ));
        let empty = owner.file("empty", b"");
        let large = owner.file("large", b"abcd");
        for selected in [&empty, &large, &owner.0] {
            assert!(matches!(
                read_registered_file(selected, 3, ABC_SHA, PackingArtifactPart::Manifest),
                Err(DevicePackingAssetError::FileRejected { .. })
            ));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn regular_file_reader_refuses_symbolic_link_even_when_target_hash_matches() {
        let owner = OwnedFixtureDirectory::new();
        let source = owner.file("abc", b"abc");
        let link = owner.0.join("link");
        std::os::unix::fs::symlink(source, &link).unwrap();
        assert!(matches!(
            read_registered_file(&link, 3, ABC_SHA, PackingArtifactPart::Manifest),
            Err(DevicePackingAssetError::FileRejected { .. })
        ));
    }

    fn declared_fixture_resources() -> DevicePackingResourceDeclaration {
        // Explicit finite fixture declarations, not observed native resources.
        DevicePackingResourceDeclaration {
            packing_session_bytes: Some(4096),
            additional_owner_metadata_host_bytes: Some(256),
            max_owned_artifact_host_bytes: 3 * 1024 * 1024,
            max_declared_host_bytes: 4 * 1024 * 1024,
            max_declared_device_bytes: 2 * 1024 * 1024,
        }
    }

    fn fixture_budget() -> PackingGraphInspectionBudget {
        PackingGraphInspectionBudget {
            max_inspection_host_bytes: PACKING_GRAPH_MAX_INSPECTION_HOST_BYTES,
            max_wire_fields: PACKING_GRAPH_MAX_WIRE_FIELDS,
        }
    }

    #[test]
    fn public_loader_requires_absolute_paths_before_any_file_admission() {
        let error = load_checked_fixed_packing_graph(
            Path::new("relative.json"),
            Path::new("relative.onnx"),
            ABC_SHA,
            ABC_SHA,
            declared_fixture_resources(),
            fixture_budget(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            DevicePackingAssetError::InvalidPath {
                part: PackingArtifactPart::Manifest,
                ..
            }
        ));
        assert_eq!(error.stage(), "arguments");
    }

    #[test]
    fn public_loader_retains_failed_part_stage_and_filesystem_cause() {
        use std::error::Error as _;
        let owner = OwnedFixtureDirectory::new();
        let missing = owner.0.join("missing");
        let error = load_checked_fixed_packing_graph(
            &missing,
            &missing,
            ABC_SHA,
            ABC_SHA,
            declared_fixture_resources(),
            fixture_budget(),
        )
        .unwrap_err();
        assert_eq!(error.stage(), "path_metadata");
        assert!(error.source().is_some());
        assert!(matches!(error, DevicePackingAssetError::Filesystem {
            part: PackingArtifactPart::Manifest, cause, ..
        } if cause.kind() == io::ErrorKind::NotFound));
    }

    #[test]
    fn invalid_scalar_resource_and_inspection_declarations_fail_before_missing_file_io() {
        let missing = std::env::temp_dir().join("rz-packing-unread-preflight.json");
        for axis in ["session", "metadata", "overflow", "owner", "host", "device"] {
            let mut declaration = declared_fixture_resources();
            match axis {
                "session" => declaration.packing_session_bytes = None,
                "metadata" => declaration.additional_owner_metadata_host_bytes = Some(0),
                "overflow" => declaration.packing_session_bytes = Some(u64::MAX),
                "owner" => declaration.max_owned_artifact_host_bytes = 0,
                "host" => declaration.max_declared_host_bytes = 0,
                "device" => declaration.max_declared_device_bytes = 0,
                _ => unreachable!(),
            }
            let error = load_checked_fixed_packing_graph(
                &missing,
                &missing,
                ABC_SHA,
                ABC_SHA,
                declaration,
                fixture_budget(),
            )
            .unwrap_err();
            assert!(
                matches!(error, DevicePackingAssetError::ByteAdmission(_)),
                "{axis}: {error:?}"
            );
        }
        let invalid_budget = PackingGraphInspectionBudget {
            max_inspection_host_bytes: 0,
            max_wire_fields: 1,
        };
        let error = load_checked_fixed_packing_graph(
            &missing,
            &missing,
            ABC_SHA,
            ABC_SHA,
            declared_fixture_resources(),
            invalid_budget,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            DevicePackingAssetError::GraphBody(PackingGraphBodyError::InvalidBudget)
        ));
    }
}
