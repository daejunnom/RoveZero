//! Focused source/process-boundary fixtures. No model or engine is created.
use super::*;
use std::{
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

fn control() -> RoventStartupControl {
    let until = Instant::now() + Duration::from_secs(30);
    RoventStartupControl {
        until,
        cleanup_until: until + Duration::from_secs(1),
        cancellation: Arc::new(AtomicBool::new(false)),
    }
}

#[test]
fn exact_embedded_components_accept_and_unknown_or_changed_bytes_reject() {
    let actual = compiled_process_source_components().unwrap();
    verify_component_declaration(&actual, None, false).unwrap();
    assert_eq!(
        actual.get(PROCESS_SOURCE_COMPONENT),
        Some(&hex(compiled_process_boundary_source_digest()))
    );
    assert_eq!(
        actual.get("rovent_bootstrap"),
        Some(&hex(compiled_bootstrap_source_digest()))
    );
    let mut unknown = actual.clone();
    unknown.insert("unregistered-final-session".into(), "1".repeat(64));
    assert_eq!(
        verify_component_declaration(&unknown, None, false)
            .unwrap_err()
            .kind,
        ProcessBoundaryFailure::SourceComponent
    );
    let mut wrong = actual.clone();
    wrong.insert("rules".into(), "0".repeat(64));
    assert!(verify_component_declaration(&wrong, None, false).is_err());
    let mut missing = actual;
    missing.remove(PROCESS_SOURCE_COMPONENT);
    assert!(verify_component_declaration(&missing, None, false).is_err());
}

#[test]
fn own_researcher_sources_are_required_without_requiring_r_for_pc_only() {
    let mut declared = compiled_process_source_components().unwrap();
    for key in ["cpu_search", "cpu_worker", "cpu_credit"] {
        declared.remove(key);
    }
    verify_component_declaration(&declared, None, false).unwrap();
    let own = RoventResearcher::Own {
        implementation_sha256: "1".repeat(64),
        search_configuration_sha256: "2".repeat(64),
    };
    assert!(verify_component_declaration(&declared, Some(&own), false).is_err());
    // PC-only NNUEC readout registers its implementation, without R execution.
    assert!(verify_component_declaration(&declared, None, true).is_err());
    let actual = compiled_process_source_components().unwrap();
    declared.insert("cpu_search".into(), actual["cpu_search"].clone());
    verify_component_declaration(&declared, None, true).unwrap();
}

#[test]
fn commit_registration_is_not_replaced_by_a_fabricated_valid_sha() {
    let registered = env!("RZ_PALS_COMPILED_SOURCE_COMMIT");
    if registered.len() == 40 {
        verify_commit_registration(registered).unwrap();
    } else {
        assert_eq!(
            verify_commit_registration(registered).unwrap_err().kind,
            ProcessBoundaryFailure::SourceCommitRegistration
        );
    }
    let other = if registered == "0".repeat(40) {
        "1".repeat(40)
    } else {
        "0".repeat(40)
    };
    assert_eq!(
        verify_commit_registration(&other).unwrap_err().kind,
        ProcessBoundaryFailure::SourceCommitRegistration
    );
    assert!(verify_commit_registration("unregistered").is_err());
}

#[test]
fn original_control_is_not_renewed_and_cancel_precedes_expiry() {
    let active = control();
    check_control(&active).unwrap();
    let mut expired = active.clone();
    expired.until = Instant::now() - Duration::from_millis(1);
    assert_eq!(
        check_control(&expired).unwrap_err().kind,
        ProcessBoundaryFailure::Expired
    );
    active.cancellation.store(true, Ordering::Release);
    assert!(Arc::ptr_eq(&active.cancellation, &expired.cancellation));
    assert_eq!(
        check_control(&expired).unwrap_err().kind,
        ProcessBoundaryFailure::Cancelled
    );
}

#[cfg(not(target_os = "linux"))]
#[test]
fn unsupported_platform_never_promotes_current_exe_file_to_mapped_proof() {
    assert_eq!(
        require_mapped_executable_profile().unwrap_err().kind,
        ProcessBoundaryFailure::UnsupportedMappedExecutable
    );
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::{
        fs,
        io::{Seek, SeekFrom},
        path::PathBuf,
        sync::atomic::AtomicU64,
    };
    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "rz-rovent-process-fixture-{}-{time}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&root).unwrap();
            Self { root }
        }
        fn capability(&self) -> Dir {
            Dir::open_ambient_dir(&self.root, cap_std::ambient_authority()).unwrap()
        }
        fn cwd_file(&self) -> File {
            File::open(&self.root).unwrap()
        }
        fn binary(&self, bytes: &[u8]) -> (ArtifactRef, File) {
            let path = self.root.join("binary");
            fs::write(&path, bytes).unwrap();
            (
                ArtifactRef {
                    path: "inputs/binary".into(),
                    bytes: bytes.len() as u64,
                    sha256: hex(Sha256::digest(bytes).into()),
                    source: "https://example.org/unexecuted-process-boundary-fixture".into(),
                    license: "fixture-only".into(),
                },
                File::open(path).unwrap(),
            )
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            if self
                .root
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("rz-rovent-process-fixture-"))
                && fs::symlink_metadata(&self.root)
                    .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
            {
                // Fixed owned fixture names only; no recursive/shared cleanup.
                for name in ["binary", "original", ".git"] {
                    let _ = fs::remove_file(self.root.join(name));
                }
                let _ = fs::remove_dir(&self.root);
            }
        }
    }

    #[test]
    fn original_binary_hash_preserves_cursor_and_rejects_changed_bytes_and_bounds() {
        let fixture = Fixture::new();
        let bytes = b"unexecuted original binary fixture";
        let (artifact, mut file) = fixture.binary(bytes);
        file.seek(SeekFrom::Start(7)).unwrap();
        verify_binary_descriptor(&artifact, &file, &control()).unwrap();
        assert_eq!(file.stream_position().unwrap(), 7);
        let mut too_big = artifact.clone();
        too_big.bytes = MAX_EXECUTABLE_BYTES + 1;
        assert_eq!(
            verify_binary_descriptor(&too_big, &file, &control())
                .unwrap_err()
                .kind,
            ProcessBoundaryFailure::BinaryDescriptor
        );
        fs::write(fixture.root.join("binary"), vec![b'x'; bytes.len()]).unwrap();
        assert_eq!(
            verify_binary_descriptor(&artifact, &file, &control())
                .unwrap_err()
                .kind,
            ProcessBoundaryFailure::BinaryDescriptor
        );
    }

    #[test]
    fn binary_cancellation_keeps_original_handle_and_cursor() {
        let fixture = Fixture::new();
        let (artifact, mut file) = fixture.binary(b"unexecuted canceled binary fixture");
        file.seek(SeekFrom::Start(5)).unwrap();
        let control = control();
        control.cancellation.store(true, Ordering::Release);
        assert_eq!(
            verify_binary_descriptor(&artifact, &file, &control)
                .unwrap_err()
                .kind,
            ProcessBoundaryFailure::Cancelled
        );
        assert_eq!(file.stream_position().unwrap(), 5);
    }

    #[test]
    fn mapped_binary_comparison_requires_actual_linux_origin_and_exact_bytes() {
        let fixture = Fixture::new();
        let (artifact, _file) = fixture.binary(b"unexecuted identity comparison fixture");
        let mut observed = ExecutableIdentityV1 {
            sha256: artifact.sha256.clone(),
            bytes: artifact.bytes,
            identity_source: ExecutableIdentitySourceV1::LinuxProcSelfExe,
        };
        verify_mapped_executable(&artifact, &observed).unwrap();
        observed.identity_source = ExecutableIdentitySourceV1::CurrentExecutableFile;
        assert_eq!(
            verify_mapped_executable(&artifact, &observed)
                .unwrap_err()
                .kind,
            ProcessBoundaryFailure::MappedExecutable
        );
        observed.identity_source = ExecutableIdentitySourceV1::LinuxProcSelfExe;
        observed.bytes += 1;
        assert!(verify_mapped_executable(&artifact, &observed).is_err());
        observed.bytes = artifact.bytes;
        observed.sha256 = "0".repeat(64);
        assert!(verify_mapped_executable(&artifact, &observed).is_err());
    }

    #[test]
    fn private_directory_requires_exact_path_and_original_cwd_and_root_capabilities() {
        let first = Fixture::new();
        let second = Fixture::new();
        let actual = first.cwd_file();
        let root = first.capability();
        verify_private_directory_at(&first.root, &first.root, &actual, &root).unwrap();
        assert!(verify_private_directory_at(&second.root, &first.root, &actual, &root).is_err());
        assert!(
            verify_private_directory_at(&first.root, &first.root, &second.cwd_file(), &root)
                .is_err()
        );
        assert!(
            verify_private_directory_at(&first.root, &first.root, &actual, &second.capability())
                .is_err()
        );
        assert!(
            verify_private_directory_at(
                Path::new("relative"),
                Path::new("relative"),
                &actual,
                &root
            )
            .is_err()
        );
    }

    #[test]
    fn source_checkout_marker_is_rejected_without_reading_repository_contents() {
        let fixture = Fixture::new();
        fs::write(
            fixture.root.join(".git"),
            b"marker-only; contents are never read",
        )
        .unwrap();
        assert_eq!(
            verify_private_directory_at(
                &fixture.root,
                &fixture.root,
                &fixture.cwd_file(),
                &fixture.capability()
            )
            .unwrap_err()
            .kind,
            ProcessBoundaryFailure::SourceCheckout
        );
    }

    #[test]
    fn replaced_named_root_does_not_rebind_the_original_cwd_descriptor() {
        let fixture = Fixture::new();
        let root = fixture.capability();
        let actual = fixture.cwd_file();
        let original = fixture.root.with_extension("original");
        fs::rename(&fixture.root, &original).unwrap();
        fs::create_dir(&fixture.root).unwrap();
        let result = verify_private_directory_at(&fixture.root, &fixture.root, &actual, &root);
        // Restore the owned names before assertion, including failure cases.
        fs::remove_dir(&fixture.root).unwrap();
        fs::rename(&original, &fixture.root).unwrap();
        assert_eq!(
            result.unwrap_err().kind,
            ProcessBoundaryFailure::WorkingDirectory
        );
    }
}
