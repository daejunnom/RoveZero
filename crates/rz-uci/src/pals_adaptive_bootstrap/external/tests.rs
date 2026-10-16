#[cfg(target_os = "linux")]
mod linux {
    use super::super::*;
    use crate::pals_adaptive_startup::PinnedStartupAssetV5;
    use cap_std::fs::Dir;
    use rz_experiments::{ArtifactRef, PalsStartupAssetV5};
    use sha2::{Digest as _, Sha256};
    use std::fs::File;
    use std::io::{Seek, SeekFrom, Write};
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1);
    struct OwnedTemp(PathBuf);
    impl OwnedTemp {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "rz-adaptive-external-admission-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&root).unwrap();
            std::fs::create_dir(root.join("inputs")).unwrap();
            Self(root)
        }
        fn capability(&self) -> Dir {
            Dir::open_ambient_dir(&self.0, cap_std::ambient_authority()).unwrap()
        }
        fn asset(&self, data: &[u8]) -> PinnedStartupAssetV5 {
            let name = "inputs/engine";
            std::fs::write(self.0.join(name), data).unwrap();
            PinnedStartupAssetV5 {
                declaration: PalsStartupAssetV5 {
                    artifact: ArtifactRef {
                        path: "external/engine".into(),
                        sha256: format!("{:x}", Sha256::digest(data)),
                        bytes: data.len() as u64,
                        source: "https://example.org/authored-external-admission".into(),
                        license: "MIT".into(),
                    },
                    snapshot_path: name.into(),
                    purpose: PalsStartupAssetPurposeV5::ExternalBinary,
                },
                file: File::open(self.0.join(name)).unwrap(),
            }
        }
    }
    impl Drop for OwnedTemp {
        fn drop(&mut self) {
            let (Ok(path), Ok(parent)) =
                (self.0.canonicalize(), std::env::temp_dir().canonicalize())
            else {
                return;
            };
            if path.parent() == Some(parent.as_path())
                && path.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with("rz-adaptive-external-admission-test-")
                })
            {
                let _ = std::fs::remove_dir_all(path);
            }
        }
    }

    fn verify_fixture_snapshot(
        root: &Dir,
        asset: &PinnedStartupAssetV5,
        maximum: u64,
        retain: bool,
    ) -> Result<Vec<u8>, CheckerError> {
        verify_held_snapshot(
            root,
            asset,
            maximum,
            retain,
            Instant::now() + std::time::Duration::from_secs(2),
            &AtomicBool::new(false),
        )
    }

    #[test]
    fn expired_or_canceled_original_startup_does_not_read_held_snapshot() {
        let root = OwnedTemp::new();
        let mut asset = root.asset(b"\x7fELFfixture-only-no-execution");
        asset.file.seek(SeekFrom::Start(7)).unwrap();
        let directory = root.capability();
        let cancel = AtomicBool::new(false);
        assert_eq!(
            verify_held_snapshot(&directory, &asset, 1024, true, Instant::now(), &cancel)
                .unwrap_err(),
            fail("original_startup_expired"),
        );
        cancel.store(true, Ordering::Release);
        assert_eq!(
            verify_held_snapshot(
                &directory,
                &asset,
                1024,
                true,
                Instant::now() + std::time::Duration::from_secs(2),
                &cancel,
            )
            .unwrap_err(),
            fail("original_startup_canceled"),
        );
        assert_eq!(asset.file.stream_position().unwrap(), 7);
    }

    #[test]
    fn retained_descriptor_hashing_preserves_cursor_and_detects_path_replacement() {
        let root = OwnedTemp::new();
        let mut asset = root.asset(b"\x7fELFfixture-only-no-execution");
        asset.file.seek(SeekFrom::Start(7)).unwrap();
        let directory = root.capability();
        let raw = verify_fixture_snapshot(&directory, &asset, 1024, true).unwrap();
        assert_eq!(raw, b"\x7fELFfixture-only-no-execution");
        assert_eq!(asset.file.stream_position().unwrap(), 7);
        assert!(
            verify_fixture_snapshot(&directory, &asset, 1024, false)
                .unwrap()
                .is_empty()
        );
        assert_eq!(asset.file.stream_position().unwrap(), 7);
        assert!(verify_fixture_snapshot(&directory, &asset, 4, false).is_err());
        std::fs::rename(root.0.join("inputs/engine"), root.0.join("inputs/held")).unwrap();
        std::fs::write(root.0.join("inputs/engine"), &raw).unwrap();
        assert_eq!(
            verify_fixture_snapshot(&directory, &asset, 1024, false).unwrap_err(),
            fail("snapshot_path_no_longer_names_held_descriptor")
        );
        assert_eq!(asset.file.stream_position().unwrap(), 7);
    }

    #[test]
    fn symlink_hash_change_and_nonexecutable_script_do_not_admit_external_snapshot() {
        let root = OwnedTemp::new();
        let mut asset = root.asset(b"\x7fELFfixture-only-no-execution");
        let directory = root.capability();
        asset.file.seek(SeekFrom::Start(6)).unwrap();
        assert!(verify_native_snapshot_header(&asset.file).is_err());
        std::fs::set_permissions(
            root.0.join("inputs/engine"),
            std::fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        assert!(verify_native_snapshot_header(&asset.file).is_ok());
        assert_eq!(asset.file.stream_position().unwrap(), 6);
        std::fs::rename(root.0.join("inputs/engine"), root.0.join("inputs/held")).unwrap();
        symlink("held", root.0.join("inputs/engine")).unwrap();
        assert!(verify_fixture_snapshot(&directory, &asset, 1024, false).is_err());
        std::fs::remove_file(root.0.join("inputs/engine")).unwrap();
        std::fs::rename(root.0.join("inputs/held"), root.0.join("inputs/engine")).unwrap();
        let mut writer = std::fs::OpenOptions::new()
            .write(true)
            .open(root.0.join("inputs/engine"))
            .unwrap();
        writer.write_all(b"#!sh").unwrap();
        writer.sync_all().unwrap();
        assert_eq!(
            verify_fixture_snapshot(&directory, &asset, 1024, false).unwrap_err(),
            fail("held_snapshot_changed_or_hash_mismatch")
        );
        assert!(verify_native_snapshot_header(&asset.file).is_err());
        assert_eq!(asset.file.stream_position().unwrap(), 6);
    }
}
