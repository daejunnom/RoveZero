//! Original teacher source descriptors for explicit Collection preloading.
//!
//! This permission only admits native P/C source loading. It starts no teacher
//! process and grants no target, dispatch or completion authority. Each actual
//! root's collection owner consumes the retained files after Frontier admission.

use super::{PinnedBlob, open_blob, read_blob};
use crate::rovent_bootstrap::{RoventStartupControl, fail};
use crate::rovent_external_launch::{PreparedRoventExternalLaunch, RoventExternalSlot};
use cap_std::fs::Dir;
use rz_contracts::Digest;
use rz_eval::error::{BackendError, FailureKind as K};
use rz_experiments::{ArtifactRef, RoventInputLock, RoventPurpose};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs::File, path::Path, sync::Arc};

const PROFILE_MAX: usize = 256 << 10;
const BINARY_MAX: u64 = 256 << 20;
const ASSET_MAX: u64 = 1 << 30;

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventTeacherAssetSelection {
    pub purpose: String,
    pub artifact: ArtifactRef,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventTeacherSourceSelection {
    pub index: usize,
    /// A directory inside the same closed private input snapshot.
    pub working_directory_snapshot_path: String,
    pub assets: Vec<RoventTeacherAssetSelection>,
}

struct PinnedTeacherSource {
    profile: PinnedBlob,
    binary: PinnedBlob,
    cwd: File,
    assets: Vec<(String, PinnedBlob)>,
}

/// Only `open` can create this source permission from the full original lock
/// and its exact retained profile, executable, cwd and separate-asset files.
pub struct RoventCollectionPreloadSources {
    lock: Arc<RoventInputLock>,
    _original_root: Dir,
    teachers: Vec<PinnedTeacherSource>,
}

/// Original descriptors transferred to an upper collection session owner.
/// It still needs the actual per-go root/Frontier and real teacher startup.
pub struct RoventCollectionTeacherSources {
    pub index: usize,
    pub prepared: PreparedRoventExternalLaunch,
    pub original_profile_bytes: Vec<u8>,
    pub executable: File,
    pub working_directory: File,
    pub assets: Vec<(String, ArtifactRef, File)>,
    original: Arc<RoventCollectionPreloadSources>,
}
/// Retain this after the worker consumes its executable/profile descriptors.
/// It is source custody only and cannot submit or validate a teacher result.
pub struct RoventCollectionTeacherSourceCustody {
    original: Arc<RoventCollectionPreloadSources>,
    assets: Vec<(String, ArtifactRef, File)>,
}
impl RoventCollectionTeacherSourceCustody {
    pub fn original_input_lock_digest(&self) -> Digest {
        self.original.input_lock_digest()
    }
    pub fn separate_assets(&self) -> &[(String, ArtifactRef, File)] {
        &self.assets
    }
}
impl RoventCollectionTeacherSources {
    pub fn original_input_lock_digest(&self) -> Digest {
        self.original.input_lock_digest()
    }
    pub fn into_parts(
        self,
    ) -> (
        usize,
        PreparedRoventExternalLaunch,
        Vec<u8>,
        File,
        File,
        RoventCollectionTeacherSourceCustody,
    ) {
        (
            self.index,
            self.prepared,
            self.original_profile_bytes,
            self.executable,
            self.working_directory,
            RoventCollectionTeacherSourceCustody {
                original: self.original,
                assets: self.assets,
            },
        )
    }
}

impl RoventCollectionPreloadSources {
    pub fn open(
        lock: Arc<RoventInputLock>,
        root: Dir,
        private_root_path: &Path,
        selections: Vec<RoventTeacherSourceSelection>,
        control: &RoventStartupControl,
    ) -> Result<Self, BackendError> {
        control.check()?;
        lock.validate()
            .map_err(|_| invalid("original Collection lock"))?;
        if lock.manifest.purpose != RoventPurpose::Collection
            || lock.manifest.engine.teachers.is_empty()
            || lock.manifest.engine.teachers.len() > 16
            || selections.len() != lock.manifest.engine.teachers.len()
            || !private_root_path.is_absolute()
        {
            return Err(invalid(
                "explicit Collection teacher source cardinality/purpose",
            ));
        }
        let mut teachers = Vec::new();
        teachers.try_reserve_exact(selections.len()).map_err(|_| {
            fail(
                K::ResourceExhausted,
                "teacher source roster allocation failed",
            )
        })?;
        for (index, selection) in selections.into_iter().enumerate() {
            let pin = &lock.manifest.engine.teachers[index];
            if selection.index != index
                || !rz_experiments::pals_private_snapshot_path_v5(&pin.profile.path)
                || !rz_experiments::pals_private_snapshot_path_v5(&pin.binary.path)
                || !rz_experiments::pals_private_snapshot_path_v5(
                    &selection.working_directory_snapshot_path,
                )
                || selection.assets.len() > 32
            {
                return Err(invalid(
                    "teacher sources must cover original slots in exact order",
                ));
            }
            let profile = open_blob(&root, pin.profile.clone(), PROFILE_MAX as u64)?;
            let bytes = read_blob(&profile, PROFILE_MAX)?;
            control.check()?;
            let launch = PreparedRoventExternalLaunch::from_lock(
                &lock,
                RoventExternalSlot::Teacher { index },
                &bytes,
            )
            .map_err(|_| invalid("original teacher profile/capability declaration"))?;
            let declaration = launch.profile();
            if Path::new(&declaration.program) != private_root_path.join(&pin.binary.path)
                || Path::new(&declaration.working_directory)
                    != private_root_path.join(&selection.working_directory_snapshot_path)
            {
                return Err(invalid(
                    "teacher program/cwd differs from the original snapshot mapping",
                ));
            }
            let binary = open_blob(&root, pin.binary.clone(), BINARY_MAX)?;
            control.check()?;
            let cwd = root
                .open_dir(&selection.working_directory_snapshot_path)
                .map_err(|_| invalid("original teacher working-directory descriptor"))?
                .into_std_file();
            if !cwd.metadata().is_ok_and(|metadata| metadata.is_dir()) {
                return Err(invalid(
                    "teacher working-directory descriptor is not a directory",
                ));
            }
            let expected = &declaration.identity.assets;
            if selection.assets.len() != expected.len() {
                return Err(invalid("teacher separate asset roster coverage differs"));
            }
            let mut seen = BTreeSet::new();
            let mut assets = Vec::new();
            for (selected, expected) in selection.assets.into_iter().zip(expected) {
                if selected.purpose != expected.purpose
                    || selected.artifact.sha256 != expected.sha256
                    || !rz_experiments::pals_private_snapshot_path_v5(&selected.artifact.path)
                    || !lock.manifest.provenance.contains(&selected.artifact)
                    || !seen.insert((selected.purpose.clone(), selected.artifact.path.clone()))
                {
                    return Err(invalid(
                        "teacher separate asset purpose/hash/source roster differs",
                    ));
                }
                assets.push((
                    selected.purpose,
                    open_blob(&root, selected.artifact, ASSET_MAX)?,
                ));
                control.check()?;
            }
            teachers.push(PinnedTeacherSource {
                profile,
                binary,
                cwd,
                assets,
            });
        }
        control.check()?;
        Ok(Self {
            lock,
            _original_root: root,
            teachers,
        })
    }

    pub fn input_lock_digest(&self) -> Digest {
        // The original lock was fully validated before this source owner arose.
        let bytes = rz_eval::asset::parse_sha256(&self.lock.canonical_sha256)
            .expect("validated original Collection lock digest");
        Digest(bytes)
    }
    pub fn matches_input_lock(&self, lock: &RoventInputLock) -> bool {
        self.lock.as_ref() == lock
    }
    pub fn teacher_count(&self) -> usize {
        self.teachers.len()
    }

    /// Duplicate only the original retained descriptors for a root's unique
    /// teacher owner. No pathname is reopened and no startup is represented.
    pub fn teacher_sources(
        self: &Arc<Self>,
        index: usize,
        control: &RoventStartupControl,
    ) -> Result<RoventCollectionTeacherSources, BackendError> {
        control.check()?;
        let source = self
            .teachers
            .get(index)
            .ok_or_else(|| invalid("teacher source slot"))?;
        let original_profile_bytes = read_blob(&source.profile, PROFILE_MAX)?;
        let prepared = PreparedRoventExternalLaunch::from_lock(
            &self.lock,
            RoventExternalSlot::Teacher { index },
            &original_profile_bytes,
        )
        .map_err(|_| invalid("retained original teacher profile changed"))?;
        let duplicate = |file: &File| {
            file.try_clone()
                .map_err(|_| fail(K::Io, "original teacher descriptor duplication failed"))
        };
        let executable = duplicate(&source.binary.file)?;
        let working_directory = duplicate(&source.cwd)?;
        let mut assets = Vec::new();
        for (purpose, blob) in &source.assets {
            assets.push((
                purpose.clone(),
                blob.declaration.clone(),
                duplicate(&blob.file)?,
            ));
        }
        control.check()?;
        Ok(RoventCollectionTeacherSources {
            index,
            prepared,
            original_profile_bytes,
            executable,
            working_directory,
            assets,
            original: Arc::clone(self),
        })
    }
}
fn invalid(detail: &'static str) -> BackendError {
    fail(K::IdentityMismatch, detail)
}
