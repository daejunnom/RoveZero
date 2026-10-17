//! Original ExternalR descriptors from the native startup's exact input lock.
//!
//! Admission only retains and verifies sources. It starts no process and grants
//! no root, dispatch, completion, target or Rules authority. The actual external
//! owner must retain the returned custody through physical worker cleanup.

use super::{PinnedBlob, RoventPinnedNativeSources, open_blob, read_blob};
use crate::rovent_bootstrap::{RoventStartupControl, fail};
use crate::rovent_external_launch::{PreparedRoventExternalLaunch, RoventExternalSlot};
use rz_contracts::Digest;
use rz_eval::error::{BackendError, FailureKind as K};
use rz_experiments::{ArtifactRef, RoventExternalPurpose, RoventResearcher};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs::File, path::Path, sync::Arc};

const PROFILE_MAX: usize = 256 << 10;
const BINARY_MAX: u64 = 256 << 20;
const ASSET_MAX: u64 = 1 << 30;
const ASSET_COUNT_MAX: usize = 32;

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventExternalRAssetSelection {
    pub purpose: String,
    pub artifact: ArtifactRef,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventExternalRSourceSelection {
    /// A closed `inputs/` directory in the same original private snapshot.
    pub working_directory_snapshot_path: String,
    /// Exact profile order; every separate asset retains its original artifact.
    pub assets: Vec<RoventExternalRAssetSelection>,
}

/// Only `open` creates this bundle from the actual External researcher slot.
/// P/C, teacher and arena declarations cannot be relabeled as ExternalR.
pub struct RoventPinnedExternalRSource {
    original: Arc<RoventPinnedNativeSources>,
    purpose: RoventExternalPurpose,
    prepared: PreparedRoventExternalLaunch,
    original_profile_bytes: Vec<u8>,
    profile: PinnedBlob,
    binary: PinnedBlob,
    working_directory: File,
    assets: Vec<(String, PinnedBlob)>,
}

/// Opaque original source custody for the physical external worker's owner.
/// Keeping it alive is not evidence that the worker loaded any declared asset.
pub struct RoventExternalRSourceCustody {
    original: Arc<RoventPinnedNativeSources>,
    purpose: RoventExternalPurpose,
    _profile: PinnedBlob,
    _assets: Vec<(String, PinnedBlob)>,
}

impl RoventPinnedExternalRSource {
    /// Reuse the native owner's retained root capability and bounded artifact
    /// reader. Only this admission opens the selected snapshot paths; later
    /// consumption moves the original descriptors without reopening a path.
    pub fn open(
        original: &Arc<RoventPinnedNativeSources>,
        selection: &RoventExternalRSourceSelection,
        control: &RoventStartupControl,
    ) -> Result<Self, BackendError> {
        control.check()?;
        original.validate_lock()?;
        control.check()?;
        let pin = match &original.lock.manifest.engine.researcher {
            Some(RoventResearcher::External { profile }) => profile.as_ref(),
            _ => return Err(invalid("original lock has no ExternalR source slot")),
        };
        if pin.purpose != RoventExternalPurpose::ResearchChecker
            || !original.private_root_path.is_absolute()
            || !rz_experiments::pals_private_snapshot_path_v5(&pin.profile.path)
            || !rz_experiments::pals_private_snapshot_path_v5(&pin.binary.path)
            || !rz_experiments::pals_private_snapshot_path_v5(
                &selection.working_directory_snapshot_path,
            )
            || pin.profile.path == pin.binary.path
            || selection.assets.len() > ASSET_COUNT_MAX
        {
            return Err(invalid(
                "ExternalR original purpose or closed source mapping differs",
            ));
        }

        let root = &original._root;
        let profile = open_blob(root, pin.profile.clone(), PROFILE_MAX as u64)?;
        let original_profile_bytes = read_blob(&profile, PROFILE_MAX)?;
        control.check()?;
        let prepared = PreparedRoventExternalLaunch::from_lock(
            &original.lock,
            RoventExternalSlot::Researcher,
            &original_profile_bytes,
        )
        .map_err(|_| invalid("original ExternalR profile/capability declaration differs"))?;
        let declaration = prepared.profile();
        if Path::new(&declaration.program) != original.private_root_path.join(&pin.binary.path)
            || Path::new(&declaration.working_directory)
                != original
                    .private_root_path
                    .join(&selection.working_directory_snapshot_path)
            || selection.assets.len() != declaration.identity.assets.len()
        {
            return Err(invalid(
                "ExternalR program/cwd or separate asset coverage differs",
            ));
        }

        // Validate every supplied artifact before opening any separate asset.
        // A profile/binary alias or duplicate path cannot supply another slot.
        let mut paths = BTreeSet::from([pin.profile.path.as_str(), pin.binary.path.as_str()]);
        for (selected, expected) in selection.assets.iter().zip(&declaration.identity.assets) {
            if selected.purpose != expected.purpose
                || selected.artifact.sha256 != expected.sha256
                || !rz_experiments::pals_private_snapshot_path_v5(&selected.artifact.path)
                || !original
                    .lock
                    .manifest
                    .provenance
                    .contains(&selected.artifact)
                || !paths.insert(selected.artifact.path.as_str())
            {
                return Err(invalid(
                    "ExternalR asset purpose/hash/original provenance differs",
                ));
            }
        }

        // open_blob verifies the executable bytes against the original pin and
        // retains that same FD for the worker's independent startup hash check.
        let binary = open_blob(root, pin.binary.clone(), BINARY_MAX)?;
        control.check()?;
        let working_directory = root
            .open_dir(&selection.working_directory_snapshot_path)
            .map_err(|_| invalid("original ExternalR cwd descriptor is unavailable"))?
            .into_std_file();
        if !working_directory
            .metadata()
            .is_ok_and(|metadata| metadata.is_dir())
        {
            return Err(invalid(
                "original ExternalR cwd descriptor is not a directory",
            ));
        }
        control.check()?;
        let mut assets = Vec::new();
        assets
            .try_reserve_exact(selection.assets.len())
            .map_err(|_| {
                fail(
                    K::ResourceExhausted,
                    "ExternalR source roster allocation failed",
                )
            })?;
        for selected in &selection.assets {
            assets.push((
                selected.purpose.clone(),
                open_blob(root, selected.artifact.clone(), ASSET_MAX)?,
            ));
            control.check()?;
        }
        control.check()?;
        Ok(Self {
            original: Arc::clone(original),
            purpose: pin.purpose,
            prepared,
            original_profile_bytes,
            profile,
            binary,
            working_directory,
            assets,
        })
    }

    pub fn input_lock_digest(&self) -> Digest {
        self.original.lock_digest()
    }

    /// The actual original manifest pin's purpose, never a caller-set label.
    pub fn source_purpose(&self) -> RoventExternalPurpose {
        self.purpose
    }

    /// Move the original files to the actual worker owner without new IO,
    /// descriptor duplication, profile mutation or any authority manufacture.
    pub fn into_parts(
        self,
    ) -> (
        PreparedRoventExternalLaunch,
        Vec<u8>,
        File,
        File,
        RoventExternalRSourceCustody,
    ) {
        (
            self.prepared,
            self.original_profile_bytes,
            self.binary.file,
            self.working_directory,
            RoventExternalRSourceCustody {
                original: self.original,
                purpose: self.purpose,
                _profile: self.profile,
                _assets: self.assets,
            },
        )
    }
}

impl RoventExternalRSourceCustody {
    pub fn input_lock_digest(&self) -> Digest {
        self.original.lock_digest()
    }

    pub fn source_purpose(&self) -> RoventExternalPurpose {
        self.purpose
    }
}

fn invalid(detail: &'static str) -> BackendError {
    fail(K::IdentityMismatch, detail)
}
