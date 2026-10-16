//! Exact Rovent asset selection over retained, verified descriptor owners.
//! No legacy manifest or V5 success record is synthesized for this model.

use super::{
    RoventStartupControl, RoventStartupTopology, admitted_topology,
    compiled_bootstrap_source_digest, fail, hex,
};
use crate::pals_native::virtual_encoding::validate_pals_virtual_rules_descriptor_bytes;
use cap_std::fs::Dir;
use rz_contracts::Digest;
use rz_eval::asset::parse_sha256;
use rz_eval::error::{BackendError, FailureKind as K};
use rz_eval::pals_onnx::virtual_review::{
    VirtualReviewArtifactKind, VirtualReviewExecutionMode, VirtualReviewOwnedSources,
    validate_virtual_review_export_manifest_kind,
};
use rz_eval::runtime_pin::{RuntimeBundleFileRole, RuntimeLibraryPin};
use rz_experiments::{ArtifactRef, RoventInputLock};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeSet,
    fs::File,
    path::{Component, Path},
    sync::Arc,
};

const MAX_GRAPH_BYTES: u64 = 1 << 30;
const MAX_WEIGHTS_BYTES: u64 = 8 << 30;
const MAX_PROVENANCE_BYTES: u64 = 16 << 20;

#[derive(Clone, Debug)]
pub struct RoventNativeGraphAsset {
    pub role: String,
    /// The original export basename, distinct from the private snapshot path.
    pub registered_file: String,
    pub artifact: ArtifactRef,
}
#[derive(Clone, Debug)]
pub struct RoventNativeAssetSelection {
    pub export_manifest: ArtifactRef,
    pub rules_descriptor: ArtifactRef,
    pub graphs: Vec<RoventNativeGraphAsset>,
    pub cuda_control_inventory: Option<ArtifactRef>,
}

struct PinnedBlob {
    declaration: ArtifactRef,
    file: File,
}
struct PinnedGraph {
    role: String,
    registered_file: String,
    blob: PinnedBlob,
}

/// Descriptors remain bound to the exact new lock. Path replacement cannot
/// redirect a later read. Participating owners must still prevent mutation of
/// these private files; metadata checks are not a hostile same-UID sandbox.
pub struct RoventPinnedNativeSources {
    lock: Arc<RoventInputLock>,
    topology: RoventStartupTopology,
    _root: Dir,
    export: PinnedBlob,
    rules: PinnedBlob,
    graphs: Vec<PinnedGraph>,
    control: Option<PinnedBlob>,
    _weights: PinnedBlob,
    _provenance: PinnedBlob,
}
impl RoventPinnedNativeSources {
    /// Pure byte/descriptor admission. No ORT runtime, model or R worker starts.
    pub fn open(
        lock: Arc<RoventInputLock>,
        root: Dir,
        selection: RoventNativeAssetSelection,
        control: &RoventStartupControl,
    ) -> Result<Self, BackendError> {
        control.check()?;
        let topology = admitted_topology(&lock)?;
        if selection.graphs.len() != 2 {
            return Err(fail(
                K::UnsupportedModel,
                "Fresh V-free native startup needs exactly two original graphs",
            ));
        }
        let compiled = rz_experiments::compiled_source_components_v5().map_err(|_| {
            fail(
                K::IdentityMismatch,
                "compiled native source inventory unavailable",
            )
        })?;
        let model = &lock.manifest.engine.pc.proposer;
        if lock
            .manifest
            .engine
            .source_components
            .get("rovent_bootstrap")
            != Some(&hex(compiled_bootstrap_source_digest()))
            || compiled.get("model_native") != Some(&model.implementation_sha256)
            || lock.manifest.engine.source_components.get("model_native")
                != Some(&model.implementation_sha256)
        {
            return Err(fail(
                K::IdentityMismatch,
                "native and bootstrap source pins differ from compiled bytes",
            ));
        }
        let mut selected_paths = BTreeSet::new();
        for artifact in std::iter::once(&selection.export_manifest)
            .chain(std::iter::once(&selection.rules_descriptor))
            .chain(selection.graphs.iter().map(|graph| &graph.artifact))
            .chain(selection.cuda_control_inventory.iter())
        {
            if !lock.manifest.provenance.contains(artifact)
                || !selected_paths.insert(artifact.path.as_str())
            {
                return Err(fail(
                    K::IdentityMismatch,
                    "selected native asset is absent or duplicated in the Rovent lock",
                ));
            }
        }
        let export = open_blob(&root, selection.export_manifest, 256 << 10)?;
        control.check()?;
        let rules = open_blob(&root, selection.rules_descriptor, 64 << 10)?;
        control.check()?;
        let mut graphs = Vec::new();
        let mut roles = BTreeSet::new();
        for graph in selection.graphs {
            if !matches!(graph.role.as_str(), "public" | "shared_pc")
                || !roles.insert(graph.role.clone())
                || !basename(&graph.registered_file)
            {
                return Err(fail(
                    K::IdentityMismatch,
                    "native graph role/basename is unsupported or duplicated",
                ));
            }
            graphs.push(PinnedGraph {
                role: graph.role,
                registered_file: graph.registered_file,
                blob: open_blob(&root, graph.artifact, MAX_GRAPH_BYTES)?,
            });
            control.check()?;
        }
        let control_blob = selection
            .cuda_control_inventory
            .map(|artifact| open_blob(&root, artifact, 4 << 20))
            .transpose()?;
        let weights = open_blob(&root, model.weights.clone(), MAX_WEIGHTS_BYTES)?;
        control.check()?;
        let provenance = open_blob(&root, model.provenance.clone(), MAX_PROVENANCE_BYTES)?;
        control.check()?;
        let owner = Self {
            lock,
            topology,
            _root: root,
            export,
            rules,
            graphs,
            control: control_blob,
            _weights: weights,
            _provenance: provenance,
        };
        owner.validate_lock()?;
        control.check()?;
        Ok(owner)
    }
    pub fn lock(&self) -> &RoventInputLock {
        &self.lock
    }
    pub fn topology(&self) -> RoventStartupTopology {
        self.topology
    }
    pub fn lock_digest(&self) -> Digest {
        // Construction has already validated this exact immutable lock string.
        Digest(parse_sha256(&self.lock.canonical_sha256).expect("admitted lock digest"))
    }
    pub fn export_digest_text(&self) -> String {
        self.export.declaration.sha256.clone()
    }
    pub fn weights_digest(&self) -> [u8; 32] {
        parse_sha256(&self._weights.declaration.sha256).expect("admitted weights digest")
    }
    pub fn cuda_inventory_digest(&self) -> Option<[u8; 32]> {
        self.control
            .as_ref()
            .map(|blob| parse_sha256(&blob.declaration.sha256).expect("admitted inventory digest"))
    }
    pub(crate) fn cuda_inventory_bytes(&self) -> Result<Option<(Vec<u8>, [u8; 32])>, BackendError> {
        self.control
            .as_ref()
            .map(|blob| {
                Ok((
                    read_blob(blob, 4 << 20)?,
                    parse_sha256(&blob.declaration.sha256)?,
                ))
            })
            .transpose()
    }
    pub fn validate_lock(&self) -> Result<(), BackendError> {
        if admitted_topology(&self.lock)? != self.topology {
            return Err(fail(
                K::IdentityMismatch,
                "native topology changed after pinning",
            ));
        }
        let raw = read_blob(&self.export, 256 << 10)?;
        let summary = validate_virtual_review_export_manifest_kind(
            &raw,
            VirtualReviewExecutionMode::Deployment,
            VirtualReviewArtifactKind::Fresh,
        )?;
        let rules_raw = read_blob(&self.rules, 64 << 10)?;
        let rules = validate_pals_virtual_rules_descriptor_bytes(&rules_raw)?;
        let model = &self.lock.manifest.engine.pc.proposer;
        if summary.checkpoint_sha256 != self.weights_digest()
            || summary.validator_present
            || summary.model_semantics != model.semantics
            || summary.encoding_schema != model.encoding
            || summary.graphs.len() != self.graphs.len()
            || summary.config != rules.config
            || summary.rules_input_semantic_sha256 != rules.semantic_sha256
            || summary.rules_encoder_source_sha256 != rules.encoder_source_sha256
            || summary.rules_profile_descriptor_sha256 != rules.descriptor_sha256()
        {
            return Err(fail(
                K::IdentityMismatch,
                "actual export, checkpoint and exact Rules descriptor disagree",
            ));
        }
        for expected in &summary.graphs {
            let matching = self
                .graphs
                .iter()
                .find(|graph| graph.role == expected.role)
                .ok_or_else(|| {
                    fail(
                        K::IdentityMismatch,
                        "native graph role missing from retained roster",
                    )
                })?;
            if matching.registered_file != expected.file
                || parse_sha256(&matching.blob.declaration.sha256)? != expected.sha256
            {
                return Err(fail(
                    K::IdentityMismatch,
                    "original graph identity differs from retained selected artifact",
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, pin: &RuntimeLibraryPin) -> Result<(), BackendError> {
        let artifacts = &self.lock.manifest.engine.pc.runtime_artifacts;
        match pin.bundle_files() {
            None => {
                if artifacts.len() != 1
                    || parse_sha256(&artifacts[0].sha256)? != pin.binary_digest()
                    || artifacts[0].bytes != pin.storage().bytes
                {
                    return Err(fail(
                        K::IdentityMismatch,
                        "actual CPU runtime pin differs from exact Rovent roster",
                    ));
                }
            }
            Some(files) => {
                if artifacts.len() != files.len() {
                    return Err(fail(
                        K::IdentityMismatch,
                        "CUDA runtime retained roster size differs",
                    ));
                }
                let mut matched = BTreeSet::new();
                for file in files {
                    let matches: Vec<_> = artifacts
                        .iter()
                        .enumerate()
                        .filter(|(_, artifact)| {
                            Path::new(&artifact.path)
                                .file_name()
                                .and_then(|name| name.to_str())
                                == Some(file.filename.as_str())
                                && artifact.sha256 == file.sha256
                                && artifact.bytes == file.bytes
                        })
                        .collect();
                    if matches.len() != 1
                        || !matched.insert(matches[0].0)
                        || (file.role == RuntimeBundleFileRole::Core
                            && parse_sha256(&file.sha256)? != pin.binary_digest())
                    {
                        return Err(fail(
                            K::IdentityMismatch,
                            "actual CUDA runtime file differs from original Rovent pin",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
impl VirtualReviewOwnedSources for RoventPinnedNativeSources {
    fn read_manifest(&self, maximum: usize) -> Result<Vec<u8>, BackendError> {
        read_blob(&self.export, maximum)
    }
    fn read_graph(
        &self,
        role: &str,
        registered_file: &str,
        expected: [u8; 32],
        maximum: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let graph = self
            .graphs
            .iter()
            .find(|graph| graph.role == role)
            .ok_or_else(|| {
                fail(
                    K::IdentityMismatch,
                    "native owner requested an unpinned graph role",
                )
            })?;
        if graph.registered_file != registered_file
            || parse_sha256(&graph.blob.declaration.sha256)? != expected
        {
            return Err(fail(
                K::IdentityMismatch,
                "native owner requested a different original graph",
            ));
        }
        read_blob(&graph.blob, maximum)
    }
}

fn open_blob(
    root: &Dir,
    declaration: ArtifactRef,
    maximum: u64,
) -> Result<PinnedBlob, BackendError> {
    let file = declaration.open_verified_at(root, maximum).map_err(|_| {
        fail(
            K::IdentityMismatch,
            "cannot admit exact retained Rovent artifact",
        )
    })?;
    Ok(PinnedBlob { declaration, file })
}
fn basename(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 256
        && matches!(
            Path::new(name).components().collect::<Vec<_>>().as_slice(),
            [Component::Normal(_)]
        )
}
fn read_blob(blob: &PinnedBlob, maximum: usize) -> Result<Vec<u8>, BackendError> {
    let before = blob
        .file
        .metadata()
        .map_err(|_| fail(K::Io, "retained artifact metadata unavailable"))?;
    let length = usize::try_from(blob.declaration.bytes)
        .map_err(|_| fail(K::ResourceExhausted, "retained artifact size overflow"))?;
    if !before.is_file()
        || length == 0
        || length > maximum
        || before.len() != blob.declaration.bytes
    {
        return Err(fail(
            K::ResourceExhausted,
            "retained artifact exceeds exact bounded read",
        ));
    }
    let mut raw = Vec::new();
    raw.try_reserve_exact(length)
        .map_err(|_| fail(K::ResourceExhausted, "retained artifact allocation failed"))?;
    raw.resize(length, 0);
    #[cfg(windows)]
    let reader =
        rz_native_loader::pinned_reader::independent_read_handle(&blob.file).map_err(|_| {
            fail(
                K::Io,
                "retained artifact independent read handle unavailable",
            )
        })?;
    #[cfg(windows)]
    let reader = &reader;
    #[cfg(not(windows))]
    let reader = &blob.file;
    let mut at = 0;
    while at < length {
        let read = positional_read(reader, &mut raw[at..], at as u64)
            .map_err(|_| fail(K::Io, "retained artifact positional read failed"))?;
        if read == 0 {
            return Err(fail(
                K::IdentityMismatch,
                "retained artifact ended before its declared size",
            ));
        }
        at += read;
    }
    let after = blob
        .file
        .metadata()
        .map_err(|_| fail(K::Io, "retained artifact metadata disappeared"))?;
    if before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
        || <[u8; 32]>::from(Sha256::digest(&raw)) != parse_sha256(&blob.declaration.sha256)?
    {
        return Err(fail(
            K::IdentityMismatch,
            "retained artifact changed during its exact read",
        ));
    }
    Ok(raw)
}
#[cfg(unix)]
fn positional_read(file: &File, out: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(out, offset)
}
#[cfg(windows)]
fn positional_read(file: &File, out: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(out, offset)
}
#[cfg(not(any(unix, windows)))]
fn positional_read(_: &File, _: &mut [u8], _: u64) -> std::io::Result<usize> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "positional asset read unavailable",
    ))
}
