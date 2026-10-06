//! E-owned declarations for one actual CUDA neural integration pair.
//!
//! This schema is independent of the strength-experiment schema and of the
//! engine contract. A lock validates declarations, not A legality, immutable
//! file contents, provider initialization, neural execution or resource limits.
//! The executor must retain its verified input/output capabilities and compare
//! separately observed startup and termination receipts against this lock.

use crate::{
    ArtifactRef, HistoryCompleteness, InitialPosition, ManifestError,
    NATIVE_ENGINE_CONTRACT_REVISION, NativeArtifactBinding, NativeArtifactRole, NativeEngineRole,
    NativeHistoryFill, NativeMovetimeV1, NativePrecision, NativeTimeoutsV1, OpeningSpec,
    ToolIdentity, Violation, decode_json, digest,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CUDA_NATIVE_LAUNCH_SCHEMA_VERSION: u32 = 1;
pub const CUDA_NATIVE_LAUNCH_DOMAIN: &str = "rz-e-native-cuda-integration-pair-v1";
pub const CUDA_NATIVE_LAUNCH_CANONICALIZATION: &str = "rz-e-native-cuda-json-v1";
pub const MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES: usize = 64 * 1024;

const MAX_WALL_MS: u64 = 86_400_000;
const MAX_PLIES: u32 = 4095;
pub const CUDA_NATIVE_ARENA_BYTES: u64 = 1024 * 1024 * 1024;
pub const CUDA_BUNDLE_FILE_MAX_BYTES: u64 = 1024 * 1024 * 1024;
pub const CUDA_BUNDLE_MAX_BYTES: u64 = 4 * CUDA_BUNDLE_FILE_MAX_BYTES;
const CUDA_RECEIPT_RESERVE_PER_PROCESS_BYTES: u64 = 512 * 1024;
const CUDA_PLACEMENT_TRACE_RESERVE_PER_PROCESS_BYTES: u64 = 4 * 1024 * 1024;
const CUDA_PROCESS_START_RESERVATIONS: u64 = 4;

/// Metadata copied from C's closed first Linux runtime profile. These constants
/// neither inspect native bytes nor attest ELF dependency resolution.
pub const CUDA_BUNDLE_FILENAMES: [(&str, CudaBundleFileRoleV1); 19] = [
    ("libonnxruntime.so.1.22.0", CudaBundleFileRoleV1::Core),
    (
        "libonnxruntime_providers_shared.so",
        CudaBundleFileRoleV1::ProvidersShared,
    ),
    (
        "libonnxruntime_providers_cuda.so",
        CudaBundleFileRoleV1::ProvidersCuda,
    ),
    ("libcudart.so.12", CudaBundleFileRoleV1::NvidiaDependency),
    ("libcublas.so.12", CudaBundleFileRoleV1::NvidiaDependency),
    ("libcublasLt.so.12", CudaBundleFileRoleV1::NvidiaDependency),
    ("libcufft.so.11", CudaBundleFileRoleV1::NvidiaDependency),
    ("libcurand.so.10", CudaBundleFileRoleV1::NvidiaDependency),
    ("libnvrtc.so.12", CudaBundleFileRoleV1::NvidiaDependency),
    (
        "libnvrtc-builtins.so.12.8",
        CudaBundleFileRoleV1::NvidiaDependency,
    ),
    ("libnvJitLink.so.12", CudaBundleFileRoleV1::NvidiaDependency),
    ("libcudnn.so.9", CudaBundleFileRoleV1::NvidiaDependency),
    ("libcudnn_adv.so.9", CudaBundleFileRoleV1::NvidiaDependency),
    ("libcudnn_cnn.so.9", CudaBundleFileRoleV1::NvidiaDependency),
    ("libcudnn_ops.so.9", CudaBundleFileRoleV1::NvidiaDependency),
    (
        "libcudnn_engines_precompiled.so.9",
        CudaBundleFileRoleV1::NvidiaDependency,
    ),
    (
        "libcudnn_engines_runtime_compiled.so.9",
        CudaBundleFileRoleV1::NvidiaDependency,
    ),
    (
        "libcudnn_graph.so.9",
        CudaBundleFileRoleV1::NvidiaDependency,
    ),
    (
        "libcudnn_heuristic.so.9",
        CudaBundleFileRoleV1::NvidiaDependency,
    ),
];

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CudaBundleFileRoleV1 {
    Core,
    ProvidersShared,
    ProvidersCuda,
    NvidiaDependency,
}

impl CudaBundleFileRoleV1 {
    const fn codec_tag(self) -> u8 {
        match self {
            Self::Core => 1,
            Self::ProvidersShared => 2,
            Self::ProvidersCuda => 3,
            Self::NvidiaDependency => 4,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CudaBundleFileBindingV1 {
    pub role: CudaBundleFileRoleV1,
    pub filename: String,
    pub artifact: ArtifactRef,
}

/// The manifest artifact binds raw JSON bytes; canonical_sha256 separately binds
/// C's filename-sorted binary codec. Neither hash proves that a file was read.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CudaBundleBindingV1 {
    pub manifest: ArtifactRef,
    pub canonical_sha256: String,
    pub files: Vec<CudaBundleFileBindingV1>,
}

impl CudaBundleBindingV1 {
    pub fn file(
        &self,
        role: CudaBundleFileRoleV1,
        filename: &str,
    ) -> Result<&ArtifactRef, ManifestError> {
        let mut found = self
            .files
            .iter()
            .filter(|file| file.role == role && file.filename == filename);
        let file = found.next().ok_or_else(|| {
            violation(
                "cuda_bundle.files",
                "MissingCudaBundleFile",
                "required CUDA bundle role and filename are absent",
            )
        })?;
        require(
            found.next().is_none(),
            "cuda_bundle.files",
            "DuplicateCudaBundleFile",
            "CUDA bundle role and filename must be unique",
        )?;
        Ok(&file.artifact)
    }

    pub fn total_bytes(&self) -> Result<u64, ManifestError> {
        self.validate()?;
        self.metadata_bytes()
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        self.metadata_bytes()?;
        hash(&self.canonical_sha256, 64, "cuda_bundle.canonical_sha256")?;
        require(
            self.canonical_digest()? == self.canonical_sha256,
            "cuda_bundle.canonical_sha256",
            "CudaBundleDigestMismatch",
            "CUDA bundle canonical digest differs from C's binary metadata codec",
        )
    }

    fn metadata_bytes(&self) -> Result<u64, ManifestError> {
        self.manifest.validate()?;
        require(
            self.manifest.bytes <= MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES as u64,
            "cuda_bundle.manifest.bytes",
            "CudaBundleManifestBudget",
            "CUDA bundle raw manifest must fit 64 KiB",
        )?;
        require(
            self.files.len() == CUDA_BUNDLE_FILENAMES.len(),
            "cuda_bundle.files",
            "CudaBundleFileCount",
            "CUDA bundle requires exactly nineteen first-profile libraries",
        )?;
        let parent = self
            .manifest
            .path
            .rsplit_once('/')
            .map_or("", |(parent, _)| parent);
        let mut filenames = BTreeSet::new();
        let mut total = 0_u64;
        for file in &self.files {
            require(
                CUDA_BUNDLE_FILENAMES.contains(&(file.filename.as_str(), file.role)),
                "cuda_bundle.files",
                "UnsupportedCudaBundleFile",
                "CUDA bundle filename and role must match C's closed first profile",
            )?;
            require(
                filenames.insert(file.filename.as_str()),
                "cuda_bundle.files",
                "DuplicateCudaBundleFile",
                "CUDA bundle filenames must be unique",
            )?;
            file.artifact.validate()?;
            let expected_path = if parent.is_empty() {
                file.filename.clone()
            } else {
                format!("{parent}/{}", file.filename)
            };
            require(
                file.artifact.path == expected_path && file.artifact.path != self.manifest.path,
                "cuda_bundle.files.artifact.path",
                "CudaBundleSiblingPath",
                "CUDA libraries must be distinct siblings of the manifest with the exact declared basename",
            )?;
            require(
                file.artifact.bytes <= CUDA_BUNDLE_FILE_MAX_BYTES,
                "cuda_bundle.files.artifact.bytes",
                "CudaBundleFileBudget",
                "CUDA bundle library exceeds 1 GiB",
            )?;
            total = total.checked_add(file.artifact.bytes).ok_or_else(|| {
                violation(
                    "cuda_bundle.files",
                    "BudgetOverflow",
                    "CUDA bundle total byte count overflow",
                )
            })?;
        }
        require(
            total <= CUDA_BUNDLE_MAX_BYTES,
            "cuda_bundle.files",
            "CudaBundleBudget",
            "CUDA bundle libraries exceed 4 GiB",
        )?;
        Ok(total)
    }

    /// Exact C codec: domain, u32-BE schema/count, filename-sorted entries with
    /// u32-BE name length, name bytes, u8 role, u64-BE size, 32 decoded SHA bytes.
    pub fn canonical_digest(&self) -> Result<String, ManifestError> {
        self.metadata_bytes()?;
        let mut files: Vec<_> = self.files.iter().collect();
        files.sort_unstable_by(|left, right| left.filename.cmp(&right.filename));
        let mut digest = Sha256::new();
        digest.update(b"RoveZero/native-cuda-bundle\0binary-codec-v1\0");
        digest.update(1_u32.to_be_bytes());
        digest.update((files.len() as u32).to_be_bytes());
        for file in files {
            digest.update((file.filename.len() as u32).to_be_bytes());
            digest.update(file.filename.as_bytes());
            digest.update([file.role.codec_tag()]);
            digest.update(file.artifact.bytes.to_be_bytes());
            // ArtifactRef metadata validation has already checked exact lowercase
            // hex length, so each pair decodes without fallible external input.
            let mut raw = [0_u8; 32];
            for (index, pair) in file.artifact.sha256.as_bytes().chunks_exact(2).enumerate() {
                raw[index] = (hex_digit(pair[0]) << 4) | hex_digit(pair[1]);
            }
            digest.update(raw);
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn same_inputs_as(&self, other: &Self) -> Result<bool, ManifestError> {
        self.validate()?;
        other.validate()?;
        if self.manifest != other.manifest || self.canonical_sha256 != other.canonical_sha256 {
            return Ok(false);
        }
        for file in &self.files {
            if &file.artifact != other.file(file.role, &file.filename)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn hex_digit(byte: u8) -> u8 {
    if byte.is_ascii_digit() {
        byte - b'0'
    } else {
        byte - b'a' + 10
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativeCudaIntegrationPurpose {
    CudaNnIntegration,
    CudaSearchPilot,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativeCudaProvider {
    OnnxRuntimeCuda,
}
/// Expected static profile, never evidence that the requested provider ran.
/// ORT placement trace hashes include execution data and are verified after
/// execution; they cannot be declared as an expected immutable launch input.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaProfileV1 {
    pub provider: NativeCudaProvider,
    pub precision: NativePrecision,
    pub batch_size: u32,
    pub intra_threads: u32,
    pub search_workers: u32,
    pub history_fill: NativeHistoryFill,
    pub fresh_only: bool,
    pub full_steps: u32,
    pub device_id: i32,
    /// ORT arena request, not a kernel-enforced VRAM quota.
    pub arena_bytes: u64,
    pub tf32: bool,
    pub expected_backend_sha256: String,
    pub expected_encoding_sha256: String,
}

impl NativeCudaProfileV1 {
    pub fn fixed(backend_sha256: String, encoding_sha256: String) -> Self {
        Self {
            provider: NativeCudaProvider::OnnxRuntimeCuda,
            precision: NativePrecision::Fp32,
            batch_size: 1,
            intra_threads: 1,
            search_workers: 1,
            history_fill: NativeHistoryFill::No,
            fresh_only: true,
            full_steps: 1,
            device_id: 0,
            arena_bytes: CUDA_NATIVE_ARENA_BYTES,
            tf32: false,
            expected_backend_sha256: backend_sha256,
            expected_encoding_sha256: encoding_sha256,
        }
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        self.validate_arena(CUDA_NATIVE_ARENA_BYTES)
    }
    pub(crate) fn validate_arena(&self, arena_bytes: u64) -> Result<(), ManifestError> {
        self.validate_batch_arena(arena_bytes, 1)
    }
    pub(crate) fn validate_batch_arena(
        &self,
        arena_bytes: u64,
        batch: u32,
    ) -> Result<(), ManifestError> {
        require(
            self.batch_size == batch
                && (1..=16).contains(&batch)
                && self.intra_threads == 1
                && self.search_workers == 1
                && self.fresh_only
                && self.full_steps == 1
                && self.device_id == 0
                && self.arena_bytes == arena_bytes
                && !self.tf32,
            "profile",
            "UnsupportedNativeProfile",
            "requires the selected CUDA ONNX arena, device=0 tf32=false FP32 batch=1 intra_threads=1 worker=1 fresh full_steps=1 history-fill No",
        )?;
        hash(
            &self.expected_backend_sha256,
            64,
            "profile.expected_backend_sha256",
        )?;
        hash(
            &self.expected_encoding_sha256,
            64,
            "profile.expected_encoding_sha256",
        )
    }
}

/// Fixed supported launch recipe. There is no arbitrary argv or UCI option map.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CudaNativeLaunchSpec<P> {
    pub role: NativeEngineRole,
    pub engine_id: String,
    pub source_commit: String,
    pub target: String,
    pub artifacts: Vec<NativeArtifactBinding>,
    pub profile: P,
    pub cuda_bundle: CudaBundleBindingV1,
}

impl<P: CudaLaunchProfile> CudaNativeLaunchSpec<P> {
    pub fn artifact(&self, role: NativeArtifactRole) -> Result<&ArtifactRef, ManifestError> {
        let mut found = self.artifacts.iter().filter(|binding| binding.role == role);
        let first = found.next().ok_or_else(|| {
            violation(
                "artifacts",
                "MissingArtifactRole",
                "required native artifact role is absent",
            )
        })?;
        require(
            found.next().is_none(),
            "artifacts",
            "DuplicateArtifactRole",
            "native artifact roles must be unique",
        )?;
        Ok(&first.artifact)
    }

    /// Equality includes path and provenance, not only equal content hashes.
    /// Engine ID and role are the only permitted differences in this first gate.
    pub fn same_inputs_as(&self, other: &Self) -> Result<bool, ManifestError> {
        let pinned = self.same_pinned_inputs_as(other)?;
        Ok(self.profile == other.profile && pinned)
    }

    /// Exact artifacts/provenance without interpreting a search-policy delta.
    /// The sealed pair profile must independently validate that delta.
    pub fn same_pinned_inputs_as(&self, other: &Self) -> Result<bool, ManifestError> {
        self.validate()?;
        other.validate()?;
        if self.source_commit != other.source_commit
            || self.target != other.target
            || !self.cuda_bundle.same_inputs_as(&other.cuda_bundle)?
        {
            return Ok(false);
        }
        for role in NativeArtifactRole::ALL {
            if self.artifact(role)? != other.artifact(role)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        id(&self.engine_id, "engine_id")?;
        hash(&self.source_commit, 40, "source_commit")?;
        text(&self.target, 128, "target")?;
        self.profile.validate_profile()?;
        self.cuda_bundle.validate()?;
        require(
            self.artifacts.len() == NativeArtifactRole::ALL.len(),
            "artifacts",
            "ArtifactRoleCount",
            "requires exactly the five native artifact roles",
        )?;
        let mut roles = BTreeSet::new();
        let mut paths = BTreeSet::new();
        for binding in &self.artifacts {
            require(
                roles.insert(binding.role),
                "artifacts",
                "DuplicateArtifactRole",
                "native artifact roles must be unique",
            )?;
            binding.artifact.validate()?;
            require(
                binding.artifact.bytes <= P::artifact_limit(binding.role),
                "artifacts.bytes",
                "ArtifactRoleBudget",
                "native artifact exceeds its role byte ceiling",
            )?;
            require(
                paths.insert(&binding.artifact.path),
                "artifacts.path",
                "ArtifactRoleAlias",
                "different native roles must not alias one logical file",
            )?;
        }
        self.profile
            .validate_model(self.artifact(NativeArtifactRole::SourceWeights)?)?;
        let core = self
            .cuda_bundle
            .file(CudaBundleFileRoleV1::Core, "libonnxruntime.so.1.22.0")?;
        require(
            self.artifact(NativeArtifactRole::OrtLibrary)? == core,
            "artifacts.ort_library",
            "CudaCoreBindingMismatch",
            "OrtLibrary must equal the complete Core ArtifactRef, including provenance",
        )?;
        for binding in &self.artifacts {
            require(
                binding.artifact.path != self.cuda_bundle.manifest.path
                    && (binding.role == NativeArtifactRole::OrtLibrary
                        || self
                            .cuda_bundle
                            .files
                            .iter()
                            .all(|file| file.artifact.path != binding.artifact.path)),
                "artifacts.path",
                "ArtifactRoleAlias",
                "native inputs may alias only the explicitly shared CUDA Core library",
            )?;
        }
        Ok(())
    }
}

/// Requested bounds. The executor must record observed enforcement separately.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaResourceBudgetV1 {
    pub max_input_bytes: u64,
    /// Combined retained streams, PGN and E metadata; native output tree is separate.
    pub max_output_bytes: u64,
    /// All files below native output roots, including ORT copies and native receipts.
    pub max_runtime_bytes: u64,
    /// Input/output/runtime reservations, not a claim of disk enforcement.
    pub max_artifact_bytes: u64,
    /// Includes the external runner leader and both engines.
    pub max_child_processes: u32,
    /// Bound on native-tree entries, counting directories as well as regular files.
    /// The executor additionally budgets fixed input and E output entries.
    pub max_runtime_files: u32,
    pub max_runtime_depth: u8,
    /// Per-process address-space request, never an aggregate RAM or VRAM quota.
    pub address_space_per_process_bytes: u64,
}

/// One pair, exactly two games, no retries and no strength interpretation.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, bound = "P: CudaLaunchProfile")]
pub struct CudaIntegrationPairSpec<P: CudaLaunchProfile> {
    pub schema_version: u32,
    pub purpose: NativeCudaIntegrationPurpose,
    pub strength_eligible: bool,
    pub contract_revision: String,
    pub run_id: String,
    pub pair_id: String,
    pub engines: [CudaNativeLaunchSpec<P>; 2],
    /// White's role in execution order; must contain each role exactly once.
    /// The board and the complete opening trace remain identical for both games.
    pub white_order: [NativeEngineRole; 2],
    pub opening: OpeningSpec,
    pub opening_artifact: ArtifactRef,
    pub runner: ToolIdentity,
    pub clock: P::Clock,
    /// Includes the full opening prefix. A's 4095-ply audit ceiling applies.
    pub max_plies: u32,
    pub timeouts: NativeTimeoutsV1,
    pub budget: NativeCudaResourceBudgetV1,
    /// Absent in V1/V2; required only by the separately sealed pilot profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pilot: Option<crate::NativeSearchPilotProtocolV3>,
}

impl<P: CudaLaunchProfile> CudaIntegrationPairSpec<P> {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        bounded_json(input)?;
        let spec: Self = decode_json(input)?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn engine(
        &self,
        role: NativeEngineRole,
    ) -> Result<&CudaNativeLaunchSpec<P>, ManifestError> {
        let mut matches = self.engines.iter().filter(|engine| engine.role == role);
        let engine = matches.next().ok_or_else(|| {
            violation(
                "engines",
                "MissingEngineRole",
                "required engine role is absent",
            )
        })?;
        require(
            matches.next().is_none(),
            "engines",
            "DuplicateEngineRole",
            "engine roles must be unique",
        )?;
        Ok(engine)
    }

    pub fn declared_artifacts(&self) -> Vec<&ArtifactRef> {
        let mut artifacts = vec![&self.runner.binary, &self.opening_artifact];
        if let Some(patch) = &self.runner.dirty_patch {
            artifacts.push(patch);
        }
        if let Some(pilot) = &self.pilot {
            artifacts.push(&pilot.opening_cohort);
        }
        for engine in &self.engines {
            artifacts.extend(engine.artifacts.iter().map(|binding| &binding.artifact));
            artifacts.push(&engine.cuda_bundle.manifest);
            artifacts.extend(engine.cuda_bundle.files.iter().map(|file| &file.artifact));
        }
        artifacts
    }

    pub fn unique_input_bytes(&self) -> Result<u64, ManifestError> {
        unique_input_bytes(self.declared_artifacts())
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.schema_version == P::VERSION
                && self.purpose == P::PURPOSE
                && !self.strength_eligible,
            "schema_version",
            "UnsupportedIntegrationScope",
            "only the selected schema integration-only strength=false is supported",
        )?;
        require(
            self.contract_revision == NATIVE_ENGINE_CONTRACT_REVISION,
            "contract_revision",
            "UnsupportedContract",
            "native engine contract revision must be exactly 0.1",
        )?;
        id(&self.run_id, "run_id")?;
        id(&self.pair_id, "pair_id")?;
        for engine in &self.engines {
            engine.validate()?;
        }
        let baseline = self.engine(NativeEngineRole::Baseline)?;
        let candidate = self.engine(NativeEngineRole::Candidate)?;
        require(
            baseline.engine_id != candidate.engine_id,
            "engines.engine_id",
            "DuplicateEngineId",
            "baseline and candidate must have distinct protocol identities",
        )?;
        require(
            baseline.same_pinned_inputs_as(candidate)?
                && P::pair_profiles_match(&baseline.profile, &candidate.profile),
            "engines",
            "IntegrationInputMismatch",
            "this first integration pair requires identical executable, model, runtime and profile inputs",
        )?;
        require(
            self.white_order[0] != self.white_order[1],
            "white_order",
            "InvalidColorOrder",
            "each engine role must play white once",
        )?;
        validate_opening(&self.opening, self.max_plies)?;
        P::validate_protocol(self.pilot.as_ref(), self.max_plies, self.white_order)?;
        self.opening_artifact.validate()?;
        self.runner.binary.validate()?;
        require(
            self.runner.binary.bytes <= NativeArtifactRole::Binary.byte_limit(),
            "runner.binary",
            "ArtifactRoleBudget",
            "runner binary exceeds native binary declaration ceiling",
        )?;
        P::validate_runner(&self.runner)?;
        hash(&self.runner.source_commit, 40, "runner.source_commit")?;
        for (name, value) in [
            ("version", &self.runner.version),
            ("source_url", &self.runner.source_url),
            ("build_mode", &self.runner.build_mode),
            ("compiler", &self.runner.compiler),
            ("target", &self.runner.target),
            ("isa", &self.runner.isa),
        ] {
            text(value, 4096, &format!("runner.{name}"))?;
        }
        require(
            self.runner.source_url == self.runner.binary.source,
            "runner.source_url",
            "RunnerSourceMismatch",
            "runner source URL must match its artifact provenance",
        )?;
        // Same-role sharing across engines is permitted; cross-role pathname aliases
        // are rejected before collecting a map so no identity is silently overwritten.
        let mut paths: BTreeSet<_> = baseline
            .artifacts
            .iter()
            .map(|binding| &binding.artifact.path)
            .collect();
        paths.insert(&baseline.cuda_bundle.manifest.path);
        paths.extend(
            baseline
                .cuda_bundle
                .files
                .iter()
                .map(|file| &file.artifact.path),
        );
        require(
            self.runner.binary.path != self.opening_artifact.path
                && !paths.contains(&self.runner.binary.path)
                && !paths.contains(&self.opening_artifact.path),
            "artifacts.path",
            "ArtifactRoleAlias",
            "runner, opening and native input roles must be distinct logical files",
        )?;
        if let Some(patch) = &self.runner.dirty_patch {
            require(
                patch.path != self.runner.binary.path
                    && patch.path != self.opening_artifact.path
                    && !paths.contains(&patch.path),
                "runner.dirty_patch.path",
                "ArtifactRoleAlias",
                "runner patch must be a distinct verified input",
            )?;
        }
        if let Some(pilot) = &self.pilot {
            require(
                pilot.opening_cohort.path != self.runner.binary.path
                    && pilot.opening_cohort.path != self.opening_artifact.path
                    && self
                        .runner
                        .dirty_patch
                        .as_ref()
                        .is_none_or(|patch| patch.path != pilot.opening_cohort.path)
                    && !paths.contains(&pilot.opening_cohort.path),
                "pilot.opening_cohort.path",
                "ArtifactRoleAlias",
                "cohort must be a distinct verified input",
            )?;
        }
        let bytes = self.unique_input_bytes()?;
        validate_budgets(self, bytes)?;
        bounded_json(&serde_json::to_string(self).map_err(serialization_error)?)
    }

    pub fn lock(mut self) -> Result<LockedCudaIntegrationPairSpec<P>, ManifestError> {
        self.validate()?;
        // Native role order is semantically irrelevant; normalize it before hashing.
        self.engines.sort_by_key(|engine| engine.role);
        for engine in &mut self.engines {
            engine.artifacts.sort_by_key(|binding| binding.role);
            engine
                .cuda_bundle
                .files
                .sort_by(|left, right| left.filename.cmp(&right.filename));
        }
        let sha256 = digest(&canonical_bytes(&self)?);
        let locked = LockedCudaIntegrationPairSpec {
            input: self,
            sha256,
        };
        // The exported envelope, not only its embedded input, must fit the cap.
        locked.to_json()?;
        Ok(locked)
    }
}

#[derive(Clone, Debug)]
pub struct LockedCudaIntegrationPairSpec<P: CudaLaunchProfile> {
    input: CudaIntegrationPairSpec<P>,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields, bound = "P: CudaLaunchProfile")]
struct NativeLockEnvelope<P: CudaLaunchProfile> {
    lock_version: u32,
    domain: String,
    canonicalization: String,
    execution_ready: bool,
    input_sha256: String,
    input: CudaIntegrationPairSpec<P>,
}

impl<P: CudaLaunchProfile> LockedCudaIntegrationPairSpec<P> {
    pub fn input(&self) -> &CudaIntegrationPairSpec<P> {
        &self.input
    }
    pub fn spec(&self) -> &CudaIntegrationPairSpec<P> {
        &self.input
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn declared_artifacts(&self) -> Vec<&ArtifactRef> {
        self.input.declared_artifacts()
    }
    pub fn unique_input_bytes(&self) -> Result<u64, ManifestError> {
        self.input.unique_input_bytes()
    }

    pub fn to_json(&self) -> Result<String, ManifestError> {
        let json = serde_json::to_string(&NativeLockEnvelope {
            lock_version: 1,
            domain: P::DOMAIN.into(),
            canonicalization: P::CANONICALIZATION.into(),
            execution_ready: false,
            input_sha256: self.sha256.clone(),
            input: self.input.clone(),
        })
        .map_err(serialization_error)?;
        bounded_json(&json)?;
        Ok(json)
    }

    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        bounded_json(input)?;
        let envelope: NativeLockEnvelope<P> = decode_json(input)?;
        require(
            envelope.lock_version == 1
                && envelope.domain == P::DOMAIN
                && envelope.canonicalization == P::CANONICALIZATION
                && !envelope.execution_ready,
            "lock",
            "UnsupportedNativeLock",
            "native declaration lock version/domain must match and cannot authorize execution",
        )?;
        hash(&envelope.input_sha256, 64, "input_sha256")?;
        let locked = envelope.input.lock()?;
        require(
            locked.sha256 == envelope.input_sha256,
            "input_sha256",
            "NativeLockDigestMismatch",
            "native integration input digest differs",
        )?;
        Ok(locked)
    }
}

fn canonical_bytes<P: CudaLaunchProfile>(
    input: &CudaIntegrationPairSpec<P>,
) -> Result<Vec<u8>, ManifestError> {
    let mut value = serde_json::json!({"domain": P::DOMAIN,
        "schema_version": P::VERSION, "execution_ready": false, "input": input});
    value.sort_all_objects();
    let bytes = serde_json::to_vec(&value).map_err(serialization_error)?;
    require(
        bytes.len() <= MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES,
        "payload",
        "NativePayloadLimit",
        "native canonical payload exceeds 64 KiB",
    )?;
    Ok(bytes)
}

fn unique_input_bytes(artifacts: Vec<&ArtifactRef>) -> Result<u64, ManifestError> {
    let mut unique = BTreeMap::<&str, &ArtifactRef>::new();
    for artifact in artifacts {
        artifact.validate()?;
        if let Some(previous) = unique.get(artifact.path.as_str()) {
            require(
                *previous == artifact,
                "artifacts.path",
                "ArtifactIdentityConflict",
                "shared logical path must have identical digest, bytes and provenance",
            )?;
        } else {
            unique.insert(&artifact.path, artifact);
        }
    }
    unique.values().try_fold(0u64, |bytes, artifact| {
        bytes.checked_add(artifact.bytes).ok_or_else(|| {
            violation(
                "artifacts.bytes",
                "BudgetOverflow",
                "unique artifact byte sum overflow",
            )
        })
    })
}

fn validate_budgets<P: CudaLaunchProfile>(
    input: &CudaIntegrationPairSpec<P>,
    bytes: u64,
) -> Result<(), ManifestError> {
    let timeouts = input.timeouts;
    for (name, value) in [
        ("startup_ms", timeouts.startup_ms),
        ("handshake_ms", timeouts.handshake_ms),
        ("runtime_ms", timeouts.runtime_ms),
        ("drain_ms", timeouts.drain_ms),
        ("shutdown_ms", timeouts.shutdown_ms),
    ] {
        require(
            value > 0 && value <= MAX_WALL_MS && value <= timeouts.runtime_ms,
            &format!("timeouts.{name}"),
            "NativeTimeoutBudget",
            "timeouts must be positive, bounded and fit the pair runtime wall limit",
        )?;
    }
    require(
        timeouts.max_supervised_wall_ms()? <= MAX_WALL_MS,
        "timeouts",
        "NativeTimeoutBudget",
        "pair runtime plus two shutdown graces must fit one day",
    )?;
    P::validate_clock(input.clock, timeouts.runtime_ms)?;
    if matches!(P::VERSION, 4 | 5) {
        require(
            timeouts.shutdown_ms == 30_000,
            "timeouts.shutdown_ms",
            "NativeTimeoutBudget",
            "whole-clock memory/batch checks require 30s cleanup",
        )?;
    }
    let budget = input.budget;
    require(
        budget.max_input_bytes > 0
            && budget.max_input_bytes <= CUDA_BUNDLE_MAX_BYTES + 128 * 1024 * 1024
            && bytes <= budget.max_input_bytes,
        "budget.max_input_bytes",
        "NativeInputBudget",
        "unique inputs exceed the positive input byte ceiling",
    )?;
    require(
        budget.max_output_bytes > 0
            && budget.max_runtime_bytes > 0
            && budget.max_artifact_bytes > 0
            && budget.address_space_per_process_bytes > 0,
        "budget",
        "NativeResourceBudget",
        "output, runtime, artifact and per-process address-space bounds must be positive",
    )?;
    require(
        budget.max_output_bytes <= 64 * 1024 * 1024
            && budget.max_runtime_bytes <= 16 * CUDA_BUNDLE_FILE_MAX_BYTES + 64 * 1024 * 1024
            && budget.max_artifact_bytes <= 21 * CUDA_BUNDLE_FILE_MAX_BYTES
            && budget.address_space_per_process_bytes <= 128 * CUDA_BUNDLE_FILE_MAX_BYTES,
        "budget",
        "NativeResourceBudget",
        "CUDA declaration ceilings are output 64MiB, runtime 16GiB+64MiB, artifact 21GiB and per-process address space 128GiB",
    )?;
    let bundle_bytes = input
        .engine(NativeEngineRole::Baseline)?
        .cuda_bundle
        .total_bytes()?;
    let runtime_reservation = (if P::VERSION == 1 { bundle_bytes } else { 0 })
        .checked_mul(CUDA_PROCESS_START_RESERVATIONS)
        .and_then(|bytes| {
            bytes.checked_add(
                (CUDA_PLACEMENT_TRACE_RESERVE_PER_PROCESS_BYTES
                    + if P::VERSION == 4 {
                        64 * 1024 * 1024 + 768 * 1024
                    } else if P::VERSION == 1 {
                        CUDA_RECEIPT_RESERVE_PER_PROCESS_BYTES
                    } else {
                        768 * 1024
                    })
                    * CUDA_PROCESS_START_RESERVATIONS,
            )
        })
        .ok_or_else(|| {
            violation(
                "budget.max_runtime_bytes",
                "BudgetOverflow",
                "CUDA runtime reservation overflow",
            )
        })?;
    require(
        budget.max_runtime_bytes >= runtime_reservation,
        "budget.max_runtime_bytes",
        "CudaRuntimeReservation",
        "runtime tree must reserve the selected storage policy, four 4MiB placement traces and four 512KiB receipt allowances",
    )?;
    require(
        (3..=16).contains(&budget.max_child_processes)
            && (128..=1024).contains(&budget.max_runtime_files)
            && (4..=8).contains(&budget.max_runtime_depth),
        "budget",
        "NativeResourceBudget",
        "requires 3..16 children, 128..1024 runtime files and runtime depth 4..8",
    )?;
    let total = budget
        .max_input_bytes
        .checked_add(budget.max_output_bytes)
        .and_then(|total| total.checked_add(budget.max_runtime_bytes))
        .ok_or_else(|| {
            violation(
                "budget",
                "BudgetOverflow",
                "input/output/runtime byte sum overflow",
            )
        })?;
    require(
        total <= budget.max_artifact_bytes,
        "budget.max_artifact_bytes",
        "NativeArtifactBudget",
        "input, retained output and native runtime reservations must fit the aggregate artifact ceiling",
    )
}

fn validate_opening(opening: &OpeningSpec, max_plies: u32) -> Result<(), ManifestError> {
    id(&opening.id, "opening.id")?;
    text(&opening.history_origin, 128, "opening.history_origin")?;
    require(
        max_plies > 0 && max_plies <= MAX_PLIES && opening.moves.len() < max_plies as usize,
        "max_plies",
        "NativePlyBudget",
        "full opening prefix must leave searched plies within the 4095-ply ceiling",
    )?;
    require(
        (max_plies as usize - opening.moves.len()) % 2 == 0,
        "max_plies",
        "NativePlyBudget",
        "Fastchess maxmoves requires a positive even searched-ply remainder",
    )?;
    for movement in &opening.moves {
        let bytes = movement.as_bytes();
        require(
            matches!(bytes.len(), 4 | 5)
                && (b'a'..=b'h').contains(&bytes[0])
                && (b'1'..=b'8').contains(&bytes[1])
                && (b'a'..=b'h').contains(&bytes[2])
                && (b'1'..=b'8').contains(&bytes[3])
                && (bytes.len() == 4 || b"qrbn".contains(&bytes[4])),
            "opening.moves",
            "MalformedMove",
            "opening moves must have bounded UCI syntax; legality remains A-owned",
        )?;
    }
    match opening.initial {
        InitialPosition::Startpos => require(
            opening.fen.is_none() && opening.history == HistoryCompleteness::Complete,
            "opening",
            "HistoryMismatch",
            "startpos trace has complete history and no FEN",
        ),
        InitialPosition::Fen => {
            require(
                opening.history == HistoryCompleteness::UnknownPrefix,
                "opening",
                "HistoryMismatch",
                "FEN origin must retain unknown-prefix history",
            )?;
            let fen = opening
                .fen
                .as_ref()
                .ok_or_else(|| violation("opening.fen", "MalformedFen", "FEN is required"))?;
            text(fen, 256, "opening.fen")?;
            require(
                fen.split_whitespace().count() == 6,
                "opening.fen",
                "MalformedFen",
                "FEN must have six fields; semantics remain A-owned",
            )
        }
    }
}

fn bounded_json(input: &str) -> Result<(), ManifestError> {
    require(
        input.len() <= MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES,
        "payload",
        "NativePayloadLimit",
        "native declaration JSON exceeds 64 KiB",
    )
}
fn serialization_error(error: serde_json::Error) -> ManifestError {
    ManifestError::Integrity(error.to_string())
}
fn violation(path: &str, code: &'static str, message: &str) -> ManifestError {
    ManifestError::Validation(vec![Violation {
        path: path.into(),
        code,
        message: message.into(),
    }])
}
fn require(
    condition: bool,
    path: &str,
    code: &'static str,
    message: &str,
) -> Result<(), ManifestError> {
    if condition {
        Ok(())
    } else {
        Err(violation(path, code, message))
    }
}
fn text(value: &str, limit: usize, path: &str) -> Result<(), ManifestError> {
    require(
        !value.is_empty()
            && value.len() <= limit
            && value.trim() == value
            && !value.chars().any(char::is_control)
            && !["tbd", "todo", "unknown", "latest", "placeholder", "unset"]
                .contains(&value.to_ascii_lowercase().as_str()),
        path,
        "UnresolvedValue",
        "requires bounded explicit text without controls or placeholders",
    )
}
fn id(value: &str, path: &str) -> Result<(), ManifestError> {
    text(value, 128, path)?;
    require(
        value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte)),
        path,
        "InvalidId",
        "identity must contain ASCII letters, digits, hyphen, underscore or dot",
    )
}
fn hash(value: &str, length: usize, path: &str) -> Result<(), ManifestError> {
    require(
        value.len() == length
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            && value.bytes().any(|byte| byte != b'0'),
        path,
        "InvalidDigest",
        "requires a nonzero lowercase hexadecimal digest of the exact length",
    )
}

pub(crate) mod profile_sealed {
    pub trait Sealed {}
}
/// Closed profiles share ownership/locking algorithms. V1/V2 are A/A;
/// V3 is a separately locked, bounded S0/S1 pilot with a whole-engine clock.
/// arbitrary profiles cannot widen budgets or substitute a wire domain.
pub trait CudaLaunchProfile:
    profile_sealed::Sealed
    + Clone
    + std::fmt::Debug
    + Eq
    + Serialize
    + DeserializeOwned
    + Send
    + 'static
{
    type Clock: Copy + std::fmt::Debug + Eq + Serialize + DeserializeOwned + Send;
    const VERSION: u32;
    const DOMAIN: &'static str;
    const CANONICALIZATION: &'static str;
    const PURPOSE: NativeCudaIntegrationPurpose = NativeCudaIntegrationPurpose::CudaNnIntegration;
    fn clock_view(clock: Self::Clock) -> crate::NativePairClock;
    fn validate_clock(clock: Self::Clock, runtime_ms: u64) -> Result<(), ManifestError>;
    fn pair_profiles_match(baseline: &Self, candidate: &Self) -> bool {
        baseline == candidate
    }
    fn validate_protocol(
        protocol: Option<&crate::NativeSearchPilotProtocolV3>,
        _max_plies: u32,
        _white_order: [NativeEngineRole; 2],
    ) -> Result<(), ManifestError> {
        require(
            protocol.is_none(),
            "pilot",
            "UnsupportedIntegrationScope",
            "V1/V2 integration declarations cannot adopt a pilot protocol",
        )
    }
    fn validate_runner(runner: &ToolIdentity) -> Result<(), ManifestError> {
        require(
            !runner.dirty && runner.dirty_patch.is_none(),
            "runner",
            "DirtyNativeRunner",
            "the integration profile requires a clean pinned runner",
        )
    }
    fn validate_profile(&self) -> Result<(), ManifestError>;
    fn runtime_profile(&self) -> &NativeCudaProfileV1;
    fn search_options(&self) -> Option<crate::NativeCudaSearchV2> {
        None
    }
    fn max_batch(&self) -> usize {
        1
    }
    fn artifact_limit(role: NativeArtifactRole) -> u64;
    fn validate_model(&self, _source: &ArtifactRef) -> Result<(), ManifestError> {
        Ok(())
    }
}
impl profile_sealed::Sealed for NativeCudaProfileV1 {}
impl CudaLaunchProfile for NativeCudaProfileV1 {
    type Clock = NativeMovetimeV1;
    const VERSION: u32 = CUDA_NATIVE_LAUNCH_SCHEMA_VERSION;
    const DOMAIN: &'static str = CUDA_NATIVE_LAUNCH_DOMAIN;
    const CANONICALIZATION: &'static str = CUDA_NATIVE_LAUNCH_CANONICALIZATION;
    fn clock_view(clock: Self::Clock) -> crate::NativePairClock {
        crate::NativePairClock::Movetime(clock)
    }
    fn validate_clock(clock: Self::Clock, runtime_ms: u64) -> Result<(), ManifestError> {
        require(
            clock.movetime_ms > 0 && clock.movetime_ms <= runtime_ms,
            "clock.movetime_ms",
            "NativeMovetimeBudget",
            "positive per-move milliseconds must fit the independent pair wall limit",
        )
    }
    fn validate_profile(&self) -> Result<(), ManifestError> {
        self.validate()
    }
    fn runtime_profile(&self) -> &NativeCudaProfileV1 {
        self
    }
    fn artifact_limit(role: NativeArtifactRole) -> u64 {
        if role == NativeArtifactRole::OrtLibrary {
            CUDA_BUNDLE_FILE_MAX_BYTES
        } else {
            role.byte_limit()
        }
    }
}
pub type CudaNativeLaunchSpecV1 = CudaNativeLaunchSpec<NativeCudaProfileV1>;
pub type CudaIntegrationPairSpecV1 = CudaIntegrationPairSpec<NativeCudaProfileV1>;
pub type LockedCudaIntegrationPairSpecV1 = LockedCudaIntegrationPairSpec<NativeCudaProfileV1>;
