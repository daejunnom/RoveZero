//! E-owned declarations for one actual CPU neural integration pair.
//!
//! This schema is independent of the strength-experiment schema and of the
//! engine contract. A lock validates declarations, not A legality, immutable
//! file contents, provider initialization, neural execution or resource limits.
//! The executor must retain its verified input/output capabilities and compare
//! separately observed startup and termination receipts against this lock.

use crate::{
    ArtifactRef, HistoryCompleteness, InitialPosition, ManifestError, OpeningSpec, ToolIdentity,
    Violation, decode_json, digest,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const NATIVE_LAUNCH_SCHEMA_VERSION: u32 = 1;
pub const NATIVE_LAUNCH_DOMAIN: &str = "rz-e-native-integration-pair-v1";
pub const NATIVE_LAUNCH_CANONICALIZATION: &str = "rz-e-native-json-v1";
pub const MAX_NATIVE_LAUNCH_JSON_BYTES: usize = 64 * 1024;
pub const NATIVE_ENGINE_CONTRACT_REVISION: &str = "0.1";
const MAX_WALL_MS: u64 = 86_400_000;
const MAX_PLIES: u32 = 4095;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum NativeEngineRole {
    Baseline,
    Candidate,
}

impl NativeEngineRole {
    pub const fn other(self) -> Self {
        match self {
            Self::Baseline => Self::Candidate,
            Self::Candidate => Self::Baseline,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum NativeArtifactRole {
    Binary,
    SourceWeights,
    Onnx,
    ExportManifest,
    OrtLibrary,
}

impl NativeArtifactRole {
    pub const ALL: [Self; 5] = [
        Self::Binary,
        Self::SourceWeights,
        Self::Onnx,
        Self::ExportManifest,
        Self::OrtLibrary,
    ];

    /// Declaration ceilings; actual C loader support is separately checked.
    pub const fn byte_limit(self) -> u64 {
        match self {
            Self::Binary => 64 * 1024 * 1024,
            Self::SourceWeights => 4 * 1024 * 1024,
            Self::Onnx => 16 * 1024 * 1024,
            Self::ExportManifest => 64 * 1024,
            Self::OrtLibrary => 512 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeArtifactBinding {
    pub role: NativeArtifactRole,
    pub artifact: ArtifactRef,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativeIntegrationPurpose {
    CpuNnIntegration,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativeCpuProvider {
    OnnxRuntimeCpu,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativePrecision {
    Fp32,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativeHistoryFill {
    No,
}

/// Expected profile, never evidence that the requested provider ran.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeCpuProfileV1 {
    pub provider: NativeCpuProvider,
    pub precision: NativePrecision,
    pub batch_size: u32,
    pub intra_threads: u32,
    pub search_workers: u32,
    pub history_fill: NativeHistoryFill,
    pub fresh_only: bool,
    pub full_steps: u32,
    pub expected_backend_sha256: String,
    pub expected_encoding_sha256: String,
}

impl NativeCpuProfileV1 {
    pub fn fixed(backend_sha256: String, encoding_sha256: String) -> Self {
        Self {
            provider: NativeCpuProvider::OnnxRuntimeCpu,
            precision: NativePrecision::Fp32,
            batch_size: 1,
            intra_threads: 1,
            search_workers: 1,
            history_fill: NativeHistoryFill::No,
            fresh_only: true,
            full_steps: 1,
            expected_backend_sha256: backend_sha256,
            expected_encoding_sha256: encoding_sha256,
        }
    }

    fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.batch_size == 1
                && self.intra_threads == 1
                && self.search_workers == 1
                && self.fresh_only
                && self.full_steps == 1,
            "profile",
            "UnsupportedNativeProfile",
            "requires CPU ONNX FP32 batch=1 intra_threads=1 worker=1 fresh full_steps=1 history-fill No",
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
pub struct CpuNativeLaunchSpecV1 {
    pub role: NativeEngineRole,
    pub engine_id: String,
    pub source_commit: String,
    pub target: String,
    pub artifacts: Vec<NativeArtifactBinding>,
    pub profile: NativeCpuProfileV1,
}

impl CpuNativeLaunchSpecV1 {
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
        self.validate()?;
        other.validate()?;
        if self.source_commit != other.source_commit
            || self.target != other.target
            || self.profile != other.profile
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
        self.profile.validate()?;
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
                binding.artifact.bytes <= binding.role.byte_limit(),
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
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeMovetimeV1 {
    /// Milliseconds per searched move; executor maps this to Fastchess st=seconds.
    /// This declaration neither proves movetime delivery nor clock fairness.
    pub movetime_ms: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeTimeoutsV1 {
    pub startup_ms: u64,
    pub handshake_ms: u64,
    /// Whole pair supervisor wall limit, including startup and play, independent
    /// of per-move time. The two bounded outer shutdown graces are additional.
    pub runtime_ms: u64,
    /// Requested outer drain bound; actual B drain limits are separately observed.
    pub drain_ms: u64,
    pub shutdown_ms: u64,
}

impl NativeTimeoutsV1 {
    pub fn max_supervised_wall_ms(&self) -> Result<u64, ManifestError> {
        self.shutdown_ms
            .checked_mul(2)
            .and_then(|graces| self.runtime_ms.checked_add(graces))
            .ok_or_else(|| {
                violation(
                    "timeouts",
                    "BudgetOverflow",
                    "supervisor wall budget overflow",
                )
            })
    }
}

/// Requested bounds. The executor must record observed enforcement separately.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeResourceBudgetV1 {
    pub max_input_bytes: u64,
    /// Combined retained streams, PGN and E metadata; native output tree is separate.
    pub max_output_bytes: u64,
    /// All files below native output roots, including ORT copies and native receipts.
    pub max_runtime_bytes: u64,
    /// Unique input bytes + both output categories, not a claim of disk enforcement.
    pub max_artifact_bytes: u64,
    /// Includes the external runner leader and both engines.
    pub max_child_processes: u32,
    /// Bound on native-tree entries, counting directories as well as regular files.
    /// The executor additionally budgets fixed input and E output entries.
    pub max_runtime_files: u32,
    pub max_runtime_depth: u8,
    /// Per-process address-space request, never an aggregate RAM quota.
    pub address_space_per_process_bytes: u64,
}

/// One pair, exactly two games, no retries and no strength interpretation.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct IntegrationPairSpecV1 {
    pub schema_version: u32,
    pub purpose: NativeIntegrationPurpose,
    pub strength_eligible: bool,
    pub contract_revision: String,
    pub run_id: String,
    pub pair_id: String,
    pub engines: [CpuNativeLaunchSpecV1; 2],
    /// White's role in execution order; must contain each role exactly once.
    /// The board and the complete opening trace remain identical for both games.
    pub white_order: [NativeEngineRole; 2],
    pub opening: OpeningSpec,
    pub opening_artifact: ArtifactRef,
    pub runner: ToolIdentity,
    pub clock: NativeMovetimeV1,
    /// Includes the full opening prefix. A's 4095-ply audit ceiling applies.
    pub max_plies: u32,
    pub timeouts: NativeTimeoutsV1,
    pub budget: NativeResourceBudgetV1,
}

impl IntegrationPairSpecV1 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        bounded_json(input)?;
        let spec: Self = decode_json(input)?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn engine(&self, role: NativeEngineRole) -> Result<&CpuNativeLaunchSpecV1, ManifestError> {
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
        for engine in &self.engines {
            artifacts.extend(engine.artifacts.iter().map(|binding| &binding.artifact));
        }
        artifacts
    }

    pub fn unique_input_bytes(&self) -> Result<u64, ManifestError> {
        unique_input_bytes(self.declared_artifacts())
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.schema_version == NATIVE_LAUNCH_SCHEMA_VERSION && !self.strength_eligible,
            "schema_version",
            "UnsupportedIntegrationScope",
            "only schema 1 integration-only strength=false is supported",
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
            baseline.same_inputs_as(candidate)?,
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
        self.opening_artifact.validate()?;
        self.runner.binary.validate()?;
        require(
            self.runner.binary.bytes <= NativeArtifactRole::Binary.byte_limit(),
            "runner.binary",
            "ArtifactRoleBudget",
            "runner binary exceeds native binary declaration ceiling",
        )?;
        require(
            !self.runner.dirty && self.runner.dirty_patch.is_none(),
            "runner",
            "DirtyNativeRunner",
            "the first native integration profile requires a clean pinned runner",
        )?;
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
        let paths: BTreeSet<_> = baseline
            .artifacts
            .iter()
            .map(|binding| &binding.artifact.path)
            .collect();
        require(
            self.runner.binary.path != self.opening_artifact.path
                && !paths.contains(&self.runner.binary.path)
                && !paths.contains(&self.opening_artifact.path),
            "artifacts.path",
            "ArtifactRoleAlias",
            "runner, opening and native input roles must be distinct logical files",
        )?;
        let bytes = self.unique_input_bytes()?;
        validate_budgets(self, bytes)?;
        bounded_json(&serde_json::to_string(self).map_err(serialization_error)?)
    }

    pub fn lock(mut self) -> Result<LockedIntegrationPairSpecV1, ManifestError> {
        self.validate()?;
        // Native role order is semantically irrelevant; normalize it before hashing.
        self.engines.sort_by_key(|engine| engine.role);
        for engine in &mut self.engines {
            engine.artifacts.sort_by_key(|binding| binding.role);
        }
        let sha256 = digest(&canonical_bytes(&self)?);
        let locked = LockedIntegrationPairSpecV1 {
            input: self,
            sha256,
        };
        // The exported envelope, not only its embedded input, must fit the cap.
        locked.to_json()?;
        Ok(locked)
    }
}

#[derive(Clone, Debug)]
pub struct LockedIntegrationPairSpecV1 {
    input: IntegrationPairSpecV1,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeLockEnvelopeV1 {
    lock_version: u32,
    domain: String,
    canonicalization: String,
    execution_ready: bool,
    input_sha256: String,
    input: IntegrationPairSpecV1,
}

impl LockedIntegrationPairSpecV1 {
    pub fn input(&self) -> &IntegrationPairSpecV1 {
        &self.input
    }
    pub fn spec(&self) -> &IntegrationPairSpecV1 {
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
        let json = serde_json::to_string(&NativeLockEnvelopeV1 {
            lock_version: 1,
            domain: NATIVE_LAUNCH_DOMAIN.into(),
            canonicalization: NATIVE_LAUNCH_CANONICALIZATION.into(),
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
        let envelope: NativeLockEnvelopeV1 = decode_json(input)?;
        require(
            envelope.lock_version == 1
                && envelope.domain == NATIVE_LAUNCH_DOMAIN
                && envelope.canonicalization == NATIVE_LAUNCH_CANONICALIZATION
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

fn canonical_bytes(input: &IntegrationPairSpecV1) -> Result<Vec<u8>, ManifestError> {
    let mut value = serde_json::json!({"domain": NATIVE_LAUNCH_DOMAIN,
        "schema_version": NATIVE_LAUNCH_SCHEMA_VERSION, "execution_ready": false, "input": input});
    value.sort_all_objects();
    let bytes = serde_json::to_vec(&value).map_err(serialization_error)?;
    require(
        bytes.len() <= MAX_NATIVE_LAUNCH_JSON_BYTES,
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

fn validate_budgets(input: &IntegrationPairSpecV1, bytes: u64) -> Result<(), ManifestError> {
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
    require(
        input.clock.movetime_ms > 0 && input.clock.movetime_ms <= timeouts.runtime_ms,
        "clock.movetime_ms",
        "NativeMovetimeBudget",
        "positive per-move milliseconds must fit the independent pair wall limit",
    )?;
    let budget = input.budget;
    require(
        budget.max_input_bytes > 0 && bytes <= budget.max_input_bytes,
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
        (3..=16).contains(&budget.max_child_processes)
            && (1..=128).contains(&budget.max_runtime_files)
            && (2..=8).contains(&budget.max_runtime_depth),
        "budget",
        "NativeResourceBudget",
        "requires 3..16 children, 1..128 runtime files and runtime depth 2..8",
    )?;
    let total = bytes
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
        "unique inputs plus retained outputs and native runtime tree must fit the aggregate artifact ceiling",
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
        (max_plies as usize - opening.moves.len()).is_multiple_of(2),
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
        input.len() <= MAX_NATIVE_LAUNCH_JSON_BYTES,
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
