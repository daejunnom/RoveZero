//! Selected LC0 assets, loaded once from bounded, hashed bytes.
//!
//! This manifest is C's export provenance, not the shared model registry or a
//! general ONNX loader. Original weights and converted files stay external.

use crate::error::{BackendError, FailureKind as K, FailureStage as S};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const SOURCE_GZIP_SHA256: &str =
    "e2f565f42d7cd9f122557e6dc4eb84e5bbaedceda1d404dc485d3611c7c97a12";
pub const SOURCE_PROTOBUF_SHA256: &str =
    "e8fe5a7f35594d4190c440a2716fda57ba9d071b26175d509476be7a4fda052a";
pub const SOURCE_GZIP_BYTES: usize = 1_262_607;
pub const SOURCE_PROTOBUF_BYTES: usize = 1_738_564;
pub const CONVERTER_COMMIT: &str = "fd71a2d921b689c5f479d3227c3806c8e272d9c5";
pub const MAX_ONNX_BYTES: usize = 16 * 1024 * 1024;
pub const INPUT_NAME: &str = "/input/planes";
pub const POLICY_NAME: &str = "/output/policy";
pub const WDL_NAME: &str = "/output/wdl";
pub const MLH_NAME: &str = "/output/mlh";
pub const BT4_GZIP_SHA256: &str =
    "e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618";
pub const BT4_PROTOBUF_SHA256: &str =
    "d6e4bbf289bea1fe312b7a0286106aeb713760b604c932ef8cdeebf16a23f36c";
pub const BT4_GZIP_BYTES: usize = 382_645_315;
pub const BT4_PROTOBUF_BYTES: usize = 382_616_086;
pub const BT4_MAX_ONNX_BYTES: usize = 768 * 1024 * 1024;
pub const BT4_LICENSE_STATUS: &str = "UNVERIFIED-local-research-only";

/// A different source is a different model identity, never a relaxed Maia pin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssetProfile {
    Maia1900,
    Bt4It332,
}
impl AssetProfile {
    pub fn gzip_sha256(self) -> &'static str {
        match self {
            Self::Maia1900 => SOURCE_GZIP_SHA256,
            Self::Bt4It332 => BT4_GZIP_SHA256,
        }
    }
    pub fn protobuf_sha256(self) -> &'static str {
        match self {
            Self::Maia1900 => SOURCE_PROTOBUF_SHA256,
            Self::Bt4It332 => BT4_PROTOBUF_SHA256,
        }
    }
    fn source_sizes(self) -> (usize, usize) {
        match self {
            Self::Maia1900 => (SOURCE_GZIP_BYTES, SOURCE_PROTOBUF_BYTES),
            Self::Bt4It332 => (BT4_GZIP_BYTES, BT4_PROTOBUF_BYTES),
        }
    }
    pub fn max_onnx_bytes(self) -> usize {
        match self {
            Self::Maia1900 => MAX_ONNX_BYTES,
            Self::Bt4It332 => BT4_MAX_ONNX_BYTES,
        }
    }
    pub fn has_moves_left_head(self) -> bool {
        self == Self::Bt4It332
    }
    /// ORT's arena declaration is neither total VRAM nor an external hard cap.
    pub fn cuda_arena_bytes(self) -> usize {
        match self {
            Self::Maia1900 => 1024 * 1024 * 1024,
            Self::Bt4It332 => 3 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExportManifest {
    pub schema: u32,
    pub source_gzip_sha256: String,
    pub source_protobuf_sha256: String,
    pub onnx_sha256: String,
    pub onnx_bytes: usize,
    pub converter_commit: String,
    pub converter_binary_sha256: String,
    pub command: Vec<String>,
    pub opset: u32,
    pub dtype: String,
    pub input_name: String,
    pub policy_name: String,
    pub wdl_name: String,
    pub weights_license: String,
    pub redistribution_ready: bool,
}

impl ExportManifest {
    pub fn profile(&self) -> Result<AssetProfile, BackendError> {
        [AssetProfile::Maia1900, AssetProfile::Bt4It332]
            .into_iter()
            .find(|profile| {
                self.source_gzip_sha256 == profile.gzip_sha256()
                    && self.source_protobuf_sha256 == profile.protobuf_sha256()
            })
            .ok_or_else(|| {
                BackendError::new(
                    K::UnsupportedModel,
                    S::Asset,
                    "unknown selected asset identity",
                )
            })
    }
    pub fn validate(&self) -> Result<(), BackendError> {
        let profile = self.profile()?;
        let license = match profile {
            AssetProfile::Maia1900 => "GPL-3.0",
            AssetProfile::Bt4It332 => BT4_LICENSE_STATUS,
        };
        if self.schema != 1
            || self.converter_commit != CONVERTER_COMMIT
            || self.opset != 17
            || self.dtype != "float32"
            || self.input_name != INPUT_NAME
            || self.policy_name != POLICY_NAME
            || self.wdl_name != WDL_NAME
            || self.weights_license != license
            || self.redistribution_ready
        {
            return Err(BackendError::new(
                K::UnsupportedModel,
                S::Asset,
                "only selected LC0 FP32 export profiles are supported",
            ));
        }
        parse_sha256(&self.onnx_sha256)?;
        parse_sha256(&self.converter_binary_sha256)?;
        if self.onnx_bytes == 0 || self.onnx_bytes > profile.max_onnx_bytes() {
            return Err(BackendError::new(
                K::ResourceExhausted,
                S::Asset,
                "ONNX asset size exceeds profile",
            ));
        }
        if self.command.is_empty()
            || self.command.len() > 32
            || self
                .command
                .iter()
                .any(|arg| arg.is_empty() || arg.len() > 4096 || arg.contains('\0'))
        {
            return Err(BackendError::new(
                K::InvalidInput,
                S::Asset,
                "missing or oversized conversion provenance",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct SelectedAsset {
    manifest: ExportManifest,
    manifest_digest: [u8; 32],
    onnx: Vec<u8>,
    profile: AssetProfile,
}

/// Compatibility name for existing consumers of the first selected profile.
pub type MaiaAsset = SelectedAsset;

impl SelectedAsset {
    /// Checks the original compressed AND decompressed identity, then verifies
    /// the exact ONNX bytes passed to ORT. No verify-path/reopen-path race.
    pub fn load(original: &Path, onnx: &Path, manifest: &Path) -> Result<Self, BackendError> {
        let manifest_bytes = read_bounded(manifest, 64 * 1024)?;
        let manifest: ExportManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|_| BackendError::new(K::InvalidInput, S::Asset, "invalid export manifest"))?;
        manifest.validate()?;
        let profile = manifest.profile()?;
        let (gzip_bytes, protobuf_bytes) = profile.source_sizes();
        let source = read_bounded(original, gzip_bytes)?;
        check_bytes(&source, gzip_bytes, profile.gzip_sha256())?;
        // Stream decompression through a fixed buffer. BT4 does not need a
        // second 365 MiB source allocation while verifying the gzip identity.
        let mut decoded = GzDecoder::new(source.as_slice()).take((protobuf_bytes + 1) as u64);
        let mut hash = Sha256::new();
        let mut size = 0usize;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = decoded
                .read(&mut buffer)
                .map_err(|_| BackendError::new(K::InvalidInput, S::Asset, "invalid gzip source"))?;
            if read == 0 {
                break;
            }
            size += read;
            hash.update(&buffer[..read]);
        }
        if size != protobuf_bytes
            || <[u8; 32]>::from(hash.finalize()) != parse_sha256(profile.protobuf_sha256())?
        {
            return Err(BackendError::new(
                K::IdentityMismatch,
                S::Asset,
                "decoded source length or SHA-256 differs",
            ));
        }
        drop(decoded);
        drop(source);
        let onnx = read_bounded(onnx, manifest.onnx_bytes)?;
        check_bytes(&onnx, manifest.onnx_bytes, &manifest.onnx_sha256)?;
        Ok(Self {
            manifest,
            manifest_digest: sha256(&manifest_bytes),
            onnx,
            profile,
        })
    }

    pub fn manifest(&self) -> &ExportManifest {
        &self.manifest
    }
    pub fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }
    pub fn onnx_bytes(&self) -> &[u8] {
        &self.onnx
    }
    pub fn profile(&self) -> AssetProfile {
        self.profile
    }
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub fn hex_sha256(bytes: &[u8]) -> String {
    sha256(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn parse_sha256(text: &str) -> Result<[u8; 32], BackendError> {
    if text.len() != 64
        || !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(BackendError::new(
            K::InvalidInput,
            S::Asset,
            "expected lowercase SHA-256",
        ));
    }
    let mut result = [0; 32];
    for (out, pair) in result.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
        let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
        *out = digit(pair[0]) * 16 + digit(pair[1]);
    }
    Ok(result)
}

pub fn read_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>, BackendError> {
    let limit = max_bytes.checked_add(1).ok_or(BackendError::new(
        K::ResourceExhausted,
        S::Asset,
        "file limit overflow",
    ))?;
    let file =
        File::open(path).map_err(|_| BackendError::new(K::Io, S::Asset, "cannot open asset"))?;
    let metadata = file
        .metadata()
        .map_err(|_| BackendError::new(K::Io, S::Asset, "cannot inspect asset"))?;
    if !metadata.is_file() || metadata.len() > max_bytes as u64 {
        return Err(BackendError::new(
            K::ResourceExhausted,
            S::Asset,
            "asset is not a bounded regular file",
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(metadata.len() as usize)
        .map_err(|_| {
            BackendError::new(K::ResourceExhausted, S::Asset, "asset allocation failed")
        })?;
    file.take(limit as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| BackendError::new(K::Io, S::Asset, "asset read failed"))?;
    if bytes.len() > max_bytes {
        return Err(BackendError::new(
            K::ResourceExhausted,
            S::Asset,
            "asset grew beyond limit",
        ));
    }
    Ok(bytes)
}

fn check_bytes(bytes: &[u8], len: usize, digest: &str) -> Result<(), BackendError> {
    if bytes.len() != len || sha256(bytes) != parse_sha256(digest)? {
        return Err(BackendError::new(
            K::IdentityMismatch,
            S::Asset,
            "asset length or SHA-256 differs",
        ));
    }
    Ok(())
}
