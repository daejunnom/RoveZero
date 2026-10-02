//! The single selected Maia asset, loaded once from bounded, hashed bytes.
//!
//! This manifest is C's export provenance, not the shared model registry or a
//! general ONNX loader. Original GPL weights and converted files stay external.

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
    pub fn validate(&self) -> Result<(), BackendError> {
        if self.schema != 1
            || self.source_gzip_sha256 != SOURCE_GZIP_SHA256
            || self.source_protobuf_sha256 != SOURCE_PROTOBUF_SHA256
            || self.converter_commit != CONVERTER_COMMIT
            || self.opset != 17
            || self.dtype != "float32"
            || self.input_name != INPUT_NAME
            || self.policy_name != POLICY_NAME
            || self.wdl_name != WDL_NAME
            || self.weights_license != "GPL-3.0"
            || self.redistribution_ready
        {
            return Err(BackendError::new(
                K::UnsupportedModel,
                S::Asset,
                "only the selected Maia FP32 export profile is supported",
            ));
        }
        parse_sha256(&self.onnx_sha256)?;
        parse_sha256(&self.converter_binary_sha256)?;
        if self.onnx_bytes == 0 || self.onnx_bytes > MAX_ONNX_BYTES {
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
pub struct MaiaAsset {
    manifest: ExportManifest,
    manifest_digest: [u8; 32],
    onnx: Vec<u8>,
}

impl MaiaAsset {
    /// Checks the original compressed AND decompressed identity, then verifies
    /// the exact ONNX bytes passed to ORT. No verify-path/reopen-path race.
    pub fn load(original: &Path, onnx: &Path, manifest: &Path) -> Result<Self, BackendError> {
        let manifest_bytes = read_bounded(manifest, 64 * 1024)?;
        let manifest: ExportManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|_| BackendError::new(K::InvalidInput, S::Asset, "invalid export manifest"))?;
        manifest.validate()?;
        let source = read_bounded(original, SOURCE_GZIP_BYTES)?;
        check_bytes(&source, SOURCE_GZIP_BYTES, SOURCE_GZIP_SHA256)?;
        let mut decoded = Vec::new();
        GzDecoder::new(source.as_slice())
            .take((SOURCE_PROTOBUF_BYTES + 1) as u64)
            .read_to_end(&mut decoded)
            .map_err(|_| BackendError::new(K::InvalidInput, S::Asset, "invalid gzip source"))?;
        check_bytes(&decoded, SOURCE_PROTOBUF_BYTES, SOURCE_PROTOBUF_SHA256)?;
        let onnx = read_bounded(onnx, manifest.onnx_bytes)?;
        check_bytes(&onnx, manifest.onnx_bytes, &manifest.onnx_sha256)?;
        Ok(Self {
            manifest,
            manifest_digest: sha256(&manifest_bytes),
            onnx,
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
