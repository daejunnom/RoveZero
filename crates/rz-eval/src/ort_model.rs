//! Opt-in, derived ORT serialization. Original weights and export identity stay
//! unchanged. The owned flatbuffer must outlive its native session and tensors.
use crate::{
    asset::{self, AssetMetadata, AssetProfile},
    error::{BackendError, FailureKind as K, FailureStage as S},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OrtModelManifest {
    pub schema: u32,
    pub source_onnx_sha256: String,
    pub source_export_manifest_sha256: String,
    pub ort_sha256: String,
    pub ort_bytes: usize,
    pub runtime_core_sha256: String,
    pub runtime_bundle_sha256: String,
    pub runtime_version: String,
    pub optimization_level: u32,
    pub provider: String,
    pub precision: String,
    pub tf32: bool,
    pub redistribution_ready: bool,
}
impl OrtModelManifest {
    pub fn validate(&self, source: &AssetMetadata) -> Result<(), BackendError> {
        if source.profile() != AssetProfile::Bt4It332
            || self.schema != 1
            || self.source_onnx_sha256 != source.manifest().onnx_sha256
            || asset::parse_sha256(&self.source_export_manifest_sha256)? != source.manifest_digest()
            || self.ort_bytes < 8
            || self.ort_bytes > source.profile().max_onnx_bytes()
            || self.runtime_version != "1.22.0"
            || self.optimization_level != 1
            || self.provider != "cuda"
            || self.precision != "fp32"
            || self.tf32
            || self.redistribution_ready
        {
            return Err(BackendError::new(
                K::UnsupportedModel,
                S::Asset,
                "derived ORT profile or original export identity differs",
            ));
        }
        for hash in [
            &self.ort_sha256,
            &self.runtime_core_sha256,
            &self.runtime_bundle_sha256,
        ] {
            asset::parse_sha256(hash)?;
        }
        Ok(())
    }
}

pub struct OwnedOrtAsset {
    pub(crate) source: AssetMetadata,
    pub(crate) bytes: Vec<u8>,
    pub(crate) manifest: OrtModelManifest,
    pub(crate) manifest_digest: [u8; 32],
}
impl OwnedOrtAsset {
    /// Verify original provenance without allocating another full ONNX model.
    /// Inference consumes only the exact, hashed Rust-owned ORT flatbuffer.
    pub fn load(
        source: &Path,
        onnx: &Path,
        export: &Path,
        ort: &Path,
        manifest: &Path,
    ) -> Result<Self, BackendError> {
        let source = AssetMetadata::load_source(source, export)?;
        verify_file(
            onnx,
            source.manifest().onnx_bytes,
            &source.manifest().onnx_sha256,
        )?;
        let json = asset::read_bounded(manifest, 64 * 1024)?;
        let manifest: OrtModelManifest = serde_json::from_slice(&json).map_err(|_| {
            BackendError::new(K::InvalidInput, S::Asset, "invalid derived ORT manifest")
        })?;
        manifest.validate(&source)?;
        let bytes = asset::read_bounded(ort, manifest.ort_bytes)?;
        if bytes.len() != manifest.ort_bytes
            || asset::sha256(&bytes) != asset::parse_sha256(&manifest.ort_sha256)?
            || bytes.get(4..8) != Some(b"ORTM".as_slice())
        {
            return Err(BackendError::new(
                K::IdentityMismatch,
                S::Asset,
                "derived ORT bytes, length or flatbuffer marker differs",
            ));
        }
        Ok(Self {
            source,
            bytes,
            manifest,
            manifest_digest: asset::sha256(&json),
        })
    }
    pub fn metadata(&self) -> &AssetMetadata {
        &self.source
    }
    pub fn manifest(&self) -> &OrtModelManifest {
        &self.manifest
    }
    pub fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }
}

/// One bounded held descriptor; no original-path reopening for inference.
fn verify_file(path: &Path, len: usize, expected: &str) -> Result<(), BackendError> {
    let file = File::open(path)
        .map_err(|_| BackendError::new(K::Io, S::Asset, "cannot open original ONNX"))?;
    let metadata = file
        .metadata()
        .map_err(|_| BackendError::new(K::Io, S::Asset, "cannot inspect original ONNX"))?;
    if !metadata.is_file() || metadata.len() > len as u64 {
        return Err(BackendError::new(
            K::ResourceExhausted,
            S::Asset,
            "original ONNX is not a bounded regular file",
        ));
    }
    let limit = len.checked_add(1).ok_or_else(|| {
        BackendError::new(K::ResourceExhausted, S::Asset, "ONNX file limit overflow")
    })?;
    let mut reader = file.take(limit as u64);
    let mut digest = Sha256::new();
    let mut count = 0;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = reader
            .read(&mut buffer)
            .map_err(|_| BackendError::new(K::Io, S::Asset, "original ONNX read failed"))?;
        if n == 0 {
            break;
        }
        count += n;
        if count > len {
            return Err(BackendError::new(
                K::ResourceExhausted,
                S::Asset,
                "original ONNX grew",
            ));
        }
        digest.update(&buffer[..n]);
    }
    let actual: [u8; 32] = digest.finalize().into();
    if count != len || actual != asset::parse_sha256(expected)? {
        return Err(BackendError::new(
            K::IdentityMismatch,
            S::Asset,
            "original ONNX length or SHA256 differs",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_stream_rejects_growth_truncation_and_digest_change() {
        let path = std::env::temp_dir().join(format!("rz-ort-original-{}", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        let digest = format!("{:x}", Sha256::digest(b"abc"));
        verify_file(&path, 3, &digest).unwrap();
        assert_eq!(
            verify_file(&path, 2, &digest).unwrap_err().kind,
            K::ResourceExhausted
        );
        assert_eq!(
            verify_file(&path, 4, &digest).unwrap_err().kind,
            K::IdentityMismatch
        );
        assert_eq!(
            verify_file(&path, 3, &"00".repeat(32)).unwrap_err().kind,
            K::IdentityMismatch
        );
        std::fs::remove_file(path).unwrap();
    }
}
