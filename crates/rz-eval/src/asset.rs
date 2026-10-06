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
    metadata: AssetMetadata,
    onnx: Vec<u8>,
}

/// Verified identity retained after the serialized model buffer is consumed.
/// Only the bounded asset loader can construct this provenance.
#[derive(Debug)]
pub struct AssetMetadata {
    manifest: ExportManifest,
    manifest_digest: [u8; 32],
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
        verify_source_stream(
            original,
            gzip_bytes,
            profile.gzip_sha256(),
            protobuf_bytes,
            profile.protobuf_sha256(),
        )?;
        #[cfg(not(feature = "experimental-onnx-read-hash"))]
        let onnx = {
            let bytes = read_bounded(onnx, manifest.onnx_bytes)?;
            check_bytes(&bytes, manifest.onnx_bytes, &manifest.onnx_sha256)?;
            bytes
        };
        #[cfg(feature = "experimental-onnx-read-hash")]
        let onnx = read_verified_onnx(onnx, manifest.onnx_bytes, &manifest.onnx_sha256)?;
        Ok(Self {
            metadata: AssetMetadata {
                manifest,
                manifest_digest: sha256(&manifest_bytes),
                profile,
            },
            onnx,
        })
    }

    pub fn manifest(&self) -> &ExportManifest {
        self.metadata.manifest()
    }
    pub fn manifest_digest(&self) -> [u8; 32] {
        self.metadata.manifest_digest()
    }
    pub fn onnx_bytes(&self) -> &[u8] {
        &self.onnx
    }
    pub fn profile(&self) -> AssetProfile {
        self.metadata.profile()
    }
    pub fn metadata(&self) -> &AssetMetadata {
        &self.metadata
    }
    #[cfg(feature = "onnx")]
    pub(crate) fn into_parts(self) -> (AssetMetadata, Vec<u8>) {
        (self.metadata, self.onnx)
    }
}

impl AssetMetadata {
    pub fn manifest(&self) -> &ExportManifest {
        &self.manifest
    }
    pub fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }
    pub fn profile(&self) -> AssetProfile {
        self.profile
    }
}

struct HashedReader<R> {
    reader: R,
    digest: Sha256,
    bytes: usize,
    io_failed: bool,
}

impl<R: Read> Read for HashedReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        match self.reader.read(buffer) {
            Ok(size) => {
                self.bytes += size; // The inner Take bounds this by the profile.
                self.digest.update(&buffer[..size]);
                Ok(size)
            }
            Err(error) => {
                self.io_failed = true;
                Err(error)
            }
        }
    }
}

/// Hash compressed and decoded bytes from one held file, without a whole gzip
/// allocation or reopening a verified path. Preserve compressed-identity error
/// precedence, including malformed/trailing data and bounded decoded expansion.
fn verify_source_stream(
    path: &Path,
    gzip_bytes: usize,
    gzip_digest: &str,
    decoded_bytes: usize,
    decoded_digest: &str,
) -> Result<(), BackendError> {
    let compressed_limit = gzip_bytes.checked_add(1).ok_or_else(|| {
        BackendError::new(K::ResourceExhausted, S::Asset, "source limit overflow")
    })?;
    let decoded_limit = decoded_bytes.checked_add(1).ok_or_else(|| {
        BackendError::new(K::ResourceExhausted, S::Asset, "decoded limit overflow")
    })?;
    let file =
        File::open(path).map_err(|_| BackendError::new(K::Io, S::Asset, "cannot open asset"))?;
    let metadata = file
        .metadata()
        .map_err(|_| BackendError::new(K::Io, S::Asset, "cannot inspect asset"))?;
    if !metadata.is_file() || metadata.len() > gzip_bytes as u64 {
        return Err(BackendError::new(
            K::ResourceExhausted,
            S::Asset,
            "asset is not a bounded regular file",
        ));
    }
    let compressed = HashedReader {
        reader: file.take(compressed_limit as u64),
        digest: Sha256::new(),
        bytes: 0,
        io_failed: false,
    };
    let mut decoded = GzDecoder::new(compressed);
    let mut limited = (&mut decoded).take(decoded_limit as u64);
    let mut digest = Sha256::new();
    let mut bytes = 0;
    let mut buffer = [0_u8; 64 * 1024];
    let decoded_error = loop {
        match limited.read(&mut buffer) {
            Ok(0) => break None,
            Ok(size) => {
                bytes += size;
                digest.update(&buffer[..size]);
            }
            Err(error) => break Some(error),
        }
    };
    // GzDecoder can stop at a member boundary or read ahead. Recover its same
    // hashing reader and drain the remaining compressed bytes once, so trailing
    // bytes and exact file length remain covered by the original source pin.
    let mut compressed = decoded.into_inner();
    loop {
        let size = compressed
            .read(&mut buffer)
            .map_err(|_| BackendError::new(K::Io, S::Asset, "asset read failed"))?;
        if size == 0 {
            break;
        }
    }
    if compressed.io_failed {
        return Err(BackendError::new(K::Io, S::Asset, "asset read failed"));
    }
    if compressed.bytes > gzip_bytes {
        return Err(BackendError::new(
            K::ResourceExhausted,
            S::Asset,
            "asset grew beyond limit",
        ));
    }
    if compressed.bytes != gzip_bytes
        || <[u8; 32]>::from(compressed.digest.finalize()) != parse_sha256(gzip_digest)?
    {
        return Err(BackendError::new(
            K::IdentityMismatch,
            S::Asset,
            "asset length or SHA-256 differs",
        ));
    }
    if decoded_error.is_some() {
        return Err(BackendError::new(
            K::InvalidInput,
            S::Asset,
            "invalid gzip source",
        ));
    }
    if bytes != decoded_bytes
        || <[u8; 32]>::from(digest.finalize()) != parse_sha256(decoded_digest)?
    {
        return Err(BackendError::new(
            K::IdentityMismatch,
            S::Asset,
            "decoded source length or SHA-256 differs",
        ));
    }
    Ok(())
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
    read_bounded_using(path, max_bytes, |file, limit, bytes| {
        file.take(limit).read_to_end(bytes).map(|_| ())
    })
    .map(|(bytes, ())| bytes)
}

// Both paths keep one held file, exact metadata-based reservation, the same
// limit+1 growth probe and error precedence. The reader writes directly into
// the owned Vec; the hashing path needs no second model-sized allocation.
fn read_bounded_using<T>(
    path: &Path,
    max_bytes: usize,
    read: impl FnOnce(File, u64, &mut Vec<u8>) -> std::io::Result<T>,
) -> Result<(Vec<u8>, T), BackendError> {
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
    let result = read(file, limit as u64, &mut bytes)
        .map_err(|_| BackendError::new(K::Io, S::Asset, "asset read failed"))?;
    if bytes.len() > max_bytes {
        return Err(BackendError::new(
            K::ResourceExhausted,
            S::Asset,
            "asset grew beyond limit",
        ));
    }
    Ok((bytes, result))
}

#[cfg(any(feature = "experimental-onnx-read-hash", test))]
fn read_hashed_onnx(file: File, limit: u64, bytes: &mut Vec<u8>) -> std::io::Result<[u8; 32]> {
    let mut reader = HashedReader {
        reader: file.take(limit),
        digest: Sha256::new(),
        bytes: 0,
        io_failed: false,
    };
    reader.read_to_end(bytes)?;
    Ok(reader.digest.finalize().into())
}

#[cfg(any(feature = "experimental-onnx-read-hash", test))]
fn read_verified_onnx(path: &Path, len: usize, digest: &str) -> Result<Vec<u8>, BackendError> {
    let (bytes, actual_digest) = read_bounded_using(path, len, read_hashed_onnx)?;
    if bytes.len() != len || actual_digest != parse_sha256(digest)? {
        return Err(BackendError::new(
            K::IdentityMismatch,
            S::Asset,
            "asset length or SHA-256 differs",
        ));
    }
    // ORT still commits these verified, Rust-owned bytes. No verified-path
    // reopen or mutable file mapping is introduced, and post-commit release
    // continues to belong to commit_verified_model.
    Ok(bytes)
}

#[cfg(any(not(feature = "experimental-onnx-read-hash"), test))]
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

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use flate2::{write::GzEncoder, Compression};
    use std::io::Write;

    #[test]
    fn onnx_stream_hash_matches_owned_bytes_and_identity_error_precedence() {
        let path = std::env::temp_dir().join(format!("rz-onnx-read-hash-{}", std::process::id()));
        for len in [0, 1, 65_535, 65_536, 65_537, 131_113] {
            let input: Vec<u8> = (0..len).map(|n| (n % 251) as u8).collect();
            std::fs::write(&path, &input).unwrap();
            let baseline = read_bounded(&path, len).unwrap();
            let digest = hex_sha256(&input);
            check_bytes(&baseline, len, &digest).unwrap();
            assert_eq!(read_verified_onnx(&path, len, &digest).unwrap(), baseline);
            for (expected_len, expected_hash) in [
                (len, "00".repeat(32)),
                (len, "bad".into()),
                (len + 1, digest.clone()),
                (len + 1, "bad".into()),
            ] {
                let old = check_bytes(&baseline, expected_len, &expected_hash).unwrap_err();
                let new = read_verified_onnx(&path, expected_len, &expected_hash).unwrap_err();
                assert_eq!(new, old);
            }
        }
        let input = b"abc";
        std::fs::write(&path, input).unwrap();
        let bytes = read_verified_onnx(&path, input.len(), &hex_sha256(input)).unwrap();
        std::fs::write(&path, b"mutated after verification").unwrap();
        assert_eq!(bytes, input);
        assert_eq!(
            read_verified_onnx(&path, input.len(), &hex_sha256(input))
                .unwrap_err()
                .kind,
            K::ResourceExhausted
        );
        assert_eq!(
            read_verified_onnx(&path, usize::MAX, "bad").unwrap_err(),
            read_bounded(&path, usize::MAX).unwrap_err()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn onnx_hash_read_preserves_growth_truncation_and_io_failure() {
        let path = std::env::temp_dir().join(format!("rz-onnx-grow-hash-{}", std::process::id()));
        for hashed in [false, true] {
            std::fs::write(&path, b"abc").unwrap();
            let grown = read_bounded_using(&path, 3, |file, limit, bytes| {
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(&path)?
                    .write_all(b"d")?;
                if hashed {
                    read_hashed_onnx(file, limit, bytes).map(|_| ())
                } else {
                    file.take(limit).read_to_end(bytes).map(|_| ())
                }
            });
            assert_eq!(grown.unwrap_err().kind, K::ResourceExhausted);
            std::fs::write(&path, b"abc").unwrap();
            let (bytes, digest) = read_bounded_using(&path, 3, |file, limit, bytes| {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(&path)?
                    .set_len(1)?;
                read_hashed_onnx(file, limit, bytes)
            })
            .unwrap();
            assert_eq!(bytes, b"a");
            assert_eq!(digest, sha256(b"a"));
            assert_eq!(
                check_bytes(&bytes, 3, &hex_sha256(b"abc"))
                    .unwrap_err()
                    .kind,
                K::IdentityMismatch
            );
        }
        let error = read_bounded_using(&path, 3, |_file, _limit, _bytes| {
            Err::<(), _>(std::io::Error::other("injected read error"))
        })
        .unwrap_err();
        assert_eq!(
            error,
            BackendError::new(K::Io, S::Asset, "asset read failed")
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn streamed_source_covers_exact_compressed_and_decoded_bytes_and_error_precedence() {
        let path = std::env::temp_dir().join(format!("rz-stream-source-{}", std::process::id()));
        let decoded: Vec<u8> = (0..131_113).map(|n| (n % 251) as u8).collect();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&decoded).unwrap();
        let mut source = encoder.finish().unwrap();
        // Old identity checking hashed all trailing compressed bytes as well.
        source.extend_from_slice(b"bound trailing bytes");
        std::fs::write(&path, &source).unwrap();
        let source_digest = hex_sha256(&source);
        let decoded_digest = hex_sha256(&decoded);
        let verify = |size, hash: &str, output_size, output_hash: &str| {
            verify_source_stream(&path, size, hash, output_size, output_hash)
        };
        verify(source.len(), &source_digest, decoded.len(), &decoded_digest).unwrap();
        assert_eq!(
            verify(
                source.len() - 1,
                &source_digest,
                decoded.len(),
                &decoded_digest
            )
            .unwrap_err()
            .kind,
            K::ResourceExhausted
        );
        assert_eq!(
            verify(
                source.len() + 1,
                &source_digest,
                decoded.len(),
                &decoded_digest
            )
            .unwrap_err()
            .kind,
            K::IdentityMismatch
        );
        assert_eq!(
            verify(
                source.len(),
                &"00".repeat(32),
                decoded.len(),
                &decoded_digest
            )
            .unwrap_err()
            .kind,
            K::IdentityMismatch
        );
        for output_size in [decoded.len() - 1, decoded.len() + 1] {
            assert_eq!(
                verify(source.len(), &source_digest, output_size, &decoded_digest)
                    .unwrap_err()
                    .kind,
                K::IdentityMismatch
            );
        }
        assert_eq!(
            verify(
                source.len(),
                &source_digest,
                decoded.len(),
                &"00".repeat(32)
            )
            .unwrap_err()
            .kind,
            K::IdentityMismatch
        );
        source[0] ^= 0xff;
        std::fs::write(&path, &source).unwrap();
        // A corrupt header still reports the changed compressed pin first.
        assert_eq!(
            verify(source.len(), &source_digest, decoded.len(), &decoded_digest)
                .unwrap_err()
                .kind,
            K::IdentityMismatch
        );
        assert_eq!(
            verify(
                source.len(),
                &hex_sha256(&source),
                decoded.len(),
                &decoded_digest
            )
            .unwrap_err()
            .kind,
            K::InvalidInput
        );
        std::fs::remove_file(path).unwrap();
    }
}
