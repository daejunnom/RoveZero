use rz_eval::asset::{self, ExportManifest};
use rz_eval::error::FailureKind;

fn manifest() -> ExportManifest {
    ExportManifest {
        schema: 1,
        source_gzip_sha256: asset::SOURCE_GZIP_SHA256.into(),
        source_protobuf_sha256: asset::SOURCE_PROTOBUF_SHA256.into(),
        onnx_sha256: "12".repeat(32),
        onnx_bytes: 1024,
        converter_commit: asset::CONVERTER_COMMIT.into(),
        converter_binary_sha256: "ab".repeat(32),
        command: vec!["lc0".into(), "leela2onnx".into()],
        opset: 17,
        dtype: "float32".into(),
        input_name: asset::INPUT_NAME.into(),
        policy_name: asset::POLICY_NAME.into(),
        wdl_name: asset::WDL_NAME.into(),
        weights_license: "GPL-3.0".into(),
        redistribution_ready: false,
    }
}

#[test]
fn rejects_other_weights_precision_heads_and_unreviewed_redistribution() {
    let original = manifest();
    original.validate().unwrap();
    for change in [
        |m: &mut ExportManifest| m.source_gzip_sha256 = "00".repeat(32),
        |m: &mut ExportManifest| m.source_protobuf_sha256 = "00".repeat(32),
        |m: &mut ExportManifest| m.dtype = "float16".into(),
        |m: &mut ExportManifest| m.opset = 18,
        |m: &mut ExportManifest| m.wdl_name = "/value".into(),
        |m: &mut ExportManifest| m.redistribution_ready = true,
    ] {
        let mut changed = original.clone();
        change(&mut changed);
        assert_eq!(
            changed.validate().unwrap_err().kind,
            FailureKind::UnsupportedModel
        );
    }
    let mut too_large = original;
    too_large.onnx_bytes = asset::MAX_ONNX_BYTES + 1;
    assert_eq!(
        too_large.validate().unwrap_err().kind,
        FailureKind::ResourceExhausted
    );
}

#[test]
fn strict_digest_and_manifest_parsing() {
    assert_eq!(
        asset::hex_sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    for bad in [
        "a".repeat(63),
        "G0".repeat(32),
        "AB".repeat(32),
        "é".repeat(32),
    ] {
        assert!(asset::parse_sha256(&bad).is_err());
    }
    let mut json = serde_json::to_value(manifest()).unwrap();
    json["silently_ignore_me"] = true.into();
    assert!(serde_json::from_value::<ExportManifest>(json).is_err());
}

#[test]
fn file_read_has_a_real_byte_limit() {
    let path = std::env::temp_dir().join(format!("rz-c-asset-limit-{}", std::process::id()));
    std::fs::write(&path, b"bounded data").unwrap();
    assert_eq!(asset::read_bounded(&path, 12).unwrap(), b"bounded data");
    assert_eq!(
        asset::read_bounded(&path, 11).unwrap_err().kind,
        FailureKind::ResourceExhausted
    );
    assert!(asset::read_bounded(&path, usize::MAX).is_err());
    std::fs::remove_file(path).unwrap();
}
