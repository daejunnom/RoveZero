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
fn bt4_is_an_independent_pinned_profile_with_its_own_size_and_rights_status() {
    let mut bt4 = manifest();
    bt4.source_gzip_sha256 = asset::BT4_GZIP_SHA256.into();
    bt4.source_protobuf_sha256 = asset::BT4_PROTOBUF_SHA256.into();
    bt4.onnx_bytes = 741_143_425;
    bt4.weights_license = asset::BT4_LICENSE_STATUS.into();
    bt4.validate().unwrap();
    assert_eq!(bt4.profile().unwrap(), asset::AssetProfile::Bt4It332);
    assert!(bt4.profile().unwrap().has_moves_left_head());
    // Neither a mixed source identity nor a code-license guess authorizes
    // the BT4 asset. Maia keeps its original 16 MiB export bound.
    for change in [
        |m: &mut ExportManifest| m.source_protobuf_sha256 = asset::SOURCE_PROTOBUF_SHA256.into(),
        |m: &mut ExportManifest| m.weights_license = "GPL-3.0".into(),
        |m: &mut ExportManifest| m.redistribution_ready = true,
    ] {
        let mut changed = bt4.clone();
        change(&mut changed);
        assert_eq!(
            changed.validate().unwrap_err().kind,
            FailureKind::UnsupportedModel
        );
    }
    bt4.onnx_bytes = asset::BT4_MAX_ONNX_BYTES + 1;
    assert_eq!(
        bt4.validate().unwrap_err().kind,
        FailureKind::ResourceExhausted
    );
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
