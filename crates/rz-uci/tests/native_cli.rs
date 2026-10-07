//! Provider and asset arguments must be rejected safely before UCI starts.
//! Build-capability registration and argument admission do not load models or prove inference.

use std::{
    io::{self, Read},
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc::{self, Receiver},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const WAIT: Duration = Duration::from_secs(5);
const MAX_OUTPUT: usize = 4096;

struct Attempt {
    child: Child,
    readers: Vec<JoinHandle<()>>,
}

impl Drop for Attempt {
    fn drop(&mut self) {
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
        let until = Instant::now() + Duration::from_secs(1);
        while self.readers.iter().any(|reader| !reader.is_finished()) && Instant::now() < until {
            thread::sleep(Duration::from_millis(10));
        }
        for reader in self.readers.drain(..) {
            if reader.is_finished() {
                let _ = reader.join();
            }
        }
    }
}

fn capture(reader: impl Read + Send + 'static) -> (Receiver<io::Result<Vec<u8>>>, JoinHandle<()>) {
    let (sender, receiver) = mpsc::sync_channel(1);
    let producer = thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = reader
            .take((MAX_OUTPUT + 1) as u64)
            .read_to_end(&mut bytes)
            .map(|_| bytes);
        let _ = sender.send(result);
    });
    (receiver, producer)
}

fn rejected(arguments: &[&str]) -> (ExitStatus, String, String) {
    let child = Command::new(env!("CARGO_BIN_EXE_rz-uci"))
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the Cargo-provided rz-uci binary");
    let mut attempt = Attempt {
        child,
        readers: Vec::new(),
    };
    let (stdout, output_reader) = capture(attempt.child.stdout.take().expect("piped stdout"));
    let (stderr, diagnostic_reader) = capture(attempt.child.stderr.take().expect("piped stderr"));
    attempt.readers = vec![output_reader, diagnostic_reader];
    let until = Instant::now() + WAIT;
    let status = loop {
        if let Some(status) = attempt.child.try_wait().expect("poll child exit") {
            break status;
        }
        assert!(
            Instant::now() < until,
            "unsupported native provider did not exit"
        );
        thread::sleep(Duration::from_millis(10));
    };
    let output = stdout
        .recv_timeout(WAIT)
        .expect("stdout producer reaches finite EOF")
        .expect("read child stdout");
    let diagnostic = stderr
        .recv_timeout(WAIT)
        .expect("stderr producer reaches finite EOF")
        .expect("read child stderr");
    assert!(
        output.len() <= MAX_OUTPUT,
        "unbounded rejected-provider stdout"
    );
    assert!(
        diagnostic.len() <= MAX_OUTPUT,
        "unbounded rejected-provider stderr"
    );
    (
        status,
        String::from_utf8(output).expect("ASCII protocol"),
        String::from_utf8(diagnostic).expect("public UTF-8 startup error"),
    )
}

#[test]
fn build_capability_json_exactly_registers_this_feature_and_target_build_without_loading() {
    let (status, output, diagnostic) = rejected(&["--build-capabilities-json"]);
    assert!(
        status.success(),
        "capability registration failed: {diagnostic}"
    );
    assert!(diagnostic.is_empty());
    let expected = format!(
        "{{\"schema\":\"rz-uci-build-capability/1\",\"target_os\":\"{}\",\"target_arch\":\"{}\",\"package_version\":\"{}\",\"compile_features\":{{\"onnx_cpu\":{},\"onnx_cuda\":{},\"experimental_io_binding\":{}}},\"cuda_path_compiled\":{},\"native_runtime_loaded\":false,\"native_model_loaded\":false}}\n",
        std::env::consts::OS,
        std::env::consts::ARCH,
        env!("CARGO_PKG_VERSION"),
        cfg!(feature = "onnx-cpu"),
        cfg!(feature = "onnx-cuda"),
        cfg!(feature = "experimental-io-binding"),
        cfg!(all(feature = "onnx-cuda", target_os = "linux")),
    );
    assert_eq!(output, expected);
}

#[test]
fn build_capability_mode_rejects_all_additional_arguments_without_provider_or_mock_fallback() {
    for arguments in [
        vec!["--build-capabilities-json", "--cpu-mock"],
        vec!["--build-capabilities-json", "--onnx-cpu"],
        vec!["--onnx-cuda", "--build-capabilities-json"],
        vec!["--search=pals", "--build-capabilities-json"],
        vec!["--build-capabilities-json", "--build-capabilities-json"],
        vec![
            "--build-capabilities-json",
            "--pals-export-manifest=private-asset-marker/manifest.json",
        ],
    ] {
        let (status, output, diagnostic) = rejected(&arguments);
        assert_eq!(status.code(), Some(2));
        assert!(
            output.is_empty(),
            "mixed registration mode emitted engine output"
        );
        assert!(diagnostic.contains("must be the only argument"));
        assert!(diagnostic.contains("no runtime, model or engine was started"));
        assert!(!diagnostic.contains("private-asset-marker"));
        assert!(diagnostic.len() <= 512);
    }
}

#[test]
#[cfg(not(feature = "onnx-cpu"))]
fn disabled_native_cpu_provider_is_explicitly_rejected_before_protocol() {
    let (status, protocol, diagnostic) = rejected(&["--onnx-cpu"]);
    assert_eq!(status.code(), Some(2));
    assert!(
        protocol.is_empty(),
        "unsupported provider emitted UCI: {protocol}"
    );
    assert!(diagnostic.contains("unsupported feature") && diagnostic.contains("onnx-cpu"));
    assert!(diagnostic.len() <= 512, "startup error must remain bounded");
}

#[test]
#[cfg(not(feature = "onnx-cpu"))]
fn native_request_does_not_fall_back_to_mock_or_echo_private_asset_arguments() {
    for arguments in [
        [
            "--cpu-mock",
            "--onnx-cpu",
            "--weights=private-asset-marker/model.pb.gz",
        ],
        [
            "--onnx-cpu",
            "--cpu-mock",
            "--weights=private-asset-marker/model.pb.gz",
        ],
    ] {
        let (status, protocol, diagnostic) = rejected(&arguments);
        assert_eq!(status.code(), Some(2));
        assert!(
            protocol.is_empty(),
            "native admission silently selected mock: {protocol}"
        );
        assert!(diagnostic.contains("unsupported feature"));
        assert!(
            !diagnostic.contains("private-asset-marker"),
            "asset argument leaked: {diagnostic}"
        );
        assert!(diagnostic.len() <= 512, "startup error must remain bounded");
    }
}

#[test]
fn native_asset_without_provider_is_rejected_without_private_value_output() {
    let (status, protocol, diagnostic) =
        rejected(&["--source-weights=private-asset-marker/model.pb.gz"]);
    assert_eq!(status.code(), Some(2));
    assert!(protocol.is_empty(), "rejected native asset started UCI");
    assert!(diagnostic.contains("unsupported argument"));
    assert!(diagnostic.contains("supported flags"));
    assert!(!diagnostic.contains("private-asset-marker"));
    assert!(!diagnostic.contains("model.pb.gz"));
    assert!(diagnostic.len() <= 512, "startup error must remain bounded");
}

#[test]
fn native_runtime_asset_mixed_with_mock_is_rejected_without_private_value_output() {
    let (status, protocol, diagnostic) = rejected(&[
        "--cpu-mock",
        "--ort-library=private-runtime-marker/vendor-library",
    ]);
    assert_eq!(status.code(), Some(2));
    assert!(
        protocol.is_empty(),
        "rejected runtime asset started mock UCI"
    );
    assert!(diagnostic.contains("unsupported argument"));
    assert!(diagnostic.contains("supported flags"));
    assert!(!diagnostic.contains("private-runtime-marker"));
    assert!(!diagnostic.contains("vendor-library"));
    assert!(diagnostic.len() <= 512, "startup error must remain bounded");
}
