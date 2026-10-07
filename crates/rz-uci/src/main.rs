use rz_contracts::ProcessEpoch;
use rz_uci::search_driver::SearchSessionDriver;
#[cfg(feature = "onnx-cpu")]
use rz_uci::{
    EngineIdentity,
    engine::EvaluatorFactory,
    native_attestation::{ReceiptWriter, StartupReceiptV1, TerminationReceiptV1},
    native_bootstrap::{NativeConfig, NativeCpuFactory, NativeProvider},
    native_profile::{ProfileWriter, ProfiledWrite},
};
use rz_uci::{
    bootstrap::CpuMockFactory,
    engine::{self, EngineProcess, EngineSettings, OwnerRegistry, ProcessClock},
    forward_lines,
};
#[cfg(all(feature = "onnx-cuda", target_os = "linux"))]
use rz_uci::{
    native_bootstrap::NativeCudaFactory,
    native_cuda_attestation::{CudaReceiptWriter, CudaStartupReceiptV1, CudaTerminationReceiptV1},
};
use std::{io, sync::Arc, thread, time::Duration};

fn main() {
    if let Err(error) = run() {
        eprintln!("RoveZero: {error}");
        std::process::exit(2);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments
        .iter()
        .any(|argument| argument == "--build-capabilities-json")
    {
        if arguments.len() != 1 {
            return Err("--build-capabilities-json must be the only argument; no runtime, model or engine was started".into());
        }
        return print_build_capabilities();
    }
    let work_receipts = SearchWorkOptions::take(&mut arguments)?;
    let mut search = None;
    arguments.retain(|argument| {
        if let Some(value) = argument.strip_prefix("--search=") {
            if search.is_some() {
                search = Some("duplicate".to_owned());
            } else {
                search = Some(value.to_owned());
            }
            false
        } else {
            true
        }
    });
    match search.as_deref().unwrap_or("puct") {
        "puct" => {}
        "cpu" => return run_own_cpu(arguments, work_receipts),
        "pals" => return run_pals(arguments, work_receipts),
        _ => return Err("unsupported or duplicate --search; expected puct, cpu, or pals".into()),
    }
    if work_receipts.is_some() {
        return Err("search work receipts require the CPU or PALS driver; legacy PUCT retains its existing receipt path".into());
    }
    if arguments
        .iter()
        .any(|argument| argument == "--onnx-cpu" || argument == "--onnx-cuda")
    {
        return run_native(arguments);
    }
    run_mock(arguments)
}

fn print_build_capabilities() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // This branch runs before every provider/model/driver parser. These are
    // compile-time facts and the current no-load state, never runtime/provider
    // availability or an inference-admission/placement witness. Semver and
    // Rust target constants are JSON-safe without the optional serde feature.
    writeln!(
        io::stdout().lock(),
        "{{\"schema\":\"rz-uci-build-capability/1\",\"target_os\":\"{}\",\"target_arch\":\"{}\",\"package_version\":\"{}\",\"compile_features\":{{\"onnx_cpu\":{},\"onnx_cuda\":{},\"experimental_io_binding\":{}}},\"cuda_path_compiled\":{},\"native_runtime_loaded\":false,\"native_model_loaded\":false}}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        env!("CARGO_PKG_VERSION"),
        cfg!(feature = "onnx-cpu"),
        cfg!(feature = "onnx-cuda"),
        cfg!(feature = "experimental-io-binding"),
        cfg!(all(feature = "onnx-cuda", target_os = "linux")),
    )?;
    Ok(())
}

fn run_pals(
    arguments: Vec<String>,
    work_receipts: Option<SearchWorkOptions>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = None;
    let mut rounds = None;
    let mut cpu_nodes = None;
    let mut cpu_depth = None;
    let mut situations = None;
    let mut native = PalsNativeOptions::default();
    for argument in arguments {
        if let Some(value) = argument.strip_prefix("--pals-model=") {
            if model.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS model".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-max-rounds=") {
            if rounds.replace(value.parse::<u64>()?).is_some() {
                return Err("duplicate PALS round limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-max-cpu-nodes=") {
            if cpu_nodes.replace(value.parse::<u64>()?).is_some() {
                return Err("duplicate PALS CPU node limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cpu-depth=") {
            if cpu_depth.replace(value.parse::<u16>()?).is_some() {
                return Err("duplicate PALS CPU depth".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cpu-checker=") {
            if native.checker.selection.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CPU checker selection".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cpu-profile=") {
            if native.checker.profile.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CPU checker profile".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cpu-profile-sha256=") {
            if native
                .checker
                .profile_hash
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CPU checker profile digest".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-max-situations=") {
            if situations.replace(value.parse::<usize>()?).is_some() {
                return Err("duplicate PALS situation capacity".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-export-manifest=") {
            if native.manifest.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS export manifest".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-export-sha256=") {
            if native.manifest_hash.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS export hash".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-runtime-path=") {
            if native.runtime.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS runtime path".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-runtime-sha256=") {
            if native.runtime_hash.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS runtime hash".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-runtime-cache-root=") {
            if native
                .runtime_cache_root
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS runtime cache root".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-output-root=") {
            if native.output_root.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS output root".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-launch-sha256=") {
            if native.launch_hash.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS launch digest".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-endpoint-id=") {
            if native.endpoint_id.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS endpoint ID".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-provider=") {
            if native.provider.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS provider".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-bundle=") {
            if native.cuda_bundle.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CUDA bundle".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-bundle-sha256=") {
            if native.cuda_bundle_hash.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CUDA bundle hash".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-device=") {
            if native.cuda_device.replace(value.parse::<i32>()?).is_some() {
                return Err("duplicate PALS CUDA device".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-session-arena-bytes=") {
            if native
                .cuda_session_arena
                .replace(value.parse::<u64>()?)
                .is_some()
            {
                return Err("duplicate PALS CUDA session allocator limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-host-record-pages=") {
            if native
                .host_record_pages
                .replace(value.parse::<bool>()?)
                .is_some()
            {
                return Err("duplicate PALS host record page option".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-private-warm=") {
            if native
                .private_warm
                .replace(value.parse::<bool>()?)
                .is_some()
            {
                return Err("duplicate PALS private warm option".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-device-public-memory=") {
            if native
                .device_public_memory
                .replace(value.parse::<bool>()?)
                .is_some()
            {
                return Err("duplicate PALS device public memory option".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-control-mode=") {
            if native.cuda_control_mode.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CUDA control mode".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-control-inventory=") {
            if native
                .cuda_control_inventory
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CUDA control inventory".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-control-inventory-sha256=") {
            if native
                .cuda_control_inventory_hash
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CUDA control inventory hash".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-profile-root=") {
            if native.cuda_profile_root.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CUDA profile root".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-profile-parent=") {
            if native
                .cuda_profile_parent
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CUDA profile parent".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-startup-probe-timeout-ms=") {
            let value = value
                .parse::<u64>()
                .map_err(|_| "invalid PALS startup probe timeout")?;
            if !(1..=180_000).contains(&value) {
                return Err("PALS startup probe timeout must be 1..180000 milliseconds".into());
            }
            if native.startup_probe_timeout_ms.replace(value).is_some() {
                return Err("duplicate PALS startup probe timeout".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-loading-profile=") {
            if native
                .cuda_loading_profile
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CUDA loading profile".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-loading-profile-sha256=") {
            if native
                .cuda_loading_profile_hash
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CUDA loading profile SHA-256".into());
            }
        } else {
            return Err("PALS requires an explicit model and finite PALS flags; LC0/ORT arguments are not implicitly reused".into());
        }
    }
    let max_cpu_nodes = cpu_nodes.unwrap_or(100_000);
    let rounds = rounds.unwrap_or(16);
    let cpu_depth = cpu_depth.unwrap_or(2);
    let config = rz_search::pals::engine::PalsConfig {
        max_nodes: situations.unwrap_or(4096),
        ..Default::default()
    };
    config.validate()?;
    let external_checker = native.checker.is_external()?;
    if external_checker && model.as_deref() != Some("onnx") {
        return Err("external PALS CPU checker requires the explicit contextual-WDL ONNX model; the legal-order mock has no model-value endpoint".into());
    }
    if rounds == 0
        || rounds > 1_000_000
        || max_cpu_nodes == 0
        || cpu_depth == 0
        || cpu_depth > rz_search::cpu::CpuConfig::default().max_depth
    {
        return Err(
            "PALS round/CPU work limits must be finite and within the registered profile".into(),
        );
    }
    if model.as_deref() == Some("onnx") {
        if work_receipts.is_some() {
            return Err("native PALS uses pals-output-root/launch-sha256/endpoint-id for one combined native/search receipt".into());
        }
        return run_native_pals(native, config, rounds, max_cpu_nodes, cpu_depth);
    }
    if model.as_deref() != Some("legal-order-mock") {
        return Err("PALS requires explicit --pals-model=legal-order-mock or onnx; no model fallback was started".into());
    }
    if native.any() {
        return Err(
            "PALS mock selection cannot accept neural assets or a provider declaration".into(),
        );
    }
    let driver = Arc::new(rz_uci::search_driver::PalsSessionDriver::new(
        config,
        rz_search::pals::engine::LegalOrderRoleMock,
        rz_search::cpu::CpuConfig::default(),
        rounds,
        max_cpu_nodes,
        cpu_depth,
        rz_uci::EngineIdentity {
            name: "RoveZero PALS explicit legal-order CPU mock + own CPU_R".into(),
            author: "RoveZero contributors".into(),
        },
    )?);
    let mut settings = EngineSettings::default();
    settings.search.max_simulations = max_cpu_nodes;
    serve_search_process(driver, settings, work_receipts)?;
    Ok(())
}

#[derive(Default)]
struct PalsNativeOptions {
    checker: PalsCheckerOptions,
    manifest: Option<String>,
    manifest_hash: Option<String>,
    runtime: Option<String>,
    runtime_hash: Option<String>,
    runtime_cache_root: Option<String>,
    provider: Option<String>,
    output_root: Option<String>,
    launch_hash: Option<String>,
    endpoint_id: Option<String>,
    cuda_bundle: Option<String>,
    cuda_bundle_hash: Option<String>,
    cuda_device: Option<i32>,
    cuda_session_arena: Option<u64>,
    host_record_pages: Option<bool>,
    private_warm: Option<bool>,
    device_public_memory: Option<bool>,
    cuda_control_mode: Option<String>,
    cuda_control_inventory: Option<String>,
    cuda_control_inventory_hash: Option<String>,
    cuda_profile_root: Option<String>,
    cuda_profile_parent: Option<String>,
    cuda_loading_profile: Option<String>,
    cuda_loading_profile_hash: Option<String>,
    startup_probe_timeout_ms: Option<u64>,
}
#[derive(Default)]
struct PalsCheckerOptions {
    selection: Option<String>,
    profile: Option<String>,
    profile_hash: Option<String>,
}
impl PalsCheckerOptions {
    fn is_external(&self) -> Result<bool, Box<dyn std::error::Error>> {
        match (
            self.selection.as_deref().unwrap_or("own"),
            &self.profile,
            &self.profile_hash,
        ) {
            ("own", None, None) => Ok(false),
            ("external-uci", Some(path), Some(hash)) if !path.is_empty() && !hash.is_empty() => Ok(true),
            _ => Err("PALS CPU checker expects own without a profile, or external-uci with both absolute profile path and SHA-256; no fallback was started".into()),
        }
    }
    #[cfg(feature = "onnx-cpu")]
    fn load(
        &self,
    ) -> Result<
        Option<rz_uci::pals_checker_profile::LoadedPalsCheckerProfile>,
        Box<dyn std::error::Error>,
    > {
        if !self.is_external()? {
            return Ok(None);
        }
        Ok(Some(
            rz_uci::pals_checker_profile::PalsCheckerProfile::load(
                std::path::Path::new(self.profile.as_deref().ok_or("missing checker profile")?),
                self.profile_hash
                    .as_deref()
                    .ok_or("missing checker profile pin")?,
            )?,
        ))
    }
}
#[cfg(test)]
mod pals_checker_option_tests {
    use super::PalsCheckerOptions;

    #[test]
    fn helper_selection_requires_an_explicit_complete_profile_without_fallback() {
        assert!(!PalsCheckerOptions::default().is_external().unwrap());
        for (selection, path, hash, accepted) in [
            ("own", None, None, true),
            ("own", Some("/profile.json"), Some("pin"), false),
            ("external-uci", None, None, false),
            ("external-uci", Some("/profile.json"), None, false),
            ("external-uci", None, Some("pin"), false),
            ("external-uci", Some(""), Some("pin"), false),
            ("external-uci", Some("/profile.json"), Some("pin"), true),
            ("unknown", None, None, false),
        ] {
            let options = PalsCheckerOptions {
                selection: Some(selection.into()),
                profile: path.map(str::to_owned),
                profile_hash: hash.map(str::to_owned),
            };
            assert_eq!(options.is_external().is_ok(), accepted, "{selection}");
        }
        // This parser only selects a branch. Absolute paths, byte/canonical pin,
        // assets and supported options are independently checked by the loader.
    }
}
impl PalsNativeOptions {
    fn any(&self) -> bool {
        self.manifest.is_some()
            || self.manifest_hash.is_some()
            || self.runtime.is_some()
            || self.runtime_hash.is_some()
            || self.runtime_cache_root.is_some()
            || self.provider.is_some()
            || self.output_root.is_some()
            || self.launch_hash.is_some()
            || self.endpoint_id.is_some()
            || self.cuda_bundle.is_some()
            || self.cuda_bundle_hash.is_some()
            || self.cuda_device.is_some()
            || self.cuda_session_arena.is_some()
            || self.device_public_memory.is_some()
            || self.host_record_pages.is_some()
            || self.private_warm.is_some()
            || self.cuda_control_mode.is_some()
            || self.cuda_control_inventory.is_some()
            || self.cuda_control_inventory_hash.is_some()
            || self.cuda_profile_root.is_some()
            || self.cuda_profile_parent.is_some()
            || self.cuda_loading_profile.is_some()
            || self.cuda_loading_profile_hash.is_some()
            || self.startup_probe_timeout_ms.is_some()
    }
}

#[cfg(not(feature = "onnx-cpu"))]
fn run_native_pals(
    _native: PalsNativeOptions,
    _config: rz_search::pals::engine::PalsConfig,
    _rounds: u64,
    _cpu_nodes: u64,
    _cpu_depth: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "PALS ONNX startup requires feature onnx-cpu; no provider or model fallback was started"
            .into(),
    )
}

#[cfg(feature = "onnx-cpu")]
fn run_native_pals(
    native: PalsNativeOptions,
    config: rz_search::pals::engine::PalsConfig,
    rounds: u64,
    cpu_nodes: u64,
    cpu_depth: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    // PALS owns a distinct manifest and tensors. LC0 NativeConfig and its
    // attestation cannot authorize or describe this model.
    let checker_profile = native.checker.load()?;
    let checker = checker_profile
        .as_ref()
        .map(|profile| profile.create_checker())
        .transpose()?;
    let checker_handshake_timeout = checker_profile
        .as_ref()
        .map(|profile| profile.config().map(|config| config.handshake_timeout))
        .transpose()?
        .unwrap_or(Duration::ZERO); // unused for the synchronous own checker
    let provider = match native.provider.as_deref() {
        Some("cpu") => "cpu",
        Some("cuda") if cfg!(all(feature = "onnx-cuda", target_os = "linux")) => "cuda",
        Some("cuda") => {
            return Err(
                "PALS CUDA requires onnx-cuda on Linux; no CPU fallback was started".into(),
            );
        }
        _ => {
            return Err(
                "PALS requires explicit --pals-provider=cpu or cuda; no fallback was started"
                    .into(),
            );
        }
    };
    if provider == "cpu"
        && (native.cuda_bundle.is_some()
            || native.cuda_bundle_hash.is_some()
            || native.cuda_device.is_some()
            || native.cuda_session_arena.is_some()
            || native.device_public_memory.unwrap_or(false)
            || native.cuda_control_mode.is_some()
            || native.cuda_control_inventory.is_some()
            || native.cuda_control_inventory_hash.is_some()
            || native.cuda_profile_root.is_some()
            || native.cuda_profile_parent.is_some()
            || native.cuda_loading_profile.is_some()
            || native.cuda_loading_profile_hash.is_some()
            || native.startup_probe_timeout_ms.is_some())
    {
        return Err("PALS CPU selection cannot accept CUDA bundle/device/allocator options".into());
    }
    if native.device_public_memory.unwrap_or(false) && !cfg!(feature = "experimental-io-binding") {
        return Err("PALS device public memory requires explicit experimental-io-binding; no binding fallback was started".into());
    }
    if native.host_record_pages.unwrap_or(false) && native.device_public_memory.unwrap_or(false) {
        return Err("PALS host record pages and device public memory are mutually exclusive; no fallback was started".into());
    }
    if native.private_warm.unwrap_or(false)
        && (provider != "cpu" || native.device_public_memory.unwrap_or(false))
    {
        return Err("PALS private warm requires a separately pinned CPU/host warm export; CUDA/device/V are unsupported".into());
    }
    let loading_profile = pals_cuda_loading_profile(
        native.cuda_loading_profile.as_deref(),
        native.cuda_loading_profile_hash.as_deref(),
        provider,
        native.cuda_control_mode.as_deref() == Some("inventory-v2"),
        native.device_public_memory.unwrap_or(false),
    )?;
    let profile_root = pals_profile_root(native.cuda_profile_root, native.cuda_profile_parent)?;
    let cuda_control = match (
        native.cuda_control_mode,
        native.cuda_control_inventory,
        native.cuda_control_inventory_hash,
        profile_root,
    ) {
        (None, None, None, None) => None,
        (Some(mode), Some(inventory), Some(hash), Some(profile))
            if mode == "inventory-v2" && provider == "cuda" =>
        {
            if native.device_public_memory.unwrap_or(false) {
                return Err("PALS inventory-v2 control mode requires host public memory; no device-binding fallback was started".into());
            }
            let inventory = std::path::PathBuf::from(inventory);
            if !inventory.is_absolute() || !profile.is_absolute() {
                return Err("PALS control inventory and profile root must be absolute".into());
            }
            rz_eval::asset::parse_sha256(&hash)?;
            let policy = rz_eval::pals_onnx::PalsCudaControlPolicy::from_inventory(&inventory, &hash)?;
            Some((policy, profile))
        }
        _ => return Err("PALS CUDA control mode requires inventory-v2, inventory path, inventory SHA-256 and a fresh owned profile root or exclusive process-profile parent together; strict mode is the default".into()),
    };
    let receipt_config =
        match (native.output_root, native.launch_hash, native.endpoint_id) {
            (None, None, None) => None,
            (Some(root), Some(launch), Some(endpoint)) => {
                let root = std::path::PathBuf::from(root);
                if !root.is_absolute() {
                    return Err("PALS output root must be absolute".into());
                }
                rz_eval::asset::parse_sha256(&launch)?;
                Some((root, launch, endpoint))
            }
            _ => return Err(
                "PALS receipt output requires output root, launch digest and endpoint ID together"
                    .into(),
            ),
        };
    if checker_profile.is_some() && receipt_config.is_none() {
        return Err("external PALS checker requires the combined output root, launch digest and endpoint ID for independent owner evidence".into());
    }
    let manifest =
        std::path::PathBuf::from(native.manifest.ok_or("PALS export manifest is required")?);
    let runtime_path =
        std::path::PathBuf::from(native.runtime.ok_or("PALS runtime path is required")?);
    if !manifest.is_absolute() || !runtime_path.is_absolute() {
        return Err("PALS model/runtime paths must be absolute".into());
    }
    let manifest_hash = native
        .manifest_hash
        .ok_or("PALS export SHA-256 is required")?;
    let runtime_hash = native
        .runtime_hash
        .ok_or("PALS runtime SHA-256 is required")?;
    rz_eval::asset::parse_sha256(&manifest_hash)?;
    rz_eval::asset::parse_sha256(&runtime_hash)?;
    let cache = match native.runtime_cache_root {
        Some(path) => {
            let path = std::path::PathBuf::from(path);
            if !path.is_absolute() {
                return Err("PALS runtime cache root must be absolute".into());
            }
            rz_eval::runtime_pin::RuntimeCache::open(&path)?
        }
        None => rz_eval::runtime_pin::RuntimeCache::for_user()?,
    };
    let mut backend_config = rz_eval::pals_onnx::PalsOnnxConfig::cpu();
    let pin = if provider == "cpu" {
        cache.library(&runtime_path, &runtime_hash)?
    } else {
        let bundle_path =
            std::path::PathBuf::from(native.cuda_bundle.ok_or("PALS CUDA bundle is required")?);
        if !bundle_path.is_absolute() {
            return Err("PALS CUDA bundle path must be absolute".into());
        }
        let bundle_hash = native
            .cuda_bundle_hash
            .ok_or("PALS CUDA bundle SHA-256 is required")?;
        let expected_bundle = rz_eval::asset::parse_sha256(&bundle_hash)?;
        let bytes = rz_eval::asset::read_bounded(&bundle_path, 64 * 1024)?;
        if rz_eval::asset::sha256(&bytes) != expected_bundle {
            return Err("PALS CUDA bundle descriptor differs from its SHA-256 pin".into());
        }
        let spec =
            rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&bytes)?)?;
        let core = spec
            .files
            .iter()
            .find(|entry| entry.role == rz_eval::runtime_pin::RuntimeBundleFileRole::Core)
            .ok_or("PALS CUDA bundle core is absent")?;
        if runtime_path.file_name().and_then(|name| name.to_str()) != Some(core.filename.as_str())
            || core.sha256 != runtime_hash
        {
            return Err("PALS explicit ORT core filename/hash differs from CUDA bundle".into());
        }
        let device_id = native
            .cuda_device
            .ok_or("PALS CUDA device ID is required")?;
        let arena = native
            .cuda_session_arena
            .ok_or("PALS per-session CUDA allocator limit is required")?;
        if device_id != 0 || arena != 2 * 1024 * 1024 * 1024 {
            return Err("first PALS CUDA profile requires device 0 and 2GiB per-session allocator declaration; this is not a measured peak".into());
        }
        backend_config.provider = rz_eval::onnx::Provider::Cuda {
            device_id,
            arena_bytes: usize::try_from(arena)?,
        };
        backend_config.device_public_memory = native.device_public_memory.unwrap_or(false);
        cache.cuda_bundle(
            runtime_path
                .parent()
                .ok_or("PALS CUDA core parent is absent")?,
            &spec,
        )?
    };
    let mut settings = EngineSettings::default();
    settings.search.max_simulations = cpu_nodes;
    let owner_options = rz_uci::pals_native::NativeOwnerOptions {
        drain_limit: settings.shutdown_limit,
        host_record_pages: native.host_record_pages.unwrap_or(false).then_some(
            rz_uci::pals_native::NativeHostRecordPageLimits {
                max_page_entries: 257,
                max_page_bytes: 2 * 1024 * 1024,
                max_transient_bytes: 2 * 1024 * 1024,
            },
        ),
    };
    // Observe actual runtime/model factory cost independently of the probe
    // deadline. Earlier pin/cache preparation belongs to overall CLI cost.
    let model_loading_started = std::time::Instant::now();
    let mut model = match cuda_control {
        Some((policy, profile)) => match loading_profile {
            Some(loading) => {
                let runtime =
                    rz_eval::onnx::OrtRuntime::load_with_cuda_loading_profile(&pin, loading)?;
                rz_uci::pals_native::NativeRoleModel::load_with_runtime_and_cuda_control_policy_with_options(
                    &manifest,
                    &manifest_hash,
                    runtime,
                    backend_config,
                    policy,
                    &profile,
                    owner_options,
                )?
            }
            None => {
                let runtime = rz_eval::onnx::OrtRuntime::load(&pin)?;
                rz_uci::pals_native::NativeRoleModel::load_with_runtime_and_cuda_control_policy_with_options(
                    &manifest,
                    &manifest_hash,
                    runtime,
                    backend_config,
                    policy,
                    &profile,
                    owner_options,
                )?
            }
        },
        None if native.private_warm.unwrap_or(false) => {
            rz_uci::pals_native::NativeRoleModel::load_pinned_cpu_private_warm_with_options(
                &manifest,
                &manifest_hash,
                &pin,
                backend_config,
                owner_options,
            )?
        }
        None => rz_uci::pals_native::NativeRoleModel::load_pinned_with_options(
            &manifest,
            &manifest_hash,
            &pin,
            backend_config,
            owner_options,
        )?,
    };
    let model_loading_elapsed = model_loading_started.elapsed();
    let finish = model.finish_handle();
    let startup = (|| {
        let budget = model.configure_startup_probe_timeout(native.startup_probe_timeout_ms)?;
        model.observe_startup_loading(model_loading_elapsed)?;
        model.prepare_startup(std::time::Instant::now() + budget)
    })();
    if let Err(primary) = startup {
        let mut publication = None;
        let mut failure_writer = match receipt_config.as_ref() {
            Some((root, launch, endpoint)) => {
                match rz_uci::pals_attestation::PalsReceiptWriter::open(
                    root,
                    endpoint,
                    launch,
                    &runtime_hash,
                ) {
                    Ok(mut writer) => match writer.failed_startup(finish.receipt(), &primary, None)
                    {
                        Ok(()) => Some(writer),
                        Err(error) => {
                            publication = Some(error);
                            None
                        }
                    },
                    Err(error) => {
                        publication = Some(error);
                        None
                    }
                }
            }
            None => None,
        };
        let cleanup = finish
            .finish(std::time::Instant::now() + settings.shutdown_limit)
            .err();
        if let Some(writer) = failure_writer.as_mut() {
            publication = writer.termination(finish.receipt(), false, None).err();
        }
        return Err(Box::new(PalsFinishError {
            primary: Some(Box::new(primary)),
            cleanup,
            checker_cleanup: None,
            checker_observation: None,
            publication,
            work_observation: None,
        }));
    }
    let identity = rz_uci::EngineIdentity {
        name: format!(
            "RoveZero PALS P/C ONNX {} + {} CPU_R",
            if provider == "cpu" { "CPU" } else { "CUDA" },
            if checker.is_some() {
                "external UCI"
            } else {
                "own"
            },
        ),
        author: "RoveZero contributors".into(),
    };
    // The helper is unstarted through every fallible constructor. No foreign
    // process can disappear into a constructor Drop without a shutdown receipt.
    let driver = match match checker {
        Some(checker) => rz_uci::search_driver::PalsSessionDriver::new_with_checker(
            config, model, checker, rounds, cpu_nodes, cpu_depth, identity,
        ),
        None => rz_uci::search_driver::PalsSessionDriver::new(
            config,
            model,
            rz_search::cpu::CpuConfig::default(),
            rounds,
            cpu_nodes,
            cpu_depth,
            identity,
        ),
    } {
        Ok(driver) => Arc::new(driver),
        Err(primary) => {
            let cleanup = finish
                .finish(std::time::Instant::now() + settings.shutdown_limit)
                .err();
            return Err(Box::new(PalsFinishError {
                primary: Some(Box::new(primary)),
                cleanup,
                checker_cleanup: None,
                checker_observation: None,
                publication: None,
                work_observation: None,
            }));
        }
    };
    if let Some(profile) = checker_profile.as_ref() {
        let control = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let startup = driver.start_checker(
            std::time::Instant::now() + checker_handshake_timeout,
            control,
        );
        if let Err(primary) = startup {
            // Both owners are attempted independently. A failed helper start is
            // not a Native startup failure and does not invent a ready helper.
            let (checker_cleanup, cleanup) = finish_pals_owners_within(
                settings.shutdown_limit,
                |until| driver.finish_checker(until).err(),
                |until| finish.finish(until).err(),
            );
            let observed_work = driver.work_receipt();
            let observed_checker =
                rz_uci::pals_attestation::checker::PalsCheckerProcessReceipt::observe(
                    &driver,
                    profile.file_sha256(),
                    profile.canonical_sha256(),
                );
            let mut publication = None;
            if let Some((root, launch, endpoint)) = receipt_config.as_ref() {
                let result = (|| {
                    let mut writer = rz_uci::pals_attestation::PalsReceiptWriter::open(
                        root,
                        endpoint,
                        launch,
                        &runtime_hash,
                    )?;
                    if let Ok(evidence) = &observed_checker {
                        writer.observe_checker(evidence.clone())?;
                    }
                    writer.startup(
                        finish.receipt(),
                        observed_work.as_ref().ok().cloned().flatten(),
                    )?;
                    writer.termination(
                        finish.receipt(),
                        false,
                        observed_work.as_ref().ok().cloned().flatten(),
                    )
                })();
                publication = result.err();
            }
            return Err(Box::new(PalsFinishError {
                primary: Some(Box::new(primary)),
                cleanup,
                checker_cleanup,
                checker_observation: observed_checker.err(),
                publication,
                work_observation: observed_work.err(),
            }));
        }
    }
    let startup_checker = checker_profile
        .as_ref()
        .map(|profile| {
            rz_uci::pals_attestation::checker::PalsCheckerProcessReceipt::observe(
                &driver,
                profile.file_sha256(),
                profile.canonical_sha256(),
            )
        })
        .transpose();
    let startup_checker = match startup_checker {
        Ok(evidence) => evidence,
        Err(checker_observation) => {
            let (checker_cleanup, cleanup) = finish_pals_owners_within(
                settings.shutdown_limit,
                |until| driver.finish_checker(until).err(),
                |until| finish.finish(until).err(),
            );
            return Err(Box::new(PalsFinishError {
                primary: None,
                cleanup,
                checker_cleanup,
                checker_observation: Some(checker_observation),
                publication: None,
                work_observation: None,
            }));
        }
    };
    let startup_work = match driver.work_receipt() {
        Ok(work) => work,
        Err(work_observation) => {
            let (checker_cleanup, cleanup) = finish_pals_owners_within(
                settings.shutdown_limit,
                |until| driver.finish_checker(until).err(),
                |until| finish.finish(until).err(),
            );
            return Err(Box::new(PalsFinishError {
                primary: None,
                cleanup,
                checker_cleanup,
                checker_observation: None,
                publication: None,
                work_observation: Some(work_observation),
            }));
        }
    };
    let mut receipts = match receipt_config {
        Some((root, launch, endpoint)) => {
            let opened = rz_uci::pals_attestation::PalsReceiptWriter::open(
                &root,
                &endpoint,
                &launch,
                &runtime_hash,
            );
            match opened {
                Ok(mut writer) => match (|| {
                    if let Some(evidence) = startup_checker.as_ref() {
                        writer.observe_checker(evidence.clone())?;
                    }
                    writer.startup(finish.receipt(), startup_work.clone())
                })() {
                    Ok(()) => Some(writer),
                    Err(publication) => {
                        let (checker_cleanup, cleanup) = finish_pals_owners_within(
                            settings.shutdown_limit,
                            |until| driver.finish_checker(until).err(),
                            |until| finish.finish(until).err(),
                        );
                        return Err(Box::new(PalsFinishError {
                            primary: None,
                            cleanup,
                            checker_cleanup,
                            checker_observation: None,
                            publication: Some(publication),
                            work_observation: None,
                        }));
                    }
                },
                Err(publication) => {
                    let (checker_cleanup, cleanup) = finish_pals_owners_within(
                        settings.shutdown_limit,
                        |until| driver.finish_checker(until).err(),
                        |until| finish.finish(until).err(),
                    );
                    return Err(Box::new(PalsFinishError {
                        primary: None,
                        cleanup,
                        checker_cleanup,
                        checker_observation: None,
                        publication: Some(publication),
                        work_observation: None,
                    }));
                }
            }
        }
        None => None,
    };
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let process = EngineProcess::with_search_driver(driver.clone(), owners, clock);
    let served = serve_process(process, settings);
    let (checker_finished, finished) = finish_pals_owners_within(
        settings.shutdown_limit,
        |until| driver.finish_checker(until),
        |until| finish.finish(until),
    );
    let observed_work = driver.work_receipt();
    let observed_checker = checker_profile
        .as_ref()
        .map(|profile| {
            rz_uci::pals_attestation::checker::PalsCheckerProcessReceipt::observe(
                &driver,
                profile.file_sha256(),
                profile.canonical_sha256(),
            )
        })
        .transpose();
    let publication = receipts.as_mut().and_then(|writer| {
        if let Ok(Some(evidence)) = &observed_checker {
            if let Err(error) = writer.observe_checker(evidence.clone()) {
                return Some(error);
            }
        }
        writer
            .termination(
                finish.receipt(),
                served.is_ok()
                    && finished.is_ok()
                    && checker_finished.is_ok()
                    && observed_work.is_ok()
                    && observed_checker.is_ok(),
                observed_work.as_ref().ok().cloned().flatten(),
            )
            .err()
    });
    if publication.is_some()
        || observed_work.is_err()
        || checker_finished.is_err()
        || observed_checker.is_err()
    {
        return Err(Box::new(PalsFinishError {
            primary: served
                .err()
                .map(|error| Box::new(error) as Box<dyn std::error::Error>),
            cleanup: finished.err(),
            checker_cleanup: checker_finished.err(),
            checker_observation: observed_checker.err(),
            publication,
            work_observation: observed_work.err(),
        }));
    }
    match (served, finished) {
        (Ok(()), Ok(receipt)) => {
            eprintln!(
                "PALS native role shutdown: completed={} delivered={} consumed={} canceled={} expired={} failed={} invalid={} in_flight={} joined={} buffers_released={} quarantined={}",
                receipt.completed_role_inputs,
                receipt.delivered_role_inputs,
                receipt.search_consumed_role_inputs,
                receipt.canceled_requests,
                receipt.expired_requests,
                receipt.failed_physical_role_calls,
                receipt.invalid_role_outputs,
                receipt.physical_runs_in_flight,
                receipt.physical_shutdown_confirmed,
                receipt.native_buffers_released,
                receipt.quarantined,
            );
            Ok(())
        }
        (Err(primary), Ok(_)) => Err(Box::new(primary)),
        (primary, Err(cleanup)) => Err(Box::new(PalsFinishError {
            primary: primary
                .err()
                .map(|error| Box::new(error) as Box<dyn std::error::Error>),
            cleanup: Some(cleanup),
            checker_cleanup: None,
            checker_observation: None,
            publication: None,
            work_observation: None,
        })),
    }
}

/// Both owners are attempted independently against the same overall deadline.
/// A slow helper must not grant the Native owner a fresh cleanup window.
#[cfg(feature = "onnx-cpu")]
fn finish_pals_owners_within<C, N>(
    limit: Duration,
    checker: impl FnOnce(std::time::Instant) -> C,
    native: impl FnOnce(std::time::Instant) -> N,
) -> (C, N) {
    let now = std::time::Instant::now();
    let until = now.checked_add(limit).unwrap_or(now);
    let checker_result = checker(until);
    let native_result = native(until);
    (checker_result, native_result)
}

#[cfg(feature = "onnx-cpu")]
fn pals_cuda_loading_profile(
    profile: Option<&str>,
    expected_hash: Option<&str>,
    provider: &str,
    inventory_control: bool,
    device_public_memory: bool,
) -> Result<Option<rz_eval::onnx::NativeLoadingProfile>, Box<dyn std::error::Error>> {
    match (profile, expected_hash) {
        (None, None) => Ok(None),
        (Some("experimental-cudnn-shim-lazy-v1"), Some(hash))
            if provider == "cuda" && inventory_control && !device_public_memory =>
        {
            let profile = rz_eval::onnx::NativeLoadingProfile::CuDnnShimLazyV1;
            if rz_eval::asset::parse_sha256(hash)?
                != rz_eval::asset::sha256(profile.canonical_descriptor().as_bytes())
            {
                return Err("PALS CUDA loading profile canonical SHA-256 differs; no native loading was started".into());
            }
            Ok(Some(profile))
        }
        _ => Err("PALS experimental CUDA loading profile and canonical SHA-256 must be selected together with CUDA inventory-v2 host control; no loading-profile or CPU fallback was started".into()),
    }
}

#[cfg(feature = "onnx-cpu")]
fn pals_profile_root(
    root: Option<String>,
    parent: Option<String>,
) -> Result<Option<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let path = match (root, parent) {
        (None, None) => None,
        (Some(root), None) => Some(std::path::PathBuf::from(root)),
        (None, Some(parent)) => {
            let parent = std::path::PathBuf::from(parent);
            if !parent.is_absolute() {
                return Err("PALS CUDA profile parent must be absolute".into());
            }
            let metadata = std::fs::symlink_metadata(&parent)?;
            if !metadata.is_dir() || metadata.file_type().is_symlink() {
                return Err("PALS CUDA profile parent must be an existing owned directory".into());
            }
            // Fastchess uses the same launch arguments across games. Derive a
            // fresh process-owned child here, then leave exclusive creation to
            // the backend: a reused PID or nonce never authorizes overwriting.
            let nonce = u64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos(),
            )?;
            Some(parent.join(format!(
                "pals-cuda-profile-{}-{nonce:016x}",
                std::process::id()
            )))
        }
        (Some(_), Some(_)) => {
            return Err("PALS CUDA profile root and parent are mutually exclusive".into());
        }
    };
    Ok(path)
}

#[cfg(all(test, feature = "onnx-cpu"))]
mod pals_profile_tests {
    use super::{finish_pals_owners_within, pals_cuda_loading_profile, pals_profile_root};

    #[test]
    fn both_owner_cleanup_attempts_share_one_deadline_even_after_helper_failure() {
        use std::cell::RefCell;
        use std::time::{Duration, Instant};
        for limit in [Duration::from_secs(1), Duration::ZERO] {
            let observed = RefCell::new(Vec::new());
            let started = Instant::now();
            let (checker, native) = finish_pals_owners_within(
                limit,
                |until| {
                    observed.borrow_mut().push(("checker", until));
                    Err::<(), _>("helper cleanup failed")
                },
                |until| {
                    observed.borrow_mut().push(("native", until));
                    Err::<(), _>("native cleanup failed")
                },
            );
            assert_eq!(checker, Err("helper cleanup failed"));
            assert_eq!(native, Err("native cleanup failed"));
            let observed = observed.into_inner();
            assert_eq!(observed.len(), 2);
            assert_eq!(observed[0].0, "checker");
            assert_eq!(observed[1].0, "native");
            assert_eq!(observed[0].1, observed[1].1);
            assert!(observed[0].1 >= started);
            assert!(observed[0].1 <= Instant::now() + limit);
            if limit.is_zero() {
                assert!(observed[1].1 <= Instant::now());
            }
        }
    }

    #[test]
    fn loading_profile_is_an_explicit_canonical_host_control_selection() {
        const SHIM: &str = "experimental-cudnn-shim-lazy-v1";
        const SHA: &str = "3084f27e678d6a6750713265bfb921e90687f199fb71484ff2c475000982ee6d";
        assert_eq!(
            pals_cuda_loading_profile(None, None, "cpu", false, false).unwrap(),
            None
        );
        assert_eq!(
            pals_cuda_loading_profile(None, None, "cuda", false, false).unwrap(),
            None
        );
        assert_eq!(
            pals_cuda_loading_profile(Some(SHIM), Some(SHA), "cuda", true, false).unwrap(),
            Some(rz_eval::onnx::NativeLoadingProfile::CuDnnShimLazyV1)
        );
        // These are pure option checks. No cache, file, model or native runtime
        // is opened, including the unsupported device/public-memory case.
        for (profile, digest, provider, control, device) in [
            (Some(SHIM), None, "cuda", true, false),
            (None, Some(SHA), "cuda", true, false),
            (Some(SHIM), Some(SHA), "cpu", true, false),
            (Some(SHIM), Some(SHA), "cuda", false, false),
            (Some(SHIM), Some(SHA), "cuda", true, true),
            (
                Some("eager-cuda12-cudnn9-v1"),
                Some(SHA),
                "cuda",
                true,
                false,
            ),
            (Some(SHIM), Some("00"), "cuda", true, false),
            (
                Some(SHIM),
                Some("0000000000000000000000000000000000000000000000000000000000000000"),
                "cuda",
                true,
                false,
            ),
        ] {
            assert!(pals_cuda_loading_profile(profile, digest, provider, control, device).is_err());
        }
    }

    #[test]
    fn exact_root_and_absent_control_preserve_existing_arguments() {
        assert_eq!(pals_profile_root(None, None).unwrap(), None);
        let exact = std::env::temp_dir().join("pals-explicit-profile-root");
        assert_eq!(
            pals_profile_root(Some(exact.to_string_lossy().into_owned()), None).unwrap(),
            Some(exact)
        );
        assert!(pals_profile_root(Some("root".into()), Some("parent".into())).is_err());
    }

    #[test]
    fn process_profile_parent_derives_but_does_not_create_a_child() {
        let parent = std::env::temp_dir();
        let child = pals_profile_root(None, Some(parent.to_string_lossy().into_owned()))
            .unwrap()
            .unwrap();
        assert_eq!(child.parent(), Some(parent.as_path()));
        assert!(
            child
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(&format!("pals-cuda-profile-{}-", std::process::id()))
        );
        assert!(!child.exists());
        assert!(pals_profile_root(None, Some("relative-profile-parent".into())).is_err());
    }
}

#[cfg(feature = "onnx-cpu")]
#[derive(Debug)]
struct PalsFinishError {
    primary: Option<Box<dyn std::error::Error>>,
    cleanup: Option<rz_search::pals::engine::RoleError>,
    checker_cleanup: Option<rz_uci::search_driver::SearchSessionFailure>,
    checker_observation: Option<rz_uci::search_driver::SearchSessionFailure>,
    publication: Option<rz_uci::process_receipts::ProcessReceiptError>,
    work_observation: Option<rz_contracts::ContractError>,
}
#[cfg(feature = "onnx-cpu")]
impl std::fmt::Display for PalsFinishError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(primary) = &self.primary {
            write!(formatter, "{primary}; ")?;
        }
        if let Some(cleanup) = &self.cleanup {
            write!(formatter, "PALS physical shutdown failed: {cleanup}; ")?;
        }
        if let Some(cleanup) = &self.checker_cleanup {
            write!(formatter, "PALS CPU helper shutdown failed: {cleanup}; ")?;
        }
        if let Some(observation) = &self.checker_observation {
            write!(
                formatter,
                "PALS CPU helper observation failed: {observation}; "
            )?;
        }
        if let Some(publication) = &self.publication {
            write!(formatter, "PALS receipt publication failed: {publication}")?;
        }
        if let Some(observation) = &self.work_observation {
            write!(formatter, "PALS work observation failed: {observation}")?;
        }
        Ok(())
    }
}
#[cfg(feature = "onnx-cpu")]
impl std::error::Error for PalsFinishError {}

fn run_own_cpu(
    arguments: Vec<String>,
    work_receipts: Option<SearchWorkOptions>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut max_depth = None;
    let mut max_nodes = None;
    let mut tt_entries = None;
    for argument in arguments {
        if let Some(value) = argument.strip_prefix("--cpu-max-depth=") {
            if max_depth.replace(value.parse::<u16>()?).is_some() {
                return Err("duplicate CPU depth limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--cpu-max-nodes=") {
            if max_nodes.replace(value.parse::<u64>()?).is_some() {
                return Err("duplicate CPU node limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--cpu-tt-entries=") {
            if tt_entries.replace(value.parse::<usize>()?).is_some() {
                return Err("duplicate CPU TT limit".into());
            }
        } else {
            return Err("CPU search only accepts --cpu-max-depth, --cpu-max-nodes and --cpu-tt-entries; neural provider/model flags are inapplicable".into());
        }
    }
    let max_nodes = max_nodes.unwrap_or(100_000);
    // Explicit finite limits: default profile is own bootstrap CPU_R, not NNUE.
    let config = rz_search::cpu::CpuConfig {
        max_depth: max_depth.unwrap_or(8),
        tt_entries: tt_entries.unwrap_or(65_536),
        ..Default::default()
    };
    let driver = Arc::new(rz_uci::search_driver::CpuSessionDriver::new(
        config, max_nodes,
    )?);
    let mut settings = EngineSettings::default();
    // Legacy field is the UCI work admission cap. CPU interprets go nodes as
    // actual CPU nodes; no completed simulation or NN receipt is fabricated.
    settings.search.max_simulations = max_nodes;
    serve_search_process(driver, settings, work_receipts)?;
    Ok(())
}

struct SearchWorkOptions {
    root: std::path::PathBuf,
    launch_sha256: String,
    endpoint_id: String,
}
impl SearchWorkOptions {
    fn take(arguments: &mut Vec<String>) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let mut root = None;
        let mut launch = None;
        let mut endpoint = None;
        let mut duplicate = false;
        arguments.retain(|argument| {
            let replaced = if let Some(value) = argument.strip_prefix("--search-work-output-root=")
            {
                root.replace(value.to_owned())
            } else if let Some(value) = argument.strip_prefix("--search-work-launch-sha256=") {
                launch.replace(value.to_owned())
            } else if let Some(value) = argument.strip_prefix("--search-work-endpoint-id=") {
                endpoint.replace(value.to_owned())
            } else {
                return true;
            };
            duplicate |= replaced.is_some();
            false
        });
        if duplicate {
            return Err("duplicate search work receipt argument".into());
        }
        match (root, launch, endpoint) {
            (None, None, None) => Ok(None),
            (Some(root), Some(launch_sha256), Some(endpoint_id)) => {
                let root = std::path::PathBuf::from(root);
                if !root.is_absolute() {
                    return Err("search work output root must be absolute".into());
                }
                rz_eval::asset::parse_sha256(&launch_sha256)?;
                #[cfg(not(feature = "search-work-receipts"))]
                {
                    let _ = (root, launch_sha256, endpoint_id);
                    Err("unsupported feature: search work receipts require feature search-work-receipts; no driver was started".into())
                }
                #[cfg(feature = "search-work-receipts")]
                {
                    Ok(Some(Self {
                        root,
                        launch_sha256,
                        endpoint_id,
                    }))
                }
            }
            _ => Err(
                "search work output requires root, launch digest and endpoint ID together".into(),
            ),
        }
    }
}

fn serve_search_process(
    driver: Arc<dyn SearchSessionDriver>,
    settings: EngineSettings,
    receipt: Option<SearchWorkOptions>,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "search-work-receipts")]
    let mut receipts = match receipt {
        Some(receipt) => {
            let mut writer = rz_uci::pals_attestation::SearchWorkReceiptWriter::open(
                &receipt.root,
                &receipt.endpoint_id,
                &receipt.launch_sha256,
            )?;
            writer.startup(driver.work_receipt()?)?;
            Some(writer)
        }
        None => None,
    };
    #[cfg(not(feature = "search-work-receipts"))]
    if receipt.is_some() {
        return Err("unsupported feature: search-work-receipts".into());
    }
    let process = EngineProcess::with_search_driver(
        Arc::clone(&driver),
        Arc::new(OwnerRegistry::default()),
        ProcessClock::new(ProcessEpoch(1)),
    );
    let served = serve_process(process, settings);
    #[cfg(feature = "search-work-receipts")]
    {
        let observed = driver.work_receipt();
        let publication = receipts.as_mut().and_then(|writer| {
            writer
                .termination(observed.as_ref().ok().cloned().flatten(), served.is_ok())
                .err()
        });
        if observed.is_err() || publication.is_some() {
            return Err(Box::new(SearchWorkFinishError {
                primary: served.err(),
                observation: observed.err(),
                publication,
            }));
        }
    }
    served?;
    Ok(())
}
#[cfg(feature = "search-work-receipts")]
#[derive(Debug)]
struct SearchWorkFinishError {
    primary: Option<engine::EngineError>,
    observation: Option<rz_contracts::ContractError>,
    publication: Option<rz_uci::process_receipts::ProcessReceiptError>,
}
#[cfg(feature = "search-work-receipts")]
impl std::fmt::Display for SearchWorkFinishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "search finish failed: primary={:?}, observation={:?}, publication={:?}",
            self.primary, self.observation, self.publication
        )
    }
}
#[cfg(feature = "search-work-receipts")]
impl std::error::Error for SearchWorkFinishError {}

fn run_mock(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut explicit_mock = false;
    let mut model_adapter = None;
    let mut delay = Duration::ZERO;
    for argument in arguments {
        if argument == "--cpu-mock" {
            explicit_mock = true;
        } else if let Some(value) = argument.strip_prefix("--model-adapter=") {
            if model_adapter.replace(value.to_owned()).is_some() {
                return Err("duplicate model adapter".into());
            }
        } else if let Some(value) = argument.strip_prefix("--mock-delay-ms=") {
            delay = Duration::from_millis(value.parse()?);
        } else {
            return Err(
                "unsupported argument; supported flags: --cpu-mock [--mock-delay-ms=0..1000]"
                    .into(),
            );
        }
    }
    if !explicit_mock {
        return Err("this executable requires --cpu-mock; no neural backend is enabled".into());
    }
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1)); // Issued once for this executable process.
    match model_adapter.as_deref().unwrap_or("lc0") {
        "lc0" => {
            let factory = Arc::new(CpuMockFactory::new(&owners, delay)?);
            serve_process(
                EngineProcess::new(factory, owners, clock),
                EngineSettings::default(),
            )?;
        }
        "entity-candidate-mock" => {
            if !delay.is_zero() {
                return Err("entity mock does not support scripted delay".into());
            }
            let factory = Arc::new(rz_uci::bootstrap::EntityMockFactory::new(&owners)?);
            let process = EngineProcess::new(factory.clone(), owners, clock).with_identity(
                rz_uci::EngineIdentity {
                    name: "RoveZero entity candidate CPU mock".into(),
                    author: "RoveZero contributors".into(),
                },
            );
            let served = serve_process(process, EngineSettings::default());
            let finished = factory.finish(std::time::Instant::now() + Duration::from_secs(2));
            served?;
            finished?;
        }
        _ => return Err("unsupported model adapter".into()),
    }
    Ok(())
}

#[cfg(not(feature = "onnx-cpu"))]
fn run_native(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    // Do not inspect or echo native asset arguments when this provider is absent.
    if arguments.iter().any(|argument| argument == "--onnx-cuda") {
        return Err("unsupported feature: --onnx-cuda requires build feature onnx-cuda on Linux; no provider was started".into());
    }
    Err(
        "unsupported feature: --onnx-cpu requires build feature onnx-cpu; no provider was started"
            .into(),
    )
}

#[cfg(feature = "onnx-cpu")]
fn run_native(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    // NativeConfig owns exact flag/asset/hash validation and bounded public errors.
    let config = NativeConfig::parse(arguments)?;
    match config.provider() {
        NativeProvider::Cpu => run_cpu(config),
        NativeProvider::Cuda => run_cuda(config),
    }
}
#[cfg(feature = "onnx-cpu")]
fn run_cpu(config: NativeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let factory = Arc::new(NativeCpuFactory::load(&owners, &config)?);
    let trace = factory.source_trace();
    let mut profile_writer = if config.profiling_requested() {
        Some(ProfileWriter::open(&config)?)
    } else {
        None
    };
    let mut receipts = if config.attestation_requested() {
        Some(ReceiptWriter::open(&config)?)
    } else {
        None
    };
    let startup = if let Some(writer) = receipts.as_mut() {
        let startup = StartupReceiptV1::capture(&factory, &clock)?;
        writer.startup(&startup)?;
        Some(startup)
    } else {
        None
    };
    let process =
        EngineProcess::new(factory.clone(), owners, clock).with_identity(EngineIdentity {
            name: "RoveZero LC0 weights ONNX CPU integration".into(),
            author: "RoveZero contributors".into(),
        });
    let settings = config.engine_settings();
    // Retain the native owner outside EngineProcess. Report/drain acceptance runs
    // for both successful protocol service and EngineError; neither implies GPU support.
    let served = serve_native_process(process, settings, trace.clone());
    let finished = factory.finish(served);
    if let (Some(writer), Some(startup)) = (receipts.as_mut(), startup.as_ref()) {
        let receipt = match TerminationReceiptV1::from_result(startup, &finished) {
            Ok(receipt) => receipt,
            Err(error) => return Err(Box::new(error.retaining(finished))),
        };
        if let Err(error) = writer.termination(&receipt) {
            // Publication must not replace an original service/drain failure.
            return Err(Box::new(error.retaining(finished)));
        }
    }
    if let (Some(writer), Some(trace)) = (profile_writer.as_mut(), trace.as_ref())
        && let Err(error) = writer.publish(trace, &finished)
    {
        return Err(Box::new(error.retaining(finished)));
    }
    let report = finished?;
    eprintln!("{report}");
    Ok(())
}

#[cfg(all(
    feature = "onnx-cpu",
    not(all(feature = "onnx-cuda", target_os = "linux"))
))]
fn run_cuda(_config: NativeConfig) -> Result<(), Box<dyn std::error::Error>> {
    Err("explicit CUDA native startup requires the onnx-cuda feature on Linux; no provider was started".into())
}
#[cfg(all(feature = "onnx-cuda", target_os = "linux"))]
fn run_cuda(config: NativeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let factory = Arc::new(NativeCudaFactory::load(&owners, &config)?);
    let trace = factory.source_trace();
    #[cfg(feature = "experimental-batch")]
    let mut batch_writer = if config.batch_attestation_requested() {
        Some(rz_uci::native_batch_attestation::BatchReceiptWriter::open(
            &config,
        )?)
    } else {
        None
    };
    #[cfg(feature = "experimental-batch")]
    let batch_startup = if let Some(writer) = batch_writer.as_mut() {
        Some(writer.startup(
            CudaStartupReceiptV1::capture(&factory, &clock)?,
            &config.engine_settings(),
        )?)
    } else {
        None
    };
    let mut profile_writer = if config.profiling_requested() {
        Some(ProfileWriter::open(&config)?)
    } else {
        None
    };
    let mut receipts = if config.attestation_requested() {
        Some(CudaReceiptWriter::open(&config)?)
    } else {
        None
    };
    let startup = if let Some(writer) = receipts.as_mut() {
        let startup = CudaStartupReceiptV1::capture(&factory, &clock)?;
        writer.startup(&startup)?;
        // Capture the exact settings subsequently served. Experimental paths
        // do not issue this fixed-profile attestation (factory guard above).
        let search = rz_uci::native_cuda_attestation::CudaSearchReceiptV1::capture(
            &startup,
            &config.engine_settings(),
        )?;
        writer.search_config(&search)?;
        Some(startup)
    } else {
        None
    };
    let process =
        EngineProcess::new(factory.clone(), owners, clock).with_identity(EngineIdentity {
            name: "RoveZero LC0 weights ONNX CUDA integration".into(),
            author: "RoveZero contributors".into(),
        });
    let served = serve_native_process(process, config.engine_settings(), trace.clone());
    let finished = factory.finish(served);
    #[cfg(feature = "experimental-batch")]
    if let (Some(writer), Some(startup)) = (batch_writer.as_mut(), batch_startup.as_ref()) {
        if let Err(error) = writer.termination(startup, &factory, &finished) {
            return Err(Box::new(error.retaining(finished)));
        }
    }
    if let (Some(writer), Some(startup)) = (receipts.as_mut(), startup.as_ref()) {
        let receipt = match CudaTerminationReceiptV1::from_result(startup, &finished) {
            Ok(receipt) => receipt,
            Err(error) => return Err(Box::new(error.retaining(finished))),
        };
        if let Err(error) = writer.termination(&receipt) {
            return Err(Box::new(error.retaining(finished)));
        }
    }
    if let (Some(writer), Some(trace)) = (profile_writer.as_mut(), trace.as_ref())
        && let Err(error) = writer.publish(trace, &finished)
    {
        return Err(Box::new(error.retaining(finished)));
    }
    let report = finished?;
    eprintln!("{report}");
    Ok(())
}

fn serve_process(
    process: EngineProcess,
    settings: EngineSettings,
) -> Result<(), engine::EngineError> {
    let (sender, events) = engine::event_channel();
    let input = sender.clone();
    // A blocked OS stdin read is owned by this independent producer. It cannot
    // delay stop/search/drain, and process exit terminates it after quit.
    thread::spawn(move || {
        let _ = forward_lines(
            &mut io::stdin().lock(),
            &input,
            settings.parser.max_line_bytes,
        );
    });
    engine::serve(
        events,
        sender,
        &mut io::stdout().lock(),
        &mut io::stderr().lock(),
        process,
        settings,
    )
}

#[cfg(feature = "onnx-cpu")]
fn serve_native_process(
    process: EngineProcess,
    settings: EngineSettings,
    trace: Option<rz_telemetry::source::SourceJournal<rz_contracts::CompletionContext>>,
) -> Result<(), engine::EngineError> {
    let (sender, events) = engine::event_channel();
    let input = sender.clone();
    thread::spawn(move || {
        let _ = forward_lines(
            &mut io::stdin().lock(),
            &input,
            settings.parser.max_line_bytes,
        );
    });
    engine::serve(
        events,
        sender,
        &mut ProfiledWrite::new(io::stdout().lock(), trace),
        &mut io::stderr().lock(),
        process,
        settings,
    )
}
