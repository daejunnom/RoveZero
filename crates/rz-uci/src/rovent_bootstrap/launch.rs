//! Separate Rovent UCI entry. No legacy V5 DTO, command-line attestation or
//! fabricated weights path participates in this source-bound native assembly.

use super::configuration::{ExecutionConfiguration, ResearchValueConfiguration};
use super::nnuec_source::LoadedRoventNnuecSource;
use super::process_boundary::verify_actual_process_boundary;
use super::product::{
    RoventNativeProductDriver, RoventProductStartupOwners, RoventProductStartupRejection,
};
use super::{
    RoventNativeAssetSelection, RoventNativeGraphAsset, RoventNativeResources, RoventNativeService,
    RoventNativeStartupRejection, RoventPinnedNativeSources, RoventStartupControl, prepare_native,
};
use crate::pals_adaptive::NativeQuestionProjection;
use crate::pals_adaptive_bootstrap::protocol::serve_driver;
use crate::pals_native::ticketed::TicketNativeEvent;
use crate::rovent_root_admission::RoventRootAdmissionPolicy;
use crate::search_driver::SearchSessionDriver;
use crate::{
    EngineIdentity,
    engine::{EngineSettings, OwnerRegistry, ProcessClock},
};
use cap_fs_ext::DirExt;
use cap_std::fs::Dir;
use rz_contracts::ProcessEpoch;
use rz_eval::pals_onnx::virtual_review::VirtualReviewProfileDirectory;
use rz_experiments::{ArtifactRef, RoventDevice, RoventInputLock};
use rz_search::pals::engine::{adaptive_port::AdaptivePortQuestion, pc_port::PcPortQuestion};
use serde::Deserialize;
use std::{
    error::Error,
    fmt,
    fs::File,
    io::Read,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

const SELECTION_DOMAIN: &str = "rz-rovent-native-assets/1";
const MAX_LOCK_BYTES: u64 = 1 << 20;
const MAX_SELECTION_BYTES: u64 = 64 << 10;

#[derive(Debug)]
pub struct RoventLaunchError {
    pub stage: &'static str,
    pub detail: String,
}
impl fmt::Display for RoventLaunchError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(output, "Rovent {}: {}", self.stage, self.detail)
    }
}
impl Error for RoventLaunchError {}
fn error(stage: &'static str, detail: impl fmt::Display) -> RoventLaunchError {
    let mut detail = detail.to_string();
    if detail.len() > 512 {
        let mut end = 512;
        while !detail.is_char_boundary(end) {
            end -= 1;
        }
        detail.truncate(end);
    }
    RoventLaunchError { stage, detail }
}

#[derive(Debug, Eq, PartialEq)]
struct Arguments {
    input_lock: String,
    native_assets: String,
    process_epoch: u64,
    startup_ms: u64,
    root_policy: RoventRootAdmissionPolicy,
}
impl Arguments {
    fn parse(arguments: &[String]) -> Result<Self, RoventLaunchError> {
        let mut input_lock = None;
        let mut native_assets = None;
        let mut process_epoch = None;
        let mut startup_ms = None;
        let mut root_policy = None;
        for argument in arguments {
            let (name, value) = argument
                .split_once('=')
                .ok_or_else(|| error("Arguments", "every option requires its explicit =value"))?;
            match name {
                "--input-lock" if input_lock.is_none() => input_lock = Some(private_path(value)?),
                "--native-assets" if native_assets.is_none() => {
                    native_assets = Some(private_path(value)?)
                }
                "--process-epoch" if process_epoch.is_none() => {
                    process_epoch = Some(number(value, u64::MAX)?)
                }
                "--startup-ms" if startup_ms.is_none() => {
                    startup_ms = Some(number(value, 3_600_000)?)
                }
                "--root-policy" if root_policy.is_none() => {
                    root_policy = Some(match value {
                        "exact-registered" => RoventRootAdmissionPolicy::ExactRegistered,
                        "dynamic-go" => RoventRootAdmissionPolicy::DynamicGo,
                        _ => {
                            return Err(error(
                                "Arguments",
                                "unknown explicit root admission policy",
                            ));
                        }
                    })
                }
                _ => return Err(error("Arguments", "unknown or repeated option")),
            }
        }
        Ok(Self {
            input_lock: input_lock.ok_or_else(|| error("Arguments", "--input-lock is required"))?,
            native_assets: native_assets
                .ok_or_else(|| error("Arguments", "--native-assets is required"))?,
            process_epoch: process_epoch
                .ok_or_else(|| error("Arguments", "--process-epoch is required"))?,
            startup_ms: startup_ms.ok_or_else(|| error("Arguments", "--startup-ms is required"))?,
            root_policy: root_policy
                .ok_or_else(|| error("Arguments", "--root-policy is required"))?,
        })
    }
}
fn private_path(value: &str) -> Result<String, RoventLaunchError> {
    if value.len() > 4096 || !rz_experiments::pals_private_snapshot_path_v5(value) {
        return Err(error(
            "Arguments",
            "input must use the closed inputs snapshot namespace",
        ));
    }
    Ok(value.into())
}
fn number(value: &str, maximum: u64) -> Result<u64, RoventLaunchError> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(error("Arguments", "integer syntax"));
    }
    let number: u64 = value
        .parse()
        .map_err(|_| error("Arguments", "integer overflow"))?;
    if number == 0 || number > maximum {
        return Err(error("Arguments", "integer bound"));
    }
    Ok(number)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectionWire {
    domain: String,
    startup_configuration: ArtifactRef,
    export_manifest: ArtifactRef,
    rules_descriptor: ArtifactRef,
    graphs: Vec<GraphWire>,
    cuda_control_inventory: Option<ArtifactRef>,
    runtime_bundle_manifest: Option<ArtifactRef>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphWire {
    role: String,
    registered_file: String,
    artifact: ArtifactRef,
}
impl SelectionWire {
    fn parse(bytes: &[u8]) -> Result<RoventNativeAssetSelection, RoventLaunchError> {
        let wire: Self = serde_json::from_slice(bytes)
            .map_err(|_| error("Assets", "strict native asset selection JSON"))?;
        if wire.domain != SELECTION_DOMAIN || wire.graphs.len() != 2 {
            return Err(error(
                "Assets",
                "native selection domain or graph cardinality",
            ));
        }
        Ok(RoventNativeAssetSelection {
            startup_configuration: wire.startup_configuration,
            export_manifest: wire.export_manifest,
            rules_descriptor: wire.rules_descriptor,
            graphs: wire
                .graphs
                .into_iter()
                .map(|graph| RoventNativeGraphAsset {
                    role: graph.role,
                    registered_file: graph.registered_file,
                    artifact: graph.artifact,
                })
                .collect(),
            cuda_control_inventory: wire.cuda_control_inventory,
            runtime_bundle_manifest: wire.runtime_bundle_manifest,
        })
    }
}

/// The caller passes only process arguments. No account, token or external
/// credential file is inspected by this native UCI entry.
pub fn run(arguments: Vec<String>) -> Result<(), RoventLaunchError> {
    if arguments == ["--build-capabilities-json"] {
        let sources = super::process_boundary::compiled_process_source_components()
            .map_err(|failure| error("CompiledSources", failure))?;
        println!(
            "{}",
            serde_json::json!({
                "domain":"rz-rovent-build-capabilities/1",
                "source_commit":env!("RZ_PALS_COMPILED_SOURCE_COMMIT"),
                "source_components":sources,
                "target_os":std::env::consts::OS,
                "target_arch":std::env::consts::ARCH,
                "onnx_cpu_compiled":cfg!(feature="onnx-cpu"),
                "onnx_cuda_compiled":cfg!(feature="onnx-cuda"),
                "native_runtime_loaded":false,
                "native_model_loaded":false,
                "neural_execution_observed":false,
            })
        );
        return Ok(());
    }
    let arguments = Arguments::parse(&arguments)?;
    let cwd = std::env::current_dir().map_err(|failure| error("Cwd", failure))?;
    let root = Dir::open_ambient_dir(&cwd, cap_std::ambient_authority())
        .map_err(|failure| error("Cwd", failure))?;
    let lock_file = root
        .open_with(
            &arguments.input_lock,
            cap_std::fs::OpenOptions::new().read(true),
        )
        .map_err(|failure| error("InputLock", failure))?
        .into_std();
    let lock_bytes = read_bounded(&lock_file, MAX_LOCK_BYTES)?;
    let lock = Arc::new(
        RoventInputLock::from_json(
            std::str::from_utf8(&lock_bytes)
                .map_err(|_| error("InputLock", "UTF-8 input required"))?,
        )
        .map_err(|failure| error("InputLock", failure))?,
    );
    if arguments.startup_ms > lock.manifest.wall_time_max_ms {
        return Err(error(
            "StartupWindow",
            "startup exceeds the original declared window",
        ));
    }
    let begin = Instant::now();
    let clock = ProcessClock::new(ProcessEpoch(arguments.process_epoch));
    let until = begin
        .checked_add(Duration::from_millis(arguments.startup_ms))
        .ok_or_else(|| error("StartupWindow", "startup deadline overflow"))?;
    let cleanup_until = until
        .checked_add(Duration::from_millis(lock.manifest.cleanup_max_ms))
        .ok_or_else(|| error("StartupWindow", "cleanup deadline overflow"))?;
    let control = RoventStartupControl {
        until,
        cleanup_until,
        cancellation: Arc::new(AtomicBool::new(false)),
    };
    let binary = lock
        .manifest
        .engine
        .binary
        .open_verified_at(&root, 128 << 20)
        .map_err(|failure| error("Binary", failure))?;
    let boundary = verify_actual_process_boundary(
        &lock,
        &lock.manifest.engine.binary,
        &binary,
        &cwd,
        &root,
        &control,
    )
    .map_err(|failure| error("ProcessBoundary", failure))?;
    let selected: Vec<_> = lock
        .manifest
        .provenance
        .iter()
        .filter(|artifact| artifact.path == arguments.native_assets)
        .collect();
    if selected.len() != 1 {
        return Err(error(
            "Assets",
            "selection is not a unique original provenance artifact",
        ));
    }
    let selection_file = selected[0]
        .open_verified_at(&root, MAX_SELECTION_BYTES)
        .map_err(|failure| error("Assets", failure))?;
    let selection = SelectionWire::parse(&read_bounded(&selection_file, MAX_SELECTION_BYTES)?)?;
    let sources = Arc::new(
        RoventPinnedNativeSources::open(
            Arc::clone(&lock),
            root.try_clone().map_err(|failure| error("Cwd", failure))?,
            cwd.clone(),
            selection,
            &control,
        )
        .map_err(|failure| error("Assets", failure))?,
    );
    sources
        .configuration()
        .require_supported_native_capabilities()
        .map_err(|failure| error("Capability", failure))?;
    match sources.configuration().fields().execution {
        ExecutionConfiguration::RolesOnly { .. } => {}
        ExecutionConfiguration::OwnResearcher { .. } => {}
        ExecutionConfiguration::ExternalResearcher { .. } => {
            return Err(error(
                "Capability",
                "external-R product assembly is not yet admitted by this entry",
            ));
        }
    }
    let cuda_profile = if matches!(lock.manifest.engine.pc.device, RoventDevice::Cuda { .. }) {
        let name = format!(
            "rovent-cuda-profile-{}-{}",
            std::process::id(),
            arguments.process_epoch
        );
        root.create_dir(&name)
            .map_err(|failure| error("ProfileDirectory", failure))?;
        let directory = root
            .open_dir_nofollow(&name)
            .map_err(|failure| error("ProfileDirectory", failure))?;
        Some(
            VirtualReviewProfileDirectory::from_owned_directory(directory.into_std_file())
                .map_err(|failure| error("ProfileDirectory", failure))?,
        )
    } else {
        None
    };
    let configuration = sources.configuration();
    let resources = RoventNativeResources {
        runtime: sources
            .pin_runtime()
            .map_err(|failure| error("Runtime", failure))?,
        configuration: configuration
            .fields()
            .native
            .onnx(&lock.manifest.engine.pc.device, cuda_profile.is_some())
            .map_err(|failure| error("Configuration", failure))?,
        loading_profile: configuration
            .fields()
            .native
            .loading_profile
            .map(|profile| profile.actual()),
        cuda_profile,
        drain_limit: Duration::from_millis(configuration.fields().native.drain_ms),
        tickets: configuration
            .fields()
            .native
            .tickets(configuration.fields().review.token_budget)
            .map_err(|failure| error("Configuration", failure))?,
        command_capacity: configuration.fields().native.command_capacity,
    };
    control
        .check()
        .map_err(|failure| error("StartupWindow", failure))?;
    let loaded_nnuec = if matches!(
        sources.configuration().fields().execution,
        ExecutionConfiguration::OwnResearcher {
            value: ResearchValueConfiguration::NnuecV2 { .. },
            ..
        }
    ) {
        Some(
            LoadedRoventNnuecSource::load(Arc::clone(&sources), &control)
                .map_err(|failure| error("NnuecSource", failure))?,
        )
    } else {
        None
    };
    let owners = Arc::new(OwnerRegistry::default());
    let identity = EngineIdentity {
        name: format!("Rovent {:?}", lock.manifest.engine.mode),
        author: "Rovent contributors".into(),
    };
    let driver = if lock.manifest.engine.mode.has_native_r() {
        let prepared = prepare_native::<AdaptivePortQuestion>(
            Arc::clone(&sources),
            resources,
            clock.runtime_clock(),
            control.clone(),
        )
        .map_err(|failure| reject_factory(failure, cleanup_until))?;
        match loaded_nnuec {
            Some(loaded) => RoventNativeProductDriver::own_nnuec_pcr(
                prepared,
                loaded,
                clock.clone(),
                Arc::clone(&owners),
                identity,
                arguments.root_policy,
            ),
            None => RoventNativeProductDriver::own_board_pcr(
                prepared,
                clock.clone(),
                Arc::clone(&owners),
                identity,
                arguments.root_policy,
            ),
        }
    } else {
        let prepared = prepare_native::<PcPortQuestion>(
            Arc::clone(&sources),
            resources,
            clock.runtime_clock(),
            control.clone(),
        )
        .map_err(|failure| reject_factory(failure, cleanup_until))?;
        RoventNativeProductDriver::pc(
            prepared,
            clock.clone(),
            Arc::clone(&owners),
            identity,
            arguments.root_policy,
        )
    }
    .map_err(|failure| reject_product(failure, cleanup_until))?;
    let mut settings = EngineSettings::default();
    settings.shutdown_limit = Duration::from_millis(lock.manifest.cleanup_max_ms);
    settings.search.untimed_limit = Duration::from_millis(lock.manifest.wall_time_max_ms);
    let driver: Arc<dyn SearchSessionDriver> = Arc::new(driver);
    // Keep original startup lock/selection/binary/cwd descriptors alive through
    // the same UCI whole-session physical finish, rather than a detached SSH job.
    let result = serve_driver(driver, clock, owners, settings)
        .map_err(|failure| error("UciSession", failure));
    drop(boundary);
    drop((selection_file, binary, lock_file, root));
    result
}

fn read_bounded(file: &File, maximum: u64) -> Result<Vec<u8>, RoventLaunchError> {
    let metadata = file
        .metadata()
        .map_err(|failure| error("BoundedInput", failure))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > maximum {
        return Err(error("BoundedInput", "not a bounded regular input"));
    }
    let mut bytes = Vec::new();
    (&*file)
        .take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|failure| error("BoundedInput", failure))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(error("BoundedInput", "input changed while reading"));
    }
    Ok(bytes)
}
fn reject_factory<Q: NativeQuestionProjection>(
    mut failure: RoventNativeStartupRejection<Q>,
    until: Instant,
) -> RoventLaunchError {
    let detail = error("NativeStartup", &failure.error);
    if failure
        .native
        .as_mut()
        .is_some_and(|native| !drain_native(native, until))
    {
        let _quarantine = Box::leak(Box::new(failure));
        return error("PhysicalCompletionUnknown", detail);
    }
    detail
}
fn reject_product(mut failure: RoventProductStartupRejection, until: Instant) -> RoventLaunchError {
    let detail = error("ProductAssembly", &failure.failure);
    let confirmed = match &mut failure.owners {
        RoventProductStartupOwners::Pc(prepared) => drain_native(&mut prepared.service, until),
        RoventProductStartupOwners::Pcr(prepared) => drain_native(&mut prepared.service, until),
        RoventProductStartupOwners::NnuecRejected(rejected) => {
            drain_native(&mut rejected.prepared.service, until)
        }
        RoventProductStartupOwners::PcrRejected(rejected) => {
            drain_native(&mut rejected.sources.native, until)
        }
        RoventProductStartupOwners::PcrDriver(driver) => {
            driver.finish_session(until).is_ok_and(|completion| {
                completion == Some(crate::search_driver::DriverPhysicalCompletion::Confirmed)
            })
        }
    };
    if !confirmed {
        let _quarantine = Box::leak(Box::new(failure));
        return error("PhysicalCompletionUnknown", detail);
    }
    detail
}
fn drain_native<Q: NativeQuestionProjection>(
    native: &mut RoventNativeService<Q>,
    until: Instant,
) -> bool {
    native.begin_shutdown(until);
    let mut finished = None;
    let mut retained = Vec::new();
    loop {
        let signal = native.signal();
        let observed = signal.version();
        while let Some(event) = native.poll() {
            match event {
                TicketNativeEvent::Finished {
                    physical_shutdown_confirmed,
                } => finished = Some(finished.unwrap_or(true) && physical_shutdown_confirmed),
                TicketNativeEvent::Ready { .. } | TicketNativeEvent::RootQuiesced(_) => {}
                other => retained.push(other),
            }
        }
        let joined = native.try_join();
        if matches!(joined, Ok(true)) && finished == Some(true) {
            return true;
        }
        if joined.is_err() || finished == Some(false) || Instant::now() >= until {
            // Unexpected payloads retain their original leases on unknown
            // completion; no ordinary startup error frees them speculatively.
            std::mem::forget(retained);
            return false;
        }
        signal.wait_changed(
            observed,
            until
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(10)),
        );
    }
}

#[cfg(test)]
#[path = "launch/tests.rs"]
mod tests;
