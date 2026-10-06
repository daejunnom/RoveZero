//! V2 endpoints use the existing native snapshot/process/PGN lifecycle.
use crate::{
    ArenaError, NativeEngineView, NativeLaunchDeclaration, NativeLaunchOwner, NativePairView,
    NativeProviderDeclaration, NativeProviderSessionAudit,
};
use rz_experiments::{
    ArtifactRef, EngineEndpointV2, LockedManifestV2, NativeEngineRole, RoveLaunchV2,
};
use serde::Serialize;
use std::path::Path;

pub const STOCKFISH19_SOURCE_COMMIT: &str = "edb0d9db6731067ec50ce619ff372b463bc4dd5d";
/// The installer separately pins the official archive and resulting universal ELF.
pub fn stockfish19_endpoint(
    binary: ArtifactRef,
    role: NativeEngineRole,
) -> rz_experiments::ExternalUciEndpointV2 {
    use std::collections::BTreeMap;
    rz_experiments::ExternalUciEndpointV2 {
        id: "stockfish-19-t2-h256".into(),
        role,
        family: "stockfish".into(),
        version: "19".into(),
        expected_uci_name: "Stockfish 19".into(),
        binary,
        source: Some(rz_experiments::ExternalSourceV2 {
            url: "https://github.com/official-stockfish/Stockfish/tree/sf_19".into(),
            commit: STOCKFISH19_SOURCE_COMMIT.into(),
            license: "GPL-3.0-only".into(),
        }),
        arguments: vec![],
        assets: vec![],
        requested_options: BTreeMap::from([
            ("Threads".into(), "2".into()),
            ("Hash".into(), "256".into()),
            ("NumaPolicy".into(), "none".into()),
            ("Ponder".into(), "false".into()),
            ("MultiPV".into(), "1".into()),
            ("Skill Level".into(), "20".into()),
            ("UCI_LimitStrength".into(), "false".into()),
            ("UCI_Chess960".into(), "false".into()),
            ("Move Overhead".into(), "10".into()),
            ("nodestime".into(), "0".into()),
            ("SyzygyProbeLimit".into(), "0".into()),
        ]),
        handshake_timeout_ms: 30_000,
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "provider", content = "audit", rename_all = "snake_case")]
pub enum ModelEndpointSessionAudit {
    Cpu(NativeProviderSessionAudit),
    #[cfg(feature = "native-cuda")]
    Cuda(Box<crate::NativeCudaProviderSessionAudit>),
}
fn unsupported() -> ArenaError {
    ArenaError::Invalid("V2 endpoint has no supported executable recipe in this build".into())
}
impl crate::native_launch::sealed::Sealed for LockedManifestV2 {}
impl NativeLaunchDeclaration for LockedManifestV2 {
    fn rove_tree_max_edges(&self) -> Option<u32> {
        Some(self.input().rove_tree_max_edges)
    }
    fn input_sha256(&self) -> &str {
        self.sha256()
    }
    fn view(&self) -> NativePairView<'_> {
        let m = self.input();
        NativePairView {
            pair_id: &m.pair_id,
            white_order: m.white_order,
            opening: &m.opening,
            opening_artifact: &m.opening_artifact,
            runner: &m.runner,
            clock: rz_experiments::NativePairClock::Game(m.clock),
            max_plies: m.max_plies,
            timeouts: m.timeouts,
            budget: m.budget,
        }
    }
    fn declared_inputs(&self) -> Vec<&ArtifactRef> {
        self.input().declared_artifacts()
    }
    fn unique_bytes(&self) -> Result<u64, rz_experiments::ManifestError> {
        self.input().unique_input_bytes()
    }
    fn engine_view(
        &self,
        role: NativeEngineRole,
    ) -> Result<NativeEngineView<'_>, rz_experiments::ManifestError> {
        match self.input().engine(role)? {
            EngineEndpointV2::ExternalUci(e) => Ok(NativeEngineView {
                engine_id: &e.id,
                artifacts: &[],
                cuda_bundle: None,
                search: None,
                batch_experiment: None,
                external: Some(e),
            }),
            EngineEndpointV2::RoveZero(e) => {
                let launch = e.launch.as_ref().ok_or_else(|| {
                    rz_experiments::ManifestError::Integrity("unsupported V2 launch recipe".into())
                })?;
                let (artifacts, cuda_bundle, search) = match launch {
                    RoveLaunchV2::Lc0Cpu(l) => (l.artifacts.as_slice(), None, None),
                    RoveLaunchV2::Lc0CudaMaia(l) => {
                        (l.artifacts.as_slice(), Some(&l.cuda_bundle), None)
                    }
                    RoveLaunchV2::Lc0Cuda(l) => (
                        l.artifacts.as_slice(),
                        Some(&l.cuda_bundle),
                        Some(l.profile.search),
                    ),
                };
                Ok(NativeEngineView {
                    engine_id: &e.id,
                    artifacts,
                    cuda_bundle,
                    search,
                    batch_experiment: None,
                    external: None,
                })
            }
        }
    }
    fn seed(&self) -> u64 {
        self.input().seed
    }
    fn provider_name(&self) -> &'static str {
        "V2-endpoints"
    }
    fn validate_execution(&self) -> Result<(), ArenaError> {
        self.input().validate()?;
        if !matches!(self.expected_provider_sessions(), 2 | 4) {
            return Err(unsupported());
        }
        let mut bundle = None;
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            let view = self.engine_view(role)?;
            if let Some(cuda) = view.cuda_bundle {
                if !cfg!(feature = "native-cuda") {
                    return Err(unsupported());
                }
                if bundle.is_some_and(|prior| prior != cuda) {
                    return Err(ArenaError::Invalid("this V2 executor requires one identical immutable CUDA runtime bundle; differing backend packages require a separate supported recipe".into()));
                }
                bundle = Some(cuda);
            }
        }
        Ok(())
    }

    fn advise_drop_input_cache(&self) -> bool {
        true
    }
    fn additional_manifest(&self) -> Option<&ArtifactRef> {
        self.input().engines.iter().find_map(|e| match e {
            EngineEndpointV2::RoveZero(e) => match e.launch.as_ref() {
                Some(RoveLaunchV2::Lc0Cuda(l)) => Some(&l.cuda_bundle.manifest),
                Some(RoveLaunchV2::Lc0CudaMaia(l)) => Some(&l.cuda_bundle.manifest),
                _ => None,
            },
            _ => None,
        })
    }
    fn validate_additional_manifest(&self, bytes: &[u8]) -> Result<(), ArenaError> {
        let bundle = self
            .input()
            .engines
            .iter()
            .find_map(|e| match e {
                EngineEndpointV2::RoveZero(e) => match e.launch.as_ref() {
                    Some(RoveLaunchV2::Lc0Cuda(l)) => Some(&l.cuda_bundle),
                    Some(RoveLaunchV2::Lc0CudaMaia(l)) => Some(&l.cuda_bundle),
                    _ => None,
                },
                _ => None,
            })
            .ok_or_else(unsupported)?;
        #[cfg(feature = "native-cuda")]
        {
            crate::validate_cuda_bundle_manifest_fields(bytes, bundle)
        }
        #[cfg(not(feature = "native-cuda"))]
        {
            let _ = (bytes, bundle);
            Err(unsupported())
        }
    }
}
impl NativeProviderDeclaration for LockedManifestV2 {
    type Audit = ModelEndpointSessionAudit;
    fn scope(&self) -> &'static str {
        "v2_model_endpoints_paired_uci_process_provider_clock_and_rules"
    }
    fn receipt_filename(&self) -> &'static str {
        "model-endpoints-pair-receipt.v2.json"
    }
    fn receipt_version(&self) -> u32 {
        2
    }
    fn expected_provider_sessions(&self) -> usize {
        2 * self
            .input()
            .engines
            .iter()
            .filter(|e| matches!(e, EngineEndpointV2::RoveZero(_)))
            .count()
    }
    fn startup_filename(&self) -> &'static str {
        "native-cpu-startup.v1.json"
    }
    fn termination_filename(&self) -> &'static str {
        "native-cpu-termination.v1.json"
    }
    fn role_startup_filename(&self, role: NativeEngineRole) -> &'static str {
        if self
            .engine_view(role)
            .is_ok_and(|e| e.cuda_bundle.is_some())
        {
            "native-cuda-startup.v1.json"
        } else {
            self.startup_filename()
        }
    }
    fn role_termination_filename(&self, role: NativeEngineRole) -> &'static str {
        if self
            .engine_view(role)
            .is_ok_and(|e| e.cuda_bundle.is_some())
        {
            "native-cuda-termination.v1.json"
        } else {
            self.termination_filename()
        }
    }
    fn claim_policy(&self) -> rz_experiments::ClaimPolicy {
        rz_experiments::ClaimPolicy::AutomaticAcceptance
    }
    fn validate_clock_trace(
        &self,
        stdout: &[u8],
        pgn: &crate::PairPgnAudit,
    ) -> Result<Option<crate::NativePilotClockAudit>, ArenaError> {
        crate::native_pilot::validate_pilot_clock_trace(
            stdout,
            pgn,
            &self.input().opening,
            self.input().clock,
        )
        .map(Some)
    }
    fn audit_failure_trace(
        &self,
        stdout: &[u8],
        pair: &crate::PairSpec,
    ) -> Result<Option<crate::NativePilotFailureAudit>, ArenaError> {
        crate::native_pilot::audit_pilot_startup_failure_trace(stdout, pair)
    }
    #[cfg(target_os = "linux")]
    fn validate_records(
        &self,
        role: NativeEngineRole,
        startup: &[u8],
        termination: &[u8],
        session: &str,
    ) -> Result<(Self::Audit, u32), ArenaError> {
        let EngineEndpointV2::RoveZero(e) = self.input().engine(role)? else {
            return Err(unsupported());
        };
        match e.launch.as_ref().ok_or_else(unsupported)? {
            RoveLaunchV2::Lc0Cpu(l) => {
                let a = crate::validate_native_provider_record_fields(
                    startup,
                    termination,
                    l,
                    session,
                )?;
                let pid = a.process_id;
                Ok((ModelEndpointSessionAudit::Cpu(a), pid))
            }
            #[cfg(feature = "native-cuda")]
            RoveLaunchV2::Lc0Cuda(l) => {
                let a = crate::validate_native_cuda_provider_record_fields(
                    startup,
                    termination,
                    l,
                    session,
                )?;
                let pid = a.process_id;
                Ok((ModelEndpointSessionAudit::Cuda(Box::new(a)), pid))
            }
            #[cfg(feature = "native-cuda")]
            RoveLaunchV2::Lc0CudaMaia(l) => {
                let a = crate::validate_native_cuda_provider_record_fields(
                    startup,
                    termination,
                    l,
                    session,
                )?;
                let pid = a.process_id;
                Ok((ModelEndpointSessionAudit::Cuda(Box::new(a)), pid))
            }
            #[cfg(not(feature = "native-cuda"))]
            _ => Err(unsupported()),
        }
    }
    #[cfg(target_os = "linux")]
    fn verify_session_evidence(
        &self,
        role: NativeEngineRole,
        pid: u32,
        startup: &[u8],
        directory: &cap_std::fs::Dir,
    ) -> Result<Vec<(String, Vec<u8>)>, ArenaError> {
        let EngineEndpointV2::RoveZero(e) = self.input().engine(role)? else {
            return Err(unsupported());
        };
        match e.launch.as_ref().ok_or_else(unsupported)? {
            RoveLaunchV2::Lc0Cpu(_) => Ok(Vec::new()),
            #[cfg(feature = "native-cuda")]
            RoveLaunchV2::Lc0Cuda(l) => {
                crate::native_cuda::verify_cuda_endpoint_evidence_with_tree_limit(
                    l,
                    pid,
                    startup,
                    directory,
                    self.input().rove_tree_max_edges,
                )
            }
            #[cfg(feature = "native-cuda")]
            RoveLaunchV2::Lc0CudaMaia(l) => {
                crate::native_cuda::verify_cuda_endpoint_evidence_with_tree_limit(
                    l,
                    pid,
                    startup,
                    directory,
                    self.input().rove_tree_max_edges,
                )
            }
            #[cfg(not(feature = "native-cuda"))]
            _ => {
                let _ = (pid, startup, directory);
                Err(unsupported())
            }
        }
    }
    #[cfg(target_os = "linux")]
    fn external_exit_ids(&self, stdout: &[u8], pids: &[u32]) -> Result<Vec<u32>, ArenaError> {
        crate::native_exit::validate_endpoint_exit_trace(
            stdout,
            pids,
            4 - self.expected_provider_sessions(),
        )
    }
    #[cfg(target_os = "linux")]
    fn validate_process_exit_trace(&self, stdout: &[u8], pids: &[u32]) -> Result<(), ArenaError> {
        crate::native_exit::validate_endpoint_exit_trace(
            stdout,
            pids,
            4 - self.expected_provider_sessions(),
        )
        .map(|_| ())
    }
}
/// A metadata lock is not execution authority. This supported recipe checks
/// both endpoint declarations before allocating the private snapshot tree.
pub fn prepare_model_endpoint_launch(
    spec: &LockedManifestV2,
    source_root: &Path,
    output_root: &Path,
    label: &str,
) -> Result<NativeLaunchOwner<LockedManifestV2>, Box<crate::NativePreparationFailure>> {
    crate::native_launch::prepare_native_launch_for(spec, source_root, output_root, label)
}

pub fn run_model_endpoint_pair(
    owner: NativeLaunchOwner<LockedManifestV2>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<
    crate::NativePairOutput<LockedManifestV2>,
    Box<crate::NativePairFailure<LockedManifestV2>>,
> {
    crate::native_runner::run_native_pair_for(owner, cancel)
}
