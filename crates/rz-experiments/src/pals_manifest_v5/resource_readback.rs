//! Item-level resource observations. Unknown readings stay unknown and a
//! cloud functional run does not thereby prove the strict local resource plan.
use super::*;

choices!(PalsResourceEvidenceProfileV5 {
    StrictLocal,
    CloudObserved
});
choices!(PalsCpuAffinityScopeV5 {
    Process,
    ThreadGroupLeader,
    Unknown
});
choices!(PalsCpuQuotaScopeV5 {
    ProcessTree,
    VisibleAncestors,
    ImmediateJobObject
});

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsCpuQuotaReadbackV5 {
    Limited {
        quota_us: u64,
        period_us: u64,
        scope: PalsCpuQuotaScopeV5,
    },
    Unlimited {
        scope: PalsCpuQuotaScopeV5,
    },
    /// Windows CPU hard-cap rate describes the current Job Object. It does
    /// not have Linux cgroup millicpu equivalence or establish nested Job limits.
    WindowsHardCap {
        rate_per_10000: u32,
        scope: PalsCpuQuotaScopeV5,
    },
}
impl PalsCpuQuotaReadbackV5 {
    fn validate(&self) -> Result<(), ManifestError> {
        require(
            match self {
                Self::Limited {
                    quota_us,
                    period_us,
                    ..
                } => *quota_us > 0 && *period_us > 0,
                Self::Unlimited { .. } => true,
                Self::WindowsHardCap {
                    rate_per_10000,
                    scope,
                } => {
                    (1..=10_000).contains(rate_per_10000)
                        && *scope == PalsCpuQuotaScopeV5::ImmediateJobObject
                }
            },
            "invalid exact CPU quota observation",
        )
    }
    pub fn matches_millicpus(&self, requested: u32) -> bool {
        matches!(self, Self::Limited { quota_us, period_us, scope: PalsCpuQuotaScopeV5::ProcessTree }
            if *quota_us > 0 && *period_us > 0
                && u128::from(*quota_us) * 1000 == u128::from(requested) * u128::from(*period_us))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsMemoryLimitReadbackV5 {
    Limited { bytes: u64 },
    Unlimited,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
#[serde(tag = "actor", rename_all = "snake_case", deny_unknown_fields)]
pub enum PalsCpuActorV5 {
    Manager,
    NativeCaller,
    OwnWorker { index: u32 },
    ExternalEngine { index: u32 },
    RuntimeHelper { index: u32 },
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsThreadAffinityReadbackV5 {
    /// Multiple sequential responsibilities still count as one physical thread.
    pub actors: BTreeSet<PalsCpuActorV5>,
    pub process_id: u64,
    pub thread_id: u64,
    pub affinity: PalsObservedV3<Vec<u32>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsOrtSettingsReadbackV5 {
    pub intra_threads: u32,
    pub inter_threads: u32,
    pub spinning: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsGpuIdentityReadbackV5 {
    /// Registered model identity, separate from the driver's literal name.
    pub model_id: String,
    pub reported_name: String,
    pub uuid: String,
    pub total_bytes: u64,
    pub device_count: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsResourceReadbackV5 {
    pub profile: PalsResourceEvidenceProfileV5,
    /// A read-only topology probe returns false/unknown here; it never installs
    /// cgroups, Job Objects, affinity, or ORT settings by observing them.
    pub limits_applied: PalsObservedV3<bool>,
    pub cpu_quota: PalsObservedV3<PalsCpuQuotaReadbackV5>,
    pub quota_ancestor_chain_complete: PalsObservedV3<bool>,
    pub allowed_logical_cpus: PalsObservedV3<Vec<u32>>,
    pub affinity_scope: PalsCpuAffinityScopeV5,
    pub thread_inventory_complete: PalsObservedV3<bool>,
    pub threads: Vec<PalsThreadAffinityReadbackV5>,
    pub memory_high: PalsObservedV3<PalsMemoryLimitReadbackV5>,
    pub memory_max: PalsObservedV3<PalsMemoryLimitReadbackV5>,
    pub ort_settings: PalsObservedV3<PalsOrtSettingsReadbackV5>,
    pub smt_sibling_exclusion_verified: PalsObservedV3<bool>,
    /// Snapshot count has explicit scope; it is not a lifetime peak claim.
    pub total_observed_kernel_threads: PalsObservedV3<u32>,
    pub gpu_identity: PalsObservedV3<PalsGpuIdentityReadbackV5>,
    pub cuda_provider_ready: PalsObservedV3<bool>,
    pub evidence: ArtifactRef,
}
impl PalsResourceReadbackV5 {
    pub fn with_unknown_checks(
        profile: PalsResourceEvidenceProfileV5,
        evidence: ArtifactRef,
    ) -> Self {
        Self {
            profile,
            limits_applied: PalsObservedV3::Unknown,
            cpu_quota: PalsObservedV3::Unknown,
            quota_ancestor_chain_complete: PalsObservedV3::Unknown,
            allowed_logical_cpus: PalsObservedV3::Unknown,
            affinity_scope: PalsCpuAffinityScopeV5::Unknown,
            thread_inventory_complete: PalsObservedV3::Unknown,
            threads: Vec::new(),
            memory_high: PalsObservedV3::Unknown,
            memory_max: PalsObservedV3::Unknown,
            ort_settings: PalsObservedV3::Unknown,
            smt_sibling_exclusion_verified: PalsObservedV3::Unknown,
            total_observed_kernel_threads: PalsObservedV3::Unknown,
            gpu_identity: PalsObservedV3::Unknown,
            cuda_provider_ready: PalsObservedV3::Unknown,
            evidence,
        }
    }

    fn validate_observations(&self) -> Result<(), ManifestError> {
        self.evidence.validate()?;
        shared::observation(&self.limits_applied)?;
        shared::observation(&self.cpu_quota)?;
        shared::observation(&self.quota_ancestor_chain_complete)?;
        shared::observation(&self.allowed_logical_cpus)?;
        shared::observation(&self.thread_inventory_complete)?;
        shared::observation(&self.memory_high)?;
        shared::observation(&self.memory_max)?;
        shared::observation(&self.ort_settings)?;
        shared::observation(&self.smt_sibling_exclusion_verified)?;
        shared::observation(&self.total_observed_kernel_threads)?;
        shared::observation(&self.gpu_identity)?;
        shared::observation(&self.cuda_provider_ready)?;
        if let Some(q) = observed(&self.cpu_quota) {
            q.validate()?;
        }
        if let Some(cpus) = observed(&self.allowed_logical_cpus) {
            require(
                !cpus.is_empty() && cpus.len() <= 4096 && unique_cpus(cpus),
                "invalid CPU set readback",
            )?;
        }
        for limit in [&self.memory_high, &self.memory_max] {
            if let Some(PalsMemoryLimitReadbackV5::Limited { bytes }) = observed(limit) {
                require(*bytes > 0, "zero memory limit is not an unlimited readback")?;
            }
        }
        if let Some(ort) = observed(&self.ort_settings) {
            require(
                ort.intra_threads > 0 && ort.inter_threads > 0,
                "invalid applied ORT thread settings",
            )?;
        }
        if let Some(count) = observed(&self.total_observed_kernel_threads) {
            require(
                (1..=1024).contains(count) && self.threads.len() <= *count as usize,
                "invalid bounded ready thread inventory",
            )?;
        }
        require(
            observed(&self.thread_inventory_complete) != Some(&true)
                || observed(&self.total_observed_kernel_threads)
                    .is_some_and(|count| self.threads.len() == *count as usize),
            "complete ready inventory omits observed threads",
        )?;
        require(
            self.threads.len() <= 1024,
            "thread inventory exceeds bounded readback",
        )?;
        let mut entries = BTreeSet::new();
        let mut actors = BTreeSet::new();
        for thread in &self.threads {
            require(
                thread.process_id > 0
                    && thread.thread_id > 0
                    && entries.insert((thread.process_id, thread.thread_id))
                    && !thread.actors.is_empty()
                    && thread.actors.len() <= 16
                    && thread
                        .actors
                        .iter()
                        .filter(|actor| matches!(actor, PalsCpuActorV5::OwnWorker { .. }))
                        .count()
                        <= 1
                    && thread.actors.iter().all(|actor| actors.insert(*actor)),
                "duplicate or invalid thread identity",
            )?;
            shared::observation(&thread.affinity)?;
            if let Some(cpus) = observed(&thread.affinity) {
                require(
                    !cpus.is_empty() && cpus.len() <= 4096 && unique_cpus(cpus),
                    "invalid thread affinity",
                )?;
            }
        }
        if let Some(gpu) = observed(&self.gpu_identity) {
            require(
                text(&gpu.model_id)
                    && text(&gpu.reported_name)
                    && text(&gpu.uuid)
                    && gpu.total_bytes > 0
                    && gpu.device_count > 0,
                "invalid GPU identity readback",
            )?;
        }
        Ok(())
    }

    /// Full, matching applied resource evidence. It excludes rounded quotas,
    /// hidden ancestors, a group-leader mask presented as all thread affinity,
    /// and unobserved settings. Cloud functional acceptance may lack this proof.
    pub fn matches_applied_plan(&self, d: &PalsEngineV5) -> bool {
        if self.validate_observations().is_err() || !self.manager_roles_compatible(d) {
            return false;
        }
        let plan = &d.resources;
        let Some(cpus) = observed(&self.allowed_logical_cpus) else {
            return false;
        };
        let cpu_scope_complete = self.affinity_scope == PalsCpuAffinityScopeV5::Process
            || (self.affinity_scope == PalsCpuAffinityScopeV5::ThreadGroupLeader
                && observed(&self.thread_inventory_complete) == Some(&true));
        let ort = observed(&self.ort_settings).is_some_and(|o| {
            o.intra_threads == plan.ort_intra_threads
                && o.inter_threads == plan.ort_inter_threads
                && o.spinning == plan.ort_spinning
        });
        let ort_required = matches!(
            d.model.backend,
            PalsModelBackendV3::OrtCpu | PalsModelBackendV3::OrtCuda
        );
        let actors_present = self
            .threads
            .iter()
            .any(|t| t.actors.contains(&PalsCpuActorV5::Manager))
            && self
                .threads
                .iter()
                .any(|t| t.actors.contains(&PalsCpuActorV5::NativeCaller))
            && (0..d.coordinator.slices.workers).all(|i| {
                self.threads
                    .iter()
                    .any(|t| t.actors.contains(&PalsCpuActorV5::OwnWorker { index: i }))
            })
            && (d.cpu.external.is_none()
                || self.threads.iter().any(|t| {
                    t.actors
                        .contains(&PalsCpuActorV5::ExternalEngine { index: 0 })
                }));
        let thread_masks_match = self.threads.iter().all(|t| {
            observed(&t.affinity).is_some_and(|mask| {
                mask.iter().all(|c| cpus.contains(c))
                    && (plan.topology != PalsTopologyPolicyV5::DedicatedControl
                        || t.actors.iter().all(|actor| match actor {
                            PalsCpuActorV5::Manager => {
                                mask.as_slice() == plan.dedicated_manager_cpu.as_slice()
                            }
                            PalsCpuActorV5::OwnWorker { .. } => {
                                mask.iter().all(|c| plan.worker_logical_cpus.contains(c))
                            }
                            _ => plan
                                .dedicated_manager_cpu
                                .is_none_or(|manager| !mask.contains(&manager)),
                        }))
            })
        });
        let gpu_match = plan.gpu_count == 0
            || observed(&self.gpu_identity).is_some_and(|gpu| {
                Some(&gpu.model_id) == plan.gpu_model.as_ref() && gpu.device_count == plan.gpu_count
            });
        observed(&self.limits_applied) == Some(&true)
            && observed(&self.cpu_quota).is_some_and(|q| q.matches_millicpus(plan.cpu_quota_millis))
            && observed(&self.quota_ancestor_chain_complete) == Some(&true)
            && cpu_scope_complete
            && !cpus.is_empty()
            && (plan.allowed_logical_cpus.is_empty() || *cpus == plan.allowed_logical_cpus)
            && observed(&self.memory_high)
                == Some(&PalsMemoryLimitReadbackV5::Limited {
                    bytes: plan.memory_high_bytes,
                })
            && observed(&self.memory_max)
                == Some(&PalsMemoryLimitReadbackV5::Limited {
                    bytes: plan.memory_max_bytes,
                })
            && (!ort_required || ort)
            && observed(&self.thread_inventory_complete) == Some(&true)
            && observed(&self.total_observed_kernel_threads)
                .is_some_and(|count| *count <= plan.kernel_threads_max)
            && actors_present
            && thread_masks_match
            && gpu_match
            && (d.model.backend != PalsModelBackendV3::OrtCuda
                || observed(&self.cuda_provider_ready) == Some(&true))
            && (plan.topology != PalsTopologyPolicyV5::DedicatedControl
                || observed(&self.smt_sibling_exclusion_verified) == Some(&true))
    }

    fn manager_roles_compatible(&self, d: &PalsEngineV5) -> bool {
        d.coordinator.policy == PalsCoordinatorPolicyV5::Synchronous
            || self.threads.iter().all(|thread| {
                !thread.actors.contains(&PalsCpuActorV5::Manager)
                    || thread.actors.iter().all(|actor| {
                        matches!(
                            actor,
                            PalsCpuActorV5::Manager | PalsCpuActorV5::RuntimeHelper { .. }
                        )
                    })
            })
    }

    pub fn validate_against(
        &self,
        d: &PalsEngineV5,
        run: &PalsRunReceiptV5,
    ) -> Result<(), ManifestError> {
        self.validate_with_context(d, &EndpointAcceptanceContextV5::from(run))
    }

    pub(super) fn validate_with_context(
        &self,
        d: &PalsEngineV5,
        run: &EndpointAcceptanceContextV5<'_>,
    ) -> Result<(), ManifestError> {
        self.validate_observations()?;
        require(
            self.manager_roles_compatible(d)
                || (!run.accepted && run.failures.contains(&PalsFailureV5::ResourceAdmission)),
            "asynchronous control manager cannot also be a blocking CPU or native caller thread",
        )?;
        require(
            self.profile == d.resources.evidence_profile
                || (!run.accepted && run.failures.contains(&PalsFailureV5::ResourceAdmission)),
            "resource readback profile contradicts declaration without an admission failure",
        )?;
        if run.accepted {
            require(
                self.profile != PalsResourceEvidenceProfileV5::StrictLocal
                    || self.matches_applied_plan(d),
                "strict-local acceptance requires matching applied quota/affinity/memory/ORT evidence",
            )?;
            if self.profile == PalsResourceEvidenceProfileV5::CloudObserved {
                require(
                    d.resources.topology == PalsTopologyPolicyV5::SharedControl
                        && (observed(&self.cpu_quota).is_some()
                            || observed(&self.allowed_logical_cpus).is_some()
                            || observed(&self.total_observed_kernel_threads).is_some()),
                    "cloud observation needs actual CPU evidence and SharedControl",
                )?;
            }
            require(
                d.model.backend != PalsModelBackendV3::OrtCuda
                    || observed(&self.cuda_provider_ready) == Some(&true),
                "accepted CUDA execution requires actual provider readiness",
            )?;
            if let Some(gpu) = observed(&self.gpu_identity) {
                require(
                    d.resources.gpu_model.as_ref() == Some(&gpu.model_id)
                        && d.resources.gpu_count == gpu.device_count,
                    "observed GPU differs from declared model/count",
                )?;
            }
        }
        require(
            (!run.performance_eligible && !run.strength_eligible) || self.matches_applied_plan(d),
            "partial cloud readback cannot support controlled performance or strength evidence",
        )
    }
}
