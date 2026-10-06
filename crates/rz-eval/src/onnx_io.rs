//! Fixed shape/address bindings, exclusive to the synchronous physical owner.
//! All native buffers remain owned here on uncertain copy/run/synchronization.
use super::*;
use ort::io_binding::IoBinding;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use rz_native_loader::ort_binding::run_fixed_binding;

pub(super) enum BoundFailure {
    Native(ort::Error),
    Output(BackendError),
}
impl From<ort::Error> for BoundFailure {
    fn from(e: ort::Error) -> Self {
        Self::Native(e)
    }
}

fn owned_alias(value: &Tensor<f32>) -> ort::Result<Tensor<f32>> {
    // Shares the existing native value; Clone would instead perform a device
    // copy. The alias never leaves the exclusive binding owner.
    value
        .view()
        .try_upgrade()
        .map_err(|_| ort::Error::new("owned tensor view cannot upgrade"))
}

struct CopySession {
    binding: IoBinding,
    session: Session,
}
impl CopySession {
    fn new(device_id: i32, to_device: bool) -> ort::Result<Self> {
        let cpu = CPUExecutionProvider::default()
            .with_arena_allocator(false)
            .build()
            .error_on_failure();
        let cuda = CUDAExecutionProvider::default()
            .with_device_id(device_id)
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
            .with_conv_max_workspace(false)
            .with_tf32(false)
            .build()
            .error_on_failure();
        let providers = if to_device { [cpu, cuda] } else { [cuda, cpu] };
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_intra_op_spinning(false)?
            .with_inter_op_spinning(false)?
            .with_memory_pattern(false)?
            .with_allocator(MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )?)?
            .with_no_environment_execution_providers()?
            .with_execution_providers(providers)?
            .commit_from_memory(COPY_MODEL)?;
        Ok(Self {
            binding: session.create_binding()?,
            session,
        })
    }
    fn copy(&mut self, source: &Tensor<f32>, target: &mut Tensor<f32>) -> ort::Result<()> {
        if source.shape() != target.shape() {
            return Err(ort::Error::new("copy target shape differs"));
        }
        self.binding.bind_input("input", source)?;
        self.binding.bind_output("output", owned_alias(target)?)?;
        self.binding.synchronize_inputs()?;
        run_fixed_binding(&mut self.session, &self.binding)?;
        self.binding.synchronize_outputs()?;
        // Success only: failure leaves every native alias with the physical
        // owner's quarantine. No process-global helper retains these tensors.
        self.binding.clear();
        Ok(())
    }
}

// Self-authored ONNX IR8/opset13 Identity, FLOAT input/output of unknown rank.
// Used only for synchronous tensor copies; BT4 weights/precision are unchanged.
const COPY_MODEL: &[u8] = &[
    8, 8, 18, 8, 82, 111, 118, 101, 90, 101, 114, 111, 58, 71, 10, 25, 10, 5, 105, 110, 112, 117,
    116, 18, 6, 111, 117, 116, 112, 117, 116, 34, 8, 73, 100, 101, 110, 116, 105, 116, 121, 18, 11,
    114, 122, 45, 99, 111, 112, 121, 45, 102, 51, 50, 90, 13, 10, 5, 105, 110, 112, 117, 116, 18,
    4, 10, 2, 8, 1, 98, 14, 10, 6, 111, 117, 116, 112, 117, 116, 18, 4, 10, 2, 8, 1, 66, 2, 16, 13,
];
pub(super) struct BoundBuffers {
    pub batch: usize,
    binding: IoBinding,
    // Private copy bindings drop before all allocator-backed tensors.
    copy_in: Option<CopySession>,
    copy_out: Option<CopySession>,
    device_input: Option<Tensor<f32>>,
    policy: Option<Tensor<f32>>,
    wdl: Option<Tensor<f32>>,
    host_policy: Option<Tensor<f32>>,
    host_wdl: Option<Tensor<f32>>,
    // ort rc.10's Tensor::new does not retain the supplied Allocator. Native
    // tensor copies/destruction still use its callbacks. Keep this LAST so all
    // binding-held values and device_input drop before the allocator, also when
    // construction fails after only some of the buffers have been created.
    allocator: Allocator,
}
impl BoundBuffers {
    pub fn new(session: &Session, provider: Provider, batch: usize) -> ort::Result<Self> {
        let allocator = match provider {
            Provider::Cpu => Allocator::default(),
            Provider::Cuda { device_id, .. } => Allocator::new(
                session,
                MemoryInfo::new(
                    AllocationDevice::CUDA,
                    device_id,
                    AllocatorType::Device,
                    MemoryType::Default,
                )?,
            )?,
        };
        // Install the allocator owner before creating any dependent tensor.
        // Local reverse drop order would otherwise release it before binding
        // on a later allocation/bind failure.
        let mut buffers = Self {
            batch,
            binding: session.create_binding()?,
            copy_in: None,
            copy_out: None,
            device_input: None,
            policy: None,
            wdl: None,
            host_policy: None,
            host_wdl: None,
            allocator,
        };
        if let Provider::Cuda { device_id, .. } = provider {
            buffers.copy_in = Some(CopySession::new(device_id, true)?);
            buffers.copy_out = Some(CopySession::new(device_id, false)?);
            buffers.device_input = Some(Tensor::new(&buffers.allocator, [batch, 112, 8, 8])?);
            buffers.binding.bind_input(
                INPUT_NAME,
                buffers.device_input.as_ref().expect("device input"),
            )?;
            let cpu = Allocator::default();
            buffers.host_policy = Some(Tensor::new(&cpu, [batch, POLICY_SIZE])?);
            buffers.host_wdl = Some(Tensor::new(&cpu, [batch, 3])?);
        }
        // These output addresses remain fixed through capture/replay. Binding
        // owns them; results are copied before the next exclusive Run.
        buffers.policy = Some(Tensor::new(&buffers.allocator, [batch, POLICY_SIZE])?);
        buffers.wdl = Some(Tensor::new(&buffers.allocator, [batch, 3])?);
        buffers.binding.bind_output(
            POLICY_NAME,
            owned_alias(buffers.policy.as_ref().expect("policy"))?,
        )?;
        buffers
            .binding
            .bind_output(WDL_NAME, owned_alias(buffers.wdl.as_ref().expect("WDL"))?)?;
        Ok(buffers)
    }
    pub fn run(
        &mut self,
        session: &mut Session,
        input: &Tensor<f32>,
        pool: &mut Vec<RawOutput>,
        reuse: bool,
        mut timing: Option<&mut IoTimings>,
    ) -> Result<Vec<RawOutput>, BoundFailure> {
        let transfer_started = std::time::Instant::now();
        if let Some(device) = &mut self.device_input {
            // Synchronous private copy into the same graph input address.
            self.copy_in
                .as_mut()
                .expect("CUDA input copy")
                .copy(input, device)?;
        } else {
            // CPU binding must be rebound after host contents change.
            self.binding.bind_input(INPUT_NAME, input)?;
        }
        self.binding.synchronize_inputs()?;
        if let Some(timing) = timing.as_deref_mut() {
            timing.transfer_in = Some(transfer_started.elapsed());
        }
        let run_started = std::time::Instant::now();
        run_fixed_binding(session, &self.binding)?;
        if let Some(timing) = timing.as_deref_mut() {
            timing.run = run_started.elapsed();
        }
        let fence_started = std::time::Instant::now();
        self.binding.synchronize_outputs()?;
        if let Some(timing) = timing.as_deref_mut() {
            timing.output_fence = Some(fence_started.elapsed());
        }
        let copied = if let (Some(policy), Some(wdl)) = (&mut self.host_policy, &mut self.host_wdl)
        {
            let transfer_started = std::time::Instant::now();
            let copy = self.copy_out.as_mut().expect("CUDA output copy");
            copy.copy(self.policy.as_ref().expect("owned policy"), policy)?;
            copy.copy(self.wdl.as_ref().expect("owned WDL"), wdl)?;
            if let Some(timing) = timing.as_deref_mut() {
                timing.transfer_out = Some(transfer_started.elapsed());
            }
            let own_started = std::time::Instant::now();
            let (_, policy) = policy.try_extract_tensor::<f32>()?;
            let (_, wdl) = wdl.try_extract_tensor::<f32>()?;
            let outputs = pack_outputs(policy, wdl, self.batch, pool, reuse);
            if let Some(timing) = timing.as_deref_mut() {
                timing.own_outputs = own_started.elapsed();
            }
            outputs
        } else {
            let own_started = std::time::Instant::now();
            let (_, policy) = self
                .policy
                .as_ref()
                .expect("owned policy")
                .try_extract_tensor::<f32>()?;
            let (_, wdl) = self
                .wdl
                .as_ref()
                .expect("owned WDL")
                .try_extract_tensor::<f32>()?;
            let outputs = pack_outputs(policy, wdl, self.batch, pool, reuse);
            if let Some(timing) = timing {
                timing.transfer_out = Some(std::time::Duration::ZERO);
                timing.own_outputs = own_started.elapsed();
            }
            outputs
        };
        if self.device_input.is_none() {
            // Physical Run/output fence and copying have completed. Release the
            // CPU input alias before the parent returns its exclusive input.
            self.binding.clear_inputs();
        }
        copied.map_err(BoundFailure::Output)
    }
}
