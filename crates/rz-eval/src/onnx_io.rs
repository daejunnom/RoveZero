//! Fixed shape/address bindings, exclusive to the synchronous physical owner.
//! All native buffers remain owned here on uncertain copy/run/synchronization.
use super::*;
use ort::io_binding::IoBinding;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::value::TensorValueType;

pub(super) enum BoundFailure {
    Native(ort::Error),
    Output(BackendError),
}
impl From<ort::Error> for BoundFailure {
    fn from(e: ort::Error) -> Self {
        Self::Native(e)
    }
}
pub(super) struct BoundBuffers {
    pub batch: usize,
    binding: IoBinding,
    device_input: Option<Tensor<f32>>,
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
            device_input: None,
            host_policy: None,
            host_wdl: None,
            allocator,
        };
        if matches!(provider, Provider::Cuda { .. }) {
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
        buffers.binding.bind_output(
            POLICY_NAME,
            Tensor::<f32>::new(&buffers.allocator, [batch, POLICY_SIZE])?,
        )?;
        buffers.binding.bind_output(
            WDL_NAME,
            Tensor::<f32>::new(&buffers.allocator, [batch, 3])?,
        )?;
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
            // ORT's synchronous Identity copy targets the existing allocation;
            // it does not replace the graph's input address. No async option.
            input.copy_into(device)?;
        } else {
            // CPU binding must be rebound after host contents change.
            self.binding.bind_input(INPUT_NAME, input)?;
        }
        self.binding.synchronize_inputs()?;
        if let Some(timing) = timing.as_deref_mut() {
            timing.transfer_in = Some(transfer_started.elapsed());
        }
        let run_started = std::time::Instant::now();
        let mut outputs = session.run_binding(&self.binding)?;
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
            outputs
                .remove(POLICY_NAME)
                .ok_or_else(|| ort::Error::new("missing bound policy"))?
                .downcast::<TensorValueType<f32>>()?
                .copy_into(policy)?;
            outputs
                .remove(WDL_NAME)
                .ok_or_else(|| ort::Error::new("missing bound WDL"))?
                .downcast::<TensorValueType<f32>>()?
                .copy_into(wdl)?;
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
            let (_, policy) = outputs
                .get(POLICY_NAME)
                .ok_or_else(|| ort::Error::new("missing policy"))?
                .try_extract_tensor::<f32>()?;
            let (_, wdl) = outputs
                .get(WDL_NAME)
                .ok_or_else(|| ort::Error::new("missing WDL"))?
                .try_extract_tensor::<f32>()?;
            let outputs = pack_outputs(policy, wdl, self.batch, pool, reuse);
            if let Some(timing) = timing {
                timing.transfer_out = Some(std::time::Duration::ZERO);
                timing.own_outputs = own_started.elapsed();
            }
            outputs
        };
        copied.map_err(BoundFailure::Output)
    }
}
