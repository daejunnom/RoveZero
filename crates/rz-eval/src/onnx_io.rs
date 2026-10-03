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
}
impl BoundBuffers {
    pub fn new(session: &Session, provider: Provider, batch: usize) -> ort::Result<Self> {
        let mut binding = session.create_binding()?;
        let cpu = Allocator::default();
        let (allocator, device_input, host_policy, host_wdl) = match provider {
            Provider::Cpu => (cpu, None, None, None),
            Provider::Cuda { device_id, .. } => {
                let allocator = Allocator::new(
                    session,
                    MemoryInfo::new(
                        AllocationDevice::CUDA,
                        device_id,
                        AllocatorType::Device,
                        MemoryType::Default,
                    )?,
                )?;
                let input = Tensor::<f32>::new(&allocator, [batch, 112, 8, 8])?;
                binding.bind_input(INPUT_NAME, &input)?;
                (
                    allocator,
                    Some(input),
                    Some(Tensor::new(&cpu, [batch, POLICY_SIZE])?),
                    Some(Tensor::new(&cpu, [batch, 3])?),
                )
            }
        };
        // These output addresses remain fixed through capture/replay. Binding
        // owns them; results are copied before the next exclusive Run.
        binding.bind_output(
            POLICY_NAME,
            Tensor::<f32>::new(&allocator, [batch, POLICY_SIZE])?,
        )?;
        binding.bind_output(WDL_NAME, Tensor::<f32>::new(&allocator, [batch, 3])?)?;
        Ok(Self {
            batch,
            binding,
            device_input,
            host_policy,
            host_wdl,
        })
    }
    pub fn run(
        &mut self,
        session: &mut Session,
        input: &Tensor<f32>,
        pool: &mut Vec<RawOutput>,
        reuse: bool,
    ) -> Result<Vec<RawOutput>, BoundFailure> {
        if let Some(device) = &mut self.device_input {
            // ORT's synchronous Identity copy targets the existing allocation;
            // it does not replace the graph's input address. No async option.
            input.copy_into(device)?;
        } else {
            // CPU binding must be rebound after host contents change.
            self.binding.bind_input(INPUT_NAME, input)?;
        }
        self.binding.synchronize_inputs()?;
        let mut outputs = session.run_binding(&self.binding)?;
        self.binding.synchronize_outputs()?;
        let copied = if let (Some(policy), Some(wdl)) = (&mut self.host_policy, &mut self.host_wdl)
        {
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
            let (_, policy) = policy.try_extract_tensor::<f32>()?;
            let (_, wdl) = wdl.try_extract_tensor::<f32>()?;
            pack_outputs(policy, wdl, self.batch, pool, reuse)
        } else {
            let (_, policy) = outputs
                .get(POLICY_NAME)
                .ok_or_else(|| ort::Error::new("missing policy"))?
                .try_extract_tensor::<f32>()?;
            let (_, wdl) = outputs
                .get(WDL_NAME)
                .ok_or_else(|| ort::Error::new("missing WDL"))?
                .try_extract_tensor::<f32>()?;
            pack_outputs(policy, wdl, self.batch, pool, reuse)
        };
        copied.map_err(BoundFailure::Output)
    }
}
