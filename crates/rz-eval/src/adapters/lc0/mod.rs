//! Supported LC0 profiles and their input/output/native session semantics.
pub mod asset;
#[cfg(feature = "contracts")]
pub mod contracts;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod output;
#[cfg(feature = "contracts")]
pub mod rules_projection;

/// Untrusted output at the model adapter boundary. Missing heads, bad shapes,
/// and non-finite values are deliberately representable for fault injection.
#[derive(Clone, Debug, PartialEq)]
pub struct RawOutput {
    pub policy_logits: Vec<f32>,
    pub wdl: Vec<f32>,
}

#[cfg(feature = "contracts")]
use crate::{
    model_adapter::AdapterAdmission,
    native_runtime_bridge::{NATIVE_CUDA_ADMISSION_BYTES, NATIVE_RUNTIME_OVERHEAD_BYTES},
};
#[cfg(feature = "contracts")]
use rz_runtime::Resources;
/// Admission declarations, not memory measurements or native allocation caps.
/// D reserves `execution_resources` while a physical lease is outstanding.
/// Bootstrap accounts for the separate session resident declaration; D must not
/// subtract it when an individual execution completes. An injected worker may
/// exercise this policy but cannot claim a verified CUDA origin.
#[cfg(feature = "contracts")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeAdmissionPolicy {
    Cpu,
    CudaOneGiB,
    CudaThreeGiB,
}

#[cfg(feature = "contracts")]
impl NativeAdmissionPolicy {
    /// Additional D resources, beyond the EvalRequest's own ByteBudget. Adding
    /// the same device declaration to ByteBudget would reserve it twice.
    pub fn execution_resources(self) -> Resources {
        Resources {
            host_bytes: NATIVE_RUNTIME_OVERHEAD_BYTES,
            device_bytes: match self {
                Self::Cpu => 0,
                Self::CudaOneGiB => NATIVE_CUDA_ADMISSION_BYTES,
                Self::CudaThreeGiB => 3 * NATIVE_CUDA_ADMISSION_BYTES,
            },
            pinned_bytes: 0,
        }
    }

    /// Separate bootstrap declaration. It is not an observed resident peak and
    /// is not enforced by an execution's D reservation.
    pub fn session_resident_admission(self) -> Resources {
        Resources {
            host_bytes: 0,
            device_bytes: self.execution_resources().device_bytes,
            pinned_bytes: 0,
        }
    }
}

#[cfg(feature = "contracts")]
impl From<NativeAdmissionPolicy> for AdapterAdmission {
    fn from(p: NativeAdmissionPolicy) -> Self {
        Self {
            execution: p.execution_resources(),
            session: p.session_resident_admission(),
        }
    }
}
#[cfg(feature = "contracts")]
impl PartialEq<NativeAdmissionPolicy> for AdapterAdmission {
    fn eq(&self, p: &NativeAdmissionPolicy) -> bool {
        *self == AdapterAdmission::from(*p)
    }
}
