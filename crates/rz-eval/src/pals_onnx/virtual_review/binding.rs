//! Private P/C fixed I/O owner for the V3 dynamic review ABI.
//!
//! The backend installs this owner before the first input bind can enqueue a
//! transfer. A failed bind, copy, Run or output fence retains this owner, the
//! Active input and its pinned PublicMemory in quarantine. Logical cancellation
//! is not physical completion. This module adds no FFI or completion shortcut.
use super::*;
use ort::io_binding::IoBinding;
use rz_native_loader::ort_binding::run_fixed_binding;

/// Values are CPU-backed fixed outputs. ORT executes the selected CUDA graph
/// and copies its result to these buffers; synchronize_outputs is required
/// before extraction or reuse. Binding metadata/native allocator bytes are not
/// inferred from their Rust output backing capacities.
pub(super) struct Binding {
    // Drop bindings before their separately retained output owners.
    binding: IoBinding,
    candidate: Tensor<f32>,
    wdl: Tensor<f32>,
    latent: Tensor<f32>,
    divergence: Tensor<f32>,
    is_critic: Tensor<bool>,
    candidates: usize,
    divergences: usize,
    critic: bool,
    host_backing_bytes: u64,
}

/// Retains the same owned OrtValue. Tensor::clone may copy device backing and
/// is deliberately not used for either output binding or lifetime ownership.
fn alias<T: ort::tensor::PrimitiveTensorElementType + std::fmt::Debug>(
    tensor: &Tensor<T>,
) -> ort::Result<Tensor<T>> {
    tensor
        .view()
        .try_upgrade()
        .map_err(|_| ort::Error::new("V3 bound output cannot upgrade its owned view"))
}

fn float_output(shape: &[usize], elements: usize) -> ort::Result<(Tensor<f32>, u64)> {
    let mut backing = Vec::new();
    backing
        .try_reserve_exact(elements)
        .map_err(|error| ort::Error::new(format!("V3 fixed output allocation failed: {error}")))?;
    // A partially written native result must not pass a finite-value check.
    backing.resize(elements, f32::NAN);
    let bytes = (backing.capacity() as u64)
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| ort::Error::new("V3 fixed output capacity overflows"))?;
    Ok((Tensor::from_array((shape.to_vec(), backing))?, bytes))
}

impl Binding {
    pub(super) fn new(
        session: &Session,
        candidates: usize,
        divergences: usize,
        critic: bool,
    ) -> ort::Result<Self> {
        // Matches the separately validated V3 admission, including one padded
        // physical slot for an empty semantic candidate/divergence list.
        if !(1..=256).contains(&candidates) || !(1..=128).contains(&divergences) {
            return Err(ort::Error::new(
                "V3 fixed output dimensions exceed admission",
            ));
        }
        let (candidate, candidate_bytes) = float_output(&[1, candidates], candidates)?;
        let (wdl, wdl_bytes) = float_output(&[1, 3], 3)?;
        let (latent, latent_bytes) = float_output(&[1, 16, 384], 16 * 384)?;
        let (divergence, divergence_bytes) = float_output(&[1, divergences], divergences)?;
        let mut role_backing = Vec::new();
        role_backing.try_reserve_exact(1).map_err(|error| {
            ort::Error::new(format!("V3 role output allocation failed: {error}"))
        })?;
        role_backing.push(false);
        let role_bytes = role_backing.capacity() as u64 * std::mem::size_of::<bool>() as u64;
        let is_critic = Tensor::from_array((Vec::<usize>::new(), role_backing))?;
        let host_backing_bytes = [
            candidate_bytes,
            wdl_bytes,
            latent_bytes,
            divergence_bytes,
            role_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
        .ok_or_else(|| ort::Error::new("V3 fixed output byte sum overflows"))?;
        let mut binding = session.create_binding()?;
        binding.bind_output("candidate_logits", alias(&candidate)?)?;
        binding.bind_output("wdl_logits", alias(&wdl)?)?;
        binding.bind_output("private_latent", alias(&latent)?)?;
        binding.bind_output("divergence_logits", alias(&divergence)?)?;
        binding.bind_output("is_critic", alias(&is_critic)?)?;
        Ok(Self {
            binding,
            candidate,
            wdl,
            latent,
            divergence,
            is_critic,
            candidates,
            divergences,
            critic,
            host_backing_bytes,
        })
    }

    /// Known fixed CPU output backing only. Active owns and counts the input
    /// role, Warm seed/mode, and all prepared input capacities independently.
    pub(super) fn host_backing_bytes(&self) -> u64 {
        self.host_backing_bytes
    }

    /// Success attests the existing synchronous Run plus output-copy fence.
    /// Semantic validation happens separately so its failure cannot erase
    /// completed NN work. On error no input alias is cleared or reused.
    pub(super) fn run(
        &mut self,
        session: &mut Session,
        active: &Active,
        memory: &PublicMemory,
    ) -> ort::Result<()> {
        self.binding.bind_input("role_is_critic", &active.role)?;
        self.binding.bind_input("memory_key", &memory.key)?;
        self.binding.bind_input("memory_value", &memory.value)?;
        self.binding.bind_input("memory_mask", &memory.mask)?;
        self.binding.bind_input("anchor_context", &memory.anchors)?;
        self.binding.bind_input("candidates", &active.candidates)?;
        self.binding
            .bind_input("candidate_mask", &active.candidate_mask)?;
        self.binding
            .bind_input("divergence_features", &active.divergences)?;
        self.binding
            .bind_input("divergence_mask", &active.divergence_mask)?;
        self.binding.bind_input("query", &active.query)?;
        self.binding
            .bind_input("query_anchor", &active.query_anchor)?;
        self.binding
            .bind_input("query_line_tokens", &active.query_line_tokens)?;
        self.binding
            .bind_input("query_line_mask", &active.query_line_mask)?;
        self.binding
            .bind_input("record_relations", &active.record_relations)?;
        if let Some(warm) = &active.warm {
            // CUDA If conditions stay in owned CPU bool scalar memory.
            self.binding
                .bind_input("initial_latent", &warm.initial_latent)?;
            self.binding.bind_input("warm_start", &warm.warm_start)?;
        }
        self.binding.synchronize_inputs()?;
        run_fixed_binding(session, &self.binding)?;
        self.binding.synchronize_outputs()?;
        // rc.10 clear_inputs is infallible; it is reached only after the real
        // output fence. Fixed output aliases remain owned for extraction.
        self.binding.clear_inputs();
        Ok(())
    }

    pub(super) fn extract(
        &self,
        input: &PalsVirtualReviewInput,
    ) -> Result<PalsRawOutput, BackendError> {
        if input.role == PalsRole::Validator
            || self.critic != (input.role == PalsRole::Critic)
            || self.candidates != input.candidates.len().max(1)
            || self.divergences != input.divergences.len().max(1)
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Output,
                "V3 bound output request differs",
            ));
        }
        let extract = |tensor: &Tensor<f32>, expected: &[i64]| {
            let (shape, values) = tensor.try_extract_tensor::<f32>().map_err(|error| {
                native(
                    CauseCode::PolicyExtract,
                    "V3 bound output extraction failed",
                    error,
                )
            })?;
            if shape.as_ref() != expected || values.iter().any(|value| !value.is_finite()) {
                return Err(fail(
                    K::NumericalFailure,
                    S::Output,
                    "V3 bound output shape/finite check failed",
                ));
            }
            Ok(values.to_vec())
        };
        let mut candidate_logits = extract(&self.candidate, &[1, self.candidates as i64])?;
        let mut divergence_logits = extract(&self.divergence, &[1, self.divergences as i64])?;
        let (shape, tag) = self
            .is_critic
            .try_extract_tensor::<bool>()
            .map_err(|error| {
                native(
                    CauseCode::PolicyExtract,
                    "V3 bound role tag extraction failed",
                    error,
                )
            })?;
        validate_role_tag(shape.as_ref(), tag, self.critic)?;
        // Validate the physical padded divergence slot before semantic trim.
        if !self.critic && divergence_logits.iter().any(|value| *value != 0.) {
            return Err(fail(
                K::NumericalFailure,
                S::Output,
                "V3 proposer emitted nonzero critic output",
            ));
        }
        candidate_logits.truncate(input.candidates.len());
        divergence_logits.truncate(input.divergences.len());
        Ok(PalsRawOutput {
            candidate_logits,
            wdl_logits: extract(&self.wdl, &[1, 3])?
                .try_into()
                .map_err(|_| fail(K::NumericalFailure, S::Output, "V3 bound WDL shape"))?,
            private_latent: extract(&self.latent, &[1, 16, 384])?,
            divergence_logits: self.critic.then_some(divergence_logits),
            task_logits: None,
        })
    }
}
