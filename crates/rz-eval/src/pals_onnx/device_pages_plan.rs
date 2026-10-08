//! First source unit for resident public pages: ownership and admission only.
//!
//! No ORT session, CUDA allocation/copy, native Run, device fence, or physical
//! completion attestation is implemented here. The only sealed backing supplied
//! by this unit is an immutable, explicitly CPU-fixture backing. A future native
//! backing and certification boundary must remain inside this module's ownership
//! boundary and attest its actual Run/fence before publication.
//!
//! Reservations are known payloads and explicit declarations, never measured
//! heap/VRAM peaks. Allocator rounding, Arc internals and native workspaces remain
//! unobserved. All declared transient stages stay reserved even on a cache hit;
//! absence of a materialized pending subset is not permission to shrink them.
use crate::pals_model::{
    PalsModelConfig, PalsModelInput, PalsRawOutput, PalsRole,
    INDEPENDENT_RECORD_PROJECTION_SEMANTICS,
};
use rz_contracts::{ContractError, Digest, PrecisionProfile, ProcessEpoch};
use rz_runtime::pals::{
    MemoryBank, MemoryBankSnapshot, MemoryKey, MemoryPin, PublicPageKey, PublicPageKind,
};
use sha2::{Digest as _, Sha256};
use std::fmt;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "experimental-io-binding")]
pub(super) mod native;

pub const DEVICE_PAGE_RECORD_CAPACITY: usize = 128;
const BOARD: usize = 66;
const HEADS: usize = 2;
const DIM: usize = 64;
const MAX_TOKENS: usize = BOARD + DEVICE_PAGE_RECORD_CAPACITY;
const MAX_BLOCKS: usize = DEVICE_PAGE_RECORD_CAPACITY + 1;
const MAX_REGISTRY: usize = (MAX_BLOCKS + 1) * (DEVICE_PAGE_RECORD_CAPACITY + 1);

// Ownership origin only, deliberately absent from semantic projection/cache
// identities. No reset, serialization or public seal constructor is provided.
static REGISTRY_INSTANCE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RegistryInstanceSeal(u64);
fn allocate_registry_seal(sequence: &AtomicU64) -> Result<RegistryInstanceSeal> {
    let previous = sequence
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| DevicePageError::Overflow)?;
    Ok(RegistryInstanceSeal(previous + 1))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DevicePageDomain {
    /// Synthetic CPU arrays only, never a CUDA capability or physical witness.
    CpuFixture,
    /// A declaration only: this unit has no backing implementation for CUDA.
    Cuda {
        device_id: u32,
        runtime_sha256: [u8; 32],
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DevicePageNamespace {
    /// The actual common-contract process epoch belongs to semantic identity.
    /// A separate private seal binds each registry's ownership origin, even
    /// when registries intentionally share every semantic namespace field.
    pub process_epoch: ProcessEpoch,
    pub model_manifest: [u8; 32],
    pub model_epoch: [u8; 32],
    pub public_graph: [u8; 32],
    pub encoding: [u8; 32],
    pub frozen_epoch: u64,
    pub game_generation: u64,
    pub domain: DevicePageDomain,
}

#[derive(Debug)]
pub enum DevicePageError {
    InvalidInput,
    InvalidLimits,
    IdentityMismatch,
    StaleGeneration,
    UnvalidatedSlice,
    MissingProjection,
    UnknownBudget,
    BudgetExceeded,
    Overflow,
    Allocation,
    Bank(ContractError),
}
impl fmt::Display for DevicePageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "device-page source admission: {self:?}")
    }
}
impl std::error::Error for DevicePageError {}
type Result<T> = std::result::Result<T, DevicePageError>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DevicePagePayload {
    pub host: u64,
    pub device: u64,
}
impl DevicePagePayload {
    fn add(self, other: Self) -> Result<Self> {
        Ok(Self {
            host: self
                .host
                .checked_add(other.host)
                .ok_or(DevicePageError::Overflow)?,
            device: self
                .device
                .checked_add(other.device)
                .ok_or(DevicePageError::Overflow)?,
        })
    }
    pub fn total(self) -> Result<u64> {
        self.host
            .checked_add(self.device)
            .ok_or(DevicePageError::Overflow)
    }
    fn in_domain(domain: DevicePageDomain, bytes: u64) -> Self {
        match domain {
            DevicePageDomain::CpuFixture => Self {
                host: bytes,
                device: 0,
            },
            DevicePageDomain::Cuda { .. } => Self {
                host: 0,
                device: bytes,
            },
        }
    }
}
fn bytes(count: usize, width: usize) -> Result<u64> {
    let value = count.checked_mul(width).ok_or(DevicePageError::Overflow)?;
    u64::try_from(value).map_err(|_| DevicePageError::Overflow)
}
fn sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |a, b| {
        a.checked_add(*b).ok_or(DevicePageError::Overflow)
    })
}
fn kv_bytes(tokens: usize) -> Result<u64> {
    bytes(tokens, 2 * HEADS * DIM * size_of::<f32>())
}

fn projection(
    namespace: DevicePageNamespace,
    kind: PublicPageKind,
    content: [u8; 32],
) -> MemoryKey {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-device-public-projection/1");
    hash.update(namespace.process_epoch.0.to_le_bytes());
    hash.update(content);
    hash.update(namespace.model_epoch);
    hash.update(namespace.public_graph);
    hash.update(INDEPENDENT_RECORD_PROJECTION_SEMANTICS.as_bytes());
    // Keep this distinct from host pages; provider/runtime declaration is part
    // of this source namespace, not proof that CUDA is available.
    match namespace.domain {
        DevicePageDomain::CpuFixture => hash.update([0]),
        DevicePageDomain::Cuda {
            device_id,
            runtime_sha256,
        } => {
            hash.update([1]);
            hash.update(device_id.to_le_bytes());
            hash.update(runtime_sha256);
        }
    }
    MemoryKey::PublicPage(PublicPageKey {
        kind,
        content: Digest(hash.finalize().into()),
        model: Digest(namespace.model_manifest),
        encoding: Digest(namespace.encoding),
        precision: PrecisionProfile::Fp32,
        frozen_epoch: namespace.frozen_epoch,
        game_generation: namespace.game_generation,
    })
}
fn record_content(namespace: DevicePageNamespace, features: &[f32; 16]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(INDEPENDENT_RECORD_PROJECTION_SEMANTICS.as_bytes());
    hash.update(namespace.model_epoch);
    for value in features {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash.finalize().into()
}

/// Descriptor is derived from actual validated model inputs, not a caller's
/// FEN/hash-only cache declaration. Fields have no mutable public access.
pub struct DevicePublicBlockDescriptor {
    namespace: DevicePageNamespace,
    block_key: MemoryKey,
    block_generation: u64,
    board_content: [u8; 32],
    features: Vec<[f32; 16]>,
    record_mask: Vec<bool>,
}
impl DevicePublicBlockDescriptor {
    /// Empty subset is the actual zero-feature/false-mask slot, including a
    /// board-only miss in a nonempty logical view. Selected records are unique
    /// projection inputs; ordered duplicate occurrences belong to the plan.
    pub fn from_input(
        namespace: DevicePageNamespace,
        block_generation: u64,
        input: &PalsModelInput,
        config: &PalsModelConfig,
        subset_indices: &[usize],
    ) -> Result<Self> {
        if block_generation == 0
            || namespace.process_epoch.0 == 0
            || input.model_epoch != namespace.model_epoch
            || input.role == PalsRole::Validator
            || subset_indices.len() > DEVICE_PAGE_RECORD_CAPACITY
            || matches!(namespace.domain, DevicePageDomain::Cuda { device_id, .. } if device_id > i32::MAX as u32)
        {
            return Err(DevicePageError::IdentityMismatch);
        }
        let plan = input
            .independent_public_plan(config)
            .map_err(|_| DevicePageError::InvalidInput)?;
        if plan.record_contents.len() > DEVICE_PAGE_RECORD_CAPACITY {
            return Err(DevicePageError::InvalidInput);
        }
        let slots = subset_indices.len().max(1);
        let mut features = Vec::new();
        let mut record_mask = Vec::new();
        features
            .try_reserve_exact(slots)
            .map_err(|_| DevicePageError::Allocation)?;
        record_mask
            .try_reserve_exact(slots)
            .map_err(|_| DevicePageError::Allocation)?;
        if subset_indices.is_empty() {
            features.push([0.0; 16]);
            record_mask.push(false);
        } else {
            for (position, index) in subset_indices.iter().copied().enumerate() {
                if index >= plan.features.len()
                    || subset_indices[..position]
                        .iter()
                        .any(|old| plan.record_contents[*old] == plan.record_contents[index])
                {
                    return Err(DevicePageError::InvalidInput);
                }
                features.push(plan.features[index]);
                record_mask.push(plan.record_mask[index]);
            }
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-device-whole-subset-input/1");
        hash.update(plan.board_content);
        for (feature, mask) in features.iter().zip(&record_mask) {
            for value in feature {
                hash.update(value.to_bits().to_le_bytes());
            }
            hash.update([u8::from(*mask)]);
        }
        let block_key = projection(
            namespace,
            PublicPageKind::WholeInput,
            hash.finalize().into(),
        );
        Ok(Self {
            namespace,
            block_key,
            block_generation,
            board_content: plan.board_content,
            features,
            record_mask,
        })
    }
    pub fn tokens(&self) -> usize {
        BOARD + self.features.len()
    }
    pub fn block_key(&self) -> MemoryKey {
        self.block_key
    }
    pub fn generation(&self) -> u64 {
        self.block_generation
    }
    fn zero_padding(&self) -> bool {
        self.features.len() == 1
            && self.record_mask.len() == 1
            && !self.record_mask[0]
            && self.features[0].iter().all(|value| value.to_bits() == 0)
    }
    fn host_bytes(&self) -> Result<u64> {
        // Inline descriptor storage is counted once by DevicePublicBlock.
        sum(&[
            bytes(self.features.capacity(), 16 * 4)?,
            bytes(self.record_mask.capacity(), size_of::<bool>())?,
        ])
    }
}

mod sealed {
    pub trait Backing {}
}
/// Sealed whole owner. No external implementation may declare a view's short
/// byte length while retaining a larger allocation or invent CUDA completion.
pub trait DevicePublicBacking: sealed::Backing {
    fn domain(&self) -> DevicePageDomain;
    fn tokens(&self) -> usize;
    fn whole_payload(&self) -> Result<DevicePagePayload>;
}

/// Synthetic immutable CPU owner. It makes no model/NN/CUDA producer claim.
pub struct CpuOwnedPublicBacking {
    tokens: usize,
    key: Vec<f32>,
    value: Vec<f32>,
}
impl CpuOwnedPublicBacking {
    pub fn new(tokens: usize, key: Vec<f32>, value: Vec<f32>) -> Result<Self> {
        if !(BOARD + 1..=MAX_TOKENS).contains(&tokens)
            || key.len() != HEADS * tokens * DIM
            || value.len() != HEADS * tokens * DIM
        {
            return Err(DevicePageError::InvalidInput);
        }
        Ok(Self { tokens, key, value })
    }
    pub fn key(&self) -> &[f32] {
        &self.key
    }
    pub fn value(&self) -> &[f32] {
        &self.value
    }
    fn finite_slice(&self, offset: usize, count: usize) -> bool {
        if offset
            .checked_add(count)
            .is_none_or(|end| end > self.tokens)
        {
            return false;
        }
        (0..HEADS).all(|head| {
            let start = (head * self.tokens + offset) * DIM;
            let end = start + count * DIM;
            self.key[start..end]
                .iter()
                .chain(&self.value[start..end])
                .all(|v| v.is_finite())
        })
    }
}
impl sealed::Backing for CpuOwnedPublicBacking {}
impl DevicePublicBacking for CpuOwnedPublicBacking {
    fn domain(&self) -> DevicePageDomain {
        DevicePageDomain::CpuFixture
    }
    fn tokens(&self) -> usize {
        self.tokens
    }
    fn whole_payload(&self) -> Result<DevicePagePayload> {
        // Inline backing headers are counted once by DevicePublicBlock.
        Ok(DevicePagePayload {
            host: sum(&[
                bytes(self.key.capacity(), 4)?,
                bytes(self.value.capacity(), 4)?,
            ])?,
            device: 0,
        })
    }
}

/// Completed CPU slice certification is minted by actual finite-array checks.
/// A future native owner needs its own private Run/fence/finite-output factory;
/// there is intentionally no public constructor accepting a completion bool.
pub struct DevicePublicBlock<B: DevicePublicBacking> {
    descriptor: DevicePublicBlockDescriptor,
    backing: B,
    certified_board: bool,
    certified_records: [bool; DEVICE_PAGE_RECORD_CAPACITY],
}
impl DevicePublicBlock<CpuOwnedPublicBacking> {
    pub fn certify_cpu(
        descriptor: DevicePublicBlockDescriptor,
        backing: CpuOwnedPublicBacking,
        board: bool,
        record_indices: &[usize],
    ) -> Result<Self> {
        if descriptor.namespace.domain != DevicePageDomain::CpuFixture
            || backing.tokens != descriptor.tokens()
            || (!board && record_indices.is_empty())
            || record_indices.len() > DEVICE_PAGE_RECORD_CAPACITY
        {
            return Err(DevicePageError::InvalidInput);
        }
        if board && !backing.finite_slice(0, BOARD) {
            return Err(DevicePageError::UnvalidatedSlice);
        }
        let mut certified_records = [false; DEVICE_PAGE_RECORD_CAPACITY];
        for index in record_indices.iter().copied() {
            if index >= descriptor.features.len()
                || certified_records[index]
                || !backing.finite_slice(BOARD + index, 1)
            {
                return Err(DevicePageError::UnvalidatedSlice);
            }
            certified_records[index] = true;
        }
        Ok(Self {
            descriptor,
            backing,
            certified_board: board,
            certified_records,
        })
    }
}
impl<B: DevicePublicBacking> DevicePublicBlock<B> {
    pub fn backing(&self) -> &B {
        &self.backing
    }
    pub fn descriptor(&self) -> &DevicePublicBlockDescriptor {
        &self.descriptor
    }
    fn reservation(&self) -> Result<DevicePagePayload> {
        if self.backing.domain() != self.descriptor.namespace.domain
            || self.backing.tokens() != self.descriptor.tokens()
        {
            return Err(DevicePageError::IdentityMismatch);
        }
        self.backing.whole_payload()?.add(DevicePagePayload {
            host: self
                .descriptor
                .host_bytes()?
                .checked_add(size_of::<Self>() as u64)
                .ok_or(DevicePageError::Overflow)?,
            device: 0,
        })
    }
    fn accepts_offset(&self, offset: usize) -> bool {
        if offset == 0 {
            return self.certified_board;
        }
        offset >= BOARD
            && offset < self.descriptor.tokens()
            && self.certified_records[offset - BOARD]
    }
}

/// Copyable, non-owning locator: no Arc, Weak, Tensor, pin or backing reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceProjectionOffset {
    // Private Copy integer seal, never an Arc/Weak/pin or semantic cache key.
    instance_seal: RegistryInstanceSeal,
    pub block_key: MemoryKey,
    pub block_generation: u64,
    pub token_offset: u16,
}
#[derive(Clone, Copy)]
struct ProjectionEntry {
    projection: MemoryKey,
    location: DeviceProjectionOffset,
}
#[derive(Clone, Copy)]
struct BlockLocator {
    key: MemoryKey,
    generation: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct DevicePagesLimits {
    pub max_blocks: usize,
    pub max_bank_whole_payload_bytes: u64,
    pub max_registry_entries: usize,
    pub max_container_host_bytes: u64,
    pub max_invocation_host_bytes: u64,
    pub max_invocation_device_bytes: u64,
}

pub struct DevicePagesRegistry<B: DevicePublicBacking> {
    instance_seal: RegistryInstanceSeal,
    namespace: DevicePageNamespace,
    limits: DevicePagesLimits,
    bank: MemoryBank<DevicePublicBlock<B>>,
    index: Vec<ProjectionEntry>,
    blocks: Vec<BlockLocator>,
    container_host_reservation: u64,
    highest_generation: u64,
}
impl<B: DevicePublicBacking> DevicePagesRegistry<B> {
    pub fn new(namespace: DevicePageNamespace, limits: DevicePagesLimits) -> Result<Self> {
        if namespace.process_epoch.0 == 0
            || !(1..=MAX_BLOCKS).contains(&limits.max_blocks)
            || !(1..=MAX_REGISTRY).contains(&limits.max_registry_entries)
            || limits.max_bank_whole_payload_bytes < limits.max_blocks as u64
            || limits.max_container_host_bytes == 0
            || limits.max_invocation_host_bytes == 0
        {
            return Err(DevicePageError::InvalidLimits);
        }
        // Scalar limits/namespace are rejected before consuming an ID. Overflow
        // is rejected before bank/container allocation. Later allocation/budget
        // refusal may consume this unique ID: it is never rolled back or reused,
        // and changes no previously published registry or semantic cache key.
        let instance_seal = allocate_registry_seal(&REGISTRY_INSTANCE_SEQUENCE)?;
        let mut bank = MemoryBank::new(limits.max_blocks, limits.max_bank_whole_payload_bytes)
            .map_err(DevicePageError::Bank)?;
        // Reserve actual bank entry capacity before this registry is published.
        // These unique placeholder keys are never inserted or certified.
        let placeholders: Vec<_> = (0..limits.max_blocks)
            .map(|index| {
                let mut content = [0u8; 32];
                content[..8].copy_from_slice(&(index as u64).to_le_bytes());
                (
                    projection(namespace, PublicPageKind::WholeInput, content),
                    1,
                )
            })
            .collect();
        let bank_slots = bank
            .reserve_batch_backing(&placeholders, limits.max_container_host_bytes)
            .map_err(DevicePageError::Bank)?;
        let mut index = Vec::new();
        let mut blocks = Vec::new();
        index
            .try_reserve_exact(limits.max_registry_entries)
            .map_err(|_| DevicePageError::Allocation)?;
        blocks
            .try_reserve_exact(limits.max_blocks)
            .map_err(|_| DevicePageError::Allocation)?;
        if index.capacity() > limits.max_registry_entries || blocks.capacity() > limits.max_blocks {
            return Err(DevicePageError::BudgetExceeded);
        }
        // A full second index buffer is reserved even when no publication is
        // in progress, so atomic staging never creates an uncharged overlap.
        let container_host_reservation = sum(&[
            size_of::<Self>() as u64,
            bank_slots,
            bytes(index.capacity(), 2 * size_of::<ProjectionEntry>())?,
            bytes(blocks.capacity(), size_of::<BlockLocator>())?,
        ])?;
        if container_host_reservation > limits.max_container_host_bytes {
            return Err(DevicePageError::BudgetExceeded);
        }
        Ok(Self {
            instance_seal,
            namespace,
            limits,
            bank,
            index,
            blocks,
            container_host_reservation,
            highest_generation: 0,
        })
    }
    pub fn snapshot(&self) -> MemoryBankSnapshot {
        self.bank.snapshot()
    }
    pub fn certified_projection_count(&self) -> usize {
        self.index.len()
    }
    pub fn container_host_reservation(&self) -> u64 {
        self.container_host_reservation
    }
    pub fn namespace(&self) -> DevicePageNamespace {
        self.namespace
    }

    /// Publish only the certified slices; retain and charge the entire immutable
    /// block. All fallible identity/capacity/budget work precedes bank eviction.
    pub fn publish(
        &mut self,
        block: DevicePublicBlock<B>,
    ) -> Result<MemoryPin<DevicePublicBlock<B>>> {
        if block.descriptor.namespace != self.namespace {
            return Err(DevicePageError::IdentityMismatch);
        }
        if block.descriptor.block_generation <= self.highest_generation {
            return Err(DevicePageError::StaleGeneration);
        }
        let key = block.descriptor.block_key;
        let generation = block.descriptor.block_generation;
        let payload = block.reservation()?.total()?;
        self.bank
            .preflight_batch(&[(key, payload)])
            .map_err(DevicePageError::Bank)?;
        let mut candidate = Vec::new();
        candidate
            .try_reserve_exact(self.index.capacity())
            .map_err(|_| DevicePageError::Allocation)?;
        if candidate.capacity() > self.index.capacity() {
            return Err(DevicePageError::BudgetExceeded);
        }
        candidate.extend(
            self.index
                .iter()
                .copied()
                .filter(|entry| self.location_current(entry.location)),
        );
        {
            let mut append = |projection_key, offset: usize| -> Result<()> {
                let location = DeviceProjectionOffset {
                    instance_seal: self.instance_seal,
                    block_key: key,
                    block_generation: generation,
                    token_offset: u16::try_from(offset).map_err(|_| DevicePageError::Overflow)?,
                };
                if let Some(entry) = candidate
                    .iter_mut()
                    .find(|entry| entry.projection == projection_key)
                {
                    entry.location = location;
                } else {
                    if candidate.len() == self.limits.max_registry_entries {
                        return Err(DevicePageError::BudgetExceeded);
                    }
                    candidate.push(ProjectionEntry {
                        projection: projection_key,
                        location,
                    });
                }
                Ok(())
            };
            if block.certified_board {
                append(
                    projection(
                        self.namespace,
                        PublicPageKind::ContextualBoard,
                        block.descriptor.board_content,
                    ),
                    0,
                )?;
            }
            for (index, features) in block.descriptor.features.iter().enumerate() {
                if block.certified_records[index] {
                    append(
                        projection(
                            self.namespace,
                            PublicPageKind::IndependentRecordProjection,
                            record_content(self.namespace, features),
                        ),
                        BOARD + index,
                    )?;
                }
            }
        }
        let pin = self
            .bank
            .insert(key, block, payload)
            .map_err(DevicePageError::Bank)?;
        let bank = &self.bank;
        candidate.retain(|entry| {
            bank.get(entry.location.block_key).is_some_and(|owner| {
                owner.descriptor.block_generation == entry.location.block_generation
            })
        });
        self.blocks.retain(|entry| {
            bank.get(entry.key)
                .is_some_and(|owner| owner.descriptor.block_generation == entry.generation)
        });
        self.blocks.push(BlockLocator { key, generation });
        self.index = candidate;
        self.highest_generation = generation;
        Ok(pin)
    }
    fn location_current(&self, location: DeviceProjectionOffset) -> bool {
        if location.instance_seal != self.instance_seal {
            return false;
        }
        self.bank.get(location.block_key).is_some_and(|owner| {
            owner.descriptor.namespace == self.namespace
                && owner.descriptor.block_generation == location.block_generation
                && owner.accepts_offset(usize::from(location.token_offset))
        })
    }
    pub fn acquire_reference(
        &mut self,
        location: DeviceProjectionOffset,
    ) -> Result<MemoryPin<DevicePublicBlock<B>>> {
        if location.instance_seal != self.instance_seal {
            return Err(DevicePageError::IdentityMismatch);
        }
        let owner = self
            .bank
            .get(location.block_key)
            .ok_or(DevicePageError::MissingProjection)?;
        if owner.descriptor.block_generation != location.block_generation {
            return Err(DevicePageError::StaleGeneration);
        }
        if owner.descriptor.namespace != self.namespace {
            return Err(DevicePageError::IdentityMismatch);
        }
        if !owner.accepts_offset(usize::from(location.token_offset)) {
            return Err(DevicePageError::UnvalidatedSlice);
        }
        self.bank
            .acquire(location.block_key)
            .ok_or(DevicePageError::MissingProjection)
    }
    fn locate(&self, key: MemoryKey) -> Option<DeviceProjectionOffset> {
        self.index
            .iter()
            .find(|entry| entry.projection == key && self.location_current(entry.location))
            .map(|entry| entry.location)
    }
    fn zero_base(&self, board_content: [u8; 32]) -> Option<DeviceProjectionOffset> {
        self.blocks.iter().find_map(|entry| {
            let owner = self.bank.get(entry.key)?;
            (owner.descriptor.namespace == self.namespace
                && owner.descriptor.block_generation == entry.generation
                && owner.descriptor.board_content == board_content
                && owner.descriptor.zero_padding()
                && owner.certified_board
                && owner.certified_records[0])
                .then_some(DeviceProjectionOffset {
                    instance_seal: self.instance_seal,
                    block_key: entry.key,
                    block_generation: entry.generation,
                    token_offset: 0,
                })
        })
    }

    /// Returns an owned admission plan, including unique whole-block pins on
    /// hits. A missing plan cannot produce runnable fixed ports. Cache lookup
    /// does not create visits, CPU-task completion or neural execution counts.
    pub fn plan(
        &mut self,
        input: &PalsModelInput,
        config: &PalsModelConfig,
    ) -> Result<DevicePagePlan<B>> {
        if input.model_epoch != self.namespace.model_epoch {
            return Err(DevicePageError::IdentityMismatch);
        }
        if input.role == PalsRole::Validator {
            return Err(DevicePageError::InvalidInput);
        }
        let actual = input
            .independent_public_plan(config)
            .map_err(|_| DevicePageError::InvalidInput)?;
        // Keep full role/history/candidate identity separate from reusable
        // projection identities. A later native consumer must recheck this key.
        let full_input_key = input
            .canonical_input_key(config)
            .map_err(|_| DevicePageError::InvalidInput)?;
        let known_input_host_payload = sum(&[
            size_of::<PalsModelInput>() as u64,
            bytes(input.board.capacity(), size_of::<u8>())?,
            bytes(
                input.records.capacity(),
                size_of::<crate::pals_model::PalsRecordToken>(),
            )?,
            bytes(input.required_critical_records.capacity(), size_of::<u64>())?,
            bytes(
                input.candidates.capacity(),
                size_of::<crate::pals_model::PalsCandidateToken>(),
            )?,
            bytes(input.divergence_features.capacity(), size_of::<[f32; 8]>())?,
        ])?;
        let private_floats = input
            .candidates
            .len()
            .max(1)
            .checked_add(3)
            .and_then(|value| value.checked_add(config.latent_elements()))
            .and_then(|value| {
                value.checked_add(if input.role == PalsRole::Critic {
                    input.divergence_features.len().max(1)
                } else {
                    0
                })
            })
            .ok_or(DevicePageError::Overflow)?;
        // Existing DeviceRole binds CPU outputs; native arrays and extracted raw
        // arrays overlap until validation. Charge both known payloads, with raw
        // inline storage once; extra ORT metadata remains a required declaration.
        let overlapped_floats = private_floats
            .checked_mul(2)
            // Raw WDL lives inline in PalsRawOutput, not in a second Vec.
            .and_then(|value| value.checked_sub(3))
            .ok_or(DevicePageError::Overflow)?;
        let known_private_output_host_payload = bytes(overlapped_floats, 4)?
            .checked_add(size_of::<PalsRawOutput>() as u64)
            .ok_or(DevicePageError::Overflow)?;
        let count = input.records.len();
        if count > DEVICE_PAGE_RECORD_CAPACITY {
            return Err(DevicePageError::InvalidInput);
        }
        let base = if count == 0 {
            self.zero_base(actual.board_content)
        } else {
            self.locate(projection(
                self.namespace,
                PublicPageKind::ContextualBoard,
                actual.board_content,
            ))
        };
        let mut pins = Vec::new();
        let mut records = Vec::new();
        let mut missing = Vec::new();
        pins.try_reserve_exact(count + 1)
            .map_err(|_| DevicePageError::Allocation)?;
        records
            .try_reserve_exact(count)
            .map_err(|_| DevicePageError::Allocation)?;
        missing
            .try_reserve_exact(count.max(1))
            .map_err(|_| DevicePageError::Allocation)?;
        {
            let mut pin_once =
                |location: DeviceProjectionOffset, registry: &mut Self| -> Result<()> {
                    if !pins.iter().any(|pin: &MemoryPin<DevicePublicBlock<B>>| {
                        pin.descriptor.block_key == location.block_key
                            && pin.descriptor.block_generation == location.block_generation
                    }) {
                        pins.push(registry.acquire_reference(location)?);
                    }
                    Ok(())
                };
            if let Some(location) = base {
                pin_once(location, self)?;
            }
            for index in 0..count {
                let key = projection(
                    self.namespace,
                    PublicPageKind::IndependentRecordProjection,
                    actual.record_contents[index],
                );
                let location = self.locate(key).filter(|location| {
                    self.bank.get(location.block_key).is_some_and(|owner| {
                        owner.descriptor.record_mask[usize::from(location.token_offset) - BOARD]
                    })
                });
                if let Some(location) = location {
                    pin_once(location, self)?;
                } else if !missing.iter().any(|old: &usize| {
                    actual.record_contents[*old] == actual.record_contents[index]
                }) {
                    missing.push(index);
                }
                records.push(location);
            }
        }
        // Empty view needs an actual zero-padded base, not a cached arbitrary
        // record at token 66. Empty subset is explicit for its public producer.
        let zero_base_required = count == 0 && base.is_none();
        let current_mask = std::iter::repeat_n(true, BOARD)
            .chain((0..count.max(1)).map(|index| index < count))
            .collect();
        Ok(DevicePagePlan {
            instance_seal: self.instance_seal,
            namespace: self.namespace,
            base,
            records,
            missing,
            pins,
            current_mask,
            zero_base_required,
            full_input_key,
            known_input_host_payload,
            known_private_output_host_payload,
        })
    }
    pub fn clear(&mut self) -> Result<()> {
        self.bank.clear().map_err(DevicePageError::Bank)?;
        self.index.clear();
        self.blocks.clear();
        // The generation high water never resets: old locators cannot ABA.
        Ok(())
    }
    pub fn whole_owner_reservation(&self) -> Result<DevicePagePayload> {
        self.blocks
            .iter()
            .try_fold(DevicePagePayload::default(), |total, entry| {
                match self.bank.get(entry.key) {
                    Some(owner) if owner.descriptor.block_generation == entry.generation => {
                        total.add(owner.reservation()?)
                    }
                    _ => Ok(total),
                }
            })
    }
    pub fn reserve_invocation(
        &self,
        plan: &DevicePagePlan<B>,
        declaration: DevicePageInvocationDeclaration,
    ) -> Result<DevicePageReservation> {
        if plan.instance_seal != self.instance_seal || plan.namespace != self.namespace {
            return Err(DevicePageError::IdentityMismatch);
        }
        if declaration
            .original_input_and_transfer_payload
            .is_some_and(|declared| declared.host < plan.known_input_host_payload)
        {
            return Err(DevicePageError::BudgetExceeded);
        }
        if declaration
            .private_output_payload
            .is_some_and(|declared| declared.host < plan.known_private_output_host_payload)
        {
            return Err(DevicePageError::BudgetExceeded);
        }
        let reservation = DevicePageReservation::calculate(
            self.whole_owner_reservation()?,
            self.container_host_reservation,
            plan.host_bytes()?,
            size_of::<DevicePublicBlock<B>>() as u64,
            self.namespace.domain,
            declaration,
        )?;
        if reservation.total.host > self.limits.max_invocation_host_bytes
            || reservation.total.device > self.limits.max_invocation_device_bytes
        {
            return Err(DevicePageError::BudgetExceeded);
        }
        Ok(reservation)
    }
}

pub struct DevicePagePlan<B: DevicePublicBacking> {
    instance_seal: RegistryInstanceSeal,
    namespace: DevicePageNamespace,
    base: Option<DeviceProjectionOffset>,
    records: Vec<Option<DeviceProjectionOffset>>,
    missing: Vec<usize>,
    pins: Vec<MemoryPin<DevicePublicBlock<B>>>,
    current_mask: Vec<bool>,
    zero_base_required: bool,
    full_input_key: [u8; 32],
    known_input_host_payload: u64,
    known_private_output_host_payload: u64,
}
impl<B: DevicePublicBacking> DevicePagePlan<B> {
    pub fn ready(&self) -> bool {
        self.base.is_some() && self.records.iter().all(Option::is_some)
    }
    pub fn base(&self) -> Option<DeviceProjectionOffset> {
        self.base
    }
    pub fn missing_record_indices(&self) -> &[usize] {
        &self.missing
    }
    pub fn zero_base_required(&self) -> bool {
        self.zero_base_required
    }
    pub fn unique_pinned_owners(&self) -> usize {
        self.pins.len()
    }
    pub fn current_mask(&self) -> &[bool] {
        &self.current_mask
    }
    pub fn memory_tokens(&self) -> usize {
        BOARD + self.records.len().max(1)
    }
    pub fn full_input_key(&self) -> [u8; 32] {
        self.full_input_key
    }
    pub fn known_input_host_payload(&self) -> u64 {
        self.known_input_host_payload
    }
    pub fn known_private_output_host_payload(&self) -> u64 {
        self.known_private_output_host_payload
    }
    pub fn fixed_ports(&self) -> Result<[DeviceProjectionOffset; DEVICE_PAGE_RECORD_CAPACITY]> {
        let base = self.base.ok_or(DevicePageError::MissingProjection)?;
        if !self.ready() {
            return Err(DevicePageError::MissingProjection);
        }
        let inactive = DeviceProjectionOffset {
            token_offset: BOARD as u16,
            ..base
        };
        let mut ports = [inactive; DEVICE_PAGE_RECORD_CAPACITY];
        for (port, record) in ports.iter_mut().zip(&self.records) {
            *port = record.ok_or(DevicePageError::MissingProjection)?;
        }
        // Inactive ports alias a physically owned in-bounds base projection;
        // they are excluded by record_stop, not certified/published by this.
        Ok(ports)
    }
    fn host_bytes(&self) -> Result<u64> {
        sum(&[
            size_of::<Self>() as u64,
            bytes(
                self.records.capacity(),
                size_of::<Option<DeviceProjectionOffset>>(),
            )?,
            bytes(self.missing.capacity(), size_of::<usize>())?,
            bytes(
                self.pins.capacity(),
                size_of::<MemoryPin<DevicePublicBlock<B>>>(),
            )?,
            bytes(self.current_mask.capacity(), size_of::<bool>())?,
        ])
    }
}

/// All three session declarations are mandatory and nonzero. These are supplied
/// limits, not observations or a claim that three sessions already exist.
#[derive(Clone, Copy, Debug)]
pub struct DevicePageInvocationDeclaration {
    pub public_session_bytes: Option<u64>,
    pub packing_session_bytes: Option<u64>,
    pub private_session_bytes: Option<u64>,
    pub private_output_payload: Option<DevicePagePayload>,
    pub original_input_and_transfer_payload: Option<DevicePagePayload>,
    /// Additional join/binding/allocator/transport owner metadata not contained
    /// in the generic whole-block inline size. Native consumer must declare it;
    /// absent/zero is unknown, not an invented zero or physical peak witness.
    pub additional_owner_metadata_payload: Option<DevicePagePayload>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DevicePageReservation {
    pub resident_whole_owners: DevicePagePayload,
    pub container_and_plan_host_bytes: u64,
    pub pending_whole_public_payload: DevicePagePayload,
    pub joined_payload: DevicePagePayload,
    /// Complete graph arithmetic, including the final outputs; never peak.
    pub packing_maximum_node_payload_sum: u64,
    /// Internal node maxima only: separately owned joined K/V and finite output
    /// are removed from the arithmetic to avoid charging those owners twice.
    pub packing_intermediate_payload: DevicePagePayload,
    pub controls_mask_and_finite_host_bytes: u64,
    pub private_outputs: DevicePagePayload,
    pub original_inputs_and_transfer: DevicePagePayload,
    pub additional_owner_metadata: DevicePagePayload,
    pub declared_public_session: DevicePagePayload,
    pub declared_packing_session: DevicePagePayload,
    pub declared_private_session: DevicePagePayload,
    pub declared_session_total: DevicePagePayload,
    pub total: DevicePagePayload,
}
fn known_nonzero(value: Option<u64>) -> Result<u64> {
    value
        .filter(|value| *value > 0)
        .ok_or(DevicePageError::UnknownBudget)
}
/// Layout-v1 node output maxima: two Slice/Gather/Concat/Slice K/V paths,
/// IsNaN/IsInf/Or BOOLs, INT64 Cast/ReduceMax, Equal and final And. The sum is
/// deliberately not peak, allocator/workspace usage or actual stage residency.
pub fn device_packing_maximum_node_payload_sum() -> Result<u64> {
    let kv_path = bytes(
        BOARD + DEVICE_PAGE_RECORD_CAPACITY + 2 * MAX_TOKENS,
        HEADS * DIM * 4,
    )?;
    let finite_path = bytes(MAX_TOKENS, HEADS * DIM * (1 + 1 + 1 + 8))?
        .checked_add(8 + 1)
        .ok_or(DevicePageError::Overflow)?;
    kv_path
        .checked_add(finite_path)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(1))
        .ok_or(DevicePageError::Overflow)
}
impl DevicePageReservation {
    fn calculate(
        resident: DevicePagePayload,
        containers: u64,
        plan: u64,
        pending_inline: u64,
        domain: DevicePageDomain,
        declaration: DevicePageInvocationDeclaration,
    ) -> Result<Self> {
        let declared_public_session =
            DevicePagePayload::in_domain(domain, known_nonzero(declaration.public_session_bytes)?);
        let declared_packing_session =
            DevicePagePayload::in_domain(domain, known_nonzero(declaration.packing_session_bytes)?);
        let declared_private_session =
            DevicePagePayload::in_domain(domain, known_nonzero(declaration.private_session_bytes)?);
        let private_outputs = declaration
            .private_output_payload
            .ok_or(DevicePageError::UnknownBudget)?;
        let original_inputs_and_transfer = declaration
            .original_input_and_transfer_payload
            .ok_or(DevicePageError::UnknownBudget)?;
        let additional_owner_metadata = declaration
            .additional_owner_metadata_payload
            .ok_or(DevicePageError::UnknownBudget)?;
        if private_outputs.total()? == 0
            || original_inputs_and_transfer.total()? == 0
            || additional_owner_metadata.total()? == 0
        {
            return Err(DevicePageError::UnknownBudget);
        }
        let container_and_plan_host_bytes = containers
            .checked_add(plan)
            .ok_or(DevicePageError::Overflow)?;
        // Keep worst-case pending subset and joined view reserved even when
        // missing.len()==0. No "unused transient" subtraction on a hit path.
        let pending_metadata = sum(&[
            bytes(DEVICE_PAGE_RECORD_CAPACITY, 16 * 4)?,
            DEVICE_PAGE_RECORD_CAPACITY as u64,
            MAX_TOKENS as u64,
            pending_inline,
        ])?;
        let pending_whole_public_payload =
            DevicePagePayload::in_domain(domain, kv_bytes(MAX_TOKENS)?).add(DevicePagePayload {
                host: pending_metadata,
                device: 0,
            })?;
        let joined_payload = DevicePagePayload::in_domain(domain, kv_bytes(MAX_TOKENS)?);
        let packing_maximum_node_payload_sum = device_packing_maximum_node_payload_sum()?;
        let internal = packing_maximum_node_payload_sum
            .checked_sub(kv_bytes(MAX_TOKENS)?)
            .and_then(|value| value.checked_sub(1))
            .ok_or(DevicePageError::Overflow)?;
        let packing_intermediate_payload = DevicePagePayload::in_domain(domain, internal);
        let controls_mask_and_finite_host_bytes = sum(&[
            bytes(DEVICE_PAGE_RECORD_CAPACITY + 1, 8)?,
            MAX_TOKENS as u64,
            1,
            bytes(DEVICE_PAGE_RECORD_CAPACITY + 2, size_of::<Vec<u8>>())?,
            bytes(
                DEVICE_PAGE_RECORD_CAPACITY,
                size_of::<DeviceProjectionOffset>(),
            )?,
        ])?;
        let declared_session_total = declared_public_session
            .add(declared_packing_session)?
            .add(declared_private_session)?;
        let mut total = resident.add(DevicePagePayload {
            host: container_and_plan_host_bytes,
            device: 0,
        })?;
        for value in [
            pending_whole_public_payload,
            joined_payload,
            packing_intermediate_payload,
            DevicePagePayload {
                host: controls_mask_and_finite_host_bytes,
                device: 0,
            },
            private_outputs,
            original_inputs_and_transfer,
            additional_owner_metadata,
            declared_session_total,
        ] {
            total = total.add(value)?;
        }
        Ok(Self {
            resident_whole_owners: resident,
            container_and_plan_host_bytes,
            pending_whole_public_payload,
            joined_payload,
            packing_maximum_node_payload_sum,
            packing_intermediate_payload,
            controls_mask_and_finite_host_bytes,
            private_outputs,
            original_inputs_and_transfer,
            additional_owner_metadata,
            declared_public_session,
            declared_packing_session,
            declared_private_session,
            declared_session_total,
            total,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_model::{PalsRecordToken, PalsRole};
    fn namespace() -> DevicePageNamespace {
        DevicePageNamespace {
            process_epoch: ProcessEpoch(9),
            model_manifest: [1; 32],
            model_epoch: [2; 32],
            public_graph: [3; 32],
            encoding: [4; 32],
            frozen_epoch: 0,
            game_generation: 1,
            domain: DevicePageDomain::CpuFixture,
        }
    }
    fn input(values: &[f32], board: u8) -> PalsModelInput {
        PalsModelInput {
            role: PalsRole::Proposer,
            board: vec![board; 64],
            metadata: [0.; 16],
            records: values
                .iter()
                .enumerate()
                .map(|(index, value)| PalsRecordToken {
                    record_id: index as u64,
                    revision: 1,
                    critical: false,
                    features: [*value; 16],
                })
                .collect(),
            required_critical_records: vec![],
            candidates: vec![],
            divergence_features: vec![],
            query: [0.; 16],
            situation_revision: 1,
            history_digest: [5; 32],
            model_epoch: namespace().model_epoch,
        }
    }
    fn limits(blocks: usize) -> DevicePagesLimits {
        DevicePagesLimits {
            max_blocks: blocks,
            max_bank_whole_payload_bytes: 8 * 1024 * 1024,
            max_registry_entries: 512,
            max_container_host_bytes: 1024 * 1024,
            max_invocation_host_bytes: 16 * 1024 * 1024,
            max_invocation_device_bytes: 0,
        }
    }
    fn block(
        input: &PalsModelInput,
        generation: u64,
        subset: &[usize],
        board: bool,
        certified: &[usize],
    ) -> DevicePublicBlock<CpuOwnedPublicBacking> {
        let descriptor = DevicePublicBlockDescriptor::from_input(
            namespace(),
            generation,
            input,
            &PalsModelConfig::default(),
            subset,
        )
        .unwrap();
        let tokens = descriptor.tokens();
        let backing = CpuOwnedPublicBacking::new(
            tokens,
            vec![1.; HEADS * tokens * DIM],
            vec![2.; HEADS * tokens * DIM],
        )
        .unwrap();
        DevicePublicBlock::certify_cpu(descriptor, backing, board, certified).unwrap()
    }
    fn declaration() -> DevicePageInvocationDeclaration {
        DevicePageInvocationDeclaration {
            public_session_bytes: Some(10),
            packing_session_bytes: Some(20),
            private_session_bytes: Some(30),
            private_output_payload: Some(DevicePagePayload {
                host: 64 * 1024,
                device: 0,
            }),
            original_input_and_transfer_payload: Some(DevicePagePayload {
                host: 64 * 1024,
                device: 0,
            }),
            additional_owner_metadata_payload: Some(DevicePagePayload {
                host: 64,
                device: 0,
            }),
        }
    }
    #[test]
    fn ordered_duplicate_ports_pin_one_whole_owner_once() {
        let mut registry = DevicePagesRegistry::new(namespace(), limits(2)).unwrap();
        let source = input(&[10., 20.], 0);
        let pin = registry
            .publish(block(&source, 1, &[0, 1], true, &[0, 1]))
            .unwrap();
        let full = registry.snapshot().reserved_bytes;
        drop(pin);
        let request = input(&[20., 10., 20.], 0);
        let plan = registry
            .plan(&request, &PalsModelConfig::default())
            .unwrap();
        assert!(plan.ready());
        assert_eq!(plan.unique_pinned_owners(), 1);
        let ports = plan.fixed_ports().unwrap();
        assert_eq!(
            [
                ports[0].token_offset,
                ports[1].token_offset,
                ports[2].token_offset
            ],
            [67, 66, 67]
        );
        assert_eq!(ports[0], ports[2]);
        assert_eq!(registry.snapshot().reserved_bytes, full);
        assert_eq!(registry.snapshot().pinned_entries, 1);
        assert_eq!(plan.current_mask(), &[true; 69]);
        let mut different_history = request.clone();
        different_history.history_digest = [8; 32];
        let history_plan = registry
            .plan(&different_history, &PalsModelConfig::default())
            .unwrap();
        assert!(history_plan.ready());
        assert_ne!(plan.full_input_key(), history_plan.full_input_key());
    }
    #[test]
    fn zero_view_requires_actual_zero_projection_and_its_certification() {
        let mut registry = DevicePagesRegistry::new(namespace(), limits(3)).unwrap();
        drop(
            registry
                .publish(block(&input(&[9.], 0), 1, &[0], true, &[0]))
                .unwrap(),
        );
        let empty = input(&[], 0);
        let missing = registry.plan(&empty, &PalsModelConfig::default()).unwrap();
        assert!(!missing.ready() && missing.zero_base_required());
        assert!(missing.fixed_ports().is_err());
        drop(missing);
        drop(registry.publish(block(&empty, 2, &[], true, &[])).unwrap());
        assert!(!registry
            .plan(&empty, &PalsModelConfig::default())
            .unwrap()
            .ready());
        registry.clear().unwrap();
        drop(registry.publish(block(&empty, 3, &[], true, &[0])).unwrap());
        let ready = registry.plan(&empty, &PalsModelConfig::default()).unwrap();
        assert!(ready.ready());
        assert_eq!(ready.memory_tokens(), 67);
        assert!(!ready.current_mask()[66]);
        assert_eq!(ready.fixed_ports().unwrap()[0].token_offset, 66);
    }
    #[test]
    fn uncertified_record_never_enters_lookup_even_when_backing_exists() {
        let mut registry = DevicePagesRegistry::new(namespace(), limits(2)).unwrap();
        let request = input(&[1., 2.], 0);
        let descriptor = DevicePublicBlockDescriptor::from_input(
            namespace(),
            1,
            &request,
            &PalsModelConfig::default(),
            &[0, 1],
        )
        .unwrap();
        let tokens = descriptor.tokens();
        let mut key = vec![1.; HEADS * tokens * DIM];
        key[(BOARD + 1) * DIM] = f32::NAN;
        let backing =
            CpuOwnedPublicBacking::new(tokens, key, vec![2.; HEADS * tokens * DIM]).unwrap();
        let completed = DevicePublicBlock::certify_cpu(descriptor, backing, true, &[0]).unwrap();
        drop(registry.publish(completed).unwrap());
        assert_eq!(registry.certified_projection_count(), 2);
        let plan = registry
            .plan(&request, &PalsModelConfig::default())
            .unwrap();
        assert_eq!(plan.missing_record_indices(), &[1]);
        assert!(!plan.ready());
    }
    #[test]
    fn live_pin_refuses_eviction_and_clear_without_mutation() {
        let mut registry = DevicePagesRegistry::new(namespace(), limits(1)).unwrap();
        let pin = registry
            .publish(block(&input(&[1.], 0), 1, &[0], true, &[0]))
            .unwrap();
        let before = registry.snapshot();
        let references = registry.certified_projection_count();
        assert!(registry
            .publish(block(&input(&[2.], 1), 2, &[0], true, &[0]))
            .is_err());
        assert!(registry.clear().is_err());
        assert_eq!(registry.snapshot().reserved_bytes, before.reserved_bytes);
        assert_eq!(registry.snapshot().entries, before.entries);
        assert_eq!(registry.certified_projection_count(), references);
        drop(pin);
        registry.clear().unwrap();
    }
    #[test]
    fn nonowning_reference_does_not_pin_and_aba_is_rejected() {
        let mut registry = DevicePagesRegistry::new(namespace(), limits(1)).unwrap();
        let original = input(&[1.], 0);
        drop(
            registry
                .publish(block(&original, 1, &[0], true, &[0]))
                .unwrap(),
        );
        let plan = registry
            .plan(&original, &PalsModelConfig::default())
            .unwrap();
        let old = plan.fixed_ports().unwrap()[0];
        drop(plan);
        assert_eq!(registry.snapshot().pinned_entries, 0);
        drop(
            registry
                .publish(block(&input(&[2.], 1), 2, &[0], true, &[0]))
                .unwrap(),
        );
        drop(
            registry
                .publish(block(&original, 3, &[0], true, &[0]))
                .unwrap(),
        );
        assert!(matches!(
            registry.acquire_reference(old),
            Err(DevicePageError::StaleGeneration)
        ));
        let new = registry
            .plan(&original, &PalsModelConfig::default())
            .unwrap()
            .fixed_ports()
            .unwrap()[0];
        assert_eq!(old.block_key, new.block_key);
        assert_ne!(old.block_generation, new.block_generation);
    }
    #[test]
    fn same_namespace_empty_registry_rejects_foreign_plan_and_reference() {
        let mut owner = DevicePagesRegistry::new(namespace(), limits(1)).unwrap();
        let request = input(&[1.], 0);
        drop(owner.publish(block(&request, 1, &[0], true, &[0])).unwrap());
        let plan = owner.plan(&request, &PalsModelConfig::default()).unwrap();
        assert!(plan.ready());
        let location = plan.fixed_ports().unwrap()[0];
        let before = owner.snapshot();
        let mut empty =
            DevicePagesRegistry::<CpuOwnedPublicBacking>::new(namespace(), limits(1)).unwrap();
        assert_eq!(owner.namespace(), empty.namespace());
        assert!(!empty.location_current(location));
        assert!(matches!(
            empty.acquire_reference(location),
            Err(DevicePageError::IdentityMismatch)
        ));
        assert!(matches!(
            empty.reserve_invocation(&plan, declaration()),
            Err(DevicePageError::IdentityMismatch)
        ));
        assert_eq!(empty.snapshot().entries, 0);
        assert_eq!(empty.snapshot().reserved_bytes, 0);
        assert_eq!(owner.snapshot().entries, before.entries);
        assert_eq!(owner.snapshot().reserved_bytes, before.reserved_bytes);
        assert_eq!(owner.snapshot().pinned_entries, before.pinned_entries);
        drop(plan);
        assert_eq!(owner.snapshot().pinned_entries, 0);
    }
    #[test]
    fn reconstructed_key_and_generation_do_not_rebind_foreign_origin() {
        let mut owner = DevicePagesRegistry::new(namespace(), limits(1)).unwrap();
        let mut reconstructed = DevicePagesRegistry::new(namespace(), limits(1)).unwrap();
        let request = input(&[1.], 0);
        drop(owner.publish(block(&request, 1, &[0], true, &[0])).unwrap());
        let descriptor = DevicePublicBlockDescriptor::from_input(
            namespace(),
            1,
            &request,
            &PalsModelConfig::default(),
            &[0],
        )
        .unwrap();
        let tokens = descriptor.tokens();
        // Same projection input/key/generation, independently owned larger
        // arrays with different CPU fixture values. Only this bank is billed.
        let mut key = Vec::with_capacity(HEADS * MAX_TOKENS * DIM);
        key.resize(HEADS * tokens * DIM, 3.);
        let mut value = Vec::with_capacity(HEADS * MAX_TOKENS * DIM);
        value.resize(HEADS * tokens * DIM, 4.);
        let backing = CpuOwnedPublicBacking::new(tokens, key, value).unwrap();
        drop(
            reconstructed
                .publish(DevicePublicBlock::certify_cpu(descriptor, backing, true, &[0]).unwrap())
                .unwrap(),
        );
        let owner_plan = owner.plan(&request, &PalsModelConfig::default()).unwrap();
        let reconstructed_plan = reconstructed
            .plan(&request, &PalsModelConfig::default())
            .unwrap();
        let original = owner_plan.fixed_ports().unwrap()[0];
        let local = reconstructed_plan.fixed_ports().unwrap()[0];
        assert_eq!(original.block_key, local.block_key);
        assert_eq!(original.block_generation, local.block_generation);
        assert_eq!(original.token_offset, local.token_offset);
        assert_ne!(original, local);
        assert!(!reconstructed.location_current(original));
        assert!(matches!(
            reconstructed.acquire_reference(original),
            Err(DevicePageError::IdentityMismatch)
        ));
        assert!(matches!(
            reconstructed.reserve_invocation(&owner_plan, declaration()),
            Err(DevicePageError::IdentityMismatch)
        ));
        let owner_bill = owner
            .reserve_invocation(&owner_plan, declaration())
            .unwrap();
        let local_bill = reconstructed
            .reserve_invocation(&reconstructed_plan, declaration())
            .unwrap();
        assert_eq!(
            local_bill.resident_whole_owners,
            reconstructed.whole_owner_reservation().unwrap()
        );
        assert_eq!(
            local_bill.resident_whole_owners.host,
            reconstructed.snapshot().reserved_bytes
        );
        assert!(local_bill.resident_whole_owners.host > owner_bill.resident_whole_owners.host);
        assert_eq!(reconstructed.snapshot().pinned_entries, 1);
        drop(reconstructed_plan);
        drop(owner_plan);
        assert_eq!(reconstructed.snapshot().pinned_entries, 0);
        assert_eq!(owner.snapshot().pinned_entries, 0);
        assert!(reconstructed.location_current(local));
        let reacquired = reconstructed.acquire_reference(local).unwrap();
        assert_eq!(reconstructed.snapshot().pinned_entries, 1);
        drop(reacquired);
        assert_eq!(reconstructed.snapshot().pinned_entries, 0);
    }
    #[test]
    fn private_registry_sequence_is_unique_and_overflow_never_resets() {
        // Exercise the same allocator with local atomics; never reset or alter
        // the production global sequence, including under parallel fixtures.
        let local = AtomicU64::new(0);
        assert_eq!(
            allocate_registry_seal(&local).unwrap(),
            RegistryInstanceSeal(1)
        );
        assert_eq!(
            allocate_registry_seal(&local).unwrap(),
            RegistryInstanceSeal(2)
        );
        let almost_exhausted = AtomicU64::new(u64::MAX - 1);
        assert_eq!(
            allocate_registry_seal(&almost_exhausted).unwrap(),
            RegistryInstanceSeal(u64::MAX)
        );
        assert!(matches!(
            allocate_registry_seal(&almost_exhausted),
            Err(DevicePageError::Overflow)
        ));
        assert!(matches!(
            allocate_registry_seal(&almost_exhausted),
            Err(DevicePageError::Overflow)
        ));
        assert_eq!(almost_exhausted.load(Ordering::Relaxed), u64::MAX);
    }
    #[test]
    fn namespace_rejection_preserves_registry_and_missing_duplicate_subset_is_unique() {
        let mut registry = DevicePagesRegistry::new(namespace(), limits(2)).unwrap();
        let request = input(&[1., 2., 1.], 0);
        let plan = registry
            .plan(&request, &PalsModelConfig::default())
            .unwrap();
        assert_eq!(plan.missing_record_indices(), &[0, 1]);
        drop(plan);
        let mut foreign = namespace();
        foreign.game_generation += 1;
        let descriptor = DevicePublicBlockDescriptor::from_input(
            foreign,
            1,
            &request,
            &PalsModelConfig::default(),
            &[0, 1],
        )
        .unwrap();
        let tokens = descriptor.tokens();
        let backing = CpuOwnedPublicBacking::new(
            tokens,
            vec![1.; HEADS * tokens * DIM],
            vec![2.; HEADS * tokens * DIM],
        )
        .unwrap();
        assert!(matches!(
            registry.publish(
                DevicePublicBlock::certify_cpu(descriptor, backing, true, &[0, 1]).unwrap()
            ),
            Err(DevicePageError::IdentityMismatch)
        ));
        assert_eq!(registry.snapshot().entries, 0);
        assert_eq!(registry.certified_projection_count(), 0);
    }
    #[test]
    fn full_capacity_bounds_and_duplicate_subset_producer_rejection() {
        let request = input(&(0..128).map(|i| i as f32).collect::<Vec<_>>(), 0);
        let mut registry = DevicePagesRegistry::new(namespace(), limits(1)).unwrap();
        let subset: Vec<_> = (0..128).collect();
        drop(
            registry
                .publish(block(&request, 1, &subset, true, &subset))
                .unwrap(),
        );
        let plan = registry
            .plan(&request, &PalsModelConfig::default())
            .unwrap();
        assert_eq!(plan.fixed_ports().unwrap()[127].token_offset, 193);
        assert_eq!(plan.memory_tokens(), 194);
        assert!(DevicePublicBlockDescriptor::from_input(
            namespace(),
            2,
            &request,
            &PalsModelConfig::default(),
            &[128]
        )
        .is_err());
        let duplicates = input(&[3., 3.], 0);
        assert!(DevicePublicBlockDescriptor::from_input(
            namespace(),
            2,
            &duplicates,
            &PalsModelConfig::default(),
            &[0, 1]
        )
        .is_err());
    }
    #[test]
    fn hit_keeps_all_declared_unused_transient_stages_and_unknown_is_not_zero() {
        let mut registry = DevicePagesRegistry::new(namespace(), limits(1)).unwrap();
        let request = input(&[1.], 0);
        let miss = registry
            .plan(&request, &PalsModelConfig::default())
            .unwrap();
        let before = registry.reserve_invocation(&miss, declaration()).unwrap();
        drop(miss);
        drop(
            registry
                .publish(block(&request, 1, &[0], true, &[0]))
                .unwrap(),
        );
        let hit = registry
            .plan(&request, &PalsModelConfig::default())
            .unwrap();
        assert!(hit.ready());
        let after = registry.reserve_invocation(&hit, declaration()).unwrap();
        assert_eq!(
            before.pending_whole_public_payload,
            after.pending_whole_public_payload
        );
        assert_eq!(before.joined_payload, after.joined_payload);
        assert_eq!(
            before.packing_maximum_node_payload_sum,
            after.packing_maximum_node_payload_sum
        );
        assert_eq!(after.declared_session_total.host, 60);
        assert_eq!(
            [
                after.declared_public_session.host,
                after.declared_packing_session.host,
                after.declared_private_session.host
            ],
            [10, 20, 30]
        );
        assert_eq!(
            device_packing_maximum_node_payload_sum().unwrap(),
            1_142_291
        );
        assert_eq!(
            after.packing_intermediate_payload.host + after.joined_payload.host + 1,
            after.packing_maximum_node_payload_sum
        );
        let mut unknown = declaration();
        unknown.packing_session_bytes = None;
        assert!(matches!(
            registry.reserve_invocation(&hit, unknown),
            Err(DevicePageError::UnknownBudget)
        ));
        unknown.packing_session_bytes = Some(0);
        assert!(matches!(
            registry.reserve_invocation(&hit, unknown),
            Err(DevicePageError::UnknownBudget)
        ));
        unknown = declaration();
        unknown.additional_owner_metadata_payload = None;
        assert!(matches!(
            registry.reserve_invocation(&hit, unknown),
            Err(DevicePageError::UnknownBudget)
        ));
        let mut overflow = declaration();
        overflow.public_session_bytes = Some(u64::MAX);
        assert!(matches!(
            registry.reserve_invocation(&hit, overflow),
            Err(DevicePageError::Overflow)
        ));
    }
    #[test]
    fn failed_budget_or_registry_capacity_preserves_old_owners_and_refs() {
        let mut small = limits(2);
        small.max_registry_entries = 2;
        let mut registry = DevicePagesRegistry::new(namespace(), small).unwrap();
        drop(
            registry
                .publish(block(&input(&[1.], 0), 1, &[0], true, &[0]))
                .unwrap(),
        );
        let before = registry.snapshot();
        assert!(matches!(
            registry.publish(block(&input(&[2.], 1), 2, &[0], true, &[0])),
            Err(DevicePageError::BudgetExceeded)
        ));
        assert_eq!(registry.snapshot().entries, before.entries);
        assert_eq!(registry.snapshot().reserved_bytes, before.reserved_bytes);
        assert_eq!(registry.certified_projection_count(), 2);
        let plan = registry
            .plan(&input(&[1.], 0), &PalsModelConfig::default())
            .unwrap();
        let mut too_large = declaration();
        too_large.private_output_payload = Some(DevicePagePayload {
            host: 32 * 1024 * 1024,
            device: 0,
        });
        assert!(matches!(
            registry.reserve_invocation(&plan, too_large),
            Err(DevicePageError::BudgetExceeded)
        ));
        let mut undersized_input = declaration();
        undersized_input.original_input_and_transfer_payload =
            Some(DevicePagePayload { host: 1, device: 0 });
        assert!(matches!(
            registry.reserve_invocation(&plan, undersized_input),
            Err(DevicePageError::BudgetExceeded)
        ));
        let mut undersized_private = declaration();
        undersized_private.private_output_payload = Some(DevicePagePayload { host: 1, device: 0 });
        assert!(matches!(
            registry.reserve_invocation(&plan, undersized_private),
            Err(DevicePageError::BudgetExceeded)
        ));
        assert_eq!(registry.snapshot().entries, before.entries);
    }
    #[test]
    fn whole_capacity_is_charged_and_cpu_cannot_certify_cuda_namespace() {
        let descriptor = DevicePublicBlockDescriptor::from_input(
            namespace(),
            1,
            &input(&[1.], 0),
            &PalsModelConfig::default(),
            &[0],
        )
        .unwrap();
        let tokens = descriptor.tokens();
        let mut key = Vec::with_capacity(HEADS * MAX_TOKENS * DIM);
        key.resize(HEADS * tokens * DIM, 1.);
        let mut value = Vec::with_capacity(HEADS * MAX_TOKENS * DIM);
        value.resize(HEADS * tokens * DIM, 2.);
        let expected = bytes(key.capacity() + value.capacity(), 4).unwrap();
        let owner = DevicePublicBlock::certify_cpu(
            descriptor,
            CpuOwnedPublicBacking::new(tokens, key, value).unwrap(),
            true,
            &[0],
        )
        .unwrap();
        assert!(owner.reservation().unwrap().host > expected);
        let mut cuda = namespace();
        cuda.domain = DevicePageDomain::Cuda {
            device_id: 0,
            runtime_sha256: [6; 32],
        };
        let descriptor = DevicePublicBlockDescriptor::from_input(
            cuda,
            1,
            &input(&[1.], 0),
            &PalsModelConfig::default(),
            &[0],
        )
        .unwrap();
        let backing = CpuOwnedPublicBacking::new(
            tokens,
            vec![1.; HEADS * tokens * DIM],
            vec![2.; HEADS * tokens * DIM],
        )
        .unwrap();
        assert!(DevicePublicBlock::certify_cpu(descriptor, backing, true, &[0]).is_err());
    }
}
