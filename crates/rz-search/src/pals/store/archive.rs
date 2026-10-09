//! Atomic, bounded archive storage. Cold records have no growing RAM directory.
//! Lookup scans only this game's committed segments under an explicit I/O budget.
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

mod codec;
use codec::{Bundle, LineWire, ObservationWire, RoleRecordWire, StateWire};
const MAGIC: &[u8; 8] = b"RZPALS01";
const HEADER: u64 = 56;
pub const GAME_ARCHIVE_BYTES: u64 = 256 * 1024 * 1024;
pub const GLOBAL_ARCHIVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct ArchiveConfig {
    pub root: PathBuf,
    pub repository_root: PathBuf,
    pub game_bytes: u64,
    pub global_bytes: u64,
    pub record_bytes: u64,
    pub max_load_pins: usize,
}
impl ArchiveConfig {
    pub fn new(root: PathBuf, repository_root: PathBuf) -> Self {
        Self {
            root,
            repository_root,
            game_bytes: GAME_ARCHIVE_BYTES,
            global_bytes: GLOBAL_ARCHIVE_BYTES,
            record_bytes: 16 * 1024 * 1024,
            max_load_pins: 16,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ArchiveIoBudget {
    pub deadline: Instant,
    pub max_bytes: u64,
}
impl ArchiveIoBudget {
    fn check(self) -> Result<(), StoreError> {
        if Instant::now() >= self.deadline {
            Err(StoreError::ArchiveDeadline)
        } else {
            Ok(())
        }
    }
    fn debit(&mut self, bytes: u64) -> Result<(), StoreError> {
        self.check()?;
        self.max_bytes = self
            .max_bytes
            .checked_sub(bytes)
            .ok_or(StoreError::ArchiveByteBudget)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StorePins {
    pub states: BTreeSet<StateId>,
    pub lines: BTreeSet<LineId>,
    pub observations: BTreeSet<ObservationId>,
    pub situations: BTreeSet<SituationId>,
    pub executions: BTreeSet<ExecutionId>,
}
impl StorePins {
    fn extend(&mut self, other: &Self) {
        self.states.extend(&other.states);
        self.lines.extend(&other.lines);
        self.observations.extend(&other.observations);
        self.situations.extend(&other.situations);
        self.executions.extend(&other.executions);
    }
    fn count(&self) -> usize {
        self.states.len()
            + self.lines.len()
            + self.observations.len()
            + self.situations.len()
            + self.executions.len()
    }
}

/// A handle proves the live archive owner and committed segment generation.
/// The serial ID is the original logical record, never a reused hot slot.
pub struct ColdHandle<T> {
    owner: Weak<()>,
    generation: u64,
    checksum: [u8; 32],
    record: T,
    _kind: PhantomData<T>,
}
impl<T: Copy> Clone for ColdHandle<T> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            generation: self.generation,
            checksum: self.checksum,
            record: self.record,
            _kind: PhantomData,
        }
    }
}
impl<T: std::fmt::Debug> std::fmt::Debug for ColdHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColdHandle")
            .field("generation", &self.generation)
            .field("record", &self.record)
            .finish_non_exhaustive()
    }
}
impl<T: Copy> ColdHandle<T> {
    pub fn original_id(&self) -> T {
        self.record
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArchiveStats {
    pub states: usize,
    pub lines: usize,
    pub situations: usize,
    pub observations: usize,
    pub dependencies: usize,
    pub executions: usize,
    pub consumers: usize,
    pub history_bytes: usize,
}
impl ArchiveStats {
    fn hot(store: &PalsStores) -> Self {
        Self {
            states: store.states.len(),
            lines: store.lines.len(),
            situations: store.situations.len(),
            observations: store.observations.len(),
            dependencies: store.dependencies.len(),
            executions: store.tasks.len(),
            consumers: store.tasks.consumer_count(),
            history_bytes: store.states.retained_history_bytes(),
        }
    }
}
#[derive(Debug)]
pub struct ArchiveReceipt {
    owner: Weak<()>,
    generation: u64,
    checksum: [u8; 32],
    pub before: ArchiveStats,
    pub after: ArchiveStats,
    pub committed_bytes: u64,
    pub io_bytes: u64,
}
impl ArchiveReceipt {
    fn handle<T: Copy>(&self, record: T) -> ColdHandle<T> {
        ColdHandle {
            owner: self.owner.clone(),
            generation: self.generation,
            checksum: self.checksum,
            record,
            _kind: PhantomData,
        }
    }
    pub fn state_handle(&self, id: StateId) -> ColdHandle<StateId> {
        self.handle(id)
    }
    pub fn line_handle(&self, id: LineId) -> ColdHandle<LineId> {
        self.handle(id)
    }
    pub fn observation_handle(&self, id: ObservationId) -> ColdHandle<ObservationId> {
        self.handle(id)
    }
    pub fn execution_handle(&self, id: ExecutionId) -> ColdHandle<ExecutionId> {
        self.handle(id)
    }
    pub fn situation_handle(&self, id: SituationId) -> ColdHandle<SituationId> {
        self.handle(id)
    }
}
#[derive(Debug)]
pub struct ArchiveLoadPin {
    owner: Weak<()>,
    serial: u64,
}
#[derive(Debug)]
pub struct LoadedArchive {
    pub pin: ArchiveLoadPin,
    pub situations: Vec<(SituationId, SituationId)>,
    pub engine_records: Vec<crate::pals::engine::RoleRecord>,
    pub engine_nodes: Vec<EngineArchiveNode>,
    pub io_bytes: u64,
}
/// Ordered edges use original exact-state IDs; metadata is caller-defined,
/// versioned bytes for the remaining engine-owned node fields (maximum 4KiB).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineArchiveNode {
    pub state: StateId,
    pub situation: SituationId,
    pub edges: Vec<(Move16, Option<StateId>)>,
    pub metadata: Vec<u8>,
}
#[derive(Debug)]
pub struct EngineArchiveReceipt {
    pub archive: ArchiveReceipt,
    pub records: usize,
    pub nodes: usize,
    actual_root: Option<(SituationId, StateId, u64)>,
}

mod sealed {
    pub trait Sealed {}
}
/// Sealed record kinds prevent callers from inventing another archive namespace.
pub trait ArchiveRecordKind: sealed::Sealed + Copy {
    #[doc(hidden)]
    fn seed(self, pins: &mut StorePins);
}
macro_rules! kind {
    ($t:ty,$field:ident) => {
        impl sealed::Sealed for $t {}
        impl ArchiveRecordKind for $t {
            fn seed(self, p: &mut StorePins) {
                p.$field.insert(self);
            }
        }
    };
}
kind!(StateId, states);
kind!(LineId, lines);
kind!(ObservationId, observations);
kind!(ExecutionId, executions);
kind!(SituationId, situations);

#[derive(Debug)]
pub(super) struct ArchiveManager {
    config: ArchiveConfig,
    identity: Arc<()>,
    directory: PathBuf,
    next_generation: u64,
    pins: StorePins,
    load_pins: BTreeMap<u64, StorePins>,
    next_pin: u64,
    auto_budget: Option<ArchiveIoBudget>,
    state_owner: Weak<()>,
    line_owner: Weak<()>,
}
fn io(stage: &'static str, error: std::io::Error) -> StoreError {
    StoreError::ArchiveIo {
        stage,
        kind: error.kind(),
    }
}
fn no_link(path: &Path) -> Result<(), StoreError> {
    let metadata = fs::symlink_metadata(path).map_err(|e| io("archive path metadata", e))?;
    if metadata.file_type().is_symlink() {
        return Err(StoreError::ArchiveIntegrity("archive symbolic link"));
    }
    // Windows junctions are reparse points as well as ordinary symlinks.
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err(StoreError::ArchiveIntegrity("archive reparse point"));
        }
    }
    Ok(())
}
impl ArchiveManager {
    fn open(config: ArchiveConfig) -> Result<Self, StoreError> {
        if !config.root.is_absolute()
            || !config.repository_root.is_absolute()
            || config.game_bytes == 0
            || config.global_bytes == 0
            || config.game_bytes > GAME_ARCHIVE_BYTES
            || config.global_bytes > GLOBAL_ARCHIVE_BYTES
            || config.game_bytes > config.global_bytes
            || config.record_bytes == 0
            || config.record_bytes > config.game_bytes
            || config.max_load_pins == 0
            || config.max_load_pins > 16
        {
            return Err(StoreError::InvalidConditions("archive path or quotas"));
        }
        let repository =
            fs::canonicalize(&config.repository_root).map_err(|e| io("repository path", e))?;
        let mut ancestor = config.root.as_path();
        while !ancestor.exists() {
            ancestor = ancestor
                .parent()
                .ok_or(StoreError::InvalidConditions("archive root"))?;
        }
        for path in ancestor.ancestors() {
            no_link(path)?;
        }
        let ancestor = fs::canonicalize(ancestor).map_err(|e| io("archive ancestor", e))?;
        if ancestor.starts_with(&repository) {
            return Err(StoreError::InvalidConditions(
                "archive must be outside repository",
            ));
        }
        fs::create_dir_all(&config.root).map_err(|e| io("archive root create", e))?;
        no_link(&config.root)?;
        let root =
            fs::canonicalize(&config.root).map_err(|e| io("archive root canonicalize", e))?;
        if root.starts_with(&repository) {
            return Err(StoreError::InvalidConditions(
                "archive must be outside repository",
            ));
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| StoreError::InvalidConditions("archive clock"))?
            .as_nanos();
        static OWNERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let serial = OWNERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let directory = root.join(format!("game-{now}-{}-{serial}", std::process::id()));
        fs::create_dir(&directory).map_err(|e| io("archive game create", e))?;
        Ok(Self {
            config: ArchiveConfig { root, ..config },
            identity: Arc::new(()),
            directory,
            next_generation: 0,
            pins: StorePins::default(),
            load_pins: BTreeMap::new(),
            next_pin: 0,
            auto_budget: None,
            state_owner: Weak::new(),
            line_owner: Weak::new(),
        })
    }
    fn path(&self, generation: u64) -> PathBuf {
        self.directory.join(format!("seg-{generation:020}.arc"))
    }
    fn verify_owner<T>(&self, h: &ColdHandle<T>) -> Result<(), StoreError> {
        if !h.owner.ptr_eq(&Arc::downgrade(&self.identity)) || h.generation >= self.next_generation
        {
            return Err(StoreError::InvalidHandle("cold owner or generation"));
        }
        Ok(())
    }
    fn verify_hot_owner(&self, store: &PalsStores) -> Result<(), StoreError> {
        if !self
            .state_owner
            .ptr_eq(&Arc::downgrade(&store.states.identity))
            || !self
                .line_owner
                .ptr_eq(&Arc::downgrade(&store.lines.identity))
        {
            return Err(StoreError::InvalidHandle("archive hot-store owner changed"));
        }
        Ok(())
    }
    fn usage(&self, budget: ArchiveIoBudget) -> Result<(u64, u64), StoreError> {
        let mut total = 0u64;
        let mut game = 0u64;
        for entry in
            fs::read_dir(&self.config.root).map_err(|e| io("global archive directory", e))?
        {
            budget.check()?;
            let entry = entry.map_err(|e| io("global archive entry", e))?;
            let path = entry.path();
            if entry.file_name() == ".quota.lock" {
                continue;
            }
            no_link(&path)?;
            if !entry
                .file_type()
                .map_err(|e| io("global archive type", e))?
                .is_dir()
                || !entry.file_name().to_string_lossy().starts_with("game-")
            {
                return Err(StoreError::ArchiveIntegrity(
                    "archive root is not exclusively managed",
                ));
            }
            for record in fs::read_dir(&path).map_err(|e| io("game archive directory", e))? {
                budget.check()?;
                let record = record.map_err(|e| io("game archive entry", e))?;
                no_link(&record.path())?;
                let metadata = record.metadata().map_err(|e| io("archive size", e))?;
                if !metadata.is_file() {
                    return Err(StoreError::ArchiveIntegrity("archive record type"));
                }
                total = total
                    .checked_add(metadata.len())
                    .ok_or(StoreError::ArchiveQuota("global archive overflow"))?;
                if path == self.directory {
                    game = game
                        .checked_add(metadata.len())
                        .ok_or(StoreError::ArchiveQuota("game archive overflow"))?;
                }
            }
        }
        Ok((game, total))
    }
    fn commit(
        &mut self,
        bundle: &Bundle,
        budget: &mut ArchiveIoBudget,
        before: ArchiveStats,
    ) -> Result<ArchiveReceipt, StoreError> {
        budget.check()?;
        let next = self
            .next_generation
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        let lock = QuotaLock::take(&self.config.root)?;
        let (game, total) = self.usage(*budget)?;
        let remaining = self
            .config
            .game_bytes
            .checked_sub(game)
            .ok_or(StoreError::ArchiveQuota("game archive"))?
            .min(
                self.config
                    .global_bytes
                    .checked_sub(total)
                    .ok_or(StoreError::ArchiveQuota("global archive"))?,
            );
        let maximum = self
            .config
            .record_bytes
            .min(remaining.saturating_sub(HEADER))
            .min(budget.max_bytes.saturating_sub(2 * HEADER) / 2);
        if maximum == 0 {
            return Err(if remaining <= HEADER {
                StoreError::ArchiveQuota("archive disk bytes")
            } else {
                StoreError::ArchiveByteBudget
            });
        }
        let mut writer = BoundedBuffer {
            bytes: Vec::new(),
            maximum,
            deadline: budget.deadline,
            error: None,
        };
        if serde_json::to_writer(&mut writer, bundle).is_err() {
            let error = writer
                .error
                .unwrap_or(StoreError::ArchiveIntegrity("archive encode"));
            return Err(
                if error == StoreError::ArchiveByteBudget
                    && maximum == remaining.saturating_sub(HEADER)
                {
                    StoreError::ArchiveQuota(
                        if self.config.game_bytes.saturating_sub(game)
                            <= self.config.global_bytes.saturating_sub(total)
                        {
                            "game archive"
                        } else {
                            "global archive"
                        },
                    )
                } else if error == StoreError::ArchiveByteBudget
                    && maximum == self.config.record_bytes
                {
                    StoreError::ArchiveQuota("archive segment bytes")
                } else {
                    error
                },
            );
        }
        let bytes = writer.bytes;
        let checksum: [u8; 32] = Sha256::digest(&bytes).into();
        let generation = self.next_generation;
        // Failed verification can leave a committed file. Reserve its number
        // before writing so a retry cannot overwrite retained evidence.
        self.next_generation = next;
        let temporary = self.directory.join(format!("seg-{generation:020}.pending"));
        let final_path = self.path(generation);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| io("archive temporary create", e))?;
        let result = (|| {
            budget.debit(HEADER + bytes.len() as u64)?;
            file.write_all(MAGIC)
                .and_then(|_| file.write_all(&generation.to_le_bytes()))
                .and_then(|_| file.write_all(&(bytes.len() as u64).to_le_bytes()))
                .and_then(|_| file.write_all(&checksum))
                .and_then(|_| file.write_all(&bytes))
                .and_then(|_| file.sync_all())
                .map_err(|e| io("archive write and sync", e))?;
            budget.check()?;
            drop(file);
            fs::rename(&temporary, &final_path).map_err(|e| io("archive atomic commit", e))?;
            #[cfg(unix)]
            File::open(&self.directory)
                .and_then(|f| f.sync_all())
                .map_err(|e| io("archive directory sync", e))?;
            self.read(generation, Some(checksum), budget)?;
            Ok(())
        })();
        if result.is_err() && temporary.exists() {
            fs::remove_file(&temporary).map_err(|e| io("archive temporary cleanup", e))?;
        }
        result?;
        drop(lock);
        Ok(ArchiveReceipt {
            owner: Arc::downgrade(&self.identity),
            generation,
            checksum,
            before,
            after: before,
            committed_bytes: HEADER + bytes.len() as u64,
            io_bytes: 2 * (HEADER + bytes.len() as u64),
        })
    }
    fn read(
        &self,
        generation: u64,
        expected: Option<[u8; 32]>,
        budget: &mut ArchiveIoBudget,
    ) -> Result<(Bundle, [u8; 32]), StoreError> {
        budget.check()?;
        let path = self.path(generation);
        no_link(&path)?;
        let mut file = File::open(&path).map_err(|e| io("archive open", e))?;
        let size = file
            .metadata()
            .map_err(|e| io("archive file size", e))?
            .len();
        if size < HEADER || size > self.config.record_bytes.saturating_add(HEADER) {
            return Err(StoreError::ArchiveIntegrity("archive file bounds"));
        }
        budget.debit(size)?;
        let mut header = [0u8; HEADER as usize];
        file.read_exact(&mut header)
            .map_err(|e| io("archive header read", e))?;
        let stored_generation = u64::from_le_bytes(header[8..16].try_into().expect("fixed header"));
        let length = u64::from_le_bytes(header[16..24].try_into().expect("fixed header"));
        let checksum: [u8; 32] = header[24..56].try_into().expect("fixed header");
        if &header[..8] != MAGIC
            || generation != stored_generation
            || length != size - HEADER
            || expected.is_some_and(|expected| expected != checksum)
        {
            return Err(StoreError::ArchiveIntegrity(
                "archive header or handle checksum",
            ));
        }
        let count = usize::try_from(length).map_err(|_| StoreError::ArchiveByteBudget)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(count)
            .map_err(|_| StoreError::Capacity("archive load buffer"))?;
        let mut digest = Sha256::new();
        let mut scratch = [0u8; 8192];
        while bytes.len() < count {
            budget.check()?;
            let take = (count - bytes.len()).min(scratch.len());
            file.read_exact(&mut scratch[..take])
                .map_err(|e| io("archive payload read", e))?;
            digest.update(&scratch[..take]);
            bytes.extend_from_slice(&scratch[..take]);
        }
        if <[u8; 32]>::from(digest.finalize()) != checksum {
            return Err(StoreError::ArchiveIntegrity("archive checksum"));
        }
        budget.check()?;
        let bundle: Bundle = serde_json::from_slice(&bytes)
            .map_err(|_| StoreError::ArchiveIntegrity("archive decode"))?;
        if bundle.version != 1
            || bundle.generation != generation
            || bundle.rules_version != rz_position::RULES_VERSION
            || bundle.variant != rz_position::RULES_VARIANT
        {
            return Err(StoreError::ArchiveIntegrity("archive schema or generation"));
        }
        Ok((bundle, checksum))
    }
}
struct QuotaLock(PathBuf);
impl QuotaLock {
    fn take(root: &Path) -> Result<Self, StoreError> {
        let path = root.join(".quota.lock");
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| io("archive quota lock", e))?;
        Ok(Self(path))
    }
}
impl Drop for QuotaLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
struct BoundedBuffer {
    bytes: Vec<u8>,
    maximum: u64,
    deadline: Instant,
    error: Option<StoreError>,
}
impl Write for BoundedBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let error = if Instant::now() >= self.deadline {
            Some(StoreError::ArchiveDeadline)
        } else if (self.bytes.len() as u64)
            .checked_add(bytes.len() as u64)
            .is_none_or(|n| n > self.maximum)
        {
            Some(StoreError::ArchiveByteBudget)
        } else {
            let needed = self.bytes.len() + bytes.len();
            let target = needed.max(
                self.bytes
                    .capacity()
                    .saturating_mul(2)
                    .max(8192)
                    .min(self.maximum as usize),
            );
            if needed > self.bytes.capacity()
                && self
                    .bytes
                    .try_reserve_exact(target - self.bytes.len())
                    .is_err()
            {
                Some(StoreError::Capacity("archive encode buffer"))
            } else {
                None
            }
        };
        if let Some(error) = error {
            self.error = Some(error);
            return Err(std::io::Error::other("bounded archive encoding"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl PalsStores {
    pub fn enable_archive(&mut self, config: ArchiveConfig) -> Result<(), StoreError> {
        if self.archive.is_some() {
            return Err(StoreError::InvalidConditions("archive already enabled"));
        }
        let mut manager = ArchiveManager::open(config)?;
        manager.state_owner = Arc::downgrade(&self.states.identity);
        manager.line_owner = Arc::downgrade(&self.lines.identity);
        self.archive = Some(manager);
        Ok(())
    }
    pub fn archive_enabled(&self) -> bool {
        self.archive.is_some()
    }
    pub(super) fn automatic_archive_enabled(&self) -> bool {
        self.archive
            .as_ref()
            .is_some_and(|manager| manager.auto_budget.is_some())
    }
    /// One search owns this explicit I/O allowance. Automatic pressure handling
    /// debits it, so multiple allocations cannot each spend the full allowance.
    pub fn set_archive_io_budget(&mut self, budget: ArchiveIoBudget) -> Result<(), StoreError> {
        budget.check()?;
        self.archive
            .as_mut()
            .ok_or(StoreError::ArchiveDisabled)?
            .auto_budget = Some(budget);
        Ok(())
    }
    pub fn set_archive_pins(&mut self, pins: StorePins) -> Result<(), StoreError> {
        self.validate_pins(&pins)?;
        self.archive
            .as_mut()
            .ok_or(StoreError::ArchiveDisabled)?
            .pins = pins;
        Ok(())
    }
    pub fn hot_stats(&self) -> ArchiveStats {
        ArchiveStats::hot(self)
    }
    pub(super) fn automatic_reclaim(
        &mut self,
        extra: StorePins,
        force: bool,
    ) -> Result<bool, StoreError> {
        let Some(manager) = self.archive.as_ref() else {
            return Ok(false);
        };
        let Some(mut budget) = manager.auto_budget else {
            return Ok(false);
        };
        let result = self.reclaim_for_allocation(extra, force, &mut budget);
        self.archive.as_mut().expect("archive restored").auto_budget = Some(budget);
        result
    }
    fn reclaim_for_allocation(
        &mut self,
        extra: StorePins,
        force: bool,
        budget: &mut ArchiveIoBudget,
    ) -> Result<bool, StoreError> {
        if !force && !self.pressure() {
            return Ok(false);
        }
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        let old_pins = manager.pins.clone();
        let mut combined = old_pins.clone();
        combined.extend(&extra);
        self.validate_pins(&combined)?;
        self.archive.as_mut().expect("archive checked").pins = combined;
        let result = self.archive_inactive_with_budget(budget);
        let manager = self.archive.as_mut().expect("archive restored");
        manager.pins = old_pins;
        result.map(|receipt| receipt.is_some())
    }
    /// A foreign physical request must remain unique across hot and cold facts.
    /// The explicit allowance covers scans and any pressure/single retry archive
    /// writes, including failed I/O. Hot-only legacy append never scans disk.
    pub fn append_observation_checked_with_archive_budget(
        &mut self,
        observation: Observation,
        budget: &mut ArchiveIoBudget,
    ) -> Result<ObservationId, StoreError> {
        budget.check()?;
        self.validate_observation_handles(&observation)?;
        if !self.archive_enabled() {
            return self.append_observation_once(observation);
        }
        if observation.external_report.is_some() {
            let manager = self.archive.as_ref().expect("archive checked");
            manager.verify_hot_owner(self)?;
            for entry in fs::read_dir(&manager.directory)
                .map_err(|e| io("foreign archive uniqueness directory", e))?
            {
                budget.check()?;
                let entry = entry.map_err(|e| io("foreign archive uniqueness entry", e))?;
                let name = entry.file_name();
                let Some(number) = name
                    .to_str()
                    .and_then(|name| name.strip_prefix("seg-"))
                    .and_then(|name| name.strip_suffix(".arc"))
                else {
                    continue;
                };
                let generation = number
                    .parse::<u64>()
                    .map_err(|_| StoreError::ArchiveIntegrity("archive segment name"))?;
                if generation >= manager.next_generation {
                    continue;
                }
                let (bundle, _) = manager.read(generation, None, budget)?;
                self.check_bundle_counts(&bundle)?;
                if bundle
                    .observations
                    .iter()
                    .any(|(_, cold)| cold.conflicts_external_request(&observation))
                {
                    return Err(StoreError::InvalidEvidence(
                        "foreign physical request belongs to another archived task",
                    ));
                }
            }
        }
        let mut pins = StorePins::default();
        pins.states.insert(observation.state);
        pins.lines.extend(observation.line);
        pins.lines.extend(observation.cpu_pv);
        pins.observations.extend(observation.supersedes);
        pins.executions.extend(observation.execution);
        pins.observations
            .extend(observation.score.model_dependencies().into_iter().flatten());
        self.reclaim_for_allocation(pins.clone(), false, budget)?;
        let result = self.append_observation_once(observation.clone());
        if matches!(result, Err(StoreError::Capacity(_)))
            && self.reclaim_for_allocation(pins, true, budget)?
        {
            self.append_observation_once(observation)
        } else {
            result
        }
    }
    fn validate_pins(&self, pins: &StorePins) -> Result<(), StoreError> {
        if pins.states.len() > self.limits.states
            || pins.lines.len() > self.limits.line_chunks
            || pins.situations.len() > self.limits.situations
            || pins.observations.len() > self.limits.observations
            || pins.executions.len() > self.limits.executions
        {
            return Err(StoreError::PinSaturated("pin index"));
        }
        for id in &pins.states {
            self.states.get(*id)?;
        }
        for id in &pins.lines {
            self.lines.get(*id)?;
        }
        for id in &pins.observations {
            self.observations.get(*id)?;
        }
        for id in &pins.situations {
            self.situations.get(*id)?;
        }
        for id in &pins.executions {
            self.tasks.get(*id)?;
        }
        Ok(())
    }
    fn expand_pins(&self, mut pins: StorePins) -> Result<StorePins, StoreError> {
        loop {
            let before = pins.count();
            for id in pins.situations.clone() {
                let s = self.situations.get(id)?;
                pins.states.insert(s.state);
                pins.lines.insert(s.focus);
                for (line, conclusion) in s.conclusions.iter() {
                    pins.lines.insert(*line);
                    if let Some(o) = conclusion.evidence {
                        pins.observations.insert(o);
                    }
                    if let ContinuationStatus::RepairedBy(line) = conclusion.status {
                        pins.lines.insert(line);
                    }
                }
            }
            for id in pins.lines.clone() {
                let line = self.lines.get(id)?;
                pins.states.insert(line.start_state);
                if let Some(parent) = line.parent {
                    pins.lines.insert(parent);
                }
            }
            for id in pins.observations.clone() {
                let o = self.observations.get(id)?;
                pins.states.insert(o.state);
                pins.lines.extend(o.line);
                pins.lines.extend(o.cpu_pv);
                pins.observations.extend(o.supersedes);
                pins.observations
                    .extend(o.score.model_dependencies().into_iter().flatten());
                pins.executions.extend(o.execution);
            }
            for id in pins.executions.clone() {
                let task = self.tasks.get(id)?;
                pins.states.insert(task.key.state);
                pins.lines.extend(task.key.line);
                pins.executions.extend(task.resumed_from);
                match task.status {
                    TaskStatus::Completed(o) => {
                        pins.observations.insert(o);
                    }
                    TaskStatus::Paused { evidence, .. }
                    | TaskStatus::RetiredPaused { evidence, .. } => {
                        pins.observations.extend(evidence);
                    }
                    _ => {}
                }
            }
            for (observation, situations) in &self.dependencies.dependents {
                if situations.iter().any(|s| pins.situations.contains(s)) {
                    pins.observations.insert(*observation);
                }
            }
            if pins.count() == before {
                break;
            }
        }
        self.validate_pins(&pins)?;
        Ok(pins)
    }
    fn protected_pins(&self, manager: &ArchiveManager) -> Result<StorePins, StoreError> {
        let mut pins = manager.pins.clone();
        pins.situations.extend(self.root);
        for loaded in manager.load_pins.values() {
            pins.extend(loaded);
        }
        for (id, task) in self.tasks.tasks.entries() {
            // CancellationRequested is still physical work, never quiescent.
            let active = matches!(
                task.status,
                TaskStatus::InFlight
                    | TaskStatus::CancellationRequested
                    | TaskStatus::Paused { .. }
            );
            let live_consumer = task.consumers.iter().any(|c| {
                !c.cancelled
                    && !c.consumed
                    && c.consumer.generation == self.generation
                    && self.situations.get(c.consumer.situation).is_ok()
            });
            if active || live_consumer {
                pins.executions.insert(ExecutionId(id));
                for c in &task.consumers {
                    if self.situations.get(c.consumer.situation).is_ok() {
                        pins.situations.insert(c.consumer.situation);
                    }
                }
            }
        }
        self.expand_pins(pins)
    }
    fn bundle_for(&self, selection: &StorePins, generation: u64) -> Result<Bundle, StoreError> {
        let copied = self.expand_pins(selection.clone())?;
        let mut bundle = Bundle::new(generation);
        for id in &copied.states {
            bundle
                .states
                .push(StateWire::encode(*id, self.states.get(*id)?)?);
        }
        for id in &copied.lines {
            bundle
                .lines
                .push(LineWire::encode(*id, self.lines.get(*id)?));
        }
        for id in &copied.observations {
            bundle
                .observations
                .push((*id, ObservationWire::encode(self.observations.get(*id)?)?));
        }
        for id in &copied.situations {
            bundle
                .situations
                .push((*id, self.situations.get(*id)?.clone()));
        }
        for id in &copied.executions {
            bundle.tasks.push((*id, self.tasks.get(*id)?.clone()));
        }
        for (observation, ids) in &self.dependencies.dependents {
            if copied.observations.contains(observation)
                || ids.iter().any(|id| copied.situations.contains(id))
            {
                bundle
                    .dependencies
                    .push((*observation, ids.iter().copied().collect()));
            }
        }
        bundle.checked_pvs = self
            .checked_cpu_pvs
            .keys()
            .filter(|line| copied.lines.contains(line))
            .copied()
            .collect();
        Ok(bundle)
    }
    fn pressure(&self) -> bool {
        let stats = self.hot_stats();
        let l = self.limits;
        [
            (stats.states, l.states),
            (stats.lines, l.line_chunks),
            (stats.situations, l.situations),
            (stats.observations, l.observations),
            (stats.dependencies, l.dependency_edges),
            (stats.executions, l.executions),
            (stats.consumers, l.consumers),
            (stats.history_bytes, l.retained_history_bytes),
        ]
        .into_iter()
        .any(|(used, maximum)| maximum != 0 && (used as u128) * 5 >= (maximum as u128) * 4)
    }
    pub fn archive_pressure(&self) -> bool {
        self.pressure()
    }
    pub fn reclaim_if_needed(
        &mut self,
        budget: ArchiveIoBudget,
    ) -> Result<Option<ArchiveReceipt>, StoreError> {
        if !self.pressure() {
            return Ok(None);
        }
        self.archive_inactive(budget)
    }
    pub fn archive_inactive(
        &mut self,
        mut budget: ArchiveIoBudget,
    ) -> Result<Option<ArchiveReceipt>, StoreError> {
        self.archive_inactive_with_budget(&mut budget)
    }
    pub fn archive_inactive_with_budget(
        &mut self,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<ArchiveReceipt>, StoreError> {
        budget.check()?;
        let mut manager = self.archive.take().ok_or(StoreError::ArchiveDisabled)?;
        let result = (|| {
            manager.verify_hot_owner(self)?;
            let pinned = self.protected_pins(&manager)?;
            let selection = StorePins {
                states: self
                    .states
                    .snapshots
                    .entries()
                    .map(|(id, _)| StateId(id))
                    .filter(|id| !pinned.states.contains(id))
                    .collect(),
                lines: self
                    .lines
                    .chunks
                    .entries()
                    .map(|(id, _)| LineId(id))
                    .filter(|id| !pinned.lines.contains(id))
                    .collect(),
                observations: self
                    .observations
                    .observations
                    .entries()
                    .map(|(id, _)| ObservationId(id))
                    .filter(|id| !pinned.observations.contains(id))
                    .collect(),
                situations: self
                    .situations
                    .iter()
                    .map(|(id, _)| id)
                    .filter(|id| !pinned.situations.contains(id))
                    .collect(),
                executions: self
                    .tasks
                    .tasks
                    .entries()
                    .map(|(id, _)| ExecutionId(id))
                    .filter(|id| !pinned.executions.contains(id))
                    .collect(),
            };
            if selection.count() == 0 {
                return Err(StoreError::PinSaturated("all hot records pinned"));
            }
            for id in &selection.situations {
                id.generation
                    .checked_add(1)
                    .ok_or(StoreError::RevisionExhausted)?;
            }
            let bundle = self.bundle_for(&selection, manager.next_generation)?;
            let mut receipt = manager.commit(&bundle, budget, self.hot_stats())?;
            // Every operation below removes an already validated hot record;
            // no RAM is released until committed bytes were reread and checked.
            for id in &selection.states {
                if let Some(state) = self.states.snapshots.remove(id.0) {
                    self.states.history_bytes -=
                        state.retained_history_bytes().expect("admitted charge");
                }
            }
            self.states.index.retain(|_, ids| {
                ids.retain(|id| !selection.states.contains(id));
                !ids.is_empty()
            });
            for id in &selection.lines {
                self.lines.chunks.remove(id.0);
                self.checked_cpu_pvs.remove(id);
            }
            self.lines
                .roots
                .retain(|_, id| !selection.lines.contains(id));
            self.lines.index.retain(|(parent, _), id| {
                !selection.lines.contains(parent) && !selection.lines.contains(id)
            });
            for id in &selection.observations {
                self.observations.observations.remove(id.0);
            }
            for id in &selection.situations {
                self.situations.remove(*id)?;
            }
            self.dependencies.dependents.retain(|o, ids| {
                if selection.observations.contains(o) {
                    return false;
                }
                ids.retain(|s| !selection.situations.contains(s));
                !ids.is_empty()
            });
            self.dependencies.edges = self
                .dependencies
                .dependents
                .values()
                .map(BTreeSet::len)
                .sum();
            for id in &selection.executions {
                if let Some(task) = self.tasks.tasks.remove(id.0) {
                    for consumer in task.consumers {
                        self.tasks.consumers.remove(&consumer.consumer.id);
                    }
                }
            }
            self.tasks
                .latest
                .retain(|_, id| !selection.executions.contains(id));
            self.states.snapshots.compact();
            self.states.index.shrink_to_fit();
            self.lines.chunks.compact();
            self.observations.observations.compact();
            self.tasks.tasks.compact();
            receipt.after = self.hot_stats();
            Ok(Some(receipt))
        })();
        self.archive = Some(manager);
        result
    }
    /// Store-independent engine payloads commit through the same quotas and
    /// checksum barrier. The parent may trim only after receiving this receipt.
    pub fn archive_engine_records(
        &mut self,
        records: &[crate::pals::engine::RoleRecord],
        nodes: &[EngineArchiveNode],
        budget: ArchiveIoBudget,
    ) -> Result<EngineArchiveReceipt, StoreError> {
        self.archive_engine_records_with_pins(records, nodes, StorePins::default(), budget)
    }
    pub fn archive_engine_records_with_pins(
        &mut self,
        records: &[crate::pals::engine::RoleRecord],
        nodes: &[EngineArchiveNode],
        selection: StorePins,
        mut budget: ArchiveIoBudget,
    ) -> Result<EngineArchiveReceipt, StoreError> {
        self.archive_engine_records_with_pins_and_budget(records, nodes, selection, &mut budget)
    }
    pub fn archive_engine_records_with_pins_and_budget(
        &mut self,
        records: &[crate::pals::engine::RoleRecord],
        nodes: &[EngineArchiveNode],
        mut selection: StorePins,
        budget: &mut ArchiveIoBudget,
    ) -> Result<EngineArchiveReceipt, StoreError> {
        budget.check()?;
        if records.len() > self.limits.observations || nodes.len() > self.limits.situations {
            return Err(StoreError::Capacity("engine archive record count"));
        }
        for record in records {
            if record.line.len() > self.limits.line_plies {
                return Err(StoreError::Capacity("engine archive line"));
            }
            selection.states.insert(record.origin_state);
            selection.observations.extend(record.cpu_observation);
        }
        for node in nodes {
            if node.edges.len() > self.limits.root_moves_per_task || node.metadata.len() > 4096 {
                return Err(StoreError::Capacity("engine archive node"));
            }
            selection.states.insert(node.state);
            selection.situations.insert(node.situation);
            for (movement, state) in &node.edges {
                movement.unpack()?;
                selection.states.extend(state);
            }
        }
        let actual_root = self.root.and_then(|root| {
            self.situations.get(root).ok().and_then(|situation| {
                nodes
                    .iter()
                    .any(|node| node.state == situation.state && node.situation == root)
                    .then_some((root, situation.state, self.generation))
            })
        });
        let mut manager = self.archive.take().ok_or(StoreError::ArchiveDisabled)?;
        let result = (|| {
            manager.verify_hot_owner(self)?;
            let mut bundle = self.bundle_for(&selection, manager.next_generation)?;
            bundle.engine_records = records
                .iter()
                .map(RoleRecordWire::encode)
                .collect::<Result<_, _>>()?;
            bundle.engine_nodes = nodes.to_vec();
            let archive = manager.commit(&bundle, budget, self.hot_stats())?;
            Ok(EngineArchiveReceipt {
                archive,
                records: records.len(),
                nodes: nodes.len(),
                actual_root,
            })
        })();
        self.archive = Some(manager);
        result
    }
    pub fn lookup_cold_state(
        &self,
        snapshot: &PositionSnapshot,
        mut budget: ArchiveIoBudget,
    ) -> Result<Option<ColdHandle<StateId>>, StoreError> {
        self.lookup_cold_state_with_budget(snapshot, &mut budget)
    }
    pub fn lookup_cold_state_with_budget(
        &self,
        snapshot: &PositionSnapshot,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<ColdHandle<StateId>>, StoreError> {
        budget.check()?;
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        for entry in
            fs::read_dir(&manager.directory).map_err(|e| io("archive lookup directory", e))?
        {
            budget.check()?;
            let entry = entry.map_err(|e| io("archive lookup entry", e))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(number) = name
                .strip_prefix("seg-")
                .and_then(|n| n.strip_suffix(".arc"))
            else {
                continue;
            };
            let generation = number
                .parse::<u64>()
                .map_err(|_| StoreError::ArchiveIntegrity("archive segment name"))?;
            if generation >= manager.next_generation {
                continue;
            }
            let (bundle, checksum) = manager.read(generation, None, budget)?;
            self.check_bundle_counts(&bundle)?;
            for state in bundle.states {
                budget.check()?;
                if state.decode()?.same_state(snapshot) {
                    return Ok(Some(ColdHandle {
                        owner: Arc::downgrade(&manager.identity),
                        generation,
                        checksum,
                        record: state.id,
                        _kind: PhantomData,
                    }));
                }
            }
        }
        Ok(None)
    }
    pub fn lookup_cold_state_id(
        &self,
        state: StateId,
        mut budget: ArchiveIoBudget,
    ) -> Result<Option<ColdHandle<StateId>>, StoreError> {
        self.lookup_cold_state_id_with_budget(state, &mut budget)
    }
    pub fn lookup_cold_state_id_with_budget(
        &self,
        state: StateId,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<ColdHandle<StateId>>, StoreError> {
        budget.check()?;
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        if state.0 >= self.states.snapshots.next_id() {
            return Err(StoreError::InvalidHandle("cold state serial"));
        }
        for entry in
            fs::read_dir(&manager.directory).map_err(|e| io("archive state lookup directory", e))?
        {
            budget.check()?;
            let entry = entry.map_err(|e| io("archive state lookup entry", e))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(number) = name
                .strip_prefix("seg-")
                .and_then(|n| n.strip_suffix(".arc"))
            else {
                continue;
            };
            let generation = number
                .parse::<u64>()
                .map_err(|_| StoreError::ArchiveIntegrity("archive segment name"))?;
            if generation >= manager.next_generation {
                continue;
            }
            let (bundle, checksum) = manager.read(generation, None, budget)?;
            self.check_bundle_counts(&bundle)?;
            if bundle.states.iter().any(|wire| wire.id == state) {
                return Ok(Some(ColdHandle {
                    owner: Arc::downgrade(&manager.identity),
                    generation,
                    checksum,
                    record: state,
                    _kind: PhantomData,
                }));
            }
        }
        Ok(None)
    }
    /// Return the newest committed engine payload for an original state serial.
    /// No partial lookup success is returned if its scan exhausts the budget.
    pub fn lookup_engine_archive(
        &self,
        state: StateId,
        mut budget: ArchiveIoBudget,
    ) -> Result<Option<EngineArchiveReceipt>, StoreError> {
        self.lookup_engine_archive_with_budget(state, &mut budget)
    }
    pub fn lookup_engine_archive_with_budget(
        &self,
        state: StateId,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<EngineArchiveReceipt>, StoreError> {
        self.lookup_engine_archive_filtered_with_budget(state, budget, false)
    }
    /// A newer record-only segment cannot hide the newest complete Node DTO.
    pub fn lookup_engine_node_archive(
        &self,
        state: StateId,
        mut budget: ArchiveIoBudget,
    ) -> Result<Option<EngineArchiveReceipt>, StoreError> {
        self.lookup_engine_node_archive_with_budget(state, &mut budget)
    }
    pub fn lookup_engine_node_archive_with_budget(
        &self,
        state: StateId,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<EngineArchiveReceipt>, StoreError> {
        self.lookup_engine_archive_filtered_with_budget(state, budget, true)
    }
    fn lookup_engine_archive_filtered_with_budget(
        &self,
        state: StateId,
        budget: &mut ArchiveIoBudget,
        require_node: bool,
    ) -> Result<Option<EngineArchiveReceipt>, StoreError> {
        budget.check()?;
        let initial = budget.max_bytes;
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        let mut newest = None;
        for entry in fs::read_dir(&manager.directory)
            .map_err(|e| io("engine archive lookup directory", e))?
        {
            budget.check()?;
            let entry = entry.map_err(|e| io("engine archive lookup entry", e))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(number) = name
                .strip_prefix("seg-")
                .and_then(|n| n.strip_suffix(".arc"))
            else {
                continue;
            };
            let generation = number
                .parse::<u64>()
                .map_err(|_| StoreError::ArchiveIntegrity("archive segment name"))?;
            if generation >= manager.next_generation {
                continue;
            }
            let (bundle, checksum) = manager.read(generation, None, budget)?;
            self.check_bundle_counts(&bundle)?;
            if (bundle.engine_nodes.iter().any(|node| node.state == state)
                || !require_node
                    && bundle
                        .engine_records
                        .iter()
                        .any(|r| r.origin_state() == state))
                && newest
                    .as_ref()
                    .is_none_or(|r: &EngineArchiveReceipt| r.archive.generation < generation)
            {
                newest = Some(EngineArchiveReceipt {
                    records: bundle.engine_records.len(),
                    nodes: bundle.engine_nodes.len(),
                    actual_root: None,
                    archive: ArchiveReceipt {
                        owner: Arc::downgrade(&manager.identity),
                        generation,
                        checksum,
                        before: self.hot_stats(),
                        after: self.hot_stats(),
                        committed_bytes: fs::metadata(manager.path(generation))
                            .map_err(|e| io("engine archive committed size", e))?
                            .len(),
                        io_bytes: 0,
                    },
                });
            }
        }
        if let Some(receipt) = newest.as_mut() {
            receipt.archive.io_bytes = initial - budget.max_bytes;
        }
        Ok(newest)
    }
    fn check_bundle_counts(&self, bundle: &Bundle) -> Result<(), StoreError> {
        let l = self.limits;
        if bundle.states.len() > l.states
            || bundle.lines.len() > l.line_chunks
            || bundle.observations.len() > l.observations
            || bundle.situations.len() > l.situations
            || bundle.tasks.len() > l.executions
            || bundle.engine_records.len() > l.observations
            || bundle.engine_nodes.len() > l.situations
            || bundle.checked_pvs.len() > l.line_chunks
            || bundle
                .dependencies
                .iter()
                .map(|(_, ids)| ids.len())
                .sum::<usize>()
                > l.dependency_edges
        {
            return Err(StoreError::ArchiveIntegrity("archive index counts"));
        }
        if bundle
            .states
            .iter()
            .map(|s| s.id)
            .collect::<BTreeSet<_>>()
            .len()
            != bundle.states.len()
            || bundle
                .lines
                .iter()
                .map(|l| l.id)
                .collect::<BTreeSet<_>>()
                .len()
                != bundle.lines.len()
            || bundle
                .observations
                .iter()
                .map(|(id, _)| *id)
                .collect::<BTreeSet<_>>()
                .len()
                != bundle.observations.len()
            || bundle
                .tasks
                .iter()
                .map(|(id, _)| *id)
                .collect::<BTreeSet<_>>()
                .len()
                != bundle.tasks.len()
            || bundle
                .situations
                .iter()
                .map(|(id, _)| *id)
                .collect::<BTreeSet<_>>()
                .len()
                != bundle.situations.len()
        {
            return Err(StoreError::ArchiveIntegrity("duplicate archive serial"));
        }
        Ok(())
    }
    pub fn release_loaded(&mut self, pin: ArchiveLoadPin) -> Result<(), StoreError> {
        let manager = self.archive.as_mut().ok_or(StoreError::ArchiveDisabled)?;
        if !pin.owner.ptr_eq(&Arc::downgrade(&manager.identity))
            || manager.load_pins.remove(&pin.serial).is_none()
        {
            return Err(StoreError::InvalidHandle("archive load pin"));
        }
        Ok(())
    }
    /// The engine first commits the old root topology. This attested handoff
    /// expires only logical root acceptance, retaining physical task ownership.
    pub fn retire_actual_root_for_archive(
        &mut self,
        receipt: &EngineArchiveReceipt,
    ) -> Result<Option<SituationId>, StoreError> {
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        manager.verify_owner(&receipt.archive.state_handle(StateId(0)))?;
        let Some(root) = self.root else {
            return Ok(None);
        };
        let state = self.situations.get(root)?.state;
        if receipt.actual_root != Some((root, state, self.generation)) {
            return Err(StoreError::InvalidConditions(
                "root retirement receipt does not attest actual root",
            ));
        }
        let next = self
            .generation
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        self.generation = next;
        self.root = None;
        Ok(Some(root))
    }

    /// Restore only the requested record's exact dependency closure. Hot reads
    /// never perform I/O implicitly. The returned pin is explicitly releasable.
    pub fn pin_load<T: ArchiveRecordKind>(
        &mut self,
        handle: &ColdHandle<T>,
        mut budget: ArchiveIoBudget,
    ) -> Result<LoadedArchive, StoreError> {
        budget.check()?;
        let mut manager = self.archive.take().ok_or(StoreError::ArchiveDisabled)?;
        let result = (|| {
            manager.verify_owner(handle)?;
            let initial = budget.max_bytes;
            let (bundle, _) =
                manager.read(handle.generation, Some(handle.checksum), &mut budget)?;
            let mut pins = StorePins::default();
            handle.record.seed(&mut pins);
            self.restore_bundle(
                &mut manager,
                bundle,
                pins,
                false,
                budget,
                initial - budget.max_bytes,
            )
        })();
        self.archive = Some(manager);
        result
    }
    pub fn load_engine_archive(
        &mut self,
        receipt: &EngineArchiveReceipt,
        mut budget: ArchiveIoBudget,
    ) -> Result<LoadedArchive, StoreError> {
        budget.check()?;
        let mut manager = self.archive.take().ok_or(StoreError::ArchiveDisabled)?;
        let result = (|| {
            let handle = receipt.archive.state_handle(StateId(0));
            manager.verify_owner(&handle)?;
            let initial = budget.max_bytes;
            let (bundle, _) =
                manager.read(handle.generation, Some(handle.checksum), &mut budget)?;
            let mut pins = StorePins::default();
            for record in &bundle.engine_records {
                record.add_pins(&mut pins);
            }
            for node in &bundle.engine_nodes {
                pins.states.insert(node.state);
                pins.situations.insert(node.situation);
                for (_, state) in &node.edges {
                    pins.states.extend(state);
                }
            }
            // Opaque Node metadata can refer to raw CPU/model evidence even
            // when no equivalent Situation dependency edge was installed.
            for (id, observation) in &bundle.observations {
                if bundle
                    .engine_nodes
                    .iter()
                    .any(|node| node.state == observation.state_id())
                {
                    pins.observations.insert(*id);
                }
            }
            self.restore_bundle(
                &mut manager,
                bundle,
                pins,
                true,
                budget,
                initial - budget.max_bytes,
            )
        })();
        self.archive = Some(manager);
        result
    }
    /// Inspect bounded immutable engine payloads without admitting hot records.
    pub fn read_engine_archive(
        &self,
        receipt: &EngineArchiveReceipt,
        mut budget: ArchiveIoBudget,
    ) -> Result<
        (
            Vec<crate::pals::engine::RoleRecord>,
            Vec<EngineArchiveNode>,
            u64,
        ),
        StoreError,
    > {
        budget.check()?;
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        let handle = receipt.archive.state_handle(StateId(0));
        manager.verify_owner(&handle)?;
        let initial = budget.max_bytes;
        let (bundle, _) = manager.read(handle.generation, Some(handle.checksum), &mut budget)?;
        self.check_bundle_counts(&bundle)?;
        let records = bundle
            .engine_records
            .into_iter()
            .map(RoleRecordWire::decode)
            .collect::<Result<Vec<_>, _>>()?;
        for record in &records {
            if record.line.len() > self.limits.line_plies {
                return Err(StoreError::ArchiveIntegrity("engine line bounds"));
            }
        }
        for node in &bundle.engine_nodes {
            if node.metadata.len() > 4096 || node.edges.len() > self.limits.root_moves_per_task {
                return Err(StoreError::ArchiveIntegrity("engine node bounds"));
            }
            for (movement, _) in &node.edges {
                movement.unpack()?;
            }
        }
        budget.check()?;
        Ok((records, bundle.engine_nodes, initial - budget.max_bytes))
    }
    /// Load the selected archived node's reachable continuation subtree. The
    /// remaining historical node graph stays cold and consumes no hot slots.
    pub fn load_engine_archive_for_state(
        &mut self,
        receipt: &EngineArchiveReceipt,
        state: StateId,
        mut budget: ArchiveIoBudget,
    ) -> Result<LoadedArchive, StoreError> {
        budget.check()?;
        let mut manager = self.archive.take().ok_or(StoreError::ArchiveDisabled)?;
        let result = (|| {
            let handle = receipt.archive.state_handle(state);
            manager.verify_owner(&handle)?;
            let initial = budget.max_bytes;
            let (mut bundle, _) =
                manager.read(handle.generation, Some(handle.checksum), &mut budget)?;
            self.check_bundle_counts(&bundle)?;
            if !bundle.engine_nodes.iter().any(|n| n.state == state) {
                return Err(StoreError::InvalidHandle("cold engine root membership"));
            }
            let mut reachable = BTreeSet::from([state]);
            loop {
                budget.check()?;
                let before = reachable.len();
                for node in &bundle.engine_nodes {
                    if reachable.contains(&node.state) {
                        for (_, child) in &node.edges {
                            reachable.extend(child);
                        }
                    }
                }
                if reachable.len() > self.limits.states {
                    return Err(StoreError::PinSaturated("engine subtree states"));
                }
                if before == reachable.len() {
                    break;
                }
            }
            bundle
                .engine_nodes
                .retain(|node| reachable.contains(&node.state));
            bundle
                .engine_records
                .retain(|record| reachable.contains(&record.origin_state()));
            let mut pins = StorePins::default();
            pins.states = reachable;
            for node in &bundle.engine_nodes {
                pins.situations.insert(node.situation);
            }
            for record in &bundle.engine_records {
                record.add_pins(&mut pins);
            }
            for (id, observation) in &bundle.observations {
                if pins.states.contains(&observation.state_id()) {
                    pins.observations.insert(*id);
                }
            }
            self.restore_bundle(
                &mut manager,
                bundle,
                pins,
                true,
                budget,
                initial - budget.max_bytes,
            )
        })();
        self.archive = Some(manager);
        result
    }
    fn restore_bundle(
        &mut self,
        manager: &mut ArchiveManager,
        bundle: Bundle,
        mut pins: StorePins,
        include_engine: bool,
        budget: ArchiveIoBudget,
        io_bytes: u64,
    ) -> Result<LoadedArchive, StoreError> {
        budget.check()?;
        self.check_bundle_counts(&bundle)?;
        if !manager
            .state_owner
            .ptr_eq(&Arc::downgrade(&self.states.identity))
            || !manager
                .line_owner
                .ptr_eq(&Arc::downgrade(&self.lines.identity))
        {
            return Err(StoreError::InvalidHandle("archive hot-store owner changed"));
        }
        if manager.load_pins.len() >= manager.config.max_load_pins {
            return Err(StoreError::PinSaturated("archive load pins"));
        }
        let pin_serial = manager.next_pin;
        let next_pin = pin_serial
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        // Membership of the seed is checked even if its numeric serial currently
        // exists hot, so a typed handle cannot attest another record in a segment.
        for id in &pins.states {
            if !bundle.states.iter().any(|s| s.id == *id) {
                return Err(StoreError::InvalidHandle("cold state membership"));
            }
        }
        for id in &pins.lines {
            if !bundle.lines.iter().any(|l| l.id == *id) {
                return Err(StoreError::InvalidHandle("cold line membership"));
            }
        }
        for id in &pins.observations {
            if !bundle.observations.iter().any(|(o, _)| o == id) {
                return Err(StoreError::InvalidHandle("cold observation membership"));
            }
        }
        for id in &pins.executions {
            if !bundle.tasks.iter().any(|(e, _)| e == id) {
                return Err(StoreError::InvalidHandle("cold execution membership"));
            }
        }
        for id in &pins.situations {
            if !bundle.situations.iter().any(|(s, _)| s == id) {
                return Err(StoreError::InvalidHandle("cold situation membership"));
            }
        }
        loop {
            budget.check()?;
            let before = pins.count();
            for id in pins.lines.clone() {
                if let Some(line) = bundle.lines.iter().find(|line| line.id == id) {
                    pins.states.insert(line.state);
                    pins.lines.extend(line.parent);
                } else {
                    let line = self.lines.get(id)?;
                    pins.states.insert(line.start_state);
                    pins.lines.extend(line.parent);
                }
            }
            for id in pins.observations.clone() {
                if let Some((_, observation)) = bundle.observations.iter().find(|(o, _)| *o == id) {
                    observation.add_pins(&mut pins);
                } else {
                    let o = self.observations.get(id)?;
                    pins.states.insert(o.state);
                    pins.lines.extend(o.line);
                    pins.lines.extend(o.cpu_pv);
                    pins.observations.extend(o.supersedes);
                    pins.observations
                        .extend(o.score.model_dependencies().into_iter().flatten());
                    pins.executions.extend(o.execution);
                }
            }
            for id in pins.executions.clone() {
                let task = if let Some((_, task)) = bundle.tasks.iter().find(|(e, _)| *e == id) {
                    task
                } else {
                    self.tasks.get(id)?
                };
                pins.states.insert(task.key.state);
                pins.lines.extend(task.key.line);
                pins.executions.extend(task.resumed_from);
                match task.status {
                    TaskStatus::Completed(o) => {
                        pins.observations.insert(o);
                    }
                    TaskStatus::Paused { evidence, .. }
                    | TaskStatus::RetiredPaused { evidence, .. } => {
                        pins.observations.extend(evidence)
                    }
                    _ => {}
                }
            }
            for id in pins.situations.clone() {
                let situation =
                    if let Some((_, s)) = bundle.situations.iter().find(|(s, _)| *s == id) {
                        s
                    } else {
                        self.situations.get(id)?
                    };
                pins.states.insert(situation.state);
                pins.lines.insert(situation.focus);
                for (line, c) in situation.conclusions.iter() {
                    pins.lines.insert(*line);
                    pins.observations.extend(c.evidence);
                    if let ContinuationStatus::RepairedBy(line) = c.status {
                        pins.lines.insert(line);
                    }
                }
            }
            for (observation, situations) in &bundle.dependencies {
                if situations.iter().any(|id| pins.situations.contains(id)) {
                    pins.observations.insert(*observation);
                }
            }
            if pins.count() == before {
                break;
            }
        }
        for (count, maximum) in [
            (
                manager
                    .load_pins
                    .values()
                    .map(|p| p.states.len())
                    .sum::<usize>()
                    + pins.states.len(),
                self.limits.states,
            ),
            (
                manager
                    .load_pins
                    .values()
                    .map(|p| p.lines.len())
                    .sum::<usize>()
                    + pins.lines.len(),
                self.limits.line_chunks,
            ),
            (
                manager
                    .load_pins
                    .values()
                    .map(|p| p.observations.len())
                    .sum::<usize>()
                    + pins.observations.len(),
                self.limits.observations,
            ),
            (
                manager
                    .load_pins
                    .values()
                    .map(|p| p.situations.len())
                    .sum::<usize>()
                    + pins.situations.len(),
                self.limits.situations,
            ),
            (
                manager
                    .load_pins
                    .values()
                    .map(|p| p.executions.len())
                    .sum::<usize>()
                    + pins.executions.len(),
                self.limits.executions,
            ),
        ] {
            if count > maximum {
                return Err(StoreError::PinSaturated("load pin index"));
            }
        }
        let mut states = BTreeMap::new();
        let mut history = 0usize;
        for wire in &bundle.states {
            if pins.states.contains(&wire.id) {
                budget.check()?;
                if wire.id.0 >= self.states.snapshots.next_id() {
                    return Err(StoreError::InvalidHandle("cold state serial"));
                }
                if let Some(hot) = self.states.snapshots.get(wire.id.0) {
                    if StateWire::encode(wire.id, hot)? != *wire {
                        return Err(StoreError::ArchiveIntegrity("state serial collision"));
                    }
                } else {
                    history = history
                        .checked_add(wire.history_charge()?)
                        .ok_or(StoreError::Capacity("archive history"))?;
                    if self
                        .states
                        .history_bytes
                        .checked_add(history)
                        .is_none_or(|n| n > self.limits.retained_history_bytes)
                    {
                        return Err(StoreError::PinSaturated("loaded history"));
                    }
                    let snapshot = wire.decode()?;
                    if states.insert(wire.id, snapshot).is_some() {
                        return Err(StoreError::ArchiveIntegrity("duplicate state"));
                    }
                }
            }
        }
        for id in &pins.states {
            if !states.contains_key(id) {
                self.states.get(*id)?;
            }
        }
        let mut lines = BTreeMap::new();
        for wire in &bundle.lines {
            if pins.lines.contains(&wire.id) {
                if wire.id.0 >= self.lines.chunks.next_id() {
                    return Err(StoreError::InvalidHandle("cold line serial"));
                }
                let line = wire.decode(self.limits.line_plies)?;
                if let Some(hot) = self.lines.chunks.get(wire.id.0) {
                    if *hot != line {
                        return Err(StoreError::ArchiveIntegrity("line serial collision"));
                    }
                } else if lines.insert(wire.id, line).is_some() {
                    return Err(StoreError::ArchiveIntegrity("duplicate line"));
                }
            }
        }
        for id in &pins.lines {
            let line = lines
                .get(id)
                .or_else(|| self.lines.chunks.get(id.0))
                .ok_or(StoreError::InvalidHandle("cold line dependency"))?;
            if !pins.states.contains(&line.start_state) {
                return Err(StoreError::ArchiveIntegrity("line state closure"));
            }
            if let Some(parent) = line.parent {
                let p = lines
                    .get(&parent)
                    .or_else(|| self.lines.chunks.get(parent.0))
                    .ok_or(StoreError::InvalidHandle("cold line parent"))?;
                if p.start_state != line.start_state
                    || p.plies.checked_add(line.len) != Some(line.plies)
                {
                    return Err(StoreError::ArchiveIntegrity("line accumulated plies"));
                }
            }
        }
        let mut observations = BTreeMap::new();
        for (id, wire) in bundle.observations {
            if pins.observations.contains(&id) {
                budget.check()?;
                if id.0 >= self.observations.observations.next_id() {
                    return Err(StoreError::InvalidHandle("cold observation serial"));
                }
                let observation = wire.decode()?;
                if let Some(hot) = self.observations.observations.get(id.0) {
                    if *hot != observation {
                        return Err(StoreError::ArchiveIntegrity("observation serial collision"));
                    }
                } else if observations.insert(id, observation).is_some() {
                    return Err(StoreError::ArchiveIntegrity("duplicate observation"));
                }
            }
        }
        let mut tasks = BTreeMap::new();
        let mut consumers = 0usize;
        let mut task_validator = TaskTable::new(
            self.limits.executions,
            self.limits.consumers,
            self.limits.root_moves_per_task,
        );
        for (id, mut task) in bundle.tasks {
            if pins.executions.contains(&id) {
                budget.check()?;
                if id.0 >= self.tasks.tasks.next_id() {
                    return Err(StoreError::InvalidHandle("cold execution serial"));
                }
                if self.tasks.tasks.get(id.0).is_none() {
                    // Serde Vec growth is an allocation detail, not part of the
                    // retained exact task key. Admit its bounded ordered data.
                    task.key.root_moves = task.key.root_moves.into_boxed_slice().into_vec();
                    if matches!(
                        task.status,
                        TaskStatus::InFlight
                            | TaskStatus::CancellationRequested
                            | TaskStatus::Paused { .. }
                    ) {
                        return Err(StoreError::ArchiveIntegrity(
                            "active execution was archived",
                        ));
                    }
                    task_validator.request(
                        task.key.clone(),
                        TaskConsumer {
                            id: id.0 as u64,
                            situation: SituationId {
                                slot: 0,
                                generation: 0,
                            },
                            revision: 0,
                            generation: 0,
                            deadline_tick: u64::MAX,
                        },
                        0,
                    )?;
                    consumers = consumers
                        .checked_add(task.consumers.len())
                        .ok_or(StoreError::Capacity("archive consumers"))?;
                    if tasks.insert(id, task).is_some() {
                        return Err(StoreError::ArchiveIntegrity("duplicate execution"));
                    }
                }
            }
        }
        for id in &pins.observations {
            let o = observations
                .get(id)
                .or_else(|| self.observations.observations.get(id.0))
                .ok_or(StoreError::InvalidHandle("cold observation dependency"))?;
            if let RawScore::ConditionalWdl { perspective, .. }
            | RawScore::ConditionalRepairWdl { perspective, .. } = o.score
            {
                let snapshot = states
                    .get(&o.state)
                    .or_else(|| self.states.snapshots.get(o.state.0))
                    .ok_or(StoreError::InvalidHandle("cold conditional state"))?;
                if perspective != snapshot.side_to_move() {
                    return Err(StoreError::ArchiveIntegrity("conditional root perspective"));
                }
                let get = |id: ObservationId| {
                    observations
                        .get(&id)
                        .or_else(|| self.observations.observations.get(id.0))
                        .ok_or(StoreError::InvalidHandle("cold conditional raw evidence"))
                };
                validate_model_derivation(o, get)?;
            }
            for line in [o.line, o.cpu_pv].into_iter().flatten() {
                let l = lines
                    .get(&line)
                    .or_else(|| self.lines.chunks.get(line.0))
                    .ok_or(StoreError::InvalidHandle("cold observation line"))?;
                if l.start_state != o.state {
                    return Err(StoreError::ArchiveIntegrity("observation line state"));
                }
            }
            if let Some(previous) = o.supersedes {
                let p = observations
                    .get(&previous)
                    .or_else(|| self.observations.observations.get(previous.0))
                    .ok_or(StoreError::InvalidHandle("cold supersedes"))?;
                if p.state != o.state {
                    return Err(StoreError::ArchiveIntegrity("supersedes state"));
                }
            }
            if let Some(execution) = o.execution {
                let task = tasks
                    .get(&execution)
                    .or_else(|| self.tasks.tasks.get(execution.0))
                    .ok_or(StoreError::InvalidHandle("cold observation execution"))?;
                if task.key.state != o.state
                    || task.key.line != o.line
                    || task.key.value_identity != o.value_identity
                    || task.key.checker_identity != o.checker_identity
                    || task.key.cpu_condition != o.cpu_condition
                {
                    return Err(StoreError::ArchiveIntegrity("observation task namespace"));
                }
            }
        }
        let selected_situations = bundle
            .situations
            .into_iter()
            .filter(|(id, _)| pins.situations.contains(id))
            .collect::<Vec<_>>();
        let new_situations = selected_situations
            .iter()
            .filter(|(id, _)| self.situations.get(*id).is_err())
            .count();
        let edges = bundle
            .dependencies
            .into_iter()
            .filter(|(o, ids)| {
                pins.observations.contains(o) || ids.iter().any(|id| pins.situations.contains(id))
            })
            .collect::<Vec<_>>();
        let additional_edges = edges
            .iter()
            .map(|(o, ids)| {
                ids.iter()
                    .filter(|s| {
                        !self
                            .dependencies
                            .dependents
                            .get(o)
                            .is_some_and(|old| old.contains(s))
                    })
                    .count()
            })
            .sum::<usize>();
        let l = self.limits;
        for (used, added, maximum, name) in [
            (self.states.len(), states.len(), l.states, "loaded states"),
            (self.lines.len(), lines.len(), l.line_chunks, "loaded lines"),
            (
                self.observations.len(),
                observations.len(),
                l.observations,
                "loaded observations",
            ),
            (
                self.tasks.len(),
                tasks.len(),
                l.executions,
                "loaded executions",
            ),
            (
                self.tasks.consumer_count(),
                consumers,
                l.consumers,
                "loaded consumers",
            ),
            (
                self.situations.len(),
                new_situations,
                l.situations,
                "loaded situations",
            ),
            (
                self.dependencies.len(),
                additional_edges,
                l.dependency_edges,
                "loaded dependencies",
            ),
            (
                self.states.history_bytes,
                history,
                l.retained_history_bytes,
                "loaded history",
            ),
        ] {
            if used.checked_add(added).is_none_or(|n| n > maximum) {
                return Err(StoreError::PinSaturated(name));
            }
        }
        // Reserve fallible record/index allocations before publishing any data.
        self.states.snapshots.reserve(states.len())?;
        self.lines.chunks.reserve(lines.len())?;
        self.observations.observations.reserve(observations.len())?;
        self.tasks.tasks.reserve(tasks.len())?;
        self.states
            .index
            .try_reserve(states.len())
            .map_err(|_| StoreError::Capacity("load state index"))?;
        self.situations
            .slots
            .try_reserve_exact(
                new_situations.saturating_sub(self.situations.slots.len() - self.situations.live),
            )
            .map_err(|_| StoreError::Capacity("load situation allocation"))?;
        let mut seen_consumers = BTreeSet::new();
        for task in tasks.values() {
            for consumer in &task.consumers {
                if self.tasks.consumers.contains(&consumer.consumer.id)
                    || !seen_consumers.insert(consumer.consumer.id)
                {
                    return Err(StoreError::ArchiveIntegrity("consumer serial collision"));
                }
            }
        }
        let mut checked = Vec::new();
        for id in bundle.checked_pvs {
            if pins.lines.contains(&id) {
                budget.check()?;
                let line = lines
                    .get(&id)
                    .or_else(|| self.lines.chunks.get(id.0))
                    .ok_or(StoreError::InvalidHandle("cold checked PV"))?;
                let snapshot = states
                    .get(&line.start_state)
                    .or_else(|| self.states.snapshots.get(line.start_state.0))
                    .ok_or(StoreError::InvalidHandle("cold checked PV state"))?;
                let trace = snapshot
                    .uci_replay(rz_position::MAX_UCI_REPLAY_PLIES)
                    .map_err(|_| StoreError::ArchiveIntegrity("PV Rules trace"))?;
                let mut position = match trace.origin {
                    rz_position::HistoryOrigin::StartPosition => Position::startpos(),
                    rz_position::HistoryOrigin::Fen => Position::from_fen(&trace.start_fen)
                        .map_err(|_| StoreError::ArchiveIntegrity("PV root"))?,
                };
                for movement in trace.moves {
                    position
                        .make_move(movement)
                        .map_err(|_| StoreError::ArchiveIntegrity("PV Rules root history"))?;
                }
                let mut at = Some(id);
                let mut chain = Vec::new();
                while let Some(id) = at {
                    let line = lines
                        .get(&id)
                        .or_else(|| self.lines.chunks.get(id.0))
                        .ok_or(StoreError::InvalidHandle("PV line chain"))?;
                    chain.push(line);
                    at = line.parent;
                }
                for line in chain.into_iter().rev() {
                    for movement in line.chunk_moves() {
                        budget.check()?;
                        position
                            .make_move(movement.unpack()?)
                            .map_err(|_| StoreError::ArchiveIntegrity("PV Rules replay"))?;
                    }
                }
                checked.push(id);
            }
        }
        for o in observations.values() {
            if o.cpu_pv
                .is_some_and(|id| !checked.contains(&id) && !self.checked_cpu_pvs.contains_key(&id))
            {
                return Err(StoreError::ArchiveIntegrity("cold CPU PV attestation"));
            }
            if let RawScore::Terminal { winner } = o.score {
                let snapshot = states
                    .get(&o.state)
                    .or_else(|| self.states.snapshots.get(o.state.0))
                    .ok_or(StoreError::InvalidHandle("cold terminal state"))?;
                let position = Position::from_snapshot(
                    snapshot,
                    rz_position::PositionLimits {
                        max_history_positions: rz_position::MAX_UCI_REPLAY_PLIES + 1,
                        ..rz_position::PositionLimits::default()
                    },
                )
                .map_err(|_| StoreError::ArchiveIntegrity("terminal Rules owner"))?;
                let legal = position.ordered_legal_moves();
                if !matches!(position.play_status_from_view(&legal),Ok(PlayStatus::Terminal {winner:actual,..}) if actual==winner)
                {
                    return Err(StoreError::ArchiveIntegrity("terminal Rules fact"));
                }
            }
        }
        let engine_records = if include_engine {
            bundle
                .engine_records
                .into_iter()
                .map(RoleRecordWire::decode)
                .collect::<Result<_, _>>()?
        } else {
            vec![]
        };
        let engine_nodes = if include_engine {
            bundle.engine_nodes
        } else {
            vec![]
        };
        for record in &engine_records {
            if record.line.len() > l.line_plies {
                return Err(StoreError::ArchiveIntegrity("engine line bounds"));
            }
        }
        for node in &engine_nodes {
            if node.metadata.len() > 4096 || node.edges.len() > l.root_moves_per_task {
                return Err(StoreError::ArchiveIntegrity("engine node bounds"));
            }
        }
        budget.check()?;
        // All fallible I/O, Rules decoding, namespace checks and capacity checks
        // precede hot publication. Old situation generations stay invalid.
        for (id, state) in states {
            let key = state.repetition_identity();
            self.states.snapshots.restore(id.0, state)?;
            self.states.index.entry(key).or_default().push(id);
        }
        self.states.history_bytes += history;
        for (id, line) in lines {
            if let Some(parent) = line.parent {
                self.lines
                    .index
                    .insert((parent, line.chunk_moves().to_vec()), id);
            } else {
                self.lines.roots.entry(line.start_state).or_insert(id);
            }
            self.lines.chunks.restore(id.0, line)?;
        }
        for (id, o) in observations {
            self.observations.observations.restore(id.0, o)?;
        }
        let mut remap = Vec::new();
        let mut mapped_pins = pins.clone();
        mapped_pins.situations.clear();
        for (old, situation) in selected_situations {
            let current = if self.situations.get(old).is_ok() {
                old
            } else {
                let new = self.situations.insert(situation.state, situation.focus)?;
                *self.situations.get_mut(new)? = situation;
                new
            };
            remap.push((old, current));
            mapped_pins.situations.insert(current);
        }
        for (id, mut task) in tasks {
            for c in &mut task.consumers {
                if let Some((_, new)) = remap.iter().find(|(old, _)| *old == c.consumer.situation) {
                    c.consumer.situation = *new;
                }
                self.tasks.consumers.insert(c.consumer.id);
            }
            self.tasks
                .latest
                .entry(task.key.clone())
                .and_modify(|old| {
                    if id.0 > old.0 {
                        *old = id
                    }
                })
                .or_insert(id);
            self.tasks.tasks.restore(id.0, task)?;
        }
        for (observation, ids) in edges {
            let entry = self.dependencies.dependents.entry(observation).or_default();
            for old in ids {
                let current = remap
                    .iter()
                    .find(|(old_id, _)| *old_id == old)
                    .map_or(old, |(_, new)| *new);
                entry.insert(current);
            }
        }
        self.dependencies.edges = self
            .dependencies
            .dependents
            .values()
            .map(BTreeSet::len)
            .sum();
        for line in checked {
            self.checked_cpu_pvs.insert(
                line,
                CheckedCpuPv {
                    state_store: Arc::downgrade(&self.states.identity),
                    line_pool: Arc::downgrade(&self.lines.identity),
                },
            );
        }
        manager.load_pins.insert(pin_serial, mapped_pins);
        manager.next_pin = next_pin;
        Ok(LoadedArchive {
            pin: ArchiveLoadPin {
                owner: Arc::downgrade(&manager.identity),
                serial: pin_serial,
            },
            situations: remap,
            engine_records,
            engine_nodes,
            io_bytes,
        })
    }
}

#[cfg(test)]
mod tests;
