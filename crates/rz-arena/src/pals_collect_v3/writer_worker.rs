//! Archive serialization and file I/O stay off the coordinator manager. This
//! bounded owner never runs search, a neural model, an optimizer or cloud API.
use super::*;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::Instant;

pub enum GenerationWriteCommandV3 {
    Record(Box<GenerationRecordV3>),
    Evidence(serde_json::Value),
    Trajectory(GenerationTrajectoryAuthorityV3),
    Finish {
        authority: GenerationProducerAuthorityV3,
        splits: BTreeMap<String, GenerationSplitV3>,
    },
    PreservePartial {
        authority: GenerationProducerAuthorityV3,
        splits: BTreeMap<String, GenerationSplitV3>,
        reason: String,
    },
}
pub enum GenerationWriterEventV3 {
    Ready,
    Failed(String),
    Finished {
        directory: PathBuf,
        manifest: GenerationChunkManifestV3,
    },
    PartialPreserved(GenerationPartialArchiveV3),
}
pub struct GenerationWriterWorkerV3 {
    commands: mpsc::SyncSender<GenerationWriteCommandV3>,
    events: mpsc::Receiver<GenerationWriterEventV3>,
    cancel: Arc<AtomicBool>,
    closed: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}
impl GenerationWriterWorkerV3 {
    pub fn spawn(
        directory: PathBuf,
        run_id: String,
        maximum_output_bytes: u64,
        deadline: Instant,
    ) -> Result<Self, ArenaError> {
        check(
            deadline > Instant::now(),
            "generation writer requires a live finite deadline",
        )?;
        let (commands, receiver) = mpsc::sync_channel(32);
        let (event_sender, events) = mpsc::sync_channel(2);
        let cancel = Arc::new(AtomicBool::new(false));
        let closed = Arc::new(AtomicBool::new(false));
        let ready = Arc::new(AtomicBool::new(false));
        let task_cancel = cancel.clone();
        let task_closed = closed.clone();
        let task_ready = ready.clone();
        let join = std::thread::Builder::new()
            .name("pals-generation-writer".into())
            .spawn(move || {
                let result = (|| -> Result<(), ArenaError> {
                    let mut writer =
                        GenerationChunkWriterV3::create(&directory, run_id, maximum_output_bytes)?;
                    task_ready.store(true, Ordering::Release);
                    event_sender
                        .send(GenerationWriterEventV3::Ready)
                        .map_err(|_| invalid("generation writer observer disconnected"))?;
                    loop {
                        check(
                            !task_cancel.load(Ordering::Acquire) && Instant::now() < deadline,
                            "generation writer canceled/expired; partial files preserved",
                        )?;
                        let wait = deadline
                            .saturating_duration_since(Instant::now())
                            .min(std::time::Duration::from_millis(25));
                        let command = match receiver.recv_timeout(wait) {
                            Ok(c) => c,
                            Err(mpsc::RecvTimeoutError::Timeout) => continue,
                            Err(mpsc::RecvTimeoutError::Disconnected) => {
                                return Err(invalid(
                                    "generation writer input ended without sealed publication",
                                ));
                            }
                        };
                        match command {
                            GenerationWriteCommandV3::PreservePartial {
                                authority,
                                splits,
                                reason,
                            } => {
                                writer.publish_authority(&authority, &splits)?;
                                let partial = writer.finish_partial(&reason)?;
                                check(
                                    !task_cancel.load(Ordering::Acquire)
                                        && Instant::now() <= deadline,
                                    "partial preservation outside archive deadline",
                                )?;
                                event_sender
                                    .send(GenerationWriterEventV3::PartialPreserved(partial))
                                    .map_err(|_| {
                                        invalid("partial archive observer disconnected")
                                    })?;
                                return Ok(());
                            }
                            GenerationWriteCommandV3::Record(record) => {
                                writer.append_record(&record)?
                            }
                            GenerationWriteCommandV3::Evidence(receipt) => {
                                writer.append_evidence(&receipt)?;
                            }
                            GenerationWriteCommandV3::Trajectory(authority) => {
                                writer.append_trajectory(&authority)?;
                            }
                            GenerationWriteCommandV3::Finish { authority, splits } => {
                                writer.publish_authority(&authority, &splits)?;
                                let manifest = writer.finish()?;
                                check(
                                    !task_cancel.load(Ordering::Acquire)
                                        && Instant::now() <= deadline,
                                    "publication finished outside its original writer deadline",
                                )?;
                                event_sender
                                    .send(GenerationWriterEventV3::Finished {
                                        directory: writer.directory().to_path_buf(),
                                        manifest,
                                    })
                                    .map_err(|_| {
                                        invalid(
                                            "generation final publication observer disconnected",
                                        )
                                    })?;
                                return Ok(());
                            }
                        }
                    }
                })();
                task_closed.store(true, Ordering::Release);
                if let Err(error) = result {
                    let _ =
                        event_sender.try_send(GenerationWriterEventV3::Failed(error.to_string()));
                }
            })
            .map_err(|e| ArenaError::Io(e.to_string()))?;
        Ok(Self {
            commands,
            events,
            cancel,
            closed,
            ready,
            join: Some(join),
        })
    }
    /// No manager serialization or archive write occurs here. On saturation the
    /// caller retains the owned command and stops admission rather than dropping
    /// an actual row or counting unpersisted evidence as published.
    #[allow(
        clippy::result_large_err,
        reason = "Saturated admission returns the original command inline without another allocation or losing unpublished evidence"
    )]
    pub fn try_submit(
        &self,
        command: GenerationWriteCommandV3,
    ) -> Result<(), (ArenaError, GenerationWriteCommandV3)> {
        if self.closed.load(Ordering::Acquire) || !self.ready.load(Ordering::Acquire) {
            return Err((invalid("generation writer is not ready/open"), command));
        }
        let finishing = matches!(
            &command,
            GenerationWriteCommandV3::Finish { .. }
                | GenerationWriteCommandV3::PreservePartial { .. }
        );
        match self.commands.try_send(command) {
            Ok(()) => {
                if finishing {
                    self.closed.store(true, Ordering::Release);
                }
                Ok(())
            }
            Err(mpsc::TrySendError::Full(c)) => Err((
                ArenaError::Budget("generation writer queue capacity".into()),
                c,
            )),
            Err(mpsc::TrySendError::Disconnected(c)) => {
                Err((invalid("generation writer owner disconnected"), c))
            }
        }
    }
    pub fn try_event(&self) -> Option<GenerationWriterEventV3> {
        self.events.try_recv().ok()
    }
    pub fn cancel(&self) {
        self.closed.store(true, Ordering::Release);
        self.cancel.store(true, Ordering::Release);
    }
    /// Join only an observed finished thread. A deadline or cancel request does
    /// not prove the archive owner stopped, and does not permit cleanup.
    pub fn try_join(&mut self) -> Result<bool, ArenaError> {
        if self.join.as_ref().is_none_or(|join| !join.is_finished()) {
            return Ok(self.join.is_none());
        }
        self.join
            .take()
            .unwrap()
            .join()
            .map_err(|_| invalid("generation writer owner panicked; partial files preserved"))?;
        Ok(true)
    }
}
impl Drop for GenerationWriterWorkerV3 {
    fn drop(&mut self) {
        self.cancel(); /* caller must observe terminal+join before cleanup */
    }
}
