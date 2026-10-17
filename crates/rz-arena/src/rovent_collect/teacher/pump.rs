//! native 퇴역 outbox와 독립 CPU_T session 사이의 유한 caller pump다.
//! manager 안에서 호출하지 않으며 실제 pipe 처리는 teacher worker가 담당한다.
use super::{session::*, *};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

struct WaitingRow {
    row: RoventRetiredRecord,
    issued: bool,
}
pub struct RoventCollectionTeacherFailure {
    pub input_sha256: String,
    pub detail: String,
    original_completion: Option<RoventCollectionTeacherCompleted>,
}
impl RoventCollectionTeacherFailure {
    pub fn completion(
        &self,
    ) -> Option<&rz_search::external_cpu::collection_teacher::TeacherCompletion> {
        self.original_completion.as_ref().map(|v| v.completion())
    }
}
pub struct RoventCollectionTeacherPump {
    owner: RoventCollectionOwner,
    session: RoventCollectionTeacherSession,
    waiting: BTreeMap<String, WaitingRow>,
    ready: VecDeque<RoventRetiredRecord>,
    failures: VecDeque<RoventCollectionTeacherFailure>,
    quarantined_rows: VecDeque<RoventRetiredRecord>,
    closed: bool,
}
impl RoventCollectionTeacherPump {
    pub fn new(owner: RoventCollectionOwner, session: RoventCollectionTeacherSession) -> Self {
        Self {
            owner,
            session,
            waiting: BTreeMap::new(),
            ready: VecDeque::new(),
            failures: VecDeque::new(),
            quarantined_rows: VecDeque::new(),
            closed: false,
        }
    }
    pub fn session(&self) -> &RoventCollectionTeacherSession {
        &self.session
    }
    pub fn session_mut(&mut self) -> &mut RoventCollectionTeacherSession {
        &mut self.session
    }
    /// native root 종료 전 별도 caller가 반복 호출한다. 한 poll은 최대 16개
    /// 원 row와 actor당 한 terminal만 처리하고 실제 teacher 완료를 기다리지 않는다.
    pub fn poll(
        &mut self,
        manifest: &RoventCorpusManifest,
        constraint: rz_search::external_cpu::ExternalGoConstraint,
        quantum: Duration,
    ) -> Result<usize, ArenaError> {
        check(
            self.failures.len() < 16 && self.ready.len() < 16 && self.quarantined_rows.len() < 16,
            "Collection caller outbox full; drain original evidence",
        )?;
        let mut changed = 0;
        while self.waiting.len() + self.ready.len() + self.quarantined_rows.len() < 16 {
            let Some(row) = self.owner.take_retired()? else {
                break;
            };
            let sha = row.input_sha256().to_owned();
            if self.waiting.contains_key(&sha) {
                self.quarantined_rows.push_back(row);
                self.closed = true;
                return Err(invalid(
                    "duplicate original teacher input occurrence retained",
                ));
            }
            if row.record.lineage.teacher_actor_id.is_none()
                || row.record.input.legal_moves.is_empty()
            {
                self.ready.push_back(row);
            } else {
                self.waiting.insert(sha, WaitingRow { row, issued: false });
            }
            changed += 1;
        }
        for actor in manifest.teachers.keys() {
            if self.failures.len() >= 16 || self.ready.len() >= 16 {
                break;
            }
            if let Some(completed) = self.session.take_completed(actor)? {
                let sha = completed.input_sha256().to_owned();
                let Some(mut waiting) = self.waiting.remove(&sha) else {
                    self.failures.push_back(RoventCollectionTeacherFailure {
                        input_sha256: sha,
                        detail: "orphan Collection completion".into(),
                        original_completion: Some(completed),
                    });
                    return Err(invalid("orphan Collection completion retained"));
                };
                let result = self.owner.with_producer(|producer| {
                    self.session
                        .attach_policy(producer, &mut waiting.row, &completed)
                });
                if let Err(error) = result {
                    self.failures.push_back(RoventCollectionTeacherFailure {
                        input_sha256: sha,
                        detail: error.to_string(),
                        original_completion: Some(completed),
                    });
                }
                // 실패/부분 의견은 policy=None이다. 원 row 및 원 완료는 각각
                // 보존하며 실패를 성공 label로 바꾸지 않는다.
                self.ready.push_back(waiting.row);
                changed += 1;
            }
        }
        if self.closed {
            return Ok(changed);
        }
        if !self.session.startup_ready(manifest)? {
            return Ok(changed);
        }
        for actor in manifest.teachers.keys() {
            if self.failures.len() >= 16 || self.ready.len() >= 16 {
                break;
            }
            let has_pending = self
                .waiting
                .values()
                .any(|v| v.issued && v.row.record.lineage.teacher_actor_id.as_ref() == Some(actor));
            if has_pending {
                continue;
            }
            let next = self
                .waiting
                .iter()
                .find(|(_, v)| {
                    !v.issued && v.row.record.lineage.teacher_actor_id.as_ref() == Some(actor)
                })
                .map(|(sha, _)| sha.clone());
            let Some(sha) = next else {
                continue;
            };
            let waiting = self
                .waiting
                .get_mut(&sha)
                .ok_or_else(|| invalid("Collection waiting row disappeared"))?;
            match self.owner.with_producer(|producer| {
                self.session
                    .submit_retired(producer, &waiting.row, constraint, quantum)
            }) {
                Ok(_) => {
                    waiting.issued = true;
                    changed += 1;
                }
                Err(error) => {
                    // 원 deadline이 종료한 입력을 새 go로 재태그하지 않는다.
                    self.failures.push_back(RoventCollectionTeacherFailure {
                        input_sha256: sha.clone(),
                        detail: error.to_string(),
                        original_completion: None,
                    });
                    let waiting = self
                        .waiting
                        .remove(&sha)
                        .ok_or_else(|| invalid("Collection failed row disappeared"))?;
                    self.ready.push_back(waiting.row);
                    changed += 1;
                }
            }
        }
        Ok(changed)
    }
    pub fn take_ready(&mut self) -> Option<RoventRetiredRecord> {
        self.ready.pop_front()
    }
    pub fn take_failure(&mut self) -> Option<RoventCollectionTeacherFailure> {
        self.failures.pop_front()
    }
    pub fn take_quarantined_row(&mut self) -> Option<RoventRetiredRecord> {
        self.quarantined_rows.pop_front()
    }
    pub fn request_shutdown(&mut self, until: Instant) {
        self.closed = true;
        self.session.request_shutdown(until);
    }
    pub fn poll_shutdown(&mut self) -> Vec<RoventCollectionTeacherShutdown> {
        self.session.poll_shutdown()
    }
    pub fn pending_rows(&self) -> usize {
        self.waiting.len()
    }
}
