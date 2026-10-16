//! 원 native owner의 submitted callback과 manager 수집기를 연결한다.
//! 큐에는 검증한 입력만 이동하며, 이 채널 자체는 NN 완료·수락 증거가 아니다.
use super::*;
use rz_search::pals::coordinator::RoleFailure;
use rz_uci::pals_native::ticketed::{
    NativeTicketQuestion, TicketExecutionEvidence, TicketPreparation,
    VirtualReviewCollectionObserver, VirtualReviewPhysicalInvocationEvidence,
    VirtualReviewPreparedTicket,
};
use std::cell::Cell;
use std::sync::{Arc, Mutex, mpsc};

type Registrations = Arc<Mutex<BTreeMap<RequestId, RoventInputCaptureReservation>>>;
pub struct RoventNativeCaptureRegistration {
    registered: Registrations,
}
pub struct RoventNativeInputObserver {
    registered: Registrations,
    sender: mpsc::SyncSender<RoventCapturedInput>,
}
pub struct RoventCapturedInputReceiver {
    receiver: mpsc::Receiver<RoventCapturedInput>,
    disconnected: Cell<bool>,
}
pub fn rovent_native_capture_channel() -> (
    RoventNativeCaptureRegistration,
    RoventNativeInputObserver,
    RoventCapturedInputReceiver,
) {
    let registered = Arc::new(Mutex::new(BTreeMap::new()));
    let (sender, receiver) = mpsc::sync_channel(16);
    (
        RoventNativeCaptureRegistration {
            registered: Arc::clone(&registered),
        },
        RoventNativeInputObserver { registered, sender },
        RoventCapturedInputReceiver {
            receiver,
            disconnected: Cell::new(false),
        },
    )
}
impl RoventNativeCaptureRegistration {
    /// 원 dispatch observer가 ticket 발급 후 execute 전에 호출한다.
    pub fn register(
        &self,
        ticket: &RoleTicketEnvelope,
        reservation: RoventInputCaptureReservation,
    ) -> Result<(), ArenaError> {
        let mut registered = self
            .registered
            .lock()
            .map_err(|_| invalid("Rovent capture registration poisoned"))?;
        check(
            registered.len() < 16 && !registered.contains_key(&ticket.id),
            "capture input registration duplicate/cap",
        )?;
        registered.insert(ticket.id, reservation);
        Ok(())
    }
    /// 아직 native owner가 읽지 않은 예약만 반환한다. 반환 sequence는
    /// producer.discard_reserved_input에 전달하고 물리 취소는 원 owner가 한다.
    pub fn cancel_unstarted(
        &self,
        id: RequestId,
    ) -> Result<Option<RoventInputCaptureReservation>, ArenaError> {
        Ok(self
            .registered
            .lock()
            .map_err(|_| invalid("Rovent capture registration poisoned"))?
            .remove(&id))
    }
}
impl RoventCapturedInputReceiver {
    pub fn input_owner_disconnected(&self) -> bool {
        self.disconnected.get()
    }
    pub fn take(&self) -> Result<Option<RoventCapturedInput>, ArenaError> {
        match self.receiver.try_recv() {
            Ok(input) => Ok(Some(input)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.disconnected.set(true);
                Ok(None)
            }
        }
    }
    /// 제출 입력은 같은 native owner가 Completion event를 발행하기 전에
    /// 전송한다. manager는 원 Completion을 처리하기 전에 이 큐를 비운다.
    pub fn admit_available(
        &self,
        producer: &mut RoventCorpusProducer,
    ) -> Result<usize, ArenaError> {
        let mut admitted = 0;
        while let Some(input) = self.take()? {
            producer.admit_captured_input(input)?;
            admitted += 1;
        }
        Ok(admitted)
    }
}
impl VirtualReviewCollectionObserver for RoventNativeInputObserver {
    fn submitted(
        &mut self,
        ticket: &RoleTicketEnvelope,
        preparation: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        prepared: &VirtualReviewPreparedTicket,
    ) -> Result<(), RoleFailure> {
        if prepared.representation != ticket.representation {
            return Err(RoleFailure::Backend(
                "Rovent prepared input representation differs from original ticket".into(),
            ));
        }
        let reservation = self
            .registered
            .lock()
            .map_err(|_| RoleFailure::Backend("Rovent capture registration poisoned".into()))?
            .remove(&ticket.id)
            .ok_or_else(|| {
                RoleFailure::Backend(
                    "Rovent native input lacks original manager reservation".into(),
                )
            })?;
        let captured = reservation
            .capture(
                *ticket,
                &preparation.view,
                &prepared.input,
                &prepared.identity,
            )
            .map_err(|e| RoleFailure::Backend(e.to_string()))?;
        self.sender.try_send(captured).map_err(|e| {
            RoleFailure::Backend(match e {
                mpsc::TrySendError::Full(_) => "Rovent captured input channel full".into(),
                mpsc::TrySendError::Disconnected(_) => {
                    "Rovent captured input owner disconnected".into()
                }
            })
        })
    }
    fn completed(
        &mut self,
        _: &RoleTicketEnvelope,
        _: PhysicalRoleCompletion,
        _: Option<&RoleOutput>,
        _: Option<TicketExecutionEvidence>,
        _: Option<VirtualReviewPhysicalInvocationEvidence>,
    ) -> Result<(), RoleFailure> {
        // 완료 권한은 manager가 받은 원 TicketOwnerRunWitness와 native event가
        // 발급한다. 이 callback의 공개 flag로 완료 권한을 재구성하지 않는다.
        Ok(())
    }
    fn retired(&mut self, _: &RoleTicketEnvelope, _: bool) -> Result<(), RoleFailure> {
        // 수락·퇴역은 원 registry의 AcceptedRoleStep을 따로 빌려 검증한다.
        Ok(())
    }
}
