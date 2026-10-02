# D02 내부 trace 계측

`profile::TraceCollector<RequestKey, ExecutionKey>`는 caller의 ID를 그대로 받는
수동 관측기입니다. scheduler·search·backend를 제어하거나 공통 계약 schema를
복제하지 않습니다. 관측 메서드의 오류는 trace 검증 실패이며 엔진의 실행 결과를
바꾸는 명령이 아닙니다. 동시 사용의 동기화는 caller가 담당합니다.

`try_new(ProfileConfig, domain)`는 active request/execution 수, 최근 퇴역 ID 수,
각 stage의 표본 수와 batch 표본 수를 유한한 설정으로 제한합니다. 모든 layout의
checked byte 계산·`isize::MAX` 한계를 첫 할당 전에 검사하고, `try_reserve_exact`로
할당 실패를 typed `CapacityError`로 전달합니다. 요청 경로에서 정렬·report 계산은
하지 않습니다. tracker 조회는 유한한 vector에서 수행하며 caller ID 비교·소멸 비용은
caller 타입에 달려 있습니다. ring, tracker, 퇴역 ID, report의 복사·map 메모리 예산은
inference buffer 예산과 별도로 관리합니다.

모든 `Timepoint {domain, elapsed}`는 같은 monotonic clock domain이어야 합니다.
점 관측과 interval 끝은 마지막 수락한 관측 이상이어야 하고, interval 시작은 끝보다
늦을 수 없습니다. 겹치는 phase의 시작이 이전 관측보다 이른 것은 허용합니다.
잘못된 domain·시점·중복 전이는 typed `TraceError`이며, 보고서의 오류 수에 남습니다.
계측이 유실되었으면 caller는 `note_event_loss`로 observation transport의 손실을
함께 기록합니다. 손실·시간 오류·미퇴역 tracker가 있는 trace를 완전한 종단 증거로
보고하지 않습니다.

caller는 CPU preparation 시작부터 `begin_request`로 span을 열고 queue·validation을
`record_stage`로 기록합니다. `finish_request`는 논리 결과만 기록하며 성공 validation을
소비로 취급하지 않습니다. 실제 소비 시 `consume_request`, backup·output의 explicit
span 기록, `close_request` 순으로 end-to-end를 마칩니다. canceled·expired·stale·failed는
성공 결과를 소비하지 않고 close할 수 있습니다.
request가 실제로 해당 execution의 결과를 받은 것인지와 search consume token이
유효한지는 caller가 검증합니다. 관측기는 엔진의 요청-실행 연결 계약을 소유하지 않습니다.

`begin_execution`의 시점은 backend physical phase의 시작입니다. 기본 mock 연결에서는
dispatch 종료 시점을 사용한 뒤 `record_dispatch(dispatch_start, dispatch_end)`로 앞선
제출 구간을 기록합니다. `finish_execution`은 backend 시작부터 물리 완료까지를
측정합니다. 취소된 request가 close된 뒤 늦은 execution 완료·회수도 정상 관측입니다.
backend label은 `CpuMock`, `Cpu`, `Gpu`로 명시하며 실제 provider hook이 없는
`Transfer`는 absent이고 0 시간으로 꾸미지 않습니다. 실제 GPU transfer만
`record_transfer`로 기록하며 backend physical과 겹칠 수 있습니다.

request와 물리 execution을 각각 한 번 셉니다. 성공 logical 결과, 실제 consumed request,
현재 execution에 귀속한 consumed item, cache-only 소비를 분리합니다. execution의
완료 직후 미소비 item은 아직 소비 가능하므로 `completed_unconsumed_items`이며,
caller가 모든 소비자가 종료했음을 확인해 `retire_execution`할 때만 최종
`unused_items_finalized`에 포함합니다. 현재 `consume_request`의 물리 실행 연결은
dedup 없는 runtime의 item 하나당 소비 한 번 계약입니다. 향후 하나의 physical item을
여러 subscriber가 소비하는 dedup은 explicit item index 계약을 추가한 뒤 계측합니다.
중복 관측 거부 수는 의미가 동일한 모델 입력의 재실행 수와 다릅니다.

중복 finish·consume·backup·execution 완료를 active tracker와 최근 퇴역 ID 범위에서
검사합니다. 이는 엔진의 exactly-once authority가 아닙니다. caller는 전체 trace에서 ID를
재사용하지 않아야 하며 퇴역 ID가 eviction되면 이전 ID의 중복을 더는 판별할 수
없습니다. 보고서에 eviction 수, duplicate check window와 caller uniqueness 전제를
노출합니다. 실제 runtime의 consume token 검사는 별도 책임입니다.

stage별 nearest-rank P50/P95/P99는 최근 보존한 표본의 값입니다. 각 stage·class·backend의
전체/보존/유실 수와 batch 분포의 같은 수를 함께 확인합니다. labels는 cold/warm,
parent-child, sibling, transposition, eviction, long trace로 고정합니다. 유실된 이전 표본이
있는 분포를 전체 실행 분포로 읽지 않습니다. 처리량은 가장 이른 수락 span 시작부터
마지막 수락 관측까지의 단일 wall span을 사용하며 겹친 stage 시간을 합산하지 않습니다.
execution 시작 뒤 앞선 dispatch span을 수락하면 wall 시작도 그 span의 시작으로 확장합니다.
거부된 span은 측정 구간을 바꾸지 않습니다. caller가
deterministic fixture clock을 주입했다면 이 값은 fixture clock의 처리량이며 실제 호스트
성능이 아닙니다. 실제 `Instant` 종단 측정과 GPU 성능은 별도 근거가 필요합니다.

host/device/pinned `MemoryReservations`는 예약 ledger의 현재값·domain별 peak입니다.
`observe_reservation_peak`는 현재값을 바꾸지 않고 보존된 high-water mark를 합칩니다.
device 예약 peak는 실제 VRAM 사용량이 아니며 event 사이 예약 변경은 caller의
ledger peak 없이는 복원할 수 없습니다. 모든 누적 counter는 포화 후 overflow flag를
남기며 overflow 결과를 정확한 집계로 해석하지 않습니다.
