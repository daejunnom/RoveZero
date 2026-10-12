# PALS Adaptive Coordinator 구현·인수

## 실행 범위와 기준

기준은 PR #23·#24가 병합된 `develop`의 `b3ba8dd2eed8d777a6b95f9135db067bd958a98a`다.
후속 작업은 `feature/pals-adaptive-coordinator`에서 진행한다. 이전 PR의 CPU·GPU·
수치·대국 증거를 새 coordinator의 실행 결과로 재명명하지 않는다.

현재 사용자 지시는 **플랜 전체의 구현을 진행하되 아직 학습을 실행하지 않는 것**이다.
실제 세대 수집·학습·checkpoint·export·평가 경로와 검사를 구현하지만 optimizer update,
128판 본수집과 G1 학습 후 32쌍 강도 평가는 실행되지 않았다. 학습 성공이나 세대 완료를
CPU/mock·schema·diagnostic 자료로 대신하지 않는다. 전체 구현 목표는 계속 진행 중이다.

첨부 Adaptive Coordinator 핸드오프와 PR23 변경 체크리스트는 설계 근거다.
이번 사용자 확정 플랜의 선택을 적용하며 기존 Rules·store·CPU·runtime·물리 수명을
재사용한다. 새 모델, context 선택은 A, slice·event order·adaptive는 S, 동일 의미의
복제 제거와 버퍼 재사용만 E로 구분한다. PALS 전체에 5% 성능 문턱을 적용하지 않는다.

## M0 공통 접점

`rz-contracts::pals_adaptive`는 검색/runtime 실행 envelope를 추가한다.
기존 `EvalRequest/EvalOutput`, `PALS_CONTRACT_REVISION=0.1`, V1~V4 bytes/digest는 유지한다.
새 coordinator revision은 `pals-adaptive/0.1`이다. 구체 Rules·pin·CPU stack·tensor는
해당 소유자가 관리하며 공통 타입에 역의존을 추가하지 않는다.

| 접점 | 필수 의미 |
|---|---|
| `CpuJobId` / `CpuJobEnvelope` | process/game/root authority·정확 상태·질문·root 제한·평가기·조건·총 native credit·원 hard deadline |
| `SliceGrant` | 별도 `ExecutionId`, 같은 Job·worker owner, 이번 native credit·soft lease·control sequence |
| `ReviewViewSeal` | exact root identity, authority·view sequence, context policy·내용 digest, 실제 토큰·record·anchor·L bucket |
| `RoleTicketEnvelope` | 제출 당시 immutable view·질문·representation·role·원 deadline; 후속 제출은 앞 ticket을 stale로 만들지 않음 |

ID 발급과 digest는 실제 소유자의 책임이다. root generation은 root 동안 고정하고
RequestId sequence만 제출마다 증가한다. view digest에는 선택 순서·anchor·mask·수열·
parent/supersedes 관계와 실제 읽는 query가 포함된다. 실제 기보는 Rules와 기록에
보존하고 review의 누적 토큰으로 넣지 않는다. 같은 가상 수열을 query와 record에서
중복해 읽으면 토큰 예산에도 각각 청구한다.

credit은 manager만 `spent + outstanding_reserved <= limit`로 관리한다.
progress 누적값을 다시 합산하지 않고 terminal/ack의 실제 delta를 한 번 청구한다.
불명확한 실패는 0 사용량이나 예약 환불이 아니다. 외부 reported nodes는 별도 원시
관측이며 native credit으로 환산하지 않는다. soft lease의 안전 경계 초과는 실제
시간으로 기록하고 원 hard deadline·취소 의미를 바꾸지 않는다.

## 초기 등록 조건

- Review: token 512, 비교용 128/256; selected record 64; 활성 bundle 256; anchor 4.
- 문맥 high-water 80%·목표 55%; 새 반례/복구는 실제 eligible view 소비 두 번까지 보호.
- L bucket 8/16/32/64/128/256 중 최소 크기; 하나의 dynamic-L graph와 session.
- CPU worker 기본 1; private paused job 1·8MiB; 고정 slice 4096 native node/8ms.
- 시작 전 Job 64·worker command 8·role ticket 16. 물리 GPU single-flight, 즉시 dispatch.
- 비용 EWMA alpha 1/8·class/stage 최근 128·p95. 완료 표본 8개 미만은 fixed.
- CPU quota 2·SharedControl·ORT intra/inter 1/1·spinning off. 실제 affinity와 topology는 readback.
- memory.high 6GiB/max 12GiB는 선언과 실제 적용을 구분한다. unknown을 0이나 성공으로 바꾸지 않는다.

worker 2/4는 별도 자원 프로필로 검증한다. paused token과 진행 중 frame은 worker 사이에
이동하지 않는다. 아직 시작하지 않은 Job만 재배정할 수 있다. queue full에서도 별도
atomic 제어 mailbox로 cancel/yield를 전달한다. terminal·오류·ack·물리 완료는 보존하고
progress만 coalesce한다. manager가 forward·archive I/O·외부 drain을 동기 실행하지 않는다.

## 21개 항목과 완료 증거

| 항목 | 소유·결과 | 필수 인수 |
|---|---|---|
| VR-01 | Search bounded bundle/index/view | 실제 기보 비유입·관계·digest·budget·2회 보호 |
| VR-02 | Rules exact rebase·store lease | anchor mismatch·suffix replay·archive pin 수명 |
| VR-03 | native selected-view 소비 | 제출 seal과 실제 tensor/소비 view 동일 |
| CY-01 | CPU PublishProgress/Yield/Cancel | partial bound 공개 금지·독립 Yield·terminal token 폐기 |
| CY-02 | Job/Slice/manager credit | 중복 청구·과다 예약·unknown 환불 없음 |
| CY-03 | private CPU worker | same-owner stack/TT/조건·depth/score/PV/nodes 대조 |
| CO-01 | continuation coordinator | 동기 순서와 fixed event-order 대조·Repair 한도 |
| CO-02 | native ticket별 async facade | pending 질문 독립·Fresh pair frozen view·늦은/중복 응답 |
| CO-03 | UCI 시작/stop/newgame/shutdown | admission close→revoke→cancel→physical drain→join→archive→receipt |
| AD-01 | class/stage 비용·backlog/hysteresis | censored·unknown tail·overhead 하한·starvation |
| AD-02 | topology/resource plan | caller/ORT/worker/외부 전체 계수·affinity readback |
| AD-03 | bounded channel·notification | full queue 제어·terminal 보존·lost wakeup |
| MI-01 | virtual_review_v1 V3 입력 | feature8·anchor·관계·mask·L bucket·구형 의미 보존 |
| MI-02 | Torch/ONNX/Rust reference/export | 모든 bucket/anchor/padding·네 승격·WDL 수치 대조 |
| MI-03 | cache/pages/Fresh/Warm 수명 | 정확 key·accepted seed·물리 완료 후 재사용·unknown 격리 |
| EX-01 | 외부 stop/drain·capability | old-go·자연 완료·pipe·stdout·unsupported |
| EX-02 | CPU_T/CPU_R 선택·자원 | Collection 전체 P/C/V+교사; Deployment P/C+own CPU_R·V-free |
| EV-01 | 실제 view/data V3·trainer/checkpoint | head provenance·WDL anchor·미래 누출·V known pair·shared moments |
| EV-02 | bounded feedback 계측 | queue/service/yield/accept/drain·실제 work·NN/소비 분리 |
| EV-03 | 독립 V5 manifest/lock/receipt·실행기 | 구형 parser/digest 보존·source pins·선택과 실제 관측 |
| EV-04 | 문서·PR·완료 근거 | source/CPU CI/GPU/학습/대국별 실제 범위 |

각 행은 실제 통합 소스·검사 결과와 연결되기 전에는 완료가 아니다. M0→M1/M2→고정
coordinator→adaptive/compact→외부·자료→GPU 순서를 따른다. 실제 학습은 사용자 지시로
보류하며 구현·CPU 검사에 계속 진척한다. 본격적 학습·기력 개선·새 PR 병합은 별도 인수다.
