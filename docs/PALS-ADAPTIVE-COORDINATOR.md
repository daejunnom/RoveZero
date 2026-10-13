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

## 실제 수집 실행 접점과 보존 범위

`rz-arena`의 `pals-collection-onnx` feature는 `rz-pals-selfplay-collect` 부모와
`rz-pals-generation-game-worker` 단일 게임 child를 제공한다. 부모는
`--declaration <등록한 절대 경로> --declaration-sha256 <원래 파일 SHA-256>`만 받는다.
선언 형식은 `rz-pals-selfplay-cli-declaration/1`이며, 128개 게임의 ordinal, 시작 상태,
원래 실행·정리 창, 바이너리, private cwd, startup config/lock, 자원·출력 상한을
실행 전에 고정한다. 이 진입점은 학습·cloud 생성·mount·자산 복제를 암묵적으로 실행하지 않는다.

자산을 담은 private cwd는 미리 준비해야 한다. 실제 동일 inode가 공유되는 경우에는
양쪽 alias의 읽기 전용 mount 관측을 요구한다. 고유 inode bytes와 logical alias bytes를
따로 기록하고, 고유 staging과 실행 출력은 18GiB 상한을 지킨다. 전체 모델/runtime을
128번 복제한 준비 자료가 상한을 넘으면 용량 오류다. 공유되었다는 선언으로 실제 관측을
대신하거나 symlink/hardlink 검사를 완화하지 않는다.

각 게임은 새 OS child를 한 번 실행한다. 부모는 원래 바이너리·입력·실제 loaded-image,
물리 종료·process group 정리, 저장된 입력·관측, Rules 재생과 시계·SAN 기보를 대조한다.
native session epoch의 변화는 OS 재시작 증거가 아니다. 제출·완료·수락·해제와 원래 child
실행 증거를 가진 opaque 결과만 최종 집계에 넣으며 JSON에서 실행 권한을 재구성하지 않는다.

검증이 끝난 게임은 큰 원시 record/evidence 맵을 RAM에서 해제하고 원래 파일 핸들과
무결성 pin, 실행·종료 소유자와 작은 seal 목록을 유지한다. 최종 집계는 정확히 128판,
64개 흑백 교환 쌍, 사전 등록 train/validation/holdout 48/8/8을 확인한다. 최종 index의
domain은 `rz-pals-whole-game-series-index/1`이다. 포함된 dataset manifest는 기존
`rz-pals-generation-chunk-manifest/1`의 의미를 유지하며 원래 게임별 JSONL의 상대 경로를
참조한다. 동일 evidence가 반복되면 내용·seal을 검증한 뒤 정의만 재사용하고 모든 원래
발생 위치를 보존한다. 반복 관측을 새 물리 완료·새 witness·새 학습 sample로 세지 않는다.

출판 재검증의 시간·누적 읽기 credit은 선언에서 따로 고정한다. 이 읽기 전용 창은 원래
child의 실행·정리 창을 연장하지 않는다. 각 게임의 읽기 상한도 그대로 적용하며 실제
소비량을 누적한다. 재검증 중 실패한 소비량을 확인할 수 없으면 남은 예약 credit을
보수적으로 소진하고 추가 read allowance를 지급하지 않는다. 저장 용량과 누적 I/O bytes는
다른 값이다. 이 소스 연결은 실행 시간이나 메모리 개선의 측정 증거가 아니다.

첫 실패에서 해당 series를 중단하고 완료 index를 발급하지 않는다. 원시 stdout/stderr,
부분 index와 소유자는 보존하며 저장 자체의 오류도 별도 기록한다. GPU 수치·물리 수명,
실제 128판, optimizer update·resume, G1 export와 최종 32쌍은 각각 별도 인수다.

## 기존 V2 호환 경로와 현재 인수 상태

실제 startup factory는 V2 `FullLineInteractionV2`의 Full/Selected와 V3
`virtual_review_v1`을 구별한다. Full은 Search가 제출 시 고정한 원래 RoleRecord
순서·값·수열·관계와 Store lease를 사용하고, Selected는 선택된 ReviewView만 받는다.
Fresh pair의 두 endpoint는 같은 원래 Full snapshot을 소유해야 한다. 모델과 입력의
기존 의미 식별·고정 L ABI·cache namespace를 새 compact 의미로 바꾸지 않는다.

Full 경로도 coordinator의 bounded ReviewView 승인을 거치므로 일부 큰 원본 문맥은
`Unknown/ViewCapacity`로 보류될 수 있다. 원본 full 입력의 의미·수명 보존은 원래 동기
full 경로의 모든 문맥이 같은 조건에서 제출된다는 보장과 구분한다. compact/selected
기본 경로에는 full record scan과 snapshot pin을 추가하지 않는다.

V2 native 계수는 같은 실제 exclusive owner 호출 전후의 public cache·public/private
실행 차이에서 얻는다. 여러 record-page 실행은 각각 물리 NN 입력으로 센다. V 실행이
없다는 근거는 검증된 manifest와 실제 로드된 session 집합이다. 계수 포화·감소·혼합·
물리 완료 불명은 알려진 0으로 표시하지 않으며, 검색 소비와 물리 완료도 따로 기록한다.

2026-10-13의 초기 Windows CPU/mock 검사에서 Search는 639개 통과·2개 의도적
미실행, UCI는 527개 통과·1개 실패였다. 실패한 fixed trace의 상태·제한·score·PV·
실제 node 사용량은 같았으나 질문 식별값이 계산 조건에 포함된 차이가 있었다. 계산
조건과 private dispatch custody를 분리한 후 원래 fixture와 판정 기준을 유지한 Linux
fixed trace 검사는 통과했다. 이어서 Linux feature 조합의 Search 675개 통과·2개
의도적 미실행을 확인했다. 플랫폼·feature 조합이 다른 검사 수를 같은 실행으로 합산하지 않는다.

후속 관리 실행에서 Windows 기본 adaptive feature 빌드, native owner 계수의 CPU
검사 4개, runtime CPU 검사 69개가 통과했다. 신경망과 optimizer import를 차단한
trainer CLI metadata 15개와 export CLI metadata 8개도 통과했다. metadata 검사는
실제 optimizer update·ONNX export·CUDA 완료·학습 checkpoint 재개 인수가 아니다.
원격 `77051c7`의 네 CI job은 source inventory 포맷과 선택적 serde 처리에서 실패했다.
후속 `a9f5d17`은 이 두 수정만 공유했으며 네 CI job은 아직 실패 상태다. 모델 CI의
기존 원문 fixture가 고정한 Rust source pin과 새 구현의 차이를 다시 검토하고 있다.
source pin 검사를 완화하거나 hash만 자동 갱신하여 통과로 표시하지 않는다.

## 실제 세대의 소유권과 수치 검증

실제 세대 실행은 같은 원래 128판 archive 소유자를 P/C 학습·V action 수집·V 학습에
유지한다. P/C 4,000 update의 `FrozenPcCollection`과 전체 5,000 update의 G1은
다른 export domain이다. frozen 수집은 Collection과 외부 CPU_T, 최종 평가는
Deployment와 own CPU_R·V-free 조건을 사용한다. 미래 가중치 hash를 사전 결과처럼
등록하거나 JSON에서 완료 game·checkpoint·process 소유자를 재구성하지 않는다.

각 export는 실제 새 graph에서 CUDA control inventory를 만들고, 같은 export와
정확한 준비 입력을 대상으로 reference와 native 수치 검사를 수행해야 한다. G0·frozen·
G1의 수치 증거는 서로 대체하지 않는다. actual trainer/export/numerical 소유자와
수집 전 등록한 32개 opening·template 소유자를 같은 프로세스에 유지하여 최종 평가를
연결한다. 실패한 child의 권한·원시 자료·실제 소비 비용은 보존한다. 새 CLI 호출로
128판을 자동 재수집하지 않으며, 완료 단계 경계의 명시적 재개도 미해결 child를 넘지 않는다.

자원 readback은 성공한 원래 모델 SessionBuilder·EP 등록·session commit의 설정과
실제 caller·manager·worker·external process 관측을 구분한다. 전체 ORT helper 수,
숨은 quota ancestry, 실제 external Threads/Hash 적용, SMT 배제 및 VRAM peak를
확인하지 못하면 unknown이다. Linux topology probe는 원래 기한·취소와 읽기 상한을
검사하고 Windows native 호출은 전후 제어 검사 범위만 표현한다.

적응형 반환 예측은 정책의 예상 대기·반환 비용이며 실제 절약 시간이나 V utility가
아니다. 완료 표본과 원래 질문·progress·소비 경계가 없으면 fixed 정책을 유지한다.
초기 Collection+V 전용 own Yield 소비와 V-free P/C 또는 외부 CPU_T 소비는 별도
연결·검사 대상이다. 이 문서의 소스 경로를 21개 항목의 실제 GPU·학습·대국 인수로
표시하지 않는다.
