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

## 21개 항목의 제품 연결과 남은 인수

아래는 2026-10-13의 소스 연결 감사다. `SOURCE_READY`는 해당 제품 접점을 작성했다는
상태이며 최신 전체 검사 통과나 실제 실행 인수가 아니다. CPU 검사 범위와 수정 중인
실패는 뒤의 관리 실행 표에 기록한다. 실제 GPU·세대 수집·학습·대국은 아직 미실행이다.

| 항목 | 실제 제품 소비자 | 현재 소스 상태 | 남은 실제 인수 |
|---|---|---|---|
| VR-01 | [ReviewContextIndex::select/consume](../crates/rz-search/src/pals/review_context.rs) → [bootstrap 선택 정책](../crates/rz-uci/src/pals_adaptive_bootstrap.rs) | bounded bundle/view·실제 소비 두 번 보호 연결 | 실제 제출 tensor에서 기보 비유입·관계·digest·budget 확인 |
| VR-02 | [checked_rebase_offset](../crates/rz-search/src/pals/review_context.rs)·[store archive](../crates/rz-search/src/pals/store/archive.rs) | 정확 Rules suffix 재생·anchor·pin 연결 | 실제 root 변경과 archive 왕복의 문맥·lease 보존 |
| VR-03 | [adaptive role port](../crates/rz-search/src/pals/engine/adaptive_port.rs) → [native backend](../crates/rz-uci/src/pals_adaptive_bootstrap/native_backend.rs) | 제출 view와 준비 입력을 같은 ticket에 보존 | native 입력·수락·실제 소비 view의 독립 수치 대조 |
| CY-01 | [manager progress 요청](../crates/rz-uci/src/pals_adaptive/manager.rs) → [CPU worker](../crates/rz-search/src/cpu_worker.rs)·[stack](../crates/rz-search/src/cpu/stack.rs) | 원래 질문을 검증하는 명시적 PublishProgress API·Yield·Cancel 연결 `SOURCE_READY`; 일반 tick의 자동 발행과 구분 | 수정 후 UCI 재검사와 실제 요청→완료 iteration 공개→ack 순서 확인 |
| CY-02 | [manager reserve/settle](../crates/rz-uci/src/pals_adaptive/manager.rs) | 원래 Job/Slice credit·progress 비청구 연결 | 실제 종료·취소·unknown에서 이중 청구·환불 없음 확인 |
| CY-03 | [CPU worker](../crates/rz-search/src/cpu_worker.rs) → [PausedStack](../crates/rz-search/src/cpu/stack.rs) | private owner·동일 조건 resume 연결 | 실제 여러 중단 위치의 depth/score/PV/nodes 대조 |
| CO-01 | [driver](../crates/rz-uci/src/pals_adaptive/driver.rs)·[continuation machine](../crates/rz-search/src/pals/coordinator/machine.rs) | fixed 순서·Repair 한도·adaptive continuation 연결 | 같은 입력의 synchronous/fixed 비교와 실제 완료 순서 검사 |
| CO-02 | [ticket registry](../crates/rz-search/src/pals/coordinator/tickets.rs)·[native ticket owner](../crates/rz-eval/src/pals_onnx/virtual_review/tickets.rs) | 질문별 view·수락 권한·Fresh pair seal 연결 | 실제 두 pending·늦은/중복 응답·물리 완료 검사 |
| CO-03 | [session lifecycle](../crates/rz-uci/src/pals_adaptive/session_driver.rs) → [manager shutdown](../crates/rz-uci/src/pals_adaptive/manager.rs) | revoke/cancel/drain/join/archive 순서 연결 | 실제 CUDA 정상 종료와 unknown quarantine·archive 보존 |
| AD-01 | [driver 반환 예측](../crates/rz-uci/src/pals_adaptive/driver.rs) → [manager 비용·backlog 제어](../crates/rz-uci/src/pals_adaptive/manager.rs) | own V-free 및 외부 detached feedback까지 연결 `SOURCE_READY` | 실제 완료 표본·censored/tail·반환 비용·starvation·유효 feedback 측정 |
| AD-02 | [startup factory](../crates/rz-uci/src/pals_adaptive_bootstrap.rs) → [resource readback](../crates/rz-uci/src/pals_adaptive_bootstrap/resource_readback.rs) | 원래 startup cancel/deadline을 같은 probe에 전달 `SOURCE_READY`; 기본 SharedControl | 실제 caller/ORT/worker/외부 사용량·affinity·quota 관측; DedicatedControl 별도 인수 |
| AD-03 | [manager selector/control mailbox/notification](../crates/rz-uci/src/pals_adaptive/manager.rs) | bounded channel·독립 제어·terminal 보존 연결 | 실제 full queue·lost wakeup·무진척·starvation 확인 |
| MI-01 | [V3 입력](../crates/rz-eval/src/pals_model/virtual_review.rs) → [Python reference](../experiments/model-research/pals/src/rz_pals_model/virtual_review.py) | 8축 feature·anchor·관계·mask·L bucket 연결 | 모든 bucket/anchor/padding·후보/승격·흑백 입력 대조 |
| MI-02 | [ONNX export](../experiments/model-research/pals/src/rz_pals_model/onnx_virtual_review.py) → [원래 수치 실행기](../crates/rz-arena/src/pals_collect_v3/numerical_process.rs) | 같은 실제 export의 reference/native 실행 경로 연결 | 실제 Torch→ORT→Rust logits/policy/WDL 허용 오차 검사 |
| MI-03 | [V3 physical owner](../crates/rz-eval/src/pals_onnx/virtual_review.rs)·[runtime](../crates/rz-runtime/src/pals.rs) | cache·accepted Warm seed·single-flight·unknown 격리 연결 | 실제 CUDA 완료 후 재사용·취소·늦은 결과와 seeded reference |
| EX-01 | [ExternalUciCpu](../crates/rz-search/src/external_cpu.rs)·[adaptive capability](../crates/rz-search/src/external_cpu/adaptive.rs) | stop→bestmove→readyok·old-go·부분 관측 연결 | 실제 pinned Stockfish의 stop/drain·pipe 오류·unsupported 검사 |
| EX-02 | [외부 선택과 시작](../crates/rz-uci/src/pals_adaptive_bootstrap/external.rs)·[bootstrap](../crates/rz-uci/src/pals_adaptive_bootstrap.rs) | Collection CPU_T와 Deployment own CPU_R·V-free 분리 | 실제 Full 옵션·자원·V 부재·최종 PALS 착수 확인 |
| EV-01 | [실제 view capture](../crates/rz-arena/src/pals_collect_v3/capture/observer.rs) → [trainer](../experiments/model-research/pals/src/rz_pals_model/generation_training.py)·[원래 checkpoint 실행기](../crates/rz-arena/src/pals_collect_v3/training_process.rs) | head provenance·Rules WDL·known pair·공유 moments·freeze 연결 | 실제 128판·P/C 각 2000/V 1000 update·resume·target 최소량 인수 |
| EV-02 | [manager 관측](../crates/rz-uci/src/pals_adaptive/manager.rs)·[native 계수](../crates/rz-uci/src/pals_adaptive/native_accounting.rs) | queue/service/yield/accept/drain과 실제 work 연결; 분리 근거 없는 physical wait는 unknown | 실제 종단 비용·유효 witness·NN 완료/검색 소비·host/VRAM 관측 |
| EV-03 | [V5 startup parser](../crates/rz-experiments/src/pals_startup_v5.rs) → [generation pipeline CLI](../crates/rz-arena/src/bin/pals-generation-pipeline.rs)·[원래 소유자 pipeline](../crates/rz-arena/src/pals_collect_v3/generation_pipeline.rs) | V5·preregistration·Frozen4000/G15000·수치·[final32 소비자](../crates/rz-arena/src/pals_adaptive_launch/evaluation_series.rs) 연결 | 실제 등록 자산·runtime 준비, 전체 원래 소유자·failure 보존과 32쌍 인수 |
| EV-04 | 이 문서·[실행 feature 등록](../crates/rz-arena/Cargo.toml)·관리 로그 | source/검사/실패/미실행 범위를 갱신 | 최신 source SHA의 전체 검사·CI·실제 GPU/학습/대국·자료 회수 보고 |

M0→M1/M2→고정 coordinator→adaptive/compact→외부·자료→GPU 순서를 따른다.
실제 학습은 사용자 지시로 보류한다. 소스 연결, CPU 검사 성공, 실제 실행 인수와
학습·기력 개선·새 PR 병합은 각각 구분한다.

후속 관리 실행 103에서 UCI 566개가 모두 통과했다. 이전 102의 세 실패는 같은
제품 보호 조건을 유지하고 fixture의 관측 경계를 바로잡았다. token discard 처리와
전체 root 회수를 따로 확인하고, Store가 실제 발급한 root 세대로 view를 생성하며,
동일 길이 내용 변경의 SHA 거부와 명시적인 metadata 시각 변화·새 inode 거부를
구분한다. 실행 104의 Arena CPU/mock 검사 319개와 105의 Windows UCI 전체
feature·target 빌드도 통과했다. 실행 106~110의 엄격한 Clippy 실패는 보존하며,
해당 타입·함수의 소유권을 유지하는 후속 소스 수정과 재검사를 진행한다. 큰 실패
객체·완료 메시지는 bounded 원래 owner를 inline으로 반환하도록 유지하고, 선언별
예외에 추가 할당을 피하는 이유를 기록했다. 메모리 개선으로 해석하지 않는다.
실행 112는 새 filesystem fixture의 빈 배열 타입 추론에서 실패했고, 이를 고친
113의 수집·실패 자료 검사 48개는 통과했다. 이후 raw 파일의 선보유 경계를 추가로
보완했으므로 새 소스 검사를 구분한다. 실행 111의 reviewed profile 검사는 framework
import 차단으로 시작 전에 거부됐으며 통과 검사에 포함하지 않는다. 별도 AST·byte
감사 114에서 두 legacy fixture의 각 두 source pin은 현재 파일과 일치했다.
이 결과는 통합 작업 소스의 검사이며 정확한 공유 커밋 SHA의 CI 결과와 구분한다.

학습·export·수치 검증 실패 자료는 원래 등록 경로와 whole deadline·남은 read credit
안에서 읽기 전용으로 회수한다. 열린 FD는 metadata·hash 검사보다 먼저 실패 전용
owner에 보유하며, 빈 파일·여러 링크·partial을 성공 checkpoint·수치 인수 권한으로
전환하지 않는다. 원래 실행 오류와 회수 오류를 따로 남긴다. 원래 시한이 지나 새 FD를
열지 못했으면 `readback_unknown`을 기록하며 모든 실패의 무조건적 회수를 주장하지 않는다.

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

후속 `4528257`의 네 CI job도 실패로 확인했다. CPU와 bindings는 compiled source
목록이 아직 공유하지 않은 새 파일을 참조한 상태였고, 모델 job의 선행 실패 실행
`a9f5d17`은 1,038개 중 53개가 reviewed Rust fixture pin 불일치에서 중단됐다.
새 source pair의 legacy Disabled/SingleReply 경로를 별도로 검토하고 제한된 profile과
synthetic fixture를 새로 등록했다. 과거 pair·unknown successor·hybrid·mutation 거부와
원래 source/build/policy admission은 유지한다. 새 profile은 adaptive 실행 인수가 아니다.
새 소비자 소스와 compiled 목록은 함께 공유하며, 새 HEAD CI 성공은 별도로 확인한다.

추가 관리 실행에서 실제 신경망 없이 resource readback CPU 검사 6개와 V5 명세
검사 223개가 통과했다. V5 첫 실행의 222개 통과·1개 실패 자료도 보존한다. 실패는
같은 크기의 파일 변경 직후 타임스탬프가 달라진다는 fixture 가정이었으며, 내용 변경의
hash 거부·cursor 보존을 유지하고 metadata identity 변화는 명시적인 크기 변경으로
검사했다. 제품의 파일 검증 조건을 완화하지 않았다. metadata-only export CLI 10개,
control inventory 12개, generation export 계약 34개도 통과했으며 framework와 실제
실행을 차단한 검사다. 수치 CLI의 첫 검사 호출은 pytest 부재로 import 전에 중단됐다.
후속 095는 새 의존성 대신 기존 unittest 형식으로 변환한 metadata 검사 8개가 통과한
자료다. 두 실행과 실제 GPU·optimizer 인수를 구분한다.

최신 확인 범위는 관리 보고서 `reports/pals-adaptive-coordinator-20261013`의 아래
로그다. 각 결과는 해당 실행 시점의 소스와 feature 범위에 대한 증거다. 후속 수정한
전체 소스의 검사나 새 HEAD의 CI 성공으로 자동 재사용하지 않고, 실행 사이의 검사
수를 합산하지 않는다. 관리 로그의 정리 성공은 모델·학습·대국 정상 종료의 증거가 아니다.

| 관리 실행 | 확인한 결과 | 범위·남은 검사 |
|---|---|---|
| `rust-linux-experiments-092.log` | 223 통과·0 실패 | V5 parser·source pin·입력/수치 admission의 CPU 검사; 실제 tensor/provider 실행 인수 아님 |
| `python-generation-training-cli-093.log` | 15 통과 | metadata CLI 검사; 모델·optimizer update·실제 checkpoint 재개 미실행 |
| `python-generation-numerical-cli-095.log` | 8 통과 | framework import와 reference/native 실행을 차단한 metadata 검사; 실제 독립 수치 검사 미실행 |
| `rust-linux-runtime-097.log` | 73 통과·0 실패 | mock/CPU 요청·취소·물리 수명 검사; 실제 CUDA 완료 미관측 |
| `rust-linux-search-098.log` | 681 통과·0 실패·2 의도적 미실행 | Rules·CPU·coordinator·외부 transport 검사; 두 ignored 검사를 성공으로 세지 않음 |
| `rust-linux-arena-targets-101.log` | 전체 target 컴파일 성공·종료 코드 0 | 6개 warning 유지; 테스트 실행·generation CLI 실행·학습 인수 아님 |
| `rust-linux-uci-102.log` | 563 통과·3 실패·종료 코드 101 | 취소 후 forwarded paused token, old-root Yield receipt, 변경/동일 bytes runtime 파일 교체 사례 실패; 수정·재검사 중 |

102의 세 실패는 각각
`forwarded_paused_token_discard_after_cancel_never_retires_new_question`,
`old_root_yield_receipt_cannot_change_a_new_root_live_frame`,
`changed_bytes_and_same_bytes_replacement_are_both_rejected`다. 실패 원시 로그와
원래 질문·root·파일 소유권 검증을 보존한다. 후속 소스가 `SOURCE_READY`여도 이 실행을
통과로 바꾸지 않으며, 최신 UCI 재검사 결과와 CI는 별도로 확인해야 한다.

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

전체 실행 진입점은 `pals-collection-onnx` feature의 `rz-pals-generation-pipeline`이다.
`--execute --declaration <등록한 절대 경로> --declaration-sha256 <원래 SHA-256>`의
명시적인 세 인수 묶음이 없으면 실행하지 않는다. 원래 선언은 source128의 CLI 선언,
초기 checkpoint와 Rules profile, frozen action template, Frozen4000/G15000의 독립
수치 case plan·runtime·실행 소스 pin, 수집 전 등록한 32 opening과 pending System
template를 요구한다. 미래 G1 hash를 미리 알도록 요구하지 않고 실제 exporter의
held FD와 checkpoint 소유자에서만 바꾼다. pipeline→수치→최종 평가가 같은 opaque
소유자를 유지하는 소스 경로와, 실제 등록 파일·바이너리·runtime·자산이 준비돼 실행
가능하다는 확인은 구분한다. 현재 후자는 아직 인수하지 않았다.

순서는 원래 128판 archive → 250 update 경계의 P/C 합계 4000 → Frozen4000
Collection export·독립 수치 검사 → 실제 V action 자료 → V 합계 1000 → G15000
Deployment export·독립 수치 검사 → 원래 final32 template의 새 lock → 32쌍이다.
최종 평가는 G0/G1 같은 compact 구조·own CPU_R·V-free, 120초+1초·256 ply·seed 1을
고정한다. 정상 종료된 원래 pair만 execution lease를 해제하며 같은 18GiB 관리 root와
live storage flock·원시 archive 소유자는 보존한다. 첫 인프라 실패에서 중단하고,
엔진 크래시·불법 수·시간패는 결과에 남긴다. 사전 등록한 `FixedPairedHoeffding95`는
32개의 유효 쌍이 끝난 뒤에만 집계하며 partial 결과나 JSON으로 실행 권한을 발급하지 않는다.

자원 readback은 성공한 원래 모델 SessionBuilder·EP 등록·session commit의 설정과
실제 caller·manager·worker·external process 관측을 구분한다. 전체 ORT helper 수,
숨은 quota ancestry, 실제 external Threads/Hash 적용, SMT 배제 및 VRAM peak를
확인하지 못하면 unknown이다. Linux topology probe는 원래 기한·취소와 읽기 상한을
검사하고 Windows native 호출은 전후 제어 검사 범위만 표현한다. 실제 collector의
startup factory·checker·resource probe에 같은 원래 cancel을 연결한 후속 소스는
`SOURCE_READY`다. 이것은 native loader 자체의 즉시 중단이나 CUDA kernel placement
인수가 아니다. 이미 시작한 owner는 원래 startup drain/quarantine 경계로 처리한다.

적응형 반환 예측은 정책의 예상 대기·반환 비용이며 실제 절약 시간이나 V utility가
아니다. 완료 표본과 원래 질문·progress·소비 경계가 없으면 fixed 정책을 유지한다.
초기 Collection+V 전용 own Yield 소비에 더해 V-free P/C와 외부 CPU_T의 detached
feedback를 원래 질문·progress·coherent tuple에 귀속하는 후속 소스가 `SOURCE_READY`다.
own 반환·resume 비용과 외부 stop/drain 비용을 따로 사용하고 필요한 관측이 없으면
보정하지 않는다. 준비·queue·service 관측과 달리 physical fence 대기를 분리할 backend
근거가 없는 현재 `PhysicalCompletionWait`는 unknown으로 남긴다. 반환 예측을 실제
절약 시간·유효 witness·V utility로 바꾸지 않는다. 이 문서의 소스 경로를 21개 항목의
실제 GPU·학습·대국 인수로 표시하지 않는다.
