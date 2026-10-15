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

후속 CPU 빌드 검증에서는 UCI 전체 feature/target의 엄격한 Clippy(115)와 Arena의
엄격한 Clippy(117)가 통과했다. `35bf6a6`의 CI는 PALS model CPU job만 성공했고,
Linux CPU·Windows CPU·독립 bindings job은 실패했다. 각 실패는 별도로 보존한다.
독립 bindings lock에는 실제 path 의존성이 요구하는 capability 파일시스템 closure만
추가했으며 기존 PyO3와 libc 버전은 유지했다. 온라인 metadata(119)와 bindings core
빌드(120)는 통과했다. 이것은 Python 호출 성능이나 실제 학습의 검사가 아니다.

Windows 실패는 Linux-native training 선언 테스트의 `/tmp` 절대 경로 가정이었다.
fixture를 실제 플랫폼의 절대 경로로 구성하고 Windows drive prefix 거부를 그대로
검사한다. 제품의 비밀 경로·상위 이동·지원하지 않는 플랫폼 거부를 완화하지 않는다.
Linux CPU executable 실패는 관리 재현(121)에서 전체 실행 파일 해시의 128MiB
한도 초과로 확인했다. stdout EOF뿐 아니라 bounded stderr·종료 상태·파일 크기를
테스트에 보존했으며 193,004,120byte 파일의 exit 2를 기록했다.

루트 dev profile의 debug 정보만 `1`로 제한했다. test는 dev 설정을 상속하고
최적화·assertion·overflow·panic 설정은 기존 기본값을 유지한다. 타입·변수 수준의
debug 정보는 줄지만 줄/모듈 정보는 유지한다. 이는 개발 빌드 준비 변경이며 탐색·모델
성능 개선 자료가 아니다. 해시 한도·전체 bytes 해시·원래 실행 파일 소유권 검사는
유지한다. 재빌드(122)의 실제 파일 크기는 113,719,080byte였고 원래 CPU UCI 실행
검사 5개가 통과했다. 새 바이너리는 source commit·profile·실제 hash로 별도 등록하고
이전 바이너리의 GPU·대국 증거를 자동 재사용하지 않는다.

전체 Linux workspace/feature/target 검사(123)는 컴파일 중 관리 catalog의 12GiB
상한에 도달하여 종료 코드 125로 중단했다. 원시 로그와 종료·정리 증거는 보존한다.
관리기가 비활성 완료 cache를 회수한 뒤 같은 소스·조건으로 실행한 124는
2,881개 통과·0 실패·18개 의도적 미실행, 종료 코드 0이었다. Arena 326개에는
실패 출력의 원래 FD 보존 검사가 포함되며, ignored 검사는 실제 GPU 인수로 세지 않는다.
이후 strict Clippy(125)가 지적한 V5 테스트의 두 default 재할당 표현만 초기화로
정리했다. 제품 admission과 assertion은 동일하며 전체 strict Clippy(126)는 통과했다.
124 종료 시 자신의 재생성 build slot이 8GiB를 넘어서 관리기가 이를 회수했고,
cleanup verified와 작은 ownership record만 남긴 상태를 확인했다. 모델·원시 실행 자료·
보고서를 삭제한 것이 아니며, 관리 cache 정리를 성능 인수로 해석하지 않는다.

독립 bindings core의 Windows 기본 feature/전체 target strict Clippy(127)가
통과했다. Windows training 경로 fixture(128)는 1개 통과했지만 Linux 전용
구현의 미사용 경고 46개가 있어 strict Windows 인수로 보고하지 않는다.
실제 Linux 소비자와 같은 `cfg`로 import·내부 helper·constructor를 좁게
구분하고, 공개 compatibility signature와 원래 파일 보관 수명은 유지했다.
공개 거부 접점과 custody 필드는 필요한 선언에만 이유가 있는 lint 예외를
적용했으며 Windows에서 Linux-native 실행을 허용하지 않는다. 이후 V5의
수정된 fixture 검사(129)는 42개 통과·0 실패, cleanup verified였다.

Windows 전체 strict Clippy(130)는 원래 deadline을 축소하는 내부 helper의
제품 Windows 미사용 오류 1개로 실패했다. 원래 판정과 테스트를 유지하며
해당 helper에 실제 Linux 제품·단위 검사와 같은 빌드 조건을 적용했다.
실패와 정리 로그는 보존하고 후속 전체 strict 검사를 별도로 기록한다.

전체 Windows strict Clippy(131)는 통과했고, Arena all-feature library 검사(133)는
255개 통과·0 실패였다. 두 실행 모두 종료 코드 0·cleanup verified다. 그 사이
정리 완료 전에 시도한 132는 live owner 거부로 시작되지 않았으며 검사 성공으로
세지 않는다. 공유 `870dd18`의 CI에서 bindings·PALS model·Windows CPU job은
성공했다. Linux CPU job은 Search 680개 통과·1개 실패·2개 의도적 미실행이었다.
paused CPU 작업의 shutdown 확인 사이에 종료 이벤트를 건너뛰는 실제 경로를
확인했으며, 후속 수정과 결정적인 경쟁 검사를 별도로 인수한다. 기존 실패 실행을
통과로 바꾸거나 CI 세 job의 성공을 전체 CI 성공으로 합산하지 않는다.

이 경쟁의 후속 수정은 paused owner가 있으면 원래 terminal permit의 seal·종료
이벤트 경로로 돌아가도록 제한했다. 실제 Yield 뒤 두 shutdown 관측 사이의 순서를
bounded channel로 고정하며 credit·token·owner 판정을 유지한다. Windows 관리 검사
135에서 CPU worker 28개와 Search 전체 target/feature strict Clippy(136)가 통과하고
종료·정리도 확인했다. Linux 실행 134는 컴파일
도중 host/WSL 실행 환경이 종료되어 최종 관리 영수증을 얻지 못했다. 그 실행은 통과나
제품 테스트 실패로 재분류하지 않으며, 원래 lease와 중단 자료를 보존해 확인한다.

134의 lease가 현재 Linux boot보다 오래됐음을 metadata와 boot 시각으로 확인하고,
원래 owner·active record를 보고서에 보존한 뒤 자신의 중단 lease만 회수했다. cache와
scratch는 그때 보존했으며 새 137은 같은 조건에서 CPU worker 28개 통과·종료 코드 0·
cleanup verified였다. 원래 134를 정상 종료로 표시하지 않는다. 공유 `29163e4`의
CI 실행 `38426529151`은 Linux·Windows CPU·bindings·PALS model 네 job 모두 성공했다.
이 정확한 SHA의 CI와 이후 G0 소비자 변경의 검사·실제 신경망 인수는 구분한다.

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

### 학습 없이 G0 수치를 준비하는 별도 경로

`rz-pals-g0-numerical --execute --registration <절대 경로>
--registration-sha256 <원래 SHA-256>`는 미학습 compact G0의 별도 실행 접점이다.
기본 호출이나 JSON 재구성으로 모델·신경망을 실행하지 않는다. 원래 등록은 V5
startup template·Rules descriptor·고정 numerical case plan·실제 Python/native
바이너리·runtime source pin과 유한 execution/cleanup 창을 요구한다.

원래 Rust 부모가 보유한 private source와 출력 디렉터리에서 registered initializer로
seed 23·steps 0·trained false를 생성하고, Collection P/C/V export와 새 static CUDA
control inventory를 만든다. 실제 산출물의 FD·pin으로 reference와 native capture를
연결한다. weights·graph는 원래 private cwd의 별도 하위 디렉터리에 두어 큰 산출물을
다시 복제하지 않는다. 종료 후 작은 raw payload와 acceptance는 기존 bounded 경로로
결합한다. 모든 L bucket 등의 coverage, 입력 bytes와 기존 수치 오차 기준은 공통 V5
verifier가 확인한다.

성공 소유자는 **G0 numerical-only**다. startup content admission·학습·수집·arena
권한을 발급하지 않으며, 후속 실제 startup은 보존한 원래 roster·FD를 다시 검증한다.
미지원 Windows 실행은 명시적으로 거부한다. Python의 기존 initializer/export는 개별
write fence를 제공하지 않으므로, 사전 conservative phase 예약과 원래 부모의 tree
watch를 사용한다는 한계를 영수증에 남긴다. 실패·partial·완료 unknown은 같은 원래
소유자와 read credit·whole deadline 안에서 보존한다.

capture의 선택적 `cudnn_conv_search`는 생략 시 기존 implicit exhaustive 정책을
유지한다. `experimental_default`는 명시적인 eager V3 CUDA-control·host profile에서만
선택하며 CPU·shim·device Warm을 조용히 대체하지 않는다. 실제 session 설정과
물리 완료·worker join 후의 정책 관측을 별도로 남기고, 미관측 cuDNN plan·실제 TF32는
unknown이다. 이 소스 연결과 metadata/CPU 검사 성공은 실제 초기화·수치·GPU 인수가
아니다. 현재 실제 학습 보류 상태에서 generation pipeline을 G0 검사용으로 실행하지 않는다.

후속 관리 검사에서 Python G0 metadata 16개(138), native capture의 순수 설정·입력
검사 10개(139)가 통과했다. framework·initializer·NN·optimizer 실행을 차단한 검사와
실제 수치 실행을 구분한다. Windows Arena 전체 feature/target strict Clippy(142),
Linux의 같은 strict Clippy(145), Linux Arena library 328개(146), Windows Arena
library 257개(147)가 통과했으며 각 원래 command의 종료 코드 0과 관리 cleanup을
확인했다. 두 OS의 검사 수를 합산하지 않는다. 143의 owned path 비교 경고는 기존
동등성·FD 수명을 유지하는 borrowed path 비교로 수정했으며 실패 로그를 보존했다.
formatting 검사는 141·144에서 통과했다. 이 결과는 해당 working source 범위이며
새 공유 SHA의 CI, 실제 G0 초기화·export·독립 reference·GPU 인수를 대신하지 않는다.

### 최초 생성 전 선언과 생성 후 수치 범위의 분리

아직 존재하지 않는 G0 weights·export·graph hash를 startup template에 미리 넣는
순환 의존은 별도 `/2` bootstrap 경계에서 해소한다. 기존 `/1` 완료 template 경로와
V5 startup·lock 검증은 유지한다. 최초 생성에 사용할 접점은
[G0 bootstrap CLI](../crates/rz-arena/src/bin/pals-g0-bootstrap-numerical.rs)의
`rz-pals-g0-bootstrap-numerical --execute --registration <등록한 절대 경로>
--registration-sha256 <원래 SHA-256>`다. 정확한 인수 묶음과 Linux 실행 환경이 없으면
입력 읽기·자식 실행 전에 거부한다.

[최초 선언](../crates/rz-arena/src/pals_collect_v3/g0_bootstrap_process.rs)은
`rz-pals-g0-bootstrap-numerical-launch/2`, `rz-pals-g0-pre-init-numerical-scope/2`,
`rz-pals-g0-numerical-cases/2`, `rz-pals-g0-native-capture-recipe/2`로 구분한다.
이미 존재하는 source·Rules·Python·ORT/CUDA 입력과 private 실행 경로·자원·유한 기한만
등록한다. 모델 recipe와 case spec에는 미래 weights·export·graphs·model epoch 필드가
없으며 추가 필드도 거부한다. CPU_T나 전체 startup·lock을 가상의 값으로 채우지 않는다.

원래 parent는 seed 23·optimizer update 0의 실제 초기화와 Collection P/C/V export를
수행한 prepare child의 출력 descriptor를 검증한다. 그 후에만 실제 weights·export·
graph pin과 모델 namespace epoch 1을 결합한 수치 범위를 만든다. epoch 1은 초기 모델의
cache namespace이며 학습 횟수가 아니다. 독립 reference와 native capture는 이 실제
범위와 동일한 sealed case 순서·입력을 사용한다. 순수 Python case factory는 Collection의
18개 그룹·150개 사례를 준비하지만 framework·모델·추론을 실행하지 않는다.

[수치 전용 scope](../crates/rz-experiments/src/pals_startup_v5/content_admission/numerical_scope.rs)는
기존 수치 core의 입력 bytes·합법 후보 순서·Fresh B1·FP32·TF32 off·coverage·허용 오차와
물리 완료 검사를 재사용한다. 이 scope와 반환 opaque owner에는 전체 startup·lock·
CPU_T·학습·수집·대국 권한이 없다. 기존 full-config API의 projection은 같은 core를
사용하며 full startup의 별도 검증과 Frozen Collection의 Warm 거부는 유지한다.

최초 등록 FD·원래 process·deadline·잔여 read credit은 prepare→reference→native→
최종 저장까지 같은 owner가 보유한다. acceptance·parent receipt의 create-new FD는
쓰기·동기화 전에 실패 custody에 들어가며, 검증 성공 후 같은 FD만 성공 목록으로
이동한다. 최종 저장 이후에도 원래 cancel과 whole deadline을 확인한 뒤에만 성공
owner를 발급한다. partial·빈 파일·미검증 FD와 물리 완료 unknown은 실패 자료로 남긴다.
parent receipt의 `charged_read_bytes`는 자체 publication/readback 전 snapshot이며,
같은 owner의 실제 read ledger는 그 후 비용도 계속 계수한다.

소스·metadata·CPU 검사를 통과했다는 사실과 실제 G0 초기화·export·Torch/ORT/Rust
수치·GPU 인수는 각각 따로 기록한다. 최초 bootstrap 성공을 실제 학습 세대 완료나
전체 UCI readiness·대국 인수로 바꾸지 않는다.

### 같은 원 G0의 Collection과 Deployment 수치 검사

[별도 thin CLI](../crates/rz-arena/src/bin/pals-g0-collection-deployment-numerical.rs)는
다음 명시적 인수로만 두 수치 단계를 실행한다. 기본 호출과 미지원 OS는 입력 읽기나
신경망 실행 전에 거부한다.

```text
rz-pals-g0-collection-deployment-numerical --execute
  --collection-registration <Collection 등록 절대 경로>
  --collection-registration-sha256 <실제 SHA-256>
  --deployment-intent <사전 intent 절대 경로>
  --deployment-intent-sha256 <실제 SHA-256>
```

[Deployment intent](../crates/rz-arena/src/pals_collect_v3/g0_deployment_numerical_process.rs)는
`rz-pals-g0-deployment-numerical-intent/1`이며 새 출력 경로와 유한 예산을 등록한다.
아직 없는 Collection acceptance·새 가중치·owner를 받지 않는다. Collection NN 전에
최대 64KiB의 원 intent를 30초·16MiB 읽기 예산 안에서 검증하고 opaque preflight
owner가 첫 FD를 보유한다. 잘못된 SHA·정적 구조는 Collection을 시작하기 전에 거부한다.

Collection의 기존 18 group·150 사례를 바꾸지 않는다. 같은 프로세스에서 성공한 live
Collection owner·원 checkpoint FD를 유지한 뒤, 별도 30초·16MiB metadata 단계에서
실제 acceptance pin을 넣은 `rz-pals-g0-deployment-numerical-launch/1`을 create-new로
봉인한다. 원 staging parent FD 아래의 새 registration과 실제 실행이 연 첫 FD의
정확한 pin·inode/device/length/UID/single-link를 연결한다. JSON으로 owner를 복원하거나
종료된 Collection 시간·read credit을 연장하지 않는다.

Deployment 단계는 원 plan의 P/C 입력·순서·bytes를 보존한 12 group·100 사례를 사용한다.
[독립 Python 소비자](../experiments/model-research/pals/src/rz_pals_model/g0_deployment_numerical_cli.py)는
같은 원 G0 checkpoint에서 기존 `include_validator=false` export를 호출하고
public/shared_pc 두 graph만 허용한다. 새 initializer·optimizer를 실행하지 않는다.
오프라인 strict checkpoint 복원은 전체 모델을 일시적으로 구성할 수 있으며 이 비용도
기록한다. 이 수치 인수와 실제 Deployment 엔진 시작의 V 생성·로딩·호출 0 증거는 별개다.

실패 영수증·원 stdout/stderr byte·read 계수와 파일 custody를 유지한다. pending child나
group cleanup unknown이면 CLI는 전체 opaque 실패 owner를 process 종료까지 격리하고
`quarantine_required`를 기록한다. 상위 감독자의 실제 종료·회수 확인 없이 정상 물리
완료로 표시하거나 자동 재시도하지 않는다. 두 단계가 모두 성공해도 numerical-only이며
startup·실제 수집·학습·arena 권한을 발급하지 않는다. CPU 순수 검사·CI·실제 NN·GPU와
대국 인수는 각각 실제 실행 SHA와 결과를 확인한다.

2026-10-14 소스 검사에서는 Linux의 experiments default/types/admission 215/237/238개와
bootstrap 6개, Windows arena all-features 263개가 통과했다. 두 OS의 strict Clippy와
framework 실행을 차단한 Python metadata 17/16개도 통과했다. Linux 전체 arena의
fixture ID 실패는 제품 검증을 유지한 채 fixture만 수정하고 해당 6개를 재검사했다.
이는 실제 최초 생성·신경망 수치·CUDA·정상 종료 인수를 대신하지 않는다.

### 실제 학습 세대의 원래 소유자

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

## 동일 round의 own CPU 예산 소진

own CPU의 root 한도는 실제 native node의 `spent + outstanding_reserved`로 관리한다.
다음 round의 시작 검사만으로 같은 round 안의 CounterValidation·Repair 등 후속
질문을 제한할 수는 없다. 다음 CPU 질문 직전에도 settled 물리 예산과 해당 질문의
논리 조건을 대조한다. 정상 소진 시 새 작업이나 완료 증거를 만들지 않고 기존 root
quiesce·질문 권한 폐기·cancel·physical drain·owner 종료 경로로 닫는다. 결과의 완료
이유는 기존 `PalsCompletion::CpuNodeLimit`이며 마지막 유효 합법 착수를 유지한다.

물리 사용량이 원 한도에 도달하고 예약이 0인 상태와 예약만 한도에 도달한 상태를
구분한다. 후자는 실제 결과와 ack를 계속 수신하며 질문을 보류한다. 정상 종료로
인정하려면 port와 manager의 원 한도가 같고 semantic 사용량이 실제 settled 사용량을
초과하지 않아야 한다. ledger 오류·한도 초과·unknown·격리된 결과·잘못된 질문이나
depth·namespace를 예산 소진으로 숨기지 않는다. 외부 엔진의 reported nodes는 이
native credit 판정으로 환산하지 않는다.

`1e00c9e`의 단일 root CPU 실행에서 실제 한도 1,024·semantic 소비 1,024·manager
소비 1,024·예약 0 뒤 CounterValidation 질문이 `InvalidLimits`로 실패한 원자료를
보존했다. 이는 기본 128/256 한도가 적용된 실행이 아니다. 원 실행은 role 출력 14개와
NN 입력 28개를 소비했고 정상 process/archive 종료를 확인했지만, 검색 인수는 실패다.
두 root를 실행한 이전 기록의 CPU 256 진단을 이 실패 root의 사용량으로 연결하지
않는다. 실패 경로의 `SearchDiagnostic`와 성공 반환의 `DriverWork` 귀속은 다르다.

같은 SHA의 새 비학습 G0에서는 Collection 150사례·Deployment 100사례, NN 입력
300개·200개, 두 physical join과 원 자식 6개의 종료를 독립 검증했다. 새 checkpoint의
실제 바이트는 이전 G0와 같았다. 이 수치 자료는 CPU 검색 수정의 실제 재실행이나
GPU·성능·학습·대국 인수를 대신하지 않는다. 실제 학습은 계속 보류한다.

수정 소스는 CPU/mock 69개와 `pals-adaptive,onnx-cpu` 실행 파일의 타입 검사를
통과했다. 새 회귀는 실제 CpuWorker/PausedStack과 role mock의 자연 소진을 사용하며
실제 Frozen WDL 경로의 수정 후 인수는 새 바이너리 실행으로 별도 확인한다.

## 66aff9f 실제 CPU 수정 인수

`66aff9fb522a0a3f656049da486f2e39c7f39164`에서 정상 CPU 예산 소진을 수정한 뒤,
같은 CPU 1024/작업당 128/depth 2와 Fresh·B1·P/C·V-free·Frozen WDL 조건으로
단일 `position startpos` → `go movetime 1000`을 다시 실행했다. 새 native build의
552개 Git-LF source·25개 component·6개 ELF·22개 원 child 종료, 별도 caller의
원 source 3개·lock·ELF·11개 child 종료와 등록한 notebook 코드/count 1을 대조했다.
이 SHA의 Linux·Windows·CPU bindings·PALS model CI 네 job도 모두 성공했다.

새 G0는 Collection 150사례/NN 300개, Deployment 100사례/NN 200개와 입력 tensor
bytes·raw/policy/WDL 허용 오차·원 자식 종료·두 물리 join을 독립 대조했다.
404개 archive member를 해시 검증했으며 새 checkpoint bytes는 직전 G0와 같았다.
이는 새 producer 실행의 결과이며 옛 실행을 새 source의 증거로 재표시하지 않는다.

실제 검색은 CPU spent 1024/reserved 0, role 결과 14개와 NN 입력 28개를 소비하고
합법 수 `a2a4`를 반환했다. `SearchFailed`와 late canceled result가 없었고,
ticket 14개의 수락·물리 완료·해제, root quiesce 1회, 반복 stop·새 게임·quit,
정상 process/archive 종료를 원 18개 파일과 대조했다. 원 stderr의
`SearchDiagnostic` 1개는 정상 PALS 작업량 요약이며 오류로 숨기거나 삭제하지 않았다.
Deployment 영수증에 완료 enum이 직렬화되지 않으므로 `CpuNodeLimit` 직접 관측은
unknown이다. 해당 source 분기와 원 counter가 일치한다는 추론을 별도로 보존한다.

원 부모 관측의 go→bestmove 약 0.3103초와 준비를 포함한 전체 약 19.61초는
서로 다른 구간이다. cgroup은 미적용, peak·VRAM·managed VM teardown은 unknown이며
이 기능 검사를 성능 개선 자료로 사용하지 않는다. Repair·Yield·동시 pending 두 개,
GPU·학습·대국은 이 실행에서 인수하지 않았다. 실제 학습은 계속 보류한다.

별도 L4 load/mapping 진단에서는 정제한 환경의 libcuda/NVML 기본 soname 탐색이
실패했지만 retained-FD 절대 load와 매핑은 확인됐다. CUDA API·provider·NN 성공으로
해석하지 않는다. 뒤의 단일 API 요청은 코드 실행 전 capacity 실패로 종료됐으며,
원 실패와 실행하지 못한 범위를 보존하고 기존 runtime/ambient guard를 유지한다.
