# PALS 독자 엔진 구현과 인수 기록

사용자가 제공한 `RoveZero_PALS_Internal_Algorithms_KO.md`와
`RoveZero_PALS_Architecture_Flows_KO.md`는 설계 근거다. 문서 속 학습·실행 권고를
실제 실행 승인으로 확대하지 않는다. 원래 목표는 **실제 학습을 제외한 구현**이며,
기반 언어는 Rust다. 기존 Rules·PUCT·LC0 모델 경로와 과거 V1/V2 기록은 보존한다.

이 첫 목표의 비학습 인수는 아래 역사 기록에 유지한다. 2026-10-10 사용자가 추가한
PR23 후속은 새 모델·실행 계약과 diagnostic nonzero 학습 smoke를 포함한다. 이번
후속의 범위·상태는 다음 절을 기준으로 읽으며, 이전의 학습 제외 문구나 과거 CPU/CI/GPU
자료를 새 기준의 실행 승인·인수로 확대하지 않는다.

## 2026-10-10 PR23 후속 범위

[후속 연결 계약](PALS-FOLLOWUP-CONTRACTS.md)은 기존 `EvalRequest/EvalOutput`
revision 0.1과 V1/V2/V3 자료를 보존한다. 아래는 승인된 범위와 현재 준비·검사·미실행의
구분이다. 본격 학습, 제품 learned V의 연결·전략·기력 인수와 PR 병합은 이번 후속에서 제외한다.

### 모델과 실행 명세의 선택

| 접점 | 새 기본 또는 명시 선택 | 기존 경로와 구분 |
|---|---|---|
| 새 모델 init·제품 profile 생략 | `full_line_interaction_v2` / `FullLineInteractionV2` | `baseline()`, 기존 `ModelConfig()`와 legacy serialization의 의미는 유지한다. 새 기본을 과거 자산의 재해석에 쓰지 않는다. |
| 모델 제거 대조 | `full_line_v2`는 input-only, `interaction_head_v2`는 head-only, `full_line_interaction_v2`는 둘 모두 | `legacy_summary_v1`은 explicit legacy 비교다. 전체 수순 입력과 interaction head 효과를 분리한다. |
| 새 후속 등록·pilot 명세 | `PalsRunManifestV4`, `schema_version=4`, `contract_revision=pals/0.2`와 V4 lock | 새 준비 경로가 `--pals-arena-wire=v4`와 명시한 profile/policy를 child에 전달한다. V1/V2/V3 codec·digest는 보존한다. |
| 일반 제품 CLI wire | `--pals-arena-wire=v4`로 명시 선택 | wire 생략은 기존 호환 경로다. 모델의 새 기본값만으로 wire를 V4로 바꾸지 않는다. |

모델 설정·초기화는 [Python config](../experiments/model-research/pals/src/rz_pals_model/config.py),
[model](../experiments/model-research/pals/src/rz_pals_model/model.py)와
[CLI](../experiments/model-research/pals/src/rz_pals_model/cli.py), Rust 모델 계약은
[pals_model](../crates/rz-eval/src/pals_model.rs)을 따른다. V4 선언 검증은
[manifest](../crates/rz-experiments/src/pals_manifest_v4.rs), 새 준비·실제 argv는
[pair 준비](../crates/rz-arena/examples/pals_pair_prepare_v4.rs)와
[V4 launch](../crates/rz-arena/src/pals_launch/v4.rs), 제품 선택은
[UCI main](../crates/rz-uci/src/main.rs)과 [driver](../crates/rz-uci/src/search_driver.rs)에 연결한다.
선언·compiled Rules descriptor는 실제 NN·provider readiness·GPU 완료 관측과 별도다.

### checker·resolver·Fresh 재검토와 반복 queue

`CpuChecker`와 `ResolverPolicy`는 독립 선택이다. 제품 CLI의 resolver 생략 기본은
own checker의 `OwnRawRestricted`, 외부 checker의 `ModelWdlRestricted`다. own raw
정책은 자체 CPU value namespace만 소비하며, model WDL 정책은 own/외부 checker를
지원한다. 외부 CP/mate·bound·작업 단위는 own raw·Rules 사실·자체 CPU 노드와 구분한다.
새 모델과 explicit legacy의 V4 비교에서는 같은 own checker·`ModelWdlRestricted`를
고정하여 resolver 차이를 모델 효과로 섞지 않는다.

`PostRepairRecheckPolicy`의 기본은 `Disabled`다. 명시적 `FrozenModelWdlV2`는 실제
C endpoint와 새 P endpoint를 frozen 모델의 Fresh WDL로 다시 계산하고, 같은 관점·모델·
정밀도·공개 문맥 revision에서 비교한다. 두 Rules terminal은 Rules로 비교하며,
mixed/unknown은 unresolved로 보존한다. `ContextWdl` 원시 관측 두 개와 dependency를
참조하는 조건부 결론을 만든다. 엄격한 개선이 있을 때만 `ConditionalRepairWdl`과
`RepairedBy`로 기존 반박을 supersede한다. 추가 NN은 원래 전역 시간·작업 예산에 포함한다.

반복 queue는 `IterativeFrozenModelWdlV2`를 선택했을 때의 S 정책이다. first move별
Repair 최대 3회·pending 64개, 같은 질문/revision 중복 금지·새 증거가 없을 때 종료를
적용한다. 실제 새 C continuation과 Fresh 관측, 원래 first move·근거·supersedes를
보존한다. queue를 별도의 기본 활성 기능으로 해석하지 않는다. 구현 접점은
[checker](../crates/rz-search/src/cpu_checker.rs), [engine](../crates/rz-search/src/pals/engine.rs),
[frozen recheck](../crates/rz-search/src/pals/engine/frozen_recheck.rs)다.

### bounded archive와 선택적 CPU·Warm 수명

기존 hot/cold 저장소에 더하는 중탐색 controlled allocation은 Store·Engine·Native·Arena V4에
연결했다. [store](../crates/rz-search/src/pals/store.rs),
[archive](../crates/rz-search/src/pals/store/archive.rs)와
[engine archive lifecycle](../crates/rz-search/src/pals/engine/archive_lifecycle.rs)이 경계다.
`with_archive_allocation`은 원래 deadline·잔여 byte 예산과 기존/resident/temporary 핀을
operation 동안 결합하고, 성공·실패 모두 실제 사용량을 반환한 뒤 기존 핀을 복원한다.
새 state/situation·아직 Node/RoleRecord에 설치하지 않은 line/observation도 임시 보호한다.

hot 80% 또는 Capacity에서 비활성 자료를 archive commit·무결성 확인 뒤 회수하고,
Capacity 재시도는 정확히 한 번까지만 허용한다. commit 뒤 allocation/retry와 logical
acceptance 앞의 controlled callback은 `ArchiveCanceled`/`ArchiveDeadline`을 검사한다.
취소 뒤 재시도를 계속하거나 conclusion revision을 게시하지 않는다. all-pinned,
quota/I/O/byte-budget 실패는 typed 오류로 남기며 증거 삭제로 성공을 만들지 않는다.
게임 256MiB·관리 전체 4GiB와 RAM 인덱스 상한을 유지한다. 중탐색 회수에서는 Node
Vec compaction이나 usize 재발급을 하지 않는다. physical active/cancel-requested 작업의
dependency closure는 유지하며, `PhysicalCompletionUnknown`은 실제 완료 전 owner·buffer·
관련 근거를 격리·보존한다. 논리 취소나 다음 root 준비가 물리 완료를 대신하지 않는다.

새 게임은 독립 Store/cold-handle owner를 만들면서 같은 session write ledger의 ID·상한·
실제 누적 소비를 승계한다. 부분 쓰기·commit 후 실패 비용은 유지하고 미작성 예약만
돌려준다. 기존 runtime 8필드와 archive 형식을 유지하며 ledger 세 nullable u64는
Native→Arena 실제 관측에 전달한다. owner별 쓰기와 session 전체 소비, close·pending 파일·
unknown을 대조하고 선언으로 complete를 만들지 않는다. startup·staged new-game owner의
실제 관리 사용량 scan은 원래 deadline 안에서 수행하며 getter는 새 scan을 하지 않는다.
자세한 형식·보존 경계는 [controlled archive 계약](PALS-FOLLOWUP-CONTRACTS.md#중탐색-controlled-archive-계약)을 따른다.

`PausedStack`은 opt-in이며 기본은 기존 재귀 경로의 `CompletedIteration`이다. own CPU의
미완료 root/PVS/negamax/qsearch/SEE frame·cursor/PV·alpha-beta·정확한 Rules/history와
accumulator, TT/ordering owner를 보존한다. 한 실제 paused task·8MiB 한도와 일회 token을
적용하고 소비한 nodes를 재실행·재청구하지 않는다. 8MiB checkpoint frame 한도와
TT/ordering history 예산은 별도다. 매 admission에서 actual owner를
검사하며 stale pause는 역사 증거로 retire하고 Start로 회복한다. CPU 호출 밖의 논리
취소도 실제 stack을 폐기하되 TT/history와 GPU lease 수명은 별도로 유지한다. 외부 checker는
PausedStack을 지원하지 않는다. 소스는 [CPU](../crates/rz-search/src/cpu.rs),
[stack](../crates/rz-search/src/cpu/stack.rs),
[CPU lifecycle](../crates/rz-search/src/pals/engine/cpu_lifecycle.rs)다.
orphan frame의 actual disposal·token/bytes 0·complete snapshot 확인 전에는 논리 pause를
retire하지 않는다. `None`·incomplete는 성공이 아니며 resumed value 실패의 새 작업을 보존한다.

host Warm과 CUDA ApproxWarm은 별도 capability이며 둘 다 기본 off다. CUDA의 명시 선택은
`--pals-cuda-private-warm=true`이고 `onnx-cuda`·`experimental-io-binding`과 등록된 FullLine
V2 device public memory가 필요하다. accepted seed는 같은 role·모델/epoch·Rules/history·
query 문맥에서만 사용하고 공개 record revision만 변경 가능하다. Value/V는 Fresh다.
unknown/cancel의 input/output/seed/workspace owner는 물리 완료 전 보존·격리한다.
request/execution/bank/ordinal과 accepted seed의 실제 join, 마지막 physical drain·buffer 0은
소스 계약과 별도로 실제 실행에서 인수한다.
소스는 [CUDA Warm](../crates/rz-eval/src/pals_onnx/warm/cuda.rs),
[private 경계](../crates/rz-eval/src/pals_private.rs)와
[native Warm](../crates/rz-uci/src/pals_native/private_warm.rs)다.

### 진단 학습과 남은 실행 인수

이전 diagnostic nonzero smoke의 세 번째 실행은 P/C/V별 연속·재개 비교 총 24 update에서
모델·AdamW·scheduler·RNG·sampler 일치, parameter membership·동결·유한값 PASS다.
이전 1+8 update 실패를 보존하며 실제 누적은 33회, 실제 target update는 0회다.
진단에 영향 주는 source 8개를 직접 대조한 재사용 확인은 이전 실제 실행과 별도 근거다.
이번 새 optimizer update는 0회이며 제품 learned V utility/action의 실제 유용성 근거는 부족하다.
사용자가 총 update 한도를 해제했지만 실행별 자원·시간·출력 상한과 실패 기록은 유지한다.
상세 범위는 [학습 smoke 기록](TRAINING-PLAN.md#2026-10-10-pals-diagnostic-학습-smoke)을 따른다.

`8bc8e26`은 이 문서 갱신에서 대조한 원본 코드 기준이다. 이후 수정된 `662094e`의
검사·CI는 해당 SHA로 별도 인수하며, 이전 CI 실패·수정 이력을 보존한다. 현재 CI 최종
상태는 미확인이고 실제 새 CPU/NN/GPU 검사는 0회다. Windows commit 여유 재측정
2.96GiB는 확인 시점의 관측값이며 기존 6GiB 시작 조건에 못 미쳐 실행을 시작하지 않았다.

| 현재 후속의 인수 범위 | 구현·준비와 실제 실행 상태 |
|---|---|
| 저장소·checker/resolver·queue·stack·Warm·V4 | 소스 연결·fixture를 제공한다. 현재 source의 로컬 CPU 검사와 CI 성공은 아직 확인하지 않았다. |
| 현재 Rust CPU32 | Legacy 6→Both 10→Input 10→Head 6의 32사례를 준비했다. 새 binary·compiled Rules·자산 핀 등록과 실제 입력/출력 수치 인수는 pending이다. |
| 실제 GPU 수명·수치 | held pause→accepted Repair→accepted C tail→같은 owner resume, 동일 actual input·accepted seed의 독립 reference, final join/drain·buffer 0/unknown 인수는 미실행이다. |
| 한 변수 비용 | optional common6 비용 모드는 준비 중·미실행이다. 모델/queue/stack/Warm·GQA 반복·Signed192와 load/ready/transfer/cache/drain/peak를 분리한다. 모델·탐색에 전역 5% gate를 추가하지 않는다. |
| V4 paired pilot | 같은 own checker·ModelWdlRestricted, 선택 기능 off, 120+1·최대256 ply·흑백 교환 1 pair/2 games·900+30초 준비다. 각 판의 Legacy/new 두 엔진을 재시작하므로 4 PALS sessions 각각의 join/drain과 PGN 2판을 인수한다. 자산 핀은 미등록이며 실제 대국·종료 인수는 미실행이다. |

직전 통합 소스의 stack 9개·CPU lifecycle 5개 검사와 과거 CI/GPU 자료는 당시 소스·
설정의 증거로 보존한다. 현재의 compiled 선언·준비나 이전 결과로 위 pending 항목의
실행 성공을 주장하지 않으며, 후일 실제 결과는 별도 인수 기록으로 갱신한다.

새 seed 23의 네 profile은 동일한 공통 FP32 tensor 90개를 여섯 profile 쌍에서 bit 단위로
대조했다. 독립 Torch/CPU ORT shared 수치 24사례와 동일 seed의 Warm CPU 수치 6사례가
통과했으며, shared raw 출력의 최대 절대 차이는 `2.026558e-6`이다. 한 실제 CPU 창의
벽시계는 75.464초였고 optimizer·GPU 실행은 0회다. 새 자산은 미학습 baseline 후보이며,
제품 등록·Rust 입력/출력·CUDA 수치와 실제 대국 인수 전에는 arena eligible이 아니다.
실행 중 관측한 cgroup peak는 1,390,157,824 bytes지만 종료 뒤 scope가 사라져 최종 peak와
events는 미관측이다. 이 표본을 최종 peak 또는 메모리 개선률로 사용하지 않는다.

## 2026-10-10 현재 소스 인수 상태

이 절부터의 V3·선택16행·학습 제외 단계 표와 실행 수치는 이전 목표의 역사 기록이다.
당시 소스 인수와 실패·unknown을 보존하며, 위 PR23 후속의 현재 상태와 구분한다.

실제 optimizer 학습을 제외한 선택 구현과 실행 인수를 마쳤다. 기존 제품 실행의 소스는
`935519d8e2cf3ddb3fe25d52da8497700d7cf1a5`, 마지막 consumer 수정 소스는
`cbaea3b8e92669adcdb4acefd1c6a71fe4d83524`다. 두 소스의 실행을 구분하며, 이후 문서
커밋을 새 모델 실행으로 표시하지 않는다. 아래 날짜 이후의 상태가 이 문서 뒤에 보존한
과거 미완료·보류 기록을 보완한다. 실패 원문이나 당시 인수 범위는 수정하지 않는다.

[소스 CI 37973912091](https://github.com/daejunnom/RoveZero/actions/runs/37973912091)의
Linux·Windows·CPU bindings·PALS model 네 job이 모두 성공했다. 로컬 build536은 fmt,
독립 witness 4개, pair example 7개, workspace 전체 feature/target Clippy, release build를
통과했다. 등록 consumer `00884ce715e4353ee6c29f03fd0f2750ff887ae18cf403c1b29cb2c9c3933117`로
Repair 이후 Query 전이537과 동일 process의 두 action538을 각각 실제 실행했다.
최종 문서 tip의 검사·CI와 source 재사용 확인은 PR과 외부 인수 영수증에 별도로 기록한다.
상세 계수와 한계는 [현재 실행 인수 기록](research/PALS-IMPLEMENTATION.md#2026-10-10-현재-소스-인수-상태)에 둔다.

선택16행은 두 원문을 대조한 구현 범위이며 원문의 번호 목록을 대체하지 않는다.
학습 제외는 데이터 계약·strict replay·분할/누출 검사·zero-step checkpoint/resume와
V-free export 준비를 제외한다는 뜻이 아니다.

| ID | 선택된 원문 요구 | 완료한 구현·인수 범위 |
|---|---|---|
| P0-01 | Rules·typed authority·제품 탐색 교체 | CPU CI, Rules/제품 경계, R6 UCI 기능 인수 |
| P0-02 | V3 manifest/lock/receipt·V1/V2 보존 | 등록 build492, locked pair/Core·기존 codec 보존 |
| P1-01 | 자체 PVS·qsearch·TT·취소·조건부 resume | CPU CI, 자체 CPU_R 대국, R2 네 stage의 독립 재실행 |
| P1-02 | SEE ordering·profile/value/resume namespace | 명시적 선택·기본값 보존 및 조건/namespace 검사; 기력 효과 미주장 |
| P2-01 | 제한 후보 best-first·합법 prefix·quiet 응수·Repair | R5 수집503, GPU 정상 round508, R2 완료 Repair537/538 |
| P2-02 | 불변 관측·판정 revision·착수 초점·표현/사실 수명 | current-view, reset/drain, 과거 controls의 영향 소스 재사용 및 R6 종료 |
| P2-03 | Repair 뒤 동일 첫 수·actual C continuation | R5 동일 continuation의 3Reply 두 사례와 독립 R2 endpoint 재실행 |
| P3-01 | 384/GQA6:2/latent16/reader2×2·private expert·V-free | 모델 CPU CI, 독립 수치 기록, 실제 CPU NN와 GPU P/C 정상 실행 |
| P3-02 | missing subset/device join·pin/evict·예약·물리 수명 | numeric/resident 인수, GPU508 정상 종료·과거 controls 재사용; fault 미실행 범위 유지 |
| P4-01 | 당시 입력/후속 label·causal current-view·split/leak | strict frozen loader 인수, 89개 holdout/value masked, 미관측 WDL target 미부여 |
| P4-02 | 후보 비교·divergence/whole-line 감독·unknown mask | 소스·CPU 검사와 masked 데이터 소비; 관측 없는 정답 미부여 |
| P4-03 | finite zero-step recipe·RNG/sampler·checkpoint/resume·export | 모델 준비·재현 검사; 실제 optimizer 학습 0 |
| P5-01 | V-private Query·CPU action·환류·episode/prior ledger | ONE Query/two-action 준비531, 실제 next Query·ledger537/538, 다음 선택 Defer |
| P5-02 | 원래 전체 causal 비용·동일 문제/witness utility·private/public | 실제 whole-cost537/538, 조건부 비교의 음성 결과와 null/masked 보존 |
| P6-01 | 같은 시작·흑백 교환·동일 clock/resource·process restart | R6 두 판/99ply·120+1·흑백 교환·Rules/시계/PGN 기능 인수 |
| P6-02 | 실패·NN 완료/소비 구분·Core/final bytes·EOF/reap/cleanup | R6 네 failed-go 0, NN4620/role소비2310, 두 PALS drain·group/unit 종료와 회수 |

537의 비용 범위는 원래 S부터 다음 선택 완료까지 1.539106541초다. CPU556 nodes·NN56은
source 보고278/28과 독립 재실행278/28을 합친 명시적 범위다. source child의 물리 작업
attestation은 false, source scalar 일치는 unknown이다. 독립 캡처14개/NN28의 완료와
shutdown은 확인했다. 다음 Query·prior ledger는 실제 새 소유자를 통해 발급했으며 JSON을
live capability로 복원하지 않았다. 다음 선택 Defer를 neural V 선택이나 전략적 증명으로
보고하지 않는다.

538은 같은 사전 Query의 N4096/N2048 action을 독립 소유자와 원래 시계로 실행했다.
각 CPU556/NN56, 전체 비용1.605115383초/1.325331493초를 기록했다. 루트 백 관점에서
repair0/counter+16이므로 필요한 조건부 counter-lower가 성립하지 않았다. 시간 차이로
이를 덮지 않고 preference null, known-label mask false, utility/target/training 권한 false를
유지했다. 이는 실제 음성 결과 처리 인수이며 positive utility나 속도 개선의 증거가 아니다.

Frozen 자산은 미학습 모델이다. 실제 학습·강도·Elo·성능 개선을 주장하지 않는다. R6의
PALS0/2는 기능 pilot 결과다. GPU508에서 Repair NN은0이며, 개별 GPU late-completion
trace·full native quarantine fault는 미실행, VRAM peak·native parameter storage의 물리
공유와 source child scalar 일치는 unknown이다. 선택적 CUDA Warm·추가 pruning·대체
optimizer와 실제 학습은 후속 범위다. 5% 의미 보존 회귀 문턱을 PALS 도입 조건으로 쓰지 않는다.

과거 실패487/496/499/504/509 및 R2 516/524~528/532, 검사534/535의 원문을 보존했다.
진단 후 선택한 새 b2b4/b8a6 사례는 기존 result-free 등록517/522와 구분한다. 532에서 발견한
full OWNED CPU 조건과 bare search_conditions 대조 오류는 consumer만 수정했다. producer
engine/sourceExpected7·모델·원래 Query·예산은 유지했다. 935→cbaea3b의 변경은 pair example,
추가 descriptor 메서드, 독립 witness 세 파일이며 기존 제품 실행 경로의 결과를 새 GPU
실행으로 표시하지 않는다. cleanup529는 비활성 private 사본을 Linux/Windows에서 합
245,219,172bytes 논리 반환하고 공유 자산·PGN·로그·실패 자료를 보존했다. VHD 물리 축소량은 아니다.

## 실행 경계

UCI 아래에서 PUCT, PALS, 자체 CPU 탐색을 선택한다. PALS는 P의 수순 제안,
C의 이탈·반박, 자체 CPU의 조건부 평가, P의 수선과 관련 결론 갱신을 수행한다.
합법 수·정확한 이력·종료 판정은 기존 Rules가 소유한다. PALS 작업 수나 CPU 노드를
PUCT simulation·visit 수로 표시하지 않는다.

학습 설계의 P/C/V와 대국 제품의 P/C를 구분한다. 제품 export와 실행에는 V를
포함하지 않는다. 자체 CPU_T와 CPU_R는 하나의 Rust PVS 코어를 공유하면서
프로필·평가 버전·노드·깊이·시간·TT 예산을 별도로 식별한다. 외부 UCI 엔진은
상대 및 독립 비교 경로다. 초기 학습 설계에는 외부 교사·기존 가중치가 포함되지 않는다.

정확한 CPU 결과·관측 기록과 GPU 표현 캐시를 분리한다. GPU 캐시를 회수해도 완료된
CPU 문제 결과는 남는다. 취소·세대 변경 이후 늦은 결과는 현재 루트에 적용하지 않는다.
물리 완료가 불명확하거나 격리된 실행은 입력·출력·workspace를 보존하며 재사용하지 않는다.

### 첫 value resolver와 교체 범위

첫 정책은 `pals-cpu-raw-restricted/0.1`이다. 등록된 CPU value namespace의 수용된
raw scalar와 실제 Rules terminal을 사용하고, 조사한 자식의 차례 관점 값을 부호 전환해
max를 계산한다. 수용된 frontier·부분 완료 값도 해당 범위와 함께 사용할 수 있으므로
요청 깊이 완료와 같은 뜻으로 기록하지 않는다. 미관측 값은 `None`이며 0·무승부로 채우지
않는다. CP/WDL 보정이나 P/C 값과의 평균은 하지 않는다. 실제 Rules 승리 착수를 우선하고,
같은 값에서는 terminal, 나머지 동점에서는 Rules 순서를 유지한다. 제한 후보의 추정값은
전체 게임의 bound 또는 완결 증명이 아니다.

`PalsResult`와 collector 원시 결과에 버전을 기록하고, UCI process 작업 영수증에는 버전과
의미 정의의 SHA-256을 추가한다. 시작 identity와 반환 결과의 정책이 다르면 거부한다.
과거 영수증의 필드 부재는 그대로 읽으며 해당 정책을 관측한 것으로 자동 채우지 않는다.
이 식별 추가는 현재 착수·점수 정책을 바꾸지 않는다. 후속 resolver 변경은 S 실험이다.

외부 UCI 상대 연결과 PALS 내부의 외부 `CPU_R` checker는 별도 접점이다. 첫 제품은 자체
`CpuSearcher`를 사용하며, 외부 checker의 capability·관점·raw mate/bound·단일 활성 go·
미지원 resume은 별도 인수한다. 현재 host whole-input K/V page의 실제 backing 소유권은
GPU resident page·record별 증분 재사용 또는 private latent warm-start의 인수를 대신하지 않는다.

## PALS V3 명세·lock·receipt

`rz-experiments::PalsRunManifestV3`는 `rz-pals-execution-v3/1` 도메인에서 JSON을
검증한다. `PalsInputLockV3`는 도메인과 canonical SHA-256을 고정한다. 기존 V1/V2의
codec·canonicalization·digest를 자동 변환하거나 PALS 실행으로 다시 해석하지 않는다.
PALS 계약 식별은 `pals/0.1`이며 기존 policy/WDL 평가 계약 revision `0.1`과 별개다.

엔진 endpoint는 PALS, 자체 CPU, 외부 UCI 비교 대상으로 나눈다. PALS 모델은 입력·head·
구조·구현·정밀도·backend·batch·고정 epoch와 가중치 출처를 기록한다. 결정적 mock,
미학습 초기 파라미터, 기존 학습 체크포인트는 서로 다른 타입이다. 파일 메타데이터 검증과
실제 파일의 hash 확인·모델 로딩·provider readiness는 다른 증거다.

생성 PGN의 native 참조는 실행 설명을 보존한다. Core의 공통 `ArtifactRef`에 연결할
때에는 `source`만 잠긴 Fastchess producer의 공개 HTTPS 소스 URL로 투영하고,
path·SHA-256·bytes·license는 유지한다. `pgn_provenance`에 원래 native 참조와
producer 소스 commit·바이너리 SHA를 따로 남긴다. URL은 로컬 PGN의 공개 다운로드
위치를 뜻하지 않으며, 공통 URL 검증·실패·시계·물리 수명·인수 조건을 완화하지 않는다.

| 비교 질문 | 고정·변경 조건 |
|---|---|
| `System` | PALS 전체 구성과 자체 CPU 또는 BT4/LC0 구성의 효과. 모델·탐색 전체가 달라지는 비교를 단일 축 효과로 보고하지 않는다. |
| `InternalModel` | PALS 모델 구성만 변경. 자체 CPU·탐색·runtime·자원은 동일하다. |
| `InternalSearch` | PALS 탐색만 변경. 모델·자체 CPU·runtime·자원은 동일하다. |
| `Runtime` | PALS runtime·pool 구성만 변경. 모델·탐색·자체 CPU·자원은 동일하다. |

명세가 선언한 변경 축과 두 endpoint의 실제 구성 차이가 일치해야 한다. 소스 SHA와
바이너리 hash는 실행별로 기록하되, 그것만으로 구현 의미의 동등성을 주장하지 않는다.
바이너리 내용이 바뀌었는데 어떤 구성의 구현 식별도 바뀌지 않은 경우는 `Endpoint`
변경으로 남겨 단일 축 비교를 거부한다. 같은 바이너리의 저장 경로 차이는 변경 축으로
세지 않는다. 이러한 선언 대조는 실제 소스 diff와 빌드·파일 hash 인수를 대신하지 않는다.
PALS 도입은 모델·탐색 실험이므로 5% 의미 보존 회귀 문턱을 도입 통과 조건으로 사용하지 않는다.

명세와 결과 영수증은 이번 범위에서 `training_executed=false`를 요구한다. 기존 학습 자산을
참조할 수 있는 schema와 실제 이번 실행에서 학습을 수행한 기록을 구분한다. CPU/mock CI
성공은 실신경망 수치 검사·GPU 실행·학습 성공으로 바꾸지 않는다.

## 유한 실행과 관측

첫 대국 명세는 표준 시작 상태, 흑백 교환 두 판, 120초+1초 피셔, 최대 256 ply,
동시 대국 1개, 게임마다 프로세스 재시작, 전체 최대 15분과 정리 최대 30초다.
Ponder·평가값에 의한 조기 판정·두 판으로부터의 Elo 주장을 허용하지 않는다.
CPU affinity·thread 수, memory.high/max, swap, 요청 GPU와 device 할당 상한을 선언한다.
상태·수순 chunk·상황·관측·작업·역할 상태·메모리 page·큐의 최대 개수도 선언한다.

`PalsRunReceiptV3`는 요청 옵션, 광고된 지원 여부, 실제 관측 값을 구분한다. `readyok`는
프로토콜 준비 응답이며 옵션 적용 값을 관측했다는 증거가 아니다. 미관측 값과 VRAM peak는
`unknown`으로 남긴다. 물리 NN 완료 입력 수·실행 수·실제 소비 입력 수·cache 소비 수와
P/C 작업·CPU 요청/완료/소비/노드 수를 구별한다. 외부 엔진의 내부 NN 실행을 RoveZero
계측값으로 꾸며 기록하지 않는다.
확인된 CPU affinity 불일치·메모리 한도 초과는 `ResourceAdmission` 실패로 보존한다.
UCI 옵션 이름의 대소문자 중복과 Ponder·Chess960·thread 한도·typed 모델 선택을
덮어쓰는 옵션을 거부한다. GPU 요청 모델명과 실제 UUID는 같은 표현이라고 가정하지 않는다.

CUDA 제품 시작은 runtime·모델·session의 cold 준비와, 준비된 backend에서 수행하는
P/C·제어 ACK probe를 분리한다. `--pals-startup-probe-timeout-ms`는 후자의 명시적
예산이며 범위는 1~180,000ms다. 생략하면 기존 15,000ms와 기존 JSON 생략 형식을
유지한다. V3 CUDA launch의 선택값을 시작·종료 execution에서 정확히 대조하고,
외부 `handshake_max_ms`는 cold 준비와 프로토콜을 포함한다. CPU CLI에는 이 CUDA
옵션을 허용하지 않는다. probe 예산이 handshake 이하여도 실제 cold 준비 여유가
보장되는 것은 아니며, 전체 pair 창·물리 drain·quarantine 조건은 늘리지 않는다.

시작 실패는 성공 readiness와 별도 게시 경로로 보존한다. 없는 mapping·placement
ACK는 없는 상태로 남기고, 관측된 ACK는 실제 소유 identity와 대조한다. 원래 오류·
부분 계수·논리 마감·미확정 물리 fence를 저장하며 실패한 시작을 성공 종료로 바꾸지
않는다. 진단은 최대 7개 제어 명령과 2개 역할 요청·64개 backend 단계만 기록한다.
lock 경합·poison·overflow는 진단 부재로 표시하며 실행 결과나 buffer 수명을 바꾸지
않는다. 명시 예산의 정상 실행에서는 실제 factory·probe 시간을 별도로 남기고,
일반 `go`에서는 startup 단계 기록을 닫는다. 단계 반환 사건 자체는 NN 완료 계수나
독립적인 물리 완료 증거가 아니다.

각 판의 백·흑 엔진, 실제 시계, 결과·종료 이유와 PGN hash를 보존한다. 엔진 크래시·불법 수·
시간패는 해당 엔진의 패배로 남긴다. 인프라 실패·실행 시간 한도·최대 ply 도달은 incomplete로
남기며 자동 무승부나 유효 대국으로 만들지 않는다. 불완전 영수증도 검증 가능한 증거로
보존하지만 `pair_eligible=true`를 붙이지 않는다.

## 단계별 인수

| 단계 | 구현·검증 대상 | 실제 학습과의 관계 |
|---|---|---|
| P0 | 공통 PALS 경계, search session 교체, 명세·lock·receipt, V1/V2 소비자 호환 | 학습 없음 |
| P1 | 자체 PVS CPU, 정확한 종료·TT·중단·조건부 resume, 독립 CPU UCI | 초기 평가와 학습 평가를 구별 |
| P2 | CPU/mock PALS 제안·반박·수선·결론 갱신, 유한 저장소·세대·수명 | mock 결과를 신경망 기력으로 해석하지 않음 |
| P3 | 384 폭·16 latent·고정 2회 반복 모델의 Rust 입력·실행·출력, export·독립 수치 대조 | 초기 파라미터 실행과 학습 완료를 구별 |
| P4 | 자체 데이터 계약, 분할·누출·unknown label, 비용·checkpoint·resume·export 도구 | 실제 optimizer 학습 실행은 이번 목표에서 제외 |
| P5 | V 학습용 작업·보상 계약, private 정보 분리, V-free export 검사 | 실제 V 학습은 실행하지 않음 |
| P6 | 자체 CPU 상대 pilot, 전체 시스템과 단일 축 효과 기록, 시계·PGN·종료 인수 | 강도 향상은 관측 결과로 판정 |

이 표는 완료 선언이 아니다. 총괄은 실제 구현 소스·같은 integration SHA의 CPU 검사,
신경망 수치 검사·GPU 물리 수명·대국 자료를 확인한 뒤 단계 상태를 갱신한다.
Colab의 학습 실행과 비용 등록은 별도 후속 실행이다. 원시 데이터·모델·PGN·로그·
보고서는 기존 저장소 관리 규약에 따라 저장소 밖에 두고, 이번 변경에는 소스와
검토 가능한 작은 독립 검사만 포함한다.

### 전체 원문과 최소 pilot의 구분

P6의 양성 자체 CPU pair만으로 두 원문의 전체 구현 완료를 선언하지 않는다.
현재 선택된 실행 흐름의 잔여와 원문의 선택적 후보를 다음처럼 구별한다.

| 항목 | 구현·인수 경계 |
|---|---|
| CPU_T SEE 정렬 | `LegalSeeV1` Rust ordering과 private CPU_T의 명시적 `--cpu-ordering=legal-see-v1`·봉인된 `ordering_policy: legal_see_v1`, producer의 `--ordering-policy legal_see_v1`까지 연결돼 있다. legacy 기본값·식별은 유지하며 SEE는 전략적 증명이나 영구 pruning이 아니다. 등록 기준 `b6de532`의 Linux·Windows CI에서 실제 선택 engine·조건·report·독립 resume namespace를 검사했다. 새 실제 producer 실행과 기력 효과는 별도 인수한다. |
| V 결과 환류 | bounded coverage loop와 native Repair 보고의 단일 prior 자료 발급 접점이 있다. capture의 loaded binary·원래 S/E/W·Rules·raw body·부모 transport 비용을 검사하지만, 자료 발급만으로 actual native witness·다음 Query admission·episode ledger·whole-action utility를 발급하지 않는다. coverage 증가는 전략적 유용성이나 학습 target의 증거가 아니다. |
| GPU record 공유 메모리 | host whole-input page와 record별 CUDA resident packing을 구분한다. 명시적 `registered-packing-v1` lane의 missing subset 인코딩·device join·불변 read view·물리 pin·예약·격리·CLI/arena 연결 소스가 있다. `34149e2`의 실제 CUDA 독립 수치·resident anchor/derived/repeat·새 게임 초기화·정상 물리 종료는 아래 실행 범위에서 인수했다. 제품 UCI의 stop·늦은 root·미확정 완료 경로, VRAM peak와 개선 효과는 별도로 남긴다. 전체 입력 cache 성공을 record별 GPU 공유의 증거로 쓰지 않는다. |
| Repair 이후 C 재검토 | 같은 line을 한 번 재검토하는 기존 lane과 실제 새 C continuation의 별도 S lane·다중 Reply 관측 접점이 있다. 원래 첫 수·Repair line/revision·첫 ongoing 상대 anchor·실제 dispatch/소비를 대조한다. 새 source의 실제 다중 Reply 수집 인수와 전략적 강도 효과는 별도로 남기며 다음 round의 root Propose로 대신하지 않는다. |
| 실제 CPU pair | 해당 integration source/binary와 시계·NN 소비·typed failed-go 0·Rules·PGN·process·저장·cleanup을 모두 인수한다. preparation 자기 보고나 CPU CI로 대체하지 않는다. |
| 선택적 후속 기능 | CUDA private Warm, paused search stack, set-associative TT, fast pruning과 대체 optimizer는 원문의 후보·미결정 범위를 보존한다. 기본 Fresh 구현의 필수 통과 조건으로 임의 확대하지 않는다. |

실제 학습은 이번 목표에서 제외한다. 2026-10-09 재개에서는 총괄과 서브에이전트가
소유 경계를 나눠 구현하며, GPU를 포함한 최종 실행은 총괄이 등록된 소스·바이너리·
자산·환경과 유한 예산을 확인한 뒤 인수한다. 과거 GPU 보류·실패 자료는 그대로 보존한다.
구현·CPU CI·실신경망·실제 대국·GPU 인수와 미지원 옵션을 각각 기록하며 미실행을
성공 또는 skip 인수로 채우지 않는다.

### 실제 GPU 수치·resident page 인수 범위

2026-10-09의 `gpu-numeric-resident-20261009-01`은 source
`34149e2cc6e588cf32452aece0494aeb8cb05134`와 release `pals_model_check`
바이너리 SHA-256 `46c8eb1b522c45f1e0e46f34ba88d98c4c9898c31afee503b4c4759477c8bd1c`에
고정한 실제 WSL CUDA 검사다. FP32·TF32 off, 같은 미학습 P/C export와 독립 reference를
사용했다. `observed-summary.json` 및 pin이 일치하는 `numeric.json`·`execution.json`·
`launch.json`·`launcher-execution.json`을 직접 대조했다. 상세 근거와 한계는
[실제 GPU 수치·resident 인수 기록](research/PALS-IMPLEMENTATION.md#실제-gpu-수치resident-인수와-r7-잔여)에 둔다.

독립 numeric 6개, resident anchor 6개와 derived 15개·repeat 15개를 수행했다.
resident backend의 완료 입력은 public 10·private 37, 총 47이다. 별도 whole comparator
30과 기존 main numeric backend 20은 다른 계수 범위이므로 47에 더해 제품 처리량으로
표시하지 않는다. resident 대조의 최대 절대 차이는 후보 logit `2.682209e-7`, WDL
`5.960464e-8`, latent `1.430511e-6`이며 기존 허용 오차를 만족했다. 새 게임 뒤 bank의
live block·certified projection·host/device 소유 bytes가 0이고 generation은 2였다.
worker ACK와 정상 물리 shutdown, 실제 exit 0·reap·group 종료 및 unit inactive를 확인했다.

외부 벽시계는 43.172초, checker 실행은 38.096초다. 적용 affinity는 `0 2`,
memory.high/max는 6GiB/12GiB, swap은 0이며 새 cgroup `memory.peak`는
3,644,235,776 bytes, `memory.events`의 OOM 계수는 모두 0이었다. 이 peak는 VRAM
또는 Windows 전체 커밋 peak가 아니다. VRAM peak, 네이티브 파라미터 저장의 실제 공유,
device K/V를 host로 복사한 tensor 대조는 `unknown` 또는 미실행으로 남긴다.

이 인수는 독립 checker와 물리 worker의 정상 경로에 한정된다. 제품 UCI의 stop·cancel·
늦은 결과·새 root 및 미확정 물리 완료 인수, 실제 CPU 4N·Query·전체 action utility·
다중 Reply·paired 대국은 완료하지 않았다. 실제 학습·기력 또는 성능 개선을 주장하지
않으며, 후속 source의 CPU CI나 GPU 인수로 이 자료를 자동 승격하지 않는다.

### 기록 입력 witness와 새 실행의 구분

새 `captured_nn_input_reinference_and_independent_cpu_condition_reexecution` 경로는
source owner가 보존한 P/C 입력을 그대로 재추론하고, 같은 조건의 CPU 문제를 독립
Rust CPU로 다시 실행하는 경계다. `query[7]`의 deadline feature는 **과거 입력값**으로
bit와 입력 식별을 유지한다. 이를 현재의 남은 시간으로 갱신하거나 새 실행 시계로
해석하지 않는다. 새 작업의 취소·완료와 비용은 별도 실제 관측이며 원래 S/E/W 안에서
검사한다. 기록 입력을 다시 계산한 사실은 과거 child의 물리 실행을 인증하지 않는다.

원래 child의 CPU scalar score는 구조화된 원문이 없어 동등성이
`unknown_not_structurally_reported`다. 새 독립 CPU의 score·조건·종료·노드와 새 NN
출력·완료는 별도 사실로 남긴다. 과거 score를 파싱하거나 새 score로 덮어쓰지 않으며,
같은 입력의 NN raw bit 대조와 CPU score 동등성을 혼동하지 않는다. 이 source 접점의
존재는 실제 R2 실행·R3 Query admission·R4 utility·R5 수집·R6 대국의 완료 증거가 아니다.

### 현재 소스의 최종 인수 순서

재개 감사 기준은 `b6de5325d47a417461823f4a92329287c0e5af09`이며 그 SHA의 CPU CI
성공을 후속 수정이나 실제 frozen NN 실행의 성공으로 소급하지 않는다. 아래 목록은
학습 제외 목표의 잔여 인수이며 구현 담당의 개별 완료와 최종 통합 인수를 구분한다.

1. 중단된 CPU 실행의 실제 owner·lease·scratch 상태를 확인하고 원래 실패 자료를
   보존한다. 최종 source·feature·바이너리와 자산을 새 실행에 등록한다.
2. 실제 registered frozen 4N CPU caller와 독립 native witness를 인수한다. 원래
   S/E/W, 입력 전 loaded binary 확인, raw·Rules·물리 완료·정리를 각각 확인한다.
3. 실제 결과를 다음 immutable Query에 소비하는 episode·prior ledger와 전체 action
   비용·utility 경계를 연결한다. 중복·순서 역전·늦은 결과·취소를 거부하고 unknown은
   masked로 보존한다. legacy receipt를 만들어 native 관측을 대신하지 않는다.
4. 같은 통합 소스에서 actual 다중 Reply의 tensor·dispatch·소비·Repair anchor/revision,
   분할·누출·영수증·저장을 인수한다. 위 `34149e2` GPU 수치·resident 정상 종료 인수와
   별개로 제품 GPU stop·늦은 root·미확정 완료 경로를 확인한다.
5. 같은 최종 source/binary의 자체 CPU 상대 한 쌍을 기존 120초+1초·최대 256 ply·
   전체 15분+정리 30초로 실행한다. 두 판의 백/흑·시계·NN 소비·typed failed-go·Rules·
   PGN·EOF/reap/group 종료·필수 최종 저장을 확인한다. 과거 source에 고정된 준비
   명세·binary를 새 source의 인수로 사용하지 않는다.
6. 원문의 선택된 필수 요구사항을 source·CPU 검사·실제 CPU·GPU·paired 자료에
   대조하고 정확한 최종 SHA의 CI와 PR 인수 기록을 갱신한다. 미지원 선택적 연구,
   실제 학습 제외와 관측되지 않은 값은 완료 범위와 함께 명시한다.

R7의 현재 잔여는 실제 R2의 기록 입력·독립 CPU/NN witness, R3의 다음 Query/episode
ledger, R4의 전체 action 비용·utility, R5의 다중 Reply 수집·consumer, R6의 current-source
자체 CPU_R 두 판과 process/storage 인수, 제품 GPU stop/late-root 인수 및 최종 SHA CI다.
완료한 독립 GPU 검사와 소스·CPU fixture를 이 잔여 실행의 대체 증거로 사용하지 않는다.
