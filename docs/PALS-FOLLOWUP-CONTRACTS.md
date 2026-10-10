# PR #23 후속 구현 계약

사용자 승인한 후속 계획의 연결 계약이다. 기존 비학습 인수와 V1/V2/V3 자료를
보존하며, 이 문서의 게시만으로 아래 기능의 구현·실행·인수를 주장하지 않는다.
공통 `EvalRequest/EvalOutput` revision 0.1은 유지한다.

이번 범위는 diagnostic nonzero 학습 smoke까지이며 본격 학습·learned V 인수·PR 병합은
제외한다. 이전 비학습 목표와 실행 자료는 [구현 인수 이력](PALS-IMPLEMENTATION.md)에
보존한다. 현재 연결·검사·미실행 구분은
[이번 후속 범위](PALS-IMPLEMENTATION.md#2026-10-10-pr23-후속-범위)를 따른다.
중탐색 controlled archive는 additive 구현·소비자 연결 중이며, 이 계약을 구현 완료나
새 source의 실행 성공으로 표시하지 않는다.

## 소유 경계와 구현 순서

- 총괄: 계약, 탐색의 checker/resolver·반복 queue, UCI 연결, 통합 검사·PR 인수.
- 저장소: `rz-search/pals/store`의 bounded hot/cold·pin·archive와 Rules snapshot.
- CPU: `rz-search/cpu`의 별도 PausedStack 정책; 기존 재귀 탐색·iteration resume 보존.
- Rust 모델: `rz-eval/pals_model`·ONNX 입출력의 V2 profile·입력·식별.
- 모델 도구: PALS Python config/model/export/reference의 V2 profile.
- 학습 도구: 별도 nonzero checkpoint·finite update·target coverage.
- 실행 명세: PALS V4 manifest/lock/receipt와 기존 codec 호환성.
- CUDA Warm: 별도 device capability와 물리 lease; host Warm 의미 보존.

같은 파일의 편집은 단일 담당이 수행한다. 실제 연결은 총괄이 같은 integration
SHA에서 확인한다. CPU/mock, 실제 NN, GPU 수명, 학습 smoke, 대국 인수를 구분한다.

## 이번 후속의 기본값과 제외 범위

| 접점 | 기본값·명시 선택 | 호환·인수 경계 |
|---|---|---|
| 새 모델 init·제품 profile 생략 | `full_line_interaction_v2` | `legacy_summary_v1`을 명시한 legacy 비교와 기존 `baseline()`·생략 serialization의 의미를 보존한다. |
| 입력/head 제거 대조 | input-only=`full_line_v2`, head-only=`interaction_head_v2`, 둘 모두=`full_line_interaction_v2` | 모델 효과를 각각 분리하며 과거 weights를 새 profile로 재해석하지 않는다. |
| 새 후속 명세·pilot 준비 | `PalsRunManifestV4`, schema 4·`pals/0.2`와 V4 lock | 준비 경로가 `--pals-arena-wire=v4`를 명시 전달한다. 일반 CLI의 wire 생략은 기존 호환이며 V1/V2/V3 codec·digest는 유지한다. |
| checker·resolver | CLI resolver 생략은 own=`OwnRawRestricted`, 외부=`ModelWdlRestricted` | checker와 resolver는 독립 선택이다. 새 모델/explicit legacy의 V4 비교는 같은 own checker·`ModelWdlRestricted`를 명시 고정한다. |
| Repair 후 재검토 | `PostRepairRecheckPolicy::Disabled` | frozen Fresh나 iterative queue는 명시 선택이다. queue는 iterative recheck 정책 안에 포함하며 기본 활성로 해석하지 않는다. |
| CPU resume | `CompletedIteration` | 실제 `PausedStack`은 own CPU의 opt-in이다. 외부 checker에서는 거부한다. |
| host/CUDA Warm·pilot | 둘 다 off, 새 기본 pilot의 archive/queue/stack/Warm/resident packing도 off | host Warm과 CUDA ApproxWarm은 별도 capability다. 새 baseline·실제 GPU·pilot 인수는 pending이다. |

설정·제품 선택은 [model config](../experiments/model-research/pals/src/rz_pals_model/config.py),
[UCI main](../crates/rz-uci/src/main.rs)과 [driver](../crates/rz-uci/src/search_driver.rs),
V4 등록·실제 child argv는 [manifest](../crates/rz-experiments/src/pals_manifest_v4.rs),
[pair 준비](../crates/rz-arena/examples/pals_pair_prepare_v4.rs)와
[V4 launch](../crates/rz-arena/src/pals_launch/v4.rs)에 연결한다.

## 모델 V2

`PalsModelProfile`/Python profile 이름은 `legacy_summary_v1`, `full_line_v2`,
`interaction_head_v2`, `full_line_interaction_v2`다. 새 init의 기본은 마지막 profile이며,
기존 `baseline()`과 V1 JSON/export의 의미는 legacy로 유지한다.

- `PalsModelInput`의 legacy byte 형식을 유지한다. V2에는 명시적 full-line payload를
  추가한다. record마다 순서가 있는 candidate 형식의 move token(from, to, promotion),
  query의 prefix/proposal/counter 세 수열, record의 로컬 parent/supersedes 참조를 둔다.
- record 최대 128개, 수열마다 256 ply다. 빈 수열은 mask로 표현한다. 부모/대체
  참조는 현재 선택된 record 집합의 인덱스이며 임의 global ID를 의미 특징으로 쓰지 않는다.
  누락된 필수 관계·비유한 값·부정확한 길이·지원하지 않는 profile을 거부한다.
- 수열 encoder는 64폭 embedding, kernel 3·dilation 1/2의 두 temporal block,
  masked attention pooling이다. 후보 head는 candidate/context 결합의 128폭 SiLU MLP다.
- 공개 record 표현/KV는 독립 수열에서 만들고 관계는 private role graph에서 소비한다.
  exact input·record cache는 전체 수열/profile을 포함하고, 관계·순서·epoch를 생략하지 않는다.
- 모델 schema/encoding V2, PALS 실행 V4를 별도 등록한다. legacy codec/digest를 바꾸거나
  과거 자산을 재해석하지 않는다. Torch reference와 Rust/ORT 입력·수치를 대조한다.

소스 접점은 [Rust 모델](../crates/rz-eval/src/pals_model.rs),
[Python config](../experiments/model-research/pals/src/rz_pals_model/config.py),
[model](../experiments/model-research/pals/src/rz_pals_model/model.py)과
[export/artifact](../experiments/model-research/pals/src/rz_pals_model/artifacts.py)다.

## 탐색·저장소·재개

- `ResolverPolicy`: own raw는 own checker만, model WDL은 own/외부 checker 모두 지원.
  새 생성자는 명시적으로 선택한다. 기존 생성자는 기존 조합을 보존한다.
- 새 재검토는 frozen 모델의 Fresh WDL을 양 끝점의 동일 관점에서 비교한다. 두 Rules
  terminal은 Rules로 비교하고, mixed/unknown은 unresolved다. 외부 CP/mate를 own raw나
  Rules 사실로 바꾸지 않는다. 추가 NN 비용은 원래 전역 예산에 포함한다.
- 새 raw WDL에는 실제 공개 문맥 revision을 `ContextWdl`로 보존한다. 조건부 반박은
  같은 모델·정밀도·revision의 두 raw 관측을 참조한다. 반복 Repair는 실제 C endpoint와
  새 P endpoint를 다시 Fresh로 대조하고, 엄격한 개선이 있을 때만 현재 반박을
  `ConditionalRepairWdl`로 supersede하여 `RepairedBy`로 연결한다. Legacy WDL wire는 유지한다.
- 반복 queue는 별도 S 정책, first move별 Repair 3회·pending 64개다. 같은 질문/revision
  중복을 막고 새로운 증거가 없으면 종료한다. 원래 first move·근거·supersedes를 보존한다.
- hot 한도 80% 또는 Capacity에서 비활성 자료를 cold로 이동하고 allocation을 최대 1회
  재시도한다. root 전이와 중탐색 회수는 같은 원래 예산·보존 규칙을 따른다.
  root/frontier/진행 작업/paused CPU/관련 근거를 pin한다. 게임 archive 256MiB, 관리 전체
  4GiB 상한이다. archive commit·무결성 확인 전에 RAM을 해제하지 않는다.
- cold 핸들은 store 소유자/generation을 검증한다. 조회는 deadline/byte 예산이 명시된
  pin/load 접점이다. RAM 인덱스도 bounded다. quota/I/O/pin 포화는 명시 실패이며,
  보존 증거를 삭제해 성공으로 처리하지 않는다.
- `PausedStack`은 일회 opaque token, 한 실제 paused task·8MiB다. 미완료
  root/PVS/negamax/qsearch/SEE frame·cursor/PV·alpha-beta와 정확한
  Rules/accumulator·ordering·TT 소유권을 보존한다. consumed work를 다시 세지 않는다.
  다른 history/profile/root/namespace, 취소·새 게임에는 재사용하지 않는다.
  CPU admission마다 실제 owner와 token을 대조한다. stale paused 기록은 역사 증거로
  보존하고 새 작업으로 다시 시작한다. CPU 호출 밖의 논리 취소도 실제 stack을 폐기하며
  GPU lease의 물리 완료 규칙은 별도로 유지한다.
- CUDA ApproxWarm은 host Warm과 별도 capability다. accepted seed만 같은 role·모델·
  Rules/query 문맥에서 사용한다. 공개 record revision만 변경 가능하다. value/V는 Fresh다.
  논리 취소가 물리 완료를 대신하지 않는다. 완료 불명 owner/buffer는 격리·보존한다.

재검토의 실제 선택명은 다음처럼 계층별로 구분한다. 외부 CP/mate를 Raw WDL의 관점으로
직접 비교하거나 search policy identity를 CLI 값으로 전달하지 않는다.

| Rust 정책 | `--pals-post-repair-recheck` 값 | search policy identity | wire |
|---|---|---|---|
| `FrozenModelWdlV2` | `frozen-model-wdl-v2` | `post-repair-frozen-wdl-v2` | `frozen_model_wdl_v2` |
| `IterativeFrozenModelWdlV2` | `iterative-frozen-model-wdl-v2` | `post-repair-frozen-wdl-queue-v2` | `iterative_frozen_model_wdl_v2` |

[engine](../crates/rz-search/src/pals/engine.rs)와
[frozen recheck](../crates/rz-search/src/pals/engine/frozen_recheck.rs)가 실제 C continuation·
Fresh 끝점·관점/revision·원래 first move와 supersedes를 소유한다.
[store](../crates/rz-search/src/pals/store.rs)의 `ContextWdl` raw와 conditional paired WDL은
같은 모델 namespace·정밀도·문맥 revision 및 두 raw observation/dependency를 보존한다.
`LegacyWdl` wire·과거 CP 권한은 새 결론의 authority로 자동 전환하지 않는다.

### 중탐색 controlled archive 계약

현재 additive scope는 `PalsStores::with_archive_allocation(StorePins, &mut ArchiveIoBudget,
FnOnce)`와 allocating facade의 `*_controlled` 접점으로 준비·연결 중이다. operation 동안
기존 핀에 resident Node/RoleRecord·active-search 핀과 temporary 핀을 합치고, 동일한
원래 deadline·잔여 bytes만 설치한다. 성공·실패 모두 실제 남은 bytes를 돌려주고 scope
종료 시 allowance를 지운 뒤 기존 핀을 복원한다. nested scope·global allowance 덮어쓰기·
다른 budget I/O·active scope의 checked foreign 혼용은 거부한다. foreign checked 관측은
별도 명시 mutable budget을 사용한다.

새 intern state/situation과 Node/RoleRecord 설치 전의 raw line/observation은 임시 핀으로
보호한다. cold로 옮길 때도 exact history와 paired raw WDL/dependency closure를 보존한다.
controlled callback은 pressure/commit 뒤 allocation·재시도와 최종 logical acceptance 전에
취소·deadline을 다시 검사한다. `ArchiveCanceled`/`ArchiveDeadline` 뒤에는 재시도하지 않으며
실패한 edge admission은 새 conclusion revision을 게시하지 않는다. all-pinned는
`PinSaturated`, quota/I/O/byte-budget은 각 typed 오류로 끝낸다.

중탐색 회수에서 Node Vec를 compact하거나 usize를 재발급하지 않는다. 다음 root 준비는
현재 search의 임시 raw line/observation 핀을 정리하되 physical active/cancel-requested 작업과
paused CPU의 유효 owner·근거 closure를 계속 보호한다. `PhysicalCompletionUnknown`에서는
논리 취소·root 전이만으로 owner/buffer를 release·재사용하지 않는다. store 접점은
[archive](../crates/rz-search/src/pals/store/archive.rs), 소비자 경계는
[archive lifecycle](../crates/rz-search/src/pals/engine/archive_lifecycle.rs)다. 이 새 scope의
allocating call-site 연결과 회귀 검사는 아직 미완료·미실행이다.

CPU의 [stack](../crates/rz-search/src/cpu/stack.rs)과
[lifecycle](../crates/rz-search/src/pals/engine/cpu_lifecycle.rs)은 checkpoint frame 한도와
TT/ordering history 예산을 구분한다. stale pause의 소비 비용은 역사 증거로 유지하고
새 Start와 resume delta에 이중 청구하지 않는다. CUDA Warm의 별도 소유권은
[device Warm](../crates/rz-eval/src/pals_onnx/warm/cuda.rs)와
[native Warm](../crates/rz-uci/src/pals_native/private_warm.rs)이 소유한다. 명시적
`--pals-cuda-private-warm=true`에는 `onnx-cuda`·`experimental-io-binding`과 등록된
FullLine V2 device public memory가 필요하다.

## 작은 학습과 실행 인수

실제 target coverage와 진단 fixture를 분리한다. nonzero checkpoint는 zero-step
format과 별도 domain이며 실제 AdamW state/step·scheduler·RNG/sampler·freeze를 저장한다.
P/C는 공유+해당 private, V는 private V만 update한다. Repair는 P 검사에 포함한다.

CPU FP32, batch/accumulation 1, lr 1e-4, betas 0.9/0.999, eps 1e-8, decay 0.01,
clip norm 1, constant schedule을 사용한다. 역할별 continuous 4와 2+resume+2를 비교한다.
각 온전한 비교는 실제 update 24회이며 CPU 2·15분+정리30초·출력2GiB를 유지한다.
2026-10-10 사용자 승인으로 실패와 재시도를 합친 누적 update 한도는 제거했다.
실제 dispatch/completion의 누적 수와 실패·checkpoint는 계속 보존하며, 이 승인을
본격 학습이나 유료 자원 사용으로 확대하지 않는다. 진단 자산은 arena 승격 불가다.

작은 nonzero 진단의 세 번째 실행에서 P/C/V 각각 8회, 총 24회 update의 모델·AdamW·
scheduler·RNG·sampler 연속/재개 일치와 의도한 parameter membership·동결·유한값을
확인했다. 앞선 실패 1회와 8회를 포함한 누적 소비는 33회다. 실제 학습 target을 사용한
update는 0회이며, 실제 target coverage와 learned V 인수는 이 진단 결과와 구분한다.
구체적인 진단 범위와 이전 학습 제외 이력은
[TRAINING-PLAN](TRAINING-PLAN.md#2026-10-10-pals-diagnostic-학습-smoke)에 연결한다.

GPU는 RTX 4050/WSL, Windows commit 여유 6GiB 이상, memory.high 6GiB/max 12GiB다.
실제 Repair→C→재개와 seeded Warm reference를 검사하고 별도 60분 비용 창을 사용한다.
모델/queue/stack/Warm 효과를 한 번에 주장하지 않는다. legacy/새 모델 paired pilot은
120초+1초·256 ply·15분+정리30초, 선택 기능 off다. 실제 비용·OOM·CUDA·완료 실패는 보존한다.

구현/검사 완료 상태는 후속 소스·PR 기록에 추가한다. 본격 학습·learned V 기력·병합은
이번 계약의 인수에 포함하지 않는다.
실제 GPU Repair→C→paused CPU resume·seeded Warm·physical drain/unknown 수명,
새 V4/FullLineInteractionV2 baseline의 제품 등록·Rust/GPU 인수와 pilot은 아직 pending이다.
새 자산의 독립 CPU 수치 확인은 [이번 후속 범위](PALS-IMPLEMENTATION.md#진단-학습과-남은-실행-인수)에
별도로 기록한다. 기존 historical GPU·CI·
CPU 자료는 원래 소스·설정의 증거로 보존하고 새 기준의 인수로 승격하지 않는다.
