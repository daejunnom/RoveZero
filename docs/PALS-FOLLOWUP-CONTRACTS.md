# PR #23 후속 구현 계약

사용자 승인한 후속 계획의 연결 계약이다. 기존 비학습 인수와 V1/V2/V3 자료를
보존하며, 이 문서의 게시만으로 아래 기능의 구현·실행·인수를 주장하지 않는다.
공통 `EvalRequest/EvalOutput` revision 0.1은 유지한다.

이번 범위는 diagnostic nonzero 학습 smoke까지이며 본격 학습·learned V 인수·PR 병합은
제외한다. 이전 비학습 목표와 실행 자료는 [구현 인수 이력](PALS-IMPLEMENTATION.md)에
보존한다. 현재 연결·검사·미실행 구분은
[이번 후속 범위](PALS-IMPLEMENTATION.md#2026-10-10-pr23-후속-범위)를 따른다.
중탐색 controlled archive와 게임간 session ledger는 Store·Engine·Native·Arena V4에
연결한다. 소스 제공, 실제 검사, NN/GPU·대국 인수는 위 구현 기록에서 별도로 표시한다.

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
  global record ID와 공개 revision은 근거·cache identity이며 신경망의 숫자 특징이 아니다.
  padding은 embedding·각 temporal block·pooling 단계마다 차단한다.
- 모델 schema/encoding V2, PALS 실행 V4를 별도 등록한다. legacy codec/digest를 바꾸거나
  과거 자산을 재해석하지 않는다. Torch reference와 Rust/ORT 입력·수치를 대조한다.
  네 profile의 같은 seed·공통 FP32 tensor bit 일치를 확인하고 비용을 각각 기록한다.
  독립 수치 허용 오차는 logits 절대 1e-4/상대 1e-3, policy/WDL 절대 1e-4다.

소스 접점은 [Rust 모델](../crates/rz-eval/src/pals_model.rs),
[Python config](../experiments/model-research/pals/src/rz_pals_model/config.py),
[model](../experiments/model-research/pals/src/rz_pals_model/model.py)과
[export/artifact](../experiments/model-research/pals/src/rz_pals_model/artifacts.py)다.

## 탐색·저장소·재개

- `ResolverPolicy`: own raw는 own checker만, model WDL은 own/외부 checker 모두 지원.
  새 생성자는 명시적으로 선택한다. 기존 생성자는 기존 조합을 보존한다.
  현재 제품 외부 checker는 절대 Linux 경로의 native ELF·binary/argv·UCI identity를
  검증하며 Linux 이외 실행은 typed `Unsupported`다. 선언과 실제 startup/shutdown은 구분한다.
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

별도 `ResumeAfterAcceptedRepairC` 선택은 iterative Frozen WDL queue와 실제 paused
owner를 요구한다. 기존 constructor의 `ReplaceOnFresh`, 일반 FIFO와 legacy fixture
범위는 유지한다. held-parent-v2의 같은 root/first-move priority drain은 accepted
Repair와 C tail을 원래 owner 재개 앞에 연결하며 다른 pending 순서는 보존한다.

독립 reference의 입력 admission은 metadata/query/record features/divergence features의
FP32 schema leaf만 정확한 32비트 값으로 비교한다. signed zero·비유한 값·bool, 나머지
JSON 타입·ID·관계·수·owner/seed 문맥과 raw serde digest·canonical key·실제 input bits
guard를 유지한다. 이 validator 수정은 native 계산이나 수치 허용오차 변경이 아니다.

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
FnOnce)`와 allocating facade의 `*_controlled` 접점이다. operation 동안
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
[archive lifecycle](../crates/rz-search/src/pals/engine/archive_lifecycle.rs)다. 현재 source의
검사·제품 실행 상태는 구현 기록을 따른다.

새 게임은 독립 Store owner와 cold-handle generation을 만들고 원래 session write ledger를
승계한다. 같은 canonical archive root·runtime 8필드 전체와 미사용 staged owner를 검증하며,
cold-handle authority는 공유하지 않는다. session 상한은 기존 `game_bytes_max`와 같다.
`archive_write_bytes_total`은 owner별 실제 쓰기, `session_write_bytes_consumed`는 owner
교체·close 이후에도 유지되는 누적 소비다. 부분 쓰기·commit 후 실패한 bytes도 소비에서
제거하지 않으며 미작성 예약만 반환한다. 보존된 pending 파일은 디스크 quota에 남는다.
startup와 새 게임의 staged owner는 원래 deadline 안에서 실제 관리 사용량을 scan하고
scan 전후 취소를 확인한다. getter는 새 scan이나 작업량을 만들지 않는다.

기존 archive payload·runtime 8필드와 V1/V2/V3 codec·digest·선택 기능 `None` bytes를
유지한다. V4 actual scope에는 `session_write_ledger_id`, `session_write_bytes_max`,
`session_write_bytes_consumed`의 nullable u64 세 필드를 추가한다. `null`은 unknown이며
숫자 0과 다르고 typed actual scope에서 필드 생략은 거부한다. Native는 실제 owner 사실을
복사하고 Arena는 ledger ID/상한 불변·소비 단조 증가와 owner별 비용·close·partial 사실을
대조한다. 선언만으로 실제 complete를 부여하지 않는다. 회수한 Rust 소유 capacity,
pending 파일의 디스크 extent, RAM buffer·RSS는 별도 계측값이다.

CPU의 [stack](../crates/rz-search/src/cpu/stack.rs)과
[lifecycle](../crates/rz-search/src/pals/engine/cpu_lifecycle.rs)은 checkpoint frame 한도와
TT/ordering history 예산을 구분한다. stale pause의 소비 비용은 역사 증거로 유지하고
새 Start와 resume delta에 이중 청구하지 않는다. orphan frame의 실제 폐기·token/bytes 0·
complete snapshot 확인 전에는 논리 paused task를 retire하지 않는다. snapshot `None`이나
incomplete는 성공이 아니며 resumed value 실패의 새 작업·typed 오류를 보존한다.
CUDA Warm의 별도 소유권은
[device Warm](../crates/rz-eval/src/pals_onnx/warm/cuda.rs)와
[native Warm](../crates/rz-uci/src/pals_native/private_warm.rs)이 소유한다. 명시적
`--pals-cuda-private-warm=true`에는 `onnx-cuda`·`experimental-io-binding`과 등록된
FullLine V2 device public memory가 필요하다.

## 게임별 종료 소유자와 공유 정리 비용

게임별 cleanup 영수증의 소유자는 실제 game PID, 실행 work와 native 종료 근거를 함께 연결한다. `4fef868`의 producer는 전체 pair의 cleanup 구간을 명시하고 공유 pair stream의 비용을 한 번만 청구한다. 공유 pair stream은 첫 baseline game에만 청구한다. 다른 게임의 0은 중복 청구를 피하기 위한 값이며, 해당 자식 프로세스의 stream을 독립 계측한 결과가 아니다. source-proof21의 역변환 검사는 기존 정상 종료, 시계, profile, group 종료와 보존 guard의 소스 경계를 확인한 자료이며 새 실행의 물리 종료를 대신하지 않는다.

V22의 typed Core 생성 실패와 process owner 중복 오류는 원문을 유지한다. 회수된 startup·termination 자료가 개별 물리 완료를 증명하더라도, 기존 Core JSON을 수정하거나 이미 종료된 live owner를 다시 조립하여 새 대국 인수로 처리하지 않는다. 새 producer의 실제 typed Core, 별개 preflight/game 소유자, 입력 lock, PGN·시계·실패 및 종료 근거가 함께 있어야 후속 실행을 인수할 수 있다.

`f75a739`에서 도입한 Admission/Expired의 typed deadline 전달과 그 CPU 검사는 별도 근거로 보존한다. 현재 V22의 성공 반환 자료는 이 오류 분기의 실제 발동 여부를 증명하지 않으므로 해당 관측은 unknown으로 둔다. 모델 의미와 Fresh/Rules namespace, 논리 취소와 물리 완료의 수명 계약은 그대로 적용한다.

`ee6322f`의 실제 Linux CI에서는 네 게임 PID의 cleanup owner와 재사용 거부 검사, 공유 pair stream의 한 번 청구 검사 두 개가 각각 성공했고 clippy도 완료됐다. 이 CPU fixture 결과는 새 producer의 계약 검사이며 Build22·Pilot23의 실제 NN·게임 인수를 대신하지 않는다.

현재 실행 상태와 실패 이력은 [게임별 종료 소유자 수정과 현재 인수](PALS-IMPLEMENTATION.md#2026-10-12-게임별-종료-소유자-수정과-현재-인수)를 따른다.

### 2026-10-13 Pilot23의 실제 종료 관측

`4fef868`/Build22의 새 Pilot23에서 원 producer의 typed Core가 생성됐고 assembly
error·Core failure는 없었다. preflight 4개와 game 4개의 owner를 분리하고,
startup·termination 8쌍·startup/ready profile 원시 파일 32개와 새 입력 lock을
연결했다. 네 game session의 정상 physical shutdown·buffer release와 owned tree
종료, 두 PGN의 색 교환·120+1 시계·Rules 반복 무승부를 실제 자료로 확인했다.
공유 pair stream의 cleanup 비용을 한 번만 청구하는 기존 계약을 유지하며, 다른
게임의 0을 독립 cleanup 계측으로 해석하지 않는다.

원 V22의 typed Core null과 owner 중복 실패는 교정한 성공 자료로 덮어쓰지 않는다.
이번 자료는 별도 source/build/lock/실행 식별로 보존한다. typed deadline 오류 분기의
실제 발동은 unknown이며 CPU 검사와 성공 반환만으로 해당 분기를 인수하지 않는다.
새 기능 관측은 Colab-observed 범위다. strict-local·성능·기력 eligibility와 typed Core의
pair eligibility는 false를 유지한다. NN 완료·검색 소비·착수 반환은 별개 counter다.
Warm off 대국을 새 Warm 인수로 표시하지 않으며 실제 TF32 kernel/plan도 unknown이다.
구체적인 실행 범위는 [Build22와 Pilot23 실제 기능 관측](PALS-IMPLEMENTATION.md#2026-10-13-build22와-pilot23-실제-기능-관측)을 따른다.

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

이전 작은 nonzero 진단의 세 번째 실행에서 P/C/V 각각 8회, 총 24회 update의 모델·AdamW·
scheduler·RNG·sampler 연속/재개 일치와 의도한 parameter membership·동결·유한값을
확인했다. 앞선 실패 1회와 8회를 포함한 누적 소비는 33회다. 실제 학습 target을 사용한
update는 0회이며, 실제 target coverage와 learned V 인수는 이 진단 결과와 구분한다.
이 결과의 영향 source 8개 직접 대조에 따른 재사용 확인과 이번 새 optimizer update 0회를
구분한다. zero-step 형식과 nonzero 실제 AdamW state/step domain도 계속 분리한다.
구체적인 진단 범위와 이전 학습 제외 이력은
[TRAINING-PLAN](TRAINING-PLAN.md#2026-10-10-pals-diagnostic-학습-smoke)에 연결한다.

실제 실행은 자원 프로필과 유한 창을 먼저 등록한다. 최신 사용자 지시에 따라 Windows
commit 여유 6GiB의 진입·지속 stop 조건은 제거했다. CPU0/2·quota2·memory.high 6GiB/
max12GiB·swap0, 유한 작업·단일 정리 예산·물리 완료와 실패 보존의 조건은 유지한다.
Colab에서 이 조건의 실제 readback을 확보하지 못한 결과는 관측 프로필로 표시하고
strict-local 인수와 performance eligibility를 false로 둔다. 외부 GC 요청·실제 GC
실행·메모리 경합 진단을 각각 구분하며 이를 성능 개선 근거로 사용하지 않는다.

실제 Repair→C→동일 owner CPU 재개, accepted seed의 독립 Warm reference와 최종
physical drain/unknown을 각각 인수한다. 모델/queue/stack/Warm과 GQA 반복·Signed192
변환 비용을 분리하며 모델·탐색 변경에 전역 5% gate를 적용하지 않는다. legacy/새 모델
paired pilot은 같은 own checker·`ModelWdlRestricted`, 120초+1초·256 ply·흑백 교환
1 pair/2 games·15분+정리30초, 선택 기능 off다. 각 판 두 엔진을 재시작하여 네 PALS
session 각각의 join/drain과 PGN 2판을 인수한다. 실제 비용·실패는 보존한다.

native UCI의 선택 인수 `--pals-shutdown-timeout-ms`는 1~30,000ms를 받으며,
미지정 실행의 기존 2초 기본값을 유지한다. V4 arena는 전체 cleanup 예산 안에서
owner drain과 감독 회수 시간을 나누고, 30초 정리에서는 owner에 15초를 전달한다.
명시한 시간이 지났다는 사실은 물리 완료 확인을 대신하지 않는다. startup·termination·
실제 ProcessReceipt와 provider 완료·버퍼 해제·격리 조건을 별도로 대조한다.

V4 producer는 `configuration.model_v2.model_identity`를 실제 native namespace와
일치시킨다. CUDA에서 모델 export 식별 뒤에 선언한 device/session arena 식별이
붙으며, 선택 Warm은 별도 namespace다. CPU 형식의 identity를 CUDA 선언에 넣거나
누락된 값을 임의 기본값으로 채워 인수하지 않는다. metadata 교정은 정확한 leaf와
원·수정 materializer pin으로 한정하며 source·model·checkpoint와 native guard를
변경한 것처럼 취급하지 않는다. 실제 loaded identity는 별도 실행 증거로 검증한다.

구현/검사 완료 상태는 후속 소스·PR 기록에 추가한다. 본격 학습·learned V 기력·병합은
이번 계약의 인수에 포함하지 않는다.
실제 검사 코드 baseline555의 CPU fmt·Warm example32·native feature·strict Clippy와
Build14 actual은 [최신 구현 기록](PALS-IMPLEMENTATION.md#2026-10-12-후속-구현과-실행-관측)에 연결한다.
코드555·검증-only8b·이전 문서e43d/1ecf와 문서 기준bf1cc6a의 exact CI 네 job 성공은
각각 원 HEAD의 관측이다. 문서 기준bf1cc6a44dca4b5619b3540e71cfe4a548b87921과 실행
코드555·실제 native c3/Build13을 구분하며 새 assembly 패치의 실행 성공으로 세지 않는다.
PR23 OPEN/Draft develop·미해결 review thread0은 이전 e43d의 총괄 관측이다.
361행 component proof의358동일·세 변경, native c3/Build13 기능·pilot 기준과
검증-only8b overlay·비용 native555/Build14를 구분한다. Build14 repair/pair-prepare의
byte 차이를 보존하고 source 동일성을 전체 binary 동일성으로 확대하지 않는다.
총괄이 인수한 C02 한 causal chain·P06/W04의 source·seed·physical functional 근거는
원 source와 실제 단계 합성으로 보존한다. 연속 pipeline elapsed는 null이며 원 serde
FAIL·budget FAIL·causal HOLD·join 오류·export/wrapper 실패를 덮지 않는다.
Label4 첫 owner 종료·Fresh load deadline/cancel 거부와 원 전체 session physical/tree
unknown을 유지한다. Label5 wrapper FAIL과 분석 당시 Rust verify 미관측도 보존한다.
새555/Build14 capture·독립 reference를 재실행하지 않고 Label6 Rust verify가 실제
통과했고 총괄은 E04 비용 정확성·물리 회수를 functional 인수했다. consumer44와 각lane
준비3+측정5, 전체 role64/public42/NN106은 별도 계수다. 개별표본 전체 NN/cache는
unknown이므로 not_comparable·ratio null·benchmark_validation_complete=false를 유지한다.
Colab-observed·strict-local/performance false와 actual TF32/plan/retry·전체 peak unknown도
유지한다. V10 metadata PASS 뒤 export identity 실패는 snapshot·child·NN·게임 이전이다.
한 manifest wire-schema leaf 교정 후 V12는 derived artifact file budget guard에서 pre-child
중단했다. snapshot/copy·owner retained 가능성과 native physical unknown을 보존하고
outer tree gone을 native shutdown 인수로 세지 않는다. V13은 watcher1071·depth9 통과 뒤
자체 DEFAULT identification preflight의 output-root/launch-sha256/endpoint-id 누락으로
execute exit1이었다. preflight child는 실행됐지만 NN·게임 이전의 실패이며 원 자료와
native physical unknown을 보존한다. inventory 존재만으로 세 receipt 인자를 대신하지
않는다. product guard를 유지한 새 assembly source·frozen 등록·빌드와 V14 actual 실행,
E06 실제 pair·E07 후속 v8 문서 tip·CI·PR 인수는 pending이며
전체45 최종 판정은 총괄 입력 전까지 null이다. 과거 actual receipt는 원 source·binary·설정으로 보존한다.

뒤이어 수정 소스21fad37의 CPU wire12·arena lib123(새 검사5 포함)와 Build15 actual을
[최신 구현 기록](PALS-IMPLEMENTATION.md#2026-10-12-후속-구현과-실행-관측)에 연결한다.
Build15는315.3177초·일곱 명령 exit0/tree gone·다섯 ELF GLIBC 상한2.34다. 문서 기준95e7184의
[exact CI](https://github.com/daejunnom/RoveZero/actions/runs/38208720665) 네 job 성공은 CPU 범위다.
원 수치·정상 Warm c3/Build13 및 비용555/Build14를 새 source로 재라벨하지 않는다. component
재사용 검증·새 arena의 실제 두 게임/네 session과 E06/E07 최종 인수는 대기 상태다.

### V4 native receipt와 launch recipe의 인코딩 식별

V3의 native receipt는 기존 recipe의 `encoding_semantic_sha256`와 대조한다.
V4의 native receipt는 모델 입력의 Fresh namespace를 나타내므로 manifest의
`configuration.model_v2.encoding_sha256`와 대조한다. V4 loader가 검증하는
`configuration.model.encoding_semantic_sha256`는 Rules-only recipe 식별이며
두 값을 교환·자동 변환·같게 선언하지 않는다. epoch·export·adapter·trained 및
startup/termination의 정확한 identity fence와 물리 완료 조건을 모두 유지한다.
새 검사는 Legacy/FullLine·CPU/CUDA, namespace 교환·history/profile 혼입·잘못된
encoding·다른 frozen identity·V3 호환·mock의 native 주장 거부를 포함한다.
[실패·수정·Build18과 실제 인수 경계](PALS-IMPLEMENTATION.md#v4-native-영수증의-fresh-인코딩-경계)를 따른다.
