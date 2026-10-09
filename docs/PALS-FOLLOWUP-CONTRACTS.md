# PR #23 후속 구현 계약

사용자 승인한 후속 계획의 연결 계약이다. 기존 비학습 인수와 V1/V2/V3 자료를
보존하며, 이 문서의 게시만으로 아래 기능의 구현·실행·인수를 주장하지 않는다.
공통 `EvalRequest/EvalOutput` revision 0.1은 유지한다.

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
- hot 한도 80% 또는 allocation 실패에서 비활성 자료를 cold로 이동하고 1회 재시도한다.
  root/frontier/진행 작업/paused CPU/관련 근거를 pin한다. 게임 archive 256MiB, 관리 전체
  4GiB 상한이다. archive commit·무결성 확인 전에 RAM을 해제하지 않는다.
- cold 핸들은 store 소유자/generation을 검증한다. 조회는 deadline/byte 예산이 명시된
  pin/load 접점이다. RAM 인덱스도 bounded다. quota/I/O/pin 포화는 명시 실패이며,
  보존 증거를 삭제해 성공으로 처리하지 않는다.
- `PausedStack`은 opaque token, 1개·8MiB다. 미완료 PVS/qsearch/SEE frame과 정확한
  Rules/accumulator·ordering·TT 소유권을 보존한다. consumed work를 다시 세지 않는다.
  다른 history/profile/root/namespace, 취소·새 게임에는 재사용하지 않는다.
  CPU admission마다 실제 owner와 token을 대조한다. stale paused 기록은 역사 증거로
  보존하고 새 작업으로 다시 시작한다. CPU 호출 밖의 논리 취소도 실제 stack을 폐기하며
  GPU lease의 물리 완료 규칙은 별도로 유지한다.
- CUDA ApproxWarm은 host Warm과 별도 capability다. accepted seed만 같은 role·모델·
  Rules/query 문맥에서 사용한다. 공개 record revision만 변경 가능하다. value/V는 Fresh다.
  논리 취소가 물리 완료를 대신하지 않는다. 완료 불명 owner/buffer는 격리·보존한다.

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

GPU는 RTX 4050/WSL, Windows commit 여유 6GiB 이상, memory.high 6GiB/max 12GiB다.
실제 Repair→C→재개와 seeded Warm reference를 검사하고 별도 60분 비용 창을 사용한다.
모델/queue/stack/Warm 효과를 한 번에 주장하지 않는다. legacy/새 모델 paired pilot은
120초+1초·256 ply·15분+정리30초, 선택 기능 off다. 실제 비용·OOM·CUDA·완료 실패는 보존한다.

구현/검사 완료 상태는 후속 소스·PR 기록에 추가한다. 본격 학습·learned V 기력·병합은
이번 계약의 인수에 포함하지 않는다.
