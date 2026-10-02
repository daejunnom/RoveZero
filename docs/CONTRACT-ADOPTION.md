# 공통 계약 적용과 Draft PR 인수 지시서

이 문서는 총괄 TASK-I01/I02가 최초 공통 Rust 계약을 게시하고 A~F의 독립 WIP를
연결할 때 수행할 작업을 정한다. 의미 계약은 [CONTRACTS](CONTRACTS.md), 역할과
작업 순서는 [IMPLEMENTATION-DIRECTIVES](IMPLEMENTATION-DIRECTIVES.md),
협업과 원격 반영은 [CONTRIBUTING](../CONTRIBUTING.md)을 따른다.
원격 PR의 코드가 서로 병렬로 진척되므로 이 문서의 조사 SHA는 고정된 관찰 자료다.
통합 직전에 최신 head와 diff를 다시 읽고 실제 타입·수명·검사 범위를 확인한다.

## 1. 게시 revision과 증거 범위

최초 공통 crate는 `rz-contracts` **0.1.0**, engine `SchemaVersion`은 **0.1**이다.
현재는 major/minor 모두 exact match를 요구한다. Cargo package의 세 번째 숫자와
engine schema의 두 숫자는 서로 다른 버전 축이며 dataset envelope 버전과도 구분한다.
알 수 없는 minor를 같은 major라는 이유만으로 수락하지 않는다. 변경이 필요하면
총괄이 영향 consumer·호환성·인수 검사를 검토하고 revision을 게시한다.

공통 crate는 표준 라이브러리만 사용하는 CPU 기본 계약이다. concrete chess board,
search tree, scheduler, encoder tensor, GPU stream·provider를 역으로 의존하지 않는다.
공통 crate를 추가한 사실이나 crate 자체 검사의 성공은 A~F와의 실제 연결 성공이 아니다.
루트 workspace와 CPU CI의 게시·dispatch·관측 결과도 각각 분리해 기록한다.

이 문서는 consumer의 적용 지시서다. 아래 source pin은 각 담당 PR의 독립 WIP를
관찰한 자료이며 현재 총괄 integration 변경과 구분한다. 공통 계약은 PR #7로 병합됐고
총괄은 PR #9에서 실제 source adapter·workspace를 수동으로 연결 중이다.
문서 작성·source 반입으로 consumer compile/test·신경망·GPU·대국 인수가
완료된 것으로 표시하지 않는다. 실제 공통 API는 `crates/rz-contracts/src/`의 게시된
동일 SHA를 기준으로 사용하고 의미가 같은 로컬 이름을 자동으로 같은 타입으로 보지 않는다.

공통 0.1의 실제 접점은 다음과 같다.

| source | 공개 접점과 적용 범위 |
|---|---|
| [primitives.rs](../crates/rz-contracts/src/primitives.rs) | `CONTRACT_REVISION`, `SchemaVersion::validate()`, epoch별 `RequestId/SelectionId/ExecutionId`, `ClockDomain/MonotonicTick/Deadline`, `CancelToken`, `Move/Promotion/Color`, `Wdl::try_new()/normalized()`, `ContractError` |
| [state.rs](../crates/rz-contracts/src/state.rs) | `PositionSnapshot<P>::try_new()`, `LegalMoveView::try_new()`, `PositionClassification`; frozen Rules state `Arc<P>`를 보관하고 실제 chess legality/live authority는 A adapter가 확인 |
| [evaluation.rs](../crates/rz-contracts/src/evaluation.rs) | `EncodingDescriptor/ModelDescriptor`, `EvalContext/EvalRequest<P>`, `AcceptanceScope`, `LegalPolicy`, `EvalOutput::validate_for()`, `EvalResult`, `Evaluator<P>`; 논리 수락을 물리 buffer·search consume와 분리 |

최소 0.1은 **fresh 계산**의 유한 단계 예산과 computed/exact-feature/raw-cache provenance를
지원한다. `require_full=false`의 허용 단계 수에서도 완전한 policy/WDL head를 반환해야
한다. 선정한 Maia의 첫 기준선은 full 1-step이다. warm-start/approximate latent·불완전
head를 뒤이어 채우는 partial refinement는 이 revision에 없다. 다른 의미를 기존 variant나
absent 값에 숨기지 말고 총괄에게 별도 revision을 제안한다.
`Digest`, `RuleKey`, `EvalInputKey`, `LegalOrderIdentity`는 adapter가 실제 입력으로 발급하는
typed attestation이다. 공통 crate에는 hash 구현·영속 wire codec·검증된 model registry가 없다.
constructor의 coherence 검사 성공을 actual Rules/encoder/registry 검증으로 표현하지 않는다.

`EvalRequest::validate_acceptance(scope, clock, now)`와
`EvalOutput::validate_for(request, scope, clock, now)`는 실제 현재 scope를 admission/finalization과
backup 직전에 다시 받아 검증한다. mutable owner가 가진 live state/registry와 scope를 연결하는
일, SelectionTicket exactly-once consume와 physical Lease/buffer owner의 해제는 B/D adapter의
별도 책임이다. 공통 함수 호출은 consume token을 사용하거나 device completion을 확인하지 않는다.

`ContractError`는 고정 ErrorCode/Stage와 static detail이며 `EvalFailure`는 완료 문맥과
RecoveryOutcome을 붙인다. backend의 임의 String을 static detail로 복사하지 않는다.
C/D adapter는 원래 backend code/stage·실제 실행 상태·추가 복구 비용을 유한한 local
진단/인수 기록에 보존하고 typed 공통 오류에 대응한다. 새 문자열을 잃어버리거나 비밀·
사용자 경로·무제한 외부 payload를 공통 오류에 싣지 않는다.

## 2. 조사한 PR과 연결 책임

초기 2026-10-03 조사는 `develop`의
`9f0bc598f6b2d8f863fd46af6a4fd73bfef1f0b8`을 기준으로 했다. 그 뒤 공통 계약
[PR #7](https://github.com/daejunnom/RoveZero/pull/7)이 병합되어 현재 총괄 통합의
develop base는 `3e3cd80533a69ee2118f1c130e72a1cb0745ae8c`다.
다음 표는 재개 후 총괄이 포착한 담당별 source pin이다. 초기 관찰 base, 담당 branch
head와 총괄의 integration SHA를 혼용하지 않는다. E/F는 아래 full SHA의 source·tests를
읽기 전용으로 대조했고, A~D의 짧은 표시는 총괄이 제공한 captured head prefix다.
표의 head는 이후 push로 바뀔 수 있으므로 실제 인수 직전에 다시 확인한다.
PR·리뷰·checks의 부재나 source의 test 선언 수만으로 실행 성공을 인정하지 않는다.

| 담당 | PR·조사 head SHA | 총괄이 연결할 공통 경계 | 담당 영역에 유지할 구현 |
|---|---|---|---|
| A | [#3](https://github.com/daejunnom/RoveZero/pull/3), `118dc03` | Rules가 발급한 state/revision·semantic move·ordered legal view와 공통 상태 권한 | board/FEN/attacks/legal/make-unmake·이력·종료·규칙 내부 key |
| B | [#6](https://github.com/daejunnom/RoveZero/pull/6), `e80be2f` | SelectionTicket의 논리 ID·세대·deadline을 공통 요청으로 변환하고 검증된 policy/WDL을 backup 접점으로 변환 | PUCT tree·driver·selection/reservation·방문·backup·시간 배분·UCI/worker 소유 구현 |
| C | [#5](https://github.com/daejunnom/RoveZero/pull/5), `0a56367` | 모델/인코딩 descriptor, immutable state/legal view, request/result/error, backend와 runtime의 physical completion | ScriptedBackend와 raw output, 인코더·action mapping·loader·provider·수치 adapter |
| D | [#4](https://github.com/daejunnom/RoveZero/pull/4), `83c2869` | Adapter associated types, clock/deadline domain, admission/current/output 검증과 공통 오류 | Scheduler·queue·batch·Resources/Limits·Lease·reservation·telemetry |
| E | [#8](https://github.com/daejunnom/RoveZero/pull/8), `383eb05c55f65cd2df5c8642a7f374e50be084b5` | engine 계약 revision과 실행 manifest의 버전/해시/설정 provenance, Rules 검증 접점 | 입력 잠금·artifact 확인·pair 계획·attempt 원장·원시 WDL/pentanomial; 실제 runner·군집 통계는 후속 |
| F | [#2](https://github.com/daejunnom/RoveZero/pull/2), `36e619bae238a587e04c0a4bae24624eb5414aec` | engine 계약 revision, Rules/인코딩 identity bridge, move·관점·종료 의미 | Python envelope·라벨/group split·누출 audit·CPU 숫자 fixture gradient/checkpoint/export/resume; 제품 NN 학습은 후속 |

공통 타입·오류·revision을 consumer마다 복제하지 않는다. 이미 구현된 local 타입을
무조건 모두 공통 crate로 이동하지도 않는다. 공통 의미와 owner를 지킨 얇은 adapter를
각 담당 경계에 두며 총괄이 실제 PR 인수 변경에서 mapping을 수동으로 맞춘다.
현재 통합은 [Draft PR #9](https://github.com/daejunnom/RoveZero/pull/9)에서 진행한다.
루트 10개 member·path 의존·toolchain 선언과 수동 adapter 작업, 실제 통합 검사·
최종 SHA의 기록은 [INTEGRATION-STATUS](INTEGRATION-STATUS.md)에 분리해 둔다.

## 3. A: Rules authority와 합법 수

A의 실제 상태 owner가 생성·검증한 불변 snapshot 또는 수명이 고정된 읽기 view를
공통 상태 접점에 연결한다. `state/revision/order` 숫자를 caller가 채워 넣은 DTO만으로
Rules 검증이 완료되었다고 수락하지 않는다. live authority는 owner·registry/slot 세대·
revision과 실제 view의 일치를 확인한다. owner가 폐기·교체된 handle은 같은 숫자가
다시 나타나도 수락하지 않는다. mutable board를 async 요청보다 짧게 borrow하지 않는다.

조사 head의 A `position.rs`는 `PositionSnapshot`, `LegalMoveView`와
`Position::matches_snapshot()`/`make_from_view()`를 제공한다. 의미 비교인
`PositionSnapshot::same_state()`와 live owner/revision 검사를 서로 대체하지 않는다.
`known_history()`는 newest-first이며 반복 identity와 full-history state identity가 별개다.

semantic identity는 `from/to/promotion`이며 a1=0, h8=63 rank-major 방향을 명시적으로
맞춘다. A 내부 square/move/key를 공통 타입과 같은 숫자라서 자동 변환하지 않는다.
외부 UCI 문자열, Rules 내부 move, 공통 semantic move, 모델 action index는 각 경계에서
확인한다. 승격 queen/rook/bishop/knight는 같은 from/to라도 서로 다른 move다.

ordered legal view는 상태 revision에 귀속한 중복 없는 순서와 order identity를 가진다.
최소 공통 `LegalMoveView`는 A의 실제 순서를 보존한다. A의 `BoardMove::sort_key()`가
같은 from/to에서 사용하는 Q/R/B/N 순서를 유지한다. 다른 표현에 순서 변환이 필요하면
명시 adapter와 policy permutation을 함께 검사한다. 승격 네 종류·castling·en passant·같은 보드와 다른 이력·make/unmake
후 복원·owner 또는 revision 교체 뒤 stale view의 거부를 연결 검사에 포함한다.
board hash·repetition key·신경망 input identity를 한 identity로 합치지 않는다.

Rules의 ongoing/terminal, claim availability와 history evidence는 독립 축이다.
FEN 이전 이력을 unknown prefix로 보존하고 encoder fill을 실제 이력으로 기록하지 않는다.
Rules의 정확한 terminal은 신경망 queue에 넣기 전에 처리한다. dead-position shortcut은
증명한 subset의 검사 범위와 함께 인수하며 완전 판정기로 표시하지 않는다.

## 4. B: SelectionTicket와 평가 결과

총괄은 B의 SelectionTicket을 조사한 최신 source에서 읽고 공통 RequestId와
SelectionId를 분리해 대응한다. retry는 새 RequestId를 만들되 같은 selection의 소비는
한 번만 허용한다. ExecutionId는 물리 backend 작업 ID이며 selection/backup ID가 아니다.
process epoch, game/root generation, 상태 revision과 legal order를 요청에 보존한다.

B의 `tree.rs`에 있는 `SelectionTicket`의 owner/serial 검사는 그대로 유지한다.
공통 세대 검사가 B-local consume capability를 대체하지 않는다. `accept_evaluation()`에는
실제 요청의 legal move 순서·f64 prior·leaf 차례 기준 scalar를 전달하고 exact Rules terminal은
`accept_terminal()`로 evaluator를 우회한다. 초기 조사에는 rz-search만 게시되어 있었지만
후속 captured head와 현재 통합에서는 UCI/worker의 실제 구현·계약 적용·binary/trace
검사를 별도로 확인한다. 초기 조사 당시의 부재를 최신 코드의 부재로 옮기지 않는다.

B가 UCI 시계에서 산출한 deadline을 공통 ClockDomain의 단조 ns deadline으로
명시 변환한다. UCI wall-clock 또는 `Instant`의 내부 표현을 숫자로 옮기지 않는다.
변환 overflow·clock owner 불일치·현재 tick에서 이미 만료된 기한은 typed failure다.
결과 검증과 수락이 끝나는 tick은 deadline보다 작아야 한다. Search는 runtime 결과를
받은 후에도 현재 세대·마감·selection의 consume 여부를 다시 확인한다.

C의 f32 head를 넓힐 때 finite/range/shape/order, side-to-move viewpoint와 선언한 합 오차
tolerance를 먼저 검사한다. 공통 `LegalPolicy`는 f64이며 `LegalPolicy::try_new()`는 검사만
수행한다. B가 정확한 합을 요구하면 유효 f32→f64 반올림 범위의 값에 대해
`LegalPolicy::normalized()`를 **명시 호출**해 전달하고 이 정책을 model/compute identity와
검사에 기록한다. 허용 범위 밖 NaN/Inf·음수·합 오류를 clipping·uniform·무승부로 고치지 않는다.
공통 `Wdl`은 f32 W/D/L이며 `Wdl::try_new()`도 정규화하지 않는다. B driver가 policy와
WDL 모두 f64 합 오차 1e-9를 요구하므로, 허용된 head에 `Wdl::normalized()`를 명시
호출해 f64 복사본을 넘긴다. 관점은 side-to-move로 유지하고 B의 leaf scalar는 전달한
복사본의 W-L로 계산한다. 부모 한 ply backup은 W/L 교환과 부호 반전이며 centipawn이나
B search utility를 raw WDL과 섞지 않는다.

작은 독립 트리에서 값 부호·prior·방문·단 한 번 backup, 네 종류 승격의 policy mapping,
legal permutation, boundary tick, 중복/늦은 결과와 stop/newgame/root 교체를 검사한다.
최소 PUCT 또는 B의 기존 mock 검사 성공을 실제 C/D 연결 검사로 재사용하지 않는다.

## 5. C와 D: callback·physical completion·clock·buffer

C 조사 head의 `crates/rz-eval/src/mock.rs`는 caller-owned `K` ticket과
`Callback`, `CancelAcknowledged`, `DeviceCompleted`를 별개 event로 제공한다.
raw policy logits/WDL와 injected failure는 아직 검증되지 않은 backend 경계의 타입이다.
C adapter가 공통 요청 문맥을 ticket에 보존하고 모델별 인코딩/action/WDL을 검증한 뒤
공통 성공 payload를 만든다. malformed head나 injected code/stage를 정상 성공으로 숨기지 않는다.

새 C head의 `output::validate_maia()`/`ValidatedHeads`는 1858 raw policy logits와
이미 확률인 WDL을 검증한다. legal index를 순서대로 gather한 뒤 policy에만 temperature 1.0
softmax를 적용하고 WDL에는 다시 softmax를 적용하지 않는다. admissibility 합 오차 1e-5와
독립 FP32 reference parity 기준은 다른 검사다. ValidatedHeads만으로 요청 identity·세대·
deadline이 검증된 것은 아니므로 공통 EvalOutput의 metadata 검증을 추가한다.

`rz-encoding`의 `classical::Frame/Input`은 Rules state가 아닌 모델 projection이다.
A checked view에서 absolute bitboard/known history/raw EP/권리를 명시 변환한다. 흑은
rank만 반전하고 C의 모델 action mapping에서 knight 승격은 일반 from/to slot, Q/R/B는
별도 slot이다. castling의 모델 king→rook 표현과 Rules king→착수 목적지를 구별한다.
model history-fill/padded frame을 실제 game/repetition history에 되돌려 넣지 않는다.

D 조사 head의 `crates/rz-runtime/src/boundary.rs`에서 `Backend::dispatch`의 Err는
device work가 시작되지 않았음을 보장한다. 시작된 작업은 실패 예정이어도 Lease를 가진다.
`poll`의 Pending은 물리 borrow와 input/output/workspace pin을 보존하고 Ready만 physical
completion 및 Lease를 빌리지 않는 owned output을 보장한다. **C의 Callback 수신은
D의 Ready 조건이 아니다.** 총괄이 C↔D adapter에서 후보 callback을 모아 요청별 결과를
검증하고 DeviceCompleted를 확인한 뒤 Ready를 만들도록 연결한다.

취소 ack, 논리 canceled/expired/stale, search의 소비·backup, physical 완료와 buffer 해제는
서로 다른 사건이다. C callback이 취소 뒤 또는 역순으로 도착해도 현재 요청/세대/deadline을
확인한다. D는 completion 전체의 RequestId 집합을 먼저 검사하고 중복·누락·다른 요청
결과를 거부한다. 미완료 Lease의 timeout/quarantine을 성공 drain으로 기록하지 않는다.

D `Adapter`의 RequestId/ExecutionId/Tick/Request/Output/Error/TerminalResult associated
types를 게시한 공통 API에 직접 대응한다. BatchKey와 Resources/Limits, RuntimeFault,
TerminalEvent, 구체 Backend/Lease와 scheduler ledger는 D 소유에 유지하고 공통 error에
원시 stage/code와 요청·실행 문맥을 보존해 매핑한다. `terminal` construction에 search
backup이나 임의 callback을 넣지 않는다. admission 실패는 요청 소유권 반환과 오류 한 경로다.

D SystemClock의 Instant와 공통 단조 ns tick 사이에는 하나의 ClockDomain과 기준 Instant
epoch를 소유하는 adapter를 둔다. 같은 도메인의 elapsed만 계산하고 ns 변환은 checked
arithmetic을 쓴다. UTC는 로그용이며 deadline 비교에 쓰지 않는다. mock 수동 시계도
동일 domain 규약으로 동작하며 다른 clock domain의 tick은 같아 보여도 거부한다.

총괄 root 통합에서 담당의 standalone `[workspace]`와 독립 의존 선언을
root member·path dependency·feature·toolchain·lockfile로 수동 정합한다.
현재 root는 10개 crate를 member로 선언하고 compiler 1.96.0과 workspace rust-version
1.90을 사용한다. 공통 crate의 MSRV 1.85는 별도 범위이며 전체 workspace의 1.85
호환을 뜻하지 않는다. 실제 manifest·adapter·검사 범위는 통합 SHA로 확인하며
CPU/mock와 명시 provider feature의 의존 방향을 유지한다.

수동 연결 검사에는 정상/실패·역순·중복·partial/NaN/Inf, 취소 ack 이전/이후 callback,
DeviceCompleted 전 buffer pin, logical 취소 뒤 physical pin, root/model/slot 교체,
queue/memory 상한·overflow, completion receiver drop와 유한 shutdown/drain을 포함한다.
CPU mock 수명 검사와 실제 GPU provider의 물리 borrow 검사를 별개 결과로 남긴다.

## 6. F와 E: persisted wire와 실행 provenance

F `experiments/model-research/src/rz_data/schema.py`의 envelope `schema_version=1`은
engine `SchemaVersion=0.1` 또는 Cargo `rz-contracts=0.1.0`과 다른 namespace다.
manifest의 engine_contract_revision에 게시된 revision을 대응하더라도 Python envelope의
버전을 자동으로 바꾸지 않는다. dataset/teacher/split/label metadata와 DataError는 F 소유다.

F `serialization.py`의 digest는 sort_keys·compact separator·UTF-8·비유한값 거부를 사용한
Python JSON bytes의 SHA-256이다. Rust state/legal/input identity가 이 digest와 같다고
가정하지 않는다. wire bridge에서 hash algorithm·직렬화·필드 범위·revision을 명시하거나
별도 typed identity 필드를 제공하고 검사한다. SHA 문자열의 syntax 성공은 해당 상태나
인코더 입력을 검증한 증거가 아니다. Rust/Python float JSON 표현의 차이도 별도로 다룬다.

총괄은 A 복원과 정확한 legal sequence·종료/이력 비교, C의 실제 model-input encoding
identity, teacher provenance 검증을 F bridge에 수동 연결한다. ordered legal 수와 label
policy의 moves/values를 함께 비교하고 WDL/q/centipawn과 scale/viewpoint를 구분한다.
fixture의 선언값을 product Rules나 실제 NN의 결과로 인정하지 않는다.

조사 head의 F는 execution_ready를 false로 강제하고 audit의 training_eligible_records=0,
state_restore/independent_legal_moves/actual_encoding_identity 등을 not_run으로 남긴다.
**공통 계약 게시만으로 execution_ready를 true로 바꾸지 않는다.** 실제 bridge 검사와
독립 상태·인코딩 증거가 확보된 후 F의 실행 준비 조건을 별도 변경·인수한다.

F의 최신 source에는 F02 내부 lifecycle도 제공돼 있다. `rz_training/recipe.py`는
execution_scope=cpu_fixture·float64·단일 fixture adapter와 engine wire_binding=not_run을
강제하고 `data.py`는 실제 teacher/self_play 입력을 거부한다. holdout forward와 선택은
제외되며 trainer가 model·SGD momentum·sampler/RNG·진행·validation 선택을 저장·복구한다.
실패/취소 step rollback과 typed receipt, native fixture export의 import/probe 대조도
제품 호환 검사와 구별한다. 숫자 fixture의 weights 변경·loss 감소는 실제 Maia 학습·
GPU·CONTROL 대국 인수의 근거가 아니다.

E는 실제 engine binary·weights·backend·precision·options·장비·seed·완전 시작 상태·
clock와 계약 revision을 실행 manifest에서 보존한다. 계약 revision 일치만으로 binary나
weights hash를 생략하지 않는다. runner fixture/CPU 외부 엔진 검사는 먼저 수행할 수 있지만
정식 paired 결과·실패 정책·통계와 목표 장비 대국은 별도 인수한다. Rules를 E에 복제하지 않는다.

E 조사 branch의 `RunManifest::contract_revision`은 engine revision을 별도 string으로
담고 E persisted `SCHEMA_VERSION=1`/canonicalization을 유지한다. 공통 0.1 exact match의
string 대응을 명시한다. `LockedManifest`의 구조 잠금은 launch/legality/fairness가 검증되었다는
뜻이 아니며 execution_ready=false를 유지한다. 최신 E source에는 같은 OpeningSpec의
엔진 색 배정만 바꾸는 결정적 plan, infrastructure-invalid pair 전체 제한 재시도·
attempt 원장, 마지막 완료·유효 pair의 원시 WDL/n0..n4 집계가 제공됐다.
미완료 pair에 0.5를 주지 않고 이전 실패 attempt도 보존한다.

E에는 F와 같은 literal not_run 감사 필드가 없으며 execution_ready=false와
validation_scope=structural_only로 경계를 표시한다. move 문법/FEN 필드·declared identity
검사는 실제 A 복원·legal/terminal·PGN 증거와 다르다. 실제 외부 runner/launch·시계·
자원·drain 영수증, cluster bootstrap/CI/Elo 계산과 정식 대국은 아직 인수하지 않았다.
A/C bridge만 연결한 결과로 이 실행 gate를 승격하지 않는다.

읽기 전용 리뷰에서 F의 test method 105개와 E의 test 선언 74개를 확인했다.
E 74개 중 4개는 Unix 조건부다. 이는 source 정의 수이며 이 문서 변경에서 검사·
학습을 실행한 결과가 아니다. 담당자가 보고한 과거 단독 검사는 현재 root 통합의
동일 SHA·환경·feature·영향 입력 검사와 분리한다.

## 7. 총괄의 PR 인수 작업

1. 인수 직전에 대상 PR의 최신 head/base SHA·상태·파일·body·일반/inline 리뷰·checks를
   다시 읽는다. 조사 snapshot보다 후속 수정이 있으면 영향 경계와 검사 범위를 다시 판단한다.
2. 최신 develop과 분리된 총괄 integration branch에서 공통 revision·root member·의존성·
   feature·toolchain을 맞춘다. consumer의 local 타입 중 유지할 책임과 공통 타입으로 바꿀
   경계를 source diff로 확인하고 필요한 adapter를 **총괄이 수동으로 조정**한다.
3. A/B/C/D/F와 E의 실제 연결부를 compile하고 위 역할별 consumer 검사를 실행한다.
   최초 게시·단독 crate 검사·의존성만 추가한 compile·CI 요청을 통합 완료로 보고하지 않는다.
4. 실패와 수정·검증·미실행 영역을 작은 커밋으로 공유하고 게시한 계약 SHA/revision과
   각 consumer head·통합 SHA·설정·명령·실행 환경·결과를 PR 코멘트/인수 기록에 남긴다.
5. 필수 검사·계약·리뷰를 만족한 변경만 규약의 develop 대상 PR로 인수한다. 실패 시
   되돌릴 기준 조합과 미완료 작업을 유지한다. 이번 문서는 자동 merge 권한을 만들지 않는다.

각 PR 코멘트는 적용할 계약 revision/소스 SHA, 담당이 다음 작업에 사용할 경계,
총괄이 인수 때 수동으로 맞출 항목, 실제 완료/실패/미실행 검사를 구체적으로 적는다.
댓글을 남겼다는 사실은 consumer 코드가 연결되었거나 검사가 완료되었다는 증거가 아니다.

## 8. 클라우드 CPU 검사와 실제 장비 인수

A~F는 GPU가 없는 클라우드에서 CPU/mock·fixture 개발과 가능한 CPU 추론 대조를
진행한다. development host의 capability와 목표 RTX 4050 6GB 검증 환경을 분리한다.
CUDA CLI 설치나 GPU 검사 skip을 실제 장치·provider 지원 또는 성공으로 해석하지 않는다.
GPU 요구 모드를 CPU로 조용히 바꾸지 않고 BackendUnavailable 등 명시 실패를 보존한다.

인수 상태는 공통 crate 검사, consumer CPU/mock 통합, 실제 CPU NN 수치 대조,
목표 GPU NN/physical buffer 검사, GPU 종단 D02/D03, 실제 F02 학습, 정식 paired 대국으로
나누어 기록한다. CPU 기본 CI가 성공해도 NN·GPU·학습·강도 gate는 승격하지 않는다.
GPU 미할당이면 해당 검사는 미실행이고 실제 사용 가능한 지정 환경의 후속 검증자와
명령·입력/fixture·SHA·revision·backend/precision·자원 상한을 인계한다.

새 유료 GPU·장시간 실행은 기존 사용자 배정과 확정 예산을 따른다. 이 계약 연결 작업은
GPU 비용 승인이나 원시 가중치·학습 데이터·PGN·로그의 Git 반입을 추가 승인하지 않는다.
