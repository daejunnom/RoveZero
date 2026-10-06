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
진행한다. development host의 capability와 지정한 GPU 검증 환경을 분리한다.
초기 RTX 4050 인수는 해당 장비의 이력으로 보존하며, 후속 GPU 벤치마크는
[외부 RunPod 준비 계획](research/RUNPOD-BENCHMARK-PLAN.md)을 따른다.
GPU A/B 이전의 [CPU 의미 보존 최적화](research/PRE-RUNPOD-OPTIMIZATION.md)는 Rules의
실제 owner/revision/history에 묶인 `ContractPosition::fork_from_view`로 B 접점을 수동
갱신했다. 공통 계약 `0.1`·digest codec·순서·claim·WDL은 유지한다. 이 소스의 NN/GPU
인수는 별도이며 Community/no network volume/초기 약 US$200·외부 회수 계획을 따른다.
CUDA CLI 설치나 GPU 검사 skip을 실제 장치·provider 지원 또는 성공으로 해석하지 않는다.
GPU 요구 모드를 CPU로 조용히 바꾸지 않고 BackendUnavailable 등 명시 실패를 보존한다.

인수 상태는 공통 crate 검사, consumer CPU/mock 통합, 실제 CPU NN 수치 대조,
목표 GPU NN/physical buffer 검사, GPU 종단 D02/D03, 실제 F02 학습, 정식 paired 대국으로
나누어 기록한다. CPU 기본 CI가 성공해도 NN·GPU·학습·강도 gate는 승격하지 않는다.
GPU 미할당이면 해당 검사는 미실행이고 실제 사용 가능한 지정 환경의 후속 검증자와
명령·입력/fixture·SHA·revision·backend/precision·자원 상한을 인계한다.

새 유료 GPU·장시간 실행은 기존 사용자 배정과 확정 예산을 따른다. 이 계약 연결 작업은
GPU 비용 승인이나 원시 가중치·학습 데이터·PGN·로그의 Git 반입을 추가 승인하지 않는다.

## 9. PR #18 전체 최적화의 실제 소비 접점

이번 OPT-01~12는 사용자가 **모든 영역의 구현을 A(Codex)에게 배정**했다.
내부 구현 후 실제 A/C/D/B 선언과 소비자를 같은 작업 브랜치에서 맞췄다.
`rz-contracts`의 공통 필드·revision **0.1**, state/input digest codec과 기본 FP32/full
경로는 유지한다. routine API 연결을 위해 추가 승인을 요청하지 않으며 총괄의 최종
통합 리뷰는 유지한다. 구현·검사 소스와 명령은 [최적화 기록 11장](research/PERFORMANCE-OPTIMIZATION-PLAN.md#11-opt-0112-전체-구현과-검증)에 고정한다.

| 경계 | 실제 선언·소비자 | 호환·수명·한도 |
|---|---|---|
| A → C | `PositionSnapshot::piece_bitboards`, `recent_history_frames`, weak position identity → `ClassicalProjection` | 유지 중인 12 bitboard·최근 8 frame을 읽는다. 전체 known prefix의 반복 증거·unknown prefix·raw EP를 보존한다. weak identity는 같은 history node에서만 hit하며 history를 pin하지 않는다 |
| C 준비 | private immutable projection/tensor/key/ordered indices → `MaiaBinding::prepare_rules` | 1-slot prepared cache; 같은 live state와 정확한 legal 순서만 공유. 새 request의 profile/budget/input/authority 검사를 생략하지 않는다. generic `prepare`는 독립 인코딩 경로 유지 |
| B → A | `rz_search::contracts::ContractPosition::retained_bytes` → Rules adapter와 bounded search state cache | 기존 opaque consumer의 기본 반환값은 `None`이며 재사용하지 않는다. Rules는 공유 prefix를 과다 계산한 보수적 charge를 제공한다. 기본 64항목/8MiB, hard 1024항목/64MiB, root마다 폐기 |
| B 진행 조회 | `ContractSearch::best_move`, `counters` → UCI | 진행 중 전체 outcome/문자열을 만들지 않는다. 최종 outcome과 통계는 기존 경로 유지 |
| C → D raw 재사용 | `ExactRawReuse`, `ClassicalProjection::raw_cache_provider`, D `submit_reused` | 실제 tensor의 모든 float bit와 metadata 및 model/encoding/backend/precision/compute/epoch/game이 일치해야 한다. root는 namespace에 넣지 않되 새 root의 요청·legal·deadline·cancel을 다시 검사한다 |
| worker → UCI 대기 | `CompletionSignal`, `EvaluatorFactory::completion_signal`, `ManagedEvaluator::wake_after` | sequence는 predicate 검사 전 읽는다. 결과는 기존 single consumer가 받으며 signal은 결과를 운반하지 않는다. 완료 publication·취소·quarantine/close가 알린다. 아직 물리 실행이 없는 queued batch는 최대 200µs timer로 다시 pump한다 |
| C 실행 모드 | `BackendConfig.experiments: ExecutionExperiments` → native bootstrap/worker | public Rust struct literal consumer는 새 필드를 채워야 하며 `BackendConfig::cpu()`의 기본값은 모두 false다. 비기본 실행 모드는 backend identity에 포함한다. wire/ORT ABI 변경은 없다 |
| 다중 요청 | B `ContractSearch::set_parallelism` / C explicit batched owner / D mixed-legal capability → native UCI `--experimental-batch=N` | 폭 1..16, 단일 물리 worker, 최대 물리 실행 1, native batch 대기 200µs. 기본 B1 owner는 유지한다. full raw heads를 제공하는 C가 명시한 경우에만 서로 다른 legal 수를 묶는다 |

raw cache는 기본 runtime 설정에서 꺼져 있다. 기본 typed 한도는 64항목/4MiB,
hard 1024항목/64MiB이고 stage 후보와 live ID도 유한하다. 실제 Computed 출력은
**D의 최종 승인 후에만** 재사용 entry로 승격한다. 취소·만료·세대 변경·수락 실패는
stage/live 등록을 제거하며 늦은 물리 완료가 이를 다시 만들지 못한다. hit는 정상
admission과 새 single-consumer receipt를 거쳐 `RawEvalHit`으로 완료한다. 새 물리
ExecutionId·Computed event를 만들지 않고 실제 출처 execution을 별도로 보존한다.
ucinewgame은 컨테이너와 namespace를 초기화한다. B는 hit도 정상 selection 하나에
대해 한 번만 backup한다. 취소한 물리 요청의 실제 lease는 기존 fence까지 유지한다.

native typed report는 Computed 완료/초기화/backup과 raw-hit 완료/초기화/backup을
분리한다. 기존 **Computed-only CPU/CUDA V1 attestation**은 raw-hit, 실행 옵션,
B>1을 baseline으로 직렬화하지 않는다. 옵션과 V1 attestation의 조합은 CLI에서
거부한다. 성공한 bootstrap placement probe·`binding_runs()`를 검색 결과의 GPU
소비나 실제 Graph capture/replay의 증거로 승격하지 않는다.

buffer pool은 정확한 shape/capacity와 반환된 소유권에 한해 재사용한다. 다른 caller가
보존한 출력은 수정하지 않는다. I/O Binding은 고정 device input/output과 별도 host
출력을 보유한다. CPU는 host input을 매 Run 다시 bind하며 CUDA는 synchronous ORT
Identity copy로 입력/출력을 전송한다. 그 추가 세션 비용도 실험 비용이다. Graph는
CUDA B1+binding만 허용한다. Run/copy/fence 실패로 물리 완료가 불명확한 CUDA
session·input·binding은 quarantine 상태에서 보존한다. 실제 장치 수명·수치·VRAM
검사가 없는 CPU 결과로 해당 경계를 인수하지 않는다.

OPT-12는 **S 실험**이다. virtual visit/loss는 selection에만 적용하며 실제 edge 통계는
검증된 완료의 backup에서만 바꾼다. ticket마다 예약·consume 권한을 분리하고 역순·
중복·일부 취소·전체 취소·오류·drain에서 한 번만 반환한다. 같은 ExecutionId의
다른 요청이 유효하면 독립 소비한다. S1과 동기 S0의 선택/완료 순서·분포가 같다는
가정을 하지 않으며 holdout 대국과 동일 시간/메모리 자원의 인수는 별도다.

## 10. PR #17·#18 총괄 수동 통합 인수

2026-10-04 총괄 TASK-I02는 #17 `b830121`과 #18의 최신 선언을 대조하여 여섯 충돌
파일을 수동 연결했다. 통합 제품 소스는 `a93569bedb802a4eb07f19b12241595715d21df1`이며,
후속 문서 변경과 구별한다. 공통 계약 revision **0.1**과 기본 FP32/full·S0 경로를 유지한다.
#18에 #17의 코드·후속 문서 ancestry를 보존하며 공유 이력을 force-push하지 않는다.

| 접점 | 수동 연결·보완 | 인수 근거와 제한 |
|---|---|---|
| B/C/D → D02 | source journal을 batch owner·prepared input·native Run·출력 변환·guarded backup에 연결 | v1은 기본 B1·fresh Computed만 인수한다. profile+raw/batch/buffer/binding/Graph는 CLI의 asset 로딩 전과 C 경계에서 거부 |
| C 출력 수명 | Run timing과 buffer 반환을 함께 유지하고 소비 전에 raw 변환 결과를 검증 | buffer/binding CPU 수치·반복 출력 보존을 검사했다. CUDA binding/Graph·device fence의 실제 인수는 별도 |
| D 완료 → B/UCI | completion signal·queued batch timer·원래 ProcessClock·새 게임 cache 초기화를 유지 | mock 취소/역순/중복/예약 검사와 실제 CPU 다섯 모드의 합법 착수·종료를 확인 |
| D02 종료 → 파일 | producer join 이후 64KiB 버퍼로 직렬화하고 8MiB cap·flush·sync 오류를 전달 | 첫 10초 종료 실패·부분 JSON을 보존하고 새 실제 CPU 실행의 complete timeline·confirmed drain을 확인 |
| 실행 명세 → native CLI | manifest의 flag/value pair를 Popen 직전에 단일 `--flag=value` argv로 변환 | 공백/등호·누락/중복 flag 검사와 실제 CPU startup/termination binding을 확인. shell을 사용하지 않음 |

현재 통합 소스의 두 OS CI, full witness, 실제 CPU NN·UCI·runner와 원시 보존 식별은
[최적화 기록 12장](research/PERFORMANCE-OPTIMIZATION-PLAN.md)에 묶었다. 이전 GPU 실행은
이번 batch/cache/I/O 옵션의 인수로 재사용하지 않는다. v1 receipt를 실험용 raw-hit/batch
report로 확대하지 않으며, cache 재사용·물리 실행·실제 방문과 S1 품질을 각각 검사한다.

코드 병합은 **#17 → #18 순서로 develop에 Merge commit**을 권고한다. #18은 #17을
조상으로 포함하므로 #17 반영 후 추가 최적화·수동 조정이 남는다. squash/rebase로
조상을 재작성하면 base와 충돌·검사 재사용 자격을 다시 확인한다. 병합 준비를 실제
develop 병합·GPU 성능·정식 대국 인수로 보고하지 않는다.

## 11. BT4 단일 profile의 총괄 수동 접점 대조

2026-10-04 총괄은 공통 revision 0.1을 유지하면서 C asset profile·graph interface·
classical encoder/ordered policy·manifest model ID, C/D admission과 B native bootstrap/
CUDA receipt를 수동으로 맞췄다. Maia exact pin과 16 MiB/두 출력 검사는 유지하고
BT4 exact gzip+protobuf pin·768 MiB/세 출력만 별도 허용했다. MLH는 탐색 미소비다.
`MaiaAsset` 호환 별칭이나 legacy backend 이름을 모델 identity 대신 사용하지 않는다.
BT4 개별 rights 미확인과 redistribution false도 strict manifest의 검증 대상이다.

profile별 CUDA arena/session 선언은 Maia 1 GiB·BT4 3 GiB로 일치시켰다. D의 physical
예약과 별도 session 선언은 실측 VRAM·global hard cap과 구분한다. CPU V1 의미·
CUDA V1 필드 집합·Computed-only B1은 유지한다. default 128과 explicit native
1..4096 simulation, root·worker·취소/완료 순서도 별도 검증했다.

수치 gate는 `aa0cc024`의 actual A/C/D CPU/CUDA 12개, 후속 UCI는 `78b7c535`의
새 release binary·실제 CUDA·guarded root 소비·18 query·confirmed final drain이다.
두 소스 사이 변경은 B UCI 네 파일뿐임을 직접 확인했다. 이 동일성은 수치 영향
경로의 재사용 근거이며 앞 gate가 후속 B 시계/전체 race까지 검사했다는 뜻이 아니다.
전체 workspace tests·strict Clippy·두 OS CI와 모델 연결은 통과했으나 독립 deadline/
stop의 물리 fence, RZ NPS/per-root journal, 정식 paired 강도와 학습은 별도 인수다.
관련 증거와 제한은 [통합 기록](INTEGRATION-STATUS.md)·[BT4 실행 기록](research/LOCAL-MODEL-BASELINE.md)을 따른다.

E의 `NativeCudaProfileV1`은 arena 1 GiB, `NativeArtifactRole::Onnx`는 16 MiB에 고정돼
있고 receipt verifier도 1 GiB를 요구한다. **이번 BT4의 3 GiB/741 MB를 해당 E V1의
통과로 인수하지 않는다.** 기존 profile을 무조건 느슨하게 하지 않았으며 후속에는
BT4 source/export pin·별도 bounds/profile revision·C/D/bootstrap·E 생성/검증·receipt를
총괄이 함께 맞춘 뒤 같은 integration SHA로 검사한다. 이번 개발 대국은 Windows LC0와
WSL native CLI를 직접 관리한 별도 finite runner와 E/A의 사후 PGN 감사 경로다.

## 12. 종료 관측·D02 단계와 별도 최종 착수 정책의 수동 연결

2026-10-05 총괄은 `cf94d07`의 관측 접점, `3ef9171`의 parity/inference 진입점,
`9817647`의 S1과 `fabe88e`의 플랫폼 진단 보완을 같은 브랜치에서 수동 대조했다.
공통 `rz-contracts` revision **0.1**, side-to-move WDL·typed request/generation·deadline,
ordered legal view·model/input identity·Computed/raw-hit·physical lease 계약은 유지한다.

| 실제 연결 | 선언·소비와 마지막 권한 | 인수 범위 |
|---|---|---|
| A exact terminal → B | Rules classification→terminal leaf→기존 final consume guard→Node::Terminal commit | 관측값/NN Q를 종료 권한으로 쓰지 않는다. 방문된 직접 terminal child만 별도 rank; guard 거부/취소/미방문이면 승패 증명 없음 |
| terminal backup → 진단 | `observe_terminal_backups`·승인 후 single-slot observation·`take_terminal_observation` | observer는 path/side/value/selection을 보존하며 새 backup/visit를 만들지 않음. root delta=1·side 부호와 독립 replay 검사 |
| B final policy → 실제 UCI | `FinalMovePolicy`→`EngineSettings`→threaded worker의 첫 selection 전 setter | visits 기본·exact-terminal 명시적 S1, late 변경 거부, policy identity suffix. 동일 binary의 root 통계 동일성과 실제 8 UCI 결과를 각각 확인 |
| B/C/D → D02 세부 단계 | replay·legal authority·input key·terminal backup·final selection source spans→기존 bounded journal/writer | passive unkeyed terminal/final 관측과 keyed NN causal chain 구분. 같은 clock origin·8192 cap·V1 envelope 유지, 누락을 complete로 숨기지 않음 |
| actual root 평가 → parity | D 정상 승인·B root 초기화와 동일 context 이후에만 관측 출력 공개 | C factory key와 별도 input dump key의 일치 확인. 실제 tensor/ordered indices·policy/WDL을 pinned independent original과 대조 |
| native parser/진단 → 플랫폼 | explicit final-selection flag·중복/미지원 값 거부·private-path 없는 bounded Debug | Windows의 unused CUDA path를 presence/digest 관측으로 해결. parser·Debug·UCI 검사와 두 OS exact source CI 성공 |

diagnostic example은 기존 factory/worker/Rules adapter를 재사용하고 B1·simulation 1..4096·
case wall 1..60초·유한 physical shutdown으로 제한한다. 일반 UCI와 별도 example에서
실행한 근거를 구분하며, output 관측은 D/B final acceptance를 대신하지 않는다.
exact-terminal 선택은 전체 minimax solved propagation이나 mate-distance 정보를 만들지 않는다.
진단 example은 고정 1ms polling·전체 outcome 조회를 사용한다. 일반 UCI는 compile feature에
따라 notify/best-move를 선택할 수 있지만 이번 실제 B1 binary에는 둘 다 꺼져 있음을 Cargo
feature와 source branch로 대조했고, 같은 1ms polling 기준을 일반 UCI에서도 계측했다.

S0/S1 probe의 source `9817647`과 후속 실제 UCI/CI source `fabe88e`를 각각 고정했다.
후속 변경은 bounded Debug뿐이고 두 source의 search/Rules/encoding/runtime 경로 일치를
대조했다. 문서 head에서 코드·Cargo·CI 경로의 동일성을 확인한 경우에만 해당 source의
검사를 재사용한다. 자세한 수치·실패·물리 완료와 미측정 범위는
[통합 기록](INTEGRATION-STATUS.md)과 [모델 기록 8장](research/LOCAL-MODEL-BASELINE.md#8-종료-회귀조건-대조b1-계측과-별도-s1)에 둔다.

PR #20의 최신 head에서 총괄이 이 접점과 실제 public 선언을 수동으로 대조했다는 기록을
남긴다. E BT4 artifact/profile 경계는 여전히 별도 연결 대상이다. 같은 이름의 S1이나
similar receipt를 이유로 기존 E V1·정식 강도·physical deadline gate를 자동 호환으로 인수하지 않는다.

## 13. 런타임 저장 capability와 isolated E argv의 수동 연결

총괄은 소스 `55583595a90bb59f611553da103101836fc9cd88`에서 C의 `RuntimeCache`→
`RuntimeLibraryPin`→`OrtRuntime`·native loader, B native bootstrap·두 C 수치 gate,
E의 두 native engine argv를 직접 대조했다. shared `rz-contracts` revision **0.1**과
model/input identity·ordered policy/WDL·generation/deadline·physical lease·backup 계약은
유지한다. `RuntimeStorage`는 별도 저장 영수증이며 `Computed/RawEvalHit` 의미가 아니다.

| 접점 | 총괄 대조·검사 |
|---|---|
| C cache→native capability | 전체 bytes/hash·size·readonly·단일 link, source로부터 분리한 inode, hit마다 새 file pin, exact CUDA 19-file admission/mapping 유지 |
| C storage→B startup | optional absolute cache root 또는 C default slot, model/session의 실제 load 뒤 private-path 없는 storage sidecar, 기존 provider/backend identity·V1 필드 집합 유지 |
| C storage→수치 gate | 기본 cache miss와 explicit root/hit을 새 process로 검사. actual CPU/CUDA·Rules No/Repeat·mapping·물리 종료 각각 확인 |
| C default root→E launcher | 총괄이 단방향 `rz-arena`→feature-less `rz-eval` 의존을 연결하고 lock을 갱신. 새 외부 package 없음. E parent에서 private cache root를 정해 두 inner argv로 전달 |
| E process→native child | `env_clear()`·LANG/PATH 정책 유지, HOME/loader 환경을 전파하지 않음. 환경을 비운 concurrent-process fixture와 explicit root의 실제 CPU/CUDA gate 통과 |
| 저장 관측→E 예산 | C cache는 4 entry/8 GiB로 별도 제한하며 E per-attempt artifact watch와 구분해 invocation limitations에 기록. E source snapshot·권리·pin·profile·receipt 조건 유지 |

새 E launcher에는 cache option을 지원하는 engine binary가 필요하다. 과거 snapshot은
원래 source/launcher로 재현하고 새 option 거부를 성공으로 숨기지 않는다. loader/native
실패 뒤 CPU fallback이나 재시도/unload를 추가하지 않았다. corrupt cache를 덮어쓰거나
활성 pin을 자동 삭제하지 않으며 비정상 종료의 lock/staging은 검증된 owned cleanup 대상이다.

로컬 workspace 727 passed·0 failed·16 ignored, Windows/Ubuntu exact-source CI와 C/B 실제
RTX 4050 재사용·종료 증거를 확인했다. E 연결의 fixture와 실제 C/B 증거를 **새 E 대국의
성공으로 인수하지 않는다.** E의 BT4 artifact/profile·정식 강도·별도 physical race gate는
그대로 남긴다. 세부 결과는 [통합 기록](INTEGRATION-STATUS.md)과
[저장 인수](research/PERFORMANCE-OPTIMIZATION-PLAN.md#13-실행별-native-library-복제-제거와-검증된-공유-저장)에 둔다.


## 14. B03 출력과 C/D 물리 완료의 수동 연결

총괄은 stop/hard deadline의 A 합법 착수 snapshot → B Session/common guard 취소·고정 →
C의 single physical lease·D shutdown → B owned worker 정상 join → protocol 출력 경로를
대조한다. 논리 취소와 물리 완료 사이에는 요청/buffer/pin을 유지한다. 완료를 큐에 넣은
사실이나 isready 응답만으로 물리 종료를 승인하지 않는다.

B의 보류 출력은 common game/root/model/encoding/backend scope에 묶고 받아들인 교체에서
폐기한다. original failed-publication slot은 실제 owner 진단/회수 전까지 유지한다.
완료 이벤트 뒤에도 join 오류·panic을 먼저 수집하며, 실패한 drain을 성공 bestmove로
바꾸지 않는다. 기존 유한 shutdown_limit의 출력 timeout은 final serve 오류로 보존한다.
타이머의 wake는 내부 완료 확인이며 새 탐색/방문/평가 승인이 아니다.

공통 revision 0.1·C eval output·D provenance·B PUCT/terminal policy·native V1 필드 집합은
바꾸지 않았다. standalone Session/transport 검사와 production Engine Owner의 physical
fence를 구분한다. 모델 session은 process lifetime 동안 resident일 수 있고 active
평가 완료와 unload는 다르다. 출력 여유/실제 응답 시간·오류는 대국 runner가 따로 기록한다.

실제 `3d90a03` source의 C/D/B CUDA와 독립 착수·종료·source journal을 인수했다.
후속 소스에서 재사용하려면 test-only/doc 차이와 production 선언·feature·Cargo/CI·asset의
동일성을 대조한다. develop 인수 시에도 최신 head의 직접 선언과 같은 integration SHA의
소비자 검사를 다시 맞추며, E의 BT4 ONNX/bounds/arena/profile을 조용히 기존 Maia V1에
넣지 않는다. 실제 runner/artifact 연결은 별도 후속 인수다.
[상세 인수](research/LOCAL-MODEL-BASELINE.md#9-stop마감-출력의-물리-완료-경계)를 따른다.

## 15. E BT4 V2의 모델·설정·영수증 수동 연결

총괄이 E의 BT4 source 382,645,315 bytes/선정 gzip SHA·ONNX 768 MiB 상한·
3 GiB arena를 C의 selected AssetProfile과 대조한다. V2의 별도 schema/domain과
separate expected model identity를 사용한다. V1 필드·기존 byte 한도·canonical
의미와 공통 계약 0.1은 유지하며 두 형식을 서로 파싱하지 않는다.

| 접점 | 총괄 인수 조건 |
|---|---|
| E 모델 선언→C 로더 | source/export/ONNX 해시·실제 bytes·고정 BT4 source, C model namespace·shape/인코딩·FP32/No·3 GiB admission |
| E search→B | sealed V2 profile의 simulation cap·final selection을 inner argv로 전달. B 실제 served EngineSettings의 별도 receipt에서 대조 |
| B receipt→E | 정확한 startup bytes 해시·process ID에 묶인 탐색 설정, tree/time/worker/batch 한도. provider V1·placement·Computed/root/non-root·drain/exit 검사와 별도로 확인 |
| E 저장→C cache | input snapshot·runtime output·외부 공유 cache를 구분. V2 native output은 네 placement/세 receipt의 최소 19 MiB, unique inputs와 전체 artifact 예산은 별도 |
| E runner→A | 같은 완전 opening과 색 교환의 실제 PGN·시계·cutoff/종료·실패를 검사. A/A 통합 cutoff는 Incomplete, 점수 부여 없음 |

PR/develop 인수 때 최신 선언·source/binary/feature·actual startup/config/end와
같은 integration SHA 소비자를 다시 수동 대조한다. 단일 엔진·동시 상주·
GPU sample·합성 wire 검사를 성공한 E pair나 holdout 기력으로 승격하지 않는다.
실행 예산을 바꾸면 원래 실패와 새 잠금/예산을 모두 보존하고 변경을 명시한다.
V2 실행은 검증 완료 private input snapshot에만 DONTNEED 힌트를 적용한다.
원본·공유 runtime cache의 보존 정책을 바꾸지 않고, descriptor·별도 inode·readonly
권한·hash·실행 후 identity 재검사를 유지한다. 이 힌트의 존재를 실제 RAM 회수나
GPU 개선 근거로 쓰지 않으며, 실제 실패와 같은 자원 조건의 후속 실행을 대조한다.

## 16. E V3 pilot의 최종 선택·전체 시계·실패와 C 메모리 정책

총괄은 V3의 purpose/schema/domain과 source·binary·모델·runtime의 동일성을
대조하고, final selection만 달라지도록 수동 연결한다. 공통 revision 0.1과
CPU/CUDA V1·V2 A/A의 wire/domain·기존 계측 의미는 유지한다.

| 접점 | 직접 대조할 조건 |
|---|---|
| V3 선언→B 설정 | 두 역할 cap 4096·같은 PUCT·clock/resource·actual B search-config와 startup bytes/PID; final selection만 visits/exact-terminal |
| E clock→runner | exact Fastchess source·patch hash/bytes·modified binary·compiler; 초기 추가 증분 0, position 전송 전부터 bestmove, ns 올림·절대 read deadline |
| runner clock→A PGN | 역할·새 게임 reset·각 searched ply의 전후 시계·제시간 증분, 완전 opening·색 교환·A 종료/현재 claim |
| 실패→원장 | Loss PGN 또는 pre-PGN startup loss의 pinned renderer 보존; provider/clock 실패를 유지하고 좋은 판만 집계하지 않음 |
| C 할당→backend | 요청 크기 arena 확장 marker의 새 digest; 모델/FP32/TF32 off/3 GiB·cache provenance 유지 |
| 재실행→새 잠금 | source/engine/backend와 16개 input hash 갱신; 원래 실패·분모를 유지하고 두 시도를 합치지 않음 |

새 C 정책은 BT4 수치·dual-resident 메모리·normal Computed·물리 종료로 확인한다.
한 번의 GPU 메모리 감소를 기존 오류의 물리 VRAM 원인이나 일반적인 속도/기력
개선으로 바꾸지 않는다. old backend의 평가/실행 증거를 새 backend 성공으로
자동 재사용하지 않는다. source별 CPU/CI와 GPU 인수는
[pilot 인수 기록](research/BT4-FINAL-SELECTION-PILOT.md)에서 구분한다.
develop 반영 때 최신 E argv·C backend marker·B startup/search/final·A PGN·clock/failure와
같은 integration SHA의 소비자 검사를 다시 수동 대조한다.

E source `5f40654`는 main private pin의 cache hint를 owned cleanup과 최종 byte/identity
재검사 뒤에도 적용한다. CUDA V2/V3만 대상이며 CPU/CUDA V1·C shared cache·원본·
GPU buffer·공통 revision은 유지한다. child read/posthash가 다시 쌓은 private 모델
페이지를 줄이는 OS 힌트이며 실제 회수·초기 peak·buffer 오류 해결을 자동 인정하지 않는다.
postcheck hint 실패는 정상 인수를 거부하고 원래 실패가 있으면 함께 보존한다.
총괄은 새 E binary와 같은 NN binary/backend, hint phase receipt, byte/inode/cursor 검사,
실제 private page 관측·boot ID·호스트 RAM과 CUDA/clock/A 인수를 별도로 연결한다.

이번 5f40654 실행에서는 여섯 pair의 완전 인수와 pair-06의 시간 예산 중단을 구분했다.
budget failure를 engine loss·startup loss·정상 draw로 재분류하지 않고 raw PGN과
integration false/scored 0을 보존한다. 등록된 16쌍에서 미완료 10쌍을 유지하며 source가
다른 원래 두 실패와 합치지 않는다. code head 43e90fa의 CI와 GPU source 5f40654의 실제
provider/session·clock·A 재생 결과는 별도 증거다. 문서 커밋의 source 영향 없음 확인은
develop의 consumer 검사를 대체하지 않는다. 최신 integration SHA에서 위 접점을 다시
수동 정합하며 Draft/execution_ready/strength_eligible false를 유지한다.

## 17. E V3 2분+1초 시계와 PGN 진단 연결

사용자 지시에 따라 총괄은 같은 BT4·FP32·No·B1·cap 4096의 S0/S1을 경기 전체
120초+착수당 1초로 실행한다. 16쌍·32판과 전체 최대 120분을 사용하며 이전 30초+0.1초
시도와 표본을 합치지 않는다. 같은 opening cohort를 다시 쓰므로 새 미관측 holdout으로
표시하지 않는다. CCRL Blitz의 시간 형식만 참조하며 기준 CPU 보정이나 공식 rating은 없다.

V3 clock은 호출자가 고른 양의 피셔 시계를 받는다. 두 고정 시계의 whitelist는 제거하고
유한 pair wall의 checked 계산을 유지한다. PGN TimeControl 누락·차이는
`NativePilotClockAudit.pgn_time_control_warnings`에 보존하며 실행·인수 거부 조건으로
쓰지 않는다. `expected_pgn_time_control`은 실제 clock에서 만든 진단 값이다.
실제 searched ply 차감·제시간 증분·게임 초기화·역할과 색의 trace 대조는 계속 수행한다.
공통 계약 revision 0.1과 V1/V2는 유지하며 이 필드는 V3 결과 진단의 추가다.

총괄은 E clock → Fastchess `tc=120.000+1.000` argv → actual parent trace → A PGN과
B search-config/물리 종료 receipt를 수동 대조한다. raw White/Black의 RoveZero·BT4·S0/S1
ID를 유지하고 검토용 SAN 사본에 색별 program type·binary/model/source identity를 넣는다.
develop 인수 시 최신 head에서 선언·실제 소비자·동일 integration SHA를 다시 대조한다.
실행 중인 대국은 완료된 강도 인수로 기록하지 않는다.
