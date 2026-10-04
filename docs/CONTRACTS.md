# 상태·평가·캐시·요청 수명 구현 계약

이 문서는 [아키텍처](ARCHITECTURE.md)의 책임 경계를 구현자가 연결할 수 있는
필드·불변식·실패·수락 조건으로 구체화한다. 주 근거는
[RoveZero v0.2.0](reference/RoveZero_Handoff_v0.2.0_KO.md)의 3~4장, 11장,
F14/F15/F21/F22, 17.3절이다. **최소 Rust 계약 revision 0.1은 `rz-contracts`에 게시한다.**
이 문서에는 후속 기능의 의미 명세도 포함되며 모든 타입·ABI·wire schema를 구현했다는
뜻은 아니다. 실제 지원 API·Draft PR별 수동 연결은 [적용 지시서](CONTRACT-ADOPTION.md)를
따른다. 사용자가 선택한 제품 엔진은 독립 Rust 구현이며 신규 자체 코드는
MIT 방침을 따른다. 외부 가중치·데이터·도구의 권리는 각각 확인한다. 필드명은 의미
명세이며 실제 Rust 타입·필드 배치는 구현 시 정한다. 추론 backend·가중치 format·
학습 언어의 실제 연결은 [결정 기록](DECISIONS.md)의 별도 항목이다. 첫 호환 형식은
[선정 기록](WEIGHT-SELECTION.md)의 Maia1 v1.0이며 이 문서의 공통 경계는 그
protobuf·tensor 구조에 종속되지 않는다. 제품 핵심 규칙·검색을
외부 chess 라이브러리나 LC0 UCI 프로세스로 대체하지 않는다. 독립 검증 참조는
별도로 사용할 수 있다. hot path에 검증용 직렬화를 추가하는 것을 요구하지 않는다.

`필수`는 구현이 유지할 의미 계약이다. `도입 기본안`은 이번 지시서에서 제안한
변경 가능한 구현 선택이며, `실행 전 잠금`은 실행 manifest에 실제 값을 정해야 하는
항목이다. R0에서 기존 LC0를 실행할 때는 해당 LC0의 실제 계약을 기록한다. 이
문서의 새 모델 필드로 기준 LC0를 개조하거나 출력 방향을 임의 변경하지 않는다.

PR #18의 OPT-01~12에서는 사용자 배정에 따라 A가 전체 소비 경계를 구현했다.
공통 `rz-contracts` revision 0.1과 byte codec은 유지하며 raw-hit의 fresh receipt,
완료 신호·batch timer, bounded 상태/평가 cache, 실행 모드·physical lease의 실제
API 영향과 소비자 검사는 [적용 지시서 9장](CONTRACT-ADOPTION.md#9-pr-18-전체-최적화의-실제-소비-접점)에 기록한다.
실험 구현이 기본 활성화나 실제 GPU·강도 인수를 뜻하지 않는다.

## 0. 최소 공통 계약의 소유자와 게시 순서

**최소 공통 계약과 `crates/rz-contracts`의 논리 소유자는 총괄로 배정된 에이전트다.** 이 문서의
초안 작성 위임은 계약 소유권 이전이 아니다. TASK-A03은 총괄의 계약을 Position과
legal move에 적용하고 연결 검증을 수행한다. 작업자 A가 공통 계약을 독자 결정하는
단계가 아니다. 다른 작업자는 필드·버전·ID·오류·설정 의미를 독립 수정하지 않고
총괄에 변경안·호환 영향·검증 범위를 전달한다. 합의된 routine 변경에 사용자 승인을
매번 추가 요구하지 않는다. 2026-10-02 초기 계약 문서 작성에서는 실제 crates를
생성하거나 구현·실행하지 않았다. 이후 실제 구현은 배정한 TASK 범위에서 진행하고
문서 계약과 실제 타입·검사 상태를 구분한다.

총괄은 CPU mock으로도 소비 가능한 최소 v0 의미 계약을 먼저 게시한다. 여기에
ID/generation/deadline, 상태/합법 수 view의 식별·수명, Move 의미와 ordered legal
policy mapping, side-to-move WDL, EvalRequest/Result status, 공통 오류와 유한 예산의
의미가 포함된다. 구체 encoder·framework tensor·GPU stream 타입은 최소 계약에
노출하지 않는다. 이후 해당 계약을 사용하는 B의 search/UCI, C의 encoding/eval,
D의 runtime, E의 arena를 병렬 연결할 수 있다. CPU mock의 계약 통과와 실제
신경망·arena 성과의 증거는 별도로 유지한다.

A~F의 클라우드 개발 환경에는 GPU가 없을 수 있다. 같은 evaluator 경계로 CPU/mock와
가능한 CPU 신경망 참조를 명시적으로 선택하며 backend ID·실행 환경·capability·
검사 범위를 기록한다. GPU 요구 모드에 CUDA capability가 없으면 명시적인
`BackendUnavailable`로 보고하고 CPU 성공으로 숨기지 않는다. GPU 검사의 skip·
미실행은 GPU 계약 통과가 아니다. backend 코드·수치 fixture·검사 진입점은 먼저
개발할 수 있으며 실제 목표 GPU 수치·buffer 수명·자원 인수는 총괄 I02가 별도로 확인한다.

| 계획 경로 | 계약 소유와 의존 방향 |
|---|---|
| `crates/rz-contracts` | 총괄 소유의 shared primitives/errors/config/request boundary. `rz-position`, evaluator, runtime에 의존하지 않음 |
| `crates/rz-position` | A 소유의 실제 board/state/attacks/legal/make-unmake. 필요 shared primitives만 사용하며 dependency cycle 금지 |
| `crates/rz-encoding` | state/계약을 읽어 선택 모델 input/delta 구성. scheduler나 Search 통계에 의존하지 않음 |
| `crates/rz-eval` | 공통 evaluator 경계·encoding과 선택 backend를 구현. search 방문을 변경하지 않음 |
| `crates/rz-search` | position·공통 계약만으로 selection/backup. concrete runtime/eval backend에 직접 결합하지 않음 |
| `crates/rz-runtime` | 공통 evaluator 인터페이스를 통해 queue/batch/cancel 수행. 규칙 구현·Search 선택을 소유하지 않음 |
| `crates/rz-uci` | position/search/runtime의 연결과 protocol. UCI 엔진 프로세스를 neural evaluator API로 쓰지 않음 |
| `crates/rz-telemetry` | 수동적인 이벤트/metric 계약. telemetry에서 Search/Runtime을 제어하거나 역방향 cycle을 만들지 않음 |
| `crates/rz-arena` | 외부 runner·시계·독립 결과 검증. 자체 규칙 코드 복제 대신 검증 경계의 position/helper 재사용 |

최소 계약 v0 게시 이후 변경은 총괄이 영향 consumer·호환 정책·실제 compile/test
범위를 확인해 version을 갱신한다. A의 PositionState 내부 representation을 공통
crate로 모두 옮기지 않는다. `PositionHandle/StateIdentity/LegalMoveView` 같은
공통 경계로 표현해 concrete board 소유권과 shared API의 의존 방향을 보존한다.
Rust async 경계에는 request보다 짧게 사는 mutable board borrow를 넘기지 않는다.
owned snapshot/참조 계수 snapshot/수명이 고정된 pool pin 중 기반에 맞는 방식을
선택하고, `Send/Sync` 및 device borrow의 조건을 실제 구현으로 확인한다.

## 1. 구현자가 먼저 잠글 항목

다음 항목이 미결정이어도 문서·독립 계약 검토는 진행한다. 실제 연결·대국 전에
해당 항목의 값을 결정하며, 기본안 채택과 사용자 선택을 구분해 기록한다.

| 항목 | 도입 기본안 또는 결정 위치 | 잠금 시점 |
|---|---|---|
| 규칙·입력 이력 | 표준 체스, Rules 단일 소유자, FEN 이전 이력 `unknown` 보존 | 규칙/arena 연결 전 |
| 무승부 claim·자동 종료 | 정확한 규칙과 arena 정책을 별도 식별한다. claim 선택·자동 수락 정책은 미결정 | 첫 대국 전 |
| 평가 관점 | 새 평가 인터페이스는 `side_to_move`를 기본안으로 한다 | 첫 모델 연결 전 |
| 행동 인코딩 | `from/to/promotion`의 의미는 필수. 텐서 index와 move 순서는 선택 모델 계약에 고정 | encoder 연결 전 |
| 입력 history·fill | 필요한 길이·방향·알려지지 않은 history의 mask/fill을 encoding manifest에 고정 | cache 도입 전 |
| 정밀도·오차 | 실행 dtype, accumulator, parity 허용 오차와 확률 검증 오차를 별도로 수치화 | backend 비교 전 |
| 반복·fresh 경로 | 선택 모델의 완전 계산 및 요청 예산의 의미를 고정 | 반복/warm 실험 전 |
| 큐·메모리·종료 | 최대 요청·배치·host/device bytes·kernel 단위와 drain timeout을 유한 값으로 설정 | scheduler 실행 전 |

`unknown` 이력으로부터 반복 횟수나 학습 입력의 실제 과거 보드를 만들어내지
않는다. 알려진 prefix만으로 가능한 판정과 전체 이력이 있어야 가능한 판정을
구분한다. history 불충분 상태를 수락하는 분석 모드와 대국의 완전 시작 상태
계약은 [실험 규약](EXPERIMENTS.md)에서 구분한다.

## 2. 식별자·버전·검증 위치

도입 기본안의 정수 폭은 아래와 같다. 외부 직렬화는 schema version과 범위를
검증하고, 런타임 내부 표현은 기반 언어·엔진에 맞게 선택한다. 포화·wrap-around를
정상 재사용으로 처리하지 않는다. overflow는 명시 실패 또는 새 epoch 발급 사유다.

| 필드/타입 | 의미와 기본안 | 유효 범위·수명 |
|---|---|---|
| `SchemaVersion` | 계약별 `major/minor`, 각각 unsigned 16-bit | 알 수 없는 major는 거부. minor도 사용 필드의 호환성 확인 |
| `ModelHandle` / `EncodingHandle` | 검증 완료 불변 manifest를 가리키는 타입이 다른 opaque handle | owner와 registry generation 포함. evict/unload 후 재사용 거부 |
| `PositionHandle` / `LegalMovesHandle` | Rules가 만든 불변 상태·move view의 handle | state revision과 owner 검증. mutable 보드의 무기한 borrow 금지 |
| `RequestId` / `SelectionId` | 논리 subscriber 평가 요청 / Search 경로 선택의 ID. 각각 프로세스 epoch 안에서 단조 증가하는 unsigned 64-bit | 재시도는 새 RequestId. 같은 selection의 중복 backup은 금지 |
| `ExecutionId` | 필요 시 Scheduler가 발급하는 물리 backend 작업의 unsigned 64-bit ID | 여러 RequestId가 dedup으로 하나의 ExecutionId를 공유할 수 있음. Search selection/terminal ID 대체 금지 |
| `GameGeneration` / `RootGeneration` | newgame와 새로운 root/search session의 unsigned 64-bit 세대 | 모델 세대와 별도. 착수/stop/newgame 때 무효화 |
| `SlotGeneration` | cache/buffer slot 재사용을 감지하는 unsigned 64-bit 세대 | slot 번호만으로 과거 handle을 수락하지 않는다 |
| `MonotonicTick` | 프로세스의 단조 clock tick을 나타내는 unsigned 64-bit | 시간 단위·clock owner 고정. UTC 시각은 로그용이며 deadline 계산에 사용하지 않음 |
| `ByteCount` / `IterationCount` | byte 수 unsigned 64-bit / 반복 수 unsigned 32-bit | 할당·덧셈·곱셈 전 overflow 및 실행 상한 검증 |
| artifact digest | 가중치·설정·인코딩 파일의 SHA-256 등 지정한 digest | 입력 출처 식별용. hash 일치만으로 권리·정확성·ABI 안전성 인정 금지 |

모델 load 경계에서 artifact digest, tensor shape/dtype, action map, history 정책,
buffer 요구량, supported backend/precision을 검증해 불변 handle을 발급한다.
모든 평가마다 가중치 전체나 manifest JSON을 다시 hash하지 않는다. 상태는 Rules가
checked transition과 증분 식별자를 관리하고, 평가 경계는 handle의 세대·revision·
필요 옵션만 확인한다. 외부 FEN·PGN·모델·cache file은 별도 외부 경계에서 파싱·검증한다.

검색·평가 key의 빠른 hash는 충돌 가능성을 가진다. 충돌 시 실제 식별 조건의
tag/구조 비교 또는 충돌을 안전하게 miss로 만드는 정책이 필요하다. 직렬화용 digest와
hot path key를 같은 비용으로 구현할 필요는 없다. `hash 같음`만으로 다른 상태를
같은 입력이라고 단정하지 않는다.

## 3. PositionState와 종료 판정

`PositionState`의 단일 작성자는 Rules다. Search/Encoder/Arena는 검증된 상태를
읽거나 Rules의 전이 API를 호출한다. 모델 잠재 상태는 이 타입의 대체물이 아니다.

| 필드 | 필수 의미 |
|---|---|
| `rulesVersion`, `variant` | 규칙 구현 version과 `standard_chess`. 지원하지 않는 변형 거부 |
| `board` | 64칸의 실제 기물 종류·색·위치. square 번호 방향은 Rules 계약에 명시 |
| `sideToMove` | 실제 착수할 색. 모델 canonicalization과 별도 |
| `castlingRights` | 실제 잔여 권리. 기물 배치에서 다시 추정하지 않음 |
| `enPassantState` | 외부 FEN의 target 표현과 Rules가 해석한 합법 가능성을 구별. hash 목적별 의미 명시 |
| `halfmoveClock`, `fullmoveNumber` | 원본 규칙 카운터. halfmove를 반복 횟수나 모델 입력 길이로 대체하지 않음 |
| `repetitionHistory` | 반복 판정에 필요한 알려진 상태 이력/카운트와 completeness. 반복 등가 key는 평가 key와 별도 |
| `modelHistory` | encoder가 요구하는 알려진 과거 상태 또는 읽기 handle. 길이·정렬·fill은 encoding 정책 |
| `historyCompleteness`, `historyOrigin` | `complete` 또는 `unknown_prefix`, origin은 startpos+move trace / FEN 등 |
| `revision`, `positionIdentity` | checked transition의 revision과 해당 상태 식별. board hash 하나로 모든 의미를 대체하지 않음 |

Rust 규칙 계층의 구현 순서는 board·FEN 검증 → attacks → pseudo-legal moves →
왕 안전 필터를 포함한 legal moves → checked make/unmake다. attacks는 pseudo/legal
move 배열의 단순 재사용으로 대체하지 않는다. pinned piece, king 이동, castling
통과 칸, en passant로 열린 공격선은 각각 규칙의 의미에 맞게 처리한다. 정확한
특수 수 전이·king 안전을 검증한 뒤에 delta feature를 연결한다. 자신의 legal move와
동일 함수로 생성한 기대값만 비교하는 검사를 독립 규칙 검증으로 표현하지 않는다.

규칙 카운터는 도입 기본안으로 unsigned 32-bit 범위를 checked parsing한다. 입력
크기·최대 trace length도 실행 설정에서 제한한다. 잘린 이력을 `complete`로 표시하거나
자동으로 카운터를 0으로 바꾸지 않는다. encoder의 history-fill은 합의된 계산 정책이며
실제 대국 이력으로 기록하지 않는다.

Rules의 `classifyPosition`은 유효 상태의 classification과 invalid input을 구분한
`Result<PositionClassification, RuleInputError>` 의미 계약이다. `PositionClassification`은
아래 **서로 독립적인 필드**를 가진다. ongoing·claim 가능·불완전 이력이 함께 존재할
수 있으므로 이들을 하나의 배타 enum의 선택지로 만들지 않는다.

- `playStatus`: `ongoing` 또는 `terminal(reason, winner_or_draw)`.
  terminal은 checkmate/stalemate/dead position/`automaticDraw(reason)` 등이며
  판정 근거와 우선순위를 보존한다.
- `claimAvailability`: 현재 상태 또는 예정 합법 수의 claim 근거 목록.
  `threefoldClaim/fiftyMoveClaim`을 구별하고 관련 항목에 대해 `available/unavailable/
  unknown`을 표현한다. ongoing에서 claim이 available이어도 자동 terminal로 바꾸지
  않는다. arena의 claim 결정과 연결하며 신경망 WDL=draw로 곧바로 바꾸지 않는다.
- `historyEvidence`: 전체 이력 또는 unknown prefix, 해당 판정에 필요한 이력의
  충족 여부·부족 이유. unrelated 과거가 unknown이어도 확정 가능한 checkmate 등
  playStatus를 무조건 unknown으로 바꾸지 않는다.

모순 입력·미지원 상태는 `RuleInputError`로 반환하며 빈 policy·무승부로 숨기지
않는다. 판정에 필요한 과거가 부족하면 해당 claim/repetition evidence가 unknown이며
정상 진행 가능 여부와 별도로 기록한다.

FIDE 9.2/9.3의 3회 반복과 50수는 claim 조건이며, 9.6의 5회 반복과 75수는
자동 종료 조건이다. 엔진 halfmove 단위로 이미 완료된 50수/75수 조건은 각각
100/150 ply다. claim은 해당 조건을 만들 예정인 합법 수에 의해서도 가능하므로
`claimAvailability`의 evidence는 `current_position`과 `intended_move(move)`를 구분한다.
75수에 도달한 마지막 수가 checkmate이면 mate 판정이 우선한다. arena의 claim
자동 수락/engine 의사표시 정책은 첫 대국 전에 별도 잠근다.
[FIDE Laws 9.2~9.6](https://handbook.fide.com/chapter/e012023)

반복 등가 상태는 같은 차례·기물 배치·가능한 이동과 관련 권리를 보존한다.
앙파상 target 문자열 존재만으로 다른 반복 상태라고 단정하지 않고 실제 capture
가능성을 규칙 계층에서 해석한다. 반면 모델 입력이 raw EP 표현을 사용하면 그
표현은 평가 input identity에 별도로 들어간다.
[FIDE Laws 9.2.3](https://handbook.fide.com/chapter/e012023)

dead position은 어느 쪽도 가능한 합법 수의 연속으로 상대 왕을 mate할 수 없는
상태다. 단순 material shortcut은 증명 가능한 subset만 exact로 인정하며 완전한
dead-position oracle을 구현했다고 주장하지 않는다. 이력 부족·지원하지 않는
판정의 처리 범위를 규칙/arena manifest와 수락 검사에 명시한다.
[FIDE Laws 5.2.2](https://handbook.fide.com/chapter/e012023)

합법 수가 0인 상태의 checkmate/stalemate 구분, 반복 및 카운터 기반 판정,
불가능한 입력 처리는 Rules가 맡는다. mate distance나 contempt/draw utility 같은
Search의 척도는 Rules의 결과와 별도 타입이다. exact terminal outcome을 평가 queue에
넣기 전에 처리하며, 휴리스틱 draw 추정은 정확한 종료 근거로 쓰지 않는다.

## 4. MoveDelta와 orderedLegalMoves

`Move`의 의미 identity는 최소 `fromSquare`, `toSquare`, `promotionKind`다.
`promotionKind`는 `none/queen/rook/bishop/knight`를 구별한다. 모델 action index는
이 의미 identity의 versioned mapping이며, 같은 from/to의 서로 다른 승격을 합치지 않는다.
square 기본안은 a1=0, h8=63의 rank-major지만, 채택 기반의 다른 numbering은
adapter에서 명시 변환한다. 이를 핸드오프의 확정 인코딩으로 보고하지 않는다.

`MoveDelta`는 의미상 두 계층으로 구성한다. Rules의 checked make가 만든
`RuleMoveDelta`는 모델 독립적인 기물·상태 변경과 undo 근거만 소유한다. Encoder가
이를 소비해 `FeatureDelta`를 구성하며 인코딩·특징 dependency를 결합한다.
`rz-position`은 모델/encoder를 알아야 할 의존이 없으며 `rz-contracts`도 두 concrete
구현에 의존하지 않는다. 출발/도착 칸 2개만 기록하는 shortcut은 특수 수를 충족하지 않는다.

| 소유 타입 | 필드 | 계약 |
|---|---|---|
| `RuleMoveDelta` | `beforeIdentity`, `afterIdentity`, `move` | delta를 적용할 정확한 부모/자식과 실제 합법 move. 다른 부모 적용 거부 |
| `RuleMoveDelta` | `moveKind` | normal / capture / castling / en_passant / promotion의 조합 의미. promotion capture 허용 |
| `RuleMoveDelta` | `pieceRemovals`, `pieceAdditions` | 이동·포획·rook 이동·승격된 기물의 실제 칸·색·종류. en passant capture square 포함 |
| `RuleMoveDelta` | `old/newGlobalState` | 차례·캐슬링·앙파상·카운터 및 실제 상태 이력 변화. 모델 history-fill은 여기서 수행하지 않음 |
| `RuleMoveDelta` | `undoToken` | Rules가 소유하는 역전이 정보. Encoder latent 역변환을 Rules unmake로 인정하지 않음 |
| `FeatureDelta` | `ruleDeltaRef`, `encodingHandle`, `deltaVersion` | 실제 RuleMoveDelta 근거와 특징 해석의 인코딩·버전. 다른 인코딩 delta 재사용 거부 |
| `FeatureDelta` | `featureDependencies`, `removals/additions` | 해당 encoder의 특징 변경과 무효화 영역. king bucket·전역 관계·history-fill에 따른 재계산 사유 포함 |

증분 특징이 정확하게 갱신되는 영역과 전체 재인코딩이 필요한 영역을 explicit하게
표시한다. king bucket 경계, 복수 threat 변화, 특수 수, 긴 make/unmake trace에서
같은 모델의 독립 full 경로와 비교한다. 재계산은 정상 경로의 일부일 수 있으며
`reason/bytes/time`을 기록한다.

`orderedLegalMoves`는 Rules가 상태 revision에 귀속한 중복 없는 합법 move 배열과
`orderIdentity`를 제공한다. 모델별 `actionMapVersion`은 C의 Encoder/Eval adapter와
ModelManifest가 소유하며 Rules의 legal 배열에 모델 의존 필드를 넣지 않는다.
공통 계약은 순서/handle 식별을 제공하고 adapter가 합법 move와 해당 action map의
호환·gather/scatter를 검증한다. 길이는 실행 설정의 유한 상한으로
검사하되 표준 체스 전체의 최대 수를 임의로 작은 수로 가정하지 않는다. LegalMoves
handle은 immutable이다. 배치를 위해 padding한 원소는 실제 move가 아니며 mask로
제외한다. 요청 순서와 결과 정책을 맞추는 책임은 evaluator adapter다.

도입 기본안은 의미 move의 `(from, to, promotion)` 순서다. 기존 모델의 고정 action
space를 사용하면 그 mapping을 선언하고 gather/scatter 결과를 검증한다. 수 순서가
바뀌면 policy는 같은 move별 값으로 permutation되어야 하고 state value는 허용 오차
안에서 같아야 한다. 후보 순서를 네트워크 입력의 의미 특징으로 쓰는 모델은 이
기본 계약과 다른 A형 설계이므로 별도 명시한다.

## 5. ModelManifest와 계산 계약

manifest는 로드 이후 불변이다. 파일 version/digest뿐 아니라 출력 해석과 계산 경로를
식별해야 한다. 아래는 필수 의미 필드다. 첫 가중치 format은 선정 기록을 따르고,
machine-readable manifest의 직렬화 형식과 구체 실행 프레임워크는 연결 시 잠근다.

| 영역 | 필드·의미 |
|---|---|
| 식별·출처 | `schemaVersion`, logical model ID, architecture/config digest, weights digest, 공개 출처·권리 참조 |
| 입력 | encoding ID/version, tensor 이름·shape 범위·dtype·layout, board 관점·history length/fill/mask, 합법 수/action map version |
| 출력 | policy logits 또는 probabilities의 구분, action axis, WDL order·viewpoint, optional Q/uncertainty의 단위·의미 |
| 수치 | storage/compute/accumulator precision, dequant 설정, normalization, output admissibility 및 fresh/full parity tolerance |
| 반복 | 지원 step 범위, 최소/최대 반복, `full`의 정의, warm-start state version/shape, 종료 신호 해석 |
| 실행 | backend/build/kernel 계약 ID, dynamic shape/batch 범위, 필요 workspace bytes, fallback 지원 여부 |
| 호환·수명 | model registry generation, feature/raw/latent cache schema, 타 모델·encoding과의 비호환 조건 |

`full`은 선택 모델에서 정한 완전 계산 예산이다. `fresh`는 부모 latent/근사 cache 없이
해당 입력을 새로 계산한 경로다. 둘은 같은 뜻이 아니다. fresh에서도 2-step의 저예산
평가를 할 수 있고 warm에서 full step을 할 수 있다. 요청과 결과는 두 축을 각각 기록한다.

기존 pretrained net의 인코딩·head 변경은 weight compatibility를 재검토한다.
가중치가 맞지 않는데 load 실패를 무시하거나 tokenizer만 바꿨다고 보고하지 않는다.
H의 CPU helper가 다른 모델이면 별도 manifest/척도/calibration/cache를 쓴다.

## 6. EvalRequest와 EvalResult

`EvalRequest`는 Search가 선택한 평가와 Runtime이 수행할 계산의 계약이다.
터미널 판정은 먼저 수행하며, 비정상 입력은 요청 생성 경계에서 거부한다.

| EvalRequest 필드 | 필수 의미 |
|---|---|
| `requestId`, `selectionId` | 논리 subscriber 평가 요청과 선택 경로 식별을 구분. dedup 시 각 subscriber는 고유 RequestId와 해당 selection 보유. 물리 작업은 별도 ExecutionId |
| `gameGeneration`, `rootGeneration` | 수신 당시의 Search 소유권 세대 |
| `position`, `orderedLegalMoves` | immutable checked handle과 일치하는 revision/orderIdentity |
| `model`, `encoding` | prevalidated handle. arbitrary file path를 hot path에서 재로드하지 않음 |
| `precisionProfile`, `computeBudget` | 실제 정밀도와 min/max steps, 실행 mode `fresh/warm`, full 요구 여부 |
| `deadline`, `cancelToken` | 자기 시간의 monotonic 최종 기한과 owner가 취소하는 token |
| `parentCacheHandle` | 선택적 exact feature 또는 approximate latent를 타입으로 구별. 부모 identity/encoding 확인 |
| `byteBudget`, `priority` | 이미 승인된 budget reservation/queue priority. priority는 move pruning 또는 유효 방문이 아님 |
| `provenanceFlags` | fresh audit/speculative/helper 등 계산 목적. 원인별 비용·소비 여부 기록 |

`EvalResult`는 tagged outcome이다. 성공 payload는 아래 필드를 포함한다.
`failed/canceled/expired/stale`는 빈 success payload 대신 원인과 실제 실행 정보를 보존한다.

| EvalResult 필드 | 필수 의미 |
|---|---|
| request·세대·입력 | 같은 request ID, model/encoding handle, state revision, move orderIdentity, game/root generation |
| `policy` | 요청 합법 move 순서의 확률. logits 반환 backend는 검증된 adapter에서 변환 |
| `wdl`, `viewpoint` | W/D/L의 유한 확률과 선언한 관점. 도입 기본안은 실제 차례 기준 |
| `actualCompute` | 실제 dtype/정밀도, steps, fresh/warm/full 여부, backend 실행 계약 |
| `cacheProvenance` | computed / exact feature reused / raw eval hit / approximate warm. 부모 handle·경로 정보를 필요 시 포함 |
| 선택 출력 | Q, uncertainty, error estimate 각각 의미·단위·calibration version. 미제공은 absent이며 0으로 위장하지 않음 |
| `timing` | queue/compute/transfer/finish tick과 전체 소요. CPU helper 비용도 별도 누적 |
| `completion` | 수치 검증 완료 여부, normal/partial 여부, failure stage/code/recovery |

요청의 terminal outcome·deadline·취소는 각 논리 RequestId에 귀속한다. Scheduler는
dedup 물리 작업을 별도 ExecutionId로 추적할 수 있으며 telemetry의 backend 완료는
각 subscriber의 완료를 의미하지 않는다. 한 subscriber 취소와 공유 device 작업 종료를
구분한다. 동일 selection이 재요청으로 여러 RequestId를 갖게 되면 Search가
selection consume token으로 backup을 한 번만 허용한다.
새 backend 실행이 없는 cache hit는 새 ExecutionId가 없을 수 있다. 과거 cache
계산의 provenance ID를 현재 물리 실행이 발생한 것으로 세지 않는다.

`completed`는 요청 계약을 충족한 성공 평가다. min step 미달, 일부 head만 완료,
deadline 밖 결과는 정상 success로 소비하지 않는다. partial output을 refinement 힌트로
쓸 연구는 별도 tagged 타입·수치 계약으로 명시하고 full raw cache에 쓰지 않는다.

WDL은 확률 범위와 합 1을 검사하고 policy도 각 유한값·음수·정규화·합법 대응을
검사한다. 성공에 필요한 head가 NaN/Inf이면 해당 평가 전체가 실패다. 작은 수치
반올림 허용 범위와 normalization 방법은 manifest에 명시한다. 그 범위를 벗어난
값을 clip·uniform policy·WDL=(0,1,0)로 조용히 수선하지 않는다. terminal leaf는
Rules의 결과를 별도 처리하므로 합법 수 없는 상태의 policy normalization을 수행하지 않는다.

side-to-move WDL을 부모 관점으로 backup할 때, 정확히 한 실제 착수 뒤에는
`(W,D,L) → (L,D,W)`이며 scalar `v=W-L`은 부호가 바뀐다. 깊이가 다른 경로는
실제 ply와 관점 metadata로 변환한다. `D`의 utility, mate 값, centipawn score는 별도
Search 설정이며 단위 변환 없이 신경망 WDL에 더하지 않는다.

## 7. 수학적 재사용·정밀도·독립 full 경로

E형 exact reuse는 **같은 모델 수학식과 선언한 입력·계산 계약**을 보존한다는
의미다. floating-point operation 순서 변경이 bit equality를 보장하는 것은 아니다.
`bitwise`가 필요한 항목과 수치 parity가 필요한 항목을 나눈다.

| 검증 종류 | 요구 |
|---|---|
| Rules·move·버전·정수 카운터 | 의미상 정확 일치. 수치 tolerance를 사용하지 않는다 |
| FP feature/full parity | tensor별 `max_abs`와 `max_rel` 기준, rel 분모의 작은 값 floor를 실행 전 잠금 |
| policy/WDL admissibility | probability range, 합 오차, normalization 규칙을 precision별 수치화 |
| policy/WDL fresh-vs-reuse | move별 max error, WDL error와 필요한 rank/유일 방어 수 회귀 기준 |
| A형 압축·양자화·warm | above 오차 외에 P99/최대 오차·경로 편향·전술·동일 시간 강도 검증 |
| byte 동일성 요구 영역 | 직렬화 fixture 등 필요 영역만 bitwise. 전체 GPU 계산의 무조건 bitwise 요구 금지 |

선언된 threshold 없는 parity 결과는 `측정값`이며 pass가 아니다. FP32/FP16/BF16/
정수 양자화에 동일한 임계값을 자동 적용하지 않는다. model/backend가 확정되기 전
실측 근거 없이 임계값을 정밀도 성능의 확정치로 만들지 않는다. 최초 작은 calibration
세트에서 수치 오차 한도를 정하되 최종 holdout 성과를 보고 느슨하게 바꾸지 않는다.

근사 warm-start 경로를 구현하기 전에 같은 자체 모델의 fresh/full 경로를 실행
가능하게 한다. 독립 full encoder는 delta update 함수의 재호출이 아니라 현재 checked
state에서 전체 특징을 구성한다. 다른 엔진의 점수·교사 호출은 동일 모델 fresh 복구가 아니다.

## 8. CacheIdentity와 namespace

`CacheIdentity`는 재사용하려는 **값의 의미**와 **handle의 수명**을 나눠 표현한다.
전체 history를 매 요청 직렬화하지 않고 validated input identity, model/encoding handle,
정밀도·budget tag를 사용한다. key 구성의 충분성을 테스트하며 필요 입력을 누락하지 않는다.

| namespace | semantic identity와 payload | 금지 |
|---|---|---|
| `ExactFeature` | 실제 encoder 입력 identity, encoding/model 의존성, precision/feature version. payload는 full과 같은 특징 | approximate latent를 동일 타입으로 저장 |
| `ApproxLatent` | 위 조건 + 부모/경로/갱신 version/steps/precision/나이. payload는 경로 의존 초기 상태 | board hash로 경로 독립 exact dedup |
| `RawEval` | 실제 모델 입력·model weights/config·encoding·precision/compute contract·fresh budget. payload는 불변 raw policy/WDL | correction·Search Q·TT bound·다른 helper 평가와 혼합 |
| `CorrectedEval` | raw reference + game generation + correction table version + 관점/단위. 기본안은 계산된 view로 사용 | 수정 값을 RawEval에 덮어쓰기 |
| `SearchTT` | 규칙 상태/이력, 검색 version·전제·깊이·bound·score unit·generation 등 검색 계약 | heuristic lower/upper를 정확 게임이론적 결과·교사 라벨로 선언 |
| `SearchEdgeStats` | root/tree owner, parent edge identity, selection/backup accounting. visits/value/in-flight | raw eval hit·sibling bonus를 실제 완료 방문으로 변경 |

RawEval의 model identity는 가중치뿐 아니라 architecture/config를 포함한다. encoding은
history-fill·관점·mask·action mapping을 포함한다. precision identity는 저장 dtype만 아니라
실제 compute/accumulator/dequant 설정을 포함한다. 반복 수·종료 정책이 출력을 바꾸면
budget/actual compute contract도 달라진다. 같은 board라도 이 항목이 다르면 miss다.

합법 수 순서는 배열 payload의 `LegalOrderIdentity`에 포함한다. raw 출력이 고정 action
space로 저장되어 있어도 C의 Encoder/Eval adapter가 요청 순서로 gather할 때
ModelManifest의 actionMapVersion과 Rules legal view의 orderIdentity를 결합·검증한다.
다른 순서의 정책 배열을 index 그대로 돌려주지 않는다. 규칙의 반복 key와
모델 입력 key는 각각 필요한 정보가 달라 별도 목적의 타입으로 둔다.

세대 처리의 기본안은 다음과 같다.

- 모델/encoding handle과 cache slot은 owner 세대를 검증한다. unload·eviction 뒤
  같은 번호로 새 항목을 발급해도 과거 handle은 실패한다.
- `rootGeneration`은 요청·reservation·SearchEdgeStats의 수락 조건이다. 불변 raw
  evaluation의 수학적 의미를 바꾸는 입력은 아니므로 모든 raw key에 붙여 무조건
  중복 계산할 필요는 없다. 새 root에서도 재사용하려면 입력·계산 계약과 게임 범위가 맞아야 한다.
- 기본 RawEval 수명은 한 게임 안의 이미 유효하게 완료된 계산이다. `ucinewgame`은
  history/correction/Search를 초기화하고 기본 game namespace를 교체한다. 게임 간
  평가 cache 유지가 필요하면 두 엔진에 같은 명시 정책을 잠그고 opening 누출을 검토한다.
- 착수/stop으로 무효화된 요청이 늦게 만든 값은 새 root나 raw cache에 게시하지
  않는 것을 기본안으로 한다. 이전 세대에서 이미 유효하게 게시한 불변 raw 값과 구별한다.

warm output을 RawEval에 저장하는 것은 기본안에서 금지한다. 경로 독립성과 full
계산 계약을 검증한 다른 설계를 채택하려면 A/E 분류·identity·수치 근거·실패 복구를
별도 결정한다. cache eviction/miss는 정보 없음이며 WDL=0·무승부나 모델 전환이 아니다.

## 9. 평가 hit와 유효 탐색 방문

`eval_hit`, `network_execution`, `leaf_selection`, `accepted_backup`, `completed_visit`을
서로 다른 카운터로 둔다. cache lookup 자체에는 Search 방문을 늘릴 권한이 없다.

새 Search selection이 실제로 선택한 경로에서, 적합한 cached 평가를 받아 같은
경로에 정상 backup하는 것은 알고리즘이 허용하면 유효 방문이 될 수 있다. 이때
방문은 **새로운 selection·경로·수락된 backup**에서 생긴다. 새 신경망 계산으로
세지 않으며, 같은 selection의 재조회·재전달·UI 표시·duplicate callback은 방문을 늘리지 않는다.

평가 dedup으로 하나의 실행에 여러 subscriber가 붙어도 각 subscriber는 독립적인
selection token과 경로를 보유한다. 동일한 경로를 실수로 중복 subscriber로 등록한 경우
동일 selection ID로 검출한다. 알고리즘이 한 leaf에 여러 selection을 허용하는지·
충돌을 병합하는지는 Search 설정이며 actual network work와 구분해 기록한다.

다른 parent edge는 `N`, `W`, `Q`, `virtualLoss`, `inFlight`를 공유하지 않는다.
정확 eval dedup부터 구현하고, DAG 자체를 도입할 때만 cycle/반복·참조 수명·GC·
동시 backup·root 이동을 별도 S형 변경으로 다룬다. 보드가 같다는 이유로 반복 이력과
경로별 종료 판정을 합치지 않는다. sibling 정보/FPU/history bonus는 추정치이며
완료 방문이나 terminal 증거로 기록하지 않는다.

## 10. 요청 수명과 race 처리

Scheduler는 물리 실행과 queue/buffer를 소유하고 Search는 reservation·경로·backup을
소유한다. GPU kernel이 즉시 중단되지 않아도 논리 취소는 가능하다. 물리 실행 완료와
탐색에 결과를 반영할 권한을 같은 flag로 표현하지 않는다.

| 논리 상태 | 진입 조건·허용 다음 상태 |
|---|---|
| `queued` | admission/budget 성공. `running/completed/canceled/expired/stale/failed`로 이동 가능. cache hit는 running을 생략할 수 있음 |
| `running` | 물리 실행 또는 dedup subscriber 대기. `completed/canceled/expired/stale/failed`로 이동 |
| `completed` | 모든 요청 조건·수치·세대·기한을 finalization에서 확인한 성공. terminal |
| `canceled` | owner 취소가 logical finalization을 선점. 실제 GPU work는 별도 drain 대상. terminal |
| `expired` | 허용 기한 이전 수락되지 않음. terminal |
| `stale` | game/root/model/handle 세대가 더 이상 유효하지 않아 수락하지 않음. terminal |
| `failed` | 입력·backend·수치·resource 등 명시 오류. terminal |

한 request의 terminal outcome은 정확히 한 번 게시한다. 원자적 상태 전이 또는
single owner event loop로 선형화 지점을 정한다. 완성 callback과 취소가 동시에 와도
두 terminal outcome을 내보내거나 두 번 reservation을 해제하지 않는다. `completed`
게시 직후 root가 바뀔 수 있으므로, Search는 backup 직전에도 selection의 현재
game/root generation과 consumed 여부를 재확인한다. 이때 거절된 결과는 consumer
측 stale rejection으로 기록하며 Scheduler의 이미 정해진 terminal 상태를 되돌리지 않는다.

deadline 도입 기본안은 **결과 검증·수락이 끝나는 tick < deadline**이다. 남은 자기
시간에서 출력 margin을 제외한 요청 deadline을 Search가 제공한다. enqueue 때만
검사하지 않고 dispatch 직전·완료 finalization·backup commit 시 검사한다. 정확한
경계 포함 여부는 엔진/arena 시간 계약과 함께 잠그며 임의 grace period를 넣지 않는다.

처리 순서의 구현 의무는 다음과 같다.

1. Search가 checked leaf·selection ID·경로를 정하고 reservation을 만든다. virtual
   loss/in-flight 증가는 reservation ledger에 귀속한다.
2. Scheduler가 handle/shape/finite budget/기한을 검사한 뒤 admit 또는 명시 거부한다.
   admission 실패도 같은 reservation의 취소 경로로 해제한다.
3. dedup subscriber 또는 실행 결과를 받은 Scheduler는 logical finalization을 한 번 수행한다.
   공유 실행의 한 subscriber 취소는 다른 유효 subscriber를 취소하지 않는다.
4. Search가 terminal outcome을 받고 현재 generation·deadline·selection 소비 여부를
   한 commit 경계에서 확인한다. 유효 성공만 exactly-once backup하고 나머지는 해제한다.
5. reservation 해제와 방문 반영을 같은 소유권 경계에서 처리한다. 실패·취소는 실제
   loss로 backup하지 않으며, 정상 완료도 virtual loss를 남기지 않는다.
6. 물리 ExecutionId의 device 완료 event까지 input/output/workspace pin을 유지한다.
   물리 buffer 해제는 ExecutionId/resource owner별 한 번이며 논리 RequestId의
   terminal/취소와 별도 accounting이다. 논리 취소만으로
   buffer를 free/reuse하지 않는다. 마지막 subscriber 종료 시 새 작업으로 논리 재사용하지 않는다.

취소·만료 후 새 재시도는 새 RequestId를 사용하지만 같은 selection을 이어받으면
한 번만 backup할 수 있다. 자동 재시도 횟수·동일 모델 fresh 복구 비용은 설정에 유한하게
명시한다. 무한 retry나 OOM 뒤 precision/batch/model을 조용히 바꾸는 정책은 없다.

## 11. stop·착수·newgame·모델 교체

`stop` 또는 실제 착수 시 Search는 해당 root의 수락 권한을 닫고 새 평가 제출을
막는다. queued 요청을 취소하고 running 요청을 logical cancel/expire하며, GPU의
취소 불가 구간을 drain 또는 격리한다. 늦은 결과가 다음 root·다음 시간·다음 게임의
latent/raw cache를 채우지 않게 한다. 잔여 실행의 bytes/time/cause를 기록한다.

`ucinewgame`은 game generation을 바꾸고 history/correction·검색 통계를 초기화한다.
모델 가중치를 온라인으로 변경하지 않는다. 모델 unload/hot swap은 이전 handle을
무효화하고 buffer owner와 running 작업의 종료를 확인한 뒤 새 세대를 발급한다.
newgame와 model generation은 서로 다른 사건이다.

main evaluation에서 상대 차례 helper/speculation을 수행하지 않는다. 출력 뒤
취소 불가 GPU 작업이 남을 수 있는 경우, 새 계산·cache 게시·다음 turn 이득을 금지하고
잔여 사용량·drain wall time을 관측한다. 대국 runner는 실제 출력/stop 및 자식 종료를
외부 시계로 확인한다. 주 시간/자원 귀속은 [실험 규약](EXPERIMENTS.md)에 따른다.

## 12. buffer·queue·finite budget

prevalidated manifest의 tensor bounds와 실행 설정의 유한 자원 한도를 모두 만족해야
admit한다. 요청당/배치당 입력·출력·latent·workspace·staging·pending bytes를 계산하고
checked arithmetic으로 overflow를 거부한다. batch 길이×shape×dtype 계산도 포함한다.

| 자원 | 실행 설정에서 실제 값을 잠글 항목 |
|---|---|
| queue | 최대 logical requests, 최대 age, priority starvation 방지, admission fail code |
| host/device | 각각 최대 live bytes, pinned bytes, cache bytes, workspace bytes. 같은 memory를 중복 소유할 때 accounting 명시 |
| batch | 최대 items·shape bucket·batch wait, deadline reserve. fill을 위해 무한 대기 금지 |
| iteration | 최소/최대 steps와 kernel/chunk 단위. budget 넘긴 결과를 full이라고 표시 금지 |
| concurrency | CPU workers, GPU streams, outstanding kernels, helper core/RAM budget |
| shutdown | 취소/drain/자식 종료 timeout, 실패 시 명시 상태. 작업 scope 밖 process 종료 금지 |

재사용 slot은 `(owner, slot, generation)` handle로 관리한다. borrower lifetime이 끝나고
device 완료가 확인될 때만 recycle한다. 취소 callback과 device 완료 callback이 같은 slot을
각각 해제하지 않게 physical resource owner가 하나여야 한다. stale handle 접근은 명시
실패이며 잘못된 slot 값을 정상 tensor로 읽지 않는다.

cache miss의 allocation이나 load도 budget에 들어간다. limit을 넘으면 정해진 admission
거부·eviction·같은 모델 fresh 경로 중 허용된 정책을 수행하고 비용과 warning을 남긴다.
의미가 다른 helper 모델·낮은 정밀도·작은 batch로의 자동 전환은 별도 실험 설정 없이 하지 않는다.

## 13. 오류·복구 계약

오류에는 `code`, `stage`, 요청 ID/세대, 영향 범위, 실제 실행 상태, recovery outcome을
보존한다. 외부 경로·계정·비밀은 공유 기록에 넣지 않는다. 런타임 로그 경로는
[개발 기준](ENGINEERING-STANDARDS.md)의 생성물 규칙을 따른다.

| 오류 계열 | 예와 허용 동작 |
|---|---|
| `InvalidInput` | 잘못된 FEN/move/shape/dtype/handle. 호출 경계에서 거부 |
| `UnsupportedContract` | 미지원 variant/encoding/backend/precision/partial mode. 다른 계약으로 숨겨 대체하지 않음 |
| `CacheMiss/CacheInvalid` | 입력·세대·parity 불일치 또는 eviction. 같은 자체 모델의 fresh 재계산 가능 |
| `ResourceExhausted` | queue/host/device/workspace 상한·OOM. 명시 거부/실패, 설정 silent 변경 금지 |
| `BackendFailure/NumericalFailure` | device 오류·NaN/Inf·policy/WDL 불일치. 정상 평가/무승부로 반환 금지 |
| `Canceled/Expired/Stale` | 수명 종결. reservation 정확 해제, 방문/새 cache 게시 금지 |

같은 모델 fresh 복구도 현재 deadline과 budget 안에서만 수행한다. 성공해도 최초
오류·복구 실행·추가 비용을 남긴다. 복구가 기한에 끝나지 않으면 실패/만료이며 정상
완료로 위장하지 않는다. 최종 착수의 안전한 기존 합법 후보 유지 또는 대국 실패
처리는 Search/UCI/arena의 잠긴 정책에 따르고, evaluator가 임의 수를 만들어내지 않는다.

H의 별도 CPU 모델은 명시 helper/fallback 정책과 자체 provenance를 가진 평가다.
외부 LC0/Stockfish 프로세스 호출은 별도 composite 실험이며 내부 오류 복구라고
숨겨 쓰지 않는다. 자체 full 대조도 자기 시간의 계산비용에 포함한다.

## 14. 독립 수락 벡터와 증거

아래는 최초 해당 기능을 도입할 때 확인할 계약 벡터다. 현재 실행된 검사 목록이나
새 영구 테스트를 모두 만들라는 요구가 아니다. 기존 기반의 의미 있는 검사를 재사용하고
없는 위험만 독립적으로 보강한다. 구현과 같은 갱신 함수를 참조로 쓰는 대조는 독립
full 검증으로 보고하지 않는다. 실제 명령·source SHA·model manifest·결과는 구현 시 기록한다.

| 벡터 | 독립 입력·실패 조건 | 수락 증거 |
|---|---|---|
| V01 규칙·특수 수 | 공개 perft/독립 규칙 참조, castling/en passant/승격·긴 make/unmake | 합법 집합·상태·카운터·undo의 정확 일치 |
| V02 동일 보드·다른 이력 | 반복/history-fill/차례·권리·카운터를 각각 바꾼 상태, FEN unknown prefix | 필요한 입력 변경마다 reuse가 차단되며 미상 이력을 창작하지 않음 |
| V03 policy 대응 | 승격 종류 4개, legal 순서 permutation, padding/mask | move별 정책 대응·value 불변성, 불합법 move 확률 소비 없음 |
| V04 관점·terminal | 부모/자식 WDL·정확 terminal, scalar/centipawn 혼용 유도 | W/L 교환과 단위 검증. 정확 terminal에서 GPU 요청 없음 |
| V05 exact delta | 독립 full encoder, king bucket·threat 경계·eviction/재계산 | 선언 precision별 parity와 fallback 사유·비용 |
| V06 warm 경로 | 서로 다른 부모 경로로 같은 입력 도착, 긴 누적 trace | ApproxLatent provenance 분리, RawEval 오염 없음, fresh 편차 기록 |
| V07 cache identity | model/encoding/precision/steps/ordered moves 변경·hash 충돌·slot 재사용 | 잘못된 hit 거부, 가능한 exact permutation만 허용, ABA 차단 |
| V08 hit·방문 | 한 selection의 반복 hit, 독립 두 parent subscriber, sibling/FPU | network/hit/selection/backup 카운터 분리. 동일 selection 한 번 backup |
| V09 동시 취소 | queued cancel, running cancel, finish와 cancel 동시, subscriber 한 명 취소 | terminal outcome·reservation 해제 각각 한 번. 다른 subscriber 독립 |
| V10 deadline·root | 경계 tick, finish 후 root 변경, stop/newgame 뒤 delayed callback | 만료/stale 차단, 새 root/게임 cache·통계 오염 없음 |
| V11 수치·지원 | NaN/Inf/정규화 오류·알 수 없는 backend/shape/dtype | typed failure. uniform policy/0점/무승부 대체 없음 |
| V12 자원·buffer | 계산 overflow·OOM·상한 초과·취소 중 device borrow | 유한 admission, physical 완료 전 recycle 없음, silent 설정 변경 없음 |
| V13 자기 시간 | helper/speculation·취소 불가 잔여 kernel·fresh audit | 모든 자기 비용 계측, 상대 차례 실행/결과 유입 방지, bounded drain |

캐시·scheduler 기능 도입 전에는 최소 V07~V12의 실패·race 경로를 검토한다. 실제
병렬 runtime은 deterministic scheduler simulation만으로 검증됐다고 보고하지 않으며
지원 환경에서의 통합 취소·buffer 수명 검사를 추가한다. GPU 사용률·throughput·NPS
향상은 계약 통과나 LC0 대비 강도 증거와 별도 결과다. 최종 강도는 잠긴 paired arena
조건으로 [실험 규약](EXPERIMENTS.md)에 따라 판정한다.
