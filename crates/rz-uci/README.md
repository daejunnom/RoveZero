# rz-uci — B01의 임시 protocol/session 경계

`TASK-B01/B03`의 UCI 파서, 세션 수명과 탐색 시간 연결을 구현한 독립 library다. 실제 체스 실행
파일·Rules·neural evaluator·runtime 구현은 이 crate에 없다. 초기 B-local 접점에 이어
[PR #7](https://github.com/daejunnom/RoveZero/pull/7)의 공통 0.1 계약을 연결했다.
`rz-contracts` 의존성은 SHA `67284c4f66f7a7ae9f46fa63dfd50e7410eb6845`에 고정한다.
`SearchTicket`은 세션 내부에서 발급하여 임의로 생성할 수 없는 소유권 토큰이며 공통 RequestId,
SelectionId, GameGeneration, RootGeneration을 대체하는 정의가 아니다.
공통 `ContractSessionOwner`가 game/root·registry·clock·cancel을 관리하고 opaque ticket과
실제 평가 요청/selection의 대응은 검색 adapter가 소유한다.

## 입력과 상태 정책

- 지원: `uci`, `isready`, `setoption`, `position startpos|fen <6 fields> [moves ...]`,
  `go movetime`, 양쪽 `wtime/btime` 및 `winc/binc/movestogo`, `nodes`, `infinite`,
  `stop`, `quit`, `ucinewgame`. `nodes`는 movetime 또는 clock과 함께 사용할 수 있다.
- `go ponder/searchmoves/depth/mate`, `ponderhit`, 알려진 명령의 잘못된 인수는
  명시적으로 거부한다. 시간 mode 충돌·중복 field·정수 overflow도 거부한다.
  별도 한도 없는 `go`는 거부한다. 다른 미지 명령은 진단하고 무시한다.
- line 기본 한도 16 KiB, trace 2,048 ply, option name 128 byte/value 1,024 byte,
  option 등록 64개, Rules legal move view 기본 한도 4,096개다. legal view 한도는
  `ParserLimits.max_legal_moves`로 실행 설정에 맞춘다. ASCII 한 줄만 허용한다.
- FEN의 외부 문법과 move text는 파서가 확인한다. 전체 FEN의 체스 의미, 이력,
  모든 수의 합법성과 정확한 terminal은 `PositionPort::prepare`가 확인한다.
  임시 snapshot과 legal view 전체를 수락한 뒤 root를 교체하므로 거절 시 원본
  root·현재 검색·option을 보존한다. Rules가 제공한 ongoing 빈 legal view는 오류다.
- 옵션은 실행 adapter가 실제 지원하는 것만 등록한다. check/spin/string/combo/button
  metadata/default와 입력값을 검증하고, 실행 중 변경은 거부한다. `isready`는 실행
  중에도 즉시 `readyok`를 반환한다.
- `stop`/deadline은 마지막 합법 progress/completion, 없으면 ordered legal 첫 수를
  한 번 출력한다. fallback 이유와 실제 검색 실패는 별도 diagnostic에 남긴다.
  exact terminal에서만 `bestmove 0000`을 출력한다.
- `go infinite`의 정상 worker 완료는 후보를 보관하고 `stop`까지 착수
  출력을 기다린다. 중복 완료·완료 후 progress는 거부하며 계산을 재시작하지 않는다.
  정확한 terminal도 infinite에서는 worker 없이 `stop`까지 기다린다. worker 실패나
  불법 완료는 진단을 남기고 합법 fallback/마지막 합법 후보로 한 번 종료한다.
- 새 `position`, `go`, `ucinewgame`, `quit`, EOF는 이전 ticket을 취소한다. 새 root/
  game/quit에서는 이전 착수를 출력하지 않는다. 늦은·중복·다른 session의 ticket을
  거부한다. `ucinewgame`은 startpos와 game-owned namespace/history/search/correction
  초기화 effect를 발행한다. 이전 raw-cache game generation을 무효화하고 model
  weights는 보존하는 것은 runtime adapter의 책임이다.

## 공통 계약 owner

`contracts::ContractSessionOwner`는 실제 process epoch·clock origin·registry의
`AcceptanceScope`와 실행 설정을 받는다. accepted `position/go/ucinewgame`에서 root를,
accepted `ucinewgame`에서 game을 checked 증가시킨다. overflow는 Session 변경 전에
거절하며 잘못된 전체 trace와 명령은 이전 세대·snapshot·token을 유지한다. registry 교체는
idle에서 명시적으로 수행한다. model/encoding/backend handle이나 digest를 만들어내지 않는다.

`handle_line`은 준비 전에 go 시각을 잡고 Rules가 확인한 실제 차례로 예산을 배분한다.
`handle_event`는 callback/deadline/EOF를 처리한다. Line event는 checked side callback을
제공하는 `handle_line`으로 보내야 한다. outcome의 `errors`는 typed 계약 원인을,
`session.diagnostics`는 원래 worker·설정 오류를 함께 보존한다.

현재 `scope`, 공통 `deadlines`, cancellation의 `token()`과 control의 simulation 한도를
`rz-search::contracts::ContractSearchConfig`에 전달한다. `IdAllocator`는 process 전체의
동일 Arc를 공유한다. 각 pump의 live scope는 이 owner와 실제 registry에서 갱신해서 읽는다.
새 root/game, stop/quit/EOF와 owner Drop은 공통 token과 기존 제어 flag를 닫는다.
두 atomic은 같은 저장소가 아니므로 취소에는 owner 또는 `ContractCancellation::cancel()`을
사용한다. baseline control flag만 직접 닫았다는 사실로 공통 worker 취소를 가정하지 않는다.

callback은 Session이 후보를 검증한 뒤 공통 세대·취소·fresh clock을 최종 검사한다.
현재 ticket의 hard timer는 worker 결과의 취소와 별도로 finite 출력을 마감한다. worker가
이미 취소됐어도 합법 후보/fallback을 한 번 반환한다. infinite 자원 만료는 계산만 닫고
stop까지 후보를 보관한다. 이른 timer와 이미 정상 완료한 infinite의 timer는 새 실패나
착수 출력을 만들지 않는다.

`serve_events_with_handler`의 handler에서 owner를 사용하고 typed 원인을 기록한다.
transport/dispatch 실패의 EOF 정리도 같은 handler를 거쳐 취소 권한을 닫은 다음
runtime Cancel/Shutdown을 시도한다. Cancel dispatch 실패 때문에 공통 수락 권한이
열린 채로 남지 않으며 원래 I/O/dispatch 오류와 정리 오류를 모두 반환한다.

## 통합 adapter의 의무

`SessionResult`의 `protocol`만 stdout에, `diagnostics`는 stderr/별도 계측에 쓴다.
`effects`를 먼저 처리하여 착수 출력 전에 취소 권한을 닫는다. Start는 snapshot,
검증된 option과 UCI `GoLimits`를 전달한다. 초기화가 끝나기 전 `Session`을
사용하지 않으며 option 변경 effect의 실제 runtime 적용 실패를 기록해야 한다.

Rules adapter는 실제 상태를 소유/pin하는 immutable snapshot과 정확한 ordered
legal UCI mapping을 제공해야 한다. 단순 move 문법 검사는 체스 합법성 증거가
아니다. Runtime/search adapter는 ticket을 공통 세대/ID에 매핑하고 실제 ply·move
순서, 계약 revision, 오류·모델/encoding 식별자를 연결해야 한다.

`bridge::SearchBinding`은 Start ticket과 `GoLimits`를 `rz-search::SearchControl`에
연결한다. Rules가 확인한 `SideToMove`를 받아 실제 자기 시계/increment를 선택하고,
soft/admission/hard 및 output/drain 여유를 적용한다. callback의 전달 시각과 실제
현재 단조 시각 중 늦은 쪽으로 검사하므로 오래된 enqueue tick을 수락 근거로 쓰지 않는다.
만료 때 검색의 공유 취소 flag를 먼저 닫고 마지막 합법 후보/fallback을 반환한다.
budget의 start는 `go`를 처리하기 전에 잡아 snapshot clone·초기화 비용도 자기 시간에 넣는다.
Session의 guarded callback은 합법 후보 검증·문자열 준비 뒤, 후보와 완료 상태를 저장하기
직전에 수락 closure를 호출한다. binding은 이 경계에서도 실제 시각과 공유 취소 flag를
확인하므로 검증 도중 마감·취소가 생겨도 이전 후보를 바꾸지 않는다.

`BuildSearchSettings`는 Default 없이 시간 설정, 최대 완료 simulation 수와 시간 없는
검색의 유한 wall 상한을 요구한다. `nodes`는 현재 B 기준선에서 root 초기화를 제외한
완료 simulation 수이며 공통 UCI node 의미는 총괄과 통일해야 한다. 설정보다 큰 nodes
요청은 조용히 줄이지 않고 거부한다. 시간 없는 nodes/infinite의 wall 한도는 UCI 시계와
별개의 자원 한도다. infinite에서는 한도 도달 시 작업을 취소·진단하고 `stop`까지
후보를 보관한다. 이미 정상 완료한 worker의 뒤늦은 자원 timer는 실패로 보고하지 않는다.

Session 자체는 공유 clock 또는 Search의 N/W를 소유하지 않는다. Cancel effect에서
해당 binding의 `cancel()`을 호출한 뒤 protocol을 출력한다. 논리 취소 후 실제 backend
완료 전 buffer 재사용을 금지하며 Runtime이 유한 drain을 소유한다.

`serve_events_with_handler`는 command·search completion·deadline을 받는 재사용 event
loop다. 호출자가 binding을 ticket별로 보관하고 event handler에서 `progress`, `complete`,
`expire`를 호출한다. 큐에서 event를 꺼낸 시점에 마감을 확인해야 하며 producer에서만
확인하면 안 된다. `serve_events`/`handle_event`는 clock 없는 세션의 기본 mapping이므로
시간 제어 연결에는 binding-aware handler를 사용한다. bounded `sync_channel`과 독립
worker를 사용한다. effect callback은 작업을 enqueue/
cancel해야 하며 전체 검색을 inline 실행하면 안 된다. I/O/dispatch 실패를 반환하며
EOF에서 Cancel/Shutdown을 처리한다. `forward_lines`는 크기를 제한한 선택적 입력
producer다. 입력 I/O 오류에서도 diagnostic과 EndOfInput을 전달한 뒤 원래 오류를
반환하므로 worker가 가진 sender가 session 종료를 막지 않는다. 실제 stdin reader의
중단과 thread join은 transport가 소유하며, quit에서
OS read를 기다리는 무한 join을 하면 안 된다.

## 총괄 통일·연결 대상

| 임시 접점 | 후속 연결 의무 |
|---|---|
| `PositionPort` / `PreparedPosition` | A의 immutable checked snapshot, 전체 trace, exact terminal과 legal UCI mapping |
| `SearchTicket` / Start·Cancel | `ContractSessionOwner`의 현재 game/root와 `ContractSearch` request/selection, 실제 D 작업 수명 |
| `ContractSessionOwner` / 설정 | 실제 process/registry owner, node 의미, E 실행 manifest의 유한 자원 한도 |
| event handler / effect dispatch | D의 bounded queue, 실제 worker·물리 drain 및 transport 종료 |

총괄 root 통합에서는 두 B crate의 고정 git 의존성을 단일 workspace/path 계약 crate로
통일하고 root member/lockfile을 맞춘다. 동일 source의 git/path 타입을 혼용하지 않는다.

## 검사

Rust 1.90.0에서 독립 package manifest로 다음 검사를 실행한다. 생성물은 저장소 밖
작업 전용 `CARGO_TARGET_DIR`에 두며 종료 전 회수한다.

```sh
cargo test --manifest-path crates/rz-uci/Cargo.toml
cargo fmt --manifest-path crates/rz-uci/Cargo.toml --check
cargo clippy --manifest-path crates/rz-uci/Cargo.toml --all-targets -- -D warnings
```

CPU fixture는 parser, 상태 원자성, ticket 취소와 착수 출력·event loop를 확인한다.
`tests/search_integration.rs`는 독립 인공 트리로 UCI→PUCT→착수와 terminal 우회,
정확 마감, root 교체, clock 선택, 잘못된 전체 trace와 stop 경계, 큐의 늦은 완료와
infinite 자원 만료의 출력 소유권을 대조한다.
`tests/contract_session.rs`는 공통 세대·취소·registry·clock owner 경계를 확인하며,
`tests/contracts_integration.rs`는 실제 공통 타입으로 UCI→비동기 평가→두 ply backup,
stop/newgame/deadline·전송 실패의 취소를 대조한다. 이 검사는 실제 A/C/D 구현 연결을
대신하지 않는다. 실제 A/C/D adapter, root workspace 통합 CI, GPU, 실제 neural 평가와
대국 강도는 아직 인수하지 않았다.
