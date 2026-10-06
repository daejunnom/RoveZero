# rz-uci — B01의 임시 protocol/session 경계

`TASK-B01/B03`의 UCI 파서, 세션 수명과 탐색 시간 연결을 구현한다. 총괄 통합본의
`engine`·`bootstrap`과 실행 파일은 A Rules → C classical projection/scripted backend →
D runtime → B 비동기 탐색을 연결한다. `--cpu-mock`은 GPU 없는 개발용 provider다.
`onnx-cpu` feature로 빌드하면 명시적인 `--onnx-cpu`가 선정 Maia·검증한 ORT CPU
FP32의 실제 신경망 평가를 연결한다. Linux에서 실제 수치·Rules·UCI 연결을 인수한
소스와 실행 근거는 [통합 인수 기록](../../docs/INTEGRATION-STATUS.md)을 따른다.
`onnx-cuda` feature는 Linux의 명시적 `--onnx-cuda` 연결을 추가한다. CUDA 코드·receipt
제공과 실제 CUDA 수치·Rules·D 전달·B backup 인수는 각각 구분한다.
목표 GPU와 정식 대국은 별도 인수다. 초기 B-local 접점에 이어
[PR #7](https://github.com/daejunnom/RoveZero/pull/7)의 공통 0.1 계약을 연결했다.
`rz-contracts` 의존성은 root workspace의 단일 `path` 계약 0.1.0을 사용한다.
게시 기준 SHA는 `67284c4f66f7a7ae9f46fa63dfd50e7410eb6845`이며 현재 root와 같은
계약 소스·revision을 소비한다.
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

총괄 통합본은 root toolchain Rust 1.96.0으로 다음 명령을 사용한다. 생성물은 저장소 밖
작업 전용 `CARGO_TARGET_DIR`에 둔다. 실제 실행 결과와 CI 상태는
[통합 인수 기록](../../docs/INTEGRATION-STATUS.md)을 따른다.

```sh
cargo test -p rz-uci --all-targets --all-features --locked
cargo fmt --all -- --check
cargo clippy -p rz-uci --all-targets --all-features --locked -- -D warnings
cargo run -p rz-uci --locked -- --cpu-mock
```

실제 CPU provider는 다음의 독립 입력을 모두 요구한다. 이름과 값은 하나의
`--이름=값` 인수로 전달하며 다른 provider로 조용히 대체하지 않는다. 경로는 실행
환경의 저장소 밖 입력·출력에 맞춘다. 가중치·변환본·ORT 파일의 권리와 출처는
MIT 엔진 소스와 분리한다.

```text
rz-uci --onnx-cpu
  --source-weights=<selected-maia.pb.gz>
  --onnx-model=<verified-export.onnx>
  --export-manifest=<verified-export-manifest.json>
  --manifest-sha256=<64 lowercase hex>
  --ort-library=<explicit-runtime-library>
  --ort-sha256=<64 lowercase hex>
  --output-root=<existing-private-output-directory-outside-Git>
```

위 표기는 필요한 인수 목록이며 줄을 나눈 shell 명령이 아니다. `--attestation`
선택 시의 구조화된 startup·종료 영수증과 E의 실제 NN paired 실행 연결은
`0f0d70d`의 Linux CPU 인수에서 확인했다. startup의 검증된 프로필과 executable,
종료의 D 완료 집계·첫/마지막 computed context·확정 drain을 서로 대조한다.
모델 로딩 성공, 실제 평가 응답, 물리 종료 확인을 구분하며 전체 root별 journal이나
물리 추론 호출 수를 이 집계로 대신하지 않는다. [인수 기록](../../docs/INTEGRATION-STATUS.md#후속-실제-cpu-신경망-pair-인수)의
소스·binary·명령·외부 증거가 실제 결과의 적용 범위다.

CUDA는 같은 공통 입력에 `--onnx-cuda`, `--cuda-bundle=<bounded-runtime-bundle.json>`,
`--cuda-bundle-sha256=<64 lowercase hex>`를 명시한다. bundle **원본 JSON 파일 SHA**와
실제 pin의 **canonical bundle digest**는 별도 identity다. core 파일명·SHA는
`--ort-library`·`--ort-sha256`와 일치하며, C의 exact nineteen-file loader와 placement
probe를 통과해야 한다. 미지원 feature/platform/provider·불완전 bundle을 CPU로
대체하지 않는다. device 0, arena 1 GiB, FP32·TF32 off, batch 1, intra thread 1,
worker 1, fresh/full step 1, HistoryFill No를 고정하고 외부 인수 전에 변경하지 않는다.
공통 native session/runtime 코어를 CPU·CUDA factory가 공유하며 실제 출처는 C owner가
발급한다. CPU constructor와 CPU V1 projection은 CUDA 출처를 거부한다.

`--attestation`의 CUDA 영수증은 역할 output root의 새 `native-process-<pid>` 아래
`native-cuda-startup.v1.json`·`native-cuda-termination.v1.json`이다. CPU V1 파일은
그대로 유지한다. CUDA startup은 실행 파일·모델·encoding·backend와 exact bundle,
placement profile SHA·CUDA node·mapping 검증을 결합한다. session resident 1 GiB와
D execution device 1 GiB는 **서로 다른 admission 선언**이며 측정 VRAM이나 전체
hard cap이 아니다. Rules 입력 ByteBudget은 host 소유이며 device를 중복 예약하지 않는다.

CUDA final은 D normal-poll 완료 집계와 B의 최종 tree guard 이후 root 초기화·non-root
backup count+first/last를 구분한다. root 초기화의 traversal은 0이며 NN 실행 수나 새
탐색 방문 수와 혼동하지 않는다. startup placement probe·늦은 물리 완료·D 전달·B 소비는
각각 다른 사건이다. scheduler와 최종 delivery event의 회수 수·drop/overflow 및 drain 중
폐기 수를 공개하지만 전체 사건 journal이나 물리 추론 총량을 이 집계로 증명하지 않는다.
미확정 physical drain·Q·원래 실패·cleanup 실패와 최종 runtime mapping 원인을 보존한다.
추가 passive observer는 backup 권한을 만들거나 기존 마감·취소·scope 검사를 완화하지 않는다.

CPU fixture는 parser, 상태 원자성, ticket 취소와 착수 출력·event loop를 확인한다.
`tests/search_integration.rs`는 독립 인공 트리로 UCI→PUCT→착수와 terminal 우회,
정확 마감, root 교체, clock 선택, 잘못된 전체 trace와 stop 경계, 큐의 늦은 완료와
infinite 자원 만료의 출력 소유권을 대조한다.
`tests/contract_session.rs`는 공통 세대·취소·registry·clock owner 경계를 확인하며,
`tests/contracts_integration.rs`는 실제 공통 타입으로 UCI→비동기 평가→두 ply backup,
stop/newgame/deadline·전송 실패의 취소를 대조한다. `tests/engine_binary.rs`는 실제
실행 파일에서 A/C/D 연결·시간 마감·root 교체·중복 stop·유한 quit/EOF를 검사한다.
standalone fixture 검사와 실제 연결 검사의 결과를 구분한다. GPU, 실제 neural 평가와
정식 대국 강도는 별도 인수다.

## 소유 물리 작업과 착수 출력

production Engine Owner는 stop/hard deadline에서 Session의 마지막 유효 착수를 고정한 뒤
모든 소유 worker의 정상 drain과 join을 확인하여 한 번 출력한다. 늦은 평가/Progress가
고정한 착수나 backup을 바꾸지 않는다. 입력과 isready는 대기 중에도 처리하고,
받아들인 root/game/registry 교체와 quit/EOF는 이전 scope의 보류 출력을 폐기한다.
별도 Session/transport reducer만 사용하면 이 physical fence를 제공한 것이 아니다.

출력 대기는 기존 유한 shutdown_limit을 사용한다. drain 오류·panic·시간 초과는
PhysicalFenceFailure와 원래 typed serve 오류로 전달하고 성공 착수를 승인하지 않는다.
소유 worker/pin을 임의 해제하지 않는다. GPU session의 모델은 계속 resident일 수 있으며
물리 평가 작업 완료와 모델 unload는 별도다. 이 수명 수정은 PUCT/backup·WDL·기본 S0와
opt-in exact-terminal S1을 바꾸지 않는다. 정확한 시간 예산과 인수 범위는
[BT4 출력 경계 기록](../../docs/research/LOCAL-MODEL-BASELINE.md#9-stop마감-출력의-물리-완료-경계)을 따른다.
