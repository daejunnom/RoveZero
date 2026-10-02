# B 담당 탐색·시간 제어

TASK-B02/B03의 CPU 기준 구현이다. 최초 B-local 기준선에 이어 총괄
[PR #7](https://github.com/daejunnom/RoveZero/pull/7)의 `rz-contracts` 0.1.0,
engine schema 0.1을 사용하는 비동기 연결부를 구현했다. 총괄 통합본의 의존성은
root workspace의 단일 `path` 계약 0.1.0이다. 게시 기준 SHA는
`67284c4f66f7a7ae9f46fa63dfd50e7410eb6845`다. 루트 workspace,
CI, 실제 체스 코어·모델·runtime은 이 crate가 소유하지 않는다.

## 구현과 경계

- `policy`: S0 PUCT와 선택 정책 교체 접점. 기본 c_puct=1.5, FPU=0, root noise 없음,
  score 정확 동률은 합법 수 순서. policy algorithm/revision/config를 기록한다.
- `tree`: 자체 selection·expansion·backup, 단일 진행 예약, 원자적인 방문 회계.
  root 최초 평가에는 edge 방문이 없다. leaf의 실제 차례 값 `W-L`은 ply마다 반전한다.
  root 착수는 N, Q, 합법 수 순서다. 정확 terminal은 NN 평가 없이 처리한다.
- `driver`: 불변 `CheckedPosition`과 `Evaluator`를 주입하는 단일 worker loop.
  상태 전이는 A의 checked API로 재생하고, 평가 정책은 해당 합법 수 순서로 받는다.
  평가가 지연·실패해도 검증된 root fallback과 원래 오류를 보존한다.
- `time`: movetime과 실제 자기 차례의 clock+increment를 구분한다. 기본 제안은
  moves horizon 30, increment 반영 80%, hard 배분 배수 3, 출력/drain 여유 각 10ms다.
  clock 배분은 현재 남은 시간을 넘지 않는다. 이 숫자는 실제 실행 manifest에서 잠근다.
- `contracts`: 실제 공통 `Evaluator::submit/poll/cancel`을 사용하는 단일 진행 탐색.
  실제 tree ticket과 공통 request/selection·snapshot·ordered legal view를 함께 보관한다.
- `contract_time`: 하나의 process clock origin 이후 elapsed ns를 checked 변환하고
  공통 deadline과 B 시간 예산, 공통 token과 기존 제어 flag의 취소를 연결한다.

확률 검증은 기본 f64 합 오차 1e-9이며 입력을 uniform/clip/draw로 수선하지 않는다.
정확 terminal scalar는 -1/0/+1, draw utility는 0이다. root history·종료와 평가 캐시를
board-only key로 공유하지 않는다. baseline은 단일 selection이며 virtual loss·DAG·
병렬 selection·raw cache 구현은 포함하지 않는다.

## 수명·예산

`SelectionTicket`은 B tree의 비위조 consume capability다. 공통 RequestId,
SelectionId, GameGeneration/RootGeneration을 대신하는 wire schema가 아니다.
reset은 과거 ticket을 무효화한다. 중복 callback·실패·취소·만료는 방문을 만들지
않으며 예약은 한 번만 해제된다. backup의 마지막 commit 경계에서 단조 시각과
취소를 다시 검사해 검증 중 마감 초과도 거절한다.

`SearchControl::from_budget`은 soft/admission/hard 경계를 연결한다. soft 또는
admission 종료 뒤 새 요청을 제출하지 않고, 이미 진행한 유효 결과는 hard deadline
전에만 수락한다. 취소 flag의 마지막 owner 검사에서 backup을 선형화한다. 해당
경계 뒤의 취소와 UCI 출력 소유권은 session ticket으로 다시 확인한다.

TreeLimits 기본값은 node 100000, edge 1000000, depth 256, legal width 4096이다.
시간·simulation과 이 구조 한도는 유한하게 설정한다. opaque Move가 외부 heap을
소유하거나 provider가 host/device buffer를 쓰는 비용은 실제 adapter/runtime의
byte budget에 별도 포함한다. 이 구조 한도만으로 전체 RAM/VRAM 상한을 증명하지 않는다.

`evaluator_calls`는 평가 함수 호출 수이며 물리 NN execution·cache hit 카운터가
아니다. root initialization, accepted backup, completed visit, 예약 해제를 구분한다.
completed visit는 유효 traversal 한 개이며 그 경로의 각 edge N은 한 번 증가한다.

## 공통 계약을 사용하는 비동기 경계

`ContractSearchConfig`는 actual model/encoding/backend, declared raw head 허용 오차,
유한 compute/byte/tree/simulation 예산, 공통 cancellation/deadlines와 공유 `IdAllocator`를
요구한다. `ContractPosition`은 A의 불변 snapshot·legal view·checked transition과 실제
owner/revision 권한을 제공한다. input-key callback은 C가 실제 모델/이력 입력으로
발급한 `EvalInputKey`를 공급한다. B는 보드 hash나 임의 숫자로 해당 증거를 만들지 않는다.

`pump`는 한 번의 제출·poll·논리 정리만 수행하며 기다리거나 thread를 join하지 않는다.
`live_scope`는 현재 owner/registry에서 읽어야 한다. 요청의 과거 context를 그대로 반환하면
현재 세대 검증이 되지 않는다. 제출과 backup 직전 Rules 권한·현재 세대·취소·시각을
다시 검사한다. pending search의 Drop/unwind는 공통 token을 닫지만 Runtime의 물리
작업을 해제하지 않는다. 명시 runtime cancel과 유한 drain은 호출자가 계속 수행한다.

각 evaluator stream의 활성 search consumer는 하나다. 여러 검색을 실행하면 dispatcher가
RequestId별 stream으로 라우팅한다. foreign/duplicate context의 `RejectedResult`는
원래 owned payload·failure·recovery를 호출자에게 반환하고 현재 예약을 유지한다.
matching 실패는 원래 typed 오류와 recovery를 보존하며 방문 없이 예약을 해제한다.

공통 성공 payload는 전체 context·실제 ordered legal 배열·compute/provenance를 검증하고,
선언된 raw head 허용 오차를 다시 검사한 뒤 `LegalPolicy::normalized()`와
`Wdl::normalized()`를 명시 호출한다. 변환은 유효 head의 f64 unit-sum 복사본이며
허용 범위 밖 값을 수선하지 않는다. 이 변환 정책과 두 허용 오차는 model/compute 실행
manifest에 잠근다. leaf scalar는 f64 복사본의 W-L이다. accepted execution ID·raw-cache
결과와 제출 시도·실제 admission을 구분하며, 이 카운터는 물리 NN 완료 계측이 아니다.

## 총괄 통일·연결 대상

| 현재 임시 접점 | 후속 소유자와 연결 의무 |
|---|---|
| `ContractPosition` | 실제 A의 checked state·legal order·정확 terminal·완전 이력·live authority |
| model/input-key 공급 | C의 실제 encoding/input identity·policy mapping·model registry·수치 허용 오차 |
| common `Evaluator` | D의 queue/batch·finalized 결과·physical lease·유한 drain |
| `SelectionTicket` / 공통 ID | 실제 registry/epoch 발급 owner; allocator는 전체 process에서 공유 |
| `driver::CheckedPosition/Evaluator` | 독립 비교용 B-local 동기 기준선; 공통 경로는 `contracts` 사용 |
| policy identity / 한도 | 공통 설정·E의 manifest·runner 자기 시간 경계 |

평가 adapter는 요청/응답 ID·입력 revision·legal order·model/encoding·정밀도·완료 상태를
검증한 뒤 B에 전달해야 한다. provider는 취소·deadline을 준수하고 물리 작업 완료 전
buffer를 해제하지 않는다. B의 논리 취소만으로 GPU 취소·물리 drain이 완료되지 않는다.
UCI 입출력 owner를 막는 evaluator 호출이나 무제한 worker join을 사용하지 않는다.
현재 두 B crate는 root의 같은 workspace/path 계약을 사용하고 member/lockfile을
공유한다. 같은 소스라도
git/path crate를 둘 다 링크하면 Rust 타입 identity가 달라지므로 하나로 통일한다.

## 검증과 재현

총괄 통합본은 root toolchain Rust 1.96.0을 사용하며 통합 MSRV는 1.90이다. 생성물은 저장소 밖
작업 전용 `CARGO_TARGET_DIR`에 둔다. 해당 작업 output은 작업 종료 전 회수하며,
소스에는 손계산 fixture와 검토 가능한 요약만 보존한다.

```sh
cargo test -p rz-search --all-targets --all-features --locked
cargo fmt --all -- --check
cargo clippy -p rz-search --all-targets --all-features --locked -- -D warnings
```

검사는 독립 score·한/두 ply WDL, root 초기화, 정확 terminal, 중복·취소·reset·
마감 중 검증 race, 자원 한도, 실패 fallback, 시간 산술 및 정책 교체를 대조한다.
fixture는 작은 인공 트리이며 표준 체스 perft·실제 신경망 수치·GPU·대국 검증이 아니다.
`tests/contract_search.rs`는 실제 공통 타입과 비동기 mock으로 context·승격·legal 순서,
명시 head 변환, final submit/backup 경계, matching/foreign 결과·recovery·Drop을 검사한다.
실제 A/C/D adapter의 CPU/mock 연결은 `rz-uci`의 engine binary에서 검사한다.
workspace 통합 CI와 목표 GPU·paired 대국의 실제 상태는
[통합 인수 기록](../../docs/INTEGRATION-STATUS.md)을 따른다.
