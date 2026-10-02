# B 담당 탐색·시간 제어

TASK-B02/B03의 CPU 기준 구현이다. 사용자 지시에 따라 총괄의 Rust 계약이 게시되기
전에 B-local 임시 API로 개발했다. 공통 계약 revision은 아직 없으며 아래 접점의
필드·식별·오류·수명을 총괄이 게시한 계약과 후속 통일해야 한다. 루트 workspace,
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

## 총괄 통일·연결 대상

| 현재 임시 접점 | 후속 소유자와 연결 의무 |
|---|---|
| `CheckedPosition` | A의 checked state·legal order·make/unmake·정확 terminal·완전 이력 |
| `Evaluation` / `Evaluator` | 총괄 요청/응답 및 C의 model·encoding·policy order·WDL 관점·수치 검증 |
| `SelectionTicket` / `SearchControl` | 공통 request/selection·game/root·모델 세대·deadline·취소 의미 |
| 동기 evaluator 호출 | D의 async queue/batch·partial/failed/canceled/expired/stale 결과와 유한 drain |
| policy identity / 한도 | 공통 설정·E의 manifest·runner 자기 시간 경계 |

평가 adapter는 요청/응답 ID·입력 revision·legal order·model/encoding·정밀도·완료 상태를
검증한 뒤 B에 전달해야 한다. provider는 취소·deadline을 준수하고 물리 작업 완료 전
buffer를 해제하지 않는다. B의 논리 취소만으로 GPU 취소·물리 drain이 완료되지 않는다.
UCI 입출력 owner를 막는 evaluator 호출이나 무제한 worker join을 사용하지 않는다.

## 검증과 재현

루트 Cargo가 아직 없으므로 crate manifest를 직접 사용한다. 개발 검증 toolchain은
Rust 1.90.0이며 프로젝트 toolchain 결정은 총괄 I01에 남아 있다. 생성물은 저장소 밖
작업 전용 `CARGO_TARGET_DIR`에 둔다. 해당 작업 output은 작업 종료 전 회수하며,
소스에는 손계산 fixture와 검토 가능한 요약만 보존한다.

```sh
cargo test --manifest-path crates/rz-search/Cargo.toml
cargo fmt --manifest-path crates/rz-search/Cargo.toml --check
cargo clippy --manifest-path crates/rz-search/Cargo.toml --all-targets -- -D warnings
```

검사는 독립 score·한/두 ply WDL, root 초기화, 정확 terminal, 중복·취소·reset·
마감 중 검증 race, 자원 한도, 실패 fallback, 시간 산술 및 정책 교체를 대조한다.
fixture는 작은 인공 트리이며 표준 체스 perft·실제 신경망 수치·GPU·대국 검증이 아니다.
실제 A/C/D adapter, workspace 통합 CI, 목표 GPU 및 paired 대국 인수는 후속 작업이다.
