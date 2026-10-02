# D02 검증 자료 보존

이 폴더는 D 소유 검증 자료의 공개 보존본이다. 새로운 엔진 기능은 추가하지 않는다.
재검사 소스는 `aa2fe1c3ac38274912017412e080a9bbd14c9485`, 공통 계약 pin은
`67284c4f66f7a7ae9f46fa63dfd50e7410eb6845`다. 계약은 runtime의 명시적
`contracts` feature에서만 사용하며 CPU trace는 기본 feature로 실행된다.

## 보존한 자료와 출처

- `traces/original/`: 작업 환경에 남아 있던 원본 TSV 8개, 184,296 bytes.
  민감정보 검사를 통과해 바이트를 변경하지 않았다. 원본 SHA-256·파일 크기·
  관측한 mtime·입력 상당값·출처의 한계는 [ledger.json](ledger.json)에 있다.
- `logs/recapture/`: 이번 보존 작업에서 기존 검사 8개를 다시 실행한 로그다.
  과거 검사 stdout의 원본이 아니다. 원래 합친 stdout/stderr를 저장한 뒤 작업
  절대경로만 공개용 placeholder로 바꾸었다. [receipts.json](receipts.json)에
  UTC 시작/종료, 정확한 argv, 환경 override, exit code, 원본/공개 SHA-256이 있다.
  조용히 성공하는 fmt 로그 2개는 빈 파일이며 영수증의 exit code는 0이다.
- [inputs.json](inputs.json): 고정 소스에서 재구성한 합성 입력 규칙과 유한 설정이다.
  원래 실행 당시 요청별 입력을 캡처한 파일이 아니다.
- [plan.json](plan.json): 이번 보존 계획이다. 원래 구현/실험 계획의 원본이 아니다.
- [results.json](results.json): 원본 TSV에서 추출한 결과와 새 검사 69개 테스트의
  결과다. 원자료의 관측 손실·미확인 provenance를 그대로 유지한다.
- [privacy-review.json](privacy-review.json): 공개 전 검사 범위·탐지 규칙·마스킹·제외 항목.
- [SHA256SUMS](SHA256SUMS): 이 파일을 제외한 보존 파일 전체의 파일 목록과 SHA-256.

전체 크기는 작아 일반 Git에 보존한다. build cache·toolchain·외부 데이터·가중치·
바이너리는 넣지 않았다. E/F의 소유 자료는 이 D 보존 범위 밖이며, 해당 자료의 보존
완료를 주장하지 않는다. 원래 TSV와 재검사 전 공개용 원본 로그는 작업 환경에도 남긴다.

## 실제/합성 구분과 해석

TSV는 프로그램이 출력한 **시나리오별 집계 profile**이다. 요청별 사건 stream의
원본 dump가 아니다. 입력·ID·관계 label과 단계 시계는 자체 합성 fixture다.
CPU 정수 계산과 `std::time::Instant` 벽시계 측정은 실제 실행이다. 실제 체스 상태,
신경망 추론, GPU 전송/VRAM, 학습, 대국, D03 개선 실험은 미실행이다.

`all-release` 및 `all-rust1.96`의 정상 32요청 trace는 각 시나리오에서 실제 소비 31,
물리 완료 32, 취소 1, 최종 host/device/pinned 예약 0을 보인다. 유한 ID 이력이
eviction되므로 `duplicate_checks_cover_all_ids=false`이며 전체 과거 ID의 중복
검사 증거로 쓰지 않는다. 예약 수치는 scheduler ledger이며 process RSS가 아니다.

`zero`, `loss`, `long-loss`, `long-zero`는 의도적인 관측 비활성/포화 진단이다.
transport 손실·tracking 거부·관측 오류 때문에 collector profile이 불완전하다.
`actual_consumed_requests`와 collector의 `consumed_requests`를 구분한다.
GPU transfer의 `na`와 누락 필드를 0 지연이나 성공 측정으로 해석하지 않는다.

## 알려진 누락과 불확실성

과거 검사 stdout/stderr, 원래 실행 영수증·정확한 실행 checkout SHA·UTC 실행 시각,
요청별 입력 및 사건 stream, 별도 원래 계획 파일은 파일로 남아 있지 않았다.
mtime은 관측한 파일 metadata이며 실행 시각의 증명이 아니다.
최종 release TSV 4개는 이전 채팅의 실행 기록상 현재 관련 소스에 연결되지만,
TSV 자체에 source SHA가 없다. 원장에 이 기록 근거와 미확인 항목을 구분한다.
초기 개발 TSV 4개의 정확한 소스와 argv는 `unknown`으로 보존한다. 현재 소스로
다시 실행해도 초기 필드 버전 및 벽시계 수치까지 같다는 보장은 없다.

## 무결성 확인과 재현

이 폴더에서 원격으로 받은 모든 보존 파일을 확인한다.

```sh
sha256sum -c SHA256SUMS
```

재현에는 Rust 1.96.0(rustfmt/clippy 포함), MSRV 검사에 Rust 1.85.0, lockfile에
고정된 git 계약 checkout이 필요하다. `--offline` 실행 전에 필요한 의존성을
준비한다. 다음 명령은 고정 소스의 별도 worktree를 만들고 기존 검사·trace recipe를
다시 실행한다. 새 결과와 빌드 출력은 지정한 저장소 밖 디렉터리에 보존한다.
작업 checkout 및 기존 원본을 삭제하지 않는다.

```sh
export RZ_D_REPO=/absolute/path/to/RoveZero
export RZ_D_REPLAY_ROOT=/absolute/path/outside/RoveZero/d-replay-new
# 필요하면 RZ_D_CARGO, RUSTUP_HOME, CARGO_HOME을 지정한다.
bash "$RZ_D_REPO/benches/runtime/evidence/aa2fe1c-d02/reproduce.sh"
```

실제로 실행한 검사 명령과 시간은 영수증을 기준으로 한다. 재현 스크립트의 trace
명령은 원장에 확인된 설정을 현재 고정 소스로 실행하는 recipe이며, 미확인 과거
argv를 복원했다는 뜻은 아니다. 스크립트는 shell 문법을 확인했으며 여기서 다시
전체 실행하지 않았다. 이번 재검사는 영수증에 기록한 검사 8개만 실행했다.
