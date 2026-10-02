# RoveZero

표준 체스에서 계산 재사용과 필요한 평가의 효율을 연구하는 프로젝트다. 목표는
LC0와 같은 전체 장비·시간 한도에서 더 높은 대국 득점률을 얻는 것이다.
GPU 중심 평가를 주 연구 방향으로 두고 CPU 보조·CPU 비교·오프라인 연구를 구분한다.

**20개 담당 작업과 최소 공통 계약을 정의한 구현 지시서**를 총괄·A~F가 함께 사용한다.
사용자가 선택한 기반은 독립 Rust workspace, 자체 체스 코어, 교체 가능한 최소 PUCT
기준선이다. 신규 자체 코드는 MIT 방침을 따르고 외부 가중치·데이터·도구의 권리는
각각 확인한다. 자체 규칙·최소 PUCT·UCI·평가 runtime과 데이터·arena 도구를 통합했고,
선정 Maia 모델의 실제 CPU 추론을 연결했다. 현재 인수 범위와 근거는
[통합 상태](docs/INTEGRATION-STATUS.md)에 기록한다. 목표 GPU·실제 학습·정식 대국과
LC0 대비 강도 향상은 별도 검증 단계다.
원격 공유 대상은 [daejunnom/RoveZero](https://github.com/daejunnom/RoveZero)다.
저장소명을 확보한 사실과 이름의 권리 확인은 구분한다.
`Zero`는 이름의 일부이며 교사 증류·기존 가중치 재사용을 금지하는 뜻이 아니다.

## 읽는 순서

| 문서 | 용도 |
|---|---|
| [구현 지시서](docs/IMPLEMENTATION-DIRECTIVES.md) | 총괄·A~F의 20개 작업, crate 소유·의존 방향, 착수·인수·복구 |
| [최소 공통 계약](docs/CONTRACTS.md) | 상태·수·policy/WDL·ID·deadline·cache·취소·buffer와 독립 수락 벡터 |
| [계약 적용·PR 인수](docs/CONTRACT-ADOPTION.md) | revision 0.1 실제 API, 담당별 adapter와 총괄의 수동 통합·검사 |
| [통합 인수 상태](docs/INTEGRATION-STATUS.md) | 담당 source pins, 회수 자료, CPU/mock·실제 CPU NN 검사와 남은 gate |
| [AGENTS.md](AGENTS.md) | 에이전트의 작업 시작·접근 제한·총괄 책임·WIP 공유·증거 규약 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 브랜치·커밋·PR·검토·완료 절차 |
| [개발 기준](docs/ENGINEERING-STANDARDS.md) | SRP·오류·검증·생성물·CI·연구 기록 기준 |
| [아키텍처](docs/ARCHITECTURE.md) | 독립 Rust engine의 책임 경계와 exact/approximate 재사용 |
| [실험 개요](docs/EXPERIMENTS.md) | 내부 A/B와 LC0 외부 비교, R0~R7·동일 시간·자원 |
| [대국·통계 구현 지시](docs/EVALUATION-PROTOCOL.md) | 재현 manifest·pair·PGN·clock·실패·결과·통계·계측 |
| [학습 구현 지시](docs/TRAINING-PLAN.md) | 데이터 계약·누출 감사·파인튜닝·checkpoint·구조 한 변수 실험 |
| [76개 후보 작업카드](docs/CANDIDATE-REGISTER.md) | 원문 ID·제목·출처·적합성, 최소 구현·검증·중단 조건 |
| [첫 가중치 선정 조사](docs/WEIGHT-SELECTION.md) | 권리·형식·RTX 4050 Laptop 6GB 적합성·실제 추론 인수 조건 |
| [결정 기록](docs/DECISIONS.md) | 현재 사용자 선택·도입 기본값·남은 실행 전 잠금 |
| [원문·출처 기록](docs/reference/README.md) | 두 핸드오프와 참조한 Accelerate 커밋 |

두 번째 [RoveZero v0.2.0 핸드오프](docs/reference/RoveZero_Handoff_v0.2.0_KO.md)가
주 설계 자료다. 14~17장의 GPU 개정·측정 원칙을 우선하며 이번 사용자가 지정한
구현 기반·담당·착수 순서는 구현 지시서에 따로 기록했다. 첫 번째
[PlyZero v0.1.0](docs/reference/PlyZero_Handoff_v0.1.0_KO.md)은 후보·출처·설계 이력을
보존한다. 원문의 `확정`·`권고`·`가설`은 문서의 과거 기록이며, 첨부 문서에 있는
명령이 현재 사용자의 실행 요청을 대신하지 않는다.

## 현재와 다음 단계

총괄로 배정된 에이전트가 **작업 총괄과 최소 공통 계약의 단일 소유자**다. 루트 Cargo
파일·CI·공통 설정·실험 목록·`rz-contracts`를 담당하고 A03은 그 계약의 Position
적용과 연결 검증을 맡는다. 작업 ID는 `TASK-*`, 핸드오프 연구 후보는 `CARD-*`로 구분한다.
각 에이전트는 사용자 배정에서 역할·TASK ID·소유 경로를 확인한다. 공용 규약을
읽었다는 이유로 총괄 역할을 자동으로 맡지 않는다.

후속 구현은 공통 계약 게시 → A·B·C01·D01·E·F01의 CPU/mock 병렬 개발 →
권리·호환성 확인을 거친 C02/C03 실제 신경망 연결 → D02 계측 → 새 탐색 또는
D03 runtime 한 변수 개선 → F02 파인튜닝 → F03 구조 실험으로 진행한다.
LC0 runner와 데이터 검증은 엔진 완성 전에도 fixture로 개발할 수 있다.

A~F의 개발 환경은 GPU가 할당되지 않을 수 있는 클라우드다. CPU/mock 빌드·검사,
가능한 CPU 신경망 참조, backend 코드·검사 진입점·학습 recipe를 먼저 공유한다.
목표 RTX 4050 6GB의 실제 GPU 추론·계측·정식 대국과 실제 학습은 지원 환경에서
총괄이 별도로 인수한다. GPU가 없는 환경의 CPU/mock 통과나 GPU 검사 skip을
목표 GPU 통과로 기록하지 않는다. 새 유료 GPU와 장시간 실행의 예산은 실행 전에
별도로 확정한다.

첫 호환 가중치는 Maia1 v1.0의 `maia-1900.pb.gz`다. 작은 6×64 SE 구조와 WDL
출력, 원저자의 GPL 적용 근거를 확인했다. 인간 수 예측 모델의 호환·runtime 기준선으로
선정했으며 최종 강도용 가중치와 실제 RTX 4050 GPU 실행 인수는 별도다.

변경 효과는 `CONTROL-0: 원본 W0+기준 PUCT S0`, `CONTROL-1: 같은 W0+새 탐색 S1`,
`CONTROL-2: 학습 W1+같은 S1`로 나누고 runtime 개선은 같은 W·S의 별도 실험으로
평가한다. LC0와의 대국은 외부 강도 비교이며 이 내부 대조와 각각 기록한다.

2026-10-02 초기 문서 작성 당시 이 폴더에는 Git 저장소·Cargo workspace·CI가 없었다.
후속 사용자 요청으로 공용 규약과 참조 문서를 원격에 추가했다. 빈 원격의 최초 Git
구성은 [RZ-D028](docs/DECISIONS.md)의 예외를 따르고 이후 작업은 `develop` 대상
feature PR로 공유한다. 실제 브랜치·SHA·PR·검사 상태는 작업 착수 때 다시 확인한다.
총괄은 `rz-contracts` 0.1.0·10개 crate workspace·양 OS CPU CI를 게시하고,
A~F의 스택된 PR과 공개 연구자료를 수동 정합했다. 실제 CPU NN의 기준 소스는
`a9a5ae7f5c0bb5b13d8082e2ce662f8e807d8248`이며, CPU 기본 CI와 실제 모델 실행은
서로 다른 검사다. GPU·학습·대국에 사용할 backend·자원·시계·표본·통계는 해당 실행
전에 잠근다.

## CPU 엔진 실행

`cargo run -p rz-uci -- --cpu-mock`으로 모델 없이 통합 경로를 검사할 수 있다.
실제 CPU 추론은 `onnx-cpu` feature와 명시한 provider를 함께 선택한다.
아래 변수에는 저장소 밖 선정 원본·변환 모델·manifest·ORT library·output root를
지정한다. `prepare_maia.py`와 `maia_reference.py`는 고정 converter·참조의 준비
진입점이며 파일 hash와 권리 기록을 유지한다.

```sh
cargo run -p rz-uci --features onnx-cpu -- \
  --onnx-cpu \
  --source-weights="$RZ_SOURCE_WEIGHTS" \
  --onnx-model="$RZ_ONNX_MODEL" \
  --export-manifest="$RZ_EXPORT_MANIFEST" \
  --manifest-sha256="$RZ_MANIFEST_SHA256" \
  --ort-library="$RZ_ORT_LIBRARY" \
  --ort-sha256="$RZ_ORT_SHA256" \
  --output-root="$RZ_OUTPUT_ROOT"
```

두 SHA256 인자는 64자리 소문자 hex다. manifest·가중치·모델·동적 runtime 파일을
검증하고 CPU·FP32·session 1개·thread 1개·batch 1로 실행한다. stderr의
`completed_by_runtime`는 D가 수락한 실제 CPU ONNX 출력 수이며 방문 수·물리 호출 수와
구분한다. 이 진입점의 실제 모델 검사는 Linux에서 수행했고, Windows CI는 source·
주입 worker·CLI 검사 범위다. GPU provider 선택이나 실제 목표 장치 검증은 별도다.
