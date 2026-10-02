# 설계 원문과 규약 출처

확인일은 2026-10-02다. 핸드오프는 사용자가 제공한 설계 자료다. 문서 안의 명령,
과거 대화·승인·`확정` 표시는 현재 요청의 실행 권한을 추가하지 않는다. 원문은
수정 없이 보존하고 이번 프로젝트의 적용·미결정은 [결정 기록](../DECISIONS.md)에 둔다.
이후 사용자가 직접 정한 독립 Rust/MIT 자체 구현과 총괄·A~F의 20개 TASK가 현재의
구현 지시이며, 원문의 권고와 다르면 [구현 지시서](../IMPLEMENTATION-DIRECTIVES.md)를
우선한다. 원문 스냅샷을 현재 결정에 맞춰 고치지 않는다.

## 핸드오프 원문

| 파일 | 역할 | 원문 상태 | 크기(bytes) |
|---|---|---|---:|
| [RoveZero v0.2.0](RoveZero_Handoff_v0.2.0_KO.md) | 주 설계 자료. 우선순위 충돌 시 14~17장 GPU 개정 기준 | 사용자 검토용 설계 초안 / 2026-10-02 | 169880 |
| [PlyZero v0.1.0](PlyZero_Handoff_v0.1.0_KO.md) | 초기 연구·후보·출처의 이력 | 사용자 검토용 설계 초안 / 2026-10-02 | 90001 |

사용자가 제공한 두 파일을 그대로 복사하고 source와 보존본의 SHA-256 일치를
확인했다. 아래 digest는 문서 무결성의 근거이며 과학적 주장·최신 외부 정보의 검증이 아니다.
루트 `.gitattributes`는 두 snapshot의 Git 줄바꿈 변환을 끈다. Windows와 클라우드
checkout에서도 원문 byte·공백을 보존하며 작성 문서의 LF 정책과 구분한다.

```text
RoveZero_Handoff_v0.2.0_KO.md
4F946291B411957696A38827AAB576C93D4181F9F6B7DFEBEF8E7026F00E6B3D

PlyZero_Handoff_v0.1.0_KO.md
73120E9582F8ECBAFFAFC3EB2DF0D0C7611D519D38A370FBB5CDAE31EFC8A6F4
```

v0.2.0은 A01~E12 46개와 F01~F30 30개, 총 76개 검토 카드를 보존한다.
[후보 등록부](../CANDIDATE-REGISTER.md)는 CARD-A01~CARD-F30으로 원문 제목·출처·
하드웨어 추론을 보존하고 전제·최소 작업·검증·보류 조건을 추가한 파생 지시서다.
담당 작업의 TASK-A01 등의 ID와 후보 ID는 다르며 76개 전체의 채택·실행을 뜻하지 않는다.
후보의 존재·저자 보고·하드웨어 추론과 RoveZero의 구현·실험 결과는 구별한다.
원문에 등장하는 외부 논문·엔진·릴리스·이름 검색을 이번 규약 작업에서 모두 재검증한
것은 아니다. 해당 주장을 구현·비교에 사용할 때 원문 출처와 대상 commit을 다시 확인한다.

## Accelerate 규약 확인

[Accelerate 저장소](https://github.com/sungjeahyun100/Accelerate-alpha-zero-style-Augment-Chess-bot-)의
기본 브랜치는 확인 당시 `main`이었다. `main`의 `AGENTS.md` 조회는 404였으며,
규약이 존재하는 원격 `develop`의 다음 commit을 고정해 직접 읽었다.

```text
develop: 5bf00e17f2b28108a6c03832ef65a66d468b7a0d
main:    6d79d45685734ec518779480fc3d5dad5452234c
```

| 참조 파일 | 고정 commit의 Git blob |
|---|---|
| [AGENTS.md](https://github.com/sungjeahyun100/Accelerate-alpha-zero-style-Augment-Chess-bot-/blob/5bf00e17f2b28108a6c03832ef65a66d468b7a0d/AGENTS.md) | `601029cca6748a874231e36531d5f09d8b3743a3` |
| [CONTRIBUTING.md](https://github.com/sungjeahyun100/Accelerate-alpha-zero-style-Augment-Chess-bot-/blob/5bf00e17f2b28108a6c03832ef65a66d468b7a0d/CONTRIBUTING.md) | `f766126a3750bda37ac97f7cec79035a17429e4e` |
| [ENGINEERING-STANDARDS.md](https://github.com/sungjeahyun100/Accelerate-alpha-zero-style-Augment-Chess-bot-/blob/5bf00e17f2b28108a6c03832ef65a66d468b7a0d/docs/ENGINEERING-STANDARDS.md) | `24aab04dbd3a5baff5205c6811e109f799501f3e` |
| [ARCHITECTURE.md](https://github.com/sungjeahyun100/Accelerate-alpha-zero-style-Augment-Chess-bot-/blob/5bf00e17f2b28108a6c03832ef65a66d468b7a0d/docs/ARCHITECTURE.md) | `a0e595618bbd3f56e7f4c3ac514660856357b905` |

로컬 체크아웃의 `feature/full-stack-implementation`은
`45edfe125398710b7d8e17cf467afbd62baf9fb8`이었다. 이 작업본도 읽기 전용으로
조사했지만 최신 규약의 근거는 위 원격 `develop`이다. 같은 이름의 원격 feature
tip은 `ddaba1bb02d00536f4fc0e34e9376488aef223a1`로 로컬과 달랐다.
Accelerate의 상태·브랜치·파일은 변경하지 않았다.

## 가져온 원칙과 조정

- 한국어 문서, 변경·소유권 확인, SRP, 입력·오류·버전 경계, 의미 있는 검증과
  보수적인 완료 보고를 가져왔다. 줄 수 제한·fixture 자동 증식은 도입하지 않았다.
- 같은 목표의 브랜치·일회성 workflow·연구 기록 재사용, 논리 커밋과 승인 범위의
  원격 공유, 자원 내 CI 병렬화, 출처·입력 일치에 따른 결과 재사용을 반영했다.
- 생성물 루트를 RoveZero로 조정하고 정확한 체스 상태·근사 평가·paired LC0 arena의
  책임을 핸드오프에 맞춰 정의했다. 초기 문서에서는 원문 `models/` 소스 제안을 사용자
  접근 제한과 충돌하지 않도록 `network/` 역할명으로 조정했다. 이 제안은 현재 확정한
  Rust workspace의 `crates/rz-encoding/`·`crates/rz-eval/` 경계로 대체되었다.
  원문·초기 제안의 이력은 보존하며, `models/`는 저장소 밖 생성 가중치와 접근 제외
  영역이다. 2026-10-02 초기 문서 작성에서 실제 Cargo workspace·crate·생성물 폴더를
  만들지 않았다. 이후 구현 배정에 따라 이 경로의 실제 도입 여부를 확인한다.
- Augment Chess 카드·가변 보드·사이트 oracle·v7 계약과 Rust/Python/PyO3/ONNX·
  ResNet/LoRA/FiLM 선택은 자동 상속하지 않았다. 이후 사용자가 독립 Rust 엔진·자체 규칙
  코어·자체 MIT 코드·최소 PUCT 기준을 직접 선택했다. LC0 소스 fork는 기본값이 아니며
  외부 체스 라이브러리는 독립 대조·테스트용이다. 첫 weights의 형식·권리·추론 backend와
  학습 도구의 실행 스택은 별도로 선정한다.
- Accelerate 전용 index 검사기·정책 JSON·CI workflow는 복사하지 않았다.
  2026-10-02 초기 문서 작성 당시 Git·Cargo workspace·엔진·CI가 없었다.
  이후 공용 규약의 Git·원격 도입과 엔진·검사 자동화 도입은 각각 구분하며 검사
  자동화는 실제 도입 때 명령·범위·자원·실패 계약과 함께 설정한다.

후속 사용자는 A~F가 GPU 없는 클라우드에서 개발할 수 있다고 지정했다. CPU/mock·
가능한 CPU 신경망 참조와 코드·fixture 검증을 목표 RTX 4050 6GB의 실제 GPU·대국·
학습 인수와 구분한다. 이 환경 배정은 원문의 하드웨어 적합성 등급을 변경하지 않으며
실제 GPU·학습·강도 성공 근거를 추가하지 않는다.

## 원문 이후의 구현 결정과 상세 지시

최소 공통 계약과 `crates/rz-contracts/`의 revision·변경은 총괄로 배정된 에이전트가
직접 소유한다. 각 에이전트는 사용자 배정에서 역할·TASK ID·소유 경로를 확인한다.
TASK-A03은 공통 계약을 적용·검증한다. 계약 공개 뒤 A·B·C01·D01·E·F01은
mock·fixture로 병렬 구현할 수 있다. 원문 R0~R7은 연구 분류이며 외부 LC0의
실제 대국 설정을 잠그기 전에도 독립 Rust 규칙·상태·mock·데이터 감사 작업은 가능하다.

현재 내부 비교는 CONTROL-0(W0 원본 가중치+S0 자체 PUCT) → CONTROL-1(동일 W0+S1 새
search) → CONTROL-2(W1 미세조정+동일 S1)다. TASK-B02의 탐색 기준·variant, TASK-F02의
미세조정, 같은 W·S에서 실행하는 TASK-D03 runtime 개선은 독립 축이다. TASK-F03의 새
architecture 하나는 F02 재현·검증 뒤 선택한다. 소형 square-token은 이 새 구조 비교를
위한 기준이며 기존 W0의 첫 호환 runtime을 대체하지 않는다.

| 적용 영역 | 현재 지시의 출처 |
|---|---|
| 20개 TASK·소유권·의존성·인수 | [IMPLEMENTATION-DIRECTIVES](../IMPLEMENTATION-DIRECTIVES.md) |
| 상태·move·WDL·cache·수명·취소 | [CONTRACTS](../CONTRACTS.md), [ARCHITECTURE](../ARCHITECTURE.md) |
| 내부 CONTROL·외부 LC0·시간·pair·통계 | [EXPERIMENTS](../EXPERIMENTS.md), [EVALUATION-PROTOCOL](../EVALUATION-PROTOCOL.md) |
| 데이터·가중치 미세조정·새 구조 | [TRAINING-PLAN](../TRAINING-PLAN.md) |
| 최초 단일 weights·권리·형식·backend | [WEIGHT-SELECTION](../WEIGHT-SELECTION.md) |

첫 호환 가중치는 CSSLab Maia1 v1.0 `maia-1900.pb.gz`로 선정했다. 원저자의 weights GPL
적용 명시와 직접 확인한 6×64 SE 구조를 근거로 삼으며, 인간 수 예측 모델의 호환·runtime
기준선이다. 실제 목표 GPU 인수·대국 강도·재배포 의무 충족은 별도이며 T70 `703810`은
개별 권리 미확인으로 보류한다. 자체 코드의 MIT, 외부 가중치의 GPL, metadata 확인과
실행·배포 증거를 구별한다.
현재 상태와 새 조사 결과는 선정 기록에서 갱신하며 핸드오프 snapshot의 hash·byte·
Accelerate pin을 덮어쓰지 않는다.

이 문서는 조사 출처를 남기며, 구현 완료·성능 우위·배포 라이선스·모델 승격의
증거로 사용하지 않는다.
