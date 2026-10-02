## 문제와 결과

구체적인 문제와 변경 후 동작을 설명합니다. 관련 결정·이슈·연구 후보 ID를 연결합니다.
20개 구현 작업의 `TASK-*`와 핸드오프 연구 `CARD-*` ID를 구별합니다.
책임·의존성과 완료 범위는 [구현 지시서](../docs/IMPLEMENTATION-DIRECTIVES.md)를 따릅니다.

## 진행 공유 상태

WIP는 draft PR로 일찍 공유하고 같은 목표의 PR을 갱신합니다. 현재 구현 범위,
협업자가 사용할 수 있는 인터페이스·제약, 알려진 빌드·테스트 실패, 미실행 항목과
다음 작업을 적습니다. WIP 공유에 기능 완성이나 전체 검사 성공을 요구하지 않습니다.
통합 준비가 되면 필수 검사·계약·리뷰 충족 여부와 실제 검증 근거를 갱신합니다.
작은 의미 단위마다 커밋·push하고 장기 작업의 15~30분 진행 공유를 유지합니다.
커밋 수 상한이나 전체 검사 성공을 중간 공유의 조건으로 두지 않습니다.

- TASK ID·담당·현재 SHA/diff와 다음 연결 작업:
- 구현됨 / 부분 구현 / 알려진 실패 / 미실행:
- 개발 환경 OS·CPU/RAM·GPU/provider capability와 목표 검증 환경:
- 공동작업자가 지금 사용할 수 있는 계약·인터페이스와 제약:

## 책임과 계약

- 담당 기능과 인터페이스, 상태·캐시·평가 관점·취소·마감에 미치는 영향:
- 관련 `crates/rz-*`·설정/도구 경로와 영향 consumer:
- 총괄이 게시한 [CONTRACTS](../docs/CONTRACTS.md) revision, 변경 전/후 revision과 호환성:
- 공통 계약 변경 제안의 이유·독립 인수 벡터·소비자 재검증 범위:
- 해당하는 경우 연구 경로 G/H/C/O 및 변경 유형 E/A/S:
- 새로운 경로·fixture·의존성·외부 코드의 필요성 및 출처·권리:
- 규약 예외와 이유·보완 검증·재검토 조건:

최소 공통 계약의 선언·revision은 총괄이 소유하고 TASK-A03은 계약 적용·검증을
담당합니다. 자체 MIT 코드와 외부 code/weights/data/opening의 권리를 각각 기록합니다.

## 실험 대조와 고정 조건

해당하는 비교만 선택해 기준/변경 manifest·고정값·실제 근거를 적습니다.

- CONTROL-0 → CONTROL-1: 같은 원본 W0, 자체 기준 PUCT S0 → 새 search S1:
- CONTROL-1 → CONTROL-2: 같은 새 search S1, 원본 W0 → fine-tuned W1:
- TASK-D03 runtime baseline/variant: 같은 weights·search·자원에서 변경 한 가지:
- 단일 architecture/ablation: 같은 data·train budget·search·출력 계약과 변경 한 가지:
- 외부 UCI LC0 paired equal-time: 내부 대조와 별도 engine/weight/backend·시계·자원:

## 검증 근거

실제 수행한 명령·설정·결과와 미실행·실패·불확정 범위를 적습니다. 로컬 검사,
관측한 CI, 재사용 근거, 대국 성과를 구분합니다. 문서 작업은 링크·결정 상태·
참조 무결성을 확인하며 존재하지 않는 test·CI의 성공을 기록하지 않습니다.

- CPU/mock·독립 규칙/perft·인공 트리·I/O·요청 수명 검사:
- 실제 weights의 독립 CPU reference 수치 대조:
- 목표 GPU의 같은 weights·policy/WDL·batch·메모리·deadline 검사:
- 실제 자체 Rust 엔진 paired 대국·실패·통계:
- CI 요청 / 관측한 CI / 재사용 확인과 기준 SHA·입력·환경:
- GPU 부재·미지원·skip으로 미실행인 검사, 지원 환경의 검증자·재현 명령·인수 기준:

mock·format 읽기·UCI 연결·GPU launch만으로 실제 NN 완료·GPU parity·arena
강도를 표시하지 않습니다. 검사 명령·toolchain·feature와 실제 시행한 범위를 적고,
총괄 TASK-I02가 영향 소비자·필수 검사·독립 리뷰의 최종 인수 근거를 확인합니다.
GPU 없는 클라우드의 CPU/mock 성공을 목표 GPU 인수로 표시하지 않습니다. 코드 제공·
실제 CPU 수치·GPU 추론·GPU 종단 계측·실제 학습·정식 대국의 상태를 따로 기록합니다.

대국·학습 변경이면 잠긴 manifest, 같은 시작 상태의 pair, 전체 자기 시간·자원,
WDL·pair 통계·신뢰구간·실패·중단과 독립 holdout 근거를 연결합니다.

## 협업·보존

- 완료·진행·미검증 범위와 남은 작업:
- 생성물의 논리 경로·보존 조건, 검토된 공유 요약:
- 브랜치·worktree·일회성 CI의 소유·종료·정리 또는 보존 이유:

원시 로그·PGN·가중치·개인 절대 경로·비밀을 본문에 복사하지 않습니다.
