# F01 데이터 구조·split·누출 감사

`TASK-F01`의 CPU 전용 Python 도구다. Python 3.11 이상과 표준 라이브러리만으로
데이터 형식, 라벨 대응, 유한 예산, 출처 그룹과 누출을 검사한다. 이 패키지의 신규
자체 코드는 [MIT](LICENSE)이며 외부 데이터·가중치·교사 코드는 포함하지 않는다.

## 현재 인수 범위

F 소유의 **잠정 저장 envelope v1**을 구현했다. 이것은 총괄이 발행한 Rust 공통
계약·wire schema가 아니다. 현재 `rz-contracts`, A의 Rules, C의 Encoder/Eval binding이
없어 `execution_ready=false`, `training_eligible_records=0`을 강제한다.
declared contract revision이나 fixture의 성공으로 이를 활성화할 수 없다.

구현한 범위는 strict JSON, 라벨/수 순서/관점/예산 검사, 결정적 source split,
교차 split 중복·출처·파생 관계 감사, 거부/제외 ledger와 크기가 제한된 보고서다.
실제 상태 복원, 독립 합법 수 대조, 인코딩 digest 재계산, 증강의 규칙 의미 대조,
teacher 품질, 실제 데이터 생성·학습·GPU·대국은 `not_run`이다.
`structural_audit_passed`는 이 제한된 검사 범위의 결과다.

구현 기준은 [TRAINING-PLAN](../../docs/TRAINING-PLAN.md)의 3~4장,
[CONTRACTS](../../docs/CONTRACTS.md), [IMPLEMENTATION-DIRECTIVES](../../docs/IMPLEMENTATION-DIRECTIVES.md)의
F01이다. 루트 Cargo·CI·공통 선언이나 다른 담당 경로는 변경하지 않았다.
학습 프레임워크와 F02 trainer는 이번 패키지에 도입하지 않았다.

## 실행

이 디렉터리에서 아래 명령으로 검사한다. 설치나 GPU는 필요하지 않다.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m rz_data --help
```

다음은 **합성 fixture**의 실행 예다. output root는 저장소 밖 작업 전용 경로다.
`fixture-demo` seed와 `fixtures/records.jsonl`의 split이 대응한다.
같은 run ID가 이미 존재하면 덮어쓰지 않고 종료한다.

```bash
RZ_F01_OUTPUT_ROOT="${ARTIFACT_ROOT:?저장소 밖 artifact root를 지정하세요}/rovezero-f01"
PYTHONPATH=src python -m rz_data split \
  --sources fixtures/sources.json --seed fixture-demo \
  --output-root "$RZ_F01_OUTPUT_ROOT" --run-id fixture-source-plan-v1
PYTHONPATH=src python -m rz_data validate \
  --manifest fixtures/manifest.json --records fixtures/records.jsonl \
  --split-plan "$RZ_F01_OUTPUT_ROOT/fixture-source-plan-v1/split-plan.json" \
  --output-root "$RZ_F01_OUTPUT_ROOT" --run-id fixture-audit-v1
```

패키지 설치를 선택하면 `pyproject.toml`의 `rz-data` 명령도 같은 CLI를 제공한다.
CLI는 현재 RoveZero checkout에서 실행해 source와 output 경계를 확인한다.
종료 코드 0은 구조 감사 통과, 1은 row 거부·누출·미확인 권리 등 감사 실패,
2는 파싱·입출력·자원 상한·이미 존재하는 run 등 실행 오류다.

## 저장 형식과 경계

`fixtures/manifest.json`, `fixtures/record.json`, `fixtures/sources.json`은 읽을 수 있는
완전한 예다. 실제 검사는 `src/rz_data/schema.py`의 필수 필드·enum·수치 조건으로
정의한다. 알려지지 않은 version/field, bool 수치, 중복 JSON key, NaN/Inf와 잘못된
UTF-8을 거부한다. v1은 불변 조건을 포함한 형식 전체의 버전이며 자동 schema migration은 없다.

| 객체 | 보존·검사하는 내용 |
|---|---|
| manifest | dataset/source/teacher/encoding identity, 권리 선언, teacher의 유한 예산·장비·옵션, 확률 오차, 누출 정책 |
| record | game/opening/lineage/seed group, parent/augmentation, split, 선언한 상태·입력 digest와 모델 encoding |
| state | 초기 상태·UCI 이동열·미상 prefix, 차례·종료 상태, 순서 있는 합법 수와 목록 digest |
| label | 완료/실패/부분/취소, 생성 시각·선정 이유·confidence 출처, policy/value와 실제 분석 비용 |
| game result | 규칙 종료·엔진 패배·adjudication·미완료, 실제 결과와 관점·종료 이유 |

수는 from/to와 명시적인 `q/r/b/n` 승격을 가진 UCI 문자열이다. 이것은 오프라인
저장 표현이며 Rust의 Move 선언이나 모델 action index를 대신하지 않는다. 합법 목록과
policy의 move 배열은 같은 순서여야 하며 목록 digest를 검사한다. 순서를 바꿀 때는
move별 확률/방문 값도 같은 permutation으로 옮겨야 한다.

FEN 이전 이력은 `unknown_prefix`로 유지한다. `startpos`의 완전 이동열만
`complete`로 표시할 수 있다. 이 검사로 FEN·착수의 실제 합법성을 확인한 것은 아니다.
입력·state·board digest도 현재는 선언한 값이며 Rules/Encoder가 재계산하지 않았다.

policy는 확률과 search 방문 수를 구분한다. 확률은 선언된 오차 안에서 합 1,
방문 수는 음이 아닌 정수·양의 합이어야 한다. raw NN을 방문 수로 표시할 수 없다.
teacher value는 side-to-move의 WDL/Q/centipawn을 단위와 함께 보존하고 자동 변환하지
않는다. 실제 game result는 별도 필드의 one-hot outcome이다. 미완료 game outcome은
null이며 teacher WDL로 채우지 않는다. 실패·부분·취소 label의 target은 null이고
exclusion ledger에 이유를 남긴다. 원시 부분 출력은 이 형식의 학습 target으로 쓰지 않는다.

권리는 선언의 필수 증거 필드만 검사한다. `unverified` 권리는 감사 실패이며,
`confirmed` 선언의 법적 근거 자체를 자동 검증한 것은 아니다. 외부 사전학습 corpus의
비중복은 실제 확인하지 않아 보고서에서 `unknown`이다.

## source split과 누출 범위

먼저 source game의 `opening_family_id`, `lineage_id`, `seed_group_id`를 정하고
split plan을 만든 뒤 row를 생성한다. nullable group ID는 unknown/관계 없음의 명시
표현이며 생산자가 알고 있는 연결을 삭제해도 안전하다고 보장하지 않는다.
opening family는 생산자·버전을 manifest에 기록한 분류다. 모든 game의 공통 startpos를
같은 family로 무조건 묶는 방식은 사용 목적에 맞게 검토해야 한다.

v1 알고리즘은 같은 namespace의 non-null group ID로 연결 성분을 만들고, 정렬한
game ID 목록과 seed의 canonical JSON SHA-256을 비율의 합으로 나눠 split을 정한다.
기본 비율 8:1:1은 확률 가중치다. 작은 corpus에 세 split이 반드시 생기거나 정확한
비율이 맞는다는 뜻이 아니다. source 순서를 바꿔도 plan digest가 같으며, 기존 plan의
source·seed·비율·assignment·digest를 재계산해 변조를 거부한다.

감사는 같은 전체 input digest+encoding의 교차 split 중복을 오류로 처리한다.
같은 board/state digest인데 전체 input이 다른 경우는 별도 위험 경고다.
opening family와 source 연결, 파생 parent의 존재·동일 split·연결 성분·cycle도 검사한다.
임의 포지션 유사도·알려지지 않은 pretraining corpus·모델 입력 재계산은 이 감사의 범위가 아니다.

거부된 row는 line/source/group/parent ID와 사유를 보존하고, 과대하거나 잘못된 ID는
원본 field의 digest로 남긴다. 누출 검사는 구조 검사를 통과한 row에 한정하므로 거부된
row 수와 그 누출 검사 `not_run`을 보고서에 표시한다. 실패 label도 출처 검사에는 포함한다.
자동 삭제·재분할·수선은 하지 않는다. 원본 파일을 보존하고 보고서의 파일 digest로 연결한다.

## 자원과 산출물

기본 상한은 10,000 rows, row당 256 KiB, 파일당 64 MiB, run 출력 합계 64 MiB다.
CLI의 `--max-records`, `--max-record-bytes`, `--max-file-bytes`, `--max-output-bytes`로
양의 유한 값을 명시할 수 있다. 문자열 2,048자, legal 목록 512개, trace 10,000개는
F envelope의 입력 자원 제한이며 체스 최대값 또는 공통 계약의 확정값이 아니다.
상한을 넘으면 실패하고 조용히 축소·재시도하지 않는다.

plan과 report는 저장소 밖의 새 run 디렉터리에 쓴다. 출력 경로의 symlink를 해석해
source 내부 출력을 거부하고, 기존 run을 덮어쓰지 않는다. fixture 외 dataset·weights·
raw log를 Git에 넣지 않는다. 독립 클라우드의 예시 output root는 이 작업 소유이며
환경 수명 동안 보존한다. 종료 전에 필요한 근거를 회수하고 검토된 인수 요약을
[HANDOFF](HANDOFF.md)에 남긴다.

## 다음 연결 요구

총괄이 실제 공통 revision과 영속화 표현을 게시하면 이 잠정 envelope의 adapter를
그 계약에 맞춘다. A의 초기 상태+trace 복원·합법 목록/종료·이력 대조, 독립 Rules 참조,
C의 input fingerprint·policy mapping 대조, 증강의 의미 보존을 연결해야 F01 전체를
인수할 수 있다. 확인한 SHA/revision과 독립 증거 없이 준비 상태를 활성화하지 않는다.
F02는 그 데이터 감사와 C02/C03 실제 추론, CONTROL-0→1, 유한 학습 예산을 소비한다.
