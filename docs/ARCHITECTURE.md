# 아키텍처와 구현 책임

현재 모델/외부 대국 교체 경계는 [모델 어댑터·외부 UCI](MODEL-ADAPTER-EXTERNAL-UCI.md)를
따른다. associated input/raw 타입으로 LC0와 entity mock을 연결하며 Rules·PUCT·평가
계약 revision 0.1·runtime 물리 수명은 유지한다. 외부 UCI는 대국 계층에 독립 연결한다.

이 문서는 [RoveZero v0.2.0 핸드오프](reference/RoveZero_Handoff_v0.2.0_KO.md)의
3~4장과 14~17장 및 이번 사용자 지시를 바탕으로 책임·의존 방향·검증 경계를 정한다.
사용자가 지정한 독립 Rust 구현의 책임과 의존 방향을 정한다. 현재 구현·실행 인수
범위는 [통합 상태](INTEGRATION-STATUS.md)에 따로 기록한다.
A~F의 클라우드 개발 환경과 목표 RTX 4050 6GB 검증 환경을 구분한다. 개발 GPU가
없어도 CPU/mock와 가능한 CPU 신경망 참조로 계약·코드·fixture를 구현할 수 있어야 한다.
실제 GPU 실행·계측·정식 대국·학습은 지원 환경에서 총괄 I02가 별도로 인수한다.
GPU capability 부재를 기록하고 GPU 요구 모드의 조용한 CPU 대체를 금지한다.
작업·인수 순서는 [IMPLEMENTATION-DIRECTIVES](IMPLEMENTATION-DIRECTIVES.md),
타입·필드·수명·실패 계약은 [CONTRACTS](CONTRACTS.md)가 구체화한다. 결정 상태는
[DECISIONS](DECISIONS.md), 변경·검증 절차는
[ENGINEERING-STANDARDS](ENGINEERING-STANDARDS.md), 대국 계약은
[EXPERIMENTS](EXPERIMENTS.md)를 따른다.

## 범위와 근거

대상은 표준 체스다. 증강체스의 카드·가변 보드·새 규칙은 초기 범위에 포함하지 않는다. 규칙 상태와 모델 입력은 분리한다. 자연어를 생성하며 직접 착수하는 LLM 엔진은 초기 구조의 목표가 아니다.

제품의 규칙·UCI·검색·추론 연결·런타임은 **독립 Rust workspace**로 작성한다.
새 자체 코드는 MIT 정책을 적용하며 외부 가중치·데이터·교사·opening·도구의 권리는
각각 확인한다. LC0 fork나 외부 chess 라이브러리를 제품 규칙 코어로 삼지 않는다.
외부 체스 구현은 독립 대조·테스트에 사용할 수 있다. Accelerate의 Rust/Python/
ONNX 경로를 옮긴 결정이 아니며 RoveZero의 사용자 지시가 근거다.

정확한 규칙·상태를 보존하고 학습된 특징·평가의 근사는 별도로 다룬다. 자체 최소
PUCT와 deterministic mock을 첫 검색 기준선으로 사용한다. 최종 검색 알고리즘은
비교 실험으로 검토하며 최소 PUCT를 최종 선택으로 고정하지 않는다. 첫 호환 형식은
[WEIGHT-SELECTION](WEIGHT-SELECTION.md)의 Maia1 v1.0 6×64 SE/WDL로 선정했다.
최종 강도용 모델 구조, 정밀도·추론 backend, 학습 도구 언어, 실행 예산은 해당
선정/실행 단계에서 잠근다. RTX 4050 6GB 자료상 적합성과 실제 지원·속도는 구분한다.

핸드오프끼리 충돌하면 v0.2.0의 14~17장을 앞 장과
[PlyZero v0.1.0](reference/PlyZero_Handoff_v0.1.0_KO.md)보다 우선한다. 최신 사용자
지시의 20개 작업과 순서는 핸드오프의 권고보다 우선한다. CPU의 작은 증분 갱신이
GPU에서도 유리하다고 가정하지 않으며 하드웨어 적합성, 공개 구현 사례, 저자 보고,
RoveZero 실측을 구별한다. `TASK-C02`는 가중치 호환 작업이고 `CARD-C02`는 반복
잠재 추론 후보다. 작업 ID와 [후보 등록부](CANDIDATE-REGISTER.md)의 연구 ID를 구분한다.

## 연구 경로

G/H/C/O는 실험과 근거의 분류다. 담당 A~F의 작업 분담 및 E/A/S 변경 분류와
별개다. 독립 Rust 구현을 확정했더라도 특정 latent·warm-start·helper·GPU kernel
구조를 모두 채택한 것은 아니다.

| 경로 | 연구 역할 | 해석 경계 |
|---|---|---|
| G | GPU로 고비용 평가를 처리하는 주 연구 | LC0와 동일한 전체 장비·시간 상한에서 검증한다. CPU 제어 코드가 남을 수 있다. |
| H | CPU 평가·전술·시간 관리·캐시로 GPU를 보조 | helper의 CPU·메모리·대기·호출 비용을 모두 포함한다. |
| C | CPU 친화적 후보의 비교 연구 | 성공해도 G의 GPU 구조가 검증됐다고 해석하지 않는다. |
| O | 데이터·학습·튜닝·검증·실험 관리 | 학습 GPU 적합성과 온라인 추론 적합성을 구별한다. |

CPU 적합성, GPU 직접 실행 적합성, GPU 중심 시스템 활용성을 각각 기록한다. 핸드오프의 높음/조건부/낮음은 미측정 추론이다. 낮은 GPU 적합성만으로 후보를 삭제하지 않으며 H/C/O 또는 다른 배치 조건에서 검토할 수 있다.

## 책임과 확정한 코드 경로

아래는 사용자 지시로 지정한 구현 경로와 소유권이다. 문서 정비만 배정된 작업에서
Cargo workspace·빈 crates를 선제 생성하지 않는다. 실제 구현 배정 시 필요한 경로를
만들고 의존 방향을 검증한다. 이전의 `engine/`, `network/`, `runtime/`
역할별 제안 대신 실제 구현에는 이 crate 경로를 사용한다. 학습 도구의 구체 언어·
패키지는 별도 결정이다. 생성 가중치는 저장소 밖 생성물 루트의 `models/`에 두고
접근 제외 규칙을 따른다. 원문 12.1절의 `models/`를 제품 모델 소스 경로로 쓰지 않는다.

| 책임·담당 | 구현 경로 | 소유하는 동작과 금지되는 혼합 |
|---|---|---|
| 최소 공통 계약 / 총괄 | `crates/rz-contracts/` | ID·버전·오류·설정·평가 경계의 단일 소유자. 구체 board·tensor·backend·Search 구현을 넣지 않는다. |
| Rules / State / A | `crates/rz-position/` | 자체 합법 수·상태 전이·이력·종료 판정. 신경망 추정이나 제품용 외부 규칙 라이브러리로 대체하지 않는다. |
| Encoder / C | `crates/rz-encoding/` | 칸·기물·관계 특징, `MoveDelta`, 모델별 입력·policy mapping. 방문 통계를 변경하지 않는다. |
| Evaluator / C | `crates/rz-eval/` | 단일 지원 형식의 policy·WDL·선택적 Q/불확실성, 로딩·실제 backend. 직접 착수·대국 판정·외부 UCI neural wrapper 금지. |
| Native loader / 총괄 I01·C03 접점 | `crates/rz-native-loader/` | 검증된 GPU 의존 라이브러리의 좁은 FFI 로딩·process lifetime pin·실제 mapping 확인. 모델·Rules·탐색·queue를 소유하지 않는다. `rz-eval`의 `forbid(unsafe_code)`를 유지하기 위해 분리한다. |
| Feature/Raw Cache / D | `crates/rz-runtime/` | 입력 식별·버전·정밀도·수명·provenance. 근사 값·Search edge 통계와 분리한다. |
| Refinement 경계 / B·C·D | Search 결정·공통 요청·Runtime 실행 | Search가 계산 요청을 선택하고 C가 supported steps/precision을 정의하며 D가 실행한다. 향후 후보이며 별도 controller crate를 선제 생성하지 않는다. |
| Search / B | `crates/rz-search/` | 최소 자체 PUCT·선택·backup·루트 결정·시간 제어. 모델 입력 형식이나 backend 구현을 소유하지 않는다. |
| UCI / B | `crates/rz-uci/` | protocol·bootstrap·position/search/runtime 연결. stdout 진단 오염·실패 수명 혼합을 막는다. |
| Runtime Scheduler / D | `crates/rz-runtime/` | bounded queue·batch·취소·자원·마감. 무효 결과 게시·double backup·device buffer 조기 재사용 금지. |
| Telemetry / D | `crates/rz-telemetry/`, `benches/runtime/` | 종단 지연·queue·전송·실행·취소·메모리 계측. 관측이 규칙/backup 의미를 바꾸지 않는다. |
| Trainer / Dataset / F | 데이터 계약·학습 도구·`experiments/model-research/` | 교사·자가대국·split·학습·resume. 구체 도구 패키지는 미결정. 최종 holdout으로 튜닝하지 않는다. |
| Experiments / E | `crates/rz-experiments/`, `experiments/baselines/` | version·digest·실행 설정·통계 계약. 대형 binary·가중치·원시 로그는 외부 생성물 루트. |
| Arena / Statistics / E | `crates/rz-arena/` | 외부 runner·시계·PGN·pair 통계·독립 결과 검증. 후보의 자체 승패 판정을 유일 근거로 삼지 않는다. |
| 독립 대조 / 각 담당 | 담당 crate의 검사·작은 fixture | independent perft·full 참조·계약·race 대조. 대형 모델·데이터와 검사 산출물을 소스에 섞지 않는다. |

최소 공통 계약은 **총괄이 직접 정의·revision 관리**한다. TASK-A03은 Position
snapshot·legal order·승격·관점·세대 적용과 연결 검증을 담당하며 계약 정의를
독립 소유하지 않는다. 다른 담당은 공통 ID/오류/요청을 복제하지 않고 총괄의 게시
계약을 소비한다. 공통 계약 공개 후 A·B·C01·D01·E·F01을 mock/fixture로 병렬
개발하며 실제 neural API·정확 규칙·대국 인수는 각각 필요한 근거를 충족한다.

의존 방향은 다음과 같다. contracts는 position/search/eval/runtime에 의존하지
않는다. position은 필요한 공통 primitive만 소비한다. encoding은 position/contracts,
search는 position/contracts와 주입한 evaluator 접점, eval은 contracts/encoding과
선택 backend를 사용한다. runtime은 공통 evaluator 접점으로 provider를 주입받으며
구체 조합은 UCI/bootstrap에서 수행한다. telemetry는 수동 이벤트·계측이고 arena는
외부 실행·검증 경계다. search↔runtime↔eval cycle이나 arena의 규칙 복제를 만들지 않는다.

실제 CPU 조합의 C `contracts` feature에는 A 상태 projection과 D의 `Backend`
접점을 구현한 `NativeRuntimeBackend`가 포함된다. 이 선택적 연결부는 C의 모델·물리 worker를
D의 요청·lease·완료 계약에 맞추며 D는 구체 C provider에 역의존하지 않는다.
B `onnx-cpu` bootstrap이 asset/runtime pin·한 session·공통 clock/ID·diagnostic owner를
조합한다. 모델 추론 코어는 D queue 정책이나 탐색 방문 통계를 소유하지 않는다.

첫 CUDA bootstrap은 C의 `RuntimeLibraryPin`이 전체 네이티브 bundle 사본을 소유하고
`rz-native-loader`가 검증한 절대 경로를 로딩하는 방향이다. 로더는 contracts·eval·runtime에
역의존하지 않는다. CPU 단일 파일 경로와 identity를 유지하고 CUDA에만 전체 bundle
identity를 추가한다. 상세 수명·오류·부분 인수 조건은
[GPU-RUNTIME-BOOTSTRAP](GPU-RUNTIME-BOOTSTRAP.md)을 따른다.

총괄은 루트 Cargo 파일·설정·CI·실험 목록과 consumer 영향을 관리한다. 불변
snapshot·수명 고정 handle로 async borrow를 안전하게 연결하며 변경 후 계약
revision·필요 consumer 검사 범위를 기록한다. 실제 source 선언과 version을
구현 전에 만들어진 것처럼 보고하지 않는다.

## 상태와 평가 계약

상세 필드·폭·제약·검증 경계는 [CONTRACTS](CONTRACTS.md)를 따른다. Runtime과
Search의 계약에 backend tensor·stream type을 노출하지 않는다. 모델·manifest는
로드 경계에서 사전 검증하고 hot path에서는 typed handle·generation·counter를
검증한다. 매 요청 가중치 hash·전체 JSON 직렬화를 강제하지 않는다.

- `PositionState`는 보드, 실제 차례, 캐슬링 권리, 앙파상, 규칙 카운터, 필요한 반복·모델 입력 이력을 포함한다. FEN 이전의 알려지지 않은 이력을 추정하지 않는다.
- 정확한 진행/종료 상태와 claim 가능성·이력 completeness는 별도 축이다. claim
  가능 상태가 여전히 진행 중일 수 있다. 무승부 claim policy는 Rules의 정확 조건과
  구별해 arena 실행 전에 잠근다. 알려진 material shortcut을 완전한 dead-position
  oracle로 보고하지 않는다.
- `MoveDelta`는 일반 이동·포획·캐슬링·앙파상·승격을 구별한다. Rules의
  `RuleMoveDelta`가 기물 제거/추가·전역 상태·undo를 소유하며 Encoder의
  `FeatureDelta`가 encoding handle·특징 dependency·delta version을 결합한다.
  Rules는 모델 입력 구조를 알 필요가 없다. make/unmake와 특수 수는 독립 전체
  재계산과 대조한다.
- `EvalRequest`는 상태 식별자, 모델·인코딩 버전, 합법 수 목록/순서 해시, 정밀도·반복 예산, deadline, 선택적 부모 cache handle을 포함한다.
- `EvalResult`는 같은 요청 ID와 합법 수 순서, policy, W/D/L, 관점, 완료 상태, 정밀도·실제 반복 수, cache provenance를 포함한다. 부분 완료·실패를 정상 완료처럼 반환하지 않는다.
- WDL은 유한하고 각 확률이 유효하며 합이 1이어야 한다. 종료 상태는 Rules가
  처리한다. 최소 공통 계약의 기본 관점은 실제 `side_to_move`다. 외부 모델의 관점은
  C가 명시 변환하며 한 실제 ply 뒤 parent/child backup은 W/L을 교환한다.
- policy는 합법 수에 대응하고 승격 종류를 구별한다. 후보 목록 순서·padding 변경으로 state value가 달라지지 않아야 한다. 허용 수치 오차는 정밀도별로 정의한다.

## 캐시와 탐색 통계 불변식

보드 배치만으로 평가 재사용 자격을 판단하지 않는다. 실제 모델 입력, history-fill, 모델·인코딩·정밀도·계산 예산을 식별해야 한다. 반복 판정 키와 평가 캐시 키는 목적과 필요한 정보가 다를 수 있다.

| 정보 종류 | 계약 |
|---|---|
| 완전 재계산과 같은 특징·평가 | 정의한 수학식과 정밀도별 오차를 만족하는 별도 타입/namespace를 사용한다. |
| 근사 warm-start 잠재 상태 | 경로·부모·모델 provenance를 유지한다. 경로 독립성을 학습했다고 exact로 승격하지 않는다. |
| 원시 신경망 출력 | 검색 중 수정되는 correction과 분리한다. 다른 모델의 평가 척도를 섞지 않는다. |
| 선택적 검색 결과·TT bound | depth·bound·검색 전제를 보존한다. 휴리스틱 검색의 bound를 게임이론적 증명이나 exact 교사 라벨로 바꾸지 않는다. |
| parent edge 통계 | visit·backup·virtual loss·in-flight를 평가 캐시와 별도로 관리한다. |

근사 warm-state를 정확한 transposition 평가로 공유하지 않는다. 정규 raw cache는
명시한 fresh 계산 계약을 만족하는 출력으로 채우는 것을 기본안으로 한다. `fresh`와
`full`은 초기값·계산 예산의 별도 축이다. exact feature/raw를 재사용해도 실제 모델
실행·cache hit·selection·backup 카운터를 따로 기록한다.

cache lookup 자체, sibling 추정, history bonus는 새 완료 방문을 만들지 않는다.
새 유효 traversal이 선택한 경로에서 적합한 cached 평가를 소비하면 알고리즘의
방문 조건에 따라 한 번 backup할 수 있다. 같은 selection의 재조회·재전달·duplicate
callback은 방문을 늘리지 않는다. 정확 eval dedup과 완전 DAG Search는 별도 변경이며
다른 parent edge의 N/W/Q·virtual loss·in-flight를 단순 합치지 않는다.

virtual loss/in-flight는 실행 예약이다. 완료 방문이나 실제 패배가 아니며 취소 시 정확히 해제한다. 요청의 generation, 모델·루트 수명과 deadline을 검사해 늦은 결과·중복 backup·잘못된 handle 재사용을 막는다. 착수 이후 잔여 GPU 작업이 다음 생각 시간에 숨은 이득을 주지 않도록 중단·격리 경계를 검증한다.

논리 취소와 device 작업 완료를 구분한다. terminal outcome·reservation 해제·정상
backup은 각 소유권 경계에서 한 번만 수행하고 physical 완료 전 buffer를 재사용하지
않는다. Request 완료 후 root가 바뀌는 race도 backup commit 직전 재검증한다.
slot 번호에 generation을 결합해 오래된 handle이 새 항목을 읽는 오류를 차단한다.

## 근사와 계산 제어

| 분류 | 변경 | 필요한 구별 |
|---|---|---|
| E | 의미 보존 최적화: 융합·정확 캐시·정확 delta | 수학적 결과 유지와 부동소수점 비트 일치를 구별한다. |
| A | 근사·모델 변경: 양자화·잠재 압축·토큰 병합·warm 조기 종료 | 평균 오차뿐 아니라 치명적 전술 오류·경로 편향을 검증한다. |
| S | 탐색·스케줄 변경: 평가 순서·선택·계산량 | 미사용·취소 작업과 중요한 경로의 대기를 포함한다. |

한 변경이 여러 분류에 해당하면 함께 표시한다. 모델 head·history 보정으로 선택 순서가 바뀌면 실행 속도 최적화로만 설명하지 않는다. CPU alpha-beta의 LMR·null-move·futility와 PUCT는 같은 알고리즘이 아니다. centipawn margin을 WDL에 바로 더하지 않는다.

초기 휴리스틱 활용은 후보 순서나 추가 계산 신호로 비교한다. 영구 후보 제외는 별도의 위험과 회귀 대조를 가진 실험이다. reduced 결과를 full-depth 결과로 캐시하지 않으며 null move를 실제 합법 수나 학습 이동으로 기록하지 않는다.

증분 특징·작은 latent·CARD-C02 recurrent·CARD-C03 warm-start는 보존한 연구
후보다. 초기 단순 Runtime이나 최소 PUCT에 동시에 넣지 않는다. warm 계산을
연구할 때 같은 자체 모델의 fresh/full 경로, 경로 provenance와 오차·tail 전술·
deadline·상주 메모리를 먼저 정의한다. model 구조·history 입력이 바뀌면 기존
가중치 호환과 학습 필요를 별도 과제로 다룬다.

## 구현·연구의 진행 축

PALS 독자 엔진의 실제 연결·책임 경계와 남은 인수는
[PALS 구현 현황과 후속 인수 지시서](research/PALS-IMPLEMENTATION.md)를 참고한다.

20개 TASK의 실제 순서와 병렬 의존성은 [구현 지시서](IMPLEMENTATION-DIRECTIVES.md)를
따른다. 총괄의 최소 공통 계약 → A/B/C01/D01/E/F01 mock 병렬 → 독립 규칙·UCI·
CPU/mock 인수 → 단일 가중치의 실제 Rust 추론 → 종단 계측 → 한 검색/런타임
변경 → 같은 새 Search의 파인튜닝 → 모델 구조 한 변수의 순서로 인수한다.
CPU/mock의 통과로 실제 neural API·GPU·대국 성과를 대신하지 않는다.

주 비교는 `CONTROL-0=원본 W0+자체 기준 S0` →
`CONTROL-1=같은 W0+새 검색 S1` → `CONTROL-2=파인튜닝 W1+같은 S1`이다.
검색 변경·가중치 변경·runtime 변경의 원인을 분리한다. LC0와 equal-time 대국은
이 내부 비교와 구분하는 외부 강도 비교다.

핸드오프 R0~R7 연구 gate는 삭제하지 않고 TASK 인수·실험 분류에 연결한다.
R0 LC0/runner 기준 재현, R1 종단 계측, R2 동일 의미 실행 변경, R3 관계 입력/latent
구조 한 변수, R4 host 잔차·시간 관리·작은 helper 한 변수, R5 반복/warm,
R6 증분 특징·검색·평가·정밀도의 CPU 비교군, R7 검증한 요소의
결합·제거 대조·holdout 확인을 유지한다. LC0 내부 최적화 제안을 자체 제품의 LC0
fork로 해석하지 않는다. 초기 Rust CPU/mock 병렬 작업은 모든 연구 gate의 실측
완료를 기다리지 않으며 실제 후보 인수는 필요한 선행 근거와 유한 자원·실험 계약을
충족한 뒤 수행한다.

## 복구와 검증 경계

재사용을 연구하면 같은 자체 모델의 fresh/full 경로를 비교·복구 기준으로 확보한다. 감사용 fresh 계산도 온라인 대국에서는 시간 비용이다. 마감 안에 복구할 수 없는 내부 오류는 기록하고, 타 엔진을 숨겨 호출해 정상 결과로 바꾸지 않는다. 자체 helper/fallback과 외부 엔진 composite는 서로 다른 실험 계약이다.

독립 규칙 참조·공개 perft·특수 수·긴 make/unmake trace로 규칙을 검증하고, 별도로 계산한 full 경로와 증분 경로를 대조한다. 캐시·평가·scheduler는 모델/이력/정밀도/경로 혼입, 수명 종료, 취소·중복 반영, deadline을 검사한다. 구현과 같은 갱신 함수를 쓰는 대조만으로 독립 검증을 주장하지 않는다.

기존 가중치로 자체 기준 검색을 실행하는 연구, 같은 가중치의 새 검색 연구, 새 검색에
맞춘 파인튜닝, 모델 구조 변경을 분리한다. LC0 기준 실행은 외부 재현·강도 비교이며
제품 코드 포크 결정이 아니다. 가중치 호환이 깨진 입력·구조 변경은 단순 tokenizer
교체가 아니다. 신규 Rust 소스의 MIT 정책과 코드·가중치·교사·opening·변환 도구의
각 권리를 구별하고 필요한 자산의 권리·출처·수치 호환은 사용 전에 확인한다.
