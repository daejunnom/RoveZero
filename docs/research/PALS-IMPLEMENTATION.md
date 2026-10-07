# PALS 구현 현황과 후속 인수 지시서

이 문서는 구현된 접점과 현재 실행 증거를 정리한다. PALS의 단계·V3 명세·비교 질문은
[기존 PALS 구현 계획](../PALS-IMPLEMENTATION.md)이 소유하며 여기서 새 계획으로 복제하지
않는다. 사용자가 제공한 `RoveZero_PALS_Internal_Algorithms_KO.md`와
`RoveZero_PALS_Architecture_Flows_KO.md`는 설계 자료다. 원문의 학습·실행 제안을 실제
승인으로 확대하지 않는다. 현재 목표는 **독자 Rust 엔진과 모델·학습 준비의 구현이며,
실제 optimizer 학습은 제외**한다. 생성 가중치는 미학습 초기화 자산이고 기력 인수는 없다.

## 처리 흐름과 소유 경계

제품 경로는 기존 UCI에서 탐색을 선택한 뒤 Rules의 정확한 상태·이력·합법 수를 사용한다.
`--search=puct`, `--search=cpu`, `--search=pals`는 별도 선택이며 PALS 작업을 기존 PUCT의
simulation·visit으로 해석하지 않는다. PALS는 P의 수순 제안 → 자체 CPU의 조건부 검사 →
C의 이탈 지점·반박 → P의 수선 → 관련 상황의 결론 갱신으로 진행한다. 마감 전 마지막
유효 착수를 선택하며, 모델의 추정은 Rules의 체크메이트·스테일메이트 판정을 대신하지 않는다.

| 소유자·소스 | 책임과 교체 경계 |
|---|---|
| Rules / [`rz-position`](../../crates/rz-position/src/lib.rs) | 합법 수, 특수 수, make/unmake, 완전 상태·이력·종료의 단일 소유자. |
| Own CPU / [`cpu.rs`](../../crates/rz-search/src/cpu.rs), [`cpu_value.rs`](../../crates/rz-search/src/cpu_value.rs) | CPU_T/CPU_R가 공유하는 유한 iterative deepening·PVS·aspiration·quiescence·TT와 평가 접점. profile·조건·depth·node·시간·완료 범위를 따로 식별한다. 초기 평가를 학습된 평가로 보고하지 않는다. |
| PALS Search / [`engine.rs`](../../crates/rz-search/src/pals/engine.rs) | 제안·반박·수선, 제한된 continuation 확장, CPU 작업 요청·소비와 루트 결정. `RoleModel`의 합법 후보 순서 logits·WDL 접점으로 모델을 시작 시 선택한다. |
| Canonical stores / [`store.rs`](../../crates/rz-search/src/pals/store.rs) | 상태·공유 line chunk·관측·상황·의존 관계·작업과 소비자별 완료를 보존한다. representation cache 회수와 검사 결과 삭제를 혼동하지 않는다. |
| PALS 계약 / [`pals.rs`](../../crates/rz-contracts/src/pals.rs) | `pals/0.1` 역할·authority·generation·situation handle·representation key·typed payload·CPU 조건. 총괄이 공통 계약을 소유한다. 기존 평가 계약 `0.1`의 가짜 `EvalOutput`으로 변환하지 않는다. |
| Runtime / [`rz-runtime/pals.rs`](../../crates/rz-runtime/src/pals.rs) | 큐·batch key·deadline·취소·물리 lease·typed 응답과 role-neutral/private memory namespace. 정상 물리 완료 전 입력·출력·session·workspace를 해제하거나 재사용하지 않는다. |
| 모델·ORT / [`pals_model.rs`](../../crates/rz-eval/src/pals_model.rs), [`pals_onnx.rs`](../../crates/rz-eval/src/pals_onnx.rs) | 입력·shape·dtype·유한값·예산·output 의미, 자산 pin, 공개 K/V 및 역할 graph, 하나의 물리 worker. cache hit·물리 NN 입력·탐색 소비를 별도로 기록한다. |
| 제품 조합 / [`pals_native.rs`](../../crates/rz-uci/src/pals_native.rs), [`main.rs`](../../crates/rz-uci/src/main.rs) | 정확한 Rules→모델 입력, 역할 요청→runtime, 모델 소비 ACK, UCI 시계·중단·새 게임·종료. 제품의 native PALS UCI 경로는 현재 명시적 CPU provider만 허용한다. |
| 실행·데이터 / [`pals_manifest.rs`](../../crates/rz-experiments/src/pals_manifest.rs), [`pals_data.rs`](../../crates/rz-experiments/src/pals_data.rs), [`pals_collect.rs`](../../crates/rz-arena/src/pals_collect.rs) | V3 명세·lock·receipt, own-source 데이터·시점·입력 seal·분할·누출 검사, 수집 및 실패 보존. 외부 상대 엔진과 내부 모델 선택은 독립이다. |

CPU profile과 모델·입력의 의미가 달라지면 기존 작업 재개·cache 자격을 다시 검사한다.
CPU가 중단됐을 때 남은 completed depth와 frontier estimate를 구분하고, 미완료 결과를
요청 depth의 완료 bound로 저장하지 않는다. CPU와 WDL의 서로 다른 척도를 임의로 더하지
않는다. 외부 UCI 상대를 선택했다고 외부 엔진이 PALS 내부 CPU_R 구현으로 연결되는 것은
아니다. 실제 교체에는 capability·문제 조건·raw score·완료 범위의 별도 adapter가 필요하다.

## 원문 설계와 이번 구현의 범위

| 설계 항목 | 현재 구현·제한 |
|---|---|
| 학습 P/C/V+CPU_T, 실전 P/C+CPU_R | 모델의 P/C/V와 역할별 private parameter를 준비했다. 제품은 V-free P/C이며 실제 P/C/V 학습은 미실행이다. |
| 역할별 private latent와 공유 reader | 폭 384, Q/KV head 6/2, head dimension 64, private latent 16×384, reader 2 block×2회, SwiGLU 1024. board 64+metadata 2 토큰 encoder 2 block, record 내부 독립 encoder 1 block이다. |
| Canonical 사실과 GPU 표현 분리 | StateStore·LinePool·ObservationStore·SituationArena·TaskTable 및 의존 관계를 구현했다. GPU 표현을 회수해도 CPU 문제의 완료 기록은 보존한다. private latent를 공통 공개 기록으로 넣지 않는다. |
| 제한된 작업·저장소·중요 기록 | 모델 입력은 record 128·후보 256·이탈 지점 128 상한이며 required critical record 누락을 거부한다. 상황·line·관측·작업·큐·실행의 별도 유한 한도를 지킨다. |
| 실제 shared P/C reader 소유 | `public_memory`+`shared_pc_if` 두 session export와 native 소비 경로를 구현했다. 공유 reader·후보 임베딩 initializer는 outer scope 한 벌이며 private 초기 latent·4 FFN·head는 6개 ONNX `If`로 hard route한다. |
| 공개 K/V 재사용과 GPU 상주 | 역할 중립 key와 cache-on/off 수치 대조 접점, device K/V/I/O binding 경로를 준비했다. 실제 GPU 입력·연산·수명, prepack 복제·VRAM 공유는 별도 인수 대상이다. |
| CPU/GPU 작업 겹치기와 private warm-start | 현재 native 역할 응답은 drain 후 반환하는 안전 경계다. CPU–GPU overlap 및 warm private latent의 의미·오차·효과는 미인수이며 현재 fresh 계산을 기준으로 남긴다. |
| 학습·resume·모델 교체 | own-source dataset·loss·recipe·zero-step AdamW·sampler/RNG/checkpoint·V-free export를 준비했다. 학습 데이터 수집 경로의 실제 신경망 사용과 optimizer update, 학습된 모델의 교체 일반화·강도는 완료하지 않았다. |

현재 가중치는 seed로 만든 **무작위 초기 파라미터**다. 실제 neural forward와 결정적
`legal-order-mock`은 별도 모델 종류로 식별하며 실패 시 서로 자동 대체하지 않는다.
신경망 실행이 된다는 사실만으로 유효한 체스 지식이나 개선된 착수 품질을 주장하지 않는다.

## 모델·입력·export 식별

구체 모델 소스와 명령은 [PALS 모델 패키지](../../experiments/model-research/pals/README.md)에
둔다. Torch 2.8.0·NumPy 2.2.6·ONNX 1.19.0·ORT 1.22.0·opset 17과 Rust
`ort 2.0.0-rc.10`의 실제 소비를 별도로 검사한다. Python은 offline 준비·export·독립 참조
용도이며 제품 탐색 노드마다 Python callback을 실행하지 않는다.

신형 artifact는 `rovezero.pals-model.v2`, `layout=shared_pc_if`, `layout_revision=1`이고
모델 의미는 v1이다. 기존 v1 separate P/C·P/C/V graph는 그대로 읽고 자동 변환하지 않는다.
제품용 `--rules-profile-json`은 실제 Rust encoder가 출력한 `rz-pals-rules-fields-v1`의
필드 순서·정규화·결측 의미를 pin한다. 58개 의미 문자열의 length-prefixed SHA,
source SHA와 선언 raw·canonical SHA를 기록한다. 의미 pin은 학습 입력 적합성의 증명이
아니므로 이를 별도 선언·데이터 검사 없이 학습 호환으로 승격하지 않는다.

P/C batch는 하나의 scalar 역할만 가지며 `If` condition은 CPU BOOL이다. 입력은 정확한
board·이력·record revision·모델 epoch·후보 순서·mask를 식별한다. 승격은 공통 Move16의
`0=none, 1=queen, 2=rook, 3=bishop, 4=knight`다. output은 요청 후보 순서와 차례 관점
WDL을 유지한다. P의 0 divergence padding은 의미 출력에서 제거한다. V 전용 head·latent가
제품 graph에 없음을 검사한다. raw logits·공개 K/V는 절대 `1e-4`+상대 `1e-3`, policy/WDL은
최대 절대 차이 `1e-4`로 독립 참조와 대조한다.

`reader_initializer_bank`의 byte·SHA와 branch-local copy 부재는 ONNX 직렬화 단일 소유의
증거다. native optimizer의 prepack·workspace 복제나 실제 VRAM 절약을 증명하지 않는다.
CPU `If` selected-only profile, 실제 native 수치·수명, GPU 실행과 메모리 관측을 구분한다.

## 자체 데이터와 zero-step 학습 준비

[`training.py`](../../experiments/model-research/pals/src/rz_pals_model/training.py)는
`rz-pals-data/2` 시점 고정 입력 seal과 native tensor sidecar의 byte SHA·encoder·history·
epoch·record 순서·critical·합법 후보를 대조한다. 독립 등록한 collection receipt와
source registry·split을 사용하며 target을 입력에 섞지 않는다. CPU raw 평가를 임의 WDL로
바꾸지 않고, policy·WDL·이탈·V 작업 각각의 명시 mask와 context를 요구한다.

PC bootstrap은 공유 encoder·reader·후보 임베딩과 선택한 P 또는 C private expert를
준비한다. V 단계는 V private expert만 선택하고 나머지 공유 및 P/C parameter를 실제
`requires_grad` 상태로 동결한다. zero-step AdamW parameter group·recipe·유한 비용,
role별 sampler permutation/cursor, Python·NumPy·Torch RNG, 모델과 optimizer 설정을
checkpoint에 보존한다. pending gradient가 없고 steps=0인 경계에서만 저장하며,
파일 저장·hash까지 닫은 final usage receipt를 재개 시 함께 요구한다.

[`preparation_check.py`](../../experiments/model-research/pals/src/rz_pals_model/preparation_check.py)는
등록된 실제 collection을 모두 한 번씩 소비하고 CPU frozen forward·masked loss·유한값·
parameter SHA 전후 일치를 확인한다. 이 CLI는 optimizer를 생성하지 않으며 backward·
optimizer step·GPU를 실행하지 않는다. 원시 collection·모델·receipt·보고서는 소스 밖
관리 루트에 두고, 공개 문서에서는 논리 경로와 식별만 사용한다.

## 현재 실행 증거

아래는 문서 작성 시점에 총괄이 확인한 실행 기록이다. 동일한 통합 commit의 CI 인수로
바꾸지 않는다. CPU reference 07의 기준은 `2e07757`+dirty source이며 개별 입력 소스
hash가 영수증에 등록돼 있다. 후속 소스 변경은 영향 검사를 다시 해야 한다.

| 실제 자료 | 확인된 결과 | 해석의 한계 |
|---|---|---|
| Workspace CPU/mock 검사 | dirty-source에서 1,011개 통과, 16개 ignored. | ignored는 미실행이다. CUDA·학습·대국 성공과 정확한 최종 commit CI를 증명하지 않는다. |
| `model-reference-07` | wrapper 영수증 `success`, CLI exit 0. 모델 13개+학습 준비 20개, 총 33개 검사 통과. | 초기화 자산의 CPU 검사다. 실제 학습·GPU 결과가 아니다. |
| `actual-collection-01` frozen 검사 | 실제 9행 모두 한 번씩 소비. P 9/C 0. policy 4행 활성·5행 mask, WDL 9행 미관측 mask. parameter SHA 전후 동일, optimizer 미생성·backward 없음·steps 0. | 실제 C 조건부 collection, V context, 신경망 collector와 학습을 검증하지 않았다. C head 숫자 fixture와 실제 C 데이터는 별도다. |
| Native CPU PALS probe | 물리 완료 NN 입력 64개, 탐색 소비 32개를 구분한 유한 probe 통과. | 기력·처리량 향상이 아니다. 완료 입력을 모두 유효한 탐색 소비로 세지 않는다. |
| GPU 시도 02 | 모델 실행 전 `native.missing_mapping`에서 실패. 실패 자료 보존. | PALS 신경망 CUDA 수치 실패로 해석하지 않는다. 입구 mapping 감사의 시점 결함은 후속 수정에서 분리했다. |
| Mapping 시점 수정 | dependency-only 입구 → 완료된 첫 native Run 이후 full audit. backend CPU seam 7개와 all-feature checker 빌드 통과. | 최종 audit 실패도 후속 실행을 막는다. CPU seam은 실제 CUDA 인수가 아니다. |
| GPU 시도 03 | 입구를 통과한 뒤 session 초기화에서 CPU EP 배정과 fallback 금지 충돌. 후속 네이티브 abort, exit `-6`. cgroup peak 3,305,578,496 bytes, OOM 0. | NN Run 이전 실패다. CPU 배정 노드와 종료 오류의 원인은 추가 확인 중이며 VRAM peak는 미관측이다. 기존 pin과 자원 한도를 조용히 바꾸지 않는다. |
| 유한 runner 종료 검사 | clock+reap combined patch를 clean upstream에 적용한 별도 binary 빌드와 production-method syscall seam 14개 통과. | 실제 paired 대국은 별도 인수다. 원래 clock-only binary·patch·과거 자료는 보존한다. |

## 남은 인수와 진행 순서

| 순서 | 필요한 확인 | 현재 상태·책임 |
|---|---|---|
| 1 | mapping 감사 시점·유한 runner wait/종료 patch의 CPU 검사, 변경 후 실제 GPU provider 초기화 | CPU seam과 runner 빌드 확인. GPU 시도 03은 session 초기화에서 실패했으며 CPU 제어 노드와 주요 NN 계산의 실제 배치를 구분한 인수가 남는다. |
| 2 | 같은 pin의 GPU 공개 K/V·P/C raw·policy/WDL, cache on/off, `If` inactive 연산, 취소·늦은 완료·drain·buffer 수명 | 미인수. C/D와 총괄이 지정 장비의 실제 자료로 확인한다. CUDA feature 빌드나 CPU 성공으로 대신하지 않는다. |
| 3 | 제품 UCI의 GPU 연결과 실제 affinity·메모리·GPU 자원 관측 | 현재 native PALS UCI는 CPU-only. 별도 GPU bootstrap·영수증·지원 조건이 필요하다. |
| 4 | 실제 NN 기반 own collection 및 C 이탈·V task context, frozen epoch·mask·split·누출 | own CPU collection의 P 9행 frozen 검사는 확인했다. 나머지는 F/E가 준비하고 총괄이 별도로 인수한다. |
| 5 | CPU/GPU overlap과 shared reader의 native prepack·VRAM residency·peak | 미인수. serialized sharing과 실제 메모리 효과를 구분하며 고정 작업량·새 세션·관측 해상도를 등록한다. |
| 6 | V-free PALS+Own CPU_R의 유한 paired 실행·시계·PGN·실패·완전 종료 | 실제 대국 인수는 남아 있다. E와 총괄이 명세·asset·resource·timeout을 잠근 뒤 수행한다. 두 판으로 Elo를 확정하지 않는다. |
| 7 | 최종 통합 SHA의 영향 feature·consumer·CPU CI와 GPU 증거 연결 | dirty-source 결과와 최종 CI를 구분한다. 모델·raw 자료는 Git 밖에 보존한다. |

능동 CPU 대체 응수의 후보·반박·수선 연결, evaluator identity를 포함한 근거 namespace,
실제 착수 뒤의 완료 근거·paused 작업 재개, CPU adapter 교체 경계도 별도로 인수한다.
노드가 남아 있다는 사실을 새 root에서 작업을 유효하게 재개했다는 증거로 바꾸지 않는다.
현재 실제 NN collector와 학습 전용 V→CPU_T 유한 dispatch producer는 후속 구현 중이며,
기존 CPU/mock collection이나 loss fixture를 해당 producer의 실행 자료로 사용하지 않는다.

실제 optimizer 학습은 이 목표의 남은 필수 실행에 포함하지 않는다. 이후 학습을 진행할
경우 own-source 데이터·권리·예산·role 순서·frozen epoch·holdout과 checkpoint 조건을
다시 확정한다. 공개 모델 자산이 아직 미학습이라는 사실은 문서와 실행 영수증에 유지한다.

PALS는 기존 PUCT·LC0와 모델·탐색을 바꾸는 구현이다. **자료형 변경 등 의미 보존 변경에
사용한 시간·메모리 5% 문턱은 이번 PALS 도입의 진행·통합 조건이 아니다.** 후속 의미 보존
E 최적화가 필요하면 해당 변경의 baseline·peak·작업량·채택 기준을 별도로 등록한다.
현재 비교는 선언한 모델·탐색·runtime 전체 또는 통제한 단일 축의 질문에 맞춰 해석한다.

## 교체와 복구 지시

모델·Own CPU profile·외부 상대를 각각 독립적으로 선택한다. 새 모델은 typed 역할 입력·
후보 순서·WDL·head·cache 의미와 실제 자산 pin을 먼저 검사한다. 새 CPU는 지원 질문·
조건·완료 범위·score 관점과 재개 capability를 맞춘다. 외부 상대의 UCI·옵션·시계·process
종료는 arena가 소유하고 PALS 내부 입력이나 역할 head를 요구하지 않는다.

미지원 형식·계약·provider·pin 불일치, 할당·물리 완료 실패가 발생하면 명확한 오류로
중단하고 실패 자료를 남긴다. session·buffer 완료가 불명확하면 격리하고 새 요청에
재사용하지 않는다. cache 재계산은 canonical 기록과 동일 입력에서 수행하며, cache
회수가 CPU 검사 재실행이나 완료된 사실 손실을 요구하지 않도록 검증한다.

복구는 등록된 이전 바이너리·자산·profile 및 기존 V1/V2·PUCT 또는 Own CPU의 명시적
실행 선택으로 한다. PALS 오류 뒤 숨은 mock·LC0·Stockfish fallback으로 정상 성공을
만들지 않는다. 계약·정확성·수명·대국 증거와 다른 작업의 WIP를 보존하며 총괄이 인수한다.
