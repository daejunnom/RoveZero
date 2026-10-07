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
| 제품 조합 / [`pals_native.rs`](../../crates/rz-uci/src/pals_native.rs), [`main.rs`](../../crates/rz-uci/src/main.rs) | 정확한 Rules→모델 입력, 역할 요청→runtime, 모델 소비 ACK, UCI 시계·중단·새 게임·종료. 명시적 CPU/CUDA provider와 GPU startup 경계를 구현했다. CUDA-control 경로는 독립 pin의 typed inventory 및 실제 placement/profile witness를 요구하며 GPU 인수 상태는 아래 실행 기록과 구별한다. |
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
| 공개 K/V 재사용과 GPU 상주 | 역할 중립 key와 cache-on/off 수치 대조 접점, bounded host page bank 및 device K/V/I/O binding 경로를 준비했다. 현재 확인한 host page는 전체 입력 단위이며 record별 증분 인코딩이 아니다. 실제 device 상주·prepack 복제·VRAM 공유는 별도 인수 대상이다. |
| CPU/GPU 작업 겹치기와 private warm-start | 현재 native 역할 응답은 drain 후 반환하는 안전 경계다. CPU–GPU overlap 및 warm private latent의 의미·오차·효과는 미인수이며 현재 fresh 계산을 기준으로 남긴다. |
| 학습·resume·모델 교체 | own-source dataset·loss·recipe·zero-step AdamW·sampler/RNG/checkpoint·V-free export를 준비했다. 실제 CPU ORT P/C 수집과 별도 training-private V→CPU_T producer의 유한 실행을 확인했다. optimizer update 및 학습된 모델의 교체 일반화·강도는 미인수다. |

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

[`pals_collect/native.rs`](../../crates/rz-arena/src/pals_collect/native.rs)는 실제 CPU ORT
P/C 호출의 준비 입력을 dispatch 전에 봉인하고, 물리 완료·raw 응답·논리 전달/거절·탐색
소비를 각각 남긴다. 독립 source registry의 binary·checkpoint·export·runtime·모델 구성·
encoding·epoch와 실제 자산을 대조한다. C 이탈 입력은 일반 합법 후보 head의 training row로
자동 변환하지 않고 별도 divergence sidecar와 계보로 보존한다. 가상 반박·수선에 실제
경기의 승패를 붙이지 않으며, backend가 raw 출력을 반환하기 전에 거부한 값은 오류 원인과
비관측 상태로 남긴다. raw가 관측됐다고 만들어 채우지 않는다.

[`verifier_producer.py`](../../experiments/model-research/pals/src/rz_pals_model/verifier_producer.py)는
등록된 공개 parent 입력을 재사용하되 별도 private query를 V forward와 CPU dispatch 전에
봉인한다. 자체 Rust CPU_T의 조건·완료 범위·비용과 후속 자료를 private bank에 보존한다.
제품 V를 활성화하거나 V private 상태를 P/C 입력에 넣지 않으며, 미실행 작업의 비교 순위·
정보 이득·WDL 목표를 만들지 않는다. 유한 producer 실행과 실제 V 학습은 별도 인수다.

## 현재 실행 증거

아래는 2026-10-07 문서 갱신 시점에 총괄이 확인한 실행 기록이다. 로컬 검사, 정확한 SHA의
CI 결과, 독립 등록 binary의 실제 실행과 dirty-source 모델 검사를 구분한다. reference 07의
기준은 `2e07757`+dirty source이며, reference 09는 `2d96a8a`+dirty source의 15개 파일 pin이다.
reference 09를 후속 최종 통합 SHA의 인수로 바꾸지 않는다. 후속 변경은 영향 검사를 다시 한다.
현재 실행 기준 소스는 `5bd8c59`이며, 그 이전 전체 workspace 검사·후속 좁은 재검사와
정확한 CI run, 등록된 실행 바이너리의 실제 결과도 각각 구분해 기록한다.
후속 `e604125`의 UCI pending-close 수정은 소스 변경 기록이며, 아래 실행 결과와 검사
완료 SHA를 그 수정의 통과 증거로 소급하지 않는다.

| 실제 자료 | 확인된 결과 | 해석의 한계 |
|---|---|---|
| 이전 Workspace CPU/mock 검사 | dirty-source에서 1,011개 통과, 16개 ignored. | 이전 실행으로 보존한다. ignored는 미실행이며 후속 변경의 검사 결과로 대체하지 않는다. |
| `975cce4` 로컬 CPU 검사 | workspace 1,061개 통과·16개 ignored, clippy 성공, default native CLI 4개 검사 통과. | 실제 CUDA·학습·대국 및 후속 SHA의 전체 CI 성공을 증명하지 않는다. |
| `975cce4` CI 및 `b04c886` 후속 수정 | 정확한 `975cce4`의 Linux·모델·CPU binding job 성공. Windows는 Unix 전용 fixture 2개 실패. `b04c886`에서 fixture를 수정했으며 해당 CI는 모델·binding 성공, Linux·Windows 실패다. | 해당 SHA의 실패 기록을 보존하며 `b04c886` CI 전체를 성공으로 표시하지 않는다. 후속 수정과 정확한 `5bd8c59`의 성공은 별도 행으로 기록한다. |
| `6979bdf` 전체 검사와 `5bd8c59` 후속 재검사 | `6979bdf` workspace 1,074개 통과·16개 ignored. 후속 `5bd8c59`의 optional CUDA launch payload boxing 2줄 수정에 대해 arena 69개 재검사와 전체 clippy 성공을 총괄이 확인했다. | 전체 workspace 실행 SHA와 좁은 재검사 SHA를 합치지 않는다. ignored는 미실행이며 GPU 종료·실제 학습·기력을 증명하지 않는다. |
| 정확한 `5bd8c59` CI | [CI run 37594871853](https://github.com/daejunnom/RoveZero/actions/runs/37594871853)의 Linux·Windows·모델·CPU bindings 4개 job 모두 성공을 총괄이 확인했다. | 해당 SHA의 CI 성공이다. 실제 CUDA 프로세스 종료, paired 대국과 source 밖 자료의 실행 인수를 대신하지 않는다. |
| `binary-registration-06` | compiler-artifact JSON의 target·profile·features·binary SHA/크기와 `rz-uci-build-capability/1` metadata를 등록했다. source는 `5bd8c59`, UCI binary는 `7b283fea62af3518275b73bd53f6a15aadbb99d789f20f9ec04d4bcda7ab5a03`, CUDA numeric binary는 `a07495dec676c4fefd3d3bc6b367c746ee08395e6b46aced22181524968b4fd5`다. | compile capability만 확인한다. 영수증의 runtime/model loaded는 false이며, `--all-features` 명령 이력이나 source SHA만으로 다른 경로의 나중 바이너리를 같은 feature 빌드로 취급하지 않는다. |
| `model-reference-07` | wrapper 영수증 `success`, CLI exit 0. 모델 13개+학습 준비 20개, 총 33개 검사 통과. | 초기화 자산의 CPU 검사다. 실제 학습·GPU 결과가 아니다. |
| `actual-collection-01` frozen 검사 | 실제 9행 모두 한 번씩 소비. P 9/C 0. policy 4행 활성·5행 mask, WDL 9행 미관측 mask. parameter SHA 전후 동일, optimizer 미생성·backward 없음·steps 0. | 실제 C 조건부 collection, V context, 신경망 collector와 학습을 검증하지 않았다. C head 숫자 fixture와 실제 C 데이터는 별도다. |
| Native CPU PALS probe | 물리 완료 NN 입력 64개, 탐색 소비 32개를 구분한 유한 probe 통과. | 기력·처리량 향상이 아니다. 완료 입력을 모두 유효한 탐색 소비로 세지 않는다. |
| `cpu-numeric-20261007-975cce4-02` | 등록 source `975cce4`, exit 0·6개 참조 사례 통과. private repeat·cache/fresh 동등성·새 게임 cache/물리 ACK·shutdown 확인. 완료 NN 입력 20개 = public 6+role 14, 전체 3.331초, cgroup peak 284,164,096 bytes, OOM 0. 새 게임 뒤 host page reserved/pinned bytes·entries 모두 0. | CPU FP32 수치·수명 증거다. host page는 `whole_input`, allocator peak와 native resident parameter·prepack 공유는 unknown이다. GPU 성능·VRAM·강도 증거가 아니다. |
| `own-onnx-collection-01/own-onnx-01` | 독립 immutable collector binary `2444980f189b1b7d1d99ed9e71339492dded44bf268f1e7d6793df52d5756983`로 실제 CPU ORT NN 수집. 1게임·2 ply·6행, CPU 966 nodes/8 jobs, NN 완료 16개 = public 8+role 8, 역할 탐색 소비 8개. finish의 물리 shutdown·buffer 해제 확인, in-flight 0·quarantine 없음·학습 0. | 2 ply 제한의 결과는 unknown이고 value 6행 모두 mask다. C/Repair 및 divergence 계보를 실제 경기 승패나 전술 증명으로 채택하지 않는다. 독립 binary 실행을 최종 소스 SHA 검사로 합치지 않는다. |
| `model-reference-09` | 등록 collection 6행을 정확히 한 번씩 소비(P 4/C 2). policy 활성 0·WDL 활성 0·WDL mask 6, parameters 전후 동일·optimizer 미생성·backward 없음·steps 0. 별도 private V producer는 자체 CPU 신규 102 nodes, dispatch 1·실제 checks 2·후속 label 1; 부모 입력·weights 불변. | 현재 자료의 masked loss가 0이라는 결과는 학습 개선이 아니다. V의 비교 순위/task loss 목표는 0행이며 private-only, native 제품 V는 비활성이다. 기준 `2d96a8a`+dirty 15개 source pin과 최종 SHA를 구분한다. |
| GPU 시도 02 | 모델 실행 전 `native.missing_mapping`에서 실패. 실패 자료 보존. | PALS 신경망 CUDA 수치 실패로 해석하지 않는다. 입구 mapping 감사의 시점 결함은 후속 수정에서 분리했다. |
| Mapping 시점 수정 | dependency-only 입구 → 완료된 첫 native Run 이후 full audit. backend CPU seam 7개와 all-feature checker 빌드 통과. | 최종 audit 실패도 후속 실행을 막는다. CPU seam은 실제 CUDA 인수가 아니다. |
| GPU 시도 03 | 입구를 통과한 뒤 session 초기화에서 CPU EP 배정과 fallback 금지 충돌. 후속 네이티브 abort, exit `-6`. cgroup peak 3,305,578,496 bytes, OOM 0. | NN Run 이전 실패다. CPU 배정 노드와 종료 오류의 원인은 추가 확인 중이며 VRAM peak는 미관측이다. 기존 pin과 자원 한도를 조용히 바꾸지 않는다. |
| `gpu-numeric-04` | source `b04c886`, CUDA-control 실행 exit 1·77.625초, cgroup peak 3,397,308,416 bytes·OOM 0·cleanup 확인. 첫 NN Run 전 public 220노드(CPU 51/CUDA 169) 배치 gate 통과, shared P/C 453노드(CPU 61/CUDA 392) gate 거부. 등록 inventory와 CPU Gather 계열 4개 및 CUDA `MemcpyFromHost` 1개의 불일치를 보존했다. NN Run 0, 종료 후 GPU 사용 0으로 복귀를 총괄이 확인했다. | 해당 실행의 pre-Run 실패는 그대로 보존한다. 후속의 명시적 CUDA transfer 근거·등록과 05의 수치 보고서를 이 실패 기록에 소급하지 않는다. 임의 whitelist 확대·CPU NN fallback·inventory 재분류로 성공 처리하지 않는다. |
| `gpu-numeric-05` | 등록 source `5bd8c59`와 CUDA numeric binary로 실제 실행했다. `numeric.json`은 6개 CUDA FP32 사례·private repeat·cache/fresh·새 게임·물리 ACK·shutdown 통과를 기록했다. 완료 NN 입력 20개 = public 6+role 14. 그러나 보고서 저장 뒤 `malloc(): unsorted double linked list corrupted`로 프로세스 exit `-6`; execution은 failed, 전체 173.926초·cgroup peak 3,693,518,848 bytes·OOM 0·cleanup 확인·종료 뒤 GPU 사용 0을 총괄이 확인했다. | **GPU 실행 전체 인수는 실패다.** 수치 보고서의 완료/ACK와 프로세스의 정상 종료는 다른 증거다. VRAM peak·native allocator peak는 unknown이며 heap 오류 원인을 확정하지 않는다. CPU 자료나 후속 재실행으로 원래 실패를 덮지 않는다. |
| `gpu-teardown-06` GDB 진단 | 05와 같은 source·binary를 GDB의 기본 ASLR 비활성 조건에서 실행했다. stdout은 inferior의 정상 종료, stderr는 `No stack.`을 기록했다. execution의 GDB exit 0, 전체 141.735초·cgroup peak 3,840,204,800 bytes·OOM 0·cleanup 확인. | 상태는 `diagnostic_only_not_acceptance`다. 총괄은 진단 controller의 exit 1을 의도한 비인수 반환으로 확인했다. GDB·ASLR 조건이 달라 05의 수정이나 정상 실행 인수로 취급하지 않으며 heap 오류가 재현되지 않았다는 사실만 남긴다. ASLR을 켠 후속 07은 이 기록 시점에 실행 중이다. |
| 유한 runner 종료 검사 | clock+reap combined patch를 clean upstream에 적용한 별도 binary 빌드와 production-method syscall seam 14개 통과. | 실제 paired 대국은 별도 인수다. 원래 clock-only binary·patch·과거 자료는 보존한다. |
| `own-paired-cpu-01` 감사 정정 | 원래 입력·실행 자료를 수정하지 않은 append-only 정정으로, 복사한 UCI hash는 정확하지만 실제 binary에 `onnx-cpu` feature가 없었음을 확인했다. identification preflight에서 exit 1, provider·NN·readiness·게임 모두 시작 전이며 게임/NN 수는 0이다. | 대국·기력 실패로 집계하지 않는다. pair/process 영수증 생성 전 실패여서 실제 native cleanup 시간은 미관측이다. 결합 process/cleanup gate 문구만으로 물리 cleanup 실패라고 판정하지 않는다. |
| `own-paired-cpu-02` 준비 | `binary-registration-06` compiler proof와 compile-only capability를 입력 등록 전에 재확인했다. lock·opening·prepare-only 3개 명령 exit 0, private snapshot 9개를 준비했다. 준비 영수증의 engines started/NN ready는 false다. | 새 유한 CPU paired 기능 실행의 준비 자료다. 경기 실행·시계·PGN·정상 종료·강도 결과는 별도 실행 영수증이 도착할 때까지 미인수로 둔다. |
| `own-paired-cpu-02` 실제 preflight | 준비 이후 실제 readiness process는 exit 2·1.017초, `group_cleanup=gone`·errors 0으로 종료했다. stderr의 `UndeliveredDiagnostics`에는 PALS rounds·CPU nodes·소비 role 모두 0이다. supervisor의 service exit는 1, 회수 완료, 전체 16.343초 = service wait 10.808초+recovery 5.520초 등으로 기록했다. 경기 0·경기 PGN 없음. | readiness 실패이며 기력 대국 결과가 아니다. supervisor 시간은 실제 경기 시계·native cleanup 시간이 아니다. 회수된 process 영수증의 cleanup 276,164ns와 group 종료 증거를 별도로 보존한다. 준비된 opening PGN을 경기 PGN으로 세지 않는다. |
| `e604125` UCI 수정 | stop 이후 pending 착수·진단을 닫고 pipelined quit을 처리하는 후속 소스 수정을 총괄이 커밋·push했다. | 이 기록 시점에는 후속 root Cargo 검사 전이다. 이전 `5bd8c59`의 CI 성공·CPU 수치 검사를 수정의 검사 결과로 재사용하지 않으며, 새 바이너리·readiness·paired 실행은 별도로 인수한다. |

작은 실행·결과 JSON을 읽기 전용으로 대조한 자료의 논리 ID와 실제 파일 SHA-256은 다음과
같다. 원시 파일은 관리 루트의 `runs/pals/` 아래에 보존하며 개인 경로·command 원문·호스트
정보는 이 문서에 옮기지 않는다.

| 논리 ID (`runs/pals/` 기준) | SHA-256 |
|---|---|
| `cpu-numeric-20261007-975cce4-02/execution.json` | `561cdf38ad77ddd0236ed13cac4826002a2d0f322ba5d11035cb92da507abba4` |
| `cpu-numeric-20261007-975cce4-02/numeric.json` | `7ed25bd97c8db285684d3dbb6bbee0662f38a8e845f3cdde5c5d6fa4dfdef6dc` |
| `own-onnx-collection-01/own-onnx-01/receipt.json` | `a2a42c3786a7f9639e98c175b483c521af92fc35482e8464e6330c482b762e61` |
| `model-reference-09/validation.json` | `9257b60dd3c69a565be56473725e23bcafa5e9ff715dbfff8171d4ab28779eaf` |
| `model-reference-09/preparation-check.json` | `91984cd2d841ac6433ec93526029d0483cb0ac3c792469ab8014673fc24d4394` |
| `model-reference-09/private-verifier/receipt.json` | `8f9227c5c34b75fa6e915c99c63cce7513c483f195bcfa13cc87d445bbe7ab4f` |
| `gpu-numeric-04/execution.json` | `1453a0ca1acd9a0196f0f79d1fd1e4b6fab2f4d6bd149b5e94da27eb22f14859` |
| `gpu-numeric-04/numeric.cuda-control-primary/public-initial-placement.json` | `0d60ee96e4aa2bfd78eb3c5395e9005bde7820874c316352e79300e4a12232a7` |
| `gpu-numeric-04/numeric.cuda-control-primary/shared_pc-initial-placement.json` | `7192905c844af038dbb5cc80ac1b8f6fc1fe342b98b3986158542c69c4db9fdd` |
| `gpu-numeric-05/execution.json` | `7e2a766401fb4adb917955c0284dd1bbeeae78d9b125f53d9368c5fcbf4ffcd1` |
| `gpu-numeric-05/numeric.json` | `5921a80c2f5982cd8a90a2a8797b90c292c28f3c555110108c00ede884e3b3c2` |
| `gpu-numeric-05/stderr.log` | `2fec9196496bf208222bf35ce8032d8968a407dd34dcce54135658bc493b7463` |
| `gpu-teardown-06/execution.json` | `a5e7d7ea94739902128f49dc9e576c0e129404c91540fe27b8f67ac89e915d2b` |
| `gpu-teardown-06/stdout.log` | `b4d43b3c5775f47920f8edc1091c39d554f884a281eb959c730eee3e3668d140` |
| `gpu-teardown-06/stderr.log` | `49e1116d47bda20a27ba350b585760440aaf899c435196f894edfc207c294ea0` |
| `binary-registration-06/registration.json` | `fad9dc73fddb2ea089d463a612869a4e6ca73d43eac9bf71502fa38335e594c2` |
| `binary-registration-06/build-capabilities.json` | `4cef04dfa2e6560226c2453fc0cdfb69afa49661608f0cb3bbb5a43c571e96ad` |
| `own-paired-cpu-01/audit-correction-01.json` | `99cc4d96bd73ecd154228526c7035843294d51101a1c098b0265ba81c50a19c0` |
| `own-paired-cpu-02/preparation-receipt.json` | `19e331e67bb6cc4b79d815c65c9fc5ffac46f64095c0225ae60a8b262696c7fc` |
| `own-paired-cpu-02/execution-result.json` | `4cd480a94e6300a6bf139f4c1b959318075ddf8f8bf02c1c606d2790a2932d94` |
| `own-paired-cpu-02/recovered/attempt-01/external-baseline-readiness-process.v2.json` | `ecacdae96c69c2f96c12fe564830bb035971bf01e3c8822f7772065ded8d55f4` |
| `own-paired-cpu-02/recovered/attempt-01/external-baseline-readiness.stderr.log` | `23d48e310be136adae0fc8d067c6a2b8b351659c32317e931d158ac97cb9b9ce` |

## 남은 인수와 진행 순서

| 순서 | 필요한 확인 | 현재 상태·책임 |
|---|---|---|
| 1 | mapping 감사·등록된 CUDA transfer 근거·실제 placement witness, 보고서 저장 이후 heap 오류와 완전 종료 | CPU seam과 runner 빌드를 확인했으며 GPU 시도 02/03/04의 원래 실패를 보존한다. 05는 보고서 뒤 heap 오류로 실패했고 06은 GDB·ASLR 비활성 진단에서 미재현했다. ASLR 활성 07은 실행 중이며 결과를 미리 인수하지 않는다. 보고서 저장·worker/session 종료·ORT/CUDA 해제의 실제 경계를 조사하고 원인을 좁힌 뒤 동일 pin의 유한 정상 실행을 별도로 등록한다. |
| 2 | 같은 pin의 GPU 공개 K/V·P/C raw·policy/WDL, cache on/off, `If` inactive 연산, 취소·늦은 완료·drain·buffer 수명 | 05의 CUDA FP32 6개 사례·cache/fresh·새 게임·물리 ACK·shutdown 보고서는 확인했다. 프로세스 종료 실패 때문에 GPU 실행 전체 인수는 남아 있으며, 제품 UCI의 취소·늦은 완료까지 통과한 증거로 확대하지 않는다. C/D와 총괄이 정상 종료와 해당 실행 경로를 별도로 확인한다. |
| 3 | 제품 UCI의 GPU 연결과 실제 affinity·메모리·GPU 자원 관측 | 명시적 CPU/CUDA 선택·startup probe·placement witness 검증·영수증 접점은 구현했다. 실제 GPU UCI 실행·중단·새 게임·drain·종료와 자원 관측은 미인수다. CPU 수치·CLI 검사를 대신 사용하지 않는다. |
| 4 | 실제 NN 기반 own collection 및 C 이탈·V task context, frozen epoch·mask·split·누출 | CPU ORT collector 6행과 frozen preparation 전체 소비, 별도 private V→CPU_T 유한 producer 실행을 확인했다. GPU collector, C divergence head의 명시 학습 context, 유효한 정책·결과·작업 효용 목표와 더 넓은 split/holdout 자료는 별도 인수다. |
| 5 | record별 증분 인코딩·device warm-start·CPU/GPU overlap과 shared reader의 native prepack·VRAM residency·peak | 미인수. 확인한 host bank는 whole-input cache다. serialized sharing과 실제 메모리 효과를 구분하며 고정 작업량·새 세션·관측 해상도를 등록한다. |
| 6 | V-free PALS+Own CPU_R의 유한 paired 실행·시계·PGN·실패·완전 종료 | `own-paired-cpu-01`의 feature 누락 preflight 실패를 정정 영수증과 함께 보존했다. 02는 준비를 마친 뒤 실제 readiness에서 `UndeliveredDiagnostics`·exit 2로 실패했고 경기 0이다. 후속 UCI 수정 `e604125`의 검사·바이너리 재등록과 새 readiness를 먼저 확인한 뒤 실제 대국·시계·PGN·정상 종료를 별도로 인수한다. 두 판으로 Elo를 확정하지 않는다. |
| 7 | 최종 통합 SHA의 영향 feature·consumer·CPU CI와 GPU 증거 연결 | `6979bdf`의 전체 CPU 검사, `5bd8c59`의 좁은 후속 검사·clippy와 정확한 CI 4개 job 성공을 각각 연결했다. 후속 `e604125`는 검사 전이며 별도 영향 검사가 필요하다. 이전 `975cce4`/`b04c886` 실패, 독립 collector binary와 reference 09의 dirty-source pin은 보존한다. 정확한 CI 성공은 GPU 05의 프로세스 실패나 paired 대국 미인수를 해소하지 않는다. 모델·raw 자료는 Git 밖에 보존한다. |

능동 CPU 대체 응수의 후보·반박·수선 연결, evaluator identity를 포함한 근거 namespace,
실제 착수 뒤의 완료 근거·paused 작업 재개, CPU adapter 교체 경계도 별도로 인수한다.
노드가 남아 있다는 사실을 새 root에서 작업을 유효하게 재개했다는 증거로 바꾸지 않는다.
실제 NN collector와 학습 전용 V→CPU_T 유한 dispatch producer는 구현했고 위 제한된 CPU
실행 자료를 확보했다. 기존 CPU/mock collection이나 loss fixture를 그 실행 자료로 바꾸지
않는다. NN 물리 완료·role 소비·CPU 신규 작업·실전 경기 결과·학습 목표의 관측 범위가 다르며,
모든 목표가 mask인 frozen forward 성공을 학습 준비 데이터의 유용성이나 기력으로 표시하지 않는다.

GPU 05의 heap 오류는 보고서의 수치·물리 ACK만으로 원인을 확정할 수 없다. 등록된 source·
binary·자산·runtime·placement 조건과 실패 자료를 보존하고, teardown의 실제 경계와 오류
발생 시점을 좁혀 확인한다. GDB·ASLR 변경 후의 미재현도 원인 제거를 증명하지 않는다.
조건 변경·CPU fallback·무한 재시도로 원래 실행을 성공 처리하지 않는다.

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
