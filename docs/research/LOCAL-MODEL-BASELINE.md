# 로컬 LC0 실행과 강도용 모델 전환 계획

기준일: 2026-10-04. 총괄 TASK-I01/I02의 조사·계획이다. 사용자는 현재 Maia의
강도 한계를 검토하고 LC0 로컬 실행 가능성을 평가한 뒤 모델 교체 또는 추가 학습
계획을 요청했다. 실제 Rust 모델 교체·학습·새 정식 대국은 이 조사와 별도 인수한다.
기준 소스는 `develop`의 `87017a27a467dd37ca711a39020f3ddf5d6a9934`다.

## 1. 현재 기준선의 한계와 확인된 사실

- 현재 Rust evaluator는 Maia1 v1.0 `maia-1900`의 6개 SE residual block·64 channel
  CNN이다. 인간의 수를 예측하는 호환·런타임 기준선으로 선정했다. 이름의 1900은
  현재 RoveZero의 Elo나 엔진 강도 보장이 아니다.
- CNN이라는 구조만으로 결함을 판정하지 않는다. 학습 목표·가중치 품질, 입력/
  policy/WDL 의미, 탐색 예산과 runtime 수명은 각각 대조한다. Maia 공식 실행 예는
  인간 수 재현을 위해 `go nodes 1`을 사용한다. 강한 대국 엔진과 목표가 다르다.
- 기존 48판은 같은 Maia·같은 PUCT의 pending/batch 폭 1/4 내부 비교였고 득점률은
  50%였다. 강한 모델이나 LC0와의 대국 근거가 아니다. WorkerLimit/legal fallback
  3건과 bestmove 뒤 physical GPU 완료·소비 journal의 공백을 보존한다.
- 현재 B의 기본값과 설정 검증에는 **128 simulation 상한**이 있다. 모델을 교체해도
  이 상한과 시간 제어·수명 문제가 자동으로 해결되지는 않는다. LC0의 UCI nodes와
  RoveZero simulation을 같은 비용·같은 방문으로 취급하지 않는다.

근거: [Maia 공식 설명](https://github.com/CSSLab/maia-chess),
[첫 선정](../WEIGHT-SELECTION.md), [기존 내부 A/B](LOCAL-STRENGTH-AB.md),
`crates/rz-eval/src/asset.rs`, `crates/rz-uci/src/engine.rs`.

## 2. 실제 로컬 LC0 실행

공식 stable LC0 v0.32.1의 Windows CUDA 12 package를 사용했다. 유료 GPU·RunPod를
사용하지 않았다. package의 공개 SHA-256과 다운로드 bytes를 대조했다.

| 항목 | 잠근 조건 / 관측 |
|---|---|
| 장비 | RTX 4050 Laptop 6 GB, driver 610.62, Windows; RAM 약 15.6 GiB |
| 엔진 | LC0 v0.32.1, classic search, 공식 Windows CUDA 12 package |
| package SHA-256 | `8d0ce17676eb15e303bea9e790742d31c94ce5d24107f6187adc58e820f6d2f7` |
| executable SHA-256 | `e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b` |
| backend | 명시적 `cuda-fp16`; UCI option과 CUDA 12.9 / 해당 GPU runtime 진단 확인 |
| 자원·옵션 | threads 1, minibatch 16, NNCache 10,000, cache history 7, Ponder off |
| 시간 | MoveOverheadMs 10, SmartPruningFactor 0; startup와 착수 시계 구분 |
| 입력 | 시작 상태, 흑 차례 오프닝, Sicilian 이력, pawn endgame, 흑/백 mate-in-one의 6개 상태 |
| 요청 | 모델별 각 상태의 `go nodes 128`, `go movetime 100`, `go movetime 1000`: 18회 |
| 모델 | Maia 원본과 T1-256x10-distilled-swa-2432500 원본을 별도 process로 순차 실행 |
| 완료 | 두 모델 모두 18회 bestmove와 exit 0, process 종료 확인 |
| GPU 메모리 | 두 모델을 순차 실행한 구간의 전체 장치 관측 최고 1,169 MiB; process peak가 아님 |
| 상한 | process별 120초, 응답별 15초, log stream별 8 MiB, model 50 MiB 다운로드 상한 |

현재 상태·bestmove/PV·WDL 독립 감사는 후속 기록에서 확정한다. 이 smoke는 실제
GPU 실행 가능성 근거이며 강도·FP32/FP16 수치 parity·동시 두 엔진 VRAM·정식 GPU
시간 공정성의 인수가 아니다. raw receipt/log는 저장소 밖
`reports/coordinator-integration/lc0-model-evaluation-20261004/`에 보존한다.

첫 Windows `--help` 호출은 20초 timeout, 재시도는 exit 0이었다. 원인은 미확정이다.
또 최초 time smoke의 LC0 기본 MoveOverheadMs는 **200 ms**여서 100 ms 요청은 nodes 1로
종료됐다. 원본 기록을 보존하고 `probe-v2`에서 위 10 ms·pruning 0·Windows QPC
단조 시계로 다시 실행했다. 기본 옵션으로 상대의 탐색 예산을 약화하지 않는다.

재현 argv는 별도 빈 config와 명시 weights 경로에 아래 옵션을 합친다.

```text
lc0 --config=<empty-config> --weights=<exact-weight-file> --backend=cuda-fp16
    --threads=1 --minibatch-size=16 --nncache=10000 --cache-history-length=7
    --show-wdl --preload --move-overhead=10 --smart-pruning-factor=0
```

[공식 release](https://github.com/LeelaChessZero/lc0/releases/tag/v0.32.1),
[공식 모델 목록](https://lczero.org/play/networks/bestnets/).

## 3. 첫 강도 후보: 동결 T1 distilled

**추천 순서는 동결된 강한 사전학습 모델의 연결·평가를 먼저 하고, 그 뒤 학습이다.**
첫 구현 후보는 공식 목록의 소형 T1-256x10-distilled-swa-2432500이다. 외부 LC0의
실행 가능성이 확인됐고, 내부 Rust 모델 승격은 아직 하지 않았다.

| 원본 필드 | 직접 검사한 값 |
|---|---|
| 공식 URL | `https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz` |
| gzip bytes / SHA-256 | 37,118,673 / `bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406` |
| protobuf bytes / SHA-256 | 40,401,217 / `6ad1b1ceae674911a7fe5a0b3369d0bc5b1f68e655fbc8b302ebb4e9feb34bc0` |
| 최소 LC0 / 저장 형식 | 0.29.0 / LINEAR16; 실행 FP16과 저장 양자화는 별개 |
| 입력 | INPUT_CLASSICAL_112_PLANE = 1 |
| body 메타데이터 | NETWORK_SE_WITH_HEADFORMAT = 4; residual 0, attention encoder 10, headcount 8, embedding 256 |
| policy / value | POLICY_ATTENTION = 3 / VALUE_WDL = 2, OUTPUT_WDL = 2 |
| moves left / activation | MOVES_LEFT_V1 = 1 / DEFAULT_ACTIVATION_RELU = 1 |
| 내장 license | 없음; 아래 직접 제작자 답변을 별도 보존 |

body enum 이름만 보고 SE CNN이라고 해석하지 않는다. 실제 encoder와 tensor 구조를
함께 검사한다. 입력 enum이 Maia와 같아도 history fill·관점·승격·캐슬링 mapping을
자동 호환으로 인정하지 않는다. LC0 명세의 고정 참조 SHA는 기존 선정 문서와 같다.

2026-08-05 제작자 `masterkni6`는 해당 파일을 제작했다고 확인하고 누구나 원하는
방식으로 사용할 수 있다고 답했다. 이번 로컬 사용·후속 호환 연구는 그 직접 허가를
근거로 삼는다. 이를 임의의 MIT/CC0/SPDX license로 바꾸지 않으며 출처·원본 digest와
답변을 보존한다. 실제 재배포 산출물·modified checkpoint의 고지 조건은 별도 기록한다.
[제작자 답변](https://github.com/orgs/LeelaChessZero/discussions/2430)

공식 표의 약 1.6 GB는 안내값이다. 이번 장치 전체 표본의 1,169 MiB와 정의가 다르다.
BT4-it332는 강한 외부 비교/교사 후보로 보존하지만 공식 약 4 GB 안내만으로 이 장치의
동시 점유·속도·권리·학습 적합성을 인수하지 않는다. 이번에는 다운로드·실행하지 않았다.

## 4. 구현·인수 순서

| 순서 / 담당 | 작업과 완료 조건 |
|---|---|
| 1 / 총괄·B·D | WorkerLimit 3건의 정확한 요청/세대/마감 원인을 재현한다. bestmove 전에 취소된 작업의 physical drain을 확인하고, game/root/request별 fresh 실행·cache 소비·유효 backup·deadline fallback을 기록한다. 실패를 숨기거나 로그만 줄이지 않는다. |
| 2 / B·총괄 | 128 simulation을 안전한 설정 상한과 T1/T2 시계로 분리한다. queue·tree·RAM·시간 상한과 stop/quit/drain을 유지한다. 기존 알고리즘 고정 상태에서 예산 변경을 따로 대조한다. |
| 3 / C·총괄 | Maia의 exact profile을 보존하며 T1용 immutable model/export descriptor와 source digest·권리 근거·새 모델 ID를 추가한다. raw/exact cache namespace를 분리하고 newgame/model 교체에서 잘못된 재사용을 차단한다. |
| 4 / C | T1 외부 ONNX export의 shape/head 의미·할당 상한을 검사한다. classical 입력은 가능한 기존 구현을 재사용하되 독립 reference로 대조한다. attention policy·WDL·미사용 moves-left head를 명시하고 미지원 모델은 거부한다. |
| 5 / C·D·총괄 | 같은 T1 원본의 CPU reference ↔ ONNX CPU/FP32 ↔ Rust CPU ↔ 로컬 CUDA/FP32 수치 대조. batch 1/2/4/8/16, 양쪽 차례·history·반복·castling/EP/네 승격을 포함한다. FP16은 통과한 FP32 기준 이후 별도 변경으로 검사한다. |
| 6 / E·총괄 | 다음 표의 대조군과 유한 manifest를 잠근 뒤 개발 대국을 실행한다. 같은 full start/history의 흑백 pair, 전체 PGN 감사, 실제 소비·실패·자원·물리 완료를 기록한다. 결과를 본 뒤 유리한 제외/중단을 선택하지 않는다. |
| 7 / F·C·총괄 | 동결 강도용 모델을 실제 Rust 엔진이 사용한 기준선과 병목·실패군을 확보한 뒤, 아래 F02 학습 경로를 하나만 선택한다. |

공통 의미 계약 0.1은 이번 조사에서 변경하지 않는다. 모델의 구체 구조를 공통 search
타입에 고정하지 않는다. 현재 `MaiaAsset`는 원본 두 digest, converter, GPL profile,
FP32 출력 이름과 **16 MiB ONNX 상한**을 고정해 검사한다. 이를 삭제하거나 기존
Maia ID 아래 T1을 넣는 구현은 거부한다. C/B/D 접점은 총괄이 최신 source와 수동 대조한다.
GPU 없는 C/F 에이전트는 parser·export·CPU fixture·training adapter를 먼저 제공할 수 있다.

### 모델·엔진·탐색·학습 효과의 분리

| 비교 | 고정할 것 / 해석 |
|---|---|
| LC0(Maia) ↔ LC0(T1) | 같은 LC0·시간·backend/정밀도·자원/옵션. 가중치와 구조 교체의 외부 엔진 안 효과 |
| RoveZero(Maia) ↔ LC0(Maia) | 같은 원본 모델/입력·전체 시계/자원. 엔진 전체 차이이며 PUCT 한 요소의 효과라고 하지 않음 |
| RoveZero(Maia) ↔ RoveZero(T1) | 같은 source·PUCT·runtime·시간/자원. 가중치/구조 교체의 내부 효과 |
| RoveZero(T1, S0) ↔ RoveZero(T1, S1) | 같은 동결 가중치에서 탐색 변경의 CONTROL-0→1 |
| RoveZero(T1, S1) ↔ RoveZero(T1-finetuned, S1) | 같은 탐색에서 학습 변경의 CONTROL-1→2 |

FP16 LC0와 기존 FP32 RoveZero, Windows와 WSL의 결과를 단일 알고리즘 차이로 부르지
않는다. 최종 같은 자원 대국에는 backend/정밀도/OS·CPU·GPU·메모리·초기화·cache·
ponder·시간 여유까지 고정한다. 외부 LC0 프로세스는 비교/오프라인 교사이며 자체 평가
API의 자동 fallback이나 Rust engine 구현의 대체가 아니다.

첫 후속 개발 대국의 제안은 24쌍·48판, worker 1, 수당 1초, 전체 30분/400 ply 상한이다.
표본·오프닝·실패/미완료·시계와 실제 자원은 실행 전에 하나의 manifest로 확정한다.
제안값은 현재 실행 권한·정식 승격 기준이 아니다. 이미 본 24오프닝은 개발 pool로
취급하고 승격용 holdout은 별도로 잠근다.

## 5. 추가 학습의 두 경로와 자원 계획

**우선 경로:** T1 원본 구조를 유지한 미세조정이다. 원본 tensor→trainable checkpoint→
동결 round-trip export가 출력을 유지해야 시작한다. 일부 parameter를 누락/랜덤으로
채우고 원본 복원에 성공했다고 하지 않는다. policy/WDL와 moves-left의 사용 여부,
관점·label 의미를 고정하고 강한 교사 분석과 실제 결과를 서로 다른 target으로 저장한다.

**후속 대안:** 추론 비용이 병목이면 작은 CNN 학생에게 강한 LC0/T1 policy·WDL를
증류한다. Maia 구조·checkpoint를 출발점으로 쓸 수 있어도 인간 수 예측을 계속 학습하는
것과 대국 강도 목표의 교사 증류를 구분한다. T1 fine-tuning과 학생 distillation을
한 실험에 섞지 않는다. CNN을 처음부터 재학습하거나 새 latent/recurrent 구조를
추가하는 F03은 이 두 기준선 이후 별도 실험이다.

현재 F의 `LinearFixture`는 합성 숫자 feature와 작은 CPU linear head의 loss·checkpoint·
resume 구현이다. 실제 Maia/T1 body를 학습하거나 Rust ONNX로 export하는 trainer는
아직 없다. recipe 파일과 lifecycle test 통과를 실제 모델 학습 인수로 승격하지 않는다.

첫 **제안** 예산은 local RTX 4050에서 2,000개 position의 교사/label audit, 256개 학습
예제로 overfit/gradient/export smoke, 최대 200 optimizer step·20분·worker 1이다.
microbatch 1부터 VRAM을 재며 accumulation으로 유효 batch를 정한다. GPU 전체 점유
상한·RAM·teacher 시간·data/checkpoint/output bytes와 cancellation을 manifest로
잠그고 한 번의 실패 원인을 보존한다. OOM 뒤 batch/정밀도를 조용히 변경하지 않는다.
추론 약 1.2 GB 관측을 6 GB training 적합성 증거로 사용하지 않는다.

소규모 smoke가 통과하면 game/opening 단위 train/validation/holdout, 중복·전이 누출
감사, frozen baseline, seed·checkpoint 선택 규칙을 고정한 bounded pilot로 확장한다.
교사 원시 policy·search visits·WDL·cp의 의미를 구분하고 필요한 교사 출력이 없으면
scalar score를 확률/방문 수로 만들어 채우지 않는다. validation loss·전술/유일 방어·
endgame·특수 수의 tail failure와 같은 시간 paired 대국을 각각 평가한다.

훈련 스택은 기존 F의 Python 경로에서 PyTorch 등 실제 autograd backend를 비교할
제안이다. Rust inference 결정과 training 언어는 별개다. 새 유료 GPU/장시간 학습은
필요한 모델·데이터·시간·총비용·보존 조건이 확정된 뒤 실행한다. 이번 조사에서는
실제 fine-tuning·교사 dataset 생성·유료 GPU 사용을 시작하지 않았다.

세부 데이터/누출/학습 조건은 [TRAINING-PLAN](../TRAINING-PLAN.md), 실제 F lifecycle
범위는 [F TRAINING](../../experiments/model-research/TRAINING.md)을 따른다.
