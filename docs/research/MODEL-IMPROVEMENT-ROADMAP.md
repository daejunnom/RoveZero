# 모델·표현·계산 배분의 후속 연구

기준일: 2026-10-06. 현재 구현 기준은 PR #20의
`f442c41aa6f1885d4ae06aab874420a4cd8f7062`다. 이 문서는 이전 대화의 모델 자체 개선과
탐색 연구를 보존하는 **미채택 연구 목록**이다. 신규 모델 구현·훈련·가중치 배포·GPU 실행은
수행하지 않았고, 이번 [메모리·실행 최적화](MEMORY-EXECUTION-OPTIMIZATION.md)와 독립적으로 관리한다.

[AGENTS](../../AGENTS.md), [CONTRIBUTING](../../CONTRIBUTING.md),
[학습 계획](../TRAINING-PLAN.md), [실험 계약](../EXPERIMENTS.md),
[기존 76개 후보](../CANDIDATE-REGISTER.md)를 따른다. `MODEL-*`와 `SEARCH-*`는
이 문서의 연구 항목 ID이며 TASK/CARD/OPT를 대체하지 않는다. 낮은 GPU 직접 적합성,
보류·기각·부정적 결과도 남긴다. 원문 성과를 RoveZero의 예상 배속·Elo로 환산하지 않는다.

## 1. 구현과 가설의 경계

현재 BT4-it332는 64칸 Transformer이며 기록된 body는 encoder 15개·attention head 32개·
embedding 1024·FFN 1536이다. 입력 `[B,112,8,8]`, policy logits `[B,1858]`,
WDL `[B,3]`, MLH `[B,1]`를 사용한다. 이 모델은 Maia-1900 CNN의 인간 수 예측 기준선과
구분한다. MLH는 graph에 존재하지만 현재 RoveZero search는 소비하지 않는다.[^rz-model]

현재 선택한 모델은 한 번의 feed-forward 평가를 수행한다. 일반 LLM의 긴 문맥처럼
매 노드의 모든 층 KV를 지속적으로 보관해 재사용하는 구조라고 가정하지 않는다.
raw 평가 캐시, Rules state 캐시, runtime 파일 공유와 학습된 KV/잠재 메모리는 다르다.[^rz-eval]

양방향 보드 attention은 한 칸 변화가 다른 칸의 깊은 표현도 바꿀 수 있다. 움직이지 않은
62칸의 모든 층 KV를 그대로 쓰는 것은 일반적으로 정확한 재사용이 아니다. 먼저 어떤 표현이
독립적으로 갱신 가능한지 설계하고, 그 조건이 깨지는 부분은 근사 A 실험으로 표시한다.
15개 서로 다른 encoder를 통과하는 현재 BT4와 가중치 공유 recurrent block을 구분한다.

| 트랙 | 고정/변경 | 판정 |
|---|---|---|
| E 실행 | 같은 weights·평가값·탐색 선택 의미, 할당·복사·실행 경로만 개선 | 메모리·실행 문서와 동일 작업량 대조 |
| A 모델 | 입력 표현·가중치·정밀도·head·KV 압축·재구성 변경 | 품질·속도·메모리·같은 시간 대국을 모두 비교 |
| S 탐색 | 선택·최종 착수·방문/정밀화 배분·배치/재사용 통계 변경 | CONTROL-0→1, 같은 모델에서 독립 비교 |
| O 오프라인 | 데이터·optimizer·교사·해석·훈련 | 학습 비용·누출·재현성, 이후 별도 export/대국 |

정밀도나 압축으로 평가가 달라지면 PUCT 소스가 같아도 엄격한 E가 아니다.
[상충 채택 기준](MEMORY-EXECUTION-OPTIMIZATION.md#adoption-gate)은 실행·메모리의 문턱이며
모델 정확도 손실이나 Elo 승격을 승인하는 규칙이 아니다. A/S 실험의 품질 인수가 추가로 필요하다.

## 2. 압축으로 모델을 키우는 실험

### 2.1 저장 공간과 활성 계산량을 분리한다

GPU 메모리는 `weights + activation/workspace + device cache + runtime`으로 나누어 기록한다.
압축 대상이 전체의 비중 f이고 r배 압축하면, 다른 조건 고정 시 전체 절약은 `f*(1-1/r)`이다.
예를 들어 f=0.1, r=4이면 전체의 7.5%만 줄어든다. 이는 가정 계산이지 실측이 아니다.
CPU raw 캐시를 줄였다고 그 bytes가 그대로 GPU 모델 공간이 되는 것은 아니다.

압축 이후 모델을 키우면 activation·workspace·KV 폭도 다시 커질 수 있으므로 최종 조합의
전체 peak를 재측정한다. Dense 폭을 키우는 방식은 주요 행렬 연산도 증가한다. 저장 파라미터,
활성 파라미터, 실제 FLOPs/MAC, 장치 이동 bytes, 유용한 평가 지연을 따로 기록한다.

| 확대 방식 | 이득 가설 | 주요 위험 |
|---|---|---|
| Dense 폭/깊이 | 더 풍부한 관계 판단 | 대부분의 추가 가중치를 매 평가에 읽고 계산 |
| 소규모 MoE/전문 head | 상황별 전문화·선택 계산 | routing·작은 GEMM·전체 VRAM·전문가 간 calibration |
| 조건부 기억 | 많은 저장 용량에서 필요한 항목만 조회 | random access·충돌·데이터 부족·offload 지연 |
| 공유 반복 블록 | 파라미터 증가 없이 가변 추론 깊이 | 반복 횟수만큼 계산; sharing 자체는 FLOPs 제거가 아님 |

### 2.2 필수 대조군

|  | 비압축 | 압축 |
|---|---|---|
| 작은 모델 | 기준선 | 압축 자체의 손익 |
| 큰 모델 | 용량 확대 자체의 손익 | 압축+용량 확대의 상호작용 |

같은 메모리를 더 큰 모델 대신 cache capacity·다른 배치에 쓰는 대조군도 별도로 둔다.
모델이 커져 같은 평가 수에서는 강해져도 동일 시간의 평가 횟수가 줄어 약해질 수 있다.
따라서 같은 고유 NN 평가 수, 같은 벽시계, 같은 RAM/VRAM 한도, 같은 학습 예산을 각각 보고한다.
NPS·파라미터 수·퍼즐 정확도를 최종 대국의 대용으로 쓰지 않는다.

## 3. 모델 연구 후보

GPU 열은 **직접 실행 적합성의 추론/미측정**이다. CPU 보조·오프라인 훈련의 가치와 다르다.
가중치 호환은 별도 검증하며, 구조 변경에 기존 BT4 weights를 단순 재배치해 같은 실력을
유지할 수 있다고 가정하지 않는다.

| ID / 후보 | 체스 적용 가설·원리 | 비용·실패 위험 / 최소 대조 | 분류 / GPU 추론 |
|---|---|---|---|
| MODEL-01 강한 교사 탐색 증류 | 최선수뿐 아니라 후보 분포·WDL·깊은 반박 수·얕은 오류 학습.[^distill] | 교사 비용·편향·불안정 라벨·game/opening 누출; 동결 기준선 유지 | A/O / 훈련 높음, 추론 모델 의존 |
| MODEL-02 혼합 정밀도/PTQ/QAT | 민감한 head는 높은 정밀도, 나머지 FP16/INT8 등 비교.[^quant] | kernel fallback·복원 비용·유일 방어 순위 역전; 같은 모델부터 대조 | A / 높음~조건부 |
| MODEL-03 칸/entity/혼합 토큰 | 안정 ID·좌표와 정확한 빈칸 정보를 보존하며 변경분 친화적 입력 | 재정렬·padding·빈칸/이동선 누락; tokenizer만의 효과 분리 | A / 조건부 |
| MODEL-04 관계 입력·GNN/hybrid | 공격·방어·차단·폰 쌍을 보조 edge/embedding/bias로 제공.[^graph] | graph 구축·희소 gather 비용; 기존 관계 처리 누락한 약한 baseline 금지 | A / 조건부 |
| MODEL-05 BPE식 패턴 토큰 | 정확 보드는 유지하고 반복 구조를 보조 특징으로 압축 | 한 칸·차례 차이 누락·사전 탐지 비용; 과도한 macro move는 ply별 검증 | A / 조건부~낮음 |
| MODEL-06 Perceiver 잠재 병목 | 작은 latent에서 깊은 FFN/attention, 정확 보드 재조회.[^perceiver] | 전술 정보 병목·cross-attention 비용; 슬롯 수/재조회 빈도 ablation | A / 높음~조건부 |
| MODEL-07 공통 인코더+후보 디코더 | z=E(s) 한 번 후 Q(s,a)=D(z,a,delta), 선택 자식만 정밀 평가 | 안 쓸 형제의 선계산 낭비; 예비 Q는 실제 방문 아님 | A/S / 조건부 |
| MODEL-08 YOCO/CLA식 층간 KV 공유 | 같은 보드 메모리의 K/V를 여러 layer/질의가 재사용.[^yoco][^cla] | attention 출력까지 공유되는 것은 아님; 기존 weights/표현 의미 변경 | A / 높음~조건부 |
| MODEL-09 MLA식 낮은 차원 메모리 | 큰 중간 상태 대신 작은 latent를 저장·복원.[^mla] | 복원·projection·cache miss; 기존 모델의 무손실 형식 변경이 아님 | A / 조건부 |
| MODEL-10 KIVI/TurboQuant 캐시 양자화 | 실제 특징 분포에 맞춘 축·block·정밀도 비교.[^kivi][^turbo] | 회전·unpack·dequant 비용, 꼬리 오차; 낮은 bits가 배속은 아님 | A / 조건부 |
| MODEL-11 공유 recurrent 잠재 추론 | 반복별 policy/WDL을 학습하고 더 어려운 상태를 정밀화.[^recurrent] | 같은 시간 추가 트리 확장보다 나은지 비교; 반복은 공짜가 아님 | A, 적응형은 S / 조건부 |
| MODEL-12 부모 latent warm-start | 이동 delta로 부모 해석을 수정하고 정확 자식 보드로 보정 | 전술 불연속·경로 의존·오차 누적; fresh와 다른 경로의 동일 상태 비교 | A / 조건부 |
| MODEL-13 DEQ/Consistency 보정 | 많은 반복의 결과를 적은 단계로 접근하도록 학습.[^deq] | 잔차 수렴은 최선수 보장 아님; 초기값·안정성·훈련 비용 | A/O / 조건부 |
| MODEL-14 DeltaCNN/NNUE식 풍부한 증분 특징 | 독립 기여는 정확 갱신, 전역 부분은 별도 계산.[^delta] | 변경 영향의 의존성 추적 비용; full 대조; 깊은 KV를 그대로 재사용하지 않음 | E 조건부 또는 A / 직접 조건부, CPU 높음 |
| MODEL-15 선형 attention 관계 요약 | 독립 k/v의 합산 기여를 빼고 더함.[^linear] | 전역 문맥화 뒤에는 국소 갱신 불가; softmax와 다른 모델, 요약 행렬 비용 | A / 조건부 |
| MODEL-16 Mamba/Gated Delta 상태 | 긴 이력 또는 warm-start용 작은 순환 기억.[^ssm] | 순서 의존·transposition 편향; 64칸의 짧은 입력 이점 불명 | A / 조건부~낮음 |
| MODEL-17 소규모 MoE/조건부 FFN | 일반 본체 후 전문 head/module·block 단위 라우팅.[^moe] | dense 대비 비용·routing·데이터 희소·전문가 전환 불연속 | A/S / 조건부 |
| MODEL-18 Engram식 조건부 기억 | 구조 index로 학습 embedding 조회, 문맥 gate로 반영.[^engram] | random access·충돌·host 전송; 작은 상주 table부터. 평가 cache와 다름 | A / 조건부, CPU 후보 |
| MODEL-19 IndexCache/ToMe/Mixture-of-Depths | 인덱스 재사용·보조 latent 병합·일부 슬롯만 갱신.[^index][^conditional] | 64토큰은 dense가 더 쌀 수 있음; 결정적 정보 누락·gather/scatter | A/S / 조건부 |
| MODEL-20 DeepCache/TeaCache 선택 재계산 | 변경량으로 cache 갱신/반복/완전 재계산 gate 학습.[^tea] | 작은 상태 변화가 큰 전술 변화를 만들 수 있음; false-reuse tail 검증 | A/S / 조건부 |
| MODEL-21 재매개변수화·희소성·저비트 학습 | RepVGG/FastViT의 병합, SparseGPT/BitNet 및 Marlin/Sage형 실행.[^compression] | 합칠 수 없는 동적 연산·sparse kernel 부재·재학습; dense 소형 모델 비교 | A 또는 검증된 E / 하드웨어 조건부 |
| MODEL-22 head·optimizer·적응 학습 | WDL·계산 효용·위협 보조 목표, SOAP/AdamW, LoRA/FiLM/Hypernetwork.[^training] | loss 충돌·optimizer memory·동적 weight 생성; 학습 향상을 추론 배속으로 보지 않음 | A/O / 모델·훈련별 상이 |
| MODEL-23 세계 모델·검색 기억·데이터 연구 | MuZero latent 전이, RAG 유사 구조, Diffusion/Flow 생성, 해석·구조화 궤적.[^offline] | 정확 Rules 대체 금지·도달성·라벨 검증·누출; LLM 직접 착수는 목표 아님 | A/O / 직접 낮음~조건부 |
| MODEL-24 DeepSeek-V4.1-Flash에서 논의한 결합 | CED/CSA2/FP4 KV/bounded replay/single-pass mHC를 연구 목록으로 보존.[^v41] | 초기 접근 실패와 후속 v1 본문 확인을 함께 기록. BT4 호환·수치·기력은 미검증 | 보류 A/S / 체스 미평가 |

원문이 체스에서 검증한 것, 타 분야에서 검증한 것, 위 표의 신규 조합을 구분한다.
MOE·토큰 축소·희소화·극저비트에 대한 부정적 결과는 기존 등록부/핸드오프에 그대로 남긴다.
학습 시간 감소, 저장 크기 감소, GPU 종단 지연 감소, 실제 Elo 상승을 서로 바꾸어 말하지 않는다.

### 3.1 우선 구조 가설: 공유 보드 메모리와 반복 판단

`M=E(s)`, `K,V=P(M)`, `z[t+1]=R(z[t], Attention(Q[t],K,V))`를 후보로 둔다.
정확한 보드는 유지하며 먼저 **동일 보드 안**의 공유만 검증한다. 부모·자식 사이의 공유는
MODEL-12/14의 독립 조건과 검증이 필요하다. 고정 반복 수로 시작한 뒤 adaptive 반복은 S로 분리한다.

메모리 정밀도, 반복 블록 폭, 반복 횟수, 교사/loss를 동시에 바꾸지 않는다. 공유 자체→압축→
용량 확대→선택적 반복 순서로 ablation하고, 기존 BT4 ONNX와의 단순 호환이라고 표시하지 않는다.
정확한 raw 결과와 경로 의존 근사 상태는 타입·namespace·provenance가 달라야 한다.

### 3.2 DeepSeek-V4.1-Flash의 접근 이력과 확인 범위

이전 대화에 등장한 CED/CSA2·FP4·bounded replay·single-pass mHC를 삭제하지 않되,
PR #21 초기 문서 작성에서 arXiv 원문 접근이 실패했던 기록을 보존한다.
특히 global GPU KV와 host/SSD persistent cache, 전체 모델 memory를 구분하고 기존 대화의
1/4·1/8 같은 비율을 RoveZero의 예상 절약으로 사용하지 않는다. 원문을 확보한 뒤 제목·버전·
정확/근사 복원·실행 hardware·조건을 기록해야 한다. 그전에도 원문 확인이 가능한 YOCO/CLA,
KIVI, MLA, Engram, IndexCache 등의 독립 가설은 평가할 수 있다.

2026-10-06 총괄 재검토에서는 **DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache
Compression**, `arXiv:2609.19969v1`(2026-09-17)의 HTML 본문에 접근했다.[^v41-verified]
이는 아래 원저자 설계의 확인이며 RoveZero 구현·benchmark 재현이 아니다.

- CSA2의 Full/Reindex/Reuse는 main KV·indexer K와 Top-K 인덱스의 공유를 구별한다.
  각 층의 query·SWA KV·attention 출력은 계속 계산한다. 인덱스만 공유하는 것과
  KV 저장량 감소를 혼동하지 않고 현재 BT4의 서로 다른 15개 층에 그대로 적용하지 않는다.
- FP4 main KV는 attention 전에 복원하는 저장 형식이며 행렬곱 가속 자체가 아니다.
  원문은 post-training QAT와 별도 SWA 정밀도를 사용한다. 동결 BT4의 무손실 E 변경으로
  분류하지 않으며 모델·수치·지원 kernel의 독립 인수가 필요하다.
- SWA bounded replay는 제한 구간을 다시 계산해 **근사 상태**를 복원한다. 원문은 cache-hit
  위치에 따른 상태 차이와 post-training 적응을 설명한다. 정확 평가 캐시와 분리한 A 연구다.
- 원문의 약 1/4은 global KV의 HBM 저장, 약 1/8은 host/SSD persistent KV의 비교다.
  전체 weights·activation·workspace나 체스의 64칸 평가에 같은 감소율을 주장하지 않는다.

CED·single-pass mHC·훈련 및 장비별 성능의 체스 전이는 후속 연구로 남긴다.
구조·정밀도·반복·스케줄을 한 번에 활성화하지 않고 한 변수 대조와 권리 확인을 거친다.

## 4. 탐색 의미가 바뀌는 항목은 별도 S 목록

이 목록은 현재 E 메모리 작업의 구현 범위에 들어가지 않는다. 모델 변경과 동시에 활성화하지 않는다.

| ID | 후보 | 최소 비교·위험 |
|---|---|---|
| SEARCH-01 FPU/Cpuct/policy temperature | 같은 BT4에서 부모 평가 기반 FPU·방문 의존 Cpuct·root 설정 비교 | 미방문 Q=0 기준과 비교; 상수 복사만으로 LC0 우위 주장 금지 |
| SEARCH-02 exact terminal/solver/저예산 착수 | 확정 종료·방문 수 선택·root prior fallback을 구분 | #20의 exact_terminal은 방문한 직접 자식만 본다. 전체 solver가 구현됐다고 하지 않음 |
| SEARCH-03 적응형 batch·DSpark식 선행 계산 | 실제 소비 확률·남은 시간·배치 latency로 폭 선택.[^dspark] | 완료 순서만 맞춰도 순차 선택과 같지 않음; 취소·낭비·deadline 포함 |
| SEARCH-04 Gumbel+Sequential Halving | root 후보에 한정한 예산 배분 비교.[^gumbel] | 초기 낮은 prior의 유일 방어 누락; 근사 value에서 보장 자동 이전 금지 |
| SEARCH-05 subtree/DAG 통계 재사용 | 실제 착수 뒤 자식 통계 보존·감쇠 비교 | raw cache와 다름; 반복·이력·edge별 통계·cycle·기존 방문 집계 |
| SEARCH-06 multi-fidelity·큰 모델/CPU 전술 검증 | 다른 후보 탐색과 현재 후보 정밀화의 선택.[^multifidelity] | 재평가를 새 방문으로 중복 집계 금지; 값 척도·신뢰도·helper 비용 포함 |

Speculative decoding의 원래 출력 분포 보장은 체스 MCTS에 자동 적용되지 않는다.[^speculative]
원래 선택·반영 순서를 유지하며 정확한 평가만 미리 준비하는 보수형도 cache identity·수치·
시간 외 계산 금지를 검증해야 한다. 실제 선택과 통계가 바뀌면 S로 기록한다.
CPU PVS/LMR/null-move/SEE·이력·correction 등의 기존 CARD-F 후보도 보존한다. GPU 직접
적합성이 낮아도 CPU 비교·오프라인·명시 helper 후보일 수 있으나 alpha-beta의 bound 의미를
PUCT 확률/방문에 그대로 대입하지 않는다.

## 5. 학습·인수 계약

F가 데이터·학습을, C가 인코딩·export·추론을, B가 평가 소비·탐색을, D가 실행·수명을,
E가 대국·통계를 맡고 공통 변경은 총괄에 제안한다. Roles나 TASK ID를 이 문서로 재배정하지 않는다.

1. 동결 BT4 기준의 모델/입력/출력 identity를 고정하고 현재 CUDA/CPU 실행 근거를 source별로 구분한다.
2. 권리가 확인된 모델·데이터·도구만 실제 학습·재배포 대상에 둔다. BT4의 개별 weight rights
   미확인 기록을 자체 MIT 코드나 다른 T1 허가로 덮지 않는다. 문서 작성은 학습 승인이 아니다.
3. 같은 data·train budget·seed 계획·search에서 한 가지 구조를 바꾸고 고정/가변 계산 비용을 측정한다.
4. 원본→trainable 복원→동결 round-trip→gradient→checkpoint/resume→export→Rust 수치 대조를 확인한다.
5. game/opening/lineage별 split·중복/transposition 누출·holdout 접근을 통제하고 교사 비용도 기록한다.
6. 순위 정확도만 아니라 유일 방어·강제 전술·WDL calibration·tail error·fresh/warm 경로 편차를 본다.
7. 내부 E runtime 비교, CONTROL-0→1 search, CONTROL-1→2 학습, 외부 LC0 대국을 분리한다.

최종 대국은 [평가 규약](../EVALUATION-PROTOCOL.md)의 동일한 완전 시작 상태에서
엔진 흑백 배정 교환, 동일한 CPU/GPU·memory·벽시계 예산, W+0.5D 득점률,
pair 단위 상관·holdout·사전 통계·실패 보존을 따른다. 코드 제공·CPU CI·8판 pilot 결과를
새 모델의 정식 Elo 인수로 바꾸지 않는다. 새 model/batch/precision마다 latency와 peak를 재측정한다.

## 6. 이전 논의와의 대응·보존

| 기존 논의 | 이 문서 / 기존 등록부 |
|---|---|
| tokenizer·entity·GNN·BPE·관계·MoE | MODEL-03~07/17, CARD-A01~A07 |
| NNUE·DeltaCNN·dataflow·선형/MLA·KV 압축·Paged/Radix/H2O | MODEL-09~16, CARD-B01~B09; 정확 저장·캐시 관리는 별도 E 문서 |
| 적응형 계산·latent·warm-start·DEQ·diffusion cache | MODEL-11~13/19~20, SEARCH-03/06, CARD-C01~C08 |
| Roofline·Flash·wavefront·커널·재매개변수화·희소/저비트·Ansor | MODEL-21, CARD-D01~D10; 실행 보존 여부에 따라 E/A/S 분류 |
| 증류·WDL·SOAP·self-play/GRPO·LoRA·FiLM·Hypernetwork | MODEL-01/22, CARD-E01~E06; 기존 학습 계획의 RL 대조 보존 |
| MuZero·RAG·생성·인과 해석·구조화 궤적·자동 연구 | MODEL-23, CARD-E07~E12; 온라인 LLM 착수는 중심 경로 아님 |
| Reckless 등 현대 엔진·CPU 탐색/NNUE 상세 | 기존 CARD-F01~F30, SEARCH-01/02/05/06; CPU 후보도 보존 |
| 최근 압축·조건부 기억·공유 KV | MODEL-08~10/18~20/24, SEARCH-03 |

이 표는 기존 원문·출처를 제거하고 새 표로 치환하는 것이 아니라 연결 색인이다.
과거의 가상 FFN=4d 계산, parameter sharing=가속, 압축률=전체 메모리 절약,
높은 GPU 점유율=높은 Elo 같은 해석은 채택하지 않는다.

## 7. 출처 주석

공식 연구의 원리와 위 표의 체스 적용 가설을 구별한다. 일부 링크는 이전 핸드오프의
추가 검토 목록을 보존한 것이며 모든 원문 코드·수식·실험을 이번에 독립 재현한 것은 아니다.
DeepSeek-V4.1의 초기 미확인과 후속 확인 범위는 아래 주석을 구분한다. 실제 채택 때 버전·commit·권리·실험 조건을 잠근다.

[^rz-model]: [LOCAL-MODEL-BASELINE @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/docs/research/LOCAL-MODEL-BASELINE.md). BT4 검사 body·shape·MLH·권리와 source별 실행 근거.
[^rz-eval]: [onnx.rs @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/onnx.rs). 평가 session·입출력·실험 옵션; persistent KV 구현의 근거로 사용하지 않는다.
[^distill]: [Searchless Chess](https://github.com/google-deepmind/searchless_chess). 교사 행동·가치 학습의 공식 구현. 최신 LC0를 동일 시간에 이겼다는 증거가 아니다.
[^quant]: [ORT FP16/mixed precision](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html). 지원 연산·수치 검증을 포함하는 실행 후보.
[^graph]: [Graph representation for chess RL](https://arxiv.org/abs/2410.23753), [LC0 Transformer Progress](https://lczero.org/blog/2024/02/transformer-progress/). 관계 표현의 기존 근거와 이미 있는 기준선.
[^perceiver]: [Perceiver IO](https://arxiv.org/abs/2107.14795). 입력/출력과 깊은 latent 계산 폭의 분리.
[^yoco]: [You Only Cache Once](https://arxiv.org/abs/2405.05254). Decoder-decoder·KV 공유 원리. 체스의 재사용 구조는 별도 가설.
[^cla]: [Cross-Layer Attention](https://arxiv.org/abs/2405.12981). 층간 K/V 공유; attention 결과의 동일성을 주장하지 않는다.
[^mla]: [DeepSeek-V2](https://arxiv.org/abs/2405.04434). MLA의 저차원 잠재 KV.
[^kivi]: [KIVI](https://arxiv.org/abs/2402.02750). 분포/축에 따른 비대칭 KV 양자화.
[^turbo]: [TurboQuant](https://arxiv.org/abs/2504.19874). 회전 기반 벡터 압축; 체스 실측 이득은 미정.
[^recurrent]: [Recurrent Depth](https://arxiv.org/abs/2502.05171). 문장 생성 대신 공유 block의 잠재 반복.
[^deq]: [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377), [Consistency DEQ](https://arxiv.org/abs/2602.03024). 수치적 고정점과 의사결정 정답성을 구별한다.
[^delta]: [DeltaCNN](https://arxiv.org/abs/2203.03996), [Stockfish NNUE](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html). 영상 변경분·체스 증분 accumulator; 임의 깊은 표현으로 일반화하지 않는다.
[^linear]: [Transformers are RNNs](https://arxiv.org/abs/2006.16236). Kernel 요약과 합산 기여의 독립 조건.
[^ssm]: [Mamba](https://arxiv.org/abs/2312.00752), [Gated Delta Networks](https://arxiv.org/abs/2412.06464). 순환 상태는 순서 독립 cache가 아니다.
[^moe]: [M2CTS](https://arxiv.org/abs/2401.16852), [Fast Feedforward Networks](https://arxiv.org/abs/2308.14711). 기준선·routing·하드웨어 조건을 분리한다.
[^engram]: [Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372). 조건부 기억·학습 lookup; 포지션 평가 cache와 다른 연구 축.
[^index]: [IndexCache](https://arxiv.org/abs/2603.12201). 층별 희소 index 재사용; 64칸 이득은 미검증.
[^conditional]: [Token Merging](https://arxiv.org/abs/2210.09461), [Mixture-of-Depths](https://arxiv.org/abs/2404.02258). 보조 latent 제한 실험으로 시작한다.
[^tea]: [DeepCache](https://arxiv.org/abs/2312.00858), [TeaCache](https://arxiv.org/abs/2411.19108). 확산 반복의 재사용을 체스 전술 연속성 보장으로 쓰지 않는다.
[^compression]: [RepVGG](https://arxiv.org/abs/2101.03697), [FastViT](https://arxiv.org/abs/2303.14189), [SparseGPT](https://arxiv.org/abs/2301.00774), [BitNet](https://arxiv.org/abs/2402.17764), [Marlin](https://github.com/IST-DASLab/marlin), [SageAttention](https://arxiv.org/abs/2410.02367). 구조·학습·저장·실행 변화를 구분한다.
[^training]: [SOAP](https://arxiv.org/abs/2409.11321), [AdamW](https://arxiv.org/abs/1711.05101), [LoRA](https://arxiv.org/abs/2106.09685), [FiLM](https://arxiv.org/abs/1709.07871), [HyperNetworks](https://arxiv.org/abs/1609.09106), [Ceres](https://github.com/dje-dev/Ceres). 학습·조건부 적응·보조 출력의 참고.
[^offline]: [MuZero](https://arxiv.org/abs/1911.08265), [RAG](https://arxiv.org/abs/2005.11401), [Flow Matching](https://arxiv.org/abs/2210.02747), [LC0 look-ahead analysis](https://arxiv.org/abs/2406.00877). 정확 규칙·도달성·라벨·인과 개입을 별도 검증한다.
[^v41]: 이전 대화의 출처 후보 [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/abs/2609.19969). **2026-10-06 이번 조회에서 abs/HTML 원문 접근 실패. 상세 원문 미확인으로 보존하며 세부 주장·수치의 확정 근거로 사용하지 않는다.**
[^v41-verified]: [DeepSeek-V4.1-Flash v1 HTML](https://arxiv.org/html/2609.19969v1). 2026-10-06 총괄이 본문 접근과 2.3.1·2.4.4·3.2.1~3.2.2의 위 범위를 확인했다. 초기 접근 실패를 성공으로 바꾸지 않으며 체스 적용·훈련·GPU 성능은 미인수다.
[^dspark]: [DSpark](https://arxiv.org/abs/2607.05147). Confidence-scheduled speculative decoding; 체스 배치 제어는 신규 적용 가설.
[^gumbel]: [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO), [Mctx](https://github.com/google-deepmind/mctx). 근사 체스 value에 원문의 개선 보장이 자동 이전되지 않는다.
[^multifidelity]: [Optimal Multi-Fidelity Best-Arm Identification](https://arxiv.org/abs/2406.03033). 체스의 편향·상관 조건은 별도 검토한다.
[^speculative]: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). 언어 생성의 동일 분포 보장은 MCTS 방문 통계 보장이 아니다.
