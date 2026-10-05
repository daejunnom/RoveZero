# BT4 최종 선택 정책 pilot

2026-10-05 사용자 확정은 **16쌍·32판, 전체 최대 120분**이다. 실행 전에 아래
조건과 별도 CUDA V3 lock을 고정한다. 기존 V2 A/A의 cutoff 두 판을 점수로 재해석하지 않는다.

| 항목 | 사전 고정 |
|---|---|
| 연구 변경 | 같은 최소 PUCT에서 S0 `visits` → S1 `exact-terminal` 최종 선택만 변경; 의미 보존 최적화와 별도인 S 변경 |
| 모델·입력 | 동일 BT4-it332 source `e6ada9d6…`, FP32 ONNX `2839171c…`; full startpos 이동열·HistoryFill No·합법 수 순서·WDL 관점 동일 |
| 실행 | 동일 source·release binary; 실제 ORT CUDA·B1·intra-thread 1·worker 1·device 0·TF32 off·raw cache off·arena 3 GiB |
| 탐색 | simulations **4096 상한**을 두 역할에 명시; 실제 방문 수·NN 완료 수는 별도로 기록 |
| 시계 | base 30,000ms + 제시간 착수 뒤 100ms; 초기 추가 증분 없음; 최대 256 ply에는 10 ply opening 포함 |
| runner | Fastchess `f618e34540f94f4719ad3817950618dabe441318` + [별도 시계 패치](../../experiments/baselines/FASTCHESS-CLOCK-PATCH.md); source·patch·수정 binary hash 분리 |
| 시계 경계 | 부모의 `steady_clock`, position 전송 전 → bestmove 수신; ns를 ms로 올림 차감; 100ms read margin은 시간패를 연장하지 않음 |
| loading | 매 판 두 fresh process, 동등한 startup/handshake/drain 예산; model loading은 착수 시계 밖, 전체 120분 안; 경기 상태 사전 NN 분석 없음 |
| 자원 | concurrency 1·전체 CPU quota 2·RAM cgroup 12 GiB·swap 0·pids 128·각 process AS 128 GiB; GPU는 같은 로컬 RTX 4050 6GB |
| GPU 관측 | 동일 주기로 device 사용량/VRAM을 관측; process별 peak나 kernel hard cap의 증거로 확대하지 않음 |
| 표본 | 아래 cohort의 16개 시작 상태; 동일 board/이력에서 흑백 교환 두 판; 짝수 ordinal은 S0 백 판 먼저, 홀수는 S1 먼저 |
| 판정 | A 규칙을 독립 재생; runner 자동 claim은 A의 현재 위치 Available 근거 필요; illegal/crash/time loss는 Loss로 보존 |
| cutoff·오류 | 256 ply cutoff는 Incomplete; provider·clock·계약·인프라 실패 시 증거/PGN/관측 패배를 보존하고 다음 쌍을 중단; 자동 retry·불리한 결과 삭제 없음 |
| 중단 | 고정 16쌍 또는 단조 전체 120분, 취소·오류·자원 상한; 승패를 보고 표본/설정/opening을 변경하지 않음 |

출발점은 [자체 작성한 16개 conventional opening trace](../../experiments/baselines/bt4-final-selection-openings-v3.json)다.
cohort SHA-256은 `df76b0f3fa2768ec3041dbdc95b5e4b701255b2295d8359c10579d2d9b2101a5`,
5955 bytes다. 각 시작에는 10 ply의 실제 이력을 보존한다. 기존 Maia 개발 cohort의
완전 이동열·완전 FEN과 일치하지 않음을 독립 python-chess로 확인했다. 같은 opening
계열 및 모델 학습 자료와의 중복이 없다는 뜻은 아니다. 이 목록은 최종 선택 정책의
개발 회귀 입력과 분리한 pilot이며 일반 opening 모집단을 대표한다고 주장하지 않는다.

통계는 실행 전에 `fixed_paired_hoeffding95`로 잠근다. S1의 두 판 점수 합을 2로
나눈 pair score를 사용하고, 쌍별 결과·WDL·pentanomial `n0..n4`를 보존한다. 독립
pair라는 가정 아래 고정 16쌍 평균의 95% Hoeffding 반폭은
`sqrt(ln(40)/(2*16)) ≈ 0.3395`다. 따라서 작은 pilot의 구간은 매우 넓으며 Elo·
정책 승격을 확정하지 않는다. opening/환경 간 의존은 이 가정의 한계로 명시한다.
미완료 또는 실패로 짝이 빠지면 등록된 16쌍의 분모를 보존하고 빠진 pair score를
0~1 범위로 남긴다. 완료된 좋은 판만 골라 전체 득점률이나 신뢰구간을 만들지 않는다.
실행된 개별 패배도 별도 원장에 남기며 중단된 pilot은 inconclusive다.

코드·GPU 실행·A 재생·전체 시계·physical drain·자원 감시·원시 증거 회수·CI를 각각
인수한다. source와 compiler/feature, engine/runner/모델/bundle/cohort 및 16개 lock SHA를
외부 보고서에 보존한다. 원본 모델·가중치·runtime cache는 유지하며 비활성인 전용
E library snapshot만 검증된 증거 회수 뒤 정리한다. 원시 PGN·로그·모델은 Git 밖에 둔다.

## 최초 시도와 실패 원인 조사

실제 source `f20a621510a69e11025bf4a9a048b06bcd1c1ff8`·engine binary
`79df6e51…a275f`·backend `2ef72d33…d6d53`의 첫 시도는 853.73초 뒤 중단했다.
처음 세 pair/6판은 process/provider/search/전체 clock/A PGN·독립 python-chess를
통과했다. 네 번째 pair의 첫 판은 176 ply의 정상 무승부 PGN을 남겼지만 두 번째
판의 S1 모델 초기화에서 6,291,456-byte BFCArena 할당 오류가 발생했다.
runner exit 1·E CLI exit 2·integration false·scored 0과 원본 로그/PGN을 보존했다.
먼저 완료된 무승부도 그 pair의 완전한 provider/clock 인수로 승격하지 않는다.

오류 당시 cgroup memory.peak는 4,222,369,792 bytes(약 3.93 GiB), max/high/oom/
oom_kill 증가는 0, owned cleanup은 완료였다. device VRAM 표본 최고치는 2254 MiB다.
이전 V2 실패의 약 11.72 GiB host 최고치와 구분한다. 이전에도 Linux OOM kill은 0이었고
동일 오류 문자열만으로 RAM/VRAM 원인을 확정하지 않는다.
[ORT BFCArena 소스](https://github.com/microsoft/onnxruntime/blob/v1.22.0/onnxruntime/core/framework/bfc_arena.cc)는
CPU/device resource allocator를 공통으로 감싸므로 그 예외 자체가 물리 VRAM 부족의
증거는 아니다. Windows host/WDDM·pinned host memory·순간 allocator 여유는 별도다.
[CUDA on WSL의 pinned system memory 제약](https://docs.nvidia.com/cuda/wsl-user-guide/)도
원인 후보이며 이번 실행의 직접 원인으로 확인한 것은 아니다.

같은 old binary·두 엔진 동시 상주·RAM 12 GiB/CPU 2/AS 128 GiB의 독립 진단 두 회는
네 process 모두 model load·32-visit search·quit exit 0을 통과했다. 오류가 재현되지
않았으므로 단순한 두 모델 동시 실행 불가로 결론 내리지 않는다. 진단의 부모 CUDA
메모리 조회는 다른 context의 상주/종료에도 같은 값을 반환해 aggregate VRAM 여유의
유효한 근거로 사용하지 않았다. device 표본의 짧은 순간 peak 누락도 가능한 한계다.

PGN이 없는 startup 실패를 숨기지 않도록 별도 closed audit를 추가했다. 총괄이
보존한 pinned runner의 정확한 game-start/FATAL renderer·역할/색/순서를 대조해
S1의 game-2 startup loss를 확인했다. parser source `d44f059`의 사후 회계이며
f20a621의 원래 E 영수증은 변경하지 않는다. 이 loss는 정상 완료/인수 점수가 아니며
failed attempt의 원장에 남긴다. 남은 pair를 좋은 결과만 골라 완성하거나 자동 재시도하지 않았다.

## C 메모리 정책 변경과 별도 인수

사용자는 오류 확인·개선 가능하면 개선·계속을 요청했다. source
`d44f05929be1a1f6565f0c91421b2e7312da60f7`은 CUDA arena를
`SameAsRequested`로 명시해 불필요한 power-of-two 확장을 줄인다.
모델·입력·FP32·TF32 off·kernel·3 GiB arena 상한·PUCT·최종 선택 정책은 유지한다.
CUDA backend identity에 `cuda-arena-extend=same-as-requested-v1`을 추가하고 old
identity와 혼용하지 않는다. CPU 설정과 identity는 유지한다.
[ort rc.10의 명시 옵션](https://github.com/pykeio/ort/blob/v2.0.0-rc.10/src/execution_providers/cuda.rs)과
ORT의 allocation 구현을 대조했다. 이는 memory 정책 개선이며 오류 근본 원인을
확정하거나 무오류 실행을 보장한 수정은 아니다.

| 검사 | source d44f059의 실제 결과 |
|---|---|
| LC0 독립 protobuf 수치 | BT4 12-case와 batch 1/2/4/8/16 통과; logits 최대 절대 오차 6.509e-5·WDL 1.789e-7·합법 policy 2.802e-6 |
| 실제 A→C→D | No/Repeat를 포함한 12-case의 입력 tensor 오차 0·합법수/순서·평가/finalization·물리 종료 통과 |
| dual resident | 같은 CPU/RAM/AS와 입력에서 old/new 각각 두 process model load·32-visit search·exit 0 통과 |
| GPU 메모리 표본 | 한 쌍의 old→new probe 최고치 2398→1710 MiB, 688 MiB 감소. observer context를 포함한 whole-device 표본, 일반적 감소율/모든 순간 peak/속도 주장 아님 |
| 로컬 source 검사 | workspace all-target/all-feature 749 passed·0 failed·16 ignored, fmt·strict Clippy·release 성공. ignored는 로컬 미실행 |
| exact-source CI | [37261878758](https://github.com/daejunnom/RoveZero/actions/runs/37261878758) Windows/Ubuntu 필수 step SUCCESS 직접 확인. CPU CI와 GPU 실행은 별도 |

재실행 engine binary SHA는
`a46e941c63253d6a898121fe662f8aa3981702ba5874698a8d7dac18259cf1c6`,
B1 backend SHA는 `e5fae5b061fab8e83bcc8aabac462c42ca710e84ac30ebffa3da002c43142857`다.
runner SHA는 `04b83c5ba8af01429db82ff1b2cabda5bf8185678ed9047134f42833a5f77d83`,
patch/cohort와 B1/clock/seed/자원은 동일하며 16개 lock을 먼저 고정했다.
두 역할 모두 같은 새 C backend를 사용한다. 기존 오류와 모든 점수는 보존하되
새 시도의 결과와 합치지 않는다. 이미 관측한 cohort를 재사용한 pilot이므로
새로운 미관측 opening holdout이라고 주장하지 않는다. 진단·재실행은 처음의 120분
안에서 끝내며 단조 시간의 추가 한도와 누적 wall deadline을 감독자가 집행한다.

source d44f059의 두 번째 시도도 818.81초 뒤 네 번째 pair/game-2의 S1 model load에서
같은 6,291,456-byte 할당 오류로 중단했다. 세 pair/6판은 완전 인수됐고 S1 W1/D3/L2,
pair score 합계 1.25지만 등록된 16쌍 중 13쌍은 빠져 있다. 고정 분모의 점수 가능 범위는
0.078125~0.890625, Hoeffding 95% 범위는 [0,1]이며 전체 강도는 inconclusive다.
첫 시도와 같은 WDL도 두 시도의 점수·표본을 합치는 근거가 아니다.

실패한 pair-03의 first game은 124 ply의 insufficient-material draw로 독립 재생했다.
game-2는 PGN 생성 전 S1 startup loss이며 새 E failure audit가 직접 보존했다.
그 pair의 integration false/scored 0을 유지한다. 실패한 pair의 RAM peak
4,222,263,296 bytes·memory max/oom 사건 0, device 표본 최고치 1630 MiB만으로
물리 VRAM 부족을 확정하지 않는다. 요청 크기 arena 변경은 측정한 점유 감소를
제공했지만 이 반복 실패를 해결하지 않았다. CPU/CUDA/pinned-host allocator를
구별할 별도 INFO 진단을 진행하며 진단 전용 바이너리·패치와 결과를 정식 대국과 분리한다.

실행 요청·잠금·부분 진행을 완료한 16쌍으로 표시하지 않는다. 원시 자료의 논리 루트는 저장소 밖
`reports/coordinator-integration/native-bt4-holdout-20261005/`다.
