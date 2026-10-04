# 로컬 RoveZero 내부 강도 A/B

기준일: 2026-10-04, Asia/Seoul. 사용자 지시에 따라 RunPod 지원 응답을 기다리지 않고
PR #17·#18을 develop에 통합한 뒤 로컬 내부 A/B만 진행한다. LC0 외부 대국과 새
클라우드 비용은 이번 실행에 포함하지 않는다. 전체 실험 계약은
[EVALUATION-PROTOCOL](../EVALUATION-PROTOCOL.md)을 따른다.

## 통합·비교 대상

PR #17 merge는 `0318e6fffd1840fd2f00ffcde0dac6a5a2372334`, #18 merge·엔진 소스는
`bdd8523a915404f333497440bc6e5cd8996d7df3`이다. #18의 검증한 제품·문서 head `d412f15`와
최종 develop의 전체 tree가 동일함을 확인했다. main은 변경하지 않는다.

같은 release binary를 `onnx-cuda,experimental-batch`로 빌드하고 **폭 1(S0) 대 폭 4(S1)**만
변경한다. 원본 Maia-1900 W0·동일 export·FP32·worker/thread 1·contract 0.1을 유지한다.
cache·I/O buffer/binding·CUDA Graph와 다른 E feature는 끈다. virtual reservation과
batching으로 선택·완료 분포가 바뀌는 **S 실험**이며, 의미 보존 E 실험으로 부르지 않는다.
기본 활성화·가중치 승격을 이번 결과로 자동 결정하지 않는다.

두 설정의 탐색 알고리즘은 모두 **`rz-puct` revision 1**이다. `c_puct=1.5`, FPU=0,
root noise off, legal-order tie와 backup 코드를 공유한다. 폭 4는 최대 네 pending 평가를
허용하고 이미 예약된 경로에 임시 방문·value 통계를 반영해 다른 leaf를 선택한다.
따라서 이번 비교는 같은 PUCT의 batch·실행 스케줄 실험이며 탐색 알고리즘 교체가 아니다.
완료 순서와 제한 시간 안에 소비하는 평가가 달라질 수 있다. 실제 처리량의 증가까지
대국 점수나 단순 응답 시간으로 확정하지 않는다.

## 첫 고정 cohort의 사전 조건

| 항목 | 잠글 값 |
|---|---|
| 장비 | 로컬 RTX 4050 Laptop 6GB, driver 610.62, WSL2 Linux; 한 번에 한 대국 |
| 시간 | T2, 매 착수 `go movetime 100`; 양쪽 현재 탐색 상한 128회 동일 |
| 표본 | 자가 작성한 24개 고유 opening prefix, 같은 완전 상태에서 색을 바꾼 48판 |
| 순서 | opening 순서는 seed 20261004로 잠금; 고정 runner의 순서로 S0 백 판 뒤 S1 백 판을 한 pair로 진행 |
| 초기화 | 같은 원본 startpos+전체 prefix, 상대 차례 평가·ponder·game 간 cache 유입 금지 |
| 종료 | 점수 adjudication·tablebase 비활성; runner의 현재 3회 반복·50수 claim 자동 수락을 명시하고 A로 감사 |
| 한도 | 최대 200 full moves; cutoff는 Incomplete이며 자동 득점 제외. 실행 wall 1200초·cleanup 10초 |
| 자원 | cohort 전용 cgroup CPU 2 core·RAM 12GiB·swap 0·pids 128; 로그·출력·단일 파일 별도 상한 |
| 실패 | 엔진 불법 수·크래시·시간패는 loss로 보존. 인프라 실패를 유리한 판의 누락으로 처리하지 않음 |
| 통계 | WDL·완료 pair·pentanomial·득점률. opening pair 단위 bootstrap 10,000회·seed 20261004 |
| 해석 | 짧은 T2·128회 상한·소표본·공유 로컬 호스트의 개발 진단; 정식 LC0 강도·승격 인수와 구별 |

실제 manifest에 source/binary·model/export/runtime/bundle·opening/runner hash와 모든
argv·한도를 고정한 뒤 시작한다. 원시 PGN·trace·metadata·실패·recipe는 저장소 밖
`runs/local-strength-20261004/`에 보존하고 Windows reports로 회수한다. 일반 CLI의
Computed-only native V1 receipt는 B>1과 조합하지 않는다. GPU·시계·취소/물리 drain의
검증 범위가 부족하면 `fairness_unverified`와 정식 인수 보류를 명시한다.

## 준비 검증

새 엔진 binary SHA-256은 `313f612eacd61ede278a79263a4aec4c897b42a05cedd8ce228ddee9e7fb29f6`,
수치 검사 binary는 `f9515a3691777ee1a5b3a9d9dd2f4b434bb3c189fe50bf223af9b2028fa8b971`이다.
새 통합 소스에서 CUDA 수치 gate가 exit 0으로 끝났고 전용 cgroup·owned process를
정리했다. 실제 12개 참조·batch 수치·provider mapping의 상세 report를 원시 근거에 연결한다.
이는 C 수치 검사이며 실제 S1 A/C/D/B·대국의 통과로 확대하지 않는다.

수치 gate는 실제 CUDA 실행 node 98개를 확인했다. 독립 참조 대비 최대 오차는
logit `1.5735626e-5`, 합법 policy `8.9406967e-7`, WDL `4.1723251e-7`이고,
single 대 B1/2/4/8/16 최대 logit 차이는 `1.5139580e-5`다. 사전에 잠근 허용 오차
내에서 12개 상태가 통과했다. 폭 1·4 실제 UCI에서 `go nodes 32`와 두 `movetime 100`
착수·종료를 확인했으며 CudaOnnx fresh 완료는 각각 97·187, fatal/overflow/boundary/poison은
0/false다. 이는 요청된 nodes가 stdout에 보고된 방문 수라는 증거는 아니다.

`rz-arena`의 `pair_pgn_audit` 예제는 기존 A Rules 감사기를 직접 호출한다. 전체 이동열·
동일 시작 상태·색 교대·종료를 검증하며 CPU/GPU runtime 계약이나 실행 권한을 만들지
않는다. C 수치, 실제 native 실행, pair PGN·실패 회계, 시계/GPU 공정성과 최종 통계를
각각 확인한다. 실제 대국과 미실행 검사 상태는 실행 후 이 기록에 추가한다.

대국 결과가 나오기 전에 pinned Fastchess의 실제 scheduling을 확인하여 pair 내
실행 순서를 S0 백 → S1 백으로 잠갔다. 양쪽의 총 백·흑 판수는 각각 24이며 전체
순서 효과까지 검증한 조건은 아니다. 두 native 엔진을 `restart=off`로 유지하고
`ucinewgame`에서 상태를 초기화하여 매 판 CUDA bundle의 복제를 피한다.

PGN 감사의 `claim_policy=automatic_acceptance`는 Fastchess의 정확한 종료 문구와
A가 제공한 **현재 상태**의 Available claim을 모두 요구한다. intended-move나 Unknown
근거, 임의 adjudication은 거부하며 결과를 `accepted_claim`으로 표시한다. 기존 native
통합 검사는 `explicit_claim`을 유지하고 별도 claim 영수증 없는 종료를 계속 거부한다.

## 잠금과 runner 연결

`cohort-v4`의 실행 전 manifest SHA-256은
`5f27c12cbfde83f544496b1abe0971a043988706dd3037c5ae8a0e21b8b696e0`,
opening PGN은 `4b5114e71522b953ff9bd95c8e3aeb562d24d3da2c7943996a58e52c78b53f02`다.
같은 24개 순서·전체 이동열은 [작은 재현 입력](../../experiments/baselines/local-strength-openings-v1.json)에 둔다.
동일 보드를 만드는 다른 이력으로 바꾸지 않으며 이 표본을 독립 holdout으로 주장하지 않는다.

외부 runner는 Fastchess `f618e34540f94f4719ad3817950618dabe441318`, binary SHA-256
`ca85b6f3cbaab62352d7c98fb684f427a15c8a03d67825f0f2909eae725d9cfe`다.
runner의 내부 로컬 사용과 외부 배포 권리 검토는 구분한다. 감사기 소스는 `3dc673f`,
binary SHA-256은 `b5ceb926decb8f810f24d10d4efc3bda3588e2e8eff28d408bfe16c93efa7c7d`다.
엔진 모델 초기화는 `startup-ms=60000`, game reset/ping은 각각 5000ms이며
대국 시계의 `st=0.1`, `timemargin=0`을 늘리지 않는다.

득점에 넣지 않는 사전 2판은 자체 A 규칙으로 18ply·색 교대·시작 이력을 감사했고
두 cutoff를 Incomplete로 분류했다. Fastchess maxmoves는 opening 이후의 수를 세므로
본 8ply prefix의 전체 400ply 한도는 `maxmoves=196`으로 맞춘다. 출력 루트 미생성,
기본 10초 startup 초과, 초기 ply ceiling 불일치와 그 과정에서 중단한 0판 본 실행의
실패를 원시 기록에 보존한다. 이 실패를 승패나 무승부로 바꾸지 않는다.

cohort의 native 파일 총합은 8GiB, trace 64MiB, PGN 4MiB, stdout/stderr 각각 8MiB,
단일 파일은 CUDA 라이브러리 복사를 허용하는 1GiB로 제한한다. supervisor·recipe·
wrapper·입력·asset 식별을 잠그고 wall·메모리·PID·종료 후 owned process 정리를 확인한다.
GPU 표본은 장치 전체 관측이며 엔진별 독점 사용량이나 device kernel 시간 증거는 아니다.

## 실행·PGN 감사 결과

`cohort-v4`는 고정한 48판·24pair를 모두 완료했다. whole-process wall은 monotonic
449.104초, runner exit 0, cgroup OOM 0·잔여 PID 0·owned cgroup 제거를 확인했다.
원본 PGN SHA-256은 `411a72742d7d8ea86c11e9f22decf06436c22c3925b05bb2fc7d7bd9b2a1d073`이다.
각 pair의 시작 상태·전체 이력·색 교대·전체 합법 수·종료를 A Rules로 감사했고
python-chess 1.11.2 독립 oracle의 이동열·최종 FEN과 48판 모두 대조했다.

| 결과 | 폭 4 후보 관점 |
|---|---|
| W/D/L | 20 / 8 / 20 |
| 득점 | 24 / 48, 50.00% |
| opening pair 95% bootstrap 구간 | 36.46~63.54%, 10,000회·seed 20261004 |
| 내부 상대 logistic Elo·같은 구간 변환 | 0.00, -96.50~+96.50 |
| pair pentanomial, 0·0.5·1·1.5·2점 | `[5,3,8,3,5]` |
| 정확 종료 / 수락 claim | 체크메이트 40 / 현재 3회 반복 8 |
| 불법 수 / process crash / runner 시간패 / cutoff | 0 / 0 / 0 / 0 |

이 구간은 개발 opening cohort에 조건부인 percentile bootstrap이며 독립 holdout·
정식 승격 구간이 아니다. 이번 표본은 폭 4의 강도 우위를 보이지 않는다. 같은 PUCT의
처리량이 몇 배 증가했는지도 측정하지 않았다. 원본 PGN의 `/0`, `n=0`은 미보고 값의
placeholder라 실제 평가 점수·depth·방문 수·NPS로 해석하지 않는다.

원시 trace에 폭 4 `SearchFailed: WorkerLimit: previous physical worker has not drained`와
legal fallback이 **3건** 남았다. 현재 game·마지막 position/bestmove로 연관한 판은
22·33·34이고 해당 판을 점수에서 제외하지 않았다. 이는 per-request ID journal이 없는
시간적 연관이며 PGN에 그 한계를 표시한다. deadline·stale·cancel 진단도 원시 로그에
보존한다. 합법 착수와 정확 종료가 이 진단의 부재나 GPU drain 통과를 의미하지 않는다.

Fastchess logger는 system-clock, 착수 판정은 steady_clock이다. 두 시계의 차이를
timeout 증거나 search/GPU 지연으로 삼지 않는다. trace에서 `go movetime 100`과
bestmove는 양쪽 각각 2077개, `ucinewgame`은 양쪽 각각 48개, quit은 각각 1개이며
미응답 go는 없다. 매 착수 physical drain·외부 자기 시간의 정식 인수는 남아 있다.
native 종료 aggregate가 Fastchess trace에 수집되지 않아 별도 실제 CUDA UCI 사전
검사의 fresh-completed 근거를 이 48판의 per-game 완료 journal로 확대하지 않는다.

PGN 정리본은 같은 48판·색·이동·결과·시작 prefix를 유지하고 batch 폭·pair/opening ID·
source/model identity·종료 감사·관측 fallback을 표시한다. 정리한 24pair PGN을 같은
A 감사기로 다시 검사하여 원본의 audit JSON과 전체 동일함을 확인했다. 원본과 정리본,
24pair PGN·index·분석·README·inventory를 외부 `reports/coordinator-integration/`
`local-strength-20261004/`에 회수한다. 원시 대국은 Git에 넣지 않는다.
