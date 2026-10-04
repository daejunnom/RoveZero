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
