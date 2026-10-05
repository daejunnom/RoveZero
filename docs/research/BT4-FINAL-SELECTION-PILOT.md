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

실제 실행 결과는 후속 인수 기록에 추가한다. 이 사전 선언은 GPU 실행·강도 인수 증거가 아니다.
