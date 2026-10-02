# 실제 runner checkpoint 1 — 부분 성공과 취소 preflight 실패

원본 source commit은 `fcbe8b25d53ee96f48ca8dafd227e7c46f8d90ab`이며 원본 `run/reproduction-receipt.json`에 기록되어 있다. A Rules source는 `118dc0311a88e143be285703940321dc16261f6a`, native contract source는 `67284c4f66f7a7ae9f46fa63dfd50e7410eb6845`, 외부 Fastchess source는 `f618e34540f94f4719ad3817950618dabe441318`이다.

normal·cutoff는 실제 CPU subprocess를 수행했고 각각 raw stdout/stderr·PGN·opening·config·ProcessReceipt·ledger·CLI command/exit/stdout/stderr를 남겼다. 엔진은 scripted synthetic fixture이며 NN/GPU/playing-strength 결과가 아니다. normal은 1개 완료 pair(W=1,L=1), cutoff는 2게임 incomplete라 제외 pair이고 무승부 점수로 집계하지 않았다.

cancel 명령은 exit code 2 및 `process.cancelled_before_spawn` stderr로 admission/preflight에서 종료했다. `attempt-cancel/opening.pgn`은 준비 artifact일 뿐 spawn 성공이나 실제 취소 drain을 증명하지 않는다. cancel ProcessReceipt·runner stdout/stderr·PGN·ledger는 존재하지 않으므로 만들지 않았다. 이 checkpoint는 실제 취소 동작 성공 evidence가 아니다. 도구 transcript의 traceback은 원본 artifact file이 아니므로 raw log로 재구성하지 않았다.

checkpoint 2는 clean-source guard 뒤 비어 있는 directory만 남아 실행 artifact를 복사하지 않았다. 이 상태는 부모 agent의 보고에 근거하며 별도의 원본 guard traceback 파일은 없다. 이후 checkpoint 3은 다른 source의 별도 실행이므로 이 snapshot에 합치지 않는다.

보존 환경 경로는 placeholder로 치환했고 Inventory가 원본·보존본 SHA/bytes와 category/count를 기록한다. 치환된 로그/receipt에서 원래 ArtifactRef SHA는 원본 bytes의 identity이므로 저장된 redacted copy의 SHA와 같다고 주장하지 않는다. 입력 lock/plan/ledger의 payload와 선언된 hash 값을 새로 쓰지 않았다. 바이너리 두 개는 제외했고 기존 binary SHA/size 메타데이터만 유지한다. 원본을 삭제하지 않았다. `derived/cases.tsv`와 재현 문서는 보존 후 생성한 것으로 원본 raw output과 구분한다.
