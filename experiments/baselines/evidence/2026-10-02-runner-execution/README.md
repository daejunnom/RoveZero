# 실제 runner 실행 및 실패 checkpoint 보존

이 snapshot은 기존 실제 CPU 실행에서 나온 입력·input lock·plan·opening·PGN·ProcessReceipt·config·ledger·raw stdout/stderr·command JSON을 checkpoint별로 보존한다. 엔진은 scripted synthetic fixture이며 실제 NN/GPU/playing strength 결과가 아니다. 각 source revision은 `INVENTORY.json`과 원본 reproduction receipt를 기준으로 구분한다.

- checkpoint 1 (`fcbe8b25d53ee96f48ca8dafd227e7c46f8d90ab`): normal/cutoff 실제 실행. cancel은 `process.cancelled_before_spawn` preflight 실패이며 ProcessReceipt가 없다. 준비 opening을 spawn/cancel 성공으로 해석하지 않는다.
- checkpoint 3 (`fdd357f08fdca16666993ebdb8afc3c72c718821`): normal/cutoff 실행과 cancel 시나리오의 실제 process 결과를 보존한다. cancel ProcessReceipt의 stop은 원문을 따른다. 취소 신호 전에 fixture가 종료되면 실제 취소 동작 검증이 되지 않는다. 일부 cancel command/후속 audit/aggregate summary가 driver 중단 뒤 저장되지 않아 missing item이다.
- checkpoint 4 (`9e49b6679aeab4e96be0f6b90e51e26a0c7a9ead`): normal/cutoff 완료. cancel은 실제 `stop=cancelled`, exit signal 15이며 `group_cleanup=unverified`다. 원래 ProcessReceipt와 excluded ledger를 보존하고 cleanup 완료나 GPU drain으로 바꾸지 않는다. outer driver는 cleanup을 Gone으로 잘못 기대한 assertion 때문에 중단했으며 그 실제 stderr를 `driver-reports/`에 그대로 보존했다. 이것은 제품의 정직한 Unverified 결과와 별개다.
- checkpoint 5 (같은 `9e49b667...`): cancel-only 재현 driver가 Unverified를 인정하고 제외·W/D/L=0을 확인했다. normal/cutoff는 이 checkpoint에서 실행하지 않았다.

checkpoint 2에는 empty directory만 남아 제외했다. clean-source guard로 실제 실행하지 않았다는 상태는 부모 agent 보고에 근거한다. 원본 guard traceback 파일은 없어 raw log를 만들지 않았다. checkpoint 1/3의 예전 outer driver 출력이 이후 덮어써져 별도 원문이 남지 않은 경우 현재 driver 로그를 과거 실행 로그로 소급하지 않는다.

`derived/checkpoint-cases.tsv`는 보존 이후 원래 ProcessReceipt와 원래 summary가 존재하는 항목만 추출한 표다. 없는 summary는 `not_in_original_receipt`로 기록한다. 저장된 실패/제외/Unverified를 성공으로 재분류하지 않았다. 별도 `2026-10-02-runner-checkpoint-1/`은 먼저 보존된 같은 checkpoint 1의 중복 역사 snapshot이며 source artifact는 삭제하지 않았다.

환경·tool debug 절대경로는 `${ARTIFACT_ROOT}`, `${REPO_ROOT}`, `${CARGO}`, `${CARGO_HOME}` 등의 placeholder로 치환했다. Inventory가 original/saved SHA·bytes·category/count를 기록한다. redacted raw copy의 SHA를 original receipt가 참조한 unredacted artifact SHA와 같다고 주장하지 않는다. input lock·plan·ledger의 hash 선언은 변경하지 않았다. 원본은 보존 작업에서 삭제하지 않았다. 바이너리8개, cache, 외부 checkout, weights/data, 개인 비밀은 복사하지 않았다.

`REPRODUCE.md`는 나중에 작성한 절차다. 실제 당시 명령은 각 `checkpoint-N/checks/*.command.json` 및 `reproduction-receipt.json`이고 raw stdout/stderr는 같은 stem의 별도 파일이다. 새로 만든 절차를 원본 실행 transcript로 표시하지 않는다. final Cargo checks와 사후 supplemental 기록을 추가로 보존했다.

## 최종 검사와 사후 입력 snapshot

`final-checks/`에는 source `9e49b6679aeab4e96be0f6b90e51e26a0c7a9ead`의 실제 원본 8개 검사 argv/exit/stdout/stderr가 있다. E01 39개+arena 101개=140개 passed이며 arena의 ignored 10개 process helper 함수는 통과 검사로 더하지 않는다. fmt/clippy와 E01 Rust1.85·arena Rust1.90 all-targets 검사도 통과했다. 이 검사는 CPU/synthetic fixtures의 소프트웨어 검사이며 NN/GPU 대국 증거가 아니다.

`Supplemental.json`, rustc version 원문 및 `inputs/end/*.Cargo.lock`는 8개 검사 완료 뒤 실제로 추가 캡처했다. 원래 checks-result.json 원본 bytes/hash가 유지된 것을 확인했으며 원래 8개 검사를 재실행하지 않았다. 시작 Cargo.lock snapshot은 수집하지 않아 시작/종료 동일성을 증명하지 않는다. `derived/final-checks.tsv`는 이후 생성한 표이고 raw stdout/stderr를 대체하지 않는다. metadata의 `original_unmodified_bytes`/원래 SHA는 원본 capture를 설명하며, 저장소의 치환본에는 Inventory의 saved SHA와 redaction 기록을 적용한다.
