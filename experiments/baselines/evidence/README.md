# E 검증 자료의 원격 보존

E01/E02와 외부 runner 개발의 작은 검증 자료를 E 소유 경로와 기존 draft PR #8에
보존한다. 입력·잠금·계획·각 단계 원장·결과·실행 영수증·PGN/config·TSV·원본 검사
stdout/stderr·실행 명령을 포함한다. D/F 자료는 각각 담당의 소유 범위다.

공개 사본은 453개 파일, 약 1.8 MB이며 최대 파일도 약 255 KB다.
448개 자료 파일은 아래 inventory에 개별 hash를 기록하고, inventory 4개와 이 안내문을
함께 보존한다. 전체 remote Git blob을 공개 사본의 SHA-256과 대조해 확인한다.

| 묶음 | 파일 목록·SHA-256·source·누락·마스킹 |
|---|---|
| E02 초기 내부 회계, E01 문맥 | [2026-10-02-e02/INVENTORY.json](2026-10-02-e02/INVENTORY.json) |
| 외부 runner 조사·합성 UCI smoke | [2026-10-02-runner-research/INVENTORY.json](2026-10-02-runner-research/INVENTORY.json) |
| 초기 실행 checkpoint | [2026-10-02-runner-checkpoint-1/INVENTORY.json](2026-10-02-runner-checkpoint-1/INVENTORY.json) |
| 현재 실행·실패 checkpoint와 최종 검사 | [2026-10-02-runner-execution/INVENTORY.json](2026-10-02-runner-execution/INVENTORY.json) |

모든 실행 엔진/결과는 합성 fixture다. 실제 프로세스·Rust 검사를 실행한 기록과
합성 사건 선언의 회계를 구분한다. E02 초기 6 pair의 W=D=L=4는 실제 체스 12판의
성적이 아니다. 현재 실제 정상 두 판은 script checkmate로 1승·1패, cutoff 두 판은
점수 없이 Incomplete다. 취소는 runner 종료를 관측했지만 자식 좀비 때문에 cleanup
Unverified를 보존하고 pair를 제외했다. 실패한 재현 driver와 빠진 영수증도 숨기지 않는다.

원본은 작업 환경에 유지한다. 공개 사본은 secret/token/key/email·개인/환경 경로·
host/account 신호를 검사하고 필요한 경로를 마스킹했다. inventory의 원본 SHA와
saved SHA를 구분한다. 마스킹한 로그의 저장 SHA를 원본 실행 영수증의 artifact SHA와
같다고 주장하지 않는다. 누락한 원본과 나중에 생성한 요약/재현 절차도 각각 표시한다.

빌드 cache·외부 바이너리·미확인 가중치/데이터는 포함하지 않는다. 원본 범위의
바이너리 사본 약 30 MB는 사전 용량 검토 후 제외했으며 source/build/license/hash
identity와 재빌드 절차만 남긴다. 공개 text는 작은 파일로 한정한다. 향후 큰 자료는
Git LFS 또는 별도 object/artifact storage의 용량·보존기간·접근권한·SHA index를
먼저 보고하고 보존 경로를 정한다. 이 checkpoint에서는 별도 업로드를 만들지 않았다.

재현 절차는 각 묶음의 REPRODUCE/README와
[run-e-fixture.py](../scripts/run-e-fixture.py), [check-e.py](../scripts/check-e.py)에 있다.
출력은 checkout 밖의 `${ARTIFACT_ROOT}`에 만들고 기존 기록을 덮어쓰지 않는다.
Fastchess는 source `f618e34540f94f4719ad3817950618dabe441318`, MIT·ZLIB disabled
도구를 별도 준비한다. binary·weight를 이 evidence에서 다운로드할 수는 없다.

검증한 Rust source는 `9e49b6679aeab4e96be0f6b90e51e26a0c7a9ead`다. 로컬 테스트
140개, 두 crate fmt/clippy, E01 1.85/arena 1.90 MSRV 검사가 통과했다. shared 계약
source `67284c4f66f7a7ae9f46fa63dfd50e7410eb6845`와 A Rules
`118dc0311a88e143be285703940321dc16261f6a`를 소비한다. 정식 시계·GPU 공정성·
runtime drain·실제 NN·강도·resume·E03 통계는 미검증이고 readiness는 false다.
