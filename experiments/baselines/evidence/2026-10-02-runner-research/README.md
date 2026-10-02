# 외부 runner 연구 보존 checkpoint

Fastchess `f618e34540f94f4719ad3817950618dabe441318`(MIT)의 실제 로컬 CPU build/probe와 synthetic UCI fixture 두 pair 실행의 원래 log·stdout·stderr·PGN·config·T1 argv를 보존했다. 당시 binary의 SHA/size는 `reports/fastchess-build-and-smoke.json`에 기록되어 있지만 binary와 외부 checkout은 포함하지 않는다.

`reports/smoke-pair.*`는 `st=0.1` probe, `reports/smoke-t1-pair.*`는 `tc=0.010+0.005` probe의 실제 외부 프로세스 출력이다. 엔진은 `reports/synthetic-uci.py`라는 synthetic fixture이고 playing strength/NN/GPU 결과가 아니다. Fastchess maxmoves에 따른 무승부와 자체 nElo/LOS는 RoveZero 강도·통계 결과로 사용하지 않는다. 정확한 RoveZero clock/claim/full-state 계약을 만족했다는 증거도 아니다. 원래 `reports/REPORT.md`의 한계를 그대로 유지한다.

경로 치환은 `INVENTORY.json`에 파일별 category/count로 기록했고 원본은 삭제하지 않았다. 실제 외부 tool 빌드 stdout/stderr는 원본 디렉터리에 남아있지 않아 build-command/version/compiler/identity receipt만 보존했다. T1 exact argv는 보존되었고 최초 `st=0.1` smoke의 exact argv 파일은 없으므로 아래 재현에서 재구성한 T2 명령을 원본 명령으로 표시하지 않는다.

Cute Chess `5e84232be4546aaedc9d87a96c91867a1da06ada`의 GPLv3+ README/manpage는 외부 문서 참고만 했고 실행하지 않았다. full 문서는 복사하지 않고 Inventory에 원래 source URL·revision·license·제외 사유를 보존했다. Fastchess help output은 MIT tool의 실제 probe 출력으로 보존한다. 다른 도구의 문서 내용을 구현이나 성공 증거로 바꾸지 않는다.

`REPRODUCE.md`와 `reproduce-runner.py`는 보존 이후 작성된 절차이며 아직 실행하지 않았다. 원본 T1 vector에 환경 root를 대체해 실행할 수 있고 출력은 새 외부 디렉터리에 기록한다. 외부 runner build는 저장소 밖에서만 수행한다.

## 추가 stdout-log probe

보존 최초 목록 이후 runner research agent가 `/proc/self/fd/1` stdout trace probe를 추가했다. `reports/smoke-stdout-log.argv.json`과 `.stdout/.stderr/.pgn`은 이 추가 실제 실행의 원본이다. 이후 변경된 `config.json`, `REPORT.md`, build-and-smoke receipt의 이전 보존본은 `reports/pre-stdout-log/`에 그대로 남겼다. stdout trace도 synthetic CPU fixture의 실제 인터페이스 출력이며 NN/GPU/성능 결과가 아니다.
