# 보존 이후 작성한 재현 명령

원본 실제 build command는 `reports/fastchess-build-and-smoke.json`의 `make -j 3`, 원본 실제 T1 argv는 `reports/smoke-t1-argv.json`이다. 아래 준비·실행 명령은 나중에 작성했으며 이 checkpoint에서 다시 실행하지 않았다.

```bash
: "${ARTIFACT_ROOT:?external root}"
git clone https://github.com/Disservin/fastchess "${ARTIFACT_ROOT}/tools/fastchess"
git -C "${ARTIFACT_ROOT}/tools/fastchess" checkout f618e34540f94f4719ad3817950618dabe441318
make -C "${ARTIFACT_ROOT}/tools/fastchess" -j 3
FASTCHESS="${ARTIFACT_ROOT}/tools/fastchess/fastchess" python3 reproduce-runner.py --phase t1
FASTCHESS="${ARTIFACT_ROOT}/tools/fastchess/fastchess" python3 reproduce-runner.py --phase t2-reconstructed
```

원래 compiler는 g++ (Debian 14.2.0-19) 14.2.0, default flags는 build receipt에 기록된 `-O3 -std=c++17 -Wall -Wextra -DNDEBUG -march=native`였다. ZLIB는 disabled였다. 다른 build/CPU에서 byte-identical binary나 timing을 보장하지 않는다. T1은 원래 저장된 exact argv의 환경 경로만 바꾸며, T2는 T1 vector와 원래 REPORT의 `st=0.1` 설명으로 만든 재구성이다. T2 exact original vector는 missing item이다. 새 raw stdout/stderr/log/PGN/config는 `${ARTIFACT_ROOT}` 아래 새 디렉터리에 생성한다.

Cute Chess는 원래 실행하지 않았으므로 Cute Chess 실행 재현 명령이나 성공 결과를 만들지 않았다. 문서 source는 https://github.com/cutechess/cutechess/tree/5e84232be4546aaedc9d87a96c91867a1da06ada 이다.

추가 실제 stdout-log probe의 exact original argv는 `reports/smoke-stdout-log.argv.json`이다. 다음 명령은 그 원본 vector를 새 외부 출력 root에 맞춰 재실행하는, 보존 후 작성한 명령이다. 이번 보존 작업에서는 실행하지 않았다.

```bash
FASTCHESS="${ARTIFACT_ROOT}/tools/fastchess/fastchess" python3 reproduce-runner.py --phase stdout-log
```
