# E02 외부 runner 실제 인터페이스 조사

Fastchess source pin은 `f618e34540f94f4719ad3817950618dabe441318` (v1.8.2-alpha)이다. 저장소 밖에서만 clone/build했고 RoveZero에 외부 소스나 바이너리를 복사하지 않았다. 로컬 binary: `${ARTIFACT_ROOT}/tools/fastchess/fastchess`; SHA-256 `32f16e6bdb88605b38c3956dd5c379957eb33ee9fdf9afe2e5b47e33024eb997`, 2,470,064 bytes. 출력 version: `fastchess alpha 1.8.2 20260726-f618e34`. g++ 14.2.0, `make -j 3`, 기본 `-O3 -std=c++17 -DNDEBUG -march=native`; CPU 전용. 다른 CPU에서 binary 호환성은 별도 확인해야 한다.

소스 본체 MIT, 기본 빌드의 Pyrrhic/fmt/chess-library는 MIT, opening classification data는 CC0. ZLIB는 활성화하지 않았다. ZLIB 활성화 빌드는 gzstream LGPL을 포함하므로 별도 권리 기록이 필요하다. Cute Chess source/doc pin은 `5e84232be4546aaedc9d87a96c91867a1da06ada`, GPLv3+, Qt 6.8/CMake 필요; 현재 환경에서 빌드하지 않았다.

## 실제 확인한 Fastchess 명령

`-engine cmd=ABSOLUTE_PATH name=ENGINE_ID option.Seed=N` 두 그룹 뒤 `-each proto=uci st=0.1 restart=on timemargin=0 option.Threads=1 option.Hash=1 option.Ponder=false`, `-openings file=PREFIX.pgn format=pgn order=sequential plies=2 start=1`, `-rounds 1 -games 2 -repeat -concurrency 1 -srand 101`, `-maxmoves 2`, `-pgnout file=RESULT.pgn append=false notation=uci min=false timeleft=true latency=true`, `-log file=RAW.log level=trace engine=true append=false`, `-autosaveinterval 0 -ratinginterval 0`, `-startup-ms 1000 -ucinewgame-ms 1000 -ping-ms 1000 -strict`가 실제 실행 성공했다.

synthetic UCI script가 2 games를 수행했고, stdout은 left/right와 right/left의 색 교환을 보여 준다. 두 게임 첫 go 전 raw log는 모두 `position startpos moves e2e4 e7e5`이다. `go movetime 100`, 엔진 별 Seed 7/11, `ucinewgame`, `stop`, `quit`도 로그에 있다. 이 결과는 프로세스 및 인터페이스 검사이며 실제 엔진 성능/학습/GPU 증거가 아니다. Fastchess가 fixture 두 무승부에서 출력한 -nan LOS/nElo 등 통계는 RoveZero 결과로 사용하지 않는다.

- Fastchess `-repeat`는 games=2 설정의 alias이다. round 1이 하나의 opening을 2회 사용한다. 색 교환은 기본이며 `-noswap`은 금지해야 한다. `-reverse`는 첫 색 배치 역전 선택에 쓸 수 있다.
- EPD는 board만 갖고 prefix history가 없다. 시작 FEN+전체 moves를 보존하려면 FEN/SetUp tags와 합법 SAN prefix의 PGN 한 게임을 생성해야 한다. `plies=N`은 exact prefix length; PGN output `notation=uci`는 결과 형식이며 PGN opening input parser는 SAN을 읽는다.
- openings malformed/terminal prefix는 reader가 early_stop하고 prefix를 줄일 수 있다. 결과로 full moves 전달을 보장하려면 독립 replay와 실제 raw log의 `position`을 대조해야 한다.
- `-srand`는 opening randomization용으로 엔진 RNG seed가 아니다. Seed option은 실제 엔진이 선언한 정확한 option name/type을 확인한 경우에만 쓸 수 있다. unknown options는 warning이고 `-strict`로 실패시킬 수 있다.
- `-maxmoves 2`는 book prefix를 제외한 searched 4plies를 센다. 실제 PGN은 book 2plies+searched 4plies=총6plies였다. manifest max_plies를 변환할 때 이 차이를 확인해야 한다. adjudication 결과를 incomplete limit failure로 쓰는 RoveZero 정책과 구분해야 한다.
- draw/resign score adjudication은 해당 -draw/-resign 인수가 없으면 기본 disabled이다. -maxmoves는 draw adjudication을 활성화한다. tablebase 인수 없음은 tablebase adjudication 없음. -recover는 이 smoke에서 사용하지 않았다.

## 공식 실험에 대해 아직 성립하지 않는 것

`app/src/matchmaking/match/match.cpp` 465-480: `go` write 후 `setupReadEngine()` 이후 `steady_clock t0`를 잡고 bestmove line read 종료까지 시간을 측정한다. locked manifest의 정확한 go_write_to_valid_bestmove_read 경계와 다르고 bestmove validation도 측정 이후이다. `game/timecontrol/timecontrol.cpp`에서 초기 remaining=base+increment, 내부 read timeout margin 100ms이다. 실제 T1 `tc=0.010+0.005` 로그 첫 go는 `wtime 15 btime 15 winc 5 binc 5`였다. 따라서 이 runner 버전의 결과는 계약 clock 준수 증거가 아니다.

자동 3fold/50move 및 insufficient-material 판정은 Fastchess/chess-library 자체 프로필이다. 별도 FIDE claim/fivefold/75move/dead-position 완전성 옵션은 문서에서 확인되지 않는다. 공정한 full-state/claim/clock 계약은 RoveZero A/E adapter가 별도 대조해야 한다. UCI setoption에 ack가 없으므로 로그는 요청전달 증거이고 backend/precision/device/실제로 적용된 옵션 증명은 엔진 출력/observation으로 추가 확인해야 한다. GPU 모델/weights/실제 대국은 실행하지 않았다.

## Cute Chess 문서 비교

동일 prefix PGN, `-games 2 -rounds 1 -repeat`, 기본 색교환을 지원하지만 `-repeat [n]`가 독립 반복숫자를 받고 opening `policy=default|encounter|round`도 가진다. Fastchess는 policy=round만 지원하는 parser와 repeat=games2 alias이므로 Cute Chess 명령을 그대로 전용하면 안 된다. Cute Chess `-pgnout file [min] [fi]`와 Fastchess key=value 출력 인수도 다르다. Cute Chess는 문서만 읽었고 실행 검증하지 않았다.

## 재검토 가능한 evidence

`fastchess-build-and-smoke.json`, `fastchess-help.txt`, `prefix-opening.pgn`, `synthetic-uci.py`, `smoke-pair.*`, `smoke-t1-argv.json`, `smoke-t1-pair.*`, `cutechess-readme.md`, `cutechess-cli-man.txt`를 이 디렉터리에 보존했다. source 링크는 https://github.com/Disservin/fastchess/blob/f618e34540f94f4719ad3817950618dabe441318/man.md 및 source 동일 SHA를 기준으로 한다.

## bounded stdout에서 원본 UCI 로그 확보

Pinned `-debug`는 지원하지 않으며 parseDebug()가 명시적으로 예외를 던진다. Linux에서 `-log file=/proc/self/fd/1 append=true level=trace realtime=true engine=true`를 사용하면 logger를 supervisor가 이미 캡처하는 동일 stdout pipe로 보낼 수 있다. 실제 subprocess pipe smoke: exit 0, stdout 31,611 bytes, stderr 0; setoption/ucinewgame/full position/uciok/readyok/bestmove/quit를 모두 확보했다. 별도 unwatched log artifact를 생성하지 않는다. 관련 증거 `smoke-stdout-log.stdout`, `.stderr`, `.argv.json`, `.pgn`. 일반 runner report와 logger는 서로 다른 mutex를 쓰므로 text interleave 가능성이 있으며 raw bytes를 보존해야 한다. 로그는 옵션 요청전달 증거이고 applied backend/options attestation은 여전히 별도다.
