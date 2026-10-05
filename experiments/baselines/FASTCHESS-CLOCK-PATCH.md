# Fastchess pilot 시계 패치

`fastchess-clock-v1.patch`는 Fastchess commit
`f618e34540f94f4719ad3817950618dabe441318`에만 적용한다. 출처는
[Fastchess](https://github.com/Disservin/fastchess/tree/f618e34540f94f4719ad3817950618dabe441318)이며,
원본 MIT 저작권 고지는 아래에 보존한다. 자체 체스 코어를 대체하지 않는다.

- 첫 시계는 base만 제공한다. 증분은 제시간의 착수 뒤에만 추가한다.
- `position` 전송 전부터 `bestmove` 수신까지 `steady_clock`으로 측정하고,
  부분 밀리초는 올림하여 차감한다. 초기 모델 로딩과 매 판 재시작은 시계 밖의
  동등한 startup 예산이며 전체 pilot 벽시계에는 포함한다.
- POSIX 응답 대기는 단조 절대 마감을 유지한다. 출력이 도착할 때마다 허용 시간을
  새로 주지 않는다. 기존 100ms read margin은 응답 회수용이며 시간패 판정을 연장하지 않는다.
- 부모가 소유한 stdout에 `RZ_CLOCK_V1`을 기록한다. E는 identity·전후 잔여 시간·
  증분·올림 차감·새 게임 초기화를 A가 재생한 PGN과 대조한다. 엔진 `info time`은
  시계 증거로 사용하지 않는다. Ponder·사전 분석·게임 간 탐색 결과 재사용은 허용하지 않는다.

V1/V2 A/A에서는 이 패치를 사용하지 않는다. V3는 source commit, patch SHA/bytes,
컴파일 설정, 수정 binary SHA를 각각 잠근다. 실행 전 실제 patch 적용·시계 fixture를
검증하며 선언을 실제 실행 증거로 보고하지 않는다. native provider·physical drain·
규칙·전체 시계는 각각 별도 gate다. 엔진 오류는 raw PGN의 Loss로 보존하고 pilot을
중단한다. cutoff는 Incomplete이며 득점률에서 무승부로 바꾸지 않는다.

## 원본 저작권 고지

MIT License

Copyright (c) 2023 disservin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
