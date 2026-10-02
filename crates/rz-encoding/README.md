# rz-encoding — Maia 모델 표현

담당: TASK-C02. [선정 기록](../../docs/WEIGHT-SELECTION.md)의 classical 112-plane,
1858-action 형식만 구현한다. Rules의 FEN parser·합법 수 생성·종료 판정을 소유하지 않는다.
I01/A03의 실제 공통 계약이 나오면 checked view → 모델 표현 adapter를 추가한다.

## Policy

`policy::index`의 입력은 현재 차례 관점의 model-space 좌표다. a1=0이며 흑 차례는
`canonical_square`로 rank만 반전한다. 일반 슬롯은 from/to 순서로 queen ray와 knight
기하를 열거해 1792개를 생성한다. 이후 7→8 rank의 목적지를 from file·to file·Q/R/B
순서로 열거해 66개를 추가한다. 외부 GPL 매핑 테이블을 복사하거나 vendoring하지 않는다.

- Knight 승격은 일반 from/to 슬롯, Q/R/B는 별도 슬롯을 쓴다.
- `castling_index`는 표준 체스의 king→rook 표현을 사용한다. king→착수 목적지와
  혼동하지 않는다. A의 checked move가 캐슬링임을 명시한 뒤 변환한다.
- 앙파상은 실제 from/to에 대응하며 합법성·포획 기물 위치는 Rules 책임이다.
- `slot`은 모델 슬롯을 반환한다. 일반 슬롯에서 knight 승격인지 알아내려면 원래
  checked legal move가 필요하므로 이 함수를 완전한 Rules move 복원으로 쓰지 않는다.
- 형식상 표현 가능한 수가 합법 수라는 뜻은 아니다. 합법 수 순서·승격·orderIdentity는
  A03의 view에서 보존하고 evaluator에서 같은 순서로 policy를 모은다.

## Classical 입력

`classical::Frame`은 절대 좌표의 색별 P/N/B/R/Q/K bitboard, 반복 plane 값, 원본 EP
target을 담는 **모델 projection**이다. mutable Position이나 새로운 Rules 상태가 아니다.
`Input`은 현재부터 과거 순서로 최대 8개 frame과 현재 차례·캐슬링 권리·halfmove를 받는다.
모든 과거 frame도 현재 요청의 차례 관점으로 변환한다.

104~107 plane은 자기/상대 queenside·kingside, 108은 실제 흑 차례, 109는 원본
halfmove counter, 110은 0, 111은 1이다. `/100` 또는 추가 canonical transform은 없다.

history-fill은 호출자가 반드시 명시한다.

- `No`: 참조 옵션 `no`. 알려진 이력 뒤는 0이며, 마지막 알려진 frame의 EP target으로
  알 수 있는 double pawn push 이전 모델 frame을 하나만 보충한다.
- `RepeatOldest`: 참조 옵션 `always`. 남은 슬롯에 가장 오래된 frame 또는 그 EP
  이전 모델 frame을 반복한다. `fen_only`는 현재 지원하지 않는다.

입력은 변경하지 않는다. 출력의 known/padded frame 수와 EP 추론 여부를 구분하며,
모델 padding을 실제 대국 이력이나 반복 근거로 되돌려 쓰지 않는다. 공통 adapter는
Rules의 `unknown_prefix`와 state/legal identity를 그대로 보존해야 한다. cache key에는
실제 입력·history-fill·인코딩/action map 버전이 포함되어야 한다.

## 독립 대조와 검사

형식 참조는 LC0 v0.32.1 commit `fd71a2d921b689c5f479d3227c3806c8e272d9c5`의
[encoder](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/neural/encoder.cc)다.
고정 참조의 action 이름과 `policy_map` 예제가 출력하는 생성 결과를 별도 작업 경로에서
1858개 모두 대조했다. 차이는 0개다. 참조 소스 SHA-256은
`f8b1fc47e5f7ab85c9d38a21d1b03f09dba98a5c6760f61f8ebacacf5afbdfae`다.
이는 **정적 action 형식 대조**이며 원본 가중치 추론 parity가 아니다.

영구 검사는 작은 참조 인덱스·승격 네 종류·캐슬링·흑 좌표·잘못된 형식과,
손으로 구성한 전체 tensor·같은 보드/다른 이력·model fill·EP 이전 상태를 확인한다.
전체 GPL table을 fixture로 넣지 않는다. 실제 LC0 encoder 실행·Maia 원본/변환본의
수치 대조와 A03 checked view 통합은 아직 미실행이다.

```sh
cargo test --manifest-path crates/rz-encoding/Cargo.toml
cargo clippy --manifest-path crates/rz-encoding/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path crates/rz-encoding/Cargo.toml -- --check
cargo run --quiet --manifest-path crates/rz-encoding/Cargo.toml --example policy_map
```

외부 dependency와 중첩 workspace 선언은 없다. standalone lockfile은 추적하지 않으며,
root workspace·공통 dependency·toolchain/lockfile/CI는 총괄 I01이 소유한다.
