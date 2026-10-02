# D01 평가 런타임

담당은 D, 작업은 TASK-D01이다. 현재 기반은 문서 커밋 `9f0bc59`이며
총괄 TASK-I01의 `rz-contracts` Rust 선언과 root Cargo workspace는 아직 없다.
이 crate의 adapter는 총괄이 게시할 RequestId/ExecutionId, 요청·결과·오류 타입을
associated type으로 받는다. 공통 ID·WDL·EvalResult·계약 revision을 복제하지 않는다.

Scheduler의 책임은 유한 FIFO queue, 호환 batch, 논리 종결과 물리 실행의 수명이다.
B는 selection/reservation/backup을 소유하고 C는 실제 evaluator와 opaque Lease를 제공한다.
입력 metadata와 raw 평가의 identity는 adapter가 그대로 보존한다. 최초 기준은
cache/dedup/speculation/warm-start 없이 fresh 요청을 실행한다.

현재 standalone `[workspace]`는 root 파일을 수정하지 않고 D 소유 crate를 검사하기
위한 임시 구성이다. I01 workspace 통합 시 이 선언을 제거하고 공통 계약 adapter,
root dependency/feature/toolchain/CI는 총괄의 integration 변경으로 연결한다.

진행 상태와 실제 검사 명령·미실행 범위는 이 문서와 같은 목표의 draft PR에 기록한다.
