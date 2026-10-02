# 보존 이후 작성한 재현 명령

이 명령은 원본 실행 log가 아니다. 실제 당시 argv·exit·stdout/stderr는 `original-command-extract.json`과 `reports/cli-receipt.json`, Cargo checks는 `reports/checks.json`이다.

1. 별도 RoveZero checkout을 `e48ff065a431bc6bdd835d75e46e2ed121661f53`에 맞춘다. 해당 checkpoint는 Rust 1.85에서 검사되었다. 현재 수정 중인 checkout에서 과거 검사 결과를 재현했다고 주장하지 않는다.
2. `${REPO_ROOT}`를 해당 checkout, `${ARTIFACT_ROOT}`를 새 외부 출력 root, `${CARGO}`를 사용할 Cargo 경로로 지정한다. `reports/rz-experiments-Cargo.lock`와 `reports/rz-arena-Cargo.lock`를 별도 checkout의 각 crate-local Cargo.lock으로 배치한다.
3. `bash reproduce-e02.sh`를 실행한다. 이 script는 실제 Cargo test/fmt/clippy, CLI lock/plan, event 순차 append와 trusted-tip audit를 수행하는 재현 명령이며 출력은 새 디렉터리에 저장한다. 원본 stderr/stdout split이 남아있지 않은 Cargo 합본 logs를 역으로 복원하지 않는다.
4. E01 원본 명령은 `e01-context/reports/e01-cli-receipt.json`·`e01-acceptance.json`에 보존되어 있다. 해당 E01 source에 맞춘 별도 checkout에서 `${CARGO} test --manifest-path crates/rz-experiments/Cargo.toml --locked`와 receipt argv를 사용한다. E01 참조 artifact root는 `${REPO_ROOT}/experiments/baselines/fixtures`이다.
