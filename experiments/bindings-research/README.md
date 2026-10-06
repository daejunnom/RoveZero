# CPU/mock Rust–Python 접점 실험

총괄 I02가 소유하는 연구 경로다. 제품 workspace에는 PyO3 의존성을 넣지 않는다.
Rust 1.96.0, CPython 3.12.3, PyO3 0.27.2, maturin 1.15.0을 사용한다.
`core::search_once`를 직접 호출·지속 JSONL CLI·Python extension이 함께 사용한다.
`position`은 기존 UCI 파서와 자체 Rules, 탐색은 기존 공통 계약과 CPU mock을 사용한다.
Python callback은 없으며 호출 입력을 Rust가 소유한 뒤 `Python::detach`로 실행한다.

`check_paths.py`는 세 경로의 착수·종료 상태·완료 simulation·소비 평가 일치와
반복·동시 호출 격리를 검사한다. 측정과 구별해 에이전트/CI에서 실행할 수 있다.
`benchmark.py`는 사용자가 실행한다. 경로별 cold process 3회, 다섯 포지션별 warm 20회,
CPU 2개·RAM 512MiB·전체 120초를 외부 cgroup supervisor에서 적용한다.
측정 중 새 패키지 설치나 빌드를 하지 않는다. 로그·wheel·target·보고서는 Git 밖에 둔다.

공통 함수는 이미 제공된 `engine::diagnostics::run`을 호출한다. 이 경로의 1ms 대기와
관측 비용도 Rust body 시간에 포함된다. 이것은 native UCI worker의 속도 측정이 아니다.
Rust 직접 경로는 함수 호출을 재며 JSON 출력 sink는 제외한다. CLI 경로는 요청 직렬화·
IPC·응답 파싱을 포함하고, binding 경로는 GIL 분리·결과 dict 생성까지 포함한다.
새 game/root/evaluator를 호출마다 구성하므로 호출 간 탐색 상태를 재사용하지 않는다.
128은 이 CPU mock 실험의 상한이다. BT4 대국의 4096 simulation과 혼동하지 않는다.

호출 오버헤드 감소는 기존 Rust 탐색 본체의 개선이나 대국 성과를 입증하지 않는다.
이번 실험의 결과로 Python 내장 엔진을 채택하지 않는다.
