"""One literal source review for NEW synthetic banks, never product enrollment.

The fourteen byte pins were reviewed at the fixed source revision below.
Reading the checkout only confirms those literal pins; it cannot register a
successor. The compiled fact is deliberately synthetic caller metadata. No
compiler, native process, Rules replay, training or GPU execution is certified.
Historical receipts and the production review registry remain unchanged.
"""
import copy
from contextlib import contextmanager
import hashlib
from unittest.mock import patch

from rz_pals_model import native_source_review as reviewed


REVIEWED_SOURCE_COMMIT = "6ca5946bceb9394d27575b8c3dc7df2ed9a48cb2"
REVIEWED_COMPONENT_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 422725, "sha256": "a86c61bc22dea02d505bc6526d482acc7dc197433b46229d6a3deed5d605e5ed"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 234273, "sha256": "83d54753084c62ecc14dc1f6c7b33e7f93f0d1cd4e2a5e5ce5b27a8a86746cbb"},
    "crates/rz-search/src/pals/engine/semantic_state.rs": {
        "bytes": 34003, "sha256": "aecd3584f37b4e4ea1105d6f60a977719e040cad0adfa832801af1089c22812c"},
    "crates/rz-search/src/pals/engine/researcher_admission.rs": {
        "bytes": 5957, "sha256": "5a1838c122e0e7eddd13b501f73cc94d9ea668333c70c1047111e9c137d3fdcd"},
    "crates/rz-search/src/pals/engine/researcher_profile.rs": {
        "bytes": 5443, "sha256": "118239c55485dcc88f4d82e29f60e68f47d3c4ba1e95b717368f41fab99ee810"},
    "crates/rz-search/src/pals/engine/review_lifecycle.rs": {
        "bytes": 37179, "sha256": "4a2059282d54f92ef05fda27d08f6383c9f6ef95d2f944b62c889217d3a8dc72"},
    "crates/rz-search/src/pals/engine/cpu_lifecycle.rs": {
        "bytes": 41911, "sha256": "2ad824a55e1f9d4b4215bf3f56663827955ac4f6efec37e5a695f0c54866fe0f"},
    "crates/rz-search/src/pals/engine/followup_execution.rs": {
        "bytes": 42122, "sha256": "72e85351261631f7bf57e95a53f95d8c8a6a079e3a320f27efcbefec2e1f938a"},
    "crates/rz-search/src/pals/engine/archive_lifecycle.rs": {
        "bytes": 154741, "sha256": "39120ddc5b9173d019e47f2f4757eda8a33676d2961b1120cd5395a1d9adb69c"},
    "crates/rz-arena/src/pals_collect.rs": {
        "bytes": 182486, "sha256": "608ed3c18d4a25e03e686c060a3c332a85443f8370406bfd9ed8f5e963b9e2e6"},
    "crates/rz-arena/src/pals_collect/full_line.rs": {
        "bytes": 18672, "sha256": "6f79dd8e05add10c258ebe873fd3db2642e39727e616ebf4e256d0c7f3b4ce30"},
    "crates/rz-arena/src/pals_collect/divergence.rs": {
        "bytes": 4980, "sha256": "f2916c4128767ffc0fe06407bc39024e2b6c2ec2e1eb952faeef998d7d7b1efd"},
    "crates/rz-arena/src/pals_collect/continuation.rs": {
        "bytes": 29927, "sha256": "22ce51044c1a4d3b61b3e62f9eae23198a004611e30ef14b20825ea5206f3963"},
    "crates/rz-arena/src/pals_collect/producer.rs": {
        "bytes": 21824, "sha256": "d72fad02feee033ce223f8d8608bd4847e6adc954232b82dd0c3abfd2065a304"},
}
REVIEWED_TRANSITION_PINS = {path: copy.deepcopy(REVIEWED_COMPONENT_PINS[path]) for path in (
    "crates/rz-search/src/pals/engine.rs", "crates/rz-arena/src/pals_collect/native.rs")}
REVIEWED_CLOSURE_SHA256 = "5a33f14ad40b1f8cec9bf5482abaa5944fbd8836c00c5d7abd808222f0d14903"
# SHA-256 of the fixed text "synthetic independently declared compiled closure
# fact\\n". This names no actual Rust compiler artifact or product binary.
SYNTHETIC_COMPILED_FACT = "f0cb4112737e170c89be3ba59c7da9dcaf893b15829f749238c5a8f5e5decb8c"


def fixture_profile(lane):
    if lane not in ("default_disabled", "legacy_single_reply"):
        raise ValueError("explicit synthetic review lane required")
    return {"schema": "rz-pals-native-reviewed-transition-profile/1",
        "review_id": "synthetic-reviewed-6ca5946-fourteen-source-" + lane,
        "lane": lane, "transition_sources": copy.deepcopy(REVIEWED_TRANSITION_PINS),
        "component_pins": copy.deepcopy(REVIEWED_COMPONENT_PINS),
        "compiled_search_implementation_sha256": SYNTHETIC_COMPILED_FACT,
        "closure_sha256": REVIEWED_CLOSURE_SHA256}


@contextmanager
def fixture_review_scope(lane):
    """Only this test owns the temporary registry; no retained enrollment."""
    profile = fixture_profile(lane)
    with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (profile,)):
        yield profile


def read_reviewed_fixture_sources(repo):
    """Observe literal reviewed bytes; never derive a new allowed source pin."""
    sources = {}
    for path, expected in REVIEWED_COMPONENT_PINS.items():
        with (repo / path).open("rb") as stream:
            actual = stream.read(expected["bytes"] + 1)
        if {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()} != expected:
            raise AssertionError("reviewed Rust fixture source pin changed: " + path)
        sources[path] = actual
    return sources


def fixture_manifest():
    # Keep the transition pair first for byte-binding mutation cases. Ordering
    # does not replace the consumer's complete fourteen-path map validation.
    return {"source_commit": REVIEWED_SOURCE_COMMIT,
        "files": [{"path": path, **copy.deepcopy(pin)} for path, pin in REVIEWED_COMPONENT_PINS.items()]}


# Separate reviewed synthetic successor. The original 6ca5946 constants and
# fixture APIs above remain unchanged. The reviewed fixed Git/LF source
# diff: engine.rs adds only the four-line intervention.rs include_str entry to
# compiled_search_implementation_sha256; all other thirteen reviewed files are
# byte-identical. This is a source-inventory change, not an observed execution
# or permission to use either legacy lane for Rovent intervention work.
SUCCESSOR_REVIEWED_SOURCE_COMMIT = "932d7ae7006803dbb655249f7fa10e13483bdc12"
SUCCESSOR_REVIEWED_COMPONENT_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 422913, "sha256": "2729ef10909bdf21182d26b2edc4555f5cad7c17f55ac322111e7d867d8cbf53"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 234273, "sha256": "83d54753084c62ecc14dc1f6c7b33e7f93f0d1cd4e2a5e5ce5b27a8a86746cbb"},
    "crates/rz-search/src/pals/engine/semantic_state.rs": {
        "bytes": 34003, "sha256": "aecd3584f37b4e4ea1105d6f60a977719e040cad0adfa832801af1089c22812c"},
    "crates/rz-search/src/pals/engine/researcher_admission.rs": {
        "bytes": 5957, "sha256": "5a1838c122e0e7eddd13b501f73cc94d9ea668333c70c1047111e9c137d3fdcd"},
    "crates/rz-search/src/pals/engine/researcher_profile.rs": {
        "bytes": 5443, "sha256": "118239c55485dcc88f4d82e29f60e68f47d3c4ba1e95b717368f41fab99ee810"},
    "crates/rz-search/src/pals/engine/review_lifecycle.rs": {
        "bytes": 37179, "sha256": "4a2059282d54f92ef05fda27d08f6383c9f6ef95d2f944b62c889217d3a8dc72"},
    "crates/rz-search/src/pals/engine/cpu_lifecycle.rs": {
        "bytes": 41911, "sha256": "2ad824a55e1f9d4b4215bf3f56663827955ac4f6efec37e5a695f0c54866fe0f"},
    "crates/rz-search/src/pals/engine/followup_execution.rs": {
        "bytes": 42122, "sha256": "72e85351261631f7bf57e95a53f95d8c8a6a079e3a320f27efcbefec2e1f938a"},
    "crates/rz-search/src/pals/engine/archive_lifecycle.rs": {
        "bytes": 154741, "sha256": "39120ddc5b9173d019e47f2f4757eda8a33676d2961b1120cd5395a1d9adb69c"},
    "crates/rz-arena/src/pals_collect.rs": {
        "bytes": 182486, "sha256": "608ed3c18d4a25e03e686c060a3c332a85443f8370406bfd9ed8f5e963b9e2e6"},
    "crates/rz-arena/src/pals_collect/full_line.rs": {
        "bytes": 18672, "sha256": "6f79dd8e05add10c258ebe873fd3db2642e39727e616ebf4e256d0c7f3b4ce30"},
    "crates/rz-arena/src/pals_collect/divergence.rs": {
        "bytes": 4980, "sha256": "f2916c4128767ffc0fe06407bc39024e2b6c2ec2e1eb952faeef998d7d7b1efd"},
    "crates/rz-arena/src/pals_collect/continuation.rs": {
        "bytes": 29927, "sha256": "22ce51044c1a4d3b61b3e62f9eae23198a004611e30ef14b20825ea5206f3963"},
    "crates/rz-arena/src/pals_collect/producer.rs": {
        "bytes": 21824, "sha256": "d72fad02feee033ce223f8d8608bd4847e6adc954232b82dd0c3abfd2065a304"},
}
SUCCESSOR_REVIEWED_TRANSITION_PINS = {path: copy.deepcopy(SUCCESSOR_REVIEWED_COMPONENT_PINS[path]) for path in (
    "crates/rz-search/src/pals/engine.rs", "crates/rz-arena/src/pals_collect/native.rs")}
SUCCESSOR_REVIEWED_CLOSURE_SHA256 = "1af475f9fb1f82557ef5ad0d02a405f55118946be5063c2c78ec2d4c6b90c61c"
# SHA-256 of this independent fixture label, not an actual compiled artifact:
# "synthetic independently declared compiled closure fact for reviewed successor 932d7ae\n".
SUCCESSOR_SYNTHETIC_COMPILED_FACT = "f566cc430f184c29fdefbdce86fbd1663c63e89f05af61616f7de4fcd09789d3"


def successor_fixture_profile(lane):
    if lane not in ("default_disabled", "legacy_single_reply"):
        raise ValueError("explicit synthetic successor review lane required")
    return {"schema": "rz-pals-native-reviewed-transition-profile/1",
        "review_id": "synthetic-reviewed-932d7ae-fourteen-source-" + lane,
        "lane": lane, "transition_sources": copy.deepcopy(SUCCESSOR_REVIEWED_TRANSITION_PINS),
        "component_pins": copy.deepcopy(SUCCESSOR_REVIEWED_COMPONENT_PINS),
        "compiled_search_implementation_sha256": SUCCESSOR_SYNTHETIC_COMPILED_FACT,
        "closure_sha256": SUCCESSOR_REVIEWED_CLOSURE_SHA256}


@contextmanager
def successor_fixture_review_scope(lane):
    """Explicit test-local successor only; restore the original registry."""
    profile = successor_fixture_profile(lane)
    with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (profile,)):
        yield profile


def read_successor_fixture_sources(repo):
    """Compare raw bytes to fixed reviewed LF pins; never normalize or enroll."""
    sources = {}
    for path, expected in SUCCESSOR_REVIEWED_COMPONENT_PINS.items():
        with (repo / path).open("rb") as stream:
            actual = stream.read(expected["bytes"] + 1)
        if {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()} != expected:
            raise AssertionError("reviewed successor Rust fixture source pin changed: " + path)
        sources[path] = actual
    return sources


def successor_fixture_manifest():
    return {"source_commit": SUCCESSOR_REVIEWED_SOURCE_COMMIT,
        "files": [{"path": path, **copy.deepcopy(pin)} for path, pin in SUCCESSOR_REVIEWED_COMPONENT_PINS.items()]}


# Separate PC-shared synthetic fixture, fixed to reviewed Git/LF bytes. Against
# 932d7ae, the fourteen-path scope changes only engine.rs: Root added the eight
# compiled-source inventory lines for shared_nnuec.rs and nnuec_v2_shared_readout.rs.
# The other thirteen files are unchanged. This table certifies no helper run,
# shared readout semantics, compiled binary, native retirement or target. Those
# remain separate actual checks; neither earlier fixture bank is rewritten.
PC_SHARED_REVIEWED_SOURCE_COMMIT = "47b602087de810dbe77c1489a7459202de3dfc9f"
PC_SHARED_REVIEWED_COMPONENT_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 423219, "sha256": "6f89838c7e95ff067bac86432c36ee221cf29acc334fe16d0f4f2b96c2d825d7"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 234273, "sha256": "83d54753084c62ecc14dc1f6c7b33e7f93f0d1cd4e2a5e5ce5b27a8a86746cbb"},
    "crates/rz-search/src/pals/engine/semantic_state.rs": {
        "bytes": 34003, "sha256": "aecd3584f37b4e4ea1105d6f60a977719e040cad0adfa832801af1089c22812c"},
    "crates/rz-search/src/pals/engine/researcher_admission.rs": {
        "bytes": 5957, "sha256": "5a1838c122e0e7eddd13b501f73cc94d9ea668333c70c1047111e9c137d3fdcd"},
    "crates/rz-search/src/pals/engine/researcher_profile.rs": {
        "bytes": 5443, "sha256": "118239c55485dcc88f4d82e29f60e68f47d3c4ba1e95b717368f41fab99ee810"},
    "crates/rz-search/src/pals/engine/review_lifecycle.rs": {
        "bytes": 37179, "sha256": "4a2059282d54f92ef05fda27d08f6383c9f6ef95d2f944b62c889217d3a8dc72"},
    "crates/rz-search/src/pals/engine/cpu_lifecycle.rs": {
        "bytes": 41911, "sha256": "2ad824a55e1f9d4b4215bf3f56663827955ac4f6efec37e5a695f0c54866fe0f"},
    "crates/rz-search/src/pals/engine/followup_execution.rs": {
        "bytes": 42122, "sha256": "72e85351261631f7bf57e95a53f95d8c8a6a079e3a320f27efcbefec2e1f938a"},
    "crates/rz-search/src/pals/engine/archive_lifecycle.rs": {
        "bytes": 154741, "sha256": "39120ddc5b9173d019e47f2f4757eda8a33676d2961b1120cd5395a1d9adb69c"},
    "crates/rz-arena/src/pals_collect.rs": {
        "bytes": 182486, "sha256": "608ed3c18d4a25e03e686c060a3c332a85443f8370406bfd9ed8f5e963b9e2e6"},
    "crates/rz-arena/src/pals_collect/full_line.rs": {
        "bytes": 18672, "sha256": "6f79dd8e05add10c258ebe873fd3db2642e39727e616ebf4e256d0c7f3b4ce30"},
    "crates/rz-arena/src/pals_collect/divergence.rs": {
        "bytes": 4980, "sha256": "f2916c4128767ffc0fe06407bc39024e2b6c2ec2e1eb952faeef998d7d7b1efd"},
    "crates/rz-arena/src/pals_collect/continuation.rs": {
        "bytes": 29927, "sha256": "22ce51044c1a4d3b61b3e62f9eae23198a004611e30ef14b20825ea5206f3963"},
    "crates/rz-arena/src/pals_collect/producer.rs": {
        "bytes": 21824, "sha256": "d72fad02feee033ce223f8d8608bd4847e6adc954232b82dd0c3abfd2065a304"},
}
PC_SHARED_REVIEWED_TRANSITION_PINS = {path: copy.deepcopy(PC_SHARED_REVIEWED_COMPONENT_PINS[path]) for path in (
    "crates/rz-search/src/pals/engine.rs", "crates/rz-arena/src/pals_collect/native.rs")}
PC_SHARED_REVIEWED_CLOSURE_SHA256 = "8ec6461fc6ee6cf7f5ab8d9a87f80e9b001286395edf89fd906cef2b78df5c7a"
# SHA-256 of this independent fixture label, not an actual compiled artifact:
# "synthetic independently declared compiled closure fact for reviewed PC shared source 47b6020\n".
PC_SHARED_SYNTHETIC_COMPILED_FACT = "1221b3e57973765549d2e489d995161c116622b24b02780adf2fb56cc3444b97"


def pc_shared_fixture_profile(lane):
    if lane not in ("default_disabled", "legacy_single_reply"):
        raise ValueError("explicit synthetic PC-shared review lane required")
    return {"schema": "rz-pals-native-reviewed-transition-profile/1",
        "review_id": "synthetic-reviewed-47b6020-pc-shared-fourteen-source-" + lane,
        "lane": lane, "transition_sources": copy.deepcopy(PC_SHARED_REVIEWED_TRANSITION_PINS),
        "component_pins": copy.deepcopy(PC_SHARED_REVIEWED_COMPONENT_PINS),
        "compiled_search_implementation_sha256": PC_SHARED_SYNTHETIC_COMPILED_FACT,
        "closure_sha256": PC_SHARED_REVIEWED_CLOSURE_SHA256}


@contextmanager
def pc_shared_fixture_review_scope(lane):
    """Explicit test-local PC-shared bank only; restore the original registry."""
    profile = pc_shared_fixture_profile(lane)
    with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (profile,)):
        yield profile


def read_pc_shared_fixture_sources(repo):
    """Compare raw bytes to the reviewed PC-shared LF pins without normalization."""
    sources = {}
    for path, expected in PC_SHARED_REVIEWED_COMPONENT_PINS.items():
        with (repo / path).open("rb") as stream:
            actual = stream.read(expected["bytes"] + 1)
        if {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()} != expected:
            raise AssertionError("reviewed PC-shared Rust fixture source pin changed: " + path)
        sources[path] = actual
    return sources


def pc_shared_fixture_manifest():
    return {"source_commit": PC_SHARED_REVIEWED_SOURCE_COMMIT,
        "files": [{"path": path, **copy.deepcopy(pin)} for path, pin in PC_SHARED_REVIEWED_COMPONENT_PINS.items()]}
