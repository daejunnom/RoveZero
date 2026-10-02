"""Deterministic source-group splits and conservative, read-only leakage audits.

Assignments use the SHA256 digest of canonical JSON containing the seed and the
sorted game IDs of a connected component. Its integer value selects a weighted
split slot. Ratios describe probabilities; small corpora may have empty splits.
Schema version 1 fixes this grouping, hash construction and assignment algorithm;
changing any of them requires a new schema version in the digested plan.
An audit never silently removes, reassigns, or repairs a record.
"""

from collections import defaultdict

from .errors import DataError
from .serialization import digest


SPLITS = ("train", "validation", "holdout")
GROUP_KEYS = ("opening_family_id", "lineage_id", "seed_group_id")
SOURCE_KEYS = {"game_id", *GROUP_KEYS}
PLAN_KEYS = {
    "schema_version", "sources", "seed", "ratios", "assignments",
    "source_digest", "digest",
}


def _sources(sources: object) -> list[dict]:
    if not isinstance(sources, list):
        raise DataError("InvalidSources", "sources", "expected a list of source games")
    result = []
    seen = set()
    for index, source in enumerate(sources):
        context = f"sources[{index}]"
        if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
            raise DataError("InvalidSource", context, "source fields must match the source schema")
        game_id = source["game_id"]
        if not isinstance(game_id, str) or not game_id.strip():
            raise DataError("InvalidGameId", context, "game_id must be a nonempty string")
        if game_id in seen:
            raise DataError("DuplicateGameId", context, f"duplicate game_id: {game_id}")
        seen.add(game_id)
        for key in GROUP_KEYS:
            value = source[key]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise DataError("InvalidGroupId", f"{context}.{key}", "expected null or a nonempty string")
        result.append({key: source[key] for key in sorted(SOURCE_KEYS)})
    return sorted(result, key=lambda source: source["game_id"])


def _ratios(ratios: object) -> tuple[int, int, int]:
    if not isinstance(ratios, (list, tuple)) or len(ratios) != len(SPLITS):
        raise DataError("InvalidRatios", "ratios", "expected three positive integer weights")
    if any(type(value) is not int or value <= 0 for value in ratios):
        raise DataError("InvalidRatios", "ratios", "weights must be positive integers; booleans are invalid")
    return tuple(ratios)


def _components(sources: list[dict]) -> list[list[str]]:
    """Union namespaced group IDs with path compression and bounded tree depth."""
    parents = {source["game_id"]: source["game_id"] for source in sources}
    sizes = {game_id: 1 for game_id in parents}

    def find(game_id: str) -> str:
        while parents[game_id] != game_id:
            parents[game_id] = parents[parents[game_id]]
            game_id = parents[game_id]
        return game_id

    first_by_group = {}
    for source in sources:
        game_id = source["game_id"]
        for key in GROUP_KEYS:
            value = source[key]
            if value is None:
                continue
            group = (key, value)
            other = first_by_group.setdefault(group, game_id)
            root, other_root = find(game_id), find(other)
            if root == other_root:
                continue
            if sizes[root] < sizes[other_root]:
                root, other_root = other_root, root
            parents[other_root] = root
            sizes[root] += sizes[other_root]
    groups = defaultdict(list)
    for game_id in sorted(parents):
        groups[find(game_id)].append(game_id)
    return sorted(groups.values())


def make_split_plan(
    sources: list[dict], *, seed: str, ratios: tuple[int, int, int] = (8, 1, 1),
) -> dict:
    """Freeze source identities before producing training rows."""
    canonical_sources = _sources(sources)
    if not isinstance(seed, str):
        raise DataError("InvalidSeed", "seed", "seed must be a string")
    weights = _ratios(ratios)
    assignments = {}
    for component in _components(canonical_sources):
        slot = int(digest({"seed": seed, "component": component}), 16) % sum(weights)
        for split, weight in zip(SPLITS, weights):
            if slot < weight:
                for game_id in component:
                    assignments[game_id] = split
                break
            slot -= weight
    plan = {
        "schema_version": 1,
        "sources": canonical_sources,
        "seed": seed,
        "ratios": list(weights),
        "assignments": dict(sorted(assignments.items())),
        "source_digest": digest(canonical_sources),
    }
    plan["digest"] = digest(plan)
    return plan


def validate_split_plan(plan: dict) -> dict:
    """Recompute all content, including assignments, to reject altered artifacts."""
    if not isinstance(plan, dict) or set(plan) != PLAN_KEYS:
        raise DataError("InvalidSplitPlan", "split_plan", "plan fields must match schema version 1")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise DataError("UnsupportedSchema", "split_plan.schema_version", "only schema version 1 is supported")
    expected = make_split_plan(plan["sources"], seed=plan["seed"], ratios=plan["ratios"])
    if plan != expected:
        raise DataError("SplitPlanTampered", "split_plan", "content or digest differs from the recomputed plan")
    return expected


def audit_leakage(records: list[dict], plan: dict) -> dict:
    """Audit identities and provenance; board equality is a risk, not input equality.

    Records normally pass the record-schema validator first. This layer still
    rejects missing source grouping fields so a caller cannot erase provenance.
    Bucket findings retain record IDs once per key, rather than enumerating pairs.
    """
    plan = validate_split_plan(plan)
    if not isinstance(records, list):
        raise DataError("InvalidRecords", "records", "expected a list of records")
    sources = {source["game_id"]: source for source in plan["sources"]}
    component_by_game = {
        game_id: index
        for index, component in enumerate(_components(plan["sources"]))
        for game_id in component
    }
    errors, warnings, ledger = [], [], []
    by_record = defaultdict(list)
    exact_buckets = defaultdict(list)
    board_buckets = defaultdict(list)
    state_buckets = defaultdict(list)
    group_buckets = {key: defaultdict(list) for key in GROUP_KEYS}
    split_counts = {split: {"records": 0, "games": 0} for split in SPLITS}
    games_by_split = {split: set() for split in SPLITS}
    failure_groups = {split: defaultdict(int) for split in SPLITS}

    def issue(target: list, code: str, context: str, message: str, **detail: object) -> None:
        target.append({"code": code, "context": context, "message": message, **detail})

    for index, record in enumerate(records):
        context = f"records[{index}]"
        if not isinstance(record, dict):
            issue(errors, "InvalidRecord", context, "expected an object")
            continue
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            issue(errors, "InvalidRecordId", context, "record_id must be a nonempty string")
            continue
        by_record[record_id].append(record)
        game_id, split = record.get("game_id"), record.get("split")
        entry = {
            "record_id": record_id, "game_id": game_id, "split": split,
            **{key: record.get(key) for key in GROUP_KEYS},
            "parent_record_id": record.get("parent_record_id"),
            "augmentation_id": record.get("augmentation_id"),
            "failure_groups": record.get("failure_groups", []),
        }
        ledger.append(entry)
        if split not in SPLITS:
            issue(errors, "InvalidSplit", record_id, "unknown split", split=split)
        else:
            split_counts[split]["records"] += 1
            if isinstance(game_id, str):
                games_by_split[split].add(game_id)
        groups = record.get("failure_groups", [])
        if isinstance(groups, list) and all(isinstance(group, str) for group in groups):
            if split in SPLITS:
                for group in set(groups):
                    failure_groups[split][group] += 1
        else:
            issue(errors, "InvalidFailureGroups", record_id, "failure_groups must be a list of strings")
        source = sources.get(game_id) if isinstance(game_id, str) else None
        if source is None:
            issue(errors, "UnknownGame", record_id, "game_id is absent from the split plan", game_id=game_id)
        elif split != plan["assignments"][game_id]:
            issue(errors, "SplitAssignmentMismatch", record_id, "record split differs from the source assignment",
                  expected=plan["assignments"][game_id], actual=split)
        for key in GROUP_KEYS:
            if key not in record:
                issue(errors, "MissingGroupField", record_id, "required provenance field is absent", field=key)
                continue
            value = record[key]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                issue(errors, "InvalidGroupId", record_id, "expected null or a nonempty string", field=key)
                continue
            if source is not None and value != source[key]:
                issue(errors, "GroupMetadataMismatch", record_id, "record grouping differs from the source plan",
                      field=key, expected=source[key], actual=value)
            if value is not None:
                group_buckets[key][value].append(record)
        identity = record.get("input_identity")
        if (isinstance(identity, dict)
                and all(isinstance(identity.get(key), str) for key in ("digest", "encoding_id", "encoding_version"))):
            identity_key = (identity["encoding_id"], identity["encoding_version"], identity["digest"])
            exact_buckets[identity_key].append(record)
        else:
            issue(errors, "InvalidInputIdentity", record_id, "full input digest and encoding identity are required")
        state = record.get("state")
        if isinstance(state, dict):
            for key, buckets in (("board_digest", board_buckets), ("state_digest", state_buckets)):
                value = state.get(key)
                if isinstance(value, str):
                    buckets[value].append(record)

    def bucket_detail(bucket: list[dict]) -> dict:
        return {
            "record_ids": sorted(record["record_id"] for record in bucket),
            "splits": sorted({record["split"] for record in bucket if record.get("split") in SPLITS}),
        }

    def cross_split(bucket: list[dict]) -> bool:
        return len({record.get("split") for record in bucket if record.get("split") in SPLITS}) > 1

    for record_id, bucket in by_record.items():
        if len(bucket) > 1:
            issue(errors, "DuplicateRecordId", record_id, "record identity is ambiguous", count=len(bucket))
    unique_records = {record_id: bucket[0] for record_id, bucket in by_record.items() if len(bucket) == 1}
    parents = {}
    for record_id, record in unique_records.items():
        parent_id = record.get("parent_record_id")
        if record.get("augmentation_id") is not None and parent_id is None:
            issue(errors, "MissingAugmentationParent", record_id, "augmentation must retain its original record relationship")
        if parent_id is None:
            continue
        if not isinstance(parent_id, str) or parent_id not in unique_records:
            issue(errors, "MissingParent", record_id, "parent record is absent or ambiguous", parent_record_id=parent_id)
            continue
        if parent_id == record_id:
            issue(errors, "SelfParent", record_id, "record cannot be its own parent")
        parent = unique_records[parent_id]
        parents[record_id] = parent_id
        if record.get("split") != parent.get("split"):
            issue(errors, "ParentSplitMismatch", record_id, "original and derived records must share a split",
                  parent_record_id=parent_id)
        game_id, parent_game = record.get("game_id"), parent.get("game_id")
        if (isinstance(game_id, str) and isinstance(parent_game, str)
                and game_id in component_by_game and parent_game in component_by_game
                and component_by_game[game_id] != component_by_game[parent_game]):
            issue(errors, "DisconnectedProvenance", record_id,
                  "parent and derived source games lack a connected source group", parent_record_id=parent_id)

    # Iterative traversal avoids recursion limits for long reanalysis chains.
    visited = set()
    for record_id in sorted(unique_records):
        if record_id in visited:
            continue
        path, position = [], {}
        current = record_id
        while current not in visited and current not in position:
            position[current] = len(path)
            path.append(current)
            if current not in parents:
                break
            current = parents[current]
        else:
            if current in position:
                cycle = sorted(path[position[current]:])
                issue(errors, "ParentCycle", cycle[0], "parent provenance contains a cycle", record_ids=cycle)
        visited.update(path)

    for identity, bucket in sorted(exact_buckets.items()):
        if cross_split(bucket):
            issue(errors, "ExactInputAcrossSplits", "input_identity",
                  "the same complete input identity occurs in multiple splits",
                  encoding_id=identity[0], encoding_version=identity[1], input_digest=identity[2],
                  **bucket_detail(bucket))
    for field, buckets in group_buckets.items():
        for group_id, bucket in sorted(buckets.items()):
            if cross_split(bucket):
                issue(errors, "SourceGroupAcrossSplits", field,
                      "connected provenance or opening family spans multiple splits",
                      group_id=group_id, **bucket_detail(bucket))
    for key, buckets in (("board_digest", board_buckets), ("state_digest", state_buckets)):
        for value, bucket in sorted(buckets.items()):
            input_keys = {
                (identity.get("encoding_id"), identity.get("encoding_version"), identity.get("digest"))
                for record in bucket
                if isinstance(identity := record.get("input_identity"), dict)
                and all(isinstance(identity.get(field), str) for field in ("digest", "encoding_id", "encoding_version"))
            }
            if cross_split(bucket) and len(input_keys) > 1:
                issue(warnings, "PositionOverlapRisk", key,
                      "position identity overlaps across splits with distinct full inputs; history or encoding may differ",
                      position_digest=value, **bucket_detail(bucket))

    for split in SPLITS:
        split_counts[split]["games"] = len(games_by_split[split])
    return {
        "schema_version": 1,
        "split_plan_digest": plan["digest"],
        "errors": sorted(errors, key=lambda item: (item["code"], item["context"], digest(item))),
        "warnings": sorted(warnings, key=lambda item: (item["code"], item["context"], digest(item))),
        "counts": {"records": len(records), "source_games": len(sources), "per_split": split_counts},
        "failure_groups": {split: dict(sorted(failure_groups[split].items())) for split in SPLITS},
        "ledger": sorted(ledger, key=lambda entry: (entry["record_id"], digest(entry))),
        "coverage": {
            "exact_input": "digest_and_encoding_identity",
            "provenance": "source_components_and_record_parent_graph",
            "opening_family": "declared_opening_family_id",
            "position_overlap": "declared_board_and_state_digests_conservative_risk",
            "arbitrary_near_positions": "not_run",
            "pretraining_overlap": "unknown",
        },
    }
