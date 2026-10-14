"""Union-linked family leakage checks, exact dedup and near-match review candidates."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher

from jev_contracts import canonical_json, validate_example

from .arabic import normalized_view
from .io import dataset_identity, digest, fields, require, strings, version


class UnionFind:
    def __init__(self) -> None:
        self.parents: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parents.setdefault(item, item)
        root = item
        while self.parents[root] != root:
            root = self.parents[root]
        while self.parents[item] != item:
            item, self.parents[item] = self.parents[item], root
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        self.parents[max(a, b)] = min(a, b)


def lineage_tokens(row: dict) -> set[str]:
    provenance = row["provenance"]
    tokens = {"row:" + row["id"], "family:" + row["family_id"]}
    tokens.update("row:" + key for key in provenance["derivation_ids"])
    lineage = provenance.get("metadata", {}).get("lineage", {"schema_version": 1})
    fields(lineage, {"schema_version"}, {
        "parent_ids", "translation_ids", "family_ids", "group_ids", "document_ids",
        "template_ids", "source_row_ids",
    })
    version(lineage["schema_version"])
    prefixes = {"parent_ids": "row:", "translation_ids": "row:", "family_ids": "family:",
                "group_ids": "group:", "document_ids": "document:",
                "template_ids": "template:", "source_row_ids": "source-row:"}
    for key, prefix in prefixes.items():
        tokens.update(prefix + item for item in strings(lineage.get(key, [])))
    return tokens


def family_groups(rows: list[dict], union_groups: list[list[str]] | None = None) -> dict[str, str]:
    dataset_identity(rows)
    union = UnionFind()
    row_ids = {row["id"] for row in rows}
    for row in rows:
        tokens = sorted(lineage_tokens(row))
        for token in tokens:
            union.union(tokens[0], token)
    for group in union_groups or []:
        strings(group)
        require(len(group) >= 2, "union group needs at least two row IDs")
        require(set(group) <= row_ids, "union group references unknown input row")
        for item in group:
            union.union("row:" + group[0], "row:" + item)
    return {row["id"]: union.find("row:" + row["id"]) for row in rows}


def split_key(row: dict) -> str:
    if row["split"] != "calibration":
        return row["split"]
    return "calibration/" + row["provenance"]["metadata"]["calibration"]["subset"]


def validate_splits(rows: list[dict], union_groups: list[list[str]] | None = None) -> dict:
    groups = family_groups(rows, union_groups)
    assignments: dict[str, str] = {}
    for row in rows:
        group, split = groups[row["id"]], split_key(row)
        require(group not in assignments or assignments[group] == split,
                "linked parents/translations/families cross protected split "
                "or calibration subgroup")
        assignments[group] = split
    return {"groups": groups, "assignments": assignments}


def content_view(row: dict) -> str:
    row = validate_example(row)
    question = row["question"]
    return canonical_json({
        "state": normalized_view(row["state"]) if type(row["state"]) is str else row["state"],
        "instructions": normalized_view(question["instructions"]),
        "type": question["type"],
        "options": sorted(normalized_view(option["description"]) for option in question["options"]),
    })


def duplicate_report(rows: list[dict], near_threshold: float = 0.92) -> dict:
    from .io import number

    number(near_threshold, 0.5, 1)
    dataset_identity(rows)
    exact: dict[str, list[str]] = defaultdict(list)
    views = {}
    for row in rows:
        views[row["id"]] = content_view(row)
        exact[digest(views[row["id"]])].append(row["id"])
    near = []
    narratives = {row["id"]: normalized_view(row["state"]) if type(row["state"]) is str
                  else canonical_json(row["state"]) for row in rows}
    comparisons = 0
    # Bounded lexical retrieval only; semantic/multilingual relations must be supplied as lineage.
    # Buckets carry (language, task, variety): cross-task surfaces are structurally
    # different rule states, and cross-variety near-pairs are same-family by
    # construction (one family renders every variety), so both need no comparison.
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["id"]):
        key = (row["language"], row["task"], row["variety"])
        for previous in buckets[key]:
            comparisons += 1
            require(comparisons <= 5000000,
                    "near-match scan budget exceeded; partition by reviewed source families")
            left, right = narratives[previous], narratives[row["id"]]
            if views[previous] == views[row["id"]]:
                continue
            matcher = SequenceMatcher(None, left, right, autojunk=False)
            if (matcher.real_quick_ratio() >= near_threshold
                    and matcher.quick_ratio() >= near_threshold
                    and matcher.ratio() >= near_threshold):
                near.append([previous, row["id"]])
        buckets[key].append(row["id"])
    return {"exact": [ids for ids in exact.values() if len(ids) > 1], "near": near}
