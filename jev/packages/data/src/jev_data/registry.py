"""Exact-release rights ledger. Repository licensing never substitutes for content rights."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from jev_contracts import validate_example

from .io import boolean, fields, read_config, require, strings, text, version


@dataclass(frozen=True)
class Source:
    id: str
    status: str
    url: str
    revision: str
    sha256: str
    license_ref: str
    attribution: str
    intended_uses: list[str]
    training: bool
    redistribution: bool
    derivatives: bool
    consent: bool
    approval_ref: str
    data_class: str
    original_language: str
    original_split: str
    collected_at: str
    related_sources: list[str]

    @classmethod
    def from_dict(cls, value: dict) -> Source:
        fields(value, set(cls.__dataclass_fields__))
        for key in set(cls.__dataclass_fields__) - {
            "intended_uses", "related_sources", "training", "redistribution",
            "derivatives", "consent",
        }:
            require(type(value[key]) is str, f"source {key} must be text")
        text(value["id"])
        require(value["status"] in {"pending", "approved", "blocked"}, "invalid rights status")
        require(value["data_class"] in {"public", "private", "commissioned"}, "invalid data class")
        uses = strings(value["intended_uses"])
        require(set(uses) <= {"training", "evaluation", "redistribution", "derivatives"},
                "unknown intended use")
        strings(value["related_sources"])
        for key in ("training", "redistribution", "derivatives", "consent"):
            boolean(value[key])
        if value["status"] == "approved":
            for key in ("url", "revision", "license_ref", "attribution", "approval_ref",
                        "original_language", "original_split", "collected_at"):
                text(value[key])
            require(value["revision"].lower() not in {"main", "master", "latest", "head"},
                    "source revision must be pinned")
            require(re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None,
                    "approved source requires SHA-256")
            require(value["consent"] is True, "approved source requires consent/privacy clearance")
            require(bool(uses), "approved source requires intended uses")
            for use in ("training", "redistribution", "derivatives"):
                require((use in uses) == value[use], "rights and intended uses disagree")
        return cls(**value)

    def allow(self, purpose: str, *, derivative: bool = True, release: bool = False) -> None:
        require(self.status == "approved", f"source {self.id}: rights not approved")
        require(purpose in self.intended_uses, f"source {self.id}: use not approved")
        require(self.consent is True, "consent/privacy clearance missing")
        if purpose == "training":
            require(self.training is True, "training prohibited")
            require(self.original_split.lower() not in {
                "test", "development", "calibration", "validation", "dev", "eval", "evaluation",
            }, "protected original source split cannot enter training")
        if derivative:
            require(self.derivatives is True, "derivatives prohibited")
        if release:
            require(self.redistribution is True, "redistribution prohibited")


class Registry:
    def __init__(self, sources: list[Source]):
        require(type(sources) is list, "sources must be a list")
        validated = [Source.from_dict(asdict(source)) for source in sources]
        self.sources = {source.id: source for source in validated}
        require(len(self.sources) == len(validated), "duplicate source ID")
        for source in validated:
            require(set(source.related_sources) <= self.sources.keys(), "unknown related source")

    @classmethod
    def from_dict(cls, value: dict) -> Registry:
        fields(value, {"schema_version", "sources"})
        version(value["schema_version"])
        require(type(value["sources"]) is list, "sources must be a list")
        return cls([Source.from_dict(source) for source in value["sources"]])

    @classmethod
    def load(cls, path: str | Path) -> Registry:
        return cls.from_dict(read_config(path))

    def to_dict(self) -> dict:
        return {"schema_version": 1, "sources": [asdict(self.sources[key])
                                                for key in sorted(self.sources)]}

    def source(self, source_id: str) -> Source:
        require(source_id in self.sources, f"unknown source {source_id}; excluded")
        return self.sources[source_id]

    def validate_row(self, row: dict, *, release: bool = False) -> dict:
        row = validate_example(row)
        provenance = row["provenance"]
        require(provenance["origin"].lower() != "unknown", "unknown origin")
        require(bool(provenance["source_ids"]), "source identity required")
        purpose = "training" if row["split"] == "train" else "evaluation"
        pins = provenance.get("metadata", {}).get("source_pins", {})
        fields(pins, {"schema_version", "sources"})
        version(pins["schema_version"])
        require(type(pins["sources"]) is dict, "source pins must be an object")
        require(set(pins["sources"]) == set(provenance["source_ids"]), "source pins incomplete")
        licenses = set()
        for source_id in provenance["source_ids"]:
            source = self.source(source_id)
            source.allow(purpose, release=release)
            require(pins["sources"][source_id] == {
                "revision": source.revision, "sha256": source.sha256,
                "license_ref": source.license_ref,
            }, "source pin differs from approved ledger")
            licenses.add(source.license_ref)
        require(provenance["license_ref"] in licenses, "unapproved license reference")
        for flag in ("native_origin", "human_authored", "localized", "synthetic"):
            boolean(provenance.get(flag))
        require(not (provenance["native_origin"] and provenance["localized"]),
                "native and localized origins conflict")
        if row["split"] == "test" and row["language"] == "ar":
            require(provenance["native_origin"] is True, "Arabic final test must be native-origin")
        return row

    def pins(self, source_ids: list[str]) -> dict:
        return {"schema_version": 1, "sources": {
            key: {"revision": self.source(key).revision, "sha256": self.source(key).sha256,
                  "license_ref": self.source(key).license_ref} for key in source_ids
        }}
