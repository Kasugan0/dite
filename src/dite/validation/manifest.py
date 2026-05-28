"""Validation manifest loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def _normalize_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError(f"constraint pair cannot reference itself: {left}")
    return tuple(sorted((left, right)))


@dataclass(frozen=True)
class ValidationFileRecord:
    """Single validation file record from a manifest."""

    path: str
    cluster_id: str
    must_link: tuple[str, ...] = ()
    must_not_link: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ConstraintSet:
    """Expanded pairwise constraints for a validation corpus."""

    must_link_pairs: frozenset[tuple[str, str]] = frozenset()
    must_not_link_pairs: frozenset[tuple[str, str]] = frozenset()


@dataclass(frozen=True)
class ValidationCorpus:
    """Loaded validation manifest plus expanded lookup structures."""

    root: Path
    name: str
    tier: str
    description: str
    owner: str
    files: tuple[ValidationFileRecord, ...]
    records_by_path: dict[str, ValidationFileRecord] = field(default_factory=dict)
    cluster_ids_by_path: dict[str, str] = field(default_factory=dict)
    constraints: ConstraintSet = field(default_factory=ConstraintSet)


def load_validation_corpus(folder: Path, *, manifest_path: Path | None = None) -> ValidationCorpus:
    """Load and validate a validation corpus manifest from a directory."""
    root = folder.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"validation corpus folder does not exist: {folder}")

    manifest = (manifest_path or (root / "manifest.json")).resolve()
    if not manifest.exists():
        raise ValueError(f"validation manifest not found: {manifest}")

    data = json.loads(manifest.read_text(encoding="utf-8"))
    files_data = data.get("files")
    if not isinstance(files_data, list):
        raise ValueError(f"validation manifest {manifest} must contain a files list")

    records: list[ValidationFileRecord] = []
    records_by_path: dict[str, ValidationFileRecord] = {}
    cluster_ids_by_path: dict[str, str] = {}
    seen_pairs: set[tuple[str, str, str]] = set()
    must_link_pairs: set[tuple[str, str]] = set()
    must_not_link_pairs: set[tuple[str, str]] = set()

    for item in files_data:
        path_value = item.get("path", "")
        cluster_id = item.get("cluster_id", "")
        if not path_value or not isinstance(path_value, str):
            raise ValueError(f"validation manifest {manifest} contains a file without path")
        if not cluster_id or not isinstance(cluster_id, str):
            raise ValueError(
                f"validation manifest {manifest} contains a file without cluster_id: {path_value}"
            )
        if path_value in records_by_path:
            raise ValueError(
                f"validation manifest {manifest} contains duplicate path: {path_value}"
            )

        file_path = root / path_value
        if not file_path.exists():
            raise ValueError(
                f"validation manifest {manifest} references missing file: {path_value}"
            )

        must_link = tuple(item.get("must_link", []) or [])
        must_not_link = tuple(item.get("must_not_link", []) or [])
        notes = item.get("notes", "") or ""
        record = ValidationFileRecord(
            path=path_value,
            cluster_id=cluster_id,
            must_link=must_link,
            must_not_link=must_not_link,
            notes=notes,
        )
        records.append(record)
        records_by_path[path_value] = record
        cluster_ids_by_path[path_value] = cluster_id

    for record in records:
        for target in record.must_link:
            if target not in records_by_path:
                raise ValueError(
                    f"validation manifest {manifest} has invalid must_link target "
                    f"{target!r} for {record.path!r}"
                )
            pair = _normalize_pair(record.path, target)
            dedupe_key = ("must_link", *pair)
            if dedupe_key in seen_pairs:
                raise ValueError(
                    f"validation manifest {manifest} contains duplicate must_link pair: {pair}"
                )
            seen_pairs.add(dedupe_key)
            must_link_pairs.add(pair)

        for target in record.must_not_link:
            if target not in records_by_path:
                raise ValueError(
                    f"validation manifest {manifest} has invalid must_not_link target "
                    f"{target!r} for {record.path!r}"
                )
            pair = _normalize_pair(record.path, target)
            dedupe_key = ("must_not_link", *pair)
            if dedupe_key in seen_pairs:
                raise ValueError(
                    f"validation manifest {manifest} contains duplicate must_not_link pair: {pair}"
                )
            seen_pairs.add(dedupe_key)
            must_not_link_pairs.add(pair)

    return ValidationCorpus(
        root=root,
        name=str(data.get("name", "")),
        tier=str(data.get("tier", "")),
        description=str(data.get("description", "")),
        owner=str(data.get("owner", "")),
        files=tuple(records),
        records_by_path=records_by_path,
        cluster_ids_by_path=cluster_ids_by_path,
        constraints=ConstraintSet(
            must_link_pairs=frozenset(must_link_pairs),
            must_not_link_pairs=frozenset(must_not_link_pairs),
        ),
    )
