from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple


# answers-students_score_detailed_1_20241211_234346
# -> answers-students_detailed_1_20241211_234346
RE_SCORE_IN_NAME = re.compile(
    r"^(?P<prefix>[A-Za-z0-9\-]+)_score_(?P<desc>simple|detailed)_(?P<idx>\d+)(?P<ts>_\d{8}_\d{6})?$"
)


def remove_score_token(name: str) -> str | None:
    """
    Remove the '_score_' token from a name of the form:
      <dataset>_score_<simple|detailed>_<idx>[_YYYYMMDD_HHMMSS]
    Returns new name or None if it doesn't match.
    """
    m = RE_SCORE_IN_NAME.match(name)
    if not m:
        return None
    return f"{m.group('prefix')}_{m.group('desc')}_{m.group('idx')}{m.group('ts') or ''}"


def rename_remove_score(root: Path, dry_run: bool = True, recursive: bool = True) -> list[Tuple[Path, Path]]:
    """
    Rename directories under root by removing '_score_' from their names.
    Returns list of (old_path, new_path) for renamed (or would-rename) dirs.
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    changes: list[Tuple[Path, Path]] = []

    it: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    for p in it:
        if not p.is_dir():
            continue

        new_name = remove_score_token(p.name)
        if not new_name:
            continue

        new_path = p.with_name(new_name)

        if new_path.exists():
            raise FileExistsError(f"Target already exists: {new_path}")

        changes.append((p, new_path))
        if not dry_run:
            p.rename(new_path)

    return changes


if __name__ == "__main__":
    ROOT = Path("results")

    # Preview
    changes = rename_remove_score(ROOT, dry_run=False, recursive=True)
    for old, new in changes:
        print(f"DRY RUN: {old}  ->  {new}")
    print(f"Would rename: {len(changes)} directories")

    # Apply (uncomment when ready)
    # changes = rename_remove_score(ROOT, dry_run=False, recursive=True)
    # print(f"Renamed: {len(changes)} directories")
