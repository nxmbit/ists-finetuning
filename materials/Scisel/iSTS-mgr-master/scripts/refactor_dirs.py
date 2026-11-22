#!/usr/bin/env python3
import argparse
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict


RUN_RE = re.compile(
    r"^(?P<dataset>[A-Za-z0-9\-_]+)_(?P<desc>simple|detailed)_(?P<idx>\d+)"
    r"(?:_lr(?P<lr>[0-9eE\.\+\-]+))?"
    r"(?:_lora_rank(?P<lora_rank>\d+))?"
    r"(?:_(?P<ts>\d{8}_\d{6}))?$"
)

def parse_run_name(name: str) -> Optional[Tuple[str, str, int, Optional[float], Optional[int], Optional[datetime]]]:
    m = RUN_RE.match(name)
    if not m:
        return None
    dataset = m.group("dataset")
    desc = m.group("desc")
    idx = int(m.group("idx"))

    lr = float(m.group("lr")) if m.group("lr") is not None else None
    lora_rank = int(m.group("lora_rank")) if m.group("lora_rank") is not None else None

    ts = m.group("ts")
    ts_dt = datetime.strptime(ts, "%Y%m%d_%H%M%S") if ts else None
    return dataset, desc, idx, lr, lora_rank, ts_dt


@dataclass(frozen=True)
class Sig:
    dataset: str
    desc: str
    idx: int
    lr: Optional[float]
    lora_rank: Optional[int]


@dataclass
class RunItem:
    path: Path
    ts: Optional[datetime]
    mtime: float


def is_flat_run_dir(root: Path, p: Path) -> bool:
    # only consider immediate children of root (flat)
    if p.parent != root or not p.is_dir():
        return False
    return parse_run_name(p.name) is not None


def sort_key(item: RunItem) -> Tuple[int, float]:
    # Primary: timestamp if present, else push to end and use mtime.
    if item.ts is not None:
        return (0, item.ts.timestamp())
    return (1, item.mtime)


def move_or_copy(src: Path, dst: Path, do_copy: bool, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    if dry_run:
        op = "COPY" if do_copy else "MOVE"
        print(f"{op}: {src}  ->  {dst}")
        return

    if do_copy:
        shutil.copytree(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. .../results_new/icl/merged or .../icl/both")
    ap.add_argument("--models", nargs="+", required=True, help="Model dir names in desired order")
    ap.add_argument("--missing-policy", choices=["drop-first", "drop-last", "strict"], default="drop-first",
                    help="If fewer runs than models in a signature group, which models to assign to")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    flat_runs = [p for p in root.iterdir() if is_flat_run_dir(root, p)]
    if not flat_runs:
        print(f"No flat run dirs found directly under: {root}")
        return

    groups: Dict[Sig, List[RunItem]] = defaultdict(list)

    for p in flat_runs:
        parsed = parse_run_name(p.name)
        assert parsed is not None
        dataset, desc, idx, lr, lora_rank, ts = parsed
        sig = Sig(dataset, desc, idx, lr, lora_rank)
        groups[sig].append(RunItem(path=p, ts=ts, mtime=p.stat().st_mtime))

    # Process each signature group independently
    for sig in sorted(groups.keys(), key=lambda s: (s.dataset, s.desc, s.idx, s.lr or -1.0, s.lora_rank or -1)):
        items = sorted(groups[sig], key=sort_key)
        n = len(items)
        m = len(args.models)

        if n > m:
            raise RuntimeError(
                f"Signature group has {n} runs but only {m} models provided.\n"
                f"Signature: {sig}\n"
                f"Runs: {[it.path.name for it in items]}"
            )

        if args.missing_policy == "strict" and n != m:
            raise RuntimeError(
                f"STRICT: expected exactly {m} runs for signature {sig}, but found {n}.\n"
                f"Runs: {[it.path.name for it in items]}"
            )

        if n == m:
            model_list = args.models
        else:
            # fewer runs than models
            if args.missing_policy == "drop-first":
                model_list = args.models[-n:]   # keep last n (often baseline missing)
            else:  # drop-last
                model_list = args.models[:n]    # keep first n

        # Move each run into its model dir
        for it, model in zip(items, model_list):
            dst = root / model / it.path.name
            move_or_copy(it.path, dst, do_copy=args.copy, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
