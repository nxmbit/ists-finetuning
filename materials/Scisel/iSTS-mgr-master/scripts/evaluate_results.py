import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from src.data.wa_converter import prepare_test_wa_file
from src.data.data import find_files_with_phrase
import pandas as pd
import numpy as np
import csv
from argparse import ArgumentParser
import subprocess
from tqdm import tqdm


RESULTS_DIR = "/data/home/ascisel/open_webui/iSTS-mgr/results_new"
TEST_DATA_DIR = "/data/home/ascisel/open_webui/iSTS-mgr/Data/test_data/wa"


_RUN_RE = re.compile(
    r"^(?P<dataset>[A-Za-z0-9\-]+)_(?P<desc>simple|detailed)_(?P<idx>\d+)(?:_(?P<ts>\d{8}_\d{6}))?$"
)


F1_PATTERNS = {
    "f1_ali": re.compile(r"\bF1\s+Ali\s+([0-9.]+)"),
    "f1_type": re.compile(r"\bF1\s+Type\s+([0-9.]+)"),
    "f1_score": re.compile(r"\bF1\s+Score\s+([0-9.]+)"),
    "f1_typescore": re.compile(r"\bF1\s+Typ\+Sco\s+([0-9.]+)"),
}


def run_evalF1_penalty(eval_script: Path, gold_wa: Path, sys_wa: Path) -> Dict[str, float]:
    """
    Runs evalF1_penalty.pl and returns dict with keys:
      f1_ali, f1_type, f1_score, f1_typescore
    """
    eval_script = eval_script.resolve()
    scripts_dir = eval_script.parent

    cmd = [
        "perl",
        f"-I{scripts_dir}",          
        str(eval_script),
        str(gold_wa),
        str(sys_wa),
    ]

    res = subprocess.run(cmd, text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            "evalF1_penalty.pl failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr}\n"
        )

    scores: Dict[str, float] = {}
    for line in res.stdout.splitlines():
        for key, pat in F1_PATTERNS.items():
            m = pat.search(line)
            if m:
                scores[key] = float(m.group(1))

    missing = [k for k in F1_PATTERNS if k not in scores]
    if missing:
        raise ValueError(
            f"Could not parse {missing} from eval output.\nSTDOUT:\n{res.stdout}"
        )

    return scores


def append_row_csv(path: Path, row: Dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


@dataclass(frozen=True)
class RunKey:
    method: str
    model: str
    dataset: str 
    desc: str    
    n_examples: int


def _parse_run_name(name: str) -> Optional[Tuple[str, str, int, Optional[datetime]]]:
    """
    Returns (dataset, desc, idx, timestamp_dt) or None if name doesn't match.
    """
    m = _RUN_RE.match(name)
    if not m:
        return None
    dataset = m.group("dataset")
    desc = m.group("desc")
    n_examples = int(m.group("idx"))
    ts = m.group("ts")
    ts_dt = datetime.strptime(ts, "%Y%m%d_%H%M%S") if ts else None
    return dataset, desc, n_examples, ts_dt


def _choose_newer(existing: Optional[Tuple[Path, Optional[datetime]]],
                  candidate: Tuple[Path, Optional[datetime]]) -> Tuple[Path, Optional[datetime]]:

    if existing is None:
        return candidate

    ex_path, ex_ts = existing
    cand_path, cand_ts = candidate

    if ex_ts is None and cand_ts is not None:
        return candidate
    if ex_ts is not None and cand_ts is None:
        return existing
    if ex_ts is not None and cand_ts is not None and cand_ts > ex_ts:
        return candidate
    return existing


def _scan_side(root: Path, method: str, side: str) -> Dict[RunKey, Tuple[Path, Optional[datetime]]]:
    """
    Scan: root/<method>/<side>/<model>/<run_name>
    side is "alignment" or "score".
    Returns map: RunKey -> (path_to_run_item, timestamp_dt)
    """
    side_dir = root / method / side
    out: Dict[RunKey, Tuple[Path, Optional[datetime]]] = {}

    if not side_dir.exists():
        return out

    for model_dir in side_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for run_item in model_dir.iterdir():
            parsed = _parse_run_name(run_item.name)
            if not parsed:
                continue
            dataset, desc, n_examples, ts_dt = parsed
            key = RunKey(method=method, model=model, dataset=dataset, desc=desc, n_examples=n_examples)

            out[key] = _choose_newer(out.get(key), (run_item, ts_dt))

    return out


def find_alignment_score_pairs(results_root: Path) -> List[Tuple[RunKey, Path, Path]]:
    """
    Finds pairs that exist in BOTH alignment and score for the same:
      (method, model, dataset, desc, idx)

    Returns list of (RunKey, alignment_path, score_path)
    """
    results_root = results_root.expanduser().resolve()

    pairs: List[Tuple[RunKey, Path, Path]] = []

    # Discover methods present (usually "icl" and/or "finetuning")
    methods = [p.name for p in results_root.iterdir() if p.is_dir()]
    for method in methods:
        alignment_map = _scan_side(results_root, method, "alignment")
        score_map = _scan_side(results_root, method, "score")
        # breakpoint()
        for key in sorted(set(alignment_map) & set(score_map), key=lambda k: (k.method, k.model, k.dataset, k.desc, k.idx)):
            align_path, _ = alignment_map[key]
            score_path, _ = score_map[key]
            pairs.append((key, align_path, score_path))

    return pairs


def main():
    parser = ArgumentParser()
    parser.add_argument("--eval_script", required=False, type=str, default="./scripts/evalF1_penalty.pl")
    args = parser.parse_args()

    root = Path(RESULTS_DIR)
    pairs = find_alignment_score_pairs(root)
    eval_script = Path(args.eval_script)  # adjust if needed
    # ^ if gold varies by dataset, pick it dynamically instead.
    script_name = eval_script.name.split(".")[0]
    # one CSV per model (or put it wherever you prefer)
    fieldnames = [
        "timestamp", "method", "model", "dataset", "desc", "idx",
        "gold_path", "sys_path",
        "f1_ali", "f1_type", "f1_score", "f1_typescore",
    ]
    for key, align_path, score_path in tqdm(pairs):
        output_path = root / "processed" / script_name / key.method / key.model
        output_path.mkdir(parents=True, exist_ok=True)

        align_pred = pd.read_csv(align_path / "results.csv")["preds"].to_list()
        align_pred = ["NOALI" if pd.isna(pred) else pred for pred in align_pred]
        
        score_pred = pd.read_csv(score_path / "results.csv")["preds"].to_list()
        score_pred = [0 if (pred is None or np.isnan(pred)) else int(pred) for pred in score_pred]
        # breakpoint()

        test_file = find_files_with_phrase(TEST_DATA_DIR, key.dataset, "wa")[0]

        out_xml = output_path / f"{key.dataset}_{key.desc}_{key.idx}.xml"
        prepare_test_wa_file(test_file, align_pred, score_pred, out_xml)

        out_wa = out_xml.with_suffix(".wa")
        if out_wa.exists():
            out_wa.unlink()
        out_xml.replace(out_wa)

        # Evaluate + save with metadata
        gold_wa = Path(find_files_with_phrase(TEST_DATA_DIR, key.dataset, "xml")[0])
        scores = run_evalF1_penalty(eval_script, gold_wa=gold_wa, sys_wa=out_wa)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "method": key.method,
            "model": key.model,
            "dataset": key.dataset,
            "desc": key.desc,
            "idx": key.idx,
            "gold_path": str(gold_wa),
            "sys_path": str(out_wa),
            **scores,
        }

        append_row_csv(output_path / "f1_results.csv", row, fieldnames)

if __name__ == "__main__":
    main()