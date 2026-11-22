import re
import csv
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.wa_converter import prepare_test_wa_file
from src.data.data import find_files_with_phrase


RESULTS_DIR = "/data/home/ascisel/open_webui/iSTS-mgr/results_new"
TEST_DATA_DIR = "/data/home/ascisel/open_webui/iSTS-mgr/Data/test_data/wa"

# Run name examples:
#   answers-students_simple_0_20260124_110330
#   answers-students_simple_0_lr0.0001_lora_rank8_20260124_114856
# lr / lora_rank optional
_RUN_RE = re.compile(
    r"^(?P<dataset>[A-Za-z0-9\-_]+)_(?P<desc>simple|detailed)_(?P<idx>\d+)"
    r"(?:_lr(?P<lr>[0-9eE\.\+\-]+))?"
    r"(?:_lora_rank(?P<lora_rank>\d+))?"
    r"(?:_(?P<ts>\d{8}_\d{6}))?$"
)

F1_PATTERNS = {
    "f1_ali": re.compile(r"\bF1\s+Ali\s+([0-9.]+)"),
    "f1_type": re.compile(r"\bF1\s+Type\s+([0-9.]+)"),
    "f1_score": re.compile(r"\bF1\s+Score\s+([0-9.]+)"),
    "f1_typescore": re.compile(r"\bF1\s+Typ\+Sco\s+([0-9.]+)"),
}


def run_evalF1_penalty(eval_script: Path, gold_wa: Path, sys_wa: Path) -> Dict[str, float]:
    eval_script = eval_script.resolve()
    scripts_dir = eval_script.parent

    cmd = ["perl", f"-I{scripts_dir}", str(eval_script), str(gold_wa), str(sys_wa)]
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
        raise ValueError(f"Could not parse {missing} from eval output.\nSTDOUT:\n{res.stdout}")

    return scores


def append_row_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


@dataclass(frozen=True)
class RunKey:
    method: str              # e.g. "finetuning" or "icl"
    run_type: str            # "merged" or "both"
    model: str               # e.g. "Llama-3.1-8B-Instruct"
    dataset: str             # e.g. "answers-students"
    desc: str                # "simple" or "detailed"
    idx: int                 # number after simple/detailed
    lr: Optional[float]
    lora_rank: Optional[int]


def _parse_run_name(name: str) -> Optional[Tuple[str, str, int, Optional[float], Optional[int], Optional[datetime]]]:
    m = _RUN_RE.match(name)
    if not m:
        return None

    dataset = m.group("dataset")
    desc = m.group("desc")
    idx = int(m.group("idx"))

    lr_s = m.group("lr")
    lr = float(lr_s) if lr_s is not None else None

    lora_s = m.group("lora_rank")
    lora_rank = int(lora_s) if lora_s is not None else None

    ts = m.group("ts")
    ts_dt = datetime.strptime(ts, "%Y%m%d_%H%M%S") if ts else None

    return dataset, desc, idx, lr, lora_rank, ts_dt


def _choose_newer(
    existing: Optional[Tuple[Path, Optional[datetime]]],
    candidate: Tuple[Path, Optional[datetime]],
) -> Tuple[Path, Optional[datetime]]:
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


def _looks_like_model_dir(p: Path) -> bool:
    # model dir contains subdirs that match run name pattern
    try:
        for child in p.iterdir():
            if child.is_dir() and _parse_run_name(child.name):
                return True
    except Exception:
        return False
    return False


def scan_runs(results_root: Path) -> List[Tuple[RunKey, Path]]:
    """
    Scans:
      results_root/<method>/{merged,both}/<model>/<run_name>/results.csv
    Also supports provider level:
      results_root/<method>/{merged,both}/<provider>/<model>/<run_name>/results.csv
    Keeps only the newest per RunKey (based on timestamp in run_name).
    """
    results_root = results_root.expanduser().resolve()
    out: Dict[RunKey, Tuple[Path, Optional[datetime]]] = {}

    for method_dir in results_root.iterdir():
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        if method == "processed":
            continue

        for run_type in ("merged", "both"):
            base = method_dir / run_type
            if not base.exists():
                continue

            def handle_model_dir(model_dir: Path):
                model = model_dir.name
                for run_dir in model_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    parsed = _parse_run_name(run_dir.name)
                    if not parsed:
                        continue
                    dataset, desc, idx, lr, lora_rank, ts_dt = parsed

                    # require results.csv inside run_dir
                    if not (run_dir / "results.csv").exists():
                        continue

                    key = RunKey(
                        method=method,
                        run_type=run_type,
                        model=model,
                        dataset=dataset,
                        desc=desc,
                        idx=idx,
                        lr=lr,
                        lora_rank=lora_rank,
                    )
                    out[key] = _choose_newer(out.get(key), (run_dir, ts_dt))

            # base can contain model dirs directly, OR provider dirs that contain model dirs
            for level1 in base.iterdir():
                if not level1.is_dir():
                    continue

                if _looks_like_model_dir(level1):
                    handle_model_dir(level1)
                else:
                    # provider dir
                    for model_dir in level1.iterdir():
                        if model_dir.is_dir() and _looks_like_model_dir(model_dir):
                            handle_model_dir(model_dir)

    # flatten
    jobs: List[Tuple[RunKey, Path]] = [(k, v[0]) for k, v in out.items()]
    jobs.sort(key=lambda x: (x[0].method, x[0].run_type, x[0].model, x[0].dataset, x[0].desc, x[0].idx, x[0].lr or -1.0, x[0].lora_rank or -1))
    return jobs


def _get_pred_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    if "pred" in lower:
        return lower["pred"]
    if "preds" in lower:
        return lower["preds"]
    raise ValueError(f"results.csv has no 'pred' or 'preds' column. Columns={cols}")


def _parse_score_alignment(cell: Any) -> Tuple[int, str]:
    """
    cell is expected as: "SCORE-ALIGNMENT"
    - SCORE: int-ish
    - ALIGNMENT: string
    If missing/NaN -> (0, "NOALI")
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or pd.isna(cell):
        return 0, "NOALI"

    s = str(cell).strip()
    if not s:
        return 0, "NOALI"

    if "-" in s:
        score_part, ali_part = s.split("-", 1)
        score_part = score_part.strip()
        ali_part = ali_part.strip() or "NOALI"
        try:
            score = int(float(score_part))
        except Exception:
            score = 0
        return score, ali_part

    # Fallback: if it is only a score
    try:
        return int(float(s)), "NOALI"
    except Exception:
        return 0, "NOALI"


def read_preds_from_results_csv(results_csv: Path) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(results_csv)
    pred_col = _get_pred_column(df)

    align_pred: List[str] = []
    score_pred: List[int] = []
    for cell in df[pred_col].to_list():
        score, ali = _parse_score_alignment(cell)
        score_pred.append(score)
        align_pred.append(ali)

    # normalize alignment
    align_pred = ["NOALI" if (a is None or str(a).strip() == "" or str(a).lower() == "nan") else str(a) for a in align_pred]
    return align_pred, score_pred


def main():
    parser = ArgumentParser()
    parser.add_argument("--eval_script", required=False, type=str, default="./scripts/evalF1_penalty.pl")
    args = parser.parse_args()

    root = Path(RESULTS_DIR)
    eval_script = Path(args.eval_script)
    script_name = eval_script.name.split(".")[0]

    fieldnames = [
        "timestamp", "method", "run_type", "model", "dataset", "desc", "idx", "lr", "lora_rank",
        "gold_path", "sys_path",
        "f1_ali", "f1_type", "f1_score", "f1_typescore",
    ]

    jobs = scan_runs(root)

    for key, run_dir in tqdm(jobs):
        output_path = root / "processed" / script_name / key.method / key.run_type / key.model
        output_path.mkdir(parents=True, exist_ok=True)

        results_csv = run_dir / "results.csv"
        align_pred, score_pred = read_preds_from_results_csv(results_csv)

        wa_files = find_files_with_phrase(TEST_DATA_DIR, key.dataset, "wa")
        xml_files = find_files_with_phrase(TEST_DATA_DIR, key.dataset, "xml")
        if not wa_files or not xml_files:
            print(f"WARNING: Missing gold/test files for dataset={key.dataset}; skipping {run_dir}")
            continue

        test_file = wa_files[0]
        gold_wa = Path(xml_files[0])

        lr_tag = f"_lr{key.lr}" if key.lr is not None else "_lrNone"
        rank_tag = f"_lora_rank{key.lora_rank}" if key.lora_rank is not None else "_lora_rankNone"
        out_xml = output_path / f"{key.dataset}_{key.desc}_{key.idx}{lr_tag}{rank_tag}.xml"
        
        prepare_test_wa_file(test_file, align_pred, score_pred, out_xml)

        out_wa = out_xml.with_suffix(".wa")
        if out_wa.exists():
            out_wa.unlink()
        out_xml.replace(out_wa)

        scores = run_evalF1_penalty(eval_script, gold_wa=gold_wa, sys_wa=out_wa)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "method": key.method,
            "run_type": key.run_type,
            "model": key.model,
            "dataset": key.dataset,
            "desc": key.desc,
            "idx": key.idx,
            "lr": key.lr,
            "lora_rank": key.lora_rank,
            "gold_path": str(gold_wa),
            "sys_path": str(out_wa),
            **scores,
        }

        append_row_csv(output_path / "f1_results.csv", row, fieldnames)


if __name__ == "__main__":
    main()
