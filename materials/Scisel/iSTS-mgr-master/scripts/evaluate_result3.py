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

# Supports:
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
    method: str
    model: str
    dataset: str
    desc: str
    idx: int
    lr: Optional[float]
    lora_rank: Optional[int]


@dataclass(frozen=True)
class EvalJob:
    # run_type: "paired" (alignment+score dirs) OR "merged" OR "both"
    run_type: str
    key: RunKey
    # for paired:
    align_dir: Optional[Path] = None
    score_dir: Optional[Path] = None
    # for merged/both:
    run_dir: Optional[Path] = None


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


# ---------- reading preds (paired dirs) ----------

def _read_alignment_preds(results_csv: Path) -> List[str]:
    df = pd.read_csv(results_csv)
    col = "preds" if "preds" in df.columns else ("pred" if "pred" in df.columns else df.columns[0])
    preds = df[col].tolist()
    return ["NOALI" if pd.isna(p) else str(p) for p in preds]


def _score_to_int(x: Any) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return 0
    s = str(x).strip()
    if not s:
        return 0
    if s.upper() == "NIL":
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0


def _read_score_preds(results_csv: Path) -> List[int]:
    df = pd.read_csv(results_csv)
    col = "preds" if "preds" in df.columns else ("pred" if "pred" in df.columns else df.columns[0])
    return [_score_to_int(p) for p in df[col].tolist()]


# ---------- reading preds (single-file merged/both) ----------

_SCORE_TOKEN_RE = re.compile(r"^(?:NIL|[0-6])(?:\.0+)?$", re.IGNORECASE)

def _is_score_token(tok: str) -> bool:
    return bool(_SCORE_TOKEN_RE.match(tok.strip()))

def _normalize_align(a: Any) -> str:
    if a is None or (isinstance(a, float) and np.isnan(a)) or pd.isna(a):
        return "NOALI"
    s = str(a).strip()
    if not s or s.lower() == "nan":
        return "NOALI"
    if s.upper() == "NIL":
        return "NOALI"
    return s

def _parse_score_alignment(cell: Any) -> Tuple[int, str]:
    """
    Expect pred like 'SCORE-ALIGNMENT' in merged/both results.csv.
    Also tolerates reversed order if it ever happens.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or pd.isna(cell):
        return 0, "NOALI"
    s = str(cell).strip()
    if not s:
        return 0, "NOALI"

    if "-" in s:
        left, right = (x.strip() for x in s.split("-", 1))
        left_is_score = _is_score_token(left)
        right_is_score = _is_score_token(right)

        # Prefer the token that looks like a score (NIL or 0-6)
        if left_is_score and not right_is_score:
            score_tok, align_tok = left, right
        elif right_is_score and not left_is_score:
            score_tok, align_tok = right, left
        else:
            # default to "SCORE-ALIGNMENT" if ambiguous
            score_tok, align_tok = left, right

        return _score_to_int(score_tok), _normalize_align(align_tok)

    # no dash => could be only score or only align
    if _is_score_token(s):
        return _score_to_int(s), "NOALI"
    return 0, _normalize_align(s)

def _get_pred_column(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    if "preds" in lower:
        return lower["preds"]
    if "pred" in lower:
        return lower["pred"]
    raise ValueError(f"results.csv missing 'preds'/'pred' column. Columns={list(df.columns)}")

def _read_combined_preds(results_csv: Path) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(results_csv)
    col = _get_pred_column(df)

    align_pred: List[str] = []
    score_pred: List[int] = []
    for cell in df[col].tolist():
        score, ali = _parse_score_alignment(cell)
        score_pred.append(score)
        align_pred.append(ali)
    return align_pred, score_pred


# ---------- scanning jobs ----------

def _scan_side(root: Path, method: str, side: str) -> Dict[RunKey, Tuple[Path, Optional[datetime]]]:
    """
    Old layout:
      root/<method>/<side>/<model>/<run_name>
    side in {"alignment","score"}
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
            if not run_item.is_dir():
                continue
            parsed = _parse_run_name(run_item.name)
            if not parsed:
                continue
            dataset, desc, idx, lr, lora_rank, ts_dt = parsed
            if not (run_item / "results.csv").exists():
                continue
            key = RunKey(method=method, model=model, dataset=dataset, desc=desc, idx=idx, lr=lr, lora_rank=lora_rank)
            out[key] = _choose_newer(out.get(key), (run_item, ts_dt))
    return out


def _looks_like_model_dir(p: Path) -> bool:
    # model dir contains subdirs matching run name
    try:
        for child in p.iterdir():
            if child.is_dir() and _parse_run_name(child.name):
                return True
    except Exception:
        return False
    return False


def _scan_single(root: Path, method: str, run_type: str) -> Dict[RunKey, Tuple[Path, Optional[datetime]]]:
    """
    New layout:
      root/<method>/{merged,both}/<model>/<run_name>/results.csv
    Also supports:
      root/<method>/{merged,both}/<provider>/<model>/<run_name>/results.csv
    """
    base = root / method / run_type
    out: Dict[RunKey, Tuple[Path, Optional[datetime]]] = {}
    if not base.exists():
        return out

    def handle_model_dir(model_dir: Path):
        model = model_dir.name
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            parsed = _parse_run_name(run_dir.name)
            if not parsed:
                continue
            dataset, desc, idx, lr, lora_rank, ts_dt = parsed
            if not (run_dir / "results.csv").exists():
                continue
            key = RunKey(method=method, model=model, dataset=dataset, desc=desc, idx=idx, lr=lr, lora_rank=lora_rank)
            out[key] = _choose_newer(out.get(key), (run_dir, ts_dt))

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

    return out


def build_jobs(root: Path) -> List[EvalJob]:
    root = root.expanduser().resolve()
    jobs: List[EvalJob] = []

    methods = [p.name for p in root.iterdir() if p.is_dir() and p.name != "processed"]

    # (A) paired alignment+score (OLD, working behavior)
    for method in methods:
        alignment_map = _scan_side(root, method, "alignment")
        score_map = _scan_side(root, method, "score")
        for key in sorted(set(alignment_map) & set(score_map),
                          key=lambda k: (k.method, k.model, k.dataset, k.desc, k.idx, k.lr or -1.0, k.lora_rank or -1)):
            align_dir, _ = alignment_map[key]
            score_dir, _ = score_map[key]
            jobs.append(EvalJob(run_type="paired", key=key, align_dir=align_dir, score_dir=score_dir))

    # (B) merged/both single-file runs
    for method in methods:
        for run_type in ("merged", "both"):
            single_map = _scan_single(root, method, run_type)
            for key in sorted(single_map.keys(),
                              key=lambda k: (k.method, k.model, k.dataset, k.desc, k.idx, k.lr or -1.0, k.lora_rank or -1)):
                run_dir, _ = single_map[key]
                jobs.append(EvalJob(run_type=run_type, key=key, run_dir=run_dir))

    return jobs


def main():
    parser = ArgumentParser()
    parser.add_argument("--eval_script", required=False, type=str, default="./scripts/evalF1_penalty.pl")
    args = parser.parse_args()

    root = Path(RESULTS_DIR)
    eval_script = Path(args.eval_script)
    script_name = eval_script.name.split(".")[0]

    fieldnames = [
        "timestamp", "method", "run_type", "model", "dataset", "desc", "idx", "lr", "lora_rank",
        "run_path", "gold_path", "sys_path",
        "f1_ali", "f1_type", "f1_score", "f1_typescore",
    ]

    jobs = build_jobs(root)

    for job in tqdm(jobs):
        key = job.key

        output_path = root / "processed" / script_name / key.method / job.run_type / key.model
        output_path.mkdir(parents=True, exist_ok=True)

        # Read preds
        if job.run_type == "paired":
            assert job.align_dir is not None and job.score_dir is not None
            align_pred = _read_alignment_preds(job.align_dir / "results.csv")
            score_pred = _read_score_preds(job.score_dir / "results.csv")
            run_path_str = f"{job.align_dir} + {job.score_dir}"
        else:
            assert job.run_dir is not None
            align_pred, score_pred = _read_combined_preds(job.run_dir / "results.csv")
            run_path_str = str(job.run_dir)

        if len(align_pred) != len(score_pred):
            print(f"WARNING: length mismatch: {run_path_str} align={len(align_pred)} score={len(score_pred)}; skipping.")
            continue

        wa_files = find_files_with_phrase(TEST_DATA_DIR, key.dataset, "wa")
        xml_files = find_files_with_phrase(TEST_DATA_DIR, key.dataset, "xml")
        if not wa_files or not xml_files:
            print(f"WARNING: Missing gold/test files for dataset={key.dataset}; skipping {run_path_str}")
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
            "run_type": job.run_type,
            "model": key.model,
            "dataset": key.dataset,
            "desc": key.desc,
            "idx": key.idx,
            "lr": key.lr,
            "lora_rank": key.lora_rank,
            "run_path": run_path_str,
            "gold_path": str(gold_wa),
            "sys_path": str(out_wa),
            **scores,
        }

        append_row_csv(output_path / "f1_results.csv", row, fieldnames)


if __name__ == "__main__":
    main()
