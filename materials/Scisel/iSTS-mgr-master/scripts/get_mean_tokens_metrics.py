from __future__ import annotations

import re
import math
import argparse
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import pandas as pd

try:
    from transformers import AutoTokenizer
except ImportError as e:
    raise SystemExit(
        "Missing dependency: transformers\n"
        "Install with: pip install transformers\n"
        "If LLaMA tokenizer errors: pip install sentencepiece\n"
    ) from e


# Scenario dir examples:
#   answers-students_simple_0_20241211_234346
#   headlines_detailed_5
#   answers-students_score_detailed_1_20241211_234346  (optional legacy - we handle it too)
SCEN_RE = re.compile(
    r"^(?P<dataset>[A-Za-z0-9\-]+)"
    r"(?:_(?P<side_token>alignment|score))?"           # optional embedded side token
    r"_(?P<desc>simple|detailed)"
    r"_(?P<idx>\d+)"
    r"(?:_(?P<date>\d{8})_(?P<time>\d{6}))?$"
)


def parse_scenario_dir(name: str) -> Dict[str, Any]:
    """
    Parse scenario directory name into dataset/desc/idx/timestamp (if present).
    Returns dict with keys: dataset, desc, idx, timestamp (optional), side_token (optional).
    If it doesn't match, returns minimal dict with scenario_raw.
    """
    m = SCEN_RE.match(name)
    if not m:
        return {"scenario_raw": name}

    d = m.groupdict()
    out: Dict[str, Any] = {
        "dataset": d["dataset"],
        "desc": d["desc"],
        "idx": int(d["idx"]),
    }
    if d.get("side_token"):
        out["side_token"] = d["side_token"]

    if d.get("date") and d.get("time"):
        out["timestamp"] = f"{d['date']}_{d['time']}"
    return out


def safe_isna(x: Any) -> bool:
    # pd.isna is safest across None/NaN/strings, but avoid importing numpy explicitly here
    return pd.isna(x)


def mean_tokens_for_texts(tokenizer, texts: List[str], batch_size: int = 256) -> float:
    """
    Compute mean token length for a list of strings using the provided tokenizer.
    """
    if not texts:
        return float("nan")

    lengths: List[int] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding=False,
            truncation=False,
        )
        # enc["input_ids"] is list[list[int]]
        lengths.extend(len(ids) for ids in enc["input_ids"])

    return float(sum(lengths)) / float(len(lengths))


def load_tokenizer_for_model(model_name: str, model_id_map: Optional[Dict[str, str]] = None):
    """
    Load tokenizer for a given model folder name.
    You can provide model_id_map to map folder names -> HF ids or local tokenizer dirs.
    """
    source = model_id_map.get(model_name, model_name) if model_id_map else model_name

    # A couple of common gotchas:
    # - trust_remote_code sometimes needed for nonstandard tokenizers
    # - use_fast=True usually preferred
    tok = AutoTokenizer.from_pretrained(source, use_fast=True, trust_remote_code=True)

    # Some tokenizers (esp. LLaMA) may not have pad token set; not needed for token counts,
    # but setting it can avoid warnings in some setups.
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    return tok


def iter_results_csv(root: Path) -> List[Path]:
    return sorted(root.rglob("results.csv"))


def infer_metadata_from_path(results_csv: Path, root: Path) -> Dict[str, Any]:
    """
    Infer method/side/model/scenario from:
      root/<method>/<side>/<model>/<scenario>/results.csv
    """
    rel = results_csv.resolve().relative_to(root.resolve())
    parts = rel.parts

    meta: Dict[str, Any] = {"results_csv": str(results_csv)}

    # Basic robust parsing
    # Expect at least: method/side/model/scenario/results.csv
    if len(parts) >= 5:
        meta["method"] = parts[0]
        meta["side"] = parts[1]
        meta["model"] = parts[2]
        scenario_dir = parts[3]
        meta.update(parse_scenario_dir(scenario_dir))
        meta["scenario_dir"] = scenario_dir
    else:
        # fallback: just keep what we can
        meta["scenario_dir"] = results_csv.parent.name
        meta.update(parse_scenario_dir(results_csv.parent.name))
        meta["model"] = results_csv.parent.parent.name if results_csv.parent.parent != root else None

    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True,
                    help="Path to your .../results directory (the one containing icl/finetuning)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV path. Default: <results-root>/mean_tokens_summary.csv")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    root = args.results_root.expanduser().resolve()
    out_path = args.out or (root / "mean_tokens_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = iter_results_csv(root)
    if not files:
        raise SystemExit(f"No results.csv found under: {root}")

    # OPTIONAL: map your folder model names to actual HF model ids or local tokenizer dirs.
    # Example:
    MODEL_ID_MAP = {
        "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        "gpt-oss-120b": "openai/gpt-oss-120b"
    }
    
    response_col = "response"
    tokenizer_cache: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    for f in files:
        meta = infer_metadata_from_path(f, root)
        model = meta.get("model") or "UNKNOWN_MODEL"

        if model not in tokenizer_cache:
            tokenizer_cache[model] = load_tokenizer_for_model(model, MODEL_ID_MAP)

        tok = tokenizer_cache[model]

        df = pd.read_csv(f)

        if response_col not in df.columns:
            # skip files that don't have response (or set mean as NaN)
            meta.update({
                "n_rows": int(len(df)),
                "mean_tokens": float("nan"),
                "note": f"missing_column:{response_col}",
            })
            rows.append(meta)
            continue

        texts = [
            "" if safe_isna(x) else str(x)
            for x in df[response_col].tolist()
        ]

        n_nans = df["preds"].isna().sum()

        mean_tok = mean_tokens_for_texts(tok, texts, batch_size=args.batch_size)

        meta.update({
            "n_rows": int(len(texts)),
            "mean_tokens": mean_tok,
            "n_nans": n_nans
        })
        rows.append(meta)

    out_df = pd.DataFrame(rows)

    # Keep output “easy and short”: just one row per scenario file with mean_tokens
    # You can drop columns if you want it even smaller.
    # Here’s a sane minimal set (only keep those that exist):
    preferred_cols = [
        "method", "side", "model", "dataset", "desc", "idx", "timestamp",
        "scenario_dir", "n_rows", "mean_tokens", "results_csv", "note"
    ]
    cols = [c for c in preferred_cols if c in out_df.columns] + [c for c in out_df.columns if c not in preferred_cols]
    out_df = out_df[cols]

    out_df.to_csv(out_path, index=False)
    print(f"Saved summary: {out_path}")
    print(f"Rows: {len(out_df)}  |  Unique models tokenized: {len(tokenizer_cache)}")


if __name__ == "__main__":
    main()
