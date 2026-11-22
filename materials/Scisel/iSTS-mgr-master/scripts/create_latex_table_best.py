from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


INPUT_CSV = Path("/data/home/ascisel/open_webui/iSTS-mgr/results/processed/evalF1_penalty/all_f1_results.csv")
OUT_TEX = Path("/data/home/ascisel/open_webui/iSTS-mgr/results/latex/best_one_table.tex")
OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

METRICS = ["f1_type", "f1_score", "f1_typescore"]
METRIC_HDR = {"f1_type": "Type", "f1_score": "Score", "f1_typescore": "Type+Score"}


def latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def fmt(v: float, bold: bool = False) -> str:
    if pd.isna(v):
        return ""
    s = f"{float(v):.4f}"
    return rf"\textbf{{{s}}}" if bold else s


def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "f1_scoretype" in df.columns and "f1_typescore" not in df.columns:
        df = df.rename(columns={"f1_scoretype": "f1_typescore"})
    if "f1_score_type" in df.columns and "f1_typescore" not in df.columns:
        df = df.rename(columns={"f1_score_type": "f1_typescore"})
    return df


def datasets_order(df: pd.DataFrame) -> List[str]:
    return sorted(df["dataset"].dropna().unique().tolist())


def compute_values_best_per_model(
    df: pd.DataFrame,
    models: List[str],
    datasets: List[str],
) -> Dict[Tuple[str, str, str], float]:
    """
    values[(model, dataset, metric)] = max over scenarios in df.
    """
    values: Dict[Tuple[str, str, str], float] = {}
    for m in models:
        sub_m = df[df["model"] == m]
        for ds in datasets:
            sub = sub_m[sub_m["dataset"] == ds]
            for metric in METRICS:
                values[(m, ds, metric)] = float(sub[metric].max(skipna=True)) if not sub.empty else np.nan
    return values


def best_per_column(
    values: Dict[Tuple[str, str, str], float],
    models: List[str],
    datasets: List[str],
) -> Dict[Tuple[str, str], float]:
    best: Dict[Tuple[str, str], float] = {}
    for ds in datasets:
        for metric in METRICS:
            vals = [values[(m, ds, metric)] for m in models]
            vals = [v for v in vals if not pd.isna(v)]
            best[(ds, metric)] = max(vals) if vals else np.nan
    return best


def winner_models_in_block(
    values: Dict[Tuple[str, str, str], float],
    best: Dict[Tuple[str, str], float],
    models: List[str],
    datasets: List[str],
) -> List[str]:
    """
    Model appears if it's best in ANY dataset×metric cell (within the block).
    """
    eps = 1e-12
    winners = []
    for m in models:
        is_winner = False
        for ds in datasets:
            for metric in METRICS:
                v = values[(m, ds, metric)]
                b = best[(ds, metric)]
                if pd.notna(v) and pd.notna(b) and abs(v - b) <= eps:
                    is_winner = True
                    break
            if is_winner:
                break
        if is_winner:
            winners.append(m)
    return winners


def render_table(
    datasets: List[str],
    blocks_rows: List[Tuple[str, List[List[str]]]],  # (block_name, rows)
) -> str:
    # columns: method, model, desc + 3*len(datasets)
    col_spec = "l l l " + " ".join(["r r r"] * len(datasets))

    lines = []
    lines.append(r"% Preamble needs: \usepackage{adjustbox}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append(r"\begin{adjustbox}{width=\textwidth}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\hline")

    # Header row 1
    h1 = [r"\textbf{Method}", r"\textbf{Model}", r"\textbf{Desc}"]
    for ds in datasets:
        h1.append(rf"\multicolumn{{3}}{{c}}{{\textbf{{{latex_escape(ds)}}}}}")
    lines.append(" & ".join(h1) + r" \\")
    # Header row 2
    h2 = ["", "", ""]
    for _ in datasets:
        h2.extend([r"\textbf{Type}", r"\textbf{Score}", r"\textbf{+TS}"])
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\hline")

    # Blocks
    for i, (block_name, rows) in enumerate(blocks_rows):
        # optional: block label row
        lines.append(rf"\multicolumn{{{3 + 3*len(datasets)}}}{{l}}{{\textbf{{{latex_escape(block_name)}}}}} \\")
        lines.append(r"\hline")
        for r in rows:
            lines.append(" & ".join(r) + r" \\")
        # thick separator between blocks
        lines.append(r"\hline\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\caption{Best-per-block results. Bold = best within block for a given dataset and metric.}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    df = pd.read_csv(INPUT_CSV)
    df = std_cols(df)

    need = {"method", "model", "dataset", "desc"} | set(METRICS)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    datasets = datasets_order(df)

    # detect gpt-oss_120b model name (adjust if you want exact)
    gpt_mask = df["model"].astype(str).str.lower().str.contains("gpt-oss", na=False) & \
               df["model"].astype(str).str.lower().str.contains("120b", na=False)
    gpt_models = sorted(df.loc[gpt_mask, "model"].unique().tolist())
    gpt_model = gpt_models[0] if gpt_models else None

    blocks_rows: List[Tuple[str, List[List[str]]]] = []

    # ---------- FINETUNING (desc = best over all) ----------
    ft = df[df["method"] == "finetuning"].copy()
    ft_models = sorted(ft["model"].unique().tolist())
    ft_values = compute_values_best_per_model(ft, ft_models, datasets)
    ft_best = best_per_column(ft_values, ft_models, datasets)
    ft_winners = winner_models_in_block(ft_values, ft_best, ft_models, datasets)

    ft_rows = []
    for model in ft_winners:
        row = [latex_escape("Finetuning"), latex_escape(model), latex_escape("best")]
        for ds in datasets:
            for metric in METRICS:
                v = ft_values[(model, ds, metric)]
                b = ft_best[(ds, metric)]
                is_best = pd.notna(v) and pd.notna(b) and abs(v - b) <= 1e-12
                row.append(fmt(v, bold=is_best))
        ft_rows.append(row)

    blocks_rows.append(("Finetuning (best over idx/desc)", ft_rows))

    # ---------- ICL (non-gpt) SIMPLE ----------
    icl = df[df["method"] == "icl"].copy()
    if gpt_model is not None:
        icl_nogpt = icl[icl["model"] != gpt_model].copy()
    else:
        icl_nogpt = icl

    icl_s = icl_nogpt[icl_nogpt["desc"] == "simple"].copy()
    icl_s_models = sorted(icl_s["model"].unique().tolist())
    icl_s_values = compute_values_best_per_model(icl_s, icl_s_models, datasets)
    icl_s_best = best_per_column(icl_s_values, icl_s_models, datasets)
    icl_s_winners = winner_models_in_block(icl_s_values, icl_s_best, icl_s_models, datasets)

    icl_s_rows = []
    for model in icl_s_winners:
        row = [latex_escape("ICL"), latex_escape(model), latex_escape("simple")]
        for ds in datasets:
            for metric in METRICS:
                v = icl_s_values[(model, ds, metric)]
                b = icl_s_best[(ds, metric)]
                is_best = pd.notna(v) and pd.notna(b) and abs(v - b) <= 1e-12
                row.append(fmt(v, bold=is_best))
        icl_s_rows.append(row)

    blocks_rows.append(("ICL (without GPT-OSS 120B) — simple", icl_s_rows))

    # ---------- ICL (non-gpt) DETAILED ----------
    icl_d = icl_nogpt[icl_nogpt["desc"] == "detailed"].copy()
    icl_d_models = sorted(icl_d["model"].unique().tolist())
    icl_d_values = compute_values_best_per_model(icl_d, icl_d_models, datasets)
    icl_d_best = best_per_column(icl_d_values, icl_d_models, datasets)
    icl_d_winners = winner_models_in_block(icl_d_values, icl_d_best, icl_d_models, datasets)

    icl_d_rows = []
    for model in icl_d_winners:
        row = [latex_escape("ICL"), latex_escape(model), latex_escape("detailed")]
        for ds in datasets:
            for metric in METRICS:
                v = icl_d_values[(model, ds, metric)]
                b = icl_d_best[(ds, metric)]
                is_best = pd.notna(v) and pd.notna(b) and abs(v - b) <= 1e-12
                row.append(fmt(v, bold=is_best))
        icl_d_rows.append(row)

    blocks_rows.append(("ICL (without GPT-OSS 120B) — detailed", icl_d_rows))

    # ---------- ICL GPT-OSS 120B (2 rows only) ----------
    if gpt_model is not None:
        gpt = icl[icl["model"] == gpt_model].copy()

        gpt_rows = []
        for desc in ["simple", "detailed"]:
            g = gpt[gpt["desc"] == desc]
            row = [latex_escape("ICL"), latex_escape(gpt_model), latex_escape(desc)]
            for ds in datasets:
                sub = g[g["dataset"] == ds]
                for metric in METRICS:
                    v = float(sub[metric].max(skipna=True)) if not sub.empty else np.nan
                    # no bolding here by default (otherwise everything would be bold with one model)
                    row.append(fmt(v, bold=False))
            gpt_rows.append(row)

        blocks_rows.append((f"ICL — {gpt_model}", gpt_rows))
    else:
        blocks_rows.append(("ICL — GPT-OSS 120B (not found)", []))

    latex = render_table(datasets, blocks_rows)
    OUT_TEX.write_text(latex, encoding="utf-8")
    print(f"Saved: {OUT_TEX}")


if __name__ == "__main__":
    main()
