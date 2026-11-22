from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

INPUT_CSV = Path("/data/home/ascisel/open_webui/iSTS-mgr/results/processed/evalF1_penalty/all_f1_results.csv")
OUTDIR = Path("/data/home/ascisel/open_webui/iSTS-mgr/results/latex")
OUTDIR.mkdir(parents=True, exist_ok=True)

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


def fmt(v) -> str:
    if pd.isna(v):
        return ""
    return f"{float(v):.4f}"


def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "f1_scoretype" in df.columns and "f1_typescore" not in df.columns:
        df = df.rename(columns={"f1_scoretype": "f1_typescore"})
    if "f1_score_type" in df.columns and "f1_typescore" not in df.columns:
        df = df.rename(columns={"f1_score_type": "f1_typescore"})
    return df


def datasets_order(df: pd.DataFrame) -> List[str]:
    return sorted(df["dataset"].dropna().unique().tolist())


def build_summary_finetuning(df: pd.DataFrame, datasets: List[str]) -> pd.DataFrame:
    """
    Finetuning: per (model, dataset) bierzemy max po scenariuszach,
    potem pivot -> (dataset, metric) + Mean(Type+Score).
    """
    best = df.groupby(["model", "dataset"], as_index=False)[METRICS].max()

    wide = best.pivot_table(index=["model"], columns="dataset", values=METRICS, aggfunc="first")
    wide = wide.swaplevel(0, 1, axis=1)  # (dataset, metric)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([datasets, METRICS]), copy=False)

    wide[("Mean", "f1_typescore")] = wide.xs("f1_typescore", level=1, axis=1).mean(axis=1)
    return wide


def build_summary_icl_best_idx(df: pd.DataFrame, datasets: List[str]) -> pd.DataFrame:
    """
    ICL: wybieramy najlepsze idx (n_examples) per (model, desc),
    gdzie "najlepsze" = max mean(Type+Score) po datasetach.
    Następnie raportujemy metryki dla tego idx.
    """
    # 1) per (model, desc, idx, dataset) -> max (gdyby były duplikaty)
    per_idx = df.groupby(["model", "desc", "idx", "dataset"], as_index=False)[METRICS].max()

    # 2) mean(Type+Score) po datasetach dla (model, desc, idx)
    mean_ts = (
        per_idx.groupby(["model", "desc", "idx"], as_index=False)["f1_typescore"]
              .mean()
              .rename(columns={"f1_typescore": "mean_typescore"})
    )

    # 3) wybór idx maksymalizującego mean_typescore per (model, desc)
    best_idx = mean_ts.loc[mean_ts.groupby(["model", "desc"])["mean_typescore"].idxmax()]
    # best_idx: columns -> model, desc, idx, mean_typescore

    # 4) dołącz wybrany idx do per_idx i przefiltruj tylko ten idx
    merged = per_idx.merge(best_idx[["model", "desc", "idx"]], on=["model", "desc", "idx"], how="inner")

    # 5) pivot -> index=(model, desc, idx) + (dataset, metric)
    wide = merged.pivot_table(index=["model", "desc", "idx"], columns="dataset", values=METRICS, aggfunc="first")
    wide = wide.swaplevel(0, 1, axis=1)  # (dataset, metric)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([datasets, METRICS]), copy=False)

    # 6) Mean(Type+Score) po datasetach (dla wybranego idx)
    wide[("Mean", "f1_typescore")] = wide.xs("f1_typescore", level=1, axis=1).mean(axis=1)
    return wide


def to_latex_table(
    wide: pd.DataFrame,
    title: str,
    include_desc: bool,
    include_n_examples: bool,
) -> str:
    ds_no_mean = [ds for ds, _ in wide.columns if ds != "Mean"]
    ds_no_mean = sorted(set(ds_no_mean))
    base_cols = ["Model"] + (["Desc"] if include_desc else []) + (["n_examples"] if include_n_examples else [])

    # column spec
    col_spec = ("l " * len(base_cols)).strip() + " " + " ".join(["r r r"] * len(ds_no_mean)) + " r"
    lines = []
    lines.append(r"% Requires in LaTeX preamble: \usepackage{adjustbox}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append(r"\begin{adjustbox}{width=\textwidth}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\hline")

    # Header row 1
    h1 = [rf"\textbf{{{c}}}" for c in base_cols]
    for ds in ds_no_mean:
        h1.append(rf"\multicolumn{{3}}{{c}}{{\textbf{{{latex_escape(ds)}}}}}")
    h1.append(r"\textbf{Mean}")
    lines.append(" & ".join(h1) + r" \\")

    # Header row 2
    h2 = [""] * len(base_cols)
    for _ in ds_no_mean:
        h2.extend([rf"\textbf{{{METRIC_HDR[m]}}}" for m in METRICS])
    h2.append(r"\textbf{Type+Score}")
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\hline")

    # Sort by mean desc
    wide_sorted = wide.copy()
    if ("Mean", "f1_typescore") in wide_sorted.columns:
        wide_sorted = wide_sorted.sort_values(("Mean", "f1_typescore"), ascending=False)

    for idx_vals, row in wide_sorted.iterrows():
        if not isinstance(idx_vals, tuple):
            idx_vals = (idx_vals,)

        # index shapes:
        # finetuning: (model,)
        # icl: (model, desc, idx)
        model = latex_escape(idx_vals[0])
        out = [model]

        cursor = 1
        if include_desc:
            out.append(latex_escape(idx_vals[cursor] if len(idx_vals) > cursor else ""))
            cursor += 1
        if include_n_examples:
            # idx stored as float sometimes; render as int if possible
            raw = idx_vals[cursor] if len(idx_vals) > cursor else ""
            try:
                raw_int = int(float(raw))
                out.append(str(raw_int))
            except Exception:
                out.append(latex_escape(raw))
            cursor += 1

        for ds in ds_no_mean:
            for m in METRICS:
                out.append(fmt(row.get((ds, m), np.nan)))
        out.append(fmt(row.get(("Mean", "f1_typescore"), np.nan)))

        lines.append(" & ".join(out) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(rf"\caption{{{title}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    df = pd.read_csv(INPUT_CSV)
    df = standardize_cols(df)

    need = {"method", "model", "dataset", "desc", "idx"} | set(METRICS)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce")

    datasets = datasets_order(df)

    # detect gpt-oss models
    lower = df["model"].astype(str).str.lower()
    gpt_mask = lower.str.contains("gpt-oss", na=False)
    gpt_models = sorted(df.loc[gpt_mask, "model"].unique().tolist())

    # -------- Finetuning table --------
    ft = df[df["method"] == "finetuning"].copy()
    ft_wide = build_summary_finetuning(ft, datasets)
    ft_tex = to_latex_table(
        ft_wide,
        title="Finetuning: best F1 per dataset (max over scenarios), with mean Type+Score across datasets.",
        include_desc=False,
        include_n_examples=False,
    )
    (OUTDIR / "finetuning_per_model.tex").write_text(ft_tex, encoding="utf-8")

    # -------- ICL (non-gpt-oss) table with n_examples --------
    icl = df[df["method"] == "icl"].copy()
    if gpt_models:
        icl_nogpt = icl[~icl["model"].isin(gpt_models)].copy()
    else:
        icl_nogpt = icl

    icl_wide = build_summary_icl_best_idx(icl_nogpt, datasets)
    icl_tex = to_latex_table(
        icl_wide,
        title="ICL (without GPT-OSS): best idx chosen by max mean(Type+Score) across datasets; table reports metrics at that n_examples.",
        include_desc=True,
        include_n_examples=True,
    )
    (OUTDIR / "icl_best_per_model_desc_nexamples.tex").write_text(icl_tex, encoding="utf-8")

    # -------- ICL GPT-OSS table (only gpt-oss models) --------
    if gpt_models:
        icl_gpt = icl[icl["model"].isin(gpt_models)].copy()
        icl_gpt_wide = build_summary_icl_best_idx(icl_gpt, datasets)
        icl_gpt_tex = to_latex_table(
            icl_gpt_wide,
            title="ICL (GPT-OSS): best idx chosen by max mean(Type+Score) across datasets; table reports metrics at that n_examples.",
            include_desc=True,
            include_n_examples=True,
        )
        (OUTDIR / "icl_gptoss_best_desc_nexamples.tex").write_text(icl_gpt_tex, encoding="utf-8")
    else:
        (OUTDIR / "icl_gptoss_best_desc_nexamples.tex").write_text(
            "% No GPT-OSS models found in input.\n", encoding="utf-8"
        )

    print("Saved:")
    print(OUTDIR / "finetuning_per_model.tex")
    print(OUTDIR / "icl_best_per_model_desc_nexamples.tex")
    print(OUTDIR / "icl_gptoss_best_desc_nexamples.tex")


if __name__ == "__main__":
    main()
