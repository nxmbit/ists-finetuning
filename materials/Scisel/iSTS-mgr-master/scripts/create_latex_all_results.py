from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

INPUT_CSV = Path("/data/home/ascisel/open_webui/iSTS-mgr/results/processed/evalF1_penalty/all_f1_results.csv")
OUT_TEX = Path("/data/home/ascisel/open_webui/iSTS-mgr/results/latex/all_results_longtable.tex")
OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

METRICS = ["f1_type", "f1_score", "f1_typescore"]
METRIC_HDR = {"f1_type": "Type", "f1_score": "Score", "f1_typescore": "+TS"}

# Optional fixed ordering (keeps your paper consistent)
DATASET_ORDER = ["answers-students", "headlines", "images"]  # set to None to use alphabetical


def latex_escape(s: str) -> str:
    """Escape LaTeX special chars (CRITICAL: underscores!)."""
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


def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "f1_scoretype" in df.columns and "f1_typescore" not in df.columns:
        df = df.rename(columns={"f1_scoretype": "f1_typescore"})
    if "f1_score_type" in df.columns and "f1_typescore" not in df.columns:
        df = df.rename(columns={"f1_score_type": "f1_typescore"})
    return df


def fmt(v) -> str:
    if pd.isna(v):
        return ""
    return f"{float(v):.4f}"


def get_datasets(df: pd.DataFrame) -> List[str]:
    ds = sorted(df["dataset"].dropna().unique().tolist())
    if DATASET_ORDER:
        out = [d for d in DATASET_ORDER if d in ds]
        out += [d for d in ds if d not in out]
        return out
    return ds


def to_wide(df: pd.DataFrame, datasets: List[str]) -> pd.DataFrame:
    """
    Pivot to wide format:
      index = (method, model, desc, idx)
      columns = (dataset, metric)
    Values are MAX over duplicates within (method, model, desc, idx, dataset).
    """
    g = (
        df.groupby(["method", "model", "desc", "idx", "dataset"], as_index=False)[METRICS]
          .max()
    )
    wide = g.pivot_table(
        index=["method", "model", "desc", "idx"],
        columns="dataset",
        values=METRICS,
        aggfunc="first",
    )
    # columns: (metric, dataset) -> (dataset, metric)
    wide = wide.swaplevel(0, 1, axis=1)

    # enforce full set + ordering
    full_cols = pd.MultiIndex.from_product([datasets, METRICS])
    wide = wide.reindex(columns=full_cols)

    return wide.sort_index()


def build_longtable(wide: pd.DataFrame, datasets: List[str]) -> str:
    # +1 for Mean +TS across datasets
    n_cols = 4 + 3 * len(datasets) + 1

    # column spec: method/model/desc/n_examples + (Type,Score,+TS)*datasets + Mean(+TS)
    col_spec = "l l l r " + " ".join(["r r r"] * len(datasets)) + " r"

    # Header row 1
    h1 = [
        r"\textbf{Method}",
        r"\textbf{Model}",
        r"\textbf{Desc}",
        r"\textbf{n\_examples}",
    ]
    for ds in datasets:
        h1.append(rf"\multicolumn{{3}}{{c}}{{\textbf{{\texttt{{{latex_escape(ds)}}}}}}}")
    h1.append(r"\textbf{Mean}")
    header_row1 = " & ".join(h1) + r" \\"

    # Header row 2
    h2 = ["", "", "", ""]
    for _ in datasets:
        h2.extend([rf"\textbf{{{METRIC_HDR[m]}}}" for m in METRICS])
    h2.append(r"\textbf{+TS}")
    header_row2 = " & ".join(h2) + r" \\"

    def render_row(idx_tuple, row) -> str:
        method, model, desc, n_examples = idx_tuple
        method_disp = "Finetuning" if str(method).lower() == "finetuning" else "ICL"

        cells = [
            latex_escape(method_disp),
            latex_escape(model),
            latex_escape(desc),
            str(int(float(n_examples))) if pd.notna(n_examples) else "",
        ]

        for ds in datasets:
            for m in METRICS:
                cells.append(fmt(row[(ds, m)]))

        # Mean +TS across datasets for this row
        ts_vals = []
        for ds in datasets:
            v = row[(ds, "f1_typescore")]
            if pd.notna(v):
                ts_vals.append(float(v))
        mean_ts = (sum(ts_vals) / len(ts_vals)) if ts_vals else np.nan
        cells.append(fmt(mean_ts))

        return " & ".join(cells) + r" \\"

    index = wide.index
    method_lvl = index.get_level_values(0).astype(str).str.lower()
    model_lvl = index.get_level_values(1).astype(str).str.lower()
    desc_lvl = index.get_level_values(2).astype(str).str.lower()

    blocks: List[Tuple[str, pd.Index]] = []
    blocks.append(("Finetuning", index[method_lvl == "finetuning"]))

    m_icl = method_lvl == "icl"
    m_no_gpt = ~model_lvl.str.contains("gpt-oss", na=False)
    blocks.append(("ICL (no GPT-OSS) — simple", index[m_icl & m_no_gpt & (desc_lvl == "simple")]))
    blocks.append(("ICL (no GPT-OSS) — detailed", index[m_icl & m_no_gpt & (desc_lvl == "detailed")]))

    m_gpt = model_lvl.str.contains("gpt-oss", na=False)
    blocks.append(("ICL — gpt-oss", index[m_icl & m_gpt]))

    lines = []
    lines.append(r"% Requires in LaTeX preamble: \usepackage{longtable}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append(r"\begin{longtable}{" + col_spec + r"}")
    lines.append(
        r"\caption{Wszystkie wyniki (pivot po datasetach). Wiersz odpowiada (method, model, desc, idx). "
        r"Ostatnia kolumna to średnia \texttt{+TS} po datasetach.}\label{tab:all_results}\\"
    )
    lines.append(r"\hline")
    lines.append(header_row1)
    lines.append(header_row2)
    lines.append(r"\hline")
    lines.append(r"\endfirsthead")

    # continuation head
    lines.append(r"\caption[]{Wszystkie wyniki (kontynuacja).}\\")
    lines.append(r"\hline")
    lines.append(header_row1)
    lines.append(header_row2)
    lines.append(r"\hline")
    lines.append(r"\endhead")

    # foot
    lines.append(r"\hline")
    lines.append(r"\multicolumn{" + str(n_cols) + r"}{r}{\small Kontynuacja na następnej stronie.}\\")
    lines.append(r"\endfoot")

    # last foot
    lines.append(r"\hline")
    lines.append(r"\endlastfoot")

    # content
    for block_title, block_index in blocks:
        lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\textbf{" + latex_escape(block_title) + r"}} \\")
        lines.append(r"\hline")

        if len(block_index) == 0:
            lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{(brak wierszy)} \\")
            lines.append(r"\hline\hline")
            continue

        sub = wide.loc[block_index]
        for idx_tuple, row in sub.iterrows():
            lines.append(render_row(idx_tuple, row))

        lines.append(r"\hline\hline")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def main():
    df = pd.read_csv(INPUT_CSV)
    df = standardize_cols(df)

    required = {"method", "model", "desc", "idx", "dataset"} | set(METRICS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    df["idx"] = pd.to_numeric(df["idx"], errors="coerce")
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    datasets = get_datasets(df)
    wide = to_wide(df, datasets)
    latex = build_longtable(wide, datasets)

    OUT_TEX.write_text(latex, encoding="utf-8")
    print(f"Saved: {OUT_TEX}")


if __name__ == "__main__":
    main()
