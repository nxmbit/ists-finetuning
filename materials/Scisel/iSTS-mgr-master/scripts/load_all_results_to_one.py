from pathlib import Path
import pandas as pd
from argparse import ArgumentParser

RESULTS_DIR = "/data/home/ascisel/open_webui/iSTS-mgr/results_new"


def load_all_f1_results(processed_dir: Path) -> pd.DataFrame:
    files = sorted(processed_dir.rglob("f1_results.csv"))
    if not files:
        raise FileNotFoundError(f"Nie znaleziono f1_results.csv w: {processed_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except:
            breakpoint()
        # Struktura: .../processed/<method>/<model>/f1_results.csv
        try:
            method = f.parents[1].name
            model = f.parents[0].name
        except Exception:
            method, model = None, None

        # Dodaj metadane jeśli brakuje
        if "method" not in df.columns:
            df["method"] = method
        if "model" not in df.columns:
            df["model"] = model

        df["source_file"] = str(f)  # opcjonalnie: skąd pochodzi rekord
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    # opcjonalnie: ustandaryzuj typy
    for c in ["f1_ali", "f1_type", "f1_score", "f1_typescore"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_script", required=False, type=str, default="./scripts/evalF1_penalty.pl")
    args = parser.parse_args()

    root = Path(RESULTS_DIR)

    eval_script = Path(args.eval_script)  # adjust if needed
    # ^ if gold varies by dataset, pick it dynamically instead.
    script_name = eval_script.name.split(".")[0]
    output_path = root / "processed" / script_name

    all_df = load_all_f1_results(output_path)
    print(all_df.shape)
    print(all_df.head())

    # Zapis zbiorczy
    out_path = output_path / "all_f1_results.csv"
    all_df.to_csv(out_path, index=False)
    print(f"Zapisano: {out_path}")