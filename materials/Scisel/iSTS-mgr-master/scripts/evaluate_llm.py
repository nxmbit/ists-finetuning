import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Tuple

from omegaconf import OmegaConf


# ---- EDIT THESE DEFAULTS (or pass via CLI) ----
DEFAULT_PROJECT_ROOT = Path("/data/home/ascisel/open_webui/iSTS-mgr")
DEFAULT_MODELS_ROOT = DEFAULT_PROJECT_ROOT / "models"
DEFAULT_OUT_CFG_DIR = DEFAULT_PROJECT_ROOT / "sweep_eval_configs"

# Set this to the module you normally run with: python -m <module>
# Example (you must replace with your real module path):
DEFAULT_EVAL_MODULE = "src.evaluation.evaluate_llm"


# If you evaluate multiple base models, you likely need a mapping from folder name (in models/)
# to full HF model id used by AutoTokenizer/AutoModel.
# If the folder name matches the basename of base_config.model_name, you don't need this.
MODEL_NAME_MAP: Dict[str, str] = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}

N_GENERATION_TOKENS = {
    "score": 5,
    "alignment": 5,
    "both": 15,
    "merged": 11
}


def looks_like_adapter_dir(p: Path) -> bool:
    """Heuristic: TRL/PEFT adapters often have adapter_config.json + adapter_model.*"""
    if not p.is_dir():
        return False
    if (p / "adapter_config.json").exists():
        return True
    if (p / "adapter_model.safetensors").exists():
        return True
    if (p / "adapter_model.bin").exists():
        return True
    # sometimes people save full checkpoints; keep it permissive
    if (p / "pytorch_model.bin").exists() or (p / "model.safetensors").exists():
        return True
    return False


def iter_adapter_runs(models_root: Path) -> Iterator[Tuple[str, str, str, Path]]:
    """
    Yield (task, dataset, base_model_folder, run_dir) for dirs matching:
      models/<task>/<dataset>/<base_model_folder>/<timestamp>/
    """
    for run_dir in models_root.glob("*/*/*/*/*/*"):
        if looks_like_adapter_dir(run_dir):
            rel = run_dir.relative_to(models_root)
            task, dataset, base_model_folder, learning_rate, lora_rank, _run_name = rel.parts[:6]
            yield task, dataset, base_model_folder, learning_rate, lora_rank, run_dir


def resolve_model_name(base_cfg, base_model_folder: str) -> str | None:
    """
    Return HF model id to use in config. Priority:
      1) if base_cfg.model_name basename matches base_model_folder -> keep it
      2) else use MODEL_NAME_MAP[base_model_folder] if present
      3) else None (skip)
    """
    cfg_model_name = getattr(base_cfg, "model_name", None)
    if isinstance(cfg_model_name, str) and os.path.split(cfg_model_name)[1] == base_model_folder:
        return cfg_model_name

    if base_model_folder in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[base_model_folder]

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", required=True, type=str, help="Path to base eval YAML config")
    ap.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT), type=str, help="Repo root (contains src/)")
    ap.add_argument("--models_root", default=str(DEFAULT_MODELS_ROOT), type=str, help="Root dir containing models/")
    ap.add_argument("--eval_module", default=DEFAULT_EVAL_MODULE, type=str, help="Module to run via python -m")
    ap.add_argument("--out_cfg_dir", default=str(DEFAULT_OUT_CFG_DIR), type=str, help="Where to write generated configs")
    ap.add_argument("--only_task", default=None, type=str, help="Optional filter: only evaluate this task")
    ap.add_argument("--only_dataset", default=None, type=str, help="Optional filter: only evaluate this dataset")
    ap.add_argument("--sleep_sec", default=0.0, type=float, help="Optional sleep between runs")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    models_root = Path(args.models_root)
    out_cfg_dir = Path(args.out_cfg_dir)
    out_cfg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OmegaConf.load(args.base_config)

    runs = list(iter_adapter_runs(models_root))
    
    if args.only_task:
        runs = [r for r in runs if r[0] == args.only_task]
    if args.only_dataset:
        runs = [r for r in runs if r[1] == args.only_dataset]

    if not runs:
        print("No adapter runs found (or all filtered out).")
        sys.exit(0)

    for i, (task, dataset, base_model_folder, learning_rate, lora_rank,  run_dir) in enumerate(runs, start=1):
        model_name = resolve_model_name(base_cfg, base_model_folder)
        if model_name is None:
            print(f"[SKIP] {run_dir} — can't resolve HF model id for base model folder '{base_model_folder}'. "
                  f"Add it to MODEL_NAME_MAP or set base_config.model_name accordingly.")
            continue

        # Merge overrides into base config
        overrides = {
            "task": task,
            "dataset": dataset,
            "model_name": model_name,
            "lora_adapter": str(run_dir),
            "generation": {"max_new_tokens": N_GENERATION_TOKENS[task]}
        }
        cfg = OmegaConf.merge(base_cfg, overrides)

        # Save per-run config
        run_tag = f"{task}_{dataset}_{base_model_folder}_{learning_rate}_{lora_rank}_{run_dir.name}"
        cfg_path = out_cfg_dir / f"{i:03d}_{run_tag}.yaml"
        OmegaConf.save(cfg, cfg_path)

        # Run evaluation via -m to keep "src" importable
        cmd = [sys.executable, "-m", args.eval_module, "--config_file", str(cfg_path)]
        print(f"\n=== [{i}/{len(runs)}] EVAL: {run_tag} ===")
        print("CMD:", " ".join(cmd))

        env = os.environ.copy()
        subprocess.run(cmd, check=True, cwd=str(project_root), env=env)

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()