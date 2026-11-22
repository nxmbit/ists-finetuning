import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


BASE_CONFIG = "/data/home/ascisel/open_webui/iSTS-mgr/configs/finetune_llm.yaml"

# repo root (folder that contains "src/")
PROJECT_ROOT = Path("/data/home/ascisel/open_webui/iSTS-mgr")

# module you normally run with -m
TRAIN_MODULE = "src.training.finetuning_sfttrainer"

OUT_CONFIG_DIR = PROJECT_ROOT / "sweep_configs"


def build_runs():
    """
    Yield tuples: (run_name, overrides_dict)

    Overrides are merged into BASE_CONFIG with OmegaConf.merge().
    Adjust keys to match your YAML structure, e.g. cfg.training.learning_rate, cfg.lora.rank, cfg.task, etc.
    """
    # tasks = ["score", "alignment"]
    tasks = ["both", "merged"]             
    lrs = [1e-5, 5e-5, 1e-4]                
    ranks = [8, 16, 32]                 
    datasets = ["images", "headlines", "answers-students"]
    models = ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.1-8B-Instruct"]

    for model, task, ds, lr, r in itertools.product(models, tasks, datasets, lrs, ranks):
        run_name = f"{task}_ds{ds}"
        overrides = {
            "task": task,
            "data": {
                "dataset": ds,
            },
            "model": {
                "base_model": model
            },
            "training": {
                "learning_rate": lr,
            },
            "lora": {
                "rank": r,
            },
        }
        yield run_name, overrides


def main():
    base_cfg = OmegaConf.load(BASE_CONFIG)
    OUT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for i, (run_name, overrides) in enumerate(build_runs(), start=1):
        cfg = OmegaConf.merge(base_cfg, overrides)
        cfg_path = OUT_CONFIG_DIR / f"{i:03d}_{run_name}.yaml"
        OmegaConf.save(cfg, cfg_path)

        env = os.environ.copy()

        cmd = [sys.executable, "-m", TRAIN_MODULE, "--config_file", str(cfg_path)]
        print("\nCMD:", " ".join(cmd))

        subprocess.run(
            cmd,
            check=True,
            env=env,
            cwd=str(PROJECT_ROOT),   # IMPORTANT: so "src" is importable
        )

        time.sleep(1)  # optional, only if your output dirs may collide by second


if __name__ == "__main__":
    main()