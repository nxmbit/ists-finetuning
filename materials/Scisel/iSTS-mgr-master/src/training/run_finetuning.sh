#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=35G
#SBATCH --ntasks 1
#SBATCH -o slurm_out/slurm-%j.log
#SBATCH --job-name=BTC_mining

# mitigates activation problems
eval "$(conda shell.bash hook)"
source .bashrc

conda activate sd3_finetune
# debug print-outs
echo USER: $USER
which conda
which python

# run the code
PYTHONPATH=. python src/training/finetuning_sfttrainer.py --config_file="/data/home/ascisel/open_webui/iSTS-mgr/configs/finetune_llm.yaml"
# PYTHONPATH=. python src/training/finetuning_trainer.py --config_file="/storage2/albert/iSTS-mgr/src/configs/finetune_llm.yaml"