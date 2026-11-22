#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=10G
#SBATCH -c 2
#SBATCH --ntasks 16
#SBATCH --output=/storage2/albert/iSTS-mgr/src/evaluation/output/%j.out

# mitigates activation problems
eval "$(conda shell.bash hook)"
source ~/.bashrc

# activate the correct environment
# conda activate pytorch
# source activate /sprint1/anaconda/envs/pytorch
# source activate /home/${USER}/.conda/envs/test-env
conda activate pythia-env

# debug print-outs
echo USER: $USER
which conda
which python

# run the code
PYTHONPATH=. python src/evaluation/evaluate_llm.py --config_file="/storage2/albert/iSTS-mgr/src/configs/evaluate_ft.yaml"