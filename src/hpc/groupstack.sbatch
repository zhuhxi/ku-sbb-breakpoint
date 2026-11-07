#!/usr/bin/env bash
#SBATCH --job-name=kusbb-groupstack
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/groupstack.sbatch   (group_stack.py: one LightGBM per feature group + second level)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONWARNINGS=ignore
PYTHONPATH=. ~/venvs/kusbb/bin/python group_stack.py
