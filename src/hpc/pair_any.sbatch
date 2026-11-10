#!/usr/bin/env bash
#SBATCH --job-name=kusbb-pair8r
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
J=64 ~/venvs/kusbb/bin/python -W ignore probe/confirm_pair.py "$1" "$2" "$3"
