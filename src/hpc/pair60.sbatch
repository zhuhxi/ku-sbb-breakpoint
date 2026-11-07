#!/usr/bin/env bash
#SBATCH --job-name=kusbb-pair60
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
J=64 ~/venvs/kusbb/bin/python -W ignore probe/confirm_pair.py 60 pair60 semiauto_bag6w
