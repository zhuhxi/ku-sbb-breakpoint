#!/usr/bin/env bash
#SBATCH --job-name=kusbb-sim
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/sim.sbatch N TAG [FOLDS...]   (generated series -> combo10 features -> train_aug on those folds)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=~/venvs/kusbb/bin/python
N=$1; TAG=$2; shift 2
PYTHONPATH=. $PY probe/sim_series.py "$N"
PYTHONPATH=. $PY aug_features.py sim
PYTHONPATH=. $PY train_aug.py --aug sim --tag "$TAG" --folds "$@"
