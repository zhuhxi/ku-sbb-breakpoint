#!/bin/bash
#SBATCH --job-name=kusbb-simpost
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/simpost.sbatch N   (clean-generator fakes -> combo10 features -> fake-trained posterior on real)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore
PY=~/venvs/kusbb/bin/python
PYTHONPATH=. $PY probe/sim_series_ku.py "$1"
PYTHONPATH=. $PY aug_features.py simku 2>&1 | grep -v -i warn
PYTHONPATH=. $PY probe/simpost_full.py 2>&1 | grep -v -i warn
