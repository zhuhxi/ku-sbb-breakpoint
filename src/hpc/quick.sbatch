#!/usr/bin/env bash
#SBATCH --job-name=kusbb-quick
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/quick.sbatch probe/SCRIPT.py [ARGS ...]   (a short probe run on a compute node)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1
PYTHONPATH=. ~/venvs/kusbb/bin/python "$@"
