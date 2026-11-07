#!/usr/bin/env bash
#SBATCH --job-name=kusbb-aug
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/aug.sbatch features crop | sbatch hpc/aug.sbatch train TAG [aug kinds...]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=~/venvs/kusbb/bin/python
if [ "$1" = features ]; then
    $PY aug_features.py "$2"
else
    TAG=$2; shift 2
    $PY train_aug.py --tag "$TAG" --aug "$@"
fi
